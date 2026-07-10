"""Tests for the shared aiohttp HTTP utilities (`metaculus_bot/research/http_fetch.py`).

Covers:
- `BROWSER_HEADERS` completeness (Safari-like UA + Accept / Accept-Language / Accept-Encoding)
- `build_session` config plumbing (ClientTimeout total+sock_read, TCPConnector limit, headers, resolver)
- `read_body_capped` under/at/over the byte cap (over -> None + WARNING)
- `FilteringResolver` filters private IPs, raises OSError when everything is filtered,
  and delegates `close()` to its inner resolver.

No real network calls: sessions are constructed, inspected, and closed.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Any

import pytest
from aiohttp.abc import AbstractResolver

from metaculus_bot.research.http_fetch import (
    BROWSER_HEADERS,
    FilteringResolver,
    build_session,
    read_body_capped,
)


def _resolve_entry(ip: str, host: str = "example.com") -> dict[str, Any]:
    """Return a dict matching aiohttp.abc.ResolveResult (TypedDict)."""
    return {
        "hostname": host,
        "host": ip,
        "port": 0,
        "family": int(socket.AF_INET if ":" not in ip else socket.AF_INET6),
        "proto": 0,
        "flags": 0,
    }


class FakeInnerResolver(AbstractResolver):
    """Inner resolver that returns fixed entries and tracks close()."""

    def __init__(self, entries: list[dict[str, Any]]):
        self._entries = entries
        self.close_called = False

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[Any]:
        del host, port, family
        return list(self._entries)  # noqa: ASYNC910

    async def close(self) -> None:  # noqa: ASYNC910
        self.close_called = True


def _all_disallowed(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    del ip
    return True


def _reject_private(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return ip.is_private or ip.is_loopback


class FakeReadResponse:
    """Minimal stub exposing the `.read()` surface `read_body_capped` consumes."""

    def __init__(self, body: bytes):
        self._body = body

    async def read(self) -> bytes:
        return self._body  # noqa: ASYNC910


class TestBrowserHeaders:
    def test_contains_required_keys(self):
        for key in ("User-Agent", "Accept", "Accept-Language", "Accept-Encoding"):
            assert key in BROWSER_HEADERS, f"missing header {key}"

    def test_user_agent_is_safari_like(self):
        ua = BROWSER_HEADERS["User-Agent"]
        assert "Safari" in ua
        assert "Mozilla/5.0" in ua

    def test_accept_negotiates_html(self):
        assert "text/html" in BROWSER_HEADERS["Accept"]

    def test_accept_encoding_excludes_brotli(self):
        # aiohttp's brotli decoder isn't a project dep (HAS_BROTLI=False), so
        # advertising `br` would have Brotli-preferring servers return content
        # aiohttp can't decode -> ClientResponseError -> silent source drop.
        # We advertise only gzip/deflate.
        assert BROWSER_HEADERS["Accept-Encoding"] == "gzip, deflate"
        assert "br" not in BROWSER_HEADERS["Accept-Encoding"]


class TestBuildSession:
    async def test_configures_timeout_connector_and_headers(self):
        async with build_session(timeout_s=12.5, connector_limit=7, headers=BROWSER_HEADERS) as session:
            assert session.timeout.total == 12.5
            assert session.timeout.sock_read == 12.5
            assert session.connector is not None
            assert session.connector.limit == 7
            assert session.headers["User-Agent"] == BROWSER_HEADERS["User-Agent"]
            assert session.headers["Accept-Language"] == BROWSER_HEADERS["Accept-Language"]

    async def test_defaults_connector_limit_20_and_no_default_headers(self):
        async with build_session(timeout_s=5.0) as session:
            assert session.timeout.total == 5.0
            assert session.timeout.sock_read == 5.0
            assert session.connector is not None
            assert session.connector.limit == 20
            # No headers arg -> no session-level User-Agent (aiohttp adds its own
            # at request time; prediction_market relies on this no-header default).
            assert "User-Agent" not in session.headers

    async def test_resolver_kwarg_flows_into_connector(self):
        # build_session must plumb the resolver into TCPConnector so aiohttp's
        # connect-time DNS lookup goes through the caller's predicate.
        inner = FakeInnerResolver([_resolve_entry("8.8.8.8")])
        resolver = FilteringResolver(disallow=_reject_private, inner=inner)
        async with build_session(timeout_s=5.0, resolver=resolver) as session:
            assert session.connector is not None
            # aiohttp stores the resolver on TCPConnector as `_resolver`. Its
            # a private attribute — we peek at it as the only reliable way to
            # assert wiring without opening a real connection.
            assert getattr(session.connector, "_resolver", None) is resolver


class TestReadBodyCapped:
    async def test_returns_body_under_cap(self):
        body = b"x" * 100
        assert await read_body_capped(FakeReadResponse(body), max_bytes=200, label="under") == body

    async def test_returns_body_at_cap_boundary(self):
        body = b"x" * 200
        assert await read_body_capped(FakeReadResponse(body), max_bytes=200, label="at-cap") == body

    async def test_over_cap_returns_none_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = await read_body_capped(FakeReadResponse(b"x" * 201), max_bytes=200, label="mylabel")
        assert result is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "mylabel" in r.getMessage()]
        assert warnings, "expected a WARNING mentioning the label for the oversized body"


class TestFilteringResolver:
    """Direct unit tests for the connect-time DNS-vetting resolver."""

    async def test_all_public_addresses_pass_through(self):
        inner = FakeInnerResolver([_resolve_entry("8.8.8.8"), _resolve_entry("1.1.1.1")])
        r = FilteringResolver(disallow=_reject_private, inner=inner)
        got = await r.resolve("example.com")
        assert [entry["host"] for entry in got] == ["8.8.8.8", "1.1.1.1"]

    async def test_private_addresses_filtered(self):
        # Public + private mixed answer (classic DNS-rebinding-lite payload).
        # The private survivor must be dropped; the public one kept.
        inner = FakeInnerResolver([_resolve_entry("8.8.8.8"), _resolve_entry("10.0.0.5")])
        r = FilteringResolver(disallow=_reject_private, inner=inner)
        got = await r.resolve("mixed.example.com")
        assert [entry["host"] for entry in got] == ["8.8.8.8"]

    async def test_all_disallowed_raises_os_error(self):
        # If every address is filtered out, mirror getaddrinfo semantics and
        # raise OSError so the fetch layer's existing except-clause catches it.
        inner = FakeInnerResolver([_resolve_entry("127.0.0.1"), _resolve_entry("10.0.0.5")])
        r = FilteringResolver(disallow=_all_disallowed, inner=inner)
        with pytest.raises(OSError, match="all resolved addresses disallowed"):
            await r.resolve("evil.example.com")

    async def test_close_delegates_to_inner(self):
        inner = FakeInnerResolver([_resolve_entry("8.8.8.8")])
        r = FilteringResolver(disallow=_reject_private, inner=inner)
        assert inner.close_called is False
        await r.close()
        assert inner.close_called is True
