"""Shared aiohttp HTTP utilities for research providers.

Right-sized extraction (2026-07 resolution-source plan): only the genuinely
generic pieces live here — session construction and a size-capped body read.
Retry/backoff logic stays provider-private (prediction_market's is JSON-API
shaped; the resolution-source fetcher deliberately does no retries).
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Any, Awaitable, Callable, cast

import aiohttp
import aiohttp.abc
import aiohttp.resolver

logger = logging.getLogger(__name__)

# Sentinel used only as default for AddressFamily; referenced in FilteringResolver.resolve.
_DEFAULT_FAMILY: socket.AddressFamily = socket.AF_INET


IpAddr = ipaddress.IPv4Address | ipaddress.IPv6Address


class FilteringResolver(aiohttp.abc.AbstractResolver):
    """DNS resolver that vets each resolved IP against a caller-supplied predicate.

    Motivation: :func:`is_public_http_url` resolves the target host and rejects
    on a private-IP hit, but ``aiohttp.TCPConnector`` performs its OWN
    resolution at connect time. A DNS-rebinding server that returned a public
    IP to the preflight and a private IP to the connect would slip past the
    guard (classic TOCTOU). This resolver runs at connect time — the same
    layer as aiohttp's DNS cache — so every IP actually dialed has been vetted
    and any redirect hop is re-resolved through the same filter.

    ``disallow(ip)`` returns True to REJECT an address. Callers pass e.g.
    ``_ip_is_disallowed`` from ``resolution_source.py``.

    If every resolved address is filtered out, ``resolve`` raises ``OSError``
    (mirroring how ``getaddrinfo``-based resolvers surface unusable results,
    so the fetch layer's existing except-clause catches it uniformly).
    """

    def __init__(
        self,
        *,
        disallow: Callable[[IpAddr], bool],
        inner: aiohttp.abc.AbstractResolver | None = None,
    ) -> None:
        self._disallow = disallow
        self._inner: aiohttp.abc.AbstractResolver = inner or aiohttp.resolver.ThreadedResolver()

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = _DEFAULT_FAMILY,
    ) -> list[Any]:
        raw = await self._inner.resolve(host, port, family)
        survivors: list[Any] = []
        for entry in raw:
            ip_str = entry["host"]
            try:
                parsed = ipaddress.ip_address(ip_str)
            except ValueError:
                # Non-parseable address string — treat conservatively as disallowed.
                continue
            if not self._disallow(parsed):
                survivors.append(entry)
        if not survivors:
            raise OSError(f"all resolved addresses disallowed for {host}")
        return survivors

    async def close(self) -> None:
        await self._inner.close()


# Safari-like UA + full Accept / Accept-Language / Accept-Encoding.
# FINDINGS (resolution_source_probe): this exact header set recovered
# 6 extra sources vs Chrome-UA-only (38/50 vs 32/50).
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # Advertise only codecs the runtime can decode: aiohttp needs the `brotli`
    # package for `br` (HAS_BROTLI=False here — not a project dep). If we
    # advertised `br` anyway, a Brotli-preferring server would send it and
    # aiohttp would raise ClientResponseError on decode, silently dropping the
    # source. Servers fall back to gzip/deflate cleanly.
    "Accept-Encoding": "gzip, deflate",
}


def build_session(
    *,
    timeout_s: float,
    connector_limit: int = 20,
    headers: dict[str, str] | None = None,
    resolver: aiohttp.abc.AbstractResolver | None = None,
) -> aiohttp.ClientSession:
    """Construct a fresh aiohttp session with total + sock_read timeouts and a connection cap.

    ``headers=None`` (the default) adds no session-level headers — prediction_market's
    JSON-API calls rely on that; the resolution-source fetcher passes BROWSER_HEADERS.

    ``resolver=None`` (the default) uses aiohttp's built-in ThreadedResolver.
    Callers that need to vet resolved IPs (SSRF-sensitive fetchers) pass a
    :class:`FilteringResolver` so aiohttp's own connect-time DNS lookup goes
    through the same predicate as the preflight guard — closing the classic
    DNS-rebinding TOCTOU.
    """
    timeout = aiohttp.ClientTimeout(total=timeout_s, sock_read=timeout_s)
    connector_kwargs: dict[str, Any] = {"limit": connector_limit}
    if resolver is not None:
        connector_kwargs["resolver"] = resolver
    connector = aiohttp.TCPConnector(**connector_kwargs)
    return aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers)


async def read_body_capped(resp: Any, *, max_bytes: int, label: str) -> bytes | None:
    """Read a response body in full, rejecting bodies over ``max_bytes``.

    Uses ``resp.read()`` (full decompressed body) then checks size, rather than
    ``resp.content.read(n)`` which returns only whatever is in the internal
    buffer — causing silent truncation on chunked/brotli responses.

    Returns None on an oversized body (logged at WARNING with ``label``).
    """
    raw = await cast("Awaitable[bytes]", resp.read())
    if len(raw) > max_bytes:
        logger.warning(f"{label} response too large ({len(raw)} bytes > {max_bytes}); dropping")
        return None  # noqa: ASYNC910
    return raw
