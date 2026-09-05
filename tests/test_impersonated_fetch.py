"""The TLS-impersonation transport (`metaculus_bot/research/impersonated_fetch.py`): its SSRF
procedure, its bounds, and the exact libcurl options it builds a session with.

Sits beside ``tests/test_http_fetch.py`` (the aiohttp transport's ``FilteringResolver``,
``build_session`` and ``read_body_capped``) and ``tests/test_rendered_fetch.py`` (the browser
transport). Most tests here open no libcurl handle: the seam is the ``AsyncSession`` NAME bound in
``impersonated_fetch``, which :func:`install_fake_curl` replaces with a fake CLASS recording the
constructor kwargs (the pin operand, the proxy options, the impersonation target, the timeout) and
serving a scripted body through the same ``content_callback`` the real session calls. The fake
patches only the transport's module attribute, so the root conftest's ``_block_native_egress``
still patches the REAL ``AsyncSession.request`` underneath, and :class:`TestEgressGuard` calls the
unfaked transport to prove that guard is what fires.

Three classes DO reach the network, on loopback only, under ``@pytest.mark.allow_network``: the
gzip-bomb and trickle-then-stall regressions and the two real-failure mapping cases, because the
facts they pin (the body cap bounds resident memory, libcurl's own ``TIMEOUT_MS`` bounds a stalled
transfer, a plaintext listener behind an ``https`` URL raises ``SSLError``) live in libcurl and
cannot be faked. They monkeypatch the pin helper to accept ``127.0.0.1`` so the SSRF vetting, which
rightly refuses loopback, does not stand in the way of the mechanism under test.

The DNS the SSRF vetting resolves through is stubbed per test on ``resolution_source.socket``, the
one module every reader of the guard resolves it on, so a hostname's addresses are whatever the
test says and nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import gzip
import logging
import socket
import time
import tracemalloc
import typing
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import certifi
import curl_cffi.requests.impersonate
import pytest
from curl_cffi import CurlError, CurlOpt
from curl_cffi.const import CurlECode
from curl_cffi.curl import CURL_WRITEFUNC_ERROR
from curl_cffi.requests import Headers, Response
from curl_cffi.requests import exceptions as curl_exceptions

from metaculus_bot.constants import IMPERSONATE_BROWSER_TARGET, RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S
from metaculus_bot.research import impersonated_fetch, rendered_fetch, resolution_source
from metaculus_bot.research.http_fetch import MAX_REDIRECTS
from metaculus_bot.research.impersonated_fetch import (
    IMPERSONATE_BLOCK_STATUSES,
    IMPERSONATE_TRIGGER_STATUSES,
    ImpersonateBodyTooLarge,
    ImpersonateBudgetExhausted,
    ImpersonateDeclined,
    ImpersonatedResponse,
    ImpersonateHopRefused,
    ImpersonatePinNotHeld,
    ImpersonateRedirectLimit,
    ImpersonateTransportError,
    ImpersonateUnpinnable,
    declared_pdf,
    fetch_impersonated,
    impersonation_enabled,
    impersonation_refused,
    note_impersonation_refused,
    note_refusal_if_block_shaped,
    reset_impersonation_memo,
)
from metaculus_bot.research.resolution_fetch_result import _NON_OK_FETCH_STATUS

_URL = "https://www.example.com/report"
_HOST = "www.example.com"
_PUBLIC_IP = "93.184.216.34"
_OTHER_HOST = "data.example.org"
# A second GLOBAL address for the second host. Not a TEST-NET range: 203.0.113.0/24 and its
# siblings are IANA special-purpose, so ``ipaddress`` reports them as not global and the vetting
# predicate would refuse them, which would test the refusal rather than the re-pin.
_OTHER_IP = "8.8.4.4"
_V6_HOST = "v6.example.com"
_V6_IP = "2606:2800:220:1:248:1893:25c8:1946"
_TRANSPORT_LOGGER = "metaculus_bot.research.impersonated_fetch"


# ---------------------------------------------------------------------------
# The fake for the curl chokepoint
# ---------------------------------------------------------------------------


class FakeResponse:
    """A stub for the completed ``curl_cffi.requests.Response`` the non-stream path returns.

    Cannot be the real class without a live curl handle. Carries what the transport reads off a
    completed response: ``status_code``, a genuine case-insensitive ``Headers``, ``primary_ip``
    and ``redirect_url``. ``chunks`` is the body the fake feeds to the transport's write callback,
    one at a time, so the cap boundary is exercised exactly as libcurl would.
    """

    def __init__(
        self,
        status: int = 200,
        *,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
        chunks: list[bytes] | None = None,
        primary_ip: str = _PUBLIC_IP,
        redirect_url: str = "",
        body_delay_s: float = 0.0,
    ) -> None:
        self.status_code = status
        self.headers = Headers(headers or {})
        self.primary_ip = primary_ip
        self.redirect_url = redirect_url
        self.chunks_read = 0
        self._chunks = list(chunks) if chunks is not None else ([body] if body else [])
        self._body_delay_s = body_delay_s

    async def feed(self, content_callback: Callable[[bytes], int]) -> None:
        """Hand each chunk to the write callback; a ``CURL_WRITEFUNC_ERROR`` return aborts.

        Mirrors libcurl calling the write function per received chunk and cutting the transfer
        the moment the function refuses one: the refused chunk raises the same
        ``RequestException`` (code 23) with this response attached that curl-cffi's non-stream
        path raises, so the transport's ``reader.tripped`` branch sees exactly what production does.
        """
        for chunk in self._chunks:
            if self._body_delay_s:
                await asyncio.sleep(self._body_delay_s)
            self.chunks_read += 1
            if content_callback(chunk) == CURL_WRITEFUNC_ERROR:
                raise curl_exceptions.RequestException("aborted by write callback", CurlECode.WRITE_ERROR, self)


# A scripted hop: a ready response, an exception to raise where ``request`` would, or a callable
# of the URL producing either (for tests whose response depends on which host was dialed).
_Scripted = FakeResponse | BaseException | Callable[[str], "FakeResponse | BaseException"]


class FakeCurl:
    """What :func:`install_fake_curl` returns: every session built, and the in-flight peaks."""

    def __init__(self, script: list[_Scripted]) -> None:
        self.script = script
        self.sessions: list[FakeCurlSession] = []
        self.host_inflight: dict[str, int] = {}
        self.host_peak: dict[str, int] = {}
        self.inflight = 0
        self.peak = 0

    def next_response(self, url: str) -> FakeResponse | BaseException:
        if not self.script:
            raise AssertionError(f"no scripted response for {url}")
        entry = self.script.pop(0)
        return entry(url) if callable(entry) and not isinstance(entry, FakeResponse) else entry

    @property
    def resolve_operands(self) -> list[list[str] | str]:
        return [session.kwargs["curl_options"][CurlOpt.RESOLVE] for session in self.sessions]


class FakeCurlSession:
    """Stands in for the ``AsyncSession`` CLASS. One instance per constructor call, exactly as the
    transport builds one per dial; ``kwargs`` is what the pin and option assertions read."""

    def __init__(self, recorder: FakeCurl, **kwargs: Any) -> None:
        self.recorder = recorder
        self.kwargs = kwargs
        self.requests: list[tuple[str, str]] = []
        self.closed = False
        recorder.sessions.append(self)

    async def __aenter__(self) -> FakeCurlSession:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        self.closed = True

    async def request(self, method: str, url: str, *, content_callback: Callable[[bytes], int]) -> FakeResponse:
        self.requests.append((method, url))
        recorder = self.recorder
        host = urlparse(url).netloc
        recorder.host_inflight[host] = recorder.host_inflight.get(host, 0) + 1
        recorder.host_peak[host] = max(recorder.host_peak.get(host, 0), recorder.host_inflight[host])
        recorder.inflight += 1
        recorder.peak = max(recorder.peak, recorder.inflight)
        try:
            # The real request yields to the loop before any body arrives.
            await asyncio.sleep(0)
            response = recorder.next_response(url)
            if isinstance(response, BaseException):
                raise response
            await response.feed(content_callback)
            return response
        finally:
            recorder.host_inflight[host] -= 1
            recorder.inflight -= 1


def install_fake_curl(monkeypatch: pytest.MonkeyPatch, *script: _Scripted) -> FakeCurl:
    """Bind ``impersonated_fetch.AsyncSession`` to a fake class serving ``script`` dial by dial.

    Patches the transport's own module attribute, never ``curl_cffi.requests.AsyncSession`` and
    never its ``request`` method, so the root conftest's native-egress guard stays armed on the
    real class underneath.
    """
    recorder = FakeCurl(list(script))
    monkeypatch.setattr(impersonated_fetch, "AsyncSession", functools.partial(FakeCurlSession, recorder))
    return recorder


def _redirect(location: str, *, redirect_url: str | None = None, status: int = 302, **kwargs: Any) -> FakeResponse:
    """A 3xx hop. ``redirect_url`` is libcurl's absolutised target; by default it equals the header."""
    return FakeResponse(
        status,
        headers={"Location": location},
        redirect_url=location if redirect_url is None else redirect_url,
        **kwargs,
    )


def _page(body: bytes = b"<html>ok</html>", **kwargs: Any) -> FakeResponse:
    headers = {"Content-Type": "text/html; charset=utf-8", "Server": "AkamaiGHost"}
    headers.update(kwargs.pop("headers", {}))
    return FakeResponse(200, headers=headers, body=body, **kwargs)


def _pdf(body: bytes, **kwargs: Any) -> FakeResponse:
    """A 200 declaring ``application/pdf``, for the two-cap re-dial tests."""
    headers = {"Content-Type": "application/pdf", "Server": "AkamaiGHost"}
    headers.update(kwargs.pop("headers", {}))
    return FakeResponse(200, headers=headers, body=body, **kwargs)


async def _fetch(
    url: str = _URL,
    *,
    host_sems: dict[str, asyncio.Semaphore] | None = None,
    deadline_in_s: float = 30.0,
    per_hop_timeout_s: float = 20.0,
    max_bytes: int = 1024 * 1024,
    document_max_bytes: int | None = None,
) -> ImpersonatedResponse:
    return await fetch_impersonated(
        url,
        host_sems={} if host_sems is None else host_sems,
        deadline_monotonic_s=time.monotonic() + deadline_in_s,
        per_hop_timeout_s=per_hop_timeout_s,
        max_bytes=max_bytes,
        document_max_bytes=max_bytes if document_max_bytes is None else document_max_bytes,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_memo():
    reset_impersonation_memo()
    yield
    reset_impersonation_memo()


@pytest.fixture(autouse=True)
def dns(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Hostname to addresses, read by the stubbed ``getaddrinfo`` the SSRF vetting resolves through.

    Autouse so no test's lookup leaves the process. Unknown hosts resolve to the one public
    example.com address; a test that wants a private, mixed or IPv6 answer writes its entry.
    """
    table: dict[str, list[str]] = {_OTHER_HOST: [_OTHER_IP], _V6_HOST: [_V6_IP]}

    def _getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        del port, args, kwargs
        infos: list[tuple[Any, ...]] = []
        for ip in table.get(host, [_PUBLIC_IP]):
            if ":" in ip:
                infos.append((socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, 0, 0, 0)))
            else:
                infos.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)))
        return infos

    monkeypatch.setattr(resolution_source.socket, "getaddrinfo", _getaddrinfo)
    return table


# ---------------------------------------------------------------------------
# The shared rung policy
# ---------------------------------------------------------------------------


class TestPolicy:
    """The module owns the trigger set, the block set, the kill switch and the memo rule, so both
    ladders read one thing. See F10 in the fix plan for why they moved off ``resolution_source``."""

    def test_the_trigger_set_is_403_only(self) -> None:
        assert frozenset({403}) == IMPERSONATE_TRIGGER_STATUSES

    def test_the_block_set_is_the_blocked_rows_plus_the_two_edge_refusals(self) -> None:
        """Every status the fetch vocabulary calls ``blocked``, read off the one table both fetch
        paths already share rather than spelled a third time, plus 401 and 503: an authentication
        wall and a challenge interstitial the edge puts up for the impersonated client. Neither is
        a ``blocked`` row, so the rung keeps stamping ``error`` for them (a telemetry table is not
        touched to widen a memo), and without them a host refusing both clients with a 503 earned a
        full impersonated dial per cited URL inside one provider wall."""
        blocked_rows = frozenset(status for status, token in _NON_OK_FETCH_STATUS.items() if token == "blocked")

        assert blocked_rows < IMPERSONATE_BLOCK_STATUSES
        assert frozenset({401, 403, 406, 429, 503}) == IMPERSONATE_BLOCK_STATUSES
        assert _NON_OK_FETCH_STATUS.get(401, "error") == "error"
        assert _NON_OK_FETCH_STATUS.get(503, "error") == "error"

    def test_the_kill_switch_defaults_on_and_honours_an_explicit_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RESOLUTION_SOURCE_IMPERSONATE_ENABLED", raising=False)
        assert impersonation_enabled() is True
        monkeypatch.setenv("RESOLUTION_SOURCE_IMPERSONATE_ENABLED", "false")
        assert impersonation_enabled() is False
        monkeypatch.setenv("RESOLUTION_SOURCE_IMPERSONATE_ENABLED", "true")
        assert impersonation_enabled() is True

    def test_a_block_shaped_answer_memoises_both_netlocs(self) -> None:
        """The host that ANSWERED the block is memoised, and the host that was DIALED, so a
        redirect-hop block bans the refusing host and does not re-walk the same chain this run."""
        wrote = note_refusal_if_block_shaped(
            dialed_url="https://first.example.com/x", answered_url="https://final.example.com/y", status=403
        )

        assert wrote is True
        assert impersonation_refused("https://final.example.com/z")
        assert impersonation_refused("https://first.example.com/w")
        assert not impersonation_refused("https://elsewhere.example.com/")

    @pytest.mark.parametrize("status", sorted(IMPERSONATE_BLOCK_STATUSES))
    def test_every_block_status_writes_the_memo(self, status: int) -> None:
        assert note_refusal_if_block_shaped(dialed_url=_URL, answered_url=_URL, status=status) is True
        assert impersonation_refused(_URL)

    @pytest.mark.parametrize("status", [200, 404, 410, 500])
    def test_a_non_block_answer_writes_nothing(self, status: int) -> None:
        assert note_refusal_if_block_shaped(dialed_url=_URL, answered_url=_URL, status=status) is False
        assert not impersonation_refused(_URL)


# ---------------------------------------------------------------------------
# The library contract (F7): facts the transport borrows from the installed curl_cffi
# ---------------------------------------------------------------------------


class TestLibraryContract:
    """No network. The transport's doubles cannot catch a curl-cffi bump that rotates the profile
    out of the literal or renames an attribute ``_dial`` reads, so pin those facts against the real
    installed package here — the sibling transport does the same in ``tests/test_rendered_fetch.py``."""

    def test_the_pinned_profile_is_a_real_concrete_profile_not_the_alias(self) -> None:
        """A curl-cffi bump that drops ``chrome146`` from ``BrowserTypeLiteral`` would make every
        dial raise ``ImpersonateError`` (a ``RequestException``), which the rung would log as a
        host fact and decline on silently. This turns that bump red at CI time instead."""
        assert IMPERSONATE_BROWSER_TARGET in typing.get_args(curl_cffi.requests.impersonate.BrowserTypeLiteral)
        assert IMPERSONATE_BROWSER_TARGET != "chrome"

    def test_a_real_response_exposes_every_attribute_the_transport_reads(self) -> None:
        """A bare real ``Response`` instance carries what ``_dial`` and ``_Hop`` read off the
        completed response. Asserted on an instance because the initialiser is where curl-cffi
        sets them; the class has none."""
        response = Response()
        for attribute in ("status_code", "headers", "primary_ip", "redirect_url"):
            assert hasattr(response, attribute), attribute

    def test_the_write_error_constants_the_cap_relies_on_exist(self) -> None:
        """The cap returns ``CURL_WRITEFUNC_ERROR`` and reads back curl error 23; a rename of
        either would let an oversized body through as a transport failure."""
        assert CURL_WRITEFUNC_ERROR == 0xFFFFFFFF
        assert int(CurlECode.WRITE_ERROR) == 23

    def test_the_write_error_carries_the_response_the_cap_reads_the_content_type_off(self) -> None:
        """``ImpersonateBodyTooLarge`` reads the declared Content-Type off the aborted request's
        attached response to decide the PDF re-dial, so the exception has to carry one."""
        response = Response()
        exc = curl_exceptions.RequestException("aborted", CurlECode.WRITE_ERROR, response)
        assert exc.response is response


# ---------------------------------------------------------------------------
# The pin operand
# ---------------------------------------------------------------------------


class TestPinOperand:
    async def test_the_pin_is_built_from_the_vetted_address(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``CURLOPT_RESOLVE`` takes a LIST, and a bare string is a silent, type-checked bug.

        ``Curl.setopt`` special-cases the option and ITERATES the value, appending each element to
        a libcurl slist. A bare ``"host:443:ip"`` therefore iterates character by character, curl
        rejects the entry ``'p'`` at perform time, and in the general case libcurl falls back to
        its own resolver so the request still succeeds: a malformed pin fails OPEN. curl-cffi's own
        type hint is ``dict[CurlOpt, str]``, so basedpyright blesses the string form. Only this
        assertion on the exact operand catches a regression.
        """

        async def _pinned(url: str) -> tuple[str, str]:
            del url
            return (_HOST, "198.18.0.1")

        monkeypatch.setattr(rendered_fetch, "resolve_pinned_host", _pinned)
        curl = install_fake_curl(monkeypatch, _page(primary_ip="198.18.0.1"))

        await _fetch()

        operand = curl.sessions[0].kwargs["curl_options"][CurlOpt.RESOLVE]
        assert isinstance(operand, list)
        assert operand == [f"{_HOST}:443:198.18.0.1"]

    @pytest.mark.parametrize(
        ("url", "port"),
        [
            ("https://www.example.com/x", 443),
            ("http://www.example.com/x", 80),
            ("https://www.example.com:8443/x", 8443),
        ],
    )
    async def test_the_port_comes_from_the_url(self, monkeypatch: pytest.MonkeyPatch, url: str, port: int) -> None:
        """libcurl matches a RESOLVE entry on host AND port, so a pin for the wrong port is inert."""
        curl = install_fake_curl(monkeypatch, _page())

        await _fetch(url)

        assert curl.resolve_operands == [[f"{_HOST}:{port}:{_PUBLIC_IP}"]]

    async def test_an_ipv6_vetted_address_is_bracketed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        curl = install_fake_curl(monkeypatch, _page(primary_ip=_V6_IP))

        await _fetch(f"https://{_V6_HOST}/x")

        assert curl.resolve_operands == [[f"{_V6_HOST}:443:[{_V6_IP}]"]]

    async def test_an_ipv6_primary_ip_in_another_spelling_still_holds_the_pin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The pin assertion compares ADDRESSES, not strings: libcurl reports the compressed form."""
        curl = install_fake_curl(monkeypatch, _page(primary_ip="2606:2800:220:1:248:1893:25C8:1946"))

        response = await _fetch(f"https://{_V6_HOST}/x")

        assert response.status == 200
        assert len(curl.sessions) == 1

    async def test_a_malformed_port_is_a_transport_error_not_an_inert_pin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Location with a port that is not a number cannot be pinned, so it is refused rather than
        dialed under a pin that matches nothing."""
        curl = install_fake_curl(monkeypatch, _redirect("https://www.example.com:abc/x"))

        with pytest.raises(ImpersonateTransportError) as excinfo:
            await _fetch()

        assert excinfo.value.failure_class is None
        assert excinfo.value.exc == "InvalidURL"
        assert len(curl.sessions) == 1


class TestSessionOptions:
    async def test_the_proxy_options_are_always_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not belt and braces: libcurl reads ``http_proxy`` from the environment itself, and
        curl-cffi 0.15.0's documented ``trust_env`` is dead code (``grep -rn trust_env`` over the
        installed package finds its annotation, its default and one assignment, and nothing reads
        it). Measured on loopback: with ``http_proxy`` set, a ``trust_env=False`` request still went
        through the proxy, which received the absolute-form request and so the HOSTNAME, and
        ``CURLOPT_RESOLVE`` never applied: the pin was bypassed entirely. ``PROXY=""`` restored the
        direct pinned connection, as did ``NOPROXY="*"``. Both are set."""
        curl = install_fake_curl(monkeypatch, _page())

        await _fetch()

        options = curl.sessions[0].kwargs["curl_options"]
        assert options[CurlOpt.PROXY] == ""
        assert options[CurlOpt.NOPROXY] == "*"

    async def test_the_proxy_options_hold_with_a_proxy_in_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            monkeypatch.setenv(name, "http://127.0.0.1:3128")
        curl = install_fake_curl(monkeypatch, _page())

        await _fetch()

        options = curl.sessions[0].kwargs["curl_options"]
        assert options[CurlOpt.PROXY] == ""
        assert options[CurlOpt.NOPROXY] == "*"
        assert "proxy" not in curl.sessions[0].kwargs
        assert "proxies" not in curl.sessions[0].kwargs

    async def test_the_impersonation_target_is_the_pinned_profile_not_the_alias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``"chrome"`` resolves through curl-cffi's ``DEFAULT_CHROME``, a source constant, so a
        routine curl-cffi bump would move the TLS and HTTP/2 fingerprint and the User-Agent the
        federal hosts see with no diff in this repo. The concrete profile makes that a reviewable
        change."""
        curl = install_fake_curl(monkeypatch, _page())

        await _fetch()

        assert curl.sessions[0].kwargs["impersonate"] == IMPERSONATE_BROWSER_TARGET == "chrome146"
        assert curl.sessions[0].kwargs["impersonate"] != "chrome"

    async def test_redirects_are_manual_and_trust_is_certifi(self, monkeypatch: pytest.MonkeyPatch) -> None:
        curl = install_fake_curl(monkeypatch, _page())

        await _fetch()

        assert curl.sessions[0].kwargs["allow_redirects"] is False
        assert curl.sessions[0].kwargs["verify"] == certifi.where()

    async def test_cookies_are_discarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The session is single-use and nothing reads its jar, and curl_cffi parses each
        ``Set-Cookie`` line with strict UTF-8 (``requests/cookies.py``), so a hostile header would
        otherwise raise a ``UnicodeDecodeError`` out of a completed response."""
        curl = install_fake_curl(monkeypatch, _page())

        await _fetch()

        assert curl.sessions[0].kwargs["discard_cookies"] is True

    async def test_no_headers_are_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The impersonation profile supplies Chrome's complete header set (Accept, Accept-Language,
        Priority, the Sec-Fetch and sec-ch-ua families). Overriding it with the direct path's
        Safari-like ``BROWSER_HEADERS`` would present a Chrome TLS fingerprint under Safari headers,
        which is precisely the incoherence an edge scores."""
        curl = install_fake_curl(monkeypatch, _page())

        await _fetch()

        assert "headers" not in curl.sessions[0].kwargs

    async def test_one_get_is_issued_per_hop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        curl = install_fake_curl(monkeypatch, _page())

        await _fetch()

        assert curl.sessions[0].requests == [("GET", _URL)]


# ---------------------------------------------------------------------------
# Refusals before any request
# ---------------------------------------------------------------------------


class TestUnpinnable:
    async def test_a_private_address_is_refused_before_any_request(
        self, monkeypatch: pytest.MonkeyPatch, dns: dict[str, list[str]]
    ) -> None:
        dns["internal.example.com"] = ["10.0.0.5"]
        curl = install_fake_curl(monkeypatch)

        with pytest.raises(ImpersonateUnpinnable):
            await _fetch("https://internal.example.com/x")

        assert curl.sessions == []

    @pytest.mark.parametrize(
        "url",
        [
            "http://169.254.169.254/latest/meta-data/",
            "https://trusted@www.example.com/x",
            "ftp://www.example.com/x",
            "https://münchen.example/x",
            "https://www.example.com./x",
        ],
    )
    async def test_a_url_the_pin_helper_refuses_is_refused_here(
        self, monkeypatch: pytest.MonkeyPatch, url: str
    ) -> None:
        curl = install_fake_curl(monkeypatch)

        with pytest.raises(ImpersonateUnpinnable):
            await _fetch(url)

        assert curl.sessions == []

    async def test_a_mixed_resolution_rejects_the_whole_hostname(
        self, monkeypatch: pytest.MonkeyPatch, dns: dict[str, list[str]]
    ) -> None:
        dns["mixed.example.com"] = [_PUBLIC_IP, "127.0.0.1"]
        curl = install_fake_curl(monkeypatch)

        with pytest.raises(ImpersonateUnpinnable):
            await _fetch("https://mixed.example.com/x")

        assert curl.sessions == []

    async def test_the_first_hop_refusal_names_the_cited_host_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On the caller's own URL there is no redirect origin to carry, so the message names the
        host and nothing else."""

        async def _unpinnable(url: str) -> None:
            del url
            return

        monkeypatch.setattr(rendered_fetch, "resolve_pinned_host", _unpinnable)
        install_fake_curl(monkeypatch)

        with pytest.raises(ImpersonateUnpinnable) as excinfo:
            await _fetch()

        assert excinfo.value.redirected_from is None
        assert _HOST in str(excinfo.value)

    async def test_the_pin_helper_is_read_off_rendered_fetch_at_call_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One pin helper for both transports, resolved on its owning module so a patch there
        covers this caller too."""

        async def _unpinnable(url: str) -> None:
            del url
            return

        monkeypatch.setattr(rendered_fetch, "resolve_pinned_host", _unpinnable)
        curl = install_fake_curl(monkeypatch)

        with pytest.raises(ImpersonateUnpinnable):
            await _fetch()

        assert curl.sessions == []


# ---------------------------------------------------------------------------
# Redirects
# ---------------------------------------------------------------------------


class TestRedirects:
    async def test_a_redirect_to_a_private_host_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, dns: dict[str, list[str]]
    ) -> None:
        dns["internal.example.com"] = ["10.0.0.5"]
        curl = install_fake_curl(monkeypatch, _redirect("https://internal.example.com/x"))

        with pytest.raises(ImpersonateHopRefused) as excinfo:
            await _fetch()

        assert excinfo.value.refusal == "ssrf_blocked"
        assert excinfo.value.hop_url == "https://internal.example.com/x"
        assert excinfo.value.from_url == _URL
        assert len(curl.sessions) == 1

    async def test_a_redirect_to_metaculus_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        curl = install_fake_curl(monkeypatch, _redirect("https://www.metaculus.com/questions/1/"))

        with pytest.raises(ImpersonateHopRefused) as excinfo:
            await _fetch()

        assert excinfo.value.refusal == "metaculus_self_ref"
        assert len(curl.sessions) == 1

    async def test_a_non_public_self_reference_reports_ssrf_blocked(
        self, monkeypatch: pytest.MonkeyPatch, dns: dict[str, list[str]]
    ) -> None:
        """The order is a telemetry contract: the public preflight runs before the self-reference
        check, so a URL that fails both has always recorded as ``ssrf_blocked``."""
        dns["www.metaculus.com"] = ["10.0.0.5"]
        install_fake_curl(monkeypatch, _redirect("https://www.metaculus.com/questions/1/"))

        with pytest.raises(ImpersonateHopRefused) as excinfo:
            await _fetch()

        assert excinfo.value.refusal == "ssrf_blocked"

    async def test_a_redirect_is_re_pinned_on_its_own_host(self, monkeypatch: pytest.MonkeyPatch) -> None:
        target = f"https://{_OTHER_HOST}/x"
        curl = install_fake_curl(monkeypatch, _redirect(target), _page(primary_ip=_OTHER_IP))

        response = await _fetch()

        assert len(curl.sessions) == 2
        assert curl.resolve_operands == [[f"{_HOST}:443:{_PUBLIC_IP}"], [f"{_OTHER_HOST}:443:{_OTHER_IP}"]]
        assert _HOST not in curl.resolve_operands[1][0]
        assert curl.sessions[1].requests == [("GET", target)]
        assert response.url == target
        assert response.primary_ip == _OTHER_IP

    async def test_the_next_hop_comes_from_the_absolutised_redirect_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``redirect_url`` is ``CURLINFO_REDIRECT_URL``, resolved against the request URL and
        populated even with ``allow_redirects=False``; the raw ``Location`` stays relative."""
        curl = install_fake_curl(
            monkeypatch,
            _redirect("/deep/other?a=1", redirect_url="https://www.example.com/deep/other?a=1"),
            _page(),
        )

        await _fetch()

        assert curl.sessions[1].requests == [("GET", "https://www.example.com/deep/other?a=1")]

    async def test_a_relative_location_is_joined_when_redirect_url_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        curl = install_fake_curl(monkeypatch, _redirect("/other", redirect_url=""), _page())

        await _fetch()

        assert curl.sessions[1].requests == [("GET", "https://www.example.com/other")]

    async def test_a_redirect_with_no_target_is_a_transport_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Mirrors the direct path's malformed-redirect branch: no failure class, the shape in ``exc``."""
        install_fake_curl(monkeypatch, FakeResponse(302))

        with pytest.raises(ImpersonateTransportError) as excinfo:
            await _fetch()

        assert excinfo.value.failure_class is None
        assert excinfo.value.exc == "MissingLocation"

    @pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
    async def test_every_redirect_status_is_followed(self, monkeypatch: pytest.MonkeyPatch, status: int) -> None:
        curl = install_fake_curl(monkeypatch, _redirect("https://www.example.com/next", status=status), _page())

        response = await _fetch()

        assert response.status == 200
        assert len(curl.sessions) == 2

    async def test_the_redirect_cap_is_max_redirects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hops = [_redirect(f"https://www.example.com/hop{i}") for i in range(MAX_REDIRECTS + 1)]
        curl = install_fake_curl(monkeypatch, *hops)

        with pytest.raises(ImpersonateRedirectLimit):
            await _fetch()

        assert len(curl.sessions) == MAX_REDIRECTS + 1

    async def test_max_redirects_hops_then_a_page_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        hops = [_redirect(f"https://www.example.com/hop{i}") for i in range(MAX_REDIRECTS)]
        curl = install_fake_curl(monkeypatch, *hops, _page())

        response = await _fetch()

        assert response.status == 200
        assert len(curl.sessions) == MAX_REDIRECTS + 1

    async def test_a_redirected_hop_that_is_unpinnable_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, dns: dict[str, list[str]]
    ) -> None:
        """The hop refusal and the pin helper agree on what is refusable; a later hop the pin helper
        declines that the refusal let through reports ``ImpersonateUnpinnable``, carrying the hop it
        was reached from so the operator sees which host failed to resolve."""

        async def _unpinnable_second_hop(url: str) -> tuple[str, str] | None:
            if "second" in url:
                return None
            return (_HOST, _PUBLIC_IP)

        monkeypatch.setattr(rendered_fetch, "resolve_pinned_host", _unpinnable_second_hop)
        curl = install_fake_curl(monkeypatch, _redirect("https://www.example.com/second"))

        with pytest.raises(ImpersonateUnpinnable) as excinfo:
            await _fetch()

        assert excinfo.value.redirected_from == _URL
        assert len(curl.sessions) == 1


# ---------------------------------------------------------------------------
# The body cap
# ---------------------------------------------------------------------------


class TestBodyCap:
    async def test_the_body_cap_aborts_the_transfer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Past the cap the write callback returns ``CURL_WRITEFUNC_ERROR``, which libcurl turns
        into a ``WRITE_ERROR`` the transport reads as ``ImpersonateBodyTooLarge``."""
        install_fake_curl(monkeypatch, _page(chunks=[b"a" * 600, b"b" * 600]))

        with pytest.raises(ImpersonateBodyTooLarge) as excinfo:
            await _fetch(max_bytes=1000)

        assert excinfo.value.max_bytes == 1000
        assert excinfo.value.bytes_read > 1000

    async def test_a_body_exactly_at_the_cap_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The boundary is ``>`` and not ``>=``, matching ``read_body_capped``."""
        install_fake_curl(monkeypatch, _page(chunks=[b"a" * 500, b"b" * 500]))

        response = await _fetch(max_bytes=1000)

        assert len(response.body) == 1000

    async def test_the_body_is_the_joined_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_curl(monkeypatch, _page(chunks=[b"<html>", b"ok", b"</html>"]))

        response = await _fetch()

        assert response.body == b"<html>ok</html>"


class TestTwoCapReDial:
    """F12: the page cap is dialed first because the write callback cannot see the headers; a body
    that aborts at it and declares a PDF earns ONE re-dial under the larger document cap, so the
    rung is no narrower than the direct path, which reads a declared PDF under the document cap."""

    async def test_an_oversize_pdf_within_the_document_cap_succeeds_after_one_re_dial(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        curl = install_fake_curl(monkeypatch, _pdf(b"P" * 3000), _pdf(b"P" * 3000))

        response = await _fetch(max_bytes=1000, document_max_bytes=5000)

        assert response.status == 200
        assert response.content_type == "application/pdf"
        assert len(response.body) == 3000
        assert len(curl.sessions) == 2
        assert curl.sessions[0].requests == curl.sessions[1].requests == [("GET", _URL)]

    async def test_an_oversize_non_document_body_is_declined_with_no_re_dial(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        curl = install_fake_curl(monkeypatch, _page(body=b"h" * 3000))

        with pytest.raises(ImpersonateBodyTooLarge):
            await _fetch(max_bytes=1000, document_max_bytes=5000)

        assert len(curl.sessions) == 1

    async def test_an_oversize_pdf_beyond_the_document_cap_is_declined_after_exactly_one_re_dial(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        curl = install_fake_curl(monkeypatch, _pdf(b"P" * 8000), _pdf(b"P" * 8000))

        with pytest.raises(ImpersonateBodyTooLarge) as excinfo:
            await _fetch(max_bytes=1000, document_max_bytes=5000)

        assert excinfo.value.max_bytes == 5000
        assert len(curl.sessions) == 2

    async def test_no_re_dial_when_the_two_caps_are_equal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both callers pass the page cap as ``max_bytes``; when they also pass it as
        ``document_max_bytes`` (an undeclared body, or a caller that never widens), a PDF over it
        is declined without the extra request."""
        curl = install_fake_curl(monkeypatch, _pdf(b"P" * 3000))

        with pytest.raises(ImpersonateBodyTooLarge):
            await _fetch(max_bytes=1000, document_max_bytes=1000)

        assert len(curl.sessions) == 1

    async def test_a_re_dial_that_no_longer_declares_a_document_is_held_to_the_page_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The larger cap is EARNED by the first dial's declared type and applied to the second
        dial, so a host could declare ``application/pdf`` once and then serve a page-cap-busting
        HTML body under the document cap. The re-dial honours the document cap only while the
        response still declares a document; anything else over the page cap is the same decline
        the first dial would have produced, with the page cap on it."""
        curl = install_fake_curl(monkeypatch, _pdf(b"P" * 3000), _page(body=b"h" * 3000))

        with pytest.raises(ImpersonateBodyTooLarge) as excinfo:
            await _fetch(max_bytes=1000, document_max_bytes=5000)

        assert excinfo.value.max_bytes == 1000
        assert excinfo.value.bytes_read == 3000
        assert excinfo.value.content_type == "text/html; charset=utf-8"
        assert len(curl.sessions) == 2

    async def test_a_re_dial_that_no_longer_declares_a_document_but_fits_the_page_cap_is_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A body under the page cap was never the abuse: whatever it declares, the first dial would
        have returned it, so the second does too."""
        curl = install_fake_curl(monkeypatch, _pdf(b"P" * 3000), _page(body=b"h" * 800))

        response = await _fetch(max_bytes=1000, document_max_bytes=5000)

        assert response.status == 200
        assert len(response.body) == 800
        assert len(curl.sessions) == 2


class TestBodyCapAgainstLibcurl:
    """The retention bound is a libcurl fact, so pin it on loopback against the real wheel."""

    @pytest.mark.allow_network
    async def test_a_gzip_bomb_is_bounded_at_the_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stream mode buffered 370 MiB of a 400 MiB body before its consumer-side cap fired; the
        write callback keeps the cap resident and aborts. Retention is asserted through
        ``tracemalloc``: a 200 MiB inflation stays orders of magnitude below its size."""
        inflated = 200 * 1024 * 1024
        compressed = gzip.compress(b"\0" * inflated, compresslevel=9)

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await _read_request_head(reader)
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Encoding: gzip\r\n"
                b"Server: bomb\r\n" + f"Content-Length: {len(compressed)}\r\n".encode() + b"Connection: close\r\n\r\n"
            )
            writer.write(compressed)
            with contextlib.suppress(*_DRAIN_ERRORS):
                await writer.drain()
            writer.close()

        cap = 2 * 1024 * 1024
        async with _loopback(monkeypatch, handler) as url:
            tracemalloc.start()
            started = time.monotonic()
            with pytest.raises(ImpersonateBodyTooLarge) as excinfo:
                await _fetch(url, max_bytes=cap, document_max_bytes=cap)
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        elapsed = time.monotonic() - started
        # The cap plus at most one crossing chunk, never the whole inflation.
        assert cap < excinfo.value.bytes_read < inflated // 2
        # Resident tracked bytes stay a small multiple of the cap, not the 200 MiB body.
        assert peak < 8 * cap, f"tracemalloc peak {peak} exceeded 8x the {cap} byte cap"
        # And it aborts promptly rather than draining the transfer.
        assert elapsed < 5.0

    @pytest.mark.allow_network
    async def test_a_trickle_then_stall_is_cut_by_libcurls_own_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-stream mode sets ``CURLOPT_TIMEOUT_MS`` from ``timeout=``, so a body that trickles
        then stalls exits at the per-hop timeout rather than the 9-to-27 s overshoot stream mode's
        low-speed cutoff produced. The wall here is libcurl's, not an ``asyncio.timeout``."""

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await _read_request_head(reader)
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 100000\r\n\r\n")
            for _ in range(3):
                writer.write(b"x" * 100)
                with contextlib.suppress(*_DRAIN_ERRORS):
                    await writer.drain()
                await asyncio.sleep(0.05)
            await asyncio.sleep(30)
            writer.close()

        per_hop = 1.0
        async with _loopback(monkeypatch, handler) as url:
            started = time.monotonic()
            with pytest.raises(ImpersonateTransportError) as excinfo:
                await _fetch(url, deadline_in_s=30.0, per_hop_timeout_s=per_hop)
            elapsed = time.monotonic() - started

        assert excinfo.value.failure_class == "timeout"
        # libcurl fires at the timeout; allow a small margin for the abort and teardown.
        assert per_hop <= elapsed < per_hop + 1.5, f"exited at {elapsed:.2f}s against a {per_hop}s timeout"


# ---------------------------------------------------------------------------
# The wall bound
# ---------------------------------------------------------------------------


class TestTimeouts:
    async def test_a_past_deadline_makes_no_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A budget already spent is its own decline, not a ``timeout`` transport error: nothing
        was dialed, so the Tier-1 rung records the provider's wall binding rather than an attempt
        that fired against the host."""
        curl = install_fake_curl(monkeypatch, _page())

        with pytest.raises(ImpersonateBudgetExhausted) as excinfo:
            await _fetch(deadline_in_s=-1.0)

        assert excinfo.value.waiting_on == "the vetting lookup"
        assert curl.sessions == []

    def test_sizing_a_hop_timeout_on_a_spent_budget_is_budget_exhausted(self) -> None:
        """The last pre-dial check, run with the gate already held: the budget spent in the queue
        behind the gate's holder is the same decline as a gate never acquired."""
        with pytest.raises(ImpersonateBudgetExhausted) as excinfo:
            impersonated_fetch._hop_timeout_s(time.monotonic() - 0.01, 20.0)

        assert excinfo.value.waiting_on == "the host gate"

    async def test_the_per_hop_timeout_is_clamped_to_the_remaining_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        curl = install_fake_curl(monkeypatch, _page())

        await _fetch(deadline_in_s=3.0, per_hop_timeout_s=20.0)

        assert curl.sessions[0].kwargs["timeout"] == pytest.approx(3.0, abs=0.25)
        assert curl.sessions[0].kwargs["timeout"] <= 20.0

    async def test_the_per_hop_timeout_never_exceeds_the_per_hop_ceiling(self, monkeypatch: pytest.MonkeyPatch) -> None:
        curl = install_fake_curl(monkeypatch, _page())

        await _fetch(deadline_in_s=60.0, per_hop_timeout_s=20.0)

        assert curl.sessions[0].kwargs["timeout"] == 20.0

    async def test_a_nearly_spent_budget_still_gets_the_floor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        curl = install_fake_curl(monkeypatch, _page())

        await _fetch(deadline_in_s=0.05, per_hop_timeout_s=20.0)

        assert curl.sessions[0].kwargs["timeout"] == RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S

    async def test_the_timeout_is_sized_after_the_host_gate_is_held(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """R22: the per-hop timeout is sized AFTER the gate, so a hop that queued behind a slow host
        does not help itself to a fresh ceiling. Hold the target host's gate, burn a slice of the
        deadline, release, and assert the recorded ``timeout`` reflects the budget remaining after
        the wait rather than the whole deadline. All three other timeout tests run uncontended, so
        moving the sizing above the gate would leave them green while reintroducing the bug."""
        curl = install_fake_curl(monkeypatch, _page())
        sems: dict[str, asyncio.Semaphore] = {}
        gate = impersonated_fetch.semaphore_for_host(_URL, sems)
        await gate.acquire()

        fetch_task = asyncio.ensure_future(_fetch(_URL, host_sems=sems, deadline_in_s=5.0, per_hop_timeout_s=20.0))
        await asyncio.sleep(1.0)
        gate.release()
        await fetch_task

        # 5s deadline minus the ~1s the gate was held: well under the deadline and the ceiling.
        assert curl.sessions[0].kwargs["timeout"] == pytest.approx(4.0, abs=0.5)
        assert curl.sessions[0].kwargs["timeout"] < 5.0

    async def test_a_gate_held_past_the_deadline_declines_without_dialing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R14: the gate acquire is bounded by the budget, so a hop that never gets the gate before
        the deadline declines as a spent budget rather than dialing late or waiting forever."""
        curl = install_fake_curl(monkeypatch, _page())
        sems: dict[str, asyncio.Semaphore] = {}
        gate = impersonated_fetch.semaphore_for_host(_URL, sems)
        await gate.acquire()

        with pytest.raises(ImpersonateBudgetExhausted) as excinfo:
            await _fetch(_URL, host_sems=sems, deadline_in_s=0.2, per_hop_timeout_s=20.0)

        assert excinfo.value.waiting_on == "the host gate"
        assert curl.sessions == []
        gate.release()

    async def test_elapsed_covers_the_whole_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_curl(monkeypatch, _redirect("https://www.example.com/next"), _page(body_delay_s=0.02))

        response = await _fetch()

        assert response.elapsed_s >= 0.02


class TestBudgetBoundedLookup:
    async def test_a_lookup_that_outlives_the_budget_declines_as_budget_exhausted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """R14: the vetting lookup is an uncancellable ``getaddrinfo`` thread, so it is awaited
        inside the remaining budget; one that does not answer in time declines without dialing."""

        async def _slow_resolve(url: str) -> tuple[str, str]:
            del url
            await asyncio.sleep(5.0)
            return (_HOST, _PUBLIC_IP)

        monkeypatch.setattr(rendered_fetch, "resolve_pinned_host", _slow_resolve)
        curl = install_fake_curl(monkeypatch, _page())

        with pytest.raises(ImpersonateBudgetExhausted) as excinfo:
            await _fetch(deadline_in_s=0.2)

        assert excinfo.value.waiting_on == "the vetting lookup"
        assert curl.sessions == []

    async def test_a_redirect_re_guard_that_outlives_the_budget_declines_without_dialing_the_hop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The third pre-dial await. ``_hop_refusal`` runs ``is_public_http_url``, another
        uncancellable ``getaddrinfo`` thread, on every redirect target; unbounded, it let the
        caller's deadline bound when the next hop STARTED rather than when the rung stopped."""

        async def _slow_refusal(candidate_url: str) -> None:
            del candidate_url
            await asyncio.sleep(5.0)

        monkeypatch.setattr(resolution_source, "_hop_refusal", _slow_refusal)
        curl = install_fake_curl(monkeypatch, _redirect(f"https://{_OTHER_HOST}/x"), _page())

        with pytest.raises(ImpersonateBudgetExhausted) as excinfo:
            await _fetch(deadline_in_s=0.3)

        assert excinfo.value.waiting_on == "the redirect re-guard"
        assert len(curl.sessions) == 1, "the first hop was dialed; the redirect target never was"


# ---------------------------------------------------------------------------
# The pin assertion
# ---------------------------------------------------------------------------


class TestPinAssertion:
    async def test_a_connection_off_the_pinned_address_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The pin is asserted after the read (the write callback cannot see the headers, so there
        is no pre-read point), and nothing from a refused response is returned. The pre-refusal
        bytes are read but capped and discarded, which is the deliberate trade for non-stream mode.
        """
        off_pin = _page(primary_ip="1.2.3.4", chunks=[b"secret"])
        install_fake_curl(monkeypatch, off_pin)

        with (
            caplog.at_level(logging.ERROR, logger=_TRANSPORT_LOGGER),
            pytest.raises(ImpersonatePinNotHeld) as excinfo,
        ):
            await _fetch()

        assert excinfo.value.expected_ip == _PUBLIC_IP
        assert excinfo.value.actual_ip == "1.2.3.4"
        assert any(record.levelno == logging.ERROR and "pin" in record.getMessage() for record in caplog.records)

    async def test_an_empty_primary_ip_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """curl-cffi fills ``primary_ip`` from ``CURLINFO_PRIMARY_IP`` when it parses the completed
        response (models.py has exactly two assignment sites and nothing re-reads it), so a
        completed transfer that cannot say where it connected is not one to trust: refuse it."""
        install_fake_curl(monkeypatch, _page(primary_ip="", chunks=[b"body"]))

        with pytest.raises(ImpersonatePinNotHeld) as excinfo:
            await _fetch()

        assert excinfo.value.actual_ip == ""

    async def test_a_redirect_off_the_pinned_address_is_refused_before_it_is_followed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        curl = install_fake_curl(monkeypatch, _redirect("https://www.example.com/next"), _page())
        curl.script[0].primary_ip = "1.2.3.4"  # type: ignore[union-attr]  # the scripted entry is a response

        with pytest.raises(ImpersonatePinNotHeld):
            await _fetch()

        assert len(curl.sessions) == 1


# ---------------------------------------------------------------------------
# Failure mapping
# ---------------------------------------------------------------------------


class TestFailureMapping:
    @pytest.mark.parametrize(
        ("exc", "failure_class"),
        [
            (curl_exceptions.Timeout("timed out", CurlECode.OPERATION_TIMEDOUT), "timeout"),
            (curl_exceptions.DNSError("resolve", CurlECode.COULDNT_RESOLVE_HOST), "dns"),
            (curl_exceptions.IncompleteRead("partial", CurlECode.PARTIAL_FILE), "decode"),
            (curl_exceptions.ConnectionError("refused", CurlECode.COULDNT_CONNECT), "connection"),
            (CurlError("bad setopt", CurlECode.UNKNOWN_OPTION), "connection"),
            # A Content-Encoding libcurl cannot decode (code 61) arrives as ``HTTPError``; a status
            # line it cannot parse (code 8) arrives as ``ConnectionError`` in curl_cffi's own map, so
            # the transport reads the code. Both are the direct path's ``malformed_response``: a
            # response refused before the body was ours.
            (curl_exceptions.HTTPError("bad content encoding", CurlECode.BAD_CONTENT_ENCODING), "malformed_response"),
            (curl_exceptions.ConnectionError("weird server reply", CurlECode.WEIRD_SERVER_REPLY), "malformed_response"),
        ],
    )
    async def test_a_code_carrying_failure_maps_to_a_failure_class(
        self, monkeypatch: pytest.MonkeyPatch, exc: Exception, failure_class: str
    ) -> None:
        """The shapes hard to provoke on loopback, hand-built with the code curl-cffi's own
        ``code2error`` would attach in non-stream mode. The bare ``CurlError`` is the rejected-option
        path (F9): neither an ``ImpersonateDeclined`` nor a ``RequestException``, and it must still
        be caught and bucketed rather than escape the transport."""
        curl = install_fake_curl(monkeypatch, exc)

        with pytest.raises(ImpersonateTransportError) as excinfo:
            await _fetch()

        assert excinfo.value.failure_class == failure_class
        assert excinfo.value.exc == type(exc).__name__
        assert excinfo.value.__cause__ is exc
        assert curl.sessions[0].closed

    def test_every_failure_class_the_direct_path_speaks_is_reachable(self) -> None:
        """The vocabulary the transport claims to mirror (``resolution_source._network_failure_class``)
        has six tokens; before the ``HTTPError`` clause ``malformed_response`` was one it could
        never emit."""
        reachable = {
            impersonated_fetch._curl_failure_class(exc)
            for exc in (
                curl_exceptions.Timeout("t", CurlECode.OPERATION_TIMEDOUT),
                curl_exceptions.SSLError("s", CurlECode.SSL_CONNECT_ERROR),
                curl_exceptions.DNSError("d", CurlECode.COULDNT_RESOLVE_HOST),
                curl_exceptions.IncompleteRead("i", CurlECode.PARTIAL_FILE),
                curl_exceptions.HTTPError("h", CurlECode.BAD_CONTENT_ENCODING),
                curl_exceptions.ConnectionError("c", CurlECode.COULDNT_CONNECT),
            )
        }

        assert reachable == {"timeout", "tls", "dns", "decode", "malformed_response", "connection"}

    async def test_a_unicode_decode_error_out_of_curl_cffi_is_a_transport_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """curl_cffi decodes server-controlled bytes with strict UTF-8 when it parses a completed
        response (the reason phrase in ``requests/session.py``, and the cookie jar unless cookies
        are discarded), so a ``UnicodeDecodeError``, a ``ValueError`` that is neither a
        ``RequestException`` nor a ``CurlError``, escaped ``_dial`` and broke the contract that
        every failure out of the transport is an ``ImpersonateDeclined``."""
        install_fake_curl(monkeypatch, UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"))

        with pytest.raises(ImpersonateTransportError) as excinfo:
            await _fetch()

        assert excinfo.value.failure_class == "malformed_response"
        assert excinfo.value.exc == "UnicodeDecodeError"
        assert isinstance(excinfo.value.__cause__, UnicodeDecodeError)

    @pytest.mark.allow_network
    async def test_a_plaintext_listener_behind_https_maps_to_tls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A real failure raised through the whole path, not a hand-built double: an ``https`` GET
        against a plaintext listener arrives as ``SSLError`` (code 35) from ``session.request``, so
        the isinstance ladder buckets it ``tls``. Under stream mode the same failure arrived as a
        bare ``RequestException`` and the ladder was unreachable."""

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            # Answer with plaintext at once, WITHOUT reading first: an https client sends a binary
            # TLS ClientHello, so waiting for an HTTP request head would never complete and libcurl
            # would time out instead of seeing the bad TLS record that makes this an SSLError.
            del reader
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            with contextlib.suppress(*_DRAIN_ERRORS):
                await writer.drain()
            writer.close()

        async with _loopback(monkeypatch, handler, scheme="https") as url:
            with pytest.raises(ImpersonateTransportError) as excinfo:
                await _fetch(url, per_hop_timeout_s=5.0)

        assert excinfo.value.failure_class == "tls"

    @pytest.mark.allow_network
    async def test_a_connection_closed_after_the_request_maps_to_connection(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A listener that reads the request and closes without answering arrives as a
        ``ConnectionError`` (``GOT_NOTHING``), the ``connection`` bucket, through the real path."""

        async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            await _read_request_head(reader)
            writer.close()

        async with _loopback(monkeypatch, handler) as url:
            with pytest.raises(ImpersonateTransportError) as excinfo:
                await _fetch(url, per_hop_timeout_s=5.0)

        assert excinfo.value.failure_class == "connection"

    async def test_a_write_error_that_did_not_trip_the_cap_maps_to_a_failure_class(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``WRITE_ERROR`` the transport did not cause (``reader.tripped`` is False) is a genuine
        transport failure, not an oversized body, so it buckets rather than declining as too-large."""

        class _WriteErrorMidBody(FakeResponse):
            async def feed(self, content_callback: Callable[[bytes], int]) -> None:
                content_callback(b"partial")
                raise curl_exceptions.IncompleteRead("cut", CurlECode.PARTIAL_FILE)

        install_fake_curl(monkeypatch, _WriteErrorMidBody(200, headers={"Content-Type": "text/html"}))

        with pytest.raises(ImpersonateTransportError) as excinfo:
            await _fetch()

        assert excinfo.value.failure_class == "decode"

    async def test_os_error_is_not_caught(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``RequestException`` subclasses ``OSError``; the transport catches curl-cffi's classes
        only, so a genuine OSError from anywhere else still crashes with its own traceback."""
        install_fake_curl(monkeypatch, OSError("not a curl failure"))

        with pytest.raises(OSError, match="not a curl failure") as excinfo:
            await _fetch()

        assert not isinstance(excinfo.value, ImpersonateDeclined)

    async def test_a_non_200_is_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_curl(monkeypatch, FakeResponse(403, headers={"Server": "AkamaiGHost"}, body=b"denied"))

        response = await _fetch()

        assert response.status == 403
        assert response.body == b"denied"
        assert response.server == "AkamaiGHost"


# ---------------------------------------------------------------------------
# The response
# ---------------------------------------------------------------------------


class TestResponse:
    async def test_content_type_is_lower_cased_and_server_is_raw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_curl(
            monkeypatch,
            FakeResponse(200, headers={"Content-Type": "Text/HTML; Charset=UTF-8", "Server": "AkamaiGHost"}),
        )

        response = await _fetch()

        assert response.content_type == "text/html; charset=utf-8"
        assert response.server == "AkamaiGHost"

    async def test_absent_headers_read_as_empty_and_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_curl(monkeypatch, FakeResponse(200, body=b"%PDF-1.4"))

        response = await _fetch()

        assert response.content_type == ""
        assert response.server is None
        assert response.body == b"%PDF-1.4"

    async def test_the_response_is_frozen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_curl(monkeypatch, _page())

        response = await _fetch()

        with pytest.raises(AttributeError):
            response.status = 500  # type: ignore[misc]  # the assignment is the assertion


# ---------------------------------------------------------------------------
# Politeness and hygiene
# ---------------------------------------------------------------------------


class TestPoliteness:
    async def test_two_calls_to_one_host_serialise_on_its_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        curl = install_fake_curl(
            monkeypatch,
            lambda url: _page(body_delay_s=0.02),
            lambda url: _page(body_delay_s=0.02),
        )
        sems: dict[str, asyncio.Semaphore] = {}

        await asyncio.gather(_fetch(_URL, host_sems=sems), _fetch(_URL + "?second", host_sems=sems))

        assert curl.host_peak[_HOST] == 1
        assert len(curl.sessions) == 2

    async def test_two_hosts_run_concurrently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _for_host(url: str) -> FakeResponse:
            ip = _OTHER_IP if _OTHER_HOST in url else _PUBLIC_IP
            return _page(body_delay_s=0.05, primary_ip=ip)

        curl = install_fake_curl(monkeypatch, _for_host, _for_host)
        sems: dict[str, asyncio.Semaphore] = {}

        await asyncio.gather(_fetch(_URL, host_sems=sems), _fetch(f"https://{_OTHER_HOST}/x", host_sems=sems))

        assert curl.peak == 2
        assert curl.host_peak == {_HOST: 1, _OTHER_HOST: 1}

    async def test_a_redirect_chain_contends_on_each_hops_own_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Strict per-hop acquire and release, never nested: an A to B to A chain would self-deadlock
        on a non-reentrant semaphore otherwise."""
        curl = install_fake_curl(
            monkeypatch,
            _redirect(f"https://{_OTHER_HOST}/x"),
            _redirect(f"https://{_HOST}/back", primary_ip=_OTHER_IP),
            _page(),
        )
        sems: dict[str, asyncio.Semaphore] = {}

        response = await asyncio.wait_for(_fetch(_URL, host_sems=sems), timeout=2.0)

        assert response.url == f"https://{_HOST}/back"
        assert len(curl.sessions) == 3
        assert set(sems) == {_HOST, _OTHER_HOST}


class TestSessionHygiene:
    @pytest.mark.parametrize(
        ("script", "expected"),
        [
            ([_page()], None),
            ([_redirect(f"https://{_OTHER_HOST}/x"), _page(primary_ip=_OTHER_IP)], None),
            ([_page(chunks=[b"a" * 2000])], ImpersonateBodyTooLarge),
            ([_page(primary_ip="1.2.3.4")], ImpersonatePinNotHeld),
            ([_redirect("https://www.metaculus.com/q/1/")], ImpersonateHopRefused),
            ([curl_exceptions.Timeout("t", CurlECode.OPERATION_TIMEDOUT)], ImpersonateTransportError),
            (
                [_redirect(f"https://www.example.com/hop{i}") for i in range(MAX_REDIRECTS + 1)],
                ImpersonateRedirectLimit,
            ),
        ],
    )
    async def test_every_session_is_closed_and_no_task_is_left_behind(
        self, monkeypatch: pytest.MonkeyPatch, script: list[Any], expected: type[BaseException] | None
    ) -> None:
        """One session per dial and ``close()`` for every one built, on every decline path, so no
        ``_force_timeout`` polling task leaks."""
        curl = install_fake_curl(monkeypatch, *script)

        if expected is None:
            await _fetch(max_bytes=1000)
        else:
            with pytest.raises(expected):
                await _fetch(max_bytes=1000)

        assert curl.sessions, "the scenario never built a session"
        assert all(session.closed for session in curl.sessions)
        assert len(curl.sessions) == len(script)
        leftovers = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        assert leftovers == []


# ---------------------------------------------------------------------------
# The declines, the memo, and the egress guard
# ---------------------------------------------------------------------------


class TestDeclineFamily:
    @pytest.mark.parametrize(
        "decline",
        [
            ImpersonateUnpinnable,
            ImpersonatePinNotHeld,
            ImpersonateHopRefused,
            ImpersonateRedirectLimit,
            ImpersonateBodyTooLarge,
            ImpersonateTransportError,
            ImpersonateBudgetExhausted,
        ],
    )
    def test_every_decline_is_an_impersonate_declined(self, decline: type[BaseException]) -> None:
        assert issubclass(decline, ImpersonateDeclined)
        assert decline is not ImpersonateDeclined

    def test_a_decline_is_not_an_os_error(self) -> None:
        """A caller's ``except OSError`` must never catch a decline by accident, and a decline must
        never be mistaken for the ``RequestException`` it wraps."""
        assert not issubclass(ImpersonateDeclined, OSError)
        assert not issubclass(ImpersonateDeclined, curl_exceptions.RequestException)


class TestMemo:
    def test_a_refusal_is_remembered_per_host(self) -> None:
        note_impersonation_refused("https://www.bls.gov/wsp/")

        assert impersonation_refused("https://www.bls.gov/news.release/pdf/wkstp.pdf")
        assert impersonation_refused("https://WWW.BLS.GOV/other")
        assert not impersonation_refused("https://www.cdc.gov/x")

    def test_the_port_is_part_of_the_host(self) -> None:
        note_impersonation_refused("https://www.example.com:8443/x")

        assert impersonation_refused("https://www.example.com:8443/y")
        assert not impersonation_refused("https://www.example.com/y")

    def test_reset_clears_it(self) -> None:
        note_impersonation_refused("https://www.bls.gov/wsp/")
        reset_impersonation_memo()

        assert not impersonation_refused("https://www.bls.gov/wsp/")

    async def test_the_transport_never_writes_the_memo_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Only the caller knows whether an answer was block-shaped; a 403 through the transport
        is data, and the memo is the caller's to write through ``note_refusal_if_block_shaped``."""
        install_fake_curl(monkeypatch, FakeResponse(403))

        await _fetch()

        assert not impersonation_refused(_URL)


class TestDeclaredPdf:
    @pytest.mark.parametrize(
        ("content_type", "expected"),
        [
            ("application/pdf", True),
            ("application/pdf; charset=binary", True),
            ("application/x-pdf", True),
            ("text/html", False),
            ("", False),
        ],
    )
    def test_declared_pdf_matches_the_direct_paths_test(self, content_type: str, expected: bool) -> None:
        """The same declared-PDF test both callers' direct paths use to pick the larger cap."""
        assert declared_pdf(content_type) is expected


class TestEgressGuard:
    async def test_the_unfaked_transport_trips_the_native_egress_guard_and_lets_it_through(
        self, native_egress_attempts: list[str]
    ) -> None:
        """With no fake installed the transport builds a REAL ``AsyncSession`` exactly as production
        does, and its ``request`` is what the root conftest guards. The guard's ``RuntimeError`` is
        not a curl-cffi class, so the transport must propagate it: a transport that swallowed it into
        a decline would disarm the money-safety backstop on precisely the path it exists for."""
        with pytest.raises(RuntimeError, match="_block_native_egress"):
            await _fetch()

        assert native_egress_attempts == [f"curl_cffi GET {_URL}"]
        native_egress_attempts.clear()


# ---------------------------------------------------------------------------
# Loopback helpers for the @allow_network regressions
# ---------------------------------------------------------------------------


_DRAIN_ERRORS = (ConnectionResetError, BrokenPipeError)


async def _read_request_head(reader: asyncio.StreamReader) -> bytes:
    """Drain the request line and headers so the handler can answer, tolerating an early close."""
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = await reader.read(4096)
        if not chunk:
            break
        head += chunk
    return head


class _Loopback:
    """A loopback listener the transport dials directly, with the pin helper stubbed to accept it.

    The SSRF vetting rightly refuses ``127.0.0.1``, so these tests, which exercise the body cap and
    the timeout mechanism rather than the guard, patch ``resolve_pinned_host`` to pin the loopback
    address. Everything else is the real transport and the real ``curl_cffi.AsyncSession``.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch, handler: Any, scheme: str) -> None:
        self._monkeypatch = monkeypatch
        self._handler = handler
        self._scheme = scheme
        self._server: asyncio.AbstractServer | None = None

    async def __aenter__(self) -> str:
        self._server = await asyncio.start_server(self._handler, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]

        async def _pin_loopback(url: str) -> tuple[str, str]:
            del url
            return ("127.0.0.1", "127.0.0.1")

        self._monkeypatch.setattr(rendered_fetch, "resolve_pinned_host", _pin_loopback)
        return f"{self._scheme}://127.0.0.1:{port}/x"

    async def __aexit__(self, *_exc: Any) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()


def _loopback(monkeypatch: pytest.MonkeyPatch, handler: Any, *, scheme: str = "http") -> _Loopback:
    return _Loopback(monkeypatch, handler, scheme)
