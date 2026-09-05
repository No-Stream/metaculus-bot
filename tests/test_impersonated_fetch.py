"""The TLS-impersonation transport (`metaculus_bot/research/impersonated_fetch.py`): its SSRF
procedure, its bounds, and the exact libcurl options it builds a session with.

Sits beside ``tests/test_http_fetch.py`` (the aiohttp transport's ``FilteringResolver``,
``build_session`` and ``read_body_capped``) and ``tests/test_rendered_fetch.py`` (the browser
transport). Nothing here opens a libcurl handle. The seam is the ``AsyncSession`` NAME bound in
``impersonated_fetch``: every test that needs a response replaces it with :func:`install_fake_curl`,
a fake CLASS that records the constructor kwargs (the pin operand, the proxy options, the
impersonation target, the timeout) and serves a scripted stream. The root conftest's
``_block_native_egress`` still patches the REAL ``AsyncSession.request`` underneath, and
:class:`TestEgressGuard` calls the unfaked transport to prove that guard is what fires, and that
the transport lets it through rather than swallowing it into a decline.

The DNS the SSRF vetting resolves through is stubbed per test on ``resolution_source.socket``,
the one module every reader of the guard resolves it on, so a hostname's addresses are whatever
the test says and nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import socket
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import certifi
import pytest
from curl_cffi import CurlOpt
from curl_cffi.const import CurlECode
from curl_cffi.requests import Headers
from curl_cffi.requests import exceptions as curl_exceptions

from metaculus_bot.constants import IMPERSONATE_BROWSER_TARGET, RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S
from metaculus_bot.research import impersonated_fetch, rendered_fetch, resolution_source
from metaculus_bot.research.http_fetch import MAX_REDIRECTS
from metaculus_bot.research.impersonated_fetch import (
    ImpersonateBodyTooLarge,
    ImpersonateDeclined,
    ImpersonatedResponse,
    ImpersonateHopRefused,
    ImpersonatePinNotHeld,
    ImpersonateRedirectLimit,
    ImpersonateTransportError,
    ImpersonateUnpinnable,
    fetch_impersonated,
    impersonation_refused,
    note_impersonation_refused,
    reset_impersonation_memo,
)

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


class _RecordingEvent:
    """Stands in for the stream response's ``quit_now`` and records WHEN it was set relative to
    ``aclose``, because that order is the whole body-cap invariant."""

    def __init__(self, log: list[str]) -> None:
        self._log = log
        self._set = False

    def set(self) -> None:
        self._set = True
        self._log.append("quit_now.set")

    def is_set(self) -> bool:
        return self._set


class FakeStreamResponse:
    """A stub for the streamed ``curl_cffi.requests.Response``.

    Cannot be the real class: stream mode needs ``queue``, ``astream_task`` and ``quit_now`` wired
    to a live curl handle. Carries what the transport reads: ``status_code``, a genuine
    case-insensitive ``Headers``, ``url``, ``redirect_url``, ``primary_ip``, an async-generator
    ``aiter_content``, a ``quit_now`` with ``set()``, and ``aclose``. ``events`` is the ordered log
    of ``quit_now.set`` and ``aclose``; ``chunks_read`` counts what the transport consumed.
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
        url: str = "",
        body_delay_s: float = 0.0,
        hang: bool = False,
        primary_ip_after_read: str | None = None,
    ) -> None:
        self.status_code = status
        self.headers = Headers(headers or {})
        self.url = url
        self.redirect_url = redirect_url
        self.primary_ip = primary_ip
        self.events: list[str] = []
        self.quit_now = _RecordingEvent(self.events)
        self.chunks_read = 0
        self._chunks = list(chunks) if chunks is not None else ([body] if body else [])
        self._body_delay_s = body_delay_s
        self._hang = hang
        self._primary_ip_after_read = primary_ip_after_read

    async def aiter_content(self):
        if self._hang:
            await asyncio.Event().wait()
        for chunk in self._chunks:
            await asyncio.sleep(self._body_delay_s)
            self.chunks_read += 1
            yield chunk
        if self._primary_ip_after_read is not None:
            self.primary_ip = self._primary_ip_after_read

    async def aclose(self) -> None:
        self.events.append("aclose")


# A scripted hop: a ready response, an exception to raise where ``request`` would, or a callable
# of the URL producing either (for tests whose response depends on which host was dialed).
_Scripted = FakeStreamResponse | BaseException | Callable[[str], "FakeStreamResponse | BaseException"]


class FakeCurl:
    """What :func:`install_fake_curl` returns: every session built, and the in-flight peaks."""

    def __init__(self, script: list[_Scripted]) -> None:
        self.script = script
        self.sessions: list[FakeCurlSession] = []
        self.host_inflight: dict[str, int] = {}
        self.host_peak: dict[str, int] = {}
        self.inflight = 0
        self.peak = 0

    def next_response(self, url: str) -> FakeStreamResponse | BaseException:
        if not self.script:
            raise AssertionError(f"no scripted response for {url}")
        entry = self.script.pop(0)
        return entry(url) if callable(entry) else entry

    @property
    def resolve_operands(self) -> list[list[str] | str]:
        return [session.kwargs["curl_options"][CurlOpt.RESOLVE] for session in self.sessions]


class FakeCurlSession:
    """Stands in for the ``AsyncSession`` CLASS. One instance per constructor call, exactly as the
    transport builds one per pinned hop; ``kwargs`` is what the pin assertions read."""

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

    @asynccontextmanager
    async def stream(self, method: str, url: str):
        self.requests.append((method, url))
        host = urlparse(url).netloc
        recorder = self.recorder
        recorder.host_inflight[host] = recorder.host_inflight.get(host, 0) + 1
        recorder.host_peak[host] = max(recorder.host_peak.get(host, 0), recorder.host_inflight[host])
        recorder.inflight += 1
        recorder.peak = max(recorder.peak, recorder.inflight)
        try:
            # The real request yields to the loop before any header arrives.
            await asyncio.sleep(0)
            response = recorder.next_response(url)
            if isinstance(response, BaseException):
                raise response
            try:
                yield response
            finally:
                # Mirrors ``AsyncSession.stream``, whose ``finally`` awaits ``rsp.aclose()``.
                await response.aclose()
        finally:
            recorder.host_inflight[host] -= 1
            recorder.inflight -= 1


def install_fake_curl(monkeypatch: pytest.MonkeyPatch, *script: _Scripted) -> FakeCurl:
    """Bind ``impersonated_fetch.AsyncSession`` to a fake class serving ``script`` hop by hop.

    Patches the transport's own module attribute, never ``curl_cffi.requests.AsyncSession`` and
    never its ``request`` method, so the root conftest's native-egress guard stays armed on the
    real class underneath.
    """
    recorder = FakeCurl(list(script))
    monkeypatch.setattr(impersonated_fetch, "AsyncSession", functools.partial(FakeCurlSession, recorder))
    return recorder


def _redirect(
    location: str, *, redirect_url: str | None = None, status: int = 302, **kwargs: Any
) -> FakeStreamResponse:
    """A 3xx hop. ``redirect_url`` is libcurl's absolutised target; by default it equals the header."""
    return FakeStreamResponse(
        status,
        headers={"Location": location},
        redirect_url=location if redirect_url is None else redirect_url,
        **kwargs,
    )


def _page(body: bytes = b"<html>ok</html>", **kwargs: Any) -> FakeStreamResponse:
    headers = {"Content-Type": "text/html; charset=utf-8", "Server": "AkamaiGHost"}
    headers.update(kwargs.pop("headers", {}))
    return FakeStreamResponse(200, headers=headers, body=body, **kwargs)


async def _fetch(
    url: str = _URL,
    *,
    host_sems: dict[str, asyncio.Semaphore] | None = None,
    deadline_in_s: float = 30.0,
    per_hop_timeout_s: float = 20.0,
    max_bytes: int = 1024 * 1024,
) -> ImpersonatedResponse:
    return await fetch_impersonated(
        url,
        host_sems={} if host_sems is None else host_sems,
        deadline_monotonic_s=time.monotonic() + deadline_in_s,
        per_hop_timeout_s=per_hop_timeout_s,
        max_bytes=max_bytes,
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
        install_fake_curl(monkeypatch, FakeStreamResponse(302))

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
        declines that the refusal let through reports ``ImpersonateUnpinnable`` too."""

        async def _unpinnable_second_hop(url: str) -> tuple[str, str] | None:
            if "second" in url:
                return None
            return (_HOST, _PUBLIC_IP)

        monkeypatch.setattr(rendered_fetch, "resolve_pinned_host", _unpinnable_second_hop)
        curl = install_fake_curl(monkeypatch, _redirect("https://www.example.com/second"))

        with pytest.raises(ImpersonateUnpinnable):
            await _fetch()

        assert len(curl.sessions) == 1


# ---------------------------------------------------------------------------
# The body cap
# ---------------------------------------------------------------------------


class TestBodyCap:
    async def test_the_body_cap_cuts_the_transfer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``quit_now.set()`` must precede ``aclose()``: ``aclose`` alone awaits the download task,
        which drains the rest of the body; the flag is what makes curl-cffi's write callback return
        ``CURL_WRITEFUNC_ERROR`` and abort the transfer."""
        oversized = _page(chunks=[b"a" * 600, b"b" * 600])
        install_fake_curl(monkeypatch, oversized)

        with pytest.raises(ImpersonateBodyTooLarge):
            await _fetch(max_bytes=1000)

        assert oversized.events == ["quit_now.set", "aclose"]

    async def test_a_body_exactly_at_the_cap_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The boundary is ``>`` and not ``>=``, matching ``read_body_capped``."""
        install_fake_curl(monkeypatch, _page(chunks=[b"a" * 500, b"b" * 500]))

        response = await _fetch(max_bytes=1000)

        assert len(response.body) == 1000

    async def test_the_body_is_the_joined_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_curl(monkeypatch, _page(chunks=[b"<html>", b"ok", b"</html>"]))

        response = await _fetch()

        assert response.body == b"<html>ok</html>"


# ---------------------------------------------------------------------------
# The wall bound
# ---------------------------------------------------------------------------


class TestTimeouts:
    async def test_a_past_deadline_makes_no_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        curl = install_fake_curl(monkeypatch, _page())

        with pytest.raises(ImpersonateTransportError) as excinfo:
            await _fetch(deadline_in_s=-1.0)

        assert excinfo.value.failure_class == "timeout"
        assert excinfo.value.exc == "TimeoutError"
        assert curl.sessions == []

    async def test_a_stream_that_never_yields_is_cut_by_the_asyncio_wall(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """libcurl's own ``timeout=`` does not bound a streamed body: in stream mode curl-cffi sets
        ``CONNECTTIMEOUT_MS`` plus a 1 byte per second ``LOW_SPEED_LIMIT``, never ``TIMEOUT_MS``.
        Measured on loopback: a streamed read with ``timeout=1.0`` pulled 192 KiB over 4.5 s with
        no exception. The ``asyncio.timeout`` around the stream is what owns the wall."""
        hanging = _page(hang=True)
        curl = install_fake_curl(monkeypatch, hanging)

        with pytest.raises(ImpersonateTransportError) as excinfo:
            await _fetch(per_hop_timeout_s=0.05)

        assert excinfo.value.failure_class == "timeout"
        assert excinfo.value.exc == "TimeoutError"
        assert hanging.events == ["quit_now.set", "aclose"]
        assert curl.sessions[0].closed

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

    async def test_elapsed_covers_the_whole_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_curl(monkeypatch, _redirect("https://www.example.com/next"), _page(body_delay_s=0.02))

        response = await _fetch()

        assert response.elapsed_s >= 0.02


# ---------------------------------------------------------------------------
# The pin assertion
# ---------------------------------------------------------------------------


class TestPinAssertion:
    async def test_a_connection_off_the_pinned_address_is_refused_unread(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        off_pin = _page(primary_ip="1.2.3.4", chunks=[b"secret"])
        install_fake_curl(monkeypatch, off_pin)

        with (
            caplog.at_level(logging.ERROR, logger=_TRANSPORT_LOGGER),
            pytest.raises(ImpersonatePinNotHeld) as excinfo,
        ):
            await _fetch()

        assert excinfo.value.expected_ip == _PUBLIC_IP
        assert excinfo.value.actual_ip == "1.2.3.4"
        assert off_pin.chunks_read == 0
        assert any(record.levelno == logging.ERROR and "pin" in record.getMessage() for record in caplog.records)

    async def test_an_empty_primary_ip_is_rechecked_after_the_read_and_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whether ``primary_ip`` is populated the moment the streamed response object appears was
        not verified, so an empty value defers the check to after the body read; still empty then
        is a refusal, and nothing from a refused response is returned."""
        unknown = _page(primary_ip="", chunks=[b"body"])
        install_fake_curl(monkeypatch, unknown)

        with pytest.raises(ImpersonatePinNotHeld) as excinfo:
            await _fetch()

        assert unknown.chunks_read == 1
        assert excinfo.value.actual_ip == ""

    async def test_an_empty_primary_ip_populated_by_the_read_holds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_curl(monkeypatch, _page(primary_ip="", chunks=[b"body"], primary_ip_after_read=_PUBLIC_IP))

        response = await _fetch()

        assert response.body == b"body"
        assert response.primary_ip == _PUBLIC_IP

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
            (curl_exceptions.SSLError("handshake", CurlECode.SSL_CONNECT_ERROR), "tls"),
            (curl_exceptions.CertificateVerifyError("verify", CurlECode.PEER_FAILED_VERIFICATION), "tls"),
            (curl_exceptions.DNSError("resolve", CurlECode.COULDNT_RESOLVE_HOST), "dns"),
            (curl_exceptions.ConnectionError("refused", CurlECode.COULDNT_CONNECT), "connection"),
            (curl_exceptions.IncompleteRead("partial", CurlECode.PARTIAL_FILE), "decode"),
            (curl_exceptions.RequestException("write", CurlECode.WRITE_ERROR), "connection"),
        ],
    )
    async def test_every_libcurl_failure_maps_to_a_failure_class(
        self, monkeypatch: pytest.MonkeyPatch, exc: Exception, failure_class: str
    ) -> None:
        curl = install_fake_curl(monkeypatch, exc)

        with pytest.raises(ImpersonateTransportError) as excinfo:
            await _fetch()

        assert excinfo.value.failure_class == failure_class
        assert excinfo.value.exc == type(exc).__name__
        assert excinfo.value.__cause__ is exc
        assert curl.sessions[0].closed

    async def test_a_failure_mid_body_maps_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``aiter_content`` re-raises a ``RequestException`` the transfer queued mid-stream."""

        class _FailsMidBody(FakeStreamResponse):
            async def aiter_content(self):
                yield b"partial"
                raise curl_exceptions.IncompleteRead("cut", CurlECode.PARTIAL_FILE)

        install_fake_curl(monkeypatch, _FailsMidBody(200, headers={"Content-Type": "text/html"}))

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
        install_fake_curl(monkeypatch, FakeStreamResponse(403, headers={"Server": "AkamaiGHost"}, body=b"denied"))

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
            FakeStreamResponse(200, headers={"Content-Type": "Text/HTML; Charset=UTF-8", "Server": "AkamaiGHost"}),
        )

        response = await _fetch()

        assert response.content_type == "text/html; charset=utf-8"
        assert response.server == "AkamaiGHost"

    async def test_absent_headers_read_as_empty_and_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        install_fake_curl(monkeypatch, FakeStreamResponse(200, body=b"%PDF-1.4"))

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
            lambda url: _page(body_delay_s=0.02, url=url),
            lambda url: _page(body_delay_s=0.02, url=url),
        )
        sems: dict[str, asyncio.Semaphore] = {}

        await asyncio.gather(_fetch(_URL, host_sems=sems), _fetch(_URL + "?second", host_sems=sems))

        assert curl.host_peak[_HOST] == 1
        assert len(curl.sessions) == 2

    async def test_two_hosts_run_concurrently(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _for_host(url: str) -> FakeStreamResponse:
            ip = _OTHER_IP if _OTHER_HOST in url else _PUBLIC_IP
            return _page(body_delay_s=0.05, primary_ip=ip, url=url)

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
        """One session per hop and ``close()`` for every one built, on every decline path, so no
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
        is data, and the memo is the caller's to write."""
        install_fake_curl(monkeypatch, FakeStreamResponse(403))

        await _fetch()

        assert not impersonation_refused(_URL)


class TestEgressGuard:
    async def test_the_unfaked_transport_trips_the_native_egress_guard_and_lets_it_through(
        self, native_egress_attempts: list[str]
    ) -> None:
        """With no fake installed the transport builds a REAL ``AsyncSession`` exactly as production
        does, and its ``stream`` lands on the ``request`` the root conftest guards. The guard's
        ``RuntimeError`` is not a curl-cffi class, so the transport must propagate it: a transport
        that swallowed it into a decline would disarm the money-safety backstop on precisely the
        path it exists for."""
        with pytest.raises(RuntimeError, match="_block_native_egress"):
            await _fetch()

        assert native_egress_attempts == [f"curl_cffi GET {_URL}"]
        native_egress_attempts.clear()
