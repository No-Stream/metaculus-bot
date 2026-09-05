"""The TLS-impersonation transport, shared by both SSRF-guarded fetch paths.

One impersonated GET, presenting a real Chrome TLS ClientHello and HTTP/2 settings fingerprint
through ``curl_cffi`` (libcurl-impersonate). It exists for one measured population: the
Akamai-fronted federal hosts (bls.gov, cdc.gov, fsis.usda.gov) that answer the bot's aiohttp
client with 403 from a GitHub Actions runner and answered the impersonated client with 200
on 2026-09-04, while the Cloudflare, CloudFront and DataDome hosts refused both, so nothing
about the request changes their answer and they stay the archive's and the paid reader's.

What this module owns is the TRANSPORT: the DNS pin, the per-hop re-guard, the manual redirect
loop, the wall bound, the body cap and the per-host politeness gate. It hands back an
:class:`ImpersonatedResponse` and knows nothing about ``FetchResult``, rung attempts or the
ladder, which is what lets the Tier-1 resolution-source rung and gap-fill v2's fetch tool share
it, exactly as they share ``rendered_fetch``. Classification stays with the caller, and so does
the memo write (:func:`note_impersonation_refused`): only the caller knows whether an answer
was block-shaped, because a 403 through this transport is DATA (the fingerprint was not the
problem), not a failure.

The SSRF invariants are carried here, because libcurl never touches aiohttp's connect-time
``FilteringResolver``. Every hop is pre-resolved through the repo's one vetting predicate,
:func:`rendered_fetch.resolve_pinned_host`, which rejects the whole hostname if ANY resolved
address is disallowed. The hop is then pinned to that address with ``CURLOPT_RESOLVE``, so
libcurl cannot resolve it again, and checked after the fact against the address libcurl
reports it connected to. No automatic redirects: every hop is re-guarded through
``resolution_source._hop_refusal``, re-resolved and re-pinned, under the shared ``MAX_REDIRECTS``
cap. A guard here fails SHUT: every refusal raises, and nothing from a refused response is
returned.

Two deliberate divergences from the aiohttp path. No ``BROWSER_HEADERS`` are sent. The
impersonation profile supplies Chrome's complete header set (``Accept``, ``Accept-Language``,
``Priority``, the ``Sec-Fetch-*`` family, the ``sec-ch-ua*`` client hints), and overriding it
with the Safari-like set would present a Chrome TLS fingerprint under Safari headers, which is
precisely the incoherence an edge scores. One consequence is that ``Accept-Encoding`` becomes the
profile's ``gzip, deflate, br, zstd`` instead of the pinned ``gzip, deflate``. That is safe here:
libcurl's bundled decoders decompress in process before the write callback, so the body cap
counts DECOMPRESSED bytes exactly as ``read_body_capped`` does, and the aiohttp path's
``Accept-Encoding`` measurement is undisturbed. The second divergence is one short-lived
``AsyncSession`` per pinned hop. ``curl_options``, the only route to ``CURLOPT_RESOLVE``, is a
constructor parameter and not a request parameter, so a pin cannot be set per request on a
shared session without a race. A session is also bound to the first loop that drives it and
spawns a polling task for its lifetime, so none is ever cached at module scope.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

import certifi
from curl_cffi import CurlOpt
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import DNSError, IncompleteRead, RequestException, SSLError, Timeout

from metaculus_bot.constants import IMPERSONATE_BROWSER_TARGET, RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S
from metaculus_bot.research import rendered_fetch
from metaculus_bot.research.http_fetch import MAX_REDIRECTS, REDIRECT_STATUSES, semaphore_for_host

if TYPE_CHECKING:
    from metaculus_bot.research.resolution_source import HopRefusal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The result and the declines
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ImpersonatedResponse:
    """One completed impersonated HTTP response, body already read and capped.

    A non-200 is returned, not raised: the caller reads a still-403 as the fact that the
    fingerprint was not the problem. ``content_type`` is the header lower-cased, ``""`` when
    absent; ``server`` is the raw header for the caller to tokenise; ``url`` is the last hop
    dialed; ``primary_ip`` is the address libcurl reports it connected to, already checked
    against the pin.
    """

    status: int
    url: str
    content_type: str
    server: str | None
    body: bytes
    elapsed_s: float
    primary_ip: str


class ImpersonateDeclined(Exception):
    """Base of every way the transport fires and produces no body. Never raised directly.

    One family so a caller can decline in one clause, and one subclass per failure shape so
    the log line and the tests can be exact. Deliberately NOT an ``OSError``: curl-cffi's
    ``RequestException`` subclasses ``OSError``, and a decline must never be caught by a
    caller's broad ``except OSError`` nor mistaken for the transport failure it wraps.
    """


class ImpersonateUnpinnable(ImpersonateDeclined):
    """The hop's host is not pinnable or does not resolve to a vetted public address.

    Raised BEFORE anything is dialed. On the first hop that is the caller's own URL; on a later
    hop the target already passed the hop refusal, whose preflight covers the same rejections,
    so this is the pin helper and the refusal disagreeing, and it fails shut either way.
    """

    def __init__(self, url: str) -> None:
        super().__init__(f"host not pinnable to a vetted public address: {urlparse(url).netloc}")
        self.url = url


class ImpersonatePinNotHeld(ImpersonateDeclined):
    """libcurl connected to an address other than the pinned one, or could not say which.

    A nonzero count is a BUILD defect, not host behaviour: an inert pin (the bare-string trap
    on ``CURLOPT_RESOLVE``) or a proxy interposed between us and the host. Logged at ERROR by
    the transport before it is raised.
    """

    def __init__(self, url: str, *, expected_ip: str, actual_ip: str) -> None:
        super().__init__(
            f"impersonated connection did not hold its pin: {urlparse(url).netloc} "
            f"pinned to {expected_ip}, libcurl connected to {actual_ip or '(unreported)'}"
        )
        self.url = url
        self.expected_ip = expected_ip
        self.actual_ip = actual_ip


class ImpersonateHopRefused(ImpersonateDeclined):
    """A redirect target failed ``resolution_source._hop_refusal``.

    ``refusal`` is the ``HopRefusal`` token (``ssrf_blocked`` or ``metaculus_self_ref``);
    ``hop_url`` is the refused target and ``from_url`` the hop that redirected to it, so the
    caller's warning can name both netlocs. A refusal is a DECLINE that leaves the direct
    fetch's own outcome standing, never a terminal result of its own.
    """

    def __init__(self, refusal: HopRefusal, *, hop_url: str, from_url: str) -> None:
        super().__init__(
            f"impersonated redirect refused ({refusal}): {urlparse(from_url).netloc} -> {urlparse(hop_url).netloc}"
        )
        self.refusal: HopRefusal = refusal
        self.hop_url = hop_url
        self.from_url = from_url


class ImpersonateRedirectLimit(ImpersonateDeclined):
    """More than ``MAX_REDIRECTS`` hops."""

    def __init__(self, url: str, *, final_url: str) -> None:
        super().__init__(f"impersonated redirect chain exceeded {MAX_REDIRECTS} hops (final={final_url})")
        self.url = url
        self.final_url = final_url


class ImpersonateBodyTooLarge(ImpersonateDeclined):
    """The body exceeded ``max_bytes`` and the transfer was cut."""

    def __init__(self, url: str, *, bytes_read: int, max_bytes: int) -> None:
        super().__init__(f"impersonated response too large ({bytes_read} bytes read > {max_bytes}): {url}")
        self.url = url
        self.bytes_read = bytes_read
        self.max_bytes = max_bytes


class ImpersonateTransportError(ImpersonateDeclined):
    """A libcurl failure, the wall bound firing, or a redirect that cannot be followed.

    ``failure_class`` speaks the direct path's small vocabulary (``timeout``, ``tls``, ``dns``,
    ``connection``, ``decode``) and is ``None`` where the direct path's own equivalent carries
    none (a redirect with no target, a target whose port cannot be pinned); ``exc`` is the
    exception class name, or the shape's name for those.
    """

    def __init__(self, *, failure_class: str | None, exc: str) -> None:
        super().__init__(f"impersonated fetch failed ({failure_class or exc})")
        self.failure_class = failure_class
        self.exc = exc


# ---------------------------------------------------------------------------
# The memo
# ---------------------------------------------------------------------------

# Hosts whose edge answered our impersonated client with a block status this run, keyed by
# NETLOC (host and port, lower-cased): what was learned is the edge's policy toward our
# fingerprint from this address, a property of the host, so the next cited URL on it is not
# going to be answered differently. Written only by the CALLER, and only for a block-shaped
# answer (an impersonated 403, 406 or 429): a 404 says the path is gone, and a 200 whose body
# classified as chrome or a JavaScript wall means the fingerprint DID get us in, so neither
# switches the host off. Unscoped, unlike the render memos, because the fact is identical for
# both callers, so gap-fill v2 saves a request on a host Tier-1 already probed.
_REFUSED_HOSTS: set[str] = set()


def _memo_key(url: str) -> str:
    return urlparse(url).netloc.lower()


def impersonation_refused(url: str) -> bool:
    """Whether a host on ``url`` already refused our impersonated client this run."""
    return _memo_key(url) in _REFUSED_HOSTS


def note_impersonation_refused(url: str) -> None:
    """Record that ``url``'s host answered the impersonated client with a block status."""
    _REFUSED_HOSTS.add(_memo_key(url))


def reset_impersonation_memo() -> None:
    """Forget every refused host. For tests: the memo outlives one provider call by design."""
    _REFUSED_HOSTS.clear()


# ---------------------------------------------------------------------------
# The fetch
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Hop:
    """What ONE pinned GET yielded, before the loop decides whether it is a redirect."""

    status: int
    url: str
    content_type: str
    server: str | None
    body: bytes
    primary_ip: str
    redirect_url: str
    location: str | None


async def fetch_impersonated(
    url: str,
    *,
    host_sems: dict[str, asyncio.Semaphore],
    deadline_monotonic_s: float,
    per_hop_timeout_s: float,
    max_bytes: int,
) -> ImpersonatedResponse:
    """GET ``url`` with a Chrome fingerprint, following redirects by hand under the SSRF guard.

    ``host_sems`` is the caller's own netloc-to-``Semaphore(1)`` map (Tier-1's process-wide
    ``http_fetch.host_semaphores()``, gap-fill v2's module global), taken as the MAP because
    redirect hops may land on other hosts and each hop must contend on ITS host's gate.
    ``deadline_monotonic_s`` is the instant the whole call must be done by; ``per_hop_timeout_s``
    is the per-request ceiling and ``max_bytes`` the body cap, both the caller's.

    Every failure that produces no body raises an :class:`ImpersonateDeclined`. A non-200 is
    returned as data. The loop runs at most ``MAX_REDIRECTS + 1`` times, exactly like the direct
    fetch, and every hop after the first is a derived URL that owes the two checks every derived
    URL owes (the public preflight, then the Metaculus self-reference refusal, in that order,
    which is a telemetry contract) before it is pinned and dialed.

    curl-cffi 0.15.0 advertises ``allow_redirects="safe"`` (``CurlFollow.SAFE``) as protection
    against internal and private addresses. It is NOT used here and this loop must not be
    "simplified" onto it: its semantics were not verified, and it would follow hops invisibly
    to our own hop refusal, our per-hop pin and our telemetry.
    """
    started = time.monotonic()
    hop_url = url
    for _hop in range(MAX_REDIRECTS + 1):
        hop = await _fetch_pinned_hop(
            hop_url,
            host_sems=host_sems,
            deadline_monotonic_s=deadline_monotonic_s,
            per_hop_timeout_s=per_hop_timeout_s,
            max_bytes=max_bytes,
        )
        if hop.status not in REDIRECT_STATUSES:
            return ImpersonatedResponse(
                status=hop.status,
                url=hop.url,
                content_type=hop.content_type,
                server=hop.server,
                body=hop.body,
                elapsed_s=time.monotonic() - started,
                primary_ip=hop.primary_ip,
            )
        next_url = _redirect_target(hop)
        await _refuse_derived_hop(next_url, from_url=hop_url)
        logger.debug(f"impersonated fetch following {hop.status} {urlparse(hop_url).netloc} -> {next_url}")
        hop_url = next_url
    raise ImpersonateRedirectLimit(url, final_url=hop_url)


async def _fetch_pinned_hop(
    hop_url: str,
    *,
    host_sems: dict[str, asyncio.Semaphore],
    deadline_monotonic_s: float,
    per_hop_timeout_s: float,
    max_bytes: int,
) -> _Hop:
    """Vet, pin, gate and dial ONE hop, and read its body under the cap.

    The order is the invariant. The host is vetted and resolved BEFORE the gate, so an
    unpinnable host never queues; the timeout is sized AFTER the gate is held, so a hop that
    queued behind a slow host does not then help itself to a fresh ceiling; and the gate is
    released before the next hop acquires its own, never nested, because ``asyncio.Semaphore``
    is not reentrant and an A to B to A chain would self-deadlock.
    """
    pinned = await rendered_fetch.resolve_pinned_host(hop_url)
    if pinned is None:
        raise ImpersonateUnpinnable(hop_url)
    host, vetted_ip = pinned
    resolve_entry = _resolve_entry(host, _hop_port(hop_url), vetted_ip)

    async with semaphore_for_host(hop_url, host_sems):
        hop_timeout_s = _hop_timeout_s(deadline_monotonic_s, per_hop_timeout_s)
        async with _pinned_session(resolve_entry, hop_timeout_s) as session:
            try:
                # The outer ``asyncio.timeout`` owns the wall. In stream mode curl-cffi deliberately
                # does NOT set ``TIMEOUT_MS``; it sets the connect timeout plus a 1 byte per second
                # low-speed cutoff over ``timeout`` seconds, so a slow-drip server holds a streamed
                # fetch far past the value (measured: 192 KiB in 4.5 s against ``timeout=1.0``).
                # ``timeout=`` stays on the session as the connect bound, and as the bound on the
                # drain that follows a cut on a STALLED stream: ``quit_now`` takes effect on the next
                # write callback, so a stream sending nothing is released by that cutoff instead.
                async with asyncio.timeout(hop_timeout_s), session.stream("GET", hop_url) as response:
                    return await _read_hop(response, hop_url, vetted_ip=vetted_ip, max_bytes=max_bytes)
            except TimeoutError:
                raise ImpersonateTransportError(failure_class="timeout", exc="TimeoutError") from None
            except RequestException as exc:
                raise ImpersonateTransportError(failure_class=_curl_failure_class(exc), exc=type(exc).__name__) from exc


def _pinned_session(resolve_entry: str, hop_timeout_s: float) -> AsyncSession:
    """One short-lived session whose every safety-relevant option is on the CONSTRUCTOR.

    Nothing per request can forget one. ``async with`` on the result is what runs ``close()``,
    which cancels the polling task ``AsyncCurl`` spawns for the session's lifetime, removes the
    loop's readers and writers and closes the pooled handles.
    """
    # ``CURLOPT_RESOLVE`` takes a LIST. ``Curl.setopt`` special-cases the option and iterates the
    # value, appending each element to a libcurl slist, so a bare ``"host:443:ip"`` iterates
    # CHARACTER BY CHARACTER and libcurl rejects the entry ``'p'`` at perform time, after which it
    # falls back to its own resolver in the general case: a malformed pin fails OPEN. curl-cffi's
    # own hint for the option's values is ``str``, which is why the dict is typed loosely here,
    # and why tests/test_impersonated_fetch.py asserts the exact operand list.
    #
    # The proxy options are mandatory, not belt and braces. libcurl reads ``http_proxy`` from the
    # environment itself, and curl-cffi 0.15.0's documented ``trust_env`` is dead code (nothing
    # reads it). Measured on loopback: with ``http_proxy`` set, the request went through the proxy,
    # which received the absolute-form request and so the HOSTNAME, and the pin never applied at
    # all. ``PROXY=""`` and ``NOPROXY="*"`` each restored the direct pinned connection; both are set
    # so an SSRF invariant does not depend on a runner's environment being clean.
    curl_options: dict[CurlOpt, Any] = {
        CurlOpt.RESOLVE: [resolve_entry],
        CurlOpt.PROXY: "",
        CurlOpt.NOPROXY: "*",
    }
    return AsyncSession(
        # The concrete profile, never the ``"chrome"`` alias (see the constant). curl-cffi types the
        # parameter as a Literal of its profile names; the constant is a plain ``str``.
        impersonate=IMPERSONATE_BROWSER_TARGET,  # pyright: ignore[reportArgumentType]  # profile name held as str
        allow_redirects=False,
        # Mirrors ``http_fetch.build_session``'s certifi pinning: trade.gov failed the handshake
        # against the machine's default store and succeeded against certifi's, so which sources are
        # reachable was a property of the machine. curl-cffi routes a ``str`` to ``CURLOPT_CAINFO``
        # and otherwise reads ``REQUESTS_CA_BUNDLE`` / ``CURL_CA_BUNDLE`` from the environment; its
        # hint says ``bool``, its runtime accepts the path.
        verify=certifi.where(),  # pyright: ignore[reportArgumentType]  # CA bundle path; the hint is narrower than the runtime
        timeout=hop_timeout_s,
        curl_options=curl_options,
    )


async def _read_hop(response: Any, hop_url: str, *, vetted_ip: str, max_bytes: int) -> _Hop:
    """Assert the pin, read the body under the cap, and abort the transfer on the way out.

    ``quit_now.set()`` runs in the ``finally`` so it always precedes the ``aclose()`` that
    ``AsyncSession.stream`` awaits on exit: ``aclose`` alone awaits the download task, which
    DRAINS the rest of the body, while the flag is what makes curl-cffi's write callback return
    ``CURL_WRITEFUNC_ERROR`` and abort the transfer. On a fully read body the flag is inert.
    """
    try:
        _check_pin(response.primary_ip, vetted_ip, hop_url, refuse_when_empty=False)
        body = await _read_body_capped(response, hop_url, max_bytes)
        # Whether ``primary_ip`` is populated the moment the streamed response object appears was
        # not verified; an empty value at that point is re-read after the body, and refused if it
        # is still empty. Refusing after a read still closes the SSRF channel, because nothing from
        # a refused response is returned.
        _check_pin(response.primary_ip, vetted_ip, hop_url, refuse_when_empty=True)
    finally:
        response.quit_now.set()
    headers = response.headers
    return _Hop(
        status=response.status_code,
        url=hop_url,
        content_type=(headers.get("content-type") or "").strip().lower(),
        server=headers.get("server"),
        body=body,
        primary_ip=response.primary_ip,
        redirect_url=response.redirect_url or "",
        location=headers.get("location"),
    )


async def _read_body_capped(response: Any, hop_url: str, max_bytes: int) -> bytes:
    """Read DECOMPRESSED bytes chunk by chunk and stop the moment the running total passes the cap.

    libcurl decompresses BEFORE the write callback, so ``aiter_content`` yields decompressed bytes
    exactly as aiohttp's ``iter_chunked`` does after its ``DeflateBuffer``; counting them against
    the same cap reproduces the direct fetch's gzip-bomb protection. The boundary is ``>``, matching
    ``read_body_capped``. The generator is closed explicitly on every exit so a cut-off iteration
    does not leave asyncio's finaliser to schedule its ``aclose`` as a stray task.
    """
    chunks: list[bytes] = []
    total = 0
    async with contextlib.aclosing(response.aiter_content()) as body:
        async for chunk in body:
            total += len(chunk)
            if total > max_bytes:
                raise ImpersonateBodyTooLarge(hop_url, bytes_read=total, max_bytes=max_bytes)
            chunks.append(chunk)
    return b"".join(chunks)


def _check_pin(primary_ip: str, vetted_ip: str, hop_url: str, *, refuse_when_empty: bool) -> None:
    """Refuse unless libcurl reports the pinned address (compared as addresses, not strings).

    A host with several A records is a benign source of a mismatch when the pin is INERT:
    ``resolve_vetted_public_ip`` returns the first vetted address while libcurl's own resolver
    may pick another. Every address was vetted, so refusing is safe if slightly lossy, and a
    nonzero count in the live QA means the pin is not working, which is the point of the check.
    """
    if not primary_ip and not refuse_when_empty:
        return
    if primary_ip and _same_address(primary_ip, vetted_ip):
        return
    logger.error(
        f"impersonated fetch pin NOT held for {urlparse(hop_url).netloc}: pinned={vetted_ip} "
        f"connected={primary_ip or '(unreported)'}; refusing the response unread"
    )
    raise ImpersonatePinNotHeld(hop_url, expected_ip=vetted_ip, actual_ip=primary_ip)


def _same_address(reported: str, vetted: str) -> bool:
    try:
        return ipaddress.ip_address(reported) == ipaddress.ip_address(vetted)
    except ValueError:
        return False


def _resolve_entry(host: str, port: int, vetted_ip: str) -> str:
    """The ``CURLOPT_RESOLVE`` entry ``host:port:address``, with an IPv6 address bracketed.

    libcurl matches host AND port exactly, so the entry pins the port this hop will dial. One
    entry, not two: a redirect that changes scheme on the same host is a new hop with a new
    session and a new pin. Both bracketed and bare IPv6 forms work; bracketed is the unambiguous one.
    """
    address = ipaddress.ip_address(vetted_ip)
    target = f"[{address}]" if address.version == 6 else str(address)
    return f"{host}:{port}:{target}"


def _hop_port(hop_url: str) -> int:
    """The port this hop dials: the URL's own, else the scheme's default.

    A port that is not a number (possible only on a redirect target, since the caller's own URL
    was already fetched once) cannot be pinned, so it is refused rather than dialed under a pin
    that matches nothing.
    """
    parsed = urlparse(hop_url)
    try:
        port = parsed.port
    except ValueError:
        raise ImpersonateTransportError(failure_class=None, exc="InvalidURL") from None
    if port is not None:
        return port
    return 443 if parsed.scheme.lower() == "https" else 80


def _hop_timeout_s(deadline_monotonic_s: float, per_hop_timeout_s: float) -> float:
    """Size this hop's timeout from the remaining budget, the direct path's arithmetic exactly."""
    remaining = deadline_monotonic_s - time.monotonic()
    if remaining <= 0.0:
        raise ImpersonateTransportError(failure_class="timeout", exc="TimeoutError")
    return min(per_hop_timeout_s, max(remaining, RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S))


def _redirect_target(hop: _Hop) -> str:
    """The next hop: libcurl's absolutised ``redirect_url``, else the raw ``Location`` joined by hand.

    ``redirect_url`` is ``CURLINFO_REDIRECT_URL``, resolved against the request URL and populated
    even with ``allow_redirects=False`` (verified: a relative ``/deep/other?a=1`` and a
    scheme-relative ``//host/ok`` both came back absolute while the header stayed raw). A redirect
    with neither mirrors the direct path's malformed-redirect branch: no failure class, the shape
    named in ``exc``.
    """
    if hop.redirect_url:
        return hop.redirect_url
    if hop.location:
        return urljoin(hop.url, hop.location)
    raise ImpersonateTransportError(failure_class=None, exc="MissingLocation")


async def _refuse_derived_hop(next_url: str, *, from_url: str) -> None:
    """Re-guard a URL that came out of a ``Location`` header before it is pinned or dialed.

    The ``resolution_source`` import is function-scoped for the two reasons
    :func:`rendered_fetch.resolve_pinned_host` states, and both must hold: it is a REAL circular
    import (that module imports this one at module scope for its rung), and it is the LATE
    BINDING the suites rely on, because the guard's predicates are monkeypatched on THAT module
    by both fetch paths' tests and it has exactly one patch surface because every reader resolves
    it there.
    """
    from metaculus_bot.research import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # real cycle + the guard's single patch surface, per the docstring
        resolution_source,
    )

    refusal = await resolution_source._hop_refusal(next_url)
    if refusal is not None:
        raise ImpersonateHopRefused(refusal, hop_url=next_url, from_url=from_url)


def _curl_failure_class(exc: RequestException) -> str:
    """Bucket a curl-cffi failure into the direct path's ``failure_class`` vocabulary.

    Mirrors ``resolution_source._network_failure_class`` so the two fetchers speak the same
    small set. The specific classes come first because ``DNSError`` and ``SSLError`` both
    subclass curl-cffi's ``ConnectionError``, and ``CertificateVerifyError`` subclasses
    ``SSLError``. Everything else, including the bare ``RequestException`` a write error
    raises, is ``connection``; no new token is invented, because the vocabulary is a marker
    contract.
    """
    if isinstance(exc, Timeout):
        return "timeout"
    if isinstance(exc, SSLError):
        return "tls"
    if isinstance(exc, DNSError):
        return "dns"
    if isinstance(exc, IncompleteRead):
        return "decode"
    return "connection"
