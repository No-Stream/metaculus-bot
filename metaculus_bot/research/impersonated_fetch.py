"""The TLS-impersonation transport, shared by both SSRF-guarded fetch paths.

One impersonated GET, presenting a real Chrome TLS ClientHello and HTTP/2 settings fingerprint
through ``curl_cffi`` (libcurl-impersonate). It exists for one measured population: the
Akamai-fronted federal hosts (bls.gov, cdc.gov, fsis.usda.gov) that answer the bot's aiohttp
client with 403 from a GitHub Actions runner and answered the impersonated client with 200
on 2026-09-04, while the Cloudflare, CloudFront and DataDome hosts refused both, so nothing
about the request changes their answer and they stay the archive's and the paid reader's.

What this module owns is the TRANSPORT and the rung POLICY the two callers share. The transport
is the DNS pin, the per-hop re-guard, the manual redirect loop, the wall bound, the body cap and
the per-host politeness gate; it hands back an :class:`ImpersonatedResponse` and knows nothing
about ``FetchResult``, rung attempts or the ladder, which is what lets the Tier-1
resolution-source rung and gap-fill v2's fetch tool share it, exactly as they share
``rendered_fetch``. The policy is the trigger set (:data:`IMPERSONATE_TRIGGER_STATUSES`), the
block-shaped statuses that switch a host off for the run (:data:`IMPERSONATE_BLOCK_STATUSES`),
the kill switch (:func:`impersonation_enabled`) and the memo write
(:func:`note_refusal_if_block_shaped`), kept beside the memo they read and write so the two
ladders cannot drift apart on what counts as a refusal. Classification stays with the caller: a
403 through this transport is DATA (the fingerprint was not the problem), not a failure, and
only the caller decides what it means for its own ladder.

The SSRF invariants are carried here, because libcurl never touches aiohttp's connect-time
``FilteringResolver``. Every hop is pre-resolved through the repo's one vetting predicate,
:func:`rendered_fetch.resolve_pinned_host`, which rejects the whole hostname if ANY resolved
address is disallowed. The hop is then pinned to that address with ``CURLOPT_RESOLVE``, so
libcurl cannot resolve it again, and checked after the fact against the address libcurl
reports it connected to. No automatic redirects: every hop is re-guarded through
``resolution_source._hop_refusal``, re-resolved and re-pinned, under the shared ``MAX_REDIRECTS``
cap. A guard here fails SHUT: every refusal raises, and nothing from a refused response is
returned.

The request runs in curl-cffi's NON-STREAM mode with a write callback, and the choice is
load-bearing. Stream mode gives up both of libcurl's own bounds: it queues chunks on an
unbounded ``asyncio.Queue`` (measured 2026-09-04: a gzip body inflating to 400 MiB had 370 MiB
resident when the consumer-side cap fired at 5 MiB), it replaces ``CURLOPT_TIMEOUT_MS`` with a
low-speed cutoff (a 2 s hop timeout raised at 9 s against a burst-then-stall server, 20 s at
27 s), and its exit drains the rest of the body. With a write callback libcurl sets
``TIMEOUT_MS`` from ``timeout=`` and calls the callback with each DECOMPRESSED chunk as it
arrives, so the callback IS the cap: it keeps chunks up to the cap and aborts the transfer past
it, and libcurl's own timer is the wall. Measured on loopback with the installed wheel: a 200 MiB
bomb under a 2 MiB cap retained exactly 2 MiB and aborted in 0.01 s, and a stalled body raised at
exactly ``timeout=``. No ``asyncio`` wall wraps the request: libcurl's timer is the measured exit
and its normal completion path, while cancelling the request future mid-transfer would leave
the teardown to curl-cffi's handle recycling, a path nothing here has measured.

Two deliberate divergences from the aiohttp path. No ``BROWSER_HEADERS`` are sent. The
impersonation profile supplies Chrome's complete header set (``Accept``, ``Accept-Language``,
``Priority``, the ``Sec-Fetch-*`` family, the ``sec-ch-ua*`` client hints), and overriding it
with the Safari-like set would present a Chrome TLS fingerprint under Safari headers, which is
precisely the incoherence an edge scores. One consequence is that ``Accept-Encoding`` becomes the
profile's ``gzip, deflate, br, zstd`` instead of the pinned ``gzip, deflate``. That is safe here:
libcurl's bundled decoders decompress in process before the write callback, so the body cap
counts DECOMPRESSED bytes exactly as ``read_body_capped`` does (verified: the bomb's callback
total exceeded its compressed size before the cap fired), and the aiohttp path's
``Accept-Encoding`` measurement is undisturbed. The second divergence is one short-lived
``AsyncSession`` per dial. ``curl_options``, the only route to ``CURLOPT_RESOLVE``, is a
constructor parameter and not a request parameter, so a pin cannot be set per request on a
shared session without a race. A session is also bound to the first loop that drives it and
spawns a polling task for its lifetime, so none is ever cached at module scope.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlparse

import certifi
from curl_cffi import CurlError, CurlOpt
from curl_cffi.curl import CURL_WRITEFUNC_ERROR
from curl_cffi.requests import AsyncSession
from curl_cffi.requests.exceptions import DNSError, IncompleteRead, RequestException, SSLError, Timeout

from metaculus_bot.constants import (
    IMPERSONATE_BROWSER_TARGET,
    RESOLUTION_SOURCE_IMPERSONATE_ENABLED_ENV,
    RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S,
    env_flag_enabled,
)
from metaculus_bot.research import rendered_fetch
from metaculus_bot.research.http_fetch import MAX_REDIRECTS, REDIRECT_STATUSES, semaphore_for_host
from metaculus_bot.research.resolution_fetch_result import _NON_OK_FETCH_STATUS

if TYPE_CHECKING:
    from metaculus_bot.research.resolution_source import HopRefusal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The policy both callers share
# ---------------------------------------------------------------------------

# The direct-fetch HTTP statuses the impersonated retry fires on. 403 only, because the
# 2026-09-04 diagnostic measured impersonation helping only on 403s: 429 is a throttle, and
# retrying at once under a different fingerprint against a host that just asked us to slow down
# is the one shape where the retry could make our position worse; 406 is a content-negotiation
# refusal, and the profile changes the ``Accept`` headers as a side effect, so a 406 rung would be
# an untested guess; 401 is an authentication requirement no fingerprint changes. Read by both
# callers as a module ATTRIBUTE at call time, so a test package can empty it to decline the rung
# suite-wide and restore this one object to exercise it.
IMPERSONATE_TRIGGER_STATUSES: frozenset[int] = frozenset({403})

# The impersonated answers that switch the host off for the rest of the run: every status the
# fetch vocabulary calls ``blocked``, derived from the one table both fetch paths already read
# rather than spelled a third time. A 404 says the path is gone and a 200 whose body classified
# as a JavaScript wall means the fingerprint DID get us in, so neither is here.
IMPERSONATE_BLOCK_STATUSES: frozenset[int] = frozenset(
    status for status, fetch_status in _NON_OK_FETCH_STATUS.items() if fetch_status == "blocked"
)

# The declared types the larger body cap applies to: the same declared-PDF test both callers'
# direct paths use to read a document under ``DOCUMENT_TEXT_PDF_MAX_BYTES`` instead of the page
# cap (``resolution_source._PDF_CONTENT_TYPES``; gap-fill v2 checks ``application/pdf`` alone).
# An undeclared body keeps the page cap there too, so it does here.
PDF_CONTENT_TYPES: tuple[str, ...] = ("application/pdf", "application/x-pdf")


def impersonation_enabled() -> bool:
    """The kill switch, ON unless ``RESOLUTION_SOURCE_IMPERSONATE_ENABLED`` says otherwise.

    The only research flag whose code default is on: the rung is free, fires on a 403 only, and
    is bounded by the per-run host memo and its budget floor. ``env_flag_enabled`` itself
    defaults to off, so the ``default=True`` here is the load-bearing half, kept in one place.
    """
    return env_flag_enabled(RESOLUTION_SOURCE_IMPERSONATE_ENABLED_ENV, default=True)


def declared_pdf(content_type: str) -> bool:
    """Whether a lower-cased Content-Type declares one of :data:`PDF_CONTENT_TYPES`."""
    return any(token in content_type for token in PDF_CONTENT_TYPES)


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

    Raised BEFORE the hop is dialed. On the first hop that is the caller's own URL, and the
    direct fetch resolved the same host through the filtering resolver moments earlier, so a
    refusal there is DNS disagreeing with the direct fetch. On a later hop it is a redirect
    target the direct fetch never resolved: the target already passed the hop refusal, whose
    preflight covers the same rejections, so this is the pin helper and the refusal disagreeing,
    and it fails shut either way. ``redirected_from`` is the hop that sent us there, so the
    message names which host failed to resolve and how it was reached.
    """

    def __init__(self, url: str, *, redirected_from: str | None = None) -> None:
        detail = f"host not pinnable to a vetted public address: {urlparse(url).netloc} ({url})"
        if redirected_from is not None:
            detail += f", a redirect target of {urlparse(redirected_from).netloc}"
        super().__init__(detail)
        self.url = url
        self.redirected_from = redirected_from


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
    """The body exceeded ``max_bytes`` and the transfer was aborted at the cap.

    ``content_type`` is the aborted response's declared Content-Type, lower-cased, which is what
    decides whether a declared document earns its one re-dial under the larger cap.
    """

    def __init__(self, url: str, *, bytes_read: int, max_bytes: int, content_type: str = "") -> None:
        super().__init__(f"impersonated response too large ({bytes_read} bytes read > {max_bytes}): {url}")
        self.url = url
        self.bytes_read = bytes_read
        self.max_bytes = max_bytes
        self.content_type = content_type


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
# going to be answered differently. Written only for a block-shaped answer, through
# :func:`note_refusal_if_block_shaped`: a 404 says the path is gone, and a 200 whose body
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


def note_refusal_if_block_shaped(*, dialed_url: str, answered_url: str, status: int) -> bool:
    """Memoise the hosts behind a block-shaped impersonated answer; True when the memo was written.

    The one memo-write rule for both callers. ``answered_url`` is the hop that produced
    ``status`` (:attr:`ImpersonatedResponse.url`) and ``dialed_url`` the hop the caller asked
    for; they differ when the impersonated client was redirected and the block came from a later
    hop. BOTH netlocs are memoised then: the answering host because it is the one whose edge
    refused our fingerprint, and the dialed host so the same chain is not walked again this run
    for the next cited URL on it. Memoising the dialed host alone, as the first version did,
    banned a netloc that never refused us and left the refusing one earning a full dial per URL.
    """
    if status not in IMPERSONATE_BLOCK_STATUSES:
        return False
    answered_netloc = urlparse(answered_url).netloc
    dialed_netloc = urlparse(dialed_url).netloc
    note_impersonation_refused(answered_url)
    note_impersonation_refused(dialed_url)
    via = f", reached from {dialed_netloc}" if _memo_key(dialed_url) != _memo_key(answered_url) else ""
    logger.info(
        f"impersonated client answered {status} by {answered_netloc}{via}; no further impersonated dial "
        "of that host this run"
    )
    return True


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
    document_max_bytes: int,
) -> ImpersonatedResponse:
    """GET ``url`` with a Chrome fingerprint, following redirects by hand under the SSRF guard.

    ``host_sems`` is the caller's own netloc-to-``Semaphore(1)`` map (Tier-1's process-wide
    ``http_fetch.host_semaphores()``, gap-fill v2's module global), taken as the MAP because
    redirect hops may land on other hosts and each hop must contend on ITS host's gate.
    ``deadline_monotonic_s`` is the instant the whole call must be done by and
    ``per_hop_timeout_s`` the per-request ceiling, both the caller's. ``max_bytes`` is the body
    cap for a page and ``document_max_bytes`` the cap for a declared PDF, the two caps the direct
    path reads under (``RESOLUTION_SOURCE_MAX_RESPONSE_BYTES`` and ``DOCUMENT_TEXT_PDF_MAX_BYTES``);
    how the second is applied is :func:`_fetch_pinned_hop`'s business.

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
    redirected_from: str | None = None
    for _hop in range(MAX_REDIRECTS + 1):
        hop = await _fetch_pinned_hop(
            hop_url,
            host_sems=host_sems,
            deadline_monotonic_s=deadline_monotonic_s,
            per_hop_timeout_s=per_hop_timeout_s,
            max_bytes=max_bytes,
            document_max_bytes=document_max_bytes,
            redirected_from=redirected_from,
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
        redirected_from = hop_url
        hop_url = next_url
    raise ImpersonateRedirectLimit(url, final_url=hop_url)


async def _fetch_pinned_hop(
    hop_url: str,
    *,
    host_sems: dict[str, asyncio.Semaphore],
    deadline_monotonic_s: float,
    per_hop_timeout_s: float,
    max_bytes: int,
    document_max_bytes: int,
    redirected_from: str | None,
) -> _Hop:
    """Vet, pin, gate and dial ONE hop, and read its body under the cap.

    The order is the invariant. The host is vetted and resolved BEFORE the gate, so an
    unpinnable host never queues; the timeout is sized AFTER the gate is held, so a hop that
    queued behind a slow host does not then help itself to a fresh ceiling; and the gate is
    released before the next hop acquires its own, never nested, because ``asyncio.Semaphore``
    is not reentrant and an A to B to A chain would self-deadlock. Both awaits that precede the
    dial are bounded by the remaining budget too (:func:`_within_budget`): the vetting lookup is
    an uncancellable ``getaddrinfo`` thread, and the gate's holder can keep it for a whole hop
    timeout, so without the bound the caller's floor would bound when this rung STARTS rather
    than when it stops.

    The two caps. The write callback cannot see the headers, so the first dial runs under
    ``max_bytes`` whatever the body is. When that dial aborts at the cap and the aborted
    response declares a PDF (:func:`declared_pdf`) and ``document_max_bytes`` is the larger cap,
    the same pinned hop is dialed ONCE more under it, still holding the gate and still against
    the same deadline: one extra request in a rare shape, and the rung is no narrower than the
    direct path it substitutes for, which reads a declared PDF under the document cap. A second
    abort, or an oversized body of any other type, is the decline.
    """
    pinned = await _within_budget(rendered_fetch.resolve_pinned_host(hop_url), deadline_monotonic_s)
    if pinned is None:
        raise ImpersonateUnpinnable(hop_url, redirected_from=redirected_from)
    host, vetted_ip = pinned
    resolve_entry = _resolve_entry(host, _hop_port(hop_url), vetted_ip)

    gate = semaphore_for_host(hop_url, host_sems)
    await _within_budget(gate.acquire(), deadline_monotonic_s)
    try:
        try:
            return await _dial(
                hop_url,
                resolve_entry,
                vetted_ip=vetted_ip,
                deadline_monotonic_s=deadline_monotonic_s,
                per_hop_timeout_s=per_hop_timeout_s,
                max_bytes=max_bytes,
            )
        except ImpersonateBodyTooLarge as too_large:
            if document_max_bytes <= max_bytes or not declared_pdf(too_large.content_type):
                raise
            logger.info(
                f"impersonated fetch of {urlparse(hop_url).netloc} declares {too_large.content_type!r} and passed "
                f"the {max_bytes} byte page cap; dialing once more under the {document_max_bytes} byte document cap"
            )
            return await _dial(
                hop_url,
                resolve_entry,
                vetted_ip=vetted_ip,
                deadline_monotonic_s=deadline_monotonic_s,
                per_hop_timeout_s=per_hop_timeout_s,
                max_bytes=document_max_bytes,
            )
    finally:
        gate.release()


async def _within_budget[T](awaitable: Awaitable[T], deadline_monotonic_s: float) -> T:
    """Await ``awaitable`` inside what is left of the budget, or decline as a timeout.

    The ``asyncio.timeout`` here bounds the WAIT, not a transfer: the vetting lookup and the
    gate acquisition are the two awaits before the dial. A lookup that outlives the budget is
    abandoned to its thread and declined; a gate not acquired by the deadline is declined
    without dialing, which is the same outcome the post-gate check would have produced, only at
    the deadline instead of whenever the holder let go.
    """
    remaining = _remaining_s(deadline_monotonic_s)
    try:
        async with asyncio.timeout(remaining):
            return await awaitable
    except TimeoutError:
        raise ImpersonateTransportError(failure_class="timeout", exc="TimeoutError") from None


async def _dial(
    hop_url: str,
    resolve_entry: str,
    *,
    vetted_ip: str,
    deadline_monotonic_s: float,
    per_hop_timeout_s: float,
    max_bytes: int,
) -> _Hop:
    """One pinned GET under one cap, with the gate already held: the session, the request, the
    body read through the write callback, and the pin assertion on what came back.

    The pin is asserted once, after the read, on the address libcurl reports it connected to.
    ``CURLOPT_RESOLVE`` is the boundary that prevents a connection to anything else; this check
    is what makes an inert pin loud. Nothing from a refused response is returned, so a refusal
    after the read still closes the channel, and the bytes read before it are bounded by the cap.
    """
    hop_timeout_s = _hop_timeout_s(deadline_monotonic_s, per_hop_timeout_s)
    reader = _CappedBodyReader(max_bytes)
    async with _pinned_session(resolve_entry, hop_timeout_s) as session:
        try:
            response = await session.request("GET", hop_url, content_callback=reader.write)
        except RequestException as exc:
            if reader.tripped:
                # curl error 23 (``CURLE_WRITE_ERROR``), raised because OUR callback returned
                # ``CURL_WRITEFUNC_ERROR``; the response curl-cffi attaches carries the headers
                # the callback could not see.
                raise ImpersonateBodyTooLarge(
                    hop_url,
                    bytes_read=reader.total,
                    max_bytes=max_bytes,
                    content_type=_content_type_of(exc.response),
                ) from None
            raise ImpersonateTransportError(failure_class=_curl_failure_class(exc), exc=type(exc).__name__) from exc
        except CurlError as exc:
            # The bare parent: ``Curl.setopt`` on an option libcurl rejects, or the multi layer.
            # Neither is a fact about the host, and neither is an ``ImpersonateDeclined``, so
            # left uncaught it would escape both callers' decline clauses and take the whole
            # provider down with it. A ``CurlError`` out of ``AsyncSession.close()`` on the way
            # out of the ``async with`` is outside this try either way.
            raise ImpersonateTransportError(failure_class=_curl_failure_class(exc), exc=type(exc).__name__) from exc
    _check_pin(response.primary_ip, vetted_ip, hop_url)
    headers = response.headers
    return _Hop(
        status=response.status_code,
        url=hop_url,
        content_type=_content_type_of(response),
        server=headers.get("server"),
        body=reader.body(),
        primary_ip=response.primary_ip,
        redirect_url=response.redirect_url or "",
        location=headers.get("location"),
    )


class _CappedBodyReader:
    """The write callback: keep DECOMPRESSED chunks up to the cap, abort the transfer past it.

    libcurl decompresses before the write callback, so the chunks arrive decompressed exactly as
    aiohttp's ``iter_chunked`` yields them after its ``DeflateBuffer``, and the cap counts the
    same bytes ``read_body_capped`` counts. The boundary is ``>``, matching it. The chunk that
    crosses the cap is dropped rather than kept, and returning ``CURL_WRITEFUNC_ERROR`` makes
    libcurl abort the transfer with ``CURLE_WRITE_ERROR`` on the spot, so resident memory is the
    cap and nothing arrives after it. ``tripped`` is how the caller tells that abort from any
    other write error.
    """

    __slots__ = ("_chunks", "max_bytes", "total", "tripped")

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.total = 0
        self.tripped = False
        self._chunks: list[bytes] = []

    def write(self, chunk: bytes) -> int:
        self.total += len(chunk)
        if self.total > self.max_bytes:
            self.tripped = True
            return CURL_WRITEFUNC_ERROR
        self._chunks.append(chunk)
        return len(chunk)

    def body(self) -> bytes:
        return b"".join(self._chunks)


def _pinned_session(resolve_entry: str, hop_timeout_s: float) -> AsyncSession:
    """One short-lived session whose every safety-relevant option is on the CONSTRUCTOR.

    Nothing per request can forget one. ``async with`` on the result is what runs ``close()``,
    which cancels the polling task ``AsyncCurl`` spawns for the session's lifetime, removes the
    loop's readers and writers and closes the pooled handles. ``timeout=`` is libcurl's
    ``CURLOPT_TIMEOUT_MS`` in the non-stream mode the transport uses, the bound on the whole
    transfer, connect included.
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


def _content_type_of(response: Any) -> str:
    """The response's declared Content-Type, lower-cased, ``""`` when absent."""
    return (response.headers.get("content-type") or "").strip().lower()


def _check_pin(primary_ip: str, vetted_ip: str, hop_url: str) -> None:
    """Refuse unless libcurl reports the pinned address (compared as addresses, not strings).

    An empty ``primary_ip`` is a refusal too: curl-cffi fills it from ``CURLINFO_PRIMARY_IP``
    when it parses the completed response, so a completed transfer that cannot say where it
    connected is not one to trust. A host with several A records is a benign source of a
    mismatch when the pin is INERT: ``resolve_vetted_public_ip`` returns the first vetted
    address while libcurl's own resolver may pick another. Every address was vetted, so
    refusing is safe if slightly lossy, and a nonzero count in the live QA means the pin is not
    working, which is the point of the check.
    """
    if primary_ip and _same_address(primary_ip, vetted_ip):
        return
    logger.error(
        f"impersonated fetch pin NOT held for {urlparse(hop_url).netloc}: pinned={vetted_ip} "
        f"connected={primary_ip or '(unreported)'}; refusing the response"
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


def _remaining_s(deadline_monotonic_s: float) -> float:
    """Seconds left before the deadline, or the timeout decline when it has passed."""
    remaining = deadline_monotonic_s - time.monotonic()
    if remaining <= 0.0:
        raise ImpersonateTransportError(failure_class="timeout", exc="TimeoutError")
    return remaining


def _hop_timeout_s(deadline_monotonic_s: float, per_hop_timeout_s: float) -> float:
    """Size this dial's timeout from the remaining budget, the direct path's arithmetic exactly."""
    return min(per_hop_timeout_s, max(_remaining_s(deadline_monotonic_s), RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S))


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


def _curl_failure_class(exc: CurlError) -> str:
    """Bucket a curl-cffi failure into the direct path's ``failure_class`` vocabulary.

    Mirrors ``resolution_source._network_failure_class`` so the two fetchers speak the same
    small set. In the non-stream mode the transport uses, curl-cffi maps the libcurl code to its
    typed subclass through ``code2error`` before raising, so the ladder below sees ``Timeout``
    for code 28, ``SSLError`` for the TLS codes, ``DNSError`` for 6 and ``IncompleteRead`` for
    18 (verified on loopback: a plaintext listener behind an ``https`` URL arrives as
    ``SSLError`` 35). The specific classes come first because ``DNSError`` and ``SSLError`` both
    subclass curl-cffi's ``ConnectionError``, and ``CertificateVerifyError`` subclasses
    ``SSLError``. Everything else, the bare ``CurlError`` a rejected option raises included, is
    ``connection``; no new token is invented, because the vocabulary is a marker contract.
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
