"""Outbound-safety leaf of the shared fetch ladder: the SSRF guard, the session, the host gate.

Every name here is a precondition on dialing a URL rather than a step of the ladder, which is
why this module imports nothing from the rest of the package: the two second transports
(``rendered_fetch``, ``impersonated_fetch``) reach for these checks, and reaching into the
fetcher for them used to be a circular import. The escalation rungs, the classification path
and the per-question bookkeeping live in their own modules beside this one.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from typing import Literal
from urllib.parse import urljoin, urlparse

import aiohttp

from metaculus_bot.constants import (
    RESOLUTION_SOURCE_GLOBAL_CONCURRENCY,
    RESOLUTION_SOURCE_HTTP_TIMEOUT,
)
from metaculus_bot.research.http_fetch import (
    BROWSER_HEADERS,
    FilteringResolver,
    build_session,
    semaphore_for_host,
)
from metaculus_bot.research.resolution_fetch_result import FetchResult
from metaculus_bot.research.resolution_url_scan import is_metaculus_self_ref

logger = logging.getLogger(__name__)


def _make_filtering_resolver() -> FilteringResolver:
    """Build a FilteringResolver seeded with :func:`_ip_is_disallowed`.

    Hoisted to module scope (from an inline lambda in ``_get_session``) so
    tests can construct one directly and to keep the import-usage adjacency
    that survives ruff's unused-import auto-format.
    """
    return FilteringResolver(disallow=_ip_is_disallowed)


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------
#
# URLs enter this module from question resolution_criteria / fine_print — user-
# authored fields that anyone can craft. Fetches run from CI runners,
# where hitting http://169.254.169.254/latest/meta-data/ (or any RFC1918 host,
# any private IP, ::1, fe80::/10, etc.) would exfiltrate instance identity into
# the research prompt AND into the public Metaculus comment. Legitimate
# resolution sources are always public websites, so a blanket public-only
# constraint costs zero functionality.
#
# Kept local to this module. `http_fetch.py` is shared with the prediction-
# market provider, which only hits a fixed allow-list of API hosts (Polymarket
# Gamma, Kalshi, Manifold) and doesn't need this. If a third caller lands in
# http_fetch that also takes user-supplied URLs, hoist this guard there.
#
# Redirect policy (hop cap + 3xx status set) lives in http_fetch.py as
# MAX_REDIRECTS / REDIRECT_STATUSES, shared with research.agentic.tools so the
# two SSRF-guarded fetchers can't drift.


def _ip_is_disallowed(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Reject every non-globally-routable IP category.

    The explicit predicates keep review clarity for the obvious classes
    (private / loopback / link-local / reserved / multicast / unspecified).
    The `not ip.is_global` clause is the catch-all — it covers ranges the
    explicit list misses, most notably CGNAT / shared address space
    100.64.0.0/10 (which is not `is_private` on ipaddress) and IPv4-mapped
    IPv6 forms of private ranges.
    """
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or not ip.is_global
    )


async def is_public_http_url(url: str) -> bool:
    """Return True iff ``url`` is safe to fetch from CI (public HTTP(S) only).

    Rejects: non-http(s) schemes, URLs carrying userinfo, IP-literal hosts that
    fall in any non-global range (private / loopback / link-local / reserved /
    multicast / unspecified), and hostnames whose DNS resolution surfaces ANY
    disallowed IP.

    This is the FAST-PATH observability guard: it lets us emit ``ssrf_blocked``
    without ever opening a session. It is NOT the DNS-rebinding trust
    boundary — the resolver aiohttp uses at connect time is (see
    :func:`_get_session` and :class:`FilteringResolver`). A rebinding server
    that returned a public IP here and a private IP to the connect-time
    resolver would still be rejected there.

    DNS failure -> False (unfetchable; would fail the fetch anyway, and we want
    the caller to uniformly emit an ``ssrf_blocked`` result).

    Async because DNS goes through ``asyncio.to_thread(socket.getaddrinfo, ...)``
    to avoid blocking the event loop.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme.lower() not in ("http", "https"):
        return False

    # Userinfo defeats hostname-based trust: `https://trusted@169.254.169.254/`
    # renders as if targeting `trusted` but actually hits the IMDS.
    if parsed.username is not None or parsed.password is not None:
        return False

    # `.hostname` strips userinfo, port, and IPv6 brackets, and lowercases —
    # harmless here: both ip_address() and getaddrinfo() are case-insensitive.
    host = parsed.hostname or ""
    if not host:
        return False

    # IP-literal branch: no DNS needed. Try IPv4 first, then IPv6.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    if ip is not None:
        return not _ip_is_disallowed(ip)

    return await _every_resolved_address_is_public(host)


async def resolve_vetted_public_ip(host: str) -> str | None:
    """Resolve ``host`` off the event loop and return its FIRST address — but only
    after vetting EVERY resolved address.

    The contract is reject-if-ANY-address-disallowed: a single disallowed address
    among the results rejects the whole hostname (DNS rebinding defense), as does a
    resolution failure, an unparseable sockaddr, or an empty result — an unfetchable
    host must reach the caller as one uniform rejection. Only when every address is
    publicly routable does the first one come back, so a caller may safely pin a
    connection to it.

    The one DNS-vetting predicate for both SSRF-guarded fetchers: the Tier-1
    preflight (:func:`is_public_http_url`) consumes the bool view below, and the
    agentic rendered rung pins Chromium's DNS to the returned IP.
    """
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
    except (socket.gaierror, OSError):
        return None
    vetted_ip: str | None = None
    for info in infos:
        # sockaddr shape: IPv4 = (ip, port); IPv6 = (ip, port, flowinfo, scopeid).
        sockaddr = info[4] if len(info) >= 5 else None
        if not sockaddr:
            return None
        try:
            resolved = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            return None
        if _ip_is_disallowed(resolved):
            return None
        if vetted_ip is None:
            vetted_ip = str(resolved)
    return vetted_ip


async def _every_resolved_address_is_public(host: str) -> bool:
    """True iff ``host`` resolves and EVERY resolved address is publicly routable.

    Bool view of :func:`resolve_vetted_public_ip`; a rejection surfaces to the
    caller as one uniform ``ssrf_blocked``.
    """
    return await resolve_vetted_public_ip(host) is not None


# ---------------------------------------------------------------------------
# Network layer (patched in tests via `_get_session`)
# ---------------------------------------------------------------------------


def _get_session() -> aiohttp.ClientSession:
    """Construct a fresh aiohttp session with browser-like headers. Patched in tests.

    The session's TCPConnector is wired to a :class:`FilteringResolver` seeded
    with :func:`_ip_is_disallowed`. This is the actual DNS-rebinding boundary:
    aiohttp's connect-time DNS lookup (and its DNS cache — see aiohttp docs)
    only ever surface IPs that pass the same predicate as
    :func:`is_public_http_url`, so the preflight guard can't be raced by a
    rebinding server between resolve and connect. The preflight guard remains
    for fast observability (it lets us emit ``ssrf_blocked`` on obviously bad
    URLs without opening a session), but it is not the trust boundary.
    """
    return build_session(
        timeout_s=RESOLUTION_SOURCE_HTTP_TIMEOUT,
        connector_limit=RESOLUTION_SOURCE_GLOBAL_CONCURRENCY,
        headers=BROWSER_HEADERS,
        resolver=_make_filtering_resolver(),
    )


def _sem_for_host(host_sems: dict[str, asyncio.Semaphore], url: str) -> asyncio.Semaphore:
    """Get-or-create the ``Semaphore(1)`` for ``url``'s netloc.

    Every task in one :func:`fetch_resolution_sources` run shares the same
    ``host_sems`` map, so every request to a given host — original URL or
    redirect hop — contends on the same semaphore object. Since 2026-09-03 that map
    is :func:`http_fetch.host_semaphores`, shared by every question running
    concurrently rather than rebuilt per provider call; the parameter stays because
    the gap-fill v2 loop reaches this function with its own map.

    Kept as a thin wrapper over the shared implementation because the test suites
    monkeypatch THIS name to observe or replace the gate.
    """
    return semaphore_for_host(url, host_sems)


HopRefusal = Literal["ssrf_blocked", "metaculus_self_ref"]


async def _hop_refusal(candidate_url: str) -> HopRefusal | None:
    """Why this module must not fetch a URL it DERIVED, or None when it may.

    The ONE home of the two checks every derived URL owes before anything dials it: a
    ``Location`` header, a meta-refresh target, the innermost URL of a Wayback capture, and
    the URL the direct fetch landed on that the browser rung is about to render. The
    ``is_public_http_url`` preflight runs FIRST (the fast-fail SSRF view; the connect-time
    resolver stays the real boundary), then the Metaculus self-reference refusal, and that
    order is a telemetry contract: a URL that is both non-public and a self-reference has always
    been recorded as ``ssrf_blocked``, never as the self-reference's ``blocked``. Shared so a rung
    cannot ship with one of the checks missing, and so a third check added here reaches every
    site at once; the callers own their log lines and what they return, because a hop that is
    refused is terminal for the redirect loop (:func:`_vetted_hop_target`) and a decline for the
    render and Wayback rungs (:func:`_rendered_rung`, :func:`_wayback_snapshot_result`).
    """
    if not await is_public_http_url(candidate_url):
        return "ssrf_blocked"
    if is_metaculus_self_ref(candidate_url):
        return "metaculus_self_ref"
    return None


async def _landing_refused(landing_url: str, cited_url: str, *, action: str) -> bool:
    """Whether a second transport must NOT be handed the URL the direct fetch LANDED on.

    The one home of the re-vet every rung that dials ``direct.url`` owes, today the browser rung
    and the impersonated retry: a landing that differs from the cited URL is a DERIVED URL, so it
    goes back through :func:`_hop_refusal` before a transport that never passed aiohttp's
    connect-time resolver dials it. A guard surface, so one copy: a copy that dropped the
    equality short-circuit or the refusal check would read exactly like the correct one, and this
    check is what stops a second transport from dialing a host the direct path refused. The
    equality short-circuit is not an optimisation but the rule that a cited URL owes nothing
    here, because :func:`_fetch_direct` already vetted every hop it followed. ``action`` is the
    verb the WARNING names; the callers own what they return, which is a decline with no attempt
    recorded, because the refusal is decided where the dialed URL is decided.
    """
    if landing_url == cited_url:
        return False
    if await _hop_refusal(landing_url) is None:
        return False
    logger.warning(
        "resolution_source: not %s %s, where the cited %s landed: a host we do not fetch",
        action,
        urlparse(landing_url).netloc,
        urlparse(cited_url).netloc,
    )
    return True


async def _vetted_hop_target(
    target: str, current_url: str, *, http_status: int, content_type: str, kind: str
) -> FetchResult | str:
    """The absolute next URL for a derived hop, or the terminal refusal it earns.

    The terminal-result form of :func:`_hop_refusal`, for a URL this module derived from a
    response inside the redirect loop, a ``Location`` header or a meta-refresh tag: the
    refusal token is mapped onto the ``FetchResult`` the loop ends on, and the two status
    strings it produces, ``ssrf_blocked`` and ``blocked``, are telemetry contracts, as is the
    ``metaculus_self_ref`` reason the second carries. ``kind`` is only there to say which hop
    shape a log line came from.
    """
    next_url = urljoin(current_url, target)
    refusal = await _hop_refusal(next_url)
    if refusal == "ssrf_blocked":
        logger.warning(
            f"resolution_source ssrf_blocked ({kind}): {urlparse(current_url).netloc} -> {urlparse(next_url).netloc}"
        )
        return FetchResult(
            url=next_url,
            status="ssrf_blocked",
            text="",
            http_status=http_status,
            content_type=content_type or None,
        )
    if refusal == "metaculus_self_ref":
        # The URL pre-filter drops self-refs, but a redirect (of either shape) can
        # still land on the question platform's own site (metaculus.com or
        # competitions.mantic.com); don't follow it (no new info, and keeps our IP
        # off the same host the critical API uses). The reason is what keeps the paid
        # rung off this URL as well (`_url_context_rung_applies`): that rung is handed
        # the CITED url, and Gemini would follow the same redirect onto the refused page.
        logger.info(
            f"resolution_source metaculus_self_ref ({kind}): "
            f"{urlparse(current_url).netloc} -> {urlparse(next_url).netloc}"
        )
        return FetchResult(
            url=next_url,
            status="blocked",
            status_reason="metaculus_self_ref",
            text="",
            http_status=http_status,
            content_type=content_type or None,
        )
    return next_url
