"""Tier-1 resolution-source fetcher.

Fetches the URL(s) explicitly cited in a Metaculus question's resolution
criteria (or fine print), extracts main content with trafilatura, and returns
a compact markdown section that every forecaster reads as the ground truth
the question will be graded against.

Scope is Tier 1 only: plain HTTP with browser-like headers, no LLM calls, no
retries. Sites behind JS walls / heavy anti-bot are deferred to a future
Tier-2 pass (see `FetchStatus` — `blocked` / `js_wall` results are retained
in the returned list to serve as that seam).

Design anchors:

- 2026-07-08 feasibility probe found 75% of questions cite an explicit source
  URL and ~62.5% of them are recoverable by a plain browser-headers fetch.
- Extraction is trafilatura in a thread (`asyncio.to_thread`) — the parse is
  CPU-bound sync C code.
- Per-host politeness: one `asyncio.Semaphore(1)` per netloc, acquired around
  each redirect hop's GET and keyed on THAT hop's host — so chains converging
  on one final host still serialize there. Distinct hosts run concurrently up
  to the connector limit.
- Char caps apply to RAW (non-LLM-processed) content only; the LLM-emitted
  research bundle is never truncated (see the resolution-source plan).
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

import aiohttp
import trafilatura
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    RESOLUTION_SOURCE_ENABLED_ENV,
    RESOLUTION_SOURCE_GLOBAL_CONCURRENCY,
    RESOLUTION_SOURCE_HTTP_TIMEOUT,
    RESOLUTION_SOURCE_JS_WALL_MIN_CHARS,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
    RESOLUTION_SOURCE_MAX_URLS,
    RESOLUTION_SOURCE_PER_URL_MAX_CHARS,
    RESOLUTION_SOURCE_TOTAL_MAX_CHARS,
    RESOLUTION_SOURCE_WALL_TIMEOUT,
    env_flag_enabled,
)
from metaculus_bot.research.http_fetch import (
    BROWSER_HEADERS,
    FilteringResolver,
    build_session,
    read_body_capped,
)
from metaculus_bot.research.providers import ResearchCallable


def _make_filtering_resolver() -> FilteringResolver:
    """Build a FilteringResolver seeded with :func:`_ip_is_disallowed`.

    Hoisted to module scope (from an inline lambda in ``_get_session``) so
    tests can construct one directly and to keep the import-usage adjacency
    that survives ruff's unused-import auto-format.
    """
    return FilteringResolver(disallow=_ip_is_disallowed)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------
#
# URLs enter this module from question resolution_criteria / fine_print — user-
# authored fields that anyone can craft. Fetches run from CI runners on AWS,
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
# Number of HTTP redirects to follow before giving up. Real-world resolution
# sources chain at most 1-2 hops (protocol/canonicalization); 5 leaves slack for
# tracker redirects while keeping the SSRF re-guard cost bounded.
_MAX_REDIRECTS: int = 5
_REDIRECT_STATUSES: frozenset[int] = frozenset({301, 302, 303, 307, 308})


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

    # Hostname branch: resolve via getaddrinfo off the event loop. Reject if
    # ANY address is disallowed (DNS rebinding defense).
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
    except (socket.gaierror, OSError):
        return False

    for info in infos:
        # sockaddr shape: IPv4 = (ip, port); IPv6 = (ip, port, flowinfo, scopeid).
        sockaddr = info[4] if len(info) >= 5 else None
        if not sockaddr:
            return False
        ip_str = sockaddr[0]
        try:
            resolved = ipaddress.ip_address(ip_str)
        except ValueError:
            return False
        if _ip_is_disallowed(resolved):
            return False

    return True


# ---------------------------------------------------------------------------
# Status enum + result dataclass (the Tier-2 seam)
# ---------------------------------------------------------------------------


FetchStatus = Literal["success", "blocked", "not_found", "js_wall", "error", "unsupported_type", "ssrf_blocked"]


@dataclass
class FetchResult:
    url: str
    status: FetchStatus
    text: str  # extracted + truncated; "" unless status == "success"
    http_status: int | None
    content_type: str | None


# ---------------------------------------------------------------------------
# Pure helpers — no I/O
# ---------------------------------------------------------------------------


# Metaculus-injected markdown escapes: `\_`, `\.`, `\&`, `\-`, `\#`, `\(`, `\)`.
# FINDINGS: 3.4% of URLs carry these; one flips 404→success once unescaped.
_MARKDOWN_ESCAPE_RE = re.compile(r"\\([_&.\-#()])")

# Markdown link: [label](https://...) — capture only the URL.
_MARKDOWN_LINK_URL_RE = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)")

# Bare URL — stops at whitespace and common closers.
_BARE_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+")

# Trailing punctuation to strip from an extracted URL.
_TRAILING_PUNCT = ".,;:)]}>\"'"


def strip_markdown_escapes(url: str) -> str:
    """Remove markdown backslash escapes Metaculus injects into rendered URLs."""
    return _MARKDOWN_ESCAPE_RE.sub(r"\1", url)


def extract_source_urls(text: str) -> list[str]:
    """Extract http(s) URLs from ``text``.

    Handles markdown links ``[label](https://…)`` and bare URLs. Strips trailing
    punctuation, applies backslash-unescape, dedupes preserving order (case-
    insensitive scheme+host; exact path and query — query params stay in the
    key because we may need them, e.g. for FRED graph_id; fragments are
    excluded because they're never sent over HTTP). Returns the FULL deduped
    list — the ``RESOLUTION_SOURCE_MAX_URLS`` cap is applied downstream by
    :func:`select_fetchable_urls`, AFTER the self-ref/FRED/Yahoo skip filter,
    so a run of leading self-refs doesn't starve the real sources out of the
    fetch budget.
    """
    if not text:
        return []

    # Collect (start_pos, url) from both regex families so the final order
    # tracks appearance in the source text — not extraction order. Markdown
    # link URLs are typically ALSO matched by the bare-URL regex (its match
    # sits inside the `[label](URL)` parens); the earlier position wins after
    # sort, and dedup drops the duplicate.
    positioned: list[tuple[int, str]] = []
    for match in _MARKDOWN_LINK_URL_RE.finditer(text):
        # Anchor at the URL group's start, not the link's start, so a
        # markdown link and a same-position bare URL rank identically.
        positioned.append((match.start(1), match.group(1)))
    for match in _BARE_URL_RE.finditer(text):
        positioned.append((match.start(), match.group(0)))
    positioned.sort(key=lambda pair: pair[0])

    cleaned: list[str] = []
    for _pos, raw in positioned:
        u = raw
        # Strip trailing punctuation (may repeat: "foo.,").
        while u and u[-1] in _TRAILING_PUNCT:
            u = u[:-1]
        u = strip_markdown_escapes(u)
        if not u.lower().startswith(("http://", "https://")):
            continue
        cleaned.append(u)

    # Dedup preserving order (first-seen URL string wins). Case-insensitive
    # scheme+netloc; exact path/query. Fragments are excluded — they're never
    # sent over HTTP, so URLs differing only by fragment are the same fetch.
    # A bare host and a bare host + "/" collapse to one entry (real questions
    # cite both forms of the same root page — observed on childmortality.org
    # in the 2026-07-09 smoke test, burning a duplicate fetch slot).
    seen: set[str] = set()
    deduped: list[str] = []
    for u in cleaned:
        try:
            parsed = urlparse(u)
        except ValueError:
            continue
        key = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path or '/'}?{parsed.query}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(u)
    return deduped


def is_metaculus_self_ref(url: str) -> bool:
    """A URL that points back at Metaculus is a self-reference (no new info)."""
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return False
    return host == "metaculus.com" or host.endswith(".metaculus.com")


def is_fred_url(url: str) -> bool:
    """FRED series URLs are already served by the financial-data provider."""
    try:
        host = urlparse(url).netloc.lower()
    except ValueError:
        return False
    return host == "fred.stlouisfed.org"


def is_yahoo_ticker_url(url: str) -> bool:
    """Yahoo Finance `/quote/…` URLs are yfinance-served; skip.

    Generic Yahoo article / news URLs remain fetchable — only the ticker
    quote pages overlap with the financial-data provider.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.netloc.lower() == "finance.yahoo.com" and parsed.path.startswith("/quote/")


def select_fetchable_urls(criteria: str | None, fine_print: str | None) -> list[str]:
    """Compose the fetchable URL list from a question's resolution criteria + fine print.

    Skips self-refs (metaculus.com), FRED, and Yahoo ticker URLs — those either
    add no new info or are covered by another provider. Caps at
    ``RESOLUTION_SOURCE_MAX_URLS`` AFTER the skip filter so a run of leading
    self-refs / FRED / Yahoo URLs (Metaculus questions often list the question
    page first) doesn't starve the real sources out of the fetch budget.
    """
    combined = f"{criteria or ''}\n\n{fine_print or ''}"
    urls = extract_source_urls(combined)
    filtered = [u for u in urls if not (is_metaculus_self_ref(u) or is_fred_url(u) or is_yahoo_ticker_url(u))]
    return filtered[:RESOLUTION_SOURCE_MAX_URLS]


def looks_like_js_wall(text: str) -> bool:
    """A 200 OK whose extracted text is shorter than the JS-wall threshold is a
    strong signal the page needs JS to render — Tier-2 candidate."""
    return len(text.strip()) < RESOLUTION_SOURCE_JS_WALL_MIN_CHARS


def format_resolution_sections(results: list[FetchResult], fetched_at: datetime) -> str:
    """Render successful fetches as a markdown body block (orchestrator adds the ``##`` header).

    Returns ``""`` if no results are successful. Enforces
    ``RESOLUTION_SOURCE_TOTAL_MAX_CHARS`` across sections: later sections are
    trimmed (or dropped) once the budget is spent. Per-URL truncation is the
    caller's responsibility (already applied in ``_fetch_one``); this cap
    covers the aggregate section length.
    """
    successes = [r for r in results if r.status == "success"]
    if not successes:
        return ""

    fetched_iso = fetched_at.strftime("%Y-%m-%d")
    caveat = f"Snapshot of the cited resolution source(s) as of {fetched_iso} — primary grading evidence."

    sections: list[str] = []
    remaining = RESOLUTION_SOURCE_TOTAL_MAX_CHARS
    for r in successes:
        # Cheap per-section budget accounting on the text body only. Section
        # overhead (URL heading + fetched-date line) is negligible relative to
        # the RESOLUTION_SOURCE_TOTAL_MAX_CHARS total budget; if the caller
        # tightens it dramatically for a test, we still cut the text
        # conservatively.
        if remaining <= 0:
            break
        body = r.text
        if len(body) > remaining:
            body = body[:remaining].rstrip()
        remaining -= len(body)
        section = f"### {r.url}\n(fetched {fetched_iso})\n\n{body}"
        sections.append(section)

    return caveat + "\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Extraction wrapper (isolated for tests; offloads trafilatura's sync API)
# ---------------------------------------------------------------------------


def _extract_main_text(body: bytes, url: str) -> str | None:
    """Trafilatura extraction. Callers wrap in ``await asyncio.to_thread(...)``.

    Returns None on empty/failed extraction so callers can classify.
    """
    try:
        out = trafilatura.extract(
            body,
            url=url,
            favor_precision=True,
            include_comments=False,
            include_tables=True,
            output_format="txt",
        )
    except (ValueError, TypeError, RuntimeError) as e:
        # Trafilatura occasionally raises on truly malformed input. We soft-fail
        # here so a single broken page doesn't take down the provider.
        logger.warning(f"trafilatura extraction failed for {url}: {e}")
        return None
    if not out or not out.strip():
        return None
    return out


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


_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
_RAW_TEXT_CONTENT_TYPES = ("text/plain", "text/csv")
_JSON_CONTENT_TYPES = ("application/json",)


def _sem_for_host(host_sems: dict[str, asyncio.Semaphore], url: str) -> asyncio.Semaphore:
    """Get-or-create the ``Semaphore(1)`` for ``url``'s netloc.

    Every task in one :func:`fetch_resolution_sources` run shares the same
    ``host_sems`` map, so every request to a given host — original URL or
    redirect hop — contends on the same semaphore object.
    """
    host = urlparse(url).netloc
    sem = host_sems.get(host)
    if sem is None:
        sem = asyncio.Semaphore(1)
        host_sems[host] = sem
    return sem


async def _fetch_one(session: Any, url: str, host_sems: dict[str, asyncio.Semaphore]) -> FetchResult:
    """Fetch a single URL, holding the per-host politeness semaphore hop by hop.

    Content-type routing:
      * HTML → trafilatura extraction (via to_thread) + JS-wall check.
      * JSON → capped raw body, no pretty-print (the data IS the content).
      * text/plain, text/csv → capped raw body.
      * anything else (PDF/binary) — including a missing/empty Content-Type
        header, by design — → ``unsupported_type``, body NOT read.

    Politeness: each hop acquires the semaphore for THAT hop's host around its
    single GET (+ body read on terminal responses) and releases it before
    following a redirect. Keying per hop — not on the original URL's host —
    preserves one-request-per-host when chains from different initial hosts
    converge on the same final host; the strict per-hop acquire/release
    pairing means an A→B→A chain never re-acquires a semaphore it still holds
    (asyncio semaphores are not reentrant).

    SSRF guard: rejects non-public URLs (private / loopback / link-local IPs,
    userinfo tricks, non-http(s) schemes) BEFORE any network I/O and again on
    every redirect Location. The connect-time :class:`FilteringResolver` (see
    :func:`_get_session`) provides the actual DNS-rebinding boundary; these
    preflight checks are fast-fail observability so we surface
    ``ssrf_blocked`` without opening a session. Redirects are followed in-band
    with a hard ``_MAX_REDIRECTS`` cap.

    No retries (Tier 1 anti-goal). Any aiohttp/asyncio error becomes ``error``.
    """
    # Guard the initial URL before any network I/O.
    if not await is_public_http_url(url):
        logger.warning(f"resolution_source ssrf_blocked (initial url): {urlparse(url).netloc}")
        return FetchResult(
            url=url,
            status="ssrf_blocked",
            text="",
            http_status=None,
            content_type=None,
        )

    current_url = url
    # Bounded redirect loop. Each iteration issues ONE GET with
    # allow_redirects=False under the current hop's host semaphore; a redirect
    # status resolves the Location, re-guards, and loops (`continue` unwinds
    # both context managers, releasing the semaphore before the next hop
    # acquires its own — no nesting, so no self-deadlock on revisited hosts).
    # Non-redirect responses fall through to the content-type routing below.
    for _hop in range(_MAX_REDIRECTS + 1):
        async with _sem_for_host(host_sems, current_url):
            try:
                async with session.get(current_url, allow_redirects=False) as resp:
                    netloc = urlparse(current_url).netloc
                    status = resp.status
                    content_type = (resp.headers.get("Content-Type") or "").lower() if resp.headers else ""

                    if status in _REDIRECT_STATUSES:
                        location = resp.headers.get("Location") if resp.headers else None
                        if not location:
                            # Malformed redirect — no Location header.
                            logger.info(f"resolution_source fetched {netloc} (error http={status} no Location)")
                            return FetchResult(
                                url=current_url,
                                status="error",
                                text="",
                                http_status=status,
                                content_type=content_type or None,
                            )
                        next_url = urljoin(current_url, location)
                        if not await is_public_http_url(next_url):
                            logger.warning(
                                f"resolution_source ssrf_blocked (redirect): "
                                f"{urlparse(current_url).netloc} -> {urlparse(next_url).netloc}"
                            )
                            return FetchResult(
                                url=next_url,
                                status="ssrf_blocked",
                                text="",
                                http_status=status,
                                content_type=content_type or None,
                            )
                        current_url = next_url
                        continue  # next hop

                    # Non-redirect response — same status routing as before.
                    if status == 403 or status == 406 or status == 429:
                        logger.info(f"resolution_source fetched {netloc} (blocked http={status})")
                        return FetchResult(
                            url=current_url,
                            status="blocked",
                            text="",
                            http_status=status,
                            content_type=content_type or None,
                        )
                    if status in (404, 410):
                        logger.info(f"resolution_source fetched {netloc} (not_found http={status})")
                        return FetchResult(
                            url=current_url,
                            status="not_found",
                            text="",
                            http_status=status,
                            content_type=content_type or None,
                        )
                    if status != 200:
                        logger.info(f"resolution_source fetched {netloc} (error http={status})")
                        return FetchResult(
                            url=current_url,
                            status="error",
                            text="",
                            http_status=status,
                            content_type=content_type or None,
                        )

                    # 200 OK: route on content type.
                    if any(ct in content_type for ct in _HTML_CONTENT_TYPES):
                        body = await read_body_capped(
                            resp,
                            max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
                            label=f"resolution_source {netloc}",
                        )
                        if body is None:
                            return FetchResult(
                                url=current_url,
                                status="error",
                                text="",
                                http_status=status,
                                content_type=content_type or None,
                            )
                        extracted = await asyncio.to_thread(_extract_main_text, body, current_url)
                        # An empty extraction on a 200 OK is a JS-wall (SPA that
                        # rendered client-side, cookie/consent gate, etc.) —
                        # exactly the Tier-2 candidate signal. Treat identically
                        # to short-but-nonempty extractions.
                        if extracted is None or looks_like_js_wall(extracted):
                            logger.info(f"resolution_source fetched {netloc} (js_wall)")
                            return FetchResult(
                                url=current_url,
                                status="js_wall",
                                text="",
                                http_status=status,
                                content_type=content_type or None,
                            )
                        truncated = extracted[:RESOLUTION_SOURCE_PER_URL_MAX_CHARS]
                        logger.info(f"resolution_source fetched {netloc} (success)")
                        return FetchResult(
                            url=current_url,
                            status="success",
                            text=truncated,
                            http_status=status,
                            content_type=content_type or None,
                        )

                    if any(ct in content_type for ct in _JSON_CONTENT_TYPES) or any(
                        ct in content_type for ct in _RAW_TEXT_CONTENT_TYPES
                    ):
                        body = await read_body_capped(
                            resp,
                            max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
                            label=f"resolution_source {netloc}",
                        )
                        if body is None:
                            return FetchResult(
                                url=current_url,
                                status="error",
                                text="",
                                http_status=status,
                                content_type=content_type or None,
                            )
                        raw = body.decode("utf-8", errors="replace")
                        truncated = raw[:RESOLUTION_SOURCE_PER_URL_MAX_CHARS]
                        logger.info(f"resolution_source fetched {netloc} (success)")
                        return FetchResult(
                            url=current_url,
                            status="success",
                            text=truncated,
                            http_status=status,
                            content_type=content_type or None,
                        )

                    # Anything else — PDF, images, etc. Do NOT read the body.
                    # INTENDED limitation: a 200 OK with a missing/empty Content-Type header
                    # also lands here (ct='') and is dropped unread. Real resolution sources
                    # send Content-Type; content-sniffing would re-open the don't-read-unknown-
                    # bodies posture for a case that mostly can't happen. The per-URL
                    # FetchStatus is the Tier-2 seam if logs ever show `unsupported_type ct=''`.
                    logger.info(f"resolution_source fetched {netloc} (unsupported_type ct={content_type!r})")
                    return FetchResult(
                        url=current_url,
                        status="unsupported_type",
                        text="",
                        http_status=status,
                        content_type=content_type or None,
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.info(f"resolution_source fetch error for {current_url}: {type(e).__name__}: {e}")
                return FetchResult(
                    url=current_url,
                    status="error",
                    text="",
                    http_status=None,
                    content_type=None,
                )

    # Fell out of the loop -> exceeded _MAX_REDIRECTS.
    logger.info(f"resolution_source redirect chain exceeded {_MAX_REDIRECTS} hops (final={current_url})")
    return FetchResult(
        url=current_url,
        status="error",
        text="",
        http_status=None,
        content_type=None,
    )


async def fetch_resolution_sources(urls: list[str]) -> list[FetchResult]:
    """Fetch each URL under per-netloc Semaphore(1) politeness.

    Distinct hosts run concurrently up to the connector limit; same-host
    requests serialize (politeness — e.g. StatCan asks Crawl-delay: 2). The
    shared ``host_sems`` map is handed to every ``_fetch_one`` task so each
    redirect hop contends on ITS host's semaphore — chains from different
    initial hosts that converge on one final host still serialize there.
    Session is closed in ``finally``.

    Teardown race guard (F5): the outer factory wraps this call in
    ``asyncio.wait_for``. When the wall-clock timeout fires, wait_for cancels
    this coroutine — but if the gather is still in flight we'd exit the
    ``async with session`` block while children are mid-request, and aiohttp
    would then close their transports out from under them (surfacing as
    scary tracebacks in logs, and in extreme cases resource-warning fires
    on connections that never got cleaned up). We use explicit Task objects
    so we can cancel + drain them in a ``finally`` before the session closes.
    """
    host_sems: dict[str, asyncio.Semaphore] = {}

    session_cm = _get_session()
    async with session_cm as session:
        tasks = [asyncio.create_task(_fetch_one(session, u, host_sems)) for u in urls]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return list(results)
        finally:
            # Whether we exit normally or via cancellation, cancel any still-
            # running task and let them settle before the session closes.
            # (No-op cost when everything already finished successfully.)
            for t in tasks:
                if not t.done():
                    t.cancel()
            # return_exceptions=True: drained tasks may surface CancelledError,
            # which is expected here.
            await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def resolution_source_provider(is_benchmarking: bool = False) -> ResearchCallable:
    """Factory returning the async ResearchCallable for the resolution-source fetcher.

    Gating (both hard):

    - ``is_benchmarking=True`` short-circuits to ``""`` (leakage guard — current
      page content post-dates any backtest window, same rationale as the
      prediction-market provider).
    - Env flag ``RESOLUTION_SOURCE_ENABLED`` must be truthy.

    Returns section BODY only; the orchestrator prepends the ``## Resolution
    Source Snapshot`` header. Inner ``### {url}`` headers stay at h3 — the
    orchestrator's heading demotion only touches h1/h2, and h3 is already
    correctly nested under the h2 provider header.
    """

    async def _fetch(question: MetaculusQuestion) -> str:
        if is_benchmarking:
            return ""  # noqa: ASYNC910
        if not env_flag_enabled(RESOLUTION_SOURCE_ENABLED_ENV):
            return ""  # noqa: ASYNC910

        urls = select_fetchable_urls(question.resolution_criteria, question.fine_print)
        if not urls:
            return ""  # noqa: ASYNC910

        try:
            results = await asyncio.wait_for(
                fetch_resolution_sources(urls),
                timeout=RESOLUTION_SOURCE_WALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(f"resolution_source: wall-clock timeout after {RESOLUTION_SOURCE_WALL_TIMEOUT}s")
            return ""  # noqa: ASYNC910

        n_fail = sum(1 for r in results if r.status != "success")
        if n_fail:
            logger.info(
                f"resolution_source: {n_fail}/{len(results)} urls unfetched (Tier-2 candidates)",
            )
        return format_resolution_sections(results, datetime.now(timezone.utc))

    return _fetch
