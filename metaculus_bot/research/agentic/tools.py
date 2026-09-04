"""The four tools the gap-fill v2 driver calls, and the fetch ladder behind ``fetch``.

This module owns the handlers (``search_news``, ``search_web``, ``fetch``,
``read_document``), the ladder spine they run on — the plain hop loop with its redirect
vetting, the headless-Chromium rendered rung with its DNS pinning, the window cache that
serves ``start_char`` continuations — and ``build_gap_fill_tools``, whose list order is the
order the driver sees.

Support pieces live next door: ``tool_descriptions`` (driver-facing text + JSON schemas),
``tool_backends`` (the AskNews / Exa / Gemini calls and their result formatting),
``fetch_outcomes`` (classifying one plain HTTP response), ``local_document`` (the local PDF
rung, the run's document cache, and the passage digest ``read_document`` serves).

The seams the suite monkeypatches
— ``_read_response_body``, ``_fetch_plain``, ``_try_rendered_fetch``, ``_resolve_pinned_host``,
``_acquire_local_document``, ``_run_document_read_sync``, ``read_document``,
``_READ_DOCUMENT_TIMEOUT_S``, ``_RENDERED_FETCH_GLOBAL_SEMAPHORE`` — are attributes of THIS
module and are resolved here at call time, so their callers stay here even where the callee
moved out.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
from collections import OrderedDict
from time import monotonic
from typing import Any
from urllib.parse import urlparse

import aiohttp

from metaculus_bot.constants import (
    ASKNEWS_BACKOFF_SECS,  # noqa: F401  # re-export: tests read the AskNews retry ladder's constants off this module
    ASKNEWS_CLIENT_ID_ENV,
    ASKNEWS_MAX_TRIES,  # noqa: F401  # re-export: see ASKNEWS_BACKOFF_SECS above
    ASKNEWS_SECRET_ENV,
    DOCUMENT_DIGEST_TOP_K,
    DOCUMENT_TEXT_PDF_MAX_BYTES,
    EXA_API_KEY_ENV,
    GOOGLE_API_KEY_ENV,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
)
from metaculus_bot.research import resolution_source
from metaculus_bot.research.agentic import local_document
from metaculus_bot.research.agentic.fetch_outcomes import (
    _FETCH_MIN_CONTENT_CHARS,
    _HTML_CONTENT_TYPE_TOKENS,
    _RETRYABLE_FETCH_BLOCK_STATUSES,  # noqa: F401  # re-export: the suite parametrizes the blocked-status set off this module
    _TEXTUAL_CONTENT_TYPE_TOKENS,
    DOCUMENT_NEEDED_METHOD,
    PlainFetchResult,
    _body_is_document,
    _content_type_is_document,
    _content_type_is_image,
    _content_type_is_pdf,
    _document_needed_result,
    _extract_links_from_html,
    _fetch_plain_url_block,
    _non_ok_status_result,
    _plain_html_outcome,
    _plain_redirect_outcome,
    _plain_textual_outcome,
    matched_throttle_phrase,
)
from metaculus_bot.research.agentic.robots_policy import google_extended_disallows
from metaculus_bot.research.agentic.tool_backends import (
    _call_asknews_search,
    _call_exa_search,
    _format_asknews_results,
    _format_exa_results,
    _run_document_read_sync,
)
from metaculus_bot.research.agentic.tool_descriptions import (
    _FETCH_PARAMETERS,
    _READ_DOCUMENT_PARAMETERS,
    _SEARCH_NEWS_PARAMETERS,
    _SEARCH_WEB_PARAMETERS,
    FETCH_DESCRIPTION,
    READ_DOCUMENT_DESCRIPTION,
    SEARCH_NEWS_DESCRIPTION,
    SEARCH_WEB_DESCRIPTION,
)
from metaculus_bot.research.agentic.types import ToolOutcome, ToolSpec
from metaculus_bot.research.document_text import is_pdf_body
from metaculus_bot.research.http_fetch import (
    BROWSER_HEADERS,
    MAX_REDIRECTS,
    REDIRECT_STATUSES,
    decode_text_body,
    read_body_capped,
)

logger = logging.getLogger(__name__)

_FETCH_WINDOW_CHARS = 8000
_FETCH_CACHE_MAX_ENTRIES = 50
_RENDERED_FETCH_TIMEOUT_MS = 35_000
# Fixed settle after the DOM is ready, in place of waiting for network idle. Measured
# 2026-09-03: 4 of the replay's 10 render rescues came from pages where ``page.goto`` raised
# TimeoutError with the DOM fully rendered (both ballotpedia questions, both fts.unocha.org
# summaries) - network idle never arrives on a page carrying a long-poll widget or an
# analytics beacon, so waiting for it discarded content Chromium already had. The worst case
# is unchanged rather than longer: the goto budget below is the 35 s cap MINUS this settle.
_RENDERED_SETTLE_MS = 2_000
_READ_DOCUMENT_TIMEOUT_S = 60.0
# read_document is acquisition-first, so its budget holds two rungs: the free local ladder,
# then the paid reader on what the total leaves it. The ToolSpec ceiling stays 70 s (see
# build_gap_fill_tools) and so does the loop's wall discipline: 25 + 40 = 65, plus 5 s of
# margin. The paid reader keeps its whole 60 s whenever acquisition failed fast, which is the
# common case for the URLs that reach it (archived all-fail p50 0.3 s).
_LOCAL_DOCUMENT_BUDGET_S = 25.0
_READ_DOCUMENT_TOTAL_BUDGET_S = 65.0
# Process-global cap on concurrent headless Chromium launches. Module-level, so
# the bound spans all questions running under the orchestrator's Semaphore(6):
# each Chromium is ~100-300MB, the driver's parallel_tool_calls can request many
# fetches in one step, and an unbounded 6·N launch would OOM the runner (an
# escape try/except cannot catch). Cap 2 covers real bursts of 1-3 rendered
# pages while bounding worst-case memory.
_RENDERED_FETCH_GLOBAL_SEMAPHORE = asyncio.Semaphore(2)
_FETCH_HOST_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
_FETCH_TEXT_CACHE: OrderedDict[str, str] = OrderedDict()
_FETCH_LINKS_CACHE: OrderedDict[str, list[str]] = OrderedDict()
# URLs where Chromium ran and extracted nothing, so the second launch a documented escalation
# would spend (a js-walled `fetch` the driver follows with `read_document`) is skipped: 100-300
# MB and up to 35 s out of the process-global launch cap, to learn what this run already knows.
# Deliberately records ONLY that outcome. A `blocked` or `error` GET is not memoized — 429 sits
# in the retryable block set and the driver is TOLD to try those URLs again, so caching them
# would suppress a retry the tool descriptions promise (and a throttle interstitial is exactly
# the q45191 case that must stay re-requestable).
_RENDERED_NO_TEXT: OrderedDict[str, None] = OrderedDict()
_PLAYWRIGHT_WARNED = False

# One robots.txt read per host, for the Google-Extended pre-check on the paid reader. The value
# is the fetched text, or None when we could not read it (which proceeds to pay). Bounded and
# FIFO like the two caches above rather than a plain dict, because this is process-global state
# that outlives one question: a whole run's hosts would otherwise accumulate here.
_ROBOTS_FETCH_TIMEOUT_S = 5.0
_ROBOTS_TXT_CACHE: OrderedDict[str, str | None] = OrderedDict()


def _host_gate(url: str) -> asyncio.Semaphore:
    return resolution_source._sem_for_host(_FETCH_HOST_SEMAPHORES, url)


def _cache_fetch_result(url: str, text: str, links: list[str]) -> None:
    _FETCH_TEXT_CACHE[url] = text
    _FETCH_TEXT_CACHE.move_to_end(url)
    _FETCH_LINKS_CACHE[url] = list(links)
    _FETCH_LINKS_CACHE.move_to_end(url)
    while len(_FETCH_TEXT_CACHE) > _FETCH_CACHE_MAX_ENTRIES:
        evicted_url, _ = _FETCH_TEXT_CACHE.popitem(last=False)
        _FETCH_LINKS_CACHE.pop(evicted_url, None)


def _slice_fetch_window(text: str, start_char: int) -> tuple[str, bool]:
    start = max(0, start_char)
    if start >= len(text):
        return "", False
    end = min(len(text), start + _FETCH_WINDOW_CHARS)
    window = text[start:end]
    if end >= len(text):
        return window, False
    marker = f"\n[truncated at {end} of {len(text)} chars — call again with start_char={end}]"
    return window + marker, True


def _fetch_from_cache(url: str, start_char: int) -> ToolOutcome | None:
    cached = _FETCH_TEXT_CACHE.get(url)
    if cached is None:
        return None
    _FETCH_TEXT_CACHE.move_to_end(url)
    links = list(_FETCH_LINKS_CACHE.get(url, []))
    window, truncated = _slice_fetch_window(cached, start_char)
    return ToolOutcome(content_markdown=window, links=links, method="cache", truncated=truncated)


def _format_fetch_error(message: str, *, status: str = "error", method: str = "plain") -> ToolOutcome:
    return ToolOutcome(content_markdown=message, method=method, status=status)


def _render_fetch_outcome(url: str, text: str, links: list[str], *, method: str, start_char: int) -> ToolOutcome:
    _cache_fetch_result(url, text, links)
    window, truncated = _slice_fetch_window(text, start_char)
    return ToolOutcome(content_markdown=window, links=links, method=method, truncated=truncated)


_NO_CONTENT_FETCH_MSG = (
    "No readable content: {url} returned HTTP 200 but produced no extractable text — "
    "neither the plain fetch nor the headless-browser render read anything (JavaScript "
    "wall, consent/anti-bot gate, or a genuinely empty page). Nothing from this URL was "
    "read; do NOT cite it as a fetched source. Try read_document(url, ask) for a targeted "
    "extraction, or find another source."
)


_THROTTLED_FETCH_MSG = (
    "Rate limited: {url} returned HTTP 200, but its body is a short interstitial carrying the "
    'throttle phrase "{phrase}", not the page. Nothing from this URL was read: do NOT cite it as '
    "a fetched source, and do NOT read it as evidence that the fact is unavailable — the page "
    "exists and we were refused for asking too fast. Do other work now (a different host, "
    "another gap) and call fetch on this URL again later in the run; a retry is a real request, "
    "not a replay of this one."
)


def _throttled_fetch_outcome(url: str, text: str, phrase: str, *, method: str) -> ToolOutcome:
    """Outcome for a 200-OK body that is the host's rate-limit interstitial, not the page.

    Mirrors :func:`_empty_fetch_outcome` in both guards — a non-``"ok"`` status AND a method
    absent from ``provenance._METHOD_TO_TIER`` — so an interstitial can never be stamped
    ``fetched`` and supersede the briefing. Deliberately NOT cached, which is the half of
    this fix that q45191 turned on: the interstitial was cached under ``method="rendered"``
    and served straight back when the driver retried the same URL, so its retry could not
    have succeeded however many slots it spent.
    """
    logger.warning(f"AGENTIC_FETCH_THROTTLED: url={url} method={method} chars={len(text.strip())} phrase={phrase}")
    return ToolOutcome(
        content_markdown=_THROTTLED_FETCH_MSG.format(url=url, phrase=phrase),
        method="throttled",
        status="throttled",
    )


def _read_content_outcome(url: str, text: str, links: list[str], *, method: str, start_char: int) -> ToolOutcome:
    """Render a body the ladder read, unless it is a throttle interstitial standing in for it.

    The one seam every successful ``fetch`` return goes through, so no success path can cache
    or tier an interstitial. The ladder itself is untouched: a throttled plain body still
    escalates to the rendered rung exactly as a thin one does, and only the outcome the
    driver receives changes.
    """
    phrase = matched_throttle_phrase(text)
    if phrase is not None:
        return _throttled_fetch_outcome(url, text, phrase, method=method)
    return _render_fetch_outcome(url, text, links, method=method, start_char=start_char)


def _empty_fetch_outcome(url: str) -> ToolOutcome:
    """Outcome for a 200-OK page the ladder could not read (zero extractable text).

    Distinct ``status``/``method`` of ``"empty"`` — never ``"ok"``/``"plain"`` — so the
    loop's tier stamping (which grants "fetched" only on a ``status == "ok"``,
    fetched-class-method outcome; see ``loop._harvest_verification_tiers``) can never
    mark an unread page authoritative. Two deterministic guards, not one: the non-"ok"
    status AND the unmapped method. Deliberately NOT cached — caching the placeholder
    would let a later paginated fetch serve it back as ``method == "cache"`` (a
    fetched-tier method) and re-launder the tier.
    """
    return ToolOutcome(content_markdown=_NO_CONTENT_FETCH_MSG.format(url=url), method="empty", status="empty")


def _warn_playwright_unavailable_once(exc: BaseException) -> None:
    global _PLAYWRIGHT_WARNED  # noqa: PLW0603  # one-shot process-wide warn latch so the rendered rung logs once per run
    if _PLAYWRIGHT_WARNED:
        return
    _PLAYWRIGHT_WARNED = True
    logger.warning("agentic fetch rendered rung unavailable: %s: %s", type(exc).__name__, exc)


async def _read_response_body(
    resp: aiohttp.ClientResponse, label: str, *, max_bytes: int = RESOLUTION_SOURCE_MAX_RESPONSE_BYTES
) -> bytes | None:
    """The response body up to ``max_bytes``, or None past it.

    The cap is a parameter because a declared PDF is read under the document cap rather than
    the page cap: the 6.7 MB report local extraction reads in 5.3 s is over the page cap, and
    refusing it here would send the one document the local rung exists for to the paid reader
    (which returned nothing for that file).
    """
    return await read_body_capped(resp, max_bytes=max_bytes, label=label)


def _body_too_large_result(current_url: str, content_type: str, *, declared_pdf: bool) -> PlainFetchResult:
    """The result for a body past its cap — which cap it was decides what the driver is told.

    A declared document gets its own method and message: it was too big to read locally AND
    too big to be worth having a model retrieve, so read_document reports the same rather than
    paying a reader for bytes we just refused. Anything else keeps the generic size error.
    """
    if declared_pdf:
        return local_document.oversize_result(current_url, content_type)
    return PlainFetchResult(
        status="error",
        method="plain",
        text="Fetch body exceeded the size limit.",
        links=[],
        url=current_url,
        content_type=content_type or None,
    )


async def _plain_response_outcome(resp: aiohttp.ClientResponse, current_url: str) -> PlainFetchResult | str:
    """Classify one HTTP response: a terminal result, or the next URL on a vetted 3xx."""
    status = resp.status
    content_type = (resp.headers.get("Content-Type") or "").lower() if resp.headers else ""
    if status in REDIRECT_STATUSES:
        return await _plain_redirect_outcome(resp, current_url, content_type)
    non_ok = _non_ok_status_result(status, current_url, content_type)
    if non_ok is not None:
        return non_ok
    if _content_type_is_image(content_type):
        # An image is the one document shape with no text a local rung could read, so its bytes
        # buy nothing and it keeps the pre-read escalation to the paid reader. A PDF no longer
        # takes this exit: its bytes are exactly what the local rung needs.
        return _document_needed_result(current_url, content_type)
    declared_pdf = _content_type_is_pdf(content_type)
    body = await _read_response_body(
        resp,
        f"agentic fetch {urlparse(current_url).netloc}",
        max_bytes=DOCUMENT_TEXT_PDF_MAX_BYTES if declared_pdf else RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
    )
    if body is None:
        return _body_too_large_result(current_url, content_type, declared_pdf=declared_pdf)
    if declared_pdf or is_pdf_body(body):
        # Local extraction first, whether the header said PDF or only the magic bytes did.
        return await local_document.pdf_fetch_result(body, url=current_url, content_type=content_type)
    if _body_is_document(body):
        return _document_needed_result(current_url, content_type)

    # Charset-honoring decode (BOM > declared charset > UTF-8), not a
    # forced UTF-8 read: a windows-1252 or UTF-16 body decoded that way
    # is `0�.�4�2�`-style mojibake that reached the driver as
    # status="ok". The ratio is the refusal signal on the textual
    # branch; the HTML branch is unaffected because its main text comes
    # from `_extract_main_text`, which decodes the raw bytes itself.
    html, undecodable_ratio = decode_text_body(body, content_type)
    if any(token in content_type for token in _HTML_CONTENT_TYPE_TOKENS) or "<html" in html.lower():
        return await _plain_html_outcome(body, html, content_type, current_url)
    if any(token in content_type for token in _TEXTUAL_CONTENT_TYPE_TOKENS) or not content_type:
        return _plain_textual_outcome(html, undecodable_ratio, content_type, current_url)
    return PlainFetchResult(
        status="error",
        method="plain",
        text=f"Unsupported content type: {content_type or 'unknown'}",
        links=[],
        url=current_url,
        content_type=content_type or None,
    )


async def _fetch_one_hop(session: aiohttp.ClientSession, current_url: str) -> PlainFetchResult | str:
    """One request against ``current_url`` under its host gate: terminal result, or the next URL."""
    async with _host_gate(current_url):
        try:
            async with session.get(current_url, allow_redirects=False) as resp:
                return await _plain_response_outcome(resp, current_url)
        except (TimeoutError, aiohttp.ClientError) as exc:
            return PlainFetchResult(
                status="error",
                method="plain",
                text=f"Fetch error: {type(exc).__name__}: {exc}",
                links=[],
                url=current_url,
            )


async def _fetch_plain(url: str) -> PlainFetchResult:
    if not await resolution_source.is_public_http_url(url):
        return PlainFetchResult(
            status="blocked",
            method="plain",
            text="Blocked non-public or unsupported URL.",
            links=[],
            url=url,
        )
    blocked = _fetch_plain_url_block(url)
    if blocked is not None:
        return blocked

    session = resolution_source._get_session()
    async with session:
        current_url = url
        for _ in range(MAX_REDIRECTS + 1):
            outcome = await _fetch_one_hop(session, current_url)
            if isinstance(outcome, PlainFetchResult):
                return outcome
            current_url = outcome
    return PlainFetchResult(status="error", method="plain", text="Redirect limit exceeded.", links=[], url=url)


def _host_resolver_rule(host: str, ip: str) -> str:
    """Build the Chromium ``--host-resolver-rules`` MAP value pinning ``host`` to ``ip``.

    IPv6 literals must be bracketed in the MAP target per Chromium's rule parser
    (``MAP host [dead::beef]``); IPv4 literals are bare. A malformed ``ip`` is
    passed through unbracketed — callers only ever feed this a value already
    vetted by :func:`_resolve_pinned_host`, so that branch is defensive only.
    """
    try:
        parsed_ip = ipaddress.ip_address(ip)
    except ValueError:
        target = ip
    else:
        target = f"[{ip}]" if parsed_ip.version == 6 else ip
    return f"--host-resolver-rules=MAP {host} {target}"


def _pinnable_url_host(url: str) -> str | None:
    """Hostname of a URL eligible for DNS pinning, or None when the URL itself disqualifies it."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme.lower() not in ("http", "https"):
        return None
    # Userinfo defeats hostname-based trust (`https://trusted@10.0.0.1/`).
    if parsed.username is not None or parsed.password is not None:
        return None
    return parsed.hostname or None


async def _resolve_pinned_host(url: str) -> tuple[str, str] | None:
    """Vet ``url``'s host and resolve it to ONE public IP for Chromium DNS pinning.

    Returns ``(host, vetted_ip)`` — the ``--host-resolver-rules=MAP`` operands — or
    ``None`` when the URL is non-public, unresolvable, or ANY resolved address is
    disallowed. Mirrors :func:`resolution_source.is_public_http_url`'s classification
    (scheme, userinfo, and the shared :func:`resolution_source.resolve_vetted_public_ip`
    predicate) so Chromium can only dial an address the airtight aiohttp
    ``FilteringResolver`` path would also accept.

    This is what closes the DNS-rebinding TOCTOU on the rendered rung: the per-request
    ``_guard_route`` preflight runs its own ``getaddrinfo`` independently of Chromium's
    socket connect, so a rebinding host (TTL 0) can pass the preflight and connect to a
    private IP. Pinning the main-frame host to a single pre-vetted IP removes that
    second resolution entirely — Chromium's connect can only reach the vetted address.

    Fails CLOSED: on any rejection the caller skips Chromium for that host and the
    fetch ladder degrades to plain / read_document.
    """
    host = _pinnable_url_host(url)
    if not host:
        return None

    # IP-literal host: no DNS to rebind. Vet directly and pin to itself.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if resolution_source._ip_is_disallowed(literal):
            return None
        return host, str(literal)

    vetted_ip = await resolution_source.resolve_vetted_public_ip(host)
    if vetted_ip is None:
        return None
    return host, vetted_ip


async def _navigate_and_read_dom(page: Any, url: str, playwright_error: type[BaseException]) -> tuple[str, str]:
    """Navigate to ``url``, let it settle, and return ``(content_type, html)``.

    Two changes from the original ``networkidle`` navigation, both measured 2026-09-03. The
    wait condition is ``domcontentloaded`` plus a fixed settle, because network idle never
    arrives on a page carrying a long-poll widget or an analytics beacon. And a goto failure is
    SALVAGED rather than treated as a dead rung: Playwright's ``TimeoutError`` subclasses
    ``Error``, and a timed-out goto routinely leaves a fully rendered DOM behind — 4 of the
    replay's 10 render rescues came from exactly that (both ballotpedia questions, both
    fts.unocha.org summaries). A genuine navigation error lands here too and salvages an empty
    ``about:blank``, which reaches the ladder as the same "rendered read nothing" as before.

    The worst case is unchanged rather than longer: the goto budget is
    ``_RENDERED_FETCH_TIMEOUT_MS`` MINUS the settle, so goto (33 s) plus settle (2 s) still
    tops out at the same 35 s cap, and on the common path DOM-ready returns far sooner than
    network idle did. ``playwright_error`` is passed in because the class comes from the
    function-scoped optional import in the caller.
    """
    try:
        response = await page.goto(
            url, wait_until="domcontentloaded", timeout=_RENDERED_FETCH_TIMEOUT_MS - _RENDERED_SETTLE_MS
        )
    except playwright_error as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # the salvage above: Playwright's own Error class, passed in from the optional import
        logger.debug("agentic rendered fetch goto failed, salvaging DOM: %s: %s", type(exc).__name__, exc)
        response = None
    await page.wait_for_timeout(_RENDERED_SETTLE_MS)
    content_type = (
        (response.headers.get("content-type") or "").lower()
        if response is not None and hasattr(response, "headers")
        else ""
    )
    return content_type, await page.content()


async def _vet_route(route: Any, request: Any, playwright_error: type[BaseException]) -> None:
    """Let one request Chromium is about to make through, or abort it.

    ``playwright_error`` is passed in because the class comes from the caller's function-scoped
    optional import. A request can still be in flight when the page/context tears down (typically
    after a goto timeout): ``continue_``/``abort`` then races the close and raises
    ``TargetClosedError`` in this detached event-listener task — the unhandled-error storm seen
    2026-07-25. It is swallowed because a closed target has no live socket, so an abort that
    "fails" because the target is already gone still lets nothing through and the SSRF guarantee
    is unaffected; ``unroute_all`` in the caller's ``finally`` is the primary drain and this is the
    residual-race backstop. Only Playwright's own Error is caught, so a genuine bug still
    propagates.
    """
    try:
        if await resolution_source.is_public_http_url(request.url):
            await route.continue_()
        else:
            await route.abort("blockedbyclient")
    except playwright_error as exc:  # the teardown race documented above, not a broad catch
        logger.debug("agentic route guard race during teardown: %s", exc)


def _note_rendered_no_text(url: str) -> None:
    """Record that Chromium rendered ``url`` and there was no text in the result."""
    _RENDERED_NO_TEXT[url] = None
    _RENDERED_NO_TEXT.move_to_end(url)
    while len(_RENDERED_NO_TEXT) > _FETCH_CACHE_MAX_ENTRIES:
        _RENDERED_NO_TEXT.popitem(last=False)


async def _try_rendered_fetch(url: str) -> PlainFetchResult | None:
    if url in _RENDERED_NO_TEXT:
        # A browser already read this URL to nothing in this run. None is the same "the rendered
        # rung gave us nothing" both callers already handle for a missing/failed Chromium.
        logger.debug("agentic rendered fetch skipped (already rendered to nothing): %s", urlparse(url).netloc)
        return None
    try:
        from playwright.async_api import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import
            Error as PlaywrightError,
        )
        from playwright.async_api import async_playwright  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # optional-dep import boundary: playwright missing/broken degrades the rendered rung, never the run
        _warn_playwright_unavailable_once(exc)
        return None

    # Pin Chromium's DNS to a single pre-vetted public IP BEFORE launch. If the
    # host can't be resolved to a public address, fail closed: skip Chromium and
    # let the ladder fall back to plain / read_document (same graceful-failure
    # signal as a playwright-unavailable / render-error return).
    pinned = await _resolve_pinned_host(url)
    if pinned is None:
        logger.warning("agentic rendered fetch skipped (host not pinnable to a public IP): %s", urlparse(url).netloc)
        return None
    host, vetted_ip = pinned

    try:
        async with _host_gate(url), _RENDERED_FETCH_GLOBAL_SEMAPHORE, async_playwright() as playwright:
            # --host-resolver-rules pins the browser's own resolution to the IP we
            # vetted above, so Chromium's socket connect cannot independently
            # re-resolve `host` to a private address (the DNS-rebinding TOCTOU that
            # the per-request preflight in _guard_route alone cannot close). A fresh
            # browser is launched per call, so per-launch host-resolver-rules is clean.
            browser = await playwright.chromium.launch(
                headless=True,
                args=[_host_resolver_rule(host, vetted_ip)],
            )
            context = await browser.new_context(
                user_agent=BROWSER_HEADERS["User-Agent"],
                extra_http_headers={key: value for key, value in BROWSER_HEADERS.items() if key != "User-Agent"},
            )

            # Defense-in-depth on top of the main-frame pin above. The route guard
            # re-checks EVERY request Chromium makes (main-frame goto, server and
            # client-side redirects, subresources) against is_public_http_url.
            # Threat model: these fetches run on GitHub-hosted Azure runners, where a
            # request to a link-local / RFC1918 host (Azure IMDS at 169.254.169.254,
            # localhost services, the internal runner network) would exfiltrate
            # internal content into the research prompt AND the public Metaculus
            # comment. The main-frame host is now pinned, so its rebinding TOCTOU is
            # closed; subresource / redirect hosts remain guarded only by this
            # per-request preflight (whose getaddrinfo resolves independently of
            # Chromium's connect), so their rebinding TOCTOU is a documented residual
            # — a filtering forward proxy would close it, deferred as its own change.
            async def _guard_route(route: Any, request: Any) -> None:
                # A thin closure so the registration keeps Playwright's expected handler shape
                # while the vetting itself stays module-level and directly testable.
                await _vet_route(route, request, PlaywrightError)

            await context.route("**/*", _guard_route)
            page = await context.new_page()
            try:
                content_type, html = await _navigate_and_read_dom(page, url, PlaywrightError)
                if _content_type_is_document(content_type):
                    return _document_needed_result(url, content_type)
                body = html.encode("utf-8", errors="replace")
                extracted = await asyncio.to_thread(resolution_source._extract_main_text, body, url)
                links = _extract_links_from_html(html, url)
                text = (extracted or "").strip()
                if not text:
                    _note_rendered_no_text(url)
                    return PlainFetchResult(status="error", method="rendered", text="", links=links, url=url)
                return PlainFetchResult(
                    status="ok",
                    method="rendered",
                    text=text,
                    links=links,
                    url=url,
                    content_type=content_type or None,
                )
            finally:
                # Drain in-flight route handlers BEFORE teardown. Without this, a
                # request still in flight when we close (common after a goto
                # timeout) fires _guard_route against the closing context and raises
                # TargetClosedError in a detached event listener — the unhandled
                # traceback storm seen 2026-07-25 that buries real fetch failures in
                # the logs. unroute_all(ignoreErrors) removes the handlers and
                # silently swallows any still mid-flight (Playwright's own remedy for
                # this exact message). SSRF is unaffected: the guard already ran for
                # every request dialed while the page was live, and a request racing
                # teardown has no live target to exfiltrate through. Guarded so a
                # teardown-race error here can't skip context/browser close (leak).
                try:
                    await context.unroute_all(behavior="ignoreErrors")
                except PlaywrightError as exc:
                    logger.debug("agentic rendered fetch unroute_all race: %s", exc)
                await context.close()
                await browser.close()
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # top-level soft-fail for the rendered fetch rung: any browser failure falls back to plain / read_document
        _warn_playwright_unavailable_once(exc)
        return None


async def search_news(query: str) -> ToolOutcome:
    client_id = os.getenv(ASKNEWS_CLIENT_ID_ENV)
    secret = os.getenv(ASKNEWS_SECRET_ENV)
    if not client_id or not secret:
        return _format_fetch_error(
            f"AskNews credentials are not configured; set {ASKNEWS_CLIENT_ID_ENV} and {ASKNEWS_SECRET_ENV}.",
            method="news",
        )
    try:
        articles = await _call_asknews_search(query)
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # tool-handler soft-fail boundary: a dead provider becomes a tool result the driver can read, never a loop crash
        return _format_fetch_error(f"AskNews search failed: {type(exc).__name__}: {exc}", method="news")
    return ToolOutcome(content_markdown=_format_asknews_results(articles), method="news")


async def search_web(query: str, end_published_date: str | None = None) -> ToolOutcome:
    if not os.getenv(EXA_API_KEY_ENV):
        return _format_fetch_error(f"Exa API key is not configured; set {EXA_API_KEY_ENV}.", method="search")
    try:
        results = await _call_exa_search(query, end_published_date)
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # tool-handler soft-fail boundary: a dead provider becomes a tool result the driver can read, never a loop crash
        return _format_fetch_error(f"Exa search failed: {type(exc).__name__}: {exc}", method="search")
    return ToolOutcome(content_markdown=_format_exa_results(results), method="search")


def _generic_document_ask(question_topic: str) -> str:
    return f"Extract the main content relevant to: {question_topic}"


def _pdf_local_outcome(url: str, plain: PlainFetchResult, *, start_char: int) -> ToolOutcome:
    """Serve a locally extracted PDF, then hold it for the rest of the run.

    The text goes through the same window/cache path an HTML page does, so ``start_char``
    paginates a 220-page report exactly as it paginates a long article. The parse is also
    re-keyed under the URL the driver asked for: the extraction cached it under the final hop,
    and a later ``read_document`` on the original URL would otherwise refetch and reparse it.
    """
    pdf = local_document.cached_document(plain.url)
    if pdf is not None:
        local_document.cache_document(url, pdf)
    local_document.log_local_document_read(
        url,
        method=local_document.PDF_LOCAL_METHOD,
        chars=len(plain.text),
        pages=None if pdf is None else pdf.pages_read,
        passages=None,
    )
    return _read_content_outcome(url, plain.text, plain.links, method=plain.method, start_char=start_char)


def _held_from_result(url: str, result: PlainFetchResult) -> local_document.HeldDocument:
    """What one ladder rung's result leaves us holding for ``url``.

    A parse from the local PDF rung wins over the flat text, because its page offsets are what
    make a digest's ``[p.N]`` labels exact; a scan is still held (page structure, no text), so
    the caller knows the free route is exhausted rather than untried. Text we do hold is cached
    for the run, so a later paginated ``fetch`` of the same URL is free.
    """
    if result.method == local_document.OVERSIZE_DOCUMENT_METHOD:
        return local_document.HeldDocument(oversize=True)
    pdf = local_document.cached_document(result.url)
    if pdf is not None:
        held = local_document.held_pdf(pdf)
    elif result.status == "ok" and result.method != DOCUMENT_NEEDED_METHOD:
        held = local_document.HeldDocument(text=result.text.strip())
    else:
        # A document_needed result is "ok" and carries a placeholder sentence telling the driver
        # to use read_document. Reading that as the page's text would hand the digest our own
        # instruction to itself, so it holds nothing, exactly like a failed rung.
        return local_document.HeldDocument()
    if held.has_text and matched_throttle_phrase(held.text) is not None:
        # A rate-limit interstitial is not the document (q45191). Hold nothing, so the paid
        # reader — which dials from Gemini's address rather than ours — gets its turn.
        return local_document.HeldDocument()
    if held.has_text:
        _cache_fetch_result(url, held.text, result.links)
    return held


async def _run_local_document_ladder(url: str) -> local_document.HeldDocument:
    """The free rungs ``fetch`` runs, plain then rendered, for a document read.

    Escalation follows ``fetch``'s own rule rather than a looser one: a page whose plain text is
    thin enough to look like a JavaScript shell goes to the browser even though we hold
    something, because digesting 100 chars of navigation chrome would answer the ask out of
    furniture. A parse ends the ladder either way — a scan is as far as the free route reaches,
    and that is worth knowing rather than re-fetching.
    """
    plain = await _fetch_plain(url)
    held = _held_from_result(url, plain)
    if held.oversize or held.pdf is not None or plain.method == DOCUMENT_NEEDED_METHOD:
        # A parse, a refusal, or a document no local rung can read: an image, or a PDF with no
        # text layer (whose parse the cache already holds). A browser reads neither, so the free
        # ladder ends here rather than spending a Chromium launch to learn that again.
        return held
    if held.has_text and not plain.escalate_rendered:
        return held
    if plain.status in ("ok", "empty"):
        rendered = await _try_rendered_fetch(plain.url)
        if rendered is not None:
            rendered_held = _held_from_result(url, rendered)
            if rendered_held.has_text:
                return rendered_held
    return held


async def _acquire_local_document(url: str) -> local_document.HeldDocument:
    """What the free ladder holds for ``url``: something already read this run, or a fresh try.

    Bounded by ``_LOCAL_DOCUMENT_BUDGET_S`` so a slow host cannot spend the paid reader's
    budget as well as its own; on expiry we hold nothing and the reader gets its turn. The
    cancelled work includes at most one in-flight extraction thread, which finishes and drops
    its result (a thread cannot be cancelled) inside its own ``max_seconds``.
    """
    cached_pdf = local_document.cached_document(url)
    if cached_pdf is not None:
        return local_document.held_pdf(cached_pdf)
    cached_text = _FETCH_TEXT_CACHE.get(url)
    if cached_text is not None:
        _FETCH_TEXT_CACHE.move_to_end(url)
        return local_document.HeldDocument(text=cached_text)
    try:
        return await asyncio.wait_for(_run_local_document_ladder(url), timeout=_LOCAL_DOCUMENT_BUDGET_S)
    except TimeoutError:
        logger.info(
            "agentic read_document local acquisition exceeded %.0fs, falling back to the reader: %s",
            _LOCAL_DOCUMENT_BUDGET_S,
            urlparse(url).netloc,
        )
        return local_document.HeldDocument()


async def _local_digest_outcome(url: str, ask: str, held: local_document.HeldDocument) -> ToolOutcome | None:
    """Answer the ask from text we hold, deterministically and for free — or None to pay instead.

    None is returned for the one shape where a digest would answer the ask out of furniture: a
    sub-floor page (under the same ``_FETCH_MIN_CONTENT_CHARS`` the fetch ladder escalates on),
    with no parse behind it, whose digest selected NO passage. That is a JavaScript shell whose
    browser rescue already failed, and digesting its navigation chrome stamped an unread page
    ``fetched`` — the one tier that supersedes the briefing — while the tool description tells
    the driver a zero-passage digest means the document does not discuss the ask (D5: a Manifold
    sidebar carrying five OTHER markets' probabilities came back as the page's content). All
    three conditions are needed: thin-but-real short sources exist and ``fetch`` serves them as
    successes, a held parse is a real local read of a document a browser cannot help with, and a
    matching passage is evidence the text is the page rather than its frame.

    The digest runs off the loop: ``select_passages`` tokenises every window of the whole
    document and holds one Counter per window, which measured a 1,365 ms contiguous stall for
    six concurrent 400-page digests — inside a research phase whose wall discards work that
    already succeeded (F47).
    """
    digest = await asyncio.to_thread(
        local_document.digest_held,
        held,
        ask=ask,
        top_k=DOCUMENT_DIGEST_TOP_K,
        max_chars=_FETCH_WINDOW_CHARS,
        source_url=url,
    )
    if len(held.text) < _FETCH_MIN_CONTENT_CHARS and held.pdf is None and digest.passages == 0:
        return None
    local_document.log_local_document_read(
        url,
        method=local_document.DIGEST_LOCAL_METHOD,
        chars=len(held.text),
        pages=None if held.pdf is None else held.pdf.pages_read,
        passages=digest.passages,
    )
    return ToolOutcome(content_markdown=digest.block, method=local_document.DIGEST_LOCAL_METHOD)


async def fetch(url: str, start_char: int = 0, *, question_topic: str = "") -> ToolOutcome:
    cached = _fetch_from_cache(url, start_char)
    if cached is not None:
        return cached

    plain = await _fetch_plain(url)
    if plain.status == "blocked":
        return ToolOutcome(content_markdown=plain.text, method=plain.method, status="blocked")
    if plain.method == local_document.PDF_LOCAL_METHOD:
        return _pdf_local_outcome(url, plain, start_char=start_char)
    if plain.method == DOCUMENT_NEEDED_METHOD:
        # Auto-escalate to the read_document backend so the driver keeps its "handled
        # automatically" contract without spending a second tool call. What reaches here is an
        # image, or a PDF the local rung already proved has no text layer. Only the PDF is free
        # of a second request: its parse is cached under the URL, whereas an image is classified
        # on Content-Type alone and its body is never downloaded on either pass, so nothing
        # caches it and the ladder would GET it again to re-derive the same verdict.
        # ``ladder_exhausted`` says so directly — the free rungs just ran here.
        return await read_document(plain.url, _generic_document_ask(question_topic), ladder_exhausted=True)
    if plain.status not in ("ok", "empty"):
        return ToolOutcome(content_markdown=plain.text, method=plain.method, status="error")
    if plain.status == "ok" and not plain.escalate_rendered:
        return _read_content_outcome(url, plain.text, plain.links, method=plain.method, start_char=start_char)
    return await _rendered_escalation_outcome(url, plain, start_char=start_char, question_topic=question_topic)


async def _rendered_escalation_outcome(
    url: str, plain: PlainFetchResult, *, start_char: int, question_topic: str
) -> ToolOutcome:
    """The ladder's last rungs: headless Chromium, then whatever the plain rung really read."""
    rendered = await _try_rendered_fetch(plain.url)
    if rendered is not None:
        if rendered.method == DOCUMENT_NEEDED_METHOD:
            return await read_document(rendered.url, _generic_document_ask(question_topic), ladder_exhausted=True)
        if rendered.status == "ok" and rendered.text:
            return _read_content_outcome(url, rendered.text, rendered.links, method="rendered", start_char=start_char)
    # Rendered was unavailable, errored, or itself extracted nothing. Fall back to
    # plain ONLY when the plain fetch actually read (thin-but-real) content; a plain
    # fetch that produced nothing has no content to hand back and must not be
    # laundered as a successful "plain"/"ok" retrieval (the companiesmarketcap.com
    # js-wall failure: an unread page stamped `fetched` and superseded the briefing).
    if plain.status == "ok":
        return _read_content_outcome(url, plain.text, plain.links, method=plain.method, start_char=start_char)
    return _empty_fetch_outcome(plain.url)


_ROBOTS_DISALLOWED_MSG = (
    "Document read not attempted: {host}'s robots.txt disallows Google-Extended, the token "
    "Gemini's url_context reader identifies as, so that read is refused at the host and returns "
    "no content whatever it costs. Nothing from this URL was read; do NOT cite it as a fetched "
    "source, and do NOT read it as evidence the fact is unavailable. Retrying will not help — "
    "look for the same fact on another host."
)


def _robots_host(url: str) -> str:
    """``url``'s netloc with any userinfo dropped: the robots cache key, and what gets logged.

    The port stays, since a host's policy is served per origin. Userinfo goes because it must
    reach neither a robots.txt request nor the archived telemetry line.
    """
    return urlparse(url).netloc.rpartition("@")[2]


async def _robots_txt_for_host(url: str) -> str | None:
    """``url``'s host's robots.txt, fetched at most once per host; None when we could not read it.

    Goes through ``_fetch_plain`` rather than its own client so the SSRF preflight, the
    filtering resolver, the redirect vetting and the body cap all apply unchanged. That path
    also classifies, so a host serving robots.txt as HTML hands back trafilatura's idea of it
    and a non-plain rung (an image, a PDF) is refused outright — both of which read as "no
    directives", i.e. proceed and pay, which is the only direction an unreadable robots.txt is
    allowed to fail in.
    """
    host = _robots_host(url)
    if host in _ROBOTS_TXT_CACHE:
        return _ROBOTS_TXT_CACHE[host]
    body: str | None = None
    try:
        result = await asyncio.wait_for(
            _fetch_plain(f"{urlparse(url).scheme}://{host}/robots.txt"), timeout=_ROBOTS_FETCH_TIMEOUT_S
        )
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # pre-check soft-fail boundary: a robots.txt we cannot read must degrade to paying, never to failing the read
        logger.debug("agentic robots.txt pre-check failed for %s: %s: %s", host, type(exc).__name__, exc)
    else:
        if result.status == "ok" and result.method == "plain":
            body = result.text
    _ROBOTS_TXT_CACHE[host] = body
    _ROBOTS_TXT_CACHE.move_to_end(host)
    while len(_ROBOTS_TXT_CACHE) > _FETCH_CACHE_MAX_ENTRIES:
        _ROBOTS_TXT_CACHE.popitem(last=False)
    return body


async def _url_context_robots_skip(url: str) -> bool:
    """True when this host tells ``Google-Extended`` to stay out of ``url``'s path.

    Only the paid ``url_context`` rung consults this: the free rungs dial from our own client
    under our own user agent, and this bot's reading of ``Content-Signal: use=reference`` is
    that reference use is permitted. Proven live 2026-09-03 — see ``robots_policy``.
    """
    robots_txt = await _robots_txt_for_host(url)
    if robots_txt is None:
        return False
    return google_extended_disallows(robots_txt, urlparse(url).path)


async def read_document(url: str, ask: str, *, ladder_exhausted: bool = False) -> ToolOutcome:
    """Answer ``ask`` about ``url``: from the page's own text where we can get it, else Gemini.

    Acquisition-first. The free ladder runs before anything is spent (this run's cache, then
    the plain and rendered rungs ``fetch`` uses), and any text it holds is answered with a
    deterministic BM25 passage digest — ``method="digest_local"``. The paid ``url_context``
    read happens only when the ladder holds nothing: a host that refuses us, a page with no
    text at all, or a PDF with no text layer. Measured 2026-09-03, that is two of 47 archived
    fetch failures, against 191 reader calls over the 2026 summer season.

    The retrieval-count guard on the paid rung stays exactly as it was, because it is what
    keeps that rung honest: ``method="document"`` maps to the ``fetched`` tier
    (``provenance._METHOD_TO_TIER``), the highest authority the artifact renderer has — only a
    ``fetched`` discrepancy enters the SUPERSEDE block that tells every forecaster to override
    the briefing. A fluent-but-ungrounded answer there is the Q38195 failure mode with a bigger
    blast radius, and the quote check cannot catch it (WARN-only for this tool, since
    paraphrase and ellipsis-joined quotes make a hard gate too false-positive-prone). So zero
    successful url_context retrievals withholds the tier, mirroring the grounded-chunk floor
    ``gemini_search`` applies. The local methods earn the same tier for the opposite reason:
    the bytes are the host's own, decoded rather than described.

    ``ladder_exhausted`` is internal and hidden from the driver-facing schema (the same way
    ``fetch`` hides ``question_topic``): ``fetch``'s own escalations set it because the free
    rungs just ran for that URL, and running them again would re-request an image the plain rung
    classified from its Content-Type without ever downloading it.
    """
    started = monotonic()
    held = local_document.HeldDocument() if ladder_exhausted else await _acquire_local_document(url)
    if held.oversize:
        return _format_fetch_error(local_document.oversize_message(url), method=local_document.OVERSIZE_DOCUMENT_METHOD)
    if held.has_text or local_document.exceeds_url_context_size_gate(held.text):
        # The size gate rides the same branch as the text it guards, so the two can never
        # disagree: a document we hold is served from the digest whatever its size, and the
        # biggest are the clearest case — the nine archived documents past the gate carried 67%
        # of the season's reader tokens and the largest of them returned nothing for the money.
        # A None here is the one shape that must not be served: sub-floor chrome that no passage
        # matched, which the paid reader below is the right rung for (see _local_digest_outcome).
        served = await _local_digest_outcome(url, ask, held)
        if served is not None:
            return served
    if not os.getenv(GOOGLE_API_KEY_ENV):
        return _format_fetch_error(f"Google API key is not configured; set {GOOGLE_API_KEY_ENV}.", method="document")
    if await _url_context_robots_skip(url):
        # Its own status token, never mapped to a tier: nothing was read, and the reason is the
        # host's policy rather than a failure worth retrying.
        logger.info(f"AGENTIC_URLCONTEXT_ROBOTS_SKIP: url={url} host={_robots_host(url)}")
        return _format_fetch_error(
            _ROBOTS_DISALLOWED_MSG.format(host=_robots_host(url)),
            status="robots_disallowed",
            method="document",
        )
    try:
        # The reader gets what the total budget has left, so a long acquisition shortens the
        # paid attempt instead of overrunning the tool's ceiling. Acquisition is itself capped at
        # _LOCAL_DOCUMENT_BUDGET_S, so this wait is 40 s at that cap and 60 s when acquisition
        # failed fast (the common case). The reader's own in-thread ceiling is FIXED at 55 s
        # (tool_backends: 2 x 26.5 s + 2 s of backoff), so past ~10 s of acquisition this wait is
        # the shorter of the two and a to_thread worker — which wait_for cannot cancel — can
        # outlive it by up to 15 s, finishing a call whose answer is discarded. What that cannot
        # do is start a NEW billed request after we stop waiting: the last attempt begins by
        # 28.5 s in, well inside the 40 s floor. Sizing the attempts off this variable wait
        # instead would cut one attempt to 19 s on the handover path and fail reads that succeed
        # today, so the arithmetic stays fixed and the overrun is documented rather than traded.
        text, n_url_success, statuses = await asyncio.wait_for(
            asyncio.to_thread(_run_document_read_sync, url, ask),
            timeout=min(_READ_DOCUMENT_TIMEOUT_S, _READ_DOCUMENT_TOTAL_BUDGET_S - (monotonic() - started)),
        )
    except TimeoutError:
        return _format_fetch_error("Document read timed out.", method="document")
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # tool-handler soft-fail boundary: a dead reader becomes a tool result the driver can read, never a loop crash
        return _format_fetch_error(f"Document read failed: {type(exc).__name__}: {exc}", method="document")
    if n_url_success == 0:
        # Greppable, mirroring gemini_search's GEMINI_UNGROUNDED_SUPPRESSED so the rate is
        # measurable from the archived run logs. ``statuses`` carries every reported
        # url_retrieval_status: a refused fetch, a retrieval timeout and a url_context tool
        # that never ran all read as zero successes, and only the status names separate them.
        # ``none`` means the SDK attached no url_metadata entry at all.
        logger.warning(f"AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED: url={url} statuses={','.join(statuses) or 'none'}")
        return _format_fetch_error(
            f"Document read retrieved no URL content: Gemini's url_context tool fetched nothing from {url}, "
            "so any answer would be unsourced recall rather than a read of the document.",
            method="document",
        )
    return ToolOutcome(content_markdown=text, method="document")


def build_gap_fill_tools(question_topic: str) -> list[ToolSpec]:
    async def _fetch_with_topic(url: str, start_char: int = 0) -> ToolOutcome:
        # Binds the question topic for rung-3 document auto-escalation; the
        # driver-facing schema stays (url, start_char) only.
        return await fetch(url, start_char, question_topic=question_topic)

    async def _read_document_public(url: str, ask: str) -> ToolOutcome:
        # (url, ask) only, for the same reason _fetch_with_topic hides question_topic: the loop
        # binds handlers with **arguments straight off the model, so an advertised — or merely
        # hallucinated — `ladder_exhausted: true` would skip the free ladder and pay. Resolves
        # `read_document` as a module attribute at call time, so the suite's patches still land.
        return await read_document(url, ask)

    return [
        ToolSpec(
            name="search_news",
            description=SEARCH_NEWS_DESCRIPTION,
            parameters=_SEARCH_NEWS_PARAMETERS,
            handler=search_news,
            timeout_s=90,
        ),
        ToolSpec(
            name="search_web",
            description=SEARCH_WEB_DESCRIPTION,
            parameters=_SEARCH_WEB_PARAMETERS,
            handler=search_web,
            timeout_s=20,
        ),
        ToolSpec(
            name="fetch",
            description=FETCH_DESCRIPTION,
            parameters=_FETCH_PARAMETERS,
            handler=_fetch_with_topic,
            # Sits above _READ_DOCUMENT_TIMEOUT_S so the rung-3 document
            # auto-escalation has room to fire and return inside this budget.
            timeout_s=90,
        ),
        ToolSpec(
            name="read_document",
            description=READ_DOCUMENT_DESCRIPTION,
            parameters=_READ_DOCUMENT_PARAMETERS,
            handler=_read_document_public,
            # UNCHANGED at 70 even though the handler now runs a free local ladder before the
            # paid read: the two share _READ_DOCUMENT_TOTAL_BUDGET_S (65) and 70 stays at
            # GAP_FILL_V2_CONCLUDE_THRESHOLD, so the loop's wall discipline is untouched.
            timeout_s=70,
        ),
    ]
