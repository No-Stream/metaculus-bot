"""The four tools the gap-fill v2 driver calls, and the fetch ladder behind ``fetch``.

This module owns the handlers (``search_news``, ``search_web``, ``fetch``,
``read_document``), the ladder spine they run on — the plain hop loop with its redirect
vetting, the mapping of a rendered page onto this ladder's result type, the window cache
that serves ``start_char`` continuations — and ``build_gap_fill_tools``, whose list order is
the order the driver sees.

Support pieces live next door: ``tool_descriptions`` (driver-facing text + JSON schemas),
``tool_backends`` (the AskNews / Exa / Gemini calls and their result formatting),
``fetch_outcomes`` (classifying one plain HTTP response), ``local_document`` (the local PDF
rung, the run's document cache, and the passage digest ``read_document`` serves). The
headless-Chromium TRANSPORT moved out one level, to ``research.rendered_fetch``, when the
Tier-1 resolution-source fetcher gained the same rung — the DNS pin, the route guard and the
process-global launch cap are all load-bearing and a second copy of any of them would drift.
``_try_rendered_fetch`` stays here because what a rendered page MEANS is this ladder's
business: extracted text plus outbound links for a driver model. The TLS-impersonating retry
(``research.impersonated_fetch``, added 2026-09-04) follows the same split: the transport and
its policy (the trigger set, the kill switch, the per-run host memo) are shared with Tier 1, and
``_try_impersonated_fetch`` here maps its response onto this ladder, from both free ladders that
run a plain GET (``fetch`` and ``read_document``'s local-document acquisition).

The seams the suite monkeypatches
— ``_read_response_body``, ``_fetch_plain``, ``_try_rendered_fetch``,
``_try_impersonated_fetch``, ``fetch_impersonated``,
``_acquire_local_document``, ``_run_document_read_sync``, ``read_document``,
``_READ_DOCUMENT_TIMEOUT_S`` — are attributes of THIS module and are resolved here at call
time, so their callers stay here even where the callee moved out. The render transport's own
seams (``resolve_pinned_host``, the launch semaphore) are attributes of ``rendered_fetch``
and are patched there.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from time import monotonic
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
    RESOLUTION_SOURCE_HTTP_TIMEOUT,
    RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
)
from metaculus_bot.research import impersonated_fetch, resolution_source
from metaculus_bot.research.agentic import local_document
from metaculus_bot.research.agentic.fetch_outcomes import (
    _FETCH_MIN_CONTENT_CHARS,
    _HTML_CONTENT_TYPE_TOKENS,
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
    MAX_REDIRECTS,
    REDIRECT_STATUSES,
    decode_text_body,
    read_body_capped,
)
from metaculus_bot.research.impersonated_fetch import (
    ImpersonateDeclined,
    fetch_impersonated,
)
from metaculus_bot.research.rendered_fetch import (
    MemoScope,
    RenderDomOverCeiling,
    RenderOffHost,
    note_rendered_no_text,
    render_page,
)
from metaculus_bot.research.robots_policy import ROBOTS_FETCH_TIMEOUT_S, google_extended_blocks_url, robots_host

logger = logging.getLogger(__name__)

# This ladder's key into the transport's render memos: its "rendered to nothing" is bare
# trafilatura emptiness, a weaker test than Tier-1's, so the two must not answer each other.
_RENDER_MEMO_SCOPE: MemoScope = "gap_fill_v2"

_FETCH_WINDOW_CHARS = 8000
_FETCH_CACHE_MAX_ENTRIES = 50
_READ_DOCUMENT_TIMEOUT_S = 60.0
# read_document is acquisition-first, so its budget holds two rungs: the free local ladder,
# then the paid reader on what the total leaves it. The ToolSpec ceiling stays 70 s (see
# build_gap_fill_tools) and so does the loop's wall discipline: 25 + 40 = 65, plus 5 s of
# margin. The paid reader keeps its whole 60 s whenever acquisition failed fast, which is the
# common case for the URLs that reach it (archived all-fail p50 0.3 s).
_LOCAL_DOCUMENT_BUDGET_S = 25.0
_READ_DOCUMENT_TOTAL_BUDGET_S = 65.0
_FETCH_HOST_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
_FETCH_TEXT_CACHE: OrderedDict[str, str] = OrderedDict()
_FETCH_LINKS_CACHE: OrderedDict[str, list[str]] = OrderedDict()


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
        # buy nothing: this exit skips the READ, and `_plain_body_outcome` reaches the same verdict
        # for a body another transport already holds. A PDF does not take this exit: its bytes are
        # exactly what the local rung needs.
        return _document_needed_result(current_url, content_type)
    declared_pdf = _content_type_is_pdf(content_type)
    body = await _read_response_body(
        resp,
        f"agentic fetch {urlparse(current_url).netloc}",
        max_bytes=DOCUMENT_TEXT_PDF_MAX_BYTES if declared_pdf else RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
    )
    if body is None:
        return _body_too_large_result(current_url, content_type, declared_pdf=declared_pdf)
    return await _plain_body_outcome(body, content_type, current_url)


async def _plain_body_outcome(body: bytes, content_type: str, current_url: str) -> PlainFetchResult:
    """Classify a body this ladder already holds, whichever transport read it.

    The bytes-level tail of :func:`_plain_response_outcome`, split from the read so the
    impersonated retry (:func:`_try_impersonated_fetch`) gets this ladder's FULL classification,
    the local PDF rung and the document escalation included, rather than a second partial copy.
    The declared-image rule lives here as well as in the wrapper's pre-read exit, because the
    wrapper's copy is a read-avoidance shortcut and this is the classification: a declared
    ``image/webp`` or ``image/svg+xml`` has no magic bytes the sniff below knows, and without the
    header clause the impersonated path reported it as an unsupported type where the aiohttp path
    escalated it to ``read_document``.
    """
    if _content_type_is_pdf(content_type) or is_pdf_body(body):
        # Local extraction first, whether the header said PDF or only the magic bytes did.
        return await local_document.pdf_fetch_result(body, url=current_url, content_type=content_type)
    if _content_type_is_image(content_type) or _body_is_document(body):
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


async def _try_rendered_fetch(url: str) -> PlainFetchResult | None:
    """Render ``url`` in headless Chromium and read it as this ladder does a plain page.

    The transport is ``research.rendered_fetch`` (shared with the Tier-1 resolution-source
    rung); what stays here is the MAPPING onto ``PlainFetchResult``, which is this ladder's
    own contract: a document content type re-enters the ``read_document`` escalation, an
    extraction with text is an ``ok`` page for the driver, and an extraction with none is an
    ``error`` whose method is still ``rendered`` — never ``ok``, because the loop grants the
    ``fetched`` verification tier on status alone.

    A ``None`` from the transport (Playwright missing, host not pinnable, browser error, or a
    URL a browser already read to nothing this run under THIS ladder's memo scope) is returned
    unchanged: it is the graceful-failure signal both call sites already degrade on. A render the
    transport CUT OFF (its DOM-read cap fired because the page kept navigating) raises
    ``TimeoutError`` instead, a rendered DOM over ``RENDERED_DOM_MAX_CHARS`` raises
    ``RenderDomOverCeiling``, and a main frame that landed on a host other than the pinned one
    raises ``RenderOffHost`` with its DOM refused unread on the transport's pre-read check, or
    discarded unpublished when the navigation commits during the read itself, so the Tier-1 rung
    can record each under its own reason; this ladder's callers only know ``None``, so all three
    are folded back into that signal here, and nothing from an off-host render reaches the driver.
    The transport memoises the cut-off itself and re-raises it on the next fetch of the same URL,
    so a second fetch of the same hostile page in this run does not pay for it again, and a failed
    navigation that landed on Chromium's own error document is memoised the same way; the oversized
    DOM and a genuine off-host landing are memoised by nobody, since the page did render. The
    ceilings this wrapper already runs under are unchanged: the ``fetch`` tool's ``timeout_s`` and
    ``_LOCAL_DOCUMENT_BUDGET_S`` on the document ladder.

    The URL handed to the browser is the plain rung's ``url``, which is the last hop of its own
    redirect loop, so the pin already covers the host that serves the content.

    The memo scope is this ladder's own because "rendered to nothing" means something weaker
    here than in Tier-1: bare trafilatura emptiness, where Tier-1 also tries the ARIA rewrite,
    the inline-chart read and the harvested feed before it gives up on a URL.
    """
    try:
        page = await render_page(url, memo_scope=_RENDER_MEMO_SCOPE, host_gate=_host_gate(url))
    except (TimeoutError, RenderDomOverCeiling, RenderOffHost):
        return None
    if page is None:
        return None
    if _content_type_is_document(page.content_type):
        return _document_needed_result(url, page.content_type)
    body = page.html.encode("utf-8", errors="replace")
    extracted = await asyncio.to_thread(resolution_source._extract_main_text, body, url)
    # Links resolve against the document the DOM came from, which after a same-host client-side
    # redirect is not the URL asked for; the memo key and the result's `url` stay the requested URL.
    links = _extract_links_from_html(page.html, page.document_url)
    text = (extracted or "").strip()
    if not text:
        note_rendered_no_text(url, memo_scope=_RENDER_MEMO_SCOPE)
        return PlainFetchResult(status="error", method="rendered", text="", links=links, url=url)
    return PlainFetchResult(
        status="ok",
        method="rendered",
        text=text,
        links=links,
        url=url,
        content_type=page.content_type or None,
    )


async def _try_impersonated_fetch(url: str, *, deadline_monotonic_s: float | None = None) -> PlainFetchResult | None:
    """Re-dial a page the plain rung was answered 403, presenting a real browser's fingerprint.

    The transport is ``research.impersonated_fetch``, shared with the Tier-1 resolution-source
    rung, which is where the measurement behind it lives (2026-09-04, from a GitHub Actions
    runner: four Akamai-fronted federal hosts answered the bot's aiohttp client 403 and the same
    GET under Chrome impersonation 200). What stays here is the MAPPING onto ``PlainFetchResult``,
    this ladder's own contract. A 200 goes through :func:`_plain_body_outcome`, the same
    classification a plain body gets, with ``method="impersonate"`` stamped on the plain-shaped
    results so the loop's tier map (``provenance._METHOD_TO_TIER``) grants ``fetched``; a document
    keeps the method its own rung stamps (``pdf_local``, ``document_needed``), which the ``fetch``
    handler keys on.

    Every decline folds back into ``None``, because this ladder's callers only know ``None``: the
    kill switch (``impersonated_fetch.impersonation_enabled``, the transport's reading of
    ``RESOLUTION_SOURCE_IMPERSONATE_ENABLED``, on by default in code), the per-run host memo
    shared with Tier 1 (a host that refused the impersonated client once this run will not answer
    the next URL on it differently), every :class:`ImpersonateDeclined` (a host that will not pin,
    a refused hop, an oversized body, a transport failure), and a non-200 answer, which the
    transport's ``note_refusal_if_block_shaped`` memoizes when it is block-shaped, for the host
    that answered and for the exact URL dialed. The direct ``blocked`` result then stands, byte
    for byte what it was before the rung existed.

    ``url`` is the plain rung's ``url``, the last hop of its own guarded redirect loop, the same
    choice :func:`_try_rendered_fetch` documents; the trigger (a host's 403) is the caller's test,
    in :func:`_fetch_plain_with_impersonated_retry`. The wall is one plain hop's worth
    (``RESOLUTION_SOURCE_HTTP_TIMEOUT``, the timeout the plain rung's session already runs under)
    for the whole retry, redirect hops included, so the retry costs the ``fetch`` tool's ceiling at
    most what one more plain hop would have. Under ``read_document``'s acquisition ladder that wall
    outlives the caller: the ladder is capped at ``_LOCAL_DOCUMENT_BUDGET_S`` by a ``wait_for``
    that would cancel the dial mid-transfer, so the ladder passes its own ``deadline_monotonic_s``,
    the dial is sized to the earlier of the two, and with less than
    ``RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S`` left (the floor Tier 1's rung claims) the retry
    declines without dialing rather than spend the paid reader's turn on a request it cannot
    finish. Strictly safer: a deadline only ever shortens or skips a dial. The host gate is this
    ladder's own map. The two body caps are the plain rung's own:
    ``RESOLUTION_SOURCE_MAX_RESPONSE_BYTES`` for a page and ``DOCUMENT_TEXT_PDF_MAX_BYTES`` for a
    declared PDF, the pair :func:`_plain_response_outcome` reads under, so a PDF between the two is
    read here as the plain rung would have read it.
    """
    if not impersonated_fetch.impersonation_enabled():
        return None
    if impersonated_fetch.impersonation_refused(url):
        return None
    netloc = urlparse(url).netloc
    wall_deadline_s = monotonic() + RESOLUTION_SOURCE_HTTP_TIMEOUT
    if deadline_monotonic_s is not None:
        remaining_s = deadline_monotonic_s - monotonic()
        if remaining_s < RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S:
            logger.info(
                "agentic fetch: skipping the impersonated retry of %s, %.1fs of the ladder's budget left",
                netloc,
                remaining_s,
            )
            return None
        wall_deadline_s = min(wall_deadline_s, deadline_monotonic_s)
    try:
        response = await fetch_impersonated(
            url,
            host_sems=_FETCH_HOST_SEMAPHORES,
            deadline_monotonic_s=wall_deadline_s,
            per_hop_timeout_s=RESOLUTION_SOURCE_HTTP_TIMEOUT,
            max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
            document_max_bytes=DOCUMENT_TEXT_PDF_MAX_BYTES,
        )
    except ImpersonateDeclined as exc:
        logger.info(
            "agentic fetch: the impersonated retry of %s produced nothing (%s: %s)", netloc, type(exc).__name__, exc
        )
        return None
    if response.status != 200:
        impersonated_fetch.note_refusal_if_block_shaped(
            dialed_url=url, answered_url=response.url, status=response.status
        )
        logger.info(
            "agentic fetch: the impersonated retry of %s was answered %d by %s",
            netloc,
            response.status,
            urlparse(response.url).netloc,
        )
        return None
    result = await _plain_body_outcome(response.body, response.content_type, response.url)
    if result.method == "plain":
        result.method = "impersonate"
    return result


async def _fetch_plain_with_impersonated_retry(
    url: str, *, deadline_monotonic_s: float | None = None
) -> PlainFetchResult:
    """The plain rung, plus the one free retry a host's 403 earns.

    The one trigger both fetchers share, read off the transport at call time so the population
    cannot drift between them (``impersonated_fetch.IMPERSONATE_TRIGGER_STATUSES``): a host's 403,
    never the ``blocked`` this ladder produces itself for a non-public URL or a Metaculus
    self-reference, both of which carry no ``http_status``. Shared by ``fetch`` and by
    ``read_document``'s local-document ladder, because the latter sits immediately in front of
    the paid reader and a cold ``read_document`` on a 403 host would otherwise pay for bytes the
    free retry fetches. A rescue replaces the plain result; every decline leaves it as it was.
    ``deadline_monotonic_s`` is the caller's own budget, handed to the retry so it never dials a
    wall its caller would cancel; ``fetch`` has none and passes nothing.
    """
    plain = await _fetch_plain(url)
    if plain.status == "blocked" and plain.http_status in impersonated_fetch.IMPERSONATE_TRIGGER_STATUSES:
        impersonated = await _try_impersonated_fetch(plain.url, deadline_monotonic_s=deadline_monotonic_s)
        if impersonated is not None:
            return impersonated
    return plain


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


async def _run_local_document_ladder(url: str, *, deadline_monotonic_s: float) -> local_document.HeldDocument:
    """The free rungs ``fetch`` runs, for a document read: plain, the impersonated retry, then rendered.

    Escalation follows ``fetch``'s own rule rather than a looser one: a page whose plain text is
    thin enough to look like a JavaScript shell goes to the browser even though we hold
    something, because digesting 100 chars of navigation chrome would answer the ask out of
    furniture. A parse ends the ladder either way — a scan is as far as the free route reaches,
    and that is worth knowing rather than re-fetching. The impersonated retry runs here for the
    same reason it runs in ``fetch``, and with more at stake: this ladder is the one in front of
    the paid ``url_context`` read, so a 403 it left standing was a paid read of a page the retry
    fetches for free (a bls.gov PDF is one of the four measured rescues). ``deadline_monotonic_s``
    is the instant :func:`_acquire_local_document`'s ``wait_for`` fires, handed to the retry so it
    sizes its dial to what is left instead of to a fresh wall the cancellation would cut short.
    """
    plain = await _fetch_plain_with_impersonated_retry(url, deadline_monotonic_s=deadline_monotonic_s)
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
    budget as well as its own; on expiry we hold nothing and the reader gets its turn. The same
    instant is handed to the ladder as its deadline, so the impersonated retry sizes its dial to
    what is left rather than to a wall this cancellation would cut short. The cancelled work
    includes at most one in-flight extraction thread, which finishes and drops its result,
    because a thread cannot be cancelled.

    That thread does NOT finish inside its own ``max_seconds``. The clock for that budget starts
    only once ``extract_pdf_text`` has read the declared page count and the whole bookmark
    outline, and the page in flight always completes, so the worker outlives this cancellation by
    that un-clocked prologue plus one page. It also hands its slot in the shared two-slot parse
    gate back as it unwinds, so a fresh parse can start alongside the abandoned one. Recorded in
    FUTURE.md under "The PDF parse overruns ``max_seconds``".
    """
    cached_pdf = local_document.cached_document(url)
    if cached_pdf is not None:
        return local_document.held_pdf(cached_pdf)
    cached_text = _FETCH_TEXT_CACHE.get(url)
    if cached_text is not None:
        _FETCH_TEXT_CACHE.move_to_end(url)
        return local_document.HeldDocument(text=cached_text)
    deadline_monotonic_s = monotonic() + _LOCAL_DOCUMENT_BUDGET_S
    try:
        return await asyncio.wait_for(
            _run_local_document_ladder(url, deadline_monotonic_s=deadline_monotonic_s),
            timeout=_LOCAL_DOCUMENT_BUDGET_S,
        )
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

    plain = await _fetch_plain_with_impersonated_retry(url)
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


async def _fetch_robots_txt(robots_url: str) -> str | None:
    """Read one robots.txt through THIS path's ladder; None when we could not read it.

    Goes through ``_fetch_plain`` rather than its own client so the SSRF preflight, the
    filtering resolver, the redirect vetting and the body cap all apply unchanged. That path
    also classifies, so a host serving robots.txt as HTML hands back trafilatura's idea of it
    and a non-plain rung (an image, a PDF) is refused outright — both of which read as "no
    directives", i.e. proceed and pay, which is the only direction an unreadable robots.txt is
    allowed to fail in. The bound and the per-host cache are ``robots_policy``'s, shared with
    the Tier-1 resolution-source reader, because a host's policy is a property of the host and
    the two paths routinely reach the same government domains in one run.
    """
    try:
        result = await asyncio.wait_for(_fetch_plain(robots_url), timeout=ROBOTS_FETCH_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # pre-check soft-fail boundary: a robots.txt we cannot read must degrade to paying, never to failing the read
        logger.debug("agentic robots.txt pre-check failed for %s: %s: %s", robots_url, type(exc).__name__, exc)
        return None
    if result.status == "ok" and result.method == "plain":
        return result.text
    return None


async def _url_context_robots_skip(url: str) -> bool:
    """True when this host tells ``Google-Extended`` to stay out of ``url``'s path.

    Only the paid ``url_context`` rung consults this: the free rungs dial from our own client
    under our own user agent, and this bot's reading of ``Content-Signal: use=reference`` is
    that reference use is permitted. Proven live 2026-09-03 — see ``robots_policy``, which owns
    the per-host cache this shares with the Tier-1 reader.
    """
    return await google_extended_blocks_url(url, fetch_text=_fetch_robots_txt)


async def read_document(url: str, ask: str, *, ladder_exhausted: bool = False) -> ToolOutcome:
    """Answer ``ask`` about ``url``: from the page's own text where we can get it, else Gemini.

    Acquisition-first. The free ladder runs before anything is spent (this run's cache, then
    the plain, impersonated-retry and rendered rungs ``fetch`` uses), and any text it holds is
    answered with a deterministic BM25 passage digest — ``method="digest_local"``. The paid
    ``url_context`` read happens only when the ladder holds nothing: a host that refuses us, a
    page with no text at all, or a PDF with no text layer. Measured 2026-09-03, that is two of 47
    archived fetch failures, against 191 reader calls over the 2026 summer season.

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
        logger.info(f"AGENTIC_URLCONTEXT_ROBOTS_SKIP: url={url} host={robots_host(url)}")
        return _format_fetch_error(
            _ROBOTS_DISALLOWED_MSG.format(host=robots_host(url)),
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
