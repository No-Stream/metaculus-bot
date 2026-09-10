"""The four tools the gap-fill v2 driver calls, and this caller's half of the fetch ladder.

The handlers are here (``search_news``, ``search_web``, ``fetch``, ``read_document``) and so is
``build_gap_fill_tools``, whose list order is the order the driver sees. What used to be here and is
not any more is the ladder itself: ``fetch`` and ``read_document``'s free acquisition both run the
SHARED ladder through ``_fetch_via_ladder``, which calls ``fetch_ladder.ladder.fetch_url`` under one
of the three gap-fill presets and maps what comes back through ``ladder_adapter``. The loop's own
former rungs (``_fetch_plain`` and friends) are unreferenced and are deleted with the rest of the
duplication.

What stays this caller's: the window cache that serves ``start_char`` continuations, the
question-platform refusal that runs before anything is dialed, the throttle-phrase check on a body
the ladder read, the auto-escalation to ``read_document``, the paid reader and its robots pre-check,
and the two shared fetch markers emitted after each tool call. Support pieces live next door:
``tool_descriptions`` (driver-facing text and JSON schemas), ``tool_backends`` (the AskNews, Exa and
Gemini calls), ``fetch_outcomes`` (this ladder's result type and its refusals), ``local_document``
(what the free ladder holds, and the digest ``read_document`` serves), ``ladder_adapter`` (the two
vocabularies' one meeting point).

The seams the suite monkeypatches — ``_fetch_via_ladder``, ``_acquire_local_document``,
``_run_document_read_sync``, ``read_document``, ``_READ_DOCUMENT_TIMEOUT_S`` — are attributes of THIS
module and resolved here at call time. The ladder's own seams (``direct_fetch._fetch_direct``,
``rungs._rendered_rung``, ``rungs.render_page``, ``rungs.fetch_impersonated``) are patched there.
Detail: ``docs/agentic_gap_fill.md`` "The fetch ladder".
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from dataclasses import replace
from datetime import UTC, datetime
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
    RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS,
)
from metaculus_bot.research import derived_api, document_cache, fetch_markers, impersonated_fetch
from metaculus_bot.research.agentic import ladder_adapter, local_document
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
from metaculus_bot.research.fetch_ladder import classify, guard
from metaculus_bot.research.fetch_ladder.context import LadderContext, QuestionRungBudget
from metaculus_bot.research.fetch_ladder.ladder import fetch_url
from metaculus_bot.research.fetch_ladder.policy import (
    GAP_FILL_DIRECT_POLICY,
    GAP_FILL_DOCUMENT_POLICY,
    GAP_FILL_FETCH_POLICY,
    LADDER_CALLER_GAP_FILL_V2,
    LadderPolicy,
)
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
    RenderedPage,
    RenderOffHost,
    note_rendered_no_text,
    render_page,
)
from metaculus_bot.research.resolution_fetch_result import FetchResult
from metaculus_bot.research.robots_policy import ROBOTS_FETCH_TIMEOUT_S, google_extended_blocks_url, robots_host
from metaculus_bot.research.wayback import (
    innermost_url,
    parse_snapshot_url,
    snapshot_age_days,
    wayback_lead,
    wayback_snapshot_url,
)

logger = logging.getLogger(__name__)

# The dead `_try_rendered_fetch`'s memo key; the live scope is `LadderPolicy.render_memo_scope`.
_RENDER_MEMO_SCOPE: MemoScope = "gap_fill_v2"

_FETCH_WINDOW_CHARS = 8000
_FETCH_CACHE_MAX_ENTRIES = 50
_READ_DOCUMENT_TIMEOUT_S = 60.0
# Two rungs share this: the free local ladder, then the paid reader on what the total leaves it.
_LOCAL_DOCUMENT_BUDGET_S = 25.0
_READ_DOCUMENT_TOTAL_BUDGET_S = 65.0
_FETCH_HOST_SEMAPHORES: dict[str, asyncio.Semaphore] = {}
_FETCH_TEXT_CACHE: OrderedDict[str, str] = OrderedDict()
_FETCH_LINKS_CACHE: OrderedDict[str, list[str]] = OrderedDict()


def _host_gate(url: str) -> asyncio.Semaphore:
    return guard._sem_for_host(_FETCH_HOST_SEMAPHORES, url)


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


async def _plain_body_outcome(body: bytes, content_type: str, current_url: str) -> PlainFetchResult | str:
    """Classify a body this ladder already holds, whichever transport read it.

    The bytes-level tail of :func:`_plain_response_outcome`, split from the read so the
    impersonated retry (:func:`_try_impersonated_fetch`) gets this ladder's FULL classification,
    the local PDF rung and the document escalation included, rather than a second partial copy.
    The declared-image rule lives here as well as in the wrapper's pre-read exit, because the
    wrapper's copy is a read-avoidance shortcut and this is the classification: a declared
    ``image/webp`` or ``image/svg+xml`` has no magic bytes the sniff below knows, and without the
    header clause the impersonated path reported it as an unsupported type where the aiohttp path
    escalated it to ``read_document``.

    Returns a ``str`` on one shape only: an HTML body whose sole content is a meta-refresh stub,
    the vetted next URL the redirect loop follows (:func:`fetch_outcomes._plain_html_outcome`).
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
        return await _plain_html_outcome(body, html, content_type, current_url, undecodable_ratio=undecodable_ratio)
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
    if not await guard.is_public_http_url(url):
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

    session = guard._get_session()
    async with session:
        current_url = url
        for _ in range(MAX_REDIRECTS + 1):
            outcome = await _fetch_one_hop(session, current_url)
            if isinstance(outcome, PlainFetchResult):
                return outcome
            current_url = outcome
    return PlainFetchResult(status="error", method="plain", text="Redirect limit exceeded.", links=[], url=url)


def _derived_api_outcome(url: str, page: RenderedPage) -> PlainFetchResult | None:
    """Serve the largest JSON feed a fruitless render already harvested, or None to stay empty.

    A JavaScript dashboard whose DOM has no text after render usually loaded its figures over XHR,
    which the render captured (``derived_api.largest_json``). Served directly with
    ``derived_api_lead`` for provenance and ``method=derived_api`` (fetched tier), it is the page's
    own feed. Only same-publisher bodies are admissible, about one of six measured dashboards
    (2026-09-04 QA), so a modest rescue against an object the render already holds.
    """
    largest = derived_api.largest_json(page.json_responses)
    if largest is None:
        return None
    endpoint = derived_api.DerivedEndpoint(endpoint_url=largest.url, discovered_on=url)
    lead = derived_api.derived_api_lead(endpoint, url)
    body_text = decode_text_body(largest.body, "application/json")[0]
    return PlainFetchResult(
        status="ok",
        method="derived_api",
        text=f"{lead}\n\n{body_text}",
        links=[],
        url=url,
        content_type="application/json",
    )


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
    extracted = await asyncio.to_thread(classify._extract_main_text, body, url)
    # Links resolve against the document the DOM came from, which after a same-host client-side
    # redirect is not the URL asked for; the memo key and the result's `url` stay the requested URL.
    links = _extract_links_from_html(page.html, page.document_url)
    text = (extracted or "").strip()
    if not text:
        derived = _derived_api_outcome(url, page)
        if derived is not None:
            return derived
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
    if isinstance(result, str):
        # A meta-refresh stub reached the impersonated 200; this retry has no redirect loop to follow it, so decline.
        return None
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


def _wayback_applies(plain: PlainFetchResult) -> bool:
    """Whether the archive is a plausible substitute for this failed fetch.

    True for a host that refused us (403/406/429, ``http_status`` set) or never answered
    (``error``), never for a URL WE refused: a non-public target or a question-platform
    self-reference comes back ``blocked`` with no ``http_status``, and handing that to the archive
    is the SSRF bypass Tier 1's ``ssrf_blocked`` exclusion prevents. A JS wall is ``empty``/``ok``,
    not here, so the unrendered shell the archive stores is never tried where the browser rung is
    the right rescue.
    """
    if plain.status == "error":
        return True
    return plain.status == "blocked" and plain.http_status is not None


async def _try_wayback_fetch(
    url: str, direct: PlainFetchResult, *, now: datetime | None = None
) -> PlainFetchResult | None:
    """The Wayback Machine as the last free rung: fetch the freshest capture, or decline.

    Reuses ``research.wayback``'s pure helpers and this ladder's own ``_fetch_plain`` for the
    snapshot GET, so the 5 MiB body cap, the redirect vetting and the classification are the same
    a live page gets. Unlike Tier 1's rung it applies NO age bound: Tier 1's 30-day cutoff is
    calibrated on a URL the question cites as its grading source, and a driver-chosen URL carries
    no such guarantee, so the capture date is SURFACED in the served text (``wayback_lead``) for the
    driver to weigh rather than silently enforced. The inner URL a capture is OF is re-guarded
    (``guard._hop_refusal``) because a capture of a platform page presents
    ``web.archive.org`` as its host and would clear a self-reference check.
    """
    now = now or datetime.now(UTC)
    snapshot = await _fetch_plain(wayback_snapshot_url(url, now=now))
    if snapshot.status != "ok":
        # No capture served, or a capture we could not read (a JS-wall shell extracts to nothing).
        return None
    parsed = parse_snapshot_url(snapshot.url)
    if parsed is None:
        # Undatable: the archive answered the year request directly, so no dated capture to date-disclose.
        return None
    if await guard._hop_refusal(innermost_url(parsed.inner_url)) is not None:
        return None
    age_days = snapshot_age_days(parsed, now)
    if age_days is None:
        return None
    lead = wayback_lead(parsed, age_days, direct.status)
    return PlainFetchResult(
        status="ok",
        method="wayback",
        text=f"{lead}\n\n{snapshot.text}",
        links=snapshot.links,
        url=url,
        content_type=snapshot.content_type,
    )


def _per_call_ctx(question_ctx: LadderContext | None, *, query: str) -> LadderContext:
    """A per-CALL context off the question's: its own wall origin, its own rung list, its own ask.

    ``started`` is the origin every rung bounds itself against and one tool call is one wall, so it
    is taken here rather than at intake. What stays the question's is ``shared``, the per-question
    rung budget the archive and paid-read caps count on, so a driver that spends both snapshots on
    one gap cannot spend two more on the next. A call with no question context (which is what the
    suite drives) gets a fresh budget, so it behaves exactly as one call always did.
    """
    base = LadderContext(host_sems=_FETCH_HOST_SEMAPHORES) if question_ctx is None else question_ctx
    return replace(base, query=query, rungs=[], started=monotonic())


async def _fetch_via_ladder(
    url: str, *, query: str, pol: LadderPolicy, ctx: LadderContext | None, record: bool = True
) -> PlainFetchResult:
    """One run of the shared fetch ladder for ``url``, as this ladder's own result.

    The question-platform refusal happens HERE rather than inside the ladder, because it is this
    caller's own policy (the resolution-source fetcher drops those URLs when it selects them) and
    because it must refuse before anything is dialed. Everything past it is the shared ladder:
    the direct fetch with its redirect vetting and local document read, then the rungs ``pol``
    enables. ``record`` is False for the robots.txt pre-check, which is not a fetch the driver made.
    """
    blocked = _fetch_plain_url_block(url)
    if blocked is not None:
        return blocked
    result = await fetch_url(url, policy=pol, ctx=_per_call_ctx(ctx, query=query))
    if record:
        _log_ladder_markers(result)
    return ladder_adapter.as_plain_result(result, requested_url=url)


def _log_ladder_markers(result: FetchResult) -> None:
    """The two shared fetch markers for one tool call, with ``question=None``.

    The loop has no question id in hand at a tool call, exactly as its three event markers do not,
    so a join to a question goes through the run id (docs/telemetry_markers.md).
    """
    logger.info(fetch_markers.fetch_marker_line(result, qid=None, caller=LADDER_CALLER_GAP_FILL_V2))
    for line in fetch_markers.escalation_marker_lines(result, qid=None, caller=LADDER_CALLER_GAP_FILL_V2):
        logger.info(line)


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
    pdf = document_cache.cached_document(plain.url)
    if pdf is not None:
        document_cache.cache_document(url, pdf)
    local_document.log_local_document_read(
        url,
        method=local_document.PDF_LOCAL_METHOD,
        chars=len(plain.text),
        pages=None if pdf is None else pdf.pages_read,
        passages=None,
    )
    return _read_content_outcome(url, plain.text, plain.links, method=plain.method, start_char=start_char)


def _blocked_outcome(blocked: PlainFetchResult) -> ToolOutcome:
    """The one ``blocked`` contract the driver reads, whichever tool or rung refused the URL."""
    return ToolOutcome(content_markdown=blocked.text, method=blocked.method, status="blocked")


def _held_from_result(url: str, result: PlainFetchResult) -> local_document.HeldDocument:
    """What one ladder rung's result leaves us holding for ``url``.

    A parse from the local PDF rung wins over the flat text, because its page offsets are what
    make a digest's ``[p.N]`` labels exact; a scan is still held (page structure, no text), so
    the caller knows the free route is exhausted rather than untried. Text we do hold is cached
    for the run, so a later paginated ``fetch`` of the same URL is free.
    """
    if result.method == local_document.OVERSIZE_DOCUMENT_METHOD:
        return local_document.HeldDocument(oversize=True)
    if result.status == "blocked" and _fetch_plain_url_block(result.url) is not None:
        # A 3xx onto a question platform, held so the paid reader (Google's address) declines the same hop.
        return local_document.HeldDocument(refused_landing=result)
    pdf = document_cache.cached_document(result.url)
    if pdf is not None:
        held = local_document.held_pdf(pdf)
    elif result.status == "ok" and result.method != DOCUMENT_NEEDED_METHOD:
        held = local_document.HeldDocument(text=result.text.strip())
    else:
        # Reading the use-read_document placeholder as the page would digest our own instruction.
        return local_document.HeldDocument()
    if held.has_text and matched_throttle_phrase(held.text) is not None:
        # An interstitial is not the document (q45191), so the paid reader gets its turn.
        return local_document.HeldDocument()
    if held.has_text:
        _cache_fetch_result(url, held.text, result.links)
    return held


async def _run_local_document_ladder(url: str, *, ctx: LadderContext | None) -> local_document.HeldDocument:
    """The free rungs a document read gets: the shared ladder under ``GAP_FILL_DOCUMENT_POLICY``.

    Its 25 s wall is what every rung bounds itself against, and its rung set is ``fetch``'s minus
    the archive: this ladder sits immediately in front of the paid ``url_context`` read, and an
    archived copy is not what a document read was asked for. The impersonated retry matters more
    here than in ``fetch`` for the same reason — a 403 left standing was a paid read of a page the
    free retry fetches (a bls.gov PDF is one of the four measured rescues) — and the browser still
    runs on a page whose text is thin enough to look like a JavaScript shell, because digesting 100
    characters of navigation chrome would answer the ask out of furniture.
    """
    plain = await _fetch_via_ladder(url, query="", pol=GAP_FILL_DOCUMENT_POLICY, ctx=ctx)
    return _held_from_result(url, plain)


async def _acquire_local_document(url: str, *, ctx: LadderContext | None = None) -> local_document.HeldDocument:
    """What the free ladder holds for ``url``: something already read this run, or a fresh try.

    Bounded by ``_LOCAL_DOCUMENT_BUDGET_S`` so a slow host cannot spend the paid reader's
    budget as well as its own; on expiry we hold nothing and the reader gets its turn. The same
    figure is the ladder's own ``total_wall_s``, so every rung inside it declines under its floor
    rather than being cancelled mid-dial. The cancelled work includes at most one in-flight
    extraction thread, which finishes and drops its result, because a thread cannot be cancelled.

    That thread does NOT finish inside its own ``max_seconds``. The clock for that budget starts
    only once ``extract_pdf_text`` has read the declared page count and the whole bookmark
    outline, and the page in flight always completes, so the worker outlives this cancellation by
    that un-clocked prologue plus one page. It also hands its slot in the shared two-slot parse
    gate back as it unwinds, so a fresh parse can start alongside the abandoned one. Recorded in
    FUTURE.md under "The PDF parse overruns ``max_seconds``".
    """
    cached_pdf = document_cache.cached_document(url)
    if cached_pdf is not None:
        return local_document.held_pdf(cached_pdf)
    cached_text = _FETCH_TEXT_CACHE.get(url)
    if cached_text is not None:
        _FETCH_TEXT_CACHE.move_to_end(url)
        return local_document.HeldDocument(text=cached_text)
    try:
        return await asyncio.wait_for(_run_local_document_ladder(url, ctx=ctx), timeout=_LOCAL_DOCUMENT_BUDGET_S)
    except TimeoutError:
        logger.info(
            "agentic read_document local acquisition exceeded %.0fs, falling back to the reader: %s",
            _LOCAL_DOCUMENT_BUDGET_S,
            urlparse(url).netloc,
        )
        return local_document.HeldDocument()


async def _local_digest_outcome(url: str, ask: str, held: local_document.HeldDocument) -> ToolOutcome | None:
    """Answer the ask from text we hold, deterministically and for free — or None to pay instead.

    None means the one shape where a digest would answer the ask out of furniture: a sub-floor page
    with no parse behind it whose digest selected NO passage. All three conditions are needed, and
    the digest runs off the event loop for a measured reason; both receipts are in
    ``docs/agentic_gap_fill.md`` "Why the free digest can refuse to answer".
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


async def fetch(
    url: str, start_char: int = 0, *, question_topic: str = "", ctx: LadderContext | None = None
) -> ToolOutcome:
    """Read ``url`` for the driver: this run's window cache, else the whole shared fetch ladder.

    The rungs — the impersonated retry on a host's 403, the browser on a page too thin to be the
    page, the archive on one our address never reached — all run inside ``fetch_url`` under
    ``GAP_FILL_FETCH_POLICY``, so what is left here is reading the outcome: refuse a blocked URL,
    paginate a locally read document, hand a document with no text layer to ``read_document``, and
    otherwise window and cache what was read. A page the browser could not rescue either comes back
    ``empty`` and NEVER as a plain success, because the loop grants the ``fetched`` tier on status
    alone (docs/agentic_gap_fill.md).
    """
    cached = _fetch_from_cache(url, start_char)
    if cached is not None:
        return cached

    plain = await _fetch_via_ladder(url, query=question_topic, pol=GAP_FILL_FETCH_POLICY, ctx=ctx)
    if plain.status == "blocked":
        return _blocked_outcome(plain)
    if plain.method == local_document.PDF_LOCAL_METHOD:
        return _pdf_local_outcome(url, plain, start_char=start_char)
    if plain.method == DOCUMENT_NEEDED_METHOD:
        # `ladder_exhausted` says the free rungs just ran, so the reader does not re-request.
        return await read_document(plain.url, _generic_document_ask(question_topic), ladder_exhausted=True, ctx=ctx)
    if plain.status == "ok":
        return _read_content_outcome(url, plain.text, plain.links, method=plain.method, start_char=start_char)
    if plain.status == "empty":
        return _empty_fetch_outcome(plain.url)
    return ToolOutcome(content_markdown=plain.text, method=plain.method, status="error")


_ROBOTS_DISALLOWED_MSG = (
    "Document read not attempted: {host}'s robots.txt disallows Google-Extended, the token "
    "Gemini's url_context reader identifies as, so that read is refused at the host and returns "
    "no content whatever it costs. Nothing from this URL was read; do NOT cite it as a fetched "
    "source, and do NOT read it as evidence the fact is unavailable. Retrying will not help — "
    "look for the same fact on another host."
)

_PAID_DOCUMENT_READ_CAP_MSG = "Document read not attempted: this question's paid document-read limit is exhausted."


async def _fetch_robots_txt(robots_url: str, *, ctx: LadderContext | None = None) -> str | None:
    """Read one robots.txt through the shared ladder's DIRECT fetch; None when we could not.

    ``GAP_FILL_DIRECT_POLICY`` is the point: one direct fetch, no escalation rung, and this
    caller's verdict, which has no content floor. A robots.txt body is 33 to 45 characters, so a
    floor would read every host as "no directives" and quietly open the paid rung on hosts that
    disallow it. Everything else is the shared path's: the SSRF preflight, the filtering resolver,
    the per-hop redirect vetting and the body cap. Bounded at ``ROBOTS_FETCH_TIMEOUT_S`` on top of
    the hop's own clamp, because an unbounded per-host acquire is not a sensible price for a
    pre-check whose only job is to avoid one paid call; a timeout reads as unreadable, which is the
    only direction this may fail in. The per-host cache is ``robots_policy``'s, shared with the
    Tier-1 reader, because a host's policy is a property of the host.
    """
    try:
        result = await asyncio.wait_for(
            _fetch_via_ladder(robots_url, query="", pol=GAP_FILL_DIRECT_POLICY, ctx=ctx, record=False),
            timeout=ROBOTS_FETCH_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # pre-check soft-fail boundary: a robots.txt we cannot read must degrade to paying, never to failing the read
        logger.debug("agentic robots.txt pre-check failed for %s: %s: %s", robots_url, type(exc).__name__, exc)
        return None
    if result.status == "ok" and result.method == "plain":
        return result.text
    return None


async def _url_context_robots_skip(url: str, *, ctx: LadderContext | None = None) -> bool:
    """True when this host tells ``Google-Extended`` to stay out of ``url``'s path.

    Only the paid ``url_context`` rung consults this: the free rungs dial from our own client
    under our own user agent, and this bot's reading of ``Content-Signal: use=reference`` is
    that reference use is permitted. Proven live 2026-09-03 — see ``robots_policy``, which owns
    the per-host cache this shares with the Tier-1 reader.
    """
    return await google_extended_blocks_url(url, fetch_text=lambda robots_url: _fetch_robots_txt(robots_url, ctx=ctx))


def _take_paid_document_read_attempt(url: str, ctx: LadderContext | None) -> bool:
    """Claim one paid read from a question context; standalone calls own an independent allowance."""
    if ctx is None or ctx.shared.take_url_context_attempt():
        return True
    logger.info(
        "agentic read_document: skipping the paid reader for %s — this question's %d paid read(s) are spent",
        urlparse(url).netloc,
        RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS,
    )
    return False


async def _free_route_outcome(url: str, ask: str, held: local_document.HeldDocument) -> ToolOutcome | None:
    """What the free ladder settles for ``read_document`` without a paid read; None gives the reader its turn.

    Three settled shapes. A URL that led onto a question platform is refused, because the paid
    reader would follow the same hop. An oversize body is an error rather than a reason to
    escalate. Text we hold is digested, and the size gate rides the same branch as the text it
    guards so the two can never disagree: a document we hold is served from the digest whatever
    its size, and the biggest are the clearest case (the nine archived documents past the gate
    carried 67% of the season's reader tokens and the largest of them returned nothing for the
    money). A None digest is the one shape that must not be served: sub-floor chrome that no
    passage matched, which the paid reader is the right rung for (see ``_local_digest_outcome``).
    """
    if held.refused_landing is not None:
        return _blocked_outcome(held.refused_landing)
    if held.oversize:
        return _format_fetch_error(local_document.oversize_message(url), method=local_document.OVERSIZE_DOCUMENT_METHOD)
    if held.has_text or local_document.exceeds_url_context_size_gate(held.text):
        return await _local_digest_outcome(url, ask, held)
    return None


async def read_document(
    url: str, ask: str, *, ladder_exhausted: bool = False, ctx: LadderContext | None = None
) -> ToolOutcome:
    """Answer ``ask`` about ``url``: from the page's own text where we can get it, else Gemini.

    Acquisition-first. The free ladder runs before anything is spent (this run's cache, then
    the plain, impersonated-retry and rendered rungs ``fetch`` uses), and any text it holds is
    answered with a deterministic BM25 passage digest — ``method="digest_local"``. The paid
    ``url_context`` read happens only when the ladder holds nothing: a host that refuses us, a
    page with no text at all, or a PDF with no text layer. Measured 2026-09-03, that is two of 47
    archived fetch failures, against 191 reader calls over the 2026 summer season.

    A question-platform URL is refused before any rung runs, with the same ``blocked`` outcome
    ``fetch`` gives it, and so is a URL that 3xxes onto one (the free ladder's refusal of that hop
    comes back as ``HeldDocument.refused_landing``). Zero successful ``url_context`` retrievals withholds the ``fetched`` tier, and
    ``ladder_exhausted`` is internal and hidden from the driver-facing schema; both receipts are in
    ``docs/agentic_gap_fill.md`` "Why the paid reader's retrieval-count guard stays".
    """
    blocked = _fetch_plain_url_block(url)
    if blocked is not None:
        return _blocked_outcome(blocked)
    started = monotonic()
    held = local_document.HeldDocument() if ladder_exhausted else await _acquire_local_document(url, ctx=ctx)
    settled = await _free_route_outcome(url, ask, held)
    if settled is not None:
        return settled
    if not os.getenv(GOOGLE_API_KEY_ENV):
        return _format_fetch_error(f"Google API key is not configured; set {GOOGLE_API_KEY_ENV}.", method="document")
    if await _url_context_robots_skip(url, ctx=ctx):
        # Its own status token, never tiered: nothing was read, and a retry cannot help.
        logger.info(f"AGENTIC_URLCONTEXT_ROBOTS_SKIP: url={url} host={robots_host(url)}")
        return _format_fetch_error(
            _ROBOTS_DISALLOWED_MSG.format(host=robots_host(url)),
            status="robots_disallowed",
            method="document",
        )
    if not _take_paid_document_read_attempt(url, ctx):
        return _format_fetch_error(_PAID_DOCUMENT_READ_CAP_MSG, method="document")
    try:
        # What the total budget has left (docs/agentic_gap_fill.md, the budget arithmetic).
        text, n_url_success, statuses = await asyncio.wait_for(
            asyncio.to_thread(_run_document_read_sync, url, ask),
            timeout=min(_READ_DOCUMENT_TIMEOUT_S, _READ_DOCUMENT_TOTAL_BUDGET_S - (monotonic() - started)),
        )
    except TimeoutError:
        return _format_fetch_error("Document read timed out.", method="document")
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # tool-handler soft-fail boundary: a dead reader becomes a tool result the driver can read, never a loop crash
        return _format_fetch_error(f"Document read failed: {type(exc).__name__}: {exc}", method="document")
    if n_url_success == 0:
        # Greppable and keyed on `statuses`, which is the only thing separating three zeroes.
        logger.warning(f"AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED: url={url} statuses={','.join(statuses) or 'none'}")
        return _format_fetch_error(
            f"Document read retrieved no URL content: Gemini's url_context tool fetched nothing from {url}, "
            "so any answer would be unsourced recall rather than a read of the document.",
            method="document",
        )
    return ToolOutcome(content_markdown=text, method="document")


def question_ladder_context() -> LadderContext:
    """The ONE fetch-ladder context a question's tool calls share.

    What it carries is the per-question half: a fresh :class:`QuestionRungBudget`, which is what
    caps this question at two archive snapshots and two paid reads however many URLs the driver
    picks, and this ladder's own per-host politeness map. Everything per call — the ask, the wall
    origin, the rung list — is derived off it in :func:`_per_call_ctx`. Its own function so the
    seam that builds it (``agentic_gap_fill.run_gap_fill_v2``) does not have to know the fields.
    """
    return LadderContext(shared=QuestionRungBudget(), host_sems=_FETCH_HOST_SEMAPHORES)


def build_gap_fill_tools(question_topic: str, *, ctx: LadderContext | None = None) -> list[ToolSpec]:
    """The four tools the driver sees, in the order it sees them.

    ``ctx`` is the question's fetch-ladder context (:func:`question_ladder_context`), captured into
    the two handlers that fetch. None means one fresh budget per call, which is what a direct call
    with no question in hand gets.
    """

    async def _fetch_with_topic(url: str, start_char: int = 0) -> ToolOutcome:
        """``fetch`` with the topic and the ladder context bound; the schema stays (url, start_char)."""
        return await fetch(url, start_char, question_topic=question_topic, ctx=ctx)

    async def _read_document_public(url: str, ask: str) -> ToolOutcome:
        """``read_document`` with (url, ask) only, so a hallucinated ``ladder_exhausted`` cannot pay.

        The loop binds handlers with ``**arguments`` straight off the model, so an advertised — or
        merely invented — ``ladder_exhausted: true`` would skip the free ladder. Resolves
        ``read_document`` as a module attribute at call time, so the suite's patches still land.
        """
        return await read_document(url, ask, ctx=ctx)

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
            # Above _READ_DOCUMENT_TIMEOUT_S, so the document escalation fits inside this budget.
            timeout_s=90,
        ),
        ToolSpec(
            name="read_document",
            description=READ_DOCUMENT_DESCRIPTION,
            parameters=_READ_DOCUMENT_PARAMETERS,
            handler=_read_document_public,
            # 70 is GAP_FILL_V2_CONCLUDE_THRESHOLD (docs/agentic_gap_fill.md, the budget arithmetic).
            timeout_s=70,
        ),
    ]
