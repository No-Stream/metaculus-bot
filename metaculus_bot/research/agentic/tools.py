"""The four tools the gap-fill v2 driver calls, and the fetch ladder behind ``fetch``.

This module owns the handlers (``search_news``, ``search_web``, ``fetch``,
``read_document``), the ladder spine they run on — the plain hop loop with its redirect
vetting, the headless-Chromium rendered rung with its DNS pinning, the window cache that
serves ``start_char`` continuations — and ``build_gap_fill_tools``, whose list order is the
order the driver sees.

Support pieces live next door: ``tool_descriptions`` (driver-facing text + JSON schemas),
``tool_backends`` (the AskNews / Exa / Gemini calls and their result formatting),
``fetch_outcomes`` (classifying one plain HTTP response). The seams the suite monkeypatches
— ``_read_response_body``, ``_fetch_plain``, ``_try_rendered_fetch``, ``_resolve_pinned_host``,
``_run_document_read_sync``, ``read_document``, ``_READ_DOCUMENT_TIMEOUT_S``,
``_RENDERED_FETCH_GLOBAL_SEMAPHORE`` — are attributes of THIS module and are resolved here at
call time, so their callers stay here even where the callee moved out.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
from collections import OrderedDict
from typing import Any
from urllib.parse import urlparse

import aiohttp

from metaculus_bot.constants import (
    ASKNEWS_BACKOFF_SECS,  # noqa: F401  # re-export: tests read the AskNews retry ladder's constants off this module
    ASKNEWS_CLIENT_ID_ENV,
    ASKNEWS_MAX_TRIES,  # noqa: F401  # re-export: see ASKNEWS_BACKOFF_SECS above
    ASKNEWS_SECRET_ENV,
    EXA_API_KEY_ENV,
    GOOGLE_API_KEY_ENV,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
)
from metaculus_bot.research import resolution_source
from metaculus_bot.research.agentic.fetch_outcomes import (
    _DOCUMENT_NEEDED_MSG,
    _HTML_CONTENT_TYPE_TOKENS,
    _RETRYABLE_FETCH_BLOCK_STATUSES,
    _TEXTUAL_CONTENT_TYPE_TOKENS,
    PlainFetchResult,
    _body_is_document,
    _content_type_is_document,
    _extract_links_from_html,
    _fetch_plain_url_block,
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
_READ_DOCUMENT_TIMEOUT_S = 60.0
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
_PLAYWRIGHT_WARNED = False


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


async def _read_response_body(resp: aiohttp.ClientResponse, label: str) -> bytes | None:
    return await read_body_capped(resp, max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES, label=label)


async def _plain_response_outcome(resp: aiohttp.ClientResponse, current_url: str) -> PlainFetchResult | str:
    """Classify one HTTP response: a terminal result, or the next URL on a vetted 3xx."""
    status = resp.status
    content_type = (resp.headers.get("Content-Type") or "").lower() if resp.headers else ""
    if status in REDIRECT_STATUSES:
        return await _plain_redirect_outcome(resp, current_url, content_type)
    if status in _RETRYABLE_FETCH_BLOCK_STATUSES:
        return PlainFetchResult(
            status="blocked",
            method="plain",
            text=f"Fetch blocked with HTTP {status}.",
            links=[],
            url=current_url,
            content_type=content_type or None,
        )
    if status != 200:
        return PlainFetchResult(
            status="error",
            method="plain",
            text=f"Fetch failed with HTTP {status}.",
            links=[],
            url=current_url,
            content_type=content_type or None,
        )
    if _content_type_is_document(content_type):
        return PlainFetchResult(
            status="ok",
            method="document_needed",
            text=_DOCUMENT_NEEDED_MSG,
            links=[],
            url=current_url,
            content_type=content_type or None,
        )
    body = await _read_response_body(resp, f"agentic fetch {urlparse(current_url).netloc}")
    if body is None:
        return PlainFetchResult(
            status="error",
            method="plain",
            text="Fetch body exceeded the size limit.",
            links=[],
            url=current_url,
            content_type=content_type or None,
        )
    if _body_is_document(body):
        return PlainFetchResult(
            status="ok",
            method="document_needed",
            text=_DOCUMENT_NEEDED_MSG,
            links=[],
            url=current_url,
            content_type=content_type or None,
        )

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


async def _try_rendered_fetch(url: str) -> PlainFetchResult | None:
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
                try:
                    if await resolution_source.is_public_http_url(request.url):
                        await route.continue_()
                    else:
                        await route.abort("blockedbyclient")
                except PlaywrightError as exc:
                    # A request can still be in flight when the page/context tears
                    # down (typically after a goto timeout): continue_/abort then
                    # races the close and raises TargetClosedError in this detached
                    # event-listener task — the unhandled-error storm seen
                    # 2026-07-25. Swallow it: a closed target has no live socket, so
                    # an abort that "fails" because the target is already gone still
                    # lets nothing through — the SSRF guarantee is unaffected.
                    # unroute_all in the finally is the primary drain; this is the
                    # residual-race backstop. Only Playwright's own Error is caught,
                    # so a genuine bug (a Python exception) still propagates.
                    logger.debug("agentic route guard race during teardown: %s", exc)

            await context.route("**/*", _guard_route)
            page = await context.new_page()
            try:
                response = await page.goto(url, wait_until="networkidle", timeout=_RENDERED_FETCH_TIMEOUT_MS)
                content_type = (
                    (response.headers.get("content-type") or "").lower()
                    if response is not None and hasattr(response, "headers")
                    else ""
                )
                if _content_type_is_document(content_type):
                    return PlainFetchResult(
                        status="ok",
                        method="document_needed",
                        text="This URL is a PDF or image — use read_document(url, ask) to read it.",
                        links=[],
                        url=url,
                        content_type=content_type or None,
                    )
                html = await page.content()
                body = html.encode("utf-8", errors="replace")
                extracted = await asyncio.to_thread(resolution_source._extract_main_text, body, url)
                links = _extract_links_from_html(html, url)
                text = (extracted or "").strip()
                if not text:
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


async def fetch(url: str, start_char: int = 0, *, question_topic: str = "") -> ToolOutcome:
    cached = _fetch_from_cache(url, start_char)
    if cached is not None:
        return cached

    plain = await _fetch_plain(url)
    if plain.status == "blocked":
        return ToolOutcome(content_markdown=plain.text, method=plain.method, status="blocked")
    if plain.method == "document_needed":
        # Rung 3: auto-escalate PDFs/images to the read_document backend so the
        # driver keeps its "handled automatically" contract without spending a
        # second tool call. read_document stays separately exposed for directed asks.
        return await read_document(plain.url, _generic_document_ask(question_topic))
    if plain.status not in ("ok", "empty"):
        return ToolOutcome(content_markdown=plain.text, method=plain.method, status="error")
    if plain.status == "ok" and not plain.escalate_rendered:
        return _read_content_outcome(url, plain.text, plain.links, method="plain", start_char=start_char)

    rendered = await _try_rendered_fetch(plain.url)
    if rendered is not None:
        if rendered.method == "document_needed":
            return await read_document(rendered.url, _generic_document_ask(question_topic))
        if rendered.status == "ok" and rendered.text:
            return _read_content_outcome(url, rendered.text, rendered.links, method="rendered", start_char=start_char)
    # Rendered was unavailable, errored, or itself extracted nothing. Fall back to
    # plain ONLY when the plain fetch actually read (thin-but-real) content; a plain
    # fetch that produced nothing has no content to hand back and must not be
    # laundered as a successful "plain"/"ok" retrieval (the companiesmarketcap.com
    # js-wall failure: an unread page stamped `fetched` and superseded the briefing).
    if plain.status == "ok":
        return _read_content_outcome(url, plain.text, plain.links, method="plain", start_char=start_char)
    return _empty_fetch_outcome(plain.url)


async def read_document(url: str, ask: str) -> ToolOutcome:
    """Read a document via Gemini url_context, granting the ``fetched`` tier only on a real read.

    ``method="document"`` maps to the ``fetched`` tier (``loop._METHOD_TO_TIER``), which is
    the highest authority the artifact renderer has: only a ``fetched`` discrepancy enters
    the SUPERSEDE block that tells every forecaster to override the briefing. So a
    fluent-but-ungrounded answer here is the Q38195 failure mode with a bigger blast
    radius, and the quote check can't catch it — it is deliberately WARN-only for this
    tool (paraphrase and ellipsis-joined quotes make a hard gate too false-positive-prone).

    Hence the retrieval-count guard below, mirroring the grounded-chunk floor
    ``gemini_search`` already applies: zero successful url_context retrievals withholds the
    tier exactly like the empty-text guard does, rather than laundering parametric recall
    as a primary-source read.
    """
    if not os.getenv(GOOGLE_API_KEY_ENV):
        return _format_fetch_error(f"Google API key is not configured; set {GOOGLE_API_KEY_ENV}.", method="document")
    try:
        text, n_url_success, statuses = await asyncio.wait_for(
            asyncio.to_thread(_run_document_read_sync, url, ask), timeout=_READ_DOCUMENT_TIMEOUT_S
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
            handler=read_document,
            timeout_s=70,
        ),
    ]
