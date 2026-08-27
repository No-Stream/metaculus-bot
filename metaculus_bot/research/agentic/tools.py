from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import socket
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp

from metaculus_bot.constants import (
    ASKNEWS_BACKOFF_SECS,
    ASKNEWS_CLIENT_ID_ENV,
    ASKNEWS_MAX_TRIES,
    ASKNEWS_SECRET_ENV,
    EXA_API_KEY_ENV,
    GAP_FILL_V2_MIN_CONTENT_CHARS,
    GAP_FILL_V2_READER_MODEL,
    GOOGLE_API_KEY_ENV,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
)
from metaculus_bot.research import providers as research_providers
from metaculus_bot.research import resolution_source
from metaculus_bot.research.agentic.types import ToolOutcome, ToolSpec
from metaculus_bot.research.http_fetch import (
    BROWSER_HEADERS,
    MAX_REDIRECTS,
    MAX_UNDECODABLE_CHAR_RATIO,
    REDIRECT_STATUSES,
    decode_text_body,
    read_body_capped,
)
from metaculus_bot.research.url_context_telemetry import extract_url_context_telemetry

logger = logging.getLogger(__name__)

_FETCH_WINDOW_CHARS = 8000
_FETCH_CACHE_MAX_ENTRIES = 50
_FETCH_LINK_CAP = 25
_FETCH_MIN_CONTENT_CHARS = GAP_FILL_V2_MIN_CONTENT_CHARS
_RENDERED_FETCH_TIMEOUT_MS = 35_000
_READ_DOCUMENT_TIMEOUT_S = 60.0
# Client-side HTTP ceilings sized just UNDER the tools' loop budgets so the
# underlying socket is torn down before the loop's asyncio.wait_for fires — a
# hung endpoint then frees its slot instead of pinning it to the wall deadline.
_EXA_HTTP_TIMEOUT_S = 18.0  # under search_web's ToolSpec timeout_s in build_gap_fill_tools
_READ_DOCUMENT_HTTP_TIMEOUT_MS = 55_000  # under _READ_DOCUMENT_TIMEOUT_S, read_document's own deadline
_EXA_RETRY_DELAYS_S = (1.0, 4.0)
_EXA_GLOBAL_SEMAPHORE = asyncio.Semaphore(4)
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

SEARCH_NEWS_DESCRIPTION = (
    "Search recent and historical NEWS coverage (AskNews). Use for: events,\n"
    "announcements, things that happened, ongoing-situation updates. Query with a\n"
    "short natural-language phrase, not keywords. Returns a digest of matching\n"
    "articles with dates and URLs. Use search_web instead for: reports, datasets,\n"
    "official documents, niche/technical facts, or anything where the best source\n"
    "is not a news article.\n"
    'Example: search_news(query="Nauru parliament treaty ratification vote")'
)

SEARCH_WEB_DESCRIPTION = (
    "Semantic web search (Exa). Use for: official documents, datasets, reports,\n"
    "organizational pages, technical/niche facts, finding a primary source you\n"
    "believe exists. Returns results with URLs and relevant excerpts. Follow up\n"
    "promising results with fetch(url) — excerpts are often not enough to verify\n"
    "a claim. Use search_news instead for event/news coverage.\n"
    'Example: search_web(query="IAEA safeguards report Iran enrichment June 2026 pdf")'
)

FETCH_DESCRIPTION = (
    "Fetch a URL and return its main content as concise markdown, plus a list of\n"
    "outbound links. Handles ordinary pages, JavaScript-heavy pages, PDFs, and\n"
    "images automatically (the result's `method` field tells you how it was\n"
    "read) — do NOT avoid a URL because of its format. Content over the size cap\n"
    "is truncated, ending with `[truncated at N of M chars — call again with\n"
    "start_char=N]`; pass start_char to read the next window (continuations are\n"
    "served from cache — they are cheap and do not refetch). Links in the result\n"
    "are leads you can fetch next.\n"
    "Use read_document instead only when you need a specific question answered\n"
    "from inside a long/complex document.\n"
    "Do NOT fetch metaculus.com URLs — the question brief already reflects them.\n"
    'Example: fetch(url="https://www.ons.gov.uk/releases/gdpquarterly")\n'
    'Example: fetch(url="https://example.gov/long-report", start_char=12000)'
)

READ_DOCUMENT_DESCRIPTION = (
    "Ask a specific question of a specific document (Gemini reads the URL —\n"
    "handles PDFs, images, and JS pages natively). Slower and costlier than\n"
    "fetch: use it when you need targeted extraction from a long or complex\n"
    "document, or when fetch returned status=blocked/js_wall/error for a URL you\n"
    "still need. Always pass a precise `ask`.\n"
    'Example: read_document(url="https://example.gov/report-q2.pdf",\n'
    '                       ask="What is the reported unemployment rate for May 2026, and what revision to April is stated?")'
)

_SEARCH_NEWS_PARAMETERS = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}

_SEARCH_WEB_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "end_published_date": {"type": ["string", "null"]},
    },
    "required": ["query"],
    "additionalProperties": False,
}

_FETCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "start_char": {"type": "integer", "minimum": 0},
    },
    "required": ["url"],
    "additionalProperties": False,
}

_READ_DOCUMENT_PARAMETERS = {
    "type": "object",
    "properties": {
        "url": {"type": "string"},
        "ask": {"type": "string"},
    },
    "required": ["url", "ask"],
    "additionalProperties": False,
}

_PDF_CONTENT_TYPE_TOKENS = ("application/pdf",)
_IMAGE_CONTENT_TYPE_PREFIXES = ("image/",)
_RETRYABLE_FETCH_BLOCK_STATUSES = {403, 406, 429}
_TEXTUAL_CONTENT_TYPE_TOKENS = ("text/plain", "text/csv", "application/json")
_HTML_CONTENT_TYPE_TOKENS = ("text/html", "application/xhtml+xml")
_RATE_LIMIT_RE = re.compile(r"\b(429|rate[\s-]?limit|too many requests|over limit|quota)\b", re.IGNORECASE)


@dataclass(slots=True)
class PlainFetchResult:
    status: str
    method: str
    text: str
    links: list[str]
    url: str
    content_type: str | None = None
    escalate_rendered: bool = False


class _LinkCollector(HTMLParser):
    def __init__(self, *, base_url: str, cap: int) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._cap = cap
        self._links: list[str] = []
        self._seen: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if len(self._links) >= self._cap or tag.lower() != "a":
            return
        href = None
        for name, value in attrs:
            if name.lower() == "href":
                href = value
                break
        if not href:
            return
        absolute = urljoin(self._base_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme not in ("http", "https"):
            return
        if absolute in self._seen:
            return
        self._seen.add(absolute)
        self._links.append(absolute)

    @property
    def links(self) -> list[str]:
        return list(self._links)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except TypeError:
            return str(value)
    return str(value)


def _mapping_or_attrs_get(item: object, *names: str) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if name in item:
                return item[name]
        return None
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return None


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


def _content_type_is_document(content_type: str | None) -> bool:
    if not content_type:
        return False
    lowered = content_type.lower()
    if any(token in lowered for token in _PDF_CONTENT_TYPE_TOKENS):
        return True
    return any(lowered.startswith(prefix) for prefix in _IMAGE_CONTENT_TYPE_PREFIXES)


def _body_is_document(body: bytes) -> bool:
    stripped = body.lstrip()
    if stripped.startswith(b"%PDF-"):
        return True
    return (
        stripped.startswith(b"\x89PNG\r\n\x1a\n")
        or stripped.startswith(b"\xff\xd8\xff")
        or stripped.startswith((b"GIF87a", b"GIF89a"))
    )


def _extract_links_from_html(html: str, base_url: str) -> list[str]:
    parser = _LinkCollector(base_url=base_url, cap=_FETCH_LINK_CAP)
    parser.feed(html)
    parser.close()
    return parser.links


def _format_asknews_results(articles: list[Any]) -> str:
    if not articles:
        return "No AskNews articles found."
    lines: list[str] = []
    for article in articles:
        title = _stringify(_mapping_or_attrs_get(article, "eng_title", "title")) or "Untitled article"
        date = _stringify(_mapping_or_attrs_get(article, "pub_date"))
        source = _stringify(_mapping_or_attrs_get(article, "source_id")) or "unknown"
        url = _stringify(_mapping_or_attrs_get(article, "article_url"))
        summary = _stringify(_mapping_or_attrs_get(article, "summary"))
        lines.append(f"### {title}")
        if date:
            lines.append(f"Date: {date}")
        lines.append(f"Source: {source}")
        if url:
            lines.append(f"URL: {url}")
        if summary:
            lines.append(f"Summary: {summary}")
        lines.append("")
    lines.pop()
    return "\n".join(lines)


def _format_exa_results(results: list[Any]) -> str:
    if not results:
        return "No Exa results found."
    lines: list[str] = []
    for result in results:
        title = _stringify(_mapping_or_attrs_get(result, "title")) or "Untitled result"
        url = _stringify(_mapping_or_attrs_get(result, "url"))
        published = _stringify(_mapping_or_attrs_get(result, "published_date", "publishedDate"))
        highlights_raw = _mapping_or_attrs_get(result, "highlights")
        if isinstance(highlights_raw, str):
            highlights = [highlights_raw]
        elif isinstance(highlights_raw, list):
            highlights = [_stringify(item) for item in highlights_raw if _stringify(item)]
        else:
            highlights = []
        lines.append(f"### {title}")
        if url:
            lines.append(f"URL: {url}")
        if published:
            lines.append(f"Published: {published}")
        if highlights:
            lines.append("Highlights:")
            lines.extend(f"- {highlight}" for highlight in highlights)
        lines.append("")
    lines.pop()
    return "\n".join(lines)


def _format_fetch_error(message: str, *, status: str = "error", method: str = "plain") -> ToolOutcome:
    return ToolOutcome(content_markdown=message, method=method, status=status)


def _render_fetch_outcome(url: str, text: str, links: list[str], method: str, start_char: int) -> ToolOutcome:
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
    global _PLAYWRIGHT_WARNED
    if _PLAYWRIGHT_WARNED:
        return
    _PLAYWRIGHT_WARNED = True
    logger.warning("agentic fetch rendered rung unavailable: %s: %s", type(exc).__name__, exc)


def _is_rate_limited_error(exc: BaseException) -> bool:
    """Generic 429/rate-limit/quota classifier (Exa and AskNews retry paths)."""
    return bool(_RATE_LIMIT_RE.search(str(exc)))


async def _call_asknews_search(query: str) -> list[Any]:
    from asknews_sdk import AsyncAskNewsSDK  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

    client_id = os.getenv(ASKNEWS_CLIENT_ID_ENV)
    secret = os.getenv(ASKNEWS_SECRET_ENV)
    if not client_id or not secret:
        raise ValueError(f"Missing AskNews credentials: {ASKNEWS_CLIENT_ID_ENV} / {ASKNEWS_SECRET_ENV}")

    tries = max(1, int(ASKNEWS_MAX_TRIES))
    backoff = float(ASKNEWS_BACKOFF_SECS)
    semaphore = research_providers.get_asknews_semaphore()

    async with semaphore, AsyncAskNewsSDK(client_id=client_id, client_secret=secret, scopes={"news"}) as sdk:
        last_exc: Exception | None = None
        for attempt in range(1, tries + 1):
            try:
                await research_providers.asknews_rate_gate()
                response = await sdk.news.search_news(
                    query=query,
                    n_articles=6,
                    return_type="both",
                    strategy="news knowledge",
                )
                return list(response.as_dicts or [])
            except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
                last_exc = exc
                # Transient-only retry, matching the primary provider's ``_is_retryable``
                # (``providers.py``). The 403011 subscription-inactive error is PERMANENT —
                # off-season billing, not throttling — so it must NOT be exempted here: re-rolling
                # it burns the whole ``ASKNEWS_MAX_TRIES`` ladder of ``ASKNEWS_BACKOFF_SECS``
                # sleeps (a double-digit fraction of GAP_FILL_V2_WALL_DEADLINE, since the sleep
                # below grows as 3**attempt) on a call that can never succeed.
                if not _is_rate_limited_error(exc):
                    msg = str(exc).lower()
                    if "concurrency limit" not in msg:
                        raise
                if attempt >= tries:
                    raise
                await asyncio.sleep(backoff * (10 + 3**attempt))
        if last_exc is not None:
            raise last_exc
    return []


async def _run_exa_search(query: str, end_published_date: str | None) -> Any:
    import httpx  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
    from exa_py import AsyncExa  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

    api_key = os.getenv(EXA_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"Missing Exa API key: {EXA_API_KEY_ENV}")
    client = AsyncExa(api_key=api_key)
    # Inject a client-side-bounded httpx client (the SDK's default timeout is
    # 600s): a hung Exa endpoint gives up at _EXA_HTTP_TIMEOUT_S instead of
    # pinning the coroutine to the loop's wall deadline. The async client also
    # means no worker thread to leak — the old to_thread path could strand a
    # hung sync `requests` call in the shared ThreadPoolExecutor.
    client._client = httpx.AsyncClient(base_url=client.base_url, headers=client.headers, timeout=_EXA_HTTP_TIMEOUT_S)
    kwargs: dict[str, Any] = {
        "query": query,
        "type": "auto",
        "num_results": 8,
        "contents": {"highlights": True},
    }
    if end_published_date is not None:
        kwargs["end_published_date"] = end_published_date
    try:
        return await client.search(**kwargs)
    finally:
        await client._client.aclose()


async def _call_exa_search(query: str, end_published_date: str | None) -> list[Any]:
    async with _EXA_GLOBAL_SEMAPHORE:
        for attempt in range(len(_EXA_RETRY_DELAYS_S) + 1):
            try:
                response = await _run_exa_search(query, end_published_date)
                results = _mapping_or_attrs_get(response, "results")
                return list(results) if isinstance(results, list) else []
            except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
                if attempt >= len(_EXA_RETRY_DELAYS_S) or not _is_rate_limited_error(exc):
                    raise
                await asyncio.sleep(_EXA_RETRY_DELAYS_S[attempt])
    return []


async def _read_response_body(resp: aiohttp.ClientResponse, label: str) -> bytes | None:
    return await read_body_capped(resp, max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES, label=label)


_METACULUS_FETCH_BLOCK_MSG = (
    "Metaculus pages are already reflected in the question brief; do not fetch metaculus.com URLs."
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
    # Block metaculus.com from our runner IP. Question pages are a JS SPA whose
    # near-empty plain fetch would auto-escalate to headless Chromium, whose
    # route guard then permits the SPA's own XHR fan-out to the Metaculus API —
    # all from our IP, on the same host the critical API calls use. Blocking here
    # (before _get_session) kills both our-IP rungs; rendered only runs after a
    # plain fetch. The brief already embeds the resolution criteria these URLs
    # would yield. (read_document is Gemini's IP, not ours, so it is not gated.)
    if resolution_source.is_metaculus_self_ref(url):
        return PlainFetchResult(
            status="blocked",
            method="plain",
            text=_METACULUS_FETCH_BLOCK_MSG,
            links=[],
            url=url,
        )

    session = resolution_source._get_session()
    async with session:
        current_url = url
        for _ in range(MAX_REDIRECTS + 1):
            async with _host_gate(current_url):
                try:
                    async with session.get(current_url, allow_redirects=False) as resp:
                        status = resp.status
                        content_type = (resp.headers.get("Content-Type") or "").lower() if resp.headers else ""
                        if status in REDIRECT_STATUSES:
                            location = resp.headers.get("Location") if resp.headers else None
                            if not location:
                                return PlainFetchResult(
                                    status="error",
                                    method="plain",
                                    text=f"Malformed redirect from {current_url}",
                                    links=[],
                                    url=current_url,
                                    content_type=content_type or None,
                                )
                            next_url = urljoin(current_url, location)
                            if not await resolution_source.is_public_http_url(next_url):
                                return PlainFetchResult(
                                    status="blocked",
                                    method="plain",
                                    text="Blocked non-public redirect target.",
                                    links=[],
                                    url=next_url,
                                    content_type=content_type or None,
                                )
                            # A 3xx to metaculus.com must not be followed either (same
                            # our-IP / no-new-info rationale as the initial-URL block).
                            if resolution_source.is_metaculus_self_ref(next_url):
                                return PlainFetchResult(
                                    status="blocked",
                                    method="plain",
                                    text=_METACULUS_FETCH_BLOCK_MSG,
                                    links=[],
                                    url=next_url,
                                    content_type=content_type or None,
                                )
                            current_url = next_url
                            continue
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
                                text="This URL is a PDF or image — use read_document(url, ask) to read it.",
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
                                text="This URL is a PDF or image — use read_document(url, ask) to read it.",
                                links=[],
                                url=current_url,
                                content_type=content_type or None,
                            )

                        # Charset-honoring decode (BOM > declared charset > UTF-8), not a
                        # forced UTF-8 read: a windows-1252 or UTF-16 body decoded that way
                        # is `0�.�4�2�`-style mojibake that reached the driver as
                        # status="ok". The ratio is the refusal signal on the textual
                        # branch below; the HTML branch is unaffected because its main
                        # text comes from `_extract_main_text`, which decodes the raw
                        # bytes itself.
                        html, undecodable_ratio = decode_text_body(body, content_type)
                        if any(token in content_type for token in _HTML_CONTENT_TYPE_TOKENS) or "<html" in html.lower():
                            extracted = await asyncio.to_thread(resolution_source._extract_main_text, body, current_url)
                            text = extracted or ""
                            links = _extract_links_from_html(html, current_url)
                            if not text.strip():
                                # No extractable text on a 200 OK (JS wall, consent
                                # gate, empty body). A distinct "empty" status keeps
                                # the ladder escalating to the rendered rung while
                                # barring this outcome from the status=="ok" tier
                                # grant — an unread page must never be "fetched".
                                return PlainFetchResult(
                                    status="empty",
                                    method="plain",
                                    text="Plain fetch returned no extractable text.",
                                    links=links,
                                    url=current_url,
                                    content_type=content_type or None,
                                    escalate_rendered=True,
                                )
                            return PlainFetchResult(
                                status="ok",
                                method="plain",
                                text=text,
                                links=links,
                                url=current_url,
                                content_type=content_type or None,
                                escalate_rendered=len(text.strip()) < _FETCH_MIN_CONTENT_CHARS,
                            )

                        if any(token in content_type for token in _TEXTUAL_CONTENT_TYPE_TOKENS) or not content_type:
                            if undecodable_ratio > MAX_UNDECODABLE_CHAR_RATIO:
                                # The decode failed rather than the text being slightly
                                # dirty (BOM-less UTF-16, an undeclared 8-bit codec):
                                # what we hold is replacement chars and NULs, not the
                                # page. Shipping it as "ok" would hand the driver
                                # mojibake as a read source. "empty" keeps the ladder
                                # escalating to the rendered rung — the browser's own
                                # charset sniffing can rescue what a declared-charset
                                # decode could not — while barring the tier grant.
                                return PlainFetchResult(
                                    status="empty",
                                    method="plain",
                                    text="Plain fetch could not decode the body as text.",
                                    links=[],
                                    url=current_url,
                                    content_type=content_type or None,
                                    escalate_rendered=True,
                                )
                            # Same allow-listed tag strip the Tier-1 raw-body branches run:
                            # a Datawrapper poll CSV measured 69% `<a href=...>` markup, so
                            # without it the driver's max_result_chars budget buys tags
                            # instead of rows, and the inflated length also defeats the
                            # short-content escalation heuristic below.
                            text = resolution_source.strip_html_tags(html).strip()
                            if not text:
                                return PlainFetchResult(
                                    status="empty",
                                    method="plain",
                                    text="Plain fetch returned no extractable text.",
                                    links=[],
                                    url=current_url,
                                    content_type=content_type or None,
                                    escalate_rendered=True,
                                )
                            return PlainFetchResult(
                                status="ok",
                                method="plain",
                                text=text,
                                links=[],
                                url=current_url,
                                content_type=content_type or None,
                                escalate_rendered=len(text) < _FETCH_MIN_CONTENT_CHARS,
                            )

                        return PlainFetchResult(
                            status="error",
                            method="plain",
                            text=f"Unsupported content type: {content_type or 'unknown'}",
                            links=[],
                            url=current_url,
                            content_type=content_type or None,
                        )
                except (TimeoutError, aiohttp.ClientError) as exc:
                    return PlainFetchResult(
                        status="error",
                        method="plain",
                        text=f"Fetch error: {type(exc).__name__}: {exc}",
                        links=[],
                        url=current_url,
                    )
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


async def _resolve_pinned_host(url: str) -> tuple[str, str] | None:
    """Vet ``url``'s host and resolve it to ONE public IP for Chromium DNS pinning.

    Returns ``(host, vetted_ip)`` — the ``--host-resolver-rules=MAP`` operands — or
    ``None`` when the URL is non-public, unresolvable, or ANY resolved address is
    disallowed. Mirrors :func:`resolution_source.is_public_http_url`'s classification
    (scheme, userinfo, and the shared :func:`resolution_source._ip_is_disallowed`
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
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    if parsed.scheme.lower() not in ("http", "https"):
        return None
    # Userinfo defeats hostname-based trust (`https://trusted@10.0.0.1/`).
    if parsed.username is not None or parsed.password is not None:
        return None
    host = parsed.hostname or ""
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

    # Hostname branch: resolve once off the event loop, reject if ANY resolved
    # address is disallowed (same rebinding-defense stance as the preflight),
    # and pin to the first survivor.
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
        if resolution_source._ip_is_disallowed(resolved):
            return None
        if vetted_ip is None:
            vetted_ip = str(resolved)
    if vetted_ip is None:
        return None
    return host, vetted_ip


async def _try_rendered_fetch(url: str) -> PlainFetchResult | None:
    try:
        from playwright.async_api import (
            Error as PlaywrightError,
        )
        from playwright.async_api import async_playwright  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
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
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
        _warn_playwright_unavailable_once(exc)
        return None


def _build_document_prompt(ask: str) -> str:
    return (
        f"{ask}\n\n"
        "Answer using verbatim quotes from the document whenever possible. Include the document's stated dates. "
        "If the document does not address the ask, say that plainly."
    )


def _run_document_read_sync(url: str, ask: str) -> tuple[str, int]:
    """Read ``url`` via Gemini url_context. Returns ``(text, n_url_retrievals_that_succeeded)``.

    The retrieval count is returned, not discarded, because the text alone cannot tell a
    real document read from a fluent answer out of parametric memory — Gemini produces
    both happily, and ``read_document`` grants the highest verification tier the artifact
    renderer has. Same reader the grounded-search provider uses for the same reason (see
    ``research/url_context_telemetry``).
    """
    from google import genai  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
    from google.genai import types as genai_types  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

    api_key = os.getenv(GOOGLE_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"Missing Google API key: {GOOGLE_API_KEY_ENV}")
    # Client-side timeout (ms) so a hung Gemini endpoint returns the thread —
    # read_document runs this sync call under asyncio.to_thread, and wait_for
    # cancels the coroutine but can't cancel the thread; without this ceiling a
    # stuck endpoint leaks the worker into the shared ThreadPoolExecutor.
    client = genai.Client(api_key=api_key, http_options=genai_types.HttpOptions(timeout=_READ_DOCUMENT_HTTP_TIMEOUT_MS))
    tools: list[Any] = [{"url_context": {}}]
    config = genai_types.GenerateContentConfig(tools=tools)
    response = client.models.generate_content(
        model=GAP_FILL_V2_READER_MODEL,
        contents=f"{_build_document_prompt(ask)}\n\nURL: {url}",
        config=config,
    )
    _, _, n_url_success, _ = extract_url_context_telemetry(response)
    return (_stringify(getattr(response, "text", "")) or "", n_url_success)


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
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
        return _format_fetch_error(f"AskNews search failed: {type(exc).__name__}: {exc}", method="news")
    return ToolOutcome(content_markdown=_format_asknews_results(articles), method="news")


async def search_web(query: str, end_published_date: str | None = None) -> ToolOutcome:
    if not os.getenv(EXA_API_KEY_ENV):
        return _format_fetch_error(f"Exa API key is not configured; set {EXA_API_KEY_ENV}.", method="search")
    try:
        results = await _call_exa_search(query, end_published_date)
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
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
        return _render_fetch_outcome(url, plain.text, plain.links, "plain", start_char)

    rendered = await _try_rendered_fetch(plain.url)
    if rendered is not None:
        if rendered.method == "document_needed":
            return await read_document(rendered.url, _generic_document_ask(question_topic))
        if rendered.status == "ok" and rendered.text:
            return _render_fetch_outcome(url, rendered.text, rendered.links, "rendered", start_char)
    # Rendered was unavailable, errored, or itself extracted nothing. Fall back to
    # plain ONLY when the plain fetch actually read (thin-but-real) content; a plain
    # fetch that produced nothing has no content to hand back and must not be
    # laundered as a successful "plain"/"ok" retrieval (the companiesmarketcap.com
    # js-wall failure: an unread page stamped `fetched` and superseded the briefing).
    if plain.status == "ok":
        return _render_fetch_outcome(url, plain.text, plain.links, "plain", start_char)
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
        text, n_url_success = await asyncio.wait_for(
            asyncio.to_thread(_run_document_read_sync, url, ask), timeout=_READ_DOCUMENT_TIMEOUT_S
        )
    except TimeoutError:
        return _format_fetch_error("Document read timed out.", method="document")
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
        return _format_fetch_error(f"Document read failed: {type(exc).__name__}: {exc}", method="document")
    if n_url_success == 0:
        # Greppable, mirroring gemini_search's GEMINI_UNGROUNDED_SUPPRESSED so the rate is
        # measurable from the archived run logs.
        logger.warning(f"AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED: url={url}")
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
