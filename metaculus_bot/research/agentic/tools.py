from __future__ import annotations

import asyncio
import logging
import os
import re
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
from metaculus_bot.research.http_fetch import BROWSER_HEADERS, MAX_REDIRECTS, REDIRECT_STATUSES, read_body_capped

logger = logging.getLogger(__name__)

_FETCH_WINDOW_CHARS = 8000
_FETCH_CACHE_MAX_ENTRIES = 50
_FETCH_LINK_CAP = 25
_FETCH_MIN_CONTENT_CHARS = GAP_FILL_V2_MIN_CONTENT_CHARS
_RENDERED_FETCH_TIMEOUT_MS = 35_000
_READ_DOCUMENT_TIMEOUT_S = 60.0
_EXA_RETRY_DELAYS_S = (1.0, 4.0)
_EXA_GLOBAL_SEMAPHORE = asyncio.Semaphore(4)
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

    async with semaphore:
        async with AsyncAskNewsSDK(client_id=client_id, client_secret=secret, scopes={"news"}) as sdk:
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
                    if not _is_rate_limited_error(exc) and not research_providers.is_asknews_subscription_error(exc):
                        msg = str(exc).lower()
                        if "concurrency limit" not in msg:
                            raise
                    if attempt >= tries:
                        raise
                    await asyncio.sleep(backoff * (10 + 3**attempt))
            if last_exc is not None:
                raise last_exc
    return []


def _run_exa_search_sync(query: str, end_published_date: str | None) -> Any:
    from exa_py import Exa  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

    api_key = os.getenv(EXA_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"Missing Exa API key: {EXA_API_KEY_ENV}")
    client = Exa(api_key=api_key)
    kwargs: dict[str, Any] = {
        "query": query,
        "type": "auto",
        "num_results": 8,
        "contents": {"highlights": True},
    }
    if end_published_date is not None:
        kwargs["end_published_date"] = end_published_date
    return client.search(**kwargs)


async def _call_exa_search(query: str, end_published_date: str | None) -> list[Any]:
    async with _EXA_GLOBAL_SEMAPHORE:
        for attempt in range(len(_EXA_RETRY_DELAYS_S) + 1):
            try:
                response = await asyncio.to_thread(_run_exa_search_sync, query, end_published_date)
                results = _mapping_or_attrs_get(response, "results")
                return list(results) if isinstance(results, list) else []
            except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
                if attempt >= len(_EXA_RETRY_DELAYS_S) or not _is_rate_limited_error(exc):
                    raise
                await asyncio.sleep(_EXA_RETRY_DELAYS_S[attempt])
    return []


async def _read_response_body(resp: aiohttp.ClientResponse, label: str) -> bytes | None:
    return await read_body_capped(resp, max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES, label=label)


async def _fetch_plain(url: str) -> PlainFetchResult:
    if not await resolution_source.is_public_http_url(url):
        return PlainFetchResult(
            status="blocked",
            method="plain",
            text="Blocked non-public or unsupported URL.",
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

                        html = body.decode("utf-8", errors="replace")
                        if any(token in content_type for token in _HTML_CONTENT_TYPE_TOKENS) or "<html" in html.lower():
                            extracted = await asyncio.to_thread(resolution_source._extract_main_text, body, current_url)
                            text = extracted or ""
                            links = _extract_links_from_html(html, current_url)
                            should_render = not text.strip() or len(text.strip()) < _FETCH_MIN_CONTENT_CHARS
                            if not text.strip():
                                text = "Plain fetch returned no extractable text."
                            return PlainFetchResult(
                                status="ok",
                                method="plain",
                                text=text,
                                links=links,
                                url=current_url,
                                content_type=content_type or None,
                                escalate_rendered=should_render,
                            )

                        if any(token in content_type for token in _TEXTUAL_CONTENT_TYPE_TOKENS) or not content_type:
                            text = html.strip()
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
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    return PlainFetchResult(
                        status="error",
                        method="plain",
                        text=f"Fetch error: {type(exc).__name__}: {exc}",
                        links=[],
                        url=current_url,
                    )
    return PlainFetchResult(status="error", method="plain", text="Redirect limit exceeded.", links=[], url=url)


async def _try_rendered_fetch(url: str) -> PlainFetchResult | None:
    try:
        from playwright.async_api import async_playwright  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
        _warn_playwright_unavailable_once(exc)
        return None

    try:
        async with _host_gate(url):
            async with async_playwright() as playwright:
                browser = await playwright.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=BROWSER_HEADERS["User-Agent"],
                    extra_http_headers={key: value for key, value in BROWSER_HEADERS.items() if key != "User-Agent"},
                )

                # Re-apply the SSRF check to EVERY request Chromium makes (main-frame
                # goto, server and client-side redirects, subresources) — Chromium does
                # its own DNS resolution and redirect-following outside the aiohttp
                # FilteringResolver boundary. Residual DNS-rebinding TOCTOU: the route
                # handler's getaddrinfo resolves independently of Chromium's connect,
                # so unlike the aiohttp FilteringResolver this is not airtight; a
                # filtering forward proxy would be — deferred as its own change.
                async def _guard_route(route: Any, request: Any) -> None:
                    if await resolution_source.is_public_http_url(request.url):
                        await route.continue_()
                    else:
                        await route.abort("blockedbyclient")

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


def _run_document_read_sync(url: str, ask: str) -> str:
    from google import genai  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
    from google.genai import types as genai_types  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

    api_key = os.getenv(GOOGLE_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"Missing Google API key: {GOOGLE_API_KEY_ENV}")
    client = genai.Client(api_key=api_key)
    tools: list[Any] = [{"url_context": {}}]
    config = genai_types.GenerateContentConfig(tools=tools)
    response = client.models.generate_content(
        model=GAP_FILL_V2_READER_MODEL,
        contents=f"{_build_document_prompt(ask)}\n\nURL: {url}",
        config=config,
    )
    return _stringify(getattr(response, "text", "")) or ""


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
    if plain.status != "ok":
        return ToolOutcome(content_markdown=plain.text, method=plain.method, status="error")
    if not plain.escalate_rendered:
        return _render_fetch_outcome(url, plain.text, plain.links, "plain", start_char)

    rendered = await _try_rendered_fetch(plain.url)
    if rendered is None:
        return _render_fetch_outcome(url, plain.text, plain.links, "plain", start_char)
    if rendered.method == "document_needed":
        return await read_document(rendered.url, _generic_document_ask(question_topic))
    if rendered.status == "ok" and rendered.text:
        return _render_fetch_outcome(url, rendered.text, rendered.links, "rendered", start_char)
    return _render_fetch_outcome(url, plain.text, plain.links, "plain", start_char)


async def read_document(url: str, ask: str) -> ToolOutcome:
    if not os.getenv(GOOGLE_API_KEY_ENV):
        return _format_fetch_error(f"Google API key is not configured; set {GOOGLE_API_KEY_ENV}.", method="document")
    try:
        text = await asyncio.wait_for(
            asyncio.to_thread(_run_document_read_sync, url, ask), timeout=_READ_DOCUMENT_TIMEOUT_S
        )
    except asyncio.TimeoutError:
        return _format_fetch_error("Document read timed out.", method="document")
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
        return _format_fetch_error(f"Document read failed: {type(exc).__name__}: {exc}", method="document")
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
            # 60s fetch budget + headroom for the rung-3 document auto-escalation
            # (read_document's internal timeout is 60s).
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
