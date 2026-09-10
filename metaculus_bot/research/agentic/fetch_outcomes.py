"""Response classification for the agentic ``fetch`` tool's plain rung.

Everything that turns one HTTP response into a :class:`PlainFetchResult` the fetch ladder
can act on: the content-type / magic-byte sniffers, the outbound-link collector, the
question-platform self-reference refusal, and the per-body-shape outcome builders (a vetted 3xx hop, an HTML
body's trafilatura main text, a raw text/CSV/JSON body). The ``status`` values these return
are load-bearing downstream — only ``"ok"`` grants the loop's ``fetched`` verification tier,
so a page we could not read is ``"empty"`` or ``"blocked"``, never ``"ok"``.

Split out of ``tools.py`` to leave that module the ladder spine (hop loop, rendered rung,
the four tool handlers, registration). The dispatcher choosing between these builders,
``tools._plain_response_outcome``, deliberately stays there: it calls
``tools._read_response_body``, which the suite monkeypatches as a module attribute of
``tools``, and a caller living here would read its own global instead. That is why the
status-set and content-type token constants below have no consumer in this file — the
dispatcher on the other side of that seam reads them.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import aiohttp

from metaculus_bot.constants import GAP_FILL_V2_MIN_CONTENT_CHARS, MANTIC_HOST, METACULUS_HOST
from metaculus_bot.research.fetch_ladder import classify, guard
from metaculus_bot.research.http_fetch import MAX_UNDECODABLE_CHAR_RATIO, meta_refresh_target
from metaculus_bot.research.resolution_body_text import strip_html_tags
from metaculus_bot.research.resolution_chart_data import render_inline_chart_data
from metaculus_bot.research.resolution_fetch_result import PDF_CONTENT_TYPES
from metaculus_bot.research.resolution_url_scan import is_metaculus_self_ref

_FETCH_LINK_CAP = 25
_FETCH_MIN_CONTENT_CHARS = GAP_FILL_V2_MIN_CONTENT_CHARS

_IMAGE_CONTENT_TYPE_PREFIXES = ("image/",)
_RETRYABLE_FETCH_BLOCK_STATUSES = {403, 406, 429}
_TEXTUAL_CONTENT_TYPE_TOKENS = ("text/plain", "text/csv", "application/json")
_HTML_CONTENT_TYPE_TOKENS = ("text/html", "application/xhtml+xml")


@dataclass(slots=True)
class PlainFetchResult:
    status: str
    method: str
    text: str
    links: list[str]
    url: str
    content_type: str | None = None
    escalate_rendered: bool = False
    # Set only by a host's response, never by a refusal this ladder made itself (see the doc).
    http_status: int | None = None


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


def _extract_links_from_html(html: str, base_url: str) -> list[str]:
    parser = _LinkCollector(base_url=base_url, cap=_FETCH_LINK_CAP)
    parser.feed(html)
    parser.close()
    return parser.links


def _content_type_is_document(content_type: str | None) -> bool:
    return _content_type_is_pdf(content_type) or _content_type_is_image(content_type)


def _content_type_is_pdf(content_type: str | None) -> bool:
    """True for a declared PDF, which the ladder reads locally rather than escalating."""
    if not content_type:
        return False
    return any(token in content_type.lower() for token in PDF_CONTENT_TYPES)


def _content_type_is_image(content_type: str | None) -> bool:
    """True for a declared image: the one document shape with no text a local rung could read."""
    if not content_type:
        return False
    return content_type.lower().startswith(_IMAGE_CONTENT_TYPE_PREFIXES)


def _body_is_document(body: bytes) -> bool:
    stripped = body.lstrip()
    if stripped.startswith(b"%PDF-"):
        return True
    return stripped.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a"))


# Named rather than spelled at each site: three producers and two consumers branch on it.
DOCUMENT_NEEDED_METHOD = "document_needed"
_DOCUMENT_NEEDED_MSG = "This URL is a PDF or image — use read_document(url, ask) to read it."


def _document_needed_result(current_url: str, content_type: str) -> PlainFetchResult:
    """The escalate-to-a-document-read outcome, for the three rungs that can reach it.

    ``status="ok"`` with a method the tier map does not carry: nothing has been read yet, so
    this can never be stamped ``fetched``, but it is not a failure either — the fetch handler
    reads the method and escalates.
    """
    return PlainFetchResult(
        status="ok",
        method=DOCUMENT_NEEDED_METHOD,
        text=_DOCUMENT_NEEDED_MSG,
        links=[],
        url=current_url,
        content_type=content_type or None,
    )


_PLATFORM_FETCH_BLOCK_MSG = (
    "Metaculus and Mantic pages are already reflected in the question brief; "
    f"do not fetch {METACULUS_HOST} or {MANTIC_HOST} URLs."
)


def _fetch_plain_url_block(url: str) -> PlainFetchResult | None:
    """Reject a URL the plain rung must not dial, or None when it is fetchable.

    Runs on the caller-supplied URL and again on every redirect hop, so a 3xx cannot walk into a
    target the initial check would have refused. What both rungs it closes would otherwise reach,
    and why the paid reader has to honour it too: docs/agentic_gap_fill.md "Why the question
    platforms' own hosts are refused".
    """
    if is_metaculus_self_ref(url):
        return PlainFetchResult(
            status="blocked",
            method="plain",
            text=_PLATFORM_FETCH_BLOCK_MSG,
            links=[],
            url=url,
        )
    return None


def _non_ok_status_result(status: int, current_url: str, content_type: str) -> PlainFetchResult | None:
    """The terminal result for a non-200, non-redirect response, or None for a 200.

    Split from the dispatcher so the body-shape rungs below it read as one sequence rather
    than as the tail of a status ladder.
    """
    if status in _RETRYABLE_FETCH_BLOCK_STATUSES:
        return PlainFetchResult(
            status="blocked",
            method="plain",
            text=f"Fetch blocked with HTTP {status}.",
            links=[],
            url=current_url,
            content_type=content_type or None,
            http_status=status,
        )
    if status != 200:
        return PlainFetchResult(
            status="error",
            method="plain",
            text=f"Fetch failed with HTTP {status}.",
            links=[],
            url=current_url,
            content_type=content_type or None,
            http_status=status,
        )
    return None


async def _plain_redirect_outcome(
    resp: aiohttp.ClientResponse, current_url: str, content_type: str
) -> PlainFetchResult | str:
    """Vet a 3xx hop: return the next URL to follow, or a terminal blocked/error result."""
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
    return await _vet_hop_target(location, current_url, content_type)


async def _vet_hop_target(target: str, current_url: str, content_type: str) -> PlainFetchResult | str:
    """The absolute next URL for a derived hop (a ``Location`` header or a meta-refresh tag), or
    the terminal refusal it earns — one home for the SSRF + platform re-guard every derived hop
    owes before the redirect loop dials it, mirroring ``guard._vetted_hop_target``."""
    next_url = urljoin(current_url, target)
    if not await guard.is_public_http_url(next_url):
        return PlainFetchResult(
            status="blocked",
            method="plain",
            text="Blocked non-public redirect target.",
            links=[],
            url=next_url,
            content_type=content_type or None,
        )
    blocked = _fetch_plain_url_block(next_url)
    if blocked is not None:
        return PlainFetchResult(
            status=blocked.status,
            method=blocked.method,
            text=blocked.text,
            links=[],
            url=next_url,
            content_type=content_type or None,
        )
    return next_url


async def _plain_html_outcome(
    body: bytes, html: str, content_type: str, current_url: str, *, undecodable_ratio: float
) -> PlainFetchResult | str:
    """Outcome for an HTML body: Tier 1's calibrated extraction plus the page's links.

    The extraction is ``classify._extract_page_text`` (ARIA-role tables rewritten to
    real tables first, default recall then a precision fallback scored by line shape), and the
    page's inline chart configuration is read on every page (``render_inline_chart_data``, led,
    the resolving series lives only there on some dashboards). A page the policy judges chrome
    with no chart block to carry its numbers is ``empty`` and escalates to the rendered rung; a
    ``<meta http-equiv=refresh>`` stub with nothing else is followed as a next hop through this
    ladder's own re-guarded redirect loop (a ``str``), the cdc.gov surveillance-stub rescue.
    """
    extraction = await asyncio.to_thread(classify._extract_page_text, html, body, current_url, undecodable_ratio)
    chart_block = await asyncio.to_thread(render_inline_chart_data, html)
    links = _extract_links_from_html(html, current_url)
    published = "" if extraction.chrome_metric_withheld else (extraction.text or "").strip()
    text = "\n\n".join(part for part in (chart_block, published) if part)
    if text:
        return PlainFetchResult(
            status="ok",
            method="plain",
            text=text,
            links=links,
            url=current_url,
            content_type=content_type or None,
            # A chart block never escalates: a render would replace the client-side series with a DOM lacking it (q43949).
            escalate_rendered=len(text) < _FETCH_MIN_CONTENT_CHARS and not chart_block,
        )
    refresh_target = meta_refresh_target(html)
    if refresh_target is not None:
        return await _vet_hop_target(refresh_target, current_url, content_type)
    # "empty" (not "ok") keeps the ladder escalating to the rendered rung while barring the tier grant on an unread page.
    return PlainFetchResult(
        status="empty",
        method="plain",
        text="Plain fetch returned no extractable text.",
        links=links,
        url=current_url,
        content_type=content_type or None,
        escalate_rendered=True,
    )


def _plain_textual_outcome(
    html: str, undecodable_ratio: float, content_type: str, current_url: str
) -> PlainFetchResult:
    """Outcome for a raw text/CSV/JSON body: tags stripped, no link extraction."""
    if undecodable_ratio > MAX_UNDECODABLE_CHAR_RATIO:
        # Replacement characters rather than the page; `empty` keeps the browser's sniff reachable.
        return PlainFetchResult(
            status="empty",
            method="plain",
            text="Plain fetch could not decode the body as text.",
            links=[],
            url=current_url,
            content_type=content_type or None,
            escalate_rendered=True,
        )
    # A Datawrapper poll CSV measured 69% `<a href=...>` markup, which buys tags instead of rows.
    text = strip_html_tags(html).strip()
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
