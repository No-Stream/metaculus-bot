"""Response classification for the agentic ``fetch`` tool's plain rung.

Everything that turns one HTTP response into a :class:`PlainFetchResult` the fetch ladder
can act on: the content-type / magic-byte sniffers, the outbound-link collector, the
metaculus.com refusal, and the per-body-shape outcome builders (a vetted 3xx hop, an HTML
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

from metaculus_bot.constants import GAP_FILL_V2_MIN_CONTENT_CHARS
from metaculus_bot.research import resolution_source
from metaculus_bot.research.http_fetch import MAX_UNDECODABLE_CHAR_RATIO

_FETCH_LINK_CAP = 25
_FETCH_MIN_CONTENT_CHARS = GAP_FILL_V2_MIN_CONTENT_CHARS

_PDF_CONTENT_TYPE_TOKENS = ("application/pdf",)
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
    return stripped.startswith((b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a"))


_DOCUMENT_NEEDED_MSG = "This URL is a PDF or image — use read_document(url, ask) to read it."

_METACULUS_FETCH_BLOCK_MSG = (
    "Metaculus pages are already reflected in the question brief; do not fetch metaculus.com URLs."
)


def _fetch_plain_url_block(url: str) -> PlainFetchResult | None:
    """Reject a URL the plain rung must not dial, or None when it is fetchable.

    Runs on the caller-supplied URL and again on every redirect hop, so a 3xx
    cannot walk into a target the initial check would have refused.
    """
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


async def _plain_html_outcome(body: bytes, html: str, content_type: str, current_url: str) -> PlainFetchResult:
    """Outcome for an HTML body: trafilatura main text plus the page's links."""
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


def _plain_textual_outcome(
    html: str, undecodable_ratio: float, content_type: str, current_url: str
) -> PlainFetchResult:
    """Outcome for a raw text/CSV/JSON body: tags stripped, no link extraction."""
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
