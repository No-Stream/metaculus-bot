"""Response classification for the agentic ``fetch`` tool's plain rung.

Everything that turns one HTTP response into a :class:`PlainFetchResult` the fetch ladder
can act on: the content-type / magic-byte sniffers, the outbound-link collector, the
metaculus.com refusal, and the per-body-shape outcome builders (a vetted 3xx hop, an HTML
body's trafilatura main text, a raw text/CSV/JSON body). The ``status`` values these return
are load-bearing downstream — only ``"ok"`` grants the loop's ``fetched`` verification tier,
so a page we could not read is ``"empty"`` or ``"blocked"``, never ``"ok"``.

``matched_throttle_phrase`` lives here for the same reason even though its caller is the
fetch handler rather than this module's dispatcher: recognising a host's rate-limit
interstitial is the same "is this body actually the page" judgment, and it has to run on
the rendered rung's text too, which never passes through here.

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

# A host that is throttling us answers HTTP 200 with a short interstitial in place of the
# page it was asked for, so every status check on the ladder passes and the driver reads the
# refusal as the page's content. Receipt: q45191 (2026-08-10), where three parallel fetches
# of ogimet.com daily summaries tripped that host's spacing rule and two came back as a
# 304-char body reading "gsynext: Limit for old data queries exceeded. Permitted a query per
# 20 seconds per IP" under status="ok" — which was then cached and replayed on the driver's
# own retry, so the exact-date reference class it published came to 4 years instead of 6 and
# the forecast under-committed to the winner it had already named.
#
# Detection needs BOTH halves, and the size half is why it is safe: the phrases alone would
# demote a real page that merely discusses rate limits, while a size floor alone would demote
# every legitimately short source (a one-line official statement), which the builders below
# deliberately keep as "ok". Bare "slow down" is left out on purpose — it is ordinary English
# ("growth will slow down") where the rest are throttle idiom, and missing a throttle only
# preserves today's behavior whereas a false positive discards a page we really did read.
# Calibration: the receipt's body is 304 chars (303 stripped, which is what the cap sees).
FETCH_THROTTLE_PAGE_MAX_CHARS = 1200
FETCH_THROTTLE_PHRASES: tuple[str, ...] = (
    "rate limit",
    "rate-limit",
    "ratelimit",
    "limit exceeded",
    "too many requests",
    "query per",
    "queries per",
    "requests per",
    "per second per ip",
    "retry after",
    "try again later",
    "please slow down",
)


def matched_throttle_phrase(text: str) -> str | None:
    """The throttle phrase in ``text`` when it reads as a host's rate-limit interstitial.

    Returns the matched phrase (evidence, so the caller can log WHICH rule fired and the
    list can be retuned on real prod fires) or ``None`` when the body is a page. A body
    longer than :data:`FETCH_THROTTLE_PAGE_MAX_CHARS` is always a page: an interstitial is
    a sentence, and a long page containing throttle vocabulary is a page about throttling.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > FETCH_THROTTLE_PAGE_MAX_CHARS:
        return None
    lowered = stripped.lower()
    return next((phrase for phrase in FETCH_THROTTLE_PHRASES if phrase in lowered), None)


@dataclass(slots=True)
class PlainFetchResult:
    status: str
    method: str
    text: str
    links: list[str]
    url: str
    content_type: str | None = None
    escalate_rendered: bool = False
    # The HTTP status behind a non-200 terminal result (`_non_ok_status_result`); None on a 200 and
    # on every refusal this ladder makes itself. It is what lets the impersonated retry key on a
    # host's 403 alone: `blocked` is also what a non-public URL and a Metaculus self-reference come
    # back as, and handing either of those to a second transport is the bypass the guard prevents.
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
    return any(token in content_type.lower() for token in _PDF_CONTENT_TYPE_TOKENS)


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


# Named rather than spelled at each site: three rungs produce this method and two consumers
# branch on it, and one of them (the local-document ladder) is WRONG if it ever treats the
# placeholder message below as the document's text.
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
