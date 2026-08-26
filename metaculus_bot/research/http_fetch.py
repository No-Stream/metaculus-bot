"""Shared aiohttp HTTP utilities for research providers.

Right-sized extraction (2026-07 resolution-source plan): only the genuinely
generic pieces live here — session construction, a size-capped body read, and
provider-agnostic HTML/URL helpers (the Datawrapper embed scan below, shared
so the resolution-source Tier-2 hop and any future agentic-fetch integration
can't drift on the route). Retry/backoff logic stays provider-private
(prediction_market's is JSON-API shaped; the resolution-source fetcher
deliberately does no retries).
"""

from __future__ import annotations

import codecs
import html as html_entities
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

import aiohttp
import aiohttp.abc
import aiohttp.resolver

logger = logging.getLogger(__name__)

# Sentinel used only as default for AddressFamily; referenced in FilteringResolver.resolve.
_DEFAULT_FAMILY: socket.AddressFamily = socket.AF_INET


IpAddr = ipaddress.IPv4Address | ipaddress.IPv6Address


class FilteringResolver(aiohttp.abc.AbstractResolver):
    """DNS resolver that vets each resolved IP against a caller-supplied predicate.

    Motivation: :func:`is_public_http_url` resolves the target host and rejects
    on a private-IP hit, but ``aiohttp.TCPConnector`` performs its OWN
    resolution at connect time. A DNS-rebinding server that returned a public
    IP to the preflight and a private IP to the connect would slip past the
    guard (classic TOCTOU). This resolver runs at connect time — the same
    layer as aiohttp's DNS cache — so every IP actually dialed has been vetted
    and any redirect hop is re-resolved through the same filter.

    ``disallow(ip)`` returns True to REJECT an address. Callers pass e.g.
    ``_ip_is_disallowed`` from ``resolution_source.py``.

    If every resolved address is filtered out, ``resolve`` raises ``OSError``
    (mirroring how ``getaddrinfo``-based resolvers surface unusable results,
    so the fetch layer's existing except-clause catches it uniformly).
    """

    def __init__(
        self,
        *,
        disallow: Callable[[IpAddr], bool],
        inner: aiohttp.abc.AbstractResolver | None = None,
    ) -> None:
        self._disallow = disallow
        self._inner: aiohttp.abc.AbstractResolver = inner or aiohttp.resolver.ThreadedResolver()

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = _DEFAULT_FAMILY,
    ) -> list[Any]:
        raw = await self._inner.resolve(host, port, family)
        survivors: list[Any] = []
        for entry in raw:
            ip_str = entry["host"]
            try:
                parsed = ipaddress.ip_address(ip_str)
            except ValueError:
                # Non-parseable address string — treat conservatively as disallowed.
                continue
            if not self._disallow(parsed):
                survivors.append(entry)
        if not survivors:
            raise OSError(f"all resolved addresses disallowed for {host}")
        return survivors

    async def close(self) -> None:
        await self._inner.close()


# Shared redirect policy for the SSRF-guarded fetchers (resolution_source and
# research.agentic.tools) — both follow redirects manually so each hop can be
# re-guarded, and both must follow the SAME policy or the two fetchers drift
# for the same URL. Real-world sources chain at most 1-2 hops; 5 leaves slack
# for tracker redirects while keeping the per-hop re-guard cost bounded.
MAX_REDIRECTS: int = 5
REDIRECT_STATUSES: frozenset[int] = frozenset({301, 302, 303, 307, 308})


# Safari-like UA + full Accept / Accept-Language / Accept-Encoding.
# FINDINGS (resolution_source_probe): this exact header set recovered
# 6 extra sources vs Chrome-UA-only (38/50 vs 32/50).
BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.5 Safari/605.1.15"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # Advertise only codecs the runtime can decode: aiohttp needs the `brotli`
    # package for `br` (HAS_BROTLI=False here — not a project dep). If we
    # advertised `br` anyway, a Brotli-preferring server would send it and
    # aiohttp would raise ClientResponseError on decode, silently dropping the
    # source. Servers fall back to gzip/deflate cleanly.
    "Accept-Encoding": "gzip, deflate",
}


def build_session(
    *,
    timeout_s: float,
    connector_limit: int = 20,
    headers: dict[str, str] | None = None,
    resolver: aiohttp.abc.AbstractResolver | None = None,
) -> aiohttp.ClientSession:
    """Construct a fresh aiohttp session with total + sock_read timeouts and a connection cap.

    ``headers=None`` (the default) adds no session-level headers — prediction_market's
    JSON-API calls rely on that; the resolution-source fetcher passes BROWSER_HEADERS.

    ``resolver=None`` (the default) uses aiohttp's built-in ThreadedResolver.
    Callers that need to vet resolved IPs (SSRF-sensitive fetchers) pass a
    :class:`FilteringResolver` so aiohttp's own connect-time DNS lookup goes
    through the same predicate as the preflight guard — closing the classic
    DNS-rebinding TOCTOU.
    """
    timeout = aiohttp.ClientTimeout(total=timeout_s, sock_read=timeout_s)
    connector_kwargs: dict[str, Any] = {"limit": connector_limit}
    if resolver is not None:
        connector_kwargs["resolver"] = resolver
    connector = aiohttp.TCPConnector(**connector_kwargs)
    return aiohttp.ClientSession(timeout=timeout, connector=connector, headers=headers)


_READ_CHUNK_BYTES = 65536


async def read_body_capped(resp: Any, *, max_bytes: int, label: str) -> bytes | None:
    """Read a response body incrementally, rejecting bodies over ``max_bytes``.

    Streams DECOMPRESSED bytes via ``resp.content.iter_chunked`` (aiohttp's
    ``DeflateBuffer`` feeds the same stream ``resp.read()`` consumes, so no
    chunked/gzip truncation) and aborts as soon as the running total exceeds
    the cap. The cap therefore bounds peak memory DURING the read — a huge or
    gzip-bombed response from an untrusted URL can't buffer fully before the
    guard fires.

    Returns None on an oversized body (logged at WARNING with ``label``).
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in resp.content.iter_chunked(_READ_CHUNK_BYTES):
        total += len(chunk)
        if total > max_bytes:
            logger.warning(f"{label} response too large ({total} bytes read > {max_bytes}); dropping")
            return None
        chunks.append(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Text decoding (raw-body branches: JSON / text/plain / text/csv / CSV datasets)
# ---------------------------------------------------------------------------
#
# A blanket `body.decode("utf-8", errors="replace")` turns a UTF-16 or
# Windows-1252 body into mojibake that still type-checks as text, and the
# caller then renders `0�.�4�2�` to a forecaster as grading evidence. The
# response ALREADY carries the two facts needed to decode it properly — a BOM
# and/or a declared charset — so both are honored, and whatever survives is
# scored so a caller can refuse a body it could not decode at all.

_CHARSET_RE = re.compile(r"charset\s*=\s*\"?([\w.:+-]+)\"?", re.IGNORECASE)

# BOM -> codec. UTF-32-LE (`ff fe 00 00`) must be tested BEFORE UTF-16-LE
# (`ff fe`), which is a prefix of it. The `utf-16` / `utf-32` / `utf-8-sig`
# codecs consume the BOM themselves, so the decoded text carries no U+FEFF.
_BOM_CODECS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)

# Above this share of undecodable characters, the decode failed rather than the
# text being slightly dirty, and the caller should refuse the body. The two
# regimes are far apart (measured on synthetic bodies; the cases are pinned in
# `tests/test_http_fetch.py`): a body decoded with the wrong codec runs ~0.5 —
# every second byte of BOM-less UTF-16
# ASCII is a NUL, and a UTF-16 BOM read as UTF-8 is two U+FFFDs up front — while
# real text carrying a few mis-encoded punctuation marks runs 0.03 on a 37-char
# line and far less on a real page. 0.10 sits in the empty gap between them, so
# it never costs us a page over one bad smart quote.
MAX_UNDECODABLE_CHAR_RATIO = 0.10


def _bom_codec(body: bytes) -> str | None:
    for bom, codec in _BOM_CODECS:
        if body.startswith(bom):
            return codec
    return None


def _declared_charset(content_type: str | None) -> str | None:
    if not content_type:
        return None
    match = _CHARSET_RE.search(content_type)
    return match.group(1) if match else None


def undecodable_char_ratio(text: str) -> float:
    """Share of ``text`` that carries no content because the decode failed.

    Counts U+FFFD (the replacement char `errors="replace"` substitutes for bytes
    the codec rejected) AND NUL. Both are needed: high bytes from an undeclared
    Windows-1252 body surface as U+FFFD, while BOM-less UTF-16 ASCII decodes as
    valid-but-meaningless NUL-interleaved text with zero replacement chars.
    """
    if not text:
        return 0.0
    return (text.count("�") + text.count("\x00")) / len(text)


def decode_text_body(body: bytes, content_type: str | None) -> tuple[str, float]:
    """Decode a raw body to ``(text, undecodable_char_ratio)``.

    Codec precedence: BOM (the body's own self-description, and the shape Excel
    exports CSV in), then the response's declared ``charset=``, then UTF-8.
    ``errors="replace"`` throughout — a partly-undecodable body still has to
    produce text, because the ratio is how the caller decides whether to keep it.
    An unknown declared charset falls back to UTF-8 rather than raising: a
    typo'd charset label is not a reason to lose the body.
    """
    codec = _bom_codec(body) or _declared_charset(content_type) or "utf-8"
    try:
        text = body.decode(codec, errors="replace")
    except LookupError:
        logger.info(f"unknown charset {codec!r} declared; decoding as utf-8")
        text = body.decode("utf-8", errors="replace")
    return text, undecodable_char_ratio(text)


# ---------------------------------------------------------------------------
# Datawrapper embed detection (resolution-source Tier-2 second hop)
# ---------------------------------------------------------------------------
#
# Poll-tracker pages routinely lock their daily series inside a Datawrapper
# iframe, which trafilatura drops at every setting — the resolving data never
# reaches the research bundle (qids 44858 / 44841, 2026-08-24 dossiers). The
# helpers here find the embeds in RAW page HTML and build the one URL that is
# safe to fetch.
#
# Route mechanism (live-verified against natesilver.net trackers 2026-08-25):
#
# - Page HTML embeds `datawrapper.dwcdn.net/<chart_id>/<version>/` with the
#   version PINNED at page-authoring time. The pinned version goes stale and
#   `<pinned>/dataset.csv` keeps serving its old snapshot with HTTP 200
#   (observed 5 and 14 MONTHS stale on the two real trackers). Reaching the
#   live version from a pinned one takes a 26+ hop chain of meta-refresh/JS
#   stubs that plain HTTP cannot follow. The versioned dataset route must
#   therefore NEVER be derived from page HTML.
# - `static.dwcdn.net/data/<chart_id>.csv` is version-free and always serves
#   the latest published dataset (Last-Modified refreshes on each republish;
#   observed hourly on live trackers). It is the URL behind the chart's own
#   "Get the data" affordance — the artifact resolution fine print names.

# Chart ids are 5 alphanumeric chars. Matches both host forms
# (`datawrapper.dwcdn.net/<id>/…`, `static.dwcdn.net/data/<id>.csv`) and
# tolerates JSON-escaped slashes (`\/`) — Substack pages carry the embed URL
# inside an HTML-escaped JSON `data-attrs` attribute.
_DATAWRAPPER_CHART_ID_RE = re.compile(
    r"(?:datawrapper|static)\.dwcdn\.net\\?/(?:data\\?/)?([A-Za-z0-9]{5})(?![A-Za-z0-9])"
)

# Title forms, in the two shapes observed in the wild:
# - JSON attrs (Substack `data-attrs`, HTML-escaped or plain):
#   `&quot;title&quot;:&quot;Do Americans …&quot;` — sits AFTER the URL. JSON
#   strings are double-quote-delimited only; a bare `'` inside the value
#   (e.g. "Trump's net approval …") is title text, not a terminator.
# - Datawrapper's own responsive embed: `<iframe title="…" … src="…">` —
#   sits BEFORE the URL, `=`-separated. The closing quote must match the
#   opening one, for the same apostrophe reason.
_DW_TITLE_JSON_RE = re.compile(r"(?:&quot;|\")title(?:&quot;|\")\s*:\s*(?:&quot;|\")(.{1,300}?)(?:&quot;|\")")
_DW_TITLE_ATTR_RE = re.compile(r"title\s*=\s*(?:\"([^\"]{1,300})\"|'([^']{1,300})')")

# Window sizes bracket the observed layouts: the Substack JSON title lands
# ~380 chars after the URL (two thumbnail URLs in between); an iframe's
# `title=` attribute sits within the same tag just before `src=`.
_DW_TITLE_FORWARD_WINDOW = 700
_DW_TITLE_BACKWARD_WINDOW = 300


@dataclass(frozen=True)
class DatawrapperChartRef:
    """A Datawrapper chart referenced by a fetched page, in document order."""

    chart_id: str
    title: str | None


def _within_one_tag(html_text: str, start: int, end: int) -> bool:
    """True when ``html_text[start:end]`` crosses no tag boundary.

    The proximity window alone cannot tell "this iframe's own title" from "a share
    button's title 200 chars earlier". A `>` or `<` between the two means they live on
    different elements, so the candidate is not this chart's title.
    """
    return "<" not in html_text[start:end] and ">" not in html_text[start:end]


def extract_datawrapper_charts(html_text: str) -> list[DatawrapperChartRef]:
    """Scan RAW page HTML for Datawrapper chart embeds.

    Must run on the raw HTML, not extracted main text — trafilatura emits
    neither iframe ``src`` attributes nor embed-script URLs at any setting
    (verified against the live tracker pages, 2026-08-24 dossier work).

    Returns first-seen-deduped refs in document order (tracker pages put the
    hero/resolving chart first). Titles are best-effort: the JSON-attrs form
    is searched forward of the URL (bounded at the next embed so a titleless
    chart can't steal its neighbour's), the iframe-attribute form backward.

    Both forms are additionally anchored to the URL's own TAG: a candidate with a
    tag boundary between it and the chart URL is rejected. Without that, the window
    alone let an unrelated `title="Share on X"` or an `og:title` on a neighbouring
    element render as the chart's identity inside the Tier-2 lead — the lead names
    the chart it is serving data for, so a borrowed title is a false claim about
    which series the forecaster is reading. Both real layouts sit inside one tag
    (the Substack `data-attrs` JSON blob, and an iframe's own `title=`), so the
    anchor costs nothing on them.
    """
    if not html_text:
        return []
    matches = list(_DATAWRAPPER_CHART_ID_RE.finditer(html_text))
    charts: list[DatawrapperChartRef] = []
    seen: set[str] = set()
    for i, m in enumerate(matches):
        chart_id = m.group(1)
        if chart_id in seen:
            continue
        seen.add(chart_id)

        forward_end = m.end() + _DW_TITLE_FORWARD_WINDOW
        if i + 1 < len(matches):
            forward_end = min(forward_end, matches[i + 1].start())
        raw_title: str | None = None
        json_match = _DW_TITLE_JSON_RE.search(html_text, m.end(), forward_end)
        if json_match is not None and _within_one_tag(html_text, m.end(), json_match.start()):
            raw_title = json_match.group(1)
        else:
            backward_start = max(0, m.start() - _DW_TITLE_BACKWARD_WINDOW)
            if i > 0:
                backward_start = max(backward_start, matches[i - 1].end())
            attr_matches = [
                match
                for match in _DW_TITLE_ATTR_RE.finditer(html_text, backward_start, m.start())
                if _within_one_tag(html_text, match.end(), m.start())
            ]
            if attr_matches:
                # The attr pattern alternates on quote style; exactly one group is set.
                raw_title = attr_matches[-1].group(1) or attr_matches[-1].group(2)

        title = html_entities.unescape(raw_title) if raw_title else None
        charts.append(DatawrapperChartRef(chart_id=chart_id, title=title))
    return charts


_DATAWRAPPER_CHART_ID_SHAPE = re.compile(r"[A-Za-z0-9]{5}\Z")


def datawrapper_live_data_url(chart_id: str) -> str:
    """The version-free, always-latest dataset URL for a Datawrapper chart.

    This is the ONLY dataset route callers may fetch (see the mechanism note
    above): the versioned `datawrapper.dwcdn.net/<id>/<version>/dataset.csv`
    form serves the pinned — potentially months-stale — snapshot when the
    version comes from page HTML. Rejects ids that don't match the 5-char
    shape so a crafted page can't turn the template into path traversal.
    """
    if not _DATAWRAPPER_CHART_ID_SHAPE.match(chart_id):
        raise ValueError(f"not a Datawrapper chart id: {chart_id!r}")
    return f"https://static.dwcdn.net/data/{chart_id}.csv"


def parse_http_last_modified(value: str) -> datetime | None:
    """Parse an HTTP ``Last-Modified`` header into an aware UTC datetime.

    Returns None on a malformed value — callers treat unverifiable freshness
    the same as stale (the serve-live-or-nothing rule).
    """
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


# Enough to carry an upstream JSON error object or the title of an HTML error page.
ERROR_SNIPPET_BYTES: int = 2048


async def read_body_snippet(resp: Any, *, max_bytes: int = ERROR_SNIPPET_BYTES) -> str:
    """Decode at most ``max_bytes`` of a body, for a log line on a non-200.

    ``resp.text()`` reads and DECOMPRESSES the whole body before the caller slices
    it, so a CDN 429/502 serving a multi-megabyte HTML error page (or a gzip bomb)
    defeats the very memory ceiling the callers enforce on their success path.
    Reads bounded chunks and stops, leaving the rest of the body unread.
    """
    buf = bytearray()
    async for chunk in resp.content.iter_chunked(max_bytes):
        buf += chunk
        if len(buf) >= max_bytes:
            break
    return bytes(buf[:max_bytes]).decode("utf-8", errors="replace")
