# SMELL-EXEMPT-monolithic-file-loc: three documented layers (pure helpers /
# network / factory); splitting the Tier-2 hop out would force FetchResult +
# the SSRF guard into a shared module — a cross-cutting refactor for its own PR.
"""Resolution-source fetcher: Tier-1 cited pages + a Tier-2 Datawrapper hop.

Fetches the URL(s) explicitly cited in a Metaculus question's resolution
criteria (or fine print), extracts main content with trafilatura, and returns
a compact markdown section that every forecaster reads as the ground truth
the question will be graded against.

Tier 1 is plain HTTP with browser-like headers, no LLM calls, no retries.
Sites behind JS walls / heavy anti-bot remain deferred (see `FetchStatus` —
`blocked` / `js_wall` results are retained in the returned list as that seam).

Tier 2 (2026-08, qids 44858/44841): when a fetched page's RAW HTML embeds a
Datawrapper chart, fetch that chart's live "Get the data" CSV — poll trackers
lock their resolving daily series inside these iframes, which trafilatura
drops at every setting. The hop uses ONLY the version-free
`static.dwcdn.net/data/<chart_id>.csv` route: the page-pinned
`datawrapper.dwcdn.net/<id>/<version>/dataset.csv` form serves months-stale
snapshots as HTTP 200 (the naive fix the 2026-08-24 verifications refuted).
A `Last-Modified` freshness guard withholds any dataset older than
`RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS` (or undatable) as `stale_data`
rather than serving stale data as live.

Success means CONTENT, on every raw-body branch (Tier-1 JSON/text/CSV and the
Tier-2 dataset alike): a body that is empty, undecodable, or — for a dataset —
not row-shaped gets a failure status via `vacuous_body_status`, never
`success`. An empty 200 body used to render an empty section under the "primary
grading evidence" caveat, suppress the all-failed notice for its siblings, and
report `ok` to provider diagnostics.

Design anchors:

- 2026-07-08 feasibility probe found 75% of questions cite an explicit source
  URL and ~62.5% of them are recoverable by a plain browser-headers fetch.
- Extraction is trafilatura in a thread (`asyncio.to_thread`) — the parse is
  CPU-bound sync C code.
- Per-host politeness: one `asyncio.Semaphore(1)` per netloc, acquired around
  each redirect hop's GET and keyed on THAT hop's host — so chains converging
  on one final host still serialize there. Distinct hosts run concurrently up
  to the connector limit.
- Char caps apply to RAW (non-LLM-processed) content only; the LLM-emitted
  research bundle is never truncated (see the resolution-source plan).
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

import aiohttp
import trafilatura
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    RESOLUTION_SOURCE_DATAWRAPPER_HOP_WALL_MARGIN_S,
    RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS,
    RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS,
    RESOLUTION_SOURCE_DATAWRAPPER_MIN_HOP_BUDGET_S,
    RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS,
    RESOLUTION_SOURCE_ENABLED_ENV,
    RESOLUTION_SOURCE_GLOBAL_CONCURRENCY,
    RESOLUTION_SOURCE_HTTP_TIMEOUT,
    RESOLUTION_SOURCE_JS_WALL_MIN_CHARS,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
    RESOLUTION_SOURCE_MAX_URLS,
    RESOLUTION_SOURCE_PER_URL_MAX_CHARS,
    RESOLUTION_SOURCE_TOTAL_MAX_CHARS,
    RESOLUTION_SOURCE_WALL_TIMEOUT,
    env_flag_enabled,
)
from metaculus_bot.research.http_fetch import (
    BROWSER_HEADERS,
    MAX_REDIRECTS,
    MAX_UNDECODABLE_CHAR_RATIO,
    REDIRECT_STATUSES,
    DatawrapperChartRef,
    FilteringResolver,
    build_session,
    datawrapper_live_data_url,
    decode_text_body,
    extract_datawrapper_charts,
    parse_http_last_modified,
    read_body_capped,
)
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.raw_log import record_raw_research


def _make_filtering_resolver() -> FilteringResolver:
    """Build a FilteringResolver seeded with :func:`_ip_is_disallowed`.

    Hoisted to module scope (from an inline lambda in ``_get_session``) so
    tests can construct one directly and to keep the import-usage adjacency
    that survives ruff's unused-import auto-format.
    """
    return FilteringResolver(disallow=_ip_is_disallowed)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------
#
# URLs enter this module from question resolution_criteria / fine_print — user-
# authored fields that anyone can craft. Fetches run from CI runners,
# where hitting http://169.254.169.254/latest/meta-data/ (or any RFC1918 host,
# any private IP, ::1, fe80::/10, etc.) would exfiltrate instance identity into
# the research prompt AND into the public Metaculus comment. Legitimate
# resolution sources are always public websites, so a blanket public-only
# constraint costs zero functionality.
#
# Kept local to this module. `http_fetch.py` is shared with the prediction-
# market provider, which only hits a fixed allow-list of API hosts (Polymarket
# Gamma, Kalshi, Manifold) and doesn't need this. If a third caller lands in
# http_fetch that also takes user-supplied URLs, hoist this guard there.
#
# Redirect policy (hop cap + 3xx status set) lives in http_fetch.py as
# MAX_REDIRECTS / REDIRECT_STATUSES, shared with research.agentic.tools so the
# two SSRF-guarded fetchers can't drift.


def _ip_is_disallowed(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Reject every non-globally-routable IP category.

    The explicit predicates keep review clarity for the obvious classes
    (private / loopback / link-local / reserved / multicast / unspecified).
    The `not ip.is_global` clause is the catch-all — it covers ranges the
    explicit list misses, most notably CGNAT / shared address space
    100.64.0.0/10 (which is not `is_private` on ipaddress) and IPv4-mapped
    IPv6 forms of private ranges.
    """
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or not ip.is_global
    )


async def is_public_http_url(url: str) -> bool:
    """Return True iff ``url`` is safe to fetch from CI (public HTTP(S) only).

    Rejects: non-http(s) schemes, URLs carrying userinfo, IP-literal hosts that
    fall in any non-global range (private / loopback / link-local / reserved /
    multicast / unspecified), and hostnames whose DNS resolution surfaces ANY
    disallowed IP.

    This is the FAST-PATH observability guard: it lets us emit ``ssrf_blocked``
    without ever opening a session. It is NOT the DNS-rebinding trust
    boundary — the resolver aiohttp uses at connect time is (see
    :func:`_get_session` and :class:`FilteringResolver`). A rebinding server
    that returned a public IP here and a private IP to the connect-time
    resolver would still be rejected there.

    DNS failure -> False (unfetchable; would fail the fetch anyway, and we want
    the caller to uniformly emit an ``ssrf_blocked`` result).

    Async because DNS goes through ``asyncio.to_thread(socket.getaddrinfo, ...)``
    to avoid blocking the event loop.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False

    if parsed.scheme.lower() not in ("http", "https"):
        return False

    # Userinfo defeats hostname-based trust: `https://trusted@169.254.169.254/`
    # renders as if targeting `trusted` but actually hits the IMDS.
    if parsed.username is not None or parsed.password is not None:
        return False

    # `.hostname` strips userinfo, port, and IPv6 brackets, and lowercases —
    # harmless here: both ip_address() and getaddrinfo() are case-insensitive.
    host = parsed.hostname or ""
    if not host:
        return False

    # IP-literal branch: no DNS needed. Try IPv4 first, then IPv6.
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    if ip is not None:
        return not _ip_is_disallowed(ip)

    return await _every_resolved_address_is_public(host)


async def _every_resolved_address_is_public(host: str) -> bool:
    """True iff ``host`` resolves and EVERY resolved address is publicly routable.

    Resolves via getaddrinfo off the event loop. Any disallowed address rejects the
    whole hostname (DNS rebinding defense); a resolution failure rejects too, since
    an unfetchable host should reach the caller as one uniform ``ssrf_blocked``.
    """
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None)
    except (socket.gaierror, OSError):
        return False

    for info in infos:
        # sockaddr shape: IPv4 = (ip, port); IPv6 = (ip, port, flowinfo, scopeid).
        sockaddr = info[4] if len(info) >= 5 else None
        if not sockaddr:
            return False
        try:
            resolved = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            return False
        if _ip_is_disallowed(resolved):
            return False

    return True


# ---------------------------------------------------------------------------
# Status enum + result dataclass (the Tier-2 seam)
# ---------------------------------------------------------------------------


# How far ahead of our clock a dataset's `Last-Modified` may sit before the freshness
# guard treats it as unusable rather than as freshest-possible. Small on purpose: this
# tolerates ordinary CDN/host clock skew and nothing more, because the only thing a
# future date can mean past that is a broken clock or a misparse — and the lead the
# stamp authorizes asserts a publication date to forecasters.
_DATAWRAPPER_CLOCK_SKEW_TOLERANCE = timedelta(hours=6)

# `stale_data` is Tier-2-only: the Datawrapper hop reached a dataset whose
# Last-Modified is outside the freshness window — older than the bound, missing,
# unparseable, or implausibly far in the FUTURE — withheld rather than served as live.
#
# `empty_body` is the 200-with-nothing-in-it case: a body that is empty or
# whitespace-only carries no information, so calling it `success` published an
# empty "primary grading evidence" section, suppressed the all-failed notice for
# every sibling URL, and reported `ok` to provider diagnostics. It is a FAILURE
# status for the same reason the HTML branch treats an empty extraction as
# `js_wall`: content is what makes a fetch a success.
FetchStatus = Literal[
    "success",
    "blocked",
    "not_found",
    "js_wall",
    "error",
    "unsupported_type",
    "ssrf_blocked",
    "stale_data",
    "empty_body",
]


@dataclass
class FetchResult:
    url: str
    status: FetchStatus
    text: str  # extracted + truncated; "" unless status == "success"
    http_status: int | None
    content_type: str | None
    # Charts seen in a fetched page's raw HTML (set on Tier-1 HTML results,
    # including js_wall pages — a JS-walled page still exposes its embeds).
    datawrapper_charts: list[DatawrapperChartRef] = field(default_factory=list)
    # Provenance for Tier-2 dataset results (None on ordinary page fetches).
    chart_id: str | None = None
    chart_title: str | None = None
    parent_url: str | None = None
    data_last_modified: str | None = None  # ISO-8601; None when the header was missing/unparseable

    def __post_init__(self) -> None:
        """Enforce the ``text`` invariant the field comment states.

        A `success` carrying blank text is the defect this guard exists for: the
        JSON/text/CSV branch used to ship an empty 200 body as `success`, which
        rendered an empty section under the "primary grading evidence" caveat,
        suppressed the all-failed notice, and told provider diagnostics `ok`.
        Every construction site now decides vacuity BEFORE it picks a status
        (:func:`vacuous_body_status`), so a blank success can only come from a
        future edit — and it should crash the provider (the orchestrator turns a
        provider exception into one `errored` provider result) rather than quietly
        publish a hole in the grading evidence.
        """
        if self.status == "success" and not self.text.strip():
            raise ValueError(f"FetchResult(status='success') with blank text for {self.url}")


# ---------------------------------------------------------------------------
# Pure helpers — no I/O
# ---------------------------------------------------------------------------


# Metaculus-injected markdown escapes: `\_`, `\.`, `\&`, `\-`, `\#`, `\(`, `\)`.
# FINDINGS: 3.4% of URLs carry these; one flips 404→success once unescaped.
_MARKDOWN_ESCAPE_RE = re.compile(r"\\([_&.\-#()])")

# Markdown link: [label](https://...) — capture only the URL.
_MARKDOWN_LINK_URL_RE = re.compile(r"\[[^\]]*\]\((https?://[^)\s]+)\)")

# Bare URL — stops at whitespace and common closers.
_BARE_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]]+")

# Trailing punctuation to strip from an extracted URL.
_TRAILING_PUNCT = ".,;:)]}>\"'"


def strip_markdown_escapes(url: str) -> str:
    """Remove markdown backslash escapes Metaculus injects into rendered URLs."""
    return _MARKDOWN_ESCAPE_RE.sub(r"\1", url)


def extract_source_urls(text: str) -> list[str]:
    """Extract http(s) URLs from ``text``.

    Handles markdown links ``[label](https://…)`` and bare URLs. Strips trailing
    punctuation, applies backslash-unescape, dedupes preserving order (case-
    insensitive scheme+host; exact path and query — query params stay in the
    key because we may need them, e.g. for FRED graph_id; fragments are
    excluded because they're never sent over HTTP). Returns the FULL deduped
    list — the ``RESOLUTION_SOURCE_MAX_URLS`` cap is applied downstream by
    :func:`select_fetchable_urls`, AFTER the self-ref/FRED/Yahoo skip filter,
    so a run of leading self-refs doesn't starve the real sources out of the
    fetch budget.
    """
    if not text:
        return []

    # Collect (start_pos, url) from both regex families so the final order
    # tracks appearance in the source text — not extraction order. Markdown
    # link URLs are typically ALSO matched by the bare-URL regex (its match
    # sits inside the `[label](URL)` parens); the earlier position wins after
    # sort, and dedup drops the duplicate.
    positioned: list[tuple[int, str]] = []
    for match in _MARKDOWN_LINK_URL_RE.finditer(text):
        # Anchor at the URL group's start, not the link's start, so a
        # markdown link and a same-position bare URL rank identically.
        positioned.append((match.start(1), match.group(1)))
    for match in _BARE_URL_RE.finditer(text):
        positioned.append((match.start(), match.group(0)))
    positioned.sort(key=lambda pair: pair[0])

    cleaned: list[str] = []
    for _pos, raw in positioned:
        u = raw
        # Strip trailing punctuation (may repeat: "foo.,").
        while u and u[-1] in _TRAILING_PUNCT:
            u = u[:-1]
        u = strip_markdown_escapes(u)
        if not u.lower().startswith(("http://", "https://")):
            continue
        cleaned.append(u)

    # Dedup preserving order (first-seen URL string wins). Case-insensitive
    # scheme+netloc; exact path/query. Fragments are excluded — they're never
    # sent over HTTP, so URLs differing only by fragment are the same fetch.
    # A bare host and a bare host + "/" collapse to one entry (real questions
    # cite both forms of the same root page — observed on childmortality.org
    # in the 2026-07-09 smoke test, burning a duplicate fetch slot).
    seen: set[str] = set()
    deduped: list[str] = []
    for u in cleaned:
        try:
            parsed = urlparse(u)
        except ValueError:
            continue
        key = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{parsed.path or '/'}?{parsed.query}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(u)
    return deduped


def is_metaculus_self_ref(url: str) -> bool:
    """A URL that points back at Metaculus is a self-reference (no new info).

    Uses ``.hostname`` (not ``.netloc``) so a port or userinfo can't slip a
    metaculus URL past the check — ``.netloc`` keeps ``:443`` / ``user@``, which
    would defeat the exact-host and suffix comparisons below.
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host == "metaculus.com" or host.endswith(".metaculus.com")


def is_fred_url(url: str) -> bool:
    """FRED series URLs are already served by the financial-data provider."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    return host == "fred.stlouisfed.org"


def is_yahoo_ticker_url(url: str) -> bool:
    """Yahoo Finance `/quote/…` URLs are yfinance-served; skip.

    Generic Yahoo article / news URLs remain fetchable — only the ticker
    quote pages overlap with the financial-data provider.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return (parsed.hostname or "").lower() == "finance.yahoo.com" and parsed.path.startswith("/quote/")


def select_fetchable_urls(criteria: str | None, fine_print: str | None) -> list[str]:
    """Compose the fetchable URL list from a question's resolution criteria + fine print.

    Skips self-refs (metaculus.com), FRED, and Yahoo ticker URLs — those either
    add no new info or are covered by another provider. Caps at
    ``RESOLUTION_SOURCE_MAX_URLS`` AFTER the skip filter so a run of leading
    self-refs / FRED / Yahoo URLs (Metaculus questions often list the question
    page first) doesn't starve the real sources out of the fetch budget.
    """
    combined = f"{criteria or ''}\n\n{fine_print or ''}"
    urls = extract_source_urls(combined)
    filtered = [u for u in urls if not (is_metaculus_self_ref(u) or is_fred_url(u) or is_yahoo_ticker_url(u))]
    return filtered[:RESOLUTION_SOURCE_MAX_URLS]


def looks_like_js_wall(text: str) -> bool:
    """A 200 OK whose extracted text is shorter than the JS-wall threshold is a
    strong signal the page needs JS to render — Tier-2 candidate."""
    return len(text.strip()) < RESOLUTION_SOURCE_JS_WALL_MIN_CHARS


# Tag names an allow-list, not `[A-Za-z]+`, and the character after the name
# must be `>`, `/`, or whitespace-then-an-attribute-assignment. Both halves are
# load-bearing: the naive `</?[A-Za-z][^>]*>` form eats a CSV cell reading
# `x <a and y > b` (measured on the 2026-08-26 diagnosis), and requiring an `=`
# in the attribute region is what keeps that same cell out of the allow-listed
# form — `<a and y >` names no attribute. A real tag either closes immediately
# (`<br/>`, `</a>`, `<td>`) or assigns something (`href=`, `style=`).
_HTML_TAG_NAMES: tuple[str, ...] = (
    "a",
    "b",
    "i",
    "u",
    "p",
    "em",
    "strong",
    "span",
    "div",
    "br",
    "td",
    "tr",
    "th",
    "table",
    "font",
    "small",
    "sub",
    "sup",
)
# The pre-`=` attribute region excludes `=` so there is exactly ONE way to reach the
# first delimiter. With `[^<>]*` on both sides of the `=`, a non-matching body (an
# allow-listed lookalike like `<b ` followed by an angle-bracket-free run of URL cells
# carrying query-string `=` signs) backtracks quadratically: 3.4s at 200 KiB, ~35 min
# at the 5 MiB response cap — synchronously on the event loop, wedging the sibling
# fetches past every wall timeout. Same matched language: any match via a later `=`
# is also a match via the first one, since the post-`=` region admits more `=`s.
_HTML_TAG_RE = re.compile(
    r"</?(?:" + "|".join(_HTML_TAG_NAMES) + r")(?:\s*/?>|\s+[^<>=]*=[^<>]*>)",
    re.IGNORECASE,
)

# An anchor whose inner text is empty carries its content in the href (a bare
# link cell), so the href is kept where the inner text would have been.
_EMPTY_ANCHOR_RE = re.compile(
    r"<a\s+[^<>]*href\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s<>]+))[^<>]*>\s*</a\s*>",
    re.IGNORECASE,
)

# Delimiters a Datawrapper "Get the data" export uses. Checked on the HEADER
# line only: a dataset's first row names its columns, so a header with no
# delimiter at all is not the CSV this route is supposed to serve.
_CSV_DELIMITERS: tuple[str, ...] = (",", ";", "\t", "|")


def strip_html_tags(text: str) -> str:
    """Remove allow-listed HTML tags from ``text``, keeping their inner text.

    For the RAW-body branches only (CSV / plain text) — trafilatura owns real
    HTML pages. Datawrapper poll tables embed a styled ``<a href=…>`` per
    pollster row, and 69% of one live tracker's 33k-char CSV was tag markup: at
    that run's actual 2,853-char budget, 9 rows survived with the tags against 30
    with them stripped (measured 2026-08-26). The tags carry nothing a forecaster
    reads — the pollster name inside them is the content — so the budget should
    buy rows.

    A no-op (byte-identical) on any text with no allow-listed tag in it, which
    covers every numeric tracker CSV checked (zero ``<`` characters).
    """
    if "<" not in text:
        return text
    without_bare_links = _EMPTY_ANCHOR_RE.sub(
        lambda m: m.group(1) or m.group(2) or m.group(3) or "",
        text,
    )
    return _HTML_TAG_RE.sub("", without_bare_links)


def looks_like_csv_rows(text: str) -> bool:
    """True when ``text`` is shaped like a dataset: a delimited header + ≥1 row.

    The precondition for asserting a dataset is LIVE. The Tier-2 lead stamps
    ``Dataset published <ts>`` off the ``Last-Modified`` header alone, so without
    this an empty or soft-404 CDN body renders under an authoritative freshness
    claim — the same shape as a venue's manufactured $0.50 price.

    Markup is rejected outright (a body whose first non-blank character is ``<``
    is an error page, not a dataset) because an HTML error page can easily carry
    a comma somewhere and pass the delimiter test.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    header = lines[0].lstrip()
    if header.startswith("<"):
        return False
    return any(delimiter in header for delimiter in _CSV_DELIMITERS)


def vacuous_body_status(text: str, undecodable_ratio: float, *, require_csv_rows: bool) -> FetchStatus | None:
    """The failure status a 200 body earns for carrying no usable content, else None.

    The one place "does this body carry information?" is decided, so every
    ``FetchResult(status="success")`` on a raw-body branch is predicated on
    content rather than on the status line. Three ways a 200 carries nothing:

    - it could not be DECODED (``undecodable_ratio`` above the bound) — mojibake
      like ``0�.�4�2�`` type-checks as text and rendered as grading evidence;
    - it is empty or whitespace-only;
    - (Tier-2 datasets only) it is not row-shaped, so nothing may claim it is the
      chart's live series.
    """
    if undecodable_ratio > MAX_UNDECODABLE_CHAR_RATIO:
        return "unsupported_type"
    if not text.strip():
        return "empty_body"
    if require_csv_rows and not looks_like_csv_rows(text):
        return "unsupported_type"
    return None


def _truncate_with_marker(text: str, cap: int, url: str) -> str:
    """Return ``text`` bounded at ``cap`` chars; on truncation, append a marker
    line naming the cap and URL so forecasters can tell the snapshot is partial.

    Invariant: ``len(return) <= cap``. When truncation fires, the emitted text
    is trimmed to ``cap - len(marker)`` before the marker is appended so the
    total stays within budget. If the marker itself is longer than the cap
    (pathologically small cap in tests), returns the raw truncation without
    the marker rather than emitting only-marker text.
    """
    if cap <= 0:
        # No budget at all — the caller's arithmetic (section allowance minus a lead line
        # minus a very long parent URL) can go negative. ``text[:cap]`` on a negative cap
        # returns nearly the WHOLE text while the invariant claims a bound, so a
        # zero-budget slot would have rendered a full page.
        return ""
    if len(text) <= cap:
        return text
    marker = f"\n[truncated at {cap} chars — full source at {url}]"
    if len(marker) >= cap:
        # Cap is too small to even fit the marker; degrade to plain truncation.
        return text[:cap]
    body_budget = cap - len(marker)
    return text[:body_budget].rstrip() + marker


def _truncate_csv_middle(text: str, cap: int, url: str) -> str:
    """Bound a CSV at ``cap`` chars, keeping the header + BOTH ends of the rows.

    The resolution-relevant rows are the most recent ones, but Datawrapper
    datasets run in either direction — the tracker model-average series are
    chronological (newest LAST) while the poll-input tables on the same pages
    are newest FIRST (observed live on both natesilver.net trackers,
    2026-08-25). Keeping both ends is ordering-agnostic: the newest rows
    survive whichever end they sit at, and only the middle is omitted. Plain
    head truncation would cut the current level off an ascending series — the
    stale-as-live failure in a different coat.

    Invariant: ``len(return) <= cap``. Degrades to plain head truncation when
    the text has too few lines to be row-shaped or the cap is too small to fit
    the header + marker + at least one row.
    """
    if len(text) <= cap:
        return text
    lines = text.rstrip("\n").split("\n")
    if len(lines) < 4:
        return _truncate_with_marker(text, cap, url)
    header = lines[0]
    rows = lines[1:]
    marker_template = "[... {} middle rows omitted — full data at {}]"
    # Reserve marker space at its worst-case width (all rows omitted).
    worst_marker = marker_template.format(len(rows), url)
    row_budget = cap - len(header) - len(worst_marker) - 4  # joining newlines
    if row_budget <= 0:
        return _truncate_with_marker(text, cap, url)
    head_budget = row_budget // 2
    head: list[str] = []
    used_head = 0
    for line in rows:
        cost = len(line) + 1
        if used_head + cost > head_budget:
            break
        head.append(line)
        used_head += cost
    tail: list[str] = []
    used_tail = 0
    for line in reversed(rows[len(head) :]):
        cost = len(line) + 1
        if used_head + used_tail + cost > row_budget:
            break
        tail.append(line)
        used_tail += cost
    if not head and not tail:
        return _truncate_with_marker(text, cap, url)
    tail.reverse()
    marker = marker_template.format(len(rows) - len(head) - len(tail), url)
    return "\n".join([header, *head, marker, *tail])


def _fetch_result_sources(results: list[FetchResult]) -> dict[str, str]:
    """Per-URL outcome map for provider diagnostics: ``{domain: "ok" | <FetchStatus>}``.

    A fetched URL normalizes to ``"ok"`` (the shared "contributed" token the
    diagnostics formatter recognizes); every other ``FetchStatus``
    (``blocked`` / ``js_wall`` / ``not_found`` / ``error`` / ``unsupported_type`` /
    ``ssrf_blocked``) is kept verbatim as the loss token so the reason survives
    into the ``lost=`` segment. Duplicate domains are disambiguated with a ``#N``
    suffix so no per-URL outcome is silently overwritten.

    Tier-2 dataset results are keyed ``datawrapper:<chart_id>`` — they are hop
    artifacts, not cited sources, and every dataset URL shares one CDN netloc, so
    domain keys would collapse them into ``static.dwcdn.net#N`` noise. Their
    ``stale_data`` verdict maps to the benign ``"none"``: that is the freshness
    guard REFUSING to serve months-old data as live, i.e. the feature working as
    designed, and reporting it in ``lost=`` would dress a by-design withhold as a
    lost cited source. A genuinely failed hop (``error``/``blocked``/``not_found``/
    ``empty_body``/``unsupported_type``) keeps its verbatim loss token — that is
    real signal about the CDN, and it is the reason the content check runs BEFORE
    the freshness guard: an empty CDN body must not borrow ``stale_data``'s
    by-design amnesty.
    """
    sources: dict[str, str] = {}
    for r in results:
        if r.chart_id is not None:
            key = f"datawrapper:{r.chart_id}"
        else:
            try:
                key = urlparse(r.url).netloc or r.url
            except ValueError:
                key = r.url
        if key in sources:
            n = 2
            while f"{key}#{n}" in sources:
                n += 1
            key = f"{key}#{n}"
        if r.status == "success":
            sources[key] = "ok"
        elif r.chart_id is not None and r.status == "stale_data":
            sources[key] = "none"
        else:
            sources[key] = r.status
    return sources


def _render_fetch_failures(failures: list[FetchResult]) -> str:
    """Render failed fetches as a compact ``"domain: status, domain: status"`` list."""
    parts: list[str] = []
    for r in failures:
        try:
            domain = urlparse(r.url).netloc or r.url
        except ValueError:
            domain = r.url
        parts.append(f"{domain}: {r.status}")
    return ", ".join(parts)


def _budgeted_success_sections(successes: list[FetchResult], fetched_iso: str) -> tuple[list[str], int]:
    """Render the success sections inside the two partitioned budgets; returns ``(sections, dropped)``.

    Cited pages and Tier-2 datasets draw on separate allowances, so a chart's rows can
    never evict the page text the section exists to serve.
    """
    sections: list[str] = []
    page_remaining = RESOLUTION_SOURCE_TOTAL_MAX_CHARS
    dataset_remaining = RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS * RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS
    dropped = 0
    for r in successes:
        # Cheap per-section budget accounting on the text body only. Section
        # overhead (URL heading + fetched-date line) is negligible relative to
        # the RESOLUTION_SOURCE_TOTAL_MAX_CHARS total budget; if the caller
        # tightens it dramatically for a test, we still cut the text
        # conservatively.
        is_dataset = r.chart_id is not None
        remaining = dataset_remaining if is_dataset else page_remaining
        if remaining <= 0:
            dropped += 1
            continue
        body = r.text
        if len(body) > remaining:
            # Through the marker-emitting truncator, not a bare slice. A bare slice cut
            # mid-sentence AND could eat the per-URL `[truncated at N chars ...]` marker the
            # fetch already appended at the end — leaving an already-truncated page rendering
            # as complete. Reachable on prod constants (5 x 6000 per-URL against an 18000
            # total). The CSV variant keeps both ends, which is what makes a dataset's newest
            # rows survive whichever direction it runs.
            body = (_truncate_csv_middle if is_dataset else _truncate_with_marker)(body, remaining, r.url)
        if is_dataset:
            dataset_remaining -= len(body)
        else:
            page_remaining -= len(body)
        sections.append(f"### {r.url}\n(fetched {fetched_iso})\n\n{body}")
    return sections, dropped


def format_resolution_sections(results: list[FetchResult], fetched_at: datetime) -> str:
    """Render fetch results as a markdown body block (orchestrator adds the ``##`` header).

    Returns ``""`` only when no URLs were attempted (empty ``results``). When
    URLs were attempted:

    - ALL failed (403 / JS wall / error / etc.) → a one-line notice naming the
      unreachable domains and their statuses, so forecasters learn the resolving
      page was never seen instead of silently getting nothing (the qid 44211
      failure: the CBP dashboard 403'd and no one in the pipeline knew).
    - SOME succeeded → the success sections as before, plus a terse trailing
      note about any that failed.

    Enforces ``RESOLUTION_SOURCE_TOTAL_MAX_CHARS`` across CITED-page success
    sections: later sections are trimmed (or dropped) once the budget is spent.
    Tier-2 dataset sections (``chart_id`` set) budget against their OWN allowance
    (``MAX_CHARTS x PER_DATASET_MAX_CHARS``) — the two classes are partitioned so
    a chart's rows can never evict the cited page text the section exists to
    serve, while a dataset still renders adjacent to its parent page. Per-URL
    truncation is the caller's responsibility (already applied in ``_fetch_one``
    and the hop); these caps cover the aggregate section length. When one or
    more sections are dropped entirely (budget spent before them), a final line
    names the dropped count so downstream readers can tell the snapshot is partial.

    Failure wording is partitioned the same way: a Datawrapper dataset is not a
    CITED resolution source, and its most common non-success — ``stale_data``,
    the freshness guard refusing to serve months-old data as live — is not a
    fetch failure at all, so datasets never ride the "cited resolution source(s)
    could not be fetched" notices and get their own withheld line instead.
    """
    if not results:
        return ""

    successes = [r for r in results if r.status == "success"]
    cited_failures = [r for r in results if r.status != "success" and r.chart_id is None]
    dataset_nonsuccesses = [r for r in results if r.status != "success" and r.chart_id is not None]

    def _dataset_withheld_note() -> str:
        n = len(dataset_nonsuccesses)
        statuses = ", ".join(sorted({r.status for r in dataset_nonsuccesses}))
        # Wording covers every non-success a dataset can carry, not just
        # `stale_data`: a body that is empty or not row-shaped is withheld under
        # the same rule (nothing may be passed off as the chart's live series).
        return (
            f"[{n} embedded chart dataset(s) not served ({statuses}) — withheld rather than "
            f"passed off as the live series; the cited page text is unaffected.]"
        )

    if not successes:
        n = len(cited_failures)
        notice = (
            f"[{n} resolution source(s) could not be fetched: {_render_fetch_failures(cited_failures)}] — "
            f"the resolving page was unreachable; weight other evidence accordingly."
        )
        if dataset_nonsuccesses:
            notice += "\n\n" + _dataset_withheld_note()
        return notice

    fetched_iso = fetched_at.strftime("%Y-%m-%d")
    caveat = f"Snapshot of the cited resolution source(s) as of {fetched_iso} — primary grading evidence."

    sections, dropped = _budgeted_success_sections(successes, fetched_iso)

    rendered = caveat + "\n\n" + "\n\n".join(sections)
    if dropped:
        rendered += f"\n\n[{dropped} additional source(s) omitted — section budget]"
    if cited_failures:
        rendered += (
            f"\n\n[Note: {len(cited_failures)} other cited resolution source(s) could not be fetched: "
            f"{_render_fetch_failures(cited_failures)} — weight accordingly.]"
        )
    if dataset_nonsuccesses:
        rendered += "\n\n" + _dataset_withheld_note()
    return rendered


# ---------------------------------------------------------------------------
# Extraction wrapper (isolated for tests; offloads trafilatura's sync API)
# ---------------------------------------------------------------------------


def _extract_main_text(body: bytes, url: str) -> str | None:
    """Trafilatura extraction. Callers wrap in ``await asyncio.to_thread(...)``.

    Returns None on empty/failed extraction so callers can classify.
    """
    try:
        out = trafilatura.extract(
            body,
            url=url,
            favor_precision=True,
            include_comments=False,
            include_tables=True,
            output_format="txt",
        )
    except (ValueError, TypeError, RuntimeError) as e:
        # Trafilatura occasionally raises on truly malformed input. We soft-fail
        # here so a single broken page doesn't take down the provider.
        logger.warning(f"trafilatura extraction failed for {url}: {e}")
        return None
    if not out or not out.strip():
        return None
    return out


# ---------------------------------------------------------------------------
# Network layer (patched in tests via `_get_session`)
# ---------------------------------------------------------------------------


def _get_session() -> aiohttp.ClientSession:
    """Construct a fresh aiohttp session with browser-like headers. Patched in tests.

    The session's TCPConnector is wired to a :class:`FilteringResolver` seeded
    with :func:`_ip_is_disallowed`. This is the actual DNS-rebinding boundary:
    aiohttp's connect-time DNS lookup (and its DNS cache — see aiohttp docs)
    only ever surface IPs that pass the same predicate as
    :func:`is_public_http_url`, so the preflight guard can't be raced by a
    rebinding server between resolve and connect. The preflight guard remains
    for fast observability (it lets us emit ``ssrf_blocked`` on obviously bad
    URLs without opening a session), but it is not the trust boundary.
    """
    return build_session(
        timeout_s=RESOLUTION_SOURCE_HTTP_TIMEOUT,
        connector_limit=RESOLUTION_SOURCE_GLOBAL_CONCURRENCY,
        headers=BROWSER_HEADERS,
        resolver=_make_filtering_resolver(),
    )


_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
_RAW_TEXT_CONTENT_TYPES = ("text/plain", "text/csv")
_JSON_CONTENT_TYPES = ("application/json",)


def _sem_for_host(host_sems: dict[str, asyncio.Semaphore], url: str) -> asyncio.Semaphore:
    """Get-or-create the ``Semaphore(1)`` for ``url``'s netloc.

    Every task in one :func:`fetch_resolution_sources` run shares the same
    ``host_sems`` map, so every request to a given host — original URL or
    redirect hop — contends on the same semaphore object.
    """
    host = urlparse(url).netloc
    sem = host_sems.get(host)
    if sem is None:
        sem = asyncio.Semaphore(1)
        host_sems[host] = sem
    return sem


async def _resolution_redirect_outcome(resp: Any, current_url: str, content_type: str) -> FetchResult | str:
    """Vet a 3xx hop: the next URL to follow, or a terminal error/blocked result."""
    status = resp.status
    location = resp.headers.get("Location") if resp.headers else None
    if not location:
        # Malformed redirect — no Location header.
        logger.info(f"resolution_source fetched {urlparse(current_url).netloc} (error http={status} no Location)")
        return FetchResult(
            url=current_url,
            status="error",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    next_url = urljoin(current_url, location)
    if not await is_public_http_url(next_url):
        logger.warning(
            f"resolution_source ssrf_blocked (redirect): {urlparse(current_url).netloc} -> {urlparse(next_url).netloc}"
        )
        return FetchResult(
            url=next_url,
            status="ssrf_blocked",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    if is_metaculus_self_ref(next_url):
        # The URL pre-filter drops metaculus self-refs, but a 3xx can
        # still land on metaculus.com; don't follow it (no new info,
        # and keeps our IP off the same host the critical API uses).
        logger.info(
            f"resolution_source metaculus_self_ref (redirect): "
            f"{urlparse(current_url).netloc} -> {urlparse(next_url).netloc}"
        )
        return FetchResult(
            url=next_url,
            status="blocked",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    return next_url


def _resolution_status_outcome(status: int, current_url: str, content_type: str) -> FetchResult | None:
    """Terminal result for a non-200 status, or None when the body should be read."""
    netloc = urlparse(current_url).netloc
    if status in (403, 406, 429):
        logger.info(f"resolution_source fetched {netloc} (blocked http={status})")
        return FetchResult(
            url=current_url,
            status="blocked",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    if status in (404, 410):
        logger.info(f"resolution_source fetched {netloc} (not_found http={status})")
        return FetchResult(
            url=current_url,
            status="not_found",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    if status != 200:
        logger.info(f"resolution_source fetched {netloc} (error http={status})")
        return FetchResult(
            url=current_url,
            status="error",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    return None


async def _resolution_html_outcome(resp: Any, current_url: str, content_type: str) -> FetchResult:
    """Trafilatura extraction plus the JS-wall check, carrying any Datawrapper embeds along."""
    status = resp.status
    netloc = urlparse(current_url).netloc
    body = await read_body_capped(
        resp,
        max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
        label=f"resolution_source {netloc}",
    )
    if body is None:
        return FetchResult(
            url=current_url,
            status="error",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    # Datawrapper embeds are only visible in the RAW HTML —
    # trafilatura drops iframes and embed scripts at every
    # setting — so the scan runs on the raw body, before
    # (and regardless of) main-text extraction. Decoded
    # through the shared helper so a BOM'd / non-UTF-8 page's
    # embeds are still findable; the page's main text is
    # trafilatura's to decode, which is why no vacuity check
    # runs on this branch (an empty extraction is `js_wall`).
    charts = extract_datawrapper_charts(decode_text_body(body, content_type)[0])
    extracted = await asyncio.to_thread(_extract_main_text, body, current_url)
    # An empty extraction on a 200 OK is a JS-wall (SPA that
    # rendered client-side, cookie/consent gate, etc.) —
    # exactly the Tier-2 candidate signal. Treat identically
    # to short-but-nonempty extractions. A walled page still
    # exposes its embeds, so the charts ride along.
    if extracted is None or looks_like_js_wall(extracted):
        logger.info(f"resolution_source fetched {netloc} (js_wall)")
        return FetchResult(
            url=current_url,
            status="js_wall",
            text="",
            http_status=status,
            content_type=content_type or None,
            datawrapper_charts=charts,
        )
    logger.info(f"resolution_source fetched {netloc} (success)")
    return FetchResult(
        url=current_url,
        status="success",
        text=_truncate_with_marker(extracted, RESOLUTION_SOURCE_PER_URL_MAX_CHARS, current_url),
        http_status=status,
        content_type=content_type or None,
        datawrapper_charts=charts,
    )


async def _resolution_text_outcome(resp: Any, current_url: str, content_type: str) -> FetchResult:
    """Capped raw body for a JSON / plain-text / CSV response, refusing a vacuous one."""
    status = resp.status
    netloc = urlparse(current_url).netloc
    body = await read_body_capped(
        resp,
        max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
        label=f"resolution_source {netloc}",
    )
    if body is None:
        return FetchResult(
            url=current_url,
            status="error",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    raw, undecodable_ratio = decode_text_body(body, content_type)
    # Markup stripping on the text branches only: a CSV or
    # plain-text body carrying `<a href=…>` per row spends the
    # per-URL budget on tags (see `strip_html_tags`), while a
    # JSON body's angle brackets sit inside string values that
    # are the data. Both text types get it because the labels
    # are demonstrably unreliable here — Datawrapper's own
    # versioned route serves CSV as application/octet-stream.
    if any(ct in content_type for ct in _RAW_TEXT_CONTENT_TYPES):
        raw = strip_html_tags(raw)
    vacuous = vacuous_body_status(raw, undecodable_ratio, require_csv_rows=False)
    if vacuous is not None:
        logger.info(
            f"resolution_source fetched {netloc} ({vacuous}: 200 with no usable content, "
            f"{len(body)} bytes, undecodable={undecodable_ratio:.2f})"
        )
        return FetchResult(
            url=current_url,
            status=vacuous,
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    logger.info(f"resolution_source fetched {netloc} (success)")
    return FetchResult(
        url=current_url,
        status="success",
        text=_truncate_with_marker(raw, RESOLUTION_SOURCE_PER_URL_MAX_CHARS, current_url),
        http_status=status,
        content_type=content_type or None,
    )


async def _resolution_response_outcome(resp: Any, current_url: str) -> FetchResult | str:
    """Classify one response: a terminal FetchResult, or the next URL on a vetted 3xx."""
    status = resp.status
    content_type = (resp.headers.get("Content-Type") or "").lower() if resp.headers else ""

    if status in REDIRECT_STATUSES:
        return await _resolution_redirect_outcome(resp, current_url, content_type)

    # Non-redirect response — same status routing as before.
    non_ok = _resolution_status_outcome(status, current_url, content_type)
    if non_ok is not None:
        return non_ok

    # 200 OK: route on content type.
    if any(ct in content_type for ct in _HTML_CONTENT_TYPES):
        return await _resolution_html_outcome(resp, current_url, content_type)
    if any(ct in content_type for ct in _JSON_CONTENT_TYPES) or any(
        ct in content_type for ct in _RAW_TEXT_CONTENT_TYPES
    ):
        return await _resolution_text_outcome(resp, current_url, content_type)

    # Anything else — PDF, images, etc. Do NOT read the body.
    # INTENDED limitation: a 200 OK with a missing/empty Content-Type header
    # also lands here (ct='') and is dropped unread. Real resolution sources
    # send Content-Type; content-sniffing would re-open the don't-read-unknown-
    # bodies posture for a case that mostly can't happen. The per-URL
    # FetchStatus is the Tier-2 seam if logs ever show `unsupported_type ct=''`.
    logger.info(f"resolution_source fetched {urlparse(current_url).netloc} (unsupported_type ct={content_type!r})")
    return FetchResult(
        url=current_url,
        status="unsupported_type",
        text="",
        http_status=status,
        content_type=content_type or None,
    )


async def _fetch_one_hop(session: Any, current_url: str, host_sems: dict[str, asyncio.Semaphore]) -> FetchResult | str:
    """ONE GET against ``current_url`` under its host semaphore: terminal result or next URL."""
    async with _sem_for_host(host_sems, current_url):
        try:
            async with session.get(current_url, allow_redirects=False) as resp:
                return await _resolution_response_outcome(resp, current_url)
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.info(f"resolution_source fetch error for {current_url}: {type(e).__name__}: {e}")
            return FetchResult(
                url=current_url,
                status="error",
                text="",
                http_status=None,
                content_type=None,
            )


async def _fetch_one(session: Any, url: str, host_sems: dict[str, asyncio.Semaphore]) -> FetchResult:
    """Fetch a single URL, holding the per-host politeness semaphore hop by hop.

    Content-type routing:
      * HTML → trafilatura extraction (via to_thread) + JS-wall check.
      * JSON → capped raw body, no pretty-print (the data IS the content).
      * text/plain, text/csv → capped raw body.
      * anything else (PDF/binary) — including a missing/empty Content-Type
        header, by design — → ``unsupported_type``, body NOT read.

    Politeness: each hop acquires the semaphore for THAT hop's host around its
    single GET (+ body read on terminal responses) and releases it before
    following a redirect. Keying per hop — not on the original URL's host —
    preserves one-request-per-host when chains from different initial hosts
    converge on the same final host; the strict per-hop acquire/release
    pairing means an A→B→A chain never re-acquires a semaphore it still holds
    (asyncio semaphores are not reentrant).

    SSRF guard: rejects non-public URLs (private / loopback / link-local IPs,
    userinfo tricks, non-http(s) schemes) BEFORE any network I/O and again on
    every redirect Location. The connect-time :class:`FilteringResolver` (see
    :func:`_get_session`) provides the actual DNS-rebinding boundary; these
    preflight checks are fast-fail observability so we surface
    ``ssrf_blocked`` without opening a session. Redirects are followed in-band
    with a hard ``MAX_REDIRECTS`` cap.

    No retries (Tier 1 anti-goal). Any aiohttp/asyncio error becomes ``error``.
    """
    # Guard the initial URL before any network I/O.
    if not await is_public_http_url(url):
        logger.warning(f"resolution_source ssrf_blocked (initial url): {urlparse(url).netloc}")
        return FetchResult(
            url=url,
            status="ssrf_blocked",
            text="",
            http_status=None,
            content_type=None,
        )

    current_url = url
    # Bounded redirect loop. Each iteration issues ONE GET with
    # allow_redirects=False under the current hop's host semaphore; a redirect
    # status resolves the Location, re-guards, and loops (each hop releases its
    # semaphore before the next acquires its own — no nesting, so no
    # self-deadlock on revisited hosts).
    # Non-redirect responses fall through to the content-type routing below.
    for _hop in range(MAX_REDIRECTS + 1):
        outcome = await _fetch_one_hop(session, current_url, host_sems)
        if isinstance(outcome, FetchResult):
            return outcome
        current_url = outcome

    # Fell out of the loop -> exceeded MAX_REDIRECTS.
    logger.info(f"resolution_source redirect chain exceeded {MAX_REDIRECTS} hops (final={current_url})")
    return FetchResult(
        url=current_url,
        status="error",
        text="",
        http_status=None,
        content_type=None,
    )


def _datawrapper_hop_status(status: int) -> FetchStatus:
    """Map the CDN's HTTP status onto a FetchStatus (200 -> ``success``)."""
    if status in (404, 410):
        return "not_found"
    if status in (403, 406, 429):
        return "blocked"
    if status != 200:
        return "error"
    return "success"


def _datawrapper_last_modified(resp: Any) -> datetime | None:
    """The dataset's parsed ``Last-Modified``, or None when absent or unparseable."""
    raw = resp.headers.get("Last-Modified") if resp.headers else None
    return parse_http_last_modified(raw) if raw else None


def _datawrapper_freshness_failure(last_modified: datetime | None) -> str | None:
    """Why ``last_modified`` fails the freshness guard, or None when it passes.

    Two-sided, deliberately. The lead this stamp authorizes asserts a
    publication date, and a FUTURE one means a broken clock or a misparse on
    one side — so it is unusable as a freshness claim, not maximally fresh.
    The old one-sided check let any future date through as the freshest
    possible dataset.
    """
    if last_modified is None:
        return "no parseable Last-Modified"
    now = datetime.now(UTC)
    if last_modified - now > _DATAWRAPPER_CLOCK_SKEW_TOLERANCE:
        return f"published {last_modified.isoformat()}, which is in the FUTURE"
    if now - last_modified > timedelta(days=RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS):
        return (
            f"published {last_modified.isoformat()}, age {(now - last_modified).days}d "
            f"> {RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS}d bound"
        )
    return None


def _datawrapper_success_text(
    chart: DatawrapperChartRef, parent_url: str, url: str, *, dataset_text: str, published: datetime
) -> str:
    """The liveness lead plus the budgeted CSV rows."""
    # Every claim in this lead is now checked: the timestamp by the
    # freshness guard above, and "dataset" itself by the row-shape
    # check — an authoritative `published <ts>` stamp over an empty or
    # soft-404 body was the same defect class as a manufactured price.
    title_part = f" ({chart.title!r})" if chart.title else ""
    lead = (
        f'Live "Get the data" dataset for Datawrapper chart {chart.chart_id}{title_part} '
        f"embedded in {parent_url}. Dataset published {published.isoformat()}."
    )
    # The DATASET cap, not the page cap: datasets budget against their own
    # section allowance so a chart's rows can never evict cited page text.
    # Tags are stripped BEFORE truncation so the budget buys rows, not markup.
    csv_budget = RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS - len(lead) - 2
    return f"{lead}\n\n{_truncate_csv_middle(dataset_text, csv_budget, url)}"


async def _datawrapper_dataset_outcome(resp: Any, chart: DatawrapperChartRef, parent_url: str, url: str) -> FetchResult:
    """Turn the CDN response into a FetchResult, serving the dataset live or not at all."""
    status = resp.status
    content_type = (resp.headers.get("Content-Type") or "").lower() if resp.headers else ""
    hop_status = _datawrapper_hop_status(status)
    if hop_status != "success":
        logger.info(f"resolution_source datawrapper hop {chart.chart_id} ({hop_status} http={status})")
        return FetchResult(
            url=url,
            status=hop_status,
            text="",
            http_status=status,
            content_type=content_type or None,
            chart_id=chart.chart_id,
            chart_title=chart.title,
            parent_url=parent_url,
        )

    body = await read_body_capped(
        resp,
        max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
        label=f"resolution_source datawrapper {chart.chart_id}",
    )
    if body is None:
        return FetchResult(
            url=url,
            status="error",
            text="",
            http_status=status,
            content_type=content_type or None,
            chart_id=chart.chart_id,
            chart_title=chart.title,
            parent_url=parent_url,
        )

    # Content BEFORE freshness, deliberately: an empty or non-CSV CDN
    # body is a failed hop whatever its Last-Modified says, and
    # `stale_data` is reported to diagnostics as the benign `none`
    # (the freshness guard working as designed), which would hide it.
    # Row-shape is decided on the PRE-strip text: looks_like_csv_rows
    # rejects markup by its leading `<`, and stripping first would remove
    # exactly the allow-listed fragment tags (`<p>`, `<div>`) a CDN
    # soft-404 opens with, letting an error page carry the authoritative
    # "Dataset published" lead if its prose holds a comma.
    dataset_text, undecodable_ratio = decode_text_body(body, content_type)
    vacuous = vacuous_body_status(dataset_text, undecodable_ratio, require_csv_rows=True)
    dataset_text = strip_html_tags(dataset_text).strip()
    if vacuous is not None:
        logger.warning(
            f"resolution_source datawrapper hop {chart.chart_id}: dataset body is not a usable "
            f"dataset ({vacuous}: {len(body)} bytes, undecodable={undecodable_ratio:.2f}) — "
            f"withheld rather than stamped live"
        )
        return FetchResult(
            url=url,
            status=vacuous,
            text="",
            http_status=status,
            content_type=content_type or None,
            chart_id=chart.chart_id,
            chart_title=chart.title,
            parent_url=parent_url,
        )

    last_modified = _datawrapper_last_modified(resp)
    freshness_failure = _datawrapper_freshness_failure(last_modified)
    if freshness_failure is not None:
        logger.warning(
            f"resolution_source datawrapper hop {chart.chart_id}: dataset failed the "
            f"freshness guard ({freshness_failure}) — withheld, not served as live"
        )
        return FetchResult(
            url=url,
            status="stale_data",
            text="",
            http_status=status,
            content_type=content_type or None,
            chart_id=chart.chart_id,
            chart_title=chart.title,
            parent_url=parent_url,
            data_last_modified=last_modified.isoformat() if last_modified else None,
        )

    assert last_modified is not None  # a passing freshness guard implies a parsed timestamp
    logger.info(f"resolution_source datawrapper hop {chart.chart_id} (success, published {last_modified.isoformat()})")
    return FetchResult(
        url=url,
        status="success",
        text=_datawrapper_success_text(chart, parent_url, url, dataset_text=dataset_text, published=last_modified),
        http_status=status,
        content_type=content_type or None,
        chart_id=chart.chart_id,
        chart_title=chart.title,
        parent_url=parent_url,
        data_last_modified=last_modified.isoformat(),
    )


async def _fetch_datawrapper_dataset(
    session: Any,
    chart: DatawrapperChartRef,
    parent_url: str,
    host_sems: dict[str, asyncio.Semaphore],
) -> FetchResult:
    """Tier-2 hop: fetch one Datawrapper chart's LIVE dataset CSV.

    Fetches ONLY the version-free ``static.dwcdn.net/data/<id>.csv`` route —
    never a page-pinned versioned ``dataset.csv``, which serves months-stale
    snapshots as HTTP 200 (see the route mechanism note in ``http_fetch``).

    Freshness guard (serve live or nothing): the dataset's ``Last-Modified``
    must be within ``RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS`` of now.
    Older, missing, or unparseable → ``stale_data`` with no text, so a dead
    chart can never masquerade as the live resolving series. The publish
    timestamp is also rendered into the section so forecasters see the data's
    age even when it passes.

    Content-Type is deliberately NOT gated here: we constructed the URL from a
    shape-validated chart id, the endpoint serves CSV (its versioned sibling
    labels the same bytes ``application/octet-stream``), and the body read is
    size-capped either way. Redirects are unexpected on this CDN and map to
    ``error`` rather than being followed.
    """
    url = datawrapper_live_data_url(chart.chart_id)
    # Uniform SSRF preflight (dwcdn is a public CDN — no exemptions added; the
    # connect-time FilteringResolver stays the real boundary).
    if not await is_public_http_url(url):
        logger.warning(f"resolution_source ssrf_blocked (datawrapper hop): {urlparse(url).netloc}")
        return FetchResult(
            url=url,
            status="ssrf_blocked",
            text="",
            http_status=None,
            content_type=None,
            chart_id=chart.chart_id,
            chart_title=chart.title,
            parent_url=parent_url,
        )

    async with _sem_for_host(host_sems, url):
        try:
            async with session.get(url, allow_redirects=False) as resp:
                return await _datawrapper_dataset_outcome(resp, chart, parent_url, url)
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.info(f"resolution_source datawrapper hop {chart.chart_id} error: {type(e).__name__}: {e}")
            return FetchResult(
                url=url,
                status="error",
                text="",
                http_status=None,
                content_type=None,
                chart_id=chart.chart_id,
                chart_title=chart.title,
                parent_url=parent_url,
            )


def _select_datawrapper_charts(page_results: list[FetchResult]) -> list[tuple[int, DatawrapperChartRef]]:
    """Pick the charts to hop to, as ``(parent_index, chart)`` pairs.

    Page order first, then document order within a page (tracker pages put the
    hero/resolving chart first), deduped by chart id across pages, capped
    globally at ``RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS``.
    """
    picks: list[tuple[int, DatawrapperChartRef]] = []
    seen: set[str] = set()
    for idx, r in enumerate(page_results):
        for chart in r.datawrapper_charts:
            if chart.chart_id in seen:
                continue
            seen.add(chart.chart_id)
            picks.append((idx, chart))
            if len(picks) >= RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS:
                return picks
    return picks


def _interleave_dataset_results(
    page_results: list[FetchResult],
    picks: list[tuple[int, DatawrapperChartRef]],
    dataset_results: list[FetchResult],
) -> list[FetchResult]:
    """Place each dataset result directly after its parent page's result, so
    the rendered section (and the total-budget trimming order) keeps a chart's
    data adjacent to the page that embeds it."""
    by_parent: dict[int, list[FetchResult]] = {}
    for (idx, _chart), ds in zip(picks, dataset_results, strict=False):
        by_parent.setdefault(idx, []).append(ds)
    merged: list[FetchResult] = []
    for idx, r in enumerate(page_results):
        merged.append(r)
        merged.extend(by_parent.get(idx, []))
    return merged


async def fetch_resolution_sources(urls: list[str]) -> list[FetchResult]:
    """Fetch each URL under per-netloc Semaphore(1) politeness, then hop to
    the live datasets of any Datawrapper charts the fetched pages embed.

    Distinct hosts run concurrently up to the connector limit; same-host
    requests serialize (politeness — e.g. StatCan asks Crawl-delay: 2). The
    shared ``host_sems`` map is handed to every ``_fetch_one`` task so each
    redirect hop contends on ITS host's semaphore — chains from different
    initial hosts that converge on one final host still serialize there; the
    Tier-2 dataset fetches contend on the dwcdn host's semaphore the same
    way. Session is closed in ``finally``.

    Teardown race guard (F5): the outer factory wraps this call in
    ``asyncio.wait_for``. When the wall-clock timeout fires, wait_for cancels
    this coroutine — but if a gather is still in flight we'd exit the
    ``async with session`` block while children are mid-request, and aiohttp
    would then close their transports out from under them (surfacing as
    scary tracebacks in logs, and in extreme cases resource-warning fires
    on connections that never got cleaned up). We use explicit Task objects
    — pages and datasets alike — so we can cancel + drain them in a
    ``finally`` before the session closes.
    """
    host_sems: dict[str, asyncio.Semaphore] = {}
    tasks: list[asyncio.Task[FetchResult]] = []
    started = time.monotonic()

    session_cm = _get_session()
    async with session_cm as session:
        try:
            page_tasks = [asyncio.create_task(_fetch_one(session, u, host_sems)) for u in urls]
            tasks.extend(page_tasks)
            page_results = list(await asyncio.gather(*page_tasks, return_exceptions=False))

            picks = _select_datawrapper_charts(page_results)
            if not picks:
                return page_results
            # The hop is a SECOND network phase inside the provider's single 45s wall,
            # and its datasets share one CDN host, so the per-host politeness semaphore
            # serializes them — worst case MAX_CHARTS x the 20s HTTP timeout, on top of
            # whatever the page phase already spent. Unbounded, a slow CDN tail would
            # blow the outer wall and cancel the WHOLE provider, discarding Tier-1
            # pages that already fetched. So the hop gets only the wall budget the
            # pages left behind (minus a margin so this path returns before the outer
            # wait_for fires), degrades to the pages on its own timeout, and is skipped
            # outright when less than one typical CDN fetch's worth remains. Typical
            # cost is trivial — a poll CSV is tens of KB off a CDN, sub-second-to-~2s
            # per dataset (the validation receipts' live runs) — so the bound exists
            # for the tail, which is exactly what a wall cap is for.
            hop_budget_s = (
                RESOLUTION_SOURCE_WALL_TIMEOUT
                - (time.monotonic() - started)
                - RESOLUTION_SOURCE_DATAWRAPPER_HOP_WALL_MARGIN_S
            )
            if hop_budget_s < RESOLUTION_SOURCE_DATAWRAPPER_MIN_HOP_BUDGET_S:
                logger.warning(
                    "resolution_source: skipping the datawrapper hop (%d chart(s)) — %.1fs of wall "
                    "budget left; serving %d Tier-1 page result(s) without datasets",
                    len(picks),
                    hop_budget_s,
                    len(page_results),
                )
                return page_results
            dataset_tasks = [
                asyncio.create_task(_fetch_datawrapper_dataset(session, chart, page_results[idx].url, host_sems))
                for idx, chart in picks
            ]
            tasks.extend(dataset_tasks)
            try:
                dataset_results = list(
                    await asyncio.wait_for(
                        asyncio.gather(*dataset_tasks, return_exceptions=False), timeout=hop_budget_s
                    )
                )
            except TimeoutError:
                logger.warning(
                    "resolution_source: datawrapper hop timed out after %.1fs; serving %d Tier-1 "
                    "page result(s) without datasets",
                    hop_budget_s,
                    len(page_results),
                )
                return page_results
            return _interleave_dataset_results(page_results, picks, dataset_results)
        finally:
            # Whether we exit normally or via cancellation, cancel any still-
            # running task and let them settle before the session closes.
            # (No-op cost when everything already finished successfully.)
            for t in tasks:
                if not t.done():
                    t.cancel()
            # return_exceptions=True: drained tasks may surface CancelledError,
            # which is expected here.
            await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def resolution_source_provider(is_benchmarking: bool = False) -> ResearchCallable:
    """Factory returning the async ResearchCallable for the resolution-source fetcher.

    Gating (both hard):

    - ``is_benchmarking=True`` short-circuits to ``""`` (leakage guard — current
      page content post-dates any backtest window, same rationale as the
      prediction-market provider).
    - Env flag ``RESOLUTION_SOURCE_ENABLED`` must be truthy.

    Returns section BODY only; the orchestrator prepends the ``## Resolution
    Source Snapshot`` header. Inner ``### {url}`` headers stay at h3 — the
    orchestrator's heading demotion only touches h1/h2, and h3 is already
    correctly nested under the h2 provider header.
    """

    async def _fetch(question: MetaculusQuestion) -> str:
        if is_benchmarking:
            return ""
        if not env_flag_enabled(RESOLUTION_SOURCE_ENABLED_ENV):
            return ""

        urls = select_fetchable_urls(question.resolution_criteria, question.fine_print)
        if not urls:
            return ""

        try:
            results = await asyncio.wait_for(
                fetch_resolution_sources(urls),
                timeout=RESOLUTION_SOURCE_WALL_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(f"resolution_source: wall-clock timeout after {RESOLUTION_SOURCE_WALL_TIMEOUT}s")
            return ""

        # CITED pages only. A withheld Tier-2 dataset is a hop artifact, not an
        # unfetched cited URL, and counting it here inflated the ratio with
        # by-design withholds (`stale_data`) on exactly the tracker questions the
        # hop serves. Datasets get their own count so both stay readable.
        cited = [r for r in results if r.chart_id is None]
        n_fail = sum(1 for r in cited if r.status != "success")
        n_datasets_withheld = sum(1 for r in results if r.chart_id is not None and r.status != "success")
        if n_fail or n_datasets_withheld:
            logger.info(
                f"resolution_source: {n_fail}/{len(cited)} cited urls unfetched "
                f"(js_wall/blocked — candidates for a future Tier-2 LLM fetch); "
                f"{n_datasets_withheld} embedded dataset(s) withheld",
            )
        qid = getattr(question, "id_of_question", None)
        record_raw_research(qid=qid, provider="resolution_source", payload=results)
        # Per-URL outcome map for the diagnostics block: even when the provider
        # returns a non-empty notice (all URLs failed → status `ok`), this surfaces
        # WHICH sources were lost so the block doesn't read as fully healthy.
        record_provider_detail(qid, "resolution_source", {"sources": _fetch_result_sources(results)})
        return format_resolution_sections(results, datetime.now(UTC))

    return _fetch
