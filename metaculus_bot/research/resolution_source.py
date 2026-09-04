# SMELL-EXEMPT-monolithic-file-loc: what stays here is fixed by the test suites'
# monkeypatch surface, not by the layer diagram. Ten `RESOLUTION_SOURCE_*` caps
# plus `_get_session`, `is_public_http_url`, `_extract_main_text` and
# `_sem_for_host` are patched on THIS module (tests/test_resolution_source_*.py,
# tests/test_agentic_tools.py), so every reader of one has to stay here to resolve
# it as a module global at call time — which pins the network layer, the section
# renderer and the provider factory. Everything with no patched read moved out:
# `resolution_url_scan` (URL extraction + skip predicates), `resolution_fetch_result`
# (FetchStatus/FetchResult, the vacuity rule, the result-list reductions), and
# `resolution_body_text` (markup stripping + the two truncators).
"""Resolution-source fetcher: Tier-1 cited pages + a Tier-2 Datawrapper hop.

Fetches the URL(s) explicitly cited in a Metaculus question's resolution
criteria (or fine print), extracts main content with trafilatura, and returns
a compact markdown section that every forecaster reads as the ground truth
the question will be graded against.

Tier 1 is plain HTTP with browser-like headers, no LLM calls, no retries. When it
cannot read a page, an ESCALATION LADDER runs (`_escalate_unresolved`), each rung
self-bounded against the same provider wall and each returning a result that went
through the SAME classification path (`_classify_html_body`), so a rescued page is
indistinguishable downstream from a directly-fetched one. The `route` on every
result says which rung produced it. Heavy anti-bot on a host that refuses our
address is the one shape no rung here fixes (see `FetchStatus` — `blocked` /
`js_wall` / `no_resolving_content` results are retained in the returned list as
that seam).

A 200-OK page whose extraction is under `RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS`
is page CHROME and is withheld as `no_resolving_content` rather than published as
grading evidence. `status_reason` says which shape: `embed_shell` when the raw
HTML names a routeless data embed (Infogram / Flourish / Tableau), so we know the
numbers exist and we have no route to them (qids 44554/44556, whose tracker
rendered 2.9k chars of forecast background as "primary grading evidence" with
zero polling numbers in it); `no_matching_passage` when a cited document read in full
discusses nothing the question asks about, the one shape that is a document rather than
a page and the one the paid rung is not allowed to re-read; `thin_page` otherwise
(q45088's 127-char SPA tab list, q45215's 385 chars of region names — five such renders
in the 2026-09-01 round, none naming a provider, which is why the floor is no longer
gated on one).
A page ABOVE the floor keeps its text when that text is content-shaped, plus a
one-line disclosure where an embed hid figures from it. Over the floor on short lines
alone (`content_share` under `RESOLUTION_SOURCE_CONTENT_SHARE_MIN`: a menu tree, a
member dropdown) it is still chrome; the same body is re-extracted under
`favor_precision`, and the page is withheld as `thin_page` when that fails too
(`_extract_page_text`).

Three free rungs sit under Tier 1, all deterministic and none of them a model call.
An ARIA-TABLE REWRITE runs before every extraction (`rewrite_aria_tables`): a
`<div role="table">` stat block is a real table trafilatura cannot see, and cdc.gov's
cyclosporiasis block published as an unlabelled "17,180 / 2" with its hospitalization
count missing entirely. A META-REFRESH HOP follows the redirect no status announces —
the same host's surveillance URLs answer 200 with a ~300-byte stub carrying only
`<meta http-equiv="refresh">`, which read as a JS wall — returning the target as the
next hop so it re-enters this same classification path under the shared `MAX_REDIRECTS`
cap and the same per-hop SSRF checks. A cited PDF is READ LOCALLY
(`research/document_text.py`, pypdf + BM25 passage selection against the question's
title and resolution criteria) instead of being dropped unread; bytes we read and could
not turn into text are `unreadable_document`, which is a different fact from
`unsupported_type` and the only one a paid document read could ever rescue. Each rung is
self-bounding against the provider wall the way the Datawrapper hop is, because the
outer `asyncio.wait_for` discards every page that already fetched when it fires.

A fourth rung leaves our own aiohttp client: a page that answered 200 with nothing
readable (`js_wall`, or the `thin_page` shape of `no_resolving_content`) is RENDERED
in headless Chromium (`research/rendered_fetch.py`, the same transport and the same
process-global Semaphore(2) launch cap the gap-fill v2 fetch ladder uses) and the DOM
re-enters the classification path. Measured 2026-09-03: Chromium rescued 6 of the 8
archived JS walls that still failed from a residential address. It runs from the
escalation ladder rather than inside the response context, so no aiohttp response is
held open across it — but it DOES re-acquire the same loop-wide per-host gate and hold
it across the launch-cap queue, the launch, the navigation, the settle and the teardown,
because Chromium dials that host itself (FUTURE.md item 5 carries the amplifier). The
transport recomputes the navigation budget once both gates are held, so a render that
queued behind them navigates on what is actually left or declines before a launch.

Inline chart configs are read straight out of the page we already hold
(`resolution_chart_data.render_inline_chart_data`): a Highcharts `data-chart`
attribute or `Highcharts.chart(...)` call carries its series as JSON, which
trafilatura drops at every setting. Zero LLM calls, no second request. It runs on
every HTML page, not only thin ones, because q43949's resolving page extracted
~80k chars of prose carrying none of the resolving figures while its annual
series — ending in the live count the question was graded on — sat in the
attribute. Chart data counts as CONTENT, so it also rescues a page the chrome
floor would otherwise withhold.

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
  to the connector limit. The map is PROCESS-WIDE (`http_fetch.host_semaphores`),
  so the gate holds across the several questions researching at once; it used to
  be rebuilt per provider call, which gave each question its own gate.
- Char caps apply to RAW (non-LLM-processed) content only; the LLM-emitted
  research bundle is never truncated (see the resolution-source plan).
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import socket
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp
import trafilatura
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    DOCUMENT_DIGEST_TOP_K,
    DOCUMENT_TEXT_MAX_PAGES,
    DOCUMENT_TEXT_MAX_SECONDS,
    DOCUMENT_TEXT_PDF_MAX_BYTES,
    GAP_FILL_V2_READER_MODEL,
    GAP_FILL_V2_READER_THINKING_LEVEL,
    GOOGLE_API_KEY_ENV,
    RESOLUTION_SOURCE_CLOCK_SKEW_TOLERANCE,
    RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS,
    RESOLUTION_SOURCE_CONTENT_SHARE_MIN,
    RESOLUTION_SOURCE_DATAWRAPPER_HOP_WALL_MARGIN_S,
    RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS,
    RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS,
    RESOLUTION_SOURCE_DATAWRAPPER_MIN_HOP_BUDGET_S,
    RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS,
    RESOLUTION_SOURCE_DERIVED_API_MIN_BUDGET_S,
    RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS,
    RESOLUTION_SOURCE_ENABLED_ENV,
    RESOLUTION_SOURCE_GLOBAL_CONCURRENCY,
    RESOLUTION_SOURCE_HTTP_TIMEOUT,
    RESOLUTION_SOURCE_JS_WALL_MIN_CHARS,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
    RESOLUTION_SOURCE_MAX_URLS,
    RESOLUTION_SOURCE_META_REFRESH_MIN_BUDGET_S,
    RESOLUTION_SOURCE_PDF_MIN_BUDGET_S,
    RESOLUTION_SOURCE_PER_URL_MAX_CHARS,
    RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S,
    RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S,
    RESOLUTION_SOURCE_TOTAL_MAX_CHARS,
    RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS,
    RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV,
    RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S,
    RESOLUTION_SOURCE_WALL_TIMEOUT,
    RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS,
    RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS,
    RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S,
    env_flag_enabled,
)
from metaculus_bot.research import derived_api
from metaculus_bot.research.document_text import (
    DocumentDigest,
    PdfText,
    digest_pdf,
    extract_pdf_text,
    has_text_layer,
    is_pdf_body,
)
from metaculus_bot.research.http_fetch import (
    BROWSER_HEADERS,
    MAX_REDIRECTS,
    REDIRECT_STATUSES,
    DatawrapperChartRef,
    FilteringResolver,
    build_session,
    datawrapper_live_data_url,
    decode_text_body,
    extract_datawrapper_charts,
    host_semaphores,
    meta_refresh_target,
    parse_http_last_modified,
    pdf_parse_semaphore,
    read_body_capped,
    rewrite_aria_tables,
    semaphore_for_host,
    unreadable_data_embed_providers,
)
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.raw_log import record_raw_research
from metaculus_bot.research.rendered_fetch import (
    RENDER_SETTLE_MS,
    RENDER_TIMEOUT_MS,
    MemoScope,
    RenderBudgetExpired,
    RenderedPage,
    _is_json_content_type,
    note_rendered_no_text,
    render_page,
    rendered_to_nothing,
)
from metaculus_bot.research.resolution_body_text import (
    _truncate_csv_middle,
    _truncate_with_marker,
    strip_html_tags,
)
from metaculus_bot.research.resolution_chart_data import render_inline_chart_data
from metaculus_bot.research.resolution_fetch_result import (
    _NON_OK_FETCH_STATUS,
    ROUTE_CAVEATS,
    FetchResult,
    FetchRoute,
    FetchStatus,
    FetchStatusReason,
    RungAttempt,
    _fetch_result_sources,
    _render_fetch_failures,
    fetch_outcome_token,
    looks_like_csv_rows,  # noqa: F401  # re-export: the Tier-1 suite imports the row-shape check from this module path
    vacuous_body_status,
)
from metaculus_bot.research.resolution_url_scan import (
    extract_source_urls,
    is_fred_url,
    is_metaculus_self_ref,
    is_yahoo_ticker_url,
    strip_markdown_escapes,  # noqa: F401  # re-export: the Tier-1 suite imports the markdown unescaper from this module path
)
from metaculus_bot.research.robots_policy import ROBOTS_FETCH_TIMEOUT_S, google_extended_blocks_url
from metaculus_bot.research.url_context_reader import run_url_context_read
from metaculus_bot.research.wayback import (
    innermost_url,
    parse_snapshot_url,
    snapshot_age_days,
    wayback_lead,
    wayback_snapshot_url,
)


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


async def resolve_vetted_public_ip(host: str) -> str | None:
    """Resolve ``host`` off the event loop and return its FIRST address — but only
    after vetting EVERY resolved address.

    The contract is reject-if-ANY-address-disallowed: a single disallowed address
    among the results rejects the whole hostname (DNS rebinding defense), as does a
    resolution failure, an unparseable sockaddr, or an empty result — an unfetchable
    host must reach the caller as one uniform rejection. Only when every address is
    publicly routable does the first one come back, so a caller may safely pin a
    connection to it.

    The one DNS-vetting predicate for both SSRF-guarded fetchers: the Tier-1
    preflight (:func:`is_public_http_url`) consumes the bool view below, and the
    agentic rendered rung pins Chromium's DNS to the returned IP.
    """
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
        if _ip_is_disallowed(resolved):
            return None
        if vetted_ip is None:
            vetted_ip = str(resolved)
    return vetted_ip


async def _every_resolved_address_is_public(host: str) -> bool:
    """True iff ``host`` resolves and EVERY resolved address is publicly routable.

    Bool view of :func:`resolve_vetted_public_ip`; a rejection surfaces to the
    caller as one uniform ``ssrf_blocked``.
    """
    return await resolve_vetted_public_ip(host) is not None


# ---------------------------------------------------------------------------
# Pure helpers — no I/O
# ---------------------------------------------------------------------------


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


def looks_like_page_chrome(text: str) -> bool:
    """True when an extraction is too thin to be anything but chrome around the content.

    The floor is what the ``no_resolving_content`` verdict rests on; a named embed
    provider only says WHERE the content went (`embed_shell` vs `thin_page`). It
    was gated on a named provider when it shipped, which withheld one shape of
    chrome and published the other: the 2026-09-01 round's five content-free
    `success` renders named no provider between them.

    Calibration re-checked for the ungated rule against the same census (89
    `resolution_source` archive records, 68 cited successes, 2026-09-02): 8 sit
    under 400 chars and all 8 are chrome — a 127-char SPA tab list
    (data.wastewaterscan.org, twice), 385 chars of Kazakh region names
    (election.gov.kz), AP's org boilerplate (355), an ABS release-date list with no
    figure (344), a mass-shooting tracker's "about the data" note (262), a
    Portuguese feedback-form blurb (157), and a clinicaltrials.gov data-element
    pointer (111). The shortest carrying the resolving content is still exactly
    401 (myfloridaelections.com's election-date table), so 400 remains the observed
    elbow and the floor stays deliberately below it: a page above it keeps its text
    and, where an embed hid figures, gets the disclosure note instead.

    Trafilatura's own precision filter drops most embed credit blocks ("Created
    with Infogram" and friends), so the char floor carries this on its own and no
    boilerplate-pattern list is needed.
    """
    return len(text.strip()) < RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS


def content_share(text: str) -> float:
    """Share of an extraction's characters that sit in content-shaped lines.

    Content is table rows (lines starting with ``|``, whatever their length: a price-history
    table is rows of 10-char cells) and lines of at least
    ``RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS``; every other line is chrome-shaped. Lines are
    stripped and blank ones dropped first. One pass over the extracted text, no second parse.

    A line-shape rule separates navigation trees from content and nothing more: prose-shaped
    boilerplate (a cookie-consent wall, a glossary) is sentences and passes, and a news ticker
    made of short headlines is withheld with the menu around it. Both are the deliberate
    trade; the calibration numbers sit on ``RESOLUTION_SOURCE_CONTENT_SHARE_MIN``.
    """
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    total = sum(len(line) for line in lines)
    if total == 0:
        return 0.0
    content = sum(
        len(line) for line in lines if line.startswith("|") or len(line) >= RESOLUTION_SOURCE_CONTENT_LINE_MIN_CHARS
    )
    return content / total


def _unreadable_embed_disclosure(providers: list[str]) -> str:
    """The one-line note a rendered page carries when it hides figures in an embed.

    Forecaster-facing and deliberately plain: the section it sits in is captioned
    "primary grading evidence", so a page whose resolving numbers are NOT in the
    text has to say so or the caveat overstates what was retrieved. No count of
    embeds — one embed can be referenced by both a container div and a loader
    script, and an overstated count in evidence prose is its own small fabrication.
    """
    return (
        f"[This page displays data through {', '.join(providers)} embed(s) that this fetch cannot read — "
        f"any figures shown inside them are NOT in the page text below.]"
    )


def _page_text_with_leads(extracted: str, url: str, providers: list[str], chart_block: str = "") -> str:
    """Per-URL-capped page text, LED by the chart-data block and the embed disclosure.

    Both leads lead (exactly like the Tier-2 dataset lead) because every truncator
    on this text is head-preserving, so anything at the tail is the first thing a
    later trim discards. As a trailer the disclosure survived the per-URL truncation
    here but not the aggregate `_budgeted_success_sections` cut, which re-truncates
    an over-budget body through `_truncate_with_marker` — on prod constants (5 x
    6000 per-URL against an 18000 total) a fourth Infogram page rendered under the
    "primary grading evidence" caption with the disclosure gone and only a generic
    truncation marker left, which is the q44554/44556 failure the disclosure exists
    to prevent. Leading it also puts the caveat ahead of the text it qualifies,
    which is why the wording says "below".

    Chart data goes ABOVE the disclosure: on a page whose prose carries none of the
    resolving figures (q43949) it is the only resolving content in the section, so
    it must be the last thing any trim reaches, and the disclosure then still sits
    immediately above the prose it qualifies.

    Both leads are budgeted out of the cap rather than added on top, so the per-URL
    bound the section budget relies on still holds — including in the pathological
    case where the leads alone exceed the cap (a test can tune the cap below the
    chart block's own).
    """
    leads = [lead for lead in (chart_block, _unreadable_embed_disclosure(providers) if providers else "") if lead]
    if not leads:
        return _truncate_with_marker(extracted, RESOLUTION_SOURCE_PER_URL_MAX_CHARS, url)
    return _lead_then_capped_body("\n\n".join(leads), extracted, url)


def _lead_then_capped_body(lead: str, body: str, url: str) -> str:
    """A provenance lead, then as much of ``body`` as the per-URL cap leaves, inside the bound.

    The one arithmetic every rung that serves an artifact under a lead uses: the chart-data /
    embed leads here, the derived feed, the archived capture, the model's reading. The lead LEADS
    because every truncator on this text is head-preserving, so anything at the tail is the first
    thing a later trim discards, and a lead trimmed off leaves the artifact passed off as the live
    page (the q44554/44556 failure). Its cost comes OUT of the cap rather than on top of it, so
    the per-URL bound ``_budgeted_success_sections`` relies on still holds — including the
    pathological case where the lead alone exceeds the cap, where the lead itself is truncated
    (a bare lead there busts the bound the aggregate budget assumes). A blank body renders the
    lead alone for the same reason.
    """
    body_cap = RESOLUTION_SOURCE_PER_URL_MAX_CHARS - len(lead) - 2
    if body_cap <= 0 or not body.strip():
        return _truncate_with_marker(lead, RESOLUTION_SOURCE_PER_URL_MAX_CHARS, url)
    return f"{lead}\n\n{_truncate_with_marker(body, body_cap, url)}"


def _budgeted_success_sections(
    successes: list[FetchResult], fetched_iso: str
) -> tuple[list[str], list[FetchResult], int]:
    """Render the success sections inside the two partitioned budgets.

    Returns ``(sections, kept, dropped)``: the rendered sections, the results they were rendered
    FROM in the same order, and how many successes the budget dropped outright. ``kept`` exists
    for the route caveats, which describe an artifact a forecaster can see and so must be
    computed over what renders rather than over every success.

    Cited pages and Tier-2 datasets draw on separate allowances, so a chart's rows can
    never evict the page text the section exists to serve.
    """
    sections: list[str] = []
    kept: list[FetchResult] = []
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
        kept.append(r)
    return sections, kept, dropped


def _route_caveats(rendered: list[FetchResult]) -> list[str]:
    """One sentence per non-direct route present in the sections that RENDER.

    Computed over the successes the section budget KEPT rather than over every result, because a
    caveat describes an artifact a forecaster can see: a rung that fired and failed left the
    direct route's own outcome, which the failure notice already names, and a success the
    aggregate budget dropped has no section below for the sentence to describe (on prod
    constants, 5 x 6000 per-URL pages against an 18000 total, a rendered page cited last was
    disclosed and then omitted). Order comes from ``ROUTE_CAVEATS``' own insertion order, so it
    is stable across questions rather than following fetch order.

    Empty for an all-direct question, which is the overwhelming majority and the case whose
    rendered section has to stay byte-identical to what it was before the ladder existed.
    """
    return [caveat for route, caveat in ROUTE_CAVEATS.items() if any(r.route == route for r in rendered)]


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
    yielded no usable content" notices and get their own withheld line instead.
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
        # "yielded no usable content", not "could not be fetched / was unreachable":
        # `no_resolving_content` and `empty_body` are pages that ANSWERED 200 and carried
        # nothing, and telling a forecaster the source was unreachable misstates the null
        # they have to weigh — "the tracker was down" and "the tracker has no reading" are
        # different pieces of evidence. The per-domain status token says which it was.
        notice = (
            f"[{n} resolution source(s) yielded no usable content: {_render_fetch_failures(cited_failures)}] — "
            f"nothing from the cited resolving page(s) is in this bundle; weight other evidence accordingly."
        )
        if dataset_nonsuccesses:
            notice += "\n\n" + _dataset_withheld_note()
        return notice

    fetched_iso = fetched_at.strftime("%Y-%m-%d")
    sections, kept, dropped = _budgeted_success_sections(successes, fetched_iso)
    caveat = "\n".join(
        [
            f"Snapshot of the cited resolution source(s) as of {fetched_iso} — primary grading evidence.",
            *_route_caveats(kept),
        ]
    )

    rendered = caveat + "\n\n" + "\n\n".join(sections)
    if dropped:
        rendered += f"\n\n[{dropped} additional source(s) omitted — section budget]"
    if cited_failures:
        rendered += (
            f"\n\n[Note: {len(cited_failures)} other cited resolution source(s) yielded no usable content: "
            f"{_render_fetch_failures(cited_failures)} — weight accordingly.]"
        )
    if dataset_nonsuccesses:
        rendered += "\n\n" + _dataset_withheld_note()
    return rendered


# ---------------------------------------------------------------------------
# Extraction wrapper (isolated for tests; offloads trafilatura's sync API)
# ---------------------------------------------------------------------------


def _extract_main_text(body: bytes | str, url: str, *, favor_precision: bool = False) -> str | None:
    """Trafilatura extraction. Callers wrap in ``await asyncio.to_thread(...)``.

    Takes bytes (the response body, letting trafilatura detect the encoding) or text
    (a body this module already decoded and rewrote — see :func:`_extract_page_text`).

    Default recall is the primary extraction and ``favor_precision=True`` the fallback, under
    the policy :func:`_extract_page_text` applies. Precision alone shipped here until
    2026-09-03 and withheld readable pages (kasa.go.kr pruned to 78 chars, two tracxn funding
    tables, manifold's market body). Default alone then shipped for a day, measured by
    character count, which is the wrong metric under a head-preserving cap: on congress.gov
    it swaps the 2,411-char bill-status card for 54,393 chars of a member-name dropdown
    (trafilatura's readability fallback replaces the main extraction when readability's
    text is over twice as long, and only precision prunes the dropdown out of that backup
    tree first), and menu trees (abs.gov.au, kasa.go.kr) clear the chrome floor as
    `success`. The receipt for running both is the 2026-09-03 calibration
    (`scratch/fetch_ladder_2026-09-03/chrome_calibration.md`: 118 bodies, five extractor
    variants on identical bytes, texts labelled by hand). ``include_comments=False`` stays
    at both settings.

    Returns None on empty/failed extraction so callers can classify.
    """
    try:
        out = trafilatura.extract(
            body,
            url=url,
            include_comments=False,
            include_tables=True,
            output_format="txt",
            favor_precision=favor_precision,
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


# Floor under the per-hop timeout `_fetch_one_hop` derives from the remaining wall budget.
# A hop reached with the budget already spent still gets a token attempt rather than a
# guaranteed-expired one: nothing downstream distinguishes "timed out at 0.0 s" from "timed
# out at 0.5 s", and a fast host answering in 200 ms is a page we would otherwise refuse for
# free. Small enough that the overshoot stays well inside RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S.
_MIN_HOP_TIMEOUT_S: float = 0.5

_HTML_CONTENT_TYPES = ("text/html", "application/xhtml+xml")
_RAW_TEXT_CONTENT_TYPES = ("text/plain", "text/csv")
_JSON_CONTENT_TYPES = ("application/json",)
_PDF_CONTENT_TYPES = ("application/pdf", "application/x-pdf")


@dataclass
class QuestionRungBudget:
    """The rung allowances one QUESTION shares across its cited URLs.

    Separate from :class:`FetchContext`, which is per-URL, because the thing being bounded is
    per-question: every Wayback snapshot shares netloc ``web.archive.org``, so the loop-wide
    per-host ``Semaphore(1)`` turns N cited URLs into N sequential archive fetches inside a wall
    that discards work already done when it fires. Its default is a fresh budget, so a
    monkeypatched fetch driven with one URL and no shared state behaves exactly as it did.
    """

    wayback_attempts_left: int = RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS
    # One browser escalation per host at a time WITHIN the question, see `browser_escalation_gate`.
    browser_escalation_gates: dict[str, asyncio.Semaphore] = field(default_factory=dict)

    def take_wayback_attempt(self) -> bool:
        """Claim one snapshot attempt for this question, or False when they are spent."""
        if self.wayback_attempts_left <= 0:
            return False
        self.wayback_attempts_left -= 1
        return True

    def browser_escalation_gate(self, url: str) -> asyncio.Semaphore:
        """The ``Semaphore(1)`` that serializes this question's derived-feed-then-browser
        escalations on ``url``'s host.

        The derived-API rung exists so a host with several cited URLs pays for ONE Chromium
        launch, but the provider fans one task out per cited URL, so every same-host URL asked
        ``endpoint_for`` before any render had finished, got None, queued on the per-host gate
        inside the render, and launched its own browser after the first had already recorded
        the endpoint. Holding this gate across the pair means the second URL re-asks once the
        first's escalation is over and takes the feed off an ordinary GET instead. Waiting here
        costs the second URL nothing it did not already pay queueing on the host gate inside the
        render, and the rungs behind it re-read their wall budget after the wait.

        Per question rather than loop-wide on purpose: the cross-question shape still
        serializes on the loop-wide host gate exactly as before, and a loop-wide gate here
        would be one more unbounded process-global acquire in front of a wall that discards
        finished work (FUTURE.md item 5, the operator's call).
        """
        return semaphore_for_host(url, self.browser_escalation_gates)


# The human phrase each rung's wall-budget skip logs, keyed by route so the one message template
# in `FetchContext.claim_rung_budget` reads the same as the six hand-copied lines it replaced.
_RUNG_WALL_SKIP_PHRASE: dict[FetchRoute, str] = {
    "meta_refresh": "the meta-refresh hop",
    "pdf_local": "the local PDF read",
    "derived_api": "the derived-feed GET",
    "rendered": "the rendered rung",
    "wayback": "the wayback rung",
    "url_context": "the url_context rung",
}


@dataclass
class FetchContext:
    """Per-URL inputs and rung bookkeeping for one :func:`_fetch_one` call.

    ONE per fetched URL, so ``rungs`` belongs to that URL and can be stamped onto its
    result; ``query``, ``started``, ``now`` and ``shared`` are the same for every URL in a
    provider call. Every field has a default so the monkeypatched fetch surface can still be
    driven with three positional arguments, and a default context is simply "no
    question text, clock starts now" — which gives a direct fetch exactly the behaviour
    it had before the ladder existed.

    ``query`` is the question's title plus its resolution criteria, and it is what
    decides WHICH passages of a 220-page PDF a forecaster sees. ``started`` is the
    provider's own wall-clock origin, so every rung can bound itself against the same
    45 s the outer ``asyncio.wait_for`` uses. ``now`` is the WALL-CLOCK counterpart, which the
    Wayback rung ages a capture against — a monotonic origin cannot date anything, and taking
    the clock inside the rung would make an archived snapshot's rendered disclosure depend on
    when it happened to run rather than on the fetch it belongs to.

    ``fast_path`` is the question's time-budget thin-window mode (``time_budget.py``), handed
    down from the orchestrator through the provider factory. The two EXPENSIVE rungs — the
    browser and the paid reader — decline on it before any side effect, recording a
    ``fast_path`` skip; the cheap rungs run as they do off it. It only ever declines, so a
    question with no fast path is byte-identical to one before the gate existed.
    """

    query: str = ""
    started: float = field(default_factory=time.monotonic)
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    shared: QuestionRungBudget = field(default_factory=QuestionRungBudget)
    rungs: list[RungAttempt] = field(default_factory=list)
    fast_path: bool = False

    def rung_budget_s(self) -> float:
        """Wall-clock seconds a rung may spend before the outer ``wait_for`` fires.

        Same arithmetic as the Datawrapper hop's, and for the same reason: that timeout
        discards every page that already fetched, so a rung that overruns costs the
        whole question's resolution evidence rather than just its own attempt.
        """
        return RESOLUTION_SOURCE_WALL_TIMEOUT - (time.monotonic() - self.started) - RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S

    def claim_rung_budget(
        self, rung: FetchRoute, from_status: FetchStatus, url: str, floor_s: float, *, note: str = ""
    ) -> float | None:
        """The remaining wall budget for ``rung``, or None once it has recorded the skip.

        The one home for the wall-budget preamble every rung copied: read the remaining wall,
        and below the rung's own floor log once, record a ``wall_budget`` skip on this context and
        return None; otherwise hand back the budget the rung sizes its work off. Structural rather
        than stylistic — an incomplete copy still returned None and simply never appeared in the
        ``rung_budget_skips`` count, so the archive under-reported how often the wall is the
        binding constraint. ``note`` is the one place the message varies (the paid rung's second
        check adds " after the robots pre-check", the PDF read's re-check " after queueing").
        """
        budget_s = self.rung_budget_s()
        if budget_s < floor_s:
            logger.warning(
                "resolution_source: skipping %s for %s — %.1fs of wall budget left%s",
                _RUNG_WALL_SKIP_PHRASE[rung],
                urlparse(url).netloc,
                budget_s,
                note,
            )
            self.skip_rung(rung, from_status, url, "wall_budget")
            return None
        return budget_s

    def start_rung(self, rung: FetchRoute, from_status: FetchStatus, url: str) -> RungAttempt:
        attempt = RungAttempt(rung=rung, from_status=from_status, url=url, started_at=time.monotonic())
        self.rungs.append(attempt)
        return attempt

    def skip_rung(self, rung: FetchRoute, from_status: FetchStatus, url: str, reason: str) -> None:
        self.rungs.append(
            RungAttempt(
                rung=rung,
                from_status=from_status,
                url=url,
                started_at=time.monotonic(),
                wall_s=0.0,
                skipped_reason=reason,
            )
        )

    def close_rungs(self, first_new: int, outcome: FetchStatus) -> None:
        """Close every attempt opened since ``first_new`` with ONE clock reading and one outcome.

        The dispatcher calls this the moment a rung's result is known — ``first_new`` is
        ``len(ctx.rungs)`` read before the rung ran — so each attempt's ``wall_s`` measures
        that rung alone and its ``outcome`` is the status that stood once it was over: the
        rescue it returned, or the direct status it left standing when it declined. A rung
        that already stamped either field for itself keeps its stamp (``finish`` respects the
        PDF read's own ``wall_s``; the browser rung's own ``outcome`` is the rendered DOM's
        verdict), which is what keeps a harvested feed's rescue from being credited to the
        render that only found the endpoint.
        """
        now = time.monotonic()
        for attempt in self.rungs[first_new:]:
            attempt.finish(now)
            if attempt.outcome is None and not attempt.skipped_reason:
                attempt.outcome = outcome


def _aux_ctx(ctx: FetchContext) -> FetchContext:
    """A child context for a request a rung makes on the cited URL's BEHALF.

    The Wayback snapshot, the remembered derived feed and the robots.txt pre-check all go
    through :func:`_fetch_direct`, whose own rungs (the meta-refresh hop, the local PDF read)
    stamp attempts onto whatever context they are handed. On the PAGE's context those stamps
    hijack the record: an archived PDF capture came back ``route="pdf_local"`` — ``route`` is
    the last rung that fired — was counted as a Wayback attempt AND a document read, and lost
    the archived-copy caveat that ``_route_caveats`` keys on the route. The child shares the
    clock, the query, the wall-clock origin and the per-question budget, and owns a rung list
    nobody reads, which is :class:`FetchContext`'s one-per-fetched-URL invariant applied to a
    URL the question never cited. Its attempts are deliberately NOT merged back: that would put
    ``pdf_local`` last again and the route would still be wrong.
    """
    return replace(ctx, rungs=[])


def _skip_for_fast_path(ctx: FetchContext, rung: FetchRoute, direct: FetchResult, url: str) -> None:
    """Record that an EXPENSIVE rung declined because the question is on the time-budget fast path.

    Its own token rather than ``wall_budget``: the wall here is the provider's fixed 45 s, which
    a fast-path question may have plenty of, and a residual round reading the counts has to be
    able to tell "the question's close left no room for a browser" from "this rung ran out of
    the provider's own clock". The saving is narrower than the fast path's name implies — a
    close-limited budget under the intake floor is skipped outright, so the band this gate
    protects is the high-pre-research-elapsed question — but a 12-35 s Chromium launch inside a
    thin window is still a launch the prediction POST would rather have.
    """
    logger.info(
        "resolution_source: declining the %s rung for %s — the question is on the time-budget fast path",
        rung,
        urlparse(url).netloc,
    )
    ctx.skip_rung(rung, direct.status, url, "fast_path")


def _stamped_with_route(result: FetchResult, ctx: FetchContext) -> FetchResult:
    """Attach the ladder bookkeeping to a finished result.

    ``route`` is the LAST rung that fired, which is the one that produced this outcome
    (a meta-refresh hop onto a PDF reads ``pdf_local``: the hop got us the bytes, the
    local read is what the text came from). Skipped rungs never claim the route.

    Every rung the dispatcher ran has already been closed with its own wall and outcome
    (:meth:`FetchContext.close_rungs`); the close here is the last resort for an attempt a
    future rung opens without bracketing, and stamps it with the final status rather than
    leaving the marker to print ``None``.
    """
    ctx.close_rungs(0, result.status)
    result.rung_attempts = list(ctx.rungs)
    fired: list[FetchRoute] = [attempt.rung for attempt in ctx.rungs if not attempt.skipped_reason]
    if fired:
        result.route = fired[-1]
    return result


def _sem_for_host(host_sems: dict[str, asyncio.Semaphore], url: str) -> asyncio.Semaphore:
    """Get-or-create the ``Semaphore(1)`` for ``url``'s netloc.

    Every task in one :func:`fetch_resolution_sources` run shares the same
    ``host_sems`` map, so every request to a given host — original URL or
    redirect hop — contends on the same semaphore object. Since 2026-09-03 that map
    is :func:`http_fetch.host_semaphores`, shared by every question running
    concurrently rather than rebuilt per provider call; the parameter stays because
    the gap-fill v2 loop reaches this function with its own map.

    Kept as a thin wrapper over the shared implementation because the test suites
    monkeypatch THIS name to observe or replace the gate.
    """
    return semaphore_for_host(url, host_sems)


async def _vetted_hop_target(
    target: str, current_url: str, *, http_status: int, content_type: str, kind: str
) -> FetchResult | str:
    """The absolute next URL for a derived hop, or the terminal refusal it earns.

    The ONE place a URL this module derived from a response — a ``Location`` header, a
    meta-refresh tag — passes the two checks every hop owes: the ``is_public_http_url``
    preflight (the fast-fail SSRF view; the connect-time resolver stays the real
    boundary) and the Metaculus self-reference refusal. Shared so a third rung cannot
    ship with one of them missing, and ``kind`` is only there to say which hop shape a
    log line came from.
    """
    next_url = urljoin(current_url, target)
    if not await is_public_http_url(next_url):
        logger.warning(
            f"resolution_source ssrf_blocked ({kind}): {urlparse(current_url).netloc} -> {urlparse(next_url).netloc}"
        )
        return FetchResult(
            url=next_url,
            status="ssrf_blocked",
            text="",
            http_status=http_status,
            content_type=content_type or None,
        )
    if is_metaculus_self_ref(next_url):
        # The URL pre-filter drops metaculus self-refs, but a redirect (of either
        # shape) can still land on metaculus.com; don't follow it (no new info,
        # and keeps our IP off the same host the critical API uses).
        logger.info(
            f"resolution_source metaculus_self_ref ({kind}): "
            f"{urlparse(current_url).netloc} -> {urlparse(next_url).netloc}"
        )
        return FetchResult(
            url=next_url,
            status="blocked",
            text="",
            http_status=http_status,
            content_type=content_type or None,
        )
    return next_url


async def _resolution_redirect_outcome(resp: Any, current_url: str, content_type: str) -> FetchResult | str:
    """Vet a 3xx hop: the next URL to follow, or a terminal error/blocked result."""
    status = resp.status
    location = resp.headers.get("Location") if resp.headers else None
    if not location:
        # Malformed redirect — no Location header.
        logger.info(f"resolution_source {urlparse(current_url).netloc}: {status} redirect with no Location header")
        return FetchResult(
            url=current_url,
            status="error",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    return await _vetted_hop_target(
        location, current_url, http_status=status, content_type=content_type, kind="redirect"
    )


def _resolution_status_outcome(status: int, current_url: str, content_type: str) -> FetchResult | None:
    """Terminal result for a non-200 status, or None when the body should be read."""
    if status == 200:
        return None
    fetch_status = _NON_OK_FETCH_STATUS.get(status, "error")
    return FetchResult(
        url=current_url,
        status=fetch_status,
        text="",
        http_status=status,
        content_type=content_type or None,
    )


@dataclass(frozen=True, slots=True)
class _PageExtraction:
    """What the extractor policy decided for one HTML body (:func:`_extract_page_text`).

    ``text`` is what the classifier sees: the precision re-extraction when it rescued the
    page, else the default one (None when nothing extracted). ``chrome_metric_withheld``
    marks a default extraction that cleared the chrome floor and failed the line-shape
    metric with no rescue, so the classifier withholds it; ``precision_rescued`` marks a
    text that came from the fallback. Both ride the result into ``details["counts"]``.
    """

    text: str | None
    chrome_metric_withheld: bool = False
    precision_rescued: bool = False


def _extract_page_text(html_text: str, body: bytes, url: str, undecodable_ratio: float) -> _PageExtraction:
    """The publishable extraction of an HTML body: ARIA tables rewritten first, default
    recall as the primary extractor, precision as the fallback, both scored by line shape.

    All of it is CPU-bound sync work over a body up to the response cap, so it runs in one
    ``asyncio.to_thread`` hop rather than several.

    The policy, calibrated 2026-09-03 (receipt on ``RESOLUTION_SOURCE_CONTENT_SHARE_MIN``):
    the default extraction publishes when it clears the chrome floor and
    :func:`content_share` is at least the threshold. When it clears the floor on chrome
    alone, the same input is re-extracted under ``favor_precision=True``, which is the one
    setting that prunes the tree readability's fallback swapped in, and that text publishes
    only if it clears the floor AND the metric. Otherwise the page is chrome and the
    classifier withholds it under ``thin_page``, so the rendered rung still fires
    (uk.finance.yahoo's direct body is a menu plus one quote line; its render is the whole
    price table). On the calibration corpus this publishes every labelled content text (46
    of 46, three of them the congress.gov status card the default alone loses) and blocks the
    navigation-tree chrome (abs, ocearch, portwatch, copernicus, the congress dropdown), at
    the cost of one extra trafilatura pass on the pages that fail the metric. What it gives
    up: prose-shaped boilerplate (a cookie-consent wall, a glossary) passes any line-shape
    metric, and kasa.go.kr's news ticker is withheld with its menu. An extraction under the
    floor skips the metric, because precision only ever shortens.

    Trafilatura gets the ORIGINAL BYTES in two cases, and in both its extraction is
    byte-identical to what it was before this rung existed: a page with no ARIA role at
    all, and a page our own decode mangled. The second is the one that matters —
    ``decode_text_body`` honours a BOM and the HTTP header's ``charset``, but a page that
    declares its encoding only in a ``<meta charset>`` decodes as UTF-8 here and comes
    back as mojibake, while trafilatura reading the bytes would have found the meta
    declaration. Handing it the rewritten mojibake instead would lose a page we can read
    today, so the rewrite is trusted ONLY on a body that decoded cleanly.

    That is why the gate is ``== 0.0`` and not ``MAX_UNDECODABLE_CHAR_RATIO``: the shared
    bound is the refuse-the-whole-body threshold and is far too loose for this decision. A
    mostly-ASCII cp1252 page whose only non-UTF-8 bytes are accented characters scores
    around 0.01 against a bound of 0.10, so under the old gate it took the rewrite and
    reached forecasters as ``R<?>sum<?> ... Qu<?>bec`` where the bytes path returns the
    accents. Any U+FFFD at all means this decode lost information trafilatura might not
    have, and the rewrite is the only thing that forecloses its own encoding detection.
    """
    rewritten = rewrite_aria_tables(html_text) if undecodable_ratio == 0.0 else None
    source = body if rewritten is None else rewritten
    default = _extract_main_text(source, url)
    if (
        default is None
        or looks_like_page_chrome(default)
        or content_share(default) >= RESOLUTION_SOURCE_CONTENT_SHARE_MIN
    ):
        return _PageExtraction(text=default)
    precision = _extract_main_text(source, url, favor_precision=True)
    if (
        precision is not None
        and not looks_like_page_chrome(precision)
        and content_share(precision) >= RESOLUTION_SOURCE_CONTENT_SHARE_MIN
    ):
        return _PageExtraction(text=precision, precision_rescued=True)
    return _PageExtraction(text=default, chrome_metric_withheld=True)


def _no_content_verdict(
    extracted: str | None, unreadable_embeds: list[str]
) -> tuple[FetchStatus, FetchStatusReason | None]:
    """Which withhold a 200 with no readable content earns, and why.

    Order is load-bearing and unchanged since the chrome floor generalised: a named
    routeless embed is the most specific thing we can say (`embed_shell` — the numbers
    exist and we have no route to them), the JS-wall floor keeps its own much lower
    threshold and its position in the middle so the chrome floor cannot swallow that
    population, and `thin_page` is everything else: under the floor, or over it on chrome
    alone (the line-shape metric in :func:`_extract_page_text`).
    """
    if unreadable_embeds:
        # Datawrapper is exempt from the embed scan (it has the Tier-2 hop), so a
        # walled tracker still comes back `js_wall` below and still hops.
        return "no_resolving_content", "embed_shell"
    if extracted is None or looks_like_js_wall(extracted):
        # An empty extraction on a 200 OK is a JS-wall (SPA that rendered client-side,
        # cookie/consent gate, etc.) — exactly the Tier-2 candidate signal. Treated
        # identically to short-but-nonempty extractions.
        return "js_wall", None
    return "no_resolving_content", "thin_page"


async def _meta_refresh_hop(
    html_text: str,
    current_url: str,
    ctx: FetchContext,
    *,
    from_status: FetchStatus,
    http_status: int,
    content_type: str,
) -> FetchResult | str | None:
    """Follow a ``<meta http-equiv="refresh">`` stub, or None when there is nothing to follow.

    A hop rather than a terminal result on purpose: the target re-enters the same
    classification path (chrome floor, JS-wall floor, chart rung, PDF read) and consumes
    one of ``MAX_REDIRECTS``, so a refresh chain is bounded exactly like a 3xx chain and
    the meta-refresh check itself works on a body only a later hop could obtain.

    Only reached with no readable content, which is what keeps it off the pages that
    already worked: a real page that ALSO carries a refresh tag (some CMSs emit one for
    a canonical URL) is served as-is rather than re-fetched.
    """
    target = meta_refresh_target(html_text)
    if target is None:
        return None
    if (
        ctx.claim_rung_budget("meta_refresh", from_status, current_url, RESOLUTION_SOURCE_META_REFRESH_MIN_BUDGET_S)
        is None
    ):
        return None
    ctx.start_rung("meta_refresh", from_status, current_url)
    logger.info(
        f"resolution_source meta_refresh: {urlparse(current_url).netloc} -> {target} (direct read was {from_status})"
    )
    return await _vetted_hop_target(
        target, current_url, http_status=http_status, content_type=content_type, kind="meta_refresh"
    )


@dataclass(frozen=True, slots=True)
class _HtmlClassification:
    """One classified HTML body, plus the decoded text the meta-refresh rung still needs.

    ``html_text`` rides along because the two callers want different things from the same
    decode: :func:`_resolution_html_outcome` looks for a refresh stub in it, while the
    rendered rung has already followed every hop a browser follows and only wants the verdict.
    Decoding twice would double the CPU on a body up to the 5 MiB response cap, or a rendered
    DOM up to ``RENDERED_DOM_MAX_CHARS`` (sized to it).
    """

    result: FetchResult
    html_text: str


async def _classify_html_body(
    body: bytes, current_url: str, content_type: str, *, http_status: int
) -> _HtmlClassification:
    """Trafilatura extraction plus the inline-chart rung and the chrome / JS-wall checks.

    The ONE classification path for an HTML body, whichever rung obtained it: the direct
    fetch, a meta-refresh hop, or a headless-Chromium render. That is what makes a rescued
    page indistinguishable from a directly-fetched one downstream — same chart read, same ARIA
    rewrite, same floors, same disclosure leads.

    Order of the three verdicts, and why:

    1. CONTENT is extracted text that clears the chrome floor and the line-shape metric
       (:func:`_extract_page_text`, which retries under precision when the default
       extraction is chrome-shaped) OR chart data read out of the raw HTML. The chart
       rung runs on every HTML page, not only thin ones, because q43949's page
       extracted ~80k chars of prose with none of the resolving figures in it — a
       thin-only gate would miss the record the rung exists for.
    2. With no content, a named routeless embed makes it `embed_shell`, an
       extraction under the JS-wall floor makes it `js_wall`, and anything else,
       under the chrome floor or over it on short lines alone, makes it `thin_page`.
       The `js_wall` check keeps its exact old meaning and its position between the
       two, so the generalised chrome floor cannot swallow the JS-wall population.
    3. Chart data therefore rescues a page the chrome floor would have withheld —
       including a JS-walled one, where the config in the raw HTML is precisely the
       data the wall was hiding. That is the one place the `js_wall` outcome moves,
       and it moves only when we actually recovered the numbers.
    """
    # Both embed scans are only possible on the RAW HTML —
    # trafilatura drops iframes and embed scripts at every
    # setting — so they run on the raw body, before (and
    # regardless of) main-text extraction. Decoded through the
    # shared helper so a BOM'd / non-UTF-8 page's embeds are
    # still findable; the page's main text is trafilatura's to
    # decode, which is why no vacuity check runs on this branch
    # (a thin extraction is classified below instead).
    html_text, undecodable_ratio = decode_text_body(body, content_type)
    charts = extract_datawrapper_charts(html_text)
    unreadable_embeds = unreadable_data_embed_providers(html_text)
    extraction = await asyncio.to_thread(_extract_page_text, html_text, body, current_url, undecodable_ratio)
    extracted = extraction.text
    # In a thread for the same reason the extraction is: it is sync CPU work (one
    # regex sweep plus a `json.loads` per config) over a body up to the 5 MiB
    # response cap — or a rendered DOM up to RENDERED_DOM_MAX_CHARS, the browser rung's
    # ceiling — and blocking the loop here would stall every sibling fetch.
    # Measured 22 ms on the 1.1 MB q43949 page, but the bound is the page, not that
    # sample. The Datawrapper / embed scans above are single regex searches and stay
    # inline.
    chart_block = await asyncio.to_thread(render_inline_chart_data, html_text)
    if (extraction.chrome_metric_withheld or looks_like_page_chrome(extracted or "")) and not chart_block:
        # No content anywhere. Which of the three withholds applies is a disclosure
        # question, not a routing one — all three retain the result as the escalation
        # seam and none of them render. A walled page still exposes its
        # embeds, so the charts ride along on every one of them.
        verdict, reason = _no_content_verdict(extracted, unreadable_embeds)
        return _HtmlClassification(
            result=FetchResult(
                url=current_url,
                status=verdict,
                text="",
                http_status=http_status,
                content_type=content_type or None,
                status_reason=reason,
                datawrapper_charts=charts,
                unreadable_embeds=unreadable_embeds,
                chrome_metric_withheld=extraction.chrome_metric_withheld,
            ),
            html_text=html_text,
        )
    return _HtmlClassification(
        result=FetchResult(
            url=current_url,
            status="success",
            text=_page_text_with_leads(extracted or "", current_url, unreadable_embeds, chart_block),
            http_status=http_status,
            content_type=content_type or None,
            datawrapper_charts=charts,
            unreadable_embeds=unreadable_embeds,
            precision_rescued=extraction.precision_rescued,
        ),
        html_text=html_text,
    )


async def _resolution_html_outcome(
    resp: Any, current_url: str, content_type: str, ctx: FetchContext
) -> FetchResult | str:
    """Classify the HTML body, then let the meta-refresh rung look for a hop no status announced.

    Only once there is no content anywhere does the meta-refresh rung run. It returns the
    target as the next hop, so this function's return type is ``FetchResult | str`` exactly
    like the redirect dispatcher's, and a refresh chain is bounded by the same
    ``MAX_REDIRECTS`` cap with the same per-hop SSRF re-guard.
    """
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
    classified = await _classify_html_body(body, current_url, content_type, http_status=status)
    if classified.result.status == "success":
        return classified.result
    hop = await _meta_refresh_hop(
        classified.html_text,
        current_url,
        ctx,
        from_status=classified.result.status,
        http_status=status,
        content_type=content_type,
    )
    if hop is not None:
        return hop
    return classified.result


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
        # Reason line, not an outcome line: the marker carries the status, this
        # carries the body size and decode score that explain it.
        logger.info(
            f"resolution_source {netloc}: 200 body carries no usable content "
            f"({vacuous}, {len(body)} bytes, undecodable={undecodable_ratio:.2f})"
        )
        return FetchResult(
            url=current_url,
            status=vacuous,
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    return FetchResult(
        url=current_url,
        status="success",
        text=_truncate_with_marker(raw, RESOLUTION_SOURCE_PER_URL_MAX_CHARS, current_url),
        http_status=status,
        content_type=content_type or None,
    )


def _pdf_unreadable_reason(pdf: PdfText) -> FetchStatusReason:
    """Why a document we read the bytes of yielded no text.

    ``encrypted`` / ``malformed`` come from the parse; ``no_text_layer`` is a document
    that parsed fine and carries images instead of text, which is the ONE shape a paid
    document read could still rescue.
    """
    if pdf.unreadable_reason == "encrypted":
        return "encrypted"
    if pdf.unreadable_reason == "malformed":
        return "malformed"
    return "no_text_layer"


@dataclass(frozen=True, slots=True)
class _PendingDocument:
    """A PDF whose bytes we hold and whose parse has not started yet.

    Exists so the parse happens OUTSIDE the per-host politeness semaphore. That map is
    loop-wide, so a 20 s parse held inside it blocked every other concurrent question's
    fetch of any URL on that host — and this population is concentrated on a handful of
    government hosts, so same-host collisions across questions in one round are the
    expected case. Two questions queued behind one parse of a shared host exhaust their own
    ``RESOLUTION_SOURCE_WALL_TIMEOUT``, and the outer ``wait_for`` then discards every page
    they had already fetched.
    """

    url: str
    body: bytes
    http_status: int
    content_type: str
    from_status: FetchStatus


def _parse_and_digest(
    body: bytes, *, max_seconds: float, query: str, source_url: str
) -> tuple[PdfText, DocumentDigest | None]:
    """pypdf parse plus BM25 passage selection: both CPU-bound, so ONE thread hop, never two.

    The digest is as CPU-bound as the parse and was running inline on the loop two lines
    below a call carefully threaded for exactly that reason: ``select_passages`` tokenises
    every window of the joined document and holds a ``Counter`` per window alive at once,
    measured at 96-235 ms per 400-page document and additive across the six concurrent
    questions — a stall that lands inside the 2 s ``RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S``
    and delays every sibling provider's I/O, not just this fetch.

    ``None`` for the digest means the document carried no text layer, which the caller
    reports rather than digests.
    """
    pdf = extract_pdf_text(body, max_pages=DOCUMENT_TEXT_MAX_PAGES, max_seconds=max_seconds)
    if not has_text_layer(pdf):
        return pdf, None
    return pdf, digest_pdf(
        pdf,
        query=query,
        top_k=DOCUMENT_DIGEST_TOP_K,
        max_chars=RESOLUTION_SOURCE_PER_URL_MAX_CHARS,
        source_url=source_url,
    )


async def _resolution_pdf_outcome(
    resp: Any, current_url: str, content_type: str, ctx: FetchContext, *, from_status: FetchStatus
) -> FetchResult | _PendingDocument:
    """Read a PDF we are already holding, locally, and render the query-relevant passages.

    Free and deterministic: pypdf plus BM25 passage selection (``research/document_text``),
    no model call and no second request. Before this rung a cited PDF was the one
    resolution source we dropped unread — measured at 833,450 chars in 5.3 s out of the
    6.7 MB 220-page document behind the constants, with the passage the reader wanted in
    it, while the paid alternative returned nothing for the same file.

    Byte cap depends on whether the server DECLARED a PDF. A declared one gets
    ``DOCUMENT_TEXT_PDF_MAX_BYTES``, not the 5 MiB response cap the text branches use,
    because the receipt file is 6.7 MB and the general cap would refuse exactly the
    document that motivated the rung. An UNDECLARED body — the sniffed case — keeps the
    5 MiB cap: it is far more likely to be an image or an archive than a document, and
    buffering 40 MiB of it per URL across every concurrent question is a memory cost
    with nothing on the other side. An undeclared PDF above 5 MiB is therefore still
    lost, which is a deliberate trade rather than an oversight.

    Self-bounding twice over. The parse is skipped outright below
    ``RESOLUTION_SOURCE_PDF_MIN_BUDGET_S`` of remaining wall (the bytes are still read —
    that already happened — but the CPU is not spent), and ``max_seconds`` is the
    remaining budget capped at ``DOCUMENT_TEXT_MAX_SECONDS``, so a 900-page document
    comes back partial-and-labelled rather than taking the outer wall down with every
    sibling page that already succeeded.

    This half runs INSIDE the response context, so it does only what needs the open
    response: the capped read, the ``%PDF-`` check and the budget-floor skip. A real
    document comes back as a :class:`_PendingDocument` and :func:`_finish_document` parses
    it once the host semaphore has been released.
    """
    status = resp.status
    netloc = urlparse(current_url).netloc
    declared_pdf = any(ct in content_type for ct in _PDF_CONTENT_TYPES)
    body = await read_body_capped(
        resp,
        max_bytes=DOCUMENT_TEXT_PDF_MAX_BYTES if declared_pdf else RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
        label=f"resolution_source pdf {netloc}",
    )
    if body is None:
        return FetchResult(
            url=current_url,
            status="error",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    if not is_pdf_body(body):
        # Declared a PDF and is not one (or carried no content type and is not one):
        # unchanged behaviour, minus the assumption that the label was right.
        logger.info(f"resolution_source {netloc}: body is not a document we can read, ct={content_type!r}")
        return FetchResult(
            url=current_url,
            status="unsupported_type",
            text="",
            http_status=status,
            content_type=content_type or None,
        )
    pending = _PendingDocument(
        url=current_url,
        body=body,
        http_status=status,
        content_type=content_type,
        from_status=from_status,
    )
    # Checked here, before the response context closes, so a question with no budget left never
    # even queues for a parse slot it would have to give back.
    if ctx.claim_rung_budget("pdf_local", from_status, current_url, RESOLUTION_SOURCE_PDF_MIN_BUDGET_S) is None:
        return _document_not_parsed(pending, "budget_skipped")
    return pending


async def _finish_document(pending: _PendingDocument, ctx: FetchContext) -> FetchResult:
    """Parse a held PDF and render its digest, with no host semaphore and no response held.

    Runs after :func:`_fetch_one_hop` has left both the ``session.get`` context and the
    per-host gate, which is the whole point: the parse is up to
    ``min(DOCUMENT_TEXT_MAX_SECONDS, budget)`` of CPU, and holding a loop-wide
    ``Semaphore(1)`` for a host through it stalls every other concurrent question's fetch of
    that host (see :class:`_PendingDocument`).

    The parse contends instead for :func:`http_fetch.pdf_parse_semaphore`, the loop-wide
    2-slot gate this route shares with the gap-fill v2 local-document ladder — the bound has
    to hold across the two routes, not inside each, because a Tier-1 fan-out alone is up to
    ``RESOLUTION_SOURCE_MAX_URLS`` documents per question across
    ``DEFAULT_MAX_CONCURRENT_RESEARCH`` questions. The wait is bounded by the remaining
    budget less the floor and degrades to the same leave-it-unread skip, since queueing until
    the outer wall fires would discard every sibling page that already succeeded.

    Never raises: ``extract_pdf_text`` returns a ``PdfText`` carrying ``unreadable_reason``
    rather than throwing, and the digest is pure.
    """
    netloc = urlparse(pending.url).netloc
    gate = pdf_parse_semaphore()
    budget_s = ctx.rung_budget_s()
    try:
        # Bounded, not a bare acquire: queueing behind two other documents until the outer
        # wall fires would discard every sibling page this question already fetched, which
        # costs strictly more than leaving one document unread. Leaving the floor unspent
        # means a slot won at the last moment still has time to parse something.
        await asyncio.wait_for(gate.acquire(), timeout=max(0.0, budget_s - RESOLUTION_SOURCE_PDF_MIN_BUDGET_S))
    except TimeoutError:
        logger.warning(
            "resolution_source: skipping the local PDF read for %s — no parse slot within %.1fs of wall budget",
            netloc,
            budget_s,
        )
        ctx.skip_rung("pdf_local", pending.from_status, pending.url, "parse_contention")
        return _document_not_parsed(pending, "parse_contention")
    try:
        # Re-read after the wait: the queue itself consumed budget, and `max_seconds` is
        # wall-clock, so a stale figure would hand pypdf a bound that already expired.
        budget_s = ctx.claim_rung_budget(
            "pdf_local", pending.from_status, pending.url, RESOLUTION_SOURCE_PDF_MIN_BUDGET_S, note=" after queueing"
        )
        if budget_s is None:
            return _document_not_parsed(pending, "budget_skipped")
        attempt = ctx.start_rung("pdf_local", pending.from_status, pending.url)
        pdf, digest = await asyncio.to_thread(
            _parse_and_digest,
            pending.body,
            max_seconds=min(DOCUMENT_TEXT_MAX_SECONDS, budget_s),
            query=ctx.query,
            source_url=pending.url,
        )
        # Stamped inside the gate so wall_s measures the parse this rung actually did, not
        # the time it spent queueing for a slot.
        attempt.wall_s = max(0.0, time.monotonic() - attempt.started_at)
    finally:
        gate.release()
    if digest is None:
        reason = _pdf_unreadable_reason(pdf)
        logger.warning(
            f"resolution_source {netloc}: PDF carried no readable text ({reason}, "
            f"{pdf.page_count} pages, {pdf.pages_read} read)"
        )
        return FetchResult(
            url=pending.url,
            status="unreadable_document",
            text="",
            http_status=pending.http_status,
            content_type=pending.content_type or None,
            status_reason=reason,
        )
    if not digest.passages:
        # A document we read END TO END that does not discuss the ask. Its block is the header,
        # the outline and one sentence saying nothing matched, which under the "primary grading
        # evidence" caption is prose standing in for an absent section: it counted the provider
        # as succeeded, defeated every downstream empty guard, and read in the run log exactly
        # like a document that handed the forecasters the resolving paragraph. Withheld like any
        # other content-free 200, with `no_matching_passage` saying which rule withheld it —
        # a document we DID read, which is why it is excluded from the paid rung's population
        # (:func:`_url_context_rung_applies`) that every other `no_resolving_content` is in.
        return FetchResult(
            url=pending.url,
            status="no_resolving_content",
            text="",
            http_status=pending.http_status,
            content_type=pending.content_type or None,
            status_reason="no_matching_passage",
        )
    return FetchResult(
        url=pending.url,
        status="success",
        text=digest.block,
        http_status=pending.http_status,
        content_type=pending.content_type or None,
    )


def _document_not_parsed(pending: _PendingDocument, reason: FetchStatusReason) -> FetchResult:
    """The result for a document we held and chose not to parse.

    ``unsupported_type`` rather than ``unreadable_document``: nothing read the bytes, so
    nothing established they carry no text, and only the latter is worth a paid document
    read later. ``reason`` says which rule declined — the same token the rung attempt's
    ``skipped_reason`` carries, repeated here because the two ride different markers
    (``RESOLUTION_SOURCE_ESCALATION`` versus ``RESOLUTION_SOURCE_FETCH``) and a reader of
    the per-fetch line should not have to join to learn we were holding a document.
    """
    return FetchResult(
        url=pending.url,
        status="unsupported_type",
        text="",
        http_status=pending.http_status,
        content_type=pending.content_type or None,
        status_reason=reason,
    )


async def _resolution_response_outcome(
    resp: Any, current_url: str, ctx: FetchContext
) -> FetchResult | _PendingDocument | str:
    """Classify one response: a terminal FetchResult, a held document, or the next hop's URL.

    The :class:`_PendingDocument` case is the PDF branch handing its parse back to the
    caller to run outside the host semaphore; every other branch is terminal or a hop.
    """
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
        return await _resolution_html_outcome(resp, current_url, content_type, ctx)
    if any(ct in content_type for ct in _JSON_CONTENT_TYPES) or any(
        ct in content_type for ct in _RAW_TEXT_CONTENT_TYPES
    ):
        return await _resolution_text_outcome(resp, current_url, content_type)

    # Everything else routes through the PDF rung, which reads the body and checks the
    # `%PDF-` magic before deciding anything. That covers a declared `application/pdf`
    # and the sniffed case: a missing/empty Content-Type header (ct=''), or a document
    # served as `application/octet-stream`, which is how several government hosts ship
    # theirs. A body that is not a PDF comes back `unsupported_type` exactly as before —
    # so the cost of sniffing is one capped read, and the benefit is that a cited PDF is
    # no longer dropped unread on the strength of a header we cannot rely on.
    return await _resolution_pdf_outcome(resp, current_url, content_type, ctx, from_status="unsupported_type")


async def _fetch_one_hop(
    session: Any, current_url: str, host_sems: dict[str, asyncio.Semaphore], ctx: FetchContext
) -> FetchResult | str:
    """ONE GET against ``current_url`` under its host semaphore: terminal result or next URL.

    The request's timeout is the REMAINING wall budget rather than the session's flat
    ``RESOLUTION_SOURCE_HTTP_TIMEOUT``, and it is computed AFTER the semaphore is acquired so
    a hop that queued behind a slow host does not then help itself to a fresh 20 s. This is
    the one choke point every hop passes through — the initial GET, each 3xx hop and the
    meta-refresh hop — so clamping here is what makes the budget arithmetic the rest of the
    ladder does actually bind: a hop admitted with 3 s left (the meta-refresh rung's floor)
    could otherwise run the full 20 s, overshoot ``RESOLUTION_SOURCE_WALL_TIMEOUT`` and let
    the provider's outer ``wait_for`` discard every sibling page that had already fetched.
    Monotonically <= the old 20 s, and an expiry lands on the existing ``TimeoutError`` path,
    so overrunning costs this one URL rather than the question.

    BOTH ``ClientTimeout`` fields are set because a per-request timeout REPLACES the
    session's wholesale rather than merging with it.

    A cited PDF is the one branch whose work does NOT finish inside the two contexts: it
    comes back as a :class:`_PendingDocument` and is parsed after both have exited, because
    that parse is seconds of CPU and the host gate is loop-wide (see
    :class:`_PendingDocument`). The HTML branch's ``to_thread`` hops still run inside the
    semaphore — trafilatura on a capped page is short next to the request it follows, and
    moving it would trade a measured hazard for an unmeasured restructure.
    """
    async with _sem_for_host(host_sems, current_url):
        hop_timeout_s = min(RESOLUTION_SOURCE_HTTP_TIMEOUT, max(ctx.rung_budget_s(), _MIN_HOP_TIMEOUT_S))
        try:
            async with session.get(
                current_url,
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=hop_timeout_s, sock_read=hop_timeout_s),
            ) as resp:
                outcome = await _resolution_response_outcome(resp, current_url, ctx)
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.info(f"resolution_source fetch error for {current_url}: {type(e).__name__}: {e}")
            return FetchResult(
                url=current_url,
                status="error",
                text="",
                http_status=None,
                content_type=None,
            )
    if isinstance(outcome, _PendingDocument):
        return await _finish_document(outcome, ctx)
    return outcome


# This fetcher's key into the transport's render memos. Its "rendered to nothing" is the strong
# form — the ARIA rewrite, the inline-chart read and the harvested feed all failed — so gap-fill
# v2's weaker verdict on the same URL must never answer for it (rendered_fetch.MemoScope).
_RENDER_MEMO_SCOPE: MemoScope = "resolution_source"


def _rendered_rung_applies(direct: FetchResult) -> bool:
    """Whether a browser could plausibly turn ``direct`` into readable content.

    Two triggers, both pages that answered 200 with nothing we could read: ``js_wall`` (the
    population the rung was measured on — Chromium rescued 6 of the 8 archived walls that
    still failed from a residential address on 2026-09-03) and the ``thin_page`` shape of
    ``no_resolving_content``, where the extraction cleared the JS-wall floor and still carried
    only chrome, which is the same client-side-assembly failure one floor up.

    ``embed_shell`` is deliberately NOT a trigger, and that is a fact about the browser rather
    than a policy choice: ``page.content()`` returns the MAIN FRAME's HTML, so an Infogram or
    Flourish iframe comes back as an ``<iframe>`` tag whose document Chromium rendered
    somewhere we never read. Rendering that page spends a 100-300 MB launch to re-derive the
    same verdict. ``blocked`` is not a trigger either: the edge refused our address before any
    HTML existed, and Chromium dials from the same address.
    """
    if direct.status == "js_wall":
        return True
    return direct.status == "no_resolving_content" and direct.status_reason == "thin_page"


async def _rendered_rung(
    url: str, direct: FetchResult, host_sems: dict[str, asyncio.Semaphore], ctx: FetchContext
) -> FetchResult | None:
    """Render an unreadable page in headless Chromium and re-classify it, or None.

    Runs from the escalation ladder, outside the aiohttp response context, so no response is
    held open across a 12-35 s render. It does NOT run outside the per-host gate: the transport
    re-acquires the same loop-wide ``Semaphore(1)`` for the URL's host and holds it across the
    launch-cap queue, the launch, the navigation, the settle and the teardown, because Chromium
    dials that host itself. Both acquires are unbounded by design (FUTURE.md item 5), which is
    why the transport recomputes the navigation budget only once both are held.

    Self-bounding on the shared pattern: skipped below ``RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S``
    of remaining wall, and the navigation gets the remaining budget less the settle, capped at
    the transport's own 35 s — as a CEILING. The transport tightens it after the gates to what is
    actually left less the settle and the DOM read, or declines under its own floor before a
    browser is launched, so a goto that runs its budget out can still be settled and read inside
    the bound below. Degrading to the direct result costs one page; overrunning the provider's
    outer ``wait_for`` costs every page the question already fetched.

    The whole transport call — queue, launch, navigation, DOM read and teardown — is ALSO held
    to the remaining budget with ``asyncio.wait_for``, on top of the transport's own DOM-read
    and teardown caps, so nothing inside the rung can outlive the wall: a cut during teardown
    cancels it and the transport's driver stop kills the browser. The receipt is ogimet.com
    (2026-09-03): the goto timed out at 33 s as designed and ``page.content()`` then blocked for
    40 s more, for a 76 s render against a 45 s wall. Either bound firing is recorded as its own
    skip, ``render_timeout``, rather than ``renderer_unavailable`` — a cut-off render says
    nothing about whether Chromium works, and must not latch that warning. The direct result is
    what stands. The transport memoises a timed-out URL itself, and only when a browser actually
    ran (the outer cut can fire while the render is still queued, which says nothing about the
    page); a memoised URL re-raises on the next question, so it is recorded the same way again.

    The rendered DOM re-enters :func:`_classify_html_body`, so a rescued page gets the same
    chart read, ARIA rewrite, floors and disclosure leads as a directly-fetched one — unless the
    browser was answered with something other than a 200. The direct GET got a 200 for this URL
    (that is the trigger), so a non-200 main-frame status is the edge telling the browser apart,
    and its markup (a 403 or 429 interstitial routinely clears the chrome floor) is not the page:
    the rung leaves the direct result standing and does not memoise, because a 429 is retryable.
    When the DOM STILL carries nothing, the JSON the page fetched for itself is the last free
    route (:func:`_derived_api_from_harvest`) — a JavaScript dashboard's numbers arrive over XHR
    and are in its HTML at no wait condition. Only once that fails too is the URL memoized
    (:func:`note_rendered_no_text`), so a second URL on the same page in this run does not spend
    another launch to learn the same thing; that memo hit is its own skip, ``rendered_no_text``,
    so it never inflates the count the operator reads as the Chromium install having failed.
    """
    if not _rendered_rung_applies(direct):
        return None
    if rendered_to_nothing(url, memo_scope=_RENDER_MEMO_SCOPE):
        ctx.skip_rung("rendered", direct.status, url, "rendered_no_text")
        return None
    budget_s = ctx.claim_rung_budget("rendered", direct.status, url, RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S)
    if budget_s is None:
        return None
    attempt = ctx.start_rung("rendered", direct.status, url)
    goto_timeout_ms = int(min(RENDER_TIMEOUT_MS, budget_s * 1000) - RENDER_SETTLE_MS)
    try:
        page = await asyncio.wait_for(
            render_page(
                url,
                memo_scope=_RENDER_MEMO_SCOPE,
                host_gate=_sem_for_host(host_sems, url),
                goto_timeout_ms=goto_timeout_ms,
                deadline_monotonic_s=time.monotonic() + budget_s,
                # Recording the page's own XHR costs one buffered body per response inside the
                # render task, which is why the transport keeps it off by default — here it is
                # exactly the rung's fallback, so it is worth the bytes.
                harvest_json=True,
            ),
            timeout=budget_s,
        )
    except RenderBudgetExpired:
        # The budget ran out in the queue behind the two gates: nothing rendered, so it is the same
        # skip the pre-gate check records, not a cut-off render and not a missing browser.
        attempt.skipped_reason = "wall_budget"
        return None
    except TimeoutError as exc:
        # One handler for both bounds: the transport's DOM-read cap raises this itself (and
        # re-raises it for a URL it already cut off this run), and the wait_for above raises it
        # when the whole render outlived the budget. The message says which.
        logger.warning(
            "resolution_source: the rendered rung for %s was cut off (%.1fs budget): %s; leaving the direct result",
            urlparse(url).netloc,
            budget_s,
            exc,
        )
        attempt.skipped_reason = "render_timeout"
        return None
    if page is None:
        # The transport declines with ONE signal for several causes — Playwright missing or
        # broken, a host that will not pin to a public IP, a DOM over its ceiling, or a browser
        # error — and its own WARN/DEBUG lines say which. Recorded as a SKIP rather than a fired rung because nothing
        # was rendered: it then claims no `route=` and emits no escalation line, while keeping
        # the measured wall_s that says what the declined launch cost.
        attempt.skipped_reason = "renderer_unavailable"
        return None
    if page.http_status is not None and page.http_status != 200:
        logger.warning(
            "resolution_source: the browser was answered %d for %s where the direct GET got 200; "
            "not reading that page as content",
            page.http_status,
            urlparse(url).netloc,
        )
        return None
    classified = await _classify_html_body(
        page.html.encode("utf-8", errors="replace"),
        url,
        page.content_type or "text/html",
        # The direct fetch's status, not the browser's: this page answered 200 and carried no
        # text, which is the fact the record should keep. Chromium reports no status at all
        # when a goto timed out and the DOM was salvaged, and a non-200 never reaches here.
        http_status=direct.http_status if direct.http_status is not None else 200,
    )
    # The render's own verdict, stamped before the harvest gets its turn: when the harvested
    # feed rescues the page, the ladder's result is `success` and the closer would otherwise
    # credit the render with a rescue the DOM never delivered.
    attempt.outcome = classified.result.status
    if classified.result.status == "success":
        return classified.result
    derived = _derived_api_from_harvest(url, direct, page, ctx)
    if derived is not None:
        return derived
    note_rendered_no_text(url, memo_scope=_RENDER_MEMO_SCOPE)
    return None


def _derived_api_from_harvest(
    url: str, direct: FetchResult, page: RenderedPage, ctx: FetchContext
) -> FetchResult | None:
    """Serve the JSON the rendered page fetched for itself, when the DOM carried nothing.

    Its own rung attempt rather than part of the render's, because ``route`` is the LAST rung
    that fired and ``derived_api`` is what actually produced the text — the render only found
    the endpoint. The endpoint is also remembered for the host, so a later cited URL on it can
    GET the feed without a second launch (:func:`_derived_api_rung`).

    Declines silently when nothing was harvested or the biggest body carries no usable content
    (:func:`vacuous_body_status`): a body we could not decode must never become the page's
    content on a section captioned primary grading evidence.
    """
    harvested = derived_api.largest_json(page.json_responses)
    if harvested is None:
        return None
    raw, undecodable_ratio = decode_text_body(harvested.body, "application/json")
    if vacuous_body_status(raw, undecodable_ratio, require_csv_rows=False) is not None:
        return None
    derived_api.remember_endpoint(url, harvested.url)
    endpoint = derived_api.DerivedEndpoint(endpoint_url=harvested.url, discovered_on=url)
    ctx.start_rung("derived_api", direct.status, url)
    return _derived_api_result(url, endpoint, raw, http_status=direct.http_status)


def _derived_api_result(
    url: str, endpoint: derived_api.DerivedEndpoint, raw: str, *, http_status: int | None
) -> FetchResult:
    """One derived-feed result: the provenance lead, then the budgeted JSON.

    The lead LEADS and its cost comes out of the per-URL cap (:func:`_lead_then_capped_body`),
    because a feed served with its provenance line trimmed off is a JSON blob nobody can check.
    """
    lead = derived_api.derived_api_lead(endpoint, url)
    return FetchResult(
        url=url,
        status="success",
        text=_lead_then_capped_body(lead, raw, url),
        http_status=http_status,
        content_type="application/json",
    )


async def _derived_api_rung(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: FetchContext
) -> FetchResult | None:
    """GET a JSON feed an earlier render on this host already found, before launching a browser.

    This is the whole point of remembering the endpoint: a host with several cited URLs in one
    run pays for one Chromium launch, not one per URL. It runs BEFORE the rendered rung for the
    same reason every ladder here is ordered cheapest-first — one GET against a known endpoint
    is a rounding error next to a browser launch. Within a question that holds even when the
    URLs are fetched concurrently, because the dispatcher runs this rung and the browser rung
    under one per-host gate (:meth:`QuestionRungBudget.browser_escalation_gate`), so a same-host
    sibling asks for the endpoint only after the first render has had its chance to record it.

    The GET goes through :func:`_fetch_direct`, so it inherits the SSRF preflight, the
    connect-time filtering resolver, the redirect re-guard, the per-host gate and the
    budget-clamped hop timeout unchanged. A feed that fails hands the URL on to the browser.
    """
    if not _rendered_rung_applies(direct):
        return None
    endpoint = derived_api.endpoint_for(url)
    if endpoint is None:
        return None
    if ctx.claim_rung_budget("derived_api", direct.status, url, RESOLUTION_SOURCE_DERIVED_API_MIN_BUDGET_S) is None:
        return None
    ctx.start_rung("derived_api", direct.status, url)
    logger.info(
        f"resolution_source derived_api: {urlparse(url).netloc} -> {endpoint.endpoint_url} "
        f"(found on {endpoint.discovered_on}, direct read was {direct.status})"
    )
    feed = await _fetch_direct(session, endpoint.endpoint_url, host_sems, _aux_ctx(ctx))
    if feed.status != "success":
        return None
    if not _is_json_content_type(feed.content_type or ""):
        # The same gate the harvest half applies at discovery, because a remembered endpoint is
        # not a promise about what it answers NEXT time: one came back 200 with an HTML "session
        # expired" portal page, which the lead below would have introduced as the JSON feed the
        # page loads its figures from. Declining hands the URL to the browser, whose own harvest
        # is gated the same way.
        logger.info(
            "resolution_source derived_api: %s answered %r rather than JSON — not served as the feed",
            endpoint.endpoint_url,
            feed.content_type,
        )
        return None
    return _derived_api_result(url, endpoint, feed.text, http_status=feed.http_status)


# A page the archive can plausibly substitute for: the host refused us, never answered, or says
# the URL is gone. Deliberately NOT `js_wall` — the archive stores the unrendered shell, so it
# rescued 0 of the 8 archived walls that still failed on 2026-09-03 while the browser rung
# rescued 6. Nor `no_resolving_content`: a page that answered 200 with chrome is one whose live
# markup we have and whose numbers are elsewhere, and an older copy of the same chrome adds
# nothing. `ssrf_blocked` is excluded because WE refused that URL, and handing it to a
# third-party fetcher is precisely the bypass the guard exists to prevent.
_WAYBACK_TRIGGER_STATUSES: frozenset[FetchStatus] = frozenset({"blocked", "error", "not_found"})


async def _wayback_snapshot_result(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: FetchContext
) -> FetchResult | None:
    """Fetch the archive's freshest capture of ``url`` and serve it, or withhold it.

    The fetch goes through :func:`_fetch_direct`, so the snapshot is classified by exactly the
    path a live page is — including the chart read and the chrome floor — and inherits the SSRF
    preflight, the per-hop re-guard and the budget-clamped hop timeout. What comes back extra is
    the FINAL URL, which is where the archive puts the 14-digit capture timestamp.

    Three outcomes, in this order, and the order is the design.

    The inner URL is UNWRAPPED — repeatedly, since a capture OF a capture presents
    ``web.archive.org`` as its own inner host — and re-checked first, because a hostname check
    on ``web.archive.org/web/…/metaculus.com/…`` sails past every self-reference filter in the
    pipeline — an archived Metaculus page in front of a forecaster is the question quoting
    itself. Then a snapshot the archive could not serve at all (no capture, or a capture that
    404s) DECLINES: there is no archived copy, which is a different fact from a stale one, and
    the direct route's own status says more about the source than a fact about the archive would.
    Only a capture we actually READ and cannot date, or can date and it is too old, is withheld
    as ``stale_data`` — because the disclosure that makes a snapshot admissible is its age, and a
    copy with no usable date cannot carry it. The direct status is not lost by that swap either:
    the ``RESOLUTION_SOURCE_ESCALATION`` line for this rung carries ``from_status``.
    """
    snapshot = await _fetch_direct(session, wayback_snapshot_url(url, now=ctx.now), host_sems, _aux_ctx(ctx))
    parsed = parse_snapshot_url(snapshot.url)
    captured_of = None if parsed is None else innermost_url(parsed.inner_url)
    if captured_of is not None and (is_metaculus_self_ref(captured_of) or not await is_public_http_url(captured_of)):
        logger.warning(
            "resolution_source wayback refused: snapshot of %s wraps a URL we do not fetch (%s)",
            urlparse(url).netloc,
            urlparse(captured_of).netloc,
        )
        return None
    if snapshot.status != "success":
        # Two different facts, and the archive's own redirect is what tells them apart: a
        # request it never redirected onto a dated capture URL means it holds no capture, while
        # a capture URL we did land on and could not use means it holds one we cannot read.
        # Both used to log "no archived copy served", so apnews.com — a capture served in full
        # whose extraction was 355 chars of AP boilerplate — read as an empty archive.
        logger.info(
            "resolution_source wayback: %s for %s (%s)",
            "no archived copy served" if parsed is None else "an archived capture was served but is unusable",
            urlparse(url).netloc,
            snapshot.status,
        )
        return None
    age_days = None if parsed is None else snapshot_age_days(parsed, ctx.now)
    if parsed is None or age_days is None or age_days > RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS:
        logger.warning(
            "resolution_source wayback: capture for %s is not usable (final=%s, age=%s) — withheld as stale",
            urlparse(url).netloc,
            snapshot.url,
            "undatable" if age_days is None else f"{age_days:.1f}d",
        )
        return FetchResult(
            url=url,
            status="stale_data",
            text="",
            http_status=snapshot.http_status,
            content_type=snapshot.content_type,
        )
    # The lead LEADS and its cost comes out of the per-URL cap (:func:`_lead_then_capped_body`):
    # an archived page whose age line has been trimmed off is being passed off as the live one.
    lead = wayback_lead(parsed, age_days, direct.status)
    return FetchResult(
        url=url,
        status="success",
        text=_lead_then_capped_body(lead, snapshot.text, url),
        http_status=snapshot.http_status,
        content_type=snapshot.content_type,
        datawrapper_charts=snapshot.datawrapper_charts,
        unreadable_embeds=snapshot.unreadable_embeds,
        precision_rescued=snapshot.precision_rescued,
    )


async def _wayback_rung(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: FetchContext
) -> FetchResult | None:
    """Try the Wayback Machine for a page our own address could not reach.

    Bounded three ways, because this rung's cost is concentrated rather than spread: below
    ``RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S`` of remaining wall it is skipped, at most
    ``RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS`` snapshots are fetched per question, and every
    snapshot contends on the one ``web.archive.org`` host gate — which is the documented trade
    for the politeness that gate exists to provide.
    """
    if direct.status not in _WAYBACK_TRIGGER_STATUSES:
        return None
    if ctx.claim_rung_budget("wayback", direct.status, url, RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S) is None:
        return None
    if not ctx.shared.take_wayback_attempt():
        logger.warning(
            "resolution_source: skipping the wayback rung for %s — this question's %d snapshot attempt(s) are spent",
            urlparse(url).netloc,
            RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS,
        )
        ctx.skip_rung("wayback", direct.status, url, "wayback_cap")
        return None
    ctx.start_rung("wayback", direct.status, url)
    return await _wayback_snapshot_result(session, url, direct, host_sems=host_sems, ctx=ctx)


# What the paid reader is allowed to be asked about. Tested against the DIRECT outcome (an
# archive withhold on the way down does not change it — see `_escalate_unresolved`): everything
# the free ladder left unresolved EXCEPT the outcomes where a model-mediated read cannot help or
# must not be tried. A 404/410 has no page to read, an empty or undecodable body and an unreadable
# document are bytes we DID get (only `no_text_layer` could ever be rescued, and that is v2's
# `read_document` job on a URL the driver chose), and `ssrf_blocked` is a URL WE refused — handing
# that to a third-party fetcher is exactly the bypass the guard exists to prevent, which is why it
# is excluded here and not merely unlisted.
# One member of the set is narrowed further by REASON rather than by status — see
# :func:`_url_context_rung_applies` — so this set is the ceiling on the population, not the
# population itself.
_URL_CONTEXT_TRIGGER_STATUSES: frozenset[FetchStatus] = frozenset(
    {"blocked", "js_wall", "error", "no_resolving_content"}
)


def _url_context_rung_applies(direct: FetchResult) -> bool:
    """Whether a model-mediated read could plausibly resolve ``direct``.

    The trigger statuses above, minus the one outcome inside them a paid read cannot help
    with: a document we read END TO END whose passage selection matched no query term
    (``no_matching_passage``). Its bytes were never the problem — we hold its full text and
    its outline — so paying Gemini to re-read the same PDF buys nothing. Scoped on the REASON
    rather than by dropping the status, because ``embed_shell`` and ``thin_page`` are pages our
    client genuinely could not read and are exactly what this rung exists for.
    """
    if direct.status not in _URL_CONTEXT_TRIGGER_STATUSES:
        return False
    return direct.status_reason != "no_matching_passage"


def _url_context_lead(live_status: FetchStatus) -> str:
    """The MANDATORY disclosure a model-mediated read carries.

    Both clauses are the point. It says WHY this route was taken, so a forecaster knows the host
    refused us rather than that we chose a model over a fetch. And it says the text is not a copy
    of the page — every other section in this snapshot is bytes the host served, and reading a
    paraphrase under the same "primary grading evidence" caption without that line would overstate
    what was retrieved by exactly the amount that matters.
    """
    return (
        f"[Read via Gemini url_context because the live page could not be fetched ({live_status}); "
        f"model-mediated, not a byte-for-byte copy.]"
    )


async def _fetch_robots_txt(
    session: Any, robots_url: str, host_sems: dict[str, asyncio.Semaphore], ctx: FetchContext
) -> str | None:
    """Read one robots.txt through THIS path's own fetch; None when we could not read it.

    Goes through :func:`_fetch_direct` rather than a second client, so the SSRF preflight, the
    connect-time filtering resolver, the per-hop redirect re-guard, the per-host gate and the
    budget-clamped hop timeout all apply to a request this pre-check makes. That path also
    CLASSIFIES, so a host serving robots.txt as HTML can come back withheld under the chrome
    floor — which reads as "no directives", i.e. proceed and pay, the only direction an
    unreadable robots.txt is allowed to fail in.

    Bounded at ``ROBOTS_FETCH_TIMEOUT_S`` on top of the hop's own clamp, the same bound gap-fill
    v2 gives the identical read: the hop clamp is the remaining WALL (up to 20 s) and the
    per-host gate in front of it is an unbounded acquire, and neither is a sensible price for a
    pre-check whose only job is to avoid one paid call. A timeout reads as unreadable.
    """
    try:
        result = await asyncio.wait_for(
            _fetch_direct(session, robots_url, host_sems, _aux_ctx(ctx)), ROBOTS_FETCH_TIMEOUT_S
        )
    except TimeoutError:
        logger.info(
            "resolution_source: robots.txt pre-check for %s did not answer in %.1fs", robots_url, ROBOTS_FETCH_TIMEOUT_S
        )
        return None
    return result.text if result.status == "success" else None


async def _url_context_robots_skip(
    session: Any, url: str, host_sems: dict[str, asyncio.Semaphore], ctx: FetchContext
) -> bool:
    """True when ``url``'s host tells ``Google-Extended`` to stay out of that path.

    Only the PAID rung consults this: the free rungs dial from our own client under our own user
    agent, and this bot's reading of ``Content-Signal: use=reference`` is that reference use is
    permitted. The per-host cache lives in ``robots_policy`` and is shared with gap-fill v2's
    reader, so a host reached by both paths in one run is read once.
    """
    return await google_extended_blocks_url(
        url, fetch_text=lambda robots_url: _fetch_robots_txt(session, robots_url, host_sems, ctx)
    )


async def _url_context_admission(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: FetchContext
) -> tuple[str, float] | None:
    """Every gate the paid read has to clear, in increasing cost order; ``(api_key, budget_s)`` or None.

    The trigger population, the flag (default off, and off in every workflow), the question's
    time-budget fast path, the API key, the wall budget, then the per-host ``Google-Extended``
    robots pre-check — the one gate that costs a request — and the wall budget AGAIN. The robots
    check is worth a request of its own because a host that disallows that token refuses the
    fetch server-side — proven live 2026-09-03, where the same call that retrieved a
    robots-allowed host came back ``URL_RETRIEVAL_STATUS_ERROR`` on
    internationalaisafetyreport.org — so the read would be spend with a known-zero return.

    The budget is checked twice because the pre-check sits between reading it and spending it,
    and it can eat real time: an unbounded per-host gate acquire and then up to
    ``ROBOTS_FETCH_TIMEOUT_S``. The read runs in a thread, which ``wait_for`` cannot cancel, so
    the client-side ceiling is the only thing that returns the worker — and a ceiling sized off
    the figure read BEFORE the pre-check could outlive the provider's wall while the money is
    spent on a result nothing reads. The second check costs nothing (the read has not started),
    and the budget returned here is the one the ceiling and the ``wait_for`` are sized off.
    """
    if not _url_context_rung_applies(direct):
        return None
    if not env_flag_enabled(RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV):
        return None
    if ctx.fast_path:
        # After the flag and before the key: recorded only for a rung that was ARMED, so a
        # flag-off run never reports spend avoided on a rung that could not have fired.
        _skip_for_fast_path(ctx, "url_context", direct, url)
        return None
    api_key = os.getenv(GOOGLE_API_KEY_ENV)
    if not api_key:
        logger.info(
            "resolution_source: url_context rung is enabled but %s is not set — skipping %s",
            GOOGLE_API_KEY_ENV,
            urlparse(url).netloc,
        )
        ctx.skip_rung("url_context", direct.status, url, "no_api_key")
        return None
    if ctx.claim_rung_budget("url_context", direct.status, url, RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S) is None:
        return None
    if await _url_context_robots_skip(session, url, host_sems, ctx):
        logger.info(f"RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP: url={url} host={urlparse(url).netloc}")
        ctx.skip_rung("url_context", direct.status, url, "robots_disallowed")
        return None
    budget_s = ctx.claim_rung_budget(
        "url_context",
        direct.status,
        url,
        RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S,
        note=" after the robots pre-check",
    )
    if budget_s is None:
        return None
    return api_key, budget_s


async def _url_context_rung(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: FetchContext
) -> FetchResult | None:
    """Ask Gemini to read a page our own client could not, or decline.

    The LAST rung and the only paid one, so every gate is checked before a cent is spent
    (:func:`_url_context_admission`, which also explains why the wall budget is read twice and
    why the figure it hands back is the one that sizes the read).

    Zero successful retrievals DISCARDS the text and reports ``ungrounded``. Gemini answers
    fluently out of parametric memory when every retrieval failed, and this section is captioned
    primary grading evidence, so a fluent unsourced answer here is the Q38195 failure with a
    forecaster-facing blast radius. That is the same floor ``gemini_search`` and v2's
    ``read_document`` apply, for the same reason.
    """
    admitted = await _url_context_admission(session, url, direct, host_sems=host_sems, ctx=ctx)
    if admitted is None:
        return None
    api_key, budget_s = admitted
    ctx.start_rung("url_context", direct.status, url)
    try:
        text, n_retrievals, statuses = await asyncio.wait_for(
            asyncio.to_thread(
                run_url_context_read,
                url,
                ctx.query,
                api_key=api_key,
                role="resolution_source",
                model=GAP_FILL_V2_READER_MODEL,
                thinking_level=GAP_FILL_V2_READER_THINKING_LEVEL,
                # The client-side ceiling is what returns the worker: wait_for cancels this
                # coroutine and not the thread it is waiting on. Sized off the remaining budget
                # so the read cannot outlive the provider's own wall by more than the margin.
                timeout_ms=int(max(0.0, budget_s - RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S) * 1000),
                attempts=RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS,
            ),
            timeout=budget_s,
        )
    except TimeoutError:
        logger.warning("resolution_source url_context read timed out for %s", urlparse(url).netloc)
        return None
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except  # paid-rung soft-fail boundary: a dead reader leaves the direct result, never takes the provider down
        logger.warning(
            "resolution_source url_context read failed for %s: %s: %s",
            urlparse(url).netloc,
            type(exc).__name__,
            exc,
        )
        return None
    if n_retrievals == 0 or not text.strip():
        # Spelled parallel to the gap-fill v2 reader's AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED (and
        # gemini_search's GEMINI_UNGROUNDED_SUPPRESSED), so the three suppression rates read as
        # one family. `statuses` carries every reported url_retrieval_status; `none` means the
        # SDK attached no url_metadata entry at all. Deliberately not a registered marker spec
        # while the flag is off in every workflow — see docs/research.md.
        logger.warning(
            f"RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url={url} statuses={','.join(statuses) or 'none'}"
        )
        return FetchResult(
            url=url,
            status="ungrounded",
            text="",
            http_status=direct.http_status,
            content_type=direct.content_type,
        )
    # The lead LEADS and is budgeted out of the cap (:func:`_lead_then_capped_body`): a model's
    # answer rendered without the disclosure reads as the page itself.
    lead = _url_context_lead(direct.status)
    return FetchResult(
        url=url,
        status="success",
        text=_lead_then_capped_body(lead, text.strip(), url),
        http_status=direct.http_status,
        content_type="text/plain",
    )


async def _escalate_unresolved(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: FetchContext
) -> FetchResult:
    """Run the escalation rungs a direct fetch's outcome earns, cheapest first.

    Returns the FIRST rung's rescue, or ``direct`` unchanged when every rung declines or fails.
    A rung that fired and produced nothing still leaves its attempt on the context, which is
    what makes ``route=rendered status=js_wall`` readable in the archive as "we tried the
    browser and this is still the answer" — the same convention the meta-refresh hop already
    follows. The Wayback rung is the one rung whose non-rescue is a VERDICT rather than None
    (``stale_data``, a capture we read and will not serve); it is kept as the fallback rather
    than as an early return, so the paid rung below is still reachable for that page.

    Each rung is closed the moment its result is known (:meth:`FetchContext.close_rungs`), so
    the attempts it opened carry that rung's own wall and outcome rather than the ladder's:
    the status it returned, or the direct status it left standing when it declined.

    ``session`` is the aiohttp session the rungs that issue an ordinary GET use; the browser
    rung ignores it, because Chromium brings its own transport.
    """
    if direct.status == "success":
        return direct
    if _rendered_rung_applies(direct):
        # The two browser-family rungs run under one per-host gate for this question, so a
        # same-host sibling asks `endpoint_for` only after this escalation has recorded (or
        # failed to record) an endpoint — see `QuestionRungBudget.browser_escalation_gate`.
        async with ctx.shared.browser_escalation_gate(url):
            first_new = len(ctx.rungs)
            derived = await _derived_api_rung(session, url, direct, host_sems=host_sems, ctx=ctx)
            ctx.close_rungs(first_new, (derived or direct).status)
            if derived is not None:
                return derived
            # Declined HERE rather than inside the rung: the rung's own gates all cost something
            # (a budget read, a memo lookup, a launch), and the fast path is a fact about the
            # question the dispatcher already holds.
            if ctx.fast_path:
                _skip_for_fast_path(ctx, "rendered", direct, url)
            else:
                first_new = len(ctx.rungs)
                rendered = await _rendered_rung(url, direct, host_sems, ctx)
                ctx.close_rungs(first_new, (rendered or direct).status)
                if rendered is not None:
                    return rendered
    # Reached only for the statuses the browser rungs do not claim — the two trigger sets are
    # disjoint by construction (see `_WAYBACK_TRIGGER_STATUSES`), so the order between them is a
    # reading choice: free-and-local first, then the route whose egress is not ours.
    first_new = len(ctx.rungs)
    wayback = await _wayback_rung(session, url, direct, host_sems=host_sems, ctx=ctx)
    ctx.close_rungs(first_new, (wayback or direct).status)
    if wayback is not None and wayback.status == "success":
        return wayback
    # Last, because it is the only rung that spends money and the only one whose product is a
    # model's answer rather than the host's bytes. Off by default and off in every workflow.
    # It is asked about the DIRECT outcome, and an archive WITHHOLD does not stand in its way: a
    # capture too old to serve is still a page we could not read fresh, which is exactly the
    # population this rung exists for. The withhold stays the fallback below, so with the flag
    # off (or the reader declining) a stale capture still reports `stale_data`.
    first_new = len(ctx.rungs)
    read = await _url_context_rung(session, url, direct, host_sems=host_sems, ctx=ctx)
    ctx.close_rungs(first_new, (read or wayback or direct).status)
    if read is not None:
        return read
    return wayback if wayback is not None else direct


async def _fetch_one(
    session: Any, url: str, host_sems: dict[str, asyncio.Semaphore], ctx: FetchContext | None = None
) -> FetchResult:
    """Fetch a single URL directly, then escalate what the direct route could not read.

    ``ctx`` carries the question text a PDF digest ranks passages against, the wall-clock
    origin each rung bounds itself with, and the rung attempts stamped onto the returned
    result. It defaults to a fresh one so the fetch surface can still be driven with three
    arguments, which is what every existing caller and test does.
    """
    ctx = FetchContext() if ctx is None else ctx
    direct = await _fetch_direct(session, url, host_sems, ctx)
    # The rungs inside the direct fetch (the meta-refresh hop, the local PDF read) are over
    # once it returns, and its status is what they left standing.
    ctx.close_rungs(0, direct.status)
    escalated = await _escalate_unresolved(session, url, direct, host_sems=host_sems, ctx=ctx)
    return _stamped_with_route(escalated, ctx)


async def _fetch_direct(
    session: Any, url: str, host_sems: dict[str, asyncio.Semaphore], ctx: FetchContext
) -> FetchResult:
    """Fetch a single URL directly, holding the per-host politeness semaphore hop by hop.

    Content-type routing:
      * HTML → ARIA-table rewrite + trafilatura extraction (via to_thread), the
        inline-chart rung, then the chrome / JS-wall checks and the meta-refresh hop.
      * JSON → capped raw body, no pretty-print (the data IS the content).
      * text/plain, text/csv → capped raw body.
      * anything else, including a missing/empty Content-Type header → capped read,
        then the ``%PDF-`` magic check: a document is read locally and rendered as a
        query-relevant digest, and anything else is ``unsupported_type`` as before.

    Politeness: each hop acquires the semaphore for THAT hop's host around its single GET,
    the body read on a terminal response, and the HTML branch's extraction, and releases it
    before following a redirect. A cited PDF's parse is the one thing deliberately outside
    the hold — it comes back as a ``_PendingDocument`` and is parsed after the semaphore is
    released, because the gate is loop-wide and the parse is seconds of CPU. Keying per hop
    — not on the original URL's host — preserves one-request-per-host when chains from
    different initial hosts converge on the same final host; the strict per-hop
    acquire/release pairing means an A→B→A chain never re-acquires a semaphore it still
    holds (asyncio semaphores are not reentrant).

    SSRF guard: rejects non-public URLs (private / loopback / link-local IPs,
    userinfo tricks, non-http(s) schemes) BEFORE any network I/O and again on
    every hop target, whether it came from a ``Location`` header or a meta-refresh
    tag (:func:`_vetted_hop_target` is the one place both are checked). The
    connect-time :class:`FilteringResolver` (see :func:`_get_session`) provides the
    actual DNS-rebinding boundary; these preflight checks are fast-fail
    observability so we surface ``ssrf_blocked`` without opening a session. Hops of
    both shapes are followed in-band and share the one ``MAX_REDIRECTS`` cap.

    No retries (Tier 1 anti-goal). Any aiohttp/asyncio error becomes ``error``. Escalation
    beyond this route is :func:`_escalate_unresolved`'s job, so this function stays exactly
    what it always was: the plain fetch, terminal on its own outcome.
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
    # status (or a meta-refresh stub) resolves the next URL, re-guards, and loops
    # (each hop releases its semaphore before the next acquires its own — no
    # nesting, so no self-deadlock on revisited hosts).
    # Non-redirect responses fall through to the content-type routing below.
    for _hop in range(MAX_REDIRECTS + 1):
        outcome = await _fetch_one_hop(session, current_url, host_sems, ctx)
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
    return "success" if status == 200 else _NON_OK_FETCH_STATUS.get(status, "error")


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
    if last_modified - now > RESOLUTION_SOURCE_CLOCK_SKEW_TOLERANCE:
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


async def fetch_resolution_sources(urls: list[str], *, query: str = "", fast_path: bool = False) -> list[FetchResult]:
    """Fetch each URL under per-netloc Semaphore(1) politeness, then hop to
    the live datasets of any Datawrapper charts the fetched pages embed.

    ``query`` is the question's title plus resolution criteria. It never touches the
    network; its one job is ranking which passages of a cited PDF a forecaster sees.
    Empty is legitimate (a caller with no question text in hand) and simply means a
    document renders its header and outline with no passages. ``fast_path`` is the
    question's time-budget thin-window mode; it rides every URL's :class:`FetchContext`
    and makes the two expensive rungs decline (see there).

    Distinct hosts run concurrently up to the connector limit; same-host
    requests serialize (politeness — e.g. StatCan asks Crawl-delay: 2). The
    host-semaphore map is now the PROCESS-WIDE one
    (:func:`http_fetch.host_semaphores`, scoped to the running loop) rather than a
    fresh dict per call: with one map per call, six questions fetching the same host
    concurrently each held their own semaphore and hit it six times at once. Every
    ``_fetch_one`` task shares it, so each hop contends on ITS host's semaphore —
    chains from different initial hosts that converge on one final host still
    serialize there; the Tier-2 dataset fetches contend on the dwcdn host's semaphore
    the same way. Session is closed in ``finally``.

    Sharing the map buys that politeness at the cost of CROSS-QUESTION serialization: a
    same-host queue now forms across the concurrent questions, inside a
    ``RESOLUTION_SOURCE_WALL_TIMEOUT`` that was not raised and that discards work which
    already succeeded when it fires, so a question that loses the queue can lose every
    page it had already fetched rather than just the contended one (reproduced; the
    archived tail says 3 of 23 all-fail fetches ran the full per-request timeout). The
    acquire wait itself is deliberately unbounded — see FUTURE.md item 5, where both
    remedies (partial harvest, or a budget-bounded wait) are the operator's call.

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
    host_sems = host_semaphores()
    tasks: list[asyncio.Task[FetchResult]] = []
    started = time.monotonic()

    session_cm = _get_session()
    async with session_cm as session:
        try:
            # One context per URL: the rung attempts belong to that URL's result, while
            # the query and the wall-clock origin are the same for all of them.
            # ONE shared rung budget across this question's URLs, and one per-URL context each:
            # the Wayback cap is per question (every snapshot shares one host gate), while the
            # rung attempts belong to the URL they were spent on.
            shared_budget = QuestionRungBudget()
            page_tasks = [
                asyncio.create_task(
                    _fetch_one(
                        session,
                        u,
                        host_sems,
                        FetchContext(query=query, started=started, shared=shared_budget, fast_path=fast_path),
                    )
                )
                for u in urls
            ]
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


def _log_fetch_outcome_markers(qid: int | None, results: list[FetchResult]) -> None:
    """Emit ONE greppable ``RESOLUTION_SOURCE_FETCH`` line per fetched URL.

    Per-URL outcomes used to live only in free-text log lines and in the published
    comment's provider-diagnostics block, so a cut like "cdc.gov is 0 successes in
    1,069 fetch records" meant re-scraping run logs that expire from GHA at 90
    days. This is the harvested form (spec ``resolution_source_fetch``,
    ``scripts/telemetry/markers.py``); the free-text outcome lines it replaces were
    deleted rather than kept beside it, so no fetch is logged twice.

    Emitted here, at the per-question aggregation point, because that is where the
    question id exists — threading it down through ``fetch_resolution_sources`` /
    ``_fetch_one`` / the response-classification helpers would change the signature
    of the whole monkeypatched fetch surface to carry a value only a log line reads.

    Tier-2 dataset hops ride the same marker and are identified by their url, which
    is always ``static.dwcdn.net/data/<chart_id>.csv`` — that host is reachable no
    other way, so a query can partition cited pages from hop artifacts on it.
    ``status`` is the shared token (``ok`` for a success, else the verbatim
    ``FetchStatus``) and ``embeds`` names the routeless data-embed providers found in
    the page's raw HTML, which is what makes an unreadable-embed page queryable even
    when its prose made it a success.

    ``reason`` is appended only where the status alone is ambiguous —
    ``no_resolving_content``'s ``embed_shell`` vs ``thin_page`` vs the
    ``no_matching_passage`` of a document read in full that discusses nothing the question
    asks about, ``unreadable_document``'s ``no_text_layer`` vs ``encrypted`` vs
    ``malformed``, and the ``budget_skipped`` / ``parse_contention`` that say an
    ``unsupported_type`` was a document we were holding.
    ``route`` is appended only when a ladder rung produced the outcome. Both are
    appended rather than always emitted so every line the archive already holds stays
    byte-identical and an absent field keeps meaning "this does not apply", not "old
    record"; both sit at the tail in the order the marker spec's optional groups do.

    Each rung that FIRED also gets one ``RESOLUTION_SOURCE_ESCALATION`` line. The
    fetch line above carries only the final outcome, so on its own it cannot say
    whether a rung rescued the page or what the attempt cost — and ``wall_s`` is what
    decides whether a rung earns its latency under a close-derived time budget. Both
    ``outcome`` and ``wall_s`` are the RUNG's own (``RungAttempt``): the status that stood
    once that rung was over and what that rung alone cost, so on a page where a dead feed
    GET was followed by a rescuing render the first line reads the direct status and the
    second reads ``success``, and neither is billed for the other's latency. The ``url``
    on an escalation line is the URL the rung was invoked ON, which for a meta-refresh hop
    is the stub rather than the target the fetch line names.
    """
    for r in results:
        reason = f" reason={r.status_reason}" if r.status_reason else ""
        route = f" route={r.route}" if r.route != "direct" else ""
        logger.info(
            f"RESOLUTION_SOURCE_FETCH: question={qid} url={r.url} status={fetch_outcome_token(r)} "
            f"http={r.http_status if r.http_status is not None else 'n/a'} "
            f"embeds={','.join(r.unreadable_embeds) if r.unreadable_embeds else 'none'}"
            f"{reason}{route}"
        )
        for attempt in r.rung_attempts:
            if attempt.skipped_reason:
                continue
            logger.info(
                f"RESOLUTION_SOURCE_ESCALATION: question={qid} url={attempt.url} "
                f"from_status={attempt.from_status} rung={attempt.rung} outcome={attempt.outcome} "
                f"wall_s={attempt.wall_s if attempt.wall_s is not None else 0.0:.2f}"
            )


def _rung_counts(results: list[FetchResult]) -> dict[str, int]:
    """Per-rung attempt counts for ``details["counts"]``.

    Zeroes are kept: they render nothing in the diagnostics line but survive into the
    archive, which is what makes "the rung existed and never fired" distinguishable
    from "this record predates the rung".
    """
    attempts = [attempt for r in results for attempt in r.rung_attempts]
    fired_by_rung = Counter(attempt.rung for attempt in attempts if not attempt.skipped_reason)
    skips_by_reason = Counter(attempt.skipped_reason for attempt in attempts if attempt.skipped_reason)
    budget_skips_by_rung = Counter(attempt.rung for attempt in attempts if attempt.skipped_reason == "wall_budget")
    return {
        "meta_refresh_hops": fired_by_rung["meta_refresh"],
        "pdf_documents_read": fired_by_rung["pdf_local"],
        "rendered_attempts": fired_by_rung["rendered"],
        "derived_api_reads": fired_by_rung["derived_api"],
        "wayback_attempts": fired_by_rung["wayback"],
        "url_context_reads": fired_by_rung["url_context"],
        "rung_budget_skips": skips_by_reason["wall_budget"],
        # The same skips broken out per rung, because the aggregate cannot say WHICH rung the
        # provider's wall is binding on — and at the paid flag's rollout "how often does the
        # paid rung get starved by the pages before it" is the question. The total stays as it
        # is, since the archive already reads it.
        **{f"{rung}_budget_skips": budget_skips_by_rung[rung] for rung in _BUDGET_GATED_RUNGS},
        # Its own count rather than folded into the budget skips: a document left unread
        # because two others were already parsing says the 2-slot gate is the binding
        # constraint, which is a different thing to fix than a question that ran late.
        "pdf_contention_skips": skips_by_reason["parse_contention"],
        # Also its own count, for the same reason: a browser rung that never rendered because
        # Chromium is missing on the runner (the install step is `continue-on-error` in every
        # workflow, so its absence is by design) says something different from a question that
        # ran out of wall, and both are invisible in `rendered_attempts`.
        "renderer_unavailable_skips": skips_by_reason["renderer_unavailable"],
        # Its own count too: a render that launched and was cut off — by the transport's DOM-read
        # cap or by the question's remaining wall budget — is a page that keeps navigating, which
        # is a fact about the page, whereas the two counts above are facts about the runner and
        # about the question's clock. Folding it into either would hide the population the
        # ogimet receipt (2026-09-03) is the first member of.
        "render_timeout_skips": skips_by_reason["render_timeout"],
        # And its own count: a browser rung skipped because an earlier question in this run
        # already rendered the same URL to nothing is the memo doing its job, not a runner without
        # Chromium — folded into `renderer_unavailable_skips` it inflated the install-failed signal.
        "rendered_no_text_skips": skips_by_reason["rendered_no_text"],
        # Also its own count: a question that spent its two snapshot attempts on earlier
        # cited URLs is a question whose per-question cap is binding, which is a different
        # thing to tune than a question that ran out of wall.
        "wayback_cap_skips": skips_by_reason["wayback_cap"],
        # Its own count: an expensive rung declined because the QUESTION's close-derived budget
        # put it on the fast path, which is a fact about the question's window rather than about
        # the provider's own 45 s wall (`rung_budget_skips`) — the two are tuned differently.
        "fast_path_skips": skips_by_reason["fast_path"],
        # Its own count because it is the free pre-check EARNING its request: a host that
        # disallows Google-Extended refuses the read server-side, so this is spend avoided
        # rather than a page lost, and it must not read as a failure.
        "url_context_robots_skips": skips_by_reason["robots_disallowed"],
        # Its own count because it is a MISCONFIGURATION rather than a tuning signal: with the
        # flag on and GOOGLE_API_KEY unset the paid rung fires nowhere, and without this key
        # that run is byte-identical in the archive to one with the flag off.
        "url_context_no_api_key_skips": skips_by_reason["no_api_key"],
        # The extractor policy's two decisions, per final result: a page withheld because its
        # extraction cleared the chrome floor on navigation alone, and a page published from
        # the precision re-extraction after the default one failed the same metric.
        "chrome_metric_withholds": sum(1 for r in results if r.chrome_metric_withheld),
        "precision_fallback_rescues": sum(1 for r in results if r.precision_rescued),
    }


# Every rung with a wall-budget floor, i.e. every rung that can record a `wall_budget` skip, in
# ladder order. `_rung_counts` breaks the aggregate `rung_budget_skips` out per member.
_BUDGET_GATED_RUNGS: tuple[FetchRoute, ...] = (
    "meta_refresh",
    "pdf_local",
    "derived_api",
    "rendered",
    "wayback",
    "url_context",
)


def _document_query(question: MetaculusQuestion) -> str:
    """The text a cited document's passages are ranked against.

    Title plus resolution criteria, because those are the two fields that say what the
    question is graded on — and the ranking is BM25 over the document, so the criteria's
    own vocabulary ("laboratory-confirmed cases", "final revised estimate") is exactly
    what should pull the right paragraph out of a 220-page report. Fine print is left
    out: it is mostly procedural boilerplate about ambiguity and annulment, which would
    dilute the term set with words no relevant passage contains.
    """
    return f"{question.question_text or ''} {question.resolution_criteria or ''}".strip()


def resolution_source_provider(is_benchmarking: bool = False, *, fast_path: bool = False) -> ResearchCallable:
    """Factory returning the async ResearchCallable for the resolution-source fetcher.

    Gating (both hard):

    - ``is_benchmarking=True`` short-circuits to ``""`` (leakage guard — current
      page content post-dates any backtest window, same rationale as the
      prediction-market provider).
    - Env flag ``RESOLUTION_SOURCE_ENABLED`` must be truthy.

    ``fast_path`` is the question's time-budget thin-window mode, the same flag the
    orchestrator uses to drop the slow search providers. This provider is cheap and
    hard-capped, so it still runs; what the flag changes is that the two EXPENSIVE rungs of
    its escalation ladder — a 12-35 s Chromium launch and the paid reader — decline, recorded
    under ``counts["fast_path_skips"]``.

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
                fetch_resolution_sources(urls, query=_document_query(question), fast_path=fast_path),
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
                f"(js_wall/blocked — the escalation ladder rescued none of them); "
                f"{n_datasets_withheld} embedded dataset(s) withheld",
            )
        qid = getattr(question, "id_of_question", None)
        _log_fetch_outcome_markers(qid, results)
        record_raw_research(qid=qid, provider="resolution_source", payload=results)
        # Per-URL outcome map for the diagnostics block: even when the provider
        # returns a non-empty notice (all URLs failed → status `ok`), this surfaces
        # WHICH sources were lost so the block doesn't read as fully healthy.
        record_provider_detail(
            qid,
            "resolution_source",
            {"sources": _fetch_result_sources(results), "counts": _rung_counts(results)},
        )
        return format_resolution_sections(results, datetime.now(UTC))

    return _fetch
