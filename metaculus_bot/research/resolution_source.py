# SMELL-EXEMPT-monolithic-file-loc: the ladder spine stays whole because
# tests/resolution_source/*.py patches its names on THIS module; the outbound guard, the
# per-URL context and the one body-classification path now live in `research/fetch_ladder/`
# (receipts: scratch_docs_and_planning/fetch_ladder_unification_plan_2026-09-09.md).
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

The IMPERSONATED RETRY (`_impersonate_rung`, transport `research/impersonated_fetch.py`)
is the one rung that leaves aiohttp without leaving our address: a page that answered
our client 403 is re-dialed once through libcurl presenting a real Chrome TLS and HTTP/2
fingerprint, and the body re-enters the same classification path (HTML, a document, or
the raw text family). Measured 2026-09-04 from a GitHub Actions runner: four of the four
Akamai-fronted federal hosts that refused our aiohttp client (bls.gov, cdc.gov,
fsis.usda.gov, one of them a PDF) answered the impersonated GET 200, so that refusal is a
fingerprint verdict rather than an egress-IP one; the hosts that refused both stay the
Wayback and paid rungs' population. It sits between the direct fetch and the archive so a
live page beats a stale capture, and before the paid reader so a rescue saves the read.
Free, 403-only, memoized per host for the run, and behind a default-on kill switch
(`RESOLUTION_SOURCE_IMPERSONATE_ENABLED`). libcurl never touches aiohttp's connect-time
resolver, so the transport carries the SSRF invariants itself (a pre-resolved, pinned
connection per hop; every redirect re-guarded under the shared `MAX_REDIRECTS` cap).

A rung that leaves our own aiohttp client AND our address is the browser: a page that
answered 200 with nothing
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
import logging
import os
import time
from collections import Counter
from collections.abc import Awaitable
from datetime import UTC, datetime
from typing import Any, Literal
from urllib.parse import urlparse

import aiohttp
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    DOCUMENT_TEXT_PDF_MAX_BYTES,
    GAP_FILL_V2_READER_MODEL,
    GAP_FILL_V2_READER_THINKING_LEVEL,
    GOOGLE_API_KEY_ENV,
    RESOLUTION_SOURCE_DATAWRAPPER_HOP_WALL_MARGIN_S,
    RESOLUTION_SOURCE_DATAWRAPPER_MIN_HOP_BUDGET_S,
    RESOLUTION_SOURCE_DERIVED_API_MIN_BUDGET_S,
    RESOLUTION_SOURCE_ENABLED_ENV,
    RESOLUTION_SOURCE_HTTP_TIMEOUT,
    RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S,
    RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
    RESOLUTION_SOURCE_MAX_URLS,
    RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S,
    RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S,
    RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S,
    RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS,
    RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV,
    RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS,
    RESOLUTION_SOURCE_URL_CONTEXT_MIN_BUDGET_S,
    RESOLUTION_SOURCE_WALL_TIMEOUT,
    RESOLUTION_SOURCE_WAYBACK_MAX_AGE_DAYS,
    RESOLUTION_SOURCE_WAYBACK_MAX_ATTEMPTS,
    RESOLUTION_SOURCE_WAYBACK_MIN_BUDGET_S,
    RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS,
    env_flag_enabled,
)
from metaculus_bot.research import derived_api, impersonated_fetch, resolution_datawrapper, resolution_presentation
from metaculus_bot.research.fetch_ladder import classify, context, guard
from metaculus_bot.research.http_fetch import (
    MAX_REDIRECTS,
    DatawrapperChartRef,
    datawrapper_live_data_url,
    decode_text_body,
    host_semaphores,
)
from metaculus_bot.research.impersonated_fetch import (
    ImpersonateBudgetExhausted,
    ImpersonateDeclined,
    ImpersonatedResponse,
    ImpersonateTransportError,
    ImpersonateUnpinnable,
    fetch_impersonated,
)
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.raw_log import record_raw_research
from metaculus_bot.research.rendered_fetch import (
    RENDER_EXIT_RESERVE_MS,
    RENDER_SETTLE_MS,
    RENDER_TIMEOUT_MS,
    MemoScope,
    RenderBudgetExpired,
    RenderDomOverCeiling,
    RenderedPage,
    RenderOffHost,
    RenderTimeout,
    is_json_content_type,
    note_rendered_no_text,
    render_page,
    rendered_to_nothing,
)
from metaculus_bot.research.resolution_fetch_result import (
    _NON_OK_FETCH_STATUS,
    FetchResult,
    FetchStatus,
    FetchStatusReason,
    RungAttempt,
    RungSkipReason,
    _fetch_result_sources,
    fetch_outcome_token,
    looks_like_csv_rows,  # noqa: F401  # re-export: the Tier-1 suite imports the row-shape check from this module path
    server_header_token,
    vacuous_body_status,
)
from metaculus_bot.research.resolution_presentation import format_resolution_sections  # noqa: F401  # public re-export
from metaculus_bot.research.resolution_url_scan import (
    extract_source_urls,
    is_fred_url,
    is_metaculus_self_ref,
    is_yahoo_ticker_url,
    strip_markdown_escapes,  # noqa: F401  # re-export: the Tier-1 suite imports the markdown unescaper from this module path
)
from metaculus_bot.research.robots_policy import ROBOTS_FETCH_TIMEOUT_S, google_extended_blocks_url
from metaculus_bot.research.url_context_reader import NOT_ADDRESSED_SENTINEL, run_url_context_read
from metaculus_bot.research.wayback import (
    innermost_url,
    parse_snapshot_url,
    snapshot_age_days,
    wayback_lead,
    wayback_snapshot_url,
)

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Pure helpers — no I/O
# ---------------------------------------------------------------------------


def select_fetchable_urls(criteria: str | None, fine_print: str | None) -> list[str]:
    """Compose the fetchable URL list from a question's resolution criteria + fine print.

    Skips self-refs (the question platform's own site: metaculus.com or
    competitions.mantic.com), FRED, and Yahoo ticker URLs — those either add no new
    info or are covered by another provider. Caps at ``RESOLUTION_SOURCE_MAX_URLS``
    AFTER the skip filter so a run of leading self-refs / FRED / Yahoo URLs (questions
    often list their own page first) doesn't starve the real sources out of the
    fetch budget.
    """
    combined = f"{criteria or ''}\n\n{fine_print or ''}"
    urls = extract_source_urls(combined)
    filtered = [u for u in urls if not (is_metaculus_self_ref(u) or is_fred_url(u) or is_yahoo_ticker_url(u))]
    return filtered[:RESOLUTION_SOURCE_MAX_URLS]


async def _fetch_one_hop(
    session: Any, current_url: str, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
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
    semaphore and the open response, and since the extractor policy they can be TWO
    trafilatura passes rather than one: the second is skipped under
    ``RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S`` of remaining wall, which bounds the
    worst case without moving the work. Moving it would trade a measured hazard for an
    unmeasured restructure (FUTURE.md carries the entry); the meta-refresh hop that follows
    the classification needs the decoded text inside this loop either way.
    """
    async with guard._sem_for_host(host_sems, current_url):
        hop_timeout_s = min(
            RESOLUTION_SOURCE_HTTP_TIMEOUT, max(ctx.rung_budget_s(), RESOLUTION_SOURCE_MIN_HOP_TIMEOUT_S)
        )
        try:
            async with session.get(
                current_url,
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=hop_timeout_s, sock_read=hop_timeout_s),
            ) as resp:
                outcome = await classify._resolution_response_outcome(resp, current_url, ctx)
        except (TimeoutError, aiohttp.ClientError) as e:
            logger.info(f"resolution_source fetch error for {current_url}: {type(e).__name__}: {e}")
            return FetchResult(
                url=current_url,
                status="error",
                text="",
                http_status=None,
                content_type=None,
                failure_class=classify._network_failure_class(e),
                exc=type(e).__name__,
            )
    if isinstance(outcome, classify._PendingDocument):
        return await classify._finish_document(outcome, ctx)
    return outcome


def _impersonate_rung_applies(direct: FetchResult) -> bool:
    """Whether a browser's TLS fingerprint could plausibly turn ``direct`` into a readable page.

    ``blocked`` with an HTTP status in ``impersonated_fetch.IMPERSONATE_TRIGGER_STATUSES`` (403
    today), and both halves are load-bearing. ``blocked`` has four producers:
    ``_NON_OK_FETCH_STATUS`` maps 403, 406 and 429 to it, and :func:`_vetted_hop_target` returns
    it for a Metaculus self-reference hop carrying the REDIRECT's 301 or 302. The 403 test
    excludes that last case exactly, which matters: handing a URL this module refused to a second
    transport is the bypass the guard exists to prevent.

    The trigger set is the TRANSPORT's, read as a module attribute at call time rather than
    imported, for two reasons. It is the one trigger both fetchers share (gap-fill v2's ``fetch``
    and ``read_document`` ladders key on the same object), so the population cannot drift between
    them; and the test packages empty that attribute to decline the rung by default and restore the
    transport's own object in the tests that exercise it, which only bites on a read that resolves
    the name at call time.

    Excluded on purpose, each for its own reason. 429 is a throttle, not a fingerprint verdict,
    and retrying at once with a different fingerprint against a host that just asked us to slow
    down is the one shape where the retry could make our position worse; the 2026-09-04
    diagnostic measured impersonation helping only on 403s. 406 is a content-negotiation refusal,
    and impersonation changes the ``Accept`` headers as a side effect, so a 406 rung would be an
    untested guess. 401 is an authentication requirement no fingerprint changes, and is not even a
    ``blocked`` shape: absent from ``_NON_OK_FETCH_STATUS``, it falls through to ``error``. A 200
    carrying a challenge or throttle interstitial is not a representable trigger today, because
    Tier 1 has no throttle-phrase check (only gap-fill v2's ``fetch_outcomes`` has one) and such a
    page classifies as ``js_wall`` or ``thin_page``; FUTURE.md carries that entry, and the
    ``error`` with ``failure_class="tls"`` widening nothing has measured.
    """
    return direct.status == "blocked" and direct.http_status in impersonated_fetch.IMPERSONATE_TRIGGER_STATUSES


async def _impersonated_body_outcome(response: ImpersonatedResponse, ctx: context.LadderContext) -> FetchResult:
    """Classify a body the impersonated retry read, on the same three-way routing as a direct 200.

    The same content-type routing as :func:`_resolution_response_outcome`, read in the same order
    and off the same vocabularies, so a rescued page is indistinguishable downstream from a
    directly-fetched one: ``_HTML_CONTENT_TYPES`` to :func:`_classify_html_body`,
    ``is_json_content_type`` or ``_RAW_TEXT_CONTENT_TYPES`` to :func:`_raw_body_outcome`, and
    everything else, an empty Content-Type included, to :func:`_document_outcome` plus the parse
    :func:`_finish_document` runs once a document is held.

    ``http_status`` is the IMPERSONATED response's 200, not the direct 403. The bytes came with a
    200 and that is the honest record: a rescue's fetch line reads ``status=ok http=200
    route=impersonate`` with no ``failure_class``, and the fact that the direct fetch was refused
    lives on the escalation line's ``from_status=blocked``, exactly as a Wayback rescue reports the
    snapshot's own status. This diverges from :func:`_rendered_rung`, which passes the direct
    status because there the direct GET also got a 200.

    No meta-refresh hop. :func:`_resolution_html_outcome` runs one after a no-content
    classification; following it from here would mean deciding which transport dials the target
    and re-entering the whole hop loop from inside a rung, so this router calls
    :func:`_classify_html_body` directly (FUTURE.md carries the entry).

    The page's own context, not :func:`_aux_ctx`. The Wayback and derived-feed rungs use the
    child context because they fetch a DIFFERENT URL on the page's behalf and must not let that
    URL's inner rungs hijack the page's route. These bytes are the cited page's own, so a
    ``pdf_local`` attempt belongs on the page's record, and a document rescue reads
    ``route=pdf_local``: the accounting a meta-refresh hop onto a PDF already produces, in the
    file's own words "the hop got us the bytes, the local read is what the text came from".
    """
    content_type = response.content_type
    if any(ct in content_type for ct in classify._HTML_CONTENT_TYPES):
        classified = await classify._classify_html_body(
            response.body,
            response.url,
            content_type,
            http_status=response.status,
            # What the retry left of the wall: the extractor's optional second pass declines under
            # its floor rather than overrunning the provider.
            remaining_wall_s=ctx.rung_budget_s(),
        )
        return classified.result
    if is_json_content_type(content_type) or any(ct in content_type for ct in classify._RAW_TEXT_CONTENT_TYPES):
        return classify._raw_body_outcome(response.body, response.url, content_type, http_status=response.status)
    # The content-type router's verdict before the `%PDF-` sniff, as `_resolution_response_outcome`
    # records it, so a document read off this rung pairs `pdf_local` with the same `from_status` a
    # directly fetched one does.
    outcome = classify._document_outcome(
        response.body, response.url, content_type, ctx, http_status=response.status, from_status="unsupported_type"
    )
    if isinstance(outcome, classify._PendingDocument):
        return await classify._finish_document(outcome, ctx)
    return outcome


async def _impersonate_or_record_the_skip(
    retry_url: str, budget_s: float, host_sems: dict[str, asyncio.Semaphore], attempt: RungAttempt
) -> ImpersonatedResponse | None:
    """Dial the transport, or turn its decline into the record the attempt should carry.

    Two declines are SKIPS rather than fired attempts, each stamped on the attempt already started
    rather than appended as a second one, the pattern :func:`_render_or_record_the_skip` uses for
    its skips. :class:`ImpersonateUnpinnable` is a hop whose host would not pin to a vetted public
    address; the pin can fail on the FIRST hop, where nothing was dialed, or on a later redirect
    hop, where the earlier hops were, so the skip says the pin failed on some hop, not that no
    wall was spent. :class:`ImpersonateBudgetExhausted` is the wall running out while the
    transport waited on a pre-dial await (the vetting lookup, a redirect re-guard, the host gate):
    nothing was dialed on that hop, so it is the ``wall_budget`` skip the pre-gate floor records,
    and not a fired attempt whose ``blocked`` outcome would read as the host refusing the
    fingerprint. Every other :class:`ImpersonateDeclined` leaves the attempt fired, so the
    dispatcher closes it on the direct status and the archive reads ``route=impersonate
    status=blocked``: we tried the fingerprint and this is still the answer. Logged at the level
    the shape deserves. A transport failure at INFO, as the direct path's own are, because a reset
    or a handshake failure is a fact about the host rather than about this rung; the spent
    budget at WARNING like every other wall skip; a refused hop, an oversized body, a redirect
    chain past the cap or a pin that did not hold at WARNING, and the transport already logged
    that last one at ERROR.

    The two body caps are the direct path's own: ``RESOLUTION_SOURCE_MAX_RESPONSE_BYTES`` for a
    page and ``DOCUMENT_TEXT_PDF_MAX_BYTES`` for a declared PDF, exactly the pair
    :func:`_resolution_pdf_outcome` reads under, so a cited PDF between the two is read on this
    rung as the direct fetch would have read it rather than declined as oversized.
    """
    netloc = urlparse(retry_url).netloc
    try:
        return await fetch_impersonated(
            retry_url,
            host_sems=host_sems,
            deadline_monotonic_s=time.monotonic() + budget_s,
            per_hop_timeout_s=RESOLUTION_SOURCE_HTTP_TIMEOUT,
            max_bytes=RESOLUTION_SOURCE_MAX_RESPONSE_BYTES,
            document_max_bytes=DOCUMENT_TEXT_PDF_MAX_BYTES,
        )
    except ImpersonateUnpinnable as exc:
        logger.warning("resolution_source: the impersonated retry of %s could not pin its host (%s)", netloc, exc)
        attempt.skipped_reason = "impersonate_unpinnable"
    except ImpersonateBudgetExhausted as exc:
        logger.warning("resolution_source: skipping the impersonated retry for %s: %s", netloc, exc)
        attempt.skipped_reason = "wall_budget"
    except ImpersonateTransportError as exc:
        logger.info(
            "resolution_source: the impersonated retry of %s failed in transport (failure_class=%s exc=%s)",
            netloc,
            exc.failure_class,
            exc.exc,
        )
    except ImpersonateDeclined as exc:
        logger.warning(
            "resolution_source: the impersonated retry of %s produced nothing (%s: %s)",
            netloc,
            type(exc).__name__,
            exc,
        )
    return None


async def _impersonate_rung(
    url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult | None:
    """Re-dial a page that answered our aiohttp client 403, presenting a real browser's fingerprint.

    Measured 2026-09-04 from a GitHub Actions runner (``scripts/probes/fetch_diagnostic.py``): four
    Akamai-fronted federal hosts (bls.gov twice, one of them a PDF, cdc.gov and fsis.usda.gov)
    answered the bot's own client 403 and the same GET through ``curl_cffi`` with Chrome
    impersonation 200, so that refusal was a TLS and HTTP/2 fingerprint verdict and is recoverable
    client-side. The four hosts that also refused the impersonated GET (Cloudflare, CloudFront and
    DataDome fronts) are the egress-IP population and stay the Wayback and paid rungs' business.
    Free: no key, no model call, no spend.

    The transport is :mod:`metaculus_bot.research.impersonated_fetch`, which carries the SSRF
    invariants itself because libcurl never touches aiohttp's connect-time ``FilteringResolver``:
    it pre-resolves the host through the repo's own vetting predicate, pins the connection to the
    vetted address, re-guards and re-pins every redirect hop under the shared ``MAX_REDIRECTS``
    cap, and caps the body as the direct fetch does: the page cap for every body, and one re-dial
    under the document cap when a declared PDF aborts on the first, so a cited PDF between the two
    caps (one of the four measured recoverable URLs is a PDF) is read here as the direct path
    would have read it (:func:`_impersonate_or_record_the_skip` names the two constants).

    Ordered gates, cheapest first, after the trigger (:func:`_impersonate_rung_applies`). The kill
    switch (``impersonated_fetch.impersonation_enabled``, the transport's own reading of
    ``RESOLUTION_SOURCE_IMPERSONATE_ENABLED``, ON by default in code unlike the paid rung's
    default-off) records ``impersonate_disabled``. The URL dialed is ``direct.url``, the hop that
    ANSWERED 403 rather than the cited URL, because :func:`_resolution_status_outcome` sets ``url``
    to the answering hop and that is the URL the host actually refused; when the two differ the
    landing is re-vetted through :func:`_landing_refused`, and a refusal is a decline with no
    attempt, the same helper :func:`_rendered_rung` runs. The rung's ATTEMPT stays keyed on the
    cited ``url``, which is what the escalation line names. The per-run memo
    (``impersonated_fetch.impersonation_refused``, keyed by the HOST that answered a block plus the
    exact URL dialed to reach it) records ``impersonate_host_refused``: a host that answered the
    impersonated client with a block status is not going to answer the next cited URL on it
    differently in the same run, and a chain that ended in one is not walked twice, while a host
    that merely redirected into the block keeps its other URLs. The memo is process-global and shared with
    gap-fill v2, whose ``fetch`` and ``read_document`` ladders write it too, so the earlier
    refusal it records may have been a v2 fetch of a URL no question ever cited. Then the wall
    budget, through :meth:`LadderContext.claim_rung_budget` with the meta-refresh hop's floor: one
    GET against a host that just answered us, no launch and no gate contended process-wide.

    Deliberately NO fast-path skip (:func:`_skip_for_fast_path`). That token separates "the
    question's close left no room for a browser" from "this rung ran out of the provider's own
    clock" and its docstring reserves it for the two EXPENSIVE rungs; this rung costs exactly what
    the meta-refresh hop costs, and the cheap rungs run on the fast path unchanged. Do not add the
    gate for symmetry.

    Outcomes. A decline from the transport (:class:`ImpersonateDeclined`) leaves the attempt FIRED
    and :func:`_run_rung` closes it on the direct status, which is the ``route=impersonate
    status=blocked`` record the archive wants: we tried the fingerprint and this is still the
    answer. The one exception is :class:`ImpersonateUnpinnable`, a hop whose host would not pin
    to a vetted public address, which is stamped on the attempt already started as its own skip
    rather than appended as a second one (the pattern :func:`_rendered_rung` uses for
    ``render_non_200``). On the first hop that is near-impossible in practice, since the direct
    fetch resolved this host through the filtering resolver moments earlier; on a later redirect
    hop it is a target the direct fetch never resolved, so the skip means the pin failed on some
    hop rather than that nothing was dialed. A non-200 answer stamps the rung's own outcome
    (``blocked`` for a still-403, so the escalation line reads ``rung=impersonate
    outcome=blocked``; ``not_found`` for a 404 or 410; ``error`` for anything else), memoizes the
    host when the status is block-shaped (``impersonated_fetch.IMPERSONATE_BLOCK_STATUSES``: the
    three ``blocked`` rows of the status table plus 401 and 503, which stamp ``error`` here and
    still switch the host off), and returns None. A 200 goes through
    :func:`_impersonated_body_outcome`, the same classification a direct 200 gets, and is returned
    ONLY when it is ``success``. An impersonated 200 that classified as unreadable stamps its
    verdict on the attempt (``outcome=js_wall``) and leaves ``blocked`` standing, because
    replacing it would change the fetch line's status without giving any later rung a way to act
    on it (the dispatcher's browser block keys on ``direct``, not on a rung's result), and
    ``blocked`` is what keeps the paid rung reachable for that URL. FUTURE.md carries letting the
    dispatcher escalate on a rung's result.
    """
    if not _impersonate_rung_applies(direct):
        return None
    if not impersonated_fetch.impersonation_enabled():
        ctx.skip_rung("impersonate", direct.status, url, "impersonate_disabled")
        return None
    retry_url = direct.url
    if await guard._landing_refused(retry_url, url, action="re-dialing"):
        return None
    if impersonated_fetch.impersonation_refused(retry_url):
        ctx.skip_rung("impersonate", direct.status, url, "impersonate_host_refused")
        return None
    budget_s = ctx.claim_rung_budget("impersonate", direct.status, url, RESOLUTION_SOURCE_IMPERSONATE_MIN_BUDGET_S)
    if budget_s is None:
        return None
    attempt = ctx.start_rung("impersonate", direct.status, url)
    response = await _impersonate_or_record_the_skip(retry_url, budget_s, host_sems, attempt)
    if response is None:
        return None
    netloc = urlparse(retry_url).netloc
    if response.status != 200:
        outcome = _NON_OK_FETCH_STATUS.get(response.status, "error")
        # The memo write is the transport's rule (`IMPERSONATE_BLOCK_STATUSES`: a 404 says the path
        # is gone, which says nothing about the host's view of our fingerprint), keyed on the host
        # that ANSWERED plus the exact URL dialed: the impersonated client follows redirects itself,
        # so the block can come from a later hop's netloc, and that is the host that refused us,
        # while the dialed host merely redirected and keeps its other URLs.
        impersonated_fetch.note_refusal_if_block_shaped(
            dialed_url=retry_url, answered_url=response.url, status=response.status
        )
        attempt.outcome = outcome
        # The Server header names which CDN refused the impersonated GET (a host that refuses both
        # clients is otherwise indistinguishable from one whose fingerprint scoring changed), and
        # the elapsed time separates an edge's instant refusal from a challenge that ran the clock.
        logger.info(
            "resolution_source: the impersonated retry of %s was answered %d by %s (%s; server=%s elapsed=%.1fs); "
            "the direct result stands",
            netloc,
            response.status,
            urlparse(response.url).netloc,
            outcome,
            server_header_token(response.server) or "none",
            response.elapsed_s,
        )
        return None
    result = await _impersonated_body_outcome(response, ctx)
    if result.chrome_metric_withheld:
        # The metric withheld the impersonated body's extraction. `chrome_metric_withholds` counts a
        # withhold anywhere on the URL's ladder, and a 403 direct fetch had no body for the metric
        # to withhold, so the fact is stamped on the direct result: `_fetch_one` carries it from
        # there onto whatever this ladder leaves standing, the direct result when nothing rescues
        # the page or a later rung's result when one does. Idempotent on a rescue, whose own
        # result already carries the flag.
        direct.chrome_metric_withheld = True
    # The rung's own verdict, stamped before deciding: a body that classified as unreadable is a
    # fact about the page the escalation line has to keep even though the direct status stands.
    attempt.outcome = result.status
    if result.status == "success":
        return result
    logger.info(
        "resolution_source: the impersonated retry of %s got a 200 that classified as %s; the direct result stands",
        netloc,
        result.status,
    )
    return None


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


async def _render_or_record_the_skip(
    url: str, budget_s: float, host_sems: dict[str, asyncio.Semaphore], attempt: RungAttempt
) -> RenderedPage | None:
    """Run the transport under the rung's wall bound; on anything but a page, record why.

    Every way the transport can stop short of a rendered page lands on ``attempt.skipped_reason``
    here, so the rung itself reads as one statement per outcome. The mapping, and why each token
    is what it is, is the :func:`_rendered_rung` docstring's business; this function only
    applies it.
    """
    goto_timeout_ms = int(min(RENDER_TIMEOUT_MS, budget_s * 1000) - RENDER_SETTLE_MS)
    try:
        page = await asyncio.wait_for(
            render_page(
                url,
                memo_scope=_RENDER_MEMO_SCOPE,
                host_gate=guard._sem_for_host(host_sems, url),
                goto_timeout_ms=goto_timeout_ms,
                # The transport's exit (shared teardown bound, launch, driver stop) runs AFTER
                # this deadline and has to land inside the wait_for below, so the deadline is
                # the budget less that reserve. Strictly safer: it can only shorten the goto or
                # decline earlier at the transport's own navigation floor.
                deadline_monotonic_s=time.monotonic() + budget_s - RENDER_EXIT_RESERVE_MS / 1000,
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
    except RenderTimeout as exc:
        # The transport's own DOM-read bound: a browser ran and the page kept navigating (or the
        # transport re-raised it for a URL it already cut off this run). A fact about the page.
        logger.warning(
            "resolution_source: the rendered rung for %s was cut off by the transport (%.1fs budget): %s; "
            "leaving the direct result",
            urlparse(url).netloc,
            budget_s,
            exc,
        )
        attempt.skipped_reason = "render_timeout"
        return None
    except TimeoutError:
        # The rung's own wait_for above: the render was still queued behind the two gates when
        # the budget ran out, or the transport overran its exit reserve. Neither says anything
        # about the page, so it is the wall binding, the same skip the pre-gate check records.
        # Ordered after the two transport exceptions, which both subclass this.
        logger.warning(
            "resolution_source: the rendered rung for %s outlived its %.1fs wall budget before the transport "
            "answered; leaving the direct result",
            urlparse(url).netloc,
            budget_s,
        )
        attempt.skipped_reason = "wall_budget"
        return None
    except RenderDomOverCeiling:
        # Chromium rendered the page and the DOM is over `RENDERED_DOM_MAX_CHARS`: a fact about
        # the page, kept out of `renderer_unavailable` so the install-failed signal stays clean.
        # The transport already logged the size.
        attempt.skipped_reason = "render_dom_too_large"
        return None
    except RenderOffHost:
        # Chromium's main frame landed on a host other than the pinned one, a server-side redirect
        # the route guard never sees, so the transport refused the DOM unread on its pre-read check,
        # or discarded it unpublished when the navigation committed during the read itself. Also a
        # fact about the page, and nothing from that render is published either way. The transport
        # already logged both hosts.
        attempt.skipped_reason = "render_off_host"
        return None
    if page is None:
        # The transport declines with ONE signal for several causes — Playwright missing or
        # broken, a host that will not pin to a public IP, or a browser error — and its own
        # WARN/DEBUG lines say which. Recorded as a SKIP rather than a fired rung because nothing
        # was rendered: it then claims no `route=` and emits no escalation line, while keeping
        # the measured wall_s that says what the declined launch cost.
        attempt.skipped_reason = "renderer_unavailable"
    return page


async def _rendered_rung(
    url: str, direct: FetchResult, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
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
    actually left of the DEADLINE handed to it less the settle and the DOM read, or declines
    under its own floor before a browser is launched, so a goto that runs its budget out can
    still be settled and read. That deadline is the remaining budget LESS the transport's exit
    reserve (``RENDER_EXIT_RESERVE_MS``: the shared teardown bound plus a second for the launch
    and the driver stop), because the transport spends those after its DOM is in hand and this
    rung's own bound has to fit them too. Degrading to the direct result costs one page;
    overrunning the provider's outer ``wait_for`` costs every page the question already fetched.

    The whole transport call — queue, launch, navigation, DOM read and teardown — is ALSO held
    to the remaining budget with ``asyncio.wait_for``. That bounds when this rung stops WAITING,
    not when the transport stops RUNNING: ``wait_for`` cancels the render and then awaits its
    unwinding teardown, so the reserve above is what keeps that teardown inside the wall, and a
    render that runs every bound out hands its DOM back before the cut instead of being
    cancelled in its own exit. The receipt is ogimet.com (2026-09-03): the goto timed out at
    33 s as designed and ``page.content()`` then blocked for 40 s more, for a 76 s render against
    a 45 s wall. The two bounds are recorded apart, by the exception class the transport raises.
    Its own DOM-read bound raises :class:`RenderTimeout`: a browser ran and the page kept
    navigating, which is a fact about the page and is its own skip, ``render_timeout``, rather
    than ``renderer_unavailable`` — a cut-off render says nothing about whether Chromium works,
    and must not latch that warning. This rung's outer bound raises a bare ``TimeoutError``: the
    render was still queued behind the two gates, or the transport overran its exit reserve,
    neither of which is about the page, so it is recorded as ``wall_budget`` like the pre-gate
    floor check and the post-gate :class:`RenderBudgetExpired`. The direct result is what stands
    either way. The transport memoises a timed-out URL itself, at the raise site and only when a
    browser actually ran; with the reserve in place its DOM-read bound lands before this rung's
    outer cut even in the salvage shape (goto ran its budget out), so the memo is written there
    too. A memoised URL re-raises on the next question, so it is recorded the same way again.

    The rendered DOM re-enters :func:`_classify_html_body`, so a rescued page gets the same
    chart read, ARIA rewrite, floors and disclosure leads as a directly-fetched one — unless the
    browser was answered with something other than a 200. The direct GET got a 200 for this URL
    (that is the trigger), so a non-200 main-frame status is the edge telling the browser apart,
    and its markup (a 403 or 429 interstitial routinely clears the chrome floor) is not the page:
    the rung leaves the direct result standing and does not memoise, because a 429 is retryable;
    that is its own skip, ``render_non_200``. A DOM over ``RENDERED_DOM_MAX_CHARS`` is likewise a
    fact about the page and its own skip, ``render_dom_too_large`` (the transport raises
    :class:`RenderDomOverCeiling` for it), so neither inflates ``renderer_unavailable``. A main
    frame that landed on a host other than the one the browser's DNS pin covers is refused by the
    transport unread on its pre-read check, or discarded unpublished when the navigation commits
    during the read itself (:class:`RenderOffHost`), and is its own skip too, ``render_off_host``:
    a server-side redirect the route guard never sees, so a fact about the page rather than the
    install, and nothing from that render is published either way.
    When the DOM STILL carries nothing, the JSON the page fetched for itself is the last free
    route (:func:`_derived_api_from_harvest`) — a JavaScript dashboard's numbers arrive over XHR
    and are in its HTML at no wait condition. Only once that fails too is the URL memoized
    (:func:`note_rendered_no_text`), so a second URL on the same page in this run does not spend
    another launch to learn the same thing; that memo hit is its own skip, ``rendered_no_text``,
    so it never inflates the count the operator reads as the Chromium install having failed.

    The browser is handed ``direct.url``, the URL the direct fetch LANDED on once its redirect
    hops were followed and re-guarded, rather than the cited ``url``: the pin then covers the
    host that actually serves the content, which is also the host the landing check holds the
    browser to, and a page whose canonical form is one ordinary hop away (``example.com`` to
    ``www.example.com``) is not refused for taking it. When the two differ, the landing is
    re-vetted through :func:`_landing_refused`, the one home of that re-vet (shared with the
    impersonated retry, which dials ``direct.url`` for the same reason), because this is where
    the URL the browser dials is decided; the refusal is a decline rather than a terminal
    result, so this site returns None with no attempt. The render memos are keyed on
    the URL rendered; the classifier's base, and so the ``FetchResult.url`` a rescue carries onto
    the ``RESOLUTION_SOURCE_FETCH`` line and into the published ``### <url>`` heading, is the URL
    the browser's main frame LANDED on (``RenderedPage.document_url``: the direct fetch's final
    hop, or a same-host hop past it); the rung's attempt stays keyed on the cited ``url``, which
    is what the escalation line names, so a per-URL join between the two lines keys on the
    escalation line; and the harvested feed is remembered for the cited URL's host, which is the
    host the next cited URL asks :func:`_derived_api_rung` about.
    """
    if not _rendered_rung_applies(direct):
        return None
    render_url = direct.url
    if await guard._landing_refused(render_url, url, action="rendering"):
        return None
    if rendered_to_nothing(render_url, memo_scope=_RENDER_MEMO_SCOPE):
        ctx.skip_rung("rendered", direct.status, url, "rendered_no_text")
        return None
    budget_s = ctx.claim_rung_budget("rendered", direct.status, url, RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S)
    if budget_s is None:
        return None
    attempt = ctx.start_rung("rendered", direct.status, url)
    page = await _render_or_record_the_skip(render_url, budget_s, host_sems, attempt)
    if page is None:
        return None
    if page.http_status is not None and page.http_status != 200:
        logger.warning(
            "resolution_source: the browser was answered %d for %s where the direct GET got 200; "
            "not reading that page as content",
            page.http_status,
            urlparse(render_url).netloc,
        )
        # Its own skip, not a fired rung: nothing about the page was read, so the attempt claims
        # no route and emits no escalation line, and the count keeps "Chromium refused where our
        # GET was not" measurable. No memo, because a 429 has to stay re-requestable.
        attempt.skipped_reason = "render_non_200"
        return None
    classified = await classify._classify_html_body(
        page.html.encode("utf-8", errors="replace"),
        # The document the DOM came from: `final_url` when the navigation committed (same host as
        # `render_url` by construction, the path may differ after a same-host client-side redirect
        # or meta refresh), so relative links and the published section URL name the real
        # document, as the direct path's last hop and the `meta_refresh` route already do.
        page.document_url,
        page.content_type or "text/html",
        # The direct fetch's status, not the browser's: this page answered 200 and carried no
        # text, which is the fact the record should keep. Chromium reports no status at all
        # when a goto timed out and the DOM was salvaged, and a non-200 never reaches here.
        http_status=direct.http_status if direct.http_status is not None else 200,
        # What the browser left of the wall: the render spent the rest, and the extractor's
        # optional second pass declines under its floor rather than overrunning the provider.
        remaining_wall_s=ctx.rung_budget_s(),
    )
    if classified.result.chrome_metric_withheld:
        # The metric withheld the rendered DOM's extraction. `chrome_metric_withholds` counts a
        # withhold anywhere on the URL's ladder, and a js_wall direct fetch had nothing for the
        # metric to withhold, so the fact is stamped on the direct result: `_fetch_one` carries
        # it from there onto whatever this ladder leaves standing, the direct result when nothing
        # rescues the page or the harvested feed when it does.
        direct.chrome_metric_withheld = True
    # The render's own verdict, stamped before the harvest gets its turn: when the harvested
    # feed rescues the page, the ladder's result is `success` and the closer would otherwise
    # credit the render with a rescue the DOM never delivered.
    attempt.outcome = classified.result.status
    if classified.result.status == "success":
        return classified.result
    derived = _derived_api_from_harvest(url, direct, page, ctx)
    if derived is not None:
        return derived
    note_rendered_no_text(render_url, memo_scope=_RENDER_MEMO_SCOPE)
    return None


def _derived_api_from_harvest(
    url: str, direct: FetchResult, page: RenderedPage, ctx: context.LadderContext
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

    The lead LEADS and its cost comes out of the per-URL cap
    (:func:`resolution_presentation._lead_then_capped_body`),
    because a feed served with its provenance line trimmed off is a JSON blob nobody can check.
    """
    lead = derived_api.derived_api_lead(endpoint, url)
    return FetchResult(
        url=url,
        status="success",
        text=resolution_presentation._lead_then_capped_body(lead, raw, url),
        http_status=http_status,
        content_type="application/json",
    )


async def _derived_api_rung(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
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
    feed = await _fetch_direct(session, endpoint.endpoint_url, host_sems, context._aux_ctx(ctx))
    if feed.status != "success":
        return None
    if not is_json_content_type(feed.content_type or ""):
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
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult | None:
    """Fetch the archive's freshest capture of ``url`` and serve it, or withhold it.

    The fetch goes through :func:`_fetch_direct`, so the snapshot is classified by exactly the
    path a live page is — including the chart read and the chrome floor — and inherits the SSRF
    preflight, the per-hop re-guard and the budget-clamped hop timeout. What comes back extra is
    the FINAL URL, which is where the archive puts the 14-digit capture timestamp.

    Three outcomes, in this order, and the order is the design.

    The inner URL is UNWRAPPED (repeatedly, since a capture OF a capture presents
    ``web.archive.org`` as its own inner host) and re-checked first through
    :func:`_hop_refusal`, the one home of the two checks every derived URL owes, because a
    hostname check on ``web.archive.org/web/…/metaculus.com/…`` sails past every self-reference
    filter in the pipeline: an archived Metaculus page in front of a forecaster is the question
    quoting itself. Then a snapshot the archive could not serve at all (no capture, or a capture that
    404s) DECLINES: there is no archived copy, which is a different fact from a stale one, and
    the direct route's own status says more about the source than a fact about the archive would.
    Only a capture we actually READ and cannot date, or can date and it is too old, is withheld
    as ``stale_data`` — because the disclosure that makes a snapshot admissible is its age, and a
    copy with no usable date cannot carry it. The direct status is not lost by that swap either:
    the ``RESOLUTION_SOURCE_ESCALATION`` line for this rung carries ``from_status``, and the
    withhold keeps the direct fetch's HTTP status and failure diagnostics, so the
    ``RESOLUTION_SOURCE_FETCH`` line it replaces the direct result on still says which host
    refused us and from which CDN.
    """
    snapshot = await _fetch_direct(session, wayback_snapshot_url(url, now=ctx.now), host_sems, context._aux_ctx(ctx))
    parsed = parse_snapshot_url(snapshot.url)
    captured_of = None if parsed is None else innermost_url(parsed.inner_url)
    if captured_of is not None and await guard._hop_refusal(captured_of) is not None:
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
            # The cited HOST's status and diagnostics, not the archive's: this verdict replaces
            # the direct result on the FETCH line, where `http=200` was the archive answering
            # and the missing `failure_class` / `server` undercounted the blocked population
            # the ladder exists for. Only the success below reports the snapshot's own status,
            # because those bytes are the archive's.
            http_status=direct.http_status,
            content_type=direct.content_type,
            failure_class=direct.failure_class,
            exc=direct.exc,
            server=direct.server,
            # A verdict names its own rung. The dispatcher otherwise stamps the LAST rung that
            # fired, and the paid rung fires after this one: a stale capture the reader then
            # failed to improve on came back `route=url_context status=stale_data`, a status that
            # rung cannot produce, on the field that partitions the archive by route.
            route="wayback",
        )
    # The lead LEADS and its cost comes out of the per-URL cap
    # (:func:`resolution_presentation._lead_then_capped_body`):
    # an archived page whose age line has been trimmed off is being passed off as the live one.
    lead = wayback_lead(parsed, age_days, direct.status)
    return FetchResult(
        url=url,
        status="success",
        text=resolution_presentation._lead_then_capped_body(lead, snapshot.text, url),
        http_status=snapshot.http_status,
        content_type=snapshot.content_type,
        datawrapper_charts=snapshot.datawrapper_charts,
        unreadable_embeds=snapshot.unreadable_embeds,
        precision_rescued=snapshot.precision_rescued,
    )


async def _wayback_rung(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
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
# Two outcomes inside the set are excluded by REASON rather than by status — see
# :func:`_url_context_rung_applies` — so this set is the ceiling on the population, not the
# population itself.
_URL_CONTEXT_TRIGGER_STATUSES: frozenset[FetchStatus] = frozenset(
    {"blocked", "js_wall", "error", "no_resolving_content"}
)

# The reasons that take an outcome OUT of the population above. Scoped on the reason rather than by
# dropping the status, because the statuses they ride are otherwise exactly what the rung exists
# for: `embed_shell` and `thin_page` are pages our client genuinely could not read, and a 403
# `blocked` is the rung's whole reason to exist.
#   `no_matching_passage` — a document we read END TO END whose passage selection matched no query
#   term. Its bytes were never the problem (we hold its full text and its outline), so paying
#   Gemini to re-read the same PDF buys nothing.
#   `metaculus_self_ref` — a redirect WE refused because it landed on the question platform's own
#   site. The rung is handed the CITED url, so Gemini would follow the same redirect and read the
#   page we refused: a paid read that by construction returns nothing new, and on Mantic the other
#   bots' forecasts read in as grading evidence. The same bypass `ssrf_blocked` is kept out of the
#   trigger set to prevent, closed here by reason because the self-reference's status is `blocked`
#   by contract.
_URL_CONTEXT_EXCLUDED_REASONS: frozenset[FetchStatusReason] = frozenset({"no_matching_passage", "metaculus_self_ref"})


def _url_context_rung_applies(direct: FetchResult) -> bool:
    """Whether a model-mediated read could plausibly resolve ``direct``.

    The trigger statuses above, minus the outcomes inside them a paid read cannot help with or
    must not be tried on (``_URL_CONTEXT_EXCLUDED_REASONS``, which says why for each).
    """
    if direct.status not in _URL_CONTEXT_TRIGGER_STATUSES:
        return False
    return direct.status_reason not in _URL_CONTEXT_EXCLUDED_REASONS


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
    session: Any, robots_url: str, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
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
            _fetch_direct(session, robots_url, host_sems, context._aux_ctx(ctx)), ROBOTS_FETCH_TIMEOUT_S
        )
    except TimeoutError:
        logger.info(
            "resolution_source: robots.txt pre-check for %s did not answer in %.1fs", robots_url, ROBOTS_FETCH_TIMEOUT_S
        )
        return None
    return result.text if result.status == "success" else None


async def _url_context_robots_skip(
    session: Any, url: str, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
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
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> tuple[str, float] | None:
    """Every gate the paid read has to clear, in increasing cost order; ``(api_key, budget_s)`` or None.

    The trigger population, the flag (default off in code, on in every bot workflow), the question's
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
        context._skip_for_fast_path(ctx, "url_context", direct, url)
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
    # Last, and only for a read that cleared every cheaper gate, so a slot is spent on a read
    # that is actually about to fire — not on one robots or the wall already declined. Mirrors
    # the Wayback per-question cap: a question citing several dead sources pays at most
    # RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS times inside one provider wall.
    if not ctx.shared.take_url_context_attempt():
        logger.info(
            "resolution_source: skipping the url_context rung for %s — this question's %d paid read(s) are spent",
            urlparse(url).netloc,
            RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS,
        )
        ctx.skip_rung("url_context", direct.status, url, "url_context_cap")
        return None
    return api_key, budget_s


def _withheld_reply_preview(reply: str) -> str:
    """The head of a paid reply we are DISCARDING, collapsed onto one log line.

    Whitespace-collapsed because a model's answer arrives with newlines and a multi-line log
    record is what makes a run log unreadable, and bounded by
    ``RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS`` because the point is to audit what the read
    said, not to keep it.
    """
    collapsed = " ".join(reply.split())
    if len(collapsed) <= RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS:
        return collapsed
    return f"{collapsed[:RESOLUTION_SOURCE_WITHHELD_REPLY_LOG_CHARS]}…"


async def _url_context_rung(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
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

    An answer that opens with ``NOT_ADDRESSED_SENTINEL`` is WITHHELD as ``no_resolving_content``
    / ``not_addressed``. The prompt asks for that opening when the retrieved page does not discuss
    the ask, so it is the designed non-answer, and rendered under the url_context lead it was
    prose standing in for an absent section, the shape :func:`_finish_document` closes for a PDF
    with ``no_matching_passage``. The page was retrieved (so it is not ``ungrounded``) and the
    read was paid for, so the verdict stays on the record as this rung's own rather than declining
    to the direct result.
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
        # SDK attached no url_metadata entry at all. A registered marker spec, named
        # resolution_source_urlcontext_ungrounded_suppressed in scripts/telemetry/markers.py, so
        # the spelling and the always-present `statuses=` field are a data contract.
        logger.warning(
            f"RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url={url} statuses={','.join(statuses) or 'none'}"
        )
        if text.strip():
            # The suppressed answer itself, on its own unregistered line (see the
            # `not_addressed` twin below for why a withheld reply is kept at all). Only when
            # there IS one: the same branch fires on an empty reply, where there is nothing to
            # audit.
            logger.info(f"url_context ungrounded reply for {urlparse(url).netloc}: {_withheld_reply_preview(text)}")
        return FetchResult(
            url=url,
            status="ungrounded",
            text="",
            # The host's status and diagnostics stay on a verdict that served nothing: a
            # model-mediated read has no status of its own, and the FETCH line this result
            # replaces the direct one on is where "which host refused us" is counted.
            http_status=direct.http_status,
            content_type=direct.content_type,
            failure_class=direct.failure_class,
            exc=direct.exc,
            server=direct.server,
        )
    answer = text.strip()
    if answer.startswith(NOT_ADDRESSED_SENTINEL):
        # A registered marker spec like its two URLCONTEXT siblings, named
        # resolution_source_urlcontext_not_addressed in scripts/telemetry/markers.py; `host=`
        # because the rollout question is which hosts Gemini can
        # reach but finds nothing on.
        logger.warning(f"RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED: url={url} host={urlparse(url).netloc}")
        # What the withheld read actually SAID, on a separate unregistered line so the marker's
        # own shape stays a data contract. Without it the verdict is unauditable: "the page does
        # not discuss this" and "the model read the bot-challenge page it was served" reach this
        # branch identically, and the text that tells them apart was being dropped on the floor.
        logger.info(f"url_context not_addressed reply for {urlparse(url).netloc}: {_withheld_reply_preview(answer)}")
        return FetchResult(
            url=url,
            status="no_resolving_content",
            text="",
            http_status=direct.http_status,
            content_type=direct.content_type,
            status_reason="not_addressed",
            failure_class=direct.failure_class,
            exc=direct.exc,
            server=direct.server,
        )
    # The lead LEADS and is budgeted out of the cap
    # (:func:`resolution_presentation._lead_then_capped_body`): a model's
    # answer rendered without the disclosure reads as the page itself.
    lead = _url_context_lead(direct.status)
    return FetchResult(
        url=url,
        status="success",
        text=resolution_presentation._lead_then_capped_body(lead, answer, url),
        http_status=direct.http_status,
        content_type="text/plain",
    )


async def _run_rung(
    ctx: context.LadderContext, fallback: FetchStatus, rung: Awaitable[FetchResult | None]
) -> FetchResult | None:
    """Await one rung and close the attempts it opened with that rung's own wall and outcome.

    The one home for the bracket every dispatcher site used to copy by hand: read
    ``len(ctx.rungs)`` before the rung runs, await it, then close every attempt opened since
    with the status that stood once it was over — its result's, or ``fallback`` (the status it
    left standing) when it declined (:meth:`LadderContext.close_rungs`). Structural rather than
    stylistic: a rung awaited without the bracket still returned its result, and its attempt fell
    through to :func:`_stamped_with_route`'s last-resort close, which stamps the ladder's FINAL
    status and the whole-ladder wall — the two figures the per-rung close exists to keep apart,
    with the marker parsing either way. ``rung`` is the coroutine created at the call site, which
    runs none of its code until it is awaited here, so the length is read first.
    """
    first_new = len(ctx.rungs)
    result = await rung
    ctx.close_rungs(first_new, fallback if result is None else result.status)
    return result


async def _escalate_unresolved(
    session: Any, url: str, direct: FetchResult, *, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
) -> FetchResult:
    """Run the escalation rungs a direct fetch's outcome earns, cheapest first.

    Returns the FIRST rung's rescue, or ``direct`` unchanged when every rung declines or fails.
    A rung that fired and produced nothing still leaves its attempt on the context, which is
    what makes ``route=rendered status=js_wall`` readable in the archive as "we tried the
    browser and this is still the answer" — the same convention the meta-refresh hop already
    follows. The Wayback rung is the one rung whose non-rescue is a VERDICT rather than None
    (``stale_data``, a capture we read and will not serve); it is kept as the fallback rather
    than as an early return, so the paid rung below is still reachable for that page.

    Each rung is closed the moment its result is known (:func:`_run_rung`), so the attempts it
    opened carry that rung's own wall and outcome rather than the ladder's: the status it
    returned, or the direct status it left standing when it declined.

    ``session`` is the aiohttp session the rungs that issue an ordinary GET use; the browser
    rung and the impersonated retry ignore it, because Chromium and libcurl each bring their own
    transport.
    """
    if direct.status == "success":
        return direct
    # First among the rungs, and the position is a reading choice rather than a functional one:
    # the trigger sets are disjoint (`_rendered_rung_applies` fires only on `js_wall` and the
    # `thin_page` shape of `no_resolving_content`, never on `blocked`), so this rung never
    # contends for the browser escalation gate. It matches the `FetchRoute` Literal's own ladder
    # order, meets a reader with the cheap free retry before the expensive ones, sits before the
    # archive so a live page beats a stale capture, and before the paid reader so a rescue saves
    # the read on that URL entirely.
    impersonated = await _run_rung(ctx, direct.status, _impersonate_rung(url, direct, host_sems=host_sems, ctx=ctx))
    if impersonated is not None:
        return impersonated
    if _rendered_rung_applies(direct):
        # The two browser-family rungs run under one per-host gate for this question, so a
        # same-host sibling asks `endpoint_for` only after this escalation has recorded (or
        # failed to record) an endpoint — see `QuestionRungBudget.browser_escalation_gate`.
        async with ctx.shared.browser_escalation_gate(url):
            derived = await _run_rung(
                ctx, direct.status, _derived_api_rung(session, url, direct, host_sems=host_sems, ctx=ctx)
            )
            if derived is not None:
                return derived
            # Declined HERE rather than inside the rung: the rung's own gates all cost something
            # (a budget read, a memo lookup, a launch), and the fast path is a fact about the
            # question the dispatcher already holds.
            if ctx.fast_path:
                context._skip_for_fast_path(ctx, "rendered", direct, url)
            else:
                rendered = await _run_rung(ctx, direct.status, _rendered_rung(url, direct, host_sems, ctx))
                if rendered is not None:
                    return rendered
    # Reached only for the statuses the browser rungs do not claim — the two trigger sets are
    # disjoint by construction (see `_WAYBACK_TRIGGER_STATUSES`), so the order between them is a
    # reading choice: free-and-local first, then the route whose egress is not ours.
    wayback = await _run_rung(ctx, direct.status, _wayback_rung(session, url, direct, host_sems=host_sems, ctx=ctx))
    if wayback is not None and wayback.status == "success":
        return wayback
    # Last, because it is the only rung that spends money and the only one whose product is a
    # model's answer rather than the host's bytes. Off by default in code, on in every bot workflow.
    # It is asked about the DIRECT outcome, and an archive WITHHOLD does not stand in its way: a
    # capture too old to serve is still a page we could not read fresh, which is exactly the
    # population this rung exists for. The withhold stays the fallback below, so with the flag
    # off (or the reader declining) a stale capture still reports `stale_data`. The paid rung's
    # own attempt closes on the DIRECT status when it declines, like every other rung: the
    # archive's verdict is not an outcome a model read can produce, and on the escalation line
    # `rung=url_context outcome=stale_data` read as if it had.
    read = await _run_rung(ctx, direct.status, _url_context_rung(session, url, direct, host_sems=host_sems, ctx=ctx))
    if read is not None:
        return read
    return wayback if wayback is not None else direct


async def _fetch_one(
    session: Any, url: str, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext | None = None
) -> FetchResult:
    """Fetch a single URL directly, then escalate what the direct route could not read.

    ``ctx`` carries the question text a PDF digest ranks passages against, the wall-clock
    origin each rung bounds itself with, and the rung attempts stamped onto the returned
    result. It defaults to a fresh one so the fetch surface can still be driven with three
    arguments, which is what every existing caller and test does.
    """
    ctx = context.LadderContext() if ctx is None else ctx
    direct = await _fetch_direct(session, url, host_sems, ctx)
    # The rungs inside the direct fetch (the meta-refresh hop, the local PDF read) are over
    # once it returns, and its status is what they left standing.
    ctx.close_rungs(0, direct.status)
    escalated = await _escalate_unresolved(session, url, direct, host_sems=host_sems, ctx=ctx)
    if direct.chrome_metric_withheld:
        # The withhold is a fact about this URL's ladder, carried onto whatever the ladder
        # returns for it: a rung's rescue has its own extraction, which the metric never
        # withheld, so summed off final results alone the withholds the ladder then paid off
        # — the policy's whole point — reached no count at all (`chrome_metric_withholds`).
        escalated.chrome_metric_withheld = True
    return context._stamped_with_route(escalated, ctx)


async def _fetch_direct(
    session: Any, url: str, host_sems: dict[str, asyncio.Semaphore], ctx: context.LadderContext
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
    tag (:func:`_hop_refusal` is the one place both checks live, and
    :func:`_vetted_hop_target` maps its verdict onto the terminal result). The
    connect-time :class:`FilteringResolver` (see :func:`_get_session`) provides the
    actual DNS-rebinding boundary; these preflight checks are fast-fail
    observability so we surface ``ssrf_blocked`` without opening a session. Hops of
    both shapes are followed in-band and share the one ``MAX_REDIRECTS`` cap.

    No retries (Tier 1 anti-goal). Any aiohttp/asyncio error becomes ``error``. Escalation
    beyond this route is :func:`_escalate_unresolved`'s job, so this function stays exactly
    what it always was: the plain fetch, terminal on its own outcome.
    """
    # Guard the initial URL before any network I/O.
    if not await guard.is_public_http_url(url):
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
    if not await guard.is_public_http_url(url):
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

    async with guard._sem_for_host(host_sems, url):
        try:
            async with session.get(url, allow_redirects=False) as resp:
                return await resolution_datawrapper._datawrapper_dataset_outcome(resp, chart, parent_url, url)
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
                failure_class=classify._network_failure_class(e),
                exc=type(e).__name__,
            )


async def fetch_resolution_sources(urls: list[str], *, query: str = "", fast_path: bool = False) -> list[FetchResult]:
    """Fetch each URL under per-netloc Semaphore(1) politeness, then hop to
    the live datasets of any Datawrapper charts the fetched pages embed.

    ``query`` is the question's title plus resolution criteria. It never touches the
    network; its one job is ranking which passages of a cited PDF a forecaster sees.
    Empty is legitimate (a caller with no question text in hand) and simply means a
    document renders its header and outline with no passages. ``fast_path`` is the
    question's time-budget thin-window mode; it rides every URL's :class:`LadderContext`
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

    session_cm = guard._get_session()
    async with session_cm as session:
        try:
            # One context per URL: the rung attempts belong to that URL's result, while
            # the query and the wall-clock origin are the same for all of them.
            # ONE shared rung budget across this question's URLs, and one per-URL context each:
            # the Wayback cap is per question (every snapshot shares one host gate), while the
            # rung attempts belong to the URL they were spent on.
            shared_budget = context.QuestionRungBudget()
            page_tasks = [
                asyncio.create_task(
                    _fetch_one(
                        session,
                        u,
                        host_sems,
                        context.LadderContext(query=query, started=started, shared=shared_budget, fast_path=fast_path),
                    )
                )
                for u in urls
            ]
            tasks.extend(page_tasks)
            page_results = list(await asyncio.gather(*page_tasks, return_exceptions=False))

            picks = resolution_datawrapper._select_datawrapper_charts(page_results)
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
            return resolution_datawrapper._interleave_dataset_results(page_results, picks, dataset_results)
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
    asks about vs the paid reader's ``not_addressed``, ``unreadable_document``'s
    ``no_text_layer`` vs ``encrypted`` vs
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
    is the stub rather than the target the fetch line names, and for a ``route=rendered``
    rescue is the cited URL while the fetch line names the URL the browser landed on (since
    2026-09-04; before that boundary the two agreed, so a per-URL join against the cited URL
    is exact only for earlier records and otherwise keys on the escalation line).
    """
    for r in results:
        reason = f" reason={r.status_reason}" if r.status_reason else ""
        route = f" route={r.route}" if r.route != "direct" else ""
        # Failure diagnostics, each appended only when present so a success and every archived
        # line stay byte-identical. Keyed and tail-positioned in a fixed order after `route`, so
        # a line carrying some but not all of them parses without a positional group claiming a
        # neighbour's value.
        failure_class = f" failure_class={r.failure_class}" if r.failure_class else ""
        exc = f" exc={r.exc}" if r.exc else ""
        server = f" server={r.server}" if r.server else ""
        logger.info(
            f"RESOLUTION_SOURCE_FETCH: question={qid} url={r.url} status={fetch_outcome_token(r)} "
            f"http={r.http_status if r.http_status is not None else 'n/a'} "
            f"embeds={','.join(r.unreadable_embeds) if r.unreadable_embeds else 'none'}"
            f"{reason}{route}{failure_class}{exc}{server}"
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
    # Typed on `RungSkipReason` so every literal key indexed below is checked: a misspelt reason
    # is a type error rather than a permanently-zero count silently absent from the archive.
    skips_by_reason: Counter[RungSkipReason | Literal[""]] = Counter(
        attempt.skipped_reason for attempt in attempts if attempt.skipped_reason
    )
    budget_skips_by_rung = Counter(attempt.rung for attempt in attempts if attempt.skipped_reason == "wall_budget")
    return {
        "meta_refresh_hops": fired_by_rung["meta_refresh"],
        # No `impersonate_rescues` key beside this: rescues are read off `route=` on the fetch
        # marker, which already partitions the population by rung. The only per-rung rescue count
        # in this dict is `chrome_metric_withholds_rescued`, and that exists because it needs a
        # join the marker cannot express.
        "impersonate_attempts": fired_by_rung["impersonate"],
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
        **{f"{rung}_budget_skips": budget_skips_by_rung[rung] for rung in context._BUDGET_GATED_RUNGS},
        # Its own count rather than folded into the budget skips: a document left unread
        # because two others were already parsing says the 2-slot gate is the binding
        # constraint, which is a different thing to fix than a question that ran late.
        "pdf_contention_skips": skips_by_reason["parse_contention"],
        # Also its own count, for the same reason: a browser rung that never rendered because
        # Chromium is missing on the runner (the install step is `continue-on-error` in every
        # workflow, so its absence is by design) says something different from a question that
        # ran out of wall, and both are invisible in `rendered_attempts`.
        "renderer_unavailable_skips": skips_by_reason["renderer_unavailable"],
        # Its own count too: a render that launched and was cut off by the transport's DOM-read
        # cap is a page that keeps navigating, which is a fact about the page, whereas the two
        # counts above are facts about the runner and about the question's clock (the rung's own
        # outer cut, which fires while the render is still queued, is a `wall_budget` skip).
        # Folding it into either would hide the population the ogimet receipt (2026-09-03) is
        # the first member of.
        "render_timeout_skips": skips_by_reason["render_timeout"],
        # Its own count: the browser was answered a non-200 where the direct GET got a 200, the
        # edge telling Chromium apart. Counted with the fired renders it read as a render that
        # produced chrome again, and the rate at which the runner's browser is refused is the
        # question the escalation ladder's case rests on.
        "render_non_200_skips": skips_by_reason["render_non_200"],
        # Its own count: Chromium rendered the page and the DOM was over `RENDERED_DOM_MAX_CHARS`,
        # a fact about the page that used to be folded into `renderer_unavailable_skips`, where it
        # pointed triage at the Playwright install.
        "render_dom_too_large_skips": skips_by_reason["render_dom_too_large"],
        # Its own count: Chromium's main frame landed on a host other than the pinned one (a
        # server-side redirect hop the route guard never sees), so the transport refused the DOM
        # unread. A fact about the page, and the one count that says how often a cited page sends
        # the browser somewhere else, which is what prices the host-equality rule.
        "render_off_host_skips": skips_by_reason["render_off_host"],
        # And its own count: a browser rung skipped because an earlier question in this run
        # already rendered the same URL to nothing is the memo doing its job, not a runner without
        # Chromium — folded into `renderer_unavailable_skips` it inflated the install-failed signal.
        "rendered_no_text_skips": skips_by_reason["rendered_no_text"],
        # Also its own count: a question that spent its two snapshot attempts on earlier
        # cited URLs is a question whose per-question cap is binding, which is a different
        # thing to tune than a question that ran out of wall.
        "wayback_cap_skips": skips_by_reason["wayback_cap"],
        # The paid rung's analogue: a question that spent its per-question paid-read budget on
        # earlier cited URLs, which is the spend cap binding rather than the wall or the flag.
        "url_context_cap_skips": skips_by_reason["url_context_cap"],
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
        # Its own count for the same reason: with the kill switch off the impersonated retry fires
        # nowhere, and without this key that run is byte-identical in the archive to one where no
        # cited page ever earned the retry.
        "impersonate_disabled_skips": skips_by_reason["impersonate_disabled"],
        # Its own count: the impersonated retry declined because a hop's host would not pin to a
        # vetted public address. On the first hop nothing was dialed, and since the direct fetch
        # resolved that host through the filtering resolver moments earlier a nonzero count means
        # DNS disagreed with itself (a flake, or a rebinding host that flipped); on a later redirect
        # hop the earlier hops WERE dialed and the target is one the direct fetch never resolved, so
        # the DNS-disagreement reading does not apply to that case.
        "impersonate_unpinnable_skips": skips_by_reason["impersonate_unpinnable"],
        # Its own count: the per-run memo declining a cited URL on a host that already answered the
        # impersonated client with a block status this run, which is the memo doing its job rather
        # than a failure, the distinction `rendered_no_text_skips` draws for the browser. The memo
        # is process-global and shared with gap-fill v2, so the earlier refusal may have been a v2
        # fetch of a URL no question cited.
        "impersonate_host_refused_skips": skips_by_reason["impersonate_host_refused"],
        # The extractor policy's decisions, per cited URL. `chrome_metric_withholds`: the
        # line-shape metric withheld an HTML extraction of the URL somewhere on its ladder — the
        # final result's own (including a chart-rescued page, whose chart block still published
        # without that text) or the direct fetch's, carried onto the rung result that replaced it
        # (`_fetch_one`). `chrome_metric_withholds_rescued`: the subset a rung past the direct
        # fetch then served (`route` is not `direct` and the result is a success) — the policy's
        # headline win, a menu tree withheld and the price table rendered, which summed off the
        # rescue's own flag alone reached neither key. `precision_fallback_rescues`: the published
        # text is the precision re-extraction, taken after the default one failed the metric.
        "chrome_metric_withholds": sum(1 for r in results if r.chrome_metric_withheld),
        "chrome_metric_withholds_rescued": sum(
            1 for r in results if r.chrome_metric_withheld and r.status == "success" and r.route != "direct"
        ),
        "precision_fallback_rescues": sum(1 for r in results if r.precision_rescued),
    }


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
        unfetched = [r for r in cited if r.status != "success"]
        n_datasets_withheld = sum(1 for r in results if r.chart_id is not None and r.status != "success")
        if unfetched or n_datasets_withheld:
            # Counts rather than a verdict. This line used to assert the ladder "rescued none of
            # them" whenever anything was unfetched, which is false on the ordinary mixed
            # question — Wayback serving one host while a second stays walled — and the summary is
            # where a run log gets read first. A rescue is a success no direct fetch produced,
            # the same reading `_rung_counts` takes; the statuses say what the losses actually
            # were instead of naming two of them by hand.
            n_rescued = sum(1 for r in cited if r.status == "success" and r.route != "direct")
            lost_statuses = ",".join(sorted({r.status for r in unfetched})) or "none"
            logger.info(
                f"resolution_source: {len(unfetched)}/{len(cited)} cited urls unfetched "
                f"({lost_statuses}); {n_rescued} rescued by a later rung; "
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
        return resolution_presentation.format_resolution_sections(results, datetime.now(UTC))

    return _fetch
