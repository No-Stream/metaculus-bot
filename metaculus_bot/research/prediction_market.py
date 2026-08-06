"""Prediction-market research provider: the seam module for ranked market retrieval.

Retrieves markets from Polymarket + Kalshi + Manifold + PredictIt that bear on a Metaculus
question, and returns a `MarketSnapshot` the forecaster reads as a peer cross-check. The
retrieval machinery lives in `metaculus_bot.research.market_retrieval`; this module owns the
provider seam, the per-session caches, the degradation counters, the aiohttp session factory,
and the one retry-wrapped LLM invocation both LLM stages share.

Four stages per question:

    1a  PREFETCH      Kalshi's complete open-events catalogue + PredictIt's whole dump
    1b  QUERY AUTHOR  one LLM call adding domain vocabulary  (concurrent with 1a)
    2   VENUE SEARCH  Manifold + Polymarket, every query, per-query failure isolation
    2.5 ENRICH        one Manifold detail GET per candidate, for the rules text
    3   POOL ASSEMBLY three channels unioned; channel order IS the ranking
    4   RANK          one LLM call over the whole pool, up to 8 rows, model's order

Two measured facts shaped this, and both are things NOT to "improve" (the bake-off receipts
are in `scratch/bakeoff_run_2026-08-03/results/`):

- **Selection binds, not generation.** A perfect ranker over the pool that already exists
  reaches 14/16 questions; the same pool's deterministic top-4 reaches 5/16. So generation is
  recall-maximal — no score floor anywhere — and all the judgment sits in one ranking call.
- **The render step was the weak link, not the LLM.** In the previous design a keep/drop
  auditor voted KEEP on 58 of 58 near-identical rows and a venue round-robin then evicted 43
  of them. There is no venue quota, no fairness interleave and no post-hoc re-scoring in this
  pipeline, at any stage.

Two behaviours the old keyword/fuzzy design had are deliberately gone. There is no fuzzy score
FLOOR (it discarded the adjacent-cut markets that carry most of the evidential value; the
fuzzy scorer survives as a way to ORDER Kalshi's ~9,762 events down to a promptable 100). And
the provider path passes `as_of=None`: the filter dropped every market closing before the
question resolved, which is precisely the "same quantity, adjacent month" class the ranked arm
scored most of its wins on, and 20 of 47 archived runs had Polymarket fetch candidates and
render nothing because of it. The parameter and the cache key survive for explicit callers
(backtests, replay tooling), where leakage defence is real; the drop itself now happens inside
pool assembly, so an ineligible row frees its width slot instead of being deleted after the
width has already been spent.

Soft-fail on every path. This provider returns an empty snapshot on any failure and never
raises — a broken venue API must never break a forecast — so the degradation counters below
are the ONLY route by which an outage reddens CI.

One analysis hazard follows from the ranker being allowed to say nothing. A question where it
returns zero rows renders no section at all, so the `## Prediction Market Snapshot` header is
absent — which means a comment- or log-backfilled archive record, whose `providers_used` is
reconstructed by scanning for that header, drops the provider entirely, while an artifact record
still lists it under `providers_attempted`. A fall in prediction-market presence across
backfilled records can therefore mean "the ranker declined" rather than "the provider broke".
The `MARKET_RANKING:` line's `outcome=` field distinguishes the two. No code change: the
header-scan reconstruction is lossy by construction and always was.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any

import aiohttp
import openai
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    KALSHI_CATALOGUE_WALL_TIMEOUT,
    MARKET_QUERY_AUTHOR_BACKOFFS,
    MARKET_QUERY_AUTHOR_WALL_TIMEOUT,
    MARKET_RANKER_BACKOFFS,
    MARKET_RANKER_WALL_TIMEOUT,
    PREDICTION_MARKET_TIMEOUT,
    PREDICTION_MARKETS_ENABLED_ENV,
    env_flag_enabled,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.llm_configs import MARKET_QUERY_AUTHOR_LLM_CONFIG, MARKET_RANKER_LLM_CONFIG
from metaculus_bot.llm_retry import invoke_with_transient_retry
from metaculus_bot.research.http_fetch import build_session
from metaculus_bot.research.market_retrieval import generation, ranking, rendering, venues
from metaculus_bot.research.market_retrieval.http import HTTP_RETRY_BACKOFF_SECS, PLATFORM_HTTP_TIMEOUT
from metaculus_bot.research.market_retrieval.queries import (
    build_query_author_prompt,
    dedupe_queries,
    deterministic_queries,
    parse_query_author,
    strip_dates_and_numbers,
)

# The row types and the liquidity vocabulary live in the market_retrieval package; this module
# re-exports them because it is the seam every consumer imports from — `raw_log`'s `asdict`,
# `provider_health`'s `getattr`, and the test suite all reach them through
# `metaculus_bot.research.prediction_market`.
from metaculus_bot.research.market_retrieval.types import (
    LIQUIDITY_DEEP_USD,  # noqa: F401  # re-export: consumed by the liquidity-contract tests
    LIQUIDITY_THIN_USD,  # noqa: F401  # re-export
    MANIFOLD_HIGH_BETTORS,  # noqa: F401  # re-export
    MANIFOLD_THIN_BETTORS,  # noqa: F401  # re-export
    MarketChild,  # noqa: F401  # re-export: the multi-outcome sub-row, archived inside MarketMatch
    MarketMatch,
    MarketSnapshot,
    ScalarEstimate,  # noqa: F401  # re-export: a scalar market's value, archived inside MarketMatch
    SettlementSource,  # noqa: F401  # re-export: archive shape + test constructions
    _FetchTally,
    _liquidity_label,  # noqa: F401  # re-export
)
from metaculus_bot.research.provider_diagnostics import is_lost_source, record_provider_detail
from metaculus_bot.research.provider_health import (
    VENUE_EXPECTED_LIQUIDITY_FIELDS,
    VenueObservation,
    record_catalogue_size,
    record_venue_observation,
)
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.raw_log import record_raw_research
from metaculus_bot.time_utils import _as_utc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Catalogue cache TTLs. Both venues are enumerated whole and re-enumerating them per question
# would dominate the snapshot's budget, so the pull is once per process per 6h.
KALSHI_CACHE_TTL_S = 6 * 60 * 60
PREDICTIT_CACHE_TTL_S = 6 * 60 * 60

# The venues whose own search index is the only way in, in the order stage 2 fans out.
_SEARCH_VENUES: tuple[str, ...] = ("polymarket", "manifold")


# Worst-case wall clock for one snapshot, computed from the same constants the stages use so a
# raised wall cannot silently outgrow PREDICTION_MARKET_TIMEOUT. A previous 31.0-versus-30.0
# overrun shipped once, which is why the arithmetic is code rather than a comment: stages 1a and
# 1b run concurrently so the chain takes their max, and the rest is serial. Per-stage worst case
# is `(len(backoffs) + 1) * wall + sum(backoffs)`.
def _llm_stage_worst(wall_timeout: float, backoffs: tuple[float, ...]) -> float:
    return (len(backoffs) + 1) * wall_timeout + sum(backoffs)


_HTTP_STAGE_WORST = PLATFORM_HTTP_TIMEOUT * 2 + HTTP_RETRY_BACKOFF_SECS
SNAPSHOT_STAGE_BUDGET_S: float = (
    max(
        _llm_stage_worst(MARKET_QUERY_AUTHOR_WALL_TIMEOUT, MARKET_QUERY_AUTHOR_BACKOFFS),
        KALSHI_CATALOGUE_WALL_TIMEOUT,
        _HTTP_STAGE_WORST,
    )
    + _HTTP_STAGE_WORST
    + generation.MANIFOLD_DETAIL_WALL_S
    + _llm_stage_worst(MARKET_RANKER_WALL_TIMEOUT, MARKET_RANKER_BACKOFFS)
)


# ---------------------------------------------------------------------------
# Per-source diagnostics tokens
# ---------------------------------------------------------------------------


def _platform_source_token(matches: list[MarketMatch], tally: _FetchTally) -> str:
    """Classify one platform's outcome as an `ok(N)` / `none` / loss source token.

    `none` is reserved for "every sub-fetch succeeded and nothing matched" — the one benign
    empty outcome `provider_diagnostics.is_lost_source` does not flag. So a lost sub-fetch has
    to produce a loss token even when other sub-fetches returned matches: otherwise a total
    outage reads as a healthy `none`, and a platform that lost one of two queries reads as a
    clean `ok(N)`.
    """
    if tally.failed:
        if tally.ok == 0:
            return "error(all_queries_failed)"
        return f"partial({tally.ok}/{tally.ok + tally.failed})"
    return f"ok({len(matches)})" if matches else "none"


# ---------------------------------------------------------------------------
# Per-session caches (module-scoped; reset via `_reset_session_caches`)
# ---------------------------------------------------------------------------

# Kalshi projected-events catalogue: (timestamp_monotonic, events_list).
_KALSHI_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
# PredictIt markets cache: (timestamp_monotonic, markets_list).
_PREDICTIT_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
# Snapshot cache keyed by (qid, as_of_iso). The as_of leg keeps backtest runs at different
# as-of instants from sharing a snapshot computed at one as-of; on the provider path as_of is
# None, which is what finally makes this cache hittable (the old `datetime.now(utc)` branch
# changed the key on every call).
_SNAPSHOT_CACHE: dict[tuple[int, str], MarketSnapshot] = {}

# Per-run count of failed Kalshi CATALOGUE pulls. The catalogue is the generation backbone —
# it feeds both the settlement-source join and the fuzzy channel — and the provider soft-fails,
# so without a counter a dead pull is INVISIBLE (the 2026-07-25 observability hole:
# research_provider_failures=0 while a Kalshi path was dead). The orchestrator folds this into
# alertable_count, so a catalogue that dies every question reddens CI. A one-off transient
# bumps it once, an accepted rare false alarm mirroring gap_fill_v2_error_count.
#
# This counter and `_SOURCE_LOSSES` both bump on a failed pull, so one catalogue outage adds 2
# to alertable_count. That is deliberate over-counting rather than a bug: the two counters
# carry different marker fields, and the point of either is that CI goes red.
#
# Module-level like the caches => accumulates per run; reset between tests.
_KALSHI_CATALOGUE_FETCH_FAILURES: int = 0


def _bump_kalshi_catalogue_failure() -> None:
    global _KALSHI_CATALOGUE_FETCH_FAILURES
    _KALSHI_CATALOGUE_FETCH_FAILURES += 1


def kalshi_catalogue_fetch_failures() -> int:
    """Per-run count of failed Kalshi catalogue pulls (folded into alertable_count)."""
    return _KALSHI_CATALOGUE_FETCH_FAILURES


def reset_series_degradation_counter() -> None:
    """Zero the catalogue-degradation counter at run start.

    The provider is a stateless callable, so the counter lives at module scope; without a
    run-start reset it would leak across runs sharing one process (and across tests, polluting
    every later alertable_count == 0 assertion). Called from forecast_questions alongside
    reset_pchip_stats — same per-run cadence. The name is unchanged from when this counter
    tracked the retired /series fetch, because the orchestrator imports it by name."""
    global _KALSHI_CATALOGUE_FETCH_FAILURES
    _KALSHI_CATALOGUE_FETCH_FAILURES = 0


# Per-run count of LOST prediction-market SOURCES: one per venue whose fan-out lost a
# sub-fetch, one per whole-provider failure (snapshot timeout, outer-except), one when the
# query author comes back unusable, and one when the ranking call does. Those last two are why
# this counts SOURCES rather than venues: a dead ranker degrades every venue's contribution
# without any venue failing. The causes are distinguished per-source in
# `MarketSnapshot.sources` (`ranking:error(...)` vs `polymarket:error(...)`), which rides both
# the published comment and the schema-v2 research archive.
#
# Operator decision 2026-07-25: alert on ANY source loss rather than only a total blackout —
# maximum sensitivity, accepting that one flaky venue can redden most runs. The provider
# soft-fails internally, so like the catalogue counter this is the only path by which an outage
# reaches CI. Module-level like the caches => accumulates per run.
_SOURCE_LOSSES: int = 0


def _bump_source_loss() -> None:
    global _SOURCE_LOSSES
    _SOURCE_LOSSES += 1


def prediction_market_source_losses() -> int:
    """Per-run count of lost prediction-market sources (folded into alertable_count)."""
    return _SOURCE_LOSSES


def reset_source_loss_counter() -> None:
    """Zero the source-loss counter at run start (see
    `reset_series_degradation_counter` for why module-scoped counters need this)."""
    global _SOURCE_LOSSES
    _SOURCE_LOSSES = 0


def _reset_session_caches() -> None:
    """Clear all per-session caches. Called between tests and at session start."""
    global _KALSHI_CATALOGUE_FETCH_FAILURES, _SOURCE_LOSSES
    _KALSHI_CACHE.clear()
    _PREDICTIT_CACHE.clear()
    _SNAPSHOT_CACHE.clear()
    _KALSHI_CATALOGUE_FETCH_FAILURES = 0
    _SOURCE_LOSSES = 0


def _get_session() -> aiohttp.ClientSession:
    """Construct a fresh aiohttp session. Patched in tests.

    No headers arg: the JSON APIs get aiohttp's default UA (flipping to a browser UA is a
    separate experiment — see the resolution-source plan).
    """
    return build_session(timeout_s=PLATFORM_HTTP_TIMEOUT)


# ---------------------------------------------------------------------------
# The one LLM invocation helper, shared by both LLM stages
# ---------------------------------------------------------------------------


async def _invoke_market_llm(
    config: dict, prompt: str, *, wall_timeout: float, backoffs: tuple[float, ...], label: str
) -> str:
    """Invoke one market-retrieval LLM stage under the elapsed-gated retry wrapper.

    ONE helper for both stages, deliberately: it is a single patch point for the tests, and it
    makes it impossible for the query author and the ranker to drift on retry policy. Each
    stage passes its OWN wall and backoff ladder, which is what keeps the serial chain of
    worst cases inside `SNAPSHOT_STAGE_BUDGET_S`.

    Both configs pin `allowed_tries=1`, so this wrapper is the SOLE retry layer: it gives the
    call its own wall cap (otherwise the snapshot-level `wait_for` is the only bound and a
    stalled stage takes the whole snapshot down with it) and replaces forecasting-tools'
    un-gated `random.uniform(5, 10)` tenacity sleep with one bounded, elapsed-gated backoff
    that never fires on a deterministic client error.

    Returns `""` on any transient failure. Both callers' parsers treat an empty completion as
    unusable, so the soft-fail and the unparseable-output path are one path rather than two.
    """
    # Constructor errors are config bugs (bad model slug, missing API key wiring) and should
    # crash loudly. Only `.invoke` is expected to face transient LLM errors.
    llm = build_llm_with_openrouter_fallback(**config)
    try:
        return await invoke_with_transient_retry(
            lambda: llm.invoke(prompt),
            wall_timeout=wall_timeout,
            label=label,
            backoffs=backoffs,
        )
    # `openai.APIError` and NOT `litellm.exceptions.APIError`: every litellm transport exception
    # (Timeout, RateLimitError, APIConnectionError, InternalServerError, ServiceUnavailableError)
    # subclasses the openai root but NOT litellm's own APIError, so catching the latter caught
    # only a bare `APIError` and let every realistic provider blip escape — past `_rank_pool`'s
    # `except RankingUnusable` to the snapshot-level net, discarding the WHOLE snapshot where the
    # documented fail-open would have rendered the pool-order slate. Same idiom as
    # `research/orchestrator.py` and `ablation/leakage_screen.py`. Deliberately NOT `Exception`:
    # the TypeError/AttributeError family must still crash (§2 fail-fast), and `CancelledError`
    # subclasses neither root so the `wait_for` boundary stays intact.
    except (openai.APIError, asyncio.TimeoutError, RuntimeError):
        logger.warning(f"{label} LLM call failed", exc_info=True)
        return ""  # noqa: ASYNC910


# ---------------------------------------------------------------------------
# Stage 1a — catalogue prefetches
# ---------------------------------------------------------------------------


async def _kalshi_catalogue(session: Any, *, qid: int | None) -> tuple[list[dict[str, Any]], str]:
    """The complete projected Kalshi open-events catalogue, cached ~6h.

    Returns ``(events, source_token)``; the token is what says whether the pull succeeded, so
    there is nothing for a separate boolean to disagree with. The completeness-gated write below
    is the ONLY writer, deliberately: the read path checks the TTL and nothing else, so any
    incremental warm would pin an error-truncated — often EMPTY — list carrying a fresh
    timestamp, and every later question in the run would then read it back as a healthy
    `ok(N)` with no HTTP and no counter bump. A partial pull is still returned to THIS question,
    which is what keeps a lost page from costing the caller the pages that did arrive.
    """
    cached = _KALSHI_CACHE.get("events")
    if cached is not None:
        timestamp, events = cached
        if (time.monotonic() - timestamp) < KALSHI_CACHE_TTL_S:
            if qid is not None:
                record_catalogue_size(qid=qid, source="kalshi_events", entries=len(events), fetch_ok=True)
            return events, f"ok({len(events)})" if events else "none"  # noqa: ASYNC910

    pull = await venues.kalshi_prefetch_events(session)
    if pull.complete:
        _KALSHI_CACHE["events"] = (time.monotonic(), pull.events)
    else:
        _bump_kalshi_catalogue_failure()

    # A pull that reports SUCCESS and hands pool assembly an empty catalogue is a contradiction
    # the pool size alone cannot show (it looks identical to "the venue had nothing to say"),
    # and an empty catalogue now zeroes the settlement join AND the fuzzy channel.
    if qid is not None:
        record_catalogue_size(qid=qid, source="kalshi_events", entries=len(pull.events), fetch_ok=pull.complete)
    if not pull.complete:
        return pull.events, pull.token or "error(unknown)"
    return pull.events, f"ok({len(pull.events)})" if pull.events else "none"


async def _predictit_universe(session: Any, *, qid: int | None) -> tuple[list[dict[str, Any]], _FetchTally]:
    """PredictIt's whole ~197-market dump, cached ~6h. One unpaginated GET.

    The tally carries the None-vs-`[]` distinction forward: a failed fetch is a lost source,
    while a successful fetch of an empty dump is Signal C's business.
    """
    cached = _PREDICTIT_CACHE.get("markets")
    if cached is not None:
        timestamp, markets = cached
        if (time.monotonic() - timestamp) < PREDICTIT_CACHE_TTL_S:
            if qid is not None:
                record_catalogue_size(qid=qid, source="predictit_markets", entries=len(markets), fetch_ok=True)
            return markets, _FetchTally(ok=1)  # noqa: ASYNC910

    markets = await venues.predictit_prefetch(session)
    if markets is None:
        # A failed fetch is already the source-loss counter's business; recording a catalogue
        # observation here too would double-count one outage (see provider_health Signal C).
        return [], _FetchTally(failed=1)
    if qid is not None:
        record_catalogue_size(qid=qid, source="predictit_markets", entries=len(markets), fetch_ok=True)
    _PREDICTIT_CACHE["markets"] = (time.monotonic(), markets)
    return markets, _FetchTally(ok=1)


# ---------------------------------------------------------------------------
# Stage 1b — the query author (concurrent with 1a)
# ---------------------------------------------------------------------------


async def _authored_queries(title: str, resolution_criteria: str) -> tuple[tuple[str, ...], str]:
    """Extra search queries from one LLM call, ADDITIVE to the deterministic set.

    Additive rather than replacing is load-bearing rather than stylistic: a replacing author's
    failure mode is an empty query set, which is indistinguishable from "no markets exist" —
    the silent failure that hid the Manifold breakage for 17+ days. Additive, its worst case is
    "no gain", so `()` costs no recall and is still reported as a lost source (a permanently
    dead author would otherwise be invisible, which is the exact failure class the degradation
    counters exist for).

    Runs concurrently with the catalogue prefetches because those need no queries at all — the
    catalogue IS the venue, and the settlement join keys on domains.
    """
    completion = await _invoke_market_llm(
        MARKET_QUERY_AUTHOR_LLM_CONFIG,
        build_query_author_prompt(title, resolution_criteria),
        wall_timeout=MARKET_QUERY_AUTHOR_WALL_TIMEOUT,
        backoffs=MARKET_QUERY_AUTHOR_BACKOFFS,
        label="market_query_author",
    )
    extra = parse_query_author(completion)
    if not extra:
        return (), "error(unusable)"
    return extra, f"ok({len(extra)})"


# ---------------------------------------------------------------------------
# Stage 2 — venue-native search
# ---------------------------------------------------------------------------


async def _search_venue(session: Any, venue: str, queries: list[str]) -> list[list[MarketMatch] | None | BaseException]:
    """Every query against one venue's own index, in parallel, results in query order.

    The per-result `None`-vs-`[]` contract is preserved all the way to pool assembly: `None`
    means the fetch failed, `[]` means it parsed to nothing. That is what makes one venue's 403
    on one long query degrade THAT QUERY and nothing else — not the venue's other queries, and
    not the question. A raised query stays in the list for the same reason — `flatten_results`
    counts it as one lost sub-query rather than letting it take the venue down.
    """
    fetcher = venues.polymarket_search if venue == "polymarket" else venues.manifold_search
    width = generation.RETRIEVAL_WIDTH[venue]
    return list(
        await asyncio.gather(*(fetcher(session, query, width=width) for query in queries), return_exceptions=True)
    )


# ---------------------------------------------------------------------------
# Stage 4 — ranking
# ---------------------------------------------------------------------------


async def _rank_pool(question: Any, pool: generation.PoolResult) -> tuple[list[MarketMatch], str, str, int]:
    """One LLM call over the whole pool. Returns ``(rows, token, outcome, prompt_chars)``.

    Fails open to the pool-order top rows, which is literally the head of what the model was
    shown — so a fail-open is a truncation of the ranker's own input rather than a different
    pipeline. `outcome` is `ranked` / `failopen` / `empty`, for the telemetry line.

    An EMPTY ARRAY from the model is a valid answer, not a failure: width is the model's choice
    in 0..8, it took 3 and 6 rows on the two measured true negatives against a fixed-8 arm's
    12, and conflating `[]` with unusable output would delete the whole adaptive-width
    mechanism. Only output that cannot be read as a JSON array at all fails open.
    """
    if not pool.candidates:
        # Nothing to rank, so the stage does not run and there is no LLM call to lose. A
        # non-loss token: an empty pool is the venues' story to tell (their own tokens, and
        # Signal C for the catalogues), not a ranking failure.
        return [], "none", "empty", 0

    prompt = ranking.build_ranker_prompt(
        ranking.RankerQuestion(
            title=getattr(question, "title", None) or getattr(question, "question_text", "") or "",
            qtype=type(question).__name__,
            unit=str(getattr(question, "unit_of_measure", "") or ""),
            resolution_criteria=getattr(question, "resolution_criteria", "") or "",
            fine_print=getattr(question, "fine_print", "") or "",
        ),
        pool.candidates,
    )
    completion = await _invoke_market_llm(
        MARKET_RANKER_LLM_CONFIG,
        prompt,
        wall_timeout=MARKET_RANKER_WALL_TIMEOUT,
        backoffs=MARKET_RANKER_BACKOFFS,
        label="market_ranker",
    )
    try:
        picks = ranking.parse_ranking(completion, len(pool.candidates))
    except ranking.RankingUnusable as exc:
        logger.warning(f"Market ranking unusable ({exc}); falling back to retrieval order")
        return ranking.fail_open_slate(pool.candidates), f"error({type(exc).__name__})", "failopen", len(prompt)
    return ranking.apply_picks(pool.candidates, picks), f"ok({len(picks)})", "ranked", len(prompt)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _as_of_cache_key(as_of: datetime | None) -> str:
    """Render `as_of` as a cache-key string, normalizing through the shared `_as_utc`.

    Two callers passing the same instant spelled differently (naive-UTC vs `+00:00`) must
    collide on one key, so the normalization has to match the one `assemble_pool` applies
    to the same value — hence both going through `time_utils._as_utc` rather than each
    respelling the naive-vs-aware branch.
    """
    if as_of is None:
        return "none"
    return _as_utc(as_of).isoformat()


async def fetch_market_snapshot(
    question: Any,
    *,
    platforms: tuple[str, ...] = ("polymarket", "kalshi", "manifold", "predictit"),
    timeout: float = PREDICTION_MARKET_TIMEOUT,  # noqa: ASYNC109
    as_of: datetime | None = None,
) -> MarketSnapshot:
    """Run the four-stage pipeline and return the ranked snapshot.

    Soft-fails on any error: returns an empty `MarketSnapshot` + a WARNING. A broken
    prediction-market API should never break a forecast run.

    The `timeout` default is the same constant the provider path passes, because any lower
    default is one the pipeline cannot succeed under — stage 1a alone outruns the 5.0s this
    carried while `SNAPSHOT_STAGE_BUDGET_S` grew past 130s, so a direct caller (backtest,
    replay tool) omitting the argument got a guaranteed `error(timeout)` and a source-loss bump.

    `as_of` (backtest leakage defence) drops candidates whose `close_time` is at or before it.
    The provider path passes None — see the module docstring for why that filter was actively
    destroying the evidence class this pipeline exists to surface — and explicit callers that
    genuinely need leakage defence pass their own instant.
    """
    qid = getattr(question, "id_of_question", None)
    cache_key = (qid, _as_of_cache_key(as_of)) if qid is not None else None
    if cache_key is not None:
        cached_snap = _SNAPSHOT_CACHE.get(cache_key)
        if cached_snap is not None:
            return cached_snap  # noqa: ASYNC910

    # Session lifecycle: create the aiohttp session at the orchestrator level so cleanup
    # happens OUTSIDE the wait_for cancellation boundary. wait_for kills inner work cleanly,
    # then the surrounding context manager runs session.close().
    session_cm = _get_session()
    try:
        async with session_cm as session:
            try:
                snapshot = await asyncio.wait_for(
                    _fetch_market_snapshot_impl(question, session=session, platforms=platforms, as_of=as_of),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Prediction-market snapshot TIMEOUT after {timeout}s for qid={qid}")
                # A whole-provider failure needs its own loss token: with an empty `sources`
                # map the diagnostics line renders no suffix at all, so a dead snapshot is
                # indistinguishable from one that was never asked for. It also counts toward
                # alertable_count — losing every source at once is strictly worse than losing
                # one, which already alerts.
                _bump_source_loss()
                return MarketSnapshot(matches=[], sources={"snapshot": "error(timeout)"})  # noqa: ASYNC910
    except Exception as e:  # HARNESS-SCAN-EXEMPT-broad-except
        # Outer safety net; should not normally fire -- investigate if seen. Re-raising after
        # logging would defeat the soft-fail contract the rest of the bot depends on, so we
        # swallow + log here. Inner narrow handlers in the venue helpers cover the common paths.
        logger.warning("Prediction-market snapshot FAILED (soft-fail returning empty)", exc_info=True)
        _bump_source_loss()
        return MarketSnapshot(matches=[], sources={"snapshot": f"error({type(e).__name__})"})  # noqa: ASYNC910

    if cache_key is not None:
        _SNAPSHOT_CACHE[cache_key] = snapshot
    return snapshot


async def _fetch_market_snapshot_impl(
    question: Any,
    *,
    session: aiohttp.ClientSession,
    platforms: tuple[str, ...],
    as_of: datetime | None,
) -> MarketSnapshot:
    """The four stages. See the module docstring for the shape and why it is that shape."""
    # Every provider-health observation is keyed on the question, so a question with no id
    # records nothing (matching record_provider_detail / record_raw_research).
    qid: int | None = getattr(question, "id_of_question", None)
    title = getattr(question, "title", None) or getattr(question, "question_text", "") or ""
    resolution_criteria = getattr(question, "resolution_criteria", "") or ""
    fine_print = getattr(question, "fine_print", "") or ""

    # Per-source outcome tokens for the provider-diagnostics line: the four venues plus
    # `manifold_detail`, `query_author` and `ranking`. Single-threaded asyncio => distinct
    # keys, no race.
    sources: dict[str, str] = {}
    lost_sources: list[str] = []

    def _record_loss(name: str, token: str) -> None:
        sources[name] = token
        lost_sources.append(f"{name}={token}")
        _bump_source_loss()

    # --- stage 1a PREFETCH ‖ stage 1b QUERY AUTHOR (concurrent) ------------
    stage_one: dict[str, asyncio.Task[Any]] = {
        "query_author": asyncio.create_task(_authored_queries(title, resolution_criteria))
    }
    if "kalshi" in platforms:
        stage_one["kalshi"] = asyncio.create_task(_kalshi_catalogue(session, qid=qid))
    if "predictit" in platforms:
        stage_one["predictit"] = asyncio.create_task(_predictit_universe(session, qid=qid))
    stage_one_results = dict(
        zip(stage_one, await asyncio.gather(*stage_one.values(), return_exceptions=True), strict=True)
    )

    def _stage_one(name: str, default: Any) -> Any:
        """One stage-one outcome, converting a residual raised error into a lost source.

        The helpers each soft-fail on the errors their transports raise, so anything landing
        here is a residual (a transport error escaping a stub, a bug). Reporting it as this
        source's loss rather than letting it kill the snapshot matches what the old per-platform
        narrow catch did, and keeps one broken venue from silencing the other three.
        """
        outcome = stage_one_results.get(name, default)
        if isinstance(outcome, BaseException):
            logger.warning(f"Market stage {name} raised (soft-fail): {type(outcome).__name__}: {outcome}")
            _record_loss(name, f"error({type(outcome).__name__})")
            return default
        return outcome

    extra_queries, author_token = _stage_one("query_author", ((), "error(unusable)"))
    kalshi_events, kalshi_token = _stage_one("kalshi", ([], "none"))
    predictit_markets, predictit_tally = _stage_one("predictit", ([], _FetchTally()))

    def _report(name: str, token: str) -> None:
        """Record one source's token, routing a loss through the counter.

        Skips a source `_stage_one` already reported: a stage that RAISED is one loss, and
        recording its default token on top would either overwrite the diagnosis or bump twice.
        """
        if name in sources:
            return
        if is_lost_source(token):
            _record_loss(name, token)
        else:
            sources[name] = token

    _report("query_author", author_token)
    if "kalshi" in platforms:
        _report("kalshi", kalshi_token)

    # --- stage 2 VENUE SEARCH ---------------------------------------------
    # The enumerable venues score against the RAW query set (a year is real signal against a
    # catalogue of dated market titles); the conjunctive venues get every query stripped of
    # digit-bearing tokens, because Manifold's `term` is a strict conjunction and one date
    # token no market's text carries returns [].
    all_queries = dedupe_queries([*deterministic_queries(title), *extra_queries])
    conjunctive_queries = dedupe_queries([strip_dates_and_numbers(query) for query in all_queries])

    search_tasks = {
        venue: asyncio.create_task(_search_venue(session, venue, conjunctive_queries))
        for venue in _SEARCH_VENUES
        if venue in platforms
    }
    search_outcomes = dict(
        zip(search_tasks, await asyncio.gather(*search_tasks.values(), return_exceptions=True), strict=True)
    )
    venue_search_results: dict[str, list[list[MarketMatch] | None | BaseException]] = {}
    for venue, outcome in search_outcomes.items():
        if isinstance(outcome, BaseException):
            logger.warning(f"Venue {venue} search raised (soft-fail): {type(outcome).__name__}: {outcome}")
            _record_loss(venue, f"error({type(outcome).__name__})")
            continue
        venue_search_results[venue] = outcome

    # --- stage 3 POOL ASSEMBLY (CPU-bound, off the event loop) -------------
    # `resolution_criteria` + `fine_print` together, because the fine print is where a question
    # often names the actual release page the settlement join keys on.
    pool = await generation.build_pool(
        criteria_text=f"{resolution_criteria}\n{fine_print}",
        queries=all_queries,
        kalshi_events=kalshi_events,
        predictit_markets=predictit_markets,
        venue_search_results=venue_search_results,
        as_of=as_of,
    )
    pool_by_venue: dict[str, list[MarketMatch]] = {}
    for row in pool.candidates:
        pool_by_venue.setdefault(row.platform, []).append(row)

    for venue in _SEARCH_VENUES:
        if venue in venue_search_results:
            tally = pool.per_venue_tally.get(venue, _FetchTally())
            _report(venue, _platform_source_token(pool_by_venue.get(venue, []), tally))
    if "predictit" in platforms:
        _report("predictit", _platform_source_token(pool_by_venue.get("predictit", []), predictit_tally))

    # --- stage 2.5 ENRICH (between assembly and ranking) -------------------
    # Manifold's search listing carries no description, so without this every Manifold
    # candidate reaches the ranker title-only — and the prompt's stated "single most reliable
    # cue" is the settlement/rules text. It mutates the pool rows in place, so it MUST run
    # before the prompt is built and before apply_picks copies them.
    enrichment = await generation.enrich_manifold(pool.candidates, session)
    if enrichment.n_attempted == 0:
        sources["manifold_detail"] = "none"
    elif enrichment.n_ok == 0:
        # A lost detail GET costs rules text, never recall, so only a TOTAL loss is reported —
        # a partial fan-out has nothing actionable to alert on.
        _record_loss("manifold_detail", "error(all_details_failed)")
    else:
        sources["manifold_detail"] = f"ok({enrichment.n_ok})"

    # --- stage 4 RANK ------------------------------------------------------
    ranked_rows, ranking_token, outcome, prompt_chars = await _rank_pool(question, pool)
    _report("ranking", ranking_token)

    if lost_sources:
        # One WARN naming every degraded source: this counter reddens CI, and a red run whose
        # cause isn't named in the log is what teaches people to ignore alerts.
        logger.warning(f"Prediction-market sources degraded (alertable): {', '.join(lost_sources)}")

    _log_ranking_telemetry(qid, pool, ranked_rows, outcome=outcome, prompt_chars=prompt_chars)
    _record_venue_health(qid, pool, ranked_rows, pool_by_venue=pool_by_venue, sources=sources, platforms=platforms)
    return MarketSnapshot(matches=ranked_rows, sources=sources)


def _log_ranking_telemetry(
    qid: int | None,
    pool: generation.PoolResult,
    ranked_rows: list[MarketMatch],
    *,
    outcome: str,
    prompt_chars: int,
) -> None:
    """One INFO line per question, carrying every ranked row's `(venue, pool_index, rank)`.

    The pool index is the point: it is the post-ship instrument for the two questions this port
    deliberately left open. Whether the ranker's attention decays down a ~400-candidate prompt
    (the measured pick-index distribution says it does NOT, which is why the input stays
    venue-grouped rather than interleaved), and whether Manifold detail enrichment changes
    which rows get picked. Both answer themselves from prod logs instead of another bake-off.
    """
    rendered = ",".join(f"{row.platform}:{index}@{row.rank}" for index, row in _pool_positions(pool, ranked_rows))
    logger.info(
        f"MARKET_RANKING: question={qid} pool={len(pool.candidates)} outcome={outcome} "
        f"rows={len(ranked_rows)} prompt_chars={prompt_chars} rendered={rendered or 'none'}"
    )


def _pool_positions(pool: generation.PoolResult, ranked_rows: list[MarketMatch]) -> list[tuple[int, MarketMatch]]:
    """Each rendered row paired with the pool index the ranker referred to it by.

    `apply_picks` returns copies stamped with a rank, not with the index they came from, so the
    index is recovered by identity of the venue-native id — which is exactly the pool's own
    dedup key, so it is unique within the pool by construction.
    """
    index_of = {
        (row.platform, row.venue_market_id or row.market_title): index for index, row in enumerate(pool.candidates)
    }
    return [(index_of.get((row.platform, row.venue_market_id or row.market_title), -1), row) for row in ranked_rows]


def _record_venue_health(
    qid: int | None,
    pool: generation.PoolResult,
    ranked_rows: list[MarketMatch],
    *,
    pool_by_venue: dict[str, list[MarketMatch]],
    sources: dict[str, str],
    platforms: tuple[str, ...],
) -> None:
    """Provider-health observations, recorded where the pool and the render are both in scope.

    Field presence AND `candidates_pre_filter` are both measured over the venue's POOL rows,
    which is the load-bearing choice and has to stay a matched pair: Signal A exists to catch a
    PARSER whose field names went stale, so a legitimate 6-row ranked render that happens to
    exclude Polymarket must not record a 100%-dead `open_interest` and redden CI on the ranker's
    judgment — and equally must not go UNREPORTED because the ranker declined that venue.
    `rows_post_filter` is the RENDERED count, recorded for the archive rather than for a rule;
    nothing alerts on it (see `VenueObservation`).

    `record_catalogue_size` (Signal C) is the two enumerable venues' only alarm for an EMPTY
    catalogue, since a healthy one always reaches the pool — which is why the catalogue
    observations in stage 1a must survive any future refactor of this function.

    Pure module-state writes: no I/O, no await, cannot raise, cannot alter the snapshot.
    """
    if qid is None:
        return
    rendered_per_venue: dict[str, int] = {}
    for row in ranked_rows:
        rendered_per_venue[row.platform] = rendered_per_venue.get(row.platform, 0) + 1

    for venue in generation.VENUE_ORDER:
        if venue not in platforms:
            continue
        # A venue that lost a sub-fetch is already alertable via _bump_source_loss, so
        # provider_health skips it rather than counting one outage twice (and "check the query
        # construction" would be the wrong remedy for a 503 anyway).
        if is_lost_source(sources.get(venue, "")):
            continue
        rows = pool_by_venue.get(venue, [])
        present = {
            field_name
            for field_name in VENUE_EXPECTED_LIQUIDITY_FIELDS.get(venue, ())
            if any(getattr(row, field_name) is not None for row in rows)
        }
        record_venue_observation(
            VenueObservation(
                qid=qid,
                venue=venue,
                candidates_pre_filter=pool.per_venue_counts.get(venue, 0),
                rows_post_filter=rendered_per_venue.get(venue, 0),
                liquidity_fields_present=frozenset(present),
            )
        )


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def format_snapshot_for_research(snapshot: MarketSnapshot) -> str:
    """The markdown block for the research bundle, or `""` when there is nothing to show.

    A thin delegate to `rendering.render_snapshot`; the degraded-ranking marker is derived from
    the snapshot's own `ranking` source token rather than passed in, so the render is
    reproducible from an archived snapshot alone.
    """
    return rendering.render_snapshot(snapshot, ranking_degraded=is_lost_source(snapshot.sources.get("ranking", "")))


# ---------------------------------------------------------------------------
# ResearchCallable factory (plugged into _select_research_providers)
# ---------------------------------------------------------------------------


def prediction_market_provider(is_benchmarking: bool = False) -> ResearchCallable:
    """Factory returning an async research callable for prediction-market data.

    The returned callable accepts a `MetaculusQuestion` and uses its full API: `id_of_question`
    for caching, `title` / `resolution_criteria` / `fine_print` for the query author, the
    settlement-source join and the ranker prompt.

    Gated on the PREDICTION_MARKETS_ENABLED env flag; disabled returns "".

    When ``is_benchmarking=True`` the provider hard-disables regardless of the env flag. There
    is no orchestrator-level backstop, so this check IS the backtest defence: markets retain
    their last-trade price after resolution, and the ``as_of`` filter alone was never
    sufficient (a market that closes between ``as_of`` and now still leaks). See CLAUDE.md and
    the ``gemini_search_provider`` / ``native_search_provider`` precedents.
    """
    if PREDICTION_MARKET_TIMEOUT < SNAPSHOT_STAGE_BUDGET_S:
        # A stale `PREDICTION_MARKET_TIMEOUT=30` left in someone's .env would otherwise surface
        # only as a generic snapshot timeout on every question, with the real cause invisible.
        logger.warning(
            f"PREDICTION_MARKET_TIMEOUT={PREDICTION_MARKET_TIMEOUT}s is BELOW the pipeline's worst-case "
            f"stage sum of {SNAPSHOT_STAGE_BUDGET_S}s — snapshots will time out under load. Raise the env "
            f"override (or unset it to take the default)."
        )

    async def _fetch(question: MetaculusQuestion) -> str:
        if is_benchmarking:
            return ""  # noqa: ASYNC910
        if not env_flag_enabled(PREDICTION_MARKETS_ENABLED_ENV):
            return ""  # noqa: ASYNC910

        snapshot = await fetch_market_snapshot(question, as_of=None, timeout=PREDICTION_MARKET_TIMEOUT)
        # Surface per-source outcomes so the orchestrator's diagnostics line shows partial
        # degradation (a live venue while a sub-source silently died). Recorded here at the
        # ResearchCallable boundary so it's keyed to the qid the orchestrator pops; no-op when
        # qid is None.
        record_provider_detail(
            getattr(question, "id_of_question", None),
            "prediction_market",
            {"sources": snapshot.sources},
        )
        record_raw_research(
            qid=getattr(question, "id_of_question", None),
            provider="prediction_market",
            payload=snapshot,
        )
        return format_snapshot_for_research(snapshot)

    return _fetch
