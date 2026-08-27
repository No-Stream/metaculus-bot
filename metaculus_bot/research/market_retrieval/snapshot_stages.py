"""The per-question snapshot context, the source ledger, and every stage that makes no LLM call.

Split out of the `prediction_market` seam, which keeps the two stages that DO call an LLM (the
query author and the ranker) because their invoker is the patch point the tests reach for. What
lives here is the rest of the pipeline: venue-native search (stage 2), pool assembly plus Manifold
enrichment (stages 3 and 2.5), and the post-rank accounting that maps rendered rows back to pool
indices and records the provider-health observations.

The context and the ledger live here rather than beside the orchestrator because every stage on
both sides of the split reads them, and the ledger is the one place a lost source is counted.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import aiohttp

from metaculus_bot.research.market_retrieval import generation, venues
from metaculus_bot.research.market_retrieval.queries import dedupe_queries, strip_dates_and_numbers
from metaculus_bot.research.market_retrieval.session_state import _bump_source_loss
from metaculus_bot.research.market_retrieval.types import MarketMatch, _FetchTally
from metaculus_bot.research.provider_diagnostics import is_lost_source
from metaculus_bot.research.provider_health import (
    VENUE_EXPECTED_LIQUIDITY_FIELDS,
    VenueObservation,
    record_venue_observation,
)

logger = logging.getLogger(__name__)

# The venues whose own search index is the only way in, in the order stage 2 fans out.
_SEARCH_VENUES: tuple[str, ...] = ("polymarket", "manifold")


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


class _SourceLedger:
    """Per-source outcome tokens for the provider-diagnostics line, plus the alertable losses.

    Owns the ONE place a loss is counted, so no stage can report a source twice or
    bump the counter without naming the source in the WARN.
    """

    def __init__(self) -> None:
        # The four venues plus `manifold_detail`, `query_author` and `ranking`.
        # Single-threaded asyncio => distinct keys, no race.
        self.tokens: dict[str, str] = {}
        self.lost: list[str] = []

    def record_loss(self, name: str, token: str) -> None:
        self.tokens[name] = token
        self.lost.append(f"{name}={token}")
        _bump_source_loss()

    def report(self, name: str, token: str) -> None:
        """Record one source's token, routing a loss through the counter.

        Skips a source already reported: a stage that RAISED is one loss, and
        recording its default token on top would either overwrite the diagnosis or bump twice.
        """
        if name in self.tokens:
            return
        if is_lost_source(token):
            self.record_loss(name, token)
        else:
            self.tokens[name] = token


@dataclass(frozen=True)
class _SnapshotContext:
    """Everything the four stages read off the question, resolved once up front."""

    question: Any
    session: aiohttp.ClientSession
    platforms: tuple[str, ...]
    as_of: datetime | None
    # Every provider-health observation is keyed on the question, so a question with no id
    # records nothing (matching record_provider_detail / record_raw_research).
    qid: int | None
    title: str
    resolution_criteria: str
    fine_print: str
    ledger: _SourceLedger

    @classmethod
    def build(
        cls,
        question: Any,
        *,
        session: aiohttp.ClientSession,
        platforms: tuple[str, ...],
        as_of: datetime | None,
    ) -> _SnapshotContext:
        return cls(
            question=question,
            session=session,
            platforms=platforms,
            as_of=as_of,
            qid=getattr(question, "id_of_question", None),
            title=getattr(question, "title", None) or getattr(question, "question_text", "") or "",
            resolution_criteria=getattr(question, "resolution_criteria", "") or "",
            fine_print=getattr(question, "fine_print", "") or "",
            ledger=_SourceLedger(),
        )


@dataclass(frozen=True)
class _PrefetchResult:
    """Stage-1 output: the LLM's extra queries plus the two enumerable catalogues."""

    extra_queries: tuple[str, ...]
    kalshi_events: list[Any]
    predictit_markets: list[dict[str, Any]]
    predictit_tally: _FetchTally


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


async def _run_venue_search_stage(
    ctx: _SnapshotContext, all_queries: list[str]
) -> dict[str, list[list[MarketMatch] | None | BaseException]]:
    """Stage 2 VENUE SEARCH over the conjunctive venues, dropping any that raised.

    The enumerable venues score against the RAW query set (a year is real signal against a
    catalogue of dated market titles); the conjunctive venues get every query stripped of
    digit-bearing tokens, because Manifold's `term` is a strict conjunction and one date
    token no market's text carries returns [].
    """
    conjunctive_queries = dedupe_queries([strip_dates_and_numbers(query) for query in all_queries])
    search_tasks = {
        venue: asyncio.create_task(_search_venue(ctx.session, venue, conjunctive_queries))
        for venue in _SEARCH_VENUES
        if venue in ctx.platforms
    }
    search_outcomes = dict(
        zip(search_tasks, await asyncio.gather(*search_tasks.values(), return_exceptions=True), strict=True)
    )
    venue_search_results: dict[str, list[list[MarketMatch] | None | BaseException]] = {}
    for venue, outcome in search_outcomes.items():
        if isinstance(outcome, BaseException):
            logger.warning(f"Venue {venue} search raised (soft-fail): {type(outcome).__name__}: {outcome}")
            ctx.ledger.record_loss(venue, f"error({type(outcome).__name__})")
            continue
        venue_search_results[venue] = outcome
    return venue_search_results


# ---------------------------------------------------------------------------
# Stage 3 — pool assembly (with stage 2.5 enrichment)
# ---------------------------------------------------------------------------


async def _assemble_pool_stage(
    ctx: _SnapshotContext,
    all_queries: list[str],
    prefetch: _PrefetchResult,
    venue_search_results: dict[str, list[list[MarketMatch] | None | BaseException]],
) -> tuple[generation.PoolResult, dict[str, list[MarketMatch]]]:
    """Stage 3 POOL ASSEMBLY (CPU-bound, off the event loop) plus stage 2.5 ENRICH.

    Reports each venue's post-assembly token, then enriches Manifold: its search listing
    carries no description, so without this every Manifold candidate reaches the ranker
    title-only — and the prompt's stated "single most reliable cue" is the settlement/rules
    text. Enrichment mutates the pool rows in place, so it MUST run before the prompt is
    built and before apply_picks copies them.
    """
    # `resolution_criteria` + `fine_print` together, because the fine print is where a question
    # often names the actual release page the settlement join keys on.
    pool = await generation.build_pool(
        criteria_text=f"{ctx.resolution_criteria}\n{ctx.fine_print}",
        queries=all_queries,
        kalshi_events=prefetch.kalshi_events,
        predictit_markets=prefetch.predictit_markets,
        venue_search_results=venue_search_results,
        as_of=ctx.as_of,
    )
    pool_by_venue: dict[str, list[MarketMatch]] = {}
    for row in pool.candidates:
        pool_by_venue.setdefault(row.platform, []).append(row)

    for venue in _SEARCH_VENUES:
        if venue in venue_search_results:
            tally = pool.per_venue_tally.get(venue, _FetchTally())
            ctx.ledger.report(venue, _platform_source_token(pool_by_venue.get(venue, []), tally))
    if "predictit" in ctx.platforms:
        ctx.ledger.report(
            "predictit", _platform_source_token(pool_by_venue.get("predictit", []), prefetch.predictit_tally)
        )

    enrichment = await generation.enrich_manifold(pool.candidates, ctx.session)
    if enrichment.n_attempted == 0:
        ctx.ledger.tokens["manifold_detail"] = "none"
    elif enrichment.n_ok == 0:
        # A lost detail GET costs rules text, never recall, so only a TOTAL loss is reported —
        # a partial fan-out has nothing actionable to alert on.
        ctx.ledger.record_loss("manifold_detail", "error(all_details_failed)")
    else:
        ctx.ledger.tokens["manifold_detail"] = f"ok({enrichment.n_ok})"
    return pool, pool_by_venue


# ---------------------------------------------------------------------------
# Post-rank accounting
# ---------------------------------------------------------------------------


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
        # A venue that lost its whole fan-out is already alertable via _bump_source_loss, so
        # provider_health skips it rather than counting one outage twice (and "check the query
        # construction" would be the wrong remedy for a 503 anyway). The skip is narrowed to
        # TOTAL losses (error/timeout-class tokens, which leave no pool rows to measure): a
        # partial(ok/total) venue still produced rows, and skipping it blinded Signal A for
        # that venue on exactly the runs where one of its queries flaked — 3 of the 4 CI-red
        # source-loss events in the 2026-08-24 residual round were partials, each of which
        # suppressed the liquidity-field read over the ~59 rows that DID parse.
        token = sources.get(venue, "")
        if is_lost_source(token) and not token.startswith("partial("):
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
