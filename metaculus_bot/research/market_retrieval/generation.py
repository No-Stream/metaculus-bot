"""Pool assembly: three retrieval channels unioned into one ranked candidate list.

**Channel order IS the ranking.** ``add()`` keeps first-seen position, so a structural
settlement-source hit always precedes a venue-index hit, which always precedes a fuzzy fill,
with no cross-channel score comparison anywhere — which matters because the scores are not
comparable (1.0 for a structural hit, an inverted venue rank, a 0-100 fuzzy score). That
ordering is used for two things and they must stay the same one: the order candidates are
PRESENTED to the ranker in, and the deterministic fail-open slate. Presenting in the
fail-open order is what makes a fail-open a truncation of what the model was shown rather
than a different pipeline.

The design is recall-maximal on purpose. The bake-off measured that **selection, not
generation, is the binding constraint**: a perfect ranker over this pool reaches 14/16
questions while the deterministic top-4 of the same pool scores 5/16. So nothing here
filters — there is no score floor, and the retired ``KALSHI_MIN_FUZZY_SCORE`` /
``PREDICTIT_MIN_FUZZY_SCORE`` floors are absent rather than defaulted, because a floor is
what discarded the adjacent-cut markets that carry most of the evidential value.

Everything CPU-bound runs in ONE ``asyncio.to_thread`` hop, and that is not hygiene: the
full-catalogue fuzzy scan measures ~0.45s of blocking CPU (9,762 events x ~16 queries), and
``publish_hardening.py`` documents how a pinned event loop misattributes forecaster
soft-deadline drops to the forecasters. Building the settlement-domain index belongs in the
same hop — it walks the same ~10k events and is not memoized.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from metaculus_bot.research.market_retrieval import venues
from metaculus_bot.research.market_retrieval.http import flatten_results
from metaculus_bot.research.market_retrieval.queries import fuzzy_best
from metaculus_bot.research.market_retrieval.settlement_join import question_domains, settlement_domain_index
from metaculus_bot.research.market_retrieval.types import MarketMatch, _FetchTally

logger = logging.getLogger(__name__)

# Venue presentation order, and therefore the pool's flat order. Kalshi's block comes first
# because it is the widest; the ranker prompt states explicitly that the blocks are ordered
# by venue and NOT by value, and the measured pick-index distribution bears that out (pick
# rate RISES with index across kalshi -> polymarket -> manifold).
VENUE_ORDER: tuple[str, ...] = ("kalshi", "polymarket", "manifold", "predictit")

# Per-venue pool width. **PredictIt is deliberately absent**: its whole universe is 197
# markets from one GET, so there is nothing to select and a width would only throw evidence
# away — the same "cannot come back by way of a defaulted argument" logic as the deleted
# score floors. A venue missing from this dict is unbounded, which is the statement.
#
# Kalshi's 100 is what fits in a prompt out of ~9,762 open events, so something has to order
# them; that is all `fuzzy_best` is for here. The two venue-search widths are ceilings on
# the union across queries (each query's own response is capped at the endpoint's `limit`).
RETRIEVAL_WIDTH: dict[str, int] = {
    "kalshi": 100,
    "polymarket": 60,
    "manifold": 60,
}

# Manifold detail-enrichment dials. Bounded at <= 60 fetches by construction (the Manifold
# pool width), 10 at a time, and the WHOLE fan-out sits under one wall — without it the worst
# case is 6 waves x 10s = 60s, which blows the snapshot's time budget on rules text.
MANIFOLD_DETAIL_CONCURRENCY = 10
MANIFOLD_DETAIL_WALL_S = 10.0
# Matches the Manifold rules cap in the ranker prompt. Kept here as well as there because
# this is where the text is stored: a longer store would be silently truncated downstream
# and the render path has its own, separate, smaller cap.
MANIFOLD_DETAIL_RULES_CHARS = 300

CHANNEL_SETTLEMENT_JOIN = "settlement_join"
CHANNEL_VENUE_SEARCH = "venue_search"
CHANNEL_UNIVERSE_FUZZY = "universe_fuzzy"


@dataclass(frozen=True, slots=True)
class PoolResult:
    """The assembled pool plus everything the caller needs to report on it.

    ``candidates`` is flat and venue-major, so a row's list index is its ranker prompt index
    and the head of the list is the fail-open slate.

    ``per_venue_counts`` is what provider-health records as ``candidates_pre_filter``: the
    per-venue POOL count, not the rendered count. Field-presence has to be measured over the
    pool rather than over the ranked rows, or a legitimate 6-row render that happens to
    exclude Polymarket records a 100%-dead ``open_interest`` field and reddens CI on the
    ranker's judgment.
    """

    candidates: tuple[MarketMatch, ...] = ()
    per_venue_counts: dict[str, int] = field(default_factory=dict)
    per_venue_tally: dict[str, _FetchTally] = field(default_factory=dict)
    degraded_venues: tuple[str, ...] = ()
    channel_counts: dict[str, int] = field(default_factory=dict)
    settlement_domains: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EnrichmentResult:
    """How the Manifold detail fan-out went.

    ``n_ok`` counts detail GETs that came BACK, not rows whose rules text changed: a market
    with a genuinely empty description is a success. The caller reports a lost source only at
    ``n_ok == 0 and n_attempted > 0`` — a partially-lost fan-out costs no recall, only rules
    text, so bumping the degradation counter on it would alert on nothing actionable.
    """

    n_attempted: int = 0
    n_ok: int = 0


def _dedup_key(match: MarketMatch) -> tuple[str, str]:
    """``(venue, casefolded venue-native id)``, falling back to the title.

    The venue-native id is the right key — it is stable across channels, where
    ``market_url`` is not (Kalshi's join channel knows the event ticker; its URL is
    series-level and shared by several events). The title fallback only matters for a row a
    venue shipped without an id, and keying those on the title at least keeps two distinct
    ones distinct instead of collapsing every id-less row into a single slot.
    """
    return (match.platform, (match.venue_market_id or match.market_title).strip().casefold())


def _settlement_join_channel(
    criteria_text: str, queries: Sequence[str], kalshi_events: Sequence[dict[str, Any]]
) -> tuple[list[MarketMatch], tuple[str, ...]]:
    """Kalshi events settling on a publisher the question's own resolution text names.

    Reaches the markets whose titles share almost no vocabulary with the question — the
    class a fuzzy scorer structurally cannot see. Re-ranked WITHIN the channel by
    ``fuzzy_best``, because the join itself has no ranking signal and returns catalogue
    order: leaving it unranked would make the fail-open slate measure the alphabet.
    """
    domains = question_domains(criteria_text)
    if not domains or not kalshi_events:
        return [], tuple(sorted(domains))

    index = settlement_domain_index(kalshi_events)
    joined: dict[str, dict[str, Any]] = {}
    for domain in sorted(domains):
        for event in index.get(domain, ()):
            joined.setdefault(str(event.get("event_ticker") or ""), event)

    scored: list[tuple[float, MarketMatch]] = []
    for event in joined.values():
        match = venues.kalshi_event_match(event, match_confidence=1.0, channel=CHANNEL_SETTLEMENT_JOIN)
        if match is None:
            continue
        scored.append((fuzzy_best(list(queries), match.market_title, match.raw_rules), match))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [match for _, match in scored], tuple(sorted(domains))


def _kalshi_universe_channel(queries: Sequence[str], kalshi_events: Sequence[dict[str, Any]]) -> list[MarketMatch]:
    """The full Kalshi catalogue, ``fuzzy_best``-ranked. Truncation happens at the width."""
    scored: list[tuple[float, MarketMatch]] = []
    for event in kalshi_events:
        if not isinstance(event, dict):
            continue
        title = event.get("title") or event.get("sub_title") or ""
        if not title:
            continue
        score = fuzzy_best(list(queries), title, venues.kalshi_event_rules(event))
        match = venues.kalshi_event_match(event, match_confidence=score, channel=CHANNEL_UNIVERSE_FUZZY)
        if match is not None:
            scored.append((score, match))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [match for _, match in scored]


def _predictit_universe_channel(predictit_markets: Sequence[dict[str, Any]]) -> list[MarketMatch]:
    """Every PredictIt market, in dump order. No scorer and no width, deliberately.

    197 markets is a prompt block, not a corpus, so there is nothing to rank and nothing to
    cut. A consequence worth knowing: provider-health's no-contribution signal can never
    fire for PredictIt (``candidates_pre_filter`` is always 197), which leaves
    ``record_catalogue_size`` as its only health signal.
    """
    out: list[MarketMatch] = []
    for market in predictit_markets:
        if not isinstance(market, dict):
            continue
        match = venues.predictit_market_match(market, match_confidence=1.0, channel=CHANNEL_UNIVERSE_FUZZY)
        if match is not None:
            out.append(match)
    return out


def assemble_pool(
    *,
    criteria_text: str,
    queries: Sequence[str],
    kalshi_events: Sequence[dict[str, Any]],
    predictit_markets: Sequence[dict[str, Any]],
    venue_search_results: Mapping[str, Sequence[list[MarketMatch] | None | BaseException]],
) -> PoolResult:
    """The synchronous, I/O-free half of pool assembly. Call it via ``build_pool``.

    Named and public rather than a closure inside ``build_pool`` so the channel ordering, the
    dedup and the widths are testable without an event loop — and so there is exactly one
    thing to hand ``asyncio.to_thread``.

    ``criteria_text`` is the question's ``resolution_criteria`` + ``fine_print``, concatenated
    by the caller: the settlement join extracts URLs from it, and the fine print is where a
    question often names the actual release page.

    ``kalshi_events`` is the PROJECTED catalogue (``venues.kalshi_prefetch_events``), not raw
    ``/events`` payloads. Pass ``[]`` for a lost pull — the caller owns that distinction, since
    it is the one holding the ``CataloguePull`` token.

    ``venue_search_results`` is per-venue, per-query, in the leaf fetchers' ``None``-vs-``[]``
    contract: ``None`` is an upstream failure, ``[]`` a search that parsed to nothing. That
    distinction is what separates a degraded venue from a venue with nothing to say, and it
    cannot be recovered from the pool afterwards. A raised query may be passed through as the
    exception itself (what ``asyncio.gather(return_exceptions=True)`` hands back); it counts as
    one lost sub-query, same as ``None``.
    """
    join_rows, domains = _settlement_join_channel(criteria_text, queries, kalshi_events)

    search_rows: list[MarketMatch] = []
    tallies: dict[str, _FetchTally] = {}
    degraded: list[str] = []
    for venue in VENUE_ORDER:
        results = venue_search_results.get(venue)
        if results is None:
            continue
        matches, tally = flatten_results(list(results), venue)
        search_rows.extend(matches)
        tallies[venue] = tally
        if tally.failed:
            degraded.append(venue)

    universe_rows = [
        *_kalshi_universe_channel(queries, kalshi_events),
        *_predictit_universe_channel(predictit_markets),
    ]

    pools: dict[str, list[MarketMatch]] = {venue: [] for venue in VENUE_ORDER}
    seen: set[tuple[str, str]] = set()
    channel_counts: dict[str, int] = {}

    def add(match: MarketMatch) -> None:
        if match.platform not in pools:
            logger.warning(f"pool assembly: dropping row from unknown venue {match.platform!r}")
            return
        key = _dedup_key(match)
        if key in seen:
            return
        seen.add(key)
        pools[match.platform].append(match)
        channel_counts[match.retrieval_channel] = channel_counts.get(match.retrieval_channel, 0) + 1

    for match in join_rows:
        add(match)
    for match in search_rows:
        add(match)
    for match in universe_rows:
        add(match)

    candidates: list[MarketMatch] = []
    per_venue_counts: dict[str, int] = {}
    for venue in VENUE_ORDER:
        width = RETRIEVAL_WIDTH.get(venue)
        rows = pools[venue] if width is None else pools[venue][:width]
        per_venue_counts[venue] = len(rows)
        candidates.extend(rows)

    logger.info(
        f"market pool: n={len(candidates)} per_venue={per_venue_counts} channels={channel_counts} "
        f"domains={list(domains)} degraded={degraded or 'none'}"
    )
    return PoolResult(
        candidates=tuple(candidates),
        per_venue_counts=per_venue_counts,
        per_venue_tally=tallies,
        degraded_venues=tuple(degraded),
        channel_counts=channel_counts,
        settlement_domains=domains,
    )


async def build_pool(
    *,
    criteria_text: str,
    queries: Sequence[str],
    kalshi_events: Sequence[dict[str, Any]],
    predictit_markets: Sequence[dict[str, Any]],
    venue_search_results: Mapping[str, Sequence[list[MarketMatch] | None | BaseException]],
) -> PoolResult:
    """``assemble_pool`` off the event loop. One hop, so the loop is yielded once, not N times.

    See the module docstring for why the offload is load-bearing rather than tidy: the
    full-catalogue scan is ~0.45s of blocking CPU, and a pinned loop shows up as forecaster
    soft-deadline drops somewhere else entirely.
    """
    return await asyncio.to_thread(
        assemble_pool,
        criteria_text=criteria_text,
        queries=queries,
        kalshi_events=kalshi_events,
        predictit_markets=predictit_markets,
        venue_search_results=venue_search_results,
    )


async def enrich_manifold(
    candidates: Sequence[MarketMatch],
    session: Any,
    *,
    concurrency: int = MANIFOLD_DETAIL_CONCURRENCY,
    wall_s: float = MANIFOLD_DETAIL_WALL_S,
) -> EnrichmentResult:
    """Fill in Manifold rules text from each candidate's detail record, in place.

    The Manifold search listing carries no description, so without this every Manifold
    candidate reaches the ranker title-only — and the ranker's stated "single most reliable
    cue" is the settlement/rules text. Runs post-dedup, on the pool, which bounds it at the
    Manifold width by construction; running it inside the search would scale with the query
    count instead.

    Soft-fails at every level. A lost detail GET leaves that row title-only, and the whole
    fan-out sits under ``wall_s`` returning whatever completed, because rules text is worth a
    bounded wait and never worth the snapshot.
    """
    rows = [row for row in candidates if row.platform == "manifold" and row.venue_market_id]
    if not rows:
        return EnrichmentResult()  # noqa: ASYNC910

    semaphore = asyncio.Semaphore(concurrency)
    n_ok = 0

    async def _enrich_one(row: MarketMatch) -> None:
        nonlocal n_ok
        async with semaphore:
            detail = await venues.manifold_market_detail(session, row.venue_market_id)
        if detail is None:
            return
        n_ok += 1
        text = str(detail.get("textDescription") or "").strip()
        # A description that merely restates the beginning of the title is not rules text; it
        # would spend ranker tokens repeating the same line's own title back at it.
        if text and not row.market_title.strip().startswith(text):
            row.raw_rules = text[:MANIFOLD_DETAIL_RULES_CHARS]

    gathered = asyncio.gather(*(_enrich_one(row) for row in rows), return_exceptions=True)
    try:
        for outcome in await asyncio.wait_for(gathered, wall_s):
            if isinstance(outcome, BaseException):
                logger.warning(f"Manifold detail enrichment raised: {type(outcome).__name__}: {outcome}")
    except asyncio.TimeoutError:
        logger.warning(
            f"Manifold detail enrichment hit its {wall_s}s wall with {n_ok}/{len(rows)} returned; "
            f"keeping what completed"
        )
    logger.info(f"manifold detail enrichment: attempted={len(rows)} ok={n_ok}")
    return EnrichmentResult(n_attempted=len(rows), n_ok=n_ok)  # noqa: ASYNC910
