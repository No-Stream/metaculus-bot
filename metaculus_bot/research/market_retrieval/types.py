"""Row and snapshot types for the prediction-market provider, plus the liquidity labels.

This module imports nothing else from this repo, deliberately: it sits at the bottom of the
package's dependency graph so the venue, generation, ranking and rendering modules can all
depend on the row type without a cycle back through the seam module. Everything here is
re-exported from ``metaculus_bot.research.prediction_market``, which is where the archive
serializer, ``provider_health`` and the test suite import it from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

# Liquidity / participation signal-label thresholds. Low-volume markets are
# often bot-dominated (roughly sub-$10k), so a "thin" label is a real noise
# warning, not a formality. These cutoffs are a tunable first pass, not
# calibrated values — the "thin" ceiling sits at $5k deliberately conservatively.
#
# The unit is USD, and that is load-bearing rather than incidental: Polymarket's
# `volumeNum` / `openInterest` are already USD, while Kalshi reports CONTRACT COUNTS
# (`FixedPointCount`, "market volume in contracts" — docs.kalshi.com, 2026-08-03). One
# shared threshold pair across both venues therefore requires converting Kalshi's
# counts to dollars at the point of parse; see `_kalshi_usd_liquidity`.
LIQUIDITY_THIN_USD = 5_000.0
LIQUIDITY_DEEP_USD = 50_000.0
MANIFOLD_THIN_BETTORS = 20
MANIFOLD_HIGH_BETTORS = 100


@dataclass(frozen=True, slots=True)
class SettlementSource:
    """One publisher a market settles on: the venue's own `{name, url}` pair, verbatim.

    A dataclass rather than the raw payload dict so `dataclasses.asdict` walks it into a plain
    `{"name": ..., "url": ...}` for the raw-research archive, and so the two fields cannot
    silently pick up whatever else a venue starts shipping in that block.
    """

    name: str = ""
    url: str = ""


@dataclass(frozen=True, slots=True)
class MarketChild:
    """One tradeable OUTCOME inside a multi-outcome parent, with its own correctly-matched price.

    A Polymarket event, a Kalshi strike family, a multi-outcome Manifold market and a PredictIt
    ballot are all one *market* to the ranker and several *prices* to a forecaster. The parent row
    carries no probability in every one of those cases — there is no single number that answers the
    parent's own title — so before this type existed the prices were either withheld or, on
    Polymarket, taken from ``markets[0]`` and rendered under the EVENT's title: the row read
    "How many Fed rate cuts in 2026? 0.89" where 0.89 was the "will no cuts happen" child.

    Deliberately leaner than ``MarketMatch``: a child is never ranked, never deduped and never
    joined on a settlement source, so it carries no id, url, tier or channel — the renderer fills
    the ``relation`` and ``why`` cells with a dash for exactly that reason. What it does carry is
    every cell that describes a PRICE: the number itself, the liquidity behind it, when it closes
    and whether it has already settled.

    ``num_bettors`` is the one field that is legitimately the PARENT's rather than the child's.
    Manifold scores participation on unique bettors and publishes no per-answer count, so its
    children inherit the market's — every answer really does share one bettor pool, and inheriting
    it renders each child the same honest label as its parent instead of a false
    ``no-liquidity-data`` on a venue that does publish per-answer volume.
    """

    title: str
    implied_prob_yes: float | None = None
    total_volume: float | None = None
    open_interest: float | None = None
    num_bettors: int | None = None
    is_resolved: bool = False
    close_time: datetime | None = None


@dataclass
class MarketMatch:
    """One market a forecaster might read as a peer cross-check.

    ADDITIVE-ONLY. Every field below the first twelve was added after a consumer already
    depended on the shape: `raw_log` archives the whole snapshot via `dataclasses.asdict` under
    an envelope whose `schema_version` is shared across all providers (so a removal changes the
    archive with no version to bump), `provider_health` reads the liquidity fields by `getattr`
    name, and several test sites construct the row with twelve positional arguments. Add at the
    end, always defaulted; never remove or reorder.

    `match_confidence` predates the ranked-retrieval design and no longer means "fuzzy
    relevance". It now carries the score of whichever RETRIEVAL CHANNEL found the row — 1.0 for
    a settlement-source join (a structural hit has no score to report), the inverted venue rank
    for a venue-index hit, the `fuzzy_best` score for an enumerable-universe hit — so it is not
    comparable across channels and is not a relevance signal. Relevance is `relation_tier` and
    `relevance_label`, both from the ranker.
    """

    platform: Literal["polymarket", "kalshi", "manifold", "predictit"]
    market_title: str
    market_url: str
    implied_prob_yes: float | None
    bid: float | None
    ask: float | None
    spread: float | None
    volume_24h: float | None
    close_time: datetime | None
    is_resolved: bool
    match_confidence: float
    raw_rules: str
    # Liquidity / participation fields. Previously received-but-discarded; now
    # parsed so the formatter can label how informative each crowd signal is.
    total_volume: float | None = None
    liquidity: float | None = None
    open_interest: float | None = None
    num_bettors: int | None = None
    # Ranked-retrieval fields. A venue parser fills `venue_market_id`, `sub_title` and
    # `settlement_sources`; pool assembly fills `retrieval_channel`; the ranker fills the rest.
    # The venue-native id (kalshi event_ticker, polymarket slug, manifold contract id, predictit
    # market id) is the pool's dedup key and the enrichment hook's join key — before this field
    # existed it was only recoverable by parsing `market_url`.
    venue_market_id: str = ""
    relation_tier: str = ""
    relevance_label: str = ""
    rank: int | None = None
    retrieval_channel: str = ""
    sub_title: str = ""
    settlement_sources: tuple[SettlementSource, ...] = ()
    # A multi-outcome Manifold market's leading `(answer_text, probability)` pairs, for the RANKER's
    # one-line candidate line: `implied_prob_yes` is null on every non-BINARY Manifold market, so
    # without these the candidate reaches the ranker with a title and nothing else. Empty on every
    # BINARY row and on a row whose detail GET was lost. Plain pairs rather than a nested dataclass
    # because the archive walks them into JSON arrays either way and there is no venue payload here
    # to guard against extra keys.
    #
    # Deliberately NOT what the render reads — `children` is, and it keeps the whole answer array
    # with each answer's own volume. Two reads of one payload, because a one-line prompt segment and
    # a table row want different slices; see `manifold_answer_children`.
    top_answers: tuple[tuple[str, float], ...] = ()
    # The multi-outcome expansion. A venue adapter fills this when the market has SEVERAL prices and
    # the parent title has none, and the renderer emits one indented sub-row per entry. The ranker's
    # view stays flat — a candidate line describes the parent alone and a pick costs one slot — so
    # this field is written by the venue parsers and read only by `rendering`.
    #
    # ADAPTER ORDER IS THE RENDER ORDER, verbatim, exactly as the ranker's order is for parents: the
    # renderer truncates a long list from the END, so each adapter orders by what its own venue
    # makes worth keeping (traded size on the real-money venues, probability on Manifold, ballot
    # order on PredictIt) and documents that choice. A row with children carries
    # `implied_prob_yes=None` on every venue — the invariant that makes "the parent has no single
    # probability" a fact about the data rather than a rendering convention.
    children: tuple[MarketChild, ...] = ()


@dataclass
class MarketSnapshot:
    matches: list[MarketMatch] = field(default_factory=list)
    # Per-source outcome tokens ({source_name: token}) for the provider-diagnostics line.
    # SEVEN keys on a complete run: the four venues, plus `query_author` (the additive query
    # stage), `manifold_detail` (the rules-text enrichment fan-out) and `ranking` (the one
    # selection call). Plus `snapshot` INSTEAD of all of them on the whole-provider failure
    # paths, where the snapshot timed out or blew up before any source reported.
    #
    # A token starting with "ok"/"none" is benign; anything else (e.g. "dropped(size_cap)",
    # "error(...)", "partial(1/2)") is a LOST source. `none` means the source was reached
    # successfully and had nothing to contribute — an outage must never land there, or the
    # published line reads healthy through a blackout. Two `none`s are load-bearing and worth
    # knowing: `manifold_detail: none` means there were no Manifold candidates to enrich, and
    # `ranking: none` means the pool was empty so the ranking call never ran. See
    # provider_diagnostics.
    sources: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _FetchTally:
    """How many of a platform's HTTP sub-fetches came back vs. were lost.

    Carried alongside the matches because an upstream outage and a genuine no-match
    both arrive as an empty match list; without these counts the diagnostics token
    cannot tell them apart (see `_platform_source_token`). The unit is one sub-fetch
    at each platform's natural granularity: a keyword query for Polymarket/Manifold,
    a page for the Kalshi events prefetch, the single dump for PredictIt.
    """

    ok: int = 0
    failed: int = 0


def liquidity_label_from_fields(
    platform: str,
    *,
    total_volume: float | None,
    open_interest: float | None,
    num_bettors: int | None,
) -> str:
    """Label how informative a price is, given its liquidity/participation.

    Real-money venues (Polymarket, Kalshi) score on dollar volume / open interest;
    Manifold (play-money) scores on unique bettor count instead. A thin market is
    a noise warning: sub-$10k volume is often bot-dominated, so its price should be
    discounted relative to a deep, actively-traded market. Thresholds are tunable.

    Takes loose fields rather than a row so a ``MarketChild`` sub-row is labelled by the SAME rule
    as its parent. Two labelling paths would let a Kalshi strike and its family disagree about what
    "thin" means.
    """
    if platform == "predictit":
        # PredictIt exposes no volume/liquidity/OI fields in its all-markets dump.
        return "no-liquidity-data"

    if platform == "manifold":
        if num_bettors is None:
            return "no-liquidity-data"
        if num_bettors < MANIFOLD_THIN_BETTORS:
            return "thin"
        if num_bettors <= MANIFOLD_HIGH_BETTORS:
            return "decent"
        return "high"

    # Real-money venues: score on the larger of total volume and open interest.
    if total_volume is None and open_interest is None:
        return "no-liquidity-data"
    score = max(total_volume or 0.0, open_interest or 0.0)
    if score < LIQUIDITY_THIN_USD:
        return "thin"
    if score <= LIQUIDITY_DEEP_USD:
        return "decent"
    return "deep"


def _liquidity_label(m: MarketMatch) -> str:
    """``liquidity_label_from_fields`` for a whole row. The name every consumer already imports."""
    return liquidity_label_from_fields(
        m.platform,
        total_volume=m.total_volume,
        open_interest=m.open_interest,
        num_bettors=m.num_bettors,
    )


def _child_liquidity_label(platform: str, child: MarketChild) -> str:
    """``liquidity_label_from_fields`` for one sub-row, under its parent's platform rule."""
    return liquidity_label_from_fields(
        platform,
        total_volume=child.total_volume,
        open_interest=child.open_interest,
        num_bettors=child.num_bettors,
    )
