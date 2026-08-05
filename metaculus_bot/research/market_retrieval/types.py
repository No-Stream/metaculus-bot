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
    # A multi-outcome market's leading `(answer_text, probability)` pairs, and the ONLY price
    # information such a row has: `implied_prob_yes` is null on every non-BINARY Manifold market,
    # so the `prob` column renders `-` and this field is what stands in for it. Empty on every
    # BINARY row and on a row whose detail GET was lost, which is also what makes it the
    # multi-outcome discriminator downstream — nothing else on the row records the outcome type.
    # Plain pairs rather than a nested dataclass because the archive walks them into JSON arrays
    # either way and there is no venue payload here to guard against extra keys.
    top_answers: tuple[tuple[str, float], ...] = ()


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


def _liquidity_label(m: MarketMatch) -> str:
    """Label how informative a market's price is, given its liquidity/participation.

    Real-money venues (Polymarket, Kalshi) score on dollar volume / open interest;
    Manifold (play-money) scores on unique bettor count instead. A thin market is
    a noise warning: sub-$10k volume is often bot-dominated, so its price should be
    discounted relative to a deep, actively-traded market. Thresholds are tunable.
    """
    if m.platform == "predictit":
        # PredictIt exposes no volume/liquidity/OI fields in its all-markets dump.
        return "no-liquidity-data"

    if m.platform == "manifold":
        if m.num_bettors is None:
            return "no-liquidity-data"
        if m.num_bettors < MANIFOLD_THIN_BETTORS:
            return "thin"
        if m.num_bettors <= MANIFOLD_HIGH_BETTORS:
            return "decent"
        return "high"

    # Real-money venues: score on the larger of total volume and open interest.
    if m.total_volume is None and m.open_interest is None:
        return "no-liquidity-data"
    score = max(m.total_volume or 0.0, m.open_interest or 0.0)
    if score < LIQUIDITY_THIN_USD:
        return "thin"
    if score <= LIQUIDITY_DEEP_USD:
        return "decent"
    return "deep"
