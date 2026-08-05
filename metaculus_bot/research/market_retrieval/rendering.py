"""The rendered snapshot: a markdown table in the ranker's order, plus the rules bullets.

Three things here are contracts rather than formatting choices:

- **Zero rendered rows returns ``""`` before any preamble is emitted.** That early return is
  what produces ``status="empty"`` downstream and the attempted-vs-succeeded distinction
  residual analysis reads off the archive. Under ranked selection a zero-row render is a
  legitimate outcome (the model is allowed to say nothing bears on the question), so this is
  a hot path, not a theoretical one.
- **The rows are rendered in the ranker's order, verbatim.** No venue interleave, no
  fairness pass, no per-venue cap, no re-scoring. Round-robin venue fairness is exactly what
  lost 43 of 58 wanted rows in the measurement this port is built on.
- **The rules-bullet section keeps its byte shape** (``- **{platform}** <{url}>: {rules}`` at
  h3). The MC per-model option parser scans for ``- <name>: NN%`` and the bullets sit close
  enough to a per-model region for that to matter, so nothing percentage-shaped may ever be
  APPENDED to a bullet.

The preamble/legend strings mirror the ones the provider ships today, minus the fuzzy-match
vocabulary the ranker replaces. Their asserted substrings are load-bearing: the ranked design
changes how a row is chosen, not what a forecaster is told to verify.
"""

from __future__ import annotations

from typing import Sequence

from metaculus_bot.research.market_retrieval.ranking import (
    DEGRADED_RANKING_MARKER,
    STRONG_TIERS,
    TIER_UNSPECIFIED,
    WHY_CHARS,
)
from metaculus_bot.research.market_retrieval.types import MarketMatch, MarketSnapshot, _liquidity_label

# Raw-rules truncation in the bullet section. Separate from, and smaller than, the ranker
# prompt's per-venue caps: this text ships to the forecaster inside a research section that
# has its own character budget, and the two must not be collapsed into one constant.
RAW_RULES_MAX_CHARS = 200

TABLE_COLUMNS: tuple[str, ...] = (
    "platform",
    "title",
    "prob",
    "total_vol",
    "OI",
    "signal",
    "close",
    "status",
    "relation",
    "why",
)

TITLE_MAX_CHARS = 80

# Shared trailing legend, shipped INSIDE the research section. The liquidity half is unchanged;
# the retired `relevance` column's content-overlap explanation is replaced by ONE sentence per
# new axis — the ranker's relation + why, and what RESOLVED means about a price. This text is
# fixed overhead on every rendered snapshot and the snapshot goes to the expensive forecaster
# models, so it earns its length: every label a cell can hold is named (a legend that omits one
# teaches forecasters to guess at it) and nothing else is.
MARKET_SIGNAL_LEGEND = (
    "The `signal` column labels each market's liquidity/participation "
    "(thin/decent/deep for real-money venues, thin/decent/high for Manifold's play-money bettor count); "
    "`total_vol` and `OI` are that market's traded volume and open interest in approximate USD on the "
    "real-money venues, and play-money mana on Manifold. "
    "`no-liquidity-data` means the venue publishes no volume figures at all (PredictIt) — it says nothing "
    "about how liquid the market is, so treat it as unknown rather than as thin. Treat "
    "deep/high-liquidity markets as a strong anchor and discount thin ones (low volume, few participants) as noisy. "
    "Rows are ordered by EVIDENTIAL VALUE, best first, and `relation` grades how each bears on THIS question — "
    "`same_quantity_same_date`, then `same_quantity_other_cut`, then `driver_or_consequence`, then `weak` "
    "(`unspecified` if ungraded); only the first two measure the quantity asked about, and `why` is the "
    "one-phrase reason. `status` is `open` or `RESOLVED`; a RESOLVED price is a realized outcome, not a forecast."
)

# Strong-evidence framing — used when at least one rendered row measures the same quantity
# (relation tier 1 or 2). "MAY be relevant" and "verify each market's resolution criteria" are
# asserted verbatim elsewhere; the old "the match below is fuzzy" clause is gone, because the
# rows are now selected by a model reading each market's rules rather than by word overlap. The
# old "a poorly-matched market may be worth little or nothing" clause is gone too — the
# per-row `relation` grade now says that per row, and saying it twice is the verbosity the
# forecaster-facing budget exists to stop.
MARKET_PREAMBLE_STRONG = (
    "The following prediction markets MAY be relevant — each was selected and ranked for THIS question, so "
    "verify each market's resolution criteria, resolution date, and topic against THIS question before "
    "weighting. A market whose criteria and date match this question is extremely strong evidence — anchor on "
    "its price; on a related but different event, date, or threshold, name the specific mismatch and discount "
    "accordingly. "
)

# Neutral framing — used when NO rendered row measures the same quantity, so the table is
# context at best. "may all be off-topic" is asserted verbatim elsewhere.
MARKET_PREAMBLE_NEUTRAL = (
    "The following prediction markets were retrieved for this question and may all be off-topic — none was "
    "judged to measure the quantity THIS question asks about, so treat them as leads to verify, not as "
    "evidence. Weight a market only after you confirm its resolution criteria, date, and topic match this "
    "question; otherwise disregard it. "
)


def _cell(text: str, *, limit: int | None = None) -> str:
    """One table cell: pipes neutralised, newlines flattened, optionally truncated.

    A raw `|` from a market title or a model-authored phrase would split the row into extra
    columns and silently shift every later cell — which is why the rendering tests select
    cells BY HEADER NAME rather than by index.
    """
    cleaned = (text or "").replace("|", "/").replace("\n", " ").strip()
    return cleaned if limit is None else cleaned[:limit]


def _row_cells(match: MarketMatch) -> dict[str, str]:
    return {
        "platform": match.platform,
        "title": _cell(match.market_title, limit=TITLE_MAX_CHARS),
        "prob": f"{match.implied_prob_yes:.2f}" if match.implied_prob_yes is not None else "-",
        "total_vol": f"{match.total_volume:.0f}" if match.total_volume is not None else "-",
        "OI": f"{match.open_interest:.0f}" if match.open_interest is not None else "-",
        "signal": _liquidity_label(match),
        "close": match.close_time.strftime("%Y-%m-%d") if match.close_time else "-",
        "status": "RESOLVED" if match.is_resolved else "open",
        "relation": _cell(match.relation_tier) or TIER_UNSPECIFIED,
        "why": _cell(match.relevance_label, limit=WHY_CHARS) or "-",
    }


def _preamble(matches: Sequence[MarketMatch]) -> str:
    """Strong when any rendered row measures the same quantity, neutral otherwise.

    The selector is the ranker's own top tier, replacing the content-overlap + confidence bar
    it retired. A fail-open slate carries no tier on any row, so it selects neutral — which
    is right: an unranked slate must never get an authoritative header.
    """
    return MARKET_PREAMBLE_STRONG if any(m.relation_tier in STRONG_TIERS for m in matches) else MARKET_PREAMBLE_NEUTRAL


def render_snapshot(snapshot: MarketSnapshot, *, ranking_degraded: bool = False) -> str:
    """The markdown block for the research bundle, or ``""`` when there is nothing to show.

    ``ranking_degraded`` prefixes the degraded marker, because the preamble and legend both
    tell the forecaster the rows are in evidential order — which is false for a fail-open
    slate, and a silently-wrong ordering claim is worse than a visibly degraded one.
    """
    matches = snapshot.matches
    if not matches:
        return ""

    lines: list[str] = []
    if ranking_degraded:
        lines.append(DEGRADED_RANKING_MARKER)
        lines.append("")
    lines.append(_preamble(matches) + MARKET_SIGNAL_LEGEND)
    lines.append("")
    lines.append("| " + " | ".join(TABLE_COLUMNS) + " |")
    lines.append("|" + "---|" * len(TABLE_COLUMNS))
    for match in matches:
        cells = _row_cells(match)
        lines.append("| " + " | ".join(cells[column] for column in TABLE_COLUMNS) + " |")

    lines.append("")
    lines.append("### Resolution criteria / rules")
    for match in matches:
        rules = (match.raw_rules or "").strip().replace("\n", " ")
        if len(rules) > RAW_RULES_MAX_CHARS:
            rules = rules[:RAW_RULES_MAX_CHARS] + "..."
        link = f" <{match.market_url}>" if match.market_url else ""
        lines.append(f"- **{match.platform}**{link}: {rules}")

    return "\n".join(lines)
