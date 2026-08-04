"""The ranking stage: build the prompt, parse the ranking, fail open deterministically.

One LLM call per question over the WHOLE deduped pool, all venues at once, returning up to 8
rows in ranked order. Three measured facts shaped this, and each one is a thing NOT to
"improve":

- **Rank, don't keep/drop.** Per-venue keep-or-drop calls voted KEEP on 58 of 58
  near-identical rows and then lost 43 of them downstream, because something has to order
  the keeps and a round-robin venue-fairness pass evicted whole Kalshi-heavy clusters. There
  is no venue quota anywhere in this module.
- **Width is the model's choice in 0..8, not a quota.** It used the whole range across 18
  questions and took 3 and 6 rows on the two true negatives, against a fixed-8 arm's 12. So
  an EMPTY ARRAY IS A VALID ANSWER, and conflating it with a failure would delete the whole
  adaptive-width mechanism.
- **Nothing re-orders, re-scores or filters the model's output.** Out-of-range and duplicate
  indices drop (a hallucinated index names no real market) and the list truncates at the
  budget. That is all. Inventing an order downstream is the exact defect this port fixes,
  which makes the order-preservation test the regression guard for the whole thing.

This module makes NO LLM call: the prompt builder and the parser are pure, and the seam
module owns the one retry-wrapped invocation helper both LLM stages share. That keeps one
patch point for the tests and stops the two stages drifting on retry policy.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from typing import Sequence

from metaculus_bot.research.market_retrieval.types import MarketMatch, _liquidity_label

logger = logging.getLogger(__name__)

# MAXIMUM rows the ranker may emit. A ceiling, not a target.
RENDER_BUDGET = 8

# The tier vocabulary the prompt asks for, in value order. Recorded per row so the forecaster
# prompt can weight by it. An unrecognised tier does NOT drop the row — recall-first, and a
# typo in a label is not evidence the market is irrelevant. The model emitted nothing outside
# this vocabulary across 214 measured rows.
TIERS: tuple[str, ...] = (
    "same_quantity_same_date",
    "same_quantity_other_cut",
    "driver_or_consequence",
    "weak",
)
TIER_UNSPECIFIED = "unspecified"

# The tiers that earn the strong-evidence preamble on the rendered snapshot.
STRONG_TIERS: frozenset[str] = frozenset({"same_quantity_same_date", "same_quantity_other_cut"})

# Per-row phrase cap. The label is one glanceable phrase a forecaster prompt can weight, not
# a second rationale.
WHY_CHARS = 120

# Question-header caps in the prompt.
RC_CHARS = 1200
FP_CHARS = 600

# Kalshi lists as many as a dozen settlement sources on one event; the first few carry the
# agency name that decides tier 1 versus tier 3 and the rest is boilerplate ("official social
# media accounts of ...").
SETTLEMENT_SOURCES_RENDERED = 3
SETTLEMENT_SOURCE_CHARS = 90

# Per-venue rules-text caps in the candidate block, set at MEASURED distributions rather than
# guessed. Kalshi `rules_primary` is p50=134 / p90=174 / max=522 over the frozen universe, so
# 700 truncates nothing real and only bounds a pathological row. Polymarket `description`
# runs 919-1100, so 900 is its median and the single biggest input-token lever. Manifold's
# 300 matches what detail enrichment stores; before enrichment existed this was 0, because
# the search listing carries no description at all. PredictIt is bounded by its contract
# count instead of by characters.
RULES_CHARS: dict[str, int] = {
    "kalshi": 700,
    "polymarket": 900,
    "manifold": 300,
}

# Prefixed to the rendered snapshot when the ranking call failed and the deterministic slate
# is standing in for it. The forecaster needs to know the rows are in retrieval order, not
# evidential order, because the prompt otherwise tells them the first row is the best.
DEGRADED_RANKING_MARKER = "[ranking unavailable — showing retrieval order]"

_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z]*\s*")
_FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")


# The prompt. Two deliberate departures from the version the bake-off measured, both of them
# fixes rather than tuning:
#
# 1. The DROP block's "a different OFFICE's or a different RACE's election result, even in the
#    same state and the same cycle" example is GONE, and inverted into tier 3 as an explicit
#    KEEP. On a Florida primary question whose pool held 10 wanted markets, the ranker
#    returned an empty array in BOTH replicates — a pool full of other Florida races is
#    exactly what that clause told it to exclude. Closing it alone takes recall to 13/16 at no
#    extra cost. The different-COUNTRY and different-COMPANY examples STAY: they are what
#    keeps Mexico's unemployment rate out of an Australia question, and the arm retained 0 of
#    39 labelled no-bearing rows with them in place.
# 2. The operator's relevance rule is carried verbatim at the top, and the signals block
#    gained the RESOLVED bullet, because resolved markets now reach the ranker (the `as_of`
#    filter that used to drop them is gone).
#
# Do not touch the four tier names. Substituted with `.replace`, not `.format`, because the
# emitted-object example is literal JSON braces.
RANKER_PROMPT = """You are selecting prediction markets for a forecaster working on the Metaculus question below.

Rank the candidates by EVIDENTIAL VALUE for forecasting this question, most valuable first.

THE RELEVANCE RULE YOU ARE APPLYING, stated by the operator this list is built for:
  Recall matters far more than precision. A market measuring the same quantity on the same
  date is the most valuable evidence there is; the same quantity at a different date or a
  different threshold IS evidence; a related driver, consequence, or sibling contest IS
  evidence; only a market with no bearing at all should be excluded. Three or four wrongly
  included markets are a good trade for one wrongly excluded one.

VALUE HIERARCHY, best first. Emit each kept row's tier verbatim from this list:
  1. "same_quantity_same_date" -- measures the SAME quantity the question asks about, resolving on the SAME date or over the same window. This is the most valuable evidence there is. If one exists it belongs at rank 1.
  2. "same_quantity_other_cut" -- the same quantity for the SAME subject, cut differently: a different resolution date, a different threshold, bucketed instead of thresholded, a different statistic of the same underlying thing (the minimum where the question asks the maximum, a component where it asks the aggregate), the same quantity published by a DIFFERENT source or agency than the question names (AAA's US gas average where the question names the EIA's), or the same asset priced on a different benchmark (WTI where the question asks Brent).
  3. "driver_or_consequence" -- a driver, a consequence, a correlated proxy, or a COMPONENT or SUB-REGION of the question's own subject: one district's turnout inside the same statewide electorate, one US metro's home prices under a US national index. A SIBLING CONTEST in the same electorate and the same cycle belongs here too -- a different office or a different race in the same state and year shares the turnout, the swing and the political environment, so a forecaster can reason from it. Not the same number, but a forecaster can reason from it.
  4. "weak" -- a loose correlate a forecaster would still plausibly glance at.

EXCLUDE a candidate with no bearing at all. In practice a no-bearing candidate is one whose SUBJECT is a different thing, not the same thing measured differently. The recurring shape is an ANALOGOUS quantity for a DIFFERENT ENTITY:
  - a different COUNTRY's version of the same statistic (Mexico's or Canada's unemployment rate, when the question asks about Australia's)
  - a different COUNTRY's or foreign city's home prices, when the question asks about a US national index
  - a different COMPANY's output or production, when the question asks about this company's
Sharing the words "unemployment rate" or "home price" is not a bearing. Measuring the SAME country's, index's, asset's, or company's outcome is. Note the asymmetry that decides the near misses: a COMPONENT, SUB-REGION or SIBLING CONTEST inside the question's own subject or electorate is a KEEP (tier 3); a different jurisdiction's parallel series is not, because nothing about it feeds this question's resolution.

WIDTH IS YOURS TO CHOOSE, from 0 up to {budget} rows.
  - Return every row that carries evidence, up to {budget}. Stop there; if a question has only two markets worth reading, return two.
  - Return an EMPTY ARRAY [] if nothing here bears on the question. Some questions genuinely have no market on these exchanges, and a list of loosely-worded near-misses is worse for the forecaster than no list.
  - Lean toward INCLUDING: a wrongly included row costs the forecaster one line of reading, a wrongly excluded row is evidence they never see. The two errors are not symmetric. But do not pad -- an irrelevant row spends the forecaster's attention on nothing.
  - When you are genuinely torn about a candidate, INCLUDE it, at the bottom, tiered honestly.

THREE SIGNALS IN THE CANDIDATE BLOCK:
  - `settles via` is the market's own settlement source. When it names the same agency, index, publication, or price feed the question's resolution criteria names, that candidate is very likely tier 1 or tier 2 EVEN IF its title shares few words with the question -- this is the single most reliable cue in the block. A market that settles on the exchange's own price feed for the asset the question asks about is measuring the same quantity.
  - `liquidity` is a QUALITY signal, never a relevance signal. Between two otherwise equally relevant rows prefer the deeper one. Never rank a thin market above a more relevant one, and never exclude a relevant market for being thin.
  - `RESOLVED` means the market has already settled. Its price is a realized outcome, not a forecast. A resolved market on an adjacent cut of the same quantity is still valuable -- it tells the forecaster what actually happened -- so keep and tier it normally; just do not treat its price as a live probability.

QUESTION
title: {title}
type: {qtype}
unit: {unit}
resolution criteria: {rc}
fine print: {fp}

CANDIDATES
{n_candidates} candidates from {n_venues} venue(s), grouped by venue: {venue_summary}. The venue blocks are ordered by venue, NOT by value -- a candidate in the last block can outrank everything in the first. Read all of them.
{candidates}

For each row you keep, emit exactly one object, IN RANKED ORDER, best first:
{"i": <index>, "tier": "<one of the four tier names above>", "why": "<one phrase under 12 words, naming the relation and quoting the deciding clause, e.g. 'near-identical: same BLS U-3 series, same month'>"}

Return ONLY a JSON array of at most {budget} objects, ranked best first. No prose before or after it. An empty array is a valid and sometimes correct answer.
JSON array:"""

# Stated once in the venue-block header rather than on all ~197 of its candidate lines. The
# segment is a constant for this venue, so repeating it per line spends ~1.6k prompt tokens
# carrying zero information. The RENDERED snapshot table still shows it per row.
_PREDICTIT_LIQUIDITY_NOTE = "this venue publishes no liquidity data"


@dataclass(frozen=True, slots=True)
class RankerQuestion:
    """The question fields the prompt needs, decoupled from any question class.

    Keeps this module free of a ``forecasting_tools`` import, so the prompt can be built and
    asserted on from a plain dataclass in tests and from a ``MetaculusQuestion`` in prod.
    """

    title: str
    qtype: str = ""
    unit: str = ""
    resolution_criteria: str = ""
    fine_print: str = ""


@dataclass(frozen=True, slots=True)
class Pick:
    index: int
    tier: str
    why: str


class RankingUnusable(ValueError):
    """The ranking output could not be read as a JSON array. The one trigger for fail-open."""


def _settlement_sources_text(match: MarketMatch) -> str:
    names = [
        (source.name or source.url).strip()
        for source in match.settlement_sources[:SETTLEMENT_SOURCES_RENDERED]  # noqa: HARNESS-SCAN-EXEMPT-subsampling
    ]
    return "; ".join(name[:SETTLEMENT_SOURCE_CHARS] for name in names if name)


def render_candidate_line(index: int, match: MarketMatch) -> str:
    """One pipe-joined candidate line, segments omitted when they carry nothing.

    PredictIt lines omit the ``liquidity`` segment entirely — see ``_PREDICTIT_LIQUIDITY_NOTE``.
    """
    parts = [f"[{index}] ({match.platform}) {match.market_title}"]
    sub_title = (match.sub_title or "").strip()
    if sub_title and sub_title != match.market_title:
        parts.append(sub_title)
    if match.close_time is not None:
        parts.append(f"closes: {match.close_time.strftime('%Y-%m-%d')}")
    if match.is_resolved:
        parts.append("RESOLVED")
    if match.platform != "predictit":
        parts.append(f"liquidity: {_liquidity_label(match)}")
    settles_via = _settlement_sources_text(match)
    if settles_via:
        parts.append(f"settles via: {settles_via}")
    rules = (match.raw_rules or "").strip().replace("\n", " ")
    cap = RULES_CHARS.get(match.platform)
    if cap is not None:
        rules = rules[:cap]
    if rules:
        parts.append(f"rules: {rules}")
    return " | ".join(parts)


def _venue_blocks(pool: Sequence[MarketMatch]) -> tuple[list[str], dict[str, int]]:
    """Candidate lines grouped by venue, in the pool's OWN venue order.

    Derived from first appearance rather than from a venue constant, so the prompt's grouping
    cannot drift from the pool's order. That identity is load-bearing: the pool's head is the
    fail-open slate, so a fail-open has to be a truncation of what the model was shown rather
    than a differently-ordered list.
    """
    order: list[str] = []
    counts: dict[str, int] = {}
    for match in pool:
        if match.platform not in counts:
            order.append(match.platform)
            counts[match.platform] = 0
        counts[match.platform] += 1

    blocks: list[str] = []
    for venue in order:
        note = f"; {_PREDICTIT_LIQUIDITY_NOTE}" if venue == "predictit" else ""
        blocks.append(f"-- {venue} ({counts[venue]} candidates{note}) --")
        blocks.extend(
            render_candidate_line(index, match) for index, match in enumerate(pool) if match.platform == venue
        )
    return blocks, counts


def build_ranker_prompt(question: RankerQuestion, pool: Sequence[MarketMatch]) -> str:
    """The full ranker prompt. Candidate indices are positions in ``pool``, so the parser's
    indices and the pool's order are the same thing by construction."""
    blocks, counts = _venue_blocks(pool)
    venue_summary = ", ".join(f"{venue} {count}" for venue, count in counts.items())
    return (
        RANKER_PROMPT.replace("{budget}", str(RENDER_BUDGET))
        .replace("{title}", question.title)
        .replace("{qtype}", question.qtype or "(none)")
        .replace("{unit}", question.unit or "(none)")
        .replace("{rc}", (question.resolution_criteria or "")[:RC_CHARS])
        .replace("{fp}", (question.fine_print or "")[:FP_CHARS])
        .replace("{n_candidates}", str(len(pool)))
        .replace("{n_venues}", str(len(counts)))
        .replace("{venue_summary}", venue_summary)
        .replace("{candidates}", "\n".join(blocks))
    )


def parse_ranking(text: str, pool_size: int) -> list[Pick]:
    """``[{"i": 12, "tier": "...", "why": "..."}]`` -> picks in the MODEL's order.

    ``[]`` returns ``[]``: an empty array is a valid answer and the whole adaptive-width
    mechanism, so only output that cannot be read as a JSON array at all raises
    ``RankingUnusable`` and fails open.

    Out-of-range indices are dropped, repeats collapse to their first (best-ranked)
    occurrence, an unrecognised tier is recorded as ``unspecified`` without dropping the row,
    and the list truncates at the budget. Nothing is re-ordered.
    """
    blob = (text or "").strip()
    if not blob:
        raise RankingUnusable("empty completion")
    if blob.startswith("```"):
        blob = _FENCE_CLOSE_RE.sub("", _FENCE_OPEN_RE.sub("", blob))
    # The widest bracket pair, so a model that wraps its array in an object or narrates around
    # it is still read. The slice always starts with `[` and ends with `]`, so `json.loads`
    # either raises or returns a list — there is no third outcome to guard against.
    start, end = blob.find("["), blob.rfind("]")
    if start < 0 or end <= start:
        raise RankingUnusable(f"no JSON array found in {blob[:160]!r}")  # noqa: HARNESS-SCAN-EXEMPT-subsampling
    try:
        parsed = json.loads(blob[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RankingUnusable(f"array did not parse: {exc}") from exc

    picks: list[Pick] = []
    seen: set[int] = set()
    for entry in parsed:
        if not isinstance(entry, dict) or "i" not in entry:
            continue
        try:
            index = int(entry["i"])
        except (TypeError, ValueError):
            continue
        if not 0 <= index < pool_size or index in seen:
            continue
        seen.add(index)
        tier = str(entry.get("tier") or "").strip()
        picks.append(
            Pick(
                index=index,
                tier=tier if tier in TIERS else TIER_UNSPECIFIED,
                why=str(entry.get("why") or "")[:WHY_CHARS],
            )
        )
    return picks[:RENDER_BUDGET]


def apply_picks(pool: Sequence[MarketMatch], picks: Sequence[Pick]) -> list[MarketMatch]:
    """The picked rows, stamped with rank / tier / label, in the model's order.

    Returns copies so the pool stays pristine — provider-health reads field presence off the
    POOL rows after this runs, and a rendered row is also a pool row.
    """
    return [
        replace(pool[pick.index], rank=position, relation_tier=pick.tier, relevance_label=pick.why)
        for position, pick in enumerate(picks)
    ]


def fail_open_slate(pool: Sequence[MarketMatch]) -> list[MarketMatch]:
    """The deterministic stand-in for a failed ranking: the pool-order top rows.

    Literally the head of what the model was shown, so a fail-open is a truncation of the
    input rather than a different pipeline. Tier and label stay empty, which is what selects
    the neutral preamble and the degraded marker downstream — a fail-open must never present
    as a confident ranking.
    """
    return [replace(match, rank=position) for position, match in enumerate(pool[:RENDER_BUDGET])]
