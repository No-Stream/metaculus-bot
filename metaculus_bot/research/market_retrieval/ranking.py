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
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from metaculus_bot.research.market_retrieval.types import MarketMatch, _liquidity_label
from metaculus_bot.structured_output_schema import extract_json_block
from metaculus_bot.time_utils import _as_utc

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

# The tiers that earn the strong-evidence preamble on the rendered snapshot: the leading two,
# and only those, measure the quantity the question asks about — tier 1 on the same date, tier 2
# on another cut of the same subject — which is precisely what that preamble asserts. Tiers 3
# and 4 are things a forecaster reasons FROM, not the quantity itself. Sliced out of TIERS
# instead of respelled so the vocabulary cannot drift between the two; the legend's "only the
# first two measure the quantity asked about" states the same fact to the forecaster.
STRONG_TIERS: frozenset[str] = frozenset(TIERS[:2])  # HARNESS-SCAN-EXEMPT-subsampling

# How far a market's close may precede the QUESTION's own open time before its top-tier grade is
# refused deterministically. `same_quantity_same_date` asserts the market resolves on the same date
# or over the same window as the question; a market that stopped trading two months before the
# question was even askable cannot be doing that, whatever its title shares with it. The measured
# case is q45163, whose rank-0 row closed 2026-02-27 against an 08-09 forecast — five months, and it
# still rendered `status=open` because Manifold's `is_resolved` was false, so the table gave a
# forecaster no cue at all.
#
# 60 days is deliberately far past any plausible same-window market and nowhere near the offender:
# the point is to catch the EGREGIOUS case without second-guessing the ranker on a question whose
# window genuinely straddles a nearby close. The cap costs the row nothing else — it keeps its rank,
# its price and its rules text, and gains a note saying what the ranker said (`cap_stale_top_tier`).
MARKET_STALENESS_TIER_CAP_DAYS = 60

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

_DECODER = json.JSONDecoder()


# The prompt. Three deliberate departures from the version the bake-off measured, all of them
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
# 3. The signals block gained the `closes` RECENCY bullet. The rendered slate is the
#    model's order verbatim — no downstream re-sort, which is the measured design decision the
#    rendering module's docstring defends — so a within-tier recency preference has nowhere to live
#    except this prompt. Measured need: q45163's rank-0 row had closed five months before the
#    forecast and was still graded above a $907k market on the question's own quantity.
#    `cap_stale_top_tier` is the deterministic backstop for the egregious case; the bullet is what
#    handles the ordinary one, since a cap cannot reorder rows.
#
# Do not touch the four tier names. Filled by `build_ranker_prompt`, never `.format`, because the
# emitted-object example below is literal JSON braces that `.format` would read as a field name.
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

FOUR SIGNALS IN THE CANDIDATE BLOCK:
  - `settles via` is the market's own settlement source. When it names the same agency, index, publication, or price feed the question's resolution criteria names, that candidate is very likely tier 1 or tier 2 EVEN IF its title shares few words with the question -- this is the single most reliable cue in the block. A market that settles on the exchange's own price feed for the asset the question asks about is measuring the same quantity.
  - `liquidity` is a QUALITY signal, never a relevance signal. Between two otherwise equally relevant rows prefer the deeper one. Never rank a thin market above a more relevant one, and never exclude a relevant market for being thin.
  - `closes` is when the market stops trading, and it is a RECENCY signal that breaks ties WITHIN a tier: between two candidates you would grade the same, rank the one still open or most recently trading ABOVE the long-closed one, whose price is old news. It also bounds the tier itself -- a market that closed months before this question's own window is not resolving on the same date, so grade it "same_quantity_other_cut" rather than "same_quantity_same_date". Never exclude a relevant market for being closed.
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

# Every slot `RANKER_PROMPT` declares. Named EXPLICITLY rather than matched as a generic `{...}`
# because the emitted-object example two lines above the end of the template is literal JSON
# braces: a generic pattern would eat `{"i": <index>, ...}` and hand the model an example it
# cannot read as JSON. Public so the tests can parametrize over this list instead of restating it,
# which is what makes an eleventh slot covered the day it is added.
RANKER_PLACEHOLDERS: tuple[str, ...] = (
    "{budget}",
    "{title}",
    "{qtype}",
    "{unit}",
    "{rc}",
    "{fp}",
    "{n_candidates}",
    "{n_venues}",
    "{venue_summary}",
    "{candidates}",
)

# Alternation order is irrelevant here: every alternative is brace-delimited, so none can be a
# proper prefix of another and no longest-match tie-break is needed.
_RANKER_PLACEHOLDER_RE = re.compile("|".join(re.escape(name) for name in RANKER_PLACEHOLDERS))

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
    """The ranking output could not be read as a ranking. The one trigger for fail-open."""


class RankingShapeRegression(RankingUnusable):
    """A well-formed but non-empty array from which NO row could be read.

    Split out from its parent so the caller's telemetry can name which failure it hit: this one
    means the model answered in a shape this parser no longer understands (a renamed index key,
    every index hallucinated past the pool), where the parent means the output could not be read
    as an array of ranking objects at all. Both fail open to the deterministic slate — the
    difference is diagnostic, and it is the difference between "our prompt/parser contract broke"
    and "the model emitted prose".
    """


def _settlement_sources_text(match: MarketMatch) -> str:
    names = [
        (source.name or source.url).strip()
        for source in match.settlement_sources[:SETTLEMENT_SOURCES_RENDERED]  # HARNESS-SCAN-EXEMPT-subsampling
    ]
    return "; ".join(name[:SETTLEMENT_SOURCE_CHARS] for name in names if name)


def render_candidate_line(index: int, match: MarketMatch) -> str:
    """One pipe-joined candidate line, segments omitted when they carry nothing.

    PredictIt lines omit the ``liquidity`` segment entirely — see ``_PREDICTIT_LIQUIDITY_NOTE``.
    Multi-outcome Manifold rows gain an ``answers:`` segment; scalar ones gain a ``scale:`` segment;
    every other row is unchanged.

    The ``scale:`` segment names the market's BOUNDS and deliberately not its value. It is here for
    the same reason the answer LABELS are — it says what the market measures, which is a relevance
    signal ("0 to 250" on a market titled for an age confirms it trades years, and it is the only
    hint that this candidate is scalar at all rather than a binary whose price was withheld) —
    whereas the value itself is a price, and this line quotes no venue's price.
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
    # Multi-outcome rows only, and the one segment on this line that quotes a price: the line
    # carries no probability for any venue, but a multi-outcome market's ANSWERS are most of what
    # it measures, the way PredictIt's contract names are. Empty on every BINARY row.
    if match.top_answers:
        parts.append("answers: " + " | ".join(f"{text} {prob:.0%}" for text, prob in match.top_answers))
    scalar = match.scalar_estimate
    if scalar is not None and (bounds := scalar.bounds_text()):
        parts.append(f"{scalar.scale_label}: {bounds}")
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
    indices and the pool's order are the same thing by construction.

    Every slot is filled in ONE pass, for the same reason ``build_query_author_prompt`` is: a
    chain of ``.replace`` calls substitutes in a fixed order, so question text is exposed to
    every LATER call in the chain. Metaculus titles quote JSON and template syntax routinely, and
    a title containing the literal ``"{candidates}"`` had the whole candidate pool spliced into
    the question header by the last call — the model then read every market twice and graded
    relevance against a question nobody asked. Nothing downstream could catch it: ``parse_ranking``
    bounds-checks an index against ``pool_size`` and never against WHICH market it names, so the
    corrupted picks parsed as valid and published as a confident ranked snapshot. One ``re.sub``
    visits each slot exactly once and never looks at what it substituted.

    The replacement is a FUNCTION, not a string, so a backslash in venue rules text or in question
    text stays literal — ``re.sub`` expands ``\\g<0>`` and ``\\1`` in a string replacement, and
    market rules and resolution criteria are arbitrary text we do not control.
    """
    blocks, counts = _venue_blocks(pool)
    substitutions = {
        "{budget}": str(RENDER_BUDGET),
        "{title}": question.title,
        "{qtype}": question.qtype or "(none)",
        "{unit}": question.unit or "(none)",
        "{rc}": (question.resolution_criteria or "")[:RC_CHARS],
        "{fp}": (question.fine_print or "")[:FP_CHARS],
        "{n_candidates}": str(len(pool)),
        "{n_venues}": str(len(counts)),
        "{venue_summary}": ", ".join(f"{venue} {count}" for venue, count in counts.items()),
        "{candidates}": "\n".join(blocks),
    }
    return _RANKER_PLACEHOLDER_RE.sub(lambda match: substitutions[match.group(0)], RANKER_PROMPT)


def _first_usable_array(text: str) -> list[Any]:
    """The first JSON array in ``text`` that could be a ranking. Raises ``RankingUnusable``.

    ``raw_decode`` from each ``[`` rather than a widest-bracket ``find``/``rfind`` slice: the
    decoder is string-literal-aware by construction, so trailing prose containing brackets no
    longer breaks the parse. The old slice spanned from the first ``[`` to the LAST ``]``
    anywhere in the output, so ``'[{"i":1}]\\nExcluded: [3] and [7].'`` — a well-formed ranking
    followed by ordinary narration — failed outright.

    "Usable" comes in three TIERS, and the scan runs to the end of the text rather than stopping
    at the first hit:

    1. A list holding a dict with an ``"i"`` key wins outright and returns immediately — only the
       ranking rows carry that key, so such a list is unambiguously the picks array.
    2. Otherwise the FIRST list holding a dict without ``"i"`` is remembered as the fallback,
       because a renamed index key has to reach ``parse_ranking``'s shape-regression WARN rather
       than pass as a valid empty answer.
    3. Otherwise an EMPTY list — the model's valid "nothing bears on this" — is remembered, and
       returned once the scan finishes with neither of the above.

    Anything else is passed over, which is what keeps a bracket inside a narrated string from
    shadowing the real array: on ``{"note": "see [3]", "picks": [{"i": 1}]}`` the first decode
    yields ``[3]``, and the scan moves on to the array that actually holds picks.

    The tiers exist because neither an empty array nor a dict-bearing helper array is evidence
    that no better one follows, and returning either where it was found made the whole ranking
    turn on JSON key order. Both halves were measured: ``{"excluded": [], "picks": [{"i": 1}]}``
    rendered zero markets while its picks sat unread two keys later, and the dict-bearing sibling
    ``{"excluded": [{"reason": "different country"}], "picks": [{"i": 0, ...}]}`` was worse still
    — the helper array won on the old any-dict test, ``parse_ranking`` then skipped every entry
    for want of an ``"i"`` key, and the question lost its whole market snapshot as ``ranked`` with
    zero rows. The mirror-image key order parsed fine off the same content in both cases.
    Deferring the empty array rather than discarding it is equally load-bearing: raising on an
    all-empty completion would trip the fail-open and render 8 near-misses on a true negative.

    One ambiguity is unresolvable here: if a helper array's dicts ALSO carry ``"i"`` (say
    ``{"excluded": [{"i": 3}], "picks": [{"i": 1}]}``) the first one still wins, since telling
    them apart needs schema knowledge this scan does not have. Acceptable because the prompt asks
    for a BARE array, so every wrapper shape is already off-spec.
    """
    index = text.find("[")
    saw_array = False
    saw_empty_array = False
    dict_bearing_fallback: list[Any] | None = None
    while index >= 0:
        try:
            value, _ = _DECODER.raw_decode(text, index)
        except ValueError:
            # ValueError, NOT json.JSONDecodeError. CPython caps int-from-string conversion at
            # 4300 digits and raises a BARE ValueError for a longer integer literal, which
            # JSONDecodeError does not cover — so the narrower catch would let a 5000-digit
            # index escape to the snapshot-level net exactly as the OverflowError above did.
            value = None
        if isinstance(value, list):
            saw_array = True
            dicts = [entry for entry in value if isinstance(entry, dict)]
            if any("i" in entry for entry in dicts):
                return value
            if dicts and dict_bearing_fallback is None:
                dict_bearing_fallback = value
            elif not value:
                saw_empty_array = True
        index = text.find("[", index + 1)
    if dict_bearing_fallback is not None:
        return dict_bearing_fallback
    if saw_empty_array:
        return []
    if saw_array:
        raise RankingUnusable(
            f"no array of ranking objects in {text[:160]!r}"
        )  # HARNESS-SCAN-EXEMPT-subsampling  # log truncation, not data sampling
    raise RankingUnusable(
        f"no JSON array found in {text[:160]!r}"
    )  # HARNESS-SCAN-EXEMPT-subsampling  # log truncation, not data sampling


def parse_ranking(text: str, pool_size: int) -> list[Pick]:
    """``[{"i": 12, "tier": "...", "why": "..."}]`` -> picks in the MODEL's order.

    ``[]`` returns ``[]``: an EMPTY array is a valid answer and the whole adaptive-width
    mechanism, so it must never fail open — a fail-open on a true negative renders 8
    near-misses.

    A NON-EMPTY array that yields no pick is the opposite case and raises
    ``RankingShapeRegression``. Returning ``[]`` there made a broken parser contract
    indistinguishable from the model's considered "nothing bears on this": both reached the
    caller as ``ok(0)``, and since the renderer gained its deliberate-empty sentence
    (2026-08-24) that conflation put an affirmative claim in front of the forecaster —
    "prediction markets were retrieved and reviewed… none was judged to bear on it" — on a
    question whose ranking we could not read. Fail-open renders the deterministic slate under
    the ``[ranking unavailable]`` marker instead, which is what the bare-int regression
    already got.

    Out-of-range indices are dropped, repeats collapse to their first (best-ranked)
    occurrence, an unrecognised tier is recorded as ``unspecified`` without dropping the row,
    and the list truncates at the budget. Nothing is re-ordered.
    """
    blob = (text or "").strip()
    if not blob:
        raise RankingUnusable("empty completion")
    # The canonical fenced-block extractor rather than a local fence regex, so a fence the rest
    # of the bot can read is a fence this parser can read. It returns None on unfenced output,
    # where the raw text is already what the array scan wants.
    parsed = _first_usable_array(extract_json_block(blob) or blob)

    picks: list[Pick] = []
    seen: set[int] = set()
    for entry in parsed:
        if not isinstance(entry, dict) or "i" not in entry:
            continue
        try:
            index = int(entry["i"])
        except (TypeError, ValueError, OverflowError):
            # OverflowError is NOT covered by the other two: `json.loads` accepts the bare
            # literals `Infinity` / `-Infinity` / `NaN` and overflowing float literals like
            # `1e400`, and `int(inf)` raises OverflowError (an ArithmeticError). Without it the
            # exception escapes `_rank_pool`'s `except RankingUnusable` to the snapshot-level
            # net and discards the WHOLE prediction-market snapshot, where dropping this one
            # entry keeps every valid sibling and the fail-open intact. `http.safe_int` guards
            # the same hazard with `math.isfinite`.
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
    if parsed and not picks:
        # A non-empty array yielding no usable pick is a SHAPE regression (a renamed key, all
        # indices hallucinated past the pool), NOT the adaptive-width `[]`. It RAISES rather than
        # returning `[]`, because `[]` reaches the caller as `ok(0)` — which
        # `provider_diagnostics.is_lost_source` does not flag — and from there the token, the
        # `MARKET_RANKING:` line and the render were byte-identical to a genuine empty answer, so
        # the forecaster got the deliberate-empty sentence's affirmative claim over unreadable
        # output. The message carries the diagnosis because only this frame sees `parsed`.
        raise RankingShapeRegression(
            f"{len(parsed)} entries yielded no usable pick (renamed index key, or every index "
            f"outside a pool of {pool_size}); first={repr(parsed[0])[:160]}"  # HARNESS-SCAN-EXEMPT-subsampling  # log truncation, not data sampling
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


def _tier_cap_note(row: MarketMatch, question_opened: datetime) -> str:
    """The demotion note for a row the staleness threshold refuses the top tier to, else ``""``.

    Split out so ``cap_stale_top_tier`` stays one comprehension: the decision and the sentence that
    explains it are the same fact, and computing them apart is how the two drift.
    """
    if row.relation_tier != TIERS[0] or row.close_time is None:
        return ""
    stale_days = (question_opened - _as_utc(row.close_time)).days
    if stale_days <= MARKET_STALENESS_TIER_CAP_DAYS:
        return ""
    return f"stale: closed {stale_days}d before the question opened (ranker said {TIERS[0]})"


def cap_stale_top_tier(rows: Sequence[MarketMatch], *, question_open_time: datetime | None) -> list[MarketMatch]:
    """Refuse ``same_quantity_same_date`` on a row that closed long before the question opened.

    DISCLOSURE, not a drop, and the distinction is the whole design. The row keeps its rank, its
    price, its liquidity cells and its rules bullet; what changes is one rung of its relation grade
    plus a ``tier_cap_note`` saying what the ranker said, so a forecaster reading the table sees
    both the demotion and the judgment it overrode. Dropping the row instead would delete evidence
    on the recall-first side of a tradeoff this pipeline has measured (a wrongly excluded market is
    evidence the forecaster never sees), and silently rewriting the tier would hide a disagreement
    between our arithmetic and the model's reading.

    Exactly ONE rung, and only from the top tier. Tier 1 makes a checkable claim — same date or
    same window — and a market that stopped trading `MARKET_STALENESS_TIER_CAP_DAYS` before the
    question was even askable cannot satisfy it. Tier 2 claims only "the same quantity, cut
    differently", and a resolved market on an adjacent cut is legitimately valuable (it says what
    actually happened), so demoting it to `driver_or_consequence` — which the forecaster prompt
    calls context rather than an anchor — would be the wrong correction. Note the measured
    consequence of that boundary: q45163's own offender was graded tier 2, so THIS pass would not
    have touched it; the render's `(Nd ago)` disclosure on the close cell is what covers that case,
    and the prompt's recency bullet is what should have ordered it lower. Across the whole archived
    corpus the cap fires ZERO times (9 rows are graded tier 1 at all, none of them stale), so read
    it as a guard on a claim a long-closed market cannot make rather than as a measured fix.

    Keyed on the question's OPEN time rather than the forecast time deliberately. Open time is a
    property of the question, so the same market is graded the same way whenever in the window the
    bot runs — a market closing mid-window stays tier-1-eligible instead of being demoted by our
    own latency.

    No-ops when the question carries no open time (nothing to compare against) and when no row is
    graded tier 1. Returns copies, matching `apply_picks`.
    """
    if question_open_time is None:
        return list(rows)
    opened = _as_utc(question_open_time)
    return [
        replace(row, relation_tier=TIERS[1], tier_cap_note=note) if (note := _tier_cap_note(row, opened)) else row
        for row in rows
    ]


def fail_open_slate(pool: Sequence[MarketMatch]) -> list[MarketMatch]:
    """The deterministic stand-in for a failed ranking: the pool-order top rows.

    Literally the head of what the model was shown, so a fail-open is a truncation of the
    input rather than a different pipeline. Tier and label stay empty, which is what selects
    the neutral preamble and the degraded marker downstream — a fail-open must never present
    as a confident ranking.
    """
    return [replace(match, rank=position) for position, match in enumerate(pool[:RENDER_BUDGET])]
