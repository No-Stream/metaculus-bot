"""The counterfactual clip-threshold sweep itself: model, math and per-window aggregation.

The pass this belongs to, and how to read its output, is documented on the CLI module
``performance_analysis.clip_threshold``; this module holds the numbers behind it. Rendering
lives in ``clip_threshold_report`` so the dependency runs one way: CLI -> report -> here.

Three rules are load-bearing throughout and are stated once here rather than at every use.

**Tightening is exact; loosening is censored.** A candidate bound at least as tight as the
one in force is fully determined by the published value, whether or not that value was
itself clamped, so ``clip_delta`` intersects the candidate with the in-force clamp before
computing anything. A candidate LOOSER than the in-force clamp cannot be priced on a record
that sits at that bound: the clamp erased the raw member value. Those records are counted
(``censored_n``) and bounded, never estimated.

**Censoring happens at the MEMBER, not the published value.** The pipeline clamps each
member and THEN aggregates, so a clamped member can sit in a median position while the
published median is above the floor (an even roster averages the two middle members:
members ``0.02 / 0.03`` publish ``0.025``). ``censored_n`` keys on the published value and
is the narrow count; ``member_censored_n`` keys on a clamped member in a median position
(any position under a mean aggregator) and is the count that actually bounds what a looser
clip could have moved. A member above the floor in a non-median position cannot move the
median however low its raw value was, which is why the member rule is exact rather than
"any member at the floor".

**Both types live in one vector shape.** ``ClipRecord.published`` is the outcome-space
probability vector, ``(p_no, p_yes)`` for binary and the option vector for MC, which is what
lets the counterfactual, the replay and the censoring rules stay single-branch. The clamp
semantics still differ (binary clamps ``p_yes`` and takes the complement; MC clamps every
option and renormalises), and ``apply_bounds`` is the one place that branches on it.

One disclosure rides beside those rules. An MC floor ``c`` cannot be DELIVERED on a ballot
with more than ``1 / c`` options (eleven options each at least 0.10 already exceed 1), and
the live clamp then returns its sub-floor fallback; such records are priced like any other
but counted in ``infeasible_n`` so a cell labelled "floor 0.10" says on how many ballots that
floor was not the floor actually applied.
"""

from __future__ import annotations

import logging
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

import numpy as np

from metaculus_bot.aggregation_strategies import aggregate_binary_median
from metaculus_bot.bootstrap import bootstrap_means
from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN, MC_PROB_MAX, MC_PROB_MIN
from metaculus_bot.mc_processing import FLOOR_FEASIBILITY_ATOL, clamp_and_renormalize_probs
from metaculus_bot.numeric.utils import aggregate_binary_mean
from metaculus_bot.performance_analysis.analysis import FT_0292_MERGED_AT, WIDENING_FLIP_MERGED_AT
from metaculus_bot.performance_analysis.parsing import _parse_probability
from metaculus_bot.performance_analysis.platform_scores import spot_peer_score
from metaculus_bot.performance_analysis.stacker_detection import base_or_per_model_forecasts, detect_stacker_fired
from metaculus_bot.scoring_common import spot_peer_delta
from metaculus_bot.time_utils import parse_iso_utc

logger: logging.Logger = logging.getLogger(__name__)

BINARY = "binary"
MULTIPLE_CHOICE = "multiple_choice"
QUESTION_TYPES: tuple[str, str] = (BINARY, MULTIPLE_CHOICE)

ClipSide = Literal["floor_only", "ceiling_only", "symmetric"]
SIDES: tuple[ClipSide, ...] = ("floor_only", "ceiling_only", "symmetric")

Aggregator = Literal["median", "mean", "unknown"]
CONFIRMED_STACKER = "confirmed_stacker"

# Candidate floors (ceiling 1 - c). Module constants so a round can widen them without
# touching logic; every c must satisfy 0 < c < 0.5 for the clamp to be a clamp.
BINARY_FLOOR_GRID: tuple[float, ...] = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10)
MC_FLOOR_GRID: tuple[float, ...] = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10)
GRID_BY_TYPE: dict[str, tuple[float, ...]] = {BINARY: BINARY_FLOOR_GRID, MULTIPLE_CHOICE: MC_FLOOR_GRID}
# At c >= 0.5 the bounds invert (lo > hi) and apply_bounds collapses every publish to 1 - c.
assert all(0.0 < c < 0.5 for grid in GRID_BY_TYPE.values() for c in grid), "a candidate floor must satisfy 0 < c < 0.5"

# ft 0.2.92's PredictedOptionList validator clamps every option into [0.01, 0.99] on
# construction, so an MC floor below MC_PROB_MIN is not shippable today whatever this sweep
# says about it. Those rows are reported and labelled rather than dropped.
MC_UNSHIPPABLE_NOTE = "not shippable under ft 0.2.92"

BOOTSTRAP_B: int = 4000
BOOTSTRAP_SEED: int = 20260902
BOOTSTRAP_CL: float = 0.95
# A record sits AT its in-force bound within this tolerance. Binary publishes are a median
# rounded to 3 dp, so a clamped one hits the floor exactly; MC options pass through a
# renormalisation that leaves 0.0101 / 0.011 where 0.01 was clamped, which is why the MC
# tolerance is coarse enough to catch that drift and nothing wider.
BINARY_CENSOR_ATOL: float = 1e-9
MC_CENSOR_ATOL: float = 0.0015

# A record counts as MOVED when its spot-peer delta clears this. MC vectors are
# renormalised, so an unaffected MC record's delta is float noise (~1e-14 points) rather
# than a hard zero; binary deltas are exactly 0 when nothing moves.
DELTA_ATOL: float = 1e-9

# The published-vector counterfactual and the per-model replay disagree when the resolving
# mass differs by more than this — half a point of probability, the resolution at which a
# disagreement could plausibly have changed a published forecast. The same tolerance decides
# whether a replayed aggregate REPRODUCES the published vector (aggregator detection).
REPLAY_DISAGREE_ATOL: float = 0.005

# Two candidates tie for the argmax within this many spot-peer points. Ties are the norm,
# not an edge case: every candidate at or below a window's in-force floor scores exactly 0
# when no publish in that window was clamped, so the winner is usually a plateau.
ARGMAX_TIE_ATOL: float = DELTA_ATOL

# The clamp in force, oldest regime first: (start_or_None, lo, hi), every row a LITERAL so a
# constant change cannot retroactively reprice the records published under the retired clamp
# (the asserts below force the APPEND). Binary [0.01, 0.99] predates the earliest archived record.
_CLAMP_HISTORY: dict[str, tuple[tuple[datetime | None, float, float], ...]] = {
    BINARY: ((None, 0.01, 0.99), (WIDENING_FLIP_MERGED_AT, 0.02, 0.98)),
    MULTIPLE_CHOICE: ((None, 0.005, 0.995), (FT_0292_MERGED_AT, 0.01, 0.99)),
}
assert _CLAMP_HISTORY[BINARY][-1][1:] == (BINARY_PROB_MIN, BINARY_PROB_MAX), (
    "BINARY_PROB_MIN/MAX moved: append a new (merge_timestamp, lo, hi) regime to _CLAMP_HISTORY "
    "and clip_threshold_windows._CLAMP_REGIME_START instead of editing the last row"
)
assert _CLAMP_HISTORY[MULTIPLE_CHOICE][-1][1:] == (MC_PROB_MIN, MC_PROB_MAX), (
    "MC_PROB_MIN/MAX moved: append a new (merge_timestamp, lo, hi) regime to _CLAMP_HISTORY "
    "and clip_threshold_windows._CLAMP_REGIME_START instead of editing the last row"
)

# Extreme-bin edges: (label, lower, upper, p_midpoint). Low bins are (lower, upper], high
# bins [lower, upper). The implied rate of the COUNTED event is the midpoint for a low bin
# and its complement for a high one, because a high bin counts NO resolutions.
LOW_PRICE_BINS: tuple[tuple[str, float, float, float], ...] = (
    ("<= 0.01", -1.0, 0.01, 0.005),
    ("(0.01, 0.02]", 0.01, 0.02, 0.015),
    ("(0.02, 0.03]", 0.02, 0.03, 0.025),
    ("(0.03, 0.05]", 0.03, 0.05, 0.04),
    ("(0.05, 0.10]", 0.05, 0.10, 0.075),
)
HIGH_PRICE_BINS: tuple[tuple[str, float, float, float], ...] = (
    (">= 0.99", 0.99, 2.0, 0.995),
    ("[0.98, 0.99)", 0.98, 0.99, 0.985),
    ("[0.97, 0.98)", 0.97, 0.98, 0.975),
    ("[0.95, 0.97)", 0.95, 0.97, 0.96),
    ("[0.90, 0.95)", 0.90, 0.95, 0.925),
)


def in_force_bounds(question_type: str, moment: datetime | None) -> tuple[float, float]:
    """The clamp live for a record of this type published at ``moment``.

    ``moment=None`` (an undatable record) gets the WIDEST historical clamp: a censoring
    claim needs to know which floor was live, so the assumption that claims the least is the
    loosest one. An unrecognised question type raises rather than guessing a clamp.
    """
    history = _CLAMP_HISTORY[question_type]
    bounds = (history[0][1], history[0][2])
    if moment is None:
        return bounds
    for start, lo, hi in history:
        if start is None or moment >= start:
            bounds = (lo, hi)
    return bounds


def _median_positions(n: int) -> tuple[int, ...]:
    """The sorted-order indices the median reads: one for odd ``n``, the middle two for even."""
    return (n // 2,) if n % 2 else (n // 2 - 1, n // 2)


def _mc_option_combine(values: list[float], aggregator: str) -> float:
    """The per-option combine ``_aggregate_mc_options`` applies: a median, or ``sum / len`` for the mean."""
    return sum(values) / len(values) if aggregator == "mean" else statistics.median(values)


def replay_members(
    members: Sequence[Sequence[float]],
    *,
    is_binary: bool,
    lo: float,
    hi: float,
    aggregator: str,
) -> tuple[float, ...]:
    """Re-run the aggregation with the members clamped at ``[lo, hi]``.

    Binary calls the LIVE aggregators (``aggregate_binary_median`` / ``aggregate_binary_mean``,
    each rounding to 3 dp) so a change to the live rounding or summation reaches this replay
    rather than being mirrored by hand; MC mirrors ``_aggregate_mc_options`` (per-member
    clamp+renormalise, per-option combine, then clamp+renormalise again), whose live form
    takes forecasting-tools option objects this vector shape does not build.
    ``aggregator="mean"`` is what the 2025-09 records were published with; ``"unknown"``
    replays as a median, the live default. The caller guarantees ``members`` is non-empty.
    """
    if is_binary:
        clamped = [min(hi, max(lo, member[1])) for member in members]
        p_yes = aggregate_binary_mean(clamped) if aggregator == "mean" else aggregate_binary_median(clamped)
        return (1.0 - p_yes, p_yes)
    per_member = [clamp_and_renormalize_probs(member, lo=lo, hi=hi) for member in members]
    n_options = len(members[0])
    combined = [_mc_option_combine([member[i] for member in per_member], aggregator) for i in range(n_options)]
    return tuple(clamp_and_renormalize_probs(combined, lo=lo, hi=hi))


def detect_aggregator(
    published: Sequence[float],
    members: Sequence[Sequence[float]],
    *,
    is_binary: bool,
    lo: float,
    hi: float,
    stacker_verdict: str,
) -> Aggregator:
    """Which aggregation of the recovered members rebuilds the published vector.

    The archive holds a 2025-09 era whose published value is the members' MEAN, not their
    median (q38797: mean(0.55, 0.60, 0.85) = 0.667 = published). Replaying those records with
    the median charges the aggregator gap to the clip, so the replay has to know which rule
    produced the publish. A stacked record or one with no recovered members is ``unknown``:
    its published value is not any aggregate of the recovered members.
    """
    if not members or stacker_verdict == CONFIRMED_STACKER:
        return "unknown"
    for candidate in ("median", "mean"):
        rebuilt = replay_members(members, is_binary=is_binary, lo=lo, hi=hi, aggregator=candidate)
        if max(abs(a - b) for a, b in zip(rebuilt, published, strict=True)) <= REPLAY_DISAGREE_ATOL:
            return candidate  # type: ignore[return-value]
    return "unknown"


@dataclass(frozen=True, slots=True)
class ClipRecord:
    """One resolved publish, reduced to what a clip counterfactual needs."""

    question_id: str
    question_type: str
    created_at: datetime | None
    in_force_lo: float
    in_force_hi: float
    published: tuple[float, ...]
    resolving_index: int
    members: tuple[tuple[float, ...], ...]
    """Per-model forecasts in the same vector shape, when the collector recovered them."""
    stacker_verdict: str
    aggregator: str
    """``median`` / ``mean`` / ``unknown``: which rule over ``members`` rebuilds ``published``."""
    spot_peer: float | None

    @property
    def is_binary(self) -> bool:
        return self.question_type == BINARY

    @property
    def published_resolving_mass(self) -> float:
        return self.published[self.resolving_index]

    @property
    def censor_atol(self) -> float:
        return BINARY_CENSOR_ATOL if self.is_binary else MC_CENSOR_ATOL

    @property
    def clampable_indices(self) -> tuple[int, ...]:
        """The vector positions the clamp acts on.

        For binary that is ``p_yes`` alone: ``p_no`` is its complement, so a floor on
        ``p_yes`` is the same constraint as a ceiling on ``p_no``, and counting both would
        report a publish at the ceiling as floor-censored. MC clamps every option.
        """
        return (1,) if self.is_binary else tuple(range(len(self.published)))

    @property
    def clampable_values(self) -> tuple[float, ...]:
        return tuple(self.published[i] for i in self.clampable_indices)

    @property
    def replayable(self) -> bool:
        """Whether the members can be re-aggregated at all (non-stacked, members recovered)."""
        return bool(self.members) and self.stacker_verdict != CONFIRMED_STACKER

    @property
    def members_rebuild_publish(self) -> bool:
        """Whether a KNOWN aggregator over the members reproduces the published value.

        The member-level censoring rule reads member positions, which only mean anything when
        the publish really is that aggregate of those members; an ``unknown`` record falls back
        to the published-value rule.
        """
        return self.replayable and self.aggregator != "unknown"

    def replay(self, lo: float, hi: float) -> tuple[float, ...]:
        return replay_members(self.members, is_binary=self.is_binary, lo=lo, hi=hi, aggregator=self.aggregator)

    def floor_infeasible(self, lo: float) -> bool:
        """Whether floor ``lo`` cannot be delivered on this ballot (MC with more than ``1 / lo`` options).

        Mirrors the degenerate test in :func:`clamp_and_renormalize_probs`: past it the live
        clamp returns a sub-floor fallback, so a cell priced at ``lo`` was not priced at ``lo``
        on this record. Binary has one free value and is always feasible.
        """
        return not self.is_binary and len(self.published) * lo > 1.0 + FLOOR_FEASIBILITY_ATOL


@dataclass(frozen=True, slots=True)
class ClipCohort:
    """The usable records of one question type, plus what was dropped getting there."""

    question_type: str
    records: tuple[ClipRecord, ...]
    n_skipped: int
    n_no_timestamp: int
    n_at_floor: int
    n_at_ceiling: int
    n_member_censored_floor: int
    """Records with a clamped member in a median position; a superset of ``n_at_floor``."""

    def to_dict(self) -> dict:
        return {
            "question_type": self.question_type,
            "n": len(self.records),
            "n_skipped": self.n_skipped,
            "n_no_timestamp": self.n_no_timestamp,
            "n_at_in_force_floor": self.n_at_floor,
            "n_at_in_force_ceiling": self.n_at_ceiling,
            "n_member_censored_floor": self.n_member_censored_floor,
        }


def _binary_clip_record(record: dict) -> ClipRecord | None:
    p_yes = record.get("our_prob_yes")
    resolution = record.get("resolution_parsed")
    if p_yes is None or not isinstance(resolution, bool):
        return None
    created_at = parse_iso_utc(record.get("bot_comment_created_at"))
    lo, hi = in_force_bounds(BINARY, created_at)
    members = tuple(
        (1.0 - parsed, parsed)
        for parsed in (_parse_probability(str(raw)) for raw in base_or_per_model_forecasts(record).values())
        if parsed is not None
    )
    published = (1.0 - float(p_yes), float(p_yes))
    verdict = detect_stacker_fired(record)
    return ClipRecord(
        question_id=str(record.get("question_id")),
        question_type=BINARY,
        created_at=created_at,
        in_force_lo=lo,
        in_force_hi=hi,
        published=published,
        resolving_index=1 if resolution else 0,
        members=members,
        stacker_verdict=verdict,
        aggregator=detect_aggregator(published, members, is_binary=True, lo=lo, hi=hi, stacker_verdict=verdict),
        spot_peer=spot_peer_score(record),
    )


def _mc_member_vectors(record: dict, options: list[str]) -> tuple[tuple[float, ...], ...]:
    """Per-model MC vectors aligned to ``options``, keeping only complete ballots.

    A partial ballot cannot be replayed through a clamp+renormalise without inventing the
    missing options, so it is dropped rather than padded.
    """
    vectors: list[tuple[float, ...]] = []
    for model_options in base_or_per_model_forecasts(record).values():
        if not isinstance(model_options, dict):
            continue
        if any(model_options.get(option) is None for option in options):
            continue
        vectors.append(tuple(float(model_options[option]) for option in options))
    return tuple(vectors)


def _mc_clip_record(record: dict) -> ClipRecord | None:
    options = list(record.get("options") or [])
    values = record.get("our_forecast_values") or []
    resolution = record.get("resolution_parsed")
    if len(options) < 2 or len(options) != len(values) or resolution not in options:
        return None
    created_at = parse_iso_utc(record.get("bot_comment_created_at"))
    lo, hi = in_force_bounds(MULTIPLE_CHOICE, created_at)
    published = tuple(float(v) for v in values)
    members = _mc_member_vectors(record, options)
    verdict = detect_stacker_fired(record)
    return ClipRecord(
        question_id=str(record.get("question_id")),
        question_type=MULTIPLE_CHOICE,
        created_at=created_at,
        in_force_lo=lo,
        in_force_hi=hi,
        published=published,
        resolving_index=options.index(resolution),
        members=members,
        stacker_verdict=verdict,
        aggregator=detect_aggregator(published, members, is_binary=False, lo=lo, hi=hi, stacker_verdict=verdict),
        spot_peer=spot_peer_score(record),
    )


def build_clip_records(data: Sequence[dict], question_type: str) -> ClipCohort:
    """Reduce a cached performance dataset to the clip records of one question type.

    Records come back OLDEST FIRST, undated ones ahead of the dated block, which is what
    lets every suffix window be a slice of the tail.
    """
    build = _binary_clip_record if question_type == BINARY else _mc_clip_record
    records: list[ClipRecord] = []
    n_skipped = 0
    for raw in data:
        if raw.get("type") != question_type:
            continue
        built = build(raw)
        if built is None:
            n_skipped += 1
            continue
        records.append(built)
    records.sort(key=lambda r: (r.created_at is not None, r.created_at or datetime.min.replace(tzinfo=UTC)))
    cohort = ClipCohort(
        question_type=question_type,
        records=tuple(records),
        n_skipped=n_skipped,
        n_no_timestamp=sum(1 for r in records if r.created_at is None),
        n_at_floor=sum(1 for r in records if at_in_force_floor(r)),
        n_at_ceiling=sum(1 for r in records if at_in_force_ceiling(r)),
        n_member_censored_floor=sum(1 for r in records if member_censored(r, floor_side=True, ceiling_side=False)),
    )
    logger.info(
        f"clip cohort {question_type}: n={len(cohort.records)} skipped={cohort.n_skipped} "
        f"undated={cohort.n_no_timestamp} at_floor={cohort.n_at_floor} at_ceiling={cohort.n_at_ceiling} "
        f"member_censored_floor={cohort.n_member_censored_floor}"
    )
    return cohort


def at_in_force_floor(record: ClipRecord) -> bool:
    """Whether any clamped PUBLISHED value sits at the floor that was live (the narrow rule)."""
    return any(v <= record.in_force_lo + record.censor_atol for v in record.clampable_values)


def at_in_force_ceiling(record: ClipRecord) -> bool:
    return any(v >= record.in_force_hi - record.censor_atol for v in record.clampable_values)


def member_censored(record: ClipRecord, *, floor_side: bool, ceiling_side: bool) -> bool:
    """Whether a clamped MEMBER could have moved the published value under a looser clip.

    Under a median aggregator only a member in a median position can move the publish, so
    the rule checks those positions; under a mean every member moves it. A record whose
    members were not recovered, whose publish is a stacker's output, or whose publish no known
    aggregator rebuilds falls back to the published-value rule, so this is never narrower than
    :func:`at_in_force_floor`.

    For MC the rule is applied per OPTION, to the members in THAT option's median position.
    It deliberately does not count a member floored on option j but sitting in option k's
    median slot with a non-floored value: renormalisation does couple the two, but the round's
    refutation pass rejected "any member component at the floor" as overstating the bound,
    and the coupled move is bounded by the floored mass a looser clip releases.
    """
    if not record.members_rebuild_publish:
        return (floor_side and at_in_force_floor(record)) or (ceiling_side and at_in_force_ceiling(record))
    tol = record.censor_atol
    for index in record.clampable_indices:
        values = sorted(member[index] for member in record.members)
        positions = range(len(values)) if record.aggregator == "mean" else _median_positions(len(values))
        if floor_side and any(values[i] <= record.in_force_lo + tol for i in positions):
            return True
        if ceiling_side and any(values[i] >= record.in_force_hi - tol for i in positions):
            return True
    return False


def _candidate_bounds(record: ClipRecord, c: float, side: ClipSide) -> tuple[float, float]:
    """The bounds the candidate ASKS for, before intersecting with what was in force."""
    lo = c if side in ("floor_only", "symmetric") else record.in_force_lo
    hi = (1.0 - c) if side in ("ceiling_only", "symmetric") else record.in_force_hi
    return lo, hi


def apply_bounds(record: ClipRecord, lo: float, hi: float) -> tuple[float, ...]:
    """The published vector re-clamped into ``[lo, hi]`` the way the pipeline would have.

    Binary clamps ``p_yes`` and takes the complement (the live per-model rule in
    ``forecaster_runners``); MC goes through the live clamp-and-renormalise, so the
    counterfactual inherits its bound-repair iteration rather than a naive divide.
    """
    if record.is_binary:
        p_yes = min(hi, max(lo, record.published[1]))
        return (1.0 - p_yes, p_yes)
    return tuple(clamp_and_renormalize_probs(record.published, lo=lo, hi=hi))


def _assumed_at_bound(record: ClipRecord, value: float, c: float, *, floor_side: bool, ceiling_side: bool) -> float:
    """``value`` if it sat at an in-force bound and the raw had really been at ``c``."""
    tol = record.censor_atol
    if floor_side and value <= record.in_force_lo + tol:
        return c
    if ceiling_side and value >= record.in_force_hi - tol:
        return 1.0 - c
    return value


def _loosen_scenario_vector(
    record: ClipRecord,
    c: float,
    *,
    floor_side: bool,
    ceiling_side: bool,
) -> tuple[float, ...]:
    """The PUBLISHED vector if every value sitting at an in-force bound had really been at ``c``.

    This is the ``sum_delta_upper`` scenario: the most a looser clip could have moved this
    record, given the clamp erased whatever its members actually said.
    """
    lo = c if floor_side else record.in_force_lo
    hi = (1.0 - c) if ceiling_side else record.in_force_hi
    if record.is_binary:
        p_yes = _assumed_at_bound(record, record.published[1], c, floor_side=floor_side, ceiling_side=ceiling_side)
        return (1.0 - p_yes, p_yes)
    assumed = [
        _assumed_at_bound(record, v, c, floor_side=floor_side, ceiling_side=ceiling_side) for v in record.published
    ]
    return tuple(clamp_and_renormalize_probs(assumed, lo=lo, hi=hi))


def _member_loosen_delta(record: ClipRecord, c: float, *, floor_side: bool, ceiling_side: bool) -> float:
    """The member-replay analogue of the upper scenario: floored MEMBERS assumed at ``c``.

    Each clamped member component is set to ``c`` (or ``1 - c``), the members are re-aggregated
    under the record's own aggregator at the looser bounds, and the delta is taken against
    the replay's own unclipped baseline so an aggregator-era gap is never charged to the clip.
    """
    lo = c if floor_side else record.in_force_lo
    hi = (1.0 - c) if ceiling_side else record.in_force_hi
    assumed: list[tuple[float, ...]] = []
    for member in record.members:
        if record.is_binary:
            p_yes = _assumed_at_bound(record, member[1], c, floor_side=floor_side, ceiling_side=ceiling_side)
            assumed.append((1.0 - p_yes, p_yes))
        else:
            assumed.append(
                tuple(_assumed_at_bound(record, v, c, floor_side=floor_side, ceiling_side=ceiling_side) for v in member)
            )
    scenario = replay_members(assumed, is_binary=record.is_binary, lo=lo, hi=hi, aggregator=record.aggregator)
    baseline = record.replay(record.in_force_lo, record.in_force_hi)
    index = record.resolving_index
    return spot_peer_delta(old_prob=baseline[index], new_prob=scenario[index], question_type=record.question_type)


@dataclass(frozen=True, slots=True)
class RecordClip:
    """One record's contribution to one ``(side, c)`` cell."""

    question_id: str
    delta: float
    """Exact spot-peer delta on the TIGHTENING path; 0 when the candidate is looser."""
    affected: bool
    expected_delta: float
    """The delta a perfectly calibrated forecaster would expect from this move under the
    record's OWN published probabilities: ``sum_k p_k * 100 ln(new_k / p_k)``, which is minus
    100 times the KL divergence and never positive. This is the properness cost of the clip;
    ``delta - expected_delta`` is how much kinder or harsher the archive was than priced."""
    best_case_delta: float
    """The delta had the outcome the clip favoured most resolved (the insurance ceiling)."""
    worst_case_delta: float
    """The delta had the outcome the clip hurt most resolved."""
    infeasible: bool
    """The floor actually applied on the tightening path cannot be delivered on this ballot
    (MC, more options than ``1 / floor``), so ``delta`` prices the live clamp's sub-floor
    fallback rather than the candidate."""
    loosening: bool
    """The candidate relaxes EITHER bound relative to the clamp in force, censored or not."""
    censored: bool
    """The candidate is looser than the clamp in force AND the publish sat at that bound."""
    loosen_at_c: float
    """Delta under the scenario "the raw published value was at or below ``c``"; 0 when uncensored."""
    member_censored: bool
    """Looser candidate AND a clamped member sat where it could move the publish."""
    loosen_members_at_c: float
    """Delta under the scenario "every clamped member was at ``c``", via the replay path."""

    @property
    def bracket_lo(self) -> float:
        """This record's worst case over the raw values the clamp could have hidden."""
        return min(0.0, self.loosen_at_c)

    @property
    def bracket_hi(self) -> float:
        return max(0.0, self.loosen_at_c)


def _outcome_deltas(record: ClipRecord, counterfactual: Sequence[float]) -> list[float]:
    """Spot-peer delta of the move for EACH possible outcome, in vector order."""
    return [
        spot_peer_delta(old_prob=old, new_prob=new, question_type=record.question_type)
        for old, new in zip(record.published, counterfactual, strict=True)
    ]


@dataclass(frozen=True, slots=True)
class _Tightening:
    delta: float
    affected: bool
    expected: float
    best: float
    worst: float
    infeasible: bool


def _tighten(record: ClipRecord, lo: float, hi: float) -> _Tightening:
    """The exact tightening path: the candidate intersected with the clamp in force."""
    floor = max(lo, record.in_force_lo)
    tight = apply_bounds(record, floor, min(hi, record.in_force_hi))
    infeasible = record.floor_infeasible(floor)
    delta = spot_peer_delta(
        old_prob=record.published_resolving_mass,
        new_prob=tight[record.resolving_index],
        question_type=record.question_type,
    )
    if abs(delta) <= DELTA_ATOL:
        return _Tightening(delta=delta, affected=False, expected=0.0, best=0.0, worst=0.0, infeasible=infeasible)
    by_outcome = _outcome_deltas(record, tight)
    return _Tightening(
        delta=delta,
        affected=True,
        expected=float(sum(p * d for p, d in zip(record.published, by_outcome, strict=True))),
        best=max(by_outcome),
        worst=min(by_outcome),
        infeasible=infeasible,
    )


@dataclass(frozen=True, slots=True)
class _Loosening:
    loosening: bool
    """The candidate relaxes EITHER bound relative to the clamp in force, censored or not."""
    censored: bool
    loosen_at_c: float
    member_censored: bool
    loosen_members_at_c: float


def _loosen(record: ClipRecord, c: float, *, floor_side: bool, ceiling_side: bool) -> _Loosening:
    """The censored loosening path, under both the published-value and the member rule.

    ``floor_side`` and ``ceiling_side`` stay separate flags all the way down (the member
    rule, the scenario vector and the member replay each read them independently), so the
    ceiling scenario is priced as a ceiling and never as a floor.
    """
    if not (floor_side or ceiling_side):
        return _Loosening(
            loosening=False, censored=False, loosen_at_c=0.0, member_censored=False, loosen_members_at_c=0.0
        )
    censored = (floor_side and at_in_force_floor(record)) or (ceiling_side and at_in_force_ceiling(record))
    loosen_at_c = 0.0
    if censored:
        scenario = _loosen_scenario_vector(record, c, floor_side=floor_side, ceiling_side=ceiling_side)
        loosen_at_c = spot_peer_delta(
            old_prob=record.published_resolving_mass,
            new_prob=scenario[record.resolving_index],
            question_type=record.question_type,
        )
    censored_m = member_censored(record, floor_side=floor_side, ceiling_side=ceiling_side)
    loosen_members = 0.0
    if censored_m:
        loosen_members = (
            _member_loosen_delta(record, c, floor_side=floor_side, ceiling_side=ceiling_side)
            if record.members_rebuild_publish
            else loosen_at_c
        )
    return _Loosening(
        loosening=True,
        censored=censored,
        loosen_at_c=loosen_at_c,
        member_censored=censored_m,
        loosen_members_at_c=loosen_members,
    )


def clip_delta(record: ClipRecord, c: float, *, side: ClipSide) -> RecordClip:
    """Price candidate floor ``c`` on one record, keeping tightening and loosening apart."""
    lo, hi = _candidate_bounds(record, c, side)
    tight = _tighten(record, lo, hi)
    loose = _loosen(record, c, floor_side=lo < record.in_force_lo, ceiling_side=hi > record.in_force_hi)
    return RecordClip(
        question_id=record.question_id,
        delta=tight.delta,
        affected=tight.affected,
        expected_delta=tight.expected,
        best_case_delta=tight.best,
        worst_case_delta=tight.worst,
        infeasible=tight.infeasible,
        loosening=loose.loosening,
        censored=loose.censored,
        loosen_at_c=loose.loosen_at_c,
        member_censored=loose.member_censored,
        loosen_members_at_c=loose.loosen_members_at_c,
    )


def bootstrap_mean_ci(deltas: Sequence[float]) -> tuple[float | None, float | None]:
    """Percentile bootstrap 95% CI of the mean delta, resampling QUESTIONS with replacement.

    The resampling itself is the shared :func:`metaculus_bot.bootstrap.bootstrap_means`
    (re-seeded per call, so a row's interval never depends on how many rows ran before it).
    ``(None, None)`` on an empty sample: there is no mean to bracket, and 0.0 would read as
    a measured zero. On an identically-zero sample the interval is exactly ``(0.0, 0.0)``;
    the report renders such rows as ``identity`` rather than as an interval, because a
    bootstrap of nothing is not a precision statement.
    """
    if not deltas:
        return None, None
    means = bootstrap_means(deltas, n_bootstrap=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, cache=True)
    half = (1.0 - BOOTSTRAP_CL) / 2.0
    lo, hi = np.quantile(means, [half, 1.0 - half])
    return float(lo), float(hi)


@dataclass(frozen=True, slots=True)
class SweepRow:
    """One ``(type, side, window, c)`` cell of the sweep."""

    question_type: str
    side: str
    window: str
    c: float
    n: int
    n_affected: int
    n_loosening: int
    """Records for which ``c`` is LOOSER than the clamp in force on either side, censored or not."""
    censored_n: int
    """Records published AT the in-force bound (the narrow, published-value rule)."""
    member_censored_n: int
    """Records with a clamped member where it could move the publish (the member rule)."""
    infeasible_n: int
    """MC records with more options than ``1 / floor``, priced at the live clamp's sub-floor
    fallback rather than at the candidate; 0 on every binary row."""
    sum_delta: float
    mean_delta: float | None
    """``sum_delta`` over the window's n, so it reads as points per question forecast."""
    ci_lo: float | None
    ci_hi: float | None
    hits_on_clipped_side: int
    """Affected records where the clip moved mass toward the outcome that resolved."""
    top1_share: float | None
    """Largest |delta| over the sum of |delta|; 1.0 means one question is the whole row."""
    top1_question_id: str | None
    """WHICH question that is. Without it a top1_share of 0.97 sends the reader to a script."""
    top1_spot_peer: float | None
    """That question's actual spot peer, so a row's driver can be recognised on sight."""
    expected_sum_delta: float
    """What the same move would cost a perfectly calibrated forecaster under the bot's own
    published prices (the properness cost); never positive."""
    best_case_sum_delta: float
    """Sum of every affected record's best outcome: the most the clip could have earned."""
    worst_case_sum_delta: float
    sum_delta_lower: float
    """Censoring scenario: every censored raw value was exactly the floor (nothing moves)."""
    sum_delta_upper: float
    """Censoring scenario: every censored raw PUBLISHED value was at or below ``c``."""
    sum_delta_upper_members: float
    """Censoring scenario on the member path: every clamped MEMBER was at ``c``."""
    bracket_lo: float
    """The identified set, each censored record contributing its own worst case."""
    bracket_hi: float
    shippable: bool

    @property
    def exact(self) -> bool:
        """True when nothing in this row is censored under EITHER rule, so ``sum_delta`` is the whole answer.

        A row with ``censored_n == 0`` but ``member_censored_n > 0`` reports an exact 0 that
        its own ``sum_delta_upper_members`` says is unobservable, so it must not compete for
        the argmax as if it were neutral.
        """
        return self.censored_n == 0 and self.member_censored_n == 0

    @property
    def identity(self) -> bool:
        """True when the candidate moved no record: the row is the do-nothing map.

        Its ``sum_delta`` is a hard 0 and its CI the bootstrap of an all-zero vector, which
        the report must not print as a tight interval.
        """
        return self.n_affected == 0

    def to_dict(self) -> dict:
        return asdict(self) | {"exact": self.exact, "identity": self.identity}


def sweep_row(
    records: Sequence[ClipRecord],
    *,
    question_type: str,
    side: ClipSide,
    window: str,
    c: float,
) -> SweepRow:
    """Aggregate one candidate over one window's records."""
    clips = [clip_delta(record, c, side=side) for record in records]
    deltas = [clip.delta for clip in clips]
    sum_delta = float(sum(deltas))
    abs_deltas = [abs(d) for d in deltas]
    total_abs = sum(abs_deltas)
    ci_lo, ci_hi = bootstrap_mean_ci(deltas)
    # Only a row that MOVED something has a driver. An MC row where the candidate is looser
    # than the clamp in force still carries ~1e-13 of renormalisation noise, and a share
    # computed over that noise reads as a real concentration (0.07) and names a question the
    # candidate never touched, on a row whose own n_affected is 0.
    moved = total_abs > DELTA_ATOL
    driver = max(zip(records, clips, strict=True), key=lambda pair: abs(pair[1].delta))[0] if moved else None
    return SweepRow(
        question_type=question_type,
        side=side,
        window=window,
        c=c,
        n=len(records),
        n_affected=sum(1 for clip in clips if clip.affected),
        n_loosening=sum(1 for clip in clips if clip.loosening),
        censored_n=sum(1 for clip in clips if clip.censored),
        member_censored_n=sum(1 for clip in clips if clip.member_censored),
        infeasible_n=sum(1 for clip in clips if clip.infeasible),
        sum_delta=sum_delta,
        mean_delta=(sum_delta / len(records) if records else None),
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        hits_on_clipped_side=sum(1 for clip in clips if clip.affected and clip.delta > 0.0),
        top1_share=(max(abs_deltas) / total_abs if moved else None),
        top1_question_id=(driver.question_id if driver else None),
        top1_spot_peer=(driver.spot_peer if driver else None),
        expected_sum_delta=float(sum(clip.expected_delta for clip in clips)),
        best_case_sum_delta=float(sum(clip.best_case_delta for clip in clips)),
        worst_case_sum_delta=float(sum(clip.worst_case_delta for clip in clips)),
        sum_delta_lower=sum_delta,
        sum_delta_upper=sum_delta + float(sum(clip.loosen_at_c for clip in clips)),
        sum_delta_upper_members=sum_delta + float(sum(clip.loosen_members_at_c for clip in clips)),
        bracket_lo=sum_delta + float(sum(clip.bracket_lo for clip in clips)),
        bracket_hi=sum_delta + float(sum(clip.bracket_hi for clip in clips)),
        shippable=(question_type != MULTIPLE_CHOICE or c >= MC_PROB_MIN),
    )


def argmax_rows(rows: Sequence[SweepRow]) -> list[SweepRow]:
    """Every EXACT candidate tied for the best ``sum_delta``, in increasing ``c``.

    Two rules, both learned from the real archive. Only EXACT rows compete: a row carrying
    censored records has an unobservable component, so its ``sum_delta`` of 0 would read as
    "this candidate is neutral" when the truth is "we cannot say". And the winner is
    routinely a PLATEAU rather than a point — every candidate at or below a window's
    in-force floor scores exactly 0 when no publish was clamped — so the set is returned
    instead of one representative, and the caller reports the tie.
    """
    exact = [row for row in rows if row.exact]
    if not exact:
        return []
    best = max(row.sum_delta for row in exact)
    return sorted((row for row in exact if best - row.sum_delta <= ARGMAX_TIE_ATOL), key=lambda row: row.c)


def argmax_row(rows: Sequence[SweepRow]) -> SweepRow | None:
    """The smallest-``c`` member of :func:`argmax_rows`: the least interventionist winner."""
    tied = argmax_rows(rows)
    return tied[0] if tied else None
