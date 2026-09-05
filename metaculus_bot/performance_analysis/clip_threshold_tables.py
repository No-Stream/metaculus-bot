"""The derived tables the clip sweep reports: calibration bins, insurance view, carry, replay.

These read the per-record and per-window primitives in ``clip_threshold_sweep`` and turn them
into the things a reader actually looks at, in the order the report prints them. The layer
split is one-way (CLI -> report -> here -> selection / windows -> sweep) and exists because
each of these tables answers a different question:

* :func:`binary_extreme_bins` / :func:`mc_extreme_bins`: how often the bot's extreme calls
  actually happened, beside how often its OWN prices said they would. This is the direct
  evidence; every sweep number is a re-expression of it.
* :func:`insurance_row`: what a floor is FOR. A floor is insurance against the sub-``c`` band
  being under-priced, so the row carries the break-even hit rate the floor needs, the
  Jeffreys interval on the observed rate, the properness cost a calibrated forecaster pays
  for the clip regardless, and the ceiling the insurance could ever have paid.
* :func:`nesting_rows`: which questions the nested windows are re-counting, so seven agreeing
  rows are not mistaken for seven replications.
* :func:`oos_row`: whether a floor fitted on the PAST carries into a later window. A clip
  level is a fitted calibration layer, and AGENTS.md's era-bucketing rule is that one ships
  only after an out-of-sample era test. A fit that moves nothing is flagged: its carry is
  vacuous, not a pass.
* :func:`replay_cohort` / :func:`cross_check_row`: the honesty check on the sweep's shortcut of
  clamping the PUBLISHED median instead of replaying members through the clamp, with the
  aggregator each record was actually published under. The per-window facts (how many
  records replay, under which aggregator, how many are even-count) live on the cohort object
  once; the per-candidate row carries only what depends on ``c``.
* :func:`single_survivor_report`: the one cohort the live single-forecaster publish floor
  (``THIN_PUBLISH_BINARY_FLOOR`` / ``_CEIL``) would fire on, priced on its own.
* :func:`compute_report`: the whole pass for both question types. Every selection-derived
  quantity the report prints (the argmax plateau, the censored ties, the sign of the replay
  error, the live floor priced on the older regime) is computed here and stored on the
  :class:`TypeReport`, so the markdown never carries a number ``--output-json`` lacks.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime

from metaculus_bot.constants import (
    CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD,
    THIN_PUBLISH_BINARY_CEIL,
    THIN_PUBLISH_BINARY_FLOOR,
)
from metaculus_bot.performance_analysis.analysis import B4E9DF0_MERGED_AT, jeffreys_ci
from metaculus_bot.performance_analysis.clip_threshold_selection import (
    OOB_BOOTSTRAP_B,
    OobArgmax,
    affected_question_ids,
    binomial_cdf,
    censored_rows_at_argmax_score,
    oob_argmax,
)
from metaculus_bot.performance_analysis.clip_threshold_sweep import (
    BINARY,
    BOOTSTRAP_B,
    BOOTSTRAP_CL,
    BOOTSTRAP_SEED,
    DELTA_ATOL,
    GRID_BY_TYPE,
    HIGH_PRICE_BINS,
    LOW_PRICE_BINS,
    QUESTION_TYPES,
    REPLAY_DISAGREE_ATOL,
    SIDES,
    ClipCohort,
    ClipRecord,
    SweepRow,
    apply_bounds,
    argmax_row,
    argmax_rows,
    build_clip_records,
    in_force_bounds,
    sweep_row,
)
from metaculus_bot.performance_analysis.clip_threshold_windows import (
    LOOKBACK_DAYS,
    WINDOW_ALL,
    WINDOW_CURRENT_CLAMP,
    Window,
    build_windows,
    nested_windows,
)
from metaculus_bot.post_processing import apply_thin_publish_floor
from metaculus_bot.scoring_common import spot_peer_delta
from metaculus_bot.spread_metrics import binary_prob_range_spread

logger: logging.Logger = logging.getLogger(__name__)

# Below this many records a complement cannot fit a clip level; the OOS row prints n/a
# rather than a number, mirroring width_monitor's MIN_N_FOR_POINT_METRICS convention.
MIN_OOS_COMPLEMENT_N: int = 30

# The label on the one MEASURED floor comparison: the floor now in force priced on the
# records published before it went live, which is the current-regime window's complement.
WINDOW_OLDER_REGIME = f"before_{WINDOW_CURRENT_CLAMP}"


@dataclass(frozen=True, slots=True)
class ExtremeBin:
    """One published-probability bin of the calibration table, for one window."""

    window: str
    """Which window's records this bin counted. Pooling windows mixes clamp regimes."""
    label: str
    side: str
    n: int
    hits: int
    """Resolutions on the side the extreme price bet against: YES for low, NO for high."""
    expected_hits: float
    """Hits the bot's OWN published prices implied for these records (sum of the counted
    event's price). The comparison that says whether the band was under-priced."""
    hit_rate: float | None
    ci_lo: float | None
    ci_hi: float | None
    implied_rate: float
    """The bin's own midpoint, expressed as the rate of the counted event."""

    def to_dict(self) -> dict:
        return asdict(self)


def jeffreys_interval(k: int, n: int) -> tuple[float | None, float | None, float | None]:
    """Observed hit rate ``k / n`` beside the equal-tailed Jeffreys(0.5, 0.5) interval on it.

    The interval is :func:`metaculus_bot.performance_analysis.analysis.jeffreys_ci` at the
    sweep's own ``BOOTSTRAP_CL`` (passed explicitly, so this module's level and the shared
    function's default cannot silently diverge); only the point estimate differs, because the
    extreme-bin table contrasts the RAW hit count with the bot's own expected hits, and the
    posterior mean would blur that. ``(None, None, None)`` on an empty bin.
    """
    if n == 0:
        return None, None, None
    _, lo, hi = jeffreys_ci(k, n, cl=BOOTSTRAP_CL)
    return k / n, lo, hi


def _bin_row(window: str, label: str, side: str, *, priced: list[tuple[float, bool]], implied: float) -> ExtremeBin:
    """``priced`` pairs each in-bin record's price OF THE COUNTED EVENT with whether it happened."""
    n = len(priced)
    hits = sum(hit for _, hit in priced)
    rate, lo, hi = jeffreys_interval(hits, n)
    return ExtremeBin(
        window=window,
        label=label,
        side=side,
        n=n,
        hits=hits,
        expected_hits=float(sum(price for price, _ in priced)),
        hit_rate=rate,
        ci_lo=lo,
        ci_hi=hi,
        implied_rate=implied,
    )


def binary_extreme_bins(records: Sequence[ClipRecord], *, window: str) -> list[ExtremeBin]:
    """How often the bot's extreme binary calls actually happened, per published price bin.

    A low bin counts YES resolutions against an implied rate of the bin midpoint; a high bin
    counts NO resolutions against the complement of its midpoint. Both columns therefore
    measure one thing: how often the outcome the bot nearly ruled out occurred. ``window``
    labels which records were counted: the table is computed per window because the pooled
    one mixes two clamp regimes, and the recent windows are what a clip decision rests on.
    """
    priced = [(r.published[1], r.resolving_index == 1) for r in records]
    rows = [
        _bin_row(window, label, "low", priced=[(p, hit) for p, hit in priced if lower < p <= upper], implied=mid)
        for label, lower, upper, mid in LOW_PRICE_BINS
    ]
    rows += [
        _bin_row(
            window,
            label,
            "high",
            priced=[(1.0 - p, not hit) for p, hit in priced if lower <= p < upper],
            implied=1.0 - mid,
        )
        for label, lower, upper, mid in HIGH_PRICE_BINS
    ]
    return rows


def mc_extreme_bins(records: Sequence[ClipRecord], *, window: str) -> list[ExtremeBin]:
    """The same table over MC OPTION prices; the unit is an option, not a question.

    Low bins only: an MC option's price does not reach the high bins in practice (the
    archive's dearest option is 0.92) and the mirrored table would be empty. Options inside
    one question are correlated, so these intervals are not question-clustered and read as
    an option-level rate.
    """
    priced = [(price, index == r.resolving_index) for r in records for index, price in enumerate(r.published)]
    return [
        _bin_row(window, label, "low", priced=[(p, hit) for p, hit in priced if lower < p <= upper], implied=mid)
        for label, lower, upper, mid in LOW_PRICE_BINS
    ]


@dataclass(frozen=True, slots=True)
class InsuranceRow:
    """What a floor at ``c`` would have to be insuring against, beside what it cost.

    Spot peer is a proper score, so a clip loses in expectation under the bot's own prices
    (``expected_sum_delta``, the properness cost) and can only pay if the records it moves
    resolve on the clipped side MORE often than priced. ``break_even_rate`` is that rate:
    ``-sum(loss) / (sum(gain) - sum(loss))`` over the moved records. A break-even above the
    Jeffreys upper bound on the observed rate rejects the floor even at the optimistic end of
    the uncertainty; ``p_hits_at_most_if_rate_c`` is the binomial chance of seeing at most the
    observed hits if the sub-``c`` band were truly under-priced at rate ``c``.
    """

    question_type: str
    window: str
    c: float
    n_affected: int
    hits: int
    hit_rate: float | None
    ci_lo: float | None
    ci_hi: float | None
    break_even_rate: float | None
    """Binary only: MC moves several options at once, so a single clipped-side rate is
    undefined there."""
    p_hits_at_most_if_rate_c: float | None
    expected_sum_delta: float
    best_case_sum_delta: float
    sum_delta: float

    @property
    def rejected_at_ci_upper(self) -> bool | None:
        if self.break_even_rate is None or self.ci_hi is None:
            return None
        return self.break_even_rate > self.ci_hi

    def to_dict(self) -> dict:
        return asdict(self) | {"rejected_at_ci_upper": self.rejected_at_ci_upper}


def insurance_row(row: SweepRow) -> InsuranceRow:
    """The insurance reading of one floor-only sweep row."""
    rate, lo, hi = jeffreys_interval(row.hits_on_clipped_side, row.n_affected)
    break_even: float | None = None
    p_at_most: float | None = None
    if row.question_type == BINARY and row.n_affected:
        spread = row.best_case_sum_delta - row.worst_case_sum_delta
        break_even = (-row.worst_case_sum_delta / spread) if spread > 0 else None
        p_at_most = binomial_cdf(row.hits_on_clipped_side, row.n_affected, row.c)
    return InsuranceRow(
        question_type=row.question_type,
        window=row.window,
        c=row.c,
        n_affected=row.n_affected,
        hits=row.hits_on_clipped_side,
        hit_rate=rate,
        ci_lo=lo,
        ci_hi=hi,
        break_even_rate=break_even,
        p_hits_at_most_if_rate_c=p_at_most,
        expected_sum_delta=row.expected_sum_delta,
        best_case_sum_delta=row.best_case_sum_delta,
        sum_delta=row.sum_delta,
    )


@dataclass(frozen=True, slots=True)
class NestingRow:
    """How many questions each NESTED window moves at ``c``, and how many distinct ones in all.

    Every nested window is a subset of ``all``, so ``n_distinct`` equals the ``all`` count by
    construction; printing it makes the re-counting visible rather than something a reader
    has to know.
    """

    c: float
    n_affected_by_window: dict[str, int]
    n_distinct: int

    def to_dict(self) -> dict:
        return asdict(self)


def nesting_rows(windows: Sequence[Window], *, grid: Sequence[float]) -> list[NestingRow]:
    rows: list[NestingRow] = []
    nested = [w for w in nested_windows(windows) if w.records]
    for c in grid:
        ids = {w.label: affected_question_ids(w.records, c, side="floor_only") for w in nested}
        union: frozenset[str] = frozenset().union(*ids.values()) if ids else frozenset()
        rows.append(NestingRow(c=c, n_affected_by_window={k: len(v) for k, v in ids.items()}, n_distinct=len(union)))
    return rows


@dataclass(frozen=True, slots=True)
class RegimeSpan:
    """Whether the clamp now in force has BOUND anything since it went live.

    If no publish in the current regime sits at or below the floor or at or above the
    ceiling, the clip question is moot for the live config whatever the pooled tables say,
    and this is the first thing a reader should know.
    """

    window: str
    n: int
    floor: float
    ceiling: float
    min_value: float | None
    max_value: float | None
    n_at_or_below_floor: int
    n_at_or_above_ceiling: int

    def to_dict(self) -> dict:
        return asdict(self)


def regime_span(window: Window, *, question_type: str) -> RegimeSpan:
    """The span of clamped published values inside the current clamp regime."""
    floor, ceiling = in_force_bounds(question_type, window.start)
    values = [v for r in window.records for v in r.clampable_values]
    tol = window.records[0].censor_atol if window.records else 0.0
    return RegimeSpan(
        window=window.label,
        n=len(window.records),
        floor=floor,
        ceiling=ceiling,
        min_value=(min(values) if values else None),
        max_value=(max(values) if values else None),
        n_at_or_below_floor=sum(1 for v in values if v <= floor + tol),
        n_at_or_above_ceiling=sum(1 for v in values if v >= ceiling - tol),
    )


@dataclass(frozen=True, slots=True)
class OosRow:
    """One suffix window's out-of-sample carry: fit on the past, evaluate inside."""

    question_type: str
    window: str
    n_window: int
    n_complement: int
    underpowered: bool
    """The complement is under ``MIN_OOS_COMPLEMENT_N``, so no fit is reported at all."""
    c_star: float | None
    fit_sum_delta: float | None
    fit_n_affected: int | None
    """How many complement records the fitted candidate moved. 0 means the fit is the
    do-nothing map and the carry below is vacuous: nothing was fitted that could fail."""
    carried_sum_delta: float | None
    carried_mean_delta: float | None
    carried_ci_lo: float | None
    carried_ci_hi: float | None
    carried_n_affected: int | None
    in_window_c_star: float | None
    """The in-window argmax, reported beside the carry: a description, not a fit."""
    in_window_sum_delta: float | None
    n_tied_in_window: int
    """How many candidates tie for the in-window argmax; > 1 means ``c_star`` is a plateau."""

    @property
    def fit_is_identity(self) -> bool | None:
        return None if self.fit_n_affected is None else self.fit_n_affected == 0

    @property
    def carry_gap(self) -> float | None:
        """Points the fitted ``c_star`` gave up against the in-window best.

        This, not a ``c_star`` mismatch, is what says whether the fit carried: when several
        candidates tie in-window, the two argmax LABELS disagree while the scores do not.
        """
        if self.carried_sum_delta is None or self.in_window_sum_delta is None:
            return None
        return self.in_window_sum_delta - self.carried_sum_delta

    def to_dict(self) -> dict:
        return asdict(self) | {"fit_is_identity": self.fit_is_identity, "carry_gap": self.carry_gap}


def oos_row(window: Window, *, question_type: str, in_window: Sequence[SweepRow]) -> OosRow:
    """Fit the floor-only argmax on ``window.complement``, then evaluate it inside ``window``.

    ``in_window`` is the window's own floor-only sweep, one row per grid candidate, handed in
    by the caller that already computed it (each row carries a 4000-draw bootstrap, so
    recomputing it here would double the pass's cost); the grid is read off those rows.
    """
    tied_in_window = argmax_rows(in_window)
    best_in_window = tied_in_window[0] if tied_in_window else None
    underpowered = len(window.complement) < MIN_OOS_COMPLEMENT_N
    fitted: SweepRow | None = None
    carried: SweepRow | None = None
    if not underpowered:
        fit_rows = [
            sweep_row(window.complement, question_type=question_type, side="floor_only", window="fit", c=row.c)
            for row in in_window
        ]
        fitted = argmax_row(fit_rows)
        if fitted is not None:
            carried = next(row for row in in_window if row.c == fitted.c)
    return OosRow(
        question_type=question_type,
        window=window.label,
        n_window=len(window.records),
        n_complement=len(window.complement),
        underpowered=underpowered,
        c_star=(fitted.c if fitted else None),
        fit_sum_delta=(fitted.sum_delta if fitted else None),
        fit_n_affected=(fitted.n_affected if fitted else None),
        carried_sum_delta=(carried.sum_delta if carried else None),
        carried_mean_delta=(carried.mean_delta if carried else None),
        carried_ci_lo=(carried.ci_lo if carried else None),
        carried_ci_hi=(carried.ci_hi if carried else None),
        carried_n_affected=(carried.n_affected if carried else None),
        in_window_c_star=(best_in_window.c if best_in_window else None),
        in_window_sum_delta=(best_in_window.sum_delta if best_in_window else None),
        n_tied_in_window=len(tied_in_window),
    )


@dataclass(frozen=True, slots=True)
class ReplayCohort:
    """The per-window facts about the replay that do not depend on the candidate.

    Stored once per window rather than once per ``(window, c)`` cell, so the JSON carries
    each number once and the renderer reads them off the window instead of off whichever
    candidate happens to be first on the grid.
    """

    question_type: str
    window: str
    n_records: int
    n_replayable: int
    """Non-stacked records with recovered members; the only ones the replay can price."""
    n_mean_aggregator: int
    """Replayable records whose published value is the members' MEAN (the 2025-09 era);
    they are replayed with the mean, so the aggregator gap is never charged to the clip."""
    n_unknown_aggregator: int
    """Replayable records neither the median nor the mean of the members rebuilds; replayed
    as a median against their own baseline and counted in ``n_baseline_mismatch``."""
    n_even_members: int
    """Replayable records with an even member count: the case where clamping the published
    median is only approximate, because the publish averages the two middle members."""
    n_baseline_mismatch: int
    """Records whose replay does not reproduce the PUBLISHED value with no clip applied. With
    aggregator detection this is the ``unknown`` residue; each path is still measured against
    its OWN unclipped baseline so recovery error never lands on a candidate that moves nothing."""
    max_baseline_gap: float

    def to_dict(self) -> dict:
        return asdict(self)


def _baseline_gap(record: ClipRecord) -> float:
    """|replayed resolving mass with NO clip - published value|, for a replayable record."""
    baseline = record.replay(record.in_force_lo, record.in_force_hi)
    return abs(baseline[record.resolving_index] - record.published_resolving_mass)


def replay_cohort(records: Sequence[ClipRecord], *, question_type: str, window: str) -> ReplayCohort:
    """The candidate-independent replay facts for one window (see :class:`ReplayCohort`)."""
    replayable = [r for r in records if r.replayable]
    gaps = [_baseline_gap(r) for r in replayable]
    return ReplayCohort(
        question_type=question_type,
        window=window,
        n_records=len(records),
        n_replayable=len(replayable),
        n_mean_aggregator=sum(1 for r in replayable if r.aggregator == "mean"),
        n_unknown_aggregator=sum(1 for r in replayable if r.aggregator == "unknown"),
        n_even_members=sum(1 for r in replayable if len(r.members) % 2 == 0),
        n_baseline_mismatch=sum(1 for gap in gaps if gap > REPLAY_DISAGREE_ATOL),
        max_baseline_gap=max(gaps, default=0.0),
    )


@dataclass(frozen=True, slots=True)
class CrossCheckRow:
    """The published-vector shortcut against a full per-model replay, at one ``c``.

    Only what depends on the candidate lives here; the per-window replay facts are on the
    window's :class:`ReplayCohort`.
    """

    question_type: str
    window: str
    c: float
    n_disagree: int
    """Records whose two counterfactual resolving masses differ by > REPLAY_DISAGREE_ATOL."""
    max_abs_gap: float
    n_routing_flips: int | None
    """Binary only: records whose member spread clears the CONDITIONAL_STACKING threshold as
    published (a strict ``>``, the live route's test) but would not under the candidate clamp,
    so the stacking route the counterfactual holds fixed would in fact have changed."""
    sum_delta_published: float
    sum_delta_replay: float
    """The clip's effect along the replay path, each record against its own unclipped baseline."""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _ReplayReading:
    """One record's two counterfactual paths at one ``c``, each against its own baseline."""

    counterfactual_gap: float
    """|replayed resolving mass - published-vector resolving mass| at this ``c``."""
    delta_published: float
    delta_replay: float
    routing_flip: bool


def _routing_flip(record: ClipRecord, lo: float, hi: float) -> bool:
    """Would the candidate clamp pull a stacker-routed binary spread under the threshold?

    Uses the live spread metric and the live route's comparison: ``stacking_route`` skips the
    stacker when ``spread <= threshold``, so a record routes only on a STRICT exceedance.
    """
    if not record.is_binary or len(record.members) < 2:
        return False
    raw = [m[1] for m in record.members]
    clamped = [min(hi, max(lo, v)) for v in raw]
    threshold = CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD
    return binary_prob_range_spread(raw) > threshold and binary_prob_range_spread(clamped) <= threshold


def _replay_reading(record: ClipRecord, c: float) -> _ReplayReading | None:
    """Both paths for one record, or None when it cannot be replayed."""
    if not record.replayable:
        return None
    lo = max(c, record.in_force_lo)
    hi = record.in_force_hi
    replayed = record.replay(lo, hi)
    baseline = record.replay(record.in_force_lo, record.in_force_hi)
    index = record.resolving_index
    published_cf = apply_bounds(record, lo, hi)[index]
    published_mass = record.published_resolving_mass
    return _ReplayReading(
        counterfactual_gap=abs(replayed[index] - published_cf),
        delta_published=spot_peer_delta(
            old_prob=published_mass, new_prob=published_cf, question_type=record.question_type
        ),
        delta_replay=spot_peer_delta(
            old_prob=baseline[index], new_prob=replayed[index], question_type=record.question_type
        ),
        routing_flip=_routing_flip(record, lo, hi),
    )


def cross_check_row(
    records: Sequence[ClipRecord],
    *,
    question_type: str,
    window: str,
    c: float,
) -> CrossCheckRow:
    """Compare the published-vector counterfactual with the per-model replay at ``c``.

    Only non-stacked records with recoverable members are replayable: a fired stacker's
    published value is not a member median, so replaying members there would price a
    forecast the bot never made.

    Each record is replayed under the aggregator that rebuilds its published value (median,
    or mean for the 2025-09 era), and each path is measured against its OWN unclipped
    baseline, so the two ``sum_delta`` columns price the same thing.
    """
    readings = [r for r in (_replay_reading(record, c) for record in records) if r is not None]
    return CrossCheckRow(
        question_type=question_type,
        window=window,
        c=c,
        n_disagree=sum(1 for r in readings if r.counterfactual_gap > REPLAY_DISAGREE_ATOL),
        max_abs_gap=max((r.counterfactual_gap for r in readings), default=0.0),
        n_routing_flips=(sum(1 for r in readings if r.routing_flip) if question_type == BINARY else None),
        sum_delta_published=float(sum(r.delta_published for r in readings)),
        sum_delta_replay=float(sum(r.delta_replay for r in readings)),
    )


@dataclass(frozen=True, slots=True)
class CrossCheckSummary:
    """The sign of the shortcut's error, over every ``(window, c)`` cell of one type."""

    n_differing: int
    """Cells where the replay path and the published-vector path differ by more than ``DELTA_ATOL``."""
    n_replay_more_negative: int
    """Of those, the cells where the replay path is the more negative of the two."""

    def to_dict(self) -> dict:
        return asdict(self)


def cross_check_summary(rows: Sequence[CrossCheckRow]) -> CrossCheckSummary:
    differing = [r for r in rows if abs(r.sum_delta_replay - r.sum_delta_published) > DELTA_ATOL]
    return CrossCheckSummary(
        n_differing=len(differing),
        n_replay_more_negative=sum(1 for r in differing if r.sum_delta_replay < r.sum_delta_published),
    )


@dataclass(frozen=True, slots=True)
class ArgmaxSelection:
    """Which candidates won one window's argmax, and which censored ones tied and lost on censoring."""

    side: str
    window: str
    argmax_c: float | None
    """The smallest-``c`` exact winner; None when no exact row exists."""
    tied_c: tuple[float, ...]
    """Every exact candidate within ``ARGMAX_TIE_ATOL`` of the best, increasing; the plateau."""
    censored_tie_c: tuple[float, ...]
    """Censored candidates at the same score, so a win-by-exclusion is visible."""

    def to_dict(self) -> dict:
        return asdict(self)


def argmax_selection(rows: Sequence[SweepRow], *, side: str, window: str) -> ArgmaxSelection:
    tied = argmax_rows(rows)
    return ArgmaxSelection(
        side=side,
        window=window,
        argmax_c=(tied[0].c if tied else None),
        tied_c=tuple(row.c for row in tied),
        censored_tie_c=tuple(row.c for row in censored_rows_at_argmax_score(rows)),
    )


@dataclass(frozen=True, slots=True)
class ThinFloorClip:
    """One genuine single-forecaster publish priced under the thin publish floor."""

    question_id: str
    created_at: str | None
    p_yes: float
    resolved_yes: bool
    clamped_p_yes: float
    delta: float
    spot_peer: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ThinFloorReport:
    """The single-survivor cohort and what ``[THIN_PUBLISH_BINARY_FLOOR, _CEIL]`` is worth on it.

    ``apply_thin_publish_floor`` fires only when exactly one forecaster survived, so its cost
    is the sum over THESE records and nothing else. A publish is a genuine single-survivor
    one only from the merge that lowered ``MIN_FORECASTERS_TO_PUBLISH`` to 1 (``b4e9df0``);
    a one-member record before that is a trimmed-comment parse artifact and is counted, not
    priced. The dataset carries no ``stacker_skip_reason`` marker, so the member count plus
    the merge timestamp is the only available detector.
    """

    floor: float
    ceiling: float
    boundary: str
    rows: tuple[ThinFloorClip, ...]
    n_single_member_before_boundary: int

    @property
    def sum_delta(self) -> float:
        return float(sum(row.delta for row in self.rows))

    def to_dict(self) -> dict:
        return asdict(self) | {"sum_delta": self.sum_delta}


def single_survivor_report(records: Sequence[ClipRecord]) -> ThinFloorReport:
    """Price the thin publish floor on every genuine single-forecaster binary publish.

    The clamp is the LIVE ``apply_thin_publish_floor``, handed the record's own member count,
    so a change to the rule's functional form (or, with data, to its ``== 1`` trigger) prices
    itself here rather than being mirrored by hand.
    """
    single = [r for r in records if r.is_binary and r.replayable and len(r.members) == 1]
    genuine = [r for r in single if r.created_at is not None and r.created_at >= B4E9DF0_MERGED_AT]
    rows: list[ThinFloorClip] = []
    for record in genuine:
        p_yes = record.published[1]
        clamped = apply_thin_publish_floor(p_yes, len(record.members))
        resolved_yes = record.resolving_index == 1
        rows.append(
            ThinFloorClip(
                question_id=record.question_id,
                created_at=(record.created_at.isoformat() if record.created_at else None),
                p_yes=p_yes,
                resolved_yes=resolved_yes,
                clamped_p_yes=clamped,
                delta=spot_peer_delta(
                    old_prob=record.published_resolving_mass,
                    new_prob=(clamped if resolved_yes else 1.0 - clamped),
                    question_type=BINARY,
                ),
                spot_peer=record.spot_peer,
            )
        )
    return ThinFloorReport(
        floor=THIN_PUBLISH_BINARY_FLOOR,
        ceiling=THIN_PUBLISH_BINARY_CEIL,
        boundary=B4E9DF0_MERGED_AT.isoformat(),
        rows=tuple(rows),
        n_single_member_before_boundary=len(single) - len(genuine),
    )


@dataclass(frozen=True, slots=True)
class TypeReport:
    """Everything the sweep says about one question type."""

    question_type: str
    cohort: ClipCohort
    grid: tuple[float, ...]
    windows: tuple[Window, ...]
    regime_span: RegimeSpan | None
    older_regime: SweepRow | None
    """The floor now in force priced on the records published BEFORE it went live (the
    current-regime window's complement, at that type's own in-force floor): the one floor
    comparison the archive measures rather than bounds. None when no such records exist."""
    extreme_bins: tuple[ExtremeBin, ...]
    sweep: tuple[SweepRow, ...]
    argmax: tuple[ArgmaxSelection, ...]
    insurance: tuple[InsuranceRow, ...]
    nesting: tuple[NestingRow, ...]
    oob: tuple[OobArgmax, ...]
    oos: tuple[OosRow, ...]
    replay_cohorts: tuple[ReplayCohort, ...]
    cross_check: tuple[CrossCheckRow, ...]
    cross_check_summary: CrossCheckSummary
    thin_floor: ThinFloorReport | None

    @property
    def populated_windows(self) -> list[Window]:
        return [w for w in self.windows if w.records]

    def sweep_rows(self, *, side: str, window: str) -> list[SweepRow]:
        return [row for row in self.sweep if row.side == side and row.window == window]

    def argmax_for(self, *, side: str, window: str) -> ArgmaxSelection:
        return next(sel for sel in self.argmax if sel.side == side and sel.window == window)

    def insurance_rows(self, window: str) -> list[InsuranceRow]:
        return [row for row in self.insurance if row.window == window]

    def extreme_bins_for(self, window: str) -> list[ExtremeBin]:
        return [b for b in self.extreme_bins if b.window == window]

    def replay_cohort_for(self, window: str) -> ReplayCohort:
        return next(cohort for cohort in self.replay_cohorts if cohort.window == window)

    def to_dict(self) -> dict:
        return {
            **self.cohort.to_dict(),
            "grid": list(self.grid),
            "windows": [w.to_dict() for w in self.windows],
            "regime_span": (self.regime_span.to_dict() if self.regime_span else None),
            "older_regime": (self.older_regime.to_dict() if self.older_regime else None),
            "extreme_bins": [b.to_dict() for b in self.extreme_bins],
            "sweep": [row.to_dict() for row in self.sweep],
            "argmax": [sel.to_dict() for sel in self.argmax],
            "insurance": [row.to_dict() for row in self.insurance],
            "nesting": [row.to_dict() for row in self.nesting],
            "oob": [row.to_dict() for row in self.oob],
            "oos": [row.to_dict() for row in self.oos],
            "replay_cohorts": [cohort.to_dict() for cohort in self.replay_cohorts],
            "cross_check": [row.to_dict() for row in self.cross_check],
            "cross_check_summary": self.cross_check_summary.to_dict(),
            "thin_floor": (self.thin_floor.to_dict() if self.thin_floor else None),
        }


@dataclass(frozen=True, slots=True)
class ClipSweepReport:
    """The whole pass: one :class:`TypeReport` per question type, plus provenance."""

    dataset_path: str
    as_of: datetime
    exclude_qids: frozenset[str]
    n_excluded: int
    types: tuple[TypeReport, ...]

    def type_report(self, question_type: str) -> TypeReport:
        return next(report for report in self.types if report.question_type == question_type)

    def to_dict(self) -> dict:
        payload: dict = {
            "meta": {
                "dataset_path": self.dataset_path,
                "as_of": self.as_of.isoformat(),
                "exclude_qids": sorted(self.exclude_qids),
                "n_excluded": self.n_excluded,
                "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "cl": BOOTSTRAP_CL, "oob_B": OOB_BOOTSTRAP_B},
                "lookback_days": LOOKBACK_DAYS,
                "min_oos_complement_n": MIN_OOS_COMPLEMENT_N,
            }
        }
        for report in self.types:
            payload[report.question_type] = report.to_dict()
        return payload


def _older_regime_row(current: Window | None, *, question_type: str) -> SweepRow | None:
    """Today's floor priced on the records published under the older clamp (see :class:`TypeReport`).

    Keyed on the current-regime window's COMPLEMENT rather than on ``era_pre_flip``: the two
    coincide for binary, whose clamp changed at the widening flip, but the MC clamp changed at
    the ft 0.2.92 unfreeze two months later, so for MC the older regime also holds the
    post-flip records and the first days of the triple era.
    """
    if current is None or not current.complement:
        return None
    floor, _ = in_force_bounds(question_type, current.start)
    return sweep_row(
        current.complement, question_type=question_type, side="floor_only", window=WINDOW_OLDER_REGIME, c=floor
    )


def _type_report(data: Sequence[dict], question_type: str, *, as_of: datetime) -> TypeReport:
    cohort = build_clip_records(data, question_type)
    grid = GRID_BY_TYPE[question_type]
    windows = build_windows(cohort.records, question_type=question_type, as_of=as_of)
    populated = [w for w in windows if w.records]
    build_bins = binary_extreme_bins if question_type == BINARY else mc_extreme_bins
    sweep = tuple(
        sweep_row(w.records, question_type=question_type, side=side, window=w.label, c=c)
        for w in populated
        for side in SIDES
        for c in grid
    )
    by_side_window = {
        (side, w.label): [row for row in sweep if row.side == side and row.window == w.label]
        for side in SIDES
        for w in populated
    }
    floor_rows = {w.label: by_side_window["floor_only", w.label] for w in populated}
    current = next((w for w in windows if w.label == WINDOW_CURRENT_CLAMP), None)
    cross_check = tuple(
        cross_check_row(w.records, question_type=question_type, window=w.label, c=c) for w in populated for c in grid
    )
    return TypeReport(
        question_type=question_type,
        cohort=cohort,
        grid=grid,
        windows=tuple(windows),
        regime_span=(regime_span(current, question_type=question_type) if current else None),
        older_regime=_older_regime_row(current, question_type=question_type),
        extreme_bins=tuple(b for w in populated for b in build_bins(w.records, window=w.label)),
        sweep=sweep,
        argmax=tuple(
            argmax_selection(by_side_window[side, w.label], side=side, window=w.label)
            for side in SIDES
            for w in populated
        ),
        insurance=tuple(insurance_row(row) for w in populated for row in floor_rows[w.label]),
        nesting=tuple(nesting_rows(windows, grid=grid)),
        oob=tuple(oob_argmax(w.records, floor_rows[w.label], side="floor_only", window=w.label) for w in populated),
        oos=tuple(
            oos_row(w, question_type=question_type, in_window=floor_rows[w.label])
            for w in populated
            if w.label != WINDOW_ALL
        ),
        replay_cohorts=tuple(replay_cohort(w.records, question_type=question_type, window=w.label) for w in populated),
        cross_check=cross_check,
        cross_check_summary=cross_check_summary(cross_check),
        thin_floor=(single_survivor_report(cohort.records) if question_type == BINARY else None),
    )


def compute_report(
    data: Sequence[dict],
    *,
    dataset_path: str,
    as_of: datetime,
    exclude_qids: frozenset[str],
) -> ClipSweepReport:
    """Run the whole sweep over a cached dataset, after applying ``exclude_qids``."""
    kept = [r for r in data if str(r.get("question_id")) not in exclude_qids]
    n_excluded = len(data) - len(kept)
    if exclude_qids:
        logger.info(f"--exclude-qids: {len(exclude_qids)} requested id(s), {n_excluded} matched a record in this pull")
    return ClipSweepReport(
        dataset_path=dataset_path,
        as_of=as_of,
        exclude_qids=exclude_qids,
        n_excluded=n_excluded,
        types=tuple(_type_report(kept, qtype, as_of=as_of) for qtype in QUESTION_TYPES),
    )
