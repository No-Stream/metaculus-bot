"""Reusable analysis functions for performance data."""

import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from scipy.stats import beta, spearmanr

from metaculus_bot.numeric.config import MAX_CDF_PROB_STEP, grid_step_constraints
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.performance_analysis.parsing import (
    MIN_SCOREABLE_ANCHORS,
    _parse_probability,
    declared_anchors,
    is_anonymous_model_key,
)
from metaculus_bot.performance_analysis.platform_scores import (
    baseline_score,
    coverage,
    peer_score,
    spot_baseline_score,
    spot_peer_score,
)
from metaculus_bot.performance_analysis.scaling import grid_zero_point
from metaculus_bot.performance_analysis.scoring import binary_log_score, brier_score
from metaculus_bot.performance_analysis.stacker_detection import detect_stacker_fired
from metaculus_bot.spread_metrics import binary_prob_range_spread
from metaculus_bot.time_utils import parse_iso_utc

logger: logging.Logger = logging.getLogger(__name__)

# Type alias: given a list of per-model binary probabilities, return a spread scalar.
BinarySpreadFn = Callable[[list[float]], float]

# Calibration bucket edges for binary questions
CALIBRATION_BUCKET_EDGES: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Case-insensitive substrings matched against the question's category tag.
_FINANCIAL_CATEGORY_SUBSTRINGS: tuple[str, ...] = (
    "finance",
    "economy",
    "business",
    "markets",
    "stock",
)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _score_stats(values: list[float]) -> dict:
    """``{count, mean, median}`` for one platform-score field; mean/median None when empty."""
    return {
        "count": len(values),
        "mean": _mean(values),
        "median": float(np.median(values)) if values else None,
    }


# The one registry for the four platform-score metrics: summary key, accessor, report
# label, in report order. platform_score_summary, the section's emptiness gate and its
# render loop all iterate this, so adding a metric in one place reaches all three
# (previously a `fields` dict here and a `_PLATFORM_SCORE_ROWS` 730 lines apart could
# drift, silently dropping a metric from the report). ``coverage`` is handled separately.
_PLATFORM_SCORE_METRICS: tuple[tuple[str, Callable[[dict], float | None], str], ...] = (
    ("spot_peer", spot_peer_score, "spot peer (PRIMARY, the leaderboard metric)"),
    ("peer", peer_score, "peer (coverage-scaled, secondary)"),
    ("spot_baseline", spot_baseline_score, "spot baseline (primary)"),
    ("baseline", baseline_score, "baseline (coverage-scaled, secondary)"),
)


def platform_score_summary(data: list[dict]) -> dict:
    """Metaculus's own scores on the published forecasts, SPOT peer first.

    The tournament leaderboard ranks on ``spot_peer_score``. ``peer_score`` is the same
    quantity scaled by coverage, which for a bot that submits once and never revises is
    mostly a function of how early it submitted — so it is reported as a labelled
    secondary and never used to rank. Same split for the two baseline scores. Full
    reasoning and the accessors live in ``performance_analysis.platform_scores``.

    Every field is counted over the records that actually carry it, so the per-metric
    ``count`` differs from ``count`` (all records) whenever a pull predates score capture.
    """
    summary: dict = {"count": len(data)}
    for name, reader, _label in _PLATFORM_SCORE_METRICS:
        summary[name] = _score_stats([v for v in (reader(r) for r in data) if v is not None])
    summary["mean_coverage"] = _mean([v for v in (coverage(r) for r in data) if v is not None])
    return summary


def per_model_cohort(data: list[dict], *, cut: str) -> list[tuple[dict, dict]]:
    """Return ``(record, per_model_forecasts)`` pairs valid for a per-model cut.

    ``per_model_forecasts`` is only a per-MODEL record on questions the bot
    published as an aggregate of individually-attributed base forecasts. Two
    kinds of entry break that, and both are dropped here:

    * **Stacker-fired records.** When the stacker LLM produced the published
      value, the bot writes ONE summary bullet holding the stacker's aggregate.
      Bucketing it under whatever key that bullet carries makes a per-model
      score that is really a stacker-vs-base-model mixture. Detection is
      ``detect_stacker_fired(record) == "confirmed_stacker"`` — the verdict
      backed by an explicit flag, marker, or body signature. ``likely_stacker``
      is deliberately NOT excluded: it is a spread-plus-delta heuristic that
      fires on any high-spread question whose published value sits far from the
      median, which is exactly what a MEAN-era aggregate looks like, so
      honoring it would drop the high-disagreement records these cuts exist to
      measure.
    * **Anonymous attribution keys.** ``Forecaster N`` keys are positional
      fallbacks, assigned when neither an explicit roster nor a ``Model:`` line
      identified the forecaster (``parsing.anonymous_model_key``). Pooled across
      questions, one positional bucket spans different models — and on
      stacking-era comments it is the stacker's own aggregate, which is how 50
      such forecasts reached the 2026-04 per-model cuts as if they were two
      extra ensemble members.

    Both exclusion counts are logged at INFO under the ``PER_MODEL_COHORT``
    marker, keyed by ``cut``, so a shrunken cohort is visible in the run log
    rather than passing for full coverage.

    Records are returned even when every key was dropped; callers already handle
    an empty per-model dict (they need <2 parseable values to mean "no spread").
    """
    cohort: list[tuple[dict, dict]] = []
    excluded_stacked_records = 0
    excluded_stacked_observations = 0
    excluded_anonymous_observations = 0

    for record in data:
        per_model = record.get("per_model_forecasts") or {}
        if detect_stacker_fired(record) == "confirmed_stacker":
            excluded_stacked_records += 1
            excluded_stacked_observations += len(per_model)
            continue
        attributed = {name: value for name, value in per_model.items() if not is_anonymous_model_key(name)}
        excluded_anonymous_observations += len(per_model) - len(attributed)
        cohort.append((record, attributed))

    logger.info(
        "PER_MODEL_COHORT: cut=%s eligible_records=%d excluded_stacked_records=%d "
        "excluded_stacked_observations=%d excluded_anonymous_observations=%d "
        "reason=a stacked record's per-model slot holds the stacker aggregate, and an "
        "anonymous Forecaster-N key is a positional bucket, not one model",
        cut,
        len(cohort),
        excluded_stacked_records,
        excluded_stacked_observations,
        excluded_anonymous_observations,
    )
    return cohort


def binary_summary(data: list[dict]) -> dict:
    """Compute summary statistics for binary questions.

    Returns dict with mean_brier, mean_log_score, calibration_buckets,
    direction_accuracy, base_rate_brier, and count.
    """
    binary = [r for r in data if r["type"] == "binary" and r["brier_score"] is not None]
    if not binary:
        return {"count": 0}

    brier_scores = [r["brier_score"] for r in binary]
    log_scores = [r["log_score"] for r in binary]

    # Calibration buckets
    buckets: dict[str, dict] = {}
    edges = CALIBRATION_BUCKET_EDGES
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        label = f"{low:.1f}-{high:.1f}"
        in_bucket = [r for r in binary if low <= r["our_prob_yes"] < high or (high == 1.0 and r["our_prob_yes"] == 1.0)]
        if in_bucket:
            actual_rate = sum(1 for r in in_bucket if r["resolution_parsed"] is True) / len(in_bucket)
            buckets[label] = {
                "predicted_mean": _mean([r["our_prob_yes"] for r in in_bucket]),
                "actual_rate": actual_rate,
                "count": len(in_bucket),
            }

    # Direction accuracy: did we predict >0.5 for Yes outcomes and <0.5 for No?
    correct_direction = sum(
        1
        for r in binary
        if (r["our_prob_yes"] >= 0.5 and r["resolution_parsed"] is True)
        or (r["our_prob_yes"] < 0.5 and r["resolution_parsed"] is False)
    )
    direction_accuracy = correct_direction / len(binary)

    # Base rate comparison: what Brier would we get always predicting the base rate?
    base_rate = sum(1 for r in binary if r["resolution_parsed"] is True) / len(binary)
    base_rate_brier = _mean([brier_score(base_rate, r["resolution_parsed"]) for r in binary])

    return {
        "count": len(binary),
        "mean_brier": _mean(brier_scores),
        "mean_log_score": _mean(log_scores),
        "calibration_buckets": buckets,
        "direction_accuracy": direction_accuracy,
        "base_rate": base_rate,
        "base_rate_brier": base_rate_brier,
    }


def per_model_binary_scores(data: list[dict]) -> dict[str, dict]:
    """Compute per-model Brier and log scores for binary questions.

    Returns dict mapping model name to {mean_brier, mean_log_score, count}.
    Only includes questions where the model's per-model forecast is parseable as a percentage.

    Restricted to the per-model cohort (see ``per_model_cohort``): stacker-fired
    records and anonymous ``Forecaster N`` keys are excluded, so every bucket
    here is one named model.
    """
    binary = [r for r in data if r["type"] == "binary" and isinstance(r["resolution_parsed"], bool)]

    model_scores: dict[str, list[tuple[float, float]]] = {}
    for r, per_model in per_model_cohort(binary, cut="per_model_binary_scores"):
        outcome = r["resolution_parsed"]
        for model_name, raw_value in per_model.items():
            prob = _parse_probability(raw_value)
            if prob is None:
                continue
            if model_name not in model_scores:
                model_scores[model_name] = []
            model_scores[model_name].append(
                (
                    brier_score(prob, outcome),
                    binary_log_score(prob, outcome),
                )
            )

    result: dict[str, dict] = {}
    for model_name, scores in sorted(model_scores.items()):
        briers = [s[0] for s in scores]
        logs = [s[1] for s in scores]
        result[model_name] = {
            "mean_brier": _mean(briers),
            "mean_log_score": _mean(logs),
            "count": len(scores),
        }
    return result


OUT_OF_RANGE_MARKER_SIDES: Mapping[str, str] = {"below_lower_bound": "low", "above_upper_bound": "high"}


@dataclass(frozen=True, slots=True)
class PitReading:
    """One record's PIT reading: a point value, or an INTERVAL of possible values.

    Metaculus reports a resolution past the displayed range as the string
    ``above_upper_bound`` / ``below_lower_bound``, so the resolution VALUE is unknown and
    ``F(resolution)`` is only pinned to a SET: ``[cdf[-1], 1]`` above the ceiling,
    ``[0, cdf[0]]`` below the floor. On an open bound that set can be wide, because our own
    CDF is free to put real mass out there — q44842 published 13% of its mass above the
    displayed ceiling, resolved ``above_upper_bound``, and won spot peer +24.4, while the old
    convention (PIT := 1.0) scored it a high-side band miss.

    Two conventions ride on this type, and they differ deliberately:

    * COVERAGE counts an interval as covered when it INTERSECTS the band — a band miss only
      when the WHOLE interval lies outside it.
    * POINT statistics (mean, std, histogram) EXCLUDE intervals and disclose how many were
      excluded. Imputing a midpoint would manufacture a reading nobody measured.

    An interval whose endpoints coincide (a closed bound, or an open one carrying no
    out-of-range mass) IS a point reading: ``is_interval`` is False and ``point`` answers the
    same value the old convention forced, so nothing changes on those records.
    """

    low: float
    high: float
    oob_side: str | None = None
    """``"low"``/``"high"`` when the resolution fell beyond the value grid, else None."""

    @classmethod
    def from_point(cls, value: float, oob_side: str | None = None) -> "PitReading":
        return cls(low=value, high=value, oob_side=oob_side)

    @property
    def is_interval(self) -> bool:
        return self.high > self.low

    @property
    def point(self) -> float | None:
        """The PIT when it is a single value; None for an interval reading."""
        return None if self.is_interval else self.low

    def intersects(self, band_low: float, band_high: float) -> bool:
        """Coverage predicate: does any of this reading fall inside ``[band_low, band_high]``?"""
        return self.high >= band_low and self.low <= band_high

    def at_or_below(self, threshold: float) -> bool:
        """Cumulative coverage ``PIT <= threshold`` — the band ``[0, threshold]``."""
        return self.low <= threshold

    def entirely_below(self, threshold: float) -> bool:
        return self.high < threshold

    def entirely_above(self, threshold: float) -> bool:
        return self.low > threshold


def out_of_range_pit_reading(resolution: object, cdf_values: Sequence[float] | np.ndarray) -> PitReading | None:
    """The interval a STRING out-of-range resolution pins ``F(resolution)`` to.

    Returns None for anything else (a numeric resolution, an annulled question), so callers
    read it as "is this the set-valued case?" and fall through otherwise. THE single home of
    the convention: ``numeric_pit_analysis`` here and ``width_monitor.compute_pit_reading``
    both go through it.
    """
    side = OUT_OF_RANGE_MARKER_SIDES.get(resolution) if isinstance(resolution, str) else None
    if side is None:
        return None
    if side == "high":
        return PitReading(low=float(cdf_values[-1]), high=1.0, oob_side="high")
    return PitReading(low=0.0, high=float(cdf_values[0]), oob_side="low")


def pit_band_count(readings: Sequence[PitReading], band_low: float, band_high: float) -> int:
    """How many readings the band covers (an interval counts when it intersects)."""
    return sum(1 for reading in readings if reading.intersects(band_low, band_high))


def pit_point_values(readings: Sequence[PitReading]) -> list[float]:
    """The point PITs, dropping the set-valued readings that point statistics exclude."""
    return [value for value in (reading.point for reading in readings) if value is not None]


def numeric_pit_analysis(data: list[dict]) -> dict:
    """PIT (Probability Integral Transform) analysis for numeric questions.

    Coverage is computed over every reading, with an out-of-range resolution counted as
    covered when its PIT INTERVAL intersects the band (see :class:`PitReading`). The
    histogram and ``pit_values`` are POINT statistics and exclude interval readings;
    ``n_oob_interval`` discloses how many were excluded, so a shrinking histogram is never
    silent.

    Keys: ``count`` (readings), ``n_point`` / ``n_oob_interval`` (its split),
    ``pit_values`` (point readings only), ``pit_intervals`` (the set-valued ones as
    ``(low, high)``), ``histogram``, ``coverage_90`` / ``coverage_50``,
    ``mean_numeric_log_score``.
    """
    numeric = [r for r in data if r["type"] in ("numeric", "discrete") and r["numeric_log_score"] is not None]
    if not numeric:
        return {"count": 0}

    readings: list[PitReading] = []
    for r in numeric:
        cdf_values = r["our_forecast_values"]
        resolution = r["resolution_parsed"]
        scaling = r["scaling"]
        lower_bound = scaling.get("range_min")
        upper_bound = scaling.get("range_max")

        if lower_bound is None or upper_bound is None or cdf_values is None:
            continue

        lower_bound = float(lower_bound)
        upper_bound = float(upper_bound)

        out_of_range = out_of_range_pit_reading(resolution, cdf_values)
        if out_of_range is not None:
            readings.append(out_of_range)
        elif isinstance(resolution, (int, float)) and not isinstance(resolution, bool):
            if upper_bound - lower_bound <= 0:
                # No range, no interpolated PIT — the record is dropped rather than
                # contributing the 0.5 that used to come back from _interpolate_pit, which
                # is the single most favorable value available (inside BOTH coverage bands).
                # Scoped to the interpolated branch: an out-of-range resolution's interval is
                # a real reading that never touched the range.
                continue
            readings.append(
                PitReading.from_point(
                    _interpolate_pit(
                        float(resolution),
                        lower_bound,
                        upper_bound,
                        cdf_values,
                        value_grid=scaling.get("continuous_range"),
                        zero_point=grid_zero_point(scaling.get("zero_point"), lower_bound),
                        per_model_percentiles=r.get("per_model_numeric_percentiles"),
                    )
                )
            )

    if not readings:
        return {"count": 0}

    point_values = pit_point_values(readings)
    num_bins = 10
    histogram = [0] * num_bins
    for pit in point_values:
        bin_idx = min(int(pit * num_bins), num_bins - 1)
        histogram[bin_idx] += 1

    # Coverage: fraction of readings the band covers (~90% / ~50% when well-calibrated).
    coverage_90 = pit_band_count(readings, 0.05, 0.95) / len(readings)
    coverage_50 = pit_band_count(readings, 0.25, 0.75) / len(readings)

    log_scores = [r["numeric_log_score"] for r in numeric]

    return {
        "count": len(readings),
        "n_point": len(point_values),
        "n_oob_interval": len(readings) - len(point_values),
        "pit_values": point_values,
        "pit_intervals": [(reading.low, reading.high) for reading in readings if reading.is_interval],
        "histogram": histogram,
        "coverage_90": coverage_90,
        "coverage_50": coverage_50,
        "mean_numeric_log_score": _mean(log_scores),
    }


def declared_percentile_pit(
    per_model_percentiles: Mapping[str, Sequence[Sequence[float]]] | None,
    resolution: float,
) -> float | None:
    """PIT from the MEDIAN of the ensemble members' declared percentile curves.

    The declared ``(percentile, value)`` pairs are the models' raw output and are NOT
    clipped to the question's displayed range, so a resolution beyond the CDF grid
    still gets a real quantile: interpolate each member's own value -> percentile
    curve at the resolution (clamping to the curve's endpoint percentiles beyond its
    ends) and take the median across members — the same median-of-members the
    published aggregate is built from, read in percentile space where the bound is
    not a wall. Returns None when no member curve is usable.

    Anonymous ``Forecaster N`` keys are EXCLUDED, matching ``max_step_clamp_screen``
    next door and ``per_model_cohort``: on a stacker-fired record that positional
    bucket holds the stacker's AGGREGATE, so pooling it into a median-of-members
    counts the aggregate as an extra member and pulls the median toward itself. The
    two sets don't intersect in today's archive, so this is a latent fix — but the
    two sibling consumers of this field already filter, and one that didn't was how
    the 50-forecast mixture got in.

    Sparse curves are NOT excluded here, deliberately — no ``MIN_SCOREABLE_ANCHORS``
    gate, unlike the ranking/clamp-screen consumers. Their ~96-log-point artifact
    comes from PCHIP-rebuilding a sparse curve onto a full CDF grid and log-scoring
    it; this path only linearly interpolates the declared pairs in percentile space
    for a single quantile, where a 3-anchor curve is coarse but not a fabricated
    distribution, and its member PIT is then medianed against its siblings' rather
    than scored on its own. Gating here would also delete the uniformly-sparse-era
    records (fall-2025 comments declare 8-percentile sets) whose PITs are valid.
    """
    curves = {
        model: pairs for model, pairs in (per_model_percentiles or {}).items() if not is_anonymous_model_key(str(model))
    }
    pits = [pit for pit in (_single_curve_pit(pairs, resolution) for pairs in curves.values()) if pit is not None]
    return float(np.median(pits)) if pits else None


def _single_curve_pit(percentile_pairs: Sequence[Sequence[float]], resolution: float) -> float | None:
    """Interpolate one member's declared (percentile 0-100, value) curve at ``resolution``."""
    if len(percentile_pairs) < 2:
        # A single recovered pair interpolates to a constant PIT at every resolution
        # (trimmed comments can lose most of a member's declared lines) — unusable.
        return None
    try:
        pcts = np.array([float(p[0]) / 100.0 for p in percentile_pairs], dtype=float)
        vals = np.array([float(p[1]) for p in percentile_pairs], dtype=float)
    except (TypeError, ValueError, IndexError):
        # Archived per-model pairs are parsed from comment text and can be malformed.
        return None
    order = np.argsort(vals, kind="stable")
    vals, pcts = vals[order], pcts[order]
    if np.any(np.diff(pcts) < 0):
        # Percentiles that DECREASE as values increase after the value sort mean the
        # member declared a non-monotonic set; interpolating it inverts the curve.
        return None
    if not np.all(np.diff(vals) > 0):
        # Duplicate declared values (e.g. a flat tail): jitter into strict monotonicity.
        eps = max(1e-9, float(np.abs(vals).max()) * 1e-9)
        vals = vals + np.arange(len(vals)) * eps
        if not np.all(np.diff(vals) > 0):
            return None
    return float(np.interp(resolution, vals, pcts))


def jeffreys_ci(k: int, n: int, cl: float = 0.95) -> tuple[float, float, float]:
    """Beta-Binomial posterior mean + equal-tailed CI under a Jeffreys(0.5, 0.5) prior.

    The one implementation for every ``k of n`` rate this package prints (the width monitor's
    coverage columns, the clip sweep's extreme-bin and insurance intervals), so the prior and
    the tail convention cannot drift between two residual tables. Mirrors ``bb`` in
    mc_numeric_calibration.py.
    """
    a = 0.5 + k
    b = 0.5 + (n - k)
    mean = a / (a + b)
    lo = float(beta.ppf((1 - cl) / 2, a, b))
    hi = float(beta.ppf(1 - (1 - cl) / 2, a, b))
    return mean, lo, hi


# ``b4e9df0`` — the merge that landed the july15 bundle on main. THE single source of
# truth for this era boundary across the package: width_monitor's ``TS_ANCHOR_ENABLE``
# aliases it, and every screen gated on the bundle's contents keys on it. Era
# boundaries are merge-to-main COMMITTER timestamps, never authoring dates.
B4E9DF0_MERGED_AT = datetime(2026, 7, 21, 17, 7, 37, tzinfo=UTC)

# 0e85e1b: numeric k_tail 1.25 -> 1.0 AND the binary clamp [0.01, 0.99] -> [0.02, 0.98].
WIDENING_FLIP_MERGED_AT = datetime(2026, 5, 18, 17, 21, 19, tzinfo=UTC)
# 325b1b0 (ft 0.2.54 -> 0.2.92): the MC option clamp [0.005, 0.995] -> [0.01, 0.99].
FT_0292_MERGED_AT = datetime(2026, 7, 24, 19, 16, 26, tzinfo=UTC)

# ``9f1175c`` (grid-scaled max-step for discrete CDF resampling) rode that merge.
# Before this instant a flat 0.2 per-bin cap applied at EVERY grid size; after it,
# coarse discrete grids get the relaxed ``grid_step_constraints`` cap.
GRID_SCALED_MAX_STEP_MERGED_AT = B4E9DF0_MERGED_AT

# A published bin counts as sitting at the cap within this tolerance, and the members
# must want at least this much MORE mass there for the record to be clamp-suspected.
_CLAMP_CAP_ATOL = 1e-6
_CLAMP_MEMBER_MARGIN = 0.10
# The min-step / ramp / discrete-snap machinery shaves the top step ~1% below the cap
# (q45065: 0.1977991526 against 0.2, 98.9%), so exact equality structurally misses every
# post-snap instance; a bin at >= this fraction of the cap is treated as cap-bound too.
_CLAMP_CAP_NEAR_FRAC = 0.90


def _resolution_bin(grid: list[float], cdf: list[float], resolution: float) -> tuple[float, float, float]:
    """``(bin_low, bin_high, published_mass)`` for the CDF bin the resolution lands in.

    side="left": a resolution sitting exactly ON a grid point belongs to the bin BELOW
    it, matching the platform scorer (``resolution_to_bucket_index`` assigns an exact
    grid point to the lower bucket); side="right" screens the bin above and misses the
    q43913 signature whenever the resolution lands on a grid edge.
    """
    grid_arr = np.asarray(grid, dtype=float)
    cdf_arr = np.maximum.accumulate(np.clip(np.asarray(cdf, dtype=float), 0.0, 1.0))
    steps = np.diff(cdf_arr)
    index = int(np.clip(np.searchsorted(grid_arr, resolution, side="left") - 1, 0, len(steps) - 1))
    return float(grid_arr[index]), float(grid_arr[index + 1]), float(steps[index])


def _member_bin_masses(record: dict, bin_low: float, bin_high: float) -> dict[str, float]:
    """Each attributed member curve's own mass on the ``[bin_low, bin_high]`` bin.

    Two members are skipped rather than measured: an anonymous positional key, which on
    a stacked record can hold the stacker's aggregate instead of a member curve (see
    ``per_model_cohort``); and a curve under ``MIN_SCOREABLE_ANCHORS`` distinct anchors,
    because the screen's verdict turns on the MINIMUM member bin mass, so one sparse
    recovery can decide it — and a 3-anchor curve interpolated across a bin is not the
    distribution the model declared. That is the same reason ranking_cohort gates its
    log-scored curves; the shared floor lives in parsing so the two cannot drift.
    """
    masses: dict[str, float] = {}
    for model, pairs in (record.get("per_model_numeric_percentiles") or {}).items():
        if is_anonymous_model_key(model) or len(declared_anchors(pairs)[0]) < MIN_SCOREABLE_ANCHORS:
            continue
        low = _single_curve_pit(pairs, bin_low)
        high = _single_curve_pit(pairs, bin_high)
        if low is not None and high is not None:
            masses[model] = high - low
    return masses


def max_step_clamp_screen(record: dict, *, member_margin: float = _CLAMP_MEMBER_MARGIN) -> dict:
    """Did a per-bin max-step cap, not the forecasters, decide the published mass at the truth?

    On a coarse discrete grid the pre-``9f1175c`` flat 0.2 cap can hold the realized
    bin far below what every member asked for (q43913: published 0.200 where members'
    own curves wanted 0.575-0.823 — spot peer -41.20, coverage-scaled peer
    -38.67). That is a pipeline defect
    masquerading as a forecast error, and it manufactures apparent dissent: each
    member keeps its concentrated mass while the published curve does not.

    The cap is ERA-CORRECT, gated on the submit timestamp: the flat 0.2 before the
    grid-scaled cap reached main (``GRID_SCALED_MAX_STEP_MERGED_AT``), the record's
    own ``grid_step_constraints(len(cdf))`` max after. Without the gate every
    post-fix coarse-grid discrete that legitimately holds a 0.2 bin false-positives.
    A missing/unparseable timestamp is treated as pre-fix — the undated records in
    the archive all predate the fix.

    Suspected requires ALL of: the realized bin CAP-BOUND — within ``_CLAMP_CAP_ATOL``
    of the era-correct cap, or at least ``_CLAMP_CAP_NEAR_FRAC`` of it, since the
    min-step / ramp / snap machinery shaves a saturated bin ~1% under the cap — at
    least two attributed member curves, and the LEAST concentrated member wanting at
    least ``member_margin`` more mass on that bin — "every member" is the point; a
    clamp overrides the whole ensemble, unlike a median.

    The cap is the PLATFORM's per-bin rule (``0.2 * 200 / N``), so a cap-bound
    realized bin is not automatically our defect. Pre-``d4ee57f`` records additionally
    carry the slack-proportional smear; post-``d4ee57f`` the excess is packed into the
    adjacent bins, and a cap-bound bin means only that the platform constraint set the
    published mass at the truth.
    """
    out: dict = {"applicable": record.get("type") in ("numeric", "discrete"), "suspected": False}
    if not out["applicable"]:
        return out
    grid = (record.get("scaling") or {}).get("continuous_range")
    cdf = record.get("our_forecast_values")
    resolution = record.get("resolution_parsed")
    if not isinstance(resolution, (int, float)) or isinstance(resolution, bool):
        out["reason"] = "non-numeric resolution"
        return out
    if not grid or not cdf or len(grid) != len(cdf):
        out["reason"] = "no usable grid"
        return out

    bin_low, bin_high, published_bin_mass = _resolution_bin(grid, cdf, float(resolution))

    submitted = parse_iso_utc(record.get("bot_comment_created_at"))
    before_fix = submitted is None or submitted < GRID_SCALED_MAX_STEP_MERGED_AT
    cap = MAX_CDF_PROB_STEP if before_fix else grid_step_constraints(len(grid))[1]

    member_bin_masses = _member_bin_masses(record, bin_low, bin_high)
    min_member = min(member_bin_masses.values()) if member_bin_masses else None

    at_cap = abs(published_bin_mass - cap) <= _CLAMP_CAP_ATOL
    cap_fraction = published_bin_mass / cap
    cap_bound = at_cap or cap_fraction >= _CLAMP_CAP_NEAR_FRAC
    out |= {
        "n_grid_points": len(grid),
        "max_step_cap": cap,
        "submitted_before_grid_scaled_cap": before_fix,
        "resolution_bin": [bin_low, bin_high],
        "published_bin_mass": published_bin_mass,
        "resolution_bin_at_cap": at_cap,
        "resolution_bin_cap_fraction": cap_fraction,
        "resolution_bin_cap_bound": cap_bound,
        "member_bin_masses": member_bin_masses,
        "min_member_bin_mass": min_member,
        "suspected": bool(
            cap_bound
            and min_member is not None
            and min_member > published_bin_mass + member_margin
            and len(member_bin_masses) >= 2
        ),
    }
    return out


def _interpolate_pit(
    resolution: float,
    lower_bound: float,
    upper_bound: float,
    cdf_values: list[float],
    *,
    value_grid: list[float] | None = None,
    zero_point: float | None = None,
    per_model_percentiles: Mapping[str, Sequence[Sequence[float]]] | None = None,
) -> float:
    """Interpolate the PIT value ``F(resolution)`` for a numeric resolution given its CDF.

    The CDF is defined on a value grid that is linear for linear-scaled questions but
    GEOMETRIC for log-scaled questions (``zero_point`` set). Evaluate ``F`` by interpolating
    the resolution against the actual value grid the CDF lives on, not against a linear index
    map -- the latter mis-buckets log-scaled questions (PIT off by up to ~0.24).

    ``value_grid`` is the authoritative grid (``scaling.continuous_range``) when present; it
    is used directly if its length matches ``cdf_values``. Otherwise the grid is reconstructed
    via :func:`build_cdf_value_grid` using ``zero_point``.
    """
    if upper_bound - lower_bound <= 0:
        # A zero-width question has no PIT, so this raises rather than answering. The old
        # ``return 0.5`` was the single most favorable value available — it falls inside BOTH
        # coverage bands, so a degenerate record silently improved every calibration
        # statistic it entered. Callers screen the range before calling (see
        # ``numeric_pit_analysis``); reaching here means one didn't.
        raise ValueError(f"degenerate question range [{lower_bound}, {upper_bound}] has no PIT")

    if value_grid is not None and len(value_grid) == len(cdf_values):
        grid = np.asarray(value_grid, dtype=float)
    else:
        grid = build_cdf_value_grid(lower_bound, upper_bound, zero_point, num_points=len(cdf_values))

    return pit_on_grid(resolution, grid, np.asarray(cdf_values, dtype=float), per_model_percentiles)[0]


def pit_on_grid(
    resolution: float,
    grid: np.ndarray,
    cdf: np.ndarray,
    per_model_percentiles: Mapping[str, Sequence[Sequence[float]]] | None,
) -> tuple[float, str | None]:
    """``(pit, oob_side)`` for a numeric resolution against a value grid.

    THE single home of the out-of-grid rule shared by both PIT paths
    (``_interpolate_pit`` here and ``width_monitor.compute_pit_details``).
    ``np.interp`` clamps beyond the grid to ``cdf[0]`` / ``cdf[-1]`` — the correct
    PIT for a resolution exactly AT a bound (F(bound) = cdf[0]) but NOT for one
    BEYOND the grid: with below/above-bound mass expressible on open bounds,
    ``cdf[0]`` can be ~0.9, so the clamp reads a below-grid resolution — a
    low-tail event — as a HIGH PIT, flipping the sign of the miss. Beyond the
    grid the PIT comes off the members' declared-percentile curves; the clamp is
    kept only when no curve is usable, and ``oob_side`` (``"low"``/``"high"``/None)
    reports the beyond-grid case either way.
    """
    oob_side = "low" if resolution < grid[0] else ("high" if resolution > grid[-1] else None)
    if oob_side is not None:
        fallback = declared_percentile_pit(per_model_percentiles, resolution)
        if fallback is not None:
            return fallback, oob_side
    return float(np.interp(resolution, grid, cdf)), oob_side


def mc_summary(data: list[dict]) -> dict:
    """Summary statistics for multiple-choice questions.

    Returns dict with accuracy, mean_prob_correct, mean_mc_log_score, and count.
    """
    mc = [r for r in data if r["type"] == "multiple_choice" and r["mc_log_score"] is not None]
    if not mc:
        return {"count": 0}

    correct_count = 0
    prob_on_correct: list[float] = []
    log_scores: list[float] = []

    for r in mc:
        resolution = r["resolution_parsed"]
        options = r.get("options") or []
        forecast_values = r["our_forecast_values"]

        if resolution in options and forecast_values:
            correct_idx = options.index(resolution)
            if correct_idx < len(forecast_values):
                prob_on_correct.append(forecast_values[correct_idx])
            else:
                # A forecast vector shorter than the option list cannot say what probability
                # we put on the winner. The old ``else 0.0`` recorded that as "we gave the
                # correct option zero" — the worst possible value — dragging
                # mean_prob_correct down on a PARSE gap rather than on a forecast. The
                # mc_log_score gate upstream makes this unreachable today; if it ever isn't,
                # the record leaves this one statistic instead of poisoning it.
                logger.warning(
                    f"MC forecast vector shorter than its option list: post_id={r.get('post_id')} "
                    f"options={len(options)} values={len(forecast_values)}; dropped from mean_prob_correct"
                )

            # "Correct" = highest predicted probability was on the correct option
            max_prob_idx = forecast_values.index(max(forecast_values))
            if max_prob_idx == correct_idx:
                correct_count += 1

        log_scores.append(r["mc_log_score"])

    return {
        "count": len(mc),
        "accuracy": correct_count / len(mc) if mc else None,
        "mean_prob_correct": _mean(prob_on_correct),
        "mean_mc_log_score": _mean(log_scores),
    }


def no_bias_check(data: list[dict]) -> dict:
    """Detect systematic NO-bias on binary predictions.

    Returns dict with overall bias_pp (mean_predicted - actual_yes_rate, in
    percentage points) and a low_range subset (P(yes) in [0.10, 0.30]).
    """
    binary = [r for r in data if r["type"] == "binary" and isinstance(r["resolution_parsed"], bool)]
    if not binary:
        return {"count": 0}

    probs = [r["our_prob_yes"] for r in binary]
    outcomes = [1.0 if r["resolution_parsed"] else 0.0 for r in binary]
    mean_predicted = _mean(probs)
    actual_yes_rate = _mean(outcomes)
    assert mean_predicted is not None
    assert actual_yes_rate is not None
    bias_pp = (mean_predicted - actual_yes_rate) * 100.0

    low_range = [r for r in binary if 0.10 <= r["our_prob_yes"] <= 0.30]
    low_range_summary: dict = {"count": 0}
    if low_range:
        lr_probs = [r["our_prob_yes"] for r in low_range]
        lr_outcomes = [1.0 if r["resolution_parsed"] else 0.0 for r in low_range]
        lr_mean_pred = _mean(lr_probs)
        lr_actual = _mean(lr_outcomes)
        assert lr_mean_pred is not None
        assert lr_actual is not None
        low_range_summary = {
            "count": len(low_range),
            "mean_predicted": lr_mean_pred,
            "actual_yes_rate": lr_actual,
            "bias_pp": (lr_mean_pred - lr_actual) * 100.0,
        }

    return {
        "count": len(binary),
        "mean_predicted": mean_predicted,
        "actual_yes_rate": actual_yes_rate,
        "bias_pp": bias_pp,
        "low_range": low_range_summary,
    }


def _is_financial(category: str | None) -> bool:
    if not category:
        return False
    cat_lower = category.lower()
    return any(sub in cat_lower for sub in _FINANCIAL_CATEGORY_SUBSTRINGS)


def financial_vs_nonfinancial_pit(data: list[dict]) -> dict:
    """Split numeric PIT analysis by financial vs non-financial category."""
    numeric = [r for r in data if r["type"] in ("numeric", "discrete")]
    financial = [r for r in numeric if _is_financial((r.get("metadata") or {}).get("category"))]
    nonfinancial = [r for r in numeric if not _is_financial((r.get("metadata") or {}).get("category"))]
    return {
        "financial": numeric_pit_analysis(financial),
        "nonfinancial": numeric_pit_analysis(nonfinancial),
    }


def stacking_effectiveness(
    data: list[dict],
    threshold: float,
    spread_fn: BinarySpreadFn = binary_prob_range_spread,
) -> dict:
    """Bucket binary questions by whether their spread would have triggered stacking.

    On each question, compute the binary spread across per-model forecasts using
    ``spread_fn`` (default: probability range, matching the production trigger).
    If that spread is strictly greater than ``threshold``, count it as triggered
    (comparison uses ``>``, matching the production trigger in main.py).
    Returns triggered/skipped counts and mean Brier per bucket.

    Note: this does NOT tell us whether the stored ensemble prediction was
    actually produced via stacking or base aggregation. We can't distinguish
    those from stored data alone; this is a counterfactual cohort cut showing
    how the trigger metric correlates with outcome difficulty.

    Restricted to the per-model cohort (see ``per_model_cohort``), so the spread
    is always measured across named base models. Records whose spread can't be
    computed (fewer than two parseable per-model values) land in ``skipped``, as
    they always have.
    """
    binary = [r for r in data if r["type"] == "binary" and isinstance(r["resolution_parsed"], bool)]

    triggered: list[dict] = []
    skipped: list[dict] = []
    for r, per_model in per_model_cohort(binary, cut="stacking_effectiveness"):
        parsed = [_parse_probability(raw) for raw in per_model.values()]
        probs = [p for p in parsed if p is not None]
        if len(probs) < 2:
            skipped.append(r)
            continue
        spread = spread_fn(probs)
        # strict ">" matches the production trigger in main.py
        if spread > threshold:
            triggered.append(r)
        else:
            skipped.append(r)

    return {
        "triggered_count": len(triggered),
        "skipped_count": len(skipped),
        "triggered_mean_brier": _mean([r["brier_score"] for r in triggered]),
        "skipped_mean_brier": _mean([r["brier_score"] for r in skipped]),
    }


def _spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation. Returns None for n<3 or degenerate rankings.

    Delegates to scipy's implementation (which handles ties via average-rank).
    scipy returns NaN for degenerate inputs (constant input, etc.); we surface
    that as None to match the original semantics.
    """
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    result = spearmanr(xs, ys)
    rho = float(result.statistic)
    if math.isnan(rho):
        return None
    return rho


def disagreement_predicts_error(
    data: list[dict],
    spread_fn: BinarySpreadFn = binary_prob_range_spread,
) -> dict:
    """Spearman correlation between per-model disagreement and prediction error.

    Pass ``spread_fn=binary_log_odds_spread`` for an alternative spread metric
    (default is probability range, which correlates more strongly with Brier
    error in practice).

    Returns dict with computed rho, n, and mean Brier per spread quartile.

    Restricted to the per-model cohort (see ``per_model_cohort``), so the
    disagreement being correlated is disagreement among named base models.
    """
    binary = [r for r in data if r["type"] == "binary" and r["brier_score"] is not None]
    paired: list[tuple[float, float]] = []
    for r, per_model in per_model_cohort(binary, cut="disagreement_predicts_error"):
        parsed = [_parse_probability(raw) for raw in per_model.values()]
        probs = [p for p in parsed if p is not None]
        if len(probs) < 2:
            continue
        spread = spread_fn(probs)
        paired.append((spread, r["brier_score"]))

    if not paired:
        return {"count": 0, "spearman_rho": None}

    spreads = [p[0] for p in paired]
    briers = [p[1] for p in paired]
    rho = _spearman_rho(spreads, briers)

    # Quartile buckets on spread
    quartile_briers: dict[str, float | None] = {}
    if len(paired) >= 4:
        sorted_pairs = sorted(paired, key=lambda t: t[0])
        n = len(sorted_pairs)
        q1 = sorted_pairs[: n // 4]
        q2 = sorted_pairs[n // 4 : n // 2]
        q3 = sorted_pairs[n // 2 : 3 * n // 4]
        q4 = sorted_pairs[3 * n // 4 :]
        for label, bucket in [("q1_low", q1), ("q2", q2), ("q3", q3), ("q4_high", q4)]:
            quartile_briers[label] = _mean([p[1] for p in bucket])

    return {
        "count": len(paired),
        "spearman_rho": rho,
        "mean_spread": _mean(spreads),
        "mean_brier": _mean(briers),
        "quartile_briers": quartile_briers,
    }


def _question_count_lines(data: list[dict]) -> list[str]:
    """Total question count plus the per-type breakdown, types alphabetical."""
    type_counts: dict[str, int] = {}
    for r in data:
        type_counts[r["type"]] = type_counts.get(r["type"], 0) + 1

    lines = [f"**Total questions:** {len(data)}"]
    lines.extend(f"- {t}: {count}" for t, count in sorted(type_counts.items()))
    lines.append("")
    return lines


def _platform_score_section_lines(data: list[dict]) -> list[str]:
    """Metaculus's own scores, spot peer first. Empty when no record carries any.

    Every other section of this report is a BOT-side score (Brier, log score) computed
    here from the resolution. This one is the only section computed by the platform
    rather than by us; it is still a pooled mean over whatever records the caller
    passed, so it is not the leaderboard standing.
    """
    ps = platform_score_summary(data)
    if not any(ps[key]["count"] for key, _reader, _label in _PLATFORM_SCORE_METRICS):
        return []

    lines = [
        "## Metaculus Platform Scores",
        "",
        "The tournament leaderboard ranks on SPOT peer. `peer` is the same quantity scaled by "
        "coverage (`peer ~= spot_peer * coverage`); this bot submits once and never revises, so "
        "its coverage is mostly a function of how early it submitted. Rank on spot, read peer as "
        "a diagnostic only.",
        "",
        "These figures are pooled over every record handed in — no config-era split, and no "
        "exclusion of the known-bug (`KNOWN_BUG_QIDS`) or degraded-run (`DEGRADED_RUN_QIDS` / "
        "`PARTIAL_DEGRADED_QIDS`) cohorts. Read them as this pull's mean, not as tournament "
        "standing; pass a pre-filtered `data` or use `width_monitor --exclude-qids` for a "
        "cohort-controlled read.",
        "",
        "| metric | n | mean | median |",
        "|--------|---|------|--------|",
    ]
    for key, _reader, label in _PLATFORM_SCORE_METRICS:
        stats = ps[key]
        if not stats["count"]:
            continue
        lines.append(f"| {label} | {stats['count']} | {stats['mean']:+.2f} | {stats['median']:+.2f} |")
    if ps["mean_coverage"] is not None:
        lines.append("")
        lines.append(f"- Mean coverage: {ps['mean_coverage']:.3f}")
    lines.append("")
    return lines


def _binary_section_lines(data: list[dict]) -> list[str]:
    """Binary headline scores plus the calibration-bucket table. Empty when no binaries."""
    bs = binary_summary(data)
    if bs["count"] <= 0:
        return []

    lines = [
        "## Binary Questions",
        f"- Count: {bs['count']}",
        f"- Mean Brier: {bs['mean_brier']:.4f}",
        f"- Mean Log Score: {bs['mean_log_score']:.2f}",
        f"- Direction Accuracy: {bs['direction_accuracy']:.1%}",
        f"- Base Rate: {bs['base_rate']:.1%}",
        f"- Base Rate Brier: {bs['base_rate_brier']:.4f}",
        "",
        "### Calibration",
        "| Bucket | Predicted | Actual | Count |",
        "|--------|-----------|--------|-------|",
    ]
    lines.extend(
        f"| {label} | {bucket['predicted_mean']:.2f} | {bucket['actual_rate']:.2f} | {bucket['count']} |"
        for label, bucket in bs["calibration_buckets"].items()
    )
    lines.append("")
    return lines


def _per_model_section_lines(data: list[dict]) -> list[str]:
    """Per-member binary scores. Empty when no record yields an attributed member."""
    pm = per_model_binary_scores(data)
    if not pm:
        return []

    lines = [
        "## Per-Model Binary Scores",
        "| Model | Mean Brier | Mean Log Score | Count |",
        "|-------|-----------|----------------|-------|",
    ]
    lines.extend(
        f"| {model_name} | {scores['mean_brier']:.4f} | {scores['mean_log_score']:.2f} | {scores['count']} |"
        for model_name, scores in pm.items()
    )
    lines.append("")
    return lines


def _numeric_section_lines(data: list[dict]) -> list[str]:
    """Numeric coverage headline plus the ten-bin PIT histogram. Empty when no PITs.

    The set-valued count is rendered unconditionally: those records DO count toward the two
    coverage lines (on band intersection) and do NOT appear in the histogram, so a reader
    comparing the two needs the split stated rather than inferred.
    """
    na = numeric_pit_analysis(data)
    if na["count"] <= 0:
        return []

    lines = [
        "## Numeric Questions",
        f"- Count: {na['count']}",
        f"- Mean Numeric Log Score: {na['mean_numeric_log_score']:.2f}",
        f"- 90% Coverage (PIT in [0.05, 0.95]): {na['coverage_90']:.1%}",
        f"- 50% Coverage (PIT in [0.25, 0.75]): {na['coverage_50']:.1%}",
        f"- Out-of-range resolutions (set-valued PIT; counted in coverage, "
        f"excluded from the histogram): {na['n_oob_interval']}",
        "",
        "### PIT Histogram",
        "| Bin | Count |",
        "|-----|-------|",
    ]
    lines.extend(f"| {i / 10.0:.1f}-{(i + 1) / 10.0:.1f} | {count} |" for i, count in enumerate(na["histogram"]))
    lines.append("")
    return lines


def _mc_section_lines(data: list[dict]) -> list[str]:
    """Multiple-choice headline scores. Empty when no MC record carries a log score."""
    ms = mc_summary(data)
    if ms["count"] <= 0:
        return []

    return [
        "## Multiple Choice Questions",
        f"- Count: {ms['count']}",
        f"- Accuracy (top pick correct): {ms['accuracy']:.1%}",
        f"- Mean Prob on Correct: {ms['mean_prob_correct']:.2f}",
        f"- Mean MC Log Score: {ms['mean_mc_log_score']:.2f}",
        "",
    ]


def generate_report(data: list[dict]) -> str:
    """Baseline markdown report (platform scores, binary, numeric, MC + per-model binary).

    Platform scores lead: they're the only section that maps to tournament standing, and
    the spot/coverage-scaled distinction is easy to get backwards when it appears only in
    a per-question dossier.

    For extended analyses -- NO-bias check, financial split, stacking effectiveness,
    disagreement-error correlation -- call those functions directly; see
    scratch/analysis_2026-04/compute_delta.py for an example.
    """
    lines: list[str] = ["# Performance Analysis Report", ""]
    lines.extend(_question_count_lines(data))
    lines.extend(_platform_score_section_lines(data))
    lines.extend(_binary_section_lines(data))
    lines.extend(_per_model_section_lines(data))
    lines.extend(_numeric_section_lines(data))
    lines.extend(_mc_section_lines(data))
    return "\n".join(lines)
