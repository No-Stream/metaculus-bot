"""Reusable analysis functions for performance data."""

import logging
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Callable

import numpy as np
from scipy.stats import spearmanr

from metaculus_bot.numeric.config import MAX_CDF_PROB_STEP, grid_step_constraints
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.performance_analysis.parsing import (
    MIN_SCOREABLE_ANCHORS,
    _parse_probability,
    declared_anchors,
    is_anonymous_model_key,
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


def numeric_pit_analysis(data: list[dict]) -> dict:
    """PIT (Probability Integral Transform) analysis for numeric questions.

    Returns dict with pit_values, coverage stats, and histogram bin counts.
    """
    numeric = [r for r in data if r["type"] in ("numeric", "discrete") and r["numeric_log_score"] is not None]
    if not numeric:
        return {"count": 0}

    pit_values: list[float] = []
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

        if resolution == "above_upper_bound":
            pit = 1.0
        elif resolution == "below_lower_bound":
            pit = 0.0
        elif isinstance(resolution, (int, float)):
            if upper_bound - lower_bound <= 0:
                # No range, no interpolated PIT — the record is dropped rather than
                # contributing the 0.5 that used to come back from _interpolate_pit, which
                # is the single most favorable value available (inside BOTH coverage bands).
                # Scoped to the interpolated branch: an out-of-bound resolution's 0.0/1.0 is
                # a real reading that never touched the range.
                continue
            pit = _interpolate_pit(
                float(resolution),
                lower_bound,
                upper_bound,
                cdf_values,
                value_grid=scaling.get("continuous_range"),
                zero_point=grid_zero_point(scaling.get("zero_point"), lower_bound),
                per_model_percentiles=r.get("per_model_numeric_percentiles"),
            )
        else:
            continue

        pit_values.append(pit)

    if not pit_values:
        return {"count": 0}

    num_bins = 10
    histogram = [0] * num_bins
    for pit in pit_values:
        bin_idx = min(int(pit * num_bins), num_bins - 1)
        histogram[bin_idx] += 1

    # Coverage: fraction of PIT values in [0.05, 0.95] (should be ~90% for well-calibrated)
    coverage_90 = sum(1 for p in pit_values if 0.05 <= p <= 0.95) / len(pit_values)
    coverage_50 = sum(1 for p in pit_values if 0.25 <= p <= 0.75) / len(pit_values)

    log_scores = [r["numeric_log_score"] for r in numeric]

    return {
        "count": len(pit_values),
        "pit_values": pit_values,
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


# ``b4e9df0`` — the merge that landed the july15 bundle on main. THE single source of
# truth for this era boundary across the package: width_monitor's ``TS_ANCHOR_ENABLE``
# aliases it, and every screen gated on the bundle's contents keys on it. Era
# boundaries are merge-to-main COMMITTER timestamps, never authoring dates.
B4E9DF0_MERGED_AT = datetime(2026, 7, 21, 17, 7, 37, tzinfo=timezone.utc)

# ``9f1175c`` (grid-scaled max-step for discrete CDF resampling) rode that merge.
# Before this instant a flat 0.2 per-bin cap applied at EVERY grid size; after it,
# coarse discrete grids get the relaxed ``grid_step_constraints`` cap.
GRID_SCALED_MAX_STEP_MERGED_AT = B4E9DF0_MERGED_AT

# A published bin counts as sitting at the cap within this tolerance, and the members
# must want at least this much MORE mass there for the record to be clamp-suspected.
_CLAMP_CAP_ATOL = 1e-6
_CLAMP_MEMBER_MARGIN = 0.10


def max_step_clamp_screen(record: dict, *, member_margin: float = _CLAMP_MEMBER_MARGIN) -> dict:
    """Did a per-bin max-step cap, not the forecasters, decide the published mass at the truth?

    On a coarse discrete grid the pre-``9f1175c`` flat 0.2 cap can hold the realized
    bin far below what every member asked for (q43913: published 0.200 where members'
    own curves wanted 0.575-0.823 — peer −38.67). That is a pipeline defect
    masquerading as a forecast error, and it manufactures apparent dissent: each
    member keeps its concentrated mass while the published curve does not.

    The cap is ERA-CORRECT, gated on the submit timestamp: the flat 0.2 before the
    grid-scaled cap reached main (``GRID_SCALED_MAX_STEP_MERGED_AT``), the record's
    own ``grid_step_constraints(len(cdf))`` max after. Without the gate every
    post-fix coarse-grid discrete that legitimately holds a 0.2 bin false-positives.
    A missing/unparseable timestamp is treated as pre-fix — the undated records in
    the archive all predate the fix.

    Suspected requires ALL of: the realized bin within ``_CLAMP_CAP_ATOL`` of the
    era-correct cap, at least two attributed member curves, and the LEAST
    concentrated member wanting at least ``member_margin`` more mass on that bin —
    "every member" is the point; a clamp overrides the whole ensemble, unlike a
    median.
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

    grid_arr = np.asarray(grid, dtype=float)
    cdf_arr = np.maximum.accumulate(np.clip(np.asarray(cdf, dtype=float), 0.0, 1.0))
    steps = np.diff(cdf_arr)
    # side="left": a resolution sitting exactly ON a grid point belongs to the bin
    # BELOW it, matching the platform scorer (resolution_to_bucket_index assigns an
    # exact grid point to the lower bucket); side="right" screens the bin above and
    # misses the q43913 signature whenever the resolution lands on a grid edge.
    index = int(np.clip(np.searchsorted(grid_arr, float(resolution), side="left") - 1, 0, len(steps) - 1))
    published_bin_mass = float(steps[index])
    bin_low, bin_high = float(grid_arr[index]), float(grid_arr[index + 1])

    submitted = parse_iso_utc(record.get("bot_comment_created_at"))
    before_fix = submitted is None or submitted < GRID_SCALED_MAX_STEP_MERGED_AT
    cap = MAX_CDF_PROB_STEP if before_fix else grid_step_constraints(len(grid_arr))[1]

    member_bin_masses: dict[str, float] = {}
    for model, pairs in (record.get("per_model_numeric_percentiles") or {}).items():
        if is_anonymous_model_key(model):
            # A positional key on a stacked record can hold the stacker's aggregate,
            # which is not a member curve (see per_model_cohort).
            continue
        if len(declared_anchors(pairs)[0]) < MIN_SCOREABLE_ANCHORS:
            # The screen's verdict turns on the MINIMUM member bin mass, so one sparse
            # recovery can decide it — and a 3-anchor curve interpolated across a bin is
            # not the distribution the model declared. That is the same reason
            # ranking_cohort gates its log-scored curves; the shared floor lives in
            # parsing so the two cannot drift.
            continue
        low = _single_curve_pit(pairs, bin_low)
        high = _single_curve_pit(pairs, bin_high)
        if low is not None and high is not None:
            member_bin_masses[model] = high - low
    min_member = min(member_bin_masses.values()) if member_bin_masses else None

    at_cap = abs(published_bin_mass - cap) <= _CLAMP_CAP_ATOL
    out |= {
        "n_grid_points": len(grid_arr),
        "max_step_cap": cap,
        "submitted_before_grid_scaled_cap": before_fix,
        "resolution_bin": [bin_low, bin_high],
        "published_bin_mass": published_bin_mass,
        "resolution_bin_at_cap": at_cap,
        "member_bin_masses": member_bin_masses,
        "min_member_bin_mass": min_member,
        "suspected": bool(
            at_cap
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
    assert mean_predicted is not None and actual_yes_rate is not None
    bias_pp = (mean_predicted - actual_yes_rate) * 100.0

    low_range = [r for r in binary if 0.10 <= r["our_prob_yes"] <= 0.30]
    low_range_summary: dict = {"count": 0}
    if low_range:
        lr_probs = [r["our_prob_yes"] for r in low_range]
        lr_outcomes = [1.0 if r["resolution_parsed"] else 0.0 for r in low_range]
        lr_mean_pred = _mean(lr_probs)
        lr_actual = _mean(lr_outcomes)
        assert lr_mean_pred is not None and lr_actual is not None
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


def generate_report(data: list[dict]) -> str:
    """Baseline markdown report (binary, numeric, MC summaries + per-model binary).

    For extended analyses -- NO-bias check, financial split, stacking effectiveness,
    disagreement-error correlation -- call those functions directly; see
    scratch/analysis_2026-04/compute_delta.py for an example.
    """
    lines: list[str] = []
    lines.append("# Performance Analysis Report")
    lines.append("")

    type_counts: dict[str, int] = {}
    for r in data:
        t = r["type"]
        type_counts[t] = type_counts.get(t, 0) + 1

    lines.append(f"**Total questions:** {len(data)}")
    for t, count in sorted(type_counts.items()):
        lines.append(f"- {t}: {count}")
    lines.append("")

    # Binary
    bs = binary_summary(data)
    if bs["count"] > 0:
        lines.append("## Binary Questions")
        lines.append(f"- Count: {bs['count']}")
        lines.append(f"- Mean Brier: {bs['mean_brier']:.4f}")
        lines.append(f"- Mean Log Score: {bs['mean_log_score']:.2f}")
        lines.append(f"- Direction Accuracy: {bs['direction_accuracy']:.1%}")
        lines.append(f"- Base Rate: {bs['base_rate']:.1%}")
        lines.append(f"- Base Rate Brier: {bs['base_rate_brier']:.4f}")
        lines.append("")

        lines.append("### Calibration")
        lines.append("| Bucket | Predicted | Actual | Count |")
        lines.append("|--------|-----------|--------|-------|")
        for label, bucket in bs["calibration_buckets"].items():
            lines.append(
                f"| {label} | {bucket['predicted_mean']:.2f} | {bucket['actual_rate']:.2f} | {bucket['count']} |"
            )
        lines.append("")

    # Per-model
    pm = per_model_binary_scores(data)
    if pm:
        lines.append("## Per-Model Binary Scores")
        lines.append("| Model | Mean Brier | Mean Log Score | Count |")
        lines.append("|-------|-----------|----------------|-------|")
        for model_name, scores in pm.items():
            lines.append(
                f"| {model_name} | {scores['mean_brier']:.4f} | {scores['mean_log_score']:.2f} | {scores['count']} |"
            )
        lines.append("")

    # Numeric
    na = numeric_pit_analysis(data)
    if na["count"] > 0:
        lines.append("## Numeric Questions")
        lines.append(f"- Count: {na['count']}")
        lines.append(f"- Mean Numeric Log Score: {na['mean_numeric_log_score']:.2f}")
        lines.append(f"- 90% Coverage (PIT in [0.05, 0.95]): {na['coverage_90']:.1%}")
        lines.append(f"- 50% Coverage (PIT in [0.25, 0.75]): {na['coverage_50']:.1%}")
        lines.append("")
        lines.append("### PIT Histogram")
        lines.append("| Bin | Count |")
        lines.append("|-----|-------|")
        for i, count in enumerate(na["histogram"]):
            low = i / 10.0
            high = (i + 1) / 10.0
            lines.append(f"| {low:.1f}-{high:.1f} | {count} |")
        lines.append("")

    # MC
    ms = mc_summary(data)
    if ms["count"] > 0:
        lines.append("## Multiple Choice Questions")
        lines.append(f"- Count: {ms['count']}")
        lines.append(f"- Accuracy (top pick correct): {ms['accuracy']:.1%}")
        lines.append(f"- Mean Prob on Correct: {ms['mean_prob_correct']:.2f}")
        lines.append(f"- Mean MC Log Score: {ms['mean_mc_log_score']:.2f}")
        lines.append("")

    return "\n".join(lines)
