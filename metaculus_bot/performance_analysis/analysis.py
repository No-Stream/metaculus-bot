"""Reusable analysis functions for performance data."""

import logging
import math
from typing import Callable

from scipy.stats import spearmanr

from metaculus_bot.performance_analysis.parsing import _parse_probability
from metaculus_bot.performance_analysis.scoring import binary_log_score, brier_score
from metaculus_bot.spread_metrics import binary_prob_range_spread

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
    """
    binary = [r for r in data if r["type"] == "binary" and isinstance(r["resolution_parsed"], bool)]

    model_scores: dict[str, list[tuple[float, float]]] = {}
    for r in binary:
        per_model = r.get("per_model_forecasts") or {}
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
            pit = _interpolate_pit(float(resolution), lower_bound, upper_bound, cdf_values)
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


def _interpolate_pit(resolution: float, lower_bound: float, upper_bound: float, cdf_values: list[float]) -> float:
    """Interpolate the PIT value for a numeric resolution given its CDF."""
    total_range = upper_bound - lower_bound
    if total_range <= 0:
        return 0.5

    fraction = (resolution - lower_bound) / total_range
    n = len(cdf_values)
    idx_float = fraction * (n - 1)
    idx_low = max(0, min(int(math.floor(idx_float)), n - 2))
    idx_high = idx_low + 1
    weight = idx_float - idx_low
    return cdf_values[idx_low] * (1 - weight) + cdf_values[idx_high] * weight


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
            prob = forecast_values[correct_idx] if correct_idx < len(forecast_values) else 0.0
            prob_on_correct.append(prob)

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
    """
    binary = [r for r in data if r["type"] == "binary" and isinstance(r["resolution_parsed"], bool)]

    triggered: list[dict] = []
    skipped: list[dict] = []
    for r in binary:
        per_model = r.get("per_model_forecasts") or {}
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
    """
    binary = [r for r in data if r["type"] == "binary" and r["brier_score"] is not None]
    paired: list[tuple[float, float]] = []
    for r in binary:
        per_model = r.get("per_model_forecasts") or {}
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
