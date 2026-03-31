"""Reusable analysis functions for performance data."""

import logging
import math

from metaculus_bot.performance_analysis.scoring import binary_log_score, brier_score

logger: logging.Logger = logging.getLogger(__name__)

# Calibration bucket edges for binary questions
CALIBRATION_BUCKET_EDGES: list[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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
            prob = _parse_percentage(raw_value)
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


def _parse_percentage(raw: str) -> float | None:
    """Parse a percentage string like '72.0%' into a probability."""
    raw = raw.strip()
    if raw.endswith("%"):
        try:
            return float(raw[:-1]) / 100.0
        except ValueError:
            return None
    try:
        val = float(raw)
        # Heuristic: if > 1 it's probably a percentage
        if val > 1.0:
            return val / 100.0
        return val
    except ValueError:
        return None


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


def generate_report(data: list[dict]) -> str:
    """Generate a markdown summary report combining all analysis functions."""
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
