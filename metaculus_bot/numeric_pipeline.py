"""Numeric percentile sanitisation and distribution construction helpers."""

from __future__ import annotations

import logging
from typing import List, Tuple

from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.bounds_clamping import (
    calculate_bounds_buffer,
    clamp_values_to_bounds,
    log_cluster_spreading_summary,
    log_corrections_summary,
    log_heavy_clamping_diagnostics,
)
from metaculus_bot.cluster_processing import (
    apply_cluster_spreading,
    apply_jitter_for_duplicates,
    compute_cluster_parameters,
    detect_count_like_pattern,
    ensure_strictly_increasing_bounded,
)
from metaculus_bot.numeric_config import (
    PCHIP_CDF_POINTS,
    TAIL_WIDEN_K_TAIL,
    TAIL_WIDEN_SPAN_FLOOR_GAMMA,
    TAIL_WIDEN_TAIL_START,
    TAIL_WIDENING_ENABLE,
)
from metaculus_bot.numeric_diagnostics import log_pchip_fallback, validate_cdf_construction
from metaculus_bot.numeric_validation import (
    check_discrete_question_properties,
    filter_to_standard_percentiles,
    sort_percentiles_by_value,
    validate_percentile_count_and_values,
)
from metaculus_bot.pchip_processing import (
    create_fallback_numeric_distribution,
    create_pchip_numeric_distribution,
    generate_pchip_cdf_with_smoothing,
)
from metaculus_bot.tail_widening import widen_declared_percentiles

logger = logging.getLogger(__name__)


def sanitize_percentiles(
    percentile_list: List[Percentile],
    question: NumericQuestion,
) -> Tuple[List[Percentile], float | None]:
    """Filter, validate, sort, jitter, and optionally widen percentile declarations."""

    filtered = filter_to_standard_percentiles(percentile_list)
    validate_percentile_count_and_values(filtered)
    ordered = sort_percentiles_by_value(filtered)
    adjusted = _apply_jitter_and_clamp(ordered, question)
    widened = _maybe_widen_tails(adjusted, question)

    _, should_force_zero_point_none = check_discrete_question_properties(question, PCHIP_CDF_POINTS)
    zero_point = getattr(question, "zero_point", None)
    if should_force_zero_point_none:
        zero_point = None

    return widened, zero_point


def build_numeric_distribution(
    percentile_list: List[Percentile],
    question: NumericQuestion,
    zero_point: float | None,
) -> NumericDistribution:
    """Create a numeric distribution, falling back to a heuristic on failure."""

    try:
        pchip_cdf, _smoothing_applied, _aggressive = generate_pchip_cdf_with_smoothing(
            percentile_list,
            question,
            zero_point,
        )
        prediction = create_pchip_numeric_distribution(pchip_cdf, percentile_list, question, zero_point)
    except Exception as exc:
        log_pchip_fallback(question, exc)
        prediction = create_fallback_numeric_distribution(percentile_list, question, zero_point)

    validate_cdf_construction(prediction, question)
    return prediction


def _apply_jitter_and_clamp(percentile_list: List[Percentile], question: NumericQuestion) -> List[Percentile]:
    range_size = question.upper_bound - question.lower_bound
    buffer = calculate_bounds_buffer(question)

    values = [p.value for p in percentile_list]
    modified_values = list(values)

    count_like = detect_count_like_pattern(values)
    span = (max(values) - min(values)) if values else 0.0
    value_eps, base_delta, spread_delta = compute_cluster_parameters(range_size, count_like, span)

    pre_deltas = [b - a for a, b in zip(values, values[1:])]
    min(pre_deltas) if pre_deltas else float("inf")  # keep behaviour identical for potential side effects

    modified_values, clusters_applied = apply_cluster_spreading(
        modified_values,
        question,
        value_eps,
        spread_delta,
        range_size,
    )

    modified_values = apply_jitter_for_duplicates(modified_values, question, range_size, percentile_list)
    modified_values, corrections_made = clamp_values_to_bounds(modified_values, percentile_list, question, buffer)

    log_cluster_spreading_summary(
        modified_values,
        values,
        question,
        clusters_applied,
        spread_delta,
        count_like,
    )
    log_corrections_summary(modified_values, values, question, corrections_made)
    log_heavy_clamping_diagnostics(modified_values, values, question, buffer)

    modified_values = ensure_strictly_increasing_bounded(modified_values, question, range_size)

    return [Percentile(value=v, percentile=p.percentile) for v, p in zip(modified_values, percentile_list)]


def _maybe_widen_tails(percentile_list: List[Percentile], question: NumericQuestion) -> List[Percentile]:
    if not TAIL_WIDENING_ENABLE:
        return percentile_list
    return widen_declared_percentiles(
        percentile_list,
        question,
        k_tail=TAIL_WIDEN_K_TAIL,
        tail_start=TAIL_WIDEN_TAIL_START,
        span_floor_gamma=TAIL_WIDEN_SPAN_FLOOR_GAMMA,
    )


__all__ = ["sanitize_percentiles", "build_numeric_distribution"]
