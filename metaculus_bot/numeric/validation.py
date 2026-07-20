"""Validate numeric predictions against question constraints."""

from __future__ import annotations

import logging

from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion
from pydantic import ValidationError

from metaculus_bot.numeric.config import (
    EXPECTED_PERCENTILE_COUNT,
    MIN_BOUNDARY_DISTANCE,
    STANDARD_PERCENTILES_CSV,
    STRICT_ORDERING_EPSILON,
)
from metaculus_bot.numeric.percentile_set import EXPECTED_KEYS, percentile_key

logger = logging.getLogger(__name__)


def validate_percentile_count_and_values(percentile_list: list[Percentile]) -> None:
    """Validate that we have exactly the expected number of percentiles with the correct values."""
    # Check count
    if len(percentile_list) != EXPECTED_PERCENTILE_COUNT:
        raise ValidationError.from_exception_data(
            "NumericDistribution",
            [
                {
                    "type": "value_error",
                    "loc": ("declared_percentiles",),
                    "input": percentile_list,
                    "ctx": {
                        "error": f"Expected {EXPECTED_PERCENTILE_COUNT} declared percentiles ({STANDARD_PERCENTILES_CSV}), got {len(percentile_list)}.",
                    },
                }
            ],
        )

    # Check values with tolerance for rounding (canonical key convention from percentile_set)
    actual_percentiles = {percentile_key(p.percentile) for p in percentile_list}
    if actual_percentiles != EXPECTED_KEYS:
        raise ValidationError.from_exception_data(
            "NumericDistribution",
            [
                {
                    "type": "value_error",
                    "loc": ("declared_percentiles",),
                    "input": percentile_list,
                    "ctx": {
                        "error": f"Expected percentile set {{{STANDARD_PERCENTILES_CSV}}}, got {sorted(p.percentile * 100 for p in percentile_list)}.",
                    },
                }
            ],
        )


def sort_percentiles_by_value(percentile_list: list[Percentile]) -> list[Percentile]:
    """Sort percentiles by percentile value to ensure proper order."""
    return sorted(percentile_list, key=lambda p: p.percentile)


def filter_to_standard_percentiles(percentile_list: list[Percentile]) -> list[Percentile]:
    """Keep only the standard percentile set (see ``STANDARD_PERCENTILES``).

    If extras are present, drop them before validation. If duplicates occur (same percentile
    repeated), keep the first occurrence.
    """
    seen: set[float] = set()
    filtered: list[Percentile] = []
    for p in percentile_list:
        key = percentile_key(p.percentile)
        if key in EXPECTED_KEYS and key not in seen:
            filtered.append(p)
            seen.add(key)
    return filtered


def detect_unit_mismatch(
    percentile_list: list[Percentile],
    question: NumericQuestion,
    *,
    span_ratio_threshold: float = 1e-5,
    min_step_ratio_threshold: float = 1e-8,
    max_magnitude_ratio_threshold: float = 1e-5,
) -> tuple[bool, str]:
    """
    Heuristically detect likely unit/scale mismatch.

    Returns (is_mismatch, reason). No network or community stats required.
    - span_ratio_threshold: flag if (highest_declared - lowest_declared) / range < threshold
    - min_step_ratio_threshold: flag if min adjacent diff / range < threshold
    - max_magnitude_ratio_threshold: flag if max(|value|) / range < threshold
    """
    try:
        values = [float(p.value) for p in percentile_list]
        if not values:
            return True, "empty percentile values"
        values_sorted = sorted(values)
        lower = float(getattr(question, "lower_bound", 0.0))
        upper = float(getattr(question, "upper_bound", 0.0))
        rng = max(upper - lower, 1e-12)

        # Span between lowest and highest declared percentiles (use indices of sorted by percentile, but we
        # receive list sorted by percentile earlier in flow; still compute robustly)
        v05 = values_sorted[0]
        v95 = values_sorted[-1]
        span = v95 - v05

        # Min adjacent diff
        diffs = [b - a for a, b in zip(values_sorted, values_sorted[1:])]
        min_step = min(diffs) if diffs else 0.0

        # Max magnitude
        vmax = max(abs(v) for v in values_sorted)

        span_ratio = span / rng
        min_step_ratio = (min_step / rng) if rng > 0 else 0.0
        vmax_ratio = (vmax / rng) if rng > 0 else 0.0

        # Any of these triggers → mismatch
        if span_ratio < span_ratio_threshold:
            return True, f"tiny span vs range (span_ratio={span_ratio:.3e} < {span_ratio_threshold:.1e})"

        # Near-duplicate (min adjacent step) rule, jitter-aware.
        #
        # ``sanitize_percentiles`` deliberately separates equal / clustered declarations
        # (e.g. a low-count discrete question where a model declares P20=P40=P50=1) into a
        # strictly-increasing set, using a jitter epsilon on the order of
        # ``MIN_BOUNDARY_DISTANCE * range`` and, when adjacent clusters collide, compressing
        # them no tighter than that epsilon spread across the percentile set
        # (``epsilon / EXPECTED_PERCENTILE_COUNT``). Those sub-threshold gaps are an expected
        # artifact of building a valid CDF, not a scale error, so a fixed relative threshold
        # alone misfires on faithful concentrated forecasts and silently drops them. Require
        # the gap to also be tighter than anything the pipeline can produce — i.e. values so
        # collapsed it could not separate them (clamped onto a bound). Genuine
        # order-of-magnitude unit errors still surface via the span/magnitude ratios, which
        # this leaves untouched.
        pipeline_min_gap = max(MIN_BOUNDARY_DISTANCE * rng, STRICT_ORDERING_EPSILON) / EXPECTED_PERCENTILE_COUNT
        if min_step_ratio < min_step_ratio_threshold and min_step < 0.5 * pipeline_min_gap:
            return True, f"near-duplicate values (min_step_ratio={min_step_ratio:.3e} < {min_step_ratio_threshold:.1e})"

        if vmax_ratio < max_magnitude_ratio_threshold:
            return True, f"values tiny vs range (max_mag_ratio={vmax_ratio:.3e} < {max_magnitude_ratio_threshold:.1e})"

        return False, ""
    except (AttributeError, TypeError, ValueError) as e:
        logger.warning(f"Unit mismatch detection failed: {e}")
        return False, ""


def check_discrete_question_properties(question: NumericQuestion, cdf_points: int) -> tuple[bool, bool]:
    """Check if a question is discrete and determine zero_point handling."""
    cdf_size = getattr(question, "cdf_size", None)
    is_discrete = cdf_size is not None and cdf_size != cdf_points
    zero_point = getattr(question, "zero_point", None)

    force_zero_point_none = False

    if is_discrete and zero_point is not None:
        logger.debug(
            f"Question {getattr(question, 'id_of_question', 'N/A')}: Forcing zero_point=None for discrete question"
        )
        force_zero_point_none = True
    elif zero_point is not None and zero_point == question.lower_bound:
        logger.warning(
            f"Question {getattr(question, 'id_of_question', 'N/A')}: zero_point ({zero_point}) is equal to lower_bound "
            f"({question.lower_bound}). Forcing linear scale for CDF generation."
        )
        force_zero_point_none = True

    return is_discrete, force_zero_point_none
