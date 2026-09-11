"""Clamp numeric percentile values to question bounds."""

from __future__ import annotations

import logging
from itertools import pairwise

from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric.config import BOUNDARY_SAFETY_MARGIN, grid_bin_width, minimum_separation

logger = logging.getLogger(__name__)


def calculate_bounds_buffer(question: NumericQuestion) -> float:
    """How far outside a CLOSED bound a declared value may sit and still be clamped in.

    The larger of the range-based tolerance (1% of the range, a flat 1.0 once the range exceeds
    100) and one grid bin (``grid_bin_width``), on every grid including the 201-point continuous
    one. A value within one bin of the edge is indistinguishable from the edge once the CDF is
    bucketed, so it is clamped instead of dropping the forecaster, while a value several bins out
    still reads as a scale error and raises. The bin floor is what a date question needs: its axis
    is epoch seconds, where the flat 1.0 is a ONE-SECOND tolerance, and a date named one day
    outside a closed bound (the natural granularity of a date answer) used to drop the member. On
    a 201-point question the floor binds once the range exceeds 200 (range / 200 instead of 1.0),
    which removes the old discontinuity where the flat 1.0 was 1% of a 100-wide range and 0.005%
    of a 20,000-wide one.

    This is the accept-or-raise tolerance ONLY. Where an accepted value lands is
    ``_boundary_inset``.
    """
    range_size = question.upper_bound - question.lower_bound
    range_buffer = 1.0 if range_size > 100 else range_size * BOUNDARY_SAFETY_MARGIN
    return max(range_buffer, grid_bin_width(question.lower_bound, question.upper_bound, question.cdf_size))


def _boundary_inset(question: NumericQuestion) -> float:
    """Where a value accepted by the tolerance lands: just inside the closed bound.

    Deliberately not the tolerance. Landing a value one tolerance inside moved it a whole bin on a
    coarse grid, and the left-to-right strict-ordering pass then dragged every percentile declared
    inside that first bin up behind it (post 651: 6.3 points of mass left the day the member
    declared; a [0, 20000] Metaculus question published four percentiles at 100.0 when three were
    in range).
    """
    return minimum_separation(question.upper_bound - question.lower_bound)


def clamp_values_to_bounds(
    modified_values: list[float],
    percentile_list: list[Percentile],
    question: NumericQuestion,
    buffer: float,
) -> tuple[list[float], bool]:
    """Clamp values outside a CLOSED bound by at most ``buffer`` (``calculate_bounds_buffer``) to just inside it.

    A violation beyond the tolerance raises: that is the scale-error signal the caller turns into
    a dropped member.
    """
    corrections_made = False
    inset = _boundary_inset(question)

    for i in range(len(modified_values)):
        original_value = modified_values[i]

        if not question.open_lower_bound and modified_values[i] < question.lower_bound:
            if question.lower_bound - modified_values[i] <= buffer:
                modified_values[i] = question.lower_bound + inset
                corrections_made = True
                logger.info(
                    "Clamped lower for Q %s: percentile %s value %s -> %s (tolerance %s)",
                    question.id_of_question,
                    percentile_list[i].percentile,
                    original_value,
                    modified_values[i],
                    buffer,
                )
            else:
                raise ValueError(
                    f"Value {original_value} too far below lower bound {question.lower_bound} (tolerance: {buffer})"
                )

        if not question.open_upper_bound and modified_values[i] > question.upper_bound:
            if modified_values[i] - question.upper_bound <= buffer:
                modified_values[i] = question.upper_bound - inset
                corrections_made = True
                logger.info(
                    "Clamped upper for Q %s: percentile %s value %s -> %s (tolerance %s)",
                    question.id_of_question,
                    percentile_list[i].percentile,
                    original_value,
                    modified_values[i],
                    buffer,
                )
            else:
                raise ValueError(
                    f"Value {original_value} too far above upper bound {question.upper_bound} (tolerance: {buffer})"
                )

    return modified_values, corrections_made


def log_heavy_clamping_diagnostics(
    modified_values: list[float],
    original_values: list[float],
    question: NumericQuestion,
) -> None:
    """Warn when more than half the values sit at a closed bound's inset, where the clamp puts them.

    Measured against ``_boundary_inset`` rather than the tolerance: with the tolerance as the
    window (a whole day on a 13-point date grid) a fully in-range forecast concentrated near a
    bound logged as heavily clamped, in exactly the logs the residual analysis reads.
    """
    if not original_values:
        return

    inset = _boundary_inset(question)
    clamped_lower = sum(
        1 for v in modified_values if not question.open_lower_bound and v <= question.lower_bound + inset
    )
    clamped_upper = sum(
        1 for v in modified_values if not question.open_upper_bound and v >= question.upper_bound - inset
    )

    if clamped_lower / len(original_values) > 0.5 or clamped_upper / len(original_values) > 0.5:
        logger.warning(
            "Heavy bound clamping for Q %s | URL %s | clamped_to_lower=%d%% | clamped_to_upper=%d%% | bounds=[%s, %s]",
            question.id_of_question,
            question.page_url,
            int(100 * clamped_lower / len(original_values)),
            int(100 * clamped_upper / len(original_values)),
            question.lower_bound,
            question.upper_bound,
        )


def log_corrections_summary(
    modified_values: list[float],
    original_values: list[float],
    question: NumericQuestion,
    corrections_made: bool,
) -> None:
    """Log summary of corrections made to the distribution."""
    if corrections_made or any(v != orig for v, orig in zip(modified_values, original_values, strict=False)):
        logger.warning(f"Corrected numeric distribution for question {question.id_of_question}")


def log_cluster_spreading_summary(
    modified_values: list[float],
    original_values: list[float],
    question: NumericQuestion,
    *,
    clusters_applied: int,
    spread_delta: float,
    count_like: bool,
) -> None:
    """Log summary of cluster spreading operations."""
    if clusters_applied > 0:
        pre_deltas = [b - a for a, b in pairwise(original_values)]
        post_deltas = [b - a for a, b in pairwise(modified_values)]

        min_value_delta_before = min(pre_deltas) if pre_deltas else float("inf")
        min_value_delta_after = min(post_deltas) if post_deltas else float("inf")

        logger.warning(
            "Cluster spread applied for Q %s | URL %s | clusters=%d | delta_used=%.6g | min_value_delta_before=%.6g | min_value_delta_after=%.6g | count_like=%s",
            question.id_of_question,
            question.page_url,
            clusters_applied,
            spread_delta,
            min_value_delta_before,
            min_value_delta_after,
            count_like,
        )
