"""Spread clustered percentile values for CDF smoothness."""

from __future__ import annotations

import logging

import numpy as np
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.constants import NUM_SPREAD_DELTA_MULT, NUM_VALUE_EPSILON_MULT
from metaculus_bot.numeric.config import (
    CLUSTER_DETECTION_ATOL,
    CLUSTER_SPREAD_BASE_DELTA,
    COUNT_LIKE_DELTA_MULTIPLIER,
    COUNT_LIKE_THRESHOLD,
    MIN_BOUNDARY_DISTANCE,
    STRICT_ORDERING_EPSILON,
)

logger = logging.getLogger(__name__)


def detect_count_like_pattern(values: list[float]) -> bool:
    """Detect if all values are near integers (count-like pattern)."""
    try:
        if not values:
            return False
        return all(abs(v - round(v)) <= COUNT_LIKE_THRESHOLD for v in values)
    except (TypeError, ValueError):
        return False


def is_degenerate_cluster(values: list[float], value_eps: float) -> bool:
    """True when EVERY declared value sits inside ONE ``value_eps`` cluster.

    That is a point mass: the model declared (near-)identical values at all
    percentiles, so the declaration carries no distribution width at all. It is
    matched with the same adjacent-gap chaining ``apply_cluster_spreading`` uses
    to grow a cluster, so the two agree on what "one cluster" means by
    construction.

    Callers need this separately from the spreader because the whole-set case is
    the one where spreading would INVENT the width instead of separating
    genuinely-plateaued neighbours: with no unclustered neighbour to compress
    against, the spread runs the full ``+-(k-1)/2 * spread_delta``, which on a
    count-like question is a full unit per position (a 13-percentile point mass
    on [0, 100] came out 12 units wide). That fabricated span is also exactly
    what let the point mass PASS ``detect_unit_mismatch``'s span-ratio test,
    which would otherwise have withheld the degenerate declaration.
    """
    if len(values) < 2:
        return False
    return all(abs(b - a) <= value_eps for a, b in zip(values, values[1:]))


def compute_cluster_parameters(
    range_size: float, count_like: bool, span: float | None = None
) -> tuple[float, float, float]:
    """Compute parameters for cluster detection and spreading."""
    value_eps = max(range_size * NUM_VALUE_EPSILON_MULT, CLUSTER_DETECTION_ATOL)
    base_delta = max(range_size * NUM_SPREAD_DELTA_MULT, CLUSTER_SPREAD_BASE_DELTA)
    # Prefer a spread relative to the raw span when available to avoid range-driven explosions
    if span is not None and span > 0:
        span_based = max(0.02 * span, CLUSTER_SPREAD_BASE_DELTA)
    else:
        span_based = base_delta
    spread_delta = max(base_delta, span_based, COUNT_LIKE_DELTA_MULTIPLIER if count_like else base_delta)
    return value_eps, base_delta, spread_delta


def apply_cluster_spreading(
    modified_values: list[float],
    question: NumericQuestion,
    value_eps: float,
    spread_delta: float,
    range_size: float,
) -> tuple[list[float], int]:
    """Spread epsilon-clustered values apart so the set can carry a CDF.

    Separates genuinely-plateaued neighbours (a count-like question where a model
    declares P20 = P40 = P50 = 1) from each other; it does NOT invent a
    distribution where the model declared none. A whole-set collapse — every
    value inside one epsilon cluster — is left ALONE and reported as 0 clusters
    applied: downstream jitter / strict-ordering give it the minimum separation
    the CDF format needs, and ``detect_unit_mismatch`` then sees the honest
    (essentially zero) span and withholds the forecaster. See
    ``is_degenerate_cluster``.
    """
    if is_degenerate_cluster(modified_values, value_eps):
        return modified_values, 0

    clusters_applied = 0
    i = 0

    while i < len(modified_values) - 1:
        j = i
        # Grow cluster while adjacent values within epsilon
        while j + 1 < len(modified_values) and abs(modified_values[j + 1] - modified_values[j]) <= value_eps:
            j += 1

        if j > i:
            # We have a cluster from i..j inclusive
            clusters_applied += 1
            k = j - i + 1

            # Base center value: mean of the cluster
            center = float(np.mean(modified_values[i : j + 1]))

            # Offsets: symmetric around center
            # Example for k=3: -d, 0, +d; for k=4: -1.5d, -0.5d, +0.5d, +1.5d
            offsets = [((idx - (k - 1) / 2.0) * spread_delta) for idx in range(k)]
            new_vals = [center + off for off in offsets]

            # Enforce bounds softly during spread to avoid later large clamps
            tiny = max(MIN_BOUNDARY_DISTANCE * range_size, CLUSTER_DETECTION_ATOL)
            if not question.open_lower_bound:
                new_vals = [max(v, question.lower_bound + tiny) for v in new_vals]
            if not question.open_upper_bound:
                new_vals = [min(v, question.upper_bound - tiny) for v in new_vals]

            # Apply while preserving non-decreasing relation to neighbors
            # If previous value exists and is >= first new, shift all up minimally
            if i - 1 >= 0 and new_vals[0] <= modified_values[i - 1]:
                shift = (modified_values[i - 1] + max(STRICT_ORDERING_EPSILON, value_eps)) - new_vals[0]
                new_vals = [v + shift for v in new_vals]

            # If next value exists and last new exceeds it, compress offsets
            if j + 1 < len(modified_values) and new_vals[-1] >= modified_values[j + 1]:
                # Compress spread to fit in available gap
                available = max(
                    modified_values[j + 1] - (new_vals[0]),
                    max(value_eps, STRICT_ORDERING_EPSILON),
                )
                if k > 1:
                    step = available / k
                    new_vals = [new_vals[0] + step * idx for idx in range(k)]

            # Assign new values
            for t in range(k):
                modified_values[i + t] = new_vals[t]

            i = j + 1
        else:
            i += 1

    return modified_values, clusters_applied


def apply_jitter_for_duplicates(
    modified_values: list[float],
    question: NumericQuestion,
    range_size: float,
    percentile_list: list[Percentile],
) -> list[float]:
    """Apply jitter to eliminate any remaining duplicate values."""
    for i in range(1, len(modified_values)):
        if modified_values[i] <= modified_values[i - 1]:
            epsilon = max(MIN_BOUNDARY_DISTANCE * range_size, STRICT_ORDERING_EPSILON)
            target = modified_values[i - 1] + epsilon

            if not question.open_upper_bound:
                target = min(target, question.upper_bound - epsilon)

            # Increase if possible; otherwise allow equality (PCHIP will handle de-dup)
            new_val = max(modified_values[i], target)

            # Also respect lower bound on closed lower
            if not question.open_lower_bound:
                new_val = max(new_val, question.lower_bound + epsilon)

            modified_values[i] = new_val
            logger.debug(
                f"Applied jitter: percentile {percentile_list[i].percentile} value {modified_values[i]} -> {new_val}"
            )

    return modified_values


def ensure_strictly_increasing_bounded(
    modified_values: list[float], question: NumericQuestion, range_size: float
) -> list[float]:
    """Final pass to ensure all values are strictly increasing within bounds."""
    epsilon = max(MIN_BOUNDARY_DISTANCE * range_size, STRICT_ORDERING_EPSILON)

    # Re-ensure increasing after clamping, bounded (left-to-right)
    for i in range(1, len(modified_values)):
        if modified_values[i] <= modified_values[i - 1]:
            target = modified_values[i - 1] + epsilon
            if not question.open_upper_bound:
                target = min(target, question.upper_bound - epsilon)
            if not question.open_lower_bound:
                target = max(target, question.lower_bound + epsilon)
            modified_values[i] = max(modified_values[i], target)

    # Additional pass (right-to-left) to make room near closed upper bound
    # If upper bound is closed and strict increase is capped, slide earlier values down by epsilon
    for i in range(len(modified_values) - 2, -1, -1):
        if modified_values[i] >= modified_values[i + 1]:
            target = modified_values[i + 1] - epsilon
            if not question.open_lower_bound:
                target = max(target, question.lower_bound + epsilon)
            modified_values[i] = min(modified_values[i], target)

    return modified_values
