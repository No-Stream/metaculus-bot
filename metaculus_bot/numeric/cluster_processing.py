"""Spread clustered percentile values for CDF smoothness."""

from __future__ import annotations

import logging
from itertools import pairwise

import numpy as np
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.constants import NUM_SPREAD_DELTA_MULT, NUM_VALUE_EPSILON_MULT
from metaculus_bot.numeric.config import (
    CLUSTER_DETECTION_ATOL,
    CLUSTER_SPREAD_BASE_DELTA,
    COUNT_LIKE_DELTA_MULTIPLIER,
    COUNT_LIKE_THRESHOLD,
    STRICT_ORDERING_EPSILON,
    grid_bin_width,
    grid_is_outcome_space,
    minimum_separation,
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

    That is a point mass: the model declared (near-)identical values at all percentiles, so
    the declaration carries no distribution width. Matched with the same adjacent-gap
    chaining ``apply_cluster_spreading`` uses to grow a cluster, so the two agree on what
    "one cluster" means by construction.

    Callers need it separately from the spreader because on the 201-point continuous grid
    the whole-set case is the one where spreading would INVENT the width (a 13-percentile
    point mass on [0, 100] came out 12 units wide), and that fabricated span is exactly what
    let a point mass PASS ``detect_unit_mismatch``. Where the grid's bins are the outcome
    space the spreader does spread it, under its one-bin cap (``apply_cluster_spreading``).
    """
    if len(values) < 2:
        return False
    return all(abs(b - a) <= value_eps for a, b in pairwise(values))


def compute_cluster_parameters(
    range_size: float, count_like: bool, span: float | None = None
) -> tuple[float, float, float]:
    """Compute parameters for cluster detection and spreading."""
    value_eps = max(range_size * NUM_VALUE_EPSILON_MULT, CLUSTER_DETECTION_ATOL)
    base_delta = max(range_size * NUM_SPREAD_DELTA_MULT, CLUSTER_SPREAD_BASE_DELTA)
    # Prefer a spread relative to the raw span when available to avoid range-driven explosions
    span_based = max(0.02 * span, CLUSTER_SPREAD_BASE_DELTA) if span is not None and span > 0 else base_delta
    spread_delta = max(base_delta, span_based, COUNT_LIKE_DELTA_MULTIPLIER if count_like else base_delta)
    return value_eps, base_delta, spread_delta


def _cluster_end_index(values: list[float], start: int, value_eps: float) -> int:
    """Last index of the epsilon-chained run beginning at ``start`` (``start`` if none)."""
    end = start
    while end + 1 < len(values) and abs(values[end + 1] - values[end]) <= value_eps:
        end += 1
    return end


def _symmetric_plateau(center: float, size: int, spread_delta: float) -> list[float]:
    """``size`` values ``spread_delta`` apart, symmetric about ``center`` (size 3: -d, 0, +d; size 4: -1.5d..+1.5d)."""
    return [center + (idx - (size - 1) / 2.0) * spread_delta for idx in range(size)]


def _shifted_inside(new_vals: list[float], low: float, high: float) -> list[float]:
    """``new_vals`` translated, spacing intact, until it lies within ``[low, high]``; the moved end lands exactly on the bound."""
    if new_vals[0] < low:
        return [low + (v - new_vals[0]) for v in new_vals]
    if new_vals[-1] > high:
        return [high - (new_vals[-1] - v) for v in new_vals]
    return new_vals


def _declared_beyond_an_open_bound(declared: float, question: NumericQuestion) -> bool:
    """True when the declared value lies strictly outside an OPEN bound: real out-of-range mass."""
    return (question.open_lower_bound and declared < question.lower_bound) or (
        question.open_upper_bound and declared > question.upper_bound
    )


def _spread_whole_set_collapse(values: list[float], question: NumericQuestion, *, spread_delta: float) -> list[float]:
    """The whole-set collapse (``is_degenerate_cluster``) placed by its declared value on an outcome-space grid.

    The declared value is the median ELEMENT of the sorted set, compared exactly against the
    bounds: an exactly-equal collapse compares exactly, an epsilon-chain of near-equal values is
    centred rather than placed off its first value, and no mean is taken (``np.mean([1.3] * 13)``
    is ``1.3000000000000003``, which read a collapse ON an open bound of 1.3 as beyond it). Inside
    the range or exactly on a bound the full span goes inside, under the one-bin cap and
    translated as needed, because the bound value buckets into the terminal bin. Strictly
    beyond an OPEN bound the spread stays symmetric about the value, a real out-of-range
    declaration. Strictly beyond a CLOSED bound the plateau starts at the value, so
    ``clamp_values_to_bounds``, which runs after this, judges the declared distance. A collapse
    has no neighbours, so no repair re-spaces it. Receipts: ``docs/numeric_pipeline.md`` Step 3.
    """
    size = len(values)
    declared = sorted(values)[size // 2]
    bin_width = grid_bin_width(question.lower_bound, question.upper_bound, question.cdf_size)
    plateau = _symmetric_plateau(declared, size, min(spread_delta, bin_width / (size - 1)))
    if _declared_beyond_an_open_bound(declared, question):
        return plateau
    return _shifted_inside(plateau, min(question.lower_bound, declared), max(question.upper_bound, declared))


def _spread_cluster_values(
    values: list[float],
    start: int,
    end: int,
    question: NumericQuestion,
    *,
    value_eps: float,
    spread_delta: float,
    range_size: float,
) -> list[float]:
    """Replacement values for the PARTIAL cluster ``values[start:end + 1]``, symmetric about its mean.

    Clamped to the ``minimum_separation`` standoff at a CLOSED edge, then shifted up if it would
    collide with the preceding value, then compressed if it would overrun the following one, so
    the spread never reorders the set. Where the published bins are the outcome space
    (``grid_is_outcome_space``) the plateau's total spread is first capped at one bin width: the
    grid points are bin edges, so a plateau at integer k that spilled past k +- 0.5 handed the
    mass the forecaster put on k to the neighbouring bins (Mantic post 253 published 0.558 for a
    declared 90%). Nothing else differs by grid: on an OPEN bound a partial plateau may straddle
    it, the behaviour benchmarked on Metaculus and deliberately kept (three rounds of
    translating partial plateaus each opened a new hole; ``docs/numeric_pipeline.md`` Step 3).
    The whole-set collapse never reaches this: ``apply_cluster_spreading`` routes it to
    ``_spread_whole_set_collapse``.
    """
    size = end - start + 1
    if grid_is_outcome_space(question):
        bin_width = grid_bin_width(question.lower_bound, question.upper_bound, question.cdf_size)
        spread_delta = min(spread_delta, bin_width / (size - 1))

    center = float(np.mean(values[start : end + 1]))
    new_vals = _symmetric_plateau(center, size, spread_delta)

    # Enforce bounds softly during spread to avoid later large clamps
    tiny = minimum_separation(range_size)
    if not question.open_lower_bound:
        new_vals = [max(v, question.lower_bound + tiny) for v in new_vals]
    if not question.open_upper_bound:
        new_vals = [min(v, question.upper_bound - tiny) for v in new_vals]

    # If a previous value exists and is >= first new, shift all up minimally
    if start - 1 >= 0 and new_vals[0] <= values[start - 1]:
        shift = (values[start - 1] + max(STRICT_ORDERING_EPSILON, value_eps)) - new_vals[0]
        new_vals = [v + shift for v in new_vals]

    # If a next value exists and the last new exceeds it, compress into the available gap
    if end + 1 < len(values) and new_vals[-1] >= values[end + 1]:
        available = max(values[end + 1] - new_vals[0], value_eps, STRICT_ORDERING_EPSILON)
        if size > 1:
            step = available / size
            new_vals = [new_vals[0] + step * idx for idx in range(size)]

    return new_vals


def apply_cluster_spreading(
    modified_values: list[float],
    question: NumericQuestion,
    *,
    value_eps: float,
    spread_delta: float,
    range_size: float,
) -> tuple[list[float], int]:
    """Spread epsilon-clustered values apart so the set can carry a CDF.

    Separates genuinely-plateaued neighbours (a count-like question where a model declares
    P20 = P40 = P50 = 1); it does NOT invent a distribution where the model declared none.
    On the 201-point continuous grid a whole-set collapse (``is_degenerate_cluster``) is
    left ALONE and reported as 0 clusters applied: the jitter / strict-ordering passes give
    it the format minimum and ``detect_unit_mismatch`` then sees the honest (zero) span and
    withholds the forecaster. Where the published bins are the outcome space
    (``grid_is_outcome_space``) the whole-set case is placed by ``_spread_whole_set_collapse``
    under the one-bin cap: "100% on 2026-09-16" is fully expressible on post 651's twelve
    one-day bins, so the member publishes with its mass in that day. Partial plateaus take
    ``_spread_cluster_values`` on every grid. Receipts: ``docs/numeric_pipeline.md`` Step 3.

    Mutates and returns ``modified_values``.
    """
    if is_degenerate_cluster(modified_values, value_eps):
        if not grid_is_outcome_space(question):
            return modified_values, 0
        modified_values[:] = _spread_whole_set_collapse(modified_values, question, spread_delta=spread_delta)
        return modified_values, 1

    clusters_applied = 0
    i = 0

    while i < len(modified_values) - 1:
        cluster_end = _cluster_end_index(modified_values, i, value_eps)
        if cluster_end == i:
            i += 1
            continue

        clusters_applied += 1
        new_vals = _spread_cluster_values(
            modified_values,
            i,
            cluster_end,
            question,
            value_eps=value_eps,
            spread_delta=spread_delta,
            range_size=range_size,
        )
        modified_values[i : cluster_end + 1] = new_vals
        i = cluster_end + 1

    return modified_values, clusters_applied


def apply_jitter_for_duplicates(
    modified_values: list[float],
    question: NumericQuestion,
    range_size: float,
    percentile_list: list[Percentile],
) -> list[float]:
    """Apply jitter to eliminate any remaining duplicate values."""
    epsilon = minimum_separation(range_size)
    for i in range(1, len(modified_values)):
        if modified_values[i] <= modified_values[i - 1]:
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
    epsilon = minimum_separation(range_size)

    # Re-ensure increasing after clamping, bounded (left-to-right)
    for i in range(1, len(modified_values)):
        if modified_values[i] <= modified_values[i - 1]:
            target = modified_values[i - 1] + epsilon
            if not question.open_upper_bound:
                target = min(target, question.upper_bound - epsilon)
            if not question.open_lower_bound:
                target = max(target, question.lower_bound + epsilon)
            modified_values[i] = max(modified_values[i], target)

    # Right-to-left: where a closed upper bound capped the increase, slide earlier values down by epsilon
    for i in range(len(modified_values) - 2, -1, -1):
        if modified_values[i] >= modified_values[i + 1]:
            target = modified_values[i + 1] - epsilon
            if not question.open_lower_bound:
                target = max(target, question.lower_bound + epsilon)
            modified_values[i] = min(modified_values[i], target)

    return modified_values
