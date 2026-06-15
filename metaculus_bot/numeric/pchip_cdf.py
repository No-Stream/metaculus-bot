"""
PCHIP-based CDF construction for robust numeric forecasting.

Based on the battle-tested implementation from panchul (Q2 2025 competition winner).
Provides smooth, monotonic CDF construction with strict constraints enforcement.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import PchipInterpolator

from metaculus_bot.constants import NUM_MAX_STEP

logger = logging.getLogger(__name__)


def _redistribute_excess_probability(cdf: np.ndarray, max_step: float) -> np.ndarray:
    """
    Redistribute probability mass so that no single step exceeds max_step while
    preserving the original total mass and monotonicity.
    """
    if cdf.size <= 1:
        return cdf

    steps = np.diff(cdf)
    if not np.any(steps > max_step + 1e-12):
        return cdf

    original_total = float(steps.sum())
    steps = np.clip(steps, 0.0, max_step)
    deficit = original_total - float(steps.sum())

    iteration = 0
    max_iterations = max(5, steps.size * 5)

    while deficit > 1e-12 and iteration < max_iterations:
        slack = max_step - steps
        positive_slack = slack > 1e-12
        if not np.any(positive_slack):
            # No room left to redistribute
            break

        allocation = np.zeros_like(steps)
        slack_sum = float(slack[positive_slack].sum())
        if slack_sum <= 1e-18:
            break

        allocation[positive_slack] = deficit * slack[positive_slack] / slack_sum
        allocation = np.minimum(allocation, slack)

        steps += allocation
        deficit = original_total - float(steps.sum())
        iteration += 1

    if deficit > 1e-8:
        raise RuntimeError(
            f"Failed to redistribute CDF probability mass within max step constraint "
            f"(remaining deficit={deficit:.12f}, iterations={iteration}, max_step={max_step})"
        )

    new_cdf = np.empty_like(cdf)
    new_cdf[0] = cdf[0]
    new_cdf[1:] = cdf[0] + np.cumsum(steps)
    return new_cdf


def safe_cdf_bounds(cdf: np.ndarray, open_lower: bool, open_upper: bool) -> np.ndarray:
    """
    Ensure CDF respects Metaculus boundary constraints:
    • For *open* bounds: cdf[0] ≥ 0.001, cdf[-1] ≤ 0.999
    • No single step may exceed 0.2
    """
    # Work on a copy to avoid mutating callers unexpectedly
    cdf = cdf.copy()

    # Pin tails to legal open-bound limits
    if open_lower:
        cdf[0] = max(cdf[0], 0.001)
    if open_upper:
        cdf[-1] = min(cdf[-1], 0.999)

    # Enforce the maximum step rule iteratively
    pre_max_step = float(np.max(np.diff(cdf))) if cdf.size > 1 else 0.0
    if pre_max_step > NUM_MAX_STEP + 1e-12:
        cdf = _redistribute_excess_probability(cdf, NUM_MAX_STEP)
        post_max_step = float(np.max(np.diff(cdf))) if cdf.size > 1 else 0.0
        logger.debug(
            "CDF max-step redistribution applied | pre_max_step=%.8f | post_max_step=%.8f | max_step=%.8f",
            pre_max_step,
            post_max_step,
            NUM_MAX_STEP,
        )

    # Ensure monotonicity and clamp to legal probability range
    np.maximum.accumulate(cdf, out=cdf)
    np.clip(cdf, 0.0, 1.0, out=cdf)

    # Re-apply open bounds in case redistribution nudged them
    if open_lower:
        cdf[0] = max(cdf[0], 0.001)
    if open_upper:
        cdf[-1] = min(cdf[-1], 0.999)

    if cdf.size > 1:
        np.maximum.accumulate(cdf, out=cdf)

    return cdf


def enforce_strict_increasing(
    percentile_dict: dict[int | float, float],
) -> dict[int | float, float]:
    """Ensure strictly increasing values by adding tiny jitter if necessary."""
    sorted_items = sorted(percentile_dict.items())
    last_val = -float("inf")
    new_dict = {}

    for p, v in sorted_items:
        if v <= last_val:
            v = last_val + 1e-8  # Add a tiny epsilon
        new_dict[p] = v
        last_val = v

    return new_dict


def enforce_min_steps(
    y_values: np.ndarray,
    min_step: float,
    *,
    upper_cap: float = 1.0,
    lower_cap: float = 0.0,
) -> np.ndarray:
    """Enforce minimum step size between adjacent points (panchul-style sweep).

    Forward pass lifts each point to be at least ``prev + min_step`` (capped at
    ``upper_cap``). When the upper cap pins the last point before the grid
    ends, a backward pass pulls earlier points down so every step still meets
    ``min_step``. Used by both the PCHIP pipeline and the mixture-CDF builder
    to keep CDFs strictly increasing under the Metaculus min-step constraint.
    """
    n = len(y_values)
    result = y_values.copy()
    for i in range(1, n):
        if result[i] < result[i - 1] + min_step:
            result[i] = result[i - 1] + min_step
        if result[i] > upper_cap:
            result[i] = upper_cap
    for j in range(n - 2, -1, -1):
        if result[j] > result[j + 1] - min_step:
            result[j] = result[j + 1] - min_step
        if result[j] < lower_cap:
            result[j] = lower_cap
    return result


def build_cdf_value_grid(
    lower_bound: float,
    upper_bound: float,
    zero_point: float | None,
    num_points: int = 201,
) -> np.ndarray:
    """Build the CDF evaluation value grid: linear for linear-scaled questions, geometric
    when ``zero_point`` is set (log-scaled questions).

    This is the canonical grid the production CDF lives on and the grid the Metaculus
    scorer buckets resolutions against (see ``scoring_common.resolution_to_bucket_index``).
    Offline pooling primitives that consume x-values (Vincentization, CRPS) must build their
    grid here so a zero_point question's pooled CDF aligns with the scorer's buckets.

    The geometric branch matches the Metaculus backend's non-linear spacing:
    ``x(t) = lower + (upper - lower) * (ratio**t - 1) / (ratio - 1)`` where
    ``ratio = (upper - zero_point) / (lower - zero_point)`` and ``t`` ranges over a uniform
    [0, 1] grid of ``num_points`` points.
    """
    t = np.linspace(0, 1, num_points)

    if zero_point is None:
        # Linear grid
        return lower_bound + (upper_bound - lower_bound) * t

    # Non-linear grid based on zero_point
    ratio = (upper_bound - zero_point) / (lower_bound - zero_point)
    # Handle potential numerical issues
    if abs(ratio - 1.0) < 1e-10:
        return lower_bound + (upper_bound - lower_bound) * t
    return np.array([lower_bound + (upper_bound - lower_bound) * ((ratio**tt - 1) / (ratio - 1)) for tt in t])


def generate_pchip_cdf(
    percentile_values: dict[int | float, float],
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None = None,
    *,
    min_step: float = 5.0e-5,
    num_points: int = 201,
    question_id: int | str | None = None,
    question_url: str | None = None,
) -> tuple[list[float], bool]:
    """
    Generate a robust continuous CDF using PCHIP interpolation with strict constraint enforcement.

    Based on the panchul implementation with enhancements for robustness. ``percentile_values``
    maps percentiles in (0, 100) to values; ``zero_point`` enables non-linear grid scaling.
    Returns ``(cdf_values, aggressive_enforcement_used)`` where the second element flags whether
    aggressive step enforcement was required to satisfy the min-step constraint.

    Raises:
        ValueError: If input validation fails
        RuntimeError: If constraint enforcement fails
    """
    # Validate inputs
    if not percentile_values:
        raise ValueError("Empty percentile values dictionary")

    if upper_bound <= lower_bound:
        raise ValueError(f"Upper bound ({upper_bound}) must be greater than lower bound ({lower_bound})")

    if zero_point is not None:
        if abs(zero_point - lower_bound) < 1e-6 or abs(zero_point - upper_bound) < 1e-6:
            raise ValueError(f"zero_point ({zero_point}) too close to bounds [{lower_bound}, {upper_bound}]")

    # Clean and validate percentile values
    pv = {}
    for k, v in percentile_values.items():
        try:
            k_float = float(k)
            v_float = float(v)

            if not (0 < k_float < 100):
                continue  # Skip invalid percentiles

            if not np.isfinite(v_float):
                continue  # Skip non-finite values

            pv[k_float] = v_float
        except (ValueError, TypeError):
            continue  # Skip non-numeric entries

    if len(pv) < 2:
        raise ValueError(f"Need at least 2 valid percentile points (got {len(pv)})")

    # Handle duplicate values by adding small offsets
    # First, sort all items to process in order
    sorted_items = sorted(pv.items())
    last_value = -float("inf")

    for k, v in sorted_items:
        if v <= last_value:
            # Add a small epsilon to ensure strictly increasing
            v = last_value + 1e-9
        pv[k] = v
        last_value = v

    # Create arrays of percentiles and values
    percentiles, values = zip(*sorted(pv.items()))
    percentiles = np.array(percentiles) / 100.0  # Convert to [0,1] range
    values = np.array(values)

    # Check if values are strictly increasing after de-duplication
    if np.any(np.diff(values) <= 0):
        raise ValueError("Percentile values must be strictly increasing after de-duplication")

    # Add boundary points if needed
    if not open_lower_bound and lower_bound < values[0] - 1e-9:
        percentiles = np.insert(percentiles, 0, 0.0)
        values = np.insert(values, 0, lower_bound)

    if not open_upper_bound and upper_bound > values[-1] + 1e-9:
        percentiles = np.append(percentiles, 1.0)
        values = np.append(values, upper_bound)

    # Determine if log scaling is appropriate (all values positive and lower bound > 0)
    use_log = np.all(values > 0) and zero_point is None and lower_bound > 0
    x_vals = np.log(values) if use_log else values

    # Create interpolator with fallback
    try:
        spline = PchipInterpolator(x_vals, percentiles, extrapolate=True)
    except Exception:
        logger.warning("PchipInterpolator failed, falling back to linear interpolation", exc_info=True)

        def spline(x):
            return np.interp(x, x_vals, percentiles)

    # Generate the grid (linear, or geometric when zero_point is set) and evaluate.
    cdf_x = build_cdf_value_grid(lower_bound, upper_bound, zero_point, num_points)

    # Handle log transformation for evaluation
    eval_x = np.log(cdf_x) if use_log else cdf_x

    # Clamp values to avoid extrapolation issues
    eval_x_clamped = np.clip(eval_x, x_vals[0], x_vals[-1])

    # Generate initial CDF values and clamp to [0,1]
    cdf_y = spline(eval_x_clamped).clip(0.0, 1.0)

    # Ensure monotonicity (non-decreasing)
    cdf_y = np.maximum.accumulate(cdf_y)

    # Set boundary values if bounds are closed
    if not open_lower_bound:
        cdf_y[0] = 0.0
    if not open_upper_bound:
        cdf_y[-1] = 1.0

    # Uniform mixture for min-step compliance (same approach as discrete_snap.py)
    _ALPHA_SAFETY_MARGIN = 1.1
    total_range = float(cdf_y[-1] - cdf_y[0])
    if total_range > 1e-12:
        min_alpha = min_step * num_points / total_range * _ALPHA_SAFETY_MARGIN
    else:
        min_alpha = 1.0
    alpha = min(1.0, min_alpha)
    uniform_cdf = np.linspace(float(cdf_y[0]), float(cdf_y[-1]), num_points)
    cdf_y = (1.0 - alpha) * cdf_y + alpha * uniform_cdf

    # First pass: enforce min-step via the shared module-level helper.
    cdf_y = enforce_min_steps(cdf_y, min_step, upper_cap=1.0, lower_cap=0.0)

    # Second pass (legacy panchul redistribution): if the CDF saturated to 1.0
    # before the grid endpoint, ramp the remaining points uniformly back up to
    # 1.0 then re-enforce min-step with a back-fill on overflow. Kept inline
    # because the shape of the redistribution is specific to the PCHIP path.
    if cdf_y[-1] > 1.0:
        overflow_idx_arr = np.where(cdf_y > 1.0)[0]
        if len(overflow_idx_arr) > 0:
            overflow_idx = overflow_idx_arr[0]
            steps_remaining = len(cdf_y) - overflow_idx

            for i in range(overflow_idx, len(cdf_y)):
                t = (i - overflow_idx) / max(1, steps_remaining - 1)
                cdf_y[i] = min(
                    1.0,
                    cdf_y[overflow_idx - 1] + (1.0 - cdf_y[overflow_idx - 1]) * t,
                )

            for i in range(overflow_idx, len(cdf_y)):
                if i > overflow_idx and cdf_y[i] < cdf_y[i - 1] + min_step:
                    cdf_y[i] = cdf_y[i - 1] + min_step
                    if cdf_y[i] > 1.0:
                        cdf_y[i] = 1.0
                        for j in range(i - 1, overflow_idx - 1, -1):
                            max_allowed = cdf_y[j + 1] - min_step
                            if cdf_y[j] > max_allowed:
                                cdf_y[j] = max_allowed

    # Apply boundary constraints and max jump rules
    cdf_y = safe_cdf_bounds(cdf_y, open_lower_bound, open_upper_bound)

    # Check if we have enough room for minimum steps
    required_range = (len(cdf_y) - 1) * min_step
    available_range = cdf_y[-1] - cdf_y[0]

    # Double-check minimum step size requirement
    steps = np.diff(cdf_y)
    aggressive_enforcement_used = False
    if np.any(steps < min_step):
        aggressive_enforcement_used = True
        # Log detailed context before aggressive enforcement
        violated_steps = np.sum(steps < min_step)
        min_violated_step = np.min(steps)
        violation_percentage = 100.0 * violated_steps / len(steps)

        logger.warning(
            "PCHIP minimum step enforcement required for Q %s | URL %s | violated_steps=%d/%d (%.1f%%) | min_step_found=%.8f | min_step_required=%.8f | available_range=%.6f | required_range=%.6f",
            question_id or "N/A",
            question_url or "N/A",
            violated_steps,
            len(steps),
            violation_percentage,
            min_violated_step,
            min_step,
            available_range,
            required_range,
        )

        # Create a strictly monotonic sequence
        if not open_lower_bound:
            start_val = 0.0
        else:
            start_val = cdf_y[0]

        if not open_upper_bound:
            end_val = 1.0
        else:
            end_val = min(cdf_y[-1], 1.0)

        available_range = end_val - start_val
        # Ensure we have enough room for all steps
        required_range = (len(cdf_y) - 1) * min_step

        if required_range > available_range:
            # We don't have enough room for minimum steps
            raise ValueError(
                f"Cannot satisfy minimum step requirement: need {required_range:.6f} "
                f"but only have {available_range:.6f} available in CDF range"
            )

        # Create a new CDF with exactly min_step between points where needed
        # and distribute remaining range proportionally
        new_cdf = np.zeros_like(cdf_y)
        new_cdf[0] = start_val

        # Get the shape from original CDF but enforce minimum steps
        if len(cdf_y) > 2:
            # Calculate normalized shape from original CDF
            orig_shape = np.diff(cdf_y)
            orig_shape = np.maximum(orig_shape, min_step)  # Enforce minimum
            orig_shape = orig_shape / np.sum(orig_shape)  # Normalize

            # Allocate the available range according to shape but ensure minimum steps
            remaining = available_range - (len(cdf_y) - 1) * min_step
            extra_steps = remaining * orig_shape

            for i in range(1, len(new_cdf)):
                new_cdf[i] = new_cdf[i - 1] + min_step + extra_steps[i - 1]
        else:
            # Simple linear spacing if original shape is unavailable
            for i in range(1, len(new_cdf)):
                new_cdf[i] = new_cdf[i - 1] + (available_range / (len(new_cdf) - 1))

        # Final validation
        if np.any(np.diff(new_cdf) < min_step - 1e-10):
            raise RuntimeError("Internal error: Step size enforcement failed")

        cdf_y = new_cdf

        # Log successful aggressive enforcement
        new_steps = np.diff(cdf_y)
        new_min_step = np.min(new_steps)
        new_max_step = np.max(new_steps)
        total_range_redistributed = available_range

        logger.info(
            "PCHIP aggressive enforcement completed for Q %s | URL %s | new_min_step=%.8f | new_max_step=%.8f | total_range_redistributed=%.6f | shape_preserved=True",
            question_id or "N/A",
            question_url or "N/A",
            new_min_step,
            new_max_step,
            total_range_redistributed,
        )

    # Final checks
    if np.any(np.diff(cdf_y) < min_step - 1e-10):
        problematic_indices = np.where(np.diff(cdf_y) < min_step - 1e-10)[0]
        error_msg = (
            f"Failed to enforce minimum step size at indices: {problematic_indices}, "
            f"values: {np.diff(cdf_y)[problematic_indices]}"
        )
        raise RuntimeError(error_msg)

    if not open_lower_bound and abs(cdf_y[0]) > 1e-10:
        raise RuntimeError(f"Failed to enforce lower bound: {cdf_y[0]}")

    if not open_upper_bound and abs(cdf_y[-1] - 1.0) > 1e-10:
        raise RuntimeError(f"Failed to enforce upper bound: {cdf_y[-1]}")

    return cdf_y.tolist(), aggressive_enforcement_used


def percentiles_to_pchip_format(percentiles: list) -> dict[float, float]:
    """
    Convert forecasting-tools Percentile objects to PCHIP input format.

    Args:
        percentiles: List of Percentile objects with .percentile and .value attributes

    Returns:
        Dictionary mapping percentile (0-100) to value
    """
    result = {}
    for p in percentiles:
        percentile_key = p.percentile * 100  # Convert from [0,1] to [0,100]
        result[percentile_key] = p.value
    return result
