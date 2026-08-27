"""
PCHIP-based CDF construction for robust numeric forecasting.

Based on the battle-tested implementation from panchul (Q2 2025 competition winner).
Provides smooth, monotonic CDF construction with strict constraints enforcement.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import PchipInterpolator

from metaculus_bot.constants import NUM_MAX_STEP, NUM_MIN_PROB_STEP

logger = logging.getLogger(__name__)

# Headroom on the uniform-mixture weight so the blended CDF clears the min-step
# constraint rather than landing exactly on it (mirrors ``discrete_snap``).
_ALPHA_SAFETY_MARGIN: float = 1.1


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


def safe_cdf_bounds(
    cdf: np.ndarray,
    open_lower: bool,
    open_upper: bool,
    *,
    min_step: float = NUM_MIN_PROB_STEP,
    max_step: float = NUM_MAX_STEP,
) -> np.ndarray:
    """
    Ensure CDF respects Metaculus boundary constraints:
    • For *open* bounds: cdf[0] ≥ 0.001, cdf[-1] ≤ 0.999
    • No single step may exceed ``max_step``
    • Adjacent steps stay ≥ ``min_step`` (re-enforced after pin+cummax)

    ``min_step`` / ``max_step`` default to the 201-point-grid constants
    (``NUM_MIN_PROB_STEP`` / ``NUM_MAX_STEP``). A caller building a CDF on a non-201 grid
    (a discrete question with ``cdf_size != 201``) must pass the grid-scaled values so the
    per-bin constraints match the server's ``round(0.01 / N, 9)`` min-step and
    ``0.2 * 200 / N`` max-step formulas, where ``N = cdf_size - 1``.
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
    if pre_max_step > max_step + 1e-12:
        cdf = _redistribute_excess_probability(cdf, max_step)
        post_max_step = float(np.max(np.diff(cdf))) if cdf.size > 1 else 0.0
        logger.debug(
            "CDF max-step redistribution applied | pre_max_step=%.8f | post_max_step=%.8f | max_step=%.8f",
            pre_max_step,
            post_max_step,
            max_step,
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
        # Pinning cdf[0] up to 0.001 + cummax flattens any sub-0.001 prefix into
        # 0-step bins, violating the server's min-step (the framework then
        # drops the prediction on open-bound fallback questions). Re-enforce.
        upper_cap = 0.999 if open_upper else 1.0
        lower_cap = 0.001 if open_lower else 0.0
        cdf = enforce_min_steps(cdf, min_step, upper_cap=upper_cap, lower_cap=lower_cap)

    return cdf


def enforce_strict_increasing(
    percentile_dict: dict[int | float, float],
) -> dict[int | float, float]:
    """Ensure strictly increasing values by adding tiny jitter if necessary."""
    sorted_items = sorted(percentile_dict.items())
    last_val = -float("inf")
    new_dict = {}

    for p, v in sorted_items:
        # A tiny epsilon above the previous value when the declaration repeats or dips.
        adjusted = last_val + 1e-8 if v <= last_val else v
        new_dict[p] = adjusted
        last_val = adjusted

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
    return lower_bound + (upper_bound - lower_bound) * ((ratio**t - 1) / (ratio - 1))


def _validate_pchip_bounds(
    percentile_values: dict[int | float, float],
    lower_bound: float,
    upper_bound: float,
    zero_point: float | None,
) -> None:
    """Reject inputs the interpolation cannot be built on at all."""
    if not percentile_values:
        raise ValueError("Empty percentile values dictionary")

    if upper_bound <= lower_bound:
        raise ValueError(f"Upper bound ({upper_bound}) must be greater than lower bound ({lower_bound})")

    if zero_point is not None and (abs(zero_point - lower_bound) < 1e-6 or abs(zero_point - upper_bound) < 1e-6):
        raise ValueError(f"zero_point ({zero_point}) too close to bounds [{lower_bound}, {upper_bound}]")


def _clean_percentile_values(percentile_values: dict[int | float, float]) -> dict[float, float]:
    """Drop out-of-range percentile LABELS and reject unusable percentile VALUES.

    The KEY filter is a genuine filter and must stay one: ``_postprocess_ensemble_cdf``'s
    discrete branch deliberately passes labels of 0.0 and 100.0 (prob*100 over a 0..1
    span) and relies on them being dropped here before the boundary points are re-added.
    Raising on those would break every discrete question.

    A bad VALUE is the opposite case and raises. Silently skipping it built a
    12-of-13-point CDF while ``declared_percentiles`` still advertised 13 — a distribution
    missing an anchor the model declared, with nothing recording the loss. It is
    reachable: ``json.loads`` accepts a bare ``NaN``, and the strictly-increasing check
    (``value <= prev``) is False for NaN, so NaN used to pass the block schema and publish.
    Every caller already handles ValueError (``build_numeric_distribution`` falls back to
    forecasting-tools' builder, which re-validates), and the extraction ladder's own
    finiteness check now closes the upstream path.
    """
    cleaned: dict[float, float] = {}
    for label, raw_value in percentile_values.items():
        try:
            label_float = float(label)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"non-numeric percentile label {label!r} in declared percentiles") from exc
        if not (0 < label_float < 100):
            continue  # Boundary/out-of-range labels: dropped by design (see above).
        try:
            value_float = float(raw_value)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"non-numeric value {raw_value!r} at percentile {label_float}") from exc
        if not np.isfinite(value_float):
            raise ValueError(f"non-finite value {value_float} at percentile {label_float}")
        cleaned[label_float] = value_float

    if len(cleaned) < 2:
        raise ValueError(f"Need at least 2 valid percentile points (got {len(cleaned)})")
    return cleaned


def _nudge_duplicate_values_apart(percentile_values: dict[float, float]) -> dict[float, float]:
    """Offset repeated values by 1e-9 so PCHIP sees a strictly increasing x-axis."""
    sorted_items = sorted(percentile_values.items())
    last_value = -float("inf")

    for label, value in sorted_items:
        adjusted = last_value + 1e-9 if value <= last_value else value
        percentile_values[label] = adjusted
        last_value = adjusted

    return percentile_values


def _percentile_arrays_with_boundaries(
    percentile_values: dict[float, float],
    *,
    open_lower_bound: bool,
    open_upper_bound: bool,
    lower_bound: float,
    upper_bound: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the declaration into (percentile fractions, values), adding closed-bound anchors."""
    percentile_labels, declared_values = zip(*sorted(percentile_values.items()), strict=False)
    percentiles = np.array(percentile_labels) / 100.0  # Convert to [0,1] range
    values = np.array(declared_values)

    if np.any(np.diff(values) <= 0):
        raise ValueError("Percentile values must be strictly increasing after de-duplication")

    if not open_lower_bound and lower_bound < values[0] - 1e-9:
        percentiles = np.insert(percentiles, 0, 0.0)
        values = np.insert(values, 0, lower_bound)

    if not open_upper_bound and upper_bound > values[-1] + 1e-9:
        percentiles = np.append(percentiles, 1.0)
        values = np.append(values, upper_bound)

    return percentiles, values


def _evaluate_pchip_on_grid(
    percentiles: np.ndarray,
    values: np.ndarray,
    *,
    open_lower_bound: bool,
    open_upper_bound: bool,
    lower_bound: float,
    upper_bound: float,
    zero_point: float | None,
    num_points: int,
) -> np.ndarray:
    """Interpolate the declaration onto the submission grid and pin closed bounds."""
    # Log scaling is appropriate when all values are positive and the lower bound is too.
    use_log = np.all(values > 0) and zero_point is None and lower_bound > 0
    x_vals = np.log(values) if use_log else values

    try:
        spline = PchipInterpolator(x_vals, percentiles, extrapolate=True)
    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except  # intentional: logged fallback to linear interp
        logger.warning("PchipInterpolator failed, falling back to linear interpolation", exc_info=True)

        def spline(x):
            return np.interp(x, x_vals, percentiles)

    # Generate the grid (linear, or geometric when zero_point is set) and evaluate,
    # clamping the evaluation points to avoid extrapolation issues.
    cdf_x = build_cdf_value_grid(lower_bound, upper_bound, zero_point, num_points)
    eval_x = np.log(cdf_x) if use_log else cdf_x
    eval_x_clamped = np.clip(eval_x, x_vals[0], x_vals[-1])

    cdf_y = spline(eval_x_clamped).clip(0.0, 1.0)
    cdf_y = np.maximum.accumulate(cdf_y)

    if not open_lower_bound:
        cdf_y[0] = 0.0
    if not open_upper_bound:
        cdf_y[-1] = 1.0

    return cdf_y


def _blend_with_uniform(cdf_y: np.ndarray, min_step: float, num_points: int) -> np.ndarray:
    """Mix in just enough of a uniform CDF to pre-satisfy the min-step constraint.

    This is the mechanism that keeps every downstream repair tier off real forecasts
    (same approach as ``discrete_snap``); the tiers below it are pathological-input
    guards, not routine passes.
    """
    total_range = float(cdf_y[-1] - cdf_y[0])
    min_alpha = min_step * num_points / total_range * _ALPHA_SAFETY_MARGIN if total_range > 1e-12 else 1.0
    alpha = min(1.0, min_alpha)
    uniform_cdf = np.linspace(float(cdf_y[0]), float(cdf_y[-1]), num_points)
    return (1.0 - alpha) * cdf_y + alpha * uniform_cdf


def _ramp_tail_to_one(cdf_y: np.ndarray, overflow_idx: int) -> None:
    """Replace the saturated tail with a uniform ramp up to 1.0, in place."""
    steps_remaining = len(cdf_y) - overflow_idx
    for i in range(overflow_idx, len(cdf_y)):
        t = (i - overflow_idx) / max(1, steps_remaining - 1)
        cdf_y[i] = min(1.0, cdf_y[overflow_idx - 1] + (1.0 - cdf_y[overflow_idx - 1]) * t)


def _pull_earlier_points_down(cdf_y: np.ndarray, from_idx: int, floor_idx: int, min_step: float) -> None:
    """Walk backwards from ``from_idx`` making room for ``min_step`` above each point."""
    for j in range(from_idx - 1, floor_idx - 1, -1):
        max_allowed = cdf_y[j + 1] - min_step
        if cdf_y[j] > max_allowed:
            cdf_y[j] = max_allowed


def _reenforce_min_steps_from(cdf_y: np.ndarray, start_idx: int, min_step: float) -> None:
    """Lift each point after ``start_idx`` to ``prev + min_step``, back-filling on overflow."""
    for i in range(start_idx + 1, len(cdf_y)):
        if cdf_y[i] >= cdf_y[i - 1] + min_step:
            continue
        cdf_y[i] = cdf_y[i - 1] + min_step
        if cdf_y[i] <= 1.0:
            continue
        cdf_y[i] = 1.0
        _pull_earlier_points_down(cdf_y, i, start_idx, min_step)


def _redistribute_saturated_tail(cdf_y: np.ndarray, min_step: float) -> None:
    """Legacy panchul redistribution for a CDF that overshot 1.0 before the grid end.

    Kept as its own pass because the ramp shape is specific to the PCHIP path.
    ``enforce_min_steps``' ``upper_cap=1.0`` already caps the forward sweep, so on the
    current pipeline this is a no-op guard rather than a live repair tier.
    """
    if cdf_y[-1] <= 1.0:
        return

    overflow_indices = np.where(cdf_y > 1.0)[0]
    if len(overflow_indices) == 0:
        return

    overflow_idx = int(overflow_indices[0])
    _ramp_tail_to_one(cdf_y, overflow_idx)
    _reenforce_min_steps_from(cdf_y, overflow_idx, min_step)


def _rebuild_with_min_steps(
    cdf_y: np.ndarray,
    min_step: float,
    *,
    open_lower_bound: bool,
    open_upper_bound: bool,
    question_id: int | str | None,
    question_url: str | None,
) -> np.ndarray:
    """Last repair tier: rebuild the CDF from its own step SHAPE with min-step floors.

    Fires only when the min-step is still violated after ``safe_cdf_bounds``. AGENTS.md
    records zero fires across 1182 archived numeric forecasts — on the production
    201-point grid ``_blend_with_uniform`` gets there first. It stays reachable on a
    coarse grid whose available range is exactly saturated by the min-step.

    Raises:
        ValueError: the CDF range cannot hold one ``min_step`` per bin.
        RuntimeError: the rebuild itself failed to satisfy the min-step.
    """
    steps = np.diff(cdf_y)
    violated_steps = np.sum(steps < min_step)
    logger.warning(
        "PCHIP minimum step enforcement required for Q %s | URL %s | violated_steps=%d/%d (%.1f%%) | min_step_found=%.8f | min_step_required=%.8f | available_range=%.6f | required_range=%.6f",
        question_id or "N/A",
        question_url or "N/A",
        violated_steps,
        len(steps),
        100.0 * violated_steps / len(steps),
        np.min(steps),
        min_step,
        cdf_y[-1] - cdf_y[0],
        (len(cdf_y) - 1) * min_step,
    )

    # Create a strictly monotonic sequence over the legal endpoint range.
    start_val = cdf_y[0] if open_lower_bound else 0.0
    end_val = min(cdf_y[-1], 1.0) if open_upper_bound else 1.0

    available_range = end_val - start_val
    required_range = (len(cdf_y) - 1) * min_step

    if required_range > available_range:
        raise ValueError(
            f"Cannot satisfy minimum step requirement: need {required_range:.6f} "
            f"but only have {available_range:.6f} available in CDF range"
        )

    new_cdf = np.zeros_like(cdf_y)
    new_cdf[0] = start_val

    if len(cdf_y) > 2:
        # Keep the original shape but floor every step, then hand out what is left over.
        orig_shape = np.diff(cdf_y)
        orig_shape = np.maximum(orig_shape, min_step)
        orig_shape = orig_shape / np.sum(orig_shape)

        extra_steps = (available_range - required_range) * orig_shape
        for i in range(1, len(new_cdf)):
            new_cdf[i] = new_cdf[i - 1] + min_step + extra_steps[i - 1]
    else:
        # Simple linear spacing if original shape is unavailable
        for i in range(1, len(new_cdf)):
            new_cdf[i] = new_cdf[i - 1] + (available_range / (len(new_cdf) - 1))

    if np.any(np.diff(new_cdf) < min_step - 1e-10):
        raise RuntimeError("Internal error: Step size enforcement failed")

    new_steps = np.diff(new_cdf)
    logger.info(
        "PCHIP aggressive enforcement completed for Q %s | URL %s | new_min_step=%.8f | new_max_step=%.8f | total_range_redistributed=%.6f | shape_preserved=True",
        question_id or "N/A",
        question_url or "N/A",
        np.min(new_steps),
        np.max(new_steps),
        available_range,
    )
    return new_cdf


def _assert_pchip_constraints(
    cdf_y: np.ndarray,
    min_step: float,
    *,
    open_lower_bound: bool,
    open_upper_bound: bool,
) -> None:
    """Fail loudly rather than submit a CDF the Metaculus validators would reject."""
    if np.any(np.diff(cdf_y) < min_step - 1e-10):
        problematic_indices = np.where(np.diff(cdf_y) < min_step - 1e-10)[0]
        raise RuntimeError(
            f"Failed to enforce minimum step size at indices: {problematic_indices}, "
            f"values: {np.diff(cdf_y)[problematic_indices]}"
        )

    if not open_lower_bound and abs(cdf_y[0]) > 1e-10:
        raise RuntimeError(f"Failed to enforce lower bound: {cdf_y[0]}")

    if not open_upper_bound and abs(cdf_y[-1] - 1.0) > 1e-10:
        raise RuntimeError(f"Failed to enforce upper bound: {cdf_y[-1]}")


def generate_pchip_cdf(
    percentile_values: dict[int | float, float],
    *,
    open_upper_bound: bool,
    open_lower_bound: bool,
    upper_bound: float,
    lower_bound: float,
    zero_point: float | None = None,
    min_step: float = 5.0e-5,
    max_step: float = NUM_MAX_STEP,
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

    ``min_step`` / ``max_step`` default to the 201-point-grid constants. A caller building a CDF
    on a non-201 grid (a discrete question with ``num_points != 201``) must pass the grid-scaled
    values (see ``numeric.config.grid_step_constraints``) so the per-bin constraints match the
    server's ``round(0.01 / N, 9)`` min-step and ``0.2 * 200 / N`` max-step, where
    ``N = num_points - 1``. Passing the 201-grid ``max_step`` (0.2) on a coarse discrete grid
    wrongly clips each bin to 20% and shoves the excess onto higher bins.

    Raises:
        ValueError: If input validation fails
        RuntimeError: If constraint enforcement fails
    """
    _validate_pchip_bounds(percentile_values, lower_bound, upper_bound, zero_point)
    cleaned = _nudge_duplicate_values_apart(_clean_percentile_values(percentile_values))

    percentiles, values = _percentile_arrays_with_boundaries(
        cleaned,
        open_lower_bound=open_lower_bound,
        open_upper_bound=open_upper_bound,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    cdf_y = _evaluate_pchip_on_grid(
        percentiles,
        values,
        open_lower_bound=open_lower_bound,
        open_upper_bound=open_upper_bound,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        zero_point=zero_point,
        num_points=num_points,
    )

    # Repair ladder, cheapest first: uniform pre-mix, min-step sweep, saturated-tail
    # ramp, then boundary + max-step enforcement.
    cdf_y = _blend_with_uniform(cdf_y, min_step, num_points)
    cdf_y = enforce_min_steps(cdf_y, min_step, upper_cap=1.0, lower_cap=0.0)
    _redistribute_saturated_tail(cdf_y, min_step)
    cdf_y = safe_cdf_bounds(cdf_y, open_lower_bound, open_upper_bound, min_step=min_step, max_step=max_step)

    aggressive_enforcement_used = bool(np.any(np.diff(cdf_y) < min_step))
    if aggressive_enforcement_used:
        cdf_y = _rebuild_with_min_steps(
            cdf_y,
            min_step,
            open_lower_bound=open_lower_bound,
            open_upper_bound=open_upper_bound,
            question_id=question_id,
            question_url=question_url,
        )

    _assert_pchip_constraints(
        cdf_y,
        min_step,
        open_lower_bound=open_lower_bound,
        open_upper_bound=open_upper_bound,
    )

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
