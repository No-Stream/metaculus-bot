"""Discrete integer CDF snapping for continuous questions with integer outcomes.

When Metaculus labels a question as "continuous" (cdf_size=201) but the
resolution values are naturally integers (e.g. "How many X will happen?"),
the smooth CDF wastes probability mass between integers. Snapping the CDF
to a step function that concentrates mass on integer values improves scoring.

Server-side constraints (from Metaculus backend):
  - Min step between adjacent CDF values: 0.01 / N (default 5e-5 for N=200)
  - Max step: 0.2 * 200 / N (default 0.2)
  - CDF must be strictly increasing (no flat segments)

The snapping algorithm:
  1. Extract integer PMF from smooth CDF via half-integer interpolation
  2. Reconstruct step-function CDF with steps at integer positions
  3. Apply uniform mixture for min-step compliance (primary min-step mechanism)
  4. Apply max-step redistribution + boundary pinning via _safe_cdf_bounds()
     (note: _safe_cdf_bounds handles max-step and boundaries only, not min-step)
  5. Post-hoc min-step verification (guard for concentrated distributions)
"""

import logging
import math

import numpy as np
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import NumericQuestion
from pydantic import BaseModel

from metaculus_bot.constants import DISCRETE_SNAP_MAX_INTEGERS, DISCRETE_SNAP_UNIFORM_MIX, NUM_MIN_PROB_STEP
from metaculus_bot.pchip_cdf import _safe_cdf_bounds
from metaculus_bot.pchip_processing import create_pchip_numeric_distribution

logger = logging.getLogger(__name__)

# Algorithm-internal constants
_NEAR_ZERO_EPSILON: float = 1e-12
_ALPHA_SAFETY_MARGIN: float = 1.1


class OutcomeTypeResult(BaseModel):
    """Structured output for discrete/continuous classification."""

    is_discrete_integer: bool


def majority_votes_discrete(votes: list[bool]) -> bool:
    """Return True if a strict majority of votes are True (DISCRETE)."""
    if not votes:
        return False
    return sum(votes) > len(votes) / 2


def snap_cdf_to_integers(
    cdf_values: list[float],
    lower_bound: float,
    upper_bound: float,
    open_lower_bound: bool,
    open_upper_bound: bool,
) -> list[float] | None:
    """Convert a smooth 201-point CDF to a step function at integer boundaries.

    Returns snapped CDF values (list of 201 floats), or None if snapping
    should be skipped (too many integers, etc.).
    """
    n_points = len(cdf_values)
    x_grid = np.linspace(lower_bound, upper_bound, n_points)
    p_smooth = np.array(cdf_values, dtype=float)

    integers = np.arange(math.ceil(lower_bound), math.floor(upper_bound) + 1)
    if len(integers) > DISCRETE_SNAP_MAX_INTEGERS:
        logger.info(
            "Discrete snap skipped: %d integers > max %d | bounds=[%.1f, %.1f]",
            len(integers),
            DISCRETE_SNAP_MAX_INTEGERS,
            lower_bound,
            upper_bound,
        )
        return None

    if len(integers) == 0:
        logger.warning("Discrete snap skipped: no integers in bounds [%.4f, %.4f]", lower_bound, upper_bound)
        return None

    # --- Step 1: Extract integer PMF via half-integer interpolation ---
    pmf = np.zeros(len(integers), dtype=float)
    for i, k in enumerate(integers):
        upper_half = min(k + 0.5, upper_bound)
        lower_half = max(k - 0.5, lower_bound)
        cdf_upper = float(np.interp(upper_half, x_grid, p_smooth))
        cdf_lower = float(np.interp(lower_half, x_grid, p_smooth))
        pmf[i] = max(0.0, cdf_upper - cdf_lower)

    # Normalize PMF to match the total mass in the smooth CDF
    total_smooth = float(p_smooth[-1] - p_smooth[0])
    total_pmf = float(pmf.sum())
    if total_pmf > _NEAR_ZERO_EPSILON:
        pmf *= total_smooth / total_pmf

    # --- Step 2: Reconstruct step CDF at grid points ---
    cumulative_pmf = np.cumsum(pmf)
    tail_lower = float(p_smooth[0])

    step_positions = integers.astype(float)
    step_cdf = np.full(n_points, tail_lower, dtype=float)
    indices = np.searchsorted(step_positions, x_grid, side="right")
    mask = indices > 0
    step_cdf[mask] = tail_lower + cumulative_pmf[indices[mask] - 1]

    # Pin cdf[0] for closed lower bound (step at k=lower_bound makes searchsorted
    # assign mass to bucket 0, violating cdf[0]=0; pin pushes mass into bucket 1)
    if not open_lower_bound:
        step_cdf[0] = tail_lower

    # Pin final endpoint to preserve upper tail mass
    step_cdf[-1] = float(p_smooth[-1])

    # --- Step 3: Uniform mixture for min-step compliance ---
    if total_smooth > _NEAR_ZERO_EPSILON:
        min_alpha_for_step = NUM_MIN_PROB_STEP * n_points / total_smooth * _ALPHA_SAFETY_MARGIN
    else:
        min_alpha_for_step = 1.0
    alpha = min(1.0, max(DISCRETE_SNAP_UNIFORM_MIX, min_alpha_for_step))
    uniform_cdf = np.linspace(p_smooth[0], p_smooth[-1], n_points)
    mixed_cdf = (1.0 - alpha) * step_cdf + alpha * uniform_cdf

    # --- Step 4: Max-step redistribution + boundary pinning ---
    # _safe_cdf_bounds handles max-step and boundary constraints; min-step relies on the uniform mixture above
    enforced_cdf = _safe_cdf_bounds(mixed_cdf, open_lower_bound, open_upper_bound, NUM_MIN_PROB_STEP)

    result = enforced_cdf.tolist()

    # Guard: uniform mixture may not satisfy min-step for very concentrated distributions
    diffs = np.diff(enforced_cdf)
    min_diff = float(np.min(diffs))
    max_diff = float(np.max(diffs))
    if min_diff < NUM_MIN_PROB_STEP - 1e-10:
        logger.error(
            "Discrete snap min-step violation after enforcement: %.8f < %.8f",
            min_diff,
            NUM_MIN_PROB_STEP,
        )
        return None

    logger.info(
        "Discrete snap applied | integers=%d | min_step=%.6f | max_step=%.6f | alpha=%.4f",
        len(integers),
        min_diff,
        max_diff,
        alpha,
    )
    return result


def snap_distribution_to_integers(
    distribution: NumericDistribution,
    question: NumericQuestion,
) -> NumericDistribution | None:
    """Snap a NumericDistribution's CDF to integer boundaries.

    Returns a new distribution with the snapped CDF, or None if snapping
    should be skipped.
    """
    if not (np.isfinite(question.lower_bound) and np.isfinite(question.upper_bound)):
        logger.warning(
            "Discrete snap skipped: non-finite bounds lower=%s upper=%s", question.lower_bound, question.upper_bound
        )
        return None

    if question.cdf_size is not None and question.cdf_size != 201:
        logger.info("Discrete snap skipped: question already labeled discrete (cdf_size=%d)", question.cdf_size)
        return None

    # All NumericDistributions in our pipeline are PchipNumericDistribution (pchip_processing.py)
    cdf_probs: list[float] = list(distribution._pchip_cdf_values)  # type: ignore[attr-defined]

    snapped_cdf = snap_cdf_to_integers(
        cdf_values=cdf_probs,
        lower_bound=question.lower_bound,
        upper_bound=question.upper_bound,
        open_lower_bound=question.open_lower_bound,
        open_upper_bound=question.open_upper_bound,
    )

    if snapped_cdf is None:
        return None

    return create_pchip_numeric_distribution(
        pchip_cdf=snapped_cdf,
        percentile_list=distribution.declared_percentiles,
        question=question,
        zero_point=question.zero_point,
    )
