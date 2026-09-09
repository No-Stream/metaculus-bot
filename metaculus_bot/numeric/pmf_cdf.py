"""A per-bin declaration to the CDF the platform accepts, on an enumerable grid.

On a grid ``config.elicit_per_bin`` admits, a forecaster declares one probability per bin plus the
reserved ``below_range`` / ``above_range`` masses where a bound is open: the platform's own PMF
shape, ``[below, p_0, ..., p_{N-1}, above]`` with ``N + 2`` entries and a closed tail's entry at
0.0. There are no percentiles to sanitise, interpolate or guard against unit errors, so the whole
repair is one step: the smallest blend toward the server's exact per-cell floors that makes every
cell legal (``_blend_to_cell_floors``; the arithmetic and the receipts are in
``docs/numeric_pipeline.md``, "Per-bin elicitation on enumerable grids"). A bin the model set to 0
lands at exactly the platform minimum, a certain member keeps 0.988 to 0.992 on its bin, and a legal
declaration comes back unchanged. Mass declared in a CLOSED tail is refused rather than pinned over,
because a pin after the cumulative sum silently moves it into the first in-range bin.

The assembled CDF runs through ``safe_cdf_bounds`` with the grid's own step limits (the one
implementation of the open-bound pins and the max-step packing every published CDF reaches), has
its closed bounds re-pinned, and is checked by :func:`validate_grid_cdf`, a fail-SHUT replica of the
server's rules, before it is wrapped in the same ``create_pchip_numeric_distribution`` object the
discrete percentile path produces, so nothing downstream sees a new shape.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric.config import OPEN_TAIL_MIN_MASS, PMF_FLOOR_MARGIN, grid_step_constraints
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid, safe_cdf_bounds
from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution
from metaculus_bot.numeric.validation import resolve_zero_point

__all__ = ["build_pmf_distribution", "published_pmf", "server_rounded_pmf", "validate_grid_cdf"]

# The server compares the CDF rounded to 10 decimals and its PMF rounded to 9.
_SERVER_CDF_DECIMALS: int = 10
_SERVER_PMF_DECIMALS: int = 9


def server_rounded_pmf(cdf: Sequence[float] | np.ndarray) -> np.ndarray:
    """The in-range steps of ``cdf`` as the server compares them: the CDF rounded to 10 decimals, differenced, rounded to 9."""
    return np.round(np.diff(np.round(np.asarray(cdf, dtype=float), _SERVER_CDF_DECIMALS)), _SERVER_PMF_DECIMALS)


def build_pmf_distribution(
    declared: Sequence[float], view: NumericQuestion, *, model_name: str = ""
) -> NumericDistribution:
    """The distribution to publish for a per-bin declaration on ``view``'s grid.

    ``declared`` is the platform's ``N + 2`` PMF shape (``N = view.cdf_size - 1`` bins):
    ``[below, p_0, ..., p_{N-1}, above]``, a closed tail's entry 0.0, every entry finite and
    non-negative, the sum positive. The extraction ladder bounds the declared sum to
    ``structured_output_schema.pmf_prob_sum_tolerance(len(grid.keys))`` of 1.0, a 0.02 floor plus
    0.005 per key (0.06 on post 651's 12-key grid, 0.165 at the 33-key maximum); it is normalised
    here, so only the shape is load-bearing. ``view`` is the numeric-pipeline view of the question
    (the epoch adapter for a date question); ``model_name`` only labels the ``CDF_MAXSTEP_CLIP``
    marker ``safe_cdf_bounds`` emits if the grid's cap ever binds.

    Raises:
        ValueError: the declaration is malformed, or carries mass in a closed tail.
        RuntimeError: the built CDF fails a server rule (``validate_grid_cdf``); a guard fails shut.
    """
    open_lower, open_upper = view.open_lower_bound, view.open_upper_bound
    pmf = _validated_declaration(declared, cdf_size=view.cdf_size, open_lower=open_lower, open_upper=open_upper)
    pmf = _blend_to_cell_floors(pmf / pmf.sum(), cdf_size=view.cdf_size, open_lower=open_lower, open_upper=open_upper)

    cdf = np.cumsum(pmf[:-1])
    _pin_closed_bounds(cdf, open_lower=open_lower, open_upper=open_upper)
    min_step, max_step = grid_step_constraints(view.cdf_size)
    cdf = safe_cdf_bounds(
        cdf,
        open_lower,
        open_upper,
        min_step=min_step,
        max_step=max_step,
        question_id=view.id_of_question,
        model_name=model_name,
    )
    _pin_closed_bounds(cdf, open_lower=open_lower, open_upper=open_upper)
    validate_grid_cdf(cdf, cdf_size=view.cdf_size, open_lower=open_lower, open_upper=open_upper)

    zero_point = resolve_zero_point(view)
    edges = build_cdf_value_grid(view.lower_bound, view.upper_bound, zero_point, view.cdf_size)
    declared_percentiles = [
        Percentile(percentile=float(height), value=float(edge)) for edge, height in zip(edges, cdf, strict=True)
    ]
    return create_pchip_numeric_distribution(
        pchip_cdf=[float(height) for height in cdf],
        percentile_list=declared_percentiles,
        question=view,
        zero_point=zero_point,
    )


def published_pmf(prediction: NumericDistribution) -> list[float]:
    """The ``N + 2`` PMF a built distribution publishes: ``[cdf[0], diff(cdf)..., 1 - cdf[-1]]``.

    The inverse of the assembly in :func:`build_pmf_distribution`, in the shape the platform itself
    exposes for every stored forecast and the ``MEMBER_FORECAST`` marker records as ``published``.
    """
    heights = np.fromiter((point.percentile for point in prediction.get_cdf()), dtype=float)
    return [float(heights[0]), *(float(step) for step in np.diff(heights)), 1.0 - float(heights[-1])]


def validate_grid_cdf(cdf: Sequence[float] | np.ndarray, *, cdf_size: int, open_lower: bool, open_upper: bool) -> None:
    """Raise ``RuntimeError`` where the platform's ``continuous_cdf`` validator would reject ``cdf``.

    The production twin of the test helper ``assert_server_accepts_cdf``: the CDF is rounded to 10
    decimals and its PMF to 9, every rounded step must lie in ``grid_step_constraints(cdf_size)``
    (the 9-decimal min step is the server's own; the 9-decimal-floored max step admits exactly the
    9-decimal values the server's unrounded cap admits), a closed bound must be exactly 0.0 / 1.0
    and an open one at least 0.001 inside. Nothing is repaired here: this is the last thing before
    a per-bin distribution is handed downstream, and a guard fails shut.
    """
    heights = np.asarray(cdf, dtype=float)
    if heights.size != cdf_size:
        raise RuntimeError(f"CDF length {heights.size} != cdf_size {cdf_size} (inbound bins + 1)")
    if np.any(np.isnan(heights)):
        raise RuntimeError(f"CDF carries NaN at {np.flatnonzero(np.isnan(heights)).tolist()}")
    _check_steps(server_rounded_pmf(heights), cdf_size=cdf_size)
    rounded = np.round(heights, _SERVER_CDF_DECIMALS)
    _check_bounds(float(rounded[0]), float(rounded[-1]), open_lower=open_lower, open_upper=open_upper)


def _check_steps(pmf: np.ndarray, *, cdf_size: int) -> None:
    min_step, max_step = grid_step_constraints(cdf_size)
    if np.any(pmf < min_step):
        offender = int(np.argmin(pmf))
        raise RuntimeError(f"CDF step {pmf[offender]} at bin {offender} is under the server min step {min_step}")
    if np.any(pmf > max_step):
        offender = int(np.argmax(pmf))
        raise RuntimeError(f"CDF step {pmf[offender]} at bin {offender} is over the server max step {max_step}")


def _check_bounds(first: float, last: float, *, open_lower: bool, open_upper: bool) -> None:
    if open_lower and first < OPEN_TAIL_MIN_MASS:
        raise RuntimeError(f"open lower bound cdf[0]={first} < {OPEN_TAIL_MIN_MASS}")
    if not open_lower and first != 0.0:
        raise RuntimeError(f"closed lower bound cdf[0]={first} != 0.0")
    if open_upper and last > 1.0 - OPEN_TAIL_MIN_MASS:
        raise RuntimeError(f"open upper bound cdf[-1]={last} > {1.0 - OPEN_TAIL_MIN_MASS}")
    if not open_upper and last != 1.0:
        raise RuntimeError(f"closed upper bound cdf[-1]={last} != 1.0")


def _validated_declaration(
    declared: Sequence[float], *, cdf_size: int, open_lower: bool, open_upper: bool
) -> np.ndarray:
    """The declaration as an array, or ``ValueError`` where it is not a PMF this grid can carry."""
    pmf = np.asarray(declared, dtype=float)
    expected = cdf_size + 1
    if pmf.size != expected:
        raise ValueError(f"per-bin declaration has {pmf.size} entries; this {cdf_size - 1}-bin grid takes {expected}")
    if not np.all(np.isfinite(pmf)):
        raise ValueError(f"per-bin declaration is not finite at {np.flatnonzero(~np.isfinite(pmf)).tolist()}")
    if np.any(pmf < 0.0):
        raise ValueError(f"per-bin declaration is negative at {np.flatnonzero(pmf < 0.0).tolist()}")
    if pmf.sum() <= 0.0:
        raise ValueError("per-bin declaration must sum to a positive mass")
    if not open_lower and pmf[0] > 0.0:
        raise ValueError(f"per-bin declaration puts {pmf[0]} below a closed lower bound")
    if not open_upper and pmf[-1] > 0.0:
        raise ValueError(f"per-bin declaration puts {pmf[-1]} above a closed upper bound")
    return pmf


def _cell_floors(cdf_size: int, *, open_lower: bool, open_upper: bool) -> np.ndarray:
    """The server's minimum per cell, each non-zero floor raised by the margin; a closed tail's floor is 0."""
    min_step, _ = grid_step_constraints(cdf_size)
    floors = np.full(cdf_size + 1, min_step + PMF_FLOOR_MARGIN)
    floors[0] = OPEN_TAIL_MIN_MASS + PMF_FLOOR_MARGIN if open_lower else 0.0
    floors[-1] = OPEN_TAIL_MIN_MASS + PMF_FLOOR_MARGIN if open_upper else 0.0
    return floors


def _blend_to_cell_floors(pmf: np.ndarray, *, cdf_size: int, open_lower: bool, open_upper: bool) -> np.ndarray:
    """``(1 - alpha) pmf + alpha t`` toward the floor distribution ``t``, with the smallest legal ``alpha``.

    ``alpha = max((f_i - p_i) / (t_i - p_i))`` over the deficient cells; it exists and is at most
    ``sum(f)`` (about 0.012) because ``t_i = f_i / sum(f) > f_i > p_i`` on each of them. The same
    idea as ``pchip_cdf._blend_with_uniform``, applied to the cell floors exactly.
    """
    floors = _cell_floors(cdf_size, open_lower=open_lower, open_upper=open_upper)
    target = floors / floors.sum()
    deficient = pmf < floors
    if not np.any(deficient):
        return pmf
    alpha = float(np.max((floors[deficient] - pmf[deficient]) / (target[deficient] - pmf[deficient])))
    return (1.0 - alpha) * pmf + alpha * target


def _pin_closed_bounds(cdf: np.ndarray, *, open_lower: bool, open_upper: bool) -> None:
    """A closed bound is exactly 0.0 / 1.0 in place; the blend left its tail at 0, so only float residue moves."""
    if not open_lower:
        cdf[0] = 0.0
    if not open_upper:
        cdf[-1] = 1.0
