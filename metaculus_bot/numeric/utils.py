"""Numeric aggregation and helper utilities used by TemplateForecaster.

This module centralises logic for combining numeric forecasts and constructing
user-friendly bound messages so that the core forecaster class stays small.
"""

import logging
from typing import Literal, Sequence

import numpy as np
from forecasting_tools import PredictedOptionList
from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    Percentile,
)
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.constants import NUM_RAMP_K_FACTOR
from metaculus_bot.mc_processing import clamp_and_renormalize_probs
from metaculus_bot.numeric.config import PCHIP_CDF_POINTS, grid_step_constraints
from metaculus_bot.numeric.pchip_cdf import generate_pchip_cdf, safe_cdf_bounds
from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution

__all__ = [
    "aggregate_numeric",
    "aggregate_binary_mean",
    "bound_messages",
    "nominal_bounds",
    "clamp_and_renormalize_mc",
]


logger = logging.getLogger(__name__)


def aggregate_binary_mean(predictions: Sequence[float]) -> float:
    """Return the mean of *binary* forecasts rounded to three decimals.

    This matches the old behaviour from `TemplateForecaster`.
    """

    if not predictions:
        raise ValueError("Cannot aggregate empty list of binary predictions")

    mean_prediction = sum(predictions) / len(predictions)
    return round(mean_prediction, 3)


def _pin_endpoints(p_vals: np.ndarray, question: NumericQuestion) -> None:
    """Pin CDF endpoints in-place according to open/closed bound semantics."""
    if question.open_lower_bound:
        p_vals[0] = max(p_vals[0], 0.001)
    else:
        p_vals[0] = 0.0
    if question.open_upper_bound:
        p_vals[-1] = min(p_vals[-1], 0.999)
    else:
        p_vals[-1] = 1.0


def _postprocess_ensemble_cdf(
    x_vals: np.ndarray,
    p_vals: np.ndarray,
    question: NumericQuestion,
    method_label: str,
) -> NumericDistribution:
    """Shared CDF post-processing for both mean and median aggregation.

    Handles endpoint pinning, monotonic enforcement, ramp smoothing (continuous),
    and PCHIP resampling (discrete). ``method_label`` is used only in log messages
    (e.g. ``"mean"`` or ``"median"``).
    """
    p_vals = np.clip(p_vals, 0.0, 1.0)
    p_vals = np.maximum.accumulate(p_vals)
    _pin_endpoints(p_vals, question)

    target_cdf_size = int(getattr(question, "cdf_size", len(x_vals)) or len(x_vals))
    is_discrete = target_cdf_size != len(x_vals)
    min_step_required, max_step_required = grid_step_constraints(target_cdf_size)

    if not is_discrete:
        diffs_before = np.diff(p_vals)
        min_delta_before = float(np.min(diffs_before)) if len(diffs_before) else 1.0
        if min_delta_before < min_step_required:
            ramp = np.linspace(0.0, min_step_required * NUM_RAMP_K_FACTOR, len(p_vals))
            p_vals = np.maximum.accumulate(p_vals + ramp)
            _pin_endpoints(p_vals, question)

            diffs_after = np.diff(p_vals)
            min_delta_after = float(np.min(diffs_after)) if len(diffs_after) else 1.0
            logger.warning(
                "Ensemble CDF ramp smoothing (%s) | Q %s | URL %s | min_prob_delta_before=%.8f | min_prob_delta_after=%.8f",
                method_label,
                getattr(question, "id_of_question", None),
                getattr(question, "page_url", None),
                min_delta_before,
                min_delta_after,
            )

        # The ramp (and even a raw concentrated median) can push interior CDF values above
        # 1.0 and can leave bins above the grid-scaled max-step (binding once cdf_size >= 42).
        # Route the aggregated CDF through the same bounds/monotonic/min-step/max-step
        # enforcement the per-model ramp path (pchip_processing._apply_ramp_smoothing) and the
        # discrete branch below already apply, so the result is a valid submission rather than
        # a downstream Percentile validation crash (percentile <= 1) that drops the question.
        p_vals = safe_cdf_bounds(
            p_vals,
            open_lower=question.open_lower_bound,
            open_upper=question.open_upper_bound,
            min_step=min_step_required,
            max_step=max_step_required,
        )

        declared_percentiles = [Percentile(percentile=float(p), value=float(v)) for v, p in zip(x_vals, p_vals)]
        return create_pchip_numeric_distribution(
            pchip_cdf=list(map(float, p_vals)),
            percentile_list=declared_percentiles,
            question=question,
            zero_point=question.zero_point,
        )

    # Discrete path: resample to question.cdf_size and strictly enforce min-step
    logger.info(
        "Discrete aggregation detected (%s) | Q %s | URL %s | target_cdf_size=%d | min_step_required=%.8f",
        method_label,
        getattr(question, "id_of_question", None),
        getattr(question, "page_url", None),
        target_cdf_size,
        min_step_required,
    )

    percentile_values = {float(prob * 100.0): float(val) for val, prob in zip(x_vals, p_vals)}
    pchip_cdf_values, _ = generate_pchip_cdf(
        percentile_values=percentile_values,
        open_upper_bound=question.open_upper_bound,
        open_lower_bound=question.open_lower_bound,
        upper_bound=question.upper_bound,
        lower_bound=question.lower_bound,
        zero_point=question.zero_point,
        min_step=min_step_required,
        max_step=max_step_required,
        num_points=target_cdf_size,
        question_id=getattr(question, "id_of_question", None),
        question_url=getattr(question, "page_url", None),
    )

    x_disc = np.linspace(question.lower_bound, question.upper_bound, target_cdf_size)
    declared_percentiles = [Percentile(percentile=float(p), value=float(v)) for v, p in zip(x_disc, pchip_cdf_values)]
    return create_pchip_numeric_distribution(
        pchip_cdf=list(map(float, pchip_cdf_values)),
        percentile_list=declared_percentiles,
        question=question,
        zero_point=question.zero_point,
    )


def _canonical_cdf_length(question: NumericQuestion) -> int:
    """Points the question's submitted CDF has: ``cdf_size``, else the 201-point default.

    Mirrors ``build_numeric_distribution``'s target (pipeline.py reads a None
    ``cdf_size`` as the standard grid the same way), so per-model CDFs and the
    ensemble CDF live on the same grid by construction. An out-of-range value
    raises LOUDLY rather than silently substituting 201: the substitution would
    only defer the crash to ``_postprocess_ensemble_cdf``, which re-reads
    ``question.cdf_size`` and would then disagree with the grid the models were
    just aligned onto.
    """
    if question.cdf_size is None:
        return PCHIP_CDF_POINTS
    target = int(question.cdf_size)
    if target < 2:
        raise ValueError(f"NumericQuestion.cdf_size must be >= 2 to define a CDF grid, got {target}")
    return target


def _cdf_heights_on_canonical_grid(
    prediction: NumericDistribution,
    n_points: int,
    question: NumericQuestion,
    model_index: int,
) -> np.ndarray:
    """One model's CDF heights, resampled to ``n_points`` if it arrived on another grid.

    Grid index ``i`` means the same thing in every CDF — Metaculus bucket
    ``i / (n - 1)`` of the question's range — so a length mismatch is resolved by
    interpolating in that shared cdf-location space, NOT in value space: a
    log-scaled (``zero_point``) question's PCHIP CDF carries a linear value axis
    while forecasting-tools' fallback builder carries a geometric one, so the
    x-values disagree by construction even when bucket ``i`` matches.
    """
    heights = np.asarray([float(p.percentile) for p in prediction.get_cdf()], dtype=float)
    if heights.size < 2:
        raise ValueError(f"Model {model_index} contributed a {heights.size}-point CDF; cannot aggregate")
    if heights.size == n_points:
        return heights
    logger.warning(
        "NUMERIC_AGGREGATE_GRID_MISMATCH: question=%s model_index=%d got_points=%d expected_points=%d — "
        "resampling in cdf-location space before aggregation",
        getattr(question, "id_of_question", None),
        model_index,
        heights.size,
        n_points,
    )
    return np.interp(
        np.linspace(0.0, 1.0, n_points),
        np.linspace(0.0, 1.0, heights.size),
        heights,
    )


def aggregate_numeric(
    predictions: Sequence[NumericDistribution],
    question: NumericQuestion,
    method: str | Literal["mean", "median"] = "mean",
) -> NumericDistribution:
    """Aggregate numeric distributions by mean or median, pointwise in CDF space.

    Every model contributes to every grid point. The aggregation is POSITIONAL
    (grid index i across all models), because grouping on the float ``value``
    axis silently medianed over a SUBSET: the PCHIP path's ``np.linspace`` grid
    and forecasting-tools' fallback ``min + span*i/(n-1)`` grid are equal in
    exact arithmetic but differ in the last bits, so a mixed-path ensemble
    produced ~225 distinct x-values for 201 buckets and roughly a quarter of them
    had fewer than n contributors — with nothing recording the partial
    membership, and the resulting length mismatch logging a spurious "Discrete
    aggregation detected". A model that genuinely arrives on a different-length
    grid is resampled first (logged, see ``_cdf_heights_on_canonical_grid``).

    Parameters
    ----------
    predictions
        List of `NumericDistribution` objects as produced by individual LLMs.
    question
        The original `NumericQuestion` – needed for bounds metadata.
    method
        "mean" (default) or "median" to pick aggregation strategy.
    """

    if not predictions:
        raise ValueError("Cannot aggregate empty list of numeric predictions")

    if method not in ("mean", "median"):
        raise ValueError(f"Invalid aggregation method: {method}")

    n_points = _canonical_cdf_length(question)
    heights = np.vstack(
        [
            _cdf_heights_on_canonical_grid(prediction, n_points, question, index)
            for index, prediction in enumerate(predictions)
        ]
    )
    if heights.shape != (len(predictions), n_points):
        raise ValueError(f"Aligned CDF matrix has shape {heights.shape}, expected {(len(predictions), n_points)}")

    p_vals = heights.mean(axis=0) if method == "mean" else np.median(heights, axis=0)
    x_vals = np.linspace(question.lower_bound, question.upper_bound, n_points)

    return _postprocess_ensemble_cdf(x_vals, p_vals, question, method_label=method)


def nominal_bounds(question: NumericQuestion) -> tuple[float, float]:
    """Return (upper, lower) nominal/displayed bounds; derive from half-step for discrete Qs.

    The half-step branch requires ``cdf_size > 1`` — the API derives cdf_size as
    ``inbound_outcome_count + 1`` so real questions are always >= 2, but a degenerate
    stub would otherwise divide by zero.
    """
    nominal_upper = getattr(question, "nominal_upper_bound", None)
    nominal_lower = getattr(question, "nominal_lower_bound", None)
    cdf_size = getattr(question, "cdf_size", None)
    if (
        nominal_upper is None
        and nominal_lower is None
        and cdf_size is not None
        and cdf_size > 1
        and cdf_size != PCHIP_CDF_POINTS
    ):
        step = (question.upper_bound - question.lower_bound) / (cdf_size - 1)
        nominal_upper = question.upper_bound - step / 2
        nominal_lower = question.lower_bound + step / 2
    upper = nominal_upper if nominal_upper is not None else question.upper_bound
    lower = nominal_lower if nominal_lower is not None else question.lower_bound
    return upper, lower


def bound_messages(question: NumericQuestion) -> tuple[str, str]:
    """Return upper & lower bound helper messages for numeric prompts.

    For discrete questions, if nominal bounds are missing, derive them using half-step logic.
    """

    upper_bound_number, lower_bound_number = nominal_bounds(question)

    if question.open_upper_bound:
        upper_bound_message = (
            f"The upper bound is open: {upper_bound_number} is the top of the displayed range, not a hard limit, "
            f"so the outcome can resolve above {upper_bound_number}. Your percentiles are the ONLY way you express "
            f"probability mass, including mass beyond the displayed range. To put N% of your probability above the "
            f"open ceiling, place that fraction of your percentiles above it: if you believe there is a ~75% chance "
            f"the outcome exceeds {upper_bound_number}, then your P50 (median) must be ABOVE {upper_bound_number} and "
            f"only your lower percentiles (P1, P2.5, P5, P10, P20) sit inside or below the range. Put percentiles at "
            f"or above {upper_bound_number} where you actually believe the value lies, even far outside the displayed "
            f"range. Do not pile percentiles at the boundary."
        )
    else:
        upper_bound_message = f"The upper bound is closed: the outcome can not be higher than {upper_bound_number}."

    if question.open_lower_bound:
        lower_bound_message = (
            f"The lower bound is open: {lower_bound_number} is the bottom of the displayed range, not a hard limit, "
            f"so the outcome can resolve below {lower_bound_number}. Your percentiles are the ONLY way you express "
            f"probability mass, including mass beyond the displayed range. To put N% of your probability below the "
            f"open floor, place that fraction of your percentiles below it: if you believe there is a ~75% chance the "
            f"outcome is below {lower_bound_number}, then your P50 (median) must be BELOW {lower_bound_number} and "
            f"only your upper percentiles (P80, P90, P95, P97.5, P99) sit inside or above the range. Put percentiles "
            f"at or below {lower_bound_number} where you actually believe the value lies, even far outside the "
            f"displayed range. Do not pile percentiles at the boundary."
        )
    else:
        lower_bound_message = f"The lower bound is closed: the outcome can not be lower than {lower_bound_number}."
    return upper_bound_message, lower_bound_message


def clamp_and_renormalize_mc(
    predicted_option_list: PredictedOptionList,
) -> PredictedOptionList:
    """Clamp MC option probabilities into [MC_PROB_MIN, MC_PROB_MAX] and renormalize in-place.

    Delegates the math to ``clamp_and_renormalize_probs`` so the drift-free clamp is a
    single source shared with every pre-construction clamp site. Renormalization can no
    longer push a floored option back below the floor (the ``0.984 + 8x0.002`` case), so
    every option lands in [MC_PROB_MIN, MC_PROB_MAX] with the list summing to 1.0 —
    **provided ``n * MC_PROB_MIN < 1.0``** (n <= 100 at the 0.01 floor). Above that no
    in-bounds sum-1 solution exists and the delegate returns sub-floor values by necessity;
    see ``clamp_and_renormalize_probs`` for the full contract and the ft-validator
    consequences. Returns the same `PredictedOptionList` for convenience.
    """
    clamped = clamp_and_renormalize_probs([option.probability for option in predicted_option_list.predicted_options])
    for option, probability in zip(predicted_option_list.predicted_options, clamped):
        option.probability = probability

    return predicted_option_list
