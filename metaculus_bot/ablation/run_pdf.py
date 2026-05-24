"""Deterministic structured-math aggregation runner for the ARM_PDF ablation arm.

Replaces the LLM stacker with pure probability math. For each surviving
forecaster, extracts the structured JSON block (via
``structured_output_schema.parse_structured_block``), computes a prediction
from the declared math, and aggregates via pointwise median.

Per-forecaster prediction cascade:

**Binary** — take the first applicable:
  1. Hazard (time-to-event): constant-hazard or Gamma-conjugate survival.
  2. Bayes (reference class + evidence): Beta-binomial posterior + LR updates.
  3. Prior blend (prior.prob + evidence): log-odds shift from evidence items.
  4. No fallback to declared ``posterior_prob``. Drop if none apply.

**Numeric** — single path:
  1. If ``mixture_components`` set: constraint-enforced 201-point CDF via mixture.
  2. Else if ``declared_percentiles`` set: fit parametric family, build CDF on grid.
  3. No fallback. Drop if neither parseable.

**Multiple choice** — single path:
  1. If ``option_probs`` well-formed (sums ~1): use directly.
  2. No fallback. Drop if malformed.

Aggregation: pointwise median across surviving forecasters (same code path as
ARM_MEDIAN). Min-forecasters guard: ABLATION_MIN_FORECASTERS = 2.

Cost: zero LLM calls. Wall-clock: O(milliseconds per qid) for binary/MC,
O(seconds) for numeric (distribution fitting).
"""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
from datetime import datetime
from typing import Any, Literal

import numpy as np
from forecasting_tools import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.forecasters import question_type_for_serialization, serialize_prediction_value
from metaculus_bot.ablation.run_stacker import ABLATION_MIN_FORECASTERS, _surviving_forecasters
from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN, MC_PROB_MAX, MC_PROB_MIN
from metaculus_bot.numeric_utils import clamp_and_renormalize_mc
from metaculus_bot.probabilistic_tools.base_rate import beta_binomial_update
from metaculus_bot.probabilistic_tools.mixtures import (
    MixtureComponent,
    MixtureOfNormals,
    percentiles_to_metaculus_cdf_via_mixture,
)
from metaculus_bot.probabilistic_tools.survival import prob_event_before
from metaculus_bot.structured_output_schema import (
    BinaryStructured,
    MultipleChoiceStructured,
    NumericStructured,
    parse_structured_block,
)

logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["run_pdf_for_qid"]

ARM_PDF = "pdf"
ARM_PDF_MIN1 = "pdf_min1"
ARM_PDF_MIN2 = "pdf_min2"
STRUCTURED_MATH_LABEL = "structured_math"

# Likelihood-ratio mapping for evidence strength (used in Bayes cascade).
_STRENGTH_TO_LR: dict[str, float] = {
    "weak": 1.5,
    "moderate": 3.0,
    "strong": 10.0,
}


# ---------------------------------------------------------------------------
# Per-forecaster structured-math computation: Binary
# ---------------------------------------------------------------------------


def _compute_binary_from_hazard(block: BinaryStructured) -> float | None:
    """Cascade step 1: constant-hazard survival model."""
    hazard = block.hazard
    if hazard is None:
        return None
    if hazard.rate_per_unit <= 0:
        return None
    result = prob_event_before(
        hazard_rate=hazard.rate_per_unit,
        elapsed_fraction=hazard.elapsed_fraction,
        remaining_fraction=hazard.remaining_fraction,
        window_length=hazard.window_duration_units,
    )
    return result.conditional_prob_given_no_event_yet


def _apply_evidence_lr(base_prob: float, evidence_items: list[Any]) -> float:
    """Apply sequential log-odds updates from evidence items."""
    if not evidence_items:
        return base_prob
    # Work in log-odds space
    if base_prob <= 0.0:
        base_prob = 0.001
    if base_prob >= 1.0:
        base_prob = 0.999
    log_odds = math.log(base_prob / (1.0 - base_prob))
    for item in evidence_items:
        if item.direction == "neutral":
            continue
        lr = _STRENGTH_TO_LR.get(item.strength, 1.0)
        if item.likelihood_ratio is not None and item.likelihood_ratio > 0:
            lr = item.likelihood_ratio
        if item.direction == "down":
            lr = 1.0 / lr
        log_odds += math.log(lr)
    # Convert back to probability
    prob = 1.0 / (1.0 + math.exp(-log_odds))
    return prob


def _compute_binary_from_bayes(block: BinaryStructured) -> float | None:
    """Cascade step 2: Beta-binomial posterior + evidence LR updates."""
    base_rate = block.base_rate
    if base_rate is None:
        return None
    if not block.evidence:
        return None
    # Compute Beta-binomial posterior mean
    result = beta_binomial_update(k=base_rate.k, n=base_rate.n)
    posterior_mean = result.posterior_mean
    # Apply evidence as sequential LR updates
    return _apply_evidence_lr(posterior_mean, block.evidence)


def _compute_binary_from_prior_blend(block: BinaryStructured) -> float | None:
    """Cascade step 3: prior.prob + evidence via log-odds shift."""
    if block.prior is None:
        return None
    if not block.evidence:
        return None
    return _apply_evidence_lr(block.prior.prob, block.evidence)


def _compute_binary_prediction(block: BinaryStructured) -> float | None:
    """Apply the binary cascade: hazard > Bayes > prior_blend. None = drop."""
    # Step 1: Hazard
    result = _compute_binary_from_hazard(block)
    if result is not None:
        return result
    # Step 2: Bayes
    result = _compute_binary_from_bayes(block)
    if result is not None:
        return result
    # Step 3: Prior blend
    result = _compute_binary_from_prior_blend(block)
    if result is not None:
        return result
    # No applicable structured math
    return None


# ---------------------------------------------------------------------------
# Per-forecaster structured-math computation: Numeric
# ---------------------------------------------------------------------------


def _compute_numeric_from_mixture(block: NumericStructured, question: NumericQuestion) -> list[Percentile] | None:
    """Path 1: Build 201-point CDF from declared mixture components."""
    if block.mixture_components is None:
        return None
    if len(block.mixture_components) < 2:
        return None
    components = tuple(MixtureComponent(weight=c.weight, mean=c.mean, sd=c.sd) for c in block.mixture_components)
    mix = MixtureOfNormals(components)
    return percentiles_to_metaculus_cdf_via_mixture(mix, question)


def _compute_numeric_from_percentiles(block: NumericStructured, question: NumericQuestion) -> list[Percentile] | None:
    """Path 2: Fit parametric family to declared percentiles, build CDF on grid."""
    from metaculus_bot.numeric_config import PCHIP_CDF_POINTS
    from metaculus_bot.pchip_cdf import enforce_min_steps, safe_cdf_bounds
    from metaculus_bot.probabilistic_tools.distributions import (
        FitType,
        eval_cdf,
        fit_lognormal_from_percentiles,
        fit_normal_from_percentiles,
        fit_student_t_from_percentiles,
    )

    percentiles = block.declared_percentiles
    if not percentiles or len(percentiles) < 2:
        return None

    hint = block.distribution_family_hint
    fit: FitType | None = None
    try:
        if hint == "lognormal":
            # Check all values positive
            if all(v > 0 for v in percentiles.values()):
                fit = fit_lognormal_from_percentiles(percentiles)
            else:
                fit = fit_student_t_from_percentiles(percentiles, df=block.student_t_df or 5.0)
        elif hint == "normal":
            fit = fit_normal_from_percentiles(percentiles)
        else:
            # Default: Student-t with df from block or 5.0
            df = block.student_t_df if block.student_t_df is not None else 5.0
            fit = fit_student_t_from_percentiles(percentiles, df=df)
    except (ValueError, RuntimeError):
        logger.debug("Parametric fit failed for numeric block; dropping forecaster")
        return None

    if fit is None:
        return None

    # Build 201-point CDF on question grid
    lower = float(question.lower_bound)
    upper = float(question.upper_bound)
    open_lower = bool(question.open_lower_bound)
    open_upper = bool(question.open_upper_bound)
    num_points = PCHIP_CDF_POINTS

    grid = np.linspace(lower, upper, num_points)
    cdf_values = np.array([eval_cdf(fit, float(x)) for x in grid], dtype=float)

    # Apply bound constraints
    from metaculus_bot.numeric_config import MIN_CDF_PROB_STEP

    hi_cap = 0.999 if open_upper else 1.0
    lo_cap = 0.001 if open_lower else 0.0
    cdf_values = np.clip(cdf_values, lo_cap, hi_cap)

    if not open_lower:
        cdf_values[0] = 0.0
    if not open_upper:
        cdf_values[-1] = 1.0

    cdf_values = np.maximum.accumulate(cdf_values)
    cdf_values = enforce_min_steps(cdf_values, MIN_CDF_PROB_STEP, upper_cap=hi_cap, lower_cap=lo_cap)
    cdf_values = np.maximum.accumulate(cdf_values)
    cdf_values = safe_cdf_bounds(cdf_values, open_lower, open_upper, MIN_CDF_PROB_STEP)

    return [Percentile(value=float(grid[i]), percentile=float(cdf_values[i])) for i in range(num_points)]


def _compute_numeric_prediction(block: NumericStructured, question: NumericQuestion) -> list[Percentile] | None:
    """Numeric cascade: mixture > percentile fit. None = drop."""
    result = _compute_numeric_from_mixture(block, question)
    if result is not None:
        return result
    return _compute_numeric_from_percentiles(block, question)


# ---------------------------------------------------------------------------
# Per-forecaster structured-math computation: Multiple choice
# ---------------------------------------------------------------------------


def _compute_mc_prediction(block: MultipleChoiceStructured) -> dict[str, float] | None:
    """Extract option_probs if well-formed. None = drop."""
    if not block.option_probs:
        return None
    # Pydantic already validated sum ~1.0 within tolerance
    return dict(block.option_probs)


# ---------------------------------------------------------------------------
# Question-type dispatch
# ---------------------------------------------------------------------------


def _question_type_label(
    question: MetaculusQuestion,
) -> Literal["binary", "numeric", "multiple_choice"]:
    if isinstance(question, BinaryQuestion):
        return "binary"
    if isinstance(question, MultipleChoiceQuestion):
        return "multiple_choice"
    if isinstance(question, NumericQuestion):
        return "numeric"
    raise ValueError(f"Unsupported question type: {type(question).__name__}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_pdf_for_qid(
    *,
    qid: int,
    question: MetaculusQuestion,
    forecaster_payloads: dict[str, dict],
    cache: AblationCache,
    force: bool = False,
    min_forecasters: int = ABLATION_MIN_FORECASTERS,
    arm_label: str = ARM_PDF,
) -> dict:
    """Run the ARM_PDF deterministic structured-math arm for one question.

    Steps:
    1. Cache check. Hit + not force returns as-is.
    2. Filter to surviving forecasters (same filter as other arms).
    3. For each survivor, parse structured block + compute prediction via cascade.
    4. Drop forecasters where structured math yields None.
    5. Min-forecasters guard (configurable via ``min_forecasters``, default 2).
    6. Aggregate via pointwise median.
    7. Cache + return payload.

    The ``arm_label`` kwarg controls the arm name written to cache and to the
    output payload. Use ``ARM_PDF_MIN1`` / ``ARM_PDF_MIN2`` to write separate
    cache files for the dual-panel pdf analysis.
    """
    if not force:
        cached = cache.read_stacker_output(qid=qid, arm=arm_label)
        if cached is not None:
            await asyncio.sleep(0)
            return cached

    surviving = _surviving_forecasters(forecaster_payloads)
    qtype_label = _question_type_label(question)

    # Per-forecaster structured-math predictions
    structured_predictions: list[Any] = []

    for slug, payload in surviving.items():
        reasoning = payload.get("reasoning", "")
        block = parse_structured_block(reasoning, qtype_label)
        if block is None:
            continue

        if isinstance(block, BinaryStructured) and isinstance(question, BinaryQuestion):
            pred = _compute_binary_prediction(block)
            if pred is not None and math.isfinite(pred):
                structured_predictions.append(pred)

        elif isinstance(block, NumericStructured) and isinstance(question, NumericQuestion):
            pred = _compute_numeric_prediction(block, question)
            if pred is not None:
                structured_predictions.append(pred)

        elif isinstance(block, MultipleChoiceStructured) and isinstance(question, MultipleChoiceQuestion):
            pred = _compute_mc_prediction(block)
            if pred is not None:
                structured_predictions.append(pred)

    n_structured = len(structured_predictions)

    if n_structured < min_forecasters:
        error_payload = {
            "success": False,
            "arm": arm_label,
            "reason": "insufficient_structured_forecasters",
            "stacker_prediction": None,
            "stacker_meta_reasoning": "",
            "computed_quantities": {},
            "cross_model_aggregation": None,
            "stacker_model_used": STRUCTURED_MATH_LABEL,
            "n_forecasters_used": n_structured,
            "ran_at": datetime.now().isoformat(),
            "tools_enabled_at_runtime": False,
            "errors": [],
        }
        cache.write_stacker_output(qid=qid, arm=arm_label, payload=error_payload)
        await asyncio.sleep(0)
        return error_payload

    # Aggregate
    aggregated: Any
    if isinstance(question, BinaryQuestion):
        aggregated = _aggregate_binary(structured_predictions)
    elif isinstance(question, MultipleChoiceQuestion):
        aggregated = _aggregate_mc(structured_predictions, question)
    elif isinstance(question, NumericQuestion):
        aggregated = await _aggregate_numeric_predictions(structured_predictions, question)
    else:
        raise ValueError(f"Unsupported question type: {type(question).__name__}")

    serialized_prediction = serialize_prediction_value(aggregated, question_type_for_serialization(question))

    success_payload = {
        "success": True,
        "arm": arm_label,
        "stacker_prediction": serialized_prediction,
        "stacker_meta_reasoning": "",
        "computed_quantities": {},
        "cross_model_aggregation": None,
        "stacker_model_used": STRUCTURED_MATH_LABEL,
        "n_forecasters_used": n_structured,
        "ran_at": datetime.now().isoformat(),
        "tools_enabled_at_runtime": False,
        "errors": [],
    }
    cache.write_stacker_output(qid=qid, arm=arm_label, payload=success_payload)
    await asyncio.sleep(0)
    return success_payload


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _aggregate_binary(predictions: list[float]) -> float:
    """Median of binary probabilities, clamped to [BINARY_PROB_MIN, BINARY_PROB_MAX]."""
    median_prob = statistics.median(predictions)
    return max(BINARY_PROB_MIN, min(BINARY_PROB_MAX, float(median_prob)))


def _aggregate_mc(
    predictions: list[dict[str, float]],
    question: MultipleChoiceQuestion,
) -> PredictedOptionList:
    """Option-wise median across forecasters, then clamp + renormalize."""
    option_order = list(question.options)
    per_option_values: dict[str, list[float]] = {name: [] for name in option_order}
    for pred in predictions:
        for name in option_order:
            if name in pred:
                per_option_values[name].append(pred[name])

    raw_probs = {}
    for name in option_order:
        values = per_option_values[name]
        raw_probs[name] = float(statistics.median(values)) if values else 1.0 / len(option_order)

    clamped = {name: max(MC_PROB_MIN, min(MC_PROB_MAX, p)) for name, p in raw_probs.items()}
    total = sum(clamped.values())
    normalized = {name: (p / total) for name, p in clamped.items()} if total > 0 else clamped
    aggregated_options = [PredictedOption(option_name=name, probability=normalized[name]) for name in option_order]
    aggregated_list = PredictedOptionList(predicted_options=aggregated_options)
    return clamp_and_renormalize_mc(aggregated_list)


async def _aggregate_numeric_predictions(
    predictions: list[list[Percentile]],
    question: NumericQuestion,
) -> Any:
    """Pointwise median of per-forecaster 201-point CDFs.

    Operates directly on the probability arrays rather than wrapping in
    NumericDistribution objects (which enforce strict-monotonicity on
    declared_percentiles — a constraint our 201-point CDFs can violate
    at the tails). Takes the pointwise median, then wraps the result.
    """
    await asyncio.sleep(0)  # cooperative yield for flake8-async ASYNC910
    from metaculus_bot.pchip_processing import create_pchip_numeric_distribution  # noqa: PLC0415

    # All predictions have the same grid (201 points on [lower, upper])
    n_points = len(predictions[0])
    prob_arrays = np.array([[p.percentile for p in perc_list] for perc_list in predictions], dtype=float)
    median_probs = np.median(prob_arrays, axis=0)

    # Ensure monotonic + clipped to [0, 1]
    median_probs = np.clip(median_probs, 0.0, 1.0)
    median_probs = np.maximum.accumulate(median_probs)

    # Build the result as a PchipNumericDistribution
    grid = np.linspace(float(question.lower_bound), float(question.upper_bound), n_points)
    # declared_percentiles for the wrapper must be strictly increasing — filter
    # to a subset that satisfies this constraint (use every 20th point as stand-in).
    stride = max(1, n_points // 10)
    subset_indices = list(range(0, n_points, stride))
    if subset_indices[-1] != n_points - 1:
        subset_indices.append(n_points - 1)
    # Ensure strictly increasing values in subset
    declared_subset: list[Percentile] = []
    prev_p = -1.0
    for idx in subset_indices:
        p = float(median_probs[idx])
        if p > prev_p:
            declared_subset.append(Percentile(percentile=min(p, 1.0), value=float(grid[idx])))
            prev_p = p

    return create_pchip_numeric_distribution(
        pchip_cdf=list(map(float, median_probs)),
        percentile_list=declared_subset,
        question=question,
        zero_point=question.zero_point,
    )
