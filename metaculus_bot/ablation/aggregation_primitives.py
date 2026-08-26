"""Shared aggregation primitives for binary and MC predictions.

Used by both run_simple_agg.py (PredictedOptionList inputs) and run_pdf.py
(raw dict inputs). Each caller accumulates its own input format into the
normalized form these functions accept.
"""

from __future__ import annotations

import statistics
from typing import Any, Literal

from forecasting_tools import PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN
from metaculus_bot.mc_processing import clamp_and_renormalize_probs

__all__ = ["aggregate_binary", "aggregate_mc"]

_AGG_FUNC: dict[str, Any] = {
    "mean": statistics.mean,
    "median": statistics.median,
}


def aggregate_binary(predictions: list[float], method: Literal["mean", "median"]) -> float:
    """Central tendency of binary probabilities, clamped to [BINARY_PROB_MIN, BINARY_PROB_MAX]."""
    central = _AGG_FUNC[method](predictions)
    return max(BINARY_PROB_MIN, min(BINARY_PROB_MAX, float(central)))


def aggregate_mc(
    per_option_values: dict[str, list[float]],
    option_order: list[str],
    method: Literal["mean", "median"],
) -> PredictedOptionList:
    """Option-wise central tendency, then clamp + renormalize.

    Accepts pre-accumulated per-option value lists. **Raises when an option has no
    values**: a uniform ``1/N`` there is a probability no model declared, imputed into
    an aggregate that then reads as a real forecast — the same defect the production
    ``build_mc_prediction`` stopped doing. Upstream accumulation covers every option in
    ``option_order`` (each contributing ballot is validated against the question's full
    option set), so an empty list means that invariant broke and the run should say so
    rather than average in an invented share.
    """
    agg_fn = _AGG_FUNC[method]
    ordered_raw: list[float] = []
    for name in option_order:
        values = per_option_values.get(name, [])
        if not values:
            raise ValueError(
                f"no model declared a probability for option {name!r}; "
                f"cannot aggregate without imputing one (options={list(option_order)})"
            )
        ordered_raw.append(float(agg_fn(values)))

    # Drift-free clamp + renormalize (the single shared helper build_mc_prediction
    # uses). Guarantees every option lands in [MC_PROB_MIN, MC_PROB_MAX] summing to
    # 1.0, so the PredictedOptionList validator is a no-op on construction — unlike
    # the old manual clamp-then-divide, where renormalization could drag a floored
    # option back below the floor.
    normalized = clamp_and_renormalize_probs(ordered_raw)
    aggregated_options = [PredictedOption(option_name=name, probability=p) for name, p in zip(option_order, normalized)]
    return PredictedOptionList(predicted_options=aggregated_options)
