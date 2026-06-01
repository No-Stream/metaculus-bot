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

from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN, MC_PROB_MAX, MC_PROB_MIN
from metaculus_bot.numeric.utils import clamp_and_renormalize_mc

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

    Accepts pre-accumulated per-option value lists. Handles missing options
    with a uniform fallback (1/N).
    """
    agg_fn = _AGG_FUNC[method]
    n_options = len(option_order)
    raw_probs: dict[str, float] = {}
    for name in option_order:
        values = per_option_values.get(name, [])
        if not values:
            raw_probs[name] = 1.0 / n_options
        else:
            raw_probs[name] = float(agg_fn(values))

    clamped = {name: max(MC_PROB_MIN, min(MC_PROB_MAX, p)) for name, p in raw_probs.items()}
    total = sum(clamped.values())
    normalized = {name: (p / total) for name, p in clamped.items()} if total > 0 else clamped
    aggregated_options = [PredictedOption(option_name=name, probability=normalized[name]) for name in option_order]
    aggregated_list = PredictedOptionList(predicted_options=aggregated_options)
    return clamp_and_renormalize_mc(aggregated_list)
