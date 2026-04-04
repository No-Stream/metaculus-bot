"""Spread metrics measuring forecaster disagreement across question types.

Used to decide whether to invoke an expensive stacking meta-analysis when
individual model predictions diverge significantly.
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any

from forecasting_tools import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import MetaculusQuestion

logger: logging.Logger = logging.getLogger(__name__)

# Clamp bounds for log-odds conversion to avoid log(0)
_LOG_ODDS_CLAMP_MIN: float = 0.001
_LOG_ODDS_CLAMP_MAX: float = 0.999

# Indices into the standard 11-percentile list (2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5)
# that correspond to the 10th, 50th, and 90th percentiles.
_KEY_PERCENTILE_INDICES: list[int] = [2, 5, 8]


def binary_log_odds_spread(prediction_values: list[float]) -> float:
    """Compute the log-odds range across binary probability predictions.

    Converts each probability to log-odds space where tail disagreement is
    amplified (e.g. 1% vs 19% is a much larger spread than 50% vs 68% despite
    similar absolute gaps). Returns max(log_odds) - min(log_odds).
    """
    if len(prediction_values) < 2:
        raise ValueError("binary_log_odds_spread requires at least 2 predictions")

    def _to_log_odds(p: float) -> float:
        clamped = max(_LOG_ODDS_CLAMP_MIN, min(_LOG_ODDS_CLAMP_MAX, p))
        return math.log(clamped / (1.0 - clamped))

    log_odds = [_to_log_odds(p) for p in prediction_values]
    return max(log_odds) - min(log_odds)


def mc_max_option_spread(prediction_values: list[PredictedOptionList]) -> float:
    """Compute the maximum per-option probability range across MC predictions.

    For each option name, collects all models' probabilities and computes
    max - min. Returns the largest such range across all options.
    """
    if len(prediction_values) < 2:
        raise ValueError("mc_max_option_spread requires at least 2 predictions")

    first_options = {o.option_name for o in prediction_values[0].predicted_options}
    for i, pred in enumerate(prediction_values[1:], start=2):
        pred_options = {o.option_name for o in pred.predicted_options}
        if pred_options != first_options:
            raise ValueError(
                f"mc_max_option_spread: model {i} has options {sorted(pred_options)}, expected {sorted(first_options)}"
            )

    option_probabilities: dict[str, list[float]] = {}
    for pred in prediction_values:
        for option in pred.predicted_options:
            option_probabilities.setdefault(option.option_name, []).append(option.probability)

    max_spread = 0.0
    for probs in option_probabilities.values():
        spread = max(probs) - min(probs)
        max_spread = max(max_spread, spread)

    return max_spread


def numeric_percentile_spread(
    prediction_values: list[list[Percentile]],
    question: NumericQuestion,
) -> float:
    """Compute the max normalized spread at key percentiles (10th, 50th, 90th).

    For closed-bound questions, normalizes by the question range. For
    open-ended questions, falls back to the ensemble interquartile range
    (median of 90th percentiles minus median of 10th percentiles) as the
    denominator.
    """
    if len(prediction_values) < 2:
        raise ValueError("numeric_percentile_spread requires at least 2 predictions")

    min_required = max(_KEY_PERCENTILE_INDICES) + 1  # 9
    for model_pcts in prediction_values:
        if len(model_pcts) < min_required:
            raise ValueError(
                f"numeric_percentile_spread requires at least {min_required} percentiles per model,"
                f" got {len(model_pcts)}"
            )

    has_finite_range = (
        not question.open_lower_bound
        and not question.open_upper_bound
        and math.isfinite(question.upper_bound)
        and math.isfinite(question.lower_bound)
    )

    if has_finite_range:
        denominator = question.upper_bound - question.lower_bound
    else:
        # Fallback: ensemble IQR from 10th and 90th percentiles across all models
        p10_values = [model_pcts[_KEY_PERCENTILE_INDICES[0]].value for model_pcts in prediction_values]
        p90_values = [model_pcts[_KEY_PERCENTILE_INDICES[2]].value for model_pcts in prediction_values]
        denominator = statistics.median(p90_values) - statistics.median(p10_values)

    if denominator <= 0:
        logger.warning(f"numeric_percentile_spread: non-positive {denominator=}, returning 0.0")
        return 0.0

    max_normalized_spread = 0.0
    for idx in _KEY_PERCENTILE_INDICES:
        values_at_percentile = [model_pcts[idx].value for model_pcts in prediction_values]
        raw_spread = max(values_at_percentile) - min(values_at_percentile)
        normalized = raw_spread / denominator
        max_normalized_spread = max(max_normalized_spread, normalized)

    return max_normalized_spread


def compute_spread(question: MetaculusQuestion, prediction_values: list[Any]) -> float:
    """Dispatch to the appropriate spread metric based on question type."""
    if isinstance(question, BinaryQuestion):
        return binary_log_odds_spread(prediction_values)
    if isinstance(question, MultipleChoiceQuestion):
        return mc_max_option_spread(prediction_values)
    if isinstance(question, NumericQuestion):
        percentile_lists = [pv.declared_percentiles for pv in prediction_values]
        return numeric_percentile_spread(percentile_lists, question)
    raise ValueError(f"Unsupported question type for spread metrics: {type(question).__name__}")
