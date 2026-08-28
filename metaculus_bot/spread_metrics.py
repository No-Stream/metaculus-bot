"""Spread metrics measuring forecaster disagreement across question types.

Used to decide whether to invoke an expensive stacking meta-analysis when
individual model predictions diverge significantly.
"""

from __future__ import annotations

import logging
import math
import statistics
from typing import Any

import numpy as np
from forecasting_tools import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.numeric.percentile_set import EXPECTED_KEYS, PercentileSet, percentile_key
from metaculus_bot.prob_math_utils import logit

logger: logging.Logger = logging.getLogger(__name__)

# Percentile LABELS (not positions) at which numeric disagreement is measured:
# the 10th, 50th, and 90th percentiles. Accessed by label so growing the standard
# percentile set (e.g. 11 -> 13) cannot silently shift them.
_KEY_SPREAD_PERCENTILES: list[float] = [0.10, 0.50, 0.90]


def _has_standard_labels(model_pcts: list[Percentile]) -> bool:
    """True iff the model's percentile labels are exactly the standard percentile set.

    Uses ``percentile_set``'s canonical rounded key set (``EXPECTED_KEYS``) to
    distinguish continuous forecaster output (the standard set) from a
    discrete-resampled cumulative-probability CDF grid. The length check guards
    against a duplicate label masquerading as the standard set.
    """
    return (
        len(model_pcts) == len(EXPECTED_KEYS)
        and frozenset(percentile_key(p.percentile) for p in model_pcts) == EXPECTED_KEYS
    )


def _key_percentile_values(model_pcts: list[Percentile]) -> tuple[float, float, float]:
    """Return the (P10, P50, P90) VALUES for one model's declared percentiles.

    Continuous forecaster output declares the standard percentile set, so P10/P50/P90
    resolve to exact label nodes via ``PercentileSet`` — byte-identical to the old strict
    path.

    Discrete numeric questions are different: ``build_numeric_distribution`` OVERWRITES a
    model's ``declared_percentiles`` with a resampled CDF grid whose labels are CUMULATIVE
    PROBABILITIES (0.0..1.0), not the standard set. For those, read the value at cumulative
    probability 0.10/0.50/0.90 by treating each ``(value, percentile)`` point as an empirical
    CDF and interpolating with ``np.interp`` (which requires the probability axis sorted
    ascending).

    A list that is neither the standard set nor a CDF grid spanning the key percentiles
    (e.g. a truncated percentile list) cannot yield P10/P50/P90 without extrapolating
    garbage, so it fails fast.
    """
    if _has_standard_labels(model_pcts):
        percentile_set = PercentileSet.from_percentiles(model_pcts)
        return (
            percentile_set.value_at(0.10),
            percentile_set.value_at(0.50),
            percentile_set.value_at(0.90),
        )

    ordered = sorted(model_pcts, key=lambda p: p.percentile)
    labels = [p.percentile for p in ordered]
    values = [p.value for p in ordered]
    if not labels:
        raise ValueError("numeric_percentile_spread: declared percentiles are empty; cannot compute spread")
    # CDF grids from discrete resampling have many points (typically 50-201) with
    # cumulative-probability labels. Open-bound questions can legitimately have
    # heavy out-of-bound mass (e.g. labels[0]=0.40 means 40% mass below the
    # displayed lower bound — the "Toy Story" scenario). np.interp clamps: if
    # a key percentile (0.10) is below labels[0], it returns values[0] (the
    # displayed bound), which is the honest answer.
    # Only reject short non-grid lists that clearly can't interpolate key
    # percentiles (< 5 points AND doesn't span P10-P90).
    _MIN_GRID_POINTS = 5
    is_plausible_grid = len(labels) >= _MIN_GRID_POINTS
    spans_key_percentiles = labels[0] <= 0.10 and labels[-1] >= 0.90
    if not is_plausible_grid and not spans_key_percentiles:
        raise ValueError(
            "numeric_percentile_spread: declared percentiles are neither the standard percentiles "
            f"nor a CDF grid spanning P10-P90; cannot read key percentiles from labels "
            f"(len={len(labels)}, range=[{labels[0]:.4f}, {labels[-1]:.4f}])"
        )
    p10, p50, p90 = np.interp(_KEY_SPREAD_PERCENTILES, labels, values)
    return float(p10), float(p50), float(p90)


def binary_prob_range_spread(prediction_values: list[float]) -> float:
    """Compute the probability range (max - min) across binary predictions.

    Active trigger metric for conditional stacking. Plain probability range
    avoids the log-odds failure mode where clamped-extreme predictions (~0 or ~1)
    produce huge spreads even when the ensemble-median answer is correct.
    """
    if len(prediction_values) < 2:
        raise ValueError("binary_prob_range_spread requires at least 2 predictions")
    return max(prediction_values) - min(prediction_values)


def binary_log_odds_spread(prediction_values: list[float]) -> float:
    """Log-odds range across binary predictions.

    Uses the shared ``logit`` (canonical eps=1e-4) to avoid log(0). Available as
    an alternative spread metric; see binary_prob_range_spread.
    """
    if len(prediction_values) < 2:
        raise ValueError("binary_log_odds_spread requires at least 2 predictions")

    log_odds = [logit(p) for p in prediction_values]
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

    A non-positive denominator makes the ratio UNDEFINED — the reachable case is
    an open-bound question whose models all interpolate to the same displayed
    bound at P10 and P90 (heavy out-of-bound mass clamped by ``np.interp``). This
    returns ``math.inf`` there, never ``0.0``: a failed measurement must not read
    as an affirmative "the models agree" (which is exactly how
    ``route_after_forecasts`` reads a spread of 0 — MEDIAN, skip stacking, marker
    ``spread_below_threshold``). The caller reads ``inf`` as its own case rather than as a
    huge spread: it routes to MEDIAN without spending a crux extraction, a targeted search
    and a stacker call on no measurement, and stamps the skip reason ``spread_undefined`` so
    the marker never claims agreement. The ``SPREAD_UNDEFINED`` WARN names the question.
    """
    if len(prediction_values) < 2:
        raise ValueError("numeric_percentile_spread requires at least 2 predictions")

    # Read the (P10, P50, P90) VALUES per model by label. Continuous forecaster output
    # uses the strict PercentileSet path; discrete-resampled CDF grids (cumulative-
    # probability labels) are interpolated. Both resolve the same three percentiles.
    key_values_by_model = [_key_percentile_values(model_pcts) for model_pcts in prediction_values]
    p10_values = [values[0] for values in key_values_by_model]
    p50_values = [values[1] for values in key_values_by_model]
    p90_values = [values[2] for values in key_values_by_model]
    values_by_percentile = {0.10: p10_values, 0.50: p50_values, 0.90: p90_values}

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
        denominator = statistics.median(p90_values) - statistics.median(p10_values)

    if denominator <= 0:
        logger.warning(
            "SPREAD_UNDEFINED: question=%s qtype=numeric denominator=%.6g models=%d — key-percentile spread is "
            "unmeasurable (non-positive denominator); reporting inf so it cannot read as agreement",
            getattr(question, "id_of_question", None),
            denominator,
            len(prediction_values),
        )
        return math.inf

    max_normalized_spread = 0.0
    for pct in _KEY_SPREAD_PERCENTILES:
        values_at_percentile = values_by_percentile[pct]
        raw_spread = max(values_at_percentile) - min(values_at_percentile)
        normalized = raw_spread / denominator
        max_normalized_spread = max(max_normalized_spread, normalized)

    return max_normalized_spread


def compute_spread(question: MetaculusQuestion, prediction_values: list[Any]) -> float:
    """Dispatch to the appropriate spread metric based on question type."""
    if isinstance(question, BinaryQuestion):
        return binary_prob_range_spread(prediction_values)
    if isinstance(question, MultipleChoiceQuestion):
        return mc_max_option_spread(prediction_values)
    if isinstance(question, NumericQuestion):
        percentile_lists = [pv.declared_percentiles for pv in prediction_values]
        return numeric_percentile_spread(percentile_lists, question)
    raise ValueError(f"Unsupported question type for spread metrics: {type(question).__name__}")
