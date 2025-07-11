from __future__ import annotations

"""Numeric aggregation and helper utilities used by TemplateForecaster.

This module centralises logic for combining numeric forecasts and constructing
user-friendly bound messages so that the core forecaster class stays small.
"""

from typing import Literal, Sequence, cast

import numpy as np
import pandas as pd

from forecasting_tools.data_models.numeric_report import (
    NumericDistribution,
    NumericReport,
    Percentile,
)
from forecasting_tools.data_models.questions import NumericQuestion

__all__ = [
    "aggregate_numeric",
    "aggregate_binary_mean",
    "bound_messages",
]


def aggregate_binary_mean(predictions: Sequence[float]) -> float:
    """Return the mean of *binary* forecasts rounded to three decimals.

    This matches the old behaviour from `TemplateForecaster`.
    """

    if not predictions:
        raise ValueError("Cannot aggregate empty list of binary predictions")

    mean_prediction = sum(predictions) / len(predictions)
    return round(mean_prediction, 3)


async def aggregate_numeric(
    predictions: Sequence[NumericDistribution],
    question: NumericQuestion,
    method: str | Literal["mean", "median"] = "mean",
) -> NumericDistribution:
    """Aggregate numeric distributions by mean or median.

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

    if method == "median":
        # Delegate to helper from forecasting_tools – preserves previous behaviour.
        return await NumericReport.aggregate_predictions(list(predictions), question)  # type: ignore[arg-type]

    if method != "mean":
        raise ValueError(f"Invalid aggregation method: {method}")

    # ---- Mean aggregation (matches previous in-class implementation) ----
    numeric_predictions = list(predictions)

    # We now use the full interpolated CDF for aggregation, same as the median method.
    cdfs_as_dfs = [pd.DataFrame([p.model_dump() for p in pred.cdf]) for pred in numeric_predictions]
    combined_cdf = pd.concat(cdfs_as_dfs)
    mean_series = combined_cdf.groupby("value")["percentile"].mean()

    mean_cdf = [
        Percentile(value=cast(float, v), percentile=cast(float, p))
        for v, p in mean_series.items()
    ]

    return NumericDistribution(
        declared_percentiles=mean_cdf,
        open_upper_bound=question.open_upper_bound,
        open_lower_bound=question.open_lower_bound,
        # DO NOT INFER UPPER AND LOWER BOUNDS FROM THE CDF! that will break b/c each model's CDF will not be aligned.
        upper_bound=question.upper_bound,
        lower_bound=question.lower_bound,
        zero_point=question.zero_point,
    )


def bound_messages(question: NumericQuestion) -> tuple[str, str]:
    """Return upper & lower bound helper messages for numeric prompts."""

    upper_msg = "" if question.open_upper_bound else f"The outcome can not be higher than {question.upper_bound}."
    lower_msg = "" if question.open_lower_bound else f"The outcome can not be lower than {question.lower_bound}."
    return upper_msg, lower_msg 