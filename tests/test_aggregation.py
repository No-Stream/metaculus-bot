import logging
from datetime import datetime
from typing import cast

import numpy as np
import pytest
from forecasting_tools import GeneralLlm
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.numeric.config import PCHIP_CDF_POINTS, STANDARD_PERCENTILES
from metaculus_bot.numeric.pchip_processing import create_fallback_numeric_distribution
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric.utils import aggregate_numeric


@pytest.mark.asyncio
async def test_numeric_aggregation_configurable():
    """
    Tests that the aggregation method for numeric questions can be configured
    and correctly applies either mean or median as specified.
    """
    # 1. Arrange
    # Create a mock numeric question to provide context for the aggregation.
    question = NumericQuestion(
        id_of_question=1,
        id_of_post=1,
        page_url="https://www.metaculus.com/questions/1/test",
        question_text="Test question for numeric aggregation?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=datetime.fromisoformat("2023-01-01T00:00:00"),
        close_time=datetime.fromisoformat("2025-01-01T00:00:00"),
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
    )

    # Create two different numeric distributions to be aggregated.
    # `dist1` is a uniform distribution.
    # `dist2` is skewed towards the upper bound.
    x_axis = [p * 100 for p in np.linspace(0, 1, 11) if 0 < p < 1]
    dist1_percentiles = [Percentile(value=v, percentile=p) for v, p in zip(x_axis, np.linspace(0, 1, 11)[1:-1])]
    dist2_percentiles = [Percentile(value=v, percentile=p**0.5) for v, p in zip(x_axis, np.linspace(0, 1, 11)[1:-1])]

    # Third distribution: quadratic skew towards lower bound.
    dist3_percentiles = [Percentile(value=v, percentile=p**2) for v, p in zip(x_axis, np.linspace(0, 1, 11)[1:-1])]

    common_args = {
        "open_lower_bound": question.open_lower_bound,
        "open_upper_bound": question.open_upper_bound,
        "lower_bound": question.lower_bound,
        "upper_bound": question.upper_bound,
        "zero_point": question.zero_point,
    }

    pred1 = NumericDistribution(declared_percentiles=dist1_percentiles, **common_args)
    pred2 = NumericDistribution(declared_percentiles=dist2_percentiles, **common_args)
    pred3 = NumericDistribution(declared_percentiles=dist3_percentiles, **common_args)
    predictions: list[PredictionTypes] = [pred1, pred2, pred3]

    # Initialize two forecaster instances with different aggregation methods.
    llms_min: dict[str, str | GeneralLlm] = {
        "default": "mock",
        "parser": "mock",
        "researcher": "mock",
        "summarizer": "mock",
    }
    forecaster_mean = TemplateForecaster(llms=llms_min, aggregation_strategy=AggregationStrategy.MEAN)
    forecaster_median = TemplateForecaster(llms=llms_min, aggregation_strategy=AggregationStrategy.MEDIAN)

    # 2. Act
    # Run the aggregation for both the 'mean' and 'median' configurations.
    mean_agg_result_uncast = await forecaster_mean._aggregate_predictions(predictions, question)
    median_agg_result_uncast = await forecaster_median._aggregate_predictions(predictions, question)
    mean_agg_result = cast(NumericDistribution, mean_agg_result_uncast)
    median_agg_result = cast(NumericDistribution, median_agg_result_uncast)

    # 3. Assert
    # Manually calculate the expected CDFs to verify the aggregation logic.
    expected_mean_cdf_percentiles = np.mean(
        [
            [p.percentile for p in pred1.cdf],
            [p.percentile for p in pred2.cdf],
            [p.percentile for p in pred3.cdf],
        ],
        axis=0,
    )
    expected_median_cdf_percentiles = np.median(
        [
            [p.percentile for p in pred1.cdf],
            [p.percentile for p in pred2.cdf],
            [p.percentile for p in pred3.cdf],
        ],
        axis=0,
    )

    # Extract the percentile values from the results.
    result_mean_percentiles = [p.percentile for p in mean_agg_result.declared_percentiles]
    result_median_percentiles = [p.percentile for p in median_agg_result.declared_percentiles]

    # Verify that the aggregated distributions match the expected values.
    assert np.allclose(result_mean_percentiles, expected_mean_cdf_percentiles)
    assert np.allclose(result_median_percentiles, expected_median_cdf_percentiles)

    # Ensure the two aggregation methods produce different results.
    assert not np.allclose(result_mean_percentiles, result_median_percentiles)


_MIXED_GRID_QUESTION = NumericQuestion(
    id_of_question=77,
    id_of_post=77,
    page_url="https://www.metaculus.com/questions/77/test",
    question_text="Mixed-grid ensemble?",
    background_info="",
    resolution_criteria="",
    fine_print="",
    lower_bound=12.3,
    upper_bound=987.65,
    open_lower_bound=False,
    open_upper_bound=False,
    unit_of_measure="",
    zero_point=None,
    cdf_size=PCHIP_CDF_POINTS,
)

_BASE_VALUES = [50.0, 90.0, 130.0, 180.0, 280.0, 420.0, 500.0, 580.0, 720.0, 830.0, 890.0, 930.0, 960.0]


def _declared(values: list[float]) -> list[Percentile]:
    return [Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, values)]


def _pchip_prediction(values: list[float]) -> NumericDistribution:
    sanitized, zero_point = sanitize_percentiles(_declared(values), _MIXED_GRID_QUESTION)
    return build_numeric_distribution(sanitized, _MIXED_GRID_QUESTION, zero_point)


def _ft_fallback_prediction(values: list[float]) -> NumericDistribution:
    """A distribution off the forecasting-tools fallback builder — the path whose
    value axis is ``min + span*i/(n-1)`` rather than ``np.linspace``."""
    sanitized, zero_point = sanitize_percentiles(_declared(values), _MIXED_GRID_QUESTION)
    return create_fallback_numeric_distribution(sanitized, _MIXED_GRID_QUESTION, zero_point)


class _ShortGridDistribution(NumericDistribution):
    """A model whose CDF genuinely arrives on a coarser grid than the question's."""

    def get_cdf(self) -> list[Percentile]:
        heights = np.linspace(0.0, 1.0, 51)
        x_values = np.linspace(self.lower_bound, self.upper_bound, 51)
        return [Percentile(percentile=float(h), value=float(x)) for x, h in zip(x_values, heights)]


class TestEnsembleCdfGridAlignment:
    """Every model contributes to every aggregated grid point (M9, 2026-08-25).

    Aggregation used to concatenate the models' CDFs and ``groupby`` the float
    ``value`` axis. The PCHIP path builds that axis with ``np.linspace`` and the
    forecasting-tools fallback builder with ``min + span*i/(n-1)`` — equal in
    exact arithmetic, different in the last bits — so a mixed-path ensemble
    produced a ~270-value union of which most points had FEWER than n
    contributors. The "median" was then a median over a rotating subset of the
    ensemble (measured: 136 of 269 groups short-handed, aggregate CDF off the true
    median by up to 0.0078), and the length mismatch pushed a continuous question
    through the discrete-resample branch, logging "Discrete aggregation detected".
    """

    def _mixed_ensemble(self) -> list[NumericDistribution]:
        return [
            _pchip_prediction(_BASE_VALUES),
            _pchip_prediction([v * 0.97 for v in _BASE_VALUES]),
            _ft_fallback_prediction([v + 8.0 for v in _BASE_VALUES]),
        ]

    @pytest.mark.parametrize("method", ["mean", "median"])
    def test_aggregate_is_the_exact_pointwise_statistic(self, method: str) -> None:
        predictions = self._mixed_ensemble()
        heights = np.array([[p.percentile for p in pred.get_cdf()] for pred in predictions], dtype=float)
        expected = heights.mean(axis=0) if method == "mean" else np.median(heights, axis=0)

        aggregated = aggregate_numeric(predictions, _MIXED_GRID_QUESTION, method)
        actual = np.array([p.percentile for p in aggregated.get_cdf()], dtype=float)

        assert len(actual) == PCHIP_CDF_POINTS
        assert np.allclose(actual, expected, atol=1e-12), f"max deviation {np.max(np.abs(actual - expected))}"

    def test_continuous_question_does_not_log_discrete_aggregation(self, caplog) -> None:
        with caplog.at_level(logging.INFO, logger="metaculus_bot.numeric.utils"):
            aggregate_numeric(self._mixed_ensemble(), _MIXED_GRID_QUESTION, "median")

        assert not [r for r in caplog.records if "Discrete aggregation detected" in r.getMessage()]

    def test_short_grid_model_is_resampled_and_reported(self, caplog) -> None:
        sanitized, zero_point = sanitize_percentiles(_declared(_BASE_VALUES), _MIXED_GRID_QUESTION)
        short = _ShortGridDistribution(
            declared_percentiles=sanitized,
            open_upper_bound=_MIXED_GRID_QUESTION.open_upper_bound,
            open_lower_bound=_MIXED_GRID_QUESTION.open_lower_bound,
            upper_bound=_MIXED_GRID_QUESTION.upper_bound,
            lower_bound=_MIXED_GRID_QUESTION.lower_bound,
            zero_point=zero_point,
            cdf_size=PCHIP_CDF_POINTS,
            strict_validation=False,
            standardize_cdf=False,
        )
        predictions = [_pchip_prediction(_BASE_VALUES), short]

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.numeric.utils"):
            aggregated = aggregate_numeric(predictions, _MIXED_GRID_QUESTION, "median")

        markers = [r.getMessage() for r in caplog.records if "NUMERIC_AGGREGATE_GRID_MISMATCH:" in r.getMessage()]
        assert len(markers) == 1
        assert "got_points=51" in markers[0]
        assert f"expected_points={PCHIP_CDF_POINTS}" in markers[0]

        heights = np.array([p.percentile for p in aggregated.get_cdf()], dtype=float)
        assert len(heights) == PCHIP_CDF_POINTS
        assert np.all(np.diff(heights) > 0)
