"""Tests for metaculus_bot.forecaster_runners — extracted per-type forecast functions.

Exercises the three public functions (run_binary_forecast, run_mc_forecast,
run_numeric_forecast) to verify they produce the same results as the original
TemplateForecaster methods they replaced.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)

from metaculus_bot.forecaster_runners import run_binary_forecast, run_mc_forecast, run_numeric_forecast


@pytest.fixture
def parser_llm():
    return GeneralLlm(model="test-parser")


@pytest.fixture
def forecaster_llm():
    return GeneralLlm(model="test-forecaster")


@pytest.fixture
def binary_question():
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = 1001
    q.page_url = "https://metaculus.com/questions/1001"
    q.question_text = "Will X happen?"
    return q


@pytest.fixture
def mc_question():
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.id_of_question = 2001
    q.page_url = "https://metaculus.com/questions/2001"
    q.question_text = "Which outcome?"
    q.options = ["Option A", "Option B", "Option C"]
    return q


@pytest.fixture
def numeric_question():
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = 3001
    q.page_url = "https://metaculus.com/questions/3001"
    q.lower_bound = 0
    q.upper_bound = 1000
    q.open_lower_bound = False
    q.open_upper_bound = True
    q.unit_of_measure = "widgets"
    return q


class TestRunBinaryForecast:
    @pytest.mark.asyncio
    async def test_returns_reasoned_prediction_with_clamped_value(self, binary_question, forecaster_llm, parser_llm):
        """Binary forecast clamps to [BINARY_PROB_MIN, BINARY_PROB_MAX] and returns ReasonedPrediction."""
        from forecasting_tools import BinaryPrediction

        reasoning_text = "Analysis: likely yes.\n\nProbability: 75%"

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
            patch(
                "metaculus_bot.forecaster_runners.structure_output",
                new=AsyncMock(return_value=BinaryPrediction(prediction_in_decimal=0.75)),
            ),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert isinstance(result, ReasonedPrediction)
        assert result.prediction_value == 0.75
        assert result.reasoning == reasoning_text

    @pytest.mark.asyncio
    async def test_clamps_below_minimum(self, binary_question, forecaster_llm, parser_llm):
        """Values below BINARY_PROB_MIN are clamped up."""
        from forecasting_tools import BinaryPrediction

        from metaculus_bot.constants import BINARY_PROB_MIN

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="Very unlikely")),
            patch(
                "metaculus_bot.forecaster_runners.structure_output",
                new=AsyncMock(return_value=BinaryPrediction(prediction_in_decimal=0.001)),
            ),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert result.prediction_value == BINARY_PROB_MIN

    @pytest.mark.asyncio
    async def test_clamps_above_maximum(self, binary_question, forecaster_llm, parser_llm):
        """Values above BINARY_PROB_MAX are clamped down."""
        from forecasting_tools import BinaryPrediction

        from metaculus_bot.constants import BINARY_PROB_MAX

        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="Nearly certain")),
            patch(
                "metaculus_bot.forecaster_runners.structure_output",
                new=AsyncMock(return_value=BinaryPrediction(prediction_in_decimal=0.999)),
            ),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert result.prediction_value == BINARY_PROB_MAX


def _make_option_list(options: list[tuple[str, float]]) -> PredictedOptionList:
    from forecasting_tools.data_models.multiple_choice_report import PredictedOption

    return PredictedOptionList(predicted_options=[PredictedOption(option_name=n, probability=p) for n, p in options])


class TestRunMcForecast:
    @pytest.mark.asyncio
    async def test_returns_reasoned_prediction_with_option_list(self, mc_question, forecaster_llm, parser_llm):
        """MC forecast returns a ReasonedPrediction with PredictedOptionList."""
        reasoning_text = "Option A most likely."
        option_list = _make_option_list([("Option A", 0.6), ("Option B", 0.3), ("Option C", 0.1)])

        with (
            patch("metaculus_bot.forecaster_runners.multiple_choice_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
            patch(
                "metaculus_bot.forecaster_runners.structure_output",
                new=AsyncMock(return_value=option_list),
            ),
            patch("metaculus_bot.forecaster_runners.clamp_and_renormalize_mc", return_value=option_list),
        ):
            result = await run_mc_forecast(mc_question, "research", forecaster_llm, parser_llm)

        assert isinstance(result, ReasonedPrediction)
        assert result.prediction_value == option_list
        assert result.reasoning == reasoning_text

    @pytest.mark.asyncio
    async def test_fallback_to_build_mc_prediction_on_validation_error(self, mc_question, forecaster_llm, parser_llm):
        """When primary parse fails with ValidationError, falls back to build_mc_prediction."""
        from pydantic import ValidationError

        from metaculus_bot.simple_types import OptionProbability

        reasoning_text = "Option B is best."
        raw_options = [
            OptionProbability(option_name="Option A", probability=0.2),
            OptionProbability(option_name="Option B", probability=0.5),
            OptionProbability(option_name="Option C", probability=0.3),
        ]
        fallback_result = _make_option_list([("Option A", 0.2), ("Option B", 0.5), ("Option C", 0.3)])

        call_count = 0

        async def mock_structure_output(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValidationError.from_exception_data(title="test", line_errors=[])
            return raw_options

        with (
            patch("metaculus_bot.forecaster_runners.multiple_choice_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
            patch("metaculus_bot.forecaster_runners.structure_output", new=mock_structure_output),
            patch("metaculus_bot.forecaster_runners.build_mc_prediction", return_value=fallback_result),
        ):
            result = await run_mc_forecast(mc_question, "research", forecaster_llm, parser_llm)

        assert result.prediction_value == fallback_result


class TestRunNumericForecast:
    @pytest.mark.asyncio
    async def test_percentile_branch_returns_prediction_and_discrete_vote(
        self, numeric_question, forecaster_llm, parser_llm
    ):
        """Numeric forecast returns (prediction, discrete_vote) tuple via the percentile branch."""
        from forecasting_tools.data_models.numeric_report import Percentile

        from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
        from metaculus_bot.numeric_format_router import RoutedNumericForecast

        reasoning_text = "OUTCOME_TYPE: DISCRETE\n\nPercentile 2.5: 50"

        percentiles = [
            Percentile(percentile=0.025, value=50),
            Percentile(percentile=0.05, value=100),
            Percentile(percentile=0.10, value=150),
            Percentile(percentile=0.20, value=200),
            Percentile(percentile=0.40, value=350),
            Percentile(percentile=0.50, value=450),
            Percentile(percentile=0.60, value=550),
            Percentile(percentile=0.80, value=700),
            Percentile(percentile=0.90, value=800),
            Percentile(percentile=0.95, value=900),
            Percentile(percentile=0.975, value=950),
        ]

        mock_prediction = MagicMock(spec=NumericDistribution)

        call_count = 0

        async def mock_structure_output(text, output_type, model, additional_instructions=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return OutcomeTypeResult(is_discrete_integer=True)
            return percentiles

        routed = RoutedNumericForecast(
            format="percentile",
            cdf_percentiles=percentiles,
            declared_percentiles=percentiles,
            mixture=None,
        )

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value=reasoning_text)),
            patch("metaculus_bot.forecaster_runners.structure_output", new=mock_structure_output),
            patch("metaculus_bot.forecaster_runners.route_numeric_output", return_value=routed),
            patch(
                "metaculus_bot.forecaster_runners.sanitize_percentiles",
                return_value=(percentiles, None),
            ),
            patch("metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=mock_prediction),
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
        ):
            prediction, discrete_vote = await run_numeric_forecast(
                numeric_question, "research", forecaster_llm, parser_llm
            )

        assert prediction.prediction_value == mock_prediction
        assert discrete_vote is True

    @pytest.mark.asyncio
    async def test_unit_mismatch_raises(self, numeric_question, forecaster_llm, parser_llm):
        """When detect_unit_mismatch returns True, raises UnitMismatchError."""
        from forecasting_tools.data_models.numeric_report import Percentile

        from metaculus_bot.exceptions import UnitMismatchError
        from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
        from metaculus_bot.numeric_format_router import RoutedNumericForecast

        percentiles = [
            Percentile(percentile=p / 100, value=v)
            for p, v in zip(
                [2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5],
                [50, 100, 150, 200, 350, 450, 550, 700, 800, 900, 950],
            )
        ]

        call_count = 0

        async def mock_structure_output(text, output_type, model, additional_instructions=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return OutcomeTypeResult(is_discrete_integer=False)
            return percentiles

        routed = RoutedNumericForecast(
            format="percentile",
            cdf_percentiles=percentiles,
            declared_percentiles=percentiles,
            mixture=None,
        )

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="reasoning")),
            patch("metaculus_bot.forecaster_runners.structure_output", new=mock_structure_output),
            patch("metaculus_bot.forecaster_runners.route_numeric_output", return_value=routed),
            patch("metaculus_bot.forecaster_runners.sanitize_percentiles", return_value=(percentiles, None)),
            patch("metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=MagicMock()),
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(True, "off by 1000x")),
        ):
            with pytest.raises(UnitMismatchError, match="off by 1000x"):
                await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm)

    @pytest.mark.asyncio
    async def test_discrete_vote_none_when_parse_fails(self, numeric_question, forecaster_llm, parser_llm):
        """When OUTCOME_TYPE parsing fails, discrete_vote is None."""
        from forecasting_tools.data_models.numeric_report import Percentile
        from pydantic import ValidationError

        from metaculus_bot.numeric_format_router import RoutedNumericForecast

        percentiles = [
            Percentile(percentile=p / 100, value=v)
            for p, v in zip(
                [2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5],
                [50, 100, 150, 200, 350, 450, 550, 700, 800, 900, 950],
            )
        ]

        call_count = 0

        async def mock_structure_output(text, output_type, model, additional_instructions=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValidationError.from_exception_data(title="test", line_errors=[])
            return percentiles

        routed = RoutedNumericForecast(
            format="percentile",
            cdf_percentiles=percentiles,
            declared_percentiles=percentiles,
            mixture=None,
        )

        with (
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="reasoning")),
            patch("metaculus_bot.forecaster_runners.structure_output", new=mock_structure_output),
            patch("metaculus_bot.forecaster_runners.route_numeric_output", return_value=routed),
            patch("metaculus_bot.forecaster_runners.sanitize_percentiles", return_value=(percentiles, None)),
            patch("metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=MagicMock()),
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=(False, "")),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
        ):
            _, discrete_vote = await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm)

        assert discrete_vote is None
