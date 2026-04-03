"""Integration tests for conditional stacking pipeline.

Tests the full conditional stacking flow: spread computation -> threshold check ->
crux extraction -> targeted research -> stacking (or skip to median aggregation).
"""

from unittest.mock import AsyncMock, Mock, patch

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
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy

# Standard 11-percentile values used throughout numeric tests
_STANDARD_PERCENTILES = [0.025, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975]


def _make_bot(
    *,
    stacking_spread_thresholds: dict[str, float] | None = None,
    stacking_fallback_on_failure: bool = True,
) -> TemplateForecaster:
    """Create a TemplateForecaster with CONDITIONAL_STACKING and mock-compatible LLMs."""
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    return TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
        llms={
            "forecasters": [test_llm, test_llm],
            "stacker": test_llm,
            "analyzer": test_llm,
            "default": test_llm,
            "parser": test_llm,
            "researcher": test_llm,
            "summarizer": test_llm,
        },
        is_benchmarking=True,
        stacking_spread_thresholds=stacking_spread_thresholds,
        stacking_fallback_on_failure=stacking_fallback_on_failure,
    )


def _make_binary_question(question_id: int = 101) -> Mock:
    question = Mock(spec=BinaryQuestion)
    question.id_of_question = question_id
    question.question_text = "Will the event happen?"
    question.background_info = "Background info"
    question.resolution_criteria = "Resolves YES if it happens"
    question.fine_print = ""
    question.page_url = "https://test.com/1"
    return question


def _make_mc_question(question_id: int = 201) -> Mock:
    question = Mock(spec=MultipleChoiceQuestion)
    question.id_of_question = question_id
    question.question_text = "Which option will occur?"
    question.options = ["A", "B", "C"]
    question.background_info = "MC background"
    question.resolution_criteria = "Based on final outcome"
    question.fine_print = ""
    question.page_url = "https://test.com/2"
    return question


class TestConditionalStackingBinaryTrigger:
    """Tests that stacking triggers correctly for binary questions with high spread."""

    @pytest.mark.asyncio
    async def test_high_spread_triggers_stacking(self):
        """High log-odds spread (>1.2) should trigger the full stacking pipeline."""
        bot = _make_bot()
        question = _make_binary_question()

        # Predictions: 0.10 and 0.85 -> log-odds spread ~3.6, well above 1.2 threshold
        pred_low = ReasonedPrediction(prediction_value=0.10, reasoning="Model: m1\n\nLikely no.")
        pred_high = ReasonedPrediction(prediction_value=0.85, reasoning="Model: m2\n\nLikely yes.")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="base research text"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
                return_value="The crux is whether X happened",
            ) as mock_crux,
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
                return_value="Targeted search found that X did happen",
            ) as mock_targeted,
            patch.object(bot, "_aggregate_predictions", return_value=0.72) as mock_aggregate,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred_low, pred_high], [], None)

            result = await bot._research_and_make_predictions(question)

            # Crux extraction was called
            mock_crux.assert_called_once()
            crux_call_args = mock_crux.call_args
            assert crux_call_args[0][0] == bot._analyzer_llm
            assert crux_call_args[0][1] == question.question_text

            # Targeted search was called with the crux
            mock_targeted.assert_called_once()
            assert mock_targeted.call_args[0][0] == "The crux is whether X happened"

            # Stacking aggregation was called with combined research
            mock_aggregate.assert_called_once()
            agg_kwargs = mock_aggregate.call_args[1]
            assert "Targeted Research" in agg_kwargs["research"]
            assert "base research text" in agg_kwargs["research"]

            # Result has exactly 1 prediction (stacked)
            assert len(result.predictions) == 1
            assert result.predictions[0].prediction_value == 0.72

            # Diagnostic counter incremented
            assert bot._conditional_stacking_triggered_count == 1
            assert bot._conditional_stacking_skipped_count == 0

    @pytest.mark.asyncio
    async def test_low_spread_skips_stacking(self):
        """Low log-odds spread (<1.2) should skip stacking and return all individual predictions."""
        bot = _make_bot()
        question = _make_binary_question()

        # Predictions: 0.45 and 0.55 -> log-odds spread ~0.40, below 1.2 threshold
        pred1 = ReasonedPrediction(prediction_value=0.45, reasoning="Model: m1\n\nSlightly no.")
        pred2 = ReasonedPrediction(prediction_value=0.55, reasoning="Model: m2\n\nSlightly yes.")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="base research text"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
            ) as mock_crux,
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
            ) as mock_targeted,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred1, pred2], [], None)

            result = await bot._research_and_make_predictions(question)

            # Stacking pipeline NOT invoked
            mock_crux.assert_not_called()
            mock_targeted.assert_not_called()

            # All individual predictions preserved
            assert len(result.predictions) == 2
            assert result.predictions[0].prediction_value == 0.45
            assert result.predictions[1].prediction_value == 0.55

            # Diagnostic counter incremented
            assert bot._conditional_stacking_skipped_count == 1
            assert bot._conditional_stacking_triggered_count == 0


class TestConditionalStackingFallbacks:
    """Tests for graceful degradation when pipeline components fail."""

    @pytest.mark.asyncio
    async def test_crux_extraction_failure_falls_through(self):
        """Crux extraction failure should still proceed to stacking with base research only."""
        bot = _make_bot()
        question = _make_binary_question()

        pred_low = ReasonedPrediction(prediction_value=0.10, reasoning="Model: m1\n\nLikely no.")
        pred_high = ReasonedPrediction(prediction_value=0.85, reasoning="Model: m2\n\nLikely yes.")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="base research text"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Analyzer LLM timed out"),
            ) as mock_crux,
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
            ) as mock_targeted,
            patch.object(bot, "_aggregate_predictions", return_value=0.65) as mock_aggregate,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred_low, pred_high], [], None)

            result = await bot._research_and_make_predictions(question)

            # Crux extraction was attempted and failed
            mock_crux.assert_called_once()
            # Targeted search NOT called (no crux to search for)
            mock_targeted.assert_not_called()

            # Stacking still proceeded with base research only
            mock_aggregate.assert_called_once()
            agg_kwargs = mock_aggregate.call_args[1]
            assert "Targeted Research" not in agg_kwargs["research"]

            # No crash, result is valid
            assert len(result.predictions) == 1
            assert result.predictions[0].prediction_value == 0.65

    @pytest.mark.asyncio
    async def test_targeted_search_failure_falls_through(self):
        """Targeted search failure should still proceed to stacking with base research only."""
        bot = _make_bot()
        question = _make_binary_question()

        pred_low = ReasonedPrediction(prediction_value=0.10, reasoning="Model: m1\n\nLikely no.")
        pred_high = ReasonedPrediction(prediction_value=0.85, reasoning="Model: m2\n\nLikely yes.")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="base research text"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
                return_value="The crux is about X",
            ),
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Search API down"),
            ) as mock_targeted,
            patch.object(bot, "_aggregate_predictions", return_value=0.60) as mock_aggregate,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred_low, pred_high], [], None)

            result = await bot._research_and_make_predictions(question)

            # Targeted search was attempted and failed
            mock_targeted.assert_called_once()

            # Stacking still proceeded with base research only
            mock_aggregate.assert_called_once()
            agg_kwargs = mock_aggregate.call_args[1]
            assert "Targeted Research" not in agg_kwargs["research"]

            # No crash, result is valid
            assert len(result.predictions) == 1
            assert result.predictions[0].prediction_value == 0.60

    @pytest.mark.asyncio
    async def test_stacking_failure_falls_back_to_median(self):
        """When stacking itself fails, fallback to MEDIAN aggregation (via MEAN fallback path)."""
        bot = _make_bot(stacking_fallback_on_failure=True)
        question = _make_binary_question()

        pred_low = ReasonedPrediction(prediction_value=0.10, reasoning="Model: m1\n\nLikely no.")
        pred_high = ReasonedPrediction(prediction_value=0.85, reasoning="Model: m2\n\nLikely yes.")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="base research text"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
                return_value="crux text",
            ),
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
                return_value="targeted results",
            ),
            patch.object(
                bot,
                "_run_stacking",
                side_effect=RuntimeError("Stacker LLM crashed"),
            ),
            patch(
                "metaculus_bot.numeric_utils.aggregate_binary_mean",
                return_value=0.475,
            ),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred_low, pred_high], [], None)

            result = await bot._research_and_make_predictions(question)

            # Result should be the MEAN-fallback value
            assert len(result.predictions) == 1
            assert result.predictions[0].prediction_value == 0.475

            # Fallback counter incremented
            assert bot._stacking_fallback_count == 1


class TestConditionalStackingMC:
    """Tests for conditional stacking with multiple choice questions."""

    @pytest.mark.asyncio
    async def test_mc_high_spread_triggers(self):
        """MC question with >0.20 max option spread should trigger stacking."""
        bot = _make_bot()
        question = _make_mc_question()

        # Predictions where option A has 0.30 spread across models (>0.20 threshold)
        pred1 = ReasonedPrediction(
            prediction_value=PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="A", probability=0.60),
                    PredictedOption(option_name="B", probability=0.25),
                    PredictedOption(option_name="C", probability=0.15),
                ]
            ),
            reasoning="Model: m1\n\nOption A is dominant.",
        )
        pred2 = ReasonedPrediction(
            prediction_value=PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="A", probability=0.30),
                    PredictedOption(option_name="B", probability=0.45),
                    PredictedOption(option_name="C", probability=0.25),
                ]
            ),
            reasoning="Model: m2\n\nOption B is stronger.",
        )

        mock_stacked_result = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.45),
                PredictedOption(option_name="B", probability=0.35),
                PredictedOption(option_name="C", probability=0.20),
            ]
        )

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="mc research"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
                return_value="Disagreement about option A vs B",
            ) as mock_crux,
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
                return_value="Search results about A vs B",
            ) as mock_targeted,
            patch.object(bot, "_aggregate_predictions", return_value=mock_stacked_result) as mock_aggregate,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred1, pred2], [], None)

            result = await bot._research_and_make_predictions(question)

            # Full pipeline triggered
            mock_crux.assert_called_once()
            mock_targeted.assert_called_once()
            mock_aggregate.assert_called_once()

            # Single stacked prediction returned
            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1


class TestConditionalStackingThresholds:
    """Tests for custom and boundary threshold behavior."""

    @pytest.mark.asyncio
    async def test_custom_thresholds(self):
        """Very high custom thresholds should prevent stacking even with moderate spread."""
        bot = _make_bot(
            stacking_spread_thresholds={"binary": 5.0, "mc": 0.90, "numeric": 0.90},
        )
        question = _make_binary_question()

        # Predictions: 0.20 and 0.80 -> log-odds spread ~2.77, below custom threshold of 5.0
        pred1 = ReasonedPrediction(prediction_value=0.20, reasoning="Model: m1\n\nUnlikely.")
        pred2 = ReasonedPrediction(prediction_value=0.80, reasoning="Model: m2\n\nLikely.")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="research"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
            ) as mock_crux,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred1, pred2], [], None)

            result = await bot._research_and_make_predictions(question)

            # Stacking NOT triggered despite moderate spread (threshold too high)
            mock_crux.assert_not_called()
            assert len(result.predictions) == 2
            assert bot._conditional_stacking_skipped_count == 1

    @pytest.mark.asyncio
    async def test_spread_just_above_threshold(self):
        """Spread barely above threshold should trigger stacking."""
        # Default binary threshold is 1.2 log-odds
        # p=0.35 -> log_odds = ln(0.35/0.65) = -0.619
        # p=0.65 -> log_odds = ln(0.65/0.35) = +0.619
        # spread = 0.619 - (-0.619) = 1.238 > 1.2 -> triggers
        bot = _make_bot()
        question = _make_binary_question()

        pred1 = ReasonedPrediction(prediction_value=0.35, reasoning="Model: m1\n\nAnalysis 1")
        pred2 = ReasonedPrediction(prediction_value=0.65, reasoning="Model: m2\n\nAnalysis 2")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="research"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
                return_value="crux",
            ) as mock_crux,
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
                return_value="search results",
            ),
            patch.object(bot, "_aggregate_predictions", return_value=0.50),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred1, pred2], [], None)

            result = await bot._research_and_make_predictions(question)

            mock_crux.assert_called_once()
            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1

    @pytest.mark.asyncio
    async def test_spread_just_below_threshold(self):
        """Spread barely below threshold should skip stacking."""
        # Default binary threshold is 1.2 log-odds
        # p=0.36 -> log_odds = ln(0.36/0.64) = -0.575
        # p=0.64 -> log_odds = ln(0.64/0.36) = +0.575
        # spread = 0.575 - (-0.575) = 1.151 < 1.2 -> skips
        bot = _make_bot()
        question = _make_binary_question()

        pred1 = ReasonedPrediction(prediction_value=0.36, reasoning="Model: m1\n\nAnalysis 1")
        pred2 = ReasonedPrediction(prediction_value=0.64, reasoning="Model: m2\n\nAnalysis 2")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="research"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
            ) as mock_crux,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred1, pred2], [], None)

            result = await bot._research_and_make_predictions(question)

            mock_crux.assert_not_called()
            assert len(result.predictions) == 2
            assert bot._conditional_stacking_skipped_count == 1


def _make_numeric_question(question_id: int = 301) -> Mock:
    question = Mock(spec=NumericQuestion)
    question.id_of_question = question_id
    question.question_text = "How many units will be sold?"
    question.background_info = "Sales question"
    question.resolution_criteria = "Resolves to actual unit count"
    question.fine_print = ""
    question.page_url = "https://test.com/3"
    question.lower_bound = 0.0
    question.upper_bound = 100.0
    question.open_lower_bound = False
    question.open_upper_bound = False
    question.zero_point = None
    question.unit_of_measure = "units"
    question.cdf_size = 201
    return question


def _make_numeric_distribution(median_value: float, spread_factor: float = 1.0) -> NumericDistribution:
    """Build a NumericDistribution with 11 percentiles centered on median_value.

    spread_factor controls how wide the percentiles are spread around the median.
    """
    offsets = [-20, -17, -14, -10, -4, 0, 4, 10, 14, 17, 20]
    percentiles = [
        Percentile(value=max(0.0, min(100.0, median_value + offset * spread_factor)), percentile=pct)
        for offset, pct in zip(offsets, _STANDARD_PERCENTILES)
    ]
    return NumericDistribution(
        declared_percentiles=percentiles,
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
    )


class TestConditionalStackingNumeric:
    """Tests for conditional stacking with numeric questions."""

    @pytest.mark.asyncio
    async def test_numeric_high_spread_triggers(self):
        """Numeric predictions with large median disagreement (spread/range > 0.15) should trigger stacking."""
        bot = _make_bot()
        question = _make_numeric_question()

        # Model 1 centered at 30, model 2 centered at 70
        # At the 50th percentile (index 5): values 30 vs 70 -> raw spread 40, normalized 40/100 = 0.40 >> 0.15
        dist_low = _make_numeric_distribution(median_value=30.0)
        dist_high = _make_numeric_distribution(median_value=70.0)

        pred_low = ReasonedPrediction(prediction_value=dist_low, reasoning="Model: m1\n\nLow estimate.")
        pred_high = ReasonedPrediction(prediction_value=dist_high, reasoning="Model: m2\n\nHigh estimate.")

        mock_stacked_dist = _make_numeric_distribution(median_value=50.0)

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="numeric research"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
                return_value="Disagreement about expected sales volume",
            ) as mock_crux,
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
                return_value="Search results about sales volume",
            ) as mock_targeted,
            patch.object(bot, "_aggregate_predictions", return_value=mock_stacked_dist) as mock_aggregate,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred_low, pred_high], [], None)

            result = await bot._research_and_make_predictions(question)

            mock_crux.assert_called_once()
            mock_targeted.assert_called_once()
            mock_aggregate.assert_called_once()

            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1
            assert bot._conditional_stacking_skipped_count == 0

    @pytest.mark.asyncio
    async def test_numeric_low_spread_skips(self):
        """Numeric predictions with similar medians (spread/range < 0.15) should skip stacking."""
        bot = _make_bot()
        question = _make_numeric_question()

        # Model 1 centered at 48, model 2 centered at 52
        # At 50th percentile (index 5): values 48 vs 52 -> raw spread 4, normalized 4/100 = 0.04 << 0.15
        dist1 = _make_numeric_distribution(median_value=48.0)
        dist2 = _make_numeric_distribution(median_value=52.0)

        pred1 = ReasonedPrediction(prediction_value=dist1, reasoning="Model: m1\n\nAround 48.")
        pred2 = ReasonedPrediction(prediction_value=dist2, reasoning="Model: m2\n\nAround 52.")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="numeric research"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
            ) as mock_crux,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred1, pred2], [], None)

            result = await bot._research_and_make_predictions(question)

            mock_crux.assert_not_called()
            assert len(result.predictions) == 2
            assert bot._conditional_stacking_skipped_count == 1
            assert bot._conditional_stacking_triggered_count == 0


class TestConditionalStackingAggregation:
    """Tests that exercise the real _aggregate_predictions path (not mocked)."""

    @pytest.mark.asyncio
    async def test_high_spread_flows_through_real_stacking(self):
        """High spread should route through _aggregate_predictions -> _run_stacking for real."""
        bot = _make_bot()
        question = _make_binary_question()

        pred_low = ReasonedPrediction(prediction_value=0.10, reasoning="Model: m1\n\nLikely no.")
        pred_high = ReasonedPrediction(prediction_value=0.85, reasoning="Model: m2\n\nLikely yes.")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="base research text"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
                return_value="The crux is X",
            ),
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
                return_value="Targeted results about X",
            ),
            # Mock _run_stacking but let _aggregate_predictions run for real
            patch.object(bot, "_run_stacking", return_value=0.65) as mock_run_stacking,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred_low, pred_high], [], None)

            result = await bot._research_and_make_predictions(question)

            # _run_stacking was called by _aggregate_predictions
            mock_run_stacking.assert_called_once()
            call_args = mock_run_stacking.call_args
            assert call_args[0][0] == question  # question is first arg
            assert "Targeted Research" in call_args[0][1]  # combined research
            assert len(call_args[0][2]) == 2  # reasoned_predictions

            assert len(result.predictions) == 1
            assert result.predictions[0].prediction_value == 0.65

    @pytest.mark.asyncio
    async def test_low_spread_uses_median_aggregation(self):
        """Low spread should skip stacking and use MEDIAN aggregation in _aggregate_predictions."""
        bot = _make_bot()
        question = _make_binary_question()

        # 0.40, 0.50, 0.60 -> log-odds spread ~0.81 (max - min), below 1.2 threshold
        test_llm = GeneralLlm(model="test-model", temperature=0.0)
        bot._forecaster_llms = [test_llm, test_llm, test_llm]

        pred1 = ReasonedPrediction(prediction_value=0.40, reasoning="Model: m1\n\nAnalysis 1")
        pred2 = ReasonedPrediction(prediction_value=0.50, reasoning="Model: m2\n\nAnalysis 2")
        pred3 = ReasonedPrediction(prediction_value=0.60, reasoning="Model: m3\n\nAnalysis 3")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="research"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
            ) as mock_crux,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred1, pred2, pred3], [], None)

            result = await bot._research_and_make_predictions(question)

            # Stacking not triggered
            mock_crux.assert_not_called()
            assert bot._conditional_stacking_skipped_count == 1

            # All 3 predictions returned for framework to aggregate
            assert len(result.predictions) == 3

            # Now call _aggregate_predictions directly with asymmetric values
            # to verify MEDIAN is used (not MEAN)
            # MEDIAN of [0.30, 0.50, 0.60] = 0.50; MEAN = 0.467
            aggregated = await bot._aggregate_predictions(
                predictions=[0.30, 0.50, 0.60],
                question=question,
            )
            assert aggregated == pytest.approx(0.50)


class TestConditionalStackingBenchmarkingFlag:
    """Tests that is_benchmarking flag is correctly passed through."""

    @pytest.mark.asyncio
    async def test_is_benchmarking_passed_to_targeted_search(self):
        """is_benchmarking=True should be forwarded to run_targeted_search."""
        bot = _make_bot()  # _make_bot already sets is_benchmarking=True
        question = _make_binary_question()

        pred_low = ReasonedPrediction(prediction_value=0.10, reasoning="Model: m1\n\nLikely no.")
        pred_high = ReasonedPrediction(prediction_value=0.85, reasoning="Model: m2\n\nLikely yes.")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="base research"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
                return_value="The crux",
            ),
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
                return_value="targeted results",
            ) as mock_targeted,
            patch.object(bot, "_aggregate_predictions", return_value=0.50),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred_low, pred_high], [], None)

            await bot._research_and_make_predictions(question)

            mock_targeted.assert_called_once()
            assert mock_targeted.call_args.kwargs["is_benchmarking"] is True


class TestConditionalStackingModelTagStripping:
    """Tests that model tags are stripped before crux extraction."""

    @pytest.mark.asyncio
    async def test_model_tags_stripped_before_crux_extraction(self):
        """Predictions with 'Model: ...' prefix should have it stripped before passing to crux extraction."""
        bot = _make_bot()
        question = _make_binary_question()

        pred_low = ReasonedPrediction(
            prediction_value=0.10, reasoning="Model: gpt-5.4\n\nThis model thinks it's unlikely."
        )
        pred_high = ReasonedPrediction(
            prediction_value=0.85, reasoning="Model: claude-4\n\nThis model thinks it's likely."
        )

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="base research"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
                return_value="The crux of disagreement",
            ) as mock_crux,
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
                return_value="targeted results",
            ),
            patch.object(bot, "_aggregate_predictions", return_value=0.50),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred_low, pred_high], [], None)

            await bot._research_and_make_predictions(question)

            mock_crux.assert_called_once()
            # Third positional arg is base_prediction_texts
            base_prediction_texts = mock_crux.call_args[0][2]
            for text in base_prediction_texts:
                assert not text.startswith("Model:"), f"Model tag not stripped: {text[:50]}"
            assert "This model thinks it's unlikely." in base_prediction_texts[0]
            assert "This model thinks it's likely." in base_prediction_texts[1]


class TestConditionalStackingFailureCounters:
    """Tests that failure counters are incremented correctly."""

    @pytest.mark.asyncio
    async def test_crux_failure_increments_counter(self):
        """Crux extraction failure should increment _conditional_stacking_crux_failures."""
        bot = _make_bot()
        question = _make_binary_question()

        pred_low = ReasonedPrediction(prediction_value=0.10, reasoning="Model: m1\n\nLikely no.")
        pred_high = ReasonedPrediction(prediction_value=0.85, reasoning="Model: m2\n\nLikely yes.")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="base research"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LLM timed out"),
            ),
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
            ) as mock_targeted,
            patch.object(bot, "_aggregate_predictions", return_value=0.50),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred_low, pred_high], [], None)

            await bot._research_and_make_predictions(question)

            assert bot._conditional_stacking_crux_failures == 1
            # Targeted search NOT called since crux failed
            mock_targeted.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_failure_increments_counter(self):
        """Targeted search failure should increment _conditional_stacking_search_failures."""
        bot = _make_bot()
        question = _make_binary_question()

        pred_low = ReasonedPrediction(prediction_value=0.10, reasoning="Model: m1\n\nLikely no.")
        pred_high = ReasonedPrediction(prediction_value=0.85, reasoning="Model: m2\n\nLikely yes.")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="base research"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
                return_value="The crux",
            ),
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Search API down"),
            ),
            patch.object(bot, "_aggregate_predictions", return_value=0.50),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([pred_low, pred_high], [], None)

            await bot._research_and_make_predictions(question)

            assert bot._conditional_stacking_search_failures == 1


class TestConditionalStackingThresholdValidation:
    """Tests for threshold key validation."""

    def test_invalid_threshold_key_raises(self):
        """Unknown threshold keys should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown stacking_spread_thresholds keys"):
            _make_bot(stacking_spread_thresholds={"binary": 1.0, "multiple_choice": 0.3, "numeric": 0.2})


class TestConditionalStackingSixModelEnsemble:
    """Tests for larger ensembles (>2 models) with conditional stacking."""

    @pytest.mark.asyncio
    async def test_six_model_ensemble_spread_computation(self):
        """Spread should be computed across all 6 models, not just the first 2."""
        test_llm = GeneralLlm(model="test-model", temperature=0.0)
        bot = _make_bot()
        # Override to 6 forecaster LLMs
        bot._forecaster_llms = [test_llm] * 6

        question = _make_binary_question()

        # 6 predictions with wide spread: [0.10, 0.20, 0.30, 0.70, 0.80, 0.90]
        # Log-odds range: log(0.90/0.10) - log(0.10/0.90) = 2*ln(9) ~= 4.39
        predictions = [
            ReasonedPrediction(prediction_value=v, reasoning=f"Model: m{i}\n\nAnalysis {i}")
            for i, v in enumerate([0.10, 0.20, 0.30, 0.70, 0.80, 0.90], start=1)
        ]

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="research"),
            patch.object(bot, "_gather_results_and_exceptions") as mock_gather,
            patch(
                "main.extract_disagreement_crux",
                new_callable=AsyncMock,
                return_value="crux",
            ) as mock_crux,
            patch(
                "main.run_targeted_search",
                new_callable=AsyncMock,
                return_value="search results",
            ),
            patch.object(bot, "_aggregate_predictions", return_value=0.50),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = (predictions, [], None)

            result = await bot._research_and_make_predictions(question)

            # Stacking must trigger: spread ~4.39 >> 1.2 threshold
            mock_crux.assert_called_once()
            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1
