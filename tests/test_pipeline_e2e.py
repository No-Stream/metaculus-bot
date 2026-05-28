"""End-to-end pipeline tests for binary question type.

Exercises CONDITIONAL_STACKING spread-check, stacker fallback chain,
and the min-forecasters guard through the full _research_and_make_predictions flow.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest
from forecasting_tools import ReasonedPrediction

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from tests.pipeline_test_helpers import make_e2e_bot, make_real_binary_question


def _make_bot(
    n_forecasters: int = 3,
    strategy: AggregationStrategy = AggregationStrategy.CONDITIONAL_STACKING,
    **overrides,
) -> TemplateForecaster:
    return make_e2e_bot(strategy, n_forecasters=n_forecasters, **overrides)


class TestBinaryLowSpreadSkipsStacking:
    """Low binary spread (< 0.15) skips stacking and returns raw predictions."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_binary_low_spread_skips_stacking(self):
        bot = _make_bot(n_forecasters=3)
        question = make_real_binary_question(qid=6001)

        predictions = [
            ReasonedPrediction(prediction_value=0.30, reasoning="Model: m1\n\nLow spread binary"),
            ReasonedPrediction(prediction_value=0.32, reasoning="Model: m2\n\nLow spread binary"),
            ReasonedPrediction(prediction_value=0.31, reasoning="Model: m3\n\nLow spread binary"),
        ]

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="Canned research"),
            patch.object(bot, "_gather_predictions_with_wall_clock") as mock_gather,
            patch.object(
                bot,
                "_forecaster_with_soft_deadline",
                new=AsyncMock(return_value=predictions[0]),
            ),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = (predictions, [], None)

            result = await bot._research_and_make_predictions(question)

        assert len(result.predictions) == 3
        qid = question.id_of_question
        assert bot._stacker_outcome[qid] == "skipped"
        assert bot._conditional_stacking_skipped_count == 1


class TestBinaryHighSpreadTriggersStacking:
    """High binary spread (>= 0.15) triggers stacking via crux + targeted search."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_binary_high_spread_triggers_stacking(self):
        bot = _make_bot(n_forecasters=3)
        question = make_real_binary_question(qid=6002)

        predictions = [
            ReasonedPrediction(prediction_value=0.20, reasoning="Model: m1\n\nLow binary"),
            ReasonedPrediction(prediction_value=0.80, reasoning="Model: m2\n\nHigh binary"),
            ReasonedPrediction(prediction_value=0.50, reasoning="Model: m3\n\nMid binary"),
        ]

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="Canned research"),
            patch.object(bot, "_gather_predictions_with_wall_clock") as mock_gather,
            patch.object(
                bot,
                "_forecaster_with_soft_deadline",
                new=AsyncMock(return_value=predictions[0]),
            ),
            patch("main.extract_disagreement_crux", new_callable=AsyncMock, return_value="Crux text") as mock_crux,
            patch("main.run_targeted_search", new_callable=AsyncMock, return_value="Targeted results") as mock_search,
            patch.object(bot, "_run_stacking", return_value=0.45) as mock_stacking,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = (predictions, [], None)

            result = await bot._research_and_make_predictions(question)

        mock_crux.assert_called_once()
        mock_search.assert_called_once()
        mock_stacking.assert_called_once()

        assert len(result.predictions) == 1
        assert bot._conditional_stacking_triggered_count == 1


class TestBinaryStackerPrimaryFailsFallbackSucceeds:
    """Primary stacker fails with timeout; fallback stacker succeeds."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_binary_stacker_primary_fails_fallback_succeeds(self):
        bot = _make_bot(n_forecasters=3, strategy=AggregationStrategy.STACKING)
        question = make_real_binary_question(qid=6003)

        predictions = [
            ReasonedPrediction(prediction_value=0.30, reasoning="Model: m1\n\nBinary reasoning"),
            ReasonedPrediction(prediction_value=0.40, reasoning="Model: m2\n\nBinary reasoning"),
            ReasonedPrediction(prediction_value=0.35, reasoning="Model: m3\n\nBinary reasoning"),
        ]

        call_count = 0

        async def stacking_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError()
            return 0.35

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="Canned research"),
            patch.object(bot, "_gather_predictions_with_wall_clock") as mock_gather,
            patch.object(
                bot,
                "_forecaster_with_soft_deadline",
                new=AsyncMock(return_value=predictions[0]),
            ),
            patch.object(bot, "_run_stacking", side_effect=stacking_side_effect),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = (predictions, [], None)

            await bot._research_and_make_predictions(question)

        qid = question.id_of_question
        assert bot._stacker_outcome[qid] == "fallback_llm"
        assert bot._stacker_primary_failed_count == 1


class TestBinaryBothStackersFailMedianFallback:
    """Both primary and fallback stackers fail; falls back to MEDIAN of base predictions."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_binary_both_stackers_fail_median_fallback(self):
        bot = _make_bot(n_forecasters=3, strategy=AggregationStrategy.STACKING)
        question = make_real_binary_question(qid=6004)

        predictions = [
            ReasonedPrediction(prediction_value=0.30, reasoning="Model: m1\n\nBinary reasoning"),
            ReasonedPrediction(prediction_value=0.50, reasoning="Model: m2\n\nBinary reasoning"),
            ReasonedPrediction(prediction_value=0.40, reasoning="Model: m3\n\nBinary reasoning"),
        ]

        async def always_fail(*args, **kwargs):
            raise RuntimeError("stacker failure")

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="Canned research"),
            patch.object(bot, "_gather_predictions_with_wall_clock") as mock_gather,
            patch.object(
                bot,
                "_forecaster_with_soft_deadline",
                new=AsyncMock(return_value=predictions[0]),
            ),
            patch.object(bot, "_run_stacking", side_effect=always_fail),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = (predictions, [], None)

            result = await bot._research_and_make_predictions(question)

        qid = question.id_of_question
        assert bot._stacker_outcome[qid] == "fallback_median"
        assert bot._stacker_fallback_failed_count == 1

        final_prediction = result.predictions[0].prediction_value
        expected_median = sorted([0.30, 0.50, 0.40])[1]
        assert abs(final_prediction - expected_median) < 0.02


class TestMinForecasterGuardRaises:
    """Min-forecasters guard raises RuntimeError when too few forecasters succeed."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_min_forecasters_guard_raises(self):
        bot = _make_bot(n_forecasters=3, min_forecasters_to_publish=2)
        question = make_real_binary_question(qid=6005)

        single_prediction = [
            ReasonedPrediction(prediction_value=0.50, reasoning="Model: m1\n\nOnly survivor"),
        ]

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="Canned research"),
            patch.object(bot, "_gather_predictions_with_wall_clock") as mock_gather,
            patch.object(
                bot,
                "_forecaster_with_soft_deadline",
                new=AsyncMock(return_value=single_prediction[0]),
            ),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = (single_prediction, ["error1", "error2"], None)

            with pytest.raises(RuntimeError, match="Only 1/3 forecasters succeeded"):
                await bot._research_and_make_predictions(question)

        assert bot._questions_failed_to_publish == 1
