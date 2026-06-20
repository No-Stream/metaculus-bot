"""Tests for the extracted aggregation pipeline module.

Exercises AggregationPipeline's three main paths:
1. Base-combine re-entry (stacking already done, parent class calls aggregate)
2. Stacking fallback chain (primary -> fallback LLM -> median)
3. Simple MEAN/MEDIAN aggregation
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

from metaculus_bot.aggregation_pipeline import AggregationCounters, AggregationPipeline
from metaculus_bot.aggregation_strategies import AggregationStrategy
from tests.conftest import make_mock_numeric_question


def _make_binary_question(qid: int = 100) -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will X happen?",
        id_of_question=qid,
        page_url="https://metaculus.com/questions/100",
        api_json={"type": "binary"},
    )


def _make_mc_question(qid: int = 200) -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which will happen?",
        id_of_question=qid,
        page_url="https://metaculus.com/questions/200",
        options=["A", "B", "C"],
        api_json={"type": "multiple_choice"},
    )


def _make_pipeline(
    strategy: AggregationStrategy = AggregationStrategy.CONDITIONAL_STACKING,
    stacking_fallback_on_failure: bool = True,
) -> AggregationPipeline:
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    return AggregationPipeline(
        strategy=strategy,
        stacker_llm=test_llm,
        parser_llm=test_llm,
        stacking_fallback_on_failure=stacking_fallback_on_failure,
        stacking_randomize_order=False,
        stacking_spread_thresholds={"binary": 0.15, "mc": 0.20, "numeric": 0.15},
        discrete_integer_votes=defaultdict(list),
    )


class TestAggregationCounters:
    def test_initial_values_are_zero(self):
        counters = AggregationCounters()
        assert counters.stacking_expected_combine_count == 0
        assert counters.stacking_unexpected_combine_count == 0
        assert counters.stacking_fallback_count == 0
        assert counters.stacker_primary_failed_count == 0
        assert counters.stacker_fallback_used_count == 0
        assert counters.stacker_fallback_failed_count == 0


class TestBaseCombineReentry:
    """Test the base-combine path: reasoned_predictions=None and research=None."""

    @pytest.mark.asyncio
    async def test_single_prediction_returns_as_is(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=101)
        pipeline.register_expected_base_combine(question)

        result = await pipeline.aggregate(
            predictions=[0.42],
            question=question,
            research=None,
            reasoned_predictions=None,
        )

        assert result == 0.42
        assert pipeline.counters.stacking_expected_combine_count == 1

    @pytest.mark.asyncio
    async def test_unexpected_combine_increments_counter(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=102)
        # Do NOT register expected combine

        result = await pipeline.aggregate(
            predictions=[0.55],
            question=question,
            research=None,
            reasoned_predictions=None,
        )

        assert result == 0.55
        assert pipeline.counters.stacking_unexpected_combine_count == 1

    @pytest.mark.asyncio
    async def test_multiple_binary_uses_median_for_conditional_stacking(self):
        pipeline = _make_pipeline(strategy=AggregationStrategy.CONDITIONAL_STACKING)
        question = _make_binary_question(qid=103)
        pipeline.register_expected_base_combine(question)

        result = await pipeline.aggregate(
            predictions=[0.30, 0.50, 0.70],
            question=question,
            research=None,
            reasoned_predictions=None,
        )

        # Median of [0.30, 0.50, 0.70] = 0.50
        assert result == 0.50

    @pytest.mark.asyncio
    async def test_multiple_binary_uses_mean_for_stacking(self):
        pipeline = _make_pipeline(strategy=AggregationStrategy.STACKING)
        question = _make_binary_question(qid=104)
        pipeline.register_expected_base_combine(question)

        result = await pipeline.aggregate(
            predictions=[0.30, 0.50, 0.70],
            question=question,
            research=None,
            reasoned_predictions=None,
        )

        # Mean of [0.30, 0.50, 0.70] = 0.50 (coincidentally same as median here)
        assert abs(cast(float, result) - 0.50) < 0.01

    @pytest.mark.asyncio
    async def test_multiple_mc_uses_median_for_conditional_stacking(self):
        pipeline = _make_pipeline(strategy=AggregationStrategy.CONDITIONAL_STACKING)
        question = _make_mc_question(qid=105)
        pipeline.register_expected_base_combine(question)

        pol1 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.5),
                PredictedOption(option_name="B", probability=0.3),
                PredictedOption(option_name="C", probability=0.2),
            ]
        )
        pol2 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.6),
                PredictedOption(option_name="B", probability=0.2),
                PredictedOption(option_name="C", probability=0.2),
            ]
        )
        pol3 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.7),
                PredictedOption(option_name="B", probability=0.1),
                PredictedOption(option_name="C", probability=0.2),
            ]
        )

        result = await pipeline.aggregate(
            predictions=[pol1, pol2, pol3],
            question=question,
            research=None,
            reasoned_predictions=None,
        )

        assert isinstance(result, PredictedOptionList)
        probs = {o.option_name: o.probability for o in result.predicted_options}
        # Median of A: [0.5, 0.6, 0.7] = 0.6
        # Median of B: [0.3, 0.2, 0.1] = 0.2
        # Median of C: [0.2, 0.2, 0.2] = 0.2
        # After renormalization: A=0.6, B=0.2, C=0.2
        assert abs(probs["A"] - 0.6) < 0.01
        assert abs(probs["B"] - 0.2) < 0.01


class TestStackingFallbackChain:
    """Test the primary -> fallback -> median chain."""

    @pytest.mark.asyncio
    async def test_primary_success_sets_outcome(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=201)
        predictions: list[PredictionTypes] = [0.20, 0.80, 0.50]
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.20, reasoning="Model: m1\n\nLow"),
            ReasonedPrediction(prediction_value=0.80, reasoning="Model: m2\n\nHigh"),
            ReasonedPrediction(prediction_value=0.50, reasoning="Model: m3\n\nMid"),
        ]

        with patch.object(pipeline, "run_stacking", new=AsyncMock(return_value=0.65)):
            result = await pipeline.aggregate(
                predictions=predictions,
                question=question,
                research="test research",
                reasoned_predictions=reasoned,
            )

        assert abs(cast(float, result) - 0.65) < 0.01
        assert pipeline.outcomes[201] == "primary"

    @pytest.mark.asyncio
    async def test_primary_failure_invokes_fallback(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=202)
        predictions: list[PredictionTypes] = [0.20, 0.80]
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.20, reasoning="Model: m1\n\nLow"),
            ReasonedPrediction(prediction_value=0.80, reasoning="Model: m2\n\nHigh"),
        ]

        call_count = 0

        async def mock_run_stacking(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError("primary timed out")
            return 0.55

        with patch.object(pipeline, "run_stacking", side_effect=mock_run_stacking):
            result = await pipeline.aggregate(
                predictions=predictions,
                question=question,
                research="test research",
                reasoned_predictions=reasoned,
            )

        assert abs(cast(float, result) - 0.55) < 0.01
        assert pipeline.outcomes[202] == "fallback_llm"
        assert pipeline.counters.stacker_primary_failed_count == 1
        assert pipeline.counters.stacker_fallback_used_count == 1

    @pytest.mark.asyncio
    async def test_both_fail_uses_median(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=203)
        predictions: list[PredictionTypes] = [0.20, 0.80, 0.50]
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.20, reasoning="Model: m1\n\nLow"),
            ReasonedPrediction(prediction_value=0.80, reasoning="Model: m2\n\nHigh"),
            ReasonedPrediction(prediction_value=0.50, reasoning="Model: m3\n\nMid"),
        ]

        with patch.object(pipeline, "run_stacking", side_effect=asyncio.TimeoutError("timed out")):
            result = await pipeline.aggregate(
                predictions=predictions,
                question=question,
                research="test research",
                reasoned_predictions=reasoned,
            )

        # Median of [0.20, 0.50, 0.80] = 0.50
        assert abs(cast(float, result) - 0.50) < 0.01
        assert pipeline.outcomes[203] == "fallback_median"
        assert pipeline.counters.stacker_primary_failed_count == 1
        assert pipeline.counters.stacker_fallback_used_count == 1
        assert pipeline.counters.stacker_fallback_failed_count == 1

    @pytest.mark.asyncio
    async def test_fallback_disabled_raises(self):
        pipeline = _make_pipeline(stacking_fallback_on_failure=False)
        question = _make_binary_question(qid=204)
        predictions: list[PredictionTypes] = [0.20, 0.80]
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.20, reasoning="Model: m1\n\nLow"),
            ReasonedPrediction(prediction_value=0.80, reasoning="Model: m2\n\nHigh"),
        ]

        with (
            patch.object(pipeline, "run_stacking", side_effect=RuntimeError("boom")),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await pipeline.aggregate(
                predictions=predictions,
                question=question,
                research="test research",
                reasoned_predictions=reasoned,
            )


class TestSimpleAggregation:
    """Test MEAN/MEDIAN strategies (no stacking)."""

    @pytest.mark.asyncio
    async def test_mean_binary(self):
        pipeline = _make_pipeline(strategy=AggregationStrategy.MEAN)
        question = _make_binary_question(qid=301)

        result = await pipeline.aggregate(
            predictions=[0.20, 0.40, 0.60],
            question=question,
        )

        # Mean of [0.20, 0.40, 0.60] = 0.40
        assert abs(cast(float, result) - 0.40) < 0.01

    @pytest.mark.asyncio
    async def test_median_binary(self):
        pipeline = _make_pipeline(strategy=AggregationStrategy.MEDIAN)
        question = _make_binary_question(qid=302)

        result = await pipeline.aggregate(
            predictions=[0.20, 0.40, 0.60],
            question=question,
        )

        # Median of [0.20, 0.40, 0.60] = 0.40
        assert abs(cast(float, result) - 0.40) < 0.01

    @pytest.mark.asyncio
    async def test_mean_mc(self):
        pipeline = _make_pipeline(strategy=AggregationStrategy.MEAN)
        question = _make_mc_question(qid=303)

        pol1 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.4),
                PredictedOption(option_name="B", probability=0.4),
                PredictedOption(option_name="C", probability=0.2),
            ]
        )
        pol2 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.6),
                PredictedOption(option_name="B", probability=0.2),
                PredictedOption(option_name="C", probability=0.2),
            ]
        )

        result = await pipeline.aggregate(predictions=[pol1, pol2], question=question)

        assert isinstance(result, PredictedOptionList)
        probs = {o.option_name: o.probability for o in result.predicted_options}
        # Mean of A: [0.4, 0.6] = 0.5, B: [0.4, 0.2] = 0.3, C: [0.2, 0.2] = 0.2
        assert abs(probs["A"] - 0.5) < 0.01
        assert abs(probs["B"] - 0.3) < 0.01
        assert abs(probs["C"] - 0.2) < 0.01


class TestRunStacking:
    """Test the run_stacking dispatch per question type."""

    @pytest.mark.asyncio
    async def test_binary_dispatches_to_stacking_module(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=401)
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.30, reasoning="Model: m1\n\nReasons"),
            ReasonedPrediction(prediction_value=0.70, reasoning="Model: m2\n\nReasons"),
        ]

        with patch("metaculus_bot.aggregation_pipeline.stacking.run_stacking_binary") as mock_stack:
            mock_stack.return_value = (0.55, "Meta analysis text")
            result = await pipeline.run_stacking(question, "research", reasoned)

        assert result == 0.55
        assert pipeline.meta_reasoning[401] == "Meta analysis text"

    @pytest.mark.asyncio
    async def test_mc_dispatches_to_stacking_module(self):
        pipeline = _make_pipeline()
        question = _make_mc_question(qid=402)
        reasoned: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(
                prediction_value=PredictedOptionList(
                    predicted_options=[
                        PredictedOption(option_name="A", probability=0.5),
                        PredictedOption(option_name="B", probability=0.3),
                        PredictedOption(option_name="C", probability=0.2),
                    ]
                ),
                reasoning="Model: m1\n\nReasons",
            ),
        ]

        expected_pol = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.6),
                PredictedOption(option_name="B", probability=0.25),
                PredictedOption(option_name="C", probability=0.15),
            ]
        )

        with patch("metaculus_bot.aggregation_pipeline.stacking.run_stacking_mc") as mock_stack:
            mock_stack.return_value = (expected_pol, "MC meta text")
            result = await pipeline.run_stacking(question, "research", reasoned)

        assert result == expected_pol
        assert pipeline.meta_reasoning[402] == "MC meta text"


class TestThresholdLookup:
    def test_binary_threshold(self):
        pipeline = _make_pipeline()
        q = _make_binary_question()
        assert pipeline.get_threshold_for_question(q) == 0.15

    def test_mc_threshold(self):
        pipeline = _make_pipeline()
        q = _make_mc_question()
        assert pipeline.get_threshold_for_question(q) == 0.20

    def test_numeric_threshold(self):
        pipeline = _make_pipeline()
        q = make_mock_numeric_question(id_of_question=500)
        assert pipeline.get_threshold_for_question(q) == 0.15


class TestRegisterExpectedBaseCombine:
    def test_registers_and_discards(self):
        pipeline = _make_pipeline()
        question = _make_binary_question(qid=600)

        pipeline.register_expected_base_combine(question)
        assert 600 in pipeline.expected_base_combines

    def test_requires_question_id(self):
        pipeline = _make_pipeline()
        question = BinaryQuestion(
            question_text="Test",
            id_of_question=None,
            page_url="http://example.com",
            api_json={"type": "binary"},
        )

        with pytest.raises(AssertionError):
            pipeline.register_expected_base_combine(question)
