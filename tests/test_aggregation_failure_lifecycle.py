"""Regression tests for aggregation failures at lifecycle boundaries."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import BinaryQuestion, ForecastBot, GeneralLlm, ReasonedPrediction
from forecasting_tools.data_models.data_organizer import PredictionTypes

import metaculus_bot.aggregation_pipeline as aggregation_pipeline_module
from metaculus_bot.aggregation_pipeline import AggregationPipeline
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.forecaster import TemplateForecaster
from metaculus_bot.llm_configs import STACKER_FALLBACK_LLM

_QID = 48001


def _question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will the event happen?",
        id_of_question=_QID,
        page_url=f"https://www.metaculus.com/questions/{_QID}/",
        background_info="Background",
        resolution_criteria="Resolves yes if the event happens.",
        fine_print="",
        open_time=datetime(2026, 1, 1),
        scheduled_resolution_time=datetime(2026, 12, 31),
        api_json={"type": "binary"},
    )


def _bot() -> TemplateForecaster:
    forecasters = [
        GeneralLlm(model="test/forecaster-one", temperature=0.0),
        GeneralLlm(model="test/forecaster-two", temperature=0.0),
    ]
    support_llm = GeneralLlm(model="test/stacker", temperature=0.0)
    llms: dict[str, Any] = {
        "forecasters": forecasters,
        "stacker": support_llm,
        "analyzer": support_llm,
        "default": forecasters[0],
        "parser": support_llm,
        "researcher": support_llm,
        "summarizer": support_llm,
    }
    return TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        aggregation_strategy=AggregationStrategy.STACKING,
        llms=llms,
        is_benchmarking=True,
        min_forecasters_to_publish=1,
        stacking_fallback_on_failure=True,
        stacking_randomize_order=False,
    )


def _prediction(value: float, model: str) -> ReasonedPrediction[float]:
    return ReasonedPrediction(prediction_value=value, reasoning=f"Model: {model}\n\nReasoning")


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(10)
async def test_parent_comment_failure_preserves_comment_owned_state_after_final_combine() -> None:
    """The parent comment builder can fail after aggregation has fully finished."""
    bot = _bot()
    question = _question()

    async def model_call(
        _question: BinaryQuestion,
        _research: str,
        llm: GeneralLlm,
        _chart_b64: str | None = None,
    ) -> ReasonedPrediction[float]:
        values = {"test/forecaster-one": 0.20, "test/forecaster-two": 0.80}
        return _prediction(values[llm.model], llm.model)

    stacker_response = """\
## Meta-analysis

The stacker reconciles the disagreement.

```json
{"question_type": "binary", "posterior_prob": 0.65}
```
"""
    original_aggregate = bot._aggregate_predictions
    final_combined_predictions: list[PredictionTypes] = []
    seeded_skip_reason = "comment_builder_failure_fixture"

    async def aggregate_then_seed_skip_state(
        predictions: list[PredictionTypes],
        called_question: BinaryQuestion,
        research: str | None = None,
        reasoned_predictions: list[ReasonedPrediction[PredictionTypes]] | None = None,
        aggregated_tool_output: str | None = None,
    ) -> PredictionTypes:
        result = await original_aggregate(
            predictions,
            called_question,
            research=research,
            reasoned_predictions=reasoned_predictions,
            aggregated_tool_output=aggregated_tool_output,
        )
        if research is None and reasoned_predictions is None:
            final_combined_predictions.append(result)
            bot._pipeline.skip_reasons[_QID] = seeded_skip_reason
        return result

    assert bot._pipeline.stacker_llm is not None
    stacker_invoke = AsyncMock(return_value=stacker_response)
    parent_comment_failure = RuntimeError("parent comment builder failed")
    with (
        patch.object(bot, "run_research", new=AsyncMock(return_value="research")),
        patch.object(bot, "_make_prediction", new=model_call),
        patch.object(bot, "_aggregate_predictions", new=aggregate_then_seed_skip_state),
        patch.object(bot._pipeline.stacker_llm, "invoke", new=stacker_invoke),
        patch.object(ForecastBot, "_create_unified_explanation", side_effect=parent_comment_failure) as parent_builder,
        pytest.raises(RuntimeError, match="parent comment builder failed") as raised,
    ):
        await bot._run_individual_question(question)

    assert raised.value is parent_comment_failure
    assert final_combined_predictions == [pytest.approx(0.65)]
    assert bot._pipeline.outcomes == {_QID: "primary"}
    assert bot._pipeline.skip_reasons == {_QID: seeded_skip_reason}
    assert bot._contributing_forecasters == {_QID: 2}
    assert _QID not in bot._pipeline.meta_reasoning
    assert _QID not in bot._pipeline.expected_base_combines
    assert bot._pipeline.counters.stacking_expected_combine_count == 1
    assert bot._pipeline.counters.stacking_unexpected_combine_count == 0
    stacker_invoke.assert_awaited_once()
    parent_builder.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.timeout(2)
async def test_primary_wait_for_timeout_cancels_attempt_before_fallback() -> None:
    """The outer deadline cancels a stuck primary before starting fallback."""
    bot = _bot()
    question = _question()
    pipeline = AggregationPipeline(
        strategy=AggregationStrategy.STACKING,
        stacker_llm=GeneralLlm(model="test/primary-stacker", temperature=0.0),
        parser_llm=GeneralLlm(model="test/parser", temperature=0.0),
        stacking_fallback_on_failure=True,
        stacking_randomize_order=False,
        stacking_spread_thresholds={"binary": 0.15, "mc": 0.20, "numeric": 0.15},
    )
    bot._pipeline = pipeline
    reasoned_predictions: list[ReasonedPrediction[PredictionTypes]] = [
        ReasonedPrediction(prediction_value=0.20, reasoning="Model: one\n\nLow"),
        ReasonedPrediction(prediction_value=0.80, reasoning="Model: two\n\nHigh"),
    ]
    primary_started = asyncio.Event()
    primary_cancelled = asyncio.Event()
    fallback_started = asyncio.Event()
    attempt_arguments: list[tuple[GeneralLlm | None, float]] = []

    async def controlled_stacking_attempt(
        _question: BinaryQuestion,
        _research: str,
        _reasoned_predictions: list[ReasonedPrediction[PredictionTypes]],
        *,
        stacker_llm_override: GeneralLlm | None = None,
        aggregated_tool_output: str | None = None,
        stacker_wall_timeout: float,
    ) -> PredictionTypes:
        del aggregated_tool_output
        attempt_arguments.append((stacker_llm_override, stacker_wall_timeout))
        if stacker_llm_override is None:
            primary_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                primary_cancelled.set()
            raise AssertionError("unreachable")
        assert primary_started.is_set()
        assert primary_cancelled.is_set()
        fallback_started.set()
        return 0.55

    primary_deadline = 0.01
    fallback_deadline = 0.50
    with (
        patch.object(aggregation_pipeline_module, "STACKER_SOFT_DEADLINE", primary_deadline),
        patch.object(aggregation_pipeline_module, "STACKER_FALLBACK_SOFT_DEADLINE", fallback_deadline),
        patch.object(pipeline, "run_stacking", new=controlled_stacking_attempt),
    ):
        result = await bot._aggregate_predictions(
            [0.20, 0.80],
            question,
            research="research",
            reasoned_predictions=reasoned_predictions,
        )

    assert result == pytest.approx(0.55)
    assert primary_started.is_set()
    assert primary_cancelled.is_set()
    assert fallback_started.is_set()
    assert attempt_arguments == [(None, primary_deadline), (STACKER_FALLBACK_LLM, fallback_deadline)]
    assert pipeline.outcomes == {_QID: "fallback_llm"}
    assert pipeline.counters.stacker_primary_failed_count == 1
    assert pipeline.counters.stacker_fallback_used_count == 1
    assert pipeline.counters.stacker_fallback_failed_count == 0
