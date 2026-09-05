"""Characterize aggregation state across the framework's multi-report lifecycle.

The framework runs every research report for a question concurrently, flattens the
surviving reports' predictions, performs one final combine, and only then builds the
comment.  Every report for a question shares the aggregation pipeline's qid-keyed
state, so report failures and completion order are part of the effective contract.

These tests keep that framework sequence, the forecaster fan-out, routing,
aggregation, and comment construction real.  Only research and model calls are
replaced with deterministic local responses.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import BinaryQuestion, GeneralLlm, ReasonedPrediction

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.comment.markers import (
    FORECASTERS_USED_MARKER_RE,
    STACKER_OUTCOME_FALLBACK_LLM,
    STACKER_OUTCOME_PRIMARY,
    STACKER_OUTCOME_SKIPPED,
    STACKER_SKIP_REASON_RE,
)
from metaculus_bot.constants import THIN_PUBLISH_BINARY_FLOOR
from metaculus_bot.forecaster import TemplateForecaster

QID = 46001


def _question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will the event happen?",
        id_of_question=QID,
        page_url=f"https://www.metaculus.com/questions/{QID}/",
        background_info="Background",
        resolution_criteria="Resolves yes if the event happens.",
        fine_print="",
        open_time=datetime(2026, 1, 1),
        scheduled_resolution_time=datetime(2026, 12, 31),
        api_json={"type": "binary"},
    )


def _bot(*, fallback: bool, research_reports: int = 2) -> TemplateForecaster:
    forecasters = [
        GeneralLlm(model="test/forecaster-one", temperature=0.0),
        GeneralLlm(model="test/forecaster-two", temperature=0.0),
    ]
    stacker = GeneralLlm(model="test/stacker", temperature=0.0)
    llms: dict[str, Any] = {
        "forecasters": forecasters,
        "stacker": stacker,
        "analyzer": stacker,
        "default": forecasters[0],
        "parser": stacker,
        "researcher": stacker,
        "summarizer": stacker,
    }
    return TemplateForecaster(
        research_reports_per_question=research_reports,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        aggregation_strategy=AggregationStrategy.STACKING,
        llms=llms,
        is_benchmarking=True,
        min_forecasters_to_publish=1,
        stacking_fallback_on_failure=fallback,
        stacking_randomize_order=False,
    )


def _prediction(value: float, model: str) -> ReasonedPrediction[float]:
    return ReasonedPrediction(prediction_value=value, reasoning=f"Model: {model}\n\nReasoning")


def _interleave_report_tasks(
    original: Callable[[BinaryQuestion], Awaitable[Any]],
) -> Callable[[BinaryQuestion], Awaitable[Any]]:
    """Start both sibling tasks, then complete report A before report B routes.

    Events make the shared-state write order deterministic without timing sleeps.
    Both framework-created report coroutines are live before either enters the real
    per-report pipeline.
    """
    second_report_started = asyncio.Event()
    first_report_finished = asyncio.Event()
    report_number = 0

    async def interleaved(question: BinaryQuestion) -> Any:
        nonlocal report_number
        current_report = report_number
        report_number += 1
        if current_report == 0:
            await second_report_started.wait()
            try:
                return await original(question)
            finally:
                first_report_finished.set()
        if current_report == 1:
            second_report_started.set()
            await first_report_finished.wait()
            return await original(question)
        raise AssertionError("framework started more than two research-report tasks")

    return interleaved


def _assert_qid_state_consumed(bot: TemplateForecaster) -> None:
    assert QID not in bot._stacker_outcome
    assert QID not in bot._stacker_skip_reason
    assert QID not in bot._stack_meta_reasoning
    assert QID not in bot._contributing_forecasters
    assert QID not in bot._pipeline.expected_base_combines


class TestSharedQidReportLifecycle:
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.timeout(10)
    async def test_failed_sibling_stack_preserves_raw_single_state_through_final_combine_and_comment(self) -> None:
        """A discarded stack attempt must not erase its sibling's publish state.

        Report A yields one raw 3% member.  Report B reaches the real stack route
        with two members, but its external stacker call fails and fallback is off,
        so the framework discards that entire report.  Its final combine therefore
        sees A's raw single and must use A's surviving skip reason to apply the floor.
        """
        bot = _bot(fallback=False)
        question = _question()
        original_report_pipeline = bot._research_and_make_predictions
        research = AsyncMock(side_effect=["report A research", "report B research"])

        async def model_call(
            _question: BinaryQuestion,
            report_research: str,
            llm: GeneralLlm,
            _chart_b64: str | None = None,
        ) -> ReasonedPrediction[float]:
            if report_research == "report A research":
                if llm.model == "test/forecaster-one":
                    return _prediction(0.03, llm.model)
                raise RuntimeError("report A second forecaster failed")
            if report_research == "report B research":
                values = {"test/forecaster-one": 0.30, "test/forecaster-two": 0.60}
                return _prediction(values[llm.model], llm.model)
            raise AssertionError(f"unexpected research: {report_research}")

        assert bot._stacker_llm is not None
        stacker_invoke = AsyncMock(side_effect=RuntimeError("stacker unavailable"))
        with (
            patch.object(bot, "run_research", research),
            patch.object(bot, "_make_prediction", new=model_call),
            patch.object(bot, "_research_and_make_predictions", new=_interleave_report_tasks(original_report_pipeline)),
            patch.object(bot._stacker_llm, "invoke", new=stacker_invoke),
        ):
            report = await bot._run_individual_question(question)

        assert report.prediction == THIN_PUBLISH_BINARY_FLOOR
        assert "*Forecaster 1 (forecaster-one)*: 3.0%" in report.explanation
        assert "*Final Prediction*: 5.0%" in report.explanation
        assert STACKER_OUTCOME_SKIPPED in report.explanation
        skip_reason = STACKER_SKIP_REASON_RE.search(report.explanation)
        assert skip_reason is not None
        assert skip_reason.group(1) == "single_forecaster"
        used = FORECASTERS_USED_MARKER_RE.search(report.explanation)
        assert used is not None
        assert used.groups() == ("3", "2")
        assert bot._pipeline.counters.stacking_expected_combine_count == 1
        assert bot._pipeline.counters.stacking_unexpected_combine_count == 0
        stacker_invoke.assert_awaited_once()
        _assert_qid_state_consumed(bot)


class TestStackedOutputsThroughFrameworkLifecycle:
    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.timeout(10)
    async def test_primary_model_failure_falls_back_and_the_real_comment_records_the_producer(self) -> None:
        """The external fallback response traverses the real prompt parser and final combine."""
        bot = _bot(fallback=True, research_reports=1)
        question = _question()

        async def model_call(
            _question: BinaryQuestion,
            _research: str,
            llm: GeneralLlm,
            _chart_b64: str | None = None,
        ) -> ReasonedPrediction[float]:
            values = {"test/forecaster-one": 0.20, "test/forecaster-two": 0.80}
            return _prediction(values[llm.model], llm.model)

        fallback_response = """\
## Meta-analysis

The fallback model reconciles the disagreement.

```json
{"question_type": "binary", "posterior_prob": 0.04}
```
"""
        invoked_models: list[str] = []

        async def external_llm_call(
            llm: GeneralLlm,
            _prompt: str,
            _system_prompt: str | None = None,
            **_kwargs: Any,
        ) -> str:
            invoked_models.append(llm.model)
            if llm.model == "test/stacker":
                raise RuntimeError("primary stacker unavailable")
            return fallback_response

        with (
            patch.object(bot, "run_research", new=AsyncMock(return_value="research")),
            patch.object(bot, "_make_prediction", new=model_call),
            patch.object(GeneralLlm, "invoke", new=external_llm_call),
        ):
            report = await bot._run_individual_question(question)

        assert report.prediction == pytest.approx(0.04)
        assert "*Final Prediction*: 4.0%" in report.explanation
        assert STACKER_OUTCOME_FALLBACK_LLM in report.explanation
        assert invoked_models[0] == "test/stacker"
        assert len(invoked_models) == 2
        assert invoked_models[1] != invoked_models[0]
        assert bot._pipeline.counters.stacker_primary_failed_count == 1
        assert bot._pipeline.counters.stacker_fallback_used_count == 1
        assert bot._pipeline.counters.stacker_fallback_failed_count == 0
        assert bot._pipeline.counters.stacking_expected_combine_count == 1
        _assert_qid_state_consumed(bot)

    @pytest.mark.asyncio
    @pytest.mark.e2e
    @pytest.mark.timeout(10)
    async def test_pre_stacked_single_surviving_a_failed_report_is_not_treated_as_a_raw_single(self) -> None:
        """A sole pre-stacked collection shares len==1 with the raw-single path.

        Report A loses every model before routing.  Report B runs the real stacker
        prompt and structured-value parser, yielding a 3% pre-stacked result.  The
        framework's final combine must preserve 3% because no single-forecaster skip
        reason was written for the surviving report.
        """
        bot = _bot(fallback=True)
        question = _question()
        original_report_pipeline = bot._research_and_make_predictions
        research = AsyncMock(side_effect=["failed report research", "stacked report research"])

        async def model_call(
            _question: BinaryQuestion,
            report_research: str,
            llm: GeneralLlm,
            _chart_b64: str | None = None,
        ) -> ReasonedPrediction[float]:
            if report_research == "failed report research":
                raise RuntimeError(f"{llm.model} failed")
            if report_research == "stacked report research":
                values = {"test/forecaster-one": 0.20, "test/forecaster-two": 0.80}
                return _prediction(values[llm.model], llm.model)
            raise AssertionError(f"unexpected research: {report_research}")

        stacker_response = """\
## Meta-analysis

The low call remains plausible after comparing both models.

```json
{"question_type": "binary", "posterior_prob": 0.03}
```
"""
        assert bot._stacker_llm is not None
        stacker_invoke = AsyncMock(return_value=stacker_response)
        with (
            patch.object(bot, "run_research", research),
            patch.object(bot, "_make_prediction", new=model_call),
            patch.object(bot, "_research_and_make_predictions", new=_interleave_report_tasks(original_report_pipeline)),
            patch.object(bot._stacker_llm, "invoke", new=stacker_invoke),
        ):
            report = await bot._run_individual_question(question)

        assert report.prediction == pytest.approx(0.03)
        assert "*Final Prediction*: 3.0%" in report.explanation
        assert STACKER_OUTCOME_PRIMARY in report.explanation
        assert STACKER_SKIP_REASON_RE.search(report.explanation) is None
        used = FORECASTERS_USED_MARKER_RE.search(report.explanation)
        assert used is not None
        assert used.groups() == ("2", "2")
        assert bot._questions_failed_to_publish == 1
        assert bot._pipeline.counters.stacking_expected_combine_count == 1
        assert bot._pipeline.counters.stacking_unexpected_combine_count == 0
        stacker_invoke.assert_awaited_once()
        _assert_qid_state_consumed(bot)
