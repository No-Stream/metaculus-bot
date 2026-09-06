"""End-to-end tests for the single-survivor binary publish floor.

The floor (``apply_thin_publish_floor``, wired in ``AggregationPipeline._base_combine``)
depends on an ORDERING the code does not state in one place: ``route_after_forecasts``
writes ``skip_reasons[qid] = "single_forecaster"`` during
``_research_and_make_predictions``; the framework's ``_run_individual_question`` then
calls ``_aggregate_predictions`` (the base-combine re-entry that reads the reason and
applies the floor); and ``_create_unified_explanation`` pops the reason last, to publish
it as the STACKER_SKIP_REASON comment marker. These tests drive the REAL
``_run_individual_question`` on a real ``BinaryQuestion`` with only the LLM-calling
seams stubbed, so a reordering of any of those three steps fails here rather than in
prod.

They also pin the two properties the plan singles out: the per-model record stays RAW
(the comment's summary bullet keeps the member's declared value while the published
prediction is floored), and a fired stacker's single pre-stacked output — which shares
the same ``len == 1`` branch — is never floored.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
)

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.comment.markers import (
    FORECASTERS_USED_MARKER_RE,
    STACKER_OUTCOME_PRIMARY,
    STACKER_OUTCOME_SKIPPED,
    STACKER_SKIP_REASON_RE,
)
from metaculus_bot.constants import THIN_PUBLISH_BINARY_CEIL, THIN_PUBLISH_BINARY_FLOOR
from metaculus_bot.forecaster import TemplateForecaster

QID = 44874  # the question the floor is priced on: a lone 0.03 on a YES resolution
GEMINI = "Model: openrouter/google/gemini-3.1-pro-preview\n\nlow"
SOL = "Model: openrouter/openai/gpt-5.6-sol\n\nmid"
OPUS = "Model: openrouter/anthropic/claude-opus-4.8\n\nmid"


def _make_bot(strategy: AggregationStrategy, n_forecasters: int = 3) -> TemplateForecaster:
    """A bot whose research and forecaster seams the test stubs; everything else is real."""
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    llms: dict[str, Any] = {
        "forecasters": [test_llm] * n_forecasters,
        "stacker": test_llm,
        "analyzer": test_llm,
        "default": test_llm,
        "parser": test_llm,
        "researcher": test_llm,
        "summarizer": test_llm,
    }
    bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        aggregation_strategy=strategy,
        llms=llms,
        is_benchmarking=True,
        min_forecasters_to_publish=1,
    )
    bot.run_research = AsyncMock(return_value="stubbed research")
    return bot


def _binary_question(qid: int = QID) -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will it happen?",
        id_of_question=qid,
        page_url=f"https://www.metaculus.com/questions/{qid}/",
        background_info="bg",
        resolution_criteria="rc",
        fine_print="",
        api_json={"type": "binary"},
    )


def _mc_question(qid: int = 45244) -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which option occurs?",
        id_of_question=qid,
        page_url=f"https://www.metaculus.com/questions/{qid}/",
        options=["A", "B"],
        background_info="bg",
        resolution_criteria="rc",
        fine_print="",
        api_json={"type": "multiple_choice"},
    )


def _serve_survivors(bot: TemplateForecaster, survivors: list[ReasonedPrediction]) -> None:
    """Hand the REAL fan-out ``survivors`` for its first calls and a failure for the rest.

    The wall-clock gather is genuine, so the drop counter and the min-forecasters
    guard see exactly what a thinned production run sees.
    """
    served = iter(survivors)

    def forecaster(*_args: Any, **_kwargs: Any) -> ReasonedPrediction:
        try:
            return next(served)
        except StopIteration:
            raise ValueError("simulated forecaster failure") from None

    # AsyncMock wraps a plain side_effect in a coroutine, so each fan-out call awaits
    # either the next survivor or the simulated failure.
    bot._forecaster_with_soft_deadline = cast(Any, AsyncMock(side_effect=forecaster))


def _marker_lines(caplog: pytest.LogCaptureFixture, prefix: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(prefix)]


class TestSingleSurvivorPublishThroughTheRealSequence:
    @pytest.mark.asyncio
    async def test_lone_low_call_publishes_at_the_floor_and_the_comment_keeps_the_raw_bullet(
        self, caplog: pytest.LogCaptureFixture
    ):
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        question = _binary_question()
        _serve_survivors(bot, [ReasonedPrediction(prediction_value=0.03, reasoning=GEMINI)])

        with caplog.at_level("INFO"):
            report = await bot._run_individual_question(question)

        # Published value: floored. Per-model record: raw. Both in the same artifact.
        assert report.prediction == THIN_PUBLISH_BINARY_FLOOR
        assert bot._forecasters_dropped_count == 2
        assert "*Forecaster 1 (gemini-3.1-pro-preview)*: 3.0%" in report.explanation
        assert "*Final Prediction*: 5.0%" in report.explanation

        # The skip reason survived the aggregation step and reached the comment marker
        # (it is popped only at comment-building time), alongside the outcome and the
        # ensemble-size disclosure of the same degraded publish.
        assert STACKER_OUTCOME_SKIPPED in report.explanation
        skip_marker = STACKER_SKIP_REASON_RE.search(report.explanation)
        assert skip_marker is not None
        assert skip_marker.group(1) == "single_forecaster"
        used = FORECASTERS_USED_MARKER_RE.search(report.explanation)
        assert used is not None
        assert used.groups() == ("1", "3")
        assert QID not in bot._pipeline.skip_reasons, "the comment builder pops the reason once published"

        # The WARN names raw and clamped, and is logged AFTER the survivor count whose
        # k=1 it depends on, so one pass over the log reads both.
        floor_lines = _marker_lines(caplog, "THIN_PUBLISH_FLOOR:")
        assert floor_lines == [f"THIN_PUBLISH_FLOOR: question={QID} raw=0.0300 clamped=0.0500 survivors=1"]
        messages = [r.getMessage() for r in caplog.records]
        assert messages.index(floor_lines[0]) > next(
            i for i, m in enumerate(messages) if m.startswith("FORECASTERS_SURVIVED:")
        )
        assert next(r.levelname for r in caplog.records if r.getMessage() == floor_lines[0]) == "WARNING"

    @pytest.mark.asyncio
    async def test_lone_high_call_publishes_at_the_ceiling(self, caplog: pytest.LogCaptureFixture):
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        question = _binary_question()
        _serve_survivors(bot, [ReasonedPrediction(prediction_value=0.97, reasoning=SOL)])

        with caplog.at_level("WARNING"):
            report = await bot._run_individual_question(question)

        assert report.prediction == THIN_PUBLISH_BINARY_CEIL
        assert "*Forecaster 1 (gpt-5.6-sol)*: 97.0%" in report.explanation
        assert _marker_lines(caplog, "THIN_PUBLISH_FLOOR:") == [
            f"THIN_PUBLISH_FLOOR: question={QID} raw=0.9700 clamped=0.9500 survivors=1"
        ]

    @pytest.mark.asyncio
    async def test_lone_call_inside_the_band_publishes_unchanged_without_a_marker(
        self, caplog: pytest.LogCaptureFixture
    ):
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        question = _binary_question()
        _serve_survivors(bot, [ReasonedPrediction(prediction_value=0.30, reasoning=GEMINI)])

        with caplog.at_level("INFO"):
            report = await bot._run_individual_question(question)

        assert report.prediction == 0.30
        assert not _marker_lines(caplog, "THIN_PUBLISH_FLOOR:")
        # The single-survivor event itself is still fully disclosed.
        assert _marker_lines(caplog, "FORECASTERS_SURVIVED:") == [
            f"FORECASTERS_SURVIVED: question={QID} survived=1/3 models=gemini-3.1-pro-preview"
        ]
        skip_marker = STACKER_SKIP_REASON_RE.search(report.explanation)
        assert skip_marker is not None
        assert skip_marker.group(1) == "single_forecaster"

    @pytest.mark.asyncio
    async def test_a_lone_mc_survivor_publishes_its_own_distribution(self, caplog: pytest.LogCaptureFixture):
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        lone = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.03),
                PredictedOption(option_name="B", probability=0.97),
            ]
        )
        _serve_survivors(bot, [ReasonedPrediction(prediction_value=lone, reasoning=GEMINI)])

        with caplog.at_level("WARNING"):
            report = await bot._run_individual_question(_mc_question())

        assert isinstance(report.prediction, PredictedOptionList)
        assert [o.probability for o in report.prediction.predicted_options] == [0.03, 0.97]
        assert not _marker_lines(caplog, "THIN_PUBLISH_FLOOR:")


class TestMultiMemberPublishesAreUntouched:
    @pytest.mark.asyncio
    async def test_a_full_ensemble_median_below_the_floor_publishes_as_is(self, caplog: pytest.LogCaptureFixture):
        # Three members all under 0.05 agree; the median IS the aggregation, and the
        # receipt priced flooring it (the "global" variant) at -52.02 spot peer.
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        question = _binary_question()
        _serve_survivors(
            bot,
            [
                ReasonedPrediction(prediction_value=0.02, reasoning=GEMINI),
                ReasonedPrediction(prediction_value=0.03, reasoning=SOL),
                ReasonedPrediction(prediction_value=0.04, reasoning=OPUS),
            ],
        )

        with caplog.at_level("WARNING"):
            report = await bot._run_individual_question(question)

        assert report.prediction == pytest.approx(0.03)
        assert not _marker_lines(caplog, "THIN_PUBLISH_FLOOR:")
        skip_marker = STACKER_SKIP_REASON_RE.search(report.explanation)
        assert skip_marker is not None
        assert skip_marker.group(1) == "spread_below_threshold"

    @pytest.mark.asyncio
    async def test_a_lone_extreme_member_outvoted_by_two_is_absorbed_by_the_median(
        self, caplog: pytest.LogCaptureFixture
    ):
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        question = _binary_question()
        _serve_survivors(
            bot,
            [
                ReasonedPrediction(prediction_value=0.03, reasoning=GEMINI),
                ReasonedPrediction(prediction_value=0.10, reasoning=SOL),
                ReasonedPrediction(prediction_value=0.15, reasoning=OPUS),
            ],
        )

        with caplog.at_level("WARNING"):
            report = await bot._run_individual_question(question)

        assert report.prediction == pytest.approx(0.10)
        assert not _marker_lines(caplog, "THIN_PUBLISH_FLOOR:")


class TestStackedOutputIsNeverFloored:
    """A fired stacker collapses the ensemble to ONE ReasonedPrediction, which the
    framework hands back through the same ``len == 1`` base-combine branch. No
    ``single_forecaster`` reason exists for it, so an extreme stacked value publishes
    as the stacker stated it.
    """

    @pytest.mark.asyncio
    async def test_conditional_stack_fired_on_disagreement(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        monkeypatch.setenv("BINARY_STACKING_ENABLED", "true")
        bot = _make_bot(AggregationStrategy.CONDITIONAL_STACKING)
        question = _binary_question()
        _serve_survivors(
            bot,
            [
                ReasonedPrediction(prediction_value=0.02, reasoning=GEMINI),
                ReasonedPrediction(prediction_value=0.60, reasoning=SOL),
                ReasonedPrediction(prediction_value=0.85, reasoning=OPUS),
            ],
        )

        with (
            patch("metaculus_bot.stacking_route.extract_disagreement_crux", new=AsyncMock(return_value="crux")),
            patch("metaculus_bot.stacking_route.run_targeted_search", new=AsyncMock(return_value="targeted")),
            patch.object(bot._pipeline, "run_stacking", new=AsyncMock(return_value=0.03)),
            caplog.at_level("WARNING"),
        ):
            report = await bot._run_individual_question(question)

        assert bot._pipeline.counters.conditional_stacking_triggered_count == 1
        assert report.prediction == 0.03
        assert STACKER_OUTCOME_PRIMARY in report.explanation
        assert STACKER_SKIP_REASON_RE.search(report.explanation) is None
        assert not _marker_lines(caplog, "THIN_PUBLISH_FLOOR:")

    @pytest.mark.asyncio
    async def test_plain_stacking_strategy(self, caplog: pytest.LogCaptureFixture):
        bot = _make_bot(AggregationStrategy.STACKING)
        question = _binary_question()
        _serve_survivors(
            bot,
            [
                ReasonedPrediction(prediction_value=0.02, reasoning=GEMINI),
                ReasonedPrediction(prediction_value=0.04, reasoning=SOL),
                ReasonedPrediction(prediction_value=0.06, reasoning=OPUS),
            ],
        )

        with (
            patch.object(bot._pipeline, "run_stacking", new=AsyncMock(return_value=0.97)),
            caplog.at_level("WARNING"),
        ):
            report = await bot._run_individual_question(question)

        assert report.prediction == 0.97
        assert STACKER_OUTCOME_PRIMARY in report.explanation
        assert not _marker_lines(caplog, "THIN_PUBLISH_FLOOR:")
