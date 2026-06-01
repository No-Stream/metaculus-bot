"""Integration tests for conditional stacking pipeline.

Tests the full conditional stacking flow: spread computation -> threshold check ->
crux extraction -> targeted research -> stacking (or skip to median aggregation).
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest
from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericDistribution,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from tests.conftest import make_mock_numeric_question

# Standard 11-percentile values used throughout numeric tests
_STANDARD_PERCENTILES = [0.025, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975]


# ---------------------------------------------------------------------------
# Shared mock helper -- most tests patch the same 6-7 objects with minor param
# variations.  This contextmanager centralizes the setup so each test only
# specifies what differs.
# ---------------------------------------------------------------------------


@contextmanager
def mock_stacking_pipeline(
    bot,
    *,
    predictions: list,
    research: str = "base research text",
    crux_return: str | None = "The crux is X",
    crux_side_effect: BaseException | None = None,
    targeted_search_return: str | None = "Targeted search results",
    targeted_search_side_effect: BaseException | None = None,
    aggregate_return=None,
    run_stacking_return=None,
    run_stacking_side_effect: BaseException | None = None,
    extra_patches: list | None = None,
):
    """Mock the conditional stacking pipeline for testing.

    Yields a dict of all mock handles keyed by short name so callers can
    assert on call counts and args.

    Parameters
    ----------
    predictions : list
        The list of ReasonedPrediction objects that _gather_predictions returns.
    research : str
        Return value for bot.run_research.
    crux_return / crux_side_effect : str | None / Exception | None
        Controls extract_disagreement_crux behavior.  If side_effect is set it
        takes precedence (simulates failure).
    targeted_search_return / targeted_search_side_effect : same pattern for
        run_targeted_search.
    aggregate_return : any
        If provided, bot._aggregate_predictions is mocked with this return_value.
        If None, _aggregate_predictions is NOT mocked (runs for real or caller
        provides via extra_patches).
    run_stacking_return / run_stacking_side_effect : controls bot._run_stacking
        mock.  If both are None, _run_stacking is NOT mocked.
    extra_patches : list[patch context managers]
        Additional patches to enter alongside the standard set.  Useful for
        one-off mocks like aggregate_binary_mean.
    """
    crux_kwargs: dict = {"new_callable": AsyncMock}
    if crux_side_effect is not None:
        crux_kwargs["side_effect"] = crux_side_effect
    else:
        crux_kwargs["return_value"] = crux_return

    search_kwargs: dict = {"new_callable": AsyncMock}
    if targeted_search_side_effect is not None:
        search_kwargs["side_effect"] = targeted_search_side_effect
    else:
        search_kwargs["return_value"] = targeted_search_return

    with (
        patch.object(bot, "_get_notepad") as mock_notepad,
        patch.object(bot, "run_research", return_value=research) as mock_research,
        patch.object(bot, "_gather_predictions_with_wall_clock") as mock_gather,
        patch.object(
            bot,
            "_forecaster_with_soft_deadline",
            new=AsyncMock(return_value=ReasonedPrediction(prediction_value=0.5, reasoning="stub")),
        ),
        patch("main.extract_disagreement_crux", **crux_kwargs) as mock_crux,
        patch("main.run_targeted_search", **search_kwargs) as mock_targeted,
    ):
        mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
        mock_gather.return_value = (predictions, [], None)

        # Optional _aggregate_predictions mock
        agg_cm = None
        mock_aggregate = None
        if aggregate_return is not None:
            agg_cm = patch.object(bot, "_aggregate_predictions", return_value=aggregate_return)
            mock_aggregate = agg_cm.__enter__()

        # Optional _run_stacking mock
        stacking_cm = None
        mock_run_stacking = None
        if run_stacking_return is not None or run_stacking_side_effect is not None:
            stacking_kw = {}
            if run_stacking_side_effect is not None:
                stacking_kw["side_effect"] = run_stacking_side_effect
            else:
                stacking_kw["return_value"] = run_stacking_return
            stacking_cm = patch.object(bot, "_run_stacking", **stacking_kw)
            mock_run_stacking = stacking_cm.__enter__()

        # Extra patches
        extra_mocks = {}
        extra_cms = []
        if extra_patches:
            for name, cm in extra_patches:
                entered = cm.__enter__()
                extra_cms.append(cm)
                extra_mocks[name] = entered

        try:
            yield {
                "notepad": mock_notepad,
                "research": mock_research,
                "gather": mock_gather,
                "crux": mock_crux,
                "targeted": mock_targeted,
                "aggregate": mock_aggregate,
                "run_stacking": mock_run_stacking,
                **extra_mocks,
            }
        finally:
            for cm in reversed(extra_cms):
                cm.__exit__(None, None, None)
            if stacking_cm is not None:
                stacking_cm.__exit__(None, None, None)
            if agg_cm is not None:
                agg_cm.__exit__(None, None, None)


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
        min_forecasters_to_publish=1,
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


# Standard binary predictions with high spread (0.10 vs 0.85 -> range 0.75)
_HIGH_SPREAD_BINARY = [
    ReasonedPrediction(prediction_value=0.10, reasoning="Model: m1\n\nLikely no."),
    ReasonedPrediction(prediction_value=0.85, reasoning="Model: m2\n\nLikely yes."),
]

# Standard binary predictions with low spread (0.45 vs 0.55 -> range 0.10)
_LOW_SPREAD_BINARY = [
    ReasonedPrediction(prediction_value=0.45, reasoning="Model: m1\n\nSlightly no."),
    ReasonedPrediction(prediction_value=0.55, reasoning="Model: m2\n\nSlightly yes."),
]


def _make_high_spread_mc_predictions() -> list[ReasonedPrediction]:
    """MC predictions where option A has 0.30 spread (>0.20 threshold)."""
    return [
        ReasonedPrediction(
            prediction_value=PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="A", probability=0.60),
                    PredictedOption(option_name="B", probability=0.25),
                    PredictedOption(option_name="C", probability=0.15),
                ]
            ),
            reasoning="Model: m1\n\nOption A is dominant.",
        ),
        ReasonedPrediction(
            prediction_value=PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="A", probability=0.30),
                    PredictedOption(option_name="B", probability=0.45),
                    PredictedOption(option_name="C", probability=0.25),
                ]
            ),
            reasoning="Model: m2\n\nOption B is stronger.",
        ),
    ]


def _make_stacked_mc_result() -> PredictedOptionList:
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=0.45),
            PredictedOption(option_name="B", probability=0.35),
            PredictedOption(option_name="C", probability=0.20),
        ]
    )


def _make_numeric_distribution(median_value: float, spread_factor: float = 1.0) -> NumericDistribution:
    """Build a NumericDistribution with 11 percentiles centered on median_value."""
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


def _make_high_spread_numeric_predictions() -> list[ReasonedPrediction]:
    """Numeric predictions: model 1 centered at 30, model 2 at 70 -> spread 0.40 >> 0.15."""
    return [
        ReasonedPrediction(prediction_value=_make_numeric_distribution(30.0), reasoning="Model: m1\n\nLow estimate."),
        ReasonedPrediction(prediction_value=_make_numeric_distribution(70.0), reasoning="Model: m2\n\nHigh estimate."),
    ]


class TestConditionalStackingBinaryTrigger:
    """Tests that stacking triggers correctly for binary questions with high spread."""

    @pytest.mark.asyncio
    async def test_high_spread_triggers_stacking(self):
        """High prob-range spread (>0.15) should trigger the full stacking pipeline."""
        bot = _make_bot()
        question = _make_binary_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_HIGH_SPREAD_BINARY,
            crux_return="The crux is whether X happened",
            targeted_search_return="Targeted search found that X did happen",
            aggregate_return=0.72,
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            crux_call_args = mocks["crux"].call_args
            assert crux_call_args[0][0] == bot._analyzer_llm
            assert crux_call_args[0][1] == question.question_text

            mocks["targeted"].assert_called_once()
            assert mocks["targeted"].call_args[0][0] == "The crux is whether X happened"

            mocks["aggregate"].assert_called_once()
            agg_kwargs = mocks["aggregate"].call_args[1]
            assert "Targeted Research" in agg_kwargs["research"]
            assert "base research text" in agg_kwargs["research"]

            assert "## Targeted Research (addressing model disagreement)" in result.research_report
            assert "base research text" in result.research_report

            assert len(result.predictions) == 1
            assert result.predictions[0].prediction_value == 0.72

            assert bot._conditional_stacking_triggered_count == 1
            assert bot._conditional_stacking_skipped_count == 0

    @pytest.mark.asyncio
    async def test_low_spread_skips_stacking(self):
        """Low prob-range spread (<0.15) should skip stacking and return all individual predictions."""
        bot = _make_bot()
        question = _make_binary_question()

        with mock_stacking_pipeline(bot, predictions=_LOW_SPREAD_BINARY) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_not_called()
            mocks["targeted"].assert_not_called()

            assert len(result.predictions) == 2
            assert result.predictions[0].prediction_value == 0.45
            assert result.predictions[1].prediction_value == 0.55

            assert "## Targeted Research" not in result.research_report

            assert bot._conditional_stacking_skipped_count == 1
            assert bot._conditional_stacking_triggered_count == 0


class TestConditionalStackingFallbacks:
    """Tests for graceful degradation when pipeline components fail."""

    @pytest.mark.asyncio
    async def test_crux_extraction_failure_falls_through(self):
        """Crux extraction failure should still proceed to stacking with base research only."""
        bot = _make_bot()
        question = _make_binary_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_HIGH_SPREAD_BINARY,
            crux_side_effect=RuntimeError("Analyzer LLM timed out"),
            aggregate_return=0.65,
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            mocks["targeted"].assert_not_called()

            mocks["aggregate"].assert_called_once()
            agg_kwargs = mocks["aggregate"].call_args[1]
            assert "Targeted Research" not in agg_kwargs["research"]

            assert len(result.predictions) == 1
            assert result.predictions[0].prediction_value == 0.65

    @pytest.mark.asyncio
    async def test_targeted_search_failure_falls_through(self):
        """Targeted search failure should still proceed to stacking with base research only."""
        bot = _make_bot()
        question = _make_binary_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_HIGH_SPREAD_BINARY,
            crux_return="The crux is about X",
            targeted_search_side_effect=RuntimeError("Search API down"),
            aggregate_return=0.60,
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["targeted"].assert_called_once()

            mocks["aggregate"].assert_called_once()
            agg_kwargs = mocks["aggregate"].call_args[1]
            assert "Targeted Research" not in agg_kwargs["research"]

            assert len(result.predictions) == 1
            assert result.predictions[0].prediction_value == 0.60

    @pytest.mark.asyncio
    async def test_stacking_failure_falls_back_to_median(self):
        """When stacking itself fails, fallback to MEDIAN aggregation (via MEAN fallback path)."""
        bot = _make_bot(stacking_fallback_on_failure=True)
        question = _make_binary_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_HIGH_SPREAD_BINARY,
            crux_return="crux text",
            targeted_search_return="targeted results",
            run_stacking_side_effect=RuntimeError("Stacker LLM crashed"),
            extra_patches=[
                ("binary_mean", patch("metaculus_bot.numeric_utils.aggregate_binary_mean", return_value=0.475)),
            ],
        ):
            result = await bot._research_and_make_predictions(question)

            assert len(result.predictions) == 1
            assert result.predictions[0].prediction_value == 0.475
            assert bot._stacking_fallback_count == 1


class TestConditionalStackingMC:
    """Tests for conditional stacking with multiple choice questions."""

    @pytest.mark.asyncio
    async def test_mc_high_spread_triggers(self):
        """MC question with >0.20 max option spread should trigger stacking."""
        bot = _make_bot()
        question = _make_mc_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_make_high_spread_mc_predictions(),
            research="mc research",
            crux_return="Disagreement about option A vs B",
            targeted_search_return="Search results about A vs B",
            aggregate_return=_make_stacked_mc_result(),
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            mocks["targeted"].assert_called_once()
            mocks["aggregate"].assert_called_once()

            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1


class TestConditionalStackingThresholds:
    """Tests for custom and boundary threshold behavior."""

    @pytest.mark.asyncio
    async def test_custom_thresholds(self):
        """Very high custom thresholds should prevent stacking even with moderate spread."""
        bot = _make_bot(stacking_spread_thresholds={"binary": 0.95, "mc": 0.90, "numeric": 0.90})
        question = _make_binary_question()

        # 0.20 and 0.80 -> range 0.60, below custom threshold 0.95
        predictions = [
            ReasonedPrediction(prediction_value=0.20, reasoning="Model: m1\n\nUnlikely."),
            ReasonedPrediction(prediction_value=0.80, reasoning="Model: m2\n\nLikely."),
        ]

        with mock_stacking_pipeline(bot, predictions=predictions, research="research") as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_not_called()
            assert len(result.predictions) == 2
            assert bot._conditional_stacking_skipped_count == 1

    @pytest.mark.asyncio
    async def test_spread_just_above_threshold(self):
        """Spread barely above threshold should trigger stacking."""
        bot = _make_bot()
        question = _make_binary_question()

        # 0.40 - 0.24 = 0.16 > 0.15 -> triggers
        predictions = [
            ReasonedPrediction(prediction_value=0.24, reasoning="Model: m1\n\nAnalysis 1"),
            ReasonedPrediction(prediction_value=0.40, reasoning="Model: m2\n\nAnalysis 2"),
        ]

        with mock_stacking_pipeline(
            bot,
            predictions=predictions,
            research="research",
            crux_return="crux",
            targeted_search_return="search results",
            aggregate_return=0.50,
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1

    @pytest.mark.asyncio
    async def test_spread_just_below_threshold(self):
        """Spread barely below threshold should skip stacking."""
        bot = _make_bot()
        question = _make_binary_question()

        # 0.39 - 0.25 = 0.14 < 0.15 -> skips
        predictions = [
            ReasonedPrediction(prediction_value=0.25, reasoning="Model: m1\n\nAnalysis 1"),
            ReasonedPrediction(prediction_value=0.39, reasoning="Model: m2\n\nAnalysis 2"),
        ]

        with mock_stacking_pipeline(bot, predictions=predictions, research="research") as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_not_called()
            assert len(result.predictions) == 2
            assert bot._conditional_stacking_skipped_count == 1


class TestConditionalStackingNumeric:
    """Tests for conditional stacking with numeric questions."""

    @pytest.mark.asyncio
    async def test_numeric_high_spread_triggers(self):
        """Numeric predictions with large median disagreement (spread/range > 0.15) should trigger stacking."""
        bot = _make_bot()
        question = make_mock_numeric_question(
            question_text="How many units will be sold?",
            background_info="Sales question",
            resolution_criteria="Resolves to actual unit count",
            page_url="https://test.com/3",
            unit_of_measure="units",
            cdf_size=201,
        )

        with mock_stacking_pipeline(
            bot,
            predictions=_make_high_spread_numeric_predictions(),
            research="numeric research",
            crux_return="Disagreement about expected sales volume",
            targeted_search_return="Search results about sales volume",
            aggregate_return=_make_numeric_distribution(median_value=50.0),
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            mocks["targeted"].assert_called_once()
            mocks["aggregate"].assert_called_once()

            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1
            assert bot._conditional_stacking_skipped_count == 0

    @pytest.mark.asyncio
    async def test_numeric_low_spread_skips(self):
        """Numeric predictions with similar medians (spread/range < 0.15) should skip stacking."""
        bot = _make_bot()
        question = make_mock_numeric_question(
            question_text="How many units will be sold?",
            background_info="Sales question",
            resolution_criteria="Resolves to actual unit count",
            page_url="https://test.com/3",
            unit_of_measure="units",
            cdf_size=201,
        )

        predictions = [
            ReasonedPrediction(prediction_value=_make_numeric_distribution(48.0), reasoning="Model: m1\n\nAround 48."),
            ReasonedPrediction(prediction_value=_make_numeric_distribution(52.0), reasoning="Model: m2\n\nAround 52."),
        ]

        with mock_stacking_pipeline(bot, predictions=predictions, research="numeric research") as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_not_called()
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

        with mock_stacking_pipeline(
            bot,
            predictions=_HIGH_SPREAD_BINARY,
            crux_return="The crux is X",
            targeted_search_return="Targeted results about X",
            run_stacking_return=0.65,
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["run_stacking"].assert_called_once()
            call_args = mocks["run_stacking"].call_args
            assert call_args[0][0] == question
            assert "Targeted Research" in call_args[0][1]
            assert len(call_args[0][2]) == 2

            assert len(result.predictions) == 1
            assert result.predictions[0].prediction_value == 0.65

    @pytest.mark.asyncio
    async def test_low_spread_uses_median_aggregation(self):
        """Low spread should skip stacking and use MEDIAN aggregation in _aggregate_predictions."""
        bot = _make_bot()
        question = _make_binary_question()

        test_llm = GeneralLlm(model="test-model", temperature=0.0)
        bot._forecaster_llms = [test_llm, test_llm, test_llm]

        predictions = [
            ReasonedPrediction(prediction_value=0.45, reasoning="Model: m1\n\nAnalysis 1"),
            ReasonedPrediction(prediction_value=0.50, reasoning="Model: m2\n\nAnalysis 2"),
            ReasonedPrediction(prediction_value=0.55, reasoning="Model: m3\n\nAnalysis 3"),
        ]

        with mock_stacking_pipeline(bot, predictions=predictions, research="research") as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_not_called()
            assert bot._conditional_stacking_skipped_count == 1
            assert len(result.predictions) == 3

            # Verify MEDIAN is used (not MEAN): MEDIAN([0.30, 0.50, 0.60]) = 0.50
            aggregated = await bot._aggregate_predictions(predictions=[0.30, 0.50, 0.60], question=question)
            assert aggregated == pytest.approx(0.50)


class TestConditionalStackingBenchmarkingFlag:
    """Tests that is_benchmarking flag is correctly passed through."""

    @pytest.mark.asyncio
    async def test_is_benchmarking_passed_to_targeted_search(self):
        """is_benchmarking=True should be forwarded to run_targeted_search."""
        bot = _make_bot()
        question = _make_binary_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_HIGH_SPREAD_BINARY,
            research="base research",
            crux_return="The crux",
            targeted_search_return="targeted results",
            aggregate_return=0.50,
        ) as mocks:
            await bot._research_and_make_predictions(question)

            mocks["targeted"].assert_called_once()
            assert mocks["targeted"].call_args.kwargs["is_benchmarking"] is True


class TestConditionalStackingModelTagStripping:
    """Tests that model tags are stripped before crux extraction."""

    @pytest.mark.asyncio
    async def test_model_tags_stripped_before_crux_extraction(self):
        """Predictions with 'Model: ...' prefix should have it stripped before passing to crux extraction."""
        bot = _make_bot()
        question = _make_binary_question()

        predictions = [
            ReasonedPrediction(prediction_value=0.10, reasoning="Model: gpt-5.4\n\nThis model thinks it's unlikely."),
            ReasonedPrediction(prediction_value=0.85, reasoning="Model: claude-4\n\nThis model thinks it's likely."),
        ]

        with mock_stacking_pipeline(
            bot,
            predictions=predictions,
            research="base research",
            crux_return="The crux of disagreement",
            targeted_search_return="targeted results",
            aggregate_return=0.50,
        ) as mocks:
            await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            base_prediction_texts = mocks["crux"].call_args[0][2]
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

        with mock_stacking_pipeline(
            bot,
            predictions=_HIGH_SPREAD_BINARY,
            research="base research",
            crux_side_effect=RuntimeError("LLM timed out"),
            aggregate_return=0.50,
        ) as mocks:
            await bot._research_and_make_predictions(question)

            assert bot._conditional_stacking_crux_failures == 1
            mocks["targeted"].assert_not_called()

    @pytest.mark.asyncio
    async def test_search_failure_increments_counter(self):
        """Targeted search failure should increment _conditional_stacking_search_failures."""
        bot = _make_bot()
        question = _make_binary_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_HIGH_SPREAD_BINARY,
            research="base research",
            crux_return="The crux",
            targeted_search_side_effect=RuntimeError("Search API down"),
            aggregate_return=0.50,
        ):
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
        bot._forecaster_llms = [test_llm] * 6
        question = _make_binary_question()

        predictions = [
            ReasonedPrediction(prediction_value=v, reasoning=f"Model: m{i}\n\nAnalysis {i}")
            for i, v in enumerate([0.10, 0.20, 0.30, 0.70, 0.80, 0.90], start=1)
        ]

        with mock_stacking_pipeline(
            bot,
            predictions=predictions,
            research="research",
            crux_return="crux",
            targeted_search_return="search results",
            aggregate_return=0.50,
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1


class TestNumericStackingEnabled:
    """Tests for NUMERIC_STACKING_ENABLED env var gating (positive polarity)."""

    @pytest.mark.asyncio
    async def test_numeric_high_spread_skips_when_disabled(self, monkeypatch):
        """When NUMERIC_STACKING_ENABLED=false, numeric questions bypass stacking even with high spread."""
        monkeypatch.setenv("NUMERIC_STACKING_ENABLED", "false")
        bot = _make_bot()
        question = make_mock_numeric_question(
            question_text="How many units will be sold?",
            background_info="Sales question",
            resolution_criteria="Resolves to actual unit count",
            page_url="https://test.com/3",
            unit_of_measure="units",
            cdf_size=201,
        )

        with mock_stacking_pipeline(
            bot,
            predictions=_make_high_spread_numeric_predictions(),
            research="numeric research",
            crux_return="Disagreement about expected sales volume",
            targeted_search_return="Search results about sales volume",
            aggregate_return=_make_numeric_distribution(30.0),
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_not_called()
            mocks["targeted"].assert_not_called()
            mocks["aggregate"].assert_not_called()

            assert len(result.predictions) == 2
            assert bot._conditional_stacking_skipped_count == 1
            assert bot._conditional_stacking_triggered_count == 0
            assert bot._stacker_outcome[question.id_of_question] == "skipped"

    @pytest.mark.asyncio
    async def test_binary_high_spread_still_triggers_when_numeric_disabled(self, monkeypatch):
        """NUMERIC_STACKING_ENABLED=false should NOT affect binary questions -- stacking still fires."""
        monkeypatch.setenv("NUMERIC_STACKING_ENABLED", "false")
        bot = _make_bot()
        question = _make_binary_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_HIGH_SPREAD_BINARY,
            crux_return="The crux is whether X happened",
            targeted_search_return="Targeted search found that X did happen",
            aggregate_return=0.72,
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            mocks["targeted"].assert_called_once()
            mocks["aggregate"].assert_called_once()

            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1

    @pytest.mark.asyncio
    async def test_mc_high_spread_still_triggers_when_numeric_disabled(self, monkeypatch):
        """NUMERIC_STACKING_ENABLED=false should NOT affect MC questions -- stacking still fires."""
        monkeypatch.setenv("NUMERIC_STACKING_ENABLED", "false")
        bot = _make_bot()
        question = _make_mc_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_make_high_spread_mc_predictions(),
            research="mc research",
            crux_return="Disagreement about option A vs B",
            targeted_search_return="Search results about A vs B",
            aggregate_return=_make_stacked_mc_result(),
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            mocks["targeted"].assert_called_once()
            mocks["aggregate"].assert_called_once()

            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1

    @pytest.mark.asyncio
    async def test_numeric_stacking_enabled_unset_defaults_to_DISABLED(self, monkeypatch):
        """When NUMERIC_STACKING_ENABLED is unset (default=False), numeric high-spread SKIPS stacking."""
        monkeypatch.delenv("NUMERIC_STACKING_ENABLED", raising=False)
        bot = _make_bot()
        question = make_mock_numeric_question(
            question_text="How many units will be sold?",
            background_info="Sales question",
            resolution_criteria="Resolves to actual unit count",
            page_url="https://test.com/3",
            unit_of_measure="units",
            cdf_size=201,
        )

        with mock_stacking_pipeline(
            bot,
            predictions=_make_high_spread_numeric_predictions(),
            research="numeric research",
            crux_return="Disagreement about expected sales volume",
            targeted_search_return="Search results about sales volume",
            aggregate_return=_make_numeric_distribution(50.0),
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_not_called()
            mocks["targeted"].assert_not_called()
            mocks["aggregate"].assert_not_called()

            assert len(result.predictions) == 2
            assert bot._conditional_stacking_skipped_count == 1
            assert bot._conditional_stacking_triggered_count == 0
            assert bot._stacker_outcome[question.id_of_question] == "skipped"

    @pytest.mark.asyncio
    async def test_numeric_stacking_enabled_explicit_true(self, monkeypatch):
        """When NUMERIC_STACKING_ENABLED=true (explicit), numeric high-spread triggers stacking."""
        monkeypatch.setenv("NUMERIC_STACKING_ENABLED", "true")
        bot = _make_bot()
        question = make_mock_numeric_question(
            question_text="How many units will be sold?",
            background_info="Sales question",
            resolution_criteria="Resolves to actual unit count",
            page_url="https://test.com/3",
            unit_of_measure="units",
            cdf_size=201,
        )

        with mock_stacking_pipeline(
            bot,
            predictions=_make_high_spread_numeric_predictions(),
            research="numeric research",
            crux_return="Disagreement about expected sales volume",
            targeted_search_return="Search results about sales volume",
            aggregate_return=_make_numeric_distribution(50.0),
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            mocks["targeted"].assert_called_once()
            mocks["aggregate"].assert_called_once()

            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1


class TestBinaryStackingEnabled:
    """Tests for BINARY_STACKING_ENABLED env var gating (positive polarity)."""

    @pytest.mark.asyncio
    async def test_binary_high_spread_skips_when_disabled(self, monkeypatch):
        """When BINARY_STACKING_ENABLED=false, binary questions bypass stacking even with high spread."""
        monkeypatch.setenv("BINARY_STACKING_ENABLED", "false")
        bot = _make_bot()
        question = _make_binary_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_HIGH_SPREAD_BINARY,
            crux_return="The crux is whether X happened",
            targeted_search_return="Targeted search found that X did happen",
            aggregate_return=0.72,
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_not_called()
            mocks["targeted"].assert_not_called()
            mocks["aggregate"].assert_not_called()

            assert len(result.predictions) == 2
            assert bot._conditional_stacking_skipped_count == 1
            assert bot._conditional_stacking_triggered_count == 0
            assert bot._stacker_outcome[question.id_of_question] == "skipped"

    @pytest.mark.asyncio
    async def test_binary_stacking_enabled_unset_defaults_to_DISABLED(self, monkeypatch):
        """When BINARY_STACKING_ENABLED is unset (default=False), binary high-spread SKIPS stacking."""
        monkeypatch.delenv("BINARY_STACKING_ENABLED", raising=False)
        bot = _make_bot()
        question = _make_binary_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_HIGH_SPREAD_BINARY,
            crux_return="The crux is whether X happened",
            targeted_search_return="Targeted search found that X did happen",
            aggregate_return=0.72,
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_not_called()
            mocks["targeted"].assert_not_called()
            mocks["aggregate"].assert_not_called()

            assert len(result.predictions) == 2
            assert bot._conditional_stacking_skipped_count == 1
            assert bot._conditional_stacking_triggered_count == 0
            assert bot._stacker_outcome[question.id_of_question] == "skipped"

    @pytest.mark.asyncio
    async def test_binary_stacking_enabled_explicit_true(self, monkeypatch):
        """When BINARY_STACKING_ENABLED=true (explicit), binary high-spread triggers stacking."""
        monkeypatch.setenv("BINARY_STACKING_ENABLED", "true")
        bot = _make_bot()
        question = _make_binary_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_HIGH_SPREAD_BINARY,
            crux_return="The crux is whether X happened",
            targeted_search_return="Targeted search found that X did happen",
            aggregate_return=0.72,
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            mocks["targeted"].assert_called_once()
            mocks["aggregate"].assert_called_once()

            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1

    @pytest.mark.asyncio
    async def test_binary_disabled_does_not_affect_numeric(self, monkeypatch):
        """BINARY_STACKING_ENABLED=false does NOT affect numeric questions -- stacking still fires."""
        monkeypatch.setenv("BINARY_STACKING_ENABLED", "false")
        bot = _make_bot()
        question = make_mock_numeric_question(
            question_text="How many units will be sold?",
            background_info="Sales question",
            resolution_criteria="Resolves to actual unit count",
            page_url="https://test.com/3",
            unit_of_measure="units",
            cdf_size=201,
        )

        with mock_stacking_pipeline(
            bot,
            predictions=_make_high_spread_numeric_predictions(),
            research="numeric research",
            crux_return="Disagreement about sales volume",
            targeted_search_return="Search results about sales volume",
            aggregate_return=_make_numeric_distribution(50.0),
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            mocks["targeted"].assert_called_once()
            mocks["aggregate"].assert_called_once()

            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1

    @pytest.mark.asyncio
    async def test_binary_disabled_does_not_affect_mc(self, monkeypatch):
        """BINARY_STACKING_ENABLED=false does NOT affect MC questions -- stacking still fires."""
        monkeypatch.setenv("BINARY_STACKING_ENABLED", "false")
        bot = _make_bot()
        question = _make_mc_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_make_high_spread_mc_predictions(),
            research="mc research",
            crux_return="Disagreement about option A vs B",
            targeted_search_return="Search results about A vs B",
            aggregate_return=_make_stacked_mc_result(),
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            mocks["targeted"].assert_called_once()
            mocks["aggregate"].assert_called_once()

            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1


class TestMCStackingEnabled:
    """Tests for MC_STACKING_ENABLED env var gating (positive polarity)."""

    @pytest.mark.asyncio
    async def test_mc_high_spread_skips_when_disabled(self, monkeypatch):
        """When MC_STACKING_ENABLED=false, MC questions bypass stacking even with high spread."""
        monkeypatch.setenv("MC_STACKING_ENABLED", "false")
        bot = _make_bot()
        question = _make_mc_question()

        mc_preds = _make_high_spread_mc_predictions()

        with mock_stacking_pipeline(
            bot,
            predictions=mc_preds,
            research="mc research",
            crux_return="Disagreement about option A vs B",
            targeted_search_return="Search results about A vs B",
            aggregate_return=mc_preds[0].prediction_value,
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_not_called()
            mocks["targeted"].assert_not_called()
            mocks["aggregate"].assert_not_called()

            assert len(result.predictions) == 2
            assert bot._conditional_stacking_skipped_count == 1
            assert bot._conditional_stacking_triggered_count == 0
            assert bot._stacker_outcome[question.id_of_question] == "skipped"

    @pytest.mark.asyncio
    async def test_mc_stacking_enabled_unset_defaults_to_DISABLED(self, monkeypatch):
        """When MC_STACKING_ENABLED is unset (default=False), MC high-spread SKIPS stacking."""
        monkeypatch.delenv("MC_STACKING_ENABLED", raising=False)
        bot = _make_bot()
        question = _make_mc_question()

        mc_preds = _make_high_spread_mc_predictions()

        with mock_stacking_pipeline(
            bot,
            predictions=mc_preds,
            research="mc research",
            crux_return="Disagreement about option A vs B",
            targeted_search_return="Search results about A vs B",
            aggregate_return=_make_stacked_mc_result(),
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_not_called()
            mocks["targeted"].assert_not_called()
            mocks["aggregate"].assert_not_called()

            assert len(result.predictions) == 2
            assert bot._conditional_stacking_skipped_count == 1
            assert bot._conditional_stacking_triggered_count == 0
            assert bot._stacker_outcome[question.id_of_question] == "skipped"

    @pytest.mark.asyncio
    async def test_mc_stacking_enabled_explicit_true(self, monkeypatch):
        """When MC_STACKING_ENABLED=true (explicit), MC high-spread triggers stacking."""
        monkeypatch.setenv("MC_STACKING_ENABLED", "true")
        bot = _make_bot()
        question = _make_mc_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_make_high_spread_mc_predictions(),
            research="mc research",
            crux_return="Disagreement about option A vs B",
            targeted_search_return="Search results about A vs B",
            aggregate_return=_make_stacked_mc_result(),
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            mocks["targeted"].assert_called_once()
            mocks["aggregate"].assert_called_once()

            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1

    @pytest.mark.asyncio
    async def test_mc_disabled_does_not_affect_binary(self, monkeypatch):
        """MC_STACKING_ENABLED=false does NOT affect binary questions -- stacking still fires."""
        monkeypatch.setenv("MC_STACKING_ENABLED", "false")
        bot = _make_bot()
        question = _make_binary_question()

        with mock_stacking_pipeline(
            bot,
            predictions=_HIGH_SPREAD_BINARY,
            crux_return="The crux is whether X happened",
            targeted_search_return="Targeted search found that X did happen",
            aggregate_return=0.72,
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            mocks["targeted"].assert_called_once()
            mocks["aggregate"].assert_called_once()

            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1

    @pytest.mark.asyncio
    async def test_mc_disabled_does_not_affect_numeric(self, monkeypatch):
        """MC_STACKING_ENABLED=false does NOT affect numeric questions -- stacking still fires."""
        monkeypatch.setenv("MC_STACKING_ENABLED", "false")
        bot = _make_bot()
        question = make_mock_numeric_question(
            question_text="How many units will be sold?",
            background_info="Sales question",
            resolution_criteria="Resolves to actual unit count",
            page_url="https://test.com/3",
            unit_of_measure="units",
            cdf_size=201,
        )

        with mock_stacking_pipeline(
            bot,
            predictions=_make_high_spread_numeric_predictions(),
            research="numeric research",
            crux_return="Disagreement about sales volume",
            targeted_search_return="Search results about sales volume",
            aggregate_return=_make_numeric_distribution(50.0),
        ) as mocks:
            result = await bot._research_and_make_predictions(question)

            mocks["crux"].assert_called_once()
            mocks["targeted"].assert_called_once()
            mocks["aggregate"].assert_called_once()

            assert len(result.predictions) == 1
            assert bot._conditional_stacking_triggered_count == 1
