"""End-to-end pipeline tests for numeric and multiple-choice question types.

Exercises the full _research_and_make_predictions flow for numeric (percentile
branch, mixture branch, discrete snap, unit mismatch, high-spread stacking)
and MC (low-spread median, clamping/renorm, high-spread stacking) questions.
Also tests cross-cutting batch behavior and comment markers.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
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
from metaculus_bot.numeric.config import STANDARD_PERCENTILES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime.now()
_OPEN = _NOW - timedelta(days=30)
_RESOLVE = _NOW + timedelta(days=365)


def _make_numeric_question(
    *,
    qid: int = 5001,
    lower_bound: float = 0.0,
    upper_bound: float = 20.0,
    open_lower: bool = True,
    open_upper: bool = True,
    question_text: str = "What will the US unemployment rate be?",
) -> NumericQuestion:
    return NumericQuestion(
        question_text=question_text,
        id_of_post=qid,
        id_of_question=qid,
        page_url=f"https://www.metaculus.com/questions/{qid}",
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        open_lower_bound=open_lower,
        open_upper_bound=open_upper,
        zero_point=None,
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
        resolution_criteria="Resolves to the BLS U-3 rate",
        background_info="US unemployment context",
        fine_print="",
        unit_of_measure="%",
    )


def _make_mc_question(
    *,
    qid: int = 5002,
    options: list[str] | None = None,
    question_text: str = "Which color will win?",
) -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text=question_text,
        id_of_post=qid,
        id_of_question=qid,
        page_url=f"https://www.metaculus.com/questions/{qid}",
        options=options or ["Red", "Blue", "Green"],
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
        resolution_criteria="Resolves to the winning color",
        background_info="Color competition context",
        fine_print="",
    )


def _make_binary_question(*, qid: int = 5003) -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will it rain tomorrow?",
        id_of_post=qid,
        id_of_question=qid,
        page_url=f"https://www.metaculus.com/questions/{qid}",
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
        resolution_criteria="Resolves YES if it rains",
        background_info="Weather context",
        fine_print="",
    )


def _make_bot(n_forecasters: int = 3, strategy: AggregationStrategy = AggregationStrategy.CONDITIONAL_STACKING):
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    return TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        aggregation_strategy=strategy,
        llms={
            "forecasters": [test_llm] * n_forecasters,
            "stacker": test_llm,
            "analyzer": test_llm,
            "default": test_llm,
            "parser": test_llm,
            "researcher": test_llm,
            "summarizer": test_llm,
        },
        is_benchmarking=True,
        stacking_fallback_on_failure=True,
        min_forecasters_to_publish=2,
    )


def _numeric_percentiles(median: float, scale: float = 1.0, bounds: tuple[float, float] = (0.0, 20.0)):
    """Build 11 standard percentiles centered on `median`."""
    offsets = [-3.5, -3.0, -2.5, -1.8, -0.5, 0.0, 0.5, 1.8, 2.5, 3.0, 3.5]
    lo, hi = bounds
    return [
        Percentile(percentile=pct, value=max(lo + 0.01, min(hi - 0.01, median + off * scale)))
        for pct, off in zip(STANDARD_PERCENTILES, offsets)
    ]


def _build_numeric_distribution(
    median: float,
    scale: float = 1.0,
    bounds: tuple[float, float] = (0.0, 20.0),
    *,
    open_lower: bool = True,
    open_upper: bool = True,
):
    """Build a proper 201-point CDF using the real PCHIP pipeline."""
    from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles

    lo, hi = bounds
    question = NumericQuestion(
        question_text="Synthetic for test",
        id_of_post=9999,
        id_of_question=9999,
        page_url="https://www.metaculus.com/questions/9999",
        lower_bound=lo,
        upper_bound=hi,
        open_lower_bound=open_lower,
        open_upper_bound=open_upper,
        zero_point=None,
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
        resolution_criteria="test",
        background_info="test",
        fine_print="",
        unit_of_measure="%",
    )
    percentiles = _numeric_percentiles(median, scale, bounds)
    sanitized, zero_point = sanitize_percentiles(percentiles, question)
    return build_numeric_distribution(sanitized, question, zero_point)


def _mc_option_list(probs: list[float], options: list[str] | None = None) -> PredictedOptionList:
    opts = options or ["Red", "Blue", "Green"]
    return PredictedOptionList(
        predicted_options=[PredictedOption(option_name=name, probability=p) for name, p in zip(opts, probs)]
    )


# ---------------------------------------------------------------------------
# Numeric Pipeline Tests
# ---------------------------------------------------------------------------


class TestNumericPercentileBranch:
    """Numeric question through the standard 11-percentile path produces a valid CDF."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_numeric_percentile_branch_produces_valid_cdf(self):
        bot = _make_bot(n_forecasters=3)
        question = _make_numeric_question()

        predictions = [
            ReasonedPrediction(
                prediction_value=_build_numeric_distribution(4.8),
                reasoning="Model: m1\n\nOUTCOME_TYPE: CONTINUOUS\n\nPercentile 2.5: 1.3\nP50: 4.8",
            ),
            ReasonedPrediction(
                prediction_value=_build_numeric_distribution(5.0),
                reasoning="Model: m2\n\nOUTCOME_TYPE: CONTINUOUS\n\nPercentile 2.5: 1.5\nP50: 5.0",
            ),
            ReasonedPrediction(
                prediction_value=_build_numeric_distribution(4.6),
                reasoning="Model: m3\n\nOUTCOME_TYPE: CONTINUOUS\n\nPercentile 2.5: 1.1\nP50: 4.6",
            ),
        ]

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="Canned research") as _,
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
        aggregated = await bot._aggregate_predictions([p.prediction_value for p in result.predictions], question)
        assert isinstance(aggregated, NumericDistribution)

        cdf_points = aggregated.cdf
        assert len(cdf_points) == 201

        prob_values = [p.percentile for p in cdf_points]
        for i in range(len(prob_values) - 1):
            assert prob_values[i] <= prob_values[i + 1], f"CDF not monotonic at index {i}"

        assert prob_values[0] >= 0.001
        assert prob_values[-1] <= 0.999

        for i in range(len(prob_values) - 1):
            step = prob_values[i + 1] - prob_values[i]
            assert step >= 5e-5 - 1e-10, f"Min step violation at index {i}: step={step}"


class TestNumericMixtureBranch:
    """Numeric question through the mixture-of-normals path produces a valid CDF."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_numeric_mixture_branch_produces_valid_cdf(self):
        bot = _make_bot(n_forecasters=1, strategy=AggregationStrategy.MEDIAN)
        bot.min_forecasters_to_publish = 1
        question = _make_numeric_question()

        from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution
        from metaculus_bot.probabilistic_tools.mixtures import (
            MixtureComponent,
            MixtureOfNormals,
            percentiles_to_metaculus_cdf_via_mixture,
        )

        mixture = MixtureOfNormals(
            components=(
                MixtureComponent(weight=0.6, mean=5.0, sd=1.5),
                MixtureComponent(weight=0.4, mean=8.0, sd=2.0),
            )
        )
        cdf_percentiles = percentiles_to_metaculus_cdf_via_mixture(mixture, question)
        cdf_values = [float(p.percentile) for p in cdf_percentiles]

        declared = []
        for target_pct in STANDARD_PERCENTILES:
            hit = next((p for p in cdf_percentiles if p.percentile >= target_pct), cdf_percentiles[-1])
            declared.append(Percentile(percentile=target_pct, value=float(hit.value)))

        prediction_dist = create_pchip_numeric_distribution(cdf_values, declared, question, zero_point=None)

        prediction = ReasonedPrediction(
            prediction_value=prediction_dist,
            reasoning="Model: m1\n\nMixture branch output",
        )

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="Canned research") as _,
            patch.object(bot, "_gather_predictions_with_wall_clock") as mock_gather,
            patch.object(
                bot,
                "_forecaster_with_soft_deadline",
                new=AsyncMock(return_value=prediction),
            ),
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = ([prediction], [], None)

            result = await bot._research_and_make_predictions(question)

        assert len(result.predictions) == 1
        dist = result.predictions[0].prediction_value
        assert isinstance(dist, NumericDistribution)

        cdf_points = dist.cdf
        assert len(cdf_points) == 201

        prob_values = [p.percentile for p in cdf_points]
        for i in range(len(prob_values) - 1):
            assert prob_values[i] <= prob_values[i + 1], f"CDF not monotonic at index {i}"

        assert prob_values[0] >= 0.001
        assert prob_values[-1] <= 0.999

        for i in range(len(prob_values) - 1):
            step = prob_values[i + 1] - prob_values[i]
            assert step >= 5e-5 - 1e-10, f"Min step violation at index {i}: step={step}"


class TestNumericDiscreteIntegerSnap:
    """Discrete integer snap applies step-function characteristics when majority votes DISCRETE."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_numeric_discrete_integer_snap(self):
        bot = _make_bot(n_forecasters=3)
        question = _make_numeric_question(lower_bound=0.0, upper_bound=10.0, open_lower=False, open_upper=False)

        predictions = [
            ReasonedPrediction(
                prediction_value=_build_numeric_distribution(
                    5.0, scale=0.8, bounds=(0.0, 10.0), open_lower=False, open_upper=False
                ),
                reasoning="Model: m1\n\nOUTCOME_TYPE: DISCRETE",
            ),
            ReasonedPrediction(
                prediction_value=_build_numeric_distribution(
                    5.2, scale=0.8, bounds=(0.0, 10.0), open_lower=False, open_upper=False
                ),
                reasoning="Model: m2\n\nOUTCOME_TYPE: DISCRETE",
            ),
            ReasonedPrediction(
                prediction_value=_build_numeric_distribution(
                    4.8, scale=0.8, bounds=(0.0, 10.0), open_lower=False, open_upper=False
                ),
                reasoning="Model: m3\n\nOUTCOME_TYPE: DISCRETE",
            ),
        ]

        qid = question.id_of_question
        bot._discrete_integer_votes[qid] = [True, True, True]

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="Canned research") as _,
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

        aggregated = await bot._aggregate_predictions([p.prediction_value for p in result.predictions], question)
        assert isinstance(aggregated, NumericDistribution)
        assert qid not in bot._discrete_integer_votes, (
            "_maybe_snap_to_integers should pop the votes after consuming them"
        )


class TestNumericUnitMismatch:
    """Unit mismatch detection raises UnitMismatchError for values wildly outside bounds."""

    @pytest.mark.e2e
    def test_numeric_unit_mismatch_raises(self):
        question = _make_numeric_question(lower_bound=0.0, upper_bound=20_000_000.0)

        bad_percentiles = [
            Percentile(percentile=pct, value=val)
            for pct, val in zip(
                STANDARD_PERCENTILES,
                [3.2, 3.5, 3.8, 4.1, 4.5, 4.8, 5.1, 5.8, 6.5, 7.2, 8.0],
            )
        ]

        from metaculus_bot.numeric.validation import detect_unit_mismatch

        mismatch, reason = detect_unit_mismatch(bad_percentiles, question)
        assert mismatch is True
        assert reason != ""


class TestNumericHighSpreadTriggersStacking:
    """High numeric spread (normalized percentile spread > 0.15) triggers the stacker."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_numeric_high_spread_triggers_stacking(self):
        bot = _make_bot(n_forecasters=3)
        question = _make_numeric_question()

        predictions = [
            ReasonedPrediction(
                prediction_value=_build_numeric_distribution(3.0, scale=0.5),
                reasoning="Model: m1\n\nLow estimate",
            ),
            ReasonedPrediction(
                prediction_value=_build_numeric_distribution(12.0, scale=0.5),
                reasoning="Model: m2\n\nHigh estimate",
            ),
            ReasonedPrediction(
                prediction_value=_build_numeric_distribution(7.0, scale=0.5),
                reasoning="Model: m3\n\nMiddle estimate",
            ),
        ]

        stacked_dist = _build_numeric_distribution(7.5, scale=0.8)

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="Canned research") as _,
            patch.object(bot, "_gather_predictions_with_wall_clock") as mock_gather,
            patch.object(
                bot,
                "_forecaster_with_soft_deadline",
                new=AsyncMock(return_value=predictions[0]),
            ),
            patch(
                "metaculus_bot.forecaster.extract_disagreement_crux", new_callable=AsyncMock, return_value="Crux text"
            ) as mock_crux,
            patch(
                "metaculus_bot.forecaster.run_targeted_search", new_callable=AsyncMock, return_value="Targeted results"
            ) as mock_search,
            patch.object(bot, "_run_stacking", return_value=stacked_dist) as mock_stacking,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = (predictions, [], None)

            result = await bot._research_and_make_predictions(question)

        mock_crux.assert_called_once()
        mock_search.assert_called_once()
        mock_stacking.assert_called_once()

        assert len(result.predictions) == 1
        assert bot._conditional_stacking_triggered_count == 1


# ---------------------------------------------------------------------------
# MC Pipeline Tests
# ---------------------------------------------------------------------------


class TestMCLowSpreadMedianPath:
    """Low MC spread uses the MEDIAN aggregation path."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_mc_low_spread_median_path(self):
        bot = _make_bot(n_forecasters=3)
        question = _make_mc_question()

        predictions = [
            ReasonedPrediction(
                prediction_value=_mc_option_list([0.50, 0.30, 0.20]),
                reasoning="Model: m1\n\nRed leads",
            ),
            ReasonedPrediction(
                prediction_value=_mc_option_list([0.48, 0.32, 0.20]),
                reasoning="Model: m2\n\nRed leads",
            ),
            ReasonedPrediction(
                prediction_value=_mc_option_list([0.52, 0.28, 0.20]),
                reasoning="Model: m3\n\nRed leads",
            ),
        ]

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="Canned research") as _,
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
        assert bot._conditional_stacking_skipped_count == 1

        aggregated = await bot._aggregate_predictions([p.prediction_value for p in result.predictions], question)
        assert isinstance(aggregated, PredictedOptionList)

        total_prob = sum(o.probability for o in aggregated.predicted_options)
        assert abs(total_prob - 1.0) < 0.01

        probs_by_name = {o.option_name: o.probability for o in aggregated.predicted_options}
        assert abs(probs_by_name["Red"] - 0.50) < 0.05
        assert abs(probs_by_name["Blue"] - 0.30) < 0.05
        assert abs(probs_by_name["Green"] - 0.20) < 0.05


class TestMCClampingAndRenormalization:
    """MC clamping ensures no option exceeds bounds and probabilities sum to 1.0."""

    @pytest.mark.e2e
    def test_mc_clamping_and_renormalization(self):
        from metaculus_bot.numeric.utils import clamp_and_renormalize_mc

        extreme_prediction = _mc_option_list([0.99, 0.005, 0.005])
        clamped = clamp_and_renormalize_mc(extreme_prediction)

        for opt in clamped.predicted_options:
            assert opt.probability <= 0.995
            assert opt.probability >= 0.005

        total = sum(o.probability for o in clamped.predicted_options)
        assert abs(total - 1.0) < 1e-6


class TestMCHighSpreadTriggersStacking:
    """High MC spread (max option spread > 0.20) triggers the stacker."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_mc_high_spread_triggers_stacking(self):
        bot = _make_bot(n_forecasters=3)
        question = _make_mc_question()

        predictions = [
            ReasonedPrediction(
                prediction_value=_mc_option_list([0.80, 0.10, 0.10]),
                reasoning="Model: m1\n\nRed dominates",
            ),
            ReasonedPrediction(
                prediction_value=_mc_option_list([0.20, 0.60, 0.20]),
                reasoning="Model: m2\n\nBlue dominates",
            ),
            ReasonedPrediction(
                prediction_value=_mc_option_list([0.30, 0.30, 0.40]),
                reasoning="Model: m3\n\nGreen leads",
            ),
        ]

        stacked_result = _mc_option_list([0.45, 0.35, 0.20])

        with (
            patch.object(bot, "_get_notepad") as mock_notepad,
            patch.object(bot, "run_research", return_value="Canned research") as _,
            patch.object(bot, "_gather_predictions_with_wall_clock") as mock_gather,
            patch.object(
                bot,
                "_forecaster_with_soft_deadline",
                new=AsyncMock(return_value=predictions[0]),
            ),
            patch(
                "metaculus_bot.forecaster.extract_disagreement_crux", new_callable=AsyncMock, return_value="MC Crux"
            ) as mock_crux,
            patch(
                "metaculus_bot.forecaster.run_targeted_search", new_callable=AsyncMock, return_value="MC Targeted"
            ) as mock_search,
            patch.object(bot, "_run_stacking", return_value=stacked_result) as mock_stacking,
        ):
            mock_notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            mock_gather.return_value = (predictions, [], None)

            result = await bot._research_and_make_predictions(question)

        mock_crux.assert_called_once()
        mock_search.assert_called_once()
        mock_stacking.assert_called_once()

        assert len(result.predictions) == 1
        assert bot._conditional_stacking_triggered_count == 1


# ---------------------------------------------------------------------------
# Cross-cutting Tests
# ---------------------------------------------------------------------------


class TestMultiQuestionBatchPartialFailure:
    """One question type failing does not block other question types in a batch."""

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_multi_question_batch_one_fails_others_succeed(self):
        bot = _make_bot(n_forecasters=3)
        binary_q = _make_binary_question()
        numeric_q = _make_numeric_question()
        mc_q = _make_mc_question()

        binary_preds = [
            ReasonedPrediction(prediction_value=0.35, reasoning="Model: m1\n\nBinary reasoning"),
            ReasonedPrediction(prediction_value=0.40, reasoning="Model: m2\n\nBinary reasoning"),
            ReasonedPrediction(prediction_value=0.38, reasoning="Model: m3\n\nBinary reasoning"),
        ]
        mc_preds = [
            ReasonedPrediction(
                prediction_value=_mc_option_list([0.50, 0.30, 0.20]),
                reasoning="Model: m1\n\nMC reasoning",
            ),
            ReasonedPrediction(
                prediction_value=_mc_option_list([0.48, 0.32, 0.20]),
                reasoning="Model: m2\n\nMC reasoning",
            ),
            ReasonedPrediction(
                prediction_value=_mc_option_list([0.52, 0.28, 0.20]),
                reasoning="Model: m3\n\nMC reasoning",
            ),
        ]

        def mock_research_predictions(question):
            from forecasting_tools.data_models.forecast_report import ResearchWithPredictions

            if isinstance(question, NumericQuestion):
                raise RuntimeError("All numeric forecasters failed")
            elif isinstance(question, BinaryQuestion):
                fut: asyncio.Future = asyncio.get_event_loop().create_future()
                fut.set_result(
                    ResearchWithPredictions(
                        research_report="binary research",
                        summary_report="binary research",
                        errors=[],
                        predictions=binary_preds,
                    )
                )
                return fut.result()
            elif isinstance(question, MultipleChoiceQuestion):
                fut2: asyncio.Future = asyncio.get_event_loop().create_future()
                fut2.set_result(
                    ResearchWithPredictions(
                        research_report="mc research",
                        summary_report="mc research",
                        errors=[],
                        predictions=mc_preds,
                    )
                )
                return fut2.result()
            raise ValueError(f"Unexpected question type: {type(question)}")

        with patch.object(bot, "_research_and_make_predictions", side_effect=mock_research_predictions):
            results = await asyncio.gather(
                bot._research_and_make_predictions(binary_q),
                bot._research_and_make_predictions(mc_q),
                return_exceptions=True,
            )
            numeric_result = None
            try:
                numeric_result = await bot._research_and_make_predictions(numeric_q)
            except RuntimeError:
                pass

        binary_result, mc_result = results[0], results[1]

        assert not isinstance(binary_result, BaseException)
        assert len(binary_result.predictions) == 3

        assert not isinstance(mc_result, BaseException)
        assert len(mc_result.predictions) == 3

        assert numeric_result is None


class TestCommentContainsExpectedMarkers:
    """After stacking, the explanation string contains expected markers."""

    @pytest.mark.e2e
    def test_comment_contains_expected_markers(self):
        bot = _make_bot(n_forecasters=2)
        question = _make_binary_question()

        predictions = [
            ReasonedPrediction(prediction_value=0.10, reasoning="Model: m1\n\nLow"),
            ReasonedPrediction(prediction_value=0.85, reasoning="Model: m2\n\nHigh"),
        ]

        bot._stacker_outcome[question.id_of_question] = "primary"

        from forecasting_tools.data_models.forecast_report import ResearchWithPredictions

        collection = ResearchWithPredictions(
            research_report="Research text",
            summary_report="Research text",
            errors=[],
            predictions=predictions,
        )

        explanation = bot._create_unified_explanation(
            question,
            [collection],
            aggregated_prediction=0.45,
            final_cost=0.0,
            time_spent_in_minutes=1.0,
        )

        assert "STACKER_OUTCOME=primary" in explanation
        assert "STACKED=true" in explanation
        assert "TOOLS_USED=" in explanation
