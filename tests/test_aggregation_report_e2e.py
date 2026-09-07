"""Exact final-report contracts for numeric and multiple-choice aggregation.

These tests drive the public ``forecast_questions`` entrypoint through the
framework's per-question lifecycle.  Two research reports each run a real
two-model fan-out, route their members, and contribute to the framework's final
combine before the real report and comment builders run.  Only research and the
external model call boundary are replaced with deterministic local responses.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from typing import cast

import pytest
from forecasting_tools import (
    GeneralLlm,
    MultipleChoiceQuestion,
    MultipleChoiceReport,
    NumericDistribution,
    NumericQuestion,
    NumericReport,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.comment.markers import (
    FORECASTERS_USED_MARKER_RE,
    STACKER_OUTCOME_SKIPPED,
    STACKER_OUTCOME_SKIPPED_CONFIG_OFF,
    STACKER_SKIP_REASON_RE,
)
from metaculus_bot.constants import MC_PROB_MAX, MC_PROB_MIN
from metaculus_bot.forecaster import TemplateForecaster
from metaculus_bot.numeric.config import PCHIP_CDF_POINTS
from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution
from metaculus_bot.time_budget import QuestionTimeBudget
from tests.pipeline_test_helpers import make_real_mc_question, make_real_numeric_question

_NUMERIC_QID = 47001
_MC_QID = 47002
_MODEL_NAMES = ("test/forecaster-one", "test/forecaster-two")
_REPORT_NAMES = ("report-one research", "report-two research")


def _bot() -> TemplateForecaster:
    forecasters = [GeneralLlm(model=name, temperature=0.0) for name in _MODEL_NAMES]
    support_llm = GeneralLlm(model="test/support", temperature=0.0)
    llms = cast(
        "dict[str, str | GeneralLlm]",
        {
            "forecasters": forecasters,
            "stacker": support_llm,
            "analyzer": support_llm,
            "default": forecasters[0],
            "parser": support_llm,
            "researcher": support_llm,
            "summarizer": support_llm,
        },
    )
    return TemplateForecaster(
        research_reports_per_question=2,
        predictions_per_research_report=1,
        publish_reports_to_metaculus=False,
        aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
        llms=llms,
        is_benchmarking=True,
        min_forecasters_to_publish=2,
        stacking_fallback_on_failure=True,
        stacking_randomize_order=False,
    )


def _research_stub() -> tuple[Callable[..., Awaitable[str]], list[int]]:
    call_qids: list[int] = []

    async def run_research(
        question: NumericQuestion | MultipleChoiceQuestion,
        time_budget: QuestionTimeBudget | None = None,
    ) -> str:
        del time_budget
        assert question.id_of_question is not None
        call_qids.append(question.id_of_question)
        return _REPORT_NAMES[len(call_qids) - 1]

    return run_research, call_qids


def _numeric_member(question: NumericQuestion, probabilities: Sequence[float]) -> NumericDistribution:
    grid = [
        question.lower_bound + (question.upper_bound - question.lower_bound) * index / (len(probabilities) - 1)
        for index in range(len(probabilities))
    ]
    declared = [
        Percentile(percentile=float(probability), value=float(value))
        for value, probability in zip(grid, probabilities, strict=True)
    ]
    return create_pchip_numeric_distribution(
        pchip_cdf=list(probabilities),
        percentile_list=declared,
        question=question,
        zero_point=question.zero_point,
    )


def _smooth_cdf(linear_weight: float) -> list[float]:
    grid = [index / (PCHIP_CDF_POINTS - 1) for index in range(PCHIP_CDF_POINTS)]
    return [linear_weight * value + (1.0 - linear_weight) * value**2 for value in grid]


def _option_list(option_probabilities: Sequence[tuple[str, float]]) -> PredictedOptionList:
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=option_name, probability=probability)
            for option_name, probability in option_probabilities
        ]
    )


def _assert_qid_state_consumed(bot: TemplateForecaster, qid: int) -> None:
    assert qid not in bot._pipeline.outcomes
    assert qid not in bot._pipeline.skip_reasons
    assert qid not in bot._pipeline.meta_reasoning
    assert qid not in bot._contributing_forecasters
    assert qid not in bot._pipeline.expected_base_combines


def _assert_common_report_contract(
    bot: TemplateForecaster,
    report: ForecastReport,
    qid: int,
    *,
    expected_skip_reason: str,
) -> None:
    used = FORECASTERS_USED_MARKER_RE.search(report.explanation)
    assert used is not None
    assert used.groups() == ("4", "2")
    skip_reason = STACKER_SKIP_REASON_RE.search(report.explanation)
    assert skip_reason is not None
    assert skip_reason.group(1) == expected_skip_reason
    expected_outcome = (
        STACKER_OUTCOME_SKIPPED_CONFIG_OFF if expected_skip_reason == "config_off" else STACKER_OUTCOME_SKIPPED
    )
    assert expected_outcome in report.explanation
    assert "## Report 1 Summary" in report.explanation
    assert "## Report 2 Summary" in report.explanation
    assert bot._pipeline.counters.stacking_expected_combine_count == 1
    assert bot._pipeline.counters.stacking_unexpected_combine_count == 0
    _assert_qid_state_consumed(bot, qid)


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(20)
async def test_numeric_two_report_fanout_produces_exact_pointwise_median_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = make_real_numeric_question(
        _NUMERIC_QID,
        lower_bound=0.0,
        upper_bound=20.0,
        open_lower_bound=False,
        open_upper_bound=False,
    )
    bot = _bot()
    run_research, research_qids = _research_stub()
    member_cdfs = {
        (_REPORT_NAMES[0], _MODEL_NAMES[0]): _smooth_cdf(0.55),
        (_REPORT_NAMES[0], _MODEL_NAMES[1]): _smooth_cdf(0.85),
        (_REPORT_NAMES[1], _MODEL_NAMES[0]): _smooth_cdf(0.95),
        (_REPORT_NAMES[1], _MODEL_NAMES[1]): _smooth_cdf(1.20),
    }
    model_calls: list[tuple[str, str]] = []

    async def model_call(
        called_question: NumericQuestion,
        research: str,
        llm: GeneralLlm,
        _chart_b64: str | None = None,
    ) -> ReasonedPrediction[NumericDistribution]:
        assert called_question is question
        model_calls.append((research, llm.model))
        prediction = _numeric_member(question, member_cdfs[(research, llm.model)])
        return ReasonedPrediction(
            prediction_value=prediction,
            reasoning=f"Model: {llm.model}\n\nDeterministic numeric rationale for {research}",
        )

    monkeypatch.setenv("NUMERIC_STACKING_ENABLED", "false")
    monkeypatch.setattr(bot, "run_research", run_research)
    monkeypatch.setattr(bot, "_make_prediction", model_call)

    reports = await bot.forecast_questions([question])

    assert research_qids == [_NUMERIC_QID, _NUMERIC_QID]
    assert Counter(model_calls) == Counter(member_cdfs.keys())
    assert len(model_calls) == len(member_cdfs)
    assert len(reports) == 1
    report = reports[0]
    assert isinstance(report, NumericReport)
    actual_cdf = [point.percentile for point in report.prediction.get_cdf()]
    expected_cdf = [statistics.median(values) for values in zip(*member_cdfs.values(), strict=True)]
    assert actual_cdf == pytest.approx(expected_cdf, abs=1e-12)
    assert len(actual_cdf) == PCHIP_CDF_POINTS
    midpoint = PCHIP_CDF_POINTS // 2
    mean_at_midpoint = statistics.mean(cdf[midpoint] for cdf in member_cdfs.values())
    report_one_median = statistics.median(
        cdf[midpoint] for key, cdf in member_cdfs.items() if key[0] == _REPORT_NAMES[0]
    )
    report_two_median = statistics.median(
        cdf[midpoint] for key, cdf in member_cdfs.items() if key[0] == _REPORT_NAMES[1]
    )
    assert actual_cdf[midpoint] != pytest.approx(mean_at_midpoint)
    assert actual_cdf[midpoint] != pytest.approx(0.5)
    assert actual_cdf[midpoint] != pytest.approx(report_one_median)
    assert actual_cdf[midpoint] != pytest.approx(report_two_median)
    _assert_common_report_contract(bot, report, _NUMERIC_QID, expected_skip_reason="spread_below_threshold")


@pytest.mark.asyncio
@pytest.mark.e2e
@pytest.mark.timeout(20)
async def test_mc_two_report_fanout_preserves_option_order_and_normalizes_final_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = make_real_mc_question(_MC_QID, options=["Alpha", "Beta", "Gamma", "Delta"])
    bot = _bot()
    run_research, research_qids = _research_stub()
    member_ballots = {
        (_REPORT_NAMES[0], _MODEL_NAMES[0]): [
            ("Alpha", 0.49),
            ("Beta", 0.49),
            ("Gamma", 0.01),
            ("Delta", 0.01),
        ],
        (_REPORT_NAMES[0], _MODEL_NAMES[1]): [
            ("Gamma", 0.49),
            ("Delta", 0.01),
            ("Alpha", 0.49),
            ("Beta", 0.01),
        ],
        (_REPORT_NAMES[1], _MODEL_NAMES[0]): [
            ("Beta", 0.49),
            ("Alpha", 0.01),
            ("Delta", 0.01),
            ("Gamma", 0.49),
        ],
        (_REPORT_NAMES[1], _MODEL_NAMES[1]): [
            ("Delta", 0.01),
            ("Gamma", 0.01),
            ("Beta", 0.49),
            ("Alpha", 0.49),
        ],
    }
    model_calls: list[tuple[str, str]] = []

    async def model_call(
        called_question: MultipleChoiceQuestion,
        research: str,
        llm: GeneralLlm,
        _chart_b64: str | None = None,
    ) -> ReasonedPrediction[PredictedOptionList]:
        assert called_question is question
        model_calls.append((research, llm.model))
        return ReasonedPrediction(
            prediction_value=_option_list(member_ballots[(research, llm.model)]),
            reasoning=f"Model: {llm.model}\n\nDeterministic MC rationale for {research}",
        )

    monkeypatch.setenv("MC_STACKING_ENABLED", "false")
    monkeypatch.setattr(bot, "run_research", run_research)
    monkeypatch.setattr(bot, "_make_prediction", model_call)

    reports = await bot.forecast_questions([question])

    assert research_qids == [_MC_QID, _MC_QID]
    assert Counter(model_calls) == Counter(member_ballots.keys())
    assert len(model_calls) == len(member_ballots)
    assert len(reports) == 1
    report = reports[0]
    assert isinstance(report, MultipleChoiceReport)
    actual_options = report.prediction.predicted_options
    assert [option.option_name for option in actual_options] == question.options

    ballots_by_name = [dict(ballot) for ballot in member_ballots.values()]
    raw_medians = [statistics.median(ballot[option] for ballot in ballots_by_name) for option in question.options]
    clamped_medians = [max(MC_PROB_MIN, min(MC_PROB_MAX, probability)) for probability in raw_medians]
    clamped_total = sum(clamped_medians)
    assert clamped_medians[-1] / clamped_total < MC_PROB_MIN
    expected_probabilities = [
        probability / sum(clamped_medians[:-1]) * (1.0 - MC_PROB_MIN) for probability in clamped_medians[:-1]
    ] + [MC_PROB_MIN]
    actual_probabilities = [option.probability for option in actual_options]
    assert actual_probabilities == pytest.approx(expected_probabilities, abs=1e-12)
    assert sum(actual_probabilities) == pytest.approx(1.0, abs=1e-12)
    assert all(MC_PROB_MIN <= probability <= MC_PROB_MAX for probability in actual_probabilities)
    assert actual_probabilities[-1] == pytest.approx(MC_PROB_MIN)
    _assert_common_report_contract(bot, report, _MC_QID, expected_skip_reason="config_off")
