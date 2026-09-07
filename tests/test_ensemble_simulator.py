"""Behavior pins for ``ensemble_analysis.ensemble_simulator.EnsembleSimulator``.

The simulator answers "how would this set of models have scored together": it
groups benchmark reports by question, aggregates each question's predictions by
mean or median in the shape that question type demands, and hands the result to
the real baseline scorer. These tests pin the aggregation math and the dispatch
(which branch a question type takes, which questions are skipped, what the
soft-fail boundary swallows) by capturing what the scorer receives, so the
function can be decomposed without moving any of those decisions.
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
from forecasting_tools.data_models.multiple_choice_report import PredictedOption, PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.ensemble_analysis import ensemble_simulator as simulator_module
from metaculus_bot.ensemble_analysis.cdf_cache import NumericCdfCache
from metaculus_bot.ensemble_analysis.correlation_analysis import CorrelationAnalyzer
from metaculus_bot.ensemble_analysis.ensemble_simulator import EnsembleSimulator


def _question(qid: int, community: float | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        id_of_question=qid,
        page_url=f"https://example.com/{qid}",
        community_prediction_at_access_time=community,
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=False,
        open_upper_bound=False,
        zero_point=None,
        cdf_size=201,
    )


def _report(question: SimpleNamespace, prediction: Any) -> SimpleNamespace:
    return SimpleNamespace(
        question=question,
        prediction=prediction,
        explanation="reasoning",
        expected_baseline_score=10.0,
        price_estimate=0.01,
    )


def _benchmark(model_name: str, reports: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(
        name=model_name,
        total_cost=0.05,
        forecast_reports=reports,
        forecast_bot_config={"llms": {"default": {"model": f"openrouter/{model_name}"}}},
    )


def _simulator(benchmarks: list[SimpleNamespace]) -> EnsembleSimulator:
    """Simulator over a stand-in analyzer; only ``.benchmarks`` is ever read off it."""
    analyzer = cast("CorrelationAnalyzer", SimpleNamespace(benchmarks=benchmarks))
    return EnsembleSimulator(analyzer, NumericCdfCache())


def _binary_baseline(p: float, community: float) -> float:
    return 100.0 * (community * (math.log2(p) + 1.0) + (1.0 - community) * (math.log2(1.0 - p) + 1.0))


def _option_list(probs: dict[str, float]) -> PredictedOptionList:
    return PredictedOptionList(
        predicted_options=[PredictedOption(option_name=name, probability=p) for name, p in probs.items()]
    )


def _numeric(p10: float, p50: float, p90: float) -> NumericDistribution:
    return NumericDistribution(
        declared_percentiles=[
            Percentile(value=p10, percentile=0.1),
            Percentile(value=p50, percentile=0.5),
            Percentile(value=p90, percentile=0.9),
        ],
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
        cdf_size=201,
    )


class TestBinarySimulation:
    """The binary branch aggregates scalar probabilities and scores them directly."""

    def _sim(self) -> EnsembleSimulator:
        question = _question(1, community=0.6)
        return _simulator(
            [
                _benchmark("model-a", [_report(question, 0.2)]),
                _benchmark("model-b", [_report(question, 0.4)]),
                _benchmark("model-c", [_report(question, 0.9)]),
            ]
        )

    def test_mean_scores_the_mean_probability(self):
        score = self._sim().simulate_ensemble_performance(["model-a", "model-b", "model-c"], "mean")
        assert score == pytest.approx(_binary_baseline((0.2 + 0.4 + 0.9) / 3, 0.6))

    def test_median_scores_the_median_probability(self):
        score = self._sim().simulate_ensemble_performance(["model-a", "model-b", "model-c"], "median")
        assert score == pytest.approx(_binary_baseline(0.4, 0.6))

    def test_enum_strategy_matches_the_raw_string(self):
        as_enum = self._sim().simulate_ensemble_performance(["model-a", "model-b"], AggregationStrategy.MEDIAN)
        as_string = self._sim().simulate_ensemble_performance(["model-a", "model-b"], "median")
        assert as_enum == pytest.approx(as_string)

    def test_question_missing_a_member_is_skipped(self):
        shared = _question(1, community=0.6)
        solo = _question(2, community=0.6)
        sim = _simulator(
            [
                _benchmark("model-a", [_report(shared, 0.2), _report(solo, 0.7)]),
                _benchmark("model-b", [_report(shared, 0.4)]),
            ]
        )
        # Only q1 has both members, so only q1 contributes; q2 never enters the mean.
        assert sim.simulate_ensemble_performance(["model-a", "model-b"], "mean") == pytest.approx(
            _binary_baseline(0.3, 0.6)
        )

    def test_no_community_prediction_yields_no_score(self):
        question = _question(1, community=None)
        sim = _simulator(
            [
                _benchmark("model-a", [_report(question, 0.2)]),
                _benchmark("model-b", [_report(question, 0.4)]),
            ]
        )
        assert sim.simulate_ensemble_performance(["model-a", "model-b"], "mean") == 0.0


class TestMultipleChoiceSimulation:
    """The MC branch aggregates per option, in the first prediction's option order."""

    def _run(self, monkeypatch: pytest.MonkeyPatch, strategy: str) -> list[tuple[str, float]]:
        captured: list[tuple[str, float]] = []

        def _capture(report: Any, _cache: dict) -> float:
            captured.extend((opt.option_name, opt.probability) for opt in report.prediction.predicted_options)
            return 12.0

        monkeypatch.setattr(simulator_module, "calculate_multiple_choice_baseline_score", _capture)
        question = _question(1)
        sim = _simulator(
            [
                _benchmark("model-a", [_report(question, _option_list({"yes": 0.6, "maybe": 0.3, "no": 0.1}))]),
                _benchmark("model-b", [_report(question, _option_list({"yes": 0.2, "maybe": 0.3, "no": 0.5}))]),
                _benchmark("model-c", [_report(question, _option_list({"yes": 0.7, "maybe": 0.2, "no": 0.1}))]),
            ]
        )
        assert sim.simulate_ensemble_performance(["model-a", "model-b", "model-c"], strategy) == pytest.approx(12.0)
        return captured

    def test_mean_aggregates_each_option_and_keeps_declared_order(self, monkeypatch: pytest.MonkeyPatch):
        captured = self._run(monkeypatch, "mean")
        assert [name for name, _ in captured] == ["yes", "maybe", "no"]
        assert [p for _, p in captured] == pytest.approx([0.5, 0.26666667, 0.23333333])

    def test_median_aggregates_each_option_then_renormalizes(self, monkeypatch: pytest.MonkeyPatch):
        captured = self._run(monkeypatch, "median")
        # Per-option medians are 0.6/0.3/0.1 → sum 1.0, so renormalization is a no-op here.
        assert [p for _, p in captured] == pytest.approx([0.6, 0.3, 0.1])

    def test_missing_member_option_is_unscoreable(self, monkeypatch: pytest.MonkeyPatch):
        question = _question(1)
        incomplete_prediction = PredictedOptionList.model_construct(
            predicted_options=[PredictedOption(option_name="yes", probability=0.2)]
        )
        sim = _simulator(
            [
                _benchmark("model-a", [_report(question, _option_list({"yes": 0.6, "no": 0.4}))]),
                _benchmark("model-b", [_report(question, incomplete_prediction)]),
            ]
        )
        monkeypatch.setattr(simulator_module, "calculate_multiple_choice_baseline_score", pytest.fail)

        assert sim.simulate_ensemble_performance(["model-a", "model-b"], "mean") == 0.0

    def test_zero_mass_member_is_unscoreable(self):
        question = _question(1)
        zero_mass_prediction = PredictedOptionList.model_construct(
            predicted_options=[
                PredictedOption(option_name="yes", probability=0.0),
                PredictedOption(option_name="no", probability=0.0),
            ]
        )
        sim = _simulator(
            [
                _benchmark("model-a", [_report(question, zero_mass_prediction)]),
                _benchmark("model-b", [_report(question, _option_list({"yes": 0.6, "no": 0.4}))]),
            ]
        )
        with pytest.raises(ValueError, match="no positive probability mass"):
            sim._score_multiple_choice_question(
                question,
                [zero_mass_prediction, _option_list({"yes": 0.6, "no": 0.4})],
                "mean",
            )


class TestNumericSimulation:
    """The numeric branch aggregates pointwise in CDF space off the safe-CDF ladder."""

    def test_mean_averages_the_member_cdfs_pointwise(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, Any] = {}

        def _capture(report: Any, _cache: dict) -> float:
            captured["cdf"] = list(report.prediction.cdf)
            return 7.0

        monkeypatch.setattr(simulator_module, "calculate_numeric_baseline_score", _capture)
        question = _question(1)
        low = _numeric(10.0, 30.0, 50.0)
        high = _numeric(50.0, 70.0, 90.0)
        sim = _simulator(
            [
                _benchmark("model-a", [_report(question, low)]),
                _benchmark("model-b", [_report(question, high)]),
            ]
        )

        assert sim.simulate_ensemble_performance(["model-a", "model-b"], "mean") == pytest.approx(7.0)
        aggregated = captured["cdf"]
        assert len(aggregated) == len(low.cdf)
        expected = [(a.percentile + b.percentile) / 2 for a, b in zip(low.cdf, high.cdf, strict=True)]
        assert [pt.percentile for pt in aggregated] == pytest.approx(expected)
        assert [pt.value for pt in aggregated] == pytest.approx([pt.value for pt in low.cdf])

    def test_each_member_cdf_is_fetched_under_its_own_model_name(self, monkeypatch: pytest.MonkeyPatch):
        """The safe-CDF cache is keyed (model_name, qid), so each member must arrive
        under its own name — not a reverse-inferred or shared one."""
        monkeypatch.setattr(simulator_module, "calculate_numeric_baseline_score", lambda *_a, **_k: 7.0)
        question = _question(1)
        low = _numeric(10.0, 30.0, 50.0)
        high = _numeric(50.0, 70.0, 90.0)
        sim = _simulator(
            [
                _benchmark("model-a", [_report(question, low)]),
                _benchmark("model-b", [_report(question, high)]),
            ]
        )
        seen: list[tuple[str, Any]] = []

        def _capture(**kw: Any) -> list[Any]:
            seen.append((kw["model_name"], kw["prediction"]))
            return []

        monkeypatch.setattr(sim._cdf_cache, "get_safe_numeric_cdf", _capture)
        sim.simulate_ensemble_performance(["model-a", "model-b"], "mean")

        assert seen == [("model-a", low), ("model-b", high)]

    def test_unusable_numeric_prediction_is_a_warning_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ):
        monkeypatch.setattr(simulator_module, "calculate_numeric_baseline_score", lambda *_a, **_k: 7.0)
        good = _question(1)
        broken = _question(2)
        sim = _simulator(
            [
                _benchmark(
                    "model-a", [_report(good, _numeric(10.0, 30.0, 50.0)), _report(broken, _numeric(1.0, 2.0, 3.0))]
                ),
                _benchmark(
                    "model-b", [_report(good, _numeric(50.0, 70.0, 90.0)), _report(broken, _numeric(4.0, 5.0, 6.0))]
                ),
            ]
        )
        monkeypatch.setattr(
            sim._cdf_cache, "get_safe_numeric_cdf", lambda **kw: None if kw["question"] is broken else []
        )

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.ensemble_analysis.ensemble_simulator"):
            score = sim.simulate_ensemble_performance(["model-a", "model-b"], "mean")

        # The unusable question is dropped; the run continues and still returns a score.
        assert score == pytest.approx(7.0)
        assert "Failed to aggregate predictions for question 2" in caplog.text
