"""Test correlation analysis with the new ensemble naming convention."""

import hashlib
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.multiple_choice_report import PredictedOption, PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.ensemble_analysis.correlation_analysis import CorrelationAnalyzer
from metaculus_bot.numeric.config import STANDARD_PERCENTILES


def _report(question_id: int, prediction: Any) -> SimpleNamespace:
    return SimpleNamespace(
        question=SimpleNamespace(
            id_of_question=question_id,
            page_url=f"https://example.com/{question_id}",
            community_prediction_at_access_time=None,
        ),
        prediction=prediction,
        explanation="reasoning",
        expected_baseline_score=10.0,
        price_estimate=0.01,
    )


def _binary_report(question_id: int, probability: float) -> SimpleNamespace:
    return _report(question_id, probability)


def _mc_report(question_id: int, option_probs: dict[str, float]) -> SimpleNamespace:
    options = PredictedOptionList(
        predicted_options=[PredictedOption(option_name=name, probability=p) for name, p in option_probs.items()]
    )
    return _report(question_id, options)


def _numeric_report(question_id: int, percentile_values: dict[float, float]) -> SimpleNamespace:
    prediction = NumericDistribution(
        declared_percentiles=[Percentile(percentile=p, value=value) for p, value in percentile_values.items()],
        open_upper_bound=False,
        open_lower_bound=False,
        upper_bound=100.0,
        lower_bound=0.0,
        zero_point=None,
        cdf_size=201,
    )
    return _report(question_id, prediction)


def _analyzer_over(reports_by_model: dict[str, list[SimpleNamespace]]) -> CorrelationAnalyzer:
    """Analyzer loaded with one benchmark per model, each holding the given reports."""
    benchmarks = [
        SimpleNamespace(
            name=model_name,
            total_cost=0.05,
            average_expected_baseline_score=12.0,
            forecast_reports=reports,
            forecast_bot_config={"llms": {"default": {"model": f"openrouter/{model_name}"}}},
        )
        for model_name, reports in reports_by_model.items()
    ]
    analyzer = CorrelationAnalyzer()
    analyzer.add_benchmark_results(cast("list[BenchmarkForBot]", benchmarks))
    return analyzer


def test_extract_model_name_with_new_ensemble_naming():
    """Test that _extract_model_name works with new ensemble bot names."""
    analyzer = CorrelationAnalyzer()

    # Test single model bot
    single_benchmark = Mock()
    single_benchmark.name = "qwen3-235b"
    single_benchmark.forecast_bot_config = {"llms": {"forecasters": []}}

    result = analyzer._extract_model_name(single_benchmark)
    assert result == "qwen3-235b"

    # Test ensemble bot with mean aggregation
    ensemble_mean_benchmark = Mock()
    ensemble_mean_benchmark.name = "qwen3_glm_mean"
    ensemble_mean_benchmark.forecast_bot_config = {"llms": {"forecasters": []}}

    result = analyzer._extract_model_name(ensemble_mean_benchmark)
    assert result == "qwen3_glm_mean"

    # Test ensemble bot with median aggregation
    ensemble_median_benchmark = Mock()
    ensemble_median_benchmark.name = "qwen3_glm_median"
    ensemble_median_benchmark.forecast_bot_config = {"llms": {"forecasters": []}}

    result = analyzer._extract_model_name(ensemble_median_benchmark)
    assert result == "qwen3_glm_median"


def test_extract_model_name_legacy_fallback():
    """Test that _extract_model_name falls back to legacy behavior for unknown patterns."""
    analyzer = CorrelationAnalyzer()

    # Test legacy benchmark that doesn't match new patterns
    legacy_benchmark = Mock()
    legacy_benchmark.name = "Legacy Bot | Config | some-unknown-model"
    legacy_benchmark.forecast_bot_config = {"llms": {"forecasters": [{"model": "openrouter/unknown/model-xyz"}]}}

    result = analyzer._extract_model_name(legacy_benchmark)
    # Should extract from the single forecaster model (legacy behavior)
    assert result == "model-xyz"


def test_extract_model_name_ensemble_from_forecasters():
    """Test ensemble name generation from forecaster list when bot name is not available."""
    analyzer = CorrelationAnalyzer()

    # Test ensemble without explicit bot name but with multiple forecasters
    ensemble_benchmark = Mock()
    ensemble_benchmark.name = "Unknown Ensemble"
    ensemble_benchmark.forecast_bot_config = {
        "llms": {
            "forecasters": [
                {"model": "openrouter/qwen/qwen3-235b-a22b-thinking-2507"},
                {"model": "openrouter/z-ai/glm-4.5"},
            ]
        },
        "aggregation_strategy": AggregationStrategy.MEAN,
    }

    result = analyzer._extract_model_name(ensemble_benchmark)
    # Should generate ensemble name from components
    assert result == "glm_qwen3_mean"  # sorted alphabetically


def test_unidentified_model_name_uses_a_process_stable_digest():
    analyzer = CorrelationAnalyzer()
    benchmark = Mock()
    benchmark.name = "Two Fields Only"
    benchmark.forecast_bot_config = {"llms": {}}

    expected = f"model_{hashlib.sha256(benchmark.name.encode()).hexdigest()[:12]}"

    assert analyzer._extract_model_name(benchmark) == expected


def test_unsupported_prediction_is_excluded_from_analysis():
    analyzer = _analyzer_over(
        {
            "model-a": [_report(1, object())],
            "model-b": [_binary_report(1, 0.42)],
        }
    )

    assert [prediction.model_name for prediction in analyzer.predictions] == ["model-b"]
    assert analyzer._extract_prediction_value(_report(1, object())) is None


def test_numeric_components_use_framework_fractional_percentile_labels():
    values = {percentile: percentile * 100.0 for percentile in STANDARD_PERCENTILES}
    analyzer = CorrelationAnalyzer()
    extracted = analyzer._extract_prediction_components(_numeric_report(1, values))

    assert extracted is not None
    question_type, components = extracted
    assert question_type == "numeric"
    assert components == pytest.approx([values[p] for p in STANDARD_PERCENTILES if 0.1 <= p <= 0.9 and p != 0.5])


def test_numeric_components_reject_missing_required_percentiles():
    analyzer = CorrelationAnalyzer()
    extracted = analyzer._extract_prediction_components(_numeric_report(1, {0.1: 10.0, 0.9: 90.0}))

    assert extracted is None


class TestComponentWiseCorrelation:
    """Pins ``calculate_correlation_matrix_by_components`` — the mixed-type path.

    Per question it pulls each model's prediction down to a component vector and
    correlates the vectors, then averages the per-question correlations. Which
    questions contribute (and with what correlation) is the whole contract: a
    question seen by fewer than two models, or whose models disagree about the
    question TYPE, contributes nothing while still counting toward ``num_questions``.
    """

    def test_multiple_choice_correlates_option_vectors_in_name_order(self):
        analyzer = _analyzer_over(
            {
                "model-a": [_mc_report(1, {"a": 0.5, "b": 0.3, "c": 0.2})],
                "model-b": [_mc_report(1, {"a": 0.2, "b": 0.3, "c": 0.5})],
            }
        )
        matrix = analyzer.calculate_correlation_matrix_by_components()

        assert matrix.num_questions == 1
        assert sorted(matrix.model_names) == ["model-a", "model-b"]
        assert matrix.pearson_matrix.loc["model-a", "model-a"] == pytest.approx(1.0)
        # Near-perfectly anti-correlated option vectors ([.5,.3,.2] vs [.2,.3,.5]).
        assert matrix.pearson_matrix.loc["model-a", "model-b"] == pytest.approx(-0.9285714, abs=1e-6)
        assert matrix.pearson_matrix.loc["model-b", "model-a"] == pytest.approx(-0.9285714, abs=1e-6)

    def test_binary_agreement_is_one_and_disagreement_is_zero(self):
        agreeing = _analyzer_over({"model-a": [_binary_report(1, 0.42)], "model-b": [_binary_report(1, 0.42)]})
        disagreeing = _analyzer_over({"model-a": [_binary_report(1, 0.42)], "model-b": [_binary_report(1, 0.43)]})

        assert agreeing.calculate_correlation_matrix_by_components().pearson_matrix.loc[
            "model-a", "model-b"
        ] == pytest.approx(1.0)
        assert disagreeing.calculate_correlation_matrix_by_components().pearson_matrix.loc[
            "model-a", "model-b"
        ] == pytest.approx(0.0)

    def test_constant_option_vectors_score_zero_not_nan(self):
        analyzer = _analyzer_over(
            {
                "model-a": [_mc_report(1, {"a": 0.5, "b": 0.5})],
                "model-b": [_mc_report(1, {"a": 0.2, "b": 0.8})],
            }
        )
        matrix = analyzer.calculate_correlation_matrix_by_components()

        assert matrix.pearson_matrix.loc["model-a", "model-b"] == pytest.approx(0.0)

    def test_question_seen_by_one_model_contributes_no_correlation(self):
        analyzer = _analyzer_over(
            {
                "model-a": [
                    _mc_report(1, {"a": 0.5, "b": 0.3, "c": 0.2}),
                    _mc_report(2, {"a": 0.9, "b": 0.05, "c": 0.05}),
                ],
                "model-b": [_mc_report(1, {"a": 0.2, "b": 0.3, "c": 0.5})],
            }
        )
        matrix = analyzer.calculate_correlation_matrix_by_components()

        # q2 still counts as a question seen, but only q1's correlation is averaged.
        assert matrix.num_questions == 2
        assert matrix.pearson_matrix.loc["model-a", "model-b"] == pytest.approx(-0.9285714, abs=1e-6)

    def test_type_disagreement_on_one_question_is_skipped_with_a_warning(self, caplog):
        analyzer = _analyzer_over(
            {
                "model-a": [_mc_report(1, {"a": 0.5, "b": 0.3, "c": 0.2})],
                "model-b": [_binary_report(1, 0.42)],
            }
        )
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.ensemble_analysis.correlation_analysis"):
            matrix = analyzer.calculate_correlation_matrix_by_components()

        assert "has mixed types across models" in caplog.text
        assert matrix.pearson_matrix.loc["model-a", "model-b"] == pytest.approx(0.0)


class TestCorrelationReport:
    """Pins the section skeleton of ``generate_correlation_report``."""

    def test_no_predictions_returns_the_no_data_sentence(self):
        assert CorrelationAnalyzer().generate_correlation_report() == (
            "No prediction data available for correlation analysis."
        )

    def test_mixed_types_render_the_component_method_and_type_breakdown(self):
        analyzer = _analyzer_over(
            {
                "model-a": [_binary_report(1, 0.42), _mc_report(2, {"a": 0.5, "b": 0.5})],
                "model-b": [_binary_report(1, 0.43), _mc_report(2, {"a": 0.2, "b": 0.8})],
            }
        )
        report = analyzer.generate_correlation_report()

        assert "# Model Correlation Analysis Report" in report
        assert "## Question Type Distribution" in report
        assert "**Analysis Method**: Component-wise correlation" in report
        assert "## Individual Model Performance" in report
        assert "## Model Correlations (Pearson)" in report
        assert "model-a" in report
        assert "model-b" in report

    def test_applied_filters_are_disclosed_in_the_report(self):
        analyzer = _analyzer_over(
            {
                "model-a": [_binary_report(1, 0.42), _mc_report(2, {"a": 0.5, "b": 0.5})],
                "model-b": [_binary_report(1, 0.43), _mc_report(2, {"a": 0.2, "b": 0.8})],
            }
        )
        analyzer.filter_models_inplace(exclude=["nothing-matches-this"])
        report = analyzer.generate_correlation_report()

        assert "## Filters Applied" in report
        assert "Remaining models: model-a, model-b" in report

    def test_report_is_written_to_disk_when_a_path_is_given(self, tmp_path):
        analyzer = _analyzer_over(
            {
                "model-a": [_binary_report(1, 0.42)],
                "model-b": [_binary_report(1, 0.43)],
            }
        )
        out = tmp_path / "correlations.md"
        report = analyzer.generate_correlation_report(output_path=str(out))

        assert out.read_text() == report


if __name__ == "__main__":
    pytest.main([__file__])
