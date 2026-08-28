"""Minimal tests for analyze_correlations.py CLI utility functions."""

import contextlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest


def test_import_analyze_correlations():
    """Test that the CLI script can be imported without errors."""
    import analyze_correlations  # Should not raise any import errors

    assert hasattr(analyze_correlations, "main")
    assert hasattr(analyze_correlations, "extract_timestamp_from_filename")
    assert hasattr(analyze_correlations, "load_benchmarks_from_path")


def test_extract_timestamp_from_filename():
    """Test timestamp extraction from various filename formats."""
    from analyze_correlations import extract_timestamp_from_filename

    # Standard format
    assert extract_timestamp_from_filename("benchmarks_2025-08-10_15-04-51.jsonl") == "2025-08-10_15-04-51"

    # With path
    assert extract_timestamp_from_filename("benchmarks/benchmarks_2025-12-25_23-59-59.json") == "2025-12-25_23-59-59"

    # No timestamp
    assert extract_timestamp_from_filename("simple.json") is None

    # Different format
    assert extract_timestamp_from_filename("other_2024-01-01_00-00-00_suffix.jsonl") == "2024-01-01_00-00-00"


def create_mock_benchmark_data():
    """Create mock benchmark data for binary questions only."""
    return {
        "forecast_bot_class_name": "TemplateForecaster",
        "name": "Test Bot | Model | test-model | 2025-08-11_12-00-00",
        "num_input_questions": 1,
        "timestamp": datetime.now().isoformat(),
        "time_taken_in_minutes": 5.0,
        "total_cost": 0.50,
        "average_expected_baseline_score": 15.5,
        "forecast_bot_config": {
            "llms": {
                "forecasters": [{"model": "openrouter/openai/gpt-4o"}],
                "default": {"model": "openrouter/openai/gpt-4o"},
            },
            "research_reports_per_question": 1,
            "predictions_per_research_report": 1,
        },
        "forecast_reports": [
            {
                "question": {
                    "id_of_question": 1,
                    "id_of_post": 1,
                    "page_url": "https://example.com/1",
                    "question_text": "Will this binary event happen?",
                    "background_info": "",
                    "resolution_criteria": "",
                    "fine_print": "",
                    "published_time": None,
                    "close_time": None,
                },
                "prediction": 0.6,
                "explanation": "# Binary Analysis\nThis is mock reasoning for correlation analysis.",
                "price_estimate": 0.25,
                "minutes_taken": 2.5,
                "expected_baseline_score": 15.5,
                "errors": [],
            }
        ],
        "failed_report_errors": [],
        "git_commit_hash": "abc123",
        "code": "# mock code",
    }


def test_load_benchmarks_from_json_file():
    """Test loading benchmarks from a single JSON file."""
    from analyze_correlations import load_benchmarks_from_path

    mock_data = create_mock_benchmark_data()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump([mock_data], f)  # List of benchmarks
        json_path = f.name

    try:
        benchmarks = load_benchmarks_from_path(json_path)
        assert len(benchmarks) == 1
        assert benchmarks[0].forecast_bot_class_name == "TemplateForecaster"
    finally:
        Path(json_path).unlink()  # Clean up


def test_load_benchmarks_from_jsonl_file():
    """Test loading benchmarks from a JSONL file."""
    from analyze_correlations import load_benchmarks_from_path

    mock_data1 = create_mock_benchmark_data()
    mock_data2 = create_mock_benchmark_data()
    mock_data2["total_cost"] = 0.30  # Different cost

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(mock_data1) + "\n")
        f.write(json.dumps(mock_data2) + "\n")
        jsonl_path = f.name

    try:
        benchmarks = load_benchmarks_from_path(jsonl_path)
        assert len(benchmarks) == 2
        assert benchmarks[0].total_cost == 0.50
        assert benchmarks[1].total_cost == 0.30
    finally:
        Path(jsonl_path).unlink()  # Clean up


def test_load_benchmarks_from_directory():
    """Test loading benchmarks from a directory with multiple files."""
    from analyze_correlations import load_benchmarks_from_path

    mock_data = create_mock_benchmark_data()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a JSON file
        json_file = Path(temp_dir) / "bench1.json"
        with open(json_file, "w") as f:
            json.dump([mock_data], f)

        # Create a JSONL file
        jsonl_file = Path(temp_dir) / "bench2.jsonl"
        with open(jsonl_file, "w") as f:
            f.write(json.dumps(mock_data) + "\n")

        benchmarks = load_benchmarks_from_path(temp_dir)
        assert len(benchmarks) == 2


def test_load_nonexistent_path():
    """Test loading from a path that doesn't exist."""
    from analyze_correlations import load_benchmarks_from_path

    benchmarks = load_benchmarks_from_path("/nonexistent/path")
    assert len(benchmarks) == 0


def test_argument_parsing():
    """Test CLI argument parsing."""
    import argparse

    # Create parser similar to main function
    parser = argparse.ArgumentParser(description="Analyze model correlations from benchmark results")
    parser.add_argument("benchmark_path", help="Path to benchmark file (.json/.jsonl) or directory")
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for correlation report (default: correlation_analysis.md)",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=1.0,
        help="Maximum cost per question for ensemble recommendations",
    )
    parser.add_argument("--max-size", type=int, default=5, help="Maximum ensemble size")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    # Test default args
    args = parser.parse_args(["benchmarks/"])
    assert args.benchmark_path == "benchmarks/"
    assert args.max_cost == 1.0
    assert args.max_size == 5
    assert args.verbose is False

    # Test custom args
    args = parser.parse_args(["test.jsonl", "--max-cost", "0.5", "--max-size", "3", "--verbose"])
    assert args.benchmark_path == "test.jsonl"
    assert args.max_cost == 0.5
    assert args.max_size == 3
    assert args.verbose is True


def _make_report(question_id: int, score: float, cost: float = 0.25) -> SimpleNamespace:
    question = SimpleNamespace(
        id_of_question=question_id,
        page_url=f"https://example.com/{question_id}",
    )
    return SimpleNamespace(
        question=question,
        expected_baseline_score=score,
        price_estimate=cost,
        prediction=None,
        errors=[],
    )


def _make_benchmark(name: str, model_label: str, scores: list[float]) -> SimpleNamespace:
    reports = [_make_report(1000 + idx, score) for idx, score in enumerate(scores)]
    return SimpleNamespace(
        forecast_bot_class_name="TemplateForecaster",
        name=name,
        num_input_questions=len(reports),
        timestamp=datetime.now().isoformat(),
        time_taken_in_minutes=5.0,
        total_cost=sum(r.price_estimate for r in reports),
        average_expected_baseline_score=sum(scores) / len(scores) if scores else 0.0,
        forecast_bot_config={
            "llms": {
                "default": {"model": model_label},
                "forecasters": [{"model": model_label}],
            }
        },
        forecast_reports=reports,
        failed_report_errors=[],
        git_commit_hash="abc123",
        code="# mock",
    )


@patch("analyze_correlations.CorrelationAnalyzer")
@patch("analyze_correlations.load_benchmarks_from_path")
def test_main_function_flow(mock_load, mock_analyzer_class):
    """Test main function flow without actual file operations."""
    import analyze_correlations

    # Mock the loading with realistic benchmark objects
    mock_benchmarks = [
        _make_benchmark("Bot A", "model-a", [12.0, -5.0, 3.5]),
        _make_benchmark("Bot B", "model-b", [10.0, -2.5, 4.0]),
    ]
    mock_load.return_value = mock_benchmarks

    # Mock the analyzer
    mock_analyzer = Mock()
    mock_analyzer.generate_correlation_report.return_value = "Mock report"
    mock_analyzer.find_optimal_ensembles.return_value = []
    mock_analyzer.filter_models_inplace.return_value = {}
    mock_analyzer._get_question_type.side_effect = lambda report: "binary"
    mock_analyzer._is_stacking_benchmark.return_value = False
    mock_analyzer.benchmarks = mock_benchmarks

    # Mock correlation matrix with proper get_least_correlated_pairs return value
    mock_corr_matrix = Mock()
    mock_corr_matrix.get_least_correlated_pairs.return_value = [
        ("model1", "model2", 0.1),
        ("model3", "model4", 0.2),
        ("model5", "model6", 0.3),
    ]
    mock_analyzer.calculate_correlation_matrix.return_value = mock_corr_matrix
    mock_analyzer.calculate_correlation_matrix_by_components.return_value = mock_corr_matrix
    mock_analyzer._has_mixed_question_types.return_value = False  # Use simple correlation
    mock_analyzer._get_question_type_breakdown.return_value = {"binary": 10}
    mock_analyzer_class.return_value = mock_analyzer

    # Mock sys.argv to avoid parsing real command line
    with (
        patch("sys.argv", ["analyze_correlations.py", "test.jsonl"]),
        contextlib.suppress(SystemExit),
    ):
        # Should run without errors
        analyze_correlations.main()


@patch("analyze_correlations.load_benchmarks_from_path")
def test_main_with_insufficient_benchmarks(mock_load):
    """Test main function with too few benchmarks."""
    import analyze_correlations

    # Mock loading only one benchmark (need 2+ for correlation)
    mock_load.return_value = [Mock()]

    with patch("sys.argv", ["analyze_correlations.py", "test.jsonl"]), pytest.raises(SystemExit):
        analyze_correlations.main()


def test_timestamped_output_filename():
    """Test that output filename includes timestamp from input file."""

    from analyze_correlations import extract_timestamp_from_filename

    input_file = "benchmarks/benchmarks_2025-08-10_15-04-51.jsonl"
    timestamp = extract_timestamp_from_filename(input_file)

    # Simulate the logic in main()
    filename = f"correlation_analysis_{timestamp}.md" if timestamp else "correlation_analysis.md"

    expected_filename = "correlation_analysis_2025-08-10_15-04-51.md"
    assert filename == expected_filename


# ---------------------------------------------------------------------------
# _ensemble_per_type — per-question-type aggregation
#
# These pin the aggregation ARITHMETIC (which is what the ensemble diagnostic
# actually asserts), not the downstream baseline scorers: the MC and numeric
# scorers are patched so the test can read back exactly what got handed to them.
# ---------------------------------------------------------------------------


def _stub_analyzer(qtype: str, cdfs_by_model: dict[str, list] | None = None) -> Any:
    """A _StubAnalyzer typed as Any: only the three hooks below are ever exercised."""
    return cast("Any", _StubAnalyzer(qtype, cdfs_by_model))


class _StubAnalyzer:
    """Minimal stand-in exposing only the three analyzer hooks _ensemble_per_type calls."""

    def __init__(self, qtype: str, cdfs_by_model: dict[str, list] | None = None) -> None:
        self._qtype = qtype
        self._cdfs_by_model = cdfs_by_model or {}

    def _extract_model_name(self, benchmark):
        return benchmark.model_name

    def _is_stacking_benchmark(self, benchmark):
        return False

    def _get_question_type(self, report):
        return self._qtype

    def _get_safe_numeric_cdf(self, model, question, prediction):
        return self._cdfs_by_model.get(model)


def _bench(model_name: str, reports: list) -> SimpleNamespace:
    return SimpleNamespace(model_name=model_name, forecast_reports=reports)


def _report(qid: int, question, prediction) -> SimpleNamespace:
    question.id_of_question = qid
    return SimpleNamespace(question=question, prediction=prediction)


def test_ensemble_per_type_binary_scores_the_aggregated_probability():
    """Binary aggregation feeds the inline community log score, mean vs median differing."""
    from analyze_correlations import _ensemble_per_type

    question = SimpleNamespace(id_of_question=1, community_prediction_at_access_time=0.6)
    benches = [
        _bench("m1", [_report(1, question, 0.2)]),
        _bench("m2", [_report(1, question, 0.4)]),
        _bench("m3", [_report(1, question, 0.9)]),
    ]
    analyzer = _stub_analyzer("binary")

    mean_stats = _ensemble_per_type(analyzer, benches, ["m1", "m2", "m3"], "mean")
    median_stats = _ensemble_per_type(analyzer, benches, ["m1", "m2", "m3"], "median")

    assert mean_stats["binary"]["n"] == 1
    assert median_stats["binary"]["n"] == 1

    def expected(p: float) -> float:
        import numpy as np

        c = 0.6
        return 100.0 * (c * (np.log2(p) + 1.0) + (1.0 - c) * (np.log2(1.0 - p) + 1.0))

    assert mean_stats["binary"]["mean"] == pytest.approx(expected(0.5), abs=1e-9)
    assert median_stats["binary"]["mean"] == pytest.approx(expected(0.4), abs=1e-9)


def test_ensemble_per_type_binary_skips_question_without_community_prediction():
    """No community prediction means the question can't be scored at all."""
    from analyze_correlations import _ensemble_per_type

    question = SimpleNamespace(id_of_question=1, community_prediction_at_access_time=None)
    benches = [_bench("m1", [_report(1, question, 0.2)]), _bench("m2", [_report(1, question, 0.4)])]

    assert _ensemble_per_type(_stub_analyzer("binary"), benches, ["m1", "m2"], "mean") == {}


def test_ensemble_per_type_skips_questions_missing_a_model():
    """Every model in the list must have answered the question."""
    from analyze_correlations import _ensemble_per_type

    q1 = SimpleNamespace(id_of_question=1, community_prediction_at_access_time=0.5)
    q2 = SimpleNamespace(id_of_question=2, community_prediction_at_access_time=0.5)
    benches = [_bench("m1", [_report(1, q1, 0.3), _report(2, q2, 0.3)]), _bench("m2", [_report(1, q1, 0.7)])]

    stats = _ensemble_per_type(_stub_analyzer("binary"), benches, ["m1", "m2"], "mean")
    assert stats["binary"]["n"] == 1, "only q1 has both models"


def test_ensemble_per_type_mc_renormalizes_aggregated_option_probabilities():
    """MC option probs are matched BY NAME across models, aggregated, then renormalized to 1."""
    import analyze_correlations
    from analyze_correlations import _ensemble_per_type

    def _pred(pairs):
        return SimpleNamespace(
            predicted_options=[SimpleNamespace(option_name=name, probability=prob) for name, prob in pairs]
        )

    question = SimpleNamespace(id_of_question=7)
    benches = [
        # Deliberately different option ORDER on the second model: aggregation must
        # match on the name, not the position.
        _bench("m1", [_report(7, question, _pred([("a", 0.2), ("b", 0.3), ("c", 0.5)]))]),
        _bench("m2", [_report(7, question, _pred([("c", 0.1), ("b", 0.5), ("a", 0.4)]))]),
    ]

    captured = []

    def _fake_score_mc(fake):
        captured.append([(o.option_name, o.probability) for o in fake.prediction.predicted_options])
        return -12.5

    with patch.object(analyze_correlations, "_score_mc", _fake_score_mc):
        stats = _ensemble_per_type(_stub_analyzer("multiple_choice"), benches, ["m1", "m2"], "mean")

    assert stats["multiple_choice"]["n"] == 1
    assert stats["multiple_choice"]["mean"] == pytest.approx(-12.5)
    assert len(captured) == 1
    names = [name for name, _ in captured[0]]
    probs = [prob for _, prob in captured[0]]
    assert names == ["a", "b", "c"], "option order follows the first model's ballot"
    assert probs == pytest.approx([0.3, 0.4, 0.3], abs=1e-9)
    assert sum(probs) == pytest.approx(1.0)


def test_ensemble_per_type_mc_skips_prediction_without_options():
    from analyze_correlations import _ensemble_per_type

    question = SimpleNamespace(id_of_question=7)
    empty = SimpleNamespace(predicted_options=[])
    benches = [_bench("m1", [_report(7, question, empty)]), _bench("m2", [_report(7, question, empty)])]

    assert _ensemble_per_type(_stub_analyzer("multiple_choice"), benches, ["m1", "m2"], "mean") == {}


def test_ensemble_per_type_numeric_aggregates_cdfs_on_the_shortest_grid():
    """Numeric CDFs are truncated to the shortest grid, then averaged percentile-wise."""
    import analyze_correlations
    from analyze_correlations import _ensemble_per_type

    def _cdf(pairs):
        return [SimpleNamespace(value=value, percentile=perc) for value, perc in pairs]

    cdfs = {
        "m1": _cdf([(0.0, 0.1), (1.0, 0.5), (2.0, 0.9)]),
        # Longer grid: the extra point must be dropped, not zero-padded.
        "m2": _cdf([(0.0, 0.3), (1.0, 0.7), (2.0, 0.95), (3.0, 0.99)]),
    }
    question = SimpleNamespace(id_of_question=9)
    benches = [_bench("m1", [_report(9, question, object())]), _bench("m2", [_report(9, question, object())])]

    captured = []

    def _fake_score_num(fake):
        captured.append([(pt.value, pt.percentile) for pt in fake.prediction.cdf])
        return -30.0

    with patch.object(analyze_correlations, "_score_num", _fake_score_num):
        stats = _ensemble_per_type(_stub_analyzer("numeric", cdfs), benches, ["m1", "m2"], "mean")

    assert stats["numeric"]["n"] == 1
    assert stats["numeric"]["mean"] == pytest.approx(-30.0)
    assert captured[0] == pytest.approx([(0.0, 0.2), (1.0, 0.6), (2.0, 0.925)])


def test_ensemble_per_type_numeric_drops_question_when_any_cdf_is_unavailable():
    """One unrecoverable CDF disqualifies the whole question, not just that model."""
    import analyze_correlations
    from analyze_correlations import _ensemble_per_type

    cdfs = {"m1": [SimpleNamespace(value=0.0, percentile=0.5)], "m2": None}
    question = SimpleNamespace(id_of_question=9)
    benches = [_bench("m1", [_report(9, question, object())]), _bench("m2", [_report(9, question, object())])]

    with patch.object(analyze_correlations, "_score_num", lambda _fake: -1.0):
        assert _ensemble_per_type(_stub_analyzer("numeric", cdfs), benches, ["m1", "m2"], "mean") == {}


def test_load_benchmarks_from_path_returns_empty_on_malformed_json():
    """A malformed file logs and yields no benchmarks rather than raising."""
    from analyze_correlations import load_benchmarks_from_path

    with tempfile.TemporaryDirectory() as tmpdir:
        bad = Path(tmpdir) / "benchmarks_bad.json"
        bad.write_text("{not valid json")
        assert load_benchmarks_from_path(str(bad)) == []


def test_load_benchmarks_from_directory_skips_correlation_outputs():
    """Directory loads must not try to parse the report files this script itself writes."""
    from analyze_correlations import load_benchmarks_from_path

    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "correlation_analysis_2026-01-01_00-00-00.json").write_text("{not valid json")
        assert load_benchmarks_from_path(tmpdir) == []
