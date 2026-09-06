"""Minimal tests for analyze_correlations.py CLI utility functions."""

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

import analyze_correlations
from analyze_correlations import (
    _ensemble_per_type,
    _parse_args,
    _resolve_output_path,
    extract_timestamp_from_filename,
    load_benchmarks_from_path,
)


def test_import_analyze_correlations() -> None:
    """Test that the CLI script can be imported without errors."""
    assert hasattr(analyze_correlations, "main")
    assert hasattr(analyze_correlations, "extract_timestamp_from_filename")
    assert hasattr(analyze_correlations, "load_benchmarks_from_path")


def test_extract_timestamp_from_filename() -> None:
    """Test timestamp extraction from various filename formats."""
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


def test_load_benchmarks_from_json_file() -> None:
    """Test loading benchmarks from a single JSON file."""
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


def test_load_benchmarks_from_jsonl_file() -> None:
    """Test loading benchmarks from a JSONL file."""
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


def test_load_benchmarks_from_directory() -> None:
    """Test loading benchmarks from a directory with multiple files."""
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


def test_load_nonexistent_path() -> None:
    """Test loading from a path that doesn't exist."""
    benchmarks = load_benchmarks_from_path("/nonexistent/path")
    assert len(benchmarks) == 0


def test_argument_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI parser exposes its current defaults and all supported analysis flags."""
    monkeypatch.setattr("sys.argv", ["analyze_correlations.py", "benchmarks/"])
    defaults = _parse_args()
    assert defaults.benchmark_path == "benchmarks/"
    assert defaults.max_cost == 1.0
    assert defaults.max_size == 7
    assert defaults.verbose is False
    assert defaults.score_stats is True
    assert defaults.question_types is None

    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_correlations.py",
            "test.jsonl",
            "--max-cost",
            "0.5",
            "--max-size",
            "3",
            "--verbose",
            "--question-types",
            "binary",
            "numeric",
            "--no-score-stats",
            "--score-stats-per-question",
            "--include-models",
            "gpt",
            "qwen",
        ],
    )
    custom = _parse_args()
    assert custom.benchmark_path == "test.jsonl"
    assert custom.max_cost == 0.5
    assert custom.max_size == 3
    assert custom.verbose is True
    assert custom.question_types == ["binary", "numeric"]
    assert custom.score_stats is False
    assert custom.score_stats_per_question is True
    assert custom.include_models == ["gpt", "qwen"]


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


def test_main_function_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI loads real benchmark JSON and writes the generated correlation report."""
    first = create_mock_benchmark_data()
    second = create_mock_benchmark_data()
    second["name"] = "Test Bot | Model | second-model | 2025-08-11_12-00-00"
    second["forecast_bot_config"]["llms"]["forecasters"][0]["model"] = "openrouter/openai/gpt-4o-mini"
    second["forecast_bot_config"]["llms"]["default"]["model"] = "openrouter/openai/gpt-4o-mini"
    second["forecast_reports"][0]["prediction"] = 0.4

    benchmark_file = tmp_path / "benchmarks.json"
    benchmark_file.write_text(json.dumps([first, second]))
    output_file = tmp_path / "correlation_analysis.md"

    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_correlations.py",
            str(benchmark_file),
            "--output",
            str(output_file),
            "--no-score-stats",
            "--max-size",
            "2",
        ],
    )

    analyze_correlations.main()

    report = output_file.read_text()
    assert "# Model Correlation Analysis Report" in report
    assert "gpt-4o" in report
    assert "gpt-4o-mini" in report
    assert "CORRELATION ANALYSIS RESULTS" in capsys.readouterr().out


@patch("analyze_correlations.load_benchmarks_from_path")
def test_main_with_insufficient_benchmarks(mock_load):
    """Test main function with too few benchmarks."""
    import analyze_correlations

    # Mock loading only one benchmark (need 2+ for correlation)
    mock_load.return_value = [Mock()]

    with patch("sys.argv", ["analyze_correlations.py", "test.jsonl"]), pytest.raises(SystemExit):
        analyze_correlations.main()


def test_resolve_output_path_uses_timestamp_and_explicit_override(tmp_path: Path) -> None:
    """The production resolver chooses a sibling timestamped file and honors --output."""
    input_file = tmp_path / "benchmarks_2025-08-10_15-04-51.jsonl"
    input_file.write_text("{}\n")
    args = argparse.Namespace(benchmark_path=str(input_file), output=None)

    assert _resolve_output_path(args) == tmp_path / "correlation_analysis_2025-08-10_15-04-51.md"

    explicit = tmp_path / "custom.md"
    assert _resolve_output_path(argparse.Namespace(benchmark_path=str(input_file), output=str(explicit))) == str(
        explicit
    )


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


@pytest.mark.parametrize(
    "invalid_probability",
    [float("nan"), float("inf"), True, -0.1, 1.1],
    ids=["nan", "infinity", "boolean", "below-zero", "above-one"],
)
def test_ensemble_per_type_binary_skips_invalid_forecast_probability(invalid_probability: object) -> None:
    """Invalid binary forecasts are omitted instead of being clamped into a score."""
    question = SimpleNamespace(id_of_question=1, community_prediction_at_access_time=0.6)
    benches = [
        _bench("m1", [_report(1, question, invalid_probability)]),
        _bench("m2", [_report(1, question, 0.4)]),
    ]

    assert _ensemble_per_type(_stub_analyzer("binary"), benches, ["m1", "m2"], "mean") == {}


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


def _mc_prediction(option_probabilities: list[tuple[str, float]]) -> SimpleNamespace:
    return SimpleNamespace(
        predicted_options=[
            SimpleNamespace(option_name=name, probability=probability) for name, probability in option_probabilities
        ]
    )


@pytest.mark.parametrize(
    ("first_model_options", "second_model_options"),
    [
        (
            [
                ("a", 0.0),
                ("b", 0.0),
            ],
            [
                ("a", 0.0),
                ("b", 0.0),
            ],
        ),
        (
            [
                ("a", float("nan")),
                ("b", 1.0),
            ],
            [
                ("a", 0.5),
                ("b", 0.5),
            ],
        ),
        (
            [
                ("a", 0.5),
                ("b", 0.5),
            ],
            [("a", 1.0)],
        ),
    ],
    ids=["zero-total", "nonfinite", "missing-option"],
)
def test_ensemble_per_type_mc_skips_unscoreable_ballots(
    first_model_options: list[tuple[str, float]], second_model_options: list[tuple[str, float]]
) -> None:
    """MC aggregation omits zero, nonfinite, and incomplete ballots rather than imputing values."""
    question = SimpleNamespace(id_of_question=7)
    benches = [
        _bench("m1", [_report(7, question, _mc_prediction(first_model_options))]),
        _bench("m2", [_report(7, question, _mc_prediction(second_model_options))]),
    ]

    with patch.object(analyze_correlations, "_score_mc", return_value=-1.0) as score_mc:
        stats = _ensemble_per_type(_stub_analyzer("multiple_choice"), benches, ["m1", "m2"], "mean")

    assert stats == {}
    score_mc.assert_not_called()


def test_ensemble_per_type_mc_skips_member_with_zero_probability_mass() -> None:
    """A zero-mass member cannot be repaired by averaging it with a valid ballot."""
    question = SimpleNamespace(id_of_question=7)
    benches = [
        _bench(
            "m1",
            [
                _report(
                    7,
                    question,
                    _mc_prediction([("a", 0.0), ("b", 0.0)]),
                )
            ],
        ),
        _bench(
            "m2",
            [
                _report(
                    7,
                    question,
                    _mc_prediction([("a", 0.7), ("b", 0.3)]),
                )
            ],
        ),
    ]

    with patch.object(analyze_correlations, "_score_mc", return_value=-1.0) as score_mc:
        stats = _ensemble_per_type(_stub_analyzer("multiple_choice"), benches, ["m1", "m2"], "mean")

    assert stats == {}
    score_mc.assert_not_called()


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
