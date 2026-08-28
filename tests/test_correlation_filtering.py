from __future__ import annotations

from typing import Any, cast

from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot

from metaculus_bot.ensemble_analysis.correlation_analysis import CorrelationAnalyzer


class FakeQuestion:
    def __init__(self, qid: int):
        self.id_of_question = qid
        self.page_url = f"https://metaculus.com/questions/{qid}"
        self.community_prediction_at_access_time = None


class FakeReport:
    def __init__(self, qid: int, prediction: float, score: float, cost: float):
        self.question = FakeQuestion(qid)
        self.prediction = prediction  # float for binary
        self.expected_baseline_score = score
        self.price_estimate = cost
        self.explanation = ""


class FakeBenchmark:
    def __init__(self, name: str, model_path: str):
        self.name = name
        self.total_cost = 0.01
        self.forecast_reports = [FakeReport(42, 0.6, 12.3, 0.001)]
        # Emulate the llms structure used for identifier extraction
        self.forecast_bot_config: dict[str, Any] = {
            "aggregation_strategy": "mean",
            "llms": {
                "default": {"model": model_path},
                "forecasters": [
                    {"model": model_path},
                ],
            },
        }


def build_analyzer_with_models(names_and_paths: list[tuple[str, str]]) -> CorrelationAnalyzer:
    benches = [FakeBenchmark(n, p) for n, p in names_and_paths]
    analyzer = CorrelationAnalyzer()
    analyzer.add_benchmark_results(cast("list[BenchmarkForBot]", benches))
    return analyzer


def test_exclude_models_by_substring_simple():
    analyzer = build_analyzer_with_models(
        [
            ("qwen3-235b", "openrouter/qwen/qwen3-235b-a22b-thinking-2507"),
            ("o3", "openrouter/openai/o3"),
            ("grok-4", "openrouter/x-ai/grok-4"),
            ("gemini-2.5-pro", "openrouter/google/gemini-2.5-pro"),
        ]
    )

    before = analyzer.get_model_names()
    assert set(before) >= {"qwen3-235b", "o3", "grok-4", "gemini-2.5-pro"}

    analyzer.filter_models_inplace(exclude=["grok-4", "gemini-2.5-pro"])  # remove two
    after = analyzer.get_model_names()
    assert set(after) == {"qwen3-235b", "o3"}

    # Predictions should also be pruned to the remaining models (one report each)
    preds_models = {p.model_name for p in analyzer.predictions}
    assert preds_models == {"qwen3-235b", "o3"}


def test_exclude_by_model_path_substring():
    analyzer = build_analyzer_with_models(
        [
            ("grok-4", "openrouter/x-ai/grok-4"),
            ("o3", "openrouter/openai/o3"),
        ]
    )

    # Exclude via a substring of the model path, not the clean name
    analyzer.filter_models_inplace(exclude=["x-ai/grok-4"])  # substring match
    remaining = analyzer.get_model_names()
    assert remaining == ["o3"]


def test_include_only_subset():
    analyzer = build_analyzer_with_models(
        [
            ("qwen3-235b", "openrouter/qwen/qwen3-235b-a22b-thinking-2507"),
            ("o3", "openrouter/openai/o3"),
            ("grok-4", "openrouter/x-ai/grok-4"),
        ]
    )

    analyzer.filter_models_inplace(include=["o3", "qwen3-235b"])  # include only two
    names = set(analyzer.get_model_names())
    assert names == {"qwen3-235b", "o3"}


# What the filter REPORTS, not just what it keeps. A token that matched nothing is
# the operator's typo signal, and the summary lines are what the correlation report
# renders under "Filters Applied" — both are part of the contract.


def _three_model_analyzer() -> CorrelationAnalyzer:
    return build_analyzer_with_models(
        [
            ("qwen3-235b", "openrouter/qwen/qwen3-235b-a22b-thinking-2507"),
            ("o3", "openrouter/openai/o3"),
            ("grok-4", "openrouter/x-ai/grok-4"),
        ]
    )


def test_no_tokens_is_a_no_op_with_empty_summary():
    analyzer = _three_model_analyzer()
    result = analyzer.filter_models_inplace()

    assert result == {"included": [], "excluded": [], "unmatched_includes": [], "unmatched_excludes": []}
    assert analyzer._filter_summary_lines == []
    assert len(analyzer.get_model_names()) == 3


def test_unmatched_tokens_are_reported_on_both_sides():
    analyzer = _three_model_analyzer()
    result = analyzer.filter_models_inplace(include=["o3", "nonexistent"], exclude=["also-missing"])

    assert result["included"] == ["o3"]
    assert result["excluded"] == []
    assert result["unmatched_includes"] == ["nonexistent"]
    assert result["unmatched_excludes"] == ["also-missing"]


def test_exclude_wins_over_include_for_the_same_model():
    analyzer = _three_model_analyzer()
    result = analyzer.filter_models_inplace(include=["o3", "grok-4"], exclude=["grok"])

    assert result["included"] == ["o3"]
    assert result["excluded"] == ["grok-4"]
    assert analyzer.get_model_names() == ["o3"]


def test_summary_lines_name_every_token_and_the_remainder():
    analyzer = _three_model_analyzer()
    analyzer.filter_models_inplace(include=["o3", "nonexistent"], exclude=["grok"])

    assert analyzer._filter_summary_lines == [
        "Included by tokens:",
        "- o3: o3",
        "- nonexistent: (no match)",
        "Excluded by tokens:",
        "- grok: grok-4",
        "Remaining models: o3",
    ]


def test_summary_says_none_when_everything_is_filtered_out():
    analyzer = _three_model_analyzer()
    analyzer.filter_models_inplace(exclude=["o3", "grok", "qwen"])

    assert analyzer._filter_summary_lines[-1] == "Remaining models: (none)"
    assert analyzer.get_model_names() == []


def test_blank_and_non_string_tokens_are_ignored():
    analyzer = _three_model_analyzer()
    result = analyzer.filter_models_inplace(exclude=cast("list[str]", ["   ", None]))

    assert result == {"included": [], "excluded": [], "unmatched_includes": [], "unmatched_excludes": []}
    assert len(analyzer.get_model_names()) == 3


def test_matching_is_case_insensitive():
    analyzer = _three_model_analyzer()
    analyzer.filter_models_inplace(exclude=["GROK"])

    assert set(analyzer.get_model_names()) == {"qwen3-235b", "o3"}
