"""Behavior pins for ``ensemble_analysis.benchmark_identity``.

These helpers decide what a benchmark is CALLED and what substrings identify it,
which is what every include/exclude filter, per-model stat, and correlation row is
keyed on. The precedence between the four naming sources (bot name shortcut, the
``default`` LLM config, the legacy ``forecasters`` array, pipe-delimited name
parsing) is load-bearing and easy to reorder by accident, so it is pinned here
rather than only through the analyzer's delegating wrappers.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.multiple_choice_report import PredictedOption, PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.ensemble_analysis.benchmark_identity import (
    extract_clean_model_name,
    extract_model_name,
    get_question_type,
    identifiers_for_benchmark,
    is_stacking_benchmark,
)


def _benchmark(name: str, config: dict[str, Any] | None = None) -> BenchmarkForBot:
    return cast("BenchmarkForBot", SimpleNamespace(name=name, forecast_bot_config=config if config is not None else {}))


class TestCleanModelName:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("openrouter/deepseek/deepseek-r1-0528:free", "deepseek-r1-0528"),
            ("openrouter/openai/o3", "o3"),
            ("o3", "o3"),
            ("openrouter/qwen/qwen3-235b:nitro", "qwen3-235b"),
        ],
    )
    def test_last_path_segment_without_variant_suffix(self, path: str, expected: str):
        assert extract_clean_model_name(path) == expected


class TestModelNamePrecedence:
    def test_a_simple_bot_name_wins_outright(self):
        # Fewer than four dash-segments, no pipe, no space: taken verbatim, even
        # though a default LLM config is present and says something else.
        benchmark = _benchmark("qwen3-235b", {"llms": {"default": {"model": "openrouter/openai/o3"}}})
        assert extract_model_name(benchmark) == "qwen3-235b"

    def test_a_dashy_name_falls_through_to_the_default_llm(self):
        benchmark = _benchmark("a-b-c-d", {"llms": {"default": {"model": "openrouter/openai/o3"}}})
        assert extract_model_name(benchmark) == "o3"

    def test_single_forecaster_prefers_original_model_over_model(self):
        benchmark = _benchmark(
            "Bot | Config | fallback",
            {
                "llms": {
                    "forecasters": [
                        {"original_model": "openrouter/openai/o3", "model": "openrouter/openai/gpt-5"},
                    ]
                }
            },
        )
        assert extract_model_name(benchmark) == "o3"

    def test_multi_forecaster_names_are_family_tokens_sorted_and_joined(self):
        benchmark = _benchmark(
            "Unknown Ensemble",
            {
                "llms": {
                    "forecasters": [
                        {"model": "openrouter/openai/gpt-5.6"},
                        {"model": "openrouter/anthropic/claude-opus-4.8"},
                        {"model": "openrouter/deepseek/deepseek-r1"},
                    ]
                }
            },
        )
        assert extract_model_name(benchmark) == "claude_deepseek_gpt5"

    def test_unrecognized_family_falls_back_to_the_leading_name_segment(self):
        benchmark = _benchmark(
            "Unknown Ensemble",
            {
                "llms": {
                    "forecasters": [
                        {"model": "openrouter/mistralai/mistral-large"},
                        {"model": "openrouter/z-ai/glm-4.5"},
                    ]
                }
            },
        )
        assert extract_model_name(benchmark) == "glm_mistral"

    def test_string_aggregation_strategy_is_appended(self):
        benchmark = _benchmark(
            "Unknown Ensemble",
            {
                "llms": {"forecasters": [{"model": "a/qwen3-x"}, {"model": "b/glm-y"}]},
                "aggregation_strategy": "median",
            },
        )
        assert extract_model_name(benchmark) == "glm_qwen3_median"

    def test_enum_aggregation_strategy_is_appended_by_value(self):
        benchmark = _benchmark(
            "Unknown Ensemble",
            {
                "llms": {"forecasters": [{"model": "a/qwen3-x"}, {"model": "b/glm-y"}]},
                "aggregation_strategy": AggregationStrategy.MEAN,
            },
        )
        assert extract_model_name(benchmark) == f"glm_qwen3_{AggregationStrategy.MEAN.value}"

    def test_pipe_delimited_name_yields_its_third_field(self):
        benchmark = _benchmark("Bot | Config | some-model | extra", {"llms": {}})
        assert extract_model_name(benchmark) == "some-model"

    def test_nothing_identifiable_falls_back_to_a_hashed_stub(self):
        benchmark = _benchmark("Two Fields Only", {"llms": {}})
        assert extract_model_name(benchmark).startswith("model_")

    def test_a_broken_config_is_logged_and_falls_back(self, caplog: pytest.LogCaptureFixture):
        benchmark = cast("BenchmarkForBot", SimpleNamespace(name="Two Fields Only", forecast_bot_config="not a dict"))
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.ensemble_analysis.benchmark_identity"):
            assert extract_model_name(benchmark).startswith("model_")

        assert "Could not extract model name" in caplog.text


class TestIdentifiers:
    def test_every_configured_model_path_becomes_an_identifier_in_order(self):
        benchmark = _benchmark(
            "ensemble-bot",
            {
                "llms": {
                    "default": {"model": "openrouter/openai/o3"},
                    "forecasters": [
                        {"original_model": "openrouter/qwen/qwen3-235b", "model": "openrouter/qwen/aliased"},
                        {"model": "openrouter/z-ai/glm-4.5"},
                    ],
                    "stacker": {"model": "openrouter/anthropic/claude-opus-4.8"},
                }
            },
        )
        assert identifiers_for_benchmark(benchmark, "ensemble-bot") == [
            "ensemble-bot",
            "openrouter/openai/o3",
            "openrouter/qwen/qwen3-235b",
            "openrouter/qwen/aliased",
            "openrouter/z-ai/glm-4.5",
            "openrouter/anthropic/claude-opus-4.8",
        ]

    def test_duplicates_and_blanks_are_dropped(self):
        benchmark = _benchmark("o3", {"llms": {"default": {"model": "o3"}, "forecasters": [{"model": ""}]}})
        assert identifiers_for_benchmark(benchmark, "o3") == ["o3"]

    def test_a_broken_config_still_returns_what_it_read(self, caplog: pytest.LogCaptureFixture):
        benchmark = cast("BenchmarkForBot", SimpleNamespace(name="o3", forecast_bot_config={"llms": {"default": 7}}))
        with caplog.at_level(logging.DEBUG, logger="metaculus_bot.ensemble_analysis.benchmark_identity"):
            assert identifiers_for_benchmark(benchmark, "o3") == ["o3"]


class TestStackingDetection:
    @pytest.mark.parametrize("strategy", ["stacking", "STACKING", "Stacking"])
    def test_a_stacking_string_is_detected_case_insensitively(self, strategy: str):
        assert is_stacking_benchmark(_benchmark("bot", {"aggregation_strategy": strategy})) is True

    def test_a_stacking_enum_is_detected_by_value(self):
        assert is_stacking_benchmark(_benchmark("bot", {"aggregation_strategy": AggregationStrategy.STACKING})) is True

    @pytest.mark.parametrize("strategy", ["median", "mean", None])
    def test_other_strategies_are_not_stacking(self, strategy: str | None):
        assert is_stacking_benchmark(_benchmark("bot", {"aggregation_strategy": strategy})) is False

    def test_missing_benchmark_is_not_stacking(self):
        assert is_stacking_benchmark(None) is False

    def test_a_broken_config_is_not_stacking(self):
        benchmark = cast("BenchmarkForBot", SimpleNamespace(name="bot", forecast_bot_config=7))
        assert is_stacking_benchmark(benchmark) is False


class TestQuestionType:
    def test_scalar_prediction_is_binary(self):
        assert get_question_type(SimpleNamespace(prediction=0.42)) == "binary"

    def test_option_list_is_multiple_choice(self):
        options = PredictedOptionList(predicted_options=[PredictedOption(option_name="yes", probability=1.0)])
        assert get_question_type(SimpleNamespace(prediction=options)) == "multiple_choice"

    def test_numeric_distribution_is_numeric(self):
        distribution = NumericDistribution(
            declared_percentiles=[Percentile(value=1.0, percentile=0.1), Percentile(value=9.0, percentile=0.9)],
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=10.0,
            lower_bound=0.0,
            zero_point=None,
            cdf_size=201,
        )
        assert get_question_type(SimpleNamespace(prediction=distribution)) == "numeric"

    def test_anything_else_defaults_to_binary(self):
        assert get_question_type(SimpleNamespace(prediction="unrecognized")) == "binary"
