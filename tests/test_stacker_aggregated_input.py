"""Stacker-prompt injection tests.

Workstream C edit 3: when ``build_cross_model_aggregation`` returns markdown,
it must land at the TOP of the stacker prompt as a "Cross-model aggregation
(deterministic math)" block. When None or empty, the section must be absent
entirely (not a stub header).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

from metaculus_bot import stacking
from metaculus_bot.prompts import (
    stacking_binary_prompt,
    stacking_multiple_choice_prompt,
    stacking_numeric_prompt,
)
from tests.conftest import make_mock_numeric_question

# _forecasting_window_str asserts on open_time / scheduled_resolution_time;
# populate in every mock question fixture below.
_OPEN = datetime.now() - timedelta(days=30)
_RESOLVE = datetime.now() + timedelta(days=365)

# ---------------------------------------------------------------------------
# Question factories
# ---------------------------------------------------------------------------


def _make_binary_q() -> BinaryQuestion:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = 1
    q.question_text = "Will it rain?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = "https://example.com/q/1"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


def _make_mc_q() -> MultipleChoiceQuestion:
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.id_of_question = 2
    q.question_text = "Which color?"
    q.options = ["Red", "Blue"]
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = "https://example.com/q/2"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


def _make_numeric_q() -> NumericQuestion:
    q = make_mock_numeric_question(id_of_question=3)
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


AGG_BLOCK = (
    "- **Pools over 3 forecasters**: linear 0.400, log 0.370, Satopää α=2.5 0.420\n"
    "- **Blended base rate across 2 forecasters**: 0.300 (range 0.250–0.333)"
)


# ---------------------------------------------------------------------------
# Prompt-builder direct checks
# ---------------------------------------------------------------------------


class TestStackingBinaryPromptInjection:
    def test_aggregated_block_appears_at_top(self):
        prompt = stacking_binary_prompt(
            _make_binary_q(),
            research="R",
            base_predictions=["model 1 analysis", "model 2 analysis"],
            aggregated_tool_output=AGG_BLOCK,
        )
        assert "Cross-model aggregation (deterministic math)" in prompt
        assert "Pools over 3 forecasters" in prompt
        # Must precede the Multiple Expert Analyses section.
        agg_idx = prompt.find("Cross-model aggregation")
        expert_idx = prompt.find("Multiple Expert Analyses")
        assert 0 <= agg_idx < expert_idx, "Cross-model aggregation block must come before 'Multiple Expert Analyses'"

    def test_no_section_when_none(self):
        prompt = stacking_binary_prompt(
            _make_binary_q(),
            research="R",
            base_predictions=["m1"],
            aggregated_tool_output=None,
        )
        assert "Cross-model aggregation" not in prompt

    def test_no_section_when_empty_string(self):
        prompt = stacking_binary_prompt(
            _make_binary_q(),
            research="R",
            base_predictions=["m1"],
            aggregated_tool_output="",
        )
        assert "Cross-model aggregation" not in prompt

    def test_default_omits_section(self):
        # Backwards-compat: old call-sites without the kwarg get no section.
        prompt = stacking_binary_prompt(_make_binary_q(), research="R", base_predictions=["m1"])
        assert "Cross-model aggregation" not in prompt


class TestStackingMcPromptInjection:
    def test_aggregated_block_appears_at_top(self):
        prompt = stacking_multiple_choice_prompt(
            _make_mc_q(),
            research="R",
            base_predictions=["m1"],
            aggregated_tool_output=AGG_BLOCK,
        )
        assert "Cross-model aggregation (deterministic math)" in prompt
        agg_idx = prompt.find("Cross-model aggregation")
        expert_idx = prompt.find("Multiple Expert Analyses")
        assert 0 <= agg_idx < expert_idx

    def test_no_section_when_none(self):
        prompt = stacking_multiple_choice_prompt(_make_mc_q(), research="R", base_predictions=["m1"])
        assert "Cross-model aggregation" not in prompt


class TestStackingNumericPromptInjection:
    def test_aggregated_block_appears_at_top(self):
        prompt = stacking_numeric_prompt(
            _make_numeric_q(),
            research="R",
            base_predictions=["m1"],
            lower_bound_message="",
            upper_bound_message="",
            aggregated_tool_output=AGG_BLOCK,
        )
        assert "Cross-model aggregation (deterministic math)" in prompt
        agg_idx = prompt.find("Cross-model aggregation")
        expert_idx = prompt.find("Multiple Expert Analyses")
        assert 0 <= agg_idx < expert_idx

    def test_no_section_when_none(self):
        prompt = stacking_numeric_prompt(
            _make_numeric_q(),
            research="R",
            base_predictions=["m1"],
            lower_bound_message="",
            upper_bound_message="",
        )
        assert "Cross-model aggregation" not in prompt


# ---------------------------------------------------------------------------
# run_stacking_* threads the new kwarg through
# ---------------------------------------------------------------------------


class TestRunStackingBinaryThreading:
    def test_run_stacking_binary_passes_aggregated_output_into_prompt(self, monkeypatch):
        captured_prompts: list[str] = []

        class FakeStackerLLM:
            model = "stacker"

            async def invoke(self, prompt: str) -> str:
                await asyncio.sleep(0)
                captured_prompts.append(prompt)
                return "reasoning... Probability: 42%"

        class FakeParserLLM:
            model = "parser"

        async def fake_structure_output(*_args, **_kwargs):
            from forecasting_tools import BinaryPrediction

            await asyncio.sleep(0)
            return BinaryPrediction(prediction_in_decimal=0.42)

        monkeypatch.setattr("metaculus_bot.stacking.structure_output", fake_structure_output)

        asyncio.run(
            stacking.run_stacking_binary(
                stacker_llm=FakeStackerLLM(),  # type: ignore[arg-type]
                parser_llm=FakeParserLLM(),  # type: ignore[arg-type]
                question=_make_binary_q(),
                research="R",
                base_texts=["m1"],
                aggregated_tool_output=AGG_BLOCK,
            )
        )
        assert len(captured_prompts) == 1
        assert "Cross-model aggregation (deterministic math)" in captured_prompts[0]
        assert "Pools over 3 forecasters" in captured_prompts[0]

    def test_run_stacking_binary_without_aggregated_output_omits_section(self, monkeypatch):
        captured_prompts: list[str] = []

        class FakeStackerLLM:
            model = "stacker"

            async def invoke(self, prompt: str) -> str:
                await asyncio.sleep(0)
                captured_prompts.append(prompt)
                return "Probability: 42%"

        class FakeParserLLM:
            model = "parser"

        async def fake_structure_output(*_args, **_kwargs):
            from forecasting_tools import BinaryPrediction

            await asyncio.sleep(0)
            return BinaryPrediction(prediction_in_decimal=0.42)

        monkeypatch.setattr("metaculus_bot.stacking.structure_output", fake_structure_output)

        asyncio.run(
            stacking.run_stacking_binary(
                stacker_llm=FakeStackerLLM(),  # type: ignore[arg-type]
                parser_llm=FakeParserLLM(),  # type: ignore[arg-type]
                question=_make_binary_q(),
                research="R",
                base_texts=["m1"],
            )
        )
        assert "Cross-model aggregation" not in captured_prompts[0]


class TestRunStackingMcThreading:
    def test_run_stacking_mc_passes_aggregated_output_into_prompt(self, monkeypatch):
        captured_prompts: list[str] = []

        class FakeStackerLLM:
            model = "stacker"

            async def invoke(self, prompt: str) -> str:
                await asyncio.sleep(0)
                captured_prompts.append(prompt)
                return "Option_A: 50%\nOption_B: 50%"

        class FakeParserLLM:
            model = "parser"

        async def fake_structure_output(*_args, **_kwargs):
            await asyncio.sleep(0)
            return PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="Red", probability=0.5),
                    PredictedOption(option_name="Blue", probability=0.5),
                ]
            )

        monkeypatch.setattr("metaculus_bot.stacking.structure_output", fake_structure_output)

        asyncio.run(
            stacking.run_stacking_mc(
                stacker_llm=FakeStackerLLM(),  # type: ignore[arg-type]
                parser_llm=FakeParserLLM(),  # type: ignore[arg-type]
                question=_make_mc_q(),
                research="R",
                base_texts=["m1"],
                aggregated_tool_output=AGG_BLOCK,
            )
        )
        assert "Cross-model aggregation (deterministic math)" in captured_prompts[0]


class TestRunStackingNumericThreading:
    def test_run_stacking_numeric_passes_aggregated_output_into_prompt(self, monkeypatch):
        from forecasting_tools.data_models.numeric_report import Percentile

        captured_prompts: list[str] = []

        class FakeStackerLLM:
            model = "stacker"

            async def invoke(self, prompt: str) -> str:
                await asyncio.sleep(0)
                captured_prompts.append(prompt)
                return "percentiles..."

        class FakeParserLLM:
            model = "parser"

        async def fake_structure_output(*_args, **_kwargs):
            await asyncio.sleep(0)
            return [
                Percentile(percentile=0.025, value=1.0),
                Percentile(percentile=0.05, value=2.0),
                Percentile(percentile=0.1, value=3.0),
                Percentile(percentile=0.2, value=4.0),
                Percentile(percentile=0.4, value=5.0),
                Percentile(percentile=0.5, value=6.0),
                Percentile(percentile=0.6, value=7.0),
                Percentile(percentile=0.8, value=8.0),
                Percentile(percentile=0.9, value=9.0),
                Percentile(percentile=0.95, value=10.0),
                Percentile(percentile=0.975, value=11.0),
            ]

        monkeypatch.setattr("metaculus_bot.stacking.structure_output", fake_structure_output)

        asyncio.run(
            stacking.run_stacking_numeric(
                stacker_llm=FakeStackerLLM(),  # type: ignore[arg-type]
                parser_llm=FakeParserLLM(),  # type: ignore[arg-type]
                question=_make_numeric_q(),
                research="R",
                base_texts=["m1"],
                lower_bound_message="",
                upper_bound_message="",
                aggregated_tool_output=AGG_BLOCK,
            )
        )
        assert "Cross-model aggregation (deterministic math)" in captured_prompts[0]


# ---------------------------------------------------------------------------
# End-to-end runtime threading: stacker LLM receives the full aggregation block
# verbatim (Gap 5). Earlier tests above check that the kwarg threads through to
# the prompt builder; these assert that *all* aggregation lines actually reach
# the captured prompt, in order, with no rewriting / truncation.
# ---------------------------------------------------------------------------


_RICH_AGG_BLOCK = (
    "- **Pools over 4 forecasters**: linear 0.412, log 0.379, Satopää α=2.5 0.451\n"
    "- **Blended base rate across 3 forecasters**: 0.302 (range 0.250–0.410)\n"
    "- **Prior/posterior snapshot**: 4 forecasters declared priors, "
    "priors range 0.180–0.330, posteriors range 0.250–0.460"
)


class TestRunStackingBinaryFullPromptCapture:
    def test_all_aggregation_lines_present_and_ordered_in_prompt(self, monkeypatch):
        captured_prompts: list[str] = []

        class FakeStackerLLM:
            model = "stacker"

            async def invoke(self, prompt: str) -> str:
                await asyncio.sleep(0)
                captured_prompts.append(prompt)
                return "Probability: 41%"

        class FakeParserLLM:
            model = "parser"

        async def fake_structure_output(*_args, **_kwargs):
            from forecasting_tools import BinaryPrediction

            await asyncio.sleep(0)
            return BinaryPrediction(prediction_in_decimal=0.41)

        monkeypatch.setattr("metaculus_bot.stacking.structure_output", fake_structure_output)

        asyncio.run(
            stacking.run_stacking_binary(
                stacker_llm=FakeStackerLLM(),  # type: ignore[arg-type]
                parser_llm=FakeParserLLM(),  # type: ignore[arg-type]
                question=_make_binary_q(),
                research="news context",
                base_texts=["base reasoning 1", "base reasoning 2", "base reasoning 3"],
                aggregated_tool_output=_RICH_AGG_BLOCK,
            )
        )
        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]

        # The header must appear, and every aggregation bullet must be present
        # verbatim (no truncation, no escaping).
        assert "Cross-model aggregation (deterministic math)" in prompt
        assert "Pools over 4 forecasters" in prompt
        assert "linear 0.412, log 0.379, Satopää α=2.5 0.451" in prompt
        assert "Blended base rate across 3 forecasters" in prompt
        assert "range 0.250–0.410" in prompt
        assert "Prior/posterior snapshot" in prompt

        # Aggregation block precedes both expert analyses AND research context.
        agg_idx = prompt.find("Cross-model aggregation")
        analyses_idx = prompt.find("Multiple Expert Analyses")
        assert 0 <= agg_idx < analyses_idx

        # Lines must appear in the same order as the source block — guards
        # against accidental dict-iteration / set-based reformatting.
        pools_idx = prompt.find("Pools over 4 forecasters")
        blended_idx = prompt.find("Blended base rate across 3 forecasters")
        snapshot_idx = prompt.find("Prior/posterior snapshot")
        assert pools_idx < blended_idx < snapshot_idx


class TestRunStackingMcFullPromptCapture:
    def test_all_aggregation_lines_present_for_mc(self, monkeypatch):
        captured_prompts: list[str] = []
        mc_agg_block = "- **Linear pool across 3 forecasters** (top 3): Red=0.520, Blue=0.310, Green=0.170"

        class FakeStackerLLM:
            model = "stacker"

            async def invoke(self, prompt: str) -> str:
                await asyncio.sleep(0)
                captured_prompts.append(prompt)
                return "Red: 50%\nBlue: 30%\nGreen: 20%"

        class FakeParserLLM:
            model = "parser"

        async def fake_structure_output(*_args, **_kwargs):
            await asyncio.sleep(0)
            return PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="Red", probability=0.5),
                    PredictedOption(option_name="Blue", probability=0.3),
                    PredictedOption(option_name="Green", probability=0.2),
                ]
            )

        mc_q = MagicMock(spec=MultipleChoiceQuestion)
        mc_q.id_of_question = 7
        mc_q.question_text = "Color?"
        mc_q.options = ["Red", "Blue", "Green"]
        mc_q.background_info = "bg"
        mc_q.resolution_criteria = "rc"
        mc_q.fine_print = ""
        mc_q.page_url = "https://example.com/q/7"
        mc_q.open_time = _OPEN
        mc_q.scheduled_resolution_time = _RESOLVE

        monkeypatch.setattr("metaculus_bot.stacking.structure_output", fake_structure_output)

        asyncio.run(
            stacking.run_stacking_mc(
                stacker_llm=FakeStackerLLM(),  # type: ignore[arg-type]
                parser_llm=FakeParserLLM(),  # type: ignore[arg-type]
                question=mc_q,
                research="research",
                base_texts=["m1", "m2", "m3"],
                aggregated_tool_output=mc_agg_block,
            )
        )
        prompt = captured_prompts[0]
        assert "Cross-model aggregation (deterministic math)" in prompt
        assert "Linear pool across 3 forecasters" in prompt
        assert "Red=0.520, Blue=0.310, Green=0.170" in prompt
        agg_idx = prompt.find("Cross-model aggregation")
        analyses_idx = prompt.find("Multiple Expert Analyses")
        assert 0 <= agg_idx < analyses_idx


class TestRunStackingNumericFullPromptCapture:
    def test_all_aggregation_lines_present_for_numeric(self, monkeypatch):
        from forecasting_tools.data_models.numeric_report import Percentile

        captured_prompts: list[str] = []
        numeric_agg_block = (
            "- **Forecaster medians**: min 25.5, max 47.2, n=3\n"
            "- **Declared distribution families**: lognormal, normal (3 forecasters)"
        )

        class FakeStackerLLM:
            model = "stacker"

            async def invoke(self, prompt: str) -> str:
                await asyncio.sleep(0)
                captured_prompts.append(prompt)
                return "percentiles..."

        class FakeParserLLM:
            model = "parser"

        async def fake_structure_output(*_args, **_kwargs):
            await asyncio.sleep(0)
            return [
                Percentile(percentile=0.025, value=10.0),
                Percentile(percentile=0.05, value=15.0),
                Percentile(percentile=0.1, value=20.0),
                Percentile(percentile=0.2, value=25.0),
                Percentile(percentile=0.4, value=30.0),
                Percentile(percentile=0.5, value=35.0),
                Percentile(percentile=0.6, value=40.0),
                Percentile(percentile=0.8, value=50.0),
                Percentile(percentile=0.9, value=60.0),
                Percentile(percentile=0.95, value=70.0),
                Percentile(percentile=0.975, value=80.0),
            ]

        monkeypatch.setattr("metaculus_bot.stacking.structure_output", fake_structure_output)

        asyncio.run(
            stacking.run_stacking_numeric(
                stacker_llm=FakeStackerLLM(),  # type: ignore[arg-type]
                parser_llm=FakeParserLLM(),  # type: ignore[arg-type]
                question=_make_numeric_q(),
                research="research",
                base_texts=["m1", "m2", "m3"],
                lower_bound_message="",
                upper_bound_message="",
                aggregated_tool_output=numeric_agg_block,
            )
        )
        prompt = captured_prompts[0]
        assert "Cross-model aggregation (deterministic math)" in prompt
        assert "Forecaster medians" in prompt
        assert "min 25.5, max 47.2, n=3" in prompt
        assert "Declared distribution families" in prompt
        assert "lognormal, normal" in prompt
        agg_idx = prompt.find("Cross-model aggregation")
        analyses_idx = prompt.find("Multiple Expert Analyses")
        assert 0 <= agg_idx < analyses_idx


# Silence unused-import complaints: Sequence/patch kept for future expansion.
_ = Sequence
_ = patch
_ = pytest
