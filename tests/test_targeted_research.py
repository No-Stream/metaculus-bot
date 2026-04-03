"""Unit tests for targeted_research module and its prompt functions."""

from unittest.mock import AsyncMock, patch

import pytest

from metaculus_bot.prompts import disagreement_crux_prompt, targeted_search_prompt
from metaculus_bot.targeted_research import extract_disagreement_crux, run_targeted_search


class TestDisagreementCruxPrompt:
    """Tests for the disagreement_crux_prompt formatting function."""

    def test_formats_predictions(self):
        result = disagreement_crux_prompt("Will X happen?", ["Analysis 1 text", "Analysis 2 text"])

        assert "Forecaster 1" in result
        assert "Forecaster 2" in result
        assert "Will X happen?" in result
        assert "Analysis 1 text" in result
        assert "Analysis 2 text" in result
        assert "factual question" in result.lower()

    def test_handles_many_predictions(self):
        predictions = [f"Prediction {i}" for i in range(6)]
        result = disagreement_crux_prompt("Question?", predictions)

        for i in range(1, 7):
            assert f"Forecaster {i}" in result


class TestTargetedSearchPrompt:
    """Tests for the targeted_search_prompt formatting function."""

    def test_includes_crux_and_question(self):
        result = targeted_search_prompt("Is the treaty signed?", "Will X happen?")

        assert "Is the treaty signed?" in result
        assert "Will X happen?" in result

    def test_benchmarking_warning_present_when_true(self):
        result = targeted_search_prompt("crux", "question", is_benchmarking=True)

        assert "benchmarking" in result.lower()
        assert "data leakage" in result.lower()

    def test_benchmarking_warning_absent_when_false(self):
        result = targeted_search_prompt("crux", "question", is_benchmarking=False)

        assert "benchmarking" not in result.lower()
        assert "data leakage" not in result.lower()


class TestExtractDisagreementCrux:
    """Tests for the async extract_disagreement_crux function."""

    @pytest.mark.asyncio
    async def test_calls_llm_and_returns_result(self):
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "The crux is X"

        result = await extract_disagreement_crux(mock_llm, "question", ["pred1", "pred2"])

        assert result == "The crux is X"
        mock_llm.invoke.assert_called_once()
        prompt_arg = mock_llm.invoke.call_args[0][0]
        assert "pred1" in prompt_arg
        assert "pred2" in prompt_arg

    @pytest.mark.asyncio
    async def test_propagates_errors(self):
        mock_llm = AsyncMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM timeout")

        with pytest.raises(RuntimeError, match="LLM timeout"):
            await extract_disagreement_crux(mock_llm, "question", ["pred1", "pred2"])


class TestRunTargetedSearch:
    """Tests for the async run_targeted_search function."""

    @pytest.mark.asyncio
    async def test_calls_grok_and_returns_result(self):
        mock_llm_instance = AsyncMock()
        mock_llm_instance.invoke.return_value = "Search results"

        with patch("metaculus_bot.targeted_research.build_native_search_llm", return_value=mock_llm_instance):
            result = await run_targeted_search("crux text", "question text")

        assert result == "Search results"
        mock_llm_instance.invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_benchmarking_flag(self):
        mock_llm_instance = AsyncMock()
        mock_llm_instance.invoke.return_value = "results"

        with (
            patch("metaculus_bot.targeted_research.build_native_search_llm", return_value=mock_llm_instance),
            patch(
                "metaculus_bot.targeted_research.targeted_search_prompt", wraps=targeted_search_prompt
            ) as mock_prompt,
        ):
            await run_targeted_search("crux", "q", is_benchmarking=True)

        mock_prompt.assert_called_once_with("crux", "q", is_benchmarking=True)
