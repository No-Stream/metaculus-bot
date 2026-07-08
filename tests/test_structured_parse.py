"""Direct tests for metaculus_bot.structured_parse.

Exercises the real parse_structured (no mocking of parse_structured itself).
Covers:
  - constrained happy path (non-list BinaryPrediction)
  - list wrappers unwrap (list[Percentile], list[OptionProbability])
  - fallback path to forecasting_tools.structure_output on runtime error
  - fallback path on malformed JSON (model_validate_json raises)
  - F2 regression: constrained GeneralLlm is constructed with parser_llm.model
  - open-bound preservation: negative / huge values pass through the wrapper
    round-trip verbatim (no clamping).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import BinaryPrediction, GeneralLlm
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot import structured_parse as sp
from metaculus_bot.simple_types import OptionProbability
from metaculus_bot.structured_parse import (
    OptionProbabilityListWrapper,
    PercentileListWrapper,
    parse_structured,
)


@pytest.fixture
def parser_llm():
    """A real GeneralLlm instance used ONLY as the fallback-path model.

    Its .model attribute (`test-parser-model/slug`) is what parse_structured
    threads into _build_constrained_llm — we assert that in
    test_constrained_llm_receives_parser_model_slug.
    """
    llm = MagicMock(spec=GeneralLlm)
    llm.model = "openrouter/some/other-model"
    return llm


def _patch_build_constrained_llm(canned_response: str | Exception) -> MagicMock:
    """Return a MagicMock stand-in for the constrained GeneralLlm.

    Its .invoke() returns/raises canned_response. The caller is responsible for
    the patch context.
    """
    constrained = MagicMock(spec=GeneralLlm)
    if isinstance(canned_response, Exception):
        constrained.invoke = AsyncMock(side_effect=canned_response)
    else:
        constrained.invoke = AsyncMock(return_value=canned_response)
    return constrained


class TestConstrainedHappyPath:
    """Constrained primary path succeeds and returns the parsed model."""

    @pytest.mark.asyncio
    async def test_non_list_binary_prediction(self, parser_llm):
        """BinaryPrediction (non-list output_type) round-trips through model_validate_json."""
        constrained = _patch_build_constrained_llm('{"prediction_in_decimal": 0.42}')
        with patch.object(sp, "_build_constrained_llm", return_value=constrained) as build_mock:
            result = await parse_structured("some reasoning text", BinaryPrediction, parser_llm)

        assert isinstance(result, BinaryPrediction)
        assert result.prediction_in_decimal == 0.42
        # F2: constrained LLM was built with the parser_llm's own model slug
        build_mock.assert_called_once()
        _, model_arg = build_mock.call_args[0]
        assert model_arg == parser_llm.model


class TestListWrapperUnwrap:
    """Both list wrappers unwrap to raw list[T] on the constrained path."""

    @pytest.mark.asyncio
    async def test_percentile_list_wrapper_unwrap(self, parser_llm):
        canned = (
            '{"percentiles": ['
            '{"percentile": 0.1, "value": 10.0},'
            '{"percentile": 0.5, "value": 50.0},'
            '{"percentile": 0.9, "value": 90.0}'
            "]}"
        )
        constrained = _patch_build_constrained_llm(canned)
        with patch.object(sp, "_build_constrained_llm", return_value=constrained):
            result = await parse_structured("txt", list[Percentile], parser_llm)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(p, Percentile) for p in result)
        assert result[0].percentile == 0.1 and result[0].value == 10.0
        assert result[2].percentile == 0.9 and result[2].value == 90.0

    @pytest.mark.asyncio
    async def test_option_probability_list_wrapper_unwrap(self, parser_llm):
        canned = '{"options": [{"option_name": "A", "probability": 0.6},{"option_name": "B", "probability": 0.4}]}'
        constrained = _patch_build_constrained_llm(canned)
        with patch.object(sp, "_build_constrained_llm", return_value=constrained):
            result = await parse_structured("txt", list[OptionProbability], parser_llm)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(o, OptionProbability) for o in result)
        assert result[0].option_name == "A" and result[0].probability == 0.6
        assert result[1].option_name == "B" and result[1].probability == 0.4


class TestConstrainedLlmReceivesParserModel:
    """F2 regression: _build_constrained_llm is threaded parser_llm.model, not a hardcoded slug."""

    @pytest.mark.asyncio
    async def test_constrained_llm_receives_parser_model_slug(self, parser_llm):
        """The parser_llm's rotated model slug flows through to the constrained LLM builder."""
        parser_llm.model = "openrouter/some/other-model"
        constrained = _patch_build_constrained_llm('{"prediction_in_decimal": 0.5}')
        with patch.object(sp, "_build_constrained_llm", return_value=constrained) as build_mock:
            await parse_structured("txt", BinaryPrediction, parser_llm)

        build_mock.assert_called_once()
        # (response_format_model, parser_model) is the positional signature.
        _, model_arg = build_mock.call_args[0]
        assert model_arg == "openrouter/some/other-model"


class TestFallbackGuarantee:
    """The structure_output fallback fires on runtime error OR malformed JSON."""

    @pytest.mark.asyncio
    async def test_runtime_error_falls_back_to_structure_output(self, parser_llm):
        """Constrained invoke raises RuntimeError → fallback invoked with original output_type + text."""
        constrained = _patch_build_constrained_llm(RuntimeError("provider blew up"))
        fallback = AsyncMock(return_value=BinaryPrediction(prediction_in_decimal=0.3))
        with (
            patch.object(sp, "_build_constrained_llm", return_value=constrained),
            patch.object(sp, "structure_output", new=fallback),
        ):
            result = await parse_structured("original text", BinaryPrediction, parser_llm, prompt_notes="notes")

        assert isinstance(result, BinaryPrediction)
        assert result.prediction_in_decimal == 0.3
        fallback.assert_awaited_once()
        # Verify the ORIGINAL output_type and text propagate to the fallback.
        await_args = fallback.await_args
        assert await_args is not None
        call_kwargs = await_args.kwargs
        assert call_kwargs["text_to_structure"] == "original text"
        assert call_kwargs["output_type"] is BinaryPrediction
        assert call_kwargs["model"] is parser_llm
        assert call_kwargs["additional_instructions"] == "notes"

    @pytest.mark.asyncio
    async def test_malformed_json_falls_back_to_structure_output(self, parser_llm):
        """Constrained invoke returns bad JSON → model_validate_json raises → same fallback."""
        constrained = _patch_build_constrained_llm("this is not JSON at all {")
        fallback_result = [Percentile(percentile=0.5, value=42.0)]
        fallback = AsyncMock(return_value=fallback_result)
        with (
            patch.object(sp, "_build_constrained_llm", return_value=constrained),
            patch.object(sp, "structure_output", new=fallback),
        ):
            result = await parse_structured("t", list[Percentile], parser_llm)

        assert result is fallback_result
        fallback.assert_awaited_once()
        await_args = fallback.await_args
        assert await_args is not None
        assert await_args.kwargs["output_type"] == list[Percentile]

    @pytest.mark.asyncio
    async def test_wrapper_json_validation_error_falls_back(self, parser_llm):
        """Constrained returns JSON that parses but fails Percentile validation → fallback."""
        # `percentile` outside [0,1] fails Percentile's pydantic validators, so
        # PercentileListWrapper.model_validate_json raises.
        canned = '{"percentiles": [{"percentile": 1.5, "value": 10.0}]}'
        constrained = _patch_build_constrained_llm(canned)
        fallback_result = [Percentile(percentile=0.5, value=99.0)]
        fallback = AsyncMock(return_value=fallback_result)
        with (
            patch.object(sp, "_build_constrained_llm", return_value=constrained),
            patch.object(sp, "structure_output", new=fallback),
        ):
            result = await parse_structured("t", list[Percentile], parser_llm)

        assert result is fallback_result


class TestOpenBoundPreservation:
    """Percentile values outside a nominal range survive the wrapper round-trip verbatim."""

    @pytest.mark.asyncio
    async def test_negative_and_huge_values_pass_through_verbatim(self, parser_llm):
        """Extreme values (negative, 1e18) survive PercentileListWrapper unwrap — no clamping."""
        canned = (
            '{"percentiles": ['
            '{"percentile": 0.05, "value": -1000000.0},'
            '{"percentile": 0.5, "value": 42.0},'
            '{"percentile": 0.95, "value": 1e18}'
            "]}"
        )
        constrained = _patch_build_constrained_llm(canned)
        with patch.object(sp, "_build_constrained_llm", return_value=constrained):
            result = await parse_structured("txt", list[Percentile], parser_llm)

        assert len(result) == 3
        assert result[0].value == -1000000.0
        assert result[1].value == 42.0
        assert result[2].value == 1e18


class TestWrapperHelpers:
    """Direct wrapper-model coverage (unwrap paths are exercised via parse_structured above)."""

    def test_percentile_list_wrapper_shape(self):
        w = PercentileListWrapper(percentiles=[Percentile(percentile=0.5, value=1.0)])
        assert len(w.percentiles) == 1
        assert w.percentiles[0].value == 1.0

    def test_option_probability_list_wrapper_shape(self):
        w = OptionProbabilityListWrapper(
            options=[OptionProbability(option_name="A", probability=0.7)],
        )
        assert len(w.options) == 1
        assert w.options[0].option_name == "A"
