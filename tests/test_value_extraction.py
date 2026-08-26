"""Tests for metaculus_bot.value_extraction — the deterministic-first extraction ladder.

Rung contract: block (deterministic parse) → repair (json_repair) → llm
(parse_structured salvage) → ValueExtractionError. Every successful extraction
emits one EXTRACTION_RUNG INFO line.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import BinaryPrediction, PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption
from forecasting_tools.data_models.numeric_report import Percentile
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from metaculus_bot.exceptions import ValueExtractionError
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.simple_types import OptionProbability
from metaculus_bot.value_extraction import (
    ExtractionOutcome,
    extract_binary,
    extract_mc,
    extract_numeric,
)

PARSER_LLM = MagicMock()
OPTIONS = ["Option A", "Option B", "Option C"]

VALID_BINARY_BLOCK = '{"question_type": "binary", "posterior_prob": 0.28}'
VALID_MC_BLOCK = (
    '{"question_type": "multiple_choice", "option_probs": {"Option A": 0.5, "Option B": 0.3, "Option C": 0.2}}'
)
_PCTS = ", ".join(f'"{p}": {i + 1}.0' for i, p in enumerate(STANDARD_PERCENTILES))
VALID_NUMERIC_BLOCK = (
    f'{{"question_type": "numeric", "declared_percentiles": {{{_PCTS}}}, "outcome_type": "continuous"}}'
)


def rationale_with(block_json: str) -> str:
    return f"## Analysis\n\nSome careful reasoning here.\n\n```json\n{block_json}\n```\n"


def numeric_block(scale: float, *, trailing_comma: bool = False) -> str:
    """A full-13-percentile numeric block whose values are ``scale * (i + 1)``."""
    pcts = ", ".join(f'"{p}": {scale * (i + 1)}' for i, p in enumerate(STANDARD_PERCENTILES))
    tail = "," if trailing_comma else ""
    return f'{{"question_type": "numeric", "declared_percentiles": {{{pcts}}}, "outcome_type": "continuous"{tail}}}'


def mc_block(probs: list[float], *, trailing_comma: bool = False) -> str:
    body = ", ".join(f'"{name}": {prob}' for name, prob in zip(OPTIONS, probs))
    tail = "," if trailing_comma else ""
    return f'{{"question_type": "multiple_choice", "option_probs": {{{body}}}{tail}}}'


def full_percentile_list() -> list[Percentile]:
    return [Percentile(percentile=p, value=float(i + 1)) for i, p in enumerate(STANDARD_PERCENTILES)]


def make_pol(probs: list[float]) -> PredictedOptionList:
    return PredictedOptionList(
        predicted_options=[PredictedOption(option_name=n, probability=p) for n, p in zip(OPTIONS, probs)]
    )


class TestRungBlock:
    """Rung 1: schema-valid fenced block wins deterministically, no LLM touched."""

    @pytest.mark.asyncio
    async def test_binary_happy_path(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot.value_extraction")
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_binary(
                rationale_with(VALID_BINARY_BLOCK), PARSER_LLM, question_id=42, model_name="test-model"
            )
        assert outcome == ExtractionOutcome(value=0.28, rung="block", block_present=True)
        llm.assert_not_awaited()
        telemetry = [r.getMessage() for r in caplog.records if "EXTRACTION_RUNG:" in r.getMessage()]
        assert len(telemetry) == 1
        assert "question=42" in telemetry[0]
        assert "model=test-model" in telemetry[0]
        assert "qtype=binary" in telemetry[0]
        assert "rung=block" in telemetry[0]
        assert "block_present=True" in telemetry[0]

    @pytest.mark.asyncio
    async def test_mc_happy_path(self) -> None:
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_mc(rationale_with(VALID_MC_BLOCK), OPTIONS, PARSER_LLM)
        assert outcome.rung == "block"
        assert outcome.block_present is True
        probs = {o.option_name: o.probability for o in outcome.value.predicted_options}
        assert set(probs) == set(OPTIONS)
        assert probs["Option A"] == pytest.approx(0.5, abs=0.02)
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_numeric_happy_path(self) -> None:
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_numeric(rationale_with(VALID_NUMERIC_BLOCK), PARSER_LLM)
        assert outcome.rung == "block"
        assert [float(p.percentile) for p in outcome.value] == STANDARD_PERCENTILES
        values = [float(p.value) for p in outcome.value]
        assert values == sorted(values)
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_numeric_returns_canonical_13(self) -> None:
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_numeric(rationale_with(VALID_NUMERIC_BLOCK), PARSER_LLM)
        assert len(outcome.value) == 13
        assert [float(p.percentile) for p in outcome.value] == STANDARD_PERCENTILES
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_last_block_wins_when_duplicated(self) -> None:
        text = rationale_with('{"question_type": "binary", "posterior_prob": 0.1}') + rationale_with(VALID_BINARY_BLOCK)
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()):
            outcome = await extract_binary(text, PARSER_LLM)
        assert outcome.value == 0.28

    @pytest.mark.asyncio
    async def test_binary_raw_value_not_clamped(self) -> None:
        """The ladder returns the RAW decimal; callers clamp."""
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()):
            outcome = await extract_binary(
                rationale_with('{"question_type": "binary", "posterior_prob": 0.999}'), PARSER_LLM
            )
        assert outcome.value == 0.999


class TestRungRepair:
    """Rung 2: deterministic repair of malformed JSON; no LLM touched."""

    @pytest.mark.asyncio
    async def test_trailing_comma(self) -> None:
        broken = '{"question_type": "binary", "posterior_prob": 0.28,}'
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_binary(rationale_with(broken), PARSER_LLM)
        assert outcome == ExtractionOutcome(value=0.28, rung="repair", block_present=True)
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_quotes(self) -> None:
        broken = "{'question_type': 'binary', 'posterior_prob': 0.28}"
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_binary(rationale_with(broken), PARSER_LLM)
        assert outcome.value == 0.28
        assert outcome.rung == "repair"
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unclosed_brace_mc(self) -> None:
        broken = (
            '{"question_type": "multiple_choice", "option_probs": {"Option A": 0.5, "Option B": 0.3, "Option C": 0.2}'
        )
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_mc(rationale_with(broken), OPTIONS, PARSER_LLM)
        assert outcome.rung == "repair"
        assert {o.option_name for o in outcome.value.predicted_options} == set(OPTIONS)
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unfenced_json_at_tail(self) -> None:
        """No fence at all — balanced-braces scan of the tail rescues the payload."""
        text = f"## Analysis\n\nreasoning...\n\n{VALID_BINARY_BLOCK}\n"
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_binary(text, PARSER_LLM)
        assert outcome == ExtractionOutcome(value=0.28, rung="repair", block_present=False)
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_numeric_truncated_mid_object_repaired_or_llm(self) -> None:
        """Truncation may be repairable; if the repaired set is partial, it must NOT pass — llm rung."""
        truncated = VALID_NUMERIC_BLOCK[: len(VALID_NUMERIC_BLOCK) // 2]
        llm_mock = AsyncMock(return_value=full_percentile_list())
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            outcome = await extract_numeric(rationale_with(truncated), PARSER_LLM)
        # Either deterministic repair produced a full 13-set (unlikely for a
        # half-truncated block) or the llm rung salvaged. Never a partial set.
        assert len(outcome.value) == 13
        assert outcome.rung in ("repair", "llm")

    @pytest.mark.asyncio
    async def test_repair_iterates_past_unrepairable_trailing_block(self) -> None:
        """Rung 2 repairs each fenced candidate best-first, not just the last by
        position: a malformed-but-repairable real forecast (trailing comma) sitting
        behind an unrepairable trailing schema-recap (``<your value>``) is still
        recovered by repair, without an LLM call."""
        real = '{"question_type": "binary", "posterior_prob": 0.28,}'  # trailing comma — repairable
        recap = '{"question_type": "binary", "posterior_prob": <your value>}'  # json_repair → string, invalid
        text = rationale_with(real) + rationale_with(recap)
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_binary(text, PARSER_LLM)
        assert outcome.value == pytest.approx(0.28)
        assert outcome.rung == "repair"
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_repair_iterates_tail_balanced_blobs(self) -> None:
        """No fence at all: the tail-scan tries each balanced-brace blob, so a junk
        leading blob doesn't block a valid trailing payload from being repaired."""
        text = f'## Analysis\n\nreasoning...\n\n{{"note": "not a forecast"}}\n{VALID_BINARY_BLOCK}\n'
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_binary(text, PARSER_LLM)
        assert outcome.value == pytest.approx(0.28)
        assert outcome.rung == "repair"
        assert outcome.block_present is False
        llm.assert_not_awaited()


class TestFinalBlockPrecedence:
    """The model's INTENDED FINAL block outranks an earlier valid one.

    Regression (2026-07-25): validity-aware selection returned the first
    candidate that STRICTLY validated, walking backward from the end of the
    rationale. A valid DRAFT block earlier in the text therefore beat a
    malformed final answer — and because rung 2 only ran when rung 1 returned
    None, that malformed final block never reached ``json_repair``, whose whole
    job is fixing defects like a trailing comma. The bot published the
    superseded draft. The deterministic rungs now walk CANDIDATE-major: strict
    parse and repair are both applied to a candidate before a lower-ranked one
    is considered.
    """

    @pytest.mark.asyncio
    async def test_repairable_final_binary_block_beats_earlier_valid_draft(self) -> None:
        draft = '{"question_type": "binary", "posterior_prob": 0.40}'
        final = '{"question_type": "binary", "posterior_prob": 0.72,}'  # trailing comma — repairable
        text = (
            "Draft thinking below.\n"
            + rationale_with(draft)
            + "On reflection my final answer is higher.\n"
            + rationale_with(final)
        )
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_binary(text, PARSER_LLM)
        assert outcome.value == pytest.approx(0.72)
        assert outcome.rung == "repair"
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_repairable_final_numeric_block_beats_earlier_valid(self) -> None:
        text = rationale_with(numeric_block(1.0)) + rationale_with(numeric_block(10.0, trailing_comma=True))
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_numeric(text, PARSER_LLM)
        assert outcome.rung == "repair"
        assert [float(p.percentile) for p in outcome.value] == STANDARD_PERCENTILES
        assert float(outcome.value[0].value) == pytest.approx(10.0)  # the final block's scale, not the draft's
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_repairable_final_mc_block_beats_earlier_valid(self) -> None:
        text = rationale_with(mc_block([0.5, 0.3, 0.2])) + rationale_with(
            mc_block([0.1, 0.2, 0.7], trailing_comma=True)
        )
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_mc(text, OPTIONS, PARSER_LLM)
        assert outcome.rung == "repair"
        probs = {o.option_name: o.probability for o in outcome.value.predicted_options}
        assert probs["Option C"] == pytest.approx(0.7, abs=0.02)  # the final block's answer
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unrepairable_final_block_yields_to_earlier_valid_block(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A trailing schema-example block is the observed prod failure mode, and no
        deterministic route recovers a value from it — so the real forecast earlier in
        the rationale wins, with the fallback logged for drift-watching."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.value_extraction")
        forecast = '{"question_type": "binary", "posterior_prob": 0.42}'
        recap = '{"question_type": "binary", "posterior_prob": <your value>}'
        text = rationale_with(forecast) + "For reference the schema is:\n" + rationale_with(recap)
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_binary(text, PARSER_LLM, question_id=11, model_name="m")
        assert outcome.value == pytest.approx(0.42)
        assert outcome.rung == "block"
        llm.assert_not_awaited()
        fallback = [r for r in caplog.records if "BLOCK_FALLBACK:" in r.getMessage()]
        assert len(fallback) == 1
        assert fallback[0].levelno == logging.INFO
        assert "skipped=1" in fallback[0].getMessage()
        # A recovered forecast is never announced as a failure, at either layer.
        # Scoped to our own loggers: caplog.records spans every logger that propagates
        # to root, so an unrelated third-party WARNING would otherwise fail this.
        our_warnings = [
            r for r in caplog.records if r.levelno >= logging.WARNING and r.name.startswith("metaculus_bot")
        ]
        assert not our_warnings, [r.getMessage() for r in our_warnings]

    @pytest.mark.asyncio
    async def test_schema_valid_but_unusable_final_block_yields_to_earlier_valid(self) -> None:
        """A trailing block can pass the schema yet fail the ladder's own contract (the
        numeric schema needs only {0.1, 0.5, 0.9}; the ladder needs all 13). Repair
        cannot change already-valid JSON, so the walk falls back deterministically
        instead of spending an LLM salvage call."""
        partial = '{"question_type": "numeric", "declared_percentiles": {"0.1": 10.0, "0.5": 50.0, "0.9": 90.0}}'
        text = rationale_with(numeric_block(2.0)) + rationale_with(partial)
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_numeric(text, PARSER_LLM)
        assert outcome.rung == "block"
        assert len(outcome.value) == 13
        assert float(outcome.value[0].value) == pytest.approx(2.0)
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mc_unmatched_option_final_block_yields_to_earlier_valid(self) -> None:
        bad_key = '{"question_type": "multiple_choice", "option_probs": {"Option A": 0.5, "Option Z": 0.5}}'
        text = rationale_with(mc_block([0.5, 0.3, 0.2])) + rationale_with(bad_key)
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_mc(text, OPTIONS, PARSER_LLM)
        assert outcome.rung == "block"
        probs = {o.option_name: o.probability for o in outcome.value.predicted_options}
        assert probs["Option A"] == pytest.approx(0.5, abs=0.02)
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_two_valid_numeric_blocks_last_wins(self) -> None:
        text = rationale_with(numeric_block(1.0)) + rationale_with(numeric_block(10.0))
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_numeric(text, PARSER_LLM)
        assert outcome.rung == "block"
        assert float(outcome.value[0].value) == pytest.approx(10.0)
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_two_valid_mc_blocks_last_wins(self) -> None:
        text = rationale_with(mc_block([0.5, 0.3, 0.2])) + rationale_with(mc_block([0.1, 0.2, 0.7]))
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_mc(text, OPTIONS, PARSER_LLM)
        assert outcome.rung == "block"
        probs = {o.option_name: o.probability for o in outcome.value.predicted_options}
        assert probs["Option C"] == pytest.approx(0.7, abs=0.02)
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unfenced_tail_scan_prefers_the_last_blob(self) -> None:
        """The tail scan carries the same position primacy: with two bare JSON objects
        in the tail, the LAST one is the model's final answer."""
        text = (
            "## Analysis\n\nreasoning...\n\n"
            '{"question_type": "binary", "posterior_prob": 0.40}\n'
            "on reflection:\n"
            '{"question_type": "binary", "posterior_prob": 0.72}\n'
        )
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_binary(text, PARSER_LLM)
        assert outcome.value == pytest.approx(0.72)
        assert outcome.rung == "repair"
        assert outcome.block_present is False
        llm.assert_not_awaited()


class TestRungLlm:
    """Rung 3: LLM salvage, loudly logged, strictly validated."""

    @pytest.mark.asyncio
    async def test_binary_no_block_salvaged(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot.value_extraction")
        llm_mock = AsyncMock(return_value=BinaryPrediction(prediction_in_decimal=0.61))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            outcome = await extract_binary("pure prose, no JSON anywhere. Chance feels ~61%.", PARSER_LLM)
        assert outcome == ExtractionOutcome(value=0.61, rung="llm", block_present=False)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "salvage" in r.getMessage()]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_numeric_partial_block_falls_to_llm(self) -> None:
        """A schema-valid block with only {0.1, 0.5, 0.9} must NOT be padded — llm rung supplies all 13."""
        partial = '{"question_type": "numeric", "declared_percentiles": {"0.1": 10.0, "0.5": 50.0, "0.9": 90.0}}'
        llm_mock = AsyncMock(return_value=full_percentile_list())
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            outcome = await extract_numeric(rationale_with(partial), PARSER_LLM)
        assert outcome.rung == "llm"
        assert outcome.block_present is True
        assert len(outcome.value) == 13
        llm_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mc_unknown_option_key_falls_to_llm(self) -> None:
        bad_key = '{"question_type": "multiple_choice", "option_probs": {"Option A": 0.5, "Option Z": 0.5}}'
        llm_mock = AsyncMock(return_value=make_pol([0.4, 0.35, 0.25]))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            outcome = await extract_mc(rationale_with(bad_key), OPTIONS, PARSER_LLM)
        assert outcome.rung == "llm"

    @pytest.mark.asyncio
    async def test_mc_llm_tolerant_two_stage_fallback(self) -> None:
        """llm rung mirrors the old strict→tolerant two-stage parse."""
        raw = [OptionProbability(option_name=n, probability=p) for n, p in zip(OPTIONS, [0.4, 0.35, 0.25])]

        calls: list[type] = []

        async def two_stage(text, output_type, parser_llm, *, prompt_notes=""):
            calls.append(output_type)
            if output_type is PredictedOptionList:
                raise ValidationError.from_exception_data(title="test", line_errors=[])
            return raw

        with patch("metaculus_bot.value_extraction.parse_structured", new=two_stage):
            outcome = await extract_mc("no block here at all", OPTIONS, PARSER_LLM)
        assert outcome.rung == "llm"
        assert calls == [PredictedOptionList, list[OptionProbability]]
        assert {o.option_name for o in outcome.value.predicted_options} == set(OPTIONS)

    @pytest.mark.asyncio
    async def test_llm_out_of_contract_value_rejected(self) -> None:
        """rung-3 output is validated, not trusted: a partial percentile set → ValueExtractionError.

        (BinaryPrediction self-validates [0, 1], so numeric is the type that can
        return an out-of-contract shape past the parser and must be caught by the
        ladder's own post-rung validation.)
        """
        partial = [Percentile(percentile=p, value=float(i)) for i, p in enumerate([0.1, 0.5, 0.9])]
        llm_mock = AsyncMock(return_value=partial)
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            with pytest.raises(ValueExtractionError):
                await extract_numeric("prose only", PARSER_LLM)

    @pytest.mark.asyncio
    async def test_prompt_notes_forwarded(self) -> None:
        captured: dict[str, str] = {}

        async def capture(text, output_type, parser_llm, *, prompt_notes=""):
            captured["notes"] = prompt_notes
            return full_percentile_list()

        with patch("metaculus_bot.value_extraction.parse_structured", new=capture):
            await extract_numeric("prose only", PARSER_LLM, prompt_notes="THE NOTES")
        assert captured["notes"] == "THE NOTES"


class TestRungFailure:
    """Rung 4: typed, loud, rung-annotated failure."""

    @pytest.mark.asyncio
    async def test_all_rungs_fail_raises_typed_error(self) -> None:
        llm_mock = AsyncMock(side_effect=ValueError("parser exploded"))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            with pytest.raises(ValueExtractionError) as excinfo:
                await extract_binary("no json anywhere", PARSER_LLM, question_id=7, model_name="m")
        msg = str(excinfo.value)
        assert "block:" in msg
        assert "llm:" in msg
        assert "question=7" in msg

    @pytest.mark.asyncio
    async def test_oversize_block_fatal_for_deterministic_rungs(self) -> None:
        """>200KB block: no parse, no repair; only the llm rung may salvage."""
        huge = '{"question_type": "binary", "posterior_prob": 0.5, "pad": "' + "x" * 210_000 + '"}'
        llm_mock = AsyncMock(side_effect=ValueError("nope"))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            with pytest.raises(ValueExtractionError) as excinfo:
                await extract_binary(rationale_with(huge), PARSER_LLM)
        assert "size cap" in str(excinfo.value)


class TestSalvageFidelity:
    """Every rung's output must be a value the rationale could have STATED.

    The LLM rung decodes under a schema, so handed a rationale with no forecast it *must*
    emit numbers — "absent" is not expressible. Shape checks (bounds, canonical set,
    sum-to-1) therefore cannot carry the documented "the parser can never smuggle in a
    fabricated value" claim on their own, and the post-rung validators are fidelity checks.
    """

    @pytest.mark.asyncio
    async def test_a_value_disordered_llm_salvage_fails_the_rung(self) -> None:
        """One out-of-place value must fail rather than be force-monotonized.

        ``sort_percentiles_by_value`` sorts by LABEL, so downstream never reorders a
        value-disordered set — it force-monotonizes it, which on one bad value pinned 10 of
        13 percentiles at the upper bound and published. A full reversal was already caught
        by the unit-mismatch guard; PARTIAL disorder was the hole.
        """
        disordered = full_percentile_list()
        disordered[1] = Percentile(percentile=disordered[1].percentile, value=999.0)
        llm_mock = AsyncMock(return_value=disordered)
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            with pytest.raises(ValueExtractionError) as excinfo:
                await extract_numeric("prose with no block at all", PARSER_LLM)

        assert "value-disordered" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_ties_are_still_accepted(self) -> None:
        # A repeated value is a legitimate concentrated (often count-like) declaration, and
        # the cluster spreader exists to separate exactly those — only a strict DECREASE
        # with rising percentile is incoherent.
        tied = full_percentile_list()
        tied[1] = Percentile(percentile=tied[1].percentile, value=tied[0].value)
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(return_value=tied)):
            outcome = await extract_numeric("prose with no block at all", PARSER_LLM)

        assert outcome.rung == "llm"
        assert [p.value for p in outcome.value] == [p.value for p in tied]

    @pytest.mark.asyncio
    async def test_a_non_finite_salvaged_value_fails_the_rung(self) -> None:
        with_nan = full_percentile_list()
        with_nan[6] = Percentile(percentile=with_nan[6].percentile, value=float("nan"))
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(return_value=with_nan)):
            with pytest.raises(ValueExtractionError) as excinfo:
                await extract_numeric("prose with no block at all", PARSER_LLM)

        assert "non-finite" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_a_truncated_numeric_literal_is_never_repaired(self) -> None:
        """``json_repair`` completes a value as readily as it completes syntax.

        A rationale cut mid-decimal leaves ``"posterior_prob": 0.`` inside a surviving
        fence; repairing that yields ``0.0``, which passes bounds validation and would
        publish as the binary clamp floor. The digits are gone, so any repair is invention —
        the rung refuses and falls through to the LLM salvage.
        """
        truncated = '{"question_type": "binary", "posterior_prob": 0.}'
        llm_mock = AsyncMock(return_value=BinaryPrediction(prediction_in_decimal=0.72))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            outcome = await extract_binary(rationale_with(truncated), PARSER_LLM)

        assert outcome.rung == "llm"
        assert outcome.value == 0.72

    @pytest.mark.asyncio
    async def test_prose_numerals_do_not_veto_a_faithful_repair(self) -> None:
        """Hyphenated numerals in prose fields ("2019-2023") are not truncated literals.

        The fidelity check lexes numeric runs OUTSIDE string literals, and prose lives
        inside them in EITHER quote style — single-quoted blocks are exactly what the
        repair rung exists to fix (see TestRungRepair.test_single_quotes), so a
        double-quote-only lexer read their prose as value position and refused a repair
        json_repair performs value-identically.
        """
        double_quoted = (
            '{"question_type": "binary", '
            '"base_rate": {"k": 3, "n": 4, "ref_class": "cuts in 2019-2023"}, '
            '"posterior_prob": 0.28,}'
        )
        single_quoted = (
            "{'question_type': 'binary', "
            "'base_rate': {'k': 3, 'n': 4, 'ref_class': 'cuts in 2019-2023'}, "
            "'posterior_prob': 0.28}"
        )
        for candidate in (double_quoted, single_quoted):
            with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
                outcome = await extract_binary(rationale_with(candidate), PARSER_LLM)
            assert outcome.rung == "repair", f"candidate quoting style wrongly refused: {candidate[:40]}"
            assert outcome.value == 0.28
            llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_truncated_literal_still_refuses_in_a_single_quoted_block(self) -> None:
        # The single-quote handling above must not blind the check to a genuinely
        # truncated value-position literal — the digits after "0." are gone in any
        # quoting style, so the rung refuses and falls through to the LLM salvage.
        truncated = "{'question_type': 'binary', 'posterior_prob': 0.}"
        llm_mock = AsyncMock(return_value=BinaryPrediction(prediction_in_decimal=0.72))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            outcome = await extract_binary(rationale_with(truncated), PARSER_LLM)

        assert outcome.rung == "llm"
        assert outcome.value == 0.72

    @pytest.mark.asyncio
    async def test_a_repair_may_drop_a_number_but_never_introduce_one(self) -> None:
        # Syntax-only repairs stay allowed: a trailing comma changes no value, so the
        # deterministic rung still handles the common malformed-block case (see
        # TestRungRepair). This pins the direction of the asymmetry.
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_binary(
                rationale_with('{"question_type": "binary", "posterior_prob": 0.28,}'), PARSER_LLM
            )

        assert outcome.rung == "repair"
        assert outcome.value == 0.28
        llm.assert_not_awaited()


class TestMcCanonicalization:
    @pytest.mark.asyncio
    async def test_case_and_whitespace_insensitive_block_keys(self) -> None:
        sloppy = '{"question_type": "multiple_choice", "option_probs": {" option a ": 0.5, "OPTION B": 0.3, "Option C": 0.2}}'
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_mc(rationale_with(sloppy), OPTIONS, PARSER_LLM)
        assert outcome.rung == "block"
        names = [o.option_name for o in outcome.value.predicted_options]
        assert names == OPTIONS  # canonical spellings, allowed order
        llm.assert_not_awaited()


# ---------------------------------------------------------------------------
# Property-based: corrupted blocks never escape the contract
# ---------------------------------------------------------------------------

_STRUCTURAL = '{}[],":'


def _corrupt(block: str, mode: int, index: int) -> str:
    if mode == 0:  # truncate
        return block[: max(1, index % max(1, len(block)))]
    if mode == 1:  # delete structural chars
        out, removed = [], 0
        for i, ch in enumerate(block):
            if ch in _STRUCTURAL and (i + index) % 4 == 0:
                removed += 1
                continue
            out.append(ch)
        return "".join(out)
    if mode == 2:  # inject junk lines
        cut = index % max(1, len(block))
        return block[:cut] + "\nlorem ipsum {{{ noise\n" + block[cut:]
    return block + "\n" + block  # duplicate


@st.composite
def corrupted_binary_rationale(draw: st.DrawFn) -> str:
    prob = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    block = f'{{"question_type": "binary", "posterior_prob": {prob:.4f}}}'
    mode = draw(st.integers(min_value=0, max_value=3))
    index = draw(st.integers(min_value=0, max_value=500))
    return rationale_with(_corrupt(block, mode, index))


@st.composite
def corrupted_numeric_rationale(draw: st.DrawFn) -> str:
    base = draw(st.floats(min_value=0.1, max_value=1e6, allow_nan=False))
    pcts = ", ".join(f'"{p}": {base * (i + 1):.4f}' for i, p in enumerate(STANDARD_PERCENTILES))
    block = f'{{"question_type": "numeric", "declared_percentiles": {{{pcts}}}, "outcome_type": "continuous"}}'
    mode = draw(st.integers(min_value=0, max_value=3))
    index = draw(st.integers(min_value=0, max_value=800))
    return rationale_with(_corrupt(block, mode, index))


@st.composite
def corrupted_mc_rationale(draw: st.DrawFn) -> str:
    a = draw(st.floats(min_value=0.01, max_value=0.98, allow_nan=False))
    b = draw(st.floats(min_value=0.01, max_value=max(0.011, 0.99 - a), allow_nan=False))
    c = max(0.0, 1.0 - a - b)
    block = (
        f'{{"question_type": "multiple_choice", "option_probs": '
        f'{{"Option A": {a:.4f}, "Option B": {b:.4f}, "Option C": {c:.4f}}}}}'
    )
    mode = draw(st.integers(min_value=0, max_value=3))
    index = draw(st.integers(min_value=0, max_value=500))
    return rationale_with(_corrupt(block, mode, index))


_HYPO_SETTINGS = settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])


class TestPropertyContract:
    """With rung 3 disabled, the ladder returns an in-contract value or raises ValueExtractionError."""

    @given(text=corrupted_binary_rationale())
    @_HYPO_SETTINGS
    @pytest.mark.asyncio
    async def test_binary_contract(self, text: str) -> None:
        llm_dead = AsyncMock(side_effect=ValueError("rung 3 disabled"))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_dead):
            try:
                outcome = await extract_binary(text, PARSER_LLM)
            except ValueExtractionError:
                return
        assert 0.0 <= outcome.value <= 1.0
        assert outcome.rung in ("block", "repair")

    @given(text=corrupted_numeric_rationale())
    @_HYPO_SETTINGS
    @pytest.mark.asyncio
    async def test_numeric_contract(self, text: str) -> None:
        llm_dead = AsyncMock(side_effect=ValueError("rung 3 disabled"))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_dead):
            try:
                outcome = await extract_numeric(text, PARSER_LLM)
            except ValueExtractionError:
                return
        assert [float(p.percentile) for p in outcome.value] == STANDARD_PERCENTILES

    @given(text=corrupted_mc_rationale())
    @_HYPO_SETTINGS
    @pytest.mark.asyncio
    async def test_mc_contract(self, text: str) -> None:
        llm_dead = AsyncMock(side_effect=ValueError("rung 3 disabled"))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_dead):
            try:
                outcome = await extract_mc(text, OPTIONS, PARSER_LLM)
            except ValueExtractionError:
                return
        assert {o.option_name for o in outcome.value.predicted_options} == set(OPTIONS)
        total = sum(o.probability for o in outcome.value.predicted_options)
        assert total == pytest.approx(1.0, abs=0.02)
