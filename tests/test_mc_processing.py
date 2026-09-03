"""Tests for ``metaculus_bot.mc_processing`` — the MC ballot builder and clamp.

The headline contract (H1, 2026-08-25): ``build_mc_prediction`` never invents a
ballot. When no parsed option matches the question's options it raises, so the
extraction ladder falls through to its typed failure and the forecaster is
dropped, instead of publishing an exact-1/n uniform forecast the model never made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import PredictedOptionList
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

from metaculus_bot.constants import MC_PROB_MAX, MC_PROB_MIN
from metaculus_bot.exceptions import ValueExtractionError
from metaculus_bot.mc_processing import (
    accumulate_declared_option_probs,
    build_mc_prediction,
    clamp_and_renormalize_probs,
)
from metaculus_bot.simple_types import OptionProbability
from metaculus_bot.value_extraction import extract_mc

OPTIONS = ["Alpha", "Beta", "Gamma"]


def _raw(pairs: list[tuple[str, float]]) -> list[OptionProbability]:
    return [OptionProbability(option_name=name, probability=prob) for name, prob in pairs]


class TestBuildMcPredictionRefusesToInvent:
    """No match, no forecast."""

    def test_empty_parse_raises(self) -> None:
        with pytest.raises(ValueError, match="refusing to impute a uniform ballot"):
            build_mc_prediction([], OPTIONS)

    def test_all_names_unmatched_raises(self) -> None:
        with pytest.raises(ValueError, match="refusing to impute a uniform ballot"):
            build_mc_prediction(_raw([("Delta", 0.6), ("Epsilon", 0.4)]), OPTIONS)

    def test_error_names_both_option_sets(self) -> None:
        with pytest.raises(ValueError, match="no parsed option matched") as excinfo:
            build_mc_prediction(_raw([("Delta", 1.0)]), OPTIONS)
        message = str(excinfo.value)
        assert "Alpha" in message
        assert "Delta" in message


class TestBuildMcPredictionHappyPaths:
    """The behaviour the refusal must not disturb."""

    def test_canonical_order_and_probabilities(self) -> None:
        pol = build_mc_prediction(_raw([("Gamma", 0.2), ("Alpha", 0.5), ("Beta", 0.3)]), OPTIONS)
        assert [o.option_name for o in pol.predicted_options] == OPTIONS
        assert [o.probability for o in pol.predicted_options] == pytest.approx([0.5, 0.3, 0.2], abs=1e-9)

    def test_case_and_prefix_variants_match(self) -> None:
        pol = build_mc_prediction(_raw([("  alpha ", 0.5), ("BETA", 0.3), ("Option Gamma", 0.2)]), OPTIONS)
        assert [o.option_name for o in pol.predicted_options] == OPTIONS

    def test_duplicate_names_are_summed(self) -> None:
        pol = build_mc_prediction(_raw([("Alpha", 0.3), ("Alpha", 0.2), ("Beta", 0.3), ("Gamma", 0.2)]), OPTIONS)
        probs = {o.option_name: o.probability for o in pol.predicted_options}
        assert probs["Alpha"] == pytest.approx(0.5, abs=1e-9)

    def test_partial_ballot_keeps_only_matched_options(self) -> None:
        """A partial ballot is NOT padded; the ladder's ``_validate_mc`` rejects it on the
        option-set comparison, which is how a partial parse fails its rung."""
        pol = build_mc_prediction(_raw([("Alpha", 0.7), ("Beta", 0.3)]), OPTIONS)
        assert [o.option_name for o in pol.predicted_options] == ["Alpha", "Beta"]


class TestExtractionLadderDropsInsteadOfImputing:
    """End-to-end: an empty rung-3 parse is a DROP, not a uniform publish.

    ``parse_structured(text, list[OptionProbability])`` validates an empty list, so
    before the fix the ladder handed ``build_mc_prediction`` nothing and returned
    exact 1/n on every option — full option set, in bounds, sums to 1, so every
    downstream check passed and it published with ``rung=llm`` as its only trace.
    """

    @pytest.mark.asyncio
    async def test_empty_llm_ballot_raises_value_extraction_error(self) -> None:
        async def parse(text, output_type, parser_llm, *, prompt_notes=""):
            if output_type is PredictedOptionList:
                return PredictedOptionList(predicted_options=[])
            return []

        with (
            patch("metaculus_bot.value_extraction.parse_structured", new=parse),
            pytest.raises(ValueExtractionError),
        ):
            await extract_mc("prose with no ballot in it", OPTIONS, MagicMock(), question_id=5, model_name="m")

    @pytest.mark.asyncio
    async def test_unmatched_llm_ballot_raises_value_extraction_error(self) -> None:
        unmatched = [OptionProbability(option_name="Delta", probability=1.0)]

        async def parse(text, output_type, parser_llm, *, prompt_notes=""):
            if output_type is PredictedOptionList:
                return PredictedOptionList(predicted_options=[PredictedOption(option_name="Delta", probability=1.0)])
            return unmatched

        with (
            patch("metaculus_bot.value_extraction.parse_structured", new=parse),
            pytest.raises(ValueExtractionError),
        ):
            await extract_mc("prose about something else", OPTIONS, MagicMock(), question_id=5, model_name="m")

    @pytest.mark.asyncio
    async def test_good_llm_ballot_still_salvages(self) -> None:
        """The salvage rung keeps working for output that carries a real ballot."""
        llm = AsyncMock(
            return_value=PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name=name, probability=prob)
                    for name, prob in zip(OPTIONS, [0.4, 0.35, 0.25], strict=False)
                ]
            )
        )
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm):
            outcome = await extract_mc("no block here", OPTIONS, MagicMock())
        assert outcome.rung == "llm"
        probs = {o.option_name: o.probability for o in outcome.value.option_list.predicted_options}
        assert probs["Alpha"] == pytest.approx(0.4, abs=0.02)


class TestClampAndRenormalize:
    def test_in_bounds_sum_one(self) -> None:
        clamped = clamp_and_renormalize_probs([0.984, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002])
        assert sum(clamped) == pytest.approx(1.0, abs=1e-9)
        assert all(MC_PROB_MIN <= p <= MC_PROB_MAX for p in clamped)

    def test_empty_input_returns_empty(self) -> None:
        assert clamp_and_renormalize_probs([]) == []

    def test_zero_mass_raises_when_the_floor_is_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pins the day-MC_PROB_MIN-becomes-0 behaviour: raise, never [1/n] * n.

        With the 0.01 floor in place this branch is unreachable, which is exactly
        why it went unnoticed as a silent uniform-manufacture site.
        """
        monkeypatch.setattr("metaculus_bot.mc_processing.MC_PROB_MIN", 0.0)
        with pytest.raises(ValueError, match="carry no mass"):
            clamp_and_renormalize_probs([0.0, 0.0, 0.0])


class TestAccumulateDeclaredOptionProbs:
    """The shared accumulation behind build_mc_prediction and the ladder's declared vector."""

    def test_pairs_are_canonical_ordered_summed_and_unclamped(self):
        raw = [
            OptionProbability(option_name="option b", probability=0.001),
            OptionProbability(option_name="Option A", probability=0.6),
            OptionProbability(option_name="A", probability=0.3),
            OptionProbability(option_name="Zed", probability=0.099),
        ]
        pairs = accumulate_declared_option_probs(raw, ["A", "B", "C"])

        assert pairs == [("A", pytest.approx(0.9)), ("B", 0.001)]

    def test_build_mc_prediction_is_the_same_pairs_clamped(self):
        raw = [
            OptionProbability(option_name="A", probability=0.995),
            OptionProbability(option_name="B", probability=0.003),
            OptionProbability(option_name="C", probability=0.002),
        ]
        options = ["A", "B", "C"]
        pairs = accumulate_declared_option_probs(raw, options)
        pol = build_mc_prediction(raw, options)

        assert [o.option_name for o in pol.predicted_options] == [name for name, _ in pairs]
        assert [o.probability for o in pol.predicted_options] == clamp_and_renormalize_probs([p for _, p in pairs])
