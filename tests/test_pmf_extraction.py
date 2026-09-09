"""``extract_pmf``: the extraction ladder for a per-bin block on an enumerable grid.

A forecaster on a coarse Mantic grid declares one probability per bin, keyed by the bin's label,
plus ``below_range`` / ``above_range`` where the bound is open. The ladder mirrors ``extract_mc``:
block keys are fold-matched onto the grid's keys (both sides through ``fold_bin_label``), a
duplicate fold sums, an unknown key fails the rung, a reserved key on a closed bound fails the
rung, and EVERY grid key must be present so a block truncated before its last bins falls through
instead of publishing a partial declaration. The value is the platform's ``N + 2`` PMF vector
``[below, p_0, ..., p_{N-1}, above]`` with 0.0 in a closed bound's tail slot.

The grids here are built directly from the frozen ``PmfGrid`` dataclass, not from questions:
this file tests the ladder, and ``tests/test_numeric_pmf_grid.py`` tests the labelling.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metaculus_bot.constants import PMF_ABOVE_RANGE_KEY, PMF_BELOW_RANGE_KEY
from metaculus_bot.exceptions import ValueExtractionError
from metaculus_bot.numeric.pmf_grid import PmfGrid
from metaculus_bot.structured_parse import BinProbability
from metaculus_bot.value_extraction import PmfForecast, extract_pmf
from scripts.telemetry.markers import MARKER_SPECS

PARSER_LLM = MagicMock()

# Question 651's shape: twelve trading days, both bounds closed, so the keys are the labels alone.
DAY_LABELS = tuple(f"2026-09-{day:02d}" for day in range(8, 20))
DAY_GRID = PmfGrid(
    labels=DAY_LABELS,
    edges=tuple(1_788_998_400.0 + 86_400.0 * k for k in range(len(DAY_LABELS) + 1)),
    open_lower_bound=False,
    open_upper_bound=False,
    style="day",
)
# The same twelve days with both bounds open: fourteen keys, the shape a per-key sum tolerance is measured on.
OPEN_DAY_GRID = PmfGrid(
    labels=DAY_LABELS,
    edges=DAY_GRID.edges,
    open_lower_bound=True,
    open_upper_bound=True,
    style="day",
)
# A count grid 0..7 with an open ceiling: the keys are the eight counts plus ``above_range``.
COUNT_LABELS = tuple(str(k) for k in range(8))
COUNT_GRID = PmfGrid(
    labels=COUNT_LABELS,
    edges=tuple(k - 0.5 for k in range(9)),
    open_lower_bound=False,
    open_upper_bound=True,
    style="center",
)
# Post 650's label shape on a coarse grid, both bounds open: ``below_range`` first, ``above_range`` last.
PRICE_LABELS = tuple(str(55_000 + 100 * k) for k in range(5))
PRICE_GRID = PmfGrid(
    labels=PRICE_LABELS,
    edges=tuple(54_950.0 + 100.0 * k for k in range(6)),
    open_lower_bound=True,
    open_upper_bound=True,
    style="center",
)
# The one style whose labels do not survive ``fold_bin_label`` unchanged (T and Z lowercase).
TIMESTAMP_LABELS: tuple[str, ...] = (
    "2026-09-08T00:00:00Z to 2026-09-08T06:00:00Z",
    "2026-09-08T06:00:00Z to 2026-09-08T12:00:00Z",
    "2026-09-08T12:00:00Z to 2026-09-08T18:00:00Z",
)
TIMESTAMP_GRID = PmfGrid(
    labels=TIMESTAMP_LABELS,
    edges=tuple(1_788_998_400.0 + 21_600.0 * k for k in range(4)),
    open_lower_bound=False,
    open_upper_bound=False,
    style="timestamp",
)


def rationale_with(block_json: str) -> str:
    return f"## Analysis\n\nSome careful reasoning here.\n\n```json\n{block_json}\n```\n"


def pmf_block(bin_probs: dict[str, float], *, trailing_comma: bool = False) -> str:
    body = ", ".join(f"{json.dumps(key)}: {prob}" for key, prob in bin_probs.items())
    tail = "," if trailing_comma else ""
    return f'{{"question_type": "pmf", "bin_probs": {{{body}}}{tail}}}'


def even_probs(keys: tuple[str, ...]) -> dict[str, float]:
    """Round-number probabilities over ``keys`` summing to exactly 1.0 in decimal."""
    share = round(1.0 / len(keys), 6)
    probs = dict.fromkeys(keys, share)
    probs[keys[-1]] = round(1.0 - share * (len(keys) - 1), 6)
    return probs


def day_probs() -> dict[str, float]:
    """A member fairly sure of 2026-09-16 with the rest spread over three neighbours."""
    probs = dict.fromkeys(DAY_LABELS, 0.0)
    probs["2026-09-15"] = 0.1
    probs["2026-09-16"] = 0.7
    probs["2026-09-17"] = 0.15
    probs["2026-09-18"] = 0.05
    return probs


def salvage_bins(bin_probs: dict[str, float]) -> list[BinProbability]:
    return [BinProbability(label=label, probability=prob) for label, prob in bin_probs.items()]


def fourteen_key_probs(*, last: float) -> dict[str, float]:
    """Thirteen keys at 0.07 (0.91 together) plus ``last`` on the fourteenth: 0.15 sums to 1.06, 0.17 to 1.08."""
    probs = dict.fromkeys(OPEN_DAY_GRID.keys, 0.07)
    probs[OPEN_DAY_GRID.keys[-1]] = last
    return probs


def extraction_rung_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "EXTRACTION_RUNG:" in r.getMessage()]


class TestBlockRung:
    @pytest.mark.asyncio
    async def test_a_closed_grid_block_becomes_the_n_plus_2_vector_with_zero_tails(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot.value_extraction")
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_pmf(
                rationale_with(pmf_block(day_probs())), DAY_GRID, PARSER_LLM, question_id=651, model_name="m"
            )
        assert outcome.rung == "block"
        assert outcome.block_present is True
        assert isinstance(outcome.value, PmfForecast)
        expected = [0.0, *day_probs().values(), 0.0]
        assert outcome.value.declared == pytest.approx(expected)
        assert len(outcome.value.declared) == len(DAY_LABELS) + 2
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_ladder_logs_qtype_pmf_on_the_extraction_rung_marker(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``qtype=pmf`` is an additive token on an existing marker; the registry regex must harvest it."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.value_extraction")
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()):
            await extract_pmf(
                rationale_with(pmf_block(day_probs())), DAY_GRID, PARSER_LLM, question_id=651, model_name="test-model"
            )
        lines = extraction_rung_lines(caplog)
        assert len(lines) == 1
        assert "question=651 model=test-model qtype=pmf rung=block block_present=True" in lines[0]
        spec = next(spec for spec in MARKER_SPECS if spec.name == "extraction_rung")
        match = spec.regex.search(lines[0])
        assert match is not None
        assert match.group("qtype") == "pmf"
        assert match.group("rung") == "block"

    @pytest.mark.asyncio
    async def test_open_bound_mass_lands_in_the_tail_slots(self) -> None:
        probs = dict.fromkeys(PRICE_GRID.keys, 0.0)
        probs[PMF_BELOW_RANGE_KEY] = 0.1
        probs["55200"] = 0.6
        probs[PMF_ABOVE_RANGE_KEY] = 0.3
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_pmf(rationale_with(pmf_block(probs)), PRICE_GRID, PARSER_LLM)
        assert outcome.rung == "block"
        assert outcome.value.declared == pytest.approx([0.1, 0.0, 0.0, 0.6, 0.0, 0.0, 0.3])
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_closed_lower_and_open_upper_grid_zeroes_only_the_closed_tail(self) -> None:
        probs = dict.fromkeys(COUNT_GRID.keys, 0.0)
        probs["2"] = 0.5
        probs["3"] = 0.3
        probs[PMF_ABOVE_RANGE_KEY] = 0.2
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()):
            outcome = await extract_pmf(rationale_with(pmf_block(probs)), COUNT_GRID, PARSER_LLM)
        assert outcome.value.declared == pytest.approx([0.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.2])

    @pytest.mark.asyncio
    async def test_the_vector_is_in_grid_order_whatever_order_the_block_used(self) -> None:
        probs = day_probs()
        reversed_block = pmf_block(dict(reversed(list(probs.items()))))
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()):
            outcome = await extract_pmf(rationale_with(reversed_block), DAY_GRID, PARSER_LLM)
        assert outcome.value.declared == pytest.approx([0.0, *probs.values(), 0.0])


class TestKeyFolding:
    """Block keys match grid keys through ``fold_bin_label`` on BOTH sides, as the MC canonical map does."""

    @pytest.mark.asyncio
    async def test_a_count_spelled_as_a_float_matches_its_bin(self) -> None:
        probs = dict.fromkeys(COUNT_GRID.keys, 0.0)
        probs["7.0"] = 1.0
        del probs["7"]
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_pmf(rationale_with(pmf_block(probs)), COUNT_GRID, PARSER_LLM)
        assert outcome.rung == "block"
        assert outcome.value.declared[1 + 7] == 1.0
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_padded_date_matches_its_bin(self) -> None:
        probs = dict.fromkeys(DAY_LABELS, 0.0)
        del probs["2026-09-16"]
        probs[" 2026-09-16 "] = 1.0
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_pmf(rationale_with(pmf_block(probs)), DAY_GRID, PARSER_LLM)
        assert outcome.rung == "block"
        assert outcome.value.declared[1 + DAY_LABELS.index("2026-09-16")] == 1.0
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_thousands_separated_price_matches_its_bin(self) -> None:
        probs = dict.fromkeys(PRICE_GRID.keys, 0.0)
        del probs["55000"]
        probs["55,000"] = 1.0
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_pmf(rationale_with(pmf_block(probs)), PRICE_GRID, PARSER_LLM)
        assert outcome.rung == "block"
        assert outcome.value.declared[1] == 1.0
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_timestamp_label_matches_although_its_own_fold_differs_from_it(self) -> None:
        """On this style ``fold_bin_label(label) != label``, so matching the raw grid key against a
        folded block key would fail every bin; the grid's keys are folded too."""
        probs = dict.fromkeys(TIMESTAMP_LABELS, 0.0)
        probs[TIMESTAMP_LABELS[1]] = 1.0
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_pmf(rationale_with(pmf_block(probs)), TIMESTAMP_GRID, PARSER_LLM)
        assert outcome.rung == "block"
        assert outcome.value.declared == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0])
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_duplicate_folded_keys_sum_onto_one_bin(self) -> None:
        probs = dict.fromkeys(COUNT_GRID.keys, 0.0)
        probs["7"] = 0.2
        probs["7.0"] = 0.3
        probs[" 7 "] = 0.5
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()):
            outcome = await extract_pmf(rationale_with(pmf_block(probs)), COUNT_GRID, PARSER_LLM)
        assert outcome.rung == "block"
        assert outcome.value.declared[1 + 7] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_a_reserved_key_folds_case_insensitively(self) -> None:
        probs = dict.fromkeys(COUNT_GRID.keys, 0.0)
        del probs[PMF_ABOVE_RANGE_KEY]
        probs["Above_Range"] = 1.0
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()):
            outcome = await extract_pmf(rationale_with(pmf_block(probs)), COUNT_GRID, PARSER_LLM)
        assert outcome.rung == "block"
        assert outcome.value.declared[-1] == 1.0


class TestBlockRungRefusals:
    """Each refusal fails the deterministic rungs and falls through to the LLM salvage rung; when that
    rung fails too, the typed error names the reason so the drop is diagnosable from the log."""

    @pytest.mark.asyncio
    async def test_a_block_missing_a_bin_falls_through_rather_than_publishing_a_partial_declaration(self) -> None:
        probs = day_probs()
        del probs["2026-09-19"]
        probs["2026-09-16"] += 0.0  # the sum is still 1.0: only the key is absent
        llm_mock = AsyncMock(return_value=salvage_bins(day_probs()))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            outcome = await extract_pmf(rationale_with(pmf_block(probs)), DAY_GRID, PARSER_LLM)
        assert outcome.rung == "llm"
        assert outcome.block_present is True
        llm_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_missing_bin_is_named_in_the_typed_error(self) -> None:
        probs = day_probs()
        del probs["2026-09-19"]
        with (
            patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(side_effect=ValueError("no"))),
            pytest.raises(ValueExtractionError, match=r"missing bin\(s\) \['2026-09-19'\]"),
        ):
            await extract_pmf(rationale_with(pmf_block(probs)), DAY_GRID, PARSER_LLM)

    @pytest.mark.asyncio
    async def test_a_block_with_an_unknown_key_fails_the_rung(self) -> None:
        probs = day_probs()
        probs["2026-09-16"] -= 0.1
        probs["2026-09-20"] = 0.1
        with (
            patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(side_effect=ValueError("no"))),
            pytest.raises(ValueExtractionError, match=r"block key '2026-09-20' matches no bin of this grid"),
        ):
            await extract_pmf(rationale_with(pmf_block(probs)), DAY_GRID, PARSER_LLM)

    @pytest.mark.asyncio
    async def test_a_non_iso_date_spelling_is_an_unknown_key(self) -> None:
        """Dates fold as text: ``2026-9-16`` is not ``2026-09-16`` and falls to the later rungs,
        exactly as an unmatched multiple-choice key does."""
        probs = day_probs()
        del probs["2026-09-16"]
        probs["2026-9-16"] = 0.7
        llm_mock = AsyncMock(return_value=salvage_bins(day_probs()))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            outcome = await extract_pmf(rationale_with(pmf_block(probs)), DAY_GRID, PARSER_LLM)
        assert outcome.rung == "llm"

    @pytest.mark.asyncio
    async def test_below_range_on_a_closed_lower_bound_fails_the_rung(self) -> None:
        probs = dict.fromkeys(COUNT_GRID.keys, 0.0)
        probs["3"] = 0.9
        probs[PMF_BELOW_RANGE_KEY] = 0.1
        with (
            patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(side_effect=ValueError("no"))),
            pytest.raises(ValueExtractionError, match=r"'below_range' .*lower bound is closed"),
        ):
            await extract_pmf(rationale_with(pmf_block(probs)), COUNT_GRID, PARSER_LLM)

    @pytest.mark.asyncio
    async def test_above_range_on_a_closed_upper_bound_fails_the_rung(self) -> None:
        probs = day_probs()
        probs["2026-09-16"] -= 0.1
        probs[PMF_ABOVE_RANGE_KEY] = 0.1
        with (
            patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(side_effect=ValueError("no"))),
            pytest.raises(ValueExtractionError, match=r"'above_range' .*upper bound is closed"),
        ):
            await extract_pmf(rationale_with(pmf_block(probs)), DAY_GRID, PARSER_LLM)

    @pytest.mark.asyncio
    async def test_a_numeric_block_does_not_satisfy_a_per_bin_ladder(self) -> None:
        """The ``question_type`` mismatch guard refuses a percentile block on a per-bin ladder."""
        numeric_block = '{"question_type": "numeric", "declared_percentiles": {"0.1": 1.0, "0.5": 2.0, "0.9": 3.0}}'
        llm_mock = AsyncMock(return_value=salvage_bins(day_probs()))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            outcome = await extract_pmf(rationale_with(numeric_block), DAY_GRID, PARSER_LLM)
        assert outcome.rung == "llm"

    @pytest.mark.asyncio
    async def test_a_fourteen_key_block_off_by_two_decimal_rounding_passes_the_block_rung(self) -> None:
        """Two-decimal rounding drifts up to 0.005 a key, so fourteen keys get 0.07 of slack and a block
        summing to 1.06 is an honest declaration, not a fabrication; downstream normalises it."""
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_pmf(
                rationale_with(pmf_block(fourteen_key_probs(last=0.15))), OPEN_DAY_GRID, PARSER_LLM
            )
        assert outcome.rung == "block"
        assert sum(outcome.value.declared) == pytest.approx(1.06)
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_block_whose_probabilities_do_not_sum_to_one_fails_the_rung(self) -> None:
        probs = day_probs()
        probs["2026-09-16"] = 0.3  # the block now sums to 0.6
        llm_mock = AsyncMock(return_value=salvage_bins(day_probs()))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            outcome = await extract_pmf(rationale_with(pmf_block(probs)), DAY_GRID, PARSER_LLM)
        assert outcome.rung == "llm"


class TestRepairRung:
    @pytest.mark.asyncio
    async def test_a_trailing_comma_is_repaired_without_the_parser(self) -> None:
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock()) as llm:
            outcome = await extract_pmf(
                rationale_with(pmf_block(day_probs(), trailing_comma=True)), DAY_GRID, PARSER_LLM
            )
        assert outcome.rung == "repair"
        assert outcome.value.declared == pytest.approx([0.0, *day_probs().values(), 0.0])
        llm.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_block_cut_mid_number_is_never_completed(self) -> None:
        """The rationale stops inside the last probability: ``"2026-09-19": 0.`` is a truncated literal
        ``json_repair`` would complete by inventing digits, so the repair rung refuses and the ladder
        goes to the LLM rung."""
        probs = day_probs()
        probs["2026-09-16"] = 0.65
        probs["2026-09-19"] = 0.05
        block = pmf_block(probs)
        truncated = block[: block.rfind("0.05") + len("0.")]
        llm_mock = AsyncMock(return_value=salvage_bins(probs))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            outcome = await extract_pmf(rationale_with(truncated), DAY_GRID, PARSER_LLM)
        assert outcome.rung == "llm"
        assert outcome.value.declared == pytest.approx([0.0, *probs.values(), 0.0])
        llm_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_block_cut_between_bins_is_repaired_but_fails_the_every_key_rule(self) -> None:
        """A cut after a complete number leaves nothing for the fidelity check to refuse; ``json_repair``
        closes the braces and the repaired block is a valid PARTIAL declaration. Requiring every grid
        key is what stops it publishing."""
        probs = dict.fromkeys(DAY_LABELS, 0.0)
        probs["2026-09-15"] = 0.3
        probs["2026-09-16"] = 0.7
        block = pmf_block(probs)
        truncated = block[: block.rfind('"2026-09-17"') - len(", ")]
        llm_mock = AsyncMock(return_value=salvage_bins(probs))
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            outcome = await extract_pmf(rationale_with(truncated), DAY_GRID, PARSER_LLM)
        assert outcome.rung == "llm"
        llm_mock.assert_awaited_once()


class TestLlmRung:
    @pytest.mark.asyncio
    async def test_salvage_runs_the_same_conversion_as_the_block_rung(self) -> None:
        bins = salvage_bins({**dict.fromkeys(COUNT_GRID.keys, 0.0), "7.0": 0.4, " 7 ": 0.2, "above_range": 0.4})
        llm_mock = AsyncMock(return_value=bins)
        with patch("metaculus_bot.value_extraction.parse_structured", new=llm_mock):
            outcome = await extract_pmf("prose with no block", COUNT_GRID, PARSER_LLM, prompt_notes="PMF NOTES")
        assert outcome.rung == "llm"
        assert outcome.block_present is False
        assert llm_mock.await_args is not None
        assert llm_mock.await_args.args[1] == list[BinProbability]
        assert llm_mock.await_args.kwargs["prompt_notes"] == "PMF NOTES"
        assert outcome.value.declared == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.4])

    @pytest.mark.asyncio
    async def test_a_salvage_missing_a_bin_is_refused(self) -> None:
        partial = salvage_bins(day_probs())[:-1]
        with (
            patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(return_value=partial)),
            pytest.raises(ValueExtractionError, match=r"missing bin\(s\)"),
        ):
            await extract_pmf("prose only", DAY_GRID, PARSER_LLM)

    @pytest.mark.asyncio
    async def test_a_salvage_with_an_unknown_key_is_refused(self) -> None:
        bins = salvage_bins({**day_probs(), "2026-09-20": 0.0})
        with (
            patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(return_value=bins)),
            pytest.raises(ValueExtractionError, match="matches no bin"),
        ):
            await extract_pmf("prose only", DAY_GRID, PARSER_LLM)

    @pytest.mark.asyncio
    async def test_a_salvage_whose_mass_does_not_sum_to_one_is_refused(self) -> None:
        """The parser LLM decodes under a schema, so it must emit numbers; the sum check is a fidelity
        check on a value the rationale may never have stated."""
        bins = salvage_bins({**dict.fromkeys(DAY_LABELS, 0.0), "2026-09-16": 0.5})
        with (
            patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(return_value=bins)),
            pytest.raises(ValueExtractionError, match="sum to"),
        ):
            await extract_pmf("prose only", DAY_GRID, PARSER_LLM)

    @pytest.mark.asyncio
    async def test_the_sum_tolerance_scales_with_the_key_count(self) -> None:
        """Fourteen keys give 0.07 of slack: off by 0.06 is accepted, off by 0.08 is refused."""
        accepted = AsyncMock(return_value=salvage_bins(fourteen_key_probs(last=0.15)))
        with patch("metaculus_bot.value_extraction.parse_structured", new=accepted):
            outcome = await extract_pmf("prose only", OPEN_DAY_GRID, PARSER_LLM)
        assert outcome.rung == "llm"
        assert sum(outcome.value.declared) == pytest.approx(1.06)

        refused = AsyncMock(return_value=salvage_bins(fourteen_key_probs(last=0.17)))
        with (
            patch("metaculus_bot.value_extraction.parse_structured", new=refused),
            pytest.raises(ValueExtractionError, match="sum to"),
        ):
            await extract_pmf("prose only", OPEN_DAY_GRID, PARSER_LLM)

    @pytest.mark.asyncio
    async def test_the_sum_tolerance_floor_is_the_ballot_tolerance_on_a_short_grid(self) -> None:
        """Three keys would earn 0.015 of slack; the floor keeps the multiple-choice ballot's 0.02, so off by 0.03 fails."""
        bins = salvage_bins(dict(zip(TIMESTAMP_LABELS, (0.5, 0.3, 0.23), strict=True)))
        with (
            patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(return_value=bins)),
            pytest.raises(ValueExtractionError, match="sum to"),
        ):
            await extract_pmf("prose only", TIMESTAMP_GRID, PARSER_LLM)

    @pytest.mark.asyncio
    async def test_a_salvage_with_an_out_of_range_probability_is_refused(self) -> None:
        bins = salvage_bins({**dict.fromkeys(DAY_LABELS, 0.0), "2026-09-16": 1.5, "2026-09-17": -0.5})
        with (
            patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(return_value=bins)),
            pytest.raises(ValueExtractionError, match=r"outside \[0, 1\]"),
        ):
            await extract_pmf("prose only", DAY_GRID, PARSER_LLM)

    @pytest.mark.asyncio
    async def test_all_rungs_failing_raises_the_typed_error_with_qtype_pmf(self) -> None:
        with (
            patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(side_effect=ValueError("boom"))),
            pytest.raises(ValueExtractionError, match="qtype=pmf question=651"),
        ):
            await extract_pmf("no json anywhere", DAY_GRID, PARSER_LLM, question_id=651)
