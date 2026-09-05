"""Tests for the per-member MEMBER_FORECAST telemetry (metaculus_bot/member_forecast.py).

Three layers, deliberately:

* the pure formatter: one shape for all three question types and both roles, with
  whitespace-free compact JSON values;
* the round trip through the ``member_forecast`` MarkerSpec: every field, the JSON vectors
  included, survives ``parse_log_text`` verbatim and ``json.loads`` back to exactly the
  values the formatter was handed. That is the archive's contract, and it is what the
  2026-09-02 clip-threshold re-read lacked (a raw binary probability was recoverable for 74
  of 451 resolved binaries, from the trim-lossy comment alone);
* the emitters: the runner and stacker paths log the line carrying the RAW value when the
  clamp moved it, the MC vector is read before the in-place clamp rewrites it, and the
  numeric line precedes the unit-mismatch guard so a withheld member still leaves one.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
from collections import defaultdict
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
    Percentile,
    PredictedOption,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.data_organizer import PredictionTypes

from metaculus_bot import stacking
from metaculus_bot.aggregation_pipeline import AggregationPipeline
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN, MC_PROB_MIN
from metaculus_bot.exceptions import UnitMismatchError
from metaculus_bot.forecaster_runners import run_binary_forecast, run_mc_forecast, run_numeric_forecast
from metaculus_bot.member_forecast import (
    MEMBER_FORECAST_ROLE_MEMBER,
    MEMBER_FORECAST_ROLE_STACKER,
    format_member_forecast_marker,
    option_vector,
    percentile_pairs,
)
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
from metaculus_bot.simple_types import OptionProbability
from metaculus_bot.value_extraction import ExtractionOutcome, McForecast, extract_mc
from scripts.telemetry.markers import MARKER_SPECS, parse_log_text
from tests.conftest import make_mock_numeric_question

QID = 44874
MODEL = "openrouter/openai/gpt-5.6-sol"
PFX = "2026-09-02 14:23:01,123 - metaculus_bot.forecaster_runners - INFO - "

_SPEC = next(spec for spec in MARKER_SPECS if spec.name == "member_forecast")
_MC_OPTIONS = ["Option 0", "Option 1", "Option 2"]

# Thirteen declared pairs on the canonical set, with the float noise sanitize leaves behind.
_DECLARED = [Percentile(percentile=p, value=8.7 + 0.8729999999999905 * i) for i, p in enumerate(STANDARD_PERCENTILES)]
_SANITIZED = [Percentile(percentile=p.percentile, value=p.value + 0.001) for p in _DECLARED]


def _harvest(line: str) -> dict[str, Any]:
    """Run one line through the real harvester and return its single member_forecast record.

    Asserts that NO other spec claimed the line: the parser breaks on the first match, so a
    looser regex earlier in MARKER_SPECS would silently steal every record.
    """
    harvested = parse_log_text(
        PFX + line,
        run_id="900",
        workflow="run_bot_on_tournament",
        artifact="research-900",
        run_date="2026-09-02",
        log_file="x.log",
    )
    others = {name: recs for name, recs in harvested.items() if recs and name != "member_forecast"}
    assert not others, f"line was claimed by another spec: {others}"
    records = harvested["member_forecast"]
    assert len(records) == 1, records
    return records[0]


def _member_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("MEMBER_FORECAST:")]


def _pol(probabilities: list[float]) -> PredictedOptionList:
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=f"Option {i}", probability=p) for i, p in enumerate(probabilities)
        ]
    )


class TestFormatter:
    def test_binary_line_carries_both_probabilities_as_json_floats(self):
        line = format_member_forecast_marker(
            question_id=QID, model=MODEL, role=MEMBER_FORECAST_ROLE_MEMBER, qtype="binary", raw=0.005, published=0.02
        )
        assert (
            line == f"MEMBER_FORECAST: question={QID} model={MODEL} role=member qtype=binary raw=0.005 published=0.02"
        )

    def test_mc_vectors_are_compact_json_with_no_whitespace(self):
        line = format_member_forecast_marker(
            question_id=QID,
            model=MODEL,
            role=MEMBER_FORECAST_ROLE_MEMBER,
            qtype="multiple_choice",
            raw=[0.9, 0.005, 0.095],
            published=[0.891, 0.01, 0.099],
        )
        assert line.endswith("qtype=multiple_choice raw=[0.9,0.005,0.095] published=[0.891,0.01,0.099]")

    def test_numeric_pairs_keep_the_declared_decimal_percentile(self):
        line = format_member_forecast_marker(
            question_id=QID,
            model=MODEL,
            role=MEMBER_FORECAST_ROLE_MEMBER,
            qtype="numeric",
            raw=[[0.025, 9.2], [0.05, 9.6]],
            published=[[0.025, 9.2], [0.05, 9.6]],
        )
        # The block's decimal percentile, never a percent label (0.1 * 100 leaves 10.000000000000002).
        assert "raw=[[0.025,9.2],[0.05,9.6]]" in line

    def test_stacker_role_is_spelled_on_the_line(self):
        line = format_member_forecast_marker(
            question_id=QID,
            model="stacker-model",
            role=MEMBER_FORECAST_ROLE_STACKER,
            qtype="binary",
            raw=0.5,
            published=0.5,
        )
        assert " role=stacker " in line

    def test_a_thirteen_pair_numeric_line_stays_well_under_two_kilochars(self):
        line = format_member_forecast_marker(
            question_id=QID,
            model=MODEL,
            role=MEMBER_FORECAST_ROLE_MEMBER,
            qtype="numeric",
            raw=percentile_pairs(_DECLARED),
            published=percentile_pairs(_SANITIZED),
        )
        assert len(_DECLARED) == 13
        assert len(line) < 2000, len(line)

    def test_a_non_finite_value_raises_rather_than_writing_nan(self):
        """The ladder guarantees finite values, so NaN here is an upstream bug; ``NaN`` is
        not JSON and json.loads on the archive would fail months later instead."""
        with pytest.raises(ValueError, match="not JSON compliant"):
            format_member_forecast_marker(
                question_id=QID,
                model=MODEL,
                role=MEMBER_FORECAST_ROLE_MEMBER,
                qtype="binary",
                raw=math.nan,
                published=0.02,
            )

    def test_option_vector_and_percentile_pairs_read_the_ft_objects_in_order(self):
        assert option_vector(_pol([0.2, 0.5, 0.3])) == [0.2, 0.5, 0.3]
        assert percentile_pairs(_DECLARED[:2]) == [[0.01, 8.7], [0.025, 8.7 + 0.8729999999999905]]


class TestRoundTripThroughMarkerSpec:
    """Format -> harvest -> json.loads reproduces every value, for all three types."""

    @pytest.mark.parametrize(
        ("qtype", "raw", "published"),
        [
            ("binary", 0.005, BINARY_PROB_MIN),
            ("multiple_choice", [0.9, 0.005, 0.095], [0.891, 0.01, 0.099]),
            ("numeric", percentile_pairs(_DECLARED), percentile_pairs(_SANITIZED)),
        ],
    )
    def test_every_field_survives_harvesting(self, qtype, raw, published):
        line = format_member_forecast_marker(
            question_id=QID, model=MODEL, role=MEMBER_FORECAST_ROLE_MEMBER, qtype=qtype, raw=raw, published=published
        )
        rec = _harvest(line)

        assert rec["marker"] == "member_forecast"
        assert rec["qid"] == QID
        assert rec["qid_kind"] == "question_id"
        assert rec["model"] == MODEL
        assert rec["role"] == "member"
        assert rec["qtype"] == qtype
        assert json.loads(rec["raw"]) == raw
        assert json.loads(rec["published"]) == published

    def test_values_are_kept_verbatim_so_a_consumer_always_json_loads(self):
        """Without the spec's raw_fields the binary line would coerce to a float while the
        MC and numeric vectors stayed strings, and every consumer would need a branch."""
        rec = _harvest(
            format_member_forecast_marker(
                question_id=QID,
                model=MODEL,
                role=MEMBER_FORECAST_ROLE_MEMBER,
                qtype="binary",
                raw=0.005,
                published=0.02,
            )
        )
        assert rec["raw"] == "0.005"
        assert rec["published"] == "0.02"
        assert _SPEC.raw_fields == frozenset({"raw", "published"})

    def test_stacker_role_and_line_timestamp_harvest(self):
        rec = _harvest(
            format_member_forecast_marker(
                question_id=QID,
                model="anthropic/claude-opus-4.8",
                role=MEMBER_FORECAST_ROLE_STACKER,
                qtype="binary",
                raw=0.99,
                published=0.98,
            )
        )
        assert rec["role"] == "stacker"
        assert rec["model"] == "anthropic/claude-opus-4.8"
        assert rec["line_ts"] == "2026-09-02T14:23:01.123000"

    def test_thin_publish_floor_raw_field_still_coerces_to_a_float(self):
        """The per-spec raw_fields must not leak: the neighbouring marker spells a field
        ``raw`` too, and its consumers difference it as a number."""
        harvested = parse_log_text(
            "2026-09-02 14:30:00,456 - metaculus_bot.x - WARNING - THIN_PUBLISH_FLOOR: question=44874 raw=0.0300 clamped=0.0500 survivors=1",
            run_id="900",
            workflow="w",
            artifact="a",
            run_date="2026-09-02",
            log_file="x.log",
        )
        assert harvested["thin_publish_floor"][0]["raw"] == 0.03


class TestDeclaredVectorLeavesTheLadder:
    """``extract_mc`` records the probabilities as DECLARED, which the option list can never hold.

    ``PredictedOptionList`` is clamped on construction (our pre-construction clamp exists to
    make ft's validator a no-op), so the runner never sees the raw MC vector on the list
    itself; the first version of the MC marker read the list back and reported the
    published vector twice. The declared vector rides beside the list in the outcome's
    ``McForecast`` value instead, paired by type so no consumer has to check for it.
    """

    @pytest.mark.asyncio
    async def test_block_rung_carries_the_pre_clamp_probabilities(self):
        text = (
            "Option 0 nearly certain.\n\n```json\n"
            '{"question_type": "multiple_choice", "option_probs": {"Option 0": 0.9, "Option 1": 0.005, "Option 2": 0.095}}\n'
            "```"
        )
        outcome = await extract_mc(text, _MC_OPTIONS, cast("GeneralLlm", MagicMock(model="parser")))

        assert outcome.rung == "block"
        assert outcome.value.declared_probs == [0.9, 0.005, 0.095]
        published = option_vector(outcome.value.option_list)
        assert min(published) >= MC_PROB_MIN
        assert published != outcome.value.declared_probs

    @pytest.mark.asyncio
    async def test_llm_rung_tolerant_path_carries_the_parser_probabilities_in_option_order(self):
        # Strict PredictedOptionList parse fails, tolerant list[OptionProbability] parse
        # returns names in a shuffled, prefix-variant spelling that build_mc_prediction
        # canonicalises; the declared vector must land in question-option order too.
        parsed = [
            OptionProbability(option_name="option 2", probability=0.002),
            OptionProbability(option_name="Option 0", probability=0.995),
            OptionProbability(option_name="OPTION 1", probability=0.003),
        ]
        with patch(
            "metaculus_bot.value_extraction.parse_structured",
            new=AsyncMock(side_effect=[ValueError("no strict list"), parsed]),
        ):
            outcome = await extract_mc("no block here", _MC_OPTIONS, cast("GeneralLlm", MagicMock(model="parser")))

        assert outcome.rung == "llm"
        assert outcome.value.declared_probs == [0.995, 0.003, 0.002]
        assert [o.option_name for o in outcome.value.option_list.predicted_options] == _MC_OPTIONS
        assert min(option_vector(outcome.value.option_list)) >= MC_PROB_MIN

    @pytest.mark.asyncio
    async def test_llm_rung_strict_path_carries_the_parser_probabilities_after_the_ft_clamp(self):
        """The strict ``PredictedOptionList`` sub-path is tried FIRST on the LLM rung and is the
        one place ``raw`` is not genuinely pre-clamp: the parser decodes straight into an ft
        model, whose validator clamps into [0.01, 0.99] and renormalises on construction, so
        the declared vector is the parser's output AFTER that clamp. Pinned so a change in
        what the archive means on this sub-path is a failing test, not a silent shift."""
        strict = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="option 2", probability=0.002),
                PredictedOption(option_name="Option 0", probability=0.995),
                PredictedOption(option_name="OPTION 1", probability=0.003),
            ]
        )
        ft_clamped = {o.option_name: o.probability for o in strict.predicted_options}
        assert ft_clamped["Option 0"] != 0.995  # ft's construction clamp already moved the parser's values
        with patch("metaculus_bot.value_extraction.parse_structured", new=AsyncMock(return_value=strict)):
            outcome = await extract_mc("no block here", _MC_OPTIONS, cast("GeneralLlm", MagicMock(model="parser")))

        assert outcome.rung == "llm"
        assert [o.option_name for o in outcome.value.option_list.predicted_options] == _MC_OPTIONS
        # Question-option order, and the ft-clamped values rather than the 0.995/0.003/0.002 handed in.
        assert outcome.value.declared_probs == [ft_clamped["Option 0"], ft_clamped["OPTION 1"], ft_clamped["option 2"]]
        assert outcome.value.declared_probs != [0.995, 0.003, 0.002]
        assert min(option_vector(outcome.value.option_list)) >= MC_PROB_MIN


@pytest.fixture
def forecaster_llm() -> GeneralLlm:
    return GeneralLlm(model=MODEL)


@pytest.fixture
def parser_llm() -> GeneralLlm:
    return GeneralLlm(model="test-parser")


@pytest.fixture
def binary_question() -> MagicMock:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = QID
    q.page_url = f"https://metaculus.com/questions/{QID}"
    return q


@pytest.fixture
def mc_question() -> MagicMock:
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.id_of_question = 2001
    q.page_url = "https://metaculus.com/questions/2001"
    q.options = ["Option 0", "Option 1", "Option 2"]
    return q


@pytest.fixture
def numeric_question() -> MagicMock:
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = 3001
    q.page_url = "https://metaculus.com/questions/3001"
    q.lower_bound = 0
    q.upper_bound = 1000
    q.open_lower_bound = False
    q.open_upper_bound = True
    q.unit_of_measure = "widgets"
    return q


class TestMemberEmission:
    """The runners log the line with the RAW value, at the point where the clamp happens."""

    @pytest.mark.asyncio
    async def test_binary_line_carries_the_pre_clamp_probability(
        self, binary_question, forecaster_llm, parser_llm, caplog
    ):
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        with (
            patch("metaculus_bot.forecaster_runners.binary_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="Very unlikely")),
            patch(
                "metaculus_bot.forecaster_runners.extract_binary",
                new=AsyncMock(return_value=ExtractionOutcome(value=0.005, rung="block", block_present=True)),
            ),
        ):
            result = await run_binary_forecast(binary_question, "research", forecaster_llm, parser_llm)

        assert result.prediction_value == BINARY_PROB_MIN
        (line,) = _member_lines(caplog)
        rec = _harvest(line)
        assert rec["qid"] == QID
        assert rec["model"] == MODEL
        assert rec["role"] == "member"
        assert rec["qtype"] == "binary"
        assert json.loads(rec["raw"]) == 0.005
        assert json.loads(rec["published"]) == BINARY_PROB_MIN

    @pytest.mark.asyncio
    async def test_mc_raw_vector_is_read_before_the_in_place_clamp(
        self, mc_question, forecaster_llm, parser_llm, caplog
    ):
        """clamp_and_renormalize_mc rewrites option.probability on the SAME objects, so a
        vector read after it would report the published values twice."""
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        raw = [0.9, 0.005, 0.095]
        with (
            patch("metaculus_bot.forecaster_runners.multiple_choice_prompt", return_value="prompt"),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="Option 0 most likely.")),
            patch(
                "metaculus_bot.forecaster_runners.extract_mc",
                new=AsyncMock(
                    return_value=ExtractionOutcome(value=McForecast(_pol(raw), raw), rung="block", block_present=True)
                ),
            ),
        ):
            result = await run_mc_forecast(mc_question, "research", forecaster_llm, parser_llm)

        (line,) = _member_lines(caplog)
        rec = _harvest(line)
        assert rec["qtype"] == "multiple_choice"
        assert json.loads(rec["raw"]) == raw
        published = json.loads(rec["published"])
        assert published == option_vector(result.prediction_value)
        assert min(published) >= MC_PROB_MIN
        assert published != raw
        assert sum(published) == pytest.approx(1.0)

    @staticmethod
    def _numeric_patches(forecaster_llm: GeneralLlm, *, mismatch: tuple[bool, str]) -> list[Any]:
        return [
            patch("metaculus_bot.forecaster_runners.numeric_prompt", return_value="prompt"),
            patch("metaculus_bot.forecaster_runners.bound_messages", return_value=("upper msg", "lower msg")),
            patch.object(forecaster_llm, "invoke", new=AsyncMock(return_value="reasoning")),
            patch(
                "metaculus_bot.forecaster_runners.parse_structured",
                new=AsyncMock(return_value=OutcomeTypeResult(is_discrete_integer=False)),
            ),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(return_value=ExtractionOutcome(value=_DECLARED, rung="block", block_present=True)),
            ),
            patch("metaculus_bot.forecaster_runners.sanitize_percentiles", return_value=(_SANITIZED, None)),
            patch("metaculus_bot.forecaster_runners.build_numeric_distribution", return_value=MagicMock()),
            patch("metaculus_bot.forecaster_runners.detect_unit_mismatch", return_value=mismatch),
            patch("metaculus_bot.forecaster_runners.log_final_prediction"),
            patch("metaculus_bot.forecaster_runners.log_open_bound_piling_diagnostics"),
        ]

    @pytest.mark.asyncio
    async def test_numeric_line_pairs_declared_with_sanitized(
        self, numeric_question, forecaster_llm, parser_llm, caplog
    ):
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        with contextlib.ExitStack() as stack:
            for p in self._numeric_patches(forecaster_llm, mismatch=(False, "")):
                stack.enter_context(p)
            await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm)

        (line,) = _member_lines(caplog)
        rec = _harvest(line)
        assert rec["qid"] == 3001
        assert rec["qtype"] == "numeric"
        assert json.loads(rec["raw"]) == percentile_pairs(_DECLARED)
        assert json.loads(rec["published"]) == percentile_pairs(_SANITIZED)

    @pytest.mark.asyncio
    async def test_numeric_line_precedes_the_unit_mismatch_guard(
        self, numeric_question, forecaster_llm, parser_llm, caplog
    ):
        """A member the guard withholds still leaves its raw declaration: that is the
        expensive data, and the drop itself is FORECASTER_DROPS' record."""
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        with contextlib.ExitStack() as stack:
            for p in self._numeric_patches(forecaster_llm, mismatch=(True, "values look 1000x off")):
                stack.enter_context(p)
            with pytest.raises(UnitMismatchError):
                await run_numeric_forecast(numeric_question, "research", forecaster_llm, parser_llm)

        (line,) = _member_lines(caplog)
        assert json.loads(_harvest(line)["raw"]) == percentile_pairs(_DECLARED)


class _FakeStackerLLM:
    model = "anthropic/claude-opus-4.8"

    async def invoke(self, prompt: str) -> str:
        return "meta reasoning"


def _stacker_binary_question() -> MagicMock:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = 100
    q.page_url = "https://metaculus.com/questions/100"
    return q


class TestStackerEmission:
    """The stacker's output gets the same line with role=stacker, on all three types."""

    @pytest.mark.asyncio
    async def test_stacker_binary_line_carries_the_pre_clamp_value(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        monkeypatch.setattr("metaculus_bot.stacking.stacking_binary_prompt", lambda *a, **k: "prompt")
        monkeypatch.setattr(
            "metaculus_bot.stacking.extract_binary",
            AsyncMock(return_value=ExtractionOutcome(value=0.999, rung="block", block_present=True)),
        )

        value, _ = await stacking.run_stacking_binary(
            cast("GeneralLlm", _FakeStackerLLM()),
            cast("GeneralLlm", MagicMock(model="parser")),
            _stacker_binary_question(),
            research="R",
            base_texts=["m1", "m2"],
        )

        assert value == BINARY_PROB_MAX
        (line,) = _member_lines(caplog)
        rec = _harvest(line)
        assert rec["qid"] == 100
        assert rec["model"] == "anthropic/claude-opus-4.8"
        assert rec["role"] == "stacker"
        assert rec["qtype"] == "binary"
        assert json.loads(rec["raw"]) == 0.999
        assert json.loads(rec["published"]) == BINARY_PROB_MAX

    @pytest.mark.asyncio
    async def test_stacker_mc_line_reads_the_vector_before_the_clamp(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        raw = [0.995, 0.003, 0.002]
        question = MagicMock(spec=MultipleChoiceQuestion)
        question.id_of_question = 200
        question.options = ["Option 0", "Option 1", "Option 2"]
        monkeypatch.setattr("metaculus_bot.stacking.stacking_multiple_choice_prompt", lambda *a, **k: "prompt")
        monkeypatch.setattr(
            "metaculus_bot.stacking.extract_mc",
            AsyncMock(
                return_value=ExtractionOutcome(value=McForecast(_pol(raw), raw), rung="block", block_present=True)
            ),
        )

        pol, _ = await stacking.run_stacking_mc(
            cast("GeneralLlm", _FakeStackerLLM()),
            cast("GeneralLlm", MagicMock(model="parser")),
            question,
            research="R",
            base_texts=["m1", "m2"],
        )

        (line,) = _member_lines(caplog)
        rec = _harvest(line)
        assert rec["role"] == "stacker"
        assert rec["qtype"] == "multiple_choice"
        assert json.loads(rec["raw"]) == raw
        assert json.loads(rec["published"]) == option_vector(pol)
        assert min(json.loads(rec["published"])) >= MC_PROB_MIN

    @pytest.mark.asyncio
    async def test_stacker_numeric_line_is_emitted_where_its_percentiles_are_sanitised(self, caplog):
        """run_stacking_numeric returns the raw declaration by contract and the pipeline
        sanitises it, so the pipeline is the one place both halves of the line exist.
        ``sanitize_percentiles`` is patched to SHIFT the values so the two fields cannot be
        confused: the real sanitiser leaves a clean strictly-increasing set untouched, and a
        line logging the declaration twice would have passed against it."""
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        test_llm = GeneralLlm(model="test-model", temperature=0.0)
        pipeline = AggregationPipeline(
            strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_llm=test_llm,
            parser_llm=test_llm,
            stacking_fallback_on_failure=True,
            stacking_randomize_order=False,
            stacking_spread_thresholds={"binary": 0.15, "mc": 0.20, "numeric": 0.15},
            discrete_integer_votes=defaultdict(list),
        )
        question = make_mock_numeric_question(id_of_question=301, cdf_size=201)
        percentiles = [
            Percentile(percentile=p, value=5.0 + 90.0 * i / (len(STANDARD_PERCENTILES) - 1))
            for i, p in enumerate(STANDARD_PERCENTILES)
        ]
        reasoned = cast(
            "list[ReasonedPrediction[PredictionTypes]]",
            [
                ReasonedPrediction(prediction_value=0.5, reasoning="Model: m1\n\nLow."),
                ReasonedPrediction(prediction_value=0.5, reasoning="Model: m2\n\nHigh."),
            ],
        )

        sanitized = [Percentile(percentile=p.percentile, value=p.value + 0.25) for p in percentiles]

        with (
            patch(
                "metaculus_bot.aggregation_pipeline.stacking.run_stacking_numeric",
                new=AsyncMock(return_value=(percentiles, "meta")),
            ),
            patch("metaculus_bot.aggregation_pipeline.sanitize_percentiles", return_value=(sanitized, None)),
        ):
            await pipeline.run_stacking(question, "research", reasoned)

        (line,) = _member_lines(caplog)
        rec = _harvest(line)
        assert rec["qid"] == 301
        assert rec["model"] == "test-model"
        assert rec["role"] == "stacker"
        assert rec["qtype"] == "numeric"
        assert json.loads(rec["raw"]) == percentile_pairs(percentiles)
        assert json.loads(rec["published"]) == percentile_pairs(sanitized)
        assert percentile_pairs(sanitized) != percentile_pairs(percentiles)
