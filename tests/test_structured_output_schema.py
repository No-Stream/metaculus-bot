"""Tests for structured_output_schema module — the pydantic schemas for per-question-type
structured rationale blocks and their validators.

The block extraction and parsing helpers in the same module are covered by
``tests/test_structured_block_parsing.py``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import ClassVar, get_args, get_type_hints

import pytest
from pydantic import ValidationError

import metaculus_bot.structured_output_schema as schema
from metaculus_bot import structured_output_schema
from metaculus_bot.question_types import QuestionType
from metaculus_bot.structured_output_schema import (
    _NUMERIC_OUTCOME_TYPES,
    _QUESTION_TYPE_TO_MODEL,
    BaseRateAnchor,
    BinaryStructured,
    BlockType,
    CriteriaClause,
    DateStructured,
    DiscreteCountStructured,
    EvidenceItem,
    MultipleChoiceStructured,
    NumericOutcomeType,
    NumericStructured,
    PmfStructured,
    ScenarioBranch,
    StatedBaseRate,
    StatedHazard,
    StatedPrior,
    parse_structured_block,
    parse_structured_payload,
)

# Why: a plugin load registers the shared fixture chain without module-level names that shadow test parameters.
pytest_plugins = ["tests.structured_block_fixtures"]

# ===========================================================================
# Happy-path construction per question-type model
# ===========================================================================


class TestBinaryStructuredHappyPath:
    def test_full_construction(self, valid_binary_block: BinaryStructured) -> None:
        b = valid_binary_block
        assert b.question_type == "binary"
        assert b.posterior_prob == pytest.approx(0.35)
        assert isinstance(b.prior, StatedPrior)
        assert b.prior.prob == pytest.approx(0.25)
        assert isinstance(b.base_rate, StatedBaseRate)
        assert b.base_rate.k == 3
        assert b.base_rate.n == 20
        assert isinstance(b.hazard, StatedHazard)
        assert b.hazard.unit == "month"
        assert len(b.evidence) == 2
        assert b.evidence[0].direction == "up"
        assert b.evidence[1].likelihood_ratio == pytest.approx(0.7)
        assert len(b.scenarios) == 2
        assert sum(s.prob for s in b.scenarios) == pytest.approx(1.0)

    def test_only_required_fields(self) -> None:
        b = BinaryStructured(question_type="binary", posterior_prob=0.5)
        assert b.question_type == "binary"
        assert b.posterior_prob == 0.5
        assert b.prior is None
        assert b.base_rate is None
        assert b.hazard is None
        assert b.evidence == []
        assert b.scenarios == []
        # Telemetry fields are optional — old blocks without them parse fine.
        assert b.base_rate_anchor is None
        assert b.criteria_clauses == []

    def test_posterior_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            BinaryStructured(question_type="binary", posterior_prob=1.5)

    def test_extra_fields_forbidden(self) -> None:
        """extra="forbid" rejects an unknown field at runtime, so the static-typing complaint is expected."""
        with pytest.raises(ValidationError):
            BinaryStructured(question_type="binary", posterior_prob=0.5, unknown_field="oops")  # type: ignore[call-arg]


class TestBinaryTelemetryFields:
    """Optional anchor / clause telemetry fields (2026-07-08).

    Back-compat contract: blocks WITHOUT the fields must keep parsing (see
    ``TestBinaryStructuredHappyPath.test_only_required_fields``); blocks WITH
    them round-trip; malformed values are rejected by validation.
    """

    def test_anchor_and_clauses_round_trip(self) -> None:
        payload = json.dumps(
            {
                "question_type": "binary",
                "posterior_prob": 0.42,
                "base_rate_anchor": {"low": 0.15, "high": 0.35},
                "criteria_clauses": [
                    {"name": "formal instrument signed", "prob": 0.6},
                    {"name": "in-window", "prob": 0.8},
                ],
            }
        )
        block = parse_structured_block(f"```json\n{payload}\n```", "binary")
        assert isinstance(block, BinaryStructured)
        assert block.base_rate_anchor is not None
        assert block.base_rate_anchor.low == pytest.approx(0.15)
        assert block.base_rate_anchor.high == pytest.approx(0.35)
        assert [c.name for c in block.criteria_clauses] == ["formal instrument signed", "in-window"]
        assert [c.prob for c in block.criteria_clauses] == [pytest.approx(0.6), pytest.approx(0.8)]

    def test_old_block_without_telemetry_fields_still_parses(self) -> None:
        payload = json.dumps({"question_type": "binary", "posterior_prob": 0.28})
        block = parse_structured_block(f"```json\n{payload}\n```", "binary")
        assert isinstance(block, BinaryStructured)
        assert block.base_rate_anchor is None
        assert block.criteria_clauses == []

    def test_anchor_low_above_high_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BaseRateAnchor(low=0.6, high=0.4)

    def test_anchor_bounds_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BaseRateAnchor(low=-0.1, high=0.4)
        with pytest.raises(ValidationError):
            BaseRateAnchor(low=0.1, high=1.4)

    def test_clause_prob_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CriteriaClause(name="threshold met", prob=1.2)

    def test_clause_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CriteriaClause(name="", prob=0.5)

    def test_degenerate_point_anchor_allowed(self) -> None:
        anchor = BaseRateAnchor(low=0.3, high=0.3)
        assert anchor.low == anchor.high == pytest.approx(0.3)


class TestNumericStructuredHappyPath:
    def test_full_construction(self, valid_numeric_block: NumericStructured) -> None:
        n = valid_numeric_block
        assert n.question_type == "numeric"
        assert n.declared_percentiles is not None
        assert set(n.declared_percentiles.keys()) >= {0.1, 0.5, 0.9}
        assert n.declared_percentiles[0.5] == pytest.approx(50.0)

    def test_only_required_fields(self) -> None:
        n = NumericStructured(
            question_type="numeric",
            declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
        )
        assert n.prior is None
        assert n.scenarios == []

    def test_tails_field_removed(self) -> None:
        """The dead ``tails`` / ``TailMass`` slot is gone: the key is rejected and the name is unimportable."""
        with pytest.raises(ValidationError):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
                tails={"below_min_expected": 0.05, "above_max_expected": 0.05},  # type: ignore[call-arg]
            )
        assert not hasattr(schema, "TailMass")

    def test_outcome_type_discrete(self) -> None:
        """C3: outcome_type='discrete_integer' accepted."""
        n = NumericStructured(
            question_type="numeric",
            declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
            outcome_type="discrete_integer",
        )
        assert n.outcome_type == "discrete_integer"

    def test_outcome_type_continuous(self) -> None:
        """C3: outcome_type='continuous' accepted."""
        n = NumericStructured(
            question_type="numeric",
            declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
            outcome_type="continuous",
        )
        assert n.outcome_type == "continuous"

    def test_outcome_type_none_default(self) -> None:
        """C3: outcome_type defaults to None (backward compat)."""
        n = NumericStructured(
            question_type="numeric",
            declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
        )
        assert n.outcome_type is None

    @pytest.mark.parametrize("declared", ["unknown", "integer", "discrete", "count", 3, ["continuous"]])
    def test_an_unrecognised_outcome_type_reads_as_absent(
        self, declared: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A misspelling must cost ONE parser call, not the whole numeric block.

        Under the bare Literal, "integer" or "count" failed validation, and since the
        numeric block has no strip-and-retry the PERCENTILES went down with it: the
        forecast dropped to the LLM salvage rung and the OutcomeTypeResult parser call
        fired anyway for the type. Reading a stray as None leaves
        ``_resolve_discrete_vote`` on exactly the parser fallback it already has.
        """
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            n = NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
                outcome_type=declared,  # type: ignore[arg-type]
            )
        assert n.outcome_type is None
        assert n.declared_percentiles == {0.1: 1.0, 0.5: 5.0, 0.9: 9.0}
        # The raw value rides the WARNING: a spelling the roster starts using is a prompt signal.
        assert any(repr(declared) in rec.getMessage() for rec in caplog.records)

    @pytest.mark.parametrize(
        ("log_failures", "expected_level"),
        [(True, logging.WARNING), (False, logging.DEBUG)],
    )
    def test_the_stray_spelling_line_respects_the_log_failures_contract(
        self, caplog: pytest.LogCaptureFixture, log_failures: bool, expected_level: int
    ) -> None:
        """``parse_structured_payload(..., log_failures=False)`` promises no scary WARNING about
        a candidate it is about to discard, and this validator logs from inside the model where
        it cannot see the flag. The flag rides down as validation context. Without it a
        misspelling warned on every silently-probed candidate, on superseded draft blocks, and
        twice per numeric forecast on the publish path, about a block that publishes fine."""
        payload = (
            '{"question_type": "numeric", "outcome_type": "integer",'
            ' "declared_percentiles": {"0.1": 10.0, "0.5": 50.0, "0.9": 90.0}}'
        )
        with caplog.at_level(logging.DEBUG, logger="metaculus_bot.structured_output_schema"):
            parsed = parse_structured_payload(payload, "numeric", log_failures=log_failures)

        assert isinstance(parsed, NumericStructured)
        assert parsed.outcome_type is None
        stray_lines = [r for r in caplog.records if "Unrecognised outcome_type" in r.getMessage()]
        assert len(stray_lines) == 1
        assert stray_lines[0].levelno == expected_level
        # The raw value rides the line at either level: that is what makes it a prompt signal.
        assert "'integer'" in stray_lines[0].getMessage()

    def test_the_accepted_outcome_types_are_derived_from_the_literal(self) -> None:
        """One vocabulary, one definition. A restated copy fails asymmetrically: a third outcome
        type would pass the annotation and then be silently nulled by the lenient validator."""
        assert set(get_args(NumericOutcomeType)) == set(_NUMERIC_OUTCOME_TYPES)
        annotation = NumericStructured.model_fields["outcome_type"].annotation
        assert NumericOutcomeType in get_args(annotation)

    def test_accepts_extra_percentiles(self) -> None:
        n = NumericStructured(
            question_type="numeric",
            declared_percentiles={0.05: 0.5, 0.1: 1.0, 0.25: 2.5, 0.5: 5.0, 0.9: 9.0, 0.95: 9.5},
        )
        assert n.declared_percentiles is not None
        assert len(n.declared_percentiles) == 6


class TestMultipleChoiceStructuredHappyPath:
    def test_full_construction(self, valid_mc_block: MultipleChoiceStructured) -> None:
        m = valid_mc_block
        assert m.question_type == "multiple_choice"
        assert m.option_probs == {"A": 0.5, "B": 0.3, "C": 0.2}
        assert m.other_mass == pytest.approx(0.0)
        assert m.concentration == pytest.approx(5.0)

    def test_only_required_fields(self) -> None:
        m = MultipleChoiceStructured(
            question_type="multiple_choice",
            option_probs={"A": 0.5, "B": 0.5},
        )
        assert m.prior is None
        assert m.other_mass is None
        assert m.concentration is None

    def test_sum_within_tolerance(self) -> None:
        """A 0.99 option_probs sum is within the 0.02 tolerance of 1.0."""
        m = MultipleChoiceStructured(
            question_type="multiple_choice",
            option_probs={"A": 0.33, "B": 0.33, "C": 0.33},
        )
        assert sum(m.option_probs.values()) == pytest.approx(0.99)


class TestDiscreteCountStructuredHappyPath:
    def test_full_construction(self, valid_discrete_block: DiscreteCountStructured) -> None:
        d = valid_discrete_block
        assert d.question_type == "discrete_count"
        assert d.mean_estimate == pytest.approx(4.2)
        assert d.dispersion == "negbinom"
        assert d.overdispersion_factor == pytest.approx(1.5)
        assert d.declared_percentiles is not None
        assert d.declared_percentiles[0.5] == pytest.approx(4.0)

    def test_only_required_fields(self) -> None:
        d = DiscreteCountStructured(
            question_type="discrete_count",
            mean_estimate=1.0,
            dispersion="poisson",
        )
        assert d.prior is None
        assert d.ceiling is None
        assert d.overdispersion_factor is None
        assert d.declared_percentiles is None

    def test_beta_binom_with_ceiling(self) -> None:
        d = DiscreteCountStructured(
            question_type="discrete_count",
            mean_estimate=3.0,
            dispersion="beta_binom_ceiling",
            ceiling=10,
        )
        assert d.ceiling == 10


# ===========================================================================
# Validator tests: shared submodels
# ===========================================================================


class TestStatedBaseRateValidators:
    def test_k_greater_than_n_raises(self) -> None:
        with pytest.raises(ValidationError, match="k <= n"):
            StatedBaseRate(k=5, n=3, ref_class="x")

    def test_n_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="n must be >= 1"):
            StatedBaseRate(k=0, n=0, ref_class="x")

    def test_negative_k_raises(self) -> None:
        with pytest.raises(ValidationError, match="k must be >= 0"):
            StatedBaseRate(k=-1, n=5, ref_class="x")

    def test_k_equals_n_ok(self) -> None:
        br = StatedBaseRate(k=5, n=5, ref_class="x")
        assert br.k == 5
        assert br.n == 5

    def test_empty_ref_class_raises(self) -> None:
        with pytest.raises(ValidationError):
            StatedBaseRate(k=1, n=2, ref_class="")


class TestStatedHazardValidators:
    def test_fractions_sum_to_1_ok(self) -> None:
        h = StatedHazard(
            rate_per_unit=0.1,
            unit="year",
            window_duration_units=5.0,
            elapsed_fraction=0.4,
            remaining_fraction=0.6,
        )
        assert h.elapsed_fraction + h.remaining_fraction == pytest.approx(1.0)

    def test_fractions_sum_way_off_raises(self) -> None:
        with pytest.raises(ValidationError, match="elapsed_fraction \\+ remaining_fraction"):
            StatedHazard(
                rate_per_unit=0.1,
                unit="day",
                window_duration_units=30.0,
                elapsed_fraction=0.5,
                remaining_fraction=0.1,
            )

    def test_fractions_sum_within_tolerance_ok(self) -> None:
        """0.49 + 0.505 = 0.995, within the 0.01 tolerance."""
        h = StatedHazard(
            rate_per_unit=0.1,
            unit="day",
            window_duration_units=30.0,
            elapsed_fraction=0.49,
            remaining_fraction=0.505,
        )
        assert h.elapsed_fraction == pytest.approx(0.49)

    def test_invalid_unit_raises(self) -> None:
        with pytest.raises(ValidationError):
            StatedHazard(
                rate_per_unit=0.1,
                unit="decade",  # type: ignore[arg-type]
                window_duration_units=10.0,
                elapsed_fraction=0.5,
                remaining_fraction=0.5,
            )

    def test_negative_rate_raises(self) -> None:
        with pytest.raises(ValidationError):
            StatedHazard(
                rate_per_unit=-0.1,
                unit="day",
                window_duration_units=30.0,
                elapsed_fraction=0.5,
                remaining_fraction=0.5,
            )

    def test_missing_window_duration_units_raises(self) -> None:
        with pytest.raises(ValidationError, match="window_duration_units"):
            StatedHazard(
                rate_per_unit=0.1,  # type: ignore[call-arg]
                unit="day",
                elapsed_fraction=0.5,
                remaining_fraction=0.5,
            )

    def test_zero_window_duration_units_raises(self) -> None:
        with pytest.raises(ValidationError):
            StatedHazard(
                rate_per_unit=0.1,
                unit="day",
                window_duration_units=0.0,
                elapsed_fraction=0.5,
                remaining_fraction=0.5,
            )


class TestEvidenceItemValidators:
    def test_likelihood_ratio_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="likelihood_ratio must be > 0"):
            EvidenceItem(summary="x", direction="up", strength="weak", likelihood_ratio=0.0)

    def test_likelihood_ratio_negative_raises(self) -> None:
        with pytest.raises(ValidationError, match="likelihood_ratio must be > 0"):
            EvidenceItem(summary="x", direction="up", strength="weak", likelihood_ratio=-1.0)

    def test_likelihood_ratio_none_ok(self) -> None:
        e = EvidenceItem(summary="x", direction="up", strength="weak")
        assert e.likelihood_ratio is None

    def test_likelihood_ratio_positive_ok(self) -> None:
        e = EvidenceItem(summary="x", direction="up", strength="strong", likelihood_ratio=2.5)
        assert e.likelihood_ratio == pytest.approx(2.5)

    def test_invalid_direction_raises(self) -> None:
        """A direction outside the Literal is rejected at runtime, so the type-ignore is expected."""
        with pytest.raises(ValidationError):
            EvidenceItem(summary="x", direction="sideways", strength="weak")  # type: ignore[arg-type]

    def test_empty_summary_raises(self) -> None:
        with pytest.raises(ValidationError):
            EvidenceItem(summary="", direction="up", strength="weak")


class TestScenarioBranchValidators:
    def test_prob_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioBranch(name="x", prob=1.5)

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            ScenarioBranch(name="", prob=0.5)


# ===========================================================================
# Scenario-sum validator on Binary and Numeric models
# ===========================================================================


class TestScenarioSumValidator:
    def test_binary_scenarios_sum_half_raises(self) -> None:
        scenarios = [ScenarioBranch(name="a", prob=0.25), ScenarioBranch(name="b", prob=0.25)]
        with pytest.raises(ValidationError, match="scenarios must have probs summing to"):
            BinaryStructured(question_type="binary", posterior_prob=0.5, scenarios=scenarios)

    def test_numeric_scenarios_sum_half_raises(self) -> None:
        scenarios = [ScenarioBranch(name="a", prob=0.25), ScenarioBranch(name="b", prob=0.25)]
        with pytest.raises(ValidationError, match="scenarios must have probs summing to"):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
                scenarios=scenarios,
            )

    def test_binary_empty_scenarios_ok(self) -> None:
        b = BinaryStructured(question_type="binary", posterior_prob=0.5, scenarios=[])
        assert b.scenarios == []

    def test_numeric_empty_scenarios_ok(self) -> None:
        n = NumericStructured(
            question_type="numeric",
            declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
            scenarios=[],
        )
        assert n.scenarios == []

    def test_binary_scenarios_within_tolerance_ok(self) -> None:
        """0.49 + 0.52 = 1.01, within the 0.02 tolerance."""
        scenarios = [ScenarioBranch(name="a", prob=0.49), ScenarioBranch(name="b", prob=0.52)]
        b = BinaryStructured(question_type="binary", posterior_prob=0.5, scenarios=scenarios)
        assert sum(s.prob for s in b.scenarios) == pytest.approx(1.01)

    def test_numeric_scenarios_within_tolerance_ok(self) -> None:
        """0.5 + 0.485 = 0.985, within the 0.02 tolerance of 1.0."""
        scenarios = [ScenarioBranch(name="a", prob=0.5), ScenarioBranch(name="b", prob=0.485)]
        n = NumericStructured(
            question_type="numeric",
            declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
            scenarios=scenarios,
        )
        assert sum(s.prob for s in n.scenarios) == pytest.approx(0.985)


# ===========================================================================
# NumericStructured.declared_percentiles validator
# ===========================================================================


class TestNumericDeclaredPercentiles:
    def test_missing_p10_raises(self) -> None:
        with pytest.raises(ValidationError, match="declared_percentiles must include"):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.5: 5.0, 0.9: 9.0},
            )

    def test_missing_p50_raises(self) -> None:
        with pytest.raises(ValidationError, match="declared_percentiles must include"):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 1.0, 0.9: 9.0},
            )

    def test_missing_p90_raises(self) -> None:
        with pytest.raises(ValidationError, match="declared_percentiles must include"):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 1.0, 0.5: 5.0},
            )

    def test_a_decrease_with_rising_percentile_still_raises(self) -> None:
        """A value-disordered percentile set is incoherent by construction and unsalvageable.

        ``sort_by_percentile_level`` orders by LABEL, so such a set is force-monotonized rather
        than reordered, and one stray value pins most of the curve at a bound.
        """
        with pytest.raises(ValidationError, match="non-decreasing"):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 10.0, 0.5: 5.0, 0.9: 15.0},
            )

    def test_tied_values_parse(self) -> None:
        """Ties are a legitimate concentrated declaration, and the schema now says so.

        ``value_extraction._validate_numeric`` allows them by name and
        ``sanitize_percentiles``'s cluster spreader exists to separate them, so while this
        schema demanded a STRICT increase a count-like block (p10 = p50 on a quantity that
        usually reads the same low number) failed rung 1, could not be repaired — it is
        valid JSON — and reached the pipeline only through the LLM salvage rung.
        """
        block = NumericStructured(
            question_type="numeric",
            declared_percentiles={0.1: 5.0, 0.5: 5.0, 0.9: 10.0},
        )
        assert block.declared_percentiles == {0.1: 5.0, 0.5: 5.0, 0.9: 10.0}

    def test_percentile_key_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError, match="Percentile keys"):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0, 1.5: 15.0},
            )


# ===========================================================================
# DateStructured
# ===========================================================================


class TestDateStructured:
    """The date block: ISO-8601 strings parsed by ``numeric.date_axis.parse_forecast_date`` and nothing else."""

    @staticmethod
    def _block(declared: dict[str, object], **extra: object) -> DateStructured:
        """Validate a raw block the way the ladder does: string keys and string values, as JSON carries them."""
        return DateStructured.model_validate({"question_type": "date", "declared_percentiles": declared, **extra})

    def test_date_only_values_parse_to_noon_utc_and_keep_the_numeric_key_checks(self) -> None:
        block = self._block({"0.1": "2026-09-10", "0.5": "2026-09-15", "0.9": "2026-09-18T18:00:00Z"})
        assert block.declared_percentiles[0.1] == datetime(2026, 9, 10, 12, tzinfo=UTC)
        assert block.declared_percentiles[0.9] == datetime(2026, 9, 18, 18, tzinfo=UTC)

    def test_the_example_block_the_prompt_teaches_parses_through_the_registry(self) -> None:
        raw = json.dumps(
            {
                "question_type": "date",
                "declared_percentiles": {
                    "0.01": "2026-09-08",
                    "0.025": "2026-09-08",
                    "0.05": "2026-09-09",
                    "0.1": "2026-09-10",
                    "0.2": "2026-09-11",
                    "0.4": "2026-09-14",
                    "0.5": "2026-09-15",
                    "0.6": "2026-09-16",
                    "0.8": "2026-09-17",
                    "0.9": "2026-09-18",
                    "0.95": "2026-09-18",
                    "0.975": "2026-09-19",
                    "0.99": "2026-09-22",
                },
            }
        )
        block = parse_structured_payload(raw, "date")
        assert isinstance(block, DateStructured)
        assert parse_structured_block(f"reasoning\n```json\n{raw}\n```", "date") is not None

    @pytest.mark.parametrize("bad", ["2027", "2027-06", "2027-06-1", "June 1 2027", "2027/06/01", ""])
    def test_non_iso_and_truncated_spellings_are_rejected(self, bad: str) -> None:
        """This is the truncation guard for dates: the repair rung's numeric-literal check cannot
        see a string, so ``json_repair`` hands back ``"2027-06-1"`` from a cut-off rationale and only
        the strict parse here stops it publishing."""
        with pytest.raises(ValidationError, match="not a strict ISO-8601"):
            self._block({"0.1": "2026-09-10", "0.5": bad, "0.9": "2027-09-10"})

    @pytest.mark.parametrize("bad", [2027, 1788825600.0, None, True])
    def test_non_string_values_are_rejected_rather_than_read_as_unix_timestamps(self, bad: object) -> None:
        with pytest.raises(ValidationError, match="must be an ISO-8601 date string"):
            self._block({"0.1": "2026-09-10", "0.5": bad, "0.9": "2027-09-10"})

    def test_a_decrease_with_rising_percentile_raises_and_ties_parse(self) -> None:
        with pytest.raises(ValidationError, match="non-decreasing"):
            self._block({"0.1": "2026-09-15", "0.5": "2026-09-10", "0.9": "2026-09-18"})
        tied = self._block({"0.1": "2026-09-15", "0.5": "2026-09-15", "0.9": "2026-09-18"})
        assert tied.declared_percentiles[0.1] == tied.declared_percentiles[0.5]

    def test_missing_required_key_and_out_of_range_key_raise(self) -> None:
        with pytest.raises(ValidationError, match="declared_percentiles must include"):
            self._block({"0.5": "2026-09-15", "0.9": "2026-09-18"})
        with pytest.raises(ValidationError, match="Percentile keys"):
            self._block({"0.1": "2026-09-10", "0.5": "2026-09-15", "0.9": "2026-09-18", "1.5": "2026-09-19"})

    def test_extra_fields_and_outcome_type_are_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            self._block({"0.1": "2026-09-10", "0.5": "2026-09-15", "0.9": "2026-09-18"}, outcome_type="continuous")

    def test_a_numeric_block_cannot_masquerade_as_a_date_block(self) -> None:
        raw = '{"question_type": "numeric", "declared_percentiles": {"0.1": 1.0, "0.5": 2.0, "0.9": 3.0}}'
        assert parse_structured_payload(raw, "date") is None


# ===========================================================================
# MultipleChoiceStructured.option_probs validator
# ===========================================================================


class TestMultipleChoiceOptionProbs:
    def test_sum_way_off_raises(self) -> None:
        with pytest.raises(ValidationError, match="must sum to"):
            MultipleChoiceStructured(
                question_type="multiple_choice",
                option_probs={"A": 0.3, "B": 0.3},
            )

    def test_empty_option_probs_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be non-empty"):
            MultipleChoiceStructured(
                question_type="multiple_choice",
                option_probs={},
            )

    def test_empty_option_name_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-empty strings"):
            MultipleChoiceStructured(
                question_type="multiple_choice",
                option_probs={"": 0.5, "B": 0.5},
            )

    def test_whitespace_option_name_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-empty strings"):
            MultipleChoiceStructured(
                question_type="multiple_choice",
                option_probs={"   ": 0.5, "B": 0.5},
            )

    def test_option_prob_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError, match="values must be in"):
            MultipleChoiceStructured(
                question_type="multiple_choice",
                option_probs={"A": 1.5, "B": -0.5},
            )

    @pytest.mark.parametrize(
        ("option_probs", "message"),
        [
            ({}, "MultipleChoiceStructured.option_probs must be non-empty"),
            ({" ": 0.5, "B": 0.5}, "MultipleChoiceStructured.option_probs keys must be non-empty strings, got ' '"),
            ({"A": 1.5, "B": -0.5}, "MultipleChoiceStructured.option_probs values must be in [0, 1], got 1.5"),
            ({"A": 0.3, "B": 0.3}, "MultipleChoiceStructured.option_probs must sum to ~1.0 (tol 0.02), got 0.6"),
        ],
    )
    def test_the_ballot_error_messages_are_byte_identical_under_the_shared_validator(
        self, option_probs: dict[str, float], message: str
    ) -> None:
        """The ballot and the per-bin block share one probability-vector validator; the ballot's
        messages are archive-visible failure text and must not drift."""
        with pytest.raises(ValidationError) as excinfo:
            MultipleChoiceStructured(question_type="multiple_choice", option_probs=option_probs)
        assert message in str(excinfo.value)

    def test_a_boolean_probability_is_refused_rather_than_read_as_certainty(self) -> None:
        """pydantic's lax float reads ``true`` as 1.0 and ``false`` as 0.0; neither is a declared probability."""
        with pytest.raises(ValidationError, match="must be a number"):
            MultipleChoiceStructured.model_validate(
                {"question_type": "multiple_choice", "option_probs": {"A": True, "B": False}}
            )
        payload = '{"question_type": "multiple_choice", "option_probs": {"A": true, "B": false}}'
        assert parse_structured_payload(payload, "multiple_choice", log_failures=False) is None

    def test_an_integer_probability_still_reads_as_a_float(self) -> None:
        """Refusing booleans must not refuse the integer literals ``0`` and ``1`` a forecaster writes."""
        block = MultipleChoiceStructured.model_validate(
            {"question_type": "multiple_choice", "option_probs": {"A": 1, "B": 0}}
        )
        assert block.option_probs == {"A": 1.0, "B": 0.0}

    def test_concentration_zero_reads_as_absent(self) -> None:
        """q45189: gemini wrote ``"concentration": 0.0`` beside a valid ballot and the old
        ``> 0`` check took the whole block down with it. A non-positive concentration is
        not a reading, but the field is dormant and unprompted, so it now reads as absent
        rather than costing the forecast."""
        block = MultipleChoiceStructured(
            question_type="multiple_choice",
            option_probs={"A": 0.5, "B": 0.5},
            concentration=0.0,
        )
        assert block.concentration is None
        assert block.option_probs == {"A": 0.5, "B": 0.5}

    @pytest.mark.parametrize("declared", [0.0, -3.0, float("inf"), float("nan"), "twenty", True, {"alpha": 20.0}])
    def test_unusable_concentration_never_costs_the_ballot(self, declared: object) -> None:
        block = MultipleChoiceStructured(
            question_type="multiple_choice",
            option_probs={"A": 0.5, "B": 0.5},
            concentration=declared,  # type: ignore[arg-type]  # the point is a value the schema must tolerate
        )
        assert block.concentration is None

    def test_a_usable_concentration_still_round_trips(self) -> None:
        block = MultipleChoiceStructured(
            question_type="multiple_choice",
            option_probs={"A": 0.5, "B": 0.5},
            concentration=20.0,
        )
        assert block.concentration == pytest.approx(20.0)

    @pytest.mark.parametrize("declared", [-0.1, 1.5, float("nan"), "a third", True])
    def test_out_of_range_other_mass_reads_as_absent(self, declared: object) -> None:
        """Same rule as concentration: the field is a retired Dirichlet input, so an
        out-of-range declaration is worth nothing and must cost nothing. MC has no
        telemetry strip-and-retry, so a raise here means the LLM salvage rung."""
        block = MultipleChoiceStructured(
            question_type="multiple_choice",
            option_probs={"A": 0.5, "B": 0.5},
            other_mass=declared,  # type: ignore[arg-type]  # the point is a value the schema must tolerate
        )
        assert block.other_mass is None

    @pytest.mark.parametrize("field", ["concentration", "other_mass"])
    def test_an_integer_literal_too_large_for_a_float_reads_as_absent(self, field: str) -> None:
        """The lenient contract is "unusable reads as absent", and it has to hold for EVERY
        unusable input, including one that breaks the conversion itself.

        ``json.loads`` decodes an integer literal of any length into an arbitrary-precision int,
        and ``float()`` past about 308 digits raises OverflowError. Pydantic converts only
        ValueError and AssertionError into a ValidationError, so before the guard this escaped
        ``model_validate`` and ``parse_structured_payload`` (which catches ValidationError only)
        and crashed the caller, which is strictly worse than the strict code it replaced: that
        produced a clean rejection the ladder could fall through on.
        """
        payload = {
            "question_type": "multiple_choice",
            "option_probs": {"A": 0.5, "B": 0.5},
            field: int("9" * 400),
        }
        block = MultipleChoiceStructured.model_validate(payload)
        assert getattr(block, field) is None
        assert block.option_probs == {"A": 0.5, "B": 0.5}

    def test_a_usable_other_mass_still_round_trips(self) -> None:
        block = MultipleChoiceStructured(
            question_type="multiple_choice",
            option_probs={"A": 0.5, "B": 0.5},
            other_mass=0.0,
        )
        assert block.other_mass == pytest.approx(0.0)


# ===========================================================================
# DiscreteCountStructured validators
# ===========================================================================


class TestDiscreteCountValidators:
    def test_beta_binom_without_ceiling_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"beta_binom_ceiling.*requires ceiling"):
            DiscreteCountStructured(
                question_type="discrete_count",
                mean_estimate=3.0,
                dispersion="beta_binom_ceiling",
                ceiling=None,
            )

    def test_beta_binom_with_ceiling_ok(self) -> None:
        d = DiscreteCountStructured(
            question_type="discrete_count",
            mean_estimate=3.0,
            dispersion="beta_binom_ceiling",
            ceiling=7,
        )
        assert d.ceiling == 7

    def test_poisson_without_ceiling_ok(self) -> None:
        d = DiscreteCountStructured(
            question_type="discrete_count",
            mean_estimate=3.0,
            dispersion="poisson",
        )
        assert d.ceiling is None

    def test_ceiling_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="ceiling must be >= 1"):
            DiscreteCountStructured(
                question_type="discrete_count",
                mean_estimate=3.0,
                dispersion="negbinom",
                ceiling=0,
            )

    def test_overdispersion_at_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="overdispersion_factor must be > 1"):
            DiscreteCountStructured(
                question_type="discrete_count",
                mean_estimate=3.0,
                dispersion="negbinom",
                overdispersion_factor=1.0,
            )

    def test_negative_mean_raises(self) -> None:
        with pytest.raises(ValidationError):
            DiscreteCountStructured(
                question_type="discrete_count",
                mean_estimate=-1.0,
                dispersion="poisson",
            )


# ===========================================================================
# Block-type vocabulary and the per-bin PmfStructured block
# ===========================================================================


class TestQuestionTypeVocabulary:
    def test_the_parsers_accept_exactly_the_shared_question_types_plus_pmf(self) -> None:
        """One vocabulary, one definition: the ``question_type`` a block may declare is ``BlockType``,
        a WIDENING of ``question_types.QuestionType`` by the per-bin block and never a restated copy
        (a restated copy fails asymmetrically: adding ``date`` cost four synchronized edits). ``pmf``
        is an elicitation of a numeric or date question, not a question type, so ``QuestionType``
        itself stays as it is and every token of the widened vocabulary has a model."""
        assert set(_QUESTION_TYPE_TO_MODEL) == set(get_args(QuestionType)) | {"pmf"}
        assert "pmf" not in get_args(QuestionType)
        for parser in (parse_structured_block, parse_structured_payload):
            assert get_type_hints(parser)["question_type"] is BlockType
        assert not hasattr(structured_output_schema, "StructuredQuestionType"), "the restated copy is back"


class TestPmfStructured:
    """The per-bin block: one probability per bin label plus the reserved out-of-range keys."""

    EXAMPLE: ClassVar[dict[str, object]] = {
        "question_type": "pmf",
        "bin_probs": {"below_range": 0.05, "0": 0.6, "1": 0.25, "2": 0.05, "above_range": 0.05},
    }

    def test_accepts_the_example_block(self) -> None:
        block = PmfStructured.model_validate(self.EXAMPLE)
        assert block.question_type == "pmf"
        assert block.bin_probs == self.EXAMPLE["bin_probs"]

    def test_parses_through_the_shared_payload_parser(self) -> None:
        parsed = parse_structured_payload(json.dumps(self.EXAMPLE), "pmf")
        assert isinstance(parsed, PmfStructured)
        assert parsed.bin_probs["above_range"] == 0.05

    def test_a_missing_question_type_is_injected_by_the_parser(self) -> None:
        payload = {"bin_probs": self.EXAMPLE["bin_probs"]}
        assert isinstance(parse_structured_payload(json.dumps(payload), "pmf"), PmfStructured)

    def test_the_sum_tolerance_is_the_ballot_floor_or_the_per_key_drift(self) -> None:
        """0.005 a key (two-decimal rounding), never below the multiple-choice ballot's 0.02."""
        assert structured_output_schema.pmf_prob_sum_tolerance(3) == 0.02
        assert structured_output_schema.pmf_prob_sum_tolerance(4) == 0.02
        assert structured_output_schema.pmf_prob_sum_tolerance(14) == pytest.approx(0.07)
        assert structured_output_schema.pmf_prob_sum_tolerance(33) == pytest.approx(0.165)

    def test_a_fourteen_key_block_off_by_0_06_is_accepted_and_off_by_0_08_is_rejected(self) -> None:
        """Fourteen keys earn 0.07 of slack: thirteen at 0.07 plus 0.15 sums to 1.06 and passes, plus 0.17 sums
        to 1.08 and fails."""
        thirteen = {str(k): 0.07 for k in range(13)}
        block = PmfStructured(question_type="pmf", bin_probs={**thirteen, "13": 0.15})
        assert sum(block.bin_probs.values()) == pytest.approx(1.06)
        with pytest.raises(ValidationError, match="must sum to"):
            PmfStructured(question_type="pmf", bin_probs={**thirteen, "13": 0.17})

    def test_a_three_key_block_off_by_0_03_is_rejected_by_the_floor(self) -> None:
        with pytest.raises(ValidationError, match="must sum to"):
            PmfStructured(question_type="pmf", bin_probs={"0": 0.5, "1": 0.3, "2": 0.23})
        block = PmfStructured(question_type="pmf", bin_probs={"0": 0.5, "1": 0.3, "2": 0.21})
        assert sum(block.bin_probs.values()) == pytest.approx(1.01)

    def test_an_unknown_top_level_key_raises(self) -> None:
        with pytest.raises(ValidationError, match="extra_forbidden"):
            PmfStructured.model_validate({**self.EXAMPLE, "prior": {"prob": 0.5, "source": "x"}})

    def test_empty_bin_probs_raises(self) -> None:
        with pytest.raises(ValidationError, match="must be non-empty"):
            PmfStructured(question_type="pmf", bin_probs={})

    def test_a_blank_key_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-empty strings"):
            PmfStructured(question_type="pmf", bin_probs={"  ": 0.5, "1": 0.5})

    def test_a_probability_outside_the_unit_interval_raises(self) -> None:
        with pytest.raises(ValidationError, match="values must be in"):
            PmfStructured(question_type="pmf", bin_probs={"0": 1.5, "1": -0.5})

    def test_a_non_finite_probability_raises(self) -> None:
        """``json.loads`` admits ``NaN`` and ``Infinity`` literals; neither is a probability."""
        for literal in ("NaN", "Infinity"):
            payload = f'{{"question_type": "pmf", "bin_probs": {{"0": {literal}, "1": 0.5}}}}'
            assert parse_structured_payload(payload, "pmf", log_failures=False) is None

    def test_a_boolean_probability_is_refused_rather_than_read_as_certainty(self) -> None:
        """pydantic's lax float reads ``true`` as 1.0 and ``false`` as 0.0, which would publish a bin the
        forecaster never priced as certain."""
        payload = '{"question_type": "pmf", "bin_probs": {"0": true, "1": false}}'
        assert parse_structured_payload(payload, "pmf", log_failures=False) is None
        with pytest.raises(ValidationError, match="must be a number"):
            PmfStructured.model_validate({"question_type": "pmf", "bin_probs": {"0": True, "1": False}})

    def test_integer_literals_still_read_as_probabilities(self) -> None:
        block = PmfStructured.model_validate({"question_type": "pmf", "bin_probs": {"0": 1, "1": 0}})
        assert block.bin_probs == {"0": 1.0, "1": 0.0}

    def test_a_repeated_bin_key_is_refused_rather_than_read_last_wins(self) -> None:
        """``json.loads`` keeps the last of two equal keys, so ``{"0": 0.9, "0": 0.4, "1": 0.6}`` would read
        as a valid vector nobody declared; the decoder refuses the repeated key and the ladder falls through."""
        payload = '{"question_type": "pmf", "bin_probs": {"0": 0.9, "0": 0.4, "1": 0.6}}'
        assert parse_structured_payload(payload, "pmf", log_failures=False) is None

    def test_the_other_block_types_refuse_a_pmf_payload(self) -> None:
        """The discriminator keeps ``pmf`` out of the percentile and ballot models."""
        for other in ("numeric", "date", "multiple_choice", "binary"):
            assert parse_structured_payload(json.dumps(self.EXAMPLE), other, log_failures=False) is None
