"""Tests for structured_output_schema module — Pydantic schemas for per-question-type
structured rationale blocks and their extraction/parsing helpers.
"""

from __future__ import annotations

import json
import logging

import pytest
from pydantic import ValidationError

from metaculus_bot.structured_output_schema import (
    _MAX_STRUCTURED_BLOCK_BYTES,
    BaseRateAnchor,
    BinaryStructured,
    CriteriaClause,
    DiscreteCountStructured,
    EvidenceItem,
    MultipleChoiceStructured,
    NumericStructured,
    ScenarioBranch,
    StatedBaseRate,
    StatedHazard,
    StatedPrior,
    extract_first_balanced_braces,
    extract_json_block,
    extract_json_block_candidates,
    iter_balanced_braces,
    parse_structured_block,
)
from metaculus_bot.tool_runner import _aggregate_binary_lines, _parse_all_blocks

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def valid_prior() -> StatedPrior:
    return StatedPrior(prob=0.25, source="historical base rate over 20 years")


@pytest.fixture
def valid_base_rate() -> StatedBaseRate:
    return StatedBaseRate(k=3, n=20, ref_class="similar elections since 2000")


@pytest.fixture
def valid_hazard() -> StatedHazard:
    return StatedHazard(
        rate_per_unit=0.05,
        unit="month",
        window_duration_units=12.0,
        elapsed_fraction=0.3,
        remaining_fraction=0.7,
    )


@pytest.fixture
def valid_evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(summary="Recent poll shift toward candidate", direction="up", strength="moderate"),
        EvidenceItem(summary="Negative debate performance", direction="down", strength="weak", likelihood_ratio=0.7),
    ]


@pytest.fixture
def valid_scenarios_binary() -> list[ScenarioBranch]:
    return [
        ScenarioBranch(name="momentum continues", prob=0.6, conditional_outcome="yes"),
        ScenarioBranch(name="reversal", prob=0.4, conditional_outcome="no"),
    ]


@pytest.fixture
def valid_binary_block(
    valid_prior: StatedPrior,
    valid_base_rate: StatedBaseRate,
    valid_hazard: StatedHazard,
    valid_evidence: list[EvidenceItem],
    valid_scenarios_binary: list[ScenarioBranch],
) -> BinaryStructured:
    return BinaryStructured(
        question_type="binary",
        prior=valid_prior,
        base_rate=valid_base_rate,
        hazard=valid_hazard,
        evidence=valid_evidence,
        scenarios=valid_scenarios_binary,
        posterior_prob=0.35,
    )


@pytest.fixture
def valid_numeric_block(valid_prior: StatedPrior) -> NumericStructured:
    return NumericStructured(
        question_type="numeric",
        prior=valid_prior,
        declared_percentiles={0.1: 10.0, 0.5: 50.0, 0.9: 90.0},
        scenarios=[],
    )


@pytest.fixture
def valid_mc_block(valid_prior: StatedPrior) -> MultipleChoiceStructured:
    return MultipleChoiceStructured(
        question_type="multiple_choice",
        prior=valid_prior,
        option_probs={"A": 0.5, "B": 0.3, "C": 0.2},
        other_mass=0.0,
        concentration=5.0,
    )


@pytest.fixture
def valid_discrete_block(valid_prior: StatedPrior) -> DiscreteCountStructured:
    return DiscreteCountStructured(
        question_type="discrete_count",
        prior=valid_prior,
        mean_estimate=4.2,
        dispersion="negbinom",
        ceiling=None,
        overdispersion_factor=1.5,
        declared_percentiles={0.1: 1.0, 0.5: 4.0, 0.9: 9.0},
    )


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
        with pytest.raises(ValidationError):
            # The unknown_field is intentionally invalid — this test asserts
            # Pydantic's extra="forbid" rejects it. Static-typing complaint is
            # expected and correct; we care about the runtime behavior.
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


class TestBinaryTelemetryStripAndRetry:
    """Strip-and-retry recovery for malformed BINARY telemetry (2026-07-08).

    Contract: a good ``posterior_prob`` (and other core fields) must survive
    a malformed ``base_rate_anchor`` / ``criteria_clauses`` value — dropping
    the entire block on a pure telemetry formatting bug would silently shift
    stacker input via the cross-model aggregation path in ``tool_runner``.

    Fail-fast on core fields is preserved: a bad ``posterior_prob`` must
    still return None even if telemetry is well-formed.
    """

    def test_criteria_clauses_null_recovers_core_block(self, caplog: pytest.LogCaptureFixture) -> None:
        # Canonical failure: prompt says "omit" criteria_clauses when there
        # isn't a conjunctive breakdown, but LLMs frequently emit `null`
        # instead. Old behavior: whole block dropped, base-rate blend and
        # prior/posterior contributions vanish. New behavior: warn + keep
        # the core binary block.
        rationale = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "prior": {"prob": 0.2, "source": "20yr base rate"},
                    "base_rate": {"k": 4, "n": 20, "ref_class": "past 20 yrs"},
                    "posterior_prob": 0.35,
                    "criteria_clauses": None,
                }
            )
            + "\n```"
        )
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.35)
        # Core telemetry-adjacent fields preserved.
        assert result.prior is not None
        assert result.prior.prob == pytest.approx(0.2)
        assert result.base_rate is not None
        assert result.base_rate.k == 4
        # Telemetry defaults after strip-and-retry.
        assert result.base_rate_anchor is None
        assert result.criteria_clauses == []
        # WARNING logged so this recovery is visible in run logs.
        assert any(
            "malformed telemetry fields" in rec.message and "criteria_clauses" in rec.message for rec in caplog.records
        )

    def test_reversed_anchor_recovers_core_block(self, caplog: pytest.LogCaptureFixture) -> None:
        # Canonical failure: {low: 0.6, high: 0.2} is rejected by
        # BaseRateAnchor's ordering validator. Same recovery contract as
        # criteria_clauses=null.
        rationale = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "posterior_prob": 0.42,
                    "base_rate_anchor": {"low": 0.6, "high": 0.2},
                }
            )
            + "\n```"
        )
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)
        assert result.base_rate_anchor is None
        assert any(
            "malformed telemetry fields" in rec.message and "base_rate_anchor" in rec.message for rec in caplog.records
        )

    def test_both_telemetry_fields_malformed_recovers(self, caplog: pytest.LogCaptureFixture) -> None:
        # Both telemetry keys present and malformed: strip both, keep core.
        rationale = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "posterior_prob": 0.28,
                    "base_rate_anchor": {"low": 0.9, "high": 0.1},
                    "criteria_clauses": None,
                }
            )
            + "\n```"
        )
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.28)
        assert result.base_rate_anchor is None
        assert result.criteria_clauses == []
        message_text = " ".join(rec.message for rec in caplog.records)
        assert "base_rate_anchor" in message_text
        assert "criteria_clauses" in message_text

    def test_bad_core_field_still_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        # posterior_prob=1.5 is a core-field violation. Even with valid
        # telemetry alongside, the block must still be dropped — strip-and-
        # retry MUST NOT rescue a bad core field.
        rationale = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "posterior_prob": 1.5,
                    "base_rate_anchor": {"low": 0.15, "high": 0.35},
                    "criteria_clauses": [{"name": "clause", "prob": 0.5}],
                }
            )
            + "\n```"
        )
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        # Original failed-validation warning still fires; no recovery warning.
        assert any("failed validation" in rec.message for rec in caplog.records)
        assert not any("malformed telemetry fields" in rec.message for rec in caplog.records)

    def test_bad_core_and_bad_telemetry_still_none(self) -> None:
        # Neither retry variant is valid — bad core, bad telemetry. Must
        # return None (fall through to the original None return).
        rationale = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "posterior_prob": 1.5,  # out of range
                    "base_rate_anchor": {"low": 0.9, "high": 0.1},  # reversed
                }
            )
            + "\n```"
        )
        result = parse_structured_block(rationale, "binary")
        assert result is None

    def test_recovered_block_feeds_cross_model_aggregation(self) -> None:
        # Guard the invariant end-to-end: a forecaster whose ONLY validation
        # error is malformed telemetry must still contribute its base_rate
        # to the cross-model aggregation. Uses tool_runner's internal
        # _parse_all_blocks + _aggregate_binary_lines directly to keep the
        # test independent of feature-flag env state.

        good = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "base_rate": {"k": 2, "n": 10, "ref_class": "ref"},
                    "posterior_prob": 0.25,
                }
            )
            + "\n```"
        )
        # This rationale would previously drop entirely because of the null
        # criteria_clauses; after strip-and-retry it contributes to the
        # base_rate blend.
        recovered = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "base_rate": {"k": 6, "n": 10, "ref_class": "ref"},
                    "posterior_prob": 0.55,
                    "criteria_clauses": None,
                }
            )
            + "\n```"
        )
        blocks = _parse_all_blocks([good, recovered], "binary")
        assert len(blocks) == 2  # both survive — invariant restored
        lines = _aggregate_binary_lines([0.25, 0.55], [b for b in blocks if isinstance(b, BinaryStructured)])
        blend_line = next((line for line in lines if "Blended base rate" in line), None)
        assert blend_line is not None
        # Blend should reflect BOTH forecasters (n=2), not just the well-formed one.
        assert "2 forecasters" in blend_line


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
        # The dead `tails` / TailMass slot was removed (W2). NumericStructured
        # uses extra="forbid", so passing a `tails` key must now raise, and
        # TailMass must no longer be importable from the schema module.
        with pytest.raises(ValidationError):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
                tails={"below_min_expected": 0.05, "above_max_expected": 0.05},  # type: ignore[call-arg]
            )
        import metaculus_bot.structured_output_schema as schema

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

    def test_outcome_type_invalid_raises(self) -> None:
        """C3: invalid outcome_type string rejected by Literal constraint."""
        with pytest.raises(ValidationError):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
                outcome_type="unknown",  # type: ignore[arg-type]
            )

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
        # 0.99 sum is within 0.02 tolerance of 1.0
        m = MultipleChoiceStructured(
            question_type="multiple_choice",
            option_probs={"A": 0.33, "B": 0.33, "C": 0.33},
        )
        assert sum(m.option_probs.values()) == pytest.approx(0.99)


class TestRemainingWindowDaysTelemetryField:
    """The optional ``remaining_window_days`` slot (2026-09-02) behind the
    WINDOW_DECLARED marker: the days-from-now-to-deadline the forecaster says it
    priced. Two contracts. (1) Back-compat: a block without the key parses exactly as
    before, on both the binary and MC schemas. (2) It can never cost a forecast — the
    field is read leniently, because a formatting slip on a measurement field must not
    drop the block that carries the forecast (MC has no telemetry strip-and-retry, so a
    strict int there would push a good ballot down to the LLM salvage rung)."""

    @staticmethod
    def _binary_block(**extra: object) -> BinaryStructured | None:
        payload = json.dumps({"question_type": "binary", "posterior_prob": 0.28, **extra})
        block = parse_structured_block(f"```json\n{payload}\n```", "binary")
        assert block is None or isinstance(block, BinaryStructured)
        return block

    @staticmethod
    def _mc_block(**extra: object) -> MultipleChoiceStructured | None:
        payload = json.dumps({"question_type": "multiple_choice", "option_probs": {"A": 0.6, "B": 0.4}, **extra})
        block = parse_structured_block(f"```json\n{payload}\n```", "multiple_choice")
        assert block is None or isinstance(block, MultipleChoiceStructured)
        return block

    def test_binary_block_with_the_field_round_trips(self) -> None:
        block = self._binary_block(remaining_window_days=45)
        assert block is not None
        assert block.remaining_window_days == 45

    def test_binary_block_without_the_field_parses_as_absent(self) -> None:
        block = self._binary_block()
        assert block is not None
        assert block.posterior_prob == pytest.approx(0.28)
        assert block.remaining_window_days is None

    def test_mc_block_with_the_field_round_trips(self) -> None:
        block = self._mc_block(remaining_window_days=12)
        assert block is not None
        assert block.remaining_window_days == 12

    def test_mc_block_without_the_field_parses_as_absent(self) -> None:
        block = self._mc_block()
        assert block is not None
        assert block.option_probs == {"A": 0.6, "B": 0.4}
        assert block.remaining_window_days is None

    def test_numeric_schema_still_forbids_the_key(self) -> None:
        """The field ships on the binary and MC schemas only. Numeric prompts do not ask
        for it, so on a numeric block it is an unexpected key and extra="forbid" applies
        — asserted so a later prompt edit that adds it there fails loudly here."""
        with pytest.raises(ValidationError):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 1.0, 0.5: 2.0, 0.9: 3.0},
                remaining_window_days=45,  # type: ignore[call-arg]
            )

    @pytest.mark.parametrize(
        ("declared", "expected"),
        [
            (45, 45),  # the ordinary case
            ("45", 45),  # a quoted integer is the same declaration, written differently
            (45.0, 45),  # so is a float
            (44.6, 45),  # rounded rather than truncated
            ("about 45 days", None),  # prose we decline to mine for a number
            (True, None),  # a JSON bool is not a day count (and int(True) == 1)
            ({"days": 45}, None),  # nested shapes carry no reading
            (None, None),
        ],
    )
    def test_unreadable_declarations_become_absent_and_never_drop_the_block(
        self, declared: object, expected: int | None
    ) -> None:
        block = self._binary_block(remaining_window_days=declared)
        assert block is not None, "an unreadable telemetry value must never cost the forecast"
        assert block.posterior_prob == pytest.approx(0.28)
        assert block.remaining_window_days == expected

    def test_an_unreadable_mc_declaration_keeps_the_ballot(self) -> None:
        """The MC half of the same guarantee: MC has no telemetry strip-and-retry, so
        leniency at the field is the only thing standing between a malformed telemetry
        value and a ballot that has to be re-parsed by the LLM salvage rung."""
        block = self._mc_block(remaining_window_days="the whole window")
        assert block is not None
        assert block.option_probs == {"A": 0.6, "B": 0.4}
        assert block.remaining_window_days is None


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
        # 0.49 + 0.505 = 0.995, within 0.01 tolerance
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
        with pytest.raises(ValidationError):
            # "sideways" is intentionally outside the Literal — this test
            # asserts Pydantic rejects it at runtime.
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
        # 0.49 + 0.52 = 1.01, within 0.02 tolerance
        scenarios = [ScenarioBranch(name="a", prob=0.49), ScenarioBranch(name="b", prob=0.52)]
        b = BinaryStructured(question_type="binary", posterior_prob=0.5, scenarios=scenarios)
        assert sum(s.prob for s in b.scenarios) == pytest.approx(1.01)

    def test_numeric_scenarios_within_tolerance_ok(self) -> None:
        # 0.5 + 0.485 = 0.985, within 0.02 tolerance of 1.0
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

    def test_non_monotone_raises(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 10.0, 0.5: 5.0, 0.9: 15.0},
            )

    def test_equal_values_raise(self) -> None:
        # equal (not strictly increasing) should raise
        with pytest.raises(ValidationError, match="strictly increasing"):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 5.0, 0.5: 5.0, 0.9: 10.0},
            )

    def test_percentile_key_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError, match="Percentile keys"):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0, 1.5: 15.0},
            )


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

    def test_concentration_zero_raises(self) -> None:
        with pytest.raises(ValidationError, match="concentration must be > 0"):
            MultipleChoiceStructured(
                question_type="multiple_choice",
                option_probs={"A": 0.5, "B": 0.5},
                concentration=0.0,
            )


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
# extract_json_block
# ===========================================================================


class TestExtractJsonBlock:
    def test_fenced_json_block_returned_trimmed(self) -> None:
        text = 'Some text\n```json\n{"question_type": "binary", "posterior_prob": 0.5}\n```\ntail'
        body = extract_json_block(text)
        assert body is not None
        assert body.startswith("{")
        assert body.endswith("}")
        assert '"question_type"' in body

    def test_returns_last_fenced_block(self) -> None:
        text = (
            "intro\n"
            '```json\n{"question_type": "binary", "posterior_prob": 0.1}\n```\n'
            "middle\n"
            '```json\n{"question_type": "binary", "posterior_prob": 0.9}\n```\n'
        )
        body = extract_json_block(text)
        assert body is not None
        assert '"posterior_prob": 0.9' in body
        assert '"posterior_prob": 0.1' not in body

    def test_no_block_returns_none(self) -> None:
        assert extract_json_block("Plain prose with no fence.") is None

    def test_empty_input_returns_none(self) -> None:
        assert extract_json_block("") is None

    def test_unclosed_fence_returns_none(self) -> None:
        # Opening fence without closing fence should not match
        text = '```json\n{"question_type": "binary"}\n'
        assert extract_json_block(text) is None

    def test_case_insensitive_json_tag(self) -> None:
        text = '```JSON\n{"question_type": "binary", "posterior_prob": 0.5}\n```'
        body = extract_json_block(text)
        assert body is not None
        assert '"posterior_prob"' in body

    def test_mixed_case_json_tag(self) -> None:
        text = '```Json\n{"question_type": "binary", "posterior_prob": 0.5}\n```'
        body = extract_json_block(text)
        assert body is not None

    def test_whitespace_around_tag(self) -> None:
        text = '```   json   \n{"question_type": "binary", "posterior_prob": 0.5}\n```'
        body = extract_json_block(text)
        assert body is not None
        assert '"question_type"' in body

    def test_untagged_fence_with_json_object_body(self) -> None:
        text = '```\n{"question_type": "binary", "posterior_prob": 0.5}\n```'
        body = extract_json_block(text)
        assert body is not None
        assert '"question_type"' in body

    def test_untagged_fence_with_non_json_body_returns_none(self) -> None:
        text = "```\nplain prose body\n```"
        assert extract_json_block(text) is None

    def test_prefers_tagged_over_untagged(self) -> None:
        # Untagged code fence (with JSON-like content) should be ignored when a tagged json block exists.
        text = '```\n{"untagged": true}\n```\nsome text\n```json\n{"tagged": true}\n```\n'
        body = extract_json_block(text)
        assert body is not None
        assert '"tagged": true' in body
        assert "untagged" not in body

    def test_empty_body_ignored(self) -> None:
        text = "```json\n\n```"
        assert extract_json_block(text) is None


class TestExtractFirstBalancedBraces:
    """Cover the string-literal-aware balanced-brace extractor shared by
    ``_parse_gap_list`` (unfenced JSON fallback). Naive brace-counting silently
    truncates JSON that contains braces inside string values — this helper
    must not do that."""

    def test_simple_object(self) -> None:
        assert extract_first_balanced_braces('{"a": 1}') == '{"a": 1}'

    def test_returns_none_on_no_braces(self) -> None:
        assert extract_first_balanced_braces("plain prose") is None

    def test_returns_none_on_empty_input(self) -> None:
        assert extract_first_balanced_braces("") is None

    def test_object_with_prefix_and_suffix_prose(self) -> None:
        text = 'Here is the output:\n{"gap": "g"}\n\nHope that helps!'
        assert extract_first_balanced_braces(text) == '{"gap": "g"}'

    def test_brace_inside_string_value_not_counted(self) -> None:
        # The crux of F11: a naive brace counter closes the object at the `}`
        # inside the string value, producing '{"foo": "has a }'.
        text = '{"foo": "has a } brace", "b": 1}'
        assert extract_first_balanced_braces(text) == text

    def test_opening_brace_inside_string_value_not_counted(self) -> None:
        text = '{"foo": "has a { brace", "b": 1}'
        assert extract_first_balanced_braces(text) == text

    def test_mixed_braces_in_string_values(self) -> None:
        text = '{"a": "has } and { chars", "b": 1}'
        assert extract_first_balanced_braces(text) == text

    def test_escaped_quote_inside_string(self) -> None:
        # `\"` should NOT exit the string, so the `}` that follows is still
        # inside the string literal.
        text = '{"a": "quote \\" then } brace", "b": 1}'
        assert extract_first_balanced_braces(text) == text

    def test_escaped_backslash_then_quote_exits_string(self) -> None:
        # `\\` is an escaped backslash; the following `"` then exits the
        # string. Without correct escape handling we'd stay inside and miss
        # the final `}`.
        text = '{"a": "trailing slash \\\\", "b": 1}'
        assert extract_first_balanced_braces(text) == text

    def test_nested_objects(self) -> None:
        text = '{"outer": {"inner": 1}, "k": "v"}'
        assert extract_first_balanced_braces(text) == text

    def test_returns_first_balanced_block_only(self) -> None:
        # Trailing second object is not part of the first balanced block.
        text = '{"first": 1} then {"second": 2}'
        assert extract_first_balanced_braces(text) == '{"first": 1}'

    def test_unbalanced_returns_none(self) -> None:
        assert extract_first_balanced_braces('{"a": 1') is None


class TestExtractJsonBlockCandidates:
    """The candidate ranking behind validity-aware selection: tagged before
    untagged, last-by-position first within a tier, empty bodies skipped."""

    def test_empty_input_returns_empty_list(self) -> None:
        assert extract_json_block_candidates("") == []

    def test_no_fence_returns_empty_list(self) -> None:
        assert extract_json_block_candidates("plain prose, no fence") == []

    def test_tagged_ranked_last_by_position_first(self) -> None:
        text = '```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```\n'
        assert extract_json_block_candidates(text) == ['{"b": 2}', '{"a": 1}']

    def test_tagged_ranked_ahead_of_untagged(self) -> None:
        # Untagged appears LATER in the text but still ranks below the tagged one.
        text = '```json\n{"tagged": 1}\n```\n```\n{"untagged": 2}\n```\n'
        assert extract_json_block_candidates(text) == ['{"tagged": 1}', '{"untagged": 2}']

    def test_empty_bodied_fence_skipped(self) -> None:
        text = '```json\n\n```\n```json\n{"a": 1}\n```\n'
        assert extract_json_block_candidates(text) == ['{"a": 1}']

    def test_extract_json_block_returns_first_candidate(self) -> None:
        text = '```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```\n'
        assert extract_json_block(text) == '{"b": 2}'
        assert extract_json_block(text) == extract_json_block_candidates(text)[0]


class TestIterBalancedBraces:
    """iter_balanced_braces yields EVERY top-level balanced block; the repair
    rung iterates them so a junk leading blob doesn't block a valid later one."""

    def test_yields_multiple_top_level_blocks(self) -> None:
        text = 'junk {"first": 1} middle {"second": 2} tail'
        assert list(iter_balanced_braces(text)) == ['{"first": 1}', '{"second": 2}']

    def test_no_braces_yields_nothing(self) -> None:
        assert list(iter_balanced_braces("plain prose")) == []

    def test_stops_after_unbalanced_run(self) -> None:
        # First blob closes; the second run is unbalanced, so nothing further.
        text = '{"ok": 1} then {"unbalanced": '
        assert list(iter_balanced_braces(text)) == ['{"ok": 1}']

    def test_brace_inside_string_not_counted_across_blocks(self) -> None:
        text = '{"a": "has } brace"} and {"b": 2}'
        assert list(iter_balanced_braces(text)) == ['{"a": "has } brace"}', '{"b": 2}']

    def test_first_of_iter_matches_extract_first_balanced_braces(self) -> None:
        text = '{"first": 1} then {"second": 2}'
        assert next(iter_balanced_braces(text)) == extract_first_balanced_braces(text)


# ===========================================================================
# parse_structured_block
# ===========================================================================


class TestParseStructuredBlock:
    def test_valid_binary_rationale(self) -> None:
        payload = {"question_type": "binary", "posterior_prob": 0.35}
        rationale = f"My thinking...\n```json\n{json.dumps(payload)}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.35)

    def test_valid_numeric_rationale(self) -> None:
        payload = {
            "question_type": "numeric",
            "declared_percentiles": {"0.1": 1.0, "0.5": 5.0, "0.9": 9.0},
        }
        rationale = f"Analysis...\n```json\n{json.dumps(payload)}\n```"
        result = parse_structured_block(rationale, "numeric")
        assert isinstance(result, NumericStructured)
        assert result.declared_percentiles is not None
        assert result.declared_percentiles[0.5] == pytest.approx(5.0)

    def test_valid_mc_rationale(self) -> None:
        payload = {
            "question_type": "multiple_choice",
            "option_probs": {"A": 0.6, "B": 0.4},
        }
        rationale = f"```json\n{json.dumps(payload)}\n```"
        result = parse_structured_block(rationale, "multiple_choice")
        assert isinstance(result, MultipleChoiceStructured)
        assert result.option_probs == {"A": 0.6, "B": 0.4}

    def test_discrete_count_class_still_constructable(self) -> None:
        # Discrete-count dispatch is phase-3, but the class remains available
        # so prompts and future runtime wiring can use it.
        d = DiscreteCountStructured(question_type="discrete_count", mean_estimate=2.0, dispersion="poisson")
        assert d.mean_estimate == pytest.approx(2.0)

    def test_no_block_returns_none_and_info_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        rationale = "Prose with no JSON block at all."
        with caplog.at_level(logging.INFO, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        # A0b: lifted from DEBUG to INFO so block-reliability is visible in run logs
        assert any(
            record.levelno == logging.INFO and "No JSON block found" in record.message for record in caplog.records
        )
        # Scoped to our own loggers: caplog.records spans every logger that propagates
        # to root, so an unrelated third-party WARNING would otherwise fail this.
        our_warnings = [
            r for r in caplog.records if r.levelno >= logging.WARNING and r.name.startswith("metaculus_bot")
        ]
        assert not our_warnings, [r.getMessage() for r in our_warnings]

    def test_log_failures_false_silences_the_strict_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """The publish-path telemetry read (``_log_window_declared`` in
        ``forecaster_runners``) runs BESIDE the extraction ladder, which repairs blocks
        this strict parse rejects. Logging the strict failure there would warn about a
        forecast that published fine, so that caller passes ``log_failures=False`` and
        both the no-block INFO and the validation WARNING go quiet. The return value is
        unchanged — only the logging is."""
        for rationale in ("Prose with no JSON block at all.", "```json\n{this is not valid json\n```"):
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="metaculus_bot.structured_output_schema"):
                result = parse_structured_block(rationale, "binary", log_failures=False)
            assert result is None
            ours = [r for r in caplog.records if r.name.startswith("metaculus_bot")]
            assert not ours, [r.getMessage() for r in ours]

    def test_malformed_json_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        rationale = "```json\n{this is not valid json\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        assert any("Malformed JSON" in record.message for record in caplog.records)
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_missing_required_field_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        # Missing posterior_prob
        payload = {"question_type": "binary"}
        rationale = f"```json\n{json.dumps(payload)}\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        assert any("failed validation" in record.message for record in caplog.records)
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_question_type_mismatch_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        payload = {"question_type": "binary", "posterior_prob": 0.5}
        rationale = f"```json\n{json.dumps(payload)}\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "numeric")
        assert result is None
        assert any("question_type mismatch" in record.message for record in caplog.records)
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_missing_question_type_in_payload_injected(self) -> None:
        # question_type missing in payload — parser should inject it.
        payload = {"posterior_prob": 0.42}
        rationale = f"```json\n{json.dumps(payload)}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.question_type == "binary"
        assert result.posterior_prob == pytest.approx(0.42)

    def test_json_array_payload_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        rationale = "```json\n[1, 2, 3]\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        assert any("must decode to a JSON object" in record.message for record in caplog.records)

    def test_roundtrip_binary(self, valid_binary_block: BinaryStructured) -> None:
        dumped = valid_binary_block.model_dump_json()
        rationale = f"Reasoning here.\n```json\n{dumped}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.model_dump() == valid_binary_block.model_dump()

    def test_roundtrip_numeric(self, valid_numeric_block: NumericStructured) -> None:
        dumped = valid_numeric_block.model_dump_json()
        rationale = f"```json\n{dumped}\n```"
        result = parse_structured_block(rationale, "numeric")
        assert isinstance(result, NumericStructured)
        assert result.model_dump() == valid_numeric_block.model_dump()

    def test_roundtrip_mc(self, valid_mc_block: MultipleChoiceStructured) -> None:
        dumped = valid_mc_block.model_dump_json()
        rationale = f"```json\n{dumped}\n```"
        result = parse_structured_block(rationale, "multiple_choice")
        assert isinstance(result, MultipleChoiceStructured)
        assert result.model_dump() == valid_mc_block.model_dump()

    def test_discrete_count_roundtrip_via_model(self, valid_discrete_block: DiscreteCountStructured) -> None:
        # Discrete-count does not flow through parse_structured_block (phase-3),
        # but the pydantic class must still round-trip via JSON on its own.
        dumped = valid_discrete_block.model_dump_json()
        loaded = DiscreteCountStructured.model_validate_json(dumped)
        assert loaded.model_dump() == valid_discrete_block.model_dump()


class TestValidityAwareBlockSelection:
    """Selection keeps the last block that VALIDATES, not the last by position.

    Regression: a trailing schema-recap / example block (the model echoing the
    STRUCTURED FORECAST schema after its real forecast) used to shadow a valid
    earlier block, because ``extract_json_block`` returned the last block
    unconditionally and ``parse_structured_block`` validated only that one. A
    model swap could surface this without warning, and a recap block with
    DIFFERENT numbers would publish the wrong forecast. Selection now walks all
    candidates and keeps the first that validates for the requested type.
    """

    def test_valid_block_then_malformed_trailing_block_selects_valid(self) -> None:
        # The team lead's reproduction, adapted to the real schema field
        # (``posterior_prob`` — the verbatim repro used ``prediction_in_decimal``,
        # which is not a BinaryStructured field, so BOTH its blocks are invalid).
        # A valid forecast block followed by a malformed schema-recap.
        text = (
            "reasoning here\n"
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            "Note, the schema looks like:\n"
            '```json\n{"question_type": "binary", "posterior_prob": <your value>}\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_valid_block_then_schema_invalid_trailing_selects_valid(self) -> None:
        # Trailing block is well-formed JSON but fails schema validation
        # (posterior_prob outside [0, 1]); the valid earlier block is kept.
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            '```json\n{"question_type": "binary", "posterior_prob": 1.5}\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_valid_block_then_wrong_qtype_trailing_selects_valid(self) -> None:
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            '```json\n{"question_type": "numeric", "declared_percentiles": '
            '{"0.1": 1.0, "0.5": 5.0, "0.9": 9.0}}\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_two_valid_blocks_last_by_position_wins(self) -> None:
        # Tiebreak preserved: among VALID blocks the last by position wins
        # (the prompt asks for the forecast block last).
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.1}\n```\n'
            '```json\n{"question_type": "binary", "posterior_prob": 0.9}\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.9)

    def test_only_malformed_block_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        # Honest-failure path unchanged: no valid candidate → None at WARNING.
        text = "```json\n{this is not valid json\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(text, "binary")
        assert result is None
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_valid_untagged_recovered_when_tagged_all_invalid(self) -> None:
        # Tagged blocks are tried first; when none validate, a valid untagged
        # ``` fence is recovered (previously any tagged block suppressed untagged).
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": <bad>}\n```\n'
            '```\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_valid_tagged_outranks_valid_untagged(self) -> None:
        # Preference order preserved: a valid tagged block wins over a valid
        # untagged one even when the untagged appears later in the text.
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            '```\n{"question_type": "binary", "posterior_prob": 0.9}\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_skip_then_recover_logs_info_not_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            '```json\n{"question_type": "binary", "posterior_prob": <your value>}\n```\n'
        )
        with caplog.at_level(logging.INFO, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        # A skipped-then-recovered trailing block is an INFO signal (the prompt
        # contract eroding), never a scary WARNING — extraction succeeded.
        assert any(record.levelno == logging.INFO and "skip" in record.message.lower() for record in caplog.records)
        # Scoped to our own loggers: caplog.records spans every logger that propagates
        # to root, so an unrelated third-party WARNING would otherwise fail this.
        our_warnings = [
            r for r in caplog.records if r.levelno >= logging.WARNING and r.name.startswith("metaculus_bot")
        ]
        assert not our_warnings, [r.getMessage() for r in our_warnings]

    def test_truncated_closed_final_block_skipped_for_valid(self) -> None:
        # Model hit its token limit mid-block but the fence still closed: the
        # truncated JSON fails to parse and the earlier valid block is kept.
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            '```json\n{"question_type": "binary", "posterior_pr\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_truncated_unclosed_final_fence_ignored(self) -> None:
        # An unclosed final fence never matches the fence pattern, so it can't
        # shadow the valid earlier block.
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            '```json\n{"question_type": "binary", "posterior_pr'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_numeric_valid_then_malformed_trailing_selects_valid(self) -> None:
        text = (
            '```json\n{"question_type": "numeric", "declared_percentiles": '
            '{"0.1": 1.0, "0.5": 5.0, "0.9": 9.0}}\n```\n'
            '```json\n{"question_type": "numeric", "declared_percentiles": '
            '{"0.1": <a>, "0.5": <b>, "0.9": <c>}}\n```\n'
        )
        result = parse_structured_block(text, "numeric")
        assert isinstance(result, NumericStructured)
        assert result.declared_percentiles is not None
        assert result.declared_percentiles[0.5] == pytest.approx(5.0)

    def test_mc_valid_then_malformed_trailing_selects_valid(self) -> None:
        text = (
            '```json\n{"question_type": "multiple_choice", "option_probs": {"A": 0.6, "B": 0.4}}\n```\n'
            '```json\n{"question_type": "multiple_choice", "option_probs": {"A": <x>, "B": <y>}}\n```\n'
        )
        result = parse_structured_block(text, "multiple_choice")
        assert isinstance(result, MultipleChoiceStructured)
        assert result.option_probs == {"A": 0.6, "B": 0.4}


# ===========================================================================
# Realistic rationale fixture
# ===========================================================================


REALISTIC_BINARY_RATIONALE = """\
Question: Will Country X's inflation rate exceed 5% by year-end?

Relevant base rate: Over the past 20 years, Country X has had inflation above 5%
in 4 out of 20 full years, giving a rough prior of 20%.

Recent signals:
- The central bank raised rates by 75bp in the last two meetings, which pushes DOWN
- Food prices (30% of CPI basket) have surged 8% YoY, which pushes UP
- Wage growth accelerating to 6% nominal, pushing UP
- Energy subsidies extended through Q4, pushing DOWN

Weighting the evidence, I think this is above the base rate but below 50%. The
rate hikes are lagging; inflation pressure is real but partially offset by policy.

Probability: 35%

```json
{
    "question_type": "binary",
    "prior": {"prob": 0.20, "source": "20-year base rate for Country X"},
    "base_rate": {"k": 4, "n": 20, "ref_class": "past 20 annual CPI readings"},
    "posterior_prob": 0.35
}
```
"""


class TestRealisticRationale:
    def test_extract_picks_json_block(self) -> None:
        body = extract_json_block(REALISTIC_BINARY_RATIONALE)
        assert body is not None
        assert '"question_type": "binary"' in body
        # The "Probability: 35%" line should NOT leak into the extracted body.
        assert "Probability: 35%" not in body
        # Body should be parseable as JSON.
        parsed = json.loads(body)
        assert parsed["posterior_prob"] == pytest.approx(0.35)

    def test_parse_structured_block_from_realistic_rationale(self) -> None:
        result = parse_structured_block(REALISTIC_BINARY_RATIONALE, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.35)
        assert result.prior is not None
        assert result.prior.prob == pytest.approx(0.20)
        assert result.base_rate is not None
        assert result.base_rate.k == 4
        assert result.base_rate.n == 20


# ===========================================================================
# Schema robustness — deep nesting, size cap, unicode, fence edge cases
# ===========================================================================


class TestSchemaRobustness:
    def test_deeply_nested_json_parses_without_crash(self) -> None:
        # Pydantic accepts extra fields from the nested block at the top level
        # only for fields it recognizes. Since BinaryStructured has
        # ``extra="forbid"``, put the nested dict under an unknown key → it
        # will fail validation, which is fine; we're testing the extractor
        # handles deeply nested JSON without blowing up Python's parser.
        nested: dict[str, object] = {"leaf": 1}
        for _ in range(100):
            nested = {"next": nested}
        outer = {"question_type": "binary", "posterior_prob": 0.3, "nested_payload": nested}
        rationale = f"```json\n{json.dumps(outer)}\n```"
        # The block should extract cleanly (no recursion-limit crash). Pydantic
        # will then reject the unknown field, which is not what we're testing.
        body = extract_json_block(rationale)
        assert body is not None
        parsed = json.loads(body)
        # Verify nesting depth preserved.
        cursor = parsed["nested_payload"]
        depth = 0
        while isinstance(cursor, dict) and "next" in cursor:
            cursor = cursor["next"]
            depth += 1
        assert depth == 100

    def test_size_cap_rejects_huge_well_formed_block(self, caplog: pytest.LogCaptureFixture) -> None:
        # Pad a legitimate payload with a huge string field beyond the cap
        # (200KB). The schema has ``extra="forbid"``, so an unknown field
        # would fail — but the size cap kicks in BEFORE pydantic validation.
        huge_body = {"question_type": "binary", "posterior_prob": 0.5, "padding": "x" * 250_000}
        rationale = f"```json\n{json.dumps(huge_body)}\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        assert any("size cap" in rec.message for rec in caplog.records)

    def test_unicode_field_values_parse(self) -> None:
        # Non-ASCII characters in strings and dict keys should round-trip fine.
        payload = {
            "question_type": "binary",
            "prior": {"prob": 0.3, "source": "日本の基準"},
            "posterior_prob": 0.4,
        }
        rationale = f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.prior is not None
        assert result.prior.source == "日本の基準"

    def test_unicode_emoji_option_keys(self) -> None:
        payload = {
            "question_type": "multiple_choice",
            "option_probs": {"🔴 red": 0.5, "🔵 blue": 0.5},
        }
        rationale = f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
        result = parse_structured_block(rationale, "multiple_choice")
        assert isinstance(result, MultipleChoiceStructured)
        assert "🔴 red" in result.option_probs
        assert "🔵 blue" in result.option_probs

    def test_multiple_fenced_blocks_last_wins(self) -> None:
        # Two valid blocks with different posteriors — the extractor should
        # return the last (per docstring contract).
        first = {"question_type": "binary", "posterior_prob": 0.1}
        last = {"question_type": "binary", "posterior_prob": 0.9}
        rationale = f"Draft:\n```json\n{json.dumps(first)}\n```\nRevision:\n```json\n{json.dumps(last)}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.9)

    def test_untagged_fence_with_json_body_parses(self) -> None:
        # Per extract_json_block docstring: untagged ``` fence with a body
        # starting with `{` should still match (fallback to tagged).
        payload = {"question_type": "binary", "posterior_prob": 0.42}
        rationale = f"```\n{json.dumps(payload)}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)


class TestSizeCapBoundary:
    """Boundary coverage for the _MAX_STRUCTURED_BLOCK_BYTES guard."""

    def _padded_binary_payload(self, padding_size: int) -> str:
        # ref_class on StatedBaseRate has no max_length, so padding it is
        # a safe way to hit the size cap without tripping extra="forbid".
        payload = {
            "question_type": "binary",
            "posterior_prob": 0.5,
            "base_rate": {"k": 1, "n": 10, "ref_class": "x" * padding_size},
        }
        return json.dumps(payload)

    def test_just_below_cap_parses_ok(self) -> None:
        # Pad to land just under the cap (leave ~1KB slack for JSON overhead).
        padding = _MAX_STRUCTURED_BLOCK_BYTES - 1000
        raw = self._padded_binary_payload(padding)
        assert len(raw) < _MAX_STRUCTURED_BLOCK_BYTES
        rationale = f"```json\n{raw}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)

    def test_just_over_cap_rejected(self, caplog: pytest.LogCaptureFixture) -> None:
        padding = _MAX_STRUCTURED_BLOCK_BYTES + 100
        raw = self._padded_binary_payload(padding)
        assert len(raw) > _MAX_STRUCTURED_BLOCK_BYTES
        rationale = f"```json\n{raw}\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        assert any("size cap" in rec.message for rec in caplog.records)
