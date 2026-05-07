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
    BinaryStructured,
    DiscreteCountStructured,
    EvidenceItem,
    MultipleChoiceStructured,
    NumericStructured,
    ScenarioBranch,
    StatedBaseRate,
    StatedHazard,
    StatedPrior,
    TailMass,
    extract_first_balanced_braces,
    extract_json_block,
    parse_structured_block,
)

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
def valid_tails() -> TailMass:
    return TailMass(below_min_expected=0.05, above_max_expected=0.05)


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
def valid_numeric_block(valid_prior: StatedPrior, valid_tails: TailMass) -> NumericStructured:
    return NumericStructured(
        question_type="numeric",
        prior=valid_prior,
        declared_percentiles={0.1: 10.0, 0.5: 50.0, 0.9: 90.0},
        distribution_family_hint="normal",
        student_t_df=None,
        tails=valid_tails,
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

    def test_posterior_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            BinaryStructured(question_type="binary", posterior_prob=1.5)

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            # The unknown_field is intentionally invalid — this test asserts
            # Pydantic's extra="forbid" rejects it. Static-typing complaint is
            # expected and correct; we care about the runtime behavior.
            BinaryStructured(question_type="binary", posterior_prob=0.5, unknown_field="oops")  # type: ignore[call-arg]


class TestNumericStructuredHappyPath:
    def test_full_construction(self, valid_numeric_block: NumericStructured) -> None:
        n = valid_numeric_block
        assert n.question_type == "numeric"
        assert set(n.declared_percentiles.keys()) >= {0.1, 0.5, 0.9}
        assert n.declared_percentiles[0.5] == pytest.approx(50.0)
        assert n.distribution_family_hint == "normal"
        assert isinstance(n.tails, TailMass)
        assert n.tails.below_min_expected == pytest.approx(0.05)

    def test_only_required_fields(self) -> None:
        n = NumericStructured(
            question_type="numeric",
            declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
        )
        assert n.prior is None
        assert n.tails is None
        assert n.scenarios == []
        assert n.student_t_df is None

    def test_accepts_extra_percentiles(self) -> None:
        n = NumericStructured(
            question_type="numeric",
            declared_percentiles={0.05: 0.5, 0.1: 1.0, 0.25: 2.5, 0.5: 5.0, 0.9: 9.0, 0.95: 9.5},
        )
        assert len(n.declared_percentiles) == 6

    def test_student_t_df_valid(self) -> None:
        n = NumericStructured(
            question_type="numeric",
            declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
            student_t_df=4.0,
        )
        assert n.student_t_df == pytest.approx(4.0)

    def test_mixture_family_hint_rejected(self) -> None:
        # "mixture" was dropped from the distribution_family_hint literal
        # along with MixtureComponent; the Pydantic literal validator must
        # now reject it.
        with pytest.raises(ValidationError):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 10.0, 0.5: 50.0, 0.9: 90.0},
                distribution_family_hint="mixture",  # type: ignore[arg-type]
            )


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


class TestTailMassValidators:
    def test_sum_exactly_at_ceiling_raises(self) -> None:
        with pytest.raises(ValidationError, match="TailMass sum"):
            TailMass(below_min_expected=0.25, above_max_expected=0.25)

    def test_sum_above_ceiling_raises(self) -> None:
        with pytest.raises(ValidationError, match="TailMass sum"):
            TailMass(below_min_expected=0.3, above_max_expected=0.3)

    def test_small_sum_ok(self) -> None:
        t = TailMass(below_min_expected=0.1, above_max_expected=0.1)
        assert t.below_min_expected + t.above_max_expected == pytest.approx(0.2)

    def test_zero_ok(self) -> None:
        t = TailMass(below_min_expected=0.0, above_max_expected=0.0)
        assert t.below_min_expected == 0.0


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

    def test_student_t_df_at_boundary_raises(self) -> None:
        with pytest.raises(ValidationError, match="student_t_df must be > 1"):
            NumericStructured(
                question_type="numeric",
                declared_percentiles={0.1: 1.0, 0.5: 5.0, 0.9: 9.0},
                student_t_df=1.0,
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
        with pytest.raises(ValidationError, match="beta_binom_ceiling.*requires ceiling"):
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

    def test_no_block_returns_none_and_debug_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        rationale = "Prose with no JSON block at all."
        with caplog.at_level(logging.DEBUG, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        # Should be DEBUG-level, not WARNING
        assert any(
            record.levelno == logging.DEBUG and "No JSON block found" in record.message for record in caplog.records
        )
        assert not any(record.levelno >= logging.WARNING for record in caplog.records)

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
