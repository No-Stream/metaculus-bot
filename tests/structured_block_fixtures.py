"""Valid structured-block fixtures shared by the two structured-block test modules.

The block models and the block parser are tested in separate files
(``tests/test_structured_output_schema.py`` and ``tests/test_structured_block_parsing.py``)
and both need the same known-good blocks, so the chain lives here and each module loads it
as a plugin rather than duplicating the bodies.
"""

from __future__ import annotations

import pytest

from metaculus_bot.structured_output_schema import (
    BinaryStructured,
    DiscreteCountStructured,
    EvidenceItem,
    MultipleChoiceStructured,
    NumericStructured,
    ScenarioBranch,
    StatedBaseRate,
    StatedHazard,
    StatedPrior,
)


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
