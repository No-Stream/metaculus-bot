"""Fine-grid (``cdf_size > 201``) behaviour of the numeric CDF pipeline.

Mantic quantitative questions run up to 2,000 bins (2,001-point CDFs) and a live preseason
question has ``inbound_outcome_count = 450`` (451 points). Metaculus grids are 201 points or
fewer, so nothing above 201 had ever been exercised. Two things are pinned here.

1. ``grid_step_constraints`` reproduces the server's per-bin formulas exactly
   (``round(0.01 / N, 9)`` min-step, ``0.2 * 200 / N`` max-step, ``N = cdf_size - 1``), with
   no floor. The retired ``5e-5`` floor was a no-op for ``N <= 200`` and STRICTER than the
   server above it: at 450 bins it demanded ``5e-5`` per step where the server demands
   ``2.22e-5``, forcing ~2.25% of total mass into a uniform mixture instead of 1%; at 2,000
   bins it forced 10% instead of 1%. That distorted the tails of every fine-grid forecast.
2. The per-model build (``sanitize_percentiles`` -> ``build_numeric_distribution``, the
   runner path) and the ensemble aggregation (``aggregate_numeric``, the aggregation-pipeline
   path) publish a CDF at 451 and 2,001 points that the server accepts after its own
   rounding, and whose open-bound outer tails carry only the mass the server's min-step
   forces, below what the old floor forced.

The server check is replicated verbatim from the open-source Metaculus backend
(``questions/serializers/common.py``), which Mantic forked with identical constants.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import NumericQuestion
from scipy.stats import norm

from metaculus_bot.constants import NUM_MIN_PROB_STEP
from metaculus_bot.numeric.config import (
    MAX_CDF_PROB_STEP,
    MIN_CDF_PROB_STEP,
    PCHIP_CDF_POINTS,
    STANDARD_PERCENTILES,
    grid_step_constraints,
)
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric.utils import aggregate_numeric


def _server_min_step(inbound: int) -> float:
    return round(0.01 / inbound, 9)


def _server_max_step(inbound: int) -> float:
    return 0.2 * 200 / inbound


def _assert_server_accepts(probs: np.ndarray, *, cdf_size: int, open_lower: bool, open_upper: bool) -> None:
    """Fail exactly where ``questions/serializers/common.py`` would reject the submission."""
    inbound = cdf_size - 1
    assert len(probs) == inbound + 1, f"len(continuous_cdf)={len(probs)} != inbound_outcome_count + 1={inbound + 1}"
    assert not np.any(np.isnan(probs))

    rounded = np.round(probs, 10)
    pmf = np.round(np.diff(rounded), 9)
    min_diff = _server_min_step(inbound)
    max_diff = _server_max_step(inbound)
    assert np.all(pmf >= min_diff), f"step below server min {min_diff}: min pmf {pmf.min()} at {int(np.argmin(pmf))}"
    assert np.all(pmf <= max_diff), f"step above server max {max_diff}: max pmf {pmf.max()} at {int(np.argmax(pmf))}"

    if open_lower:
        assert rounded[0] >= 0.001, f"open lower bound cdf[0]={rounded[0]} < 0.001"
    else:
        assert rounded[0] == 0.0, f"closed lower bound cdf[0]={rounded[0]} != 0.0"
    if open_upper:
        assert rounded[-1] <= 0.999, f"open upper bound cdf[-1]={rounded[-1]} > 0.999"
    else:
        assert rounded[-1] == 1.0, f"closed upper bound cdf[-1]={rounded[-1]} != 1.0"


class TestGridStepConstraintsMatchServer:
    @pytest.mark.parametrize("num_points", [9, 13, 51, 201, 451, 2001])
    def test_exact_server_formulas(self, num_points: int) -> None:
        inbound = num_points - 1
        min_step, max_step = grid_step_constraints(num_points)
        assert min_step == _server_min_step(inbound)
        assert max_step == min(1.0, _server_max_step(inbound))

    def test_min_step_carries_the_server_rounding(self) -> None:
        """The server compares ``round(diff, 9) >= round(0.01 / N, 9)``.

        An unrounded ``0.01 / 450`` (2.2222222222e-5) would be a stricter floor than the
        server's 2.2222e-5, so the rounding is part of the contract.
        """
        assert grid_step_constraints(451)[0] == 2.2222e-05
        assert grid_step_constraints(13)[0] == 0.000833333
        assert grid_step_constraints(2001)[0] == 5e-06

    @pytest.mark.parametrize("num_points", [451, 2001])
    def test_fine_grid_min_step_is_below_the_201_grid_value(self, num_points: int) -> None:
        """No floor: a fine grid demands LESS per bin than the 201-point grid, as the server does."""
        min_step, max_step = grid_step_constraints(num_points)
        assert min_step < NUM_MIN_PROB_STEP
        assert max_step < MAX_CDF_PROB_STEP

    def test_standard_grid_is_unchanged(self) -> None:
        assert grid_step_constraints(PCHIP_CDF_POINTS) == (MIN_CDF_PROB_STEP, MAX_CDF_PROB_STEP)
        assert MIN_CDF_PROB_STEP == NUM_MIN_PROB_STEP == 5e-5


class _GridSpec(NamedTuple):
    """One fine-grid question plus the ``(mean, sd)`` of three forecasters that broadly agree.

    Every declared percentile sits inside the displayed range and well clear of both edges,
    so on an open side the outermost bins carry no declared mass.
    """

    cdf_size: int
    lower_bound: float
    upper_bound: float
    members: tuple[tuple[float, float], ...]


# The live Mantic preseason bitcoin question: 450 bins over $54,950 to $99,950.
_BITCOIN_451 = _GridSpec(451, 54_950.0, 99_950.0, ((78_000.0, 6_000.0), (80_000.0, 7_000.0), (76_500.0, 5_500.0)))
# A 2,000-bin count over [0, 2000], the largest grid Mantic issues.
_COUNT_2001 = _GridSpec(2001, 0.0, 2000.0, ((1_000.0, 120.0), (1_040.0, 140.0), (960.0, 110.0)))

_GRID_CASES = [
    pytest.param(_BITCOIN_451, True, True, id="bitcoin-451-open-open"),
    pytest.param(_BITCOIN_451, False, False, id="bitcoin-451-closed-closed"),
    pytest.param(_COUNT_2001, True, True, id="count-2001-open-open"),
    pytest.param(_COUNT_2001, False, False, id="count-2001-closed-closed"),
    pytest.param(_COUNT_2001, False, True, id="count-2001-closed-open"),
]


def _question(spec: _GridSpec, *, open_lower: bool, open_upper: bool) -> NumericQuestion:
    return NumericQuestion(
        id_of_question=650,
        id_of_post=650,
        page_url="https://competitions.mantic.com/questions/650",
        question_text="Fine-grid quantitative question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=spec.lower_bound,
        upper_bound=spec.upper_bound,
        open_lower_bound=open_lower,
        open_upper_bound=open_upper,
        unit_of_measure="",
        zero_point=None,
        cdf_size=spec.cdf_size,
    )


def _normal_declaration(mean: float, sd: float) -> list[Percentile]:
    """The 13 standard percentiles of a normal forecast, as a forecaster would declare them."""
    return [Percentile(percentile=p, value=float(norm.ppf(p, loc=mean, scale=sd))) for p in STANDARD_PERCENTILES]


def _build_members(spec: _GridSpec, question: NumericQuestion) -> list[NumericDistribution]:
    """Run each forecaster's declaration through the runner path: sanitize, then build."""
    built: list[NumericDistribution] = []
    for index, (mean, sd) in enumerate(spec.members):
        model_name = f"member-{index}"
        sanitized, zero_point = sanitize_percentiles(_normal_declaration(mean, sd), question, model_name=model_name)
        built.append(build_numeric_distribution(sanitized, question, zero_point, model_name=model_name))
    return built


def _probs(distribution: NumericDistribution) -> np.ndarray:
    return np.asarray([p.percentile for p in distribution.get_cdf()], dtype=float)


def _outer_tail_masses(probs: np.ndarray, fraction: float = 0.05) -> tuple[int, float, float]:
    """(bin count, low-side mass, high-side mass) of the outermost ``fraction`` of bins per side."""
    n_bins = len(probs) - 1
    n_outer = round(fraction * n_bins)
    return n_outer, float(probs[n_outer] - probs[0]), float(probs[-1] - probs[-1 - n_outer])


@pytest.mark.parametrize(("spec", "open_lower", "open_upper"), _GRID_CASES)
class TestFineGridPublishPath:
    def test_each_member_cdf_is_server_valid(self, spec: _GridSpec, open_lower: bool, open_upper: bool) -> None:
        question = _question(spec, open_lower=open_lower, open_upper=open_upper)
        expected_grid = np.linspace(spec.lower_bound, spec.upper_bound, spec.cdf_size)
        for member in _build_members(spec, question):
            _assert_server_accepts(_probs(member), cdf_size=spec.cdf_size, open_lower=open_lower, open_upper=open_upper)
            values = np.asarray([p.value for p in member.get_cdf()], dtype=float)
            np.testing.assert_allclose(values, expected_grid, rtol=0, atol=1e-9)

    @pytest.mark.parametrize("method", ["median", "mean"])
    def test_aggregate_is_server_valid(self, spec: _GridSpec, open_lower: bool, open_upper: bool, method: str) -> None:
        question = _question(spec, open_lower=open_lower, open_upper=open_upper)
        aggregated = aggregate_numeric(_build_members(spec, question), question, method)
        probs = _probs(aggregated)
        _assert_server_accepts(probs, cdf_size=spec.cdf_size, open_lower=open_lower, open_upper=open_upper)

        grid_values = np.linspace(spec.lower_bound, spec.upper_bound, spec.cdf_size)
        aggregate_median = float(np.interp(0.5, probs, grid_values))
        member_medians = sorted(mean for mean, _ in spec.members)
        assert member_medians[0] <= aggregate_median <= member_medians[-1]

    def test_open_tails_carry_only_the_server_floor(self, spec: _GridSpec, open_lower: bool, open_upper: bool) -> None:
        """On an open side the outermost 5% of bins hold no declared mass, so their mass is the floor.

        The retired ``5e-5`` floor would have forced at least ``n_outer * 5e-5`` there (twice the
        server's requirement at 450 bins, ten times at 2,000). The bound is computed from the grid,
        not hardcoded, so it tracks whatever grid the case uses.
        """
        if not (open_lower or open_upper):
            pytest.skip("both bounds closed: PCHIP anchors the CDF at the bound, so the tails carry declared mass")
        question = _question(spec, open_lower=open_lower, open_upper=open_upper)
        probs = _probs(aggregate_numeric(_build_members(spec, question), question, "median"))
        n_outer, low_mass, high_mass = _outer_tail_masses(probs)
        server_min_step = grid_step_constraints(spec.cdf_size)[0]
        old_floor_forced = n_outer * NUM_MIN_PROB_STEP
        new_floor_forced = n_outer * server_min_step
        assert old_floor_forced > new_floor_forced

        if open_lower:
            assert new_floor_forced - 1e-9 <= low_mass < old_floor_forced, (
                f"low tail mass {low_mass} not in [{new_floor_forced}, {old_floor_forced})"
            )
        if open_upper:
            assert new_floor_forced - 1e-9 <= high_mass < old_floor_forced, (
                f"high tail mass {high_mass} not in [{new_floor_forced}, {old_floor_forced})"
            )
