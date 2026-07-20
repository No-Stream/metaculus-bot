"""End-to-end guard for the 13-percentile standard set (P1..P99).

The set was grown 11 -> 13 (adding P1=0.01 and P99=0.99) so forecasters can
express probability mass BELOW an open lower bound. Before the change, the
lowest anchor was P2.5 and the parser clamped out-of-range values, so a
distribution that mostly lived below an open floor could not be represented —
this is the Minions & Monsters miss (open lower bound $75M, actual gross <$75M).

These tests prove a forecaster supplying all 13 declared percentiles round-trips
through the real PCHIP pipeline to a valid 201-point Metaculus CDF, including the
critical below-open-bound case where the CDF must carry large mass at the floor.
"""

from __future__ import annotations

import numpy as np
import pytest
from forecasting_tools import NumericQuestion
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.numeric.config import EXPECTED_PERCENTILE_COUNT, STANDARD_PERCENTILES
from metaculus_bot.numeric.pchip_processing import create_fallback_numeric_distribution
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles

_MIN_STEP = 5e-5
_MAX_STEP = 0.2


def _question(
    *,
    lower: float,
    upper: float,
    open_lower: bool,
    open_upper: bool,
    cdf_size: int = 201,
) -> NumericQuestion:
    return NumericQuestion(
        question_text="What will the value be?",
        id_of_question=4242,
        id_of_post=4242,
        page_url="https://www.metaculus.com/questions/4242/",
        background_info="",
        resolution_criteria="",
        fine_print="",
        lower_bound=lower,
        upper_bound=upper,
        open_lower_bound=open_lower,
        open_upper_bound=open_upper,
        zero_point=None,
        unit_of_measure="USD",
        cdf_size=cdf_size,
    )


def _declared(values: list[float]) -> list[Percentile]:
    """Build the standard 13-percentile list from 13 strictly-increasing values."""
    assert len(values) == EXPECTED_PERCENTILE_COUNT == 13
    return [Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, values)]


def _assert_valid_metaculus_cdf(cdf: list[Percentile], *, open_lower: bool, open_upper: bool) -> np.ndarray:
    """Assert the 201-point CDF satisfies the Metaculus server-side constraints. Returns the prob array."""
    assert len(cdf) == 201
    probs = np.array([p.percentile for p in cdf], dtype=float)
    steps = np.diff(probs)
    assert np.all(steps >= _MIN_STEP - 1e-10), f"min-step violation: {float(steps.min())}"
    assert np.all(steps <= _MAX_STEP + 1e-9), f"max-step violation: {float(steps.max())}"
    if open_lower:
        assert probs[0] >= 0.001 - 1e-12
    else:
        assert probs[0] == pytest.approx(0.0, abs=1e-9)
    if open_upper:
        assert probs[-1] <= 0.999 + 1e-12
    else:
        assert probs[-1] == pytest.approx(1.0, abs=1e-9)
    return probs


class TestThirteenPercentileRoundTrip:
    def test_in_range_closed_bounds_produces_valid_cdf(self):
        """A normal in-range 13-percentile forecast on closed bounds → valid CDF pinned to [0, 1]."""
        question = _question(lower=0.0, upper=100.0, open_lower=False, open_upper=False)
        declared = _declared([5, 8, 12, 18, 28, 42, 50, 58, 72, 82, 88, 92, 96])

        sanitized, zero_point = sanitize_percentiles(declared, question)
        distribution = build_numeric_distribution(sanitized, question, zero_point)

        _assert_valid_metaculus_cdf(distribution.cdf, open_lower=False, open_upper=False)

    def test_in_range_open_bounds_produces_valid_cdf(self):
        """A normal in-range 13-percentile forecast on open bounds → valid CDF within [0.001, 0.999]."""
        question = _question(lower=0.0, upper=100.0, open_lower=True, open_upper=True)
        declared = _declared([5, 8, 12, 18, 28, 42, 50, 58, 72, 82, 88, 92, 96])

        sanitized, zero_point = sanitize_percentiles(declared, question)
        distribution = build_numeric_distribution(sanitized, question, zero_point)

        _assert_valid_metaculus_cdf(distribution.cdf, open_lower=True, open_upper=True)

    def test_mass_below_open_lower_bound_produces_large_cdf0(self):
        """The Minions & Monsters case: P1..P60 sit BELOW an open lower floor.

        With the 13-percentile set and no clamping, the pipeline must carry the
        forecaster's below-floor belief into the CDF — cdf[0] (mass at the floor)
        must be large (> 0.5). The old 11-set + clamping could not express this.
        """
        # Open bounds [75M, 150M]; the outcome can resolve far below the floor.
        question = _question(lower=75_000_000.0, upper=150_000_000.0, open_lower=True, open_upper=True)
        # Median (P50=68M) and everything up through P60 (74M) live BELOW the 75M floor;
        # only the upper percentiles (P80..P99) sit inside/above the displayed range.
        declared = _declared(
            [
                30_000_000,  # P1
                35_000_000,  # P2.5
                40_000_000,  # P5
                45_000_000,  # P10
                52_000_000,  # P20
                63_000_000,  # P40
                68_000_000,  # P50 (median below floor)
                74_000_000,  # P60
                90_000_000,  # P80 (inside range)
                110_000_000,  # P90
                130_000_000,  # P95
                145_000_000,  # P97.5
                160_000_000,  # P99 (above range)
            ]
        )

        sanitized, zero_point = sanitize_percentiles(declared, question)
        distribution = build_numeric_distribution(sanitized, question, zero_point)

        cdf = distribution.cdf
        probs = _assert_valid_metaculus_cdf(cdf, open_lower=True, open_upper=True)

        # cdf[0] is the probability mass AT the lower bound (75M) — i.e. the fraction
        # of the distribution at or below the open floor. Since P60 sits below 75M, this
        # must exceed 0.5, proving the out-of-bound mass the old system couldn't express.
        assert cdf[0].value == pytest.approx(75_000_000.0)
        assert probs[0] > 0.5, f"expected large below-floor mass at cdf[0], got {probs[0]}"

    def test_mass_above_open_upper_bound_produces_small_cdf_last(self):
        """The Toy Story 5 case (symmetric): P40..P99 sit ABOVE an open ceiling.

        The actual gross ($159.68M) exceeded the open upper bound ($150M). With the
        13-percentile set and no clamping, the pipeline must carry the forecaster's
        above-ceiling belief into the CDF — the terminal cdf value (mass at/below the
        ceiling) must be well under 1.0, i.e. large above-ceiling tail mass.
        """
        question = _question(lower=75_000_000.0, upper=150_000_000.0, open_lower=True, open_upper=True)
        # Median (P50=165M) and everything from P40 up live ABOVE the 150M ceiling;
        # only the lower percentiles (P1..P20) sit inside/below the displayed range.
        declared = _declared(
            [
                90_000_000,  # P1 (inside range)
                100_000_000,  # P2.5
                110_000_000,  # P5
                125_000_000,  # P10
                140_000_000,  # P20
                158_000_000,  # P40 (above ceiling)
                165_000_000,  # P50 (median above ceiling)
                172_000_000,  # P60
                185_000_000,  # P80
                200_000_000,  # P90
                220_000_000,  # P95
                240_000_000,  # P97.5
                260_000_000,  # P99
            ]
        )

        sanitized, zero_point = sanitize_percentiles(declared, question)
        distribution = build_numeric_distribution(sanitized, question, zero_point)

        cdf = distribution.cdf
        probs = _assert_valid_metaculus_cdf(cdf, open_lower=True, open_upper=True)

        # cdf[-1] is the mass AT the upper bound (150M) — the fraction at or below the
        # open ceiling. Since P40 sits above 150M, this must be well under 0.5, so the
        # above-ceiling tail mass (1 - probs[-1]) is large — what the old system couldn't express.
        assert cdf[-1].value == pytest.approx(150_000_000.0)
        assert probs[-1] < 0.5, f"expected large above-ceiling mass, got cdf[-1]={probs[-1]}"


class TestFallbackCdfRespectsOpenBounds:
    """Regression: the forecasting-tools fallback CDF path must respect open-bound endpoints.

    Its native builder anchors an open lower bound at int(0.5 * percentile_min)%, which
    rounds to 0% once the standard set includes P1 (percentile_min == 1.0 → int(0.5) == 0),
    producing cdf[0] == 0.0. Metaculus rejects open-bound CDFs with cdf[0] < 0.001, so the
    fallback wrapper must re-pin endpoints into the legal range.
    """

    def test_fallback_open_bounds_endpoints_pinned(self):
        question = _question(lower=0.0, upper=100.0, open_lower=True, open_upper=True)
        declared = _declared([5, 8, 12, 18, 28, 42, 50, 58, 72, 82, 88, 92, 96])

        distribution = create_fallback_numeric_distribution(declared, question, zero_point=None)
        cdf = distribution.cdf

        assert len(cdf) == 201
        probs = np.array([p.percentile for p in cdf], dtype=float)
        assert probs[0] >= 0.001
        assert probs[-1] <= 0.999
        assert np.all(np.diff(probs) >= 0.0), "fallback CDF must be monotone non-decreasing"

    def test_fallback_open_lower_min_step_satisfied(self):
        """Regression: with P1 well inside the range, the native builder puts sub-0.001
        probabilities on the leading bins; pinning cdf[0] to 0.001 + cummax then flattens
        them into 0-step bins, tripping the framework's ``assert diff >= 5e-05`` and
        dropping the model's prediction. safe_cdf_bounds must re-enforce the min-step."""
        question = _question(lower=0.0, upper=100.0, open_lower=True, open_upper=False)
        declared = _declared([10, 14, 18, 24, 32, 44, 50, 56, 68, 78, 84, 90, 96])

        distribution = create_fallback_numeric_distribution(declared, question, zero_point=None)
        cdf = distribution.cdf

        assert len(cdf) == 201
        probs = np.array([p.percentile for p in cdf], dtype=float)
        steps = np.diff(probs)
        assert float(steps.min()) >= _MIN_STEP - 1e-10, f"min-step violation: {float(steps.min())}"
        assert probs[0] >= 0.001

    def test_fallback_coarse_grid_uses_grid_scaled_constraints(self):
        """F4 regression: on a coarse discrete grid (cdf_size < 201) the fallback must scale
        its ``safe_cdf_bounds`` constraints to the grid, not use the 201-grid ``max_step=0.2``.

        The native builder emits ``cdf_size`` points, so a concentrated open-bound
        low-count forecast puts well over 0.2 of mass in one bin. With the 201-grid cap the
        fallback would wrongly redistribute that bin down to 0.2; the grid-scaled cap
        (``0.2 * 200 / (cdf_size - 1)``, vacuous at cdf_size=9) must let the peak stand while
        still honouring the grid's min-step and open-bound endpoints."""
        question = _question(lower=-0.5, upper=7.5, open_lower=False, open_upper=True, cdf_size=9)
        declared = _declared([0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.45, 0.9, 1.5, 2.5, 4.0, 6.0])

        distribution = create_fallback_numeric_distribution(declared, question, zero_point=None)
        cdf = distribution.cdf

        assert len(cdf) == question.cdf_size
        probs = np.array([p.percentile for p in cdf], dtype=float)
        steps = np.diff(probs)
        grid_min_step = 0.01 / (question.cdf_size - 1)
        grid_max_step = min(1.0, 0.2 * 200.0 / (question.cdf_size - 1))
        assert float(steps.max()) > _MAX_STEP, "coarse-grid peak must exceed the old 201-grid 0.2 cap"
        assert float(steps.min()) >= grid_min_step - 1e-10, f"grid min-step violation: {float(steps.min())}"
        assert float(steps.max()) <= grid_max_step + 1e-9, f"grid max-step violation: {float(steps.max())}"
        assert bool(np.all(steps > 0.0)), "fallback CDF must be strictly increasing"
        assert probs[-1] <= 0.999 + 1e-12
