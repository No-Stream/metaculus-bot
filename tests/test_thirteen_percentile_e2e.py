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
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles

_MIN_STEP = 5e-5
_MAX_STEP = 0.2


def _question(
    *,
    lower: float,
    upper: float,
    open_lower: bool,
    open_upper: bool,
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


class TestFallbackCdfRespectsOpenBounds:
    """Regression: the forecasting-tools fallback CDF path must respect open-bound endpoints.

    Its native builder anchors an open lower bound at int(0.5 * percentile_min)%, which
    rounds to 0% once the standard set includes P1 (percentile_min == 1.0 → int(0.5) == 0),
    producing cdf[0] == 0.0. Metaculus rejects open-bound CDFs with cdf[0] < 0.001, so the
    fallback wrapper must re-pin endpoints into the legal range.
    """

    def test_fallback_open_bounds_endpoints_pinned(self):
        from metaculus_bot.numeric.pchip_processing import create_fallback_numeric_distribution

        question = _question(lower=0.0, upper=100.0, open_lower=True, open_upper=True)
        declared = _declared([5, 8, 12, 18, 28, 42, 50, 58, 72, 82, 88, 92, 96])

        distribution = create_fallback_numeric_distribution(declared, question, zero_point=None)
        cdf = distribution.cdf

        assert len(cdf) == 201
        probs = np.array([p.percentile for p in cdf], dtype=float)
        assert probs[0] >= 0.001
        assert probs[-1] <= 0.999
        assert np.all(np.diff(probs) >= 0.0), "fallback CDF must be monotone non-decreasing"
