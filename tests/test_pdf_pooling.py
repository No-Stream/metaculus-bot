"""Tests for Vincentization / tail-floor / log-pool numeric-CDF aggregation primitives.

These offline functions are an alternative to the production VERTICAL CDF-averaging
(`metaculus_bot/ablation/run_pdf.py::_aggregate_numeric_predictions`, which averages F(x)
at each grid value and over-disperses bimodal ensembles). A later harness compares them
against median on the tournament log-score metric. Nothing here runs live.
"""

from __future__ import annotations

import numpy as np
from forecasting_tools.data_models.numeric_report import Percentile
from scipy.stats import norm

from metaculus_bot.numeric.config import MIN_CDF_PROB_STEP, PCHIP_CDF_POINTS
from metaculus_bot.probabilistic_tools.pdf_pooling import (
    apply_tail_floor,
    log_pool_cdfs,
    vincentize_cdfs,
)
from tests.conftest import make_mock_numeric_question


def _normal_cdf_on_grid(grid: np.ndarray, mean: float, sd: float) -> list[Percentile]:
    """Build a forecaster CDF (list[Percentile]) for N(mean, sd) sampled on a shared x-grid."""
    probs = norm.cdf(grid, loc=mean, scale=sd)
    return [Percentile(value=float(x), percentile=float(p)) for x, p in zip(grid, probs)]


def _probs(cdf: list[Percentile]) -> np.ndarray:
    return np.array([p.percentile for p in cdf], dtype=float)


def _values(cdf: list[Percentile]) -> np.ndarray:
    return np.array([p.value for p in cdf], dtype=float)


def _interquantile_spread(cdf: list[Percentile], lo: float = 0.1, hi: float = 0.9) -> float:
    """Width of the central (hi - lo) probability interval in x-units (e.g. p10..p90)."""
    probs = _probs(cdf)
    vals = _values(cdf)
    x_lo = float(np.interp(lo, probs, vals))
    x_hi = float(np.interp(hi, probs, vals))
    return x_hi - x_lo


def _vertical_average(cdfs: list[list[Percentile]]) -> list[Percentile]:
    """Reference implementation of the production VERTICAL averaging we contrast against:
    mean of F(x) at each shared grid value (mirrors run_pdf.py)."""
    vals = _values(cdfs[0])
    prob_matrix = np.array([_probs(c) for c in cdfs], dtype=float)
    mean_probs = np.maximum.accumulate(np.clip(prob_matrix.mean(axis=0), 0.0, 1.0))
    return [Percentile(value=float(x), percentile=float(p)) for x, p in zip(vals, mean_probs)]


def _assert_valid_cdf(
    cdf: list[Percentile],
    question,
    *,
    n_points: int = PCHIP_CDF_POINTS,
    min_step: float = MIN_CDF_PROB_STEP,
) -> None:
    """Validity = right length, strictly increasing probs, in [0,1], endpoints respect
    open/closed bounds, min-step satisfied, x-grid matches the question's linspace."""
    assert len(cdf) == n_points
    probs = _probs(cdf)
    vals = _values(cdf)

    expected_grid = np.linspace(float(question.lower_bound), float(question.upper_bound), n_points)
    np.testing.assert_allclose(vals, expected_grid, rtol=0, atol=1e-9)

    assert np.all(probs >= 0.0 - 1e-12)
    assert np.all(probs <= 1.0 + 1e-12)

    steps = np.diff(probs)
    assert np.all(steps >= min_step - 1e-10), f"min-step violated: min diff {steps.min()}"

    if question.open_lower_bound:
        assert probs[0] >= 0.001 - 1e-9
    else:
        assert abs(probs[0]) <= 1e-9
    if question.open_upper_bound:
        assert probs[-1] <= 0.999 + 1e-9
    else:
        assert abs(probs[-1] - 1.0) <= 1e-9


class TestVincentizeCdfs:
    def test_preserves_sharpness_vs_vertical_on_bimodal_ensemble(self):
        # Two confident-but-disagreeing forecasters: one peaked low, one peaked high.
        # Vertical averaging of F(x) smears this into a wide low-information distribution;
        # Vincentization (quantile averaging) keeps the spread near the component sharpness.
        question = make_mock_numeric_question(lower_bound=0.0, upper_bound=100.0)
        grid = np.linspace(0.0, 100.0, PCHIP_CDF_POINTS)
        sharp_sd = 5.0
        low = _normal_cdf_on_grid(grid, mean=25.0, sd=sharp_sd)
        high = _normal_cdf_on_grid(grid, mean=75.0, sd=sharp_sd)

        vincent = vincentize_cdfs([low, high], question, method="mean")
        vertical = _vertical_average([low, high])

        _assert_valid_cdf(vincent, question)

        vincent_spread = _interquantile_spread(vincent)
        vertical_spread = _interquantile_spread(vertical)
        # Vertical averaging over-disperses: its central interval is far wider.
        assert vincent_spread < vertical_spread
        # Vincent's central interval should sit near the average component location (~50),
        # tight relative to the 50-unit gap between the two component means.
        assert vincent_spread < 0.6 * vertical_spread

    def test_identical_forecasters_roundtrip_to_same_cdf(self):
        # Averaging identical quantile functions must return (essentially) the same CDF.
        question = make_mock_numeric_question(lower_bound=0.0, upper_bound=100.0)
        grid = np.linspace(0.0, 100.0, PCHIP_CDF_POINTS)
        one = _normal_cdf_on_grid(grid, mean=50.0, sd=12.0)

        out = vincentize_cdfs([one, one, one], question, method="mean")
        _assert_valid_cdf(out, question)
        # Median crossing (x where F=0.5) should match the original ~50.
        x50 = float(np.interp(0.5, _probs(out), _values(out)))
        assert abs(x50 - 50.0) < 1.0

    def test_median_method_returns_valid_cdf(self):
        question = make_mock_numeric_question(
            lower_bound=-50.0, upper_bound=50.0, open_lower_bound=True, open_upper_bound=True
        )
        grid = np.linspace(-50.0, 50.0, PCHIP_CDF_POINTS)
        cdfs = [
            _normal_cdf_on_grid(grid, mean=-10.0, sd=8.0),
            _normal_cdf_on_grid(grid, mean=0.0, sd=8.0),
            _normal_cdf_on_grid(grid, mean=12.0, sd=8.0),
        ]
        out = vincentize_cdfs(cdfs, question, method="median")
        _assert_valid_cdf(out, question)

    def test_single_forecaster_returns_valid_cdf(self):
        question = make_mock_numeric_question(lower_bound=0.0, upper_bound=100.0)
        grid = np.linspace(0.0, 100.0, PCHIP_CDF_POINTS)
        out = vincentize_cdfs([_normal_cdf_on_grid(grid, mean=40.0, sd=10.0)], question)
        _assert_valid_cdf(out, question)


class TestApplyTailFloor:
    def test_floors_saturated_lower_tail_and_stays_valid(self):
        # A CDF that saturates: almost all mass piled at the low end, near-zero tail mass.
        # apply_tail_floor must lift boundary-bucket mass to >= floor_eps.
        question = make_mock_numeric_question(lower_bound=0.0, upper_bound=100.0)
        n = PCHIP_CDF_POINTS
        # Build a CDF that jumps to ~1.0 almost immediately (saturating low).
        raw = np.clip(np.linspace(0.0, 1.0, n) ** 8, 0.0, 1.0)
        raw[0] = 0.0
        raw[-1] = 1.0
        floor_eps = 1e-3

        out = apply_tail_floor(raw, question, floor_eps=floor_eps)
        out_arr = np.asarray(out, dtype=float)
        steps = np.diff(out_arr)

        # Every interior bucket carries at least floor_eps mass (anti-saturation).
        assert steps.min() >= floor_eps - 1e-9
        # Still a valid CDF.
        assert np.all(np.diff(out_arr) > 0)
        assert out_arr[0] == 0.0
        assert abs(out_arr[-1] - 1.0) < 1e-9

    def test_floors_open_upper_tail_mass(self):
        # Open upper bound: there is PMF mass ABOVE the grid (1 - cdf[-1]).
        # With a CDF that hits the 0.999 cap, the floor still leaves headroom and stays valid.
        question = make_mock_numeric_question(lower_bound=0.0, upper_bound=100.0, open_upper_bound=True)
        n = PCHIP_CDF_POINTS
        grid = np.linspace(0.0, 100.0, n)
        raw = norm.cdf(grid, loc=80.0, scale=6.0)
        floor_eps = 5e-3

        out = np.asarray(apply_tail_floor(raw, question, floor_eps=floor_eps), dtype=float)
        # Open upper => some mass reserved above the grid.
        assert out[-1] <= 0.999 + 1e-9
        # Monotonic + min-step preserved.
        assert np.all(np.diff(out) >= MIN_CDF_PROB_STEP - 1e-10)

    def test_already_diffuse_cdf_barely_changes(self):
        # A near-uniform CDF already exceeds floor_eps everywhere; output stays close.
        question = make_mock_numeric_question(lower_bound=0.0, upper_bound=100.0)
        n = PCHIP_CDF_POINTS
        raw = np.linspace(0.0, 1.0, n)
        floor_eps = 1e-4  # uniform step is 1/200 = 5e-3 >> floor

        out = np.asarray(apply_tail_floor(raw, question, floor_eps=floor_eps), dtype=float)
        np.testing.assert_allclose(out, raw, atol=1e-6)


class TestLogPoolCdfs:
    def test_returns_valid_cdf(self):
        question = make_mock_numeric_question(lower_bound=0.0, upper_bound=100.0)
        grid = np.linspace(0.0, 100.0, PCHIP_CDF_POINTS)
        cdfs = [
            _normal_cdf_on_grid(grid, mean=40.0, sd=10.0),
            _normal_cdf_on_grid(grid, mean=55.0, sd=12.0),
            _normal_cdf_on_grid(grid, mean=50.0, sd=9.0),
        ]
        out = log_pool_cdfs(cdfs, question)
        _assert_valid_cdf(out, question)

    def test_log_pool_sharper_than_linear_on_agreeing_forecasters(self):
        # Geometric (log) pooling of agreeing densities concentrates mass: the pooled
        # central interval should be no wider than the vertical (linear) average.
        question = make_mock_numeric_question(lower_bound=0.0, upper_bound=100.0)
        grid = np.linspace(0.0, 100.0, PCHIP_CDF_POINTS)
        cdfs = [
            _normal_cdf_on_grid(grid, mean=50.0, sd=10.0),
            _normal_cdf_on_grid(grid, mean=50.0, sd=12.0),
            _normal_cdf_on_grid(grid, mean=50.0, sd=11.0),
        ]
        pooled = log_pool_cdfs(cdfs, question)
        vertical = _vertical_average(cdfs)
        assert _interquantile_spread(pooled) <= _interquantile_spread(vertical) + 1e-6

    def test_weights_shift_pool_toward_weighted_forecaster(self):
        # Heavily weighting the high-mean forecaster pulls the median crossing upward.
        question = make_mock_numeric_question(lower_bound=0.0, upper_bound=100.0)
        grid = np.linspace(0.0, 100.0, PCHIP_CDF_POINTS)
        low = _normal_cdf_on_grid(grid, mean=30.0, sd=8.0)
        high = _normal_cdf_on_grid(grid, mean=70.0, sd=8.0)

        balanced = log_pool_cdfs([low, high], question, weights=[1.0, 1.0])
        high_weighted = log_pool_cdfs([low, high], question, weights=[0.1, 0.9])

        x50_balanced = float(np.interp(0.5, _probs(balanced), _values(balanced)))
        x50_high = float(np.interp(0.5, _probs(high_weighted), _values(high_weighted)))
        assert x50_high > x50_balanced
