"""Tests for the era-bucketed numeric width / calibration monitor.

The coverage math is verified against hand-computed values on synthetic
records with linear CDFs (so PIT = (resolution - lower) / (upper - lower)).
"""

import numpy as np
import pytest

from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.performance_analysis.width_monitor import (
    TS_ANCHOR_ENABLE,
    WIDENING_FLIP,
    _cdf_and_grid,
    _grid_zero_point,
    assign_era,
    compute_all_eras,
    compute_era_metrics,
    compute_pit,
    default_eras,
    jeffreys_ci,
    relative_band_width,
    render_markdown,
)

GRID_N = 201


def _linear_cdf_record(
    *,
    resolution,
    lower: float = 0.0,
    upper: float = 100.0,
    created_at: str | None = "2026-01-01T00:00:00Z",
    q_type: str = "numeric",
) -> dict:
    """A numeric record whose published CDF is the identity ramp on a linear
    grid over ``[lower, upper]``. For such a CDF, F(x) = (x - lower) / (upper -
    lower), so PIT of a resolution ``r`` is exactly ``(r - lower) / (upper -
    lower)`` and the p-quantile value is ``lower + p * (upper - lower)``.
    """
    cdf = np.linspace(0.0, 1.0, GRID_N).tolist()
    return {
        "type": q_type,
        "our_forecast_values": cdf,
        "resolution_parsed": resolution,
        "scaling": {"range_min": lower, "range_max": upper, "zero_point": None},
        "open_lower_bound": True,
        "open_upper_bound": True,
        "bot_comment_created_at": created_at,
    }


class TestPit:
    def test_pit_matches_linear_cdf(self):
        # F(x) = x/100 for the identity ramp, so PIT == resolution/100.
        for res, expected in [(10.0, 0.10), (25.0, 0.25), (50.0, 0.50), (90.0, 0.90)]:
            rec = _linear_cdf_record(resolution=res)
            assert compute_pit(rec) == pytest.approx(expected, abs=1e-9)

    def test_pit_out_of_bounds(self):
        assert compute_pit(_linear_cdf_record(resolution="below_lower_bound")) == 0.0
        assert compute_pit(_linear_cdf_record(resolution="above_upper_bound")) == 1.0

    def test_pit_none_when_unscorable(self):
        # Missing bounds -> can't build a grid.
        rec = _linear_cdf_record(resolution=50.0)
        rec["scaling"] = {}
        assert compute_pit(rec) is None
        # Non-numeric, non-OOB resolution.
        assert compute_pit(_linear_cdf_record(resolution="annulled")) is None


class TestRelativeBandWidth:
    def test_linear_cdf_band_width(self):
        # P10=10, P50=50, P90=90 -> (90-10)/|50| = 1.6.
        rec = _linear_cdf_record(resolution=50.0)
        assert relative_band_width(rec) == pytest.approx(1.6, abs=1e-6)

    def test_median_floor_excludes_near_zero(self):
        # A distribution centred on 0 (symmetric about 0) has |P50| ~ 0.
        rec = _linear_cdf_record(resolution=0.0, lower=-50.0, upper=50.0)
        # P50 = -50 + 0.5*100 = 0 -> excluded.
        assert relative_band_width(rec) is None


class TestJeffreysCi:
    def test_posterior_mean(self):
        # Jeffreys(0.5,0.5): a=0.5+k, b=0.5+(n-k); mean=a/(a+b).
        mean, lo, hi = jeffreys_ci(3, 5)
        assert mean == pytest.approx(3.5 / 6.0, abs=1e-9)
        assert lo < mean < hi
        assert 0.0 < lo and hi < 1.0

    def test_all_successes(self):
        mean, lo, hi = jeffreys_ci(10, 10)
        assert mean == pytest.approx(10.5 / 11.0, abs=1e-9)
        assert lo < mean <= hi


class TestEraAssignment:
    def test_boundaries(self):
        eras = default_eras()
        # Strictly before the widening flip -> widening_on.
        assert assign_era({"bot_comment_created_at": "2026-05-11T23:59:59Z"}, eras) == "widening_on (k_tail=1.25)"
        # On the flip instant -> widening_off (half-open [start, end)).
        assert assign_era({"bot_comment_created_at": WIDENING_FLIP.isoformat()}, eras) == "widening_off (k_tail=1.0)"
        # Between the two flips -> widening_off.
        assert assign_era({"bot_comment_created_at": "2026-07-01T00:00:00Z"}, eras) == "widening_off (k_tail=1.0)"
        # On/after the TS-anchor enable -> ts_anchor.
        assert assign_era({"bot_comment_created_at": TS_ANCHOR_ENABLE.isoformat()}, eras) == "ts_anchor (sharpen)"

    def test_missing_timestamp(self):
        assert assign_era({"bot_comment_created_at": None}, default_eras()) == "no_timestamp"
        assert assign_era({}, default_eras()) == "no_timestamp"


class TestEraMetrics:
    def test_coverage_counts_hand_computed(self):
        # PITs = [0.05, 0.15, 0.50, 0.85, 0.95] via resolutions [5,15,50,85,95].
        recs = [_linear_cdf_record(resolution=r) for r in (5.0, 15.0, 50.0, 85.0, 95.0)]
        m = compute_era_metrics("test", recs)
        assert m is not None
        assert m.n_pit == 5
        # cov80: PIT in [0.10, 0.90] -> {0.15, 0.50, 0.85} = 3/5.
        assert m.cov80[0] == pytest.approx(jeffreys_ci(3, 5)[0], abs=1e-9)
        # cov50: PIT in [0.25, 0.75] -> {0.50} = 1/5.
        assert m.cov50[0] == pytest.approx(jeffreys_ci(1, 5)[0], abs=1e-9)
        # cov@10: PIT <= 0.10 -> {0.05} = 1/5 = 0.20.
        assert m.cov_at_10 == pytest.approx(0.20, abs=1e-9)
        # cov@50: PIT <= 0.50 -> {0.05,0.15,0.50} = 3/5 = 0.60.
        assert m.cov_at_50 == pytest.approx(0.60, abs=1e-9)
        # cov@90: PIT <= 0.90 -> {0.05,0.15,0.50,0.85} = 4/5 = 0.80.
        assert m.cov_at_90 == pytest.approx(0.80, abs=1e-9)
        # mean PIT = 2.5/5 = 0.50; std = population std of the five PITs.
        pits = np.array([0.05, 0.15, 0.50, 0.85, 0.95])
        assert m.mean_pit == pytest.approx(0.50, abs=1e-9)
        assert m.pit_std == pytest.approx(pits.std(), abs=1e-9)
        # median rel width: all identical linear CDFs -> 1.6.
        assert m.median_rel_width == pytest.approx(1.6, abs=1e-6)
        assert m.n_width == 5

    def test_oob_counts(self):
        recs = [
            _linear_cdf_record(resolution="below_lower_bound"),
            _linear_cdf_record(resolution=50.0),
            _linear_cdf_record(resolution="above_upper_bound"),
        ]
        m = compute_era_metrics("test", recs)
        assert m is not None
        assert m.n_oob_low == 1
        assert m.n_oob_high == 1

    def test_returns_none_without_numeric(self):
        assert compute_era_metrics("empty", [{"type": "binary"}]) is None

    def test_no_post_id_makes_n_eff_equal_n(self):
        # Synthetic records without post_id => each is its own cluster => n_eff == n,
        # so the coverage CIs match the naive jeffreys_ci(cov_k, n) exactly.
        recs = [_linear_cdf_record(resolution=r) for r in (5.0, 15.0, 50.0, 85.0, 95.0)]
        m = compute_era_metrics("test", recs)
        assert m is not None
        assert m.n_eff == m.n_pit == 5
        assert m.cov80 == pytest.approx(jeffreys_ci(3, 5))
        assert m.cov50 == pytest.approx(jeffreys_ci(1, 5))


class TestEraMetricsClustering:
    """F3: coverage CIs use n_eff (distinct post_ids), not the raw question count.
    Correlated question families (multiple sub-questions per post) otherwise make
    the Jeffreys CI too narrow."""

    def test_six_questions_two_posts_widen_ci_to_n_eff(self):
        # 6 questions across 2 posts; all PITs land inside [0.10, 0.90] so cov80_k=6.
        recs = []
        for i, res in enumerate((20.0, 30.0, 40.0, 60.0, 70.0, 80.0)):
            rec = _linear_cdf_record(resolution=res)
            rec["post_id"] = 1 if i < 3 else 2  # 3 sub-questions per post
            recs.append(rec)
        m = compute_era_metrics("test", recs)
        assert m is not None
        assert m.n_pit == 6
        assert m.n_eff == 2
        # Point estimate uses round(cov_k * n_eff / n) over n_eff: round(6*2/6)=2 of 2.
        assert m.cov80 == pytest.approx(jeffreys_ci(2, 2))
        # The clustered CI is materially WIDER than the naive n=6 CI.
        naive_lo, naive_hi = jeffreys_ci(6, 6)[1], jeffreys_ci(6, 6)[2]
        _mean, clustered_lo, clustered_hi = m.cov80
        assert (clustered_hi - clustered_lo) > (naive_hi - naive_lo)

    def test_missing_post_id_counts_as_own_cluster(self):
        # Two records share a post; one has no post_id => 2 clusters total.
        recs = [_linear_cdf_record(resolution=50.0) for _ in range(3)]
        recs[0]["post_id"] = 7
        recs[1]["post_id"] = 7
        # recs[2] intentionally has no post_id.
        m = compute_era_metrics("test", recs)
        assert m is not None
        assert m.n_pit == 3
        assert m.n_eff == 2

    def test_n_eff_rendered_in_markdown(self):
        recs = []
        for i, res in enumerate((20.0, 40.0, 60.0, 80.0)):
            rec = _linear_cdf_record(resolution=res)
            rec["post_id"] = 1 if i < 2 else 2
            recs.append(rec)
        md = render_markdown(compute_all_eras(recs))
        assert "n_eff" in md


class TestComputeAllEras:
    def test_buckets_by_era_and_emits_all_row(self):
        data = [
            _linear_cdf_record(resolution=50.0, created_at="2026-03-01T00:00:00Z"),  # widening_on
            _linear_cdf_record(resolution=50.0, created_at="2026-03-15T00:00:00Z"),  # widening_on
            _linear_cdf_record(resolution=50.0, created_at="2026-06-01T00:00:00Z"),  # widening_off
            {"type": "binary", "our_prob_yes": 0.5},  # ignored: not numeric
        ]
        metrics = compute_all_eras(data)
        by_label = {m.label: m for m in metrics}
        assert by_label["widening_on (k_tail=1.25)"].n_pit == 2
        assert by_label["widening_off (k_tail=1.0)"].n_pit == 1
        assert by_label["all"].n_pit == 3
        # ts_anchor era has no records -> omitted from output.
        assert "ts_anchor (sharpen)" not in by_label

    def test_render_markdown_smoke(self):
        data = [_linear_cdf_record(resolution=r) for r in (10.0, 50.0, 90.0)]
        md = render_markdown(compute_all_eras(data))
        assert "Numeric width / calibration monitor" in md
        assert "cov@10" in md
        assert "| all |" in md


class TestLogScaleGrid:
    """Regression: a log-scale question serializes ``zero_point == 0`` with a
    positive ``range_min`` (the geometric grid uses ``ratio = range_max /
    range_min``). The old code treated 0 as the linear sentinel and rebuilt a
    linear grid, corrupting the value grid by up to ~0.55 span-normalized on
    9 real questions in the 2026-07-18 width audit. The fix (a) prefers the
    API's grid-exact ``continuous_range`` when present, (b) otherwise
    reconstructs the geometric grid instead of a linear one.
    """

    def test_grid_zero_point_treats_zero_as_log_when_range_min_positive(self):
        # zero_point==0 with a positive floor => genuine log scale (keep 0.0).
        assert _grid_zero_point(0, 100.0) == 0.0
        assert _grid_zero_point(0.0, 100.0) == 0.0
        # zero_point==0 with a non-positive floor can't be a log transform => drop.
        assert _grid_zero_point(0, 0.0) is None
        assert _grid_zero_point(0, -5.0) is None
        # A genuinely-absent zero_point is linear.
        assert _grid_zero_point(None, 100.0) is None
        # A real (nonzero) zero_point is passed through.
        assert _grid_zero_point(50, 100.0) == 50.0

    def test_reconstructed_grid_is_geometric_for_zero_point_zero(self):
        # No continuous_range in the record -> _cdf_and_grid must reconstruct the
        # GEOMETRIC grid (ratio = range_max / range_min), not a linear ramp.
        lower, upper = 100.0, 1000.0
        cdf = np.linspace(0.0, 1.0, GRID_N).tolist()
        rec = {
            "type": "numeric",
            "our_forecast_values": cdf,
            "resolution_parsed": 500.0,
            "scaling": {"range_min": lower, "range_max": upper, "zero_point": 0},
        }
        built = _cdf_and_grid(rec)
        assert built is not None
        _cdf_arr, grid = built
        expected_geometric = build_cdf_value_grid(lower, upper, 0.0, GRID_N)
        expected_linear = build_cdf_value_grid(lower, upper, None, GRID_N)
        # Matches the geometric grid, and is materially different from linear.
        np.testing.assert_allclose(grid, expected_geometric, rtol=0, atol=1e-9)
        assert float(np.max(np.abs(expected_geometric - expected_linear))) > 1.0

    def test_continuous_range_is_preferred_when_present(self):
        # When the API grid is present it is used verbatim (already log/linear
        # correct), regardless of the zero_point sentinel.
        lower, upper = 100.0, 1000.0
        api_grid = build_cdf_value_grid(lower, upper, 0.0, GRID_N)
        cdf = np.linspace(0.0, 1.0, GRID_N).tolist()
        rec = {
            "type": "numeric",
            "our_forecast_values": cdf,
            "resolution_parsed": 500.0,
            "scaling": {
                "range_min": lower,
                "range_max": upper,
                "zero_point": 0,
                "continuous_range": api_grid.tolist(),
            },
        }
        built = _cdf_and_grid(rec)
        assert built is not None
        _cdf_arr, grid = built
        np.testing.assert_array_equal(grid, api_grid)
