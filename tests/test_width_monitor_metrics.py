"""The per-era metrics and how a row renders: coverage, band width, the Jeffreys CI and its
cluster correction, the underpowered-row disclosure and the band-miss tail split.
"""

import numpy as np
import pytest

from metaculus_bot.performance_analysis.width_monitor import (
    MIN_N_FOR_POINT_METRICS,
    compute_all_eras,
    compute_era_metrics,
    jeffreys_ci,
    relative_band_width,
    render_markdown,
)
from tests.width_monitor_fakes import (
    _below_bound_mass_record,
    _linear_cdf_record,
    _record_with_pit,
    _row_cells,
)


class TestRelativeBandWidth:
    def test_linear_cdf_band_width(self):
        """P10=10, P50=50, P90=90 on the identity ramp, so the width is (90-10)/|50| = 1.6."""
        rec = _linear_cdf_record(resolution=50.0)
        assert relative_band_width(rec) == pytest.approx(1.6, abs=1e-6)

    def test_median_floor_excludes_near_zero(self):
        """A distribution centred on 0 (symmetric about 0) has |P50| ~ 0, so it is excluded."""
        rec = _linear_cdf_record(resolution=0.0, lower=-50.0, upper=50.0)
        # P50 = -50 + 0.5*100 = 0 -> excluded.
        assert relative_band_width(rec) is None


class TestJeffreysCi:
    def test_posterior_mean(self):
        """Jeffreys(0.5, 0.5): a = 0.5 + k, b = 0.5 + (n - k), mean = a/(a+b)."""
        mean, lo, hi = jeffreys_ci(3, 5)
        assert mean == pytest.approx(3.5 / 6.0, abs=1e-9)
        assert lo < mean < hi
        assert lo > 0.0
        assert hi < 1.0

    def test_all_successes(self):
        mean, lo, hi = jeffreys_ci(10, 10)
        assert mean == pytest.approx(10.5 / 11.0, abs=1e-9)
        assert lo < mean <= hi


class TestEraMetrics:
    def test_coverage_counts_hand_computed(self):
        """Hand-computed counts for the PITs [0.05, 0.15, 0.50, 0.85, 0.95].

        Those come from resolutions 5, 15, 50, 85 and 95 on the identity ramp.
        """
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

    def test_oob_counts_numeric_out_of_grid_resolution(self):
        """A NUMERIC resolution beyond the grid counts as OOB even with a PIT off the bounds.

        Its PIT comes off the declared-percentile curves rather than being pinned at 0.0/1.0, and
        the pre-fix counters tested PIT == 0.0/1.0 and read 0/0 on exactly this shape.
        """
        recs = [
            _below_bound_mass_record(
                resolution=50.0,
                per_model_percentiles={"model-a": [[10.0, 80.0], [50.0, 90.0], [90.0, 105.0]]},
            ),
            _linear_cdf_record(resolution=50.0),
        ]
        m = compute_era_metrics("test", recs)
        assert m is not None
        assert m.n_pit == 2
        assert m.n_oob_low == 1
        assert m.n_oob_high == 0

    def test_in_grid_pit_of_zero_is_not_counted_oob(self):
        """A closed-bound resolution AT the minimum has PIT exactly 0.0 but is not out of grid.

        The old value-equality counter miscounted this one as OOB.
        """
        m = compute_era_metrics("test", [_linear_cdf_record(resolution=0.0)])
        assert m is not None
        assert m.n_oob_low == 0

    def test_returns_none_without_numeric(self):
        assert compute_era_metrics("empty", [{"type": "binary"}]) is None

    def test_no_post_id_makes_n_eff_equal_n(self):
        """Synthetic records without post_id are each their own cluster, so n_eff == n.

        The coverage CIs then match the naive ``jeffreys_ci(cov_k, n)`` exactly.
        """
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
        """6 questions across 2 posts, with every PIT inside [0.10, 0.90] so cov80_k = 6."""
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
        """Two records share a post and the third carries no post_id, so 2 clusters total."""
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


class TestClusterCorrectionDisclosure:
    """The cluster correction is inert on every archived dataset (one record per
    post in all five pulls), while the legend used to tell the operator the CIs
    had been cluster-widened and a code comment claimed "~62% of records share a
    post" — a figure no dataset supports. The mechanism stays (a group post can
    still resolve); the table now says per row whether it fired."""

    def test_one_record_per_post_is_marked_inert(self):
        recs = []
        for i, res in enumerate((20.0, 40.0, 60.0, 80.0)):
            rec = _linear_cdf_record(resolution=res)
            rec["post_id"] = 100 + i
            recs.append(rec)
        [m] = [row for row in compute_all_eras(recs) if row.label == "all"]
        assert m.ci_clustered is False
        assert m.cov80 == pytest.approx(jeffreys_ci(4, 4))  # identical to the naive CI
        md = render_markdown(compute_all_eras(recs))
        assert _row_cells(md, "all")[4] == f"{m.n_eff} (=n)"

    def test_multi_record_post_is_marked_widened(self):
        recs = []
        for i, res in enumerate((20.0, 30.0, 40.0, 60.0, 70.0, 80.0)):
            rec = _linear_cdf_record(resolution=res)
            rec["post_id"] = 1 if i < 3 else 2
            recs.append(rec)
        [m] = [row for row in compute_all_eras(recs) if row.label == "all"]
        assert m.ci_clustered is True
        md = render_markdown(compute_all_eras(recs))
        assert _row_cells(md, "all")[4] == "2 (widened)"

    def test_legend_states_the_marker_convention_rather_than_asserting_widening(self):
        md = render_markdown(compute_all_eras([_record_with_pit(0.5)]))
        assert "`(widened)`" in md
        assert "`(=n)`" in md
        assert "Every archived pull to date is `(=n)`." in md

    def test_serialized_row_carries_the_flag(self):
        m = compute_era_metrics("test", [_record_with_pit(0.5)])
        assert m is not None
        assert m.to_dict()["ci_clustered"] is False


class TestUnderpoweredPointMetrics:
    """cov@k / PIT std / mean PIT / band_miss carry no CI, so at small n they read
    as estimates while their resolution (1/n) is coarser than the target they are
    compared against. The worst case shipped: pit_std == 0.0 at n=1, which reads
    as "maximally too WIDE"."""

    def test_single_record_row_renders_na_not_a_zero_pit_std(self):
        metrics = compute_all_eras([_record_with_pit(0.5)])
        md = render_markdown(metrics)
        cells = _row_cells(md, "all")
        # cov@10, cov@50, cov@90, PIT std, mean PIT, then band_miss.
        assert cells[7:12] == ["n/a"] * 5
        assert cells[13] == "n/a"
        # The CI columns still render: their width is the honest small-n signal.
        assert cells[5].startswith("0.")

    def test_underpowered_flag_and_raw_values_survive_in_json(self):
        m = compute_era_metrics("test", [_record_with_pit(0.5)])
        assert m is not None
        assert m.underpowered is True
        d = m.to_dict()
        assert d["underpowered"] is True
        assert d["pit_std"] == pytest.approx(0.0)  # kept for scripts, hidden from readers

    def test_row_at_the_threshold_renders_numbers(self):
        recs = [_record_with_pit(p) for p in np.linspace(0.05, 0.95, MIN_N_FOR_POINT_METRICS)]
        m = compute_era_metrics("test", recs)
        assert m is not None
        assert m.underpowered is False
        cells = _row_cells(render_markdown(compute_all_eras(recs)), "all")
        assert cells[10] == f"{m.pit_std:.3f}"
        assert "n/a" not in cells[7:12]


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


class TestBandMissSplit:
    """``band_miss`` splits the out-of-band rate into tails, which separates a
    band that is too TIGHT from one that is the right width but MIS-CENTERED.
    ``cov80`` alone cannot express that distinction.
    """

    @staticmethod
    def _records(n_low: int, n_high: int, n_inside: int) -> list[dict]:
        pits = [0.05] * n_low + [0.95] * n_high + [0.50] * n_inside
        return [_record_with_pit(p) for p in pits]

    def test_band_miss_equals_one_minus_raw_cov80(self):
        m = compute_era_metrics("test", self._records(n_low=1, n_high=1, n_inside=18))
        assert m is not None
        assert m.band_miss == pytest.approx(0.10, abs=1e-9)
        assert m.band_lo == pytest.approx(0.05, abs=1e-9)
        assert m.band_hi == pytest.approx(0.05, abs=1e-9)

    def test_split_discriminates_tight_from_miscentered_at_identical_cov80(self):
        tight = compute_era_metrics("tight", self._records(n_low=3, n_high=3, n_inside=14))
        miscentered = compute_era_metrics("miscentered", self._records(n_low=0, n_high=6, n_inside=14))
        assert tight is not None
        assert miscentered is not None
        # cov80 is IDENTICAL between the two cases, so no cov80-based read can tell them apart.
        assert tight.cov80 == pytest.approx(miscentered.cov80)
        assert tight.band_miss == pytest.approx(miscentered.band_miss, abs=1e-9)
        assert tight.band_miss == pytest.approx(0.30, abs=1e-9)
        # The tails do tell them apart: symmetric misses vs. all-high misses.
        assert tight.band_lo == pytest.approx(tight.band_hi, abs=1e-9)
        assert miscentered.band_lo == pytest.approx(0.0, abs=1e-9)
        assert miscentered.band_hi == pytest.approx(0.30, abs=1e-9)

    def test_band_miss_rendered_and_serialized(self):
        data = [_record_with_pit(p) for p in (0.05, 0.50, 0.95)]
        metrics = compute_all_eras(data)
        md = render_markdown(metrics)
        assert "band_miss" in md
        d = metrics[0].to_dict()
        assert {"band_miss", "band_lo", "band_hi"} <= set(d)
