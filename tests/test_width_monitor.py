"""Tests for the era-bucketed numeric width / calibration monitor.

The coverage math is verified against hand-computed values on synthetic
records with linear CDFs (so PIT = (resolution - lower) / (upper - lower)).
"""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.performance_analysis.width_monitor import (
    KNOWN_BUG_QIDS,
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
    question_id: object = None,
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
        "question_id": question_id,
    }


def _record_with_pit(pit: float, **kwargs) -> dict:
    """A record whose PIT is exactly ``pit`` (identity ramp over [0, 100])."""
    return _linear_cdf_record(resolution=pit * 100.0, **kwargs)


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
        """Bucketing at literal wall-clock instants either side of each boundary.

        These used ``WIDENING_FLIP.isoformat()`` / ``TS_ANCHOR_ENABLE.isoformat()``
        as inputs, which fed each constant back to itself and made the assertion
        pass for any value the constants happened to hold (2099-01-01 included).
        That is how the wrong TS-anchor boundary survived a green suite. Literal
        timestamps here; the constants' own values are pinned to their merge
        commits in ``TestEraBoundariesAreMergeDates``.
        """
        eras = default_eras()
        assert assign_era({"bot_comment_created_at": "2026-05-11T23:59:59Z"}, eras) == "widening_on (k_tail=1.25)"
        assert assign_era({"bot_comment_created_at": "2026-05-18T17:21:19Z"}, eras) == "widening_off (k_tail=1.0)"
        assert assign_era({"bot_comment_created_at": "2026-07-01T00:00:00Z"}, eras) == "widening_off (k_tail=1.0)"
        assert assign_era({"bot_comment_created_at": "2026-07-21T17:07:37Z"}, eras) == "ts_anchor (sharpen)"
        assert assign_era({"bot_comment_created_at": "2026-08-01T00:00:00Z"}, eras) == "ts_anchor (sharpen)"

    def test_boundary_instant_is_half_open(self):
        """``[start, end)``: the boundary instant itself belongs to the LATER era,
        and one microsecond earlier to the earlier one.

        Value-independent by construction — it asserts the interval convention,
        not the dates — so it is deliberately kept separate from the assertions
        that pin the dates themselves.
        """
        eras = default_eras()
        for boundary, earlier_label, later_label in (
            (WIDENING_FLIP, "widening_on (k_tail=1.25)", "widening_off (k_tail=1.0)"),
            (TS_ANCHOR_ENABLE, "widening_off (k_tail=1.0)", "ts_anchor (sharpen)"),
        ):
            assert assign_era({"bot_comment_created_at": boundary.isoformat()}, eras) == later_label
            just_before = boundary - timedelta(microseconds=1)
            assert assign_era({"bot_comment_created_at": just_before.isoformat()}, eras) == earlier_label

    def test_missing_timestamp(self):
        assert assign_era({"bot_comment_created_at": None}, default_eras()) == "no_timestamp"
        assert assign_era({}, default_eras()) == "no_timestamp"


class TestEraBoundariesAreMergeDates:
    """Era boundaries must be MERGE-TO-MAIN timestamps, never authoring dates.

    Prod runs from ``main``, so a config change is live only once its merge
    commit lands there. Every assertion here anchors to a fact the constant
    cannot define — either the merge commit's committer timestamp or the roster
    that the same merge retired — because the pre-existing boundary test fed the
    constant back to itself and therefore passed for any value (including
    2099-01-01).
    """

    def test_boundaries_equal_merge_commit_timestamps(self):
        """Both constants equal the committer date of the merge that landed them.

        Re-derive with ``TZ=UTC git log -1 --date=iso-local --format='%h %cd' <sha>``:

          * ``0e85e1b`` 2026-05-18 17:21:19 +0000 — flipped ``TAIL_WIDEN_K_TAIL``
            1.25 -> 1.0 (confirmed by value across ``0e85e1b^1``/``0e85e1b``).
            Authored ``b8d730f`` 2026-05-12, six days earlier.
          * ``b4e9df0`` 2026-07-21 17:07:37 +0000 — the july15 bundle: TS anchor
            provider + prompt clause + the ``TS_ANCHOR_ENABLED: 'true'`` yaml
            flip, all authored 2026-07-17, four days earlier.
        """
        assert WIDENING_FLIP == datetime(2026, 5, 18, 17, 21, 19, tzinfo=timezone.utc)
        assert TS_ANCHOR_ENABLE == datetime(2026, 7, 21, 17, 7, 37, tzinfo=timezone.utc)

    def test_pre_merge_roster_record_is_not_in_post_merge_era(self):
        """A record that provably ran the retired 6-model roster cannot be in the
        post-``b4e9df0`` era.

        This is qid 44795 verbatim: published 2026-07-17T21:16:47Z, four days
        after the anchor was authored and four days before it reached ``main``.
        Its own comment names ``gpt-5.5``, ``claude-opus-4.6`` and ``grok-4.5``
        — and ``b4e9df0`` dropped the roster from six models to the
        latest-per-vendor triple in the same merge that landed the anchor, so
        that combination is impossible post-merge. The assertion therefore holds
        independently of what value the constant happens to carry.
        """
        record = {
            "bot_comment_created_at": "2026-07-17T21:16:47.573093+00:00",
            "bot_comment": (
                "*Forecaster 1 (gpt-5.6-sol)*: 12.0\n"
                "*Forecaster 2 (gpt-5.5)*: 13.0\n"
                "*Forecaster 3 (claude-opus-4.8)*: 11.5\n"
                "*Forecaster 4 (claude-opus-4.6)*: 12.5\n"
                "*Forecaster 5 (gemini-3.1-pro-preview)*: 12.2\n"
                "*Forecaster 6 (grok-4.5)*: 14.0\n"
            ),
        }
        assert assign_era(record, default_eras()) == "widening_off (k_tail=1.0)"

    @pytest.mark.parametrize(
        "created_at",
        [
            "2026-07-17T03:24:24Z",  # the anchor provider's own authoring instant
            "2026-07-19T12:00:00Z",
            "2026-07-21T00:00:00Z",
            "2026-07-21T17:07:36Z",  # one second before the merge landed
        ],
    )
    def test_july15_gap_window_buckets_pre_anchor(self, created_at):
        """Nothing on ``main`` changed between the 2026-07-12 merge (``f084bf7``)
        and ``b4e9df0``, so every run in the author-to-merge gap used the
        identical pre-anchor config and belongs in ``widening_off``."""
        assert assign_era({"bot_comment_created_at": created_at}, default_eras()) == "widening_off (k_tail=1.0)"

    @pytest.mark.parametrize(
        "created_at",
        [
            "2026-05-12T10:32:02Z",  # b8d730f authoring instant
            "2026-05-15T12:00:00Z",
            "2026-05-18T17:21:18Z",  # one second before 0e85e1b landed
        ],
    )
    def test_widening_gap_window_buckets_pre_flip(self, created_at):
        """Same defect class as the TS-anchor boundary, six days wide. Zero
        resolved records fall in this window today, so it is latent — a future
        backfill recovering May 12-18 records would activate it silently."""
        assert assign_era({"bot_comment_created_at": created_at}, default_eras()) == "widening_on (k_tail=1.25)"


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


class TestExcludeQids:
    """``exclude_qids`` drops named questions from every row, and says so.

    The bug pair 43746/43747 (Minions / Toy Story 5 opening-weekend gross) is
    excluded from every other dimension of the residual analysis; the width
    monitor was the one place that still counted it. Both records are
    PIT-extreme and sit in opposite tails, so leaving them in makes the active
    era read mildly too narrow.
    """

    def test_known_bug_qids_names_the_documented_pair(self):
        assert KNOWN_BUG_QIDS == frozenset({"43746", "43747"})

    def test_default_keeps_every_record(self):
        """Exclusion is opt-in: callers pass the set explicitly."""
        data = [
            _record_with_pit(0.5, created_at="2026-06-01T00:00:00Z"),
            _record_with_pit(0.025, created_at="2026-06-01T00:00:00Z", question_id=43746),
            _record_with_pit(0.975, created_at="2026-06-01T00:00:00Z", question_id=43747),
        ]
        by_label = {m.label: m for m in compute_all_eras(data)}
        assert by_label["widening_off (k_tail=1.0)"].n_pit == 3
        assert by_label["all"].n_pit == 3
        assert by_label["all"].n_excluded == 0

    def test_excluded_qids_drop_from_era_and_all_rows(self):
        """Integer ``question_id`` is the real dataset shape — the collector
        writes ``q["id"]`` straight through — so the match must coerce rather
        than compare an int against a string set and silently no-op."""
        data = [
            _record_with_pit(0.5, created_at="2026-06-01T00:00:00Z"),
            _record_with_pit(0.5, created_at="2026-03-01T00:00:00Z"),
            _record_with_pit(0.025, created_at="2026-06-01T00:00:00Z", question_id=43746),
            _record_with_pit(0.975, created_at="2026-06-01T00:00:00Z", question_id="43747"),
        ]
        by_label = {m.label: m for m in compute_all_eras(data, exclude_qids=KNOWN_BUG_QIDS)}
        assert by_label["widening_off (k_tail=1.0)"].n_pit == 1
        assert by_label["widening_off (k_tail=1.0)"].n_excluded == 2
        assert by_label["all"].n_pit == 2
        assert by_label["all"].n_excluded == 2
        # The untouched era is unaffected and reports no exclusions.
        assert by_label["widening_on (k_tail=1.25)"].n_pit == 1
        assert by_label["widening_on (k_tail=1.25)"].n_excluded == 0

    def test_excluded_count_surfaces_in_rendered_table(self):
        """A silent exclusion is the same failure mode as a silent degradation:
        the reader must be able to see that rows were dropped."""
        data = [
            _record_with_pit(0.5, created_at="2026-06-01T00:00:00Z"),
            _record_with_pit(0.025, created_at="2026-06-01T00:00:00Z", question_id=43746),
        ]
        md = render_markdown(compute_all_eras(data, exclude_qids=KNOWN_BUG_QIDS))
        assert "excl" in md
        # The dropped record is visible as a count, not just absent.
        assert "| 1 | 1 |" in md


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
        assert tight is not None and miscentered is not None
        # The point of the new column: cov80 is IDENTICAL between the two cases,
        # so no cov80-based read can tell them apart.
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
