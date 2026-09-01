"""Tests for the era-bucketed numeric width / calibration monitor.

The coverage math is verified against hand-computed values on synthetic
records with linear CDFs (so PIT = (resolution - lower) / (upper - lower)).
"""

import json
import logging
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.performance_analysis.analysis import B4E9DF0_MERGED_AT, GRID_SCALED_MAX_STEP_MERGED_AT
from metaculus_bot.performance_analysis.cohorts import (
    DEGRADED_RUN_QIDS,
    EXCLUSION_COHORTS,
    KNOWN_BUG_QIDS,
    PARTIAL_DEGRADED_QIDS,
    parse_exclude_qids,
)
from metaculus_bot.performance_analysis.width_monitor import (
    MIN_N_FOR_POINT_METRICS,
    STARVED_OUTER_TAIL_FLOOR_MULTIPLE,
    TS_ANCHOR_ENABLE,
    WIDENING_FLIP,
    OuterTailReading,
    OuterTailVerdict,
    _cdf_and_grid,
    _grid_zero_point,
    assign_era,
    compute_all_eras,
    compute_era_metrics,
    compute_pit,
    compute_pit_reading,
    default_eras,
    jeffreys_ci,
    main,
    measure_outer_tails,
    relative_band_width,
    render_markdown,
    render_starved_outer_tails,
    scan_outer_tails,
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


def _row_cells(md: str, label: str) -> list[str]:
    """The stripped cells of one rendered table row, indexed as in the header.

    1=era, 2=n, 3=excl, 4=n_eff, 5=cov80, 6=cov50, 7=cov@10, 8=cov@50, 9=cov@90,
    10=PIT std, 11=mean PIT, 12=med rel width, 13=band_miss, 14=OOB, 15=set-valued (pt n).
    """
    [row] = [line for line in md.splitlines() if line.startswith(f"| {label} |")]
    return [cell.strip() for cell in row.split("|")]


def _pit_and_side(record: dict) -> tuple[float | None, str | None]:
    """``(point PIT, oob_side)`` for a record, for the point-valued cases below."""
    reading = compute_pit_reading(record)
    return (None, None) if reading is None else (reading.point, reading.oob_side)


class TestPit:
    def test_pit_matches_linear_cdf(self):
        # F(x) = x/100 for the identity ramp, so PIT == resolution/100.
        for res, expected in [(10.0, 0.10), (25.0, 0.25), (50.0, 0.50), (90.0, 0.90)]:
            rec = _linear_cdf_record(resolution=res)
            assert compute_pit(rec) == pytest.approx(expected, abs=1e-9)

    def test_pit_out_of_bounds_degenerates_to_a_point_when_no_mass_is_out_there(self):
        # The identity ramp spans the full [0, 1], so cdf[0] == 0 and cdf[-1] == 1: the
        # out-of-range INTERVAL collapses to the single value the old convention forced,
        # and `compute_pit` still answers it.
        assert compute_pit(_linear_cdf_record(resolution="below_lower_bound")) == 0.0
        assert compute_pit(_linear_cdf_record(resolution="above_upper_bound")) == 1.0

    def test_pit_none_when_unscorable(self):
        # Missing bounds -> can't build a grid.
        rec = _linear_cdf_record(resolution=50.0)
        rec["scaling"] = {}
        assert compute_pit(rec) is None
        # Non-numeric, non-OOB resolution.
        assert compute_pit(_linear_cdf_record(resolution="annulled")) is None


def _out_of_range_mass_record(*, resolution, cdf_start: float = 0.0, cdf_end: float = 1.0, **kwargs) -> dict:
    """A record whose published CDF ramps ``cdf_start -> cdf_end`` over the displayed range.

    ``1 - cdf_end`` is the mass declared ABOVE the displayed ceiling and ``cdf_start`` the
    mass below the floor, which is what an out-of-range resolution's PIT interval is read
    off.
    """
    rec = _linear_cdf_record(resolution=resolution, **kwargs)
    rec["our_forecast_values"] = np.linspace(cdf_start, cdf_end, GRID_N).tolist()
    return rec


class TestSetValuedOutOfRangePit:
    """An out-of-range resolution pins the PIT to a SET, not to 1.0 / 0.0.

    Metaculus reports "beyond the displayed range" as a string, so the resolution VALUE is
    unknown; all that is known is that ``F(resolution)`` lies in ``[cdf[-1], 1]`` (above) or
    ``[0, cdf[0]]`` (below). On an open bound our own CDF is free to put real mass out there:
    q44842 published 13% of its mass above the displayed ceiling, resolved
    ``above_upper_bound`` and won spot peer +24.4, while the old PIT-1.0 convention scored it
    a high-side band miss. The shape here is that record's (``cdf[-1] = 0.87``).
    """

    def test_above_upper_bound_reads_as_the_interval_above_the_cdf_end(self):
        reading = compute_pit_reading(_out_of_range_mass_record(resolution="above_upper_bound", cdf_end=0.87))
        assert reading is not None
        assert reading.is_interval
        assert (reading.low, reading.high) == pytest.approx((0.87, 1.0))
        assert reading.oob_side == "high"
        # There is no point PIT to report, and none is invented.
        assert reading.point is None
        assert compute_pit(_out_of_range_mass_record(resolution="above_upper_bound", cdf_end=0.87)) is None

    def test_below_lower_bound_reads_as_the_interval_below_the_cdf_start(self):
        reading = compute_pit_reading(_out_of_range_mass_record(resolution="below_lower_bound", cdf_start=0.13))
        assert reading is not None
        assert reading.is_interval
        assert (reading.low, reading.high) == pytest.approx((0.0, 0.13))
        assert reading.oob_side == "low"

    def test_the_q44842_shape_counts_as_covered_at_cov80(self):
        # [0.87, 1] intersects [0.10, 0.90], so the record is covered rather than a miss.
        m = compute_era_metrics("test", [_out_of_range_mass_record(resolution="above_upper_bound", cdf_end=0.87)])
        assert m is not None
        assert m.n_pit == 1
        assert m.cov80 == pytest.approx(jeffreys_ci(1, 1))
        assert m.band_hi == pytest.approx(0.0)
        assert m.band_miss == pytest.approx(0.0)
        # cov@90 = P(PIT <= 0.90): the interval reaches below 0.90, so it counts.
        assert m.cov_at_90 == pytest.approx(1.0)
        assert m.cov_at_10 == pytest.approx(0.0)

    def test_a_starved_tail_is_still_a_high_side_band_miss(self):
        # cdf[-1] = 0.999 (the open-bound structural floor): the whole interval sits above
        # 0.90, so the record misses the band exactly as it should.
        m = compute_era_metrics("test", [_out_of_range_mass_record(resolution="above_upper_bound", cdf_end=0.999)])
        assert m is not None
        assert m.cov80 == pytest.approx(jeffreys_ci(0, 1))
        assert m.band_hi == pytest.approx(1.0)
        assert m.band_lo == pytest.approx(0.0)
        assert m.cov_at_90 == pytest.approx(0.0)

    def test_a_starved_low_tail_is_still_a_low_side_band_miss(self):
        m = compute_era_metrics("test", [_out_of_range_mass_record(resolution="below_lower_bound", cdf_start=0.001)])
        assert m is not None
        assert m.cov80 == pytest.approx(jeffreys_ci(0, 1))
        assert m.band_lo == pytest.approx(1.0)
        assert m.cov_at_10 == pytest.approx(1.0)

    def test_the_q44842_low_side_mirror_counts_as_covered(self):
        m = compute_era_metrics("test", [_out_of_range_mass_record(resolution="below_lower_bound", cdf_start=0.13)])
        assert m is not None
        assert m.cov80 == pytest.approx(jeffreys_ci(1, 1))
        assert m.band_lo == pytest.approx(0.0)

    def test_interval_records_are_excluded_from_point_metrics_and_the_count_is_disclosed(self):
        # Nine point PITs spread across the unit interval plus one set-valued record.
        recs = [_record_with_pit(p) for p in np.linspace(0.05, 0.95, MIN_N_FOR_POINT_METRICS)]
        recs.append(_out_of_range_mass_record(resolution="above_upper_bound", cdf_end=0.87))
        m = compute_era_metrics("test", recs)
        assert m is not None
        assert m.n_pit == MIN_N_FOR_POINT_METRICS + 1
        assert m.n_point == MIN_N_FOR_POINT_METRICS
        assert m.n_oob_interval == 1
        # Point statistics see only the ten point readings — an imputed midpoint (0.935)
        # would have pulled both of these.
        points = np.linspace(0.05, 0.95, MIN_N_FOR_POINT_METRICS)
        assert m.mean_pit == pytest.approx(points.mean())
        assert m.pit_std == pytest.approx(points.std())
        # The interval still counts in coverage: 8 of the 10 point PITs are inside
        # [0.10, 0.90] (0.05 and 0.95 are not) and [0.87, 1] intersects the band, so 9 of 11.
        assert m.cov80 == pytest.approx(jeffreys_ci(9, 11))

    def test_an_all_interval_era_reports_no_point_statistics_rather_than_nan(self):
        recs = [_out_of_range_mass_record(resolution="above_upper_bound", cdf_end=0.87)]
        m = compute_era_metrics("test", recs)
        assert m is not None
        assert m.n_point == 0
        assert m.pit_std is None
        assert m.mean_pit is None
        cells = _row_cells(render_markdown([m]), "test")
        assert cells[10] == "n/a"
        assert cells[11] == "n/a"

    def test_the_disclosure_count_is_rendered_and_serialized(self):
        recs = [_record_with_pit(0.5), _out_of_range_mass_record(resolution="above_upper_bound", cdf_end=0.87)]
        metrics = compute_all_eras(recs)
        [m] = [row for row in metrics if row.label == "all"]
        assert m.to_dict()["n_oob_interval"] == 1
        assert m.to_dict()["n_point"] == 1
        md = render_markdown(metrics)
        assert "set-valued" in md
        # Last column: set-valued readings, with the point-metric denominator beside them.
        assert _row_cells(md, "all")[15] == "1 (1)"

    def test_a_numeric_out_of_grid_resolution_stays_a_point_reading(self):
        # Only the STRING markers are set-valued: when the platform gives the value, the
        # members' declared curves read a real quantile off it (see TestOutOfGridPit).
        rec = _below_bound_mass_record(
            resolution=50.0,
            per_model_percentiles={"model-a": [[10.0, 80.0], [50.0, 90.0], [90.0, 105.0]]},
        )
        reading = compute_pit_reading(rec)
        assert reading is not None
        assert not reading.is_interval
        assert reading.point == pytest.approx(0.10, abs=1e-9)
        assert reading.oob_side == "low"


def _below_bound_mass_record(*, resolution, per_model_percentiles=None, **kwargs) -> dict:
    """The q44218 shape: open lower bound with most of the mass declared BELOW it.

    Published CDF ramps 0.90 -> 0.975 over [100, 200], i.e. F(100) = 0.90: 90% of the
    mass sits below the displayed lower bound. A resolution under 100 is therefore a
    LOW-tail event, but grid interpolation clamps it to cdf[0] = 0.90 — the sign flip
    the declared-percentile fallback exists to prevent.
    """
    rec = _linear_cdf_record(resolution=resolution, lower=100.0, upper=200.0, **kwargs)
    rec["our_forecast_values"] = np.linspace(0.90, 0.975, GRID_N).tolist()
    if per_model_percentiles is not None:
        rec["per_model_numeric_percentiles"] = per_model_percentiles
    return rec


class TestOutOfGridPit:
    """A numeric resolution BEYOND the value grid must not be censored at cdf[0]/cdf[-1]."""

    def test_below_grid_resolution_reads_low_tail_not_the_clamp(self):
        # Resolution 50 is below every declared value of every member, so each member
        # curve reads its lowest declared percentile (P10 -> 0.10). The grid clamp
        # would have said 0.90 — the opposite tail.
        rec = _below_bound_mass_record(
            resolution=50.0,
            per_model_percentiles={
                "model-a": [[10.0, 80.0], [50.0, 90.0], [90.0, 105.0]],
                "model-b": [[10.0, 85.0], [50.0, 95.0], [90.0, 110.0]],
            },
        )
        pit, oob_side = _pit_and_side(rec)
        assert oob_side == "low"
        assert pit == pytest.approx(0.10, abs=1e-9)
        assert compute_pit(rec) == pytest.approx(0.10, abs=1e-9)

    def test_fallback_is_median_of_member_curves(self):
        # Resolution 95: model-a interpolates 0.50 + (95-90)/(105-90)*0.40 = 0.6333,
        # model-b reads exactly its P50 = 0.50; the median of the two is their mean.
        rec = _below_bound_mass_record(
            resolution=95.0,
            per_model_percentiles={
                "model-a": [[10.0, 80.0], [50.0, 90.0], [90.0, 105.0]],
                "model-b": [[10.0, 85.0], [50.0, 95.0], [90.0, 110.0]],
            },
        )
        pit, oob_side = _pit_and_side(rec)
        assert oob_side == "low"
        assert pit == pytest.approx((0.6333333 + 0.50) / 2, abs=1e-6)

    def test_above_grid_resolution_reads_high_tail(self):
        rec = _below_bound_mass_record(
            resolution=250.0,
            per_model_percentiles={"model-a": [[10.0, 120.0], [50.0, 150.0], [90.0, 220.0]]},
        )
        pit, oob_side = _pit_and_side(rec)
        assert oob_side == "high"
        assert pit == pytest.approx(0.90, abs=1e-9)

    def test_no_member_curves_keeps_grid_read_but_flags_oob(self):
        # Degraded path (e.g. stacked-era records with no per-model bullets): the
        # grid-endpoint read is kept, but the OOB side still surfaces the record.
        rec = _below_bound_mass_record(resolution=50.0)
        pit, oob_side = _pit_and_side(rec)
        assert oob_side == "low"
        assert pit == pytest.approx(0.90, abs=1e-9)

    def test_in_grid_resolution_has_no_oob_side(self):
        pit, oob_side = _pit_and_side(_linear_cdf_record(resolution=50.0))
        assert oob_side is None
        assert pit == pytest.approx(0.50, abs=1e-9)

    def test_resolution_exactly_at_bound_keeps_endpoint_read(self):
        # AT a bound the clamp IS the correct PIT: F(bound) = cdf[0].
        rec = _below_bound_mass_record(resolution=100.0)
        pit, oob_side = _pit_and_side(rec)
        assert oob_side is None
        assert pit == pytest.approx(0.90, abs=1e-9)


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
        assert lo > 0.0
        assert hi < 1.0

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

    def test_unparseable_timestamp_is_not_attributed_to_an_era(self):
        # A record whose timestamp can't be parsed must land in no_timestamp rather
        # than silently defaulting into the first (or last) era — mis-attributing one
        # record's config era is exactly what the shared parser exists to prevent.
        for raw in ("not-a-date", "2026-13-45T99:00:00Z", ""):
            assert assign_era({"bot_comment_created_at": raw}, default_eras()) == "no_timestamp"

    def test_offset_and_naive_timestamps_land_in_the_same_era(self):
        # The archive carries all three ISO shapes (Z, explicit offset, naive). They
        # must agree, since a naive read of a -07:00 instant is 7 hours off and can
        # cross a boundary.
        eras = default_eras()
        instant_utc = "2026-07-21T18:07:37Z"
        same_instant_offset = "2026-07-21T11:07:37-07:00"
        naive_utc = "2026-07-21T18:07:37"
        labels = {
            assign_era({"bot_comment_created_at": raw}, eras) for raw in (instant_utc, same_instant_offset, naive_utc)
        }
        assert labels == {"ts_anchor (sharpen)"}


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
        assert datetime(2026, 5, 18, 17, 21, 19, tzinfo=UTC) == WIDENING_FLIP
        assert datetime(2026, 7, 21, 17, 7, 37, tzinfo=UTC) == TS_ANCHOR_ENABLE

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

    def test_every_b4e9df0_gate_reads_the_same_instant(self):
        """The monitor's era boundary and the clamp screen's era gate are the SAME
        merge, so they are aliases of one constant.

        Both mark ``b4e9df0``: the era split the width rows are bucketed by, and the
        instant after which a coarse discrete grid's max-step cap stopped being a flat
        0.2. Two independently-edited copies could drift, which would file one record
        into the anchor era while screening it under the pre-fix cap.
        """
        assert TS_ANCHOR_ENABLE is B4E9DF0_MERGED_AT
        assert GRID_SCALED_MAX_STEP_MERGED_AT is B4E9DF0_MERGED_AT

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

    def test_oob_counts_numeric_out_of_grid_resolution(self):
        # A NUMERIC resolution beyond the grid counts as OOB even though its PIT is
        # no longer pinned at 0.0/1.0 (it comes off the declared-percentile curves).
        # The pre-fix counters tested PIT == 0.0/1.0 and read 0/0 on exactly this shape.
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
        # A closed-bound resolution AT the minimum has PIT exactly 0.0 but is not
        # out of grid; the old value-equality counter miscounted this as OOB.
        m = compute_era_metrics("test", [_linear_cdf_record(resolution=0.0)])
        assert m is not None
        assert m.n_oob_low == 0

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


class TestExcludeQids:
    """``exclude_qids`` drops named questions from every row, and says so.

    The known-pipeline-bug cohort is excluded from every other dimension of the
    residual analysis; the width monitor was the one place that still counted it.
    43746/43747 (Minions / Toy Story 5 opening-weekend gross) are both PIT-extreme
    and sit in opposite tails, so leaving them in makes the active era read mildly
    too narrow.
    """

    def test_known_bug_qids_pins_the_documented_cohort(self):
        """Membership is a deliberate, dated decision per question, so it is pinned
        here rather than left to whatever a caller happens to pass.

        43913 (WSOP bracelets) joined 2026-08-25: pre-`9f1175c` discrete max-step cap,
        with all six forecasters stating 79.5-83% on the outcome that resolved while
        the published CDF carried 20.00% on that bin — pinned at exactly 0.200000, the
        201-grid ceiling misapplied to an 11-point grid. Receipts in
        `scratch/residual_2026-08-24/dossiers/43913_dossier.md`.

        43147 and 41798 joined 2026-09-01: the same defect on pre_flip discrete
        records (34- and 12-point grids, true caps 1.0), flagged by the shipped
        `max_step_clamp_screen`. Receipts in
        `scratch/residual_2026-08-31/dim_numeric-width.md`.
        """
        assert frozenset({"43746", "43747", "43913", "43147", "41798"}) == KNOWN_BUG_QIDS

    def test_43913_drops_from_the_rows_it_was_added_for(self):
        # The reclassification is only worth anything if the id actually matches: the
        # collector writes question_id as an int, and 43913 is a discrete record, so
        # both the coercion and the discrete/numeric type gate have to hold.
        data = [
            _record_with_pit(0.5, created_at="2026-06-11T00:00:00Z"),
            _record_with_pit(0.99, created_at="2026-06-11T00:00:00Z", question_id=43913),
        ]
        by_label = {m.label: m for m in compute_all_eras(data, exclude_qids=KNOWN_BUG_QIDS)}
        assert by_label["all"].n_pit == 1
        assert by_label["all"].n_excluded == 1

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


class TestParseExcludeQids:
    """The ``known_bug`` shorthand COMPOSES with explicit ids.

    It used to expand only as the whole argument, so ``--exclude-qids known_bug,43800``
    produced the literal ``{"known_bug", "43800"}``: no question id matches the word, so the
    bug pair stayed in every row while the ``excl`` column reported one exclusion and made the
    run look like the shorthand had worked. That is exactly the silent-exclusion failure the
    column exists to prevent.
    """

    def test_empty_excludes_nothing(self):
        assert parse_exclude_qids("") == frozenset()
        assert parse_exclude_qids("  ,  ") == frozenset()

    def test_shorthand_alone_expands_to_the_pair(self):
        assert parse_exclude_qids("known_bug") == KNOWN_BUG_QIDS
        assert parse_exclude_qids("  known_bug  ") == KNOWN_BUG_QIDS

    def test_shorthand_mixed_with_explicit_ids_expands_and_keeps_both(self):
        assert parse_exclude_qids("known_bug,43800") == KNOWN_BUG_QIDS | {"43800"}
        assert parse_exclude_qids("43800, known_bug ,43801") == KNOWN_BUG_QIDS | {"43800", "43801"}

    def test_the_shorthand_token_never_survives_as_a_literal_id(self):
        """The word itself must not reach ``compute_all_eras`` — it matches no question id, so
        its only effect there is an exclusion the table reports and never performs."""
        assert "known_bug" not in parse_exclude_qids("known_bug,43800")

    def test_explicit_ids_alone_are_passed_through(self):
        assert parse_exclude_qids("43800,43801") == frozenset({"43800", "43801"})

    def test_the_mixed_form_actually_drops_all_three_questions(self):
        """End-to-end through the metrics, not just the parse: the shorthand's ids and the
        explicit id all leave the rows."""
        data = [
            _record_with_pit(0.5, created_at="2026-06-01T00:00:00Z"),
            _record_with_pit(0.025, created_at="2026-06-01T00:00:00Z", question_id=43746),
            _record_with_pit(0.975, created_at="2026-06-01T00:00:00Z", question_id=43747),
            _record_with_pit(0.1, created_at="2026-06-01T00:00:00Z", question_id=43800),
        ]
        by_label = {m.label: m for m in compute_all_eras(data, exclude_qids=parse_exclude_qids("known_bug,43800"))}

        assert by_label["all"].n_pit == 1
        assert by_label["all"].n_excluded == 3

    def test_the_help_text_states_that_the_shorthand_composes(self, capsys):
        """The composing behavior is only discoverable from ``--help``, and a help string that
        still described the sole-value form is what made the old bug invisible."""
        with pytest.raises(SystemExit):
            main(["--help"])

        help_text = capsys.readouterr().out
        assert "composes" in help_text
        assert "known_bug,43800" in help_text

    def test_the_help_text_names_every_cohort_shorthand(self, capsys):
        """A cohort nobody can discover from ``--help`` gets hardcoded in a round script
        instead, which is how the degraded-run ids ended up copied three times."""
        with pytest.raises(SystemExit):
            main(["--help"])

        help_text = capsys.readouterr().out
        for name in EXCLUSION_COHORTS:
            assert name in help_text


class TestDegradedRunCohorts:
    """The dry-donated-key incident cohorts (2026-07-26 .. 07-28), now tracked constants.

    They were standing scoring exclusions living only in playbook prose, and three separate
    analysis rounds hardcoded private copies of the ids. Membership is a dated decision per
    question, so it is pinned here rather than left to whatever a caller retypes.
    """

    def test_degraded_run_qids_pins_the_eight_one_of_three_publishes(self):
        assert frozenset({"44870", "44871", "44872", "44873", "44874", "44875", "44876", "44877"}) == DEGRADED_RUN_QIDS

    def test_partial_degraded_qids_pins_the_three_two_of_three_publishes(self):
        assert frozenset({"44841", "44856", "44912"}) == PARTIAL_DEGRADED_QIDS

    def test_the_cohorts_are_disjoint_from_each_other_and_from_the_bug_pair(self):
        """Overlap would double-count a question in the excluded tally and make the two
        forecaster-count arms non-exclusive."""
        assert not DEGRADED_RUN_QIDS & PARTIAL_DEGRADED_QIDS
        assert not DEGRADED_RUN_QIDS & KNOWN_BUG_QIDS
        assert not PARTIAL_DEGRADED_QIDS & KNOWN_BUG_QIDS

    def test_the_ids_are_question_ids_not_the_post_ids_of_the_same_questions(self):
        """The eight questions carry post ids 44721-44728. Storing those instead would make
        every question-id-keyed join miss, and minibench POST ids 44873-44877 sit inside the
        question-id range, so a "match either id" join admits five unrelated questions."""
        post_ids = {str(pid) for pid in range(44721, 44729)}
        assert not DEGRADED_RUN_QIDS & post_ids

    def test_every_cohort_is_reachable_by_its_shorthand(self):
        assert EXCLUSION_COHORTS == {
            "known_bug": KNOWN_BUG_QIDS,
            "degraded_run": DEGRADED_RUN_QIDS,
            "partial_degraded": PARTIAL_DEGRADED_QIDS,
        }

    def test_each_shorthand_expands_and_composes(self):
        assert parse_exclude_qids("degraded_run") == DEGRADED_RUN_QIDS
        assert parse_exclude_qids("partial_degraded") == PARTIAL_DEGRADED_QIDS
        assert parse_exclude_qids("degraded_run,partial_degraded") == DEGRADED_RUN_QIDS | PARTIAL_DEGRADED_QIDS
        assert parse_exclude_qids("known_bug, degraded_run ,43800") == (KNOWN_BUG_QIDS | DEGRADED_RUN_QIDS | {"43800"})

    def test_an_unrecognized_non_numeric_token_raises_instead_of_excluding_nothing(self):
        """With one shorthand a typo was survivable; with three, ``degraded`` would drop
        nothing while the ``excl`` column read 0 — indistinguishable from a cohort whose
        questions aren't in this pull."""
        with pytest.raises(ValueError, match="neither a question id nor a cohort shorthand"):
            parse_exclude_qids("degraded")
        with pytest.raises(ValueError, match="known_bug"):
            parse_exclude_qids("43800,knownbug")
        # str.isdigit() alone accepts a fullwidth digit, which would pass the guard and then
        # match no question id: exactly the silent no-op the guard exists to prevent.
        fullwidth_43800 = "".join(chr(0xFF10 + int(digit)) for digit in "43800")
        with pytest.raises(ValueError, match="neither a question id"):
            parse_exclude_qids(fullwidth_43800)

    def test_a_degraded_run_question_actually_leaves_the_rows(self):
        """End-to-end through the metrics: the constant is only worth anything if the id
        matches the int question_id the collector writes."""
        data = [
            _record_with_pit(0.5, created_at="2026-08-01T00:00:00Z"),
            _record_with_pit(0.025, created_at="2026-08-01T00:00:00Z", question_id=44872),
            _record_with_pit(0.975, created_at="2026-08-01T00:00:00Z", question_id=44841),
        ]
        by_label = {
            m.label: m for m in compute_all_eras(data, exclude_qids=parse_exclude_qids("degraded_run,partial_degraded"))
        }
        assert by_label["all"].n_pit == 1
        assert by_label["all"].n_excluded == 2


class TestExcludeQidsCliReporting:
    """``main`` reports requested-vs-matched and warns only on the id-space confusion.

    A bare numeric id matching no record used to be a silent no-op — pasting the
    degraded cohort's POST ids rendered byte-identically to ``--exclude-qids ''``. The
    numeric half of the failure the shorthand raise closed stays reportable here without
    alarming on a cohort id that simply isn't in the pull.
    """

    def _write(self, tmp_path, records: list[dict]) -> str:
        path = tmp_path / "data.json"
        path.write_text(json.dumps(records))
        return str(path)

    def test_reports_requested_and_matched_counts(self, tmp_path, caplog):
        records = [
            _record_with_pit(0.5, created_at="2026-08-01T00:00:00Z", question_id=99999),
            _record_with_pit(0.5, created_at="2026-08-01T00:00:00Z", question_id=44872),
        ]
        path = self._write(tmp_path, records)
        with caplog.at_level(logging.INFO):
            main(["--cached", path, "--exclude-qids", "degraded_run"])
        # 8 requested (the whole degraded_run cohort), 1 present in this pull.
        assert any("8 requested id(s), 1 matched" in r.message for r in caplog.records)

    def test_warns_when_an_explicit_id_is_a_post_id_not_a_question_id(self, tmp_path, caplog):
        # 44721 is the POST id of question 44870 (a degraded_run member). Pasting post ids
        # is the collision the cohort comment warns about.
        record = _record_with_pit(0.5, created_at="2026-08-01T00:00:00Z", question_id=44870)
        record["post_id"] = 44721
        path = self._write(tmp_path, [record])
        with caplog.at_level(logging.WARNING):
            main(["--cached", path, "--exclude-qids", "44721"])
        assert any(
            "matched no question_id but IS a post_id" in r.message
            for r in caplog.records
            if r.levelno == logging.WARNING
        )

    def test_no_post_id_warning_on_a_correct_cohort_pass(self, tmp_path, caplog):
        record = _record_with_pit(0.5, created_at="2026-08-01T00:00:00Z", question_id=44870)
        record["post_id"] = 44721
        path = self._write(tmp_path, [record])
        with caplog.at_level(logging.WARNING):
            main(["--cached", path, "--exclude-qids", "degraded_run"])
        assert not any("IS a post_id" in r.message for r in caplog.records if r.levelno == logging.WARNING)


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


def _rig_count_record(
    *,
    band_bin_multiple: float,
    declared_top: tuple[float, ...] = (602.0, 603.0, 606.0),
    top_label: float = 99.0,
    lower_label: float = 1.0,
    declared_bottom: tuple[float, ...] = (566.0, 568.0, 570.0),
    open_upper: bool = True,
    open_lower: bool = True,
    question_id: object = 45218,
    created_at: str = "2026-08-13T06:11:38Z",
) -> dict:
    """The q45218 geometry: the record whose flat -219.5 zone motivated the detector.

    A 72-point discrete grid over [559.5, 630.5] (one rig per step), open on both ends, three
    members declaring p99 at 602 / 603 / 606 (median 603) and p1 at 566 / 568 / 570 (median
    568), ``cdf[0] = 0.01`` and ``cdf[-1] = 0.99`` — the canonical p1/p99 out-of-range mass.
    Each bin fully above the declared p99 carries ``band_bin_multiple`` times the platform's
    minimum step (``0.01 / 71``), so a caller dials starvation directly: the real record sat
    at ~1.12, every one of its 27 upper bins pinned at the structural floor.

    The mass below the declared p1 is left at a healthy density so a test can read one side at
    a time; ``declared_bottom`` mirrors the geometry when the low side is the subject.
    """
    n_points = 72
    grid = np.linspace(559.5, 630.5, n_points)
    min_step = 0.01 / (n_points - 1)
    cdf_start, cdf_end = 0.01, 0.99

    anchor_high = float(np.median(declared_top))
    first_band_bin = min(int(np.searchsorted(grid, anchor_high, side="left")), n_points - 1)
    band_bins = (n_points - 1) - first_band_bin

    if band_bins == 0:
        # The declared p99 sits at or past the displayed ceiling (the q44842 shape): there is no
        # in-range band to starve, so the CDF is a plain ramp across the whole displayed range.
        cdf = np.linspace(cdf_start, cdf_end, n_points)
    else:
        band_mass = band_bins * band_bin_multiple * min_step
        cdf = np.empty(n_points, dtype=float)
        cdf[: first_band_bin + 1] = np.linspace(cdf_start, cdf_end - band_mass, first_band_bin + 1)
        cdf[first_band_bin:] = cdf[first_band_bin] + np.arange(band_bins + 1) * (band_mass / band_bins)

    percentiles = {
        f"model-{i}": [[lower_label, low], [50.0, 586.0], [top_label, high]]
        for i, (low, high) in enumerate(zip(declared_bottom, declared_top, strict=True))
    }
    return {
        "type": "discrete",
        "question_id": question_id,
        "title": "How many active US drilling rigs will there be",
        "our_forecast_values": cdf.tolist(),
        "resolution_parsed": 588.0,
        "scaling": {"range_min": 559.5, "range_max": 630.5, "zero_point": None},
        "open_lower_bound": open_lower,
        "open_upper_bound": open_upper,
        "bot_comment_created_at": created_at,
        "per_model_numeric_percentiles": percentiles,
    }


def _high_side(record: dict) -> OuterTailReading:
    [reading] = [r for r in measure_outer_tails(record) if r.side == "high"]
    return reading


def _flat_zone_score(band_bin_multiple: float, *, n_points: int = 72, n_open_bounds: int = 2) -> float:
    """The closed-form Metaculus log score for a bin holding ``multiple`` x the min step.

    ``50 * ln(pmf / baseline)`` with ``pmf = multiple * 0.01 / N`` and
    ``baseline = (1 - 0.05 * open_bounds) / N``. At a multiple near 1.1 that is about -219 on
    ANY grid size, which is what makes a starved band a flat cliff rather than a gradient:
    q45218 measured -219.5 and q44182, the worst record on the board, -219.0.
    """
    n_inbound = n_points - 1
    pmf = band_bin_multiple * 0.01 / n_inbound
    baseline = (1.0 - 0.05 * n_open_bounds) / n_inbound
    return 50.0 * float(np.log(pmf / baseline))


class TestStarvedOuterTail:
    """The open-bound p99 cliff: mass beyond the declared outer anchor starved to the floor.

    Distinct from the shipped ``CDF_MAXSTEP_CLIP`` smear. On an open bound the declared tail
    can be routed out of the displayed range entirely, leaving every in-range bin above the
    declared p99 pinned at the platform's structural minimum step — so every resolution out
    there scores at the same floor (-219.5 on q45218, -219.0 on q44182, the worst record on
    the board), a cliff nobody declared and one no modest widening walks out of, because the
    defect is a step function in the declared p99.
    """

    def test_the_q45218_geometry_fires(self):
        reading = _high_side(_rig_count_record(band_bin_multiple=1.12))
        assert reading.verdict is OuterTailVerdict.STARVED
        assert reading.starved
        assert reading.declared_percentile == 99.0
        assert reading.declared_value == pytest.approx(603.0)
        assert reading.bound_value == pytest.approx(630.5)
        assert reading.band_bins == 27
        assert reading.tail_mass == pytest.approx(27 * 1.12 * 0.01 / 71)
        assert reading.beyond_bound_mass == pytest.approx(0.01)
        assert reading.floor_multiple == pytest.approx(1.12)
        # The whole point: any resolution in that band earns the platform's floor score, which
        # is what q45218 measured at -219.5 and q44182 at -219.0.
        assert reading.flat_zone_log_score == pytest.approx(_flat_zone_score(1.12), abs=0.5)

    def test_an_honest_tail_does_not_fire(self):
        reading = _high_side(_rig_count_record(band_bin_multiple=6.0))
        assert reading.verdict is OuterTailVerdict.HEALTHY
        assert not reading.starved
        assert reading.floor_multiple == pytest.approx(6.0)
        # Still a bad place to resolve, but ~84 points off the cliff rather than sitting on it.
        assert reading.flat_zone_log_score == pytest.approx(_flat_zone_score(6.0), abs=0.5)
        assert reading.flat_zone_log_score is not None
        assert reading.flat_zone_log_score - _flat_zone_score(1.12) > 80.0

    def test_the_threshold_is_the_named_constant(self):
        just_under = _high_side(_rig_count_record(band_bin_multiple=STARVED_OUTER_TAIL_FLOOR_MULTIPLE * 0.99))
        just_over = _high_side(_rig_count_record(band_bin_multiple=STARVED_OUTER_TAIL_FLOOR_MULTIPLE * 1.01))
        assert just_under.starved
        assert not just_over.starved
        # Calibrated over the archived performance dataset (271 numeric/discrete records, 417
        # measurable open-bound sides; receipts in scratch/next_season_bundle_2026-09/item19/):
        # q45218 reads 1.12 on both sides and q44182 1.46 high / 1.13 low, so both fire, while
        # q44842 is not measurable on either side because its declaration routes the tail past
        # the ceiling. The measured distribution is bimodal — 44 sides at [1.00, 1.25), then
        # ~8 per 0.25 bucket to 3.0, median 10.3 — so the cut is not load-bearing, and 2.0
        # keeps margin over q44182's 1.46.
        assert STARVED_OUTER_TAIL_FLOOR_MULTIPLE == 2.0

    def test_an_explicit_floor_multiple_overrides_the_default(self):
        record = _rig_count_record(band_bin_multiple=3.0)
        assert not _high_side(record).starved
        [strict] = [r for r in measure_outer_tails(record, floor_multiple=4.0) if r.side == "high"]
        assert strict.starved

    def test_the_q44842_shape_is_unmeasurable_rather_than_starved(self):
        # q44842 declared p99 at 20500 against a 14000 ceiling: the declaration itself routes
        # the outer tail past the displayed range, which is the honest open-bound shape (that
        # record won spot peer +24.4). There is no in-range band above the anchor to starve.
        reading = _high_side(_rig_count_record(band_bin_multiple=1.12, declared_top=(640.0, 660.0, 700.0)))
        assert reading.verdict is OuterTailVerdict.DECLARED_BEYOND_BOUND
        assert reading.starved is False
        assert reading.tail_mass is None

    def test_the_low_side_mirror_fires_on_a_starved_floor(self):
        # The same geometry read from below: the bins under the declared p1 (568) at 1.2x the
        # min step, with the canonical 1% below the displayed floor.
        record = _rig_count_record(band_bin_multiple=6.0)
        cdf = np.asarray(record["our_forecast_values"])
        grid = np.linspace(559.5, 630.5, 72)
        last_bin = int(np.searchsorted(grid, 568.0, side="right")) - 1
        min_step = 0.01 / 71
        band_mass = last_bin * 1.2 * min_step
        cdf[: last_bin + 1] = 0.01 + np.arange(last_bin + 1) * (band_mass / last_bin)
        record["our_forecast_values"] = cdf.tolist()

        [low] = [r for r in measure_outer_tails(record) if r.side == "low"]
        assert low.verdict is OuterTailVerdict.STARVED
        assert low.declared_percentile == 1.0
        assert low.declared_value == pytest.approx(568.0)
        assert low.bound_value == pytest.approx(559.5)
        assert low.band_bins == last_bin
        assert low.beyond_bound_mass == pytest.approx(0.01)
        assert low.floor_multiple == pytest.approx(1.2)
        assert low.flat_zone_log_score == pytest.approx(_flat_zone_score(1.2), abs=0.5)

    def test_a_closed_bound_side_is_never_scanned(self):
        readings = measure_outer_tails(_rig_count_record(band_bin_multiple=1.12, open_upper=False))
        assert [r.side for r in readings] == ["low"]

    def test_a_record_without_member_curves_is_unmeasurable_not_healthy(self):
        # Stacked-era and trimmed comments lose the per-model percentile lines. The absence of
        # a declaration must not read as "this tail is fine" — that silent pass is exactly
        # what the verdict enum exists to prevent.
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {}
        assert _high_side(record).verdict is OuterTailVerdict.NO_MEMBER_CURVE

    def test_a_curve_whose_shared_anchor_is_not_extreme_is_unmeasurable(self):
        # A trimmed recovery can leave the members sharing only their p50. Reading the band as
        # "everything above the median" would call almost every record starved.
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {
            "model-0": [[10.0, 575.0], [50.0, 586.0]],
            "model-1": [[50.0, 586.0], [99.0, 603.0]],
        }
        assert _high_side(record).verdict is OuterTailVerdict.ANCHOR_NOT_EXTREME

    def test_anonymous_forecaster_keys_are_excluded_from_the_anchor(self):
        # On a stacker-fired record the positional `Forecaster N` bucket holds the stacker's
        # AGGREGATE, not a member; pooling it into the median-of-members moves the anchor.
        # Same rule as declared_percentile_pit and max_step_clamp_screen next door.
        record = _rig_count_record(band_bin_multiple=1.12)
        record["per_model_numeric_percentiles"] = {
            "Forecaster 1": [[1.0, 560.0], [50.0, 586.0], [99.0, 628.0]],
        }
        assert _high_side(record).verdict is OuterTailVerdict.NO_MEMBER_CURVE

    def test_a_record_without_a_usable_cdf_is_unmeasurable(self):
        record = _rig_count_record(band_bin_multiple=1.12)
        record["scaling"] = {}
        assert {r.verdict for r in measure_outer_tails(record)} == {OuterTailVerdict.NO_USABLE_CDF}

    def test_non_numeric_records_are_skipped_by_the_scan(self):
        scan = scan_outer_tails([{"type": "binary", "our_prob_yes": 0.5}])
        assert scan.readings == []
        assert scan.starved == []

    def test_the_scan_counts_verdicts_and_ranks_the_flagged_rows(self):
        starved = _rig_count_record(band_bin_multiple=1.12, question_id=45218)
        healthy = _rig_count_record(band_bin_multiple=6.0, question_id=44182)
        blind = _rig_count_record(band_bin_multiple=1.12, question_id=44553)
        blind["per_model_numeric_percentiles"] = {}
        scan = scan_outer_tails([starved, healthy, blind])

        assert scan.n_scanned == 6  # three records x two open sides
        assert [r.question_id for r in scan.starved] == [45218]
        assert scan.verdict_counts[OuterTailVerdict.STARVED] == 1
        assert scan.verdict_counts[OuterTailVerdict.NO_MEMBER_CURVE] == 2

    def test_excluded_qids_leave_the_scan(self):
        scan = scan_outer_tails(
            [_rig_count_record(band_bin_multiple=1.12, question_id=43913)],
            exclude_qids=KNOWN_BUG_QIDS,
        )
        assert scan.readings == []
        assert scan.n_excluded == 1

    def test_the_rendered_section_states_every_reported_field(self):
        scan = scan_outer_tails([_rig_count_record(band_bin_multiple=1.12)])
        md = render_starved_outer_tails(scan)
        assert "Starved outer tails" in md
        assert "45218" in md
        assert "603" in md  # the declared p99
        assert "630.5" in md  # the displayed ceiling
        assert "-219" in md  # the flat-zone log score
        assert "| 27 |" in md  # band bins
        assert "0.0043" in md  # tail mass, 27 bins x 1.12 x the min step

    def test_a_scan_with_nothing_starved_says_so(self):
        md = render_starved_outer_tails(scan_outer_tails([_rig_count_record(band_bin_multiple=6.0)]))
        assert "Starved outer tails" in md
        assert "none" in md.lower()

    def test_the_unmeasurable_tally_is_disclosed(self):
        blind = _rig_count_record(band_bin_multiple=1.12)
        blind["per_model_numeric_percentiles"] = {}
        md = render_starved_outer_tails(scan_outer_tails([blind]))
        assert "no_member_curve: 2" in md

    def test_the_cli_renders_the_section(self, tmp_path, capsys):
        path = tmp_path / "data.json"
        path.write_text(json.dumps([_rig_count_record(band_bin_multiple=1.12)]))
        main(["--cached", str(path)])
        out = capsys.readouterr().out
        assert "Numeric width / calibration monitor" in out
        assert "Starved outer tails" in out

    def test_the_cli_can_write_the_scan_as_json(self, tmp_path):
        data = tmp_path / "data.json"
        data.write_text(json.dumps([_rig_count_record(band_bin_multiple=1.12)]))
        out_json = tmp_path / "starved.json"
        main(["--cached", str(data), "--output-starved-json", str(out_json)])
        payload = json.loads(out_json.read_text())
        [high] = [row for row in payload["readings"] if row["side"] == "high"]
        assert high["verdict"] == "starved"
        assert high["declared_value"] == pytest.approx(603.0)
        assert high["flat_zone_log_score"] < -215.0
        assert payload["verdict_counts"]["starved"] == 1
