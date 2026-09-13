"""How the width monitor reads one record's PIT.

Three shapes: an ordinary point reading off the published CDF, the SET-valued reading an
out-of-range string resolution forces, and a numeric resolution that lands beyond the value grid
and has to come off the members' declared percentile curves.
"""

import numpy as np
import pytest

from metaculus_bot.performance_analysis.width_monitor import (
    MIN_N_FOR_POINT_METRICS,
    compute_all_eras,
    compute_era_metrics,
    compute_pit,
    compute_pit_reading,
    jeffreys_ci,
    render_markdown,
)
from tests.width_monitor_fakes import (
    _below_bound_mass_record,
    _linear_cdf_record,
    _out_of_range_mass_record,
    _pit_and_side,
    _record_with_pit,
    _row_cells,
)


class TestPit:
    def test_pit_matches_linear_cdf(self):
        """PIT is resolution/100 on the identity ramp, whose F(x) = x/100."""
        for res, expected in [(10.0, 0.10), (25.0, 0.25), (50.0, 0.50), (90.0, 0.90)]:
            rec = _linear_cdf_record(resolution=res)
            assert compute_pit(rec) == pytest.approx(expected, abs=1e-9)

    def test_pit_out_of_bounds_degenerates_to_a_point_when_no_mass_is_out_there(self):
        """An out-of-range marker on a full-span ramp still answers as a single point.

        The identity ramp spans the whole [0, 1], so cdf[0] == 0 and cdf[-1] == 1: the
        out-of-range INTERVAL collapses to the one value the old convention forced, and
        ``compute_pit`` still answers it.
        """
        assert compute_pit(_linear_cdf_record(resolution="below_lower_bound")) == 0.0
        assert compute_pit(_linear_cdf_record(resolution="above_upper_bound")) == 1.0

    def test_pit_none_when_unscorable(self):
        """No PIT without bounds to build a grid from, and none for a non-numeric, non-OOB
        resolution."""
        rec = _linear_cdf_record(resolution=50.0)
        rec["scaling"] = {}
        assert compute_pit(rec) is None
        assert compute_pit(_linear_cdf_record(resolution="annulled")) is None


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
        """[0.87, 1] intersects [0.10, 0.90], so the record is covered rather than a miss."""
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
        """A cdf[-1] of 0.999, the open-bound structural floor, still misses the band high.

        The whole interval sits above 0.90, so the record misses exactly as it should.
        """
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
        """Point statistics skip the set-valued reading, and the row discloses how many it skipped.

        The dataset is point PITs spread across the unit interval plus one set-valued record.
        """
        recs = [_record_with_pit(p) for p in np.linspace(0.05, 0.95, MIN_N_FOR_POINT_METRICS)]
        recs.append(_out_of_range_mass_record(resolution="above_upper_bound", cdf_end=0.87))
        m = compute_era_metrics("test", recs)
        assert m is not None
        assert m.n_pit == MIN_N_FOR_POINT_METRICS + 1
        assert m.n_point == MIN_N_FOR_POINT_METRICS
        assert m.n_oob_interval == 1
        # An imputed midpoint (0.935) would have pulled both of these.
        points = np.linspace(0.05, 0.95, MIN_N_FOR_POINT_METRICS)
        assert m.mean_pit == pytest.approx(points.mean())
        assert m.pit_std == pytest.approx(points.std())
        # 8 of the 10 point PITs are inside [0.10, 0.90] and [0.87, 1] intersects it: 9 of 11.
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
        """Only the STRING markers are set-valued, never a numeric value beyond the grid.

        When the platform gives the value, the members' declared curves read a real quantile off
        it (see TestOutOfGridPit).
        """
        rec = _below_bound_mass_record(
            resolution=50.0,
            per_model_percentiles={"model-a": [[10.0, 80.0], [50.0, 90.0], [90.0, 105.0]]},
        )
        reading = compute_pit_reading(rec)
        assert reading is not None
        assert not reading.is_interval
        assert reading.point == pytest.approx(0.10, abs=1e-9)
        assert reading.oob_side == "low"


class TestOutOfGridPit:
    """A numeric resolution BEYOND the value grid must not be censored at cdf[0]/cdf[-1]."""

    def test_below_grid_resolution_reads_low_tail_not_the_clamp(self):
        """Resolution 50 is below every declared value of every member, so each member curve
        reads its lowest declared percentile (P10 -> 0.10).

        The grid clamp would have said 0.90, the opposite tail.
        """
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
        """The median of the member curves' reads, which for two members is their mean.

        Resolution 95: model-a interpolates 0.50 + (95-90)/(105-90)*0.40 = 0.6333, model-b reads
        exactly its P50 = 0.50.
        """
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
        """Without member curves the grid-endpoint read is kept, but the OOB side still surfaces
        the record.

        The degraded path is a stacked-era record with no per-model bullets to interpolate.
        """
        rec = _below_bound_mass_record(resolution=50.0)
        pit, oob_side = _pit_and_side(rec)
        assert oob_side == "low"
        assert pit == pytest.approx(0.90, abs=1e-9)

    def test_in_grid_resolution_has_no_oob_side(self):
        pit, oob_side = _pit_and_side(_linear_cdf_record(resolution=50.0))
        assert oob_side is None
        assert pit == pytest.approx(0.50, abs=1e-9)

    def test_resolution_exactly_at_bound_keeps_endpoint_read(self):
        """AT a bound the clamp IS the correct PIT: F(bound) = cdf[0]."""
        rec = _below_bound_mass_record(resolution=100.0)
        pit, oob_side = _pit_and_side(rec)
        assert oob_side is None
        assert pit == pytest.approx(0.90, abs=1e-9)
