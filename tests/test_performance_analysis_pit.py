"""Tests for the PIT reading: the value grid, the declared-percentile fallback, out-of-range sets.

Covers ``_interpolate_pit`` against the actual (linear or geometric) value grid, the
declared-percentile curves it falls back to beyond the grid and their junk tolerance,
``numeric_pit_analysis`` end to end on a mixed cohort, and the set-valued reading an out-of-range
resolution gets plus its exclusion from the point statistics.
"""

from __future__ import annotations

from typing import ClassVar, cast

import numpy as np
import pytest

from metaculus_bot import performance_analysis
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.performance_analysis.analysis import (
    PitReading,
    _interpolate_pit,
    _single_curve_pit,
    declared_percentile_pit,
    numeric_pit_analysis,
    out_of_range_pit_reading,
)
from tests.performance_analysis_fakes import _old_interpolate_pit


class TestInterpolatePit:
    """PIT = F(resolution). F must be read against the ACTUAL value grid the CDF
    lives on (linear for linear questions, geometric for zero_point questions),
    not against a linear index map. The old linear-index map mis-buckets
    log-scaled resolutions by up to ~0.24."""

    def test_linear_question_matches_old_behavior(self):
        """On a linear grid the two maps are mathematically equivalent, so the value-grid
        interpolation must equal the old linear-index one within float tolerance."""
        lower, upper = 0.0, 100.0
        cdf = [float(value) for value in np.linspace(0.0, 1.0, 201)]  # straight-line CDF
        grid = list(build_cdf_value_grid(lower, upper, None, num_points=201))
        for resolution in (0.0, 12.3, 25.0, 50.0, 73.7, 100.0):
            new = _interpolate_pit(resolution, lower, upper, cdf, value_grid=grid, zero_point=None)
            old = _old_interpolate_pit(resolution, lower, upper, cdf)
            assert new == pytest.approx(old, abs=1e-9)

    def test_linear_endpoints_and_midpoint(self):
        lower, upper = 0.0, 100.0
        cdf = [float(value) for value in np.linspace(0.0, 1.0, 201)]
        grid = list(build_cdf_value_grid(lower, upper, None, num_points=201))
        assert _interpolate_pit(lower, lower, upper, cdf, value_grid=grid) == pytest.approx(cdf[0])
        assert _interpolate_pit(upper, lower, upper, cdf, value_grid=grid) == pytest.approx(cdf[-1])
        assert _interpolate_pit(50.0, lower, upper, cdf, value_grid=grid) == pytest.approx(0.5)

    def test_log_scaled_question_differs_and_is_correct(self):
        """On a log-scaled (zero_point) question the value grid is geometric, so the resolution
        lands on a different CDF index than the linear-index map put it.

        Resolution 31.6 is only ~0.1% along the range linearly but a meaningful chunk of
        probability on the geometric grid, which is where the fix has to bite: the new reading
        is the geometric-midpoint PIT (~0.5), not the near-zero PIT the linear-index map gives.
        """
        lower, upper, zero_point = 1.0, 1000.0, 0.0
        cdf = [float(value) for value in np.linspace(0.0, 1.0, 201)]  # uniform-in-index CDF
        geo_grid = build_cdf_value_grid(lower, upper, zero_point, num_points=201)

        resolution = 31.6  # ~10^1.5 -> roughly the geometric midpoint of [1, 1000]

        new = _interpolate_pit(resolution, lower, upper, cdf, value_grid=list(geo_grid), zero_point=zero_point)
        old = _old_interpolate_pit(resolution, lower, upper, cdf)

        expected = float(np.interp(resolution, geo_grid, np.asarray(cdf, dtype=float)))
        assert new == pytest.approx(expected, abs=1e-12)

        # The fix must bite: geometric vs linear-index map differ materially here.
        assert abs(new - old) > 0.2
        # And the new value is the geometric-midpoint PIT, not the linear-index map's near-zero one.
        assert new == pytest.approx(0.5, abs=0.02)
        assert old < 0.05

    def test_falls_back_to_zero_point_grid_when_value_grid_absent(self):
        """With no continuous_range supplied the geometric grid is reconstructed from zero_point,
        and the result must match interpolation against that rebuilt grid."""
        lower, upper, zero_point = 1.0, 1000.0, 0.0
        cdf = [float(value) for value in np.linspace(0.0, 1.0, 201)]
        resolution = 31.6

        no_grid = _interpolate_pit(resolution, lower, upper, cdf, value_grid=None, zero_point=zero_point)
        rebuilt = build_cdf_value_grid(lower, upper, zero_point, num_points=201)
        expected = float(np.interp(resolution, rebuilt, np.asarray(cdf, dtype=float)))
        assert no_grid == pytest.approx(expected, abs=1e-12)

    def test_mismatched_value_grid_length_falls_back(self):
        """A value_grid whose length disagrees with the CDF is ignored, and the grid is rebuilt
        from the bounds and zero_point instead."""
        lower, upper = 0.0, 100.0
        cdf = [float(value) for value in np.linspace(0.0, 1.0, 201)]
        bad_grid = [0.0, 50.0, 100.0]  # wrong length
        result = _interpolate_pit(50.0, lower, upper, cdf, value_grid=bad_grid, zero_point=None)
        assert result == pytest.approx(0.5)

    def test_degenerate_range_raises_instead_of_answering_with_the_best_case(self):
        """A zero-width question has no PIT. This used to return 0.5, which is the single most
        favorable value available — inside BOTH coverage bands — so a degenerate record
        silently improved every calibration statistic it entered. The caller screens the
        range now (see ``TestDeclaredPercentilePitDropsDegenerateRanges``)."""
        cdf = [float(value) for value in np.linspace(0.0, 1.0, 201)]

        with pytest.raises(ValueError, match="degenerate question range"):
            _interpolate_pit(5.0, 10.0, 10.0, cdf)


class TestInterpolatePitOutOfGrid:
    """The q44218 shape: a resolution BEYOND the grid must not be censored at cdf[0]/cdf[-1].

    With below-bound mass expressible on open bounds, cdf[0] can be ~0.9, so the grid
    clamp reads a below-grid resolution — a LOW-tail event — as a high PIT. Beyond the
    grid the PIT must come off the members' declared-percentile curves instead.
    """

    _LOWER: ClassVar[float] = 100.0
    _UPPER: ClassVar[float] = 200.0
    # 90% of the mass below the open lower bound (F(100) = 0.90), like q44218's 0.9168.
    _CDF: ClassVar[list[float]] = list(np.linspace(0.90, 0.975, 201))
    _PERCENTILES: ClassVar[dict[str, list[list[float]]]] = {
        "model-a": [[10.0, 80.0], [50.0, 90.0], [90.0, 105.0]],
        "model-b": [[10.0, 85.0], [50.0, 95.0], [90.0, 110.0]],
    }

    def _grid(self) -> list[float]:
        return list(build_cdf_value_grid(self._LOWER, self._UPPER, None, num_points=201))

    def test_below_grid_resolution_reads_low_tail_not_the_clamp(self):
        """Resolution 50 sits below every declared value of every member, so each curve reads its
        lowest declared percentile (P10, giving 0.10). The grid clamp would have said 0.90."""
        pit = _interpolate_pit(
            50.0,
            self._LOWER,
            self._UPPER,
            self._CDF,
            value_grid=self._grid(),
            per_model_percentiles=self._PERCENTILES,
        )
        assert pit == pytest.approx(0.10, abs=1e-9)

    def test_fallback_is_median_of_member_curves(self):
        """At resolution 95 model-a interpolates to 0.6333 and model-b reads its P50 of 0.50."""
        pit = _interpolate_pit(
            95.0,
            self._LOWER,
            self._UPPER,
            self._CDF,
            value_grid=self._grid(),
            per_model_percentiles=self._PERCENTILES,
        )
        assert pit == pytest.approx((0.6333333 + 0.50) / 2, abs=1e-6)

    def test_no_member_curves_keeps_grid_read(self):
        """The degraded path, with no per-model percentiles recoverable, keeps the grid-endpoint
        read."""
        pit = _interpolate_pit(50.0, self._LOWER, self._UPPER, self._CDF, value_grid=self._grid())
        assert pit == pytest.approx(0.90, abs=1e-9)

    def test_at_bound_resolution_keeps_endpoint_read(self):
        """AT a bound the clamp IS the correct PIT, since F(bound) equals cdf[0], so the
        declared-percentile fallback must engage only strictly beyond the grid."""
        pit = _interpolate_pit(
            self._LOWER,
            self._LOWER,
            self._UPPER,
            self._CDF,
            value_grid=self._grid(),
            per_model_percentiles=self._PERCENTILES,
        )
        assert pit == pytest.approx(0.90, abs=1e-9)

    def test_numeric_pit_analysis_uses_declared_fallback(self):
        record = {
            "post_id": 1,
            "type": "numeric",
            "our_forecast_values": self._CDF,
            "resolution_parsed": 50.0,
            "scaling": {
                "range_min": self._LOWER,
                "range_max": self._UPPER,
                "zero_point": None,
                "continuous_range": self._grid(),
            },
            "open_lower_bound": True,
            "open_upper_bound": True,
            "numeric_log_score": 0.0,
            "per_model_numeric_percentiles": self._PERCENTILES,
            "metadata": {"category": None},
        }
        result = numeric_pit_analysis([record])
        assert result["count"] == 1
        assert result["pit_values"][0] == pytest.approx(0.10, abs=1e-9)


class TestDeclaredPercentilePitDropsDegenerateRanges:
    """A zero-width question contributes no PIT rather than the most favorable one.

    ``_interpolate_pit`` used to answer 0.5 there, which is inside both coverage bands, so a
    degenerate record silently improved every calibration statistic it entered.
    """

    @staticmethod
    def _record(range_min: float, range_max: float) -> dict:
        return {
            "post_id": 1,
            "type": "numeric",
            "our_forecast_values": list(np.linspace(0.0, 1.0, 201)),
            "resolution_parsed": 5.0,
            "scaling": {"range_min": range_min, "range_max": range_max, "zero_point": None},
            "open_lower_bound": False,
            "open_upper_bound": False,
            "numeric_log_score": 0.0,
            "metadata": {"category": None},
        }

    def test_zero_width_record_is_dropped_not_scored_at_half(self):
        assert numeric_pit_analysis([self._record(10.0, 10.0)]) == {"count": 0}

    def test_an_inverted_range_is_dropped_too(self):
        assert numeric_pit_analysis([self._record(10.0, 5.0)]) == {"count": 0}

    def test_a_real_range_still_scores(self):
        result = numeric_pit_analysis([self._record(0.0, 100.0)])

        assert result["count"] == 1


class TestDeclaredPercentileCurveTolerance:
    """Member curves come out of comment TEXT, so the fallback must tolerate junk.

    Every unusable curve reads as no-curve (dropped from the median) rather than
    raising or contributing a garbage quantile — the callers then either median the
    surviving curves or fall back to the grid read.
    """

    _GOOD: ClassVar[list[list[float]]] = [[10.0, 85.0], [50.0, 95.0], [90.0, 110.0]]

    def test_non_numeric_declared_value_drops_only_that_curve(self):
        """One percentile line parsed to a non-number, so the median is taken over the surviving
        curve alone (model-b at 50 reads its P10 of 0.10) rather than over a coerced zero that
        would drag the quantile."""
        curves = cast(
            "dict[str, list[list[float]]]",
            {"model-a": [[10.0, "n/a"], [50.0, 90.0], [90.0, 105.0]], "model-b": self._GOOD},
        )
        assert declared_percentile_pit(curves, 50.0) == pytest.approx(0.10, abs=1e-9)

    def test_pair_missing_its_value_is_unusable(self):
        """A truncated line recovered as a bare percentile with no value beside it."""
        assert _single_curve_pit([[10.0], [50.0]], 50.0) is None

    def test_anonymous_keys_are_excluded_from_the_median_of_members(self):
        """A positional ``Forecaster N`` bucket on a stacker-fired record holds the STACKER's
        aggregate, so pooling it into a median-of-members counts the aggregate as an extra
        member and pulls the median toward itself. ``max_step_clamp_screen`` next door and
        ``per_model_cohort`` both filter these; this consumer used not to."""
        curves = cast(
            "dict[str, list[list[float]]]",
            {"model-a": self._GOOD, "Forecaster 1": [[10.0, 10.0], [50.0, 12.0], [90.0, 14.0]]},
        )

        # The anonymous curve would read ~0.90 at resolution 50 and swing the median.
        assert declared_percentile_pit(curves, 50.0) == pytest.approx(0.10, abs=1e-9)

    def test_an_all_anonymous_record_yields_none_rather_than_the_stacker_curve(self):
        curves = cast("dict[str, list[list[float]]]", {"Forecaster 1": self._GOOD})

        assert declared_percentile_pit(curves, 50.0) is None

    def test_duplicate_declared_values_stay_usable(self):
        """A flat tail, where P10 equals P50, is legitimate model output, so jitter it into
        strict monotonicity rather than discarding the whole curve."""
        flat_tail = [[10.0, 80.0], [50.0, 80.0], [90.0, 105.0]]
        assert _single_curve_pit(flat_tail, 50.0) == pytest.approx(0.10, abs=1e-9)
        # Between the duplicated value and P90 the curve still interpolates.
        mid = _single_curve_pit(flat_tail, 90.0)
        assert mid is not None
        assert 0.5 < mid < 0.9

    def test_non_finite_declared_values_are_unusable(self):
        """Jitter cannot rescue non-finite values, so the curve must read as no-curve instead of
        returning a nan PIT into the median.

        ``np.errstate`` only silences the expected nan arithmetic inside the guard, which is the
        code under test here.
        """
        with np.errstate(invalid="ignore"):
            assert _single_curve_pit([[10.0, float("inf")], [90.0, float("inf")]], 50.0) is None
            assert _single_curve_pit([[10.0, float("nan")], [50.0, 90.0]], 50.0) is None

    def test_all_curves_unusable_reads_as_no_fallback(self):
        """``declared_percentile_pit`` returning None is what makes ``_interpolate_pit`` and
        ``compute_pit_details`` keep the grid-endpoint read."""
        junk = cast("dict[str, list[list[float]]]", {"model-a": [[50.0, "junk"]]})
        assert declared_percentile_pit(junk, 50.0) is None
        assert declared_percentile_pit(None, 50.0) is None


class TestNumericPitAnalysisValueGrid:
    """End-to-end numeric_pit_analysis on a small mixed cohort: one linear-scaled
    record and one log-scaled (zero_point) record carrying continuous_range."""

    def _record(self, post_id, cdf, resolution, lower, upper, zero_point, continuous_range):
        return {
            "post_id": post_id,
            "type": "numeric",
            "our_forecast_values": cdf,
            "resolution_parsed": resolution,
            "scaling": {
                "range_min": lower,
                "range_max": upper,
                "zero_point": zero_point,
                "continuous_range": continuous_range,
            },
            "open_lower_bound": False,
            "open_upper_bound": False,
            "brier_score": None,
            "log_score": None,
            "numeric_log_score": 0.0,
            "mc_log_score": None,
            "per_model_forecasts": {},
            "metadata": {"category": None},
        }

    def test_continuous_range_used_directly_for_log_scaled(self):
        """One linear record at its midpoint resolution and one log-scaled record at its
        geometric midpoint both read PIT ~0.5, where the linear-index map called the log-scaled
        one ~0.03."""
        cdf = list(np.linspace(0.0, 1.0, 201))
        # Linear question, midpoint resolution -> PIT 0.5.
        lin_grid = list(build_cdf_value_grid(0.0, 100.0, None, num_points=201))
        linear_rec = self._record(1, cdf, 50.0, 0.0, 100.0, None, lin_grid)

        geo_grid = list(build_cdf_value_grid(1.0, 1000.0, 0.0, num_points=201))
        log_rec = self._record(2, cdf, 31.6, 1.0, 1000.0, 0.0, geo_grid)

        result = numeric_pit_analysis([linear_rec, log_rec])
        assert result["count"] == 2
        assert result["pit_values"][0] == pytest.approx(0.5)
        assert result["pit_values"][1] == pytest.approx(0.5, abs=0.02)
        # Both PITs land in the central coverage band.
        assert result["coverage_50"] == pytest.approx(1.0)

    def test_zero_point_zero_without_continuous_range_reconstructs_geometric(self):
        """Regression for the zero_point sentinel bug on the analysis fallback path.

        A log-scale record serializes ``zero_point == 0`` with a positive ``range_min`` but
        carries NO continuous_range, from an old archive or schema drift, and
        ``numeric_pit_analysis`` must then reconstruct the GEOMETRIC grid via
        ``grid_zero_point`` rather than a linear one. On [1, 1000] the geometric midpoint
        (~31.6) is PIT ~0.5, where the buggy linear-grid reconstruction called it near-zero.
        """
        cdf = list(np.linspace(0.0, 1.0, 201))
        log_rec = self._record(1, cdf, 31.6, 1.0, 1000.0, 0, None)
        result = numeric_pit_analysis([log_rec])
        assert result["count"] == 1
        assert result["pit_values"][0] == pytest.approx(0.5, abs=0.02)


class TestSetValuedOutOfRangePit:
    """An out-of-range resolution's PIT is a SET, and point statistics exclude it.

    The platform reports "beyond the displayed range" as a string, so the resolution VALUE
    is unknown and ``F(resolution)`` is only pinned to ``[cdf[-1], 1]`` (above) or
    ``[0, cdf[0]]`` (below). Forcing it to 1.0 / 0.0 counted q44842 as a high-side band
    miss: an open-bound record that deliberately published 13% of its mass above the
    displayed ceiling, resolved ``above_upper_bound``, and won spot peer +24.4.
    """

    @staticmethod
    def _record(resolution, *, cdf_start: float = 0.0, cdf_end: float = 1.0) -> dict:
        cdf = list(np.linspace(cdf_start, cdf_end, 201))
        return {
            "post_id": 1,
            "type": "numeric",
            "our_forecast_values": cdf,
            "resolution_parsed": resolution,
            "scaling": {"range_min": 0.0, "range_max": 100.0, "zero_point": None},
            "open_lower_bound": True,
            "open_upper_bound": True,
            "numeric_log_score": 0.0,
            "metadata": {"category": None},
        }

    def test_the_interval_is_read_off_our_own_published_tail_mass(self):
        above = out_of_range_pit_reading("above_upper_bound", list(np.linspace(0.0, 0.87, 201)))
        assert above is not None
        assert (above.low, above.high) == pytest.approx((0.87, 1.0))
        assert above.oob_side == "high"
        assert above.is_interval
        assert above.point is None

        below = out_of_range_pit_reading("below_lower_bound", list(np.linspace(0.13, 1.0, 201)))
        assert below is not None
        assert (below.low, below.high) == pytest.approx((0.0, 0.13))
        assert below.oob_side == "low"

        # Not an out-of-range marker at all.
        assert out_of_range_pit_reading("annulled", [0.0, 1.0]) is None
        assert out_of_range_pit_reading(50.0, [0.0, 1.0]) is None

    def test_a_closed_bound_interval_collapses_to_the_old_point_convention(self):
        """With no mass beyond the bound, ``[cdf[-1], 1]`` is ``[1, 1]``, so the set-valued
        reading degenerates to exactly the 1.0 the old convention forced and nothing changes on
        records that put nothing out of range."""
        reading = out_of_range_pit_reading("above_upper_bound", list(np.linspace(0.0, 1.0, 201)))
        assert reading is not None
        assert not reading.is_interval
        assert reading.point == pytest.approx(1.0)

    def test_a_point_reading_answers_the_band_predicates_like_a_scalar(self):
        point = PitReading.from_point(0.42)
        assert point.point == pytest.approx(0.42)
        assert point.intersects(0.10, 0.90)
        assert point.at_or_below(0.50)
        assert not point.at_or_below(0.40)
        assert not point.entirely_below(0.10)
        assert not point.entirely_above(0.90)

    def test_q44842_shape_is_covered_and_excluded_from_the_histogram(self):
        result = numeric_pit_analysis([self._record("above_upper_bound", cdf_end=0.87)])
        assert result["count"] == 1
        assert result["n_point"] == 0
        assert result["n_oob_interval"] == 1
        assert result["pit_values"] == []
        assert result["pit_intervals"] == [(pytest.approx(0.87), 1.0)]
        # [0.87, 1] intersects [0.05, 0.95] and [0.25, 0.75] it does not.
        assert result["coverage_90"] == pytest.approx(1.0)
        assert result["coverage_50"] == pytest.approx(0.0)
        assert sum(result["histogram"]) == 0

    def test_a_starved_tail_is_still_outside_the_coverage_band(self):
        """``cdf[-1] = 0.999`` is the open-bound structural floor, so [0.999, 1] lies wholly
        above 0.95 and this record is the band miss that the q44842 shape is not."""
        result = numeric_pit_analysis([self._record("above_upper_bound", cdf_end=0.999)])
        assert result["coverage_90"] == pytest.approx(0.0)

    def test_the_below_bound_mirror(self):
        covered = numeric_pit_analysis([self._record("below_lower_bound", cdf_start=0.13)])
        assert covered["coverage_90"] == pytest.approx(1.0)
        assert covered["n_oob_interval"] == 1
        starved = numeric_pit_analysis([self._record("below_lower_bound", cdf_start=0.001)])
        assert starved["coverage_90"] == pytest.approx(0.0)

    def test_point_records_and_intervals_share_the_coverage_denominator(self):
        data = [
            self._record(50.0),  # PIT 0.50 — covered
            self._record(1.0),  # PIT 0.01 — outside [0.05, 0.95]
            self._record("above_upper_bound", cdf_end=0.87),  # interval — covered
        ]
        result = numeric_pit_analysis(data)
        assert result["count"] == 3
        assert result["n_point"] == 2
        assert result["n_oob_interval"] == 1
        assert result["coverage_90"] == pytest.approx(2 / 3)
        # The histogram (a point statistic) counts only the two point readings.
        assert sum(result["histogram"]) == 2

    def test_the_report_discloses_the_excluded_count(self):
        report = performance_analysis.generate_report(
            [self._record(50.0), self._record("above_upper_bound", cdf_end=0.87)]
        )
        assert "## Numeric Questions" in report
        assert "Out-of-range resolutions (set-valued PIT" in report
        assert "excluded from the histogram): 1" in report
