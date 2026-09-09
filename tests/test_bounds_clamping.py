"""The closed-bound clamp (``metaculus_bot/numeric/bounds_clamping.py``).

Two numbers govern it and they are deliberately different. ``calculate_bounds_buffer`` is the
accept-or-raise TOLERANCE: how far outside a closed bound a declared value may sit and still be
clamped in rather than dropping the forecaster. ``minimum_separation`` (``numeric/config.py``) is
where an accepted value LANDS: just inside the bound, the same standoff the cluster spreader and
the strict-ordering passes keep. Landing a value one tolerance inside instead moved it a whole bin
on a coarse grid, and the left-to-right strict-ordering pass then dragged every percentile declared
inside that first bin up with it, so the per-grid-shape pins at the bottom run the whole sanitizer
and assert that a clamped value stays inside its terminal bin and no in-range percentile moves.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric.bounds_clamping import (
    calculate_bounds_buffer,
    clamp_values_to_bounds,
    log_cluster_spreading_summary,
    log_corrections_summary,
    log_heavy_clamping_diagnostics,
)
from metaculus_bot.numeric.config import STANDARD_PERCENTILES, grid_bin_width, minimum_separation
from metaculus_bot.numeric.pipeline import sanitize_percentiles
from tests.pipeline_test_helpers import make_real_numeric_question

_DAY = 86_400.0


def _make_question(open_upper=False, open_lower=False, lower=0.0, upper=100.0, cdf_size=201) -> NumericQuestion:
    return cast(
        NumericQuestion,
        SimpleNamespace(
            open_upper_bound=open_upper,
            open_lower_bound=open_lower,
            upper_bound=upper,
            lower_bound=lower,
            cdf_size=cdf_size,
            id_of_question=999,
            page_url="https://example.com/q/999",
        ),
    )


def _inset(question: NumericQuestion) -> float:
    return minimum_separation(question.upper_bound - question.lower_bound)


class TestCalculateBoundsBuffer:
    def test_large_range_is_the_flat_tolerance(self):
        assert calculate_bounds_buffer(_make_question(lower=0.0, upper=200.0)) == 1.0

    def test_small_range_is_one_percent_of_range(self):
        assert calculate_bounds_buffer(_make_question(lower=0.0, upper=50.0)) == 50.0 * 0.01

    def test_is_at_least_one_grid_bin(self):
        """A coarse grid widens the tolerance to one bin: a value inside one bin of a closed edge
        is indistinguishable from the edge once bucketed, so it clamps instead of dropping the
        forecaster. Twelve one-day bins on an epoch-seconds axis is the case that motivated it,
        where the flat 1.0 was a one-second tolerance."""
        question = _make_question(lower=0.0, upper=12 * _DAY, cdf_size=13)
        assert calculate_bounds_buffer(question) == _DAY
        # Whenever the range-based tolerance is the larger, nothing changes.
        assert calculate_bounds_buffer(_make_question(lower=0.0, upper=200.0)) == 1.0
        assert calculate_bounds_buffer(_make_question(lower=0.0, upper=50.0)) == 0.5

    def test_the_bin_floor_binds_on_the_201_grid_too_once_the_range_exceeds_200(self):
        """No scoping branch: the rationale holds on the continuous grid, and the floor removes
        the old discontinuity where the flat 1.0 was 1% of a 100-wide range and 0.005% of a
        20,000-wide one. On a Metaculus question with range > 200 the drop-versus-clamp
        threshold is range / 200 rather than 1.0; it changes only that threshold, never where
        the clamped value lands."""
        for lower, upper in ((0.0, 1000.0), (0.0, 20000.0), (-500.0, 4500.0)):
            question = _make_question(lower=lower, upper=upper)
            assert calculate_bounds_buffer(question) == (upper - lower) / 200
            assert calculate_bounds_buffer(question) == grid_bin_width(lower, upper, 201)


class TestClampValuesToBounds:
    def test_no_violations(self):
        values = [10.0, 20.0, 30.0]
        percentiles = _percentiles(values)
        question = _make_question(lower=0.0, upper=100.0)

        result, corrections_made = clamp_values_to_bounds(values.copy(), percentiles, question, 1.0)

        assert not corrections_made
        assert result == values

    def test_lower_violation_within_tolerance_lands_just_inside_the_bound(self):
        values = [-0.5, 20.0, 30.0]
        question = _make_question(lower=0.0, upper=100.0)

        result, corrections_made = clamp_values_to_bounds(values.copy(), _percentiles(values), question, 1.0)

        assert corrections_made
        assert result[0] == question.lower_bound + _inset(question)
        assert question.lower_bound < result[0] < question.lower_bound + 1.0
        assert result[1:] == values[1:]

    def test_upper_violation_within_tolerance_lands_just_inside_the_bound(self):
        values = [10.0, 20.0, 100.5]
        question = _make_question(upper=100.0)

        result, corrections_made = clamp_values_to_bounds(values.copy(), _percentiles(values), question, 1.0)

        assert corrections_made
        assert result[0:2] == values[0:2]
        assert result[2] == question.upper_bound - _inset(question)
        assert question.upper_bound - 1.0 < result[2] < question.upper_bound

    def test_violation_exceeding_the_tolerance_raises(self):
        values = [-5.0, 20.0, 30.0]
        question = _make_question(lower=0.0, upper=100.0)

        with pytest.raises(ValueError, match="too far below lower bound"):
            clamp_values_to_bounds(values.copy(), _percentiles(values), question, 1.0)

    def test_a_value_one_bin_outside_a_wide_201_grid_clamps_instead_of_raising(self):
        """[0, 20000] has 100-wide bins, so -40 is within the tolerance (it raised under the
        flat 1.0) and lands just inside 0, not at 100."""
        question = _make_question(lower=0.0, upper=20000.0)
        values = [-40.0, 5.0, 20.0]
        buffer = calculate_bounds_buffer(question)

        result, corrections_made = clamp_values_to_bounds(values.copy(), _percentiles(values), question, buffer)

        assert corrections_made
        assert result[0] == _inset(question)
        assert result[1:] == [5.0, 20.0]
        with pytest.raises(ValueError, match="too far below lower bound"):
            clamp_values_to_bounds([-101.0, 5.0, 20.0], _percentiles(values), question, buffer)

    def test_open_bounds_are_never_clamped(self):
        values = [-1.0, 20.0, 101.0]
        question = _make_question(open_lower=True, open_upper=True, lower=0.0, upper=100.0)

        result, corrections_made = clamp_values_to_bounds(values.copy(), _percentiles(values), question, 1.0)

        assert not corrections_made
        assert result == values


class TestHeavyClampingDiagnostics:
    """The WARNING fires when more than half the values sit where the clamp puts them, at the
    inset. Measured against the inset rather than the tolerance: with the tolerance as the window
    (a whole day on a 13-point date grid) a fully in-range forecast concentrated near a bound
    logged as heavily clamped, in exactly the logs the residual analysis reads."""

    def test_light_clamping_is_silent(self, caplog):
        question = _make_question(lower=0.0, upper=100.0)
        modified_values = [question.lower_bound + _inset(question), 20.0, 30.0]

        caplog.clear()
        log_heavy_clamping_diagnostics(modified_values, [0.0, 20.0, 30.0], question)

        assert not any("Heavy bound clamping" in record.message for record in caplog.records)

    def test_heavy_lower_clamping_warns(self, caplog):
        question = _make_question(lower=0.0, upper=100.0)
        at_inset = question.lower_bound + _inset(question)
        modified_values = [at_inset, at_inset, 30.0]

        caplog.clear()
        caplog.set_level("WARNING")
        log_heavy_clamping_diagnostics(modified_values, [-0.5, -0.2, 30.0], question)

        (record,) = [r for r in caplog.records if "Heavy bound clamping" in r.message]
        assert "clamped_to_lower=66%" in record.getMessage()
        assert "Q 999" in record.getMessage()

    def test_an_in_range_forecast_inside_the_terminal_bin_is_not_clamping(self, caplog):
        """Three values in the first day of a 12-day grid, none of them clamped: one tolerance
        (a day) as the window counted all three as clamped and warned on a healthy forecast."""
        question = _make_question(lower=0.0, upper=12 * _DAY, cdf_size=13)
        values = [0.25 * _DAY, 0.5 * _DAY, 0.75 * _DAY]

        caplog.clear()
        caplog.set_level("WARNING")
        log_heavy_clamping_diagnostics(values, values, question)

        assert not any("Heavy bound clamping" in record.message for record in caplog.records)


class TestSummaries:
    def test_log_corrections_summary_with_corrections(self, caplog):
        caplog.clear()
        caplog.set_level("WARNING")
        log_corrections_summary([1.0, 20.0, 30.0], [0.0, 20.0, 30.0], _make_question(), corrections_made=True)

        assert any("Corrected numeric distribution for question 999" in r.message for r in caplog.records)

    def test_log_corrections_summary_no_corrections(self, caplog):
        values = [10.0, 20.0, 30.0]

        caplog.clear()
        log_corrections_summary(values, values, _make_question(), corrections_made=False)

        assert not any("Corrected numeric distribution" in record.message for record in caplog.records)

    def test_log_cluster_spreading_summary_with_clusters(self, caplog):
        caplog.clear()
        caplog.set_level("WARNING")
        log_cluster_spreading_summary(
            [10.0, 20.1, 20.2, 30.0],
            [10.0, 20.0, 20.0, 30.0],
            _make_question(),
            clusters_applied=1,
            spread_delta=0.1,
            count_like=False,
        )

        assert any("Cluster spread applied for Q 999" in record.message for record in caplog.records)

    def test_log_cluster_spreading_summary_no_clusters(self, caplog):
        values = [10.0, 20.0, 30.0]

        caplog.clear()
        log_cluster_spreading_summary(
            values, values, _make_question(), clusters_applied=0, spread_delta=0.1, count_like=False
        )

        assert not any("Cluster spread applied" in record.message for record in caplog.records)


class TestAClampedValueStaysInItsTerminalBin:
    """Through the whole sanitizer, one pin per grid shape.

    Each declares one percentile just outside a closed bound (inside the tolerance) with the
    neighbouring percentiles declared INSIDE the terminal bin. The clamped value must land in that
    bin and the in-range values must publish exactly as declared. Landing the clamped value one
    tolerance inside put it a whole bin in, and ``ensure_strictly_increasing_bounded`` then dragged
    the in-range neighbours up behind it: on post 651 that moved 6.3 points of mass out of the day
    the member declared, on the 22-point grid 20.7 landed on the bin 19/20 edge, and on a
    [0, 20000] Metaculus question four percentiles published at 100.0 when three were in range.
    """

    @staticmethod
    def _sanitized_values(question: NumericQuestion, values: list[float]) -> list[float]:
        declared = [Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, values, strict=True)]
        sanitized, _zero_point = sanitize_percentiles(declared, question)
        return [p.value for p in sanitized]

    def test_thirteen_point_date_grid_closed_both_sides(self):
        question = make_real_numeric_question(lower_bound=0.0, upper_bound=12 * _DAY, open_upper_bound=False)
        question = question.model_copy(update={"cdf_size": 13})
        first_day = [0.25 * _DAY, 0.5 * _DAY, 0.75 * _DAY]
        later = [
            1.5 * _DAY,
            2.5 * _DAY,
            3.5 * _DAY,
            4.5 * _DAY,
            5.5 * _DAY,
            6.5 * _DAY,
            7.5 * _DAY,
            8.5 * _DAY,
            9.5 * _DAY,
        ]
        declared = [-0.5 * _DAY, *first_day, *later]

        result = self._sanitized_values(question, declared)

        assert 0.0 < result[0] < first_day[0]
        assert result[0] < grid_bin_width(0.0, 12 * _DAY, 13)
        assert result[1:] == declared[1:]

    def test_twenty_two_point_discrete_grid(self):
        """Bins centred on 0..20 over [-0.5, 20.5]; the top bin is (19.5, 20.5]. A declared 20.7
        must stay in it, not land on its 19.5 edge, which right-closed bucketing scores as 19."""
        question = make_real_numeric_question(lower_bound=-0.5, upper_bound=20.5, open_upper_bound=False)
        question = question.model_copy(update={"cdf_size": 22})
        in_range = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 19.0, 20.0]
        declared = [*in_range, 20.7]

        result = self._sanitized_values(question, declared)

        assert 20.0 < result[-1] < 20.5
        assert result[:-1] == in_range

    def test_two_hundred_and_one_point_grid_over_a_wide_range(self):
        question = make_real_numeric_question(lower_bound=0.0, upper_bound=20000.0, open_upper_bound=False)
        in_range = [5.0, 20.0, 60.0, 500.0, 1000.0, 3000.0, 5000.0, 8000.0, 12000.0, 15000.0, 18000.0, 19000.0]
        declared = [-40.0, *in_range]

        result = self._sanitized_values(question, declared)

        assert 0.0 < result[0] < in_range[0]
        assert result[0] < grid_bin_width(0.0, 20000.0, 201)
        assert result[1:] == in_range


def _percentiles(values: list[float]) -> list[Percentile]:
    labels = [0.10, 0.20, 0.30]
    return [Percentile(percentile=label, value=value) for label, value in zip(labels, values, strict=True)]
