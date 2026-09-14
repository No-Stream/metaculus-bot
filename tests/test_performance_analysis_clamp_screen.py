"""Tests for ``max_step_clamp_screen``: did the per-bin max-step cap override the ensemble?

The screen answers one question per numeric or discrete record: was the published mass on the
realized bin pinned at the era-correct max-step cap while the members' own declared curves wanted
materially more there. These tests pin the era boundary, the near-cap ratio, the bin-edge
convention, and every way a member curve is ruled unusable.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from metaculus_bot.performance_analysis.analysis import max_step_clamp_screen


class TestMaxStepClampScreen:
    """The q43913 signature: a published bin pinned at the per-bin max-step cap while
    every member's own declared curve wanted materially more mass there.

    The cap is era-correct — flat 0.2 before the grid-scaled cap reached main (b4e9df0),
    the grid's own ``grid_step_constraints`` max after — so a post-fix coarse-grid
    discrete that legitimately holds a 0.2 bin must NOT fire. The fixture member curves
    are 11-ANCHOR on purpose: the screen drops any member under MIN_SCOREABLE_ANCHORS,
    because its verdict turns on the MINIMUM member bin mass and a 3-anchor interpolation
    across one bin is not the distribution the model declared.
    """

    # 11-point integer grid; steps[1] (the [1, 2] bin) is exactly 0.20.
    _GRID: ClassVar[list[float]] = [float(v) for v in range(11)]
    _CDF: ClassVar[list[float]] = [0.0, 0.05, 0.25, 0.45, 0.65, 0.85, 0.90, 0.93, 0.96, 0.98, 1.0]
    # Both members concentrate ~0.70 of their mass on the [1, 2] bin.
    _LABELS: ClassVar[list[float]] = [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0]
    _MEMBERS: ClassVar[dict[str, list[list[float]]]] = {
        "model-a": [
            [label, value]
            for label, value in zip(
                _LABELS,
                [0.90, 1.00, 1.15, 1.30, 1.45, 1.55, 1.70, 1.85, 2.00, 2.30, 2.60],
                strict=True,
            )
        ],
        "model-b": [
            [label, value]
            for label, value in zip(
                _LABELS,
                [0.95, 1.05, 1.18, 1.32, 1.46, 1.56, 1.72, 1.90, 2.05, 2.35, 2.65],
                strict=True,
            )
        ],
    }
    _PRE_FIX_TS = "2026-06-11T00:00:00Z"
    _POST_FIX_TS = "2026-08-01T00:00:00Z"

    def _record(
        self, *, submitted, members=None, cdf=None, grid=None, resolution: float | str = 1.4, q_type="discrete"
    ) -> dict:
        grid = grid if grid is not None else self._GRID
        return {
            "type": q_type,
            "our_forecast_values": cdf if cdf is not None else self._CDF,
            "resolution_parsed": resolution,
            "scaling": {"range_min": grid[0], "range_max": grid[-1], "continuous_range": grid},
            "bot_comment_created_at": submitted,
            "per_model_numeric_percentiles": members if members is not None else self._MEMBERS,
        }

    def test_pre_fix_coarse_grid_clamp_is_suspected(self):
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS))
        assert screen["suspected"] is True
        assert screen["submitted_before_grid_scaled_cap"] is True
        assert screen["max_step_cap"] == pytest.approx(0.2)
        assert screen["published_bin_mass"] == pytest.approx(0.2, abs=1e-9)
        assert screen["min_member_bin_mass"] > 0.6

    def test_post_fix_coarse_grid_point_two_bin_does_not_fire(self):
        """After 9f1175c an 11-point grid's cap is 1.0, so a 0.2 bin means nothing."""
        screen = max_step_clamp_screen(self._record(submitted=self._POST_FIX_TS))
        assert screen["suspected"] is False
        assert screen["submitted_before_grid_scaled_cap"] is False
        assert screen["max_step_cap"] == pytest.approx(1.0)
        assert screen["resolution_bin_at_cap"] is False

    def test_post_fix_standard_grid_cap_still_fires(self):
        """On the 201-point grid the era-correct cap is still 0.2, so the screen keeps catching
        genuine clamps after the fix."""
        steps = np.full(200, 0.8 / 199)
        steps[100] = 0.2
        cdf = np.concatenate([[0.0], np.cumsum(steps)]).tolist()
        grid = np.linspace(0.0, 200.0, 201).tolist()
        # 11-anchor curves (see _MEMBERS): each puts ~0.70 on the [100, 101] bin.
        members = {
            "model-a": [
                [label, value]
                for label, value in zip(
                    self._LABELS,
                    [99.90, 100.00, 100.15, 100.30, 100.45, 100.55, 100.70, 100.85, 101.00, 101.30, 101.60],
                    strict=True,
                )
            ],
            "model-b": [
                [label, value]
                for label, value in zip(
                    self._LABELS,
                    [99.95, 100.05, 100.18, 100.32, 100.46, 100.56, 100.72, 100.90, 101.05, 101.35, 101.65],
                    strict=True,
                )
            ],
        }
        screen = max_step_clamp_screen(
            self._record(submitted=self._POST_FIX_TS, members=members, cdf=cdf, grid=grid, resolution=100.5)
        )
        assert screen["max_step_cap"] == pytest.approx(0.2)
        assert screen["suspected"] is True

    def _fine_grid_members(self) -> dict[str, list[list[float]]]:
        """Three 11-anchor curves each putting ~0.65-0.77 on the [100, 101] bin."""
        value_rows = [
            [99.90, 100.00, 100.15, 100.30, 100.45, 100.55, 100.70, 100.85, 101.00, 101.30, 101.60],
            [99.95, 100.05, 100.18, 100.32, 100.46, 100.56, 100.72, 100.90, 101.05, 101.35, 101.65],
            [99.85, 99.98, 100.12, 100.28, 100.43, 100.53, 100.68, 100.83, 100.98, 101.28, 101.55],
        ]
        return {
            f"model-{name}": [[label, value] for label, value in zip(self._LABELS, values, strict=True)]
            for name, values in zip("abc", value_rows, strict=True)
        }

    def _fine_grid_record(self, realized_bin_mass: float) -> dict:
        steps = np.full(200, (1.0 - realized_bin_mass) / 199)
        steps[100] = realized_bin_mass
        cdf = np.concatenate([[0.0], np.cumsum(steps)]).tolist()
        grid = np.linspace(0.0, 200.0, 201).tolist()
        return self._record(
            submitted=self._POST_FIX_TS, members=self._fine_grid_members(), cdf=cdf, grid=grid, resolution=100.5
        )

    def test_post_snap_near_cap_bin_is_suspected(self):
        """The q45065 shape: the snap alpha shaves the realized bin ~1.1% under the 0.2
        cap (0.1977991526), so the exact-equality screen read it as clear while all
        three members declared 0.65-0.77 there. The near-cap ratio must catch it."""
        screen = max_step_clamp_screen(self._fine_grid_record(0.1977991526))
        assert screen["suspected"] is True
        assert screen["resolution_bin_at_cap"] is False
        assert screen["resolution_bin_cap_bound"] is True
        assert screen["resolution_bin_cap_fraction"] == pytest.approx(0.989, abs=1e-3)

    def test_bin_well_below_the_cap_is_not_cap_bound(self):
        """0.15 against a 0.2 cap is 75%, below ``_CLAMP_CAP_NEAR_FRAC``, so the bin is not
        cap-bound even though every member wanted materially more mass there."""
        screen = max_step_clamp_screen(self._fine_grid_record(0.15))
        assert screen["resolution_bin_cap_bound"] is False
        assert screen["suspected"] is False

    def test_missing_timestamp_treated_as_pre_fix(self):
        screen = max_step_clamp_screen(self._record(submitted=None))
        assert screen["submitted_before_grid_scaled_cap"] is True
        assert screen["max_step_cap"] == pytest.approx(0.2)

    def test_unparseable_timestamp_treated_as_pre_fix(self):
        """Same rule as a missing timestamp: the undated (and undatable) archive records all
        predate the fix, so an unreadable timestamp must not be read as post-fix."""
        screen = max_step_clamp_screen(self._record(submitted="not-a-date"))
        assert screen["submitted_before_grid_scaled_cap"] is True
        assert screen["max_step_cap"] == pytest.approx(0.2)

    def test_timestamp_with_an_offset_is_compared_in_utc(self):
        """A post-fix instant written with a local offset must read as post-fix, because a naive
        offset-dropping comparison shifts it hours across the era boundary."""
        screen = max_step_clamp_screen(self._record(submitted="2026-07-21T11:07:37-07:00"))
        assert screen["submitted_before_grid_scaled_cap"] is False

    def test_members_not_materially_more_does_not_fire(self):
        """The members' own curves put ~0.2 on the bin too, so the cap coincided with what the
        ensemble wanted and nothing was overridden."""
        diffuse = {
            "model-a": [[10.0, 0.0], [50.0, 3.0], [90.0, 8.0]],
            "model-b": [[10.0, 0.5], [50.0, 3.5], [90.0, 8.5]],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=diffuse))
        assert screen["resolution_bin_at_cap"] is True
        assert screen["suspected"] is False

    def test_single_member_curve_does_not_fire(self):
        one = {"model-a": self._MEMBERS["model-a"]}
        assert max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=one))["suspected"] is False

    def test_anonymous_member_keys_are_excluded(self):
        """A positional key on a stacked record can hold the stacker's aggregate."""
        anon = {"Forecaster 1": self._MEMBERS["model-a"], "Forecaster 2": self._MEMBERS["model-b"]}
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=anon))
        assert screen["member_bin_masses"] == {}
        assert screen["suspected"] is False

    def test_resolution_exactly_on_grid_point_screens_the_bin_below(self):
        """A resolution sitting exactly ON a grid edge belongs to the bin BELOW it, which is the
        platform scorer's convention in ``resolution_to_bucket_index``.

        On this fixture resolution 2.0 must screen the [1, 2] bin whose 0.20 step sits at the
        pre-fix cap; the old ``side="right"`` screened [2, 3] and missed the q43913 signature
        entirely.
        """
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, resolution=2.0))
        assert screen["resolution_bin"] == [1.0, 2.0]
        assert screen["published_bin_mass"] == pytest.approx(0.2, abs=1e-9)
        assert screen["suspected"] is True

    def test_single_pair_member_curve_is_unusable(self):
        """One recovered (percentile, value) pair interpolates to a constant PIT at every
        resolution, so the member is dropped, leaving one usable curve, which is below the
        two-curve minimum the screen requires."""
        one_pair = {
            "model-a": [[50.0, 90.0]],
            "model-b": self._MEMBERS["model-b"],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=one_pair))
        assert list(screen["member_bin_masses"]) == ["model-b"]
        assert screen["suspected"] is False

    def test_a_sparse_member_curve_is_excluded_from_the_min(self):
        """The verdict turns on the MINIMUM member bin mass, so one sparse recovery can
        decide it — and a 3-anchor interpolation across one bin is not the distribution the
        model declared. q43913's KNOWN_BUG_QIDS entry survives this gate on its own
        11-anchor member; the 3-anchor sibling never decided that verdict."""
        mixed = {
            "model-a": self._MEMBERS["model-a"],
            "model-sparse": [[10.0, 0.0], [50.0, 3.0], [90.0, 8.0]],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=mixed))

        assert list(screen["member_bin_masses"]) == ["model-a"]
        # One usable curve is below the >=2-curves requirement, so nothing is suspected.
        assert screen["suspected"] is False

    def test_a_uniformly_sparse_record_reports_no_member_masses(self):
        """The sparse-era shape: no curve clears the anchor floor, so the screen has no member
        evidence at all rather than ranking equals against each other. A bin-mass comparison is
        absolute, unlike ``ranking_cohort``'s relative one."""
        sparse = {
            "model-a": [[10.0, 0.9], [50.0, 1.3], [90.0, 2.1]],
            "model-b": [[10.0, 0.95], [50.0, 1.4], [90.0, 2.2]],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=sparse))

        assert screen["member_bin_masses"] == {}
        assert screen["min_member_bin_mass"] is None
        assert screen["suspected"] is False

    def test_non_monotonic_member_curve_is_unusable(self):
        """Percentiles that DECREASE as values increase invert the curve, so it is dropped."""
        inverted = {
            "model-a": [[90.0, 0.9], [50.0, 1.3], [10.0, 2.1]],
            "model-b": self._MEMBERS["model-b"],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=inverted))
        assert list(screen["member_bin_masses"]) == ["model-b"]

    def test_non_numeric_resolution_and_type_gates(self):
        assert max_step_clamp_screen({"type": "binary"})["applicable"] is False
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, resolution="below_lower_bound"))
        assert screen["suspected"] is False
        assert screen["reason"] == "non-numeric resolution"

    def test_records_without_a_usable_grid_report_that_reason(self):
        """The screen needs the question's own value grid to locate the realized bin.

        Comment-backfilled records often carry no continuous_range, and a grid whose length
        disagrees with the published CDF cannot be indexed either, so both must report a reason
        instead of screening an arbitrary bin.
        """
        no_grid = self._record(submitted=self._PRE_FIX_TS)
        no_grid["scaling"] = {"range_min": 0.0, "range_max": 10.0}
        assert max_step_clamp_screen(no_grid)["reason"] == "no usable grid"

        mismatched = self._record(submitted=self._PRE_FIX_TS, grid=[0.0, 1.0, 2.0])
        assert max_step_clamp_screen(mismatched)["reason"] == "no usable grid"
        assert max_step_clamp_screen(mismatched)["suspected"] is False
