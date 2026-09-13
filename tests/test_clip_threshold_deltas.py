"""Tests for the clip-threshold sweep's delta arithmetic.

Every figure is pinned against a hand-computed ``100 * ln(new / old)`` spot-peer value on
synthetic records, so a change in the sweep's units or sign convention fails here first.
"""

from __future__ import annotations

import math

import pytest

from metaculus_bot.mc_processing import clamp_and_renormalize_probs
from metaculus_bot.performance_analysis.clip_threshold_sweep import (
    DELTA_ATOL,
    build_clip_records,
    clip_delta,
    sweep_row,
)
from metaculus_bot.performance_analysis.clip_threshold_windows import WINDOW_ALL
from tests.clip_threshold_fakes import binary_record, one_binary, one_mc


class TestTighteningArithmetic:
    """A tighter clip's spot-peer delta is exactly ``100 * ln(new / old)``."""

    def test_no_resolution_gains_when_the_floor_lifts_off_it(self):
        """A 0.05 floor moves a NO published at the in-force post-flip 0.02 floor 0.98 -> 0.95, a LOSS."""
        record = one_binary(question_id=1, p_yes=0.02, resolution=False)
        clip = clip_delta(record, 0.05, side="floor_only")
        assert clip.delta == pytest.approx(100.0 * math.log(0.95 / 0.98), abs=1e-9)
        assert clip.delta < 0.0
        assert clip.affected is True
        assert clip.censored is False

    def test_yes_resolution_gains_from_the_same_floor(self):
        record = one_binary(question_id=2, p_yes=0.02, resolution=True)
        clip = clip_delta(record, 0.05, side="floor_only")
        assert clip.delta == pytest.approx(100.0 * math.log(0.05 / 0.02), abs=1e-9)
        assert clip.delta > 0.0

    def test_unaffected_record_is_exactly_zero(self):
        record = one_binary(question_id=3, p_yes=0.30, resolution=True)
        clip = clip_delta(record, 0.05, side="floor_only")
        assert clip.delta == 0.0
        assert clip.affected is False

    def test_floor_only_leaves_the_high_side_alone(self):
        record = one_binary(question_id=4, p_yes=0.98, resolution=True)
        assert clip_delta(record, 0.05, side="floor_only").delta == 0.0
        ceiling = clip_delta(record, 0.05, side="ceiling_only")
        assert ceiling.delta == pytest.approx(100.0 * math.log(0.95 / 0.98), abs=1e-9)

    def test_symmetric_moves_both_sides(self):
        low = one_binary(question_id=5, p_yes=0.02, resolution=False)
        high = one_binary(question_id=6, p_yes=0.98, resolution=False)
        assert clip_delta(low, 0.05, side="symmetric").delta == pytest.approx(100.0 * math.log(0.95 / 0.98), abs=1e-9)
        # A NO at 0.98 gains: the ceiling pulls our (tiny) NO mass 0.02 -> 0.05.
        assert clip_delta(high, 0.05, side="symmetric").delta == pytest.approx(100.0 * math.log(0.05 / 0.02), abs=1e-9)

    def test_hits_on_clipped_side_counts_the_records_the_floor_paid_on(self):
        records = [
            one_binary(question_id=10, p_yes=0.02, resolution=True),
            one_binary(question_id=11, p_yes=0.02, resolution=False),
            one_binary(question_id=12, p_yes=0.60, resolution=True),
        ]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05)
        assert row.n == 3
        assert row.n_affected == 2
        assert row.hits_on_clipped_side == 1
        assert row.sum_delta == pytest.approx(100.0 * math.log(0.05 / 0.02) + 100.0 * math.log(0.95 / 0.98), abs=1e-9)
        assert row.mean_delta == pytest.approx(row.sum_delta / 3, abs=1e-9)

    def test_mc_floor_renormalises_and_prices_the_resolving_option(self):
        record = one_mc(
            question_id=20,
            options=["A", "B", "C", "D"],
            probs=[0.70, 0.27, 0.02, 0.01],
            resolution="A",
        )
        clip = clip_delta(record, 0.05, side="floor_only")
        expected_vector = clamp_and_renormalize_probs([0.70, 0.27, 0.02, 0.01], lo=0.05, hi=0.99)
        assert clip.delta == pytest.approx(100.0 * math.log(expected_vector[0] / 0.70), abs=1e-9)
        # Raising two floors steals mass from the leader, so the leader-resolving case loses.
        assert clip.delta < 0.0

    def test_mc_floor_pays_when_the_floored_option_resolves(self):
        record = one_mc(
            question_id=21,
            options=["A", "B", "C", "D"],
            probs=[0.70, 0.27, 0.02, 0.01],
            resolution="D",
        )
        clip = clip_delta(record, 0.05, side="floor_only")
        expected_vector = clamp_and_renormalize_probs([0.70, 0.27, 0.02, 0.01], lo=0.05, hi=0.99)
        assert clip.delta == pytest.approx(100.0 * math.log(expected_vector[3] / 0.01), abs=1e-9)
        assert clip.delta > 0.0

    def test_top1_share_is_one_when_a_single_question_carries_the_row(self):
        records = [
            one_binary(question_id=30, p_yes=0.02, resolution=True),
            one_binary(question_id=31, p_yes=0.60, resolution=True),
        ]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05)
        assert row.top1_share == pytest.approx(1.0, abs=1e-12)

    def test_top1_share_is_none_when_nothing_moved(self):
        records = [one_binary(question_id=32, p_yes=0.60, resolution=True)]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05)
        assert row.top1_share is None
        assert row.top1_question_id is None
        assert row.top1_spot_peer is None

    def test_mc_renormalisation_noise_is_not_a_driver(self):
        """A candidate looser than the clamp in force moves nothing, so it has no top1.

        The MC counterfactual runs the live clamp-and-renormalise, and a published vector
        whose floats sum to 1 + 2e-16 comes back perturbed by ~1e-14 points even when the
        bounds are exactly the ones in force. Computing a share over that noise reported a
        concentration of 0.07 and named a question the candidate never touched, on a row
        whose own n_affected was 0.
        """
        record = one_mc(
            question_id=43052,
            options=["A", "B", "C", "D"],
            probs=[0.020833333333333336, 0.03125, 0.13541666666666669, 0.8125000000000001],
            resolution="A",
        )
        row = sweep_row([record], question_type="multiple_choice", side="ceiling_only", window=WINDOW_ALL, c=0.005)
        assert row.n_affected == 0
        assert row.sum_delta != 0.0
        assert abs(row.sum_delta) < DELTA_ATOL
        assert row.top1_share is None
        assert row.top1_question_id is None
        assert row.top1_spot_peer is None

    def test_top1_names_the_driving_question(self):
        """A top1_share near 1.0 is only actionable if the row says WHICH question it is."""
        records = [
            build_clip_records(
                [
                    {
                        **binary_record(question_id=33, p_yes=0.02, resolution=True),
                        "metaculus_scores": {"spot_peer_score": -105.27},
                    }
                ],
                "binary",
            ).records[0],
            one_binary(question_id=34, p_yes=0.60, resolution=True),
        ]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05)
        assert row.top1_question_id == "33"
        assert row.top1_spot_peer == pytest.approx(-105.27)
        assert row.top1_share == pytest.approx(1.0, abs=1e-12)


class TestExpectedAndBestCase:
    """The properness cost and the insurance ceiling, pinned by hand on one record."""

    def test_binary_expected_is_minus_kl_and_best_worst_are_the_two_outcomes(self):
        record = one_binary(question_id=9010, p_yes=0.02, resolution=False)
        clip = clip_delta(record, 0.05, side="floor_only")
        gain = 100.0 * math.log(0.05 / 0.02)
        loss = 100.0 * math.log(0.95 / 0.98)
        assert clip.best_case_delta == pytest.approx(gain, abs=1e-9)
        assert clip.worst_case_delta == pytest.approx(loss, abs=1e-9)
        assert clip.expected_delta == pytest.approx(0.02 * gain + 0.98 * loss, abs=1e-9)
        assert clip.expected_delta < 0.0

    def test_unaffected_record_carries_zero_everywhere(self):
        clip = clip_delta(one_binary(question_id=9011, p_yes=0.30, resolution=True), 0.05, side="floor_only")
        assert (clip.expected_delta, clip.best_case_delta, clip.worst_case_delta) == (0.0, 0.0, 0.0)

    def test_mc_expected_is_never_positive(self):
        record = one_mc(question_id=9012, options=["A", "B", "C", "D"], probs=[0.70, 0.27, 0.02, 0.01], resolution="A")
        clip = clip_delta(record, 0.05, side="floor_only")
        assert clip.expected_delta < 0.0
        assert clip.best_case_delta > 0.0 > clip.worst_case_delta

    def test_row_sums_the_per_record_quantities(self):
        records = [
            one_binary(question_id=9013, p_yes=0.02, resolution=False),
            one_binary(question_id=9014, p_yes=0.02, resolution=True),
        ]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.05)
        gain = 100.0 * math.log(0.05 / 0.02)
        loss = 100.0 * math.log(0.95 / 0.98)
        assert row.best_case_sum_delta == pytest.approx(2 * gain, abs=1e-9)
        assert row.worst_case_sum_delta == pytest.approx(2 * loss, abs=1e-9)
        assert row.expected_sum_delta == pytest.approx(2 * (0.02 * gain + 0.98 * loss), abs=1e-9)


class TestFloorFeasibility:
    """An MC floor with more options than ``1 / c`` cannot be delivered, and the row says so."""

    def test_eleven_options_cannot_take_a_tenth_floor(self):
        options = [f"O{i}" for i in range(11)]
        probs = [0.30, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01]
        record = one_mc(question_id=9160, options=options, probs=probs, resolution="O0")
        assert clip_delta(record, 0.10, side="floor_only").infeasible is True
        assert clip_delta(record, 0.05, side="floor_only").infeasible is False
        row = sweep_row([record], question_type="multiple_choice", side="floor_only", window=WINDOW_ALL, c=0.10)
        # Priced at the live clamp's sub-floor fallback, still counted as moved, and disclosed.
        assert (row.infeasible_n, row.n_affected) == (1, 1)
        assert (
            sweep_row(
                [record], question_type="multiple_choice", side="floor_only", window=WINDOW_ALL, c=0.05
            ).infeasible_n
            == 0
        )

    def test_ten_options_at_a_tenth_are_exactly_feasible(self):
        options = [f"O{i}" for i in range(10)]
        probs = [0.30, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.03, 0.02]
        record = one_mc(question_id=9162, options=options, probs=probs, resolution="O0")
        clip = clip_delta(record, 0.10, side="floor_only")
        assert clip.infeasible is False
        # And the counterfactual really is the 0.10 floor: the uniform vector, not a sub-floor one.
        assert clip.delta == pytest.approx(100.0 * math.log(0.10 / 0.30), abs=1e-9)

    def test_binary_rows_are_never_infeasible(self):
        row = sweep_row(
            [one_binary(question_id=9161, p_yes=0.02, resolution=False)],
            question_type="binary",
            side="floor_only",
            window=WINDOW_ALL,
            c=0.10,
        )
        assert row.infeasible_n == 0
