"""Tests for censoring: below the clamp in force the raw member value is unobservable.

A candidate LOOSER than the clamp that was live cannot be priced, only bracketed, because the
clamp destroyed the raw value. The published-value rule and the member-position rule are both here.
"""

from __future__ import annotations

import math

import pytest

from metaculus_bot.performance_analysis.clip_threshold_sweep import clip_delta, member_censored, sweep_row
from metaculus_bot.performance_analysis.clip_threshold_windows import WINDOW_ALL
from tests.clip_threshold_fakes import BEFORE_WIDENING, one_binary, one_mc


class TestLoosenBounds:
    """Below the in-force floor the raw member value is unobservable."""

    def test_record_at_the_floor_is_censored_and_bounded(self):
        record = one_binary(question_id=40, p_yes=0.02, resolution=False)
        clip = clip_delta(record, 0.01, side="floor_only")
        assert clip.censored is True
        assert clip.affected is False
        assert clip.delta == 0.0
        # Lower bound: raw was exactly the floor, nothing moves. Upper: raw was <= 0.01.
        assert clip.loosen_at_c == pytest.approx(100.0 * math.log(0.99 / 0.98), abs=1e-9)

    def test_censored_yes_loses_under_the_upper_scenario(self):
        record = one_binary(question_id=41, p_yes=0.02, resolution=True)
        clip = clip_delta(record, 0.01, side="floor_only")
        assert clip.censored is True
        assert clip.loosen_at_c == pytest.approx(100.0 * math.log(0.01 / 0.02), abs=1e-9)
        assert clip.loosen_at_c < 0.0

    def test_record_above_the_floor_contributes_zero_to_both_bounds(self):
        record = one_binary(question_id=42, p_yes=0.30, resolution=False)
        clip = clip_delta(record, 0.01, side="floor_only")
        assert clip.censored is False
        assert clip.delta == 0.0
        assert clip.loosen_at_c == 0.0

    def test_row_reports_both_bounds_and_the_censored_count(self):
        """The identified bracket takes each censored record's own best and worst case, which is why
        it is wider than the two named scenarios when the signs disagree.
        """
        records = [
            one_binary(question_id=43, p_yes=0.02, resolution=False),
            one_binary(question_id=44, p_yes=0.02, resolution=True),
            one_binary(question_id=45, p_yes=0.30, resolution=False),
        ]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.01)
        assert row.censored_n == 2
        assert row.n_loosening == 3
        assert row.sum_delta == 0.0
        assert row.sum_delta_lower == 0.0
        gain = 100.0 * math.log(0.99 / 0.98)
        loss = 100.0 * math.log(0.01 / 0.02)
        assert row.sum_delta_upper == pytest.approx(gain + loss, abs=1e-9)
        assert row.bracket_lo == pytest.approx(loss, abs=1e-9)
        assert row.bracket_hi == pytest.approx(gain, abs=1e-9)

    def test_pre_flip_record_tightens_at_a_candidate_above_its_own_floor(self):
        """In-force floor 0.01 before the widening flip, so c = 0.015 TIGHTENS this record even
        though it sits at its own floor: a tighter clip is exact regardless of the raw value.
        """
        record = one_binary(question_id=46, p_yes=0.01, resolution=False, created_at=BEFORE_WIDENING)
        clip = clip_delta(record, 0.015, side="floor_only")
        assert clip.censored is False
        assert clip.delta == pytest.approx(100.0 * math.log(0.985 / 0.99), abs=1e-9)

    def test_mc_option_at_the_floor_censors_the_record(self):
        record = one_mc(
            question_id=47,
            options=["A", "B", "C"],
            probs=[0.88, 0.11, 0.01],
            resolution="A",
        )
        clip = clip_delta(record, 0.005, side="floor_only")
        assert clip.censored is True
        assert clip.delta == 0.0
        assert clip.loosen_at_c > 0.0

    def test_mc_record_clear_of_the_floor_is_not_censored(self):
        record = one_mc(
            question_id=48,
            options=["A", "B", "C"],
            probs=[0.50, 0.30, 0.20],
            resolution="A",
        )
        clip = clip_delta(record, 0.005, side="floor_only")
        assert clip.censored is False
        assert clip.loosen_at_c == 0.0

    def test_ceiling_only_loosening_is_counted_on_the_ceiling(self):
        """A publish at the in-force 0.98 ceiling swept on the ceiling side at c = 0.005 asks for
        a 0.995 ceiling: looser than the one in force, so the record is censored AND a loosening
        record. ``n_loosening`` used to inspect only the floor and read 0 on every ceiling-only
        row while ``censored_n`` on the same row read 1.

        The same candidate on the floor side is a loosening too (0.005 < 0.02), but this publish
        sits nowhere near the floor, so it is counted as loosening and NOT as censored.
        """
        record = one_binary(question_id=49, p_yes=0.98, resolution=True)
        row = sweep_row([record], question_type="binary", side="ceiling_only", window=WINDOW_ALL, c=0.005)
        assert row.censored_n == 1
        assert row.n_loosening == 1
        assert row.sum_delta == 0.0
        assert row.exact is False
        assert row.sum_delta_upper == pytest.approx(100.0 * math.log(0.995 / 0.98), abs=1e-9)
        floor_side = sweep_row([record], question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.005)
        assert (floor_side.n_loosening, floor_side.censored_n) == (1, 0)


class TestMemberCensoring:
    """A clamped MEMBER in a median position censors a looser clip even when the publish is above the floor."""

    def test_even_roster_with_a_floored_middle_member_is_member_censored(self):
        """Members 0.02 / 0.03 publish 0.025 post-flip: above the 0.02 floor, so the published
        rule says nothing is censored, yet the 0.02 member is one of the two middle values and a
        looser floor could have moved the publish."""
        record = one_binary(
            question_id=9020,
            p_yes=0.025,
            resolution=False,
            per_model={"a": "2.0%", "b": "3.0%"},
        )
        clip = clip_delta(record, 0.01, side="floor_only")
        assert clip.censored is False
        assert clip.member_censored is True
        # Member replay: median(0.01, 0.03) = 0.02 against the baseline median 0.025, on a NO.
        assert clip.loosen_members_at_c == pytest.approx(100.0 * math.log(0.98 / 0.975), abs=1e-9)
        assert clip.loosen_at_c == 0.0

    def test_floored_member_outside_the_median_positions_cannot_move_the_publish(self):
        record = one_binary(
            question_id=9021,
            p_yes=0.06,
            resolution=False,
            per_model={"a": "2.0%", "b": "6.0%", "c": "10.0%"},
        )
        assert member_censored(record, floor_side=True, ceiling_side=False) is False
        clip = clip_delta(record, 0.01, side="floor_only")
        assert clip.member_censored is False
        assert clip.loosen_members_at_c == 0.0

    def test_mean_aggregator_makes_every_floored_member_censoring(self):
        """Members 0.02 / 0.10 / 0.30 published as their mean 0.14, whose median would be 0.10."""
        record = one_binary(
            question_id=9022,
            p_yes=0.14,
            resolution=False,
            per_model={"a": "2.0%", "b": "10.0%", "c": "30.0%"},
        )
        assert record.aggregator == "mean"
        assert member_censored(record, floor_side=True, ceiling_side=False) is True

    def test_no_members_falls_back_to_the_published_rule(self):
        at_floor = one_binary(question_id=9023, p_yes=0.02, resolution=False)
        above = one_binary(question_id=9024, p_yes=0.30, resolution=False)
        assert member_censored(at_floor, floor_side=True, ceiling_side=False) is True
        assert member_censored(above, floor_side=True, ceiling_side=False) is False
        clip = clip_delta(at_floor, 0.01, side="floor_only")
        assert clip.member_censored is True
        assert clip.loosen_members_at_c == pytest.approx(clip.loosen_at_c, abs=1e-12)

    def test_unknown_aggregator_falls_back_to_the_published_rule(self):
        """Members whose median and mean both miss the publish say nothing about member positions."""
        at_floor = one_binary(
            question_id=9029, p_yes=0.02, resolution=False, per_model={"a": "10.0%", "b": "30.0%", "c": "80.0%"}
        )
        assert at_floor.aggregator == "unknown"
        assert member_censored(at_floor, floor_side=True, ceiling_side=False) is True
        clip = clip_delta(at_floor, 0.01, side="floor_only")
        assert clip.member_censored is True
        assert clip.loosen_members_at_c == pytest.approx(clip.loosen_at_c, abs=1e-12)

    def test_row_counts_both_rules_and_the_member_bound_is_never_narrower(self):
        records = [
            one_binary(question_id=9025, p_yes=0.02, resolution=False),
            one_binary(question_id=9026, p_yes=0.025, resolution=False, per_model={"a": "2.0%", "b": "3.0%"}),
            one_binary(question_id=9027, p_yes=0.30, resolution=False),
        ]
        row = sweep_row(records, question_type="binary", side="floor_only", window=WINDOW_ALL, c=0.01)
        assert row.censored_n == 1
        assert row.member_censored_n == 2
        assert row.sum_delta_upper_members > row.sum_delta_upper > 0.0

    def test_mc_even_roster_with_a_floored_middle_option_is_member_censored(self):
        """Two MC ballots put option C at 0.01 and 0.03; the per-option median publishes 0.02,
        above the 0.01 floor, so the published rule sees nothing while the member rule does."""
        record = one_mc(
            question_id=9028,
            options=["A", "B", "C"],
            probs=[0.60, 0.38, 0.02],
            resolution="A",
            per_model={
                "a": {"A": 0.60, "B": 0.39, "C": 0.01},
                "b": {"A": 0.60, "B": 0.37, "C": 0.03},
            },
        )
        assert record.aggregator == "median"
        clip = clip_delta(record, 0.005, side="floor_only")
        assert clip.censored is False
        assert clip.member_censored is True
        assert clip.loosen_members_at_c != 0.0
