"""Tests for the per-model cross-check: replay the members through the clamp, then aggregate."""

from __future__ import annotations

import math

import pytest

from metaculus_bot.performance_analysis.clip_threshold_tables import cross_check_row, replay_cohort
from metaculus_bot.performance_analysis.clip_threshold_windows import WINDOW_ALL
from tests.clip_threshold_fakes import BEFORE_WIDENING, one_binary, one_mc


class TestPerModelCrossCheck:
    """Replaying the members through the clamp, then the median.

    ``median(clamp(members)) == clamp(median(members))`` holds exactly for an odd member
    count (the median is an order statistic and clamping is monotone). For an even count
    the published median averages the two middle members, so the two paths can differ —
    which is the point of reporting the gap rather than asserting it away.
    """

    def test_three_member_replay_agrees_with_the_published_vector_path(self):
        record = one_binary(
            question_id=6001,
            p_yes=0.02,
            resolution=False,
            per_model={"a": "1.0%", "b": "2.0%", "c": "40.0%"},
        )
        row = cross_check_row([record], question_type="binary", window=WINDOW_ALL, c=0.05)
        assert replay_cohort([record], question_type="binary", window=WINDOW_ALL).n_replayable == 1
        assert row.n_disagree == 0
        assert row.max_abs_gap == pytest.approx(0.0, abs=1e-9)
        assert row.sum_delta_replay == pytest.approx(row.sum_delta_published, abs=1e-9)

    def test_two_member_replay_can_disagree_and_the_gap_is_reported(self):
        """Members 0.02 and 0.08 publish a median 0.05, and a 0.06 floor lifts only the low member.

        The replayed median is then 0.07 while clamping the published 0.05 gives 0.06. That 0.01
        gap is the even-count approximation, reported rather than hidden.
        """
        record = one_binary(
            question_id=6002,
            p_yes=0.05,
            resolution=True,
            per_model={"a": "2.0%", "b": "8.0%"},
        )
        row = cross_check_row([record], question_type="binary", window=WINDOW_ALL, c=0.06)
        assert replay_cohort([record], question_type="binary", window=WINDOW_ALL).n_even_members == 1
        assert row.max_abs_gap == pytest.approx(0.01, abs=1e-9)
        assert row.n_disagree == 1
        assert row.sum_delta_replay > row.sum_delta_published

    def test_stacked_records_are_not_replayable(self):
        record = one_binary(
            question_id=6003,
            p_yes=0.02,
            resolution=False,
            per_model={"stacker": "2.0%"},
            stacker_outcome="primary",
        )
        cohort = replay_cohort([record], question_type="binary", window=WINDOW_ALL)
        assert (cohort.n_records, cohort.n_replayable) == (1, 0)

    def test_mean_era_record_is_replayed_with_the_mean(self):
        """A pre_flip record whose published value is the members' MEAN, not their median.

        q38797 is the real shape: members 0.55 / 0.60 / 0.85 published as 0.667. The replay
        detects the mean aggregator and rebuilds the publish exactly, so there is no baseline
        mismatch to disclose and nothing is charged to a clip that moves none of the members.
        Before aggregator detection this record read as a 0.067 baseline gap and a median
        replay put -122.33 points on candidates that move nothing at all.
        """
        record = one_binary(
            question_id=6005,
            p_yes=0.667,
            resolution=True,
            created_at=BEFORE_WIDENING,
            per_model={"a": "55.0%", "b": "60.0%", "c": "85.0%"},
        )
        assert record.aggregator == "mean"
        cohort = replay_cohort([record], question_type="binary", window=WINDOW_ALL)
        assert cohort.n_replayable == 1
        assert cohort.n_mean_aggregator == 1
        assert cohort.n_unknown_aggregator == 0
        assert cohort.n_baseline_mismatch == 0
        assert cohort.max_baseline_gap == pytest.approx(0.0, abs=1e-9)
        row = cross_check_row([record], question_type="binary", window=WINDOW_ALL, c=0.05)
        assert row.sum_delta_replay == 0.0
        assert row.sum_delta_published == 0.0

    def test_mean_era_record_clips_along_the_mean_path(self):
        """Members 0.01 / 0.04 / 0.10 published as their mean 0.05 (median 0.04) under the 0.01 floor.

        A 0.03 floor lifts the low member to 0.03 and the MEAN moves to round(0.17/3, 3) = 0.057;
        the published-vector path sees a 0.05 publish that the 0.03 floor does not touch at all.
        """
        record = one_binary(
            question_id=6006,
            p_yes=0.05,
            resolution=False,
            created_at=BEFORE_WIDENING,
            per_model={"a": "1.0%", "b": "4.0%", "c": "10.0%"},
        )
        assert record.aggregator == "mean"
        row = cross_check_row([record], question_type="binary", window=WINDOW_ALL, c=0.03)
        assert row.sum_delta_published == 0.0
        assert row.sum_delta_replay == pytest.approx(100.0 * math.log(0.943 / 0.95), abs=1e-9)
        assert row.n_disagree == 1

    def test_unknown_aggregator_is_the_baseline_mismatch_residue(self):
        """Published 0.20 from members whose median is 0.30 and mean is 0.40: neither rebuilds it."""
        record = one_binary(
            question_id=6007,
            p_yes=0.20,
            resolution=True,
            per_model={"a": "10.0%", "b": "30.0%", "c": "80.0%"},
        )
        assert record.aggregator == "unknown"
        cohort = replay_cohort([record], question_type="binary", window=WINDOW_ALL)
        assert cohort.n_unknown_aggregator == 1
        assert cohort.n_baseline_mismatch == 1
        assert cohort.max_baseline_gap == pytest.approx(0.10, abs=1e-9)

    def test_even_member_count_and_routing_flips_are_counted(self):
        """Members 0.02 / 0.20: the 0.18 spread clears the 0.15 stacking threshold as published,
        but a 0.10 floor squeezes it to 0.10, so the stacking route would have changed.
        """
        record = one_binary(
            question_id=6008,
            p_yes=0.11,
            resolution=False,
            per_model={"a": "2.0%", "b": "20.0%"},
        )
        row = cross_check_row([record], question_type="binary", window=WINDOW_ALL, c=0.10)
        assert replay_cohort([record], question_type="binary", window=WINDOW_ALL).n_even_members == 1
        assert row.n_routing_flips == 1
        untouched = cross_check_row([record], question_type="binary", window=WINDOW_ALL, c=0.03)
        assert untouched.n_routing_flips == 0

    def test_mc_rows_carry_no_routing_count(self):
        record = one_mc(
            question_id=6009,
            options=["A", "B"],
            probs=[0.60, 0.40],
            resolution="A",
            per_model={"a": {"A": 0.60, "B": 0.40}},
        )
        row = cross_check_row([record], question_type="multiple_choice", window=WINDOW_ALL, c=0.05)
        assert row.n_routing_flips is None

    def test_mc_replay_runs_on_option_vectors(self):
        record = one_mc(
            question_id=6004,
            options=["A", "B", "C"],
            probs=[0.70, 0.29, 0.01],
            resolution="C",
            per_model={
                "a": {"A": 0.70, "B": 0.29, "C": 0.01},
                "b": {"A": 0.70, "B": 0.29, "C": 0.01},
                "c": {"A": 0.70, "B": 0.29, "C": 0.01},
            },
        )
        row = cross_check_row([record], question_type="multiple_choice", window=WINDOW_ALL, c=0.05)
        assert replay_cohort([record], question_type="multiple_choice", window=WINDOW_ALL).n_replayable == 1
        assert row.n_disagree == 0
