"""Tests for what ``build_clip_records`` recovers from a raw performance record.

Which member forecasts it reads, which aggregator it infers from them, and what its cohort
counters say about the records it discarded.
"""

from __future__ import annotations

from metaculus_bot.performance_analysis.clip_threshold_sweep import build_clip_records
from tests.clip_threshold_fakes import binary_record, mc_record, one_binary, one_mc


class TestMemberRecovery:
    """Which recovered forecasts the replay sees: base models over a collapsed aggregate, complete ballots only."""

    def test_per_base_model_forecasts_win_over_a_collapsed_per_model_aggregate(self):
        """On a record that carried a stacker, ``per_model_forecasts`` collapses to the single
        aggregate while ``per_base_model_forecasts`` keeps the roster; the sweep must read the
        roster, or every member-level reading is a 1-member replay of the publish itself."""
        record = one_binary(
            question_id=9150,
            p_yes=0.30,
            resolution=True,
            per_model={"stacker": "30.0%"},
            per_base_model={"a": "10.0%", "b": "30.0%", "c": "80.0%"},
        )
        assert [member[1] for member in record.members] == [0.1, 0.3, 0.8]
        assert record.aggregator == "median"

    def test_mc_ballot_missing_an_option_is_dropped_not_padded(self):
        record = one_mc(
            question_id=9151,
            options=["A", "B", "C"],
            probs=[0.60, 0.30, 0.10],
            resolution="A",
            per_model={
                "a": {"A": 0.60, "B": 0.30, "C": 0.10},
                "b": {"A": 0.70, "B": 0.30},
                "c": {"C": 0.10, "A": 0.60, "B": 0.30},
            },
        )
        # The partial ballot is gone; the complete ones survive in the question's option order.
        assert record.members == ((0.60, 0.30, 0.10), (0.60, 0.30, 0.10))


class TestAggregatorDetection:
    def test_median_mean_stacker_and_neither(self):
        median = one_binary(
            question_id=9030, p_yes=0.30, resolution=True, per_model={"a": "10.0%", "b": "30.0%", "c": "80.0%"}
        )
        mean = one_binary(
            question_id=9031, p_yes=0.40, resolution=True, per_model={"a": "10.0%", "b": "30.0%", "c": "80.0%"}
        )
        neither = one_binary(
            question_id=9032, p_yes=0.20, resolution=True, per_model={"a": "10.0%", "b": "30.0%", "c": "80.0%"}
        )
        stacker = one_binary(
            question_id=9033, p_yes=0.30, resolution=True, per_model={"stacker": "30.0%"}, stacker_outcome="primary"
        )
        assert (median.aggregator, mean.aggregator, neither.aggregator, stacker.aggregator) == (
            "median",
            "mean",
            "unknown",
            "unknown",
        )
        assert stacker.replayable is False


class TestCohortAccounting:
    """The five ``ClipCohort`` counters are the sweep's only record of what it discarded.

    Split by question type because ``build_clip_records`` filters on the record's ``type`` BEFORE
    calling the builder, so a malformed MC shape can never reach a binary cohort's ``n_skipped``.
    """

    def test_binary_skips_and_counters(self):
        cohort = build_clip_records(
            [
                binary_record(question_id=9170, p_yes=0.30, resolution=True),
                {**binary_record(question_id=9171, p_yes=0.30, resolution=True), "our_prob_yes": None},
                {**binary_record(question_id=9172, p_yes=0.30, resolution=True), "resolution_parsed": "annulled"},
                binary_record(question_id=9173, p_yes=0.02, resolution=False),
                binary_record(question_id=9174, p_yes=0.98, resolution=True),
                binary_record(question_id=9175, p_yes=0.50, resolution=True, created_at=None),
            ],
            "binary",
        )
        assert [r.question_id for r in cohort.records] == ["9175", "9170", "9173", "9174"]
        # No members recovered, so the member rule falls back to the published one: 1, not 0.
        assert cohort.to_dict() == {
            "question_type": "binary",
            "n": 4,
            "n_skipped": 2,
            "n_no_timestamp": 1,
            "n_at_in_force_floor": 1,
            "n_at_in_force_ceiling": 1,
            "n_member_censored_floor": 1,
        }

    def test_mc_skips_count_only_records_of_the_cohorts_own_type(self):
        """The stray binary is filtered by type, not counted as a skip: the header's "skipped N"
        claim is about records of this cohort's own type.
        """
        cohort = build_clip_records(
            [
                mc_record(question_id=9180, options=["A", "B", "C"], probs=[0.6, 0.3, 0.1], resolution="A"),
                mc_record(question_id=9181, options=["A"], probs=[1.0], resolution="A"),
                mc_record(question_id=9182, options=["A", "B", "C"], probs=[0.6, 0.4], resolution="A"),
                mc_record(question_id=9183, options=["A", "B", "C"], probs=[0.6, 0.3, 0.1], resolution="Z"),
                binary_record(question_id=9184, p_yes=0.30, resolution=True),
            ],
            "multiple_choice",
        )
        assert [r.question_id for r in cohort.records] == ["9180"]
        assert cohort.n_skipped == 3
        assert (
            build_clip_records(
                [mc_record(question_id=9185, options=["A", "B"], probs=[0.5, 0.5], resolution="A")], "binary"
            ).n_skipped
            == 0
        )
