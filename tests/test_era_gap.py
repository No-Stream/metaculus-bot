"""Tests for the standing era-gap read: type-and-horizon-adjusted spot peer between two eras.

The estimator is pinned on synthetic arms where the answer is known by construction: a
comparison arm whose score falls with the submit-to-resolve lag, so the type-adjusted gap is
positive while the horizon-matched gap is exactly zero; a type mix that inflates the raw gap
while the within-type gap is zero; and clusters that collapse the bootstrap when every record
shares one resolution day.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from metaculus_bot.performance_analysis.cohorts import DEGRADED_RUN_QIDS, KNOWN_BUG_QIDS
from metaculus_bot.performance_analysis.era_gap import (
    CONCERN_GAP_POINTS,
    ERA_FIELD,
    Arm,
    EraGapReport,
    Verdict,
    build_arm,
    compute_era_gap_report,
    horizon_match,
    lag_quartiles,
    main,
    per_type_gaps,
    render_report,
    summarize_arm,
    two_sided_watch,
    type_adjusted_gap,
    type_lag_adjusted_gap,
    unadjusted_gap,
)
from metaculus_bot.time_utils import parse_iso_utc

SUBMITTED = "2026-07-01T12:00:00Z"
FAST_DRAWS = 400


def _record(
    qid: int,
    *,
    spot: float | None,
    lag_days: float,
    era: str = "treated",
    q_type: str = "binary",
    submitted: str | None = SUBMITTED,
    tournament: str = "summer-futureeval-2026",
) -> dict:
    """A collector-shaped record trimmed to the fields the era read touches."""
    resolved = parse_iso_utc(SUBMITTED)
    assert resolved is not None
    resolved += timedelta(days=lag_days)
    return {
        "question_id": qid,
        "post_id": qid,
        "type": q_type,
        ERA_FIELD: era,
        "bot_comment_created_at": submitted,
        "metadata": {"actual_resolve_time": resolved.isoformat().replace("+00:00", "Z")},
        "metaculus_scores": {"spot_peer_score": spot},
        "source_tournament": tournament,
    }


def _score_falls_with_lag(lag_days: float) -> float:
    return 40.0 - 0.3 * lag_days


def _horizon_confounded_arms() -> tuple[Arm, Arm]:
    """Comparison lags run 2..120 days and treated 2..40, with an identical lag-to-score law.

    Capping the comparison arm at the treated arm's maximum lag leaves exactly the treated
    arm's lag set, so the horizon-matched gap is zero by construction while the unmatched one
    is +12 (the comparison arm carries 61 mean lag days against 21).
    """
    comparison = [
        _record(1000 + i, spot=_score_falls_with_lag(lag), lag_days=lag, era="comparison")
        for i, lag in enumerate(range(2, 121, 2))
    ]
    treated = [
        _record(2000 + i, spot=_score_falls_with_lag(lag), lag_days=lag) for i, lag in enumerate(range(2, 41, 2))
    ]
    return build_arm("treated", treated), build_arm("comparison", comparison)


class TestBuildArm:
    def test_scores_lag_and_cluster_off_the_record(self):
        arm = build_arm("t", [_record(1, spot=12.5, lag_days=3.5)])
        (scored,) = arm.records
        assert scored.question_id == "1"
        assert scored.spot == pytest.approx(12.5)
        assert scored.lag_days == pytest.approx(3.5)
        # Submitted 2026-07-01T12:00Z plus 3.5 days resolves on the 5th, the cluster key.
        assert scored.cluster == "2026-07-05"

    def test_unscoreable_records_are_dropped_and_counted_not_zeroed(self):
        records = [
            _record(1, spot=10.0, lag_days=1),
            _record(2, spot=None, lag_days=1),
            _record(3, spot=10.0, lag_days=1, submitted=None),
        ]
        arm = build_arm("t", records)
        assert arm.n == 1
        assert arm.n_unscoreable == 2

    def test_strict_drops_every_exclusion_cohort_by_question_id(self):
        bug_qid = int(next(iter(KNOWN_BUG_QIDS)))
        degraded_qid = int(next(iter(DEGRADED_RUN_QIDS)))
        records = [
            _record(bug_qid, spot=-40.0, lag_days=1),
            _record(degraded_qid, spot=-100.0, lag_days=2),
            _record(1, spot=10.0, lag_days=3),
        ]
        assert build_arm("t", records).n == 3
        strict = build_arm("t", records, strict=True)
        assert strict.n == 1
        assert strict.n_excluded == 2

    def test_missing_type_raises_rather_than_filing_the_record_nowhere(self):
        record = _record(1, spot=1.0, lag_days=1)
        del record["type"]
        with pytest.raises(KeyError):
            build_arm("t", [record])


class TestArmSummary:
    def test_effective_n_counts_distinct_resolution_days(self):
        same_day = [_record(i, spot=float(i), lag_days=5) for i in range(4)]
        spread = [_record(10 + i, spot=float(i), lag_days=5 + i) for i in range(4)]
        assert summarize_arm(build_arm("same", same_day)).n_clusters == 1
        assert summarize_arm(build_arm("spread", spread)).n_clusters == 4

    def test_distribution_fields(self):
        records = [_record(i, spot=s, lag_days=lag) for i, (s, lag) in enumerate([(-10.0, 1), (5.0, 3), (20.0, 9)])]
        summary = summarize_arm(build_arm("t", records))
        assert summary.n == 3
        assert summary.spot_mean == pytest.approx(5.0)
        assert summary.spot_median == pytest.approx(5.0)
        assert summary.n_negative == 1
        assert summary.frac_negative == pytest.approx(1 / 3)
        assert summary.lag_median_days == pytest.approx(3.0)
        assert summary.lag_max_days == pytest.approx(9.0)
        assert summary.type_counts == {"binary": 3}
        assert summary.tournament_counts == {"summer-futureeval-2026": 3}


class TestHorizonConfound:
    def test_type_adjusted_gap_is_positive_but_horizon_matched_gap_is_zero(self):
        treated, comparison = _horizon_confounded_arms()
        unmatched = type_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=1)
        assert unmatched.gap == pytest.approx(12.0)
        assert unmatched.ci_low > 0

        matched_arm = horizon_match(comparison, treated.max_lag_days)
        assert matched_arm.n == treated.n
        matched = type_adjusted_gap(treated, matched_arm, draws=FAST_DRAWS, seed=1)
        assert matched.gap == pytest.approx(0.0, abs=1e-9)
        assert matched.ci_low < 0 < matched.ci_high

    def test_lag_quintile_adjustment_attenuates_the_confound_without_capping(self):
        """Stratifying on five lag bins is coarse: it removes the between-bin trend and leaves the
        within-bin slope, so the gap shrinks from +12 to under a point rather than to zero."""
        treated, comparison = _horizon_confounded_arms()
        unmatched = type_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=1)
        adjusted = type_lag_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=1)
        assert abs(adjusted.gap) < 0.1 * unmatched.gap
        assert adjusted.ci_low < 0 < adjusted.ci_high

    def test_horizon_match_keeps_a_record_exactly_at_the_cap(self):
        comparison = build_arm("c", [_record(1, spot=1.0, lag_days=10), _record(2, spot=1.0, lag_days=10.5)])
        assert horizon_match(comparison, 10.0).n == 1

    def test_comparison_arm_lag_quartiles_fall_monotonically(self):
        _, comparison = _horizon_confounded_arms()
        rows = lag_quartiles(comparison)
        assert [row.quartile for row in rows] == [1, 2, 3, 4]
        assert sum(row.n for row in rows) == comparison.n
        means = [row.spot_mean for row in rows]
        assert means == sorted(means, reverse=True)


class TestTypeMix:
    def test_raw_gap_from_type_mix_alone_vanishes_under_type_adjustment(self):
        """Treated is half numeric (scoring 30), comparison a quarter; within type both arms tie."""

        def arm(label: str, n_binary: int, n_numeric: int, base: int) -> Arm:
            records = [_record(base + i, spot=10.0, lag_days=1 + i, q_type="binary") for i in range(n_binary)]
            records += [_record(base + 50 + i, spot=30.0, lag_days=1 + i, q_type="numeric") for i in range(n_numeric)]
            return build_arm(label, records)

        treated = arm("t", 10, 10, 100)
        comparison = arm("c", 15, 5, 300)
        assert unadjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=3).gap == pytest.approx(5.0)
        assert type_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=3).gap == pytest.approx(0.0, abs=1e-9)

    def test_per_type_gaps_only_where_both_arms_carry_the_type(self):
        treated = build_arm(
            "t",
            [_record(1, spot=20.0, lag_days=1), _record(2, spot=5.0, lag_days=2, q_type="numeric")],
        )
        comparison = build_arm(
            "c",
            [_record(3, spot=10.0, lag_days=1), _record(4, spot=0.0, lag_days=2, q_type="discrete")],
        )
        rows = per_type_gaps(treated, comparison)
        assert [row.q_type for row in rows] == ["binary"]
        assert rows[0].gap == pytest.approx(10.0)
        assert (rows[0].n_treated, rows[0].n_comparison) == (1, 1)


class TestBootstrap:
    def test_same_seed_reproduces_the_interval_and_a_new_seed_moves_it(self):
        treated, comparison = _horizon_confounded_arms()
        first = type_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=7)
        again = type_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=7)
        other = type_adjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=8)
        assert (first.ci_low, first.ci_high, first.p_negative) == (again.ci_low, again.ci_high, again.p_negative)
        assert (first.ci_low, first.ci_high) != (other.ci_low, other.ci_high)
        assert first.ci_low <= first.gap <= first.ci_high
        assert 0.0 <= first.p_negative <= 1.0

    def test_one_cluster_per_arm_collapses_the_interval_onto_the_point(self):
        """Resampling a single cluster returns the same records every draw, so the CI has no width.

        Pins that the bootstrap resamples CLUSTERS rather than records: eight records on one
        resolution day would otherwise give a wide interval.
        """
        treated = build_arm("t", [_record(i, spot=float(10 + i), lag_days=5) for i in range(8)])
        comparison = build_arm("c", [_record(100 + i, spot=float(i), lag_days=5) for i in range(8)])
        estimate = unadjusted_gap(treated, comparison, draws=FAST_DRAWS, seed=1)
        assert estimate.ci_low == pytest.approx(estimate.gap)
        assert estimate.ci_high == pytest.approx(estimate.gap)


class TestTwoSidedWatch:
    def test_concern_needs_a_point_below_the_threshold_and_an_interval_excluding_zero(self):
        assert two_sided_watch(-8.0, (-15.0, -1.0)) is Verdict.CONCERN
        assert two_sided_watch(-8.0, (-15.0, 2.0)) is Verdict.NO_MEASURABLE_DIFFERENCE
        assert two_sided_watch(-3.0, (-5.0, -1.0)) is Verdict.NO_MEASURABLE_DIFFERENCE
        assert two_sided_watch(CONCERN_GAP_POINTS, (-9.0, -1.0)) is Verdict.NO_MEASURABLE_DIFFERENCE

    def test_favourable_is_reported_not_flagged(self):
        assert two_sided_watch(10.71, (1.72, 19.69)) is Verdict.FAVOURABLE
        assert two_sided_watch(8.19, (-3.21, 19.90)) is Verdict.NO_MEASURABLE_DIFFERENCE

    def test_a_positive_gap_never_reads_as_concern_whatever_its_size(self):
        assert two_sided_watch(30.0, (20.0, 40.0)) is Verdict.FAVOURABLE


class TestReport:
    def test_report_carries_every_read_and_the_watch_row(self):
        treated, comparison = _horizon_confounded_arms()
        report = compute_era_gap_report(treated, comparison, draws=FAST_DRAWS, seed=1)
        assert isinstance(report, EraGapReport)
        assert [g.label for g in report.gaps] == [
            "unadjusted",
            "type-adjusted",
            "type-adjusted, horizon-matched",
            "type x lag-quintile adjusted",
            "type x lag-quintile adjusted, horizon-matched",
        ]
        assert report.watch.label == "type-adjusted, horizon-matched"
        assert report.watch.verdict is Verdict.NO_MEASURABLE_DIFFERENCE
        assert report.comparison_horizon_matched.n == treated.n
        assert set(report.lag_quartiles) == {"treated", "comparison"}
        as_json = json.dumps(report.to_dict())
        assert "no measurable difference" in as_json

    def test_empty_arm_fails_shut(self):
        treated, _ = _horizon_confounded_arms()
        with pytest.raises(ValueError, match="no scoreable records"):
            compute_era_gap_report(treated, build_arm("empty", []), draws=FAST_DRAWS, seed=1)

    def test_render_names_the_rule_and_the_watch_verdict(self):
        treated, comparison = _horizon_confounded_arms()
        text = render_report(compute_era_gap_report(treated, comparison, draws=FAST_DRAWS, seed=1))
        assert "| treated |" in text
        assert "| type-adjusted, horizon-matched |" in text
        assert "Standing watch" in text
        assert "no measurable difference" in text
        assert f"below {CONCERN_GAP_POINTS:+.0f}" in text


class TestCli:
    def _write(self, tmp_path, records: list[dict]) -> str:
        path = tmp_path / "tagged.json"
        path.write_text(json.dumps(records))
        return str(path)

    def _records(self) -> list[dict]:
        bug_qid = int(next(iter(KNOWN_BUG_QIDS)))
        records = [_record(i, spot=15.0 + i, lag_days=2 + i, era="triple_era") for i in range(6)]
        records += [_record(100 + i, spot=5.0 + i, lag_days=2 + 3 * i, era="post_flip") for i in range(8)]
        records.append(_record(bug_qid, spot=-80.0, lag_days=4, era="post_flip"))
        records.append(_record(900, spot=0.0, lag_days=1, era="pre_flip"))
        return records

    def test_smoke_prints_markdown_and_writes_json(self, tmp_path, capsys):
        path = self._write(tmp_path, self._records())
        out_json = tmp_path / "era_gap.json"
        main(
            [
                "--dataset",
                path,
                "--treated-era",
                "triple_era",
                "--comparison-era",
                "post_flip",
                "--strict",
                "--output-json",
                str(out_json),
            ]
        )
        text = capsys.readouterr().out
        assert "# Era gap: triple_era vs post_flip (STRICT" in text
        assert "| post_flip | 8 |" in text  # the known-bug record dropped, and its count shown
        assert "Standing watch" in text
        written = json.loads(out_json.read_text())
        assert written["comparison"]["n_excluded"] == 1
        assert written["watch"]["label"] == "type-adjusted, horizon-matched"

    def test_without_strict_the_cohort_record_stays_in(self, tmp_path, capsys):
        path = self._write(tmp_path, self._records())
        main(["--dataset", path, "--treated-era", "triple_era", "--comparison-era", "post_flip"])
        text = capsys.readouterr().out
        assert "(all records" in text
        assert "| post_flip | 9 |" in text

    def test_unknown_era_fails_shut(self, tmp_path):
        path = self._write(tmp_path, self._records())
        with pytest.raises(ValueError, match="no scoreable records"):
            main(["--dataset", path, "--treated-era", "fall_config", "--comparison-era", "post_flip"])
