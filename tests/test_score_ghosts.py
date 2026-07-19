"""Tests for the ghost-forecast scorer scaffold (scripts/score_ghosts.py).

The scorer joins harvested GHOST_FORECAST markers (gap-fill v2's unpublished dry-run
forecast) to resolved questions and computes paired log-score deltas ghost-vs-published
— the named gate for retiring gap-fill v1. Today it finds ~0 resolved v2-era questions
(v2 shipped 2026-07-17), so the n=0 path is a first-class, tested outcome.
"""

from scripts.score_ghosts import join_and_score, parse_ghost_summary, render_report


def _ghost(qid: int, qtype: str, summary: str, run_date: str = "2026-07-17T00:00:00Z") -> dict:
    return {"marker": "ghost_forecast", "qid": qid, "qtype": qtype, "summary": summary, "run_date": run_date, "seq": 0}


def _binary_record(qid: int, resolution, our_prob_yes) -> dict:
    return {
        "question_id": qid,
        "type": "binary",
        "resolution_parsed": resolution,
        "our_prob_yes": our_prob_yes,
        "our_forecast_values": [our_prob_yes] if our_prob_yes is not None else None,
    }


class TestParseGhostSummary:
    def test_binary(self):
        assert parse_ghost_summary("binary", "posterior_prob=0.4200") == 0.42

    def test_multiple_choice(self):
        assert parse_ghost_summary("multiple_choice", "Blue=0.300, Red=0.700") == {"Blue": 0.3, "Red": 0.7}

    def test_numeric_median(self):
        assert parse_ghost_summary("numeric", "median=42.5") == {"median": 42.5}

    def test_unknown_or_empty(self):
        assert parse_ghost_summary("unknown", "") is None
        assert parse_ghost_summary("numeric", "") is None


class TestJoinAndScore:
    def test_n_zero_when_no_ghosts(self):
        summary = join_and_score([], [])
        assert summary["n_ghosts"] == 0
        assert summary["n_scored"] == 0
        assert "n=0" in render_report(summary)

    def test_binary_ghost_scored_against_resolution(self):
        ghosts = [_ghost(1, "binary", "posterior_prob=0.90")]
        records = [_binary_record(1, True, 0.50)]  # ghost 0.90 beats published 0.50 on a YES
        summary = join_and_score(ghosts, records)
        assert summary["n_ghosts"] == 1
        assert summary["n_scored"] == 1
        assert summary["binary"]["n"] == 1
        # Ghost (0.90) is more confident+correct than published (0.50) -> positive delta.
        assert summary["binary"]["mean_delta"] > 0

    def test_unmatched_ghost_not_scored(self):
        ghosts = [_ghost(999, "binary", "posterior_prob=0.5")]
        summary = join_and_score(ghosts, [_binary_record(1, True, 0.5)])
        assert summary["n_ghosts"] == 1
        assert summary["n_joined"] == 0
        assert summary["n_scored"] == 0

    def test_unresolved_record_joined_but_not_scored(self):
        ghosts = [_ghost(1, "binary", "posterior_prob=0.5")]
        summary = join_and_score(ghosts, [_binary_record(1, None, 0.5)])
        assert summary["n_joined"] == 1
        assert summary["n_scored"] == 0

    def test_numeric_ghost_joined_but_not_scored_median_only(self):
        # The GHOST_FORECAST marker exposes only the numeric median, not the full
        # percentile set, so a numeric log score isn't computable from telemetry alone.
        ghosts = [_ghost(2, "numeric", "median=42.5")]
        records = [{"question_id": 2, "type": "numeric", "resolution_parsed": 40.0}]
        summary = join_and_score(ghosts, records)
        assert summary["n_joined"] == 1
        assert summary["n_scored"] == 0
        assert summary["numeric"]["n_unscoreable"] == 1

    def test_latest_ghost_per_qid_wins(self):
        ghosts = [
            _ghost(1, "binary", "posterior_prob=0.10", run_date="2026-07-10T00:00:00Z"),
            _ghost(1, "binary", "posterior_prob=0.90", run_date="2026-07-17T00:00:00Z"),
        ]
        summary = join_and_score(ghosts, [_binary_record(1, True, 0.50)])
        # Only the latest ghost (0.90) is scored -> one scored row, positive delta.
        assert summary["n_scored"] == 1
        assert summary["binary"]["mean_delta"] > 0
