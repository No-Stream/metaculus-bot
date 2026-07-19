"""Tests for the ghost-forecast scorer (scripts/score_ghosts.py).

The scorer joins harvested ghost markers (gap-fill v2's unpublished dry-run forecast)
to resolved questions and computes paired log-score deltas ghost-vs-published — the
named gate for retiring gap-fill v1. Two marker sources: the full-fidelity
``GHOST_FORECAST_JSON`` (preferred; makes numeric ghosts scoreable) and the legacy
lossy ``GHOST_FORECAST`` summary. Today it finds ~0 resolved v2-era questions (v2
shipped 2026-07-17), so the n=0 path is a first-class, tested outcome.
"""

import json
import math

from metaculus_bot.numeric.pchip_cdf import generate_pchip_cdf
from scripts.score_ghosts import join_and_score, parse_ghost_summary, render_report


def _legacy_ghost(qid: int, qtype: str, summary: str, run_date: str = "2026-07-17T00:00:00Z") -> dict:
    return {"marker": "ghost_forecast", "qid": qid, "qtype": qtype, "summary": summary, "run_date": run_date, "seq": 0}


def _json_ghost(qid: int, payload: dict, run_date: str = "2026-07-17T00:00:00Z") -> dict:
    return {
        "marker": "ghost_forecast_json",
        "qid": qid,
        "run_date": run_date,
        "seq": 0,
        "forecast_json": json.dumps(payload, separators=(",", ":")),
    }


def _binary_record(qid: int, resolution, our_prob_yes) -> dict:
    return {
        "question_id": qid,
        "type": "binary",
        "resolution_parsed": resolution,
        "our_prob_yes": our_prob_yes,
        "our_forecast_values": [our_prob_yes] if our_prob_yes is not None else None,
    }


def _numeric_record(
    qid: int,
    resolution: float,
    published_cdf: list[float],
    *,
    lower=0.0,
    upper=100.0,
    open_lower=False,
    open_upper=False,
    zero_point=None,
) -> dict:
    return {
        "question_id": qid,
        "type": "numeric",
        "resolution_parsed": resolution,
        "our_forecast_values": published_cdf,
        "scaling": {"range_min": lower, "range_max": upper, "zero_point": zero_point},
        "open_lower_bound": open_lower,
        "open_upper_bound": open_upper,
    }


def _pchip_cdf(
    percent_percentiles: dict[float, float],
    lower=0.0,
    upper=100.0,
    *,
    open_lower=False,
    open_upper=False,
    zero_point=None,
) -> list[float]:
    """Build a 201-point CDF from percent-keyed percentiles (test helper).

    generate_pchip_cdf takes the open-bound flags as (open_upper, open_lower); pass them
    in that order so the published CDF matches the record's declared bounds/scaling.
    """
    cdf, _ = generate_pchip_cdf(percent_percentiles, open_upper, open_lower, upper, lower, zero_point)
    return cdf


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
        summary = join_and_score([], [], [])
        assert summary["n_ghosts"] == 0
        assert summary["n_scored"] == 0
        assert "n=0" in render_report(summary)

    def test_legacy_binary_ghost_scored_against_resolution(self):
        legacy = [_legacy_ghost(1, "binary", "posterior_prob=0.90")]
        records = [_binary_record(1, True, 0.50)]  # ghost 0.90 beats published 0.50 on a YES
        summary = join_and_score([], legacy, records)
        assert summary["n_ghosts"] == 1
        assert summary["n_scored"] == 1
        assert summary["binary"]["n"] == 1
        assert summary["source_counts"] == {"json": 0, "legacy": 1}
        # Ghost (0.90) is more confident+correct than published (0.50) -> positive delta.
        assert summary["binary"]["mean_delta"] > 0

    def test_unmatched_ghost_not_scored(self):
        legacy = [_legacy_ghost(999, "binary", "posterior_prob=0.5")]
        summary = join_and_score([], legacy, [_binary_record(1, True, 0.5)])
        assert summary["n_ghosts"] == 1
        assert summary["n_joined"] == 0
        assert summary["n_scored"] == 0

    def test_unresolved_record_joined_but_not_scored(self):
        legacy = [_legacy_ghost(1, "binary", "posterior_prob=0.5")]
        summary = join_and_score([], legacy, [_binary_record(1, None, 0.5)])
        assert summary["n_joined"] == 1
        assert summary["n_scored"] == 0

    def test_legacy_numeric_ghost_unscoreable_median_only(self):
        # The legacy GHOST_FORECAST marker exposes only the numeric median, so a
        # numeric log score isn't computable from it — reported as an honest gap.
        legacy = [_legacy_ghost(2, "numeric", "median=42.5")]
        records = [_numeric_record(2, 40.0, _pchip_cdf({5: 10, 50: 40, 95: 90}))]
        summary = join_and_score([], legacy, records)
        assert summary["n_joined"] == 1
        assert summary["n_scored"] == 0
        assert summary["numeric"]["n_unscoreable"] == 1
        assert summary["numeric"]["unscoreable_reasons"] == {"legacy_median_only": 1}

    def test_latest_ghost_per_qid_wins(self):
        legacy = [
            _legacy_ghost(1, "binary", "posterior_prob=0.10", run_date="2026-07-10T00:00:00Z"),
            _legacy_ghost(1, "binary", "posterior_prob=0.90", run_date="2026-07-17T00:00:00Z"),
        ]
        summary = join_and_score([], legacy, [_binary_record(1, True, 0.50)])
        # Only the latest ghost (0.90) is scored -> one scored row, positive delta.
        assert summary["n_scored"] == 1
        assert summary["binary"]["mean_delta"] > 0


class TestJsonSourceGhosts:
    def test_json_binary_scored(self):
        json_ghosts = [_json_ghost(1, {"qtype": "binary", "prob": 0.90})]
        summary = join_and_score(json_ghosts, [], [_binary_record(1, True, 0.50)])
        assert summary["source_counts"] == {"json": 1, "legacy": 0}
        assert summary["binary"]["n"] == 1
        assert summary["binary"]["mean_delta"] > 0

    def test_json_mc_scored(self):
        json_ghosts = [_json_ghost(1, {"qtype": "multiple_choice", "option_probs": {"Blue": 0.2, "Red": 0.8}})]
        record = {
            "question_id": 1,
            "type": "multiple_choice",
            "resolution_parsed": "Red",
            "options": ["Blue", "Red"],
            "our_forecast_values": [0.5, 0.5],
        }
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["multiple_choice"]["n"] == 1
        # Ghost put 0.8 on the correct option vs published 0.5 -> positive delta.
        assert summary["multiple_choice"]["mean_delta"] > 0

    def test_json_wins_over_legacy_for_same_qid(self):
        # A malformed-but-present legacy ghost and a JSON ghost on the same qid: JSON wins.
        legacy = [_legacy_ghost(1, "binary", "posterior_prob=0.10")]
        json_ghosts = [_json_ghost(1, {"qtype": "binary", "prob": 0.90})]
        summary = join_and_score(json_ghosts, legacy, [_binary_record(1, True, 0.50)])
        assert summary["n_ghosts"] == 1
        assert summary["source_counts"] == {"json": 1, "legacy": 0}
        # The JSON ghost (0.90, correct) scored, not the legacy 0.10.
        assert summary["binary"]["mean_delta"] > 0

    def test_malformed_json_falls_back_to_legacy(self):
        legacy = [_legacy_ghost(1, "binary", "posterior_prob=0.90")]
        bad_json = {"marker": "ghost_forecast_json", "qid": 1, "run_date": "2026-07-17T00:00:00Z", "seq": 0}
        bad_json["forecast_json"] = "{not valid json"
        summary = join_and_score([bad_json], legacy, [_binary_record(1, True, 0.50)])
        assert summary["source_counts"] == {"json": 0, "legacy": 1}
        assert summary["binary"]["n"] == 1


class TestNumericPairedScoring:
    def test_tight_json_numeric_ghost_beats_wide_published(self):
        # Published is wide across [0,100]; the ghost is tight around the median.
        # The resolution lands at 50, so the tight ghost concentrates more mass on
        # the resolution bucket -> higher numeric log score -> positive delta.
        published_cdf = _pchip_cdf({5: 10, 25: 30, 50: 50, 75: 70, 95: 90})
        record = _numeric_record(1, 50.0, published_cdf)
        json_ghosts = [
            _json_ghost(
                1,
                {
                    "qtype": "numeric",
                    "declared_percentiles": {0.05: 45, 0.25: 48, 0.5: 50, 0.75: 52, 0.95: 55},
                    "median": 50,
                },
            )
        ]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 1
        assert summary["numeric"]["n_joined"] == 1
        assert summary["numeric"]["n_unscoreable"] == 0
        row = summary["numeric"]["rows"][0]
        assert isinstance(row["ghost_log_score"], float)
        assert isinstance(row["published_log_score"], float)
        # Tight+correct ghost out-scores the wide published forecast.
        assert row["delta"] > 0
        assert summary["n_scored"] == 1

    def test_numeric_unscoreable_when_no_published_cdf(self):
        record = {
            "question_id": 1,
            "type": "numeric",
            "resolution_parsed": 50.0,
            "our_forecast_values": None,
            "scaling": {"range_min": 0.0, "range_max": 100.0, "zero_point": None},
            "open_lower_bound": False,
            "open_upper_bound": False,
        }
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.1: 10, 0.5: 50, 0.9: 90}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 0
        assert summary["numeric"]["unscoreable_reasons"] == {"no_published_cdf": 1}

    def test_report_surfaces_numeric_coverage(self):
        published_cdf = _pchip_cdf({5: 10, 50: 50, 95: 90})
        record = _numeric_record(1, 50.0, published_cdf)
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.1: 40, 0.5: 50, 0.9: 60}})]
        summary = join_and_score(json_ghosts, [], [record])
        report = render_report(summary)
        assert "numeric coverage" in report
        assert "numeric: n=1" in report

    def test_open_lower_bound_numeric_scored(self):
        # open_lower_bound threads to generate_pchip_cdf and numeric_log_score in different
        # positional slots; exercise the open-lower wiring end to end (both scores finite).
        published_cdf = _pchip_cdf({5: 20, 50: 50, 95: 90}, open_lower=True)
        record = _numeric_record(1, 50.0, published_cdf, open_lower=True)
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.05: 25, 0.5: 55, 0.95: 85}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 1
        assert summary["numeric"]["n_unscoreable"] == 0
        row = summary["numeric"]["rows"][0]
        assert math.isfinite(row["ghost_log_score"])
        assert math.isfinite(row["published_log_score"])
        assert math.isfinite(row["delta"])

    def test_open_upper_bound_numeric_scored(self):
        # Same wiring check for the open-upper flag (the other positional slot).
        published_cdf = _pchip_cdf({5: 20, 50: 50, 95: 90}, open_upper=True)
        record = _numeric_record(1, 50.0, published_cdf, open_upper=True)
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.05: 15, 0.5: 45, 0.95: 95}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 1
        assert summary["numeric"]["n_unscoreable"] == 0
        row = summary["numeric"]["rows"][0]
        assert math.isfinite(row["ghost_log_score"])
        assert math.isfinite(row["published_log_score"])
        assert math.isfinite(row["delta"])

    def test_log_scale_zero_point_numeric_scored(self):
        # Log-scaled question: zero_point=0 with a positive floor => geometric grid. The
        # zero_point must thread identically through the ghost CDF build and both scores.
        published_cdf = _pchip_cdf({5: 30, 50: 100, 95: 300}, lower=1.0, upper=1000.0, zero_point=0.0)
        record = _numeric_record(1, 100.0, published_cdf, lower=1.0, upper=1000.0, zero_point=0.0)
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.05: 40, 0.5: 110, 0.95: 260}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 1
        assert summary["numeric"]["n_unscoreable"] == 0
        row = summary["numeric"]["rows"][0]
        assert math.isfinite(row["ghost_log_score"])
        assert math.isfinite(row["published_log_score"])
        assert math.isfinite(row["delta"])

    def test_numeric_unscoreable_when_scaling_missing_bounds(self):
        # scaling without range_min => resolve_numeric_record_to_score_inputs returns None,
        # so a valid published CDF + valid ghost percentiles still can't be paired.
        published_cdf = _pchip_cdf({5: 10, 50: 50, 95: 90})
        record = {
            "question_id": 1,
            "type": "numeric",
            "resolution_parsed": 50.0,
            "our_forecast_values": published_cdf,
            "scaling": {"range_min": None, "range_max": 100.0, "zero_point": None},
            "open_lower_bound": False,
            "open_upper_bound": False,
        }
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.1: 10, 0.5: 50, 0.9: 90}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 0
        assert summary["numeric"]["unscoreable_reasons"] == {"no_score_inputs": 1}

    def test_numeric_unscoreable_when_cdf_build_fails(self):
        # Degenerate ghost percentiles at the fraction bounds (0.0 -> pct 0, 1.0 -> pct 100):
        # generate_pchip_cdf filters both out and raises, classified as cdf_build_failed.
        published_cdf = _pchip_cdf({5: 10, 50: 50, 95: 90})
        record = _numeric_record(1, 50.0, published_cdf)
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.0: 10, 1.0: 90}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 0
        assert summary["numeric"]["unscoreable_reasons"] == {"cdf_build_failed": 1}
