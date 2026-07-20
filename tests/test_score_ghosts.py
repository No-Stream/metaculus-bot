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

import numpy as np
import pytest

import metaculus_bot.numeric.pchip_cdf as pchip_mod
from metaculus_bot.numeric.config import grid_step_constraints
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
    # The scorer joins ghosts on ``post_id`` (a ghost's qid is the Metaculus post id
    # parsed from page_url). ``question_id`` is the disjoint sub-question id, so set it
    # to a deliberately-different value to keep the fixtures honest about the join key.
    return {
        "post_id": qid,
        "question_id": qid + 100_000,
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
    # post_id is the join key (ghost qid == post id); question_id is the disjoint
    # sub-question id — kept different on purpose (see _binary_record).
    return {
        "post_id": qid,
        "question_id": qid + 100_000,
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

    def test_join_keys_on_post_id_not_question_id(self):
        # A ghost's qid is the Metaculus POST id (from page_url); the collector emits
        # post_id and the disjoint sub-question question_id separately. The join MUST
        # key on post_id: a record whose question_id equals the ghost qid but whose
        # post_id does not must NOT join (this is exactly what the old question_id-keyed
        # code got wrong — it joined on the disjoint id space and always missed).
        ghost = [_legacy_ghost(42, "binary", "posterior_prob=0.90")]
        wrong_key = {
            "post_id": 999,  # ghost qid (42) != post_id -> no join
            "question_id": 42,  # equals ghost qid, but question_id is NOT the join key
            "type": "binary",
            "resolution_parsed": True,
            "our_prob_yes": 0.5,
            "our_forecast_values": [0.5],
        }
        summary = join_and_score([], ghost, [wrong_key])
        assert summary["n_joined"] == 0
        assert summary["n_scored"] == 0

        # Positive case: post_id == ghost qid joins even though question_id differs.
        right_key = {**wrong_key, "post_id": 42, "question_id": 42_000}
        summary = join_and_score([], ghost, [right_key])
        assert summary["n_joined"] == 1
        assert summary["n_scored"] == 1

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
            "post_id": 1,
            "question_id": 100_001,
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
            "post_id": 1,
            "question_id": 100_001,
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
            "post_id": 1,
            "question_id": 100_001,
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

    def test_native_discrete_ghost_scored_on_reduced_grid(self):
        # Native-discrete questions (Metaculus type == "discrete") publish a CDF on a
        # reduced grid (cdf_size != 201). Prod resamples the aggregate onto that grid;
        # _score_numeric mirrors it by building the ghost with num_points=len(published_cdf),
        # so the reduced-grid case pairs cleanly instead of being dropped. (This is the
        # discrete mechanism the scorer CAN reproduce from the record — integer-snap on
        # 201-point continuous questions is a separate, prod-side-only decision; see the
        # _score_numeric docstring.)
        min_step = round(0.01 / 20, 9)
        published_cdf, _ = generate_pchip_cdf(
            {5: 5, 50: 10, 95: 15}, False, False, 20.0, 0.0, None, min_step=min_step, num_points=21
        )
        assert len(published_cdf) == 21  # reduced grid, not the continuous 201
        record = _numeric_record(1, 10.0, published_cdf, lower=0.0, upper=20.0)
        record["type"] = "discrete"
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": {0.05: 6, 0.5: 11, 0.95: 14}})]
        summary = join_and_score(json_ghosts, [], [record])
        assert summary["numeric"]["n"] == 1
        assert summary["numeric"]["n_unscoreable"] == 0
        row = summary["numeric"]["rows"][0]
        assert math.isfinite(row["ghost_log_score"])
        assert math.isfinite(row["published_log_score"])
        assert math.isfinite(row["delta"])


class TestGhostGridScaledMaxStep:
    """F1 regression: the ghost CDF must be rebuilt with the grid-SCALED max-step.

    On a native-discrete question the published CDF lives on a coarse grid
    (cdf_size < 201) where the server's per-bin max step relaxes above 0.2 (e.g. 1.0
    on a 9-point grid). The scorer rebuilds the ghost on that same grid; if it inherited
    the 201-grid 0.2 cap it would clip a concentrated integer's mass while the published
    side (built by the fixed prod path) stayed uncapped — an asymmetric paired log-score
    biased against the ghost on concentrated discrete questions. The ghost build must pass
    the grid-scaled (min_step, max_step) so a concentrated bin survives.
    """

    def test_concentrated_discrete_ghost_retains_bin_above_020(self, monkeypatch):
        # grok's Q38880 concentrated low-count shape (~30% mass on integer 0), count 0-7,
        # open upper. On a 9-point grid the P(0) bin (cdf[1]-cdf[0]) must stay above 0.25.
        min_step, max_step = grid_step_constraints(9)
        # Any valid 9-point published CDF fixes num_points=9; its shape only feeds the
        # published log score, not the ghost CDF whose bins we assert on.
        published_cdf, _ = generate_pchip_cdf(
            {5: 1, 50: 3, 95: 6}, True, False, 7.5, -0.5, None, min_step=min_step, max_step=max_step, num_points=9
        )
        assert len(published_cdf) == 9

        captured: dict = {}
        real = pchip_mod.generate_pchip_cdf

        def spy(*args, **kwargs):
            cdf, flag = real(*args, **kwargs)
            captured["cdf"] = cdf
            captured["max_step"] = kwargs.get("max_step")
            return cdf, flag

        # _score_numeric imports generate_pchip_cdf lazily from this module, so patching
        # the module attribute captures the ghost build (published_cdf was built above,
        # before the patch, with the real function).
        monkeypatch.setattr(pchip_mod, "generate_pchip_cdf", spy)

        record = _numeric_record(1, 0.0, published_cdf, lower=-0.5, upper=7.5, open_upper=True)
        record["type"] = "discrete"
        concentrated = {0.2: 0.30, 0.4: 0.65, 0.5: 0.90, 0.8: 2.20, 0.9: 3.20, 0.99: 6.60}
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": concentrated})]
        summary = join_and_score(json_ghosts, [], [record])

        assert summary["numeric"]["n"] == 1  # scoreable
        # The ghost build received the grid-scaled max-step (1.0), not the 201-grid 0.2 cap.
        assert captured["max_step"] == pytest.approx(max_step)
        assert captured["max_step"] > 0.2
        ghost_cdf = np.asarray(captured["cdf"], dtype=float)
        assert ghost_cdf[0] == pytest.approx(0.0, abs=1e-9)  # closed lower
        p_zero = ghost_cdf[1] - ghost_cdf[0]  # F(0.5) = P(0)
        assert p_zero > 0.25, f"ghost P(0)={p_zero} was clipped by the 0.2 cap"

    def test_paired_score_symmetric_on_concentrated_discrete_grid(self):
        # Build the published CDF and the ghost from the SAME concentrated percentiles on
        # the same coarse (9-point) grid. With the grid-scaled max-step the scorer rebuilds
        # the ghost identically to the published side, so the paired delta is exactly 0.
        # Under the bug the ghost was clipped at 0.2 while the published side was not — a
        # spurious nonzero delta penalizing the concentrated ghost.
        concentrated_pct = {20.0: 0.30, 40.0: 0.65, 50.0: 0.90, 80.0: 2.20, 90.0: 3.20, 99.0: 6.60}
        min_step, max_step = grid_step_constraints(9)
        published_cdf, _ = generate_pchip_cdf(
            concentrated_pct, True, False, 7.5, -0.5, None, min_step=min_step, max_step=max_step, num_points=9
        )
        record = _numeric_record(1, 0.0, published_cdf, lower=-0.5, upper=7.5, open_upper=True)
        record["type"] = "discrete"
        ghost_declared = {p / 100.0: v for p, v in concentrated_pct.items()}
        json_ghosts = [_json_ghost(1, {"qtype": "numeric", "declared_percentiles": ghost_declared})]
        summary = join_and_score(json_ghosts, [], [record])

        assert summary["numeric"]["n"] == 1
        assert summary["numeric"]["n_unscoreable"] == 0
        row = summary["numeric"]["rows"][0]
        assert row["delta"] == pytest.approx(0.0, abs=1e-9)
