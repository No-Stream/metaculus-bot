"""Tests for extended performance-analysis cuts added for residual analysis.

Covers: no_bias_check, financial_vs_nonfinancial_pit, stacking_effectiveness,
disagreement_predicts_error.
"""

from __future__ import annotations

import numpy as np
import pytest

from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.performance_analysis.analysis import (
    _interpolate_pit,
    disagreement_predicts_error,
    financial_vs_nonfinancial_pit,
    no_bias_check,
    numeric_pit_analysis,
    stacking_effectiveness,
)


def _old_interpolate_pit(resolution: float, lower_bound: float, upper_bound: float, cdf_values: list[float]) -> float:
    """The pre-fix linear-index implementation, kept here only to prove the regression.

    Maps the resolution to a CDF index assuming a LINEAR value grid. Correct for
    linear-scaled questions, wrong for log-scaled (zero_point) ones.
    """
    total_range = upper_bound - lower_bound
    if total_range <= 0:
        return 0.5
    fraction = (resolution - lower_bound) / total_range
    n = len(cdf_values)
    idx_float = fraction * (n - 1)
    idx_low = max(0, min(int(idx_float // 1), n - 2))
    idx_high = idx_low + 1
    weight = idx_float - idx_low
    return cdf_values[idx_low] * (1 - weight) + cdf_values[idx_high] * weight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_record(
    post_id: int,
    prob_yes: float,
    resolution: bool,
    per_model: dict[str, str] | None = None,
    category: str | None = None,
) -> dict:
    return {
        "post_id": post_id,
        "type": "binary",
        "our_prob_yes": prob_yes,
        "our_forecast_values": [1.0 - prob_yes, prob_yes],
        "resolution_parsed": resolution,
        "brier_score": (prob_yes - (1.0 if resolution else 0.0)) ** 2,
        "log_score": 0.0,
        "numeric_log_score": None,
        "mc_log_score": None,
        "per_model_forecasts": per_model or {},
        "metadata": {"category": category},
    }


def _numeric_record(
    post_id: int,
    cdf: list[float],
    resolution: float,
    lower: float = 0.0,
    upper: float = 100.0,
    category: str | None = None,
) -> dict:
    return {
        "post_id": post_id,
        "type": "numeric",
        "our_forecast_values": cdf,
        "resolution_parsed": resolution,
        "scaling": {"range_min": lower, "range_max": upper},
        "open_lower_bound": False,
        "open_upper_bound": False,
        "brier_score": None,
        "log_score": None,
        "numeric_log_score": 0.0,
        "mc_log_score": None,
        "per_model_forecasts": {},
        "metadata": {"category": category},
    }


# ---------------------------------------------------------------------------
# no_bias_check
# ---------------------------------------------------------------------------


class TestNoBiasCheck:
    def test_detects_no_bias(self):
        # Predict 30% when actual YES rate is 43% -> -13pp NO-bias
        records = [_binary_record(i, 0.30, True) for i in range(43)] + [
            _binary_record(100 + i, 0.30, False) for i in range(57)
        ]
        result = no_bias_check(records)
        assert result["count"] == 100
        assert result["mean_predicted"] == pytest.approx(0.30)
        assert result["actual_yes_rate"] == pytest.approx(0.43)
        assert result["bias_pp"] == pytest.approx(-13.0)

    def test_reports_low_range_subset(self):
        # 20 records inside the 0.10-0.30 bucket: mean predicted ~0.205, actual
        # yes-rate 0.50 (10 of 20 resolve YES). Plus 5 records at 0.70 outside
        # the bucket, which must not leak into the low_range stats.
        low_range = (
            [_binary_record(i, 0.15, True) for i in range(4)]
            + [_binary_record(10 + i, 0.25, True) for i in range(6)]
            + [_binary_record(20 + i, 0.20, False) for i in range(10)]
        )
        other = [_binary_record(100 + i, 0.70, True) for i in range(5)]
        result = no_bias_check(low_range + other)
        assert "low_range" in result
        lr = result["low_range"]
        assert lr["count"] == 20
        assert lr["mean_predicted"] == pytest.approx(0.205, abs=0.01)
        assert lr["actual_yes_rate"] == pytest.approx(0.50)

    def test_empty_data(self):
        assert no_bias_check([])["count"] == 0


# ---------------------------------------------------------------------------
# financial_vs_nonfinancial_pit
# ---------------------------------------------------------------------------


class TestFinancialVsNonfinancialPit:
    def test_splits_by_category(self):
        # Simple linear CDFs so PIT is predictable
        linear_cdf = [i / 200 for i in range(201)]
        records = [
            _numeric_record(1, linear_cdf, resolution=25.0, category="Economy & Business"),
            _numeric_record(2, linear_cdf, resolution=75.0, category="Economy & Business"),
            _numeric_record(3, linear_cdf, resolution=50.0, category="Science & Tech"),
        ]
        result = financial_vs_nonfinancial_pit(records)
        assert result["financial"]["count"] == 2
        assert result["nonfinancial"]["count"] == 1

    def test_unknown_category_goes_to_nonfinancial(self):
        linear_cdf = [i / 200 for i in range(201)]
        records = [_numeric_record(1, linear_cdf, resolution=50.0, category=None)]
        result = financial_vs_nonfinancial_pit(records)
        assert result["nonfinancial"]["count"] == 1
        assert result["financial"]["count"] == 0


# ---------------------------------------------------------------------------
# stacking_effectiveness
# ---------------------------------------------------------------------------


class TestStackingEffectiveness:
    def test_computes_counterfactual_mean_brier_on_triggered(self):
        # Triggered = per-model probability range exceeds threshold.
        high_spread = _binary_record(
            1,
            prob_yes=0.50,
            resolution=True,
            per_model={"m1": "10%", "m2": "90%"},  # prob range 0.80
        )
        low_spread = _binary_record(
            2,
            prob_yes=0.50,
            resolution=True,
            per_model={"m1": "48%", "m2": "52%"},  # prob range 0.04
        )
        result = stacking_effectiveness([high_spread, low_spread], threshold=0.20)
        assert result["triggered_count"] == 1
        assert result["skipped_count"] == 1

    def test_empty_data(self):
        assert stacking_effectiveness([], threshold=0.15)["triggered_count"] == 0

    def test_boundary_exact_match_skips(self):
        exact_match = _binary_record(
            1,
            prob_yes=0.50,
            resolution=True,
            per_model={"m1": "40%", "m2": "60%"},  # prob range exactly 0.20
        )
        result = stacking_effectiveness([exact_match], threshold=0.20)
        assert result["triggered_count"] == 0
        assert result["skipped_count"] == 1


# ---------------------------------------------------------------------------
# disagreement_predicts_error
# ---------------------------------------------------------------------------


class TestDisagreementPredictsError:
    def test_positive_correlation_on_disagreement_and_error(self):
        # Build records where high-spread questions are also the high-Brier ones.
        records = []
        for i in range(10):
            spread_tight = {"m1": f"{50 + i}%", "m2": f"{50 - i}%"}  # low spread
            records.append(_binary_record(i, 0.50, resolution=True, per_model=spread_tight))
        for i in range(10):
            # High spread, Brier gets large when prob_yes is wrong
            spread_wide = {"m1": "90%", "m2": "10%"}
            records.append(
                _binary_record(100 + i, 0.10, resolution=True, per_model=spread_wide)  # Brier = 0.81
            )
        result = disagreement_predicts_error(records)
        # High-spread bucket should have worse (higher) Brier
        assert result["count"] >= 20
        assert result["spearman_rho"] is not None
        assert result["spearman_rho"] > 0.3

    def test_handles_few_records(self):
        # With <3 records, can't compute meaningful correlation. Pair each
        # record with a per_model dict so it actually contributes to the
        # spread correlation.
        records = [
            _binary_record(1, 0.5, True, per_model={"m1": "40%", "m2": "60%"}),
            _binary_record(2, 0.6, True, per_model={"m1": "50%", "m2": "70%"}),
        ]
        result = disagreement_predicts_error(records)
        assert result["count"] == 2
        assert result["spearman_rho"] is None  # n<3


# ---------------------------------------------------------------------------
# collector — bot_comment_created_at field
# ---------------------------------------------------------------------------


class TestCollectorCommentCreatedAt:
    """Records produced by the collector should surface the comment's
    ``created_at`` timestamp so cohort cuts can filter by submit-date (vs the
    coarser actual_resolve_time on the question)."""

    def _post_data(self, post_id: int, question_id: int, resolution: str = "yes") -> dict:
        return {
            "id": post_id,
            "title": f"Q{post_id}",
            "question": {
                "id": question_id,
                "type": "binary",
                "resolution": resolution,
                "my_forecasts": {
                    "latest": {
                        "forecast_values": [0.3, 0.7],
                        "score_data": {"peer_score": 1.0},
                    },
                },
                "scaling": {},
                "options": None,
                "open_lower_bound": False,
                "open_upper_bound": False,
                "nr_forecasters": 5,
                "title": f"Q{post_id}",
            },
            "projects": {},
        }

    def test_record_includes_bot_comment_created_at(self):
        from metaculus_bot.performance_analysis.collector import _process_post

        post = self._post_data(1, 11)
        comment = {
            "id": 999,
            "text": "*Forecaster 1*: 70%\n",
            "on_post": 1,
            "created_at": "2026-04-30T12:34:56Z",
        }
        records = _process_post(post, {1: comment})
        assert len(records) == 1
        assert records[0]["bot_comment_created_at"] == "2026-04-30T12:34:56Z"

    def test_record_has_none_when_comment_missing(self):
        from metaculus_bot.performance_analysis.collector import _process_post

        post = self._post_data(2, 22)
        records = _process_post(post, {})
        assert len(records) == 1
        assert records[0]["bot_comment_created_at"] is None

    def test_record_has_none_when_comment_lacks_created_at(self):
        from metaculus_bot.performance_analysis.collector import _process_post

        post = self._post_data(3, 33)
        comment = {"id": 1000, "text": "*Forecaster 1*: 70%\n", "on_post": 3}
        records = _process_post(post, {3: comment})
        assert len(records) == 1
        assert records[0]["bot_comment_created_at"] is None


# ---------------------------------------------------------------------------
# collector — stacker_outcome / stacker_outcome_source fields
# ---------------------------------------------------------------------------


class TestCollectorStackerOutcome:
    """Records produced by the collector should expose the tri-state
    ``stacker_outcome`` plus its provenance, computed from
    ``parse_inferred_stacker_outcome`` over the comment text. The legacy
    ``was_stacked`` field collapses median-fallback into False, so analyses
    that need to distinguish "stacker LLM ran" from "MEDIAN fallback" must
    consume ``stacker_outcome``.
    """

    def _post_data(self, post_id: int, question_id: int) -> dict:
        return {
            "id": post_id,
            "title": f"Q{post_id}",
            "question": {
                "id": question_id,
                "type": "binary",
                "resolution": "yes",
                "my_forecasts": {
                    "latest": {
                        "forecast_values": [0.3, 0.7],
                        "score_data": {"peer_score": 1.0},
                    },
                },
                "scaling": {},
                "options": None,
                "open_lower_bound": False,
                "open_upper_bound": False,
                "nr_forecasters": 5,
                "title": f"Q{post_id}",
            },
            "projects": {},
        }

    def _run(self, post_id: int, question_id: int, comment_text: str | None) -> dict:
        from metaculus_bot.performance_analysis.collector import _process_post

        post = self._post_data(post_id, question_id)
        if comment_text is None:
            records = _process_post(post, {})
        else:
            records = _process_post(post, {post_id: {"id": 999, "text": comment_text, "on_post": post_id}})
        assert len(records) == 1
        return records[0]

    def test_outcome_marker_primary(self):
        rec = self._run(1, 11, "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=primary -->\n")
        assert rec["stacker_outcome"] == "primary"
        assert rec["stacker_outcome_source"] == "marker_outcome"

    def test_outcome_marker_fallback_median_distinguished_from_skipped(self):
        # The load-bearing case: pre-fix this would round-trip as STACKED=true
        # → was_stacked=True with no way to tell median-fallback from primary.
        # Now stacker_outcome="fallback_median" is preserved on the record.
        rec = self._run(2, 22, "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=fallback_median -->\n")
        assert rec["stacker_outcome"] == "fallback_median"
        assert rec["stacker_outcome_source"] == "marker_outcome"

    def test_outcome_marker_skipped(self):
        rec = self._run(3, 33, "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=skipped -->\n")
        assert rec["stacker_outcome"] == "skipped"
        assert rec["stacker_outcome_source"] == "marker_outcome"

    def test_legacy_marker_only_maps_to_primary(self):
        rec = self._run(4, 44, "*Forecaster 1*: 70%\n<!-- STACKED=true -->\n")
        assert rec["stacker_outcome"] == "primary"
        assert rec["stacker_outcome_source"] == "marker_legacy"

    def test_legacy_marker_false_maps_to_skipped(self):
        rec = self._run(5, 55, "*Forecaster 1*: 70%\n<!-- STACKED=false -->\n")
        assert rec["stacker_outcome"] == "skipped"
        assert rec["stacker_outcome_source"] == "marker_legacy"

    def test_historical_body_inferred_primary(self):
        # Pre-marker comment from spring-aib-2026 dataset: no STACKED= or
        # STACKER_OUTCOME= marker, but the Forecaster 1 body opens with
        # "## Stacker Meta-Analysis", which only the stacker pipeline produces.
        comment = (
            "# SUMMARY\n"
            "*Forecaster 1*: 70%\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "## Stacker Meta-Analysis\n\n"
            "Synthesis of 6 base models below.\n"
        )
        rec = self._run(6, 66, comment)
        assert rec["stacker_outcome"] == "primary"
        assert rec["stacker_outcome_source"] == "historical_body"

    def test_no_signal_returns_none(self):
        rec = self._run(7, 77, "*Forecaster 1*: 70%\n")
        assert rec["stacker_outcome"] is None
        assert rec["stacker_outcome_source"] == "none"

    def test_missing_comment_returns_none(self):
        rec = self._run(8, 88, None)
        assert rec["stacker_outcome"] is None
        assert rec["stacker_outcome_source"] == "none"

    def test_outcome_marker_takes_precedence_over_legacy(self):
        # Both markers coexist for one round of back-compat. The collector
        # must prefer the richer STACKER_OUTCOME= signal so median-fallback
        # isn't silently downgraded to "primary".
        comment = "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=fallback_median -->\n<!-- STACKED=false -->\n"
        rec = self._run(9, 99, comment)
        assert rec["stacker_outcome"] == "fallback_median"
        assert rec["stacker_outcome_source"] == "marker_outcome"


# ---------------------------------------------------------------------------
# _interpolate_pit — value-grid-aware PIT (regression: log-scaled questions)
# ---------------------------------------------------------------------------


class TestInterpolatePit:
    """PIT = F(resolution). F must be read against the ACTUAL value grid the CDF
    lives on (linear for linear questions, geometric for zero_point questions),
    not against a linear index map. The old linear-index map mis-buckets
    log-scaled resolutions by up to ~0.24."""

    def test_linear_question_matches_old_behavior(self):
        # Linear grid: new value-grid interpolation must equal the old linear-index
        # interpolation within float tolerance (mathematically equivalent).
        lower, upper = 0.0, 100.0
        cdf = list(np.linspace(0.0, 1.0, 201))  # straight-line CDF
        grid = list(build_cdf_value_grid(lower, upper, None, num_points=201))
        for resolution in (0.0, 12.3, 25.0, 50.0, 73.7, 100.0):
            new = _interpolate_pit(resolution, lower, upper, cdf, value_grid=grid, zero_point=None)
            old = _old_interpolate_pit(resolution, lower, upper, cdf)
            assert new == pytest.approx(old, abs=1e-9)

    def test_linear_endpoints_and_midpoint(self):
        lower, upper = 0.0, 100.0
        cdf = list(np.linspace(0.0, 1.0, 201))
        grid = list(build_cdf_value_grid(lower, upper, None, num_points=201))
        assert _interpolate_pit(lower, lower, upper, cdf, value_grid=grid) == pytest.approx(cdf[0])
        assert _interpolate_pit(upper, lower, upper, cdf, value_grid=grid) == pytest.approx(cdf[-1])
        assert _interpolate_pit(50.0, lower, upper, cdf, value_grid=grid) == pytest.approx(0.5)

    def test_log_scaled_question_differs_and_is_correct(self):
        # Log-scaled (zero_point) question: the value grid is geometric, so the
        # resolution lands on a different CDF index than the linear-index map.
        lower, upper, zero_point = 1.0, 1000.0, 0.0
        cdf = list(np.linspace(0.0, 1.0, 201))  # uniform-in-index CDF
        geo_grid = build_cdf_value_grid(lower, upper, zero_point, num_points=201)

        # Resolution near the low end of a log scale: linearly it's ~0.1% of the
        # range, but on the geometric grid it's a meaningful chunk of probability.
        resolution = 31.6  # ~10^1.5 -> roughly the geometric midpoint of [1, 1000]

        new = _interpolate_pit(resolution, lower, upper, cdf, value_grid=list(geo_grid), zero_point=zero_point)
        old = _old_interpolate_pit(resolution, lower, upper, cdf)

        expected = float(np.interp(resolution, geo_grid, np.asarray(cdf, dtype=float)))
        assert new == pytest.approx(expected, abs=1e-12)

        # The fix must bite: geometric vs linear-index map differ materially here.
        assert abs(new - old) > 0.2
        # And the new value is the geometric-midpoint-ish PIT (~0.5), not the
        # near-zero PIT the linear-index map produces.
        assert new == pytest.approx(0.5, abs=0.02)
        assert old < 0.05

    def test_falls_back_to_zero_point_grid_when_value_grid_absent(self):
        # No continuous_range supplied -> reconstruct the geometric grid from
        # zero_point. Result must match interpolation against the rebuilt grid.
        lower, upper, zero_point = 1.0, 1000.0, 0.0
        cdf = list(np.linspace(0.0, 1.0, 201))
        resolution = 31.6

        no_grid = _interpolate_pit(resolution, lower, upper, cdf, value_grid=None, zero_point=zero_point)
        rebuilt = build_cdf_value_grid(lower, upper, zero_point, num_points=201)
        expected = float(np.interp(resolution, rebuilt, np.asarray(cdf, dtype=float)))
        assert no_grid == pytest.approx(expected, abs=1e-12)

    def test_mismatched_value_grid_length_falls_back(self):
        # A value_grid whose length != cdf is ignored; we rebuild from bounds/zero_point.
        lower, upper = 0.0, 100.0
        cdf = list(np.linspace(0.0, 1.0, 201))
        bad_grid = [0.0, 50.0, 100.0]  # wrong length
        result = _interpolate_pit(50.0, lower, upper, cdf, value_grid=bad_grid, zero_point=None)
        assert result == pytest.approx(0.5)

    def test_degenerate_range_returns_half(self):
        cdf = list(np.linspace(0.0, 1.0, 201))
        assert _interpolate_pit(5.0, 10.0, 10.0, cdf) == pytest.approx(0.5)


class TestNumericPitAnalysisValueGrid:
    """End-to-end numeric_pit_analysis on a small mixed cohort: one linear-scaled
    record and one log-scaled (zero_point) record carrying continuous_range."""

    def _record(self, post_id, cdf, resolution, lower, upper, zero_point, continuous_range):
        return {
            "post_id": post_id,
            "type": "numeric",
            "our_forecast_values": cdf,
            "resolution_parsed": resolution,
            "scaling": {
                "range_min": lower,
                "range_max": upper,
                "zero_point": zero_point,
                "continuous_range": continuous_range,
            },
            "open_lower_bound": False,
            "open_upper_bound": False,
            "brier_score": None,
            "log_score": None,
            "numeric_log_score": 0.0,
            "mc_log_score": None,
            "per_model_forecasts": {},
            "metadata": {"category": None},
        }

    def test_continuous_range_used_directly_for_log_scaled(self):
        cdf = list(np.linspace(0.0, 1.0, 201))
        # Linear question, midpoint resolution -> PIT 0.5.
        lin_grid = list(build_cdf_value_grid(0.0, 100.0, None, num_points=201))
        linear_rec = self._record(1, cdf, 50.0, 0.0, 100.0, None, lin_grid)

        # Log-scaled question; resolution at geometric midpoint -> PIT ~0.5,
        # which the linear-index map would have called ~0.03.
        geo_grid = list(build_cdf_value_grid(1.0, 1000.0, 0.0, num_points=201))
        log_rec = self._record(2, cdf, 31.6, 1.0, 1000.0, 0.0, geo_grid)

        result = numeric_pit_analysis([linear_rec, log_rec])
        assert result["count"] == 2
        assert result["pit_values"][0] == pytest.approx(0.5)
        assert result["pit_values"][1] == pytest.approx(0.5, abs=0.02)
        # Both PITs land in the central coverage band.
        assert result["coverage_50"] == pytest.approx(1.0)
