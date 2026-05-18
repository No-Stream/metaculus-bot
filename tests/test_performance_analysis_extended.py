"""Tests for extended performance-analysis cuts added for residual analysis.

Covers: no_bias_check, financial_vs_nonfinancial_pit, stacking_effectiveness,
disagreement_predicts_error.
"""

from __future__ import annotations

import pytest

from metaculus_bot.performance_analysis.analysis import (
    disagreement_predicts_error,
    financial_vs_nonfinancial_pit,
    no_bias_check,
    stacking_effectiveness,
)

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
