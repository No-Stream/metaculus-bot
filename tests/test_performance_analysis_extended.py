"""Tests for extended performance-analysis cuts added for residual analysis.

Covers: no_bias_check, financial_vs_nonfinancial_pit, stacking_effectiveness,
disagreement_predicts_error.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from metaculus_bot import performance_analysis
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.performance_analysis import collector
from metaculus_bot.performance_analysis.analysis import (
    _interpolate_pit,
    _single_curve_pit,
    binary_summary,
    declared_percentile_pit,
    disagreement_predicts_error,
    financial_vs_nonfinancial_pit,
    max_step_clamp_screen,
    mc_summary,
    no_bias_check,
    numeric_pit_analysis,
    per_model_binary_scores,
    per_model_cohort,
    stacking_effectiveness,
)
from metaculus_bot.performance_analysis.collector import (
    _process_post,
    build_performance_dataset,
    load_dataset,
    rescore_records,
    resolve_numeric_record_to_score_inputs,
)
from metaculus_bot.performance_analysis.parsing import anonymous_model_key, is_anonymous_model_key


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
    **stacker_fields: object,
) -> dict:
    """Build a binary record. ``stacker_fields`` sets the stacker-detection
    signals (``was_stacked``, ``stacker_outcome``, ``comment_text``) that
    ``per_model_cohort`` reads; omit them for an ordinary unstacked record."""
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
        **stacker_fields,
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


class TestMcSummary:
    def test_a_short_forecast_vector_is_dropped_from_mean_prob_correct(self, caplog):
        """A forecast vector shorter than its option list cannot say what probability was
        on the winner. The old ``else 0.0`` scored that PARSE gap as "we gave the correct
        option zero", dragging mean_prob_correct down on a defect rather than a forecast;
        the record must still count in count / mean_mc_log_score."""
        normal = {
            "type": "multiple_choice",
            "mc_log_score": -0.5,
            "resolution_parsed": "B",
            "options": ["A", "B"],
            "our_forecast_values": [0.3, 0.7],
            "post_id": 1,
        }
        short = {
            "type": "multiple_choice",
            "mc_log_score": -0.9,
            "resolution_parsed": "C",
            "options": ["A", "B", "C"],
            "our_forecast_values": [0.6, 0.4],
            "post_id": 2,
        }
        with caplog.at_level("WARNING"):
            summary = mc_summary([normal, short])
        assert summary["count"] == 2
        assert summary["mean_prob_correct"] == pytest.approx(0.7), "the short record contributes nothing"
        assert summary["mean_mc_log_score"] == pytest.approx(-0.7)
        assert summary["accuracy"] == pytest.approx(0.5)
        assert "shorter than its option list" in caplog.text


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
# per_model_cohort — phantom "Forecaster N" buckets and stacked records
# ---------------------------------------------------------------------------


class TestPerModelCohort:
    """Per-model cuts must see only named base models.

    Two ways a non-model entry reaches ``per_model_forecasts``: an anonymous
    positional key (no ``Model:`` line to attribute the bullet) and a
    stacker-fired record (the one summary bullet holds the stacker's aggregate,
    not a base model's forecast). Measured on the 2026-04 dataset, 50 such
    forecasts were being scored as if ``Forecaster 1`` and ``Forecaster 2`` were
    ensemble members, making that bucket a stacker-vs-base-model mixture.
    """

    def test_anonymous_keys_dropped_named_models_kept(self):
        record = _binary_record(
            1,
            prob_yes=0.60,
            resolution=True,
            per_model={"gpt-5.6-sol": "70%", "Forecaster 1": "50%", "Forecaster 2 base": "40%"},
        )
        [(returned, per_model)] = per_model_cohort([record], cut="unit_test")
        assert returned is record
        assert per_model == {"gpt-5.6-sol": "70%"}

    @pytest.mark.parametrize(
        "stacker_fields",
        [
            {"was_stacked": True},
            {"stacker_outcome": "primary"},
            {"stacker_outcome": "fallback_llm"},
            {"comment_text": "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=primary -->\n"},
            {"comment_text": "*Forecaster 1*: 70%\n<!-- STACKED=true -->\n"},
        ],
    )
    def test_stacker_fired_records_excluded_entirely(self, stacker_fields):
        stacked = _binary_record(
            1, prob_yes=0.60, resolution=True, per_model={"claude-opus-4.8": "70%"}, **stacker_fields
        )
        assert per_model_cohort([stacked], cut="unit_test") == []

    def test_median_records_kept(self):
        # The mirror of the case above: a record the detector confirms ran on
        # MEDIAN keeps its per-model bullets.
        unstacked = _binary_record(
            1,
            prob_yes=0.60,
            resolution=True,
            per_model={"claude-opus-4.8": "70%"},
            stacker_outcome="skipped",
        )
        [(_record, per_model)] = per_model_cohort([unstacked], cut="unit_test")
        assert per_model == {"claude-opus-4.8": "70%"}

    def test_high_spread_record_without_stacker_signals_is_kept(self):
        # ``likely_stacker`` (high spread + published value far from the median)
        # must NOT exclude: that shape is also what a MEAN-era aggregate looks
        # like, and dropping it would silently remove the high-disagreement
        # records these cuts exist to measure.
        wide = _binary_record(1, prob_yes=0.10, resolution=True, per_model={"m1": "90%", "m2": "10%"})
        [(_record, per_model)] = per_model_cohort([wide], cut="unit_test")
        assert per_model == {"m1": "90%", "m2": "10%"}

    def test_exclusions_are_logged_with_counts_and_reason(self, caplog):
        records = [
            _binary_record(1, 0.6, True, per_model={"gpt-5.6-sol": "70%", "Forecaster 1": "50%"}),
            _binary_record(2, 0.6, True, per_model={"Forecaster 1": "50%", "Forecaster 2": "40%"}),
            _binary_record(3, 0.6, True, per_model={"claude-opus-4.8": "70%"}, was_stacked=True),
        ]
        with caplog.at_level(logging.INFO, logger="metaculus_bot.performance_analysis.analysis"):
            per_model_cohort(records, cut="my_cut")

        [line] = [r.getMessage() for r in caplog.records if "PER_MODEL_COHORT" in r.getMessage()]
        assert "cut=my_cut" in line
        assert "eligible_records=2" in line
        assert "excluded_stacked_records=1" in line
        assert "excluded_stacked_observations=1" in line
        assert "excluded_anonymous_observations=3" in line
        assert "reason=" in line

    def test_per_model_binary_scores_excludes_phantoms(self):
        records = [
            _binary_record(1, 0.6, True, per_model={"gpt-5.6-sol": "70%", "Forecaster 1": "10%"}),
            _binary_record(2, 0.4, False, per_model={"gpt-5.6-sol": "30%", "Forecaster 1": "90%"}),
            # Stacker-fired: its bullet is the aggregate, not a base model.
            _binary_record(3, 0.6, True, per_model={"gemini-3.1-pro-preview": "70%"}, was_stacked=True),
        ]
        scores = per_model_binary_scores(records)
        assert set(scores) == {"gpt-5.6-sol"}
        assert scores["gpt-5.6-sol"]["count"] == 2

    def test_aggregate_cuts_still_include_excluded_records(self):
        # The operator keeps the aggregates over stacked / anonymously-attributed
        # records — only the per-MODEL cuts drop them. Same three records, both
        # aggregate paths must count all three.
        records = [
            _binary_record(1, 0.6, True, per_model={"Forecaster 1": "60%"}),
            _binary_record(2, 0.6, True, per_model={"claude-opus-4.8": "70%"}, was_stacked=True),
            _binary_record(3, 0.4, False, per_model={"gpt-5.6-sol": "40%"}),
        ]
        assert binary_summary(records)["count"] == 3
        assert no_bias_check(records)["count"] == 3
        # ...while the per-model cut sees one named model on one question.
        assert set(per_model_binary_scores(records)) == {"gpt-5.6-sol"}

    def test_spread_cuts_skip_stacked_records(self):
        stacked_wide = _binary_record(
            1, 0.5, True, per_model={"Forecaster 1": "10%", "Forecaster 2": "90%"}, was_stacked=True
        )
        named_wide = _binary_record(2, 0.5, True, per_model={"m1": "10%", "m2": "90%"})
        effectiveness = stacking_effectiveness([stacked_wide, named_wide], threshold=0.20)
        assert effectiveness["triggered_count"] == 1
        assert effectiveness["skipped_count"] == 0

        correlation = disagreement_predicts_error([stacked_wide, named_wide])
        assert correlation["count"] == 1


class TestAnonymousModelKey:
    """The producer and the predicate must agree — they are what keeps the
    phantom filter from drifting away from the key format it filters on."""

    @pytest.mark.parametrize("index", [1, 3, 12])
    @pytest.mark.parametrize("is_base_model", [False, True])
    def test_produced_keys_are_recognized(self, index, is_base_model):
        assert is_anonymous_model_key(anonymous_model_key(index, is_base_model=is_base_model))

    @pytest.mark.parametrize(
        "key",
        [
            "gpt-5.6-sol",
            "claude-opus-4.8",
            "gemini-3.1-pro-preview",
            # Near-misses: real display names that merely start the same way, and
            # a bullet-shaped string, must not be swept up.
            "Forecaster",
            "Forecaster One",
            "Forecaster 1 (gpt-5.6-sol)",
            "*Forecaster 1*",
        ],
    )
    def test_model_names_are_not_anonymous(self, key):
        assert not is_anonymous_model_key(key)


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
        post = self._post_data(2, 22)
        records = _process_post(post, {})
        assert len(records) == 1
        assert records[0]["bot_comment_created_at"] is None

    def test_crowd_size_is_read_off_the_post_not_the_question(self):
        """``nr_forecasters`` is a POST field; reading it off the question dict with a 0
        default made it read 0 in all 2196 archived records — "never read", rendered as a
        measured empty crowd. This fixture deliberately keeps the decoy on the question."""
        post = self._post_data(5, 55)
        post["nr_forecasters"] = 170
        records = _process_post(post, {})

        assert records[0]["metadata"]["nr_forecasters"] == 170

    def test_a_post_with_no_crowd_field_reads_none_not_zero(self):
        # None means "the post didn't say", which a crowd-size cut can drop. A 0 would
        # average a fabricated empty crowd into the cut, and would also silently kill
        # audit.py's `n/a` fallback (a real 0 is not a missing key).
        post = self._post_data(6, 66)
        post.pop("nr_forecasters", None)
        records = _process_post(post, {})

        assert records[0]["metadata"]["nr_forecasters"] is None

    def test_record_has_none_when_comment_lacks_created_at(self):
        post = self._post_data(3, 33)
        comment = {"id": 1000, "text": "*Forecaster 1*: 70%\n", "on_post": 3}
        records = _process_post(post, {3: comment})
        assert len(records) == 1
        assert records[0]["bot_comment_created_at"] is None

    def test_stacker_skip_reason_marker_round_trips_onto_record(self):
        # The additive STACKER_SKIP_REASON marker must reach the record dict —
        # its documented durable path is the comment, not the run log.
        post = self._post_data(4, 44)
        comment = {
            "id": 1001,
            "text": "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=skipped -->\n<!-- STACKER_SKIP_REASON=single_forecaster -->\n",
            "on_post": 4,
        }
        records = _process_post(post, {4: comment})
        assert len(records) == 1
        assert records[0]["stacker_skip_reason"] == "single_forecaster"
        assert records[0]["stacker_outcome"] == "skipped"

    def test_stacker_skip_reason_none_without_marker(self):
        post = self._post_data(5, 55)
        comment = {"id": 1002, "text": "*Forecaster 1*: 70%\n", "on_post": 5}
        records = _process_post(post, {5: comment})
        assert records[0]["stacker_skip_reason"] is None

    def test_practice_posts_produce_no_records(self):
        # Practice questions are not tournament scoring surface; they must never enter
        # the dataset (they would otherwise land in every calibration cut).
        post = self._post_data(6, 66)
        post["title"] = "[PRACTICE] Will this be scored?"
        comment = {"id": 1003, "text": "*Forecaster 1*: 70%\n", "on_post": 6}
        assert _process_post(post, {6: comment}) == []


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

    def test_outcome_marker_skipped_config_off(self):
        # Config-suppressed skip (per-type gate off despite high spread) must
        # survive the collector round-trip distinct from plain "skipped" — this
        # is the field the 0/22-numeric-suppression re-attribution needed.
        rec = self._run(3, 33, "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=skipped_config_off -->\n")
        assert rec["stacker_outcome"] == "skipped_config_off"
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

    def test_degenerate_range_raises_instead_of_answering_with_the_best_case(self):
        """A zero-width question has no PIT. This used to return 0.5, which is the single most
        favorable value available — inside BOTH coverage bands — so a degenerate record
        silently improved every calibration statistic it entered. The caller screens the
        range now (see ``TestDeclaredPercentilePitDropsDegenerateRanges``)."""
        cdf = list(np.linspace(0.0, 1.0, 201))

        with pytest.raises(ValueError, match="degenerate question range"):
            _interpolate_pit(5.0, 10.0, 10.0, cdf)


class TestInterpolatePitOutOfGrid:
    """The q44218 shape: a resolution BEYOND the grid must not be censored at cdf[0]/cdf[-1].

    With below-bound mass expressible on open bounds, cdf[0] can be ~0.9, so the grid
    clamp reads a below-grid resolution — a LOW-tail event — as a high PIT. Beyond the
    grid the PIT must come off the members' declared-percentile curves instead.
    """

    _LOWER, _UPPER = 100.0, 200.0
    # 90% of the mass below the open lower bound (F(100) = 0.90), like q44218's 0.9168.
    _CDF = list(np.linspace(0.90, 0.975, 201))
    _PERCENTILES = {
        "model-a": [[10.0, 80.0], [50.0, 90.0], [90.0, 105.0]],
        "model-b": [[10.0, 85.0], [50.0, 95.0], [90.0, 110.0]],
    }

    def _grid(self) -> list[float]:
        return list(build_cdf_value_grid(self._LOWER, self._UPPER, None, num_points=201))

    def test_below_grid_resolution_reads_low_tail_not_the_clamp(self):
        # Resolution 50 is below every declared value of every member: each curve reads
        # its lowest declared percentile (P10 -> 0.10). The clamp would say 0.90.
        pit = _interpolate_pit(
            50.0,
            self._LOWER,
            self._UPPER,
            self._CDF,
            value_grid=self._grid(),
            per_model_percentiles=self._PERCENTILES,
        )
        assert pit == pytest.approx(0.10, abs=1e-9)

    def test_fallback_is_median_of_member_curves(self):
        # Resolution 95: model-a interpolates 0.6333, model-b reads its P50 = 0.50.
        pit = _interpolate_pit(
            95.0,
            self._LOWER,
            self._UPPER,
            self._CDF,
            value_grid=self._grid(),
            per_model_percentiles=self._PERCENTILES,
        )
        assert pit == pytest.approx((0.6333333 + 0.50) / 2, abs=1e-6)

    def test_no_member_curves_keeps_grid_read(self):
        # Degraded path (no per-model percentiles recoverable): grid-endpoint read kept.
        pit = _interpolate_pit(50.0, self._LOWER, self._UPPER, self._CDF, value_grid=self._grid())
        assert pit == pytest.approx(0.90, abs=1e-9)

    def test_at_bound_resolution_keeps_endpoint_read(self):
        # AT a bound the clamp IS the correct PIT (F(bound) = cdf[0]); the fallback
        # must only engage strictly beyond the grid.
        pit = _interpolate_pit(
            self._LOWER,
            self._LOWER,
            self._UPPER,
            self._CDF,
            value_grid=self._grid(),
            per_model_percentiles=self._PERCENTILES,
        )
        assert pit == pytest.approx(0.90, abs=1e-9)

    def test_numeric_pit_analysis_uses_declared_fallback(self):
        record = {
            "post_id": 1,
            "type": "numeric",
            "our_forecast_values": self._CDF,
            "resolution_parsed": 50.0,
            "scaling": {
                "range_min": self._LOWER,
                "range_max": self._UPPER,
                "zero_point": None,
                "continuous_range": self._grid(),
            },
            "open_lower_bound": True,
            "open_upper_bound": True,
            "numeric_log_score": 0.0,
            "per_model_numeric_percentiles": self._PERCENTILES,
            "metadata": {"category": None},
        }
        result = numeric_pit_analysis([record])
        assert result["count"] == 1
        assert result["pit_values"][0] == pytest.approx(0.10, abs=1e-9)


class TestDeclaredPercentilePitDropsDegenerateRanges:
    """A zero-width question contributes no PIT rather than the most favorable one.

    ``_interpolate_pit`` used to answer 0.5 there, which is inside both coverage bands, so a
    degenerate record silently improved every calibration statistic it entered.
    """

    @staticmethod
    def _record(range_min: float, range_max: float) -> dict:
        return {
            "post_id": 1,
            "type": "numeric",
            "our_forecast_values": list(np.linspace(0.0, 1.0, 201)),
            "resolution_parsed": 5.0,
            "scaling": {"range_min": range_min, "range_max": range_max, "zero_point": None},
            "open_lower_bound": False,
            "open_upper_bound": False,
            "numeric_log_score": 0.0,
            "metadata": {"category": None},
        }

    def test_zero_width_record_is_dropped_not_scored_at_half(self):
        assert numeric_pit_analysis([self._record(10.0, 10.0)]) == {"count": 0}

    def test_an_inverted_range_is_dropped_too(self):
        assert numeric_pit_analysis([self._record(10.0, 5.0)]) == {"count": 0}

    def test_a_real_range_still_scores(self):
        result = numeric_pit_analysis([self._record(0.0, 100.0)])

        assert result["count"] == 1


class TestDeclaredPercentileCurveTolerance:
    """Member curves come out of comment TEXT, so the fallback must tolerate junk.

    Every unusable curve reads as no-curve (dropped from the median) rather than
    raising or contributing a garbage quantile — the callers then either median the
    surviving curves or fall back to the grid read.
    """

    _GOOD = [[10.0, 85.0], [50.0, 95.0], [90.0, 110.0]]

    def test_non_numeric_declared_value_drops_only_that_curve(self):
        # A percentile line that parsed to a non-number: the median is taken over the
        # surviving curve alone (model-b at 50 reads its P10 = 0.10), not over a
        # coerced zero that would drag the quantile.
        curves = cast(
            "dict[str, list[list[float]]]",
            {"model-a": [[10.0, "n/a"], [50.0, 90.0], [90.0, 105.0]], "model-b": self._GOOD},
        )
        assert declared_percentile_pit(curves, 50.0) == pytest.approx(0.10, abs=1e-9)

    def test_pair_missing_its_value_is_unusable(self):
        # A truncated line recovered as a bare percentile with no value.
        assert _single_curve_pit([[10.0], [50.0]], 50.0) is None

    def test_anonymous_keys_are_excluded_from_the_median_of_members(self):
        """A positional ``Forecaster N`` bucket on a stacker-fired record holds the STACKER's
        aggregate, so pooling it into a median-of-members counts the aggregate as an extra
        member and pulls the median toward itself. ``max_step_clamp_screen`` next door and
        ``per_model_cohort`` both filter these; this consumer used not to."""
        curves = cast(
            "dict[str, list[list[float]]]",
            {"model-a": self._GOOD, "Forecaster 1": [[10.0, 10.0], [50.0, 12.0], [90.0, 14.0]]},
        )

        # The anonymous curve would read ~0.90 at resolution 50 and swing the median.
        assert declared_percentile_pit(curves, 50.0) == pytest.approx(0.10, abs=1e-9)

    def test_an_all_anonymous_record_yields_none_rather_than_the_stacker_curve(self):
        curves = cast("dict[str, list[list[float]]]", {"Forecaster 1": self._GOOD})

        assert declared_percentile_pit(curves, 50.0) is None

    def test_duplicate_declared_values_stay_usable(self):
        # A flat tail (P10 == P50) is legitimate model output: jitter it into strict
        # monotonicity rather than discarding the whole curve.
        flat_tail = [[10.0, 80.0], [50.0, 80.0], [90.0, 105.0]]
        assert _single_curve_pit(flat_tail, 50.0) == pytest.approx(0.10, abs=1e-9)
        # Between the duplicated value and P90 the curve still interpolates.
        mid = _single_curve_pit(flat_tail, 90.0)
        assert mid is not None
        assert 0.5 < mid < 0.9

    def test_non_finite_declared_values_are_unusable(self):
        # Jitter cannot rescue non-finite values; the curve must read as no-curve
        # instead of returning a nan PIT into the median. errstate only silences the
        # expected nan arithmetic inside the guard, which is the code under test.
        with np.errstate(invalid="ignore"):
            assert _single_curve_pit([[10.0, float("inf")], [90.0, float("inf")]], 50.0) is None
            assert _single_curve_pit([[10.0, float("nan")], [50.0, 90.0]], 50.0) is None

    def test_all_curves_unusable_reads_as_no_fallback(self):
        # declared_percentile_pit returning None is what makes _interpolate_pit /
        # compute_pit_details keep the grid-endpoint read.
        junk = cast("dict[str, list[list[float]]]", {"model-a": [[50.0, "junk"]]})
        assert declared_percentile_pit(junk, 50.0) is None
        assert declared_percentile_pit(None, 50.0) is None


class TestMaxStepClampScreen:
    """The q43913 signature: a published bin pinned at the per-bin max-step cap while
    every member's own declared curve wanted materially more mass there. The cap is
    era-correct — flat 0.2 before the grid-scaled cap reached main (b4e9df0), the
    grid's own ``grid_step_constraints`` max after — so a post-fix coarse-grid
    discrete that legitimately holds a 0.2 bin must NOT fire."""

    # 11-point integer grid; steps[1] (the [1, 2] bin) is exactly 0.20.
    _GRID = [float(v) for v in range(11)]
    _CDF = [0.0, 0.05, 0.25, 0.45, 0.65, 0.85, 0.90, 0.93, 0.96, 0.98, 1.0]
    # Both members concentrate ~0.70 of their mass on the [1, 2] bin. Curves are
    # 11-ANCHOR on purpose: the screen now drops any member under
    # MIN_SCOREABLE_ANCHORS, because its verdict turns on the MINIMUM member bin mass
    # and a 3-anchor interpolation across one bin is not the declared distribution.
    _LABELS = [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0]
    _MEMBERS = {
        "model-a": [
            [label, value]
            for label, value in zip(_LABELS, [0.90, 1.00, 1.15, 1.30, 1.45, 1.55, 1.70, 1.85, 2.00, 2.30, 2.60])
        ],
        "model-b": [
            [label, value]
            for label, value in zip(_LABELS, [0.95, 1.05, 1.18, 1.32, 1.46, 1.56, 1.72, 1.90, 2.05, 2.35, 2.65])
        ],
    }
    _PRE_FIX_TS = "2026-06-11T00:00:00Z"
    _POST_FIX_TS = "2026-08-01T00:00:00Z"

    def _record(
        self, *, submitted, members=None, cdf=None, grid=None, resolution: float | str = 1.4, q_type="discrete"
    ) -> dict:
        grid = grid if grid is not None else self._GRID
        return {
            "type": q_type,
            "our_forecast_values": cdf if cdf is not None else self._CDF,
            "resolution_parsed": resolution,
            "scaling": {"range_min": grid[0], "range_max": grid[-1], "continuous_range": grid},
            "bot_comment_created_at": submitted,
            "per_model_numeric_percentiles": members if members is not None else self._MEMBERS,
        }

    def test_pre_fix_coarse_grid_clamp_is_suspected(self):
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS))
        assert screen["suspected"] is True
        assert screen["submitted_before_grid_scaled_cap"] is True
        assert screen["max_step_cap"] == pytest.approx(0.2)
        assert screen["published_bin_mass"] == pytest.approx(0.2, abs=1e-9)
        assert screen["min_member_bin_mass"] > 0.6

    def test_post_fix_coarse_grid_point_two_bin_does_not_fire(self):
        # After 9f1175c an 11-point grid's cap is 1.0, so a 0.2 bin means nothing.
        screen = max_step_clamp_screen(self._record(submitted=self._POST_FIX_TS))
        assert screen["suspected"] is False
        assert screen["submitted_before_grid_scaled_cap"] is False
        assert screen["max_step_cap"] == pytest.approx(1.0)
        assert screen["resolution_bin_at_cap"] is False

    def test_post_fix_standard_grid_cap_still_fires(self):
        # On the 201-point grid the era-correct cap is still 0.2, so the screen keeps
        # catching genuine clamps after the fix.
        steps = np.full(200, 0.8 / 199)
        steps[100] = 0.2
        cdf = np.concatenate([[0.0], np.cumsum(steps)]).tolist()
        grid = np.linspace(0.0, 200.0, 201).tolist()
        # 11-anchor curves (see _MEMBERS): each puts ~0.70 on the [100, 101] bin.
        members = {
            "model-a": [
                [label, value]
                for label, value in zip(
                    self._LABELS,
                    [99.90, 100.00, 100.15, 100.30, 100.45, 100.55, 100.70, 100.85, 101.00, 101.30, 101.60],
                )
            ],
            "model-b": [
                [label, value]
                for label, value in zip(
                    self._LABELS,
                    [99.95, 100.05, 100.18, 100.32, 100.46, 100.56, 100.72, 100.90, 101.05, 101.35, 101.65],
                )
            ],
        }
        screen = max_step_clamp_screen(
            self._record(submitted=self._POST_FIX_TS, members=members, cdf=cdf, grid=grid, resolution=100.5)
        )
        assert screen["max_step_cap"] == pytest.approx(0.2)
        assert screen["suspected"] is True

    def test_missing_timestamp_treated_as_pre_fix(self):
        screen = max_step_clamp_screen(self._record(submitted=None))
        assert screen["submitted_before_grid_scaled_cap"] is True
        assert screen["max_step_cap"] == pytest.approx(0.2)

    def test_unparseable_timestamp_treated_as_pre_fix(self):
        # Same rule as a missing one: the undated (and undatable) archive records all
        # predate the fix, so an unreadable timestamp must not be read as post-fix.
        screen = max_step_clamp_screen(self._record(submitted="not-a-date"))
        assert screen["submitted_before_grid_scaled_cap"] is True
        assert screen["max_step_cap"] == pytest.approx(0.2)

    def test_timestamp_with_an_offset_is_compared_in_utc(self):
        # A post-fix instant written with a local offset must read as post-fix: a naive
        # (offset-dropping) comparison shifts it hours across the boundary.
        screen = max_step_clamp_screen(self._record(submitted="2026-07-21T11:07:37-07:00"))
        assert screen["submitted_before_grid_scaled_cap"] is False

    def test_members_not_materially_more_does_not_fire(self):
        # Members' own curves put ~0.2 on the bin too: the cap coincided with what
        # the ensemble wanted, so nothing was overridden.
        diffuse = {
            "model-a": [[10.0, 0.0], [50.0, 3.0], [90.0, 8.0]],
            "model-b": [[10.0, 0.5], [50.0, 3.5], [90.0, 8.5]],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=diffuse))
        assert screen["resolution_bin_at_cap"] is True
        assert screen["suspected"] is False

    def test_single_member_curve_does_not_fire(self):
        one = {"model-a": self._MEMBERS["model-a"]}
        assert max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=one))["suspected"] is False

    def test_anonymous_member_keys_are_excluded(self):
        # A positional key on a stacked record can hold the stacker's aggregate.
        anon = {"Forecaster 1": self._MEMBERS["model-a"], "Forecaster 2": self._MEMBERS["model-b"]}
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=anon))
        assert screen["member_bin_masses"] == {}
        assert screen["suspected"] is False

    def test_resolution_exactly_on_grid_point_screens_the_bin_below(self):
        # A resolution sitting exactly ON a grid edge belongs to the bin BELOW it —
        # the platform scorer's convention (resolution_to_bucket_index). On this
        # fixture, resolution 2.0 must screen the [1, 2] bin whose 0.20 step sits at
        # the pre-fix cap; the old side="right" screened [2, 3] and missed the
        # q43913 signature entirely.
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, resolution=2.0))
        assert screen["resolution_bin"] == [1.0, 2.0]
        assert screen["published_bin_mass"] == pytest.approx(0.2, abs=1e-9)
        assert screen["suspected"] is True

    def test_single_pair_member_curve_is_unusable(self):
        # One recovered (percentile, value) pair interpolates to a constant PIT at
        # every resolution — the member is dropped, leaving one usable curve, which
        # is below the >=2-curves requirement.
        one_pair = {
            "model-a": [[50.0, 90.0]],
            "model-b": self._MEMBERS["model-b"],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=one_pair))
        assert list(screen["member_bin_masses"]) == ["model-b"]
        assert screen["suspected"] is False

    def test_a_sparse_member_curve_is_excluded_from_the_min(self):
        """The verdict turns on the MINIMUM member bin mass, so one sparse recovery can
        decide it — and a 3-anchor interpolation across one bin is not the distribution the
        model declared. q43913's KNOWN_BUG_QIDS entry survives this gate on its own
        11-anchor member; the 3-anchor sibling never decided that verdict."""
        mixed = {
            "model-a": self._MEMBERS["model-a"],
            "model-sparse": [[10.0, 0.0], [50.0, 3.0], [90.0, 8.0]],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=mixed))

        assert list(screen["member_bin_masses"]) == ["model-a"]
        # One usable curve is below the >=2-curves requirement, so nothing is suspected.
        assert screen["suspected"] is False

    def test_a_uniformly_sparse_record_reports_no_member_masses(self):
        # The sparse-era shape: no curve clears the floor, so the screen has no member
        # evidence at all rather than ranking equals against each other (a bin-mass
        # comparison is absolute, unlike ranking_cohort's relative one).
        sparse = {
            "model-a": [[10.0, 0.9], [50.0, 1.3], [90.0, 2.1]],
            "model-b": [[10.0, 0.95], [50.0, 1.4], [90.0, 2.2]],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=sparse))

        assert screen["member_bin_masses"] == {}
        assert screen["min_member_bin_mass"] is None
        assert screen["suspected"] is False

    def test_non_monotonic_member_curve_is_unusable(self):
        # Percentiles that DECREASE as values increase invert the curve — drop it.
        inverted = {
            "model-a": [[90.0, 0.9], [50.0, 1.3], [10.0, 2.1]],
            "model-b": self._MEMBERS["model-b"],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=inverted))
        assert list(screen["member_bin_masses"]) == ["model-b"]

    def test_non_numeric_resolution_and_type_gates(self):
        assert max_step_clamp_screen({"type": "binary"})["applicable"] is False
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, resolution="below_lower_bound"))
        assert screen["suspected"] is False
        assert screen["reason"] == "non-numeric resolution"

    def test_records_without_a_usable_grid_report_that_reason(self):
        # The screen needs the question's own value grid to locate the realized bin.
        # Comment-backfilled records often carry no continuous_range, and a grid whose
        # length disagrees with the published CDF can't be indexed either — both must
        # report a reason instead of screening an arbitrary bin.
        no_grid = self._record(submitted=self._PRE_FIX_TS)
        no_grid["scaling"] = {"range_min": 0.0, "range_max": 10.0}
        assert max_step_clamp_screen(no_grid)["reason"] == "no usable grid"

        mismatched = self._record(submitted=self._PRE_FIX_TS, grid=[0.0, 1.0, 2.0])
        assert max_step_clamp_screen(mismatched)["reason"] == "no usable grid"
        assert max_step_clamp_screen(mismatched)["suspected"] is False


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

    def test_zero_point_zero_without_continuous_range_reconstructs_geometric(self):
        # Regression for the zero_point sentinel bug on the analysis fallback path:
        # a log-scale record serializes zero_point==0 with range_min>0 but carries
        # NO continuous_range (old archive / schema drift). numeric_pit_analysis
        # must reconstruct the GEOMETRIC grid (via grid_zero_point), not a linear
        # one. On [1, 1000] the geometric midpoint (~31.6) is PIT ~0.5; the buggy
        # linear-grid reconstruction would call it near-zero.
        cdf = list(np.linspace(0.0, 1.0, 201))
        log_rec = self._record(1, cdf, 31.6, 1.0, 1000.0, 0, None)
        result = numeric_pit_analysis([log_rec])
        assert result["count"] == 1
        assert result["pit_values"][0] == pytest.approx(0.5, abs=0.02)


class TestRescoreRecords:
    """Stale stored scores in cached datasets must self-heal on load.

    Scores are pure functions of fields the record carries, but a cached JSON's
    score VALUES are whatever the scorer computed when the file was written — a
    scorer fix never reaches previously-saved files. The checked-in q38991
    fixture is the real record that carried a linear-bucket numeric_log_score of
    −193.29 for a month after the zero_point coercion fix, against a platform
    spot_baseline_score of 165.54.
    """

    _FIXTURE = Path(__file__).parent / "data" / "q38991_stale_zero_point_score.json"

    def _stale_record(self) -> dict:
        return json.loads(self._FIXTURE.read_text())

    def test_zero_point_record_rescores_to_platform_value(self):
        record = self._stale_record()
        assert record["scaling"]["zero_point"] == 0
        assert record["numeric_log_score"] == pytest.approx(-193.292, abs=1e-3)

        changed = rescore_records([record])

        assert changed == 1
        assert record["numeric_log_score"] == pytest.approx(record["metaculus_scores"]["spot_baseline_score"], abs=1e-9)

    def test_fresh_scores_left_untouched(self):
        record = self._stale_record()
        rescore_records([record])
        healed = record["numeric_log_score"]
        assert rescore_records([record]) == 0
        assert record["numeric_log_score"] == healed

    def test_unrecomputable_score_is_never_deleted(self):
        # Missing scaling bounds -> recomputation yields None -> keep the stored value.
        record = self._stale_record()
        record["scaling"] = {}
        stored = record["numeric_log_score"]
        assert rescore_records([record]) == 0
        assert record["numeric_log_score"] == stored

    def test_load_dataset_heals_stale_scores(self, tmp_path: Path):
        path = tmp_path / "cached.json"
        path.write_text(json.dumps([self._stale_record()]))
        (loaded,) = load_dataset(str(path))
        assert loaded["numeric_log_score"] == pytest.approx(loaded["metaculus_scores"]["spot_baseline_score"], abs=1e-9)

    def test_malformed_record_is_skipped(self):
        assert rescore_records([{"no_type": True}, "not-a-dict"]) == 0  # type: ignore[list-item]

    def test_partial_records_are_skipped_not_crashed(self):
        # rescore takes arbitrary cached JSON: every record missing a field that
        # _compute_scores subscripts must be skipped, never raise KeyError.
        partial = [
            {"type": "binary", "resolution_parsed": True},  # no our_forecast_values
            {"type": "binary", "resolution_parsed": True, "our_forecast_values": [0.3, 0.7]},  # no our_prob_yes
            {  # numeric without open-bound flags
                "type": "numeric",
                "resolution_parsed": 5.0,
                "our_forecast_values": [0.0, 0.5, 1.0],
                "scaling": {"range_min": 0.0, "range_max": 10.0},
            },
        ]
        assert rescore_records(partial) == 0
        assert "brier_score" not in partial[0]

    def test_load_dataset_survives_partial_records(self, tmp_path: Path):
        path = tmp_path / "cached.json"
        path.write_text(json.dumps([{"type": "binary", "resolution_parsed": True}, self._stale_record()]))
        loaded = load_dataset(str(path))
        assert len(loaded) == 2
        assert loaded[1]["numeric_log_score"] == pytest.approx(
            loaded[1]["metaculus_scores"]["spot_baseline_score"], abs=1e-9
        )

    def test_load_dataset_is_idempotent_on_an_already_healed_file(self, tmp_path: Path):
        # Re-loading a file whose scores already agree with the scorer must leave every
        # value byte-identical: healing is a repair, not a rewrite of live data.
        healed = self._stale_record()
        rescore_records([healed])
        path = tmp_path / "healed.json"
        path.write_text(json.dumps([healed]))
        (reloaded,) = load_dataset(str(path))
        assert reloaded["numeric_log_score"] == healed["numeric_log_score"]

    def test_scoring_failure_on_a_record_without_post_id_only_warns(self, caplog):
        # rescore walks arbitrary cached JSON, so a record can lack post_id entirely.
        # The scoring-failure log lines must read it defensively — a subscript there
        # turns one unscoreable record into a KeyError that kills the whole load.
        unscoreable_numeric = {
            "type": "numeric",
            "resolution_parsed": 5.0,
            "our_forecast_values": [0.5],  # < 2 CDF points -> numeric_log_score raises
            "open_lower_bound": False,
            "open_upper_bound": False,
            "scaling": {"range_min": 0.0, "range_max": 10.0, "zero_point": None},
            "numeric_log_score": -1.0,
        }
        unscoreable_mc = {
            "type": "multiple_choice",
            "resolution_parsed": "B",
            "our_forecast_values": [1.0],  # fewer probabilities than options
            "options": ["A", "B"],
            "mc_log_score": -1.0,
        }
        with caplog.at_level(logging.WARNING):
            assert rescore_records([unscoreable_numeric, unscoreable_mc]) == 0

        messages = [r.getMessage() for r in caplog.records]
        assert any("Failed numeric scoring for post None" in m for m in messages)
        assert any("Failed MC scoring for post None" in m for m in messages)
        # The stored values survive: healing never deletes a score it can't recompute.
        assert unscoreable_numeric["numeric_log_score"] == -1.0
        assert unscoreable_mc["mc_log_score"] == -1.0


class TestBuildPerformanceDatasetResearchTags:
    """The dataset the analysis reads must arrive with the treatment tags stamped.

    ``attach_research_tags`` is a single call inside ``build_performance_dataset``;
    unit-testing the tagger alone leaves that pass-through deletable with a green
    suite, and every treated/untreated calibration cut reads these fields off the
    built dataset.
    """

    def _post(self, post_id: int, question_id: int) -> dict:
        return {
            "id": post_id,
            "title": f"Q{post_id}",
            "question": {
                "id": question_id,
                "type": "binary",
                "resolution": "yes",
                "my_forecasts": {"latest": {"forecast_values": [0.3, 0.7], "score_data": {}}},
                "scaling": {},
                "options": None,
                "open_lower_bound": False,
                "open_upper_bound": False,
                "nr_forecasters": 5,
                "title": f"Q{post_id}",
            },
            "projects": {},
        }

    def test_records_carry_tags_from_the_archive_dir(self, tmp_path: Path, monkeypatch):
        (tmp_path / "11.json").write_text(
            json.dumps(
                {
                    "research_text": "## Time Series Anchor\nband\n## Agentic Research Findings\nfindings\n",
                    "source": "artifact",
                    "gap_fill_v2": {"steps": 4},
                }
            )
        )
        monkeypatch.setattr(collector, "fetch_resolved_questions", lambda tournament, token: [self._post(1, 11)])
        monkeypatch.setattr(
            collector,
            "fetch_bot_comments",
            lambda author_id, token: [{"id": 9, "text": "*Forecaster 1*: 70%\n", "on_post": 1}],
        )

        records = build_performance_dataset(tournament="t", token="fake", research_archive_dir=tmp_path)

        assert len(records) == 1
        assert records[0]["anchor_present"] is True
        assert records[0]["gfv2_present"] is True
        assert records[0]["gfv2_loop_ran"] is True
        assert records[0]["research_source_class"] == "artifact"

    def test_question_without_an_archive_record_gets_none_not_false(self, tmp_path: Path, monkeypatch):
        # Absence of evidence is not an untreated record: a missing archive file (or a
        # whole missing archive) must never look like a measured False in the cuts.
        monkeypatch.setattr(collector, "fetch_resolved_questions", lambda tournament, token: [self._post(2, 22)])
        monkeypatch.setattr(collector, "fetch_bot_comments", lambda author_id, token: [])

        records = build_performance_dataset(tournament="t", token="fake", research_archive_dir=tmp_path)

        assert records[0]["anchor_present"] is None
        assert records[0]["gfv2_present"] is None
        assert records[0]["anchor_confidence"] is None


class TestPackageExports:
    """The residual rounds' out-of-band scripts import these off the package root, so
    the re-export list is a contract, not bookkeeping."""

    def test_new_analysis_helpers_are_re_exported(self):
        for name in (
            "attach_research_tags",
            "research_tags_for_qid",
            "research_tags_for_record",
            "max_step_clamp_screen",
            "rescore_records",
            "parse_stacker_skip_reason_marker",
        ):
            assert name in performance_analysis.__all__, name
            assert getattr(performance_analysis, name) is not None


class TestResolveNumericScoreInputsZeroPoint:
    """Regression for the zero_point sentinel bug in the record-scoring coercion:
    ``resolve_numeric_record_to_score_inputs`` must keep a serialized
    ``zero_point == 0`` (with a positive ``range_min``) as a genuine log-scale
    value, not collapse it to the linear ``None`` sentinel."""

    def _record(self, zero_point: float | int | None, range_min: float, range_max: float) -> dict:
        return {
            "type": "numeric",
            "resolution_parsed": (range_min + range_max) / 2.0,
            "scaling": {"range_min": range_min, "range_max": range_max, "zero_point": zero_point},
        }

    def test_zero_point_zero_stays_log_when_range_min_positive(self):
        # This is the sibling of the width_monitor fix: a log-scale question with a
        # positive floor carries zero_point==0, which must survive as 0.0 so
        # numeric_log_score buckets on the geometric grid.
        inputs = resolve_numeric_record_to_score_inputs(self._record(0, 1.0, 1000.0))
        assert inputs is not None
        _res, _lo, _hi, zero_point = inputs
        assert zero_point == 0.0

    def test_zero_point_zero_dropped_when_range_min_nonpositive(self):
        # A non-positive floor rules out a log transform -> linear (None).
        inputs = resolve_numeric_record_to_score_inputs(self._record(0, 0.0, 100.0))
        assert inputs is not None
        assert inputs[3] is None

    def test_absent_zero_point_is_linear(self):
        inputs = resolve_numeric_record_to_score_inputs(self._record(None, 0.0, 100.0))
        assert inputs is not None
        assert inputs[3] is None

    def test_nonzero_zero_point_passthrough(self):
        inputs = resolve_numeric_record_to_score_inputs(self._record(50, 0.0, 100.0))
        assert inputs is not None
        assert inputs[3] == 50.0
