"""Tests for the aggregate analysis cuts the residual rounds read.

Covers ``mc_summary``, ``no_bias_check``, ``financial_vs_nonfinancial_pit``,
``stacking_effectiveness``, ``disagreement_predicts_error``, and the ``per_model_cohort`` filter
that keeps phantom "Forecaster N" buckets and stacker-fired records out of the per-model cuts.
"""

from __future__ import annotations

import logging

import pytest

from metaculus_bot.performance_analysis.analysis import (
    binary_summary,
    disagreement_predicts_error,
    financial_vs_nonfinancial_pit,
    mc_summary,
    no_bias_check,
    per_model_binary_scores,
    per_model_cohort,
    stacking_effectiveness,
)
from metaculus_bot.performance_analysis.parsing import anonymous_model_key, is_anonymous_model_key
from tests.performance_analysis_fakes import _binary_record, _numeric_record


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
        """Predicting 30% when the actual YES rate is 43% is a -13pp NO-bias."""
        records = [_binary_record(i, 0.30, True) for i in range(43)] + [
            _binary_record(100 + i, 0.30, False) for i in range(57)
        ]
        result = no_bias_check(records)
        assert result["count"] == 100
        assert result["mean_predicted"] == pytest.approx(0.30)
        assert result["actual_yes_rate"] == pytest.approx(0.43)
        assert result["bias_pp"] == pytest.approx(-13.0)

    def test_reports_low_range_subset(self):
        """20 records inside the 0.10-0.30 bucket: mean predicted ~0.205, actual yes-rate 0.50
        (10 of 20 resolve YES). The 5 records at 0.70 sit outside the bucket and must not leak
        into the low_range stats."""
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


class TestFinancialVsNonfinancialPit:
    def test_splits_by_category(self):
        """Simple linear CDFs, so the PIT each record reads is predictable."""
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


class TestStackingEffectiveness:
    def test_computes_counterfactual_mean_brier_on_triggered(self):
        """Triggered means the per-model probability range exceeds the threshold."""
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


class TestDisagreementPredictsError:
    def test_positive_correlation_on_disagreement_and_error(self):
        """The records are built so the high-spread questions are also the high-Brier ones."""
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
        """Under 3 records there is no meaningful correlation to compute.

        Each record still carries a per_model dict, so it actually contributes to the spread
        correlation and the None comes from the record count rather than from empty input.
        """
        records = [
            _binary_record(1, 0.5, True, per_model={"m1": "40%", "m2": "60%"}),
            _binary_record(2, 0.6, True, per_model={"m1": "50%", "m2": "70%"}),
        ]
        result = disagreement_predicts_error(records)
        assert result["count"] == 2
        assert result["spearman_rho"] is None  # n<3


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
        """The mirror of the stacker-fired case: a record the detector confirms ran on MEDIAN
        keeps its per-model bullets."""
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
        """``likely_stacker`` (high spread plus a published value far from the median) must NOT
        exclude a record: that shape is also what a MEAN-era aggregate looks like, and dropping
        it would silently remove the high-disagreement records these cuts exist to measure."""
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
        """The aggregates keep stacked and anonymously-attributed records by decision; only the
        per-MODEL cuts drop them, so both aggregate paths must count all three records here."""
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
            # Near-misses, spelled out in the docstring below.
            "Forecaster",
            "Forecaster One",
            "Forecaster 1 (gpt-5.6-sol)",
            "*Forecaster 1*",
        ],
    )
    def test_model_names_are_not_anonymous(self, key):
        """A real model name is never anonymous, and neither are the near-misses: display names
        that merely start the same way as the positional format, and a bullet-shaped string."""
        assert not is_anonymous_model_key(key)
