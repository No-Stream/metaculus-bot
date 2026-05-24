"""Tests for metaculus_bot.performance_analysis.stacker_detection module.

Verifies the multi-signal stacker-detection helper that combines explicit flags,
body markers, spread thresholds, and production-vs-median deltas to classify
whether a given historical record was stacked.
"""

from __future__ import annotations

import pytest

from metaculus_bot.performance_analysis.stacker_detection import (
    compute_production_vs_median_delta,
    detect_stacker_fired,
    exceeded_spread_threshold,
    get_stacker_outcome_field,
    has_stacker_body_marker,
    has_was_stacked_flag,
)

# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


class TestHasWasStackedFlag:
    def test_true(self):
        assert has_was_stacked_flag({"was_stacked": True}) is True

    def test_false(self):
        assert has_was_stacked_flag({"was_stacked": False}) is False

    def test_missing(self):
        assert has_was_stacked_flag({"question_id": 123}) is None

    def test_none_value(self):
        assert has_was_stacked_flag({"was_stacked": None}) is None


class TestGetStackerOutcomeField:
    def test_primary(self):
        assert get_stacker_outcome_field({"stacker_outcome": "primary"}) == "primary"

    def test_fallback_llm(self):
        assert get_stacker_outcome_field({"stacker_outcome": "fallback_llm"}) == "fallback_llm"

    def test_fallback_median(self):
        assert get_stacker_outcome_field({"stacker_outcome": "fallback_median"}) == "fallback_median"

    def test_skipped(self):
        assert get_stacker_outcome_field({"stacker_outcome": "skipped"}) == "skipped"

    def test_missing(self):
        assert get_stacker_outcome_field({"question_id": 123}) is None

    def test_none_value(self):
        assert get_stacker_outcome_field({"stacker_outcome": None}) is None

    def test_empty_string(self):
        assert get_stacker_outcome_field({"stacker_outcome": ""}) is None


class TestHasStackerBodyMarker:
    def test_stacker_outcome_marker(self):
        record = {"comment_text": "some text <!-- STACKER_OUTCOME=primary --> more text"}
        assert has_stacker_body_marker(record) is True

    def test_stacker_outcome_fallback_llm(self):
        record = {"comment_text": "text <!-- STACKER_OUTCOME=fallback_llm --> end"}
        assert has_stacker_body_marker(record) is True

    def test_stacker_outcome_skipped(self):
        # skipped means it did NOT fire — this should return False
        record = {"comment_text": "text <!-- STACKER_OUTCOME=skipped --> end"}
        assert has_stacker_body_marker(record) is False

    def test_stacker_outcome_fallback_median(self):
        # fallback_median means stacker failed, fell back to median — NOT stacker
        record = {"comment_text": "text <!-- STACKER_OUTCOME=fallback_median --> end"}
        assert has_stacker_body_marker(record) is False

    def test_stacked_true_legacy(self):
        record = {"comment_text": "text <!-- STACKED=true --> end"}
        assert has_stacker_body_marker(record) is True

    def test_stacked_false_legacy(self):
        record = {"comment_text": "text <!-- STACKED=false --> end"}
        assert has_stacker_body_marker(record) is False

    def test_tools_used_alone_not_stacker(self):
        # TOOLS_USED alone doesn't indicate stacking
        record = {"comment_text": "text <!-- TOOLS_USED=true --> end"}
        assert has_stacker_body_marker(record) is None

    def test_historical_body_signature(self):
        record = {
            "comment_text": (
                "## R1: Forecaster 1 Reasoning\n"
                "Model: openrouter/anthropic/claude-opus-4.5\n"
                "## Stacker Meta-Analysis\n"
                "The models disagree..."
            )
        }
        assert has_stacker_body_marker(record) is True

    def test_no_marker(self):
        record = {"comment_text": "just a regular comment with no markers"}
        assert has_stacker_body_marker(record) is None

    def test_missing_comment_text(self):
        record = {"question_id": 123}
        assert has_stacker_body_marker(record) is None


class TestComputeProductionVsMedianDelta:
    def test_binary_record(self):
        record = {
            "type": "binary",
            "our_prob_yes": 0.70,
            "per_model_forecasts": {
                "gpt-5.5": "60%",
                "claude-opus-4.7": "80%",
                "gemini-3.1-pro": "70%",
            },
        }
        # median of [0.6, 0.7, 0.8] = 0.7, delta = |0.70 - 0.70| = 0.0
        assert compute_production_vs_median_delta(record) == pytest.approx(0.0, abs=1e-6)

    def test_binary_record_with_delta(self):
        record = {
            "type": "binary",
            "our_prob_yes": 0.85,
            "per_model_forecasts": {
                "gpt-5.5": "60%",
                "claude-opus-4.7": "80%",
                "gemini-3.1-pro": "70%",
            },
        }
        # median of [0.6, 0.7, 0.8] = 0.7, delta = |0.85 - 0.70| = 0.15
        assert compute_production_vs_median_delta(record) == pytest.approx(0.15, abs=1e-6)

    def test_missing_production_value(self):
        record = {
            "type": "binary",
            "per_model_forecasts": {"gpt-5.5": "60%", "claude-opus-4.7": "80%"},
        }
        assert compute_production_vs_median_delta(record) is None

    def test_insufficient_models(self):
        record = {
            "type": "binary",
            "our_prob_yes": 0.70,
            "per_model_forecasts": {"gpt-5.5": "60%"},
        }
        assert compute_production_vs_median_delta(record) is None

    def test_numeric_record_returns_none_when_no_cdf(self):
        # Numeric delta requires CDF comparison which is complex; without
        # our_forecast_values it should return None
        record = {
            "type": "numeric",
            "per_model_numeric_percentiles": {"a": [[2.5, 10], [50, 50], [97.5, 90]]},
        }
        assert compute_production_vs_median_delta(record) is None


class TestExceededSpreadThreshold:
    def test_binary_high_spread(self):
        record = {
            "type": "binary",
            "per_model_forecasts": {
                "gpt-5.5": "90%",
                "claude-opus-4.7": "50%",
                "gemini-3.1-pro": "70%",
            },
        }
        # spread = 0.9 - 0.5 = 0.4 > 0.15
        assert exceeded_spread_threshold(record) is True

    def test_binary_low_spread(self):
        record = {
            "type": "binary",
            "per_model_forecasts": {
                "gpt-5.5": "70%",
                "claude-opus-4.7": "75%",
                "gemini-3.1-pro": "72%",
            },
        }
        # spread = 0.75 - 0.70 = 0.05 < 0.15
        assert exceeded_spread_threshold(record) is False

    def test_binary_at_threshold(self):
        # Use values that compute to exactly the threshold in float
        # 0.70 - 0.55 = 0.15000000000000002 in float64 (> threshold due to precision)
        # In production, the strict > comparison means float-precision boundary cases trigger.
        # Use values that clearly don't exceed: 0.72 - 0.58 = 0.14
        record = {
            "type": "binary",
            "per_model_forecasts": {
                "gpt-5.5": "58%",
                "claude-opus-4.7": "72%",
            },
        }
        # spread = 0.72 - 0.58 = 0.14 < 0.15
        assert exceeded_spread_threshold(record) is False

    def test_missing_per_model(self):
        record = {"type": "binary"}
        assert exceeded_spread_threshold(record) is None

    def test_mc_high_spread(self):
        record = {
            "type": "multiple_choice",
            "per_model_mc_forecasts": {
                "gpt-5.5": [{"option_name": "A", "probability": 0.8}, {"option_name": "B", "probability": 0.2}],
                "claude-opus-4.7": [{"option_name": "A", "probability": 0.5}, {"option_name": "B", "probability": 0.5}],
            },
        }
        # max option spread: A = 0.8-0.5=0.3, B = 0.5-0.2=0.3. max=0.3 > 0.20
        assert exceeded_spread_threshold(record) is True

    def test_mc_low_spread(self):
        record = {
            "type": "multiple_choice",
            "per_model_mc_forecasts": {
                "gpt-5.5": [{"option_name": "A", "probability": 0.6}, {"option_name": "B", "probability": 0.4}],
                "claude-opus-4.7": [
                    {"option_name": "A", "probability": 0.65},
                    {"option_name": "B", "probability": 0.35},
                ],
            },
        }
        # max option spread: A = 0.65-0.6=0.05, B = 0.4-0.35=0.05. max=0.05 < 0.20
        assert exceeded_spread_threshold(record) is False


# ---------------------------------------------------------------------------
# Main detect_stacker_fired tests
# ---------------------------------------------------------------------------


class TestDetectStackerFired:
    def test_was_stacked_true_confirmed_stacker(self):
        """Signal 1: explicit was_stacked=True → confirmed_stacker."""
        record = {"was_stacked": True, "type": "binary"}
        assert detect_stacker_fired(record) == "confirmed_stacker"

    def test_was_stacked_false_low_spread_confirmed_median(self):
        """was_stacked=False AND low spread → confirmed_median."""
        record = {
            "was_stacked": False,
            "type": "binary",
            "per_model_forecasts": {
                "gpt-5.5": "70%",
                "claude-opus-4.7": "72%",
            },
        }
        assert detect_stacker_fired(record) == "confirmed_median"

    def test_was_stacked_false_but_high_spread_and_production_differs(self):
        """was_stacked=False but high spread + production differs from median → likely_stacker.

        This is the "flag may be wrong" case — the record says no stacking but
        the evidence says otherwise.
        """
        record = {
            "was_stacked": False,
            "type": "binary",
            "our_prob_yes": 0.85,
            "per_model_forecasts": {
                "gpt-5.5": "60%",
                "claude-opus-4.7": "80%",
                "gemini-3.1-pro": "50%",
            },
        }
        # median = 0.60, spread = 0.80-0.50 = 0.30 > 0.15, |0.85 - 0.60| = 0.25 > 0.05
        assert detect_stacker_fired(record) == "likely_stacker"

    def test_no_flags_low_spread_likely_median(self):
        """No explicit flags + low spread → likely_median."""
        record = {
            "type": "binary",
            "our_prob_yes": 0.71,
            "per_model_forecasts": {
                "gpt-5.5": "70%",
                "claude-opus-4.7": "72%",
                "gemini-3.1-pro": "71%",
            },
        }
        assert detect_stacker_fired(record) == "likely_median"

    def test_no_flags_high_spread_production_matches_median(self):
        """No flags + high spread + production matches median → likely_median.

        Stacker may have failed and fallen back to median.
        """
        record = {
            "type": "binary",
            "our_prob_yes": 0.70,
            "per_model_forecasts": {
                "gpt-5.5": "50%",
                "claude-opus-4.7": "90%",
                "gemini-3.1-pro": "70%",
            },
        }
        # median = 0.70, spread = 0.9-0.5 = 0.4 > 0.15, |0.70 - 0.70| = 0.0 <= 0.05
        assert detect_stacker_fired(record) == "likely_median"

    def test_no_fields_unknown(self):
        """Completely empty record → unknown."""
        record = {"question_id": 999}
        assert detect_stacker_fired(record) == "unknown"

    def test_body_marker_stacker_outcome_primary(self):
        """Signal 3: body marker STACKER_OUTCOME=primary → confirmed_stacker."""
        record = {
            "type": "binary",
            "comment_text": "some text <!-- STACKER_OUTCOME=primary --> more",
        }
        assert detect_stacker_fired(record) == "confirmed_stacker"

    def test_body_marker_stacker_outcome_fallback_llm(self):
        """STACKER_OUTCOME=fallback_llm also means stacker fired."""
        record = {
            "type": "binary",
            "comment_text": "text <!-- STACKER_OUTCOME=fallback_llm --> end",
        }
        assert detect_stacker_fired(record) == "confirmed_stacker"

    def test_stacker_outcome_field_primary(self):
        """Signal 2: stacker_outcome field present → confirmed_stacker."""
        record = {"stacker_outcome": "primary", "type": "binary"}
        assert detect_stacker_fired(record) == "confirmed_stacker"

    def test_stacker_outcome_field_skipped(self):
        """stacker_outcome=skipped + low spread → confirmed_median."""
        record = {
            "stacker_outcome": "skipped",
            "type": "binary",
            "per_model_forecasts": {
                "gpt-5.5": "70%",
                "claude-opus-4.7": "72%",
            },
        }
        assert detect_stacker_fired(record) == "confirmed_median"

    def test_stacker_outcome_field_fallback_median(self):
        """stacker_outcome=fallback_median → confirmed_median."""
        record = {
            "stacker_outcome": "fallback_median",
            "type": "binary",
        }
        assert detect_stacker_fired(record) == "confirmed_median"

    def test_numeric_record_high_spread_stacker_marker(self):
        """Numeric record with stacker marker → confirmed_stacker."""
        record = {
            "type": "numeric",
            "comment_text": "text <!-- STACKED=true --> end",
        }
        assert detect_stacker_fired(record) == "confirmed_stacker"

    def test_custom_threshold(self):
        """Custom default_threshold for production-vs-median delta."""
        record = {
            "type": "binary",
            "our_prob_yes": 0.73,
            "per_model_forecasts": {
                "gpt-5.5": "60%",
                "claude-opus-4.7": "80%",
                "gemini-3.1-pro": "70%",
            },
        }
        # median = 0.70, spread = 0.80-0.60 = 0.20 > 0.15
        # |0.73 - 0.70| = 0.03
        # With default_threshold=0.05: 0.03 < 0.05, so likely_median (production matches)
        assert detect_stacker_fired(record, default_threshold=0.05) == "likely_median"
        # With default_threshold=0.02: 0.03 > 0.02, so likely_stacker
        assert detect_stacker_fired(record, default_threshold=0.02) == "likely_stacker"

    def test_historical_body_signature_confirmed(self):
        """Historical body signature (pre-marker era) → confirmed_stacker."""
        record = {
            "type": "binary",
            "comment_text": (
                "## R1: Forecaster 1 Reasoning\n"
                "Model: openrouter/anthropic/claude-opus-4.5\n"
                "## Stacker Meta-Analysis\n"
                "The models show disagreement on this question..."
            ),
        }
        assert detect_stacker_fired(record) == "confirmed_stacker"
