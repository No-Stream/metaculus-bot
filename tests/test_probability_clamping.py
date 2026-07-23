"""
Tests for custom probability clamping in prediction extraction.

Tests that probabilities are correctly clamped after LLM extraction:
- Binary questions: 2% to 98% (0.02 to 0.98) — see note below
- Multiple choice questions: 1% to 99% (0.01 to 0.99) with renormalization, aligned to
  ft 0.2.92's PredictedOptionList validator (see the MC clamp constants)

Binary bounds were widened from [0.01, 0.99] to [0.02, 0.98] on 2026-05-12
following Preseen-Atlas (spring-AIB-2026 leader), whose comments publish
`submitted = 0.96 * model_estimate + 0.02` on every forecast. We adopted the
clip-only portion (tail protection from log-loss blowup on misses) without
the linear shrink. See:
scratch_docs_and_planning/atlas_inspired_improvements.md (Workstream B).
"""

import pytest
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOption,
    PredictedOptionList,
)

from metaculus_bot.constants import MC_PROB_MAX, MC_PROB_MIN
from metaculus_bot.numeric.utils import clamp_and_renormalize_mc


class TestProbabilityClamping:
    """Test custom probability clamping logic directly."""

    def test_binary_clamping_logic(self):
        """Test binary prediction clamping logic at the Atlas-inspired [0.02, 0.98]."""
        # Mirror the clamp in main.py:_run_forecast_on_binary and stacking.py:run_stacking_binary.

        # Test extreme low value (should be clamped to 0.02)
        raw_prediction = 0.0001
        clamped = max(0.02, min(0.98, raw_prediction))
        assert clamped == 0.02, f"Expected 0.02, got {clamped}"

        # Test extreme high value (should be clamped to 0.98)
        raw_prediction = 0.9999
        clamped = max(0.02, min(0.98, raw_prediction))
        assert clamped == 0.98, f"Expected 0.98, got {clamped}"

        # Test normal value (should be preserved)
        raw_prediction = 0.65
        clamped = max(0.02, min(0.98, raw_prediction))
        assert clamped == 0.65, f"Expected 0.65, got {clamped}"

        # Test boundary values
        raw_prediction = 0.02
        clamped = max(0.02, min(0.98, raw_prediction))
        assert clamped == 0.02, f"Expected 0.02, got {clamped}"

        raw_prediction = 0.98
        clamped = max(0.02, min(0.98, raw_prediction))
        assert clamped == 0.98, f"Expected 0.98, got {clamped}"

    def test_mc_clamping_clamps_extremes_to_bounds(self):
        """clamp_and_renormalize_mc pins extreme options into [MC_PROB_MIN, MC_PROB_MAX] and renormalizes."""
        pol = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Option A", probability=0.0001),  # -> floored to MC_PROB_MIN
                PredictedOption(option_name="Option B", probability=0.9999),  # -> ceilinged to MC_PROB_MAX
            ]
        )
        clamp_and_renormalize_mc(pol)

        for option in pol.predicted_options:
            assert MC_PROB_MIN <= option.probability <= MC_PROB_MAX, (
                f"{option.option_name} probability {option.probability} outside [{MC_PROB_MIN}, {MC_PROB_MAX}]"
            )
        total_prob = sum(o.probability for o in pol.predicted_options)
        assert abs(total_prob - 1.0) < 1e-10, f"Probabilities should sum to 1, got {total_prob}"

    def test_mc_clamping_holds_floor_under_renormalization(self):
        """Regression for renormalize-below-floor drift.

        With a dominant option plus near-floor siblings, the old clamp-then-divide
        pushed the floored siblings back UNDER MC_PROB_MIN (dividing by a >1 total).
        clamp_and_renormalize_mc now guarantees EVERY option stays within
        [MC_PROB_MIN, MC_PROB_MAX] after renormalization.
        """
        pol = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Option A", probability=0.0001),  # near-floor sibling
                PredictedOption(option_name="Option B", probability=0.0001),  # near-floor sibling
                PredictedOption(option_name="Option C", probability=0.9998),  # dominant
            ]
        )
        clamp_and_renormalize_mc(pol)

        for option in pol.predicted_options:
            assert option.probability >= MC_PROB_MIN, (
                f"{option.option_name} probability {option.probability} drifted below MC_PROB_MIN ({MC_PROB_MIN})"
            )
            assert option.probability <= MC_PROB_MAX
        total_prob = sum(o.probability for o in pol.predicted_options)
        assert abs(total_prob - 1.0) < 1e-10, f"Probabilities should sum to 1, got {total_prob}"

    def test_mc_clamping_preserves_normal_values(self):
        """Comfortably in-bounds options are approximately preserved (only renormalization epsilon)."""
        pol = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Option A", probability=0.3),
                PredictedOption(option_name="Option B", probability=0.7),
            ]
        )
        clamp_and_renormalize_mc(pol)

        option_a = next(o for o in pol.predicted_options if o.option_name == "Option A")
        option_b = next(o for o in pol.predicted_options if o.option_name == "Option B")
        assert option_a.probability == pytest.approx(0.3, abs=1e-9), (
            f"Option A should be ~0.3, got {option_a.probability}"
        )
        assert option_b.probability == pytest.approx(0.7, abs=1e-9), (
            f"Option B should be ~0.7, got {option_b.probability}"
        )

        total_prob = sum(o.probability for o in pol.predicted_options)
        assert abs(total_prob - 1.0) < 1e-10, f"Probabilities should sum to 1, got {total_prob}"

    def test_boundary_conditions(self):
        """Test clamping at exact boundary values."""
        # Binary boundaries — Atlas-inspired [0.02, 0.98], see module docstring.
        assert max(0.02, min(0.98, 0.02)) == 0.02
        assert max(0.02, min(0.98, 0.98)) == 0.98
        assert max(0.02, min(0.98, 0.019)) == 0.02
        assert max(0.02, min(0.98, 0.981)) == 0.98

        # MC boundaries — aligned to ft 0.2.92's [0.01, 0.99]; MC_PROB_MIN/MAX are the single source.
        assert max(MC_PROB_MIN, min(MC_PROB_MAX, MC_PROB_MIN)) == MC_PROB_MIN
        assert max(MC_PROB_MIN, min(MC_PROB_MAX, MC_PROB_MAX)) == MC_PROB_MAX
        assert max(MC_PROB_MIN, min(MC_PROB_MAX, 0.004)) == MC_PROB_MIN
        assert max(MC_PROB_MIN, min(MC_PROB_MAX, 0.996)) == MC_PROB_MAX


if __name__ == "__main__":
    pytest.main([__file__])
