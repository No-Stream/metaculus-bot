"""
Tests for custom probability clamping in prediction extraction.

Tests that probabilities are correctly clamped after LLM extraction:
- Binary questions: 2% to 98% (0.02 to 0.98) — see note below
- Multiple choice questions: 0.5% to 99.5% (0.005 to 0.995) with renormalization

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

    def test_mc_clamping_logic(self):
        """Test multiple choice prediction clamping logic."""
        # Test the exact clamping and renormalization logic used in main.py:401-409

        # Create options with extreme values
        options = [
            PredictedOption(option_name="Option A", probability=0.0001),  # Should be clamped to 0.005
            PredictedOption(option_name="Option B", probability=0.9999),  # Should be clamped to 0.995
        ]
        predicted_option_list = PredictedOptionList(predicted_options=options)

        # Apply custom clamping to 0.5%/99.5% for multiple choice questions
        for option in predicted_option_list.predicted_options:
            option.probability = max(0.005, min(0.995, option.probability))

        # Renormalize to ensure probabilities sum to 1 after clamping
        total_prob = sum(option.probability for option in predicted_option_list.predicted_options)
        if total_prob > 0:
            for option in predicted_option_list.predicted_options:
                option.probability /= total_prob

        # Check that values were clamped and renormalized
        option_a = next(opt for opt in predicted_option_list.predicted_options if opt.option_name == "Option A")
        option_b = next(opt for opt in predicted_option_list.predicted_options if opt.option_name == "Option B")

        # Values should be clamped to [0.005, 0.995]
        assert option_a.probability >= 0.005, f"Option A probability {option_a.probability} should be >= 0.005"
        assert option_b.probability <= 0.995, f"Option B probability {option_b.probability} should be <= 0.995"

        # Probabilities should sum to 1 after renormalization
        total_prob = sum(opt.probability for opt in predicted_option_list.predicted_options)
        assert abs(total_prob - 1.0) < 1e-10, f"Probabilities should sum to 1, got {total_prob}"

    def test_mc_clamping_with_renormalization(self):
        """Test multiple choice clamping with renormalization."""
        # Create options that will need renormalization after clamping
        options = [
            PredictedOption(option_name="Option A", probability=0.0001),  # Will become 0.005
            PredictedOption(option_name="Option B", probability=0.0001),  # Will become 0.005
            PredictedOption(option_name="Option C", probability=0.9998),  # Will become 0.995
        ]
        predicted_option_list = PredictedOptionList(predicted_options=options)

        # Apply custom clamping to 0.5%/99.5% for multiple choice questions
        for option in predicted_option_list.predicted_options:
            option.probability = max(0.005, min(0.995, option.probability))

        # Renormalize to ensure probabilities sum to 1 after clamping
        total_prob = sum(option.probability for option in predicted_option_list.predicted_options)
        if total_prob > 0:
            for option in predicted_option_list.predicted_options:
                option.probability /= total_prob

        # After renormalization, values may go slightly below the minimum bound
        # but should still be reasonable (close to minimum)
        for option in predicted_option_list.predicted_options:
            assert option.probability >= 0.001, (
                f"Option {option.option_name} probability {option.probability} too low after renormalization"
            )
            assert option.probability <= 1.0, (
                f"Option {option.option_name} probability {option.probability} too high after renormalization"
            )

        # Probabilities should sum to 1 after renormalization
        total_prob = sum(opt.probability for opt in predicted_option_list.predicted_options)
        assert abs(total_prob - 1.0) < 1e-10, f"Probabilities should sum to 1, got {total_prob}"

    def test_mc_clamping_preserves_normal_values(self):
        """Test multiple choice clamping preserves normal values."""
        # Create options with normal values (should be preserved)
        options = [
            PredictedOption(option_name="Option A", probability=0.3),
            PredictedOption(option_name="Option B", probability=0.7),
        ]
        predicted_option_list = PredictedOptionList(predicted_options=options)

        # Apply custom clamping to 0.5%/99.5% for multiple choice questions
        for option in predicted_option_list.predicted_options:
            option.probability = max(0.005, min(0.995, option.probability))

        # Renormalize to ensure probabilities sum to 1 after clamping
        total_prob = sum(option.probability for option in predicted_option_list.predicted_options)
        if total_prob > 0:
            for option in predicted_option_list.predicted_options:
                option.probability /= total_prob

        # Values should be approximately preserved (within small renormalization error)
        option_a = next(opt for opt in predicted_option_list.predicted_options if opt.option_name == "Option A")
        option_b = next(opt for opt in predicted_option_list.predicted_options if opt.option_name == "Option B")

        assert abs(option_a.probability - 0.3) < 0.01, f"Option A should be ~0.3, got {option_a.probability}"
        assert abs(option_b.probability - 0.7) < 0.01, f"Option B should be ~0.7, got {option_b.probability}"

        # Probabilities should sum to 1
        total_prob = sum(opt.probability for opt in predicted_option_list.predicted_options)
        assert abs(total_prob - 1.0) < 1e-10, f"Probabilities should sum to 1, got {total_prob}"

    def test_boundary_conditions(self):
        """Test clamping at exact boundary values."""
        # Binary boundaries — Atlas-inspired [0.02, 0.98], see module docstring.
        assert max(0.02, min(0.98, 0.02)) == 0.02
        assert max(0.02, min(0.98, 0.98)) == 0.98
        assert max(0.02, min(0.98, 0.019)) == 0.02
        assert max(0.02, min(0.98, 0.981)) == 0.98

        # MC boundaries (unchanged — Workstream B is binary-only).
        assert max(0.005, min(0.995, 0.005)) == 0.005
        assert max(0.005, min(0.995, 0.995)) == 0.995
        assert max(0.005, min(0.995, 0.004)) == 0.005
        assert max(0.005, min(0.995, 0.996)) == 0.995


if __name__ == "__main__":
    pytest.main([__file__])
