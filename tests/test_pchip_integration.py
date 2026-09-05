"""
Integration tests for PCHIP CDF with forecasting-tools NumericDistribution.

Tests that our CDF override approach works correctly with the framework.
"""

from itertools import pairwise

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid, generate_pchip_cdf, percentiles_to_pchip_format
from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution


def _build_question(
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 100.0,
    zero_point: float | None = None,
    cdf_size: int = 201,
) -> NumericQuestion:
    return NumericQuestion(
        id_of_question=1,
        id_of_post=1,
        page_url="https://example.com/q/1",
        question_text="Test numeric question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="units",
        zero_point=zero_point,
        cdf_size=cdf_size,
    )


class TestPchipIntegration:
    """Test integration between PCHIP CDF and NumericDistribution."""

    def test_cdf_override_format(self) -> None:
        """Test that our CDF override produces the expected format."""
        # Create test percentiles (our 8-percentile standard)
        percentiles = [
            Percentile(percentile=0.05, value=5.0),
            Percentile(percentile=0.10, value=10.0),
            Percentile(percentile=0.20, value=20.0),
            Percentile(percentile=0.40, value=40.0),
            Percentile(percentile=0.60, value=60.0),
            Percentile(percentile=0.80, value=80.0),
            Percentile(percentile=0.90, value=90.0),
            Percentile(percentile=0.95, value=95.0),
        ]

        question = _build_question()

        # Generate PCHIP CDF
        pchip_percentiles = percentiles_to_pchip_format(percentiles)
        pchip_cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=pchip_percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )

        prediction = create_pchip_numeric_distribution(pchip_cdf, percentiles, question, question.zero_point)
        pchip_percentile_objects = prediction.get_cdf()

        # Validate the format
        assert len(pchip_percentile_objects) == 201
        assert all(isinstance(p, Percentile) for p in pchip_percentile_objects)

        # Validate probability values are in [0,1]
        prob_values = [p.percentile for p in pchip_percentile_objects]
        assert all(0.0 <= p <= 1.0 for p in prob_values)

        # Validate question values are in bounds
        question_values = [p.value for p in pchip_percentile_objects]
        assert all(question.lower_bound <= v <= question.upper_bound for v in question_values)

        # Validate monotonicity (CDF requirements)
        assert all(a <= b for a, b in pairwise(prob_values))
        assert all(a <= b for a, b in pairwise(question_values))

    def test_cdf_override_uses_geometric_grid_for_zero_point(self) -> None:
        """The production wrapper must pair PCHIP heights with the canonical value grid."""
        percentiles = [
            Percentile(percentile=0.05, value=10.0),
            Percentile(percentile=0.50, value=100.0),
            Percentile(percentile=0.95, value=900.0),
        ]
        question = _build_question(lower_bound=1.0, upper_bound=1000.0, zero_point=0.0)

        pchip_cdf, _ = generate_pchip_cdf(
            percentile_values=percentiles_to_pchip_format(percentiles),
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )
        prediction = create_pchip_numeric_distribution(pchip_cdf, percentiles, question, question.zero_point)

        actual_values = [p.value for p in prediction.get_cdf()]
        expected_values = build_cdf_value_grid(
            question.lower_bound, question.upper_bound, question.zero_point, len(pchip_cdf)
        )

        np.testing.assert_allclose(actual_values, expected_values)
        assert actual_values[1] == pytest.approx(1.035142, abs=1e-6)

    def test_spacing_assertion_compliance(self) -> None:
        """Test that our PCHIP CDF satisfies the 5e-5 spacing requirement."""
        percentiles = [
            Percentile(percentile=0.05, value=10.0),
            Percentile(percentile=0.50, value=50.0),
            Percentile(percentile=0.95, value=90.0),
        ]

        question = _build_question()

        # Generate PCHIP CDF
        pchip_percentiles = percentiles_to_pchip_format(percentiles)
        pchip_cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=pchip_percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )

        # Check that spacing assertion would pass
        for i in range(len(pchip_cdf) - 1):
            spacing = abs(pchip_cdf[i + 1] - pchip_cdf[i])
            assert spacing >= 5e-05, f"Spacing violation at index {i}: {spacing}"

    def test_pchip_subclass_approach_works(self) -> None:
        """Test that our PchipNumericDistribution subclass approach works."""
        percentiles = [
            Percentile(percentile=0.05, value=5.0),
            Percentile(percentile=0.50, value=50.0),
            Percentile(percentile=0.95, value=95.0),
        ]

        # Generate PCHIP CDF (simulating our main.py logic)
        question = _build_question()

        pchip_percentiles = percentiles_to_pchip_format(percentiles)
        pchip_cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=pchip_percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )

        prediction = create_pchip_numeric_distribution(pchip_cdf, percentiles, question, question.zero_point)

        # Test that the subclass works
        result_cdf = prediction.cdf
        assert len(result_cdf) == 201
        assert all(isinstance(p, Percentile) for p in result_cdf)

        # Test that spacing assertion would pass
        for i in range(len(result_cdf) - 1):
            spacing = abs(result_cdf[i + 1].percentile - result_cdf[i].percentile)
            assert spacing >= 5e-05, f"Spacing violation at index {i}: {spacing}"

    def test_problematic_distribution_case(self) -> None:
        """Test with a distribution similar to the one that was failing."""
        # Create a distribution that would likely fail the original spacing check
        percentiles = [
            Percentile(percentile=0.05, value=70.0),
            Percentile(percentile=0.10, value=70.5),  # Very close values
            Percentile(percentile=0.20, value=71.0),
            Percentile(percentile=0.40, value=72.0),
            Percentile(percentile=0.60, value=73.0),
            Percentile(percentile=0.80, value=74.0),
            Percentile(percentile=0.90, value=74.5),  # Close values again
            Percentile(percentile=0.95, value=75.0),
        ]

        question = _build_question()

        # This should work with PCHIP even though values are close
        pchip_percentiles = percentiles_to_pchip_format(percentiles)
        pchip_cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=pchip_percentiles,
            open_upper_bound=question.open_upper_bound,
            open_lower_bound=question.open_lower_bound,
            upper_bound=question.upper_bound,
            lower_bound=question.lower_bound,
            zero_point=question.zero_point,
        )

        # Verify no spacing violations
        for i in range(len(pchip_cdf) - 1):
            spacing = abs(pchip_cdf[i + 1] - pchip_cdf[i])
            assert spacing >= 5e-05, f"PCHIP failed to fix spacing at index {i}: {spacing}"


if __name__ == "__main__":
    pytest.main([__file__])
