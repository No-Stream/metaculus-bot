"""
Tests for PCHIP CDF validation (QA checks that replace forecasting-tools validation).
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from forecasting_tools import GeneralLlm, NumericQuestion
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.constants import NUM_MAX_STEP, NUM_MIN_PROB_STEP
from metaculus_bot.numeric.config import PCHIP_CDF_POINTS
from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
from metaculus_bot.numeric.pchip_cdf import generate_pchip_cdf
from metaculus_bot.value_extraction import ExtractionOutcome


def _stub_open_time() -> datetime:
    return datetime.now() - timedelta(days=30)


def _stub_resolve_time() -> datetime:
    return datetime.now() + timedelta(days=365)


class TestPchipValidation:
    """Test our custom PCHIP CDF validation logic."""

    def create_mock_template_forecaster(self):
        """Create a mock TemplateForecaster for testing validation."""
        # Import here to avoid circular imports in tests
        from main import TemplateForecaster

        # The "default" forecaster LLM's .invoke must be an AsyncMock (awaitable):
        # run_numeric_forecast wraps it in invoke_with_broad_retry, which awaits the
        # result. A bare MagicMock().invoke() returns a non-awaitable, which raises a
        # (broadly-retryable) TypeError and incurs the full ~41s retry backoff. These
        # tests exercise CDF validation, not the LLM call, so a trivial async stub is
        # the right fixture; structure_output is patched per-test for the parse step.
        default_llm = MagicMock()
        default_llm.invoke = AsyncMock(return_value="reasoning text with percentiles")
        mock_llms = {
            "default": default_llm,
            "parser": MagicMock(),
            "researcher": MagicMock(),
            "summarizer": MagicMock(),
        }
        return TemplateForecaster(llms=cast(dict[str, str | GeneralLlm], mock_llms), publish_reports_to_metaculus=False)

    def create_dummy_llm(self):
        """Create a dummy LLM for testing."""

        class DummyLLM:
            def __init__(self, reasoning: str):
                self._reasoning = reasoning
                self.model = "dummy-test-model"

            async def invoke(self, prompt: str):
                return self._reasoning

        return DummyLLM(
            "Test reasoning with percentiles:\nPercentile 5: 5.0\nPercentile 10: 10.0\nPercentile 20: 20.0\nPercentile 40: 40.0\nPercentile 60: 60.0\nPercentile 80: 80.0\nPercentile 90: 90.0\nPercentile 95: 95.0"
        )

    def create_mock_question(self, open_upper=False, open_lower=False, upper=100.0, lower=0.0):
        """Create a mock question for testing."""
        return SimpleNamespace(
            open_upper_bound=open_upper,
            open_lower_bound=open_lower,
            upper_bound=upper,
            lower_bound=lower,
            zero_point=None,
            id_of_question=123,
            question_text="Test numeric question",
            background_info="Test background",
            resolution_criteria="Test resolution criteria",
            fine_print="Test fine print",
            unit_of_measure="units",
            page_url="https://example.com/question/123",
            open_time=_stub_open_time(),
            scheduled_resolution_time=_stub_resolve_time(),
        )

    @pytest.mark.asyncio
    @patch("metaculus_bot.numeric.pchip_cdf.generate_pchip_cdf")
    @patch("metaculus_bot.numeric.pchip_cdf.percentiles_to_pchip_format")
    async def test_valid_pchip_cdf_passes_validation(self, mock_format, mock_generate):
        """Test that a valid PCHIP CDF passes all validation checks."""
        forecaster = self.create_mock_template_forecaster()
        question = self.create_mock_question(open_upper=False, open_lower=False)

        # Create valid CDF (201 points, monotonic, proper spacing)
        valid_cdf = np.linspace(0.0, 1.0, 201).tolist()
        mock_generate.return_value = (valid_cdf, False)
        mock_format.return_value = {}

        percentiles = [
            Percentile(percentile=0.01, value=1.0),
            Percentile(percentile=0.025, value=3.0),
            Percentile(percentile=0.05, value=5.0),
            Percentile(percentile=0.10, value=10.0),
            Percentile(percentile=0.20, value=20.0),
            Percentile(percentile=0.40, value=40.0),
            Percentile(percentile=0.50, value=50.0),
            Percentile(percentile=0.60, value=60.0),
            Percentile(percentile=0.80, value=80.0),
            Percentile(percentile=0.90, value=90.0),
            Percentile(percentile=0.95, value=95.0),
            Percentile(percentile=0.975, value=97.0),
            Percentile(percentile=0.99, value=99.0),
        ]

        # Should not raise any exceptions. parse_structured still serves the C3
        # outcome_type read; percentiles now come from the extraction ladder.
        llm = self.create_dummy_llm()
        with (
            patch(
                "metaculus_bot.forecaster_runners.parse_structured",
                return_value=OutcomeTypeResult(is_discrete_integer=False),
            ),
            patch(
                "metaculus_bot.forecaster_runners.extract_numeric",
                new=AsyncMock(return_value=ExtractionOutcome(value=percentiles, rung="block", block_present=True)),
            ),
        ):
            result = await forecaster._run_forecast_on_numeric(
                cast(NumericQuestion, question), "test research", cast(GeneralLlm, llm)
            )
            assert result is not None

    # ---------------------------------------------------------------------
    # Constraint enforcement, against the REAL generator.
    #
    # Seven tests used to live here, one per constraint (wrong length, probs
    # outside [0,1], non-monotonic, min-step, max-step, closed-bound, open-bound).
    # Each mocked `generate_pchip_cdf` to return a hand-built invalid CDF and
    # asserted `pytest.raises(Exception)`. All seven were vacuous: they patched
    # `forecaster_runners.parse_structured` to return a percentile LIST, but at
    # that point in `run_numeric_forecast` that call serves the C3 outcome_type
    # read, so production did `outcome_result.is_discrete_integer` on a list and
    # died with `AttributeError` BEFORE any CDF work. Probed directly:
    # `mock_generate.call_count == 0` — the invalid CDFs never reached anything,
    # and the bare `raises(Exception)` swallowed the AttributeError. Introduced
    # accidentally by bdbd452 (2026-02-19), which inserted that production call
    # and fixed every test it visibly broke; these were invisible because
    # `raises(Exception)` kept them green.
    #
    # Wiring the mocks correctly does NOT rescue them: `build_numeric_distribution`
    # (numeric/pipeline.py) catches a failing CDF build and substitutes
    # `create_fallback_numeric_distribution`, and `validate_cdf_construction`
    # deliberately skips PCHIP distributions — so a forced-invalid CDF produces a
    # fallback distribution, not an exception. The premise "an invalid CDF raises
    # out of _run_forecast_on_numeric" was never true on this path.
    #
    # So the constraint claim is asserted where it is actually decidable: on the
    # real generator's OUTPUT. This is the property the seven tests were reaching
    # for — the pipeline cannot emit a CDF that violates Metaculus's submission
    # rules — and it fails loudly if any enforcement tier regresses.
    # ---------------------------------------------------------------------

    @pytest.mark.parametrize("open_bounds", [False, True], ids=["closed_bounds", "open_bounds"])
    @pytest.mark.parametrize(
        "shape,percentile_values",
        [
            ("spread", {1.0: 1.0, 25.0: 25.0, 50.0: 50.0, 75.0: 75.0, 99.0: 99.0}),
            ("concentrated", {1.0: 49.0, 25.0: 49.8, 50.0: 50.0, 75.0: 50.2, 99.0: 51.0}),
            ("skewed_low", {1.0: 0.5, 25.0: 1.0, 50.0: 2.0, 75.0: 10.0, 99.0: 95.0}),
            ("all_duplicate_values", {1.0: 50.0, 25.0: 50.0, 50.0: 50.0, 75.0: 50.0, 99.0: 50.1}),
        ],
    )
    def test_generated_cdf_satisfies_every_submission_constraint(
        self, shape: str, percentile_values: dict[float, float], open_bounds: bool
    ) -> None:
        """The real generator's output satisfies all six server-side CDF rules.

        Input shapes span the cases that stress different enforcement tiers: a
        well-spread distribution, one concentrated in a hair-thin band (stresses
        min-step), a heavily skewed one (stresses max-step), and one whose
        declared values are all identical (the degenerate case). ``shape`` is
        unused in the body — it names the case in the test id so a failure says
        which shape broke.
        """
        del shape  # named for the parametrize id only

        cdf, _aggressive_enforcement = generate_pchip_cdf(
            percentile_values=percentile_values,
            open_upper_bound=open_bounds,
            open_lower_bound=open_bounds,
            upper_bound=100.0,
            lower_bound=0.0,
            zero_point=None,
            min_step=NUM_MIN_PROB_STEP,
        )
        steps = np.diff(cdf)

        assert len(cdf) == PCHIP_CDF_POINTS
        assert all(0.0 <= v <= 1.0 for v in cdf), f"probability outside [0,1]: {min(cdf)}..{max(cdf)}"
        # Strict, not >=: the min-step rule below already forbids flat segments,
        # and Metaculus rejects a CDF with any zero-width bin.
        assert bool(np.all(steps > 0.0)), f"non-monotonic: min step {np.min(steps)}"
        assert float(np.min(steps)) >= NUM_MIN_PROB_STEP - 1e-12, f"min-step violated: {np.min(steps)}"
        assert float(np.max(steps)) <= NUM_MAX_STEP + 1e-12, f"max-step violated: {np.max(steps)}"

        if open_bounds:
            assert cdf[0] >= 0.001, f"open lower bound needs >= 0.001 mass, got {cdf[0]}"
            assert cdf[-1] <= 0.999, f"open upper bound caps at 0.999, got {cdf[-1]}"
        else:
            assert cdf[0] == 0.0
            assert cdf[-1] == 1.0


if __name__ == "__main__":
    pytest.main([__file__])
