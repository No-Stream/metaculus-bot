"""Tests for metaculus_bot.comment.formatting — standalone helper functions
that produce the formatted comment text.

These tests exercise the helpers directly (no class hierarchy patching needed).
The existing test_main_comment_output.py tests continue to exercise the full
integration through TemplateForecaster's thin wrapper methods.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.comment.markers import (
    STACKED_MARKER_FALSE,
    STACKED_MARKER_TRUE,
    STACKER_OUTCOME_FALLBACK_LLM,
    STACKER_OUTCOME_FALLBACK_MEAN,
    STACKER_OUTCOME_FALLBACK_MEDIAN,
    STACKER_OUTCOME_PRIMARY,
    STACKER_OUTCOME_SKIPPED,
    TOOLS_USED_MARKER_FALSE,
    TOOLS_USED_MARKER_TRUE,
)

# ---------------------------------------------------------------------------
# format_research_summary_with_models
# ---------------------------------------------------------------------------


class TestFormatResearchSummaryWithModels:
    """Test the standalone helper that injects model names and trims."""

    def test_injects_model_names_into_bullets(self):
        from metaculus_bot.comment.formatting import format_research_summary_with_models

        base_text = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1*: 72.0%\n"
            "*Forecaster 2*: 68.0%\n"
            "### Research Summary\nSome research.\n"
        )
        predictions = [
            MagicMock(reasoning="Model: openrouter/openai/gpt-5.5\n\nanalysis"),
            MagicMock(reasoning="Model: openrouter/anthropic/claude-opus-4.7\n\nanalysis"),
        ]
        result = format_research_summary_with_models(base_text, predictions, report_number=1)
        assert "*Forecaster 1 (gpt-5.5)*: 72.0%" in result
        assert "*Forecaster 2 (claude-opus-4.7)*: 68.0%" in result

    def test_missing_model_prefix_leaves_bullet_unannotated(self):
        from metaculus_bot.comment.formatting import format_research_summary_with_models

        base_text = "## Report 1 Summary\n### Forecasts\n*Forecaster 1*: 72.0%\n### Research Summary\nSome research.\n"
        predictions = [MagicMock(reasoning="just analysis, no model prefix")]
        result = format_research_summary_with_models(base_text, predictions, report_number=1)
        assert "*Forecaster 1*: 72.0%" in result
        assert "(" not in result.split("### Forecasts")[1].split("### Research Summary")[0]

    def test_trims_oversized_text(self):
        from metaculus_bot.comment.formatting import format_research_summary_with_models
        from metaculus_bot.constants import REPORT_SECTION_CHAR_LIMIT

        huge_text = "## Report 1 Summary\n" + ("X" * (REPORT_SECTION_CHAR_LIMIT + 5000))
        predictions: list = []
        result = format_research_summary_with_models(huge_text, predictions, report_number=1)
        assert len(result) <= REPORT_SECTION_CHAR_LIMIT


# ---------------------------------------------------------------------------
# format_main_research_section
# ---------------------------------------------------------------------------


class TestFormatMainResearchSection:
    def test_trims_oversized_text(self):
        from metaculus_bot.comment.formatting import format_main_research_section
        from metaculus_bot.constants import REPORT_SECTION_CHAR_LIMIT

        huge_text = "## Research\n" + ("Y" * (REPORT_SECTION_CHAR_LIMIT + 3000))
        result = format_main_research_section(huge_text, report_number=1)
        assert len(result) <= REPORT_SECTION_CHAR_LIMIT

    def test_short_text_passes_through(self):
        from metaculus_bot.comment.formatting import format_main_research_section

        short_text = "## Research\nSome content."
        result = format_main_research_section(short_text, report_number=1)
        assert result == short_text


# ---------------------------------------------------------------------------
# format_forecaster_rationales_section
# ---------------------------------------------------------------------------


class TestFormatForecasterRationalesSection:
    def test_trims_oversized_text(self):
        from metaculus_bot.comment.formatting import format_forecaster_rationales_section
        from metaculus_bot.constants import REPORT_SECTION_CHAR_LIMIT

        huge_text = "## Rationale\n" + ("Z" * (REPORT_SECTION_CHAR_LIMIT + 2000))
        result = format_forecaster_rationales_section(huge_text, report_number=1)
        assert len(result) <= REPORT_SECTION_CHAR_LIMIT

    def test_short_text_passes_through(self):
        from metaculus_bot.comment.formatting import format_forecaster_rationales_section

        short_text = "## Rationale\nSome reasoning."
        result = format_forecaster_rationales_section(short_text, report_number=1)
        assert result == short_text


# ---------------------------------------------------------------------------
# build_unified_explanation
# ---------------------------------------------------------------------------


class TestBuildUnifiedExplanation:
    """Test the standalone build_unified_explanation helper."""

    def _make_question(self, cls=BinaryQuestion, qid: int = 12345) -> MagicMock:
        q = MagicMock(spec=cls)
        q.id_of_question = qid
        return q

    def test_non_stacking_strategy_just_trims(self):
        from metaculus_bot.comment.formatting import build_unified_explanation

        base_text = "# SUMMARY\n\nBody text."
        result = build_unified_explanation(
            base_text=base_text,
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.MEAN,
            stacker_outcome=None,
        )
        assert "STACKED=" not in result
        assert "STACKER_OUTCOME=" not in result
        assert "Body text." in result

    def test_median_strategy_just_trims(self):
        from metaculus_bot.comment.formatting import build_unified_explanation

        base_text = "# SUMMARY\nContent."
        result = build_unified_explanation(
            base_text=base_text,
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.MEDIAN,
            stacker_outcome=None,
        )
        assert "STACKED=" not in result

    def test_stacking_primary_emits_correct_markers(self):
        from metaculus_bot.comment.formatting import build_unified_explanation

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.STACKING,
            stacker_outcome="primary",
        )
        assert STACKER_OUTCOME_PRIMARY in result
        assert STACKED_MARKER_TRUE in result
        assert STACKED_MARKER_FALSE not in result

    def test_stacking_fallback_llm_emits_correct_markers(self):
        from metaculus_bot.comment.formatting import build_unified_explanation

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.STACKING,
            stacker_outcome="fallback_llm",
        )
        assert STACKER_OUTCOME_FALLBACK_LLM in result
        assert STACKED_MARKER_TRUE in result

    def test_stacking_fallback_median_emits_correct_markers(self):
        from metaculus_bot.comment.formatting import build_unified_explanation

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_outcome="fallback_median",
        )
        assert STACKER_OUTCOME_FALLBACK_MEDIAN in result
        assert STACKED_MARKER_FALSE in result
        assert STACKED_MARKER_TRUE not in result

    def test_stacking_fallback_mean_emits_correct_markers(self):
        from metaculus_bot.comment.formatting import build_unified_explanation

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.STACKING,
            stacker_outcome="fallback_mean",
        )
        assert STACKER_OUTCOME_FALLBACK_MEAN in result
        assert STACKED_MARKER_FALSE in result
        assert STACKED_MARKER_TRUE not in result

    def test_stacking_skipped_emits_correct_markers(self):
        from metaculus_bot.comment.formatting import build_unified_explanation

        result = build_unified_explanation(
            base_text="# SUMMARY\nBody.",
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_outcome="skipped",
        )
        assert STACKER_OUTCOME_SKIPPED in result
        assert STACKED_MARKER_FALSE in result
        assert STACKED_MARKER_TRUE not in result

    def test_unknown_outcome_raises_valueerror(self):
        from metaculus_bot.comment.formatting import build_unified_explanation

        with pytest.raises(ValueError, match="Unknown stacker outcome"):
            build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(),
                aggregation_strategy=AggregationStrategy.STACKING,
                stacker_outcome="bogus_value",
            )

    def test_stacking_with_none_outcome_raises_assertion(self):
        from metaculus_bot.comment.formatting import build_unified_explanation

        with pytest.raises(AssertionError, match="stacker_outcome must be provided"):
            build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(),
                aggregation_strategy=AggregationStrategy.STACKING,
                stacker_outcome=None,
            )

    def test_tools_used_marker_emitted_for_binary_when_enabled(self):
        from metaculus_bot.comment.formatting import build_unified_explanation

        with patch("metaculus_bot.comment.formatting._tool_runner_feature_enabled", return_value=True):
            result = build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(cls=BinaryQuestion),
                aggregation_strategy=AggregationStrategy.STACKING,
                stacker_outcome="primary",
            )
        assert TOOLS_USED_MARKER_TRUE in result

    def test_tools_used_marker_false_when_disabled(self):
        from metaculus_bot.comment.formatting import build_unified_explanation

        with patch("metaculus_bot.comment.formatting._tool_runner_feature_enabled", return_value=False):
            result = build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(cls=BinaryQuestion),
                aggregation_strategy=AggregationStrategy.STACKING,
                stacker_outcome="primary",
            )
        assert TOOLS_USED_MARKER_FALSE in result

    def test_numeric_question_type_passed_to_feature_enabled(self):
        from metaculus_bot.comment.formatting import build_unified_explanation

        with patch("metaculus_bot.comment.formatting._tool_runner_feature_enabled", return_value=False) as mock_fe:
            build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(cls=NumericQuestion),
                aggregation_strategy=AggregationStrategy.STACKING,
                stacker_outcome="primary",
            )
        mock_fe.assert_called_once_with("numeric")

    def test_mc_question_type_passed_to_feature_enabled(self):
        from metaculus_bot.comment.formatting import build_unified_explanation

        with patch("metaculus_bot.comment.formatting._tool_runner_feature_enabled", return_value=False) as mock_fe:
            build_unified_explanation(
                base_text="# SUMMARY\nBody.",
                question=self._make_question(cls=MultipleChoiceQuestion),
                aggregation_strategy=AggregationStrategy.STACKING,
                stacker_outcome="primary",
            )
        mock_fe.assert_called_once_with("multiple_choice")

    def test_trims_oversized_comment(self):
        from metaculus_bot.comment.formatting import build_unified_explanation
        from metaculus_bot.constants import COMMENT_CHAR_LIMIT

        huge_base = "# SUMMARY\n" + ("X" * (COMMENT_CHAR_LIMIT + 5000))
        result = build_unified_explanation(
            base_text=huge_base,
            question=self._make_question(),
            aggregation_strategy=AggregationStrategy.STACKING,
            stacker_outcome="primary",
        )
        assert len(result) <= COMMENT_CHAR_LIMIT
        assert STACKER_OUTCOME_PRIMARY in result
