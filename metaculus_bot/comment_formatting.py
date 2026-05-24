"""Standalone helpers for formatting Metaculus bot comment text.

These are the logic-bearing implementations called by TemplateForecaster's thin
override methods. Extracted from main.py to keep that file focused on pipeline
orchestration.
"""

from __future__ import annotations

import logging
from typing import Literal, Sequence

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.comment_markers import (
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
from metaculus_bot.comment_trimming import trim_comment, trim_section
from metaculus_bot.performance_analysis.parsing import (
    annotate_forecaster_bullets_with_models,
    extract_model_display_name_from_reasoning,
)
from metaculus_bot.tool_runner import _feature_enabled as _tool_runner_feature_enabled

logger = logging.getLogger(__name__)


def format_research_summary_with_models(
    base_text: str,
    predictions: Sequence,
    report_number: int,
) -> str:
    """Inject model display names into summary bullets, then trim to section limit."""
    model_names_by_index: dict[int, str] = {}
    for j, forecast in enumerate(predictions):
        name = extract_model_display_name_from_reasoning(forecast.reasoning)
        if name is not None:
            model_names_by_index[j + 1] = name
    text = annotate_forecaster_bullets_with_models(base_text, model_names_by_index)
    return trim_section(text, f"report_{report_number}_summary")


def format_main_research_section(base_text: str, report_number: int) -> str:
    """Trim the main research section to the configured section limit."""
    return trim_section(base_text, f"report_{report_number}_research")


def format_forecaster_rationales_section(base_text: str, report_number: int) -> str:
    """Trim the forecaster rationales section to the configured section limit."""
    return trim_section(base_text, f"report_{report_number}_rationales")


def build_unified_explanation(
    base_text: str,
    question: object,
    aggregation_strategy: AggregationStrategy,
    stacker_outcome: str | None,
) -> str:
    """Build the final Metaculus comment with stacker/tools markers appended.

    For non-stacking strategies, just trims and returns. For STACKING /
    CONDITIONAL_STACKING, appends STACKER_OUTCOME, legacy STACKED, and
    TOOLS_USED markers.
    """
    if aggregation_strategy not in (AggregationStrategy.STACKING, AggregationStrategy.CONDITIONAL_STACKING):
        return trim_comment(base_text)

    assert stacker_outcome is not None, (
        "stacker_outcome must be provided for STACKING/CONDITIONAL_STACKING strategies; "
        "every reachable code path in _aggregate_predictions sets it. Missing entry = real bug."
    )

    match stacker_outcome:
        case "primary":
            outcome_marker, legacy_marker = STACKER_OUTCOME_PRIMARY, STACKED_MARKER_TRUE
        case "fallback_llm":
            outcome_marker, legacy_marker = STACKER_OUTCOME_FALLBACK_LLM, STACKED_MARKER_TRUE
        case "fallback_median":
            outcome_marker, legacy_marker = STACKER_OUTCOME_FALLBACK_MEDIAN, STACKED_MARKER_FALSE
        case "fallback_mean":
            outcome_marker, legacy_marker = STACKER_OUTCOME_FALLBACK_MEAN, STACKED_MARKER_FALSE
        case "skipped":
            outcome_marker, legacy_marker = STACKER_OUTCOME_SKIPPED, STACKED_MARKER_FALSE
        case other:
            raise ValueError(f"Unknown stacker outcome {other!r}")

    if isinstance(question, BinaryQuestion):
        qtype: Literal["binary", "numeric", "multiple_choice"] | None = "binary"
    elif isinstance(question, NumericQuestion):
        qtype = "numeric"
    elif isinstance(question, MultipleChoiceQuestion):
        qtype = "multiple_choice"
    else:
        qtype = None

    tools_marker = TOOLS_USED_MARKER_TRUE if _tool_runner_feature_enabled(qtype) else TOOLS_USED_MARKER_FALSE
    return trim_comment(f"{base_text}\n{outcome_marker}\n{legacy_marker}\n{tools_marker}\n")


__all__ = [
    "build_unified_explanation",
    "format_forecaster_rationales_section",
    "format_main_research_section",
    "format_research_summary_with_models",
]
