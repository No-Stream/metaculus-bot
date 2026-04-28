"""Performance analysis module for the Metaculus forecasting bot.

Collects resolved question data from the Metaculus API, scores predictions,
and provides reusable analysis functions.
"""

from metaculus_bot.performance_analysis.analysis import (
    binary_summary,
    disagreement_predicts_error,
    financial_vs_nonfinancial_pit,
    generate_report,
    mc_summary,
    no_bias_check,
    numeric_pit_analysis,
    per_model_binary_scores,
    stacking_effectiveness,
)
from metaculus_bot.performance_analysis.audit import (
    emit_combined_report,
    emit_external_comment_stub,
    emit_miss_markdown,
    emit_synthesis,
    load_combined_dataset,
    rank_our_models_by_accuracy,
    select_worst_misses,
)
from metaculus_bot.performance_analysis.collector import (
    build_performance_dataset,
    fetch_bot_comments,
    fetch_resolved_questions,
    load_dataset,
    save_dataset,
)
from metaculus_bot.performance_analysis.parsing import (
    annotate_forecaster_bullets_with_models,
    extract_model_display_name_from_reasoning,
    parse_forecaster_model_map,
    parse_per_model_forecasts,
    parse_per_model_numeric_percentiles,
    parse_per_model_reasoning_text,
    parse_resolution,
    parse_stacked_marker,
)
from metaculus_bot.performance_analysis.scoring import (
    binary_log_score,
    brier_score,
    mc_log_score,
    numeric_log_score,
)

__all__ = [
    "annotate_forecaster_bullets_with_models",
    "binary_log_score",
    "binary_summary",
    "brier_score",
    "build_performance_dataset",
    "disagreement_predicts_error",
    "emit_combined_report",
    "emit_external_comment_stub",
    "emit_miss_markdown",
    "emit_synthesis",
    "extract_model_display_name_from_reasoning",
    "fetch_bot_comments",
    "fetch_resolved_questions",
    "financial_vs_nonfinancial_pit",
    "generate_report",
    "load_combined_dataset",
    "load_dataset",
    "mc_log_score",
    "mc_summary",
    "no_bias_check",
    "numeric_log_score",
    "numeric_pit_analysis",
    "parse_forecaster_model_map",
    "parse_per_model_forecasts",
    "parse_per_model_numeric_percentiles",
    "parse_per_model_reasoning_text",
    "parse_resolution",
    "parse_stacked_marker",
    "per_model_binary_scores",
    "rank_our_models_by_accuracy",
    "save_dataset",
    "select_worst_misses",
    "stacking_effectiveness",
]
