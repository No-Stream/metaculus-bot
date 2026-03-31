"""Performance analysis module for the Metaculus forecasting bot.

Collects resolved question data from the Metaculus API, scores predictions,
and provides reusable analysis functions.
"""

from metaculus_bot.performance_analysis.analysis import (
    binary_summary,
    generate_report,
    mc_summary,
    numeric_pit_analysis,
    per_model_binary_scores,
)
from metaculus_bot.performance_analysis.collector import (
    build_performance_dataset,
    fetch_bot_comments,
    fetch_resolved_questions,
    load_dataset,
    save_dataset,
)
from metaculus_bot.performance_analysis.parsing import (
    MODEL_NAMES,
    parse_per_model_forecasts,
    parse_resolution,
)
from metaculus_bot.performance_analysis.scoring import (
    binary_log_score,
    brier_score,
    mc_log_score,
    numeric_log_score,
)

__all__ = [
    "MODEL_NAMES",
    "binary_log_score",
    "binary_summary",
    "brier_score",
    "build_performance_dataset",
    "fetch_bot_comments",
    "fetch_resolved_questions",
    "generate_report",
    "load_dataset",
    "mc_log_score",
    "mc_summary",
    "numeric_log_score",
    "numeric_pit_analysis",
    "parse_per_model_forecasts",
    "parse_resolution",
    "per_model_binary_scores",
    "save_dataset",
]
