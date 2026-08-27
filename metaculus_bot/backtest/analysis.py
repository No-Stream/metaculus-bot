"""Aggregate scores, compare bot vs community, generate reports."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from metaculus_bot.backtest.scoring import GroundTruth, QuestionScore

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    bot_name: str
    scores: list[QuestionScore]
    num_questions: int
    num_scored: int
    num_failed: int


def aggregate_scores(scores: list[QuestionScore]) -> dict[str, dict]:
    """Group scores by metric_name and compute summary statistics for each.

    Returns dict mapping metric_name -> {bot_mean, bot_std, community_mean, bot_minus_community, n}.
    """
    if not scores:
        return {}

    by_metric: dict[str, list[QuestionScore]] = {}
    for s in scores:
        by_metric.setdefault(s.metric_name, []).append(s)

    aggregated: dict[str, dict] = {}
    for metric_name, metric_scores in by_metric.items():
        bot_values = np.array([s.bot_score for s in metric_scores], dtype=float)
        community_values = [s.community_score for s in metric_scores if s.community_score is not None]

        bot_mean = float(np.mean(bot_values))
        bot_std = float(np.std(bot_values))

        if community_values:
            community_mean = float(np.mean(community_values))
            bot_minus_community = bot_mean - community_mean
        else:
            community_mean = None
            bot_minus_community = None

        aggregated[metric_name] = {
            "bot_mean": bot_mean,
            "bot_std": bot_std,
            "community_mean": community_mean,
            "bot_minus_community": bot_minus_community,
            "n": len(metric_scores),
        }

    return aggregated


_METRIC_TABLE_HEADER: tuple[str, str] = (
    "| Metric | Bot Mean | Bot Std | Community Mean | Bot - Community | N |",
    "|--------|----------|---------|----------------|-----------------|---|",
)


def _metric_table_lines(aggregated: dict[str, dict]) -> list[str]:
    """Markdown table of per-metric bot/community stats; empty list when nothing aggregated.

    Community columns render "N/A" rather than a number when Metaculus published no
    comparable community score for that metric (numeric log score, notably).
    """
    if not aggregated:
        return []
    lines = list(_METRIC_TABLE_HEADER)
    for metric_name, stats in aggregated.items():
        community_mean_str = f"{stats['community_mean']:.4f}" if stats["community_mean"] is not None else "N/A"
        bot_minus_str = f"{stats['bot_minus_community']:.4f}" if stats["bot_minus_community"] is not None else "N/A"
        lines.append(
            f"| {metric_name} "
            f"| {stats['bot_mean']:.4f} "
            f"| {stats['bot_std']:.4f} "
            f"| {community_mean_str} "
            f"| {bot_minus_str} "
            f"| {stats['n']} |"
        )
    return lines


def _question_set_summary_lines(question_set: Any) -> list[str]:
    """Question count + type distribution bullets, when the question set carries metadata."""
    if question_set is None or not hasattr(question_set, "fetch_metadata"):
        return []
    metadata = question_set.fetch_metadata
    lines = [f"- Questions: {metadata.get('total_clean', '?')}"]
    type_dist = metadata.get("type_distribution")
    if type_dist:
        dist_parts = [f"{qtype}: {count}" for qtype, count in type_dist.items()]
        lines.append(f"- Type distribution: {', '.join(dist_parts)}")
    return lines


def _per_type_breakdown_lines(results: list[BacktestResult]) -> list[str]:
    """One metric table per (bot, question type), question types in sorted order."""
    lines: list[str] = []
    for result in results:
        by_type: dict[str, list[QuestionScore]] = {}
        for score in result.scores:
            by_type.setdefault(score.question_type, []).append(score)

        for question_type, type_scores in sorted(by_type.items()):
            lines.append("")
            lines.append(f"### {result.bot_name} - {question_type}")
            lines.extend(_metric_table_lines(aggregate_scores(type_scores)))
    return lines


def generate_backtest_report(
    results: list[BacktestResult],
    question_set: Any = None,
    output_path: str | None = None,
) -> str:
    """Generate a markdown backtest report string.

    If output_path is provided, writes the report to that file (creating parent dirs).
    Returns the markdown string.
    """
    # astimezone() attaches the local zone without shifting the wall clock, so the
    # rendered stamp stays local-time (what the operator reads) and tz-aware.
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    lines.append("# Backtest Report")
    lines.append(f"Generated: {timestamp}")
    lines.append("")

    lines.append("## Summary")
    lines.append(f"- {len(results)} bots evaluated")
    lines.extend(_question_set_summary_lines(question_set))
    lines.append("")

    lines.append("## Results by Bot")
    for result in results:
        lines.append("")
        lines.append(f"### {result.bot_name}")
        lines.append(f"- Questions scored: {result.num_scored}/{result.num_questions} (failed: {result.num_failed})")
        lines.append("")
        lines.extend(_metric_table_lines(aggregate_scores(result.scores)))
        lines.append("")

    lines.append("## Per-Type Breakdown")
    lines.extend(_per_type_breakdown_lines(results))

    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- Community prediction is the final CP at resolution time, which benefits from late-stage convergence"
    )
    lines.append("- Numeric community log scores unavailable (Metaculus removed aggregations from list API)")
    lines.append("")

    report = "\n".join(lines)

    if output_path is not None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report)
        logger.info(f"Backtest report written to {output_path}")

    return report


def _serialize_ground_truth(gt: GroundTruth) -> dict[str, Any]:
    """Serialize a GroundTruth to a JSON-safe dict."""
    return {
        "question_id": gt.question_id,
        "question_type": gt.question_type,
        "resolution": str(gt.resolution),
        "resolution_string": gt.resolution_string,
        "community_prediction": gt.community_prediction,
        "actual_resolution_time": gt.actual_resolution_time.isoformat() if gt.actual_resolution_time else None,
        "question_text": gt.question_text,
        "page_url": gt.page_url,
    }


def save_backtest_data(
    question_set: Any,
    results: list[BacktestResult],
    output_dir: str,
) -> str:
    """Save ground truths and scores as JSON for offline re-analysis.

    Returns the file path of the saved JSON.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Local-time stamp (tz-aware via astimezone, same wall clock) so the filename
    # matches what the operator sees in the report header.
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = output_path / f"backtest_data_{timestamp}.json"

    ground_truths_serialized: dict[str, dict] = {}
    if hasattr(question_set, "ground_truths"):
        for qid, gt in question_set.ground_truths.items():
            ground_truths_serialized[str(qid)] = _serialize_ground_truth(gt)

    results_serialized: list[dict[str, Any]] = []
    for result in results:
        scores_list = [asdict(s) for s in result.scores]
        results_serialized.append(
            {
                "bot_name": result.bot_name,
                "num_questions": result.num_questions,
                "num_scored": result.num_scored,
                "num_failed": result.num_failed,
                "scores": scores_list,
                "aggregated": aggregate_scores(result.scores),
            }
        )

    data = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "fetch_metadata": question_set.fetch_metadata if hasattr(question_set, "fetch_metadata") else {},
        "ground_truths": ground_truths_serialized,
        "results": results_serialized,
    }

    filepath.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Backtest data saved to {filepath}")

    return str(filepath)
