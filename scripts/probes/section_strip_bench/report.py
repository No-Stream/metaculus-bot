"""The call rows and what is computed from them: arm means, paired deltas, the Markdown summary.

Everything here is a pure function of the rows and the per-question summaries ``run.json`` keeps, which
is what lets ``--rescore`` rebuild a run's results without the archive, the questions or the network.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from metaculus_bot.ablation.scoring import bootstrap_mean_ci, sign_test
from scripts.probes.section_strip_bench.bundle import FULL_ARM
from scripts.probes.section_strip_bench.scoring import peer_scale_factor

BOOTSTRAP_DRAWS = 5000

STATUS_SCORED = "scored"
STATUS_EXTRACTION_FAILED = "extraction_failed"
STATUS_BUILD_FAILED = "build_failed"
STATUS_UNIT_MISMATCH = "unit_mismatch"
STATUS_API_ERROR = "api_error"
STATUS_SKIPPED_SPEND_CAP = "skipped_spend_cap"

ALL_TYPES = "all"
PEER_SCALE = "all_peer_points"
PUBLISHED_COLUMN = "published"


@dataclass
class CallRow:
    """One (question, arm, replicate) call as written to ``calls.jsonl``."""

    question_id: int
    qtype: str
    n_options: int | None
    arm: str
    seed: int
    status: str
    score: float | None = None
    forecast: Any = None
    rung: str | None = None
    block_present: bool | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    elapsed_s: float = 0.0
    error: str | None = None
    rationale: str | None = None


def _mean_scores_by_question_arm(rows: Sequence[CallRow]) -> dict[tuple[int, str], float]:
    """Each (question, arm)'s mean score over its scored replicates."""
    scores: dict[tuple[int, str], list[float]] = {}
    for row in rows:
        if row.status == STATUS_SCORED and row.score is not None:
            scores.setdefault((row.question_id, row.arm), []).append(row.score)
    return {key: statistics.fmean(values) for key, values in scores.items()}


def _arm_summary(
    means: dict[tuple[int, str], float], summaries: Sequence[dict[str, Any]], arms: Sequence[str]
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Per arm (and the published reference), the mean and median question score by type and pooled."""
    types = sorted({s["qtype"] for s in summaries})
    type_of = {s["question_id"]: s["qtype"] for s in summaries}
    columns: dict[str, dict[int, float]] = {
        arm: {qid: value for (qid, row_arm), value in means.items() if row_arm == arm} for arm in arms
    }
    columns[PUBLISHED_COLUMN] = {
        s["question_id"]: s["published_score"] for s in summaries if s["published_score"] is not None
    }
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for label, by_qid in columns.items():
        out[label] = {}
        for qtype in [*types, ALL_TYPES]:
            values = [v for qid, v in by_qid.items() if qtype == ALL_TYPES or type_of[qid] == qtype]
            if values:
                out[label][qtype] = {
                    "n": len(values),
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                }
    return out


def _delta_stats(deltas: Sequence[float], *, seed: int) -> dict[str, float | int]:
    mean, low, high = bootstrap_mean_ci(list(deltas), n_bootstrap=BOOTSTRAP_DRAWS, seed=seed)
    return {
        "n": len(deltas),
        "mean_delta": mean,
        "ci95_low": low,
        "ci95_high": high,
        "median_delta": statistics.median(deltas),
        "n_full_better": sum(1 for d in deltas if d > 0),
        "n_arm_better": sum(1 for d in deltas if d < 0),
        "sign_test_p": sign_test(list(deltas)),
    }


def _paired_deltas(
    means: dict[tuple[int, str], float], summaries: Sequence[dict[str, Any]], arms: Sequence[str], *, seed: int
) -> dict[str, dict[str, dict[str, float | int]]]:
    """full-minus-arm per question, by type on the native score scale and pooled on spot-peer points."""
    types = sorted({s["qtype"] for s in summaries})
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for index, arm in enumerate(a for a in arms if a != FULL_ARM):
        native: dict[str, list[float]] = {qtype: [] for qtype in types}
        peer: list[float] = []
        for summary in summaries:
            qid = summary["question_id"]
            if (qid, FULL_ARM) not in means or (qid, arm) not in means:
                continue
            delta = means[(qid, FULL_ARM)] - means[(qid, arm)]
            native[summary["qtype"]].append(delta)
            peer.append(delta * peer_scale_factor(summary["qtype"], summary["n_options"]))
        groups = {**native, PEER_SCALE: peer}
        out[arm] = {
            label: _delta_stats(deltas, seed=seed + 1000 * index + offset)
            for offset, (label, deltas) in enumerate(groups.items())
            if deltas
        }
    return out


def _per_question_table(
    means: dict[tuple[int, str], float], summaries: Sequence[dict[str, Any]], arms: Sequence[str]
) -> list[dict[str, Any]]:
    table = []
    for summary in summaries:
        qid = summary["question_id"]
        full = means.get((qid, FULL_ARM))
        deltas = {
            arm: (full - means[(qid, arm)]) if full is not None and (qid, arm) in means else None
            for arm in arms
            if arm != FULL_ARM
        }
        table.append({**summary, "arm_scores": {arm: means.get((qid, arm)) for arm in arms}, "deltas": deltas})
    return table


def _spend_summary(rows: Sequence[CallRow]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, float]] = {}
    for row in rows:
        arm = by_arm.setdefault(
            row.arm, {"usd": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        )
        arm["usd"] += row.cost_usd or 0.0
        arm["prompt_tokens"] += row.prompt_tokens
        arm["completion_tokens"] += row.completion_tokens
        arm["reasoning_tokens"] += row.reasoning_tokens
    return {
        "forecaster_usd": sum(a["usd"] for a in by_arm.values()),
        "costed_calls": sum(1 for row in rows if row.cost_usd is not None),
        "by_arm": by_arm,
    }


def aggregate(
    rows: Sequence[CallRow], summaries: Sequence[dict[str, Any]], *, arms: Sequence[str], bootstrap_seed: int
) -> dict[str, Any]:
    """Everything ``results.json`` carries, computed from the call rows and the per-question summaries alone."""
    means = _mean_scores_by_question_arm(rows)
    return {
        "arm_summary": _arm_summary(means, summaries, arms),
        "paired_deltas": _paired_deltas(means, summaries, arms, seed=bootstrap_seed),
        "per_question": _per_question_table(means, summaries, arms),
        "call_status_by_arm": {arm: dict(Counter(row.status for row in rows if row.arm == arm)) for arm in arms},
        "extraction_rungs": dict(Counter(f"{row.arm}:{row.rung}" for row in rows if row.rung is not None)),
        "spend": _spend_summary(rows),
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": bootstrap_seed},
    }


def render_markdown(results: dict[str, Any], *, model: str, seeds: int) -> str:
    """The human summary: the paired deltas first, then the arm means, the call outcomes and the spend."""
    lines = [
        "# Section-strip bench",
        "",
        f"Model `{model}`, {seeds} replicate(s) per arm; scores are the platform log scores (binary and MC on their",
        "log-K baseline scale, numeric already halved). A positive delta means the FULL bundle scored higher than the arm.",
        "",
        "## Paired deltas, full minus arm",
        "",
        "| arm | group | n | mean Δ | 95% CI | median Δ | full better / arm better | sign test p |",
        "|---|---|--:|--:|---|--:|--:|--:|",
    ]
    for arm, groups in results["paired_deltas"].items():
        for label, stats in groups.items():
            lines.append(
                f"| {arm} | {label} | {stats['n']} | {stats['mean_delta']:+.2f} | "
                f"[{stats['ci95_low']:+.2f}, {stats['ci95_high']:+.2f}] | {stats['median_delta']:+.2f} | "
                f"{stats['n_full_better']} / {stats['n_arm_better']} | {stats['sign_test_p']:.3f} |"
            )
    lines += ["", "## Mean question score by arm", "", "| arm | type | n | mean | median |", "|---|---|--:|--:|--:|"]
    for arm, by_type in results["arm_summary"].items():
        for qtype, stats in by_type.items():
            lines.append(f"| {arm} | {qtype} | {stats['n']} | {stats['mean']:.2f} | {stats['median']:.2f} |")
    lines += ["", "## Call outcomes by arm", ""]
    lines += [f"- {arm}: {statuses}" for arm, statuses in results["call_status_by_arm"].items()]
    spend = results["spend"]
    lines += ["", "## Spend", ""]
    lines.append(f"- forecaster charges: ${spend['forecaster_usd']:.4f} over {spend['costed_calls']} costed calls")
    if "parser_usd" in spend:
        lines.append(f"- parser salvage charges: ${spend['parser_usd']:.4f}")
    lines += ["", "Per-question scores and deltas: `results.json` (`per_question`); every call: `calls.jsonl`.", ""]
    return "\n".join(lines)
