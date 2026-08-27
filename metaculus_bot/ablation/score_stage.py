"""The ablation's score stage: cached arm payloads in, paired-comparison artifacts out.

Everything between the on-disk arm payloads and the two files a scored run leaves behind:

* the adaptation layer — which cache arm each scoring-arm label reads from, which arm
  sets a given cache directory supports, and the per-question ``MagicMock`` reports whose
  ``isinstance`` identity drives ``ablation.scoring``'s metric dispatch;
* ``_build_paired_scores`` — per-comparison N, so a qid enters on any 2 succeeding arms
  rather than the full-arm intersection;
* ``_stage_score`` — bootstrap aggregation plus the markdown summary and the
  machine-readable score-run record written beside it.

Nothing here calls an LLM or the network — ``score_arm_for_qid`` / ``aggregate_paired`` /
``render_summary_markdown`` are pure functions over already-cached payloads. That is why
the stage sits here rather than in ``ablation.cli``: none of the three is a monkeypatch
target anywhere in the repo, so no test's ``monkeypatch.setattr`` needs their call sites
to live in the CLI module. Patch them at this module if a future test needs to.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
from forecasting_tools.data_models.numeric_report import NumericReport

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.forecasters import deserialize_prediction_value
from metaculus_bot.ablation.run_pdf import ARM_PDF_MIN1, ARM_PDF_MIN2
from metaculus_bot.ablation.run_stacker import ARM_MEAN, ARM_MEDIAN, ARM_STACK, ARM_STACK_AUG
from metaculus_bot.ablation.run_state import WorkingSet
from metaculus_bot.ablation.scoring import PairedScore, PairedStats, aggregate_paired, score_arm_for_qid
from metaculus_bot.ablation.scoring_report import render_summary_markdown


def _build_report_shim(qid: int, question: Any, payload: dict) -> Any:
    """Build a MagicMock report whose isinstance() check matches the question type.

    The score function in ``metaculus_bot.ablation.scoring`` uses ``isinstance``
    to dispatch to the right metric set; ``MagicMock(spec=BinaryReport)`` etc.
    pass that check.
    """
    stacker_pred = payload["stacker_prediction"]
    pred_type = stacker_pred["type"]

    if pred_type == "binary":
        report = MagicMock(spec=BinaryReport)
        report.prediction = float(stacker_pred["prob"])
        return report

    if pred_type == "multiple_choice":
        report = MagicMock(spec=MultipleChoiceReport)
        prediction = MagicMock()
        predicted_options: list[Any] = []
        for opt in stacker_pred["options"]:
            po = MagicMock()
            po.option_name = opt["option_name"]
            po.probability = float(opt["probability"])
            predicted_options.append(po)
        prediction.predicted_options = predicted_options
        report.prediction = prediction
        report.question = question
        return report

    if pred_type == "numeric":
        # Post-Bucket-1: ``deserialize_prediction_value`` returns a
        # ``PchipNumericDistribution`` whose ``.cdf`` already provides the
        # constraint-enforced 201-point CDF as a list of Percentile objects
        # (monotonic by construction — PCHIP enforces strict monotonicity in
        # the value axis). No defensive sort or duplicate-check needed; that
        # was a workaround for the old ``list[Percentile]`` return type when
        # free-model stackers emitted out-of-order declared percentiles.
        report = MagicMock(spec=NumericReport)
        deserialized = deserialize_prediction_value(stacker_pred, question)
        cdf_points: list[Any] = []
        for percentile in deserialized.cdf:
            point = MagicMock()
            point.value = float(percentile.value)
            point.percentile = float(percentile.percentile)
            cdf_points.append(point)
        prediction = MagicMock()
        prediction.cdf = cdf_points
        report.prediction = prediction
        report.question = question
        return report

    raise ValueError(f"Unknown prediction type {pred_type} for qid {qid}")


# Which cache arm each scoring-arm label reads from, in the canonical order
# ``score_arm_for_qid`` requires its ``arm_reports`` list to arrive in.
_SCORING_ARM_CACHE_KEYS: dict[str, str] = {
    "stack": ARM_STACK,
    "stack_aug": ARM_STACK_AUG,
    "pdf_min1": ARM_PDF_MIN1,
    "pdf_min2": ARM_PDF_MIN2,
    "median": ARM_MEDIAN,
    "mean": ARM_MEAN,
}
_SCORING_ARM_LABELS_6: tuple[str, ...] = ("stack", "stack_aug", "pdf_min1", "pdf_min2", "median", "mean")
_SCORING_ARM_LABELS_5: tuple[str, ...] = ("stack", "stack_aug", "pdf_min1", "pdf_min2", "median")
_SCORING_ARM_LABELS_3: tuple[str, ...] = ("stack", "stack_aug", "median")


def _scoring_arm_labels(*, has_pdf_arms: bool, has_mean_arm: bool) -> tuple[str, ...]:
    """Which arms enter the paired comparison for this cache directory.

    6-arm adds the deterministic mean arm on top of the 5-arm set, and is used only
    when BOTH pdf AND mean payloads are present. Old free-tier cache dirs (pdf, no
    mean) stay on the 5-arm path; pre-pdf data stays on the 3-arm path.
    """
    if has_pdf_arms and has_mean_arm:
        return _SCORING_ARM_LABELS_6
    if has_pdf_arms:
        return _SCORING_ARM_LABELS_5
    return _SCORING_ARM_LABELS_3


def _arm_score_inputs(working: WorkingSet, qid: int, question: Any) -> dict[str, tuple[Any, dict | None]]:
    """(report, payload) per scoring-arm label; report is None when the arm is missing or failed."""
    inputs: dict[str, tuple[Any, dict | None]] = {}
    for label, arm_key in _SCORING_ARM_CACHE_KEYS.items():
        payload = working.stacker_payloads.get(arm_key, {}).get(qid)
        report = _build_report_shim(qid, question, payload) if payload is not None and payload.get("success") else None
        inputs[label] = (report, payload)
    return inputs


def _score_run_payload(
    metadata: dict,
    stats: list[PairedStats],
    paired_scores: list[PairedScore],
) -> dict:
    """The machine-readable score-run record written alongside the markdown summary."""
    return {
        "metadata": metadata,
        "stats": [
            {
                "metric": s.metric,
                "question_type": s.question_type,
                "n": s.n,
                "mean_delta": s.mean_delta,
                "bootstrap_ci_low": s.bootstrap_ci_low,
                "bootstrap_ci_high": s.bootstrap_ci_high,
                "sign_test_p": s.sign_test_p,
                "wilcoxon_p": s.wilcoxon_p,
                "higher_is_better": s.higher_is_better,
            }
            for s in stats
        ],
        "paired_scores": [
            {
                "qid": s.qid,
                "question_type": s.question_type,
                "metric": s.metric,
                "comparison": s.comparison,
                "score_stack": s.score_stack,
                "score_stack_aug": s.score_stack_aug,
                "score_median": s.score_median,
                "delta": s.delta,
                "higher_is_better": s.higher_is_better,
            }
            for s in paired_scores
        ],
    }


def _build_paired_scores(
    working: WorkingSet,
    qids: list[int],
    arm_labels: tuple[str, ...],
) -> tuple[list[PairedScore], int]:
    """Score every qid that has at least 2 present arms; returns (rows, n_scored).

    The 2-arm floor is counted over EVERY arm, not just the ones in ``arm_labels``,
    so which qids clear the gate doesn't move when the cache directory gains an arm.
    ``n_scored`` counts the qids that produced at least one comparison row.
    """
    paired_scores: list[PairedScore] = []
    n_scored = 0
    for qid in qids:
        question = working.questions.get(qid)
        gt = working.ground_truths.get(qid)
        if question is None or gt is None:
            continue

        arm_inputs = _arm_score_inputs(working, qid, question)
        if sum(1 for report, _payload in arm_inputs.values() if report is not None) < 2:
            continue

        scores = score_arm_for_qid([(label, *arm_inputs[label]) for label in arm_labels], gt)
        if scores:
            n_scored += 1
        paired_scores.extend(scores)
    return paired_scores, n_scored


def _stage_score(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
) -> Path:
    """Build paired scores, aggregate, render summary, write run + summary files.

    Uses per-comparison N: each pairwise comparison only requires the two arms in
    that comparison to have succeeded for a qid. A qid is included if at least 2
    arms succeeded (enabling at least one comparison). This avoids collapsing N to
    the 5-way intersection of all arms.
    """
    # Union of all qids that have ANY arm payload.
    qid_set: set[int] = set()
    for arm_key in _SCORING_ARM_CACHE_KEYS.values():
        qid_set.update(working.stacker_payloads.get(arm_key, {}).keys())

    arm_labels = _scoring_arm_labels(
        has_pdf_arms=bool(working.stacker_payloads.get(ARM_PDF_MIN1))
        or bool(working.stacker_payloads.get(ARM_PDF_MIN2)),
        has_mean_arm=bool(working.stacker_payloads.get(ARM_MEAN)),
    )
    paired_scores, n_scored = _build_paired_scores(working, sorted(qid_set), arm_labels)

    stats = aggregate_paired(paired_scores, n_bootstrap=5000, seed=args.seed)

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    metadata = {
        "timestamp": timestamp,
        "n_questions": n_scored,
        "tournaments": ", ".join(args.tournaments),
        "resolved_after": args.resolved_after,
    }
    summary_md = render_summary_markdown(stats, paired_scores, metadata)
    cache.write_score_run(timestamp, _score_run_payload(metadata, stats, paired_scores))
    return cache.write_score_summary(timestamp, summary_md)
