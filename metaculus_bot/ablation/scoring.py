"""Paired-difference scoring + report generation for the probabilistic-tools ablation benchmark.

For each question we score arms stack (rationale-only stacker), pdf (rationale + tools stacker),
and median (deterministic simple-aggregation baseline) against the same ground truth, then
compute three pairwise comparisons: pdf-stack, median-stack, median-pdf. Aggregates by metric,
question type, and comparison, with paired-bootstrap CI, sign test, and Wilcoxon signed-rank test.

Pure functions: score → aggregate → render. No state.

Note: the production score_report dispatcher in metaculus_bot/backtest/scoring.py deliberately
omits CRPS for numeric questions (it scores PMF-bucket log score only). This module re-uses
the underlying primitives directly so the numeric arm gets both metrics.
"""

import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
from forecasting_tools.data_models.numeric_report import NumericReport
from forecasting_tools.data_models.questions import OutOfBoundsResolution
from scipy import stats as scipy_stats

from metaculus_bot.backtest.scoring import (
    GroundTruth,
    mc_log_score_from_report,
    numeric_crps_from_report,
    numeric_log_score_from_report,
)
from metaculus_bot.scoring_common import PROB_CLAMP_MIN, binary_log_score, brier_score

logger: logging.Logger = logging.getLogger(__name__)

WILCOXON_MIN_N: int = 6
BOOTSTRAP_MIN_N: int = 5

# Comparison labels — every PairedScore row carries one of these and Δ is the
# corresponding signed difference.
# 3-arm comparisons (backward compat for callers that pass 3 arms).
COMPARISONS_3ARM: tuple[str, ...] = ("stack_aug-stack", "median-stack", "median-stack_aug")

# 5-arm comparisons (full 10 pairwise for pdf_min1/pdf_min2 dual-panel analysis).
COMPARISONS_5ARM: tuple[str, ...] = (
    "stack_aug-stack",
    "pdf_min1-stack",
    "pdf_min2-stack",
    "median-stack",
    "pdf_min1-stack_aug",
    "pdf_min2-stack_aug",
    "median-stack_aug",
    "pdf_min2-pdf_min1",
    "median-pdf_min1",
    "median-pdf_min2",
)

# Default: 3-arm for backward compatibility. Active comparison set is determined
# by whether score_arm_for_qid receives 3 or 5 arm tuples.
COMPARISONS: tuple[str, ...] = COMPARISONS_3ARM

METRIC_DIRECTION: dict[str, bool] = {
    "brier": False,
    "binary_log_score": True,
    "mc_log_score": True,
    "numeric_log_score": True,
    "crps": False,
}

SATURATION_FRAC_LOG_SCORE: float = 0.05
SATURATION_EPS_BRIER: float = 0.01

# Empirical floor for numeric log score on 201-pt CDF with min-step PMF.
# Plan formula gave ~-200; latest summary (qids 43129, 42747, 43171) shows
# -219.9756 in practice. Use empirical, not formula.
NUMERIC_LOG_SCORE_FLOOR: float = -220.0

# 100*(log2(PROB_CLAMP_MIN) + 1) ≈ -1228.77 with PROB_CLAMP_MIN=1e-4 (canonical eps).
BINARY_LOG_SCORE_FLOOR: float = 100.0 * (math.log2(PROB_CLAMP_MIN) + 1.0)

# (1 - PROB_CLAMP_MIN)^2 ≈ 0.9998 with PROB_CLAMP_MIN=1e-4.
BRIER_FLOOR: float = (1.0 - PROB_CLAMP_MIN) ** 2


def _mc_log_score_floor(n_options: int) -> float:
    """Floor for Metaculus-style MC log score with K options at clip prob 0.02."""
    if n_options < 2:
        raise ValueError(f"MC needs >= 2 options, got {n_options}")
    return 100.0 * (math.log2(PROB_CLAMP_MIN) / math.log2(n_options) + 1.0)


def is_score_saturated(metric: str, score: float, n_mc_options: int | None = None) -> bool:
    """True iff a score sits within ε of its schema's max-confident-wrong floor.

    Log-score metrics use a fractional eps (5% of the distance from 0 to floor) so the
    saturation cutoff scales with each metric's range. Brier keeps an absolute eps because
    it lives on the [0, 1] scale where a fixed 0.01 buffer is well-defined. This avoids
    the prior calibration drift where absolute eps=25 produced 1.0% of brier's range but
    only 2.0% of binary log score's distance to floor — letting the two metrics disagree
    on saturation status for the same forecast.

    Higher-is-better metrics saturate when ``score <= floor * (1 - SATURATION_FRAC_LOG_SCORE)``.
    Lower-is-better brier saturates when ``score >= BRIER_FLOOR - SATURATION_EPS_BRIER``.
    Returns False for metrics without a comparable floor (crps).

    n_mc_options is required for mc_log_score; ignored otherwise.
    """
    if metric == "numeric_log_score":
        cutoff = NUMERIC_LOG_SCORE_FLOOR * (1.0 - SATURATION_FRAC_LOG_SCORE)
        return score <= cutoff
    if metric == "binary_log_score":
        cutoff = BINARY_LOG_SCORE_FLOOR * (1.0 - SATURATION_FRAC_LOG_SCORE)
        return score <= cutoff
    if metric == "mc_log_score":
        if n_mc_options is None:
            return False
        cutoff = _mc_log_score_floor(n_mc_options) * (1.0 - SATURATION_FRAC_LOG_SCORE)
        return score <= cutoff
    if metric == "brier":
        return score >= BRIER_FLOOR - SATURATION_EPS_BRIER
    return False


def bootstrap_median_ci(
    deltas: list[float], n_bootstrap: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap on deltas — resample, recompute MEDIAN, take CI percentiles.

    For n < BOOTSTRAP_MIN_N, returns (median, median, median).
    """
    n = len(deltas)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    arr = np.asarray(deltas, dtype=float)
    median = float(np.median(arr))

    if n < BOOTSTRAP_MIN_N:
        return (median, median, median)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_medians = np.median(arr[indices], axis=1)

    lo = float(np.percentile(boot_medians, 100 * alpha / 2))
    hi = float(np.percentile(boot_medians, 100 * (1 - alpha / 2)))
    return median, lo, hi


@dataclass
class PairedScore:
    qid: int
    question_type: str  # "binary" | "multiple_choice" | "numeric"
    metric: str  # "brier" | "binary_log_score" | "mc_log_score" | "numeric_log_score" | "crps"
    comparison: str  # one of COMPARISONS_3ARM or COMPARISONS_5ARM
    # All arm scores populated on every row regardless of comparison so per-arm
    # raw stats and the per-question diagnostic can derive numbers from any subset.
    score_stack: float
    score_stack_aug: float
    score_median: float
    delta: float  # comparison-specific signed diff (X-Y: score_X - score_Y)
    higher_is_better: bool
    # pdf_min1 and pdf_min2 arm scores (NaN when using 3-arm mode).
    score_pdf_min1: float = float("nan")
    score_pdf_min2: float = float("nan")
    # Optional confounder fields populated by score_arm_for_qid when stacker
    # payloads are supplied. None on legacy callers; surfaced into the summary
    # report when present so the user can read fallback rate, n-forecasters
    # delta, and treatment activation per question.
    stack_model_used: str | None = None
    stack_aug_model_used: str | None = None
    pdf_min1_model_used: str | None = None
    pdf_min2_model_used: str | None = None
    median_model_used: str | None = None
    n_forecasters_stack: int | None = None
    n_forecasters_stack_aug: int | None = None
    n_forecasters_pdf_min1: int | None = None
    n_forecasters_pdf_min2: int | None = None
    n_forecasters_median: int | None = None
    stack_aug_tools_fired: bool | None = None
    median_tools_fired: bool | None = None
    # Per-arm saturation flags. True iff the score sits within ε of the
    # schema's max-confident-wrong floor (see is_score_saturated). Both-True
    # rows are mechanical Δ≈0 draws and don't carry signal.
    is_saturated_stack: bool = False
    is_saturated_stack_aug: bool = False
    is_saturated_pdf_min1: bool = False
    is_saturated_pdf_min2: bool = False
    is_saturated_median: bool = False


@dataclass
class PairedStats:
    metric: str
    question_type: str | None  # None = overall (all types for this metric)
    comparison: str  # "stack_aug-stack" | "median-stack" | "median-stack_aug"
    n: int
    mean_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    sign_test_p: float
    wilcoxon_p: float | None
    higher_is_better: bool
    median_delta: float = float("nan")
    median_ci_low: float = float("nan")
    median_ci_high: float = float("nan")
    n_clean: int = 0  # count of pairs where neither arm in this comparison saturated
    mean_delta_clean: float = float("nan")  # mean Δ over non-saturated pairs (NaN if n_clean == 0)


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------


def _score_binary_arm(report: BinaryReport, outcome: bool) -> dict[str, float]:
    p = float(report.prediction)
    return {
        "brier": brier_score(p, outcome),
        "binary_log_score": binary_log_score(p, outcome),
    }


def _score_numeric_arm(report: NumericReport, resolution: Any) -> dict[str, float] | None:
    log_score = numeric_log_score_from_report(report, resolution)
    crps = numeric_crps_from_report(report, resolution)
    if log_score is None or crps is None:
        return None
    return {"numeric_log_score": log_score, "crps": crps}


def _score_mc_arm(report: MultipleChoiceReport, correct_option: str) -> dict[str, float] | None:
    log_score = mc_log_score_from_report(report, correct_option)
    if log_score is None:
        return None
    return {"mc_log_score": log_score}


def _confounders_for_arm(payload: dict | None, arm: str) -> dict[str, Any]:
    """Extract the confounder fields for a single arm from its cached stacker payload.

    Returns the fields keyed for that arm (e.g., ``stack_model_used``,
    ``n_forecasters_stack``, ``stack_aug_tools_fired``). Empty dict when payload is None.
    Arm median contributes ``median_tools_fired`` (always False — deterministic baseline).
    """
    if payload is None:
        return {}
    out: dict[str, Any] = {f"{arm}_model_used": payload.get("stacker_model_used")}
    n_used = payload.get("n_forecasters_used")
    out[f"n_forecasters_{arm}"] = int(n_used) if n_used is not None else None
    if arm == "stack_aug":
        out["stack_aug_tools_fired"] = bool(payload.get("cross_model_aggregation"))
    elif arm == "median":
        # ARM_MEDIAN is deterministic — never fires LLM tools. Reflect this explicitly.
        out["median_tools_fired"] = bool(payload.get("cross_model_aggregation"))
    return out


def _saturation_for_metrics(metrics: dict[str, float], n_mc_options: int | None = None) -> dict[str, bool]:
    return {metric: is_score_saturated(metric, score, n_mc_options) for metric, score in metrics.items()}


def _build_paired_scores(
    qid: int,
    question_type: str,
    metrics_stack: dict[str, float],
    metrics_pdf: dict[str, float],
    metrics_median: dict[str, float],
    saturation_stack: dict[str, bool],
    saturation_pdf: dict[str, bool],
    saturation_median: dict[str, bool],
    confounders: dict[str, Any],
    *,
    metrics_pdf_min1: dict[str, float] | None = None,
    metrics_pdf_min2: dict[str, float] | None = None,
    saturation_pdf_min1: dict[str, bool] | None = None,
    saturation_pdf_min2: dict[str, bool] | None = None,
) -> list[PairedScore]:
    """Emit one PairedScore per (metric, comparison).

    In 3-arm mode (pdf_min1/pdf_min2 = None): 3 comparisons per metric.
    In 5-arm mode: 10 comparisons per metric.

    Every row carries all arm scores + saturation flags so per-arm raw stats can be
    derived from the union of rows. ``delta`` is comparison-specific.
    """
    is_five_arm = metrics_pdf_min1 is not None and metrics_pdf_min2 is not None

    if is_five_arm:
        comparisons = COMPARISONS_5ARM
    else:
        comparisons = COMPARISONS_3ARM

    # Score accessor by arm label
    def _score(metric: str, arm: str) -> float:
        if arm == "stack":
            return metrics_stack[metric]
        if arm == "stack_aug":
            return metrics_pdf[metric]
        if arm == "pdf_min1" and metrics_pdf_min1 is not None:
            return metrics_pdf_min1[metric]
        if arm == "pdf_min2" and metrics_pdf_min2 is not None:
            return metrics_pdf_min2[metric]
        if arm == "median":
            return metrics_median[metric]
        raise ValueError(f"Unknown arm {arm!r}")

    def _sat(metric: str, arm: str) -> bool:
        if arm == "stack":
            return saturation_stack[metric]
        if arm == "stack_aug":
            return saturation_pdf[metric]
        if arm == "pdf_min1" and saturation_pdf_min1 is not None:
            return saturation_pdf_min1[metric]
        if arm == "pdf_min2" and saturation_pdf_min2 is not None:
            return saturation_pdf_min2[metric]
        if arm == "median":
            return saturation_median[metric]
        if arm in ("pdf_min1", "pdf_min2"):
            return False
        raise ValueError(f"Unknown arm {arm!r}")

    paired: list[PairedScore] = []
    for metric, s_stack in metrics_stack.items():
        s_stack_aug = metrics_pdf[metric]
        s_median = metrics_median[metric]
        s_pdf_min1 = metrics_pdf_min1[metric] if metrics_pdf_min1 else float("nan")
        s_pdf_min2 = metrics_pdf_min2[metric] if metrics_pdf_min2 else float("nan")

        for comparison in comparisons:
            left, right = comparison.split("-", 1)
            delta = _score(metric, left) - _score(metric, right)
            paired.append(
                PairedScore(
                    qid=qid,
                    question_type=question_type,
                    metric=metric,
                    comparison=comparison,
                    score_stack=s_stack,
                    score_stack_aug=s_stack_aug,
                    score_median=s_median,
                    score_pdf_min1=s_pdf_min1,
                    score_pdf_min2=s_pdf_min2,
                    delta=delta,
                    higher_is_better=METRIC_DIRECTION[metric],
                    is_saturated_stack=saturation_stack[metric],
                    is_saturated_stack_aug=saturation_pdf[metric],
                    is_saturated_pdf_min1=_sat(metric, "pdf_min1"),
                    is_saturated_pdf_min2=_sat(metric, "pdf_min2"),
                    is_saturated_median=saturation_median[metric],
                    **confounders,
                )
            )
    return paired


def score_arm_for_qid(
    arm_reports: list[tuple[str, ForecastReport | None, dict | None]],
    ground_truth: GroundTruth,
) -> list[PairedScore]:
    """Score arms vs ground truth, return PairedScores for pairwise comparisons where both arms are present.

    ``arm_reports`` is either a 3-element or 5-element list:

    **3-arm** (backward compat): labels in order [stack, stack_aug, median].
      - Up to 3 comparisons per metric.

    **5-arm**: labels in order [stack, stack_aug, pdf_min1, pdf_min2, median].
      - Up to 10 comparisons per metric.

    Each element is ``(label, report, payload)`` where report may be None
    (arm failed/missing for this qid) and payload is the cached stacker payload
    dict or None.

    Comparisons are only emitted when BOTH arms in the comparison have non-None
    reports that score successfully. This enables per-comparison N rather than
    requiring all arms to succeed for every qid.

    Output rows per metric per question type (for N_present comparisons):
      - Binary: 2 metrics * N_present rows.
      - MC: 1 metric * N_present rows.
      - Numeric: 2 metrics * N_present rows.

    Returns empty list if fewer than 2 arms are present, or if resolution type
    is incompatible (canceled, mismatched type, parse error).
    """
    if len(arm_reports) == 3:
        expected_labels = ["stack", "stack_aug", "median"]
    elif len(arm_reports) == 5:
        expected_labels = ["stack", "stack_aug", "pdf_min1", "pdf_min2", "median"]
    else:
        raise ValueError(f"score_arm_for_qid expects 3 or 5 arm tuples, got {len(arm_reports)}")

    actual_labels = [t[0] for t in arm_reports]
    if actual_labels != expected_labels:
        raise ValueError(f"score_arm_for_qid expects arms in order {expected_labels}, got {actual_labels}")

    is_five_arm = len(arm_reports) == 5

    _, report_stack, payload_stack = arm_reports[0]
    _, report_stack_aug, payload_stack_aug = arm_reports[1]
    if is_five_arm:
        _, report_pdf_min1, payload_pdf_min1 = arm_reports[2]
        _, report_pdf_min2, payload_pdf_min2 = arm_reports[3]
        _, report_median, payload_median = arm_reports[4]
    else:
        report_pdf_min1 = report_pdf_min2 = None
        payload_pdf_min1 = payload_pdf_min2 = None
        _, report_median, payload_median = arm_reports[2]

    # Count present arms — need at least 2 for any comparison to be possible.
    present_reports = [
        r for r in [report_stack, report_stack_aug, report_pdf_min1, report_pdf_min2, report_median] if r is not None
    ]
    if len(present_reports) < 2:
        return []

    qid = ground_truth.question_id
    resolution = ground_truth.resolution
    confounders: dict[str, Any] = {}
    confounders.update(_confounders_for_arm(payload_stack, "stack"))
    confounders.update(_confounders_for_arm(payload_stack_aug, "stack_aug"))
    if is_five_arm:
        confounders.update(_confounders_for_arm(payload_pdf_min1, "pdf_min1"))
        confounders.update(_confounders_for_arm(payload_pdf_min2, "pdf_min2"))
    confounders.update(_confounders_for_arm(payload_median, "median"))

    # Determine question type from the first non-None report.
    first_report = present_reports[0]

    # Validate type consistency among non-None reports. Use isinstance against
    # concrete types rather than type() comparison, because MagicMock(spec=X)
    # passes isinstance(_, X) but type() returns a unique MagicMock subclass.
    if isinstance(first_report, BinaryReport):
        expected_type = BinaryReport
    elif isinstance(first_report, NumericReport):
        expected_type = NumericReport
    elif isinstance(first_report, MultipleChoiceReport):
        expected_type = MultipleChoiceReport
    else:
        logger.warning(f"Q{qid}: unsupported report type: {type(first_report).__name__}")
        return []
    for r in present_reports[1:]:
        if not isinstance(r, expected_type):
            logger.warning(f"Q{qid}: mismatched report types among present arms")
            return []

    # Score each present arm; collect metrics per arm label.
    arm_metrics: dict[str, dict[str, float] | None] = {}
    arm_saturation: dict[str, dict[str, bool] | None] = {}
    n_mc_options: int | None = None

    if isinstance(first_report, BinaryReport):
        if not isinstance(resolution, bool):
            logger.warning(f"Q{qid}: expected bool for binary resolution, got {type(resolution).__name__}")
            return []
        question_type = "binary"
        for label, report in [
            ("stack", report_stack),
            ("stack_aug", report_stack_aug),
            ("pdf_min1", report_pdf_min1),
            ("pdf_min2", report_pdf_min2),
            ("median", report_median),
        ]:
            if report is not None:
                metrics_for_arm = _score_binary_arm(report, resolution)
                arm_metrics[label] = metrics_for_arm
                arm_saturation[label] = _saturation_for_metrics(metrics_for_arm)
            else:
                arm_metrics[label] = None
                arm_saturation[label] = None

    elif isinstance(first_report, NumericReport):
        if isinstance(resolution, bool):
            logger.warning(
                f"Q{qid}: bool resolution {resolution!r} routed to numeric scorer; skipping. "
                "This indicates an upstream type-routing bug."
            )
            return []
        if not isinstance(resolution, (int, float, OutOfBoundsResolution)):
            logger.warning(
                f"Q{qid}: expected numeric or OutOfBoundsResolution for numeric question, "
                f"got {type(resolution).__name__}"
            )
            return []
        question_type = "numeric"
        for label, report in [
            ("stack", report_stack),
            ("stack_aug", report_stack_aug),
            ("pdf_min1", report_pdf_min1),
            ("pdf_min2", report_pdf_min2),
            ("median", report_median),
        ]:
            if report is not None:
                scored = _score_numeric_arm(report, resolution)
                if scored is None:
                    # Scoring failed (e.g., CDF issue) — treat this arm as absent.
                    arm_metrics[label] = None
                    arm_saturation[label] = None
                else:
                    arm_metrics[label] = scored
                    arm_saturation[label] = _saturation_for_metrics(scored)
            else:
                arm_metrics[label] = None
                arm_saturation[label] = None

    elif isinstance(first_report, MultipleChoiceReport):
        if not isinstance(resolution, str):
            logger.warning(f"Q{qid}: expected str for MC resolution, got {type(resolution).__name__}")
            return []
        question_type = "multiple_choice"
        n_mc_options = len(first_report.question.options)
        for label, report in [
            ("stack", report_stack),
            ("stack_aug", report_stack_aug),
            ("pdf_min1", report_pdf_min1),
            ("pdf_min2", report_pdf_min2),
            ("median", report_median),
        ]:
            if report is not None:
                scored = _score_mc_arm(report, resolution)
                if scored is None:
                    arm_metrics[label] = None
                    arm_saturation[label] = None
                else:
                    arm_metrics[label] = scored
                    arm_saturation[label] = _saturation_for_metrics(scored, n_mc_options=n_mc_options)
            else:
                arm_metrics[label] = None
                arm_saturation[label] = None
    else:
        logger.warning(f"Q{qid}: unsupported report type: {type(first_report).__name__}")
        return []

    # Determine which arms actually scored successfully.
    scored_arms = {label for label, m in arm_metrics.items() if m is not None}
    if len(scored_arms) < 2:
        return []

    # Choose comparison set based on mode.
    if is_five_arm:
        comparisons = COMPARISONS_5ARM
    else:
        comparisons = COMPARISONS_3ARM

    # Filter to comparisons where both arms scored successfully.
    active_comparisons = []
    for comparison in comparisons:
        left, right = comparison.split("-", 1)
        if left in scored_arms and right in scored_arms:
            active_comparisons.append(comparison)

    if not active_comparisons:
        return []

    # Build PairedScore rows for active comparisons only.
    # Collect the canonical metric set from the first scored arm.
    first_scored_label = next(iter(scored_arms))
    first_scored_metrics = arm_metrics[first_scored_label]
    assert first_scored_metrics is not None  # guaranteed by scored_arms membership
    metric_names = list(first_scored_metrics.keys())

    paired: list[PairedScore] = []
    for metric in metric_names:
        # Extract scores for all arms (NaN for absent arms).
        s_stack = arm_metrics["stack"][metric] if arm_metrics["stack"] else float("nan")
        s_stack_aug = arm_metrics["stack_aug"][metric] if arm_metrics["stack_aug"] else float("nan")
        s_pdf_min1 = arm_metrics["pdf_min1"][metric] if arm_metrics["pdf_min1"] else float("nan")
        s_pdf_min2 = arm_metrics["pdf_min2"][metric] if arm_metrics["pdf_min2"] else float("nan")
        s_median = arm_metrics["median"][metric] if arm_metrics["median"] else float("nan")

        def _get_score(arm_label: str) -> float:
            m = arm_metrics[arm_label]
            return m[metric] if m else float("nan")

        def _get_sat(arm_label: str) -> bool:
            s = arm_saturation[arm_label]
            return s[metric] if s else False

        for comparison in active_comparisons:
            left, right = comparison.split("-", 1)
            delta = _get_score(left) - _get_score(right)
            paired.append(
                PairedScore(
                    qid=qid,
                    question_type=question_type,
                    metric=metric,
                    comparison=comparison,
                    score_stack=s_stack,
                    score_stack_aug=s_stack_aug,
                    score_median=s_median,
                    score_pdf_min1=s_pdf_min1,
                    score_pdf_min2=s_pdf_min2,
                    delta=delta,
                    higher_is_better=METRIC_DIRECTION[metric],
                    is_saturated_stack=_get_sat("stack"),
                    is_saturated_stack_aug=_get_sat("stack_aug"),
                    is_saturated_pdf_min1=_get_sat("pdf_min1"),
                    is_saturated_pdf_min2=_get_sat("pdf_min2"),
                    is_saturated_median=_get_sat("median"),
                    **confounders,
                )
            )
    return paired


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------


def bootstrap_mean_ci(
    deltas: list[float], n_bootstrap: int = 5000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap on the deltas: resample with replacement, recompute mean, take CI percentiles.

    For n < 5, returns (mean, mean, mean) — too few samples for a meaningful CI.
    """
    n = len(deltas)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    arr = np.asarray(deltas, dtype=float)
    mean = float(arr.mean())

    if n < BOOTSTRAP_MIN_N:
        return (mean, mean, mean)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = arr[indices].mean(axis=1)

    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, lo, hi


def sign_test(deltas: list[float]) -> float:
    """Two-sided sign test on Δ vs 0. Ties (delta == 0) excluded.

    Returns p-value. If all values are ties, returns 1.0.
    """
    nonzero = [d for d in deltas if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0

    n_pos = sum(1 for d in nonzero if d > 0)
    result = scipy_stats.binomtest(n_pos, n, p=0.5, alternative="two-sided")
    return float(result.pvalue)


def wilcoxon_signed_rank(deltas: list[float]) -> float | None:
    """Wilcoxon signed-rank, two-sided. Returns p-value, or None if n < WILCOXON_MIN_N.

    All-zero or all-equal arrays raise inside scipy; we return None in that case as well.
    """
    if len(deltas) < WILCOXON_MIN_N:
        return None

    if all(d == 0.0 for d in deltas):
        return None

    try:
        result = scipy_stats.wilcoxon(deltas, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return None
    return float(result.pvalue)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _sat_for_arm(score: PairedScore, arm: str) -> bool:
    """Return the saturation flag for a named arm on a PairedScore row."""
    if arm == "stack":
        return score.is_saturated_stack
    if arm == "stack_aug":
        return score.is_saturated_stack_aug
    if arm == "pdf_min1":
        return score.is_saturated_pdf_min1
    if arm == "pdf_min2":
        return score.is_saturated_pdf_min2
    if arm == "median":
        return score.is_saturated_median
    raise ValueError(f"Unknown arm {arm!r}")


def _saturation_pair_for_comparison(score: PairedScore) -> bool:
    """True iff either arm in this row's comparison saturated."""
    left, right = score.comparison.split("-", 1)
    return _sat_for_arm(score, left) or _sat_for_arm(score, right)


def _stats_for_group(
    metric: str,
    question_type: str | None,
    comparison: str,
    group: list[PairedScore],
    n_bootstrap: int,
    mean_seed: int,
    median_seed: int,
) -> PairedStats:
    raw_deltas = [s.delta for s in group]
    valid_pairs = [(s, d) for s, d in zip(group, raw_deltas, strict=True) if not math.isnan(d)]
    n_dropped = len(raw_deltas) - len(valid_pairs)
    if n_dropped > 0:
        logger.warning(
            "_stats_for_group: dropped %d NaN delta(s) from metric=%s qtype=%s comparison=%s "
            "(propagating NaN through bootstrap would zero out the entire group)",
            n_dropped,
            metric,
            question_type if question_type is not None else "__overall__",
            comparison,
        )

    if not valid_pairs:
        return PairedStats(
            metric=metric,
            question_type=question_type,
            comparison=comparison,
            n=0,
            mean_delta=float("nan"),
            bootstrap_ci_low=float("nan"),
            bootstrap_ci_high=float("nan"),
            sign_test_p=float("nan"),
            wilcoxon_p=None,
            higher_is_better=group[0].higher_is_better,
            median_delta=float("nan"),
            median_ci_low=float("nan"),
            median_ci_high=float("nan"),
            n_clean=0,
            mean_delta_clean=float("nan"),
        )

    valid_group = [s for s, _ in valid_pairs]
    deltas = [d for _, d in valid_pairs]
    mean, lo, hi = bootstrap_mean_ci(deltas, n_bootstrap=n_bootstrap, seed=mean_seed)
    median, mlo, mhi = bootstrap_median_ci(deltas, n_bootstrap=n_bootstrap, seed=median_seed)

    clean_deltas = [s.delta for s in valid_group if not _saturation_pair_for_comparison(s)]
    n_clean = len(clean_deltas)
    mean_delta_clean = float(np.mean(clean_deltas)) if n_clean > 0 else float("nan")

    return PairedStats(
        metric=metric,
        question_type=question_type,
        comparison=comparison,
        n=len(valid_group),
        mean_delta=mean,
        bootstrap_ci_low=lo,
        bootstrap_ci_high=hi,
        sign_test_p=sign_test(deltas),
        wilcoxon_p=wilcoxon_signed_rank(deltas),
        higher_is_better=group[0].higher_is_better,
        median_delta=median,
        median_ci_low=mlo,
        median_ci_high=mhi,
        n_clean=n_clean,
        mean_delta_clean=mean_delta_clean,
    )


def _group_seed(
    parent_seed: int,
    metric: str,
    question_type: str | None,
    comparison: str,
    subgroup: str = "",
) -> int:
    """Derive a deterministic 32-bit per-group seed from (parent_seed, metric, qtype, comparison, subgroup).

    Ensures each (metric, qtype, comparison, subgroup) bootstrap draws from a distinct RNG
    stream so per-group CIs are not artificially correlated by sharing top-level state, while
    keeping reproducibility from the parent seed. ``subgroup`` lets the caller derive a
    separate seed for, e.g., the median bootstrap vs the mean bootstrap.

    Uses ``hashlib.sha256`` rather than Python's builtin ``hash()`` because the
    builtin tuple/string hash is randomized per process unless ``PYTHONHASHSEED``
    is set. SHA-256 keeps the seed reproducible across distinct Python processes.
    """
    qtype_str = question_type if question_type is not None else "__overall__"
    key = f"{metric}|{qtype_str}|{comparison}|{subgroup}".encode("utf-8")
    digest_int = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    seed_seq = np.random.SeedSequence([parent_seed, digest_int])
    return int(seed_seq.generate_state(1)[0])


def aggregate_paired(scores: list[PairedScore], n_bootstrap: int = 5000, seed: int = 0) -> list[PairedStats]:
    """Group by (metric, question_type, comparison), compute paired stats. Append per-(metric, comparison) overall rows.

    Returns one PairedStats per (metric, question_type, comparison) plus one PairedStats per
    (metric, comparison) with question_type=None (the overall aggregation across all types
    for that metric/comparison).

    Each group gets a distinct bootstrap seed derived from `seed` so the per-group
    CIs are not correlated by sharing RNG state. Reproducibility holds: identical
    `seed` reproduces identical PairedStats across all groups.
    """
    if not scores:
        return []

    by_metric_comparison: dict[tuple[str, str], list[PairedScore]] = {}
    by_metric_type_comparison: dict[tuple[str, str, str], list[PairedScore]] = {}
    for s in scores:
        by_metric_comparison.setdefault((s.metric, s.comparison), []).append(s)
        by_metric_type_comparison.setdefault((s.metric, s.question_type, s.comparison), []).append(s)

    out: list[PairedStats] = []
    for (metric, qtype, comparison), group in sorted(by_metric_type_comparison.items()):
        out.append(
            _stats_for_group(
                metric,
                qtype,
                comparison,
                group,
                n_bootstrap,
                mean_seed=_group_seed(seed, metric, qtype, comparison, subgroup="mean"),
                median_seed=_group_seed(seed, metric, qtype, comparison, subgroup="median"),
            )
        )

    for (metric, comparison), group in sorted(by_metric_comparison.items()):
        out.append(
            _stats_for_group(
                metric,
                None,
                comparison,
                group,
                n_bootstrap,
                mean_seed=_group_seed(seed, metric, None, comparison, subgroup="mean"),
                median_seed=_group_seed(seed, metric, None, comparison, subgroup="median"),
            )
        )

    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _format_score(x: float) -> str:
    if math.isnan(x):
        return "NaN"
    return f"{x:.4f}"


def _format_p(x: float | None) -> str:
    if x is None:
        return "N/A"
    if math.isnan(x):
        return "NaN"
    return f"{x:.3f}"


def _format_n(n: int) -> str:
    return f"{n:.0f}"


def _stats_table_header() -> list[str]:
    return [
        (
            "| Metric | Type | N | Mean Δ | Mean 95% CI | Median Δ | Median 95% CI "
            "| NoSat Δ (n_clean) | Sign p | Wilcoxon p | Better when Δ |"
        ),
        (
            "|--------|------|---|--------|-------------|----------|---------------"
            "|-------------------|--------|------------|---------------|"
        ),
    ]


LOW_N_THRESHOLD: int = 30


def _format_clean_mean(stats: PairedStats) -> str:
    if stats.n_clean == 0:
        return f"NaN ({stats.n_clean})"
    return f"{_format_score(stats.mean_delta_clean)} ({stats.n_clean})"


def _direction_label(s: PairedStats) -> str:
    """Which arm wins when Δ has the favorable sign for this comparison + metric.

    Comparison X-Y means Δ = score_X - score_Y. With higher-is-better metrics (log scores),
    Δ > 0 means X wins. With lower-is-better metrics (brier, crps), Δ < 0 means X wins.
    Either way the *left* arm wins on the favorable sign.
    """
    left = s.comparison.split("-")[0]
    arrow = "Δ > 0" if s.higher_is_better else "Δ < 0"
    return f"{left} when {arrow}"


def _stats_row(s: PairedStats) -> str:
    qtype = s.question_type if s.question_type is not None else "overall"
    mean_ci_str = f"[{_format_score(s.bootstrap_ci_low)}, {_format_score(s.bootstrap_ci_high)}]"
    median_ci_str = f"[{_format_score(s.median_ci_low)}, {_format_score(s.median_ci_high)}]"
    n_str = _format_n(s.n)
    if s.question_type is not None and s.n < LOW_N_THRESHOLD:
        n_str = f"{n_str}*"
    return (
        f"| {s.metric} "
        f"| {qtype} "
        f"| {n_str} "
        f"| {_format_score(s.mean_delta)} "
        f"| {mean_ci_str} "
        f"| {_format_score(s.median_delta)} "
        f"| {median_ci_str} "
        f"| {_format_clean_mean(s)} "
        f"| {_format_p(s.sign_test_p)} "
        f"| {_format_p(s.wilcoxon_p)} "
        f"| {_direction_label(s)} |"
    )


def _has_confounder_data(scores: list[PairedScore]) -> bool:
    """True iff at least one PairedScore has a confounder field populated."""
    return any(
        s.stack_model_used is not None
        or s.stack_aug_model_used is not None
        or s.median_model_used is not None
        or s.stack_aug_tools_fired is not None
        or s.median_tools_fired is not None
        or s.n_forecasters_stack is not None
        or s.n_forecasters_stack_aug is not None
        or s.n_forecasters_median is not None
        for s in scores
    )


def _md_escape_cell(value: Any) -> str:
    """Escape pipes in a markdown table cell so embedded ``|`` don't break the table layout.

    Currently only an issue if a confounder field carries a raw model identifier with ``|``
    (e.g., ``"claude-opus-4.7|via-openrouter"``). Production code only writes literal
    "primary"/"fallback"/"simple_aggregation" — escaping is defensive against future config changes.
    """
    if value is None:
        return "-"
    return str(value).replace("|", "\\|")


def _per_question_diagnostic_table(scores: list[PairedScore]) -> list[str]:
    """One row per (qid, metric) — dedupe across the three comparisons.

    Each row carries all three scores + three deltas inline. Saturation flags per arm.
    Confounder columns appended when payload data is present.
    """
    show_confounders = _has_confounder_data(scores)
    if show_confounders:
        lines = [
            (
                "| qid | type | metric | stack | stack_aug | median | Δ_stack_aug-stack | Δ_median-stack | Δ_median-stack_aug "
                "| sat_stack | sat_stack_aug | sat_median | stack_model | stack_aug_model | median_model | stack_aug_tools | median_tools |"
            ),
            (
                "|-----|------|--------|-------|-----|--------|-------------|----------------|-------------"
                "|-----------|---------|-----------|-------------|-----------|--------------|-----------|--------------|"
            ),
        ]
    else:
        lines = [
            (
                "| qid | type | metric | stack | stack_aug | median | Δ_stack_aug-stack | Δ_median-stack | Δ_median-stack_aug "
                "| sat_stack | sat_stack_aug | sat_median |"
            ),
            (
                "|-----|------|--------|-------|-----|--------|-------------|----------------|-------------"
                "|-----------|---------|-----------|"
            ),
        ]

    # Dedupe to one row per (qid, metric). Pick the pdf-stack row as the canonical source of
    # the comparison-independent fields (scores, saturation, confounders); compute deltas
    # directly from score_stack/pdf/median. Sort by absolute Δ_pdf-stack descending so the
    # loudest rows rise to the top — matches the prior 2-arm diagnostic ordering.
    by_key: dict[tuple[int, str], PairedScore] = {}
    for s in scores:
        key = (s.qid, s.metric)
        if key not in by_key or s.comparison == "stack_aug-stack":
            by_key[key] = s

    def _sat_label(flag: bool) -> str:
        return "Y" if flag else "n"

    rows = list(by_key.values())
    rows.sort(key=lambda s: abs(s.score_stack_aug - s.score_stack), reverse=True)
    for s in rows:
        delta_pdf_stack = s.score_stack_aug - s.score_stack
        delta_median_stack = s.score_median - s.score_stack
        delta_median_pdf = s.score_median - s.score_stack_aug
        base_row = (
            f"| {s.qid} "
            f"| {s.question_type} "
            f"| {s.metric} "
            f"| {_format_score(s.score_stack)} "
            f"| {_format_score(s.score_stack_aug)} "
            f"| {_format_score(s.score_median)} "
            f"| {_format_score(delta_pdf_stack)} "
            f"| {_format_score(delta_median_stack)} "
            f"| {_format_score(delta_median_pdf)} "
            f"| {_sat_label(s.is_saturated_stack)} "
            f"| {_sat_label(s.is_saturated_stack_aug)} "
            f"| {_sat_label(s.is_saturated_median)} "
        )
        if show_confounders:
            stack_marker = _md_escape_cell(s.stack_model_used)
            pdf_marker = _md_escape_cell(s.stack_aug_model_used)
            median_marker = _md_escape_cell(s.median_model_used)
            pdf_tools_marker = "-" if s.stack_aug_tools_fired is None else ("yes" if s.stack_aug_tools_fired else "no")
            median_tools_marker = "-" if s.median_tools_fired is None else ("yes" if s.median_tools_fired else "no")
            lines.append(
                f"{base_row}| {stack_marker} | {pdf_marker} | {median_marker} "
                f"| {pdf_tools_marker} | {median_tools_marker} |"
            )
        else:
            lines.append(f"{base_row}|")
    return lines


def _confounder_summary_lines(scores: list[PairedScore]) -> list[str]:
    """Build the per-arm fallback rate + treatment-activation block from paired scores.

    Aggregates over unique qids (not per-metric rows): a binary question contributes
    one observation, not two (brier + log score share the same arm payload). With three
    arms and three comparisons, the same qid contributes 6 rows for binary; we still
    want one observation per qid.
    """
    by_qid: dict[int, PairedScore] = {}
    for s in scores:
        by_qid.setdefault(s.qid, s)
    sample = list(by_qid.values())
    if not sample:
        return []

    total = len(sample)

    def _count(arm_attr: str, value: str) -> int:
        return sum(1 for s in sample if getattr(s, arm_attr) == value)

    stack_primary = _count("stack_model_used", "primary")
    stack_fallback = _count("stack_model_used", "fallback")
    pdf_primary = _count("stack_aug_model_used", "primary")
    pdf_fallback = _count("stack_aug_model_used", "fallback")
    median_simple = _count("median_model_used", "simple_aggregation")

    n_stack_values = [s.n_forecasters_stack for s in sample if s.n_forecasters_stack is not None]
    n_pdf_values = [s.n_forecasters_stack_aug for s in sample if s.n_forecasters_stack_aug is not None]
    n_median_values = [s.n_forecasters_median for s in sample if s.n_forecasters_median is not None]
    avg_n_stack = sum(n_stack_values) / len(n_stack_values) if n_stack_values else None
    avg_n_pdf = sum(n_pdf_values) / len(n_pdf_values) if n_pdf_values else None
    avg_n_median = sum(n_median_values) / len(n_median_values) if n_median_values else None

    pdf_tools_observations = [s.stack_aug_tools_fired for s in sample if s.stack_aug_tools_fired is not None]
    pdf_tools_fired_count = sum(1 for fired in pdf_tools_observations if fired)
    pdf_tools_total = len(pdf_tools_observations)

    median_tools_observations = [s.median_tools_fired for s in sample if s.median_tools_fired is not None]
    median_tools_fired_count = sum(1 for fired in median_tools_observations if fired)
    median_tools_total = len(median_tools_observations)

    lines: list[str] = ["", "## Confounder summary"]
    lines.append(f"- stack: {stack_primary}/{total} primary, {stack_fallback}/{total} fallback.")
    lines.append(f"- stack_aug: {pdf_primary}/{total} primary, {pdf_fallback}/{total} fallback.")
    if median_simple > 0:
        lines.append(f"- median: {median_simple}/{total} simple_aggregation (deterministic, no fallback).")
    if avg_n_stack is not None and avg_n_pdf is not None and avg_n_median is not None:
        lines.append(
            f"- Average n_forecasters_used: stack {avg_n_stack:.2f}, stack_aug {avg_n_pdf:.2f}, median {avg_n_median:.2f}."
        )
    elif avg_n_stack is not None and avg_n_pdf is not None:
        lines.append(f"- Average n_forecasters_used: stack {avg_n_stack:.2f}, stack_aug {avg_n_pdf:.2f}.")
    if pdf_tools_total > 0:
        line_pdf = (
            f"- Treatment activation: stack_aug fired tools on {pdf_tools_fired_count}/{pdf_tools_total} questions"
        )
        empty = pdf_tools_total - pdf_tools_fired_count
        if empty > 0:
            line_pdf += f" ({empty} had empty cross_model_aggregation — likely structured-block parse failures)"
        lines.append(line_pdf + ".")
    if median_tools_total > 0:
        lines.append(
            f"- Treatment activation: median fired tools on {median_tools_fired_count}/{median_tools_total} "
            "(deterministic aggregation — no LLM tools)."
        )
    return lines


def _per_arm_raw_stats_lines(scores: list[PairedScore]) -> list[str]:
    """Per-arm raw mean/median score per (arm, metric, type), sanity-check section.

    Aggregates over unique qids per (arm, metric, type) — the score values are redundantly
    carried on every PairedScore row regardless of comparison, so dedupe by (qid, metric).
    Reports overall + per-type breakdown.
    """
    # Dedupe to (qid, metric) → score triple. Use any row (pdf-stack by convention).
    by_key: dict[tuple[int, str, str], PairedScore] = {}
    for s in scores:
        by_key[(s.qid, s.metric, s.question_type)] = s

    if not by_key:
        return []

    # Detect 5-arm mode: any score has non-NaN pdf_min1/pdf_min2.
    has_pdf_arms = any(not math.isnan(s.score_pdf_min1) for s in by_key.values())

    # (arm, metric, type) → list[score]. Type "overall" aggregates across types.
    arm_extractors: dict[str, Any] = {
        "stack": lambda s: s.score_stack,
        "stack_aug": lambda s: s.score_stack_aug,
        "median": lambda s: s.score_median,
    }
    if has_pdf_arms:
        arm_extractors["pdf_min1"] = lambda s: s.score_pdf_min1
        arm_extractors["pdf_min2"] = lambda s: s.score_pdf_min2
    bucket: dict[tuple[str, str, str], list[float]] = {}
    for s in by_key.values():
        for arm, extractor in arm_extractors.items():
            score = extractor(s)
            if not math.isnan(score):
                bucket.setdefault((arm, s.metric, s.question_type), []).append(score)
                bucket.setdefault((arm, s.metric, "overall"), []).append(score)

    lines: list[str] = [
        "",
        "## Per-arm raw stats",
        "",
        "| Arm | Metric | Type | N | Mean | Median |",
        "|-----|--------|------|---|------|--------|",
    ]
    for (arm, metric, qtype), values in sorted(bucket.items()):
        n = len(values)
        if n == 0:
            continue
        mean = float(np.mean(values))
        median = float(np.median(values))
        lines.append(f"| {arm} | {metric} | {qtype} | {n} | {_format_score(mean)} | {_format_score(median)} |")
    return lines


def _dual_panel_pdf_section(paired_scores: list[PairedScore], stats: list[PairedStats]) -> list[str]:
    """Render the dual-panel PDF arm comparison section.

    Shows two panels (min_forecasters=1 vs min_forecasters=2) side by side with
    loud caveats about free-model structured-block emit rates.
    """
    lines: list[str] = []
    lines.append("## PDF arm: dual-panel comparison")
    lines.append("")
    lines.append(
        "> **Caveat:** the 'pdf' arm uses strict structured-math-only per forecaster. On the n=88 "
        "free-OpenRouter ensemble, only ~58% of (forecaster, question) pairs emit parseable "
        "structured blocks (mean 2.32 of 4 forecasters survive). We render two views:"
    )
    lines.append(">")
    lines.append(
        "> - **min_forecasters=1** (any-structured): qids where >=1 forecaster emitted structured math. "
        "Single-forecaster qids are NOT aggregation -- they are that single forecaster's structured prediction."
    )
    lines.append(
        "> - **min_forecasters=2** (proper aggregation): qids where >=2 forecasters survived. "
        "These are honest pointwise-median aggregations."
    )
    lines.append(">")
    lines.append(
        "> A re-run on the prod ensemble (Opus 4.6/4.7, GPT-5.5, Gemini 3 Pro, etc.) is expected "
        "to produce structured blocks much more reliably -- these free-model results should be "
        "interpreted as a lower bound on what the pdf arm can do."
    )
    lines.append("")

    # Filter stats to pdf-relevant comparisons
    pdf_min1_comparisons = {"pdf_min1-stack", "pdf_min1-stack_aug", "median-pdf_min1"}
    pdf_min2_comparisons = {"pdf_min2-stack", "pdf_min2-stack_aug", "median-pdf_min2"}

    # Count N for each panel from per-type stats
    overall_stats = [s for s in stats if s.question_type is None]

    # Panel 1: min_forecasters=1
    min1_stats = [s for s in overall_stats if s.comparison in pdf_min1_comparisons]
    if min1_stats:
        n_min1 = min1_stats[0].n if min1_stats else 0
        lines.append(f"### min_forecasters=1 panel (n={n_min1})")
        lines.extend(_stats_table_header())
        for s in sorted(min1_stats, key=lambda x: (x.comparison, x.metric)):
            lines.append(_stats_row(s))
        lines.append("")

    # Panel 2: min_forecasters=2
    min2_stats = [s for s in overall_stats if s.comparison in pdf_min2_comparisons]
    if min2_stats:
        n_min2 = min2_stats[0].n if min2_stats else 0
        lines.append(f"### min_forecasters=2 panel (n={n_min2})")
        lines.extend(_stats_table_header())
        for s in sorted(min2_stats, key=lambda x: (x.comparison, x.metric)):
            lines.append(_stats_row(s))
        lines.append("")

    # Survival-rate table: shows the underlying data shape driving the policy
    # difference between min_forecasters=1 and min_forecasters=2.
    # pdf_min1 and pdf_min2 produce identical predictions on qids where both
    # succeed (>=2 surviving forecasters); they only differ on qids where
    # exactly 1 forecaster survived, which pdf_min1 keeps and pdf_min2 drops.
    lines.append("### Structured-forecaster survival distribution")
    lines.append("")
    lines.append(
        "Shows per-qid structured-forecaster survival counts and which policy "
        "(min=1 vs min=2) produces a prediction. On qids where both succeed, "
        "predictions are identical by construction (same forecasters, same aggregation)."
    )
    lines.append("")

    # Dedupe to one observation per qid using the first metric row.
    by_qid_pdf: dict[int, PairedScore] = {}
    for s in paired_scores:
        if s.qid not in by_qid_pdf:
            by_qid_pdf[s.qid] = s

    # Classify each qid by pdf arm availability.
    n_both_fail = 0
    n_min1_only = 0
    n_both_succeed = 0
    for s in by_qid_pdf.values():
        has_min1 = not math.isnan(s.score_pdf_min1)
        has_min2 = not math.isnan(s.score_pdf_min2)
        if has_min1 and has_min2:
            n_both_succeed += 1
        elif has_min1 and not has_min2:
            n_min1_only += 1
        else:
            n_both_fail += 1

    lines.append("| Survival | n_qids | pdf_min1 | pdf_min2 |")
    lines.append("|----------|--------|----------|----------|")
    lines.append(f"| 0 surviving forecasters | {n_both_fail} | fail | fail |")
    lines.append(f"| 1 surviving forecaster | {n_min1_only} | success (single) | fail |")
    lines.append(f"| 2+ surviving forecasters | {n_both_succeed} | success | success |")
    lines.append("")
    lines.append(
        f"Policy impact: pdf_min1 covers {n_min1_only + n_both_succeed} qids, "
        f"pdf_min2 covers {n_both_succeed} qids. The {n_min1_only} single-forecaster "
        "qids are the policy-relevant difference."
    )
    lines.append("")

    return lines


def render_summary_markdown(stats: list[PairedStats], paired_scores: list[PairedScore], metadata: dict) -> str:
    """Generate the human-readable paired-difference summary.

    Sections:
      - Header (metadata)
      - Overall summary, three subsections (B-A, C-A, C-B)
      - Per-type breakdown, three subsections
      - Per-arm raw stats (single table, all arms × metrics × types)
      - Per-question diagnostic (one row per qid+metric, all three deltas inline)
      - Confounder summary (when payloads present)
      - Caveats
    """
    lines: list[str] = []

    lines.append("# Probabilistic-Tools Ablation — Paired-Difference Summary")
    lines.append("")

    # ``timestamp`` and ``n_questions`` are always populated by the orchestrator's
    # _stage_score caller. Direct subscript surfaces a contract violation.
    timestamp = metadata["timestamp"]
    n_questions = metadata["n_questions"]
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- N questions: {n_questions}")

    if "model_lineup" in metadata:
        lineup = metadata["model_lineup"]
        if isinstance(lineup, (list, tuple)):
            lineup_str = ", ".join(str(m) for m in lineup)
        else:
            lineup_str = str(lineup)
        lines.append(f"- Model lineup: {lineup_str}")
    for key, value in metadata.items():
        if key in {"timestamp", "n_questions", "model_lineup"}:
            continue
        lines.append(f"- {key}: {value}")

    lines.append("")

    overall_stats = [s for s in stats if s.question_type is None]
    per_type_stats = [s for s in stats if s.question_type is not None]

    # Derive active comparisons from the data (supports both 3-arm and 5-arm).
    active_comparisons = sorted({s.comparison for s in stats}) if stats else list(COMPARISONS_3ARM)

    lines.append("## Overall summary")
    if overall_stats:
        for comparison in active_comparisons:
            comp_stats = [s for s in overall_stats if s.comparison == comparison]
            if not comp_stats:
                continue
            lines.append(f"### Comparison: {comparison}")
            lines.extend(_stats_table_header())
            for s in sorted(comp_stats, key=lambda x: x.metric):
                lines.append(_stats_row(s))
            lines.append("")
    else:
        lines.append("(no scores; n=0)")
        lines.append("")

    lines.append("## Per-type breakdown")
    if per_type_stats:
        for comparison in active_comparisons:
            comp_stats = [s for s in per_type_stats if s.comparison == comparison]
            if not comp_stats:
                continue
            lines.append(f"### Comparison: {comparison}")
            lines.extend(_stats_table_header())
            for s in sorted(comp_stats, key=lambda x: (x.question_type, x.metric)):
                lines.append(_stats_row(s))
            lines.append("")
    else:
        lines.append("(no per-type scores; n=0)")
        lines.append("")

    # Saturation-excluded panels: re-aggregate on rows where neither arm in the
    # comparison saturated. Per-comparison filter (a stack-saturated row drops
    # from pdf-stack and median-stack but NOT from median-pdf, etc.). N varies
    # by comparison — that's the point.
    nosat_scores = [s for s in paired_scores if not _saturation_pair_for_comparison(s)]
    if nosat_scores:
        nosat_stats = aggregate_paired(nosat_scores, n_bootstrap=5000, seed=0)
        nosat_overall = [s for s in nosat_stats if s.question_type is None]
        nosat_per_type = [s for s in nosat_stats if s.question_type is not None]

        lines.append("## Overall summary (saturation-excluded)")
        if nosat_overall:
            for comparison in active_comparisons:
                comp_stats = [s for s in nosat_overall if s.comparison == comparison]
                if not comp_stats:
                    continue
                lines.append(f"### Comparison: {comparison}")
                lines.extend(_stats_table_header())
                for s in sorted(comp_stats, key=lambda x: x.metric):
                    lines.append(_stats_row(s))
                lines.append("")
        else:
            lines.append("(no non-saturated scores; n=0)")
            lines.append("")

        lines.append("## Per-type breakdown (saturation-excluded)")
        if nosat_per_type:
            for comparison in active_comparisons:
                comp_stats = [s for s in nosat_per_type if s.comparison == comparison]
                if not comp_stats:
                    continue
                lines.append(f"### Comparison: {comparison}")
                lines.extend(_stats_table_header())
                for s in sorted(comp_stats, key=lambda x: (x.question_type, x.metric)):
                    lines.append(_stats_row(s))
                lines.append("")
        else:
            lines.append("(no non-saturated per-type scores; n=0)")
            lines.append("")

    if paired_scores:
        lines.extend(_per_arm_raw_stats_lines(paired_scores))
        lines.append("")

    # Dual-panel PDF section: only rendered when pdf_min1/pdf_min2 scores are present.
    has_pdf_scores = any(not math.isnan(s.score_pdf_min1) for s in paired_scores)
    if has_pdf_scores:
        lines.extend(_dual_panel_pdf_section(paired_scores, stats))
        lines.append("")

    lines.append("## Per-question diagnostic")
    if paired_scores:
        lines.extend(_per_question_diagnostic_table(paired_scores))
    else:
        lines.append("(no per-question scores; n=0)")
    lines.append("")

    if _has_confounder_data(paired_scores):
        lines.extend(_confounder_summary_lines(paired_scores))
        lines.append("")

    lines.append("## Caveats")
    overall_min_n = min((s.n for s in stats if s.question_type is None), default=0)
    per_type_min_n = min((s.n for s in stats if s.question_type is not None), default=0)
    has_per_type_stats = any(s.question_type is not None for s in stats)

    if stats and overall_min_n < LOW_N_THRESHOLD:
        lines.append(
            f"- **Overall directional only (min overall n={overall_min_n}).** "
            "Bootstrap CIs are wide and significance tests have limited power. "
            "Append more questions before drawing conclusions."
        )
    if has_per_type_stats and per_type_min_n < LOW_N_THRESHOLD:
        lines.append(
            f"- **Per-type directional only (min per-type n={per_type_min_n}).** "
            f"Per-type rows marked with `*` have n<{LOW_N_THRESHOLD}; treat their CIs and p-values as descriptive. "
            "The overall row aggregates across types and is the primary read."
        )
    n_tests = len(stats)
    lines.append(
        f"- **Multiple testing.** k={n_tests} tests reported below; no multiple-testing correction. "
        "Treat individual p-values as descriptive, not as evidence of significance. "
        "To call something significant, look at the overall (metric, all-types) row, not a per-type cell."
    )
    lines.append(
        "- **Percentile-bootstrap CIs** may under-cover the nominal level when n<30, especially with skewed Δ. "
        "Treat width as a lower bound on uncertainty."
    )
    lines.append(
        "- **Selection bias.** Survivors of the leakage screen may be systematically harder or more obscure "
        "than the original tournament population. Δ generalizes to *this* class of questions, not to the "
        "unscreened population. Leakage rate (proportion of qids dropped) is a good proxy for how much "
        "selection bias matters."
    )
    lines.append(
        "- **Saturation flagging.** A score is *saturated* when it sits within ε of the schema's "
        "max-confident-wrong floor (e.g., a numeric log score of -220 means the resolution fell in a "
        "bucket where the prediction had ~min-step PMF mass — there's no way to be more wrong on this "
        "schema). Both-saturated rows are mechanical Δ≈0 draws and don't carry signal; they inflate "
        "sign-test power without informing direction. The `NoSat Δ (n_clean)` column reports the mean "
        "Δ over only the pairs where neither arm in the comparison saturated."
    )
    lines.append(
        "- **(saturation-excluded) panels** filter rows per comparison: a row is dropped if either "
        "arm in that comparison saturated (numeric_log_score near -220, brier near 1.0, etc., where "
        "the schema's max-confident-wrong floor is hit). Use these panels when interpreting numeric "
        "metrics — saturation can dominate means without reflecting calibration differences. N varies "
        "by comparison because saturation patterns differ across arms."
    )
    lines.append("- Higher is better for log scores; lower is better for Brier and CRPS.")
    lines.append("- Δ for comparison X-Y = score_X - score_Y. Bootstrap is paired (resampling qids with replacement).")

    return "\n".join(lines)
