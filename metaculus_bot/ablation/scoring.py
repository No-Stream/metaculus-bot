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
from typing import Any, cast

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
from metaculus_bot.bootstrap import bootstrap_means, bootstrap_medians
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

# 6-arm comparisons: the full 5-arm set plus mean-vs-everything. ``mean-median``
# is the headline contrast (the two deterministic baselines head to head); the
# remaining four pit the deterministic mean against each LLM/pdf arm. The arm
# order in score_arm_for_qid for this mode is
# [stack, stack_aug, pdf_min1, pdf_min2, median, mean].
COMPARISONS_6ARM: tuple[str, ...] = (
    *COMPARISONS_5ARM,
    "mean-stack",
    "mean-stack_aug",
    "mean-pdf_min1",
    "mean-pdf_min2",
    "mean-median",
)

# Every arm label a PairedScore row carries a score for, in the canonical order the
# ``arm_reports`` list must arrive in. 3- and 5-arm callers omit the trailing arms;
# those land as None and are treated as absent.
ALL_ARM_LABELS: tuple[str, ...] = ("stack", "stack_aug", "pdf_min1", "pdf_min2", "median", "mean")

# The arm-count modes ``score_arm_for_qid`` accepts, and the comparison set each one
# emits. Keyed by len(arm_reports) so the mode is read once, not re-derived per branch.
_ARM_LABELS_BY_COUNT: dict[int, tuple[str, ...]] = {
    3: ("stack", "stack_aug", "median"),
    5: ("stack", "stack_aug", "pdf_min1", "pdf_min2", "median"),
    6: ALL_ARM_LABELS,
}
_COMPARISONS_BY_ARM_COUNT: dict[int, tuple[str, ...]] = {
    3: COMPARISONS_3ARM,
    5: COMPARISONS_5ARM,
    6: COMPARISONS_6ARM,
}

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

    boot_medians = bootstrap_medians(arr, n_bootstrap=n_bootstrap, seed=seed)

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
    # mean arm score (deterministic pointwise mean baseline; NaN unless 6-arm mode).
    score_mean: float = float("nan")
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
    is_saturated_mean: bool = False


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


def _index_arm_reports(
    arm_reports: list[tuple[str, ForecastReport | None, dict | None]],
) -> tuple[dict[str, ForecastReport | None], dict[str, dict | None]]:
    """Key the caller's positional arm tuples by label, padding absent arms with None.

    The incoming list is positional, so a mis-ordered list would silently mislabel
    every score; both the count and the order are validated here. Returns reports and
    payloads covering every label in ``ALL_ARM_LABELS`` so downstream code can read any
    arm without re-deriving which arm-count mode the caller used.
    """
    expected_labels = _ARM_LABELS_BY_COUNT.get(len(arm_reports))
    if expected_labels is None:
        raise ValueError(f"score_arm_for_qid expects 3, 5, or 6 arm tuples, got {len(arm_reports)}")
    actual_labels = [label for label, _report, _payload in arm_reports]
    if actual_labels != list(expected_labels):
        raise ValueError(f"score_arm_for_qid expects arms in order {list(expected_labels)}, got {actual_labels}")

    supplied = {label: (report, payload) for label, report, payload in arm_reports}
    reports = {label: supplied.get(label, (None, None))[0] for label in ALL_ARM_LABELS}
    payloads = {label: supplied.get(label, (None, None))[1] for label in ALL_ARM_LABELS}
    return reports, payloads


def _report_type_of(report: ForecastReport) -> type[ForecastReport] | None:
    """Concrete ft report class backing ``report``, or None when unsupported.

    isinstance rather than ``type()`` comparison because ``MagicMock(spec=X)`` passes
    ``isinstance(_, X)`` but ``type()`` returns a unique MagicMock subclass.
    """
    for report_type in (BinaryReport, NumericReport, MultipleChoiceReport):
        if isinstance(report, report_type):
            return report_type
    return None


def _question_type_for_resolution(report_type: type[ForecastReport], resolution: Any, qid: int) -> str | None:
    """Question-type label for ``report_type``, or None (after a WARN) if the GT can't score.

    The resolution has to match the report family: a bool resolution reaching the
    numeric scorer is an upstream routing bug, not a scoreable question.
    """
    if report_type is BinaryReport:
        if not isinstance(resolution, bool):
            logger.warning(f"Q{qid}: expected bool for binary resolution, got {type(resolution).__name__}")
            return None
        return "binary"
    if report_type is NumericReport:
        if isinstance(resolution, bool):
            logger.warning(
                f"Q{qid}: bool resolution {resolution!r} routed to numeric scorer; skipping. "
                "This indicates an upstream type-routing bug."
            )
            return None
        if not isinstance(resolution, (int, float, OutOfBoundsResolution)):
            logger.warning(
                f"Q{qid}: expected numeric or OutOfBoundsResolution for numeric question, "
                f"got {type(resolution).__name__}"
            )
            return None
        return "numeric"
    if not isinstance(resolution, str):
        logger.warning(f"Q{qid}: expected str for MC resolution, got {type(resolution).__name__}")
        return None
    return "multiple_choice"


def _score_one_arm(report: ForecastReport, *, question_type: str, resolution: Any) -> dict[str, float] | None:
    """Metrics for one arm's report, or None when the report can't be scored."""
    if question_type == "binary":
        return _score_binary_arm(cast(BinaryReport, report), cast(bool, resolution))
    if question_type == "numeric":
        return _score_numeric_arm(cast(NumericReport, report), resolution)
    return _score_mc_arm(cast(MultipleChoiceReport, report), cast(str, resolution))


def _score_present_arms(
    reports: dict[str, ForecastReport | None],
    *,
    question_type: str,
    resolution: Any,
    n_mc_options: int | None,
) -> tuple[dict[str, dict[str, float] | None], dict[str, dict[str, bool] | None]]:
    """Score every canonical arm; unscoreable arms map to None in both dicts.

    An arm that is absent OR whose report fails to score (e.g. a numeric CDF the
    scorer rejects) lands as None and is therefore treated as absent by the
    comparison filter downstream.
    """
    arm_metrics: dict[str, dict[str, float] | None] = {}
    arm_saturation: dict[str, dict[str, bool] | None] = {}
    for label in ALL_ARM_LABELS:
        report = reports[label]
        metrics = (
            _score_one_arm(report, question_type=question_type, resolution=resolution) if report is not None else None
        )
        arm_metrics[label] = metrics
        arm_saturation[label] = (
            _saturation_for_metrics(metrics, n_mc_options=n_mc_options) if metrics is not None else None
        )
    return arm_metrics, arm_saturation


def _collect_confounders(payloads: dict[str, dict | None]) -> dict[str, Any]:
    """Per-arm confounder fields from the cached stacker payloads.

    The mean arm is a deterministic baseline with no dedicated PairedScore confounder
    fields, so its payload is intentionally not surfaced.
    """
    confounders: dict[str, Any] = {}
    for label in ("stack", "stack_aug", "pdf_min1", "pdf_min2", "median"):
        confounders.update(_confounders_for_arm(payloads[label], label))
    return confounders


def _metric_or_nan(metrics: dict[str, float] | None, metric: str) -> float:
    return metrics[metric] if metrics else float("nan")


def _saturated_or_false(saturation: dict[str, bool] | None, metric: str) -> bool:
    return saturation[metric] if saturation else False


def _build_paired_rows(
    *,
    qid: int,
    question_type: str,
    active_comparisons: list[str],
    arm_metrics: dict[str, dict[str, float] | None],
    arm_saturation: dict[str, dict[str, bool] | None],
    confounders: dict[str, Any],
) -> list[PairedScore]:
    """One PairedScore per (metric, active comparison).

    Every row carries all six arm scores (NaN when the arm is absent) so per-arm raw
    stats and the per-question diagnostic can be derived from any subset of rows. The
    canonical metric set comes off the first scored arm; every arm of a given question
    type produces the same metric keys.
    """
    canonical_metrics = next(metrics for metrics in arm_metrics.values() if metrics is not None)

    paired: list[PairedScore] = []
    for metric in canonical_metrics:
        score = {label: _metric_or_nan(arm_metrics[label], metric) for label in ALL_ARM_LABELS}
        saturated = {label: _saturated_or_false(arm_saturation[label], metric) for label in ALL_ARM_LABELS}
        for comparison in active_comparisons:
            left, right = comparison.split("-", 1)
            paired.append(
                PairedScore(
                    qid=qid,
                    question_type=question_type,
                    metric=metric,
                    comparison=comparison,
                    score_stack=score["stack"],
                    score_stack_aug=score["stack_aug"],
                    score_median=score["median"],
                    score_pdf_min1=score["pdf_min1"],
                    score_pdf_min2=score["pdf_min2"],
                    score_mean=score["mean"],
                    delta=score[left] - score[right],
                    higher_is_better=METRIC_DIRECTION[metric],
                    is_saturated_stack=saturated["stack"],
                    is_saturated_stack_aug=saturated["stack_aug"],
                    is_saturated_pdf_min1=saturated["pdf_min1"],
                    is_saturated_pdf_min2=saturated["pdf_min2"],
                    is_saturated_median=saturated["median"],
                    is_saturated_mean=saturated["mean"],
                    **confounders,
                )
            )
    return paired


def score_arm_for_qid(
    arm_reports: list[tuple[str, ForecastReport | None, dict | None]],
    ground_truth: GroundTruth,
) -> list[PairedScore]:
    """Score arms vs ground truth, return PairedScores for pairwise comparisons where both arms are present.

    ``arm_reports`` is a 3-, 5-, or 6-element list:

    **3-arm** (backward compat): labels in order [stack, stack_aug, median].
      - Up to 3 comparisons per metric.

    **5-arm**: labels in order [stack, stack_aug, pdf_min1, pdf_min2, median].
      - Up to 10 comparisons per metric.

    **6-arm**: labels in order [stack, stack_aug, pdf_min1, pdf_min2, median, mean].
      - Up to 15 comparisons per metric (the 10 5-arm pairs + 5 mean comparisons).

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
    reports, payloads = _index_arm_reports(arm_reports)

    present_reports = [report for report in (reports[label] for label in ALL_ARM_LABELS) if report is not None]
    if len(present_reports) < 2:
        return []

    qid = ground_truth.question_id
    resolution = ground_truth.resolution
    first_report = present_reports[0]

    expected_type = _report_type_of(first_report)
    if expected_type is None:
        logger.warning(f"Q{qid}: unsupported report type: {type(first_report).__name__}")
        return []
    if any(not isinstance(report, expected_type) for report in present_reports[1:]):
        logger.warning(f"Q{qid}: mismatched report types among present arms")
        return []

    question_type = _question_type_for_resolution(expected_type, resolution, qid)
    if question_type is None:
        return []

    n_mc_options = (
        len(cast(MultipleChoiceReport, first_report).question.options) if question_type == "multiple_choice" else None
    )
    arm_metrics, arm_saturation = _score_present_arms(
        reports,
        question_type=question_type,
        resolution=resolution,
        n_mc_options=n_mc_options,
    )

    scored_arms = {label for label, metrics in arm_metrics.items() if metrics is not None}
    if len(scored_arms) < 2:
        return []

    active_comparisons = [
        comparison
        for comparison in _COMPARISONS_BY_ARM_COUNT[len(arm_reports)]
        if all(arm in scored_arms for arm in comparison.split("-", 1))
    ]
    if not active_comparisons:
        return []

    return _build_paired_rows(
        qid=qid,
        question_type=question_type,
        active_comparisons=active_comparisons,
        arm_metrics=arm_metrics,
        arm_saturation=arm_saturation,
        confounders=_collect_confounders(payloads),
    )


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

    boot_means = bootstrap_means(arr, n_bootstrap=n_bootstrap, seed=seed)

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
    if arm == "mean":
        return score.is_saturated_mean
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
    *,
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
    *,
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
    key = f"{metric}|{qtype_str}|{comparison}|{subgroup}".encode()
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
                n_bootstrap=n_bootstrap,
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
                n_bootstrap=n_bootstrap,
                mean_seed=_group_seed(seed, metric, None, comparison, subgroup="mean"),
                median_seed=_group_seed(seed, metric, None, comparison, subgroup="median"),
            )
        )

    return out
