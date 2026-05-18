"""Paired-difference scoring + report generation for the probabilistic-tools ablation benchmark.

For each question we score arm A (tools off) and arm B (tools on) against the same ground
truth, then compute Δ = score_B - score_A. Aggregates by metric and question type, with
paired-bootstrap CI, sign test, and Wilcoxon signed-rank test.

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
    score_a: float
    score_b: float
    delta: float
    higher_is_better: bool
    # Optional confounder fields populated by score_arm_for_qid when stacker
    # payloads are supplied. None on legacy callers; surfaced into the summary
    # report when present so the user can read fallback rate, n-forecasters
    # delta, and treatment activation per question.
    stacker_a_model_used: str | None = None
    stacker_b_model_used: str | None = None
    n_forecasters_a: int | None = None
    n_forecasters_b: int | None = None
    tools_b_fired: bool | None = None
    # Per-arm saturation flags. True iff the score sits within ε of the
    # schema's max-confident-wrong floor (see is_score_saturated). Both-True
    # rows are mechanical Δ≈0 draws and don't carry signal.
    is_saturated_a: bool = False
    is_saturated_b: bool = False


@dataclass
class PairedStats:
    metric: str
    question_type: str | None  # None = overall (all types for this metric)
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
    n_clean: int = 0  # count of pairs where neither arm saturated
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


def _confounders_from_payloads(arm_a_payload: dict | None, arm_b_payload: dict | None) -> dict[str, Any]:
    """Extract confounder fields from cached stacker payloads. Empty dict when neither given."""
    if arm_a_payload is None and arm_b_payload is None:
        return {}
    confounders: dict[str, Any] = {}
    if arm_a_payload is not None:
        confounders["stacker_a_model_used"] = arm_a_payload.get("stacker_model_used")
        n_a = arm_a_payload.get("n_forecasters_used")
        confounders["n_forecasters_a"] = int(n_a) if n_a is not None else None
    if arm_b_payload is not None:
        confounders["stacker_b_model_used"] = arm_b_payload.get("stacker_model_used")
        n_b = arm_b_payload.get("n_forecasters_used")
        confounders["n_forecasters_b"] = int(n_b) if n_b is not None else None
        confounders["tools_b_fired"] = bool(arm_b_payload.get("cross_model_aggregation"))
    return confounders


def _saturation_for_metrics(metrics: dict[str, float], n_mc_options: int | None = None) -> dict[str, bool]:
    return {metric: is_score_saturated(metric, score, n_mc_options) for metric, score in metrics.items()}


def _build_paired_scores(
    qid: int,
    question_type: str,
    metrics_a: dict[str, float],
    metrics_b: dict[str, float],
    saturation_a: dict[str, bool],
    saturation_b: dict[str, bool],
    confounders: dict[str, Any] | None = None,
) -> list[PairedScore]:
    extra_fields = confounders or {}
    paired: list[PairedScore] = []
    for metric, score_a in metrics_a.items():
        score_b = metrics_b[metric]
        paired.append(
            PairedScore(
                qid=qid,
                question_type=question_type,
                metric=metric,
                score_a=score_a,
                score_b=score_b,
                delta=score_b - score_a,
                higher_is_better=METRIC_DIRECTION[metric],
                is_saturated_a=saturation_a[metric],
                is_saturated_b=saturation_b[metric],
                **extra_fields,
            )
        )
    return paired


def score_arm_for_qid(
    report_a: ForecastReport,
    report_b: ForecastReport,
    ground_truth: GroundTruth,
    arm_a_payload: dict | None = None,
    arm_b_payload: dict | None = None,
) -> list[PairedScore]:
    """Score both arms vs ground truth, return one PairedScore per metric.

    Binary: brier + binary_log_score (2 entries).
    MC: mc_log_score (1 entry).
    Numeric: numeric_log_score + crps (2 entries).

    Optional ``arm_{a,b}_payload`` are the cached stacker payloads for each arm
    (the dicts written to ``stacker_outputs/<qid>/arm_{A,B}.json``). When passed,
    each PairedScore is annotated with confounder fields:
    ``stacker_{a,b}_model_used``, ``n_forecasters_{a,b}``, and ``tools_b_fired``
    (True iff arm B's ``cross_model_aggregation`` is non-empty). When omitted,
    those fields stay None.

    Returns empty list if either arm fails to score (canceled, mismatched type, parse error).
    """
    qid = ground_truth.question_id
    resolution = ground_truth.resolution
    confounders = _confounders_from_payloads(arm_a_payload, arm_b_payload)

    if isinstance(report_a, BinaryReport) and isinstance(report_b, BinaryReport):
        if not isinstance(resolution, bool):
            logger.warning(f"Q{qid}: expected bool for binary resolution, got {type(resolution).__name__}")
            return []
        metrics_a = _score_binary_arm(report_a, resolution)
        metrics_b = _score_binary_arm(report_b, resolution)
        sat_a = _saturation_for_metrics(metrics_a)
        sat_b = _saturation_for_metrics(metrics_b)
        return _build_paired_scores(qid, "binary", metrics_a, metrics_b, sat_a, sat_b, confounders)

    if isinstance(report_a, NumericReport) and isinstance(report_b, NumericReport):
        # ``isinstance(True, int)`` is True in Python, so the bool check must precede
        # the int check to avoid silently scoring bool resolutions (which would coerce
        # to 0.0/1.0 and fall into a boundary bucket).
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
        metrics_a = _score_numeric_arm(report_a, resolution)
        metrics_b = _score_numeric_arm(report_b, resolution)
        if metrics_a is None or metrics_b is None:
            return []
        sat_a = _saturation_for_metrics(metrics_a)
        sat_b = _saturation_for_metrics(metrics_b)
        return _build_paired_scores(qid, "numeric", metrics_a, metrics_b, sat_a, sat_b, confounders)

    if isinstance(report_a, MultipleChoiceReport) and isinstance(report_b, MultipleChoiceReport):
        if not isinstance(resolution, str):
            logger.warning(f"Q{qid}: expected str for MC resolution, got {type(resolution).__name__}")
            return []
        metrics_a = _score_mc_arm(report_a, resolution)
        metrics_b = _score_mc_arm(report_b, resolution)
        if metrics_a is None or metrics_b is None:
            return []
        n_mc_options = len(report_a.question.options)
        sat_a = _saturation_for_metrics(metrics_a, n_mc_options=n_mc_options)
        sat_b = _saturation_for_metrics(metrics_b, n_mc_options=n_mc_options)
        return _build_paired_scores(qid, "multiple_choice", metrics_a, metrics_b, sat_a, sat_b, confounders)

    logger.warning(
        f"Q{qid}: unsupported or mismatched report types: A={type(report_a).__name__}, B={type(report_b).__name__}"
    )
    return []


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


def _stats_for_group(
    metric: str,
    question_type: str | None,
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
            "_stats_for_group: dropped %d NaN delta(s) from metric=%s qtype=%s "
            "(propagating NaN through bootstrap would zero out the entire group)",
            n_dropped,
            metric,
            question_type if question_type is not None else "__overall__",
        )

    if not valid_pairs:
        return PairedStats(
            metric=metric,
            question_type=question_type,
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

    clean_deltas = [s.delta for s in valid_group if not (s.is_saturated_a or s.is_saturated_b)]
    n_clean = len(clean_deltas)
    mean_delta_clean = float(np.mean(clean_deltas)) if n_clean > 0 else float("nan")

    return PairedStats(
        metric=metric,
        question_type=question_type,
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


def _group_seed(parent_seed: int, metric: str, question_type: str | None, subgroup: str = "") -> int:
    """Derive a deterministic 32-bit per-group seed from (parent_seed, metric, qtype, subgroup).

    Ensures each (metric, qtype, subgroup) bootstrap draws from a distinct RNG stream so
    per-group CIs are not artificially correlated by sharing top-level state,
    while keeping reproducibility from the parent seed. ``subgroup`` lets the caller
    derive a separate seed for, e.g., the median bootstrap vs the mean bootstrap.

    Uses ``hashlib.sha256`` rather than Python's builtin ``hash()`` because the
    builtin tuple/string hash is randomized per process unless ``PYTHONHASHSEED``
    is set. SHA-256 keeps the seed reproducible across distinct Python processes.
    """
    qtype_str = question_type if question_type is not None else "__overall__"
    key = f"{metric}|{qtype_str}|{subgroup}".encode("utf-8")
    digest_int = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    seed_seq = np.random.SeedSequence([parent_seed, digest_int])
    return int(seed_seq.generate_state(1)[0])


def aggregate_paired(scores: list[PairedScore], n_bootstrap: int = 5000, seed: int = 0) -> list[PairedStats]:
    """Group by (metric, question_type), compute paired stats. Append per-metric overall rows.

    Returns one PairedStats per (metric, question_type) plus one PairedStats per metric with
    question_type=None (the overall aggregation across all types for that metric).

    Each group gets a distinct bootstrap seed derived from `seed` so the per-group
    CIs are not correlated by sharing RNG state. Reproducibility holds: identical
    `seed` reproduces identical PairedStats across all groups.
    """
    if not scores:
        return []

    by_metric: dict[str, list[PairedScore]] = {}
    by_metric_type: dict[tuple[str, str], list[PairedScore]] = {}
    for s in scores:
        by_metric.setdefault(s.metric, []).append(s)
        by_metric_type.setdefault((s.metric, s.question_type), []).append(s)

    out: list[PairedStats] = []
    for (metric, qtype), group in sorted(by_metric_type.items()):
        out.append(
            _stats_for_group(
                metric,
                qtype,
                group,
                n_bootstrap,
                mean_seed=_group_seed(seed, metric, qtype, subgroup="mean"),
                median_seed=_group_seed(seed, metric, qtype, subgroup="median"),
            )
        )

    for metric, group in sorted(by_metric.items()):
        out.append(
            _stats_for_group(
                metric,
                None,
                group,
                n_bootstrap,
                mean_seed=_group_seed(seed, metric, None, subgroup="mean"),
                median_seed=_group_seed(seed, metric, None, subgroup="median"),
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
            "| NoSat Δ (n_clean) | Sign p | Wilcoxon p | B better when Δ |"
        ),
        (
            "|--------|------|---|--------|-------------|----------|---------------"
            "|-------------------|--------|------------|-----------------|"
        ),
    ]


LOW_N_THRESHOLD: int = 30


def _format_clean_mean(stats: PairedStats) -> str:
    if stats.n_clean == 0:
        return f"NaN ({stats.n_clean})"
    return f"{_format_score(stats.mean_delta_clean)} ({stats.n_clean})"


def _stats_row(s: PairedStats) -> str:
    qtype = s.question_type if s.question_type is not None else "overall"
    direction = "Δ > 0" if s.higher_is_better else "Δ < 0"
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
        f"| {direction} |"
    )


def _has_confounder_data(scores: list[PairedScore]) -> bool:
    """True iff at least one PairedScore has a confounder field populated."""
    return any(
        s.stacker_a_model_used is not None
        or s.stacker_b_model_used is not None
        or s.tools_b_fired is not None
        or s.n_forecasters_a is not None
        or s.n_forecasters_b is not None
        for s in scores
    )


def _saturation_label(score: PairedScore) -> str:
    if score.is_saturated_a and score.is_saturated_b:
        return "both"
    if score.is_saturated_a:
        return "a_sat"
    if score.is_saturated_b:
        return "b_sat"
    return "clean"


def _md_escape_cell(value: Any) -> str:
    """Escape pipes in a markdown table cell so embedded ``|`` don't break the table layout.

    Currently only an issue if a confounder field carries a raw model identifier with ``|``
    (e.g., ``"claude-opus-4.7|via-openrouter"``). Production code only writes literal
    "primary"/"fallback" — escaping is defensive against future config changes.
    """
    if value is None:
        return "-"
    return str(value).replace("|", "\\|")


def _per_question_diagnostic_table(scores: list[PairedScore]) -> list[str]:
    show_confounders = _has_confounder_data(scores)
    if show_confounders:
        lines = [
            "| qid | type | metric | A | B | Δ | direction | saturation | A_stacker | B_stacker | B_tools |",
            "|-----|------|--------|---|---|---|-----------|------------|-----------|-----------|---------|",
        ]
    else:
        lines = [
            "| qid | type | metric | A | B | Δ | direction | saturation |",
            "|-----|------|--------|---|---|---|-----------|------------|",
        ]
    sorted_scores = sorted(scores, key=lambda s: abs(s.delta), reverse=True)
    for s in sorted_scores:
        direction = "Δ > 0" if s.higher_is_better else "Δ < 0"
        sat_label = _saturation_label(s)
        base_row = (
            f"| {s.qid} "
            f"| {s.question_type} "
            f"| {s.metric} "
            f"| {_format_score(s.score_a)} "
            f"| {_format_score(s.score_b)} "
            f"| {_format_score(s.delta)} "
            f"| {direction} "
            f"| {sat_label} "
        )
        if show_confounders:
            a_marker = _md_escape_cell(s.stacker_a_model_used)
            b_marker = _md_escape_cell(s.stacker_b_model_used)
            tools_marker = "-" if s.tools_b_fired is None else ("yes" if s.tools_b_fired else "no")
            lines.append(f"{base_row}| {a_marker} | {b_marker} | {tools_marker} |")
        else:
            lines.append(f"{base_row}|")
    return lines


def _confounder_summary_lines(scores: list[PairedScore]) -> list[str]:
    """Build the per-arm fallback rate + treatment-activation block from paired scores.

    Aggregates over unique qids (not per-metric rows): a binary question contributes
    one observation, not two (brier + log score share the same arm payload).
    """
    by_qid: dict[int, PairedScore] = {}
    for s in scores:
        by_qid.setdefault(s.qid, s)
    sample = list(by_qid.values())
    if not sample:
        return []

    total = len(sample)
    a_primary = sum(1 for s in sample if s.stacker_a_model_used == "primary")
    a_fallback = sum(1 for s in sample if s.stacker_a_model_used == "fallback")
    b_primary = sum(1 for s in sample if s.stacker_b_model_used == "primary")
    b_fallback = sum(1 for s in sample if s.stacker_b_model_used == "fallback")

    n_a_values = [s.n_forecasters_a for s in sample if s.n_forecasters_a is not None]
    n_b_values = [s.n_forecasters_b for s in sample if s.n_forecasters_b is not None]
    avg_n_a = sum(n_a_values) / len(n_a_values) if n_a_values else None
    avg_n_b = sum(n_b_values) / len(n_b_values) if n_b_values else None

    tools_observations = [s.tools_b_fired for s in sample if s.tools_b_fired is not None]
    tools_fired = sum(1 for fired in tools_observations if fired)
    tools_total = len(tools_observations)
    tools_empty = tools_total - tools_fired

    lines: list[str] = ["", "## Confounder summary"]
    lines.append(f"- Arm A: {a_primary}/{total} primary, {a_fallback}/{total} fallback.")
    lines.append(f"- Arm B: {b_primary}/{total} primary, {b_fallback}/{total} fallback.")
    if avg_n_a is not None and avg_n_b is not None:
        lines.append(f"- Average n_forecasters_used: arm A {avg_n_a:.2f}, arm B {avg_n_b:.2f}.")
    if tools_total > 0:
        line = f"- Treatment activation: arm B fired tools on {tools_fired}/{tools_total} questions"
        if tools_empty > 0:
            line += f" ({tools_empty} had empty cross_model_aggregation — likely structured-block parse failures)."
        else:
            line += "."
        lines.append(line)
    return lines


def render_summary_markdown(stats: list[PairedStats], paired_scores: list[PairedScore], metadata: dict) -> str:
    """Generate the human-readable paired-difference summary.

    Sections: header (metadata), Overall summary, Per-type breakdown, Per-question diagnostic,
    Caveats.
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

    lines.append("## Overall summary")
    if overall_stats:
        lines.extend(_stats_table_header())
        for s in sorted(overall_stats, key=lambda x: x.metric):
            lines.append(_stats_row(s))
    else:
        lines.append("(no scores; n=0)")
    lines.append("")

    lines.append("## Per-type breakdown")
    if per_type_stats:
        lines.extend(_stats_table_header())
        for s in sorted(per_type_stats, key=lambda x: (x.question_type, x.metric)):
            lines.append(_stats_row(s))
    else:
        lines.append("(no per-type scores; n=0)")
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
        "Δ over only the pairs where neither arm saturated."
    )
    lines.append("- Higher is better for log scores; lower is better for Brier and CRPS.")
    lines.append("- Δ = score_B - score_A. Bootstrap is paired (resampling qids with replacement).")

    return "\n".join(lines)
