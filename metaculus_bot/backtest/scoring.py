"""Ground truth scoring functions for backtest evaluation.

Handles binary (Brier + log score), numeric (PMF-bucket log score), and
multiple-choice (log score) question types. All functions are pure with no side effects.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

PROB_CLAMP_MIN: float = 1e-4
PROB_CLAMP_MAX: float = 1.0 - 1e-4


@dataclass
class GroundTruth:
    question_id: int
    question_type: str  # "binary", "numeric", "multiple_choice"
    resolution: Any  # bool | float | OutOfBoundsResolution | str
    resolution_string: str
    # NOTE: community_prediction is no longer available for newly-fetched questions.
    # Metaculus removed the aggregations field from their list API, so this will be None
    # for new data. Historical/resolved tournament data may still populate this field.
    community_prediction: Any  # float (binary), list[float] (numeric CDF), list[float] (MC probs)
    actual_resolution_time: datetime | None
    question_text: str
    page_url: str | None = None


@dataclass
class QuestionScore:
    question_id: int
    question_type: str
    bot_score: float
    community_score: float | None
    metric_name: str  # "brier", "log_score", "numeric_log_score", "mc_log_score"


def _clamp_prob(p: float) -> float:
    return max(PROB_CLAMP_MIN, min(PROB_CLAMP_MAX, p))


# ---------------------------------------------------------------------------
# Binary scoring
# ---------------------------------------------------------------------------


def brier_score(predicted_prob: float, outcome: bool) -> float:
    """Brier score: (clamp(p) - y)^2. Lower is better."""
    p = _clamp_prob(predicted_prob)
    y = 1.0 if outcome else 0.0
    return (p - y) ** 2


def binary_log_score(predicted_prob: float, outcome: bool) -> float:
    """Metaculus-style log score for binary questions.

    Formula: 100 * (y * (log2(p) + 1) + (1 - y) * (log2(1 - p) + 1))
    Higher is better. Uniform prediction (0.5) scores 0.
    """
    p = _clamp_prob(predicted_prob)
    y = 1.0 if outcome else 0.0
    return 100.0 * (y * (math.log2(p) + 1.0) + (1.0 - y) * (math.log2(1.0 - p) + 1.0))


# ---------------------------------------------------------------------------
# Numeric scoring (CRPS)
# ---------------------------------------------------------------------------


def numeric_crps(x_values: list[float], cdf_values: list[float], resolution: float) -> float:
    """Continuous Ranked Probability Score using trapezoidal integration.

    CRPS = integral((CDF(x) - H(x - resolution))^2 dx), normalized by range.
    Lower is better.
    """
    xs = np.array(x_values, dtype=float)
    cdfs = np.array(cdf_values, dtype=float)

    if len(xs) < 2 or len(xs) != len(cdfs):
        raise ValueError(f"Invalid CDF: need >= 2 matched x/cdf pairs, got {len(xs)}/{len(cdfs)}")

    heaviside = np.where(xs >= resolution, 1.0, 0.0)
    integrand = (cdfs - heaviside) ** 2

    raw_crps = float(np.trapezoid(integrand, xs))

    total_range = float(xs[-1] - xs[0])
    if total_range <= 0:
        raise ValueError(f"Invalid CDF range: x[0]={xs[0]}, x[-1]={xs[-1]}")

    return raw_crps / total_range


def numeric_crps_from_report(report: Any, resolution: Any) -> float | None:
    """Extract CDF from a NumericReport and compute CRPS against resolution.

    Handles OutOfBoundsResolution by mapping to the appropriate bound.
    """
    from forecasting_tools.data_models.questions import OutOfBoundsResolution

    try:
        cdf_points = report.prediction.cdf
        if not cdf_points or len(cdf_points) < 2:
            logger.warning(f"NumericReport has insufficient CDF points: {len(cdf_points) if cdf_points else 0}")
            return None

        x_values = [float(p.value) for p in cdf_points]
        cdf_values = [float(p.percentile) for p in cdf_points]

        if isinstance(resolution, OutOfBoundsResolution):
            if resolution == OutOfBoundsResolution.ABOVE_UPPER_BOUND:
                resolution_float = x_values[-1]
            elif resolution == OutOfBoundsResolution.BELOW_LOWER_BOUND:
                resolution_float = x_values[0]
            else:
                logger.warning(f"Unknown OutOfBoundsResolution: {resolution}")
                return None
        elif isinstance(resolution, (int, float)):
            resolution_float = float(resolution)
        else:
            logger.warning(f"Unsupported numeric resolution type: {type(resolution)}")
            return None

        return numeric_crps(x_values, cdf_values, resolution_float)

    except Exception:
        logger.exception("Failed to compute CRPS from report")
        return None


# ---------------------------------------------------------------------------
# Numeric scoring (Metaculus PMF-bucket log score)
# ---------------------------------------------------------------------------

BOUNDARY_BASELINE: float = 0.05


def _resolution_to_bucket_index(
    resolution: float,
    lower_bound: float,
    upper_bound: float,
    n_inbound: int,
    zero_point: float | None = None,
) -> int:
    """Map a numeric resolution value to a PMF bucket index.

    Replicates Metaculus backend's unscaled_location_to_bucket_index.
    Returns bucket in [0, n_inbound+1] where 0 = below-lower-bound, n_inbound+1 = above-upper-bound.
    """
    total_range = upper_bound - lower_bound
    if total_range <= 0:
        raise ValueError(f"Invalid bounds: lower={lower_bound}, upper={upper_bound}")

    if zero_point is not None:
        deriv_ratio = (upper_bound - zero_point) / (lower_bound - zero_point)
        scaled_offset = (resolution - lower_bound) * (deriv_ratio - 1) + total_range
        if scaled_offset <= 0:
            return 0
        unscaled = math.log(scaled_offset / total_range) / math.log(deriv_ratio)
    else:
        unscaled = (resolution - lower_bound) / total_range

    if unscaled < 0:
        return 0
    if unscaled > 1:
        return n_inbound + 1
    if unscaled == 1.0:
        return n_inbound
    return max(int(unscaled * n_inbound + 1 - 1e-10), 1)


def numeric_log_score(
    cdf_values: list[float],
    resolution: float,
    lower_bound: float,
    upper_bound: float,
    open_lower_bound: bool,
    open_upper_bound: bool,
    zero_point: float | None = None,
) -> float:
    """Metaculus-style PMF-bucket log score for numeric questions.

    Formula: 50 * ln(pmf[resolution_bucket] / baseline)
    Higher is better. Uniform prediction scores 0.

    CDF is converted to a PMF with len(cdf)+1 buckets (boundary + interior).
    The resolution maps to one bucket; the score is the log of the PMF mass
    in that bucket relative to a uniform baseline.
    """
    n_cdf = len(cdf_values)
    if n_cdf < 2:
        raise ValueError(f"Need at least 2 CDF values, got {n_cdf}")

    n_inbound = n_cdf - 1  # 200 for standard 201-point CDF

    pmf = [cdf_values[0]]
    for i in range(1, n_cdf):
        pmf.append(cdf_values[i] - cdf_values[i - 1])
    pmf.append(1.0 - cdf_values[-1])

    bucket = _resolution_to_bucket_index(resolution, lower_bound, upper_bound, n_inbound, zero_point)

    n_open_bounds = int(open_lower_bound) + int(open_upper_bound)
    if bucket in (0, len(pmf) - 1):
        baseline = BOUNDARY_BASELINE
    else:
        baseline = (1.0 - BOUNDARY_BASELINE * n_open_bounds) / n_inbound

    pmf_value = max(pmf[bucket], 1e-15)
    return 50.0 * math.log(pmf_value / baseline)


def numeric_log_score_from_report(report: Any, resolution: Any) -> float | None:
    """Extract CDF from a NumericReport and compute PMF-bucket log score.

    Handles OutOfBoundsResolution by mapping to the appropriate boundary bucket.
    """
    from forecasting_tools.data_models.questions import OutOfBoundsResolution

    try:
        cdf_points = report.prediction.cdf
        if not cdf_points or len(cdf_points) < 2:
            logger.warning(f"NumericReport has insufficient CDF points: {len(cdf_points) if cdf_points else 0}")
            return None

        cdf_values = [float(p.percentile) for p in cdf_points]

        question = report.question
        lower_bound = float(question.lower_bound)
        upper_bound = float(question.upper_bound)
        open_lower = bool(question.open_lower_bound)
        open_upper = bool(question.open_upper_bound)
        zero_point = float(question.zero_point) if question.zero_point is not None else None

        if isinstance(resolution, OutOfBoundsResolution):
            if resolution == OutOfBoundsResolution.BELOW_LOWER_BOUND:
                resolution_float = lower_bound - 1.0
            elif resolution == OutOfBoundsResolution.ABOVE_UPPER_BOUND:
                resolution_float = upper_bound + 1.0
            else:
                logger.warning(f"Unknown OutOfBoundsResolution: {resolution}")
                return None
        elif isinstance(resolution, (int, float)):
            resolution_float = float(resolution)
        else:
            logger.warning(f"Unsupported numeric resolution type: {type(resolution)}")
            return None

        return numeric_log_score(
            cdf_values, resolution_float, lower_bound, upper_bound, open_lower, open_upper, zero_point
        )

    except Exception:
        logger.exception("Failed to compute numeric log score from report")
        return None


# ---------------------------------------------------------------------------
# Multiple-choice scoring
# ---------------------------------------------------------------------------


def mc_log_score(predicted_probs: list[float], correct_option_index: int) -> float:
    """Metaculus-style log score for multiple-choice questions.

    Formula: 100 * (log2(clamp(p_correct)) / log2(K) + 1)
    Higher is better. Uniform prediction scores 0.
    """
    k = len(predicted_probs)
    if k < 2:
        raise ValueError(f"Need at least 2 options, got {k}")
    if correct_option_index < 0 or correct_option_index >= k:
        raise ValueError(f"correct_option_index {correct_option_index} out of range [0, {k})")

    p_correct = _clamp_prob(predicted_probs[correct_option_index])
    return 100.0 * (math.log2(p_correct) / math.log2(k) + 1.0)


def mc_log_score_from_report(report: Any, correct_option: str) -> float | None:
    """Extract probabilities from a MultipleChoiceReport and compute log score."""
    try:
        options = report.question.options
        if not options:
            logger.warning("MultipleChoiceReport has no question options")
            return None

        option_probs: dict[str, float] = {
            opt.option_name: opt.probability for opt in report.prediction.predicted_options
        }

        predicted_probs = [option_probs.get(opt, 0.0) for opt in options]
        try:
            correct_index = list(options).index(correct_option)
        except ValueError:
            logger.warning(f"Correct option '{correct_option}' not found in question options: {options}")
            return None

        return mc_log_score(predicted_probs, correct_index)

    except Exception:
        logger.exception("Failed to compute MC log score from report")
        return None


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------


def score_report(report: Any, ground_truth: GroundTruth) -> list[QuestionScore]:
    """Score a single forecast report against ground truth.

    Dispatches to the appropriate scoring function based on report type.
    Returns list of QuestionScore (binary returns both Brier + log_score).
    Also computes community scores using ground_truth.community_prediction.
    """
    from forecasting_tools.data_models.binary_report import BinaryReport
    from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
    from forecasting_tools.data_models.numeric_report import NumericReport

    scores: list[QuestionScore] = []
    qid = ground_truth.question_id

    if isinstance(report, BinaryReport):
        outcome = ground_truth.resolution
        if not isinstance(outcome, bool):
            logger.warning(f"Q{qid}: expected bool resolution for binary, got {type(outcome)}")
            return scores

        bot_prob = float(report.prediction)

        # Brier score
        bot_brier = brier_score(bot_prob, outcome)
        community_brier = None
        if isinstance(ground_truth.community_prediction, (int, float)):
            community_brier = brier_score(float(ground_truth.community_prediction), outcome)
        scores.append(QuestionScore(qid, "binary", bot_brier, community_brier, "brier"))

        # Log score
        bot_log = binary_log_score(bot_prob, outcome)
        community_log = None
        if isinstance(ground_truth.community_prediction, (int, float)):
            community_log = binary_log_score(float(ground_truth.community_prediction), outcome)
        scores.append(QuestionScore(qid, "binary", bot_log, community_log, "log_score"))

    elif isinstance(report, NumericReport):
        resolution = ground_truth.resolution
        bot_log = numeric_log_score_from_report(report, resolution)
        if bot_log is not None:
            community_log = _compute_community_numeric_log_score(ground_truth, report)
            scores.append(QuestionScore(qid, "numeric", bot_log, community_log, "numeric_log_score"))

    elif isinstance(report, MultipleChoiceReport):
        correct_option = ground_truth.resolution
        if not isinstance(correct_option, str):
            logger.warning(f"Q{qid}: expected str resolution for MC, got {type(correct_option)}")
            return scores

        bot_mc_log = mc_log_score_from_report(report, correct_option)
        if bot_mc_log is not None:
            community_mc_log = _compute_community_mc_log_score(ground_truth, report)
            scores.append(QuestionScore(qid, "multiple_choice", bot_mc_log, community_mc_log, "mc_log_score"))

    else:
        logger.warning(f"Q{qid}: unsupported report type {type(report).__name__}")

    return scores


def _compute_community_numeric_log_score(ground_truth: GroundTruth, report: Any) -> float | None:
    """Compute PMF-bucket log score for community CDF against ground truth resolution.

    Community prediction for numeric is stored as raw CDF values. We need bound info
    from the report's question to compute the score. Returns None if community data
    is unavailable (Metaculus removed aggregations from the list API).
    """
    from forecasting_tools.data_models.questions import OutOfBoundsResolution

    community_cdf = ground_truth.community_prediction
    if not isinstance(community_cdf, list) or len(community_cdf) < 2:
        return None

    resolution = ground_truth.resolution

    try:
        question = report.question
        lower_bound = float(question.lower_bound)
        upper_bound = float(question.upper_bound)
        open_lower = bool(question.open_lower_bound)
        open_upper = bool(question.open_upper_bound)
        zero_point = float(question.zero_point) if question.zero_point is not None else None
    except (AttributeError, TypeError):
        return None

    if isinstance(resolution, OutOfBoundsResolution):
        if resolution == OutOfBoundsResolution.BELOW_LOWER_BOUND:
            resolution_float = lower_bound - 1.0
        elif resolution == OutOfBoundsResolution.ABOVE_UPPER_BOUND:
            resolution_float = upper_bound + 1.0
        else:
            return None
    elif isinstance(resolution, (int, float)):
        resolution_float = float(resolution)
    else:
        return None

    try:
        cdf_values = [float(v) for v in community_cdf]
        return numeric_log_score(
            cdf_values, resolution_float, lower_bound, upper_bound, open_lower, open_upper, zero_point
        )
    except (ValueError, TypeError):
        return None


def _compute_community_mc_log_score(ground_truth: GroundTruth, report: Any) -> float | None:
    """Compute MC log score for community prediction against ground truth."""
    community_probs = ground_truth.community_prediction
    if not isinstance(community_probs, list) or len(community_probs) < 2:
        return None

    correct_option = ground_truth.resolution
    if not isinstance(correct_option, str):
        return None

    try:
        options = report.question.options
        correct_index = list(options).index(correct_option)
        return mc_log_score(community_probs, correct_index)
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to compute community MC log score: {e}")
        return None
