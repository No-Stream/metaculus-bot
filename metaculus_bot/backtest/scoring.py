"""Ground truth scoring functions for backtest evaluation.

Handles binary (Brier + log score), numeric (PMF-bucket log score), and
multiple-choice (log score) question types.

Pure scoring primitives (clamp_prob, brier_score, binary_log_score,
resolution_to_bucket_index, numeric_log_score, mc_log_score) live in
metaculus_bot.scoring_common and are re-exported here for backward compatibility.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
from forecasting_tools.data_models.numeric_report import NumericReport
from forecasting_tools.data_models.questions import OutOfBoundsResolution

from metaculus_bot.scoring_common import (
    binary_log_score,
    brier_score,
    mc_log_score,
    numeric_log_score,
)

logger: logging.Logger = logging.getLogger(__name__)

# Resolution type for numeric questions after cancelled resolutions are filtered out
NumericResolutionValue = float | OutOfBoundsResolution


@dataclass
class GroundTruth:
    question_id: int
    question_type: str  # "binary", "numeric", "multiple_choice"
    resolution: bool | float | OutOfBoundsResolution | str
    resolution_string: str
    # NOTE: community_prediction is no longer available for newly-fetched questions.
    # Metaculus removed the aggregations field from their list API, so this will be None
    # for new data. Historical/resolved tournament data may still populate this field.
    community_prediction: float | list[float] | None
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


def numeric_crps_from_report(report: NumericReport, resolution: NumericResolutionValue) -> float | None:
    """Extract CDF from a NumericReport and compute CRPS against resolution.

    Handles OutOfBoundsResolution by mapping to the appropriate bound.
    """

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
        else:
            resolution_float = float(resolution)

        return numeric_crps(x_values, cdf_values, resolution_float)

    except Exception:
        logger.exception("Failed to compute CRPS from report")
        return None


# ---------------------------------------------------------------------------
# Numeric scoring (Metaculus PMF-bucket log score)
# ---------------------------------------------------------------------------


def numeric_log_score_from_report(report: NumericReport, resolution: NumericResolutionValue) -> float | None:
    """Extract CDF from a NumericReport and compute PMF-bucket log score.

    Handles OutOfBoundsResolution by mapping to the appropriate boundary bucket.
    """

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
        else:
            resolution_float = float(resolution)

        return numeric_log_score(
            cdf_values, resolution_float, lower_bound, upper_bound, open_lower, open_upper, zero_point
        )

    except Exception:
        logger.exception("Failed to compute numeric log score from report")
        return None


# ---------------------------------------------------------------------------
# Multiple-choice scoring
# ---------------------------------------------------------------------------


def mc_log_score_from_report(report: MultipleChoiceReport, correct_option: str) -> float | None:
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


def score_report(report: ForecastReport, ground_truth: GroundTruth) -> list[QuestionScore]:
    """Score a single forecast report against ground truth.

    Dispatches to the appropriate scoring function based on report type.
    Returns list of QuestionScore (binary returns both Brier + log_score).
    Also computes community scores using ground_truth.community_prediction.
    """

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


def _compute_community_numeric_log_score(ground_truth: GroundTruth, report: NumericReport) -> float | None:
    """Compute PMF-bucket log score for community CDF against ground truth resolution.

    Community prediction for numeric is stored as raw CDF values. We need bound info
    from the report's question to compute the score. Returns None if community data
    is unavailable (Metaculus removed aggregations from the list API).
    """

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


def _compute_community_mc_log_score(ground_truth: GroundTruth, report: MultipleChoiceReport) -> float | None:
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
