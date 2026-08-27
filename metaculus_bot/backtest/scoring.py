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
            cdf_values,
            resolution_float,
            lower_bound,
            upper_bound,
            open_lower_bound=open_lower,
            open_upper_bound=open_upper,
            zero_point=zero_point,
        )

    except Exception:
        logger.exception("Failed to compute numeric log score from report")
        return None


# ---------------------------------------------------------------------------
# Multiple-choice scoring
# ---------------------------------------------------------------------------


def _canonicalize_mc_option(s: str) -> str:
    """Best-effort normalize numeric MC options. '3.0' -> '3', ' 3 ' -> '3'."""
    stripped = s.strip()
    # Try to coerce to int via float (handles '3.0', '3', ' 3 ', '+3').
    try:
        f = float(stripped)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except (ValueError, TypeError):
        return stripped


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

        # A missing option means the ballot doesn't cover the question, so there is no
        # probability to score. The old `.get(opt, 0.0)` scored it as "we assigned the
        # option zero" — and if that option is the one that resolved, the log score is the
        # worst value the scale has. Unreachable today (upstream validation forces the full
        # option set), so this returns None the way every other unscoreable shape here does.
        missing_options = [opt for opt in options if opt not in option_probs]
        if missing_options:
            logger.warning(
                f"MultipleChoiceReport ballot is missing question options {missing_options}; cannot score this question"
            )
            return None

        predicted_probs = [option_probs[opt] for opt in options]
        options_list = list(options)
        try:
            correct_index = options_list.index(correct_option)
        except ValueError:
            # Try canonical-form fallback before reporting failure. Resolution strings
            # sometimes come through as float-formatted ('3.0') while options are
            # integer-formatted ('3'); canonicalize both sides and retry.
            canonical_correct = _canonicalize_mc_option(correct_option)
            canonical_options = [_canonicalize_mc_option(o) for o in options_list]
            try:
                correct_index = canonical_options.index(canonical_correct)
            except ValueError:
                logger.warning(
                    f"Correct option '{correct_option}' (canonical {canonical_correct!r}) not found in "
                    f"question options: {options_list} (canonical {canonical_options!r})"
                )
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

    if isinstance(report, BinaryReport):
        return _score_binary_report(report, ground_truth)
    if isinstance(report, NumericReport):
        return _score_numeric_report(report, ground_truth)
    if isinstance(report, MultipleChoiceReport):
        return _score_mc_report(report, ground_truth)
    logger.warning(f"Q{ground_truth.question_id}: unsupported report type {type(report).__name__}")
    return []


def _score_binary_report(report: BinaryReport, ground_truth: GroundTruth) -> list[QuestionScore]:
    """Brier + log score for a binary report, each paired with the community's own score."""
    qid = ground_truth.question_id
    outcome = ground_truth.resolution
    if not isinstance(outcome, bool):
        logger.warning(f"Q{qid}: expected bool resolution for binary, got {type(outcome)}")
        return []

    bot_prob = float(report.prediction)
    community_prob = (
        float(ground_truth.community_prediction)
        if isinstance(ground_truth.community_prediction, (int, float))
        else None
    )
    return [
        QuestionScore(
            qid,
            "binary",
            brier_score(bot_prob, outcome),
            brier_score(community_prob, outcome) if community_prob is not None else None,
            "brier",
        ),
        QuestionScore(
            qid,
            "binary",
            binary_log_score(bot_prob, outcome),
            binary_log_score(community_prob, outcome) if community_prob is not None else None,
            "log_score",
        ),
    ]


def _score_numeric_report(report: NumericReport, ground_truth: GroundTruth) -> list[QuestionScore]:
    """PMF-bucket log score for a numeric report; empty when the resolution isn't scoreable."""
    qid = ground_truth.question_id
    resolution = ground_truth.resolution
    # ``bool`` is a subclass of ``int`` in Python, so it would slip through the
    # ``isinstance(..., int)`` check and be silently coerced to 1.0/0.0; exclude it
    # explicitly so a boolean resolution is reported as the type error it is.
    if isinstance(resolution, bool) or not isinstance(resolution, (float, int, OutOfBoundsResolution)):
        logger.warning(f"Q{qid}: expected numeric resolution for numeric, got {type(resolution)}")
        return []
    resolution_value: NumericResolutionValue = (
        resolution if isinstance(resolution, OutOfBoundsResolution) else float(resolution)
    )
    bot_log = numeric_log_score_from_report(report, resolution_value)
    if bot_log is None:
        return []
    community_log = _compute_community_numeric_log_score(ground_truth, report)
    return [QuestionScore(qid, "numeric", bot_log, community_log, "numeric_log_score")]


def _score_mc_report(report: MultipleChoiceReport, ground_truth: GroundTruth) -> list[QuestionScore]:
    """MC log score for a multiple-choice report; empty when the resolution isn't scoreable."""
    qid = ground_truth.question_id
    correct_option = ground_truth.resolution
    if not isinstance(correct_option, str):
        logger.warning(f"Q{qid}: expected str resolution for MC, got {type(correct_option)}")
        return []

    bot_mc_log = mc_log_score_from_report(report, correct_option)
    if bot_mc_log is None:
        return []
    community_mc_log = _compute_community_mc_log_score(ground_truth, report)
    return [QuestionScore(qid, "multiple_choice", bot_mc_log, community_mc_log, "mc_log_score")]


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
            cdf_values,
            resolution_float,
            lower_bound,
            upper_bound,
            open_lower_bound=open_lower,
            open_upper_bound=open_upper,
            zero_point=zero_point,
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
