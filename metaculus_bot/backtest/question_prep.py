"""Fetch resolved questions from past tournaments and prepare them for backtesting.

Handles question fetching, ground truth extraction, and question sanitization
so bots see clean OPEN-state questions with no resolution info.
"""

import asyncio
import copy
import http.client
import logging
import random
import sys
from dataclasses import dataclass, field
from typing import Any

from forecasting_tools import ApiFilter, MetaculusApi
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    CanceledResolution,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    QuestionState,
)

from metaculus_bot.backtest.scoring import GroundTruth
from metaculus_bot.constants import (
    BACKTEST_DEFAULT_MIN_FORECASTERS,
    BACKTEST_DEFAULT_RESOLVED_AFTER,
    BACKTEST_DEFAULT_TOURNAMENT,
    BACKTEST_OVERFETCH_RATIO,
    FETCH_RETRY_BACKOFFS,
)

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class BacktestQuestionSet:
    questions: list[MetaculusQuestion]
    ground_truths: dict[int, GroundTruth]
    research_cache: dict[int, str] = field(default_factory=dict)
    fetch_metadata: dict[str, Any] = field(default_factory=dict)


async def fetch_resolved_questions(
    total_questions: int,
    resolved_after: str = BACKTEST_DEFAULT_RESOLVED_AFTER,
    tournament: str = BACKTEST_DEFAULT_TOURNAMENT,
    min_forecasters: int = BACKTEST_DEFAULT_MIN_FORECASTERS,
) -> BacktestQuestionSet:
    """Fetch resolved questions from a tournament, extract ground truths, prepare for backtesting.

    Over-fetches by BACKTEST_OVERFETCH_RATIO since the API can't filter by actual_resolution_time.
    Filters locally, then samples down to total_questions.
    """
    from datetime import datetime

    resolved_after_dt = datetime.strptime(resolved_after, "%Y-%m-%d")

    overfetch_count = total_questions * BACKTEST_OVERFETCH_RATIO
    logger.info(
        f"Fetching ~{overfetch_count} resolved questions from tournament '{tournament}' "
        f"(target: {total_questions}, resolved after: {resolved_after})"
    )

    raw_questions = await _fetch_with_retries(
        tournament=tournament,
        count=overfetch_count,
        min_forecasters=min_forecasters,
    )

    logger.info(f"API returned {len(raw_questions)} resolved questions")

    # Local filter: actual_resolution_time > resolved_after, skip canceled/ambiguous
    filtered_questions: list[MetaculusQuestion] = []
    ground_truths: dict[int, GroundTruth] = {}
    skipped_no_resolution_time = 0
    skipped_too_early = 0
    skipped_canceled = 0

    for q in raw_questions:
        qid = q.id_of_question
        if qid is None:
            continue

        if q.actual_resolution_time is None:
            skipped_no_resolution_time += 1
            continue

        if q.actual_resolution_time <= resolved_after_dt:
            skipped_too_early += 1
            continue

        gt = _extract_ground_truth(q)
        if gt is None:
            skipped_canceled += 1
            continue

        filtered_questions.append(q)
        ground_truths[qid] = gt

    logger.info(
        f"After local filtering: {len(filtered_questions)} usable questions "
        f"(skipped: {skipped_no_resolution_time} no resolution time, "
        f"{skipped_too_early} resolved too early, "
        f"{skipped_canceled} canceled/ambiguous)"
    )

    # Sample down to target if we have more
    if len(filtered_questions) > total_questions:
        sampled = random.sample(filtered_questions, total_questions)
        sampled_ids = {q.id_of_question for q in sampled}
        ground_truths = {qid: gt for qid, gt in ground_truths.items() if qid in sampled_ids}
        filtered_questions = sampled
    elif len(filtered_questions) < total_questions:
        logger.warning(
            f"Only {len(filtered_questions)} questions available after filtering "
            f"(requested {total_questions}). Proceeding with what's available."
        )

    # Prepare clean questions (strip resolution, set state=OPEN)
    clean_questions = [_prepare_question_for_backtest(q) for q in filtered_questions]
    random.shuffle(clean_questions)

    type_counts: dict[str, int] = {}
    for q in clean_questions:
        q_type = type(q).__name__
        type_counts[q_type] = type_counts.get(q_type, 0) + 1
    logger.info(f"Final backtest question distribution: {type_counts}")

    return BacktestQuestionSet(
        questions=clean_questions,
        ground_truths=ground_truths,
        fetch_metadata={
            "tournament": tournament,
            "resolved_after": resolved_after,
            "total_fetched": len(raw_questions),
            "total_filtered": len(filtered_questions),
            "total_clean": len(clean_questions),
            "type_distribution": type_counts,
            "skipped_no_resolution_time": skipped_no_resolution_time,
            "skipped_too_early": skipped_too_early,
            "skipped_canceled": skipped_canceled,
        },
    )


async def _fetch_with_retries(
    tournament: str,
    count: int,
    min_forecasters: int,
) -> list[MetaculusQuestion]:
    """Fetch resolved questions with retry logic matching community_benchmark.py."""
    from requests import exceptions as req_exc
    from urllib3 import exceptions as ul3_exc

    api_filter = ApiFilter(
        allowed_statuses=["resolved"],
        allowed_tournaments=[tournament],
        num_forecasters_gte=min_forecasters,
    )

    def _is_retryable_error(err: Exception) -> bool:
        retryables = (
            req_exc.ConnectionError,
            req_exc.Timeout,
            ul3_exc.ProtocolError,
            http.client.RemoteDisconnected,
        )
        if isinstance(err, retryables):
            return True
        msg = str(err).lower()
        return any(tok in msg for tok in ["429", "too many requests", "502", "503", "504", "timeout"])

    attempts = 0
    backoffs = list(FETCH_RETRY_BACKOFFS)
    while True:
        try:
            logger.info(f"Attempt {attempts + 1}: fetching {count} resolved questions from tournament '{tournament}'")
            sys.stdout.flush()
            questions = await MetaculusApi.get_questions_matching_filter(
                api_filter,
                num_questions=count,
                randomly_sample=False,
            )
            if not questions:
                raise RuntimeError("API returned 0 resolved questions")
            return questions
        except Exception as e:
            if attempts < 2 and _is_retryable_error(e):
                sleep_s = backoffs[attempts] if attempts < len(backoffs) else backoffs[-1]
                logger.warning(
                    f"Retryable error fetching resolved questions (attempt {attempts + 1}/3): {e}. "
                    f"Backing off {sleep_s}s."
                )
                sys.stdout.flush()
                await asyncio.sleep(sleep_s)
                attempts += 1
                continue
            logger.error(f"Failed to fetch resolved questions: {e}")
            raise RuntimeError(f"Unable to fetch resolved questions after {attempts + 1} attempts") from e


def _extract_ground_truth(question: MetaculusQuestion) -> GroundTruth | None:
    """Extract ground truth from a resolved question. Returns None for canceled/ambiguous."""
    qid = question.id_of_question
    if qid is None:
        return None

    typed_res = question.typed_resolution
    if typed_res is None:
        return None

    if isinstance(typed_res, CanceledResolution):
        return None

    # Extract community prediction where available.
    # NOTE: Metaculus removed aggregations from their list API, so CP data is no longer
    # available for newly-fetched questions. Resolved tournament questions fetched via the
    # individual question endpoint may still have this data populated.
    if isinstance(question, BinaryQuestion):
        question_type = "binary"
        community_prediction = question.community_prediction_at_access_time
    elif isinstance(question, NumericQuestion):
        question_type = "numeric"
        community_prediction = _extract_numeric_community_cdf(question)
    elif isinstance(question, MultipleChoiceQuestion):
        question_type = "multiple_choice"
        community_prediction = _extract_mc_community_probs(question)
    else:
        logger.warning(f"Q{qid}: unsupported question type {type(question).__name__}")
        return None

    return GroundTruth(
        question_id=qid,
        question_type=question_type,
        resolution=typed_res,
        resolution_string=question.resolution_string or str(typed_res),
        community_prediction=community_prediction,
        actual_resolution_time=question.actual_resolution_time,
        question_text=question.question_text,
        page_url=question.page_url,
    )


def _extract_numeric_community_cdf(question: MetaculusQuestion) -> list[float] | None:
    """Extract numeric community CDF from api_json (same path as scoring_patches.py)."""
    api_json = getattr(question, "api_json", None)
    if not isinstance(api_json, dict):
        return None

    question_obj = api_json.get("question") if isinstance(api_json.get("question"), dict) else api_json
    if not isinstance(question_obj, dict):
        return None

    aggregations = question_obj.get("aggregations")
    if not isinstance(aggregations, dict):
        return None

    rw = aggregations.get("recency_weighted")
    rw_latest = rw.get("latest") if isinstance(rw, dict) else None
    if not isinstance(rw_latest, dict):
        return None

    fv = rw_latest.get("forecast_values")
    if isinstance(fv, list) and len(fv) >= 2:
        try:
            return [float(x) for x in fv]
        except (TypeError, ValueError):
            return None

    return None


def _extract_mc_community_probs(question: MetaculusQuestion) -> list[float] | None:
    """Extract MC community probabilities from api_json (same path as scoring_patches.py)."""
    api_json = getattr(question, "api_json", None)
    if not isinstance(api_json, dict):
        return None

    question_obj = api_json.get("question") if isinstance(api_json.get("question"), dict) else api_json
    if not isinstance(question_obj, dict):
        return None

    options = getattr(question, "options", None)
    if options is None and isinstance(question_obj.get("options"), list):
        options = question_obj.get("options")

    aggregations = question_obj.get("aggregations")
    if not isinstance(aggregations, dict):
        return None

    rw = aggregations.get("recency_weighted")
    rw_latest = rw.get("latest") if isinstance(rw, dict) else None
    if not isinstance(rw_latest, dict):
        return None

    # Try forecast_values first
    fv = rw_latest.get("forecast_values")
    if isinstance(fv, list) and options and isinstance(options, list) and len(fv) == len(options):
        try:
            probs = [float(x) for x in fv]
            total = sum(probs)
            if abs(total - 1.0) > 1e-6 and total > 0:
                probs = [p / total for p in probs]
            return probs
        except (TypeError, ValueError):
            pass

    # Try probability_yes_per_category
    pyc = rw_latest.get("probability_yes_per_category")
    if isinstance(pyc, dict) and options and isinstance(options, list):
        try:
            probs = [float(pyc.get(opt, 0.0)) for opt in options]
            total = sum(probs)
            if abs(total - 1.0) > 1e-6 and total > 0:
                probs = [p / total for p in probs]
            return probs
        except (TypeError, ValueError):
            pass

    return None


def _prepare_question_for_backtest(question: MetaculusQuestion) -> MetaculusQuestion:
    """Deep copy a resolved question and sanitize it to look like an open question.

    Sets state=OPEN, clears resolution fields and background_info.
    """
    clean = copy.deepcopy(question)
    clean.state = QuestionState.OPEN
    clean.resolution_string = None
    clean.actual_resolution_time = None
    clean.background_info = None
    return clean
