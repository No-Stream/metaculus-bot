"""Data collection from the Metaculus API.

Fetches resolved questions and bot comments, matches them, parses per-model
predictions, computes scores, and returns a structured dataset.
"""

import json
import logging
import os
import time

import requests
from dotenv import load_dotenv

from metaculus_bot.performance_analysis.parsing import (
    parse_per_model_forecasts,
    parse_per_model_numeric_percentiles,
    parse_resolution,
    parse_stacked_marker,
)
from metaculus_bot.performance_analysis.scoring import binary_log_score, brier_score, mc_log_score, numeric_log_score

logger: logging.Logger = logging.getLogger(__name__)

load_dotenv()

BASE_URL = "https://www.metaculus.com/api"
DEFAULT_BOT_USER_ID = 275109
DEFAULT_TOURNAMENT = "spring-aib-2026"

FETCH_DELAY_SECS: float = 0.5
MAX_RETRIES: int = 3
RETRY_BACKOFF_SECS: float = 5.0
PAGE_SIZE: int = 100


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def _make_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Token {token}"}


def _api_get(path: str, token: str, params: dict | None = None) -> dict:
    url = f"{BASE_URL}{path}"
    headers = _make_headers(token)
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code == 429 and attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF_SECS * (attempt + 1)
            logger.warning(f"Rate limited (429), retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Exhausted retries on rate-limited request")


# ---------------------------------------------------------------------------
# Fetch functions
# ---------------------------------------------------------------------------


def fetch_resolved_questions(tournament: str, token: str) -> list[dict]:
    """Fetch all resolved questions from a tournament.

    Fetches post IDs from the list API, then fetches each individually for full
    details (resolution, my_forecasts, scaling, etc.). Returns list of raw API
    response dicts.
    """
    post_ids = _fetch_resolved_post_ids(tournament, token)
    logger.info(f"Found {len(post_ids)} resolved posts in tournament '{tournament}'")

    posts: list[dict] = []
    for i, pid in enumerate(post_ids):
        logger.info(f"  [{i + 1}/{len(post_ids)}] Fetching post {pid}...")
        post_data = _api_get(f"/posts/{pid}/", token)
        posts.append(post_data)
        if i < len(post_ids) - 1:
            time.sleep(FETCH_DELAY_SECS)

    return posts


def _fetch_resolved_post_ids(tournament: str, token: str) -> list[int]:
    """Paginate the tournament posts list to get all resolved post IDs."""
    post_ids: list[int] = []
    offset = 0
    while True:
        data = _api_get(
            "/posts/",
            token,
            params={
                "tournaments": tournament,
                "statuses": "resolved",
                "limit": PAGE_SIZE,
                "offset": offset,
            },
        )
        results = data.get("results", [])
        for post in results:
            post_ids.append(post["id"])
        logger.info(f"Fetched post listing page: {offset=}, got {len(results)} posts")
        if not results or data.get("next") is None:
            break
        offset += PAGE_SIZE
        time.sleep(FETCH_DELAY_SECS)
    return post_ids


def fetch_bot_comments(author_id: int, token: str) -> list[dict]:
    """Fetch all comments by a given author, paginated. Returns list of raw comment dicts."""
    comments: list[dict] = []
    offset = 0
    while True:
        data = _api_get(
            "/comments/",
            token,
            params={"author": author_id, "limit": PAGE_SIZE, "offset": offset},
        )
        results = data.get("results", [])
        comments.extend(results)
        logger.info(f"Fetched comments page: {offset=}, got {len(results)} comments")
        if not results or data.get("next") is None:
            break
        offset += PAGE_SIZE
        time.sleep(FETCH_DELAY_SECS)
    return comments


# ---------------------------------------------------------------------------
# Comment lookup
# ---------------------------------------------------------------------------


def _build_comment_lookup(comments: list[dict]) -> dict[int, dict]:
    """Build post_id -> comment mapping. Takes the most recent comment per post."""
    lookup: dict[int, dict] = {}
    for c in comments:
        post_id = c.get("on_post")
        if post_id is None:
            continue
        existing = lookup.get(post_id)
        if existing is None or c["id"] > existing["id"]:
            lookup[post_id] = c
    return lookup


# ---------------------------------------------------------------------------
# Record processing
# ---------------------------------------------------------------------------


def _process_post(post_data: dict, comment_lookup: dict[int, dict]) -> list[dict]:
    """Process a single post into one or more question records."""
    post_id = post_data["id"]
    title = post_data.get("title", "")

    if title.startswith("[PRACTICE]"):
        logger.info(f"  Skipping PRACTICE post {post_id}: {title[:60]}")
        return []

    group = post_data.get("group_of_questions")
    if group is not None:
        questions = group.get("questions", [])
    else:
        q = post_data.get("question")
        questions = [q] if q is not None else []

    if not questions:
        logger.warning(f"  Post {post_id} has no question data")
        return []

    comment = comment_lookup.get(post_id)
    comment_text = comment.get("text") or comment.get("comment_text") if comment else None
    comment_id = comment["id"] if comment else None
    per_model = parse_per_model_forecasts(comment_text) if comment_text else {}
    per_model_numeric_percentiles = parse_per_model_numeric_percentiles(comment_text) if comment_text else {}
    was_stacked = parse_stacked_marker(comment_text) if comment_text else None

    # Cross-signal sanity: a stacked comment should expose at least one
    # per-model entry via the rationale parsers. If the marker says stacked
    # but we recovered nothing, the producer-side delimiter likely drifted —
    # worth surfacing during triage, but only as DEBUG since legitimate cases
    # (trimmed comments, binary/MC stacked Qs with no percentile restatement)
    # look the same.
    if was_stacked is True and not per_model_numeric_percentiles and not per_model:
        logger.debug(
            f"Stacked comment yielded no per-model entries: post_id={post_id}, "
            f"comment_len={len(comment_text) if comment_text else 0}"
        )

    records: list[dict] = []
    for q in questions:
        record = _process_single_question(
            post_id,
            title,
            q,
            comment_text,
            comment_id,
            per_model,
            per_model_numeric_percentiles,
            was_stacked,
            post_data,
        )
        if record is not None:
            records.append(record)
    return records


def _process_single_question(
    post_id: int,
    title: str,
    q: dict,
    comment_text: str | None,
    comment_id: int | None,
    per_model: dict[str, str],
    per_model_numeric_percentiles: dict[str, list[tuple[float, float]]],
    was_stacked: bool | None,
    post_data: dict,
) -> dict | None:
    """Process a single question dict into a scored record."""
    question_id = q.get("id")
    q_type = q.get("type", "")
    resolution_raw = q.get("resolution")

    if resolution_raw is None:
        logger.info(f"  Skipping Q{question_id} (no resolution)")
        return None

    resolution_parsed, should_skip = parse_resolution(str(resolution_raw), q_type)
    if should_skip:
        logger.info(f"  Skipping Q{question_id}: resolution={resolution_raw}")
        return None

    my_forecasts = q.get("my_forecasts")
    forecast_values = None
    prob_yes = None
    metaculus_scores: dict | None = None
    if my_forecasts and my_forecasts.get("latest"):
        forecast_values = my_forecasts["latest"].get("forecast_values")
        if q_type == "binary" and forecast_values and len(forecast_values) >= 2:
            prob_yes = forecast_values[1]
        raw_sd = my_forecasts.get("score_data")
        if raw_sd:
            metaculus_scores = {
                "peer_score": raw_sd.get("peer_score"),
                "spot_peer_score": raw_sd.get("spot_peer_score"),
                "baseline_score": raw_sd.get("baseline_score"),
                "spot_baseline_score": raw_sd.get("spot_baseline_score"),
                "coverage": raw_sd.get("coverage"),
                "weighted_coverage": raw_sd.get("weighted_coverage"),
                "relative_legacy_score": raw_sd.get("relative_legacy_score"),
            }

    if forecast_values is None:
        logger.info(f"  Skipping Q{question_id}: no forecast from us")
        return None

    scaling = q.get("scaling") or {}
    open_lower = q.get("open_lower_bound", False)
    open_upper = q.get("open_upper_bound", False)
    options = q.get("options")

    category = None
    projects = post_data.get("projects", {})
    category_list = projects.get("category", [])
    if category_list and isinstance(category_list, list) and len(category_list) > 0:
        category = category_list[0].get("name")

    q_title = q.get("title") or title

    record = {
        "post_id": post_id,
        "question_id": question_id,
        "title": q_title,
        "type": q_type,
        "resolution_raw": str(resolution_raw),
        "resolution_parsed": resolution_parsed,
        "our_forecast_values": forecast_values,
        "our_prob_yes": prob_yes,
        "per_model_forecasts": per_model,
        # Per-forecaster percentile lists for numeric/discrete questions.
        # {model_name: [(percentile, value), ...]}. Empty for binary/MC.
        "per_model_numeric_percentiles": per_model_numeric_percentiles,
        # Tri-state: True/False when the STACKED=<bool> marker is present in
        # the comment, None for older comments where the marker didn't exist
        # and stacking status can't be determined.
        "was_stacked": was_stacked,
        "scaling": scaling,
        "open_lower_bound": open_lower,
        "open_upper_bound": open_upper,
        "options": options,
        "comment_text": comment_text,
        "comment_id": comment_id,
        # Metaculus-computed scores from my_forecasts.score_data. Contains
        # peer_score (ascending: negative = worse than crowd), spot_peer_score,
        # baseline_score, spot_baseline_score, coverage, weighted_coverage,
        # relative_legacy_score. None for records fetched before score data
        # was captured; always populated on fresh pulls of resolved questions.
        "metaculus_scores": metaculus_scores,
        "metadata": {
            "nr_forecasters": q.get("nr_forecasters", 0),
            "open_time": q.get("open_time"),
            "actual_resolve_time": q.get("actual_resolve_time"),
            "scheduled_resolve_time": q.get("scheduled_resolve_time"),
            "category": category,
        },
        "brier_score": None,
        "log_score": None,
        "numeric_log_score": None,
        "mc_log_score": None,
    }

    _compute_scores(record)
    return record


def _compute_scores(record: dict) -> None:
    """Compute and set score fields on a record dict in place."""
    q_type = record["type"]
    resolution = record["resolution_parsed"]
    forecast_values = record["our_forecast_values"]

    if forecast_values is None:
        return

    if q_type == "binary" and isinstance(resolution, bool):
        prob_yes = record["our_prob_yes"]
        if prob_yes is not None:
            record["brier_score"] = brier_score(prob_yes, resolution)
            record["log_score"] = binary_log_score(prob_yes, resolution)

    elif q_type in ("numeric", "discrete"):
        scaling = record["scaling"]
        lower_bound = scaling.get("range_min")
        upper_bound = scaling.get("range_max")
        zero_point_raw = scaling.get("zero_point")
        zero_point = float(zero_point_raw) if zero_point_raw not in (None, 0, 0.0) else None

        if lower_bound is not None and upper_bound is not None:
            lower_bound = float(lower_bound)
            upper_bound = float(upper_bound)

            if resolution == "above_upper_bound":
                res_float = upper_bound + 1.0
            elif resolution == "below_lower_bound":
                res_float = lower_bound - 1.0
            elif isinstance(resolution, (int, float)):
                res_float = float(resolution)
            else:
                return

            try:
                record["numeric_log_score"] = numeric_log_score(
                    forecast_values,
                    res_float,
                    lower_bound,
                    upper_bound,
                    record["open_lower_bound"],
                    record["open_upper_bound"],
                    zero_point,
                )
            except (ValueError, ZeroDivisionError) as e:
                logger.warning(f"Failed numeric scoring for post {record['post_id']}: {e}")

    elif q_type == "multiple_choice" and isinstance(resolution, str):
        options = record.get("options") or []
        if resolution in options and forecast_values:
            try:
                correct_idx = options.index(resolution)
                record["mc_log_score"] = mc_log_score(forecast_values, correct_idx)
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed MC scoring for post {record['post_id']}: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_performance_dataset(
    tournament: str = DEFAULT_TOURNAMENT,
    token: str | None = None,
    author_id: int = DEFAULT_BOT_USER_ID,
) -> list[dict]:
    """Fetch questions + comments, match them, parse per-model predictions, compute scores.

    Returns the structured dataset as a list of record dicts.
    Token defaults to METACULUS_TOKEN env var.
    """
    if token is None:
        token = os.environ["METACULUS_TOKEN"]

    logger.info(f"Fetching resolved questions from tournament '{tournament}'...")
    posts = fetch_resolved_questions(tournament, token)

    logger.info(f"Fetching all bot comments (author_id={author_id})...")
    all_comments = fetch_bot_comments(author_id, token)
    logger.info(f"Fetched {len(all_comments)} total comments")
    comment_lookup = _build_comment_lookup(all_comments)
    logger.info(f"Comments mapped to {len(comment_lookup)} unique posts")

    records: list[dict] = []
    for post_data in posts:
        post_records = _process_post(post_data, comment_lookup)
        records.extend(post_records)

    logger.info(f"Collected {len(records)} question records")
    return records


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def save_dataset(data: list[dict], path: str) -> None:
    """Save dataset to a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Saved {len(data)} records to {path}")


def load_dataset(path: str) -> list[dict]:
    """Load dataset from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} records from {path}")
    return data
