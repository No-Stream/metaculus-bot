"""Extract research text from Metaculus bot comments for backtest replay.

The bot posts a structured comment on each question it forecasts. This script
fetches all such comments, extracts the research section, and writes JSONL
compatible with the research archive system.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.backfill_research_from_logs import detect_gap_fill, detect_providers  # noqa: E402

logger = logging.getLogger(__name__)

BOT_USER_ID = 275109
BASE_URL = "https://www.metaculus.com/api"
PAGE_SIZE = 100
FETCH_DELAY = 2.0

RESEARCH_START_MARKER = "# RESEARCH"
FORECASTS_MARKER = "# FORECASTS"


def get_token() -> str:
    """Load METACULUS_TOKEN from environment or .env file."""
    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        env_path = Path(".env")
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("METACULUS_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not token:
        logger.error("METACULUS_TOKEN not found in environment or .env file")
        sys.exit(1)
    return token


def api_get(endpoint: str, token: str, params: dict | None = None) -> dict:
    """Authenticated GET to the Metaculus API."""
    url = f"{BASE_URL}{endpoint}"
    headers = {"Authorization": f"Token {token}"}
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_all_comments(token: str) -> list[dict]:
    """Paginate through all bot comments."""
    comments: list[dict] = []
    offset = 0
    while True:
        data = api_get(
            "/comments/",
            token,
            params={
                "author": BOT_USER_ID,
                "limit": PAGE_SIZE,
                "offset": offset,
            },
        )
        results = data.get("results", [])
        comments.extend(results)
        logger.info(f"Fetched {len(comments)} comments so far (offset={offset})")
        if not results or data.get("next") is None:
            break
        offset += PAGE_SIZE
        time.sleep(FETCH_DELAY)
    return comments


def get_question_id_for_post(post_id: int, token: str) -> int | None:
    """Resolve a post_id to a question_id via the posts API."""
    try:
        data = api_get(f"/posts/{post_id}/", token)
        question = data.get("question")
        if question and isinstance(question, dict):
            return question.get("id")
        if isinstance(question, int):
            return question
        return None
    except Exception as e:
        logger.warning(f"Failed to resolve post {post_id} to question: {e}")
        return None


def extract_research_text(comment_text: str) -> str | None:
    """Extract the research section from a bot comment.

    Returns the text between '# RESEARCH' and '# FORECASTS' markers,
    stripping the '# RESEARCH' and '## Report 1 Research' header lines.
    Returns None if no research section found or if it's empty.
    """
    research_start = comment_text.find(RESEARCH_START_MARKER)
    if research_start == -1:
        return None

    # Start after the "# RESEARCH" line
    content_start = research_start + len(RESEARCH_START_MARKER)

    forecasts_start = comment_text.find(FORECASTS_MARKER, content_start)
    if forecasts_start == -1:
        research_block = comment_text[content_start:]
    else:
        research_block = comment_text[content_start:forecasts_start]

    # Strip leading header lines: blank lines and "## Report 1 Research"
    lines = research_block.split("\n")
    content_start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped in ("", "## Report 1 Research"):
            content_start_idx = i + 1
        else:
            break

    research_text = "\n".join(lines[content_start_idx:]).strip()
    return research_text if research_text else None


def detect_trimmed(research_text: str) -> bool:
    """Check if research was trimmed due to length."""
    lower = research_text.lower()
    return "[... trimmed for length]" in lower or "[...trimmed" in lower


def build_record(comment: dict, question_id: int | None) -> dict | None:
    """Build a JSONL record from a comment dict. Returns None if no research found."""
    comment_text = comment.get("text") or comment.get("comment_text") or ""
    if not comment_text:
        return None

    research_text = extract_research_text(comment_text)
    if not research_text:
        return None

    post_id = comment.get("on_post")
    comment_id = comment.get("id")

    return {
        "schema_version": 1,
        "qid": question_id,
        "page_url": f"https://www.metaculus.com/questions/{question_id}/" if question_id else "",
        "question_text": "",
        "research_text": research_text,
        "providers_used": detect_providers(research_text),
        "run_mode": "tournament",
        "tournament_id": "",
        "timestamp": comment.get("created_at", ""),
        "run_id": f"comment-{comment_id}",
        "research_chars": len(research_text),
        "gap_fill_used": detect_gap_fill(research_text),
        "on_post": post_id,
        "is_trimmed": detect_trimmed(research_text),
    }


def main():
    parser = argparse.ArgumentParser(description="Backfill research archive from Metaculus bot comments.")
    parser.add_argument("--output-dir", default="backtests/research_archive/backfill", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Fetch comments but don't write")
    parser.add_argument("--limit", type=int, default=None, help="Max comments to process (for testing)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    token = get_token()

    logger.info("Fetching bot comments from Metaculus API...")
    comments = fetch_all_comments(token)
    logger.info(f"Total comments fetched: {len(comments)}")

    if args.limit:
        comments = comments[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "comments_backfill.jsonl"

    # Load existing post IDs to deduplicate
    existing_post_ids: set[int] = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if "on_post" in record:
                        existing_post_ids.add(record["on_post"])

    # Cache for post_id -> question_id mapping
    post_to_qid: dict[int, int | None] = {}

    records_written = 0
    trimmed_count = 0
    no_research_count = 0

    with open(output_file, "a") as f:
        for idx, comment in enumerate(comments, 1):
            post_id: int | None = comment.get("on_post")
            if post_id is None:
                continue
            if post_id in existing_post_ids:
                continue

            # Resolve post_id -> question_id (cached)
            if post_id not in post_to_qid:
                post_to_qid[post_id] = get_question_id_for_post(post_id, token)
                time.sleep(FETCH_DELAY)

            qid = post_to_qid[post_id]

            record = build_record(comment, question_id=qid)
            if record is None:
                no_research_count += 1
                continue

            if record["is_trimmed"]:
                trimmed_count += 1

            if not args.dry_run:
                f.write(json.dumps(record) + "\n")
                existing_post_ids.add(post_id)
            records_written += 1

            if idx % 50 == 0:
                logger.info(f"Processed {idx}/{len(comments)} comments, {records_written} records written")

    logger.info(
        f"Done. {records_written} new records, {trimmed_count} trimmed, {no_research_count} without research section"
    )


if __name__ == "__main__":
    main()
