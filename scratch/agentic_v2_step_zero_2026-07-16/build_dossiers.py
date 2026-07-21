"""Build per-question classification dossiers for the gap-fill v2 step-zero miss audit.

Selects the worst recent misses (summer-futureeval-2026 + spring-aib-2026 comments from
2026-03-01 onward) by Metaculus peer score, plus a seeded random control sample of good
outcomes, then assembles one markdown dossier per question containing:

- question title, description, resolution criteria, fine print (pulled read-only from the
  Metaculus API via post_id; auth via METACULUS_TOKEN)
- resolution + scores + the bot's published forecast and per-model forecasts
- the research bundle the forecasters saw (from backtests/research_archive/latest/<qid>.json)
- the per-model forecaster rationales (from the bot's own comment text)

Free/read-only: hits only the Metaculus API. No LLM or research-provider calls.

Usage: uv run python scratch/agentic_v2_step_zero_2026-07-16/build_dossiers.py
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

from metaculus_bot.performance_analysis.collector import _api_get

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "dossiers"
PERF_JSON = REPO / "scratch" / "coherence_2026-07-15" / "perf_all_tagged.json"
ARCHIVE_DIR = REPO / "backtests" / "research_archive" / "latest"

N_WORST = 32
N_CONTROL = 10
CONTROL_SEED = 20260716
RECENT_SPRING_CUTOFF = "2026-03-01"
FETCH_DELAY_SECS = 0.5


def is_recent(rec: dict) -> bool:
    if rec.get("source_tournament") == "summer-futureeval-2026":
        return True
    return (
        rec.get("source_tournament") == "spring-aib-2026"
        and (rec.get("bot_comment_created_at") or "") >= RECENT_SPRING_CUTOFF
    )


def peer(rec: dict) -> float | None:
    return (rec.get("metaculus_scores") or {}).get("peer_score")


def select_questions(records: list[dict]) -> tuple[list[dict], list[dict]]:
    recent = [r for r in records if is_recent(r) and peer(r) is not None]
    misses = sorted(recent, key=lambda r: peer(r) or 0.0)[:N_WORST]
    miss_qids = {r["question_id"] for r in misses}
    good_pool = [r for r in recent if (peer(r) or 0.0) > 20 and r["question_id"] not in miss_qids]
    rng = random.Random(CONTROL_SEED)
    # Finite-N control draw for the hindsight-bias check, not a computation shortcut.
    controls = rng.sample(good_pool, N_CONTROL)  # noqa: HARNESS-SCAN-EXEMPT-subsampling
    return misses, controls


def fetch_question_detail(post_id: int, question_id: int, token: str) -> dict:
    post = _api_get(f"/posts/{post_id}/", token)
    candidates = []
    if post.get("question"):
        candidates.append(post["question"])
    if post.get("group_of_questions"):
        candidates.extend(post["group_of_questions"].get("questions", []))
    for q in candidates:
        if q.get("id") == question_id:
            return {
                "description": q.get("description", ""),
                "resolution_criteria": q.get("resolution_criteria", ""),
                "fine_print": q.get("fine_print", ""),
                "resolution": q.get("resolution"),
                "actual_close_time": q.get("actual_close_time"),
                "scheduled_close_time": q.get("scheduled_close_time"),
                "post_title": post.get("title", ""),
            }
    raise KeyError(f"question {question_id} not found on post {post_id}")


def load_research_text(qid: int) -> tuple[str, dict]:
    path = ARCHIVE_DIR / f"{qid}.json"
    if not path.exists():
        return "", {}
    artifact = json.loads(path.read_text())
    meta = {k: artifact.get(k) for k in ("run_id", "run_mode", "timestamp", "providers_used", "research_chars")}
    return artifact.get("research_text", ""), meta


def split_comment(comment_text: str) -> dict[str, str]:
    """Split the bot comment into SUMMARY / RESEARCH / FORECASTS sections."""
    sections: dict[str, str] = {}
    markers = [("# SUMMARY", "summary"), ("# RESEARCH", "research"), ("# FORECASTS", "forecasts")]
    positions = []
    for marker, name in markers:
        idx = comment_text.find(marker)
        if idx >= 0:
            positions.append((idx, name))
    positions.sort()
    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(comment_text)
        sections[name] = comment_text[start:end]
    return sections


def build_dossier(rec: dict, detail: dict, research_text: str, research_meta: dict, cohort: str) -> str:
    qid = rec["question_id"]
    sections = split_comment(rec.get("comment_text") or "")
    scores = rec.get("metaculus_scores") or {}
    per_model = rec.get("per_model_forecasts") or {}
    lines = [
        f"# Dossier — qid {qid} ({cohort})",
        "",
        f"**Title**: {rec.get('title')}",
        f"**Type**: {rec.get('type')} | **Tournament**: {rec.get('source_tournament')}",
        f"**Forecast submitted**: {rec.get('bot_comment_created_at')}",
        f"**Question close**: {detail.get('actual_close_time') or detail.get('scheduled_close_time')}",
        f"**Peer score**: {scores.get('peer_score'):.1f} | **Baseline**: {scores.get('baseline_score'):.1f}"
        if scores.get("peer_score") is not None
        else "**Peer score**: n/a",
        f"**Resolution**: {rec.get('resolution_raw')} (parsed: {rec.get('resolution_parsed')})",
        f"**Research artifact**: {research_meta or 'MISSING'}",
        "",
        "## Question description",
        detail.get("description", "(unavailable)"),
        "",
        "## Resolution criteria",
        detail.get("resolution_criteria", "(unavailable)"),
        "",
        "## Fine print",
        detail.get("fine_print", "(none)"),
        "",
        "## Published forecast (SUMMARY section of bot comment)",
        sections.get("summary", "(missing)"),
        "",
        "## Per-model final forecasts (parsed)",
        "```json",
        json.dumps(per_model, indent=2, default=str),
        "```",
        "",
        "## Research bundle (what forecasters saw)",
        research_text
        if research_text
        else sections.get("research", "(no research archive artifact; no RESEARCH section)"),
        "",
        "## Forecaster rationales (FORECASTS section of bot comment)",
        sections.get("forecasts", "(missing)"),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    load_dotenv()
    token = os.environ["METACULUS_TOKEN"]
    records = json.loads(PERF_JSON.read_text())
    misses, controls = select_questions(records)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    index = []
    for cohort, recs in (("miss", misses), ("control", controls)):
        for rec in recs:
            qid = rec["question_id"]
            detail = fetch_question_detail(rec["post_id"], qid, token)
            time.sleep(FETCH_DELAY_SECS)
            research_text, research_meta = load_research_text(qid)
            dossier = build_dossier(rec, detail, research_text, research_meta, cohort)
            out_path = OUT_DIR / f"{cohort}_{qid}.md"
            out_path.write_text(dossier)
            index.append(
                {
                    "qid": qid,
                    "cohort": cohort,
                    "type": rec.get("type"),
                    "tournament": rec.get("source_tournament"),
                    "peer_score": peer(rec),
                    "title": rec.get("title"),
                    "submitted": rec.get("bot_comment_created_at"),
                    "resolution": rec.get("resolution_raw"),
                    "has_research_artifact": bool(research_text),
                    "dossier": str(out_path.relative_to(REPO)),
                }
            )
            logger.info(f"wrote {out_path.name} ({len(dossier)} chars)")

    (OUT_DIR.parent / "index.json").write_text(json.dumps(index, indent=2))
    logger.info(f"done: {len(index)} dossiers, index at {OUT_DIR.parent / 'index.json'}")


if __name__ == "__main__":
    main()
