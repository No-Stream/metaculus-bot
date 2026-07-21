"""Stratified sampler for the base-rate audit.

Pulls ~20 binary + ~10 MC + ~10 numeric resolved questions from
scratch/coherence_2026-07-15/perf_all_tagged.json (spring + summer eras
preferred), stratified by peer-score tercile, and writes:

- sample_manifest.json: question ids, type, tercile, scores, resolution (for
  the later correlation step — NOT shown to extractors).
- rationales/<post_id>.md: title + question metadata + 2-3 per-model reasoning
  sections, with resolution/scores stripped (hindsight-bias mitigation for the
  extraction pass).

Free/local only: reads a file already on disk, no API calls.
"""

import ast
import json
import random
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "scratch/coherence_2026-07-15/perf_all_tagged.json"
OUT = Path(__file__).resolve().parent
RATIONALE_DIR = OUT / "rationales"

TARGETS = {"binary": 20, "multiple_choice": 10, "numeric": 10}
MODELS_PER_QUESTION = 3
SEED = 20260716

REASONING_HEADER = re.compile(r"^## R1: Forecaster (\d+) Reasoning\s*$", re.MULTILINE)
MODEL_LINE = re.compile(r"^Model: (\S+)\s*$", re.MULTILINE)


def parse_scores(rec: dict) -> dict:
    ms = rec.get("metaculus_scores")
    if isinstance(ms, str):
        ms = ast.literal_eval(ms)
    return ms or {}


def extract_reasoning_sections(comment_text: str) -> list[dict]:
    """Split out each '## R1: Forecaster N Reasoning' block with its model name."""
    matches = list(REASONING_HEADER.finditer(comment_text))
    sections = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(comment_text)
        body = comment_text[start:end]
        # Header scan only: the Model: line sits in the first few lines of the block.
        model_m = MODEL_LINE.search(body[:500])  # noqa: HARNESS-SCAN-EXEMPT-subsampling
        model = model_m.group(1) if model_m else "unknown"
        sections.append({"forecaster_idx": int(m.group(1)), "model": model, "text": body})
    return sections


def main() -> None:
    rng = random.Random(SEED)
    data = json.loads(SRC.read_text())

    pools: dict[str, list[dict]] = {t: [] for t in TARGETS}
    for rec in data:
        qtype = rec["type"]
        # Fold discrete into numeric (same rationale shape: percentile forecasts).
        pool_key = "numeric" if qtype == "discrete" else qtype
        if pool_key not in pools:
            continue
        if rec["source_tournament"] not in ("spring-aib-2026", "summer-futureeval-2026"):
            continue
        scores = parse_scores(rec)
        if scores.get("peer_score") is None:
            continue
        if "## R1: Forecaster" not in (rec.get("comment_text") or ""):
            continue
        pools[pool_key].append(rec)

    RATIONALE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    model_rotation_offset = 0

    for qtype, target in TARGETS.items():
        pool = sorted(pools[qtype], key=lambda r: parse_scores(r)["peer_score"])
        n = len(pool)
        terciles = {
            "worst": pool[: n // 3],
            "middling": pool[n // 3 : 2 * n // 3],
            "good": pool[2 * n // 3 :],
        }
        per_tercile = [target // 3] * 3
        for i in range(target - sum(per_tercile)):
            per_tercile[i] += 1
        for (tname, bucket), k in zip(terciles.items(), per_tercile):
            # Deliberate, operator-specified stratified sampling (audit design), not
            # silent subsampling of a computation.
            picks = rng.sample(bucket, min(k, len(bucket)))  # noqa: HARNESS-SCAN-EXEMPT-subsampling
            for rec in picks:
                sections = extract_reasoning_sections(rec["comment_text"])
                if not sections:
                    continue
                # Rotate which models we read so coverage varies across questions.
                order = (
                    sections[model_rotation_offset % len(sections) :]
                    + sections[: model_rotation_offset % len(sections)]
                )
                model_rotation_offset += 1
                chosen = order[:MODELS_PER_QUESTION]

                scores = parse_scores(rec)
                manifest.append(
                    {
                        "post_id": rec["post_id"],
                        "question_id": rec["question_id"],
                        "title": rec["title"],
                        "type": qtype,
                        "raw_type": rec["type"],
                        "tercile": tname,
                        "peer_score": scores.get("peer_score"),
                        "spot_peer_score": scores.get("spot_peer_score"),
                        "baseline_score": scores.get("baseline_score"),
                        "resolution": rec.get("resolution_parsed"),
                        "resolution_raw": rec.get("resolution_raw"),
                        "our_prob_yes": rec.get("our_prob_yes"),
                        "per_model_forecasts": rec.get("per_model_forecasts"),
                        "source_tournament": rec["source_tournament"],
                        "config_era": rec.get("config_era"),
                        "bot_comment_created_at": rec.get("bot_comment_created_at"),
                        "models_read": [s["model"] for s in chosen],
                        "options": rec.get("options"),
                    }
                )

                lines = [
                    f"# {rec['title']}",
                    "",
                    f"- post_id: {rec['post_id']}",
                    f"- type: {qtype}",
                    f"- forecast date: {rec.get('bot_comment_created_at', '')[:10]}",
                ]
                if rec.get("options") and rec["options"] != "None":
                    lines.append(f"- options: {rec['options']}")
                lines.append("")
                lines.append("NOTE FOR EXTRACTOR: resolution and scores are deliberately withheld.")
                lines.append("")
                for s in chosen:
                    lines.append(f"---\n\n## MODEL: {s['model']}\n")
                    lines.append(s["text"])
                (RATIONALE_DIR / f"{rec['post_id']}.md").write_text("\n".join(lines))

    (OUT / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"sampled {len(manifest)} questions")
    print(Counter((m["type"], m["tercile"]) for m in manifest))
    print("rationale files:", len(list(RATIONALE_DIR.glob("*.md"))))


if __name__ == "__main__":
    main()
