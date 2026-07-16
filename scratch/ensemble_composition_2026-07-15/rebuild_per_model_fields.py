"""Re-derive per-model fields in perf_all_tagged.json with the current parser.

The 2026-07-15 screening consumed per-model fields produced by the strict
collector path, which silently lost block-only forecasts whose fenced JSON
carried retired tier-2 fields (see the CORRECTION header in RESULTS.md and the
tolerant-salvage rung in ``performance_analysis/parsing.py``, commit f530968).
Records store the raw ``comment_text``, so the parse is fully reproducible
offline: this script re-runs the collector's parsing logic over each record's
stored comment and rewrites ``per_model_forecasts`` /
``per_model_numeric_percentiles`` / ``per_base_model_forecasts`` in place,
leaving every other field (scores, resolutions, tags) untouched.

Prints a coverage diff (records/models gained or lost per field) so parser
changes are auditable. A one-time backup of the original file is kept next to
it as ``perf_all_tagged.pre_tolerant_backup.json`` (never overwritten).

100% offline. Run from repo root:
  uv run python scratch/ensemble_composition_2026-07-15/rebuild_per_model_fields.py
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from pathlib import Path

from metaculus_bot.performance_analysis.parsing import (
    parse_per_base_model_forecasts,
    parse_per_model_forecasts,
    parse_per_model_mc_option_probs,
    parse_per_model_numeric_percentiles,
)

logging.basicConfig(level=logging.WARNING)

DATA = Path("scratch/coherence_2026-07-15/perf_all_tagged.json")
BACKUP = DATA.with_name("perf_all_tagged.pre_tolerant_backup.json")

REBUILT_FIELDS = ("per_model_forecasts", "per_model_numeric_percentiles", "per_base_model_forecasts")


def rebuild_record_fields(record: dict) -> dict:
    """Return {field: new_value} for one record, mirroring collector._process_post."""
    comment_text = record.get("comment_text")
    qid = record.get("question_id")
    if not comment_text:
        return {field: record.get(field) or {} for field in REBUILT_FIELDS}

    numeric_percentiles = parse_per_model_numeric_percentiles(comment_text, question_id=qid)
    mc_option_probs = parse_per_model_mc_option_probs(comment_text)
    per_model = mc_option_probs if mc_option_probs else parse_per_model_forecasts(comment_text)
    per_base = parse_per_base_model_forecasts(comment_text, record.get("type", ""))
    return {
        "per_model_forecasts": per_model,
        "per_model_numeric_percentiles": numeric_percentiles,
        "per_base_model_forecasts": per_base,
    }


def _json_comparable(value: object) -> object:
    """Normalize via a JSON round-trip (tuples -> lists) for old-vs-new comparison."""
    return json.loads(json.dumps(value, default=str))


def main() -> None:
    with open(DATA) as f:
        records = json.load(f)

    gains: Counter[str] = Counter()
    losses: Counter[str] = Counter()
    value_changes: Counter[str] = Counter()
    touched_qids: set[int] = set()

    for record in records:
        new_fields = rebuild_record_fields(record)
        for field, new_value in new_fields.items():
            old_value = record.get(field) or {}
            old_models = set(old_value)
            new_models = set(new_value)
            for model in sorted(new_models - old_models):
                gains[f"{record['type']}/{field}/{model}"] += 1
                touched_qids.add(record["question_id"])
            for model in sorted(old_models - new_models):
                losses[f"{record['type']}/{field}/{model}"] += 1
                touched_qids.add(record["question_id"])
            for model in sorted(old_models & new_models):
                if _json_comparable(old_value[model]) != _json_comparable(new_value[model]):
                    value_changes[f"{record['type']}/{field}/{model}"] += 1
                    touched_qids.add(record["question_id"])
            record[field] = new_value

    if not BACKUP.exists():
        shutil.copy2(DATA, BACKUP)
        print(f"Backup written: {BACKUP}")

    with open(DATA, "w") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"Rewrote {len(records)} records -> {DATA}")
    print(f"Touched question_ids ({len(touched_qids)}): {sorted(touched_qids)}")
    for name, counter in (("GAINED", gains), ("LOST", losses), ("VALUE-CHANGED", value_changes)):
        print(f"\n{name} model entries:")
        if not counter:
            print("  (none)")
        for key, count in sorted(counter.items()):
            print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
