"""Shared JSONL reader for the research/telemetry download + archive scripts.

``download_research``, ``download_raw_research``, and the telemetry ``archive``
each grew their own copy of the same "read a JSONL file, skip blank lines,
WARN-and-skip malformed lines" loop. This is the one canonical version.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_jsonl_records(path: Path) -> list[dict]:
    """Load all records from a JSONL file, skipping blank and malformed lines.

    A missing file returns ``[]`` — the telemetry archive relies on this for
    not-yet-written marker files; the download scripts only ever pass
    globbed/just-downloaded paths, so the guard is a harmless superset there.
    Malformed lines are logged at WARNING and skipped, never raised, so one
    corrupt line can't sink the whole file.
    """
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Malformed JSON at {path}:{line_num}, skipping")
    return records
