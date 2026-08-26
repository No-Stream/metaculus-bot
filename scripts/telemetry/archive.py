"""Durable telemetry archive: one JSONL per marker type + a runs.jsonl manifest.

Layout under ``backtests/telemetry_archive/`` (gitignored via the ``backtests/`` rule):

    extraction_rung.jsonl        # one record per harvested marker line
    gap_fill_v2.jsonl
    ghost_forecast.jsonl
    ghost_forecast_json.jsonl    # full-fidelity companion to ghost_forecast
    ...
    runs.jsonl                   # manifest: one record per harvested run

The per-marker file set is derived from ``MARKER_SPECS`` (via ``MARKER_NAMES``),
so adding a marker spec automatically gives it its own JSONL here — no change
to this module is needed.

IDEMPOTENCY — REPLACE-BY-RUN: :func:`merge_and_write` drops every existing record
whose ``run_id`` is present in the incoming harvest, then appends the freshly-parsed
records. A run's uploaded log is immutable, so re-parsing yields byte-identical
records; runs absent from the new harvest (e.g. their GHA artifact expired) are left
untouched. Re-running the sync therefore neither duplicates nor loses records.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from scripts.telemetry.jsonl import load_jsonl_records
from scripts.telemetry.markers import MARKER_SPECS

logger = logging.getLogger(__name__)

MARKER_NAMES: tuple[str, ...] = tuple(spec.name for spec in MARKER_SPECS)
RUNS_MANIFEST_FILE = "runs.jsonl"


@dataclass
class HarvestedRun:
    """One run's harvested telemetry: run metadata + per-marker record lists."""

    run_id: str
    workflow: str
    artifact: str
    run_date: str
    log_files: list[str]
    records: dict[str, list[dict]] = field(default_factory=dict)

    def total_records(self) -> int:
        return sum(len(v) for v in self.records.values())


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Atomically write ``records`` to ``path`` as JSONL (temp sibling + ``os.replace``).

    ``merge_and_write`` rewrites the ONLY durable telemetry archive in place. An in-place
    ``open(path, "w")`` truncates the file at the moment it opens, so a crash mid-write
    (serialization error, SIGKILL) would leave the archive truncated or empty. We stream
    into a sibling temp file first and atomically rename it over the target — the original
    stays intact until the full new content is on disk. The temp file shares ``path``'s
    directory so the rename is same-filesystem (a prerequisite for ``os.replace`` atomicity).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_marker_records(archive_dir: Path, marker: str) -> list[dict]:
    """Load all archived records for one marker type."""
    return load_jsonl_records(Path(archive_dir) / f"{marker}.jsonl")


def load_run_manifest(archive_dir: Path) -> list[dict]:
    """Load the run manifest (one record per harvested run)."""
    return load_jsonl_records(Path(archive_dir) / RUNS_MANIFEST_FILE)


def _sort_key(record: dict) -> tuple:
    """Stable ordering so re-harvests produce identical files: run_date, run_id, seq."""
    return (str(record.get("run_date", "")), str(record.get("run_id", "")), record.get("seq", 0))


def merge_and_write(archive_dir: Path, runs: list[HarvestedRun]) -> dict[str, int]:
    """Merge harvested runs into the archive (replace-by-run) and return per-marker totals.

    Returns a ``{marker_name: total_record_count}`` map reflecting the archive AFTER
    the merge — handy for the sync's harvest report.
    """
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    replaced_run_ids = {run.run_id for run in runs}

    totals: dict[str, int] = {}
    for marker in MARKER_NAMES:
        existing = load_marker_records(archive_dir, marker)
        kept = [r for r in existing if r.get("run_id") not in replaced_run_ids]
        incoming: list[dict] = []
        for run in runs:
            incoming.extend(run.records.get(marker, []))
        merged = sorted(kept + incoming, key=_sort_key)
        _write_jsonl(archive_dir / f"{marker}.jsonl", merged)
        totals[marker] = len(merged)

    _merge_manifest(archive_dir, runs, replaced_run_ids)
    return totals


def _merge_manifest(archive_dir: Path, runs: list[HarvestedRun], replaced_run_ids: set[str]) -> None:
    existing = load_run_manifest(archive_dir)
    existing_workflows = {str(r.get("run_id", "")): str(r.get("workflow", "")) for r in existing}
    downgraded = sum(
        1 for run in runs if run.workflow == "unknown" and existing_workflows.get(run.run_id, "") not in ("", "unknown")
    )
    if downgraded:
        logger.warning(
            f"{downgraded} of {len(runs)} harvested run(s) overwrite a concrete archived workflow label with "
            f"'unknown' in {archive_dir} — the resolved workflow map supplied no label for runs the archive "
            f"already attributes"
        )
    kept = [r for r in existing if r.get("run_id") not in replaced_run_ids]
    incoming = [
        {
            "run_id": run.run_id,
            "workflow": run.workflow,
            "artifact": run.artifact,
            "run_date": run.run_date,
            "log_files": run.log_files,
            "marker_counts": {marker: len(recs) for marker, recs in run.records.items() if recs},
            "total_records": run.total_records(),
        }
        for run in runs
    ]
    merged = sorted(kept + incoming, key=lambda r: (str(r.get("run_date", "")), str(r.get("run_id", ""))))
    _write_jsonl(archive_dir / RUNS_MANIFEST_FILE, merged)
