"""Tests for the telemetry archive writer (scripts/telemetry/archive.py).

The archive is idempotent by REPLACE-BY-RUN: re-harvesting a run replaces exactly
that run's records (a run's uploaded log is immutable, so a re-parse is byte-identical),
while runs not in the new harvest — including ones whose GHA artifact has since
expired — are preserved. That's what makes ``make sync_telemetry`` safe to run on a
schedule without duplicating or losing records.
"""

import json
import logging
from pathlib import Path

import pytest

from scripts.telemetry.archive import (
    HarvestedRun,
    _write_jsonl,
    load_marker_records,
    load_run_manifest,
    merge_and_write,
)


def _run(run_id: str, records: dict[str, list[dict]], *, workflow: str = "tournament") -> HarvestedRun:
    return HarvestedRun(
        run_id=run_id,
        workflow=workflow,
        artifact=f"research-{run_id}",
        run_date=f"2026-07-{int(run_id[-2:]) % 28 + 1:02d}T00:00:00Z",
        log_files=[f"run_{run_id}.log"],
        records=records,
    )


def _extraction(run_id: str, qid: int, seq: int) -> dict:
    return {"marker": "extraction_rung", "run_id": run_id, "qid": qid, "seq": seq, "rung": "block"}


class TestMergeAndWrite:
    def test_fresh_write_creates_files_and_manifest(self, tmp_path: Path):
        run = _run(
            "100",
            {
                "extraction_rung": [_extraction("100", 1, 0), _extraction("100", 2, 1)],
                "gap_fill_v2": [{"marker": "gap_fill_v2", "run_id": "100", "qid": 1, "seq": 0, "steps": 7}],
            },
        )
        merge_and_write(tmp_path, [run])

        extractions = load_marker_records(tmp_path, "extraction_rung")
        assert len(extractions) == 2
        gap_fills = load_marker_records(tmp_path, "gap_fill_v2")
        assert len(gap_fills) == 1

        manifest = load_run_manifest(tmp_path)
        assert len(manifest) == 1
        assert manifest[0]["run_id"] == "100"
        assert manifest[0]["workflow"] == "tournament"
        assert manifest[0]["artifact"] == "research-100"
        assert manifest[0]["marker_counts"]["extraction_rung"] == 2
        assert manifest[0]["marker_counts"]["gap_fill_v2"] == 1

    def test_reharvest_same_run_is_idempotent(self, tmp_path: Path):
        run = _run("100", {"extraction_rung": [_extraction("100", 1, 0), _extraction("100", 2, 1)]})
        merge_and_write(tmp_path, [run])
        merge_and_write(tmp_path, [run])  # re-harvest identical
        assert len(load_marker_records(tmp_path, "extraction_rung")) == 2
        assert len(load_run_manifest(tmp_path)) == 1

    def test_reharvest_updated_run_replaces_old_records(self, tmp_path: Path):
        merge_and_write(tmp_path, [_run("100", {"extraction_rung": [_extraction("100", 1, 0)]})])
        # Same run re-parsed now yields 3 records (e.g. a re-uploaded/expanded log).
        updated = _run(
            "100",
            {"extraction_rung": [_extraction("100", 1, 0), _extraction("100", 2, 1), _extraction("100", 3, 2)]},
        )
        merge_and_write(tmp_path, [updated])
        recs = load_marker_records(tmp_path, "extraction_rung")
        assert len(recs) == 3, "old run-100 records must be replaced, not appended"

    def test_new_run_added_old_run_preserved(self, tmp_path: Path):
        merge_and_write(tmp_path, [_run("100", {"extraction_rung": [_extraction("100", 1, 0)]})])
        merge_and_write(tmp_path, [_run("200", {"extraction_rung": [_extraction("200", 5, 0)]})])
        recs = load_marker_records(tmp_path, "extraction_rung")
        run_ids = {r["run_id"] for r in recs}
        assert run_ids == {"100", "200"}
        assert len(load_run_manifest(tmp_path)) == 2

    def test_expired_run_not_in_new_harvest_is_preserved(self, tmp_path: Path):
        # Run 100 was harvested while live; later it expires and only run 200 is
        # re-harvestable. Run 100's records must survive the sync.
        merge_and_write(tmp_path, [_run("100", {"extraction_rung": [_extraction("100", 1, 0)]})])
        merge_and_write(tmp_path, [_run("200", {"extraction_rung": [_extraction("200", 5, 0)]})])
        # A third sync sees only run 200 again (100 expired) — 100 stays.
        merge_and_write(tmp_path, [_run("200", {"extraction_rung": [_extraction("200", 5, 0)]})])
        run_ids = {r["run_id"] for r in load_marker_records(tmp_path, "extraction_rung")}
        assert "100" in run_ids

    def test_records_written_as_jsonl(self, tmp_path: Path):
        merge_and_write(tmp_path, [_run("100", {"extraction_rung": [_extraction("100", 1, 0)]})])
        path = tmp_path / "extraction_rung.jsonl"
        assert path.exists()
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        assert lines[0]["qid"] == 1

    def test_returns_per_marker_totals(self, tmp_path: Path):
        run = _run(
            "100",
            {
                "extraction_rung": [_extraction("100", 1, 0), _extraction("100", 2, 1)],
                "ghost_forecast": [{"marker": "ghost_forecast", "run_id": "100", "qid": 1, "seq": 0}],
            },
        )
        totals = merge_and_write(tmp_path, [run])
        assert totals["extraction_rung"] == 2
        assert totals["ghost_forecast"] == 1


class TestAtomicWrite:
    """``_write_jsonl`` rewrites the only durable archive in place, so it must be atomic:
    a crash mid-serialization must leave the previous file intact and leave no temp residue.
    """

    def test_roundtrip_and_no_temp_residue(self, tmp_path: Path):
        path = tmp_path / "x.jsonl"
        _write_jsonl(path, [{"a": 1}, {"a": 2}])
        lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        assert lines == [{"a": 1}, {"a": 2}]
        assert {p.name for p in tmp_path.iterdir()} == {"x.jsonl"}, "no temp sibling should survive a normal write"

    def test_crash_mid_write_leaves_original_intact(self, tmp_path: Path):
        path = tmp_path / "x.jsonl"
        _write_jsonl(path, [{"good": 1}])
        original = path.read_text()

        class _Unserializable:
            pass

        # The second record can't be JSON-serialized: json.dumps raises AFTER the temp
        # file is partly written but BEFORE os.replace, so the original must survive and
        # the temp file must be cleaned up in the finally block.
        with pytest.raises(TypeError):
            _write_jsonl(path, [{"good": 2}, {"bad": _Unserializable()}])

        assert path.read_text() == original, "a mid-write crash must not truncate the durable archive"
        assert {p.name for p in tmp_path.iterdir()} == {"x.jsonl"}, "temp file must be cleaned up on failure"


class TestManifestDowngradeWarning:
    """Replacing a concrete archived workflow label with ``unknown`` warns; anything else stays quiet.

    This is the merge-side signal for a workflow map whose archive under-layer came up
    empty (missing/partial manifest) — the replace-by-run merge still proceeds, but the
    downgrade is no longer silent.
    """

    _WARN_TOKEN = "overwrite a concrete archived workflow label"

    def test_warns_when_concrete_label_replaced_by_unknown(self, tmp_path: Path, caplog):
        merge_and_write(tmp_path, [_run("100", {}, workflow="tournament")])
        with caplog.at_level(logging.WARNING):
            merge_and_write(tmp_path, [_run("100", {}, workflow="unknown")])
        assert any(self._WARN_TOKEN in r.getMessage() for r in caplog.records)

    def test_quiet_when_label_stays_concrete(self, tmp_path: Path, caplog):
        merge_and_write(tmp_path, [_run("100", {}, workflow="tournament")])
        with caplog.at_level(logging.WARNING):
            merge_and_write(tmp_path, [_run("100", {}, workflow="tournament")])
        assert not any(self._WARN_TOKEN in r.getMessage() for r in caplog.records)

    def test_quiet_on_first_harvest_of_an_unknown_run(self, tmp_path: Path, caplog):
        with caplog.at_level(logging.WARNING):
            merge_and_write(tmp_path, [_run("100", {}, workflow="unknown")])
        assert not any(self._WARN_TOKEN in r.getMessage() for r in caplog.records)
