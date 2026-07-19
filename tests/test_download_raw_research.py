"""Tests for the raw-research downloader/archiver (scripts/download_raw_research.py).

Network-free: exercises the local ``raw_research_*.jsonl`` -> per-run-archive harvest,
the replace-by-run idempotency, and within-run dedup against temp files. The GitHub
download path reuses scripts/download_run_logs.py's enumeration + both-prefix download
(already tested there) and is not re-tested here.
"""

import json
from pathlib import Path

from scripts.download_raw_research import (
    RAW_LOG_FILENAME_RE,
    harvest_raw_logs_from_dir,
    merge_and_write,
)
from scripts.download_research import research_jsonl_files


def _write(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def _rec(qid: int, provider: str, phase: str | None = None, fetched_at: str = "2026-07-19T00:00:00+00:00") -> dict:
    return {"qid": qid, "provider": provider, "phase": phase, "fetched_at": fetched_at, "payload": {"k": qid}}


class TestFilenameRegex:
    def test_extracts_run_id(self):
        m = RAW_LOG_FILENAME_RE.search("raw_research_123456.jsonl")
        assert m is not None
        assert m.group("run_id") == "123456"

    def test_extracts_local_run_id(self):
        m = RAW_LOG_FILENAME_RE.search("raw_research_local.jsonl")
        assert m is not None and m.group("run_id") == "local"

    def test_does_not_match_research_outputs(self):
        assert RAW_LOG_FILENAME_RE.search("research_20260719T000000Z.jsonl") is None


class TestHarvest:
    def test_groups_records_by_run_id_from_filename(self, tmp_path: Path):
        run_dir = tmp_path / "123456"
        _write(
            run_dir / "run_logs" / "raw_research_123456.jsonl",
            [_rec(1, "asknews", "hot"), _rec(1, "asknews", "historical")],
        )

        harvested = harvest_raw_logs_from_dir(run_dir)
        assert set(harvested) == {"123456"}
        assert len(harvested["123456"]) == 2

    def test_ignores_non_raw_jsonl(self, tmp_path: Path):
        run_dir = tmp_path / "123456"
        _write(run_dir / "research_outputs" / "research_20260719.jsonl", [{"qid": 1, "research_text": "x"}])

        assert harvest_raw_logs_from_dir(run_dir) == {}

    def test_multiple_runs(self, tmp_path: Path):
        run_dir = tmp_path / "root"
        _write(run_dir / "raw_research_500.jsonl", [_rec(1, "native_search")])
        _write(run_dir / "raw_research_501.jsonl", [_rec(2, "gemini_search")])

        harvested = harvest_raw_logs_from_dir(run_dir)
        assert set(harvested) == {"500", "501"}

    def test_skips_malformed_lines(self, tmp_path: Path):
        run_dir = tmp_path / "500"
        path = run_dir / "raw_research_500.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(_rec(1, "asknews", "hot")) + "\n{ this is not json\n")

        harvested = harvest_raw_logs_from_dir(run_dir)
        assert len(harvested["500"]) == 1


class TestMergeAndWrite:
    def test_writes_one_file_per_run(self, tmp_path: Path):
        archive = tmp_path / "raw"
        merge_and_write(archive, {"500": [_rec(1, "asknews", "hot")], "501": [_rec(2, "gemini_search")]})

        assert (archive / "500.jsonl").exists()
        assert (archive / "501.jsonl").exists()

    def test_replace_by_run_is_idempotent(self, tmp_path: Path):
        archive = tmp_path / "raw"
        harvested = {"500": [_rec(1, "asknews", "hot"), _rec(1, "asknews", "historical")]}

        merge_and_write(archive, harvested)
        first = (archive / "500.jsonl").read_text()
        merge_and_write(archive, harvested)
        second = (archive / "500.jsonl").read_text()

        assert first == second

    def test_dedups_within_run_keeping_latest(self, tmp_path: Path):
        archive = tmp_path / "raw"
        older = _rec(1, "asknews", "hot", fetched_at="2026-07-19T00:00:00+00:00")
        newer = _rec(1, "asknews", "hot", fetched_at="2026-07-19T05:00:00+00:00")

        merge_and_write(archive, {"500": [older, newer]})

        lines = [json.loads(x) for x in (archive / "500.jsonl").read_text().splitlines() if x.strip()]
        assert len(lines) == 1
        assert lines[0]["fetched_at"] == "2026-07-19T05:00:00+00:00"

    def test_distinct_phases_are_not_deduped(self, tmp_path: Path):
        archive = tmp_path / "raw"
        merge_and_write(archive, {"500": [_rec(1, "asknews", "hot"), _rec(1, "asknews", "historical")]})

        lines = (archive / "500.jsonl").read_text().splitlines()
        assert len([x for x in lines if x.strip()]) == 2

    def test_second_harvest_leaves_other_runs_untouched(self, tmp_path: Path):
        archive = tmp_path / "raw"
        merge_and_write(archive, {"500": [_rec(1, "asknews", "hot")]})
        merge_and_write(archive, {"501": [_rec(2, "gemini_search")]})

        assert (archive / "500.jsonl").exists()
        assert (archive / "501.jsonl").exists()


class TestMainArchiveExcludesRawLogs:
    """download_research.py globs the whole research-* artifact for *.jsonl; the raw
    log now rides in run_logs/ inside that artifact, so the main-archive glob must
    skip it or the raw records corrupt the per-question archive (dedup on qid/run_id).
    """

    def test_research_jsonl_files_excludes_raw_research(self, tmp_path: Path):
        _write(tmp_path / "research_outputs" / "research_20260719.jsonl", [{"qid": 1, "research_text": "x"}])
        _write(tmp_path / "run_logs" / "raw_research_500.jsonl", [_rec(1, "asknews", "hot")])

        found = {p.name for p in research_jsonl_files(tmp_path)}
        assert found == {"research_20260719.jsonl"}
