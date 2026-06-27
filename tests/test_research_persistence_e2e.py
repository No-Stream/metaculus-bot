"""End-to-end tests for the research persistence pipeline.

Tests the full flow across all components:
  Write path: ResearchPersistenceWriter.record() → .flush() → JSONL on disk
  Read path: _load_research_from_archive(dir, questions) → dict[int, str] cache
  Backfill: parse_research_blocks(log_text, run_id) → list[dict] records
  Download/merge: deduplicate_records() + build_archive() → archive structure
"""

import json
import subprocess
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from backtest import _load_research_from_archive
from metaculus_bot.research.persistence import ResearchPersistenceWriter
from scripts.backfill_research_from_logs import existing_records, parse_research_blocks
from scripts.download_research import (
    build_archive,
    deduplicate_records,
    download_research_artifacts,
    list_research_artifacts,
)
from scripts.download_research import main as download_research_main

# --- Realistic test data ---

REALISTIC_RESEARCH_TEXT = """\
## News Articles (AskNews)

**Article 1: Major Climate Summit Reaches Agreement (2026-05-20)**
World leaders at the 2026 Climate Summit in Geneva reached a landmark agreement to reduce emissions
by 45% by 2035. The deal includes binding commitments from 190 nations and a $500B green transition fund.
Sources: Reuters, AP News, The Guardian.

**Article 2: Implementation Challenges Remain (2026-05-18)**
Experts warn that while the agreement is historic, implementation faces significant hurdles including
political turnover, economic pressures, and technological readiness in developing nations.

---

## Web Research (Native Search)

According to the UN Climate Change Secretariat, the agreement builds on the Paris Accord framework
but introduces stronger enforcement mechanisms. Key provisions include:
- Quarterly progress reporting requirements
- Financial penalties for non-compliance (up to 2% of GDP)
- Technology transfer mechanisms for developing nations

Historical precedent: The Kyoto Protocol achieved ~60% compliance among signatories.
The Paris Accord achieved ~75% partial compliance by 2025.

---

## Targeted Gap-Fill (second pass)

**Gap: What is the current baseline emissions trajectory without new policy?**
The IPCC AR7 (2026) projects a baseline of +2.8C by 2100 without additional policy intervention.
Current committed policies put the world on a +2.1C trajectory. The new agreement targets +1.5C.

**Gap: What are the enforcement precedents for international climate agreements?**
The Montreal Protocol (ozone) achieved near-universal compliance via trade sanctions.
The Paris Agreement's "ratchet mechanism" produced increased ambition in 72% of NDC updates."""

MULTI_QUESTION_LOG = """\
forecast_job\tRun forecasts\t2026-05-22T14:00:00.1234567Z 2026-05-22 14:00:00,123 - metaculus_bot.research.orchestrator - INFO - Found Research for URL https://www.metaculus.com/questions/43613/will-climate-target-be-met/:
forecast_job\tRun forecasts\t2026-05-22T14:00:00.1234568Z ## News Articles (AskNews)
forecast_job\tRun forecasts\t2026-05-22T14:00:00.1234569Z Climate summit reached agreement on emissions targets.
forecast_job\tRun forecasts\t2026-05-22T14:00:00.1234570Z ## Web Research (Native Search)
forecast_job\tRun forecasts\t2026-05-22T14:00:00.1234571Z Historical compliance rates suggest 60-75% probability.
forecast_job\tRun forecasts\t2026-05-22T14:00:00.1234572Z ## Targeted Gap-Fill (second pass)
forecast_job\tRun forecasts\t2026-05-22T14:00:00.1234573Z Enforcement mechanisms are stronger than Paris Accord.
forecast_job\tRun forecasts\t2026-05-22T14:05:00.0000000Z 2026-05-22 14:05:00,000 - metaculus_bot.research.orchestrator - INFO - Found Research for URL https://www.metaculus.com/questions/50001/will-ai-achieve-agi-by-2030/:
forecast_job\tRun forecasts\t2026-05-22T14:05:00.0000001Z ## News Articles (AskNews)
forecast_job\tRun forecasts\t2026-05-22T14:05:00.0000002Z Leading AI labs report breakthrough in reasoning.
forecast_job\tRun forecasts\t2026-05-22T14:05:00.0000003Z ## Web Research (Google Search via Gemini)
forecast_job\tRun forecasts\t2026-05-22T14:05:00.0000004Z Expert surveys show wide disagreement on timelines.
forecast_job\tRun forecasts\t2026-05-22T14:10:00.0000000Z 2026-05-22 14:10:00,000 - metaculus_bot.research.orchestrator - INFO - Starting forecaster fan-out
"""


def _make_question(qid: int) -> types.SimpleNamespace:
    """Create a minimal mock question with id_of_question attribute."""
    return types.SimpleNamespace(id_of_question=qid)


class TestWritePathE2E:
    """Test that the orchestrator's research_sink callback produces valid JSONL via the writer."""

    def test_record_and_flush_produces_valid_jsonl(self, tmp_path: Path) -> None:
        writer = ResearchPersistenceWriter(run_mode="tournament", tournament_id="3672", run_id="12345")

        writer.record(
            qid=43613,
            page_url="https://www.metaculus.com/questions/43613/will-climate-target-be-met/",
            question_text="Will the 2026 climate agreement emissions target be met by 2035?",
            research_text=REALISTIC_RESEARCH_TEXT,
            providers_used=["asknews", "native_search"],
            gap_fill_used=True,
        )

        output_dir = tmp_path / "research_outputs"
        result = writer.flush(output_dir=str(output_dir))

        assert result is not None
        assert result.exists()
        assert result.suffix == ".jsonl"

        lines = result.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["schema_version"] == 1
        assert record["qid"] == 43613
        assert record["page_url"] == "https://www.metaculus.com/questions/43613/will-climate-target-be-met/"
        assert record["question_text"] == "Will the 2026 climate agreement emissions target be met by 2035?"
        assert record["research_text"] == REALISTIC_RESEARCH_TEXT
        assert record["providers_used"] == ["asknews", "native_search"]
        assert record["gap_fill_used"] is True
        assert record["run_mode"] == "tournament"
        assert record["tournament_id"] == "3672"
        assert record["run_id"] == "12345"
        assert record["research_chars"] == len(REALISTIC_RESEARCH_TEXT)
        assert "timestamp" in record

    def test_research_text_preserved_exactly_no_truncation(self, tmp_path: Path) -> None:
        writer = ResearchPersistenceWriter(run_mode="tournament", tournament_id="3672", run_id="99")

        long_research = "A" * 50_000 + "\n## Targeted Gap-Fill (second pass)\n" + "B" * 50_000
        writer.record(
            qid=1,
            page_url="https://www.metaculus.com/questions/1/",
            question_text="Test?",
            research_text=long_research,
            providers_used=["asknews"],
            gap_fill_used=True,
        )

        result = writer.flush(output_dir=str(tmp_path))
        assert result is not None
        record = json.loads(result.read_text().strip())
        assert record["research_text"] == long_research
        assert record["research_chars"] == len(long_research)

    def test_sink_callback_signature_matches_orchestrator_call_site(self) -> None:
        """The orchestrator calls research_sink with these exact kwargs. Verify writer.record accepts them."""
        writer = ResearchPersistenceWriter(run_mode="tournament", tournament_id="t", run_id="r")

        # These are the exact kwargs from research_orchestrator.py:100-107
        writer.record(
            qid=43613,
            page_url="https://www.metaculus.com/questions/43613/",
            question_text="Will X happen?",
            research_text="## News Articles (AskNews)\nContent here",
            providers_used=["asknews", "native_search", "gemini_search"],
            gap_fill_used=False,
        )

        assert len(writer._records) == 1
        assert writer._records[0]["qid"] == 43613

    def test_multiple_questions_in_single_flush(self, tmp_path: Path) -> None:
        writer = ResearchPersistenceWriter(run_mode="minibench", tournament_id="bench-1", run_id="local-42")

        for qid in [43613, 50001, 38880]:
            writer.record(
                qid=qid,
                page_url=f"https://www.metaculus.com/questions/{qid}/",
                question_text=f"Question {qid}?",
                research_text=f"Research for {qid}",
                providers_used=["asknews"],
                gap_fill_used=qid == 43613,
            )

        result = writer.flush(output_dir=str(tmp_path))
        assert result is not None

        lines = result.read_text().strip().splitlines()
        assert len(lines) == 3

        records = [json.loads(line) for line in lines]
        qids = [r["qid"] for r in records]
        assert qids == [43613, 50001, 38880]
        assert records[0]["gap_fill_used"] is True
        assert records[1]["gap_fill_used"] is False


class TestReadPathE2E:
    """Test _load_research_from_archive reads archive dir into a research_cache dict."""

    def test_loads_multiple_questions_from_archive(self, tmp_path: Path) -> None:
        latest_dir = tmp_path / "latest"
        latest_dir.mkdir()

        # Write archive JSON files
        record_1 = {
            "schema_version": 1,
            "qid": 43613,
            "research_text": "## News Articles (AskNews)\nClimate summit reached agreement.",
            "providers_used": ["asknews", "native_search"],
            "timestamp": "2026-05-22T14:00:00Z",
        }
        record_2 = {
            "schema_version": 1,
            "qid": 50001,
            "research_text": "## Web Research (Native Search)\nAI progress report.",
            "providers_used": ["native_search"],
            "timestamp": "2026-05-22T15:00:00Z",
        }

        (latest_dir / "43613.json").write_text(json.dumps(record_1))
        (latest_dir / "50001.json").write_text(json.dumps(record_2))

        questions = [_make_question(43613), _make_question(50001)]
        cache = _load_research_from_archive(str(latest_dir), questions)

        assert cache == {
            43613: "## News Articles (AskNews)\nClimate summit reached agreement.",
            50001: "## Web Research (Native Search)\nAI progress report.",
        }

    def test_partial_coverage_missing_questions_excluded(self, tmp_path: Path) -> None:
        latest_dir = tmp_path / "latest"
        latest_dir.mkdir()

        record = {"qid": 43613, "research_text": "Some research"}
        (latest_dir / "43613.json").write_text(json.dumps(record))

        questions = [_make_question(43613), _make_question(99999)]
        cache = _load_research_from_archive(str(latest_dir), questions)

        assert 43613 in cache
        assert 99999 not in cache
        assert cache[43613] == "Some research"

    def test_missing_dir_returns_empty_dict(self, tmp_path: Path) -> None:
        nonexistent = str(tmp_path / "does_not_exist")
        questions = [_make_question(12345)]
        cache = _load_research_from_archive(nonexistent, questions)
        assert cache == {}

    def test_empty_research_text_still_loaded(self, tmp_path: Path) -> None:
        latest_dir = tmp_path / "latest"
        latest_dir.mkdir()

        record = {"qid": 77777, "research_text": ""}
        (latest_dir / "77777.json").write_text(json.dumps(record))

        questions = [_make_question(77777)]
        cache = _load_research_from_archive(str(latest_dir), questions)
        assert cache[77777] == ""


class TestBackfillE2E:
    """Test backfill: log text -> parsed records -> dedup -> JSONL output."""

    def test_multi_question_log_produces_correct_records(self) -> None:
        records = parse_research_blocks(MULTI_QUESTION_LOG, run_id="gh-action-555")

        assert len(records) == 2

        r1 = records[0]
        assert r1["qid"] == 43613
        assert r1["page_url"] == "https://www.metaculus.com/questions/43613/will-climate-target-be-met/"
        assert r1["timestamp"] == "2026-05-22T14:00:00Z"
        assert r1["run_id"] == "gh-action-555"
        assert "asknews" in r1["providers_used"]
        assert "native_search" in r1["providers_used"]
        assert r1["gap_fill_used"] is True
        assert "Climate summit reached agreement" in r1["research_text"]
        assert "Enforcement mechanisms" in r1["research_text"]

        r2 = records[1]
        assert r2["qid"] == 50001
        assert r2["page_url"] == "https://www.metaculus.com/questions/50001/will-ai-achieve-agi-by-2030/"
        assert "asknews" in r2["providers_used"]
        assert "gemini_search" in r2["providers_used"]
        assert r2["gap_fill_used"] is False

    def test_dedup_with_existing_records(self, tmp_path: Path) -> None:
        """A second parse of the same log produces no new records when checked against existing."""
        records = parse_research_blocks(MULTI_QUESTION_LOG, run_id="gh-action-555")

        # Write first batch
        output_dir = tmp_path / "backfill"
        output_dir.mkdir()
        jsonl_file = output_dir / "backfill.jsonl"
        with open(jsonl_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # Check existing records
        seen = existing_records(output_dir)
        assert (43613, "gh-action-555") in seen
        assert (50001, "gh-action-555") in seen

        # Second parse — filter against seen
        records_2 = parse_research_blocks(MULTI_QUESTION_LOG, run_id="gh-action-555")
        new_records = [r for r in records_2 if (r["qid"], r["run_id"]) not in seen]
        assert len(new_records) == 0

    def test_jsonl_round_trip_integrity(self, tmp_path: Path) -> None:
        records = parse_research_blocks(MULTI_QUESTION_LOG, run_id="run-789")

        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Read back
        loaded = []
        with open(jsonl_file) as f:
            for line in f:
                loaded.append(json.loads(line.strip()))

        assert len(loaded) == len(records)
        for original, roundtripped in zip(records, loaded):
            assert original == roundtripped


class TestDownloadMergeE2E:
    """Test deduplicate_records + build_archive produce correct archive structure."""

    def test_deduplicate_keeps_distinct_run_ids(self) -> None:
        records = [
            {"qid": 43613, "run_id": "run-1", "timestamp": "2026-05-20T10:00:00Z", "providers_used": ["asknews"]},
            {"qid": 43613, "run_id": "run-2", "timestamp": "2026-05-21T10:00:00Z", "providers_used": ["asknews"]},
            {"qid": 50001, "run_id": "run-1", "timestamp": "2026-05-20T11:00:00Z", "providers_used": ["native_search"]},
            {"qid": 50001, "run_id": "run-2", "timestamp": "2026-05-21T11:00:00Z", "providers_used": ["gemini_search"]},
        ]

        deduped = deduplicate_records(records)
        assert len(deduped) == 4

    def test_deduplicate_removes_true_duplicates(self) -> None:
        records = [
            {"qid": 43613, "run_id": "run-1", "timestamp": "2026-05-20T10:00:00Z", "research_text": "old"},
            {"qid": 43613, "run_id": "run-1", "timestamp": "2026-05-20T12:00:00Z", "research_text": "newer"},
        ]

        deduped = deduplicate_records(records)
        assert len(deduped) == 1
        assert deduped[0]["research_text"] == "newer"

    def test_build_archive_structure(self, tmp_path: Path) -> None:
        records = [
            {
                "qid": 43613,
                "run_id": "run-1",
                "timestamp": "2026-05-20T10:00:00Z",
                "research_text": "Older research for 43613",
                "providers_used": ["asknews"],
            },
            {
                "qid": 43613,
                "run_id": "run-2",
                "timestamp": "2026-05-21T10:00:00Z",
                "research_text": "Newer research for 43613",
                "providers_used": ["asknews", "native_search"],
            },
            {
                "qid": 50001,
                "run_id": "run-1",
                "timestamp": "2026-05-20T11:00:00Z",
                "research_text": "Research for 50001",
                "providers_used": ["gemini_search"],
            },
            {
                "qid": 50001,
                "run_id": "run-2",
                "timestamp": "2026-05-21T11:00:00Z",
                "research_text": "Newer research for 50001",
                "providers_used": ["native_search", "gemini_search"],
            },
        ]

        archive_dir = tmp_path / "archive"
        build_archive(records, archive_dir)

        # latest/<qid>.json contains the newest record
        latest_43613 = json.loads((archive_dir / "latest" / "43613.json").read_text())
        assert latest_43613["research_text"] == "Newer research for 43613"
        assert latest_43613["timestamp"] == "2026-05-21T10:00:00Z"

        latest_50001 = json.loads((archive_dir / "latest" / "50001.json").read_text())
        assert latest_50001["research_text"] == "Newer research for 50001"
        assert latest_50001["timestamp"] == "2026-05-21T11:00:00Z"

        # by_qid/<qid>.jsonl has all versions, newest first
        by_qid_43613 = (archive_dir / "by_qid" / "43613.jsonl").read_text().strip().splitlines()
        assert len(by_qid_43613) == 2
        first_record = json.loads(by_qid_43613[0])
        assert first_record["timestamp"] == "2026-05-21T10:00:00Z"  # newest first
        second_record = json.loads(by_qid_43613[1])
        assert second_record["timestamp"] == "2026-05-20T10:00:00Z"

        by_qid_50001 = (archive_dir / "by_qid" / "50001.jsonl").read_text().strip().splitlines()
        assert len(by_qid_50001) == 2

        # manifest.json has correct metadata
        manifest = json.loads((archive_dir / "manifest.json").read_text())
        assert "43613" in manifest
        assert manifest["43613"]["versions_count"] == 2
        assert manifest["43613"]["latest_timestamp"] == "2026-05-21T10:00:00Z"
        assert sorted(manifest["43613"]["providers"]) == ["asknews", "native_search"]

        assert "50001" in manifest
        assert manifest["50001"]["versions_count"] == 2
        assert sorted(manifest["50001"]["providers"]) == ["gemini_search", "native_search"]


def _iso(dt: datetime) -> str:
    """Render a datetime as a GH-style 'YYYY-MM-DDTHH:MM:SSZ' createdAt string."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _artifact(name: str, run_id: int, created: datetime, expired: bool = False) -> dict:
    """Build an artifact object as emitted by `list_research_artifacts`'s --jq projection."""
    return {
        "id": run_id * 10,
        "name": name,
        "created_at": _iso(created),
        "expires_at": _iso(created + timedelta(days=90)),
        "expired": expired,
        "size_in_bytes": 1234,
        "run_id": run_id,
    }


def _gh_stdout(artifacts: list[dict]) -> str:
    """Mimic `gh api --paginate ... --jq` output: one JSON object per line."""
    return "\n".join(json.dumps(a) for a in artifacts) + "\n"


class TestArtifactsEndpointSweep:
    """Test the complete artifacts-endpoint download path: filtering, expiry logging, dedup.

    All `gh` calls are mocked — no live GitHub access, so the suite stays self-contained.
    """

    def test_list_research_artifacts_parses_paginated_stream(self) -> None:
        """`gh api --paginate ... --jq` emits one object per line; parse them all."""
        now = datetime.now(timezone.utc)
        artifacts = [
            _artifact("research-100", 100, now - timedelta(days=1)),
            _artifact("research-200", 200, now - timedelta(days=2)),
            _artifact("some-other-artifact", 300, now - timedelta(days=3)),
        ]
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=_gh_stdout(artifacts), stderr="")
        with mock.patch("scripts.download_research.subprocess.run", return_value=completed) as run:
            parsed = list_research_artifacts("repo")

        # The REST endpoint is NOT workflow-scoped: it returns ALL artifacts (filtering
        # to research-* happens downstream), so all three rows parse.
        assert [a["name"] for a in parsed] == ["research-100", "research-200", "some-other-artifact"]
        # We page via the artifacts endpoint with --paginate, not `gh run list`.
        cmd = run.call_args.args[0]
        assert cmd[:3] == ["gh", "api", "--paginate"]
        assert "/repos/repo/actions/artifacts?per_page=100" in cmd

    def test_list_research_artifacts_exits_on_gh_failure(self) -> None:
        """A gh error is fail-fast (sys.exit), never silently swallowed."""
        failed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
        with mock.patch("scripts.download_research.subprocess.run", return_value=failed):
            try:
                list_research_artifacts("repo")
            except SystemExit as exc:
                assert exc.code == 1
            else:
                raise AssertionError("expected SystemExit on gh failure")

    def test_only_live_research_artifacts_downloaded_expired_logged(self, caplog) -> None:  # noqa: ANN001
        """Live research-* downloaded; expired logged (not downloaded); non-research ignored."""
        now = datetime.now(timezone.utc)
        artifacts = [
            _artifact("research-100", 100, now - timedelta(days=1)),
            _artifact("research-200", 200, now - timedelta(days=2)),
            _artifact("research-999", 999, now - timedelta(days=95), expired=True),  # past retention
            _artifact("benchmark-results", 300, now - timedelta(days=1)),  # not research-*
        ]

        records_by_run = {
            100: [{"qid": 1, "run_id": "100", "timestamp": "t1"}],
            200: [{"qid": 2, "run_id": "200", "timestamp": "t2"}],
        }

        def fake_download(run_id, repo, artifact_name, dest_dir):  # noqa: ANN001, ANN202
            return [run_id] if records_by_run.get(run_id) else []

        with (
            mock.patch("scripts.download_research.verify_gh_cli"),
            mock.patch("scripts.download_research.list_research_artifacts", return_value=artifacts),
            mock.patch("scripts.download_research.download_artifact", side_effect=fake_download) as dl,
            mock.patch("scripts.download_research.load_jsonl_records", side_effect=lambda f: records_by_run[f]),
            caplog.at_level("WARNING", logger="scripts.download_research"),
        ):
            records = download_research_artifacts(repo="repo", since_days=0)

        # Only the two LIVE research artifacts were downloaded.
        attempted = sorted(c.args[0] for c in dl.call_args_list)
        assert attempted == [100, 200]

        # The expired research artifact was NOT downloaded but WAS logged loudly by name + date.
        assert 999 not in attempted
        assert any("research-999" in r.message and "LOST" in r.message for r in caplog.records)

        # Records aggregated only from the live downloads.
        assert sorted(r["qid"] for r in records) == [1, 2]

    def test_dedup_by_run_id_downloads_once(self) -> None:
        """Two artifact rows sharing a workflow_run id are downloaded at most once."""
        now = datetime.now(timezone.utc)
        artifacts = [
            _artifact("research-100", 100, now - timedelta(days=1)),
            _artifact("research-100-retry", 100, now - timedelta(hours=1)),  # same run_id
            _artifact("research-200", 200, now - timedelta(days=2)),
        ]

        def fake_download(run_id, repo, artifact_name, dest_dir):  # noqa: ANN001, ANN202
            return [run_id]

        with (
            mock.patch("scripts.download_research.verify_gh_cli"),
            mock.patch("scripts.download_research.list_research_artifacts", return_value=artifacts),
            mock.patch("scripts.download_research.download_artifact", side_effect=fake_download) as dl,
            mock.patch(
                "scripts.download_research.load_jsonl_records",
                side_effect=lambda f: [{"qid": f, "run_id": str(f), "timestamp": "t"}],
            ),
        ):
            download_research_artifacts(repo="repo", since_days=0)

        attempted = sorted(c.args[0] for c in dl.call_args_list)
        assert attempted == [100, 200]  # run 100 downloaded once despite two artifact rows

    def test_since_days_post_filter_excludes_old_artifacts(self) -> None:
        """--since-days windows out live artifacts created before the cutoff."""
        now = datetime.now(timezone.utc)
        artifacts = [
            _artifact("research-100", 100, now - timedelta(days=1)),  # inside 7-day window
            _artifact("research-200", 200, now - timedelta(days=30)),  # outside 7-day window
        ]

        def fake_download(run_id, repo, artifact_name, dest_dir):  # noqa: ANN001, ANN202
            return [run_id]

        with (
            mock.patch("scripts.download_research.verify_gh_cli"),
            mock.patch("scripts.download_research.list_research_artifacts", return_value=artifacts),
            mock.patch("scripts.download_research.download_artifact", side_effect=fake_download) as dl,
            mock.patch(
                "scripts.download_research.load_jsonl_records",
                side_effect=lambda f: [{"qid": f, "run_id": str(f), "timestamp": "t"}],
            ),
        ):
            download_research_artifacts(repo="repo", since_days=7)

        attempted = sorted(c.args[0] for c in dl.call_args_list)
        assert attempted == [100]  # only the recent artifact

    def test_full_flow_rebuilds_manifest(self, tmp_path: Path) -> None:
        """The downloaded records feed build_archive and produce a manifest (rebuild intact)."""
        now = datetime.now(timezone.utc)
        artifacts = [_artifact("research-100", 100, now - timedelta(days=1))]

        record = {
            "qid": 43613,
            "run_id": "100",
            "timestamp": "2026-06-20T10:00:00Z",
            "research_text": "Downloaded research",
            "providers_used": ["asknews"],
        }

        with (
            mock.patch("scripts.download_research.verify_gh_cli"),
            mock.patch("scripts.download_research.list_research_artifacts", return_value=artifacts),
            mock.patch("scripts.download_research.download_artifact", return_value=[100]),
            mock.patch("scripts.download_research.load_jsonl_records", return_value=[record]),
        ):
            records = download_research_artifacts(repo="repo", since_days=0)

        deduped = deduplicate_records(records)
        archive_dir = tmp_path / "archive"
        build_archive(deduped, archive_dir)

        manifest = json.loads((archive_dir / "manifest.json").read_text())
        assert "43613" in manifest
        assert manifest["43613"]["latest_timestamp"] == "2026-06-20T10:00:00Z"
        assert manifest["43613"]["providers"] == ["asknews"]


class TestSyncResearchNoClobber:
    """Regression: `make sync_research`'s final build must contain BOTH artifacts AND backfill.

    The clobber bug: comment-backfill records (run_id `comment-<id>`) live in the backfill
    dir, but freshly DOWNLOADED artifact records (numeric run_id) live only in the build's
    in-memory list — never written to the backfill dir. A standalone `--skip-download`
    rebuild therefore drops every artifact run_id. The fix is a single `main()` build that
    downloads artifacts AND loads backfill, so the archive ends up with both. These tests
    lock that invariant: every live artifact's run_id survives into the rebuilt archive.

    All `gh`/subprocess access is mocked — no live GitHub calls.
    """

    @staticmethod
    def _write_backfill(backfill_dir: Path) -> None:
        """Populate the backfill dir as `backfill_research_from_comments.py` would.

        Comment records use a `comment-<id>` run_id namespace, distinct from the numeric
        workflow-run ids artifacts carry, so the two never collide in dedup by (qid, run_id).
        Includes one record for qid 43613 (an artifact also covers it) and one comment-only
        qid 90000 (no artifact) to prove both sources survive.
        """
        backfill_dir.mkdir(parents=True, exist_ok=True)
        comment_records = [
            {
                "qid": 43613,
                "run_id": "comment-855787",
                "timestamp": "2026-05-19T13:58:49Z",
                "research_text": "Older comment-sourced research for 43613",
                "providers_used": ["asknews"],
            },
            {
                "qid": 90000,
                "run_id": "comment-999000",
                "timestamp": "2026-05-19T14:00:00Z",
                "research_text": "Comment-only research for 90000",
                "providers_used": ["native_search"],
            },
        ]
        with open(backfill_dir / "comments_backfill.jsonl", "w") as f:
            for r in comment_records:
                f.write(json.dumps(r) + "\n")

    def test_full_sync_build_keeps_artifacts_and_backfill(self, tmp_path: Path) -> None:
        """main() (download + load backfill + build) preserves artifact run_ids AND comments."""
        backfill_dir = tmp_path / "backfill"
        output_dir = tmp_path / "archive"
        self._write_backfill(backfill_dir)

        now = datetime.now(timezone.utc)
        artifacts = [_artifact("research-28256121132", 28256121132, now - timedelta(days=1))]
        artifact_record = {
            "qid": 43613,
            "run_id": "28256121132",  # numeric workflow-run id — must survive
            "timestamp": "2026-06-26T18:00:00Z",
            "research_text": "Fresh artifact research for 43613",
            "providers_used": ["asknews", "native_search"],
        }

        # Mock only the gh-touching pieces; let load_jsonl_records run for real so it
        # parses BOTH the downloaded artifact file and the on-disk comment backfill (the
        # earlier mistake of mocking load_jsonl_records globally hijacked backfill loading).
        def fake_download(run_id, repo, artifact_name, dest_dir):  # noqa: ANN001, ANN202
            run_dir = Path(dest_dir) / str(run_id)
            run_dir.mkdir(parents=True, exist_ok=True)
            jsonl = run_dir / f"{artifact_name}.jsonl"
            jsonl.write_text(json.dumps(artifact_record) + "\n")
            return [jsonl]

        argv = [
            "download_research.py",
            "--repo",
            "repo",
            "--backfill-dir",
            str(backfill_dir),
            "--output-dir",
            str(output_dir),
        ]
        with (
            mock.patch("sys.argv", argv),
            mock.patch("scripts.download_research.verify_gh_cli"),
            mock.patch("scripts.download_research.list_research_artifacts", return_value=artifacts),
            mock.patch("scripts.download_research.download_artifact", side_effect=fake_download),
        ):
            download_research_main()

        manifest = json.loads((output_dir / "manifest.json").read_text())

        # qid 43613 has BOTH the artifact version and the comment version (distinct run_ids).
        assert manifest["43613"]["versions_count"] == 2
        # latest is the fresh artifact (newer than the comment version) — no regression.
        assert manifest["43613"]["latest_timestamp"] == "2026-06-26T18:00:00Z"

        # The artifact run_id is present in the archive (this is what the clobber dropped).
        versions_43613 = [
            json.loads(line) for line in (output_dir / "by_qid" / "43613.jsonl").read_text().strip().splitlines()
        ]
        run_ids = {v["run_id"] for v in versions_43613}
        assert "28256121132" in run_ids  # artifact survived
        assert "comment-855787" in run_ids  # comment also survived

        # Comment-only qid (no artifact) is still present too.
        assert "90000" in manifest

    def test_skip_download_alone_drops_artifacts(self, tmp_path: Path) -> None:
        """Documents the clobber: --skip-download rebuilds from backfill ONLY (no artifacts).

        This is WHY the old third sync step clobbered — it ran exactly this. The fix is to
        not run a standalone --skip-download rebuild after the artifact download.
        """
        backfill_dir = tmp_path / "backfill"
        output_dir = tmp_path / "archive"
        self._write_backfill(backfill_dir)

        argv = [
            "download_research.py",
            "--skip-download",
            "--backfill-dir",
            str(backfill_dir),
            "--output-dir",
            str(output_dir),
        ]
        with mock.patch("sys.argv", argv):
            download_research_main()

        manifest = json.loads((output_dir / "manifest.json").read_text())
        # Only comment-sourced records — the artifact run_id is absent (the clobber).
        assert manifest["43613"]["versions_count"] == 1
        versions = [
            json.loads(line) for line in (output_dir / "by_qid" / "43613.jsonl").read_text().strip().splitlines()
        ]
        assert {v["run_id"] for v in versions} == {"comment-855787"}


class TestFullRoundTrip:
    """Test the complete lifecycle: write -> flush -> build archive -> load for backtest."""

    def test_write_flush_archive_load(self, tmp_path: Path) -> None:
        # --- Step 1: Writer records 3 questions ---
        writer = ResearchPersistenceWriter(run_mode="tournament", tournament_id="3672", run_id="gh-action-100")

        questions_data = [
            {
                "qid": 43613,
                "page_url": "https://www.metaculus.com/questions/43613/will-climate-target-be-met/",
                "question_text": "Will the 2026 climate agreement target be met?",
                "research_text": "## News Articles (AskNews)\nClimate research for 43613.",
                "providers_used": ["asknews", "native_search"],
                "gap_fill_used": True,
            },
            {
                "qid": 50001,
                "page_url": "https://www.metaculus.com/questions/50001/will-ai-achieve-agi/",
                "question_text": "Will AI achieve AGI by 2030?",
                "research_text": "## Web Research (Native Search)\nAI research for 50001.",
                "providers_used": ["native_search", "gemini_search"],
                "gap_fill_used": False,
            },
            {
                "qid": 38880,
                "page_url": "https://www.metaculus.com/c/diffusion-community/38880/some-question/",
                "question_text": "Will diffusion models surpass GANs?",
                "research_text": "## Web Research (Google Search via Gemini)\nDiffusion research for 38880.",
                "providers_used": ["gemini_search"],
                "gap_fill_used": False,
            },
        ]

        for q in questions_data:
            writer.record(**q)

        # --- Step 2: Flush to JSONL ---
        research_outputs_dir = tmp_path / "research_outputs"
        jsonl_path = writer.flush(output_dir=str(research_outputs_dir))
        assert jsonl_path is not None
        assert jsonl_path.exists()

        # --- Step 3: Read JSONL and build archive ---
        records = []
        with open(jsonl_path) as f:
            for line in f:
                records.append(json.loads(line.strip()))

        assert len(records) == 3

        archive_dir = tmp_path / "archive"
        build_archive(records, archive_dir)

        # Verify archive structure
        assert (archive_dir / "latest" / "43613.json").exists()
        assert (archive_dir / "latest" / "50001.json").exists()
        assert (archive_dir / "latest" / "38880.json").exists()
        assert (archive_dir / "manifest.json").exists()

        # --- Step 4: Load archive for backtest ---
        questions = [_make_question(qid) for qid in [43613, 50001, 38880]]
        cache = _load_research_from_archive(str(archive_dir / "latest"), questions)

        assert len(cache) == 3
        assert cache[43613] == "## News Articles (AskNews)\nClimate research for 43613."
        assert cache[50001] == "## Web Research (Native Search)\nAI research for 50001."
        assert cache[38880] == "## Web Research (Google Search via Gemini)\nDiffusion research for 38880."

    def test_round_trip_preserves_long_multiline_research(self, tmp_path: Path) -> None:
        """Verify that the full realistic research text survives the entire pipeline."""
        writer = ResearchPersistenceWriter(run_mode="tournament", tournament_id="3672", run_id="test-long")

        writer.record(
            qid=43613,
            page_url="https://www.metaculus.com/questions/43613/",
            question_text="Climate question",
            research_text=REALISTIC_RESEARCH_TEXT,
            providers_used=["asknews", "native_search"],
            gap_fill_used=True,
        )

        jsonl_path = writer.flush(output_dir=str(tmp_path / "outputs"))
        assert jsonl_path is not None

        records = []
        with open(jsonl_path) as f:
            for line in f:
                records.append(json.loads(line.strip()))

        archive_dir = tmp_path / "archive"
        build_archive(records, archive_dir)

        questions = [_make_question(43613)]
        cache = _load_research_from_archive(str(archive_dir / "latest"), questions)

        assert cache[43613] == REALISTIC_RESEARCH_TEXT
