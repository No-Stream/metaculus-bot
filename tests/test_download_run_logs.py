"""Tests for the run-log downloader (scripts/download_run_logs.py).

Network-free: exercises the pure artifact-filtering / workflow-inference logic and
the local log->HarvestedRun harvest against temp files. The GitHub enumeration +
download path lives in the shared core (scripts.gha_artifacts) — the resilience tests
here monkeypatch that module's seams; the enumeration internals are tested there.
"""

import json
import logging
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scripts.download_run_logs as dl
import scripts.gha_artifacts as gha
from scripts.download_run_logs import (
    RUN_LOG_ARTIFACT_PREFIXES,
    filter_run_log_artifacts,
    harvest_run_logs_from_dir,
    infer_workflow,
    workflow_slug_from_path,
)
from scripts.telemetry.archive import (
    RUNS_MANIFEST_FILE,
    load_marker_records,
    load_run_manifest,
    merge_and_write,
)

EXTRACTION_LINE = (
    "2026-07-17 14:23:01,123 - metaculus_bot.value_extraction - INFO - "
    "EXTRACTION_RUNG: question=12345 model=openai/gpt-5.6-sol qtype=binary rung=block block_present=True"
)
GHOST_LINE = (
    "2026-07-17 14:25:11,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GHOST_FORECAST: qtype=binary summary=posterior_prob=0.42"
)
# The newest spec additions, in their emitted shapes: q45085's 405-closed publish
# failure, and the additive skip-reason companion carried in the published comment.
PUBLISH_HARDENING_LINE = (
    "2026-08-24 09:00:00,000 - metaculus_bot.publish_hardening - WARNING - "
    "PUBLISH_HARDENING: _post_question_prediction attempt 2/2 failed "
    '(HTTPError: Error while posting prediction: Status code: 405. Response: {"error":"closed"})'
)
STACKER_SKIP_REASON_LINE = "<!-- STACKER_SKIP_REASON=single_forecaster -->"


class TestWorkflowSlugFromPath:
    def test_prod_workflows(self):
        assert workflow_slug_from_path(".github/workflows/run_bot_on_tournament.yaml") == "tournament"
        assert workflow_slug_from_path(".github/workflows/run_bot_on_metaculus_cup.yaml") == "metaculus_cup"
        assert workflow_slug_from_path(".github/workflows/run_bot_on_minibench.yaml") == "minibench"

    def test_test_bot(self):
        assert workflow_slug_from_path(".github/workflows/test_bot.yaml") == "test_bot"

    def test_unknown_path_falls_back_to_stem(self):
        assert workflow_slug_from_path(".github/workflows/some_other.yaml") == "some_other"


class TestInferWorkflow:
    def test_uses_workflow_map_when_present(self):
        wf_map = {123: "tournament"}
        assert infer_workflow("research-123", 123, wf_map) == "tournament"

    def test_logs_prefix_defaults_to_test_bot(self):
        assert infer_workflow("logs-999", 999, {}) == "test_bot"

    def test_research_prefix_without_map_is_unknown(self):
        assert infer_workflow("research-999", 999, {}) == "unknown"


def _seed_manifest(archive_dir: Path, entries: dict[int, str]) -> None:
    """Write a minimal run manifest mapping each run_id to a workflow slug."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"run_id": str(run_id), "workflow": workflow, "artifact": f"research-{run_id}"})
        for run_id, workflow in entries.items()
    ]
    (archive_dir / RUNS_MANIFEST_FILE).write_text("\n".join(lines) + "\n")


class TestResolveWorkflowMap:
    """The resolved map layers GitHub's fresh window OVER the archive-derived map (see resolve_workflow_map)."""

    def test_archive_fills_runs_github_no_longer_returns(self, tmp_path: Path, monkeypatch):
        _seed_manifest(tmp_path, {100: "tournament"})
        monkeypatch.setattr(dl, "build_workflow_map", lambda repo: {200: "minibench"})

        resolved = dl.resolve_workflow_map("owner/repo", tmp_path)

        assert resolved == {100: "tournament", 200: "minibench"}
        # The downstream consequence the merge depends on: the aged-out run keeps its
        # archived slug instead of degrading to unknown via prefix inference.
        assert infer_workflow("research-100", 100, resolved) == "tournament"

    def test_github_wins_where_both_know_a_run(self, tmp_path: Path, monkeypatch):
        _seed_manifest(tmp_path, {100: "test_bot"})
        monkeypatch.setattr(dl, "build_workflow_map", lambda repo: {100: "tournament"})
        assert dl.resolve_workflow_map("owner/repo", tmp_path) == {100: "tournament"}

    def test_archived_unknown_never_enters_the_map(self, tmp_path: Path, monkeypatch):
        _seed_manifest(tmp_path, {100: "unknown"})
        monkeypatch.setattr(dl, "build_workflow_map", lambda repo: {})
        assert dl.resolve_workflow_map("owner/repo", tmp_path) == {}

    def test_from_store_never_calls_github(self, tmp_path: Path, monkeypatch):
        _seed_manifest(tmp_path, {100: "tournament"})

        def _forbidden(repo: str) -> dict[int, str]:
            raise AssertionError("from_store must not hit GitHub")

        monkeypatch.setattr(dl, "build_workflow_map", _forbidden)
        assert dl.resolve_workflow_map("owner/repo", tmp_path, from_store=True) == {100: "tournament"}


class TestFilterRunLogArtifacts:
    def test_keeps_both_prefixes_and_splits_expired(self):
        artifacts = [
            {"name": "research-1", "run_id": 1, "expired": False},
            {"name": "logs-2", "run_id": 2, "expired": False},
            {"name": "research-3", "run_id": 3, "expired": True},
            {"name": "some-other-artifact", "run_id": 4, "expired": False},
        ]
        live, expired = filter_run_log_artifacts(artifacts)
        assert {a["name"] for a in live} == {"research-1", "logs-2"}
        assert {a["name"] for a in expired} == {"research-3"}

    def test_prefixes_constant_covers_both_families(self):
        assert "research-" in RUN_LOG_ARTIFACT_PREFIXES
        assert "logs-" in RUN_LOG_ARTIFACT_PREFIXES


class TestHarvestRunLogsFromDir:
    def test_parses_logs_and_builds_run(self, tmp_path: Path):
        run_logs = tmp_path / "run_logs"
        run_logs.mkdir()
        (run_logs / "run_500_a.log").write_text(EXTRACTION_LINE + "\n")
        (run_logs / "run_500_b.log").write_text(GHOST_LINE + "\n")

        run = harvest_run_logs_from_dir(
            tmp_path, run_id="500", workflow="tournament", artifact="research-500", run_date="2026-07-17T00:00:00Z"
        )
        assert run is not None
        assert run.run_id == "500"
        assert run.workflow == "tournament"
        assert len(run.records["extraction_rung"]) == 1
        assert len(run.records["ghost_forecast"]) == 1
        assert run.records["ghost_forecast"][0]["qid"] == 38975
        assert run.log_files == ["run_500_a.log", "run_500_b.log"]

    def test_seq_is_contiguous_across_files(self, tmp_path: Path):
        run_logs = tmp_path / "run_logs"
        run_logs.mkdir()
        (run_logs / "run_1.log").write_text(EXTRACTION_LINE + "\n")
        (run_logs / "run_2.log").write_text(EXTRACTION_LINE + "\n")
        run = harvest_run_logs_from_dir(
            tmp_path, run_id="500", workflow="tournament", artifact="research-500", run_date="2026-07-17T00:00:00Z"
        )
        assert run is not None
        seqs = sorted(r["seq"] for r in run.records["extraction_rung"])
        assert seqs == [0, 1], "seq must be contiguous across a run's multiple log files"

    def test_no_log_files_returns_none(self, tmp_path: Path):
        # research-* artifacts contain research_outputs/ but sometimes no run_logs/.
        (tmp_path / "research_outputs").mkdir()
        run = harvest_run_logs_from_dir(
            tmp_path, run_id="500", workflow="tournament", artifact="research-500", run_date="2026-07-17T00:00:00Z"
        )
        assert run is None


class TestNewMarkerSpecsReachTheArchive:
    """A newly-added MarkerSpec must survive the whole durable path: run log ->
    ``harvest_run_logs_from_dir`` -> ``merge_and_write`` -> its own JSONL.

    The archive derives its file set from ``MARKER_SPECS``, so this is what makes a
    spec-only change (2026-08-24's PUBLISH_HARDENING and STACKER_SKIP_REASON) actually
    queryable after the 90-day GHA log expiry. Parsing is tested per-spec in
    ``tests/test_telemetry_markers.py``; this pins the plumbing around it.
    """

    def test_publish_hardening_and_skip_reason_round_trip_to_jsonl(self, tmp_path: Path):
        run_logs = tmp_path / "run_logs"
        run_logs.mkdir()
        (run_logs / "run.log").write_text(
            "\n".join([EXTRACTION_LINE, PUBLISH_HARDENING_LINE, STACKER_SKIP_REASON_LINE]) + "\n"
        )

        run = harvest_run_logs_from_dir(
            tmp_path, run_id="900", workflow="tournament", artifact="research-900", run_date="2026-08-24T00:00:00Z"
        )
        assert run is not None
        assert [r["method"] for r in run.records["publish_hardening"]] == ["_post_question_prediction"]
        assert [r["reason"] for r in run.records["stacker_skip_reason"]] == ["single_forecaster"]

        archive_dir = tmp_path / "archive"
        totals = merge_and_write(archive_dir, [run])
        assert totals["publish_hardening"] == 1
        assert totals["stacker_skip_reason"] == 1
        archived = load_marker_records(archive_dir, "publish_hardening")
        assert len(archived) == 1
        assert archived[0]["run_id"] == "900"
        assert archived[0]["workflow"] == "tournament"
        assert archived[0]["error_type"] == "HTTPError"


class TestDownloadTimeoutResilience:
    """A single slow `gh` call must not sink the whole weekly harvest: a per-artifact
    download timeout is skipped (loop continues), and the run-enumeration timeout falls
    back to the archive-derived workflow map instead of raising.
    """

    def test_per_artifact_timeout_returns_none_not_raises(self, tmp_path: Path, monkeypatch, caplog):
        # ``_download_artifact_to`` now lives in the shared core (scripts.gha_artifacts);
        # monkeypatch its subprocess there.
        def _raise_timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd=["gh", "run", "download"], timeout=gha.ARTIFACT_DOWNLOAD_TIMEOUT_S)

        monkeypatch.setattr(gha.subprocess, "run", _raise_timeout)
        with caplog.at_level(logging.WARNING):
            result = gha._download_artifact_to(7, "owner/repo", "research-7", tmp_path)
        assert result is None
        assert any("Timed out" in r.getMessage() for r in caplog.records)

    def test_workflow_map_timeout_returns_empty_map(self, monkeypatch, caplog):
        def _raise_timeout(*_args, **_kwargs):
            raise subprocess.TimeoutExpired(cmd=["gh", "api"], timeout=dl.GH_API_TIMEOUT_S)

        monkeypatch.setattr(dl.subprocess, "run", _raise_timeout)
        with caplog.at_level(logging.WARNING):
            result = dl.build_workflow_map("owner/repo")
        assert result == {}
        assert any("timed out" in r.getMessage().lower() for r in caplog.records)

    def test_workflow_map_gh_failure_returns_empty_map(self, monkeypatch, caplog):
        # A failed `gh api` (auth expired, rate limit, network) must degrade to the
        # empty fresh map — the resolver then serves the archive-derived map alone —
        # rather than raising out of the weekly harvest.
        failed = subprocess.CompletedProcess(args=["gh", "api"], returncode=1, stdout="", stderr="HTTP 401\n")
        monkeypatch.setattr(dl.subprocess, "run", lambda *_a, **_k: failed)
        with caplog.at_level(logging.WARNING):
            result = dl.build_workflow_map("owner/repo")
        assert result == {}
        messages = [r.getMessage() for r in caplog.records]
        assert any("HTTP 401" in m and "archive-derived map" in m for m in messages)

    def test_workflow_map_skips_unparseable_and_incomplete_rows(self, monkeypatch):
        # `gh api --jq` streams one JSON object per line; a truncated or field-less row
        # must be skipped instead of aborting the whole map.
        stdout = "\n".join(
            [
                '{"id": 1, "path": ".github/workflows/run_bot_on_tournament.yaml"}',
                "{not json",
                '{"id": 2}',  # no path
                '{"path": ".github/workflows/test_bot.yaml"}',  # no id
                "",
            ]
        )
        ok = subprocess.CompletedProcess(args=["gh", "api"], returncode=0, stdout=stdout, stderr="")
        monkeypatch.setattr(dl.subprocess, "run", lambda *_a, **_k: ok)
        assert dl.build_workflow_map("owner/repo") == {1: "tournament"}

    def test_enumeration_stays_bounded_by_a_created_cutoff(self, monkeypatch):
        # The walk must carry a `created:>=` filter: unbounded --paginate crawls the
        # repo's entire run history and stalls the harvest for minutes. The window is a
        # module constant now (the 1000-item pagination cap dominates it anyway), so the
        # cutoff the command asks for has to come from that constant.
        seen: dict[str, list[str]] = {}

        def _capture(cmd, **_kwargs):
            seen["cmd"] = list(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        monkeypatch.setattr(dl.subprocess, "run", _capture)
        dl.build_workflow_map("owner/repo")

        expected_cutoff = (datetime.now(timezone.utc) - timedelta(days=dl._RUNS_ENUMERATION_WINDOW_DAYS)).strftime(
            "%Y-%m-%d"
        )
        assert f"created=>={expected_cutoff}" in seen["cmd"]
        assert "/repos/owner/repo/actions/runs" in seen["cmd"]

    def test_timed_out_artifact_does_not_abort_harvest(self, tmp_path: Path, monkeypatch):
        # Two live artifacts; the first "times out" (download returns None), the second
        # succeeds. The good one must still be harvested — the loop skips, never aborts.
        artifacts = [
            {"name": "research-1", "run_id": 1, "expired": False, "created_at": "2026-07-18T00:00:00Z"},
            {"name": "research-2", "run_id": 2, "expired": False, "created_at": "2026-07-19T00:00:00Z"},
        ]
        # Enumeration + download go through the shared core now; monkeypatch those seams
        # in scripts.gha_artifacts. build_workflow_map + merge_and_write stay in dl.
        monkeypatch.setattr(gha, "verify_gh_cli", lambda: None)
        monkeypatch.setattr(gha, "list_research_artifacts", lambda repo: artifacts)
        monkeypatch.setattr(dl, "build_workflow_map", lambda repo: {})
        # Patch the archive writer so this test is independent of the archive module.
        monkeypatch.setattr(dl, "merge_and_write", lambda archive_dir, runs: {})

        def _fake_download(run_id, repo, name, dest_dir):
            if run_id == 1:
                return None  # simulates a TimeoutExpired-skipped artifact
            log_dir = dest_dir / str(run_id) / "run_logs"
            log_dir.mkdir(parents=True)
            (log_dir / "run.log").write_text(EXTRACTION_LINE + "\n")
            return dest_dir / str(run_id)

        monkeypatch.setattr(gha, "_download_artifact_to", _fake_download)

        # store_dir is explicit: without it the harvest persists these fixture artifacts
        # into the operator's real backtests/gha_artifact_store/ (the conftest redirect is
        # the backstop, this is the local statement of intent).
        _totals, runs, _expired = dl.download_and_harvest("owner/repo", 0, tmp_path, store_dir=tmp_path / "store")
        assert {r.run_id for r in runs} == {"2"}, "the surviving artifact must still be harvested"
        assert len(runs[0].records["extraction_rung"]) == 1


class TestDownloadAndHarvestKeepsArchivedLabels:
    def test_online_harvest_keeps_archived_label_for_aged_out_run(self, tmp_path: Path, monkeypatch):
        """A run absent from GitHub's window keeps its archived label through the real merge.

        ``merge_and_write`` is deliberately NOT patched: the assertion covers the whole
        online path, resolver through manifest round-trip, in the standalone driver
        (``make sync_telemetry`` runs it directly, without going through sync_all).
        """
        archive_dir = tmp_path / "archive"
        _seed_manifest(archive_dir, {1: "tournament"})
        artifacts = [{"name": "research-1", "run_id": 1, "expired": False, "created_at": "2026-06-01T00:00:00Z"}]
        monkeypatch.setattr(gha, "verify_gh_cli", lambda: None)
        monkeypatch.setattr(gha, "list_research_artifacts", lambda repo: artifacts)
        monkeypatch.setattr(dl, "build_workflow_map", lambda repo: {})

        def _download(run_id, repo, name, dest_dir):
            log_dir = dest_dir / str(run_id) / "run_logs"
            log_dir.mkdir(parents=True)
            (log_dir / "run.log").write_text(EXTRACTION_LINE + "\n")
            return dest_dir / str(run_id)

        monkeypatch.setattr(gha, "_download_artifact_to", _download)

        _totals, runs, _expired = dl.download_and_harvest("owner/repo", 0, archive_dir, store_dir=tmp_path / "store")

        assert [r.workflow for r in runs] == ["tournament"]
        manifest = load_run_manifest(archive_dir)
        assert {r["run_id"]: r["workflow"] for r in manifest} == {"1": "tournament"}
