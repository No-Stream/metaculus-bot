"""Tests for the single-pass sync driver (scripts/sync_all.py).

Network-free: mocks the shared-core enumeration + per-artifact download seams
(scripts.gha_artifacts) and the workflow-map lookup, then drives ``run_sync`` over a
fake set of downloaded run dirs. Asserts the single-pass invariants the driver exists
to guarantee: ONE enumeration, each unique artifact downloaded ONCE (even though three
harvesters read it), all three archives written from that one pass, and a failed
per-artifact download skipping that run for every harvester without aborting.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TypedDict
from unittest import mock

import scripts.sync_all as sync_all
from scripts.telemetry.archive import load_marker_records, load_run_manifest

# A valid EXTRACTION_RUNG log line (mirrors the run-log tests) so the telemetry harvest
# produces a real marker record from a fake run's run_logs/*.log.
EXTRACTION_LINE = (
    "2026-07-17 14:23:01,123 - metaculus_bot.value_extraction - INFO - "
    "EXTRACTION_RUNG: question=12345 model=openai/gpt-5.6-sol qtype=binary rung=block block_present=True"
)


def _artifact(name: str, run_id: int, created: datetime, expired: bool = False) -> dict:
    """Build an artifact object as emitted by ``list_research_artifacts``'s --jq projection."""
    return {
        "id": run_id * 10,
        "name": name,
        "created_at": created.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": (created + timedelta(days=90)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expired": expired,
        "size_in_bytes": 1234,
        "run_id": run_id,
    }


def _write_research_run_dir(dest_dir: Path, run_id: int, *, qid: int) -> Path:
    """A prod research-* run dir: research_outputs/ + run_logs/ (log marker + raw-research log)."""
    run_dir = Path(dest_dir) / str(run_id)
    outputs = run_dir / "research_outputs"
    logs = run_dir / "run_logs"
    outputs.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    (outputs / f"research_{run_id}.jsonl").write_text(
        json.dumps(
            {"qid": qid, "run_id": str(run_id), "timestamp": "2026-07-17T00:00:00Z", "providers_used": ["asknews"]}
        )
        + "\n"
    )
    (logs / f"run_{run_id}.log").write_text(EXTRACTION_LINE + "\n")
    (logs / f"raw_research_{run_id}.jsonl").write_text(
        json.dumps({"qid": qid, "provider": "asknews", "phase": "hot", "fetched_at": "2026-07-17T00:00:00Z"}) + "\n"
    )
    return run_dir


def _write_logs_run_dir(dest_dir: Path, run_id: int) -> Path:
    """A test_bot logs-* run dir: run_logs/ only (no research_outputs/)."""
    run_dir = Path(dest_dir) / str(run_id)
    logs = run_dir / "run_logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"run_{run_id}.log").write_text(EXTRACTION_LINE + "\n")
    return run_dir


class SyncDirs(TypedDict):
    """``run_sync``'s directory kwargs. Typed so ``**dirs`` doesn't widen to ``Path``."""

    research_dir: Path
    backfill_dir: Path
    telemetry_dir: Path
    raw_dir: Path
    store_dir: Path


def _dirs(tmp_path: Path) -> SyncDirs:
    return {
        "research_dir": tmp_path / "research_archive",
        "backfill_dir": tmp_path / "research_archive" / "backfill",  # absent -> empty backfill
        "telemetry_dir": tmp_path / "telemetry_archive",
        "raw_dir": tmp_path / "research_archive" / "raw",
        "store_dir": tmp_path / "gha_artifact_store",
    }


class TestSinglePassDriver:
    def test_enumerates_once_downloads_each_run_once_writes_all_three(self, tmp_path: Path) -> None:
        """One enumeration; each unique run downloaded once; research + telemetry + raw all written.

        run 100 is a prod research-* run feeding ALL THREE harvesters from a single
        downloaded dir; run 300 is a legacy pre-2026-08-03 test_bot logs-* run, which carried
        only run_logs/ (telemetry only). The prod run
        must appear exactly once in the download calls despite three consumers reading it.
        """
        now = datetime.now(timezone.utc)
        artifacts = [
            _artifact("research-100", 100, now - timedelta(days=1)),
            _artifact("logs-300", 300, now - timedelta(days=2)),
            _artifact("benchmark-results", 400, now - timedelta(days=1)),  # not a run-log family
        ]

        def fake_download(run_id, repo, artifact_name, dest_dir):  # noqa: ANN001, ANN202
            if run_id == 100:
                return _write_research_run_dir(dest_dir, 100, qid=43613)
            if run_id == 300:
                return _write_logs_run_dir(dest_dir, 300)
            return None

        dirs = _dirs(tmp_path)
        with (
            mock.patch("scripts.gha_artifacts.verify_gh_cli"),
            mock.patch("scripts.gha_artifacts.list_research_artifacts", return_value=artifacts) as list_mock,
            mock.patch("scripts.gha_artifacts._download_artifact_to", side_effect=fake_download) as dl_mock,
            mock.patch("scripts.download_run_logs.build_workflow_map", return_value={}),
        ):
            summary = sync_all.run_sync("repo", 0, **dirs)

        # (a) exactly ONE enumeration for the whole sync (not one per archive).
        assert list_mock.call_count == 1

        # (b) each unique run downloaded exactly once; the benchmark artifact is out of family.
        downloaded_run_ids = [c.args[0] for c in dl_mock.call_args_list]
        assert sorted(downloaded_run_ids) == [100, 300]
        assert downloaded_run_ids.count(100) == 1, "the prod run feeds 3 harvesters but is downloaded once"

        # (c) all three archives written from the one pass.
        # Research: research-100 contributed qid 43613.
        manifest = json.loads((dirs["research_dir"] / "manifest.json").read_text())
        assert "43613" in manifest
        # Telemetry: both runs carried an EXTRACTION_RUNG marker.
        extraction = load_marker_records(dirs["telemetry_dir"], "extraction_rung")
        assert {r["run_id"] for r in extraction} == {"100", "300"}
        # Raw research: only the prod run carried a raw_research log.
        assert (dirs["raw_dir"] / "100.jsonl").exists()
        assert not (dirs["raw_dir"] / "300.jsonl").exists()

        assert summary.research_questions == 1
        assert summary.telemetry_runs == 2

    def test_failed_download_skips_run_for_all_harvesters_but_does_not_abort(self, tmp_path: Path) -> None:
        """A per-artifact download that returns None is skipped for research + telemetry + raw.

        The surviving run must still land in all three archives; the failed run in none —
        and the whole sync completes without raising.
        """
        now = datetime.now(timezone.utc)
        artifacts = [
            _artifact("research-100", 100, now - timedelta(days=2)),  # download "fails"
            _artifact("research-200", 200, now - timedelta(days=1)),  # succeeds
        ]

        def fake_download(run_id, repo, artifact_name, dest_dir):  # noqa: ANN001, ANN202
            if run_id == 100:
                return None  # simulate a TimeoutExpired-skipped artifact
            return _write_research_run_dir(dest_dir, 200, qid=50001)

        dirs = _dirs(tmp_path)
        with (
            mock.patch("scripts.gha_artifacts.verify_gh_cli"),
            mock.patch("scripts.gha_artifacts.list_research_artifacts", return_value=artifacts),
            mock.patch("scripts.gha_artifacts._download_artifact_to", side_effect=fake_download),
            mock.patch("scripts.download_run_logs.build_workflow_map", return_value={}),
        ):
            summary = sync_all.run_sync("repo", 0, **dirs)

        # Only the surviving run is represented across all three archives.
        manifest = json.loads((dirs["research_dir"] / "manifest.json").read_text())
        assert set(manifest) == {"50001"}
        extraction = load_marker_records(dirs["telemetry_dir"], "extraction_rung")
        assert {r["run_id"] for r in extraction} == {"200"}
        assert (dirs["raw_dir"] / "200.jsonl").exists()
        assert not (dirs["raw_dir"] / "100.jsonl").exists()

        assert summary.research_questions == 1
        assert summary.telemetry_runs == 1

    def test_expired_artifacts_reported_and_split_by_family(self, tmp_path: Path) -> None:
        """Expired run-log artifacts are surfaced in the summary, split research-* vs logs-*."""
        now = datetime.now(timezone.utc)
        artifacts = [
            _artifact("research-100", 100, now - timedelta(days=1)),
            _artifact("research-900", 900, now - timedelta(days=95), expired=True),
            _artifact("logs-901", 901, now - timedelta(days=96), expired=True),
        ]

        def fake_download(run_id, repo, artifact_name, dest_dir):  # noqa: ANN001, ANN202
            return _write_research_run_dir(dest_dir, 100, qid=1)

        dirs = _dirs(tmp_path)
        with (
            mock.patch("scripts.gha_artifacts.verify_gh_cli"),
            mock.patch("scripts.gha_artifacts.list_research_artifacts", return_value=artifacts),
            mock.patch("scripts.gha_artifacts._download_artifact_to", side_effect=fake_download) as dl_mock,
            mock.patch("scripts.download_run_logs.build_workflow_map", return_value={}),
        ):
            summary = sync_all.run_sync("repo", 0, **dirs)

        # Expired artifacts are never downloaded, only reported.
        assert [c.args[0] for c in dl_mock.call_args_list] == [100]
        assert len(summary.expired) == 2
        research_expired, logs_expired = sync_all._expired_by_family(summary.expired)
        assert (research_expired, logs_expired) == (1, 1)


class TestOfflineReharvestFromTheStore:
    """``--from-store`` rebuilds all three archives off local disk, with no network at all.

    This is what makes an ingest fix cheap: the artifacts are already persisted, so the
    corrected parser re-runs over the same bytes for free instead of re-downloading (and
    instead of being unable to, once the 90-day retention has passed).
    """

    def _seed_store(self, tmp_path: Path, dirs: SyncDirs) -> None:
        """Run one ONLINE sync so the store holds two artifacts, exactly as a real pull leaves it."""
        now = datetime.now(timezone.utc)
        artifacts = [
            _artifact("research-100", 100, now - timedelta(days=2)),
            _artifact("research-200", 200, now - timedelta(days=1)),
        ]

        def fake_download(run_id, repo, artifact_name, dest_dir):  # noqa: ANN001, ANN202
            return _write_research_run_dir(dest_dir, run_id, qid=40000 + run_id)

        with (
            mock.patch("scripts.gha_artifacts.verify_gh_cli"),
            mock.patch("scripts.gha_artifacts.list_research_artifacts", return_value=artifacts),
            mock.patch("scripts.gha_artifacts._download_artifact_to", side_effect=fake_download),
            mock.patch(
                "scripts.download_run_logs.build_workflow_map", return_value={100: "tournament", 200: "minibench"}
            ),
        ):
            sync_all.run_sync("repo", 0, **dirs)

    def test_rebuilds_every_archive_with_no_network_call(self, tmp_path: Path) -> None:
        dirs = _dirs(tmp_path)
        self._seed_store(tmp_path, dirs)

        # Wipe the three archives so only a genuine re-harvest off the store can restore them.
        for key in ("research_dir", "telemetry_dir", "raw_dir"):
            shutil.rmtree(dirs[key], ignore_errors=True)

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError(f"--from-store must not shell out: {args!r}")

        with (
            mock.patch("scripts.gha_artifacts.subprocess.run", side_effect=forbidden),
            mock.patch("scripts.download_run_logs.subprocess.run", side_effect=forbidden),
        ):
            summary = sync_all.run_sync("repo", 0, from_store=True, **dirs)

        manifest = json.loads((dirs["research_dir"] / "manifest.json").read_text())
        assert set(manifest) == {"40100", "40200"}
        assert {r["run_id"] for r in load_marker_records(dirs["telemetry_dir"], "extraction_rung")} == {"100", "200"}
        assert (dirs["raw_dir"] / "100.jsonl").exists() and (dirs["raw_dir"] / "200.jsonl").exists()
        assert (summary.research_questions, summary.telemetry_runs) == (2, 2)
        assert summary.expired == [], "the store cannot hold an expired artifact"

    def test_offline_reharvest_keeps_workflow_attribution(self, tmp_path: Path) -> None:
        """An offline re-harvest replays runs.jsonl's own attribution instead of degrading to ``unknown``."""
        dirs = _dirs(tmp_path)
        self._seed_store(tmp_path, dirs)

        sync_all.run_sync("repo", 0, from_store=True, **dirs)

        manifest = load_run_manifest(dirs["telemetry_dir"])
        assert {r["run_id"]: r["workflow"] for r in manifest} == {"100": "tournament", "200": "minibench"}

    def test_re_running_the_online_sync_downloads_nothing_new(self, tmp_path: Path) -> None:
        """A second online pull re-enumerates but re-fetches nothing already in the store."""
        dirs = _dirs(tmp_path)
        self._seed_store(tmp_path, dirs)
        now = datetime.now(timezone.utc)
        artifacts = [
            _artifact("research-100", 100, now - timedelta(days=2)),
            _artifact("research-200", 200, now - timedelta(days=1)),
        ]

        with (
            mock.patch("scripts.gha_artifacts.verify_gh_cli"),
            mock.patch("scripts.gha_artifacts.list_research_artifacts", return_value=artifacts),
            mock.patch("scripts.gha_artifacts._download_artifact_to") as dl_mock,
            mock.patch("scripts.download_run_logs.build_workflow_map", return_value={}),
        ):
            summary = sync_all.run_sync("repo", 0, **dirs)

        assert dl_mock.call_count == 0
        assert summary.research_questions == 2, "and the harvest still sees both runs, off the store"

    def test_online_resync_keeps_archived_workflow_labels_for_aged_out_runs(self, tmp_path: Path) -> None:
        """An ONLINE re-sync whose GitHub window misses an archived run (100) keeps that run's
        archived label, while GitHub wins for the in-window run (200)."""
        dirs = _dirs(tmp_path)
        self._seed_store(tmp_path, dirs)
        now = datetime.now(timezone.utc)
        artifacts = [
            _artifact("research-100", 100, now - timedelta(days=30)),  # aged out of the runs window
            _artifact("research-200", 200, now - timedelta(days=1)),  # still inside it
        ]

        with (
            mock.patch("scripts.gha_artifacts.verify_gh_cli"),
            mock.patch("scripts.gha_artifacts.list_research_artifacts", return_value=artifacts),
            mock.patch("scripts.gha_artifacts._download_artifact_to"),
            mock.patch("scripts.download_run_logs.build_workflow_map", return_value={200: "minibench"}),
        ):
            sync_all.run_sync("repo", 0, **dirs)

        manifest = load_run_manifest(dirs["telemetry_dir"])
        assert {r["run_id"]: r["workflow"] for r in manifest} == {"100": "tournament", "200": "minibench"}
