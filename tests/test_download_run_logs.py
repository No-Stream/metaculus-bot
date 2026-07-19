"""Tests for the run-log downloader (scripts/download_run_logs.py).

Network-free: exercises the pure artifact-filtering / workflow-inference logic and
the local log->HarvestedRun harvest against temp files. The GitHub download path
mirrors scripts/download_research.py (shared ``list_research_artifacts`` /
``verify_gh_cli``) and is not re-tested here.
"""

from pathlib import Path

from scripts.download_run_logs import (
    RUN_LOG_ARTIFACT_PREFIXES,
    filter_run_log_artifacts,
    harvest_run_logs_from_dir,
    infer_workflow,
    workflow_slug_from_path,
)

EXTRACTION_LINE = (
    "2026-07-17 14:23:01,123 - metaculus_bot.value_extraction - INFO - "
    "EXTRACTION_RUNG: question=12345 model=openai/gpt-5.6-sol qtype=binary rung=block block_present=True"
)
GHOST_LINE = (
    "2026-07-17 14:25:11,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GHOST_FORECAST: qtype=binary summary=posterior_prob=0.42"
)


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
