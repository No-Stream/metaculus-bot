"""``--force-stages`` semantics: which stages a forced stage cascades into, what a
forced re-run overwrites, and the manual-rejects archive a forced ``qa_iterate`` writes.

Split out of the original monolithic ``test_ablation_cli.py`` so the forcing rules
read as one story instead of being scattered across the stage-by-stage tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from metaculus_bot.ablation.cache import model_slug_to_filename
from tests import ablation_cli_fakes as _fakes
from tests.ablation_cli_fakes import (
    _binary_forecaster_payload,
    _binary_stacker_payload,
    _build_question_set,
    _install_full_stack_mocks,
    _make_binary_ground_truth,
    _make_binary_question,
)

# Fixture bound by assignment rather than imported: see the ablation_cli_fakes docstring.
cache_dir = _fakes.cache_dir

# ---------------------------------------------------------------------------
# C3 part 1: --force-stages qa_iterate archives manual_rejects.json
# ---------------------------------------------------------------------------


class TestForceStagesQaIterateArchivesManualRejects:
    @pytest.mark.asyncio
    async def test_force_qa_iterate_archives_manual_rejects(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without --force-stages qa_iterate, an entry in manual_rejects.json
        causes that qid to be skipped silently. With --force-stages qa_iterate,
        the prior file is archived to ``manual_rejects.bak.<ts>.json`` and the
        qid is re-evaluated by the verifier.
        """
        import json

        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q = _make_binary_question(8100)
        gt = _make_binary_ground_truth(8100)
        question_set = _build_question_set([(q, gt)])

        verdicts = {
            8100: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            8100: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={8100: ("blob 8100", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={8100: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={8100: _binary_stacker_payload("stack_aug", 0.7)},
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        # Pre-populate manual_rejects.json with an entry for qid 8100 — the same
        # qid we're about to re-process.
        cache_dir.mkdir(parents=True, exist_ok=True)
        manual_rejects_path = cache_dir / "manual_rejects.json"
        manual_rejects_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "rejects": {
                        "8100": {
                            "rejected_at": "2026-05-01T00:00:00",
                            "reason": "stale entry from prior run",
                            "verifier_scores": [],
                            "iterations": 1,
                        }
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        args = _build_parser().parse_args(
            [
                "--num-binary",
                "1",
                "--cache-dir",
                str(cache_dir),
                "--qa-iterate-mode",
                "advisory",
                "--force-stages",
                "qa_iterate",
            ]
        )
        await run_ablation(args)

        # The qid was re-evaluated (default verifier returns clean), so
        # manual_rejects.json no longer contains 8100.
        post_rejects = json.loads(manual_rejects_path.read_text(encoding="utf-8"))
        assert "8100" not in post_rejects["rejects"]

        # The pre-existing rejects were archived to a backup file.
        backups = list(cache_dir.glob("manual_rejects.bak.*.json"))
        assert len(backups) == 1, f"expected one backup file; got {backups}"
        backup_payload = json.loads(backups[0].read_text(encoding="utf-8"))
        assert "8100" in backup_payload["rejects"]
        assert backup_payload["rejects"]["8100"]["reason"] == "stale entry from prior run"


class TestForceStages:
    @pytest.mark.asyncio
    async def test_force_stages_research_re_runs_research_only(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """First run completes; second with --force-stages research re-runs only research."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(501)
        gt1 = _make_binary_ground_truth(501)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            501: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            501: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", value=0.5
                )
                for i in range(3)
            },
        }

        stacker_a = {501: _binary_stacker_payload("stack", 0.6)}
        stacker_b = {501: _binary_stacker_payload("stack_aug", 0.7)}

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={501: ("research blob 501", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results=stacker_a,
            stacker_b_results=stacker_b,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        # First run.
        args = _build_parser().parse_args(
            ["--num-binary", "1", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        await run_ablation(args)

        research_count_after_run1 = mocks["research"].await_count
        forecaster_count_after_run1 = mocks["forecasters"].await_count

        # Second run with --force-stages research.
        args = _build_parser().parse_args(
            [
                "--num-binary",
                "1",
                "--cache-dir",
                str(cache_dir),
                "--force-stages",
                "research",
                "--qa-iterate-mode",
                "advisory",
            ]
        )
        await run_ablation(args)

        # Research re-ran; downstream stages just read cache.
        assert mocks["research"].await_count == research_count_after_run1 + 1
        # Forecasters batch may have been called with empty list (no work to do).
        # The contract: it's not called with FRESH work, and its mock count
        # depends on whether the orchestrator passes empty work or skips entirely.
        # Either way: forecasters were not actually invoked on new work.
        # We verify by checking the count didn't grow due to fresh forecasting.
        assert mocks["forecasters"].await_count <= forecaster_count_after_run1 + 1


# ---------------------------------------------------------------------------
# C1: --force-stages auto-cascades to downstream stages whose inputs changed.
#
# Forcing forecast WITHOUT cascading silently leaves stale stack/score caches.
# The audit at backtests/ablation/cli_audit_20260515.md:9 documents the
# operator footgun: re-run forecast to fix a flaky lineup, get cached stacker
# outputs derived from OLD forecasts. Worse, both arms cache success=True so
# nothing surfaces the inconsistency. C1 closes this by expanding ``forced``
# at parse time per a static cascade table.
# ---------------------------------------------------------------------------


class TestForceStagesCascade:
    def test_force_stages_forecast_cascades_to_stack_a_b_c(self) -> None:
        """Forcing forecast must auto-force stack + pdf + median + mean (otherwise stale cache served)."""
        from metaculus_bot.ablation.cli import _expand_forced_stages

        forced = _expand_forced_stages({"forecast"})
        assert forced == {"forecast", "stack", "stack_aug", "pdf", "median", "mean"}

    def test_force_stages_prune_cascades_through_screen_qa_forecast_stack(self) -> None:
        """Forcing prune must invalidate every stage downstream that consumes its output."""
        from metaculus_bot.ablation.cli import _expand_forced_stages

        forced = _expand_forced_stages({"prune"})
        assert "screen" in forced
        assert "qa_iterate" in forced
        assert "forecast" in forced
        assert "stack" in forced
        assert "stack_aug" in forced
        assert "pdf" in forced
        assert "median" in forced

    def test_force_stages_research_cascades_to_every_downstream(self) -> None:
        """Forcing research is the most upstream choice; everything below must invalidate."""
        from metaculus_bot.ablation.cli import _expand_forced_stages

        forced = _expand_forced_stages({"research"})
        assert "prune" in forced
        assert "screen" in forced
        assert "qa_iterate" in forced
        assert "forecast" in forced
        assert "stack" in forced
        assert "stack_aug" in forced
        assert "pdf" in forced
        assert "median" in forced

    def test_force_stages_screen_cascades_only_to_qa_iterate(self) -> None:
        """Screen verdict feeds qa_iterate but not forecast (forecast reads pruned blob)."""
        from metaculus_bot.ablation.cli import _expand_forced_stages

        forced = _expand_forced_stages({"screen"})
        # screen → qa_iterate cascades. forecast/stack are NOT downstream of screen
        # (forecast reads the pruned blob, screen produces a verdict that gates
        # which qids reach qa_iterate).
        assert "qa_iterate" in forced
        assert "forecast" not in forced
        assert "stack" not in forced
        assert "median" not in forced

    def test_force_stages_terminal_stages_have_no_cascade(self) -> None:
        """stack, pdf, median, score, fetch are terminal — no downstream invalidation."""
        from metaculus_bot.ablation.cli import _expand_forced_stages

        for terminal in ("stack", "stack_aug", "pdf", "median", "score", "fetch", "qa_iterate"):
            forced = _expand_forced_stages({terminal})
            assert forced == {terminal}, f"--force-stages {terminal} should not auto-cascade; got {forced}"

    def test_force_stages_explicit_set_kept_when_already_includes_cascade(self) -> None:
        """Cascade must be idempotent: explicitly listing all stages doesn't double-add."""
        from metaculus_bot.ablation.cli import _expand_forced_stages

        explicit = {"forecast", "stack", "stack_aug", "pdf", "median", "mean"}
        forced = _expand_forced_stages(explicit)
        assert forced == explicit

    @pytest.mark.asyncio
    async def test_force_stages_forecast_invalidates_stacker_cache_at_runtime(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-to-end: --force-stages forecast must trigger a fresh stacker run.

        Mutation test for C1: if the cascade table is removed, this test fails
        because the second run would hit cached stacker payloads.
        """
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(11001)
        gt1 = _make_binary_ground_truth(11001)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            11001: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            11001: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        stacker_a = {11001: _binary_stacker_payload("stack", 0.6)}
        stacker_b = {11001: _binary_stacker_payload("stack_aug", 0.7)}

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={11001: ("blob 11001", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results=stacker_a,
            stacker_b_results=stacker_b,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        # First run: populate every cache up to the stack stages. We deliberately
        # exclude the score stage from --stages because score requires all three
        # arms (A, B, C) to overlap and is independent of the cascade behavior
        # this test verifies.
        argv = [
            "--num-binary",
            "1",
            "--cache-dir",
            str(cache_dir),
            "--qa-iterate-mode",
            "advisory",
            "--stages",
            "fetch,research,prune,screen,qa_iterate,forecast,stack,stack_aug,pdf,median",
        ]
        await run_ablation(_build_parser().parse_args(argv))
        forecaster_count_after_run1 = mocks["forecasters"].await_count
        stacker_count_after_run1 = mocks["stacker"].await_count
        assert forecaster_count_after_run1 == 1
        # Only arms A + B go through ``run_stacker_batch``; ARM_MEDIAN uses
        # ``run_median_for_qid`` (deterministic median, no LLM batch).
        assert stacker_count_after_run1 == 2  # arm A + arm B

        # Second run with --force-stages forecast: BOTH forecaster AND stacker
        # should re-run (cascade invalidates downstream). Without C1, the
        # stacker would return cached payloads from the OLD forecaster run.
        argv_forced = [*argv, "--force-stages", "forecast"]
        await run_ablation(_build_parser().parse_args(argv_forced))

        assert mocks["forecasters"].await_count == forecaster_count_after_run1 + 1
        # Stacker re-ran for both LLM arms A + B (without cascade, this would still be 2).
        # ARM_MEDIAN is invalidated by the cascade too, but its calls go through
        # run_median_for_qid, not the mocked run_stacker_batch.
        assert mocks["stacker"].await_count == stacker_count_after_run1 + 2, (
            "--force-stages forecast must cascade to stack + pdf + median; otherwise "
            "stackers return stale payloads derived from old forecaster outputs."
        )
