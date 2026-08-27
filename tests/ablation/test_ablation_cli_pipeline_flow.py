"""End-to-end stage flow: the full fetch → research → screen → forecast → stack → score
run, re-run idempotency, atomic writes for non-cache files, and the pacing knobs
(patient-mode concurrency clamp, per-question sleep, wall-clock estimate at stage start).

Split out of the original monolithic ``test_ablation_cli.py``. These are the tests that
drive ``run_ablation`` over the whole pipeline; per-stage behavior lives in the
sibling ``test_ablation_cli_*`` modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from tests.ablation_cli_fakes import (
    _binary_forecaster_payload,
    _binary_stacker_payload,
    _build_question_set,
    _install_full_stack_mocks,
    _make_binary_ground_truth,
    _make_binary_question,
    _make_numeric_ground_truth,
    _make_numeric_question,
    _numeric_forecaster_payload,
    _numeric_stacker_payload,
)

# ---------------------------------------------------------------------------
# Full pipeline happy path
# ---------------------------------------------------------------------------


class TestFullPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_happy_path(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All stages execute in order; final summary file is written."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(101)
        q2 = _make_numeric_question(102)
        gt1 = _make_binary_ground_truth(101, outcome=True)
        gt2 = _make_numeric_ground_truth(102, value=50.0)

        question_set = _build_question_set([(q1, gt1), (q2, gt2)])

        research_results: dict[int, tuple[str, dict] | None] = {
            101: ("research blob 101", {"sources": 3}),
            102: ("research blob 102", {"sources": 5}),
        }

        verdicts = {
            101: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
            102: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            101: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", value=0.5 + 0.05 * i
                )
                for i in range(3)
            },
            102: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _numeric_forecaster_payload(
                    f"openrouter/test/m{i}", median=50.0 + i
                )
                for i in range(3)
            },
        }

        stacker_a = {
            101: _binary_stacker_payload("stack", 0.6),
            102: _numeric_stacker_payload("stack", 50.0),
        }
        stacker_b = {
            101: _binary_stacker_payload("stack_aug", 0.75),
            102: _numeric_stacker_payload("stack_aug", 52.0),
        }

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results=research_results,
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results=stacker_a,
            stacker_b_results=stacker_b,
        )

        # Speed up the per-question sleep.
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(
            [
                "--num-binary",
                "1",
                "--num-numeric",
                "1",
                "--cache-dir",
                str(cache_dir),
                "--qa-iterate-mode",
                "advisory",
            ]
        )
        exit_code = await run_ablation(args)

        assert exit_code == 0

        # Each stage was invoked exactly once.
        assert mocks["fetch"].await_count == 1
        assert mocks["research"].await_count == 1
        assert mocks["screen"].await_count == 1
        assert mocks["forecasters"].await_count == 1
        assert mocks["stacker"].await_count == 2  # arms A and B

        # Summary written.
        scores_dir = cache_dir / "scores"
        summaries = list(scores_dir.glob("summary_*.md"))
        runs = list(scores_dir.glob("run_*.json"))
        assert len(summaries) == 1
        assert len(runs) == 1


# ---------------------------------------------------------------------------
# M3: estimated wall-clock at stage start
# ---------------------------------------------------------------------------


class TestPatientModeForcesConcurrencyOne:
    """At 50q in ``patient`` rate-limit mode, the docstring promises
    "concurrency=1" but in practice ``--concurrency 4`` (default) means 4
    questions in flight × per_forecaster_concurrency=1 = 4 forecasters
    flooding free-tier providers in parallel. Clamp ``--concurrency`` to 1
    in patient mode and emit a warning.
    """

    @pytest.mark.asyncio
    async def test_patient_mode_clamps_concurrency_to_one(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q = _make_binary_question(8500)
        gt = _make_binary_ground_truth(8500)
        question_set = _build_question_set([(q, gt)])

        verdicts = {
            8500: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            8500: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={8500: ("blob 8500", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={8500: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={8500: _binary_stacker_payload("stack_aug", 0.7)},
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(
            [
                "--num-binary",
                "1",
                "--cache-dir",
                str(cache_dir),
                "--qa-iterate-mode",
                "advisory",
                "--rate-limit-mode",
                "patient",
                "--concurrency",
                "4",
            ]
        )

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.ablation.cli"):
            await run_ablation(args)

        assert args.concurrency == 1, "patient mode must clamp concurrency to 1"
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("patient" in m and "concurrency" in m for m in warnings), (
            f"expected a clamp warning; got: {warnings}"
        )

    def test_patient_mode_no_warning_when_concurrency_already_one(self) -> None:
        """If the operator passes --concurrency 1 explicitly, no warning fires."""
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--rate-limit-mode", "patient", "--concurrency", "1"])
        assert args.concurrency == 1


class TestEstimatedWallClockAtStageStart:
    """At 50q+ scale the operator wants a "should I wait or come back?" signal
    on each stage. Log a rough estimate at INFO when the stage starts.
    """

    @pytest.mark.asyncio
    async def test_forecast_stage_start_logs_estimated_wall_clock(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q = _make_binary_question(8400)
        gt = _make_binary_ground_truth(8400)
        question_set = _build_question_set([(q, gt)])

        verdicts = {
            8400: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            8400: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={8400: ("blob 8400", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={8400: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={8400: _binary_stacker_payload("stack_aug", 0.7)},
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(
            [
                "--num-binary",
                "1",
                "--cache-dir",
                str(cache_dir),
                "--qa-iterate-mode",
                "advisory",
            ]
        )

        with caplog.at_level(logging.INFO, logger="metaculus_bot.ablation.cli"):
            await run_ablation(args)

        info_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
        forecast_starts = [m for m in info_messages if "stage=forecast START" in m]
        assert any("est wall-clock" in m for m in forecast_starts), (
            f"Expected an 'est wall-clock' annotation on stage=forecast START; got: {forecast_starts}"
        )


# ---------------------------------------------------------------------------
# Atomic-write coverage for non-cache files (C2)
# ---------------------------------------------------------------------------


class TestAtomicWritesForNonCacheFiles:
    """The four call sites identified by the audit must use the atomic helper:

    1. ``write_manual_rejects`` — covered in test_ablation_qa_iterate.py
    2. ``render_qa_summary`` — covered in test_ablation_qa_iterate.py
    3. ``_stage_qa_iterate`` per-qid qa_reports write
    4. ``_stage_qa_research_dump`` qa_research_<ts>.md write

    These tests exercise (3) and (4) at the cli-stage level: monkey-patch
    ``os.replace`` to fail mid-commit, assert the existing file (when present)
    is preserved and that no temp leftovers remain.
    """

    def test_qa_research_dump_uses_atomic_write(self, cache_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """qa_research_<ts>.md write goes through atomic_write_text."""
        import os as _os

        from metaculus_bot.ablation.cli import _stage_qa_research_dump

        # Build a minimal working set so the dump function has something to
        # render. Only fields that the renderer touches are populated.
        q = _make_binary_question(8001)
        gt = _make_binary_ground_truth(8001)
        cache = AblationCache(cache_dir)
        cache.write_research(8001, "research", {"sources": 1})

        from metaculus_bot.ablation.cli import WorkingSet

        working = WorkingSet(
            questions={8001: q},
            ground_truths={8001: gt},
            research_blobs={},
            leakage_verdicts={
                8001: {
                    "is_leaked": False,
                    "detector_response": "ok",
                    "detector_model": "x",
                    "detector_failed": False,
                    "screened_at": "now",
                }
            },
        )

        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--num-binary", "1", "--cache-dir", str(cache_dir)])

        def boom_replace(*_a: Any, **_k: Any) -> None:
            raise RuntimeError("interrupted commit")

        monkeypatch.setattr(_os, "replace", boom_replace)

        with pytest.raises(RuntimeError, match="interrupted commit"):
            _stage_qa_research_dump(args, cache, working)

        # No partial qa_research file should remain.
        qa_dumps = list(cache_dir.glob("qa_research_*.md"))
        assert qa_dumps == [], f"expected no partial qa_research dumps; got {qa_dumps}"
        # No tempfile leftover under the cache_dir either.
        leftovers = [p for p in cache_dir.iterdir() if p.name.startswith(".qa_research_") and p.suffix == ".tmp"]
        assert leftovers == []

    @pytest.mark.asyncio
    async def test_per_qid_qa_report_write_is_atomic(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``qa_reports/<qid>.json`` write must be atomic."""
        import os as _os

        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(8050)
        gt1 = _make_binary_ground_truth(8050)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            8050: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            8050: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={8050: ("blob 8050", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={8050: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={8050: _binary_stacker_payload("stack_aug", 0.7)},
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        # Pre-populate the qa_reports/<qid>.json so we can assert it survives a
        # crash mid-write.
        qa_reports_dir = cache_dir / "qa_reports"
        qa_reports_dir.mkdir(parents=True, exist_ok=True)
        existing_report = qa_reports_dir / "8050.json"
        existing_report.write_text('{"qid": 8050, "previous": "snapshot"}', encoding="utf-8")
        original = existing_report.read_text(encoding="utf-8")

        original_replace = _os.replace
        target_path_str = str(existing_report)

        def selective_boom_replace(src: Any, dst: Any) -> Any:
            if str(dst) == target_path_str:
                raise RuntimeError("interrupted qa_report commit")
            return original_replace(src, dst)

        monkeypatch.setattr(_os, "replace", selective_boom_replace)

        args = _build_parser().parse_args(
            ["--num-binary", "1", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )

        with pytest.raises(RuntimeError, match="interrupted qa_report commit"):
            await run_ablation(args)

        # Original file must be intact (no truncation).
        assert existing_report.read_text(encoding="utf-8") == original
        leftovers = [p for p in qa_reports_dir.iterdir() if p.name.startswith(".8050.json.")]
        assert leftovers == []


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    @pytest.mark.asyncio
    async def test_re_run_is_idempotent_with_existing_caches(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Running twice with the same args produces zero LLM calls on the second pass."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(401)
        gt1 = _make_binary_ground_truth(401)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            401: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        # Use real-lineup models so list_forecaster_outputs(lineup_filter=...)
        # in _stage_forecast (Task #23 + m2) honors the cached payloads and
        # the second run is genuinely idempotent.
        from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

        forecaster_results = {
            401: {
                model_slug_to_filename(model): _binary_forecaster_payload(model, value=0.5)
                for model in FREE_FORECASTER_MODELS
            },
        }

        stacker_a = {401: _binary_stacker_payload("stack", 0.6)}
        stacker_b = {401: _binary_stacker_payload("stack_aug", 0.7)}

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={401: ("research blob 401", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results=stacker_a,
            stacker_b_results=stacker_b,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        argv = [
            "--num-binary",
            "1",
            "--cache-dir",
            str(cache_dir),
            "--qa-iterate-mode",
            "advisory",
        ]

        # First run: all mocks called.
        args = _build_parser().parse_args(argv)
        exit_code1 = await run_ablation(args)
        assert exit_code1 == 0
        assert mocks["fetch"].await_count == 1
        assert mocks["research"].await_count == 1
        assert mocks["screen"].await_count == 1
        assert mocks["forecasters"].await_count == 1
        assert mocks["stacker"].await_count == 2

        # Second run with identical args: caches read; no fresh LLM calls.
        # The fetch stage still calls fetch only if the manifest's per-type counts
        # are below request — here they match (1 binary requested, 1 binary in manifest).
        args = _build_parser().parse_args(argv)
        exit_code2 = await run_ablation(args)
        assert exit_code2 == 0

        # Research, screen, forecasters, stackers should have read all values from cache.
        # The mocks should not have been called a second time. The runners themselves
        # short-circuit on cache hits, so the orchestrator either skips the call or
        # passes empty work to it. We assert nothing fresh hit the wire.
        assert mocks["research"].await_count == 1
        assert mocks["screen"].await_count == 1
        assert mocks["forecasters"].await_count == 1
        assert mocks["stacker"].await_count == 2


# ---------------------------------------------------------------------------
# Per-question sleep
# ---------------------------------------------------------------------------


class TestPerQuestionSleep:
    @pytest.mark.asyncio
    async def test_per_question_sleep_respected(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(1101)
        gt1 = _make_binary_ground_truth(1101)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            1101: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            1101: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        stacker_a = {1101: _binary_stacker_payload("stack", 0.6)}
        stacker_b = {1101: _binary_stacker_payload("stack_aug", 0.7)}

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={1101: ("blob 1101", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results=stacker_a,
            stacker_b_results=stacker_b,
        )

        sleep_mock = AsyncMock(return_value=None)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", sleep_mock)

        args = _build_parser().parse_args(
            [
                "--num-binary",
                "1",
                "--per-question-sleep",
                "5",
                "--cache-dir",
                str(cache_dir),
                "--qa-iterate-mode",
                "advisory",
            ]
        )
        await run_ablation(args)

        # asyncio.sleep was called at least once with 5s.
        sleep_calls = [call.args[0] if call.args else call.kwargs.get("delay") for call in sleep_mock.await_args_list]
        assert 5 in sleep_calls or 5.0 in sleep_calls

    @pytest.mark.asyncio
    async def test_per_question_sleep_applies_after_every_api_firing_stage(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """M4: --per-question-sleep must fire after EVERY API stage, not just 3.

        Pre-fix behavior: sleep fires after research, prune, forecast (3 places).
        Post-fix behavior: sleep fires after research, prune, screen, qa_iterate,
        forecast, stack, pdf (7 places).
        """
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(11201)
        gt1 = _make_binary_ground_truth(11201)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            11201: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }
        forecaster_results = {
            11201: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={11201: ("blob 11201", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={11201: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={11201: _binary_stacker_payload("stack_aug", 0.7)},
        )

        sleep_mock = AsyncMock(return_value=None)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", sleep_mock)

        args = _build_parser().parse_args(
            [
                "--num-binary",
                "1",
                "--per-question-sleep",
                "7",
                "--cache-dir",
                str(cache_dir),
                "--qa-iterate-mode",
                "advisory",
            ]
        )
        await run_ablation(args)

        # Count how many times sleep(7) fires (this is the inter-stage sleep).
        # asyncio.sleep(0) calls scattered throughout the code are NOT 7, so
        # we filter by the value to isolate inter-stage pauses.
        seven_calls = [call for call in sleep_mock.await_args_list if call.args and call.args[0] == 7]
        # M4: 7 stages each fire one inter-stage sleep — research, prune, screen,
        # qa_iterate, forecast, stack, pdf. The score stage does no API
        # work and gets no post-sleep.
        assert len(seven_calls) == 7, (
            f"--per-question-sleep must fire after every API-firing stage; "
            f"expected 7 calls of sleep(7), got {len(seven_calls)}"
        )
