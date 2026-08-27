"""Research and prune stages: the kwargs the CLI threads into
``run_gemini_research_for_qids``, and the prune stage's batching, cache reads and
failure handling.

Split out of the original monolithic ``test_ablation_cli.py``; the leakage screen and
``qa_iterate`` halves of the research phase live in
``test_ablation_cli_screen.py`` and ``test_ablation_cli_qa_iterate.py``.
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
# _stage_research kwarg threading
# ---------------------------------------------------------------------------


class TestStageResearchKwargs:
    @pytest.mark.asyncio
    async def test_stage_research_passes_gemini_model_to_runner(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--gemini-model`` flag flows through to ``run_gemini_research_for_qids`` as a kwarg."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(3001)
        gt1 = _make_binary_ground_truth(3001)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            3001: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            3001: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={3001: ("blob 3001", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={3001: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={3001: _binary_stacker_payload("stack_aug", 0.7)},
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
                "--gemini-model",
                "gemini-2.5-flash",
                "--qa-iterate-mode",
                "advisory",
            ]
        )
        await run_ablation(args)

        assert mocks["research"].await_count == 1
        await_args = mocks["research"].await_args
        assert await_args is not None
        assert await_args.kwargs.get("gemini_model") == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_stage_research_passes_gap_fill_disabled_to_runner(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default (``--no-gap-fill``) passes ``enable_gap_fill=False`` to ``run_gemini_research_for_qids``."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(3010)
        gt1 = _make_binary_ground_truth(3010)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            3010: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            3010: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={3010: ("blob 3010", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={3010: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={3010: _binary_stacker_payload("stack_aug", 0.7)},
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        # Default is --no-gap-fill (no flag set).
        args = _build_parser().parse_args(
            ["--num-binary", "1", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        await run_ablation(args)

        assert mocks["research"].await_count == 1
        await_args = mocks["research"].await_args
        assert await_args is not None
        assert await_args.kwargs.get("enable_gap_fill") is False

    @pytest.mark.asyncio
    async def test_stage_research_passes_gap_fill_enabled_to_runner(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit ``--gap-fill`` passes ``enable_gap_fill=True``."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(3020)
        gt1 = _make_binary_ground_truth(3020)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            3020: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            3020: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={3020: ("blob 3020", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={3020: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={3020: _binary_stacker_payload("stack_aug", 0.7)},
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(
            ["--num-binary", "1", "--cache-dir", str(cache_dir), "--gap-fill", "--qa-iterate-mode", "advisory"]
        )
        await run_ablation(args)

        assert mocks["research"].await_count == 1
        await_args = mocks["research"].await_args
        assert await_args is not None
        assert await_args.kwargs.get("enable_gap_fill") is True

    @pytest.mark.asyncio
    async def test_rate_limit_mode_slow_threads_kwargs_to_forecaster_batch(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--rate-limit-mode slow`` plumbs (per_forecaster_concurrency=1, max_retries=5) into ``run_forecasters_batch``."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(3030)
        gt1 = _make_binary_ground_truth(3030)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            3030: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            3030: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={3030: ("blob 3030", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={3030: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={3030: _binary_stacker_payload("stack_aug", 0.7)},
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
                "--rate-limit-mode",
                "slow",
                "--qa-iterate-mode",
                "advisory",
            ]
        )
        await run_ablation(args)

        assert mocks["forecasters"].await_count == 1
        await_args = mocks["forecasters"].await_args
        assert await_args is not None
        assert await_args.kwargs.get("per_forecaster_concurrency") == 1
        assert await_args.kwargs.get("max_retries") == 5


# ---------------------------------------------------------------------------
# Prune stage
# ---------------------------------------------------------------------------


class TestPruneStage:
    def test_prune_in_stages_list_between_research_and_screen(self) -> None:
        from metaculus_bot.ablation.cli import STAGES

        assert "prune" in STAGES
        research_idx = STAGES.index("research")
        prune_idx = STAGES.index("prune")
        screen_idx = STAGES.index("screen")
        assert research_idx < prune_idx < screen_idx

    def test_parser_accepts_prune_stage(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--stages", "fetch,research,prune"])
        assert "prune" in args.stages

    def test_parser_force_stages_accepts_prune(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--force-stages", "prune"])
        assert "prune" in args.force_stages

    @pytest.mark.asyncio
    async def test_stage_prune_runs_after_research_and_before_screen(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Prune mock fires once, between research and screen, with the raw blobs."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(2001)
        gt1 = _make_binary_ground_truth(2001)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            2001: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            2001: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        stacker_a = {2001: _binary_stacker_payload("stack", 0.6)}
        stacker_b = {2001: _binary_stacker_payload("stack_aug", 0.7)}

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={2001: ("raw research blob 2001", {"sources": 1})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results=stacker_a,
            stacker_b_results=stacker_b,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(
            ["--num-binary", "1", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        await run_ablation(args)

        # Prune was called exactly once with the qid + raw blob.
        assert mocks["prune"].await_count == 1
        prune_call = mocks["prune"].await_args
        assert prune_call is not None
        triples = prune_call.args[0] if prune_call.args else prune_call.kwargs["questions_with_gt_and_blob"]
        assert len(triples) == 1
        question, gt, raw_blob = triples[0]
        assert question.id_of_question == 2001
        assert gt.question_id == 2001
        assert raw_blob == "raw research blob 2001"

    @pytest.mark.asyncio
    async def test_stage_prune_swaps_research_blobs_in_working_set(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After prune, the screen stage receives the SANITIZED blob, not the raw one."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(2002)
        gt1 = _make_binary_ground_truth(2002)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            2002: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        prune_meta = {
            "qid": 2002,
            "original_chars": 100,
            "sanitized_chars": 50,
            "redactions": [{"original_excerpt": "ANSWER", "reason": "leak"}],
            "redactor_invocation_id": "abc",
            "pruned_at": "2026-05-13T18:00:00",
        }

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={2002: ("raw blob with ANSWER inside", {})},
            prune_results={2002: ("sanitized blob without leak", prune_meta)},
            leakage_verdicts=verdicts,
            forecaster_results={
                2002: {
                    model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                        f"openrouter/test/m{i}", 0.5
                    )
                    for i in range(3)
                },
            },
            stacker_a_results={2002: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={2002: _binary_stacker_payload("stack_aug", 0.7)},
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(
            ["--num-binary", "1", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        await run_ablation(args)

        # Screen stage saw the sanitized blob.
        screen_call = mocks["screen"].await_args
        assert screen_call is not None
        # Position 2 is the research_blobs dict; position 0 is questions, 1 is ground_truths.
        research_blobs_arg = (
            screen_call.args[2] if len(screen_call.args) >= 3 else screen_call.kwargs.get("research_cache_payloads")
        )
        assert research_blobs_arg is not None
        assert research_blobs_arg[2002] == "sanitized blob without leak"

        # Forecasters also saw the sanitized blob.
        forecaster_call = mocks["forecasters"].await_args
        assert forecaster_call is not None
        questions_with_research = (
            forecaster_call.args[0] if forecaster_call.args else forecaster_call.kwargs["questions_with_research"]
        )
        for q, blob in questions_with_research:
            if q.id_of_question == 2002:
                assert blob == "sanitized blob without leak"

    @pytest.mark.asyncio
    async def test_stage_prune_drops_validation_failures_from_working_set(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A qid whose prune returned None is dropped before screen runs."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q_ok = _make_binary_question(2010)
        q_fail = _make_binary_question(2011)
        question_set = _build_question_set(
            [
                (q_ok, _make_binary_ground_truth(2010)),
                (q_fail, _make_binary_ground_truth(2011)),
            ]
        )

        verdicts = {
            2010: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            2010: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        stacker_a = {2010: _binary_stacker_payload("stack", 0.6)}
        stacker_b = {2010: _binary_stacker_payload("stack_aug", 0.7)}

        prune_meta = {
            "qid": 2010,
            "original_chars": 100,
            "sanitized_chars": 50,
            "redactions": [],
            "redactor_invocation_id": "abc",
            "pruned_at": "2026-05-13T18:00:00",
        }

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={
                2010: ("raw 2010", {}),
                2011: ("raw 2011", {}),
            },
            prune_results={
                2010: ("sanitized 2010", prune_meta),
                2011: None,  # validation failure for 2011
            },
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results=stacker_a,
            stacker_b_results=stacker_b,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(
            ["--num-binary", "2", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        await run_ablation(args)

        # Screen ONLY saw 2010 (2011 dropped after prune validation failure).
        screen_call = mocks["screen"].await_args
        assert screen_call is not None
        questions_arg = screen_call.args[0] if len(screen_call.args) > 0 else screen_call.kwargs["questions"]
        screened_qids = sorted(q.id_of_question for q in questions_arg)
        assert screened_qids == [2010]

    @pytest.mark.asyncio
    async def test_stage_prune_cached_hit_increments_spend_counter(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Re-running with cached pruned blob bumps cached_prune_hits and zero redactor invocations."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(2020)
        gt1 = _make_binary_ground_truth(2020)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            2020: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            2020: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        stacker_a = {2020: _binary_stacker_payload("stack", 0.6)}
        stacker_b = {2020: _binary_stacker_payload("stack_aug", 0.7)}

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={2020: ("raw 2020", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results=stacker_a,
            stacker_b_results=stacker_b,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        argv = ["--num-binary", "1", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]

        # First run.
        await run_ablation(_build_parser().parse_args(argv))
        out1 = capsys.readouterr().out
        # Fresh redactor invocation.
        assert "Redactor" in out1
        # Cache hit on prune is zero on first run.
        assert "prune=0" in out1

        # Second run with same cache.
        await run_ablation(_build_parser().parse_args(argv))
        out2 = capsys.readouterr().out
        # Pruned cache hit.
        assert "prune=1" in out2

    @pytest.mark.asyncio
    async def test_stage_prune_qa_research_dump_includes_raw_and_sanitized(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The QA dump shows BOTH raw and sanitized blobs for operator review."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(2030)
        gt1 = _make_binary_ground_truth(2030, outcome=True)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            2030: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        prune_meta = {
            "qid": 2030,
            "original_chars": 30,
            "sanitized_chars": 14,
            "redactions": [{"original_excerpt": "RAW_LEAK_TOKEN", "reason": "states resolution"}],
            "redactor_invocation_id": "abc",
            "pruned_at": "2026-05-13T18:00:00",
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={2030: ("raw blob with RAW_LEAK_TOKEN inside", {})},
            prune_results={2030: ("sanitized blob", prune_meta)},
            leakage_verdicts=verdicts,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(
            [
                "--num-binary",
                "1",
                "--qa-research",
                "--cache-dir",
                str(cache_dir),
            ]
        )
        await run_ablation(args)

        qa_dump = next(cache_dir.glob("qa_research_*.md"))
        text = qa_dump.read_text(encoding="utf-8")
        # Raw blob present.
        assert "RAW_LEAK_TOKEN" in text
        # Sanitized blob present.
        assert "sanitized blob" in text
        # Redaction metadata visible.
        assert "states resolution" in text
