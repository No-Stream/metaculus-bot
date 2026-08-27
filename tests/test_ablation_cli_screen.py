"""Leakage screen and the questions it removes: the screen cache's research-blob hash
check, the ``--qa-research`` halt-after-screen path and its dump, plus the
leaked / failed-research drop paths.

Split out of the original monolithic ``test_ablation_cli.py`` — everything here is
about which questions survive into the forecast stage and why.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
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
# C3 part 2: screen cache invalidation when blob hash changes
# ---------------------------------------------------------------------------


class TestScreenCacheBlobHashCheck:
    """When the prune stage re-runs (--force-stages prune), the screen verdict
    cache may hold a verdict against an OLD pruned blob. The screen stage must
    detect that the blob has changed (via a sha hash) and re-run instead of
    silently returning a stale verdict.

    Backwards compat: a cached verdict written before the sha field was added
    must also re-screen (treat the missing field as "stale").
    """

    @pytest.mark.asyncio
    async def test_screen_re_runs_when_cached_verdict_blob_sha_is_stale(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import hashlib

        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q = _make_binary_question(8200)
        gt = _make_binary_ground_truth(8200)
        question_set = _build_question_set([(q, gt)])

        # Pre-populate a screen verdict that hashes a DIFFERENT blob than the
        # one the new run will compute. The stale verdict says "leaked", but
        # the fresh blob is benign.
        cache = AblationCache(cache_dir)
        cache.write_leakage_screen(
            qid=8200,
            payload={
                "is_leaked": True,
                "detector_response": "stale leak verdict",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-01T00:00:00",
                "research_blob_sha": hashlib.sha256(b"OLD blob content").hexdigest()[:16],
            },
        )

        # The new (fresh) verdict says clean.
        verdicts = {
            8200: {
                "is_leaked": False,
                "detector_response": "fresh blob is clean",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
                "research_blob_sha": hashlib.sha256(b"FRESH blob 8200").hexdigest()[:16],
            },
        }

        forecaster_results = {
            8200: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={8200: ("FRESH blob 8200", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={8200: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={8200: _binary_stacker_payload("stack_aug", 0.7)},
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
        await run_ablation(args)

        # screen_batch must have been called (the cached stale verdict was
        # invalidated, so the screen re-runs).
        assert mocks["screen"].await_count >= 1

    @pytest.mark.asyncio
    async def test_screen_uses_cache_when_blob_sha_matches(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the cached verdict's sha matches the current blob, no re-screen."""
        import hashlib

        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q = _make_binary_question(8210)
        gt = _make_binary_ground_truth(8210)
        question_set = _build_question_set([(q, gt)])

        # Pre-populate research and a verdict with a MATCHING sha.
        cache = AblationCache(cache_dir)
        blob_text = "fresh blob 8210"
        cache.write_research(8210, blob_text, {"sources": 1})
        cache.write_pruned_research(
            qid=8210,
            sanitized_blob=blob_text,
            meta={
                "qid": 8210,
                "original_chars": len(blob_text),
                "sanitized_chars": len(blob_text),
                "redactions": [],
                "redactor_invocation_id": "x",
                "pruned_at": "2026-05-13T18:00:00",
            },
        )
        cache.write_leakage_screen(
            qid=8210,
            payload={
                "is_leaked": False,
                "detector_response": "cached clean verdict",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
                "research_blob_sha": hashlib.sha256(blob_text.encode()).hexdigest()[:16],
            },
        )

        forecaster_results = {
            8210: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={8210: (blob_text, {})},
            leakage_verdicts={
                8210: {
                    "is_leaked": False,
                    "detector_response": "should not have re-screened",
                    "detector_model": "test",
                    "detector_failed": False,
                    "screened_at": "2026-05-14T00:00:00",
                    "research_blob_sha": hashlib.sha256(blob_text.encode()).hexdigest()[:16],
                }
            },
            forecaster_results=forecaster_results,
            stacker_a_results={8210: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={8210: _binary_stacker_payload("stack_aug", 0.7)},
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
        await run_ablation(args)

        # screen_batch must NOT have been called (cache hit was honored).
        assert mocks["screen"].await_count == 0

    @pytest.mark.asyncio
    async def test_screen_re_runs_when_cached_verdict_missing_blob_sha(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A pre-existing cached verdict with no ``research_blob_sha`` field
        (written before C3 part 2 landed) must trigger a re-screen as a
        defensive cache-invalidation path.
        """
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q = _make_binary_question(8220)
        gt = _make_binary_ground_truth(8220)
        question_set = _build_question_set([(q, gt)])

        # Pre-populate a verdict without the new field.
        cache = AblationCache(cache_dir)
        cache.write_leakage_screen(
            qid=8220,
            payload={
                "is_leaked": True,
                "detector_response": "old verdict pre-C3",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-04-01T00:00:00",
            },
        )

        forecaster_results = {
            8220: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={8220: ("blob 8220", {})},
            leakage_verdicts={
                8220: {
                    "is_leaked": False,
                    "detector_response": "fresh clean",
                    "detector_model": "test",
                    "detector_failed": False,
                    "screened_at": "2026-05-14T00:00:00",
                    "research_blob_sha": "fresh-hash-stub",
                }
            },
            forecaster_results=forecaster_results,
            stacker_a_results={8220: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={8220: _binary_stacker_payload("stack_aug", 0.7)},
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
        await run_ablation(args)

        # Missing-field cache entry must invalidate; re-screen fires.
        assert mocks["screen"].await_count >= 1


# ---------------------------------------------------------------------------
# QA-research halts after screen
# ---------------------------------------------------------------------------


class TestQaResearch:
    @pytest.mark.asyncio
    async def test_qa_research_halts_after_screen(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(201)
        gt1 = _make_binary_ground_truth(201)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            201: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={201: ("research blob 201", {})},
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
        exit_code = await run_ablation(args)
        assert exit_code == 0

        # Forecasters and stackers must NOT have been called.
        assert mocks["forecasters"].await_count == 0
        assert mocks["stacker"].await_count == 0

        # QA dump must exist.
        qa_dumps = list(cache_dir.glob("qa_research_*.md"))
        assert len(qa_dumps) == 1

    @pytest.mark.asyncio
    async def test_qa_research_dump_contains_question_text_and_verdict(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(301)
        gt1 = _make_binary_ground_truth(301, outcome=True)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            301: {
                "is_leaked": False,
                "detector_response": "Looks clean. No resolution leak.",
                "detector_model": "openrouter/test/detector",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={301: ("This is a long research blob with multiple paragraphs.", {})},
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
        assert "Will Q301 happen" in text
        assert "YES" in text
        assert "research blob" in text
        assert "Looks clean" in text


# ---------------------------------------------------------------------------
# Leaked qid dropped downstream
# ---------------------------------------------------------------------------


class TestLeakedQidDropped:
    @pytest.mark.asyncio
    async def test_leaked_qid_dropped_from_downstream(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q_clean = _make_binary_question(701)
        q_leaked = _make_binary_question(702)
        question_set = _build_question_set(
            [
                (q_clean, _make_binary_ground_truth(701)),
                (q_leaked, _make_binary_ground_truth(702)),
            ]
        )

        verdicts = {
            701: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
            702: {
                "is_leaked": True,
                "detector_response": "Leak detected: news article cites resolution.",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }
        forecaster_results = {
            701: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        stacker_a = {701: _binary_stacker_payload("stack", 0.6)}
        stacker_b = {701: _binary_stacker_payload("stack_aug", 0.7)}

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={701: ("blob 701", {}), 702: ("blob 702", {})},
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

        # Forecaster batch only saw qid 701.
        forecaster_call = mocks["forecasters"].await_args
        assert forecaster_call is not None
        questions_with_research = (
            forecaster_call.args[0] if forecaster_call.args else forecaster_call.kwargs["questions_with_research"]
        )
        forecaster_qids = sorted(q.id_of_question for q, _ in questions_with_research)
        assert forecaster_qids == [701]

        # Stacker only saw 701.
        stacker_args = mocks["stacker"].await_args
        assert stacker_args is not None
        qid_to_data = stacker_args.args[0] if stacker_args.args else stacker_args.kwargs["qid_to_data"]
        assert sorted(qid_to_data.keys()) == [701]


# ---------------------------------------------------------------------------
# Failed research drops downstream
# ---------------------------------------------------------------------------


class TestFailedResearchDropped:
    @pytest.mark.asyncio
    async def test_failed_research_qid_dropped(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q_ok = _make_binary_question(901)
        q_fail = _make_binary_question(902)
        question_set = _build_question_set(
            [
                (q_ok, _make_binary_ground_truth(901)),
                (q_fail, _make_binary_ground_truth(902)),
            ]
        )

        # 902 returns None (research failure).
        research_results: dict[int, tuple[str, dict] | None] = {
            901: ("blob 901", {}),
            902: None,
        }

        verdicts = {
            901: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            901: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        stacker_a = {901: _binary_stacker_payload("stack", 0.6)}
        stacker_b = {901: _binary_stacker_payload("stack_aug", 0.7)}

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results=research_results,
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

        # Screen only saw 901.
        screen_call = mocks["screen"].await_args
        assert screen_call is not None
        # Look up the research_blobs arg (positional or kwarg).
        questions_arg = screen_call.args[0] if len(screen_call.args) > 0 else screen_call.kwargs["questions"]
        screened_qids = sorted(q.id_of_question for q in questions_arg)
        assert screened_qids == [901]


# ---------------------------------------------------------------------------
# Bug-5 regression: QA research dump must include the actual blob for leaked
# qids (the operator needs to verify the screener's verdict). The screen
# stage pops leaked qids from in-memory ``working.research_blobs`` to keep
# them out of forecasting; the dump still needs to read the blob from disk.
# ---------------------------------------------------------------------------


class TestQaResearchDumpIncludesLeakedBlobs:
    def test_qa_research_dump_includes_leaked_blobs(
        self,
        cache_dir: Path,
    ) -> None:
        """A leaked qid's blob must appear in the QA dump verbatim.

        After the screen stage runs, leaked qids are popped from
        ``working.research_blobs`` so downstream forecaster/stacker stages
        skip them. But the QA dump exists specifically for the operator to
        review the screener's verdicts — meaning the blob that the screener
        flagged must be visible alongside the verdict. Reading from
        ``cache.read_research(qid)`` (still on disk) preserves the blob
        regardless of in-memory pops.
        """
        from metaculus_bot.ablation.cli import WorkingSet, _stage_qa_research_dump

        cache = AblationCache(str(cache_dir))

        leaked_qid = 9101
        leaked_blob = "BREAKING: result was Y. Here is the resolution news article body."
        cache.write_research(leaked_qid, leaked_blob, {"sources": 2})

        leaked_question = _make_binary_question(leaked_qid)
        leaked_gt = _make_binary_ground_truth(leaked_qid, outcome=True)
        leaked_verdict = {
            "is_leaked": True,
            "detector_response": "Leak detected: news article cites resolution.",
            "detector_model": "test",
            "detector_failed": False,
            "screened_at": "2026-05-13T12:00:00",
        }

        working = WorkingSet()
        working.questions[leaked_qid] = leaked_question
        working.ground_truths[leaked_qid] = leaked_gt
        # Leaked qids are POPPED from ``research_blobs`` by ``_stage_screen``.
        # The QA dump must still surface the actual blob for the operator.
        working.leakage_verdicts[leaked_qid] = leaked_verdict

        args = MagicMock()
        args.num_binary = 1
        args.num_multiple_choice = 0
        args.num_numeric = 0

        target_path = _stage_qa_research_dump(args, cache, working)

        text = target_path.read_text(encoding="utf-8")
        assert leaked_blob in text, (
            f"Leaked blob must appear in QA dump so operator can verify the screener; dump content: {text[:500]}"
        )
        assert "Leak detected" in text
        assert f"Q{leaked_qid}" in text
