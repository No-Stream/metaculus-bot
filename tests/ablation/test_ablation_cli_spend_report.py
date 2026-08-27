"""Spend accounting and the end-of-run summary: ``_print_spend_report`` aggregation
across stages and arms, and the content of the rendered run summary.

Split out of the original monolithic ``test_ablation_cli.py``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from metaculus_bot.ablation.cache import model_slug_to_filename
from tests.ablation_cli_fakes import (
    _binary_forecaster_payload,
    _binary_stacker_payload,
    _build_question_set,
    _install_full_stack_mocks,
    _make_binary_ground_truth,
    _make_binary_question,
    _make_mc_question,
    _make_numeric_question,
)

# ---------------------------------------------------------------------------
# Spend report
# ---------------------------------------------------------------------------


class TestSpendReport:
    @pytest.mark.asyncio
    async def test_spend_report_aggregates_correctly(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Two fresh questions: every counter reflects real call count; cache hits all zero.

        Then re-run with the same cache: every counter zero, every cache hit at the
        previous fresh-call total.
        """
        from metaculus_bot.ablation.cli import _build_parser, run_ablation
        from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

        n_forecasters = len(FREE_FORECASTER_MODELS)

        q1 = _make_binary_question(801)
        q2 = _make_binary_question(802)
        gt1 = _make_binary_ground_truth(801)
        gt2 = _make_binary_ground_truth(802)
        question_set = _build_question_set([(q1, gt1), (q2, gt2)])

        verdicts = {
            801: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
            802: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        # Mirror the real lineup (6 free forecasters by default).
        forecaster_results = {
            qid: {
                model_slug_to_filename(model): _binary_forecaster_payload(model, 0.5)
                for model in FREE_FORECASTER_MODELS
            }
            for qid in (801, 802)
        }

        stacker_a = {qid: _binary_stacker_payload("stack", 0.6) for qid in (801, 802)}
        stacker_b = {qid: _binary_stacker_payload("stack_aug", 0.7) for qid in (801, 802)}

        # Research with gap-fill used (3 gaps each).
        research_results: dict[int, tuple[str, dict] | None] = {
            801: ("research blob 801", {"gap_fill_used": True, "gap_count": 3}),
            802: ("research blob 802", {"gap_fill_used": True, "gap_count": 3}),
        }

        _install_full_stack_mocks(
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

        captured = capsys.readouterr()
        out = captured.out

        # Header still present.
        assert "ABLATION RUN COMPLETE" in out
        assert "Spend report" in out

        # Two fresh research calls + 2*3 gap-fill searches.
        assert "primary: 2 calls" in out
        assert "gap-fill: 6 calls" in out
        # Two leakage detector calls (one per qid).
        assert "Leakage detector     2 LLM calls" in out
        # 2 qids * 6 forecasters each.
        expected_forecaster = 2 * n_forecasters
        assert f"Forecasters          {expected_forecaster} LLM calls" in out
        # 2 qids per arm.
        assert "Stacker (stack)      2 calls (0 fallback)" in out
        assert "Stacker (stack_aug)        2 calls (0 fallback)" in out
        # 4 stacker calls -> 4 parser calls.
        assert "Parser               4 calls" in out
        # All cache hits zero on first fresh run.
        assert "research=0" in out
        assert "screen=0" in out
        assert "forecast=0" in out
        assert "stack=0" in out
        assert "stack_aug=0" in out

        # Second run with same args: every artifact cached, every fresh-call counter zero.
        args2 = _build_parser().parse_args(
            ["--num-binary", "2", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        await run_ablation(args2)

        captured2 = capsys.readouterr()
        out2 = captured2.out

        # All fresh calls zero.
        assert "primary: 0 calls" in out2
        assert "gap-fill: 0 calls" in out2
        assert "Leakage detector     0 LLM calls" in out2
        assert "Forecasters          0 LLM calls" in out2
        assert "Stacker (stack)      0 calls (0 fallback)" in out2
        assert "Stacker (stack_aug)        0 calls (0 fallback)" in out2
        assert "Parser               0 calls" in out2

        # Cache hits reflect what was cached: 2 qids fully cached at every stage.
        assert "research=2" in out2
        assert "screen=2" in out2
        assert f"forecast={2 * n_forecasters}" in out2
        assert "stack=2" in out2
        assert "stack_aug=2" in out2

    @pytest.mark.asyncio
    async def test_spend_report_counts_fallback_stacker(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """A stacker payload with stacker_model_used='fallback' increments the fallback counter."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation
        from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

        q1 = _make_binary_question(810)
        gt1 = _make_binary_ground_truth(810)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            810: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            810: {
                model_slug_to_filename(model): _binary_forecaster_payload(model, 0.5)
                for model in FREE_FORECASTER_MODELS
            },
        }

        # Arm A used the fallback stacker; arm B used primary.
        payload_stack = _binary_stacker_payload("stack", 0.6)
        payload_stack["stacker_model_used"] = "fallback"
        payload_stack_aug = _binary_stacker_payload("stack_aug", 0.7)
        payload_stack_aug["stacker_model_used"] = "primary"

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={810: ("blob 810", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={810: payload_stack},
            stacker_b_results={810: payload_stack_aug},
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(
            ["--num-binary", "1", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        await run_ablation(args)

        captured = capsys.readouterr()
        out = captured.out
        assert "Stacker (stack)      1 calls (1 fallback)" in out
        assert "Stacker (stack_aug)        1 calls (0 fallback)" in out

    @pytest.mark.asyncio
    async def test_spend_report_skips_empty_research_in_leakage_count(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """A verdict with the empty-blob sentinel is NOT a real LLM call; skip in count."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation
        from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS
        from metaculus_bot.ablation.leakage_screen import _EMPTY_BLOB_RESPONSE

        q1 = _make_binary_question(820)
        q2 = _make_binary_question(821)
        gt1 = _make_binary_ground_truth(820)
        gt2 = _make_binary_ground_truth(821)
        question_set = _build_question_set([(q1, gt1), (q2, gt2)])

        # Q820: real LLM verdict. Q821: empty-research short-circuit (no LLM call).
        verdicts = {
            820: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
            821: {
                "is_leaked": False,
                "detector_response": _EMPTY_BLOB_RESPONSE,
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            qid: {
                model_slug_to_filename(model): _binary_forecaster_payload(model, 0.5)
                for model in FREE_FORECASTER_MODELS
            }
            for qid in (820, 821)
        }

        stacker_a = {qid: _binary_stacker_payload("stack", 0.6) for qid in (820, 821)}
        stacker_b = {qid: _binary_stacker_payload("stack_aug", 0.7) for qid in (820, 821)}

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={820: ("blob 820", {}), 821: ("blob 821", {})},
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

        captured = capsys.readouterr()
        out = captured.out
        # Only ONE leakage detector call (for q820); q821's empty-blob sentinel skipped.
        assert "Leakage detector     1 LLM calls" in out

    def test_spend_report_n_clean_subtracts_qa_iterate_drops(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """M6: n_clean must reflect the post-ALL-drops surviving set (research_blobs),
        not n_total - n_leaked which over-counts.

        Synthesize a working set: 10 manifest qids, 3 leaked, 2 dropped at qa_iterate
        (popped from research_blobs), so the surviving set has 5 qids. The headline
        line must say ``5 questions in working set`` and surface the additional drops.
        """
        from metaculus_bot.ablation.cli import SpendReport as _SpendReport
        from metaculus_bot.ablation.cli import WorkingSet as _WorkingSet
        from metaculus_bot.ablation.cli import _print_spend_report

        working = _WorkingSet()
        # 10 manifest qids; 3 leaked, 2 qa_iterate-dropped.
        for qid in range(1, 11):
            working.questions[qid] = _make_binary_question(qid)
        leaked_qids = {1, 2, 3}
        qa_dropped_qids = {4, 5}
        for qid in range(1, 11):
            working.leakage_verdicts[qid] = {
                "is_leaked": qid in leaked_qids,
                "detector_response": "x",
                "detector_model": "x",
                "detector_failed": False,
                "screened_at": "now",
            }
        # research_blobs reflects the post-screen + post-qa_iterate state:
        # leaked qids + qa_iterate-dropped qids removed.
        for qid in range(1, 11):
            if qid not in leaked_qids and qid not in qa_dropped_qids:
                working.research_blobs[qid] = "blob"

        spend = _SpendReport()
        _print_spend_report(spend, working, summary_path=None)

        out = capsys.readouterr().out
        # n_clean must be 5 (the actual surviving research_blobs count), NOT 7
        # (10 - 3 leaked, which ignores qa_iterate drops).
        assert "5 questions in working set" in out, (
            f"M6: n_clean must equal len(research_blobs)=5; got headline:\n{out}"
        )
        # The 2 qa_iterate-dropped qids must be visible somewhere in the report
        # so the operator can reconcile (n_total=10 → 5 surviving = 3 leaked + 2 other drops).
        assert "2 other drops" in out or "n_dropped_other=2" in out or "2 dropped" in out, (
            f"M6: 2 qa_iterate drops must be surfaced separately; got:\n{out}"
        )

    def test_spend_report_n_dropped_other_is_non_negative_on_resume(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Regression: Phase B resume produced -3 other drops via double-counting leaked qids.

        When --stages forecast,stack,pdf,score runs against an already-hydrated
        cache, working.research_blobs contains all on-disk pruned blobs INCLUDING
        leaked qids (because _hydrate_working_set_from_cache doesn't filter by
        leakage verdict). The spend report's n_clean must exclude leaked qids
        so n_dropped_other = n_total - n_clean - n_leaked is non-negative.
        """
        from metaculus_bot.ablation.cli import SpendReport as _SpendReport
        from metaculus_bot.ablation.cli import WorkingSet as _WorkingSet
        from metaculus_bot.ablation.cli import _print_spend_report

        # Mirror the bug-report scenario: 19 manifest qids, 4 leaked, 18 pruned blobs
        # on disk (one qid lost research; the other 18 include the 4 leaked qids).
        working = _WorkingSet()
        leaked_qids = {1, 2, 3, 4}
        no_blob_qid = 19  # research failed, so this qid never landed in research_blobs
        for qid in range(1, 20):
            working.questions[qid] = _make_binary_question(qid)
            working.leakage_verdicts[qid] = {
                "is_leaked": qid in leaked_qids,
                "detector_response": "x",
                "detector_model": "x",
                "detector_failed": False,
                "screened_at": "now",
            }
            if qid != no_blob_qid:
                working.research_blobs[qid] = "blob"

        _print_spend_report(_SpendReport(), working, summary_path=None)

        out = capsys.readouterr().out
        # The Results headline must not contain a negative count anywhere.
        results_lines = [line for line in out.splitlines() if line.startswith("Results:")]
        assert results_lines, f"expected a 'Results:' headline in output:\n{out}"
        assert "-" not in results_lines[0], (
            f"resume invocation produced negative count in headline: {results_lines[0]!r}"
        )
        # n_clean = 19 manifest - 4 leaked - 1 research-failure = 14.
        # n_dropped_other = 19 - 14 - 4 = 1 (the research-failure qid).
        assert "14 questions in working set" in out, f"expected n_clean=14, got:\n{out}"
        assert "4 leaked" in out
        assert "1 other drops" in out

    def test_spend_report_n_clean_excludes_leaked_qids(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Even when research_blobs contains leaked qids, n_clean reports only non-leaked."""
        from metaculus_bot.ablation.cli import SpendReport as _SpendReport
        from metaculus_bot.ablation.cli import WorkingSet as _WorkingSet
        from metaculus_bot.ablation.cli import _print_spend_report

        working = _WorkingSet()
        leaked_qids = {1, 2}
        for qid in range(1, 6):
            working.questions[qid] = _make_binary_question(qid)
            working.leakage_verdicts[qid] = {
                "is_leaked": qid in leaked_qids,
                "detector_response": "x",
                "detector_model": "x",
                "detector_failed": False,
                "screened_at": "now",
            }
            # Resume-style: research_blobs hydrated from disk, leaked qids included.
            working.research_blobs[qid] = "blob"

        _print_spend_report(_SpendReport(), working, summary_path=None)

        out = capsys.readouterr().out
        # n_clean = 5 - 2 leaked = 3; n_dropped_other = 5 - 3 - 2 = 0.
        assert "3 questions in working set" in out, f"expected n_clean=3, got:\n{out}"
        assert "2 leaked" in out
        assert "0 other drops" in out

    def test_spend_report_by_type_counts_match_n_clean(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """The by-type counts (Binary/MC/Numeric) sum to n_clean, not to len(research_blobs)."""
        from metaculus_bot.ablation.cli import SpendReport as _SpendReport
        from metaculus_bot.ablation.cli import WorkingSet as _WorkingSet
        from metaculus_bot.ablation.cli import _print_spend_report

        working = _WorkingSet()
        # 1 binary (leaked), 1 MC (clean), 1 numeric (clean). research_blobs has all 3.
        binary_qid, mc_qid, numeric_qid = 100, 200, 300
        working.questions[binary_qid] = _make_binary_question(binary_qid)
        working.questions[mc_qid] = _make_mc_question(mc_qid)
        working.questions[numeric_qid] = _make_numeric_question(numeric_qid)
        working.leakage_verdicts = {
            binary_qid: {
                "is_leaked": True,
                "detector_response": "leaked",
                "detector_model": "x",
                "detector_failed": False,
                "screened_at": "now",
            },
            mc_qid: {
                "is_leaked": False,
                "detector_response": "x",
                "detector_model": "x",
                "detector_failed": False,
                "screened_at": "now",
            },
            numeric_qid: {
                "is_leaked": False,
                "detector_response": "x",
                "detector_model": "x",
                "detector_failed": False,
                "screened_at": "now",
            },
        }
        # Resume-style: research_blobs hydrated, leaked qid still present on disk.
        working.research_blobs[binary_qid] = "blob"
        working.research_blobs[mc_qid] = "blob"
        working.research_blobs[numeric_qid] = "blob"

        _print_spend_report(_SpendReport(), working, summary_path=None)

        out = capsys.readouterr().out
        # n_clean = 2 (the leaked binary qid is excluded from the working set).
        assert "2 questions in working set" in out, f"expected n_clean=2, got:\n{out}"
        # By-type must reflect post-leak filtering: leaked binary qid NOT counted.
        assert "Binary:  0 questions" in out, f"binary count wrong:\n{out}"
        assert "MC:      1 questions" in out, f"MC count wrong:\n{out}"
        assert "Numeric: 1 questions" in out, f"numeric count wrong:\n{out}"


# ---------------------------------------------------------------------------
# Summary file content
# ---------------------------------------------------------------------------


class TestSummaryContent:
    @pytest.mark.asyncio
    async def test_summary_file_includes_paired_scores(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(1201)
        gt1 = _make_binary_ground_truth(1201, outcome=True)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            1201: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            1201: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        stacker_a = {1201: _binary_stacker_payload("stack", 0.6)}
        stacker_b = {1201: _binary_stacker_payload("stack_aug", 0.8)}

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={1201: ("blob 1201", {})},
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

        summary_path = next((cache_dir / "scores").glob("summary_*.md"))
        text = summary_path.read_text(encoding="utf-8")
        assert "Overall summary" in text
        assert "Per-type breakdown" in text
        assert "Per-question diagnostic" in text
