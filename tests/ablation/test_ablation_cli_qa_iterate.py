"""The ``qa_iterate`` stage: verifier/re-redactor loop, thresholds, iteration caps,
manual-reject bookkeeping and the reports it writes.

Split out of the original monolithic ``test_ablation_cli.py``. These tests patch the
two LLM seams on ``metaculus_bot.ablation.qa_iterate`` (``_invoke_verifier`` /
``_invoke_re_redactor``) rather than on the CLI, so the patch targets are the
qa_iterate module's own attributes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
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
)

# ---------------------------------------------------------------------------
# qa_iterate stage
# ---------------------------------------------------------------------------


class TestQaIterateStage:
    def test_qa_iterate_in_stages_list_between_screen_and_forecast(self) -> None:
        from metaculus_bot.ablation.cli import STAGES

        assert "qa_iterate" in STAGES
        screen_idx = STAGES.index("screen")
        qa_iterate_idx = STAGES.index("qa_iterate")
        forecast_idx = STAGES.index("forecast")
        assert screen_idx < qa_iterate_idx < forecast_idx

    def test_parser_qa_iterate_mode_default_is_halt(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.qa_iterate_mode == "halt"

    def test_parser_qa_iterate_mode_accepts_advisory(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--qa-iterate-mode", "advisory"])
        assert args.qa_iterate_mode == "advisory"

    def test_parser_qa_iterate_mode_accepts_skip(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--qa-iterate-mode", "skip"])
        assert args.qa_iterate_mode == "skip"

    def test_parser_qa_iterate_mode_rejects_invalid(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--qa-iterate-mode", "nonsense"])

    def test_parser_qa_iterate_max_iterations_default(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.qa_iterate_max_iterations == 3

    def test_parser_qa_iterate_leakage_threshold_default(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.qa_iterate_leakage_threshold == 0.3

    def test_parser_qa_iterate_forecastability_threshold_default(self) -> None:
        """The forecastability threshold defaults to qa_iterate.DEFAULT_FORECASTABILITY_THRESHOLD."""
        from metaculus_bot.ablation.cli import _build_parser
        from metaculus_bot.ablation.qa_iterate import DEFAULT_FORECASTABILITY_THRESHOLD

        args = _build_parser().parse_args([])
        assert args.qa_iterate_forecastability_threshold == DEFAULT_FORECASTABILITY_THRESHOLD

    def test_parser_prune_batch_size_default(self) -> None:
        """The prune batch size defaults to prune.DEFAULT_BATCH_SIZE."""
        from metaculus_bot.ablation.cli import _build_parser
        from metaculus_bot.ablation.prune import DEFAULT_BATCH_SIZE

        args = _build_parser().parse_args([])
        assert args.prune_batch_size == DEFAULT_BATCH_SIZE

    def test_parser_prune_batch_size_override(self) -> None:
        """Operators can shrink batch_size to bound blast radius on flaky runs."""
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--prune-batch-size", "5"])
        assert args.prune_batch_size == 5

    def test_parser_qa_iterate_forecastability_threshold_override(self) -> None:
        """Operators can tune the forecastability threshold (smoke runs at boundary)."""
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--qa-iterate-forecastability-threshold", "0.15"])
        assert args.qa_iterate_forecastability_threshold == 0.15

    @pytest.mark.asyncio
    async def test_qa_iterate_forecastability_threshold_flows_through_to_batch(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The CLI flag must flow through to ``run_qa_iterate_batch``."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q = _make_binary_question(8300)
        gt = _make_binary_ground_truth(8300)
        question_set = _build_question_set([(q, gt)])

        verdicts = {
            8300: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            8300: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={8300: ("blob 8300", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={8300: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={8300: _binary_stacker_payload("stack_aug", 0.7)},
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        captured: dict[str, Any] = {}

        async def capturing_batch(*args: Any, **kwargs: Any) -> dict:
            await asyncio.sleep(0)
            captured.update(kwargs)
            return {}

        monkeypatch.setattr("metaculus_bot.ablation.cli.run_qa_iterate_batch", capturing_batch)

        args = _build_parser().parse_args(
            [
                "--num-binary",
                "1",
                "--cache-dir",
                str(cache_dir),
                "--qa-iterate-mode",
                "advisory",
                "--qa-iterate-forecastability-threshold",
                "0.15",
            ]
        )
        await run_ablation(args)

        assert captured.get("forecastability_threshold") == 0.15

    @pytest.mark.asyncio
    async def test_skip_mode_is_noop(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """skip mode: stage runs but does nothing — no file IO, no subprocess."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(7001)
        gt1 = _make_binary_ground_truth(7001)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            7001: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }
        forecaster_results = {
            7001: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={7001: ("blob 7001", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={7001: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={7001: _binary_stacker_payload("stack_aug", 0.7)},
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        verifier_mock = AsyncMock()
        redactor_mock = AsyncMock()
        monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", verifier_mock)
        monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", redactor_mock)

        args = _build_parser().parse_args(
            ["--num-binary", "1", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "skip"]
        )
        exit_code = await run_ablation(args)

        assert exit_code == 0
        # Skip mode = no verifier/redactor invocations.
        assert verifier_mock.await_count == 0
        assert redactor_mock.await_count == 0
        # No qa_summary or manual_rejects emitted.
        assert not list(cache_dir.glob("qa_summary_*.md"))
        assert not (cache_dir / "manual_rejects.json").exists()

    @pytest.mark.asyncio
    async def test_advisory_mode_logs_but_continues(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """advisory mode: writes summary, proceeds to forecast even with rejects."""
        import json

        from metaculus_bot.ablation.cli import _build_parser, run_ablation
        from metaculus_bot.ablation.qa_iterate import _invoke_verifier  # noqa: F401

        q1 = _make_binary_question(7010)
        q2 = _make_binary_question(7011)
        gt1 = _make_binary_ground_truth(7010)
        gt2 = _make_binary_ground_truth(7011)
        question_set = _build_question_set([(q1, gt1), (q2, gt2)])

        verdicts = {
            7010: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
            7011: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            qid: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            }
            for qid in (7010, 7011)
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={7010: ("blob clean", {}), 7011: ("blob leaky", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={
                7010: _binary_stacker_payload("stack", 0.6),
                7011: _binary_stacker_payload("stack", 0.6),
            },
            stacker_b_results={
                7010: _binary_stacker_payload("stack_aug", 0.7),
                7011: _binary_stacker_payload("stack_aug", 0.7),
            },
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        def _verifier_response_for(qid: int, leakage: float, forecastability: float) -> str:
            return json.dumps(
                {
                    "verdicts": [
                        {
                            "qid": qid,
                            "leakage_risk": leakage,
                            "forecastability": forecastability,
                            "hallucination_risk": 0.2,
                            "notes": "",
                        }
                    ]
                }
            )

        # 7010 clean on first pass; 7011 always leaky → rejected.
        async def _fake_verifier(prompt: str, **_kwargs: Any) -> str:
            await asyncio.sleep(0)
            if "qid=7010" in prompt:
                return _verifier_response_for(7010, leakage=0.05, forecastability=0.8)
            return _verifier_response_for(7011, leakage=0.5, forecastability=0.6)

        async def _fake_redactor(prompt: str, **_kwargs: Any) -> str:
            await asyncio.sleep(0)
            return json.dumps(
                {
                    "results": [
                        {"qid": 7011, "sanitized_blob": "still leaky", "redactions": []},
                    ]
                }
            )

        monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", _fake_verifier)
        monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", _fake_redactor)

        args = _build_parser().parse_args(
            ["--num-binary", "2", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        exit_code = await run_ablation(args)

        assert exit_code == 0
        summaries = list(cache_dir.glob("qa_summary_*.md"))
        assert len(summaries) == 1
        rejects_path = cache_dir / "manual_rejects.json"
        assert rejects_path.exists()
        rejects = json.loads(rejects_path.read_text(encoding="utf-8"))
        assert "7011" in rejects["rejects"]

    @pytest.mark.asyncio
    async def test_halt_mode_raises_after_summary_written(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """halt mode: writes summary first, then raises RuntimeError."""
        import json

        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(7020)
        gt1 = _make_binary_ground_truth(7020)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            7020: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={7020: ("leaky blob", {})},
            leakage_verdicts=verdicts,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        async def _fake_verifier(prompt: str, **_kwargs: Any) -> str:
            await asyncio.sleep(0)
            return json.dumps(
                {
                    "verdicts": [
                        {
                            "qid": 7020,
                            "leakage_risk": 0.5,
                            "forecastability": 0.6,
                            "hallucination_risk": 0.2,
                            "notes": "still leaky",
                        }
                    ]
                }
            )

        async def _fake_redactor(prompt: str, **_kwargs: Any) -> str:
            await asyncio.sleep(0)
            return json.dumps({"results": [{"qid": 7020, "sanitized_blob": "still leaky 2", "redactions": []}]})

        monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", _fake_verifier)
        monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", _fake_redactor)

        # Halt mode is the default but pass explicitly for clarity.
        args = _build_parser().parse_args(
            ["--num-binary", "1", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "halt"]
        )
        with pytest.raises(RuntimeError, match="QA iteration"):
            await run_ablation(args)

        # Summary was written BEFORE the raise.
        summaries = list(cache_dir.glob("qa_summary_*.md"))
        assert len(summaries) == 1
        rejects_path = cache_dir / "manual_rejects.json"
        assert rejects_path.exists()

    @pytest.mark.asyncio
    async def test_halt_mode_resume_message_documents_manual_rejects_caveat(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """M5: the halt-mode RuntimeError must explain that manual_rejects.json
        is only honored when qa_iterate is in --stages.

        Pre-fix: the message said "edit manual_rejects.json if needed, then
        resume with --stages forecast,...". Operators followed the literal
        instruction, edited the file, and forecast on rejected qids anyway
        because the resume command bypassed qa_iterate.
        """
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(7501)
        gt1 = _make_binary_ground_truth(7501)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            7501: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={7501: ("blob", {})},
            leakage_verdicts=verdicts,
        )
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        args = _build_parser().parse_args(
            ["--num-binary", "1", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "halt"]
        )

        with pytest.raises(RuntimeError) as excinfo:
            await run_ablation(args)

        message = str(excinfo.value)
        # Must mention reviewing the QA summary AND the manual_rejects caveat.
        assert "manual_rejects" in message
        assert "qa_iterate" in message, (
            f"resume message must explain that manual_rejects is only consulted "
            f"when qa_iterate is in --stages; got: {message}"
        )
        # Must include the resume command.
        assert "forecast,stack,stack_aug,pdf,median,score" in message

    @pytest.mark.asyncio
    async def test_advisory_mode_writes_per_qid_qa_report(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The qa_iterate stage must write one qa_reports/<qid>.json per qid.

        The plan doc (``scratch_docs_and_planning/ablation_phase_a3_plan.md:290``)
        specifies these per-qid reports, but the implementation only wrote the
        aggregate summary + manual_rejects.json. Audit at
        ``backtests/ablation/audit_smoke_20260515.md:243-263`` flagged the gap.
        """
        import json

        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(7030)
        gt1 = _make_binary_ground_truth(7030)
        question_set = _build_question_set([(q1, gt1)])

        verdicts = {
            7030: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        forecaster_results = {
            7030: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            },
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={7030: ("blob 7030", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={7030: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={7030: _binary_stacker_payload("stack_aug", 0.7)},
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(
            ["--num-binary", "1", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        exit_code = await run_ablation(args)

        assert exit_code == 0
        qa_reports_dir = cache_dir / "qa_reports"
        assert qa_reports_dir.is_dir(), "expected qa_reports/ subdirectory under cache root"
        report_path = qa_reports_dir / "7030.json"
        assert report_path.exists(), f"expected per-qid report at {report_path}"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["qid"] == 7030
        assert report["final_status"] == "clean"
        assert "verifier_scores" in report
        assert "iterations" in report
        assert isinstance(report["verifier_scores"], list)
        assert len(report["verifier_scores"]) >= 1

    @pytest.mark.asyncio
    async def test_advisory_mode_writes_qa_report_for_rejected_qids_too(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rejected qids must get qa_reports too — they're the most important to review."""
        import json

        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q_clean = _make_binary_question(7040)
        q_rejected = _make_binary_question(7041)
        gt_clean = _make_binary_ground_truth(7040)
        gt_rejected = _make_binary_ground_truth(7041)
        question_set = _build_question_set([(q_clean, gt_clean), (q_rejected, gt_rejected)])

        verdicts = {
            qid: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            }
            for qid in (7040, 7041)
        }

        forecaster_results = {
            qid: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            }
            for qid in (7040, 7041)
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={7040: ("blob 7040", {}), 7041: ("blob 7041", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={
                7040: _binary_stacker_payload("stack", 0.6),
                7041: _binary_stacker_payload("stack", 0.6),
            },
            stacker_b_results={
                7040: _binary_stacker_payload("stack_aug", 0.7),
                7041: _binary_stacker_payload("stack_aug", 0.7),
            },
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        # 7040 clean (low leakage); 7041 always leaky → rejected_leakage.
        async def _fake_verifier(prompt: str, **_kwargs: Any) -> str:
            await asyncio.sleep(0)
            if "qid=7040" in prompt:
                return json.dumps(
                    {
                        "verdicts": [
                            {
                                "qid": 7040,
                                "leakage_risk": 0.05,
                                "forecastability": 0.8,
                                "hallucination_risk": 0.1,
                                "notes": "",
                            }
                        ]
                    }
                )
            return json.dumps(
                {
                    "verdicts": [
                        {
                            "qid": 7041,
                            "leakage_risk": 0.5,
                            "forecastability": 0.6,
                            "hallucination_risk": 0.2,
                            "notes": "still leaky",
                        }
                    ]
                }
            )

        async def _fake_redactor(prompt: str, **_kwargs: Any) -> str:
            await asyncio.sleep(0)
            return json.dumps({"results": [{"qid": 7041, "sanitized_blob": "still leaky 2", "redactions": []}]})

        monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", _fake_verifier)
        monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", _fake_redactor)

        args = _build_parser().parse_args(
            ["--num-binary", "2", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        await run_ablation(args)

        qa_reports_dir = cache_dir / "qa_reports"
        assert (qa_reports_dir / "7040.json").exists()
        assert (qa_reports_dir / "7041.json").exists()
        rejected_report = json.loads((qa_reports_dir / "7041.json").read_text(encoding="utf-8"))
        # The reject status varies by reason but it must NOT be "clean".
        assert rejected_report["final_status"] != "clean"
        assert rejected_report["reject_reason"] is not None
