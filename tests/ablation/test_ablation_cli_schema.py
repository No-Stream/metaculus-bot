"""Schema strictness at the cache and manifest boundaries: a payload missing a required
field, or carrying an unexpected one, must be rejected rather than silently half-read.

Split out of the original monolithic ``test_ablation_cli.py``. Round-trip fidelity of the
manifest shims themselves is covered in ``test_ablation_cli_shims.py``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.backtest.question_prep import BacktestQuestionSet
from tests.ablation_cli_fakes import (
    _build_question_set,
    _install_full_stack_mocks,
    _make_binary_ground_truth,
    _make_binary_question,
)

# ---------------------------------------------------------------------------
# Schema strictness — defensive .get() / getattr() removed in favor of fail-fast
# ---------------------------------------------------------------------------


class TestSchemaStrictness:
    """Verify the orchestrator surfaces schema drift rather than silently masking it.

    Each defensive ``getattr(..., default)`` / ``.get(key, default)`` pattern hides a
    failure mode: if forecasting-tools renames a question attribute or the cache
    payload schema changes shape, the orchestrator must crash with a clear
    AttributeError / KeyError rather than emit ``None`` / ``0.0`` / ``False``-tinged
    silent corruption that biases the ablation result.
    """

    def test_serialize_question_metadata_raises_when_numeric_zero_point_attribute_missing(self) -> None:
        """If forecasting-tools removes ``zero_point`` from NumericQuestion, fail loud."""
        from metaculus_bot.ablation.cli import _serialize_question_metadata

        q = MagicMock(spec=NumericQuestion)
        q.lower_bound = 0.0
        q.upper_bound = 100.0
        q.open_lower_bound = False
        q.open_upper_bound = False
        q.zero_point = None
        del q.zero_point  # Simulate schema drift: attribute removed.

        with pytest.raises(AttributeError):
            _serialize_question_metadata(q)

    def test_serialize_question_metadata_raises_when_mc_options_attribute_missing(self) -> None:
        """If forecasting-tools removes ``options`` from MultipleChoiceQuestion, fail loud."""
        from metaculus_bot.ablation.cli import _serialize_question_metadata

        q = MagicMock(spec=MultipleChoiceQuestion)
        q.options = ["Red", "Blue"]
        del q.options  # Simulate schema drift: attribute removed.

        with pytest.raises(AttributeError):
            _serialize_question_metadata(q)

    def test_build_manifest_entry_raises_when_page_url_attribute_missing(self) -> None:
        """If forecasting-tools renames ``page_url``, manifest writer must crash, not emit None."""
        from metaculus_bot.ablation.cli import _build_manifest_entry

        q = MagicMock(spec=BinaryQuestion)
        q.question_text = "Will it happen?"
        q.page_url = "https://example.com/q/1"
        del q.page_url  # Simulate schema drift: attribute removed.

        gt = _make_binary_ground_truth(1)

        with pytest.raises(AttributeError):
            _build_manifest_entry(q, gt, "spring-aib-2026")

    def test_build_question_shim_raises_when_manifest_entry_missing_page_url(self) -> None:
        """Manifest is written by this same module — a missing key means schema drift."""
        from metaculus_bot.ablation.cli import _build_question_shim_from_manifest_entry

        entry = {
            "type": "binary",
            "tournament": "spring-aib-2026",
            "question_text": "Will it happen?",
            # page_url deliberately missing — would never happen if manifest came from
            # the current _build_manifest_entry, so absence indicates drift.
            "ground_truth": {},
            "question_metadata": {},
        }

        with pytest.raises(KeyError):
            _build_question_shim_from_manifest_entry(1, entry)

    def test_build_question_shim_raises_when_manifest_entry_missing_question_metadata(self) -> None:
        """``question_metadata`` is always written by _build_manifest_entry; missing → drift."""
        from metaculus_bot.ablation.cli import _build_question_shim_from_manifest_entry

        entry = {
            "type": "numeric",
            "tournament": "spring-aib-2026",
            "question_text": "What value?",
            "page_url": "https://example.com/q/1",
            "ground_truth": {},
            "resolution_criteria": "Resolves to a number.",
            "fine_print": "",
            "background_info": "",
            # question_metadata deliberately missing.
        }

        with pytest.raises(KeyError):
            _build_question_shim_from_manifest_entry(1, entry)

    def test_build_question_shim_raises_when_numeric_metadata_missing_lower_bound(self) -> None:
        """Numeric questions ALWAYS have ``lower_bound``; missing → drift."""
        from metaculus_bot.ablation.cli import _build_question_shim_from_manifest_entry

        entry = {
            "type": "numeric",
            "tournament": "spring-aib-2026",
            "question_text": "What value?",
            "page_url": "https://example.com/q/1",
            "ground_truth": {},
            "resolution_criteria": "Resolves to a number.",
            "fine_print": "",
            "background_info": "",
            "question_metadata": {
                "open_time": "2026-01-01T00:00:00",
                "scheduled_resolution_time": "2026-05-01T00:00:00",
                # lower_bound deliberately missing.
                "upper_bound": 100.0,
                "open_lower_bound": False,
                "open_upper_bound": False,
                "zero_point": None,
                "unit_of_measure": None,
            },
        }

        with pytest.raises(KeyError):
            _build_question_shim_from_manifest_entry(1, entry)

    def test_build_question_shim_raises_when_numeric_metadata_missing_upper_bound(self) -> None:
        from metaculus_bot.ablation.cli import _build_question_shim_from_manifest_entry

        entry = {
            "type": "numeric",
            "tournament": "spring-aib-2026",
            "question_text": "What value?",
            "page_url": "https://example.com/q/1",
            "ground_truth": {},
            "resolution_criteria": "Resolves to a number.",
            "fine_print": "",
            "background_info": "",
            "question_metadata": {
                "open_time": "2026-01-01T00:00:00",
                "scheduled_resolution_time": "2026-05-01T00:00:00",
                "lower_bound": 0.0,
                # upper_bound missing.
                "open_lower_bound": False,
                "open_upper_bound": False,
                "zero_point": None,
                "unit_of_measure": None,
            },
        }

        with pytest.raises(KeyError):
            _build_question_shim_from_manifest_entry(1, entry)

    def test_build_question_shim_raises_when_mc_metadata_missing_options(self) -> None:
        """MC questions ALWAYS have ``options``; missing → drift."""
        from metaculus_bot.ablation.cli import _build_question_shim_from_manifest_entry

        entry = {
            "type": "multiple_choice",
            "tournament": "spring-aib-2026",
            "question_text": "Which option?",
            "page_url": "https://example.com/q/1",
            "ground_truth": {},
            "resolution_criteria": "Resolves to the correct option.",
            "fine_print": "",
            "background_info": "",
            "question_metadata": {
                "open_time": "2026-01-01T00:00:00",
                "scheduled_resolution_time": "2026-05-01T00:00:00",
            },  # options missing.
        }

        with pytest.raises(KeyError):
            _build_question_shim_from_manifest_entry(1, entry)

    def test_build_question_shim_raises_when_resolution_criteria_missing(self) -> None:
        """``resolution_criteria`` is always written by ``_build_manifest_entry``.

        The leakage detector at ``backtest/leakage.py:86`` reads
        ``question.resolution_criteria``; a missing key is schema drift.
        """
        from metaculus_bot.ablation.cli import _build_question_shim_from_manifest_entry

        entry = {
            "type": "binary",
            "tournament": "spring-aib-2026",
            "question_text": "Will it happen?",
            "page_url": "https://example.com/q/1",
            "ground_truth": {},
            # resolution_criteria deliberately missing — schema drift.
            "fine_print": "",
            "background_info": "",
            "question_metadata": {
                "open_time": "2026-01-01T00:00:00",
                "scheduled_resolution_time": "2026-05-01T00:00:00",
            },
        }

        with pytest.raises(KeyError):
            _build_question_shim_from_manifest_entry(1, entry)

    def test_build_question_shim_raises_when_fine_print_missing(self) -> None:
        """``fine_print`` is always written by ``_build_manifest_entry``."""
        from metaculus_bot.ablation.cli import _build_question_shim_from_manifest_entry

        entry = {
            "type": "binary",
            "tournament": "spring-aib-2026",
            "question_text": "Will it happen?",
            "page_url": "https://example.com/q/1",
            "ground_truth": {},
            "resolution_criteria": "Resolves YES if ...",
            # fine_print deliberately missing.
            "background_info": "",
            "question_metadata": {
                "open_time": "2026-01-01T00:00:00",
                "scheduled_resolution_time": "2026-05-01T00:00:00",
            },
        }

        with pytest.raises(KeyError):
            _build_question_shim_from_manifest_entry(1, entry)

    def test_build_question_shim_raises_when_background_info_missing(self) -> None:
        """``background_info`` is always written by ``_build_manifest_entry``."""
        from metaculus_bot.ablation.cli import _build_question_shim_from_manifest_entry

        entry = {
            "type": "binary",
            "tournament": "spring-aib-2026",
            "question_text": "Will it happen?",
            "page_url": "https://example.com/q/1",
            "ground_truth": {},
            "resolution_criteria": "Resolves YES if ...",
            "fine_print": "",
            # background_info deliberately missing.
            "question_metadata": {
                "open_time": "2026-01-01T00:00:00",
                "scheduled_resolution_time": "2026-05-01T00:00:00",
            },
        }

        with pytest.raises(KeyError):
            _build_question_shim_from_manifest_entry(1, entry)

    def test_build_question_shim_keeps_zero_point_optional_for_numeric(self) -> None:
        """``zero_point`` IS legitimately optional (None for linear-scale numerics)."""
        from metaculus_bot.ablation.cli import _build_question_shim_from_manifest_entry

        entry_without_zero_point = {
            "type": "numeric",
            "tournament": "spring-aib-2026",
            "question_text": "What value?",
            "page_url": "https://example.com/q/1",
            "ground_truth": {},
            "resolution_criteria": "Resolves to a number.",
            "fine_print": "",
            "background_info": "",
            "question_metadata": {
                "open_time": "2026-01-01T00:00:00",
                "scheduled_resolution_time": "2026-05-01T00:00:00",
                "lower_bound": 0.0,
                "upper_bound": 100.0,
                "open_lower_bound": False,
                "open_upper_bound": False,
                "unit_of_measure": None,
                # zero_point key absent — must be tolerated (linear-scale).
            },
        }

        q = _build_question_shim_from_manifest_entry(1, entry_without_zero_point)
        assert q.zero_point is None

    @pytest.mark.asyncio
    async def test_stage_fetch_raises_when_question_set_missing_ground_truth_for_qid(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fetcher invariant: every question has a matching ground_truth.

        The orchestrator must not silently skip questions whose ground_truth was
        dropped — that masks a fetcher bug and produces a smaller-than-expected
        ablation set.
        """
        from metaculus_bot.ablation.cli import WorkingSet as _WorkingSet
        from metaculus_bot.ablation.cli import _build_parser, _stage_fetch

        q1 = _make_binary_question(901)
        q2 = _make_binary_question(902)
        # Build a question_set where 902 is missing from ground_truths.
        question_set = BacktestQuestionSet(
            questions=[q1, q2],
            ground_truths={901: _make_binary_ground_truth(901)},
            fetch_metadata={
                "tournaments": ["spring-aib-2026"],
                "resolved_after": "2026-01-01",
                "resolved_before": None,
                "total_clean": 2,
                "type_distribution": {},
                "per_tournament_raw_counts": {},
                "per_type_targets": {},
                "per_type_actual": {},
                "skipped_no_resolution_time": 0,
                "skipped_too_early": 0,
                "skipped_too_late": 0,
                "skipped_canceled": 0,
            },
        )
        fetch_mock = AsyncMock(return_value=question_set)
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.fetch_resolved_questions_stratified",
            fetch_mock,
        )
        # _stage_fetch preflights the API identity (real network) before fetching; stub it.
        monkeypatch.setattr("metaculus_bot.ablation.cli.verify_metaculus_api_identity", MagicMock())

        cache = AblationCache(str(cache_dir))
        working = _WorkingSet()
        args = _build_parser().parse_args(["--num-binary", "2", "--cache-dir", str(cache_dir)])

        with pytest.raises(KeyError):
            await _stage_fetch(args, cache, working)

    def test_print_spend_report_raises_when_verdict_missing_is_leaked_key(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Verdict dicts are built by leakage_screen._build_verdict with a fixed schema.

        ``verdict.get("is_leaked")`` returning None (falsy) under-counts leaked qids
        and biases the ablation result. A missing key is an invariant violation —
        we want a KeyError, not silent under-count.
        """
        from metaculus_bot.ablation.cli import SpendReport as _SpendReport
        from metaculus_bot.ablation.cli import WorkingSet as _WorkingSet
        from metaculus_bot.ablation.cli import _print_spend_report

        working = _WorkingSet()
        working.questions = {1: _make_binary_question(1)}
        working.leakage_verdicts = {
            1: {
                # is_leaked deliberately missing — schema drift.
                "detector_response": "...",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T00:00:00",
            }
        }
        spend = _SpendReport()

        with pytest.raises(KeyError):
            _print_spend_report(spend, working, summary_path=None)

    @pytest.mark.asyncio
    async def test_run_ablation_screen_loop_raises_when_verdict_missing_is_leaked_key(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The post-screen log line counts leaked qids; a missing key is drift."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(1001)
        gt1 = _make_binary_ground_truth(1001)
        question_set = _build_question_set([(q1, gt1)])

        # Verdict missing the is_leaked key entirely.
        verdicts = {
            1001: {
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={1001: ("blob 1001", {})},
            leakage_verdicts=verdicts,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        # Run only up to screen so we hit the screen-stage post-loop counter.
        args = _build_parser().parse_args(
            [
                "--num-binary",
                "1",
                "--cache-dir",
                str(cache_dir),
                "--stages",
                "fetch,research,screen",
            ]
        )

        with pytest.raises(KeyError):
            await run_ablation(args)

    def test_qa_research_dump_skips_qids_without_research_blob(
        self,
        cache_dir: Path,
    ) -> None:
        """When the QA dump runs, qids without a research blob must be skipped explicitly.

        The old defensive default emitted a sentinel string ``(no research blob — research stage failed)``
        as if the research had been attempted; a missing blob means the qid was
        dropped upstream and should not be reported as a "Q<n>" section in the
        QA file.
        """
        from metaculus_bot.ablation.cli import WorkingSet as _WorkingSet
        from metaculus_bot.ablation.cli import _build_parser, _stage_qa_research_dump

        cache = AblationCache(str(cache_dir))
        working = _WorkingSet()
        working.questions = {1: _make_binary_question(1)}
        working.ground_truths = {1: _make_binary_ground_truth(1)}
        # research_blobs deliberately empty — research stage dropped qid 1.
        working.leakage_verdicts = {
            1: {
                "is_leaked": False,
                "detector_response": "ok",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T00:00:00",
            }
        }

        args = _build_parser().parse_args(["--num-binary", "1", "--cache-dir", str(cache_dir), "--qa-research"])

        path = _stage_qa_research_dump(args, cache, working)
        text = path.read_text(encoding="utf-8")
        # The qid is marked as skipped, NOT presented as a normal Q-section.
        assert "(skipped — no research blob)" in text
        # The sentinel string the old code emitted must not appear.
        assert "(no research blob — research stage failed)" not in text
        # The regular Q-section body fields (Leaked, Detector verdict) must not appear.
        assert "- Leaked:" not in text
        assert "### Detector verdict" not in text


# ---------------------------------------------------------------------------
# run_stacker.py — fixed forecaster payload schema
# ---------------------------------------------------------------------------


class TestStackerForecasterPayloadSchema:
    @pytest.mark.asyncio
    async def test_run_stacker_for_arm_raises_when_forecaster_payload_missing_model_key(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Forecaster payloads have a fixed schema (forecasters.run_forecasters_batch).

        A missing ``model`` key indicates schema drift; we want KeyError, not a
        silent fallback to the slug filename.
        """
        from forecasting_tools import BinaryPrediction

        from metaculus_bot.ablation.run_stacker import run_stacker_for_arm

        q = _make_binary_question(2001)

        # Two payloads (>= ABLATION_MIN_FORECASTERS) so we get past the survival check.
        # Each payload deliberately missing the ``model`` key.
        broken_payload_a = {
            "prediction_value": {"type": "binary", "prob": 0.6},
            "reasoning": "Model: openrouter/test/m0\n\nrationale text",
            "errors": [],
            "ran_at": "2026-05-13T12:00:00",
            "duration_seconds": 1.5,
        }
        broken_payload_b = {
            "prediction_value": {"type": "binary", "prob": 0.55},
            "reasoning": "Model: openrouter/test/m1\n\nrationale text",
            "errors": [],
            "ran_at": "2026-05-13T12:00:00",
            "duration_seconds": 1.5,
        }

        cache = AblationCache(str(cache_dir))

        async def _noop_run_stacking_binary(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0)
            return MagicMock(spec=BinaryPrediction, prediction_value=0.7), "stacker reasoning"

        monkeypatch.setattr(
            "metaculus_bot.ablation.run_stacker.stacking.run_stacking_binary",
            _noop_run_stacking_binary,
        )

        with pytest.raises(KeyError):
            await run_stacker_for_arm(
                question=q,
                research_blob="research blob",
                forecaster_payloads={
                    "openrouter_test_m0": broken_payload_a,
                    "openrouter_test_m1": broken_payload_b,
                },
                arm="stack",
                cache=cache,
            )
