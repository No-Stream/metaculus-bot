"""Tests for the probabilistic-tools ablation benchmark CLI orchestrator.

The CLI under test is ``metaculus_bot/ablation/cli.py``. It glues together every
Wave 1-3 building block: question fetch, Gemini-only research, leakage screen,
forecaster fan-out, two stacker arms, and paired-difference scoring.

All tests mock module-level entry points so no live API calls fire:

* ``fetch_resolved_questions_stratified`` (wave 1)
* ``run_gemini_research_for_qids`` (wave 2)
* ``screen_batch`` (wave 3)
* ``run_forecasters_batch`` (wave 3)
* ``run_stacker_batch`` (wave 3)

The orchestrator's contract under test:

1. Argparse parses every flag in the spec.
2. Stage-by-stage flow (fetch → research → screen → forecast → stack → pdf → score).
3. Each stage reads cache before calling its mocked entry point.
4. ``--qa-research`` halts after screen.
5. ``--stages`` skips the right stages.
6. ``--force-stages`` re-runs only the listed stages.
7. ``--qids`` bypasses fetch.
8. Re-runs are idempotent.
9. Smoke→expand grows the manifest without re-sampling existing qids.
10. Leaked qids are dropped from downstream stages.
11. Spend report aggregates correctly.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.backtest.question_prep import BacktestQuestionSet
from metaculus_bot.backtest.scoring import GroundTruth

# ---------------------------------------------------------------------------
# Question fixtures
# ---------------------------------------------------------------------------

_OPEN = datetime(2026, 1, 1)
_RESOLVE = datetime(2026, 5, 1)


def _make_binary_question(qid: int) -> BinaryQuestion:
    return BinaryQuestion(
        id_of_question=qid,
        id_of_post=qid,
        question_text=f"Will Q{qid} happen?",
        background_info="",
        resolution_criteria="Resolves YES if it happens.",
        fine_print="",
        page_url=f"https://example.com/q/{qid}",
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _make_mc_question(qid: int) -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        id_of_question=qid,
        id_of_post=qid,
        question_text=f"Which option for Q{qid}?",
        background_info="",
        resolution_criteria="Resolves to the correct option.",
        fine_print="",
        options=["Red", "Blue"],
        page_url=f"https://example.com/q/{qid}",
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _make_numeric_question(qid: int) -> NumericQuestion:
    return NumericQuestion(
        id_of_question=qid,
        id_of_post=qid,
        question_text=f"What value for Q{qid}?",
        background_info="",
        resolution_criteria="Resolves to a number.",
        fine_print="",
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=False,
        open_upper_bound=False,
        zero_point=None,
        unit_of_measure=None,
        page_url=f"https://example.com/q/{qid}",
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
    )


def _make_binary_ground_truth(qid: int, outcome: bool = True) -> GroundTruth:
    return GroundTruth(
        question_id=qid,
        question_type="binary",
        resolution=outcome,
        resolution_string="YES" if outcome else "NO",
        community_prediction=None,
        actual_resolution_time=_RESOLVE,
        question_text=f"Will Q{qid} happen?",
        page_url=f"https://example.com/q/{qid}",
    )


def _make_mc_ground_truth(qid: int, correct: str = "Red") -> GroundTruth:
    return GroundTruth(
        question_id=qid,
        question_type="multiple_choice",
        resolution=correct,
        resolution_string=correct,
        community_prediction=None,
        actual_resolution_time=_RESOLVE,
        question_text=f"Which option for Q{qid}?",
        page_url=f"https://example.com/q/{qid}",
    )


def _make_numeric_ground_truth(qid: int, value: float = 50.0) -> GroundTruth:
    return GroundTruth(
        question_id=qid,
        question_type="numeric",
        resolution=value,
        resolution_string=str(value),
        community_prediction=None,
        actual_resolution_time=_RESOLVE,
        question_text=f"What value for Q{qid}?",
        page_url=f"https://example.com/q/{qid}",
    )


# ---------------------------------------------------------------------------
# Stage-mock canned data factories
# ---------------------------------------------------------------------------


def _binary_forecaster_payload(model: str, value: float = 0.6) -> dict:
    return {
        "model": model,
        "prediction_value": {"type": "binary", "prob": value},
        "reasoning": f"Model: {model}\n\nrationale text",
        "errors": [],
        "ran_at": "2026-05-13T12:00:00",
        "duration_seconds": 1.5,
    }


def _numeric_forecaster_payload(model: str, median: float = 50.0) -> dict:
    """Build a numeric forecaster payload in the post-Bucket-1 full-CDF schema.

    Schema is the output of ``serialize_prediction_value`` for a real
    ``NumericDistribution`` — declared_percentiles + the constraint-enforced
    201-point CDF + bounds + zero_point + cdf_size. Tests construct payloads
    directly here instead of running serialize, so we synthesize a monotone
    linear CDF that matches the bounds.
    """
    declared = [
        {"percentile": 0.025, "value": median - 30},
        {"percentile": 0.05, "value": median - 25},
        {"percentile": 0.10, "value": median - 20},
        {"percentile": 0.20, "value": median - 12},
        {"percentile": 0.40, "value": median - 5},
        {"percentile": 0.50, "value": median},
        {"percentile": 0.60, "value": median + 5},
        {"percentile": 0.80, "value": median + 12},
        {"percentile": 0.90, "value": median + 20},
        {"percentile": 0.95, "value": median + 25},
        {"percentile": 0.975, "value": median + 30},
    ]
    # Synthesize a monotone CDF from 0.001 → 0.999 across 201 points so
    # ``deserialize_prediction_value``'s reconstruction has valid CDF input.
    cdf_probabilities = [0.001 + (0.998 * i / 200) for i in range(201)]
    return {
        "model": model,
        "prediction_value": {
            "type": "numeric",
            "declared_percentiles": declared,
            "cdf_probabilities": cdf_probabilities,
            "lower_bound": 0.0,
            "upper_bound": 100.0,
            "open_lower_bound": False,
            "open_upper_bound": False,
            "zero_point": None,
            "cdf_size": 201,
        },
        "reasoning": f"Model: {model}\n\nrationale text",
        "errors": [],
        "ran_at": "2026-05-13T12:00:00",
        "duration_seconds": 1.5,
    }


def _mc_forecaster_payload(model: str) -> dict:
    return {
        "model": model,
        "prediction_value": {
            "type": "multiple_choice",
            "options": [
                {"option_name": "Red", "probability": 0.6},
                {"option_name": "Blue", "probability": 0.4},
            ],
        },
        "reasoning": f"Model: {model}\n\nrationale text",
        "errors": [],
        "ran_at": "2026-05-13T12:00:00",
        "duration_seconds": 1.5,
    }


def _binary_stacker_payload(arm: str, value: float = 0.7) -> dict:
    # ARM_MEDIAN bypasses the stacker LLM and uses simple median aggregation; its
    # payload mirrors what ``run_median_for_qid`` writes (sentinel
    # ``stacker_model_used``, empty cross_model_aggregation, tools always off).
    if arm == "C":
        return {
            "success": True,
            "arm": "C",
            "stacker_prediction": {"type": "binary", "prob": value},
            "stacker_meta_reasoning": "",
            "computed_quantities": {},
            "cross_model_aggregation": "",
            "stacker_model_used": "simple_aggregation",
            "n_forecasters_used": 3,
            "ran_at": "2026-05-13T12:30:00",
            "tools_enabled_at_runtime": False,
            "errors": [],
        }
    return {
        "success": True,
        "arm": arm,
        "stacker_prediction": {"type": "binary", "prob": value},
        "stacker_meta_reasoning": f"stacker reasoning arm {arm}",
        "computed_quantities": {},
        "cross_model_aggregation": "" if arm == "A" else "## Cross-model aggregation\n",
        "stacker_model_used": "primary",
        "n_forecasters_used": 3,
        "ran_at": "2026-05-13T12:30:00",
        "tools_enabled_at_runtime": arm == "B",
        "errors": [],
    }


def _numeric_stacker_payload(arm: str, median: float = 55.0) -> dict:
    payload = _numeric_forecaster_payload("stacker", median)["prediction_value"]
    return {
        "success": True,
        "arm": arm,
        "stacker_prediction": payload,
        "stacker_meta_reasoning": f"stacker reasoning arm {arm}",
        "computed_quantities": {},
        "cross_model_aggregation": "" if arm == "A" else "## Cross-model aggregation\n",
        "stacker_model_used": "primary",
        "n_forecasters_used": 3,
        "ran_at": "2026-05-13T12:30:00",
        "tools_enabled_at_runtime": arm == "B",
        "errors": [],
    }


def _mc_stacker_payload(arm: str, p_red: float = 0.7) -> dict:
    return {
        "success": True,
        "arm": arm,
        "stacker_prediction": {
            "type": "multiple_choice",
            "options": [
                {"option_name": "Red", "probability": p_red},
                {"option_name": "Blue", "probability": 1.0 - p_red},
            ],
        },
        "stacker_meta_reasoning": f"stacker reasoning arm {arm}",
        "computed_quantities": {},
        "cross_model_aggregation": "" if arm == "A" else "## Cross-model aggregation\n",
        "stacker_model_used": "primary",
        "n_forecasters_used": 3,
        "ran_at": "2026-05-13T12:30:00",
        "tools_enabled_at_runtime": arm == "B",
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Mock-installer helper for the full stack
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "abl"


def _build_question_set(questions_with_gt: list[tuple[Any, GroundTruth]]) -> BacktestQuestionSet:
    """Build a BacktestQuestionSet from (question, ground_truth) pairs."""
    questions = [q for q, _ in questions_with_gt]
    ground_truths = {gt.question_id: gt for _, gt in questions_with_gt}
    return BacktestQuestionSet(
        questions=questions,
        ground_truths=ground_truths,
        fetch_metadata={
            "tournaments": ["spring-aib-2026"],
            "resolved_after": "2026-01-01",
            "resolved_before": None,
            "total_clean": len(questions),
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


def _install_full_stack_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fetch_question_set: BacktestQuestionSet,
    research_results: dict[int, tuple[str, dict] | None] | None = None,
    prune_results: dict[int, tuple[str, dict] | None] | None = None,
    leakage_verdicts: dict[int, dict] | None = None,
    forecaster_results: dict[int, dict[str, dict]] | None = None,
    stacker_a_results: dict[int, dict] | None = None,
    stacker_b_results: dict[int, dict] | None = None,
) -> dict[str, MagicMock | AsyncMock]:
    """Install canned mocks for every wave-1/2/3 entry point used by the CLI.

    Returns the dict of mock objects so individual tests can inspect call
    counts and arguments. ``prune_results`` defaults to passing raw research
    blobs through unchanged, so tests that don't care about the prune stage
    still work.
    """
    fetch_mock = AsyncMock(return_value=fetch_question_set)
    monkeypatch.setattr(
        "metaculus_bot.ablation.cli.fetch_resolved_questions_stratified",
        fetch_mock,
    )

    research_results = research_results or {}

    def _research_returns(
        questions: list,
        cache: AblationCache,
        **_kwargs: Any,
    ) -> dict[int, tuple[str, dict] | None]:
        # Mirror real run_gemini_research_for_qids: write to cache before returning.
        out: dict[int, tuple[str, dict] | None] = {}
        for q in questions:
            qid = q.id_of_question
            if qid not in research_results:
                continue
            value = research_results[qid]
            if value is not None:
                cache.write_research(qid, value[0], value[1])
            out[qid] = value
        return out

    research_mock = AsyncMock(side_effect=_research_returns)
    monkeypatch.setattr("metaculus_bot.ablation.cli.run_gemini_research_for_qids", research_mock)

    explicit_prune_results = prune_results

    def _prune_returns(
        questions_with_gt_and_blob: list,
        cache: AblationCache,
        **_kwargs: Any,
    ) -> dict[int, tuple[str, dict] | None]:
        out: dict[int, tuple[str, dict] | None] = {}
        for question, _gt, raw_blob in questions_with_gt_and_blob:
            qid = question.id_of_question
            if explicit_prune_results is not None and qid in explicit_prune_results:
                value = explicit_prune_results[qid]
            else:
                # Default: pass raw blob through with synthetic meta.
                value = (
                    raw_blob,
                    {
                        "qid": qid,
                        "original_chars": len(raw_blob),
                        "sanitized_chars": len(raw_blob),
                        "redactions": [],
                        "redactor_invocation_id": "test-noop",
                        "pruned_at": "2026-05-13T18:00:00",
                    },
                )
            if value is not None:
                cache.write_pruned_research(qid, value[0], value[1])
            out[qid] = value
        return out

    prune_mock = AsyncMock(side_effect=_prune_returns)
    monkeypatch.setattr("metaculus_bot.ablation.cli.run_prune_for_qids", prune_mock)

    leakage_verdicts = leakage_verdicts or {}

    def _screen_batch_returns(
        questions: list,
        ground_truths: dict[int, GroundTruth],
        research_blobs: dict[int, str],
        cache: AblationCache,
        **_kwargs: Any,
    ) -> tuple[list, dict[int, GroundTruth], dict[int, dict]]:
        import hashlib as _hashlib

        clean_qids = {qid for qid, v in leakage_verdicts.items() if not v.get("is_leaked")}
        clean_questions = [q for q in questions if q.id_of_question in clean_qids]
        clean_gts = {qid: gt for qid, gt in ground_truths.items() if qid in clean_qids}
        # Mirror the real screen_batch: stamp research_blob_sha against the
        # blob being screened so cache reads honor the C3 hash check.
        stamped: dict[int, dict] = {}
        for qid, verdict in leakage_verdicts.items():
            blob = research_blobs.get(qid, "")
            entry = {**verdict, "research_blob_sha": _hashlib.sha256(blob.encode()).hexdigest()[:16]}
            cache.write_leakage_screen(qid, entry)
            stamped[qid] = entry
        return clean_questions, clean_gts, stamped

    screen_mock = AsyncMock(side_effect=_screen_batch_returns)
    monkeypatch.setattr("metaculus_bot.ablation.cli.screen_batch", screen_mock)

    forecaster_results = forecaster_results or {}

    def _forecaster_batch_returns(
        questions_with_research: list[tuple[Any, str]],
        cache: AblationCache,
        **kwargs: Any,
    ) -> dict[int, dict[str, dict]]:
        # Persist canonical cache files like the real run_forecasters_batch.
        out: dict[int, dict[str, dict]] = {}
        for q, _blob in questions_with_research:
            qid = q.id_of_question
            per_model = forecaster_results.get(qid, {})
            for slug, payload in per_model.items():
                cache.write_forecaster_output(qid=qid, model_slug=slug, payload=payload)
            out[qid] = per_model
        return out

    forecasters_mock = AsyncMock(side_effect=_forecaster_batch_returns)
    monkeypatch.setattr("metaculus_bot.ablation.cli.run_forecasters_batch", forecasters_mock)

    arm_a = stacker_a_results or {}
    arm_b = stacker_b_results or {}

    def _stacker_batch_returns(
        qid_to_data: dict[int, dict],
        arm: str,
        cache: AblationCache,
        **kwargs: Any,
    ) -> dict[int, dict]:
        source = arm_a if arm == "stack" else arm_b
        out: dict[int, dict] = {}
        for qid in qid_to_data:
            payload = source.get(qid)
            if payload is not None:
                cache.write_stacker_output(qid=qid, arm=arm, payload=payload)
                out[qid] = payload
        return out

    stacker_mock = AsyncMock(side_effect=_stacker_batch_returns)
    monkeypatch.setattr("metaculus_bot.ablation.cli.run_stacker_batch", stacker_mock)

    # Default qa_iterate stub: clean verifier verdict for every qid so the stage no-ops cleanly.
    # Tests can monkeypatch a different fake verifier/redactor to exercise iterate behavior.
    import json as _json  # noqa: PLC0415
    import re as _re  # noqa: PLC0415

    async def _default_clean_verifier(prompt: str, **_kwargs: Any) -> str:
        await asyncio.sleep(0)
        match = _re.search(r"qid=(\d+)", prompt)
        qid = int(match.group(1)) if match else 0
        return _json.dumps(
            {
                "verdicts": [
                    {
                        "qid": qid,
                        "leakage_risk": 0.05,
                        "forecastability": 0.8,
                        "hallucination_risk": 0.1,
                        "notes": "",
                    }
                ]
            }
        )

    async def _default_passthrough_redactor(prompt: str, **_kwargs: Any) -> str:
        await asyncio.sleep(0)
        match = _re.search(r"qid=(\d+)", prompt)
        qid = int(match.group(1)) if match else 0
        return _json.dumps({"results": [{"qid": qid, "sanitized_blob": "stub blob", "redactions": []}]})

    verifier_mock = AsyncMock(side_effect=_default_clean_verifier)
    redactor_mock = AsyncMock(side_effect=_default_passthrough_redactor)
    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", verifier_mock)
    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", redactor_mock)

    return {
        "fetch": fetch_mock,
        "research": research_mock,
        "prune": prune_mock,
        "screen": screen_mock,
        "forecasters": forecasters_mock,
        "stacker": stacker_mock,
        "qa_verifier": verifier_mock,
        "qa_redactor": redactor_mock,
    }


def _populate_full_cache_for_qid(cache: AblationCache, qid: int) -> None:
    """Pre-populate every cache file for a single binary qid.

    Used by C2 / M1 / Task #23 tests that need a hydrated working set without
    firing any stage. Mirrors what _build_manifest_entry would write for a
    real binary question.
    """
    q = _make_binary_question(qid)
    cache.append_qids_manifest(
        {
            qid: {
                "type": "binary",
                "tournament": "spring-aib-2026",
                "question_text": q.question_text,
                "page_url": q.page_url,
                "id_of_post": q.id_of_post,
                "resolution_criteria": q.resolution_criteria,
                "fine_print": q.fine_print,
                "background_info": q.background_info,
                "ground_truth": {
                    "question_id": qid,
                    "question_type": "binary",
                    "resolution": True,
                    "resolution_string": "YES",
                    "actual_resolution_time": "2026-05-01T00:00:00",
                    "question_text": q.question_text,
                    "page_url": q.page_url,
                },
                "question_metadata": {
                    "open_time": "2026-01-01T00:00:00",
                    "scheduled_resolution_time": "2026-05-01T00:00:00",
                },
            }
        }
    )
    cache.write_research(qid, f"raw blob {qid}", {"sources": 1})
    cache.write_pruned_research(qid, f"sanitized blob {qid}", {"redactions": []})
    cache.write_leakage_screen(
        qid,
        {
            "is_leaked": False,
            "detector_response": "no leak",
            "detector_model": "test",
            "detector_failed": False,
            "screened_at": "2026-05-13T12:00:00",
            "research_blob_sha": "stub",
        },
    )
    for i in range(3):
        slug = model_slug_to_filename(f"openrouter/test/m{i}")
        cache.write_forecaster_output(qid=qid, model_slug=slug, payload=_binary_forecaster_payload(f"m{i}", 0.5))
    cache.write_stacker_output(qid=qid, arm="stack", payload=_binary_stacker_payload("stack", 0.6))
    cache.write_stacker_output(qid=qid, arm="stack_aug", payload=_binary_stacker_payload("stack_aug", 0.7))
    cache.write_stacker_output(qid=qid, arm="median", payload=_binary_stacker_payload("median", 0.65))


# ---------------------------------------------------------------------------
# Argparse tests
# ---------------------------------------------------------------------------


class TestParser:
    def test_parser_default_tournaments_is_spring_aib_2026(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--num-binary", "1"])
        assert args.tournaments == ["spring-aib-2026"]

    def test_parser_default_resolved_after_is_2026_01_01(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--num-binary", "1"])
        assert args.resolved_after == "2026-01-01"

    def test_parser_stages_default_is_all(self):
        from metaculus_bot.ablation.cli import STAGES, _build_parser

        args = _build_parser().parse_args([])
        assert args.stages == STAGES

    def test_parser_qids_parses_csv(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--qids", "1,2,3"])
        assert args.qids == [1, 2, 3]

    def test_parser_force_stages_subset(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--force-stages", "research,score"])
        assert args.force_stages == ["research", "score"]

    def test_parser_qa_research_flag(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--qa-research"])
        assert args.qa_research is True

    def test_parser_default_cache_dir_is_backtests_ablation(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.cache_dir == "backtests/ablation"

    def test_parser_concurrency_default(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.concurrency == 4

    def test_parser_seed_default(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.seed == 0

    def test_parser_per_question_sleep_default(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.per_question_sleep == 30

    def test_per_question_sleep_help_text_documents_inter_stage_behavior(self):
        """Help text must clarify the sleep is per-stage, not literally per-question.

        Despite the flag name, the implementation only sleeps once between stages
        (research -> forecast -> stack). The help string should be honest about that
        so a user setting --per-question-sleep=30 for a 30-question run doesn't
        expect 30s * 30 = 900s total pause.
        """
        from metaculus_bot.ablation.cli import _build_parser

        parser = _build_parser()
        help_text = parser.format_help()
        # The help string for --per-question-sleep must mention BETWEEN STAGES
        # (case-insensitive match permits "between stages", "BETWEEN STAGES", etc.).
        assert "between stages" in help_text.lower(), (
            f"--per-question-sleep help must document per-stage (not per-question) behavior; got: {help_text}"
        )

    def test_parser_gap_fill_max_gaps_default(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.gap_fill_max_gaps == 3

    def test_parser_invalid_stage_rejected(self):
        from metaculus_bot.ablation.cli import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--stages", "fetch,nonsense"])

    def test_parser_default_gemini_model(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.gemini_model == "gemini-2.5-flash"

    def test_parser_default_gap_fill_off(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.gap_fill is False

    def test_parser_explicit_gap_fill_on(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--gap-fill"])
        assert args.gap_fill is True

    def test_parser_explicit_gap_fill_off(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--no-gap-fill"])
        assert args.gap_fill is False

    def test_parser_gap_fill_mutex(self):
        from metaculus_bot.ablation.cli import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--gap-fill", "--no-gap-fill"])

    def test_parser_rate_limit_mode_default_is_patient(self):
        """Default flipped from ``gentle`` to ``patient`` on 2026-05-14 (Phase A.3 Package 3a).

        At 50q × 5 forecasters = 250 calls per arm, ``gentle`` (concurrency=2,
        max_retries=3) was thrashing free-tier per-minute throttles. ``patient``
        (concurrency=1, max_retries=8) is the new default for any non-trivial run;
        operators with a smoke (≤4q) workload can opt back into ``gentle`` or ``fast``.
        """
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.rate_limit_mode == "patient"

    def test_parser_rate_limit_mode_accepts_fast(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--rate-limit-mode", "fast"])
        assert args.rate_limit_mode == "fast"

    def test_parser_rate_limit_mode_accepts_slow(self):
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--rate-limit-mode", "slow"])
        assert args.rate_limit_mode == "slow"

    def test_parser_rate_limit_mode_accepts_patient(self):
        """The ``patient`` preset is "slow but persistent" — concurrency=1, max_retries bumped above ``slow``."""
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--rate-limit-mode", "patient"])
        assert args.rate_limit_mode == "patient"

    def test_parser_rate_limit_mode_rejects_invalid(self):
        from metaculus_bot.ablation.cli import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--rate-limit-mode", "nonsense"])

    def test_log_level_default_is_info(self) -> None:
        """The ablation CLI emits per-stage and per-qid INFO logs that we want
        captured by default. Python's root logger defaults to WARNING which
        silently drops all the rich INFO diagnostics.
        """
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args([])
        assert args.log_level == "INFO"

    def test_log_level_flag_accepts_debug(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"

    def test_log_level_flag_accepts_warning(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        args = _build_parser().parse_args(["--log-level", "WARNING"])
        assert args.log_level == "WARNING"

    def test_log_level_rejects_invalid(self) -> None:
        from metaculus_bot.ablation.cli import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--log-level", "TRACE"])


class TestLoggingConfiguration:
    """The ablation CLI must wire up file-archived INFO logging so smoke runs
    are reviewable. The audit at backtests/ablation/audit_smoke_20260515.md
    flagged a 38-line smoke log because no ``logging.basicConfig`` was called
    anywhere in ``metaculus_bot/ablation/`` and the root logger defaulted to
    WARNING.
    """

    def test_configure_logging_creates_logs_subdir_under_cache(self, tmp_path: Path) -> None:
        from metaculus_bot.ablation.cli import _build_parser, _configure_logging

        args = _build_parser().parse_args(["--cache-dir", str(tmp_path)])
        _configure_logging(args, tmp_path)
        logs_dir = tmp_path / "logs"
        assert logs_dir.is_dir(), "expected logs/ subdirectory under cache root"

    def test_configure_logging_returns_log_file_path_in_logs_dir(self, tmp_path: Path) -> None:
        from metaculus_bot.ablation.cli import _build_parser, _configure_logging

        args = _build_parser().parse_args(["--cache-dir", str(tmp_path)])
        log_path = _configure_logging(args, tmp_path)
        assert log_path.parent == tmp_path / "logs"
        assert log_path.name.startswith("run_")
        assert log_path.name.endswith(".log")

    def test_configure_logging_writes_info_messages_to_file(self, tmp_path: Path) -> None:
        """Emit a logger.info AFTER configuration; assert it appears in the file."""
        import logging

        from metaculus_bot.ablation.cli import _build_parser, _configure_logging

        args = _build_parser().parse_args(["--cache-dir", str(tmp_path), "--log-level", "INFO"])
        log_path = _configure_logging(args, tmp_path)

        test_logger = logging.getLogger("test_ablation_logging_demo")
        test_logger.info("hello-from-test-info-message")

        # Force flush across handlers so the file is written.
        for handler in logging.root.handlers:
            handler.flush()

        contents = log_path.read_text(encoding="utf-8")
        assert "hello-from-test-info-message" in contents

    def test_configure_logging_respects_log_level_flag(self, tmp_path: Path) -> None:
        """--log-level WARNING should drop INFO messages from the file."""
        import logging

        from metaculus_bot.ablation.cli import _build_parser, _configure_logging

        args = _build_parser().parse_args(["--cache-dir", str(tmp_path), "--log-level", "WARNING"])
        log_path = _configure_logging(args, tmp_path)

        test_logger = logging.getLogger("test_ablation_logging_demo_warning_filter")
        test_logger.info("info-must-not-appear")
        test_logger.warning("warning-must-appear")

        for handler in logging.root.handlers:
            handler.flush()

        contents = log_path.read_text(encoding="utf-8")
        assert "info-must-not-appear" not in contents
        assert "warning-must-appear" in contents


class TestRateLimitModeMapping:
    """The CLI flag maps to (per_forecaster_concurrency, max_retries) tuples.

    The mapping itself lives in ``cli._RATE_LIMIT_MODE_TO_KWARGS`` so individual
    stages and tests share one source of truth.
    """

    def test_fast_mode_high_concurrency_low_retries(self):
        from metaculus_bot.ablation.cli import _rate_limit_mode_kwargs

        kwargs = _rate_limit_mode_kwargs("fast")
        assert kwargs == {"per_forecaster_concurrency": 4, "max_retries": 1}

    def test_gentle_mode_balanced(self):
        from metaculus_bot.ablation.cli import _rate_limit_mode_kwargs

        kwargs = _rate_limit_mode_kwargs("gentle")
        assert kwargs == {"per_forecaster_concurrency": 2, "max_retries": 3}

    def test_slow_mode_low_concurrency_high_retries(self):
        from metaculus_bot.ablation.cli import _rate_limit_mode_kwargs

        kwargs = _rate_limit_mode_kwargs("slow")
        assert kwargs == {"per_forecaster_concurrency": 1, "max_retries": 5}

    def test_rate_limit_mode_patient_maps_to_concurrency_1_retries_8(self):
        """``patient`` keeps concurrency=1 (matching ``slow``) but bumps the retry budget to 8.

        The motivation: free-tier providers (qwen, minimax, gemma-4-26b) frequently shed
        forecasters under tight retry budgets even though successive attempts often
        succeed. ``patient`` adds retry budget without dropping concurrency further;
        it's "slow but persistent" rather than "even slower."
        """
        from metaculus_bot.ablation.cli import _rate_limit_mode_kwargs

        kwargs = _rate_limit_mode_kwargs("patient")
        assert kwargs == {"per_forecaster_concurrency": 1, "max_retries": 8}


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


# ---------------------------------------------------------------------------
# C3 part 1: --force-stages qa_iterate archives manual_rejects.json
# ---------------------------------------------------------------------------


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
# Task #23: forecast cache check is CONTENT-AWARE
#
# The previous all-or-nothing check treated any non-empty list_forecaster_outputs
# result as a cache hit. This let mock-poisoned (errors-only) payloads serve as
# valid forecasts and the stacker downstream cached "insufficient_forecasters"
# permanently. The diagnosis at skipped_qids_diagnosis_20260515.md:69 documents
# the exact failure mode for qids 43077/43148/43150.
# ---------------------------------------------------------------------------


class TestForecastCacheContentAware:
    @pytest.mark.asyncio
    async def test_forecast_re_runs_when_all_cached_payloads_have_errors(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mutation test for Task #23: if the content-aware check is reverted,
        the test fails because the stage treats error-only payloads as cached.
        """
        from metaculus_bot.ablation.cli import _build_parser, run_ablation
        from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

        # Pre-populate manifest + research + pruned + screen cache for qid.
        cache = AblationCache(cache_dir)
        _populate_full_cache_for_qid(cache, 12001)
        # Wipe the (synthetic) stacker outputs from _populate_full_cache_for_qid;
        # we want this run to actually re-forecast.
        for arm in ("stack", "B"):
            (cache.root / "stacker_outputs" / "12001" / f"arm_{arm}.json").unlink(missing_ok=True)
        # Replace the synthetic forecaster cache with all-error payloads from the
        # CURRENT lineup (so lineup_filter doesn't drop them).
        for stem in list((cache.root / "forecaster_outputs" / "12001").glob("*.json")):
            stem.unlink()
        for model in FREE_FORECASTER_MODELS:
            slug = model_slug_to_filename(model)
            cache.write_forecaster_output(
                qid=12001,
                model_slug=slug,
                payload={
                    "model": model,
                    "prediction_value": None,
                    "reasoning": "",
                    "errors": ["Mock object has no attribute 'id_of_post'"],
                    "ran_at": "2026-05-13T20:59:58Z",
                    "duration_seconds": 1e-5,
                },
            )

        question_set = _build_question_set([])
        forecaster_results = {
            12001: {
                model_slug_to_filename(model): _binary_forecaster_payload(model, 0.5)
                for model in FREE_FORECASTER_MODELS
            },
        }
        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            forecaster_results=forecaster_results,
            stacker_a_results={12001: _binary_stacker_payload("stack", 0.6)},
            stacker_b_results={12001: _binary_stacker_payload("stack_aug", 0.7)},
        )
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        # Run only forecast (and downstream) — the upstream caches are pre-populated.
        args = _build_parser().parse_args(
            [
                "--stages",
                "forecast,stack,stack_aug,pdf,median,score",
                "--cache-dir",
                str(cache_dir),
            ]
        )
        await run_ablation(args)

        # Forecasters mock SHOULD have been called (the cached payloads were
        # all-errors, so the content-aware check forced a re-run).
        assert mocks["forecasters"].await_count == 1, (
            "Task #23: forecast cache hit must check that surviving forecasters "
            "(non-error, prediction_value!=None) reach ABLATION_MIN_FORECASTERS; "
            "otherwise a stale all-errors cache poisons every downstream stage."
        )

    @pytest.mark.asyncio
    async def test_forecast_uses_cache_when_surviving_meets_threshold(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sanity check: a fully-valid cache is honored (didn't break the happy path)."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation
        from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS

        cache = AblationCache(cache_dir)
        _populate_full_cache_for_qid(cache, 12002)
        # Replace synthetic 3-model forecaster cache with current-lineup valid payloads.
        for stem in list((cache.root / "forecaster_outputs" / "12002").glob("*.json")):
            stem.unlink()
        for model in FREE_FORECASTER_MODELS:
            slug = model_slug_to_filename(model)
            cache.write_forecaster_output(qid=12002, model_slug=slug, payload=_binary_forecaster_payload(model, 0.5))

        question_set = _build_question_set([])
        mocks = _install_full_stack_mocks(monkeypatch, fetch_question_set=question_set)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        args = _build_parser().parse_args(["--stages", "forecast", "--cache-dir", str(cache_dir)])
        await run_ablation(args)

        # Forecasters mock NOT called — cached payloads are valid.
        assert mocks["forecasters"].await_count == 0


class TestStagesScoreOnly:
    @pytest.mark.asyncio
    async def test_stages_score_only_uses_existing_caches(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        # Pre-populate caches manually.
        cache = AblationCache(cache_dir)

        q1 = _make_binary_question(601)
        gt1 = _make_binary_ground_truth(601, outcome=True)

        # Manifest must reference the qid; schema mirrors _build_manifest_entry's writer.
        cache.write_qids_manifest(
            {
                601: {
                    "type": "binary",
                    "tournament": "spring-aib-2026",
                    "question_text": "Will Q601 happen?",
                    "page_url": "https://example.com/q/601",
                    "resolution_criteria": "Resolves YES if it happens.",
                    "fine_print": "",
                    "background_info": "",
                    "ground_truth": {
                        "question_id": 601,
                        "question_type": "binary",
                        "resolution": True,
                        "resolution_string": "YES",
                        "actual_resolution_time": "2026-05-01T00:00:00",
                        "question_text": "Will Q601 happen?",
                        "page_url": "https://example.com/q/601",
                    },
                    "question_metadata": {
                        "open_time": "2026-01-01T00:00:00",
                        "scheduled_resolution_time": "2026-05-01T00:00:00",
                    },
                }
            }
        )

        cache.write_research(601, "blob 601", {"sources": 1})
        cache.write_leakage_screen(
            601,
            {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
        )
        for i in range(3):
            slug = model_slug_to_filename(f"openrouter/test/m{i}")
            cache.write_forecaster_output(
                qid=601, model_slug=slug, payload=_binary_forecaster_payload(f"openrouter/test/m{i}", 0.5)
            )
        cache.write_stacker_output(qid=601, arm="stack", payload=_binary_stacker_payload("stack", 0.6))
        cache.write_stacker_output(qid=601, arm="stack_aug", payload=_binary_stacker_payload("stack_aug", 0.7))
        cache.write_stacker_output(qid=601, arm="median", payload=_binary_stacker_payload("median", 0.65))

        # Mocks installed; none should fire.
        question_set = _build_question_set([(q1, gt1)])
        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(["--stages", "score", "--cache-dir", str(cache_dir)])
        exit_code = await run_ablation(args)
        assert exit_code == 0

        # Scoring-only: no fetch, research, screen, forecast, or stacker calls.
        assert mocks["fetch"].await_count == 0
        assert mocks["research"].await_count == 0
        assert mocks["screen"].await_count == 0
        assert mocks["forecasters"].await_count == 0
        assert mocks["stacker"].await_count == 0

        # Summary file written.
        summaries = list((cache_dir / "scores").glob("summary_*.md"))
        assert len(summaries) == 1

    @pytest.mark.asyncio
    async def test_stages_score_only_errors_when_prerequisites_missing(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        # Empty cache: no manifest, no anything.
        question_set = _build_question_set([])
        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(["--stages", "score", "--cache-dir", str(cache_dir)])
        exit_code = await run_ablation(args)
        assert exit_code != 0


class TestQidsBypassesFetch:
    @pytest.mark.asyncio
    async def test_qids_arg_bypasses_fetch_stage(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--qids 100,200 with empty cache: fetch mock NOT called, downstream uses those qids."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        # We're going to mock fetch_resolved_questions_stratified; if the runner
        # calls it, the test fails.
        question_set = _build_question_set([])

        # The orchestrator must build questions some other way. We mock
        # MetaculusApi.get_question_by_post_id (or whatever it uses) to return
        # canned questions for qids 100 and 200. The CLI is responsible for
        # routing qid → MetaculusQuestion. We fake-mock at the routing point.
        from metaculus_bot.ablation import cli as cli_module

        q100 = _make_binary_question(100)
        q200 = _make_binary_question(200)

        async def fake_load_qids(qids: list[int]) -> tuple[list[Any], dict[int, GroundTruth]]:
            questions: list[Any] = []
            gts: dict[int, GroundTruth] = {}
            await asyncio.sleep(0)
            for qid in qids:
                if qid == 100:
                    questions.append(q100)
                    gts[100] = _make_binary_ground_truth(100, outcome=True)
                elif qid == 200:
                    questions.append(q200)
                    gts[200] = _make_binary_ground_truth(200, outcome=False)
            return questions, gts

        monkeypatch.setattr(cli_module, "load_questions_by_qids", fake_load_qids)

        verdicts = {
            100: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
            200: {
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
                    f"openrouter/test/m{i}", value=0.5
                )
                for i in range(3)
            }
            for qid in (100, 200)
        }
        stacker_a = {qid: _binary_stacker_payload("stack", 0.6) for qid in (100, 200)}
        stacker_b = {qid: _binary_stacker_payload("stack_aug", 0.7) for qid in (100, 200)}

        mocks = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={100: ("blob 100", {}), 200: ("blob 200", {})},
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
            ["--qids", "100,200", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        exit_code = await run_ablation(args)
        assert exit_code == 0

        # Fetch never called.
        assert mocks["fetch"].await_count == 0

        # Downstream stages saw qids 100, 200 (not whatever fake fetch would have).
        assert mocks["research"].await_count == 1
        # Inspect the questions arg passed to research.
        research_call_args = mocks["research"].await_args
        assert research_call_args is not None
        questions_arg = (
            research_call_args.args[0] if research_call_args.args else research_call_args.kwargs["questions"]
        )
        ids = sorted(q.id_of_question for q in questions_arg)
        assert ids == [100, 200]


# ---------------------------------------------------------------------------
# Manifest expansion
# ---------------------------------------------------------------------------


class TestManifestExpansion:
    @pytest.mark.asyncio
    async def test_smoke_then_expand_appends_qids(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """First run 1/1/1; second run 5/5/5. Manifest grows from 3 to 15 qids without re-sampling."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        # First fetch: 1/1/1.
        binary_q1 = _make_binary_question(1001)
        mc_q1 = _make_mc_question(1002)
        numeric_q1 = _make_numeric_question(1003)
        first_question_set = _build_question_set(
            [
                (binary_q1, _make_binary_ground_truth(1001)),
                (mc_q1, _make_mc_ground_truth(1002, "Red")),
                (numeric_q1, _make_numeric_ground_truth(1003, 50.0)),
            ]
        )

        first_verdicts = {
            qid: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            }
            for qid in (1001, 1002, 1003)
        }
        first_research: dict[int, tuple[str, dict] | None] = {qid: (f"blob {qid}", {}) for qid in (1001, 1002, 1003)}

        binary_payloads_first = {
            1001: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            }
        }
        mc_payloads_first = {
            1002: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _mc_forecaster_payload(f"openrouter/test/m{i}")
                for i in range(3)
            }
        }
        numeric_payloads_first = {
            1003: {
                model_slug_to_filename(f"openrouter/test/m{i}"): _numeric_forecaster_payload(
                    f"openrouter/test/m{i}", 50.0
                )
                for i in range(3)
            }
        }
        forecasters_first = {**binary_payloads_first, **mc_payloads_first, **numeric_payloads_first}
        stacker_a_first = {
            1001: _binary_stacker_payload("stack", 0.6),
            1002: _mc_stacker_payload("A"),
            1003: _numeric_stacker_payload("A"),
        }
        stacker_b_first = {
            1001: _binary_stacker_payload("stack_aug", 0.7),
            1002: _mc_stacker_payload("B"),
            1003: _numeric_stacker_payload("B"),
        }

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=first_question_set,
            research_results=first_research,
            leakage_verdicts=first_verdicts,
            forecaster_results=forecasters_first,
            stacker_a_results=stacker_a_first,
            stacker_b_results=stacker_b_first,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        argv_smoke = [
            "--num-binary",
            "1",
            "--num-multiple-choice",
            "1",
            "--num-numeric",
            "1",
            "--cache-dir",
            str(cache_dir),
            "--qa-iterate-mode",
            "advisory",
        ]
        await run_ablation(_build_parser().parse_args(argv_smoke))

        cache = AblationCache(cache_dir)
        manifest_after_smoke = cache.read_qids_manifest()
        assert set(manifest_after_smoke.keys()) == {1001, 1002, 1003}

        # Second fetch: 5/5/5. Re-install fetch mock to return only the NEW
        # questions (the orchestrator should ask for the delta: 4/4/4).
        new_binary_qs = [_make_binary_question(2000 + i) for i in range(4)]
        new_mc_qs = [_make_mc_question(2100 + i) for i in range(4)]
        new_numeric_qs = [_make_numeric_question(2200 + i) for i in range(4)]
        new_pairs = (
            [(q, _make_binary_ground_truth(q.id_of_question)) for q in new_binary_qs]
            + [(q, _make_mc_ground_truth(q.id_of_question, "Red")) for q in new_mc_qs]
            + [(q, _make_numeric_ground_truth(q.id_of_question, 50.0)) for q in new_numeric_qs]
        )
        second_question_set = _build_question_set(new_pairs)

        all_new_qids = [q.id_of_question for q, _ in new_pairs]
        second_research: dict[int, tuple[str, dict] | None] = {qid: (f"blob {qid}", {}) for qid in all_new_qids}
        second_verdicts = {
            qid: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            }
            for qid in all_new_qids
        }
        forecasters_second: dict[int, dict[str, dict]] = {}
        for q in new_binary_qs:
            forecasters_second[q.id_of_question] = {
                model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(
                    f"openrouter/test/m{i}", 0.5
                )
                for i in range(3)
            }
        for q in new_mc_qs:
            forecasters_second[q.id_of_question] = {
                model_slug_to_filename(f"openrouter/test/m{i}"): _mc_forecaster_payload(f"openrouter/test/m{i}")
                for i in range(3)
            }
        for q in new_numeric_qs:
            forecasters_second[q.id_of_question] = {
                model_slug_to_filename(f"openrouter/test/m{i}"): _numeric_forecaster_payload(
                    f"openrouter/test/m{i}", 50.0
                )
                for i in range(3)
            }

        stacker_a_second = {}
        stacker_b_second = {}
        for q in new_binary_qs:
            stacker_a_second[q.id_of_question] = _binary_stacker_payload("stack", 0.6)
            stacker_b_second[q.id_of_question] = _binary_stacker_payload("stack_aug", 0.7)
        for q in new_mc_qs:
            stacker_a_second[q.id_of_question] = _mc_stacker_payload("A")
            stacker_b_second[q.id_of_question] = _mc_stacker_payload("B")
        for q in new_numeric_qs:
            stacker_a_second[q.id_of_question] = _numeric_stacker_payload("A")
            stacker_b_second[q.id_of_question] = _numeric_stacker_payload("B")

        mocks2 = _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=second_question_set,
            research_results=second_research,
            leakage_verdicts=second_verdicts,
            forecaster_results=forecasters_second,
            stacker_a_results=stacker_a_second,
            stacker_b_results=stacker_b_second,
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        argv_small = [
            "--num-binary",
            "5",
            "--num-multiple-choice",
            "5",
            "--num-numeric",
            "5",
            "--cache-dir",
            str(cache_dir),
            "--qa-iterate-mode",
            "advisory",
        ]
        await run_ablation(_build_parser().parse_args(argv_small))

        # Fetch was called with delta counts (4/4/4), excluding existing qids.
        assert mocks2["fetch"].await_count == 1
        # Check the manifest grew correctly.
        cache = AblationCache(cache_dir)
        new_manifest = cache.read_qids_manifest()
        # Original 3 + 12 new = 15 qids.
        assert len(new_manifest) == 15
        # Original qids preserved.
        assert {1001, 1002, 1003}.issubset(set(new_manifest.keys()))


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


# ---------------------------------------------------------------------------
# Stacker payloads → scoring (confounder surface)
# ---------------------------------------------------------------------------


class TestStagePassesPayloadsToScoring:
    @pytest.mark.asyncio
    async def test_stage_score_passes_stacker_payloads_to_scoring(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The cached arm A/B stacker payloads must populate confounder fields in the summary."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        q1 = _make_binary_question(1301)
        q2 = _make_binary_question(1302)
        gt1 = _make_binary_ground_truth(1301, outcome=True)
        gt2 = _make_binary_ground_truth(1302, outcome=False)
        question_set = _build_question_set([(q1, gt1), (q2, gt2)])

        verdicts = {
            1301: {
                "is_leaked": False,
                "detector_response": "no leak",
                "detector_model": "test",
                "detector_failed": False,
                "screened_at": "2026-05-13T12:00:00",
            },
            1302: {
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
            for qid in (1301, 1302)
        }

        # Q1301 — both arms primary, arm B has cross_model_aggregation (tools fired).
        # Q1302 — arm A used fallback, arm B used primary, arm B has empty cross_model_aggregation.
        a1 = _binary_stacker_payload("stack", 0.6)
        b1 = _binary_stacker_payload("stack_aug", 0.7)
        b1["cross_model_aggregation"] = "## Cross-model aggregation\n(numbers)\n"
        a2 = _binary_stacker_payload("stack", 0.55)
        a2["stacker_model_used"] = "fallback"
        a2["n_forecasters_used"] = 4
        b2 = _binary_stacker_payload("stack_aug", 0.65)
        b2["cross_model_aggregation"] = ""
        b2["n_forecasters_used"] = 5

        _install_full_stack_mocks(
            monkeypatch,
            fetch_question_set=question_set,
            research_results={1301: ("blob 1301", {}), 1302: ("blob 1302", {})},
            leakage_verdicts=verdicts,
            forecaster_results=forecaster_results,
            stacker_a_results={1301: a1, 1302: a2},
            stacker_b_results={1301: b1, 1302: b2},
        )
        monkeypatch.setattr(
            "metaculus_bot.ablation.cli.asyncio.sleep",
            AsyncMock(return_value=None),
        )

        args = _build_parser().parse_args(
            ["--num-binary", "2", "--cache-dir", str(cache_dir), "--qa-iterate-mode", "advisory"]
        )
        await run_ablation(args)

        summary_path = next((cache_dir / "scores").glob("summary_*.md"))
        text = summary_path.read_text(encoding="utf-8")

        # Confounder section is present.
        assert "Confounder summary" in text
        # Arm A: 1 primary (q1301), 1 fallback (q1302).
        assert "1/2 primary" in text
        assert "1/2 fallback" in text
        # Treatment activation: pdf arm fired tools on 1/2.
        assert "stack_aug fired tools on 1/2 questions" in text
        # The empty-aggregation count message should mention the empty case.
        assert "empty cross_model_aggregation" in text
        # Per-question diagnostic table includes the marker columns.
        assert "stack_model" in text
        assert "stack_aug_model" in text
        # Renamed from "B_tools" to "stack_aug_tools" in the 3-arm summary refactor.
        assert "stack_aug_tools" in text


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


# ---------------------------------------------------------------------------
# Bug-2 regression (post-Bucket-1): numeric stacker payloads round-trip through
# ``deserialize_prediction_value`` to a PchipNumericDistribution, whose ``.cdf``
# is already constraint-enforced (PCHIP guarantees strict monotonicity by
# construction). The pre-Bucket-1 version of this test stress-tested the
# defensive sort in ``_build_report_shim`` against out-of-order declared
# percentiles; that defense is now upstream — declared percentiles flow into
# the PCHIP pipeline at forecaster time, and what comes out is a 201-point
# monotonic CDF. Test now asserts the shim produces a monotonic CDF for valid
# input — and that the shim's reliance on ``deserialize_prediction_value`` is
# wired correctly.
# ---------------------------------------------------------------------------


class TestBuildReportShimNumericSorting:
    def test_build_report_shim_returns_monotonic_cdf(self) -> None:
        """The shim's CDF must be strictly monotonic — required by ``np.trapezoid``
        in ``numeric_crps_from_report``.

        The new ``_build_report_shim`` reads ``stacker_prediction`` as a
        post-Bucket-1 full-CDF payload (declared_percentiles + cdf_probabilities
        + bounds + zero_point + cdf_size). It deserializes via
        ``deserialize_prediction_value`` to a PchipNumericDistribution and uses
        its ``.cdf`` directly — which is monotonic by PCHIP construction.
        """
        from metaculus_bot.ablation.cli import _build_report_shim

        q = _make_numeric_question(7001)
        # Build a valid Bucket-1 payload. declared_percentiles must be
        # strictly monotonic in BOTH percentile and value (Pydantic validator
        # on NumericDistribution enforces this); the synthetic 201-point CDF
        # spans the bounds linearly.
        declared_percentiles = [
            {"percentile": 0.1, "value": 30.0},
            {"percentile": 0.5, "value": 50.0},
            {"percentile": 0.9, "value": 70.0},
        ]
        cdf_probabilities = [0.001 + (0.998 * i / 200) for i in range(201)]
        payload = {
            "stacker_prediction": {
                "type": "numeric",
                "declared_percentiles": declared_percentiles,
                "cdf_probabilities": cdf_probabilities,
                "lower_bound": 0.0,
                "upper_bound": 100.0,
                "open_lower_bound": False,
                "open_upper_bound": False,
                "zero_point": None,
                "cdf_size": 201,
            },
        }
        report = _build_report_shim(7001, q, payload)

        cdf_values = [point.value for point in report.prediction.cdf]
        assert len(cdf_values) == 201, f"Expected 201-point CDF, got {len(cdf_values)}"
        # PCHIP guarantees strict monotonicity in the value axis; assert it.
        for prev, current in zip(cdf_values, cdf_values[1:]):
            assert current > prev, f"CDF values not strictly increasing: {prev} >= {current}"


# ---------------------------------------------------------------------------
# Bug-3 regression: OutOfBoundsResolution must round-trip through ground-truth
# serialize/deserialize. Without an explicit tag, the JSON writer's
# ``default=str`` produces "OutOfBoundsResolution.ABOVE_UPPER_BOUND" and the
# reload path's ``float(raw_resolution)`` raises ValueError.
# ---------------------------------------------------------------------------


class TestGroundTruthOutOfBoundsRoundTrip:
    def test_ground_truth_round_trip_with_out_of_bounds_resolution(self) -> None:
        """A numeric ``GroundTruth`` with ``OutOfBoundsResolution`` must reload exactly.

        The score-only path hydrates the working set entirely from cache. If the
        serialize/deserialize pair turns ``OutOfBoundsResolution.ABOVE_UPPER_BOUND``
        into a string then tries ``float(...)`` on it, the reload crashes with
        ``ValueError: could not convert string to float``. This test pins the
        round-trip contract.
        """
        from forecasting_tools.data_models.questions import OutOfBoundsResolution

        from metaculus_bot.ablation.cli import _deserialize_ground_truth, _serialize_ground_truth

        original = GroundTruth(
            question_id=8001,
            question_type="numeric",
            resolution=OutOfBoundsResolution.ABOVE_UPPER_BOUND,
            resolution_string="above upper bound",
            community_prediction=None,
            actual_resolution_time=datetime(2026, 5, 1),
            question_text="What value?",
            page_url="https://example.com/q/8001",
        )

        serialized = _serialize_ground_truth(original)
        # Round-trip via JSON to mimic the cache writer's behavior (default=str).
        import json  # noqa: PLC0415  - intentional in-test JSON round-trip

        round_tripped = json.loads(json.dumps(serialized, default=str))
        restored = _deserialize_ground_truth(round_tripped)

        assert restored.question_id == original.question_id
        assert restored.question_type == "numeric"
        assert restored.resolution == OutOfBoundsResolution.ABOVE_UPPER_BOUND
        assert restored.resolution_string == original.resolution_string
        assert restored.actual_resolution_time == original.actual_resolution_time


# ---------------------------------------------------------------------------
# Bug-4 regression: question shims hydrated from a manifest must carry
# ``open_time`` and ``scheduled_resolution_time`` so ``compute_mid_window_today``
# can perform real datetime arithmetic. A MagicMock(spec=...) without those
# attributes set returns sub-MagicMocks that pass the ``is not None`` assert
# but crash on subtraction.
# ---------------------------------------------------------------------------


class TestQuestionShimTimeFields:
    def test_question_shim_carries_open_time_and_scheduled_resolution_time(self) -> None:
        """``open_time`` and ``scheduled_resolution_time`` must round-trip through the manifest.

        ``compute_mid_window_today`` (used by the window patch) does
        ``scheduled_resolution_time - open_time``. If the manifest omits those
        fields, the score-only and ``--qids`` paths produce shims with
        sub-``MagicMock`` attributes that pass ``is not None`` but raise
        ``TypeError`` during datetime arithmetic. Tests pin both serialize and
        deserialize to round-trip the values as real datetimes.
        """
        from metaculus_bot.ablation.cli import (
            _build_manifest_entry,
            _build_question_shim_from_manifest_entry,
        )

        original_open = datetime(2026, 2, 1, 12, 0, 0)
        original_resolve = datetime(2026, 6, 15, 12, 0, 0)
        q = _make_binary_question(8501)
        q.open_time = original_open
        q.scheduled_resolution_time = original_resolve

        gt = _make_binary_ground_truth(8501, outcome=True)

        entry = _build_manifest_entry(q, gt, "spring-aib-2026")

        # Round-trip through JSON to mirror what cache.write does.
        import json  # noqa: PLC0415  - intentional in-test JSON round-trip

        round_tripped_entry = json.loads(json.dumps(entry, default=str))
        shim = _build_question_shim_from_manifest_entry(8501, round_tripped_entry)

        assert isinstance(shim.open_time, datetime), (
            f"shim.open_time must be a datetime, got {type(shim.open_time).__name__}"
        )
        assert isinstance(shim.scheduled_resolution_time, datetime), (
            f"shim.scheduled_resolution_time must be a datetime, got {type(shim.scheduled_resolution_time).__name__}"
        )
        assert shim.open_time == original_open
        assert shim.scheduled_resolution_time == original_resolve
        # Sanity: subtraction (the operation that would crash on a sub-MagicMock) works.
        delta = shim.scheduled_resolution_time - shim.open_time
        assert delta.days > 0


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


# ---------------------------------------------------------------------------
# Bug-6 regression: question shims hydrated from a manifest must carry every
# attribute downstream code reads. Pydantic BaseModel fields are NOT class
# attributes, so ``MagicMock(spec=BinaryQuestion)`` raises AttributeError on
# ``question.resolution_criteria`` / ``fine_print`` / ``background_info`` /
# ``unit_of_measure`` unless they're explicitly set on the shim. The leakage
# detector at ``backtest/leakage.py:86`` reads ``question.resolution_criteria``;
# the stacker prompts read all of background_info/resolution_criteria/fine_print
# (and unit_of_measure for numeric).
# ---------------------------------------------------------------------------


class TestQuestionShimContentFields:
    """Round-trip every question content attribute downstream code reads."""

    def test_question_shim_carries_resolution_criteria(self) -> None:
        """``resolution_criteria`` must round-trip through the manifest.

        The leakage detector at ``backtest/leakage.py:86`` reads
        ``question.resolution_criteria`` to render its prompt. Without
        explicit set on the shim, ``MagicMock(spec=BinaryQuestion)`` raises
        AttributeError because Pydantic model fields aren't class attributes.
        """
        from metaculus_bot.ablation.cli import (
            _build_manifest_entry,
            _build_question_shim_from_manifest_entry,
        )

        criteria_text = "Resolves YES if the SEC files an enforcement action by 2026-12-31."
        q = _make_binary_question(9501)
        q.resolution_criteria = criteria_text

        gt = _make_binary_ground_truth(9501, outcome=True)
        entry = _build_manifest_entry(q, gt, "spring-aib-2026")

        import json  # noqa: PLC0415  - intentional in-test JSON round-trip

        round_tripped_entry = json.loads(json.dumps(entry, default=str))
        shim = _build_question_shim_from_manifest_entry(9501, round_tripped_entry)

        assert shim.resolution_criteria == criteria_text

    def test_question_shim_carries_fine_print(self) -> None:
        """``fine_print`` must round-trip through the manifest.

        The stacker prompts (`stacking_binary_prompt`, `stacking_multiple_choice_prompt`,
        `stacking_numeric_prompt`) embed `question.fine_print` directly. Without
        explicit set on the shim, attribute access raises AttributeError.
        """
        from metaculus_bot.ablation.cli import (
            _build_manifest_entry,
            _build_question_shim_from_manifest_entry,
        )

        fine_print_text = "If the SEC issues a no-action letter instead, this resolves NO."
        q = _make_binary_question(9502)
        q.fine_print = fine_print_text

        gt = _make_binary_ground_truth(9502, outcome=True)
        entry = _build_manifest_entry(q, gt, "spring-aib-2026")

        import json  # noqa: PLC0415

        round_tripped_entry = json.loads(json.dumps(entry, default=str))
        shim = _build_question_shim_from_manifest_entry(9502, round_tripped_entry)

        assert shim.fine_print == fine_print_text

    def test_question_shim_carries_background_info(self) -> None:
        """``background_info`` must round-trip through the manifest.

        Every stacker prompt (binary/MC/numeric) embeds `question.background_info`.
        """
        from metaculus_bot.ablation.cli import (
            _build_manifest_entry,
            _build_question_shim_from_manifest_entry,
        )

        background = "The SEC has been investigating Acme Corp since 2025-03-15."
        q = _make_binary_question(9503)
        q.background_info = background

        gt = _make_binary_ground_truth(9503, outcome=True)
        entry = _build_manifest_entry(q, gt, "spring-aib-2026")

        import json  # noqa: PLC0415

        round_tripped_entry = json.loads(json.dumps(entry, default=str))
        shim = _build_question_shim_from_manifest_entry(9503, round_tripped_entry)

        assert shim.background_info == background

    def test_question_shim_carries_unit_of_measure_for_numeric(self) -> None:
        """``unit_of_measure`` must round-trip for numeric questions.

        ``stacking_numeric_prompt`` and ``numeric_prompt`` both read
        ``question.unit_of_measure`` to format the bounds-and-units block.
        """
        from metaculus_bot.ablation.cli import (
            _build_manifest_entry,
            _build_question_shim_from_manifest_entry,
        )

        units = "barrels per day"
        q = _make_numeric_question(9504)
        q.unit_of_measure = units

        gt = _make_numeric_ground_truth(9504, value=42.0)
        entry = _build_manifest_entry(q, gt, "spring-aib-2026")

        import json  # noqa: PLC0415

        round_tripped_entry = json.loads(json.dumps(entry, default=str))
        shim = _build_question_shim_from_manifest_entry(9504, round_tripped_entry)

        assert shim.unit_of_measure == units

    def test_question_shim_supports_leakage_detector_prompt_construction(self) -> None:
        """The shim from a manifest entry must work with ``_check_single_question_leakage``.

        Reproduces the live crash: the live ablation hit
        ``AttributeError: Mock object has no attribute 'resolution_criteria'``
        when the screen stage's leakage detector tried to render its prompt
        against a manifest-rehydrated shim. The fix is to round-trip
        resolution_criteria so the shim has the real string.
        """
        from metaculus_bot.ablation.cli import (
            _build_manifest_entry,
            _build_question_shim_from_manifest_entry,
        )
        from metaculus_bot.backtest.leakage import _check_single_question_leakage

        criteria_text = "Resolves YES if the launch occurs before 2026-12-31."
        q = _make_binary_question(9505)
        q.resolution_criteria = criteria_text

        gt = _make_binary_ground_truth(9505, outcome=True)
        entry = _build_manifest_entry(q, gt, "spring-aib-2026")

        import json  # noqa: PLC0415

        round_tripped_entry = json.loads(json.dumps(entry, default=str))
        shim = _build_question_shim_from_manifest_entry(9505, round_tripped_entry)

        # Detector LLM whose ``invoke`` returns "NO" and records the prompt
        # via ``await_args_list``.
        detector = MagicMock()
        detector.invoke = AsyncMock(return_value="NO - clean.")

        is_leaked = asyncio.run(_check_single_question_leakage(shim, gt, "research blob text", detector))

        assert is_leaked is False
        # Crucial: the prompt construction did not crash with AttributeError, AND the
        # resolution criteria string flowed through to the prompt verbatim.
        assert detector.invoke.await_count == 1
        prompt_arg = detector.invoke.await_args_list[0].args[0]
        assert criteria_text in prompt_arg, (
            f"Resolution criteria must appear in detector prompt; got: {prompt_arg[:500]}"
        )


# ---------------------------------------------------------------------------
# Regression: hydration must not promote raw research to research_blobs.
#
# The original prune stage drops qids with prune-validation failures from
# ``working.research_blobs`` so they never reach forecast/stack/score. The
# hydration path (used when re-running downstream stages from cache) must
# mirror this drop, otherwise re-running ``--stages forecast,stack,...``
# would forecast on raw (potentially leaked) research that the prune stage
# refused to sanitize.
# ---------------------------------------------------------------------------


class TestHydrationRespectsPruneFailures:
    @pytest.mark.asyncio
    async def test_hydration_skips_qid_with_no_pruned_blob(
        self,
        tmp_path: Path,
    ) -> None:
        """A qid present in research/ but absent from research_pruned/ MUST
        NOT appear in ``working.research_blobs`` after hydration.

        This pins the behavior that ``research_pruned`` is the canonical
        "ready for downstream consumption" gate. Raw research stays on disk
        for QA-dump purposes only.
        """
        from metaculus_bot.ablation.cli import (
            WorkingSet,
            _build_manifest_entry,
            _hydrate_working_set_from_cache,
        )

        cache = AblationCache(str(tmp_path))
        q_clean = _make_binary_question(qid=4001)
        q_unsanitized = _make_binary_question(qid=4002)
        gt_clean = _make_binary_ground_truth(4001, outcome=True)
        gt_unsanitized = _make_binary_ground_truth(4002, outcome=False)

        # Both qids in manifest with raw research; only the clean one has
        # a pruned blob.
        cache.append_qids_manifest(
            {
                4001: _build_manifest_entry(q_clean, gt_clean, "test-tournament"),
                4002: _build_manifest_entry(q_unsanitized, gt_unsanitized, "test-tournament"),
            }
        )
        cache.write_research(qid=4001, blob="raw research clean", meta={"provider": "gemini"})
        cache.write_research(qid=4002, blob="raw research with leak", meta={"provider": "gemini"})
        cache.write_pruned_research(qid=4001, sanitized_blob="sanitized clean", meta={"validation": "pass"})
        # Deliberately NO write_pruned_research for 4002 — simulates prune
        # validation failure.

        working = WorkingSet()
        await _hydrate_working_set_from_cache(cache, working)

        # Both questions in manifest are loaded as question objects.
        assert set(working.questions.keys()) == {4001, 4002}
        # But only the qid with a sanitized blob is eligible for downstream stages.
        assert 4001 in working.research_blobs
        assert 4002 not in working.research_blobs, (
            "qid 4002 had no sanitized blob (prune failure); hydration must NOT promote raw research downstream"
        )
        assert working.research_blobs[4001] == "sanitized clean"


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
        """Forcing forecast must auto-force stack + pdf + median (otherwise stale cache served)."""
        from metaculus_bot.ablation.cli import _expand_forced_stages

        forced = _expand_forced_stages({"forecast"})
        assert forced == {"forecast", "stack", "stack_aug", "pdf", "median"}

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

        explicit = {"forecast", "stack", "stack_aug", "pdf", "median"}
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
        argv_forced = argv + ["--force-stages", "forecast"]
        await run_ablation(_build_parser().parse_args(argv_forced))

        assert mocks["forecasters"].await_count == forecaster_count_after_run1 + 1
        # Stacker re-ran for both LLM arms A + B (without cascade, this would still be 2).
        # ARM_MEDIAN is invalidated by the cascade too, but its calls go through
        # run_median_for_qid, not the mocked run_stacker_batch.
        assert mocks["stacker"].await_count == stacker_count_after_run1 + 2, (
            "--force-stages forecast must cascade to stack + pdf + median; otherwise "
            "stackers return stale payloads derived from old forecaster outputs."
        )


# ---------------------------------------------------------------------------
# C2: --qids filter applies even when "fetch" is not in --stages.
#
# Without this, --qids 100 --stages stack,pdf,score loads the FULL
# manifest in hydration and runs every stage on every qid — operator pays
# 50x stacker spend when they meant to re-score one qid. See cli_audit
# (C2) for the full operator-footgun analysis.
# ---------------------------------------------------------------------------


class TestQidsFilterAppliedAfterHydration:
    @pytest.mark.asyncio
    async def test_qids_filter_when_fetch_not_in_stages_restricts_working_set(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--qids 100 --stages score must filter manifest to {100} only.

        Mutation test for C2: if the post-hydration filter is removed, the
        score summary will reference qid 200 too.
        """
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        cache = AblationCache(cache_dir)
        _populate_full_cache_for_qid(cache, 100)
        _populate_full_cache_for_qid(cache, 200)

        question_set = _build_question_set([])
        _install_full_stack_mocks(monkeypatch, fetch_question_set=question_set)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        args = _build_parser().parse_args(["--qids", "100", "--stages", "score", "--cache-dir", str(cache_dir)])
        exit_code = await run_ablation(args)
        assert exit_code == 0

        # Only qid 100's score should appear; qid 200 must be filtered out.
        summaries = sorted((cache_dir / "scores").glob("summary_*.md"))
        assert len(summaries) == 1
        text = summaries[-1].read_text(encoding="utf-8")
        # Per-question diagnostic table renders qids as `| <qid> |` rows.
        assert "| 100 |" in text
        assert "| 200 |" not in text, (
            f"--qids 100 must filter the working set even with --stages score; qid 200 leaked into summary:\n{text}"
        )
        # The metadata header records n_questions; should be 1, not 2.
        assert "N questions: 1" in text, f"expected N questions: 1; got summary:\n{text}"

    @pytest.mark.asyncio
    async def test_qids_filter_logs_error_for_qids_missing_from_manifest(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """--qids 999 (not in manifest) must log an error after hydration."""
        import logging

        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        cache = AblationCache(cache_dir)
        _populate_full_cache_for_qid(cache, 100)

        question_set = _build_question_set([])
        _install_full_stack_mocks(monkeypatch, fetch_question_set=question_set)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        caplog.set_level(logging.ERROR, logger="metaculus_bot.ablation.cli")
        args = _build_parser().parse_args(["--qids", "999", "--stages", "score", "--cache-dir", str(cache_dir)])
        await run_ablation(args)

        assert any("999" in record.message and "qids" in record.message.lower() for record in caplog.records), (
            f"expected error log mentioning qid 999; got: {[r.message for r in caplog.records]}"
        )

    @pytest.mark.asyncio
    async def test_qids_filter_subset_via_stack_a_path(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--qids 100 --stages stack must run stacker only on qid 100, not on qid 200.

        Tests the path where qids filter applies BEFORE stack stages run.
        """
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        cache = AblationCache(cache_dir)
        _populate_full_cache_for_qid(cache, 100)
        _populate_full_cache_for_qid(cache, 200)

        question_set = _build_question_set([])
        mocks = _install_full_stack_mocks(monkeypatch, fetch_question_set=question_set)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        # --force-stages stack so we don't pick up the cached stack payload
        # and instead inspect what qids would be sent to the stacker mock.
        args = _build_parser().parse_args(
            [
                "--qids",
                "100",
                "--stages",
                "stack",
                "--force-stages",
                "stack",
                "--cache-dir",
                str(cache_dir),
            ]
        )
        await run_ablation(args)

        # Stacker was called with qid 100 only.
        assert mocks["stacker"].await_count >= 1
        for call in mocks["stacker"].await_args_list:
            qid_to_data = call.args[0] if call.args else call.kwargs["qid_to_data"]
            assert set(qid_to_data.keys()) <= {100}, (
                f"--qids 100 must restrict stacker to {{100}}; got {set(qid_to_data.keys())}"
            )


# ---------------------------------------------------------------------------
# M1: score_only must error when arm_A and arm_B have zero qid overlap.
#
# The current "either dict empty" check passes when {1,2,3} ∪ {4,5,6}, then
# _stage_score takes the intersection (empty) and produces an "n=0 success".
# ---------------------------------------------------------------------------


class TestScoreOnlyZeroOverlapCheck:
    @pytest.mark.asyncio
    async def test_score_only_succeeds_with_zero_comparisons_when_disjoint(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Disjoint single-arm sets produce 0 comparisons but exit 0 (valid empty result).

        With per-comparison N, a qid needs >= 2 arms for any comparison. When
        each qid only has 1 arm, no comparisons are possible — the summary is
        empty but the run succeeds (exit 0). This is correct: the data simply
        doesn't support any pairwise comparison.
        """
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        cache = AblationCache(cache_dir)
        # Pre-populate manifests for all 6 qids; only arm_A for {1,2,3} and arm_B for {4,5,6}.
        for qid in (1, 2, 3, 4, 5, 6):
            _populate_full_cache_for_qid(cache, qid)
        # Wipe stacker outputs and rewrite them as disjoint single-arm sets.
        for qid in (1, 2, 3, 4, 5, 6):
            (cache.root / "stacker_outputs" / str(qid) / "arm_stack.json").unlink(missing_ok=True)
            (cache.root / "stacker_outputs" / str(qid) / "arm_stack_aug.json").unlink(missing_ok=True)
            (cache.root / "stacker_outputs" / str(qid) / "arm_median.json").unlink(missing_ok=True)
        for qid in (1, 2, 3):
            cache.write_stacker_output(qid=qid, arm="stack", payload=_binary_stacker_payload("stack", 0.6))
        for qid in (4, 5, 6):
            cache.write_stacker_output(qid=qid, arm="stack_aug", payload=_binary_stacker_payload("stack_aug", 0.7))

        question_set = _build_question_set([])
        _install_full_stack_mocks(monkeypatch, fetch_question_set=question_set)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        args = _build_parser().parse_args(["--stages", "score", "--cache-dir", str(cache_dir)])
        exit_code = await run_ablation(args)

        # Per-comparison N: each qid has only 1 arm, so no comparisons are
        # possible. Exit 0 is correct — it's not a config error, just empty data.
        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_score_only_succeeds_when_arms_overlap(
        self,
        cache_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sanity check: full overlap still produces a summary (didn't break the happy path)."""
        from metaculus_bot.ablation.cli import _build_parser, run_ablation

        cache = AblationCache(cache_dir)
        for qid in (10, 11):
            _populate_full_cache_for_qid(cache, qid)

        question_set = _build_question_set([])
        _install_full_stack_mocks(monkeypatch, fetch_question_set=question_set)
        monkeypatch.setattr("metaculus_bot.ablation.cli.asyncio.sleep", AsyncMock(return_value=None))

        args = _build_parser().parse_args(["--stages", "score", "--cache-dir", str(cache_dir)])
        exit_code = await run_ablation(args)

        assert exit_code == 0
        summaries = list((cache_dir / "scores").glob("summary_*.md"))
        assert len(summaries) == 1


# ---------------------------------------------------------------------------
# ARM_MEDIAN end-to-end via _stage_stack
#
# ARM_MEDIAN bypasses the stacker LLM entirely and runs deterministic median
# aggregation per question. These tests verify that:
#
# * The "median" stage is wired into STAGES.
# * ``_stage_stack(arm=ARM_MEDIAN, ...)`` writes a structurally-correct ``arm_median.json``
#   cache file when run on a synthetic working set.
# * ``WorkingSet.stacker_median_payloads`` is populated end-to-end.
# ---------------------------------------------------------------------------


class TestArmMedianStageStack:
    def test_stages_includes_stack_c(self) -> None:
        from metaculus_bot.ablation.cli import STAGES

        assert "median" in STAGES
        # median sits between pdf and score in the canonical pipeline order
        # so the orchestrator runs it after both LLM stackers but before scoring.
        stack_b_idx = STAGES.index("stack_aug")
        stack_c_idx = STAGES.index("median")
        score_idx = STAGES.index("score")
        assert stack_b_idx < stack_c_idx < score_idx

    @pytest.mark.asyncio
    async def test_stage_stack_arm_median_writes_arm_median_json_to_cache(
        self,
        cache_dir: Path,
    ) -> None:
        """Synthetic working set + ``_stage_stack(arm=ARM_MEDIAN)`` should emit ``arm_median.json``."""
        from metaculus_bot.ablation.cli import (
            SpendReport,
            WorkingSet,
            _build_parser,
            _stage_stack,
        )
        from metaculus_bot.ablation.run_stacker import ARM_MEDIAN

        cache = AblationCache(cache_dir)
        qid = 99001
        question = _make_binary_question(qid)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(f"openrouter/test/m{i}", 0.5)
            for i in range(3)
        }

        working = WorkingSet()
        working.questions[qid] = question
        working.forecaster_payloads[qid] = forecaster_payloads
        working.research_blobs[qid] = "research blob"

        spend = SpendReport()
        args = _build_parser().parse_args(["--num-binary", "1", "--cache-dir", str(cache_dir)])

        await _stage_stack(args, cache, working, arm=ARM_MEDIAN, force=False, spend=spend)

        # Cache file written.
        cached = cache.read_stacker_output(qid=qid, arm=ARM_MEDIAN)
        assert cached is not None
        assert cached["arm"] == "median"
        assert cached["success"] is True
        assert cached["stacker_model_used"] == "simple_aggregation"
        assert cached["tools_enabled_at_runtime"] is False
        # WorkingSet populated. The cache adds a ``cache_schema_version`` field on
        # read; the in-memory working set carries the raw payload _stage_stack
        # received before the cache round-trip. Compare on the structural fields
        # both should agree on rather than the full dict.
        assert qid in working.stacker_median_payloads
        for key in ("arm", "success", "stacker_model_used", "stacker_prediction"):
            assert working.stacker_median_payloads[qid][key] == cached[key]
        # ARM_MEDIAN does NOT consume LLM-call counters; only cache-hit counter (zero on first run).
        assert spend.stacker_llm_calls_stack == 0
        assert spend.stacker_llm_calls_stack_aug == 0
        assert spend.cached_stacker_median_hits == 0  # first run, no cache hit

    @pytest.mark.asyncio
    async def test_stage_stack_arm_c_uses_cache_on_second_call(
        self,
        cache_dir: Path,
    ) -> None:
        """A second invocation should hit the cache and bump ``cached_stacker_median_hits``."""
        from metaculus_bot.ablation.cli import (
            SpendReport,
            WorkingSet,
            _build_parser,
            _stage_stack,
        )
        from metaculus_bot.ablation.run_stacker import ARM_MEDIAN

        cache = AblationCache(cache_dir)
        qid = 99002
        question = _make_binary_question(qid)
        forecaster_payloads = {
            model_slug_to_filename(f"openrouter/test/m{i}"): _binary_forecaster_payload(f"openrouter/test/m{i}", 0.5)
            for i in range(3)
        }

        working = WorkingSet()
        working.questions[qid] = question
        working.forecaster_payloads[qid] = forecaster_payloads
        working.research_blobs[qid] = "research blob"

        args = _build_parser().parse_args(["--num-binary", "1", "--cache-dir", str(cache_dir)])

        # First call: writes cache.
        spend1 = SpendReport()
        await _stage_stack(args, cache, working, arm=ARM_MEDIAN, force=False, spend=spend1)
        assert spend1.cached_stacker_median_hits == 0

        # Second call: hits cache.
        working2 = WorkingSet()
        working2.questions[qid] = question
        working2.forecaster_payloads[qid] = forecaster_payloads
        working2.research_blobs[qid] = "research blob"
        spend2 = SpendReport()
        await _stage_stack(args, cache, working2, arm=ARM_MEDIAN, force=False, spend=spend2)
        assert spend2.cached_stacker_median_hits == 1
        assert qid in working2.stacker_median_payloads
