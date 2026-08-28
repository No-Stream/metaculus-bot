"""Shared fakes, payload factories and the full-stack mock installer for the ablation-CLI tests.

The CLI under test is ``metaculus_bot/ablation/cli.py``. It glues together every
Wave 1-3 building block: question fetch, Gemini-only research, leakage screen,
forecaster fan-out, two stacker arms, and paired-difference scoring. Eleven
``tests/ablation/test_ablation_cli_*.py`` modules exercise one aspect each; every
one of them builds questions, ground truths and stage payloads from this module, so
one canonical copy keeps the shapes from drifting apart.

``_install_full_stack_mocks`` is the reason no test here fires a live API call.
It monkeypatches every wave-1/2/3 entry point ON ``metaculus_bot.ablation.cli``
by string target, so the CLI's own module attributes are what get replaced:

* ``fetch_resolved_questions_stratified`` (wave 1)
* ``run_gemini_research_for_qids`` (wave 2)
* ``screen_batch`` (wave 3)
* ``run_forecasters_batch`` (wave 3)
* ``run_stacker_batch`` (wave 3)

Not named ``test_*`` on purpose: pytest must import this module without collecting it.
Holds no fixtures — the ``cache_dir`` these tests take lives in
``tests/ablation/conftest.py`` alongside the rest of the directory's fixtures.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.ablation.run_stacker import DEFAULT_STACKER_MODEL
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
    # The fetch stage runs a real unauthenticated API-identity preflight before
    # fetching; stub it so the ablation stack stays hermetic (its own behavior
    # is covered in test_api_preflight.py).
    monkeypatch.setattr("metaculus_bot.ablation.cli.verify_metaculus_api_identity", MagicMock())

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
        # Honor the per-stacker cache slug the real run_stacker_batch threads
        # through, so cache-hit idempotency on re-run matches production.
        stacker_slug = kwargs.get("stacker_slug")
        out: dict[int, dict] = {}
        for qid in qid_to_data:
            payload = source.get(qid)
            if payload is not None:
                cache.write_stacker_output(qid=qid, arm=arm, payload=payload, stacker_slug=stacker_slug)
                out[qid] = payload
        return out

    stacker_mock = AsyncMock(side_effect=_stacker_batch_returns)
    monkeypatch.setattr("metaculus_bot.ablation.cli.run_stacker_batch", stacker_mock)

    # Default qa_iterate stub: clean verifier verdict for every qid so the stage no-ops cleanly.
    # Tests can monkeypatch a different fake verifier/redactor to exercise iterate behavior.
    import json as _json
    import re as _re

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
    # stack/stack_aug are slugged by the active stacker (mirrors a real run); the
    # default `--lineup free` path reads them under the opus-4.5 DEFAULT_STACKER_MODEL
    # slug, so seed them under that slug. The deterministic median arm stays unslugged.
    free_stacker_slug = model_slug_to_filename(DEFAULT_STACKER_MODEL)
    cache.write_stacker_output(
        qid=qid, arm="stack", payload=_binary_stacker_payload("stack", 0.6), stacker_slug=free_stacker_slug
    )
    cache.write_stacker_output(
        qid=qid, arm="stack_aug", payload=_binary_stacker_payload("stack_aug", 0.7), stacker_slug=free_stacker_slug
    )
    cache.write_stacker_output(qid=qid, arm="median", payload=_binary_stacker_payload("median", 0.65))
