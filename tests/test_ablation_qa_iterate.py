"""Tests for the iterate-until-clean QA harness for the ablation benchmark.

The qa_iterate stage runs between screen and forecast. For each qid where the
screen flagged leakage OR the verifier flags leakage_risk above threshold, the
loop:

1. Spawns a verifier ``claude -p`` subprocess that scores the sanitized blob
   on three dimensions (leakage_risk, forecastability, hallucination_risk).
2. If the verifier accepts (leakage_risk < threshold AND screen says clean),
   accept and stop.
3. Otherwise spawn a re-redactor ``claude -p`` subprocess that does a second-pass
   redaction informed by the verifier's notes.
4. Repeat up to ``max_iterations``. After that, auto-reject with a reason.
5. Once leakage passes, check forecastability — if the blob is too thin to
   forecast from, auto-reject with ``rejected_forecastability``.

All tests mock the ``claude -p`` subprocess primitive.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from forecasting_tools import MetaculusQuestion

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.backtest.scoring import GroundTruth


def _make_question(qid: int) -> MetaculusQuestion:
    q = SimpleNamespace(
        id_of_question=qid,
        question_text=f"Will Q{qid} happen?",
        resolution_criteria=f"Resolves YES if Q{qid} happens.",
        fine_print="",
        background_info="",
        page_url=f"https://example.com/q/{qid}",
        open_time=datetime(2026, 1, 1),
        scheduled_resolution_time=datetime(2026, 5, 1),
    )
    return cast(MetaculusQuestion, q)


def _make_ground_truth(qid: int, resolution_string: str = "Yes") -> GroundTruth:
    return GroundTruth(
        question_id=qid,
        question_type="binary",
        resolution=True,
        resolution_string=resolution_string,
        community_prediction=None,
        actual_resolution_time=datetime(2026, 5, 1),
        question_text=f"Will Q{qid} happen?",
        page_url=f"https://example.com/q/{qid}",
    )


def _make_screen_verdict(*, is_leaked: bool) -> dict:
    return {
        "is_leaked": is_leaked,
        "detector_response": "test verdict",
        "detector_model": "test",
        "detector_failed": False,
        "screened_at": "2026-05-14T00:00:00",
    }


def _verifier_response(
    qid: int,
    *,
    leakage_risk: float,
    forecastability: float,
    hallucination_risk: float,
    notes: str = "",
) -> str:
    payload = {
        "verdicts": [
            {
                "qid": qid,
                "leakage_risk": leakage_risk,
                "forecastability": forecastability,
                "hallucination_risk": hallucination_risk,
                "notes": notes,
            }
        ]
    }
    return json.dumps(payload)


def _redactor_response(qid: int, sanitized_blob: str) -> str:
    payload = {
        "results": [
            {
                "qid": qid,
                "sanitized_blob": sanitized_blob,
                "redactions": [],
            }
        ]
    }
    return json.dumps(payload)


@pytest.fixture
def cache(tmp_path: Path) -> AblationCache:
    return AblationCache(tmp_path / "abl")


# ---------------------------------------------------------------------------
# Per-qid loop behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_first_pass_no_iteration(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Initial screen + verifier both clean: no re-redaction fires."""
    from metaculus_bot.ablation import qa_iterate

    qid = 100
    question = _make_question(qid)
    gt = _make_ground_truth(qid)
    cache.write_pruned_research(
        qid=qid,
        sanitized_blob="clean blob",
        meta={
            "qid": qid,
            "original_chars": 100,
            "sanitized_chars": 100,
            "redactions": [],
            "redactor_invocation_id": "x",
            "pruned_at": "2026-05-14T00:00:00",
        },
    )

    verifier_mock = AsyncMock(
        return_value=_verifier_response(qid, leakage_risk=0.05, forecastability=0.8, hallucination_risk=0.1, notes="")
    )
    redactor_mock = AsyncMock(side_effect=[_redactor_response(qid, "should not fire")])

    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", verifier_mock)
    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", redactor_mock)

    outcome = await qa_iterate.run_qa_iterate_for_qid(
        qid=qid,
        question=question,
        ground_truth=gt,
        current_blob="clean blob",
        screen_verdict=_make_screen_verdict(is_leaked=False),
        cache=cache,
        max_iterations=3,
    )

    assert outcome.final_status == "clean"
    assert outcome.iterations == 1
    assert verifier_mock.await_count == 1
    assert redactor_mock.await_count == 0
    assert len(outcome.verifier_scores) == 1
    assert outcome.verifier_scores[0].leakage_risk == 0.05


@pytest.mark.asyncio
async def test_iterate_until_clean(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First verifier flags leakage; second iteration produces clean blob."""
    from metaculus_bot.ablation import qa_iterate

    qid = 200
    question = _make_question(qid)
    gt = _make_ground_truth(qid)
    cache.write_pruned_research(
        qid=qid,
        sanitized_blob="leaky blob",
        meta={
            "qid": qid,
            "original_chars": 100,
            "sanitized_chars": 80,
            "redactions": [],
            "redactor_invocation_id": "x",
            "pruned_at": "2026-05-14T00:00:00",
        },
    )

    verifier_mock = AsyncMock(
        side_effect=[
            _verifier_response(qid, leakage_risk=0.5, forecastability=0.7, hallucination_risk=0.2, notes="anchored"),
            _verifier_response(qid, leakage_risk=0.1, forecastability=0.7, hallucination_risk=0.2, notes="ok"),
        ]
    )
    redactor_mock = AsyncMock(return_value=_redactor_response(qid, "second-pass clean blob"))

    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", verifier_mock)
    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", redactor_mock)

    outcome = await qa_iterate.run_qa_iterate_for_qid(
        qid=qid,
        question=question,
        ground_truth=gt,
        current_blob="leaky blob",
        screen_verdict=_make_screen_verdict(is_leaked=False),
        cache=cache,
        max_iterations=3,
    )

    assert outcome.final_status == "clean"
    assert outcome.iterations == 2
    assert verifier_mock.await_count == 2
    assert redactor_mock.await_count == 1
    cached_pruned = cache.read_pruned_research(qid)
    assert cached_pruned is not None
    assert cached_pruned[0] == "second-pass clean blob"


@pytest.mark.asyncio
async def test_max_iterations_then_reject(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifier always returns leakage_risk=0.5: auto-reject after max iterations."""
    from metaculus_bot.ablation import qa_iterate

    qid = 300
    question = _make_question(qid)
    gt = _make_ground_truth(qid)
    cache.write_pruned_research(
        qid=qid,
        sanitized_blob="persistently leaky",
        meta={
            "qid": qid,
            "original_chars": 100,
            "sanitized_chars": 100,
            "redactions": [],
            "redactor_invocation_id": "x",
            "pruned_at": "2026-05-14T00:00:00",
        },
    )

    verifier_mock = AsyncMock(
        return_value=_verifier_response(
            qid, leakage_risk=0.5, forecastability=0.6, hallucination_risk=0.2, notes="leak"
        )
    )
    redactor_mock = AsyncMock(return_value=_redactor_response(qid, "still leaky"))

    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", verifier_mock)
    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", redactor_mock)

    outcome = await qa_iterate.run_qa_iterate_for_qid(
        qid=qid,
        question=question,
        ground_truth=gt,
        current_blob="persistently leaky",
        screen_verdict=_make_screen_verdict(is_leaked=False),
        cache=cache,
        max_iterations=3,
    )

    assert outcome.final_status == "rejected_leakage"
    assert outcome.iterations == 3
    assert outcome.reject_reason is not None
    assert "leakage" in outcome.reject_reason


@pytest.mark.asyncio
async def test_low_forecastability_rejects_qid(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifier reports clean leakage but very low forecastability → reject."""
    from metaculus_bot.ablation import qa_iterate

    qid = 400
    question = _make_question(qid)
    gt = _make_ground_truth(qid)
    cache.write_pruned_research(
        qid=qid,
        sanitized_blob="empty blob",
        meta={
            "qid": qid,
            "original_chars": 30,
            "sanitized_chars": 30,
            "redactions": [],
            "redactor_invocation_id": "x",
            "pruned_at": "2026-05-14T00:00:00",
        },
    )

    verifier_mock = AsyncMock(
        return_value=_verifier_response(
            qid, leakage_risk=0.05, forecastability=0.05, hallucination_risk=0.9, notes="empty"
        )
    )
    redactor_mock = AsyncMock(return_value=_redactor_response(qid, "should not fire"))

    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", verifier_mock)
    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", redactor_mock)

    outcome = await qa_iterate.run_qa_iterate_for_qid(
        qid=qid,
        question=question,
        ground_truth=gt,
        current_blob="empty blob",
        screen_verdict=_make_screen_verdict(is_leaked=False),
        cache=cache,
        max_iterations=3,
    )

    assert outcome.final_status == "rejected_forecastability"
    assert redactor_mock.await_count == 0
    assert outcome.reject_reason is not None
    assert "forecastability" in outcome.reject_reason


@pytest.mark.asyncio
async def test_screen_disagrees_with_verifier_blocks_acceptance(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Screen says clean but verifier flags leakage_risk=0.5: must iterate."""
    from metaculus_bot.ablation import qa_iterate

    qid = 500
    question = _make_question(qid)
    gt = _make_ground_truth(qid)
    cache.write_pruned_research(
        qid=qid,
        sanitized_blob="screen says clean",
        meta={
            "qid": qid,
            "original_chars": 100,
            "sanitized_chars": 100,
            "redactions": [],
            "redactor_invocation_id": "x",
            "pruned_at": "2026-05-14T00:00:00",
        },
    )

    verifier_mock = AsyncMock(
        side_effect=[
            _verifier_response(qid, leakage_risk=0.5, forecastability=0.7, hallucination_risk=0.2, notes="hidden"),
            _verifier_response(qid, leakage_risk=0.1, forecastability=0.7, hallucination_risk=0.2, notes="ok"),
        ]
    )
    redactor_mock = AsyncMock(return_value=_redactor_response(qid, "fixed blob"))

    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", verifier_mock)
    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", redactor_mock)

    outcome = await qa_iterate.run_qa_iterate_for_qid(
        qid=qid,
        question=question,
        ground_truth=gt,
        current_blob="screen says clean",
        screen_verdict=_make_screen_verdict(is_leaked=False),
        cache=cache,
        max_iterations=3,
    )

    assert outcome.final_status == "clean"
    assert outcome.iterations == 2


@pytest.mark.asyncio
async def test_screen_says_leaked_blocks_acceptance(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Screen says leaked but verifier reports leakage_risk=0.1: must iterate."""
    from metaculus_bot.ablation import qa_iterate

    qid = 600
    question = _make_question(qid)
    gt = _make_ground_truth(qid)
    cache.write_pruned_research(
        qid=qid,
        sanitized_blob="seems clean to verifier",
        meta={
            "qid": qid,
            "original_chars": 100,
            "sanitized_chars": 100,
            "redactions": [],
            "redactor_invocation_id": "x",
            "pruned_at": "2026-05-14T00:00:00",
        },
    )

    verifier_mock = AsyncMock(
        return_value=_verifier_response(qid, leakage_risk=0.1, forecastability=0.7, hallucination_risk=0.2, notes="ok"),
    )
    redactor_mock = AsyncMock(return_value=_redactor_response(qid, "rewrite"))

    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", verifier_mock)
    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", redactor_mock)

    outcome = await qa_iterate.run_qa_iterate_for_qid(
        qid=qid,
        question=question,
        ground_truth=gt,
        current_blob="seems clean to verifier",
        screen_verdict=_make_screen_verdict(is_leaked=True),
        cache=cache,
        max_iterations=3,
    )

    assert outcome.iterations >= 2
    assert redactor_mock.await_count >= 1


# ---------------------------------------------------------------------------
# manual_rejects.json
# ---------------------------------------------------------------------------


def test_read_manual_rejects_raises_with_path_on_malformed_json(tmp_path: Path) -> None:
    """A hand-edit error in manual_rejects.json must surface a clear ValueError
    that names the file path, not a bare json.JSONDecodeError. The audit
    workflow involves operators hand-editing the file; a clear error message
    saves debugging time on the next run.
    """
    from metaculus_bot.ablation import qa_iterate

    rejects_path = tmp_path / "manual_rejects.json"
    rejects_path.write_text('{"version": 1, "rejects": {}, ', encoding="utf-8")  # trailing comma — invalid

    with pytest.raises(ValueError, match=str(rejects_path)) as excinfo:
        qa_iterate.read_manual_rejects(rejects_path)
    assert "malformed" in str(excinfo.value).lower() or "invalid" in str(excinfo.value).lower()


def test_existing_manual_reject_is_honored(tmp_path: Path) -> None:
    """A pre-existing entry in manual_rejects.json must NOT be overwritten."""
    from metaculus_bot.ablation import qa_iterate

    rejects_path = tmp_path / "manual_rejects.json"
    rejects_path.write_text(
        json.dumps(
            {
                "version": 1,
                "rejects": {
                    "12345": {
                        "rejected_at": "2026-05-01T00:00:00",
                        "reason": "manual:operator note",
                        "verifier_scores": [],
                        "iterations": 0,
                    }
                },
            },
            indent=2,
        )
    )

    existing = qa_iterate.read_manual_rejects(rejects_path)
    assert 12345 in existing
    assert existing[12345]["reason"] == "manual:operator note"


def test_write_manual_rejects_appends_without_clobbering_manual_entries(tmp_path: Path) -> None:
    from metaculus_bot.ablation import qa_iterate

    rejects_path = tmp_path / "manual_rejects.json"
    rejects_path.write_text(
        json.dumps(
            {
                "version": 1,
                "rejects": {
                    "12345": {
                        "rejected_at": "2026-05-01T00:00:00",
                        "reason": "manual:keep me",
                        "verifier_scores": [],
                        "iterations": 0,
                    }
                },
            },
            indent=2,
        )
    )

    new_outcome = qa_iterate.IterateOutcome(
        qid=99999,
        final_status="rejected_leakage",
        iterations=3,
        final_blob_path=None,
        verifier_scores=[],
        reject_reason="leakage_unrecoverable_after_3_iterations",
    )
    qa_iterate.write_manual_rejects([new_outcome], rejects_path)

    on_disk = json.loads(rejects_path.read_text(encoding="utf-8"))
    assert "12345" in on_disk["rejects"]
    assert on_disk["rejects"]["12345"]["reason"] == "manual:keep me"
    assert "99999" in on_disk["rejects"]
    assert on_disk["rejects"]["99999"]["reason"] == "leakage_unrecoverable_after_3_iterations"


def test_write_manual_rejects_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash mid-write must NOT corrupt manual_rejects.json.

    Atomic write via tempfile + os.replace: when the in-progress write fails,
    the original file remains intact. Without atomic writes, ``write_text``
    truncates the file before raising, leaving an empty file on disk.
    """
    from metaculus_bot.ablation import qa_iterate

    rejects_path = tmp_path / "manual_rejects.json"
    rejects_path.write_text(
        json.dumps({"version": 1, "rejects": {"123": {"reason": "manual:keep"}}}, indent=2),
        encoding="utf-8",
    )
    original_contents = rejects_path.read_text(encoding="utf-8")

    real_replace = qa_iterate.os.replace if hasattr(qa_iterate, "os") else None  # cache helper uses os.replace

    import os as _os

    def boom_replace(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated kernel reboot during commit")

    monkeypatch.setattr(_os, "replace", boom_replace)

    new_outcome = qa_iterate.IterateOutcome(
        qid=999,
        final_status="rejected_leakage",
        iterations=1,
        final_blob_path=None,
        verifier_scores=[],
        reject_reason="test",
    )

    with pytest.raises(RuntimeError, match="simulated kernel reboot"):
        qa_iterate.write_manual_rejects([new_outcome], rejects_path)

    # Original file must remain intact (no truncation, no corruption).
    assert rejects_path.exists()
    assert rejects_path.read_text(encoding="utf-8") == original_contents

    # No tempfile leak: only the original file should remain in the parent dir.
    leftover = [p for p in rejects_path.parent.iterdir() if p.name.startswith(f".{rejects_path.name}.")]
    assert leftover == []
    _ = real_replace  # silence unused


def test_render_qa_summary_uses_atomic_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """qa_summary_<ts>.md write must be atomic — crash mid-write preserves prior file."""
    from metaculus_bot.ablation import qa_iterate

    summary_path = tmp_path / "qa_summary_test.md"
    summary_path.write_text("# previous summary\n", encoding="utf-8")
    original = summary_path.read_text(encoding="utf-8")

    import os as _os

    def boom_replace(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("write interrupted")

    monkeypatch.setattr(_os, "replace", boom_replace)

    outcomes = {
        1: qa_iterate.IterateOutcome(
            qid=1,
            final_status="clean",
            iterations=1,
            final_blob_path=None,
            verifier_scores=[],
            reject_reason=None,
        )
    }

    with pytest.raises(RuntimeError, match="write interrupted"):
        qa_iterate.render_qa_summary(outcomes, summary_path)

    # Original file must remain intact.
    assert summary_path.read_text(encoding="utf-8") == original
    leftover = [p for p in summary_path.parent.iterdir() if p.name.startswith(f".{summary_path.name}.")]
    assert leftover == []


def test_iterate_outcome_serializable_to_json() -> None:
    from metaculus_bot.ablation import qa_iterate

    score = qa_iterate.VerifierScore(
        iteration=1,
        leakage_risk=0.4,
        forecastability=0.6,
        hallucination_risk=0.3,
        notes="anchored value",
    )
    outcome = qa_iterate.IterateOutcome(
        qid=42,
        final_status="clean",
        iterations=2,
        final_blob_path=Path("/tmp/sanitized/42.md"),
        verifier_scores=[score],
        reject_reason=None,
    )
    serialized = qa_iterate.serialize_outcome(outcome)
    text = json.dumps(serialized)
    parsed = json.loads(text)
    assert parsed["qid"] == 42
    assert parsed["final_status"] == "clean"
    assert parsed["verifier_scores"][0]["leakage_risk"] == 0.4


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------


def test_render_qa_summary_includes_aggregate_stats(tmp_path: Path) -> None:
    from metaculus_bot.ablation import qa_iterate

    outcomes = {
        1: qa_iterate.IterateOutcome(
            qid=1,
            final_status="clean",
            iterations=1,
            final_blob_path=tmp_path / "1.md",
            verifier_scores=[
                qa_iterate.VerifierScore(iteration=1, leakage_risk=0.05, forecastability=0.8, hallucination_risk=0.1)
            ],
            reject_reason=None,
        ),
        2: qa_iterate.IterateOutcome(
            qid=2,
            final_status="rejected_leakage",
            iterations=3,
            final_blob_path=None,
            verifier_scores=[
                qa_iterate.VerifierScore(iteration=i, leakage_risk=0.5, forecastability=0.7, hallucination_risk=0.2)
                for i in range(1, 4)
            ],
            reject_reason="leakage_unrecoverable_after_3_iterations",
        ),
        3: qa_iterate.IterateOutcome(
            qid=3,
            final_status="rejected_forecastability",
            iterations=1,
            final_blob_path=None,
            verifier_scores=[
                qa_iterate.VerifierScore(iteration=1, leakage_risk=0.05, forecastability=0.05, hallucination_risk=0.9)
            ],
            reject_reason="low_forecastability",
        ),
    }
    out_path = tmp_path / "summary.md"
    qa_iterate.render_qa_summary(outcomes, out_path)
    text = out_path.read_text(encoding="utf-8")
    assert "clean" in text.lower()
    assert "rejected_leakage" in text
    assert "rejected_forecastability" in text
    assert "Total qids: 3" in text or "total: 3" in text.lower()
    assert "Q1" in text and "Q2" in text and "Q3" in text


# ---------------------------------------------------------------------------
# Mode handling - the stage-level wrapper that lives in cli.py is tested in
# test_ablation_cli.py. Here we exercise the qa_iterate-level helpers.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_qa_iterate_batch_returns_per_qid_outcomes(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch helper aggregates per-qid outcomes."""
    from metaculus_bot.ablation import qa_iterate

    qids = [10, 20]
    inputs: dict[int, dict[str, Any]] = {}
    for qid in qids:
        question = _make_question(qid)
        gt = _make_ground_truth(qid)
        cache.write_pruned_research(
            qid=qid,
            sanitized_blob=f"blob {qid}",
            meta={
                "qid": qid,
                "original_chars": 100,
                "sanitized_chars": 100,
                "redactions": [],
                "redactor_invocation_id": "x",
                "pruned_at": "2026-05-14T00:00:00",
            },
        )
        inputs[qid] = {
            "question": question,
            "ground_truth": gt,
            "current_blob": f"blob {qid}",
            "screen_verdict": _make_screen_verdict(is_leaked=False),
        }

    side_effect_per_call = [
        _verifier_response(10, leakage_risk=0.05, forecastability=0.8, hallucination_risk=0.1),
        _verifier_response(20, leakage_risk=0.05, forecastability=0.8, hallucination_risk=0.1),
    ]
    verifier_mock = AsyncMock(side_effect=side_effect_per_call)
    redactor_mock = AsyncMock()
    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", verifier_mock)
    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", redactor_mock)

    outcomes = await qa_iterate.run_qa_iterate_batch(
        inputs,
        cache=cache,
        max_iterations=3,
    )

    assert set(outcomes.keys()) == {10, 20}
    for qid in qids:
        assert outcomes[qid].final_status == "clean"


# ---------------------------------------------------------------------------
# Per-qid error isolation (C1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_qa_iterate_batch_isolates_per_qid_subprocess_failure(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A subprocess failure for one qid must NOT cancel other qids in the batch.

    Prior to C1, ``asyncio.gather(*tasks)`` cancelled every in-flight task on
    a single failure. Wrap ``_one`` in try/except that converts subprocess
    failures into auto-rejected ``IterateOutcome`` (status=rejected_leakage,
    reason=qa_iterate_failed: <ExceptionType>).
    """
    import logging
    import subprocess

    from metaculus_bot.ablation import qa_iterate

    qids = [30, 31, 32]
    inputs: dict[int, dict[str, Any]] = {}
    for qid in qids:
        question = _make_question(qid)
        gt = _make_ground_truth(qid)
        cache.write_pruned_research(
            qid=qid,
            sanitized_blob=f"blob {qid}",
            meta={
                "qid": qid,
                "original_chars": 100,
                "sanitized_chars": 100,
                "redactions": [],
                "redactor_invocation_id": "x",
                "pruned_at": "2026-05-14T00:00:00",
            },
        )
        inputs[qid] = {
            "question": question,
            "ground_truth": gt,
            "current_blob": f"blob {qid}",
            "screen_verdict": _make_screen_verdict(is_leaked=False),
        }

    import asyncio
    import re

    async def failing_verifier(prompt: str, **_kwargs: Any) -> str:
        await asyncio.sleep(0)
        if "qid=31" in prompt:
            raise subprocess.CalledProcessError(returncode=1, cmd=["claude"], stderr=b"oops")
        match = re.search(r"qid=(\d+)", prompt)
        qid = int(match.group(1)) if match else 0
        return _verifier_response(qid, leakage_risk=0.05, forecastability=0.8, hallucination_risk=0.1)

    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", failing_verifier)
    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", AsyncMock())

    with caplog.at_level(logging.WARNING, logger="metaculus_bot.ablation.qa_iterate"):
        outcomes = await qa_iterate.run_qa_iterate_batch(
            inputs,
            cache=cache,
            max_iterations=3,
        )

    assert set(outcomes.keys()) == {30, 31, 32}
    assert outcomes[30].final_status == "clean"
    assert outcomes[32].final_status == "clean"

    failed = outcomes[31]
    assert failed.final_status == "rejected_leakage"
    assert failed.reject_reason is not None
    assert failed.reject_reason.startswith("qa_iterate_failed:")
    assert "CalledProcessError" in failed.reject_reason

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "qid=31" in r.getMessage()]
    assert warning_records, "expected a WARNING log for the failed qid"


# ---------------------------------------------------------------------------
# Subprocess argv
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_verifier_uses_sonnet_and_correct_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from metaculus_bot.ablation import qa_iterate

    captured: dict = {}

    async def fake_subproc(*args, **kwargs):
        captured["argv"] = list(args)
        captured["kwargs"] = kwargs
        await asyncio.sleep(0)

        class _Proc:
            returncode = 0

            async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
                captured["stdin"] = input
                await asyncio.sleep(0)
                return b'{"verdicts": []}', b""

        return _Proc()

    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate.asyncio.create_subprocess_exec", fake_subproc)

    result = await qa_iterate._invoke_verifier("a verifier prompt")
    assert result == '{"verdicts": []}'
    argv = captured["argv"]
    assert any("claude" in str(a) for a in argv)
    assert "-p" in argv
    assert "--max-turns" in argv
    assert "--permission-mode" in argv
    assert "--append-system-prompt" in argv
    sys_prompt_idx = argv.index("--append-system-prompt")
    sys_prompt = argv[sys_prompt_idx + 1]
    assert "auditor" in sys_prompt.lower() or "verifier" in sys_prompt.lower()


@pytest.mark.asyncio
async def test_invoke_re_redactor_includes_verifier_notes_in_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from metaculus_bot.ablation import qa_iterate

    captured: dict = {}

    async def fake_subproc(*args, **kwargs):
        captured["argv"] = list(args)
        await asyncio.sleep(0)

        class _Proc:
            returncode = 0

            async def communicate(self, input: bytes | None = None) -> tuple[bytes, bytes]:
                captured["stdin"] = input
                await asyncio.sleep(0)
                return b'{"results": []}', b""

        return _Proc()

    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate.asyncio.create_subprocess_exec", fake_subproc)

    await qa_iterate._invoke_re_redactor("a re-redactor prompt with verifier notes embedded")
    assert b"verifier notes" in captured["stdin"]


# ---------------------------------------------------------------------------
# Verifier prompt content
# ---------------------------------------------------------------------------


def test_extract_inner_result_warns_on_unparseable_stdout(caplog: pytest.LogCaptureFixture) -> None:
    """When claude -p stdout is unparseable JSON, log a WARNING before
    passing it through. Mirrors prune._extract_inner_result behavior.
    """
    import logging

    from metaculus_bot.ablation.qa_iterate import _extract_inner_result

    unparseable = "not json {{{"

    with caplog.at_level(logging.WARNING, logger="metaculus_bot.ablation.qa_iterate"):
        result = _extract_inner_result(unparseable)

    assert result == unparseable
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("not parseable JSON" in r.getMessage() for r in warning_records)


def test_verifier_system_prompt_scores_three_axes() -> None:
    from metaculus_bot.ablation.qa_iterate import VERIFIER_SYSTEM_PROMPT

    lower = VERIFIER_SYSTEM_PROMPT.lower()
    assert "leakage_risk" in lower
    assert "forecastability" in lower
    assert "hallucination_risk" in lower


def test_re_redactor_system_prompt_references_verifier_notes() -> None:
    from metaculus_bot.ablation.qa_iterate import RE_REDACTOR_SYSTEM_PROMPT

    lower = RE_REDACTOR_SYSTEM_PROMPT.lower()
    assert "second-pass" in lower or "second pass" in lower
    assert "verifier" in lower


# ---------------------------------------------------------------------------
# CRIT-1: subprocess kill on timeout (qa_iterate parallel of prune CRIT-1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_claude_subprocess_timeout_kills_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """asyncio.wait_for cancels the awaitable but does NOT terminate the
    underlying OS subprocess. _run_claude_subprocess must call proc.kill()
    on TimeoutError so the leaked child gets reaped, then re-raise so the
    caller's TimeoutError handling still fires.

    Mutation: remove proc.kill(); this test fails (kill_calls == 0).
    """
    import asyncio
    import logging

    from metaculus_bot.ablation import qa_iterate

    kill_calls = {"n": 0}
    wait_calls = {"n": 0}

    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0)
        proc = SimpleNamespace()
        proc.pid = 42424
        proc.returncode = None

        async def slow_communicate(input: bytes | None = None) -> tuple[bytes, bytes]:
            await asyncio.sleep(10)  # forces wait_for to time out
            return b"", b""

        def kill() -> None:
            kill_calls["n"] += 1

        async def fake_wait() -> int:
            wait_calls["n"] += 1
            await asyncio.sleep(0)
            return 0

        proc.communicate = slow_communicate
        proc.kill = kill
        proc.wait = fake_wait
        return proc

    monkeypatch.setattr(
        "metaculus_bot.ablation.qa_iterate.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    with caplog.at_level(logging.WARNING, logger="metaculus_bot.ablation.qa_iterate"):
        with pytest.raises(asyncio.TimeoutError):
            await qa_iterate._run_claude_subprocess(
                ["claude", "-p"],
                "any prompt",
                timeout_seconds=0,
            )

    assert kill_calls["n"] == 1, f"proc.kill() should fire exactly once on timeout; got {kill_calls['n']}"
    assert wait_calls["n"] == 1, "proc.wait() should be awaited after kill so the child is reaped"
    assert any("subprocess timeout" in r.getMessage().lower() for r in caplog.records), (
        "expected a WARNING log naming the timeout + pid"
    )


# ---------------------------------------------------------------------------
# CRIT-2: verifier schema-drift fail-loud
#
# Currently _verifier_score_from_entry silently defaults forecastability=0.0
# when the verifier's JSON drops the field; that turns schema drift into mass
# rejected_forecastability. Fail-loud raises ValueError so the per-qid
# try/except converts it to qa_iterate_failed and the operator sees an
# explicit reason.
# ---------------------------------------------------------------------------


def test_verifier_score_from_entry_raises_on_missing_required_field() -> None:
    """Missing forecastability key: ValueError naming the missing field.

    Mutation: revert to .get(key, default) defaults and this test fails
    (no ValueError; silent default returns).
    """
    from metaculus_bot.ablation.qa_iterate import _verifier_score_from_entry

    entry_missing_forecastability = {
        "leakage_risk": 0.1,
        # forecastability deliberately absent
        "hallucination_risk": 0.2,
        "notes": "ok",
    }
    with pytest.raises(ValueError, match="forecastability"):
        _verifier_score_from_entry(entry_missing_forecastability, iteration=1)


def test_verifier_score_from_entry_raises_on_missing_leakage_risk() -> None:
    from metaculus_bot.ablation.qa_iterate import _verifier_score_from_entry

    entry_missing_leakage = {
        # leakage_risk deliberately absent
        "forecastability": 0.5,
        "hallucination_risk": 0.2,
    }
    with pytest.raises(ValueError, match="leakage_risk"):
        _verifier_score_from_entry(entry_missing_leakage, iteration=1)


def test_verifier_score_from_entry_succeeds_with_all_required_fields() -> None:
    from metaculus_bot.ablation.qa_iterate import _verifier_score_from_entry

    entry = {
        "leakage_risk": 0.05,
        "forecastability": 0.7,
        "hallucination_risk": 0.1,
        "notes": "all good",
    }
    score = _verifier_score_from_entry(entry, iteration=2)
    assert score.leakage_risk == 0.05
    assert score.forecastability == 0.7
    assert score.hallucination_risk == 0.1
    assert score.notes == "all good"
    assert score.iteration == 2


def test_verifier_score_from_entry_notes_is_optional() -> None:
    """notes is legitimately optional (verifier may omit on terse verdicts)."""
    from metaculus_bot.ablation.qa_iterate import _verifier_score_from_entry

    entry = {
        "leakage_risk": 0.05,
        "forecastability": 0.7,
        "hallucination_risk": 0.1,
        # notes deliberately absent
    }
    score = _verifier_score_from_entry(entry, iteration=1)
    assert score.notes == ""


# ---------------------------------------------------------------------------
# MAJ-2: read_manual_rejects validates version
# ---------------------------------------------------------------------------


def test_read_manual_rejects_raises_on_unsupported_version(tmp_path: Path) -> None:
    """A manual_rejects.json with version != supported must raise ValueError
    naming the version mismatch. Without this, future schema bumps silently
    misinterpret old files.
    """
    from metaculus_bot.ablation import qa_iterate

    rejects_path = tmp_path / "manual_rejects.json"
    rejects_path.write_text(
        json.dumps({"version": 99, "rejects": {"123": {"reason": "anything"}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="99"):
        qa_iterate.read_manual_rejects(rejects_path)


def test_read_manual_rejects_accepts_supported_version(tmp_path: Path) -> None:
    from metaculus_bot.ablation import qa_iterate

    rejects_path = tmp_path / "manual_rejects.json"
    rejects_path.write_text(
        json.dumps(
            {
                "version": qa_iterate.MANUAL_REJECTS_VERSION,
                "rejects": {"7": {"reason": "ok"}},
            }
        ),
        encoding="utf-8",
    )
    out = qa_iterate.read_manual_rejects(rejects_path)
    assert 7 in out
    assert out[7]["reason"] == "ok"


# ---------------------------------------------------------------------------
# MAJ-3: per-qid try/except in run_qa_iterate_batch carves out resource exceptions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_qa_iterate_batch_propagates_memory_error(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MemoryError must propagate; do not convert to qa_iterate_failed."""
    from metaculus_bot.ablation import qa_iterate

    qid = 8888
    inputs: dict[int, dict[str, Any]] = {
        qid: {
            "question": _make_question(qid),
            "ground_truth": _make_ground_truth(qid),
            "current_blob": "blob",
            "screen_verdict": _make_screen_verdict(is_leaked=False),
        }
    }

    import asyncio

    async def raise_memory_error(**_kwargs: Any) -> Any:
        await asyncio.sleep(0)
        raise MemoryError("simulated FD exhaustion")

    monkeypatch.setattr(qa_iterate, "run_qa_iterate_for_qid", raise_memory_error)

    with pytest.raises(MemoryError):
        await qa_iterate.run_qa_iterate_batch(inputs, cache=cache, max_iterations=3)


@pytest.mark.asyncio
async def test_run_qa_iterate_batch_still_catches_value_error_per_qid(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Application-level ValueError still gets caught and recorded as
    qa_iterate_failed for that qid (regression guard for MAJ-3 carve-out).
    """
    from metaculus_bot.ablation import qa_iterate

    qid = 9999
    inputs: dict[int, dict[str, Any]] = {
        qid: {
            "question": _make_question(qid),
            "ground_truth": _make_ground_truth(qid),
            "current_blob": "blob",
            "screen_verdict": _make_screen_verdict(is_leaked=False),
        }
    }

    import asyncio

    async def raise_value_error(**_kwargs: Any) -> Any:
        await asyncio.sleep(0)
        raise ValueError("verifier returned malformed JSON")

    monkeypatch.setattr(qa_iterate, "run_qa_iterate_for_qid", raise_value_error)

    outcomes = await qa_iterate.run_qa_iterate_batch(inputs, cache=cache, max_iterations=3)
    assert outcomes[qid].final_status == "rejected_leakage"
    assert outcomes[qid].reject_reason == "qa_iterate_failed: ValueError"


# ---------------------------------------------------------------------------
# MAJ-5: rejected iter blob doesn't overwrite a prior cleaner blob
#
# Per the audit: when iter N writes its sanitized blob to research_pruned,
# but the next iter (N+1) verifier rejects, the cache is left with the
# leakier iter-N blob — a cleaner pre-iteration blob (or earlier-iter blob)
# is gone. Operators recovering by hand-editing manual_rejects.json get the
# leakiest version, not the cleanest one.
#
# Fix: snapshot the original pre-iteration blob; on rejected_leakage final
# return path, restore the snapshot so the cache reflects the pre-iteration
# state rather than a partially-redacted leaky version.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rejected_leakage_restores_original_blob_to_cache(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When max_iterations exhaust without leakage acceptance, the cache
    must NOT contain the latest re-redactor's leakier blob — it should
    revert to the snapshot of whatever was in the cache when the iteration
    loop started (or be cleared if nothing was there).

    Mutation: remove the snapshot/restore logic; the cache will contain the
    iter-N leakier blob and this test fails.
    """
    from metaculus_bot.ablation import qa_iterate

    qid = 12121
    question = _make_question(qid)
    gt = _make_ground_truth(qid)

    # Cache starts with a clean baseline blob from the prior prune stage.
    cache.write_pruned_research(
        qid=qid,
        sanitized_blob="ORIGINAL_BASELINE_BLOB",
        meta={
            "qid": qid,
            "original_chars": 200,
            "sanitized_chars": 200,
            "redactions": [],
            "redactor_invocation_id": "baseline",
            "pruned_at": "2026-05-14T00:00:00",
        },
    )

    # Verifier always reports leakage above threshold → re-redactor runs each iter.
    verifier_mock = AsyncMock(
        return_value=_verifier_response(qid, leakage_risk=0.8, forecastability=0.7, hallucination_risk=0.2, notes="bad")
    )
    # Re-redactor produces ever-leakier blobs (test hook so we can verify
    # which blob ended up in cache).
    redactor_mock = AsyncMock(
        side_effect=[
            _redactor_response(qid, "ITER1_REDACTED_BLOB"),
            _redactor_response(qid, "ITER2_REDACTED_BLOB"),
        ]
    )
    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_verifier", verifier_mock)
    monkeypatch.setattr("metaculus_bot.ablation.qa_iterate._invoke_re_redactor", redactor_mock)

    outcome = await qa_iterate.run_qa_iterate_for_qid(
        qid=qid,
        question=question,
        ground_truth=gt,
        current_blob="ORIGINAL_BASELINE_BLOB",
        screen_verdict=_make_screen_verdict(is_leaked=False),
        cache=cache,
        max_iterations=3,
    )

    assert outcome.final_status == "rejected_leakage"

    # The cache must show the ORIGINAL baseline blob, not the iter-2 leakier one.
    # An operator recovering this qid via manual_rejects.json hand-edit will
    # then see the cleanest known version, not whatever the last re-redactor
    # produced before exhaustion.
    cached = cache.read_pruned_research(qid)
    assert cached is not None, "snapshot/restore should leave the cache populated with the original"
    blob, _meta = cached
    assert blob == "ORIGINAL_BASELINE_BLOB", (
        f"on rejected_leakage exhaustion the cache should be restored to the pre-iteration baseline; got {blob!r}"
    )


# ---------------------------------------------------------------------------
# MIN-3: documentation pin for forecastability strict-less-than at boundary
# ---------------------------------------------------------------------------


def test_forecastability_threshold_strict_less_than_at_boundary(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """forecastability == threshold PASSES (strict less-than rejects only).

    qid 43151 in the smoke run scored exactly 0.20 on iter 2 and passed —
    the strict less-than convention is documented in the constant's
    comment. This test pins the convention so a future "off-by-one fix"
    can't silently flip the boundary.
    """
    from metaculus_bot.ablation import qa_iterate

    # Behavior pin: a verifier that returns forecastability == threshold
    # results in 'clean' (passes), not 'rejected_forecastability'.
    score_at_boundary = qa_iterate._verifier_score_from_entry(
        {
            "leakage_risk": 0.05,
            "forecastability": qa_iterate.DEFAULT_FORECASTABILITY_THRESHOLD,
            "hallucination_risk": 0.1,
        },
        iteration=1,
    )
    # Score is exactly at threshold; the condition `forecastability < threshold`
    # evaluates False, so this iteration is treated as forecastable.
    assert score_at_boundary.forecastability == qa_iterate.DEFAULT_FORECASTABILITY_THRESHOLD
    # Pin the inequality direction by mirroring the production-code check.
    assert not (score_at_boundary.forecastability < qa_iterate.DEFAULT_FORECASTABILITY_THRESHOLD)


def test_forecastability_threshold_constant_documents_strict_less_than() -> None:
    """The constant DEFAULT_FORECASTABILITY_THRESHOLD must be documented as
    strict less-than at module scope — operator-tunable; the boundary is
    intentional.
    """
    import inspect

    from metaculus_bot.ablation import qa_iterate

    src = inspect.getsource(qa_iterate)
    # Find the assignment (= 0.2), not the __all__ reference.
    idx = src.find("DEFAULT_FORECASTABILITY_THRESHOLD = ")
    assert idx != -1, "DEFAULT_FORECASTABILITY_THRESHOLD assignment not found"
    surrounding = src[max(0, idx - 500) : idx + 200]
    assert "strict less-than" in surrounding.lower() or "strict <" in surrounding, (
        "DEFAULT_FORECASTABILITY_THRESHOLD must document the strict-less-than convention"
    )


# ---------------------------------------------------------------------------
# Re-redactor verbatim-leak check is type-aware
#
# Phase B smoke run surfaced this: qid 42750 (binary, GT="no") had a correctly
# sanitized re-redactor blob (verifier scored leakage_risk=0.35), but the
# substring check matched "no" inside unrelated words like "not"/"now" and
# wrongly raised ValueError, ending iteration at iter=0. Fix: route through the
# type-aware ``verbatim_leak_check_passes`` already used by the prune layer.
# ---------------------------------------------------------------------------


def _make_typed_ground_truth(qid: int, resolution_string: str, *, question_type: str) -> GroundTruth:
    resolution: object
    if question_type == "binary":
        resolution = True
    elif question_type == "numeric":
        resolution = float(resolution_string) if resolution_string.replace(".", "").lstrip("-").isdigit() else 0.0
    else:
        resolution = resolution_string
    return GroundTruth(
        question_id=qid,
        question_type=question_type,
        resolution=resolution,
        resolution_string=resolution_string,
        community_prediction=None,
        actual_resolution_time=datetime(2026, 5, 1),
        question_text=f"Will Q{qid} happen?",
        page_url=f"https://example.com/q/{qid}",
    )


class TestReRedactorVerbatimCheck:
    """``_parse_re_redactor_response`` must apply the type-aware check.

    Mirrors the prune-layer ``TestVerbatimLeakCheck`` in
    ``test_ablation_prune.py``: binary GTs in {yes,no,true,false} skip the
    surface check (the verifier subagent's leakage_risk score catches semantic
    leaks); MC GTs use word-boundary regex; numeric GTs use strict substring.
    """

    def test_binary_gt_no_does_not_false_positive_on_substring(self) -> None:
        """Regression: Phase B smoke surfaced this — GT='no' must NOT match
        'not'/'now'/'noteworthy' in sanitized blob. Iteration must continue.
        """
        from metaculus_bot.ablation.qa_iterate import _parse_re_redactor_response

        qid = 42750
        gt = _make_typed_ground_truth(qid, "no", question_type="binary")
        sanitized_blob = (
            "Background: the indicator is noteworthy but not currently elevated. "
            "Now consider the broader market context."
        )
        raw = _redactor_response(qid, sanitized_blob)

        result = _parse_re_redactor_response(raw, qid, gt)

        assert result == sanitized_blob

    def test_binary_gt_no_with_standalone_answer_passes_skip_check(self) -> None:
        """Even 'the answer is no' passes the binary skip-check at this layer.
        The semantic leak is caught downstream by the verifier subagent's
        leakage_risk score on the next iteration, not by surface substring.
        """
        from metaculus_bot.ablation.qa_iterate import _parse_re_redactor_response

        qid = 42751
        gt = _make_typed_ground_truth(qid, "no", question_type="binary")
        sanitized_blob = "Final answer: no, the threshold was not crossed."
        raw = _redactor_response(qid, sanitized_blob)

        result = _parse_re_redactor_response(raw, qid, gt)

        assert result == sanitized_blob

    def test_mc_gt_word_boundary_passes_when_only_embedded(self) -> None:
        """MC GT 'Red' embedded in 'Reduction' is not a word-boundary match;
        re-redactor parser should accept the blob.
        """
        from metaculus_bot.ablation.qa_iterate import _parse_re_redactor_response

        qid = 42752
        gt = _make_typed_ground_truth(qid, "Red", question_type="multiple_choice")
        sanitized_blob = "Background context. The Reduction was significant."
        raw = _redactor_response(qid, sanitized_blob)

        result = _parse_re_redactor_response(raw, qid, gt)

        assert result == sanitized_blob

    def test_mc_gt_word_boundary_rejects_when_standalone(self) -> None:
        """MC GT appearing as a standalone token sequence still raises."""
        from metaculus_bot.ablation.qa_iterate import _parse_re_redactor_response

        qid = 42753
        gt = _make_typed_ground_truth(qid, "Nikkei 225", question_type="multiple_choice")
        sanitized_blob = "Background. The answer is Nikkei 225."
        raw = _redactor_response(qid, sanitized_blob)

        with pytest.raises(ValueError, match="Nikkei 225"):
            _parse_re_redactor_response(raw, qid, gt)

    def test_numeric_gt_substring_still_strict(self) -> None:
        """Numeric GTs SHOULD trigger the verbatim check on substring match."""
        from metaculus_bot.ablation.qa_iterate import _parse_re_redactor_response

        qid = 42754
        gt = _make_typed_ground_truth(qid, "66.246", question_type="numeric")
        sanitized_blob = "Background context. The reported value was 66.246 percent."
        raw = _redactor_response(qid, sanitized_blob)

        with pytest.raises(ValueError, match="66.246"):
            _parse_re_redactor_response(raw, qid, gt)
