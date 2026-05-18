"""Tests for the redactor-driven prune stage of the ablation benchmark.

The prune stage strips resolution-revealing content from cached Gemini research
blobs by invoking ``claude -p`` (headless Claude Code) as a subprocess. Each
batch of up to 10 questions is sent as one subprocess call. The subagent
returns JSON with one entry per qid; we validate, write to cache, and return.

All tests mock the subprocess primitive — no live ``claude -p`` invocations.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from forecasting_tools import MetaculusQuestion

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.backtest.scoring import GroundTruth

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_question(qid: int, *, text: str | None = None) -> MetaculusQuestion:
    q = SimpleNamespace(
        id_of_question=qid,
        question_text=text or f"Will Q{qid} happen?",
        resolution_criteria=f"Resolves YES if Q{qid} happens.",
        fine_print=f"See bls.gov for Q{qid}.",
        background_info=f"Background for Q{qid}",
        page_url=f"https://example.com/q/{qid}",
        open_time=datetime(2026, 1, 1),
        scheduled_resolution_time=datetime(2026, 5, 1),
    )
    return cast(MetaculusQuestion, q)


def _make_ground_truth(
    qid: int,
    resolution_string: str = "Yes",
    *,
    question_type: str = "binary",
) -> GroundTruth:
    # Type-aware verbatim detector branches on question_type so callers can
    # exercise the binary {yes,no,true,false} skip, MC word-boundary check,
    # and numeric strict-substring branches independently.
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


@pytest.fixture
def cache(tmp_path: Path) -> AblationCache:
    return AblationCache(tmp_path / "abl")


def _patch_subprocess(monkeypatch: pytest.MonkeyPatch, stdout_payloads: list) -> AsyncMock:
    mock = AsyncMock(side_effect=stdout_payloads)
    monkeypatch.setattr("metaculus_bot.ablation.prune._invoke_claude_redactor", mock)
    return mock


def _build_response(per_qid: list) -> str:
    payload = {
        "results": [{"qid": qid, "sanitized_blob": blob, "redactions": redactions} for qid, blob, redactions in per_qid]
    }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# run_prune_for_qids — cache behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_prune_for_qids_cache_hit_short_circuits(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from metaculus_bot.ablation.prune import run_prune_for_qids

    cache.write_pruned_research(
        qid=42,
        sanitized_blob="cached sanitized blob",
        meta={
            "qid": 42,
            "original_chars": 100,
            "sanitized_chars": 50,
            "redactions": [],
            "redactor_invocation_id": "z",
            "pruned_at": "2026-05-13T18:00:00",
        },
    )

    question = _make_question(42)
    gt = _make_ground_truth(42)
    blob = "raw research blob with some content"

    mock = _patch_subprocess(monkeypatch, [])

    results = await run_prune_for_qids([(question, gt, blob)], cache)

    mock.assert_not_called()
    assert results[42] is not None
    sanitized, _meta = results[42]
    assert sanitized == "cached sanitized blob"


@pytest.mark.asyncio
async def test_run_prune_for_qids_force_bypasses_cache(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from metaculus_bot.ablation.prune import run_prune_for_qids

    cache.write_pruned_research(
        qid=42,
        sanitized_blob="OLD sanitized",
        meta={
            "qid": 42,
            "original_chars": 100,
            "sanitized_chars": 50,
            "redactions": [],
            "redactor_invocation_id": "z",
            "pruned_at": "2026-05-13T18:00:00",
        },
    )

    question = _make_question(42)
    gt = _make_ground_truth(42, "Yes")
    blob = "raw research blob"

    mock = _patch_subprocess(
        monkeypatch,
        [_build_response([(42, "FRESH sanitized", [{"original_excerpt": "x", "reason": "y"}])])],
    )

    results = await run_prune_for_qids([(question, gt, blob)], cache, force=True)

    mock.assert_called_once()
    assert results[42] is not None
    sanitized, _meta = results[42]
    assert sanitized == "FRESH sanitized"

    cached = cache.read_pruned_research(qid=42)
    assert cached is not None
    assert cached[0] == "FRESH sanitized"


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_prune_for_qids_batches_correctly(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """25 questions, batch_size=10 → 3 subprocess invocations (10, 10, 5)."""
    from metaculus_bot.ablation.prune import run_prune_for_qids

    triples = []
    for qid in range(1000, 1025):
        triples.append((_make_question(qid), _make_ground_truth(qid, f"answer-{qid}"), f"raw blob {qid}"))

    batch_qids = [list(range(1000, 1010)), list(range(1010, 1020)), list(range(1020, 1025))]
    canned_responses = [_build_response([(qid, f"sanitized-{qid}", []) for qid in batch]) for batch in batch_qids]
    mock = _patch_subprocess(monkeypatch, canned_responses)

    results = await run_prune_for_qids(triples, cache, batch_size=10)

    assert mock.await_count == 3
    assert len(results) == 25
    for qid in range(1000, 1025):
        result = results[qid]
        assert result is not None
        sanitized, _meta = result
        assert sanitized == f"sanitized-{qid}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_prune_for_qids_validates_sanitized_blob_excludes_ground_truth(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from metaculus_bot.ablation.prune import run_prune_for_qids

    triples = [
        (_make_question(1), _make_ground_truth(1, "17,237,442"), "raw blob 1"),
        (_make_question(2), _make_ground_truth(2, "PMI 53.6"), "raw blob 2"),
    ]

    response = _build_response(
        [
            (1, "Background context. The weekly total was 17,237,442 passengers.", []),
            (2, "Background context with no resolution.", []),
        ]
    )
    _patch_subprocess(monkeypatch, [response])

    results = await run_prune_for_qids(triples, cache)

    assert results[1] is None
    assert results[2] is not None
    sanitized, _meta = results[2]
    assert sanitized == "Background context with no resolution."

    assert cache.read_pruned_research(qid=1) is None
    assert cache.read_pruned_research(qid=2) is not None


@pytest.mark.asyncio
async def test_run_prune_for_qids_validates_qid_set_match(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from metaculus_bot.ablation.prune import run_prune_for_qids

    triples = [
        (_make_question(1), _make_ground_truth(1), "raw blob 1"),
        (_make_question(2), _make_ground_truth(2), "raw blob 2"),
    ]

    response = _build_response(
        [
            (1, "sanitized 1", []),
            (2, "sanitized 2", []),
            (99, "sanitized 99 — this should be discarded", []),
        ]
    )
    _patch_subprocess(monkeypatch, [response])

    import logging

    with caplog.at_level(logging.WARNING, logger="metaculus_bot.ablation.prune"):
        results = await run_prune_for_qids(triples, cache)

    assert results[1] is not None
    assert results[2] is not None
    assert 99 not in results
    assert any("99" in r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)


@pytest.mark.asyncio
async def test_run_prune_for_qids_validates_empty_sanitized_blob_rejected(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from metaculus_bot.ablation.prune import run_prune_for_qids

    triples = [
        (_make_question(1), _make_ground_truth(1), "raw blob 1"),
        (_make_question(2), _make_ground_truth(2), "raw blob 2"),
    ]

    response = _build_response(
        [
            (1, "", []),
            (2, "real sanitized content", []),
        ]
    )
    _patch_subprocess(monkeypatch, [response])

    results = await run_prune_for_qids(triples, cache)

    assert results[1] is None
    assert results[2] is not None


@pytest.mark.asyncio
async def test_run_prune_for_qids_handles_subprocess_failure(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One batch's subprocess raises CalledProcessError. Other batches still complete.

    Post M4: when a batch fails entirely the per-qid recovery path retries
    each qid in a 1-qid batch. If those retries also fail, the qid lands as
    None (same end state as before). This test feeds matching number of
    failures: 1 failed batch (10 qids) + 10 failing per-qid retries +
    1 successful 5-qid batch.
    """
    import subprocess

    from metaculus_bot.ablation.prune import run_prune_for_qids

    triples = [(_make_question(qid), _make_ground_truth(qid), f"raw blob {qid}") for qid in range(1, 16)]

    failing_batch = subprocess.CalledProcessError(returncode=1, cmd=["claude"], stderr=b"oops")
    failing_retry = subprocess.CalledProcessError(returncode=1, cmd=["claude"], stderr=b"retry oops")
    success_response = _build_response([(qid, f"sanitized-{qid}", []) for qid in range(11, 16)])
    _patch_subprocess(
        monkeypatch,
        [failing_batch] + [failing_retry] * 10 + [success_response],
    )

    results = await run_prune_for_qids(triples, cache, batch_size=10)

    for qid in range(1, 11):
        assert results[qid] is None

    for qid in range(11, 16):
        assert results[qid] is not None


@pytest.mark.asyncio
async def test_run_prune_for_qids_handles_invalid_json(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from metaculus_bot.ablation.prune import run_prune_for_qids

    triples = [
        (_make_question(1), _make_ground_truth(1), "raw blob 1"),
        (_make_question(2), _make_ground_truth(2), "raw blob 2"),
    ]
    _patch_subprocess(monkeypatch, ["this is definitely not valid JSON {{{{"])

    import logging

    with caplog.at_level(logging.ERROR, logger="metaculus_bot.ablation.prune"):
        results = await run_prune_for_qids(triples, cache)

    assert results[1] is None
    assert results[2] is None
    assert any(r.levelno == logging.ERROR for r in caplog.records)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_build_redactor_prompt_includes_question_fields() -> None:
    from metaculus_bot.ablation.prune import _build_redactor_prompt

    triples = [
        (
            _make_question(123, text="Will the cat jump?"),
            _make_ground_truth(123, "Yes - by 12 cm"),
            "raw research blob 123 contains relevant context",
        ),
    ]
    prompt = _build_redactor_prompt(triples)

    assert "Will the cat jump?" in prompt
    assert "Resolves YES if Q123 happens." in prompt
    assert "Yes - by 12 cm" in prompt
    assert "raw research blob 123 contains relevant context" in prompt
    assert "123" in prompt


def test_build_redactor_prompt_warns_redactor_about_ground_truth() -> None:
    from metaculus_bot.ablation.prune import _build_redactor_prompt

    triples = [
        (
            _make_question(7),
            _make_ground_truth(7, "Yes"),
            "blob",
        ),
    ]
    prompt = _build_redactor_prompt(triples)

    lower = prompt.lower()
    assert "ground truth" in lower
    assert "must not" in lower or "do not" in lower or "must never" in lower


def test_build_redactor_prompt_describes_json_schema() -> None:
    from metaculus_bot.ablation.prune import _build_redactor_prompt

    triples = [(_make_question(7), _make_ground_truth(7), "blob")]
    prompt = _build_redactor_prompt(triples)
    assert "qid" in prompt
    assert "sanitized_blob" in prompt
    assert "redactions" in prompt


def test_redactor_system_prompt_warns_about_implication_leakage() -> None:
    """The redactor needs to know that anchored / comparative phrasing (e.g.
    "X PMI was unchanged from its March reading of 52.7%") leaks the answer
    even when no April value is named directly.
    """
    from metaculus_bot.ablation.prune import REDACTOR_SYSTEM_PROMPT

    lower = REDACTOR_SYSTEM_PROMPT.lower()
    # At least one of the implication-leak terms should be present.
    implication_terms = ["implication", "anchor", "anchored", "bracket", "threshold framing"]
    assert any(term in lower for term in implication_terms), (
        "REDACTOR_SYSTEM_PROMPT should explicitly warn about implication leakage; "
        f"none of {implication_terms} appear in it"
    )


def test_redactor_system_prompt_includes_example() -> None:
    """The prompt should ground its rules in a worked example."""
    from metaculus_bot.ablation.prune import REDACTOR_SYSTEM_PROMPT

    upper = REDACTOR_SYSTEM_PROMPT.upper()
    assert "EXAMPLE" in upper, "REDACTOR_SYSTEM_PROMPT should contain a worked EXAMPLE"


def test_redactor_system_prompt_includes_second_example() -> None:
    """A second worked example is required to cover a real failure mode the
    first example didn't catch (subtler whole-sentence implication leak).
    """
    from metaculus_bot.ablation.prune import REDACTOR_SYSTEM_PROMPT

    assert "EXAMPLE 2" in REDACTOR_SYSTEM_PROMPT, (
        "REDACTOR_SYSTEM_PROMPT must contain a second worked example marked 'EXAMPLE 2' "
        "to cover whole-sentence implication-leak failure modes that the first example doesn't capture."
    )


def test_redactor_system_prompt_explains_anchored_value_failure_mode() -> None:
    """The prompt must explain WHY the redactor needs to redact ENTIRE comparative
    sentences, not just the leading clauses — the failure mode observed in qa_research_9.log.
    """
    from metaculus_bot.ablation.prune import REDACTOR_SYSTEM_PROMPT

    upper = REDACTOR_SYSTEM_PROMPT.upper()
    # A phrase like "ALGEBRAICALLY DETERMINES" or equivalent must surface the
    # whole-sentence redaction principle.
    indicators = [
        "ALGEBRAICALLY DETERMINES",
        "WHOLE SENTENCE",
        "ENTIRE COMPARATIVE",
        "ENTIRE SENTENCE",
    ]
    assert any(term in upper for term in indicators), (
        "REDACTOR_SYSTEM_PROMPT must contain explicit whole-sentence redaction guidance "
        f"(one of {indicators}). The phrase 'X was unchanged from PRIOR_VALUE' algebraically "
        "determines the current value and must be redacted in full, not just the leading clause."
    )


# ---------------------------------------------------------------------------
# Subprocess argv
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_claude_redactor_uses_correct_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from metaculus_bot.ablation.prune import _invoke_claude_redactor

    captured: dict = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured["argv"] = list(args)
        captured["kwargs"] = kwargs
        await asyncio.sleep(0)
        proc = MagicMock()

        async def communicate(input=None):
            captured["stdin"] = input
            await asyncio.sleep(0)
            return b'{"results": []}', b""

        proc.communicate = communicate
        proc.returncode = 0
        return proc

    monkeypatch.setattr("metaculus_bot.ablation.prune.asyncio.create_subprocess_exec", fake_create_subprocess_exec)

    result = await _invoke_claude_redactor("a redactor prompt")
    assert result == '{"results": []}'

    argv = captured["argv"]
    assert any("claude" in str(a) for a in argv)
    # Deliberately NOT --bare (cargo-cult fraud_research's known-good pattern).
    assert "--bare" not in argv
    # Deliberately NOT --allowedTools (known-fragile in non-interactive mode).
    assert "--allowedTools" not in argv
    assert "-p" in argv or "--print" in argv
    assert "--output-format" in argv
    output_idx = argv.index("--output-format")
    # Plain text output — the redactor's response IS JSON because the prompt
    # asks for it; we don't need an outer JSON envelope.
    assert argv[output_idx + 1] == "text"
    assert "--max-turns" in argv
    max_idx = argv.index("--max-turns")
    assert argv[max_idx + 1] == "1"
    assert "--permission-mode" in argv
    perm_idx = argv.index("--permission-mode")
    assert argv[perm_idx + 1] == "bypassPermissions"
    # Force-disable prompt-caching beta (Mantle gateway rejects it in headless;
    # diagnosed by fraud_research team 2026-05-06).
    assert "--settings" in argv
    settings_idx = argv.index("--settings")
    settings_payload = argv[settings_idx + 1]
    assert "ENABLE_PROMPT_CACHING_1H" in settings_payload
    assert '"0"' in settings_payload  # value is the string "0", not the int 0
    assert "--append-system-prompt" in argv

    assert captured["stdin"] is not None
    assert b"a redactor prompt" in captured["stdin"]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def test_parse_redactor_response_structured_blob() -> None:
    from metaculus_bot.ablation.prune import _parse_redactor_response

    raw = _build_response(
        [
            (1, "sanitized 1", [{"original_excerpt": "ex", "reason": "states resolution"}]),
            (2, "sanitized 2", []),
        ]
    )
    gts = {1: _make_ground_truth(1, "GT_1"), 2: _make_ground_truth(2, "GT_2")}
    parsed = _parse_redactor_response(raw, expected_qids=[1, 2], ground_truths=gts)

    assert parsed[1] is not None
    blob1, redactions1 = parsed[1]
    assert blob1 == "sanitized 1"
    assert redactions1 == [{"original_excerpt": "ex", "reason": "states resolution"}]

    assert parsed[2] is not None
    blob2, redactions2 = parsed[2]
    assert blob2 == "sanitized 2"
    assert redactions2 == []


def test_parse_redactor_response_skips_qids_with_empty_sanitized_blob() -> None:
    from metaculus_bot.ablation.prune import _parse_redactor_response

    raw = _build_response([(1, "", []), (2, "real content", [])])
    gts = {1: _make_ground_truth(1, "GT_1"), 2: _make_ground_truth(2, "GT_2")}
    parsed = _parse_redactor_response(raw, expected_qids=[1, 2], ground_truths=gts)

    assert parsed[1] is None
    assert parsed[2] is not None


def test_parse_redactor_response_rejects_qid_with_gt_in_blob() -> None:
    from metaculus_bot.ablation.prune import _parse_redactor_response

    raw = _build_response(
        [
            (1, "Background. The total was 17,237,442 passengers.", []),
            (2, "Background. No resolution here.", []),
        ]
    )
    gts = {
        1: _make_ground_truth(1, "17,237,442"),
        2: _make_ground_truth(2, "ABCDEF12345"),
    }
    parsed = _parse_redactor_response(raw, expected_qids=[1, 2], ground_truths=gts)
    assert parsed[1] is None
    assert parsed[2] is not None


def test_parse_redactor_response_case_insensitive_gt_match() -> None:
    """For NUMERIC/MC GTs the substring check is case-insensitive and rejects on match.

    Binary {yes,no,true,false} GTs now skip the substring check entirely (per
    the audit at backtests/ablation/audit_smoke_20260515.md:154 — "no" and
    "yes" appear in many non-leak words and the substring check has zero
    discriminative power). To exercise the case-insensitive substring branch
    we use a non-binary GT type.
    """
    from metaculus_bot.ablation.prune import _parse_redactor_response

    raw = _build_response([(1, "the answer was Tuesday", [])])
    gts = {1: _make_ground_truth(1, "TUESDAY", question_type="multiple_choice")}
    parsed = _parse_redactor_response(raw, expected_qids=[1], ground_truths=gts)
    assert parsed[1] is None


# ---------------------------------------------------------------------------
# Type-aware verbatim-leak detector
#
# Audit at backtests/ablation/audit_smoke_20260515.md:176-194 documents the
# bug: the original substring-only check false-positives on binary "yes"/"no"
# GTs (substring "no" appears 8x in "non-manufacturing") and on MC options
# that share words with the question (e.g., "March 2026" appears in question
# phrasing "Jan/Feb/Mar 2026"). 5 of 11 problematic qids in the smoke run
# failed at this layer alone.
# ---------------------------------------------------------------------------


class TestVerbatimLeakCheck:
    """Type-aware verbatim-leak detector (``verbatim_leak_check_passes``).

    - binary with GT in {yes, no, true, false}: skip check; rely on LLM
      screen + qa_iterate verifier for semantic leakage.
    - multiple_choice: word-boundary regex; reject only when GT appears as
      a standalone token sequence (not as ambient question phrasing).
    - numeric: substring check on normalized GT (numeric resolution strings
      are unique enough that any verbatim hit is a real leak).
    """

    def test_binary_no_gt_skips_substring_check_when_in_non_word(self) -> None:
        """Smoke-run reality: "no" GT + sanitized blob containing
        "non-manufacturing" → previously rejected by substring check.
        Should now PASS (skip).
        """
        from metaculus_bot.ablation.prune import verbatim_leak_check_passes

        gt = _make_ground_truth(1, "no", question_type="binary")
        sanitized = "Background: the China non-manufacturing PMI methodology covers retail trade."
        passes, reason = verbatim_leak_check_passes(sanitized, gt, "binary")
        assert passes is True
        assert reason is None

    def test_binary_yes_gt_skips_substring_check(self) -> None:
        from metaculus_bot.ablation.prune import verbatim_leak_check_passes

        gt = _make_ground_truth(1, "yes", question_type="binary")
        sanitized = "Background context — yesterday's report showed mixed signals."
        passes, reason = verbatim_leak_check_passes(sanitized, gt, "binary")
        assert passes is True
        assert reason is None

    def test_binary_true_gt_skips_substring_check(self) -> None:
        from metaculus_bot.ablation.prune import verbatim_leak_check_passes

        gt = _make_ground_truth(1, "true", question_type="binary")
        sanitized = "Background. Truer statements about market structure are hard to find."
        passes, reason = verbatim_leak_check_passes(sanitized, gt, "binary")
        assert passes is True
        assert reason is None

    def test_binary_false_gt_skips_substring_check(self) -> None:
        from metaculus_bot.ablation.prune import verbatim_leak_check_passes

        gt = _make_ground_truth(1, "false", question_type="binary")
        sanitized = "Background context with no resolution-relevant content."
        passes, reason = verbatim_leak_check_passes(sanitized, gt, "binary")
        assert passes is True
        assert reason is None

    def test_uncommon_binary_gt_falls_through_to_substring(self) -> None:
        """Rare binary GTs not in {yes,no,true,false} (e.g., "draw") fall
        through to the substring check.
        """
        from metaculus_bot.ablation.prune import verbatim_leak_check_passes

        gt = _make_ground_truth(1, "draw", question_type="binary")
        sanitized = "Match notes: the result was a draw between the two teams."
        passes, reason = verbatim_leak_check_passes(sanitized, gt, "binary")
        assert passes is False
        assert reason is not None

    def test_mc_gt_word_boundary_passes_when_only_ambient_phrasing(self) -> None:
        """MC GT "March 2026" appears as ambient question phrasing in
        "comparing Jan 2026, Feb 2026, March 2026 revenues". With word-boundary
        match the GT is present as a token sequence so a strict word-boundary
        match still rejects. The audit recommends this — ambient mentions ARE
        word-boundary matches; MC option-specific leakage is caught by the
        downstream LLM screen + qa_iterate verifier. The narrower test is the
        OPPOSITE: GT must NOT appear at word boundary.
        """
        from metaculus_bot.ablation.prune import verbatim_leak_check_passes

        # Ambient: GT does NOT appear at word boundary, only embedded in another word.
        # E.g., question option "ABC" appearing inside "ABCD".
        gt = _make_ground_truth(1, "Red", question_type="multiple_choice")
        sanitized = "Background context. The Reduction was significant."
        # "Red" is a substring of "Reduction" but not a word-boundary match.
        passes, reason = verbatim_leak_check_passes(sanitized, gt, "multiple_choice")
        assert passes is True
        assert reason is None

    def test_mc_gt_word_boundary_rejects_when_standalone(self) -> None:
        from metaculus_bot.ablation.prune import verbatim_leak_check_passes

        gt = _make_ground_truth(1, "Nikkei 225", question_type="multiple_choice")
        sanitized = "Background. The answer is Nikkei 225."
        passes, reason = verbatim_leak_check_passes(sanitized, gt, "multiple_choice")
        assert passes is False
        assert reason is not None

    def test_numeric_gt_substring_check_strict(self) -> None:
        from metaculus_bot.ablation.prune import verbatim_leak_check_passes

        gt = _make_ground_truth(1, "66.246", question_type="numeric")
        sanitized = "Background context. The reported value was 66.246 percent."
        passes, reason = verbatim_leak_check_passes(sanitized, gt, "numeric")
        assert passes is False
        assert reason is not None

    def test_numeric_gt_not_in_sanitized_passes(self) -> None:
        from metaculus_bot.ablation.prune import verbatim_leak_check_passes

        gt = _make_ground_truth(1, "66.246", question_type="numeric")
        sanitized = "Background. Pre-event guidance was 63-65 percent."
        passes, reason = verbatim_leak_check_passes(sanitized, gt, "numeric")
        assert passes is True
        assert reason is None

    def test_numeric_gt_negative_substring_check_strict(self) -> None:
        from metaculus_bot.ablation.prune import verbatim_leak_check_passes

        gt = _make_ground_truth(1, "-87.9", question_type="numeric")
        sanitized = "Background. The advance trade balance was -87.9 billion."
        passes, reason = verbatim_leak_check_passes(sanitized, gt, "numeric")
        assert passes is False
        assert reason is not None

    def test_empty_gt_string_passes(self) -> None:
        from metaculus_bot.ablation.prune import verbatim_leak_check_passes

        gt = _make_ground_truth(1, "", question_type="numeric")
        sanitized = "anything goes here"
        passes, reason = verbatim_leak_check_passes(sanitized, gt, "numeric")
        assert passes is True
        assert reason is None


# ---------------------------------------------------------------------------
# REDACTOR_SYSTEM_PROMPT preserves pre-event analyst guidance (audit fix)
# ---------------------------------------------------------------------------


def test_redactor_prompt_mentions_pre_event_guidance() -> None:
    """The audit at backtests/ablation/audit_smoke_20260515.md:158-174
    recommends preserving pre-event analyst guidance ranges (e.g., TSMC
    issued Q1 2026 gross margin guidance of 63.0-65.0%) as forecaster
    context. The prompt must mention this concept so the redactor knows
    not to strip such pre-resolution context.
    """
    from metaculus_bot.ablation.prune import REDACTOR_SYSTEM_PROMPT

    assert "guidance" in REDACTOR_SYSTEM_PROMPT.lower(), (
        "REDACTOR_SYSTEM_PROMPT must mention 'guidance' (pre-event analyst guidance) "
        "to instruct the redactor to preserve such ranges; see audit"
    )


# ---------------------------------------------------------------------------
# M4: bound redactor batch failure blast radius
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_prune_for_qids_batch_size_is_parametrized(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--prune-batch-size`` flows through to the batching loop."""
    from metaculus_bot.ablation.prune import run_prune_for_qids

    triples = [
        (_make_question(qid), _make_ground_truth(qid, f"answer-{qid}"), f"raw {qid}") for qid in range(2000, 2007)
    ]

    import asyncio

    batch_qids_seen: list[list[int]] = []

    async def capturing_invoke(prompt: str, **_kwargs: Any) -> str:
        await asyncio.sleep(0)
        ids = [int(line.split("qid=")[-1]) for line in prompt.splitlines() if line.startswith("## qid=")]
        batch_qids_seen.append(ids)
        return _build_response([(qid, f"sanitized-{qid}", []) for qid in ids])

    monkeypatch.setattr("metaculus_bot.ablation.prune._invoke_claude_redactor", capturing_invoke)

    results = await run_prune_for_qids(triples, cache, batch_size=3)

    # 7 qids / 3 per batch = 3 batches (3, 3, 1).
    assert len(batch_qids_seen) == 3
    assert [len(b) for b in batch_qids_seen] == [3, 3, 1]
    assert all(results[qid] is not None for qid in range(2000, 2007))


@pytest.mark.asyncio
async def test_run_prune_for_qids_batch_failure_recovers_per_qid(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the batch subprocess raises, retry each qid in a 1-qid batch.

    Without per-qid recovery, a single intermittent failure drops 10 qids.
    With recovery, the per-qid retry catches transient failures while still
    bounding cost (each retry is its own subprocess invocation).
    """
    import logging
    import subprocess

    from metaculus_bot.ablation.prune import run_prune_for_qids

    triples = [
        (_make_question(qid), _make_ground_truth(qid, f"answer-{qid}"), f"raw {qid}") for qid in range(3000, 3003)
    ]

    import asyncio

    call_count = {"n": 0}

    async def flaky_invoke(prompt: str, **_kwargs: Any) -> str:
        await asyncio.sleep(0)
        call_count["n"] += 1
        # First call (the 3-qid batch) fails entirely.
        if call_count["n"] == 1:
            raise subprocess.CalledProcessError(returncode=1, cmd=["claude"], stderr=b"transient")
        # Subsequent per-qid retries succeed.
        ids = [int(line.split("qid=")[-1]) for line in prompt.splitlines() if line.startswith("## qid=")]
        return _build_response([(qid, f"sanitized-{qid}", []) for qid in ids])

    monkeypatch.setattr("metaculus_bot.ablation.prune._invoke_claude_redactor", flaky_invoke)

    with caplog.at_level(logging.ERROR, logger="metaculus_bot.ablation.prune"):
        results = await run_prune_for_qids(triples, cache, batch_size=3)

    # All three qids must succeed via per-qid recovery.
    for qid in range(3000, 3003):
        result = results[qid]
        assert result is not None, f"qid {qid} should have been recovered"
        sanitized, _meta = result
        assert sanitized == f"sanitized-{qid}"

    # 1 batch failure + 3 per-qid recoveries = 4 invocations.
    assert call_count["n"] == 4


@pytest.mark.asyncio
async def test_run_prune_for_qids_per_qid_recovery_partial_success(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If some per-qid recoveries succeed and others fail, only the failures
    end up as None. Bounds blast radius to specific qids.
    """
    import subprocess

    from metaculus_bot.ablation.prune import run_prune_for_qids

    triples = [
        (_make_question(qid), _make_ground_truth(qid, f"answer-{qid}"), f"raw {qid}") for qid in range(4000, 4003)
    ]

    import asyncio

    async def flaky_invoke(prompt: str, **_kwargs: Any) -> str:
        await asyncio.sleep(0)
        ids = [int(line.split("qid=")[-1]) for line in prompt.splitlines() if line.startswith("## qid=")]
        # Initial multi-qid batch always fails.
        if len(ids) > 1:
            raise subprocess.CalledProcessError(returncode=1, cmd=["claude"], stderr=b"batch fail")
        # Per-qid recovery: 4001 fails, 4000 and 4002 succeed.
        if ids == [4001]:
            raise subprocess.CalledProcessError(returncode=1, cmd=["claude"], stderr=b"qid fail")
        return _build_response([(qid, f"sanitized-{qid}", []) for qid in ids])

    monkeypatch.setattr("metaculus_bot.ablation.prune._invoke_claude_redactor", flaky_invoke)

    results = await run_prune_for_qids(triples, cache, batch_size=3)

    assert results[4000] is not None
    assert results[4001] is None
    assert results[4002] is not None


# ---------------------------------------------------------------------------
# M7: warn on unparseable claude -p stdout envelope
# ---------------------------------------------------------------------------


def test_extract_inner_result_warns_on_unparseable_stdout(caplog: pytest.LogCaptureFixture) -> None:
    """When claude -p stdout is unparseable JSON, log a WARNING before
    passing it through. Without this warning, downstream parsers (e.g.
    _parse_redactor_response) emit confusing errors about missing fields
    rather than the real envelope-shape failure.
    """
    import logging

    from metaculus_bot.ablation.prune import _extract_inner_result

    unparseable = "this is definitely not JSON {{{ }"

    with caplog.at_level(logging.WARNING, logger="metaculus_bot.ablation.prune"):
        result = _extract_inner_result(unparseable)

    # Passthrough still happens for backwards compat with raw-JSON test stubs.
    assert result == unparseable
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("not parseable JSON" in r.getMessage() for r in warning_records)


# ---------------------------------------------------------------------------
# CRIT-1: subprocess kill on timeout
#
# At 50q x 3 iterations x 2 calls = 300 subprocess invocations. Each timeout
# (DEFAULT_TIMEOUT_SECONDS=600) without proc.kill() leaks a child process,
# holding 3 pipe FDs + a process slot until the OS reaps it. Subsequent
# fork() calls compete with leaked children for FDs/process slots.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_claude_redactor_timeout_kills_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """asyncio.wait_for cancels the awaitable but does NOT terminate the
    underlying OS subprocess. A try/except around wait_for must call
    proc.kill() so the leaked child gets reaped, then re-raise so the
    caller's TimeoutError handling still fires.

    Mutation: remove proc.kill(); this test fails (kill_calls == 0).
    """
    import asyncio
    import logging

    from metaculus_bot.ablation.prune import _invoke_claude_redactor

    kill_calls = {"n": 0}
    wait_calls = {"n": 0}

    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0)
        proc = MagicMock()
        proc.pid = 12345
        proc.returncode = None

        async def slow_communicate(input: bytes | None = None) -> tuple[bytes, bytes]:
            await asyncio.sleep(10)  # forces the wait_for to time out
            return b"", b""

        proc.communicate = slow_communicate

        def kill() -> None:
            kill_calls["n"] += 1

        async def fake_wait() -> int:
            wait_calls["n"] += 1
            await asyncio.sleep(0)
            return 0

        proc.kill = kill
        proc.wait = fake_wait
        return proc

    monkeypatch.setattr(
        "metaculus_bot.ablation.prune.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    with caplog.at_level(logging.WARNING, logger="metaculus_bot.ablation.prune"):
        with pytest.raises(asyncio.TimeoutError):
            # timeout_seconds is typed as int; cast a tiny float through int(0)
            # which still triggers wait_for's TimeoutError immediately because
            # slow_communicate awaits 10 seconds.
            await _invoke_claude_redactor("any prompt", timeout_seconds=0)

    assert kill_calls["n"] == 1, f"proc.kill() should fire exactly once on timeout; got {kill_calls['n']}"
    assert wait_calls["n"] == 1, "proc.wait() should be awaited after kill so the child is reaped"
    assert any("subprocess timeout" in r.getMessage().lower() for r in caplog.records), (
        "expected a WARNING log naming the timeout + pid"
    )


# ---------------------------------------------------------------------------
# MAJ-1: redactor batch prompt size guard with binary split
#
# 10 x 80 KB blobs near 800 KB ~ 200K tokens at the edge of Claude's input
# context. Without a size guard, hot batches fail with subprocess.CalledProcessError
# carrying input-too-long; with a guard, recursively split the batch in half
# until each fits.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_batch_splits_oversized_prompt(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A batch with two ~500K-char blobs exceeds APPROX_CHAR_LIMIT and must
    recursively split into 2 single-qid batches. Both qids land in the
    result dict (the mocked claude -p returns happy stubs) and a WARN log
    fires naming the split sizes.
    """
    import asyncio
    import logging

    from metaculus_bot.ablation import prune
    from metaculus_bot.ablation.prune import _process_batch

    big_blob = "x" * 500_000  # 500 KB; 2 of these blow the ~720KB guard
    triples: list[tuple[MetaculusQuestion, GroundTruth, str]] = [
        (_make_question(qid), _make_ground_truth(qid, f"answer-{qid}"), big_blob) for qid in (5000, 5001)
    ]

    invocations: list[list[int]] = []

    async def stub_invoke(prompt: str, **_kwargs: Any) -> str:
        await asyncio.sleep(0)
        ids = [int(line.split("qid=")[-1]) for line in prompt.splitlines() if line.startswith("## qid=")]
        invocations.append(ids)
        return _build_response([(qid, f"sanitized-{qid}", []) for qid in ids])

    monkeypatch.setattr("metaculus_bot.ablation.prune._invoke_claude_redactor", stub_invoke)

    with caplog.at_level(logging.WARNING, logger="metaculus_bot.ablation.prune"):
        results = await _process_batch(
            triples,
            cache,
            claude_executable=prune.DEFAULT_CLAUDE_EXECUTABLE,
            timeout_seconds=prune.DEFAULT_TIMEOUT_SECONDS,
        )

    # Both qids land via the recursive split.
    assert results[5000] is not None
    assert results[5001] is not None

    # Recursive split should have produced 2 single-qid invocations
    # (no successful 2-qid call because the original prompt was over the limit).
    assert [5000] in invocations and [5001] in invocations
    assert all(len(ids) == 1 for ids in invocations), (
        f"expected only single-qid invocations after split; saw batch sizes {[len(i) for i in invocations]}"
    )

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        "too large" in r.getMessage().lower() and "splitting" in r.getMessage().lower() for r in warning_records
    ), "expected a WARNING log naming the prompt-too-large split"


@pytest.mark.asyncio
async def test_process_batch_singleton_oversized_fails_loud(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A 1-qid batch whose prompt is still over the limit cannot be split
    further. The function must mark that single qid as None (not invoke
    claude -p with an over-context prompt) and emit an ERROR log so the
    operator knows to truncate the blob upstream.
    """
    import asyncio
    import logging

    from metaculus_bot.ablation import prune
    from metaculus_bot.ablation.prune import _process_batch

    huge_blob = "y" * 1_000_000  # 1 MB; over the limit even alone
    triples: list[tuple[MetaculusQuestion, GroundTruth, str]] = [
        (_make_question(6000), _make_ground_truth(6000, "answer"), huge_blob)
    ]

    invoke_calls = {"n": 0}

    async def stub_invoke(_prompt: str, **_kwargs: Any) -> str:
        await asyncio.sleep(0)
        invoke_calls["n"] += 1
        return _build_response([(6000, "sanitized", [])])

    monkeypatch.setattr("metaculus_bot.ablation.prune._invoke_claude_redactor", stub_invoke)

    with caplog.at_level(logging.ERROR, logger="metaculus_bot.ablation.prune"):
        results = await _process_batch(
            triples,
            cache,
            claude_executable=prune.DEFAULT_CLAUDE_EXECUTABLE,
            timeout_seconds=prune.DEFAULT_TIMEOUT_SECONDS,
        )

    assert results[6000] is None
    assert invoke_calls["n"] == 0, "must not invoke claude -p with an over-limit prompt"
    error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("too large" in r.getMessage().lower() and "qid=6000" in r.getMessage() for r in error_records), (
        "expected an ERROR log naming the over-limit single qid"
    )
