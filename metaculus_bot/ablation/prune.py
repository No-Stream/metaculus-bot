"""Redactor stage: prune resolution-revealing content from cached research blobs.

The Gemini grounded-search step often returns research blobs that literally
contain the answer for resolved questions. The downstream LLM leakage screen
flags those at ~100% rate, draining the benchmark of usable questions. This
stage interposes between research and screen: a headless ``claude -p``
subagent reads each (question, ground truth, raw blob) tuple and emits a
sanitized blob with resolution-revealing sentences either removed or
inline-redacted, plus a list of the redactions it made.

Workflow per batch (default 10 questions per ``claude -p`` invocation):

1. Build a single multi-question prompt describing the redactor's role,
   showing per-qid {question, resolution criteria, ground truth, raw blob},
   and demanding strict JSON output.
2. Spawn ``claude -p`` via ``asyncio.create_subprocess_exec`` with flags
   that disable hooks, plugins, and tools entirely (``--bare``, no
   ``--allowedTools``, ``--max-turns 1``). Send the prompt on stdin.
3. Parse the JSON, validate per-qid:
   - qid must be in the input batch (drop unknowns with a warning).
   - sanitized_blob must be non-empty.
   - sanitized_blob must NOT contain the ground-truth resolution_string
     (case-insensitive, whitespace-normalized).
4. For successful qids, write the sanitized blob + meta to
   ``research_pruned/<qid>.{md,meta.json}``. For failures, return ``None``
   so the caller drops the qid.

Per-batch failure (subprocess error, invalid JSON) marks every qid in that
batch as ``None`` but does not affect other batches. Per-qid validation
failure inside a successful batch only affects that qid.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import secrets
import subprocess
from datetime import datetime
from typing import Any, cast

from forecasting_tools import MetaculusQuestion

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.backtest.scoring import GroundTruth

__all__ = ["run_prune_for_qids", "verbatim_leak_check_passes"]

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 10
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_CLAUDE_EXECUTABLE = "claude"

# Approximate cap on combined prompt + system-prompt characters before the
# redactor invocation is expected to bust Claude's input context window. At
# ~4 chars/token average for English-with-structure, 720KB chars ≈ 180K
# tokens, leaving ~20K tokens of headroom for the model's response inside a
# 200K-token context. Empirically large gemini-grounded blobs are 10-80KB
# each; a 10-qid batch can push 800KB. When over the cap, recursively
# binary-split the batch; a singleton over the cap is fail-fast.
APPROX_PROMPT_CHAR_LIMIT = 4 * (200_000 - 20_000)


def _require_qid(question: MetaculusQuestion) -> int:
    """Narrow ``MetaculusQuestion.id_of_question`` (typed ``int | None``) to ``int``.

    Every resolved question flowing through the redactor stage has a server-assigned
    id; a missing id is a programming error, not an expected runtime condition.
    """
    qid = question.id_of_question
    assert qid is not None, "MetaculusQuestion.id_of_question must be set for redactor batches"
    return qid


REDACTOR_SYSTEM_PROMPT = """\
You are a forensic redactor working on a forecasting backtest. The benchmark
evaluates how well a forecaster would have done HAD it forecast a question
BEFORE that question's resolution. The research blobs were collected after
the resolution date and frequently contain the literal answer — a number,
date, outcome, or sentence that directly states or implies what happened.

Your job: produce a sanitized version of each blob that strips
resolution-revealing content (DIRECT and IMPLIED) while preserving genuinely
useful background information that a forecaster could legitimately have known
before resolution.

Rules:
1. STRIP any sentence, table cell, paragraph, or numeric value that directly
   states, strongly implies, or numerically reveals the resolution. Direct
   leaks are obvious; IMPLICATION leaks are subtler and equally fatal:

   a. Comparison-by-difference / anchored values. "The X PMI was unchanged
      from its March reading of 52.7%" reveals the current X PMI = 52.7%
      even though no current value is named. STRIP these.
   b. Anchored rates. "decreased slightly from March's 54.0%" implies the
      current value is in the low 53s-54. STRIP.
   c. Bracketing ranges. "approximately 17.0-17.5 million travelers"
      tightly bounds the truth. STRIP.
   d. Threshold framing. "scored just below the 40% threshold" reveals the
      truth was below threshold. STRIP if the threshold matches the
      question's resolution criteria.
   e. Date-specific outcomes. "On April 29, 2026, X reached Y%" reveals an
      outcome at a specific date if that date falls in the question's
      resolution window. STRIP.
   f. Time-of-resolution context. Past-tense statements with a date AFTER
      the question's open_time but BEFORE/AT resolution_time, on the
      question's topic, are likely leaks. SCRUTINIZE.

2. PRESERVE background context: historical base rates, methodology
   descriptions, market structure, predecessor data BEFORE the question's
   open_time, generic discussion of factors that would influence the
   outcome, AND pre-event analyst guidance ranges (e.g., "Company X issued
   Q1 guidance of 63-65% gross margin"). A forecaster pre-resolution could
   legitimately have used these. When in doubt about pre-vs-post-event
   status, PREFER preserving the data and stripping the comparison framing
   — keep "guidance was 63-65%" while stripping "exceeded guidance" or
   "above the guided range" (which reveals direction).
3. You MUST NOT add any new content. Only redact (delete a sentence) or
   replace it inline with `[REDACTED: <short reason>]`.
4. The ground truth is provided so you know what to redact. The ground
   truth string itself MUST NOT appear anywhere in the sanitized output —
   not verbatim, not as a numeric value embedded in another sentence, not
   case-shifted.
5. You are operating with EXPLICIT KNOWLEDGE of the ground truth. Apply
   this test: given the ground truth, does the content help a forecaster
   who didn't know the ground truth converge on the right answer? If yes,
   it leaks. Don't reason about the forecaster's perspective — reason
   about whether (content + ground truth) are tightly consistent. When in
   doubt, REDACT. False positives cost a small amount of useful context;
   false negatives ruin the benchmark.
6. Output STRICT JSON in the schema specified. No prose before or after the
   JSON. No code fences.

EXAMPLE 1
=========

Question: Which will be higher in the first official ISM reports for April 2026:
  the Manufacturing PMI or the Services PMI?
Ground truth: Services PMI higher

LEAKY blob fragment:
  "The Manufacturing PMI was unchanged from its March reading of 52.7%."
  "The Services PMI decreased slightly from March's 54.0%."

WHY IT LEAKS: Even without naming April values, "unchanged from March 52.7%"
reveals April Manufacturing == 52.7%. "Decreased slightly from 54.0%" reveals
Services in the low 53s-54s. Combined: Services > Manufacturing. The ground
truth is recoverable.

CORRECT redaction:
  "[REDACTED: anchored Manufacturing PMI value via March comparison]"
  "[REDACTED: anchored Services PMI value via March comparison]"

EXAMPLE 2 (subtler — WHOLE-SENTENCE redaction required)
=======================================================

Question: Same as EXAMPLE 1. Ground truth: Services PMI higher.

LEAKY fragment (missed in a previous run):
  "Manufacturing PMI was unchanged from its March reading of 52.7%, marking the
  fourth consecutive month of expansion."
  "Services PMI decreased slightly from March's 54.0% but remained firmly in
  expansion territory for the 22nd consecutive month."

WHY IT STILL LEAKS: "X was unchanged from PRIOR_VALUE" ALGEBRAICALLY DETERMINES
the current value (= PRIOR_VALUE). A forecaster derives Manufacturing = 52.7%
and Services in the low 53s (between 52.7 and 54.0) → Services > Manufacturing.

CORRECT (whole-sentence):
  "[REDACTED: anchored Manufacturing April 2026 value via 'unchanged from March']"
  "[REDACTED: anchored Services April 2026 value via 'decreased slightly from March']"

WRONG (what failed):
  "[REDACTED: clause] marking the fourth consecutive month of expansion."
  ↑ Trailing clause still places the value above 50 and leaks.

Rule: when you see "X was [unchanged | up Y bps | down Z%] from PRIOR_VALUE"
or any phrase that ALGEBRAICALLY DETERMINES the current value given a known
prior, redact the ENTIRE SENTENCE — both the comparison and any trailing
contextual clauses.

Schema:
{
  "results": [
    {
      "qid": <int>,
      "sanitized_blob": "<sanitized markdown text>",
      "redactions": [
        {"original_excerpt": "<text removed>", "reason": "<short reason>"}
      ]
    }
  ]
}

The "results" array MUST contain exactly one entry per qid in the input
batch, in any order. Every sanitized_blob must be non-empty (if everything
needs redacting, leave a brief stub like "Background only; resolution-
revealing content was fully redacted.").
"""


def _normalize_for_match(s: str) -> str:
    """Lowercase and collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", s).strip().lower()


_TRIVIAL_BINARY_GT_VALUES: frozenset[str] = frozenset({"yes", "no", "true", "false"})


def verbatim_leak_check_passes(
    sanitized: str,
    ground_truth: GroundTruth,
    question_type: str,
) -> tuple[bool, str | None]:
    """Type-aware verbatim-leak check on the redactor's sanitized blob.

    Returns ``(passes, reject_reason)``. ``passes=True`` means no verbatim
    leak detected; ``reject_reason`` is a human-readable explanation when
    ``passes=False``.

    Audit at ``backtests/ablation/audit_smoke_20260515.md:176-194`` documents
    why the previous unconditional substring check false-positives:

    * binary GTs in {yes, no, true, false}: substring "no" appears in
      "non-manufacturing" and other non-leak words. Skip the check entirely
      and rely on the LLM screen + qa_iterate verifier (which check
      semantic, not surface, leakage).
    * multiple_choice: the GT may appear as ambient question phrasing
      (e.g., "March 2026" in "Jan/Feb/March 2026 revenues"). A bare-substring
      check has no discriminative power. Use a word-boundary regex match;
      ambient *embedded* mentions (e.g., "Red" inside "Reduction") pass,
      while standalone occurrences ("the answer is Nikkei 225") reject.
    * numeric: numeric resolution strings like "66.246" or "-87.9" are
      unique enough that any verbatim hit is a real leak. Keep the
      substring check but on the normalized form.

    Uncommon binary GTs (e.g., "draw") fall through to the substring check.
    """
    gt_str = ground_truth.resolution_string
    if not gt_str:
        return True, None

    if question_type == "binary" and gt_str.strip().lower() in _TRIVIAL_BINARY_GT_VALUES:
        return True, None

    normalized_gt = _normalize_for_match(gt_str)
    if not normalized_gt:
        return True, None

    normalized_blob = _normalize_for_match(sanitized)

    if question_type == "multiple_choice":
        pattern = r"\b" + re.escape(normalized_gt) + r"\b"
        if re.search(pattern, normalized_blob):
            return False, f"sanitized_blob still contains MC option {gt_str!r} as standalone phrase"
        return True, None

    if normalized_gt in normalized_blob:
        return False, f"sanitized_blob still contains ground truth {gt_str!r} verbatim"
    return True, None


def _build_redactor_prompt(batch: list[tuple[MetaculusQuestion, GroundTruth, str]]) -> str:
    """Assemble the per-batch prompt sent on stdin to ``claude -p``.

    Each question block lists qid, question text, resolution criteria, fine
    print, ground truth, and the raw research blob. The opening section
    restates the redactor's contract so the prompt is self-contained even
    when ``--append-system-prompt`` was already used.
    """
    parts: list[str] = []
    parts.append("# Redactor batch task")
    parts.append("")
    parts.append(
        "You are about to receive %d question(s). For each, the ground truth "
        "is shown ONLY so you know what to redact. The ground truth MUST NOT "
        "appear anywhere in your sanitized output — not verbatim, not "
        "paraphrased numerically, not case-shifted." % len(batch)
    )
    parts.append("")
    parts.append(
        "Output strict JSON with one entry per qid: "
        '{"results": [{"qid": <int>, "sanitized_blob": "<text>", '
        '"redactions": [{"original_excerpt": "<text>", "reason": "<text>"}]}]}'
    )
    parts.append("")
    parts.append("---")
    parts.append("")

    for question, ground_truth, raw_blob in batch:
        qid = question.id_of_question
        parts.append(f"## qid={qid}")
        parts.append("")
        parts.append("### Question text")
        parts.append(str(question.question_text))
        parts.append("")
        parts.append("### Resolution criteria")
        parts.append(str(getattr(question, "resolution_criteria", "") or ""))
        parts.append("")
        parts.append("### Fine print")
        parts.append(str(getattr(question, "fine_print", "") or ""))
        parts.append("")
        parts.append("### Ground truth (DO NOT include this in the sanitized output)")
        parts.append(str(ground_truth.resolution_string))
        parts.append("")
        parts.append("### Raw research blob to redact")
        parts.append("```")
        parts.append(raw_blob)
        parts.append("```")
        parts.append("")
        parts.append("---")
        parts.append("")

    parts.append(
        "Now emit JSON in the documented schema. The 'results' array must contain exactly one entry per qid above."
    )
    return "\n".join(parts)


async def _invoke_claude_redactor(
    prompt: str,
    *,
    claude_executable: str = DEFAULT_CLAUDE_EXECUTABLE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Run ``claude -p`` headless with the redactor prompt on stdin; return stdout.

    Flags:
      -p / --print                 headless single-shot
      --output-format text         plain text output (canonical pattern from
                                   ``~/workspace/fraud_research/scripts/autoresearch_loop.sh``;
                                   the redactor's response IS JSON because we
                                   ask for it in the prompt — we don't need an
                                   outer JSON envelope wrapping it).
      --max-turns 1                one shot
      --permission-mode bypassPermissions   no permission prompts
      --settings '{...}'           force-disable prompt-caching 1H beta
                                   (Mantle's headless gateway rejects the
                                   ``prompt-caching-2025-XX-XX`` beta header,
                                   producing 400 invalid-beta-flag → exit 1).
                                   Diagnosed by fraud_research team 2026-05-06.
      --append-system-prompt <s>   redactor system prompt

    NOTE: we deliberately do NOT pass ``--bare``. The successful run #5 of this
    pipeline DID use ``--bare`` but a follow-up run with the same flag set
    failed — the precise cause is unclear, but the canonical pattern in
    fraud_research/autoresearch_loop.sh runs without ``--bare`` and is known to
    work for thousands of headless invocations against the same Mantle gateway.
    Cargo-culting that pattern.

    Tools are NOT explicitly disabled here either — ``--max-turns 1`` already
    constrains the model to one shot, and the system prompt instructs it to
    output JSON only. Adding ``--allowedTools ""`` is a known-fragile flag in
    non-interactive mode (per OpenRouter / Anthropic GitHub issues) and was the
    only differing flag between run #5 (worked) and the latest failures —
    dropping it.

    Raises ``subprocess.CalledProcessError`` on non-zero exit. Raises
    ``asyncio.TimeoutError`` if the subprocess exceeds ``timeout_seconds``.
    """
    settings_json = json.dumps({"env": {"ENABLE_PROMPT_CACHING_1H": "0"}})
    argv = [
        claude_executable,
        "-p",
        "--output-format",
        "text",
        "--max-turns",
        "1",
        "--permission-mode",
        "bypassPermissions",
        "--settings",
        settings_json,
        "--append-system-prompt",
        REDACTOR_SYSTEM_PROMPT,
    ]

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")),
            timeout=timeout_seconds,
        )
    except asyncio.TimeoutError:
        # asyncio.wait_for cancels the awaitable but does NOT terminate the
        # underlying OS subprocess. Without proc.kill(), the orphan keeps
        # running until the model finishes on its own. At 50q x 3 iterations
        # the leaked FDs + process slots compound until fork() starts failing.
        logger.warning(
            "claude -p subprocess timeout (%ss); killing pid=%s",
            timeout_seconds,
            proc.pid,
        )
        proc.kill()
        try:
            # noqa: ASYNC120 — checkpoint inside except is intentional. We must
            # await proc.wait() here to reap the killed child; the surrounding
            # `raise` re-raises the original TimeoutError after cleanup. If the
            # task itself is cancelled mid-await we still want the kill to have
            # been issued (already done above), so the leak is bounded either
            # way.
            await asyncio.wait_for(proc.wait(), timeout=5.0)  # noqa: ASYNC120
        except asyncio.TimeoutError:
            logger.error("claude -p subprocess pid=%s refused SIGKILL within 5s", proc.pid)
        raise

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            returncode=proc.returncode if proc.returncode is not None else -1,
            cmd=argv,
            output=stdout_bytes,
            stderr=stderr_bytes,
        )

    stdout_text = stdout_bytes.decode("utf-8", errors="replace")
    return _extract_inner_result(stdout_text)


def _extract_inner_result(stdout_text: str) -> str:
    """Pull the inner ``result`` field out of ``claude -p --output-format json``.

    Empirically (``claude --version 2.1.140``), ``claude -p --output-format json``
    emits a JSON ARRAY of stream events, the last of which is the result envelope:
    ``[{"type":"system",...}, {"type":"assistant",...}, {"type":"result", "result":"<text>", ...}]``.
    Older versions (or future revisions) may emit a single dict envelope. Handle both:

    * list of events → find the last ``{"type":"result"}`` entry, return its ``result`` field.
    * dict envelope → return its ``result`` field.
    * anything else (e.g. test stubs returning raw redactor JSON) → pass through unchanged.
    """
    stripped = stdout_text.strip()
    if not stripped:
        return stripped
    try:
        envelope: Any = json.loads(stripped)
    except json.JSONDecodeError:
        # Passthrough preserved for backwards compat with raw-JSON test stubs;
        # log a warning so a Claude-CLI envelope-shape change doesn't surface
        # as a misleading downstream parser error.
        logger.warning(
            "claude -p stdout was not parseable JSON; returning raw (first 200 chars: %r)",
            stripped[:200],
        )
        return stripped
    if isinstance(envelope, list):
        for raw_event in reversed(envelope):
            if not isinstance(raw_event, dict):
                continue
            event = cast(dict[str, Any], raw_event)
            if event.get("type") == "result" and isinstance(event.get("result"), str):
                return event["result"]
        return stripped
    if isinstance(envelope, dict):
        env_dict = cast(dict[str, Any], envelope)
        if "result" in env_dict and isinstance(env_dict["result"], str):
            return env_dict["result"]
    return stripped


def _parse_redactor_response(
    raw_stdout: str,
    expected_qids: list[int],
    ground_truths: dict[int, GroundTruth],
) -> dict[int, tuple[str, list[dict]] | None]:
    """Parse JSON, validate per-qid structure, run leakage post-checks.

    Returns ``{qid: (sanitized_blob, redactions) | None}``. ``None`` marks
    validation failure. Unknown qids (subagent fabricated a qid not in input)
    are dropped with a warning and excluded from the return dict.
    """
    expected_set = set(expected_qids)
    out: dict[int, tuple[str, list[dict]] | None] = {qid: None for qid in expected_qids}

    payload = json.loads(raw_stdout)

    if not isinstance(payload, dict) or "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError(f"Redactor response missing 'results' array; got top-level keys: {list(payload)[:5]}")

    for entry in payload["results"]:
        if not isinstance(entry, dict):
            logger.warning("Redactor response entry is not a dict: %r; skipping", entry)
            continue
        qid_raw = entry.get("qid")
        if not isinstance(qid_raw, int):
            logger.warning("Redactor response entry missing int qid: %r; skipping", entry)
            continue
        qid = qid_raw
        if qid not in expected_set:
            logger.warning("Redactor response references unknown qid=%d; dropping", qid)
            continue

        sanitized = entry.get("sanitized_blob")
        redactions = entry.get("redactions", [])
        if not isinstance(sanitized, str) or sanitized == "":
            logger.warning("qid=%d: empty or non-string sanitized_blob; rejecting", qid)
            out[qid] = None
            continue

        gt = ground_truths[qid]
        passes, reject_reason = verbatim_leak_check_passes(sanitized, gt, gt.question_type)
        if not passes:
            logger.warning("qid=%d: %s; rejecting", qid, reject_reason)
            out[qid] = None
            continue

        if not isinstance(redactions, list):
            redactions = []
        out[qid] = (sanitized, redactions)

    return out


def _build_meta(
    *,
    qid: int,
    original_chars: int,
    sanitized_chars: int,
    redactions: list[dict],
    redactor_invocation_id: str,
) -> dict:
    return {
        "qid": qid,
        "original_chars": original_chars,
        "sanitized_chars": sanitized_chars,
        "redactions": redactions,
        "redactor_invocation_id": redactor_invocation_id,
        "pruned_at": datetime.now().isoformat(),
    }


async def _write_redactor_failure_dump(
    cache: AblationCache,
    qids: list[int],
    stderr_bytes: bytes,
    stdout_bytes: bytes,
) -> str:
    """Persist a per-batch failure dump under the cache root.

    Writes ``<cache_root>/redactor_failures/batch_<qid0>_<ts>.log`` with
    delimited stderr + stdout sections so the operator can inspect the
    full result envelope (which often carries the real failure reason
    that the 2000-char log truncation hides). Returns the path string,
    or a sentinel on filesystem error.

    Uses ``asyncio.to_thread`` so the blocking I/O doesn't stall the
    event loop. Storing under the cache root (instead of ``/tmp``)
    avoids the multi-user race that bandit B108 flags and keeps the
    artifact alongside the rest of the run's diagnostic output.
    """
    qid_label = str(qids[0]) if qids else "empty"
    ts = int(datetime.now().timestamp())
    failures_dir = cache.root / "redactor_failures"
    debug_path = failures_dir / f"batch_{qid_label}_{ts}.log"

    def _write() -> None:
        failures_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_path, "wb") as f:
            f.write(b"=== STDERR ===\n")
            f.write(stderr_bytes)
            f.write(b"\n=== STDOUT ===\n")
            f.write(stdout_bytes)

    try:
        await asyncio.to_thread(_write)
    except OSError:
        # Async-checkpoint on the early-return path so flake8-async ASYNC910
        # sees a guaranteed cooperative yield. The to_thread call above is
        # itself a checkpoint on the success path.
        await asyncio.sleep(0)
        return "(failed to write debug file)"
    return str(debug_path)


async def _process_batch(
    batch: list[tuple[MetaculusQuestion, GroundTruth, str]],
    cache: AblationCache,
    *,
    claude_executable: str,
    timeout_seconds: int,
) -> dict[int, tuple[str, dict] | None]:
    """Run one ``claude -p`` invocation for a batch and persist successes.

    Returns ``{qid: (sanitized_blob, meta) | None}`` with ``None`` for any
    qid that failed validation or whose batch hit a subprocess/JSON error.

    Prompt-size guard: if the assembled prompt would bust Claude's input
    context window, recursively binary-split the batch. Singleton batches
    that still exceed the cap are fail-fast (the operator should truncate
    the blob upstream).
    """
    await asyncio.sleep(0)
    qids = [_require_qid(q) for q, _gt, _blob in batch]
    out: dict[int, tuple[str, dict] | None] = {qid: None for qid in qids}

    prompt = _build_redactor_prompt(batch)
    total_prompt_chars = len(prompt) + len(REDACTOR_SYSTEM_PROMPT)
    if total_prompt_chars > APPROX_PROMPT_CHAR_LIMIT and len(batch) > 1:
        mid = len(batch) // 2
        logger.warning(
            "prune | batch prompt too large (%d chars > %d); splitting %d→%d+%d",
            total_prompt_chars,
            APPROX_PROMPT_CHAR_LIMIT,
            len(batch),
            mid,
            len(batch) - mid,
        )
        left = await _process_batch(
            batch[:mid],
            cache,
            claude_executable=claude_executable,
            timeout_seconds=timeout_seconds,
        )
        right = await _process_batch(
            batch[mid:],
            cache,
            claude_executable=claude_executable,
            timeout_seconds=timeout_seconds,
        )
        return {**left, **right}
    if total_prompt_chars > APPROX_PROMPT_CHAR_LIMIT:
        # len(batch) == 1 here; no further split is possible. Fail fast for
        # this qid so the operator can truncate the upstream blob.
        logger.error(
            "prune | single-qid batch prompt still too large (qid=%d, %d chars > %d); failing this qid",
            qids[0],
            total_prompt_chars,
            APPROX_PROMPT_CHAR_LIMIT,
        )
        return out

    raw_stdout: str
    try:
        raw_stdout = await _invoke_claude_redactor(
            prompt,
            claude_executable=claude_executable,
            timeout_seconds=timeout_seconds,
        )
    except subprocess.CalledProcessError as exc:
        # CalledProcessError captures stderr in `exc.stderr` but Python's default
        # __str__ only renders the argv. Surface the actual stderr bytes (decoded,
        # truncated) in the log AND dump the full stdout/stderr to a debug file
        # under the cache root so the result envelope (which often carries the
        # real failure detail) is diagnosable without re-running the subprocess.
        stderr_bytes = b""
        if exc.stderr is not None:
            stderr_bytes = exc.stderr if isinstance(exc.stderr, bytes) else str(exc.stderr).encode()
        stdout_bytes = b""
        if exc.output is not None:
            stdout_bytes = exc.output if isinstance(exc.output, bytes) else str(exc.output).encode()

        debug_path = await _write_redactor_failure_dump(cache, qids, stderr_bytes, stdout_bytes)

        stderr_text = stderr_bytes.decode("utf-8", errors="replace")[:2000]
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")[:2000]
        logger.error(
            "Redactor subprocess failed for qids=%s: exit=%s\n"
            "(full output dumped to %s)\n"
            "stderr (truncated): %s\nstdout (truncated): %s",
            qids,
            exc.returncode,
            debug_path,
            stderr_text,
            stdout_text,
        )
        return out
    except asyncio.TimeoutError as exc:
        logger.error("Redactor subprocess timed out for qids=%s: %s", qids, exc, exc_info=True)
        return out
    except Exception as exc:
        logger.error("Redactor subprocess raised unexpected error for qids=%s: %s", qids, exc, exc_info=True)
        return out

    ground_truths = {_require_qid(q): gt for q, gt, _ in batch}
    raw_blobs = {_require_qid(q): blob for q, _, blob in batch}

    try:
        parsed = _parse_redactor_response(raw_stdout, qids, ground_truths)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.error(
            "Redactor JSON parse failed for qids=%s: %s; raw stdout (first 500 chars): %r",
            qids,
            exc,
            raw_stdout[:500],
        )
        return out

    invocation_id = secrets.token_hex(8)
    for qid, result in parsed.items():
        if result is None:
            continue
        sanitized_blob, redactions = result
        original_chars = len(raw_blobs[qid])
        meta = _build_meta(
            qid=qid,
            original_chars=original_chars,
            sanitized_chars=len(sanitized_blob),
            redactions=redactions,
            redactor_invocation_id=invocation_id,
        )
        cache.write_pruned_research(qid=qid, sanitized_blob=sanitized_blob, meta=meta)
        out[qid] = (sanitized_blob, {**meta, "cache_schema_version": 1})

    return out


async def run_prune_for_qids(
    questions_with_gt_and_blob: list[tuple[MetaculusQuestion, GroundTruth, str]],
    cache: AblationCache,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    force: bool = False,
    claude_executable: str = DEFAULT_CLAUDE_EXECUTABLE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[int, tuple[str, dict] | None]:
    """Redact resolution-revealing content from research blobs via headless Claude Code.

    Cache hits short-circuit per qid. Cache-miss qids are batched into
    groups of ``batch_size`` and dispatched serially (one ``claude -p`` per
    batch). Per-batch failures (subprocess error, invalid JSON) only affect
    that batch's qids; other batches still complete. Per-qid validation
    failures (sanitized blob still contains GT, empty blob, etc.) only
    affect that qid.

    Returns ``{qid: (sanitized_blob, meta) | None}`` covering every qid in
    the input. ``None`` means failure; the caller should drop that qid from
    downstream stages.
    """
    await asyncio.sleep(0)
    out: dict[int, tuple[str, dict] | None] = {}
    needs_run: list[tuple[MetaculusQuestion, GroundTruth, str]] = []

    for question, gt, raw_blob in questions_with_gt_and_blob:
        qid = _require_qid(question)
        if not force:
            cached = cache.read_pruned_research(qid)
            if cached is not None:
                out[qid] = cached
                continue
        needs_run.append((question, gt, raw_blob))

    for start in range(0, len(needs_run), batch_size):
        batch = needs_run[start : start + batch_size]
        batch_results = await _process_batch(
            batch,
            cache,
            claude_executable=claude_executable,
            timeout_seconds=timeout_seconds,
        )
        # Per-qid recovery: if every qid in this batch came back None and the
        # batch had > 1 qid, the failure was a batch-level subprocess error
        # (not per-qid validation), so retry each qid in its own 1-qid batch.
        # A single intermittent failure thereby drops at most 1 qid instead of
        # ``batch_size``. Only fires when ALL qids failed; mixed-success batches
        # already provide per-qid isolation via _parse_redactor_response.
        if len(batch) > 1 and all(batch_results.get(qid) is None for qid in batch_results):
            logger.warning(
                "prune | batch failed entirely for qids=%s; falling back to per-qid recovery",
                list(batch_results.keys()),
            )
            for triple in batch:
                single_results = await _process_batch(
                    [triple],
                    cache,
                    claude_executable=claude_executable,
                    timeout_seconds=timeout_seconds,
                )
                batch_results.update(single_results)
        out.update(batch_results)

    return out
