"""Iterate-until-clean QA harness for the ablation benchmark.

The ``qa_iterate`` stage runs between ``screen`` and ``forecast``. For each qid
it spawns a verifier ``claude -p`` subprocess that scores the sanitized blob
on three axes (leakage_risk, forecastability, hallucination_risk). When the
verifier flags leakage above threshold OR the screen disagrees, a re-redactor
``claude -p`` subprocess does a second-pass redaction informed by the
verifier's notes. The loop runs up to ``max_iterations`` (default 3); after
that the qid auto-rejects with an entry in ``manual_rejects.json``. After
leakage acceptance, low-forecastability blobs auto-reject with a different
reason — this catches the "research empty → forecaster hallucinates" failure
mode that the leakage screen alone cannot.

Verifier independence: the verifier uses the SAME headless Claude binary as
the redactor (sonnet-class) but a DIFFERENT system prompt that frames it as
an independent auditor. Two prompts, one model — the prompt framing supplies
the independence.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from forecasting_tools import MetaculusQuestion

from metaculus_bot.ablation.cache import AblationCache, atomic_write_text
from metaculus_bot.ablation.prune import verbatim_leak_check_passes
from metaculus_bot.backtest.scoring import GroundTruth

__all__ = [
    "DEFAULT_LEAKAGE_THRESHOLD",
    "DEFAULT_FORECASTABILITY_THRESHOLD",
    "DEFAULT_MAX_ITERATIONS",
    "VERIFIER_SYSTEM_PROMPT",
    "RE_REDACTOR_SYSTEM_PROMPT",
    "VerifierScore",
    "IterateOutcome",
    "run_qa_iterate_for_qid",
    "run_qa_iterate_batch",
    "read_manual_rejects",
    "write_manual_rejects",
    "render_qa_summary",
    "serialize_outcome",
]

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_MAX_ITERATIONS = 3
DEFAULT_LEAKAGE_THRESHOLD = 0.3
# Score values BELOW (strict less-than) DEFAULT_FORECASTABILITY_THRESHOLD
# reject as low_forecastability; values AT or ABOVE the threshold pass.
# qid 43151 in the smoke run scored exactly 0.20 on iter 2 and passed —
# the strict-less-than convention is intentional, not an off-by-one. Pick
# the threshold so this alignment is what you want.
DEFAULT_FORECASTABILITY_THRESHOLD = 0.2
DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_CLAUDE_EXECUTABLE = "claude"

FinalStatus = Literal["clean", "rejected_leakage", "rejected_forecastability"]


VERIFIER_SYSTEM_PROMPT = """\
You are an independent forecasting-benchmark auditor. Your job: judge whether
a sanitized research blob is safe and useful for forecasting a resolved
question, treating the research as if it were collected BEFORE the question's
resolution.

For EACH question, score three independent properties on the closed interval
[0.0, 1.0]:

1. leakage_risk (0=no leakage detectable, 1=blob clearly reveals resolution):
   - DIRECT: blob states the answer numerically or by name
   - IMPLIED: anchored values, comparison-by-difference, bracketing ranges,
     post-resolution citations, references to events that only happened after
     the resolution date
   - PHRASING leaks: questions in research that wouldn't be asked unless the
     answer were known

2. forecastability (0=cannot forecast at all from this blob, 1=blob is
   substantive and forecast-ready):
   - Does the blob contain ANY useful background information?
   - Or does it just say "no data available" / "not provided"?
   - For numeric questions: are there ANY anchor values, ranges, prior trends?

3. hallucination_risk (0=blob is clearly grounded, 1=blob's emptiness invites
   forecasters to fabricate):
   - High when the blob is short, vague, or evasive
   - High when the question requires specific data the blob doesn't have

Return STRICT JSON: { "verdicts": [ { "qid": <int>, "leakage_risk": <float>,
"forecastability": <float>, "hallucination_risk": <float>, "notes": <string,
1-3 sentences> } ] }

No prose before or after the JSON. No code fences.
"""


RE_REDACTOR_SYSTEM_PROMPT = """\
You are a forensic redactor on a forecasting benchmark, doing a SECOND-PASS
redaction. The previous redaction left leakage cues that the verifier
flagged. You will receive the verifier's notes as part of the per-question
brief.

Re-redact the blob to remove the cues the verifier flagged PLUS any others
you spot. Be aggressive: prefer dropping a paragraph over keeping it if you
are unsure. The ground truth is provided ONLY so you know what to redact;
the ground truth string MUST NOT appear anywhere in your sanitized output —
not verbatim, not as a numeric value embedded in another sentence, not
case-shifted.

Output STRICT JSON: { "results": [ { "qid": <int>, "sanitized_blob":
<string>, "redactions": [{"original_excerpt": <string>, "reason": <string>}]
} ] }

No prose before or after the JSON. No code fences.
"""


@dataclass(frozen=True)
class VerifierScore:
    iteration: int
    leakage_risk: float
    forecastability: float
    hallucination_risk: float
    notes: str = ""


@dataclass
class IterateOutcome:
    qid: int
    final_status: FinalStatus
    iterations: int
    final_blob_path: Path | None
    verifier_scores: list[VerifierScore] = field(default_factory=list)
    reject_reason: str | None = None


def serialize_outcome(outcome: IterateOutcome) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "qid": outcome.qid,
        "final_status": outcome.final_status,
        "iterations": outcome.iterations,
        "final_blob_path": str(outcome.final_blob_path) if outcome.final_blob_path is not None else None,
        "verifier_scores": [asdict(s) for s in outcome.verifier_scores],
        "reject_reason": outcome.reject_reason,
    }
    return payload


# ---------------------------------------------------------------------------
# Subprocess invocation
# ---------------------------------------------------------------------------


def _settings_payload() -> str:
    return json.dumps({"env": {"ENABLE_PROMPT_CACHING_1H": "0"}})


def _build_argv(system_prompt: str, *, claude_executable: str = DEFAULT_CLAUDE_EXECUTABLE) -> list[str]:
    return [
        claude_executable,
        "-p",
        "--output-format",
        "text",
        "--max-turns",
        "1",
        "--permission-mode",
        "bypassPermissions",
        "--settings",
        _settings_payload(),
        "--append-system-prompt",
        system_prompt,
    ]


async def _run_claude_subprocess(
    argv: list[str],
    prompt: str,
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
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
            # noqa: ASYNC120 — checkpoint inside except is intentional. We
            # must await proc.wait() here to reap the killed child; the
            # outer `raise` re-raises the original TimeoutError after
            # cleanup. Bounded by an inner 5s timeout so a child that
            # refuses SIGKILL doesn't pin us.
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
    """Pull the inner ``result`` field if Claude emitted a JSON envelope.

    Mirrors ``prune._extract_inner_result``: ``--output-format text`` usually
    returns the raw model output, but some Claude builds wrap it in a stream
    of typed events. Handle both.
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


async def _invoke_verifier(
    prompt: str,
    *,
    claude_executable: str = DEFAULT_CLAUDE_EXECUTABLE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    argv = _build_argv(VERIFIER_SYSTEM_PROMPT, claude_executable=claude_executable)
    return await _run_claude_subprocess(argv, prompt, timeout_seconds=timeout_seconds)


async def _invoke_re_redactor(
    prompt: str,
    *,
    claude_executable: str = DEFAULT_CLAUDE_EXECUTABLE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    argv = _build_argv(RE_REDACTOR_SYSTEM_PROMPT, claude_executable=claude_executable)
    return await _run_claude_subprocess(argv, prompt, timeout_seconds=timeout_seconds)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _build_verifier_prompt(
    qid: int,
    question: MetaculusQuestion,
    sanitized_blob: str,
) -> str:
    parts: list[str] = []
    parts.append("# Verifier task")
    parts.append("")
    parts.append("Score the sanitized research blob on leakage_risk, forecastability, hallucination_risk.")
    parts.append("")
    parts.append(f"## qid={qid}")
    parts.append("")
    parts.append("### Question text")
    parts.append(str(question.question_text))
    parts.append("")
    parts.append("### Resolution criteria")
    parts.append(str(getattr(question, "resolution_criteria", "") or ""))
    parts.append("")
    parts.append("### Sanitized research blob")
    parts.append("```")
    parts.append(sanitized_blob)
    parts.append("```")
    parts.append("")
    parts.append('Now emit JSON: {"verdicts": [{"qid": <int>, "leakage_risk": ..., "forecastability": ...}]}')
    return "\n".join(parts)


def _build_re_redactor_prompt(
    qid: int,
    question: MetaculusQuestion,
    ground_truth: GroundTruth,
    sanitized_blob: str,
    verifier_notes: str,
) -> str:
    parts: list[str] = []
    parts.append("# Second-pass redactor task")
    parts.append("")
    parts.append("This is a SECOND-PASS redaction. The previous redaction left leakage cues.")
    parts.append("Verifier notes from the previous iteration are below — re-redact to remove them.")
    parts.append("")
    parts.append("## verifier notes")
    parts.append(verifier_notes or "(no notes provided)")
    parts.append("")
    parts.append(f"## qid={qid}")
    parts.append("")
    parts.append("### Question text")
    parts.append(str(question.question_text))
    parts.append("")
    parts.append("### Resolution criteria")
    parts.append(str(getattr(question, "resolution_criteria", "") or ""))
    parts.append("")
    parts.append("### Ground truth (DO NOT include in sanitized output)")
    parts.append(str(ground_truth.resolution_string))
    parts.append("")
    parts.append("### Current sanitized blob (still leaky per verifier)")
    parts.append("```")
    parts.append(sanitized_blob)
    parts.append("```")
    parts.append("")
    parts.append('Now emit JSON: {"results": [{"qid": <int>, "sanitized_blob": ..., "redactions": [...]}]}')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_verifier_response(raw: str, qid: int) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or "verdicts" not in payload or not isinstance(payload["verdicts"], list):
        raise ValueError(f"verifier response missing 'verdicts' list: {raw[:200]!r}")
    for entry in payload["verdicts"]:
        if not isinstance(entry, dict):
            continue
        if entry.get("qid") == qid:
            return cast(dict[str, Any], entry)
    raise ValueError(f"verifier response has no entry for qid={qid}: {raw[:200]!r}")


def _parse_re_redactor_response(raw: str, qid: int, ground_truth: GroundTruth) -> str:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or "results" not in payload or not isinstance(payload["results"], list):
        raise ValueError(f"re-redactor response missing 'results' list: {raw[:200]!r}")
    for entry in payload["results"]:
        if not isinstance(entry, dict) or entry.get("qid") != qid:
            continue
        sanitized = entry.get("sanitized_blob")
        if not isinstance(sanitized, str) or sanitized == "":
            raise ValueError(f"re-redactor produced empty sanitized_blob for qid={qid}")
        # Sanity check that the re-redactor LLM didn't leave ground truth verbatim.
        # This is a belt-and-suspenders check; the real leakage detection is the
        # verifier subagent's leakage_risk score on the next iteration. The
        # substring-only check false-positived on binary GTs ("no" inside "not"/
        # "now") and ended iteration at iter=0; the type-aware path (binary skip,
        # MC word-boundary, numeric strict-substring) lives in prune.py and is
        # shared by both stages.
        passes, reject_reason = verbatim_leak_check_passes(sanitized, ground_truth, ground_truth.question_type)
        if not passes:
            raise ValueError(f"re-redactor for qid={qid}: {reject_reason}")
        return sanitized
    raise ValueError(f"re-redactor response has no entry for qid={qid}")


# ---------------------------------------------------------------------------
# Per-qid iteration loop
# ---------------------------------------------------------------------------


def _verifier_score_from_entry(entry: dict[str, Any], iteration: int) -> VerifierScore:
    """Parse a verifier verdict entry into a VerifierScore, fail-loud on missing keys.

    Schema-drift in the verifier's JSON output (Anthropic ships a new ``claude``
    build that nudges output formatting; or the model hallucinates a slightly
    different shape) used to silently default ``forecastability=0.0`` —
    rejecting every qid as ``rejected_forecastability`` for an invented
    reason. Fail-fast with a ValueError instead; the per-qid try/except in
    ``run_qa_iterate_batch`` converts it to ``qa_iterate_failed: ValueError``
    so the operator sees the schema drift explicitly.
    """
    required = ("leakage_risk", "forecastability", "hallucination_risk")
    missing = [k for k in required if k not in entry]
    if missing:
        raise ValueError(f"verifier entry missing required keys {missing}; full entry: {entry!r}")
    return VerifierScore(
        iteration=iteration,
        leakage_risk=float(entry["leakage_risk"]),
        forecastability=float(entry["forecastability"]),
        hallucination_risk=float(entry["hallucination_risk"]),
        notes=str(entry.get("notes", "")),  # notes is legitimately optional
    )


async def run_qa_iterate_for_qid(
    *,
    qid: int,
    question: MetaculusQuestion,
    ground_truth: GroundTruth,
    current_blob: str,
    screen_verdict: dict[str, Any],
    cache: AblationCache,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    leakage_threshold: float = DEFAULT_LEAKAGE_THRESHOLD,
    forecastability_threshold: float = DEFAULT_FORECASTABILITY_THRESHOLD,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> IterateOutcome:
    """Run the verify → re-redact loop for a single qid.

    Acceptance gate (BOTH must hold):
      - screen_verdict["is_leaked"] is False
      - latest verifier leakage_risk < leakage_threshold

    On accept, check forecastability against ``forecastability_threshold``;
    below it, return ``rejected_forecastability``.

    On max-iterations exhaustion, return ``rejected_leakage``. The pre-iteration
    cached blob (if any) is snapshot at entry and restored to the cache on
    the rejected_leakage exhaustion path so that operator-recovery via
    hand-editing ``manual_rejects.json`` (the documented workflow) gets the
    cleanest known blob, not the leakiest re-redactor iteration.
    """
    await asyncio.sleep(0)
    blob = current_blob
    scores: list[VerifierScore] = []
    screen_says_clean = not screen_verdict.get("is_leaked", False)
    # Snapshot whatever's in the cache at entry. If the iteration loop hits
    # max_iterations without reaching leakage acceptance, we restore this
    # snapshot so the cache reflects the pre-iteration state. None means the
    # cache was empty at entry; the operator-recovery path won't surface a
    # blob that was never there in the first place.
    pre_iter_cache = cache.read_pruned_research(qid)

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        verifier_raw = await _invoke_verifier(
            _build_verifier_prompt(qid, question, blob),
            timeout_seconds=timeout_seconds,
        )
        entry = _parse_verifier_response(verifier_raw, qid)
        score = _verifier_score_from_entry(entry, iteration)
        scores.append(score)
        logger.info(
            "qa_iterate | qid=%d iter=%d leakage=%.2f forecastability=%.2f hallucination=%.2f",
            qid,
            iteration,
            score.leakage_risk,
            score.forecastability,
            score.hallucination_risk,
        )

        leakage_clean = score.leakage_risk < leakage_threshold and screen_says_clean
        if leakage_clean:
            if score.forecastability < forecastability_threshold:
                return IterateOutcome(
                    qid=qid,
                    final_status="rejected_forecastability",
                    iterations=iteration,
                    final_blob_path=cache._pruned_research_blob_path(qid),
                    verifier_scores=scores,
                    reject_reason="low_forecastability",
                )
            return IterateOutcome(
                qid=qid,
                final_status="clean",
                iterations=iteration,
                final_blob_path=cache._pruned_research_blob_path(qid),
                verifier_scores=scores,
                reject_reason=None,
            )

        if iteration >= max_iterations:
            break

        re_redactor_raw = await _invoke_re_redactor(
            _build_re_redactor_prompt(qid, question, ground_truth, blob, score.notes),
            timeout_seconds=timeout_seconds,
        )
        new_blob = _parse_re_redactor_response(re_redactor_raw, qid, ground_truth)
        blob = new_blob
        cache.write_pruned_research(
            qid=qid,
            sanitized_blob=blob,
            meta={
                "qid": qid,
                "original_chars": len(current_blob),
                "sanitized_chars": len(blob),
                "redactions": [],
                "redactor_invocation_id": f"qa_iterate_iter_{iteration}",
                "pruned_at": datetime.now().isoformat(),
                "qa_iterate_iteration": iteration,
            },
        )
        screen_says_clean = True  # re-redaction supersedes the screen's verdict on the old blob

    # Restore the pre-iteration snapshot to the cache. The iteration loop has
    # been overwriting research_pruned/<qid>.{md,meta.json} with each
    # re-redactor pass; on max-iter exhaustion the LATEST blob is the
    # leakiest one (verifier rejected it). Operator-recovery via
    # manual_rejects.json should pull the original baseline (or nothing, if
    # there was no baseline) instead of the leaky re-redactor output.
    if pre_iter_cache is not None:
        original_blob, original_meta = pre_iter_cache
        # Strip cache_schema_version (write_pruned_research re-injects it).
        meta_for_write = {k: v for k, v in original_meta.items() if k != "cache_schema_version"}
        cache.write_pruned_research(qid=qid, sanitized_blob=original_blob, meta=meta_for_write)

    return IterateOutcome(
        qid=qid,
        final_status="rejected_leakage",
        iterations=iteration,
        final_blob_path=None,
        verifier_scores=scores,
        reject_reason=f"leakage_unrecoverable_after_{max_iterations}_iterations",
    )


async def run_qa_iterate_batch(
    inputs: dict[int, dict[str, Any]],
    *,
    cache: AblationCache,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    leakage_threshold: float = DEFAULT_LEAKAGE_THRESHOLD,
    forecastability_threshold: float = DEFAULT_FORECASTABILITY_THRESHOLD,
    concurrency: int = 4,
) -> dict[int, IterateOutcome]:
    """Run ``run_qa_iterate_for_qid`` over a batch under a semaphore."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(qid: int, payload: dict[str, Any]) -> tuple[int, IterateOutcome]:
        async with semaphore:
            try:
                outcome = await run_qa_iterate_for_qid(
                    qid=qid,
                    question=payload["question"],
                    ground_truth=payload["ground_truth"],
                    current_blob=payload["current_blob"],
                    screen_verdict=payload["screen_verdict"],
                    cache=cache,
                    max_iterations=max_iterations,
                    leakage_threshold=leakage_threshold,
                    forecastability_threshold=forecastability_threshold,
                )
            except (KeyboardInterrupt, SystemExit, MemoryError, asyncio.CancelledError):
                # System-level resource exhaustion / operator interrupts /
                # cancellation are NOT per-qid failures. Let them propagate so
                # the run halts visibly instead of trickle-rejecting every
                # subsequent qid with the same root cause.
                raise
            except Exception as exc:
                # Conservative-drop on any per-qid failure (subprocess, timeout,
                # JSON parse). Mirrors forecasters._run_one's posture: a single
                # qid's failure must not cancel the rest of the batch via
                # asyncio.gather. The qid lands in manual_rejects.json with a
                # `qa_iterate_failed:` reason so the operator can re-evaluate.
                logger.warning(
                    "qa_iterate | qid=%d failed entirely | %s: %s",
                    qid,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                outcome = IterateOutcome(
                    qid=qid,
                    final_status="rejected_leakage",
                    iterations=0,
                    final_blob_path=None,
                    verifier_scores=[],
                    reject_reason=f"qa_iterate_failed: {type(exc).__name__}",
                )
        return qid, outcome

    tasks = [_one(qid, payload) for qid, payload in inputs.items()]
    results = await asyncio.gather(*tasks)
    return {qid: outcome for qid, outcome in results}


# ---------------------------------------------------------------------------
# manual_rejects.json
# ---------------------------------------------------------------------------


MANUAL_REJECTS_VERSION = 1


def read_manual_rejects(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    raw_text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"manual_rejects.json at {path} is malformed JSON ({exc}). "
            f"Either fix it manually or delete the file to start fresh."
        ) from exc
    on_disk_version = payload.get("version", 1)
    if on_disk_version != MANUAL_REJECTS_VERSION:
        raise ValueError(
            f"manual_rejects.json at {path} has version {on_disk_version}; "
            f"this code supports version {MANUAL_REJECTS_VERSION} only. "
            f"Either upgrade the code or migrate the file."
        )
    rejects_raw = payload.get("rejects", {})
    return {int(qid): entry for qid, entry in rejects_raw.items()}


def write_manual_rejects(outcomes: list[IterateOutcome], path: Path) -> None:
    """Append rejected outcomes to manual_rejects.json. Existing entries are preserved."""
    existing = read_manual_rejects(path)
    rejected = [o for o in outcomes if o.final_status != "clean"]
    for outcome in rejected:
        if outcome.qid in existing:
            continue
        existing[outcome.qid] = {
            "rejected_at": datetime.now().isoformat(),
            "reason": outcome.reject_reason or outcome.final_status,
            "verifier_scores": [asdict(s) for s in outcome.verifier_scores],
            "iterations": outcome.iterations,
        }
    payload = {
        "version": MANUAL_REJECTS_VERSION,
        "rejects": {str(qid): entry for qid, entry in sorted(existing.items())},
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------


def render_qa_summary(outcomes: dict[int, IterateOutcome], path: Path) -> Path:
    qids = sorted(outcomes.keys())
    n_total = len(qids)
    n_clean = sum(1 for o in outcomes.values() if o.final_status == "clean")
    n_rejected_leakage = sum(1 for o in outcomes.values() if o.final_status == "rejected_leakage")
    n_rejected_forecastability = sum(1 for o in outcomes.values() if o.final_status == "rejected_forecastability")
    total_iterations = sum(o.iterations for o in outcomes.values())
    mean_iterations = (total_iterations / n_total) if n_total > 0 else 0.0

    lines: list[str] = [
        "# QA Iterate Summary",
        "",
        f"Total qids: {n_total}",
        f"- clean: {n_clean}",
        f"- rejected_leakage: {n_rejected_leakage}",
        f"- rejected_forecastability: {n_rejected_forecastability}",
        f"- total iterations: {total_iterations}",
        f"- mean iterations per qid: {mean_iterations:.2f}",
        "",
        "## Per-qid outcomes",
        "",
    ]

    for qid in qids:
        outcome = outcomes[qid]
        lines.append(f"### Q{qid}")
        lines.append(f"- final_status: `{outcome.final_status}`")
        lines.append(f"- iterations: {outcome.iterations}")
        if outcome.reject_reason:
            lines.append(f"- reject_reason: {outcome.reject_reason}")
        if outcome.final_blob_path:
            lines.append(f"- final_blob_path: `{outcome.final_blob_path}`")
        if outcome.verifier_scores:
            lines.append("- verifier scores:")
            for score in outcome.verifier_scores:
                lines.append(
                    f"  - iter {score.iteration}: leakage={score.leakage_risk:.2f} "
                    f"forecastability={score.forecastability:.2f} "
                    f"hallucination={score.hallucination_risk:.2f}" + (f" — {score.notes}" if score.notes else "")
                )
        lines.append("")

    atomic_write_text(path, "\n".join(lines))
    return path
