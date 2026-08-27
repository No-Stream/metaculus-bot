"""Shared headless ``claude -p`` subprocess driver for the ablation harness.

Two ablation stages drive the same headless Claude Code binary: the redactor
(``ablation.prune``) and the verify → re-redact loop (``ablation.qa_iterate``).
They differ only in the system prompt and the prompt body, so the argv flag
set, the timeout/orphan-reap path, and the stdout-envelope unwrapping live
here and are maintained once. Both stages re-import these names so their
existing patch surfaces keep resolving.

This is a separate module rather than a helper on either stage because
``qa_iterate`` imports ``prune.verbatim_leak_check_passes``, so hanging the
shared driver off either one would close an import cycle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from typing import Any, cast

__all__ = [
    "DEFAULT_CLAUDE_EXECUTABLE",
    "DEFAULT_TIMEOUT_SECONDS",
]

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 600
DEFAULT_CLAUDE_EXECUTABLE = "claude"


def _settings_payload() -> str:
    return json.dumps({"env": {"ENABLE_PROMPT_CACHING_1H": "0"}})


def _build_argv(system_prompt: str, *, claude_executable: str = DEFAULT_CLAUDE_EXECUTABLE) -> list[str]:
    """Assemble the ``claude -p`` argv for a single headless invocation.

    Flags:
      -p / --print                 headless single-shot
      --output-format text         plain text output (canonical pattern from a
                                   sibling headless-Claude research harness; a
                                   stage's response IS JSON because we ask for
                                   it in the prompt — we don't need an outer
                                   JSON envelope wrapping it).
      --max-turns 1                one shot
      --permission-mode bypassPermissions   no permission prompts
      --settings '{...}'           force-disable prompt-caching 1H beta (the
                                   headless gateway rejects the
                                   ``prompt-caching-2025-XX-XX`` beta header,
                                   producing 400 invalid-beta-flag → exit 1).
                                   Diagnosed 2026-05-06.
      --append-system-prompt <s>   the calling stage's system prompt

    NOTE: we deliberately do NOT pass ``--bare``. The successful run #5 of the
    redactor pipeline DID use ``--bare`` but a follow-up run with the same flag
    set failed — the precise cause is unclear, but the canonical pattern in that
    sibling harness runs without ``--bare`` and is known to work for thousands
    of headless invocations against the same gateway. Cargo-culting that
    pattern.

    Tools are NOT explicitly disabled here either — ``--max-turns 1`` already
    constrains the model to one shot, and the system prompts instruct it to
    output JSON only. Adding ``--allowedTools ""`` is a known-fragile flag in
    non-interactive mode (per OpenRouter / Anthropic GitHub issues) and was the
    only differing flag between run #5 (worked) and the latest failures —
    dropping it.
    """
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
    """Run ``argv`` with ``prompt`` on stdin; return the unwrapped stdout text.

    Raises ``subprocess.CalledProcessError`` on non-zero exit. Raises
    ``asyncio.TimeoutError`` if the subprocess exceeds ``timeout_seconds``,
    after killing and reaping the child.
    """
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
    except TimeoutError:
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
            # must await proc.wait() here to reap the killed child; the outer
            # `raise` re-raises the original TimeoutError after cleanup. If the
            # task itself is cancelled mid-await the kill has already been
            # issued, so the leak is bounded either way. Bounded by an inner
            # 5s timeout so a child that refuses SIGKILL doesn't pin us.
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except TimeoutError:
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
    """Pull the inner ``result`` field out if Claude emitted a JSON envelope.

    Empirically (``claude --version 2.1.140``), ``claude -p`` can emit a JSON
    ARRAY of stream events, the last of which is the result envelope:
    ``[{"type":"system",...}, {"type":"assistant",...}, {"type":"result", "result":"<text>", ...}]``.
    Older versions (or future revisions) may emit a single dict envelope, and
    ``--output-format text`` usually returns the raw model output. Handle all three:

    * list of events → find the last ``{"type":"result"}`` entry, return its ``result`` field.
    * dict envelope → return its ``result`` field.
    * anything else (e.g. test stubs returning raw stage JSON) → pass through unchanged.
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
