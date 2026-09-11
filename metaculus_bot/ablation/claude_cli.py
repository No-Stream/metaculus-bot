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

# An enumerated deny list, not the empty --allowedTools whose parse shape correlated with past run failures.
_DISALLOWED_TOOLS = (
    "Bash,Edit,Write,MultiEdit,NotebookEdit,Read,Grep,Glob,LS,WebFetch,WebSearch,Task,Agent,TodoWrite,KillShell"
)


def _settings_payload() -> str:
    return json.dumps({"env": {"ENABLE_PROMPT_CACHING_1H": "0"}})


def _build_argv(system_prompt: str, *, claude_executable: str = DEFAULT_CLAUDE_EXECUTABLE) -> list[str]:
    """Assemble the ``claude -p`` argv for a single headless invocation.

    Three of the flags carry receipts. ``--settings`` force-disables the 1H
    prompt-caching beta, whose header the headless gateway rejects with a 400
    invalid-beta-flag and exit 1 (diagnosed 2026-05-06). ``--bare`` is absent on
    purpose: one redactor run succeeded with it and the next failed on the same
    flag, while the sibling headless harness that runs thousands of invocations
    against this gateway passes it never. ``--disallowedTools`` denies every tool
    with the permission mode left at the headless default, hardened 2026-08-27
    after a security review: the prompt embeds web-derived research text, so an
    injection steering one tool call was a local-execution risk under the old
    ``--permission-mode bypassPermissions``. The empty ``--allowedTools ""``
    shape stays avoided, being the one correlated with run failures.
    """
    return [
        claude_executable,
        "-p",
        "--output-format",
        "text",
        "--max-turns",
        "1",
        "--disallowedTools",
        _DISALLOWED_TOOLS,
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
        # wait_for cancels the await, not the child: at 50 questions x 3 iterations the orphans exhaust fork().
        logger.warning(
            "claude -p subprocess timeout (%ss); killing pid=%s",
            timeout_seconds,
            proc.pid,
        )
        proc.kill()
        try:
            # Reap the killed child, bounded so one refusing SIGKILL cannot pin the run.
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


def excerpt(text: str, *, limit: int) -> str:
    """The head of ``text`` for a log line or error message; a subagent's blob runs to megabytes."""
    return text[:limit]


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
        # Passthrough kept for raw-JSON test stubs; the warning keeps a CLI envelope change from reading as a parser error.
        logger.warning(
            "claude -p stdout was not parseable JSON; returning raw (first 200 chars: %r)",
            excerpt(stripped, limit=200),
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
