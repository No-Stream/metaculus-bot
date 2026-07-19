from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError

from metaculus_bot.research.agentic.artifact import detachment_lint, render_findings
from metaculus_bot.research.agentic.llm import build_default_llm_call
from metaculus_bot.research.agentic.types import (
    Finding,
    GhostForecast,
    LoopConfig,
    LoopResult,
    LoopTelemetry,
    ToolOutcome,
    ToolSpec,
)
from metaculus_bot.structured_output_schema import (
    BinaryStructured,
    MultipleChoiceStructured,
    NumericStructured,
    parse_structured_block,
)

logger = logging.getLogger(__name__)

LlmCall = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], Awaitable[Any]]

_INTERNAL_TOOL_TIMEOUT_S = 5.0
_INTERNAL_TOOL_NAMES = ("record_findings", "conclude")
_NUDGE = "call conclude or use tools"


@dataclass(slots=True)
class _LoopState:
    messages: list[dict[str, Any]]
    started_at_s: float
    deadline_at_s: float
    telemetry: LoopTelemetry = field(default_factory=LoopTelemetry)
    findings: list[Finding] = field(default_factory=list)
    pending_leads: list[str] = field(default_factory=list)
    seen_tool_calls: set[tuple[str, str]] = field(default_factory=set)
    nudged_for_no_action: bool = False
    explicit_conclude: bool = False
    stop_loop: bool = False


@dataclass(slots=True)
class _ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(slots=True)
class _ToolExecutionResult:
    tool_call_id: str
    tool_name: str
    content: str
    method: str = ""


def _tool_schema(name: str, description: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _internal_tool_schemas() -> list[dict[str, Any]]:
    finding_schema = {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "source_url": {"type": "string"},
                        "quote": {"type": "string"},
                        "date": {"type": "string"},
                        "retrieved_how": {"type": "string"},
                        "topic": {"type": "string"},
                        "discrepancy": {"type": "boolean"},
                    },
                    "required": ["claim", "source_url", "quote"],
                    "additionalProperties": True,
                },
            }
        },
        "required": ["findings"],
        "additionalProperties": False,
    }
    conclude_schema = {
        "type": "object",
        "properties": {
            "pending_leads": {"type": "array", "items": {"type": "string"}},
            "final_findings": {
                "type": "array",
                "items": finding_schema["properties"]["findings"]["items"],
            },
        },
        "additionalProperties": False,
    }
    return [
        _tool_schema(
            "record_findings",
            "Bank detached findings. Claims must stay citation-only and avoid likelihood or verdict language.",
            finding_schema,
        ),
        _tool_schema(
            "conclude",
            "Finish the loop, optionally banking final findings and leaving pending leads for follow-up telemetry.",
            conclude_schema,
        ),
    ]


def _tool_schemas(tools: list[ToolSpec], must_conclude: bool) -> list[dict[str, Any]]:
    internal = _internal_tool_schemas()
    if must_conclude:
        return internal
    return internal + [_tool_schema(tool.name, tool.description, tool.parameters) for tool in tools]


def _get_field(value: Any, field: str) -> Any:
    if isinstance(value, dict):
        return value.get(field)
    return getattr(value, field, None)


def _parse_response_message(response: Any) -> dict[str, Any]:
    choices = _get_field(response, "choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices[0].message")
    message = _get_field(choices[0], "message")
    if message is None:
        raise ValueError("LLM response missing choices[0].message")

    content = _get_field(message, "content")
    tool_calls_raw = _get_field(message, "tool_calls")
    tool_calls: list[dict[str, Any]] = []
    if tool_calls_raw:
        if not isinstance(tool_calls_raw, list):
            raise ValueError("assistant tool_calls was not a list")
        tool_calls = [_normalize_tool_call(entry, index) for index, entry in enumerate(tool_calls_raw)]

    assistant: dict[str, Any] = {"role": "assistant", "content": content if isinstance(content, str) else ""}
    if tool_calls:
        assistant["tool_calls"] = tool_calls
    return assistant


def _normalize_tool_call(raw: Any, index: int) -> dict[str, Any]:
    function = _get_field(raw, "function")
    name = _get_field(function, "name")
    arguments = _get_field(function, "arguments")
    if not isinstance(name, str) or not name:
        raise ValueError(f"assistant tool call {index} missing function.name")
    call_id = _get_field(raw, "id")
    normalized: dict[str, Any] = {
        "id": call_id if isinstance(call_id, str) and call_id else f"tool_{index}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments if isinstance(arguments, str) else "{}",
        },
    }
    return normalized


def _extract_tool_calls(assistant_message: dict[str, Any]) -> list[_ToolCall]:
    raw_calls = assistant_message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return []
    return [
        _ToolCall(
            id=str(entry["id"]),
            name=str(entry["function"]["name"]),
            arguments=str(entry["function"]["arguments"]),
        )
        for entry in raw_calls
    ]


def _remaining_s(state: _LoopState, now: Callable[[], float]) -> float:
    return max(0.0, state.deadline_at_s - now())


def _must_conclude(state: _LoopState, config: LoopConfig, now: Callable[[], float]) -> bool:
    return _remaining_s(state, now) < config.conclude_threshold_s or state.telemetry.tool_calls >= config.max_tool_calls


def _truncate_content(content: str, max_chars: int) -> tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False
    clipped = content[:max_chars].rstrip()
    return f"{clipped}\n[truncated at {len(content)} chars]", True


def _budget_line(state: _LoopState, config: LoopConfig, now: Callable[[], float]) -> str:
    remaining = int(_remaining_s(state, now))
    if _must_conclude(state, config, now):
        return f"\n[budget: {remaining}s remaining — you must conclude now]"
    return f"\n[budget: {remaining}s remaining, {state.telemetry.tool_calls}/{config.max_tool_calls} tool calls used]"


def _format_tool_content(tool_name: str, outcome: ToolOutcome, max_chars: int) -> str:
    body, truncated = _truncate_content(outcome.content_markdown, max_chars)
    effective = outcome.model_copy(update={"content_markdown": body, "truncated": outcome.truncated or truncated})
    lines = [f"tool: {tool_name}", f"status: {effective.status}"]
    if effective.method:
        lines.append(f"method: {effective.method}")
    if effective.links:
        lines.append("links:")
        lines.extend(f"- {link}" for link in effective.links)
    if effective.truncated and "[truncated at " not in effective.content_markdown:
        lines.append("truncated: true")
    if effective.content_markdown:
        lines.append("")
        lines.append(effective.content_markdown)
    return "\n".join(lines)


def _parse_arguments(arguments: str) -> dict[str, Any]:
    if not arguments.strip():
        return {}
    parsed = json.loads(arguments)
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _validate_findings_payload(
    raw_findings: Any,
    *,
    label: Literal["findings", "final_findings"],
) -> tuple[list[Finding], list[str], int]:
    if raw_findings is None:
        return [], [], 0
    if not isinstance(raw_findings, list):
        return [], [f"{label} must be a list"], 0

    accepted: list[Finding] = []
    rejected: list[str] = []
    lint_rejections = 0
    for index, raw_finding in enumerate(raw_findings):
        try:
            finding = Finding.model_validate(raw_finding)
        except ValidationError as exc:
            rejected.append(f"{label}[{index}] invalid: {exc.errors()[0]['msg']}")
            continue
        violations = detachment_lint(finding)
        if violations:
            lint_rejections += 1
            rejected.append(f"{label}[{index}] rejected: {'; '.join(violations)}")
            continue
        accepted.append(finding)
    return accepted, rejected, lint_rejections


def _coerce_pending_leads(raw_pending_leads: Any) -> tuple[list[str], list[str]]:
    if raw_pending_leads is None:
        return [], []
    if not isinstance(raw_pending_leads, list):
        return [], ["pending_leads must be a list of strings"]

    pending_leads: list[str] = []
    issues: list[str] = []
    for index, item in enumerate(raw_pending_leads):
        if isinstance(item, str):
            pending_leads.append(item)
        else:
            issues.append(f"pending_leads[{index}] invalid: expected string")
    return pending_leads, issues


async def _record_findings_tool(state: _LoopState, arguments: dict[str, Any]) -> ToolOutcome:
    accepted, rejected, lint_rejections = _validate_findings_payload(arguments.get("findings"), label="findings")
    if accepted:
        state.findings.extend(accepted)
    state.telemetry.lint_rejections += lint_rejections

    lines = [f"Recorded {len(accepted)} finding(s)."]
    if rejected:
        lines.append("Rejected:")
        lines.extend(f"- {item}" for item in rejected)
    return ToolOutcome(content_markdown="\n".join(lines), method="internal")


async def _conclude_tool(state: _LoopState, arguments: dict[str, Any]) -> ToolOutcome:
    accepted, rejected, lint_rejections = _validate_findings_payload(
        arguments.get("final_findings"), label="final_findings"
    )
    pending_leads, pending_errors = _coerce_pending_leads(arguments.get("pending_leads"))
    if accepted:
        state.findings.extend(accepted)
    state.pending_leads = pending_leads
    state.telemetry.lint_rejections += lint_rejections
    state.explicit_conclude = True
    state.stop_loop = True

    lines = [f"Concluded with {len(accepted)} final finding(s) and {len(pending_leads)} pending lead(s)."]
    if rejected or pending_errors:
        lines.append("Rejected:")
        lines.extend(f"- {item}" for item in [*rejected, *pending_errors])
    return ToolOutcome(content_markdown="\n".join(lines), method="internal")


async def _execute_one_tool_call(
    tool_call: _ToolCall,
    tools_by_name: dict[str, ToolSpec],
    state: _LoopState,
    config: LoopConfig,
) -> _ToolExecutionResult:
    try:
        arguments = _parse_arguments(tool_call.arguments)
    except (json.JSONDecodeError, ValueError) as exc:
        outcome = ToolOutcome(
            content_markdown=f"Invalid tool arguments: {exc}",
            method="internal",
            status="error",
        )
        return _ToolExecutionResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            content=_format_tool_content(tool_call.name, outcome, config.max_result_chars),
        )

    if tool_call.name in _INTERNAL_TOOL_NAMES:
        timeout_s = _INTERNAL_TOOL_TIMEOUT_S
    else:
        spec = tools_by_name.get(tool_call.name)
        if spec is None:
            outcome = ToolOutcome(
                content_markdown=f"Unknown tool: {tool_call.name}",
                method="internal",
                status="error",
            )
            return _ToolExecutionResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                content=_format_tool_content(tool_call.name, outcome, config.max_result_chars),
            )
        timeout_s = spec.timeout_s

    try:
        # Instantiate the handler coroutine INSIDE the boundary. External-tool
        # handlers have concrete signatures, and async-def binds kwargs eagerly:
        # a missing/typo'd/extra key in the LLM-emitted `arguments` raises
        # TypeError at bind time, before any await. Doing the bind here means
        # that failure becomes a status="error" outcome (via the except below)
        # instead of escaping the batch gather and aborting the whole pass —
        # matching the unknown-tool path. Internal tools bind positionally and
        # can't hit this.
        if tool_call.name == "record_findings":
            handler = _record_findings_tool(state, arguments)
        elif tool_call.name == "conclude":
            handler = _conclude_tool(state, arguments)
        else:
            handler = tools_by_name[tool_call.name].handler(**arguments)
        raw_outcome = await asyncio.wait_for(handler, timeout=timeout_s)
        outcome = ToolOutcome.model_validate(raw_outcome)
    except asyncio.TimeoutError:
        outcome = ToolOutcome(
            content_markdown=f"Tool timed out after {timeout_s:.2f}s.",
            method="internal",
            status="timeout",
        )
    except ValidationError as exc:
        outcome = ToolOutcome(
            content_markdown=f"Tool returned invalid outcome: {exc.errors()[0]['msg']}",
            method="internal",
            status="error",
        )
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # tool-execution boundary: any tool failure becomes an error outcome, never a loop crash
        outcome = ToolOutcome(
            content_markdown=f"{type(exc).__name__}: {exc}",
            method="internal",
            status="error",
        )

    return _ToolExecutionResult(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=_format_tool_content(tool_call.name, outcome, config.max_result_chars),
        method=outcome.method,
    )


def _normalized_call_key(tool_call: _ToolCall) -> tuple[str, str]:
    """(tool, normalized-args) identity for exact-duplicate detection.

    JSON args are re-serialized with sorted keys so key-order shuffles still
    count as the same call; unparseable args fall back to the raw string.
    """
    try:
        normalized = json.dumps(json.loads(tool_call.arguments or "{}"), sort_keys=True)
    except (json.JSONDecodeError, ValueError):
        normalized = tool_call.arguments
    return (tool_call.name, normalized)


_DUPLICATE_CALL_WARNING = (
    "\n[note: this exact tool call was already made earlier in this run — "
    "its result will not have changed. Vary the query/URL or move on.]"
)


def _budget_rejected_content(tool_name: str, config: LoopConfig) -> str:
    outcome = ToolOutcome(
        content_markdown=(
            f"Tool call rejected: the {config.max_tool_calls}-call research budget is exhausted. "
            "No further external tool calls will run — call conclude to finish, "
            "or record_findings to bank what you already have."
        ),
        method="internal",
        status="error",
    )
    return _format_tool_content(tool_name, outcome, config.max_result_chars)


async def _execute_tool_batch(
    tool_calls: list[_ToolCall],
    *,
    tools_by_name: dict[str, ToolSpec],
    state: _LoopState,
    config: LoopConfig,
    now: Callable[[], float],
) -> None:
    duplicate_call_ids: set[str] = set()
    rejected_call_ids: set[str] = set()
    accepted: list[_ToolCall] = []

    # Clamp the batch to the remaining call slots. With parallel_tool_calls a
    # single turn can emit more calls than budget allows; without this an
    # over-budget batch executes (and bills) every external call, overshooting
    # the max_tool_calls anytime ceiling. Internal bookkeeping tools
    # (record_findings/conclude) are never rejected so the driver can always
    # bank/finish. Rejected calls are NOT counted as executed, so
    # telemetry.tool_calls stays consistent with _must_conclude's gate.
    for tool_call in tool_calls:
        is_internal = tool_call.name in _INTERNAL_TOOL_NAMES
        if not is_internal and state.telemetry.tool_calls >= config.max_tool_calls:
            rejected_call_ids.add(tool_call.id)
            continue

        state.telemetry.tool_calls += 1
        state.telemetry.per_tool_counts[tool_call.name] = state.telemetry.per_tool_counts.get(tool_call.name, 0) + 1
        call_key = _normalized_call_key(tool_call)
        if call_key in state.seen_tool_calls:
            state.telemetry.dup_tool_calls += 1
            duplicate_call_ids.add(tool_call.id)
        else:
            state.seen_tool_calls.add(call_key)
        accepted.append(tool_call)

    results = await asyncio.gather(
        *[_execute_one_tool_call(tool_call, tools_by_name, state, config) for tool_call in accepted]
    )
    for result in results:
        if result.method == "rendered":
            state.telemetry.rendered_fetches += 1
    results_by_id = {result.tool_call_id: result for result in results}

    # Exactly one tool message per tool_call_id, in the assistant's original
    # order, or the next LLM turn 400s. Rejected calls get a synthetic
    # budget-exhausted error response.
    budget_line = _budget_line(state, config, now)
    for tool_call in tool_calls:
        if tool_call.id in rejected_call_ids:
            content = _budget_rejected_content(tool_call.name, config)
        else:
            result = results_by_id[tool_call.id]
            content = result.content + (_DUPLICATE_CALL_WARNING if tool_call.id in duplicate_call_ids else "")
        state.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_call.name,
                "content": content + budget_line,
            }
        )


def _freeze_result(state: _LoopState, findings_markdown: str, ghost: GhostForecast | None) -> LoopResult:
    state.telemetry.findings_count = len(state.findings)
    state.telemetry.pending_leads_count = len(state.pending_leads)
    state.telemetry.concluded_early = state.explicit_conclude and not state.telemetry.deadline_hit
    state.telemetry.wall_s = max(0.0, state.telemetry.wall_s)
    return LoopResult(
        findings_markdown=findings_markdown,
        ghost=ghost,
        telemetry=state.telemetry,
        transcript=copy.deepcopy(state.messages),
    )


def _log_completion(state: _LoopState, log_prefix: str) -> None:
    # Marker shape per plan §6: model + per-surface counters make the run_logs
    # grep enough for the driver vibe-eval (no research-archive JSON needed).
    per_tool = state.telemetry.per_tool_counts
    searches = per_tool.get("search_news", 0) + per_tool.get("search_web", 0)
    logger.info(
        "%sGAP_FILL_V2: model=%s steps=%s tool_calls=%s searches=%s fetches=%s rendered=%s reads=%s "
        "dup_tool_calls=%s deadline_hit=%s concluded_early=%s wall_s=%.2f findings=%s "
        "pending_leads=%s lint_rejections=%s",
        log_prefix,
        state.telemetry.model,
        state.telemetry.steps,
        state.telemetry.tool_calls,
        searches,
        per_tool.get("fetch", 0),
        state.telemetry.rendered_fetches,
        per_tool.get("read_document", 0),
        state.telemetry.dup_tool_calls,
        state.telemetry.deadline_hit,
        state.explicit_conclude and not state.telemetry.deadline_hit,
        state.telemetry.wall_s,
        len(state.findings),
        len(state.pending_leads),
        state.telemetry.lint_rejections,
    )


def _summarize_ghost(raw_text: str) -> tuple[str, str]:
    for qtype in ("binary", "multiple_choice", "numeric"):
        block = parse_structured_block(raw_text, qtype)
        if isinstance(block, BinaryStructured):
            return "binary", f"posterior_prob={float(block.posterior_prob):.4f}"
        if isinstance(block, MultipleChoiceStructured):
            probs = ", ".join(f"{name}={prob:.3f}" for name, prob in sorted(block.option_probs.items()))
            return "multiple_choice", probs
        if isinstance(block, NumericStructured) and block.declared_percentiles:
            median = block.declared_percentiles.get(0.5)
            return "numeric", "" if median is None else f"median={median}"
    return "unknown", ""


async def _run_ghost_phase(
    *,
    state: _LoopState,
    ghost_prompt: str,
    llm_call: LlmCall,
    log_prefix: str,
) -> GhostForecast | None:
    state.messages.append({"role": "user", "content": ghost_prompt})
    try:
        response = await asyncio.wait_for(llm_call(state.messages, None), timeout=60.0)
        assistant_message = _parse_response_message(response)
        state.messages.append(assistant_message)
    except asyncio.TimeoutError:
        logger.warning("%sGhost phase timed out after 60s", log_prefix)
        return None
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # telemetry-only phase
        logger.warning("%sGhost phase failed: %s: %s", log_prefix, type(exc).__name__, exc)
        return None

    raw_text = assistant_message["content"]
    qtype, parsed_summary = _summarize_ghost(raw_text)
    ghost = GhostForecast(qtype=qtype, raw_text=raw_text, parsed_summary=parsed_summary)
    logger.info("%sGHOST_FORECAST: qtype=%s summary=%s", log_prefix, ghost.qtype, ghost.parsed_summary)
    return ghost


async def _run_loop_body(
    *,
    state: _LoopState,
    tools: list[ToolSpec],
    config: LoopConfig,
    llm_call: LlmCall,
    ghost_prompt: str | None,
    log_prefix: str,
    now: Callable[[], float],
) -> LoopResult:
    tools_by_name = {tool.name: tool for tool in tools}

    while not state.stop_loop and state.telemetry.steps < config.max_steps:
        tools_json = _tool_schemas(tools, _must_conclude(state, config, now))
        response = await llm_call(state.messages, tools_json)
        assistant_message = _parse_response_message(response)
        state.messages.append(assistant_message)
        state.telemetry.steps += 1

        tool_calls = _extract_tool_calls(assistant_message)
        if tool_calls:
            await _execute_tool_batch(tool_calls, tools_by_name=tools_by_name, state=state, config=config, now=now)
            continue

        if state.nudged_for_no_action:
            state.stop_loop = True
            break

        state.messages.append({"role": "user", "content": _NUDGE})
        state.nudged_for_no_action = True

    state.telemetry.wall_s = now() - state.started_at_s
    findings_markdown = render_findings(state.findings, state.pending_leads)
    ghost: GhostForecast | None = None
    if ghost_prompt is not None and state.explicit_conclude:
        ghost = await _run_ghost_phase(state=state, ghost_prompt=ghost_prompt, llm_call=llm_call, log_prefix=log_prefix)

    _log_completion(state, log_prefix)
    return _freeze_result(state, findings_markdown, ghost)


async def run_agentic_loop(
    system_prompt: str,
    user_brief: str,
    tools: list[ToolSpec],
    config: LoopConfig,
    llm_call: LlmCall | None = None,
    ghost_prompt: str | None = None,
    *,
    log_prefix: str = "",
    now: Callable[[], float] | None = None,
) -> LoopResult:
    now_fn = now or time.monotonic
    call = llm_call or build_default_llm_call(config)
    state = _LoopState(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_brief},
        ],
        started_at_s=now_fn(),
        deadline_at_s=now_fn() + config.wall_deadline_s,
    )
    state.telemetry.model = config.model

    try:
        return await asyncio.wait_for(
            _run_loop_body(
                state=state,
                tools=tools,
                config=config,
                llm_call=call,
                ghost_prompt=ghost_prompt,
                log_prefix=log_prefix,
                now=now_fn,
            ),
            timeout=config.wall_deadline_s,
        )
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        state.telemetry.deadline_hit = True
        state.telemetry.wall_s = now_fn() - state.started_at_s
        findings_markdown = render_findings(state.findings, state.pending_leads)
        _log_completion(state, log_prefix)
        return _freeze_result(state, findings_markdown, None)
    except Exception:  # noqa: BLE001, HARNESS-SCAN-EXEMPT-broad-except  # sanctioned package boundary: mirror v1 soft-fail contract and never raise past the harness except on cancellation
        logger.exception("%sAgentic loop failed; soft-failing to banked findings if any", log_prefix)
        state.telemetry.wall_s = now_fn() - state.started_at_s
        findings_markdown = render_findings(state.findings, state.pending_leads)
        _log_completion(state, log_prefix)
        return _freeze_result(state, findings_markdown, None)


__all__ = ["LlmCall", "run_agentic_loop"]
