"""Shared fake-LLM scaffolding for the agentic gap-fill v2 tests.

litellm-shaped response doubles (choices[0].message with optional tool_calls)
plus a scripted ``FakeLlm`` used by ``tests/test_agentic_loop.py``,
``tests/test_agentic_gates.py`` and ``tests/test_agentic_gap_fill.py``. Also the
loop-driving helpers (``tool_spec`` / ``loop_config`` / ``tool_messages`` /
``gap_accounting``) those suites share. One copy here so they can't drift.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from metaculus_bot.research.agentic.types import LoopConfig, ToolSpec


def tool_spec(name: str, handler: Any, *, timeout_s: float = 0.1) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} description",
        parameters={"type": "object", "properties": {}, "additionalProperties": True},
        handler=handler,
        timeout_s=timeout_s,
    )


def loop_config(
    *,
    model: str = "openai/gpt-5.6-luna",
    reasoning_effort: str = "medium",
    max_tool_calls: int = 14,
    wall_deadline_s: float = 1.0,
    conclude_threshold_s: float = 0.1,
    max_result_chars: int = 8000,
    max_steps: int = 5,
    max_conclude_gate_rejections: int = 2,
) -> LoopConfig:
    return LoopConfig(
        model=model,
        reasoning_effort=reasoning_effort,
        max_tool_calls=max_tool_calls,
        wall_deadline_s=wall_deadline_s,
        conclude_threshold_s=conclude_threshold_s,
        max_result_chars=max_result_chars,
        max_steps=max_steps,
        max_conclude_gate_rejections=max_conclude_gate_rejections,
    )


def tool_messages(result: Any) -> list[dict[str, Any]]:
    return [message for message in result.transcript if message["role"] == "tool"]


def gap_accounting(
    *gap_ids: str, actions: str = "searched and fetched the source", status: str = "resolved"
) -> list[dict[str, str]]:
    """gap_accounting entries for the given gap ids (default actions cite a fetch,
    satisfying the per-gap fetch-floor clause)."""
    return [{"gap_id": gap_id, "actions_taken": actions, "status": status} for gap_id in gap_ids]


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice]


def tool_call(tool_id: str, name: str, arguments: dict[str, Any] | None = None) -> FakeToolCall:
    return FakeToolCall(tool_id, FakeFunction(name=name, arguments=json.dumps(arguments or {})))


def plan_call(
    tool_id: str = "plan0",
    *,
    gaps: list[dict[str, Any]] | None = None,
    sensitive_assumptions: list[str] | None = None,
    dry_run_forecast: dict[str, Any] | None = None,
) -> FakeToolCall:
    """A ``set_research_plan`` tool call — the W1 gate opener every loop needs
    before any external tool runs. Defaults to a single generic gap (a zero-gap
    plan is now rejected, F3a, so the default must carry one to open the gate);
    pass ``gaps`` to exercise plan telemetry / the conclude gate. Tests that then
    conclude early without satisfying the W2 gap-accounting floor should pass
    ``max_conclude_gate_rejections=0`` to keep the conclude gate out of the way."""
    default_gaps = [{"id": "g1", "question": "What is the load-bearing fact?"}]
    arguments: dict[str, Any] = {"gaps": gaps if gaps is not None else default_gaps}
    if sensitive_assumptions is not None:
        arguments["sensitive_assumptions"] = sensitive_assumptions
    if dry_run_forecast is not None:
        arguments["dry_run_forecast"] = dry_run_forecast
    return tool_call(tool_id, "set_research_plan", arguments)


def response(content: str = "", tool_calls: list[FakeToolCall] | None = None) -> FakeResponse:
    return FakeResponse(choices=[FakeChoice(message=FakeMessage(content=content, tool_calls=tool_calls))])


class FakeLlm:
    """Scripted llm_call double; records every invocation.

    ``calls`` entries are ``{"messages": ..., "tools": ...}`` dicts,
    deep-copied at call time so append-only-transcript assertions compare
    genuine snapshots rather than aliased lists.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> Any:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)})
        if not self._responses:
            raise AssertionError("FakeLlm ran out of scripted responses")
        response_or_exc = self._responses.pop(0)
        if isinstance(response_or_exc, Exception):
            raise response_or_exc
        return response_or_exc  # noqa: ASYNC910 - scripted double; async-by-contract, no checkpoint needed
