"""Shared fake-LLM scaffolding for the agentic gap-fill v2 tests.

litellm-shaped response doubles (choices[0].message with optional tool_calls)
plus a scripted ``FakeLlm`` used by both ``tests/test_agentic_loop.py`` and
``tests/test_agentic_gap_fill.py``. One copy here so the two suites can't
drift.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any


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
    before any external tool runs. Defaults to an empty gap list (enough to open
    the gate); pass fields to exercise plan telemetry."""
    arguments: dict[str, Any] = {"gaps": gaps if gaps is not None else []}
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
