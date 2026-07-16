from __future__ import annotations

import asyncio
import copy
import json
import logging
from dataclasses import dataclass
from typing import Any

import pytest

from metaculus_bot.research.agentic.loop import run_agentic_loop
from metaculus_bot.research.agentic.types import LoopConfig, ToolOutcome, ToolSpec


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction


@dataclass
class _FakeMessage:
    content: str
    tool_calls: list[_FakeToolCall] | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


class FakeLlm:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> Any:
        self.calls.append({"messages": copy.deepcopy(messages), "tools": copy.deepcopy(tools)})
        if not self._responses:
            raise AssertionError("FakeLlm ran out of scripted responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, delta: float) -> None:
        self.value += delta


def _tool_call(tool_id: str, name: str, arguments: dict[str, Any] | None = None) -> _FakeToolCall:
    return _FakeToolCall(tool_id, _FakeFunction(name=name, arguments=json.dumps(arguments or {})))


def _response(content: str = "", tool_calls: list[_FakeToolCall] | None = None) -> _FakeResponse:
    return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content=content, tool_calls=tool_calls))])


def _tool_spec(
    name: str,
    handler: Any,
    *,
    timeout_s: float = 0.1,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} description",
        parameters={"type": "object", "properties": {}, "additionalProperties": True},
        handler=handler,
        timeout_s=timeout_s,
    )


def _config(
    *,
    model: str = "openai/gpt-5.4-mini",
    reasoning_effort: str = "medium",
    max_tool_calls: int = 14,
    wall_deadline_s: float = 1.0,
    conclude_threshold_s: float = 0.1,
    max_result_chars: int = 8000,
    max_steps: int = 5,
) -> LoopConfig:
    return LoopConfig(
        model=model,
        reasoning_effort=reasoning_effort,
        max_tool_calls=max_tool_calls,
        wall_deadline_s=wall_deadline_s,
        conclude_threshold_s=conclude_threshold_s,
        max_result_chars=max_result_chars,
        max_steps=max_steps,
    )


def _tool_messages(result: Any) -> list[dict[str, Any]]:
    return [message for message in result.transcript if message["role"] == "tool"]


def _tool_names(tools_json: list[dict[str, Any]] | None) -> list[str]:
    return [] if tools_json is None else [tool["function"]["name"] for tool in tools_json]


@pytest.mark.asyncio
async def test_happy_path_records_findings_and_emits_telemetry(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")

    async def search_web(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="Authoritative page text.", links=["https://example.com"], method="search")

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_tool_call("search1", "search_web", {"query": "report"})]),
            _response(
                tool_calls=[
                    _tool_call(
                        "record1",
                        "record_findings",
                        {
                            "findings": [
                                {
                                    "claim": "The report was published on July 1.",
                                    "source_url": "https://example.com",
                                    "quote": "Published on July 1, 2026.",
                                    "date": "2026-07-01",
                                    "retrieved_how": "search_web",
                                    "topic": "timeline",
                                }
                            ]
                        },
                    )
                ]
            ),
            _response(tool_calls=[_tool_call("done1", "conclude", {"pending_leads": ["Check the appendix."]})]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("search_web", search_web)],
        _config(),
        llm_call=fake_llm,
    )

    assert "## Agentic Research Findings" in result.findings_markdown
    assert "The report was published on July 1." in result.findings_markdown
    assert result.telemetry.steps == 3
    assert result.telemetry.tool_calls == 3
    assert result.telemetry.per_tool_counts == {"search_web": 1, "record_findings": 1, "conclude": 1}
    assert result.telemetry.findings_count == 1
    assert result.telemetry.pending_leads_count == 1
    assert result.telemetry.concluded_early is True
    assert any("GAP_FILL_V2:" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_parallel_tool_calls_execute_concurrently_and_append_budget_line() -> None:
    started: list[str] = []
    finished: list[str] = []
    release = asyncio.Event()

    def _parallel_handler(name: str) -> Any:
        async def _handler(**_: Any) -> ToolOutcome:
            started.append(name)
            if len(started) == 3:
                release.set()
            await release.wait()
            finished.append(name)
            return ToolOutcome(content_markdown=f"{name} done", method="plain")

        return _handler

    fake_llm = FakeLlm(
        [
            _response(
                tool_calls=[
                    _tool_call("a", "alpha"),
                    _tool_call("b", "beta"),
                    _tool_call("c", "gamma"),
                ]
            ),
            _response(tool_calls=[_tool_call("done", "conclude")]),
        ]
    )

    result = await asyncio.wait_for(
        run_agentic_loop(
            "system",
            "user",
            [
                _tool_spec("alpha", _parallel_handler("alpha")),
                _tool_spec("beta", _parallel_handler("beta")),
                _tool_spec("gamma", _parallel_handler("gamma")),
            ],
            _config(),
            llm_call=fake_llm,
        ),
        timeout=0.5,
    )

    assert started == ["alpha", "beta", "gamma"]
    assert set(finished) == {"alpha", "beta", "gamma"}
    tool_messages = _tool_messages(result)[:3]
    assert [message["name"] for message in tool_messages] == ["alpha", "beta", "gamma"]
    budget_lines = [message["content"].splitlines()[-1] for message in tool_messages]
    assert len(set(budget_lines)) == 1
    assert budget_lines[0].startswith("[budget: ")
    assert budget_lines[0].endswith("3/14 tool calls used]")


@pytest.mark.asyncio
async def test_budget_threshold_switches_to_conclude_mode_and_shrinks_tools() -> None:
    clock = FakeClock()

    async def search_web(**_: Any) -> ToolOutcome:
        clock.advance(95.0)
        return ToolOutcome(content_markdown="late result", method="search")

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_tool_call("search1", "search_web")]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("search_web", search_web)],
        _config(wall_deadline_s=100.0, conclude_threshold_s=10.0),
        llm_call=fake_llm,
        now=clock,
    )

    assert "[budget: 5s remaining — you must conclude now]" in _tool_messages(result)[0]["content"]
    assert _tool_names(fake_llm.calls[1]["tools"]) == ["record_findings", "conclude"]


@pytest.mark.asyncio
async def test_deadline_mid_tool_still_returns_banked_findings() -> None:
    async def slow_tool(**_: Any) -> ToolOutcome:
        await asyncio.sleep(0.2)
        return ToolOutcome(content_markdown="too slow", method="plain")

    fake_llm = FakeLlm(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "record1",
                        "record_findings",
                        {
                            "findings": [
                                {
                                    "claim": "The statute took effect in 2025.",
                                    "source_url": "https://example.com/statute",
                                    "quote": "This act takes effect in 2025.",
                                    "topic": "law",
                                }
                            ]
                        },
                    ),
                    _tool_call("slow1", "slow_tool"),
                ]
            )
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("slow_tool", slow_tool, timeout_s=1.0)],
        _config(wall_deadline_s=0.05, max_steps=2),
        llm_call=fake_llm,
    )

    assert "The statute took effect in 2025." in result.findings_markdown
    assert result.telemetry.deadline_hit is True
    assert result.ghost is None


@pytest.mark.asyncio
async def test_tool_timeout_is_reported_and_loop_continues() -> None:
    async def slow_tool(**_: Any) -> ToolOutcome:
        await asyncio.sleep(0.05)
        return ToolOutcome(content_markdown="never seen", method="plain")

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_tool_call("slow1", "slow_tool")]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("slow_tool", slow_tool, timeout_s=0.01)],
        _config(),
        llm_call=fake_llm,
    )

    assert "status: timeout" in _tool_messages(result)[0]["content"]
    assert result.telemetry.steps == 2


@pytest.mark.asyncio
async def test_record_findings_rejects_lint_and_banks_clean_findings() -> None:
    fake_llm = FakeLlm(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "record1",
                        "record_findings",
                        {
                            "findings": [
                                {
                                    "claim": "This likely resolves soon.",
                                    "source_url": "https://example.com/bad",
                                    "quote": "Likely to resolve soon.",
                                    "topic": "general",
                                },
                                {
                                    "claim": "The board published the minutes.",
                                    "source_url": "https://example.com/good",
                                    "quote": "Minutes were published on Tuesday.",
                                    "topic": "minutes",
                                },
                            ]
                        },
                    )
                ]
            ),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop("system", "user", [], _config(), llm_call=fake_llm)

    assert "The board published the minutes." in result.findings_markdown
    assert "This likely resolves soon." not in result.findings_markdown
    assert result.telemetry.lint_rejections == 1
    assert "findings[0] rejected" in _tool_messages(result)[0]["content"]


@pytest.mark.asyncio
async def test_cap_exhaustion_restricts_next_step_to_internal_tools() -> None:
    async def search_web(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="done", method="search")

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_tool_call("search1", "search_web")]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("search_web", search_web)],
        _config(max_tool_calls=1),
        llm_call=fake_llm,
    )

    assert _tool_names(fake_llm.calls[1]["tools"]) == ["record_findings", "conclude"]
    assert result.telemetry.tool_calls == 2


@pytest.mark.asyncio
async def test_append_only_message_history_is_preserved_across_steps() -> None:
    async def search_web(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="done", method="search")

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_tool_call("search1", "search_web")]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("search_web", search_web)],
        _config(),
        llm_call=fake_llm,
    )

    for earlier, later in zip(fake_llm.calls, fake_llm.calls[1:]):
        assert later["messages"][: len(earlier["messages"])] == earlier["messages"]


@pytest.mark.asyncio
async def test_soft_fail_preserves_banked_findings_on_broken_llm_response() -> None:
    fake_llm = FakeLlm(
        [
            _response(
                tool_calls=[
                    _tool_call(
                        "record1",
                        "record_findings",
                        {
                            "findings": [
                                {
                                    "claim": "The agency released the bulletin.",
                                    "source_url": "https://example.com/bulletin",
                                    "quote": "Bulletin released Friday.",
                                    "topic": "bulletin",
                                }
                            ]
                        },
                    )
                ]
            ),
            object(),
        ]
    )

    result = await run_agentic_loop("system", "user", [], _config(), llm_call=fake_llm)

    assert "The agency released the bulletin." in result.findings_markdown
    assert result.ghost is None


@pytest.mark.asyncio
async def test_soft_fail_returns_empty_result_on_injected_loop_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    async def search_web(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="done", method="search")

    fake_llm = FakeLlm([_response(tool_calls=[_tool_call("search1", "search_web")])])
    monkeypatch.setattr(
        "metaculus_bot.research.agentic.loop._tool_schemas",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("search_web", search_web)],
        _config(),
        llm_call=fake_llm,
    )

    assert result.findings_markdown == ""
    assert result.ghost is None


@pytest.mark.asyncio
async def test_ghost_phase_runs_after_conclude_and_logs_marker(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
    ghost_text = 'analysis\n```json\n{"question_type":"binary","posterior_prob":0.42}\n```'
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_tool_call("done1", "conclude")]),
            _response(content=ghost_text),
        ]
    )

    result = await run_agentic_loop("system", "user", [], _config(), llm_call=fake_llm, ghost_prompt="ghost now")

    assert result.ghost is not None
    assert result.ghost.qtype == "binary"
    assert result.ghost.parsed_summary == "posterior_prob=0.4200"
    assert any("GHOST_FORECAST:" in record.getMessage() for record in caplog.records)
