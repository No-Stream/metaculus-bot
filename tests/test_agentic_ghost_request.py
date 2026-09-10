"""The ghost forecasts' REQUEST shapes: same tool list as the research turns, tools forbidden.

The ghost is a telemetry-only call appended to the driver transcript after ``conclude``. OpenAI's
prompt cache keys on the rendered prefix, tool definitions included, so a ghost sent with
``tools=None`` re-paid full input price on the whole transcript (about $0.09 a question, 29% of
the v2 driver line in the 2026-09-09 cost pass). Offering the last turn's tool list with
``tool_choice="none"`` keeps the prefix matching without letting the ghost call a tool. The ghost's
OUTPUT and its two markers are pinned in tests/test_agentic_loop.py and must not change.

The v1 ghost (``run_ghost_v1``) is the same request branched off the transcript BEFORE the plain
ghost's prompt, with gap-fill v1's section in its brief: same cached prefix, same tool list, tools
forbidden, and the plain ghost's answer nowhere in sight, so the pair measures v1's section alone.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from metaculus_bot.research.agentic.loop import run_agentic_loop, run_ghost_v1
from metaculus_bot.research.agentic.tool_schemas import _INTERNAL_TOOL_NAMES
from metaculus_bot.research.agentic.types import ToolOutcome
from tests.agentic_fakes import FakeLlm
from tests.agentic_fakes import gap_accounting as _accounting
from tests.agentic_fakes import loop_config as _config
from tests.agentic_fakes import plan_call as _plan_call
from tests.agentic_fakes import response as _response
from tests.agentic_fakes import tool_call as _tool_call
from tests.agentic_fakes import tool_spec as _tool_spec

GHOST_TEXT = 'analysis\n```json\n{"question_type":"binary","posterior_prob":0.42}\n```'


def _tool_names(tools_json: list[dict[str, Any]] | None) -> list[str]:
    return [] if tools_json is None else [tool["function"]["name"] for tool in tools_json]


async def _fetch(**_: Any) -> ToolOutcome:
    return ToolOutcome(content_markdown="Authoritative page text.", method="rendered")


@pytest.mark.asyncio
async def test_ghost_request_reuses_the_last_turns_tools_and_forbids_tool_use(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://example.com"})]),
            _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1")})]),
            _response(content=GHOST_TEXT),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("fetch", _fetch)],
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
        ghost_prompt="ghost now",
    )

    *research_turns, ghost_call = fake_llm.calls
    assert len(research_turns) == 3
    assert all(turn["tool_choice"] is None for turn in research_turns)
    assert "fetch" in _tool_names(research_turns[-1]["tools"])
    assert ghost_call["tools"] == research_turns[-1]["tools"]
    assert ghost_call["tool_choice"] == "none"
    assert ghost_call["messages"][-1] == {"role": "user", "content": "ghost now"}

    assert result.ghost is not None
    assert result.ghost.raw_text == GHOST_TEXT
    messages = [record.getMessage() for record in caplog.records]
    assert any(line.startswith("GHOST_FORECAST: qtype=binary summary=posterior_prob=0.4200") for line in messages)
    (json_line,) = [line for line in messages if line.startswith("GHOST_FORECAST_JSON:")]
    assert json.loads(json_line.split("GHOST_FORECAST_JSON:", 1)[1]) == {"qtype": "binary", "prob": 0.42}


@pytest.mark.asyncio
async def test_ghost_request_after_a_forced_conclude_reuses_the_internal_only_list() -> None:
    """When the tool budget forces conclude, the last turn offered only the internal tools; the
    ghost re-sends exactly that list, since that is the prefix the cache holds."""
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://example.com"})]),
            _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1")})]),
            _response(content=GHOST_TEXT),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("fetch", _fetch)],
        _config(max_tool_calls=1, max_conclude_gate_rejections=0),
        llm_call=fake_llm,
        ghost_prompt="ghost now",
    )

    assert result.ghost is not None
    conclude_turn, ghost_call = fake_llm.calls[-2:]
    assert set(_tool_names(conclude_turn["tools"])) == set(_INTERNAL_TOOL_NAMES)
    assert ghost_call["tools"] == conclude_turn["tools"]
    assert ghost_call["tool_choice"] == "none"


@pytest.mark.asyncio
async def test_no_ghost_prompt_means_no_ghost_call() -> None:
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1")})]),
        ]
    )

    result = await run_agentic_loop("system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm)

    assert result.ghost is None
    assert len(fake_llm.calls) == 2
    assert all(call["tool_choice"] is None for call in fake_llm.calls)


V1_GHOST_TEXT = 'analysis\n```json\n{"question_type":"binary","posterior_prob":0.55}\n```'
V1_LOG_PREFIX = "question=https://www.metaculus.com/questions/650/ "


def _happy_loop_with_two_ghosts() -> FakeLlm:
    """Plan, fetch, conclude, the plain ghost's answer, then one more answer left for the v1 ghost."""
    return FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://example.com"})]),
            _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1")})]),
            _response(content=GHOST_TEXT),
            _response(content=V1_GHOST_TEXT),
        ]
    )


@pytest.mark.asyncio
async def test_v1_ghost_branches_off_the_pre_ghost_transcript_with_the_same_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
    fake_llm = _happy_loop_with_two_ghosts()
    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("fetch", _fetch)],
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
        ghost_prompt="ghost now",
    )
    assert result.ghost_context is not None

    ghost_v1 = await run_ghost_v1(result.ghost_context, "ghost with v1", log_prefix=V1_LOG_PREFIX)

    *_, plain_ghost_call, v1_call = fake_llm.calls
    # Same prefix as the plain ghost up to its prompt, then the v1 brief; the plain ghost's own answer is absent.
    assert v1_call["messages"] == [*plain_ghost_call["messages"][:-1], {"role": "user", "content": "ghost with v1"}]
    assert all(message.get("content") != GHOST_TEXT for message in v1_call["messages"])
    assert v1_call["tools"] == plain_ghost_call["tools"]
    assert v1_call["tool_choice"] == "none"
    # The loop's own transcript is untouched by the branch: it still ends on the plain ghost's answer.
    assert result.transcript[-1]["content"] == GHOST_TEXT

    assert ghost_v1 is not None
    assert (ghost_v1.qtype, ghost_v1.raw_text, ghost_v1.parsed_summary) == (
        "binary",
        V1_GHOST_TEXT,
        "posterior_prob=0.5500",
    )
    messages = [record.getMessage() for record in caplog.records]
    assert f"{V1_LOG_PREFIX}GHOST_FORECAST_V1: qtype=binary summary=posterior_prob=0.5500" in messages
    (json_line,) = [line for line in messages if "GHOST_FORECAST_V1_JSON:" in line]
    assert json_line.startswith(V1_LOG_PREFIX)
    assert json.loads(json_line.split("GHOST_FORECAST_V1_JSON:", 1)[1]) == {"qtype": "binary", "prob": 0.55}
    # The plain ghost's pair is byte-identical to before: one line each, untouched by the second ghost.
    assert len([line for line in messages if "GHOST_FORECAST: " in line]) == 1
    assert len([line for line in messages if "GHOST_FORECAST_JSON:" in line]) == 1


@pytest.mark.asyncio
async def test_no_plain_ghost_means_no_ghost_context() -> None:
    """The v1 ghost exists only as one half of a pair, so a loop whose plain ghost did not run hands out no context."""
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1")})]),
        ]
    )

    result = await run_agentic_loop("system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm)

    assert result.ghost is None
    assert result.ghost_context is None


@pytest.mark.asyncio
async def test_v1_ghost_failure_is_swallowed_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="metaculus_bot.research.agentic.loop")
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1")})]),
            _response(content=GHOST_TEXT),
            RuntimeError("provider hiccup"),
        ]
    )
    result = await run_agentic_loop(
        "system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm, ghost_prompt="ghost now"
    )
    assert result.ghost_context is not None

    ghost_v1 = await run_ghost_v1(result.ghost_context, "ghost with v1", log_prefix=V1_LOG_PREFIX)

    assert ghost_v1 is None
    assert any("v1 ghost phase failed: RuntimeError: provider hiccup" in r.getMessage() for r in caplog.records)
    assert not any("GHOST_FORECAST_V1" in r.getMessage() for r in caplog.records)
