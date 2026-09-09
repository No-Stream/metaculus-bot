"""The ghost forecast's REQUEST shape: same tool list as the research turns, tools forbidden.

The ghost is a telemetry-only call appended to the driver transcript after ``conclude``. OpenAI's
prompt cache keys on the rendered prefix, tool definitions included, so a ghost sent with
``tools=None`` re-paid full input price on the whole transcript (about $0.09 a question, 29% of
the v2 driver line in the 2026-09-09 cost pass). Offering the last turn's tool list with
``tool_choice="none"`` keeps the prefix matching without letting the ghost call a tool. The ghost's
OUTPUT and its two markers are pinned in tests/test_agentic_loop.py and must not change.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from metaculus_bot.research.agentic.loop import run_agentic_loop
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
