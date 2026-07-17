from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from metaculus_bot.research.agentic.loop import _summarize_ghost, run_agentic_loop
from metaculus_bot.research.agentic.types import LoopConfig, ToolOutcome, ToolSpec
from metaculus_bot.structured_output_schema import NumericStructured
from tests.agentic_fakes import FakeLlm
from tests.agentic_fakes import response as _response
from tests.agentic_fakes import tool_call as _tool_call


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, delta: float) -> None:
        self.value += delta


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

    async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
        return ToolOutcome(  # noqa: ASYNC910
            content_markdown="Authoritative page text.", links=["https://example.com"], method="search"
        )

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
    assert result.telemetry.model == "openai/gpt-5.4-mini"
    assert result.telemetry.steps == 3
    assert result.telemetry.tool_calls == 3
    assert result.telemetry.per_tool_counts == {"search_web": 1, "record_findings": 1, "conclude": 1}
    assert result.telemetry.rendered_fetches == 0
    assert result.telemetry.dup_tool_calls == 0
    assert result.telemetry.findings_count == 1
    assert result.telemetry.pending_leads_count == 1
    assert result.telemetry.concluded_early is True
    marker_lines = [record.getMessage() for record in caplog.records if "GAP_FILL_V2:" in record.getMessage()]
    assert len(marker_lines) == 1
    # Plan-§6 marker shape: model + per-surface counters, grep-able in run_logs.
    marker = marker_lines[0]
    assert "model=openai/gpt-5.4-mini" in marker
    assert "searches=1" in marker
    assert "fetches=0" in marker
    assert "rendered=0" in marker
    assert "reads=0" in marker
    assert "dup_tool_calls=0" in marker


@pytest.mark.asyncio
async def test_duplicate_tool_calls_counted_and_warned(caplog: pytest.LogCaptureFixture) -> None:
    """Exact-duplicate (tool, normalized-args) repeats bump dup_tool_calls and
    append a gentle warning to the duplicate's tool result — no enforcement."""
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")

    async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
        return ToolOutcome(content_markdown="done", method="search")  # noqa: ASYNC910

    # Same args with shuffled key order still count as the same call.
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report", "extra": 1})]),
            _response(tool_calls=[_tool_call("s2", "search_web", {"extra": 1, "query": "report"})]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("search_web", search_web)],
        _config(),
        llm_call=fake_llm,
    )

    assert result.telemetry.dup_tool_calls == 1
    tool_messages = _tool_messages(result)
    first, second = tool_messages[0], tool_messages[1]
    assert "already made earlier in this run" not in first["content"]
    assert "already made earlier in this run" in second["content"]
    # The duplicate still executed — counter + warning only, no enforcement.
    assert "done" in second["content"]
    marker = next(record.getMessage() for record in caplog.records if "GAP_FILL_V2:" in record.getMessage())
    assert "dup_tool_calls=1" in marker


@pytest.mark.asyncio
async def test_rendered_fetch_outcomes_counted_in_telemetry() -> None:
    async def fetch(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
        return ToolOutcome(content_markdown="rendered body", method="rendered")  # noqa: ASYNC910

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://example.com"})]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("fetch", fetch)],
        _config(),
        llm_call=fake_llm,
    )

    assert result.telemetry.per_tool_counts["fetch"] == 1
    assert result.telemetry.rendered_fetches == 1


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

    async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
        clock.advance(95.0)
        return ToolOutcome(content_markdown="late result", method="search")  # noqa: ASYNC910

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
    async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
        return ToolOutcome(content_markdown="done", method="search")  # noqa: ASYNC910

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
    async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
        return ToolOutcome(content_markdown="done", method="search")  # noqa: ASYNC910

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
    async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
        return ToolOutcome(content_markdown="done", method="search")  # noqa: ASYNC910

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


class TestSummarizeGhost:
    """Branch coverage for _summarize_ghost (MC + numeric; binary is covered
    by test_ghost_phase_runs_after_conclude_and_logs_marker above)."""

    def test_multiple_choice_formats_sorted_option_probs(self) -> None:
        raw = (
            "analysis\n```json\n"
            '{"question_type": "multiple_choice", "option_probs": {"Zeta": 0.5, "Alpha": 0.3, "Mid": 0.2}}'
            "\n```"
        )

        qtype, summary = _summarize_ghost(raw)

        assert qtype == "multiple_choice"
        assert summary == "Alpha=0.300, Mid=0.200, Zeta=0.500"

    def test_numeric_reports_median(self) -> None:
        raw = (
            "analysis\n```json\n"
            '{"question_type": "numeric", "declared_percentiles": {"0.1": 10.0, "0.5": 20.5, "0.9": 30.0}}'
            "\n```"
        )

        qtype, summary = _summarize_ghost(raw)

        assert qtype == "numeric"
        assert summary == "median=20.5"

    def test_numeric_missing_median_yields_empty_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The median-None guard. Unreachable through a schema-valid parse
        (NumericStructured requires percentile 0.5), so build the block via
        model_construct and stub the parse to return it."""
        block = NumericStructured.model_construct(question_type="numeric", declared_percentiles={0.1: 10.0, 0.9: 30.0})

        def fake_parse(raw_text: str, qtype: str) -> Any:
            return block if qtype == "numeric" else None

        monkeypatch.setattr("metaculus_bot.research.agentic.loop.parse_structured_block", fake_parse)

        qtype, summary = _summarize_ghost("whatever")

        assert qtype == "numeric"
        assert summary == ""

    def test_unparseable_text_reports_unknown(self) -> None:
        qtype, summary = _summarize_ghost("no structured block here")

        assert qtype == "unknown"
        assert summary == ""
