from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from metaculus_bot.research.agentic.loop import _summarize_ghost, run_agentic_loop
from metaculus_bot.research.agentic.types import LoopConfig, ToolOutcome, ToolSpec
from metaculus_bot.structured_output_schema import NumericStructured
from tests.agentic_fakes import FakeLlm
from tests.agentic_fakes import plan_call as _plan_call
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
            _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "When was the report published?"}])]),
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
            _response(
                tool_calls=[
                    _tool_call(
                        "done1",
                        "conclude",
                        {
                            "pending_leads": ["Check the appendix."],
                            # W2 conclude gate: one gap, so its accounting must
                            # cover g1 and (per-gap fetch-floor clause) cite a
                            # fetch/read action in actions_taken.
                            "gap_accounting": [
                                {
                                    "gap_id": "g1",
                                    "actions_taken": "fetched the primary source and confirmed the date",
                                    "status": "resolved",
                                }
                            ],
                        },
                    )
                ]
            ),
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
    assert result.telemetry.steps == 4
    assert result.telemetry.tool_calls == 4
    assert result.telemetry.per_tool_counts == {
        "set_research_plan": 1,
        "search_web": 1,
        "record_findings": 1,
        "conclude": 1,
    }
    assert result.telemetry.plan_gaps == 1
    assert result.telemetry.plan_skipped is False
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
    assert "plan_gaps=1" in marker
    assert "plan_skipped=False" in marker
    assert "conclude_gate_rejections=0" in marker


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
            _response(tool_calls=[_plan_call()]),
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
    # tool_messages[0] is the set_research_plan result; the two searches follow.
    tool_messages = _tool_messages(result)
    first, second = tool_messages[1], tool_messages[2]
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
            _response(tool_calls=[_plan_call()]),
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
            _response(tool_calls=[_plan_call()]),
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
            # cap=0 disables the W2 conclude gate: this test exercises parallel
            # execution + budget-line rendering, and concludes without accounting.
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        ),
        timeout=0.5,
    )

    assert started == ["alpha", "beta", "gamma"]
    assert set(finished) == {"alpha", "beta", "gamma"}
    # tool_messages[0] is the set_research_plan result; the parallel batch follows.
    tool_messages = _tool_messages(result)[1:4]
    assert [message["name"] for message in tool_messages] == ["alpha", "beta", "gamma"]
    budget_lines = [message["content"].splitlines()[-1] for message in tool_messages]
    assert len(set(budget_lines)) == 1
    assert budget_lines[0].startswith("[budget: ")
    # plan (1) + alpha/beta/gamma (3) = 4 tool calls used. The default plan's gap
    # id trails as the W1 work-list suffix.
    assert "4/14 tool calls used" in budget_lines[0]


@pytest.mark.asyncio
async def test_budget_threshold_switches_to_conclude_mode_and_shrinks_tools() -> None:
    clock = FakeClock()

    async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
        clock.advance(95.0)
        return ToolOutcome(content_markdown="late result", method="search")  # noqa: ASYNC910

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
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

    # The plan turn (tool message [0]) runs at full budget; the search (message
    # [1]) advances the clock to 5s remaining, tipping into must-conclude mode.
    # The default plan's gap id trails as the W1 work-list suffix.
    assert (
        "[budget: 5s remaining — you must conclude now unaddressed_gaps=[g1]]" in _tool_messages(result)[1]["content"]
    )
    # calls[2] is the conclude turn — the first LLM call made after the clock
    # crossed the threshold, so its tool schema is shrunk to internals only.
    assert _tool_names(fake_llm.calls[2]["tools"]) == ["set_research_plan", "record_findings", "conclude"]


@pytest.mark.asyncio
async def test_deadline_mid_tool_still_returns_banked_findings() -> None:
    async def slow_tool(**_: Any) -> ToolOutcome:
        await asyncio.sleep(0.2)
        return ToolOutcome(content_markdown="too slow", method="plain")

    # Turn 1 sets the plan (so the external slow_tool clears the W1 gate); turn 2
    # banks a finding and fires the slow tool that trips the wall deadline.
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
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
            ),
        ]
    )

    result = await run_agentic_loop(
        "system",
        # Briefing embeds the cited URL, so the finding is provenance-grounded
        # (a non-discrepancy finding may cite a briefing URL).
        "briefing cites https://example.com/statute",
        [_tool_spec("slow_tool", slow_tool, timeout_s=1.0)],
        _config(wall_deadline_s=0.05, max_steps=3),
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
            _response(tool_calls=[_plan_call()]),
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

    # tool_messages[0] is the plan result; the timed-out slow_tool is [1].
    assert "status: timeout" in _tool_messages(result)[1]["content"]
    assert result.telemetry.steps == 3


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

    # Briefing embeds the clean finding's URL so it clears the provenance gate;
    # the lint rejection is what this test exercises.
    result = await run_agentic_loop(
        "system", "briefing cites https://example.com/good", [], _config(), llm_call=fake_llm
    )

    assert "The board published the minutes." in result.findings_markdown
    assert "This likely resolves soon." not in result.findings_markdown
    assert result.telemetry.lint_rejections == 1
    assert "findings[0] rejected" in _tool_messages(result)[0]["content"]


@pytest.mark.asyncio
async def test_derivation_banned_phrase_is_not_a_lint_rejection() -> None:
    """W3: a banned-register phrase confined to the ``derivation`` field is the
    arithmetic-synthesis carve-out — it clears the lint, banks, and does NOT bump
    lint_rejections. The same phrase in ``claim`` would still be rejected."""
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "What are the yearly records?"}])]),
            _response(
                tool_calls=[
                    _tool_call(
                        "record1",
                        "record_findings",
                        {
                            "findings": [
                                {
                                    "claim": "Per-year oldest-human bound reconstructed from the quoted record.",
                                    "source_url": "https://example.com/grg",
                                    "quote": "1997: 122; 1998: 116; 1999: 119.",
                                    "date": "2026-07-01",
                                    "retrieved_how": "fetch",
                                    "topic": "oldest-human bounds",
                                    "derivation": (
                                        "1997 max 122; the annual record probably tops the prior year by 1 — "
                                        "arithmetic only, inputs above."
                                    ),
                                }
                            ]
                        },
                    )
                ]
            ),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    # max_conclude_gate_rejections=0 disables the W2 gate: this test exercises
    # the W3 lint carve-out, not the conclude gate, and concludes without
    # research/accounting.
    result = await run_agentic_loop(
        "system",
        "briefing cites https://example.com/grg",
        [],
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
    )

    assert result.telemetry.lint_rejections == 0
    assert "Per-year oldest-human bound reconstructed from the quoted record." in result.findings_markdown
    assert "Derived analysis (arithmetic from quoted sources)" in result.findings_markdown


_FINDING_UN = {
    "claim": "The UN projects a population peak of ~10.3 billion in the mid-2080s.",
    "source_url": "https://example.com/un",
    "quote": "peak of around 10.3 billion people in the mid-2080s",
    "date": "2024-07-11",
    "retrieved_how": "fetch",
    "topic": "demographics",
}
_FINDING_NASA = {
    "claim": "NASA reports no known >140m asteroid with significant 100-year impact risk.",
    "source_url": "https://example.com/nasa",
    "quote": "no known asteroid larger than 140 meters",
    "date": "2023-09-08",
    "retrieved_how": "fetch",
    "topic": "asteroids",
}


@pytest.mark.asyncio
async def test_findings_recorded_then_relisted_in_conclude_are_not_duplicated() -> None:
    """Regression (Q578): a driver that banks findings with record_findings and
    then re-lists the SAME findings in conclude's final_findings must not double
    them (observed: 8 findings rendered 16 times). Banking is idempotent by
    full-field identity."""
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_tool_call("rec1", "record_findings", {"findings": [_FINDING_UN, _FINDING_NASA]})]),
            _response(tool_calls=[_tool_call("done1", "conclude", {"final_findings": [_FINDING_UN, _FINDING_NASA]})]),
        ]
    )

    # cap=0 disables the W2 conclude gate: this test never plans (dedup is the
    # subject) and concludes without gap accounting.
    result = await run_agentic_loop(
        "system",
        "briefing cites https://example.com/un and https://example.com/nasa",
        [],
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
    )

    md = result.findings_markdown
    assert md.count("The UN projects a population peak") == 1
    assert md.count("NASA reports no known") == 1
    assert result.telemetry.findings_count == 2
    conclude_message = _tool_messages(result)[-1]
    assert "Skipped 2 final finding(s) already recorded earlier in this run." in conclude_message["content"]
    assert "Concluded with 0 final finding(s)" in conclude_message["content"]


@pytest.mark.asyncio
async def test_findings_sharing_source_but_distinct_claim_are_both_kept() -> None:
    """Dedup must key on the WHOLE finding, not just source_url: two findings
    from the same page with different claims/topics are genuinely distinct and
    must both survive (the Q578 log had exactly this shape for the NASA page)."""
    long_claim = {
        **_FINDING_NASA,
        "topic": "asteroids (with NEO Surveyor context)",
        "claim": "NASA reports no known >140m asteroid impact risk and says NEO Surveyor will accelerate discovery.",
    }
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_tool_call("rec1", "record_findings", {"findings": [_FINDING_NASA, long_claim]})]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system", "briefing cites https://example.com/nasa", [], _config(), llm_call=fake_llm
    )

    assert result.telemetry.findings_count == 2
    assert "NEO Surveyor" in result.findings_markdown


@pytest.mark.asyncio
async def test_cap_exhaustion_restricts_next_step_to_internal_tools() -> None:
    async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
        return ToolOutcome(content_markdown="done", method="search")  # noqa: ASYNC910

    # Budget of 2: the plan call (1) plus one external search (2) exhaust it, so
    # the conclude turn's tool schema shrinks to internals only.
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("search1", "search_web")]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "user",
        [_tool_spec("search_web", search_web)],
        _config(max_tool_calls=2),
        llm_call=fake_llm,
    )

    # calls[2] is the conclude turn — offered only internal tools once the cap hit.
    assert _tool_names(fake_llm.calls[2]["tools"]) == ["set_research_plan", "record_findings", "conclude"]
    assert result.telemetry.tool_calls == 3


_CLEAN_FINDING = {
    "claim": "The report was published on July 1.",
    "source_url": "https://example.com",
    "quote": "Published on July 1, 2026.",
    "topic": "timeline",
}


@pytest.mark.asyncio
async def test_external_tool_bind_error_extra_key_becomes_error_and_loop_continues() -> None:
    """An LLM-emitted extra kwarg makes spec.handler(**arguments) raise TypeError
    at bind time (async-def binds eagerly). It must surface as a status=error
    tool result inside the per-tool boundary, not crash the batch gather and
    abort the whole pass. The loop keeps running and banked findings survive."""
    invoked: list[str] = []

    async def strict_tool(*, query: str) -> ToolOutcome:  # noqa: ASYNC124 - concrete signature, no **kwargs
        invoked.append(query)
        return ToolOutcome(content_markdown="ran", method="search")  # noqa: ASYNC910

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("bad1", "strict_tool", {"query": "x", "unexpected": 1})]),
            _response(tool_calls=[_tool_call("record1", "record_findings", {"findings": [_CLEAN_FINDING]})]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "briefing cites https://example.com",
        [_tool_spec("strict_tool", strict_tool)],
        _config(),
        llm_call=fake_llm,
    )

    assert result.telemetry.steps == 4  # loop ran all four turns — no abort
    # tool_messages[0] is the plan result; the bind-error tool result is [1].
    bad_message = _tool_messages(result)[1]
    assert bad_message["tool_call_id"] == "bad1"
    assert "status: error" in bad_message["content"]
    assert "TypeError" in bad_message["content"]
    assert invoked == []  # handler body never ran; failure was at bind time
    assert "The report was published on July 1." in result.findings_markdown


@pytest.mark.asyncio
async def test_external_tool_bind_error_missing_key_becomes_error_and_loop_continues() -> None:
    """A missing required kwarg raises TypeError at bind time too; same contract:
    error tool result, loop continues, banked findings survive."""
    invoked: list[str] = []

    async def strict_tool(*, query: str) -> ToolOutcome:  # noqa: ASYNC124 - concrete signature, no **kwargs
        invoked.append(query)
        return ToolOutcome(content_markdown="ran", method="search")  # noqa: ASYNC910

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("bad1", "strict_tool", {})]),  # no `query`
            _response(tool_calls=[_tool_call("record1", "record_findings", {"findings": [_CLEAN_FINDING]})]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "briefing cites https://example.com",
        [_tool_spec("strict_tool", strict_tool)],
        _config(),
        llm_call=fake_llm,
    )

    assert result.telemetry.steps == 4
    # tool_messages[0] is the plan result; the bind-error tool result is [1].
    bad_message = _tool_messages(result)[1]
    assert bad_message["tool_call_id"] == "bad1"
    assert "status: error" in bad_message["content"]
    assert "TypeError" in bad_message["content"]
    assert invoked == []
    assert "The report was published on July 1." in result.findings_markdown


@pytest.mark.asyncio
async def test_batch_clamped_to_remaining_budget_rejects_overflow_external_calls() -> None:
    """A single turn can emit more calls than budget allows (parallel_tool_calls).
    The batch is clamped to the remaining external-call slots: in-budget calls run,
    overflow external calls are rejected with a synthetic budget-exhausted error,
    every tool_call_id still gets exactly one response, and internal tools
    (record_findings) keep working past the ceiling."""
    invoked: list[str] = []

    async def search_web(*, query: str) -> ToolOutcome:  # noqa: ASYNC124 - concrete signature, no **kwargs
        invoked.append(query)
        return ToolOutcome(content_markdown=f"result {query}", method="search")  # noqa: ASYNC910

    # Budget of 3: the plan call (1) leaves two external slots, so q1/q2 run and
    # the overflow q3 is budget-rejected — the shape this test exercises.
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(
                tool_calls=[
                    _tool_call("c1", "search_web", {"query": "q1"}),
                    _tool_call("c2", "search_web", {"query": "q2"}),
                    _tool_call("c3", "search_web", {"query": "q3"}),
                ]
            ),
            _response(tool_calls=[_tool_call("record1", "record_findings", {"findings": [_CLEAN_FINDING]})]),
            _response(tool_calls=[_tool_call("done1", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "briefing cites https://example.com",
        [_tool_spec("search_web", search_web)],
        _config(max_tool_calls=3),
        llm_call=fake_llm,
    )

    # Only the two in-budget external calls executed; the third never bound.
    assert set(invoked) == {"q1", "q2"}
    assert len(invoked) == 2
    assert result.telemetry.per_tool_counts["search_web"] == 2

    # tool_messages[0] is the plan result; the clamped parallel batch is [1:4].
    batch_messages = _tool_messages(result)[1:4]
    assert [message["tool_call_id"] for message in batch_messages] == ["c1", "c2", "c3"]

    rejected_message = next(message for message in batch_messages if message["tool_call_id"] == "c3")
    assert "status: error" in rejected_message["content"]
    assert "budget is exhausted" in rejected_message["content"]
    accepted_messages = [message for message in batch_messages if message["tool_call_id"] in {"c1", "c2"}]
    assert all("budget is exhausted" not in message["content"] for message in accepted_messages)

    # record_findings ran past the ceiling — internal tools are not budget-gated.
    assert "The report was published on July 1." in result.findings_markdown


@pytest.mark.asyncio
async def test_append_only_message_history_is_preserved_across_steps() -> None:
    async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
        return ToolOutcome(content_markdown="done", method="search")  # noqa: ASYNC910

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
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

    result = await run_agentic_loop(
        "system", "briefing cites https://example.com/bulletin", [], _config(), llm_call=fake_llm
    )

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

    result = await run_agentic_loop(
        "system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm, ghost_prompt="ghost now"
    )

    assert result.ghost is not None
    assert result.ghost.qtype == "binary"
    assert result.ghost.parsed_summary == "posterior_prob=0.4200"
    assert any("GHOST_FORECAST:" in record.getMessage() for record in caplog.records)


def _ghost_json_payload(caplog: pytest.LogCaptureFixture) -> dict:
    """Extract and parse the single GHOST_FORECAST_JSON marker line from captured logs."""
    lines = [record.getMessage() for record in caplog.records if "GHOST_FORECAST_JSON:" in record.getMessage()]
    assert len(lines) == 1, f"expected exactly one GHOST_FORECAST_JSON line, got {lines}"
    blob = lines[0].split("GHOST_FORECAST_JSON:", 1)[1].strip()
    return json.loads(blob)


@pytest.mark.asyncio
async def test_ghost_phase_emits_full_fidelity_json_marker_binary(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
    ghost_text = 'analysis\n```json\n{"question_type":"binary","posterior_prob":0.42}\n```'
    fake_llm = FakeLlm([_response(tool_calls=[_tool_call("d", "conclude")]), _response(content=ghost_text)])

    await run_agentic_loop(
        "system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm, ghost_prompt="ghost now"
    )

    payload = _ghost_json_payload(caplog)
    assert payload == {"qtype": "binary", "prob": 0.42}


@pytest.mark.asyncio
async def test_ghost_phase_emits_full_fidelity_json_marker_mc(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
    ghost_text = 'analysis\n```json\n{"question_type":"multiple_choice","option_probs":{"Blue":0.3,"Red":0.7}}\n```'
    fake_llm = FakeLlm([_response(tool_calls=[_tool_call("d", "conclude")]), _response(content=ghost_text)])

    await run_agentic_loop(
        "system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm, ghost_prompt="ghost now"
    )

    payload = _ghost_json_payload(caplog)
    assert payload == {"qtype": "multiple_choice", "option_probs": {"Blue": 0.3, "Red": 0.7}}


@pytest.mark.asyncio
async def test_ghost_phase_emits_full_fidelity_json_marker_numeric(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
    ghost_text = (
        'analysis\n```json\n{"question_type":"numeric","declared_percentiles":{"0.1":10.0,"0.5":20.5,"0.9":30.0}}\n```'
    )
    fake_llm = FakeLlm([_response(tool_calls=[_tool_call("d", "conclude")]), _response(content=ghost_text)])

    await run_agentic_loop(
        "system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm, ghost_prompt="ghost now"
    )

    payload = _ghost_json_payload(caplog)
    # JSON serializes float keys as strings; the full percentile set survives round-trip.
    assert payload == {
        "qtype": "numeric",
        "declared_percentiles": {"0.1": 10.0, "0.5": 20.5, "0.9": 30.0},
        "median": 20.5,
    }


@pytest.mark.asyncio
async def test_ghost_phase_suppresses_json_marker_when_unparseable(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
    fake_llm = FakeLlm([_response(tool_calls=[_tool_call("d", "conclude")]), _response(content="no block here")])

    await run_agentic_loop(
        "system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm, ghost_prompt="ghost now"
    )

    # Legacy marker still fires (qtype=unknown); the JSON companion is suppressed.
    assert any("GHOST_FORECAST:" in r.getMessage() for r in caplog.records)
    assert not any("GHOST_FORECAST_JSON:" in r.getMessage() for r in caplog.records)


class TestSummarizeGhost:
    """Branch coverage for _summarize_ghost (MC + numeric; binary is covered
    by test_ghost_phase_runs_after_conclude_and_logs_marker above). Also asserts
    the third tuple element — the full-fidelity forecast payload."""

    def test_multiple_choice_formats_sorted_option_probs(self) -> None:
        raw = (
            "analysis\n```json\n"
            '{"question_type": "multiple_choice", "option_probs": {"Zeta": 0.5, "Alpha": 0.3, "Mid": 0.2}}'
            "\n```"
        )

        qtype, summary, payload = _summarize_ghost(raw)

        assert qtype == "multiple_choice"
        assert summary == "Alpha=0.300, Mid=0.200, Zeta=0.500"
        assert payload == {"qtype": "multiple_choice", "option_probs": {"Zeta": 0.5, "Alpha": 0.3, "Mid": 0.2}}

    def test_numeric_reports_median(self) -> None:
        raw = (
            "analysis\n```json\n"
            '{"question_type": "numeric", "declared_percentiles": {"0.1": 10.0, "0.5": 20.5, "0.9": 30.0}}'
            "\n```"
        )

        qtype, summary, payload = _summarize_ghost(raw)

        assert qtype == "numeric"
        assert summary == "median=20.5"
        assert payload == {
            "qtype": "numeric",
            "declared_percentiles": {0.1: 10.0, 0.5: 20.5, 0.9: 30.0},
            "median": 20.5,
        }

    def test_numeric_missing_median_yields_empty_summary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The median-None guard. Unreachable through a schema-valid parse
        (NumericStructured requires percentile 0.5), so build the block via
        model_construct and stub the parse to return it."""
        block = NumericStructured.model_construct(question_type="numeric", declared_percentiles={0.1: 10.0, 0.9: 30.0})

        def fake_parse(_raw_text: str, qtype: str) -> Any:
            return block if qtype == "numeric" else None

        monkeypatch.setattr("metaculus_bot.research.agentic.loop.parse_structured_block", fake_parse)

        qtype, summary, payload = _summarize_ghost("whatever")

        assert qtype == "numeric"
        assert summary == ""
        # Payload still carries the full (median-less) percentile set for scoring.
        assert payload == {"qtype": "numeric", "declared_percentiles": {0.1: 10.0, 0.9: 30.0}, "median": None}

    def test_unparseable_text_reports_unknown(self) -> None:
        qtype, summary, payload = _summarize_ghost("no structured block here")

        assert qtype == "unknown"
        assert summary == ""
        assert payload is None

    @pytest.mark.parametrize(
        ("qtype", "block"),
        [
            ("numeric", '{"question_type":"numeric","declared_percentiles":{"0.1":10.0,"0.5":20.5,"0.9":30.0}}'),
            ("multiple_choice", '{"question_type":"multiple_choice","option_probs":{"Blue":0.3,"Red":0.7}}'),
            ("binary", '{"question_type":"binary","posterior_prob":0.42}'),
        ],
    )
    def test_no_qtype_mismatch_warnings_on_declared_block(
        self, caplog: pytest.LogCaptureFixture, qtype: str, block: str
    ) -> None:
        """Regression (BUG 2): the probe used to try all three qtypes, tripping
        the shared parser's question_type-mismatch WARN for the non-matching
        ones (5 spurious WARNs/run). Reading the declared type first parses only
        the matching type — zero mismatch WARNs on the expected path."""
        caplog.set_level(logging.WARNING, logger="metaculus_bot.structured_output_schema")

        result_qtype, _summary, _payload = _summarize_ghost(f"analysis\n```json\n{block}\n```")

        assert result_qtype == qtype
        mismatch_warns = [r for r in caplog.records if "question_type mismatch" in r.getMessage()]
        assert mismatch_warns == []


@pytest.mark.asyncio
async def test_ghost_phase_numeric_emits_no_qtype_mismatch_warnings(caplog: pytest.LogCaptureFixture) -> None:
    """End-to-end guard: a numeric ghost run through the full loop leaves no
    question_type-mismatch WARN in the structured_output_schema logger."""
    caplog.set_level(logging.WARNING, logger="metaculus_bot.structured_output_schema")
    ghost_text = (
        'analysis\n```json\n{"question_type":"numeric","declared_percentiles":{"0.1":10.0,"0.5":20.5,"0.9":30.0}}\n```'
    )
    fake_llm = FakeLlm([_response(tool_calls=[_tool_call("d", "conclude")]), _response(content=ghost_text)])

    result = await run_agentic_loop(
        "system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm, ghost_prompt="ghost now"
    )

    assert result.ghost is not None and result.ghost.qtype == "numeric"
    assert [r for r in caplog.records if "question_type mismatch" in r.getMessage()] == []


def _finding(source_url: str, *, quote: str = "Verbatim quote from the source.", discrepancy: bool = False) -> dict:
    return {
        "claim": "The board published the minutes on July 1.",
        "source_url": source_url,
        "quote": quote,
        "date": "2026-07-01",
        "retrieved_how": "fetch",
        "topic": "minutes",
        "discrepancy": discrepancy,
    }


def _search_returning(content: str, links: list[str] | None = None):
    async def _search(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
        return ToolOutcome(content_markdown=content, links=links or [], method="search")  # noqa: ASYNC910

    return _search


def _returns_method(content: str, *, method: str, status: str = "ok", links: list[str] | None = None):
    """A tool handler returning a ToolOutcome with a specific method/status —
    used by the W4 tier-stamping tests to drive fetched vs snippet vs failed
    retrievals through the loop."""

    async def _handler(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
        return ToolOutcome(  # noqa: ASYNC910
            content_markdown=content, links=links or [], method=method, status=status
        )

    return _handler


class TestProvenanceGate:
    """Tier-1 hard URL gate + Tier-2 warn-only quote check on agentic findings.

    A finding is rendered under a "supersedes-the-briefing" banner shown to
    every base forecaster, so a hallucinated/mistyped citation from the
    low-effort driver would silently override correct research. The URL check is
    a hard gate; the quote check only warns."""

    @pytest.mark.asyncio
    async def test_hallucinated_url_is_rejected_and_counted(self) -> None:
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://hallucinated.example/nowhere")]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop("system", "briefing with no URLs", [], _config(), llm_call=fake_llm)

        assert "The board published the minutes" not in result.findings_markdown
        assert result.telemetry.findings_count == 0
        assert result.telemetry.provenance_rejections == 1
        # The rejection reason is fed back through the existing "Rejected:" channel.
        rejection = _tool_messages(result)[0]["content"]
        assert "findings[0] rejected" in rejection
        assert "did not appear in any tool result or the briefing" in rejection

    @pytest.mark.asyncio
    async def test_url_from_tool_result_is_accepted(self) -> None:
        """A URL the driver retrieved via a tool (here, embedded in search
        result content) grounds a non-discrepancy finding."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "minutes"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://agency.example/minutes")]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning("Found it at https://agency.example/minutes — details follow.")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert "The board published the minutes" in result.findings_markdown
        assert result.telemetry.findings_count == 1
        assert result.telemetry.provenance_rejections == 0

    @pytest.mark.asyncio
    async def test_briefing_url_accepted_for_non_discrepancy(self) -> None:
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _tool_call("rec1", "record_findings", {"findings": [_finding("https://gov.example/report")]})
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system", "briefing cites https://gov.example/report as a source", [], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.provenance_rejections == 0

    @pytest.mark.asyncio
    async def test_briefing_only_url_rejected_for_discrepancy(self) -> None:
        """A discrepancy must rest on a fresh primary-source check, so a URL that
        appears ONLY in the briefing (never in a tool result) does not ground it."""
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://gov.example/report", discrepancy=True)]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system", "briefing cites https://gov.example/report as a source", [], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 0
        assert result.telemetry.provenance_rejections == 1
        rejection = _tool_messages(result)[0]["content"]
        assert "discrepancy source_url" in rejection
        assert "fresh primary-source check" in rejection

    @pytest.mark.asyncio
    async def test_tool_sourced_url_accepted_for_discrepancy(self) -> None:
        """The same discrepancy is accepted when the URL DID come from a tool."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://gov.example/report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://gov.example/report", discrepancy=True)]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        # The fetch's URL ARGUMENT is what the driver saw — enough to ground it,
        # even though the result body carries no URL.
        fetch = _search_returning("The report body, no links here.")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("fetch", fetch)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.provenance_rejections == 0

    @pytest.mark.asyncio
    async def test_url_variant_normalizes_to_seen_url_and_is_accepted(self) -> None:
        """Trailing slash, utm params, and a fragment on the cited URL all
        normalize to the tool-seen URL, so the finding is grounded."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://Agency.example/report/?utm_source=news#section-2")]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning("See https://agency.example/report for the figures.")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.provenance_rejections == 0

    @pytest.mark.asyncio
    async def test_quote_mismatch_warns_but_accepts(self, caplog: pytest.LogCaptureFixture) -> None:
        """A quote not found in the tool contents is warn-only: the finding is
        still accepted, quote_mismatch_warnings increments, and a WARN is logged."""
        caplog.set_level(logging.WARNING, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        quote="A sentence that never appears in the tool result body.",
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _search_returning("See https://agency.example/report — the body says something else entirely.")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1  # accepted despite the quote miss
        assert result.telemetry.provenance_rejections == 0
        assert result.telemetry.quote_mismatch_warnings == 1
        assert any(
            "GAP_FILL_V2 quote_mismatch" in r.getMessage()
            for r in caplog.records
            if r.name == "metaculus_bot.research.agentic.loop"
        )

    @pytest.mark.asyncio
    async def test_quote_found_in_tool_content_emits_no_warning(self) -> None:
        """A quote present (modulo whitespace/quote-glyph normalization) in the
        tool content does not warn."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://agency.example/report",
                                        quote="The rate was 4.1 percent.",
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        # Curly quotes + extra whitespace in the source still match the finding.
        search = _search_returning(
            "https://agency.example/report says:  “The rate   was 4.1 percent.”  Full text follows."
        )

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert result.telemetry.quote_mismatch_warnings == 0

    @pytest.mark.asyncio
    async def test_provenance_counters_surface_in_log_line(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {"findings": [_finding("https://hallucinated.example/nowhere")]},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        await run_agentic_loop("system", "briefing with no URLs", [], _config(), llm_call=fake_llm)

        marker = next(r.getMessage() for r in caplog.records if "GAP_FILL_V2:" in r.getMessage())
        assert "provenance_rejections=1" in marker
        assert "quote_mismatch_warnings=0" in marker


class TestHarvestVerificationTiers:
    """W4: the per-call tier harvester. Fetched-class methods (document/rendered/
    plain/cache) tier only the URLs actually retrieved (the arguments) "fetched";
    snippet-class methods (search/news) tier every surfaced URL "snippet"; a
    failed/blocked outcome or a non-retrieval method grants no tier at all."""

    def test_fetched_class_methods_tier_argument_urls_fetched(self) -> None:
        from metaculus_bot.research.agentic.loop import _harvest_verification_tiers

        for method in ("document", "rendered", "plain", "cache"):
            tiers = _harvest_verification_tiers(
                "fetch",
                {"url": "https://x.example/a"},
                ToolOutcome(content_markdown="body", method=method),
            )
            assert tiers == {"https://x.example/a": "fetched"}, method

    def test_fetched_class_does_not_tier_body_or_link_urls(self) -> None:
        """A fetch's outbound links and in-body URLs are leads, not pages we
        read — only the requested URL earns the fetched tier from this call."""
        from metaculus_bot.research.agentic.loop import _harvest_verification_tiers

        tiers = _harvest_verification_tiers(
            "fetch",
            {"url": "https://x.example/a"},
            ToolOutcome(
                content_markdown="body cites https://z.example/c",
                links=["https://y.example/b"],
                method="plain",
            ),
        )
        assert tiers == {"https://x.example/a": "fetched"}

    def test_read_document_ask_url_is_not_tiered_fetched(self) -> None:
        """F1: read_document takes a free-text ``ask``. A URL the driver types
        into ``ask`` (e.g. "compare with https://other") was never retrieved —
        only the ``url`` page was — so only ``url`` earns the fetched tier."""
        from metaculus_bot.research.agentic.loop import _harvest_verification_tiers

        tiers = _harvest_verification_tiers(
            "read_document",
            {"url": "https://x.example/a", "ask": "how does this compare to https://other.example/b?"},
            ToolOutcome(content_markdown="the answer", method="document"),
        )
        assert tiers == {"https://x.example/a": "fetched"}

    def test_search_class_methods_tier_all_surfaced_urls_snippet(self) -> None:
        from metaculus_bot.research.agentic.loop import _harvest_verification_tiers

        for method in ("search", "news"):
            tiers = _harvest_verification_tiers(
                "search_web",
                {"query": "q"},
                ToolOutcome(
                    content_markdown="see https://x.example/a",
                    links=["https://y.example/b"],
                    method=method,
                ),
            )
            assert tiers == {"https://x.example/a": "snippet", "https://y.example/b": "snippet"}, method

    def test_failed_outcome_grants_no_tier(self) -> None:
        """The 131.3 mechanism: a fetch that 403s (status != ok) confers no
        authority, so a later search snippet of the same fact stays snippet."""
        from metaculus_bot.research.agentic.loop import _harvest_verification_tiers

        tiers = _harvest_verification_tiers(
            "fetch",
            {"url": "https://x.example/a"},
            ToolOutcome(content_markdown="403", method="plain", status="blocked"),
        )
        assert tiers == {}

    def test_internal_tool_and_unknown_method_grant_no_tier(self) -> None:
        from metaculus_bot.research.agentic.loop import _harvest_verification_tiers

        assert (
            _harvest_verification_tiers("record_findings", {}, ToolOutcome(content_markdown="ok", method="internal"))
            == {}
        )
        # document_needed is an intermediate fetch state, not a real retrieval.
        assert (
            _harvest_verification_tiers(
                "fetch", {"url": "https://x.example/a"}, ToolOutcome(content_markdown="", method="document_needed")
            )
            == {}
        )


class TestVerificationTierStamping:
    """W4 end-to-end: the loop stamps each banked finding's verification_tier
    from the URL->best-method map — CODE-derived, not driver-claimed. A search-
    then-fetch upgrades the tier; a failed retrieval leaves a discrepancy
    snippet-tier and demoted (the 131.3 fix). Observed via the rendered markdown
    (tier token on topic findings; block placement on discrepancies)."""

    @pytest.mark.asyncio
    async def test_fetch_sourced_finding_renders_fetched_tier(self) -> None:
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://agency.example/report"})]),
                _response(
                    tool_calls=[
                        _tool_call("rec1", "record_findings", {"findings": [_finding("https://agency.example/report")]})
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        fetch = _returns_method("The report body.", method="plain")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("fetch", fetch)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert "[verification: fetched]" in result.findings_markdown
        assert "[verification: snippet]" not in result.findings_markdown

    @pytest.mark.asyncio
    async def test_search_sourced_finding_renders_snippet_tier(self) -> None:
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call("rec1", "record_findings", {"findings": [_finding("https://agency.example/report")]})
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _returns_method("See https://agency.example/report for the figures.", method="search")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        assert "[verification: snippet]" in result.findings_markdown
        assert "[verification: fetched]" not in result.findings_markdown

    @pytest.mark.asyncio
    async def test_search_then_fetch_upgrades_tier_to_fetched(self) -> None:
        """A URL first surfaced in a search snippet, then actually fetched,
        upgrades to fetched (best-tier wins) — even though search ran first."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://agency.example/report"})]),
                _response(
                    tool_calls=[
                        _tool_call("rec1", "record_findings", {"findings": [_finding("https://agency.example/report")]})
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _returns_method("See https://agency.example/report — snippet only.", method="search")
        fetch = _returns_method("The full report body.", method="plain")

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("search_web", search), _tool_spec("fetch", fetch)],
            _config(),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        assert "[verification: fetched]" in result.findings_markdown
        assert "[verification: snippet]" not in result.findings_markdown

    @pytest.mark.asyncio
    async def test_failed_fetch_then_snippet_leaves_discrepancy_demoted(self) -> None:
        """The 131.3 scenario end-to-end: the direct fetch 403s (no tier), a
        search snippet then surfaces the contradicting figure, and the discrepancy
        banked on that URL is snippet-tier — so it renders in the DEMOTED block,
        never superseding the briefing."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://gov.example/figure"})]),
                _response(tool_calls=[_tool_call("s1", "search_news", {"query": "figure"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    _finding(
                                        "https://gov.example/figure",
                                        quote="A snippet reports the figure is 131.3.",
                                        discrepancy=True,
                                    )
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        # The direct fetch is blocked (403) — grants no tier but still marks the
        # URL tool-seen, so the discrepancy clears the provenance gate.
        blocked_fetch = _returns_method("403 Forbidden", method="plain", status="blocked")
        # The search snippet surfaces the contradicting figure at the same URL.
        snippet_search = _returns_method("https://gov.example/figure says the figure is 131.3.", method="news")

        result = await run_agentic_loop(
            "system",
            "briefing with no URLs",
            [_tool_spec("fetch", blocked_fetch), _tool_spec("search_news", snippet_search)],
            _config(),
            llm_call=fake_llm,
        )

        assert result.telemetry.findings_count == 1
        md = result.findings_markdown
        assert "### ⚠ Possible corrections (snippet-sourced — recheck advised)" in md
        assert "### ⚠ Corrections to the briefing" not in md
        assert "do NOT supersede the briefing" in md

    @pytest.mark.asyncio
    async def test_driver_supplied_tier_is_overwritten_by_code(self) -> None:
        """A driver that puts verification_tier in its record_findings payload
        cannot self-promote: the loop overwrites it from the URL->method map. A
        snippet-only URL falsely claimed "fetched" renders as snippet."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call()]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "report"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "rec1",
                            "record_findings",
                            {
                                "findings": [
                                    {
                                        **_finding("https://agency.example/report"),
                                        "verification_tier": "fetched",  # driver lie
                                    }
                                ]
                            },
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )
        search = _returns_method("See https://agency.example/report for the figures.", method="search")

        result = await run_agentic_loop(
            "system", "briefing with no URLs", [_tool_spec("search_web", search)], _config(), llm_call=fake_llm
        )

        assert result.telemetry.findings_count == 1
        # Code-derived snippet wins over the driver's "fetched" claim.
        assert "[verification: snippet]" in result.findings_markdown
        assert "[verification: fetched]" not in result.findings_markdown


class TestResearchPlanGate:
    """W1: set_research_plan is required before any external tool call. Until it
    runs, external calls are rejected with a nudge; the plan-nudge cap forces a
    soft-continue (plan_skipped) so a driver that never plans can't wedge. The
    plan also emits the GHOST_PRE / GHOST_PRE_JSON pre-research telemetry."""

    @pytest.mark.asyncio
    async def test_external_tool_before_plan_is_rejected_with_nudge(self) -> None:
        invoked: list[str] = []

        async def search_web(**kwargs: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            invoked.append(kwargs.get("query", ""))
            return ToolOutcome(content_markdown="ran", method="search")  # noqa: ASYNC910

        # Turn 1 calls an external tool with no plan set -> rejected. Turn 2 sets
        # the plan. Turn 3's search then runs. Turn 4 concludes.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("early", "search_web", {"query": "too soon"})]),
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "now allowed"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # W1 test: cap=0 disables the W2 conclude gate so the early conclude here
        # isn't blocked (the plan-gate nudge is what this exercises).
        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        # The premature call was rejected: its handler never ran, and its tool
        # message carries the plan-required nudge.
        assert invoked == ["now allowed"]
        early_message = _tool_messages(result)[0]
        assert early_message["tool_call_id"] == "early"
        assert "status: error" in early_message["content"]
        assert "call set_research_plan first" in early_message["content"]
        # A rejected pre-plan call is NOT counted against the tool-call budget.
        assert result.telemetry.per_tool_counts.get("search_web") == 1
        assert result.telemetry.plan_skipped is False
        assert "The" not in result.findings_markdown or result.telemetry.findings_count == 0

    @pytest.mark.asyncio
    async def test_plan_accepted_then_external_tool_runs(self) -> None:
        invoked: list[str] = []

        async def search_web(**kwargs: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            invoked.append(kwargs.get("query", ""))
            return ToolOutcome(content_markdown="ran", method="search")  # noqa: ASYNC910

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "allowed"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # W1 test: cap=0 disables the W2 conclude gate (early conclude, no
        # gap_accounting) so the plan-accepted path is what's under test.
        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        assert invoked == ["allowed"]
        assert result.telemetry.plan_gaps == 1
        assert result.telemetry.plan_skipped is False

    @pytest.mark.asyncio
    async def test_plan_emits_ghost_pre_json_marker(self, caplog: pytest.LogCaptureFixture) -> None:
        """The dry-run forecast passed to set_research_plan is logged as GHOST_PRE
        / GHOST_PRE_JSON — the pre-research counterpart to the concluding ghost."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _plan_call(
                            gaps=[{"id": "g1", "question": "q?"}],
                            sensitive_assumptions=["the incumbent runs again", "turnout stays flat"],
                            dry_run_forecast={"question_type": "binary", "posterior_prob": 0.37},
                        )
                    ]
                ),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # cap=0 disables the W2 gate: this test asserts the GHOST_PRE telemetry,
        # not the conclude gate, and concludes without research/accounting.
        await run_agentic_loop(
            "system",
            "user",
            [],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
            log_prefix="question=https://q/1/ ",
        )

        pre_json_lines = [m.getMessage() for m in caplog.records if "GHOST_PRE_JSON:" in m.getMessage()]
        assert len(pre_json_lines) == 1
        blob = pre_json_lines[0].split("GHOST_PRE_JSON:", 1)[1].strip()
        assert json.loads(blob) == {"qtype": "binary", "prob": 0.37}
        # The GHOST_PRE summary line carries the plan shape and the question ref.
        pre_lines = [m.getMessage() for m in caplog.records if "GHOST_PRE:" in m.getMessage()]
        # GHOST_PRE_JSON also contains "GHOST_PRE" as a substring, so filter it out.
        pre_summary = [m for m in pre_lines if "GHOST_PRE_JSON:" not in m]
        assert len(pre_summary) == 1
        assert "gaps=1" in pre_summary[0]
        assert "sensitive_assumptions=2" in pre_summary[0]
        assert "question=https://q/1/" in pre_summary[0]

    @pytest.mark.asyncio
    async def test_ghost_pre_json_suppressed_when_no_dry_run(self, caplog: pytest.LogCaptureFixture) -> None:
        """No dry_run_forecast -> the GHOST_PRE summary still fires but the JSON
        companion is suppressed (nothing to serialize), mirroring the ghost path."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # cap=0 disables the W2 gate: GHOST_PRE-suppression telemetry is the subject.
        await run_agentic_loop("system", "user", [], _config(max_conclude_gate_rejections=0), llm_call=fake_llm)

        assert any("GHOST_PRE:" in m.getMessage() for m in caplog.records)
        assert not any("GHOST_PRE_JSON:" in m.getMessage() for m in caplog.records)

    @pytest.mark.asyncio
    async def test_plan_nudge_cap_soft_continues_without_plan(self, caplog: pytest.LogCaptureFixture) -> None:
        """A driver that never plans hits the plan-nudge cap (2), after which the
        loop soft-continues: the external tool runs un-gated and plan_skipped is
        logged. This is the anti-wedge guard."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        invoked: list[str] = []

        async def search_web(**kwargs: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            invoked.append(kwargs.get("query", ""))
            return ToolOutcome(content_markdown="ran", method="search")  # noqa: ASYNC910

        # Three external-only turns: turns 1 and 2 are plan-rejected (2 nudges ==
        # cap), turn 3 runs un-gated, turn 4 concludes.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("c1", "search_web", {"query": "q1"})]),
                _response(tool_calls=[_tool_call("c2", "search_web", {"query": "q2"})]),
                _response(tool_calls=[_tool_call("c3", "search_web", {"query": "q3"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("search_web", search_web)], _config(max_steps=6), llm_call=fake_llm
        )

        # First two calls were rejected pre-plan; the third ran once the cap
        # forced a soft-continue.
        assert invoked == ["q3"]
        assert result.telemetry.plan_skipped is True
        assert result.telemetry.plan_gaps == 0
        marker = next(m.getMessage() for m in caplog.records if "GAP_FILL_V2:" in m.getMessage())
        assert "plan_skipped=True" in marker
        assert "plan_gaps=0" in marker

    @pytest.mark.asyncio
    async def test_unaddressed_gaps_appear_in_budget_line(self) -> None:
        """After a plan is set, the per-turn budget line lists the plan's gap ids
        as the driver's outstanding work-list (W1 coarse accounting)."""

        async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            return ToolOutcome(content_markdown="ran", method="search")  # noqa: ASYNC910

        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[
                        _plan_call(gaps=[{"id": "gap-a", "question": "q1?"}, {"id": "gap-b", "question": "q2?"}])
                    ]
                ),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # cap=0 disables the W2 gate: the budget-line work-list (W1) is the subject.
        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        # The search result's budget line carries both gap ids.
        search_message = _tool_messages(result)[1]
        budget_line = search_message["content"].splitlines()[-1]
        assert "unaddressed_gaps=[gap-a, gap-b]" in budget_line

    @pytest.mark.asyncio
    async def test_gap_list_capped_at_max_gaps(self) -> None:
        """set_research_plan drops gaps beyond config.max_gaps (the ranked tail),
        keeping the top-N; plan_gaps reflects the kept count."""
        gaps = [{"id": f"g{i}", "question": f"q{i}?"} for i in range(6)]
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=gaps)]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        # cap=0 disables the W2 gate: the max_gaps cap (W1) is the subject here.
        result = await run_agentic_loop(
            "system", "user", [], _config(max_tool_calls=14, max_conclude_gate_rejections=0), llm_call=fake_llm
        )

        # Default max_gaps is 4; only the top 4 survive.
        assert result.telemetry.plan_gaps == 4
        plan_message = _tool_messages(result)[0]
        assert "kept the top 4 of 6 gaps" in plan_message["content"]

    @pytest.mark.asyncio
    async def test_zero_gap_plan_is_rejected_and_w1_gate_stays_armed(self) -> None:
        """F3a: a zero-gap set_research_plan is rejected — research_plan is left
        None so W1's external-tool gate stays armed. A follow-up external tool
        call is still plan-rejected; the empty plan did not open the gate."""
        invoked: list[str] = []

        async def search_web(**kwargs: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            invoked.append(kwargs.get("query", ""))
            return ToolOutcome(content_markdown="ran", method="search")  # noqa: ASYNC910

        # Turn 1: empty-gap plan (rejected). Turn 2: an external tool that must
        # still be plan-gated (only 1 nudge so far, cap not yet hit). Turn 3: a
        # real plan. Turn 4: the external tool finally runs. Turn 5: conclude.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call("p0", gaps=[])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "too soon"})]),
                _response(tool_calls=[_plan_call("p1", gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("s2", "search_web", {"query": "now allowed"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=7, max_conclude_gate_rejections=0),
            llm_call=fake_llm,
        )

        # The empty plan was rejected and did NOT open the gate, so the turn-2
        # search was plan-rejected; only the post-real-plan search ran.
        assert invoked == ["now allowed"]
        plan_reject = _tool_messages(result)[0]
        assert plan_reject["tool_call_id"] == "p0"
        assert "status: error" in plan_reject["content"]
        assert "Research plan rejected" in plan_reject["content"]
        early_search = _tool_messages(result)[1]
        assert early_search["tool_call_id"] == "s1"
        assert "call set_research_plan first" in early_search["content"]
        assert result.telemetry.plan_gaps == 1
        assert result.telemetry.plan_skipped is False

    @pytest.mark.asyncio
    async def test_replan_to_empty_does_not_clobber_a_prior_valid_plan(self) -> None:
        """F3a corollary: once a valid plan is set, a later empty re-plan is
        rejected rather than clobbering it — plan_gaps keeps the real count."""
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call("p0", gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_plan_call("p1", gaps=[])]),
                _response(tool_calls=[_tool_call("done1", "conclude", {"gap_accounting": _accounting("g1")})]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [], _config(max_steps=6, max_conclude_gate_rejections=0), llm_call=fake_llm
        )

        assert result.telemetry.plan_gaps == 1
        replan = _tool_messages(result)[1]
        assert "Research plan rejected" in replan["content"]


def _accounting(*gap_ids: str, actions: str = "searched and fetched the source", status: str = "resolved") -> list:
    """gap_accounting entries for the given gap ids (default actions cite a fetch,
    satisfying the per-gap fetch-floor clause)."""
    return [{"gap_id": gap_id, "actions_taken": actions, "status": status} for gap_id in gap_ids]


class TestConcludeGate:
    """W2: the conclude gate blocks an EARLY conclusion until the driver's
    gap_accounting covers every plan gap, the run made ≥1 external call per gap,
    and the fetch floor is met (top-2 gaps cite a fetch, OR ≥2 fetches/reads).
    A rejection continues the loop (no stop, no banking) and bumps
    conclude_gate_rejections; the cap accepts unconditionally after 2 rejections;
    must_conclude (deadline) and plan_skipped bypass it entirely."""

    @pytest.mark.asyncio
    async def test_reject_missing_gap_id_then_accept_on_complete_accounting(self) -> None:
        """A conclude whose accounting omits a plan gap is rejected (loop
        continues); a follow-up conclude that covers every gap is accepted."""

        async def fetch(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            return ToolOutcome(content_markdown="body", method="rendered")  # noqa: ASYNC910

        # Plan has two gaps; two fetches meet the per-gap call count + fetch floor.
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(
                    tool_calls=[
                        _tool_call("f1", "fetch", {"url": "https://a.example"}),
                        _tool_call("f2", "fetch", {"url": "https://b.example"}),
                    ]
                ),
                # First conclude accounts for only g1 -> rejected.
                _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1")})]),
                # Second conclude accounts for both -> accepted.
                _response(tool_calls=[_tool_call("c2", "conclude", {"gap_accounting": _accounting("g1", "g2")})]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("fetch", fetch)], _config(max_steps=8), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 1
        first_conclude = _tool_messages(result)[-2]
        assert "Conclude rejected" in first_conclude["content"]
        assert "missing entries for gap(s): g2" in first_conclude["content"]
        # The accepted conclude actually stopped the loop.
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_reject_too_few_external_calls(self) -> None:
        """Two plan gaps but only one external tool call: the ≥1-call-per-gap
        invariant is unmet, so the conclude is rejected even with full accounting."""

        async def fetch(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            return ToolOutcome(content_markdown="body", method="rendered")  # noqa: ASYNC910

        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://a.example"})]),
                # cap=0 keeps the rejection observable (the loop would otherwise
                # loop; here it soft-fails after the single scripted conclude).
                _response(tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1", "g2")})]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("fetch", fetch)],
            _config(max_steps=6, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 1
        conclude_message = _tool_messages(result)[-1]
        assert "only 1 external tool call(s) made for 2 plan gap(s)" in conclude_message["content"]
        # Rejected -> the loop did not conclude early; it soft-failed on exhaustion.
        assert result.telemetry.concluded_early is False

    @pytest.mark.asyncio
    async def test_reject_when_both_fetch_floor_clauses_fail(self) -> None:
        """One gap, one external call, but it's a search (no fetch/read) and the
        accounting doesn't cite a fetch: both fetch-floor clauses fail -> reject."""

        async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            return ToolOutcome(content_markdown="snippet", method="search")  # noqa: ASYNC910

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "c1",
                            "conclude",
                            {"gap_accounting": _accounting("g1", actions="searched only, no primary source")},
                        )
                    ]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=6, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 1
        conclude_message = _tool_messages(result)[-1]
        assert "fetch floor unmet" in conclude_message["content"]

    @pytest.mark.asyncio
    async def test_accept_via_per_gap_fetch_clause(self) -> None:
        """Fetch-floor clause (a): the top-ranked gap's accounting cites a fetch,
        even though the run's own fetch count is below the global floor (one
        rendered fetch). The conclude is accepted."""

        async def fetch(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            return ToolOutcome(content_markdown="body", method="rendered")  # noqa: ASYNC910

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(tool_calls=[_tool_call("f1", "fetch", {"url": "https://a.example"})]),
                _response(
                    tool_calls=[
                        _tool_call(
                            "c1",
                            "conclude",
                            {"gap_accounting": _accounting("g1", actions="fetched the primary source PDF")},
                        )
                    ]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("fetch", fetch)], _config(max_steps=6), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_accept_via_global_fetch_count_clause(self) -> None:
        """Fetch-floor clause (b): the run made ≥2 fetches/reads, so a conclude
        whose accounting does NOT cite a fetch per gap is still accepted."""

        async def fetch(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            return ToolOutcome(content_markdown="body", method="rendered")  # noqa: ASYNC910

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(
                    tool_calls=[
                        _tool_call("f1", "fetch", {"url": "https://a.example"}),
                        _tool_call("f2", "fetch", {"url": "https://b.example"}),
                    ]
                ),
                _response(
                    tool_calls=[
                        _tool_call(
                            "c1",
                            "conclude",
                            {"gap_accounting": _accounting("g1", actions="reviewed the search snippets")},
                        )
                    ]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("fetch", fetch)], _config(max_steps=6), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_must_conclude_bypasses_gate(self) -> None:
        """Budget exhaustion (_must_conclude) overrides the gate: a conclude with
        no gap_accounting at all is accepted, and the rejection counter stays 0."""

        async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            return ToolOutcome(content_markdown="snippet", method="search")  # noqa: ASYNC910

        # max_tool_calls=2: plan (1) + one search (2) exhausts the budget, so the
        # conclude turn is in must-conclude mode and the gate is bypassed.
        fake_llm = FakeLlm(
            [
                _response(
                    tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}, {"id": "g2", "question": "q2?"}])]
                ),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                _response(tool_calls=[_tool_call("c1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("search_web", search_web)], _config(max_tool_calls=2), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_rejection_cap_accepts_after_two_rejections(self) -> None:
        """After max_conclude_gate_rejections (2) blocked concludes, the third is
        accepted unconditionally so a pathological driver can't loop forever."""

        async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            return ToolOutcome(content_markdown="snippet", method="search")  # noqa: ASYNC910

        # A single gap, one snippet-only search, and three bare concludes that all
        # fail the fetch floor. The first two are rejected; the third is accepted
        # by the cap.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                _response(
                    tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
                _response(
                    tool_calls=[_tool_call("c2", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
                _response(
                    tool_calls=[_tool_call("c3", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("search_web", search_web)], _config(max_steps=8), llm_call=fake_llm
        )

        assert result.telemetry.conclude_gate_rejections == 2
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_plan_skipped_bypasses_gate(self) -> None:
        """When the plan-nudge cap forced a soft-continue (plan_skipped), there is
        no plan to enforce, so the conclude gate is bypassed."""

        async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            return ToolOutcome(content_markdown="snippet", method="search")  # noqa: ASYNC910

        # Two plan-less external turns hit the nudge cap; turn 3 runs un-gated;
        # turn 4 concludes with no accounting -> accepted (plan_skipped bypass).
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("c1", "search_web", {"query": "q1"})]),
                _response(tool_calls=[_tool_call("c2", "search_web", {"query": "q2"})]),
                _response(tool_calls=[_tool_call("c3", "search_web", {"query": "q3"})]),
                _response(tool_calls=[_tool_call("done1", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system", "user", [_tool_spec("search_web", search_web)], _config(max_steps=8), llm_call=fake_llm
        )

        assert result.telemetry.plan_skipped is True
        assert result.telemetry.conclude_gate_rejections == 0
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_conclude_with_no_plan_is_rejected(self) -> None:
        """F3b: conclude is not plan-gated, so a driver could conclude on turn 1
        with no plan set (plan_skipped still False). The gate now rejects that —
        planning can't be skipped by never calling set_research_plan."""

        async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            return ToolOutcome(content_markdown="snippet", method="search")  # noqa: ASYNC910

        # Turn 1 concludes with no plan -> rejected; turn 2's bare conclude is
        # accepted by the rejection cap (=1), proving the loop can't wedge.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("c1", "conclude")]),
                _response(tool_calls=[_tool_call("c2", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=6, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 1
        first_conclude = _tool_messages(result)[0]
        assert "Conclude rejected" in first_conclude["content"]
        assert "no research plan was set" in first_conclude["content"]
        # The cap accepted the second conclude — no wedge.
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_valid_plan_after_nudge_cap_re_arms_the_gate(self) -> None:
        """F3c: plan_skipped was checked before the plan and never cleared, so a
        valid plan set after the plan-nudge cap fired stayed permanently exempt.
        Now a real plan re-arms the gate even once plan_skipped is True."""

        async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            return ToolOutcome(content_markdown="snippet", method="search")  # noqa: ASYNC910

        # Turns 1-2: plan-less external calls hit the nudge cap (plan_skipped=True).
        # Turn 3: a real plan registers. Turn 4: an early conclude with no
        # gap_accounting must now be rejected (the gate is re-armed despite
        # plan_skipped). Turn 5: the cap (=1) accepts, so no wedge.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_tool_call("c1", "search_web", {"query": "q1"})]),
                _response(tool_calls=[_tool_call("c2", "search_web", {"query": "q2"})]),
                _response(tool_calls=[_plan_call("p1", gaps=[{"id": "g1", "question": "q?"}])]),
                _response(tool_calls=[_tool_call("d1", "conclude")]),
                _response(tool_calls=[_tool_call("d2", "conclude")]),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=8, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.plan_skipped is True
        assert result.telemetry.plan_gaps == 1
        # The gate re-armed: the first post-plan conclude was rejected on the
        # missing gap accounting, not waved through by the stale plan_skipped.
        assert result.telemetry.conclude_gate_rejections == 1
        rejected_conclude = _tool_messages(result)[-2]
        assert "Conclude rejected" in rejected_conclude["content"]
        assert "missing entries for gap(s): g1" in rejected_conclude["content"]
        assert result.telemetry.concluded_early is True

    @pytest.mark.asyncio
    async def test_conclude_gate_rejections_surface_in_marker(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")

        async def search_web(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            return ToolOutcome(content_markdown="snippet", method="search")  # noqa: ASYNC910

        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(tool_calls=[_tool_call("s1", "search_web", {"query": "x"})]),
                # One rejected conclude (fetch floor unmet), then the cap accepts.
                _response(
                    tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
                _response(
                    tool_calls=[_tool_call("c2", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
            ]
        )

        await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("search_web", search_web)],
            _config(max_steps=8, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        marker = next(r.getMessage() for r in caplog.records if "GAP_FILL_V2:" in r.getMessage())
        assert "conclude_gate_rejections=1" in marker


class TestActionsCiteFetch:
    """F2: the per-gap fetch-floor self-report clause matches fetch/read verbs on
    word boundaries. A bare substring test fired on "already"/"spread" (contain
    "read") and on the honest failed-fetch note "could not fetch the source" —
    both cleared the floor with zero real retrievals."""

    def test_narration_words_containing_read_do_not_clear(self) -> None:
        from metaculus_bot.research.agentic.loop import _actions_cite_fetch

        for phrase in (
            "already covered by the briefing",
            "spread the estimates across scenarios",
            "the thread was inconclusive",
            "ready to conclude",
            "readily available in the digest",
        ):
            assert _actions_cite_fetch(phrase) is False, phrase

    def test_failed_or_attempted_fetch_note_does_not_clear(self) -> None:
        from metaculus_bot.research.agentic.loop import _actions_cite_fetch

        for phrase in (
            "could not fetch the primary source",
            "tried to fetch but it 403'd",
            "will fetch next run",
        ):
            assert _actions_cite_fetch(phrase) is False, phrase

    def test_real_fetch_read_mentions_clear(self) -> None:
        from metaculus_bot.research.agentic.loop import _actions_cite_fetch

        for phrase in (
            "read the PDF and confirmed the figure",
            "fetched the release",
            "fetches the dataset and cross-checked",
            "fetching the report resolved it",
            "used read_document on the filing",
            "reads the appendix table",
        ):
            assert _actions_cite_fetch(phrase) is True, phrase


class TestFetchFloorCountsSuccessfulRetrievals:
    """F4: the global fetch-floor clause counts successful fetched-tier
    retrievals (url_best_tier), not fetch/read tool CALLS. per_tool_counts
    increments at accept time regardless of outcome, so two 403'd fetches would
    otherwise clear the floor though they reached nothing — the 131.3 mechanism."""

    @pytest.mark.asyncio
    async def test_two_blocked_fetches_do_not_clear_the_global_floor(self) -> None:
        async def fetch(**_: Any) -> ToolOutcome:  # noqa: ASYNC124 - async-by-contract test handler
            # Both fetches 403 (status=blocked): no fetched tier is granted, so
            # url_best_tier stays empty and the global floor is unmet.
            return ToolOutcome(content_markdown="Fetch blocked with HTTP 403.", method="plain", status="blocked")  # noqa: ASYNC910

        # One gap; the accounting cites "searched" (no fetch verb), so the per-gap
        # clause fails too. With two fetch CALLS but zero successful retrievals,
        # the conclude must be rejected on the fetch floor.
        fake_llm = FakeLlm(
            [
                _response(tool_calls=[_plan_call(gaps=[{"id": "g1", "question": "q1?"}])]),
                _response(
                    tool_calls=[
                        _tool_call("f1", "fetch", {"url": "https://a.example"}),
                        _tool_call("f2", "fetch", {"url": "https://b.example"}),
                    ]
                ),
                _response(
                    tool_calls=[_tool_call("c1", "conclude", {"gap_accounting": _accounting("g1", actions="searched")})]
                ),
            ]
        )

        result = await run_agentic_loop(
            "system",
            "user",
            [_tool_spec("fetch", fetch)],
            _config(max_steps=6, max_conclude_gate_rejections=1),
            llm_call=fake_llm,
        )

        assert result.telemetry.conclude_gate_rejections == 1
        conclude_message = _tool_messages(result)[-1]
        assert "fetch floor unmet" in conclude_message["content"]
        # made 0 — the two calls happened but neither retrieved a primary source.
        assert "made 0" in conclude_message["content"]


class TestNormalizeUrl:
    def test_lowercases_scheme_and_host_only(self) -> None:
        from metaculus_bot.research.agentic.loop import _normalize_url

        assert _normalize_url("HTTPS://Example.COM/Path") == "https://example.com/Path"

    def test_strips_trailing_slash_fragment_and_trackers(self) -> None:
        from metaculus_bot.research.agentic.loop import _normalize_url

        assert (
            _normalize_url("https://example.com/report/?utm_source=x&id=7&gclid=abc#frag")
            == "https://example.com/report?id=7"
        )

    def test_rejects_non_http_scheme(self) -> None:
        from metaculus_bot.research.agentic.loop import _normalize_url

        assert _normalize_url("ftp://example.com/file") is None
        assert _normalize_url("not a url") is None
