"""Integration tests for the gap-fill v2 seam (research/agentic_gap_fill.py).

Covers the seam contract (flag-gating, benchmarking hard-off, soft-fail) plus
the orchestrator-level wiring (v1+v2 concurrent sections, archive payload).
Everything is mocked — zero LLM/network calls.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import GeneralLlm

from metaculus_bot.research.agentic.types import ToolOutcome, ToolSpec
from metaculus_bot.research.agentic_gap_fill import run_gap_fill_v2
from metaculus_bot.research.orchestrator import ResearchOrchestrator
from metaculus_bot.research.persistence import ResearchPersistenceWriter
from tests.pipeline_test_helpers import (
    make_real_binary_question,
    make_real_mc_question,
    make_real_numeric_question,
)

BUNDLE = "## News Articles (AskNews)\nSome first-pass research prose about unemployment."


# --- fake litellm-shaped responses (mirrors tests/test_agentic_loop.py) ---


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


def _tool_call(tool_id: str, name: str, arguments: dict[str, Any]) -> _FakeToolCall:
    return _FakeToolCall(tool_id, _FakeFunction(name=name, arguments=json.dumps(arguments)))


def _response(content: str = "", tool_calls: list[_FakeToolCall] | None = None) -> _FakeResponse:
    return _FakeResponse(choices=[_FakeChoice(message=_FakeMessage(content=content, tool_calls=tool_calls))])


class FakeLlm:
    """Scripted llm_call double; records every invocation."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    async def __call__(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> Any:
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("FakeLlm ran out of scripted responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


_FINDING = {
    "claim": "BLS reported the unemployment rate at 4.1% for June 2026.",
    "source_url": "https://www.bls.gov/news.release/empsit.nr0.htm",
    "quote": "The unemployment rate was 4.1 percent in June.",
    "date": "2026-07-03",
    "retrieved_how": "fetch of BLS release",
    "topic": "labor-market",
}

_GHOST_BLOCK = '```json\n{"forecast_type": "binary", "posterior_prob": 0.12}\n```'


def _happy_path_llm() -> FakeLlm:
    """One search step, then conclude with a finding, then the ghost turn."""
    return FakeLlm(
        [
            _response(tool_calls=[_tool_call("c1", "search_web", {"query": "BLS unemployment June 2026"})]),
            _response(tool_calls=[_tool_call("c2", "conclude", {"final_findings": [_FINDING]})]),
            _response(content=_GHOST_BLOCK),
        ]
    )


def _fake_tools() -> list[ToolSpec]:
    async def _search(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="result: BLS release found", status="ok")

    return [
        ToolSpec(
            name="search_web",
            description="fake exa",
            parameters={"type": "object", "properties": {}, "additionalProperties": True},
            handler=_search,
            timeout_s=1.0,
        )
    ]


def _patch_loop_internals(fake_llm: FakeLlm):
    """Patch the seam's LLM binding and tool construction (no network, no keys)."""
    return (
        patch("metaculus_bot.research.agentic.loop.build_default_llm_call", return_value=fake_llm),
        patch("metaculus_bot.research.agentic_gap_fill.build_gap_fill_tools", return_value=_fake_tools()),
    )


class TestRunGapFillV2Seam:
    @pytest.mark.asyncio
    async def test_flag_off_returns_empty_with_zero_llm_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GAP_FILL_V2_ENABLED", raising=False)
        fake_llm = _happy_path_llm()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch, tools_patch:
            result = await run_gap_fill_v2(make_real_binary_question(), BUNDLE, is_benchmarking=False)
        assert result == ""
        assert fake_llm.calls == []

    @pytest.mark.asyncio
    async def test_benchmarking_returns_empty_with_zero_llm_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        fake_llm = _happy_path_llm()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch, tools_patch:
            result = await run_gap_fill_v2(make_real_binary_question(), BUNDLE, is_benchmarking=True)
        assert result == ""
        assert fake_llm.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "question_factory",
        [make_real_binary_question, make_real_mc_question, make_real_numeric_question],
        ids=["binary", "mc", "numeric"],
    )
    async def test_happy_path_returns_findings_and_emits_markers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        question_factory,
    ) -> None:
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        caplog.set_level(logging.INFO, logger="metaculus_bot.research.agentic.loop")
        fake_llm = _happy_path_llm()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch, tools_patch:
            result = await run_gap_fill_v2(question_factory(), BUNDLE, is_benchmarking=False)

        assert "## Agentic Research Findings" in result
        assert _FINDING["claim"] in result
        messages = [record.getMessage() for record in caplog.records]
        assert any("GAP_FILL_V2:" in m for m in messages)
        assert any("GHOST_FORECAST:" in m for m in messages)
        # log_prefix carries the question reference on the marker lines.
        assert any("question=https://www.metaculus.com/questions/" in m for m in messages if "GAP_FILL_V2:" in m)

    @pytest.mark.asyncio
    async def test_user_brief_embeds_question_and_bundle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The frozen prefix must carry resolution criteria, the real template, and the bundle."""
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        question = make_real_numeric_question()
        fake_llm = _happy_path_llm()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch, tools_patch:
            await run_gap_fill_v2(question, BUNDLE, is_benchmarking=False)

        first_messages = fake_llm.calls[0]
        assert first_messages[0]["role"] == "system"
        assert "research analyst" in first_messages[0]["content"]
        brief = first_messages[1]["content"]
        assert question.resolution_criteria in brief
        assert BUNDLE in brief
        # Real numeric bounds/units render in the skeleton (OPEN_BOUND_PILING lesson).
        assert "Units: %" in brief
        assert "The upper bound is open" in brief

    @pytest.mark.asyncio
    async def test_seam_exception_soft_fails_to_empty(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        caplog.set_level(logging.ERROR, logger="metaculus_bot.research.agentic_gap_fill")
        with patch(
            "metaculus_bot.research.agentic_gap_fill.build_gap_fill_tools",
            side_effect=RuntimeError("tool construction exploded"),
        ):
            result = await run_gap_fill_v2(make_real_binary_question(), BUNDLE, is_benchmarking=False)
        assert result == ""
        assert any("Gap-fill v2 seam failed" in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_archive_sink_receives_transcript_and_telemetry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        captured: list[dict] = []
        fake_llm = _happy_path_llm()
        llm_patch, tools_patch = _patch_loop_internals(fake_llm)
        with llm_patch, tools_patch:
            await run_gap_fill_v2(
                make_real_binary_question(),
                BUNDLE,
                is_benchmarking=False,
                archive_sink=captured.append,
            )
        assert len(captured) == 1
        payload = captured[0]
        assert payload["telemetry"]["findings_count"] == 1
        assert payload["telemetry"]["tool_calls"] == 2  # search_web + conclude
        roles = [
            message["role"] for message in payload["transcript"]
        ]  # HARNESS-SCAN-EXEMPT-object-explosion — tiny transcript list, not a DataFrame
        assert roles[0] == "system"
        assert "tool" in roles


@pytest.fixture
def mock_llm() -> GeneralLlm:
    return GeneralLlm(model="test/model", temperature=0.0)


class TestOrchestratorBothFlags:
    """Orchestrator-level wiring: v1 and v2 sections coexist; archive carries the trace."""

    @pytest.mark.asyncio
    async def test_both_sections_render_and_v2_appends_after_v1(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(
            # Long enough to clear GAP_FILL_MIN_RESEARCH_CHARS (200) so the v1 gate opens.
            return_value="First-pass research prose long enough to pass the gap-fill min-chars gate. " * 4
        )
        v2_section = "## Agentic Research Findings\n\n### labor-market\n\nClaim: BLS says 4.1%."

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass",
                new_callable=AsyncMock,
                return_value="v1 gap-fill addendum text",
            ) as v1_mock,
            patch(
                "metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2",
                new_callable=AsyncMock,
                return_value=v2_section,
            ) as v2_mock,
        ):
            research = await orch.run_research(make_real_binary_question())

        v1_mock.assert_awaited_once()
        v2_mock.assert_awaited_once()
        assert "## Targeted Gap-Fill (second pass)" in research
        assert "## Agentic Research Findings" in research
        assert research.index("## Targeted Gap-Fill (second pass)") < research.index("## Agentic Research Findings")

    @pytest.mark.asyncio
    async def test_v2_flag_off_leaves_v1_path_untouched(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.delenv("GAP_FILL_V2_ENABLED", raising=False)

        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(
            # Long enough to clear GAP_FILL_MIN_RESEARCH_CHARS (200) so the v1 gate opens.
            return_value="First-pass research prose long enough to pass the gap-fill min-chars gate. " * 4
        )

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass",
                new_callable=AsyncMock,
                return_value="v1 gap-fill addendum text",
            ),
            patch(
                "metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2",
                new_callable=AsyncMock,
            ) as v2_mock,
        ):
            research = await orch.run_research(make_real_binary_question())

        # The orchestrator gates on the v2 flag before awaiting the seam, so
        # run_gap_fill_v2 is never called when the flag is off.
        assert v2_mock.await_count == 0
        assert "## Targeted Gap-Fill (second pass)" in research
        assert "## Agentic Research Findings" not in research

    @pytest.mark.asyncio
    async def test_archive_payload_carries_gap_fill_v2_key(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

        captured: dict = {}

        def sink(**kwargs) -> None:  # noqa: ANN003
            captured.update(kwargs)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            allow_research_fallback=False,
            research_sink=sink,
        )
        provider = AsyncMock(return_value="research prose")
        trace = {"transcript": [{"role": "system", "content": "x"}], "telemetry": {"steps": 2}}

        async def _fake_v2(question, bundle, *, is_benchmarking, archive_sink=None):
            if archive_sink is not None:
                archive_sink(trace)
            return "## Agentic Research Findings\n\nClaim: something."

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch("metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2", side_effect=_fake_v2),
        ):
            await orch.run_research(make_real_binary_question())

        assert captured["gap_fill_v2"] == trace

    @pytest.mark.asyncio
    async def test_archive_payload_omits_key_when_v2_off(
        self, mock_llm: GeneralLlm, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
        monkeypatch.delenv("GAP_FILL_V2_ENABLED", raising=False)

        captured: dict = {}

        def sink(**kwargs) -> None:  # noqa: ANN003
            captured.update(kwargs)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            allow_research_fallback=False,
            research_sink=sink,
        )
        provider = AsyncMock(return_value="research prose")
        with patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]):
            await orch.run_research(make_real_binary_question())

        assert captured["gap_fill_v2"] is None

    def test_persistence_writer_serializes_gap_fill_v2(self, tmp_path) -> None:
        """Writer round-trips the v2 trace and omits the key when absent."""
        writer = ResearchPersistenceWriter(run_mode="tournament", tournament_id="t", run_id="r")
        trace = {"transcript": [{"role": "system", "content": "x"}], "telemetry": {"steps": 3}}
        writer.record(
            qid=1,
            page_url="https://www.metaculus.com/questions/1/",
            question_text="Q?",
            research_text="research",
            providers_used=["asknews"],
            gap_fill_used=False,
            gap_fill_v2=trace,
        )
        writer.record(
            qid=2,
            page_url="https://www.metaculus.com/questions/2/",
            question_text="Q2?",
            research_text="research",
            providers_used=["asknews"],
            gap_fill_used=False,
        )
        out = writer.flush(output_dir=str(tmp_path))
        assert out is not None
        records = [json.loads(line) for line in out.read_text().strip().splitlines()]
        assert records[0]["gap_fill_v2"] == trace
        assert "gap_fill_v2" not in records[1]
