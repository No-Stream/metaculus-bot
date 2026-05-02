"""Tests for the second-pass gap-fill pipeline in ``targeted_research``.

Mocks ``_run_analyzer`` at the module level and ``invoke_gemini_grounded`` (the
per-gap search fn). No live API calls.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import MetaculusQuestion

from metaculus_bot.constants import GAP_FILL_MAX_GAPS
from metaculus_bot.targeted_research import _parse_gap_list, run_gap_fill_pass


@dataclass
class MockQuestion:
    """Minimal stand-in for MetaculusQuestion for gap-fill tests.

    Duck-typed to match the attribute access in ``run_gap_fill_pass``: it reads
    ``question_text``, ``resolution_criteria``, and ``fine_print`` via ``getattr``.
    """

    question_text: str = "Will X happen by 2026?"
    resolution_criteria: str | None = "Resolves YES if X happens before Dec 31, 2026."
    fine_print: str | None = "See bls.gov for data."
    id_of_question: int = 42
    page_url: str = "https://example.com/q/42"

    _unused: dict[str, str] = field(default_factory=dict)


def _q(mock: MockQuestion) -> MetaculusQuestion:
    """Cast a MockQuestion to MetaculusQuestion for static type checkers.

    The real code only uses duck-typed attribute access, so this is a runtime
    no-op; it exists solely to keep Pyright happy about the function signature.
    """
    return cast(MetaculusQuestion, mock)


# ---------------------------------------------------------------------------
# _parse_gap_list unit tests
# ---------------------------------------------------------------------------


class TestParseGapList:
    """Cover the various shapes of analyzer output the parser must tolerate."""

    def test_empty_string_returns_empty(self) -> None:
        assert _parse_gap_list("") == []
        assert _parse_gap_list("   \n  ") == []

    def test_plain_valid_json(self) -> None:
        raw = '{"gaps": [{"gap": "g1", "why_matters": "wm1", "search_query": "sq1"}]}'
        out = _parse_gap_list(raw)

        assert len(out) == 1
        assert out[0]["gap"] == "g1"
        assert out[0]["search_query"] == "sq1"
        assert out[0]["why_matters"] == "wm1"

    def test_json_fenced_code_block(self) -> None:
        raw = '```json\n{"gaps": [{"gap": "fenced gap", "why_matters": "wm", "search_query": "sq"}]}\n```'
        out = _parse_gap_list(raw)

        assert len(out) == 1
        assert out[0]["gap"] == "fenced gap"

    def test_json_with_trailing_commentary(self) -> None:
        raw = 'Here is the output:\n{"gaps": [{"gap": "g", "why_matters": "wm", "search_query": "sq"}]}\n\nHope that helps!'
        out = _parse_gap_list(raw)

        assert len(out) == 1
        assert out[0]["gap"] == "g"

    def test_malformed_json_returns_empty(self) -> None:
        out = _parse_gap_list("not json at all, just words")

        assert out == []

    def test_gap_missing_gap_field_is_dropped(self) -> None:
        raw = '{"gaps": [{"why_matters": "wm", "search_query": "sq"}]}'
        out = _parse_gap_list(raw)

        assert out == []

    def test_gap_missing_search_query_falls_back_to_gap_text(self) -> None:
        raw = '{"gaps": [{"gap": "my gap text", "why_matters": "wm"}]}'
        out = _parse_gap_list(raw)

        assert len(out) == 1
        assert out[0]["gap"] == "my gap text"
        assert out[0]["search_query"] == "my gap text"

    def test_clips_at_max_gaps(self) -> None:
        """With max_gaps=5, a 7-gap payload is clipped to 5 — this is the real clip."""
        gap_objs = [{"gap": f"g{i}", "search_query": f"q{i}", "why_matters": f"wm{i}"} for i in range(7)]
        raw = json.dumps({"gaps": gap_objs})

        out = _parse_gap_list(raw, max_gaps=5)

        assert len(out) == 5
        # Order is preserved: first five, not last five.
        assert [g["gap"] for g in out] == ["g0", "g1", "g2", "g3", "g4"]

    def test_no_clip_when_max_gaps_none(self) -> None:
        """Default max_gaps=None returns every parsed gap; clipping is opt-in."""
        gap_objs = [{"gap": f"g{i}", "search_query": f"q{i}", "why_matters": f"wm{i}"} for i in range(7)]
        raw = json.dumps({"gaps": gap_objs})

        out = _parse_gap_list(raw)

        assert len(out) == 7


# ---------------------------------------------------------------------------
# run_gap_fill_pass integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_gaps_returns_empty_string() -> None:
    """Analyzer returns no gaps → addendum is "" and the search fn is never called."""
    question = MockQuestion()

    fake_search = AsyncMock(return_value="should not be called")
    with (
        patch("metaculus_bot.targeted_research._run_analyzer", AsyncMock(return_value=[])),
        patch("metaculus_bot.targeted_research.invoke_gemini_grounded", fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()


@pytest.mark.asyncio
async def test_two_gaps_run_in_parallel() -> None:
    """Analyzer returns 2 gaps → 2 grounded calls that run concurrently.

    The slow-first, fast-second ordering guards against a sequential refactor: if the
    code ever did ``for gap in gaps: await invoke_gemini_grounded(...)``, gap 1 would
    finish before gap 2. With ``asyncio.gather``, the short sleep wins the race.
    """
    question = MockQuestion()

    gaps = [
        {"gap": "gap one text", "search_query": "q1", "why_matters": "wm1"},
        {"gap": "gap two text", "search_query": "q2", "why_matters": "wm2"},
    ]

    completion_order: list[str] = []

    async def search_side_effect(prompt: str) -> str:
        if "q1" in prompt:
            await asyncio.sleep(0.1)
            completion_order.append("q1")
            return "result for gap 1"
        if "q2" in prompt:
            await asyncio.sleep(0.01)
            completion_order.append("q2")
            return "result for gap 2"
        raise AssertionError(f"unexpected prompt: {prompt!r}")

    fake_search = AsyncMock(side_effect=search_side_effect)

    with (
        patch("metaculus_bot.targeted_research._run_analyzer", AsyncMock(return_value=gaps)),
        patch("metaculus_bot.targeted_research.invoke_gemini_grounded", fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert fake_search.await_count == 2
    assert "### Gap 1: gap one text" in out
    assert "### Gap 2: gap two text" in out
    assert "result for gap 1" in out
    assert "result for gap 2" in out
    # Concurrency check: the fast (q2) search must finish before the slow (q1) one.
    # A sequential implementation would always produce ["q1", "q2"].
    assert completion_order == ["q2", "q1"]


@pytest.mark.asyncio
async def test_run_gap_fill_pass_does_not_clip_analyzer_output() -> None:
    """``run_gap_fill_pass`` itself must not truncate the analyzer's gap list.

    Clipping lives inside ``_parse_gap_list`` (see ``TestParseGapList.test_clips_at_max_gaps``).
    By patching ``_run_analyzer`` to return more than ``GAP_FILL_MAX_GAPS`` gaps we bypass the
    parser; every gap the analyzer yields must trigger a grounded search.
    """
    question = MockQuestion()

    oversized_count = GAP_FILL_MAX_GAPS + 2
    gaps = [{"gap": f"gap {i}", "search_query": f"q{i}", "why_matters": f"wm{i}"} for i in range(oversized_count)]
    fake_search = AsyncMock(side_effect=[f"res{i}" for i in range(oversized_count)])

    with (
        patch("metaculus_bot.targeted_research._run_analyzer", AsyncMock(return_value=gaps)),
        patch("metaculus_bot.targeted_research.invoke_gemini_grounded", fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert fake_search.await_count == oversized_count
    # Every gap — including those past GAP_FILL_MAX_GAPS — shows up in the addendum.
    assert f"### Gap {oversized_count}:" in out
    assert f"res{oversized_count - 1}" in out


@pytest.mark.asyncio
async def test_malformed_analyzer_output_soft_fails() -> None:
    """If the analyzer returns no parseable gaps, we get "" and no searches run.

    This simulates ``_parse_gap_list`` returning [] inside ``_run_analyzer``.
    """
    question = MockQuestion()

    fake_search = AsyncMock(return_value="should not be called")
    with (
        patch("metaculus_bot.targeted_research._run_analyzer", AsyncMock(return_value=[])),
        patch("metaculus_bot.targeted_research.invoke_gemini_grounded", fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()


@pytest.mark.asyncio
async def test_partial_search_failure_returns_successful_results(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """3 gaps, middle one raises → addendum contains gap 1 and gap 3 only; gap 2 logged."""
    question = MockQuestion()

    gaps = [
        {"gap": "g1", "search_query": "q1", "why_matters": "wm1"},
        {"gap": "g2", "search_query": "q2", "why_matters": "wm2"},
        {"gap": "g3", "search_query": "q3", "why_matters": "wm3"},
    ]

    async def search_side_effect(prompt: str) -> str:
        await asyncio.sleep(0)
        if "q2" in prompt:
            raise RuntimeError("boom for gap 2")
        if "q1" in prompt:
            return "result 1"
        if "q3" in prompt:
            return "result 3"
        raise AssertionError(f"unexpected prompt: {prompt!r}")

    with (
        patch("metaculus_bot.targeted_research._run_analyzer", AsyncMock(return_value=gaps)),
        patch("metaculus_bot.targeted_research.invoke_gemini_grounded", side_effect=search_side_effect),
        caplog.at_level(logging.WARNING, logger="metaculus_bot.targeted_research"),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert "### Gap 1: g1" in out
    assert "result 1" in out
    assert "### Gap 3: g3" in out
    assert "result 3" in out
    # Gap 2 must NOT appear in the addendum.
    assert "### Gap 2:" not in out
    # And the failure was logged as a warning.
    assert any("gap #2" in rec.message.lower() or "boom" in rec.message.lower() for rec in caplog.records)


@pytest.mark.asyncio
async def test_all_searches_fail_returns_empty(caplog: pytest.LogCaptureFixture) -> None:
    """All grounded searches raise → addendum is "" and each failure logs a warning.

    The soft-fail contract depends on failures being observable in logs, so we assert
    exactly one warning per failed gap with the "gap #<n>" marker.
    """
    question = MockQuestion()

    gaps = [
        {"gap": "g1", "search_query": "q1", "why_matters": "wm1"},
        {"gap": "g2", "search_query": "q2", "why_matters": "wm2"},
    ]
    fake_search = AsyncMock(side_effect=RuntimeError("all fail"))

    with (
        patch("metaculus_bot.targeted_research._run_analyzer", AsyncMock(return_value=gaps)),
        patch("metaculus_bot.targeted_research.invoke_gemini_grounded", fake_search),
        caplog.at_level(logging.WARNING, logger="metaculus_bot.targeted_research"),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""

    gap_failure_records = [
        rec for rec in caplog.records if rec.levelno == logging.WARNING and "gap #" in rec.message.lower()
    ]
    assert len(gap_failure_records) == len(gaps)
    assert any("gap #1" in rec.message.lower() for rec in gap_failure_records)
    assert any("gap #2" in rec.message.lower() for rec in gap_failure_records)


@pytest.mark.asyncio
async def test_benchmarking_flag_threaded_to_analyzer_and_searches() -> None:
    """is_benchmarking=True: analyzer + each gap search receives the benchmarking warning."""
    question = MockQuestion()

    gaps = [
        {"gap": "g1", "search_query": "q1", "why_matters": "wm1"},
        {"gap": "g2", "search_query": "q2", "why_matters": "wm2"},
    ]
    fake_analyzer = AsyncMock(return_value=gaps)
    fake_search = AsyncMock(side_effect=["r1", "r2"])

    with (
        patch("metaculus_bot.targeted_research._run_analyzer", fake_analyzer),
        patch("metaculus_bot.targeted_research.invoke_gemini_grounded", fake_search),
    ):
        await run_gap_fill_pass(_q(question), "first-pass", is_benchmarking=True)

    # Analyzer was called with is_benchmarking=True.
    fake_analyzer.assert_awaited_once()
    assert fake_analyzer.call_args.kwargs["is_benchmarking"] is True

    # Each per-gap search prompt includes the benchmarking warning string.
    for call in fake_search.call_args_list:
        prompt = call.args[0]
        assert "benchmarking run" in prompt


@pytest.mark.asyncio
async def test_analyzer_timeout_returns_empty() -> None:
    """_run_analyzer raising TimeoutError → soft-fail with "" and no search calls."""
    question = MockQuestion()

    fake_search = AsyncMock()
    with (
        patch("metaculus_bot.targeted_research._run_analyzer", AsyncMock(side_effect=asyncio.TimeoutError())),
        patch("metaculus_bot.targeted_research.invoke_gemini_grounded", fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()


@pytest.mark.asyncio
async def test_analyzer_missing_key_returns_empty() -> None:
    """_run_analyzer raising ValueError (e.g., missing GOOGLE_API_KEY) → "" and no searches."""
    question = MockQuestion()

    fake_search = AsyncMock()
    with (
        patch(
            "metaculus_bot.targeted_research._run_analyzer",
            AsyncMock(side_effect=ValueError("GOOGLE_API_KEY must be set")),
        ),
        patch("metaculus_bot.targeted_research.invoke_gemini_grounded", fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()


@pytest.mark.asyncio
async def test_analyzer_gemini_api_error_returns_empty() -> None:
    """google-genai APIError from the analyzer → soft-fail with "" (no raising out).

    Covers ClientError (4xx) and ServerError (5xx) since both subclass APIError.
    """
    import httpx
    from google.genai.errors import APIError

    question = MockQuestion()

    # APIError signature: APIError(code, response_json, response=None). We fake a 500 payload.
    response = httpx.Response(status_code=500, text='{"error": {"message": "internal"}}')
    fake_exc = APIError(code=500, response_json={"error": {"message": "internal"}}, response=response)

    fake_search = AsyncMock()
    with (
        patch("metaculus_bot.targeted_research._run_analyzer", AsyncMock(side_effect=fake_exc)),
        patch("metaculus_bot.targeted_research.invoke_gemini_grounded", fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()


@pytest.mark.asyncio
async def test_analyzer_httpx_error_returns_empty() -> None:
    """Raw httpx.HTTPError from the analyzer → soft-fail with "" (covers mid-SDK network failures)."""
    import httpx

    question = MockQuestion()
    fake_search = AsyncMock()
    with (
        patch(
            "metaculus_bot.targeted_research._run_analyzer",
            AsyncMock(side_effect=httpx.ConnectError("connection refused")),
        ),
        patch("metaculus_bot.targeted_research.invoke_gemini_grounded", fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()


@pytest.mark.asyncio
async def test_analyzer_os_error_returns_empty() -> None:
    """OSError (DNS / socket) from the analyzer → soft-fail with ""."""
    question = MockQuestion()
    fake_search = AsyncMock()
    with (
        patch(
            "metaculus_bot.targeted_research._run_analyzer",
            AsyncMock(side_effect=OSError("name resolution failed")),
        ),
        patch("metaculus_bot.targeted_research.invoke_gemini_grounded", fake_search),
    ):
        out = await run_gap_fill_pass(_q(question), "first-pass research")

    assert out == ""
    fake_search.assert_not_called()


# ---------------------------------------------------------------------------
# Prompt wiring (resolution_criteria / fine_print thread through)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyzer_receives_resolution_criteria_and_fine_print() -> None:
    """The analyzer call passes through resolution_criteria + fine_print from the question."""
    question = MockQuestion(
        question_text="Will bitcoin exceed $200k?",
        resolution_criteria="Resolves YES if BTC USD > 200k on Jan 1 2027.",
        fine_print="Data source: Coinbase Pro BTC-USD.",
    )

    fake_analyzer = AsyncMock(return_value=[])
    fake_search = AsyncMock()
    with (
        patch("metaculus_bot.targeted_research._run_analyzer", fake_analyzer),
        patch("metaculus_bot.targeted_research.invoke_gemini_grounded", fake_search),
    ):
        await run_gap_fill_pass(_q(question), "some first-pass research")

    # _run_analyzer is called with (question, first_pass_research, is_benchmarking=...)
    fake_analyzer.assert_awaited_once()
    positional_q = fake_analyzer.call_args.args[0]
    assert positional_q.resolution_criteria == "Resolves YES if BTC USD > 200k on Jan 1 2027."
    assert positional_q.fine_print == "Data source: Coinbase Pro BTC-USD."
