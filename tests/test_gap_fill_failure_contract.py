"""The analyzer must enforce its grades and report failures without losing good research."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metaculus_bot.research.targeted import run_gap_fill_pass
from tests.test_gap_fill_pass import MockQuestion, _gap, _patch_resolver, _q


@pytest.mark.asyncio
async def test_analyzer_requests_strict_grades_and_resolves_valid_output() -> None:
    gap = _gap("What is the latest dated reading of the official tracker?")
    analyzer = MagicMock(invoke=AsyncMock(return_value=json.dumps({"gaps": [gap]})))
    resolver = AsyncMock(return_value="The official tracker reports 17 as of September 10.")
    errors: list[BaseException] = []
    with (
        patch("metaculus_bot.fallback_openrouter.build_llm_with_openrouter_fallback", return_value=analyzer) as builder,
        _patch_resolver(resolver),
    ):
        result = await run_gap_fill_pass(_q(MockQuestion()), "First-pass research", on_error=errors.append)

    response_format = builder.call_args.kwargs["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    gap_schema = schema["$defs"]["GapCandidate"]
    assert set(gap_schema["required"]) == {
        "gap",
        "search_query",
        "why_matters",
        "answerable_now",
        "already_in_first_pass",
        "same_need_as",
    }
    assert gap_schema["properties"]["answerable_now"]["type"] == "boolean"
    assert gap_schema["properties"]["already_in_first_pass"]["type"] == "boolean"
    assert gap_schema["additionalProperties"] is False
    assert schema["additionalProperties"] is False
    assert builder.call_args.kwargs["extra_body"]["provider"]["require_parameters"] is True
    resolver.assert_awaited_once()
    assert "reports 17" in result
    assert errors == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not JSON",
        "null",
        "[]",
        "{}",
        '{"gaps": null}',
        json.dumps(
            {
                "gaps": [
                    {
                        "gap": f"Missing fact {index}",
                        "search_query": f"official fact {index}",
                        "why_matters": "Changes forecast",
                    }
                    for index in range(4)
                ]
            }
        ),
    ],
)
async def test_invalid_analyzer_output_reports_failure_without_searching(raw: str) -> None:
    """Includes the smoke run's four ungraded objects, through the real parser and triage."""
    analyzer = MagicMock(invoke=AsyncMock(return_value=raw))
    resolver = AsyncMock()
    errors: list[BaseException] = []
    with (
        patch("metaculus_bot.fallback_openrouter.build_llm_with_openrouter_fallback", return_value=analyzer),
        _patch_resolver(resolver),
    ):
        result = await run_gap_fill_pass(_q(MockQuestion()), "First-pass research", on_error=errors.append)

    assert result == ""
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    resolver.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("gaps", [[], [_gap("Already dated fact", already_in_first_pass=True)]])
async def test_legitimate_empty_addendum_is_not_failure(gaps: list[dict]) -> None:
    analyzer = MagicMock(invoke=AsyncMock(return_value=json.dumps({"gaps": gaps})))
    errors: list[BaseException] = []
    with patch("metaculus_bot.fallback_openrouter.build_llm_with_openrouter_fallback", return_value=analyzer):
        result = await run_gap_fill_pass(_q(MockQuestion()), "First-pass research", on_error=errors.append)
    assert result == ""
    assert errors == []


@pytest.mark.asyncio
async def test_partial_failure_reports_once_and_preserves_successful_search() -> None:
    gaps = [_gap("Good fact"), _gap("Unavailable fact"), {"gap": "Ungraded fact"}]
    errors: list[BaseException] = []
    resolver = AsyncMock(side_effect=["Verified dated reading: 17.", TimeoutError("resolver timed out")])
    with (
        patch("metaculus_bot.research.targeted._run_analyzer", return_value=gaps),
        _patch_resolver(resolver),
    ):
        result = await run_gap_fill_pass(_q(MockQuestion()), "First-pass research", on_error=errors.append)
    assert "Verified dated reading: 17." in result
    assert len(errors) == 1
    assert resolver.await_count == 2


@pytest.mark.asyncio
async def test_analyzer_timeout_reports_failure() -> None:
    errors: list[BaseException] = []
    with patch("metaculus_bot.research.targeted._run_analyzer", side_effect=TimeoutError("analyzer timed out")):
        result = await run_gap_fill_pass(_q(MockQuestion()), "First-pass research", on_error=errors.append)
    assert result == ""
    assert len(errors) == 1
    assert isinstance(errors[0], TimeoutError)


@pytest.mark.asyncio
async def test_cancelling_the_pass_cancels_searches_without_reporting_a_failure() -> None:
    started = asyncio.Event()
    errors: list[BaseException] = []

    async def pending_search(_prompt: str) -> str:
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    with (
        patch("metaculus_bot.research.targeted._run_analyzer", return_value=[_gap("Current reading")]),
        _patch_resolver(AsyncMock(side_effect=pending_search)),
    ):
        task = asyncio.create_task(run_gap_fill_pass(_q(MockQuestion()), "First pass", on_error=errors.append))
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert errors == []
