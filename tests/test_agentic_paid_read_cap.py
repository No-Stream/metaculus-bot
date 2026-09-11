from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from metaculus_bot.constants import GOOGLE_API_KEY_ENV, RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS
from metaculus_bot.research.agentic import local_document, tools


def _leave_only_the_paid_reader(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    monkeypatch.setenv(GOOGLE_API_KEY_ENV, "test-key")
    monkeypatch.setattr(
        tools,
        "_acquire_local_document",
        AsyncMock(return_value=local_document.HeldDocument()),
    )
    monkeypatch.setattr(tools, "_url_context_robots_skip", AsyncMock(return_value=False))
    reader = MagicMock(return_value=("The retrieved document answers the question.", 1, ["success"]))
    monkeypatch.setattr(tools, "_run_document_read_sync", reader)
    return reader


@pytest.mark.asyncio
async def test_shared_question_context_caps_concurrent_paid_document_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = _leave_only_the_paid_reader(monkeypatch)
    question_ctx = tools.question_ladder_context()

    outcomes = await asyncio.gather(
        *[
            tools.read_document(f"https://example.com/document-{index}", "What does it say?", ctx=question_ctx)
            for index in range(RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS + 1)
        ]
    )

    assert reader.call_count == RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS
    assert [outcome.status for outcome in outcomes].count("ok") == RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS
    denied = [outcome for outcome in outcomes if outcome.status == "error"]
    assert len(denied) == 1
    assert denied[0].method == "document"
    assert "paid document-read limit is exhausted" in denied[0].content_markdown


@pytest.mark.asyncio
async def test_direct_document_reads_without_a_question_context_get_independent_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _leave_only_the_paid_reader(monkeypatch)

    outcomes = await asyncio.gather(
        tools.read_document("https://example.com/first", "First ask"),
        tools.read_document("https://example.com/second", "Second ask"),
    )

    assert reader.call_count == 2
    assert [outcome.status for outcome in outcomes] == ["ok", "ok"]
