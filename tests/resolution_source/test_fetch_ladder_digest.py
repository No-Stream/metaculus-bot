"""Shared page-digest behavior for fresh, cached and held HTML reads."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from metaculus_bot.research.agentic import local_document
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.document_text import PdfText
from metaculus_bot.research.fetch_ladder import classify
from metaculus_bot.research.fetch_ladder.context import LadderContext
from metaculus_bot.research.fetch_ladder.digest import LadderDigest
from metaculus_bot.research.fetch_ladder.ladder import fetch_url
from metaculus_bot.research.fetch_ladder.policy import (
    GAP_FILL_DOCUMENT_POLICY,
    GAP_FILL_FETCH_POLICY,
    RESOLUTION_SOURCE_POLICY,
)
from metaculus_bot.research.fetch_ladder.verdict import PageExtraction
from tests.resolution_source_fakes import FakeResponse, FakeSession

_URL = "https://tracker.example.com/digest"


def _long_page() -> str:
    return "Opening context about the tracker.\n\n" + ("The tracker reports 917 admissions this week. " * 220)


@pytest.mark.asyncio
async def test_long_html_digest_receives_full_text_and_reapplies_leads_and_cap_on_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The digest sees the canonical extraction on both fresh and cached presentation."""
    full_text = _long_page()
    calls: list[tuple[str, str, float]] = []

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        calls.append((text, query, budget_seconds))
        return LadderDigest(
            passages=["The tracker reports 917 admissions this week."],
            passages_returned=4,
            passages_grounded=3,
            fallback_used=False,
            method="llm_extractive",
        )

    monkeypatch.setattr(classify, "_extract_page_text", lambda *_args, **_kwargs: PageExtraction(text=full_text))
    monkeypatch.setattr(classify, "render_inline_chart_data", lambda *_args, **_kwargs: "Chart data lead")
    monkeypatch.setattr(classify, "unreadable_data_embed_providers", lambda *_args, **_kwargs: ["Infogram"])
    session = FakeSession({_URL: FakeResponse(200, body=b"<html><body>page</body></html>")})
    policy = replace(RESOLUTION_SOURCE_POLICY, digest=digest)

    fresh = await fetch_url(
        _URL,
        policy=policy,
        ctx=LadderContext(query="weekly admissions", session=session, host_sems={}),
    )
    cached = await fetch_url(
        _URL,
        policy=policy,
        ctx=LadderContext(query="weekly admissions", session=session, host_sems={}),
    )

    assert len(calls) == 2
    assert all(text == full_text for text, _query, _budget in calls)
    assert [query for _text, query, _budget in calls] == ["weekly admissions", "weekly admissions"]
    assert all(0.0 < budget <= policy.total_wall_s for _text, _query, budget in calls)
    for result in (fresh, cached):
        assert result.status == "success"
        assert result.text.startswith("Chart data lead\n\n")
        assert "Infogram embed(s)" in result.text
        assert "The tracker reports 917 admissions this week." in result.text
        assert len(result.text) <= policy.per_url_max_chars  # type: ignore[arg-type]
        assert result.passages_returned == 4
        assert result.passages_grounded == 3
        assert result.fallback_used is False
    assert cached.cache_hit is True
    assert session.requested == [_URL]


@pytest.mark.asyncio
async def test_short_html_does_not_invoke_digest_and_has_no_digest_marker_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        del query, budget_seconds
        calls.append(text)
        raise AssertionError("short HTML must stay on the ordinary presentation path")

    short_text = "Official tracker report.\n\n" + ("The official tracker reports 917 admissions this week. " * 12)
    monkeypatch.setattr(classify, "_extract_page_text", lambda *_args, **_kwargs: PageExtraction(text=short_text))
    session = FakeSession({_URL: FakeResponse(200, body=b"<html><body>page</body></html>")})
    result = await fetch_url(
        _URL,
        policy=replace(RESOLUTION_SOURCE_POLICY, digest=digest),
        ctx=LadderContext(query="admissions", session=session, host_sems={}),
    )

    assert result.status == "success"
    assert result.text == short_text
    assert calls == []
    assert result.passages_returned is None
    assert result.passages_grounded is None
    assert result.fallback_used is None


@pytest.mark.asyncio
async def test_loop_fetch_keeps_long_html_paginated_and_does_not_digest_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_text = _long_page()

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        del text, query, budget_seconds
        raise AssertionError("ordinary fetch pagination must not invoke the page digest")

    monkeypatch.setattr(classify, "_extract_page_text", lambda *_args, **_kwargs: PageExtraction(text=full_text))
    monkeypatch.setattr(agentic_tools, "GAP_FILL_FETCH_POLICY", replace(GAP_FILL_FETCH_POLICY, digest=digest))
    session = FakeSession({_URL: FakeResponse(200, body=b"<html><body>page</body></html>")})
    monkeypatch.setattr(agentic_tools.guard, "_get_session", lambda: session)

    first = await agentic_tools.fetch(_URL, question_topic="weekly admissions")
    second = await agentic_tools.fetch(_URL, start_char=8_000, question_topic="weekly admissions")

    assert first.method == "plain"
    assert first.truncated is True
    assert "start_char=8000" in first.content_markdown
    assert "917 admissions" in second.content_markdown
    assert second.method == "cache"
    assert session.requested == [_URL]


@pytest.mark.asyncio
async def test_held_flat_html_uses_digest_seat_with_remaining_read_document_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held_text = "Opening context.\n\n" + ("The tracker reports 917 admissions this week. " * 100)
    calls: list[tuple[str, str, float]] = []

    async def digest(text: str, query: str, *, budget_seconds: float) -> LadderDigest:
        calls.append((text, query, budget_seconds))
        return LadderDigest(
            passages=["The tracker reports 917 admissions this week."],
            passages_returned=2,
            passages_grounded=2,
            fallback_used=False,
            method="llm_extractive",
        )

    monkeypatch.setattr(
        agentic_tools,
        "GAP_FILL_DOCUMENT_POLICY",
        replace(GAP_FILL_DOCUMENT_POLICY, digest=digest),
    )
    monkeypatch.setattr(
        agentic_tools,
        "_acquire_local_document",
        AsyncMock(return_value=local_document.HeldDocument(text=held_text)),
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    outcome = await agentic_tools.read_document(_URL, "weekly admissions")

    assert outcome.method == "digest_local"
    assert "The tracker reports 917 admissions this week." in outcome.content_markdown
    assert len(calls) == 1
    text, query, budget = calls[0]
    assert text == held_text
    assert query == "weekly admissions"
    assert 0.0 < budget <= agentic_tools._READ_DOCUMENT_TOTAL_BUDGET_S


@pytest.mark.asyncio
async def test_subfloor_flat_no_match_still_falls_through_to_paid_reader_when_opening_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    held = local_document.HeldDocument(text="Opening page context with no matching fact.")
    digest = AsyncMock(
        return_value=LadderDigest(
            passages=["Opening page context with no matching fact."],
            passages_returned=0,
            passages_grounded=0,
            fallback_used=True,
            method="digest_local",
        )
    )
    monkeypatch.setattr(agentic_tools, "GAP_FILL_DOCUMENT_POLICY", replace(GAP_FILL_DOCUMENT_POLICY, digest=digest))
    monkeypatch.setattr(agentic_tools, "_acquire_local_document", AsyncMock(return_value=held))
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(agentic_tools, "_url_context_robots_skip", AsyncMock(return_value=False))
    monkeypatch.setattr(
        agentic_tools,
        "_run_document_read_sync",
        MagicMock(return_value=("Paid reader answer.", 1, ["SUCCESS"])),
    )

    outcome = await agentic_tools.read_document(_URL, "weekly admissions")

    assert outcome.method == "document"
    assert outcome.content_markdown == "Paid reader answer."
    digest.assert_awaited_once()


@pytest.mark.asyncio
async def test_held_pdf_keeps_page_aware_digest_and_never_uses_flat_digest_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = PdfText(
        page_count=2,
        pages_read=2,
        pages=("The first page reports unemployment at 4.1 percent.", "The second page reports revisions."),
        truncated_by="",
        outline=(),
    )
    digest = AsyncMock(side_effect=AssertionError("PDFs must use the page-aware local digest"))
    monkeypatch.setattr(agentic_tools, "GAP_FILL_DOCUMENT_POLICY", replace(GAP_FILL_DOCUMENT_POLICY, digest=digest))
    monkeypatch.setattr(
        agentic_tools,
        "_acquire_local_document",
        AsyncMock(return_value=local_document.HeldDocument(text="", pdf=pdf)),
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    outcome = await agentic_tools.read_document(_URL, "unemployment revisions")

    assert outcome.method == "digest_local"
    assert "[p.1]" in outcome.content_markdown or "[p.2]" in outcome.content_markdown
    digest.assert_not_awaited()
