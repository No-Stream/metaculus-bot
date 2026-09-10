"""An unreadable PDF parse remains reusable by later document reads."""

from __future__ import annotations

import pytest

from metaculus_bot.research import document_cache
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.document_text import PdfText
from metaculus_bot.research.fetch_ladder import classify, guard
from metaculus_bot.research.fetch_ladder.context import LadderContext
from tests.resolution_source_fakes import FakeResponse, FakeSession

_URL = "https://reports.example.com/scanned.pdf"


@pytest.mark.asyncio
async def test_repeated_document_acquisition_reuses_an_unreadable_pdf_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _public(_url: str) -> bool:
        return True

    parsed: list[bytes] = []

    def _extract(body: bytes, **_kwargs: object) -> PdfText:
        parsed.append(body)
        return PdfText(page_count=2, pages_read=2, pages=("", ""), truncated_by="", outline=())

    monkeypatch.setattr(guard, "is_public_http_url", _public)
    monkeypatch.setattr(classify, "extract_pdf_text", _extract)
    session = FakeSession({_URL: FakeResponse(200, body=b"%PDF-1.4\nscan", content_type="application/pdf")})
    ctx = LadderContext(session=session, host_sems={})

    first = await agentic_tools._acquire_local_document(_URL, ctx=ctx)
    second = await agentic_tools._acquire_local_document(_URL, ctx=ctx)

    assert first.pdf is not None
    assert first.text == ""
    assert second == first
    assert session.requested == [_URL]
    assert parsed == [b"%PDF-1.4\nscan"]
    assert document_cache.cached_document(_URL) == first.pdf
