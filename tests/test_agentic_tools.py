from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.agentic.loop import _tool_schemas


class _FakeResponse:
    def __init__(self, *, status: int, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, bool]] = []

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def get(self, url: str, *, allow_redirects: bool = False) -> _FakeResponse:
        self.calls.append((url, allow_redirects))
        return self._response


@pytest.fixture(autouse=True)
def _reset_tool_state() -> None:
    agentic_tools._FETCH_TEXT_CACHE.clear()
    agentic_tools._FETCH_LINKS_CACHE.clear()
    agentic_tools._FETCH_HOST_SEMAPHORES.clear()
    agentic_tools._PLAYWRIGHT_WARNED = False


def test_tool_schemas_round_trip_for_public_tools() -> None:
    tools = agentic_tools.build_gap_fill_tools("topic")
    schemas = _tool_schemas(tools, must_conclude=False)

    by_name = {entry["function"]["name"]: entry["function"] for entry in schemas}
    assert by_name["search_news"]["parameters"]["required"] == ["query"]
    assert by_name["search_web"]["parameters"]["properties"]["end_published_date"]["type"] == ["string", "null"]
    assert by_name["fetch"]["parameters"]["properties"]["start_char"]["minimum"] == 0
    assert by_name["read_document"]["parameters"]["required"] == ["url", "ask"]


@pytest.mark.asyncio
async def test_search_web_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "key")
    monkeypatch.setattr(
        agentic_tools,
        "_call_exa_search",
        AsyncMock(
            return_value=[
                SimpleNamespace(
                    title="Result title",
                    url="https://example.com/a",
                    published_date="2026-07-15",
                    highlights=["First highlight", "Second highlight"],
                )
            ]
        ),
    )

    outcome = await agentic_tools.search_web("query")

    assert outcome.status == "ok"
    assert outcome.method == "search"
    assert "Result title" in outcome.content_markdown
    assert "https://example.com/a" in outcome.content_markdown
    assert "- First highlight" in outcome.content_markdown


@pytest.mark.asyncio
async def test_search_web_retries_then_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "key")
    searcher = MagicMock(
        side_effect=[
            RuntimeError("429 too many requests"),
            RuntimeError("rate limit"),
            SimpleNamespace(results=[SimpleNamespace(title="Recovered", url="https://example.com", highlights=[])]),
        ]
    )
    exa_cls = MagicMock(return_value=SimpleNamespace(search=searcher))
    sleeps: list[float] = []

    monkeypatch.setattr("asyncio.sleep", AsyncMock(side_effect=lambda delay: sleeps.append(delay)))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))
    monkeypatch.setitem(sys.modules, "exa_py", SimpleNamespace(Exa=exa_cls))

    outcome = await agentic_tools.search_web("query")

    assert outcome.status == "ok"
    assert "Recovered" in outcome.content_markdown
    assert sleeps == [1.0, 4.0]
    assert searcher.call_count == 3


@pytest.mark.asyncio
async def test_search_web_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "key")
    searcher = MagicMock(side_effect=RuntimeError("429 too many requests"))
    exa_cls = MagicMock(return_value=SimpleNamespace(search=searcher))

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))
    monkeypatch.setitem(sys.modules, "exa_py", SimpleNamespace(Exa=exa_cls))

    outcome = await agentic_tools.search_web("query")

    assert outcome.status == "error"
    assert "Exa search failed" in outcome.content_markdown
    assert searcher.call_count == 3


@pytest.mark.asyncio
async def test_search_web_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    outcome = await agentic_tools.search_web("query")

    assert outcome.status == "error"
    assert "EXA_API_KEY" in outcome.content_markdown


@pytest.mark.asyncio
async def test_search_web_passes_end_published_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "key")
    searcher = MagicMock(return_value=SimpleNamespace(results=[]))
    exa_cls = MagicMock(return_value=SimpleNamespace(search=searcher))

    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))
    monkeypatch.setitem(sys.modules, "exa_py", SimpleNamespace(Exa=exa_cls))

    await agentic_tools.search_web("query", end_published_date="2026-01-01")

    assert searcher.call_args.kwargs["end_published_date"] == "2026-01-01"


@pytest.mark.asyncio
async def test_search_news_happy_path_uses_gate_and_semaphore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")
    gate = AsyncMock()
    semaphore_entered = False

    class RecordingSemaphore:
        async def __aenter__(self) -> None:
            nonlocal semaphore_entered
            semaphore_entered = True

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class FakeSdk:
        async def __aenter__(self) -> "FakeSdk":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        news = SimpleNamespace(
            search_news=AsyncMock(
                return_value=SimpleNamespace(
                    as_dicts=[
                        {
                            "eng_title": "Article title",
                            "pub_date": "2026-07-16",
                            "source_id": "reuters",
                            "article_url": "https://example.com/story",
                            "summary": "Short summary.",
                        }
                    ]
                )
            )
        )

    monkeypatch.setattr("metaculus_bot.research.providers._ASKNEWS_GLOBAL_SEMAPHORE", RecordingSemaphore())
    monkeypatch.setattr("metaculus_bot.research.providers._asknews_rate_gate", gate)
    monkeypatch.setitem(sys.modules, "asknews_sdk", SimpleNamespace(AsyncAskNewsSDK=lambda **_: FakeSdk()))

    outcome = await agentic_tools.search_news("query")

    assert outcome.status == "ok"
    assert outcome.method == "news"
    assert semaphore_entered is True
    gate.assert_awaited_once()
    assert "Article title" in outcome.content_markdown


@pytest.mark.asyncio
async def test_search_news_missing_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
    monkeypatch.delenv("ASKNEWS_SECRET", raising=False)

    outcome = await agentic_tools.search_news("query")

    assert outcome.status == "error"
    assert "ASKNEWS_CLIENT_ID" in outcome.content_markdown


@pytest.mark.asyncio
async def test_fetch_plain_success_path_reuses_fetch_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/html"}))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
    monkeypatch.setattr(
        agentic_tools,
        "_read_response_body",
        AsyncMock(return_value=b'<html><body><a href="/a">A</a><p>Long body</p></body></html>'),
    )
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text",
        MagicMock(return_value="Rendered plain body " * 40),
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    outcome = await agentic_tools.fetch("https://example.com/page")

    assert outcome.status == "ok"
    assert outcome.method == "plain"
    assert outcome.links == ["https://example.com/a"]
    assert "Rendered plain body" in outcome.content_markdown
    assert session.calls == [("https://example.com/page", False)]


@pytest.mark.asyncio
async def test_fetch_js_wall_escalates_to_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agentic_tools,
        "_fetch_plain",
        AsyncMock(
            return_value=SimpleNamespace(
                status="ok",
                method="plain",
                text="too short",
                links=["https://example.com/plain"],
                url="https://example.com/page",
                escalate_rendered=True,
            )
        ),
    )
    monkeypatch.setattr(
        agentic_tools,
        "_try_rendered_fetch",
        AsyncMock(
            return_value=SimpleNamespace(
                status="ok",
                method="rendered",
                text="rendered body",
                links=["https://example.com/rendered"],
                url="https://example.com/page",
            )
        ),
    )

    outcome = await agentic_tools.fetch("https://example.com/page")

    assert outcome.method == "rendered"
    assert outcome.links == ["https://example.com/rendered"]
    assert outcome.content_markdown == "rendered body"


@pytest.mark.asyncio
async def test_fetch_thin_content_escalates_to_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agentic_tools,
        "_fetch_plain",
        AsyncMock(
            return_value=SimpleNamespace(
                status="ok",
                method="plain",
                text="x" * 100,
                links=[],
                url="https://example.com/page",
                escalate_rendered=True,
            )
        ),
    )
    monkeypatch.setattr(
        agentic_tools,
        "_try_rendered_fetch",
        AsyncMock(
            return_value=SimpleNamespace(
                status="ok", method="rendered", text="x" * 600, links=[], url="https://example.com/page"
            )
        ),
    )

    outcome = await agentic_tools.fetch("https://example.com/page")

    assert outcome.method == "rendered"
    assert outcome.content_markdown == "x" * 600


@pytest.mark.asyncio
async def test_fetch_pdf_content_type_returns_document_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "application/pdf"}))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)

    outcome = await agentic_tools.fetch("https://example.com/file.pdf")

    assert outcome.status == "ok"
    assert outcome.method == "document_needed"
    assert "use read_document" in outcome.content_markdown


@pytest.mark.asyncio
async def test_fetch_pagination_second_call_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    fetch_plain = AsyncMock(
        return_value=SimpleNamespace(
            status="ok",
            method="plain",
            text="A" * (agentic_tools._FETCH_WINDOW_CHARS + 5),
            links=["https://example.com/a"],
            url="https://example.com/page",
            escalate_rendered=False,
        )
    )
    monkeypatch.setattr(agentic_tools, "_fetch_plain", fetch_plain)

    first = await agentic_tools.fetch("https://example.com/page")
    second = await agentic_tools.fetch("https://example.com/page", start_char=agentic_tools._FETCH_WINDOW_CHARS)

    assert first.truncated is True
    assert "[truncated at 8000 of 8005 chars — call again with start_char=8000]" in first.content_markdown
    assert second.method == "cache"
    assert second.content_markdown == "A" * 5
    assert fetch_plain.await_count == 1


def test_extract_links_caps_at_twenty_five() -> None:
    html = "".join(f'<a href="/{index}">link{index}</a>' for index in range(30))

    links = agentic_tools._extract_links_from_html(html, "https://example.com/root")

    assert len(links) == 25
    assert links[0] == "https://example.com/0"
    assert links[-1] == "https://example.com/24"


@pytest.mark.asyncio
async def test_fetch_ssrf_reject_returns_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=False))

    outcome = await agentic_tools.fetch("http://127.0.0.1")

    assert outcome.status == "blocked"


@pytest.mark.asyncio
async def test_fetch_playwright_missing_degrades_to_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agentic_tools,
        "_fetch_plain",
        AsyncMock(
            return_value=SimpleNamespace(
                status="ok",
                method="plain",
                text="plain body",
                links=[],
                url="https://example.com/page",
                escalate_rendered=True,
            )
        ),
    )
    monkeypatch.setattr(agentic_tools, "_try_rendered_fetch", AsyncMock(return_value=None))

    outcome = await agentic_tools.fetch("https://example.com/page")

    assert outcome.method == "plain"
    assert outcome.content_markdown == "plain body"


@pytest.mark.asyncio
async def test_try_rendered_fetch_uses_playwright_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    semaphore_entries: list[str] = []

    class RecordingSemaphore:
        async def __aenter__(self) -> None:
            semaphore_entries.append("entered")

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    class FakePage:
        async def goto(self, url: str, *, wait_until: str, timeout: int) -> SimpleNamespace:
            assert url == "https://example.com/page"
            assert wait_until == "networkidle"
            assert timeout == 35_000
            return SimpleNamespace(headers={"content-type": "text/html"})

        async def content(self) -> str:
            return '<html><body><a href="/next">Next</a><p>Rendered body</p></body></html>'

    class FakeContext:
        async def new_page(self) -> FakePage:
            return FakePage()

        async def close(self) -> None:
            return None

    class FakeBrowser:
        async def new_context(self, **kwargs) -> FakeContext:
            assert kwargs["user_agent"]
            assert "Accept-Language" in kwargs["extra_http_headers"]
            return FakeContext()

        async def close(self) -> None:
            return None

    class FakeChromium:
        async def launch(self, *, headless: bool) -> FakeBrowser:
            assert headless is True
            return FakeBrowser()

    class FakePlaywrightManager:
        chromium = FakeChromium()

        async def __aenter__(self) -> "FakePlaywrightManager":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr("metaculus_bot.research.resolution_source._sem_for_host", lambda *_: RecordingSemaphore())
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value="Rendered body")
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))
    monkeypatch.setitem(
        sys.modules, "playwright.async_api", SimpleNamespace(async_playwright=lambda: FakePlaywrightManager())
    )

    outcome = await agentic_tools._try_rendered_fetch("https://example.com/page")

    assert outcome is not None
    assert outcome.method == "rendered"
    assert outcome.links == ["https://example.com/next"]
    assert semaphore_entries == ["entered"]


@pytest.mark.asyncio
async def test_read_document_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))
    monkeypatch.setattr(agentic_tools, "_run_document_read_sync", MagicMock(return_value="Quoted answer with dates."))

    outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What does it say?")

    assert outcome.status == "ok"
    assert outcome.method == "document"
    assert outcome.content_markdown == "Quoted answer with dates."


@pytest.mark.asyncio
async def test_read_document_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setattr(agentic_tools, "_READ_DOCUMENT_TIMEOUT_S", 0.01)

    async def slow_to_thread(fn, *args):
        await asyncio.sleep(0.05)
        return fn(*args)

    monkeypatch.setattr("asyncio.to_thread", slow_to_thread)
    monkeypatch.setattr(agentic_tools, "_run_document_read_sync", MagicMock(return_value="late result"))

    outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What does it say?")

    assert outcome.status == "error"
    assert "timed out" in outcome.content_markdown


@pytest.mark.asyncio
async def test_read_document_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What does it say?")

    assert outcome.status == "error"
    assert "GOOGLE_API_KEY" in outcome.content_markdown
