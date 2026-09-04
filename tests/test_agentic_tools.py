from __future__ import annotations

import asyncio
import io
import logging
import socket
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import aiohttp
import pytest
from google.genai import types as genai_types
from playwright.async_api import Error as _PlaywrightError
from playwright.async_api import TimeoutError as _PlaywrightTimeoutError
from pypdf import PdfWriter

from metaculus_bot.constants import (
    DOCUMENT_DIGEST_TOP_K,
    DOCUMENT_TEXT_PDF_MAX_BYTES,
    GAP_FILL_V2_READER_HTTP_ATTEMPTS,
    GAP_FILL_V2_READER_MODEL,
    GAP_FILL_V2_READER_THINKING_LEVEL,
    URL_CONTEXT_SIZE_GATE_TOKENS,
)
from metaculus_bot.research import http_fetch, rendered_fetch, robots_policy
from metaculus_bot.research import providers as research_providers
from metaculus_bot.research.agentic import fetch_outcomes, local_document, tool_backends
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.agentic.loop import _harvest_verification_tiers, _method_to_tier, _tool_schemas
from metaculus_bot.research.document_text import extract_pdf_text
from metaculus_bot.research.gemini_client_config import gemini_retry_sleep_allowance_s
from tests.playwright_fakes import FakeBrowser, FakeChromium, FakePage, FakePlaywrightManager, install_fake_playwright
from tests.test_document_text import build_text_pdf


class _FakeResponse:
    def __init__(self, *, status: int, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeSession:
    """Serves a queued sequence of responses; the last response repeats.

    Single-response construction keeps the original fixed-response behavior;
    multi-response construction lets redirect tests script a chain of hops.
    """

    def __init__(self, *responses: _FakeResponse) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, bool]] = []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def get(self, url: str, *, allow_redirects: bool = False) -> _FakeResponse:
        self.calls.append((url, allow_redirects))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


def _scanned_pdf() -> bytes:
    """A structurally valid PDF with a page and no text layer at all — a scan."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _serve_pdf(monkeypatch: pytest.MonkeyPatch, body: bytes, *, content_type: str = "application/pdf") -> AsyncMock:
    """Wire the plain rung to answer one request with ``body`` under ``content_type``.

    Patches ``_read_response_body`` rather than teaching the fake response object to stream,
    because the cap that read runs under is the thing the PDF rung changes and each test wants
    to state the body it is classifying, not the transport. Returns that spy, which is how a
    test counts requests: one request has to serve both a paginated fetch and a later digest.
    """
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": content_type}))
    read_body = AsyncMock(return_value=body)
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
    monkeypatch.setattr(agentic_tools, "_read_response_body", read_body)
    return read_body


@pytest.fixture(autouse=True)
def _reset_tool_state() -> None:
    # Run-scoped state of the local-document rung: held parses, plus the pypdf parse gate it now
    # shares process-wide with the Tier-1 rung (its own reset helper, since the gate is
    # loop-scoped and one test's held slot must not gate another's).
    local_document.clear_document_cache()
    http_fetch.reset_pdf_parse_semaphore()
    agentic_tools._FETCH_TEXT_CACHE.clear()
    agentic_tools._FETCH_LINKS_CACHE.clear()
    agentic_tools._FETCH_HOST_SEMAPHORES.clear()
    # Shared with the Tier-1 reader, so it is reset through its owning module.
    robots_policy.reset_robots_cache()
    # Run-scoped state of the shared render transport: the rendered-to-nothing memo, the
    # one-shot playwright warn latch, and a FRESH launch semaphore (construction is loop-free
    # in 3.12, so rebinding prevents a contended acquire in one test's event loop from leaking
    # a loop binding into a later test).
    rendered_fetch.reset_render_state()


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


class _FakeHttpxAsyncClient:
    """httpx.AsyncClient double that records its constructor kwargs.

    Lets the Exa tests assert the client-side timeout was applied without a
    real socket. Instances append their kwargs to the shared ``captured`` list.
    """

    def __init__(self, captured: list[dict[str, Any]], **kwargs: Any) -> None:
        captured.append(kwargs)

    async def aclose(self) -> None:
        return None


def _patch_async_exa(monkeypatch: pytest.MonkeyPatch, searcher: MagicMock) -> list[dict[str, Any]]:
    """Wire fake ``exa_py.AsyncExa`` + ``httpx`` for a ``search_web`` test.

    ``searcher`` drives ``AsyncExa.search`` (call it to raise/return); the
    returned list captures each ``httpx.AsyncClient(**kwargs)`` construction.
    """
    captured: list[dict[str, Any]] = []

    class FakeAsyncExa:
        def __init__(self, api_key: str | None = None) -> None:
            self.base_url = "https://api.exa.ai"
            self.headers = {"x-api-key": api_key}
            self._client: Any = None

        async def search(self, **kwargs: Any) -> Any:
            return searcher(**kwargs)

    monkeypatch.setitem(sys.modules, "exa_py", SimpleNamespace(AsyncExa=FakeAsyncExa))
    monkeypatch.setitem(
        sys.modules, "httpx", SimpleNamespace(AsyncClient=lambda **kwargs: _FakeHttpxAsyncClient(captured, **kwargs))
    )
    return captured


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
    sleeps: list[float] = []

    monkeypatch.setattr("asyncio.sleep", AsyncMock(side_effect=sleeps.append))
    _patch_async_exa(monkeypatch, searcher)

    outcome = await agentic_tools.search_web("query")

    assert outcome.status == "ok"
    assert "Recovered" in outcome.content_markdown
    assert sleeps == [1.0, 4.0]
    assert searcher.call_count == 3


@pytest.mark.asyncio
async def test_search_web_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXA_API_KEY", "key")
    searcher = MagicMock(side_effect=RuntimeError("429 too many requests"))

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    _patch_async_exa(monkeypatch, searcher)

    outcome = await agentic_tools.search_web("query")

    assert outcome.status == "error"
    assert "Exa search failed" in outcome.content_markdown
    assert searcher.call_count == 3


@pytest.mark.asyncio
async def test_search_web_exa_client_uses_bounded_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix 2 (Exa half): the async Exa client is built with a client-side
    timeout <= the search_web tool budget, so a hung endpoint tears the socket
    down before the loop's wait_for fires — and there is no worker thread to
    leak because the sync/to_thread path is gone."""
    monkeypatch.setenv("EXA_API_KEY", "key")
    searcher = MagicMock(return_value=SimpleNamespace(results=[]))
    captured = _patch_async_exa(monkeypatch, searcher)

    outcome = await agentic_tools.search_web("query")

    assert outcome.status == "ok"
    tool_budget = next(
        tool.timeout_s for tool in agentic_tools.build_gap_fill_tools("topic") if tool.name == "search_web"
    )
    assert len(captured) == 1
    assert "timeout" in captured[0]
    assert captured[0]["timeout"] is not None
    assert captured[0]["timeout"] <= tool_budget


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
    _patch_async_exa(monkeypatch, searcher)

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
        async def __aenter__(self) -> FakeSdk:
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


class _FakeAskNewsSdk:
    """Async-context AskNews SDK double with a scripted search_news."""

    def __init__(self, search_news_mock: AsyncMock) -> None:
        self.news = SimpleNamespace(search_news=search_news_mock)

    async def __aenter__(self) -> _FakeAskNewsSdk:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _patch_asknews_env(monkeypatch: pytest.MonkeyPatch, search_news_mock: AsyncMock) -> AsyncMock:
    """Wire creds + SDK + rate gate for a _call_asknews_search test; returns the sleep recorder."""
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")
    monkeypatch.setattr("metaculus_bot.research.providers._asknews_rate_gate", AsyncMock())
    monkeypatch.setitem(
        sys.modules, "asknews_sdk", SimpleNamespace(AsyncAskNewsSDK=lambda **_: _FakeAskNewsSdk(search_news_mock))
    )
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)
    return sleep_mock


@pytest.mark.asyncio
async def test_asknews_search_retries_rate_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    search_news = AsyncMock(
        side_effect=[
            RuntimeError("429 too many requests"),
            SimpleNamespace(as_dicts=[{"eng_title": "Recovered article"}]),
        ]
    )
    sleep_mock = _patch_asknews_env(monkeypatch, search_news)

    articles = await agentic_tools._call_asknews_search("query")

    assert len(articles) == 1
    assert search_news.await_count == 2
    # One backoff between the two attempts, on the provider's schedule.
    expected_backoff = agentic_tools.ASKNEWS_BACKOFF_SECS * (10 + 3**1)
    assert [call.args[0] for call in sleep_mock.await_args_list] == [expected_backoff]


@pytest.mark.asyncio
async def test_asknews_search_retries_concurrency_limit_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    search_news = AsyncMock(
        side_effect=[
            RuntimeError("concurrency limit exceeded for plan"),
            SimpleNamespace(as_dicts=[{"eng_title": "Recovered article"}]),
        ]
    )
    sleep_mock = _patch_asknews_env(monkeypatch, search_news)

    articles = await agentic_tools._call_asknews_search("query")

    assert len(articles) == 1
    assert search_news.await_count == 2
    assert sleep_mock.await_count == 1


@pytest.mark.asyncio
async def test_asknews_search_non_retryable_error_raises_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    search_news = AsyncMock(side_effect=RuntimeError("invalid credentials"))
    sleep_mock = _patch_asknews_env(monkeypatch, search_news)

    with pytest.raises(RuntimeError, match="invalid credentials"):
        await agentic_tools._call_asknews_search("query")

    assert search_news.await_count == 1
    assert sleep_mock.await_count == 0


class _FakeAskNewsForbiddenError(Exception):
    """Stand-in for ``asknews_sdk.errors.ForbiddenError`` — matched by class name."""


# ``is_asknews_subscription_error`` keys on the class-name substring, so rename the
# attribute to the SDK's real class name (same trick as tests/test_research_providers.py).
_FakeAskNewsForbiddenError.__name__ = "ForbiddenError"


@pytest.mark.asyncio
async def test_asknews_search_subscription_inactive_raises_on_first_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """403011 subscription-inactive is PERMANENT, so it costs exactly one attempt.

    It used to be exempted from the fast-fail alongside rate limits, which re-rolled a
    call that can never succeed and burned the whole ``ASKNEWS_MAX_TRIES`` backoff ladder
    out of GAP_FILL_V2_WALL_DEADLINE. The primary provider's ``_is_retryable`` never
    retried it; this asserts the agentic path matches that policy.
    """
    subscription_exc = _FakeAskNewsForbiddenError("403011 - subscription is not currently active")
    assert research_providers.is_asknews_subscription_error(subscription_exc) is True
    search_news = AsyncMock(side_effect=subscription_exc)
    sleep_mock = _patch_asknews_env(monkeypatch, search_news)

    with pytest.raises(_FakeAskNewsForbiddenError, match="403011"):
        await agentic_tools._call_asknews_search("query")

    assert search_news.await_count == 1
    assert sleep_mock.await_count == 0


@pytest.mark.asyncio
async def test_asknews_search_rate_limit_exhausts_retries_and_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    tries = max(1, int(agentic_tools.ASKNEWS_MAX_TRIES))
    search_news = AsyncMock(side_effect=RuntimeError("429 too many requests"))
    sleep_mock = _patch_asknews_env(monkeypatch, search_news)

    with pytest.raises(RuntimeError, match="429"):
        await agentic_tools._call_asknews_search("query")

    assert search_news.await_count == tries
    assert sleep_mock.await_count == tries - 1


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
async def test_fetch_scanned_pdf_escalates_to_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PDF with no text layer is the paid reader's one remaining job on this rung."""
    _serve_pdf(monkeypatch, _scanned_pdf())
    read_document = AsyncMock(
        return_value=agentic_tools.ToolOutcome(content_markdown="Extracted PDF content.", method="document")
    )
    monkeypatch.setattr(agentic_tools, "read_document", read_document)

    outcome = await agentic_tools.fetch("https://example.com/file.pdf")

    assert outcome.status == "ok"
    assert outcome.method == "document"
    assert outcome.content_markdown == "Extracted PDF content."
    read_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_document_escalation_generic_ask_contains_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_pdf(monkeypatch, _scanned_pdf())
    read_document = AsyncMock(
        return_value=agentic_tools.ToolOutcome(content_markdown="Extracted PDF content.", method="document")
    )
    monkeypatch.setattr(agentic_tools, "read_document", read_document)

    tools = agentic_tools.build_gap_fill_tools("Will Nauru ratify the treaty?")
    fetch_handler = next(tool.handler for tool in tools if tool.name == "fetch")

    outcome = await fetch_handler(url="https://example.com/file.pdf")

    assert outcome.method == "document"
    read_document.assert_awaited_once_with(
        "https://example.com/file.pdf",
        "Extract the main content relevant to: Will Nauru ratify the treaty?",
        # The free rungs just ran here, so the escalation says so rather than running them again.
        ladder_exhausted=True,
    )


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


@pytest.mark.asyncio
async def test_fetch_plain_textual_branch_strips_allowlisted_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw-text branch runs the same allow-listed tag strip as the Tier-1 CSV path:
    a poll-tracker CSV's styled per-row anchors are markup the driver's result budget
    should not buy, and inequality signs in data cells must survive untouched."""
    csv_body = (
        b"date,pollster,margin\n"
        b"\"8/16 - 8/17, 2026\",<a href='https://poller.example/aug' style='color:#000'>Emerson College</a>,-12.8\n"
        b"note,a < 5 and b > 3,0.0\n"
    )
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/csv"}))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
    monkeypatch.setattr(agentic_tools, "_read_response_body", AsyncMock(return_value=csv_body))

    result = await agentic_tools._fetch_plain("https://example.com/data.csv")

    assert result.status == "ok"
    assert "Emerson College" in result.text
    assert "<a " not in result.text
    assert "style=" not in result.text
    assert "a < 5 and b > 3" in result.text


def test_extract_links_caps_at_twenty_five() -> None:
    html = "".join(f'<a href="/{index}">link{index}</a>' for index in range(30))

    links = agentic_tools._extract_links_from_html(html, "https://example.com/root")

    assert len(links) == 25
    assert links[0] == "https://example.com/0"
    assert links[-1] == "https://example.com/24"


@pytest.mark.asyncio
async def test_fetch_plain_follows_redirect_to_public_url(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(
        _FakeResponse(status=302, headers={"Location": "https://example.com/final"}),
        _FakeResponse(status=200, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
    monkeypatch.setattr(
        agentic_tools,
        "_read_response_body",
        AsyncMock(return_value=b"<html><body><p>Final page body</p></body></html>"),
    )
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text",
        MagicMock(return_value="Final page body " * 40),
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    result = await agentic_tools._fetch_plain("https://example.com/start")

    assert result.status == "ok"
    assert result.url == "https://example.com/final"
    assert "Final page body" in result.text
    assert session.calls == [("https://example.com/start", False), ("https://example.com/final", False)]


@pytest.mark.asyncio
async def test_fetch_plain_blocks_redirect_to_non_public_target(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(
        _FakeResponse(status=302, headers={"Location": "http://169.254.169.254/latest/meta-data/"}),
    )

    async def is_public(url: str) -> bool:
        return "169.254.169.254" not in url

    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", is_public)
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)

    result = await agentic_tools._fetch_plain("https://example.com/start")

    assert result.status == "blocked"
    assert "non-public redirect target" in result.text
    # The private hop must never be requested.
    assert session.calls == [("https://example.com/start", False)]


@pytest.mark.asyncio
async def test_fetch_plain_caps_redirect_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    hops = agentic_tools.MAX_REDIRECTS + 2
    session = _FakeSession(
        *[_FakeResponse(status=302, headers={"Location": f"https://example.com/hop{i}"}) for i in range(hops)]
    )
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)

    result = await agentic_tools._fetch_plain("https://example.com/start")

    assert result.status == "error"
    assert result.text == "Redirect limit exceeded."
    assert len(session.calls) == agentic_tools.MAX_REDIRECTS + 1


@pytest.mark.asyncio
async def test_fetch_plain_redirect_without_location_is_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(_FakeResponse(status=302, headers={}))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)

    result = await agentic_tools._fetch_plain("https://example.com/start")

    assert result.status == "error"
    assert "Malformed redirect" in result.text


@pytest.mark.parametrize(
    "url",
    [
        "https://www.metaculus.com/questions/12345/",
        "https://metaculus.com/q/12345",
        "https://www.metaculus.com:443/questions/12345/",  # port must not bypass the block
        "https://sub.metaculus.com/page",  # subdomain
    ],
)
@pytest.mark.asyncio
async def test_fetch_plain_blocks_metaculus_without_network(url: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # is_public_http_url is stubbed True so the metaculus URL clears the SSRF gate
    # (as it would in prod — metaculus is public); the real is_metaculus_self_ref
    # then blocks it. _get_session raises if reached, proving no HTTP is attempted.
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    get_session = MagicMock(side_effect=AssertionError("must not open a session for a metaculus URL"))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", get_session)

    result = await agentic_tools._fetch_plain(url)

    assert result.status == "blocked"
    assert "metaculus.com" in result.text
    get_session.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_plain_blocks_redirect_to_metaculus(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(
        _FakeResponse(status=302, headers={"Location": "https://www.metaculus.com/questions/12345/"}),
    )
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)

    result = await agentic_tools._fetch_plain("https://example.com/start")

    assert result.status == "blocked"
    assert "metaculus.com" in result.text
    # The metaculus hop must never be requested (only the initial URL was GET-ed).
    assert session.calls == [("https://example.com/start", False)]


@pytest.mark.asyncio
async def test_same_host_plain_and_rendered_fetches_serialize(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plan §5 politeness: a plain and a rendered fetch to the same host must
    contend on the same per-host Semaphore(1) and never run concurrently."""
    events: list[str] = []
    release_plain = asyncio.Event()

    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/html"}))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)

    async def blocking_read(resp: object, label: str, *, max_bytes: int = 0) -> bytes:
        events.append("plain_read_started")
        await release_plain.wait()
        events.append("plain_read_finished")
        return b"<html><body><p>Long body</p></body></html>"

    monkeypatch.setattr(agentic_tools, "_read_response_body", blocking_read)
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text",
        MagicMock(return_value="body text " * 60),
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    class _RecordingManager(FakePlaywrightManager):
        async def __aenter__(self) -> _RecordingManager:
            # Runs strictly after _try_rendered_fetch acquires the host gate.
            events.append("rendered_started")
            return self

    install_fake_playwright(
        monkeypatch,
        FakePage(html="<html><body><p>rendered body</p></body></html>"),
        pinned=("example.com", "93.184.216.34"),
        manager_cls=_RecordingManager,
    )

    plain_task = asyncio.create_task(agentic_tools._fetch_plain("https://example.com/plain-page"))
    await asyncio.sleep(0)
    assert events == ["plain_read_started"]  # plain holds the example.com gate

    rendered_task = asyncio.create_task(agentic_tools._try_rendered_fetch("https://example.com/rendered-page"))
    for _ in range(3):
        await asyncio.sleep(0)
    # Rendered must be parked on the shared host gate while plain holds it.
    assert "rendered_started" not in events

    release_plain.set()
    plain_result = await plain_task
    rendered_result = await rendered_task

    assert plain_result.status == "ok"
    assert rendered_result is not None
    assert rendered_result.method == "rendered"
    assert events.index("plain_read_finished") < events.index("rendered_started")


@pytest.mark.asyncio
async def test_rendered_fetch_drains_routes_and_guard_tolerates_teardown_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """2026-07-25 teardown fix. Three things must hold together:
    1. the SSRF route guard still ABORTS a disallowed URL and CONTINUES an
       allowed one on a live page (the guard is the SSRF boundary — never weaken);
    2. a route callback racing context teardown (continue_/abort raising a
       closed-target Playwright error) is swallowed, not re-raised as an
       unhandled event-listener error (the log storm);
    3. teardown drains handlers via unroute_all(behavior="ignoreErrors") before
       closing, and context/browser are still closed.
    """

    # A closed-target error is a subclass of Playwright's public Error (exactly
    # like the real TargetClosedError) — built locally so the test doesn't lean
    # on the private import path the storm's traceback came from.
    class _RacingClosedError(_PlaywrightError):
        pass

    async def _is_public(url: str) -> bool:
        return "evil" not in url

    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", _is_public)
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value="body text " * 60)
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    class FakeRoute:
        def __init__(self, *, raise_on_action: bool = False) -> None:
            self.aborted: str | None = None
            self.continued = False
            self._raise = raise_on_action

        async def continue_(self) -> None:
            if self._raise:
                raise _RacingClosedError("Route.continue: Target page, context or browser has been closed")
            self.continued = True

        async def abort(self, code: str | None = None) -> None:
            if self._raise:
                raise _RacingClosedError("Route.abort: Target page, context or browser has been closed")
            self.aborted = code

    page = FakePage(html="<html><body><p>rendered body</p></body></html>")
    install_fake_playwright(monkeypatch, page, pinned=("example.com", "93.184.216.34"))

    result = await agentic_tools._try_rendered_fetch("https://example.com/page")
    assert result is not None
    assert result.method == "rendered"

    # Teardown drained the handlers before close (Playwright's remedy for the storm).
    assert page.unroute_behavior == "ignoreErrors"
    assert page.teardown == ["unroute_all", "context.close", "browser.close"]

    guard = page.route_handler

    # SSRF guard intact: disallowed URL is aborted, allowed URL is continued.
    disallowed = FakeRoute()
    await guard(disallowed, SimpleNamespace(url="http://evil.internal/imds"))
    assert disallowed.aborted == "blockedbyclient"
    assert disallowed.continued is False

    allowed = FakeRoute()
    await guard(allowed, SimpleNamespace(url="https://example.com/subresource"))
    assert allowed.continued is True
    assert allowed.aborted is None

    # Teardown race: continue_/abort raising a closed-target error must be
    # swallowed, not re-raised (no unhandled event-listener storm).
    racing_allowed = FakeRoute(raise_on_action=True)
    await guard(racing_allowed, SimpleNamespace(url="https://example.com/late"))
    racing_blocked = FakeRoute(raise_on_action=True)
    await guard(racing_blocked, SimpleNamespace(url="http://evil.internal/late"))


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


# ---------------------------------------------------------------------------
# No-content fetch outcome (empty-page laundering fix). A 200 OK whose page
# yields ZERO extractable text is NOT a successful fetch: it must carry a
# distinct non-"ok" status so the loop's tier stamping can never mark an unread
# page "fetched" (the companiesmarketcap.com js-wall failure, 2026-07-25).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extracted", [None, "", "   \n  \t "])
@pytest.mark.asyncio
async def test_fetch_plain_empty_extraction_returns_empty_status(
    extracted: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 200-OK HTML page that extracts to nothing (or only whitespace) must
    report status="empty", not "ok" — while still flagging escalation so the
    ladder tries the rendered rung next."""
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/html"}))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
    monkeypatch.setattr(agentic_tools, "_read_response_body", AsyncMock(return_value=b"<html><body></body></html>"))
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value=extracted)
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    result = await agentic_tools._fetch_plain("https://example.com/js-wall")

    assert result.status == "empty"
    assert result.escalate_rendered is True
    assert "no extractable text" in result.text.lower()


@pytest.mark.asyncio
async def test_fetch_plain_thin_extraction_is_ok_not_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short-but-real extraction is genuinely read content: status stays "ok"
    (fetched-tierable) even though it's below the escalation floor. Thin != empty
    — demoting real short sources would harm legitimate official statements."""
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/html"}))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
    monkeypatch.setattr(
        agentic_tools, "_read_response_body", AsyncMock(return_value=b"<html><body><p>hi</p></body></html>")
    )
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text",
        MagicMock(return_value="Short but real official statement."),
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    result = await agentic_tools._fetch_plain("https://example.com/short")

    assert result.status == "ok"
    assert result.escalate_rendered is True  # thin -> escalate, but the content is real
    assert result.text == "Short but real official statement."


@pytest.mark.asyncio
async def test_fetch_plain_honors_declared_charset_on_textual_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A windows-1252 CSV with its charset declared decodes faithfully. The old
    forced-UTF-8 read turned every high byte into U+FFFD and shipped the
    mojibake to the driver as status="ok"."""
    body = "date,séries\n2026-08-01,0.42\n".encode("windows-1252")
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/csv; charset=windows-1252"}))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
    monkeypatch.setattr(agentic_tools, "_read_response_body", AsyncMock(return_value=body))

    result = await agentic_tools._fetch_plain("https://example.com/data.csv")

    assert result.status == "ok"
    assert "séries" in result.text
    assert "�" not in result.text


@pytest.mark.asyncio
async def test_fetch_plain_refuses_an_undecodable_textual_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A BOM-less UTF-16 body with no declared charset decodes to NUL-interleaved
    garbage — a failed decode, not text we read. It must report "empty" (never
    "ok") and escalate, so the rendered rung's browser sniffing gets a try."""
    body = "date,value\n2026-08-01,0.42\n".encode("utf-16-le")
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/plain"}))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
    monkeypatch.setattr(agentic_tools, "_read_response_body", AsyncMock(return_value=body))

    result = await agentic_tools._fetch_plain("https://example.com/data.txt")

    assert result.status == "empty"
    assert result.escalate_rendered is True
    assert "could not decode" in result.text


class TestFetchPlainTerminalStatuses:
    """Behavior pins for the non-content exit paths of the plain rung.

    Each branch here decides whether the ladder escalates, retries, or hands the
    driver a refusal, and each was previously only covered transitively through
    ``fetch``. They are pinned directly so the status/method/text triple a
    caller keys on cannot drift.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", sorted(agentic_tools._RETRYABLE_FETCH_BLOCK_STATUSES))
    async def test_anti_bot_status_is_blocked(self, status: int, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _FakeSession(_FakeResponse(status=status, headers={"Content-Type": "text/html"}))
        monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
        monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)

        result = await agentic_tools._fetch_plain("https://example.com/gated")

        assert result.status == "blocked"
        assert result.method == "plain"
        assert result.text == f"Fetch blocked with HTTP {status}."
        assert result.url == "https://example.com/gated"

    @pytest.mark.asyncio
    async def test_server_error_status_is_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _FakeSession(_FakeResponse(status=503, headers={"Content-Type": "text/html"}))
        monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
        monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)

        result = await agentic_tools._fetch_plain("https://example.com/down")

        assert result.status == "error"
        assert result.text == "Fetch failed with HTTP 503."

    @pytest.mark.asyncio
    async def test_pdf_with_a_text_layer_is_read_locally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The rung the whole change exists for: a declared PDF is decoded, not escalated."""
        _serve_pdf(
            monkeypatch,
            build_text_pdf([["The unemployment rate was 4.1 percent in May 2026, revised from 4.0 percent."]]),
        )

        result = await agentic_tools._fetch_plain("https://example.com/report.pdf")

        assert result.status == "ok"
        assert result.method == "pdf_local"
        assert "The unemployment rate was 4.1 percent in May 2026" in result.text
        # Never escalated to the rendered rung: a browser has nothing to add to a decoded PDF,
        # and a short-but-real document is a complete read.
        assert result.escalate_rendered is False

    @pytest.mark.asyncio
    async def test_scanned_pdf_asks_for_read_document(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No text layer is the one PDF shape a model still has to read."""
        _serve_pdf(monkeypatch, _scanned_pdf())

        result = await agentic_tools._fetch_plain("https://example.com/scan.pdf")

        assert result.status == "ok"
        assert result.method == "document_needed"
        assert "read_document" in result.text

    @pytest.mark.asyncio
    async def test_pdf_magic_bytes_behind_html_content_type_are_read_locally(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mislabeled body is classified off its bytes, not its Content-Type."""
        _serve_pdf(
            monkeypatch,
            build_text_pdf([["Mislabeled as HTML, but its bytes are a readable PDF document."]]),
            content_type="text/html",
        )

        result = await agentic_tools._fetch_plain("https://example.com/mislabeled")

        assert result.method == "pdf_local"
        assert "Mislabeled as HTML" in result.text

    @pytest.mark.asyncio
    async def test_oversized_body_is_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "text/html"}))
        monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
        monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
        monkeypatch.setattr(agentic_tools, "_read_response_body", AsyncMock(return_value=None))

        result = await agentic_tools._fetch_plain("https://example.com/huge")

        assert result.status == "error"
        assert result.text == "Fetch body exceeded the size limit."

    @pytest.mark.asyncio
    async def test_unsupported_content_type_is_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "application/zip"}))
        monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
        monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
        monkeypatch.setattr(agentic_tools, "_read_response_body", AsyncMock(return_value=b"PK\x03\x04payload"))

        result = await agentic_tools._fetch_plain("https://example.com/bundle.zip")

        assert result.status == "error"
        assert result.text == "Unsupported content type: application/zip"
        assert result.content_type == "application/zip"

    @pytest.mark.asyncio
    async def test_transport_error_is_reported_not_raised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _RaisingSession(_FakeSession):
            def get(self, url: str, *, allow_redirects: bool = False):  # type: ignore[override]
                self.calls.append((url, allow_redirects))
                raise aiohttp.ClientConnectorError(MagicMock(), OSError("refused"))

        session = _RaisingSession(_FakeResponse(status=200))
        monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
        monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)

        result = await agentic_tools._fetch_plain("https://example.com/unreachable")

        assert result.status == "error"
        assert result.text.startswith("Fetch error: ClientConnectorError")


@pytest.mark.asyncio
async def test_fetch_plain_redirect_to_empty_page_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A redirect chain that terminates on a 200-OK empty page is still empty —
    the final hop, not the redirect, decides the outcome."""
    session = _FakeSession(
        _FakeResponse(status=302, headers={"Location": "https://example.com/final"}),
        _FakeResponse(status=200, headers={"Content-Type": "text/html"}),
    )
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
    monkeypatch.setattr(agentic_tools, "_read_response_body", AsyncMock(return_value=b"<html><body></body></html>"))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value=None))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    result = await agentic_tools._fetch_plain("https://example.com/start")

    assert result.status == "empty"
    assert result.url == "https://example.com/final"


@pytest.mark.asyncio
async def test_fetch_empty_plain_failed_render_returns_empty_not_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core fix: an empty plain fetch whose rendered rung is unavailable must
    NOT be laundered back into a plain/ok success. It returns a distinct "empty"
    outcome, legible to the driver, that no tier map can promote to "fetched"."""
    monkeypatch.setattr(
        agentic_tools,
        "_fetch_plain",
        AsyncMock(
            return_value=SimpleNamespace(
                status="empty",
                method="plain",
                text="Plain fetch returned no extractable text.",
                links=[],
                url="https://companiesmarketcap.com/berkshire-hathaway/marketcap/",
                escalate_rendered=True,
            )
        ),
    )
    monkeypatch.setattr(agentic_tools, "_try_rendered_fetch", AsyncMock(return_value=None))

    outcome = await agentic_tools.fetch("https://companiesmarketcap.com/berkshire-hathaway/marketcap/")

    assert outcome.status == "empty"
    assert outcome.method == "empty"
    assert _method_to_tier(outcome.method) is None
    # Legible to a probabilistic consumer: it must read as "nothing was read",
    # not as a thin-but-valid page it can confabulate around.
    assert "was read" in outcome.content_markdown.lower()
    # Never cached: a cached placeholder would resurface as method="cache" (a
    # fetched-tier method) on a later paginated fetch and re-launder the tier.
    assert "https://companiesmarketcap.com/berkshire-hathaway/marketcap/" not in agentic_tools._FETCH_TEXT_CACHE


@pytest.mark.asyncio
async def test_fetch_empty_plain_and_empty_render_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact prod scenario: plain extracts nothing AND the rendered rung runs
    but also extracts nothing (status="error", empty text). The outcome stays
    "empty" rather than falling back to the empty plain placeholder as ok."""
    monkeypatch.setattr(
        agentic_tools,
        "_fetch_plain",
        AsyncMock(
            return_value=SimpleNamespace(
                status="empty",
                method="plain",
                text="Plain fetch returned no extractable text.",
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
                status="error", method="rendered", text="", links=[], url="https://example.com/page"
            )
        ),
    )

    outcome = await agentic_tools.fetch("https://example.com/page")

    assert outcome.status == "empty"
    assert _method_to_tier(outcome.method) is None


@pytest.mark.asyncio
async def test_fetch_empty_plain_still_escalates_to_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty plain fetch must NOT short-circuit: the ladder still runs the
    rendered rung, and a successful render is returned as a real fetched outcome."""
    monkeypatch.setattr(
        agentic_tools,
        "_fetch_plain",
        AsyncMock(
            return_value=SimpleNamespace(
                status="empty",
                method="plain",
                text="Plain fetch returned no extractable text.",
                links=[],
                url="https://example.com/page",
                escalate_rendered=True,
            )
        ),
    )
    rendered = AsyncMock(
        return_value=SimpleNamespace(
            status="ok", method="rendered", text="real rendered content", links=[], url="https://example.com/page"
        )
    )
    monkeypatch.setattr(agentic_tools, "_try_rendered_fetch", rendered)

    outcome = await agentic_tools.fetch("https://example.com/page")

    assert outcome.status == "ok"
    assert outcome.method == "rendered"
    assert outcome.content_markdown == "real rendered content"
    rendered.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_empty_plain_still_escalates_to_read_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty plain fetch must still permit escalation to read_document when the
    rendered rung discovers a document (PDF/image) behind the URL."""
    monkeypatch.setattr(
        agentic_tools,
        "_fetch_plain",
        AsyncMock(
            return_value=SimpleNamespace(
                status="empty",
                method="plain",
                text="Plain fetch returned no extractable text.",
                links=[],
                url="https://example.com/report",
                escalate_rendered=True,
            )
        ),
    )
    monkeypatch.setattr(
        agentic_tools,
        "_try_rendered_fetch",
        AsyncMock(
            return_value=SimpleNamespace(
                status="ok", method="document_needed", text="doc hint", links=[], url="https://example.com/report"
            )
        ),
    )
    read_document = AsyncMock(
        return_value=agentic_tools.ToolOutcome(content_markdown="Extracted doc content.", method="document")
    )
    monkeypatch.setattr(agentic_tools, "read_document", read_document)

    outcome = await agentic_tools.fetch("https://example.com/report")

    assert outcome.method == "document"
    assert outcome.content_markdown == "Extracted doc content."
    read_document.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_fetch_cannot_earn_fetched_tier_but_real_fetch_can(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end tier check through the loop's real stamping helper: a finding
    whose source URL only ever produced an empty fetch earns NO tier (so a
    discrepancy on it can't supersede the briefing), while a genuinely read page
    earns "fetched". This is the load-bearing invariant."""
    url = "https://companiesmarketcap.com/berkshire-hathaway/marketcap/"
    monkeypatch.setattr(
        agentic_tools,
        "_fetch_plain",
        AsyncMock(
            return_value=SimpleNamespace(
                status="empty",
                method="plain",
                text="Plain fetch returned no extractable text.",
                links=[],
                url=url,
                escalate_rendered=True,
            )
        ),
    )
    monkeypatch.setattr(agentic_tools, "_try_rendered_fetch", AsyncMock(return_value=None))

    empty_outcome = await agentic_tools.fetch(url)
    empty_tiers = _harvest_verification_tiers("fetch", {"url": url}, empty_outcome)
    assert empty_tiers == {}

    real_outcome = agentic_tools.ToolOutcome(content_markdown="Berkshire market cap is ...", method="plain")
    real_tiers = _harvest_verification_tiers("fetch", {"url": url}, real_outcome)
    assert real_tiers == {"https://companiesmarketcap.com/berkshire-hathaway/marketcap": "fetched"}


def test_empty_method_maps_to_no_tier() -> None:
    """Belt-and-suspenders: even if a future edit passed status=="ok" through, the
    "empty" method itself maps to no tier — the guard is doubly deterministic."""
    assert _method_to_tier("empty") is None
    assert _method_to_tier("plain") == "fetched"


# Verbatim, from the q45191 run's archived transcript
# (backtests/research_archive/latest/45191.json -> gap_fill_v2.transcript, the tool result
# the driver read at step 11 and again from cache at step 23). 304 chars, HTTP 200,
# status="ok", method="rendered": ogimet.com's throttle interstitial standing in for the
# 2022-08-31 daily summary the loop asked for.
_OGIMET_THROTTLE_BODY = (
    "| Professional information about meteorological conditions in the world |  |  | \n"
    "| WEATHER MODEL FORECAST METEOGRAMS INDEXES UNDECODED REPORTS TEXT INFORMATION BUFR "
    "REPORTS GRAPHIC INFORMATION OTHER Advertisements | gsynext: Limit for old data queries "
    "exceeded. Permitted a query per 20 seconds per IP |\n"
)


def _plain_result(text: str, *, escalate_rendered: bool = False, url: str = "https://www.ogimet.com/summary") -> Any:
    return SimpleNamespace(
        status="ok", method="plain", text=text, links=[], url=url, escalate_rendered=escalate_rendered
    )


class TestThrottleInterstitialIsNotASuccess:
    """A host that throttles us answers 200 with a sentence instead of the page.

    q45191 (2026-08-10): three parallel ogimet.com fetches tripped that host's one-query-per-
    20-seconds rule, two came back as the interstitial under ``status: ok``, the window cache
    stored it, and the driver's own retry of the same URL was served the stored copy
    (``method: cache``) — so the retry it correctly made could not have succeeded. The
    exact-date reference class it published came to 4 years instead of 6, and the forecast
    under-committed to the state it had already named as the winner.

    The fix belongs on the tool, not in the prompt: that run's own pending lead reads
    "Ogimet rate-limited further historical August 31 queries (2022 and 2023)", so the driver
    had already diagnosed the throttle and still had no way back to the page.
    """

    @pytest.mark.asyncio
    async def test_a_rendered_interstitial_is_throttled_not_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The receipt's exact ladder path: the plain rung read too little and escalated, and
        # the rendered rung returned the interstitial.
        monkeypatch.setattr(
            agentic_tools, "_fetch_plain", AsyncMock(return_value=_plain_result("nav only", escalate_rendered=True))
        )
        monkeypatch.setattr(
            agentic_tools,
            "_try_rendered_fetch",
            AsyncMock(
                return_value=SimpleNamespace(
                    status="ok",
                    method="rendered",
                    text=_OGIMET_THROTTLE_BODY,
                    links=[],
                    url="https://www.ogimet.com/summary",
                )
            ),
        )

        outcome = await agentic_tools.fetch("https://www.ogimet.com/summary")

        assert outcome.status == "throttled"
        assert outcome.method == "throttled"
        # The interstitial text itself must not reach the driver as content.
        assert "Limit for old data queries exceeded" not in outcome.content_markdown
        # What the driver is told to do instead: retry later, and do not read the refusal as
        # the fact being unavailable (the null-result reading that cost q44799).
        assert "again later in the run" in outcome.content_markdown
        assert "do NOT read it as evidence that the fact is unavailable" in outcome.content_markdown

    @pytest.mark.asyncio
    async def test_a_plain_interstitial_is_throttled_without_a_rendered_hop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same body arriving on the plain rung with enough chars not to escalate.
        rendered = AsyncMock()
        monkeypatch.setattr(agentic_tools, "_fetch_plain", AsyncMock(return_value=_plain_result(_OGIMET_THROTTLE_BODY)))
        monkeypatch.setattr(agentic_tools, "_try_rendered_fetch", rendered)

        outcome = await agentic_tools.fetch("https://www.ogimet.com/summary")

        assert outcome.status == "throttled"
        rendered.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_interstitial_is_never_cached_so_the_retry_refetches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The half of the fix q45191 turned on: the driver's retry must be a real request."""
        fetch_plain = AsyncMock(return_value=_plain_result(_OGIMET_THROTTLE_BODY))
        monkeypatch.setattr(agentic_tools, "_fetch_plain", fetch_plain)

        first = await agentic_tools.fetch("https://www.ogimet.com/summary")
        assert first.status == "throttled"
        assert agentic_tools._FETCH_TEXT_CACHE == {}

        # The host has since let us through: the retry gets the page, not the stored refusal.
        fetch_plain.return_value = _plain_result("31/08/2022  41.1  Phoenix Sky Harbor")
        second = await agentic_tools.fetch("https://www.ogimet.com/summary")

        assert fetch_plain.await_count == 2
        assert second.status == "ok"
        assert second.method == "plain"
        assert "Phoenix Sky Harbor" in second.content_markdown

    @pytest.mark.asyncio
    async def test_a_short_legitimate_page_is_still_a_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Under the char cap but with no throttle phrase: real content, cached as before.

        The size half of the rule is what keeps this safe — a one-line official statement is
        the shape ``_plain_html_outcome`` deliberately keeps as "ok", and demoting it would
        cost more than the throttle it is trying to catch.
        """
        body = "The Ministry confirmed the vote will be held on 12 October 2026."
        assert len(body) < fetch_outcomes.FETCH_THROTTLE_PAGE_MAX_CHARS
        monkeypatch.setattr(agentic_tools, "_fetch_plain", AsyncMock(return_value=_plain_result(body)))

        outcome = await agentic_tools.fetch("https://example.gov/statement")

        assert outcome.status == "ok"
        assert outcome.method == "plain"
        assert outcome.content_markdown == body
        assert agentic_tools._FETCH_TEXT_CACHE["https://example.gov/statement"] == body

    @pytest.mark.asyncio
    async def test_a_long_page_about_rate_limits_is_still_a_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The phrase half alone would demote a page that merely discusses throttling."""
        body = "This API returns 429 Too Many Requests once you exceed the rate limit. " * 40
        assert len(body) > fetch_outcomes.FETCH_THROTTLE_PAGE_MAX_CHARS
        monkeypatch.setattr(agentic_tools, "_fetch_plain", AsyncMock(return_value=_plain_result(body)))

        outcome = await agentic_tools.fetch("https://example.com/api-docs")

        assert outcome.status == "ok"
        assert outcome.method == "plain"

    @pytest.mark.asyncio
    async def test_a_throttled_fetch_earns_no_verification_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end through the loop's real stamping helper: an interstitial can never be
        stamped ``fetched``, so a "correction" resting on it cannot supersede the briefing."""
        url = "https://www.ogimet.com/summary"
        monkeypatch.setattr(
            agentic_tools, "_fetch_plain", AsyncMock(return_value=_plain_result(_OGIMET_THROTTLE_BODY, url=url))
        )

        outcome = await agentic_tools.fetch(url)

        assert _harvest_verification_tiers("fetch", {"url": url}, outcome) == {}

    def test_throttled_method_maps_to_no_tier(self) -> None:
        # Belt-and-suspenders, exactly as for "empty": even if a future edit let
        # status=="ok" through, the method itself grants nothing.
        assert _method_to_tier("throttled") is None

    @pytest.mark.asyncio
    async def test_the_marker_names_the_rule_that_fired(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One greppable WARN per throttled fetch: without it the event has no trace at all,
        and ``phrase``/``chars`` are what let a prod fire be graded true or false positive."""
        monkeypatch.setattr(
            agentic_tools,
            "_fetch_plain",
            AsyncMock(return_value=_plain_result(_OGIMET_THROTTLE_BODY, url="https://www.ogimet.com/summary")),
        )

        with caplog.at_level(logging.WARNING, logger=agentic_tools.__name__):
            await agentic_tools.fetch("https://www.ogimet.com/summary")

        # 303, not the archived body's 304: `chars` is the whitespace-stripped length the rule
        # actually measured against the cap, so the field and the comparison can never disagree.
        assert (
            "AGENTIC_FETCH_THROTTLED: url=https://www.ogimet.com/summary method=plain chars=303 phrase=query per"
            in caplog.text
        )


class TestMatchedThrottlePhrase:
    """The predicate itself, anchored on the receipt and on the shapes it must not claim."""

    def test_the_q45191_body_matches_on_the_hosts_own_wording(self) -> None:
        assert fetch_outcomes.matched_throttle_phrase(_OGIMET_THROTTLE_BODY) == "query per"

    @pytest.mark.parametrize(
        "body",
        [
            "429 Too Many Requests",
            "Rate limit exceeded. Retry after 30 seconds.",
            "You have made too many requests; please slow down.",
            "Limit: 60 queries per minute per API key.",
        ],
    )
    def test_common_interstitial_wordings_match(self, body: str) -> None:
        assert fetch_outcomes.matched_throttle_phrase(body) is not None

    @pytest.mark.parametrize(
        "body",
        [
            "",
            "   \n  ",
            # "rate" alone is not the rule: the phrases are anchored on throttle idiom.
            "The unemployment rate fell to 4.1% in August.",
            "Growth is expected to slow down through 2027.",
            "The limit of the sequence exceeded every earlier bound.",
        ],
    )
    def test_ordinary_prose_does_not_trip_the_rule(self, body: str) -> None:
        assert fetch_outcomes.matched_throttle_phrase(body) is None

    def test_a_body_over_the_cap_is_a_page_whatever_it_says(self) -> None:
        # An interstitial is a sentence. A long body carrying the same words is a page about
        # throttling, and demoting it would discard content we really did read.
        body = "Rate limit exceeded. " * 200
        assert len(body) > fetch_outcomes.FETCH_THROTTLE_PAGE_MAX_CHARS
        assert fetch_outcomes.matched_throttle_phrase(body) is None


@pytest.mark.asyncio
async def test_try_rendered_fetch_uses_playwright_objects(monkeypatch: pytest.MonkeyPatch) -> None:
    semaphore_entries: list[str] = []

    class RecordingSemaphore:
        async def __aenter__(self) -> None:
            semaphore_entries.append("entered")

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    page = FakePage(html='<html><body><a href="/next">Next</a><p>Rendered body</p></body></html>')
    chromium = install_fake_playwright(monkeypatch, page, pinned=("example.com", "93.184.216.34"))
    # Patch our own fresh global semaphore (bound in THIS test's loop) rather than
    # leaning on the autouse fixture + import order — asyncio.Semaphore binds to the
    # running loop on first await, so a stale cross-file binding would raise here.
    monkeypatch.setattr(rendered_fetch, "_RENDERED_FETCH_GLOBAL_SEMAPHORE", asyncio.Semaphore(2))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._sem_for_host", lambda *_: RecordingSemaphore())
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value="Rendered body")
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    outcome = await agentic_tools._try_rendered_fetch("https://example.com/page")

    assert outcome is not None
    assert outcome.method == "rendered"
    assert outcome.links == ["https://example.com/next"]
    assert semaphore_entries == ["entered"]
    assert page.route_patterns == ["**/*"]
    (call,) = page.goto_calls
    assert call["url"] == "https://example.com/page"
    assert call["wait_until"] == "domcontentloaded"
    # The settle is taken OUT of the goto budget, so the rung's 35 s ceiling is
    # unchanged rather than lengthened by the wait that replaced networkidle.
    assert call["timeout"] == rendered_fetch.RENDER_TIMEOUT_MS - rendered_fetch.RENDER_SETTLE_MS
    assert page.context_kwargs["user_agent"]
    assert "Accept-Language" in page.context_kwargs["extra_http_headers"]
    assert chromium.headless == [True]


@pytest.mark.asyncio
async def test_rendered_fetch_launches_bounded_by_global_semaphore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix 1: concurrent headless-Chromium launches must never exceed the
    module-global cap, even across questions (the semaphore is per-process).

    Fires more concurrent _try_rendered_fetch calls than the cap — each to a
    distinct host so the per-host gate never serializes them — and gates each
    fake launch on a barrier so we can measure the true concurrent-launch peak.
    The peak must equal the cap (proving contention was actually reached) and
    never exceed it."""
    cap = 2
    monkeypatch.setattr(rendered_fetch, "_RENDERED_FETCH_GLOBAL_SEMAPHORE", asyncio.Semaphore(cap))

    live = 0
    peak = 0
    hold = asyncio.Event()
    at_cap = asyncio.Event()

    class _BarrierChromium(FakeChromium):
        async def launch(self, *, headless: bool, args: list[str] | None = None) -> FakeBrowser:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            if live >= cap:
                at_cap.set()
            try:
                await hold.wait()
            finally:
                live -= 1
            return await super().launch(headless=headless, args=args)

    page = FakePage(html="<html><body><p>rendered body</p></body></html>")
    install_fake_playwright(monkeypatch, page, chromium=_BarrierChromium(page))

    async def _pin_each_host(url: str) -> tuple[str, str]:
        # Each render is to its own host, and the transport holds the landing to the pinned one,
        # so the pin has to be the requested host as the real resolver returns it.
        return urlparse(url).hostname or "", "93.184.216.34"

    monkeypatch.setattr(rendered_fetch, "_resolve_pinned_host", _pin_each_host)
    monkeypatch.setattr("metaculus_bot.research.resolution_source._sem_for_host", lambda *_: asyncio.Semaphore(1))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value="rendered body")
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    tasks = [
        asyncio.create_task(agentic_tools._try_rendered_fetch(f"https://host{index}.example.com/page"))
        for index in range(cap + 3)
    ]

    # Let the first wave saturate the semaphore, then confirm it plateaued at
    # the cap while the launches are still parked on the barrier.
    await asyncio.wait_for(at_cap.wait(), timeout=1.0)
    for _ in range(5):
        await asyncio.sleep(0)
    assert live == cap
    assert peak == cap

    hold.set()
    results = await asyncio.gather(*tasks)

    assert all(result is not None and result.method == "rendered" for result in results)
    assert peak == cap


@pytest.mark.asyncio
async def test_rendered_fetch_route_guard_blocks_private_redirect_target(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-hop route guard must abort requests to non-public hosts.

    Simulates a page whose client-side redirect targets a private host: the
    guard registered via context.route re-runs is_public_http_url per request,
    aborts the private hop, and no below-bound content reaches the outcome.
    """
    aborted: list[tuple[str, str]] = []
    continued: list[str] = []

    class FakeRoute:
        def __init__(self, url: str) -> None:
            self.request = SimpleNamespace(url=url)

        async def continue_(self) -> None:
            continued.append(self.request.url)

        async def abort(self, error_code: str) -> None:
            aborted.append((self.request.url, error_code))

    class _RedirectingPage(FakePage):
        async def goto(self, url: str, *, wait_until: str, timeout: int) -> Any:  # noqa: ASYNC109  # mirrors Playwright API
            # Drive the guard the way Chromium would: the public main-frame
            # request continues; the page's client-side redirect to the
            # private host is aborted.
            guard = self.route_handler
            main_route = FakeRoute(url)
            await guard(main_route, main_route.request)
            private_route = FakeRoute("http://169.254.169.254/latest/meta-data/")
            await guard(private_route, private_route.request)
            return await super().goto(url, wait_until=wait_until, timeout=timeout)

    async def fake_is_public(url: str) -> bool:
        await asyncio.sleep(0)
        return "169.254.169.254" not in url

    install_fake_playwright(
        monkeypatch,
        _RedirectingPage(html="<html><body><p>public content only</p></body></html>"),
        pinned=("example.com", "93.184.216.34"),
    )
    # Self-sufficient global semaphore bound in this test's loop (see the sibling
    # rendered-fetch test) — avoids a cross-file stale-loop-binding RuntimeError.
    monkeypatch.setattr(rendered_fetch, "_RENDERED_FETCH_GLOBAL_SEMAPHORE", asyncio.Semaphore(2))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._sem_for_host", lambda *_: asyncio.Semaphore(1))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", fake_is_public)
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value="public content only")
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    outcome = await agentic_tools._try_rendered_fetch("https://example.com/page")

    assert outcome is not None
    assert continued == ["https://example.com/page"]
    assert aborted == [("http://169.254.169.254/latest/meta-data/", "blockedbyclient")]
    assert "169.254.169.254" not in outcome.text
    assert outcome.text == "public content only"


def _addrinfo(ip: str) -> list[tuple[Any, ...]]:
    """A minimal getaddrinfo return; only sockaddr[0] (the IP string) is read."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]


def _addrinfo6(ip: str) -> list[tuple[Any, ...]]:
    """IPv6 getaddrinfo return; sockaddr is (ip, port, flowinfo, scopeid)."""
    return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, 0, 0, 0))]


def test_host_resolver_rule_ipv4_is_bare() -> None:
    assert rendered_fetch._host_resolver_rule("example.com", "93.184.216.34") == (
        "--host-resolver-rules=MAP example.com 93.184.216.34"
    )


def test_host_resolver_rule_ipv6_is_bracketed() -> None:
    # Chromium's rule parser requires IPv6 literals bracketed in the MAP target.
    assert rendered_fetch._host_resolver_rule("example.com", "2606:2800:220:1:248:1893:25c8:1946") == (
        "--host-resolver-rules=MAP example.com [2606:2800:220:1:248:1893:25c8:1946]"
    )


@pytest.mark.asyncio
async def test_resolve_pinned_host_public_ip_returns_host_and_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=_addrinfo("93.184.216.34")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    assert await rendered_fetch._resolve_pinned_host("https://example.com/page") == ("example.com", "93.184.216.34")


@pytest.mark.asyncio
async def test_resolve_pinned_host_private_ip_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=_addrinfo("10.0.0.5")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    assert await rendered_fetch._resolve_pinned_host("https://internal.example.com/page") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_link_local_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # The Azure IMDS / cloud-metadata address is link-local — must fail closed.
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=_addrinfo("169.254.169.254")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    assert await rendered_fetch._resolve_pinned_host("https://rebind.example.com/page") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_rejects_when_any_address_disallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A rebinding host that resolves to BOTH a public and a private IP must be
    # rejected wholesale (same stance as the aiohttp preflight/FilteringResolver).
    mixed = _addrinfo("93.184.216.34") + _addrinfo("127.0.0.1")
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=mixed))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    assert await rendered_fetch._resolve_pinned_host("https://mixed.example.com/page") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_ipv6_public_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=_addrinfo6("2606:2800:220:1:248:1893:25c8:1946")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    assert await rendered_fetch._resolve_pinned_host("https://v6.example.com/page") == (
        "v6.example.com",
        "2606:2800:220:1:248:1893:25c8:1946",
    )


@pytest.mark.asyncio
async def test_resolve_pinned_host_ip_literal_public_pins_to_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    # An IP-literal host needs no DNS; getaddrinfo must not even be consulted.
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(side_effect=AssertionError("getaddrinfo must not run")))

    assert await rendered_fetch._resolve_pinned_host("https://93.184.216.34/page") == ("93.184.216.34", "93.184.216.34")


@pytest.mark.asyncio
async def test_resolve_pinned_host_ip_literal_private_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(side_effect=AssertionError("getaddrinfo must not run")))

    assert await rendered_fetch._resolve_pinned_host("http://127.0.0.1/latest/meta-data/") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_userinfo_and_scheme_fail_closed() -> None:
    # Userinfo defeats hostname trust; non-http(s) schemes are never fetched.
    assert await rendered_fetch._resolve_pinned_host("https://trusted@169.254.169.254/") is None
    assert await rendered_fetch._resolve_pinned_host("ftp://example.com/x") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_dns_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(side_effect=socket.gaierror("no such host")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    assert await rendered_fetch._resolve_pinned_host("https://nxdomain.example.com/page") is None


@pytest.mark.asyncio
async def test_rendered_fetch_skips_launch_when_host_not_pinnable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vetting fails (disallowed / unresolvable host) → Chromium is NOT launched
    and the rung returns the graceful-failure ``None`` the ladder degrades on."""
    chromium = install_fake_playwright(monkeypatch, FakePage(), pinned=None)
    monkeypatch.setattr("metaculus_bot.research.resolution_source._sem_for_host", lambda *_: asyncio.Semaphore(1))

    outcome = await agentic_tools._try_rendered_fetch("https://rebind.example.com/page")

    assert outcome is None
    assert chromium.launch_args == [], "Chromium must not launch for a non-pinnable host"


@pytest.mark.asyncio
async def test_rendered_fetch_launches_with_host_resolver_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vetting succeeds → Chromium launches with a ``--host-resolver-rules=MAP``
    arg pinning the main-frame host to exactly the vetted public IP."""
    chromium = install_fake_playwright(
        monkeypatch,
        FakePage(html="<html><body><p>Rendered body</p></body></html>"),
        pinned=("example.com", "93.184.216.34"),
    )
    monkeypatch.setattr("metaculus_bot.research.resolution_source._sem_for_host", lambda *_: asyncio.Semaphore(1))
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value="Rendered body")
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    outcome = await agentic_tools._try_rendered_fetch("https://example.com/page")

    assert outcome is not None
    assert outcome.method == "rendered"
    assert chromium.launch_args == [["--host-resolver-rules=MAP example.com 93.184.216.34"]]


@pytest.fixture
def _robots_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Say this host's robots.txt does not disallow ``Google-Extended``.

    The paid rung runs a one-request robots pre-check before it spends anything, and it goes
    through ``_fetch_plain``, so a test about the reader itself would otherwise either dial the
    network or answer the pre-check out of whatever fake body it wired for the document. Its own
    behavior is covered by ``TestUrlContextRobotsPreCheck``.
    """
    monkeypatch.setattr(agentic_tools, "_url_context_robots_skip", AsyncMock(return_value=False))


@pytest.fixture
def _no_local_document(monkeypatch: pytest.MonkeyPatch, _robots_allowed: None) -> None:
    """Make ``read_document``'s acquisition-first ladder hold nothing for the URL.

    ``read_document`` runs the free rungs (cache, plain, rendered) before it pays, so every test
    below about the PAID url_context rung has to say the free ones came back empty — otherwise
    the handler would dial the network instead of reaching the code under test. Requests
    ``_robots_allowed`` for the same reason: the pre-check ahead of the paid call is a request
    too.
    """
    monkeypatch.setattr(agentic_tools, "_acquire_local_document", AsyncMock(return_value=local_document.HeldDocument()))


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_local_document")
async def test_read_document_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))
    monkeypatch.setattr(
        agentic_tools,
        "_run_document_read_sync",
        MagicMock(return_value=("Quoted answer with dates.", 1, ["URL_RETRIEVAL_STATUS_SUCCESS"])),
    )

    outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What does it say?")

    assert outcome.status == "ok"
    assert outcome.method == "document"
    assert outcome.content_markdown == "Quoted answer with dates."


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_local_document")
async def test_read_document_genai_client_uses_bounded_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix 2 (genai half): the genai Client is built with a client-side timeout
    (ms) <= the read_document internal deadline, so a hung endpoint returns the
    to_thread worker instead of stranding it in the shared ThreadPoolExecutor.

    Patches the real ``google.genai.Client`` attribute and uses the real
    ``HttpOptions`` so the asserted timeout is the value that would ship."""
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    captured: dict[str, Any] = {}

    def fake_client(**kwargs: Any) -> Any:
        captured.update(kwargs)
        # Carries a SUCCESSFUL url_context retrieval: read_document now withholds the
        # 'fetched' tier when nothing was actually retrieved, so a metadata-less response
        # would (correctly) come back as an error and this timeout assertion would be
        # asserting on the wrong outcome.
        models = SimpleNamespace(generate_content=lambda **_: _document_response("Quoted answer."))
        return SimpleNamespace(models=models)

    monkeypatch.setattr("google.genai.Client", fake_client)
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What does it say?")

    assert outcome.status == "ok"
    http_options = captured["http_options"]
    # HttpOptions.timeout is in milliseconds; the read_document internal deadline is in seconds.
    assert http_options.timeout is not None
    assert http_options.timeout <= agentic_tools._READ_DOCUMENT_TIMEOUT_S * 1000


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_local_document")
async def test_read_document_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setattr(agentic_tools, "_READ_DOCUMENT_TIMEOUT_S", 0.01)

    async def slow_to_thread(fn, *args):
        await asyncio.sleep(0.05)
        return fn(*args)

    monkeypatch.setattr("asyncio.to_thread", slow_to_thread)
    monkeypatch.setattr(
        agentic_tools,
        "_run_document_read_sync",
        MagicMock(return_value=("late result", 1, ["URL_RETRIEVAL_STATUS_SUCCESS"])),
    )

    outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What does it say?")

    assert outcome.status == "error"
    assert "timed out" in outcome.content_markdown


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_local_document")
async def test_read_document_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What does it say?")

    assert outcome.status == "error"
    assert "GOOGLE_API_KEY" in outcome.content_markdown


def _document_response(text: str, *statuses: str) -> Any:
    """A fake Gemini response with the given text and url_context retrieval statuses.

    Defaults to one SUCCESS entry (a genuine document read). Shape mirrors the typed SDK
    models ``extract_url_context_telemetry`` reads: ``candidates[0].url_context_metadata
    .url_metadata[i].url_retrieval_status`` / ``.retrieved_url``.
    """
    statuses = statuses or ("URL_RETRIEVAL_STATUS_SUCCESS",)
    url_metadata = [
        SimpleNamespace(url_retrieval_status=status, retrieved_url=f"https://example.com/doc{i}")
        for i, status in enumerate(statuses)
    ]
    candidate = SimpleNamespace(url_context_metadata=SimpleNamespace(url_metadata=url_metadata))
    return SimpleNamespace(text=text, candidates=[candidate])


class TestReadDocumentRequiresRealRetrieval:
    """A ``document`` outcome earns the ``fetched`` tier, so it must be a real read.

    ``method="document"`` maps to ``fetched`` (``loop._METHOD_TO_TIER``), and only a
    ``fetched`` discrepancy enters the SUPERSEDE block that instructs every forecaster to
    override the briefing (``artifact.render_findings``). Gemini answers fluently from
    parametric memory when every url_context retrieval failed — the Q38195 failure mode —
    and the quote check cannot catch it here (WARN-only for this tool by design). So the
    retrieval count is the guard.
    """

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_no_local_document")
    async def test_all_retrievals_failed_withholds_the_fetched_tier(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        # Non-empty, confident-looking text plus a FAILED retrieval: exactly the shape
        # that used to be stamped `fetched` and could supersede the briefing.
        monkeypatch.setattr(
            "google.genai.Client",
            lambda **_: SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **_kw: _document_response(
                        "The filing states revenue of $4.2B.", "URL_RETRIEVAL_STATUS_ERROR"
                    )
                )
            ),
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

        with caplog.at_level(logging.WARNING, logger=agentic_tools.__name__):
            outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What is revenue?")

        assert outcome.status != "ok", "an unretrieved document must not come back as a successful read"
        assert "The filing states revenue of $4.2B." not in outcome.content_markdown, (
            "the ungrounded answer text must not reach the driver as document content"
        )
        assert "AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED" in caplog.text, (
            "the suppression must be greppable in the archived run logs, mirroring GEMINI_UNGROUNDED_SUPPRESSED"
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_no_local_document")
    async def test_no_url_context_metadata_at_all_withholds_the_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # url_context never reported back (tool didn't run / SDK attached nothing). Zero
        # successful retrievals either way, so the tier is withheld the same.
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            "google.genai.Client",
            lambda **_: SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **_kw: SimpleNamespace(text="Confident recall.", candidates=[])
                )
            ),
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

        outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What is revenue?")
        assert outcome.status != "ok"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_no_local_document")
    async def test_one_success_among_failures_still_counts_as_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The guard is "did ANY retrieval land", matching gemini_search's grounded-chunk
        # floor. A partially-failed multi-URL read still rests on real retrieved content.
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            "google.genai.Client",
            lambda **_: SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **_kw: _document_response(
                        "Quoted from the filing.",
                        "URL_RETRIEVAL_STATUS_ERROR",
                        "URL_RETRIEVAL_STATUS_SUCCESS",
                    )
                )
            ),
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

        outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What is revenue?")
        assert outcome.status == "ok"
        assert outcome.method == "document"
        assert outcome.content_markdown == "Quoted from the filing."

    def test_document_method_still_maps_to_the_fetched_tier(self) -> None:
        # If this ever stopped being true the guard above would be defending nothing;
        # pin the coupling that makes it load-bearing.
        assert _method_to_tier("document") == "fetched"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_no_local_document")
    async def test_the_suppression_warn_names_the_retrieval_statuses(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Zero successes is the same number for several different problems.

        A refused fetch, a retrieval that timed out and a url_context tool that never ran
        all read as ``n_url_success == 0``, and the run log used to carry only the URL. The
        status names are what separate "this host blocked us" from "the tool did not fire".
        """
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            "google.genai.Client",
            lambda **_: SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **_kw: _document_response(
                        "Confident recall.",
                        "URL_RETRIEVAL_STATUS_ERROR",
                        "URL_RETRIEVAL_STATUS_UNSAFE",
                    )
                )
            ),
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

        with caplog.at_level(logging.WARNING, logger=agentic_tools.__name__):
            outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What is revenue?")

        assert outcome.status != "ok"
        assert (
            "AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED: url=https://example.com/file.pdf "
            "statuses=URL_RETRIEVAL_STATUS_ERROR,URL_RETRIEVAL_STATUS_UNSAFE"
        ) in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_no_local_document")
    async def test_the_suppression_warn_reads_none_when_nothing_was_reported(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No url_metadata entry at all: the tool never reported back, which is its own case."""
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            "google.genai.Client",
            lambda **_: SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **_kw: SimpleNamespace(text="Confident recall.", candidates=[])
                )
            ),
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

        with caplog.at_level(logging.WARNING, logger=agentic_tools.__name__):
            outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What is revenue?")

        assert outcome.status != "ok"
        assert "statuses=none" in caplog.text

    def test_the_backend_returns_the_status_names_beside_the_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 3-tuple is the seam that feeds the WARN above; pin its shape and order."""
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            "google.genai.Client",
            lambda **_: SimpleNamespace(
                models=SimpleNamespace(
                    generate_content=lambda **_kw: _document_response(
                        "Quoted from the filing.",
                        "URL_RETRIEVAL_STATUS_ERROR",
                        "URL_RETRIEVAL_STATUS_SUCCESS",
                    )
                )
            ),
        )

        text, n_success, statuses = tool_backends._run_document_read_sync("https://example.com/f.pdf", "ask")

        assert text == "Quoted from the filing."
        assert n_success == 1
        assert statuses == ["URL_RETRIEVAL_STATUS_ERROR", "URL_RETRIEVAL_STATUS_SUCCESS"]


class TestReadDocumentClientConfig:
    """The reader's google-genai client: bounded retries, explicit thinking, logged spend.

    A bare ``genai.Client`` retries NOTHING (``retry_args(None)`` is
    ``stop_after_attempt(1)``), which is how two production reads died outright on a
    ``503 UNAVAILABLE``. The retry has to fit inside the existing HTTP budget rather than
    extend it, because this call runs in a ``to_thread`` worker that ``read_document``'s
    ``asyncio.wait_for`` cannot cancel — a longer worst case here means a pooled thread
    pinned for longer.
    """

    @staticmethod
    def _capture_client_kwargs(monkeypatch: pytest.MonkeyPatch, response: Any) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def fake_client(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_kw: response))

        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr("google.genai.Client", fake_client)
        return captured

    def test_retry_ladder_is_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = self._capture_client_kwargs(monkeypatch, _document_response("Quoted answer."))

        tool_backends._run_document_read_sync("https://example.com/f.pdf", "ask")

        retry_options = captured["http_options"].retry_options
        assert retry_options is not None, "without retry_options the SDK stops after one attempt"
        assert retry_options.attempts == GAP_FILL_V2_READER_HTTP_ATTEMPTS
        assert 503 in (retry_options.http_status_codes or []), "503 UNAVAILABLE is the failure this recovers"

    def test_every_attempt_plus_its_backoff_fits_the_existing_budget(self) -> None:
        """The arithmetic, pinned: the retries must not lengthen the in-thread worst case.

        ``_READ_DOCUMENT_HTTP_TIMEOUT_MS`` is the whole in-thread HTTP budget and stays where it
        was; the attempts divide it after the worst-case backoff sleeps are set aside. Today:
        2 x 26_500 + 2_000 = 55_000ms, exactly the previous single-attempt ceiling.

        What that 55 s no longer sits under is ``read_document``'s own wait, which became
        ``min(60, 65 - acquisition_elapsed)`` when the free local ladder landed ahead of the paid
        read — so 40 s at the 25 s acquisition cap. The second assertion is therefore against the
        60 s constant only (the no-acquisition case), and the money-relevant invariant is the
        separate test below.
        """
        worst_case_ms = GAP_FILL_V2_READER_HTTP_ATTEMPTS * tool_backends._READ_DOCUMENT_HTTP_PER_ATTEMPT_TIMEOUT_MS
        worst_case_ms += 1000 * gemini_retry_sleep_allowance_s(GAP_FILL_V2_READER_HTTP_ATTEMPTS)

        assert worst_case_ms <= tool_backends._READ_DOCUMENT_HTTP_TIMEOUT_MS
        assert worst_case_ms <= agentic_tools._READ_DOCUMENT_TIMEOUT_S * 1000

    def test_no_billed_attempt_is_dispatched_after_the_shortest_wait_fires(self) -> None:
        """The invariant that actually costs money, on the worst case for the wait.

        ``asyncio.wait_for`` cancels the coroutine but not the ``to_thread`` worker, so past ~10 s
        of local acquisition the worker outlives the wait by up to 15 s and finishes a call whose
        answer is discarded. That is one billed call at worst. Dispatching a NEW request after we
        stopped waiting would be a second, invisible to the ``GEMINI_USAGE`` line the read logs on
        return — so the last attempt has to START inside the shortest wait the handover can
        produce (65 - 25 = 40 s), which is what this pins: the attempts before the last one, plus
        every backoff sleep, fit in that window.
        """
        dispatch_of_last_attempt_ms = (
            GAP_FILL_V2_READER_HTTP_ATTEMPTS - 1
        ) * tool_backends._READ_DOCUMENT_HTTP_PER_ATTEMPT_TIMEOUT_MS
        dispatch_of_last_attempt_ms += 1000 * gemini_retry_sleep_allowance_s(GAP_FILL_V2_READER_HTTP_ATTEMPTS)
        shortest_wait_ms = 1000 * (agentic_tools._READ_DOCUMENT_TOTAL_BUDGET_S - agentic_tools._LOCAL_DOCUMENT_BUDGET_S)

        assert dispatch_of_last_attempt_ms <= shortest_wait_ms

    def test_thinking_level_is_set_explicitly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Quoting a fetched document back is the least reasoning-heavy Gemini call we make,
        and an unset level means the model's own default (HIGH on the Gemini 3 flash line)."""
        captured: dict[str, Any] = {}

        def fake_client(**_kwargs: Any) -> Any:
            def generate_content(**kwargs: Any) -> Any:
                captured.update(kwargs)
                return _document_response("Quoted answer.")

            return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))

        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr("google.genai.Client", fake_client)

        tool_backends._run_document_read_sync("https://example.com/f.pdf", "ask")

        thinking_config = captured["config"].thinking_config
        assert thinking_config is not None
        assert thinking_config.thinking_level == genai_types.ThinkingLevel(GAP_FILL_V2_READER_THINKING_LEVEL.upper())

    def test_token_spend_is_logged(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        """This call bills the operator's personal AI Studio key and used to record nothing."""
        response = _document_response("Quoted answer.")
        response.usage_metadata = SimpleNamespace(
            prompt_token_count=8000,
            tool_use_prompt_token_count=None,
            candidates_token_count=300,
            thoughts_token_count=120,
            total_token_count=8420,
        )
        self._capture_client_kwargs(monkeypatch, response)

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.gemini_usage"):
            tool_backends._run_document_read_sync("https://example.com/f.pdf", "ask")

        assert (
            f"GEMINI_USAGE: role=read_document model={GAP_FILL_V2_READER_MODEL} prompt_tokens=8000 "
            "tool_use_prompt_tokens=n/a candidates_tokens=300 thoughts_tokens=120 total_tokens=8420 "
            "search_queries=0"
        ) in caplog.text
        assert "question=" not in caplog.text, "the document reader holds no question id to carry"


# ---------------------------------------------------------------------------
# The local-document rung (2026-09-03). A PDF is decoded and passage-selected from bytes we
# already hold, and the paid Gemini url_context reader is spent only on a document we cannot
# read at all: measured over the 2026 summer season, that was 191 reader calls, nine documents
# over 100k tokens carried 67% of the retrieved tokens, and on the one file where both routes
# were tried local pypdf pulled 833,450 chars in 5.3 s while the paid read returned nothing.
# ---------------------------------------------------------------------------


def _long_pdf() -> bytes:
    """A PDF whose text runs past one fetch window, so pagination is exercised for real."""
    line = "The reported unemployment rate for May 2026 was 4.1 percent, revised from 4.0 percent. "
    return build_text_pdf([[line] * 60, [line] * 60])


def _no_paid_reader(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the paid url_context backend with a spy that must not be called."""
    reader = MagicMock(side_effect=AssertionError("the paid reader must not run for a document we hold"))
    monkeypatch.setattr(agentic_tools, "_run_document_read_sync", reader)
    return reader


class TestLocalPdfRung:
    @pytest.mark.asyncio
    async def test_a_pdf_with_a_text_layer_is_served_locally_and_paginates(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        read_body = _serve_pdf(monkeypatch, _long_pdf())
        read_document = AsyncMock()
        monkeypatch.setattr(agentic_tools, "read_document", read_document)

        with caplog.at_level(logging.INFO, logger=local_document.__name__):
            first = await agentic_tools.fetch("https://example.gov/report.pdf")

        assert first.method == "pdf_local"
        assert _method_to_tier(first.method) == "fetched", "we decoded the bytes the host served"
        assert "unemployment rate for May 2026" in first.content_markdown
        assert first.truncated is True, "a document past one window paginates like a long page"
        read_document.assert_not_awaited()
        assert "AGENTIC_FETCH_LOCAL_DOC: url=https://example.gov/report.pdf method=pdf_local" in caplog.text
        assert "pages=2 passages=n/a" in caplog.text, "a pdf_local fetch serves the text and selects nothing"

        # The continuation is served from the run cache: no second request, no second parse.
        second = await agentic_tools.fetch("https://example.gov/report.pdf", agentic_tools._FETCH_WINDOW_CHARS)
        assert second.method == "cache"
        assert read_body.await_count == 1

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_scanned_pdfs_escalation_neither_refetches_nor_reparses(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The parse that proved there is no text layer is held, so escalating costs one request."""
        read_body = _serve_pdf(monkeypatch, _scanned_pdf())
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("Model read.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        outcome = await agentic_tools.fetch("https://example.gov/scan.pdf")

        assert outcome.method == "document"
        assert read_body.await_count == 1, "the escalation reuses the held parse"

    @pytest.mark.asyncio
    async def test_a_document_over_the_byte_cap_reports_rather_than_escalating(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Too big to read locally is also too big to be worth having a model retrieve."""
        session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "application/pdf"}))
        monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
        monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
        monkeypatch.setattr(agentic_tools, "_read_response_body", AsyncMock(return_value=None))
        reader = _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.fetch("https://example.gov/huge.pdf")

        assert outcome.status == "error"
        assert outcome.method == "oversize_document"
        assert _method_to_tier(outcome.method) is None, "nothing was read, so no tier"
        assert "too large to read" in outcome.content_markdown.lower()
        reader.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_declared_pdf_body_is_read_under_the_document_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The 6.7 MB report local extraction reads in 5.3 s is over the ordinary page cap."""
        read_body = _serve_pdf(monkeypatch, build_text_pdf([["Long enough to count as a real text layer here."]]))

        await agentic_tools._fetch_plain("https://example.gov/report.pdf")

        assert read_body.await_args is not None
        assert read_body.await_args.kwargs["max_bytes"] == DOCUMENT_TEXT_PDF_MAX_BYTES

    @pytest.mark.asyncio
    async def test_a_partial_read_says_so_in_the_text_it_serves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A truncated PDF served as ``pdf_local`` must not read as the whole document.

        Extraction stops at the page cap or the time budget and reports which in ``truncated_by``.
        The digest header discloses it; this route serves the joined page text with no header at
        all, and ``FETCH_DESCRIPTION`` tells the driver "A PDF is read here, in full text" — so a
        driver that pages to the end sees ``truncated=False`` and can state an absence over pages
        nobody read. A 405-page report served 22,290 chars without mentioning the 5 it skipped.
        """
        truncated = extract_pdf_text(_long_pdf(), max_pages=1, max_seconds=5.0)
        assert truncated.truncated_by == "pages", "the fixture has to be a genuinely partial read"
        monkeypatch.setattr(local_document, "extract_pdf_text", MagicMock(return_value=truncated))

        result = await local_document.pdf_fetch_result(
            _long_pdf(), url="https://example.gov/report.pdf", content_type="application/pdf"
        )

        note = "[Partial document read: 2 pages; stopped at the 1-page read cap]"
        assert result.text.startswith(note)
        assert "unemployment rate" in result.text, "the disclosure leads the text, it does not replace it"
        # The other writer of that text into the run cache — a later read_document digests it flat.
        assert local_document.held_pdf(truncated).text.startswith(note)
        # A complete read gets no note at all, so ordinary output is unchanged.
        whole = extract_pdf_text(_long_pdf(), max_pages=400, max_seconds=5.0)
        assert whole.truncated_by == ""
        assert not local_document.held_pdf(whole).text.startswith("[Partial")

    @pytest.mark.asyncio
    async def test_the_pypdf_gate_is_shared_with_the_tier_1_rung(self) -> None:
        """One process-wide parse gate, not one per rung.

        pypdf is pure Python, so the two rungs' parses contend for the same GIL (6 concurrent
        parses of a 220-page document took 10.2 s against 1.66 s solo) and each parse's
        ``max_seconds`` is wall-clock, so unbounded contention truncates reads on a budget
        concurrency ate rather than document size. A gate private to this module would bound
        neither.
        """
        gate = http_fetch.pdf_parse_semaphore()
        await gate.acquire()
        await gate.acquire()

        parse = asyncio.create_task(
            local_document.pdf_fetch_result(
                _long_pdf(), url="https://example.gov/queued.pdf", content_type="application/pdf"
            )
        )
        for _ in range(3):
            await asyncio.sleep(0)
        assert not parse.done(), "both slots are held, so the v2 parse must be queued behind them"

        gate.release()
        result = await parse
        gate.release()

        assert result.method == "pdf_local"

    def test_the_held_parse_cache_is_run_scoped_state_the_suite_resets(self) -> None:
        # The autouse fixture calls exactly this, which is what keeps one test's held document
        # out of the next one's ladder.
        pdf = extract_pdf_text(_scanned_pdf(), max_pages=5, max_seconds=5.0)
        local_document.cache_document("https://example.gov/a.pdf", pdf)
        assert local_document.cached_document("https://example.gov/a.pdf") is not None

        local_document.clear_document_cache()
        assert local_document.cached_document("https://example.gov/a.pdf") is None


class TestReadDocumentAcquiresBeforePaying:
    """``read_document`` answers from the page's own text wherever it can get it.

    Its old shape sent every ask straight to a paid Gemini ``url_context`` call. Now the free
    rungs run first and their text is answered with a deterministic BM25 passage digest; the
    paid read is what a host that refuses us, or a document with no text layer, still needs.
    """

    @pytest.mark.asyncio
    async def test_a_fetchable_page_is_digested_with_no_model_call(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        page_text = (
            "Background on the tracker.\n\n"
            "The unemployment rate stood at 4.1 percent in May 2026.\n\n"
            "Unrelated methodology notes about seasonal adjustment.\n\n"
        ) * 3
        monkeypatch.setattr(
            agentic_tools,
            "_fetch_plain",
            AsyncMock(
                return_value=SimpleNamespace(
                    status="ok",
                    method="plain",
                    text=page_text,
                    links=[],
                    url="https://example.gov/tracker",
                    escalate_rendered=False,
                )
            ),
        )
        reader = _no_paid_reader(monkeypatch)
        # No Google key at all: the local digest is free and must not depend on one.
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        with caplog.at_level(logging.INFO, logger=local_document.__name__):
            outcome = await agentic_tools.read_document("https://example.gov/tracker", "unemployment rate May 2026")

        assert outcome.method == "digest_local"
        assert _method_to_tier(outcome.method) == "fetched"
        assert "4.1 percent in May 2026" in outcome.content_markdown
        assert "Most relevant passages for: unemployment rate May 2026" in outcome.content_markdown
        assert "[passage]" in outcome.content_markdown, "a page has no page numbers to claim"
        reader.assert_not_called()
        assert "method=digest_local" in caplog.text
        assert "pages=n/a" in caplog.text

    @pytest.mark.asyncio
    async def test_a_pdf_digest_carries_page_numbers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Page attribution is why a held parse beats the flat text: a cited page must be true."""
        _serve_pdf(monkeypatch, _long_pdf())
        _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document("https://example.gov/report.pdf", "unemployment rate revised")

        assert outcome.method == "digest_local"
        assert "[p.1]" in outcome.content_markdown or "[p.2]" in outcome.content_markdown

    @pytest.mark.asyncio
    async def test_text_a_fetch_already_read_is_digested_without_refetching(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        read_body = _serve_pdf(monkeypatch, _long_pdf())
        await agentic_tools.fetch("https://example.gov/report.pdf")
        _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document("https://example.gov/report.pdf", "unemployment rate revised")

        assert outcome.method == "digest_local"
        assert read_body.await_count == 1, "one request served both tools"

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_blocked_url_still_reaches_the_paid_reader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reader's remaining job: a host our own client cannot read from at all."""
        monkeypatch.setattr(
            agentic_tools,
            "_fetch_plain",
            AsyncMock(
                return_value=SimpleNamespace(
                    status="blocked",
                    method="plain",
                    text="Fetch blocked with HTTP 403.",
                    links=[],
                    url="https://sagaftra.org/contract",
                    escalate_rendered=False,
                )
            ),
        )
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools,
            "_run_document_read_sync",
            MagicMock(return_value=("The contract states a 3.5 percent increase.", 1, ["SUCCESS"])),
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        outcome = await agentic_tools.read_document("https://sagaftra.org/contract", "what increase is stated?")

        assert outcome.method == "document"
        assert outcome.content_markdown == "The contract states a 3.5 percent increase."

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_throttle_interstitial_is_never_digested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """q45191 again: a rate-limit sentence under HTTP 200 is not the document.

        Digesting it would put the host's refusal in front of the driver as the page's content.
        The paid reader dials from Gemini's address rather than ours, so it is the right next
        rung for exactly this case.
        """
        monkeypatch.setattr(
            agentic_tools,
            "_fetch_plain",
            AsyncMock(
                return_value=SimpleNamespace(
                    status="ok",
                    method="plain",
                    text="Limit for old data queries exceeded. Permitted a query per 20 seconds per IP",
                    links=[],
                    url="https://www.ogimet.com/summary",
                    escalate_rendered=False,
                )
            ),
        )
        # The interstitial leaves us holding nothing, so the ladder tries the browser next; it
        # is the plain rung's classification that is under test here.
        monkeypatch.setattr(agentic_tools, "_try_rendered_fetch", AsyncMock(return_value=None))
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("Model read.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        outcome = await agentic_tools.read_document("https://www.ogimet.com/summary", "the 2022-08-31 maximum")

        assert outcome.method == "document"
        assert "Limit for old data queries" not in outcome.content_markdown

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_an_image_is_not_digested_from_its_own_placeholder(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``document_needed`` result is "ok" and carries our own instruction as its text.

        Digesting that would answer the ask out of the sentence telling the driver to call this
        very tool. An image also ends the free ladder: no browser reads one either, so the rung
        must not spend a Chromium launch to find that out.
        """
        session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "image/png"}))
        monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
        monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
        rendered = AsyncMock()
        monkeypatch.setattr(agentic_tools, "_try_rendered_fetch", rendered)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("The chart shows 41.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        outcome = await agentic_tools.read_document("https://example.gov/chart.png", "what does the chart show?")

        assert outcome.method == "document"
        assert outcome.content_markdown == "The chart shows 41."
        rendered.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_oversize_document_is_reported_rather_than_paid_for(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            agentic_tools,
            "_fetch_plain",
            AsyncMock(
                return_value=SimpleNamespace(
                    status="error",
                    method="oversize_document",
                    text=local_document.oversize_message("https://example.gov/huge.pdf"),
                    links=[],
                    url="https://example.gov/huge.pdf",
                    escalate_rendered=False,
                )
            ),
        )
        reader = _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document("https://example.gov/huge.pdf", "anything")

        assert outcome.method == "oversize_document"
        assert "too large to read" in outcome.content_markdown.lower()
        reader.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_js_walled_page_is_rescued_by_the_browser_and_digested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rung read_document exists for: a page whose text only appears after JavaScript.

        The plain rung comes back with a shell, Chromium renders the real page, and the ask is
        answered from its text for free. Untested until 2026-09-03 — deleting the rendered block
        from the ladder left the whole suite green.
        """
        monkeypatch.setattr(
            agentic_tools,
            "_fetch_plain",
            AsyncMock(
                return_value=SimpleNamespace(
                    status="ok",
                    method="plain",
                    text="Loading…",
                    links=[],
                    url="https://example.gov/tracker",
                    escalate_rendered=True,
                )
            ),
        )
        rendered_text = (
            "Weekly tracker.\n\nThe unemployment rate stood at 4.1 percent in May 2026.\n\n"
            "Methodology notes about seasonal adjustment follow.\n\n"
        ) * 3
        monkeypatch.setattr(
            agentic_tools,
            "_try_rendered_fetch",
            AsyncMock(
                return_value=SimpleNamespace(
                    status="ok",
                    method="rendered",
                    text=rendered_text,
                    links=[],
                    url="https://example.gov/tracker",
                )
            ),
        )
        reader = _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document("https://example.gov/tracker", "unemployment rate May 2026")

        assert outcome.method == "digest_local"
        assert _method_to_tier(outcome.method) == "fetched", "the browser read the host's own bytes"
        assert "4.1 percent in May 2026" in outcome.content_markdown
        reader.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_subfloor_chrome_no_passage_matched_reaches_the_paid_reader(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A JavaScript shell the browser could not rescue must not be digested (F38/D5).

        The ladder holds the shell's navigation chrome, which is under the same content floor
        ``fetch`` escalates on, and no passage of it matches the ask. Digesting it stamped an
        unread page ``fetched`` — the tier that supersedes the briefing — while the tool
        description tells the driver a zero-passage digest means the document does not discuss
        what was asked. The paid reader, which dials from Gemini's address, is the right rung.
        """
        monkeypatch.setattr(
            agentic_tools,
            "_fetch_plain",
            AsyncMock(
                return_value=SimpleNamespace(
                    status="ok",
                    method="plain",
                    text="Home | Markets | Browse | Related questions | Sign in | Newsletter | About | Terms",
                    links=[],
                    url="https://manifold.markets/q/some-market",
                    escalate_rendered=True,
                )
            ),
        )
        monkeypatch.setattr(agentic_tools, "_try_rendered_fetch", AsyncMock(return_value=None))
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("Model read.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        with caplog.at_level(logging.INFO, logger=local_document.__name__):
            outcome = await agentic_tools.read_document(
                "https://manifold.markets/q/some-market", "what unemployment rate did the department report for May"
            )

        assert outcome.method == "document"
        assert "Related questions" not in outcome.content_markdown
        assert "AGENTIC_FETCH_LOCAL_DOC" not in caplog.text, (
            "the local-read marker must fire only where a digest is actually served"
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_short_page_that_answers_the_ask_is_still_digested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The floor alone would discard real short sources, so the match is the other half.

        ``fetch`` serves a thin-but-real page as a success by design (a one-line official
        statement), and here the same text answers the ask, so the free digest stands.
        """
        monkeypatch.setattr(
            agentic_tools,
            "_fetch_plain",
            AsyncMock(
                return_value=SimpleNamespace(
                    status="ok",
                    method="plain",
                    text="Statement: the unemployment rate stood at 4.1 percent in May 2026.",
                    links=[],
                    url="https://example.gov/statement",
                    escalate_rendered=True,
                )
            ),
        )
        monkeypatch.setattr(agentic_tools, "_try_rendered_fetch", AsyncMock(return_value=None))
        reader = _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document("https://example.gov/statement", "unemployment rate May 2026")

        assert outcome.method == "digest_local"
        assert "4.1 percent" in outcome.content_markdown
        reader.assert_not_called()


class TestTheDocumentedEscalationDoesNotRepeatItself:
    """``fetch`` then ``read_document`` on the same URL is the driver's documented path.

    ``READ_DOCUMENT_DESCRIPTION`` tells the driver to call it "for a URL where fetch returned
    status=blocked/js_wall/error", and ``fetch`` auto-escalates its own document results, so this
    population is the main path rather than an edge. Nothing recorded a failed acquisition, so
    both halves of the ladder ran twice: a second GET of an image whose body is never downloaded
    on either pass, and a second Chromium launch (100-300 MB, up to 35 s, out of a
    process-global cap of 2) for a page whose render just returned nothing. Only "rendered to
    nothing" is memoized — a blocked, errored or throttled GET stays re-requestable, because the
    driver is TOLD to retry those and 429 is in the retryable block set.
    """

    @staticmethod
    def _wire_launch_counting_playwright(monkeypatch: pytest.MonkeyPatch) -> FakeChromium:
        """A Chromium that renders every page to an empty DOM; the returned launcher counts launches."""
        monkeypatch.setattr("metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value=""))
        return install_fake_playwright(
            monkeypatch, FakePage(html="<html><body></body></html>"), pinned=("example.gov", "93.184.216.34")
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_an_image_escalation_issues_one_request_not_two(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An image is classified from its Content-Type, so a second GET learns nothing.

        The branch comment used to claim the escalation cost no second request; that holds for a
        scanned PDF, whose parse is cached under the URL, and never held for an image, whose body
        is not downloaded on either pass and so is cached nowhere.
        """
        session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "image/png"}))
        read_body = AsyncMock(return_value=b"")
        monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
        monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
        monkeypatch.setattr(agentic_tools, "_read_response_body", read_body)
        rendered = AsyncMock()
        monkeypatch.setattr(agentic_tools, "_try_rendered_fetch", rendered)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("The chart shows 41.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        outcome = await agentic_tools.fetch("https://example.gov/chart.png")

        assert outcome.method == "document"
        assert len(session.calls) == 1, "the escalation must not re-GET a URL the plain rung just classified"
        assert read_body.await_count == 0, "an image's body is never downloaded, on either pass"
        rendered.assert_not_awaited()  # no browser reads an image

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_js_wall_renders_once_then_read_document_pays(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chromium = self._wire_launch_counting_playwright(monkeypatch)
        plain = AsyncMock(
            return_value=SimpleNamespace(
                status="empty",
                method="plain",
                text="",
                links=[],
                url="https://example.gov/wall",
                escalate_rendered=True,
            )
        )
        monkeypatch.setattr(agentic_tools, "_fetch_plain", plain)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("Model read.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        walled = await agentic_tools.fetch("https://example.gov/wall")
        assert walled.status == "empty", "a page nothing could read must not be laundered as a success"

        outcome = await agentic_tools.read_document("https://example.gov/wall", "what does the tracker report?")

        assert outcome.method == "document", "the paid reader is the rung left for a page we cannot read"
        assert len(chromium.launch_args) == 1, "the second launch would re-learn what this run already knows"
        assert plain.await_count == 2, (
            "the plain GET is deliberately NOT negative-cached: the driver is told to retry these URLs"
        )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("_robots_allowed")
    async def test_a_throttled_fetch_then_read_document_does_re_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """q45191's contract: a refusal we were served must stay re-requestable.

        The host answered 200 with a rate-limit interstitial, which is not evidence about the
        page — caching that outcome (as the pre-fix code cached its text) is what made the
        driver's own retry impossible.
        """
        plain = AsyncMock(
            return_value=SimpleNamespace(
                status="ok",
                method="plain",
                text="Limit for old data queries exceeded. Permitted a query per 20 seconds per IP",
                links=[],
                url="https://www.ogimet.com/summary",
                escalate_rendered=False,
            )
        )
        monkeypatch.setattr(agentic_tools, "_fetch_plain", plain)
        monkeypatch.setattr(agentic_tools, "_try_rendered_fetch", AsyncMock(return_value=None))
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(
            agentic_tools, "_run_document_read_sync", MagicMock(return_value=("Model read.", 1, ["SUCCESS"]))
        )
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))

        throttled = await agentic_tools.fetch("https://www.ogimet.com/summary")
        outcome = await agentic_tools.read_document("https://www.ogimet.com/summary", "the 2022-08-31 maximum")

        assert throttled.status == "throttled"
        assert outcome.method == "document"
        assert plain.await_count == 2, "a throttle is not a fact about the page, so nothing memoizes it"


class TestUrlContextSizeGate:
    """A document we hold is never sent to the paid reader, and the biggest are the clearest case.

    The gate rides the same branch as the text it guards, so the two cannot disagree. It is
    there because the nine archived documents past it carried 67% of the season's reader tokens
    and the largest of them returned nothing at all for the money.
    """

    def test_the_gate_reads_chars_over_four_as_tokens(self) -> None:
        at_bound = "x" * (URL_CONTEXT_SIZE_GATE_TOKENS * 4)
        assert local_document.exceeds_url_context_size_gate(at_bound) is False
        assert local_document.exceeds_url_context_size_gate(at_bound + "xxxx") is True
        assert local_document.exceeds_url_context_size_gate("") is False

    @pytest.mark.asyncio
    async def test_a_huge_held_document_is_served_locally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        held = local_document.HeldDocument(text="revision " * (URL_CONTEXT_SIZE_GATE_TOKENS + 10))
        assert local_document.exceeds_url_context_size_gate(held.text)
        monkeypatch.setattr(agentic_tools, "_acquire_local_document", AsyncMock(return_value=held))
        reader = _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document("https://example.gov/833k.pdf", "revision")

        assert outcome.method == "digest_local"
        assert len(outcome.content_markdown) <= agentic_tools._FETCH_WINDOW_CHARS
        reader.assert_not_called()

    def test_the_digest_width_is_the_configured_one(self) -> None:
        # The digest's width is a knob in constants.py, not a literal at the call site.
        digest = local_document.digest_held(
            local_document.HeldDocument(text="\n\n".join(f"paragraph {i} about revisions. " * 30 for i in range(20))),
            ask="revisions",
            top_k=DOCUMENT_DIGEST_TOP_K,
            max_chars=8000,
            source_url="https://example.gov/a",
        )
        assert digest.passages == DOCUMENT_DIGEST_TOP_K


class TestRenderedRungSalvagesATimedOutNavigation:
    """The wait-condition fix (2026-09-03), measured on the 47-URL replay.

    ``page.goto(wait_until="networkidle")`` never returns on a page carrying a long-poll widget
    or an analytics beacon, and its TimeoutError used to discard the rung: 4 of the replay's 10
    render rescues came from pages whose DOM was complete anyway (both ballotpedia questions,
    both fts.unocha.org summaries). The rung now waits for DOM-ready plus a fixed settle and
    salvages ``page.content()`` when the navigation itself fails.
    """

    @staticmethod
    def _wire_playwright(monkeypatch: pytest.MonkeyPatch, page: FakePage) -> None:
        install_fake_playwright(monkeypatch, page, pinned=("example.com", "93.184.216.34"))
        monkeypatch.setattr(rendered_fetch, "_RENDERED_FETCH_GLOBAL_SEMAPHORE", asyncio.Semaphore(2))
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    @pytest.mark.asyncio
    async def test_a_goto_timeout_still_returns_the_rendered_dom(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = FakePage(
            goto_raises=_PlaywrightTimeoutError("Timeout 33000ms exceeded."),
            html="<html><body><p>The tracker reports 41 cases this week.</p></body></html>",
        )
        self._wire_playwright(monkeypatch, page)
        monkeypatch.setattr(
            "metaculus_bot.research.resolution_source._extract_main_text",
            MagicMock(return_value="The tracker reports 41 cases this week."),
        )

        result = await agentic_tools._try_rendered_fetch("https://ballotpedia.org/race")

        assert result is not None, "a timed-out goto with a complete DOM is a rescue, not a dead rung"
        assert result.status == "ok"
        assert result.method == "rendered"
        assert result.text == "The tracker reports 41 cases this week."
        assert page.settles == [rendered_fetch.RENDER_SETTLE_MS]

    @pytest.mark.asyncio
    async def test_a_navigation_error_with_no_dom_reads_as_rendered_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A genuine navigation failure salvages an empty about:blank, which is the same
        "rendered read nothing" outcome the rung produced before, so the ladder falls through
        exactly as it did."""

        self._wire_playwright(
            monkeypatch,
            FakePage(
                goto_raises=_PlaywrightError("net::ERR_NAME_NOT_RESOLVED"),
                html="<html><head></head><body></body></html>",
            ),
        )
        monkeypatch.setattr("metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value=None))

        result = await agentic_tools._try_rendered_fetch("https://example.com/gone")

        assert result is not None
        assert result.status == "error"
        assert result.method == "rendered"
        assert result.text == ""


class TestRenderedRungTimeoutAtTheV2Wrapper:
    """P3-1's transport bound RAISES ``TimeoutError`` rather than declining with ``None``, so the
    Tier-1 rung can record its own reason. This wrapper's callers only know ``None``, so it folds
    the timeout back into that signal. The memo for a cut-off render is the transport's own —
    written only when a browser actually ran, pinned in ``tests/test_rendered_fetch.py`` — so this
    wrapper writes neither memo on a timeout. The ceilings it already ran under are unchanged: the
    ``fetch`` tool's ``timeout_s`` and ``_LOCAL_DOCUMENT_BUDGET_S`` on the document ladder.
    """

    @pytest.mark.asyncio
    async def test_a_transport_timeout_declines_without_memoising_the_url_itself(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "https://example.com/keeps-navigating"

        async def _timed_out(target: str, **kwargs: object) -> None:
            del target, kwargs
            await asyncio.sleep(0)
            raise TimeoutError("rendered fetch DOM read exceeded 5000ms")

        monkeypatch.setattr(agentic_tools, "render_page", _timed_out)

        result = await agentic_tools._try_rendered_fetch(url)

        assert result is None
        assert rendered_fetch.rendered_to_nothing(url, memo_scope="gap_fill_v2") is False

    @pytest.mark.asyncio
    async def test_a_dom_over_the_ceiling_declines_without_memoising_the_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The transport's other non-``None`` decline. The page rendered, so it is not "rendered
        to nothing" and must not be memoised as such; this wrapper folds it into ``None`` like the
        timeout, because its callers know no other signal."""
        url = "https://example.com/five-megabyte-dashboard"

        async def _too_large(target: str, **kwargs: object) -> None:
            del kwargs
            await asyncio.sleep(0)
            raise rendered_fetch.RenderDomOverCeiling(f"the rendered DOM of {target} is over the ceiling")

        monkeypatch.setattr(agentic_tools, "render_page", _too_large)

        result = await agentic_tools._try_rendered_fetch(url)

        assert result is None
        assert rendered_fetch.rendered_to_nothing(url, memo_scope="gap_fill_v2") is False

    @pytest.mark.asyncio
    async def test_an_off_host_landing_declines_without_memoising_the_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The transport's third non-``None`` decline: Chromium's main frame landed off the pinned
        host, so the DOM was never read. This wrapper folds it into ``None`` like the other two,
        because its callers know no other signal, and nothing from that render reaches the driver.
        Not memoised: the page rendered, on a host that was not the one asked for."""
        url = "https://example.com/redirects-inward"

        async def _off_host(target: str, **kwargs: object) -> None:
            del kwargs
            await asyncio.sleep(0)
            raise rendered_fetch.RenderOffHost(
                requested_url=target, final_url="http://169.254.169.254/latest/meta-data/", pinned_host="example.com"
            )

        monkeypatch.setattr(agentic_tools, "render_page", _off_host)

        result = await agentic_tools._try_rendered_fetch(url)

        assert result is None
        assert rendered_fetch.rendered_to_nothing(url, memo_scope="gap_fill_v2") is False

    @pytest.mark.asyncio
    async def test_through_the_transport_an_off_host_landing_is_never_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end through the fake browser: the landing is checked before ``page.content()``,
        so no DOM from the other host exists for this ladder to extract, memoise or publish."""
        page = FakePage(html="<html><body><p>internal status page</p></body></html>", land_on="http://10.0.0.8/status")
        install_fake_playwright(monkeypatch, page, pinned=("example.com", "93.184.216.34"))
        monkeypatch.setattr(rendered_fetch, "_RENDERED_FETCH_GLOBAL_SEMAPHORE", asyncio.Semaphore(2))
        monkeypatch.setattr("metaculus_bot.research.resolution_source._sem_for_host", lambda *_: asyncio.Semaphore(1))
        extract = MagicMock(return_value="internal status page")
        monkeypatch.setattr("metaculus_bot.research.resolution_source._extract_main_text", extract)

        result = await agentic_tools._try_rendered_fetch("https://example.com/page")

        assert result is None
        assert page.content_reads == 0
        extract.assert_not_called()
        assert rendered_fetch.rendered_to_nothing("https://example.com/page", memo_scope="gap_fill_v2") is False
        assert page.teardown == ["unroute_all", "context.close", "browser.close"]


# ---------------------------------------------------------------------------
# The Google-Extended robots pre-check on the paid reader (2026-09-03). Proven live: the
# verification probe's url_context call returned URL_RETRIEVAL_STATUS_ERROR on
# internationalaisafetyreport.org, whose (Cloudflare-managed) robots.txt carries the two
# directives below, while the identical call on a robots-allowed host retrieved. A retry cannot
# change a host-policy refusal, so that read is spend with a known-zero return.
# ---------------------------------------------------------------------------

_GOOGLE_EXTENDED_BLOCKED_ROBOTS = "User-agent: Google-Extended\nDisallow: /\n"
_GENERIC_CRAWLER_BLOCKED_ROBOTS = "User-agent: *\nDisallow: /\nCrawl-delay: 10\n"


def _plain_result_stub(url: str, *, status: str, text: str) -> Any:
    return SimpleNamespace(
        status=status, method="plain", text=text, links=[], url=url, escalate_rendered=False, content_type=None
    )


def _fetch_plain_serving_robots(robots_txt: str | None, *, calls: list[str]) -> Any:
    """A ``_fetch_plain`` double: ``/robots.txt`` gets ``robots_txt``, every other URL a 403.

    The 403 is what leaves ``read_document``'s free ladder holding nothing, which is the state the
    pre-check guards. ``robots_txt=None`` stands for a robots.txt we could not read at all.
    """

    async def fetch_plain(url: str) -> Any:
        calls.append(url)
        await asyncio.sleep(0)
        if url.endswith("/robots.txt"):
            if robots_txt is None:
                return _plain_result_stub(url, status="error", text="Fetch error: TimeoutError:")
            return _plain_result_stub(url, status="ok", text=robots_txt)
        return _plain_result_stub(url, status="blocked", text="Fetch blocked with HTTP 403.")

    return fetch_plain


class TestUrlContextRobotsPreCheck:
    """One free request decides whether the paid ``url_context`` read can work at all.

    Only the ``Google-Extended`` group is consulted: our own free rungs dial under our own user
    agent and are unaffected, and this bot reads ``Content-Signal: use=reference`` as permitting
    reference use. So a host that blocks generic crawlers is still read for us, and only a host
    that names Gemini's retrieval token is skipped.
    """

    @staticmethod
    def _wire_reader(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        reader = MagicMock(return_value=("Model read.", 1, ["URL_RETRIEVAL_STATUS_SUCCESS"]))
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(agentic_tools, "_run_document_read_sync", reader)
        monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args, **kwargs: fn(*args, **kwargs)))
        return reader

    @pytest.mark.asyncio
    async def test_a_host_that_only_blocks_generic_crawlers_is_still_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            agentic_tools, "_fetch_plain", _fetch_plain_serving_robots(_GENERIC_CRAWLER_BLOCKED_ROBOTS, calls=calls)
        )
        reader = self._wire_reader(monkeypatch)

        outcome = await agentic_tools.read_document("https://who.int/data/gho", "what does the indicator read?")

        assert outcome.method == "document"
        assert outcome.content_markdown == "Model read."
        reader.assert_called_once()
        assert "https://who.int/robots.txt" in calls

    @pytest.mark.asyncio
    async def test_a_google_extended_disallow_skips_the_paid_read(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            agentic_tools, "_fetch_plain", _fetch_plain_serving_robots(_GOOGLE_EXTENDED_BLOCKED_ROBOTS, calls=calls)
        )
        reader = _no_paid_reader(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")

        url = "https://internationalaisafetyreport.org/chapters/2/"
        with caplog.at_level(logging.INFO, logger=agentic_tools.__name__):
            outcome = await agentic_tools.read_document(url, "what does the chapter say about compute?")

        reader.assert_not_called()
        assert outcome.status == "robots_disallowed"
        assert outcome.method == "document"
        assert _harvest_verification_tiers("read_document", {"url": url}, outcome) == {}, (
            "nothing was read, so nothing may claim the fetched tier"
        )
        assert "Retrying will not help" in outcome.content_markdown
        assert (
            "AGENTIC_URLCONTEXT_ROBOTS_SKIP: url=https://internationalaisafetyreport.org/chapters/2/ "
            "host=internationalaisafetyreport.org"
        ) in caplog.text

    @pytest.mark.asyncio
    async def test_an_unreadable_robots_txt_proceeds_to_the_reader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every ambiguity resolves toward paying: a wrong skip loses a document we could read."""
        calls: list[str] = []
        monkeypatch.setattr(agentic_tools, "_fetch_plain", _fetch_plain_serving_robots(None, calls=calls))
        reader = self._wire_reader(monkeypatch)

        outcome = await agentic_tools.read_document("https://example.gov/report", "what is the figure?")

        assert outcome.method == "document"
        reader.assert_called_once()

    @pytest.mark.asyncio
    async def test_robots_txt_is_read_once_per_host_per_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []
        monkeypatch.setattr(
            agentic_tools, "_fetch_plain", _fetch_plain_serving_robots(_GOOGLE_EXTENDED_BLOCKED_ROBOTS, calls=calls)
        )
        _no_paid_reader(monkeypatch)
        monkeypatch.setenv("GOOGLE_API_KEY", "key")

        first = await agentic_tools.read_document("https://internationalaisafetyreport.org/a", "ask one")
        second = await agentic_tools.read_document("https://internationalaisafetyreport.org/b", "ask two")

        assert first.status == "robots_disallowed"
        assert second.status == "robots_disallowed"
        robots_calls = [url for url in calls if url.endswith("/robots.txt")]
        assert len(robots_calls) == 1, "the verdict is cached per host, so a run pays one request for it"

    @pytest.mark.asyncio
    async def test_the_free_rungs_are_not_gated_by_robots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The pre-check guards the PAID rung only: a page our own client can read is served.

        ``internationalaisafetyreport.org``'s pages are readable by the plain fetch, and the
        operator's reading of its ``Content-Signal: use=reference`` is that reference use is
        permitted, so a disallowing robots.txt must not withhold a free read.
        """
        page = "The report states that training compute grew fourfold in 2026. " * 12

        async def fetch_plain(url: str) -> Any:
            await asyncio.sleep(0)
            if url.endswith("/robots.txt"):
                return _plain_result_stub(url, status="ok", text=_GOOGLE_EXTENDED_BLOCKED_ROBOTS)
            return _plain_result_stub(url, status="ok", text=page)

        monkeypatch.setattr(agentic_tools, "_fetch_plain", fetch_plain)
        reader = _no_paid_reader(monkeypatch)

        outcome = await agentic_tools.read_document(
            "https://internationalaisafetyreport.org/chapters/2/", "training compute growth"
        )

        assert outcome.method == "digest_local"
        assert "training compute grew fourfold" in outcome.content_markdown
        reader.assert_not_called()


class TestGoogleExtendedRobotsRules:
    """Reading one group out of a robots.txt, biased hard toward paying rather than skipping."""

    def test_a_generic_crawler_block_is_not_a_google_extended_block(self) -> None:
        # The reason this is not ``urllib.robotparser``: its ``can_fetch("Google-Extended", ...)``
        # falls back to the ``User-agent: *`` group (verified on 3.12.12), which would skip the
        # paid read on every host that merely disallows crawlers — a far broader policy than the
        # one that was approved.
        assert robots_policy.google_extended_disallows(_GENERIC_CRAWLER_BLOCKED_ROBOTS, "/data") is False

    def test_the_receipt_host_shape_disallows_every_path(self) -> None:
        assert robots_policy.google_extended_disallows(_GOOGLE_EXTENDED_BLOCKED_ROBOTS, "/chapters/2/") is True
        assert robots_policy.google_extended_disallows(_GOOGLE_EXTENDED_BLOCKED_ROBOTS, "") is True

    def test_the_group_can_name_several_agents(self) -> None:
        robots = "User-agent: GPTBot\nUser-agent: google-extended\nDisallow: /reports\n"
        assert robots_policy.google_extended_disallows(robots, "/reports/2026") is True
        assert robots_policy.google_extended_disallows(robots, "/about") is False

    def test_rules_stop_belonging_to_a_group_at_the_next_agent_line(self) -> None:
        robots = "User-agent: Google-Extended\nDisallow: /secret\n\nUser-agent: *\nDisallow: /\n"
        assert robots_policy.google_extended_disallows(robots, "/secret/a") is True
        assert robots_policy.google_extended_disallows(robots, "/public") is False

    def test_the_longest_matching_rule_wins_and_allow_takes_a_tie(self) -> None:
        robots = "User-agent: Google-Extended\nDisallow: /docs\nAllow: /docs/public\n"
        assert robots_policy.google_extended_disallows(robots, "/docs/private") is True
        assert robots_policy.google_extended_disallows(robots, "/docs/public/a") is False
        tie = "User-agent: Google-Extended\nDisallow: /\nAllow: /\n"
        assert robots_policy.google_extended_disallows(tie, "/anything") is False

    def test_comments_and_blank_lines_are_ignored(self) -> None:
        robots = "# policy\n\nUser-agent: Google-Extended  # the AI token\nDisallow: /  # everything\n"
        assert robots_policy.google_extended_disallows(robots, "/x") is True

    def test_an_empty_disallow_allows_everything(self) -> None:
        assert robots_policy.google_extended_disallows("User-agent: Google-Extended\nDisallow:\n", "/x") is False

    @pytest.mark.parametrize("rule", ["/*/private", "/*.pdf$", "*"])
    def test_a_rule_needing_glob_matching_is_left_alone(self, rule: str) -> None:
        # Not modelled, so it cannot disallow: the read proceeds and is paid for, which is the
        # only direction an unmatched rule is allowed to fail in.
        robots = f"User-agent: Google-Extended\nDisallow: {rule}\n"
        assert robots_policy.google_extended_disallows(robots, "/reports/private/a.pdf") is False

    def test_a_trailing_star_is_the_prefix_it_decorates(self) -> None:
        robots = "User-agent: Google-Extended\nDisallow: /reports*\n"
        assert robots_policy.google_extended_disallows(robots, "/reports/2026") is True
        assert robots_policy.google_extended_disallows(robots, "/about") is False

    def test_an_empty_robots_txt_says_nothing(self) -> None:
        assert robots_policy.google_extended_disallows("", "/x") is False
