from __future__ import annotations

import asyncio
import logging
import socket
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from playwright.async_api import Error as _PlaywrightError

from metaculus_bot.research import providers as research_providers
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.agentic.loop import _harvest_verification_tiers, _method_to_tier, _tool_schemas


class _FakeResponse:
    def __init__(self, *, status: int, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.headers = headers or {}

    async def __aenter__(self) -> "_FakeResponse":
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

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def get(self, url: str, *, allow_redirects: bool = False) -> _FakeResponse:
        self.calls.append((url, allow_redirects))
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]


@pytest.fixture(autouse=True)
def _reset_tool_state() -> None:
    agentic_tools._FETCH_TEXT_CACHE.clear()
    agentic_tools._FETCH_LINKS_CACHE.clear()
    agentic_tools._FETCH_HOST_SEMAPHORES.clear()
    agentic_tools._PLAYWRIGHT_WARNED = False
    # Fresh module-global rendered-fetch semaphore per test: construction is
    # loop-free in 3.12, so this prevents a contended acquire in one test's
    # event loop from leaking a loop binding into a later test.
    agentic_tools._RENDERED_FETCH_GLOBAL_SEMAPHORE = asyncio.Semaphore(2)


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

    monkeypatch.setattr("asyncio.sleep", AsyncMock(side_effect=lambda delay: sleeps.append(delay)))
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


class _FakeAskNewsSdk:
    """Async-context AskNews SDK double with a scripted search_news."""

    def __init__(self, search_news_mock: AsyncMock) -> None:
        self.news = SimpleNamespace(search_news=search_news_mock)

    async def __aenter__(self) -> "_FakeAskNewsSdk":
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
async def test_fetch_pdf_content_type_auto_escalates_to_document(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "application/pdf"}))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
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
    session = _FakeSession(_FakeResponse(status=200, headers={"Content-Type": "application/pdf"}))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._get_session", lambda: session)
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
    monkeypatch.setattr(agentic_tools, "_resolve_pinned_host", AsyncMock(return_value=("example.com", "93.184.216.34")))

    async def blocking_read(resp: object, label: str) -> bytes:
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

    class FakePage:
        async def goto(self, url: str, *, wait_until: str, timeout: int) -> SimpleNamespace:
            return SimpleNamespace(headers={"content-type": "text/html"})

        async def content(self) -> str:
            return "<html><body><p>rendered body</p></body></html>"

    class FakeContext:
        async def route(self, pattern: str, handler) -> None:
            return None

        async def unroute_all(self, *, behavior: str | None = None) -> None:
            return None

        async def new_page(self) -> FakePage:
            return FakePage()

        async def close(self) -> None:
            return None

    class FakeBrowser:
        async def new_context(self, **kwargs) -> FakeContext:
            return FakeContext()

        async def close(self) -> None:
            return None

    class FakeChromium:
        async def launch(self, *, headless: bool, args: list[str] | None = None) -> FakeBrowser:
            return FakeBrowser()

    class FakePlaywrightManager:
        chromium = FakeChromium()

        async def __aenter__(self) -> "FakePlaywrightManager":
            # Runs strictly after _try_rendered_fetch acquires the host gate.
            events.append("rendered_started")
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "playwright.async_api",
        SimpleNamespace(async_playwright=lambda: FakePlaywrightManager(), Error=_PlaywrightError),
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
    assert rendered_result is not None and rendered_result.method == "rendered"
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
    monkeypatch.setattr(agentic_tools, "_resolve_pinned_host", AsyncMock(return_value=("example.com", "93.184.216.34")))
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value="body text " * 60)
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    captured: dict[str, Any] = {}

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

    class FakePage:
        async def goto(self, url: str, *, wait_until: str, timeout: int) -> SimpleNamespace:
            return SimpleNamespace(headers={"content-type": "text/html"})

        async def content(self) -> str:
            return "<html><body><p>rendered body</p></body></html>"

    class FakeContext:
        async def route(self, pattern: str, handler) -> None:
            captured["guard"] = handler

        async def unroute_all(self, *, behavior: str | None = None) -> None:
            captured["unroute_behavior"] = behavior

        async def new_page(self) -> FakePage:
            return FakePage()

        async def close(self) -> None:
            captured["context_closed"] = True

    class FakeBrowser:
        async def new_context(self, **kwargs) -> FakeContext:
            return FakeContext()

        async def close(self) -> None:
            captured["browser_closed"] = True

    class FakeChromium:
        async def launch(self, *, headless: bool, args: list[str] | None = None) -> FakeBrowser:
            return FakeBrowser()

    class FakePlaywrightManager:
        chromium = FakeChromium()

        async def __aenter__(self) -> "FakePlaywrightManager":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "playwright.async_api",
        SimpleNamespace(async_playwright=lambda: FakePlaywrightManager(), Error=_PlaywrightError),
    )

    result = await agentic_tools._try_rendered_fetch("https://example.com/page")
    assert result is not None and result.method == "rendered"

    # Teardown drained the handlers before close (Playwright's remedy for the storm).
    assert captured["unroute_behavior"] == "ignoreErrors"
    assert captured.get("context_closed") is True
    assert captured.get("browser_closed") is True

    guard = captured["guard"]

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

    routes: list[str] = []

    class FakeContext:
        async def route(self, pattern: str, handler) -> None:
            routes.append(pattern)

        async def unroute_all(self, *, behavior: str | None = None) -> None:
            return None

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
        async def launch(self, *, headless: bool, args: list[str] | None = None) -> FakeBrowser:
            assert headless is True
            return FakeBrowser()

    class FakePlaywrightManager:
        chromium = FakeChromium()

        async def __aenter__(self) -> "FakePlaywrightManager":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(agentic_tools, "_resolve_pinned_host", AsyncMock(return_value=("example.com", "93.184.216.34")))
    # Patch our own fresh global semaphore (bound in THIS test's loop) rather than
    # leaning on the autouse fixture + import order — asyncio.Semaphore binds to the
    # running loop on first await, so a stale cross-file binding would raise here.
    monkeypatch.setattr(agentic_tools, "_RENDERED_FETCH_GLOBAL_SEMAPHORE", asyncio.Semaphore(2))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._sem_for_host", lambda *_: RecordingSemaphore())
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value="Rendered body")
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))
    monkeypatch.setitem(
        sys.modules,
        "playwright.async_api",
        SimpleNamespace(async_playwright=lambda: FakePlaywrightManager(), Error=_PlaywrightError),
    )

    outcome = await agentic_tools._try_rendered_fetch("https://example.com/page")

    assert outcome is not None
    assert outcome.method == "rendered"
    assert outcome.links == ["https://example.com/next"]
    assert semaphore_entries == ["entered"]
    assert routes == ["**/*"]


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
    monkeypatch.setattr(agentic_tools, "_RENDERED_FETCH_GLOBAL_SEMAPHORE", asyncio.Semaphore(cap))

    live = 0
    peak = 0
    hold = asyncio.Event()
    at_cap = asyncio.Event()

    class FakePage:
        async def goto(self, url: str, *, wait_until: str, timeout: int) -> SimpleNamespace:
            return SimpleNamespace(headers={"content-type": "text/html"})

        async def content(self) -> str:
            return "<html><body><p>rendered body</p></body></html>"

    class FakeContext:
        async def route(self, pattern: str, handler) -> None:
            return None

        async def unroute_all(self, *, behavior: str | None = None) -> None:
            return None

        async def new_page(self) -> FakePage:
            return FakePage()

        async def close(self) -> None:
            return None

    class FakeBrowser:
        async def new_context(self, **kwargs) -> FakeContext:
            return FakeContext()

        async def close(self) -> None:
            return None

    class FakeChromium:
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
            return FakeBrowser()

    class FakePlaywrightManager:
        chromium = FakeChromium()

        async def __aenter__(self) -> "FakePlaywrightManager":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(
        agentic_tools, "_resolve_pinned_host", AsyncMock(return_value=("host.example.com", "93.184.216.34"))
    )
    monkeypatch.setattr("metaculus_bot.research.resolution_source._sem_for_host", lambda *_: asyncio.Semaphore(1))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", AsyncMock(return_value=True))
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value="rendered body")
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))
    monkeypatch.setitem(
        sys.modules,
        "playwright.async_api",
        SimpleNamespace(async_playwright=lambda: FakePlaywrightManager(), Error=_PlaywrightError),
    )

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

    guard_holder: list = []

    class FakePage:
        async def goto(self, url: str, *, wait_until: str, timeout: int) -> SimpleNamespace:
            # Drive the guard the way Chromium would: the public main-frame
            # request continues; the page's client-side redirect to the
            # private host is aborted.
            guard = guard_holder[0]
            main_route = FakeRoute(url)
            await guard(main_route, main_route.request)
            private_route = FakeRoute("http://169.254.169.254/latest/meta-data/")
            await guard(private_route, private_route.request)
            return SimpleNamespace(headers={"content-type": "text/html"})

        async def content(self) -> str:
            return "<html><body><p>public content only</p></body></html>"

    class FakeContext:
        async def route(self, pattern: str, handler) -> None:
            guard_holder.append(handler)

        async def unroute_all(self, *, behavior: str | None = None) -> None:
            return None

        async def new_page(self) -> FakePage:
            return FakePage()

        async def close(self) -> None:
            return None

    class FakeBrowser:
        async def new_context(self, **kwargs) -> FakeContext:
            return FakeContext()

        async def close(self) -> None:
            return None

    class FakeChromium:
        async def launch(self, *, headless: bool, args: list[str] | None = None) -> FakeBrowser:
            return FakeBrowser()

    class FakePlaywrightManager:
        chromium = FakeChromium()

        async def __aenter__(self) -> "FakePlaywrightManager":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    async def fake_is_public(url: str) -> bool:
        return "169.254.169.254" not in url

    monkeypatch.setattr(agentic_tools, "_resolve_pinned_host", AsyncMock(return_value=("example.com", "93.184.216.34")))
    # Self-sufficient global semaphore bound in this test's loop (see the sibling
    # rendered-fetch test) — avoids a cross-file stale-loop-binding RuntimeError.
    monkeypatch.setattr(agentic_tools, "_RENDERED_FETCH_GLOBAL_SEMAPHORE", asyncio.Semaphore(2))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._sem_for_host", lambda *_: asyncio.Semaphore(1))
    monkeypatch.setattr("metaculus_bot.research.resolution_source.is_public_http_url", fake_is_public)
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value="public content only")
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))
    monkeypatch.setitem(
        sys.modules,
        "playwright.async_api",
        SimpleNamespace(async_playwright=lambda: FakePlaywrightManager(), Error=_PlaywrightError),
    )

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
    assert agentic_tools._host_resolver_rule("example.com", "93.184.216.34") == (
        "--host-resolver-rules=MAP example.com 93.184.216.34"
    )


def test_host_resolver_rule_ipv6_is_bracketed() -> None:
    # Chromium's rule parser requires IPv6 literals bracketed in the MAP target.
    assert agentic_tools._host_resolver_rule("example.com", "2606:2800:220:1:248:1893:25c8:1946") == (
        "--host-resolver-rules=MAP example.com [2606:2800:220:1:248:1893:25c8:1946]"
    )


@pytest.mark.asyncio
async def test_resolve_pinned_host_public_ip_returns_host_and_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=_addrinfo("93.184.216.34")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    assert await agentic_tools._resolve_pinned_host("https://example.com/page") == ("example.com", "93.184.216.34")


@pytest.mark.asyncio
async def test_resolve_pinned_host_private_ip_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=_addrinfo("10.0.0.5")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    assert await agentic_tools._resolve_pinned_host("https://internal.example.com/page") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_link_local_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    # The Azure IMDS / cloud-metadata address is link-local — must fail closed.
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=_addrinfo("169.254.169.254")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    assert await agentic_tools._resolve_pinned_host("https://rebind.example.com/page") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_rejects_when_any_address_disallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # A rebinding host that resolves to BOTH a public and a private IP must be
    # rejected wholesale (same stance as the aiohttp preflight/FilteringResolver).
    mixed = _addrinfo("93.184.216.34") + _addrinfo("127.0.0.1")
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=mixed))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    assert await agentic_tools._resolve_pinned_host("https://mixed.example.com/page") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_ipv6_public_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(return_value=_addrinfo6("2606:2800:220:1:248:1893:25c8:1946")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    assert await agentic_tools._resolve_pinned_host("https://v6.example.com/page") == (
        "v6.example.com",
        "2606:2800:220:1:248:1893:25c8:1946",
    )


@pytest.mark.asyncio
async def test_resolve_pinned_host_ip_literal_public_pins_to_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    # An IP-literal host needs no DNS; getaddrinfo must not even be consulted.
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(side_effect=AssertionError("getaddrinfo must not run")))

    assert await agentic_tools._resolve_pinned_host("https://93.184.216.34/page") == ("93.184.216.34", "93.184.216.34")


@pytest.mark.asyncio
async def test_resolve_pinned_host_ip_literal_private_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(side_effect=AssertionError("getaddrinfo must not run")))

    assert await agentic_tools._resolve_pinned_host("http://127.0.0.1/latest/meta-data/") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_userinfo_and_scheme_fail_closed() -> None:
    # Userinfo defeats hostname trust; non-http(s) schemes are never fetched.
    assert await agentic_tools._resolve_pinned_host("https://trusted@169.254.169.254/") is None
    assert await agentic_tools._resolve_pinned_host("ftp://example.com/x") is None


@pytest.mark.asyncio
async def test_resolve_pinned_host_dns_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(side_effect=socket.gaierror("no such host")))
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))

    assert await agentic_tools._resolve_pinned_host("https://nxdomain.example.com/page") is None


@pytest.mark.asyncio
async def test_rendered_fetch_skips_launch_when_host_not_pinnable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vetting fails (disallowed / unresolvable host) → Chromium is NOT launched
    and the rung returns the graceful-failure ``None`` the ladder degrades on."""
    launched: list[Any] = []

    class FakeChromium:
        async def launch(self, *, headless: bool, args: list[str] | None = None) -> Any:
            launched.append(args)
            raise AssertionError("Chromium must not launch for a non-pinnable host")

    class FakePlaywrightManager:
        chromium = FakeChromium()

        async def __aenter__(self) -> "FakePlaywrightManager":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(agentic_tools, "_resolve_pinned_host", AsyncMock(return_value=None))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._sem_for_host", lambda *_: asyncio.Semaphore(1))
    monkeypatch.setitem(
        sys.modules,
        "playwright.async_api",
        SimpleNamespace(async_playwright=lambda: FakePlaywrightManager(), Error=_PlaywrightError),
    )

    outcome = await agentic_tools._try_rendered_fetch("https://rebind.example.com/page")

    assert outcome is None
    assert launched == []


@pytest.mark.asyncio
async def test_rendered_fetch_launches_with_host_resolver_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Vetting succeeds → Chromium launches with a ``--host-resolver-rules=MAP``
    arg pinning the main-frame host to exactly the vetted public IP."""
    launch_args: list[list[str] | None] = []

    class FakePage:
        async def goto(self, url: str, *, wait_until: str, timeout: int) -> SimpleNamespace:
            return SimpleNamespace(headers={"content-type": "text/html"})

        async def content(self) -> str:
            return "<html><body><p>Rendered body</p></body></html>"

    class FakeContext:
        async def route(self, pattern: str, handler) -> None:
            return None

        async def unroute_all(self, *, behavior: str | None = None) -> None:
            return None

        async def new_page(self) -> FakePage:
            return FakePage()

        async def close(self) -> None:
            return None

    class FakeBrowser:
        async def new_context(self, **kwargs) -> FakeContext:
            return FakeContext()

        async def close(self) -> None:
            return None

    class FakeChromium:
        async def launch(self, *, headless: bool, args: list[str] | None = None) -> FakeBrowser:
            launch_args.append(args)
            return FakeBrowser()

    class FakePlaywrightManager:
        chromium = FakeChromium()

        async def __aenter__(self) -> "FakePlaywrightManager":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr(agentic_tools, "_resolve_pinned_host", AsyncMock(return_value=("example.com", "93.184.216.34")))
    monkeypatch.setattr("metaculus_bot.research.resolution_source._sem_for_host", lambda *_: asyncio.Semaphore(1))
    monkeypatch.setattr(
        "metaculus_bot.research.resolution_source._extract_main_text", MagicMock(return_value="Rendered body")
    )
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))
    monkeypatch.setitem(
        sys.modules,
        "playwright.async_api",
        SimpleNamespace(async_playwright=lambda: FakePlaywrightManager(), Error=_PlaywrightError),
    )

    outcome = await agentic_tools._try_rendered_fetch("https://example.com/page")

    assert outcome is not None and outcome.method == "rendered"
    assert len(launch_args) == 1
    assert launch_args[0] == ["--host-resolver-rules=MAP example.com 93.184.216.34"]


@pytest.mark.asyncio
async def test_read_document_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setattr("asyncio.to_thread", AsyncMock(side_effect=lambda fn, *args: fn(*args)))
    monkeypatch.setattr(
        agentic_tools, "_run_document_read_sync", MagicMock(return_value=("Quoted answer with dates.", 1))
    )

    outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What does it say?")

    assert outcome.status == "ok"
    assert outcome.method == "document"
    assert outcome.content_markdown == "Quoted answer with dates."


@pytest.mark.asyncio
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
async def test_read_document_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "key")
    monkeypatch.setattr(agentic_tools, "_READ_DOCUMENT_TIMEOUT_S", 0.01)

    async def slow_to_thread(fn, *args):
        await asyncio.sleep(0.05)
        return fn(*args)

    monkeypatch.setattr("asyncio.to_thread", slow_to_thread)
    monkeypatch.setattr(agentic_tools, "_run_document_read_sync", MagicMock(return_value=("late result", 1)))

    outcome = await agentic_tools.read_document("https://example.com/file.pdf", "What does it say?")

    assert outcome.status == "error"
    assert "timed out" in outcome.content_markdown


@pytest.mark.asyncio
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
