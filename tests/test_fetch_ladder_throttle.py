"""End-to-end 200-interstitial checks for the shared fetch ladder."""

from __future__ import annotations

import logging

import pytest

from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.agentic.ladder_adapter import as_plain_result
from metaculus_bot.research.fetch_ladder import guard, ladder, run_cache, rungs
from metaculus_bot.research.fetch_ladder.context import LadderContext
from metaculus_bot.research.fetch_ladder.policy import GAP_FILL_DIRECT_POLICY, GAP_FILL_FETCH_POLICY
from metaculus_bot.research.rendered_fetch import RenderedPage
from tests.resolution_source_fakes import FakeResponse, FakeSession

_URL = "https://throttle.example.com/report"
_THROTTLE_HTML = (
    b"<!doctype html><html><head><title>Too many requests</title></head><body>"
    b"<h1>Rate limit exceeded</h1><p>Please try again later.</p></body></html>"
)
_THROTTLE_TEXT = b"Rate limit exceeded. Please try again later."
_JS_SHELL = b'<!doctype html><html><body><div id="root"></div><script src="/app.js"></script></body></html>'


@pytest.fixture(autouse=True)
def _clear_ladder_caches() -> None:
    run_cache.clear()


def _public_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _public(_url: str) -> bool:
        return True

    monkeypatch.setattr(guard, "is_public_http_url", _public)


@pytest.mark.asyncio
async def test_direct_html_interstitial_is_terminal_blank_and_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real 200 HTML body is classified before the verdict and never reaches another rung."""
    _public_urls(monkeypatch)
    session = FakeSession(
        {
            _URL: [
                FakeResponse(200, body=_THROTTLE_HTML, content_type="text/html"),
                FakeResponse(
                    200, body=b"<html><body><article>Actual report.</article></body></html>", content_type="text/html"
                ),
            ]
        }
    )

    async def _browser_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a direct throttle must be terminal before the browser rung")

    monkeypatch.setattr(rungs, "render_page", _browser_must_not_run)
    first = await ladder.fetch_url(
        _URL,
        policy=GAP_FILL_FETCH_POLICY,
        ctx=LadderContext(session=session, host_sems={}),
    )

    assert first.status == "throttled"
    assert first.text == ""
    assert first.route == "direct"
    assert first.rung_attempts == []
    second = await ladder.fetch_url(
        _URL,
        policy=GAP_FILL_DIRECT_POLICY,
        ctx=LadderContext(session=session, host_sems={}),
    )
    assert second.status == "success"
    assert session.requested == [_URL, _URL]


@pytest.mark.asyncio
async def test_raw_text_interstitial_maps_to_the_loop_throttle_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The raw-body path shares detection and exposes no fetched verification tier."""
    _public_urls(monkeypatch)
    session = FakeSession({_URL: FakeResponse(200, body=_THROTTLE_TEXT, content_type="text/plain")})

    result = await ladder.fetch_url(
        _URL,
        policy=GAP_FILL_DIRECT_POLICY,
        ctx=LadderContext(session=session, host_sems={}),
    )
    plain = as_plain_result(result, requested_url=_URL)

    assert result.status == "throttled"
    assert result.text == ""
    assert plain.status == "throttled"
    assert plain.method == "throttled"
    assert plain.text == ""


@pytest.mark.asyncio
async def test_pdf_bytes_with_throttle_words_are_not_throttled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Throttle detection is for HTML/raw text and must not inspect a PDF body."""
    _public_urls(monkeypatch)
    session = FakeSession(
        {
            _URL: FakeResponse(
                200,
                body=b"%PDF-1.4\nRate limit exceeded. Please try again later.",
                content_type="application/pdf",
            )
        }
    )

    result = await ladder.fetch_url(
        _URL,
        policy=GAP_FILL_DIRECT_POLICY,
        ctx=LadderContext(session=session, host_sems={}),
    )

    assert result.status != "throttled"


@pytest.mark.asyncio
async def test_http_429_behavior_remains_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new body detector only classifies 200 responses; HTTP 429 keeps its old status."""
    _public_urls(monkeypatch)
    session = FakeSession(
        {_URL: FakeResponse(429, body=_THROTTLE_TEXT, content_type="text/html", headers={"Server": "edge"})}
    )

    result = await ladder.fetch_url(
        _URL,
        policy=GAP_FILL_DIRECT_POLICY,
        ctx=LadderContext(session=session, host_sems={}),
    )

    assert result.status == "blocked"
    assert result.http_status == 429
    assert result.text == ""
    plain = as_plain_result(result, requested_url=_URL)
    assert plain.status == "blocked"
    assert plain.method == "plain"
    assert plain.text == "Fetch blocked with HTTP 429."


@pytest.mark.asyncio
async def test_rendered_interstitial_preserves_route_attempt_and_marker(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A browser-read interstitial keeps ``route=rendered`` and the fired attempt's verdict."""
    _public_urls(monkeypatch)
    render_calls: list[str] = []

    async def _render(url: str, **kwargs: object) -> RenderedPage:
        del kwargs
        render_calls.append(url)
        return RenderedPage(url=url, content_type="text/html", html=_THROTTLE_HTML.decode(), http_status=200)

    monkeypatch.setattr(rungs, "render_page", _render)
    ladder_session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})
    ladder_result = await ladder.fetch_url(
        _URL,
        policy=GAP_FILL_FETCH_POLICY,
        ctx=LadderContext(session=ladder_session, host_sems={}),
    )
    assert ladder_result.status == "throttled"
    assert ladder_result.text == ""
    assert ladder_result.route == "rendered"
    assert len(ladder_result.rung_attempts) == 1
    assert ladder_result.rung_attempts[0].rung == "rendered"
    assert ladder_result.rung_attempts[0].outcome == "throttled"

    session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})
    with caplog.at_level(logging.WARNING, logger=agentic_tools.__name__):
        outcome = await agentic_tools.fetch(
            _URL,
            ctx=LadderContext(session=session, host_sems={}),
        )

    assert render_calls == [_URL, _URL]
    assert outcome.status == "throttled"
    assert outcome.method == "throttled"
    assert "Rate limit exceeded" not in outcome.content_markdown
    assert "AGENTIC_FETCH_THROTTLED: url=https://throttle.example.com/report method=rendered" in caplog.text
    assert "phrase=rate limit" in caplog.text
