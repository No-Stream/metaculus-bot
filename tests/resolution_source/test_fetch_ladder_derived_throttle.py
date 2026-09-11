"""Throttle verdicts for JSON obtained through either derived-API path."""

from __future__ import annotations

from metaculus_bot.research import derived_api
from metaculus_bot.research.agentic.ladder_adapter import as_plain_result
from metaculus_bot.research.fetch_ladder import rungs
from metaculus_bot.research.fetch_ladder.ladder import _fetch_one
from metaculus_bot.research.rendered_fetch import HarvestedJson, RenderedPage
from tests.resolution_source_fakes import _FEED_URL, _JS_SHELL, _URL, FakeResponse, FakeSession, _fake_render

_THROTTLE_BODY = b"Rate limit exceeded. Please try again later."


async def test_harvested_derived_feed_applies_the_shared_throttle_verdict(monkeypatch) -> None:
    rendered = RenderedPage(
        url=_URL,
        content_type="text/html",
        html=_JS_SHELL.decode(),
        json_responses=(HarvestedJson(url=_FEED_URL, body=_THROTTLE_BODY),),
    )
    monkeypatch.setattr(rungs, "render_page", _fake_render(rendered, []))
    session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

    result = await _fetch_one(session, _URL, {})

    assert result.status == "throttled"
    assert result.route == "derived_api"
    assert result.text == ""
    assert result.throttle_phrase == "rate limit"
    plain = as_plain_result(result, requested_url=_URL)
    assert plain.status == "throttled"
    assert plain.method == "throttled"
    assert plain.throttle_method == "derived_api"


async def test_remembered_derived_feed_propagates_the_terminal_throttle(monkeypatch) -> None:
    derived_api.remember_endpoint(_URL, _FEED_URL)
    session = FakeSession(
        {
            _URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
            _FEED_URL: FakeResponse(200, body=_THROTTLE_BODY, content_type="application/json"),
        }
    )

    async def _browser_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a remembered feed throttle must stop escalation")

    monkeypatch.setattr(rungs, "render_page", _browser_must_not_run)

    result = await _fetch_one(session, _URL, {})

    assert result.status == "throttled"
    assert result.route == "derived_api"
    assert result.url == _URL
    assert result.text == ""
    assert result.throttle_phrase == "rate limit"
    assert _FEED_URL in session.requested
