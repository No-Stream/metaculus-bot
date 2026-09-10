"""Cross-caller follow-up behavior after a successful impersonated re-dial."""

from __future__ import annotations

import pytest

from metaculus_bot.research import impersonated_fetch
from metaculus_bot.research.fetch_ladder import guard, ladder, rungs
from metaculus_bot.research.fetch_ladder.context import LadderContext
from metaculus_bot.research.fetch_ladder.policy import GAP_FILL_FETCH_POLICY
from metaculus_bot.research.impersonated_fetch import ImpersonatedResponse
from metaculus_bot.research.rendered_fetch import RenderedPage
from tests.resolution_source_fakes import FakeResponse, FakeSession

_URL = "https://reports.example.com/dashboard"
_JS_SHELL = b'<!doctype html><html><body><div id="root"></div><script src="/app.js"></script></body></html>'
_THIN_PAGE = b"<html><body><article>Latest update.</article></body></html>"
_RENDERED_HTML = (
    "<html><body><article>"
    + ("The complete quarterly report includes verified figures. " * 20)
    + "</article></body></html>"
)


@pytest.fixture(autouse=True)
def _reset_impersonation_memo() -> None:
    impersonated_fetch.reset_impersonation_memo()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("impersonated_body", "impersonated_outcome"),
    [
        (_THIN_PAGE, "success"),
        (_JS_SHELL, "js_wall"),
    ],
)
async def test_gap_fill_renders_an_impersonated_body_that_still_needs_a_browser(
    monkeypatch: pytest.MonkeyPatch,
    impersonated_body: bytes,
    impersonated_outcome: str,
) -> None:
    async def _public(_url: str) -> bool:
        return True

    async def _impersonated(*_args: object, **_kwargs: object) -> ImpersonatedResponse:
        return ImpersonatedResponse(
            status=200,
            url=_URL,
            content_type="text/html",
            server="edge",
            body=impersonated_body,
            elapsed_s=0.1,
            primary_ip="203.0.113.10",
        )

    async def _render(url: str, **_kwargs: object) -> RenderedPage:
        return RenderedPage(url=url, content_type="text/html", html=_RENDERED_HTML, http_status=200)

    monkeypatch.setattr(guard, "is_public_http_url", _public)
    monkeypatch.setattr(rungs, "fetch_impersonated", _impersonated)
    monkeypatch.setattr(rungs, "render_page", _render)
    session = FakeSession({_URL: FakeResponse(403, body=b"denied", content_type="text/html")})

    result = await ladder.fetch_url(
        _URL,
        policy=GAP_FILL_FETCH_POLICY,
        ctx=LadderContext(session=session, host_sems={}),
    )

    assert result.status == "success"
    assert result.route == "rendered"
    assert [(attempt.rung, attempt.outcome) for attempt in result.rung_attempts] == [
        ("impersonate", impersonated_outcome),
        ("rendered", "success"),
    ]
