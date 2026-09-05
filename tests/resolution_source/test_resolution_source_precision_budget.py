"""The wall-budget floor under the extractor policy's second pass.

``_extract_page_text`` re-extracts a chrome-shaped default under ``favor_precision``. That pass
is CPU over a body already in hand, it runs after the browser rung has spent the whole remaining
budget on the rendered rung, and it costs about what the default pass did, so under
``RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S`` of remaining wall it is skipped and the default
text is withheld exactly as a failed precision pass would have been. The floor can only ever
withhold a page, never publish one.

The extractor is faked at the seam the module resolves it on, as the policy tests do: a
chrome-shaped default (short lines, over the floor) and a content-shaped precision text, so the
only thing deciding the outcome is whether the second pass ran.
"""

from __future__ import annotations

import time

from metaculus_bot.research import resolution_source
from metaculus_bot.research.rendered_fetch import RenderedPage
from metaculus_bot.research.resolution_source import (
    RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S,
    RESOLUTION_SOURCE_WALL_TIMEOUT,
    FetchContext,
    FetchResult,
    _extract_page_text,
    _fetch_one,
    _rendered_rung_applies,
    content_share,
    looks_like_page_chrome,
)
from tests.resolution_source_fakes import FakeResponse, FakeSession, _fake_render

_URL = "https://portal.example.com/statistics"
_BODY = b"<!doctype html><html><body><nav>Home</nav><main><p>irrelevant to the fake extractor</p></main></body></html>"
_HTML = _BODY.decode()

# Over the 400-char floor on 12-char lines alone: chrome to the line-shape metric.
_MENU = "\n".join(f"Menu item {i:03d}" for i in range(60))
_CARD = (
    "H.R.2913 - Ukraine Support Act\n"
    "Latest Action: Senate - 07/20/2026 Read the second time and placed on Senate Legislative Calendar "
    "under General Orders. Calendar No. 412.\n"
    "Tracker: This bill has the status Passed House. Here are the steps for Status of Legislation: "
    "Introduced, Passed House, Passed Senate, To President, Became Law.\n"
    "Sponsor: Rep. Example, Someone [D-XX-1] (Introduced 04/16/2026). Committees: House - Foreign Affairs; "
    "Financial Services; Judiciary; Ways and Means; Oversight and Government Reform."
)


def _spent_context(wall_left_s: float) -> FetchContext:
    """A per-URL context whose remaining wall (``rung_budget_s``) is about ``wall_left_s``."""
    margin = resolution_source.RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S
    return FetchContext(started=time.monotonic() - (RESOLUTION_SOURCE_WALL_TIMEOUT - margin - wall_left_s))


def _install_fake_extractor(monkeypatch, *, precision: str | None, default_cost_s: float = 0.0) -> list[bool]:
    """Fake the extractor seam; returns the ``favor_precision`` flag of every call, in order."""
    calls: list[bool] = []

    def _extract(body: bytes | str, url: str, *, favor_precision: bool = False) -> str | None:
        del body, url
        calls.append(favor_precision)
        if favor_precision:
            return precision
        if default_cost_s:
            time.sleep(default_cost_s)
        return _MENU

    monkeypatch.setattr(resolution_source, "_extract_main_text", _extract)
    return calls


class TestPrecisionRetryBudget:
    def test_the_fixtures_are_what_the_policy_decides_on(self):
        assert not looks_like_page_chrome(_MENU)
        assert content_share(_MENU) < resolution_source.RESOLUTION_SOURCE_CONTENT_SHARE_MIN
        assert content_share(_CARD) >= resolution_source.RESOLUTION_SOURCE_CONTENT_SHARE_MIN

    def test_unbounded_by_default_the_second_pass_runs_and_rescues(self, monkeypatch):
        """The two existing call shapes: no ``remaining_wall_s`` at all, and plenty of it."""
        calls = _install_fake_extractor(monkeypatch, precision=_CARD)

        unbounded = _extract_page_text(_HTML, _BODY, _URL, 0.0)
        roomy = _extract_page_text(
            _HTML, _BODY, _URL, 0.0, remaining_wall_s=RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S + 10.0
        )

        assert calls == [False, True, False, True]
        for extraction in (unbounded, roomy):
            assert extraction.text == _CARD
            assert extraction.precision_rescued is True
            assert extraction.chrome_metric_withheld is False

    def test_under_the_floor_the_second_pass_is_skipped_and_the_default_withheld(self, monkeypatch):
        """The skip takes the exit a FAILED precision pass takes: the default text carried with
        the withhold flag, so the classifier withholds it under `thin_page` and the rendered rung
        still fires. It must not publish the default text: that ships a navigation tree as
        primary grading evidence whenever the wall is short."""
        calls = _install_fake_extractor(monkeypatch, precision=_CARD)

        extraction = _extract_page_text(
            _HTML, _BODY, _URL, 0.0, remaining_wall_s=RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S - 0.1
        )

        assert calls == [False]
        assert extraction.text == _MENU
        assert extraction.chrome_metric_withheld is True
        assert extraction.precision_rescued is False

    def test_the_default_pass_s_own_cost_counts_against_the_budget(self, monkeypatch):
        """``remaining_wall_s`` is read when the body is handed over; the default pass on a 5 MiB
        DOM can itself take seconds, so the floor is checked against what is left AFTER it."""
        calls = _install_fake_extractor(monkeypatch, precision=_CARD, default_cost_s=0.05)

        extraction = _extract_page_text(
            _HTML, _BODY, _URL, 0.0, remaining_wall_s=RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S + 0.02
        )

        assert calls == [False]
        assert extraction.chrome_metric_withheld is True

    def test_a_default_that_passes_the_metric_never_consults_the_budget(self, monkeypatch):
        """The floor gates the SECOND pass only. A content-shaped default publishes with no wall
        left at all, so the change cannot withhold a page the policy already published."""

        def _content(body: bytes | str, url: str, *, favor_precision: bool = False) -> str | None:
            del body, url, favor_precision
            return _CARD

        monkeypatch.setattr(resolution_source, "_extract_main_text", _content)

        extraction = _extract_page_text(_HTML, _BODY, _URL, 0.0, remaining_wall_s=-5.0)

        assert extraction.text == _CARD
        assert extraction.chrome_metric_withheld is False


class TestPrecisionRetryBudgetThroughTheLadder:
    """Both production call sites hand the classifier what the wall has left."""

    async def test_the_direct_path_withholds_under_the_floor_and_rescues_above_it(self, monkeypatch):
        calls = _install_fake_extractor(monkeypatch, precision=_CARD)
        session = FakeSession({_URL: FakeResponse(200, body=_BODY)})

        rescued = await _fetch_one(session, _URL, {}, FetchContext())
        assert calls == [False, True]
        assert rescued.status == "success"
        assert rescued.precision_rescued is True

        calls.clear()
        withheld = await _fetch_one(
            session, _URL, {}, _spent_context(RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S - 1.0)
        )

        assert calls == [False]
        assert withheld.status == "no_resolving_content"
        assert withheld.status_reason == "thin_page"
        assert withheld.text == ""
        assert withheld.chrome_metric_withheld is True
        assert _rendered_rung_applies(withheld)

    @staticmethod
    async def _render_a_menu_tree(monkeypatch, ctx: FetchContext) -> tuple[list[bool], FetchResult]:
        """Drive the ladder with a direct body AND a rendered DOM that both extract to chrome.

        The browser floor is lowered so the rung is admitted on a nearly spent wall (the real
        12 s floor would decline it before the classification this test is about), and the
        precision pass extracts nothing, so both classifications withhold and the URL is memoised
        as rendered-to-nothing; one URL per test, because that memo outlives the call.
        """
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S", 1.0)
        calls = _install_fake_extractor(monkeypatch, precision=None)
        rendered_dom = "<html><body><nav>" + "".join(f"<a href='/{i}'>Menu item {i:03d}</a>" for i in range(60))
        rendered_dom += "</nav></body></html>"
        page = RenderedPage(url=_URL, content_type="text/html", html=rendered_dom)
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(page, []))
        session = FakeSession({_URL: FakeResponse(200, body=_BODY)})
        result = await _fetch_one(session, _URL, {}, ctx)
        return calls, result

    async def test_a_rendered_dom_gets_its_second_pass_with_wall_to_spare(self, monkeypatch):
        calls, result = await self._render_a_menu_tree(monkeypatch, FetchContext())

        # The direct classification (default, precision), then the rendered DOM's (default, precision).
        assert calls == [False, True, False, True]
        assert result.route == "rendered"
        assert result.status == "no_resolving_content"

    async def test_the_rendered_dom_is_classified_against_what_the_browser_left(self, monkeypatch):
        """The rendered rung gives the browser the whole remaining budget and classifies the DOM
        afterwards, which is the call site the floor exists for: a rendered navigation tree is
        exactly the shape that triggers the second pass."""
        calls, result = await self._render_a_menu_tree(
            monkeypatch, _spent_context(RESOLUTION_SOURCE_PRECISION_RETRY_MIN_BUDGET_S - 1.0)
        )

        # Neither classification spent a second pass on a wall that could not pay for one.
        assert calls == [False, False]
        assert result.route == "rendered"
        assert result.status == "no_resolving_content"
        assert result.chrome_metric_withheld is True
