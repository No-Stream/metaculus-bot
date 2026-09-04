"""The Tier-1 escalation ladder: the rungs that run when the direct fetch read nothing.

One module per ladder, not per rung, because what these tests mostly pin is ORDER and
TRIGGER POPULATION — which direct outcome earns which rung, which rung claims the ``route``,
and what a rung records when it declines. The direct route's own branches stay in
``test_resolution_source_fetch.py``.

Nothing here touches the network: the browser transport (``resolution_source.render_page``)
and the paid reader are faked at the seam ``resolution_source`` resolves them on, and the
package's autouse fixtures already stub DNS and reset the shared gates.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime, timedelta
from typing import get_args

import pytest

from metaculus_bot.research import rendered_fetch, resolution_source
from metaculus_bot.research.derived_api import reset_derived_endpoints
from metaculus_bot.research.provider_diagnostics import pop_provider_detail
from metaculus_bot.research.rendered_fetch import HarvestedJson, RenderedPage
from metaculus_bot.research.resolution_fetch_result import ROUTE_CAVEATS, FetchResult, FetchRoute
from metaculus_bot.research.resolution_source import (
    _WAYBACK_TRIGGER_STATUSES,
    FetchContext,
    QuestionRungBudget,
    _fetch_one,
    _rung_counts,
    format_resolution_sections,
    resolution_source_provider,
)
from metaculus_bot.research.robots_policy import reset_robots_cache
from metaculus_bot.research.wayback import wayback_snapshot_url
from tests.resolution_source_fakes import (
    _INFOGRAM_EMBED_MARKUP,
    FakeResponse,
    FakeSession,
    _embed_shell_page,
    _mock_question,
    _prose_page,
)
from tests.test_document_text import build_text_pdf

_URL = "https://tracker.example.com/senate"
_FEED_URL = "https://tracker.example.com/api/series"

# A body under RESOLUTION_SOURCE_JS_WALL_MIN_CHARS extracts to nothing: the direct route
# calls that `js_wall`, which is the rendered rung's primary trigger population.
_JS_SHELL = b'<!doctype html><html><body><div id="root"></div><script src="/app.js"></script></body></html>'

# Above RESOLUTION_SOURCE_JS_WALL_MIN_CHARS (100) and below the chrome floor (400): the band
# where a 200 carries real prose that is nothing but page furniture, which is what
# `no_resolving_content` / `thin_page` means and what the rendered rung's second trigger is.
_TAB_LIST_CHROME = (
    "Nationwide. Midwest. Northeast. South. West. Select a region above to load its series. "
    "Data updates weekly. About the data. Methodology. Contact us. Terms of use."
)

# Deliberately well ABOVE RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS (400) once extracted: the
# rendered DOM has to clear the same chrome floor a directly-fetched page does, so a fixture
# sized just under it would test the floor rather than the rung.
_RENDERED_PROSE = (
    "The Nebraska Senate polling average stands at 47.2 percent for Osborn and 45.8 percent "
    "for Ricketts as of September 2, 2026, across the eleven qualifying polls released since "
    "the primary. The average is recomputed whenever a new qualifying poll is published, and "
    "the trend over the last three weeks has been a narrowing of the gap by roughly a point. "
    "Polls are weighted by sample size, recency and the pollster's historic accuracy, and "
    "partisan-sponsored surveys are included at half weight with an adjustment for house "
    "effects. The eleven polls in the current average were fielded between August 4 and "
    "September 1 and carry sample sizes between 480 and 1,320 likely voters."
)


@pytest.fixture(autouse=True)
def _reset_render_state():
    """The render memo, the launch gate and the derived-endpoint map all outlive one provider
    call by design — that is what makes the second cited URL on a host cheap — so a test that
    inherited another's finds would pass or fail on ordering."""
    rendered_fetch.reset_render_state()
    reset_derived_endpoints()
    yield
    rendered_fetch.reset_render_state()
    reset_derived_endpoints()


def _rendered(html: str, *, content_type: str = "text/html") -> RenderedPage:
    return RenderedPage(url=_URL, content_type=content_type, html=html)


def _rendered_document(inner: str) -> RenderedPage:
    """A rendered DOM shaped like a real document.

    Trafilatura discards a bare ``<article>`` with no head element ("discarding data"), so a
    fixture that skipped the head would test the extractor's give-up path rather than the
    rung. What a browser hands us is always a whole document, so the fixture is one too.
    """
    return _rendered(
        "<!doctype html><html><head><title>Nebraska Senate polling average</title></head><body>"
        f"<nav>Home | Senate</nav><article>{inner}</article><footer>&copy; 2026</footer></body></html>"
    )


def _fake_render(page: RenderedPage | None, calls: list[dict[str, object]]):
    async def _render(
        url: str,
        *,
        memo_scope: str,
        host_gate,
        goto_timeout_ms: int,
        deadline_monotonic_s: float | None = None,
        harvest_json: bool = False,
    ):
        calls.append({"url": url, "goto_timeout_ms": goto_timeout_ms, "harvest_json": harvest_json})
        del memo_scope, host_gate, deadline_monotonic_s
        # A real yield point, so the fake schedules like the browser rung it replaces and a
        # test that races two escalations sees the same interleaving the transport would give.
        await asyncio.sleep(0)
        return page

    return _render


class TestRenderedRungTriggers:
    """Which direct outcome earns a browser launch, and which deliberately does not."""

    async def test_a_js_wall_is_rescued_and_claims_the_route(self, monkeypatch):
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            resolution_source,
            "render_page",
            _fake_render(_rendered_document(f"<h1>Polling average</h1><p>{_RENDERED_PROSE}</p>"), calls),
        )
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert result.route == "rendered"
        assert "Nebraska Senate polling average" in result.text
        # The record keeps the direct fetch's HTTP status: the page answered 200 and carried
        # no text, which is the fact worth archiving. Chromium reports no status on a salvaged
        # DOM, so borrowing its status would sometimes be None.
        assert result.http_status == 200
        assert len(calls) == 1

    async def test_a_thin_page_is_escalated_too(self, monkeypatch):
        """The `thin_page` shape of `no_resolving_content` is the same client-side-assembly
        failure one floor up: above the JS-wall floor, still nothing but chrome."""
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            resolution_source,
            "render_page",
            _fake_render(_rendered_document(f"<h1>Polling average</h1><p>{_RENDERED_PROSE}</p>"), calls),
        )
        thin = _prose_page(_TAB_LIST_CHROME)
        session = FakeSession({_URL: FakeResponse(200, body=thin, content_type="text/html")})

        direct_only = await resolution_source._classify_html_body(thin, _URL, "text/html", http_status=200)
        assert direct_only.result.status_reason == "thin_page"

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert result.route == "rendered"
        assert len(calls) == 1

    async def test_an_embed_shell_is_not_escalated(self, monkeypatch):
        """`page.content()` is the MAIN FRAME's HTML, so an Infogram iframe's document is
        somewhere a render never reads. Launching would spend 100-300 MB to re-derive the
        same verdict."""
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(_rendered("<html></html>"), calls))
        session = FakeSession(
            {_URL: FakeResponse(200, body=_embed_shell_page(_INFOGRAM_EMBED_MARKUP), content_type="text/html")}
        )

        result = await _fetch_one(session, _URL, {})

        assert result.status == "no_resolving_content"
        assert result.status_reason == "embed_shell"
        assert result.route == "direct"
        assert calls == []

    @pytest.mark.parametrize("status", [403, 404, 500])
    async def test_a_non_200_is_not_escalated_to_the_browser(self, monkeypatch, status):
        """Chromium dials from the same address the edge just refused, and a 404 has no page."""
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(_rendered("<html></html>"), calls))
        session = FakeSession({_URL: FakeResponse(status, body=b"nope", content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status != "success"
        assert result.route == "direct"
        assert calls == []

    async def test_a_successful_direct_fetch_never_escalates(self, monkeypatch, article_html):
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(_rendered("<html></html>"), calls))
        session = FakeSession({_URL: FakeResponse(200, body=article_html, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert result.route == "direct"
        assert calls == []


class TestRenderedRungBudget:
    """The rung is self-bounded against the provider wall, because the outer wait_for
    discards every page the question already fetched when it fires."""

    async def test_it_is_skipped_below_the_floor(self, monkeypatch):
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(_rendered("<html></html>"), calls))
        monkeypatch.setattr(FetchContext, "rung_budget_s", lambda self: 4.0)
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "js_wall"
        assert result.route == "direct"
        assert calls == []
        skips = [a for a in result.rung_attempts if a.rung == "rendered"]
        assert [a.skipped_reason for a in skips] == ["wall_budget"]
        counts = _rung_counts([result])
        assert counts["rung_budget_skips"] == 1
        # The aggregate cannot say WHICH rung the wall bound; the per-rung key can.
        assert counts["rendered_budget_skips"] == 1
        assert counts["url_context_budget_skips"] == 0
        assert counts["rendered_attempts"] == 0

    async def test_the_navigation_budget_comes_off_the_remaining_wall(self, monkeypatch):
        """A render admitted with 20 s left may not then help itself to the full 35 s cap."""
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(None, calls))
        monkeypatch.setattr(FetchContext, "rung_budget_s", lambda self: 20.0)
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        await _fetch_one(session, _URL, {})

        assert calls[0]["goto_timeout_ms"] == 20_000 - rendered_fetch.RENDER_SETTLE_MS

    async def test_a_generous_budget_is_still_capped_at_the_transport_ceiling(self, monkeypatch):
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(None, calls))
        monkeypatch.setattr(FetchContext, "rung_budget_s", lambda self: 600.0)
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        await _fetch_one(session, _URL, {})

        assert calls[0]["goto_timeout_ms"] == rendered_fetch.RENDER_TIMEOUT_MS - rendered_fetch.RENDER_SETTLE_MS


class TestRenderedRungDeclines:
    """A transport that hands back nothing, and a render that reads nothing, are different
    facts and are recorded differently."""

    async def test_an_unavailable_renderer_is_a_skip_not_a_fired_rung(self, monkeypatch):
        """Chromium's install step is `continue-on-error` in every workflow, so its absence is
        by design — and nothing was rendered, so the attempt must claim no route."""
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(None, []))
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "js_wall"
        assert result.route == "direct"
        attempts = [a for a in result.rung_attempts if a.rung == "rendered"]
        assert [a.skipped_reason for a in attempts] == ["renderer_unavailable"]
        counts = _rung_counts([result])
        assert counts["renderer_unavailable_skips"] == 1
        assert counts["rendered_attempts"] == 0
        assert counts["rung_budget_skips"] == 0

    async def test_a_render_that_reads_nothing_is_memoized_and_keeps_the_route(self, monkeypatch):
        """The rung FIRED, so `route=rendered status=js_wall` is the archive's way of saying
        we tried the browser and this is still the answer — the meta-refresh convention."""
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            resolution_source,
            "render_page",
            _fake_render(_rendered(_JS_SHELL.decode()), calls),
        )
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "js_wall"
        assert result.route == "rendered"
        assert rendered_fetch.rendered_to_nothing(_URL, memo_scope="resolution_source") is True
        assert _rung_counts([result])["rendered_attempts"] == 1
        assert len(calls) == 1

    async def test_a_rescued_render_is_not_memoized(self, monkeypatch):
        monkeypatch.setattr(
            resolution_source,
            "render_page",
            _fake_render(_rendered_document(f"<h1>Polling average</h1><p>{_RENDERED_PROSE}</p>"), []),
        )
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        await _fetch_one(session, _URL, {})

        assert rendered_fetch.rendered_to_nothing(_URL, memo_scope="resolution_source") is False


def _hanging_render(calls: list[str]):
    """A transport that never comes back: the ogimet shape (P3-1), seen from the caller's side."""

    async def _render(url: str, **kwargs: object) -> None:
        calls.append(url)
        del kwargs
        await asyncio.Event().wait()

    return _render


def _admit_a_render_with(monkeypatch: pytest.MonkeyPatch, budget_s: float) -> None:
    """Let the rung fire on a sub-second budget. The production floor is 12 s, and a test that
    waited it out would cost more than the suite's whole rendered-rung coverage."""
    monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_RENDER_MIN_BUDGET_S", 0.01)
    monkeypatch.setattr(FetchContext, "rung_budget_s", lambda self: budget_s)


def _assert_recorded_as_a_render_timeout(result: FetchResult) -> None:
    assert result.status == "js_wall"
    assert result.route == "direct"
    attempts = [a for a in result.rung_attempts if a.rung == "rendered"]
    assert [a.skipped_reason for a in attempts] == ["render_timeout"]
    # The timed-out memo is the TRANSPORT's, written only when a browser actually ran (pinned in
    # tests/test_rendered_fetch.py); the rung never writes the rendered-to-nothing memo on a cut.
    assert rendered_fetch.rendered_to_nothing(result.url, memo_scope="resolution_source") is False
    assert rendered_fetch._PLAYWRIGHT_WARNED is False
    counts = _rung_counts([result])
    assert counts["render_timeout_skips"] == 1
    assert counts["rendered_attempts"] == 0
    assert counts["renderer_unavailable_skips"] == 0


class TestRenderedRungTimeout:
    """P3-1 (live QA, 2026-09-03): a page that kept navigating held ``page.content()`` for 40 s
    after its goto timed out, the render ran 76 s, and the provider's 45 s wall discarded every
    page the question had fetched. The rung now bounds the whole transport call at the remaining
    budget, on top of the transport's own DOM-read cap, and a render cut off by either is its own
    skip reason: it says nothing about whether Chromium works, so it must neither read as
    ``renderer_unavailable`` nor latch that warning, and the URL is memoised so a second question
    citing the same page does not pay for it again. The direct result is what stands.
    """

    async def test_a_render_that_outlives_the_budget_is_cut_off_at_the_budget(self, monkeypatch):
        calls: list[str] = []
        monkeypatch.setattr(resolution_source, "render_page", _hanging_render(calls))
        _admit_a_render_with(monkeypatch, 0.2)
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        started = time.monotonic()
        result = await _fetch_one(session, _URL, {})
        elapsed = time.monotonic() - started

        assert 0.2 <= elapsed < 2.0
        assert calls == [_URL]
        _assert_recorded_as_a_render_timeout(result)
        (attempt,) = [a for a in result.rung_attempts if a.rung == "rendered"]
        assert attempt.wall_s is not None
        assert attempt.wall_s >= 0.2

    async def test_the_transports_own_dom_read_timeout_is_recorded_the_same_way(self, monkeypatch):
        """The inner bound RAISES rather than declining with ``None``, which is what lets the rung
        tell it from a missing browser; both bounds land on the one reason."""

        async def _timed_out(url: str, **kwargs: object) -> None:
            del url, kwargs
            await asyncio.sleep(0)
            raise TimeoutError("rendered fetch DOM read exceeded 5000ms")

        monkeypatch.setattr(resolution_source, "render_page", _timed_out)
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        _assert_recorded_as_a_render_timeout(result)


class TestASlowRenderLeavesTheSiblingPagesStanding:
    """The invariant every per-rung bound exists for: a rung that overruns costs its own page,
    never the question's. Before the bound, this shape — one hostile page beside one ordinary
    one — returned nothing at all, because the provider's outer wall fired first."""

    _NEWS_URL = "https://news.example.com/cpi-report"

    async def test_the_provider_returns_the_other_page_inside_its_wall(self, monkeypatch, article_html, caplog):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setattr(resolution_source, "render_page", _hanging_render([]))
        _admit_a_render_with(monkeypatch, 0.2)
        session = FakeSession(
            {
                _URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                self._NEWS_URL: FakeResponse(200, body=article_html, content_type="text/html"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        question = _mock_question(resolution_criteria=f"Resolves per {_URL} and {self._NEWS_URL}")

        started = time.monotonic()
        with caplog.at_level("WARNING", logger="metaculus_bot.research.rendered_fetch"):
            section = await resolution_source_provider(is_benchmarking=False)(question)
        elapsed = time.monotonic() - started

        assert elapsed < 2.0
        assert "Consumer Price Index" in section
        assert "tracker.example.com: js_wall" in section
        counts = pop_provider_detail(question.id_of_question, "resolution_source")["counts"]
        assert counts["render_timeout_skips"] == 1
        assert counts["rendered_attempts"] == 0
        assert not [message for message in caplog.messages if "rung unavailable" in message]


class TestRenderedRungClassification:
    """A rescued page goes through the SAME classification path as a directly-fetched one,
    which is what makes it indistinguishable downstream."""

    async def test_the_rendered_dom_feeds_the_datawrapper_hop(self, monkeypatch):
        """An embed only the rendered DOM carries still reaches Tier 2."""
        page = _rendered_document(
            f"<h1>Polling average</h1><p>{_RENDERED_PROSE}</p>"
            '<iframe title="Approval tracker" src="https://datawrapper.dwcdn.net/aB3dE/7/"></iframe>'
        )
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(page, []))
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert [chart.chart_id for chart in result.datawrapper_charts] == ["aB3dE"]

    async def test_the_per_url_cap_still_binds_on_a_rendered_page(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_PER_URL_MAX_CHARS", 200)
        monkeypatch.setattr(
            resolution_source,
            "render_page",
            _fake_render(_rendered_document(f"<h1>Polling average</h1><p>{_RENDERED_PROSE * 20}</p>"), []),
        )
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert len(result.text) <= 200


class TestDerivedApiRung:
    """A JavaScript dashboard's numbers arrive over XHR, so the JSON the page fetched for
    itself is the last free route once the rendered DOM turns out to be empty too."""

    # A sibling path, not a child of the page URL: a real dashboard's feed usually is one, and
    # the fake session dispatches by URL PREFIX, so a child path would be served the page.
    _FEED = '{"series":[{"date":"2026-09-01","osborn":47.2,"ricketts":45.8}]}'

    def _harvested(self, endpoint: str = _FEED_URL, body: str | None = None) -> RenderedPage:
        payload = (self._FEED if body is None else body).encode()
        return RenderedPage(
            url=_URL,
            content_type="text/html",
            html=_JS_SHELL.decode(),
            json_responses=(HarvestedJson(url=endpoint, body=payload),),
        )

    async def test_the_harvested_feed_is_served_when_the_dom_is_still_empty(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(self._harvested(), []))
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        # `derived_api`, not `rendered`: the render only found the endpoint, the feed produced
        # the text, and `route` is the rung that produced the outcome.
        assert result.route == "derived_api"
        assert _FEED_URL in result.text
        assert '"osborn":47.2' in result.text
        counts = _rung_counts([result])
        assert counts["rendered_attempts"] == 1
        assert counts["derived_api_reads"] == 1

    async def test_a_second_url_on_the_host_gets_the_feed_without_a_browser(self, monkeypatch):
        """The whole point of remembering the endpoint: a host with several cited URLs pays for
        one Chromium launch, not one per URL."""
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(self._harvested(), calls))
        second_url = "https://tracker.example.com/house"
        session = FakeSession(
            {
                _URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                second_url: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                _FEED_URL: FakeResponse(200, body=self._FEED.encode(), content_type="application/json"),
            }
        )

        first = await _fetch_one(session, _URL, {})
        second = await _fetch_one(session, second_url, {})

        assert first.route == "derived_api"
        assert second.status == "success"
        assert second.route == "derived_api"
        # One render across both URLs, and the second URL's feed came off an ordinary GET.
        assert len(calls) == 1
        assert _FEED_URL in session.requested
        # Reused from another page, so the lead says so rather than presenting it as this
        # page's own data: a host's feed is usually parameterised.
        assert "DIFFERENT page" in second.text
        assert _URL in second.text

    async def test_two_same_host_urls_fetched_concurrently_share_one_launch(self, monkeypatch):
        """The shape prod actually takes: `fetch_resolution_sources` fans one task out per cited
        URL, so both same-host URLs reach `endpoint_for` before either render has finished. The
        second must wait for the first's escalation and then take the feed off an ordinary GET,
        or the docstring's "one Chromium launch per host" is true only for sequential fetches."""
        calls: list[dict[str, object]] = []
        harvested = self._harvested()

        async def _slow_render(
            url: str,
            *,
            memo_scope: str,
            host_gate,
            goto_timeout_ms: int,
            deadline_monotonic_s: float | None = None,
            harvest_json: bool = False,
        ):
            calls.append({"url": url, "goto_timeout_ms": goto_timeout_ms, "harvest_json": harvest_json})
            del host_gate, memo_scope, deadline_monotonic_s
            # Long enough that the sibling task reaches its own escalation while this render is
            # still in flight, which is the interleaving the fan-out produces.
            await asyncio.sleep(0.02)
            return harvested

        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setattr(resolution_source, "render_page", _slow_render)
        second_url = "https://tracker.example.com/house"
        session = FakeSession(
            {
                _URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                second_url: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                _FEED_URL: FakeResponse(200, body=self._FEED.encode(), content_type="application/json"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        question = _mock_question(resolution_criteria=f"Resolves per {_URL} and {second_url}")

        section = await resolution_source_provider(is_benchmarking=False)(question)

        assert len(calls) == 1, "the second same-host URL launched its own browser"
        assert _FEED_URL in session.requested
        counts = pop_provider_detail(question.id_of_question, "resolution_source")["counts"]
        assert counts["rendered_attempts"] == 1
        assert counts["derived_api_reads"] == 2
        assert ROUTE_CAVEATS["derived_api"] in section

    async def test_a_feed_that_fails_hands_the_url_on_to_the_browser(self, monkeypatch):
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(self._harvested(), calls))
        second_url = "https://tracker.example.com/house"
        session = FakeSession(
            {
                _URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                second_url: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                # The first URL is served its feed straight off the render, so this handler is
                # only ever reached by the SECOND URL's derived-feed GET.
                _FEED_URL: FakeResponse(503, body=b"", content_type="text/html"),
            }
        )

        await _fetch_one(session, _URL, {})
        second = await _fetch_one(session, second_url, {})

        # The dead feed did not end the ladder: the browser ran for the second URL too.
        assert len(calls) == 2
        assert second.route == "derived_api"

    async def test_a_remembered_endpoint_answering_html_is_not_served_as_the_feed(self, monkeypatch):
        """The harvest half gates on a JSON content type; the reuse half — the path that fires on
        every later cited URL on the host — did not, so a remembered endpoint answering 200 with
        a "session expired" portal page was published as the page's data feed, under a lead
        saying it was the JSON the page loads its figures from."""
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(self._harvested(), calls))
        second_url = "https://tracker.example.com/house"
        portal = _prose_page("Your session has expired. " * 30)
        session = FakeSession(
            {
                _URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                second_url: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                _FEED_URL: FakeResponse(200, body=portal, content_type="text/html; charset=utf-8"),
            }
        )

        await _fetch_one(session, _URL, {})
        second = await _fetch_one(session, second_url, {})

        assert _FEED_URL in session.requested
        # Declined, so the URL fell through to the browser, whose own harvest served it.
        assert len(calls) == 2
        assert second.route == "derived_api"
        assert "session has expired" not in second.text
        assert second.text.startswith("[This page's own HTML carried no readable content.")

    async def test_the_derived_get_is_skipped_below_its_floor(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(self._harvested(), []))
        second_url = "https://tracker.example.com/house"
        session = FakeSession(
            {
                _URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                second_url: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
            }
        )

        await _fetch_one(session, _URL, {})
        monkeypatch.setattr(FetchContext, "rung_budget_s", lambda self: 1.0)
        second = await _fetch_one(session, second_url, {})

        assert second.status == "js_wall"
        assert second.route == "direct"
        assert [(a.rung, a.skipped_reason) for a in second.rung_attempts] == [
            ("derived_api", "wall_budget"),
            ("rendered", "wall_budget"),
        ]
        assert _FEED_URL not in session.requested
        counts = _rung_counts([second])
        assert counts["rung_budget_skips"] == 2
        assert counts["derived_api_budget_skips"] == 1
        assert counts["rendered_budget_skips"] == 1

    async def test_an_undecodable_feed_is_never_served_as_content(self, monkeypatch):
        """A body we could not decode must not become the page's content on a section
        captioned primary grading evidence."""
        mojibake = RenderedPage(
            url=_URL,
            content_type="text/html",
            html=_JS_SHELL.decode(),
            json_responses=(HarvestedJson(url=_FEED_URL, body=b"a\x00" * 300),),
        )
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(mojibake, []))
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "js_wall"
        assert result.route == "rendered"
        assert _rung_counts([result])["derived_api_reads"] == 0
        # Nothing usable came out of the render, so the URL is memoized against a second launch.
        assert rendered_fetch.rendered_to_nothing(_URL, memo_scope="resolution_source") is True

    async def test_the_per_url_cap_binds_on_a_served_feed(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_PER_URL_MAX_CHARS", 400)
        big = '{"series":[' + ",".join(f'{{"v":{index}}}' for index in range(500)) + "]}"
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(self._harvested(body=big), []))
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert len(result.text) <= 400
        # The lead LEADS, so a later trim reaches the JSON before the provenance line.
        assert result.text.startswith("[This page's own HTML carried no readable content.")


def _snapshot_url(page_url: str, *, captured: datetime) -> str:
    """The final URL the archive redirects a snapshot request to (live-verified shape)."""
    return f"https://web.archive.org/web/{captured.strftime('%Y%m%d%H%M%S')}id_/{page_url}"


class TestWaybackRung:
    """The archive is the one free route whose EGRESS IS NOT OURS, which is why a host that
    refuses our address earns it — and why a JavaScript wall never does."""

    _NOW = datetime(2026, 9, 4, tzinfo=UTC)

    @pytest.fixture(autouse=True)
    def _arm_the_archive(self, monkeypatch):
        """Restore the rung's own trigger set, which this package's conftest empties by default.

        The module's constant OBJECT is restored rather than a copy, so the trigger population
        these tests assert on cannot drift from the one prod uses.
        """
        monkeypatch.setattr(resolution_source, "_WAYBACK_TRIGGER_STATUSES", _WAYBACK_TRIGGER_STATUSES)

    def _ctx(self) -> FetchContext:
        return FetchContext(now=self._NOW)

    def _archive_page(self) -> bytes:
        return _prose_page(_RENDERED_PROSE)

    def _session(
        self,
        *,
        page: FakeResponse,
        captured: datetime,
        archived: FakeResponse | None = None,
        extra: dict[str, FakeResponse] | None = None,
    ) -> FakeSession:
        """A session where the cited URL fails and the archive redirects to a dated capture."""
        snapshot = _snapshot_url(_URL, captured=captured)
        return FakeSession(
            {
                _URL: page,
                wayback_snapshot_url(_URL, now=self._NOW): FakeResponse(302, headers={"Location": snapshot}),
                snapshot: archived or FakeResponse(200, body=self._archive_page(), content_type="text/html"),
                **(extra or {}),
            }
        )

    async def test_a_blocked_page_is_served_from_a_fresh_capture(self):
        session = self._session(
            page=FakeResponse(403, body=b"denied", content_type="text/html"),
            captured=self._NOW - timedelta(days=6),
        )

        result = await _fetch_one(session, _URL, {}, self._ctx())

        assert result.status == "success"
        assert result.route == "wayback"
        # The cited URL, not the archive's: the section heading and every earlier fetch record
        # for this source are filed under what the question cited.
        assert result.url == _URL
        assert result.text.startswith(
            "[Archived copy from the Wayback Machine, captured 2026-08-29, 6 days before this "
            "forecast; the live page could not be fetched (blocked).]"
        )
        assert "Nebraska Senate polling average" in result.text
        assert _rung_counts([result])["wayback_attempts"] == 1

    @pytest.mark.parametrize("status", [403, 404, 500])
    async def test_every_trigger_status_reaches_the_archive(self, status):
        session = self._session(
            page=FakeResponse(status, body=b"", content_type="text/html"), captured=self._NOW - timedelta(days=1)
        )

        result = await _fetch_one(session, _URL, {}, self._ctx())

        assert result.route == "wayback"
        assert result.status == "success"

    async def test_a_js_wall_never_reaches_the_archive(self, monkeypatch):
        """The archive stores the unrendered shell: it rescued 0 of 8 archived walls while the
        browser rung rescued 6."""
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(None, []))
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {}, self._ctx())

        assert result.status == "js_wall"
        assert not any(request.startswith("https://web.archive.org/") for request in session.requested)

    async def test_a_capture_past_the_age_bound_is_withheld_as_stale(self):
        session = self._session(
            page=FakeResponse(403, body=b"", content_type="text/html"),
            captured=self._NOW - timedelta(days=400),
        )

        result = await _fetch_one(session, _URL, {}, self._ctx())

        assert result.status == "stale_data"
        assert result.text == ""
        # The direct route's own status is not lost by the swap: the escalation line carries it.
        assert [(a.rung, a.from_status) for a in result.rung_attempts] == [("wayback", "blocked")]

    async def test_a_withheld_capture_still_hands_the_url_to_the_paid_reader(self, monkeypatch):
        """A stale archive is still a page we could not read fresh, which is exactly the
        population the paid rung was built for. Only a RESCUE ends the ladder early; the
        withhold stays the fallback when the reader is off or declines, so the `stale_data`
        marker pair above stands unchanged."""
        calls: list[dict[str, object]] = []

        def _read(url, ask, **kwargs):
            calls.append({"url": url, "ask": ask, **kwargs})
            return ("The page reports 12 major work stoppages.", 1, ["URL_RETRIEVAL_STATUS_SUCCESS"])

        monkeypatch.setenv("RESOLUTION_SOURCE_URL_CONTEXT_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(resolution_source, "run_url_context_read", _read)
        reset_robots_cache()
        session = self._session(
            page=FakeResponse(403, body=b"", content_type="text/html"),
            captured=self._NOW - timedelta(days=400),
            extra={
                "https://tracker.example.com/robots.txt": FakeResponse(
                    200, body=b"User-agent: *\nAllow: /\n", content_type="text/plain"
                )
            },
        )

        result = await _fetch_one(session, _URL, {}, FetchContext(now=self._NOW, query="ask"))

        assert [call["url"] for call in calls] == [_URL]
        assert result.status == "success"
        assert result.route == "url_context"
        # The archive attempt is still on the record: it fired, and the reader came after it.
        assert [a.rung for a in result.rung_attempts] == ["wayback", "url_context"]

    async def test_an_undatable_snapshot_is_withheld_too(self):
        """The archive answering our four-digit request means it never landed on a capture, and a
        copy with no usable date cannot carry the age disclosure."""
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"", content_type="text/html"),
                wayback_snapshot_url(_URL, now=self._NOW): FakeResponse(
                    200, body=self._archive_page(), content_type="text/html"
                ),
            }
        )

        result = await _fetch_one(session, _URL, {}, self._ctx())

        assert result.status == "stale_data"

    async def test_a_snapshot_wrapping_a_metaculus_page_is_refused(self):
        """`is_metaculus_self_ref` keys on hostname, so a wrapped metaculus.com URL sails past
        every self-reference filter in the pipeline — the question would quote itself."""
        captured = self._NOW - timedelta(days=2)
        wrapped = _snapshot_url("https://www.metaculus.com/questions/45001/", captured=captured)
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"", content_type="text/html"),
                wayback_snapshot_url(_URL, now=self._NOW): FakeResponse(302, headers={"Location": wrapped}),
                wrapped: FakeResponse(200, body=self._archive_page(), content_type="text/html"),
            }
        )

        result = await _fetch_one(session, _URL, {}, self._ctx())

        assert result.status == "blocked"
        assert result.route == "wayback"
        assert result.text == ""

    @pytest.mark.parametrize(
        "innermost",
        ["https://www.metaculus.com/questions/45001/", "http://169.254.169.254/latest/meta-data/"],
    )
    async def test_a_nested_capture_is_unwrapped_to_its_innermost_url(self, innermost):
        """A capture OF a capture presents `web.archive.org` as its inner host, which clears both
        the self-reference test and the public-URL test at one level of unwrapping."""
        captured = self._NOW - timedelta(days=2)
        inner = _snapshot_url(innermost, captured=captured - timedelta(days=1))
        wrapped = _snapshot_url(inner, captured=captured)
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"", content_type="text/html"),
                wayback_snapshot_url(_URL, now=self._NOW): FakeResponse(302, headers={"Location": wrapped}),
                wrapped: FakeResponse(200, body=self._archive_page(), content_type="text/html"),
            }
        )

        result = await _fetch_one(session, _URL, {}, self._ctx())

        assert result.status == "blocked"
        assert result.text == ""

    async def test_a_snapshot_wrapping_a_private_address_is_refused(self, monkeypatch):
        captured = self._NOW - timedelta(days=2)
        wrapped = _snapshot_url("http://169.254.169.254/latest/meta-data/", captured=captured)
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"", content_type="text/html"),
                wayback_snapshot_url(_URL, now=self._NOW): FakeResponse(302, headers={"Location": wrapped}),
                wrapped: FakeResponse(200, body=self._archive_page(), content_type="text/html"),
            }
        )

        result = await _fetch_one(session, _URL, {}, self._ctx())

        assert result.status == "blocked"
        assert result.text == ""

    async def test_a_redirect_chain_past_the_hop_cap_declines(self):
        """The archive never served a copy, which is a different fact from a stale one — so the
        direct route's own status stands rather than being overwritten by a fact about the
        archive."""
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"", content_type="text/html"),
                "https://web.archive.org/": FakeResponse(
                    302, headers={"Location": "https://web.archive.org/web/2026id_/https://x.example.com/loop"}
                ),
            }
        )

        result = await _fetch_one(session, _URL, {}, self._ctx())

        assert result.status == "blocked"
        assert result.route == "wayback"

    async def test_no_archived_copy_at_all_declines(self, caplog):
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"", content_type="text/html"),
                wayback_snapshot_url(_URL, now=self._NOW): FakeResponse(404, body=b"", content_type="text/html"),
            }
        )

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.resolution_source"):
            result = await _fetch_one(session, _URL, {}, self._ctx())

        assert result.status == "blocked"
        assert result.route == "wayback"
        # The archive never redirected onto a dated capture, so this is the only decline the
        # "no archived copy" wording is true of.
        assert "no archived copy served for tracker.example.com" in caplog.text

    async def test_the_rung_is_skipped_below_its_floor(self, monkeypatch):
        monkeypatch.setattr(FetchContext, "rung_budget_s", lambda self: 4.0)
        session = FakeSession({_URL: FakeResponse(403, body=b"", content_type="text/html")})

        result = await _fetch_one(session, _URL, {}, self._ctx())

        assert result.status == "blocked"
        assert result.route == "direct"
        assert [(a.rung, a.skipped_reason) for a in result.rung_attempts] == [("wayback", "wall_budget")]

    async def test_the_per_question_cap_binds_across_cited_urls(self):
        """Every snapshot shares netloc web.archive.org, so N cited URLs would queue into N
        sequential archive fetches behind one gate inside a 45 s wall."""
        shared = QuestionRungBudget()
        urls = [f"https://dead{index}.example.com/page" for index in range(3)]
        handlers: dict[str, object] = {}
        for url in urls:
            snapshot = _snapshot_url(url, captured=self._NOW - timedelta(days=2))
            handlers[url] = FakeResponse(403, body=b"", content_type="text/html")
            handlers[wayback_snapshot_url(url, now=self._NOW)] = FakeResponse(302, headers={"Location": snapshot})
            handlers[snapshot] = FakeResponse(200, body=self._archive_page(), content_type="text/html")
        session = FakeSession(handlers)

        # Spelled out rather than looped: the cap is order-dependent, so which call is third is
        # the assertion.
        first = await _fetch_one(session, urls[0], {}, FetchContext(now=self._NOW, shared=shared))
        second = await _fetch_one(session, urls[1], {}, FetchContext(now=self._NOW, shared=shared))
        third = await _fetch_one(session, urls[2], {}, FetchContext(now=self._NOW, shared=shared))
        results = [first, second, third]

        assert [r.route for r in results] == ["wayback", "wayback", "direct"]
        assert results[2].status == "blocked"
        assert [(a.rung, a.skipped_reason) for a in results[2].rung_attempts] == [("wayback", "wayback_cap")]
        assert _rung_counts(results)["wayback_attempts"] == 2
        assert _rung_counts(results)["wayback_cap_skips"] == 1

    async def test_an_archived_pdf_keeps_the_wayback_route_and_its_caveat(self):
        """The snapshot fetch runs on a CHILD context, so the rungs the archive's own bytes go
        through (here the local PDF read) stay off the cited URL's record.

        Sharing the page's context stamped this result `route=pdf_local`, double-counted one
        rescue as a Wayback attempt AND a document read, and — because `_route_caveats` keys on
        the route — rendered the PDF sentence while DROPPING the archived-copy disclosure the
        operator made the condition of admitting a snapshot at all."""
        session = self._session(
            page=FakeResponse(403, body=b"denied", content_type="text/html"),
            captured=self._NOW - timedelta(days=3),
            archived=FakeResponse(
                200,
                body=build_text_pdf([["Hospitalizations reported: 922", "Deaths reported: 2 as of August 24"]]),
                content_type="application/pdf",
            ),
        )

        result = await _fetch_one(session, _URL, {}, FetchContext(now=self._NOW, query="hospitalizations reported"))

        assert result.status == "success"
        assert result.route == "wayback"
        assert [a.rung for a in result.rung_attempts] == ["wayback"]
        assert result.text.startswith("[Archived copy from the Wayback Machine, captured 2026-09-01, 3 days before")
        assert "922" in result.text
        counts = _rung_counts([result])
        assert counts["wayback_attempts"] == 1
        assert counts["pdf_documents_read"] == 0
        rendered = format_resolution_sections([result], self._NOW)
        assert ROUTE_CAVEATS["wayback"] in rendered
        assert ROUTE_CAVEATS["pdf_local"] not in rendered

    async def test_an_archived_page_that_is_itself_unreadable_leaves_the_direct_status(self, caplog):
        session = self._session(
            page=FakeResponse(403, body=b"", content_type="text/html"),
            captured=self._NOW - timedelta(days=2),
            archived=FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
        )

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.resolution_source"):
            result = await _fetch_one(session, _URL, {}, self._ctx())

        assert result.status == "blocked"
        assert result.route == "wayback"
        # The decline is right and the wording has to match it: the archive DID serve this
        # capture, and calling that "no archived copy served" (which it did, until apnews.com
        # showed up in the sweep as an empty archive) describes a fact we never established.
        assert "an archived capture was served but is unusable for tracker.example.com" in caplog.text
        assert "no archived copy served" not in caplog.text


class TestUrlContextRung:
    """The LAST rung and the only paid one: every gate is checked before a cent is spent, and
    zero successful retrievals discards the text rather than rendering recall as evidence."""

    _ANSWER = (
        "The Bureau of Labor Statistics work stoppages page reports 12 major work stoppages "
        "beginning in 2026 through August, per the table dated 2026-08-28."
    )

    def _reader(self, *, text: str = "", retrievals: int = 1, statuses: list[str] | None = None):
        calls: list[dict[str, object]] = []

        def _read(url, ask, **kwargs):
            calls.append({"url": url, "ask": ask, **kwargs})
            return (text or self._ANSWER, retrievals, statuses or ["URL_RETRIEVAL_STATUS_SUCCESS"])

        return _read, calls

    def _session(self, *, robots: bytes = b"User-agent: *\nAllow: /\n") -> FakeSession:
        return FakeSession(
            {
                _URL: FakeResponse(403, body=b"denied", content_type="text/html"),
                "https://tracker.example.com/robots.txt": FakeResponse(200, body=robots, content_type="text/plain"),
            }
        )

    def _arm(self, monkeypatch, reader) -> None:
        monkeypatch.setenv("RESOLUTION_SOURCE_URL_CONTEXT_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(resolution_source, "run_url_context_read", reader)
        reset_robots_cache()

    async def test_the_flag_off_never_calls_the_reader(self, monkeypatch):
        """Ships OFF and stays off in every workflow: it is the only rung that spends money and
        the only one whose product is a model's answer rather than the host's bytes."""
        reader, calls = self._reader()
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(resolution_source, "run_url_context_read", reader)
        monkeypatch.delenv("RESOLUTION_SOURCE_URL_CONTEXT_ENABLED", raising=False)

        result = await _fetch_one(self._session(), _URL, {})

        assert result.status == "blocked"
        assert result.route == "direct"
        assert calls == []

    async def test_an_allowed_host_is_read_and_disclosed(self, monkeypatch):
        reader, calls = self._reader()
        self._arm(monkeypatch, reader)

        result = await _fetch_one(self._session(), _URL, {}, FetchContext(query="How many stoppages?"))

        assert result.status == "success"
        assert result.route == "url_context"
        assert result.text.startswith(
            "[Read via Gemini url_context because the live page could not be fetched (blocked); "
            "model-mediated, not a byte-for-byte copy.]"
        )
        assert "12 major work stoppages" in result.text
        # The ask is the question's own title plus resolution criteria, which is what decides
        # what the reader looks for.
        assert calls[0]["ask"] == "How many stoppages?"
        assert calls[0]["role"] == "resolution_source"
        assert _rung_counts([result])["url_context_reads"] == 1

    async def test_zero_retrievals_discards_the_text(self, monkeypatch, caplog):
        """Gemini answers fluently out of parametric memory when every retrieval failed (Q38195),
        and this section is captioned primary grading evidence."""
        reader, _calls = self._reader(retrievals=0, statuses=["URL_RETRIEVAL_STATUS_ERROR"])
        self._arm(monkeypatch, reader)

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.resolution_source"):
            result = await _fetch_one(self._session(), _URL, {}, FetchContext(query="ask"))

        assert result.status == "ungrounded"
        assert result.text == ""
        # Deliberately unregistered while the flag is off everywhere, so the spelling is pinned
        # HERE: parallel to AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED, the v2 reader's twin, so the
        # spec that eventually registers it matches the lines already in the logs.
        assert (
            "RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url=https://tracker.example.com/senate "
            "statuses=URL_RETRIEVAL_STATUS_ERROR"
        ) in caplog.messages

    async def test_the_ungrounded_line_says_none_when_the_sdk_reported_no_statuses(self, monkeypatch, caplog):
        def _read(url, ask, **kwargs):
            del url, ask, kwargs
            return ("", 0, [])

        self._arm(monkeypatch, _read)

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.resolution_source"):
            await _fetch_one(self._session(), _URL, {}, FetchContext(query="ask"))

        assert (
            "RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url=https://tracker.example.com/senate statuses=none"
        ) in caplog.messages

    async def test_a_google_extended_disallowing_host_is_skipped_before_paying(self, monkeypatch, caplog):
        """Proven live 2026-09-03: a host that disallows the token refuses the fetch server-side,
        so the read would be spend with a known-zero return."""
        reader, calls = self._reader()
        self._arm(monkeypatch, reader)
        session = self._session(robots=b"User-agent: Google-Extended\nDisallow: /\n")

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.resolution_source"):
            result = await _fetch_one(session, _URL, {}, FetchContext(query="ask"))

        assert result.status == "blocked"
        assert result.route == "direct"
        assert calls == []
        assert _rung_counts([result])["url_context_robots_skips"] == 1
        # Unregistered while the flag is off everywhere, so pinned here; parallel to the v2
        # reader's AGENTIC_URLCONTEXT_ROBOTS_SKIP.
        assert (
            "RESOLUTION_SOURCE_URLCONTEXT_ROBOTS_SKIP: url=https://tracker.example.com/senate host=tracker.example.com"
        ) in caplog.messages

    async def test_a_robots_file_blocking_generic_crawlers_still_pays(self, monkeypatch):
        """A different and much broader policy than the one this pre-check implements: our own
        free rungs are unaffected by robots.txt, and only the Google-Extended group is read."""
        reader, calls = self._reader()
        self._arm(monkeypatch, reader)
        session = self._session(robots=b"User-agent: *\nDisallow: /\n")

        result = await _fetch_one(session, _URL, {}, FetchContext(query="ask"))

        assert result.status == "success"
        assert len(calls) == 1

    async def test_a_document_we_read_in_full_is_never_sent_to_the_paid_reader(self, monkeypatch):
        """`no_matching_passage` is inside the trigger STATUSES and outside the trigger
        POPULATION. We hold the whole document's text and its outline, so a model re-reading the
        same PDF cannot find a passage BM25 could not — it would be spend with a known-zero
        return, which is the same test every other exclusion in that set passes. Reason-scoped,
        because `thin_page` and `embed_shell` are pages our client genuinely could not read."""
        reader, calls = self._reader()
        self._arm(monkeypatch, reader)
        pdf_url = "https://cdc.example.com/report.pdf"
        session = FakeSession(
            {
                pdf_url: FakeResponse(
                    200,
                    body=build_text_pdf([["Hospitalizations reported: 922", "Deaths reported: 2"]]),
                    content_type="application/pdf",
                ),
                # Served, and ALLOWING: without the exclusion every later gate here passes, so
                # the assertion that fails is the one that matters — the paid call was made.
                "https://cdc.example.com/robots.txt": FakeResponse(
                    200, body=b"User-agent: *\nAllow: /\n", content_type="text/plain"
                ),
            }
        )

        result = await _fetch_one(session, pdf_url, {}, FetchContext(query="corn futures settlement"))

        assert calls == [], "the paid reader was asked to re-read a document we already hold"
        assert result.status == "no_resolving_content"
        assert result.status_reason == "no_matching_passage"
        assert result.route == "pdf_local"

    async def test_a_missing_api_key_skips_without_calling(self, monkeypatch):
        reader, calls = self._reader()
        monkeypatch.setenv("RESOLUTION_SOURCE_URL_CONTEXT_ENABLED", "true")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setattr(resolution_source, "run_url_context_read", reader)

        result = await _fetch_one(self._session(), _URL, {}, FetchContext(query="ask"))

        assert result.status == "blocked"
        assert calls == []
        assert [(a.rung, a.skipped_reason) for a in result.rung_attempts] == [("url_context", "no_api_key")]
        # Its own count: without it "flag on, key missing" is byte-identical in the archive to
        # "flag off", and the moment that matters is the paid flag's rollout.
        assert _rung_counts([result])["url_context_no_api_key_skips"] == 1

    async def test_the_rung_is_skipped_below_its_floor(self, monkeypatch):
        reader, calls = self._reader()
        self._arm(monkeypatch, reader)
        monkeypatch.setattr(FetchContext, "rung_budget_s", lambda self: 9.0)

        result = await _fetch_one(self._session(), _URL, {}, FetchContext(query="ask"))

        assert result.status == "blocked"
        assert calls == []
        assert ("url_context", "wall_budget") in [(a.rung, a.skipped_reason) for a in result.rung_attempts]

    def _budget_that_drops_after_the_robots_read(self, session: FakeSession, *, before: float, after: float):
        """A wall budget that reads ``before`` until the robots.txt GET has gone out, then ``after``.

        The pre-check is the one gate that costs a request, so it is the one place the budget
        can move between being read and being spent."""

        def _budget(_self: FetchContext) -> float:
            return after if any(url.endswith("/robots.txt") for url in session.requested) else before

        return _budget

    async def test_the_reader_is_sized_off_the_budget_left_after_the_robots_pre_check(self, monkeypatch):
        """The read runs in a thread, which `wait_for` cannot cancel, so the client ceiling is the
        only thing that returns the worker — and a ceiling sized off a figure read BEFORE a real
        request went out could outlive the provider's wall while the money is spent anyway."""
        reader, calls = self._reader()
        self._arm(monkeypatch, reader)
        session = self._session()
        monkeypatch.setattr(
            FetchContext,
            "rung_budget_s",
            self._budget_that_drops_after_the_robots_read(session, before=20.0, after=16.0),
        )

        result = await _fetch_one(session, _URL, {}, FetchContext(query="ask"))

        assert result.status == "success"
        assert calls[0]["attempts"] == resolution_source.RESOLUTION_SOURCE_URL_CONTEXT_ATTEMPTS
        assert calls[0]["timeout_ms"] == int((16.0 - resolution_source.RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S) * 1000)

    async def test_a_pre_check_that_eats_the_room_skips_before_paying(self, monkeypatch):
        reader, calls = self._reader()
        self._arm(monkeypatch, reader)
        session = self._session()
        monkeypatch.setattr(
            FetchContext,
            "rung_budget_s",
            self._budget_that_drops_after_the_robots_read(session, before=20.0, after=5.0),
        )

        result = await _fetch_one(session, _URL, {}, FetchContext(query="ask"))

        assert result.status == "blocked"
        assert calls == []
        assert "https://tracker.example.com/robots.txt" in session.requested
        # One skip, recorded AFTER the pre-check ate the room, never two.
        assert [(a.rung, a.skipped_reason) for a in result.rung_attempts] == [("url_context", "wall_budget")]

    async def test_the_robots_pre_check_is_bounded_and_fails_toward_paying(self, monkeypatch):
        """A robots.txt that never answers must neither hold the paid rung nor withhold it: the
        pre-check gets the same fixed bound gap-fill v2 gives it, and an unreadable policy
        proceeds to pay, the only direction it is allowed to fail in."""

        class _HangingResponse(FakeResponse):
            async def read(self) -> bytes:
                await asyncio.Event().wait()
                return b""

        reader, calls = self._reader()
        self._arm(monkeypatch, reader)
        monkeypatch.setattr(resolution_source, "ROBOTS_FETCH_TIMEOUT_S", 0.05)
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"denied", content_type="text/html"),
                "https://tracker.example.com/robots.txt": _HangingResponse(200, content_type="text/plain"),
            }
        )

        started = time.monotonic()
        result = await _fetch_one(session, _URL, {}, FetchContext(query="ask"))

        assert time.monotonic() - started < 2.0
        assert result.status == "success"
        assert len(calls) == 1

    @pytest.mark.parametrize("status", [404, 410])
    async def test_a_missing_page_is_never_sent_to_the_reader(self, monkeypatch, status):
        """A 404 has no page to read, so the spend would buy nothing."""
        reader, calls = self._reader()
        self._arm(monkeypatch, reader)
        session = FakeSession({_URL: FakeResponse(status, body=b"", content_type="text/html")})

        result = await _fetch_one(session, _URL, {}, FetchContext(query="ask"))

        assert result.status == "not_found"
        assert calls == []

    async def test_a_reader_failure_leaves_the_direct_status(self, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise RuntimeError("reader exploded")

        self._arm(monkeypatch, _boom)

        result = await _fetch_one(self._session(), _URL, {}, FetchContext(query="ask"))

        assert result.status == "blocked"
        assert result.route == "url_context"


class TestFastPath:
    """A question on the time-budget fast path declines the two EXPENSIVE rungs — the browser and
    the paid reader — before any side effect, with its own greppable skip reason. The cheap rungs
    (meta-refresh, local PDF, derived-feed reuse, Wayback) run exactly as they do off it, and a
    question with no fast path is byte-identical to before the gate existed."""

    async def test_the_provider_declines_the_browser_on_the_fast_path(self, monkeypatch):
        calls: list[dict[str, object]] = []
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setattr(
            resolution_source,
            "render_page",
            _fake_render(_rendered_document(f"<h1>Polling average</h1><p>{_RENDERED_PROSE}</p>"), calls),
        )
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        question = _mock_question(resolution_criteria=f"Resolves per {_URL}")

        section = await resolution_source_provider(is_benchmarking=False, fast_path=True)(question)

        assert calls == []
        assert "tracker.example.com: js_wall" in section
        counts = pop_provider_detail(question.id_of_question, "resolution_source")["counts"]
        assert counts["fast_path_skips"] == 1
        assert counts["rendered_attempts"] == 0
        assert counts["renderer_unavailable_skips"] == 0

    async def test_off_the_fast_path_the_provider_renders_as_before(self, monkeypatch):
        calls: list[dict[str, object]] = []
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setattr(
            resolution_source,
            "render_page",
            _fake_render(_rendered_document(f"<h1>Polling average</h1><p>{_RENDERED_PROSE}</p>"), calls),
        )
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        question = _mock_question(resolution_criteria=f"Resolves per {_URL}")

        section = await resolution_source_provider(is_benchmarking=False)(question)

        assert len(calls) == 1
        assert "Nebraska Senate polling average" in section
        counts = pop_provider_detail(question.id_of_question, "resolution_source")["counts"]
        assert counts["fast_path_skips"] == 0
        assert counts["rendered_attempts"] == 1

    async def test_the_paid_reader_declines_on_the_fast_path_only_when_it_was_armed(self, monkeypatch):
        """Recorded only when the rung would otherwise have been considered: with the flag off a
        fast-path skip would read as spend avoided on a rung that could never have fired."""
        calls: list[dict[str, object]] = []

        def _read(url, ask, **kwargs):
            calls.append({"url": url, "ask": ask, **kwargs})
            return ("text", 1, ["URL_RETRIEVAL_STATUS_SUCCESS"])

        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(resolution_source, "run_url_context_read", _read)
        reset_robots_cache()
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"denied", content_type="text/html"),
                "https://tracker.example.com/robots.txt": FakeResponse(
                    200, body=b"User-agent: *\nAllow: /\n", content_type="text/plain"
                ),
            }
        )

        monkeypatch.setenv("RESOLUTION_SOURCE_URL_CONTEXT_ENABLED", "true")
        armed = await _fetch_one(session, _URL, {}, FetchContext(query="ask", fast_path=True))
        monkeypatch.delenv("RESOLUTION_SOURCE_URL_CONTEXT_ENABLED")
        unarmed = await _fetch_one(session, _URL, {}, FetchContext(query="ask", fast_path=True))

        assert calls == []
        assert armed.status == "blocked"
        assert [(a.rung, a.skipped_reason) for a in armed.rung_attempts] == [("url_context", "fast_path")]
        assert unarmed.rung_attempts == []
        # Declined before the pre-check, so no request went out for it either.
        assert "https://tracker.example.com/robots.txt" not in session.requested

    async def test_the_cheap_rungs_still_run_on_the_fast_path(self, monkeypatch):
        """The Wayback rung and a remembered derived feed are ordinary GETs and stay in."""
        monkeypatch.setattr(resolution_source, "_WAYBACK_TRIGGER_STATUSES", _WAYBACK_TRIGGER_STATUSES)
        now = datetime(2026, 9, 4, tzinfo=UTC)
        snapshot = _snapshot_url(_URL, captured=now - timedelta(days=2))
        feed_page = "https://tracker.example.com/house"
        resolution_source.derived_api.remember_endpoint(feed_page, _FEED_URL)
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"", content_type="text/html"),
                wayback_snapshot_url(_URL, now=now): FakeResponse(302, headers={"Location": snapshot}),
                snapshot: FakeResponse(200, body=_prose_page(_RENDERED_PROSE), content_type="text/html"),
                feed_page: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                _FEED_URL: FakeResponse(200, body=b'{"series":[{"v":1}]}', content_type="application/json"),
            }
        )

        archived = await _fetch_one(session, _URL, {}, FetchContext(now=now, fast_path=True))
        derived = await _fetch_one(session, feed_page, {}, FetchContext(now=now, fast_path=True))

        assert archived.route == "wayback"
        assert archived.status == "success"
        assert derived.route == "derived_api"
        assert derived.status == "success"
        assert [(a.rung, a.skipped_reason) for a in derived.rung_attempts] == [("derived_api", "")]


def _success(route: FetchRoute, url: str = _URL) -> FetchResult:
    return FetchResult(
        url=url, status="success", text="body text", http_status=200, content_type="text/html", route=route
    )


class TestRouteCaveats:
    """A forecaster reading a rescued section has to be able to tell how it was obtained, since
    the heading above it calls the whole snapshot primary grading evidence."""

    _AT = datetime(2026, 9, 4, tzinfo=UTC)

    def test_an_all_direct_question_renders_byte_identically(self):
        """The overwhelming majority of questions, and the case a diff against the archive has
        to keep clean: a route that adds nothing is unrepresentable in the mapping."""
        rendered = format_resolution_sections([_success("direct")], self._AT)

        assert rendered.splitlines()[0] == (
            "Snapshot of the cited resolution source(s) as of 2026-09-04 — primary grading evidence."
        )
        assert rendered.splitlines()[1] == ""
        for caveat in ROUTE_CAVEATS.values():
            assert caveat not in rendered

    def test_every_route_token_except_direct_has_a_caveat(self):
        """The completeness guard, and the reason `impersonate` carries a sentence for a rung
        that has not shipped: a future route token added to `FetchRoute` without a caveat would
        otherwise render rescued content with no disclosure at all, silently."""
        routes = set(get_args(FetchRoute))

        assert routes - {"direct"} == set(ROUTE_CAVEATS)
        assert "direct" not in ROUTE_CAVEATS

    @pytest.mark.parametrize("route", sorted(ROUTE_CAVEATS))
    def test_every_non_direct_route_says_how_its_section_was_obtained(self, route):
        rendered = format_resolution_sections([_success(route)], self._AT)

        assert ROUTE_CAVEATS[route] in rendered

    def test_one_sentence_per_route_present_in_stable_order(self):
        results = [
            _success("wayback", "https://a.example.com/p"),
            _success("rendered", "https://b.example.com/p"),
            _success("wayback", "https://c.example.com/p"),
        ]

        rendered = format_resolution_sections(results, self._AT)

        # Deduped, and ordered by the mapping rather than by fetch order.
        assert rendered.count(ROUTE_CAVEATS["wayback"]) == 1
        assert rendered.index(ROUTE_CAVEATS["rendered"]) < rendered.index(ROUTE_CAVEATS["wayback"])

    def test_a_section_dropped_by_the_budget_adds_no_caveat(self, monkeypatch):
        """The caveat describes an artifact a forecaster can SEE. Computed over every success, it
        told forecasters a section below was rendered in a browser when the aggregate budget had
        already dropped that section (reproduced on prod constants: 5 x 6000 per-URL pages
        against an 18000 total, with the rendered page cited last)."""
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_TOTAL_MAX_CHARS", 40)
        first = FetchResult(
            url="https://a.example.com/p", status="success", text="x" * 60, http_status=200, content_type="text/html"
        )
        rendered_last = _success("rendered", "https://b.example.com/p")

        rendered = format_resolution_sections([first, rendered_last], self._AT)

        assert "[1 additional source(s) omitted — section budget]" in rendered
        assert "### https://b.example.com/p" not in rendered
        assert ROUTE_CAVEATS["rendered"] not in rendered

    def test_a_kept_section_keeps_its_caveat_when_a_sibling_is_dropped(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_TOTAL_MAX_CHARS", 40)
        rendered_first = _success("rendered", "https://b.example.com/p")
        dropped = FetchResult(
            url="https://a.example.com/p", status="success", text="x" * 60, http_status=200, content_type="text/html"
        )

        rendered = format_resolution_sections([rendered_first, dropped], self._AT)

        assert "### https://b.example.com/p" in rendered
        assert ROUTE_CAVEATS["rendered"] in rendered

    def test_a_failed_rung_adds_no_caveat(self):
        """A caveat describes an artifact a forecaster can see; a rung that fired and failed left
        the direct outcome, which the failure notice already names."""
        failed = FetchResult(
            url=_URL, status="js_wall", text="", http_status=200, content_type="text/html", route="rendered"
        )

        rendered = format_resolution_sections([_success("direct", "https://ok.example.com/p"), failed], self._AT)

        assert ROUTE_CAVEATS["rendered"] not in rendered
