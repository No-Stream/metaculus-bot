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

import pytest

from metaculus_bot.research import rendered_fetch, resolution_source
from metaculus_bot.research.derived_api import reset_derived_endpoints
from metaculus_bot.research.rendered_fetch import HarvestedJson, RenderedPage
from metaculus_bot.research.resolution_source import FetchContext, _fetch_one, _rung_counts
from tests.resolution_source_fakes import (
    _INFOGRAM_EMBED_MARKUP,
    FakeResponse,
    FakeSession,
    _embed_shell_page,
    _prose_page,
)

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
    async def _render(url: str, *, host_gate, goto_timeout_ms: int, harvest_json: bool = False):
        calls.append({"url": url, "goto_timeout_ms": goto_timeout_ms, "harvest_json": harvest_json})
        del host_gate
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
        assert _rung_counts([result])["rung_budget_skips"] == 1
        assert _rung_counts([result])["rendered_attempts"] == 0

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
        assert rendered_fetch.rendered_to_nothing(_URL) is True
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

        assert rendered_fetch.rendered_to_nothing(_URL) is False


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
        assert rendered_fetch.rendered_to_nothing(_URL) is True

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
