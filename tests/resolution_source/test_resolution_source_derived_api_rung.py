"""The derived-API rung: the JSON feed a rendered page fetched for itself, remembered per host."""

from __future__ import annotations

import asyncio

from metaculus_bot.research import rendered_fetch, resolution_presentation, resolution_source
from metaculus_bot.research.provider_diagnostics import pop_provider_detail
from metaculus_bot.research.rendered_fetch import HarvestedJson, RenderedPage
from metaculus_bot.research.resolution_fetch_result import ROUTE_CAVEATS
from metaculus_bot.research.resolution_source import (
    FetchContext,
    _fetch_one,
    _rung_counts,
    resolution_source_provider,
)
from tests.resolution_source_fakes import (
    _FEED_URL,
    _JS_SHELL,
    _URL,
    FakeResponse,
    FakeSession,
    _fake_render,
    _meta_refresh_stub,
    _mock_question,
    _prose_page,
)


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

    async def test_a_remembered_endpoint_that_redirects_keeps_its_hops_off_the_pages_record(self, monkeypatch):
        """The feed GET is a request made on the CITED URL's behalf, so the rungs inside it belong
        to the feed and not to the page.

        `_fetch_direct` follows a meta-refresh stub for up to `MAX_REDIRECTS` hops and stamps a
        `meta_refresh` attempt per hop onto whatever context it is handed. Handed the PAGE's
        context, a remembered endpoint answering the session-expired-portal shape pushes hops
        onto a page that never redirected: `meta_refresh_hops` climbs off zero and `route` lands
        on the hop rather than on the rung that answered. The child context `_aux_ctx` builds is
        what keeps the page's own record to the rungs the page itself earned."""
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(self._harvested(), calls))
        second_url = "https://tracker.example.com/house"
        login_url = "https://tracker.example.com/login"
        session = FakeSession(
            {
                _URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                second_url: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                _FEED_URL: FakeResponse(200, body=_meta_refresh_stub(login_url), content_type="text/html"),
                login_url: FakeResponse(
                    200, body=_prose_page("Your session has expired. " * 30), content_type="text/html"
                ),
            }
        )

        await _fetch_one(session, _URL, {})
        second = await _fetch_one(session, second_url, {})

        # The hop DID happen, on the feed's behalf: without this the assertions below would also
        # hold for a fixture that never redirected at all.
        assert login_url in session.requested
        # The feed's own rungs, and only the page's: the reused endpoint declined (the portal is
        # not JSON), the browser ran, and its harvest served the feed.
        assert [(a.rung, a.skipped_reason) for a in second.rung_attempts] == [
            ("derived_api", ""),
            ("rendered", ""),
            ("derived_api", ""),
        ]
        assert _rung_counts([second])["meta_refresh_hops"] == 0
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
        monkeypatch.setattr(resolution_presentation, "RESOLUTION_SOURCE_PER_URL_MAX_CHARS", 400)
        big = '{"series":[' + ",".join(f'{{"v":{index}}}' for index in range(500)) + "]}"
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(self._harvested(body=big), []))
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert len(result.text) <= 400
        # The lead LEADS, so a later trim reaches the JSON before the provenance line.
        assert result.text.startswith("[This page's own HTML carried no readable content.")
