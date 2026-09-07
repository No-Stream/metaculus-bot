"""The rendered (headless-Chromium) escalation rung: triggers, budget, declines, timeout,
and that a rescued DOM re-enters the same classification path."""

from __future__ import annotations

import asyncio
import time

import pytest

from metaculus_bot.research import rendered_fetch, resolution_presentation, resolution_source
from metaculus_bot.research.provider_diagnostics import pop_provider_detail
from metaculus_bot.research.rendered_fetch import HarvestedJson, RenderedPage
from metaculus_bot.research.resolution_fetch_result import FetchResult
from metaculus_bot.research.resolution_source import (
    FetchContext,
    _fetch_one,
    _rung_counts,
    resolution_source_provider,
)
from tests.resolution_source_fakes import (
    _FEED_URL,
    _INFOGRAM_EMBED_MARKUP,
    _JS_SHELL,
    _RENDERED_PROSE,
    _URL,
    FakeResponse,
    FakeSession,
    _embed_shell_page,
    _fake_render,
    _mock_question,
    _prose_page,
    _rendered,
    _rendered_document,
)

# Above RESOLUTION_SOURCE_JS_WALL_MIN_CHARS (100) and below the chrome floor (400): the band
# where a 200 carries real prose that is nothing but page furniture, which is what
# `no_resolving_content` / `thin_page` means and what the rendered rung's second trigger is.
_TAB_LIST_CHROME = (
    "Nationwide. Midwest. Northeast. South. West. Select a region above to load its series. "
    "Data updates weekly. About the data. Methodology. Contact us. Terms of use."
)


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


def _assert_the_direct_result_stands_after_a_cut(result: FetchResult, skipped_reason: str) -> dict[str, int]:
    assert result.status == "js_wall"
    assert result.route == "direct"
    attempts = [a for a in result.rung_attempts if a.rung == "rendered"]
    assert [a.skipped_reason for a in attempts] == [skipped_reason]
    # The timed-out memo is the TRANSPORT's, written only when a browser actually ran (pinned in
    # tests/test_rendered_fetch.py); the rung never writes the rendered-to-nothing memo on a cut.
    assert rendered_fetch.rendered_to_nothing(result.url, memo_scope="resolution_source") is False
    assert rendered_fetch._PLAYWRIGHT_WARNED is False
    counts = _rung_counts([result])
    assert counts["rendered_attempts"] == 0
    assert counts["renderer_unavailable_skips"] == 0
    return counts


def _assert_recorded_as_a_render_timeout(result: FetchResult) -> None:
    """The transport's own cut: a browser ran and the page kept navigating."""
    counts = _assert_the_direct_result_stands_after_a_cut(result, "render_timeout")
    assert counts["render_timeout_skips"] == 1
    assert counts["rung_budget_skips"] == 0


def _assert_recorded_as_the_wall_binding(result: FetchResult) -> None:
    """The rung's own outer cut: the render never left the queue, so the wall was what bound."""
    counts = _assert_the_direct_result_stands_after_a_cut(result, "wall_budget")
    assert counts["render_timeout_skips"] == 0
    assert counts["rung_budget_skips"] == 1
    assert counts["rendered_budget_skips"] == 1


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


class TestRenderedRungTimeout:
    """P3-1 (live QA, 2026-09-03): a page that kept navigating held ``page.content()`` for 40 s
    after its goto timed out, the render ran 76 s, and the provider's 45 s wall discarded every
    page the question had fetched. The rung now bounds the whole transport call at the remaining
    budget, on top of the transport's own DOM-read cap, and the two cuts are recorded APART. The
    transport's cut (``RenderTimeout``) is a browser that ran on a page that kept navigating, its
    own skip reason: it says nothing about whether Chromium works, so it must neither read as
    ``renderer_unavailable`` nor latch that warning, and the transport memoises the URL so a
    second question citing the same page does not pay for it again. The rung's own cut (a bare
    ``TimeoutError`` from its ``wait_for``) fires while the render is still queued behind the
    launch gates, which says nothing about the page, so it is the wall binding: ``wall_budget``,
    the same reason the pre-gate floor and the post-gate ``RenderBudgetExpired`` record. The
    direct result is what stands either way.
    """

    async def test_a_render_still_queued_at_the_budget_is_cut_off_and_recorded_as_the_wall(self, monkeypatch):
        """The transport never answers (no browser ran), so the rung's own bound is what fires."""
        calls: list[str] = []
        monkeypatch.setattr(resolution_source, "render_page", _hanging_render(calls))
        _admit_a_render_with(monkeypatch, 0.2)
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        started = time.monotonic()
        result = await _fetch_one(session, _URL, {})
        elapsed = time.monotonic() - started

        assert 0.2 <= elapsed < 2.0
        assert calls == [_URL]
        _assert_recorded_as_the_wall_binding(result)
        (attempt,) = [a for a in result.rung_attempts if a.rung == "rendered"]
        assert attempt.wall_s is not None
        assert attempt.wall_s >= 0.2

    async def test_a_render_queued_behind_the_launch_cap_is_the_wall_binding_not_a_render_timeout(self, monkeypatch):
        """Through the REAL transport: both process-wide launch slots are held by other renders,
        so this one queues on the gate with the wall running and never reaches a launch. Recorded
        as ``render_timeout`` it inflated a count documented as a fact about the page while the
        wall-budget counts stayed at zero."""
        monkeypatch.setattr(resolution_source, "render_page", rendered_fetch.render_page)
        _admit_a_render_with(monkeypatch, 0.2)
        gate = rendered_fetch._RENDERED_FETCH_GLOBAL_SEMAPHORE
        for _ in range(rendered_fetch.RENDER_LAUNCH_CAP):
            await gate.acquire()
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        _assert_recorded_as_the_wall_binding(result)
        assert rendered_fetch.render_timed_out(_URL, memo_scope="resolution_source") is False

    async def test_the_transports_own_dom_read_cut_off_is_recorded_as_a_render_timeout(self, monkeypatch):
        """The inner bound RAISES its own class rather than declining with ``None``, which is what
        lets the rung tell it from a missing browser and from its own outer cut."""

        async def _timed_out(url: str, **kwargs: object) -> None:
            del url, kwargs
            await asyncio.sleep(0)
            raise rendered_fetch.RenderTimeout("the DOM read of tracker.example.com outlived 5000ms")

        monkeypatch.setattr(resolution_source, "render_page", _timed_out)
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        _assert_recorded_as_a_render_timeout(result)


class TestRenderedRungRefusedByTheEdge:
    """The direct GET got a 200 for the URL (that is the trigger), so a non-200 main-frame status
    from the browser is the edge telling Chromium apart, and its interstitial markup is not the
    page. Recorded as its own skip: counted with the fired renders it was byte-identical on the
    escalation line to a render that ran and produced chrome again, and "how often is the runner's
    browser refused where its GET was not" is the rate the ladder's case rests on."""

    async def test_a_browser_targeted_403_is_its_own_skip_and_claims_no_route(self, monkeypatch, caplog):
        challenge = RenderedPage(
            url=_URL,
            content_type="text/html",
            html=_rendered_document("<h1>Checking your browser</h1><p>" + "Please wait. " * 60 + "</p>").html,
            http_status=403,
        )
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(challenge, []))
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            result = await _fetch_one(session, _URL, {})
            resolution_source._log_fetch_outcome_markers(1, [result])

        assert result.status == "js_wall"
        assert result.route == "direct"
        attempts = [a for a in result.rung_attempts if a.rung == "rendered"]
        assert [a.skipped_reason for a in attempts] == ["render_non_200"]
        counts = _rung_counts([result])
        assert counts["render_non_200_skips"] == 1
        assert counts["rendered_attempts"] == 0
        assert counts["renderer_unavailable_skips"] == 0
        # A 403 or 429 is retryable, so the URL stays live for the next question.
        assert rendered_fetch.rendered_to_nothing(_URL, memo_scope="resolution_source") is False
        # A skip emits no escalation line, so nothing claims the browser rung fired.
        assert not [message for message in caplog.messages if "RESOLUTION_SOURCE_ESCALATION" in message]


class TestRenderedRungLandedOffHost:
    """A server-side redirect hop is dialed by Chromium with no check of ours, so the transport
    refuses a main frame that landed on a host other than the one its DNS pin covers
    (``RenderOffHost``). Its own skip: folded into ``renderer_unavailable`` it would point triage
    at the Playwright install, and it is a fact about the page (it answered our GET with a wall
    and sent the browser somewhere else), which is the rate a residual round would ask for. The
    direct result stands and nothing from the render is published."""

    @staticmethod
    async def _off_host_render(url: str, **kwargs: object) -> None:
        del kwargs
        await asyncio.sleep(0)
        raise rendered_fetch.RenderOffHost(
            requested_url=url, final_url="http://10.0.0.8/status", pinned_host="tracker.example.com"
        )

    async def test_an_off_host_landing_is_its_own_skip_and_claims_no_route(self, monkeypatch, caplog):
        monkeypatch.setattr(resolution_source, "render_page", self._off_host_render)
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            result = await _fetch_one(session, _URL, {})
            resolution_source._log_fetch_outcome_markers(1, [result])

        counts = _assert_the_direct_result_stands_after_a_cut(result, "render_off_host")
        assert counts["render_off_host_skips"] == 1
        assert counts["render_dom_too_large_skips"] == 0
        assert counts["render_non_200_skips"] == 0
        # Nothing was read, so nothing claims the browser rung fired.
        assert not [message for message in caplog.messages if "RESOLUTION_SOURCE_ESCALATION" in message]

    async def test_the_count_reaches_the_providers_details(self, monkeypatch):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setattr(resolution_source, "render_page", self._off_host_render)
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        question = _mock_question(resolution_criteria=f"Resolves per {_URL}")

        section = await resolution_source_provider(is_benchmarking=False)(question)

        assert "tracker.example.com: js_wall" in section
        counts = pop_provider_detail(question.id_of_question, "resolution_source")["counts"]
        assert counts["render_off_host_skips"] == 1
        assert counts["renderer_unavailable_skips"] == 0
        assert counts["rendered_attempts"] == 0


class TestRenderedRungRendersTheFinalUrl:
    """The browser is handed the URL the direct fetch LANDED on, after its redirect hops were
    followed and re-guarded (``FetchResult.url`` is the last hop's URL), so the DNS pin covers the
    host that actually serves the content and the landing-host check has the right host to hold
    the browser to. The rung's attempt stays keyed on the cited URL, because that is what the
    escalation line names, and the render memos move to the URL actually rendered."""

    _FINAL = "https://www.tracker.example.com/senate"

    def _redirected_session(self) -> FakeSession:
        return FakeSession(
            {
                _URL: FakeResponse(302, headers={"Location": self._FINAL}),
                self._FINAL: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
            }
        )

    async def test_a_redirected_direct_fetch_hands_the_browser_its_final_url(self, monkeypatch):
        calls: list[dict[str, object]] = []
        rescued = _rendered_document(f"<h1>Polling average</h1><p>{_RENDERED_PROSE}</p>")
        page = RenderedPage(url=self._FINAL, content_type="text/html", html=rescued.html, final_url=self._FINAL)
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(page, calls))

        result = await _fetch_one(self._redirected_session(), _URL, {})

        assert [call["url"] for call in calls] == [self._FINAL]
        assert result.status == "success"
        assert result.route == "rendered"
        assert result.url == self._FINAL
        (attempt,) = [a for a in result.rung_attempts if a.rung == "rendered"]
        assert attempt.url == _URL

    async def test_the_classifier_sees_the_documents_landing_url(self, monkeypatch):
        """``final_url`` is the URL the main frame actually landed on, same host as the rendered URL
        by construction and a different path after a same-host client-side redirect or meta refresh
        (6 of 22 render targets in the 2026-09-04 probe). It is the classifier's base, so relative
        links and the published section URL name the real document, as the direct path's last hop
        and the ``meta_refresh`` route already do; the attempt stays keyed on the cited URL."""
        landed = f"{self._FINAL}/2026/?tab=polls"
        rescued = _rendered_document(f"<h1>Polling average</h1><p>{_RENDERED_PROSE}</p>")
        page = RenderedPage(url=self._FINAL, content_type="text/html", html=rescued.html, final_url=landed)
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(page, []))

        result = await _fetch_one(self._redirected_session(), _URL, {})

        assert result.status == "success"
        assert result.route == "rendered"
        assert result.url == landed
        (attempt,) = [a for a in result.rung_attempts if a.rung == "rendered"]
        assert attempt.url == _URL

    async def test_the_render_memos_are_keyed_on_the_rendered_url(self, monkeypatch):
        calls: list[dict[str, object]] = []
        empty = RenderedPage(url=self._FINAL, content_type="text/html", html=_JS_SHELL.decode(), final_url=self._FINAL)
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(empty, calls))

        first = await _fetch_one(self._redirected_session(), _URL, {})
        second = await _fetch_one(self._redirected_session(), _URL, {})

        assert first.route == "rendered"
        assert rendered_fetch.rendered_to_nothing(self._FINAL, memo_scope="resolution_source") is True
        assert rendered_fetch.rendered_to_nothing(_URL, memo_scope="resolution_source") is False
        assert len(calls) == 1
        attempts = [a for a in second.rung_attempts if a.rung == "rendered"]
        assert [a.skipped_reason for a in attempts] == ["rendered_no_text"]

    @pytest.mark.parametrize(
        "landed",
        ["https://www.metaculus.com/questions/999/", "http://10.0.0.8/status"],
    )
    async def test_a_final_url_the_ladder_would_not_fetch_is_never_rendered(self, monkeypatch, caplog, landed):
        """Unreachable through ``_fetch_direct``, whose every hop is vetted, so it is driven with a
        hand-built direct result: the URL the browser dials is decided here, so the refusal lives
        here too, and it declines before any attempt is opened."""
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(_rendered("<html></html>"), calls))
        direct = FetchResult(url=landed, status="js_wall", text="", http_status=200, content_type="text/html")
        ctx = FetchContext()

        with caplog.at_level("WARNING", logger="metaculus_bot.research.resolution_source"):
            result = await resolution_source._rendered_rung(_URL, direct, {}, ctx)

        assert result is None
        assert calls == []
        assert ctx.rungs == []
        assert [message for message in caplog.messages if "not rendering" in message]


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
        # A transport that never answered is the wall binding on the rung, not a cut-off render.
        assert counts["rendered_budget_skips"] == 1
        assert counts["render_timeout_skips"] == 0
        assert counts["rendered_attempts"] == 0
        assert not [message for message in caplog.messages if "rung unavailable" in message]


def _menu_tree_dom() -> RenderedPage:
    """A rendered DOM the line-shape metric withholds: about 2,000 chars of 48-char listing lines,
    well over the chrome floor and nothing but a release archive (the abs.gov.au shape the
    extractor-policy tests measure the metric on)."""
    months = ("January", "February", "March", "April", "May", "June", "July", "August", "September")
    items = "".join(f"<li>Labour Force, Australia, {month} 2026 Archive release</li>" for month in months * 5)
    return _rendered_document(f"<h1>Labour Force, Australia</h1><ul>{items}</ul>")


class TestRenderedRungMetricWithhold:
    """`chrome_metric_withholds` counts a withhold anywhere on the URL's ladder, and the direct
    fetch of a js_wall page carries nothing for the metric to withhold. The rendered DOM is the
    first extraction of such a URL the metric sees, so its withhold has to reach the count from
    here, whether the harvested feed then rescues the page or nothing does."""

    async def test_a_withheld_rendered_dom_with_no_rescue_is_counted_once(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(_menu_tree_dom(), []))
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "js_wall"
        assert result.route == "rendered"
        assert result.chrome_metric_withheld is True
        counts = _rung_counts([result])
        assert counts["chrome_metric_withholds"] == 1
        assert counts["chrome_metric_withholds_rescued"] == 0
        # The render was tried and found chrome, so the URL is memoised like any empty render.
        assert rendered_fetch.rendered_to_nothing(_URL, memo_scope="resolution_source") is True

    async def test_a_withheld_rendered_dom_the_harvested_feed_rescued_is_counted_under_both_keys(self, monkeypatch):
        menu_tree = _menu_tree_dom()
        feed = b'{"series":[{"date":"2026-09-01","osborn":47.2,"ricketts":45.8}]}'
        page = RenderedPage(
            url=_URL,
            content_type="text/html",
            html=menu_tree.html,
            json_responses=(HarvestedJson(url=_FEED_URL, body=feed),),
        )
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(page, []))
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert result.route == "derived_api"
        assert result.chrome_metric_withheld is True
        counts = _rung_counts([result])
        assert counts["chrome_metric_withholds"] == 1
        assert counts["chrome_metric_withholds_rescued"] == 1


class TestOneJsonVocabulary:
    """The 200-response router used to recognise only `application/json`, while the harvest that
    discovers a feed and the reuse gate that serves it both accepted `text/json` and any `+json`
    suffix. A remembered `+json` endpoint therefore came back `unsupported_type` from its GET and
    never reached the reuse gate, so every later cited URL on the host paid the wasted GET and
    then a full Chromium launch: exactly the saving the derived-feed rung exists to deliver."""

    _FEED = b'{"series":[{"date":"2026-09-01","osborn":47.2,"ricketts":45.8}]}'

    async def test_a_remembered_vnd_api_json_feed_is_served_without_a_second_launch(self, monkeypatch):
        calls: list[dict[str, object]] = []
        harvested = RenderedPage(
            url=_URL,
            content_type="text/html",
            html=_JS_SHELL.decode(),
            json_responses=(HarvestedJson(url=_FEED_URL, body=self._FEED),),
        )
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(harvested, calls))
        second_url = "https://tracker.example.com/house"
        session = FakeSession(
            {
                _URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                second_url: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                _FEED_URL: FakeResponse(200, body=self._FEED, content_type="application/vnd.api+json"),
            }
        )

        first = await _fetch_one(session, _URL, {})
        second = await _fetch_one(session, second_url, {})

        assert first.route == "derived_api"
        assert second.status == "success"
        assert second.route == "derived_api"
        assert '"osborn":47.2' in second.text
        assert len(calls) == 1, "the +json feed fell through the router and the second URL launched a browser"
        assert _FEED_URL in session.requested

    @pytest.mark.parametrize("content_type", ["text/json", "application/geo+json"])
    async def test_a_directly_cited_json_url_is_read_as_text(self, content_type):
        body = b'{"series":[' + b'{"date":"2026-09-01","value":47.2},' * 20 + b"]}"
        session = FakeSession({_URL: FakeResponse(200, body=body, content_type=content_type)})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert result.route == "direct"
        assert '"value":47.2' in result.text


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
        monkeypatch.setattr(resolution_presentation, "RESOLUTION_SOURCE_PER_URL_MAX_CHARS", 200)
        monkeypatch.setattr(
            resolution_source,
            "render_page",
            _fake_render(_rendered_document(f"<h1>Polling average</h1><p>{_RENDERED_PROSE * 20}</p>"), []),
        )
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert len(result.text) <= 200
