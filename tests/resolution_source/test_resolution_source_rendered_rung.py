"""The rendered (headless-Chromium) escalation rung: triggers, budget, declines, timeout,
and that a rescued DOM re-enters the same classification path."""

from __future__ import annotations

import asyncio
import time

import pytest

from metaculus_bot.research import rendered_fetch, resolution_source
from metaculus_bot.research.provider_diagnostics import pop_provider_detail
from metaculus_bot.research.resolution_fetch_result import FetchResult
from metaculus_bot.research.resolution_source import (
    FetchContext,
    _fetch_one,
    _rung_counts,
    resolution_source_provider,
)
from tests.resolution_source_fakes import (
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
