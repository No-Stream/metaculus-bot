"""The dispatcher's rung bookkeeping: what each attempt's `outcome` and `route` say.

``_run_rung`` closes every attempt a rung opened with THAT rung's own wall and outcome, and
``_stamped_with_route`` names the rung that produced the result. Both are what the
``RESOLUTION_SOURCE_ESCALATION`` and ``RESOLUTION_SOURCE_FETCH`` markers key on, so a wrong
stamp is wrong archive data rather than a wrong log line.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import get_args

import pytest

from metaculus_bot.research import resolution_source
from metaculus_bot.research.provider_diagnostics import pop_provider_detail
from metaculus_bot.research.rendered_fetch import RenderedPage
from metaculus_bot.research.resolution_fetch_result import RungSkipReason
from metaculus_bot.research.resolution_source import (
    _BUDGET_GATED_RUNGS,
    _RUNG_WALL_SKIP_PHRASE,
    _WAYBACK_TRIGGER_STATUSES,
    FetchContext,
    FetchResult,
    FetchStatus,
    _fetch_one,
    _run_rung,
    _rung_counts,
    resolution_source_provider,
)
from metaculus_bot.research.robots_policy import reset_robots_cache
from metaculus_bot.research.wayback import wayback_snapshot_url
from tests.resolution_source_fakes import (
    _RENDERED_PROSE,
    _URL,
    FakeResponse,
    FakeSession,
    _escape_config,
    _fake_render,
    _mock_question,
    _prose_page,
    _snapshot_url,
)


def _result(status: FetchStatus = "success") -> FetchResult:
    return FetchResult(
        url=_URL, status=status, text="body" if status == "success" else "", http_status=200, content_type="text/html"
    )


class TestRunRung:
    """The bracket every dispatcher site used to copy: read the length first, close afterwards."""

    async def test_a_rung_that_returns_a_result_closes_its_attempts_on_that_status(self):
        ctx = FetchContext()

        async def _rung() -> FetchResult | None:
            ctx.start_rung("wayback", "blocked", _URL)
            await asyncio.sleep(0)
            return _result("success")

        result = await _run_rung(ctx, "blocked", _rung())

        assert result is not None
        assert result.status == "success"
        assert [(a.rung, a.outcome) for a in ctx.rungs] == [("wayback", "success")]
        assert ctx.rungs[0].wall_s is not None

    async def test_a_rung_that_declines_closes_its_attempts_on_the_fallback(self):
        ctx = FetchContext()

        async def _rung() -> FetchResult | None:
            ctx.start_rung("url_context", "blocked", _URL)
            await asyncio.sleep(0)
            return None

        assert await _run_rung(ctx, "blocked", _rung()) is None
        assert [(a.rung, a.outcome) for a in ctx.rungs] == [("url_context", "blocked")]

    async def test_only_the_attempts_the_rung_opened_are_closed(self):
        """An attempt an EARLIER rung left open keeps its own outcome; a skipped one is left alone."""
        ctx = FetchContext()
        earlier = ctx.start_rung("derived_api", "js_wall", _URL)
        earlier.outcome = "js_wall"
        ctx.skip_rung("rendered", "js_wall", _URL, "fast_path")

        async def _rung() -> FetchResult | None:
            ctx.start_rung("wayback", "js_wall", _URL)
            await asyncio.sleep(0)
            return None

        await _run_rung(ctx, "js_wall", _rung())

        assert [(a.rung, a.outcome, a.skipped_reason) for a in ctx.rungs] == [
            ("derived_api", "js_wall", ""),
            ("rendered", None, "fast_path"),
            ("wayback", "js_wall", ""),
        ]


class TestAVerdictKeepsItsOwnRoute:
    """A Wayback `stale_data` withhold is the ladder's fallback when the paid reader then fails.

    Before this pin the result came back `route=url_context status=stale_data` (the LAST rung to
    fire claimed the route) and the paid attempt closed with `outcome=stale_data` (the fallback
    was the withhold's status): a status that rung cannot produce, on both fields an analyst
    partitions the archive by. Dormant while the paid flag is off in every workflow; wrong on the
    marker regardless.
    """

    _NOW = datetime(2026, 9, 4, tzinfo=UTC)

    @pytest.fixture(autouse=True)
    def _arm_both_rungs(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "_WAYBACK_TRIGGER_STATUSES", _WAYBACK_TRIGGER_STATUSES)
        monkeypatch.setenv("RESOLUTION_SOURCE_URL_CONTEXT_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        reset_robots_cache()

    def _session(self) -> FakeSession:
        stale = _snapshot_url(_URL, captured=self._NOW - timedelta(days=400))
        return FakeSession(
            {
                _URL: FakeResponse(403, body=b"", content_type="text/html", headers={"Server": "AkamaiGHost"}),
                wayback_snapshot_url(_URL, now=self._NOW): FakeResponse(302, headers={"Location": stale}),
                stale: FakeResponse(200, body=_prose_page(_RENDERED_PROSE), content_type="text/html"),
                "https://tracker.example.com/robots.txt": FakeResponse(
                    200, body=b"User-agent: *\nAllow: /\n", content_type="text/plain"
                ),
            }
        )

    async def test_a_stale_withhold_the_reader_failed_to_improve_on_keeps_the_wayback_route(self, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise RuntimeError("reader exploded")

        monkeypatch.setattr(resolution_source, "run_url_context_read", _boom)

        result = await _fetch_one(self._session(), _URL, {}, FetchContext(now=self._NOW, query="ask"))

        assert result.status == "stale_data"
        assert result.route == "wayback"
        # Both rungs fired; each attempt closed on ITS outcome: the archive's verdict, then the
        # direct status the failed paid read left standing.
        assert [(a.rung, a.from_status, a.outcome) for a in result.rung_attempts] == [
            ("wayback", "blocked", "stale_data"),
            ("url_context", "blocked", "blocked"),
        ]
        # The withhold still carries the cited host's diagnostics onto the FETCH line.
        assert result.http_status == 403
        assert result.failure_class == "http_403"
        assert result.server == "akamaighost"

    async def test_a_reader_that_rescues_the_page_still_claims_the_route(self, monkeypatch):
        """The exception is for a returned VERDICT only: a rescue is the last rung's own result."""

        def _read(url, ask, **kwargs):
            del url, ask, kwargs
            return ("The page reports 12 major work stoppages.", 1, ["URL_RETRIEVAL_STATUS_SUCCESS"])

        monkeypatch.setattr(resolution_source, "run_url_context_read", _read)

        result = await _fetch_one(self._session(), _URL, {}, FetchContext(now=self._NOW, query="ask"))

        assert result.status == "success"
        assert result.route == "url_context"
        assert [(a.rung, a.outcome) for a in result.rung_attempts] == [
            ("wayback", "stale_data"),
            ("url_context", "success"),
        ]

    async def test_with_the_reader_declining_the_withhold_is_stamped_the_same_way(self, monkeypatch):
        """No paid attempt fires when the key is missing, so the last fired rung IS the archive;
        the pre-stamped route agrees with what the dispatcher would have chosen."""
        monkeypatch.delenv("GOOGLE_API_KEY")

        result = await _fetch_one(self._session(), _URL, {}, FetchContext(now=self._NOW, query="ask"))

        assert result.status == "stale_data"
        assert result.route == "wayback"
        assert [(a.rung, a.outcome, a.skipped_reason) for a in result.rung_attempts] == [
            ("wayback", "stale_data", ""),
            ("url_context", None, "no_api_key"),
        ]


class TestRungSkipReasonCounts:
    """Every member of the closed `RungSkipReason` vocabulary reaches `details["counts"]`.

    The counts are how a skip survives into the archive (skips emit no escalation line), and
    `_rung_counts` reads the members into keys by hand, so a member added to the Literal without
    a key would type-check and then be permanently invisible. Read off the Literal, like the
    repo's other closed vocabularies (`ROUTE_CAVEATS`, `RunMode`, `McMissingKind`).
    """

    @pytest.mark.parametrize("reason", get_args(RungSkipReason))
    def test_every_skip_reason_moves_a_counts_key(self, reason: RungSkipReason):
        ctx = FetchContext()
        ctx.skip_rung("rendered", "js_wall", _URL, reason)
        skipped = _result("js_wall")
        skipped.rung_attempts = list(ctx.rungs)

        baseline = _rung_counts([])
        counts = _rung_counts([skipped])

        assert set(counts) == set(baseline)
        moved = {key for key in counts if counts[key] != baseline[key]}
        assert moved, f"{reason!r} reaches no counts key"
        assert all(counts[key] == baseline[key] + 1 for key in moved)

    @pytest.mark.parametrize("rung", _BUDGET_GATED_RUNGS)
    def test_every_budget_gated_rung_has_its_own_budget_skip_key(self, rung):
        ctx = FetchContext()
        ctx.skip_rung(rung, "blocked", _URL, "wall_budget")
        skipped = _result("blocked")
        skipped.rung_attempts = list(ctx.rungs)

        counts = _rung_counts([skipped])

        assert counts["rung_budget_skips"] == 1
        assert counts[f"{rung}_budget_skips"] == 1

    def test_the_budget_gated_rungs_are_the_phrased_rungs(self):
        """Derived, not restated: `claim_rung_budget` indexes the phrase map, so a rung gated on
        the wall without a phrase raises inside a budget guard, and one phrased without a count
        key loses its per-rung budget skips from the archive."""
        assert tuple(_RUNG_WALL_SKIP_PHRASE) == _BUDGET_GATED_RUNGS

    def test_the_keys_are_stable_with_no_results(self):
        """Zeroes are kept so 'the rung existed and never fired' stays distinguishable from
        'this record predates the rung'."""
        counts = _rung_counts([])

        assert all(value == 0 for value in counts.values())
        assert {"rung_budget_skips", "chrome_metric_withholds", "chrome_metric_withholds_rescued"} <= set(counts)


# The abs.gov.au shape the extractor policy was calibrated on: about 2,000 chars of 48-char
# listing lines, well over the chrome floor and nothing but a menu, so real trafilatura extracts
# it and the line-shape metric withholds it.
_MENU_TREE = (
    "<!doctype html><html><head><title>Labour Force, Australia</title></head><body><nav>Home</nav><main>"
    "<h1>Labour Force, Australia</h1><ul>"
    + "".join(f"<li>Labour Force, Australia, release {i:02d} 2026 Archive release</li>" for i in range(36))
    + "</ul></main></body></html>"
).encode()


class TestChromeMetricWithholdCounts:
    """A metric withhold is a fact about the URL's ladder and is counted wherever the ladder ends.

    Summed off the final result's own flag, a menu tree the rendered rung then rescued reached
    neither key: the rescue's extraction was never withheld, so `chrome_metric_withholds` lost
    exactly the population the policy exists to hand to the browser. The direct fetch's flag is
    carried onto the rescue, and `chrome_metric_withholds_rescued` counts the carried-and-served
    subset.
    """

    async def test_a_withhold_with_no_rescue_counts_once_and_is_not_a_rescue(self):
        session = FakeSession({_URL: FakeResponse(200, body=_MENU_TREE)})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "no_resolving_content"
        assert result.chrome_metric_withheld is True
        counts = _rung_counts([result])
        assert counts["chrome_metric_withholds"] == 1
        assert counts["chrome_metric_withholds_rescued"] == 0

    async def test_a_withhold_the_rendered_rung_rescued_is_counted_under_both_keys(self, monkeypatch):
        page = RenderedPage(url=_URL, content_type="text/html", html=_prose_page(_RENDERED_PROSE).decode())
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(page, []))
        session = FakeSession({_URL: FakeResponse(200, body=_MENU_TREE)})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert result.route == "rendered"
        assert "47.2 percent" in result.text
        # Carried from the direct fetch: the rendered DOM's own extraction passed the metric.
        assert result.chrome_metric_withheld is True
        assert result.precision_rescued is False
        counts = _rung_counts([result])
        assert counts["chrome_metric_withholds"] == 1
        assert counts["chrome_metric_withholds_rescued"] == 1

    async def test_a_chart_block_publishing_alone_is_a_withhold_but_not_a_rescue(self):
        """The chart block is the direct fetch's own content, so `route` stays `direct`."""
        config = {"xAxis": [{"categories": ["2024", "2025"]}], "series": [{"name": "Unemployed", "data": [1, 2]}]}
        page = _MENU_TREE.replace(
            b"</ul></main>",
            f'</ul><div class="charts-highchart" data-chart="{_escape_config(config)}"></div></main>'.encode(),
        )
        session = FakeSession({_URL: FakeResponse(200, body=page)})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        assert result.route == "direct"
        assert result.chrome_metric_withheld is True
        counts = _rung_counts([result])
        assert counts["chrome_metric_withholds"] == 1
        assert counts["chrome_metric_withholds_rescued"] == 0

    async def test_a_page_the_metric_never_withheld_moves_neither_key(self, article_html):
        session = FakeSession({_URL: FakeResponse(200, body=article_html)})

        result = await _fetch_one(session, _URL, {})

        assert result.status == "success"
        counts = _rung_counts([result])
        assert counts["chrome_metric_withholds"] == 0
        assert counts["chrome_metric_withholds_rescued"] == 0

    async def test_the_rescued_key_reaches_the_provider_detail(self, monkeypatch):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        page = RenderedPage(url=_URL, content_type="text/html", html=_prose_page(_RENDERED_PROSE).decode())
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(page, []))
        session = FakeSession({_URL: FakeResponse(200, body=_MENU_TREE)})
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria=f"Resolves per {_URL} on release.")

        await resolution_source_provider(is_benchmarking=False)(q)

        detail = pop_provider_detail(q.id_of_question, "resolution_source")
        assert detail["sources"] == {"tracker.example.com": "ok"}
        assert detail["counts"]["chrome_metric_withholds"] == 1
        assert detail["counts"]["chrome_metric_withholds_rescued"] == 1
        assert detail["counts"]["rendered_attempts"] == 1
