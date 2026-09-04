"""The dispatcher's rung bookkeeping: what each attempt's `outcome` and `route` say.

``_run_rung`` closes every attempt a rung opened with THAT rung's own wall and outcome, and
``_stamped_with_route`` names the rung that produced the result. Both are what the
``RESOLUTION_SOURCE_ESCALATION`` and ``RESOLUTION_SOURCE_FETCH`` markers key on, so a wrong
stamp is wrong archive data rather than a wrong log line.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from metaculus_bot.research import resolution_source
from metaculus_bot.research.resolution_source import (
    _WAYBACK_TRIGGER_STATUSES,
    FetchContext,
    FetchResult,
    FetchStatus,
    _fetch_one,
    _run_rung,
)
from metaculus_bot.research.robots_policy import reset_robots_cache
from metaculus_bot.research.wayback import wayback_snapshot_url
from tests.resolution_source_fakes import _RENDERED_PROSE, _URL, FakeResponse, FakeSession, _prose_page, _snapshot_url


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
