"""Ladder dispatch and ordering: per-rung escalation lines, the fast-path gate across rungs,
and the provider-level marker lines each rung produces end to end."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime, timedelta

from metaculus_bot.constants import RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV
from metaculus_bot.research import resolution_source
from metaculus_bot.research.provider_diagnostics import pop_provider_detail
from metaculus_bot.research.rendered_fetch import HarvestedJson, RenderedPage
from metaculus_bot.research.resolution_fetch_result import ROUTE_CAVEATS
from metaculus_bot.research.resolution_source import (
    _IMPERSONATE_TRIGGER_HTTP_STATUS,
    _WAYBACK_TRIGGER_STATUSES,
    FetchContext,
    _fetch_one,
    resolution_source_provider,
)
from metaculus_bot.research.wayback import wayback_snapshot_url
from tests.resolution_source_fakes import (
    _FEED_URL,
    _JS_SHELL,
    _RENDERED_PROSE,
    _URL,
    FakeResponse,
    FakeSession,
    _fake_render,
    _impersonated,
    _meta_refresh_stub,
    _mock_question,
    _prose_page,
    _rendered_document,
    _snapshot_url,
    arm_paid_rung,
    fake_impersonated_fetch,
    paid_reader,
    refused_page_with_robots,
)


class TestEscalationLinesArePerRung:
    """``outcome`` and ``wall_s`` on the ``RESOLUTION_SOURCE_ESCALATION`` line belong to the rung
    the line is about, not to the ladder.

    Before the per-rung close, both were stamped once after the whole ladder: every line for a
    URL carried the FINAL status, so a rung that fired and failed read as having rescued the
    page, and each attempt's wall ran to the end of the last rung. Reproduced on the repo's own
    dead-feed-then-browser shape — three lines for one URL, all ``outcome=success``, the first a
    feed GET that answered 503.
    """

    _FEED = '{"series":[{"date":"2026-09-01","osborn":47.2,"ricketts":45.8}]}'
    _SECOND_URL = "https://tracker.example.com/house"

    def _harvested(self) -> RenderedPage:
        return RenderedPage(
            url=_URL,
            content_type="text/html",
            html=_JS_SHELL.decode(),
            json_responses=(HarvestedJson(url=_FEED_URL, body=self._FEED.encode()),),
        )

    async def test_a_dead_feed_get_then_a_rescuing_render_read_as_two_outcomes(self, monkeypatch, caplog):
        harvested = self._harvested()

        async def _slow_render(url: str, **kwargs: object) -> RenderedPage:
            del url, kwargs
            # Measurable, so the feed GET's wall is provably NOT the ladder's.
            await asyncio.sleep(0.02)
            return harvested

        monkeypatch.setattr(resolution_source, "render_page", _slow_render)
        session = FakeSession(
            {
                _URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                self._SECOND_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
                # Reached only by the second URL's derived-feed GET (the first is served straight
                # off its render), and dead.
                _FEED_URL: FakeResponse(503, body=b"", content_type="text/html"),
            }
        )

        await _fetch_one(session, _URL, {})
        second = await _fetch_one(session, self._SECOND_URL, {})
        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.resolution_source"):
            resolution_source._log_fetch_outcome_markers(999, [second])

        feed_get, render, harvest = second.rung_attempts
        assert [(a.rung, a.outcome) for a in (feed_get, render, harvest)] == [
            ("derived_api", "js_wall"),
            # The render's OWN verdict: the DOM was still empty, and the harvested feed is what
            # rescued the page.
            ("rendered", "js_wall"),
            ("derived_api", "success"),
        ]
        escalations = [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_ESCALATION:")]
        assert [re.findall(r"rung=(\S+) outcome=(\S+)", m) for m in escalations] == [
            [("derived_api", "js_wall")],
            [("rendered", "js_wall")],
            [("derived_api", "success")],
        ]
        # Per-rung walls: the feed GET was closed before the render opened, so it is not billed
        # for the launch that followed it; the render's wall is at least its own sleep.
        assert feed_get.wall_s is not None
        assert render.wall_s is not None
        assert feed_get.started_at + feed_get.wall_s <= render.started_at
        assert render.wall_s >= 0.02
        assert feed_get.wall_s < render.wall_s

    async def test_the_direct_fetchs_own_rungs_are_closed_with_its_status(self, monkeypatch, caplog):
        """A meta-refresh hop that led to a JS wall, then a browser that rescued it: the hop's
        line reads the status it left standing, the render's reads the rescue."""
        target = "https://cdc.example.com/data/current"
        stub = "https://cdc.example.com/surveillance"
        monkeypatch.setattr(
            resolution_source,
            "render_page",
            _fake_render(_rendered_document(f"<h1>Polling average</h1><p>{_RENDERED_PROSE}</p>"), []),
        )
        session = FakeSession(
            {
                stub: FakeResponse(200, body=_meta_refresh_stub("/data/current")),
                target: FakeResponse(200, body=_JS_SHELL, content_type="text/html"),
            }
        )

        result = await _fetch_one(session, stub, {})
        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.resolution_source"):
            resolution_source._log_fetch_outcome_markers(999, [result])

        assert result.status == "success"
        assert [(a.rung, a.outcome) for a in result.rung_attempts] == [
            ("meta_refresh", "js_wall"),
            ("rendered", "success"),
        ]
        escalations = [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_ESCALATION:")]
        assert "rung=meta_refresh outcome=js_wall" in escalations[0]
        assert "rung=rendered outcome=success" in escalations[1]


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
        reader, calls = paid_reader()
        session = refused_page_with_robots()

        # Armed, then disarmed by dropping the flag alone, so the two runs differ in nothing else.
        arm_paid_rung(monkeypatch, reader)
        armed = await _fetch_one(session, _URL, {}, FetchContext(query="ask", fast_path=True))
        monkeypatch.delenv(RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV)
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

    async def test_the_impersonated_retry_still_runs_on_the_fast_path(self, monkeypatch):
        """One GET against a host that just answered us, so it is a cheap rung like the hop above
        and records no `fast_path` skip; the gate is reserved for the browser and the paid read."""
        monkeypatch.setattr(resolution_source, "_IMPERSONATE_TRIGGER_HTTP_STATUS", _IMPERSONATE_TRIGGER_HTTP_STATUS)
        calls: list[dict[str, object]] = []
        monkeypatch.setattr(
            resolution_source,
            "fetch_impersonated",
            fake_impersonated_fetch(_impersonated(200, body=_prose_page(_RENDERED_PROSE)), calls),
        )
        session = FakeSession({_URL: FakeResponse(403, body=b"", content_type="text/html")})

        result = await _fetch_one(session, _URL, {}, FetchContext(fast_path=True))

        assert len(calls) == 1
        assert result.route == "impersonate"
        assert result.status == "success"
        assert [(a.rung, a.skipped_reason) for a in result.rung_attempts] == [("impersonate", "")]


class TestProviderLevelRungMarkers:
    """Each new rung, driven through ``resolution_source_provider`` end to end.

    The `_fetch_one`-seam tests above pin a rung's behaviour; none of them exercise the provider,
    which is the only place the `RESOLUTION_SOURCE_FETCH` line's `route=`, the
    `RESOLUTION_SOURCE_ESCALATION` line and the `details["counts"]` the rung's keys ride are all
    produced together — and the only place the one-shared-`QuestionRungBudget`-per-question
    wiring exists (dropping `shared=` in the provider would keep every seam test green while
    turning the per-question Wayback cap into a per-URL one). One test per rung, in the style of
    the meta_refresh / pdf_local pair in ``test_resolution_source_fetch.py``.

    Fast-path declines (``TestFastPath``) and the Wayback capture-timestamp shape
    (``TestWaybackRung``) are already covered; these do not repeat them.
    """

    async def test_the_rendered_rung_names_its_route_through_the_provider(self, monkeypatch, caplog):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setattr(
            resolution_source,
            "render_page",
            _fake_render(_rendered_document(f"<h1>Polling average</h1><p>{_RENDERED_PROSE}</p>"), []),
        )
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria=f"Resolves per {_URL}")

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            f"RESOLUTION_SOURCE_FETCH: question=999 url={_URL} status=ok http=200 embeds=none route=rendered"
        ]
        assert re.fullmatch(
            rf"RESOLUTION_SOURCE_ESCALATION: question=999 url={re.escape(_URL)} "
            r"from_status=js_wall rung=rendered outcome=success wall_s=\d+\.\d\d",
            next(m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_ESCALATION:")),
        )
        counts = pop_provider_detail(q.id_of_question, "resolution_source")["counts"]
        assert counts["rendered_attempts"] == 1

    async def test_the_derived_api_rung_names_its_route_through_the_provider(self, monkeypatch, caplog):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        harvested = RenderedPage(
            url=_URL,
            content_type="text/html",
            html=_JS_SHELL.decode(),
            json_responses=(HarvestedJson(url=_FEED_URL, body=b'{"series":[{"v":1}]}'),),
        )
        monkeypatch.setattr(resolution_source, "render_page", _fake_render(harvested, []))
        session = FakeSession({_URL: FakeResponse(200, body=_JS_SHELL, content_type="text/html")})
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria=f"Resolves per {_URL}")

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.resolution_source"):
            section = await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            f"RESOLUTION_SOURCE_FETCH: question=999 url={_URL} status=ok http=200 embeds=none route=derived_api"
        ]
        # Two escalation lines: the render fired (found the endpoint) and the derived-feed read
        # produced the text, and only the second is the route. Matched whole, like the sibling
        # rungs' single-line assertions, because this is the one URL that emits two lines and
        # `question`, `url`, `from_status` and `wall_s` went unpinned on both of them.
        escalations = [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_ESCALATION:")]
        assert len(escalations) == 2
        assert re.fullmatch(
            rf"RESOLUTION_SOURCE_ESCALATION: question=999 url={re.escape(_URL)} "
            r"from_status=js_wall rung=rendered outcome=js_wall wall_s=\d+\.\d\d",
            escalations[0],
        )
        assert re.fullmatch(
            rf"RESOLUTION_SOURCE_ESCALATION: question=999 url={re.escape(_URL)} "
            r"from_status=js_wall rung=derived_api outcome=success wall_s=\d+\.\d\d",
            escalations[1],
        )
        counts = pop_provider_detail(q.id_of_question, "resolution_source")["counts"]
        assert counts["derived_api_reads"] == 1
        assert counts["rendered_attempts"] == 1
        assert ROUTE_CAVEATS["derived_api"] in section

    async def test_the_wayback_rung_names_its_route_through_the_provider(self, monkeypatch, caplog):
        """One cited URL through the provider: ``route=wayback`` on the FETCH line, the escalation
        line's ``from_status`` / ``outcome`` / ``wall_s``, ``counts["wayback_attempts"]``, and the
        archived-copy caveat in the rendered section.

        It says nothing about the per-question cap, which one URL cannot: the shared
        ``QuestionRungBudget`` the provider builds is pinned by
        ``test_two_same_host_urls_fetched_concurrently_share_one_launch``
        (``test_resolution_source_derived_api_rung.py``), whose single-launch assertion rides that
        budget's browser gate and goes red when ``shared=`` is dropped, and the cap arithmetic
        itself by ``test_the_per_question_cap_binds_across_cited_urls``
        (``test_resolution_source_wayback_rung.py``) and its paid-rung twin
        ``test_the_per_question_paid_read_cap_binds_across_cited_urls``
        (``test_resolution_source_url_context_rung.py``)."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setattr(resolution_source, "_WAYBACK_TRIGGER_STATUSES", _WAYBACK_TRIGGER_STATUSES)
        # Real now, so the capture stays inside the age bound regardless of when the suite runs;
        # the provider builds the request URL off its own `datetime.now(UTC)`, same year, so the
        # prefix-keyed handler matches.
        now = datetime.now(UTC)
        captured = now - timedelta(days=2)
        snapshot = _snapshot_url(_URL, captured=captured)
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"denied", content_type="text/html"),
                wayback_snapshot_url(_URL, now=now): FakeResponse(302, headers={"Location": snapshot}),
                snapshot: FakeResponse(200, body=_prose_page(_RENDERED_PROSE), content_type="text/html"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria=f"Resolves per {_URL}")

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.resolution_source"):
            section = await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            f"RESOLUTION_SOURCE_FETCH: question=999 url={_URL} status=ok http=200 embeds=none route=wayback"
        ]
        assert re.fullmatch(
            rf"RESOLUTION_SOURCE_ESCALATION: question=999 url={re.escape(_URL)} "
            r"from_status=blocked rung=wayback outcome=success wall_s=\d+\.\d\d",
            next(m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_ESCALATION:")),
        )
        counts = pop_provider_detail(q.id_of_question, "resolution_source")["counts"]
        assert counts["wayback_attempts"] == 1
        assert ROUTE_CAVEATS["wayback"] in section

    async def test_a_withheld_capture_keeps_the_direct_fetchs_diagnostics_on_the_fetch_line(self, monkeypatch, caplog):
        """The Wayback withhold REPLACES the direct result on the FETCH line, and it used to carry
        the archive's `http=200` with no `failure_class` or `server`: the blocked population the
        ladder was justified on, undercounted by every withheld capture. A rung verdict that served
        nothing keeps the cited host's own status and diagnostics; only the success path reports
        the snapshot's status, because those bytes are the archive's."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setattr(resolution_source, "_WAYBACK_TRIGGER_STATUSES", _WAYBACK_TRIGGER_STATUSES)
        now = datetime.now(UTC)
        snapshot = _snapshot_url(_URL, captured=now - timedelta(days=400))
        session = FakeSession(
            {
                _URL: FakeResponse(403, body=b"denied", headers={"Server": "AkamaiGHost"}, content_type="text/html"),
                wayback_snapshot_url(_URL, now=now): FakeResponse(302, headers={"Location": snapshot}),
                snapshot: FakeResponse(200, body=_prose_page(_RENDERED_PROSE), content_type="text/html"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria=f"Resolves per {_URL}")

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            f"RESOLUTION_SOURCE_FETCH: question=999 url={_URL} status=stale_data http=403 embeds=none "
            "route=wayback failure_class=http_403 server=akamaighost"
        ]
        assert re.fullmatch(
            rf"RESOLUTION_SOURCE_ESCALATION: question=999 url={re.escape(_URL)} "
            r"from_status=blocked rung=wayback outcome=stale_data wall_s=\d+\.\d\d",
            next(m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_ESCALATION:")),
        )

    async def test_the_url_context_rung_names_its_route_through_the_provider(self, monkeypatch, caplog):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        reader, _calls = paid_reader()
        arm_paid_rung(monkeypatch, reader)
        session = refused_page_with_robots()
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria=f"Resolves per {_URL}")

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.resolution_source"):
            section = await resolution_source_provider(is_benchmarking=False)(q)

        # http=403: the paid read keeps the direct fetch's HTTP status (the host refused us),
        # since a model-mediated read has no status of its own.
        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            f"RESOLUTION_SOURCE_FETCH: question=999 url={_URL} status=ok http=403 embeds=none route=url_context"
        ]
        assert re.fullmatch(
            rf"RESOLUTION_SOURCE_ESCALATION: question=999 url={re.escape(_URL)} "
            r"from_status=blocked rung=url_context outcome=success wall_s=\d+\.\d\d",
            next(m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_ESCALATION:")),
        )
        counts = pop_provider_detail(q.id_of_question, "resolution_source")["counts"]
        assert counts["url_context_reads"] == 1
        assert ROUTE_CAVEATS["url_context"] in section
