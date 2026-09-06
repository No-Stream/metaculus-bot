"""The Wayback rung: an archived capture of a page our own address could not reach."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pytest

from metaculus_bot.research import resolution_presentation, resolution_source
from metaculus_bot.research.resolution_fetch_result import ROUTE_CAVEATS
from metaculus_bot.research.resolution_presentation import format_resolution_sections
from metaculus_bot.research.resolution_source import (
    _WAYBACK_TRIGGER_STATUSES,
    FetchContext,
    QuestionRungBudget,
    _fetch_one,
    _rung_counts,
)
from metaculus_bot.research.wayback import wayback_snapshot_url
from tests.resolution_source_fakes import (
    _JS_SHELL,
    _RENDERED_PROSE,
    _ROBOTS_URL,
    _URL,
    ROBOTS_ALLOW_ALL,
    FakeResponse,
    FakeSession,
    _fake_render,
    _prose_page,
    _snapshot_url,
    arm_paid_rung,
    paid_reader,
)
from tests.test_document_text import build_text_pdf


class TestWaybackRung:
    """The archive is the one free route whose EGRESS IS NOT OURS, which is why a host that
    refuses our address earns it — and why a JavaScript wall never does."""

    # Deliberately NOT the current year. `wayback_snapshot_url` interpolates only the year, and
    # every handler here is keyed by the URL that function returns for this instant, so a fixture
    # dated in the running year would key the archive handler on the same string a wall-clock read
    # produces: dropping the threaded `now=` would then keep this whole module green. Dated in a
    # past year, a rung that reads its own clock asks for a URL no handler serves and the fake
    # session says so. The clock has to be threaded rather than read inside the rung because the
    # request and the age disclosure rendered off what comes back must be one instant.
    _NOW = datetime(2025, 9, 4, tzinfo=UTC)

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
            "[Archived copy from the Wayback Machine, captured 2025-08-29, 6 days before this "
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
        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader)
        session = self._session(
            page=FakeResponse(403, body=b"", content_type="text/html"),
            captured=self._NOW - timedelta(days=400),
            extra={_ROBOTS_URL: FakeResponse(200, body=ROBOTS_ALLOW_ALL, content_type="text/plain")},
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
        assert result.text.startswith("[Archived copy from the Wayback Machine, captured 2025-09-01, 3 days before")
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

    async def test_the_archived_lead_is_truncated_when_it_alone_exceeds_the_cap(self, monkeypatch):
        """The pathological case the shared cap helper (`_lead_then_capped_body`) fixes: at a cap
        below the age-disclosure lead's own length, the earlier bare-lead return busted the
        per-URL bound `_budgeted_success_sections` relies on. The lead is now truncated to fit."""
        cap = 60
        monkeypatch.setattr(resolution_presentation, "RESOLUTION_SOURCE_PER_URL_MAX_CHARS", cap)
        session = self._session(
            page=FakeResponse(403, body=b"denied", content_type="text/html"),
            captured=self._NOW - timedelta(days=3),
        )

        result = await _fetch_one(session, _URL, {}, self._ctx())

        assert result.status == "success"
        assert result.route == "wayback"
        assert len(result.text) <= cap
