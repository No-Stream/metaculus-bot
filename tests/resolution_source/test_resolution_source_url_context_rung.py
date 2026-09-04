"""The paid url_context rung: every gate before a cent is spent, and its per-question caps."""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from metaculus_bot.research import resolution_source
from metaculus_bot.research.resolution_source import (
    FetchContext,
    QuestionRungBudget,
    _fetch_one,
    _rung_counts,
)
from metaculus_bot.research.robots_policy import reset_robots_cache
from tests.resolution_source_fakes import (
    _URL,
    FakeResponse,
    FakeSession,
)
from tests.test_document_text import build_text_pdf


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

    async def test_the_per_question_paid_read_cap_binds_across_cited_urls(self, monkeypatch):
        """The paid rung's analogue of the Wayback per-question cap: a question citing several
        dead sources pays at most RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS times inside one
        provider wall, and the read the cap declines records a `url_context_cap` skip."""
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS", 2)
        reader, calls = self._reader()
        self._arm(monkeypatch, reader)
        shared = QuestionRungBudget()
        urls = [f"https://dead{index}.example.com/page" for index in range(3)]
        handlers: dict[str, object] = {}
        for url in urls:
            handlers[url] = FakeResponse(403, body=b"denied", content_type="text/html")
            handlers[f"https://dead{urls.index(url)}.example.com/robots.txt"] = FakeResponse(
                200, body=b"User-agent: *\nAllow: /\n", content_type="text/plain"
            )
        session = FakeSession(handlers)

        results = [await _fetch_one(session, url, {}, FetchContext(query="ask", shared=shared)) for url in urls]

        assert [r.route for r in results] == ["url_context", "url_context", "direct"]
        assert len(calls) == 2, "the third cited URL paid for a read past the per-question cap"
        assert results[2].status == "blocked"
        assert [(a.rung, a.skipped_reason) for a in results[2].rung_attempts] == [("url_context", "url_context_cap")]
        assert _rung_counts(results)["url_context_reads"] == 2
        assert _rung_counts(results)["url_context_cap_skips"] == 1

    async def test_the_read_lead_is_truncated_when_it_alone_exceeds_the_cap(self, monkeypatch):
        """The same pathological case as the archive lead: at a cap below the mandatory
        model-mediated disclosure's length, the earlier bare-lead return exceeded the per-URL
        bound. The shared cap helper truncates the lead instead."""
        cap = 60
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_PER_URL_MAX_CHARS", cap)
        reader, _calls = self._reader(text="The page reports 12 major work stoppages. " * 5)
        self._arm(monkeypatch, reader)

        result = await _fetch_one(self._session(), _URL, {}, FetchContext(query="ask"))

        assert result.status == "success"
        assert result.route == "url_context"
        assert len(result.text) <= cap
