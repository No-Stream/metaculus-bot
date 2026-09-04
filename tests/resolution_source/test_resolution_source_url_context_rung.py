"""The paid url_context rung: every gate before a cent is spent, and its per-question caps."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import UTC, datetime

import pytest

from metaculus_bot.research import resolution_source
from metaculus_bot.research.resolution_source import (
    FetchContext,
    QuestionRungBudget,
    _fetch_one,
    _rung_counts,
    format_resolution_sections,
)
from tests.resolution_source_fakes import (
    _ROBOTS_URL,
    _URL,
    FakeResponse,
    FakeSession,
    arm_paid_rung,
    paid_reader,
    refused_page_with_robots,
)
from tests.test_document_text import build_text_pdf


class TestUrlContextRung:
    """The LAST rung and the only paid one: every gate is checked before a cent is spent, and
    zero successful retrievals discards the text rather than rendering recall as evidence."""

    async def test_the_flag_off_never_calls_the_reader(self, monkeypatch):
        """Ships OFF and stays off in every workflow: it is the only rung that spends money and
        the only one whose product is a model's answer rather than the host's bytes."""
        reader, calls = paid_reader()
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setattr(resolution_source, "run_url_context_read", reader)
        monkeypatch.delenv("RESOLUTION_SOURCE_URL_CONTEXT_ENABLED", raising=False)

        result = await _fetch_one(refused_page_with_robots(), _URL, {})

        assert result.status == "blocked"
        assert result.route == "direct"
        assert calls == []

    async def test_an_allowed_host_is_read_and_disclosed(self, monkeypatch):
        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader)

        result = await _fetch_one(refused_page_with_robots(), _URL, {}, FetchContext(query="How many stoppages?"))

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
        reader, _calls = paid_reader(retrievals=0, statuses=["URL_RETRIEVAL_STATUS_ERROR"])
        arm_paid_rung(monkeypatch, reader)

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.resolution_source"):
            result = await _fetch_one(refused_page_with_robots(), _URL, {}, FetchContext(query="ask"))

        assert result.status == "ungrounded"
        assert result.text == ""
        # A verdict that served nothing keeps the cited host's status and diagnostics, so the
        # FETCH line it replaces the direct result on still says which host refused us.
        assert result.http_status == 403
        assert result.failure_class == "http_403"
        # Deliberately unregistered while the flag is off everywhere, so the spelling is pinned
        # HERE: parallel to AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED, the v2 reader's twin, so the
        # spec that eventually registers it matches the lines already in the logs.
        assert (
            "RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url=https://tracker.example.com/senate "
            "statuses=URL_RETRIEVAL_STATUS_ERROR"
        ) in caplog.messages

    async def test_an_answer_that_does_not_address_the_ask_is_withheld(self, monkeypatch, caplog):
        """The prompt asks the model to open with NOT_ADDRESSED when the retrieved page does not
        discuss the ask, so that answer is the DESIGNED non-answer. Published as `success` it
        rendered under the url_context lead and the primary-grading-evidence caption, counted the
        provider as succeeded and suppressed the all-failed notice for sibling URLs: prose
        standing in for an absent section, the shape the PDF digest closes with
        `no_matching_passage`. The read was still paid for, so it stays on the record as the
        rung's own verdict."""
        reader, _calls = paid_reader(text="NOT_ADDRESSED. The page lists office hours only.")
        arm_paid_rung(monkeypatch, reader)

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.resolution_source"):
            result = await _fetch_one(refused_page_with_robots(), _URL, {}, FetchContext(query="ask"))

        assert result.status == "no_resolving_content"
        assert result.status_reason == "not_addressed"
        assert result.route == "url_context"
        assert result.text == ""
        # The cited host's status and diagnostics, as on every verdict that served nothing.
        assert result.http_status == 403
        assert result.failure_class == "http_403"
        assert _rung_counts([result])["url_context_reads"] == 1
        assert [(a.rung, a.outcome) for a in result.rung_attempts] == [("url_context", "no_resolving_content")]
        # Unregistered while the flag is off everywhere, like its two URLCONTEXT siblings.
        assert (
            "RESOLUTION_SOURCE_URLCONTEXT_NOT_ADDRESSED: url=https://tracker.example.com/senate host=tracker.example.com"
        ) in caplog.messages
        rendered = format_resolution_sections([result], datetime(2026, 9, 4, tzinfo=UTC))
        assert "Read via Gemini url_context" not in rendered
        assert "office hours" not in rendered
        assert "tracker.example.com: no_resolving_content" in rendered

    async def test_the_ungrounded_line_says_none_when_the_sdk_reported_no_statuses(self, monkeypatch, caplog):
        def _read(url, ask, **kwargs):
            del url, ask, kwargs
            return ("", 0, [])

        arm_paid_rung(monkeypatch, _read)

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.resolution_source"):
            await _fetch_one(refused_page_with_robots(), _URL, {}, FetchContext(query="ask"))

        assert (
            "RESOLUTION_SOURCE_URLCONTEXT_UNGROUNDED_SUPPRESSED: url=https://tracker.example.com/senate statuses=none"
        ) in caplog.messages

    async def test_a_google_extended_disallowing_host_is_skipped_before_paying(self, monkeypatch, caplog):
        """Proven live 2026-09-03: a host that disallows the token refuses the fetch server-side,
        so the read would be spend with a known-zero return."""
        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader)
        session = refused_page_with_robots(robots=b"User-agent: Google-Extended\nDisallow: /\n")

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
        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader)
        session = refused_page_with_robots(robots=b"User-agent: *\nDisallow: /\n")

        result = await _fetch_one(session, _URL, {}, FetchContext(query="ask"))

        assert result.status == "success"
        assert len(calls) == 1

    async def test_a_robots_file_served_as_a_pdf_keeps_its_read_off_the_pages_record(self, monkeypatch):
        """The pre-check's GET is a request made on the CITED URL's behalf, so what it reads must
        not be booked against the page.

        `_fetch_robots_txt` goes through `_fetch_direct`, which CLASSIFIES: a host answering
        `/robots.txt` with `application/pdf` sends it down the local document read, which stamps a
        `pdf_local` attempt and a `pdf_documents_read` onto whatever context it was handed. On the
        PAGE's context that is a document read attributed to a page whose own fetch was a 403, and
        `route` would move to `pdf_local` on a page no document was ever cited from. The child
        context `_aux_ctx` builds is what keeps the page's record to the rung it actually earned.
        An unreadable policy still proceeds to pay, which is the only direction this pre-check is
        allowed to fail in, so the paid read below is what a correct run reaches."""
        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader)
        session = refused_page_with_robots(
            extra={
                _ROBOTS_URL: FakeResponse(
                    200,
                    body=build_text_pdf([["User-agent: Google-Extended", "Disallow: /"]]),
                    content_type="application/pdf",
                )
            }
        )

        result = await _fetch_one(session, _URL, {}, FetchContext(query="ask"))

        # The pre-check really ran and really read the document: without this the assertions
        # below would also hold for a fixture that never fetched robots.txt at all.
        assert _ROBOTS_URL in session.requested
        assert len(calls) == 1
        assert result.route == "url_context"
        assert [(a.rung, a.skipped_reason) for a in result.rung_attempts] == [("url_context", "")]
        assert _rung_counts([result])["pdf_documents_read"] == 0

    async def test_a_document_we_read_in_full_is_never_sent_to_the_paid_reader(self, monkeypatch):
        """`no_matching_passage` is inside the trigger STATUSES and outside the trigger
        POPULATION. We hold the whole document's text and its outline, so a model re-reading the
        same PDF cannot find a passage BM25 could not — it would be spend with a known-zero
        return, which is the same test every other exclusion in that set passes. Reason-scoped,
        because `thin_page` and `embed_shell` are pages our client genuinely could not read."""
        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader)
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
        reader, calls = paid_reader()
        monkeypatch.setenv("RESOLUTION_SOURCE_URL_CONTEXT_ENABLED", "true")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setattr(resolution_source, "run_url_context_read", reader)

        result = await _fetch_one(refused_page_with_robots(), _URL, {}, FetchContext(query="ask"))

        assert result.status == "blocked"
        assert calls == []
        assert [(a.rung, a.skipped_reason) for a in result.rung_attempts] == [("url_context", "no_api_key")]
        # Its own count: without it "flag on, key missing" is byte-identical in the archive to
        # "flag off", and the moment that matters is the paid flag's rollout.
        assert _rung_counts([result])["url_context_no_api_key_skips"] == 1

    async def test_the_rung_is_skipped_below_its_floor(self, monkeypatch):
        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader)
        monkeypatch.setattr(FetchContext, "rung_budget_s", lambda self: 9.0)

        result = await _fetch_one(refused_page_with_robots(), _URL, {}, FetchContext(query="ask"))

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
        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader)
        session = refused_page_with_robots()
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
        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader)
        session = refused_page_with_robots()
        monkeypatch.setattr(
            FetchContext,
            "rung_budget_s",
            self._budget_that_drops_after_the_robots_read(session, before=20.0, after=5.0),
        )

        result = await _fetch_one(session, _URL, {}, FetchContext(query="ask"))

        assert result.status == "blocked"
        assert calls == []
        assert _ROBOTS_URL in session.requested
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

        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader)
        monkeypatch.setattr(resolution_source, "ROBOTS_FETCH_TIMEOUT_S", 0.05)
        session = refused_page_with_robots(extra={_ROBOTS_URL: _HangingResponse(200, content_type="text/plain")})

        started = time.monotonic()
        result = await _fetch_one(session, _URL, {}, FetchContext(query="ask"))

        assert time.monotonic() - started < 2.0
        assert result.status == "success"
        assert len(calls) == 1

    @pytest.mark.parametrize("status", [404, 410])
    async def test_a_missing_page_is_never_sent_to_the_reader(self, monkeypatch, status):
        """A 404 has no page to read, so the spend would buy nothing."""
        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader)
        session = FakeSession({_URL: FakeResponse(status, body=b"", content_type="text/html")})

        result = await _fetch_one(session, _URL, {}, FetchContext(query="ask"))

        assert result.status == "not_found"
        assert calls == []

    async def test_a_reader_failure_leaves_the_direct_status(self, monkeypatch):
        def _boom(*_args, **_kwargs):
            raise RuntimeError("reader exploded")

        arm_paid_rung(monkeypatch, _boom)

        result = await _fetch_one(refused_page_with_robots(), _URL, {}, FetchContext(query="ask"))

        assert result.status == "blocked"
        assert result.route == "url_context"

    async def test_the_per_question_paid_read_cap_binds_across_cited_urls(self, monkeypatch):
        """The paid rung's analogue of the Wayback per-question cap: a question citing several
        dead sources pays at most RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS times inside one
        provider wall, and the read the cap declines records a `url_context_cap` skip."""
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_URL_CONTEXT_MAX_ATTEMPTS", 2)
        reader, calls = paid_reader()
        arm_paid_rung(monkeypatch, reader)
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
        reader, _calls = paid_reader(text="The page reports 12 major work stoppages. " * 5)
        arm_paid_rung(monkeypatch, reader)

        result = await _fetch_one(refused_page_with_robots(), _URL, {}, FetchContext(query="ask"))

        assert result.status == "success"
        assert result.route == "url_context"
        assert len(result.text) <= cap
