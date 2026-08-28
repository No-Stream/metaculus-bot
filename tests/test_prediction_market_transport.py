"""The prediction-market seam's transport layer: the bounded GET, and the catalogue cache.

Two seams the venue-level unit tests cannot reach:

- the one bounded-GET helper every venue path sits on, exercised THROUGH a venue so the
  retry/`None`-vs-`[]` degradation contract is under test rather than the parser,
- the 6h catalogue cache, which the seam owns outright (the `venues` package is cache-free) and whose
  failure modes are all about what must NOT be pinned for the TTL.

Fakes, payload fixtures and the `handlers()` baseline live in `tests/market_retrieval_fakes.py`.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from metaculus_bot.research import prediction_market as pmp
from metaculus_bot.research.http_fetch import ERROR_SNIPPET_BYTES
from metaculus_bot.research.market_retrieval import venues
from metaculus_bot.research.market_retrieval.http import MAX_RESPONSE_BYTES
from metaculus_bot.research.prediction_market import _reset_session_caches
from tests import market_retrieval_fakes as _fakes
from tests.market_retrieval_fakes import KALSHI_EVENTS_URL as _KALSHI_EVENTS_URL
from tests.market_retrieval_fakes import MANIFOLD_SEARCH_URL as _MANIFOLD_SEARCH_URL
from tests.market_retrieval_fakes import POLY_URL as _POLY_URL
from tests.market_retrieval_fakes import PREDICTIT_URL as _PREDICTIT_URL
from tests.market_retrieval_fakes import FakeResponse, FakeSession
from tests.market_retrieval_fakes import fetch_snapshot as _fetch
from tests.market_retrieval_fakes import handlers as _handlers
from tests.market_retrieval_fakes import market_llm as _market_llm
from tests.market_retrieval_fakes import no_backoff as _no_backoff
from tests.market_retrieval_fakes import rank_one_per_venue as _rank_one_per_venue

# Shared fixtures have to be bound in THIS module's globals for pytest to find them, and bound by
# ASSIGNMENT rather than `from ... import`: pyflakes reads a same-named fixture parameter as an
# F811 redefinition of an import, so importing them would put a per-signature suppression on every
# test below. `reset_provider_caches` is autouse and resets the seam's caches around each test.
reset_provider_caches = _fakes.reset_provider_caches
mock_question = _fakes.mock_question
kalshi_events_payload = _fakes.kalshi_events_payload
predictit_payload = _fakes.predictit_payload


class TestVenueTransport:
    """The one bounded-GET helper every venue path sits on."""

    @pytest.mark.asyncio
    async def test_rate_limit_retries_with_backoff_then_gives_up(self, monkeypatch):
        """403 is Polymarket's rate-limit shape, so it is retryable — and retry exhaustion
        returns `None`, NOT `[]`. That distinction is the whole degradation contract: `[]`
        publishes as a benign `none` token, so laundering an outage into it is how a dead venue
        reads healthy."""
        _no_backoff(monkeypatch)
        session = FakeSession({_POLY_URL: [FakeResponse(403, text="rate limited"), FakeResponse(403, text="again")]})

        result = await venues.polymarket_search(session, "starship", width=10)

        assert result is None
        assert session._call_counts[_POLY_URL] == venues.POLYMARKET_MAX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_malformed_json_reads_as_a_failed_fetch(self, caplog):
        """A body that will not decode is an upstream failure, not an empty result."""
        session = FakeSession({_MANIFOLD_SEARCH_URL: FakeResponse(200, payload=None)})

        with caplog.at_level(logging.WARNING):
            result = await venues.manifold_search(session, "starship", width=10)

        assert result is None
        assert any("JSON decode failed" in rec.getMessage() for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_a_non_200_body_read_is_bounded(self, caplog):
        """The error-path log reads a bounded snippet off the stream. A regression to
        whole-body `resp.text()` would put a multi-megabyte error page in the log."""
        session = FakeSession({_PREDICTIT_URL: FakeResponse(404, text="x" * (ERROR_SNIPPET_BYTES * 4))})

        with caplog.at_level(logging.WARNING):
            result = await venues.predictit_prefetch(session)

        assert result is None
        logged = "\n".join(rec.getMessage() for rec in caplog.records)
        assert len(logged) < ERROR_SNIPPET_BYTES * 3

    def test_the_response_body_cap_is_shared_and_generous(self):
        """One cap for every buffered venue body; the streaming catalogue pull has its own,
        larger one, because a catalogue page is legitimately megabytes."""
        assert MAX_RESPONSE_BYTES < venues.KALSHI_PAGE_MAX_BYTES


class TestCatalogueCaching:
    @pytest.mark.asyncio
    async def test_a_complete_pull_is_cached_and_the_next_question_does_not_refetch(
        self, mock_question, kalshi_events_payload
    ):
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})
        session = FakeSession(handlers)
        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", _market_llm()),
            patch.object(pmp, "_get_session", lambda: session),
        ):
            await pmp.fetch_market_snapshot(mock_question, timeout=5.0)
            second = MagicMock(id_of_question=999, title=mock_question.title, question_text=mock_question.question_text)
            second.resolution_criteria = mock_question.resolution_criteria
            second.fine_print = ""
            second.unit_of_measure = ""
            await pmp.fetch_market_snapshot(second, timeout=5.0)

        assert session._call_counts[_KALSHI_EVENTS_URL] == 1, "the second question must ride the 6h cache"
        cached_ts, cached_events = pmp._KALSHI_CACHE["events"]
        assert [event["event_ticker"] for event in cached_events] == ["KXSPACEX-26", "KXOTHER-1"]
        assert cached_ts <= time.monotonic()

    @pytest.mark.asyncio
    async def test_a_pull_that_loses_a_LATER_page_caches_nothing(self, mock_question, kalshi_events_payload):
        """The regression that let the cache bug ship: both sibling "does not poison" tests fail
        on page ONE, so a per-page cache warm never fired in them at all.

        Page 1 succeeds with a cursor and page 2 is throttled, so the pull holds a real, truncated
        catalogue. That partial still serves THIS question — the pages it paid for are not thrown
        away — but nothing may be pinned for the 6h TTL, because the read path checks only the TTL
        and every later question would then report `ok(1)` against a true catalogue of two, with
        zero HTTP and no counter bump. Measured live in the Stage-D dry run: questions 2 and 3
        reported `kalshi: ok(8400)` against a true 10,219.
        """
        page_one = {"events": [{"event_ticker": "A", "title": "A", "markets": []}], "cursor": "next"}
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: [
                    FakeResponse(200, page_one),
                    FakeResponse(429, text="slow down"),
                    FakeResponse(200, kalshi_events_payload),
                ]
            }
        )
        second = MagicMock(id_of_question=54321, title=mock_question.title, question_text=mock_question.question_text)
        second.resolution_criteria = mock_question.resolution_criteria
        second.fine_print = ""
        second.unit_of_measure = ""

        session = FakeSession(handlers)
        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", _market_llm(ranking=_rank_one_per_venue)),
            patch.object(pmp, "_get_session", lambda: session),
        ):
            first = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)
            # Checked BETWEEN the two questions: question 2's own pull succeeds and legitimately
            # caches, so a post-hoc check would read its entry and pass either way.
            assert "events" not in pmp._KALSHI_CACHE, "a truncated catalogue must not be pinned for the TTL"
            got_second = await pmp.fetch_market_snapshot(second, timeout=5.0)

        assert first.sources["kalshi"] == "error(http_429)", "question 1 pays for the partial and reports it"
        assert session._call_counts[_KALSHI_EVENTS_URL] == 3, "question 2 must re-issue HTTP, not ride the partial"
        assert got_second.sources["kalshi"].startswith("ok(")
        assert [event["event_ticker"] for event in pmp._KALSHI_CACHE["events"][1]] == ["KXSPACEX-26", "KXOTHER-1"]

    @pytest.mark.asyncio
    async def test_a_failed_pull_does_not_poison_the_next_question(
        self, mock_question, kalshi_events_payload, monkeypatch
    ):
        """Question 1's catalogue 503s, question 2's succeeds. Nothing may be cached for the
        full 6h TTL off the failure, or one transient blip starves every later question in the
        run behind a stale empty catalogue."""
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: [
                    FakeResponse(503, text="service unavailable"),
                    FakeResponse(503, text="service unavailable"),
                    FakeResponse(200, kalshi_events_payload),
                ]
            }
        )
        second = MagicMock(id_of_question=54321, title=mock_question.title, question_text=mock_question.question_text)
        second.resolution_criteria = mock_question.resolution_criteria
        second.fine_print = ""
        second.unit_of_measure = ""

        _no_backoff(monkeypatch)
        session = FakeSession(handlers)
        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", _market_llm(ranking=_rank_one_per_venue)),
            patch.object(pmp, "_get_session", lambda: session),
        ):
            first = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)
            got_second = await pmp.fetch_market_snapshot(second, timeout=5.0)

        assert not first.sources["kalshi"].startswith(("ok", "none"))
        assert "kalshi" not in {m.platform for m in first.matches}
        assert got_second.sources["kalshi"].startswith("ok(")
        assert "kalshi" in {m.platform for m in got_second.matches}

    @pytest.mark.asyncio
    async def test_reset_clears_every_cache_and_both_counters(
        self, mock_question, kalshi_events_payload, predictit_payload
    ):
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
                _PREDICTIT_URL: FakeResponse(200, predictit_payload),
            }
        )
        await _fetch(mock_question, handlers)
        pmp._bump_kalshi_catalogue_failure()
        pmp._bump_source_loss()
        assert pmp._KALSHI_CACHE
        assert pmp._PREDICTIT_CACHE
        assert pmp._SNAPSHOT_CACHE

        _reset_session_caches()

        assert pmp._KALSHI_CACHE == {}
        assert pmp._PREDICTIT_CACHE == {}
        assert pmp._SNAPSHOT_CACHE == {}
        assert pmp.kalshi_catalogue_fetch_failures() == 0
        assert pmp.prediction_market_source_losses() == 0
