"""The prediction-market seam's transport layer: the bounded GET, and the catalogue cache.

Two seams the venue-level unit tests cannot reach:

- the one bounded-GET helper every venue path sits on, exercised THROUGH a venue so the
  retry/`None`-vs-`[]` degradation contract is under test rather than the parser,
- the 6h catalogue cache, which the seam owns outright (the `venues` package is cache-free) and whose
  failure modes are all about what must NOT be pinned for the TTL.

Fakes, payload fixtures and the `handlers()` baseline live in `tests/market_retrieval_fakes.py`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from metaculus_bot.research import prediction_market as pmp
from metaculus_bot.research.http_fetch import ERROR_SNIPPET_BYTES
from metaculus_bot.research.market_retrieval import venues
from metaculus_bot.research.market_retrieval.http import MAX_RESPONSE_BYTES
from metaculus_bot.research.prediction_market import _reset_session_caches
from metaculus_bot.research.provider_health import recorded_observations, reset_provider_health
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


class _SuspendingResponse(FakeResponse):
    """A `FakeResponse` that yields to the event loop before it serves its body.

    The single-flight tests need two callers inside one pull at once, and nothing in the fake
    transport suspends on its own: `FakeResponse.json` and the streamed `iter_chunked` never await
    a real future, so a gathered caller can run a whole unpaginated fetch without ever handing
    control back. Kalshi's own inter-page `asyncio.sleep` gives that up for free once a pull spans
    two pages, which is why only the one-GET PredictIt pull needs this.
    """

    async def __aenter__(self) -> _SuspendingResponse:
        await asyncio.sleep(0)
        return self


# Page one of a two-page catalogue: a real row plus an open cursor, so the pull paginates — and
# sleeps between pages, which is what lets a second caller reach the guard mid-pull.
_KALSHI_PAGE_ONE = {
    "events": [{"event_ticker": "KXPAGEONE", "title": "Page one event", "markets": []}],
    "cursor": "next",
}


class TestCatalogueSingleFlight:
    """The in-flight guard on the two catalogue caches.

    The 6h TTL check cannot see a pull that has STARTED and not finished, so with one pull per
    question a run's concurrent questions opened one whole-catalogue pagination each against the
    same venue — 60-75 pages apiece for Kalshi — and the venue rate-limited most of them. Every
    test here is about what N concurrent callers make the venue see, and about the outcome they
    share when the one pull they all ride goes wrong.
    """

    @pytest.mark.asyncio
    async def test_concurrent_questions_share_one_kalshi_pagination(self, kalshi_events_payload):
        """Four questions, one pagination: two page fetches total, and all four get BOTH pages.

        The per-question catalogue-size observation is asserted alongside, because sharing the
        pull must not cost the waiters their own provider-health signal — that recording is what
        Signal C reads, and it is keyed by qid.
        """
        session = FakeSession(
            {_KALSHI_EVENTS_URL: [FakeResponse(200, _KALSHI_PAGE_ONE), FakeResponse(200, kalshi_events_payload)]}
        )
        reset_provider_health()

        results = await asyncio.gather(*(pmp._kalshi_catalogue(session, qid=qid) for qid in (11, 22, 33, 44)))

        assert session._call_counts[_KALSHI_EVENTS_URL] == 2, "one shared pagination, not one per question"
        assert [[event["event_ticker"] for event in events] for events, _token in results] == [
            ["KXPAGEONE", "KXSPACEX-26", "KXOTHER-1"]
        ] * 4
        assert [token for _events, token in results] == ["ok(3)"] * 4
        _venue_observations, catalogues = recorded_observations()
        assert sorted(obs.qid for obs in catalogues) == [11, 22, 33, 44]
        assert all(obs.fetch_ok for obs in catalogues)

    @pytest.mark.asyncio
    async def test_a_rate_limited_leader_hands_every_waiter_the_same_partial(self):
        """A 429 mid-pagination is SHARED rather than retried by each waiter.

        Re-asking a rate limiter from three more questions is a second violation, not a retry
        (`_kalshi_fetch_events_page` refuses to retry a 429 for exactly that reason), so the
        waiters get what the leader got: the partial catalogue plus its error token. One lost pull
        also bumps the catalogue-failure counter ONCE, which is the honest count of pulls lost —
        the pre-guard behaviour reported the same outage once per rate-limited question.
        """
        session = FakeSession(
            {_KALSHI_EVENTS_URL: [FakeResponse(200, _KALSHI_PAGE_ONE), FakeResponse(429, text="slow down")]}
        )

        results = await asyncio.gather(*(pmp._kalshi_catalogue(session, qid=None) for _ in range(4)))

        assert session._call_counts[_KALSHI_EVENTS_URL] == 2, "the waiters must not re-ask the rate limiter"
        assert [token for _events, token in results] == ["error(http_429)"] * 4
        assert [[event["event_ticker"] for event in events] for events, _token in results] == [["KXPAGEONE"]] * 4
        assert "events" not in pmp._KALSHI_CACHE, "a truncated catalogue must not be pinned for the TTL"
        assert pmp.kalshi_catalogue_fetch_failures() == 1, "one lost pull is one failure, not one per waiter"

    @pytest.mark.asyncio
    async def test_a_leader_whose_pull_raises_does_not_strand_its_waiters(self):
        """A leader that produces NO result — its pull raised, or its own caller's deadline
        cancelled it — is the one case the waiters cannot share.

        They go back through the guard instead: the first to wake leads a fresh pull and the rest
        wait on that one, so an abandoned pull costs one more pull rather than one per waiter, and
        nobody is left awaiting a future that will never resolve.
        """
        attempts: list[int] = []

        async def flaky_prefetch(_session: Any, **_kwargs: Any) -> venues.CataloguePull:
            attempts.append(1)
            await asyncio.sleep(0)  # let the other callers queue behind this pull first
            if len(attempts) == 1:
                raise RuntimeError("connector died mid-pull")
            return venues.CataloguePull(
                events=[{"event_ticker": "KXLATE", "title": "Late event", "markets": []}],
                token="",
                tally=pmp._FetchTally(ok=1),
                complete=True,
            )

        with patch.object(venues, "kalshi_prefetch_events", flaky_prefetch):
            outcomes = await asyncio.gather(
                *(pmp._kalshi_catalogue(MagicMock(), qid=None) for _ in range(3)), return_exceptions=True
            )

        assert isinstance(outcomes[0], RuntimeError), "the leader's own caller still sees the failure"
        # A waiter that inherited the failure lands its exception in the token slot, so it fails
        # this same assertion rather than needing a second one.
        waiter_tokens = [outcome if isinstance(outcome, BaseException) else outcome[1] for outcome in outcomes[1:]]
        assert waiter_tokens == ["ok(1)"] * 2
        assert len(attempts) == 2, "one extra pull for the whole set of waiters"
        assert pmp._KALSHI_CATALOGUE_IN_FLIGHT == {}, "a settled pull leaves nothing behind to await"

    @pytest.mark.asyncio
    async def test_concurrent_questions_share_one_predictit_fetch(self, predictit_payload):
        """The PredictIt dump is one unpaginated GET, and four questions make exactly one of it."""
        session = FakeSession({_PREDICTIT_URL: _SuspendingResponse(200, predictit_payload)})

        results = await asyncio.gather(*(pmp._predictit_universe(session, qid=None) for _ in range(4)))

        assert session._call_counts[_PREDICTIT_URL] == 1, "one shared dump, not one per question"
        assert [tally.ok for _markets, tally in results] == [1] * 4
        assert all(markets == results[0][0] for markets, _tally in results)

    @pytest.mark.asyncio
    async def test_reset_clears_the_in_flight_pull_maps(self):
        """The guards are reset with the caches they guard. A future left here from an earlier
        test's event loop would be awaited by the next test and never resolve."""
        loop = asyncio.get_running_loop()
        pmp._KALSHI_CATALOGUE_IN_FLIGHT["events"] = loop.create_future()
        pmp._PREDICTIT_UNIVERSE_IN_FLIGHT["markets"] = loop.create_future()

        _reset_session_caches()

        assert pmp._KALSHI_CATALOGUE_IN_FLIGHT == {}
        assert pmp._PREDICTIT_UNIVERSE_IN_FLIGHT == {}
        await asyncio.sleep(0)  # cooperative yield for flake8-async ASYNC910
