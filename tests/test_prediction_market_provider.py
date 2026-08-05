"""Tests for the prediction-market research provider — the SEAM module.

The retrieval machinery itself (venue parsers, pool assembly, the ranker prompt and parser,
the renderer) is unit-tested per module in `tests/test_market_retrieval_*.py`. What this file
covers is everything the seam owns and nothing else:

- the four-stage pipeline wired end to end through a fake aiohttp session,
- the caches, the two degradation counters and their resets,
- the per-source diagnostics dict — SEVEN keys — and every path that turns one into a loss,
- the provider factory's gates (`is_benchmarking`, the env flag) and the `as_of=None` decision,
- the two LLM stages' budgets, their concurrency, and their config pins,
- what the provider-health rules are actually fed.

All HTTP is mocked via fake sessions and both LLM stages are patched, so nothing here opens a
socket or bills a key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from metaculus_bot.constants import (
    MARKET_QUERY_AUTHOR_BACKOFFS,
    MARKET_QUERY_AUTHOR_WALL_TIMEOUT,
    MARKET_RANKER_BACKOFFS,
    MARKET_RANKER_WALL_TIMEOUT,
    PREDICTION_MARKET_TIMEOUT,
)
from metaculus_bot.llm_configs import MARKET_QUERY_AUTHOR_LLM_CONFIG, MARKET_RANKER_LLM_CONFIG
from metaculus_bot.research import prediction_market as pmp
from metaculus_bot.research.http_fetch import ERROR_SNIPPET_BYTES
from metaculus_bot.research.market_retrieval import generation, venues
from metaculus_bot.research.market_retrieval.http import MAX_RESPONSE_BYTES
from metaculus_bot.research.market_retrieval.ranking import DEGRADED_RANKING_MARKER, RENDER_BUDGET
from metaculus_bot.research.market_retrieval.rendering import TABLE_COLUMNS
from metaculus_bot.research.prediction_market import (
    MarketMatch,
    MarketSnapshot,
    _liquidity_label,
    _reset_session_caches,
    format_snapshot_for_research,
    prediction_market_provider,
)
from metaculus_bot.research.provider_diagnostics import _partial_loss_suffix, pop_provider_detail
from metaculus_bot.research.provider_health import (
    VENUE_EXPECTED_LIQUIDITY_FIELDS,
    provider_degradation_findings,
    recorded_observations,
    reset_provider_health,
)

# The seven keys a complete run reports. Pinned as a set here because the diagnostics line, the
# archive and the source-loss counter all read this dict, and a silently-dropped key is a source
# nobody is watching any more.
_ALL_SOURCES = frozenset(
    {"polymarket", "manifold", "kalshi", "predictit", "manifold_detail", "query_author", "ranking"}
)


# ---------------------------------------------------------------------------
# Test isolation: module-level caches bleed across tests when run in certain
# orders (observed 2026-05-12: a full-suite run failed 2 tests that passed
# individually). Reset every test.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_provider_caches():
    _reset_session_caches()
    yield
    _reset_session_caches()


# ---------------------------------------------------------------------------
# Fake aiohttp session
# ---------------------------------------------------------------------------


class _FakeStreamContent:
    """Stub for `resp.content`: streams a payload as JSON bytes via `iter_chunked`.

    The Kalshi catalogue pull stream-parses `resp.content.iter_chunked()`; the other JSON
    endpoints go through `read_json_capped`, which gates on a `.read` attribute FakeResponse
    deliberately lacks, so they still use the `.json()` path. `raw_content` overrides the
    serialized bytes so a test can inject malformed JSON.
    """

    def __init__(self, payload: Any, raw_content: bytes | None = None):
        if raw_content is not None:
            self._data = raw_content
        elif payload is None:
            self._data = b""
        else:
            self._data = json.dumps(payload).encode()

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:  # noqa: ASYNC900
        step = max(1, n)
        for i in range(0, len(self._data), step):
            yield self._data[i : i + step]


class FakeResponse:
    def __init__(self, status: int, payload: Any = None, text: str | None = None, raw_content: bytes | None = None):
        self.status = status
        self._payload = payload
        # Real aiohttp serves `.text()` and `.content` from ONE body, so an error-page stub must
        # stream its text too: the non-200 log path reads a BOUNDED snippet off `.content`, and a
        # stub holding its text outside the stream would hide a regression back to whole-body
        # `resp.text()`.
        if raw_content is None and payload is None and text:
            raw_content = text.encode()
        self.content = _FakeStreamContent(payload, raw_content=raw_content)

    async def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload  # noqa: ASYNC910

    async def text(self) -> str:
        # Faithful to aiohttp: reads the WHOLE body (the memory trap the bounded snippet read
        # exists to avoid).
        chunks = [chunk async for chunk in self.content.iter_chunked(65536)]
        return b"".join(chunks).decode("utf-8", errors="replace")

    async def __aenter__(self) -> "FakeResponse":
        return self  # noqa: ASYNC910

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        return None  # noqa: ASYNC910


# A handler is either a ready-made response or a callable that builds one from the GET params.
_Handler = FakeResponse | Callable[[dict[str, Any]], FakeResponse]
# Per-URL value accepted by FakeSession: a single handler or a list of them. Mapping (not dict) so
# call-site literals like {url: FakeResponse(...)} are accepted (dict is invariant in its values).
_Handlers = Mapping[str, _Handler | list[_Handler]]


class FakeSession:
    """aiohttp.ClientSession replacement dispatching GETs to handlers keyed by URL prefix."""

    def __init__(self, handlers: _Handlers):
        self._handlers: dict[str, list[_Handler]] = {
            k: (v if isinstance(v, list) else [v]) for k, v in handlers.items()
        }
        self._call_counts: dict[str, int] = {k: 0 for k in handlers}
        self.closed = False

    def get(self, url: str, params: dict | None = None, **_kwargs: Any) -> FakeResponse:
        for prefix, handler_list in self._handlers.items():
            if url.startswith(prefix):
                idx = min(self._call_counts[prefix], len(handler_list) - 1)
                self._call_counts[prefix] += 1
                handler = handler_list[idx]
                if callable(handler):
                    return handler(params or {})
                return handler
        raise AssertionError(f"no handler for URL {url}")

    async def close(self) -> None:  # noqa: ASYNC910
        self.closed = True

    async def __aenter__(self) -> "FakeSession":
        return self  # noqa: ASYNC910

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()


_POLY_URL = "https://gamma-api.polymarket.com/public-search"
_MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
_MANIFOLD_DETAIL_URL = "https://api.manifold.markets/v0/market"
_KALSHI_EVENTS_URL = "https://api.elections.kalshi.com/trade-api/v2/events"
_PREDICTIT_URL = "https://www.predictit.org/api/marketdata/all/"


# ---------------------------------------------------------------------------
# The two LLM stages, stubbed at the `build_llm_with_openrouter_fallback` seam
# ---------------------------------------------------------------------------

# The query author's shape. `parse_query_author` strips digits at parse time, so these carry
# none — a numeric synonym would silently vanish and make the test's own intent unreadable.
_AUTHOR_JSON = json.dumps({"synonyms": ["orbital launch", "Starship"], "framings": ["Starship orbit"]})

_RANKER_CUE = "Rank the candidates by EVIDENTIAL VALUE"
_CANDIDATE_LINE_RE = re.compile(r"^\[(\d+)\] \((\w+)\)", re.MULTILINE)


def _rank_one_per_venue(prompt: str) -> str:
    """A ranking array picking the FIRST candidate of every venue block in the prompt.

    Reads the indices back out of the prompt rather than hard-coding them: `parse_ranking` drops
    out-of-range indices, so a hard-coded array would silently render fewer rows than the test
    means on a small pool and nothing at all on an empty one — a test that asserts on rows would
    then pass for the wrong reason.
    """
    first_of: dict[str, int] = {}
    for match in _CANDIDATE_LINE_RE.finditer(prompt):
        first_of.setdefault(match.group(2), int(match.group(1)))
    return json.dumps(
        [{"i": index, "tier": "same_quantity_other_cut", "why": f"{venue} pick"} for venue, index in first_of.items()]
    )


def _market_llm(
    *,
    ranking: str | Callable[[str], str] = "[]",
    author: str = _AUTHOR_JSON,
    configs: list[dict] | None = None,
):
    """A fake LLM covering BOTH market stages, routed on the prompt each one builds.

    One stub for two stages because production has one invocation helper for two stages; a
    per-stage stub would let the two drift in the test suite while they cannot drift in prod.
    `configs` collects the constructor kwargs, which is how the config pins are asserted through
    the real call path rather than by reading the dict.
    """

    class FakeLlm:
        def __init__(self, **kwargs: Any) -> None:
            if configs is not None:
                configs.append(kwargs)

        async def invoke(self, prompt: str) -> str:
            if _RANKER_CUE in prompt:
                return ranking(prompt) if callable(ranking) else ranking
            return author  # noqa: ASYNC910

    return FakeLlm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_question():
    q = MagicMock()
    q.id_of_question = 12345
    q.question_text = "Will SpaceX Starship reach orbit before July 2026?"
    q.title = "Will SpaceX Starship reach orbit before July 2026?"
    q.short_title = "Starship orbit 2026"
    q.resolution_criteria = "Resolves Yes if a SpaceX Starship reaches orbital velocity before 2026-07-01."
    # Spelled out rather than left to MagicMock's auto-attributes: the pipeline concatenates
    # fine_print into the settlement-join text and formats unit_of_measure into the ranker
    # prompt, so an auto-attribute would put a `<MagicMock ...>` repr into both.
    q.fine_print = ""
    q.unit_of_measure = ""
    q.page_url = "https://metaculus.com/q/12345"
    q.scheduled_resolution_time = None
    return q


@pytest.fixture
def polymarket_payload():
    """Realistic Polymarket gamma public-search response shape."""
    return {
        "events": [
            {
                "title": "Will SpaceX Starship reach orbit in 2026?",
                "slug": "spacex-starship-orbit-2026",
                "description": "Resolves yes if Starship reaches orbit in 2026.",
                "endDate": "2026-12-31T23:59:59Z",
                "volume": 125000.0,
                "markets": [
                    {
                        "question": "Will SpaceX Starship reach orbit in 2026?",
                        "outcomePrices": '["0.74", "0.26"]',
                        "volume24hr": 12500.0,
                        "bestBid": 0.73,
                        "bestAsk": 0.75,
                        "volumeNum": 987654.0,
                        "liquidityNum": 45000.0,
                        "openInterest": 60000.0,
                    }
                ],
            },
            {
                "title": "Will SpaceX complete 100 launches in 2026?",
                "slug": "spacex-100-launches-2026",
                "description": "Launch-count threshold market.",
                "endDate": "2026-12-31T23:59:59Z",
                "markets": [
                    {
                        "question": "Will SpaceX complete 100 launches in 2026?",
                        "outcomePrices": '["0.42", "0.58"]',
                        "volume24hr": 3000.0,
                    }
                ],
            },
        ],
        "markets": [],
    }


@pytest.fixture
def manifold_payload():
    return [
        {
            "id": "abc",
            "question": "Will Starship reach orbit before July 2026?",
            "slug": "starship-orbit-july-2026",
            "creatorUsername": "spaceFan",
            "probability": 0.62,
            "volume24Hours": 500.0,
            "volume": 8200.0,
            "totalLiquidity": 1500.0,
            "uniqueBettorCount": 42,
            "closeTime": 1782086400000,  # ms: 2026-06-20
            "textDescription": "YES if SpaceX Starship reaches orbit before July 1 2026.",
            "isResolved": False,
        }
    ]


@pytest.fixture
def kalshi_events_payload():
    """Kalshi `/events?status=open&with_nested_markets=true` shape.

    Field names match the live API (`volume_fp` / `open_interest_fp`, fixed-point STRINGS;
    `liquidity_dollars` deprecated at a constant "0.0000"). An earlier version of this fixture
    carried the bare names `volume` / `open_interest`, which exist nowhere in Kalshi's spec — it
    was written in the same commit as the parser and encoded the same wrong guess, so the pair
    stayed green while every real Kalshi row rendered `no-liquidity-data`. The field-name
    contract is additionally pinned against captured live payloads in
    `test_prediction_market_liquidity_contract.py`, which is where a rename should fail first.

    `settlement_sources` sits at the EVENT level, which is where both the projection and the
    settlement-source join read it; every market-level copy in the live payload is null.
    """
    return {
        "events": [
            {
                "event_ticker": "KXSPACEX-26",
                "title": "Will SpaceX Starship reach orbit in 2026?",
                "sub_title": "Before Dec 31 2026",
                "settlement_sources": [{"name": "SpaceX newsroom", "url": "https://www.spacex.com/updates"}],
                "markets": [
                    {
                        "ticker": "KXSPACEX-26-YES",
                        "title": "Will SpaceX Starship reach orbit in 2026?",
                        "rules_primary": "If Starship achieves orbital velocity in 2026 per SpaceX confirmation.",
                        "yes_bid_dollars": "0.68",
                        "yes_ask_dollars": "0.72",
                        "notional_value_dollars": "1.0000",
                        "volume_24h_fp": "2500.0",
                        "volume_fp": "82000.00",
                        "open_interest_fp": "15000.00",
                        "liquidity_dollars": "0.0000",
                        "close_time": "2026-12-31T23:59:59Z",
                        "status": "active",
                    }
                ],
            },
            {
                "event_ticker": "KXOTHER-1",
                "title": "Unrelated World Cup market",
                "markets": [{"ticker": "KXOTHER-1-YES", "title": "Unrelated", "status": "active"}],
            },
        ],
        "cursor": "",
    }


@pytest.fixture
def predictit_payload():
    """PredictIt `/api/marketdata/all/` shape (verified live). US-politics only, no
    volume/liquidity/OI fields anywhere."""
    return {
        "markets": [
            {
                "id": 7456,
                "name": "Will SpaceX Starship reach orbit before July 2026?",
                "shortName": "Starship orbit 2026",
                "url": "https://www.predictit.org/markets/detail/7456/starship-orbit",
                "contracts": [
                    {
                        "id": 100,
                        "name": "Yes",
                        "shortName": "Yes",
                        "lastTradePrice": 0.58,
                        "bestBuyYesCost": 0.59,
                        "bestBuyNoCost": 0.42,
                        "dateEnd": "2026-07-01T00:00:00",
                        "status": "Open",
                    }
                ],
            },
            {
                "id": 7999,
                "name": "Unrelated 2028 primary market",
                "shortName": "2028 primary",
                "url": "https://www.predictit.org/markets/detail/7999/2028-primary",
                "contracts": [
                    {"id": 200, "name": "Candidate A", "status": "Open", "dateEnd": "NA"},
                    {"id": 201, "name": "Candidate B", "status": "Open", "dateEnd": "N/A"},
                ],
            },
        ]
    }


def _handlers(**overrides: Any) -> dict[str, Any]:
    """The healthy baseline: every venue live, every venue empty.

    Empty SEARCH results and POPULATED catalogues, deliberately. A catalogue that a successful
    fetch returns empty is a different degradation (provider-health Signal C) and stubbing one
    by default would make it fire in every test here.
    """
    handlers: dict[str, Any] = {
        _POLY_URL: FakeResponse(200, {"events": [], "markets": []}),
        _MANIFOLD_SEARCH_URL: FakeResponse(200, []),
        _MANIFOLD_DETAIL_URL: FakeResponse(200, {}),
        _KALSHI_EVENTS_URL: FakeResponse(200, {"events": [_OFF_TOPIC_KALSHI_EVENT], "cursor": ""}),
        _PREDICTIT_URL: FakeResponse(200, {"markets": [_OFF_TOPIC_PREDICTIT_MARKET]}),
    }
    handlers.update(overrides)
    return handlers


_OFF_TOPIC_KALSHI_EVENT = {
    "event_ticker": "KXWORLDCUP-26",
    "title": "Who wins the 2026 World Cup?",
    "markets": [{"ticker": "KXWORLDCUP-26-BRA", "title": "Brazil", "status": "active"}],
}
_OFF_TOPIC_PREDICTIT_MARKET = {
    "id": 9001,
    "name": "Who wins the 2026 World Cup?",
    "url": "https://www.predictit.org/markets/detail/9001",
    "contracts": [{"id": 1, "name": "Brazil", "status": "Open", "lastTradePrice": 0.2}],
}


def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero BOTH copies of the retry backoff, for tests that drive retry-exhaustion paths.

    `venues.py` imports the constant by name, so patching only `http` leaves the catalogue
    pull's own retry sleeping the real 0.5s — which is exactly the kind of half-patch that makes
    a suite mysteriously slow rather than red.
    """
    monkeypatch.setattr("metaculus_bot.research.market_retrieval.http.HTTP_RETRY_BACKOFF_SECS", 0.0)
    monkeypatch.setattr(venues, "HTTP_RETRY_BACKOFF_SECS", 0.0)


async def _fetch(
    question: Any,
    handlers: dict[str, Any],
    *,
    ranking: str | Callable[[str], str] = "[]",
    author: str = _AUTHOR_JSON,
    as_of: datetime | None = None,
    timeout: float = 5.0,
    configs: list[dict] | None = None,
) -> MarketSnapshot:
    """One snapshot through the real pipeline with both LLM stages stubbed."""
    with (
        patch.object(
            pmp, "build_llm_with_openrouter_fallback", _market_llm(ranking=ranking, author=author, configs=configs)
        ),
        patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
    ):
        return await pmp.fetch_market_snapshot(question, timeout=timeout, as_of=as_of)


# ---------------------------------------------------------------------------
# Venue transport (the shared bounded GET, exercised through a venue)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Catalogue caching (the seam owns the TTL; venues.py is cache-free)
# ---------------------------------------------------------------------------


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
    async def test_pagination_warms_the_cache_page_by_page(self, mock_question):
        """A cancelled or partly-lost pull must leave whatever pages completed, so the next
        question picks up where this one stopped."""
        page_one = {"events": [{"event_ticker": "A", "title": "A", "markets": []}], "cursor": "next"}
        page_two = {"events": [{"event_ticker": "B", "title": "B", "markets": []}], "cursor": ""}
        handlers = _handlers(**{_KALSHI_EVENTS_URL: [FakeResponse(200, page_one), FakeResponse(200, page_two)]})

        await _fetch(mock_question, handlers)

        assert [event["event_ticker"] for event in pmp._KALSHI_CACHE["events"][1]] == ["A", "B"]

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
        assert pmp._KALSHI_CACHE and pmp._PREDICTIT_CACHE and pmp._SNAPSHOT_CACHE

        _reset_session_caches()

        assert pmp._KALSHI_CACHE == {}
        assert pmp._PREDICTIT_CACHE == {}
        assert pmp._SNAPSHOT_CACHE == {}
        assert pmp.kalshi_catalogue_fetch_failures() == 0
        assert pmp.prediction_market_source_losses() == 0


# ---------------------------------------------------------------------------
# The two LLM stages: configs, budgets, concurrency
# ---------------------------------------------------------------------------


def _stage_worst(wall: float, backoffs: tuple[float, ...]) -> float:
    return (len(backoffs) + 1) * wall + sum(backoffs)


class TestMarketLlmStages:
    def test_both_configs_pin_a_single_attempt(self):
        """`allowed_tries=1` is required, not decorative: the elapsed-gated `llm_retry` wrapper
        is the SOLE retry layer, and leaving this unpinned inherits forecasting-tools' default of
        2 with an un-gated `random.uniform(5, 10)` tenacity sleep inside the snapshot budget."""
        assert MARKET_RANKER_LLM_CONFIG["allowed_tries"] == 1
        assert MARKET_QUERY_AUTHOR_LLM_CONFIG["allowed_tries"] == 1

    def test_both_configs_leave_headroom_over_the_measured_completions(self):
        """A truncated completion is a TOTAL loss on both stages — the ranking fails open and the
        author's object will not parse — so `max_tokens` sits well above what the bake-off
        measured rather than just above it, and luna's output tokens are cheap enough that the
        headroom costs nothing.

        Measured maxima INCLUDING reasoning tokens, from
        scratch/bakeoff_run_2026-08-03/results/RANKED_ARM_RESULTS.md: ranker 1,042 (mean 589),
        author 588. The shipped budgets are 3,000 and 1,500, i.e. 2.9x and 2.6x the measured
        max. The floor asserted here is 2.5x — enough that a model whose reasoning budget grows
        by half still fits, and low enough that it is not a restatement of the constants.
        """
        assert MARKET_RANKER_LLM_CONFIG["max_tokens"] >= 2.5 * 1042
        assert MARKET_QUERY_AUTHOR_LLM_CONFIG["max_tokens"] >= 2.5 * 588

    def test_each_litellm_timeout_sits_above_its_elapsed_gated_wall(self):
        """The wall is meant to be the binding bound. A litellm timeout below it would fire
        first, and the stage's own budget arithmetic would describe nothing."""
        assert MARKET_RANKER_LLM_CONFIG["timeout"] > MARKET_RANKER_WALL_TIMEOUT
        assert MARKET_QUERY_AUTHOR_LLM_CONFIG["timeout"] > MARKET_QUERY_AUTHOR_WALL_TIMEOUT

    @pytest.mark.asyncio
    async def test_each_stage_passes_its_own_budget_through_the_shared_helper(self, mock_question):
        """The call-site spy, and the more valuable half of the budget coverage.

        A constants-level inequality (tests/test_llm_retry.py) proves the numbers fit; it cannot
        prove the CALLS carry them. This does: one helper, two stages, each with its own wall and
        backoff ladder, and their serial-chain sum inside the snapshot budget.
        """
        calls: list[dict[str, Any]] = []

        async def _spy(config: dict, prompt: str, **kwargs: Any) -> str:
            calls.append({"config": config, **kwargs})
            return "[]" if _RANKER_CUE in prompt else _AUTHOR_JSON

        with (
            patch.object(pmp, "_invoke_market_llm", _spy),
            patch.object(pmp, "_get_session", lambda: FakeSession(_handlers())),
        ):
            await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        by_label = {call["label"]: call for call in calls}
        assert set(by_label) == {"market_query_author", "market_ranker"}

        author = by_label["market_query_author"]
        assert author["config"] is MARKET_QUERY_AUTHOR_LLM_CONFIG
        assert author["wall_timeout"] == MARKET_QUERY_AUTHOR_WALL_TIMEOUT
        assert author["backoffs"] == MARKET_QUERY_AUTHOR_BACKOFFS

        ranker = by_label["market_ranker"]
        assert ranker["config"] is MARKET_RANKER_LLM_CONFIG
        assert ranker["wall_timeout"] == MARKET_RANKER_WALL_TIMEOUT
        assert ranker["backoffs"] == MARKET_RANKER_BACKOFFS

        chain = sum(_stage_worst(call["wall_timeout"], call["backoffs"]) for call in calls)
        assert chain < PREDICTION_MARKET_TIMEOUT

    @pytest.mark.asyncio
    async def test_the_catalogue_pull_and_the_query_author_run_concurrently(self, mock_question):
        """Stage 1a and stage 1b must be GATHERED, not awaited in sequence.

        Nothing else would catch a serial wiring: the chain still fits under 150s today, so a
        serial version would pass every budget test and every behavioural test while spending
        ~20s of wall clock per question for nothing. The prefetches need no queries at all — the
        catalogue IS the venue and the settlement join keys on domains — which is what makes the
        concurrency free.
        """
        order: list[str] = []
        real_prefetch = venues.kalshi_prefetch_events

        async def _slow_prefetch(session: Any, **kwargs: Any) -> Any:
            order.append("catalogue:start")
            await asyncio.sleep(0.02)
            order.append("catalogue:end")
            return await real_prefetch(session, **kwargs)

        async def _slow_llm(config: dict, prompt: str, **_kwargs: Any) -> str:
            order.append("author:start")
            await asyncio.sleep(0.02)
            order.append("author:end")
            return _AUTHOR_JSON

        with (
            patch.object(venues, "kalshi_prefetch_events", _slow_prefetch),
            patch.object(pmp, "_invoke_market_llm", _slow_llm),
            patch.object(pmp, "_get_session", lambda: FakeSession(_handlers())),
        ):
            await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        # Both stages START before either ENDS. Which of the two is scheduled first is a
        # gather-ordering detail worth nothing; overlap is the whole invariant.
        assert set(order[:2]) == {"catalogue:start", "author:start"}, (
            f"stage 1a and 1b were awaited serially rather than gathered: {order}"
        )


# ---------------------------------------------------------------------------
# The pipeline end to end
# ---------------------------------------------------------------------------


class TestFetchMarketSnapshot:
    @pytest.mark.asyncio
    async def test_the_pool_sees_all_four_venues_and_the_render_is_the_models_order(
        self, mock_question, polymarket_payload, manifold_payload, kalshi_events_payload, predictit_payload
    ):
        """Generation is venue-complete; SELECTION is the ranker's, and the render is its order
        verbatim.

        This replaces an older assertion that all four platforms appear in the output, which was
        the venue-fairness assumption in test form — the exact defect this pipeline fixes (a
        round-robin fairness pass evicted 43 of 58 wanted rows). A venue can now legitimately be
        absent from the render, so what is asserted is that the POOL saw every venue and that
        the rows come back in the order the model asked for.
        """
        handlers = _handlers(
            **{
                _POLY_URL: FakeResponse(200, polymarket_payload),
                _MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload),
                _KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
                _PREDICTIT_URL: FakeResponse(200, predictit_payload),
            }
        )
        captured: list[str] = []

        def _rank_reversed(prompt: str) -> str:
            """Pick one candidate per venue, then INVERT the order. A renderer that sorted by
            venue, by index, or by anything else would produce the pool's order instead."""
            captured.append(prompt)
            picks = json.loads(_rank_one_per_venue(prompt))
            return json.dumps(list(reversed(picks)))

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_reversed)

        assert captured, "the ranker was never called"
        pool_venues = {match.group(2) for match in _CANDIDATE_LINE_RE.finditer(captured[0])}
        assert pool_venues == {"kalshi", "polymarket", "manifold", "predictit"}

        expected = [pick["i"] for pick in reversed(json.loads(_rank_one_per_venue(captured[0])))]
        assert [row.rank for row in snapshot.matches] == list(range(len(snapshot.matches)))
        assert [row.platform for row in snapshot.matches] == [
            _CANDIDATE_LINE_RE.findall(captured[0])[index][1] for index in expected
        ]
        assert all(row.relation_tier == "same_quantity_other_cut" for row in snapshot.matches)
        assert all(row.relevance_label for row in snapshot.matches)

    @pytest.mark.asyncio
    async def test_an_empty_ranking_is_a_valid_answer_and_not_a_degradation(self, mock_question, kalshi_events_payload):
        """`[]` means "nothing here bears on the question", which is the whole adaptive-width
        mechanism. Conflating it with a failure would delete that mechanism AND redden CI on a
        correct answer."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        snapshot = await _fetch(mock_question, handlers, ranking="[]")

        assert snapshot.matches == []
        assert snapshot.sources["ranking"] == "ok(0)"
        assert pmp.prediction_market_source_losses() == 0
        assert format_snapshot_for_research(snapshot) == ""

    @pytest.mark.asyncio
    async def test_an_unreadable_ranking_fails_open_to_retrieval_order(self, mock_question, kalshi_events_payload):
        """Output that cannot be read as a JSON array at all renders the pool-order top rows —
        literally the head of what the model was shown, so a fail-open is a truncation of its own
        input rather than a different pipeline. It is marked as such, because the preamble and
        legend both claim evidential order and a silently-wrong ordering claim is worse than a
        visibly degraded one."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        snapshot = await _fetch(mock_question, handlers, ranking="I could not decide.")

        assert snapshot.matches, "a fail-open must still render the deterministic slate"
        assert len(snapshot.matches) <= RENDER_BUDGET
        assert all(row.relation_tier == "" for row in snapshot.matches)
        assert snapshot.sources["ranking"].startswith("error(")
        assert pmp.prediction_market_source_losses() == 1
        assert DEGRADED_RANKING_MARKER in format_snapshot_for_research(snapshot)

    @pytest.mark.asyncio
    async def test_an_empty_pool_skips_the_ranking_call_entirely(self, mock_question):
        """With nothing to rank there is no LLM call to make, so the stage reports a benign
        `none` rather than a loss: an empty pool is the venues' story to tell through their own
        tokens, not a ranking failure."""
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, {"events": [], "cursor": ""}),
                _PREDICTIT_URL: FakeResponse(200, {"markets": []}),
            }
        )
        ranker_prompts: list[str] = []

        def _record(prompt: str) -> str:
            ranker_prompts.append(prompt)
            return "[]"

        snapshot = await _fetch(mock_question, handlers, ranking=_record)

        assert ranker_prompts == []
        assert snapshot.matches == []
        assert snapshot.sources["ranking"] == "none"
        assert pmp.prediction_market_source_losses() == 0

    @pytest.mark.asyncio
    async def test_the_settlement_join_reaches_a_market_the_question_shares_no_words_with(self, mock_question):
        """The structural recall channel, end to end through the seam's own text assembly.

        The event's title shares nothing with the question; what connects them is that both name
        the same publisher. It is joined via `resolution_criteria` + `fine_print` CONCATENATED —
        the fine print is where a question often names the actual release page, so a pipeline
        that passed only the resolution criteria would miss this class entirely, which is what
        this test pins.
        """
        mock_question.title = "What will the U-3 unemployment rate be in July 2026?"
        mock_question.question_text = mock_question.title
        mock_question.resolution_criteria = "Per the official release."
        mock_question.fine_print = "Resolution uses https://data.bls.gov/timeseries/LNS14000000"

        joined_event = {
            "event_ticker": "KXJOBLESS-26JUL",
            "title": "Jobless rate, July",
            "settlement_sources": [{"name": "BLS", "url": "https://www.bls.gov/news.release/empsit.nr0.htm"}],
            "markets": [{"ticker": "KXJOBLESS-26JUL-T4", "status": "active", "close_time": "2026-08-07T00:00:00Z"}],
        }
        handlers = _handlers(
            **{_KALSHI_EVENTS_URL: FakeResponse(200, {"events": [joined_event, _OFF_TOPIC_KALSHI_EVENT], "cursor": ""})}
        )
        captured: list[str] = []

        def _capture(prompt: str) -> str:
            captured.append(prompt)
            return _rank_one_per_venue(prompt)

        snapshot = await _fetch(mock_question, handlers, ranking=_capture)

        # The join is the first channel, and channel order IS the pool order, so the joined event
        # leads the Kalshi block ahead of the alphabetically-earlier off-topic one.
        assert "[0] (kalshi) Jobless rate, July" in captured[0]
        assert "settles via: BLS" in captured[0]
        joined = next(row for row in snapshot.matches if row.venue_market_id == "KXJOBLESS-26JUL")
        assert joined.retrieval_channel == "settlement_join"

    @pytest.mark.asyncio
    async def test_timeout_returns_an_empty_snapshot_soft_fail(self, mock_question):
        """A per-question timeout must NOT raise: soft-fail with an empty snapshot, a loss token
        and a counter bump. With an empty `sources` map the diagnostics line renders no suffix at
        all, so a dead snapshot would be indistinguishable from one nobody asked for."""

        class StalledLlm:
            def __init__(self, **kwargs: Any) -> None:
                pass

            async def invoke(self, prompt: str) -> str:
                await asyncio.sleep(10)
                return "..."

        with patch.object(pmp, "build_llm_with_openrouter_fallback", StalledLlm):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=0.05)

        assert snapshot.matches == []
        assert snapshot.sources == {"snapshot": "error(timeout)"}
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_one_dead_venue_does_not_take_the_others_down(self, mock_question, manifold_payload, caplog):
        """Soft-fail isolation at venue granularity: a venue that raises outright is that
        venue's loss, and every other venue's rows still reach the pool."""

        def _boom(_params: dict[str, Any]) -> FakeResponse:
            raise RuntimeError("connection refused")

        handlers = _handlers(
            **{
                _POLY_URL: _boom,
                _MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload),
                _KALSHI_EVENTS_URL: _boom,
                _PREDICTIT_URL: _boom,
            }
        )

        with caplog.at_level(logging.WARNING):
            snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        assert "manifold" in {row.platform for row in snapshot.matches}
        assert "polymarket" not in {row.platform for row in snapshot.matches}
        assert snapshot.sources["manifold"].startswith("ok(")
        assert "degraded (alertable)" in "\n".join(rec.getMessage() for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_the_snapshot_cache_key_includes_as_of(self, mock_question, kalshi_events_payload):
        """One entry per (qid, as_of), so a backtest at as_of=A cannot reuse a snapshot computed
        at as_of=B. The provider path passes None, which is finally what makes this cache
        hittable — the old `datetime.now(utc)` default changed the key on every call."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})
        as_of_a = datetime(2026, 5, 1, tzinfo=timezone.utc)
        as_of_b = datetime(2026, 6, 1, tzinfo=timezone.utc)

        await _fetch(mock_question, handlers, as_of=as_of_a)
        await _fetch(mock_question, handlers, as_of=as_of_b)
        await _fetch(mock_question, handlers, as_of=None)

        assert (mock_question.id_of_question, as_of_a.isoformat()) in pmp._SNAPSHOT_CACHE
        assert (mock_question.id_of_question, as_of_b.isoformat()) in pmp._SNAPSHOT_CACHE
        assert (mock_question.id_of_question, "none") in pmp._SNAPSHOT_CACHE

    @pytest.mark.asyncio
    async def test_an_explicit_as_of_filters_the_pool_before_the_ranker_sees_it(self, mock_question):
        """The leakage filter survives for explicit callers, and it filters the POOL.

        Filtering the pool rather than the rendered rows is what keeps the ranker from reading a
        market that already resolved, and keeps the candidate indices the model is given
        identical to the ones its picks are applied against. The provider path passes None — that
        decision is asserted separately in TestProviderFactory.
        """
        closed_event = {
            "event_ticker": "KXSTAR-PAST",
            "title": "Will SpaceX Starship reach orbit in 2026?",
            "markets": [
                {
                    "ticker": "KXSTAR-PAST-YES",
                    "rules_primary": "orbital velocity",
                    "status": "settled",
                    "close_time": "2026-04-01T00:00:00Z",
                }
            ],
        }
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, {"events": [closed_event], "cursor": ""})})
        captured: list[str] = []

        def _capture(prompt: str) -> str:
            captured.append(prompt)
            return _rank_one_per_venue(prompt)

        snapshot = await _fetch(
            mock_question, handlers, ranking=_capture, as_of=datetime(2026, 5, 1, tzinfo=timezone.utc)
        )

        assert "kalshi" not in {row.platform for row in snapshot.matches}
        assert not any("KXSTAR-PAST" in prompt or "reach orbit in 2026" in prompt for prompt in captured)

    @pytest.mark.asyncio
    async def test_the_ranking_telemetry_names_every_rendered_rows_pool_index(
        self, mock_question, kalshi_events_payload, caplog
    ):
        """One INFO line per question, carrying `(venue, pool_index, rank)` per rendered row.

        The pool index is the point: it is the post-ship instrument for the two questions this
        port deliberately left open — whether ranker attention decays down a ~400-candidate
        prompt, and whether Manifold detail enrichment changes which rows get picked. Both then
        answer themselves from prod logs instead of another bake-off.
        """
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        with caplog.at_level(logging.INFO):
            snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        lines = [rec.getMessage() for rec in caplog.records if "MARKET_RANKING:" in rec.getMessage()]
        assert len(lines) == 1, lines
        line = lines[0]
        assert f"question={mock_question.id_of_question}" in line
        assert "outcome=ranked" in line
        assert f"rows={len(snapshot.matches)}" in line
        assert re.search(r"pool=[1-9]\d*", line), line
        assert re.search(r"prompt_chars=[1-9]\d*", line), line
        for rank, row in enumerate(snapshot.matches):
            assert re.search(rf"{row.platform}:\d+@{rank}", line), line


# ---------------------------------------------------------------------------
# prediction_market_provider factory (research-provider flag gating)
# ---------------------------------------------------------------------------


class TestProviderFactory:
    @pytest.mark.asyncio
    async def test_disabled_flag_returns_empty_at_the_provider_entrypoint(self, monkeypatch, mock_question):
        """Defence in depth: the orchestrator also gates registration on this flag, but the
        provider re-checks so a direct caller cannot bypass it."""
        monkeypatch.delenv("PREDICTION_MARKETS_ENABLED", raising=False)

        assert await prediction_market_provider()(mock_question) == ""

    @pytest.mark.asyncio
    async def test_is_benchmarking_short_circuits_regardless_of_the_env_flag(self, monkeypatch, mock_question):
        """There is no orchestrator-level backstop, so THIS check is the backtest defence.

        Markets retain their last-trade price after resolution, and the `as_of` filter alone was
        never sufficient — a market that closes between `as_of` and now still leaks. Mirrors the
        contract `gemini_search_provider` and `native_search_provider` use.
        """
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")

        assert await prediction_market_provider(is_benchmarking=True)(mock_question) == ""

    @pytest.mark.asyncio
    async def test_the_default_factory_call_gates_only_on_the_env_flag(self, monkeypatch, mock_question):
        """Control for the above: adding the parameter must not flip the default, which would
        silence the provider in prod where is_benchmarking is False."""
        monkeypatch.delenv("PREDICTION_MARKETS_ENABLED", raising=False)

        assert await prediction_market_provider()(mock_question) == ""

    @pytest.mark.asyncio
    async def test_the_provider_path_passes_no_as_of(self, monkeypatch, mock_question, kalshi_events_payload):
        """The provider passes `as_of=None`, deliberately, even for a question with a scheduled
        resolution.

        The old derivation (`scheduled_resolution_time - 1 day`) was worse than inert: it dropped
        every market closing before the question resolved, which is exactly the "same quantity,
        adjacent month" class that supplied most of the ranked arm's near-identical rows, and
        prod telemetry recorded the cost — 20 of 47 archived runs had Polymarket fetch candidates
        and render nothing because of it. The benchmarking guard is the leakage defence.
        """
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")
        mock_question.scheduled_resolution_time = datetime(2026, 8, 1, tzinfo=timezone.utc)
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        captured: list[datetime | None] = []
        original = pmp.fetch_market_snapshot

        async def _capturing(question_arg: Any, *, as_of: datetime | None = None, **kwargs: Any) -> MarketSnapshot:
            captured.append(as_of)
            return await original(question_arg, as_of=as_of, **kwargs)

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", _market_llm()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
            patch.object(pmp, "fetch_market_snapshot", _capturing),
        ):
            await pmp.prediction_market_provider()(mock_question)

        assert captured == [None]

    @pytest.mark.asyncio
    async def test_the_enabled_provider_fetches_and_formats(
        self, monkeypatch, mock_question, kalshi_events_payload, manifold_payload
    ):
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
                _MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload),
            }
        )

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", _market_llm(ranking=_rank_one_per_venue)),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            out = await pmp.prediction_market_provider()(mock_question)

        assert "MAY be relevant" in out
        assert "| " + " | ".join(TABLE_COLUMNS) + " |" in out
        assert "### Resolution criteria / rules" in out

    def test_a_timeout_below_the_stage_budget_warns_loudly_at_init(self, monkeypatch, caplog):
        """A stale `PREDICTION_MARKET_TIMEOUT=30` left in someone's .env would otherwise surface
        only as a generic snapshot timeout on every question, with the real cause invisible."""
        monkeypatch.setattr(pmp, "PREDICTION_MARKET_TIMEOUT", 30.0)

        with caplog.at_level(logging.WARNING):
            prediction_market_provider()

        logged = "\n".join(rec.getMessage() for rec in caplog.records)
        assert "BELOW the pipeline's worst-case stage sum" in logged
        assert str(pmp.SNAPSHOT_STAGE_BUDGET_S) in logged

    def test_the_default_timeout_clears_the_stage_budget(self):
        assert PREDICTION_MARKET_TIMEOUT >= pmp.SNAPSHOT_STAGE_BUDGET_S


# ---------------------------------------------------------------------------
# Formatter (a thin delegate; the render itself is tested in test_market_retrieval_rendering)
# ---------------------------------------------------------------------------


def _row(
    title: str,
    *,
    platform: str = "polymarket",
    tier: str = "same_quantity_other_cut",
    why: str = "same series, adjacent month",
    rules: str = "rules text",
) -> MarketMatch:
    return MarketMatch(
        platform=platform,  # type: ignore[arg-type]
        market_title=title,
        market_url=f"https://example.com/{platform}",
        implied_prob_yes=0.5,
        bid=None,
        ask=None,
        spread=None,
        volume_24h=None,
        close_time=datetime(2026, 12, 31, tzinfo=timezone.utc),
        is_resolved=False,
        match_confidence=1.0,
        raw_rules=rules,
        total_volume=100_000.0,
        open_interest=50_000.0,
        relation_tier=tier,
        relevance_label=why,
    )


class TestFormatterDelegate:
    def test_an_empty_snapshot_renders_nothing(self):
        """The early return is what produces `status="empty"` downstream and the
        attempted-vs-succeeded distinction residual analysis reads off the archive. Under ranked
        selection a zero-row render is a legitimate outcome, so this is a hot path."""
        assert format_snapshot_for_research(MarketSnapshot(matches=[])) == ""

    def test_the_degraded_marker_is_derived_from_the_snapshots_own_ranking_token(self):
        """Derived, not passed in, so the render is reproducible from an archived snapshot
        alone — which matters because `record_raw_research` archives the snapshot and a replay
        tool renders from it."""
        rows = [_row("A market", tier="")]

        healthy = format_snapshot_for_research(MarketSnapshot(matches=rows, sources={"ranking": "ok(1)"}))
        degraded = format_snapshot_for_research(MarketSnapshot(matches=rows, sources={"ranking": "error(x)"}))

        assert DEGRADED_RANKING_MARKER not in healthy
        assert degraded.startswith(DEGRADED_RANKING_MARKER)

    def test_a_missing_ranking_token_reads_as_degraded(self):
        """The conservative direction: an unlabelled snapshot may be in retrieval order, and
        claiming evidential order falsely is worse than a marker on a healthy table."""
        assert DEGRADED_RANKING_MARKER in format_snapshot_for_research(MarketSnapshot(matches=[_row("A market")]))

    def test_cells_are_addressable_by_header_name(self):
        """Read by HEADER NAME, never by index. The previous version of this test carried a
        comment enumerating cell positions, and every column addition silently shifted them."""
        snapshot = MarketSnapshot(matches=[_row("Will X win?", platform="predictit")], sources={"ranking": "ok(1)"})

        formatted = format_snapshot_for_research(snapshot)
        header = next(line for line in formatted.splitlines() if line.startswith("| platform |"))
        columns = [cell.strip() for cell in header.strip("|").split("|")]
        row = next(line for line in formatted.splitlines() if line.startswith("| predictit"))
        cells = dict(zip(columns, [cell.strip() for cell in row.strip("|").split("|")], strict=True))

        assert columns == list(TABLE_COLUMNS)
        assert cells["signal"] == "no-liquidity-data"
        assert cells["status"] == "open"
        assert cells["relation"] == "same_quantity_other_cut"
        assert cells["why"] == "same series, adjacent month"

    def test_the_preamble_follows_the_ranker_top_tier(self):
        """Strong when a rendered row measures the same quantity, neutral otherwise. The selector
        is the ranker's own grade, replacing the content-overlap + confidence bar it retired."""
        strong = format_snapshot_for_research(
            MarketSnapshot(matches=[_row("A", tier="same_quantity_same_date")], sources={"ranking": "ok(1)"})
        )
        neutral = format_snapshot_for_research(
            MarketSnapshot(matches=[_row("A", tier="weak")], sources={"ranking": "ok(1)"})
        )

        assert "extremely strong evidence" in strong.lower()
        assert "may all be off-topic" in neutral.lower()
        assert "extremely strong evidence" not in neutral.lower()
        # The retired vocabulary must not come back: rows are chosen by a model reading each
        # market's rules, not by word overlap.
        for rendered in (strong, neutral):
            assert "fuzzy" not in rendered.lower()
            assert "verify-carefully" not in rendered
            assert "likely-relevant" not in rendered

    @pytest.mark.parametrize(
        ("total_volume", "expected"), [(1_000.0, "thin"), (20_000.0, "decent"), (100_000.0, "deep")]
    )
    def test_liquidity_label_real_money_thresholds(self, total_volume, expected):
        row = _row("x")
        row.total_volume = total_volume
        row.open_interest = None
        assert _liquidity_label(row) == expected

    @pytest.mark.parametrize(("num_bettors", "expected"), [(5, "thin"), (50, "decent"), (200, "high")])
    def test_liquidity_label_manifold_bettor_thresholds(self, num_bettors, expected):
        row = _row("x", platform="manifold")
        row.total_volume = None
        row.open_interest = None
        row.num_bettors = num_bettors
        assert _liquidity_label(row) == expected


# ---------------------------------------------------------------------------
# Per-source diagnostics
# ---------------------------------------------------------------------------


class TestSnapshotSourceDiagnostics:
    """The seven per-source tokens on `MarketSnapshot.sources`.

    Drained by the orchestrator into the provider-diagnostics line, so PARTIAL degradation — one
    dead stage while the rest of the pipeline runs — is visible per question rather than only in
    an aggregate counter. A token starting with "ok"/"none" is benign; everything else is a lost
    source that bumps the alertable counter.
    """

    @pytest.fixture(autouse=True)
    def _no_retry_backoff(self, monkeypatch):
        """These tests drive 503 retry-exhaustion paths; the real 0.5s backoff between attempts
        would dominate their runtime."""
        _no_backoff(monkeypatch)

    @pytest.mark.asyncio
    async def test_a_healthy_run_reports_all_seven_sources_and_no_loss(self, mock_question, kalshi_events_payload):
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        assert set(snapshot.sources) == _ALL_SOURCES
        assert all(token.startswith(("ok", "none")) for token in snapshot.sources.values()), snapshot.sources
        # No lost source -> the diagnostics suffix stays empty (byte-identical to a healthy line).
        assert _partial_loss_suffix({"sources": snapshot.sources}) == ""
        assert pmp.prediction_market_source_losses() == 0

    @pytest.mark.asyncio
    async def test_all_live_but_empty_stays_benign_none(self, mock_question):
        """The other direction: live venues with nothing to say must ALL read `none` with an
        empty suffix, so the token vocabulary never cries outage on a healthy no-match question.

        The two enumerable venues are the exception and are correct to be: their catalogues
        populated, so they report `ok(N)` on the candidates they contributed even though the
        ranker kept none.
        """
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, {"events": [], "cursor": ""}),
                _PREDICTIT_URL: FakeResponse(200, {"markets": []}),
            }
        )

        snapshot = await _fetch(mock_question, handlers)

        assert snapshot.sources == {
            "polymarket": "none",
            "manifold": "none",
            "kalshi": "none",
            "predictit": "none",
            "manifold_detail": "none",
            "query_author": "ok(3)",
            "ranking": "none",
        }
        assert _partial_loss_suffix({"sources": snapshot.sources}) == ""
        assert pmp.prediction_market_source_losses() == 0

    @pytest.mark.asyncio
    async def test_a_lost_catalogue_is_a_venue_loss_and_bumps_the_degraded_counter(self, mock_question):
        """A failed catalogue pull bumps BOTH counters, which is deliberate over-counting rather
        than a bug: the two carry different marker fields, and the catalogue is now the
        generation backbone — it feeds the settlement-source join AND the fuzzy channel, so its
        loss is worse than it used to be, not better."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(503, text="service unavailable")})

        snapshot = await _fetch(mock_question, handlers)

        assert snapshot.sources["kalshi"] == "error(http_503)"
        assert "lost=kalshi:error(http_503)" in _partial_loss_suffix({"sources": snapshot.sources})
        assert pmp.kalshi_catalogue_fetch_failures() == 1
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_a_rate_limited_catalogue_reports_incomplete_without_a_second_violation(self, mock_question):
        """Re-asking a rate limiter 0.5s later is not a retry, it is a second violation. So a 429
        stops pagination dead and reports the pull incomplete, which is also what keeps a
        rate-limited exchange from being cached as a short catalogue for 6h."""
        session = FakeSession(_handlers(**{_KALSHI_EVENTS_URL: FakeResponse(429, text="slow down")}))

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", _market_llm()),
            patch.object(pmp, "_get_session", lambda: session),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        assert session._call_counts[_KALSHI_EVENTS_URL] == 1
        assert snapshot.sources["kalshi"] == "error(http_429)"
        assert "events" not in pmp._KALSHI_CACHE

    @pytest.mark.asyncio
    async def test_a_dead_query_author_is_an_additive_loss_not_a_silenced_pipeline(
        self, mock_question, kalshi_events_payload
    ):
        """The author's output is ADDITIVE, so its failure costs no recall — the deterministic
        query set still runs and the venues still answer. It is still reported, because a
        permanently dead author would otherwise be invisible, which is the exact failure class
        the degradation counters exist for."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        snapshot = await _fetch(
            mock_question, handlers, author="I could not help with that.", ranking=_rank_one_per_venue
        )

        assert snapshot.sources["query_author"] == "error(unusable)"
        assert "lost=query_author:error(unusable)" in _partial_loss_suffix({"sources": snapshot.sources})
        assert pmp.prediction_market_source_losses() == 1
        # The pipeline ran anyway: venues answered and rows rendered.
        assert snapshot.matches
        assert snapshot.sources["kalshi"].startswith("ok(")

    @pytest.mark.asyncio
    async def test_a_venue_outage_is_a_loss_and_the_healthy_venues_still_land(
        self, mock_question, manifold_payload, kalshi_events_payload
    ):
        """A 503'd venue reads as a LOSS, never as `none`."""
        handlers = _handlers(
            **{
                _POLY_URL: FakeResponse(503, text="service unavailable"),
                _MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload),
                _KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
            }
        )

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        assert snapshot.sources["polymarket"] == "error(all_queries_failed)"
        assert "lost=polymarket:" in _partial_loss_suffix({"sources": snapshot.sources})
        assert "manifold" in {row.platform for row in snapshot.matches}
        assert snapshot.sources["manifold"].startswith("ok(")
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_one_lost_query_of_several_reads_as_partial(self, mock_question, manifold_payload):
        """The venue fan-out issues every deduped query, so losing one of them used to publish a
        clean `ok(N)` off the survivors."""
        seen: list[str] = []

        def _manifold(params: dict[str, Any]) -> FakeResponse:
            term = params.get("term") or ""
            seen.append(term)
            if term == seen[0]:
                return FakeResponse(503, text="service unavailable")
            return FakeResponse(200, manifold_payload)

        handlers = _handlers(**{_MANIFOLD_SEARCH_URL: _manifold})

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        assert len(seen) > 1, "the venue must issue every deduped query, not one"
        assert snapshot.sources["manifold"].startswith("partial("), snapshot.sources["manifold"]
        assert "lost=manifold:partial(" in _partial_loss_suffix({"sources": snapshot.sources})
        assert "manifold" in {row.platform for row in snapshot.matches}, "the surviving queries' rows still land"
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_a_total_blackout_bumps_once_per_lost_source(self, mock_question):
        """Every venue down. Each is its own loss — the published diagnostics line must not read
        healthy through a blackout, and one bump for four dead venues would understate it."""
        handlers = {
            url: FakeResponse(503, text="service unavailable")
            for url in (_POLY_URL, _MANIFOLD_SEARCH_URL, _MANIFOLD_DETAIL_URL, _KALSHI_EVENTS_URL, _PREDICTIT_URL)
        }

        snapshot = await _fetch(mock_question, handlers)

        suffix = _partial_loss_suffix({"sources": snapshot.sources})
        for venue in ("polymarket", "manifold", "predictit"):
            assert snapshot.sources[venue] == "error(all_queries_failed)", venue
            assert f"{venue}:error(all_queries_failed)" in suffix, venue
        assert snapshot.sources["kalshi"] == "error(http_503)"
        assert pmp.prediction_market_source_losses() == 4

    @pytest.mark.asyncio
    async def test_manifold_enrichment_reports_none_when_there_is_nothing_to_enrich(
        self, mock_question, kalshi_events_payload
    ):
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        assert snapshot.sources["manifold_detail"] == "none"
        assert pmp.prediction_market_source_losses() == 0

    @pytest.mark.asyncio
    async def test_manifold_enrichment_fills_the_rules_text_the_search_listing_omits(
        self, mock_question, manifold_payload
    ):
        """Without this every Manifold candidate reaches the ranker title-only, and the prompt's
        stated "single most reliable cue" is the settlement/rules text. It must run BEFORE the
        prompt is built, because it mutates the pool rows in place and `apply_picks` copies
        them."""
        detail = {"textDescription": "Resolves YES on an FAA-confirmed orbital insertion."}
        handlers = _handlers(
            **{
                _MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload),
                _MANIFOLD_DETAIL_URL: FakeResponse(200, detail),
            }
        )
        captured: list[str] = []

        def _capture(prompt: str) -> str:
            captured.append(prompt)
            return _rank_one_per_venue(prompt)

        snapshot = await _fetch(mock_question, handlers, ranking=_capture)

        assert snapshot.sources["manifold_detail"].startswith("ok(")
        assert "FAA-confirmed orbital insertion" in captured[0]
        assert pmp.prediction_market_source_losses() == 0

    @pytest.mark.asyncio
    async def test_a_totally_lost_enrichment_fan_out_is_the_only_reported_case(self, mock_question, manifold_payload):
        """A lost detail GET costs rules text, never recall, so only a TOTAL loss is reported: a
        partial fan-out has nothing actionable to alert on."""
        handlers = _handlers(
            **{
                _MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload),
                _MANIFOLD_DETAIL_URL: FakeResponse(503, text="service unavailable"),
            }
        )

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        assert snapshot.sources["manifold_detail"] == "error(all_details_failed)"
        assert snapshot.sources["manifold"].startswith("ok("), "the SEARCH was healthy; only the detail GET died"
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_the_provider_records_the_sources_map_into_the_registry(
        self, monkeypatch, mock_question, kalshi_events_payload
    ):
        """The `_fetch` ResearchCallable records the map keyed by (qid, 'prediction_market') for
        the orchestrator to drain into the comment-only diagnostics block."""
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(503, text="service unavailable")})

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", _market_llm()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            await pmp.prediction_market_provider()(mock_question)

        detail = pop_provider_detail(mock_question.id_of_question, "prediction_market")
        assert set(detail.get("sources", {})) == _ALL_SOURCES
        assert detail["sources"]["kalshi"] == "error(http_503)"


# ---------------------------------------------------------------------------
# Provider-health recording
# ---------------------------------------------------------------------------


class TestProviderHealthRecording:
    """The recording seam: what a real snapshot fetch hands the degradation rules.

    The rules themselves are tested in `tests/test_provider_health.py`. These tests pin the
    numbers this module feeds them, which is where the risk actually lives — and under ranked
    selection the two counts come from DIFFERENT places, which is the whole subtlety: field
    presence and `candidates_pre_filter` from the venue's POOL rows, `rows_post_filter` from the
    RENDERED ones.
    """

    @pytest.fixture(autouse=True)
    def _reset_health(self):
        reset_provider_health()
        yield
        reset_provider_health()

    @pytest.mark.asyncio
    async def test_field_presence_is_measured_over_the_pool_not_the_ranked_rows(
        self, mock_question, kalshi_events_payload, polymarket_payload
    ):
        """The load-bearing choice, and the reason Signal A does not fire on the ranker's
        judgment.

        The ranker keeps only Kalshi here, so Polymarket renders zero rows. Measured over the
        RENDERED rows, Polymarket's declared fields would read 100% dead and redden CI on a
        legitimate 1-row ranking. Signal A exists to catch a PARSER whose field names went
        stale, so it must be immune to selection — which is what reading the pool gives it.
        """
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
                _POLY_URL: FakeResponse(200, polymarket_payload),
            }
        )

        def _kalshi_only(prompt: str) -> str:
            index = int(_CANDIDATE_LINE_RE.search(prompt).group(1))  # type: ignore[union-attr]
            return json.dumps([{"i": index, "tier": "same_quantity_same_date", "why": "same event"}])

        snapshot = await _fetch(mock_question, handlers, ranking=_kalshi_only)

        assert {row.platform for row in snapshot.matches} == {"kalshi"}
        observed = {obs.venue: obs for obs in recorded_observations()[0]}
        assert observed["polymarket"].rows_post_filter == 0
        assert observed["polymarket"].candidates_pre_filter > 0
        assert set(observed["polymarket"].liquidity_fields_present) == set(
            VENUE_EXPECTED_LIQUIDITY_FIELDS["polymarket"]
        )
        assert provider_degradation_findings() == []

    @pytest.mark.asyncio
    async def test_rendered_counts_come_from_the_ranked_rows(self, mock_question, kalshi_events_payload):
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        rendered = [row for row in snapshot.matches if row.platform == "kalshi"]
        observed = next(obs for obs in recorded_observations()[0] if obs.venue == "kalshi")
        assert observed.rows_post_filter == len(rendered)

    @pytest.mark.asyncio
    async def test_a_market_less_question_records_zero_rows_and_finds_nothing(self, mock_question):
        """The false-positive case that matters most, through the real fetch path: every venue
        answered, the ranker kept nothing, and NOTHING is alertable.

        Note the asymmetry, which is a fact about ranked retrieval rather than a quirk of this
        fixture. The two SEARCH venues sit at zero candidates, because their index had no hit.
        The two ENUMERABLE venues cannot: their whole catalogue enters the pool, so a healthy
        catalogue always yields candidates. That is why Signal B is structurally dead for them and
        `record_catalogue_size` (Signal C) is their only alarm — and why a test that forced them
        to zero would have to stub an EMPTY catalogue, which is itself a finding.
        """
        await _fetch(mock_question, _handlers())

        observed = {obs.venue: obs for obs in recorded_observations()[0]}
        assert set(observed) == {"polymarket", "manifold", "kalshi", "predictit"}
        assert all(obs.rows_post_filter == 0 for obs in observed.values())
        assert observed["polymarket"].candidates_pre_filter == 0
        assert observed["manifold"].candidates_pre_filter == 0
        assert observed["kalshi"].candidates_pre_filter > 0
        assert observed["predictit"].candidates_pre_filter > 0
        assert provider_degradation_findings() == []

    @pytest.mark.asyncio
    async def test_a_lost_venue_records_no_observation(self, mock_question, monkeypatch):
        """A venue whose fan-out lost a sub-fetch is already alertable via
        `prediction_market_source_losses`, so it is not observed here — counting the same outage
        twice would report one failure as two, and "check the query construction" is the wrong
        remedy for a 503."""
        _no_backoff(monkeypatch)
        handlers = _handlers(**{_MANIFOLD_SEARCH_URL: FakeResponse(503, text="service unavailable")})

        await _fetch(mock_question, handlers)

        assert "manifold" not in {obs.venue for obs in recorded_observations()[0]}
        assert pmp.prediction_market_source_losses() >= 1
        assert provider_degradation_findings() == []

    @pytest.mark.asyncio
    async def test_both_catalogue_sizes_are_recorded_with_their_fetch_outcome(
        self, mock_question, kalshi_events_payload, predictit_payload
    ):
        """Signal C's input, and the only health signal the enumerable venues have left."""
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
                _PREDICTIT_URL: FakeResponse(200, predictit_payload),
            }
        )

        await _fetch(mock_question, handlers)

        catalogues = {obs.source: obs for obs in recorded_observations()[1]}
        assert catalogues["kalshi_events"].entries == len(kalshi_events_payload["events"])
        assert catalogues["kalshi_events"].fetch_ok is True
        assert catalogues["predictit_markets"].entries == len(predictit_payload["markets"])
        assert catalogues["predictit_markets"].fetch_ok is True

    @pytest.mark.asyncio
    async def test_a_failed_pull_records_the_failure_not_a_phantom_empty_catalogue(self, mock_question, monkeypatch):
        """A 503'd catalogue must record `fetch_ok=False`, so Signal C stays silent and the
        source-loss counter remains the sole reporter."""
        _no_backoff(monkeypatch)
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(503, text="service unavailable")})

        await _fetch(mock_question, handlers)

        catalogues = {obs.source: obs for obs in recorded_observations()[1]}
        assert catalogues["kalshi_events"].fetch_ok is False
        assert provider_degradation_findings() == []

    @pytest.mark.asyncio
    async def test_a_question_with_no_id_records_nothing(self, mock_question):
        """Every observation is keyed on the question, so an id-less question records nothing —
        matching `record_provider_detail` / `record_raw_research`."""
        mock_question.id_of_question = None

        await _fetch(mock_question, _handlers())

        assert recorded_observations() == ((), ())

    @pytest.mark.asyncio
    async def test_recording_does_not_alter_the_snapshot(self, mock_question, kalshi_events_payload):
        """Recording is a pure module-state write on the research path: same rows, same tokens,
        with or without observations already present."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})
        first = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        _reset_session_caches()
        reset_provider_health()
        second = await _fetch(
            mock_question,
            _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)}),
            ranking=_rank_one_per_venue,
        )

        assert [(row.platform, row.market_title) for row in first.matches] == [
            (row.platform, row.market_title) for row in second.matches
        ]
        assert first.sources == second.sources


# ---------------------------------------------------------------------------
# Generation widths, through the seam
# ---------------------------------------------------------------------------


class TestRetrievalWidthIsNotTheRenderCap:
    """The port's biggest silent-failure risk, pinned end to end.

    In the old design the render cap WAS the retrieval width (`top_k = max_matches_per_platform
    + 2`), so a ranked pipeline that inherited those call sites would hand the ranker a
    5-candidate pool and measure nothing while looking finished. The pool has to be wide, and
    the render cap has to be independent of it.
    """

    @pytest.mark.asyncio
    async def test_a_large_catalogue_reaches_the_ranker_at_full_width(self, mock_question):
        big = {
            "events": [
                {
                    "event_ticker": f"KX{i:04d}",
                    "title": f"Will SpaceX Starship reach orbit, threshold {i}?",
                    "markets": [{"ticker": f"KX{i:04d}-Y", "status": "active"}],
                }
                for i in range(RETRIEVAL_WIDTH_KALSHI + 50)
            ],
            "cursor": "",
        }
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, big)})
        captured: list[str] = []

        def _capture(prompt: str) -> str:
            captured.append(prompt)
            return _rank_one_per_venue(prompt)

        snapshot = await _fetch(mock_question, handlers, ranking=_capture)

        kalshi_lines = [line for line in captured[0].splitlines() if line.startswith("[") and "(kalshi)" in line]
        assert len(kalshi_lines) == RETRIEVAL_WIDTH_KALSHI
        # And the RENDER stays the ranker's choice, bounded by the budget rather than by the width.
        assert len(snapshot.matches) <= RENDER_BUDGET


RETRIEVAL_WIDTH_KALSHI = generation.RETRIEVAL_WIDTH["kalshi"]
