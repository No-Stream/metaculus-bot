"""Tests for the prediction-market research provider (Polymarket + Kalshi + Manifold).

All HTTP is mocked via aiohttp_client fake sessions. No real network calls.

The provider module is split into:
- `fetch_market_snapshot` — orchestrator returning a `MarketSnapshot`
- `prediction_market_provider` — factory for the `ResearchCallable` plugged into
  `_select_research_providers`
- `format_snapshot_for_research` — markdown formatter for the LLM prompt

Tests cover (one per behavior):
- Polymarket search, parse, implied_prob extraction from outcomePrices
- Kalshi prefetch + client-side fuzzy match + yes_bid/yes_ask -> implied_prob
- Manifold search + direct probability field
- `as_of` filter drops post-as-of matches (backtest leakage defense)
- Keyword-extraction calls gpt-5.6-luna with max_tokens >= 800 (G0 token-budget trap)
- Malformed response -> empty snapshot + WARNING logged
- Polymarket 403 rate-limit -> bounded retry-with-backoff -> eventual empty
- Per-question timeout -> empty snapshot (soft-fail)
- `PREDICTION_MARKETS_ENABLED=false` -> orchestrator returns empty
- Formatter emits the NOT-AN-ANCHOR caveat + raw_rules
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Mapping
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import aiohttp
import litellm.exceptions
import pytest

from metaculus_bot.constants import PREDICTION_MARKET_TIMEOUT
from metaculus_bot.llm_configs import PREDICTION_MARKET_KEYWORD_LLM_CONFIG
from metaculus_bot.research import prediction_market as pmp
from metaculus_bot.research.http_fetch import ERROR_SNIPPET_BYTES
from metaculus_bot.research.prediction_market import (
    MarketMatch,
    MarketSnapshot,
    _extract_title_entities,
    _kalshi_entity_matches,
    _kalshi_prefetch_events,
    _kalshi_search_local,
    _liquidity_label,
    _manifold_search,
    _match_entities_to_series,
    _parse_manifold_matches,
    _parse_polymarket_matches,
    _polymarket_search,
    _predictit_prefetch,
    _predictit_search_local,
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


# ---------------------------------------------------------------------------
# Test isolation: module-level caches in prediction_market_provider.py bleed
# across tests when run in certain orders (observed 2026-05-12: full-suite
# run fails 2 tests that pass individually). Reset every test.
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

    The Kalshi /series fetch stream-parses `resp.content.iter_chunked()`; the
    other JSON endpoints go through `_read_json_capped`, which gates on a `.read`
    attribute FakeResponse deliberately lacks, so they still use the `.json()`
    path. `raw_content` overrides the serialized bytes so a test can inject
    malformed JSON.
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
        # Real aiohttp serves `.text()` and `.content` from ONE body, so an error-page
        # stub must stream its text too: the non-200 log path reads a BOUNDED snippet
        # off `.content`, and a stub holding its text outside the stream would hide a
        # regression back to whole-body `resp.text()`.
        if raw_content is None and payload is None and text:
            raw_content = text.encode()
        self.content = _FakeStreamContent(payload, raw_content=raw_content)

    async def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload  # noqa: ASYNC910

    async def text(self) -> str:
        # Faithful to aiohttp: reads the WHOLE body (the memory trap the bounded
        # snippet read exists to avoid).
        chunks = [chunk async for chunk in self.content.iter_chunked(65536)]
        return b"".join(chunks).decode("utf-8", errors="replace")

    async def __aenter__(self) -> "FakeResponse":
        return self  # noqa: ASYNC910

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        return None  # noqa: ASYNC910


# A handler is either a ready-made response or a callable that builds one from the GET params.
_Handler = FakeResponse | Callable[[dict[str, Any]], FakeResponse]
# Per-URL value accepted by FakeSession: a single handler or a list of them. Mapping (not dict) so
# call-site literals like {url: FakeResponse(...)} are accepted (dict is invariant in its value type).
_Handlers = Mapping[str, _Handler | list[_Handler]]


class FakeSession:
    """Simple aiohttp.ClientSession replacement that dispatches GET requests
    to handler callables keyed by URL prefix.
    """

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
    q.page_url = "https://metaculus.com/q/12345"
    # Set to None so the provider falls back to datetime.now(UTC) for as_of,
    # which keeps post-2026 close_times in the test fixtures un-filtered.
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
    """Kalshi /events?status=open&with_nested_markets=true shape.

    Field names match the live API (`volume_fp` / `open_interest_fp`, fixed-point STRINGS;
    `liquidity_dollars` deprecated at a constant "0.0000"). This fixture previously carried
    the bare names `volume` / `open_interest`, which exist nowhere in Kalshi's spec — it was
    written in the same commit as the parser and encoded the same wrong guess, so the pair
    stayed green while every real Kalshi row rendered `no-liquidity-data`. The field-name
    contract is now additionally pinned against captured live payloads in
    `test_prediction_market_liquidity_contract.py`, which is where a rename should fail
    first.
    """
    return {
        "events": [
            {
                "event_ticker": "KXSPACEX-26",
                "title": "Will SpaceX Starship reach orbit in 2026?",
                "sub_title": "Before Dec 31 2026",
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
                    }
                ],
            },
            {
                "event_ticker": "KXOTHER-1",
                "title": "Unrelated World Cup market",
                "markets": [{"ticker": "KXOTHER-1-YES", "title": "Unrelated"}],
            },
        ],
        "cursor": "",
    }


@pytest.fixture
def predictit_payload():
    """PredictIt /api/marketdata/all/ shape (verified live). US-politics only,
    no volume/liquidity/OI fields anywhere."""
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
                        "lastClosePrice": 0.57,
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
                    {
                        "id": 200,
                        "name": "Candidate A",
                        "shortName": "A",
                        "lastTradePrice": 0.25,
                        "status": "Open",
                    },
                    {
                        "id": 201,
                        "name": "Candidate B",
                        "shortName": "B",
                        "lastTradePrice": 0.20,
                        "status": "Open",
                    },
                ],
            },
        ]
    }


# ---------------------------------------------------------------------------
# Polymarket tests
# ---------------------------------------------------------------------------


class TestPolymarket:
    @pytest.mark.asyncio
    async def test_search_and_parse_implied_prob(self, polymarket_payload):
        session = FakeSession({"https://gamma-api.polymarket.com/public-search": FakeResponse(200, polymarket_payload)})
        matches = await _polymarket_search(session, "Starship orbit 2026")

        assert matches is not None  # a successful search never signals failure
        assert len(matches) == 2
        top = matches[0]
        assert top.platform == "polymarket"
        assert "Starship" in top.market_title
        assert top.implied_prob_yes == pytest.approx(0.74)
        assert top.market_url.startswith("https://polymarket.com/")
        assert top.bid == pytest.approx(0.73)
        assert top.ask == pytest.approx(0.75)
        # Previously-discarded liquidity fields now populate from volumeNum / liquidityNum / openInterest.
        assert top.total_volume == pytest.approx(987654.0)
        assert top.liquidity == pytest.approx(45000.0)
        assert top.open_interest == pytest.approx(60000.0)
        assert top.num_bettors is None  # not a Polymarket concept

    @pytest.mark.asyncio
    async def test_match_confidence_is_nonzero_for_strong_match(self, polymarket_payload):
        """F9: every Polymarket parse populates a real fuzzy-match confidence,
        not the legacy 0.0 placeholder. Confidence comes from token_set_ratio
        between the active query and the market title."""

        session = FakeSession({"https://gamma-api.polymarket.com/public-search": FakeResponse(200, polymarket_payload)})
        matches = await _polymarket_search(session, "Starship orbit 2026")
        assert matches is not None

        # The query "Starship orbit 2026" is a strong token-set-ratio match
        # against the top market title; confidence must be well above 0.0.
        assert matches[0].match_confidence > 0.5
        # Every match has SOME positive confidence (queries overlap on "2026").
        assert all(m.match_confidence > 0.0 for m in matches)

    @pytest.mark.asyncio
    async def test_rate_limit_retry_with_backoff_then_empty(self, monkeypatch):
        """403 on every attempt -> bounded retry -> eventual empty list, no exception."""

        call_count = {"n": 0}

        def handler(_params):
            call_count["n"] += 1
            return FakeResponse(403, text="rate limited")

        session = FakeSession({"https://gamma-api.polymarket.com/public-search": [handler, handler, handler, handler]})

        # Patch sleep so we don't actually wait
        sleeps: list[float] = []

        async def _no_sleep(s: float) -> None:  # noqa: ASYNC124, ASYNC910
            sleeps.append(s)

        monkeypatch.setattr(pmp.asyncio, "sleep", _no_sleep)

        matches = await pmp._polymarket_search(session, "anything")
        # None, not []: retry exhaustion is an upstream LOSS, and the caller has to be
        # able to tell it from a search that succeeded and matched nothing.
        assert matches is None
        # Should have retried a bounded number of times (>=2, <= 5)
        assert 2 <= call_count["n"] <= 5
        # And at least one backoff sleep
        assert len(sleeps) >= 1

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty_and_warns(self, caplog):
        session = FakeSession(
            {"https://gamma-api.polymarket.com/public-search": FakeResponse(200, payload={"garbage": 1})}
        )
        with caplog.at_level(logging.WARNING):
            matches = await _polymarket_search(session, "anything")
        assert matches == []

    def test_parse_implied_prob_from_list_prices(self):
        payload = {
            "events": [
                {
                    "title": "Will X happen?",
                    "slug": "will-x-happen",
                    "markets": [{"outcomePrices": [0.33, 0.67]}],
                }
            ]
        }
        matches = _parse_polymarket_matches(payload)
        assert len(matches) == 1
        assert matches[0].implied_prob_yes == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# Kalshi tests
# ---------------------------------------------------------------------------


class TestKalshi:
    @pytest.mark.asyncio
    async def test_prefetch_and_local_fuzzy_match(self, kalshi_events_payload):
        session = FakeSession(
            {"https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(200, kalshi_events_payload)}
        )
        events, tally = await _kalshi_prefetch_events(session, event_limit=100, page_sleep_s=0.0)
        assert len(events) == 2
        assert (tally.ok, tally.failed) == (1, 0)

        matches = _kalshi_search_local(events, "Starship orbit 2026", top_k=3, min_score=30.0)
        assert len(matches) >= 1
        top = matches[0]
        assert top.platform == "kalshi"
        assert "Starship" in top.market_title
        # yes_bid=0.68, yes_ask=0.72 -> midpoint 0.70
        assert top.implied_prob_yes == pytest.approx(0.70, abs=0.01)
        assert top.market_url.startswith("https://kalshi.com/")
        assert "orbital velocity" in top.raw_rules.lower()
        # Liquidity fields come from volume_fp / open_interest_fp, converted to the USD the
        # thresholds are denominated in: volume is turnover (x $0.70 midpoint price) and
        # open interest is collateral (x $1.00 notional). See _kalshi_usd_liquidity.
        assert top.total_volume == pytest.approx(82_000.0 * 0.70)
        assert top.open_interest == pytest.approx(15_000.0)
        # `liquidity_dollars` is deprecated upstream (always "0.0000"), so nothing reads it.
        assert top.liquidity is None

    @pytest.mark.asyncio
    async def test_prefetch_handles_http_error(self, caplog):
        session = FakeSession({"https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(500, text="boom")})
        with caplog.at_level(logging.WARNING):
            events, tally = await _kalshi_prefetch_events(session, event_limit=100, page_sleep_s=0.0)
        assert events == []
        # The tally carries the loss so the caller doesn't publish the outage as `none`.
        assert (tally.ok, tally.failed) == (0, 1)
        # And a failed prefetch must NOT poison the 6h cache with its empty list.
        assert "events" not in pmp._KALSHI_CACHE

    @pytest.mark.asyncio
    async def test_prefetch_writes_cache_incrementally(self):
        """F6: Kalshi prefetch updates _KALSHI_CACHE after each successful page,
        not only at the end. A partial run still warms whatever pages completed.
        """

        page_one = {
            "events": [
                {"event_ticker": "EV1", "title": "Event 1", "markets": []},
                {"event_ticker": "EV2", "title": "Event 2", "markets": []},
            ],
            "cursor": "next-page-cursor",
        }

        # Capture cache state right after page 1 completes by failing page 2.
        captured_cache_after_page_one: list[list[dict]] = []

        call_count = {"n": 0}

        def handler(_params):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # Snapshot the cache as it will be set inside prefetch right
                # after this page is consumed.
                return FakeResponse(200, page_one)
            # Return 500 on page 2 so prefetch breaks early -- cache should
            # already contain page 1.
            captured_cache_after_page_one.append(list(pmp._KALSHI_CACHE.get("events", (0, []))[1]))
            return FakeResponse(500, text="page 2 boom")

        pmp._reset_session_caches()
        session = FakeSession({"https://api.elections.kalshi.com/trade-api/v2/events": [handler, handler]})
        events, tally = await pmp._kalshi_prefetch_events(session, event_limit=100, page_sleep_s=0.0)

        # Returned list should have 2 events from page 1 (page 2 broke early).
        assert len(events) == 2
        # One page landed, one was lost -> a partial fetch, not a clean one.
        assert (tally.ok, tally.failed) == (1, 1)
        # And the cache contained page 1 events BEFORE page 2 was attempted.
        assert len(captured_cache_after_page_one) == 1
        assert len(captured_cache_after_page_one[0]) == 2


# ---------------------------------------------------------------------------
# Manifold tests
# ---------------------------------------------------------------------------


class TestManifold:
    @pytest.mark.asyncio
    async def test_search_and_parse_direct_probability(self, manifold_payload):
        session = FakeSession({"https://api.manifold.markets/v0/search-markets": FakeResponse(200, manifold_payload)})
        matches = await _manifold_search(session, "Starship orbit July 2026")

        assert matches is not None  # a successful search never signals failure
        assert len(matches) == 1
        m = matches[0]
        assert m.platform == "manifold"
        assert m.implied_prob_yes == pytest.approx(0.62)
        assert "Starship" in m.market_title
        assert m.is_resolved is False
        # Previously-discarded participation fields now populate from volume / totalLiquidity / uniqueBettorCount.
        assert m.total_volume == pytest.approx(8200.0)
        assert m.liquidity == pytest.approx(1500.0)
        assert m.num_bettors == 42

    @pytest.mark.asyncio
    async def test_non_list_payload_returns_empty(self, caplog):
        session = FakeSession(
            {"https://api.manifold.markets/v0/search-markets": FakeResponse(200, payload={"unexpected": "shape"})}
        )
        with caplog.at_level(logging.WARNING):
            matches = await _manifold_search(session, "anything")
        assert matches == []

    @pytest.mark.asyncio
    async def test_match_confidence_is_nonzero_for_strong_match(self, manifold_payload):
        """F9: Manifold parse populates real fuzzy-match confidence per row."""

        session = FakeSession({"https://api.manifold.markets/v0/search-markets": FakeResponse(200, manifold_payload)})
        matches = await _manifold_search(session, "Starship orbit July 2026")

        assert matches is not None
        assert len(matches) == 1
        # Title is "Will Starship reach orbit before July 2026?" — strong overlap.
        assert matches[0].match_confidence > 0.5

    def test_rules_fallback_to_description_tiptap_doc(self):
        """3b: the search endpoint often omits `textDescription`; the rich body
        lives in the `description` TipTap/ProseMirror doc. `raw_rules` must fall
        back to the flattened text-node content, not an empty string."""

        payload = [
            {
                "id": "xyz",
                "question": "Will the Fed cut rates in Q3?",
                "slug": "fed-cut-q3",
                "creatorUsername": "econWatcher",
                "probability": 0.4,
                # textDescription absent (or empty) — must fall back to `description`.
                "textDescription": "",
                "description": {
                    "type": "doc",
                    "content": [
                        {
                            "type": "paragraph",
                            "content": [
                                {"type": "text", "text": "Resolves YES if the FOMC cuts"},
                                {"type": "text", "text": " the target rate at any Q3 meeting."},
                            ],
                        },
                        {
                            "type": "paragraph",
                            "content": [{"type": "text", "text": "Per official Fed statements."}],
                        },
                    ],
                },
                "isResolved": False,
            }
        ]
        matches = _parse_manifold_matches(payload, query="Fed cut rates")
        assert len(matches) == 1
        rules = matches[0].raw_rules
        assert rules  # non-empty
        assert "Resolves YES if the FOMC cuts" in rules
        assert "target rate at any Q3 meeting" in rules
        assert "official Fed statements" in rules

    def test_rules_fallback_to_question_title_when_no_description(self):
        """When both textDescription and description are missing, fall back to the title."""

        payload = [
            {
                "id": "q1",
                "question": "Will it rain tomorrow?",
                "slug": "rain",
                "creatorUsername": "weather",
                "probability": 0.3,
                "isResolved": False,
            }
        ]
        matches = _parse_manifold_matches(payload, query="rain")
        assert matches[0].raw_rules == "Will it rain tomorrow?"


# ---------------------------------------------------------------------------
# PredictIt tests (prefetch full dump + client-side fuzzy match)
# ---------------------------------------------------------------------------


class TestPredictIt:
    @pytest.mark.asyncio
    async def test_prefetch_and_local_fuzzy_match(self, predictit_payload):
        session = FakeSession({"https://www.predictit.org/api/marketdata/all/": FakeResponse(200, predictit_payload)})
        markets = await _predictit_prefetch(session)
        assert markets is not None  # a successful prefetch never signals failure
        assert len(markets) == 2

        matches = _predictit_search_local(markets, "Starship orbit 2026", top_k=3, min_score=30.0)
        assert len(matches) >= 1
        top = matches[0]
        assert top.platform == "predictit"
        assert "Starship" in top.market_title
        # Priced from the order book: yes_ask = bestBuyYesCost (0.59),
        # yes_bid = 1 - bestBuyNoCost (0.58), implied = midpoint.
        assert top.implied_prob_yes == pytest.approx(0.585)
        assert top.bid == pytest.approx(0.58)
        assert top.ask == pytest.approx(0.59)
        assert top.spread == pytest.approx(0.01)
        assert top.market_url.startswith("https://www.predictit.org/")
        # Strong title match -> nonzero confidence.
        assert top.match_confidence > 0.5

    def test_predictit_has_no_liquidity_fields(self, predictit_payload):
        """PredictIt exposes no volume/liquidity/OI/bettor data -> all None."""

        matches = _predictit_search_local(predictit_payload["markets"], "Starship orbit 2026", min_score=30.0)
        assert matches
        top = matches[0]
        assert top.total_volume is None
        assert top.liquidity is None
        assert top.open_interest is None
        assert top.num_bettors is None
        assert top.volume_24h is None

    def test_multi_contract_market_names_the_contract_in_title(self, predictit_payload):
        """A market with >1 contract tags the priced contract into the title
        so the forecaster can tell which sub-outcome the price refers to."""

        matches = _predictit_search_local(predictit_payload["markets"], "2028 primary", top_k=3, min_score=30.0)
        assert matches
        top = matches[0]
        assert "2028 primary" in top.market_title.lower()
        # Some contract name is disambiguated into the title (query has no
        # candidate name, so contract choice falls back to the first).
        assert "Candidate" in top.market_title

    def test_multi_contract_prices_query_matching_contract(self):
        """A multi-contract market must price the contract whose name matches the
        query, not blindly contracts[0] (that would attach the wrong outcome's
        price to a good market match)."""

        markets = [
            {
                "id": 42,
                "name": "Who wins the 2028 Democratic nomination?",
                "shortName": "2028 Dem nom",
                "url": "https://www.predictit.org/markets/detail/42/dem-nom",
                "contracts": [
                    {"id": 1, "name": "Gavin Newsom", "lastTradePrice": 0.31, "status": "Open"},
                    {"id": 2, "name": "Alexandria Ocasio-Cortez", "lastTradePrice": 0.12, "status": "Open"},
                    {"id": 3, "name": "Josh Shapiro", "lastTradePrice": 0.09, "status": "Open"},
                ],
            }
        ]
        matches = _predictit_search_local(
            markets, "Will Josh Shapiro win the 2028 Democratic nomination?", min_score=30.0
        )
        assert matches
        top = matches[0]
        # Shapiro's contract (0.09), not Newsom's contracts[0] (0.31), must be priced.
        assert top.implied_prob_yes == pytest.approx(0.09)
        assert "Josh Shapiro" in top.market_title

    @pytest.mark.asyncio
    async def test_malformed_payload_returns_empty_and_warns(self, caplog):
        session = FakeSession({"https://www.predictit.org/api/marketdata/all/": FakeResponse(200, payload=[1, 2, 3])})
        with caplog.at_level(logging.WARNING):
            markets = await _predictit_prefetch(session)
        assert markets == []
        assert any("predictit" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_missing_markets_key_returns_empty_and_warns(self, caplog):
        session = FakeSession(
            {"https://www.predictit.org/api/marketdata/all/": FakeResponse(200, payload={"unexpected": "shape"})}
        )
        with caplog.at_level(logging.WARNING):
            markets = await _predictit_prefetch(session)
        assert markets == []
        assert any("predictit" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_reset_session_caches_clears_predictit_cache(self):
        """The autouse fixture resets caches; this pins that _PREDICTIT_CACHE is
        actually part of _reset_session_caches (regression guard)."""

        pmp._PREDICTIT_CACHE["markets"] = (0.0, [{"id": 1}])
        assert pmp._PREDICTIT_CACHE
        pmp._reset_session_caches()
        assert pmp._PREDICTIT_CACHE == {}


# ---------------------------------------------------------------------------
# Keyword extraction (S4 + S5 union, gpt-5.6-luna max_tokens >= 800)
# ---------------------------------------------------------------------------


class TestKeywordExtractor:
    @pytest.mark.asyncio
    async def test_extract_runs_s4_and_s5_with_max_tokens_800(self, mock_question):
        """G0 token-budget trap defense: the extractor MUST request max_tokens >= 800.
        reasoning=low burns 128-512 tokens on invisible reasoning (G0, measured
        on gpt-5.4-mini; luna succeeded it 2026-08-03 and is unmeasured there)."""

        captured_kwargs: list[dict] = []
        calls: list[str] = []

        class FakeLlm:
            def __init__(self, model: str, **kwargs):  # type: ignore[no-untyped-def]
                captured_kwargs.append({"model": model, **kwargs})
                self.model = model

            async def invoke(self, prompt: str) -> str:
                calls.append(prompt)
                if "noun phrases" in prompt.lower():
                    return "Starship orbit SpaceX July 2026 orbital velocity"  # noqa: ASYNC910
                return "SpaceX Starship orbit before July 2026"  # noqa: ASYNC910

        with patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm):
            extractor = pmp.KeywordExtractor(strategy="s4_s5_union")
            queries = await extractor.extract(mock_question)

        # Both S4 and S5 should run
        assert len(calls) == 2
        # Returned queries should be deduped union
        assert len(queries) >= 2
        # The constructed LLM must have max_tokens >= 800
        assert all(kw.get("max_tokens", 0) >= 800 for kw in captured_kwargs), captured_kwargs
        # And use gpt-5.6-luna
        assert all("gpt-5.6-luna" in kw.get("model", "") for kw in captured_kwargs)

    @pytest.mark.asyncio
    async def test_manifold_gets_extra_s2_query(self, mock_question):
        """Per G0 findings, Manifold prefers natural-language (S2) framings."""

        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                return "LLM query"  # noqa: ASYNC910

        with patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm):
            extractor = pmp.KeywordExtractor(strategy="s4_s5_union")
            queries = await extractor.extract(mock_question)
            manifold_queries = extractor.queries_for_platform(mock_question, queries, "manifold")

        # S2 = question_text trimmed at '?'
        assert any("Will SpaceX Starship reach orbit before July 2026" in q for q in manifold_queries)

    @pytest.mark.asyncio
    async def test_cache_by_question_id(self, mock_question):
        call_count = {"n": 0}

        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                call_count["n"] += 1
                return "starship orbit"  # noqa: ASYNC910

        with patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm):
            extractor = pmp.KeywordExtractor(strategy="s4_s5_union")
            await extractor.extract(mock_question)
            await extractor.extract(mock_question)

        # Second call should hit cache -- only 2 invocations total (one S4, one S5)
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_strategy_simple_skips_llm(self, mock_question):
        call_count = {"n": 0}

        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                call_count["n"] += 1
                return "x"  # noqa: ASYNC910

        with patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm):
            extractor = pmp.KeywordExtractor(strategy="simple")
            queries = await extractor.extract(mock_question)

        assert call_count["n"] == 0
        assert any("Starship" in q for q in queries)

    def test_config_pins_allowed_tries_to_one(self):
        """The keyword LLM must not carry forecasting-tools' own retry budget.

        Left unpinned it inherits ``allowed_tries=2`` with an UN-GATED
        ``random.uniform(5, 10)`` tenacity sleep — the exact un-gated retry
        ``llm_retry`` exists to replace, and a 5-10s blind sleep is a third of the
        30s ``PREDICTION_MARKET_TIMEOUT`` the whole snapshot runs under. Pinning to 1
        makes the gated wrapper the SOLE retry layer.
        """
        assert PREDICTION_MARKET_KEYWORD_LLM_CONFIG["allowed_tries"] == 1

    @pytest.mark.asyncio
    async def test_extract_routes_through_gated_retry_wrapper(self, mock_question):
        """Keyword extraction goes through ``invoke_with_transient_retry``.

        Asserts the wrapper is actually in the call path (label + bounded backoffs),
        not merely imported.
        """
        captured: list[dict] = []

        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                return "starship orbit"  # noqa: ASYNC910

        async def _spy(make_awaitable, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(kwargs)
            return await make_awaitable()

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm),
            patch.object(pmp, "invoke_with_transient_retry", _spy),
        ):
            queries = await pmp.KeywordExtractor(strategy="s4_s5_union").extract(mock_question)

        assert queries
        assert len(captured) == 2  # S4 + S5 each wrapped
        assert all(kw["label"] == "prediction_market_keywords" for kw in captured)
        # Backoffs must stay inside the snapshot budget: the default 1s/10s/30s
        # ladder alone exceeds PREDICTION_MARKET_TIMEOUT.
        assert all(sum(kw["backoffs"]) < PREDICTION_MARKET_TIMEOUT for kw in captured)
        assert all(kw["wall_timeout"] <= PREDICTION_MARKET_TIMEOUT for kw in captured)
        # The real ceiling, which neither assertion above can see: ``wall_timeout`` is
        # PER ATTEMPT, so the worst case is (attempts x wall) + sum(backoffs). At the
        # original 15.0 that came to 2 x 15 + 1 = 31s against a 30s
        # PREDICTION_MARKET_TIMEOUT, leaving -1s for the four-venue fan-out the budget
        # exists to protect — yet both assertions above pass at 15.0 and at 12.0 alike,
        # which is how the arithmetic bug survived them. Assert the product, and keep a
        # floor for the fan-out so a future bump cannot silently reclaim it.
        for kw in captured:
            attempts = len(kw["backoffs"]) + 1
            worst_case = attempts * kw["wall_timeout"] + sum(kw["backoffs"])
            fan_out_margin = PREDICTION_MARKET_TIMEOUT - worst_case
            assert fan_out_margin >= 5.0, (
                f"keyword extraction can consume {worst_case}s of the {PREDICTION_MARKET_TIMEOUT}s "
                f"snapshot budget ({attempts} attempts x {kw['wall_timeout']}s wall + "
                f"{sum(kw['backoffs'])}s backoff), leaving {fan_out_margin}s for the venue fan-out"
            )

    @pytest.mark.asyncio
    async def test_extract_does_not_retry_a_hard_403(self, mock_question):
        """A drained-key 403 costs the extractor ONE attempt per call site, not two.

        The 2026-07-26 run logged two keyword failures 1s apart, each having already
        burned an ungated ~7s tenacity sleep on a deterministic rejection. With the
        status-aware predicate the wrapper gives up immediately and the extractor
        soft-fails to "" as designed.
        """
        attempts = {"n": 0}

        class DeadLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                attempts["n"] += 1
                raise litellm.exceptions.APIError(
                    status_code=403,
                    message='{"error":{"message":"Key limit exceeded (total limit).","code":403}}',
                    llm_provider="openrouter",
                    model="openai/gpt-5.6-luna",
                )

        with patch.object(pmp, "build_llm_with_openrouter_fallback", DeadLlm):
            queries = await pmp.KeywordExtractor(strategy="s4_s5_union").extract(mock_question)

        assert queries == []
        assert attempts["n"] == 2  # one attempt each for S4 and S5, no retries


# ---------------------------------------------------------------------------
# fetch_market_snapshot orchestrator tests
# ---------------------------------------------------------------------------


class TestFetchMarketSnapshot:
    @pytest.mark.asyncio
    async def test_full_orchestrator_returns_matches(
        self, mock_question, polymarket_payload, manifold_payload, kalshi_events_payload, predictit_payload
    ):
        handlers = {
            "https://gamma-api.polymarket.com/public-search": FakeResponse(200, polymarket_payload),
            "https://api.manifold.markets/v0/search-markets": FakeResponse(200, manifold_payload),
            "https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(200, kalshi_events_payload),
            "https://api.elections.kalshi.com/trade-api/v2/series": FakeResponse(200, {"series": []}),
            "https://www.predictit.org/api/marketdata/all/": FakeResponse(200, predictit_payload),
        }

        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                return "Starship orbit 2026"  # noqa: ASYNC910

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        # Should find at least one match from each of the four platforms. PredictIt's
        # presence exercises the dedup/assembly footgun literals (by_platform,
        # seen_urls_per_platform, and the assembly-loop tuple).
        platforms = {m.platform for m in snapshot.matches}
        assert "polymarket" in platforms
        assert "kalshi" in platforms
        assert "manifold" in platforms
        assert "predictit" in platforms

    @pytest.mark.asyncio
    async def test_as_of_filter_drops_post_as_of_matches(self, mock_question):
        """Backtest leakage defense: matches that closed AFTER as_of are dropped.

        Resolved prediction markets keep a 'last trade' price after the question
        settled; using that during a resolved-question backtest is leakage.
        """

        # Kalshi event that closes 2026-12-31 -- after as_of 2026-05-01
        late_close_payload = {
            "events": [
                {
                    "event_ticker": "KXSTAR-26",
                    "title": "Will SpaceX Starship reach orbit in 2026?",
                    "markets": [
                        {
                            "title": "Will SpaceX Starship reach orbit in 2026?",
                            "rules_primary": "orbital velocity 2026",
                            "yes_bid_dollars": "0.60",
                            "yes_ask_dollars": "0.70",
                            "close_time": "2026-12-31T23:59:59Z",
                        }
                    ],
                }
            ],
            "cursor": "",
        }

        handlers = {
            "https://gamma-api.polymarket.com/public-search": FakeResponse(200, {"events": [], "markets": []}),
            "https://api.manifold.markets/v0/search-markets": FakeResponse(200, []),
            "https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(200, late_close_payload),
            "https://api.elections.kalshi.com/trade-api/v2/series": FakeResponse(200, {"series": []}),
            "https://www.predictit.org/api/marketdata/all/": FakeResponse(200, {"markets": []}),
        }

        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                return "Starship orbit"  # noqa: ASYNC910

        as_of = datetime(2026, 5, 1, tzinfo=timezone.utc)

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot_open = await pmp.fetch_market_snapshot(mock_question, as_of=as_of, timeout=5.0)

        # close_time=2026-12-31 is AFTER as_of=2026-05-01 so the market is still open
        # at as_of -- should remain in the snapshot.
        assert any(m.platform == "kalshi" for m in snapshot_open.matches)

        # Now flip it: close_time BEFORE as_of -> resolved-before-as-of -> drop.
        closed_payload = {
            "events": [
                {
                    "event_ticker": "KXSTAR-PAST",
                    "title": "Will SpaceX Starship reach orbit in 2026?",
                    "markets": [
                        {
                            "title": "Starship orbit",
                            "rules_primary": "orbital velocity",
                            "yes_bid_dollars": "0.60",
                            "yes_ask_dollars": "0.70",
                            "close_time": "2026-04-01T00:00:00Z",  # BEFORE as_of
                        }
                    ],
                }
            ],
            "cursor": "",
        }

        handlers2 = {
            "https://gamma-api.polymarket.com/public-search": FakeResponse(200, {"events": [], "markets": []}),
            "https://api.manifold.markets/v0/search-markets": FakeResponse(200, []),
            "https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(200, closed_payload),
            "https://api.elections.kalshi.com/trade-api/v2/series": FakeResponse(200, {"series": []}),
            "https://www.predictit.org/api/marketdata/all/": FakeResponse(200, {"markets": []}),
        }
        # Fresh cache
        pmp._reset_session_caches()

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers2)),
        ):
            snapshot2 = await pmp.fetch_market_snapshot(mock_question, as_of=as_of, timeout=5.0)

        assert len(snapshot2.matches) == 0, "Markets closed at/before as_of must be filtered"

    @pytest.mark.asyncio
    async def test_timeout_returns_empty_snapshot_soft_fail(self, mock_question, caplog):
        """A per-question timeout must NOT raise -- soft-fail with empty snapshot + WARNING."""

        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                await asyncio.sleep(10)
                return "..."

        with patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm):
            with caplog.at_level(logging.WARNING):
                snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=0.05)

        assert snapshot.matches == []
        # Soft-fail, but not SILENT: a dead whole snapshot carries a loss token and
        # reddens CI, where an empty `sources` map rendered no diagnostics suffix.
        assert snapshot.sources == {"snapshot": "error(timeout)"}
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_orchestrator_soft_fails_on_any_platform_error(self, mock_question, manifold_payload, caplog):
        """If one platform errors, other platforms' matches still come through."""

        def _boom(_params):
            raise RuntimeError("connection refused")

        handlers = {
            "https://gamma-api.polymarket.com/public-search": _boom,
            "https://api.manifold.markets/v0/search-markets": FakeResponse(200, manifold_payload),
            "https://api.elections.kalshi.com/trade-api/v2/events": _boom,
            "https://www.predictit.org/api/marketdata/all/": _boom,
        }

        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                return "Starship orbit"  # noqa: ASYNC910

        pmp._reset_session_caches()
        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
            caplog.at_level(logging.WARNING),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        platforms = {m.platform for m in snapshot.matches}
        assert "manifold" in platforms  # manifold still worked
        assert "polymarket" not in platforms

    @pytest.mark.asyncio
    async def test_snapshot_cache_key_includes_as_of(
        self, mock_question, polymarket_payload, manifold_payload, kalshi_events_payload
    ):
        """F1/F7: snapshot cache keyed by (qid, as_of_iso) so a backtest at
        as_of=A doesn't reuse a snapshot computed at as_of=B."""
        handlers = {
            "https://gamma-api.polymarket.com/public-search": FakeResponse(200, polymarket_payload),
            "https://api.manifold.markets/v0/search-markets": FakeResponse(200, manifold_payload),
            "https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(200, kalshi_events_payload),
            "https://api.elections.kalshi.com/trade-api/v2/series": FakeResponse(200, {"series": []}),
            "https://www.predictit.org/api/marketdata/all/": FakeResponse(200, {"markets": []}),
        }

        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                return "Starship orbit"  # noqa: ASYNC910

        as_of_a = datetime(2026, 5, 1, tzinfo=timezone.utc)
        as_of_b = datetime(2026, 6, 1, tzinfo=timezone.utc)

        pmp._reset_session_caches()
        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snap_a = await pmp.fetch_market_snapshot(mock_question, as_of=as_of_a, timeout=5.0)
            snap_b = await pmp.fetch_market_snapshot(mock_question, as_of=as_of_b, timeout=5.0)

        # Two distinct cache entries should exist, one per (qid, as_of_iso).
        assert (mock_question.id_of_question, as_of_a.isoformat()) in pmp._SNAPSHOT_CACHE
        assert (mock_question.id_of_question, as_of_b.isoformat()) in pmp._SNAPSHOT_CACHE
        # Both calls should have produced a snapshot (not collided as None).
        assert snap_a is not None and snap_b is not None

    @pytest.mark.asyncio
    async def test_max_matches_per_platform_respected(
        self, mock_question, polymarket_payload, manifold_payload, kalshi_events_payload
    ):
        handlers = {
            "https://gamma-api.polymarket.com/public-search": FakeResponse(200, polymarket_payload),
            "https://api.manifold.markets/v0/search-markets": FakeResponse(200, manifold_payload),
            "https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(200, kalshi_events_payload),
            "https://api.elections.kalshi.com/trade-api/v2/series": FakeResponse(200, {"series": []}),
            "https://www.predictit.org/api/marketdata/all/": FakeResponse(200, {"markets": []}),
        }

        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                return "Starship orbit"  # noqa: ASYNC910

        pmp._reset_session_caches()
        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, max_matches_per_platform=1, timeout=5.0)

        # At most 1 match per platform

        counts = Counter(m.platform for m in snapshot.matches)
        assert all(v <= 1 for v in counts.values()), counts


# ---------------------------------------------------------------------------
# prediction_market_provider factory (research-provider flag gating)
# ---------------------------------------------------------------------------


class TestProviderFactory:
    @pytest.mark.asyncio
    async def test_disabled_flag_returns_empty_at_orchestrator_level(self, monkeypatch, mock_question):
        """When PREDICTION_MARKETS_ENABLED is not set, the provider returns ''.
        This is a defense-in-depth check at the research-provider entrypoint."""

        monkeypatch.delenv("PREDICTION_MARKETS_ENABLED", raising=False)

        provider = prediction_market_provider()
        # The provider takes a MetaculusQuestion. With the flag disabled it
        # short-circuits to empty without touching the question.
        result = await provider(mock_question)
        assert result == ""

    @pytest.mark.asyncio
    async def test_is_benchmarking_short_circuits_regardless_of_env_flag(self, monkeypatch, mock_question):
        """F7: is_benchmarking=True must hard-disable the provider, even when
        PREDICTION_MARKETS_ENABLED is set in the operator's environment.

        The ``as_of`` filter only drops markets that closed BEFORE ``as_of``;
        markets that are still open OR closed between ``as_of`` and now leak
        post-``as_of`` information into a backtest. The benchmarking guard is
        the only safe defense — any other path risks data leakage. This
        mirrors the contract used by ``gemini_search_provider`` and
        ``native_search_provider``.
        """

        # Set the env flag so we'd otherwise enable the provider — the
        # is_benchmarking guard must override.
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")

        provider = prediction_market_provider(is_benchmarking=True)
        result = await provider(mock_question)
        assert result == "", "is_benchmarking=True must short-circuit to '' regardless of env flag"

    @pytest.mark.asyncio
    async def test_is_benchmarking_default_false_preserves_existing_behavior(self, monkeypatch, mock_question):
        """F7 control: factory called without is_benchmarking still gates only on env flag.

        Pins that adding the new param doesn't accidentally flip the default
        behavior (which would silence the provider in prod where
        is_benchmarking is False).
        """

        monkeypatch.delenv("PREDICTION_MARKETS_ENABLED", raising=False)

        # No is_benchmarking arg → defaults to False. Env flag off → empty.
        provider = prediction_market_provider()
        result = await provider(mock_question)
        assert result == ""

    @pytest.mark.asyncio
    async def test_factory_derives_as_of_from_scheduled_resolution(
        self, monkeypatch, mock_question, polymarket_payload, manifold_payload, kalshi_events_payload
    ):
        """F1: when the question carries scheduled_resolution_time, the
        provider derives as_of from it (with a small backward buffer) so the
        leakage filter is active even when the orchestrator caller didn't
        supply as_of explicitly."""
        # Set scheduled_resolution_time so the provider derives a real as_of.
        mock_question.scheduled_resolution_time = datetime(2026, 8, 1, tzinfo=timezone.utc)

        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")

        handlers = {
            "https://gamma-api.polymarket.com/public-search": FakeResponse(200, polymarket_payload),
            "https://api.manifold.markets/v0/search-markets": FakeResponse(200, manifold_payload),
            "https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(200, kalshi_events_payload),
            "https://api.elections.kalshi.com/trade-api/v2/series": FakeResponse(200, {"series": []}),
            "https://www.predictit.org/api/marketdata/all/": FakeResponse(200, {"markets": []}),
        }

        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                return "Starship orbit"  # noqa: ASYNC910

        captured_as_ofs: list[datetime | None] = []
        original_fetch = pmp.fetch_market_snapshot

        async def _capturing_fetch(question_arg, *, as_of=None, **kwargs):
            captured_as_ofs.append(as_of)
            return await original_fetch(question_arg, as_of=as_of, **kwargs)

        pmp._reset_session_caches()
        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
            patch.object(pmp, "fetch_market_snapshot", _capturing_fetch),
        ):
            provider = pmp.prediction_market_provider()
            await provider(mock_question)

        assert len(captured_as_ofs) == 1
        derived = captured_as_ofs[0]
        # Derived as_of should be slightly before scheduled_resolution_time
        # (backward buffer applied).
        assert derived is not None
        assert derived < mock_question.scheduled_resolution_time

    @pytest.mark.asyncio
    async def test_enabled_flag_fetches_and_formats(
        self, monkeypatch, mock_question, polymarket_payload, manifold_payload, kalshi_events_payload
    ):
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")

        handlers = {
            "https://gamma-api.polymarket.com/public-search": FakeResponse(200, polymarket_payload),
            "https://api.manifold.markets/v0/search-markets": FakeResponse(200, manifold_payload),
            "https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(200, kalshi_events_payload),
            "https://api.elections.kalshi.com/trade-api/v2/series": FakeResponse(200, {"series": []}),
            "https://www.predictit.org/api/marketdata/all/": FakeResponse(200, {"markets": []}),
        }

        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                return "Starship orbit"  # noqa: ASYNC910

        pmp._reset_session_caches()
        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", FakeLlm),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            provider = pmp.prediction_market_provider()
            out = await provider(mock_question)

        assert isinstance(out, str)
        assert "MAY be relevant" in out
        assert "polymarket" in out.lower() or "manifold" in out.lower() or "kalshi" in out.lower()


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def _fmt_match(title: str, rules: str, confidence: float, *, platform: str = "polymarket") -> MarketMatch:
    """Minimal MarketMatch for formatter/relevance tests."""
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
        match_confidence=confidence,
        raw_rules=rules,
        total_volume=100_000.0,
        open_interest=50_000.0,
    )


class TestRelevanceGate:
    """Piece 1: per-row relevance labels + conditional strong/neutral preamble."""

    _SPACEX_Q = "Will SpaceX Starship reach orbit in 2026?"
    _SPACEX_RC = "Resolves Yes if a SpaceX Starship reaches orbital velocity in 2026."

    def test_all_junk_gets_neutral_preamble_and_verify_labels(self):
        # Off-topic contract (zero content-word overlap, low conf) against a Fed-rates question.
        snap = MarketSnapshot(matches=[_fmt_match("Carnival Cruise lower berth days", "berth-day count", 0.42)])
        out = format_snapshot_for_research(
            snap,
            question_title="Will the Federal Reserve cut interest rates in September 2026?",
            resolution_criteria="Resolves Yes if the FOMC lowers the target range at its September meeting.",
        )
        # Neutral framing, NOT the strong-evidence framing.
        assert "may all be off-topic" in out.lower()
        assert "leads to verify, not as evidence" in out.lower()
        assert "extremely strong evidence" not in out.lower()
        assert "anchor on its price" not in out.lower()
        # Contract still rendered, labelled verify-carefully. (Check the pipe-delimited table
        # cell, not the whole string — the legend text names both labels.)
        assert "Carnival Cruise lower berth days" in out
        assert "| verify-carefully |" in out
        assert "| likely-relevant |" not in out

    def test_relevant_contract_fires_strong_preamble_with_per_row_labels(self):
        # One strongly-overlapping contract (clears the bar) + one off-topic contract (does not).
        snap = MarketSnapshot(
            matches=[
                _fmt_match(self._SPACEX_Q, "Starship reaches orbital velocity in 2026.", 0.88),
                _fmt_match("Carnival Cruise lower berth days", "berth-day count", 0.40, platform="kalshi"),
            ]
        )
        out = format_snapshot_for_research(snap, question_title=self._SPACEX_Q, resolution_criteria=self._SPACEX_RC)
        # >=1 clears -> strong preamble.
        assert "extremely strong evidence" in out.lower()
        assert "anchor on its price" in out.lower()
        # Both contracts rendered; per-row labels differ (pipe-delimited cells).
        assert "| likely-relevant |" in out
        assert "| verify-carefully |" in out
        assert "Carnival Cruise lower berth days" in out

    def test_high_overlap_but_low_conf_stays_verify_carefully(self):
        # Overlap clears but conf < MARKET_RELEVANCE_CONF_MIN -> not likely-relevant -> neutral.
        snap = MarketSnapshot(matches=[_fmt_match(self._SPACEX_Q, "Starship orbital velocity 2026.", 0.30)])
        out = format_snapshot_for_research(snap, question_title=self._SPACEX_Q, resolution_criteria=self._SPACEX_RC)
        assert "| verify-carefully |" in out
        assert "| likely-relevant |" not in out
        assert "may all be off-topic" in out.lower()

    def test_no_question_context_defaults_to_verify_carefully(self):
        # Without question context overlap can't be computed -> conservative default (neutral + verify).
        snap = MarketSnapshot(matches=[_fmt_match(self._SPACEX_Q, "Starship orbital velocity 2026.", 0.99)])
        out = format_snapshot_for_research(snap)
        assert "| verify-carefully |" in out
        assert "| likely-relevant |" not in out
        assert "may all be off-topic" in out.lower()
        # Contract still rendered.
        assert self._SPACEX_Q in out


class TestFormatter:
    def test_formatter_emits_caveat_and_columns(self):
        snap = MarketSnapshot(
            matches=[
                MarketMatch(
                    platform="polymarket",
                    market_title="Will SpaceX Starship reach orbit in 2026?",
                    market_url="https://polymarket.com/event/foo",
                    implied_prob_yes=0.74,
                    bid=0.73,
                    ask=0.75,
                    spread=0.02,
                    volume_24h=12500.0,
                    close_time=datetime(2026, 12, 31, tzinfo=timezone.utc),
                    is_resolved=False,
                    match_confidence=0.85,
                    raw_rules="If Starship achieves orbital velocity in 2026 per SpaceX confirmation.",
                    total_volume=987654.0,
                    liquidity=45000.0,
                    open_interest=60000.0,
                )
            ]
        )

        # Question context that strongly overlaps the match (spacex/starship/orbit/orbital/velocity)
        # so the contract clears the relevance bar and the STRONG-evidence preamble fires.
        formatted = format_snapshot_for_research(
            snap,
            question_title="Will SpaceX Starship reach orbit in 2026?",
            resolution_criteria="Resolves Yes if a SpaceX Starship reaches orbital velocity in 2026.",
        )

        # The strong-evidence header is the load-bearing framing when >=1 contract clears the bar:
        # the fuzzy match may be off-topic, so relevance must be verified before weighting — a
        # criteria+date match is extremely strong evidence, a mismatch is discounted proportionally.
        assert "MAY be relevant" in formatted
        assert "verify each market's resolution criteria" in formatted.lower()
        assert "extremely strong evidence" in formatted.lower()
        assert "anchor on its price" in formatted.lower()
        assert "resolution date" in formatted.lower()
        assert "name the specific mismatch and discount accordingly" in formatted.lower()
        assert "worth little or nothing" in formatted.lower()
        # The old unconditional header must be gone.
        assert "weight these markets heavily" not in formatted.lower()
        # New columns per the plan: total_vol + OI + signal replace the misleading 24h vol column,
        # plus the per-row relevance label.
        assert "platform" in formatted.lower()
        assert "total_vol" in formatted.lower()
        assert "| oi |" in formatted.lower()
        assert "signal" in formatted.lower()
        assert "| relevance |" in formatted.lower()
        assert "likely-relevant" in formatted
        assert "polymarket" in formatted.lower()
        assert "0.74" in formatted
        # total_volume=987654 rendered (not the 12500 24h vol), open_interest=60000, deep signal.
        assert "987654" in formatted
        assert "60000" in formatted
        assert "deep" in formatted.lower()
        # Raw rules included
        assert "orbital velocity" in formatted.lower()

    def test_formatter_empty_snapshot_returns_empty_string(self):
        assert format_snapshot_for_research(MarketSnapshot(matches=[])) == ""

    def test_formatter_truncates_long_raw_rules(self):
        long_rules = "a" * 2000
        snap = MarketSnapshot(
            matches=[
                MarketMatch(
                    platform="kalshi",
                    market_title="x",
                    market_url="https://kalshi.com/x",
                    implied_prob_yes=0.5,
                    bid=None,
                    ask=None,
                    spread=None,
                    volume_24h=None,
                    close_time=None,
                    is_resolved=False,
                    match_confidence=0.5,
                    raw_rules=long_rules,
                )
            ]
        )
        formatted = format_snapshot_for_research(snap)
        # Should NOT contain the full 2000-char string
        assert "a" * 500 not in formatted  # i.e. truncated well below full length

    @pytest.mark.parametrize(
        ("total_volume", "expected"),
        [(1_000.0, "thin"), (20_000.0, "decent"), (100_000.0, "deep")],
    )
    def test_liquidity_label_real_money_thresholds(self, total_volume, expected):
        m = MarketMatch(
            platform="polymarket",
            market_title="x",
            market_url="",
            implied_prob_yes=0.5,
            bid=None,
            ask=None,
            spread=None,
            volume_24h=None,
            close_time=None,
            is_resolved=False,
            match_confidence=0.5,
            raw_rules="",
            total_volume=total_volume,
        )
        assert _liquidity_label(m) == expected

    @pytest.mark.parametrize(
        ("num_bettors", "expected"),
        [(5, "thin"), (50, "decent"), (200, "high")],
    )
    def test_liquidity_label_manifold_bettor_thresholds(self, num_bettors, expected):
        m = MarketMatch(
            platform="manifold",
            market_title="x",
            market_url="",
            implied_prob_yes=0.5,
            bid=None,
            ask=None,
            spread=None,
            volume_24h=None,
            close_time=None,
            is_resolved=False,
            match_confidence=0.5,
            raw_rules="",
            num_bettors=num_bettors,
        )
        assert _liquidity_label(m) == expected

    def test_formatter_all_none_predictit_row_renders_no_liquidity_data(self):
        snap = MarketSnapshot(
            matches=[
                MarketMatch(
                    platform="predictit",
                    market_title="Will X win?",
                    market_url="https://www.predictit.org/markets/detail/1/x",
                    implied_prob_yes=0.4,
                    bid=None,
                    ask=None,
                    spread=None,
                    volume_24h=None,
                    close_time=None,
                    is_resolved=False,
                    match_confidence=0.6,
                    raw_rules="Will X win?",
                )
            ]
        )
        formatted = format_snapshot_for_research(snap)
        assert "predictit" in formatted.lower()
        assert "no-liquidity-data" in formatted
        # total_vol and OI render as "-" for an all-None row.
        row = [ln for ln in formatted.splitlines() if ln.startswith("| predictit")][0]
        cells = [c.strip() for c in row.split("|")]
        # cells: ['', platform, title, prob, total_vol, OI, signal, close, conf, relevance, '']
        assert cells[4] == "-"  # total_vol
        assert cells[5] == "-"  # OI
        assert cells[6] == "no-liquidity-data"  # signal
        assert cells[9] == "verify-carefully"  # relevance (no question context -> conservative default)


# ---------------------------------------------------------------------------
# Kalshi entity-based series retrieval (Piece 2)
# ---------------------------------------------------------------------------


class TestKalshiEntityRetrieval:
    def test_extract_title_entities_captures_awards_and_names(self):
        q = MagicMock()
        q.title = "Will an athlete win the ESPY for Best Male Athlete in 2026?"
        q.question_text = q.title
        entities = _extract_title_entities(q)
        # Acronym + proper-noun phrase captured; scaffolding (Will/the/for/in/month) dropped.
        assert "ESPY" in entities
        assert "Best Male Athlete" in entities
        assert "Will" not in entities

    def test_extract_title_entities_quoted_span_and_dedup(self):
        q = MagicMock()
        q.title = 'Will the film "Toy Story 5" gross over $200M and win a BET Award?'
        q.question_text = q.title
        entities = _extract_title_entities(q)
        assert "Toy Story 5" in entities
        assert "BET Award" in entities

    def test_match_entities_to_series_matches_by_title(self):
        series = [
            {"ticker": "KXESPYS", "title": "ESPY Awards Best Male Athlete", "tags": ["espy", "awards"]},
            {"ticker": "KXHIGHNY", "title": "Weekly high temperature in NYC", "tags": []},
        ]
        tickers = _match_entities_to_series(["Best Male Athlete", "ESPY"], series)
        assert tickers == ["KXESPYS"]

    @pytest.mark.asyncio
    async def test_entity_retrieval_finds_series_pure_fuzzy_missed(self):
        """ESPY-style title: the exact ESPY event is ABSENT from the prefetch dump (only junk
        there), so pure fuzzy-over-prefetch misses it. Entity retrieval matches the series and
        fetches the event by series_ticker, surfacing the exact market."""
        q = MagicMock()
        q.title = "Will an athlete win the ESPY for Best Male Athlete in 2026?"
        q.question_text = q.title

        # Prefetch dump: only an off-topic event a pure fuzzy match would surface.
        prefetch_events = [
            {
                "event_ticker": "KXCRUISE",
                "title": "Carnival Cruise lower berth days",
                "markets": [{"ticker": "KXCRUISE-Y", "title": "berth days", "rules_primary": "berth-day count"}],
            }
        ]
        espy_event = {
            "event_ticker": "KXESPYS-26-BMA",
            "title": "ESPY Award: Best Male Athlete 2026",
            "markets": [
                {
                    "ticker": "KXESPYS-26-BMA-Y",
                    "title": "ESPY Best Male Athlete 2026",
                    "rules_primary": "Resolves to the athlete who wins the 2026 ESPY for Best Male Athlete.",
                    "yes_bid_dollars": "0.30",
                    "yes_ask_dollars": "0.34",
                    "close_time": "2026-12-31T23:59:59Z",
                }
            ],
        }
        series_payload = {
            "series": [
                {"ticker": "KXESPYS", "title": "ESPY Awards Best Male Athlete", "tags": ["espy"]},
                {"ticker": "KXHIGHNY", "title": "Weekly high temperature in NYC", "tags": []},
            ]
        }

        def _events_handler(params: dict[str, Any]) -> FakeResponse:
            if params.get("series_ticker") == "KXESPYS":
                return FakeResponse(200, {"events": [espy_event], "cursor": ""})
            return FakeResponse(200, {"events": prefetch_events, "cursor": ""})

        handlers = {
            "https://api.elections.kalshi.com/trade-api/v2/series": FakeResponse(200, series_payload),
            "https://api.elections.kalshi.com/trade-api/v2/events": _events_handler,
        }
        session = FakeSession(handlers)

        # A generic query that would NOT fuzzy-match the ESPY event over the prefetch dump.
        query = "athlete nominee 2026"

        # Pure fuzzy over the prefetch dump surfaces nothing about the ESPY market.
        fuzzy_only = _kalshi_search_local(prefetch_events, query, top_k=5, min_score=40.0)
        assert not any("espy" in m.market_title.lower() for m in fuzzy_only)

        # Entity retrieval surfaces the exact ESPY market via /series -> /events?series_ticker.
        entity_matches, _tally = await _kalshi_entity_matches(session, q, [query], top_k=5)
        assert any("espy" in m.market_title.lower() for m in entity_matches)
        assert all(m.platform == "kalshi" for m in entity_matches)

    @pytest.mark.asyncio
    async def test_entity_retrieval_soft_fails_when_no_series_match(self):
        """No series matches the entities -> no /events call, empty result (no raise)."""
        q = MagicMock()
        q.title = "Will an athlete win the ESPY for Best Male Athlete in 2026?"
        q.question_text = q.title
        handlers = {
            "https://api.elections.kalshi.com/trade-api/v2/series": FakeResponse(
                200, {"series": [{"ticker": "KXHIGHNY", "title": "Weekly high temperature in NYC", "tags": []}]}
            ),
        }
        session = FakeSession(handlers)
        matches, tally = await _kalshi_entity_matches(session, q, ["athlete"], top_k=5)
        assert matches == []
        # No series matched -> no event fetch was attempted, so nothing is lost either.
        assert (tally.ok, tally.failed) == (0, 0)


# ---------------------------------------------------------------------------
# Kalshi /series streaming fetch (2026-07-25 fix: the unpaginated series body
# crossed the shared 10 MiB read cap and silently zeroed the entity-recall path;
# we now stream-parse it with a generous safety ceiling + a degradation counter)
# ---------------------------------------------------------------------------

_KALSHI_SERIES_URL = "https://api.elections.kalshi.com/trade-api/v2/series"


def _big_series_payload(n_series: int, junk_chars: int) -> dict:
    """A series catalogue whose serialized JSON is dominated by fields the
    extractor DROPS (mirrors the real payload, where per-series contract terms /
    settlement sources are the bulk we don't need for entity matching)."""
    junk = "x" * junk_chars
    series: list[dict] = [
        {
            "ticker": f"KX{i:05d}",
            "title": f"Series number {i} about topic {i}",
            "category": "Politics",
            "tags": [f"tag{i}"],
            "contract_terms_url": junk,  # heavy field, not retained
            "settlement_sources": [junk],  # heavy field, not retained
        }
        for i in range(n_series)
    ]
    series.append({"ticker": "KXESPYS", "title": "ESPY Awards Best Male Athlete", "tags": ["espy"]})
    return {"series": series}


class TestKalshiSeriesStreaming:
    @pytest.mark.asyncio
    async def test_streams_catalogue_larger_than_old_10mib_cap(self):
        """Regression for the 2026-07-25 prod failure: a /series body LARGER than
        the old 10 MiB read cap must parse fully (the old buffered read dropped
        it), and only {ticker,title,category,tags} are retained per series."""
        payload = _big_series_payload(n_series=1400, junk_chars=8000)
        # Precondition: this catalogue really is bigger than the old cap that dropped it.
        assert len(json.dumps(payload).encode()) > pmp.MAX_RESPONSE_BYTES

        session = FakeSession({_KALSHI_SERIES_URL: FakeResponse(200, payload)})
        series = await pmp._kalshi_prefetch_series(session)

        assert len(series) == 1401  # 1400 filler + the ESPY target
        assert pmp.kalshi_series_fetch_failures() == 0
        # Heavy fields dropped; only the four entity-matching fields retained.
        assert set(series[0].keys()) == {"ticker", "title", "category", "tags"}
        # Entity matching still finds the target series in the streamed result.
        assert _match_entities_to_series(["ESPY", "Best Male Athlete"], series) == ["KXESPYS"]

    @pytest.mark.asyncio
    async def test_over_ceiling_returns_empty_and_counts_degradation(self, monkeypatch, caplog):
        """When the generous safety ceiling trips, the fetch aborts LOUD + COUNTED
        (a degradation counter that reddens CI), never a silent drop."""
        monkeypatch.setattr(pmp, "KALSHI_SERIES_MAX_BYTES", 500)
        payload = {"series": [{"ticker": f"KX{i}", "title": f"t{i}"} for i in range(200)]}
        assert len(json.dumps(payload).encode()) > 500

        session = FakeSession({_KALSHI_SERIES_URL: FakeResponse(200, payload)})
        with caplog.at_level(logging.WARNING):
            series = await pmp._kalshi_prefetch_series(session)

        assert series == []
        assert pmp.kalshi_series_fetch_failures() == 1
        assert any("ceiling" in r.getMessage().lower() for r in caplog.records if r.levelno == logging.WARNING)
        # A failure is NOT cached, so the next question re-attempts.
        assert "series" not in pmp._KALSHI_CACHE

    @pytest.mark.asyncio
    async def test_http_error_counts_degradation(self, caplog):
        """A non-200 on /series is the exact 'provider silently dead' hole: it
        must bump the degradation counter (the run reported timeouts=0 while the
        path was dead)."""
        session = FakeSession({_KALSHI_SERIES_URL: FakeResponse(503, text="upstream boom")})
        with caplog.at_level(logging.WARNING):
            series = await pmp._kalshi_prefetch_series(session)

        assert series == []
        assert pmp.kalshi_series_fetch_failures() == 1
        assert any("kalshi series" in r.getMessage().lower() for r in caplog.records if r.levelno == logging.WARNING)

    @pytest.mark.asyncio
    async def test_malformed_json_counts_degradation(self, caplog):
        """A truncated/garbled body must fail loud + counted, not parse into junk."""
        truncated = b'{"series": [ {"ticker": "KXA", "title": "Alpha"'
        session = FakeSession({_KALSHI_SERIES_URL: FakeResponse(200, payload=None, raw_content=truncated)})
        with caplog.at_level(logging.WARNING):
            series = await pmp._kalshi_prefetch_series(session)

        assert series == []
        assert pmp.kalshi_series_fetch_failures() == 1
        # "Loud" is the whole point of the counted path — assert the warning, not just the count.
        assert any("parse failed" in r.getMessage().lower() for r in caplog.records if r.levelno == logging.WARNING), (
            "a garbled body must log WHY it was dropped"
        )

    @pytest.mark.asyncio
    async def test_transient_error_counts_degradation(self, caplog, monkeypatch):
        """A connection reset on every attempt must soft-fail to [] + a counted
        degradation, never escape into the provider (the fuzzy-over-events pass
        still has to run)."""
        monkeypatch.setattr(pmp, "HTTP_RETRY_BACKOFF_SECS", 0.0)

        class _ExplodingSession:
            def get(self, *_args: Any, **_kwargs: Any) -> Any:
                raise aiohttp.ClientConnectionError("connection reset by peer")

        with caplog.at_level(logging.WARNING):
            series = await pmp._kalshi_prefetch_series(_ExplodingSession())

        assert series == []
        assert pmp.kalshi_series_fetch_failures() == 1
        assert any("transient error" in r.getMessage().lower() for r in caplog.records if r.levelno == logging.WARNING)

    @pytest.mark.asyncio
    async def test_missing_series_array_is_a_counted_failure(self, caplog):
        """An HTTP 200 carrying `{"error": ...}` yields zero series items, exactly
        like an empty catalogue. Streaming can only tell them apart by watching for
        the top-level `series` ARRAY; without that the error body is cached as a
        valid empty index for 6h and entity matching dies silently."""
        session = FakeSession({_KALSHI_SERIES_URL: FakeResponse(200, {"error": "temporarily unavailable"})})
        sink: dict[str, str] = {}
        with caplog.at_level(logging.WARNING):
            series = await pmp._kalshi_prefetch_series(session, outcome_sink=sink)

        assert series == []
        assert pmp.kalshi_series_fetch_failures() == 1
        assert sink["kalshi_series"] == "error(no_series_array)"  # a LOST source, not benign `none`
        assert "series" not in pmp._KALSHI_CACHE  # never cached, so the next question re-attempts
        assert any("no top-level" in r.getMessage().lower() for r in caplog.records if r.levelno == logging.WARNING)

    @pytest.mark.asyncio
    async def test_non_array_series_is_a_counted_failure(self):
        """`{"series": null}` is the same hole as a missing key: present but unusable."""
        session = FakeSession({_KALSHI_SERIES_URL: FakeResponse(200, {"series": None})})
        sink: dict[str, str] = {}
        series = await pmp._kalshi_prefetch_series(session, outcome_sink=sink)

        assert series == []
        assert pmp.kalshi_series_fetch_failures() == 1
        assert sink["kalshi_series"] == "error(no_series_array)"

    @pytest.mark.asyncio
    async def test_transient_status_is_retried_and_can_succeed(self, monkeypatch):
        """A 503 that would have succeeded on retry must not silently drop entity
        recall for the question (the streaming rewrite dropped the old max_attempts=2)."""
        monkeypatch.setattr(pmp, "HTTP_RETRY_BACKOFF_SECS", 0.0)
        payload = {"series": [{"ticker": "KXA", "title": "Alpha", "tags": ["a"]}]}
        session = FakeSession(
            {_KALSHI_SERIES_URL: [FakeResponse(503, text="upstream boom"), FakeResponse(200, payload)]}
        )

        series = await pmp._kalshi_prefetch_series(session)

        assert [s["ticker"] for s in series] == ["KXA"]
        assert session._call_counts[_KALSHI_SERIES_URL] == 2
        assert pmp.kalshi_series_fetch_failures() == 0

    @pytest.mark.asyncio
    async def test_retry_restarts_the_parse_instead_of_resuming_it(self, monkeypatch):
        """The retry must re-issue the request with a FRESH parser and accumulator:
        a stream that died mid-array must not leave its partial series in the result
        (nor duplicate the ones the second attempt re-sends)."""
        monkeypatch.setattr(pmp, "HTTP_RETRY_BACKOFF_SECS", 0.0)
        payload = {"series": [{"ticker": f"KX{i}", "title": f"Title {i}"} for i in range(3)]}
        body = json.dumps(payload).encode()

        class _DyingContent(_FakeStreamContent):
            """A stream that dies half way through the series array."""

            async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:  # noqa: ASYNC900
                del n  # chunk size is irrelevant; this stream dies mid-array by design
                yield body[: len(body) // 2]
                raise aiohttp.ClientPayloadError("response payload truncated")

        dying = FakeResponse(200, payload)
        dying.content = _DyingContent(payload)
        session = FakeSession({_KALSHI_SERIES_URL: [dying, FakeResponse(200, payload)]})

        series = await pmp._kalshi_prefetch_series(session)

        assert [s["ticker"] for s in series] == ["KX0", "KX1", "KX2"]
        assert session._call_counts[_KALSHI_SERIES_URL] == 2

    @pytest.mark.asyncio
    async def test_non_retryable_status_is_not_retried(self, monkeypatch):
        """A 404 is a deterministic answer; retrying it just burns the budget."""
        monkeypatch.setattr(pmp, "HTTP_RETRY_BACKOFF_SECS", 0.0)
        session = FakeSession({_KALSHI_SERIES_URL: FakeResponse(404, text="no such endpoint")})

        series = await pmp._kalshi_prefetch_series(session)

        assert series == []
        assert session._call_counts[_KALSHI_SERIES_URL] == 1
        assert pmp.kalshi_series_fetch_failures() == 1

    @pytest.mark.asyncio
    async def test_retry_stays_inside_the_wall_clock_budget(self, monkeypatch, caplog):
        """KALSHI_SERIES_HTTP_TIMEOUT is the budget for the WHOLE fetch, retry
        included, so adding the retry can't push this fetch past what the
        surrounding PREDICTION_MARKET_TIMEOUT allows the entire snapshot. With the
        budget spent, neither the backoff nor the second attempt may be taken —
        asserted on elapsed WALL CLOCK, since a skipped attempt after a full-length
        sleep would still have blown the budget."""
        monkeypatch.setattr(pmp, "KALSHI_SERIES_HTTP_TIMEOUT", 0.01)
        monkeypatch.setattr(pmp, "HTTP_RETRY_BACKOFF_SECS", 5.0)
        payload = {"series": [{"ticker": "KXA", "title": "Alpha"}]}
        session = FakeSession({_KALSHI_SERIES_URL: [FakeResponse(503, text="boom"), FakeResponse(200, payload)]})

        started = time.monotonic()
        with caplog.at_level(logging.WARNING):
            series = await pmp._kalshi_prefetch_series(session)
        elapsed = time.monotonic() - started

        assert series == []
        assert elapsed < 1.0, f"fetch took {elapsed:.2f}s on a 0.01s budget"
        assert session._call_counts[_KALSHI_SERIES_URL] == 1  # retry skipped: no budget left
        assert any("budget exhausted" in r.getMessage().lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_non_200_body_read_is_bounded(self, caplog):
        """A CDN 429/502 can answer with a huge HTML error page. `resp.text()`
        buffers and decompresses ALL of it before the caller slices 200 chars,
        blowing the very ceiling the streamed read exists to enforce — so the
        error snippet must come off a bounded read."""
        served = 0

        class _HugeErrorPage(_FakeStreamContent):
            """An error page far bigger than the snippet, counting what gets served."""

            async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:  # noqa: ASYNC900
                nonlocal served
                for _ in range(200):  # 200 * n bytes of error page available
                    served += n
                    yield b"E" * n

        resp = FakeResponse(502)
        resp.content = _HugeErrorPage(None)
        session = FakeSession({_KALSHI_SERIES_URL: resp})

        with caplog.at_level(logging.WARNING):
            series = await pmp._kalshi_prefetch_series(session)

        assert series == []
        assert served <= 2 * ERROR_SNIPPET_BYTES, f"read {served} bytes of the error page; expected a bounded read"
        logged = next(r.getMessage() for r in caplog.records if "HTTP 502" in r.getMessage())
        assert len(logged) < 4096

    @pytest.mark.asyncio
    async def test_empty_but_valid_catalogue_is_not_a_degradation(self):
        """An empty (but well-formed) catalogue is a valid response, not a
        failure: return [] WITHOUT bumping the counter, and cache it."""
        session = FakeSession({_KALSHI_SERIES_URL: FakeResponse(200, {"series": []})})
        series = await pmp._kalshi_prefetch_series(session)

        assert series == []
        assert pmp.kalshi_series_fetch_failures() == 0
        assert "series" in pmp._KALSHI_CACHE  # valid empty result is cached

    @pytest.mark.asyncio
    async def test_degradation_counter_folds_into_orchestrator(self):
        """Wiring: the module counter surfaces via ResearchOrchestrator, which the
        forecaster sums into alertable_count (so a dead series path reddens CI)."""
        from metaculus_bot.research.orchestrator import (
            ResearchOrchestrator,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
        )

        orch = ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())
        assert orch.prediction_market_degraded_count == 0
        pmp._bump_kalshi_series_failure()
        pmp._bump_kalshi_series_failure()
        assert orch.prediction_market_degraded_count == 2

    @pytest.mark.asyncio
    async def test_source_loss_counter_folds_into_orchestrator(self):
        """Wiring for the source-loss counter: same path as the series counter —
        module global -> ResearchOrchestrator -> the forecaster's alertable_count. The
        two counters stay SEPARATE so a red run says which degradation fired."""
        from metaculus_bot.research.orchestrator import (
            ResearchOrchestrator,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
        )

        orch = ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())
        assert orch.prediction_market_source_loss_count == 0
        pmp._bump_source_loss()
        assert orch.prediction_market_source_loss_count == 1
        assert orch.prediction_market_degraded_count == 0  # no cross-talk

        orch.reset_run_degradation_counters()
        assert orch.prediction_market_source_loss_count == 0


class TestSnapshotSourceDiagnostics:
    """Per-source outcome tokens on MarketSnapshot.sources, drained by the
    orchestrator into the provider-diagnostics line so PARTIAL degradation (a live
    platform while the Kalshi series index silently died) is visible per-question."""

    _POLY_URL = "https://gamma-api.polymarket.com/public-search"
    _MANIFOLD_URL = "https://api.manifold.markets/v0/search-markets"
    _KALSHI_EVENTS_URL = "https://api.elections.kalshi.com/trade-api/v2/events"
    _PREDICTIT_URL = "https://www.predictit.org/api/marketdata/all/"

    # Every venue live, every venue empty — the benign baseline the loss tests deviate from.
    _HANDLERS_BASE = {
        _POLY_URL: {"events": [], "markets": []},
        _MANIFOLD_URL: [],
        _KALSHI_EVENTS_URL: {"events": [], "cursor": ""},
        _PREDICTIT_URL: {"markets": []},
    }

    @pytest.fixture(autouse=True)
    def _no_retry_backoff(self, monkeypatch):
        """These tests drive 503 retry-exhaustion paths; the real 0.5s backoff between
        attempts would dominate their runtime."""
        monkeypatch.setattr(pmp, "HTTP_RETRY_BACKOFF_SECS", 0.0)

    def _handlers(self, series_payload: dict) -> dict:
        h = {url: FakeResponse(200, payload) for url, payload in self._HANDLERS_BASE.items()}
        h["https://api.elections.kalshi.com/trade-api/v2/series"] = FakeResponse(200, series_payload)
        return h

    @staticmethod
    def _fake_llm():
        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                return "SpaceX Starship orbit"  # noqa: ASYNC910

        return FakeLlm

    @pytest.mark.asyncio
    async def test_healthy_run_records_sources_with_no_loss(self, mock_question):
        handlers = self._handlers({"series": [{"ticker": "KXSPACEX", "title": "SpaceX launches", "tags": ["spacex"]}]})
        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", self._fake_llm()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        # All 4 platforms plus the series index are recorded; the series fetch succeeded.
        assert set(snapshot.sources) == {"polymarket", "manifold", "kalshi", "predictit", "kalshi_series"}
        assert snapshot.sources["kalshi_series"] == "ok(1)"
        # No lost source -> the diagnostics suffix stays empty (byte-identical to a healthy line).

        assert _partial_loss_suffix({"sources": snapshot.sources}) == ""

    @pytest.mark.asyncio
    async def test_series_drop_surfaces_as_lost_source(self, monkeypatch, mock_question):
        # Series body over the (shrunk) ceiling: the platform stays live (events path),
        # but the entity-index source is lost -> a visible, classified loss.
        monkeypatch.setattr(pmp, "KALSHI_SERIES_MAX_BYTES", 500)
        big_series = {"series": [{"ticker": f"KX{i}", "title": f"series {i}"} for i in range(200)]}
        handlers = self._handlers(big_series)
        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", self._fake_llm()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        assert snapshot.sources["kalshi_series"] == "dropped(size_cap)"

        suffix = _partial_loss_suffix({"sources": snapshot.sources})
        assert "lost=kalshi_series:dropped(size_cap)" in suffix

    @pytest.mark.asyncio
    async def test_provider_records_sources_into_registry(self, monkeypatch, mock_question):
        """The _fetch ResearchCallable records the sources map into the provider-detail
        registry keyed by (qid, 'prediction_market') for the orchestrator to drain."""
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")
        monkeypatch.setattr(pmp, "KALSHI_SERIES_MAX_BYTES", 500)
        handlers = self._handlers({"series": [{"ticker": f"KX{i}", "title": f"series {i}"} for i in range(200)]})
        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", self._fake_llm()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            provider = pmp.prediction_market_provider()
            await provider(mock_question)

        detail = pop_provider_detail(mock_question.id_of_question, "prediction_market")
        assert detail.get("sources", {}).get("kalshi_series") == "dropped(size_cap)"

    @pytest.mark.asyncio
    async def test_dead_keyword_extractor_is_a_keywords_loss_not_a_venue_loss(self, monkeypatch, mock_question):
        """A dead keyword extractor silences all four venues without any venue failing.

        2026-07-26: the extractor's LLM took a 403 and the run reported
        ``prediction_market_platform_failures=1``, which reads as "a venue went down"
        when no venue was ever queried. The scalar is now named for what it counts (a
        lost SOURCE), and the per-source token — durable in both the published comment
        and the schema-v2 archive — is what distinguishes the two causes.
        """
        handlers = self._handlers({"series": []})

        class DeadLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                return ""  # noqa: ASYNC910  # extractor soft-fails to "" on any LLM error

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", DeadLlm),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        assert snapshot.matches == []
        # The cause is durable per-source: exactly the keywords token, no venue tokens
        # (nothing was queried), so no reader can mistake this for a venue outage.
        assert snapshot.sources == {"keywords": "error(no_queries)"}
        assert "lost=keywords:error(no_queries)" in _partial_loss_suffix({"sources": snapshot.sources})
        # One lost source, counted once, in the correctly-named scalar.
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_platform_outage_is_a_loss_and_healthy_platforms_still_land(self, mock_question, manifold_payload):
        """A 503'd venue reads as a LOSS, never as `none`. The other half of the
        contract still holds: the healthy venues' matches come through unaffected."""
        handlers = self._handlers({"series": []})
        handlers[self._POLY_URL] = FakeResponse(503, text="service unavailable")
        handlers[self._MANIFOLD_URL] = FakeResponse(200, manifold_payload)

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", self._fake_llm()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        assert snapshot.sources["polymarket"] == "error(all_queries_failed)"
        assert "lost=polymarket:" in _partial_loss_suffix({"sources": snapshot.sources})
        # Soft-fail preserved: Manifold's match is still in the snapshot.
        assert "manifold" in {m.platform for m in snapshot.matches}
        assert snapshot.sources["manifold"].startswith("ok(")
        # Operator decision 2026-07-25: ANY platform failure reddens CI, so one dead
        # venue among three healthy ones is already alertable.
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_total_blackout_reads_as_lost_even_with_a_warm_series_cache(self, mock_question):
        """The worst case that motivated the fix: every venue is down, but Kalshi's 6h
        series cache is warm from an earlier question, so the series index reports a
        healthy `ok(1)`. Every platform token still has to be a loss — otherwise the
        published diagnostics line reads actively healthy through a total blackout."""
        pmp._KALSHI_CACHE["series"] = (
            time.monotonic(),
            [{"ticker": "KXSPACEX", "title": "SpaceX launches", "tags": ["spacex"]}],
        )
        handlers = {url: FakeResponse(503, text="service unavailable") for url in self._HANDLERS_BASE}

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", self._fake_llm()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        assert snapshot.sources["kalshi_series"] == "ok(1)"  # the stale-cache trap
        suffix = _partial_loss_suffix({"sources": snapshot.sources})
        for platform in ("polymarket", "manifold", "kalshi", "predictit"):
            assert snapshot.sources[platform] == "error(all_queries_failed)", platform
            assert f"{platform}:error(all_queries_failed)" in suffix, platform
        # One alertable bump per lost venue — four, not one.
        assert pmp.prediction_market_source_losses() == 4

    @pytest.mark.asyncio
    async def test_all_live_but_empty_stays_benign_none(self, mock_question):
        """Regression guard for the other direction: four live venues that simply have
        no matching market must ALL read `none` with an empty suffix, so the widened
        token vocabulary never cries outage on a healthy no-match question."""
        handlers = self._handlers({"series": []})

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", self._fake_llm()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        assert snapshot.sources == {
            "polymarket": "none",
            "manifold": "none",
            "kalshi": "none",
            "predictit": "none",
            "kalshi_series": "none",
        }
        assert _partial_loss_suffix({"sources": snapshot.sources}) == ""
        # The load-bearing half of the alerting guard: a genuine no-match must never
        # redden CI, however sensitive the outage alert is.
        assert pmp.prediction_market_source_losses() == 0

    # The D2 tests below need PRODUCTION-SHAPED keyword output. `_fake_llm` returns a
    # 3-token query, which sits below Manifold's ~4-content-token satisfiability cliff and
    # would therefore match even on the broken code — a test that passes pre-fix. Real S4
    # and S5 prompts ask for "3-5 noun phrases" / "under 12 words" and measure at 5-8
    # content tokens, so a faithful stub has to be long.
    @staticmethod
    def _fake_llm_long_query():
        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                return "SpaceX Starship orbital velocity reach orbit July 2026 threshold"  # noqa: ASYNC910

        return FakeLlm

    @staticmethod
    def _manifold_conjunction_stub(
        seen_terms: list[str], payload: Any, *, empty_response: Any = None
    ) -> Callable[[dict[str, Any]], FakeResponse]:
        """Stub Manifold's measured STRICT-CONJUNCTION behaviour: a term over ~3 content
        tokens is unsatisfiable and returns `[]`; a short one matches. `empty_response`
        overrides what a satisfiable-but-broken rung returns (used to test failure paths)."""

        def _handler(params: dict[str, Any]) -> FakeResponse:
            term = params.get("term") or ""
            seen_terms.append(term)
            content_tokens = [t for t in term.split() if t.lower().strip(".,'") not in pmp._RELEVANCE_STOPWORDS]
            if len(content_tokens) > 3:
                return FakeResponse(200, [])
            return empty_response if empty_response is not None else FakeResponse(200, payload)

        return _handler

    @pytest.mark.asyncio
    async def test_manifold_relaxes_the_query_when_full_length_matches_nothing(self, mock_question, manifold_payload):
        """D2: Manifold's `term` is a strict conjunction, so a full-length question
        sentence is usually unsatisfiable and returns `[]`. This venue contributed zero
        rows to any bundle for 17+ days while every observability channel read healthy —
        `[]` classifies as the benign `none` token, so nothing warned.

        Pre-fix the provider only ever issued long terms, so the snapshot came back empty.
        """
        handlers = self._handlers({"series": []})
        seen_terms: list[str] = []
        handlers[self._MANIFOLD_URL] = self._manifold_conjunction_stub(seen_terms, manifold_payload)

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", self._fake_llm_long_query()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        assert "manifold" in {m.platform for m in snapshot.matches}
        assert snapshot.sources["manifold"].startswith("ok(")
        # A rescued query is NOT a degradation: the plumbing worked, so nothing alerts.
        assert pmp.prediction_market_source_losses() == 0
        # The precise queries were tried FIRST — relaxation is a fallback, not a
        # replacement, so questions that already match at full length keep their precision.
        assert any(len(t.split()) > 4 for t in seen_terms), seen_terms

    @pytest.mark.asyncio
    async def test_manifold_does_not_relax_when_a_full_length_query_already_matched(
        self, mock_question, manifold_payload
    ):
        """The fallback must cost nothing on the healthy path. A question whose full-length
        query matches must issue no relaxation calls at all — otherwise every question pays
        extra HTTP for a rescue it doesn't need, and the high-precision long-query result
        could be displaced by a looser one."""
        handlers = self._handlers({"series": []})
        seen_terms: list[str] = []

        def _manifold_handler(params: dict[str, Any]) -> FakeResponse:
            seen_terms.append(params.get("term") or "")
            return FakeResponse(200, manifold_payload)

        handlers[self._MANIFOLD_URL] = _manifold_handler

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", self._fake_llm_long_query()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        assert "manifold" in {m.platform for m in snapshot.matches}
        # Exactly the base query set (the LLM query plus S2) — no relaxation rungs.
        assert len(seen_terms) == 2, seen_terms
        assert all(len(t.split()) > 3 for t in seen_terms), seen_terms

    @pytest.mark.asyncio
    async def test_manifold_relaxation_failure_is_still_classified_as_a_loss(self, mock_question):
        """A relaxation rung that 503s must not be laundered into a benign `none`: the
        whole point of the None-vs-`[]` split is that a fetch failure stays visible."""
        handlers = self._handlers({"series": []})
        seen_terms: list[str] = []
        handlers[self._MANIFOLD_URL] = self._manifold_conjunction_stub(
            seen_terms, None, empty_response=FakeResponse(503, text="down")
        )

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", self._fake_llm_long_query()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        assert snapshot.sources["manifold"].startswith("partial("), snapshot.sources["manifold"]
        assert "lost=manifold:" in _partial_loss_suffix({"sources": snapshot.sources})
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_one_lost_query_of_two_reads_as_partial(self, mock_question, manifold_payload):
        """Manifold runs two queries (the LLM query plus the extra S2 natural-language
        one). Losing one of them used to publish a clean `ok(N)` off the survivor."""
        handlers = self._handlers({"series": []})

        def _manifold_handler(params: dict[str, Any]) -> FakeResponse:
            # The S2 query is the raw question text; the LLM query is the short one.
            if "reach orbit" in (params.get("term") or ""):
                return FakeResponse(503, text="service unavailable")
            return FakeResponse(200, manifold_payload)

        handlers[self._MANIFOLD_URL] = _manifold_handler

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", self._fake_llm()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        assert snapshot.sources["manifold"] == "partial(1/2)"
        assert "lost=manifold:partial(1/2)" in _partial_loss_suffix({"sources": snapshot.sources})
        # The surviving query's match still lands.
        assert "manifold" in {m.platform for m in snapshot.matches}
        # A partial loss is a loss: it alerts even though matches came back.
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_failed_kalshi_prefetch_does_not_poison_the_next_question(self, mock_question, kalshi_events_payload):
        """Question 1's Kalshi prefetch 503s, question 2's succeeds. The failed prefetch
        must not have cached its empty list for the full 6h TTL, or one transient blip
        starves every later question in the run behind a stale `none`."""
        calls = {"n": 0}

        def _events_handler(_params: dict[str, Any]) -> FakeResponse:
            calls["n"] += 1
            if calls["n"] == 1:
                return FakeResponse(503, text="service unavailable")
            return FakeResponse(200, kalshi_events_payload)

        handlers = self._handlers({"series": []})
        handlers[self._KALSHI_EVENTS_URL] = _events_handler

        second_question = MagicMock()
        second_question.id_of_question = 54321
        second_question.question_text = mock_question.question_text
        second_question.title = mock_question.title
        second_question.resolution_criteria = mock_question.resolution_criteria
        second_question.scheduled_resolution_time = None

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", self._fake_llm()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            first = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)
            second = await pmp.fetch_market_snapshot(second_question, timeout=5.0)

        assert first.sources["kalshi"] == "error(all_queries_failed)"
        assert "kalshi" not in {m.platform for m in first.matches}
        assert second.sources["kalshi"].startswith("ok(")
        assert "kalshi" in {m.platform for m in second.matches}


class TestProviderHealthRecording:
    """The recording seam: what a real snapshot fetch hands the degradation rules.

    The rules themselves are tested in ``tests/test_provider_health.py``. These tests
    pin the numbers this module feeds them, which is where the risk actually lives —
    both counts (pre-filter candidates and post-filter rows) are only simultaneously
    in scope at the tail of ``_fetch_market_snapshot_impl``, and re-deriving either
    one somewhere else is the kind of second derivation that let the Kalshi
    field-name guess drift from reality in the first place.
    """

    _POLY_URL = "https://gamma-api.polymarket.com/public-search"
    _MANIFOLD_URL = "https://api.manifold.markets/v0/search-markets"
    _KALSHI_EVENTS_URL = "https://api.elections.kalshi.com/trade-api/v2/events"
    _KALSHI_SERIES_URL = "https://api.elections.kalshi.com/trade-api/v2/series"
    _PREDICTIT_URL = "https://www.predictit.org/api/marketdata/all/"

    @pytest.fixture(autouse=True)
    def _reset_health(self):
        reset_provider_health()
        yield
        reset_provider_health()

    @staticmethod
    def _fake_llm():
        class FakeLlm:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

            async def invoke(self, prompt: str) -> str:
                return "SpaceX Starship orbit"  # noqa: ASYNC910

        return FakeLlm

    # A market-less question is one where every venue's catalogue POPULATED and simply
    # held nothing relevant. So the baseline stubs return non-matching entries rather
    # than empty catalogues — an empty catalogue is a different degradation (Signal C),
    # and stubbing one here would make every test in this class fire it.
    _OFF_TOPIC_KALSHI_EVENT = {
        "event_ticker": "KXWORLDCUP-26",
        "title": "Who wins the 2026 World Cup?",
        "markets": [{"ticker": "KXWORLDCUP-26-BRA", "title": "Brazil", "status": "active"}],
    }
    _OFF_TOPIC_PREDICTIT_MARKET = {
        "id": 9001,
        "name": "Who wins the 2026 World Cup?",
        "url": "https://www.predictit.org/markets/detail/9001",
        "status": "Open",
        "contracts": [{"id": 1, "name": "Brazil", "status": "Open", "lastTradePrice": 0.2}],
    }

    def _handlers(self, **overrides: Any) -> dict:
        handlers: dict[str, Any] = {
            self._POLY_URL: FakeResponse(200, {"events": [], "markets": []}),
            self._MANIFOLD_URL: FakeResponse(200, []),
            self._KALSHI_EVENTS_URL: FakeResponse(200, {"events": [self._OFF_TOPIC_KALSHI_EVENT], "cursor": ""}),
            self._KALSHI_SERIES_URL: FakeResponse(200, {"series": []}),
            self._PREDICTIT_URL: FakeResponse(200, {"markets": [self._OFF_TOPIC_PREDICTIT_MARKET]}),
        }
        handlers.update(overrides)
        return handlers

    async def _fetch(self, question: Any, handlers: dict, *, as_of: datetime | None = None) -> Any:
        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", self._fake_llm()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            return await pmp.fetch_market_snapshot(question, timeout=5.0, as_of=as_of)

    @pytest.mark.asyncio
    async def test_an_all_empty_run_records_zeros_and_finds_nothing(self, mock_question):
        """The market-less question, end to end from the HTTP seam. Every venue's
        catalogue populated and matched nothing, so all four are observed at zero rows
        and zero candidates, and NOTHING is alertable.

        The false-positive case that matters most, verified through the real fetch path
        rather than hand-recorded observations — this is the tournament day the naive
        per-run-rate design would have reddened.
        """
        await self._fetch(mock_question, self._handlers())

        observed = {obs.venue: obs for obs in recorded_observations()[0]}
        assert set(observed) == {"polymarket", "manifold", "kalshi", "predictit"}
        assert all(obs.rows_post_filter == 0 and obs.candidates_pre_filter == 0 for obs in observed.values())
        assert provider_degradation_findings() == []

    @pytest.mark.asyncio
    async def test_rows_and_liquidity_presence_come_from_the_rendered_matches(
        self, mock_question, kalshi_events_payload
    ):
        """Presence is read off the parsed ``MarketMatch`` objects that actually render,
        so the signal can never disagree with the ``signal`` column the forecaster is
        told to weight by."""
        handlers = self._handlers(**{self._KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})
        snapshot = await self._fetch(mock_question, handlers)

        rendered = [m for m in snapshot.matches if m.platform == "kalshi"]
        observed = next(obs for obs in recorded_observations()[0] if obs.venue == "kalshi")
        assert observed.rows_post_filter == len(rendered)
        for name in VENUE_EXPECTED_LIQUIDITY_FIELDS["kalshi"]:
            assert (name in observed.liquidity_fields_present) == any(
                getattr(row, name) is not None for row in rendered
            )

    @pytest.mark.asyncio
    async def test_a_dropped_candidate_carrying_the_field_does_not_mask_a_blank_rendered_row(self, mock_question):
        """The case that makes the rendered-vs-pre-filter distinction OBSERVABLE.

        Two Polymarket events: one carries ``openInterest`` but closes before ``as_of``
        (so the filter drops it), the other renders and carries no OI. Presence read
        off the PRE-filter matches would report ``open_interest`` alive and stay silent,
        while the table a forecaster reads shows ``-`` in the OI column on every
        rendered row. Reading the rendered rows is what keeps the signal and the
        rendered output from disagreeing — the whole reason presence is computed off
        the parsed ``MarketMatch`` objects rather than the upstream payload.
        """
        payload = {
            "events": [
                {
                    "title": "Will SpaceX Starship reach orbit in 2026?",
                    "slug": "starship-orbit-2026-early",
                    "description": "Closes early; dropped by the as_of filter.",
                    "endDate": "2026-01-31T23:59:59Z",
                    "volume": 50000.0,
                    "markets": [
                        {
                            "question": "Will SpaceX Starship reach orbit in 2026?",
                            "outcomePrices": '["0.60", "0.40"]',
                            "volumeNum": 50000.0,
                            "openInterest": 12345.0,
                        }
                    ],
                },
                {
                    "title": "Will SpaceX Starship reach orbit before 2028?",
                    "slug": "starship-orbit-2028",
                    "description": "Renders; carries no openInterest.",
                    "endDate": "2027-12-31T23:59:59Z",
                    "volume": 70000.0,
                    "markets": [
                        {
                            "question": "Will SpaceX Starship reach orbit before 2028?",
                            "outcomePrices": '["0.80", "0.20"]',
                            "volumeNum": 70000.0,
                        }
                    ],
                },
            ],
            "markets": [],
        }
        handlers = self._handlers(**{self._POLY_URL: FakeResponse(200, payload)})
        snapshot = await self._fetch(mock_question, handlers, as_of=datetime(2026, 6, 1, tzinfo=timezone.utc))

        rendered = [m for m in snapshot.matches if m.platform == "polymarket"]
        assert rendered, "the later-closing event must survive the as_of filter"
        assert all(row.open_interest is None for row in rendered)

        observed = next(obs for obs in recorded_observations()[0] if obs.venue == "polymarket")
        assert observed.candidates_pre_filter > observed.rows_post_filter
        assert "open_interest" not in observed.liquidity_fields_present
        assert "total_volume" in observed.liquidity_fields_present

        findings = provider_degradation_findings()
        assert [f.detail["fields"] for f in findings] == ["open_interest"]

    @pytest.mark.asyncio
    async def test_as_of_dropped_candidates_are_recorded_pre_filter(self, mock_question, polymarket_payload):
        """The leg that keeps correct as_of drops from alerting. Polymarket fetched
        markets that all close before ``as_of``, so zero rows RENDER while the
        candidate count stays positive. Recording only the post-filter number would
        make this indistinguishable from a dead venue — which is precisely the shape
        that would have fired on 20 of 47 archived runs.
        """
        handlers = self._handlers(**{self._POLY_URL: FakeResponse(200, polymarket_payload)})
        # The fixture's Polymarket events all close 2026-12-31, so an as_of past that
        # date drops every one of them at the filter — the same mechanism that dropped
        # 20 real candidates on a question resolving after the markets closed.
        snapshot = await self._fetch(mock_question, handlers, as_of=datetime(2027, 6, 1, tzinfo=timezone.utc))

        assert [m for m in snapshot.matches if m.platform == "polymarket"] == []
        observed = next(obs for obs in recorded_observations()[0] if obs.venue == "polymarket")
        assert observed.rows_post_filter == 0
        assert observed.candidates_pre_filter > 0
        assert provider_degradation_findings() == []

    @pytest.mark.asyncio
    async def test_a_lost_venue_records_no_observation(self, mock_question):
        """A venue whose fan-out lost a sub-fetch is already alertable via
        ``prediction_market_source_losses``, so it is not observed here — counting the
        same outage twice would report one failure as two, and "check the query
        construction" is the wrong remedy for a 503."""
        handlers = self._handlers(**{self._MANIFOLD_URL: FakeResponse(503, text="service unavailable")})
        with patch.object(pmp, "HTTP_RETRY_BACKOFF_SECS", 0.0):
            await self._fetch(mock_question, handlers)

        venues = {obs.venue for obs in recorded_observations()[0]}
        assert "manifold" not in venues
        assert pmp.prediction_market_source_losses() >= 1
        assert provider_degradation_findings() == []

    @pytest.mark.asyncio
    async def test_prefetch_catalogue_sizes_are_recorded_with_their_fetch_outcome(
        self, mock_question, kalshi_events_payload
    ):
        """Signal C's input. Both prefetch venues report their catalogue size AND
        whether the fetch succeeded, which is the only way an empty-but-successful
        catalogue is distinguishable from an outage."""
        predictit_payload = {"markets": [{"id": 1, "name": "Some market", "contracts": [], "status": "Open"}]}
        handlers = self._handlers(
            **{
                self._KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
                self._PREDICTIT_URL: FakeResponse(200, predictit_payload),
            }
        )
        await self._fetch(mock_question, handlers)

        catalogues = {obs.source: obs for obs in recorded_observations()[1]}
        assert catalogues["kalshi_events"].entries == len(kalshi_events_payload["events"])
        assert catalogues["kalshi_events"].fetch_ok is True
        assert catalogues["predictit_markets"].entries == 1
        assert catalogues["predictit_markets"].fetch_ok is True

    @pytest.mark.asyncio
    async def test_a_failed_prefetch_records_the_failure_not_a_phantom_empty_catalogue(self, mock_question):
        """A 503'd Kalshi prefetch must record ``fetch_ok=False``, so Signal C stays
        silent and the source-loss counter remains the sole reporter."""
        handlers = self._handlers(**{self._KALSHI_EVENTS_URL: FakeResponse(503, text="service unavailable")})
        with patch.object(pmp, "HTTP_RETRY_BACKOFF_SECS", 0.0):
            await self._fetch(mock_question, handlers)

        catalogues = {obs.source: obs for obs in recorded_observations()[1]}
        assert catalogues["kalshi_events"].fetch_ok is False
        assert provider_degradation_findings() == []

    @pytest.mark.asyncio
    async def test_a_question_with_no_id_records_nothing(self, mock_question):
        """Every observation is keyed on the question, so an id-less question records
        nothing — matching ``record_provider_detail`` / ``record_raw_research``."""
        mock_question.id_of_question = None
        await self._fetch(mock_question, self._handlers())

        assert recorded_observations()[0] == ()
        assert recorded_observations()[1] == ()

    @pytest.mark.asyncio
    async def test_recording_does_not_alter_the_snapshot(self, mock_question, kalshi_events_payload):
        """Recording is a pure module-state write on the research path: same matches,
        same source tokens, with or without observations already present."""
        handlers = self._handlers(**{self._KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})
        first = await self._fetch(mock_question, handlers)

        _reset_session_caches()
        reset_provider_health()
        handlers = self._handlers(**{self._KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})
        second = await self._fetch(mock_question, handlers)

        assert [(m.platform, m.market_title) for m in first.matches] == [
            (m.platform, m.market_title) for m in second.matches
        ]
        assert first.sources == second.sources
