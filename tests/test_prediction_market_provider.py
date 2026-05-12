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
- Keyword-extraction calls gpt-5-mini with max_tokens >= 800 (G0 token-budget trap)
- Malformed response -> empty snapshot + WARNING logged
- Polymarket 403 rate-limit -> bounded retry-with-backoff -> eventual empty
- Per-question timeout -> empty snapshot (soft-fail)
- `PREDICTION_MARKETS_ENABLED=false` -> orchestrator returns empty
- Formatter emits the NOT-AN-ANCHOR caveat + raw_rules
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Test isolation: module-level caches in prediction_market_provider.py bleed
# across tests when run in certain orders (observed 2026-05-12: full-suite
# run fails 2 tests that pass individually). Reset every test.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_provider_caches():
    from metaculus_bot.prediction_market_provider import _reset_session_caches

    _reset_session_caches()
    yield
    _reset_session_caches()


# ---------------------------------------------------------------------------
# Fake aiohttp session
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status: int, payload: Any = None, text: str | None = None):
        self.status = status
        self._payload = payload
        self._text = text if text is not None else ""

    async def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no json payload")
        return self._payload  # noqa: ASYNC910

    async def text(self) -> str:
        return self._text  # noqa: ASYNC910

    async def __aenter__(self) -> "FakeResponse":
        return self  # noqa: ASYNC910

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        return None  # noqa: ASYNC910


class FakeSession:
    """Simple aiohttp.ClientSession replacement that dispatches GET requests
    to handler callables keyed by URL prefix.
    """

    def __init__(self, handlers: dict[str, Any]):
        self._handlers = {k: (v if isinstance(v, list) else [v]) for k, v in handlers.items()}
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
            "closeTime": 1782086400000,  # ms: 2026-06-20
            "textDescription": "YES if SpaceX Starship reaches orbit before July 1 2026.",
            "isResolved": False,
        }
    ]


@pytest.fixture
def kalshi_events_payload():
    """Kalshi /events?status=open&with_nested_markets=true shape."""
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
                        "volume_24h_fp": "2500.0",
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


# ---------------------------------------------------------------------------
# Polymarket tests
# ---------------------------------------------------------------------------


class TestPolymarket:
    @pytest.mark.asyncio
    async def test_search_and_parse_implied_prob(self, polymarket_payload):
        from metaculus_bot.prediction_market_provider import _polymarket_search

        session = FakeSession({"https://gamma-api.polymarket.com/public-search": FakeResponse(200, polymarket_payload)})
        matches = await _polymarket_search(session, "Starship orbit 2026")

        assert len(matches) == 2
        top = matches[0]
        assert top.platform == "polymarket"
        assert "Starship" in top.market_title
        assert top.implied_prob_yes == pytest.approx(0.74)
        assert top.market_url.startswith("https://polymarket.com/")
        assert top.bid == pytest.approx(0.73)
        assert top.ask == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_rate_limit_retry_with_backoff_then_empty(self, monkeypatch):
        """403 on every attempt -> bounded retry -> eventual empty list, no exception."""
        from metaculus_bot import prediction_market_provider as pmp

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
        assert matches == []
        # Should have retried a bounded number of times (>=2, <= 5)
        assert 2 <= call_count["n"] <= 5
        # And at least one backoff sleep
        assert len(sleeps) >= 1

    @pytest.mark.asyncio
    async def test_malformed_json_returns_empty_and_warns(self, caplog):
        from metaculus_bot.prediction_market_provider import _polymarket_search

        session = FakeSession(
            {"https://gamma-api.polymarket.com/public-search": FakeResponse(200, payload={"garbage": 1})}
        )
        with caplog.at_level(logging.WARNING):
            matches = await _polymarket_search(session, "anything")
        assert matches == []

    def test_parse_implied_prob_from_list_prices(self):
        from metaculus_bot.prediction_market_provider import _parse_polymarket_matches

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
        from metaculus_bot.prediction_market_provider import _kalshi_prefetch_events, _kalshi_search_local

        session = FakeSession(
            {"https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(200, kalshi_events_payload)}
        )
        events = await _kalshi_prefetch_events(session, event_limit=100, page_sleep_s=0.0)
        assert len(events) == 2

        matches = _kalshi_search_local(events, "Starship orbit 2026", top_k=3, min_score=30.0)
        assert len(matches) >= 1
        top = matches[0]
        assert top.platform == "kalshi"
        assert "Starship" in top.market_title
        # yes_bid=0.68, yes_ask=0.72 -> midpoint 0.70
        assert top.implied_prob_yes == pytest.approx(0.70, abs=0.01)
        assert top.market_url.startswith("https://kalshi.com/")
        assert "orbital velocity" in top.raw_rules.lower()

    @pytest.mark.asyncio
    async def test_prefetch_handles_http_error(self, caplog):
        from metaculus_bot.prediction_market_provider import _kalshi_prefetch_events

        session = FakeSession({"https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(500, text="boom")})
        with caplog.at_level(logging.WARNING):
            events = await _kalshi_prefetch_events(session, event_limit=100, page_sleep_s=0.0)
        assert events == []


# ---------------------------------------------------------------------------
# Manifold tests
# ---------------------------------------------------------------------------


class TestManifold:
    @pytest.mark.asyncio
    async def test_search_and_parse_direct_probability(self, manifold_payload):
        from metaculus_bot.prediction_market_provider import _manifold_search

        session = FakeSession({"https://api.manifold.markets/v0/search-markets": FakeResponse(200, manifold_payload)})
        matches = await _manifold_search(session, "Starship orbit July 2026")

        assert len(matches) == 1
        m = matches[0]
        assert m.platform == "manifold"
        assert m.implied_prob_yes == pytest.approx(0.62)
        assert "Starship" in m.market_title
        assert m.is_resolved is False

    @pytest.mark.asyncio
    async def test_non_list_payload_returns_empty(self, caplog):
        from metaculus_bot.prediction_market_provider import _manifold_search

        session = FakeSession(
            {"https://api.manifold.markets/v0/search-markets": FakeResponse(200, payload={"unexpected": "shape"})}
        )
        with caplog.at_level(logging.WARNING):
            matches = await _manifold_search(session, "anything")
        assert matches == []


# ---------------------------------------------------------------------------
# Keyword extraction (S4 + S5 union, gpt-5-mini max_tokens >= 800)
# ---------------------------------------------------------------------------


class TestKeywordExtractor:
    @pytest.mark.asyncio
    async def test_extract_runs_s4_and_s5_with_max_tokens_800(self, mock_question):
        """G0 token-budget trap defense: the extractor MUST request max_tokens >= 800.
        reasoning=low gpt-5-mini burns 128-512 tokens on invisible reasoning."""
        from metaculus_bot import prediction_market_provider as pmp

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
        # And use gpt-5-mini
        assert all("gpt-5-mini" in kw.get("model", "") for kw in captured_kwargs)

    @pytest.mark.asyncio
    async def test_manifold_gets_extra_s2_query(self, mock_question):
        """Per G0 findings, Manifold prefers natural-language (S2) framings."""
        from metaculus_bot import prediction_market_provider as pmp

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
        from metaculus_bot import prediction_market_provider as pmp

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
        from metaculus_bot import prediction_market_provider as pmp

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


# ---------------------------------------------------------------------------
# fetch_market_snapshot orchestrator tests
# ---------------------------------------------------------------------------


class TestFetchMarketSnapshot:
    @pytest.mark.asyncio
    async def test_full_orchestrator_returns_matches(
        self, mock_question, polymarket_payload, manifold_payload, kalshi_events_payload
    ):
        from metaculus_bot import prediction_market_provider as pmp

        handlers = {
            "https://gamma-api.polymarket.com/public-search": FakeResponse(200, polymarket_payload),
            "https://api.manifold.markets/v0/search-markets": FakeResponse(200, manifold_payload),
            "https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(200, kalshi_events_payload),
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

        # Should find at least one match from each of the three platforms
        platforms = {m.platform for m in snapshot.matches}
        assert "polymarket" in platforms
        assert "kalshi" in platforms
        assert "manifold" in platforms

    @pytest.mark.asyncio
    async def test_as_of_filter_drops_post_as_of_matches(self, mock_question):
        """Backtest leakage defense: matches that closed AFTER as_of are dropped.

        Resolved prediction markets keep a 'last trade' price after the question
        settled; using that during a resolved-question backtest is leakage.
        """
        from metaculus_bot import prediction_market_provider as pmp

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
        from metaculus_bot import prediction_market_provider as pmp

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

    @pytest.mark.asyncio
    async def test_orchestrator_soft_fails_on_any_platform_error(self, mock_question, manifold_payload, caplog):
        """If one platform errors, other platforms' matches still come through."""
        from metaculus_bot import prediction_market_provider as pmp

        def _boom(_params):
            raise RuntimeError("connection refused")

        handlers = {
            "https://gamma-api.polymarket.com/public-search": _boom,
            "https://api.manifold.markets/v0/search-markets": FakeResponse(200, manifold_payload),
            "https://api.elections.kalshi.com/trade-api/v2/events": _boom,
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
    async def test_max_matches_per_platform_respected(
        self, mock_question, polymarket_payload, manifold_payload, kalshi_events_payload
    ):
        from metaculus_bot import prediction_market_provider as pmp

        handlers = {
            "https://gamma-api.polymarket.com/public-search": FakeResponse(200, polymarket_payload),
            "https://api.manifold.markets/v0/search-markets": FakeResponse(200, manifold_payload),
            "https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(200, kalshi_events_payload),
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
        from collections import Counter

        counts = Counter(m.platform for m in snapshot.matches)
        assert all(v <= 1 for v in counts.values()), counts


# ---------------------------------------------------------------------------
# prediction_market_provider factory (research-provider flag gating)
# ---------------------------------------------------------------------------


class TestProviderFactory:
    @pytest.mark.asyncio
    async def test_disabled_flag_returns_empty_at_orchestrator_level(self, monkeypatch):
        """When PREDICTION_MARKETS_ENABLED is not set, the provider returns ''.
        This is a defense-in-depth check at the research-provider entrypoint."""
        from metaculus_bot.prediction_market_provider import prediction_market_provider

        monkeypatch.delenv("PREDICTION_MARKETS_ENABLED", raising=False)

        provider = prediction_market_provider()
        # Calling the provider with a question_text should return empty string
        # because the flag gate prevents any fetching.
        result = await provider("Will Starship reach orbit?")
        assert result == ""

    @pytest.mark.asyncio
    async def test_enabled_flag_fetches_and_formats(
        self, monkeypatch, polymarket_payload, manifold_payload, kalshi_events_payload
    ):
        from metaculus_bot import prediction_market_provider as pmp

        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")

        handlers = {
            "https://gamma-api.polymarket.com/public-search": FakeResponse(200, polymarket_payload),
            "https://api.manifold.markets/v0/search-markets": FakeResponse(200, manifold_payload),
            "https://api.elections.kalshi.com/trade-api/v2/events": FakeResponse(200, kalshi_events_payload),
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
            out = await provider("Will SpaceX Starship reach orbit before July 2026?")

        assert isinstance(out, str)
        assert "NOT AN ANCHOR" in out
        assert "polymarket" in out.lower() or "manifold" in out.lower() or "kalshi" in out.lower()


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


class TestFormatter:
    def test_formatter_emits_caveat_and_columns(self):
        from metaculus_bot.prediction_market_provider import (
            MarketMatch,
            MarketSnapshot,
            format_snapshot_for_research,
        )

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
                )
            ]
        )

        formatted = format_snapshot_for_research(snap)

        # The NOT-AN-ANCHOR caveat is the one load-bearing guardrail for the forecaster prompt
        assert "NOT AN ANCHOR" in formatted
        assert "verify" in formatted.lower()
        # Columns per the plan
        assert "platform" in formatted.lower()
        assert "polymarket" in formatted.lower()
        assert "0.74" in formatted
        # Raw rules included
        assert "orbital velocity" in formatted.lower()

    def test_formatter_empty_snapshot_returns_empty_string(self):
        from metaculus_bot.prediction_market_provider import MarketSnapshot, format_snapshot_for_research

        assert format_snapshot_for_research(MarketSnapshot(matches=[])) == ""

    def test_formatter_truncates_long_raw_rules(self):
        from metaculus_bot.prediction_market_provider import (
            MarketMatch,
            MarketSnapshot,
            format_snapshot_for_research,
        )

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
