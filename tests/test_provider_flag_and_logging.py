import asyncio
import dataclasses
import json
import logging
from datetime import UTC, datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import MetaculusQuestion

from main import TemplateForecaster
from metaculus_bot.research.orchestrator import ResearchOrchestrator

# Imported names referenced inside test bodies below; bind to module scope
# so the auto-formatter doesn't strip them as "unused".
_DT = datetime
_TZ = timezone
_ = asyncio  # used inside nested async def stubs defined in tests below


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Skip the AskNews 10s rate-limit pre-sleep in tests; SDK is mocked anyway.

    research_providers.py inserts unconditional ``await asyncio.sleep(10.1)``
    gates before each of the HOT and HISTORICAL AskNews calls. Even when the
    SDK is mocked, those sleeps still run — adding ~20s per test, ~100s for
    this whole file. Patching only the research_providers module's asyncio
    keeps the fixture scoped to this file (no global conftest pollution).
    """

    async def _instant(*args, **kwargs):
        return None

    monkeypatch.setattr("metaculus_bot.research.providers.asyncio.sleep", _instant)


@pytest.mark.asyncio
async def test_research_provider_flag_and_logging(mock_os_getenv, caplog):
    # Force AskNews via env flag and provide required creds
    mock_os_getenv.side_effect = lambda x, default=None: {
        "RESEARCH_PROVIDER": "asknews",
        "ASKNEWS_CLIENT_ID": "client",
        "ASKNEWS_SECRET": "secret",
    }.get(x, default)

    bot = TemplateForecaster(
        llms={
            "default": "mock_default_model",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
            "summarizer": "mock_summarizer",
        }
    )
    q = MetaculusQuestion(
        question_text="Test",
        page_url="http://example.com",
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
    )

    with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
        # Mock the SDK to return our test result
        mock_sdk = AsyncMock()
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        mock_sdk.news.search_news.return_value = mock_response
        mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

        # No articles means no section at all — not a header over a "no articles" sentence.
        # This test used to assert that sentence plus the header; both were the defect,
        # since an empty AskNews read then arrived as `ok` with chars>0 and the summarizer
        # was asked to brief from prose that carried no facts.
        with (
            patch.object(ResearchOrchestrator, "_summarize_asknews", new_callable=AsyncMock) as summarize,
            caplog.at_level(logging.INFO),
        ):
            res = await bot.run_research(q)
        assert "No articles were found for this query." not in res
        assert "## News Articles (AskNews)" not in res
        # The spend-saving half of the empty-read fix: nothing to brief from, so the
        # summarizer LLM call is skipped entirely rather than asked to invent one.
        summarize.assert_not_awaited()
        assert any("ASKNEWS_NO_ARTICLES: question=" in rec.message for rec in caplog.records)
        assert any("Using research providers:" in rec.message for rec in caplog.records)


@dataclasses.dataclass
class _StubArticle:
    """AskNews-Article-shaped stub: the fields _format_single_article/dedup read."""

    eng_title: str
    summary: str
    language: str
    pub_date: datetime
    source_id: str
    article_url: str


@pytest.mark.asyncio
async def test_a_nonempty_asknews_read_is_summarized_exactly_once(mock_os_getenv):
    """The mirror of the empty-read skip above, so the skip cannot be silently widened
    into "never summarize": articles in, summarizer awaited once, briefing in the bundle."""
    mock_os_getenv.side_effect = lambda x, default=None: {
        "RESEARCH_PROVIDER": "asknews",
        "ASKNEWS_CLIENT_ID": "client",
        "ASKNEWS_SECRET": "secret",
    }.get(x, default)

    bot = TemplateForecaster(
        llms={
            "default": "mock_default_model",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
            "summarizer": "mock_summarizer",
        }
    )
    q = MetaculusQuestion(
        question_text="Test",
        page_url="http://example.com",
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
    )

    article = _StubArticle(
        eng_title="Something happened",
        summary="A decision-relevant fact.",
        language="en",
        pub_date=datetime(2026, 8, 20, tzinfo=UTC),
        source_id="src",
        article_url="http://example.com/article",
    )
    with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
        mock_sdk = AsyncMock()
        mock_response = AsyncMock()
        mock_response.as_dicts = [article]
        mock_sdk.news.search_news.return_value = mock_response
        mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

        with patch.object(
            ResearchOrchestrator,
            "_summarize_asknews",
            new_callable=AsyncMock,
            return_value="## News Analysis (AskNews)\n\nAnalyst briefing.",
        ) as summarize:
            res = await bot.run_research(q)

    summarize.assert_awaited_once()
    assert "Analyst briefing." in res


@pytest.mark.asyncio
async def test_gemini_search_flag_logs_provider_name(mock_os_getenv, caplog):
    """With GEMINI_SEARCH_ENABLED=true, the parallel providers list is logged including gemini_search."""
    mock_os_getenv.side_effect = lambda x, default=None: {
        "RESEARCH_PROVIDER": "asknews",
        "ASKNEWS_CLIENT_ID": "client",
        "ASKNEWS_SECRET": "secret",
        "GEMINI_SEARCH_ENABLED": "true",
        "GOOGLE_API_KEY": "fake-key",
    }.get(x, default)

    bot = TemplateForecaster(
        llms={
            "default": "mock_default_model",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
            "summarizer": "mock_summarizer",
        }
    )
    q = MetaculusQuestion(
        question_text="Test",
        page_url="http://example.com",
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
    )

    # Stub the AskNews SDK (primary provider) to return immediately.
    with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
        mock_sdk = AsyncMock()
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        mock_sdk.news.search_news.return_value = mock_response
        mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

        # Stub the Gemini provider fetch path — we only care that selection happened
        # and that the provider name is logged; we don't want a live API call.
        async def _fake_gemini(question_text: str) -> str:
            await asyncio.sleep(0)
            return f"stub gemini research for {question_text[:0]}"

        with (
            patch(
                "metaculus_bot.research.gemini_search.gemini_search_provider",
                return_value=_fake_gemini,
            ),
            caplog.at_level(logging.INFO),
        ):
            await bot.run_research(q)

    provider_log_messages = [rec.message for rec in caplog.records if "Using research providers:" in rec.message]
    assert provider_log_messages, "expected a 'Using research providers:' log line"
    assert any("gemini_search" in msg for msg in provider_log_messages)


@pytest.mark.asyncio
async def test_gemini_search_params_passed_through(mock_os_getenv, caplog):
    """gemini_search_provider receives the env-configured model and bot's is_benchmarking flag."""
    mock_os_getenv.side_effect = lambda x, default=None: {
        "RESEARCH_PROVIDER": "asknews",
        "ASKNEWS_CLIENT_ID": "client",
        "ASKNEWS_SECRET": "secret",
        "GEMINI_SEARCH_ENABLED": "true",
        "GEMINI_SEARCH_MODEL": "gemini-2.5-flash",
        "GOOGLE_API_KEY": "fake-key",
    }.get(x, default)

    bot = TemplateForecaster(
        llms={
            "default": "mock_default_model",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
            "summarizer": "mock_summarizer",
        }
    )
    # Explicitly set is_benchmarking to a known value so we can assert it propagates.
    bot.is_benchmarking = False
    q = MetaculusQuestion(
        question_text="Test",
        page_url="http://example.com",
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
    )

    with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
        mock_sdk = AsyncMock()
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        mock_sdk.news.search_news.return_value = mock_response
        mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

        async def _fake_gemini(question_text: str) -> str:
            await asyncio.sleep(0)
            return f"stub gemini research for {question_text[:0]}"

        with patch(
            "metaculus_bot.research.gemini_search.gemini_search_provider",
            return_value=_fake_gemini,
        ) as fake_provider:
            await bot.run_research(q)

    fake_provider.assert_called_once()
    assert fake_provider.call_args.args == ("gemini-2.5-flash",)
    assert fake_provider.call_args.kwargs == {"is_benchmarking": False}


@pytest.mark.asyncio
async def test_gemini_search_flag_disabled_excludes_provider(mock_os_getenv, caplog):
    """With GEMINI_SEARCH_ENABLED unset/false, gemini_search is not in the parallel provider list."""
    mock_os_getenv.side_effect = lambda x, default=None: {
        "RESEARCH_PROVIDER": "asknews",
        "ASKNEWS_CLIENT_ID": "client",
        "ASKNEWS_SECRET": "secret",
        # GEMINI_SEARCH_ENABLED deliberately absent
    }.get(x, default)

    bot = TemplateForecaster(
        llms={
            "default": "mock_default_model",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
            "summarizer": "mock_summarizer",
        }
    )
    q = MetaculusQuestion(
        question_text="Test",
        page_url="http://example.com",
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
    )

    with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
        mock_sdk = AsyncMock()
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        mock_sdk.news.search_news.return_value = mock_response
        mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

        with caplog.at_level(logging.INFO):
            await bot.run_research(q)

    provider_log_messages = [rec.message for rec in caplog.records if "Using research providers:" in rec.message]
    assert provider_log_messages
    # No "gemini_search" in any of the provider log lines.
    assert not any("gemini_search" in msg for msg in provider_log_messages)


# Gap 2: prediction-market provider integration with _run_providers_parallel
#
# Earlier tests in tests/test_prediction_market_provider.py cover the provider
# in isolation. The tests below verify it slots correctly into the full
# parallel research fan-out — that its output appears alongside AskNews +
# Gemini outputs, that flag-off keeps it out of the provider list, and that
# the F1 ``MetaculusQuestion`` plumbing reaches the snapshot orchestrator
# end-to-end (carrying ``scheduled_resolution_time`` for as_of derivation).


_PM_HANDLERS_SHARED: dict[str, list] = {}


def _make_polymarket_payload() -> dict:
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
            }
        ],
        "markets": [],
    }


def _make_manifold_payload() -> list:
    return [
        {
            "id": "abc",
            "question": "Will Starship reach orbit before July 2026?",
            "slug": "starship-orbit-july-2026",
            "creatorUsername": "spaceFan",
            "probability": 0.62,
            "volume24Hours": 500.0,
            "closeTime": 1782086400000,  # epoch ms, future close date
            "textDescription": "YES if SpaceX Starship reaches orbit before July 1 2026.",
            "isResolved": False,
        }
    ]


def _make_kalshi_events_payload() -> dict:
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
                        "status": "active",
                    }
                ],
            }
        ],
        "cursor": "",
    }


# The two market-retrieval LLM stages, in the shapes their parsers accept. A prose stub fails
# BOTH: `parse_query_author` returns `()` (an additive loss) and `parse_ranking` raises, which
# fails open to an untiered slate and therefore to the NEUTRAL preamble — so the research-content
# assertions below would be measuring a degraded pipeline.
_CANNED_QUERY_AUTHOR = json.dumps({"synonyms": ["orbital launch"], "framings": ["Starship orbit"]})
_CANNED_RANKING = json.dumps([{"i": 0, "tier": "same_quantity_same_date", "why": "same event, same window"}])


class _FakeMarketLlm:
    """Covers both market stages, routed on the prompt each one builds."""

    def __init__(self, **_kwargs):
        pass

    async def invoke(self, prompt: str) -> str:
        await asyncio.sleep(0)
        return _CANNED_RANKING if "Rank the candidates by EVIDENTIAL VALUE" in prompt else _CANNED_QUERY_AUTHOR


class _FakeContent:
    """Streams the payload as JSON bytes for the Kalshi catalogue stream-parse path."""

    def __init__(self, payload):
        self._data = b"" if payload is None else json.dumps(payload).encode()

    async def iter_chunked(self, n: int):
        step = max(1, n)
        for i in range(0, len(self._data), step):
            yield self._data[i : i + step]


class _FakeResponse:
    def __init__(self, status: int, payload):
        self.status = status
        self._payload = payload
        self.content = _FakeContent(payload)

    async def json(self):
        await asyncio.sleep(0)
        return self._payload

    async def text(self):
        await asyncio.sleep(0)
        return ""

    async def __aenter__(self):
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *_args):
        await asyncio.sleep(0)
        return


class _FakeSession:
    """Routes HTTP GETs to in-memory payload handlers keyed by URL prefix."""

    def __init__(self, handlers: dict):
        self._handlers = handlers
        self.closed = False

    def get(self, url: str, _params=None, **_kwargs):
        for prefix, response in self._handlers.items():
            if url.startswith(prefix):
                return response
        raise AssertionError(f"unhandled URL {url}")

    async def close(self):
        await asyncio.sleep(0)
        self.closed = True

    async def __aenter__(self):
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, *_args):
        await self.close()


@pytest.fixture(autouse=False)
def _reset_pm_caches():
    from metaculus_bot.research.prediction_market import _reset_session_caches

    _reset_session_caches()
    yield
    _reset_session_caches()


@pytest.mark.asyncio
@pytest.mark.usefixtures("_reset_pm_caches")
async def test_prediction_market_provider_integrates_with_run_providers_parallel(mock_os_getenv, caplog):
    """Flag ON + ASKNEWS creds + GEMINI creds + PREDICTION_MARKETS_ENABLED:
    all three providers should run in parallel and combined research must
    contain blocks from each. Verifies end-to-end that:
    - the prediction-market provider is registered in the providers list,
    - it receives the MetaculusQuestion (not a string), and
    - its formatted output lands in combined_research."""
    mock_os_getenv.side_effect = lambda x, default=None: {
        "RESEARCH_PROVIDER": "asknews",
        "ASKNEWS_CLIENT_ID": "client",
        "ASKNEWS_SECRET": "secret",
        "GEMINI_SEARCH_ENABLED": "true",
        "GOOGLE_API_KEY": "fake-key",
        "PREDICTION_MARKETS_ENABLED": "true",
    }.get(x, default)

    bot = TemplateForecaster(
        llms={
            "default": "mock_default_model",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
            "summarizer": "mock_summarizer",
        }
    )
    # Real MetaculusQuestion so the F1 plumbing flows through. Use a future
    # scheduled_resolution_time so as_of derivation doesn't drop the test
    # close_times (which are in 2026).
    q = MetaculusQuestion(
        question_text="Will SpaceX Starship reach orbit?",
        page_url="http://example.com/q/1",
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
    )
    q.id_of_question = 99999  # type: ignore[attr-defined]
    q.scheduled_resolution_time = datetime(2027, 1, 1, tzinfo=UTC)
    q.resolution_criteria = "Resolves Yes if a SpaceX Starship reaches orbital velocity."

    # AskNews stub.
    asknews_class_path = "asknews_sdk.AsyncAskNewsSDK"

    # Gemini provider stub: short-circuit the actual Google Search call.
    async def _fake_gemini(question_text: str) -> str:
        await asyncio.sleep(0)
        return "Gemini grounded research output for this query."

    # Prediction-market provider stubs: a fake aiohttp session + a fake LLM.
    handlers = {
        "https://gamma-api.polymarket.com/public-search": _FakeResponse(200, _make_polymarket_payload()),
        "https://api.manifold.markets/v0/search-markets": _FakeResponse(200, _make_manifold_payload()),
        "https://api.elections.kalshi.com/trade-api/v2/events": _FakeResponse(200, _make_kalshi_events_payload()),
        "https://api.manifold.markets/v0/market": _FakeResponse(200, {}),
        "https://www.predictit.org/api/marketdata/all/": _FakeResponse(200, {"markets": []}),
    }

    captured_questions: list = []
    from metaculus_bot.research import prediction_market as pmp

    original_fetch = pmp.fetch_market_snapshot

    async def _capturing_fetch(question_arg, **kwargs):
        captured_questions.append(question_arg)
        return await original_fetch(question_arg, **kwargs)

    with patch(asknews_class_path) as mock_sdk_class:
        mock_sdk = AsyncMock()
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        mock_sdk.news.search_news.return_value = mock_response
        mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

        with (
            patch(
                "metaculus_bot.research.gemini_search.gemini_search_provider",
                return_value=_fake_gemini,
            ),
            patch.object(pmp, "build_llm_with_openrouter_fallback", _FakeMarketLlm),
            patch.object(pmp, "_get_session", lambda: _FakeSession(handlers)),
            patch.object(pmp, "fetch_market_snapshot", _capturing_fetch),
            caplog.at_level(logging.INFO),
        ):
            research = await bot.run_research(q)

    # F1 plumbing: the snapshot orchestrator received the real
    # MetaculusQuestion, not a string. Confirms the ResearchCallable signature
    # widening went through to the snapshot fetch.
    assert captured_questions, "fetch_market_snapshot was never called"
    assert captured_questions[0] is q

    # Provider registration: the log line lists all three providers.
    provider_log = next(rec.message for rec in caplog.records if "Using research providers:" in rec.message)
    assert "asknews" in provider_log
    assert "gemini_search" in provider_log
    assert "prediction_market" in provider_log

    # Combined research carries the prediction-market block + Gemini block. AskNews
    # returns no articles, so it contributes NO section — the assertion here used to be
    # that its header rendered over a "No articles were found" sentence, which is the
    # sentinel that made an empty read look like research.
    assert "Gemini grounded research output" in research
    assert "## News Articles (AskNews)" not in research
    # Prediction-market formatter leads with the strong-evidence header, which the canned
    # ranking earns by grading its one row `same_quantity_same_date` — verify criteria, date and
    # topic before weighting.
    assert "MAY be relevant" in research
    assert "verify each market's resolution criteria" in research


@pytest.mark.asyncio
async def test_prediction_market_provider_disabled_flag_excludes_from_parallel(mock_os_getenv, caplog):
    """With PREDICTION_MARKETS_ENABLED unset/false, the provider must not be in
    the parallel provider list — and crucially, no HTTP calls are issued."""
    mock_os_getenv.side_effect = lambda x, default=None: {
        "RESEARCH_PROVIDER": "asknews",
        "ASKNEWS_CLIENT_ID": "client",
        "ASKNEWS_SECRET": "secret",
        # PREDICTION_MARKETS_ENABLED deliberately absent
    }.get(x, default)

    bot = TemplateForecaster(
        llms={
            "default": "mock_default_model",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
            "summarizer": "mock_summarizer",
        }
    )
    q = MetaculusQuestion(
        question_text="Test",
        page_url="http://example.com",
        open_time=datetime(2026, 1, 1, tzinfo=UTC),
    )

    # Sentinel: if the provider runs, it would try to instantiate a session
    # via _get_session. Patch it to a sentinel that raises AssertionError if
    # called — proving no HTTP work happens.
    from metaculus_bot.research import prediction_market as pmp

    def _should_not_be_called():
        raise AssertionError("PM provider's _get_session must not be called when flag is off")

    with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
        mock_sdk = AsyncMock()
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        mock_sdk.news.search_news.return_value = mock_response
        mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

        with patch.object(pmp, "_get_session", _should_not_be_called), caplog.at_level(logging.INFO):
            await bot.run_research(q)

    provider_log = next(rec.message for rec in caplog.records if "Using research providers:" in rec.message)
    assert "prediction_market" not in provider_log


@pytest.mark.asyncio
@pytest.mark.usefixtures("_reset_pm_caches")
async def test_prediction_market_provider_passes_no_as_of_through_the_orchestrator(mock_os_getenv):
    """The provider path passes ``as_of=None``, even for a question that HAS a scheduled
    resolution — asserted here at the orchestrator level, where a future caller might inject one.

    The old derivation (``scheduled_resolution_time - 1 day``) was worse than inert. It dropped
    every market closing before the question resolved, which is precisely the "same quantity,
    adjacent month" class that supplied most of the ranked arm's near-identical rows, and prod
    telemetry recorded the cost: 20 of 47 archived runs had Polymarket fetch candidates and render
    nothing because of it. Backtest leakage is defended by the ``is_benchmarking`` guard, which
    hard-disables the provider outright; ``as_of`` survives only for explicit callers.
    """
    mock_os_getenv.side_effect = lambda x, default=None: {
        "RESEARCH_PROVIDER": "asknews",
        "ASKNEWS_CLIENT_ID": "client",
        "ASKNEWS_SECRET": "secret",
        "PREDICTION_MARKETS_ENABLED": "true",
    }.get(x, default)

    bot = TemplateForecaster(
        llms={
            "default": "mock_default_model",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
            "summarizer": "mock_summarizer",
        }
    )
    q = MetaculusQuestion(question_text="Will event X happen?", page_url="http://example.com/q/2")
    q.id_of_question = 7777  # type: ignore[attr-defined]
    q.scheduled_resolution_time = datetime(2027, 8, 1, tzinfo=UTC)
    q.resolution_criteria = "Resolves yes if X happens by deadline."

    handlers = {
        "https://gamma-api.polymarket.com/public-search": _FakeResponse(200, _make_polymarket_payload()),
        "https://api.manifold.markets/v0/search-markets": _FakeResponse(200, _make_manifold_payload()),
        "https://api.manifold.markets/v0/market": _FakeResponse(200, {}),
        "https://api.elections.kalshi.com/trade-api/v2/events": _FakeResponse(200, _make_kalshi_events_payload()),
        "https://www.predictit.org/api/marketdata/all/": _FakeResponse(200, {"markets": []}),
    }

    from metaculus_bot.research import prediction_market as pmp

    captured_as_ofs: list = []
    original_fetch = pmp.fetch_market_snapshot

    async def _capturing_fetch(question_arg, *, as_of=None, **kwargs):
        captured_as_ofs.append(as_of)
        return await original_fetch(question_arg, as_of=as_of, **kwargs)

    with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
        mock_sdk = AsyncMock()
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        mock_sdk.news.search_news.return_value = mock_response
        mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", _FakeMarketLlm),
            patch.object(pmp, "_get_session", lambda: _FakeSession(handlers)),
            patch.object(pmp, "fetch_market_snapshot", _capturing_fetch),
        ):
            await bot.run_research(q)

    assert captured_as_ofs == [None]
