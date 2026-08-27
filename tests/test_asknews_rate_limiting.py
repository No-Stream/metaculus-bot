"""Test that AskNews integration properly handles rate limiting."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metaculus_bot.research.providers import _asknews_provider


def _make_q(text: str) -> MagicMock:
    """Build a minimal MetaculusQuestion-shaped mock for the new ResearchCallable
    contract."""
    q = MagicMock()
    q.question_text = text
    return q


@pytest.mark.asyncio
async def test_asknews_rate_limiting_delay():
    """Test that AskNews provider waits between API calls to respect rate limits."""

    # Mock the AsyncAskNewsSDK to track timing
    call_times = []

    async def mock_search_news(*args, **kwargs):
        call_times.append(asyncio.get_event_loop().time())
        # Create a minimal mock response
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        return mock_response

    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: {
            "ASKNEWS_CLIENT_ID": "test_client_id",
            "ASKNEWS_SECRET": "test_secret",
        }.get(key, default)

        with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
            mock_sdk = AsyncMock()
            mock_sdk.news.search_news = mock_search_news
            mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

            provider = _asknews_provider()
            await provider(_make_q("test question"))

            # Verify two calls were made
            assert len(call_times) == 2

            # Verify there was a delay between calls (should be ~1.2 seconds)
            time_diff = call_times[1] - call_times[0]
            assert time_diff >= 10.0, f"Expected delay >= 10.0s, got {time_diff:.2f}s"
            assert time_diff <= 11.1, f"Expected delay <= 11.1s, got {time_diff:.2f}s"


@pytest.mark.asyncio
async def test_asknews_calls_both_endpoints():
    """Test that AskNews provider calls both latest and historical news endpoints."""

    search_calls = []

    async def mock_search_news(*args, **kwargs):
        search_calls.append(kwargs.get("strategy", "unknown"))
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        return mock_response

    with patch("os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: {
            "ASKNEWS_CLIENT_ID": "test_client_id",
            "ASKNEWS_SECRET": "test_secret",
        }.get(key, default)

        with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
            mock_sdk = AsyncMock()
            mock_sdk.news.search_news = mock_search_news
            mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

            provider = _asknews_provider()
            result = await provider(_make_q("test question"))

            # Verify both strategies were called
            assert "latest news" in search_calls
            assert "news knowledge" in search_calls
            assert len(search_calls) == 2

            # Both endpoints fired and returned nothing, so the provider contributes "".
            # The old assertion here was the "No articles were found" sentence, which read
            # downstream as research; this test's subject is the two-endpoint call pattern
            # above, which is unchanged.
            assert result == ""


class TestAskNewsPhaseRetryBudget:
    """Behavior pins for the two-phase retry ladder and its shared attempt budget.

    Both phases retry only transient rate/concurrency errors, and the HISTORICAL
    phase's budget is what the HOT phase left over — so a HOT call that burned two
    attempts leaves HISTORICAL fewer. Those are the rules a reader of the phase
    helper has to preserve, so they are pinned directly here.
    """

    @staticmethod
    async def _run(search_news, *, tries: int = 3) -> str:
        """Drive the provider against a scripted ``sdk.news.search_news``, sleeps stubbed out."""
        with (
            patch("os.getenv") as mock_getenv,
            patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class,
            patch("metaculus_bot.research.providers.ASKNEWS_MAX_TRIES", tries),
            patch("metaculus_bot.research.providers.ASKNEWS_BACKOFF_SECS", 0.0),
            patch("asyncio.sleep", new=AsyncMock()),
        ):
            mock_getenv.side_effect = lambda key, default=None: {
                "ASKNEWS_CLIENT_ID": "test_client_id",
                "ASKNEWS_SECRET": "test_secret",
            }.get(key, default)
            mock_sdk = AsyncMock()
            mock_sdk.news.search_news = search_news
            mock_sdk_class.return_value.__aenter__.return_value = mock_sdk
            return await _asknews_provider()(_make_q("test question"))

    @staticmethod
    def _empty_response() -> AsyncMock:
        response = AsyncMock()
        response.as_dicts = []
        return response

    @pytest.mark.asyncio
    async def test_rate_limited_hot_call_retries_then_succeeds(self) -> None:
        attempts: list[str] = []

        async def search_news(*_args, **kwargs):
            strategy = kwargs["strategy"]
            attempts.append(strategy)
            if strategy == "latest news" and attempts.count("latest news") == 1:
                raise RuntimeError("429 rate limit exceeded")
            return TestAskNewsPhaseRetryBudget._empty_response()

        assert await self._run(search_news) == ""
        assert attempts == ["latest news", "latest news", "news knowledge"]

    @pytest.mark.asyncio
    async def test_non_retryable_hot_error_raises_on_the_first_attempt(self) -> None:
        attempts: list[str] = []

        async def search_news(*_args, **kwargs):
            attempts.append(kwargs["strategy"])
            raise RuntimeError("400 malformed query")

        with pytest.raises(RuntimeError, match="malformed query"):
            await self._run(search_news)
        assert attempts == ["latest news"]

    @pytest.mark.asyncio
    async def test_exhausted_hot_retries_reraise_the_last_error(self) -> None:
        attempts: list[str] = []

        async def search_news(*_args, **kwargs):
            attempts.append(kwargs["strategy"])
            raise RuntimeError(f"concurrency limit hit #{len(attempts)}")

        with pytest.raises(RuntimeError, match="#3"):
            await self._run(search_news, tries=3)
        assert attempts == ["latest news"] * 3

    @pytest.mark.asyncio
    async def test_historical_budget_is_what_hot_left_over(self) -> None:
        """HOT burning 2 of 3 attempts leaves HISTORICAL 2 (``tries - (hot_used - 1)``)."""
        attempts: list[str] = []

        async def search_news(*_args, **kwargs):
            strategy = kwargs["strategy"]
            attempts.append(strategy)
            if strategy == "latest news" and attempts.count("latest news") == 1:
                raise RuntimeError("429 rate limit")
            if strategy == "news knowledge":
                raise RuntimeError("429 rate limit")
            return TestAskNewsPhaseRetryBudget._empty_response()

        with pytest.raises(RuntimeError, match="429"):
            await self._run(search_news, tries=3)
        assert attempts.count("latest news") == 2
        assert attempts.count("news knowledge") == 2

    @pytest.mark.asyncio
    async def test_non_retryable_historical_error_raises_on_the_first_attempt(self) -> None:
        attempts: list[str] = []

        async def search_news(*_args, **kwargs):
            strategy = kwargs["strategy"]
            attempts.append(strategy)
            if strategy == "news knowledge":
                raise RuntimeError("500 upstream exploded")
            return TestAskNewsPhaseRetryBudget._empty_response()

        with pytest.raises(RuntimeError, match="upstream exploded"):
            await self._run(search_news)
        assert attempts == ["latest news", "news knowledge"]
