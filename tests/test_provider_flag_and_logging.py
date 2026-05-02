import asyncio
import logging
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import MetaculusQuestion

from main import TemplateForecaster

_ = asyncio  # used inside nested async def stubs defined in tests below


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
    q = MetaculusQuestion(question_text="Test", page_url="http://example.com")

    with patch("asknews_sdk.AsyncAskNewsSDK") as mock_sdk_class:
        # Mock the SDK to return our test result
        mock_sdk = AsyncMock()
        mock_response = AsyncMock()
        mock_response.as_dicts = []
        mock_sdk.news.search_news.return_value = mock_response
        mock_sdk_class.return_value.__aenter__.return_value = mock_sdk

        # Since no articles are returned, it should return the message with provider header
        with caplog.at_level(logging.INFO):
            res = await bot.run_research(q)
        assert "No articles were found for this query." in res
        assert "## News Articles (AskNews)" in res
        assert any("Using research providers:" in rec.message for rec in caplog.records)


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
    q = MetaculusQuestion(question_text="Test", page_url="http://example.com")

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

        with patch(
            "metaculus_bot.gemini_search_provider.gemini_search_provider",
            return_value=_fake_gemini,
        ):
            with caplog.at_level(logging.INFO):
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
    q = MetaculusQuestion(question_text="Test", page_url="http://example.com")

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
            "metaculus_bot.gemini_search_provider.gemini_search_provider",
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
    q = MetaculusQuestion(question_text="Test", page_url="http://example.com")

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
