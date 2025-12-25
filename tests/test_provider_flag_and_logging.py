import logging
from unittest.mock import AsyncMock, patch

import pytest
from forecasting_tools import MetaculusQuestion

from main import TemplateForecaster


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
