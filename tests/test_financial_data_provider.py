"""Tests for the financial data research provider (yfinance + FRED)."""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------


class TestClassifyFinancialQuestion:
    """Tests for _classify_financial_question."""

    @pytest.mark.asyncio
    async def test_classifies_stock_question_with_tickers(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: AAPL, MSFT\nFRED_SERIES: NONE"

        from metaculus_bot.financial_data_provider import _classify_financial_question

        result = await _classify_financial_question("Will Apple stock price exceed $200 by end of 2026?", mock_llm)

        assert result is not None
        assert result["tickers"] == ["AAPL", "MSFT"]
        assert result["fred_series"] == []

    @pytest.mark.asyncio
    async def test_classifies_economic_question_with_fred_series(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: NONE\nFRED_SERIES: UNRATE, CPIAUCSL"

        from metaculus_bot.financial_data_provider import _classify_financial_question

        result = await _classify_financial_question("Will US unemployment rate exceed 5% in 2026?", mock_llm)

        assert result is not None
        assert result["tickers"] == []
        assert result["fred_series"] == ["UNRATE", "CPIAUCSL"]

    @pytest.mark.asyncio
    async def test_classifies_mixed_question_with_both(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: ^GSPC\nFRED_SERIES: FEDFUNDS"

        from metaculus_bot.financial_data_provider import _classify_financial_question

        result = await _classify_financial_question("Will the S&P 500 drop if the Fed raises rates?", mock_llm)

        assert result is not None
        assert result["tickers"] == ["^GSPC"]
        assert result["fred_series"] == ["FEDFUNDS"]

    @pytest.mark.asyncio
    async def test_non_financial_question_returns_none(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: NO\nTICKERS: NONE\nFRED_SERIES: NONE"

        from metaculus_bot.financial_data_provider import _classify_financial_question

        result = await _classify_financial_question("Will it rain in London tomorrow?", mock_llm)

        assert result is None

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM timeout")

        from metaculus_bot.financial_data_provider import _classify_financial_question

        result = await _classify_financial_question("Will Apple stock exceed $200?", mock_llm)

        assert result is None

    @pytest.mark.asyncio
    async def test_malformed_llm_response_returns_none(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "I don't understand the question format."

        from metaculus_bot.financial_data_provider import _classify_financial_question

        result = await _classify_financial_question("Will Apple stock exceed $200?", mock_llm)

        assert result is None

    @pytest.mark.asyncio
    async def test_financial_yes_but_no_tickers_or_series_returns_none(self) -> None:
        """If classifier says YES but extracts nothing useful, treat as non-financial."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: NONE\nFRED_SERIES: NONE"

        from metaculus_bot.financial_data_provider import _classify_financial_question

        result = await _classify_financial_question("Will the economy improve?", mock_llm)

        assert result is None


# ---------------------------------------------------------------------------
# yfinance fetch tests
# ---------------------------------------------------------------------------


class TestFetchYfinanceData:
    """Tests for _fetch_yfinance_data."""

    def test_valid_ticker_returns_markdown_with_key_fields(self) -> None:
        dates = pd.date_range(end="2026-03-30", periods=252, freq="B")
        close_prices = np.linspace(150.0, 200.0, 252)
        mock_history = pd.DataFrame(
            {
                "Close": close_prices,
                "Open": close_prices * 0.99,
                "High": close_prices * 1.01,
                "Low": close_prices * 0.98,
            },
            index=dates,
        )

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = mock_history
        mock_ticker_instance.info = {
            "shortName": "Apple Inc.",
            "regularMarketPrice": 200.0,
            "trailingPE": 28.5,
            "marketCap": 3_000_000_000_000,
            "forwardEps": 7.5,
        }

        with patch("metaculus_bot.financial_data_provider.yfinance") as mock_yf:
            mock_yf.Ticker.return_value = mock_ticker_instance

            from metaculus_bot.financial_data_provider import _fetch_yfinance_data

            result = _fetch_yfinance_data("AAPL")

        assert result != ""
        assert "AAPL" in result
        assert "200.0" in result or "200.00" in result
        # Should contain return calculations
        assert "return" in result.lower() or "change" in result.lower()
        # Should contain volatility
        assert "volatil" in result.lower()

    def test_yfinance_exception_returns_empty_string(self) -> None:
        with patch("metaculus_bot.financial_data_provider.yfinance") as mock_yf:
            mock_yf.Ticker.side_effect = Exception("Network error")

            from metaculus_bot.financial_data_provider import _fetch_yfinance_data

            result = _fetch_yfinance_data("INVALID")

        assert result == ""

    def test_empty_history_returns_empty_string(self) -> None:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker_instance.info = {}

        with patch("metaculus_bot.financial_data_provider.yfinance") as mock_yf:
            mock_yf.Ticker.return_value = mock_ticker_instance

            from metaculus_bot.financial_data_provider import _fetch_yfinance_data

            result = _fetch_yfinance_data("FAKE")

        assert result == ""


# ---------------------------------------------------------------------------
# FRED fetch tests
# ---------------------------------------------------------------------------


class TestFetchFredData:
    """Tests for _fetch_fred_data."""

    def test_valid_series_returns_markdown_with_key_fields(self) -> None:
        dates = pd.date_range(end="2026-03-01", periods=60, freq="MS")
        values = np.linspace(3.5, 4.2, 60)
        mock_series = pd.Series(values, index=dates, name="UNRATE")

        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.return_value = mock_series
        mock_fred_instance.get_series_info.return_value = pd.DataFrame(
            {"title": ["Unemployment Rate"]}, index=["UNRATE"]
        )

        with patch("metaculus_bot.financial_data_provider.Fred") as mock_fred_class:
            mock_fred_class.return_value = mock_fred_instance

            from metaculus_bot.financial_data_provider import _fetch_fred_data

            result = _fetch_fred_data("UNRATE", "fake_api_key")

        assert result != ""
        assert "UNRATE" in result
        # Should contain the latest value
        assert "4.2" in result or "4.20" in result

    def test_fred_exception_returns_empty_string(self) -> None:
        with patch("metaculus_bot.financial_data_provider.Fred") as mock_fred_class:
            mock_fred_class.return_value.get_series.side_effect = Exception("API error")

            from metaculus_bot.financial_data_provider import _fetch_fred_data

            result = _fetch_fred_data("INVALID", "fake_api_key")

        assert result == ""


# ---------------------------------------------------------------------------
# Integration tests (full provider flow)
# ---------------------------------------------------------------------------


class TestFinancialDataProviderIntegration:
    """Integration tests for the full financial_data_provider flow."""

    @pytest.mark.asyncio
    async def test_financial_question_returns_combined_markdown(self) -> None:
        """Full flow: financial question -> classify -> fetch -> combined output."""
        # Mock the classifier LLM
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: AAPL\nFRED_SERIES: UNRATE"

        # Mock yfinance
        dates = pd.date_range(end="2026-03-30", periods=252, freq="B")
        close_prices = np.linspace(150.0, 200.0, 252)
        mock_history = pd.DataFrame(
            {"Close": close_prices},
            index=dates,
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_history
        mock_ticker.info = {"shortName": "Apple Inc.", "regularMarketPrice": 200.0}

        # Mock FRED
        fred_dates = pd.date_range(end="2026-03-01", periods=60, freq="MS")
        fred_values = np.linspace(3.5, 4.2, 60)
        mock_fred_series = pd.Series(fred_values, index=fred_dates, name="UNRATE")
        mock_fred_instance = MagicMock()
        mock_fred_instance.get_series.return_value = mock_fred_series
        mock_fred_instance.get_series_info.return_value = pd.DataFrame(
            {"title": ["Unemployment Rate"]}, index=["UNRATE"]
        )

        with (
            patch("metaculus_bot.financial_data_provider.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.financial_data_provider.yfinance") as mock_yf,
            patch("metaculus_bot.financial_data_provider.Fred") as mock_fred_class,
        ):
            mock_yf.Ticker.return_value = mock_ticker
            mock_fred_class.return_value = mock_fred_instance

            from metaculus_bot.financial_data_provider import financial_data_provider

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("FRED_API_KEY", "fake_key")
            try:
                provider = financial_data_provider()
                result = await provider("Will Apple stock price exceed $200 and will unemployment stay below 5%?")
            finally:
                monkeypatch.undo()

        assert result != ""
        assert "AAPL" in result
        assert "UNRATE" in result

    @pytest.mark.asyncio
    async def test_non_financial_question_returns_empty(self) -> None:
        """Full flow: non-financial question -> classify as NO -> return empty."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: NO\nTICKERS: NONE\nFRED_SERIES: NONE"

        with patch("metaculus_bot.financial_data_provider.build_llm_with_openrouter_fallback", return_value=mock_llm):
            from metaculus_bot.financial_data_provider import financial_data_provider

            provider = financial_data_provider()
            result = await provider("Will it rain in London tomorrow?")

        assert result == ""

    @pytest.mark.asyncio
    async def test_partial_failure_still_returns_data(self) -> None:
        """If one fetch fails, other successful fetches still produce output."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: AAPL, BADTICKER\nFRED_SERIES: NONE"

        # First ticker works, second raises
        dates = pd.date_range(end="2026-03-30", periods=252, freq="B")
        close_prices = np.linspace(150.0, 200.0, 252)
        mock_history = pd.DataFrame({"Close": close_prices}, index=dates)

        good_ticker = MagicMock()
        good_ticker.history.return_value = mock_history
        good_ticker.info = {"shortName": "Apple Inc.", "regularMarketPrice": 200.0}

        bad_ticker = MagicMock()
        bad_ticker.history.side_effect = Exception("Ticker not found")

        def ticker_factory(symbol: str) -> MagicMock:
            if symbol == "AAPL":
                return good_ticker
            return bad_ticker

        with (
            patch("metaculus_bot.financial_data_provider.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.financial_data_provider.yfinance") as mock_yf,
        ):
            mock_yf.Ticker.side_effect = ticker_factory

            from metaculus_bot.financial_data_provider import financial_data_provider

            provider = financial_data_provider()
            result = await provider("Compare Apple and BADTICKER stock performance")

        assert "AAPL" in result
        # BADTICKER should not appear (its fetch failed and returned "")
        assert "BADTICKER" not in result

    @pytest.mark.asyncio
    async def test_missing_fred_key_skips_fred_fetches(self) -> None:
        """If FRED_API_KEY is not set, FRED fetches are skipped, yfinance still works."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: AAPL\nFRED_SERIES: UNRATE"

        dates = pd.date_range(end="2026-03-30", periods=252, freq="B")
        close_prices = np.linspace(150.0, 200.0, 252)
        mock_history = pd.DataFrame({"Close": close_prices}, index=dates)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_history
        mock_ticker.info = {"shortName": "Apple Inc.", "regularMarketPrice": 200.0}

        with (
            patch("metaculus_bot.financial_data_provider.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.financial_data_provider.yfinance") as mock_yf,
        ):
            mock_yf.Ticker.return_value = mock_ticker

            from metaculus_bot.financial_data_provider import financial_data_provider

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.delenv("FRED_API_KEY", raising=False)
            try:
                provider = financial_data_provider()
                result = await provider("Will Apple stock rise and unemployment fall?")
            finally:
                monkeypatch.undo()

        assert "AAPL" in result
        # FRED data should not appear since no API key
        assert "UNRATE" not in result


# ---------------------------------------------------------------------------
# Provider selection tests (env var gating)
# ---------------------------------------------------------------------------


class TestProviderSelection:
    """Test that FINANCIAL_DATA_ENABLED env var gates provider inclusion."""

    def test_provider_included_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FINANCIAL_DATA_ENABLED", "true")
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from main import TemplateForecaster

        with (
            patch.object(TemplateForecaster, "__init__", lambda self: None),
            patch(
                "metaculus_bot.financial_data_provider.build_llm_with_openrouter_fallback",
                return_value=AsyncMock(),
            ),
        ):
            bot = TemplateForecaster.__new__(TemplateForecaster)
            bot._custom_research_provider = None
            bot.is_benchmarking = False
            bot.allow_research_fallback = True

            async def mock_provider(q: str) -> str:
                return "primary research"

            with patch.object(bot, "_select_research_provider", return_value=(mock_provider, "asknews")):
                providers = bot._select_research_providers()

        provider_names = [name for _, name in providers]
        assert "financial_data" in provider_names

    def test_provider_excluded_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FINANCIAL_DATA_ENABLED", "false")
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from main import TemplateForecaster

        with patch.object(TemplateForecaster, "__init__", lambda self: None):
            bot = TemplateForecaster.__new__(TemplateForecaster)
            bot._custom_research_provider = None
            bot.is_benchmarking = False
            bot.allow_research_fallback = True

            async def mock_provider(q: str) -> str:
                return "primary research"

            with patch.object(bot, "_select_research_provider", return_value=(mock_provider, "asknews")):
                providers = bot._select_research_providers()

        provider_names = [name for _, name in providers]
        assert "financial_data" not in provider_names

    def test_provider_excluded_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from main import TemplateForecaster

        with patch.object(TemplateForecaster, "__init__", lambda self: None):
            bot = TemplateForecaster.__new__(TemplateForecaster)
            bot._custom_research_provider = None
            bot.is_benchmarking = False
            bot.allow_research_fallback = True

            async def mock_provider(q: str) -> str:
                return "primary research"

            with patch.object(bot, "_select_research_provider", return_value=(mock_provider, "asknews")):
                providers = bot._select_research_providers()

        provider_names = [name for _, name in providers]
        assert "financial_data" not in provider_names
