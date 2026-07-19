"""Tests for the financial data research provider (yfinance + FRED)."""

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest


def _make_q(text: str, resolution_criteria: str = "", fine_print: str = "") -> MagicMock:
    """Build a minimal MetaculusQuestion-shaped mock for the ResearchCallable
    contract. resolution_criteria/fine_print default to "" (a bare MagicMock
    would auto-create truthy child mocks, breaking the `or ""` guard and regex)."""
    q = MagicMock()
    q.question_text = text
    q.resolution_criteria = resolution_criteria
    q.fine_print = fine_print
    return q


# Fixed open_time used across the benchmarking date-ceiling tests. as_of derivation
# under is_benchmarking pins to this, so the yfinance start/end and FRED ceiling are
# both deterministic: end == open_time.date() + 1 day, start == open_time.date() - 365d.
_BENCH_OPEN_TIME = datetime(2026, 3, 15, 14, 30, tzinfo=UTC)


def _make_bench_q(text: str, resolution_criteria: str = "", fine_print: str = "") -> MagicMock:
    """A _make_q with a concrete open_time so the benchmarking ceiling can be derived
    (a bare MagicMock open_time isn't a datetime → the provider soft-skips as leakage-safe)."""
    q = _make_q(text, resolution_criteria, fine_print)
    q.open_time = _BENCH_OPEN_TIME
    return q


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------


class TestClassifyFinancialQuestion:
    """Tests for _classify_financial_question."""

    @pytest.mark.asyncio
    async def test_classifies_stock_question_with_tickers(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: AAPL, MSFT\nFRED_SERIES: NONE"

        from metaculus_bot.research.financial_data import _classify_financial_question

        result = await _classify_financial_question("Will Apple stock price exceed $200 by end of 2026?", mock_llm)

        assert result is not None
        assert result["tickers"] == ["AAPL", "MSFT"]
        assert result["fred_series"] == []

    @pytest.mark.asyncio
    async def test_classifies_economic_question_with_fred_series(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: NONE\nFRED_SERIES: UNRATE, CPIAUCSL"

        from metaculus_bot.research.financial_data import _classify_financial_question

        result = await _classify_financial_question("Will US unemployment rate exceed 5% in 2026?", mock_llm)

        assert result is not None
        assert result["tickers"] == []
        assert result["fred_series"] == ["UNRATE", "CPIAUCSL"]

    @pytest.mark.asyncio
    async def test_classifies_mixed_question_with_both(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: ^GSPC\nFRED_SERIES: FEDFUNDS"

        from metaculus_bot.research.financial_data import _classify_financial_question

        result = await _classify_financial_question("Will the S&P 500 drop if the Fed raises rates?", mock_llm)

        assert result is not None
        assert result["tickers"] == ["^GSPC"]
        assert result["fred_series"] == ["FEDFUNDS"]

    @pytest.mark.asyncio
    async def test_non_financial_question_returns_none(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: NO\nTICKERS: NONE\nFRED_SERIES: NONE"

        from metaculus_bot.research.financial_data import _classify_financial_question

        result = await _classify_financial_question("Will it rain in London tomorrow?", mock_llm)

        assert result is None

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM timeout")

        from metaculus_bot.research.financial_data import _classify_financial_question

        result = await _classify_financial_question("Will Apple stock exceed $200?", mock_llm)

        assert result is None

    @pytest.mark.asyncio
    async def test_malformed_llm_response_returns_none(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "I don't understand the question format."

        from metaculus_bot.research.financial_data import _classify_financial_question

        result = await _classify_financial_question("Will Apple stock exceed $200?", mock_llm)

        assert result is None

    @pytest.mark.asyncio
    async def test_financial_yes_but_no_tickers_or_series_returns_none(self) -> None:
        """If classifier says YES but extracts nothing useful, treat as non-financial."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: NONE\nFRED_SERIES: NONE"

        from metaculus_bot.research.financial_data import _classify_financial_question

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

        with patch("metaculus_bot.research.financial_data.yfinance") as mock_yf:
            mock_yf.Ticker.return_value = mock_ticker_instance

            from metaculus_bot.research.financial_data import _fetch_yfinance_data

            result = _fetch_yfinance_data("AAPL")

        assert result != ""
        assert "AAPL" in result
        assert "200.0" in result or "200.00" in result
        # Should contain return calculations
        assert "return" in result.lower() or "change" in result.lower()
        # Should contain volatility
        assert "volatil" in result.lower()

    def test_yfinance_exception_returns_empty_string(self) -> None:
        with patch("metaculus_bot.research.financial_data.yfinance") as mock_yf:
            mock_yf.Ticker.side_effect = Exception("Network error")

            from metaculus_bot.research.financial_data import _fetch_yfinance_data

            result = _fetch_yfinance_data("INVALID")

        assert result == ""

    def test_empty_history_returns_empty_string(self) -> None:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker_instance.info = {}

        with patch("metaculus_bot.research.financial_data.yfinance") as mock_yf:
            mock_yf.Ticker.return_value = mock_ticker_instance

            from metaculus_bot.research.financial_data import _fetch_yfinance_data

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

        with patch("metaculus_bot.research.financial_data.Fred") as mock_fred_class:
            mock_fred_class.return_value = mock_fred_instance

            from metaculus_bot.research.financial_data import _fetch_fred_data

            result = _fetch_fred_data("UNRATE", "fake_api_key")

        assert result != ""
        assert "UNRATE" in result
        # Should contain the latest value
        assert "4.2" in result or "4.20" in result

    def test_fred_exception_returns_empty_string(self) -> None:
        with patch("metaculus_bot.research.financial_data.Fred") as mock_fred_class:
            mock_fred_class.return_value.get_series.side_effect = Exception("API error")

            from metaculus_bot.research.financial_data import _fetch_fred_data

            result = _fetch_fred_data("INVALID", "fake_api_key")

        assert result == ""


class TestRenderFredSeriesYoY:
    """The year-over-year line must use a DATE-based ~365-day lookup, not a fixed
    13-observation offset (F8). On a daily series 13 observations is ~2.5 weeks, so
    the old offset mislabeled a two-and-a-half-week move as year-over-year."""

    @staticmethod
    def _yoy_change_from(markdown: str) -> float:
        """Pull the signed YoY change value out of the rendered markdown line."""
        for line in markdown.splitlines():
            if line.startswith("- Year-over-year change:"):
                # "- Year-over-year change: +12.3 (+4.56%)" -> "+12.3"
                return float(line.split(":", 1)[1].strip().split(" ")[0])
        raise AssertionError(f"no year-over-year line in:\n{markdown}")

    def test_daily_series_uses_365d_ago_value_not_obs_minus_13(self) -> None:
        from metaculus_bot.research.financial_data import _render_fred_series

        # 800 business days ending 2026-03-02. The value is a linear ramp from 0.0
        # to 799.0 (one unit per observation), so the value at any date equals its
        # integer offset from the start — making the two lookups trivially distinct.
        dates = pd.bdate_range(end="2026-03-02", periods=800)
        data = pd.Series(np.arange(800.0), index=dates, name="DGS10")

        latest_value = float(data.iloc[-1])  # 799.0
        # obs[-13] (the OLD, wrong behavior) is ~2.5 weeks back, not a year.
        wrong_offset_value = float(data.iloc[-13])  # 787.0
        # The date-based lookup: last observation at or before ~365 days ago.
        year_ago = data.index[-1] - pd.Timedelta(days=365)
        correct_value = float(data.loc[:year_ago].iloc[-1])

        markdown = _render_fred_series("DGS10", data, "10Y Treasury rate")
        rendered_yoy = self._yoy_change_from(markdown)

        assert rendered_yoy == pytest.approx(latest_value - correct_value, abs=1e-6)
        # And is materially different from the old fixed-offset result.
        assert rendered_yoy != pytest.approx(latest_value - wrong_offset_value, abs=1e-6)

    def test_monthly_series_still_correct(self) -> None:
        from metaculus_bot.research.financial_data import _render_fred_series

        # 60 monthly observations; the value ~12 months back is one year ago.
        dates = pd.date_range(end="2026-03-01", periods=60, freq="MS")
        data = pd.Series(np.arange(60.0), index=dates, name="UNRATE")

        latest_value = float(data.iloc[-1])
        year_ago = data.index[-1] - pd.Timedelta(days=365)
        expected_prior = float(data.loc[:year_ago].iloc[-1])

        markdown = _render_fred_series("UNRATE", data, "unemployment rate")
        rendered_yoy = self._yoy_change_from(markdown)

        assert rendered_yoy == pytest.approx(latest_value - expected_prior, abs=1e-6)

    def test_short_series_omits_yoy_line(self) -> None:
        from metaculus_bot.research.financial_data import _render_fred_series

        # Only ~3 months of monthly data: nothing is ~365 days back, so the YoY
        # line is omitted rather than reaching for a nonexistent observation.
        dates = pd.date_range(end="2026-03-01", periods=3, freq="MS")
        data = pd.Series(np.arange(3.0), index=dates, name="UNRATE")

        markdown = _render_fred_series("UNRATE", data, "unemployment rate")

        assert "Year-over-year change" not in markdown


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
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data.yfinance") as mock_yf,
            patch("metaculus_bot.research.financial_data.Fred") as mock_fred_class,
        ):
            mock_yf.Ticker.return_value = mock_ticker
            mock_fred_class.return_value = mock_fred_instance

            from metaculus_bot.research.financial_data import financial_data_provider

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("FRED_API_KEY", "fake_key")
            try:
                provider = financial_data_provider()
                result = await provider(
                    _make_q("Will Apple stock price exceed $200 and will unemployment stay below 5%?")
                )
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

        with patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm):
            from metaculus_bot.research.financial_data import financial_data_provider

            provider = financial_data_provider()
            result = await provider(_make_q("Will it rain in London tomorrow?"))

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
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data.yfinance") as mock_yf,
        ):
            mock_yf.Ticker.side_effect = ticker_factory

            from metaculus_bot.research.financial_data import financial_data_provider

            provider = financial_data_provider()
            result = await provider(_make_q("Compare Apple and BADTICKER stock performance"))

        assert "AAPL" in result
        # BADTICKER should not appear in the rendered data body (its fetch failed).
        # The Part D routing marker legitimately records the classifier's choice,
        # so check the body before the (forecaster-invisible) HTML-comment marker.
        body = result.split("<!-- financial_routing:")[0]
        assert "BADTICKER" not in body

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
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data.yfinance") as mock_yf,
        ):
            mock_yf.Ticker.return_value = mock_ticker

            from metaculus_bot.research.financial_data import financial_data_provider

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.delenv("FRED_API_KEY", raising=False)
            try:
                provider = financial_data_provider()
                result = await provider(_make_q("Will Apple stock rise and unemployment fall?"))
            finally:
                monkeypatch.undo()

        assert "AAPL" in result
        # FRED data should not appear in the rendered body since there's no API key.
        # The routing marker still records the (skipped) FRED routing decision.
        body = result.split("<!-- financial_routing:")[0]
        assert "UNRATE" not in body


# ---------------------------------------------------------------------------
# Deterministic identifier extraction from resolution criteria / fine print
# ---------------------------------------------------------------------------


class TestExtractFinancialIdentifiers:
    """Tests for extract_financial_identifiers_from_criteria (Part A)."""

    def test_extracts_fred_series_from_url(self) -> None:
        from metaculus_bot.research.financial_data import extract_financial_identifiers_from_criteria

        text = "This resolves based on https://fred.stlouisfed.org/series/DGS10 as published."
        result = extract_financial_identifiers_from_criteria(text)

        assert result["fred_series"] == ["DGS10"]
        assert result["tickers"] == []

    def test_extracts_fred_series_with_underscores_and_digits(self) -> None:
        from metaculus_bot.research.financial_data import extract_financial_identifiers_from_criteria

        text = "High-yield spread: https://fred.stlouisfed.org/series/BAMLH0A0HYM2 and the 10y-2y https://fred.stlouisfed.org/series/T10Y2Y."
        result = extract_financial_identifiers_from_criteria(text)

        assert result["fred_series"] == ["BAMLH0A0HYM2", "T10Y2Y"]

    def test_extracts_yahoo_ticker_with_url_encoded_caret(self) -> None:
        from metaculus_bot.research.financial_data import extract_financial_identifiers_from_criteria

        text = "Resolves on the 10Y yield index at https://finance.yahoo.com/quote/%5ETNX/"
        result = extract_financial_identifiers_from_criteria(text)

        assert result["tickers"] == ["^TNX"]
        assert result["fred_series"] == []

    def test_strips_trailing_period_from_sentence_final_yahoo_url(self) -> None:
        """A URL ending a sentence captures the period into the Yahoo char class
        (`.../quote/%5ETNX.` -> `^TNX.`), which isn't in KNOWN_TICKERS and silently
        defeats the q43650 deterministic-fire guarantee. The trailing `.` must be
        stripped; internal dots (e.g. DX-Y.NYB) are preserved by rstrip."""
        from metaculus_bot.research.financial_data import extract_financial_identifiers_from_criteria

        result = extract_financial_identifiers_from_criteria("Resolves on https://finance.yahoo.com/quote/%5ETNX.")

        assert result["tickers"] == ["^TNX"]

    def test_extracts_yahoo_ticker_with_special_chars(self) -> None:
        from metaculus_bot.research.financial_data import extract_financial_identifiers_from_criteria

        text = "Crude: https://finance.yahoo.com/quote/CL=F bitcoin: https://finance.yahoo.com/quote/BTC-USD"
        result = extract_financial_identifiers_from_criteria(text)

        assert result["tickers"] == ["CL=F", "BTC-USD"]

    def test_extracts_both_fred_and_yahoo(self) -> None:
        from metaculus_bot.research.financial_data import extract_financial_identifiers_from_criteria

        text = (
            "Yield per https://fred.stlouisfed.org/series/DGS10 and proxy "
            "https://finance.yahoo.com/quote/%5ETNX for context."
        )
        result = extract_financial_identifiers_from_criteria(text)

        assert result["fred_series"] == ["DGS10"]
        assert result["tickers"] == ["^TNX"]

    def test_dedupes_preserving_order(self) -> None:
        from metaculus_bot.research.financial_data import extract_financial_identifiers_from_criteria

        text = (
            "https://fred.stlouisfed.org/series/DGS10 ... again "
            "https://fred.stlouisfed.org/series/DGS2 ... once more "
            "https://fred.stlouisfed.org/series/DGS10"
        )
        result = extract_financial_identifiers_from_criteria(text)

        assert result["fred_series"] == ["DGS10", "DGS2"]

    def test_no_match_returns_empty_lists(self) -> None:
        from metaculus_bot.research.financial_data import extract_financial_identifiers_from_criteria

        result = extract_financial_identifiers_from_criteria("Will it rain in London tomorrow?")

        assert result == {"tickers": [], "fred_series": []}

    def test_empty_string_returns_empty_lists(self) -> None:
        from metaculus_bot.research.financial_data import extract_financial_identifiers_from_criteria

        result = extract_financial_identifiers_from_criteria("")

        assert result == {"tickers": [], "fred_series": []}


# ---------------------------------------------------------------------------
# Deterministic routing integration: extraction guarantees + observability
# ---------------------------------------------------------------------------


def _stub_fred_fetch(series_id: str, api_key: str) -> str:
    """Recognizable FRED markdown so tests assert on routing, not the live API."""
    return f"### {series_id} (Test Series)\n- Latest value: 4.48 (2026-06-27)"


def _stub_yfinance_fetch(ticker: str, **kwargs: object) -> str:
    """Recognizable yfinance markdown so tests assert on routing, not the live API.

    Accepts **kwargs so the live-path call (as_of=..., is_benchmarking=...) matches."""
    return f"### {ticker}\n- Current price: 4.48"


class TestDeterministicRouting:
    """Part B/C/D: criteria-driven extraction guarantees the resolving source fires."""

    @pytest.mark.asyncio
    async def test_q43650_regression_dgs10_fires_despite_classifier_misroute(self) -> None:
        """The q43650 smoking gun: criteria name FRED DGS10 but the classifier
        emits only the Yahoo proxy ^TNX. Deterministic extraction must force a
        ### DGS10 FRED section into the output regardless of the classifier."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: ^TNX\nFRED_SERIES: NONE"

        question = _make_q(
            "What will the 10-year Treasury yield be at end of June 2026?",
            resolution_criteria="Resolves to the value at https://fred.stlouisfed.org/series/DGS10 on the close date.",
        )

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data._fetch_fred_data", side_effect=_stub_fred_fetch),
            patch("metaculus_bot.research.financial_data._fetch_yfinance_data", side_effect=_stub_yfinance_fetch),
        ):
            from metaculus_bot.research.financial_data import financial_data_provider

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("FRED_API_KEY", "fake_key")
            try:
                provider = financial_data_provider()
                result = await provider(question)
            finally:
                monkeypatch.undo()

        assert "### DGS10" in result
        # The classifier's Yahoo proxy still fetched too (extraction is additive).
        assert "### ^TNX" in result

    @pytest.mark.asyncio
    async def test_non_financial_classification_but_extracted_url_still_fetches(self) -> None:
        """classification is None but criteria name a FRED URL -> still fetch it.
        Must NOT early-return "" on the extracted path (the q43650-class guarantee)."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: NO\nTICKERS: NONE\nFRED_SERIES: NONE"

        question = _make_q(
            "A vaguely-worded question the classifier won't recognize.",
            fine_print="Resolution source: https://fred.stlouisfed.org/series/CPIAUCSL",
        )

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data._fetch_fred_data", side_effect=_stub_fred_fetch),
        ):
            from metaculus_bot.research.financial_data import financial_data_provider

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("FRED_API_KEY", "fake_key")
            try:
                provider = financial_data_provider()
                result = await provider(question)
            finally:
                monkeypatch.undo()

        assert "### CPIAUCSL" in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("classifier_line", "fake_id"),
        [
            ("FINANCIAL: YES\nTICKERS: NONE\nFRED_SERIES: TOTALSALES_FAKE", "TOTALSALES_FAKE"),
            # ZZZZ has only valid ticker chars (so F2 keeps it) but isn't in
            # KNOWN_TICKERS, exercising the unknown-but-well-formed ticker branch.
            ("FINANCIAL: YES\nTICKERS: ZZZZ\nFRED_SERIES: NONE", "ZZZZ"),
        ],
    )
    async def test_unknown_classifier_id_logs_warning_but_still_fetches(
        self, caplog: pytest.LogCaptureFixture, classifier_line: str, fake_id: str
    ) -> None:
        """An unrecognized classifier ID (FRED series OR ticker) is soft-failed
        loudly: WARNING logged, fetch still happens (may be valid-but-unlisted), and
        the id appears in the routing marker's unknown=[...] field."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = classifier_line

        question = _make_q("Will some obscure indicator move?")

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data._fetch_fred_data", side_effect=_stub_fred_fetch),
            patch("metaculus_bot.research.financial_data._fetch_yfinance_data", side_effect=_stub_yfinance_fetch),
        ):
            from metaculus_bot.research.financial_data import financial_data_provider

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("FRED_API_KEY", "fake_key")
            try:
                provider = financial_data_provider()
                with caplog.at_level("WARNING", logger="metaculus_bot.research.financial_data"):
                    result = await provider(question)
            finally:
                monkeypatch.undo()

        assert f"### {fake_id}" in result
        assert f"unknown=[{fake_id}]" in result
        assert any(fake_id in rec.message and rec.levelname == "WARNING" for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_malformed_classifier_id_dropped_and_marker_not_corrupted(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A classifier token containing chars outside the extraction char class
        (here `-->`, which would close the routing HTML comment and leak its tail as
        visible markdown) is dropped with a WARNING. A clean co-emitted id survives,
        and the rendered marker contains exactly ONE `-->` (its own terminator).

        Note: _parse_classifier_response upper-cases values, so `BAD --> leak`
        reaches the sanitizer (and the WARNING) as `BAD --> LEAK`."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: AAPL, BAD --> leak\nFRED_SERIES: NONE"

        question = _make_q("Will Apple stock move?")

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data._fetch_yfinance_data", side_effect=_stub_yfinance_fetch),
        ):
            from metaculus_bot.research.financial_data import financial_data_provider

            with caplog.at_level("WARNING", logger="metaculus_bot.research.financial_data"):
                provider = financial_data_provider()
                result = await provider(question)

        # The clean token still fetched; the malformed one is gone everywhere.
        assert "### AAPL" in result
        assert "LEAK" not in result
        # Exactly one `-->`: the marker's own terminator, nothing leaked early.
        assert result.count("-->") == 1
        assert any("BAD --> LEAK" in rec.message and rec.levelname == "WARNING" for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_routing_marker_present_with_expected_values(self) -> None:
        """Part D: a compact, forecaster-invisible HTML-comment routing marker is
        appended to the returned markdown, recording the routing decision."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: ^TNX\nFRED_SERIES: NONE"

        question = _make_q(
            "10y Treasury yield question",
            resolution_criteria="Resolves on https://fred.stlouisfed.org/series/DGS10.",
        )

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data._fetch_fred_data", side_effect=_stub_fred_fetch),
            patch("metaculus_bot.research.financial_data._fetch_yfinance_data", side_effect=_stub_yfinance_fetch),
        ):
            from metaculus_bot.research.financial_data import financial_data_provider

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("FRED_API_KEY", "fake_key")
            try:
                provider = financial_data_provider()
                result = await provider(question)
            finally:
                monkeypatch.undo()

        assert "<!-- financial_routing:" in result
        assert "fred=[DGS10]" in result
        assert "tickers=[^TNX]" in result
        assert "extracted_fred=[DGS10]" in result
        assert "extracted_tickers=[]" in result
        assert "unknown=[]" in result

    @pytest.mark.asyncio
    async def test_routing_marker_absent_on_non_financial_empty_return(self) -> None:
        """The marker is only emitted when the provider actually ran; a truly
        non-financial question (no classification, no extraction) returns ""."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: NO\nTICKERS: NONE\nFRED_SERIES: NONE"

        with patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm):
            from metaculus_bot.research.financial_data import financial_data_provider

            provider = financial_data_provider()
            result = await provider(_make_q("Will it rain in London tomorrow?"))

        assert result == ""
        assert "financial_routing" not in result


# ---------------------------------------------------------------------------
# Allowlist / prompt single-source-of-truth consistency
# ---------------------------------------------------------------------------


class TestAllowlistPromptConsistency:
    """The KNOWN_* frozensets and the CLASSIFIER_PROMPT reference table are both
    derived from the _TICKER_GROUPS / _FRED_GROUPS dicts, so they cannot drift.
    These tests guard that derivation (and would catch a future hardcode regression)."""

    def test_every_known_id_appears_in_prompt(self) -> None:
        from metaculus_bot.research.financial_data import CLASSIFIER_PROMPT, KNOWN_FRED_SERIES, KNOWN_TICKERS

        for identifier in KNOWN_TICKERS | KNOWN_FRED_SERIES:
            assert identifier in CLASSIFIER_PROMPT, f"{identifier} missing from CLASSIFIER_PROMPT reference table"

    def test_frozensets_derived_from_label_dicts(self) -> None:
        from metaculus_bot.research.financial_data import (
            FRED_LABELS,
            KNOWN_FRED_SERIES,
            KNOWN_TICKERS,
            TICKER_LABELS,
        )

        assert KNOWN_TICKERS == frozenset(TICKER_LABELS)
        assert KNOWN_FRED_SERIES == frozenset(FRED_LABELS)


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

        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)

        with patch(
            "metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback",
            return_value=AsyncMock(),
        ):
            orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
            mock_provider = AsyncMock(return_value="primary research")

            with patch.object(orch, "_select_research_provider", return_value=(mock_provider, "asknews")):
                providers = orch._select_research_providers()

        provider_names = [name for _, name in providers]
        assert "financial_data" in provider_names

    def test_provider_excluded_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FINANCIAL_DATA_ENABLED", "false")
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        mock_provider = AsyncMock(return_value="primary research")

        with patch.object(orch, "_select_research_provider", return_value=(mock_provider, "asknews")):
            providers = orch._select_research_providers()

        provider_names = [name for _, name in providers]
        assert "financial_data" not in provider_names

    def test_provider_excluded_when_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)
        mock_provider = AsyncMock(return_value="primary research")

        with patch.object(orch, "_select_research_provider", return_value=(mock_provider, "asknews")):
            providers = orch._select_research_providers()

        provider_names = [name for _, name in providers]
        assert "financial_data" not in provider_names


# ---------------------------------------------------------------------------
# Backtest leakage guard: date-ceiling under is_benchmarking
# ---------------------------------------------------------------------------


class TestBenchmarkingDateCeiling:
    """Under is_benchmarking the provider must ceiling every fetch to open_time and
    never touch live-only surfaces (yfinance .info, fredapi). Live path unchanged."""

    def test_yfinance_benchmarking_uses_start_end_ceiling_and_skips_info(self) -> None:
        """Benchmarking yfinance fetch: history is called with start/end (NOT period),
        end == open_time.date() + 1 day, start == open_time.date() - 365d, and `.info`
        is never accessed (a PropertyMock that would raise if touched)."""
        dates = pd.date_range(end="2026-03-15", periods=252, freq="B")
        close_prices = np.linspace(150.0, 200.0, 252)
        mock_history = pd.DataFrame({"Close": close_prices}, index=dates)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_history
        # A PropertyMock that raises if `.info` is read at all — the benchmarking path
        # must never touch it (it has no historical mode → leaks today's values).
        info_prop = PropertyMock(side_effect=AssertionError(".info must not be read under benchmarking"))
        type(mock_ticker).info = info_prop

        with patch("metaculus_bot.research.financial_data.yfinance") as mock_yf:
            mock_yf.Ticker.return_value = mock_ticker

            from metaculus_bot.research.financial_data import _fetch_yfinance_data

            result = _fetch_yfinance_data("AAPL", as_of=_BENCH_OPEN_TIME, is_benchmarking=True)

        info_prop.assert_not_called()
        mock_ticker.history.assert_called_once()
        _, kwargs = mock_ticker.history.call_args
        assert "period" not in kwargs
        assert kwargs["start"] == "2025-03-15"  # open_time.date() - 365d
        assert kwargs["end"] == "2026-03-16"  # open_time.date() + 1d (end EXCLUSIVE)
        assert "AAPL" in result
        assert "[omitted under backtest" in result

    def test_yfinance_live_path_unchanged_period_based(self) -> None:
        """Live path (default is_benchmarking=False): period-based call preserved, no
        start/end, and `.info` still consulted for fundamentals."""
        dates = pd.date_range(end="2026-03-30", periods=252, freq="B")
        close_prices = np.linspace(150.0, 200.0, 252)
        mock_history = pd.DataFrame({"Close": close_prices}, index=dates)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_history
        mock_ticker.info = {"shortName": "Apple Inc.", "trailingPE": 28.5}

        with patch("metaculus_bot.research.financial_data.yfinance") as mock_yf:
            mock_yf.Ticker.return_value = mock_ticker

            from metaculus_bot.research.financial_data import _fetch_yfinance_data

            result = _fetch_yfinance_data("AAPL")

        mock_ticker.history.assert_called_once()
        _, kwargs = mock_ticker.history.call_args
        assert kwargs == {"period": "365d"}
        assert "P/E ratio" in result  # .info fundamentals rendered

    def test_fred_benchmarking_routes_through_ts_fetch_with_ceiling(self) -> None:
        """Benchmarking FRED fetch routes through ts_fetch.fetch_series (keyless), with
        the ceiling == open_time.date(), and a REVISING series (CPIAUCSL, not in the
        non-revising allowlist) gets a vintage spec (revises=True)."""
        fred_dates = pd.date_range(end="2026-03-01", periods=60, freq="MS")
        fred_values = np.linspace(280.0, 310.0, 60)
        mock_series = pd.Series(fred_values, index=fred_dates, name="CPIAUCSL")

        captured: dict = {}

        def fake_fetch_series(spec, ceiling, **kwargs):
            captured["spec"] = spec
            captured["ceiling"] = ceiling
            return mock_series

        with patch("metaculus_bot.research.financial_data.fetch_series", side_effect=fake_fetch_series):
            from metaculus_bot.research.financial_data import _fetch_fred_data_ceiling

            result = _fetch_fred_data_ceiling("CPIAUCSL", _BENCH_OPEN_TIME)

        assert captured["ceiling"] == date(2026, 3, 15)  # open_time.date()
        assert captured["spec"].source == "fred"
        assert captured["spec"].series_id == "CPIAUCSL"
        assert captured["spec"].revises is True  # revising macro series → ALFRED vintage
        assert "### CPIAUCSL" in result
        assert "Latest value" in result

    def test_fred_benchmarking_non_revising_series_no_vintage(self) -> None:
        """A non-revising allowlisted series (DGS10) must NOT be routed to a vintage
        fetch (revises=False) — plain fredgraph is leakage-safe for it."""
        dgs_dates = pd.date_range(end="2026-03-15", periods=300, freq="B")
        dgs_values = np.linspace(3.8, 4.4, 300)
        mock_series = pd.Series(dgs_values, index=dgs_dates, name="DGS10")

        captured: dict = {}

        def fake_fetch_series(spec, ceiling, **kwargs):
            captured["spec"] = spec
            return mock_series

        with patch("metaculus_bot.research.financial_data.fetch_series", side_effect=fake_fetch_series):
            from metaculus_bot.research.financial_data import _fetch_fred_data_ceiling

            result = _fetch_fred_data_ceiling("DGS10", _BENCH_OPEN_TIME)

        assert captured["spec"].revises is False
        assert "### DGS10" in result

    def test_fred_ceiling_fetch_soft_fails_on_fetch_error(self) -> None:
        """A ts_fetch FetchError soft-fails to "" (never propagates)."""
        from metaculus_bot.research.ts_fetch import FetchError

        with patch("metaculus_bot.research.financial_data.fetch_series", side_effect=FetchError("bad id")):
            from metaculus_bot.research.financial_data import _fetch_fred_data_ceiling

            result = _fetch_fred_data_ceiling("BOGUS", _BENCH_OPEN_TIME)

        assert result == ""

    @pytest.mark.asyncio
    async def test_provider_benchmarking_no_fred_key_still_fetches_fred(self) -> None:
        """End-to-end benchmarking flow: FRED_API_KEY is UNSET yet the FRED series is
        still fetched (keyless ts_fetch path), and yfinance never touches `.info`."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: AAPL\nFRED_SERIES: UNRATE"

        dates = pd.date_range(end="2026-03-15", periods=252, freq="B")
        mock_history = pd.DataFrame({"Close": np.linspace(150.0, 200.0, 252)}, index=dates)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_history
        info_prop = PropertyMock(side_effect=AssertionError(".info must not be read under benchmarking"))
        type(mock_ticker).info = info_prop

        unrate_dates = pd.date_range(end="2026-03-01", periods=60, freq="MS")
        mock_unrate = pd.Series(np.linspace(3.5, 4.2, 60), index=unrate_dates, name="UNRATE")

        def fake_fetch_series(spec, ceiling, **kwargs):
            assert ceiling == date(2026, 3, 15)
            return mock_unrate

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data.yfinance") as mock_yf,
            patch("metaculus_bot.research.financial_data.fetch_series", side_effect=fake_fetch_series) as mock_fetch,
        ):
            mock_yf.Ticker.return_value = mock_ticker

            from metaculus_bot.research.financial_data import financial_data_provider

            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.delenv("FRED_API_KEY", raising=False)
            try:
                provider = financial_data_provider(is_benchmarking=True)
                result = await provider(_make_bench_q("Will AAPL rise and unemployment stay below 5%?"))
            finally:
                monkeypatch.undo()

        info_prop.assert_not_called()
        mock_fetch.assert_called_once()  # keyless FRED fetch fired despite no API key
        assert "### AAPL" in result
        assert "### UNRATE" in result

    @pytest.mark.asyncio
    async def test_provider_benchmarking_missing_open_time_soft_skips(self) -> None:
        """No open_time under benchmarking → can't ceiling → soft-skip to "" (never
        fetch, since fetching would risk today's data leaking into a resolved question)."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: AAPL\nFRED_SERIES: NONE"

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data.yfinance") as mock_yf,
        ):
            from metaculus_bot.research.financial_data import financial_data_provider

            provider = financial_data_provider(is_benchmarking=True)
            # _make_q leaves open_time as an auto-child MagicMock (not a datetime).
            result = await provider(_make_q("Will Apple stock exceed $200?"))

        assert result == ""
        mock_yf.Ticker.assert_not_called()

    @pytest.mark.asyncio
    async def test_provider_default_is_live(self) -> None:
        """Default (no is_benchmarking arg) → live path: period-based yfinance, `.info`
        consulted, and no open_time needed."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: AAPL\nFRED_SERIES: NONE"

        dates = pd.date_range(end="2026-03-30", periods=252, freq="B")
        mock_history = pd.DataFrame({"Close": np.linspace(150.0, 200.0, 252)}, index=dates)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_history
        mock_ticker.info = {"shortName": "Apple Inc.", "trailingPE": 28.5}

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data.yfinance") as mock_yf,
        ):
            mock_yf.Ticker.return_value = mock_ticker

            from metaculus_bot.research.financial_data import financial_data_provider

            provider = financial_data_provider()
            result = await provider(_make_q("Will Apple stock exceed $200?"))

        _, kwargs = mock_ticker.history.call_args
        assert kwargs == {"period": "365d"}
        assert "### AAPL" in result
