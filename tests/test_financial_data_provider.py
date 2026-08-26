"""Tests for the financial data research provider (yfinance + FRED)."""

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
from forecasting_tools import GeneralLlm

from metaculus_bot.constants import FINANCIAL_YFINANCE_LOOKBACK_DAYS, MAX_FINANCIAL_IDENTIFIERS
from metaculus_bot.research.financial_data import (
    _PERIOD_ROW_OFFSETS,
    CLASSIFIER_PROMPT,
    FRED_LABELS,
    KNOWN_FRED_SERIES,
    KNOWN_TICKERS,
    TICKER_LABELS,
    _cap_identifiers,
    _classify_financial_question,
    _fetch_fred_data,
    _fetch_fred_data_ceiling,
    _fetch_yfinance_data,
    _render_fred_series,
    extract_financial_identifiers_from_criteria,
    financial_data_provider,
)
from metaculus_bot.research.orchestrator import ResearchOrchestrator
from metaculus_bot.research.provider_diagnostics import _is_lost_source, pop_provider_detail
from metaculus_bot.research.ts_estimators import (
    CALENDAR_DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    observed_periods_per_year,
)
from metaculus_bot.research.ts_fetch import FetchError


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

        result, _error = await _classify_financial_question(
            "Will Apple stock price exceed $200 by end of 2026?", mock_llm
        )

        assert result is not None
        assert result["tickers"] == ["AAPL", "MSFT"]
        assert result["fred_series"] == []

    @pytest.mark.asyncio
    async def test_classifies_economic_question_with_fred_series(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: NONE\nFRED_SERIES: UNRATE, CPIAUCSL"

        result, _error = await _classify_financial_question("Will US unemployment rate exceed 5% in 2026?", mock_llm)

        assert result is not None
        assert result["tickers"] == []
        assert result["fred_series"] == ["UNRATE", "CPIAUCSL"]

    @pytest.mark.asyncio
    async def test_classifies_mixed_question_with_both(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: ^GSPC\nFRED_SERIES: FEDFUNDS"

        result, _error = await _classify_financial_question("Will the S&P 500 drop if the Fed raises rates?", mock_llm)

        assert result is not None
        assert result["tickers"] == ["^GSPC"]
        assert result["fred_series"] == ["FEDFUNDS"]

    @pytest.mark.asyncio
    async def test_non_financial_question_returns_none(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: NO\nTICKERS: NONE\nFRED_SERIES: NONE"

        result, _error = await _classify_financial_question("Will it rain in London tomorrow?", mock_llm)

        assert result is None

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none_and_names_the_error(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM timeout")

        result, error = await _classify_financial_question("Will Apple stock exceed $200?", mock_llm)

        assert result is None
        # A DEAD classifier and a non-financial question both return None. Without this
        # second value the caller cannot tell them apart, so a model retirement (the
        # 2026-05-15 grok 404 precedent), a schema change, or a quota reads as "no
        # financial angle" forever.
        assert error == "RuntimeError"

    @pytest.mark.asyncio
    async def test_non_financial_reports_no_error(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: NO\nTICKERS: NONE\nFRED_SERIES: NONE"

        result, error = await _classify_financial_question("Will it rain in London tomorrow?", mock_llm)

        assert result is None
        assert error is None, "a working classifier reading 'not financial' is not a failure"

    @pytest.mark.asyncio
    async def test_malformed_llm_response_returns_none(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "I don't understand the question format."

        result, _error = await _classify_financial_question("Will Apple stock exceed $200?", mock_llm)

        assert result is None

    @pytest.mark.asyncio
    async def test_financial_yes_but_no_tickers_or_series_returns_none(self) -> None:
        """If classifier says YES but extracts nothing useful, treat as non-financial."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: NONE\nFRED_SERIES: NONE"

        result, _error = await _classify_financial_question("Will the economy improve?", mock_llm)

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

            result = _fetch_yfinance_data("INVALID")

        assert result == ""

    def test_empty_history_returns_empty_string(self) -> None:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker_instance.info = {}

        with patch("metaculus_bot.research.financial_data.yfinance") as mock_yf:
            mock_yf.Ticker.return_value = mock_ticker_instance

            result = _fetch_yfinance_data("FAKE")

        assert result == ""


# ---------------------------------------------------------------------------
# FRED fetch tests
# ---------------------------------------------------------------------------


class TestAnnualizationBasis:
    """The vol/window basis must come from the OBSERVED series density, not a fixed 252.

    24/7-traded assets (crypto) print ~365 daily bars a year, so sqrt(252)
    understates their annualized vol by ~17% (sqrt(252/365) ~= 0.83) and the
    252-row "1y"/"52-week" windows span only ~8.2 calendar months — the q44882
    (ETH-USD) defect. Exchange-traded series (5 rows/week) must stay byte-identical
    to the historical 252 behavior, and an unmeasurable density must degrade to
    252, never crash.
    """

    @staticmethod
    def _fetch_with_history(close: pd.Series) -> str:
        history = pd.DataFrame(
            {"Close": close, "Open": close * 0.99, "High": close * 1.01, "Low": close * 0.98},
            index=close.index,
        )
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = history
        mock_ticker_instance.info = {"shortName": "Test Asset"}
        with patch("metaculus_bot.research.financial_data.yfinance") as mock_yf:
            mock_yf.Ticker.return_value = mock_ticker_instance
            return _fetch_yfinance_data("TEST")

    @staticmethod
    def _line_value(markdown: str, prefix: str) -> str:
        for line in markdown.splitlines():
            if line.strip().startswith(prefix):
                return line.split(":", 1)[1].strip()
        raise AssertionError(f"no {prefix!r} line in:\n{markdown}")

    def test_business_day_series_keeps_trading_day_basis(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range(end="2026-03-30", periods=300)
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=dates)
        assert observed_periods_per_year(close.index) == 252

        result = self._fetch_with_history(close)
        expected_vol = close.pct_change().dropna().iloc[-30:].std() * np.sqrt(252) * 100
        assert self._line_value(result, "- 30-trading-day annualized volatility") == f"{expected_vol:.1f}%"
        expected_1y = (close.iloc[-1] / close.iloc[-253] - 1) * 100
        assert self._line_value(result, "- 1y") == f"{expected_1y:+.2f}%"
        year_slice = close.iloc[-252:]
        assert self._line_value(result, "- 52-week range") == f"{year_slice.min():.2f} - {year_slice.max():.2f}"

    def test_24_7_series_gets_calendar_basis_and_full_year_windows(self):
        rng = np.random.default_rng(11)
        dates = pd.date_range(end="2026-07-31", periods=366, freq="D")
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, 366))), index=dates)
        # Plant a spike only the 365-row window can see: 300 rows back is beyond
        # the old 252-row slice but inside the true 52 calendar weeks.
        close.iloc[-300] = close.max() * 3
        assert observed_periods_per_year(close.index) == 365

        result = self._fetch_with_history(close)
        expected_vol = close.pct_change().dropna().iloc[-30:].std() * np.sqrt(365) * 100
        assert self._line_value(result, "- 30-calendar-day annualized volatility") == f"{expected_vol:.1f}%"
        # A 252-basis vol would be ~17% lower; make sure that's not what rendered.
        wrong_vol = close.pct_change().dropna().iloc[-30:].std() * np.sqrt(252) * 100
        assert f"{wrong_vol:.1f}%" != f"{expected_vol:.1f}%"
        # 1y = 365 calendar rows back, 1w = 7 rows back.
        expected_1y = (close.iloc[-1] / close.iloc[-366] - 1) * 100
        assert self._line_value(result, "- 1y") == f"{expected_1y:+.2f}%"
        expected_1w = (close.iloc[-1] / close.iloc[-8] - 1) * 100
        assert self._line_value(result, "- 1w") == f"{expected_1w:+.2f}%"
        # The 52-week high must include the spike the 252-row slice misses.
        assert self._line_value(result, "- 52-week range").endswith(f"{close.iloc[-300]:.2f}")

    def test_the_production_fetch_window_still_renders_the_1y_row(self):
        """The boundary the lookback constant exists to clear: the 1y offset needs STRICTLY MORE
        rows than 365 (`close.iloc[-(days + 1)]`), and a 24/7 series prints ~one bar per calendar
        day of the fetch window — so at a 365-day window the 1y row silently vanished from
        exactly the crypto snapshots the calendar-basis fix targets. Built at the production
        frame size so a future trim of the constant back to 365 fails here, not in prod."""
        assert FINANCIAL_YFINANCE_LOOKBACK_DAYS > 365, "the 1y offset needs at least 366 bars"
        rng = np.random.default_rng(7)
        n_rows = FINANCIAL_YFINANCE_LOOKBACK_DAYS
        dates = pd.date_range(end="2026-07-31", periods=n_rows, freq="D")
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_rows))), index=dates)
        assert observed_periods_per_year(close.index) == 365

        result = self._fetch_with_history(close)

        expected_1y = (close.iloc[-1] / close.iloc[-366] - 1) * 100
        assert self._line_value(result, "- 1y") == f"{expected_1y:+.2f}%"

    def test_the_two_offset_tables_are_the_documented_conventions(self):
        # The tables ARE the behavior: every period-return label on every financial snapshot is
        # only a true calendar period because of these numbers. Pinned verbatim so an edit is a
        # deliberate act — the 252 row is the trading-day convention (5/wk), the 365 row is plain
        # calendar days, and swapping one number silently mislabels one snapshot family.
        assert _PERIOD_ROW_OFFSETS[TRADING_DAYS_PER_YEAR] == [
            ("1d", 1),
            ("1w", 5),
            ("1m", 21),
            ("3m", 63),
            ("6m", 126),
            ("1y", 252),
        ]
        assert _PERIOD_ROW_OFFSETS[CALENDAR_DAYS_PER_YEAR] == [
            ("1d", 1),
            ("1w", 7),
            ("1m", 30),
            ("3m", 91),
            ("6m", 182),
            ("1y", 365),
        ]
        assert set(_PERIOD_ROW_OFFSETS) == {TRADING_DAYS_PER_YEAR, CALENDAR_DAYS_PER_YEAR}

    @pytest.mark.parametrize("basis", [TRADING_DAYS_PER_YEAR, CALENDAR_DAYS_PER_YEAR])
    def test_every_period_row_reads_its_bases_own_offset(self, basis: int):
        # The two bases are pinned end-to-end above only on 1w/1y; walk ALL six labels so a
        # mis-keyed intermediate offset (3m reading 63 rows on a 24/7 series = 9 calendar weeks
        # under a "3m" label) can't hide between the two tested ones. Table-driven, so a new
        # period label is covered the moment it lands.
        rng = np.random.default_rng(3)
        # Enough rows for the longest offset plus the strictly-more-rows requirement, on an index
        # whose observed density lands the series on the basis under test.
        index = (
            pd.bdate_range(end="2026-07-31", periods=400)
            if basis == TRADING_DAYS_PER_YEAR
            else pd.date_range(end="2026-07-31", periods=400, freq="D")
        )
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index)))), index=index)
        assert observed_periods_per_year(close.index) == basis

        result = self._fetch_with_history(close)

        for label, offset in _PERIOD_ROW_OFFSETS[basis]:
            expected = (close.iloc[-1] / close.iloc[-(offset + 1)] - 1) * 100
            assert self._line_value(result, f"- {label}") == f"{expected:+.2f}%", (basis, label)

    def test_short_series_degrades_to_trading_day_basis(self):
        dates = pd.date_range(end="2026-07-31", periods=10, freq="D")
        close = pd.Series(np.linspace(100.0, 110.0, 10), index=dates)
        assert observed_periods_per_year(close.index) == 252
        # And the full fetch still renders (no vol line under 30 rows, no crash).
        result = self._fetch_with_history(close)
        assert "- Current price:" in result
        assert "volatility" not in result

    def test_non_datetime_index_degrades_to_trading_day_basis(self):
        close = pd.Series(np.linspace(100.0, 110.0, 50))  # RangeIndex
        assert observed_periods_per_year(close.index) == 252

    def test_zero_span_index_degrades_to_trading_day_basis(self):
        dates = pd.DatetimeIndex(["2026-07-31"] * 50)
        close = pd.Series(np.linspace(100.0, 110.0, 50), index=dates)
        assert observed_periods_per_year(close.index) == 252


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

            result = _fetch_fred_data("UNRATE", "fake_api_key")

        assert result != ""
        assert "UNRATE" in result
        # Should contain the latest value
        assert "4.2" in result or "4.20" in result

    def test_fred_exception_returns_empty_string(self) -> None:
        with patch("metaculus_bot.research.financial_data.Fred") as mock_fred_class:
            mock_fred_class.return_value.get_series.side_effect = Exception("API error")

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
        text = "This resolves based on https://fred.stlouisfed.org/series/DGS10 as published."
        result = extract_financial_identifiers_from_criteria(text)

        assert result["fred_series"] == ["DGS10"]
        assert result["tickers"] == []

    def test_extracts_fred_series_with_underscores_and_digits(self) -> None:
        text = "High-yield spread: https://fred.stlouisfed.org/series/BAMLH0A0HYM2 and the 10y-2y https://fred.stlouisfed.org/series/T10Y2Y."
        result = extract_financial_identifiers_from_criteria(text)

        assert result["fred_series"] == ["BAMLH0A0HYM2", "T10Y2Y"]

    def test_extracts_yahoo_ticker_with_url_encoded_caret(self) -> None:
        text = "Resolves on the 10Y yield index at https://finance.yahoo.com/quote/%5ETNX/"
        result = extract_financial_identifiers_from_criteria(text)

        assert result["tickers"] == ["^TNX"]
        assert result["fred_series"] == []

    def test_strips_trailing_period_from_sentence_final_yahoo_url(self) -> None:
        """A URL ending a sentence captures the period into the Yahoo char class
        (`.../quote/%5ETNX.` -> `^TNX.`), which isn't in KNOWN_TICKERS and silently
        defeats the q43650 deterministic-fire guarantee. The trailing `.` must be
        stripped; internal dots (e.g. DX-Y.NYB) are preserved by rstrip."""
        result = extract_financial_identifiers_from_criteria("Resolves on https://finance.yahoo.com/quote/%5ETNX.")

        assert result["tickers"] == ["^TNX"]

    def test_extracts_yahoo_ticker_with_special_chars(self) -> None:
        text = "Crude: https://finance.yahoo.com/quote/CL=F bitcoin: https://finance.yahoo.com/quote/BTC-USD"
        result = extract_financial_identifiers_from_criteria(text)

        assert result["tickers"] == ["CL=F", "BTC-USD"]

    def test_extracts_both_fred_and_yahoo(self) -> None:
        text = (
            "Yield per https://fred.stlouisfed.org/series/DGS10 and proxy "
            "https://finance.yahoo.com/quote/%5ETNX for context."
        )
        result = extract_financial_identifiers_from_criteria(text)

        assert result["fred_series"] == ["DGS10"]
        assert result["tickers"] == ["^TNX"]

    def test_dedupes_preserving_order(self) -> None:
        text = (
            "https://fred.stlouisfed.org/series/DGS10 ... again "
            "https://fred.stlouisfed.org/series/DGS2 ... once more "
            "https://fred.stlouisfed.org/series/DGS10"
        )
        result = extract_financial_identifiers_from_criteria(text)

        assert result["fred_series"] == ["DGS10", "DGS2"]

    def test_no_match_returns_empty_lists(self) -> None:
        result = extract_financial_identifiers_from_criteria("Will it rain in London tomorrow?")

        assert result == {"tickers": [], "fred_series": []}

    def test_empty_string_returns_empty_lists(self) -> None:
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
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("FRED_API_KEY", "fake_key")
            try:
                provider = financial_data_provider()
                result = await provider(question)
            finally:
                monkeypatch.undo()

        assert "### CPIAUCSL" in result

    @pytest.mark.asyncio
    async def test_records_per_identifier_detail_for_diagnostics(self) -> None:
        """A ticker that returns no data is recorded as a lost source, so a partial
        financial_data fetch (one ticker ok, one empty) is visible in diagnostics
        even though the provider still returns usable output (status `ok`)."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: AAPL, FAKE\nFRED_SERIES: NONE"

        def _yf(ticker: str, **kwargs: object) -> str:
            return "### AAPL\n- Current price: 190" if ticker == "AAPL" else ""

        question = _make_q("Will Apple stock exceed $200 by end of 2026?")
        question.id_of_question = 7777

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data._fetch_yfinance_data", side_effect=_yf),
        ):
            result = await financial_data_provider()(question)

        assert "### AAPL" in result  # the good ticker contributed
        sources = pop_provider_detail(question.id_of_question, "financial_data")["sources"]
        assert sources["AAPL"] == "ok"  # contributed data
        assert sources["FAKE"] == "empty"  # requested but returned nothing — a lost source

    @pytest.mark.asyncio
    async def test_records_error_detail_when_a_fetch_task_raises(self) -> None:
        """A raising fetch (yfinance HTTP failure) is gathered as an exception, not an
        empty string, so it needs its own `error` token: without it the identifier is
        absent from the source map entirely and the diagnostics line reads as fully
        healthy while one requested series never arrived."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: AAPL, MSFT\nFRED_SERIES: NONE"

        def _yf(ticker: str, **kwargs: object) -> str:
            if ticker == "AAPL":
                return "### AAPL\n- Current price: 190"
            raise RuntimeError("yfinance upstream 503")

        question = _make_q("Will Apple stock exceed $200 by end of 2026?")
        question.id_of_question = 7778

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data._fetch_yfinance_data", side_effect=_yf),
        ):
            result = await financial_data_provider()(question)

        assert "### AAPL" in result  # the healthy ticker still contributes
        sources = pop_provider_detail(question.id_of_question, "financial_data")["sources"]
        assert sources["AAPL"] == "ok"
        assert sources["MSFT"] == "error"  # the raised fetch is a LOST source, not a missing key

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
        for identifier in KNOWN_TICKERS | KNOWN_FRED_SERIES:
            assert identifier in CLASSIFIER_PROMPT, f"{identifier} missing from CLASSIFIER_PROMPT reference table"

    def test_frozensets_derived_from_label_dicts(self) -> None:
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

            result = _fetch_yfinance_data("AAPL", as_of=_BENCH_OPEN_TIME, is_benchmarking=True)

        info_prop.assert_not_called()
        mock_ticker.history.assert_called_once()
        _, kwargs = mock_ticker.history.call_args
        assert "period" not in kwargs
        assert kwargs["start"] == "2025-03-08"  # open_time.date() - FINANCIAL_YFINANCE_LOOKBACK_DAYS (372d)
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

            result = _fetch_yfinance_data("AAPL")

        mock_ticker.history.assert_called_once()
        _, kwargs = mock_ticker.history.call_args
        assert kwargs == {"period": f"{FINANCIAL_YFINANCE_LOOKBACK_DAYS}d"}
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
            result = _fetch_fred_data_ceiling("DGS10", _BENCH_OPEN_TIME)

        assert captured["spec"].revises is False
        assert "### DGS10" in result

    def test_fred_ceiling_fetch_soft_fails_on_fetch_error(self) -> None:
        """A ts_fetch FetchError soft-fails to "" (never propagates)."""
        with patch("metaculus_bot.research.financial_data.fetch_series", side_effect=FetchError("bad id")):
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
            provider = financial_data_provider(is_benchmarking=True)
            # _make_q leaves open_time as an auto-child MagicMock (not a datetime).
            result = await provider(_make_q("Will Apple stock exceed $200?"))

        assert result == ""
        mock_yf.Ticker.assert_not_called()

    def test_identifier_cap_drops_classifier_extras_and_keeps_extracted(self) -> None:
        """Bound the per-question fan-out without ever dropping the resolving source.

        Each identifier gets its own asyncio.to_thread, and the list was unbounded — it is
        whatever an LLM named plus whatever URL extraction found. Those threads land in the
        process-wide default executor shared with ts_fetch, resolution_source, the agentic
        fetch ladder, and the /auth/key probe, and a task queued behind a full pool burns
        its wait_for budget without running, so an over-eager classification on one
        question degrades providers on others.

        The cap is asymmetric on purpose: URL-EXTRACTED ids are the load-bearing guarantee
        that the source a question actually resolves on gets fetched even when the
        classifier misroutes, so only classifier-only extras are trimmed.
        """
        extracted = {"tickers": ["AAPL"], "fred_series": ["UNRATE"]}
        # Far more classifier ids than the cap, with the two extracted ones in the middle.
        tickers = [f"TK{i}" for i in range(10)] + ["AAPL"] + [f"ZZ{i}" for i in range(10)]
        fred_series = [f"FS{i}" for i in range(10)] + ["UNRATE"]

        kept_tickers, kept_fred = _cap_identifiers(tickers, fred_series, extracted)

        assert len(kept_tickers) + len(kept_fred) == MAX_FINANCIAL_IDENTIFIERS
        assert "AAPL" in kept_tickers, "an extracted ticker must never be dropped"
        assert "UNRATE" in kept_fred, "an extracted FRED series must never be dropped"
        # Relative order preserved so rendered sections stay stable.
        assert kept_tickers == [t for t in tickers if t in kept_tickers]
        assert kept_fred == [f for f in fred_series if f in kept_fred]

    def test_identifier_cap_is_a_no_op_under_the_limit(self) -> None:
        extracted = {"tickers": ["AAPL"], "fred_series": []}
        kept_tickers, kept_fred = _cap_identifiers(["AAPL", "MSFT"], ["UNRATE"], extracted)
        assert kept_tickers == ["AAPL", "MSFT"]
        assert kept_fred == ["UNRATE"]

    def test_identifier_cap_keeps_all_extracted_even_past_the_limit(self) -> None:
        # Correctness beats the bound when they conflict: dropping a resolving source to
        # honor a capacity cap would silently answer the wrong question.
        many = [f"EX{i}" for i in range(MAX_FINANCIAL_IDENTIFIERS + 5)]
        extracted = {"tickers": many, "fred_series": []}
        kept_tickers, kept_fred = _cap_identifiers([*many, "CLASSIFIER_EXTRA"], [], extracted)
        assert kept_tickers == many, "every extracted id survives"
        assert "CLASSIFIER_EXTRA" not in kept_tickers
        assert kept_fred == []

    @pytest.mark.asyncio
    async def test_dead_classifier_records_a_loss_token_before_the_empty_early_return(self) -> None:
        """A dead classifier must not look like "no financial angle" on the diagnostics line.

        The classifier's ``except`` logs one WARNING and returns None; with no extracted
        identifiers the provider then returns "" — and the per-identifier
        ``record_provider_detail`` call lives AFTER that early return, so the failure used
        to produce no diagnostics detail at all. That makes a model retirement, a schema
        change, or a quota indistinguishable from a weather question.
        """
        mock_llm = AsyncMock()
        mock_llm.invoke.side_effect = RuntimeError("classifier model retired")

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data.yfinance") as mock_yf,
        ):
            question = _make_q("Will it rain in London tomorrow?")
            question.id_of_question = 4242
            question.resolution_criteria = ""
            question.fine_print = ""
            provider = financial_data_provider()
            result = await provider(question)

        assert result == "", "no identifiers means no financial section, as before"
        mock_yf.Ticker.assert_not_called()
        sources = pop_provider_detail(4242, "financial_data")["sources"]
        assert _is_lost_source(sources["classifier"]), f"a dead classifier must read as a LOST source; got {sources}"
        assert "RuntimeError" in sources["classifier"]

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

            provider = financial_data_provider()
            result = await provider(_make_q("Will Apple stock exceed $200?"))

        _, kwargs = mock_ticker.history.call_args
        assert kwargs == {"period": f"{FINANCIAL_YFINANCE_LOOKBACK_DAYS}d"}
        assert "### AAPL" in result
