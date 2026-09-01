"""Tests for the financial data research provider (yfinance + FRED)."""

import math
import re
from datetime import UTC, date, datetime, timedelta
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
from urllib.error import URLError
from xml.etree.ElementTree import ParseError

import numpy as np
import pandas as pd
import pytest
from forecasting_tools import GeneralLlm
from pandas.tseries.holiday import USFederalHolidayCalendar

from metaculus_bot.constants import FINANCIAL_YFINANCE_LOOKBACK_DAYS, MAX_FINANCIAL_IDENTIFIERS
from metaculus_bot.research.financial_data import (
    _PERIOD_SLIP_GRACE_DAYS,
    _PERIOD_TARGET_DAYS,
    CLASSIFIER_PROMPT,
    FRED_LABELS,
    HARD_PEG_ANCHORS,
    KNOWN_FRED_SERIES,
    KNOWN_TICKERS,
    TICKER_LABELS,
    _cap_identifiers,
    _classify_financial_question,
    _fetch_fred_data,
    _fetch_fred_data_ceiling,
    _fetch_yfinance_data,
    _format_fred_change,
    _format_fred_value,
    _render_fred_series,
    extract_financial_identifiers_from_criteria,
    financial_data_provider,
)
from metaculus_bot.research.orchestrator import ResearchOrchestrator
from metaculus_bot.research.provider_diagnostics import _is_lost_source, pop_provider_detail
from metaculus_bot.research.ts_estimators import (
    CALENDAR_DAYS_PER_YEAR,
    TRADING_DAYS_PER_YEAR,
    annualized_realized_vol_pct,
    multi_period_annualized_vol_pct,
    observed_periods_per_year,
    stale_latest_age_days,
    variance_ratio,
)
from metaculus_bot.research.ts_fetch import FetchError
from tests.ts_anchor_fakes import noise_dominated_close_series, random_walk_close_series


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
# both deterministic: end == open_time.date() + 1 day, start == open_time.date() minus
# the FINANCIAL_YFINANCE_LOOKBACK_DAYS calendar window.
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
    def _fetch_with_history(close: pd.Series, **kwargs) -> str:
        history = pd.DataFrame(
            {"Close": close, "Open": close * 0.99, "High": close * 1.01, "Low": close * 0.98},
            index=close.index,
        )
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = history
        mock_ticker_instance.info = {"shortName": "Test Asset"}
        with patch("metaculus_bot.research.financial_data.yfinance") as mock_yf:
            mock_yf.Ticker.return_value = mock_ticker_instance
            return _fetch_yfinance_data("TEST", **kwargs)

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
        # The 1y row and the 52-week band both read a true calendar year, not a
        # 252-row count (which on a holiday-bearing exchange index spans ~50 weeks).
        year_ago = close.index[-1] - pd.Timedelta(days=365)
        expected_1y = (close.iloc[-1] / close.loc[:year_ago].iloc[-1] - 1) * 100
        assert self._line_value(result, "- 1y") == f"{expected_1y:+.2f}%"
        year_slice = close[close.index >= year_ago]
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
        # startswith, not equality: the implanted 3x spike-and-revert above is a one-bar
        # reversal, so this fixture legitimately trips the variance-ratio noise screen and
        # the line picks up a trailing noise-suspect label. The claim under test is the
        # VALUE and its annualization basis, which are unchanged.
        assert self._line_value(result, "- 30-calendar-day annualized volatility").startswith(f"{expected_vol:.1f}%")
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
        """The boundary the lookback constant exists to clear: the 1y lookup needs an
        observation at least 365 days back, and the fetch window is what supplies it.
        Built WINDOW-shaped — an end-inclusive span of LOOKBACK+1 calendar dates, exactly
        what the start-date fetch returns for a gap-free 24/7 series — so a future trim
        of the constant fails here, not in prod. (The old version built `periods=LOOKBACK`
        ROWS, silently encoding the gap-free bar-count assumption this replaced.)"""
        assert FINANCIAL_YFINANCE_LOOKBACK_DAYS > 365, "the 1y lookup needs a >365-day window"
        rng = np.random.default_rng(7)
        n_rows = FINANCIAL_YFINANCE_LOOKBACK_DAYS + 1  # end-inclusive window, one bar per date
        dates = pd.date_range(end="2026-07-31", periods=n_rows, freq="D")
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_rows))), index=dates)
        assert observed_periods_per_year(close.index) == 365

        result = self._fetch_with_history(close)

        expected_1y = (close.iloc[-1] / close.iloc[-366] - 1) * 100
        assert self._line_value(result, "- 1y") == f"{expected_1y:+.2f}%"

    def test_one_more_closure_than_worst_observed_density_keeps_the_1y_row(self):
        """The listed-asset margin that measured EXACTLY ZERO under the old sizing: real
        SPY windows bottomed out at 253 bars per 373-date window, so one more unscheduled
        closure (a mourning-day, a Sandy) left 252 bars — and the old row-offset 1y lookup
        (needs strictly more than 252 rows) silently dropped the row. The date-based
        lookup reads the window's SPAN, so the same frame keeps its 1y return."""
        end = pd.Timestamp("2026-08-21")  # a Friday, like a normal trading day
        idx = pd.bdate_range(start=end - pd.Timedelta(days=372), end=end)
        n_closures = len(idx) - 252
        assert n_closures > 0
        closed = np.linspace(30, len(idx) - 30, n_closures).astype(int)
        idx = idx[np.setdiff1d(np.arange(len(idx)), closed)]
        assert len(idx) == 252, "worst observed density plus one extra closure"
        rng = np.random.default_rng(11)
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 252))), index=idx)
        assert observed_periods_per_year(close.index) == 252

        result = self._fetch_with_history(close)

        # Value-pinned, not presence-pinned: a lookup that lands weeks off target would
        # still put a "- 1y" substring in the render.
        start = close.loc[: close.index[-1] - pd.Timedelta(days=365)]
        expected = (close.iloc[-1] / start.iloc[-1] - 1) * 100
        assert self._line_value(result, "- 1y") == f"{expected:+.2f}%"

    def test_scattered_crypto_gaps_keep_the_1y_row(self):
        """Yahoo data holes on a 24/7 series subtract rows instead of extending the window
        (a persistent one-day BTC-USD hole was observed live). Thirty scattered holes take
        a full production window to 361 rows — under the old 366-row requirement, which
        dropped the 1y row; the date-based lookup still finds the year-ago observation."""
        window_dates = FINANCIAL_YFINANCE_LOOKBACK_DAYS + 1
        idx = pd.date_range(end="2026-08-21", periods=window_dates, freq="D")
        holes = np.linspace(30, window_dates - 11, 30).astype(int)
        idx = idx[np.setdiff1d(np.arange(window_dates), holes)]
        assert len(idx) == window_dates - 30
        assert len(idx) < 366, "the fixture must be thinner than the old row requirement"
        rng = np.random.default_rng(13)
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, len(idx)))), index=idx)
        assert observed_periods_per_year(close.index) == 365

        result = self._fetch_with_history(close)

        start = close.loc[: close.index[-1] - pd.Timedelta(days=365)]
        expected = (close.iloc[-1] / start.iloc[-1] - 1) * 100
        assert self._line_value(result, "- 1y") == f"{expected:+.2f}%"

    def test_lookback_constant_covers_both_bases_with_headroom(self):
        """Both binding constraints, so no future trim can under-size either basis: the
        365 basis needs 366 rows plus gap headroom; the 252 basis needs 253 bars plus a
        week of closures, converted at the trading-day density. The old 372 passed the
        first and cleared the second by arithmetic accident (margin zero on real
        windows)."""
        assert FINANCIAL_YFINANCE_LOOKBACK_DAYS >= 365 + 14
        assert math.ceil((252 + 7) * 365.25 / 252) <= FINANCIAL_YFINANCE_LOOKBACK_DAYS
        # Documented crypto tolerance: end-inclusive window dates minus the 366 rows the
        # 1y lookup wants on a gap-free frame.
        assert (FINANCIAL_YFINANCE_LOOKBACK_DAYS + 1) - 366 >= 24

    def test_the_offset_table_is_the_documented_convention(self):
        # The table IS the behavior: every period-return label on every financial snapshot is
        # only a true calendar period because of these numbers. Pinned verbatim so an edit is a
        # deliberate act — ONE table of calendar days for both bases (business-day steps on the
        # 252 basis silently landed ~10 trading days short of a year across market holidays),
        # each resolved to a target DATE and matched at-or-before. What stays per-basis is the
        # slip grace: 3 days absorbs a weekend-plus-holiday landing on the 252 basis, while a
        # 24/7 series should print every date, so ANY slip there disclosves as a data gap.
        assert _PERIOD_TARGET_DAYS == [
            ("1d", 1),
            ("1w", 7),
            ("1m", 30),
            ("3m", 91),
            ("6m", 182),
            ("1y", 365),
        ]
        assert _PERIOD_SLIP_GRACE_DAYS == {TRADING_DAYS_PER_YEAR: 3, CALENDAR_DAYS_PER_YEAR: 0}

    @pytest.mark.parametrize("basis", [TRADING_DAYS_PER_YEAR, CALENDAR_DAYS_PER_YEAR])
    def test_every_period_row_reads_a_true_calendar_period_on_both_bases(self, basis: int):
        # The bases are pinned end-to-end above only on 1w/1y; walk ALL six labels so a
        # mis-keyed intermediate target (3m reading 63 days = 9 calendar weeks under a
        # "3m" label) can't hide between the two tested ones. Table-driven, so a new
        # period label is covered the moment it lands. The expected start is pandas' own
        # at-or-before read of `last - days` — pinning the table plus the match semantics
        # — and on the gap-free daily index the 365-basis reads are additionally asserted
        # bit-identical to the historical row offsets, so the calendar-day rewrite
        # provably left 24/7 numbers unchanged. The marker asserts at the bottom pin the
        # render delta to purely additive (no span disclosures, no staleness).
        rng = np.random.default_rng(3)
        # Enough rows to cover the 365-day 1y target on either index density.
        index = (
            pd.bdate_range(end="2026-07-31", periods=400)
            if basis == TRADING_DAYS_PER_YEAR
            else pd.date_range(end="2026-07-31", periods=400, freq="D")
        )
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index)))), index=index)
        assert observed_periods_per_year(close.index) == basis

        # as_of one day past the last bar: fresh enough that no staleness note can fire,
        # not the bar's own date so no partial-bar marker fires.
        result = self._fetch_with_history(close, as_of=datetime(2026, 8, 1, 12, 0, tzinfo=UTC))

        for label, days in _PERIOD_TARGET_DAYS:
            start = close.loc[: close.index[-1] - pd.Timedelta(days=days)]
            expected = (close.iloc[-1] / start.iloc[-1] - 1) * 100
            assert self._line_value(result, f"- {label}") == f"{expected:+.2f}%", (basis, label)
            if basis == CALENDAR_DAYS_PER_YEAR:
                assert start.iloc[-1] == close.iloc[-(days + 1)], (basis, label)
        assert "(actual" not in result, "weekend slips stay inside the 252 grace; gap-free 365 has none"
        assert "⚠" not in result, "a fresh latest must not read as stale"
        assert "in progress" not in result

    def test_short_series_degrades_to_trading_day_basis(self):
        dates = pd.date_range(end="2026-07-31", periods=10, freq="D")
        close = pd.Series(np.linspace(100.0, 110.0, 10), index=dates)
        assert observed_periods_per_year(close.index) == 252
        # And the full fetch still renders (no vol line under 30 rows, no crash).
        result = self._fetch_with_history(close)
        assert "- Latest price:" in result
        assert "volatility" not in result

    def test_non_datetime_index_degrades_to_trading_day_basis(self):
        close = pd.Series(np.linspace(100.0, 110.0, 50))  # RangeIndex
        assert observed_periods_per_year(close.index) == 252

    def test_zero_span_index_degrades_to_trading_day_basis(self):
        dates = pd.DatetimeIndex(["2026-07-31"] * 50)
        close = pd.Series(np.linspace(100.0, 110.0, 50), index=dates)
        assert observed_periods_per_year(close.index) == 252


class TestDatedLatestAndStaleness:
    """Every rendered "latest" carries its observation date, a still-forming bar says so,
    and a latest older than the series' own cadence allows is flagged as stale.

    The undated "Current price" read as live even when Yahoo's newest bar was days old:
    a weekend Friday close, or a null-close consolidation hole silently dropped from the
    frame by yfinance's keepna=False default — indistinguishable to the reader either way.
    """

    _fetch_with_history = staticmethod(TestAnnualizationBasis._fetch_with_history)
    _line_value = staticmethod(TestAnnualizationBasis._line_value)

    @staticmethod
    def _daily_series(end: str, n: int = 366, freq: str = "D") -> pd.Series:
        rng = np.random.default_rng(5)
        dates = pd.date_range(end=end, periods=n, freq=freq)
        return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)

    def test_latest_price_line_carries_its_observation_date(self):
        close = self._daily_series("2026-08-24")
        result = self._fetch_with_history(close, as_of=datetime(2026, 8, 25, 12, 0, tzinfo=UTC))
        assert "- Latest price:" in result
        assert "(as of 2026-08-24)" in self._line_value(result, "- Latest price")

    def test_todays_bar_is_marked_in_progress(self):
        close = self._daily_series("2026-08-26")
        result = self._fetch_with_history(close, as_of=datetime(2026, 8, 26, 4, 52, tzinfo=UTC))
        assert "(as of 2026-08-26) — today's bar, in progress" in result

    def test_a_completed_bar_carries_no_in_progress_marker(self):
        close = self._daily_series("2026-08-25")
        result = self._fetch_with_history(close, as_of=datetime(2026, 8, 26, 4, 52, tzinfo=UTC))
        assert "in progress" not in result

    def test_benchmarking_never_marks_in_progress(self):
        # Under a backtest the as_of-dated bar is a COMPLETED historical bar fetched
        # later; calling it in-progress would be false.
        close = self._daily_series("2026-08-26")
        result = self._fetch_with_history(close, as_of=datetime(2026, 8, 26, 4, 52, tzinfo=UTC), is_benchmarking=True)
        assert "(as of 2026-08-26)" in result
        assert "in progress" not in result

    def test_stale_calendar_basis_latest_is_flagged_and_warned(self, caplog: pytest.LogCaptureFixture):
        # A 24/7 series should print a bar every date; a 3-day-old latest is beyond the
        # 1-step + 1-grace-day allowance and must be flagged in render AND run logs.
        close = self._daily_series("2026-08-23")
        with caplog.at_level("WARNING"):
            result = self._fetch_with_history(close, as_of=datetime(2026, 8, 26, 4, 52, tzinfo=UTC))
        assert "⚠ Latest observation is 3 days old" in result
        assert "FINANCIAL_STALE_LATEST: surface=financial_data symbol=TEST age_d=3 cadence=calendar-day" in caplog.text

    def test_weekend_friday_close_on_trading_basis_stays_silent(self):
        # Friday close read on Sunday is 2 days old — routine on the 252 basis (the
        # allowance absorbs a weekend plus a holiday), so no flag and no WARN.
        rng = np.random.default_rng(5)
        dates = pd.bdate_range(end="2026-08-21", periods=300)
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=dates)
        result = self._fetch_with_history(close, as_of=datetime(2026, 8, 23, 12, 0, tzinfo=UTC))
        assert "⚠" not in result

    def test_a_tz_aware_exchange_index_ages_in_its_own_timezone(self):
        # yfinance serves listed-asset bars on a tz-aware exchange-local index (ts_fetch
        # normalizes the same frames), while window_end is UTC. Comparing dates across the
        # two zones inflates a US-equity age by one for part of every day: a Friday close
        # read at Wednesday 02:00 UTC (Tuesday 22:00 ET) is 4 ET days old — inside the
        # 252-basis allowance — but 5 UTC days old, a false staleness warning.
        rng = np.random.default_rng(5)
        dates = pd.bdate_range(end="2026-08-21", periods=300, tz="America/New_York")
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))), index=dates)
        result = self._fetch_with_history(close, as_of=datetime(2026, 8, 26, 2, 0, tzinfo=UTC))
        assert "⚠" not in result

    def test_stale_helper_thresholds_per_basis(self):
        # The shared helper's exact boundaries: >2 days on the 365 basis, >4 on the 252.
        as_of = date(2026, 8, 26)
        assert stale_latest_age_days(date(2026, 8, 24), as_of, CALENDAR_DAYS_PER_YEAR) is None
        assert stale_latest_age_days(date(2026, 8, 23), as_of, CALENDAR_DAYS_PER_YEAR) == 3
        assert stale_latest_age_days(date(2026, 8, 22), as_of, TRADING_DAYS_PER_YEAR) is None
        assert stale_latest_age_days(date(2026, 8, 21), as_of, TRADING_DAYS_PER_YEAR) == 5


class TestDateBasedPeriodReturns:
    """Period returns look up their start value by DATE (at-or-before the label's target),
    so a gapped index cannot shift every row by the hole count.

    The row-offset arithmetic this replaced rendered a "1d" return spanning 53 hours when
    Yahoo dropped a null-close bar: every offset walked one row too far, and nothing in
    the render said so.
    """

    _fetch_with_history = staticmethod(TestAnnualizationBasis._fetch_with_history)
    _line_value = staticmethod(TestAnnualizationBasis._line_value)

    @staticmethod
    def _gapped_btc_shape() -> pd.Series:
        # A long healthy 24/7 series whose second-to-last DATE is missing — the observed
        # Yahoo consolidation-hole shape (a null-close bar deleted by keepna=False).
        idx = pd.date_range(end="2026-08-26", periods=400, freq="D")
        idx = idx.drop([pd.Timestamp("2026-08-25")])
        rng = np.random.default_rng(7)
        return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)

    def test_1d_over_a_hole_discloses_the_actual_span(self):
        close = self._gapped_btc_shape()
        result = self._fetch_with_history(close, as_of=datetime(2026, 8, 26, 12, 0, tzinfo=UTC))
        one_d = next(line.strip() for line in result.splitlines() if line.strip().startswith("- 1d"))
        assert one_d.startswith("- 1d (actual 2d):"), one_d

    def test_longer_rows_hit_the_same_calendar_dates_as_an_ungapped_index(self):
        # Row offsets would shift 1m/3m/6m/1y one row deep past the hole; the date-based
        # lookup must read the observation dated exactly N days before the last bar.
        close = self._gapped_btc_shape()
        result = self._fetch_with_history(close, as_of=datetime(2026, 8, 26, 12, 0, tzinfo=UTC))
        last_ts = close.index[-1]
        for label, days in [("1m", 30), ("3m", 91), ("6m", 182), ("1y", 365)]:
            expected = (close.iloc[-1] / close.loc[last_ts - pd.Timedelta(days=days)] - 1) * 100
            assert self._line_value(result, f"- {label}") == f"{expected:+.2f}%", label

    def test_a_nan_close_row_renders_identically_to_a_dropped_row(self):
        """Yahoo's consolidation hole arrives in TWO representations: yfinance's
        keepna=False default usually deletes the row, but a NaN close that does arrive
        must not poison the tail-anchored stats — without the dropna, "Latest price:
        nan" and NaN period returns go straight into a forecaster prompt."""
        dropped = self._gapped_btc_shape()
        as_nan_row = dropped.reindex(pd.date_range(end="2026-08-26", periods=400, freq="D"))
        assert as_nan_row.isna().sum() == 1

        as_of = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
        result_nan = self._fetch_with_history(as_nan_row, as_of=as_of)

        assert result_nan == self._fetch_with_history(dropped, as_of=as_of)
        assert "- 1d (actual 2d):" in result_nan
        assert "nan" not in result_nan  # an f-string'd NaN renders exactly "nan"

    def test_an_all_null_close_frame_renders_nothing(self, caplog: pytest.LogCaptureFixture):
        idx = pd.date_range(end="2026-08-26", periods=40, freq="D")
        all_nan = pd.Series(np.nan, index=idx)
        with caplog.at_level("WARNING"):
            result = self._fetch_with_history(all_nan, as_of=datetime(2026, 8, 26, 12, 0, tzinfo=UTC))
        assert result == ""
        assert "no non-null closes" in caplog.text

    def test_a_holiday_bearing_exchange_index_still_reads_a_full_calendar_year(self):
        """A real exchange index is missing every market HOLIDAY, not just weekends.

        Resolving the 252-basis targets in business days (`pd.offsets.BDay`) landed the
        "1y" match ~10 trading days short on a NYSE-shaped index — ~352 calendar days
        back under a label that claims a year, with slip 0 so the `(actual Nd)`
        disclosure never fired. Targets resolve in calendar days on BOTH bases, so a
        holiday under the target is absorbed by the at-or-before match within the slip
        grace instead of silently shortening every long horizon.
        """

        end = pd.Timestamp("2026-08-21")  # a Friday
        idx = pd.bdate_range(start=end - pd.Timedelta(days=420), end=end)
        holidays = USFederalHolidayCalendar().holidays(start=idx[0], end=idx[-1])
        idx = idx.difference(holidays)
        rng = np.random.default_rng(17)
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
        assert observed_periods_per_year(close.index) == 252

        result = self._fetch_with_history(close, as_of=datetime(2026, 8, 22, 12, 0, tzinfo=UTC))

        last_ts = close.index[-1]
        for label, days in [("1m", 30), ("3m", 91), ("6m", 182), ("1y", 365)]:
            start = close.loc[: last_ts - pd.Timedelta(days=days)]
            expected = (close.iloc[-1] / start.iloc[-1] - 1) * 100
            assert self._line_value(result, f"- {label}") == f"{expected:+.2f}%", label
            actual_span = (last_ts - start.index[-1]).days
            assert days - 3 <= actual_span <= days, (label, actual_span)
        assert "(actual" not in result, "weekend/holiday slip is routine on the 252 basis, not a mislabel"

    def test_a_holiday_under_a_trading_basis_target_stays_undisclosed(self):
        # A market holiday sitting exactly under the 1m target slips the match one
        # business day — routine on the 252 basis, so the label must stay plain AND
        # still read the nearest prior observation.
        idx = pd.bdate_range(end="2026-08-21", periods=400)
        holiday = pd.Timestamp("2026-07-22")  # the exact 30-calendar-day 1m target, a Wednesday
        assert holiday in idx
        idx = idx.drop([holiday])
        rng = np.random.default_rng(9)
        close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
        result = self._fetch_with_history(close, as_of=datetime(2026, 8, 22, 12, 0, tzinfo=UTC))
        one_m = next(line.strip() for line in result.splitlines() if line.strip().startswith("- 1m"))
        assert one_m.startswith("- 1m:"), one_m
        expected = (close.iloc[-1] / close.loc[:holiday].iloc[-1] - 1) * 100
        assert one_m == f"- 1m: {expected:+.2f}%"


def _history_frame(close: pd.Series) -> pd.DataFrame:
    """A yfinance-shaped OHLC frame around a close series."""
    return pd.DataFrame(
        {"Close": close, "Open": close * 0.99, "High": close * 1.01, "Low": close * 0.98},
        index=close.index,
    )


def _yfinance_by_symbol(closes: dict[str, pd.Series], *, missing: tuple[str, ...] = ()) -> MagicMock:
    """A yfinance module mock whose Ticker() dispatches per symbol.

    Symbols in ``missing`` return an empty history, which is how a peg anchor that Yahoo
    does not serve reaches the provider's soft-fail path. Any symbol neither mapped nor
    listed raises, so a test can never silently assert against the wrong series. Every
    ``history()`` call is recorded on ``module.history_calls`` as ``(symbol, kwargs)``,
    since each Ticker() call returns a fresh mock whose calls are invisible to the module."""
    history_calls: list[tuple[str, dict]] = []

    def ticker_factory(symbol: str) -> MagicMock:
        instance = MagicMock()
        instance.info = {"shortName": f"{symbol} name"}
        frame = pd.DataFrame() if symbol in missing else None
        if frame is None:
            if symbol not in closes:
                raise AssertionError(f"unexpected yfinance fetch for {symbol!r}")
            frame = _history_frame(closes[symbol])

        def history(**kwargs) -> pd.DataFrame:
            history_calls.append((symbol, kwargs))
            return frame

        instance.history.side_effect = history
        return instance

    module = MagicMock()
    module.Ticker.side_effect = ticker_factory
    module.history_calls = history_calls
    return module


def _clean_close(**kwargs) -> pd.Series:
    return random_walk_close_series(**kwargs)


def _noisy_close(**kwargs) -> pd.Series:
    return noise_dominated_close_series(**kwargs)


class TestPeggedCrossAnchor:
    """A hard-pegged FX cross must arrive labeled, beside its liquid anchor.

    q44797: `USDSZL=X`'s 17.8% "30-day annualized volatility" — 79% vendor noise on a cross
    fixed 1:1 to the rand — went to all six forecasters, four of whom multiplied it into
    their interval width. The honest like-for-like figure off `ZAR=X` was 10.6%. The
    requirement is disclosure plus the anchor, never a silent substitution: the question
    still resolves on the pegged pair, so its own quote has to stay on the page.
    """

    def test_pegged_ticker_renders_both_blocks_and_names_the_peg(self) -> None:
        closes = {"USDSZL=X": _noisy_close(seed=3), "ZAR=X": _clean_close(seed=3)}
        with patch("metaculus_bot.research.financial_data.yfinance", _yfinance_by_symbol(closes)):
            result = _fetch_yfinance_data("USDSZL=X")

        assert "### USDSZL=X" in result, "the pegged pair's own block must survive"
        assert "### ZAR=X" in result, "the liquid anchor must be rendered beside it"
        assert "⚠ Pegged pair: USD/SZL is fixed at par with the South African rand since 1974" in result
        assert "Peg anchor: `ZAR=X`" in result
        assert "_Peg anchor for USDSZL=X" in result
        # The pegged block keeps its own latest price: disclosure, not substitution.
        assert result.index("### USDSZL=X") < result.index("- Latest price:") < result.index("### ZAR=X")

    def test_the_peg_disclosure_precedes_the_statistics_it_qualifies(self) -> None:
        closes = {"USDSZL=X": _noisy_close(seed=4), "ZAR=X": _clean_close(seed=4)}
        with patch("metaculus_bot.research.financial_data.yfinance", _yfinance_by_symbol(closes)):
            result = _fetch_yfinance_data("USDSZL=X")

        assert result.index("⚠ Pegged pair") < result.index("annualized volatility")
        assert result.index("⚠ Pegged pair") < result.index("52-week range")

    def test_both_yahoo_spellings_and_lower_case_resolve_to_the_same_peg(self) -> None:
        """A resolution URL may cite `SZL=X` or `USDSZL=X`, in either case."""
        for ticker in ("SZL=X", "USDSZL=X", "usdszl=x"):
            closes = {ticker: _noisy_close(seed=5), "ZAR=X": _clean_close(seed=5)}
            with patch("metaculus_bot.research.financial_data.yfinance", _yfinance_by_symbol(closes)):
                result = _fetch_yfinance_data(ticker)
            assert "### ZAR=X" in result, f"{ticker} missed its peg anchor"

    def test_a_usd_pegged_currency_says_there_is_no_anchor_to_read(self) -> None:
        """AED/SAR/QAR/HKD are pegged to the USD leg itself, so no third cross exists. The
        honest statement is that the pair has no independent dynamics — not a second block,
        and not silence."""
        closes = {"USDAED=X": _noisy_close(seed=6)}
        with patch("metaculus_bot.research.financial_data.yfinance", _yfinance_by_symbol(closes)) as module:
            result = _fetch_yfinance_data("USDAED=X")

        assert "pegged to the US dollar at 3.6725 since November 1997" in result
        assert "no liquid third-currency cross to read instead" in result
        assert "Do not size a forecast interval" in result
        assert module.Ticker.call_count == 1, "a USD peg must not trigger a second fetch"

    def test_an_unpegged_ticker_renders_no_peg_lines(self) -> None:
        closes = {"EURUSD=X": _clean_close(seed=7)}
        with patch("metaculus_bot.research.financial_data.yfinance", _yfinance_by_symbol(closes)) as module:
            result = _fetch_yfinance_data("EURUSD=X")

        assert "Pegged pair" not in result
        assert "Peg anchor" not in result
        assert module.Ticker.call_count == 1

    def test_an_unfetchable_anchor_degrades_to_a_visible_notice(self) -> None:
        """The anchor is enrichment: losing it must not take the pegged pair's block down,
        and must not pass silently either."""
        closes = {"USDSZL=X": _noisy_close(seed=8)}
        module = _yfinance_by_symbol(closes, missing=("ZAR=X",))
        with patch("metaculus_bot.research.financial_data.yfinance", module):
            result = _fetch_yfinance_data("USDSZL=X")

        assert "### USDSZL=X" in result
        assert "### ZAR=X" not in result
        assert "⚠ The peg anchor `ZAR=X` could not be fetched" in result

    def test_a_failed_primary_fetch_is_still_an_empty_string(self) -> None:
        module = _yfinance_by_symbol({}, missing=("USDSZL=X",))
        with patch("metaculus_bot.research.financial_data.yfinance", module):
            assert _fetch_yfinance_data("USDSZL=X") == ""

    def test_benchmarking_ceiling_applies_to_the_anchor_fetch_too(self) -> None:
        """A leaky anchor would leak just as hard as a leaky primary."""
        closes = {"USDSZL=X": _noisy_close(seed=9), "ZAR=X": _clean_close(seed=9)}
        module = _yfinance_by_symbol(closes)
        with patch("metaculus_bot.research.financial_data.yfinance", module):
            _fetch_yfinance_data("USDSZL=X", as_of=_BENCH_OPEN_TIME, is_benchmarking=True)

        assert module.Ticker.call_count == 2
        # Both fetches carry the same explicit start/end window.
        assert [symbol for symbol, _ in module.history_calls] == ["USDSZL=X", "ZAR=X"]
        expected_start = (_BENCH_OPEN_TIME - timedelta(days=FINANCIAL_YFINANCE_LOOKBACK_DAYS)).date().isoformat()
        for _symbol, kwargs in module.history_calls:
            assert kwargs["end"] == "2026-03-16"  # open_time.date() + 1d, end EXCLUSIVE
            assert kwargs["start"] == expected_start

    def test_no_anchor_is_itself_pegged_so_the_render_cannot_recurse(self) -> None:
        """The one-level recursion bound `_fetch_yfinance_data` documents."""
        anchors = {peg.anchor_ticker for peg in HARD_PEG_ANCHORS.values() if peg.anchor_ticker}
        assert anchors
        assert not (anchors & set(HARD_PEG_ANCHORS)), "an anchor that is itself pegged would recurse"

    def test_every_peg_entry_carries_both_spellings_and_a_dated_regime(self) -> None:
        for ticker, peg in HARD_PEG_ANCHORS.items():
            assert ticker.endswith("=X"), f"{ticker} is not a Yahoo FX ticker form"
            assert HARD_PEG_ANCHORS[f"{peg.currency}=X"] is peg
            assert HARD_PEG_ANCHORS[f"USD{peg.currency}=X"] is peg
            # A regime with no date is an unsourced claim in a forecaster prompt.
            assert re.search(r"\b(19|20)\d{2}\b|1980s", peg.regime), f"{ticker}: {peg.regime!r} has no date"


class TestVolatilityHorizonsAndNoiseFlag:
    """Two volatility horizons, and the variance-ratio noise flag that reorders them."""

    @staticmethod
    def _fetch(close: pd.Series, ticker: str = "TEST") -> str:
        with patch("metaculus_bot.research.financial_data.yfinance", _yfinance_by_symbol({ticker: close})):
            return _fetch_yfinance_data(ticker)

    def test_a_clean_series_prints_short_then_long_with_no_flag(self) -> None:
        clean = _clean_close(seed=3)
        result = self._fetch(clean)

        short_expected = annualized_realized_vol_pct(clean, window=30, periods_per_year=252)
        long_expected = annualized_realized_vol_pct(clean, window=252, periods_per_year=252)
        assert short_expected is not None
        assert long_expected is not None
        assert f"- 30-trading-day annualized volatility: {short_expected:.1f}%" in result
        assert f"- 252-trading-day annualized volatility: {long_expected:.1f}%" in result
        assert result.index("- 30-trading-day") < result.index("- 252-trading-day")
        assert "Vendor-noise flag" not in result

    def test_the_long_horizon_line_names_the_rows_it_actually_holds(self) -> None:
        """264 returns is under a year, so the label must say 264, not 252."""
        clean = _clean_close(seed=3, n=200)
        result = self._fetch(clean)
        assert "- 199-trading-day annualized volatility:" in result

    def test_too_little_history_prints_only_the_short_window(self) -> None:
        short = _clean_close(seed=3, n=31)
        result = self._fetch(short)
        assert "- 30-trading-day annualized volatility:" in result
        assert result.count("annualized volatility") == 1
        assert "Vendor-noise flag" not in result

    def test_a_noise_dominated_series_flags_and_leads_with_the_robust_figure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        noisy = _noisy_close(seed=3)
        with caplog.at_level("INFO", logger="metaculus_bot.research.financial_data"):
            result = self._fetch(noisy)

        ratio = variance_ratio(noisy, lag=5, min_returns=120)
        robust = multi_period_annualized_vol_pct(noisy, lag=5, periods_per_year=252, min_returns=120)
        assert ratio is not None
        assert robust is not None
        assert f"variance ratio VR(5) = {ratio:.2f}" in result
        assert "Noise-robust annualized volatility, from overlapping 5-trading-day returns" in result
        assert f"{robust:.1f}% — size intervals from THIS figure" in result
        # Ordering: robust first, then the long window, then the noise-suspect short window.
        assert result.index("Noise-robust") < result.index("- 252-trading-day") < result.index("- 30-trading-day")
        assert "(from one-day returns, noise included; noise-suspect)" in result
        assert "FINANCIAL_NOISE_FLAG" in caplog.text

    def test_the_robust_figure_is_the_short_one_scaled_by_the_root_ratio(self) -> None:
        """The remedy is the flag's own arithmetic, not an unrelated number: the multi-day
        volatility equals the one-day figure times sqrt(VR) in log-return terms."""
        noisy = _noisy_close(seed=11)
        ratio = variance_ratio(noisy, lag=5, min_returns=120)
        robust = multi_period_annualized_vol_pct(noisy, lag=5, periods_per_year=252, min_returns=120)
        assert ratio is not None
        assert robust is not None
        log_returns = np.diff(np.log(noisy.to_numpy(dtype="float64")))
        one_day = float(log_returns.std(ddof=1) * np.sqrt(252) * 100.0)
        assert robust == pytest.approx(one_day * math.sqrt(ratio), rel=1e-9)


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


class TestRenderFredSeriesZeroBasePercent:
    """A base of exactly 0 has no percent change; it must not render as 0.00%.

    FRED spread series cross zero routinely (T10Y2Y inverted through 2023-24), and the
    old ``else 0`` put a fabricated "unchanged" percentage next to a genuine absolute
    move in a forecaster prompt.
    """

    def test_zero_previous_observation_omits_the_percent_clause(self) -> None:
        dates = pd.date_range(end="2026-03-01", periods=3, freq="MS")
        data = pd.Series([0.5, 0.0, 0.31], index=dates, name="T10Y2Y")

        markdown = _render_fred_series("T10Y2Y", data, "10Y-2Y spread")
        change_line = next(line for line in markdown.splitlines() if line.startswith("- Change from previous:"))

        assert change_line == "- Change from previous: +0.31"
        assert "0.00%" not in markdown

    def test_zero_year_ago_observation_omits_only_the_yoy_percent(self) -> None:
        # 25 monthly observations so the ~365-day lookup lands on a real row, which is
        # set to exactly 0. The month-over-month clause is unaffected.
        dates = pd.date_range(end="2026-03-01", periods=25, freq="MS")
        values = [1.0] * 25
        values[12] = 0.0  # the observation ~365 days before the last one
        data = pd.Series(values, index=dates, name="T10Y3M")
        data.iloc[-1] = 0.4
        data.iloc[-2] = 0.2

        markdown = _render_fred_series("T10Y3M", data, "10Y-3M spread")
        yoy_line = next(line for line in markdown.splitlines() if line.startswith("- Year-over-year change:"))
        mom_line = next(line for line in markdown.splitlines() if line.startswith("- Change from previous:"))

        assert yoy_line == "- Year-over-year change: +0.4"
        assert "(+100.00%)" in mom_line

    def test_a_nonzero_base_still_renders_its_percent(self) -> None:
        dates = pd.date_range(end="2026-03-01", periods=2, freq="MS")
        data = pd.Series([2.0, 3.0], index=dates, name="UNRATE")

        markdown = _render_fred_series("UNRATE", data, "unemployment rate")

        assert "- Change from previous: +1 (+50.00%)" in markdown


def _monthly_fred(values: list[float], name: str = "CSUSHPISA", end: str = "2026-06-01") -> pd.Series:
    """A month-start FRED series, oldest first — the shape both render paths hand over."""
    return pd.Series(values, index=pd.date_range(end=end, periods=len(values), freq="MS"), name=name)


class TestFredValuePrecision:
    """FRED levels must render at the precision the agency published them at.

    q44944 resolved on a Case-Shiller print of 331.893 inside a displayed range four index
    points wide with 0.02-point buckets, and the provider — the one component designed to
    read the resolving series directly — rendered it through `:.4g` as "331.9". The exact
    value reached the forecasters only because two gap-fill passes independently quoted the
    FRED page.
    """

    def test_a_case_shiller_scale_level_keeps_all_its_digits(self) -> None:
        data = _monthly_fred([330.873, 331.359, 331.020, 331.893])

        markdown = _render_fred_series("CSUSHPISA", data, "Case-Shiller home price index")

        assert "- Latest value: 331.893 (2026-06-01)" in markdown
        assert "- Previous value: 331.02" in markdown
        assert "  - 2026-05-01: 331.02" in markdown
        assert "331.9\n" not in markdown, "the `:.4g` rounding must be gone from every line"

    def test_the_change_line_keeps_its_precision_and_loses_float_noise(self) -> None:
        data = _monthly_fred([331.020, 331.893])

        markdown = _render_fred_series("CSUSHPISA", data, "Case-Shiller home price index")

        # 331.893 - 331.02 is 0.8729999999999905 in binary floating point.
        assert "- Change from previous: +0.873 (+0.26%)" in markdown

    def test_a_large_level_never_renders_in_scientific_notation(self) -> None:
        """WALCL (the Fed balance sheet, in millions) went out as "6.7e+06" under `:.4g`."""
        data = _monthly_fred([6_698_123.0, 6_699_580.0], name="WALCL")

        markdown = _render_fred_series("WALCL", data, "Fed balance sheet")

        assert "- Latest value: 6699580 (2026-06-01)" in markdown
        assert "e+0" not in markdown

    def test_a_rate_still_renders_without_trailing_zeros(self) -> None:
        data = _monthly_fred([4.15, 4.2], name="DGS10")

        markdown = _render_fred_series("DGS10", data, "10Y Treasury rate")

        assert "- Latest value: 4.2 (2026-06-01)" in markdown
        assert "- Change from previous: +0.05" in markdown

    def test_the_formatters_own_contract(self) -> None:
        """Unit-level, because the interesting inputs (float noise, a magnitude that rounds
        away, a true zero) do not arise from real FRED decimals. A negative sign on a
        magnitude that rounds to zero would not be information."""
        assert _format_fred_value(331.893) == "331.893"
        assert _format_fred_value(6_699_580.0) == "6699580"
        assert _format_fred_value(4.2) == "4.2"
        assert _format_fred_value(0.0) == "0"
        assert _format_fred_value(331.893 - 331.020) == "0.873"
        assert _format_fred_change(-0.749) == "-0.749"
        assert _format_fred_change(0.0) == "+0"
        assert _format_fred_change(-0.0) == "+0"
        assert _format_fred_change(-1e-9) == "+0"


class TestFredFirstReleaseTable:
    """The first-release-vs-current-vintage table for a revising resolving series.

    q44944's resolving quantity was the FIRST published Case-Shiller print while every level
    the provider rendered was today's revised vintage; anchoring on a revision-adjusted May
    was worth +66.6 spot peer. The table turns the revision channel from a symmetric-noise
    assumption into a signed input — and carries the double-count guard, because stacking it
    on a same-source leading indicator overshot by 0.7 index points and lost 15 spot peer.
    """

    @staticmethod
    def _fred_mock(
        current: pd.Series, first_releases: pd.Series | None, vintage_error: Exception | None = None
    ) -> MagicMock:
        """A fredapi mock whose get_series answers the plain and initial-release calls.

        The initial-release call is the one carrying ``output_type``; recorded on
        ``mock.first_release_calls`` so a test can assert the request shape."""
        first_release_calls: list[dict] = []

        def get_series(series_id: str, **kwargs) -> pd.Series:
            del series_id
            if "output_type" in kwargs:
                first_release_calls.append(kwargs)
                if first_releases is None:
                    raise vintage_error or ValueError("Bad Request. Invalid output_type.")
                return first_releases
            return current

        instance = MagicMock()
        instance.get_series.side_effect = get_series
        instance.get_series_info.return_value = pd.DataFrame({"title": ["S&P Case-Shiller"]}, index=["CSUSHPISA"])
        instance.first_release_calls = first_release_calls
        return instance

    # Five months of Case-Shiller, current vintage against first release. The last four
    # pairs are the table's rows: +0.873, +0.43, 0.0 (unrevised) and -0.749 — the same
    # both-directions ±0.3-0.8 revision channel the dossier measured across three instances.
    _CURRENT: ClassVar[list[float]] = [330.44, 330.873, 331.359, 331.020, 331.893]
    _FIRST: ClassVar[list[float]] = [330.16, 331.622, 331.359, 330.590, 331.020]

    def _fetch(
        self,
        *,
        is_resolving_source: bool,
        first_releases: pd.Series | None = None,
        vintage_error: Exception | None = None,
    ) -> tuple[str, MagicMock]:
        current = _monthly_fred(self._CURRENT)
        if vintage_error is not None:
            releases = None
        else:
            releases = _monthly_fred(self._FIRST) if first_releases is None else first_releases
        instance = self._fred_mock(current, releases, vintage_error)
        with patch("metaculus_bot.research.financial_data.Fred", return_value=instance) as fred_class:
            fred_class.earliest_realtime_start = "1776-07-04"
            fred_class.latest_realtime_end = "9999-12-31"
            markdown = _fetch_fred_data("CSUSHPISA", "fake_key", is_resolving_source=is_resolving_source)
        return markdown, instance

    def test_a_resolving_series_renders_the_table_with_the_double_count_guard(self) -> None:
        markdown, instance = self._fetch(is_resolving_source=True)

        assert "- First release vs current vintage" in markdown
        assert "  - 2026-06-01: first release 331.02 → current vintage 331.893 (revised +0.873)" in markdown
        assert "  - 2026-05-01: first release 330.59 → current vintage 331.02 (revised +0.43)" in markdown
        # An unrevised print says so rather than rendering "revised +0".
        assert "  - 2026-04-01: first release 331.359 → current vintage 331.359 (unrevised)" in markdown
        # 4 prints: +0.873, +0.43, 0.0 (331.359 unrevised), -0.749.
        assert "Of these 4 prints, 2 were revised up, 1 down and 1 not at all" in markdown
        assert "mean revision +0.1385" in markdown
        assert "⚠ Do not double-count" in markdown
        assert "Apply one of them, not both" in markdown
        assert len(instance.first_release_calls) == 1

    def test_the_initial_release_request_opens_the_full_real_time_window(self) -> None:
        """Both real-time bounds default to TODAY, which would restrict the answer to prints
        that were never revised — exactly the ones with nothing to report."""
        _markdown, instance = self._fetch(is_resolving_source=True)

        (kwargs,) = instance.first_release_calls
        assert kwargs["output_type"] == 4
        assert kwargs["realtime_start"] == "1776-07-04"
        assert kwargs["realtime_end"] == "9999-12-31"
        # Bounded to the prints the table renders, read off the dates already in hand.
        assert kwargs["observation_start"] == pd.Timestamp("2026-03-01")

    def test_a_classifier_only_series_makes_no_vintage_request(self) -> None:
        """The revision channel matters for the series a question GRADES against; every
        other identifier would just be another HTTP round trip inside the fetch thread."""
        markdown, instance = self._fetch(is_resolving_source=False)

        assert "### CSUSHPISA" in markdown
        assert "First release vs current vintage" not in markdown
        assert instance.first_release_calls == []

    def test_a_non_revising_series_makes_no_vintage_request(self) -> None:
        """DGS10 is a market rate on the non-revising allowlist: a first-release table there
        would be a column of zeros dressed as a finding."""
        current = _monthly_fred([4.15, 4.2], name="DGS10")
        instance = self._fred_mock(current, current)
        with patch("metaculus_bot.research.financial_data.Fred", return_value=instance):
            markdown = _fetch_fred_data("DGS10", "fake_key", is_resolving_source=True)

        assert "### DGS10" in markdown
        assert instance.first_release_calls == []

    @pytest.mark.parametrize(
        "vintage_error",
        [
            # fredapi re-raises the API's own error message as ValueError...
            ValueError("Bad Request. Invalid output_type."),
            # ...a transport failure arrives as URLError, an OSError...
            URLError("connection reset"),
            # ...and a non-XML body (a proxy or status page answering instead) reaches
            # ET.fromstring, whose ParseError is a SyntaxError and matches neither above.
            ParseError("syntax error: line 1, column 0"),
        ],
        ids=["api_error", "transport", "unparseable_body"],
    )
    def test_a_failed_vintage_fetch_leaves_the_primary_block_standing(self, vintage_error: Exception) -> None:
        """The table is enrichment; the series itself is the source, and no failure mode of
        the extra call may take it down."""
        markdown, instance = self._fetch(is_resolving_source=True, vintage_error=vintage_error)

        assert "- Latest value: 331.893 (2026-06-01)" in markdown
        assert "First release vs current vintage" not in markdown
        assert len(instance.first_release_calls) == 1

    def test_a_response_missing_the_latest_print_drops_the_table(self) -> None:
        """The guard on the one inference in this path — that opening the real-time window
        really does return revised prints' first releases. If FRED ever answers with only
        never-revised observations, the newest print is absent and a table of older ones
        under a "recent prints" label would be a different claim than the one being made."""
        stale_releases = _monthly_fred(self._FIRST[:-1], end="2026-05-01")
        markdown, _instance = self._fetch(is_resolving_source=True, first_releases=stale_releases)

        assert "- Latest value: 331.893 (2026-06-01)" in markdown
        assert "First release vs current vintage" not in markdown

    def test_the_benchmarking_path_renders_the_same_shape_without_a_table(self) -> None:
        """One renderer for both paths. The keyless ALFRED CSV serves a series AS OF a
        vintage, not each print's first release, so a backtest cannot measure this feature —
        the same limitation prediction_market and resolution_source carry."""
        current = _monthly_fred(self._CURRENT)
        with patch("metaculus_bot.research.financial_data.fetch_series", return_value=current):
            markdown = _fetch_fred_data_ceiling("CSUSHPISA", _BENCH_OPEN_TIME)

        assert "- Latest value: 331.893 (2026-06-01)" in markdown
        assert "First release vs current vintage" not in markdown

    @pytest.mark.asyncio
    async def test_the_provider_marks_url_extracted_series_as_resolving(self) -> None:
        """End-to-end wiring: only the URL-extracted series gets is_resolving_source=True,
        the classifier's extra context series does not."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: NONE\nFRED_SERIES: HOUST"
        question = _make_q(
            "Where will the Case-Shiller index print for June?",
            resolution_criteria="Resolves per https://fred.stlouisfed.org/series/CSUSHPISA.",
        )
        seen: dict[str, bool] = {}

        def _fred(series_id: str, api_key: str, *, is_resolving_source: bool = False) -> str:
            del api_key
            seen[series_id] = is_resolving_source
            return f"### {series_id} (stub)"

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data._fetch_fred_data", side_effect=_fred),
        ):
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("FRED_API_KEY", "fake_key")
            try:
                await financial_data_provider()(question)
            finally:
                monkeypatch.undo()

        assert seen == {"HOUST": False, "CSUSHPISA": True}


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


def _stub_fred_fetch(series_id: str, api_key: str, **_kwargs: object) -> str:
    """Recognizable FRED markdown so tests assert on routing, not the live API.

    ``**_kwargs`` absorbs the real fetcher's keyword-only flags (``is_resolving_source``),
    which these routing tests do not exercise."""
    del api_key
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
            caplog.at_level("WARNING", logger="metaculus_bot.research.financial_data"),
        ):
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
        assert frozenset(TICKER_LABELS) == KNOWN_TICKERS
        assert frozenset(FRED_LABELS) == KNOWN_FRED_SERIES


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
        end == open_time.date() + 1 day, start == open_time.date() - the lookback
        constant, and `.info` is never accessed (a PropertyMock that would raise if
        touched)."""
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
        expected_start = (_BENCH_OPEN_TIME - timedelta(days=FINANCIAL_YFINANCE_LOOKBACK_DAYS)).date()
        assert kwargs["start"] == expected_start.isoformat()
        assert kwargs["end"] == "2026-03-16"  # open_time.date() + 1d (end EXCLUSIVE)
        assert "AAPL" in result
        assert "[omitted under backtest" in result

    def test_yfinance_live_path_fetches_by_start_date_with_no_end_ceiling(self) -> None:
        """Live path (default is_benchmarking=False): fetch by explicit start date only.
        No ``period=`` — Yahoo reads a bare custom period as trading BARS for listed
        assets, a different unit than the calendar days the backtest branch spends the
        same constant as. No ``end=`` — yfinance defaults it to now, keeping today's
        partial bar. `.info` still consulted for fundamentals."""
        dates = pd.date_range(end="2026-03-30", periods=252, freq="B")
        close_prices = np.linspace(150.0, 200.0, 252)
        mock_history = pd.DataFrame({"Close": close_prices}, index=dates)

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_history
        mock_ticker.info = {"shortName": "Apple Inc.", "trailingPE": 28.5}

        as_of = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)
        with patch("metaculus_bot.research.financial_data.yfinance") as mock_yf:
            mock_yf.Ticker.return_value = mock_ticker

            result = _fetch_yfinance_data("AAPL", as_of=as_of)

        mock_ticker.history.assert_called_once()
        _, kwargs = mock_ticker.history.call_args
        expected_start = (as_of - timedelta(days=FINANCIAL_YFINANCE_LOOKBACK_DAYS)).date()
        assert kwargs == {"start": expected_start.isoformat()}
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
        """Default (no is_benchmarking arg) → live path: start-date yfinance fetch with
        no end ceiling, `.info` consulted, and no open_time needed."""
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

            before = datetime.now(UTC)
            provider = financial_data_provider()
            result = await provider(_make_q("Will Apple stock exceed $200?"))
            after = datetime.now(UTC)

        _, kwargs = mock_ticker.history.call_args
        # The provider derives start from its own now(); bracket it so a midnight
        # rollover between our two clock reads can't flake the assert.
        expected_starts = {
            (t - timedelta(days=FINANCIAL_YFINANCE_LOOKBACK_DAYS)).date().isoformat() for t in (before, after)
        }
        assert set(kwargs) == {"start"}
        assert kwargs["start"] in expected_starts
        assert "### AAPL" in result
