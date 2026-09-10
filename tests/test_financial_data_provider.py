"""Tests for the financial data research provider (yfinance + FRED)."""

import math
import re
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
from forecasting_tools import GeneralLlm
from pandas.tseries.holiday import USFederalHolidayCalendar

from metaculus_bot.constants import (
    FINANCIAL_VARIANCE_RATIO_FLOOR,
    FINANCIAL_VARIANCE_RATIO_LAG,
    FINANCIAL_YFINANCE_LOOKBACK_DAYS,
    MAX_FINANCIAL_IDENTIFIERS,
)
from metaculus_bot.research.financial_data import (
    _PERIOD_SLIP_GRACE_DAYS,
    _PERIOD_TARGET_DAYS,
    CLASSIFIER_PROMPT,
    FRED_LABELS,
    FRED_SKIPPED_NO_KEY_TOKEN,
    KNOWN_FRED_SERIES,
    KNOWN_TICKERS,
    TICKER_LABELS,
    _cap_identifiers,
    _classify_financial_question,
    _fetch_yfinance_data,
    extract_financial_identifiers_from_criteria,
    financial_data_provider,
)
from metaculus_bot.research.fred_rendering import _fetch_fred_data_ceiling
from metaculus_bot.research.noise_flag import NoiseScreen, noise_flag_line, screen_for_quote_noise
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
from scripts.telemetry.markers import MARKER_SPECS
from tests.financial_fakes import (
    _BENCH_OPEN_TIME,
    _clean_close,
    _make_bench_q,
    _make_q,
    _noisy_close,
    _yfinance_by_symbol,
)

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

    def test_the_emitted_line_parses_under_the_marker_spec(self, caplog: pytest.LogCaptureFixture) -> None:
        """The harvest contract, checked against the REAL logged line rather than a literal.

        `"FINANCIAL_NOISE_FLAG" in caplog.text` was the only assertion on this emitter, so a
        field reorder — or deleting the logger.info outright — left the suite green while the
        archive silently harvested nothing. The parser-side tests cannot close that: they parse
        hand-typed strings whose comments claim "in the shape X emits" with nothing enforcing
        it. Same pattern as tests/test_ts_routing.py's ts_anchor_route test.
        """
        noisy = _noisy_close(seed=3)
        with caplog.at_level("INFO", logger="metaculus_bot.research.financial_data"):
            self._fetch(noisy, ticker="USDSZL=X")

        ratio = variance_ratio(noisy, lag=FINANCIAL_VARIANCE_RATIO_LAG, min_returns=120)
        robust = multi_period_annualized_vol_pct(
            noisy, lag=FINANCIAL_VARIANCE_RATIO_LAG, periods_per_year=252, min_returns=120
        )
        short = annualized_realized_vol_pct(noisy, window=30, periods_per_year=252)
        long_vol = annualized_realized_vol_pct(noisy, window=252, periods_per_year=252)
        assert ratio is not None
        assert robust is not None
        assert short is not None
        assert long_vol is not None

        (line,) = [r.getMessage() for r in caplog.records if "FINANCIAL_NOISE_FLAG" in r.getMessage()]
        spec = next(s for s in MARKER_SPECS if s.name == "financial_noise_flag")
        match = spec.regex.search(line)
        assert match is not None
        assert match.groupdict() == {
            "surface": "financial_data",
            "symbol": "USDSZL=X",
            "vr_lag": str(FINANCIAL_VARIANCE_RATIO_LAG),
            "vr": f"{ratio:.3f}",
            "floor": str(FINANCIAL_VARIANCE_RATIO_FLOOR),
            "short_vol": f"{short:.1f}",
            "long_vol": str(round(long_vol, 1)),
            "robust_vol": str(round(robust, 1)),
        }

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


class TestSharedNoiseScreen:
    """One owner for the screen and the telemetry line, so the two surfaces cannot drift.

    The screen and the FINANCIAL_NOISE_FLAG format string were duplicated across
    `financial_data._volatility_lines` and `ts_render._realized_vol_lines`, which is how the
    two copies of the vol estimator drifted before (the q44882 sqrt(252)-on-a-24/7-series
    defect was fixed in one copy weeks before the other). Only the forecaster-facing prose is
    local to each renderer now.
    """

    @staticmethod
    def _spec_regex() -> re.Pattern[str]:
        return next(s for s in MARKER_SPECS if s.name == "financial_noise_flag").regex

    def test_the_screen_fires_only_below_the_floor(self) -> None:
        assert screen_for_quote_noise(_clean_close(seed=3), periods_per_year=252) is None
        fired = screen_for_quote_noise(_noisy_close(seed=3), periods_per_year=252)
        assert fired is not None
        assert fired.ratio < FINANCIAL_VARIANCE_RATIO_FLOOR
        # Same refusal policy for both estimators, so a fired screen always carries a remedy.
        assert fired.robust_vol_pct is not None

    def test_a_sample_the_estimator_refuses_is_not_a_flag(self) -> None:
        """An administratively fixed quote has no measurable return variance, so the ratio
        would be float noise over float noise — and "no measurement" must not read as a flag."""
        flat = pd.Series([7.4604] * 300, index=pd.date_range(end="2026-03-02", periods=300, freq="B"))
        assert screen_for_quote_noise(flat, periods_per_year=252) is None

    def test_both_surfaces_emit_the_same_field_set_under_the_spec_regex(self) -> None:
        """The anti-drift property R12 exists for: one shape, every field required, so a
        reorder harvests as a clean zero rather than recording None for an emitted value."""
        screen = NoiseScreen(ratio=0.412, robust_vol_pct=9.44)
        yfinance_line = noise_flag_line(
            screen, surface="financial_data", symbol="USDSZL=X", short_vol=17.85, long_vol=15.24
        )
        anchor_line = noise_flag_line(screen, surface="ts_anchor", symbol="CSUSHPISA", short_vol=14.6, long_vol=None)

        regex = self._spec_regex()
        yfinance_match = regex.search(yfinance_line)
        anchor_match = regex.search(anchor_line)
        assert yfinance_match is not None
        assert anchor_match is not None
        assert yfinance_match.groupdict() == {
            "surface": "financial_data",
            "symbol": "USDSZL=X",
            "vr_lag": str(FINANCIAL_VARIANCE_RATIO_LAG),
            "vr": "0.412",
            "floor": str(FINANCIAL_VARIANCE_RATIO_FLOOR),
            "short_vol": "17.9",
            "long_vol": "15.2",
            "robust_vol": "9.4",
        }
        # The anchor computes no long window; `surface` is what tells that apart from a
        # yfinance series too short to hold one, so the field itself stays present.
        assert anchor_match.groupdict() | {"surface": "financial_data"} == yfinance_match.groupdict() | {
            "symbol": "CSUSHPISA",
            "short_vol": "14.6",
            "long_vol": "None",
        }

    def test_a_refused_remedy_renders_the_none_sentinel(self) -> None:
        line = noise_flag_line(
            NoiseScreen(ratio=0.369, robust_vol_pct=None),
            surface="financial_data",
            symbol="USDSZL=X",
            short_vol=17.85,
            long_vol=None,
        )
        match = self._spec_regex().search(line)
        assert match is not None
        # "None", not 0.0: the archive coerces the sentinel to null rather than a fabricated
        # zero volatility.
        assert match.group("robust_vol") == "None"
        assert match.group("long_vol") == "None"


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
            patch("metaculus_bot.research.fred_rendering.Fred") as mock_fred_class,
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
# Exchange-rate routing: hallucinated FRED ids, the Yahoo cross, and the disclosure
# ---------------------------------------------------------------------------


class TestExchangeRateRouting:
    """q45363 published with no financial block at all on a currency question.

    The classifier proposed the FRED series ``DEXBOUS``, which does not exist -- FRED carries no
    Bolivia daily FX series -- and named no Yahoo cross beside it, so the forecasters got no level
    and no realized volatility on an exchange-rate question. The verification pass measured what
    that cost: a member sized off the resolving series' own 30-print volatility would have scored
    +55.35 spot peer alone, better than every member that actually ran.

    Three things have to hold. A nonexistent id is distinguishable from a live-but-empty series;
    the Yahoo cross the prompt now asks for renders on its own when FRED has nothing; and a pair
    NEITHER vendor serves leaves the section ABSENT while its loss shows up in the diagnostics
    detail. The absence is deliberate and is the AskNews ``No articles were found`` rule: prose
    standing in for a provider's absent output would flip the orchestrator's status from ``empty``
    to ``ok`` and count in ``providers_succeeded``, so the count is where the signal lives.
    """

    FRED_400 = ValueError("The series does not exist.")

    @staticmethod
    def _fx_question(qid: int) -> MagicMock:
        question = _make_q("What will be the Boliviano-USD exchange rate on August 31, 2026?")
        question.id_of_question = qid
        return question

    @staticmethod
    async def _run(question: MagicMock, classifier_response: str, yahoo: dict[str, str]) -> str:
        """Run the provider with a canned classification, a stubbed Yahoo, and a dead FRED."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = classifier_response

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch(
                "metaculus_bot.research.financial_data._fetch_yfinance_data",
                side_effect=lambda ticker, **_: yahoo.get(ticker, ""),
            ),
            patch("metaculus_bot.research.fred_rendering.Fred") as mock_fred_class,
        ):
            mock_fred_class.return_value.get_series.side_effect = TestExchangeRateRouting.FRED_400
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.setenv("FRED_API_KEY", "fake_key")
            try:
                return await financial_data_provider()(question)
            finally:
                monkeypatch.undo()

    @pytest.mark.asyncio
    async def test_a_hallucinated_fred_id_gets_its_own_token_and_the_yahoo_cross_still_renders(self) -> None:
        """The realized q45363 shape, with the fix in place: FRED is dead, Yahoo carries the pair."""
        question = self._fx_question(45363)

        result = await self._run(
            question,
            "FINANCIAL: YES\nTICKERS: USDBOB=X\nFRED_SERIES: DEXBOUS",
            {"USDBOB=X": "### USDBOB=X\n- Latest price: 11.62 (as of 2026-08-14)"},
        )

        assert "### USDBOB=X" in result
        assert "Latest price: 11.62" in result
        detail = pop_provider_detail(question.id_of_question, "financial_data")
        # `unknown_series`, not `empty`: the id does not exist, which is a defect to chase rather
        # than a live source with no observations in the window.
        assert detail["sources"]["DEXBOUS"] == "unknown_series"
        assert _is_lost_source(detail["sources"]["DEXBOUS"])
        assert detail["sources"]["USDBOB=X"] == "ok"
        # The count is a per-identifier vendor outcome, not a property of the section: the FRED id
        # carried nothing even though the Yahoo cross rendered, so the partial gap is still visible.
        assert detail["counts"]["fx_identifiers_empty"] == 1

    @pytest.mark.asyncio
    async def test_a_pair_neither_vendor_serves_leaves_the_section_absent(self) -> None:
        """Both vendors tried, neither carried anything: no section, and the loss in the detail.

        Nothing renders -- not even the routing marker, which only ever rides a body. What a
        residual round reads instead is the two loss tokens plus the count of exchange-rate
        identifiers among them.
        """
        question = self._fx_question(45364)

        result = await self._run(question, "FINANCIAL: YES\nTICKERS: USDBOB=X\nFRED_SERIES: DEXBOUS", yahoo={})

        assert result == ""
        detail = pop_provider_detail(question.id_of_question, "financial_data")
        assert detail["sources"] == {"USDBOB=X": "empty", "DEXBOUS": "unknown_series"}
        assert detail["counts"]["fx_identifiers_empty"] == 2

    @pytest.mark.asyncio
    async def test_a_fred_only_currency_question_is_absent_but_counted(self) -> None:
        """The literal q45363 classification: one bogus FX series, no ticker at all."""
        question = self._fx_question(45365)

        result = await self._run(question, "FINANCIAL: YES\nTICKERS: NONE\nFRED_SERIES: DEXBOUS", yahoo={})

        assert result == ""
        detail = pop_provider_detail(question.id_of_question, "financial_data")
        assert detail["sources"] == {"DEXBOUS": "unknown_series"}
        assert detail["counts"]["fx_identifiers_empty"] == 1

    @pytest.mark.asyncio
    async def test_a_non_currency_question_with_nothing_to_render_counts_nothing(self) -> None:
        """A stock and a macro series carrying nothing are not exchange rates, so the count stays 0.

        The 0 is itself a reading: it says the check ran on this question and found no lost FX
        identifier, which is what separates it from a record where the check never ran.
        """
        question = _make_q("Will Apple stock exceed $200 by end of 2026?")
        question.id_of_question = 45366

        result = await self._run(question, "FINANCIAL: YES\nTICKERS: AAPL\nFRED_SERIES: NOTASERIES", yahoo={})

        assert result == ""
        detail = pop_provider_detail(question.id_of_question, "financial_data")
        assert detail["counts"]["fx_identifiers_empty"] == 0
        assert detail["sources"] == {"AAPL": "empty", "NOTASERIES": "unknown_series"}

    @pytest.mark.asyncio
    async def test_a_skipped_fred_series_is_recorded_as_a_lost_source(self) -> None:
        """Without ``FRED_API_KEY`` the series never becomes a job, and used to leave NO token —
        so N requested series vanished from the source map and the line read as fully healthy."""
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: AAPL\nFRED_SERIES: UNRATE"
        question = _make_q("Will Apple stock rise and unemployment fall?")
        question.id_of_question = 45367

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch(
                "metaculus_bot.research.financial_data._fetch_yfinance_data",
                side_effect=lambda ticker, **_: f"### {ticker}\n- Latest price: 190",
            ),
        ):
            monkeypatch = pytest.MonkeyPatch()
            monkeypatch.delenv("FRED_API_KEY", raising=False)
            try:
                result = await financial_data_provider()(question)
            finally:
                monkeypatch.undo()

        assert "### AAPL" in result
        sources = pop_provider_detail(question.id_of_question, "financial_data")["sources"]
        assert sources["UNRATE"] == FRED_SKIPPED_NO_KEY_TOKEN
        assert _is_lost_source(sources["UNRATE"])

    @pytest.mark.asyncio
    async def test_a_keyless_fred_only_fetch_set_still_records_the_skip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No ``FRED_API_KEY`` and nothing but FRED series: zero jobs, and the skip is the record.

        This is the arm the retired ``if not jobs: return ""`` short-circuited — it returned before
        the per-identifier ``record_provider_detail`` call, so ``pop_provider_detail`` gave ``{}``
        and the requested series vanished from the archive entirely. Every other no-key test
        classifies a ticker, so a job always existed and none of them reached this path. Deliberately
        not routed through ``_run``, which pins ``FRED_API_KEY`` to a fake value.
        """
        mock_llm = AsyncMock()
        mock_llm.invoke.return_value = "FINANCIAL: YES\nTICKERS: NONE\nFRED_SERIES: UNRATE"
        question = _make_q("Will unemployment fall below 4% in 2026?")
        question.id_of_question = 45368
        monkeypatch.delenv("FRED_API_KEY", raising=False)

        with patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm):
            result = await financial_data_provider()(question)

        assert result == ""
        assert pop_provider_detail(question.id_of_question, "financial_data") == {
            "sources": {"UNRATE": FRED_SKIPPED_NO_KEY_TOKEN},
            "counts": {"fx_identifiers_empty": 0},
        }

    def test_the_prompt_routes_exchange_rates_to_a_yahoo_cross(self) -> None:
        """The cause of q45363 was a reference table with no FX coverage at all, so the classifier
        had nothing to route to and invented an id. The routing rule is the fix at that cause."""
        assert "USD<ISO>=X" in CLASSIFIER_PROMPT
        assert "<ISO>USD=X" in CLASSIFIER_PROMPT
        assert "NEVER invent a FRED series ID" in CLASSIFIER_PROMPT


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

        with patch("metaculus_bot.research.fred_rendering.fetch_series", side_effect=fake_fetch_series):
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

        with patch("metaculus_bot.research.fred_rendering.fetch_series", side_effect=fake_fetch_series):
            result = _fetch_fred_data_ceiling("DGS10", _BENCH_OPEN_TIME)

        assert captured["spec"].revises is False
        assert "### DGS10" in result

    def test_fred_ceiling_fetch_soft_fails_on_fetch_error(self) -> None:
        """A ts_fetch FetchError soft-fails to "" (never propagates)."""
        with patch("metaculus_bot.research.fred_rendering.fetch_series", side_effect=FetchError("bad id")):
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
            patch("metaculus_bot.research.fred_rendering.fetch_series", side_effect=fake_fetch_series) as mock_fetch,
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
    async def test_dead_classifier_token_survives_beside_a_fetched_identifier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The classifier loss has to reach the FINAL record too, not only the pre-return one.

        ``record_provider_detail`` assigns the whole dict, so the per-identifier record written
        after the fetches used to overwrite the classifier token wholesale. A dead classifier on a
        question whose resolving FRED series was still recovered from its own resolution URL then
        archived ``{"sources": {"CPIAUCSL": "ok"}}`` — a diagnostics line byte-identical to a fully
        healthy run, leaving the outage's only trace an unstructured run-log WARN that dies with
        the 90-day GHA artifact.
        """
        mock_llm = AsyncMock()
        mock_llm.invoke.side_effect = RuntimeError("classifier model retired")
        question = _make_q(
            "A question the classifier never got to read.",
            resolution_criteria="Resolves to https://fred.stlouisfed.org/series/CPIAUCSL on the close date.",
        )
        question.id_of_question = 4243
        monkeypatch.setenv("FRED_API_KEY", "fake_key")

        with (
            patch("metaculus_bot.research.financial_data.build_llm_with_openrouter_fallback", return_value=mock_llm),
            patch("metaculus_bot.research.financial_data._fetch_fred_data", side_effect=_stub_fred_fetch),
        ):
            result = await financial_data_provider()(question)

        assert "### CPIAUCSL" in result, "the URL-extracted resolving series still fetches"
        detail = pop_provider_detail(4243, "financial_data")
        assert detail["sources"]["CPIAUCSL"] == "ok"
        assert _is_lost_source(detail["sources"]["classifier"])
        assert "RuntimeError" in detail["sources"]["classifier"]
        # The token is not an identifier, so it must not read as a lost exchange rate.
        assert detail["counts"]["fx_identifiers_empty"] == 0

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
