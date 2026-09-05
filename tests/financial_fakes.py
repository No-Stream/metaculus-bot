"""Shared fixtures for the financial-data provider tests, mirroring ``ts_anchor_fakes``.

The provider's tests live in three files after the module split — the provider itself
(``test_financial_data_provider``), the peg table (``test_currency_pegs``) and the FRED
renderer (``test_fred_rendering``) — and all three need the same yfinance module mock and the
same synthetic series, so they live here rather than in three copies that can drift.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pandas as pd

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


def _monthly_fred(values: list[float], name: str = "CSUSHPISA", end: str = "2026-06-01") -> pd.Series:
    """A month-start FRED series, oldest first — the shape both render paths hand over."""
    return pd.Series(values, index=pd.date_range(end=end, periods=len(values), freq="MS"), name=name)
