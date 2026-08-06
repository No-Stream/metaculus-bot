"""Tests for the point-in-time time-series fetch layer (``research/ts_fetch.py``).

One seam: ``ts_fetch._http_get`` (the single synchronous HTTP seam returning raw CSV bytes)
is patched with ``tests.ts_anchor_fakes.FakeHttp``, so the real ``fetch_series`` parse,
cache and leakage guard all run. The yfinance path patches ``yfinance.Ticker`` instead,
since that branch synthesizes its CSV from a history frame rather than going through
``_http_get``. No network.

Coverage (one behavior per test):
- FRED: fredgraph for non-revising vs alfredgraph (vintage) for revising; "." -> NaN
  dropped; malformed HTML body -> FetchError; post-ceiling row -> LeakageError; cache reuse
  (one HTTP call for a repeat key).
- yfinance: High vs Close column selection; empty frame and missing column -> FetchError;
  the exclusive-end ceiling arithmetic end-to-end.
- Politeness pacing: the gate spaces concurrent requests, does not toll a lone fetch, and
  is cleared by a session reset.

The rendered-section / routing / provider tests live in ``test_timeseries_anchor_provider.py``
and ``test_ts_routing.py``; they mock ``timeseries_anchor.fetch_series`` wholesale instead.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import pytest

from metaculus_bot.research import timeseries_anchor as ts
from metaculus_bot.research import ts_fetch as tf
from metaculus_bot.research.ts_fetch import (
    ALFRED_CSV_URL,
    FRED_CSV_URL,
    FetchError,
    LeakageError,
    SeriesSpec,
    fetch_series,
)
from tests.ts_anchor_fakes import FakeHttp, _csv


# Test isolation: the fetch layer keeps a parsed-series cache and a process-wide pacing
# clock; both bleed across tests otherwise.
@pytest.fixture(autouse=True)
def _reset_fetch_caches():
    ts._reset_session_caches()
    yield
    ts._reset_session_caches()


class TestPolitenessPacing:
    """The politeness wait must space real requests, not just occupy worker threads.

    ``_http_get`` used to open with an unconditional ``time.sleep(POLITENESS_SLEEP_S)``,
    which was strictly worse on both counts. ``fetch_series`` runs under
    ``asyncio.to_thread`` from two providers, so N concurrent fetches all slept in
    parallel and then fired within milliseconds of each other (measured: 6 concurrent
    calls, gaps of 0.000-0.009s) — no pacing at all — while each held a slot in the
    process-wide default executor for 0.5s doing nothing. That executor is shared with
    financial_data's fan-out, resolution_source, the agentic fetch ladder, and the
    /auth/key probe, and a task queued behind a full pool burns its ``wait_for`` budget
    without running, so idle sleeps convert into timeouts elsewhere.
    """

    def test_concurrent_fetches_are_actually_spaced(self, monkeypatch):
        tf._reset_politeness_clock()
        monkeypatch.setattr(tf, "POLITENESS_SLEEP_S", 0.05)
        fired: list[float] = []
        lock = threading.Lock()

        def record() -> None:
            tf._politeness_gate()
            with lock:
                fired.append(time.monotonic())

        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in [pool.submit(record) for _ in range(4)]:
                future.result()

        gaps = [b - a for a, b in zip(sorted(fired), sorted(fired)[1:])]
        assert len(gaps) == 3
        for gap in gaps:
            assert gap >= 0.04, f"consecutive requests must be spaced, got gaps {gaps}"

    def test_a_lone_fetch_does_not_wait(self, monkeypatch):
        # The gate is a MINIMUM INTERVAL, not a fixed toll: the common case (one fetch,
        # or fetches already far apart) must not pay for spacing it doesn't need.
        tf._reset_politeness_clock()
        monkeypatch.setattr(tf, "POLITENESS_SLEEP_S", 0.5)
        started = time.monotonic()
        tf._politeness_gate()
        assert time.monotonic() - started < 0.1

    def test_session_reset_clears_the_pacing_clock(self):
        # Otherwise a fresh session's (or test's) first fetch waits out the last one's
        # interval for no reason. The reset is owned by the provider module, which clears
        # the fetch layer's clock along with its own caches.
        tf._politeness_gate()
        ts._reset_session_caches()
        started = time.monotonic()
        tf._politeness_gate()
        assert time.monotonic() - started < 0.1


# Fetch layer (real fetch_series over a faked _http_get).
class TestFetchLayer:
    def test_non_revising_hits_fredgraph_not_alfred(self, monkeypatch):
        fake = FakeHttp({FRED_CSV_URL: _csv("DGS10", [("2026-06-01", "4.20"), ("2026-06-02", "4.25")])})
        monkeypatch.setattr(tf, "_http_get", fake)

        series = fetch_series(SeriesSpec(source="fred", series_id="DGS10"), date(2026, 6, 30))

        assert len(series) == 2
        assert float(series.iloc[-1]) == pytest.approx(4.25)
        ((url, params),) = fake.calls
        assert url == FRED_CSV_URL  # fredgraph, not alfredgraph
        assert "vintage_date" not in params  # no vintage on a non-revising fetch

    def test_revising_hits_alfredgraph_with_vintage(self, monkeypatch):
        # ALFRED value column carries a vintage suffix; the parser matches by prefix.
        fake = FakeHttp({ALFRED_CSV_URL: _csv("CPIAUCSL_20260630", [("2026-05-01", "283.1"), ("2026-06-01", "283.9")])})
        monkeypatch.setattr(tf, "_http_get", fake)

        series = fetch_series(SeriesSpec(source="fred", series_id="CPIAUCSL", revises=True), date(2026, 6, 30))

        assert float(series.iloc[-1]) == pytest.approx(283.9)
        ((url, params),) = fake.calls
        assert url == ALFRED_CSV_URL
        # vintage defaults to the ceiling for a revising series with no explicit vintage.
        assert params["vintage_date"] == "2026-06-30"

    def test_missing_values_dropped(self, monkeypatch):
        rows = [("2026-06-01", "4.20"), ("2026-06-02", "."), ("2026-06-03", "4.30")]
        monkeypatch.setattr(tf, "_http_get", FakeHttp({FRED_CSV_URL: _csv("DGS10", rows)}))

        series = fetch_series(SeriesSpec(source="fred", series_id="DGS10"), date(2026, 6, 30))

        assert len(series) == 2  # the "." row is dropped, no interior NaN
        assert not series.isna().any()

    def test_malformed_html_body_raises_fetch_error(self, monkeypatch):
        monkeypatch.setattr(tf, "_http_get", FakeHttp({FRED_CSV_URL: b"<!DOCTYPE html><html>bad series id</html>"}))
        with pytest.raises(FetchError):
            fetch_series(SeriesSpec(source="fred", series_id="DGS10"), date(2026, 6, 30))

    def test_post_ceiling_row_raises_leakage_error(self, monkeypatch):
        # A row dated after the ceiling means the endpoint ignored the coed bound.
        rows = [("2026-06-01", "4.20"), ("2026-07-15", "4.30")]
        monkeypatch.setattr(tf, "_http_get", FakeHttp({FRED_CSV_URL: _csv("DGS10", rows)}))
        with pytest.raises(LeakageError):
            fetch_series(SeriesSpec(source="fred", series_id="DGS10"), date(2026, 6, 30))

    def test_cache_reuse_avoids_second_http_call(self, monkeypatch):
        fake = FakeHttp({FRED_CSV_URL: _csv("DGS10", [("2026-06-01", "4.20"), ("2026-06-02", "4.25")])})
        monkeypatch.setattr(tf, "_http_get", fake)

        first = fetch_series(SeriesSpec(source="fred", series_id="DGS10"), date(2026, 6, 30))
        second = fetch_series(SeriesSpec(source="fred", series_id="DGS10"), date(2026, 6, 30))

        assert len(fake.calls) == 1  # second call served from the in-memory cache
        pd.testing.assert_series_equal(first, second)


# yfinance fetch path (real fetch_series over a faked yfinance.Ticker).


def _yf_ohlc(dates: list[str], *, close: list[float], high: list[float]) -> pd.DataFrame:
    """Canned yfinance history frame: tz-aware DatetimeIndex + full OHLCV columns,
    mirroring what ``yfinance.Ticker(...).history()`` returns."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize("America/New_York")
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": close, "Close": close, "Volume": [0] * len(dates)},
        index=idx,
    )


def _fake_yf_ticker(frame: pd.DataFrame) -> tuple[type, list[dict[str, str]]]:
    """Return a (Ticker-class, calls-list) pair; the class records every history() kwargs."""
    calls: list[dict[str, str]] = []

    class _Ticker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, **kwargs: str) -> pd.DataFrame:
            calls.append(kwargs)
            return frame

    return _Ticker, calls


class TestYfinanceFetch:
    def test_high_column_spec_reads_high(self, monkeypatch):
        frame = _yf_ohlc(["2026-06-29", "2026-06-30"], close=[18.0, 19.0], high=[20.0, 22.0])
        ticker, _ = _fake_yf_ticker(frame)
        monkeypatch.setattr("yfinance.Ticker", ticker)

        series = fetch_series(SeriesSpec(source="yfinance", series_id="^VIX", column="High"), date(2026, 6, 30))

        assert float(series.iloc[-1]) == pytest.approx(22.0)  # High, not Close
        assert float(series.iloc[0]) == pytest.approx(20.0)

    def test_default_spec_reads_close(self, monkeypatch):
        frame = _yf_ohlc(["2026-06-29", "2026-06-30"], close=[18.0, 19.0], high=[20.0, 22.0])
        ticker, _ = _fake_yf_ticker(frame)
        monkeypatch.setattr("yfinance.Ticker", ticker)

        series = fetch_series(SeriesSpec(source="yfinance", series_id="^VIX"), date(2026, 6, 30))

        assert float(series.iloc[-1]) == pytest.approx(19.0)  # Close (default column)

    def test_empty_frame_raises_fetch_error(self, monkeypatch):
        ticker, _ = _fake_yf_ticker(pd.DataFrame())
        monkeypatch.setattr("yfinance.Ticker", ticker)

        with pytest.raises(FetchError, match="empty history"):
            fetch_series(SeriesSpec(source="yfinance", series_id="^VIX"), date(2026, 6, 30))

    def test_missing_requested_column_raises_fetch_error(self, monkeypatch):
        # A frame with no High column, but the spec asks for High -> FetchError.
        frame = _yf_ohlc(["2026-06-30"], close=[19.0], high=[22.0]).drop(columns=["High"])
        ticker, _ = _fake_yf_ticker(frame)
        monkeypatch.setattr("yfinance.Ticker", ticker)

        with pytest.raises(FetchError, match="no 'High' column"):
            fetch_series(SeriesSpec(source="yfinance", series_id="^VIX", column="High"), date(2026, 6, 30))

    def test_ceiling_respected_end_to_end(self, monkeypatch):
        # yfinance end is EXCLUSIVE, so the fetch must request end = ceiling + 1 day,
        # and the returned series must carry no observation after the ceiling.
        frame = _yf_ohlc(["2026-06-28", "2026-06-29", "2026-06-30"], close=[17.0, 18.0, 19.0], high=[19.0, 20.0, 22.0])
        ticker, calls = _fake_yf_ticker(frame)
        monkeypatch.setattr("yfinance.Ticker", ticker)

        ceiling = date(2026, 6, 30)
        series = fetch_series(SeriesSpec(source="yfinance", series_id="^VIX"), ceiling)

        assert calls[0]["end"] == date(2026, 7, 1).isoformat()  # ceiling + 1 (exclusive end)
        assert series.index.max().date() <= ceiling  # leakage guard held on the yfinance path
