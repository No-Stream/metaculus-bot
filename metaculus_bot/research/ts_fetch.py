"""Point-in-time, leakage-proof time-series fetcher for the time-series-anchor provider.

Returns a univariate float series with NO observation after a given ceiling date. For
revised macro series (CPI, payrolls, GDP) a plain date ceiling still leaks — today's
FRED data contains REVISED historical values not known at forecast time — so those go
through ALFRED point-in-time vintages instead.

Adapted from ``scratch/ts_anchor_replay_2026-07-16/fetch.py`` for the prod provider:

- The disk cache is replaced by an in-memory per-session dict keyed on
  ``(source, series_id, column, ceiling, vintage, lookback_years)``; reruns within a
  session are network-free and deterministic. ``_reset_series_cache()`` clears it
  (tests + session start).
- ``fetch_series`` stays synchronous (all HTTP is blocking ``requests`` / ``yfinance``);
  the async provider wraps each call in ``asyncio.to_thread`` — the same pattern
  ``financial_data.py`` uses.
- Browser headers are reused from the shared ``http_fetch`` module so the User-Agent /
  Accept-Language / Accept-Encoding set stays in one place.

Endpoint facts (live-verified 2026-07-16):

- FRED keyless CSV ``fredgraph.csv?id=&cosd=&coed=`` returns ``observation_date,<ID>``
  rows; cosd/coed are inclusive. It silently IGNORES ``vintage_date`` — never use it
  for vintage-sensitive fetches.
- ALFRED keyless CSV ``alfredgraph.csv?...&vintage_date=YYYY-MM-DD`` honors vintages;
  the value column header carries a vintage suffix (``CPIAUCSL_20250601``).
- Bad series ids return an HTML page (observed HTTP 404, but historically 200) —
  validate the body starts with ``observation_date,`` rather than trusting status alone.
- A vintage predating the series' first vintage → HTTP 404 with an empty body.
- Missing values are encoded as ``"."`` — dropped here so interior NaNs never reach
  the band estimators.
- yfinance 1.x: ``history(end=...)`` is EXCLUSIVE (we pass ceiling + 1 day); data does
  not revise, so a plain ceiling suffices. auto_adjust defaults to True (adjusted Close).
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import pandas as pd
import requests

from metaculus_bot.constants import TS_ANCHOR_HTTP_TIMEOUT
from metaculus_bot.research.http_fetch import BROWSER_HEADERS

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ALFRED_CSV_URL = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
CSV_HEADER_PREFIX = "observation_date,"
# Per-request HTTP timeout for one FRED/ALFRED/yfinance fetch. Single source of
# truth lives in constants (TS_ANCHOR_HTTP_TIMEOUT); mirrored here as the module
# default so the fetch seam has no magic literal.
HTTP_TIMEOUT_S = TS_ANCHOR_HTTP_TIMEOUT
POLITENESS_SLEEP_S = 0.5


class FetchError(Exception):
    """Bad series id, empty/malformed response, or transport failure. Fail fast."""


class LeakageError(FetchError):
    """An observation dated after the ceiling reached the output — the one invariant
    everything downstream trusts. Should be unreachable; if it fires, the source
    endpoint changed behavior and the fetch path must not be trusted."""


@dataclass(frozen=True)
class SeriesSpec:
    source: Literal["fred", "yfinance"]
    series_id: str
    revises: bool = False  # fred only; True → ALFRED vintage fetch
    column: Literal["Close", "High", "Low", "Open"] = "Close"  # yfinance only; "High" for intraday-max Qs

    def __post_init__(self) -> None:
        if self.source == "yfinance" and self.revises:
            raise ValueError(f"revises=True is only meaningful for fred series, got {self!r}")
        if self.source == "fred" and self.column != "Close":
            raise ValueError(f"column={self.column!r} is only meaningful for yfinance series, got {self!r}")


# In-memory per-session cache: parsed series keyed on the full fetch identity.
_SERIES_CACHE: dict[tuple[str, str, str, str, str, int], pd.Series] = {}


def _reset_series_cache() -> None:
    """Clear the in-memory series cache (tests + session start)."""
    _SERIES_CACHE.clear()


def _cache_key(spec: SeriesSpec, ceiling: date, vintage: date | None, lookback_years: int) -> tuple:
    return (
        spec.source,
        spec.series_id,
        spec.column,
        ceiling.isoformat(),
        vintage.isoformat() if vintage else "none",
        lookback_years,
    )


def fetch_series(
    spec: SeriesSpec,
    ceiling: date,
    *,
    vintage: date | None = None,
    lookback_years: int = 15,
) -> pd.Series:
    """Float series indexed by naive dates; guaranteed ``max(index) <= ceiling``.

    For ``spec.revises`` fred series the fetch goes through ALFRED with
    ``vintage_date=vintage``; when ``vintage`` is omitted it defaults to ``ceiling``
    (point-in-time as-of the ceiling). Passing a vintage for a non-revising spec is a
    caller error and raises ``ValueError``.

    Cached in-memory keyed on (source, series_id, column, ceiling, vintage, lookback),
    so reruns within a session are network-free and deterministic. Raises ``FetchError``
    on bad id / empty / malformed response — never returns a silent empty series.
    """
    if spec.source == "fred":
        if spec.revises:
            if vintage is None:
                vintage = ceiling
                logger.info(f"{spec.series_id}: revising series with no vintage given; defaulting vintage={ceiling}")
        elif vintage is not None:
            raise ValueError(f"vintage={vintage} passed for non-revising spec {spec!r}; drop it or set revises=True")
    elif vintage is not None:
        raise ValueError(f"vintage={vintage} is not applicable to yfinance spec {spec!r}")

    key = _cache_key(spec, ceiling, vintage, lookback_years)
    cached = _SERIES_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    start = ceiling - timedelta(days=round(lookback_years * 365.25))
    raw = _fetch_raw(spec, start, ceiling, vintage)
    series = _parse_csv(raw, spec)
    _assert_no_leakage(series, ceiling, spec)
    _SERIES_CACHE[key] = series
    return series.copy()


# ---------------------------------------------------------------------------
# Fetch (uncached path)
# ---------------------------------------------------------------------------


def _fetch_raw(spec: SeriesSpec, start: date, ceiling: date, vintage: date | None) -> bytes:
    """Fetch and validate raw CSV bytes for the spec. Both sources are normalized to
    ``observation_date,<ID>`` CSV so one parser serves the FRED and yfinance paths."""
    if spec.source == "fred":
        return _fetch_fred_csv(spec, start, ceiling, vintage)
    if spec.source == "yfinance":
        return _fetch_yfinance_csv(spec, start, ceiling)
    raise FetchError(f"unknown source {spec.source!r}")  # unreachable via Literal, guards raw dict construction


def _yf_price_column(frame: pd.DataFrame, spec: SeriesSpec) -> pd.Series:
    """Pick the requested OHLC column (default Close) from a yfinance history frame.

    ``max_window`` intraday-max questions (^VIX, CL=F, SI=F) resolve on the daily
    *High*, not the Close; ``spec.column`` selects it. yfinance ``auto_adjust=True``
    adjusts every OHLC column consistently, so High stays comparable to the adjusted
    Close used elsewhere."""
    if spec.column not in frame.columns:
        raise FetchError(f"{spec.series_id}: yfinance history has no {spec.column!r} column ({list(frame.columns)})")
    return frame[spec.column].dropna()


def _fetch_fred_csv(spec: SeriesSpec, start: date, ceiling: date, vintage: date | None) -> bytes:
    params = {"id": spec.series_id, "cosd": start.isoformat(), "coed": ceiling.isoformat()}
    if vintage is not None:
        url = ALFRED_CSV_URL
        params["vintage_date"] = vintage.isoformat()
    else:
        url = FRED_CSV_URL

    body = _http_get(url, params)
    text_head = body[:200].decode("utf-8", errors="replace")  # HARNESS-SCAN-EXEMPT-subsampling: header sniff, not data
    if not text_head.startswith(CSV_HEADER_PREFIX):
        raise FetchError(
            f"{spec.series_id}: response is not observation CSV (bad series id, or vintage "
            f"{vintage} predates the first vintage). Body starts: {text_head[:80]!r}"  # HARNESS-SCAN-EXEMPT-subsampling: error-message truncation
        )
    return body


def _fetch_yfinance_csv(spec: SeriesSpec, start: date, ceiling: date) -> bytes:
    import yfinance  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import  # optional-dep pattern, matches financial_data.py
    from yfinance.exceptions import YFException  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

    try:
        # end is EXCLUSIVE in yfinance 1.x → ceiling + 1 day makes the ceiling inclusive.
        frame = yfinance.Ticker(spec.series_id).history(
            start=start.isoformat(), end=(ceiling + timedelta(days=1)).isoformat()
        )
    except YFException as exc:
        raise FetchError(f"{spec.series_id}: yfinance fetch failed: {exc}") from exc

    if frame.empty:
        raise FetchError(f"{spec.series_id}: yfinance returned empty history (bad or delisted ticker?)")

    price = _yf_price_column(frame, spec)
    if price.empty:
        raise FetchError(f"{spec.series_id}: yfinance history has no usable {spec.column} values")
    price.index = pd.DatetimeIndex(price.index).tz_localize(None).normalize()

    buf = io.StringIO()
    price.rename(spec.series_id).rename_axis("observation_date").to_csv(buf, date_format="%Y-%m-%d")
    return buf.getvalue().encode("utf-8")


def _http_get(url: str, params: dict[str, str]) -> bytes:
    """Single HTTP seam: politeness sleep + browser UA + status check. Tests mock this."""
    time.sleep(POLITENESS_SLEEP_S)
    try:
        response = requests.get(url, params=params, headers=BROWSER_HEADERS, timeout=HTTP_TIMEOUT_S)
    except requests.RequestException as exc:
        raise FetchError(f"HTTP request failed for {url} params={params}: {exc}") from exc
    if response.status_code != 200:
        raise FetchError(
            f"HTTP {response.status_code} for {url} params={params} "
            f"(bad series id, or vintage predating the first vintage)"
        )
    return response.content


# ---------------------------------------------------------------------------
# Parse + invariants
# ---------------------------------------------------------------------------


def _parse_csv(raw: bytes, spec: SeriesSpec) -> pd.Series:
    """Parse ``observation_date,<ID>[ _YYYYMMDD]`` CSV bytes into a float series.

    FRED encodes missing values as "." — those rows are dropped so interior NaNs never
    reach the band estimators. ALFRED value columns carry a vintage suffix
    (``CPIAUCSL_20250601``); the column is matched by prefix, not equality.
    """
    text_head = raw[:200].decode("utf-8", errors="replace")  # HARNESS-SCAN-EXEMPT-subsampling: header sniff, not data
    if not text_head.startswith(CSV_HEADER_PREFIX):
        raise FetchError(
            f"{spec.series_id}: malformed CSV, header starts: {text_head[:80]!r}"  # HARNESS-SCAN-EXEMPT-subsampling: error-message truncation
        )

    try:
        frame = pd.read_csv(io.BytesIO(raw), na_values=["."])
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise FetchError(f"{spec.series_id}: CSV parse failed: {exc}") from exc

    if frame.shape[1] != 2:
        raise FetchError(f"{spec.series_id}: expected 2 CSV columns, got {list(frame.columns)}")
    value_col = str(frame.columns[1])
    if not value_col.upper().startswith(spec.series_id.upper()):
        raise FetchError(f"{spec.series_id}: unexpected value column {value_col!r}")

    frame = frame.dropna(subset=[value_col])
    if frame.empty:
        raise FetchError(
            f"{spec.series_id}: series is empty after dropping missing values "
            f"(window has no data, or vintage predates the first release)"
        )

    try:
        values = pd.to_numeric(frame[value_col], errors="raise").astype("float64")
        index = pd.DatetimeIndex(pd.to_datetime(frame["observation_date"], format="%Y-%m-%d"))
    except (ValueError, TypeError) as exc:
        raise FetchError(f"{spec.series_id}: non-numeric values or malformed dates in CSV: {exc}") from exc

    series = pd.Series(values.to_numpy(), index=index, name=spec.series_id)
    return series.sort_index()


def _assert_no_leakage(series: pd.Series, ceiling: date, spec: SeriesSpec) -> None:
    """Belt-and-suspenders invariant: no observation may postdate the ceiling."""
    last = series.index.max()
    if last.date() > ceiling:
        raise LeakageError(
            f"{spec.series_id}: observation at {last.date()} postdates ceiling {ceiling} — "
            f"source endpoint ignored the date bound; do not trust this fetch path"
        )
