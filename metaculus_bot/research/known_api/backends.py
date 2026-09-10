"""The known-API backends: FRED, Yahoo Finance, prediction markets and SEC EDGAR.

Each backend takes a resolved identifier (a FRED id, a Yahoo symbol, a venue + market, or an
EDGAR call), reads it through the same client code the research providers use, and returns a
neutral :class:`KnownApiResult`. No LLM, no paid key. A backend NEVER raises to its caller:
an unknown id is ``not_found`` with the provider's message, an empty window is ``empty``, a
transport or quota failure is ``error`` naming the exception class. Detail: docs/research.md
"Known-API registry".

The bounds are the ones the 2026-09-09 cost pass measured a need for: a windowed read capped at
400 observations newest-kept, 15 s per FRED or Yahoo call (fredapi's ``urlopen`` carries no
timeout of its own, so the whole worker is bounded by :func:`asyncio.wait_for`), five market rows,
and at most four Kalshi detail GETs per question (a semaphore the caller constructs per question,
since the loop has no per-question object yet -- see :func:`market_snapshot`).
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any, cast
from xml.etree.ElementTree import ParseError

import pandas as pd

from metaculus_bot.constants import FRED_API_KEY_ENV
from metaculus_bot.research import fred_rendering, ts_fetch
from metaculus_bot.research.known_api.result import KnownApiResult
from metaculus_bot.research.number_format import format_decimal_change, format_decimal_value

logger = logging.getLogger(__name__)

# fredapi's urlopen has no timeout of its own, so asyncio.wait_for bounds the await, not the worker.
FRED_YAHOO_CALL_TIMEOUT_S = 15.0
# The default read is the recent window; the driver widens it with start/end when it needs history.
DEFAULT_OBSERVATIONS = 30
# The hard cap: a windowed read never renders more than this, newest kept.
MAX_OBSERVATIONS = 400
# FRED free-text search returns at most this many candidate series.
FRED_SEARCH_LIMIT = 5


def _today() -> date:
    return datetime.now(tz=UTC).date()


def _window(series: pd.Series, *, start: date | None, end: date | None) -> pd.Series:
    """Slice ``series`` to ``[start, end]`` then to the recent/cap bound, newest observations kept.

    With no ``start`` the default is the last :data:`DEFAULT_OBSERVATIONS`; an explicit window is
    honoured and then capped at :data:`MAX_OBSERVATIONS`, so a wide range can never render an
    unbounded table.
    """
    clean = series.dropna().sort_index()
    if start is not None:
        clean = clean[clean.index >= pd.Timestamp(start)]
    if end is not None:
        clean = clean[clean.index <= pd.Timestamp(end)]
    limit = MAX_OBSERVATIONS if start is not None else min(DEFAULT_OBSERVATIONS, MAX_OBSERVATIONS)
    return clean.tail(limit)


def _summary_lines(series: pd.Series) -> list[str]:
    latest = float(series.iloc[-1])
    latest_date = pd.Timestamp(series.index[-1]).strftime("%Y-%m-%d")
    lines = [f"- Latest: {format_decimal_value(latest)} ({latest_date})"]
    if len(series) >= 2:
        previous = float(series.iloc[-2])
        change = latest - previous
        lines.append(f"- Previous: {format_decimal_value(previous)} (change {format_decimal_change(change)})")
    return lines


def _observation_lines(series: pd.Series, *, total: int) -> list[str]:
    """Newest-first observation rows, with a caption when the window was capped."""
    newest_first = series.iloc[::-1]
    caption = f"- Observations (newest first, {len(series)} of {total}"
    caption += f"; capped at {MAX_OBSERVATIONS})" if total > MAX_OBSERVATIONS else ")"
    rows = [
        f"  - {cast(pd.Timestamp, ts).strftime('%Y-%m-%d')}: {format_decimal_value(float(value))}"
        for ts, value in newest_first.items()
    ]
    return [caption + "\n" + "\n".join(rows)]


def _render_series_block(*, header: str, source_url: str, series: pd.Series, total: int) -> str:
    parts = [header, f"Source: {source_url}", *_summary_lines(series), *_observation_lines(series, total=total)]
    return "\n".join(parts)


async def _bounded(sync_call: Callable[[], KnownApiResult], *, label: str) -> KnownApiResult:
    """Run a blocking backend in a thread, bounded by :data:`FRED_YAHOO_CALL_TIMEOUT_S`."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(sync_call), timeout=FRED_YAHOO_CALL_TIMEOUT_S)
    except TimeoutError:
        return KnownApiResult(
            status="error", content_markdown=f"{label} timed out after {FRED_YAHOO_CALL_TIMEOUT_S}s.", source_url=""
        )


def _column(frame: pd.DataFrame, name: str) -> list[Any]:
    """A search column's values, or blanks when FRED omitted it, aligned to the frame's rows."""
    return frame[name].tolist() if name in frame.columns else [""] * len(frame)


def _fred_search_sync(text: str, api_key: str) -> KnownApiResult:
    fred = fred_rendering.Fred(api_key=api_key)
    # Fred.search caps at FRED_SEARCH_LIMIT, so the whole frame renders; columns zip, no row loop.
    hits = fred.search(text, limit=FRED_SEARCH_LIMIT)
    if not isinstance(hits, pd.DataFrame) or hits.empty:
        return KnownApiResult(
            status="empty", content_markdown=f"FRED search for {text!r} matched no series.", source_url=""
        )
    zipped = zip(
        list(hits.index), _column(hits, "title"), _column(hits, "frequency"), _column(hits, "units"), strict=True
    )
    rows = [f"FRED series matching {text!r}:"]
    rows += [f"- {series_id}: {title} ({frequency}; {units})" for series_id, title, frequency, units in zipped]
    return KnownApiResult(status="ok", content_markdown="\n".join(rows), source_url="https://fred.stlouisfed.org/")


def _fred_series_keyed_sync(
    series_id: str, api_key: str, *, start: date | None, end: date | None, first_release: bool
) -> KnownApiResult:
    source_url = f"https://fred.stlouisfed.org/series/{series_id}"
    fred = fred_rendering.Fred(api_key=api_key)
    # A bad id raises ValueError("... does not exist"); fred_series's runner classifies it not_found.
    data = fred.get_series(series_id, observation_start=start, observation_end=end)
    return _render_fred(series_id, data, source_url, fred=fred, start=start, end=end, first_release=first_release)


def _fred_series_keyless_sync(series_id: str, *, start: date | None, end: date | None) -> KnownApiResult:
    source_url = f"https://fred.stlouisfed.org/series/{series_id}"
    revises = series_id.upper() not in ts_fetch.FRED_NON_REVISING_SERIES
    spec = ts_fetch.SeriesSpec(source="fred", series_id=series_id, revises=revises)
    data = ts_fetch.fetch_series(spec, end or _today())
    return _render_fred(series_id, data, source_url, fred=None, start=start, end=end, first_release=False)


def _render_fred(
    series_id: str,
    data: pd.Series,
    source_url: str,
    *,
    fred: Any,
    start: date | None,
    end: date | None,
    first_release: bool,
) -> KnownApiResult:
    total = int(data.dropna().shape[0])
    windowed = _window(data, start=start, end=end)
    if windowed.empty:
        return KnownApiResult(
            status="empty",
            content_markdown=f"FRED series {series_id!r} has no observations in the window.",
            source_url=source_url,
        )
    title = _fred_title(fred, series_id)
    header = f"### {series_id} ({title})" + (" [first release]" if first_release else "")
    block = _render_series_block(header=header, source_url=source_url, series=windowed, total=total)
    if first_release and fred is not None:
        block += _fred_first_release_note(fred, series_id, windowed)
    return KnownApiResult(status="ok", content_markdown=block, source_url=source_url, links=[source_url])


def _fred_title(fred: Any, series_id: str) -> str:
    if fred is None:
        return series_id
    try:
        info = fred.get_series_info(series_id)
    except (ValueError, OSError, ParseError):  # title is cosmetic enrichment; fall back to the id
        return series_id
    if isinstance(info, pd.DataFrame) and "title" in info.columns:
        return str(info["title"].iloc[0])
    if isinstance(info, pd.Series) and "title" in info.index:
        return str(info["title"])
    return series_id


def _fred_first_release_note(fred: Any, series_id: str, windowed: pd.Series) -> str:
    observation_start = windowed.index[0]
    first_releases = fred_rendering._fetch_fred_first_releases(fred, series_id, observation_start)
    if first_releases is None or first_releases.empty:
        return ""
    return "\n" + "\n".join(fred_rendering._first_release_lines(windowed, first_releases))


async def fred_series(
    *,
    series_id: str | None = None,
    search: str | None = None,
    start: date | None = None,
    end: date | None = None,
    first_release: bool = False,
) -> KnownApiResult:
    """One FRED series over a date window, or a free-text search over FRED's catalogue.

    ``search`` runs ``Fred.search`` (five hits: id, title, frequency, units). ``series_id`` reads
    the observations through the keyed API when ``FRED_API_KEY`` is set (with the initial-release
    table when ``first_release``), and through the keyless ``fredgraph`` CSV otherwise. One series
    per call; the window is capped at :data:`MAX_OBSERVATIONS`.
    """
    api_key = os.getenv(FRED_API_KEY_ENV)
    source_url = f"https://fred.stlouisfed.org/series/{series_id}" if series_id else ""

    def _run() -> KnownApiResult:
        try:
            if search:
                if not api_key:
                    return KnownApiResult(
                        status="error",
                        content_markdown=f"FRED search needs {FRED_API_KEY_ENV}, which is unset.",
                        source_url="",
                    )
                return _fred_search_sync(search, api_key)
            if not series_id:
                return KnownApiResult(
                    status="error", content_markdown="fred_series needs a series_id or a search string.", source_url=""
                )
            if api_key:
                return _fred_series_keyed_sync(series_id, api_key, start=start, end=end, first_release=first_release)
            return _fred_series_keyless_sync(series_id, start=start, end=end)
        except ts_fetch.FetchError as exc:
            return KnownApiResult(
                status="not_found", content_markdown=f"FRED {series_id!r}: {exc}", source_url=source_url
            )
        except ValueError as exc:
            if fred_rendering.is_unknown_fred_series_error(exc):
                return KnownApiResult(
                    status="not_found",
                    content_markdown=f"FRED has no series {series_id!r}: {exc}",
                    source_url=source_url,
                )
            return KnownApiResult(
                status="error", content_markdown=f"FRED fetch failed (ValueError): {exc}", source_url=source_url
            )
        except OSError as exc:
            return KnownApiResult(
                status="error",
                content_markdown=f"FRED fetch failed ({type(exc).__name__}): {exc}",
                source_url=source_url,
            )

    return await _bounded(_run, label=f"FRED {search or series_id}")


def _yahoo_history_sync(ticker: str, *, start: date | None, end: date | None, column: str) -> KnownApiResult:
    source_url = f"https://finance.yahoo.com/quote/{ticker}/history/"
    spec = ts_fetch.SeriesSpec(source="yfinance", series_id=ticker, column=column)  # type: ignore[arg-type]
    data = ts_fetch.fetch_series(spec, end or _today())
    total = int(data.dropna().shape[0])
    windowed = _window(data, start=start, end=end)
    if windowed.empty:
        return KnownApiResult(
            status="empty",
            content_markdown=f"Yahoo Finance {ticker!r} has no {column} data in the window.",
            source_url=source_url,
        )
    header = f"### {ticker} (Yahoo Finance {column})"
    block = _render_series_block(header=header, source_url=source_url, series=windowed, total=total)
    return KnownApiResult(status="ok", content_markdown=block, source_url=source_url, links=[source_url])


async def yahoo_history(
    *,
    ticker: str,
    start: date | None = None,
    end: date | None = None,
    column: str = "Close",
) -> KnownApiResult:
    """One Yahoo Finance symbol's adjusted price history over a date window.

    Reads through ``ts_fetch.fetch_series`` (the yfinance client the financial-data provider
    uses), ``column`` in Close/High/Low/Open. One ticker per call, window capped at
    :data:`MAX_OBSERVATIONS`; an empty history (a bad or delisted ticker) is ``not_found``.
    """
    source_url = f"https://finance.yahoo.com/quote/{ticker}/history/"

    def _run() -> KnownApiResult:
        try:
            return _yahoo_history_sync(ticker, start=start, end=end, column=column)
        except ts_fetch.FetchError as exc:
            return KnownApiResult(
                status="not_found", content_markdown=f"Yahoo Finance {ticker!r}: {exc}", source_url=source_url
            )
        except (OSError, ValueError) as exc:
            return KnownApiResult(
                status="error",
                content_markdown=f"Yahoo fetch failed ({type(exc).__name__}): {exc}",
                source_url=source_url,
            )

    return await _bounded(_run, label=f"Yahoo {ticker}")
