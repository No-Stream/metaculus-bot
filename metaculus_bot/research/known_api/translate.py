"""``translate(url) -> KnownApiCall | None``: one URL shape to one deterministic known-API call.

Pure and dependency-light: it reads a URL into the identifier, window and venue a backend needs,
and returns None for anything the registry does not own (which leaves the fetch ladder's later
rungs to try the page). It shares ``parse`` with the extraction seams so the two cannot disagree
about what counts as a FRED, Yahoo or Kalshi URL. Detail: docs/research.md "Known-API registry".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Literal
from urllib.parse import parse_qs, urlparse

from metaculus_bot.research.known_api import parse

# At most two FRED series per URL: the one archived multi-id URL is a NOB spread, served as two.
_MAX_FRED_IDS = 2

_FRED_HOSTS = frozenset({parse._FRED_HOST, parse._ALFRED_HOST, parse._FRED_API_HOST})
_ALFRED_HOST = parse._ALFRED_HOST
_EDGAR_HOSTS = frozenset({"www.sec.gov", "data.sec.gov", "efts.sec.gov"})

EdgarKind = Literal["filing_document", "company_submissions"]


@dataclass(frozen=True, slots=True)
class KnownApiCall:
    """A resolved known-API call: which host, which identifier, and any window or venue with it.

    One of ``kind``'s four families is populated at a time. ``canonical_url`` is the human page
    for the identifier, surfaced so provenance can cite the series/market rather than the raw
    data endpoint.
    """

    kind: Literal["fred", "yahoo", "kalshi", "edgar"]
    canonical_url: str = ""
    fred_series_ids: tuple[str, ...] = ()
    fred_first_release: bool = False
    yahoo_symbol: str | None = None
    window_start: date | None = None
    window_end: date | None = None
    kalshi_ticker: str | None = None
    edgar_kind: EdgarKind | None = None
    edgar_url: str | None = None
    edgar_arg: str | None = field(default=None)


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def _iso_or_none(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _epoch_or_none(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromtimestamp(int(value), tz=UTC).date()
    except (ValueError, OverflowError, OSError):
        return None


def _translate_fred(url: str) -> KnownApiCall | None:
    ids = parse.fred_series_ids(url)[:_MAX_FRED_IDS]
    if not ids:
        return None
    query = parse_qs(urlparse(url).query)
    start = _iso_or_none(_first(query, "cosd") or _first(query, "observation_start"))
    end = _iso_or_none(_first(query, "coed") or _first(query, "observation_end"))
    # ALFRED shapes read the initial release (output_type=4); fred.stlouisfed.org reads the vintage.
    first_release = _host(url) == _ALFRED_HOST
    return KnownApiCall(
        kind="fred",
        canonical_url=f"https://fred.stlouisfed.org/series/{ids[0]}",
        fred_series_ids=tuple(ids),
        fred_first_release=first_release,
        window_start=start,
        window_end=end,
    )


def _translate_yahoo(url: str) -> KnownApiCall | None:
    symbols = parse.yahoo_symbols(url)
    if not symbols:
        return None
    query = parse_qs(urlparse(url).query)
    exclusive_end = _epoch_or_none(_first(query, "period2"))
    return KnownApiCall(
        kind="yahoo",
        canonical_url=f"https://finance.yahoo.com/quote/{symbols[0]}/history/",
        yahoo_symbol=symbols[0],
        window_start=_epoch_or_none(_first(query, "period1")),
        # Yahoo's period2 excludes its date; backend windows use inclusive dates.
        window_end=exclusive_end - timedelta(days=1) if exclusive_end is not None else None,
    )


def _translate_kalshi(url: str) -> KnownApiCall | None:
    ticker = parse.kalshi_ticker(url)
    if ticker is None:
        return None
    return KnownApiCall(
        kind="kalshi",
        canonical_url=f"https://kalshi.com/markets/{ticker}",
        kalshi_ticker=ticker,
    )


def _translate_edgar(url: str) -> KnownApiCall | None:
    parsed = urlparse(url)
    if parsed.path.startswith("/Archives/edgar/data/"):
        return KnownApiCall(kind="edgar", canonical_url=url, edgar_kind="filing_document", edgar_url=url)
    if parsed.path.startswith("/cgi-bin/browse-edgar"):
        cik = _first(parse_qs(parsed.query), "CIK")
        if cik:
            return KnownApiCall(kind="edgar", canonical_url=url, edgar_kind="company_submissions", edgar_arg=cik)
    return None


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def translate(url: str) -> KnownApiCall | None:
    """Map one URL to a known-API call, or None when the registry does not own its host."""
    host = _host(url)
    if host in _FRED_HOSTS:
        return _translate_fred(url)
    if host == parse._YAHOO_CHART_HOST or parse._YAHOO_QUOTE_HOST_RE.match(host):
        return _translate_yahoo(url)
    if host == "kalshi.com":
        return _translate_kalshi(url)
    if host in _EDGAR_HOSTS:
        return _translate_edgar(url)
    return None
