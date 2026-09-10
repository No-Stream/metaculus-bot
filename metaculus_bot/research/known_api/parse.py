"""Stdlib-only URL parsers for the known-API registry.

The one place a FRED, Yahoo Finance or Kalshi URL is read into an identifier. Kept
dependency-free so the two extraction seams that must agree with it -- the financial-data
provider's id extraction and the resolution-source fetcher's Yahoo skip -- can import it
without dragging in fredapi, yfinance or aiohttp, and so ``translate`` and those seams
cannot drift on what counts as one of these hosts. Detail: docs/research.md
"Known-API registry".

Markdown-escape and percent-decode handling mirrors ``resolution_source.strip_markdown_escapes``:
a criteria URL Metaculus rendered as ``%5ETYX`` or ``DGS10\\_`` must read the same id the
live page would.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, unquote, urlparse

# The set of markdown escapes Metaculus injects into rendered URLs; undone before matching.
_MARKDOWN_ESCAPE_RE = re.compile(r"\\([_&.\-#()])")

# FRED series ids are alphanumeric + underscore; Yahoo symbols add ^ = . -
_FRED_ID = r"[A-Za-z0-9_]+"
_YAHOO_SYMBOL = r"[A-Za-z0-9.^=\-]+"

_FRED_HOST = "fred.stlouisfed.org"
_ALFRED_HOST = "alfred.stlouisfed.org"
_FRED_API_HOST = "api.stlouisfed.org"
_YAHOO_CHART_HOST = "query1.finance.yahoo.com"

# Yahoo Finance quote host: the bare host or a two-letter regional prefix (uk., ca.).
_YAHOO_QUOTE_HOST_RE = re.compile(r"\A(?:[a-z]{2}\.)?finance\.yahoo\.com\Z", re.IGNORECASE)

# `/series/{id}` and `/data/{id}` FRED page paths.
_FRED_PAGE_PATH_RE = re.compile(rf"/(?:series|data)/({_FRED_ID})")
# `query1.finance.yahoo.com/v8/finance/chart/{symbol}` data endpoint path.
_YAHOO_CHART_PATH_RE = re.compile(rf"/v8/finance/chart/({_YAHOO_SYMBOL})")
# `/quote/{symbol}` Yahoo page path (regional or bare host).
_YAHOO_QUOTE_PATH_RE = re.compile(rf"/quote/({_YAHOO_SYMBOL})")
# The path segments after /markets/ or /api/v#/markets/; the last is the market ticker.
_KALSHI_MARKETS_RE = re.compile(r"/(?:api/v\d+/)?markets/(.+)")
# At most two FRED series per URL: the one archived multi-id URL is a NOB spread, served as two.
_MAX_FRED_IDS_PER_URL = 2


def strip_markdown_escapes(text: str) -> str:
    """Remove the markdown backslash escapes Metaculus injects into rendered URLs."""
    return _MARKDOWN_ESCAPE_RE.sub(r"\1", text)


def normalize(text: str) -> str:
    """Percent-decode then unescape, so ``%5ETYX`` and ``DGS10\\_`` read as the live id would."""
    return strip_markdown_escapes(unquote(text))


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


def is_fred_url(url: str) -> bool:
    """A ``fred.stlouisfed.org`` URL: the financial-data provider already serves these."""
    return _host(url) == _FRED_HOST


def is_yahoo_ticker_url(url: str) -> bool:
    """A Yahoo Finance ``/quote/...`` page, regional hosts (``uk.``, ``ca.``) included.

    The financial-data provider renders these from the cited symbol, so the resolution-source
    fetcher skips them; the regional hosts are here because a ``uk.finance.yahoo.com`` criteria
    URL was both rendered by the provider and attempted as a page by the fetcher. Generic Yahoo
    news/help URLs stay fetchable.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return bool(_YAHOO_QUOTE_HOST_RE.match(host)) and parsed.path.startswith("/quote/")


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def fred_series_ids(text: str) -> list[str]:
    """Every FRED series id cited in ``text``, order-preserving and deduped.

    Reads the page paths (``/series/{id}``, ``/data/{id}``), the graph exports
    (``fredgraph.csv``/``.xls?id=A,B`` -- up to two ids -- and ``alfredgraph.csv?id=``), the
    ALFRED series page (``?seid=``) and the API observations endpoint (``?series_id=``).
    """
    decoded = normalize(text)
    ids: list[str] = []
    for match in re.finditer(r"https?://[^\s)>\]]+", decoded):
        url = match.group(0)
        host = _host(url)
        if host not in (_FRED_HOST, _ALFRED_HOST, _FRED_API_HOST):
            continue
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        if "id" in query:
            # Cap the id list per URL, so one graph URL cannot expand into an unbounded fetch set.
            per_url = [part for value in query["id"] for part in value.split(",") if part]
            ids.extend(per_url[:_MAX_FRED_IDS_PER_URL])
        elif "seid" in query:
            ids.extend(query["seid"])
        elif "series_id" in query:
            ids.extend(query["series_id"])
        else:
            page = _FRED_PAGE_PATH_RE.search(parsed.path)
            if page:
                ids.append(page.group(1))
    return _dedupe(ids)


def yahoo_symbols(text: str) -> list[str]:
    """Every Yahoo Finance symbol cited in ``text``, order-preserving and deduped.

    Reads the ``/quote/{symbol}`` pages (regional hosts included) and the
    ``query1.finance.yahoo.com/v8/finance/chart/{symbol}`` data endpoint. A trailing dot picked
    up from sentence-final punctuation is stripped; internal dots (``DX-Y.NYB``) stay.
    """
    decoded = normalize(text)
    symbols: list[str] = []
    for match in re.finditer(r"https?://[^\s)>\]]+", decoded):
        url = match.group(0)
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if host == _YAHOO_CHART_HOST:
            chart = _YAHOO_CHART_PATH_RE.search(parsed.path)
            if chart:
                symbols.append(chart.group(1))
        elif _YAHOO_QUOTE_HOST_RE.match(host):
            quote = _YAHOO_QUOTE_PATH_RE.search(parsed.path)
            if quote:
                symbols.append(quote.group(1))
    return _dedupe([symbol.rstrip(".") for symbol in symbols])


def kalshi_ticker(url: str) -> str | None:
    """The market ticker of a Kalshi market URL, upper-cased, or None.

    Reads ``/markets/{ticker}``, ``/markets/{series}/{slug}/{ticker}`` and the API path
    ``/api/v#/markets/{ticker}``. On a multi-segment path the last segment is the ticker only when
    it starts with the series (first) segment; otherwise the URL points at a series/slug page with
    no market ticker (``/markets/kxu3/unemployment``), which returns None so the fetcher does not
    burn detail GETs on a 404. Only ``kalshi.com`` market URLs qualify.
    """
    if _host(url) != "kalshi.com":
        return None
    match = _KALSHI_MARKETS_RE.search(urlparse(url).path)
    if not match:
        return None
    segments = [segment for segment in match.group(1).split("/") if segment]
    if not segments:
        return None
    if len(segments) == 1:
        return segments[0].upper()
    ticker = segments[-1].upper()
    return ticker if ticker.startswith(segments[0].upper()) else None
