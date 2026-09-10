"""URL parsing for the known-API registry: the shared predicates and id extractors.

Every observed FRED/Yahoo/Kalshi URL shape (the 2026-09-09 cost pass's
``url_shapes_results.json``) and the negative cases (Yahoo help pages, news
articles, the FRED release calendar) live here, so the one place URL parsing
happens cannot silently regress on a shape the driver really tried.
"""

from __future__ import annotations

import pytest

from metaculus_bot.research.known_api import parse


class TestIsFredUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://fred.stlouisfed.org/series/DGS30",
            "https://fred.stlouisfed.org/data/CSUSHPISA",
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CSUSHPISA",
            "https://fred.stlouisfed.org:443/series/DGS10",
        ],
    )
    def test_matches_fred_host(self, url: str):
        assert parse.is_fred_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://stlouisfed.org/other",
            "https://alfred.stlouisfed.org/series?seid=CSUSHPISA",
            "https://example.com/fred.stlouisfed.org",
        ],
    )
    def test_rejects_non_fred_host(self, url: str):
        assert parse.is_fred_url(url) is False


class TestIsYahooTickerUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://finance.yahoo.com/quote/AAPL",
            "https://finance.yahoo.com/quote/BTC-USD/history",
            "https://finance.yahoo.com/quote/SPCX/history/",
            "https://uk.finance.yahoo.com/quote/ETH-USD/history/",
            "https://ca.finance.yahoo.com/quote/SPCX/history/",
            "https://finance.yahoo.com:443/quote/AAPL",
            "https://finance.yahoo.com/quote/%5ETYX/history/?period1=1&period2=2",
        ],
    )
    def test_matches_quote_pages_including_regional(self, url: str):
        assert parse.is_yahoo_ticker_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "https://finance.yahoo.com/news/some-article",
            "https://help.yahoo.com/kb/SLN28256.html",
            "https://finance.yahoo.com/personal-finance/investing/article/x.html",
            "https://uk.finance.yahoo.com/news/politics",
            "https://notyahoo.com/quote/AAPL",
        ],
    )
    def test_rejects_non_quote_pages(self, url: str):
        assert parse.is_yahoo_ticker_url(url) is False


class TestFredSeriesIds:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("https://fred.stlouisfed.org/series/DGS30", ["DGS30"]),
            ("https://fred.stlouisfed.org/data/CSUSHPISA", ["CSUSHPISA"]),
            ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=CSUSHPISA", ["CSUSHPISA"]),
            (
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS30,DGS10&cos=Close&coed=2026-07-24",
                ["DGS30", "DGS10"],
            ),
            ("https://fred.stlouisfed.org/graph/fredgraph.xls?id=CSUSHPISA", ["CSUSHPISA"]),
            ("https://alfred.stlouisfed.org/series?seid=CSUSHPISA", ["CSUSHPISA"]),
            ("https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=CSUSHPISA&vintage_date=2026-07-28", ["CSUSHPISA"]),
            ("https://api.stlouisfed.org/fred/series/observations?series_id=CSUSHPISA&file_type=json", ["CSUSHPISA"]),
        ],
    )
    def test_extracts_from_every_fred_shape(self, text: str, expected: list[str]):
        assert parse.fred_series_ids(text) == expected

    def test_dedupes_preserving_order(self):
        text = "https://fred.stlouisfed.org/series/DGS10 and https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
        assert parse.fred_series_ids(text) == ["DGS10"]

    def test_url_encoded_and_markdown_escaped(self):
        """A criteria URL Metaculus rendered with a backslash-escaped id survives the unescape."""
        assert parse.fred_series_ids(r"https://fred.stlouisfed.org/series/DGS10\_") == ["DGS10_"]

    def test_ignores_the_release_calendar(self):
        assert parse.fred_series_ids("https://fred.stlouisfed.org/releases/calendar?rid=199&y=2026") == []

    def test_caps_the_id_list_per_url_at_two(self):
        """One graph URL cannot expand into an unbounded fetch set."""
        assert parse.fred_series_ids("https://fred.stlouisfed.org/graph/fredgraph.csv?id=A,B,C,D") == ["A", "B"]


class TestYahooSymbols:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("https://finance.yahoo.com/quote/SPCX/history/", ["SPCX"]),
            ("https://uk.finance.yahoo.com/quote/ETH-USD/history/", ["ETH-USD"]),
            ("https://finance.yahoo.com/quote/BZ=F/", ["BZ=F"]),
            ("https://finance.yahoo.com/quote/%5ETYX/history/?period1=1&period2=2", ["^TYX"]),
            ("https://query1.finance.yahoo.com/v8/finance/chart/SPCX?period1=1&period2=2&interval=1d", ["SPCX"]),
            ("https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC", ["^GSPC"]),
        ],
    )
    def test_extracts_from_quote_and_chart_shapes(self, text: str, expected: list[str]):
        assert parse.yahoo_symbols(text) == expected

    def test_strips_a_sentence_final_dot(self):
        assert parse.yahoo_symbols("resolves on https://finance.yahoo.com/quote/%5ETYX.") == ["^TYX"]

    def test_ignores_help_and_news(self):
        assert parse.yahoo_symbols("https://help.yahoo.com/kb/SLN2311.html") == []
        assert parse.yahoo_symbols("https://finance.yahoo.com/news/some-article-123.html") == []


class TestKalshiTicker:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://kalshi.com/markets/KXTRUMPAPPROVALBELOW-26DEC31", "KXTRUMPAPPROVALBELOW-26DEC31"),
            ("https://kalshi.com/markets/kxu3/unemployment/kxu3-26aug", "KXU3-26AUG"),
            ("https://kalshi.com/markets/KXGROK/KXGROK-GROK47", "KXGROK-GROK47"),
            ("https://kalshi.com/api/v2/markets/KXPGACOMPETE-PRESCUP26SEP", "KXPGACOMPETE-PRESCUP26SEP"),
            ("https://kalshi.com/api/v1/markets/KXPGACOMPETE-PRESCUP26SEP", "KXPGACOMPETE-PRESCUP26SEP"),
            (
                "https://kalshi.com/markets/kxhurctotmaj/number-of-major-hurricanes/kxhurctotmaj-26dec01",
                "KXHURCTOTMAJ-26DEC01",
            ),
        ],
    )
    def test_reads_the_ticker_off_every_kalshi_shape(self, url: str, expected: str):
        assert parse.kalshi_ticker(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://kalshi-public-docs.s3.amazonaws.com/contract_terms/U3.pdf",
            "https://kalshi.com/",
            "https://kalshi.com/account",
        ],
    )
    def test_rejects_non_market_urls(self, url: str):
        assert parse.kalshi_ticker(url) is None

    def test_rejects_a_two_segment_series_slug(self):
        """A /markets/{series}/{slug} page has no market ticker; its last segment is a slug, not a ticker."""
        assert parse.kalshi_ticker("https://kalshi.com/markets/kxu3/unemployment") is None
