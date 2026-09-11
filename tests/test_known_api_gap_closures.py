"""The two URL-parsing gaps the known-API registry closes, tested at the delegating seams.

The parsers themselves are tested in ``test_known_api_parse.py``; here we assert the two existing
consumers now delegate to them: the resolution-source fetcher skips regional Yahoo quote pages, and
the financial-data provider extracts ids from the fredgraph CSV/XLS and the query1 chart endpoint.
"""

from __future__ import annotations

from metaculus_bot.research.financial_data import extract_financial_identifiers_from_criteria
from metaculus_bot.research.resolution_url_scan import is_yahoo_ticker_url


class TestResolutionFetcherYahooSkip:
    def test_skips_regional_quote_pages(self):
        assert is_yahoo_ticker_url("https://uk.finance.yahoo.com/quote/%5EGSPC/history/") is True
        assert is_yahoo_ticker_url("https://ca.finance.yahoo.com/quote/SPCX/history/") is True

    def test_still_fetches_regional_news(self):
        assert is_yahoo_ticker_url("https://uk.finance.yahoo.com/news/politics") is False


class TestFinancialExtractionGaps:
    def test_reads_fred_ids_from_a_fredgraph_csv_url(self):
        text = "NOB spread: https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS30,DGS10&coed=2026-07-24"
        assert extract_financial_identifiers_from_criteria(text)["fred_series"] == ["DGS30", "DGS10"]

    def test_reads_a_yahoo_symbol_from_the_chart_endpoint(self):
        text = "https://query1.finance.yahoo.com/v8/finance/chart/SPCX?period1=1&period2=2&interval=1d"
        assert extract_financial_identifiers_from_criteria(text)["tickers"] == ["SPCX"]
