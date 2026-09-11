"""``translate``: one observed URL shape to one deterministic known-API call, or None.

Parametrised over every translatable shape in the 2026-09-09 cost pass's
``url_shapes_results.json`` plus the negative cases that must fall through to the page ladder
(Yahoo help pages, news articles, the Kalshi contract-terms PDF, the FRED release calendar).
"""

from __future__ import annotations

from datetime import date

import pytest

from metaculus_bot.research.known_api import translate
from metaculus_bot.research.known_api.translate import KnownApiCall


class TestFredShapes:
    def test_series_page(self):
        call = translate("https://fred.stlouisfed.org/series/DGS30")
        assert call == KnownApiCall(
            kind="fred", fred_series_ids=("DGS30",), canonical_url="https://fred.stlouisfed.org/series/DGS30"
        )

    def test_data_page(self):
        call = translate("https://fred.stlouisfed.org/data/CSUSHPISA")
        assert call is not None
        assert call.kind == "fred"
        assert call.fred_series_ids == ("CSUSHPISA",)

    def test_fredgraph_csv_two_ids_with_end(self):
        call = translate("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS30,DGS10&cos=Close&coed=2026-07-24")
        assert call is not None
        assert call.fred_series_ids == ("DGS30", "DGS10")
        assert call.window_start is None
        assert call.window_end == date(2026, 7, 24)

    def test_fredgraph_csv_observation_bounds(self):
        call = translate(
            "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS30,DGS10"
            "&observation_start=2026-06-01&observation_end=2026-07-31"
        )
        assert call is not None
        assert call.window_start == date(2026, 6, 1)
        assert call.window_end == date(2026, 7, 31)

    def test_fredgraph_caps_at_two_ids(self):
        call = translate("https://fred.stlouisfed.org/graph/fredgraph.csv?id=A,B,C,D")
        assert call is not None
        assert call.fred_series_ids == ("A", "B")

    def test_fredgraph_xls(self):
        call = translate("https://fred.stlouisfed.org/graph/fredgraph.xls?id=CSUSHPISA")
        assert call is not None
        assert call.fred_series_ids == ("CSUSHPISA",)

    def test_alfred_series_page_is_first_release(self):
        call = translate("https://alfred.stlouisfed.org/series?seid=CSUSHPISA")
        assert call is not None
        assert call.fred_series_ids == ("CSUSHPISA",)
        assert call.fred_first_release is True

    def test_alfred_graph_csv_is_first_release(self):
        call = translate("https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=CSUSHPISA&vintage_date=2026-07-28")
        assert call is not None
        assert call.fred_first_release is True

    def test_fred_api_observations_endpoint(self):
        call = translate("https://api.stlouisfed.org/fred/series/observations?series_id=CSUSHPISA&file_type=json")
        assert call is not None
        assert call.kind == "fred"
        assert call.fred_series_ids == ("CSUSHPISA",)

    def test_release_calendar_is_not_translatable(self):
        assert translate("https://fred.stlouisfed.org/releases/calendar?rid=199&y=2026") is None


class TestYahooShapes:
    def test_quote_history_page(self):
        call = translate("https://finance.yahoo.com/quote/SPCX/history/")
        assert call == KnownApiCall(
            kind="yahoo", yahoo_symbol="SPCX", canonical_url="https://finance.yahoo.com/quote/SPCX/history/"
        )

    def test_regional_quote_page(self):
        call = translate("https://uk.finance.yahoo.com/quote/ETH-USD/history/")
        assert call is not None
        assert call.yahoo_symbol == "ETH-USD"

    def test_url_encoded_symbol(self):
        call = translate("https://finance.yahoo.com/quote/%5ETYX/history/?period1=1785542400&period2=1786233600")
        assert call is not None
        assert call.yahoo_symbol == "^TYX"
        assert call.window_start == date(2026, 8, 1)
        assert call.window_end == date(2026, 8, 8)

    def test_bare_equals_symbol(self):
        call = translate("https://finance.yahoo.com/quote/BZ=F/")
        assert call is not None
        assert call.yahoo_symbol == "BZ=F"

    def test_chart_endpoint(self):
        call = translate(
            "https://query1.finance.yahoo.com/v8/finance/chart/SPCX?period1=1784592000&period2=1785024000&interval=1d"
        )
        assert call is not None
        assert call.kind == "yahoo"
        assert call.yahoo_symbol == "SPCX"
        assert call.window_start == date(2026, 7, 21)
        assert call.window_end == date(2026, 7, 25)

    @pytest.mark.parametrize("period2", ["", "invalid", "999999999999999999999"])
    def test_invalid_end_leaves_window_unbounded(self, period2: str) -> None:
        call = translate(f"https://finance.yahoo.com/quote/SPCX/history/?period2={period2}")
        assert call is not None
        assert call.window_end is None

    @pytest.mark.parametrize(
        "url",
        [
            "https://help.yahoo.com/kb/SLN28256.html",
            "https://finance.yahoo.com/personal-finance/investing/article/x.html",
            "https://www.yahoo.com/news/politics/articles/musks-america-pac-plans-100-205928122.html",
        ],
    )
    def test_help_and_news_are_not_translatable(self, url: str):
        assert translate(url) is None


class TestKalshiShapes:
    @pytest.mark.parametrize(
        ("url", "ticker"),
        [
            ("https://kalshi.com/markets/KXTRUMPAPPROVALBELOW-26DEC31", "KXTRUMPAPPROVALBELOW-26DEC31"),
            ("https://kalshi.com/markets/kxu3/unemployment/kxu3-26aug", "KXU3-26AUG"),
            ("https://kalshi.com/api/v2/markets/KXPGACOMPETE-PRESCUP26SEP", "KXPGACOMPETE-PRESCUP26SEP"),
        ],
    )
    def test_market_urls(self, url: str, ticker: str):
        call = translate(url)
        assert call is not None
        assert call.kind == "kalshi"
        assert call.kalshi_ticker == ticker

    def test_s3_contract_terms_pdf_is_not_translatable(self):
        assert translate("https://kalshi-public-docs.s3.amazonaws.com/contract_terms/U3.pdf") is None


class TestEdgarShapes:
    def test_archives_filing_document(self):
        url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
        call = translate(url)
        assert call is not None
        assert call.kind == "edgar"
        assert call.edgar_kind == "filing_document"
        assert call.edgar_url == url

    def test_browse_edgar_company_page(self):
        call = translate("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=10-K")
        assert call is not None
        assert call.kind == "edgar"
        assert call.edgar_kind == "company_submissions"
        assert call.edgar_arg == "0000320193"

    def test_sec_homepage_is_not_translatable(self):
        assert translate("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent") is None


class TestUnknownHosts:
    @pytest.mark.parametrize(
        "url",
        [
            "https://en.wikipedia.org/wiki/Foo",
            "https://example.gov/report",
            "not a url",
            "",
        ],
    )
    def test_returns_none(self, url: str):
        assert translate(url) is None
