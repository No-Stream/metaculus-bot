"""Tests for the exchange-rate identifier shapes.

The predicates decide which identifiers ``financial_data`` counts under
``counts["fx_identifiers_empty"]``, the one field that says a currency question's vendors carried
nothing. Both directions matter: a false negative loses q45363's signal from the archive, and a
false positive files an ordinary empty stock or macro fetch as a missing exchange rate. Nothing
here renders to a forecaster -- a financial section with nothing in it is absent, per the AskNews
``No articles were found`` rule.
"""

import pytest

from metaculus_bot.research.fx_identifiers import is_fred_fx_series, is_fx_identifier, is_yahoo_fx_ticker


class TestFredFxSeriesShape:
    @pytest.mark.parametrize(
        "series_id",
        [
            "DEXBZUS",  # Brazilian reals per US dollar — a real series, the q45363 control
            "DEXUSEU",  # US dollars per euro
            "DEXBOUS",  # the hallucinated Bolivia id; shape-legal, which is why the fetch decides
            "dexbzus",  # a URL-extracted id keeps the page's own casing
        ],
    )
    def test_the_fx_family_is_recognized(self, series_id: str) -> None:
        assert is_fred_fx_series(series_id)

    @pytest.mark.parametrize("series_id", ["UNRATE", "CPIAUCSL", "DGS10", "T10Y2Y", "DEX", "DEXBZ", "DEXBRAZUS"])
    def test_every_other_fred_series_is_not(self, series_id: str) -> None:
        assert not is_fred_fx_series(series_id)


class TestYahooFxTickerShape:
    @pytest.mark.parametrize(
        "ticker",
        [
            "BOB=X",  # units of the foreign currency per US dollar, USD implied
            "USDBOB=X",  # the same cross spelled out
            "BOBUSD=X",  # the inverse: US dollars per unit
            "EURUSD=X",
            "zar=x",  # currency_pegs stores anchors uppercase; a URL may not
        ],
    )
    def test_both_spellings_of_a_cross_are_recognized(self, ticker: str) -> None:
        assert is_yahoo_fx_ticker(ticker)

    @pytest.mark.parametrize(
        "ticker",
        [
            "DX-Y.NYB",  # the dollar INDEX, not a cross
            "CL=F",  # a future, not FX
            "AAPL",
            "^GSPC",
            "BTC-USD",  # crypto is a Yahoo pair but not an `=X` FX cross
            "USDBOBB=X",  # seven letters is no currency pair
        ],
    )
    def test_indices_futures_and_equities_are_not(self, ticker: str) -> None:
        assert not is_yahoo_fx_ticker(ticker)

    def test_either_vendor_shape_counts_as_an_exchange_rate(self) -> None:
        assert is_fx_identifier("DEXBOUS")
        assert is_fx_identifier("USDBOB=X")
        assert not is_fx_identifier("UNRATE")
