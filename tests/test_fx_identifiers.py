"""Tests for the exchange-rate identifier shapes and the no-data disclosure.

The predicates are what decides whether ``financial_data`` says anything at all when every one of
its identifiers came back empty, so both halves matter: a false negative reproduces q45363 (silence
on a currency question), and a false positive turns any empty stock or macro fetch into prose, which
is the AskNews ``No articles were found`` anti-pattern.
"""

import pytest

from metaculus_bot.research.fx_identifiers import (
    FX_NO_DATA_HEADER,
    fx_no_data_disclosure,
    is_fred_fx_series,
    is_fx_identifier,
    is_yahoo_fx_ticker,
)


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


class TestNoDataDisclosure:
    def test_it_names_every_identifier_tried_and_why_each_carried_nothing(self) -> None:
        rendered = fx_no_data_disclosure({"DEXBOUS": "unknown_series", "USDBOB=X": "empty"})

        assert rendered.startswith(FX_NO_DATA_HEADER)
        assert "`DEXBOUS` (FRED reports no such series)" in rendered
        assert "`USDBOB=X` (the vendor returned no history)" in rendered
        # One block, and short: it is a disclosure, not a research section.
        assert rendered.count("###") == 1
        assert len(rendered) < 400

    def test_the_request_order_is_preserved(self) -> None:
        rendered = fx_no_data_disclosure({"USDBOB=X": "empty", "DEXBOUS": "unknown_series"})
        assert rendered.index("USDBOB=X") < rendered.index("DEXBOUS")

    def test_an_unmapped_token_renders_verbatim_rather_than_being_smoothed(self) -> None:
        """A token added later must show up as itself, not as prose that does not describe it."""
        rendered = fx_no_data_disclosure({"USDBOB=X": "deadline"})
        assert "`USDBOB=X` (deadline)" in rendered

    def test_nothing_attempted_renders_nothing(self) -> None:
        assert fx_no_data_disclosure({}) == ""
