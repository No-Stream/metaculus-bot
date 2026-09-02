"""Tests for the hard-currency-peg table and its forecaster-facing disclosure.

A pegged FX cross's measured daily volatility is largely vendor noise, so the yfinance block
has to say what is fixed and render the liquid anchor beside it — never instead of it, since
the question still resolves on the pegged pair.
"""

import re
from datetime import timedelta
from unittest.mock import patch

from metaculus_bot.constants import FINANCIAL_YFINANCE_LOOKBACK_DAYS
from metaculus_bot.research.currency_pegs import HARD_PEG_ANCHORS
from metaculus_bot.research.financial_data import _fetch_yfinance_data
from tests.financial_fakes import _BENCH_OPEN_TIME, _clean_close, _noisy_close, _yfinance_by_symbol


class TestPeggedCrossAnchor:
    """A hard-pegged FX cross must arrive labeled, beside its liquid anchor.

    q44797: `USDSZL=X`'s 17.8% "30-day annualized volatility" — 79% vendor noise on a cross
    fixed 1:1 to the rand — went to all six forecasters, four of whom multiplied it into
    their interval width. The honest like-for-like figure off `ZAR=X` was 10.6%. The
    requirement is disclosure plus the anchor, never a silent substitution: the question
    still resolves on the pegged pair, so its own quote has to stay on the page.
    """

    def test_pegged_ticker_renders_both_blocks_and_names_the_peg(self) -> None:
        closes = {"USDSZL=X": _noisy_close(seed=3), "ZAR=X": _clean_close(seed=3)}
        with patch("metaculus_bot.research.financial_data.yfinance", _yfinance_by_symbol(closes)):
            result = _fetch_yfinance_data("USDSZL=X")

        assert "### USDSZL=X" in result, "the pegged pair's own block must survive"
        assert "### ZAR=X" in result, "the liquid anchor must be rendered beside it"
        assert "⚠ Pegged pair: USD/SZL — SZL is fixed at par with the South African rand since 1974" in result
        assert "Peg anchor: `ZAR=X`" in result
        assert "quotes ZAR per US dollar" in result
        assert "_Peg anchor for USDSZL=X" in result
        # The pegged block keeps its own latest price: disclosure, not substitution.
        assert result.index("### USDSZL=X") < result.index("- Latest price:") < result.index("### ZAR=X")

    def test_the_peg_disclosure_precedes_the_statistics_it_qualifies(self) -> None:
        closes = {"USDSZL=X": _noisy_close(seed=4), "ZAR=X": _clean_close(seed=4)}
        with patch("metaculus_bot.research.financial_data.yfinance", _yfinance_by_symbol(closes)):
            result = _fetch_yfinance_data("USDSZL=X")

        assert result.index("⚠ Pegged pair") < result.index("annualized volatility")
        assert result.index("⚠ Pegged pair") < result.index("52-week range")

    def test_both_yahoo_spellings_and_lower_case_resolve_to_the_same_peg(self) -> None:
        """A resolution URL may cite `SZL=X` or `USDSZL=X`, in either case."""
        for ticker in ("SZL=X", "USDSZL=X", "usdszl=x"):
            closes = {ticker: _noisy_close(seed=5), "ZAR=X": _clean_close(seed=5)}
            with patch("metaculus_bot.research.financial_data.yfinance", _yfinance_by_symbol(closes)):
                result = _fetch_yfinance_data(ticker)
            assert "### ZAR=X" in result, f"{ticker} missed its peg anchor"

    def test_a_usd_pegged_currency_says_there_is_no_anchor_to_read(self) -> None:
        """AED/SAR/QAR/HKD are pegged to the USD leg itself, so no third cross exists. The
        honest statement is that the pair has no independent dynamics — not a second block,
        and not silence."""
        closes = {"USDAED=X": _noisy_close(seed=6)}
        with patch("metaculus_bot.research.financial_data.yfinance", _yfinance_by_symbol(closes)) as module:
            result = _fetch_yfinance_data("USDAED=X")

        assert "pegged to the US dollar at 3.6725 since November 1997" in result
        assert "no liquid third-currency cross to read instead" in result
        assert "Do not size a forecast interval" in result
        assert module.Ticker.call_count == 1, "a USD peg must not trigger a second fetch"

    def test_an_unpegged_ticker_renders_no_peg_lines(self) -> None:
        closes = {"EURUSD=X": _clean_close(seed=7)}
        with patch("metaculus_bot.research.financial_data.yfinance", _yfinance_by_symbol(closes)) as module:
            result = _fetch_yfinance_data("EURUSD=X")

        assert "Pegged pair" not in result
        assert "Peg anchor" not in result
        assert module.Ticker.call_count == 1

    def test_an_unfetchable_anchor_degrades_to_a_visible_notice(self) -> None:
        """The anchor is enrichment: losing it must not take the pegged pair's block down,
        and must not pass silently either."""
        closes = {"USDSZL=X": _noisy_close(seed=8)}
        module = _yfinance_by_symbol(closes, missing=("ZAR=X",))
        with patch("metaculus_bot.research.financial_data.yfinance", module):
            result = _fetch_yfinance_data("USDSZL=X")

        assert "### USDSZL=X" in result
        assert "### ZAR=X" not in result
        assert "⚠ The peg anchor `ZAR=X` could not be fetched" in result

    def test_a_failed_primary_fetch_is_still_an_empty_string(self) -> None:
        module = _yfinance_by_symbol({}, missing=("USDSZL=X",))
        with patch("metaculus_bot.research.financial_data.yfinance", module):
            assert _fetch_yfinance_data("USDSZL=X") == ""

    def test_benchmarking_ceiling_applies_to_the_anchor_fetch_too(self) -> None:
        """A leaky anchor would leak just as hard as a leaky primary."""
        closes = {"USDSZL=X": _noisy_close(seed=9), "ZAR=X": _clean_close(seed=9)}
        module = _yfinance_by_symbol(closes)
        with patch("metaculus_bot.research.financial_data.yfinance", module):
            _fetch_yfinance_data("USDSZL=X", as_of=_BENCH_OPEN_TIME, is_benchmarking=True)

        assert module.Ticker.call_count == 2
        # Both fetches carry the same explicit start/end window.
        assert [symbol for symbol, _ in module.history_calls] == ["USDSZL=X", "ZAR=X"]
        expected_start = (_BENCH_OPEN_TIME - timedelta(days=FINANCIAL_YFINANCE_LOOKBACK_DAYS)).date().isoformat()
        for _symbol, kwargs in module.history_calls:
            assert kwargs["end"] == "2026-03-16"  # open_time.date() + 1d, end EXCLUSIVE
            assert kwargs["start"] == expected_start

    def test_a_euro_pegged_row_reads_the_per_usd_euro_cross_and_names_the_currency(self) -> None:
        """The euro pegs must anchor on `EUR=X`, not the inverted `EURUSD=X`.

        Yahoo's `EURUSD=X` quotes US DOLLARS PER EURO, the opposite way round to USD/DKK
        (DKK per USD = 7.46038 / EURUSD), so its signed percent moves are the NEGATIVE of the
        pegged pair's while both rendered sentences claim they transfer unchanged. `EUR=X` is
        the per-USD spelling (euros per USD, like the existing `ZAR=X`/`SGD=X` rows), which
        makes the claim true instead of needing a sign-flip caveat. The regime string also
        states a rate against the EURO, so the disclosure's subject must be the currency: the
        old "USD/DKK is held near ... 7.46038 per euro" printed a ~16% level error directly
        above that block's own "Latest price: 6.44".
        """
        for pegged, currency in (("USDDKK=X", "DKK"), ("USDXOF=X", "XOF"), ("USDXAF=X", "XAF")):
            closes = {pegged: _noisy_close(seed=12), "EUR=X": _clean_close(seed=12)}
            with patch("metaculus_bot.research.financial_data.yfinance", _yfinance_by_symbol(closes)):
                result = _fetch_yfinance_data(pegged)

            assert "### EUR=X" in result, f"{pegged} must anchor on the per-USD euro cross"
            assert "EURUSD=X" not in result, f"{pegged} must not anchor on the inverted cross"
            assert "Peg anchor: `EUR=X`" in result
            assert "quotes EUR per US dollar" in result
            assert f"⚠ Pegged pair: USD/{currency} — {currency} is " in result
            # The peg rate is quoted per EURO; attaching it to the USD cross is the level error.
            assert f"USD/{currency} is held near" not in result
            assert f"USD/{currency} is fixed at 655.957" not in result

    def test_no_anchor_is_itself_pegged_so_the_render_cannot_recurse(self) -> None:
        """The one-level recursion bound `_fetch_yfinance_data` documents."""
        anchors = {peg.anchor_ticker for peg in HARD_PEG_ANCHORS.values() if peg.anchor_ticker}
        assert anchors
        assert not (anchors & set(HARD_PEG_ANCHORS)), "an anchor that is itself pegged would recurse"

    def test_every_anchor_is_the_per_usd_form(self) -> None:
        """`XXX=X` is Yahoo's units-per-US-dollar spelling, the same orientation as the
        `USD<currency>=X` pair a question resolves on. An inverted `EURUSD=X`-shaped anchor
        would hand the forecaster percent moves with the wrong SIGN under a sentence saying
        they transfer unchanged, so the shape is forbidden here rather than caveated in prose.
        """
        anchors = {peg.anchor_ticker for peg in HARD_PEG_ANCHORS.values() if peg.anchor_ticker}
        assert anchors
        for anchor in anchors:
            assert re.fullmatch(r"[A-Z]{3}=X", anchor), f"{anchor} is not Yahoo's per-USD form"

    def test_every_peg_entry_carries_both_spellings_and_a_dated_regime(self) -> None:
        for ticker, peg in HARD_PEG_ANCHORS.items():
            assert ticker.endswith("=X"), f"{ticker} is not a Yahoo FX ticker form"
            assert HARD_PEG_ANCHORS[f"{peg.currency}=X"] is peg
            assert HARD_PEG_ANCHORS[f"USD{peg.currency}=X"] is peg
            # A regime with no date is an unsourced claim in a forecaster prompt.
            assert re.search(r"\b(19|20)\d{2}\b|1980s", peg.regime), f"{ticker}: {peg.regime!r} has no date"
