"""Hard currency pegs: which USD FX crosses are a fixed quote, and where their dynamics live.

A pegged cross's measured daily volatility is largely vendor noise, so the yfinance block
renders a disclosure above its own statistics and appends the liquid anchor cross's block
below. Split out of ``financial_data`` because it is a static reference table plus two
accessors, sharing nothing with the rest of that module (stdlib ``dataclass`` is its only
import, so it can never cycle).

q44797 is the realized failure this table exists for: ``USDSZL=X``'s 17.8% "30-day annualized
volatility" went to all six forecasters, 79% of that series' return variance was quote noise,
and the like-for-like figure off the liquid rand cross was 10.6%.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CurrencyPeg:
    """A hard peg on a USD FX cross: where the pair's real dynamics live, and the regime.

    ``anchor_ticker`` is the LIQUID cross the pegged leg is fixed to, which is the honest
    read of the pegged pair's volatility and percent moves; ``None`` means the leg is
    pegged to the USD leg itself (HKD, AED, SAR, QAR), where no substitute cross exists and
    the only honest statement is that the pair has no independent market dynamics.
    ``regime`` names the peg rate and its start date and is rendered verbatim.

    Every ``anchor_ticker`` is Yahoo's per-USD ``XXX=X`` form — units of the anchor currency
    per US dollar, the same way round as the ``USD<currency>=X`` pair the question resolves
    on. That orientation is the whole reason the block can hand the anchor's SIGNED percent
    moves and its volatility to a forecaster as the pegged pair's own; only the price LEVELS
    are a different quantity (and only when the peg is not at par). An inverted anchor such
    as ``EURUSD=X`` (US dollars per euro) would move the opposite way and is forbidden by
    test — see ``test_every_anchor_is_the_per_usd_form``.
    """

    currency: str
    regime: str
    anchor_ticker: str | None


def _peg_entries(peg: CurrencyPeg) -> dict[str, CurrencyPeg]:
    """Both Yahoo spellings of USD/<currency>: the explicit pair and the USD-implied form.

    Yahoo serves the same cross as ``USDSZL=X`` and as ``SZL=X``; a question's resolution
    URL may cite either, and a lookup that knew only one would miss the peg silently.
    """
    return {f"USD{peg.currency}=X": peg, f"{peg.currency}=X": peg}


# Currencies whose USD cross is a fixed or band-bounded quote rather than a traded market,
# so its measured daily volatility is largely vendor noise. q44797 is the realized failure:
# `USDSZL=X`'s 17.8% "30-day annualized volatility" went to all six forecasters, 79% of that
# series' return variance was quote noise, and the like-for-like figure off the liquid rand
# cross was 10.6% — the single cheapest fix point in that question's ~32 peer-point loss.
# Deliberately a STATIC table, not a correlation detector: the failure class is hard pegs,
# which are published policy and do not need inferring. Peg facts verified 2026-09-01
# against each regime's own authority (HKMA, ECB/Danmarks Nationalbank, BCEAO, the CMA
# treaty history, and Reuters' FX-regime factbox for the Gulf pegs).
HARD_PEG_ANCHORS: dict[str, CurrencyPeg] = {
    # Common Monetary Area: each member issues its own currency at par with the rand, so
    # USD/<member> IS USD/ZAR plus quote noise. Dates are each currency's own par entry.
    **_peg_entries(
        CurrencyPeg(
            "SZL", "fixed at par with the South African rand since 1974 under the Common Monetary Area", "ZAR=X"
        )
    ),
    **_peg_entries(
        CurrencyPeg(
            "LSL", "fixed at par with the South African rand since 1980 under the Common Monetary Area", "ZAR=X"
        )
    ),
    **_peg_entries(
        CurrencyPeg(
            "NAD", "fixed at par with the South African rand since 1993 under the Common Monetary Area", "ZAR=X"
        )
    ),
    # Pegged to the euro, so the euro's own USD cross carries the dynamics. The anchor is
    # `EUR=X` — Yahoo's per-USD spelling, euros per US dollar (~0.86) — NOT `EURUSD=X`, which
    # quotes US dollars per euro (~1.16) and therefore moves the OPPOSITE way to USD/DKK:
    # USD/DKK = 7.46038 * (EUR per USD), so a 2% rise in `EUR=X` is a ~2% rise in the pegged
    # pair, while a 2% rise in `EURUSD=X` is a ~2% FALL in it. With the per-USD form the
    # anchor's signed percent moves and its volatility are the pegged pair's own; only the
    # levels differ.
    **_peg_entries(
        CurrencyPeg(
            "DKK",
            "held near the ERM II central rate of 7.46038 per euro (band 7.29252-7.62824) under Denmark's "
            "fixed-exchange-rate policy, in force since the 1980s",
            "EUR=X",
        )
    ),
    **_peg_entries(
        CurrencyPeg(
            "XOF",
            "fixed at 655.957 per euro since 1 January 1999, guaranteed by the French Treasury",
            "EUR=X",
        )
    ),
    **_peg_entries(
        CurrencyPeg(
            "XAF",
            "fixed at 655.957 per euro since 1 January 1999, guaranteed by the French Treasury",
            "EUR=X",
        )
    ),
    # Interchangeable at par with the Singapore dollar, which is the traded cross.
    **_peg_entries(
        CurrencyPeg(
            "BND",
            "interchangeable at par with the Singapore dollar under the 1967 Currency Interchangeability Agreement",
            "SGD=X",
        )
    ),
    # Pegged to the USD leg itself: no third currency to read instead.
    **_peg_entries(
        CurrencyPeg(
            "HKD",
            "held inside the HKMA's 7.75-7.85 Convertibility Zone (linked to the US dollar at 7.80 since "
            "17 October 1983; the two-sided band since May 2005)",
            None,
        )
    ),
    **_peg_entries(CurrencyPeg("AED", "pegged to the US dollar at 3.6725 since November 1997", None)),
    **_peg_entries(CurrencyPeg("SAR", "pegged to the US dollar at 3.75 since 1986", None)),
    **_peg_entries(
        CurrencyPeg("QAR", "pegged to the US dollar at 3.64 since 1980 (official band 3.6385-3.6415)", None)
    ),
}


def peg_for_ticker(ticker: str) -> CurrencyPeg | None:
    """The peg record for a Yahoo FX ticker, or None when the pair is a traded market."""
    return HARD_PEG_ANCHORS.get(ticker.upper())


def peg_disclosure_lines(peg: CurrencyPeg) -> list[str]:
    """The forecaster-facing peg warning: what is fixed, and what to read instead.

    Takes the peg, not the ticker: the pair it names is derived from ``peg.currency``, which is
    correct whichever of the two Yahoo spellings the question cited (the ``ticker`` parameter
    this used to carry was never read).

    The regime's subject is the CURRENCY, not the USD cross: every ``regime`` string states a
    rate against the peg's own reference (7.46038 DKK per euro, 655.957 XOF per euro), which
    is not the USD cross's level — rendering it as "USD/XOF is fixed at 655.957 per euro" put
    a ~16% level error directly above that block's own "Latest price: 565.31".
    """
    pair = f"USD/{peg.currency}"
    lines = [
        f"- ⚠ Pegged pair: {pair} — {peg.currency} is {peg.regime}. The volatility, 52-week "
        f"range and daily closes in this block are a thin vendor quote on a fixed cross, so "
        f"most of their day-to-day movement is quote noise rather than exchange-rate risk."
    ]
    if peg.anchor_ticker is None:
        lines.append(
            f"- There is no liquid third-currency cross to read instead — {peg.currency} is pegged to the US "
            "dollar itself, so this pair has no independent market dynamics. Do not size a forecast interval "
            "from any volatility figure below."
        )
    else:
        anchor_currency = peg.anchor_ticker.split("=")[0]
        lines.append(
            f"- Peg anchor: `{peg.anchor_ticker}` is the liquid cross the peg is fixed to, and its block is "
            f"appended directly below. It quotes {anchor_currency} per US dollar — the same way round as "
            f"{pair}, so read ITS volatility and signed percent moves as {pair}'s own."
        )
    return lines
