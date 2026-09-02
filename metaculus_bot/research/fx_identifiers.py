"""What an exchange-rate identifier looks like on each vendor, and what to say when neither has one.

q45363 ("What will be the Boliviano-USD exchange rate on August 31, 2026?") is the realized failure
this module exists for. The classifier proposed the FRED series ``DEXBOUS``, which does not exist --
FRED carries no Bolivia daily FX series, and its real ones follow ``DEXBZUS`` -- and named no Yahoo
cross beside it, so ``financial_data`` rendered NOTHING on a currency question: no level, no 52-week
range, no realized volatility. The forecasters hand-sized their intervals off the gap-fill's TCO
series, and the adversarial verification found that a member sized off the resolving series' own
30-print realized volatility would have scored +55.35 spot peer alone, better than every member that
actually ran. The only trace of any of it was the diagnostics token ``DEXBOUS:empty``, which reads
identically to a live series with no observations.

Two shape predicates and one disclosure line. Stdlib-only (``re``), like ``currency_pegs``, so it can
never cycle with the provider modules that import it.
"""

import re

# FRED's daily exchange-rate family: ``DEX`` plus two 2-letter country codes, one of which is US
# (``DEXBZUS`` is Brazilian reals per US dollar, ``DEXUSEU`` is US dollars per euro). Deliberately a
# SHAPE check rather than a membership list of the real series: FRED's country codes are its own
# (BZ Brazil, SZ Switzerland, CH China), so enumerating them would be a table nothing in this repo
# can verify offline, and the shape is all a caller needs. Whether the series EXISTS is answered by
# the fetch itself -- see ``fred_rendering.UnknownFredSeries``.
_FRED_FX_SERIES_RE = re.compile(r"^DEX[A-Z]{4}$")

# Yahoo's FX suffix. Both spellings of one cross are legal and mean opposite directions:
# ``USDBOB=X`` (equivalently ``BOB=X``) quotes units of the foreign currency per US dollar, while
# ``BOBUSD=X`` quotes US dollars per unit. ``currency_pegs`` already relies on the same two forms,
# and the ``=X`` suffix is what separates a cross from an index (``DX-Y.NYB``) or a future (``CL=F``).
_YAHOO_FX_TICKER_RE = re.compile(r"^(?:[A-Z]{3}|[A-Z]{6})=X$")


def is_fred_fx_series(identifier: str) -> bool:
    """Whether ``identifier`` has the shape of a FRED daily exchange-rate series."""
    return bool(_FRED_FX_SERIES_RE.fullmatch(identifier.strip().upper()))


def is_yahoo_fx_ticker(identifier: str) -> bool:
    """Whether ``identifier`` has the shape of a Yahoo FX cross, in either spelling."""
    return bool(_YAHOO_FX_TICKER_RE.fullmatch(identifier.strip().upper()))


def is_fx_identifier(identifier: str) -> bool:
    """Whether ``identifier`` names an exchange rate on either vendor."""
    return is_fred_fx_series(identifier) or is_yahoo_fx_ticker(identifier)


# Why one attempted identifier carried no data, in the disclosure's own words. Keys are the
# ``details["sources"]`` tokens ``financial_data`` records; an unlisted token renders verbatim so a
# newly-added one shows up as itself rather than being smoothed into prose that does not describe it.
_OUTCOME_PHRASES: dict[str, str] = {
    "unknown_series": "FRED reports no such series",
    "empty": "the vendor returned no history",
    "error": "the fetch failed",
    "skipped(no_fred_api_key)": "not fetched, no FRED API key is configured",
}

FX_NO_DATA_HEADER = "### Exchange rate: no vendor data available"


def fx_no_data_disclosure(attempted: dict[str, str]) -> str:
    """One forecaster-facing block naming every exchange-rate source tried and why it carried nothing.

    ``attempted`` maps identifier to its ``details["sources"]`` token, in the order the provider
    requested them. Empty string for an empty map, so a caller can render it unconditionally.
    """
    if not attempted:
        return ""
    reasons = ", ".join(
        f"`{identifier}` ({_OUTCOME_PHRASES.get(token, token)})" for identifier, token in attempted.items()
    )
    return (
        f"{FX_NO_DATA_HEADER}\n\n"
        "- ⚠ This bundle carries no level, range or volatility figure for this exchange rate. Every "
        f"exchange-rate source attempted returned nothing: {reasons}."
    )
