"""What an exchange-rate identifier looks like on each vendor.

q45363 ("What will be the Boliviano-USD exchange rate on August 31, 2026?") is the realized failure
this module exists for. The classifier proposed the FRED series ``DEXBOUS``, which does not exist --
FRED carries no Bolivia daily FX series, and its real ones follow ``DEXBZUS`` -- and named no Yahoo
cross beside it, so ``financial_data`` rendered NOTHING on a currency question: no level, no 52-week
range, no realized volatility. The forecasters hand-sized their intervals off the gap-fill's TCO
series, and the adversarial verification found that a member sized off the resolving series' own
30-print realized volatility would have scored +55.35 spot peer alone, better than every member that
actually ran. The only trace of any of it was the diagnostics token ``DEXBOUS:empty``, which reads
identically to a live series with no observations.

The fix at the cause is the exchange-rate routing rule in ``financial_data``'s classifier prompt; the
fix at the trace is ``fred_rendering.UnknownFredSeries``, which turns that ``empty`` into
``unknown_series``. These predicates are the third part: they let ``financial_data`` count how many
attempted EXCHANGE-RATE identifiers carried nothing (``counts["fx_identifiers_empty"]``), so a
currency question that got no financial block is a query over the archive rather than a
re-derivation from identifier names.

They deliberately do NOT feed a forecaster-facing "no exchange-rate data available" line. An earlier
revision rendered one, and prose standing in for a provider's absent output is the AskNews
``No articles were found`` shape (``research/providers.py``): the non-empty return flips the
orchestrator's status from ``empty`` to ``ok``, counts in ``providers_succeeded``, and defeats every
downstream empty guard. A financial section with nothing in it is ABSENT, and the per-identifier
loss tokens are the whole record.

Three shape predicates. Stdlib-only (``re``), like ``currency_pegs``, so it can never cycle with the
provider modules that import it -- which is also why the count is computed in ``financial_data``,
where ``provider_diagnostics.is_lost_source`` already lives.
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
