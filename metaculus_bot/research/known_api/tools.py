"""The three explicit known-API tools the gap-fill v2 driver gets, ready to append to its list.

``build_known_api_tools`` returns three ``ToolSpec`` objects -- ``fred_series``, ``yahoo_history``
and ``market_snapshot`` -- each pairing a driver-facing description and JSON schema with a handler
that parses the ISO date window off the model's arguments, calls the backend, and adapts the
result to the loop's ``ToolOutcome``. The market handler binds the per-question session, the run's
already-pulled Kalshi catalogue and PredictIt dump, and the per-question Kalshi detail-GET budget
(the seam the loop wiring fills, since the loop has no per-question object yet). Detail:
docs/agentic_gap_fill.md "The known-API tools".

The descriptions steer: these tools return a date window on demand, and the same FRED/Yahoo/Kalshi
URL forms are translated automatically at rung 0, so the driver may also just ``fetch`` the URL.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from metaculus_bot.research.agentic.types import ToolOutcome, ToolSpec
from metaculus_bot.research.known_api import adapters, backends

# Above each backend's own 15 s FRED/Yahoo bound and the market snapshot's per-venue timeouts.
_FRED_YAHOO_TOOL_TIMEOUT_S = 20.0
_MARKET_TOOL_TIMEOUT_S = 30.0

FRED_SERIES_DESCRIPTION = (
    "Read one FRED (Federal Reserve Economic Data) series over a date window, or search FRED's\n"
    "catalogue for a series id. Pass `series_id` (e.g. DGS10, CSUSHPISA) with optional `start`\n"
    "and `end` ISO dates for a windowed table (default: the recent observations), or `search`\n"
    "with free text to find the right id. `first_release` reads the initial-release vintage\n"
    "rather than today's revised values. Deterministic, no cost. A `fred.stlouisfed.org` URL\n"
    "is translated automatically, so you can also just fetch(url).\n"
    'Example: fred_series(series_id="DGS10", start="2026-06-01", end="2026-07-31")'
)

YAHOO_HISTORY_DESCRIPTION = (
    "Read one Yahoo Finance symbol's price history over a date window. Pass `ticker` (e.g.\n"
    "^GSPC, BTC-USD, CL=F) with optional `start`/`end` ISO dates and `column` (Close, High,\n"
    "Low, Open; default Close). Returns a dated table, newest first. Deterministic, no cost. A\n"
    "`finance.yahoo.com/quote/...` URL is translated automatically, so you can also just\n"
    "fetch(url).\n"
    'Example: yahoo_history(ticker="^GSPC", start="2026-08-01", column="Close")'
)

MARKET_SNAPSHOT_DESCRIPTION = (
    "Read a prediction-market snapshot for one venue. Pass `venue` (kalshi, polymarket,\n"
    "manifold, predictit) and `market` as either a venue id/ticker (e.g. Kalshi\n"
    "KXU3-26AUG) or free-text to search. Returns up to five ranked rows with prices.\n"
    "Deterministic, no cost. A `kalshi.com/markets/...` URL is translated automatically, so you\n"
    "can also just fetch(url).\n"
    'Example: market_snapshot(venue="kalshi", market="unemployment rate August")'
)

_FRED_SERIES_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "series_id": {"type": "string"},
        "search": {"type": "string"},
        "start": {"type": "string", "description": "ISO date, inclusive"},
        "end": {"type": "string", "description": "ISO date, inclusive"},
        "first_release": {"type": "boolean"},
    },
    "additionalProperties": False,
}

_YAHOO_HISTORY_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string"},
        "start": {"type": "string", "description": "ISO date, inclusive"},
        "end": {"type": "string", "description": "ISO date, inclusive"},
        "column": {"type": "string", "enum": ["Close", "High", "Low", "Open"]},
    },
    "required": ["ticker"],
    "additionalProperties": False,
}

_MARKET_SNAPSHOT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "venue": {"type": "string", "enum": ["kalshi", "polymarket", "manifold", "predictit"]},
        "market": {"type": "string"},
    },
    "required": ["venue", "market"],
    "additionalProperties": False,
}


def _iso(value: str | None) -> date | None:
    """An ISO date off a model argument, or None for absent or unparseable input."""
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def build_known_api_tools(
    *,
    session: Any,
    kalshi_catalogue: list[dict[str, Any]] | None = None,
    predictit_markets: list[dict[str, Any]] | None = None,
    kalshi_detail_budget: backends.KalshiGetBudget | None = None,
) -> list[ToolSpec]:
    """The three known-API ToolSpecs, with the per-question market dependencies bound in.

    One :class:`KalshiGetBudget` is shared across every ``market_snapshot`` call these tools make,
    so the per-question Kalshi detail-GET ceiling holds across the whole loop, not per call.
    """
    budget = kalshi_detail_budget or backends.KalshiGetBudget()

    async def _fred(
        *,
        series_id: str | None = None,
        search: str | None = None,
        start: str | None = None,
        end: str | None = None,
        first_release: bool = False,
    ) -> ToolOutcome:
        result = await backends.fred_series(
            series_id=series_id, search=search, start=_iso(start), end=_iso(end), first_release=bool(first_release)
        )
        return adapters.to_tool_outcome(result)

    async def _yahoo(
        ticker: str, start: str | None = None, end: str | None = None, column: str = "Close"
    ) -> ToolOutcome:
        result = await backends.yahoo_history(ticker=ticker, start=_iso(start), end=_iso(end), column=column)
        return adapters.to_tool_outcome(result)

    async def _market(venue: str, market: str) -> ToolOutcome:
        result = await backends.market_snapshot(
            venue=venue,
            market=market,
            session=session,
            kalshi_catalogue=kalshi_catalogue,
            predictit_markets=predictit_markets,
            kalshi_detail_budget=budget,
        )
        return adapters.to_tool_outcome(result)

    return [
        ToolSpec(
            name="fred_series",
            description=FRED_SERIES_DESCRIPTION,
            parameters=_FRED_SERIES_PARAMETERS,
            handler=_fred,
            timeout_s=_FRED_YAHOO_TOOL_TIMEOUT_S,
        ),
        ToolSpec(
            name="yahoo_history",
            description=YAHOO_HISTORY_DESCRIPTION,
            parameters=_YAHOO_HISTORY_PARAMETERS,
            handler=_yahoo,
            timeout_s=_FRED_YAHOO_TOOL_TIMEOUT_S,
        ),
        ToolSpec(
            name="market_snapshot",
            description=MARKET_SNAPSHOT_DESCRIPTION,
            parameters=_MARKET_SNAPSHOT_PARAMETERS,
            handler=_market,
            timeout_s=_MARKET_TOOL_TIMEOUT_S,
        ),
    ]
