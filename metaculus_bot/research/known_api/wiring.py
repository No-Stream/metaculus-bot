"""Compose the known-API registry into the shared fetch-ladder rung-0 callback.

The registry deliberately exposes neutral backend results. This module owns the small amount of
caller wiring that translates a URL, dispatches its backend, and returns a successful
``FetchResult``. The ladder supplies the current per-URL wall bound around the callback.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from metaculus_bot.research.known_api import backends
from metaculus_bot.research.known_api.result import KnownApiResult
from metaculus_bot.research.known_api.translate import KnownApiCall, translate
from metaculus_bot.research.resolution_fetch_result import FetchResult


def _unique_links(results: list[KnownApiResult]) -> list[str]:
    seen: set[str] = set()
    links: list[str] = []
    for result in results:
        for link in result.links or ([result.source_url] if result.source_url else []):
            if link not in seen:
                seen.add(link)
                links.append(link)
    return links


async def _fred_fanout(call: KnownApiCall) -> KnownApiResult | None:
    results = await asyncio.gather(
        *(
            backends.fred_series(
                series_id=series_id,
                start=call.window_start,
                end=call.window_end,
                first_release=call.fred_first_release,
            )
            for series_id in call.fred_series_ids
        )
    )
    if not results or any(result.status != "ok" for result in results):
        return None
    return KnownApiResult(
        status="ok",
        content_markdown="\n\n".join(result.content_markdown for result in results),
        source_url=call.canonical_url,
        links=_unique_links(results),
    )


async def _dispatch(
    call: KnownApiCall,
    *,
    session: Any,
    kalshi_catalogue: list[dict[str, Any]] | None,
    predictit_markets: list[dict[str, Any]] | None,
    kalshi_detail_budget: backends.KalshiGetBudget,
) -> KnownApiResult | None:
    if call.kind == "fred":
        return await _fred_fanout(call)
    if call.kind == "yahoo":
        if call.yahoo_symbol is None:
            raise AssertionError("Translated Yahoo call has no symbol")
        return await backends.yahoo_history(
            ticker=call.yahoo_symbol,
            start=call.window_start,
            end=call.window_end,
        )
    if call.kind == "kalshi":
        if call.kalshi_ticker is None:
            raise AssertionError("Translated Kalshi call has no ticker")
        return await backends.market_snapshot(
            venue="kalshi",
            market=call.kalshi_ticker,
            session=session,
            kalshi_catalogue=kalshi_catalogue,
            predictit_markets=predictit_markets,
            kalshi_detail_budget=kalshi_detail_budget,
        )
    if call.kind == "edgar":
        return await backends.edgar(call)
    raise AssertionError(f"Unhandled known-API call kind: {call.kind}")


def build_known_api_fetcher(
    *,
    session: Any,
    kalshi_catalogue: list[dict[str, Any]] | None = None,
    predictit_markets: list[dict[str, Any]] | None = None,
    kalshi_detail_budget: backends.KalshiGetBudget | None = None,
) -> Callable[[str], Awaitable[FetchResult | None]]:
    """Build one URL-only rung callback with its question's market resources bound.

    The callback returns ``None`` for unsupported URLs and non-success backend results, allowing the
    normal page ladder to continue. The caller's ``fetch_url`` entry point supplies the current
    per-URL wall timeout around this callback.
    """
    budget = kalshi_detail_budget or backends.KalshiGetBudget()

    async def _fetch(url: str) -> FetchResult | None:
        call = translate(url)
        if call is None:
            return None
        result = await _dispatch(
            call,
            session=session,
            kalshi_catalogue=kalshi_catalogue,
            predictit_markets=predictit_markets,
            kalshi_detail_budget=budget,
        )
        if result is None or result.status != "ok":
            return None
        # Local import avoids the agentic package-init cycle: adapters imports agentic.types while
        # agentic.__init__ imports agentic.tools, which constructs this callback.
        from metaculus_bot.research.known_api import adapters  # noqa: PLC0415  # real circular import

        return adapters.to_fetch_result(result, url=url)

    return _fetch
