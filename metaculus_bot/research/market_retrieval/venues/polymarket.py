"""Polymarket: Gamma's public search, parsed from either the events or the markets list.

``events_status=active`` is load-bearing rather than tidy: nothing downstream filters on close
date any more (the pipeline passes ``as_of=None``), so without the parameter ``Ethereum`` puts 7
closed events out of 10 straight into the pool.

**A multi-market EVENT quotes no price of its own.** An event is a set of mutually-exclusive
outcomes ("How many Fed rate cuts in 2026?" → 0, 1, 2, ... cuts), each a separate nested market
with its own book, so no single number answers the event's title. Reading ``markets[0]`` for the
price legs — what this module did until 2026-08-05 — rendered the FIRST child's probability under
the EVENT's title: on this package's own committed fixture, "How many Fed rate cuts in 2026?" with
prob 0.89, which is the "will no cuts happen" child. The forecaster prompts tell the model to
anchor on a matched market's price, so that row mislabelled an anchor. Every nested market now
renders as its own ``MarketChild`` sub-row instead, and the event's price legs are ``None``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

from metaculus_bot.research.market_retrieval.http import http_get_with_backoff, parse_iso, safe_float
from metaculus_bot.research.market_retrieval.types import MarketChild, MarketMatch
from metaculus_bot.research.market_retrieval.venues._shared import RULES_TEXT_MAX_CHARS, VENUE_SEARCH_LIMIT

logger = logging.getLogger(__name__)

POLYMARKET_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"

POLYMARKET_MAX_ATTEMPTS = 2


def _prob_from_prices(prices: Any) -> float | None:
    if isinstance(prices, str):
        try:
            parsed = json.loads(prices)
        except (json.JSONDecodeError, ValueError):
            return None
        if isinstance(parsed, list) and parsed:
            try:
                return float(parsed[0])
            except (TypeError, ValueError):
                return None
    if isinstance(prices, list) and prices:
        try:
            return float(prices[0])
        except (TypeError, ValueError):
            return None
    return None


def _polymarket_market_volume(market: dict[str, Any]) -> float | None:
    """One nested market's all-time volume: ``volumeNum``, else the string-typed ``volume``."""
    total_volume = safe_float(market.get("volumeNum"))
    return total_volume if total_volume is not None else safe_float(market.get("volume"))


def polymarket_event_children(markets: Sequence[dict[str, Any]]) -> tuple[MarketChild, ...]:
    """One child per nested market, MOST-TRADED FIRST, each priced off its OWN ``outcomePrices``.

    The title is ``groupItemTitle`` ("0 (0 bps)", "1 (25 bps)") in preference to ``question``
    ("Will no Fed rate cuts happen in 2026?"): it is the label Polymarket's own event page shows
    against each outcome, it is a third the length, and under the parent's title it reads as the
    ladder rung it is. ``question`` is the fallback for the events that ship no group label.

    Volume-descending because the renderer truncates a long child list from the end, so the order
    decides what survives the budget — the most-traded outcome is the one a forecaster most needs.
    The sort is STABLE, so an event whose children all report the same volume (or none) keeps
    Gamma's own order rather than an arbitrary one.

    A market with no usable title is dropped rather than rendered as a blank row; it would spend a
    child slot saying nothing.
    """
    children = [
        MarketChild(
            title=str(title),
            implied_prob_yes=_prob_from_prices(market.get("outcomePrices")),
            total_volume=_polymarket_market_volume(market),
            open_interest=safe_float(market.get("openInterest")),
            is_resolved=bool(market.get("closed")) or bool(market.get("resolved")),
            close_time=parse_iso(market.get("endDate") or market.get("end_date_iso") or ""),
        )
        for market in markets
        if (title := market.get("groupItemTitle") or market.get("question") or market.get("title") or "")
    ]
    children.sort(key=lambda child: max(child.total_volume or 0.0, child.open_interest or 0.0), reverse=True)
    return tuple(children)


def _polymarket_event_match(event: dict[str, Any], rank: int) -> MarketMatch:
    """One Gamma event as a candidate row: a single-outcome event's own price, or its children's.

    A MULTI-market event quotes no price of its own — see the module docstring for the mislabelled
    anchor that came of reading ``markets[0]`` — and its money columns come from the EVENT rather
    than from a sum over the children. That is not a stylistic preference: Gamma's public-search
    response TRUNCATES the nested markets list, so on the committed fixture the event reports
    $46.2M of volume against $9.1M across the two children it shipped. Summing the visible subset
    would understate the family by 5x.

    A single-market event is unchanged, deliberately: event and market ask the same question there,
    so the price legs are the market's and the row renders exactly as it always has.
    """
    title = event.get("title") or event.get("question") or ""
    slug = str(event.get("slug") or "")
    volume = safe_float(event.get("volume"))
    # Gamma carries `openInterest` on the EVENT, not the nested market (5/5 live events vs
    # 0/42 nested, verified 2026-08-03) — reading it only at the market level left the
    # rendered OI column blank on every Polymarket row ever archived.
    event_open_interest = safe_float(event.get("openInterest"))

    implied: float | None = None
    bid: float | None = None
    ask: float | None = None
    vol_24h: float | None = None
    total_volume: float | None = None
    liquidity: float | None = None
    open_interest: float | None = None
    children: tuple[MarketChild, ...] = ()
    markets = [market for market in (event.get("markets") or []) if isinstance(market, dict)]
    if len(markets) > 1:
        children = polymarket_event_children(markets)
        vol_24h = safe_float(event.get("volume24hr"))
        total_volume = volume
        liquidity = safe_float(event.get("liquidity"))
        open_interest = event_open_interest
    elif markets:
        first = markets[0]
        implied = _prob_from_prices(first.get("outcomePrices"))
        bid = safe_float(first.get("bestBid"))
        ask = safe_float(first.get("bestAsk"))
        vol_24h = safe_float(first.get("volume24hr"))
        # volumeNum is Gamma's all-time total; fall back to the event-level or market-level
        # `volume` when it is absent.
        total_volume = safe_float(first.get("volumeNum"))
        if total_volume is None:
            total_volume = volume if volume is not None else safe_float(first.get("volume"))
        liquidity = safe_float(first.get("liquidityNum"))
        if liquidity is None:
            liquidity = safe_float(first.get("liquidity"))
        # Market-level first (more specific when present), event-level as the fallback that
        # actually populates on today's payloads.
        open_interest = safe_float(first.get("openInterest"))
        if open_interest is None:
            open_interest = event_open_interest
    else:
        total_volume = volume
        open_interest = event_open_interest

    return MarketMatch(
        platform="polymarket",
        market_title=title,
        market_url=f"https://polymarket.com/event/{slug}" if slug else "",
        implied_prob_yes=implied,
        bid=bid,
        ask=ask,
        spread=(ask - bid) if (bid is not None and ask is not None) else None,
        volume_24h=vol_24h if vol_24h is not None else volume,
        close_time=parse_iso(event.get("endDate") or event.get("end_date_iso") or ""),
        is_resolved=bool(event.get("closed")) or bool(event.get("resolved")),
        match_confidence=100.0 - rank,
        raw_rules=(event.get("description") or "")[:RULES_TEXT_MAX_CHARS],
        total_volume=total_volume,
        liquidity=liquidity,
        open_interest=open_interest,
        venue_market_id=slug,
        retrieval_channel="venue_search",
        children=children,
    )


def _polymarket_market_match(market: dict[str, Any], rank: int) -> MarketMatch:
    slug = str(market.get("slug") or "")
    total_volume = _polymarket_market_volume(market)
    liquidity = safe_float(market.get("liquidityNum"))
    if liquidity is None:
        liquidity = safe_float(market.get("liquidity"))
    return MarketMatch(
        platform="polymarket",
        market_title=market.get("question") or market.get("title") or "",
        market_url=f"https://polymarket.com/market/{slug}" if slug else "",
        implied_prob_yes=_prob_from_prices(market.get("outcomePrices")),
        bid=safe_float(market.get("bestBid")),
        ask=safe_float(market.get("bestAsk")),
        spread=None,
        volume_24h=safe_float(market.get("volume24hr")),
        close_time=parse_iso(market.get("endDate") or ""),
        is_resolved=bool(market.get("closed")),
        match_confidence=100.0 - rank,
        raw_rules=(market.get("description") or "")[:RULES_TEXT_MAX_CHARS],
        total_volume=total_volume,
        liquidity=liquidity,
        open_interest=safe_float(market.get("openInterest")),
        venue_market_id=slug,
        retrieval_channel="venue_search",
    )


def parse_polymarket_matches(payload: Any, *, width: int) -> list[MarketMatch] | None:
    """Parse a Gamma public-search response into candidate rows, venue-rank order.

    Gamma returns ``{events: [...], markets: [...]}``; events are preferred and the
    top-level markets list is the fallback for when there were none. BOTH branches read
    ``openInterest``, because both are live paths and the field sits in a different place in
    each.

    ``match_confidence`` is the venue's OWN rank inverted, never a local re-score: the point
    of the venue-index channel is to stop trying to out-rank the index.

    ``None`` when the TOP-LEVEL shape is not the documented dict, honouring the package's
    ``None``-means-fetch-failed contract. ``http_get_with_backoff`` has already turned every
    non-200 and every undecodable body into ``None`` before this runs, so the only payload that
    reaches here shapeless is a 200 whose contract changed — a loss, not a search that found
    nothing. Returning ``[]`` for it counted as a clean query in ``flatten_results`` and
    published as a benign ``none``. Per-ROW malformation is different and still skips the row:
    one unreadable event among ten is not a venue-wide failure.
    """
    if not isinstance(payload, dict):
        logger.warning(f"Polymarket returned a {type(payload).__name__} payload, not the documented object")
        return None

    out: list[MarketMatch] = []
    for rank, event in enumerate(payload.get("events") or []):
        if len(out) >= width:
            break
        if isinstance(event, dict):
            out.append(_polymarket_event_match(event, rank))
    if out:
        return out

    for rank, market in enumerate(payload.get("markets") or []):
        if len(out) >= width:
            break
        if isinstance(market, dict):
            out.append(_polymarket_market_match(market, rank))
    return out


async def polymarket_search(session: Any, query: str, *, width: int) -> list[MarketMatch] | None:
    """Search Polymarket for one query. ``None`` when the fetch failed OR the shape changed.

    ``events_status=active`` is load-bearing, not tidiness: nothing downstream filters on
    close date any more, so without it ``Ethereum`` puts 7 closed events out of 10 into the
    pool.

    The None-vs-``[]`` split is the whole degradation contract: a retry-exhausted 503 would
    otherwise arrive at the caller as an ordinary empty result and publish as a benign
    ``none``. A 200 whose top-level shape is not Gamma's object is the same class of loss, and
    the parser reports it the same way.
    """
    payload = await http_get_with_backoff(
        session,
        POLYMARKET_SEARCH_URL,
        {"q": query, "limit_per_type": str(VENUE_SEARCH_LIMIT), "events_status": "active"},
        max_attempts=POLYMARKET_MAX_ATTEMPTS,
        label=f"Polymarket q={query[:40]!r}",  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # log-label truncation
    )
    if payload is None:
        return None
    return parse_polymarket_matches(payload, width=width)
