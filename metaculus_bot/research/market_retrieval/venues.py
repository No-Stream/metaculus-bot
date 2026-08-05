"""The four venue fetch/parse paths: raw venue JSON in, ``MarketMatch`` rows out.

Every path here serves a pipeline that hands its WHOLE candidate pool to one ranking call,
which is what makes the four decisions below the shape they are. Recall is generation's job,
selection is the ranker's, and no path here fuzzy-selects, score-floors or width-caps on the
venue's behalf.

- **Kalshi pulls the COMPLETE open-events catalogue** (~9,762 events, ~49 pages) and streams
  each page through a TIERED projection, so peak memory tracks the retained fields rather
  than the ~3 MB/page raw body — 49 pages of raw JSON is ~150 MB, and the caller holds the
  result for 6h in the process that runs forecasters. Each event's first nested market keeps
  the full field set and every later one keeps only ``close_time`` and ``status``, the only
  two fields anything reads across nested markets: 85.5 MB down to 43.7 MB on the frozen
  universe, with byte-identical consumer output.
- **Kalshi's event-level derivations read EVERY nested market**: ``close_time`` is the MAX
  over them and ``is_resolved`` requires ALL of them resolved. Reading ``nested[0]`` instead
  misclassifies ~305 live event families as RESOLVED and ~72 settled ones as open on the
  frozen universe, and both fields are now a rendered column and a ranker input. Liquidity
  and the price legs DO stay on ``nested[0]``, deliberately: they describe one tradeable
  contract, and there is no defensible way to add up four strikes' worth of open interest.
- **Polymarket sends ``events_status=active``**, and it is load-bearing rather than tidy:
  nothing downstream filters on close date any more (the pipeline passes ``as_of=None``), so
  without the parameter ``Ethereum`` puts 7 closed events out of 10 straight into the pool.
- **PredictIt parses ``dateEnd`` into ``close_time``, renders contract names, and quotes no
  price.** The whole ~197-market universe reaches the pool, so there is no per-question query
  to select a contract with and pricing an arbitrary one would misreport the market; the
  contract NAMES are most of a PredictIt market's semantic content, since its own title is
  often just "Which party will win ...".

Two structural rules hold throughout. **Every function takes ``session`` as a parameter** —
the session factory stays in the seam module, where four test files patch it. And **nothing
here reads or writes a cache**: the caches are module globals in the seam module that the
orchestrator imports by name, so the caller owns the TTL and this module stays pure I/O plus
parse. That is why the catalogue pull reports ``complete`` rather than writing
``_KALSHI_CACHE`` itself.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiohttp
import ijson

from metaculus_bot.constants import (
    KALSHI_CATALOGUE_WALL_TIMEOUT,
    KALSHI_PAGE_SLEEP_S,
    KALSHI_PREFETCH_EVENT_LIMIT,
    KALSHI_PREFETCH_MAX_PAGES,
)
from metaculus_bot.research.http_fetch import read_body_snippet
from metaculus_bot.research.market_retrieval.http import (
    HTTP_RETRY_BACKOFF_SECS,
    http_get_with_backoff,
    parse_iso,
    parse_iso_guarded,
    safe_float,
    safe_int,
    settlement_sources,
)
from metaculus_bot.research.market_retrieval.types import MarketMatch, _FetchTally

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoints and per-venue bounds
# ---------------------------------------------------------------------------

POLYMARKET_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
MANIFOLD_MARKET_URL = "https://api.manifold.markets/v0/market"
KALSHI_EVENTS_URL = "https://api.elections.kalshi.com/trade-api/v2/events"
PREDICTIT_URL = "https://www.predictit.org/api/marketdata/all/"

POLYMARKET_MAX_ATTEMPTS = 2
MANIFOLD_MAX_ATTEMPTS = 2
PREDICTIT_MAX_ATTEMPTS = 2

# The venue-search endpoints' own `limit`. NOT a retrieval width: the pool's per-venue width
# is generation's business, and the parsers take theirs as an explicit argument, so no hard
# slice in a parser can silently cap a wider pool — a `payload[:10]` left in one would make
# "width 60" mean 10 per query with nothing to see at the call site.
VENUE_SEARCH_LIMIT = 10

# Kalshi catalogue streaming bounds. The size ceiling is a last-resort guard against a
# runaway or compressed-bomb body, sitting well above a real page; a breach is loud and
# refuses to cache rather than silently truncating.
KALSHI_PAGE_MAX_BYTES = 64 * 1024 * 1024
KALSHI_PAGE_MAX_ATTEMPTS = 2
_KALSHI_READ_CHUNK_BYTES = 65536
# Mirrors the default retryable set (403 is Kalshi's rate-limit shape); >= 500 also retries.
_KALSHI_RETRYABLE_STATUSES = frozenset({403, 429, 500, 502, 503, 504})
# ijson structural events, i.e. the ones carrying no scalar `value` to keep.
_IJSON_CONTAINER_EVENTS = frozenset({"start_map", "map_key", "end_map", "start_array", "end_array"})

# The projection. Everything outside these tuples (plus `settlement_sources`) is dropped as the
# page streams in, which is the whole point: a raw market object carries ~38 keys and the
# catalogue is held for 6h in the process that runs forecasters.
KALSHI_EVENT_FIELDS: tuple[str, ...] = ("event_ticker", "series_ticker", "title", "sub_title")
# The full nested-market set, kept for the FIRST nested market of each event. It stays the
# accurate union of what any consumer can read off a market object.
KALSHI_MARKET_FIELDS: tuple[str, ...] = (
    "rules_primary",
    "close_time",
    "yes_bid_dollars",
    "yes_ask_dollars",
    "last_price_dollars",
    "notional_value_dollars",
    "volume_fp",
    "open_interest_fp",
    "volume_24h_fp",
    "status",
)
# Nested markets AFTER the first keep only these two, because only these two are ever read
# across nested markets — `close_time` by the max-over-nested derivation and `status` by the
# all-resolved one. Every other field is read exclusively off `nested[0]` (`kalshi_event_match`
# and `kalshi_event_rules`), so on the frozen universe 69,207 of 78,969 nested markets' worth of
# price/rules fields would be dead weight: measured through the real collector, 85.5 MB untiered
# versus 43.7 MB tiered, with `rules_primary` on the tail alone accounting for 13.2 MB. At the
# `KALSHI_PREFETCH_EVENT_LIMIT` guard that is ~175 MB versus ~90 MB.
#
# A NEW field has to pick a tier deliberately: put it here only if something reads it off a
# nested market other than the first. The saving requires genuinely OMITTING the keys — seeding
# the tail with all ten set to None recovers only 35.7 of the 41.8 MB. `sys.intern` is NOT the
# alternative: it saves 34.9 MB but PEP-683 makes interned strings effectively permanent, so a
# 6h cache refresh would never reclaim them.
KALSHI_NESTED_TAIL_FIELDS: tuple[str, ...] = ("close_time", "status")

# A Kalshi market whose status is one of these has settled. Load-bearing on the EVENT too: an
# event is resolved only when every nested market is.
KALSHI_RESOLVED_STATUSES: frozenset[str] = frozenset({"settled", "finalized", "closed"})

# PredictIt bundles several binary contracts per market, and the contract names are most of
# its semantic content. Eight covers a crowded primary field without letting one market's
# ballot dominate the ranker prompt.
PREDICTIT_CONTRACTS_RENDERED = 8

# Rules-text retention at parse time. Generous here and tightened per venue in the ranker
# prompt: this bound only stops a pathological row from being carried around.
RULES_TEXT_MAX_CHARS = 2000


# ---------------------------------------------------------------------------
# Kalshi: the full open-events catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CataloguePull:
    """The outcome of one paginated Kalshi catalogue fetch.

    ``token`` is a provider-diagnostics source token, empty exactly when the pull finished
    clean. ``complete`` says whether pagination exited on its own terms — the cursor was
    exhausted — and therefore whether the result is safe to cache for the TTL. Stopping at
    ``max_pages`` or ``event_limit`` with a cursor still open is NOT complete: the catalogue is
    truncated, so pinning it for 6h would serve a short universe to every later question.

    The pair is why this does NOT use the ``None``-vs-``[]`` contract the leaf fetchers use:
    a pull that lost page 30 of 49 still holds 29 pages of usable catalogue, and neither
    ``None`` (throwing them away) nor ``[]`` (claiming an empty exchange) says that.
    ``events == [] and token == ""`` is the one genuinely empty catalogue.
    """

    events: list[dict[str, Any]]
    token: str
    tally: _FetchTally
    complete: bool


def _kalshi_page_collector(kept: list[dict[str, Any]], state: dict[str, Any]) -> Any:
    """An ijson push target that projects an ``/events`` page down as it streams.

    Parses at the EVENT level rather than materializing each event object, for two reasons.
    Peak memory tracks the kept fields (~14 per event) instead of a ~3 MB page. And the parse
    events expose the TOP-LEVEL shape, which item-level extraction cannot see: an HTTP 200
    carrying ``{"events": null}`` or ``{"error": "temporarily unavailable"}`` yields zero
    items, exactly like a legitimately empty catalogue, and without ``saw_events_array`` that
    lands in the 6h cache as a valid empty index.

    ``state`` carries ``saw_events_array`` and ``cursor`` back out, because the caller needs
    both after the stream has closed.
    """

    @ijson.coroutine
    def _collect():  # noqa: ANN202  # untyped ijson push-parser target
        event: dict[str, Any] | None = None
        market: dict[str, Any] | None = None
        source: dict[str, Any] | None = None
        while True:
            prefix, parse_event, value = yield
            if prefix == "events":
                if parse_event == "start_array":
                    state["saw_events_array"] = True
                continue
            if prefix == "cursor" and parse_event == "string":
                state["cursor"] = value
                continue
            if prefix == "events.item":
                if parse_event == "start_map":
                    event = {field: None for field in KALSHI_EVENT_FIELDS}
                    event["settlement_sources"] = []
                    event["markets"] = []
                elif parse_event == "end_map":
                    # A ticketless event is unusable downstream — the ticker is the pool's
                    # dedup key and the settlement index's key — so it is dropped here
                    # rather than carried as a row nothing can reference.
                    if event is not None and event.get("event_ticker"):
                        kept.append(event)
                    event = None
                continue
            if event is None:
                continue
            if prefix == "events.item.settlement_sources.item":
                if parse_event == "start_map":
                    source = {"name": None, "url": None}
                elif parse_event == "end_map":
                    if source is not None:
                        event["settlement_sources"].append(source)
                    source = None
                continue
            if source is not None:
                if prefix == "events.item.settlement_sources.item.name":
                    source["name"] = value
                elif prefix == "events.item.settlement_sources.item.url":
                    source["url"] = value
                continue
            if prefix == "events.item.markets.item":
                if parse_event == "start_map":
                    # Tiered off the list already being built, which reads the index for free:
                    # the first nested market gets the full set, every later one the tail.
                    fields = KALSHI_MARKET_FIELDS if not event["markets"] else KALSHI_NESTED_TAIL_FIELDS
                    market = {field: None for field in fields}
                elif parse_event == "end_map":
                    if market is not None:
                        event["markets"].append(market)
                    market = None
                continue
            if market is not None:
                key = prefix.removeprefix("events.item.markets.item.")
                # The pre-seeded dict IS the membership set, so the tier decided at `start_map`
                # decides what this market retains without a second tuple lookup.
                if key in market and parse_event not in _IJSON_CONTAINER_EVENTS:
                    market[key] = value
                continue
            key = prefix.removeprefix("events.item.")
            if key in KALSHI_EVENT_FIELDS and parse_event not in _IJSON_CONTAINER_EVENTS:
                event[key] = value

    return _collect()


async def _kalshi_fetch_events_page(
    session: Any, params: dict[str, str], *, timeout_s: float
) -> tuple[list[dict[str, Any]] | None, str | None, str, bool]:
    """One streamed ``/events`` page. Returns ``(events, cursor, reason, retryable)``.

    ``retryable`` is True for transport failures, transient statuses, and a missing
    ``events`` array (the "temporarily unavailable" 200 is a transient upstream state and
    such a body is tiny). It is False for the size ceiling and for malformed JSON, where a
    second identical request only burns budget — and for a 429: re-asking a rate limiter
    0.5s later is not a retry, it is a second violation, so the pull stops early and refuses
    to cache.
    """
    kept: list[dict[str, Any]] = []
    state: dict[str, Any] = {"saw_events_array": False, "cursor": None}

    timeout = aiohttp.ClientTimeout(total=timeout_s, sock_read=timeout_s)
    total = 0
    try:
        async with session.get(KALSHI_EVENTS_URL, params=params, timeout=timeout) as resp:
            if resp.status != 200:
                snippet = await read_body_snippet(resp)
                logger.warning(f"Kalshi events HTTP {resp.status}: {snippet}")
                retryable = resp.status != 429 and (resp.status in _KALSHI_RETRYABLE_STATUSES or resp.status >= 500)
                return None, None, f"error(http_{resp.status})", retryable
            parser = ijson.parse_coro(_kalshi_page_collector(kept, state))
            try:
                async for chunk in resp.content.iter_chunked(_KALSHI_READ_CHUNK_BYTES):
                    total += len(chunk)
                    if total > KALSHI_PAGE_MAX_BYTES:
                        logger.warning(
                            f"Kalshi events page exceeded safety ceiling "
                            f"({total} bytes read > {KALSHI_PAGE_MAX_BYTES}); aborting stream"
                        )
                        return None, None, "dropped(size_cap)", False
                    parser.send(chunk)
                parser.close()
            except ijson.JSONError as exc:
                logger.warning(f"Kalshi events stream parse failed: {type(exc).__name__}: {exc}")
                return None, None, "error(parse)", False
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        logger.warning(f"Kalshi events transient error: {type(exc).__name__}: {exc}")
        return None, None, f"error({type(exc).__name__})", True  # noqa: ASYNC910

    if not state["saw_events_array"]:
        logger.warning(
            f"Kalshi events payload carried no top-level 'events' array ({total} bytes); "
            f"treating as a lost catalogue, not an empty one"
        )
        return None, None, "error(no_events_array)", True
    return kept, state["cursor"], "", False


async def kalshi_prefetch_events(
    session: Any,
    *,
    event_limit: int = KALSHI_PREFETCH_EVENT_LIMIT,
    max_pages: int = KALSHI_PREFETCH_MAX_PAGES,
    page_sleep_s: float = KALSHI_PAGE_SLEEP_S,
    wall_timeout: float = KALSHI_CATALOGUE_WALL_TIMEOUT,
) -> CataloguePull:
    """Paginate the COMPLETE open-events catalogue, projected down per page.

    ``/events?status=open&with_nested_markets=true`` rather than ``/markets``, which is
    dominated by sports-parlay rows. ``max_pages`` is the real bound and ``event_limit`` a
    runaway guard; the wall is wall-clock over the whole pull, retries included, so
    pagination can never overrun the snapshot's own timeout.

    A 429 stops pagination and reports the pull incomplete, so a rate-limited exchange can
    never be cached as a short catalogue for the TTL. The events accumulated up to a failure
    are still returned, so the question that paid for the partial pull keeps it — but
    ``complete=False`` tells the caller not to pin that partial list for 6h. Caching a
    truncated (often empty) list is how one transient blip on the first question starved every
    later question in the run.
    """
    params = {"status": "open", "limit": "200", "with_nested_markets": "true"}
    deadline = time.monotonic() + wall_timeout
    all_events: list[dict[str, Any]] = []
    cursor: str | None = None
    pages_ok = 0
    pages_failed = 0
    token = ""
    complete = True

    while len(all_events) < event_limit and pages_ok < max_pages:
        if time.monotonic() >= deadline:
            logger.warning(f"Kalshi catalogue wall of {wall_timeout}s exhausted after {pages_ok} page(s)")
            token, complete = "error(wall_timeout)", False
            pages_failed += 1
            break

        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor

        events: list[dict[str, Any]] | None = None
        reason = "error(unknown)"
        for attempt in range(KALSHI_PAGE_MAX_ATTEMPTS):
            attempt_budget = deadline - time.monotonic()
            if attempt_budget <= 0.0:
                reason = "error(wall_timeout)"
                break
            events, cursor_value, reason, retryable = await _kalshi_fetch_events_page(
                session, page_params, timeout_s=attempt_budget
            )
            if events is not None:
                cursor = cursor_value
                break
            if not retryable or attempt + 1 >= KALSHI_PAGE_MAX_ATTEMPTS:
                break
            if deadline - time.monotonic() <= HTTP_RETRY_BACKOFF_SECS:
                logger.warning(f"Kalshi events {reason}: retry budget exhausted; giving up")
                break
            logger.warning(
                f"Kalshi events {reason}; retry {attempt + 2}/{KALSHI_PAGE_MAX_ATTEMPTS} "
                f"after {HTTP_RETRY_BACKOFF_SECS:.2f}s"
            )
            await asyncio.sleep(HTTP_RETRY_BACKOFF_SECS)

        if events is None:
            token, complete = reason or "error(unknown)", False
            pages_failed += 1
            break

        pages_ok += 1
        all_events.extend(events)
        if not cursor or not events:
            break
        if page_sleep_s > 0:
            await asyncio.sleep(page_sleep_s)

    if complete and cursor and (pages_ok >= max_pages or len(all_events) >= event_limit):
        # A bound reached with the cursor still open means the catalogue is TRUNCATED, so the
        # pull is not complete: Kalshi halving its effective page size, or the open-events count
        # growing past `event_limit`, would otherwise pin a short universe for the 6h TTL and
        # report `ok(n)` with `fetch_ok=True`. Signal C cannot see it either — it skips any
        # source whose successful observation has entries > 0, and there is no size floor. The
        # token rides with the flag because the seam treats the token as the verdict, so a
        # token-only variant would put the two in contradiction.
        logger.warning(
            f"Kalshi catalogue stopped at a runaway bound (pages={pages_ok}, events={len(all_events)}) "
            f"with a cursor still open"
        )
        token, complete = "dropped(runaway_bound)", False
    logger.info(
        f"Kalshi catalogue: events={len(all_events)} pages={pages_ok} complete={complete} token={token or 'ok'}"
    )
    return CataloguePull(  # noqa: ASYNC910
        events=all_events, token=token, tally=_FetchTally(pages_ok, pages_failed), complete=complete
    )


def kalshi_usd_liquidity(market: dict[str, Any]) -> tuple[float | None, float | None]:
    """Convert a Kalshi market's contract counts to the USD the thresholds expect.

    Returns ``(volume_usd, open_interest_usd)``, either leg ``None`` when its count is absent
    — the distinction ``_liquidity_label`` needs to tell "upstream sent no field"
    (``no-liquidity-data``) from "a real zero" (``thin``).

    Field names and units are from the Kalshi OpenAPI spec (docs.kalshi.com, re-read
    2026-08-03). Two details drive the arithmetic:

    - ``volume_fp`` and ``open_interest_fp`` are ``FixedPointCount`` — "market volume in
      contracts" and "number of contracts bought ... disconsidering netting". They are
      COUNTS, not dollars. The bare names ``volume`` / ``open_interest`` appear nowhere in
      the spec and were absent from every archived payload, which is why reading them
      blanked the label on 100% of Kalshi rows.
    - The two counts convert with DIFFERENT multipliers, because they measure different
      things. Volume is turnover, so it scales by trade price. Open interest is collateral:
      both sides of a contract together post the full settlement value, so it scales by
      ``notional_value_dollars``. Using trade price for both would call a market holding
      119k contracts of open interest at $0.001 "thin" when ~$119k of capital is locked in
      it.

    ``liquidity_dollars`` is deliberately not consulted: upstream marks it deprecated and
    documents it to "always return 0.0000" (confirmed on 1,504 live markets and all 127
    archived rows). Scoring off it would wire in a constant zero.

    Three details here are UNPINNED by any test: the ``last_price_dollars`` fallback, the
    ``or 1.0`` notional default, and the ``if not price:`` truthiness (which routes a genuine
    0.0 midpoint to the last trade). "Simplifying" any of them blanks or misstates labels with
    a green suite.
    """
    volume = safe_float(market.get("volume_fp"))
    open_interest = safe_float(market.get("open_interest_fp"))
    if volume is None and open_interest is None:
        return None, None

    # Kalshi prices every market in dollars-per-contract. Prefer the live bid/ask midpoint
    # (what the row already reports as its implied probability) and fall back to the last
    # trade, so a market with quotes but no trades still converts.
    bid, ask = safe_float(market.get("yes_bid_dollars")), safe_float(market.get("yes_ask_dollars"))
    price = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
    if not price:
        price = safe_float(market.get("last_price_dollars"))
    notional = safe_float(market.get("notional_value_dollars")) or 1.0

    volume_usd = None if volume is None else volume * (price or 0.0)
    open_interest_usd = None if open_interest is None else open_interest * notional
    return volume_usd, open_interest_usd


def kalshi_event_rules(event: dict[str, Any]) -> str:
    """``rules_primary`` off the first nested market — the event's settlement clause.

    Every strike inside one event shares a settlement rule and differs only in the
    threshold, so ``nested[0]`` is representative here in a way it is not for close time or
    status.
    """
    nested = event.get("markets") or []
    if nested and isinstance(nested[0], dict):
        return (nested[0].get("rules_primary") or "")[:RULES_TEXT_MAX_CHARS]
    return ""


def kalshi_event_match(event: dict[str, Any], *, match_confidence: float, channel: str) -> MarketMatch | None:
    """One projected Kalshi event as a candidate row. None when it has no usable title.

    ``close_time`` is the MAX over nested markets and ``is_resolved`` requires ALL of them
    resolved. Prices and liquidity stay on ``nested[0]``, which describes one tradeable
    contract and is the only thing a single ``prob`` cell can honestly report.
    """
    title = event.get("title") or event.get("sub_title") or ""
    if not title:
        return None

    nested = [market for market in (event.get("markets") or []) if isinstance(market, dict)]
    first = nested[0] if nested else {}

    closes = [parsed for market in nested if (parsed := parse_iso(market.get("close_time") or ""))]
    is_resolved = bool(nested) and all(
        (market.get("status") or "").lower() in KALSHI_RESOLVED_STATUSES for market in nested
    )

    yes_bid = safe_float(first.get("yes_bid_dollars"))
    yes_ask = safe_float(first.get("yes_ask_dollars"))
    implied: float | None = None
    spread: float | None = None
    if yes_bid is not None and yes_ask is not None:
        implied = (yes_bid + yes_ask) / 2.0
        spread = yes_ask - yes_bid
    total_volume, open_interest = kalshi_usd_liquidity(first)

    event_ticker = str(event.get("event_ticker") or "")
    return MarketMatch(
        platform="kalshi",
        market_title=title,
        market_url=f"https://kalshi.com/markets/{event_ticker}" if event_ticker else "",
        implied_prob_yes=implied,
        bid=yes_bid,
        ask=yes_ask,
        spread=spread,
        volume_24h=safe_float(first.get("volume_24h_fp")),
        close_time=max(closes) if closes else None,
        is_resolved=is_resolved,
        match_confidence=match_confidence,
        raw_rules=kalshi_event_rules(event),
        total_volume=total_volume,
        open_interest=open_interest,
        venue_market_id=event_ticker,
        retrieval_channel=channel,
        sub_title=str(event.get("sub_title") or ""),
        settlement_sources=settlement_sources(event.get("settlement_sources")),
    )


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------


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


def _polymarket_event_match(event: dict[str, Any], rank: int) -> MarketMatch:
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
    markets = event.get("markets") or []
    if markets and isinstance(markets[0], dict):
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
    )


def _polymarket_market_match(market: dict[str, Any], rank: int) -> MarketMatch:
    slug = str(market.get("slug") or "")
    total_volume = safe_float(market.get("volumeNum"))
    if total_volume is None:
        total_volume = safe_float(market.get("volume"))
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


def parse_polymarket_matches(payload: Any, *, width: int) -> list[MarketMatch]:
    """Parse a Gamma public-search response into candidate rows, venue-rank order.

    Gamma returns ``{events: [...], markets: [...]}``; events are preferred and the
    top-level markets list is the fallback for when there were none. BOTH branches read
    ``openInterest``, because both are live paths and the field sits in a different place in
    each.

    ``match_confidence`` is the venue's OWN rank inverted, never a local re-score: the point
    of the venue-index channel is to stop trying to out-rank the index.
    """
    if not isinstance(payload, dict):
        logger.warning("Polymarket returned non-dict payload")
        return []

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
    """Search Polymarket for one query. ``None`` when the fetch itself failed.

    ``events_status=active`` is load-bearing, not tidiness: nothing downstream filters on
    close date any more, so without it ``Ethereum`` puts 7 closed events out of 10 into the
    pool.

    The None-vs-``[]`` split is the whole degradation contract: a retry-exhausted 503 would
    otherwise arrive at the caller as an ordinary empty result and publish as a benign
    ``none``.
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


# ---------------------------------------------------------------------------
# Manifold
# ---------------------------------------------------------------------------


def _walk_tiptap_text(node: Any) -> list[str]:
    """Depth-first collect every leaf ``text`` string from a TipTap/ProseMirror doc."""
    out: list[str] = []
    if isinstance(node, dict):
        text = node.get("text")
        if isinstance(text, str) and text:
            out.append(text)
        content = node.get("content")
        if isinstance(content, list):
            for child in content:
                out.extend(_walk_tiptap_text(child))
    elif isinstance(node, list):
        for child in node:
            out.extend(_walk_tiptap_text(child))
    return out


def manifold_rules_text(market: dict[str, Any]) -> str:
    """A Manifold market's rules text: ``textDescription``, else the flattened doc, else "".

    Deliberately NO fall back to the question title: that renders a candidate line whose
    ``rules:`` segment repeats its own title, spending ranker tokens on nothing. The
    enrichment hook is what fills real description text in, and a blank here is the honest
    signal that it has not run or found nothing.
    """
    text_description = market.get("textDescription")
    if isinstance(text_description, str) and text_description.strip():
        return text_description[:RULES_TEXT_MAX_CHARS]

    description = market.get("description")
    if isinstance(description, dict):
        collected = _walk_tiptap_text(description)
        return " ".join(collected)[:RULES_TEXT_MAX_CHARS] if collected else ""
    if isinstance(description, str) and description.strip():
        return description[:RULES_TEXT_MAX_CHARS]
    return ""


def parse_manifold_matches(payload: Any, *, width: int) -> list[MarketMatch]:
    """Parse a Manifold search response into candidate rows, venue-rank order.

    Close time and every liquidity field are read off EACH SEARCH ROW. That is the fix for
    the blank close/liquidity cells the bake-off measured on 52 of 94 Manifold rows: those
    came from looking each row up in a cached universe it was never in.
    """
    if not isinstance(payload, list):
        logger.warning("Manifold returned non-list payload")
        return []

    out: list[MarketMatch] = []
    for rank, market in enumerate(payload):
        if len(out) >= width:
            break
        if not isinstance(market, dict):
            continue
        slug = market.get("slug") or ""
        creator = market.get("creatorUsername") or ""
        url = f"https://manifold.markets/{creator}/{slug}" if slug and creator else (market.get("url") or "")

        close_ms = market.get("closeTime")
        close_time: datetime | None = None
        if isinstance(close_ms, (int, float)):
            try:
                close_time = datetime.fromtimestamp(float(close_ms) / 1000.0, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                close_time = None

        out.append(
            MarketMatch(
                platform="manifold",
                market_title=market.get("question") or "",
                market_url=url,
                implied_prob_yes=safe_float(market.get("probability")),
                bid=None,
                ask=None,
                spread=None,
                volume_24h=safe_float(market.get("volume24Hours")),
                close_time=close_time,
                is_resolved=bool(market.get("isResolved")),
                match_confidence=100.0 - rank,
                raw_rules=manifold_rules_text(market),
                total_volume=safe_float(market.get("volume")),
                liquidity=safe_float(market.get("totalLiquidity")),
                num_bettors=safe_int(market.get("uniqueBettorCount")),
                venue_market_id=str(market.get("id") or ""),
                retrieval_channel="venue_search",
            )
        )
    return out


async def manifold_search(session: Any, query: str, *, width: int) -> list[MarketMatch] | None:
    """Search Manifold for one query. ``None`` when the fetch itself failed.

    ``contractType=BINARY`` costs a known ~30% recall ceiling — multi-outcome markets are
    invisible — and is kept with eyes open, because flipping to ALL needs a multi-outcome
    render path: the search response carries no per-answer data, so every row would need a
    detail GET, which the enrichment hook below makes nearly free later. Tracked in FUTURE.md.
    """
    payload = await http_get_with_backoff(
        session,
        MANIFOLD_SEARCH_URL,
        {"term": query, "contractType": "BINARY", "limit": str(VENUE_SEARCH_LIMIT)},
        max_attempts=MANIFOLD_MAX_ATTEMPTS,
        retryable_statuses=(429, 500, 502, 503, 504),
        label=f"Manifold q={query[:40]!r}",  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # log-label truncation
    )
    if payload is None:
        return None
    return parse_manifold_matches(payload, width=width)


async def manifold_market_detail(session: Any, market_id: str) -> dict[str, Any] | None:
    """One market's full record, for the ``textDescription`` the search listing omits.

    ``None`` on any failure, which the enrichment hook treats as "leave the row title-only":
    a lost detail GET costs no recall, only rules text. One attempt, no retry — 60 of these
    fire per question under a 10s fan-out wall, and a row's description is not worth
    spending that wall on twice.
    """
    payload = await http_get_with_backoff(
        session,
        f"{MANIFOLD_MARKET_URL}/{market_id}",
        {},
        max_attempts=1,
        retryable_statuses=(429, 500, 502, 503, 504),
        label=f"Manifold detail id={market_id[:40]!r}",  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # log-label truncation
    )
    return payload if isinstance(payload, dict) else None


# ---------------------------------------------------------------------------
# PredictIt
# ---------------------------------------------------------------------------


async def predictit_prefetch(session: Any) -> list[dict[str, Any]] | None:
    """Fetch the full PredictIt market dump. ``None`` when the fetch itself failed.

    One unpaginated ``/marketdata/all/`` GET, ~197 markets, no auth and no query param — so
    "prefetch the universe" is one request's worth of work, and the ranked design shows all
    of it rather than fuzzy-selecting 20.
    """
    payload = await http_get_with_backoff(
        session,
        PREDICTIT_URL,
        {},
        max_attempts=PREDICTIT_MAX_ATTEMPTS,
        label="PredictIt prefetch",
    )
    if payload is None:
        return None
    if not isinstance(payload, dict):
        logger.warning("PredictIt returned non-dict payload")
        return []

    markets = payload.get("markets")
    if not isinstance(markets, list):
        logger.warning("PredictIt payload missing 'markets' list")
        return []
    return [market for market in markets if isinstance(market, dict)]


def predictit_contract_names(market: dict[str, Any]) -> str:
    """The contract list for one PredictIt market, comma-joined.

    PredictIt bundles one binary contract per outcome, and the contract NAMES are most of
    the market's semantic content — its own title is often just "Which party will win ...".
    So they are both the fuzzy-scoring text and the rules text the ranker reads. One helper
    for both, so the two cannot describe the same market differently.
    """
    contracts = market.get("contracts")
    if not isinstance(contracts, list):
        return ""
    names = [
        str(contract.get("name") or "")
        for contract in contracts[:PREDICTIT_CONTRACTS_RENDERED]  # noqa: HARNESS-SCAN-EXEMPT-subsampling
        if isinstance(contract, dict)
    ]
    return ", ".join(name for name in names if name)


def predictit_market_match(market: dict[str, Any], *, match_confidence: float, channel: str) -> MarketMatch | None:
    """One PredictIt market as a candidate row. None when it has no usable name.

    One row per market, matching Kalshi's one-row-per-event, and NO price. The whole
    ~197-market universe reaches the pool, so there is no per-question query to pick a
    contract with; relevance is the ranker's job, and quoting one arbitrary contract's price
    on a multi-contract ballot would misreport the market. Rendered close coverage therefore
    reads ~26% — the universe's own rate, not the ~64% a fuzzy pre-filter selects for.

    Both event-level derivations read EVERY contract rather than ``contracts[0]``, mirroring
    ``kalshi_event_match`` four functions up: a market whose contracts are ``[Closed, Open]``
    is live, and reading the first one makes the verdict depend on the order untrusted
    external JSON happened to arrive in — reverse the list and the same market flips. Both
    fields are load-bearing now: ``status`` is a rendered column AND a ranker prompt signal,
    telling the model a RESOLVED price is a realized outcome rather than a forecast. Derived
    from the contracts rather than the market-level ``status`` field (present on 197/197 live
    markets, ignored here) so both venues answer the question the same way, in one place.
    """
    name = market.get("name") or ""
    short_name = market.get("shortName") or ""
    if not name and not short_name:
        return None

    raw_contracts = market.get("contracts")
    # Filtered to dicts ONCE and gated on the filtered list. Folding the isinstance check inside
    # the `all()` instead would take `all()` over an EMPTY sequence when every contract is a
    # non-dict — reporting RESOLVED where an unusable contracts block must stay unresolved.
    contracts = [entry for entry in raw_contracts if isinstance(entry, dict)] if isinstance(raw_contracts, list) else []
    statuses = [(entry.get("status") or "").lower() for entry in contracts]
    # A missing status is not evidence of settlement, so it reads as open.
    is_resolved = bool(statuses) and all(status not in ("", "open") for status in statuses)
    closes = [parsed for entry in contracts if (parsed := parse_iso_guarded(entry.get("dateEnd"))) is not None]

    contract_names = predictit_contract_names(market)
    market_id = market.get("id")
    return MarketMatch(
        platform="predictit",
        market_title=name or short_name,
        market_url=market.get("url") or "",
        implied_prob_yes=None,
        bid=None,
        ask=None,
        spread=None,
        volume_24h=None,
        close_time=max(closes) if closes else None,
        is_resolved=is_resolved,
        match_confidence=match_confidence,
        raw_rules=f"contracts: {contract_names}" if contract_names else "",
        venue_market_id="" if market_id is None else str(market_id),
        retrieval_channel=channel,
    )
