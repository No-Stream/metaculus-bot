"""Kalshi: the COMPLETE open-events catalogue, streamed and projected, then read per family.

Two decisions shape everything here.

- **The pull is the whole catalogue** (~9,762 events, ~49 pages), streamed so each page passes
  through a TIERED projection and peak memory tracks the retained fields rather than the ~3
  MB/page raw body — 49 pages of raw JSON is ~150 MB, and the caller holds the result for 6h in
  the process that runs forecasters. Only the rules text is head-only; see
  ``KALSHI_NESTED_TAIL_FIELDS``.
- **EVERY event-level derivation reads EVERY nested market.** An event is a threshold FAMILY
  (86.5% of the catalogue is multi-strike), so ``close_time`` is the MAX over its strikes,
  ``is_resolved`` requires ALL of them resolved, and the liquidity legs SUM each live strike's own
  dollar conversion. Reading ``nested[0]`` instead misclassifies ~305 live families as RESOLVED,
  ~72 settled ones as open, and mislabels the liquidity of 1,634 events. ``implied_prob_yes`` and
  its bid/ask/spread legs are the one thing that CANNOT be aggregated: a family has no single
  probability, so they populate only for a one-tradeable-strike event and are ``None`` otherwise.
- **A family with no single probability shows EVERY strike's, rather than hiding all of them.**
  That is the 2026-08-05 reversal. Withholding the price was right about the arithmetic and wrong
  about the remedy: 8,103 of 9,417 frozen-universe events rendered a bare ``-`` in the ``prob``
  column while the prices sat one field away, unshown. ``kalshi_strike_children`` now renders one
  ``MarketChild`` sub-row per tradeable strike, each with its own ``yes_sub_title`` and its own
  midpoint, which is what makes the family-level ``None`` a statement about the family rather than
  a loss of the data. A one-strike family is untouched — it still quotes that strike directly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
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
    parse_iso,
    safe_float,
    settlement_sources,
)
from metaculus_bot.research.market_retrieval.types import MarketChild, MarketMatch, _FetchTally
from metaculus_bot.research.market_retrieval.venues._shared import RULES_TEXT_MAX_CHARS

logger = logging.getLogger(__name__)

KALSHI_EVENTS_URL = "https://api.elections.kalshi.com/trade-api/v2/events"

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

# The projection. Everything outside these tuples (plus `settlement_sources`) is dropped as the page
# streams in, which is the whole point: a raw market object carries ~38 keys and the catalogue is held
# for 6h in the process that runs forecasters.
KALSHI_EVENT_FIELDS: tuple[str, ...] = ("event_ticker", "series_ticker", "title", "sub_title")
# The full nested-market set, kept for the FIRST nested market of each event. It stays the accurate
# union of what any consumer can read off a market object.
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
    # The per-strike LABEL, kept only since 2026-08-05, when a multi-strike family started
    # rendering one sub-row per strike and each row needed a title of its own. `yes_sub_title` is
    # short and self-describing ("Republican Party", "Tie", "Before Nov 1, 2026" — 3-18 chars on
    # the committed family fixture, 5/5 present) and reads as the ladder rung it is under the
    # event's own title. The strike's `title` is deliberately NOT kept: it is the full binary
    # question ("Will the difference between the number of Republican governors and ...", ~120
    # chars), so three siblings truncate to the same 80 chars and render indistinguishable.
    # `ticker` is the guaranteed-present fallback — opaque, but it is the strike's primary key, and
    # a row labelled `KXGOVWINS-27JAN01-R1` beats a row labelled nothing.
    "yes_sub_title",
    "ticker",
)
# Nested markets AFTER the first keep everything EXCEPT `rules_primary`, DERIVED by subtraction so a
# new field joins the tail by default. Every field but the rules text IS read across nested markets:
# `close_time` by the max-over-nested derivation, `status` by the all-resolved one and by
# `kalshi_tradeable_strikes`, and the money fields by the per-strike conversion the liquidity sum runs
# over the family (`volume_24h_fp` included — it rides with whichever strike the price legs quote,
# often not the first). `rules_primary` earns head-only status twice over: every strike shares one
# settlement rule, so tail copies say nothing new, and it is by far the largest field.
#
# The tail was `("close_time", "status")` until 2026-08-05, when the liquidity legs moved off
# `nested[0]` onto a family sum and needed the money fields on every strike. Measured through the real
# collector on the frozen universe (9,417 events / 72,479 nested markets): 75.6 MB untiered, 39.1 MB
# at the old tail, 64.0 MB now — correctness re-spent 24.9 of the 36.5 MB the tiering saved, and
# `rules_primary`'s 11.6 MB is what the tier still buys. At the `KALSHI_PREFETCH_EVENT_LIMIT` runaway
# guard that is ~136 MB rather than ~83 MB. Worth paying: the cheaper tail was cheap because it fed a
# derivation that mislabelled 1,634 events. The saving requires genuinely OMITTING the key — seeding
# the tail with every field set to None recovers only 35.7 of the 41.8 MB the omission does.
#
# `yes_sub_title` + `ticker` (the strike-children labels) add an estimated 8.6 MB on top, taking the
# projection to ~72.6 MB. That figure is an EXTRAPOLATION, not a collector run like the ones above:
# the two strings cost a measured 119 bytes per nested market on the committed family fixture
# (`sys.getsizeof`, and the projected dict absorbs two more keys with no resize), times 72,479
# markets. The mechanism bounds it — one short string each, on every strike — so the real figure
# tracks average label length rather than anything that can blow up.
# `sys.intern` is NOT the alternative: it saves 34.9 MB but PEP-683 makes interned strings effectively
# permanent, so a 6h refresh never reclaims them.
KALSHI_NESTED_HEAD_ONLY_FIELDS: frozenset[str] = frozenset({"rules_primary"})
KALSHI_NESTED_TAIL_FIELDS: tuple[str, ...] = tuple(
    field for field in KALSHI_MARKET_FIELDS if field not in KALSHI_NESTED_HEAD_ONLY_FIELDS
)

# A Kalshi market whose status is one of these has settled. Load-bearing on the EVENT twice over: an
# event is resolved only when every nested market is, and its money/price legs read only the rest.
KALSHI_RESOLVED_STATUSES: frozenset[str] = frozenset({"settled", "finalized", "closed"})


@dataclass(frozen=True, slots=True)
class CataloguePull:
    """The outcome of one paginated Kalshi catalogue fetch.

    ``token`` is a provider-diagnostics source token, empty exactly when the pull finished
    clean. ``complete`` says whether pagination exited on its own terms — the cursor was
    exhausted — and therefore whether the result is safe to cache for the TTL. Stopping at
    ``max_pages`` or ``event_limit`` with a cursor still open is NOT complete, and neither is
    an empty page that still handed back a token: in both cases the catalogue is truncated, so
    pinning it for 6h would serve a short universe to every later question.

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

    An EMPTY page whose cursor is still open takes the same route. Only an exhausted cursor
    ends pagination on its own terms; a page that returned no rows while still handing back a
    token has stopped short of the catalogue, and reporting that as complete would cache the
    truncation for the TTL behind a green ``ok(n)``. The runaway-bound re-check below cannot
    see it either — that fires only on ``max_pages`` / ``event_limit``, and an empty page
    reaches neither.
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
        if not cursor:
            break
        if not events:
            # A page with rows-of-zero and a token still open has stopped short of the
            # catalogue rather than finished it, so the pull is not cacheable. The token rides
            # with the flag because the seam reads the token as the verdict.
            logger.warning(
                f"Kalshi catalogue got an empty page with a cursor still open after {pages_ok} page(s); "
                f"treating the pull as truncated rather than finished"
            )
            token, complete = "dropped(empty_page)", False
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

    ONE strike's arithmetic. ``kalshi_event_usd_liquidity`` is what an event-level caller wants; a
    caller reaching here directly reports one threshold as a whole family.

    Returns ``(volume_usd, open_interest_usd)``, either leg ``None`` when its count is absent — the
    distinction ``_liquidity_label`` needs to tell "upstream sent no field" (``no-liquidity-data``)
    from "a real zero" (``thin``).

    Field names and units are from the Kalshi OpenAPI spec (docs.kalshi.com, re-read 2026-08-03).
    Two details drive the arithmetic:

    - ``volume_fp`` and ``open_interest_fp`` are ``FixedPointCount`` — "market volume in contracts"
      and "number of contracts bought ... disconsidering netting". They are COUNTS, not dollars. The
      bare names ``volume`` / ``open_interest`` appear nowhere in the spec and were absent from every
      archived payload, which is why reading them blanked the label on 100% of Kalshi rows.
    - The two counts convert with DIFFERENT multipliers, because they measure different things.
      Volume is turnover, so it scales by trade price. Open interest is collateral: both sides of a
      contract together post the full settlement value, so it scales by ``notional_value_dollars``.
      Using trade price for both would call a market holding 119k contracts of open interest at
      $0.001 "thin" when ~$119k of capital is locked in it.

    ``liquidity_dollars`` is deliberately not consulted: upstream marks it deprecated and documents
    it to "always return 0.0000" (confirmed on 1,504 live markets and all 127 archived rows).
    Scoring off it would wire in a constant zero.

    Three details here are UNPINNED by any test: the ``last_price_dollars`` fallback, the ``or 1.0``
    notional default, and the ``if not price:`` truthiness (which routes a genuine 0.0 midpoint to
    the last trade). "Simplifying" any of them blanks or misstates labels with a green suite.
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


def kalshi_tradeable_strikes(nested: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The strikes an event's money and price legs may be read off: the LIVE ones.

    One scope shared by both legs, because a settled strike misreports both the same way. A settled
    Kalshi market publishes an EMPTY book — ``yes_bid 0.0000`` / ``yes_ask 1.0000`` on 1,063 of the
    1,066 settled strikes inside open events, versus 44 of 71,413 live ones — so
    ``kalshi_usd_liquidity``'s midpoint reads a synthetic $0.50 for it rather than a price anybody
    traded at, overstating its own last trade by 71x at the 90th percentile and moving 9 events'
    rendered liquidity label on the frozen universe. Dropping settled strikes costs a family's
    traded history, which is real evidence, but a $0.50 stand-in for a $0.007 market is not the way
    to keep it.

    An all-settled family falls back to its own strikes rather than reporting nothing: the event is
    genuinely over, the render marks it ``RESOLVED``, and its realized figures are the honest
    answer. Zero frozen-universe events take that branch (a fully-settled family leaves the
    open-events catalogue), so it serves resolved rows arriving by other channels.
    """
    live = [market for market in nested if (market.get("status") or "").lower() not in KALSHI_RESOLVED_STATUSES]
    return live or list(nested)


def kalshi_event_usd_liquidity(nested: Sequence[dict[str, Any]]) -> tuple[float | None, float | None]:
    """The event FAMILY's liquidity: each tradeable strike's own USD conversion, summed.

    Each strike converts at its OWN price and notional, so this sums the per-market output of
    ``kalshi_usd_liquidity`` rather than adding counts and multiplying once — a threshold at $0.02
    and one at $0.90 turn the same contract count into wildly different dollars. The ``nested[0]``
    read this replaced on 2026-08-05 reported one strike's money as the family's, mislabelling
    1,634 of 9,417 frozen-universe events; the sum runs a median 2.6x its first strike's volume and
    140x at the 90th percentile.

    Either leg is ``None`` when NO strike published its count, preserving the "upstream sent no
    field" vs. "a real zero" distinction ``_liquidity_label`` needs; a strike publishing one leg and
    not the other contributes only the leg it has. No frozen-universe row ends up without a label.
    """
    legs = [kalshi_usd_liquidity(market) for market in kalshi_tradeable_strikes(nested)]
    volumes = [volume for volume, _ in legs if volume is not None]
    open_interests = [open_interest for _, open_interest in legs if open_interest is not None]
    return (sum(volumes) if volumes else None), (sum(open_interests) if open_interests else None)


def kalshi_strike_price(market: dict[str, Any]) -> float | None:
    """One strike's implied yes-probability: its bid/ask midpoint, or ``None`` if a side is missing.

    The single pricing rule for the venue, shared by the family's own price legs and by every
    strike sub-row, so a family and its children can never disagree about what a strike is worth.
    Deliberately no ``last_price_dollars`` fallback: a strike with no live quote on one side has no
    current price, and its last trade may be arbitrarily stale. (``kalshi_usd_liquidity`` does fall
    back to the last trade, because converting a contract COUNT to dollars at a stale price is far
    better than dropping the count.)
    """
    bid, ask = safe_float(market.get("yes_bid_dollars")), safe_float(market.get("yes_ask_dollars"))
    return (bid + ask) / 2.0 if (bid is not None and ask is not None) else None


def kalshi_strike_children(nested: Sequence[dict[str, Any]]) -> tuple[MarketChild, ...]:
    """Each tradeable strike as its own sub-row, OPEN-FIRST then PRICE-DESCENDING. ``()`` for a
    one-strike family.

    The multi-strike answer, and the exact complement of ``kalshi_price_strike``: children populate
    when that returns ``None`` and are empty when it returns a strike, so a family either quotes one
    price at the family level or quotes every strike's below it, never both and never neither.

    Scoped to ``kalshi_tradeable_strikes`` for the same reason the money legs are: a settled strike
    publishes an EMPTY book (``yes_bid 0.0000`` / ``yes_ask 1.0000`` on 1,063 of 1,066 settled
    strikes inside open events), so its midpoint is a synthetic $0.50 nobody traded at. Rendering
    that as a sub-row would be inventing a price, which is worse than the withholding this replaces.
    An all-settled family falls back to its own strikes there, so its realized prices do render.

    Price-descending — the same rule as ``manifold_answer_children`` — because the renderer
    truncates from the end and a strike family's children are a distribution over one question's
    outcomes: the rungs carrying the price mass are the forecast, whatever traded. Traded-size
    ordering let near-zero-probability rungs with open interest evict the priced brackets, which
    on a truncated family deleted exactly the region a forecaster extrapolates from (q45189: the
    six omitted rungs held 0.365 of price mass, all on one side). A strike with no live quote
    sorts with the zero-priced rungs, so the priced rows are the ones that survive the budget.
    The sort is STABLE, so equal-priced (and quoteless) strikes keep the catalogue's own
    (threshold) order.
    """
    tradeable = kalshi_tradeable_strikes(nested)
    if len(tradeable) < 2:
        return ()

    children: list[MarketChild] = []
    for market in tradeable:
        title = str(market.get("yes_sub_title") or market.get("ticker") or "")
        if not title:
            continue
        volume_usd, open_interest_usd = kalshi_usd_liquidity(market)
        children.append(
            MarketChild(
                title=title,
                implied_prob_yes=kalshi_strike_price(market),
                total_volume=volume_usd,
                open_interest=open_interest_usd,
                is_resolved=(market.get("status") or "").lower() in KALSHI_RESOLVED_STATUSES,
                close_time=parse_iso(market.get("close_time") or ""),
            )
        )
    children.sort(key=lambda child: (child.is_resolved, -(child.implied_prob_yes or 0.0)))
    return tuple(children)


def kalshi_price_strike(nested: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """The one strike whose price may be quoted as the event's, or ``None`` when there is none.

    A strike family has no single probability: quoting one threshold's price under the event's title
    says "72%" about a question the row's own title never asked. So a FAMILY-level price is reported
    only when the family has ONE tradeable strike — exactly when event and strike ask the same
    question. That is the 1,271 single-strike frozen-universe events (13.5%) plus 43 families whose
    siblings have all settled; the other 8,103 report ``None`` here.

    Reporting ``None`` no longer means the prices are LOST. Until 2026-08-05 it did — the row
    rendered a bare ``-`` and its 17 strike prices went unshown — and the reasoning for that
    (``prob`` has no room to name which threshold a number came from, so withheld beats wrong) was
    sound about the column and wrong about the remedy. ``kalshi_strike_children`` is the remedy: a
    row that has no single probability now carries one sub-row per strike, each naming its own
    threshold. The family-level ``None`` this function returns is what keeps the parent honest.

    The strike is found by STATUS, not position: on a collapsed family ``nested[0]`` is usually the
    settled one, so a positional read would quote a realized outcome as a live price.
    """
    tradeable = kalshi_tradeable_strikes(nested)
    return tradeable[0] if len(tradeable) == 1 else None


def kalshi_event_rules(event: dict[str, Any]) -> str:
    """``rules_primary`` off the first nested market — the event's settlement clause.

    Every strike inside one event shares a settlement rule and differs only in the threshold, so
    ``nested[0]`` is representative here in a way it is not for close time, status, or money — which
    is also why the projection keeps this field on the head alone.
    """
    nested = event.get("markets") or []
    if nested and isinstance(nested[0], dict):
        return (nested[0].get("rules_primary") or "")[:RULES_TEXT_MAX_CHARS]
    return ""


def kalshi_event_match(event: dict[str, Any], *, match_confidence: float, channel: str) -> MarketMatch | None:
    """One projected Kalshi event as a candidate row. None when it has no usable title.

    EVERY field here is derived across the whole strike family, because the row's title is the
    EVENT's: ``close_time`` is the max, ``is_resolved`` the all-resolved conjunction, the liquidity
    legs the per-strike USD sum, and the price legs one tradeable strike's or nothing
    (``kalshi_event_usd_liquidity`` and ``kalshi_price_strike`` carry the reasoning). ``volume_24h``
    rides with the price legs rather than the sum: it is a display column saying how much traded
    lately, so it belongs to the quoted strike and is ``None`` on the families that quote none.

    A family that quotes none carries ``children`` instead, one sub-row per strike, so the prices the
    family level cannot state are stated where they are true rather than dropped.
    """
    title = event.get("title") or event.get("sub_title") or ""
    if not title:
        return None

    nested = [market for market in (event.get("markets") or []) if isinstance(market, dict)]

    closes = [parsed for market in nested if (parsed := parse_iso(market.get("close_time") or ""))]
    is_resolved = bool(nested) and all(
        (market.get("status") or "").lower() in KALSHI_RESOLVED_STATUSES for market in nested
    )

    priced = kalshi_price_strike(nested) or {}
    yes_bid = safe_float(priced.get("yes_bid_dollars"))
    yes_ask = safe_float(priced.get("yes_ask_dollars"))
    implied = kalshi_strike_price(priced)
    spread = (yes_ask - yes_bid) if (yes_bid is not None and yes_ask is not None) else None
    total_volume, open_interest = kalshi_event_usd_liquidity(nested)

    event_ticker = str(event.get("event_ticker") or "")
    return MarketMatch(
        platform="kalshi",
        market_title=title,
        market_url=f"https://kalshi.com/markets/{event_ticker}" if event_ticker else "",
        implied_prob_yes=implied,
        bid=yes_bid,
        ask=yes_ask,
        spread=spread,
        volume_24h=safe_float(priced.get("volume_24h_fp")),
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
        children=kalshi_strike_children(nested),
    )
