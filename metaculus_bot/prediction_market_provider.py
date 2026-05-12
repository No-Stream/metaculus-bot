"""Prediction-market research provider.

Queries Polymarket + Kalshi + Manifold for markets that resolve on the same
(or adjacent) event as a given Metaculus question, and returns a
`MarketSnapshot` the forecaster can read as a peer cross-check.

Design anchors (from the G0 empirical study, 2026-05-12 -- see
`scratch_docs_and_planning/prediction_market_keyword_extraction_experiment.md`):

- Default keyword extraction is S4 (LLM noun phrases) + S5 (LLM entity + event
  + deadline) run in parallel via gpt-5-mini with `max_tokens=800` and
  `reasoning=low`. Hit rate 67% vs 33% for a naive baseline. The 800-token
  budget is load-bearing: gpt-5-mini burns 128-512 tokens on invisible
  reasoning before emitting any response.

- Manifold gets an extra S2 query (question text trimmed at '?') because its
  search prefers natural-language framings.

- Kalshi has no keyword-search endpoint. Prefetch ~3k events via
  `/trade-api/v2/events?status=open&with_nested_markets=true` once per
  session (~22s paginated, cached for ~6h) and fuzzy-match client-side.

- Polymarket Gamma public-search occasionally returns 403 (IP rate limit).
  Bounded retry-with-backoff; fail soft to empty.

- `as_of` filter drops matches with `close_time <= as_of`. Critical for
  resolved-question backtests: a market that closed BEFORE the as-of instant
  holds a post-resolution price that would leak into a backtest.

- Soft-fail on every error path. This provider returns empty on any failure;
  it never raises. A broken prediction-market API should never break a
  forecast.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import aiohttp
from rapidfuzz import fuzz

from metaculus_bot.constants import (
    PREDICTION_MARKET_KEYWORD_STRATEGY_ENV,
    PREDICTION_MARKET_KEYWORD_STRATEGY_VALID,
    PREDICTION_MARKETS_ENABLED_ENV,
    env_flag_enabled,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.research_providers import ResearchCallable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Keyword-extraction model (per G0 finding: gpt-5-mini reasoning=low).
KEYWORD_EXTRACT_MODEL = "openrouter/openai/gpt-5-mini"
# G0 token-budget trap: gpt-5-mini burns 128-512 tokens on invisible reasoning.
# 800 provides headroom so the visible response fits inside the budget.
KEYWORD_EXTRACT_MAX_TOKENS = 800

POLYMARKET_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
KALSHI_EVENTS_URL = "https://api.elections.kalshi.com/trade-api/v2/events"

# Bounded retry-with-backoff for transient 403/429/5xx from Polymarket.
POLYMARKET_MAX_ATTEMPTS = 3
MANIFOLD_MAX_ATTEMPTS = 3

# Client-side Kalshi fuzzy-match threshold below which we drop candidates.
KALSHI_MIN_FUZZY_SCORE = 40.0

# Per-platform search timeout (s). Wrapped in an outer `timeout` in
# fetch_market_snapshot; this is the per-HTTP-call cap.
PLATFORM_HTTP_TIMEOUT = 10.0

# Kalshi events cache TTL.
KALSHI_CACHE_TTL_S = 6 * 60 * 60  # 6h

# Max events to prefetch from Kalshi (G0 used 3k; cap matches).
KALSHI_PREFETCH_EVENT_LIMIT = 3000

# Raw-rules truncation for formatter output.
RAW_RULES_MAX_CHARS = 200


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MarketMatch:
    platform: Literal["polymarket", "kalshi", "manifold"]
    market_title: str
    market_url: str
    implied_prob_yes: float | None
    bid: float | None
    ask: float | None
    spread: float | None
    volume_24h: float | None
    close_time: datetime | None
    is_resolved: bool
    match_confidence: float
    raw_rules: str


@dataclass
class MarketSnapshot:
    matches: list[MarketMatch] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-session caches (module-scoped; reset via `_reset_session_caches`)
# ---------------------------------------------------------------------------

# Kalshi events cache: (timestamp_monotonic, events_list)
_KALSHI_CACHE: dict[str, tuple[float, list[dict]]] = {}
# Keyword-extraction cache: qid -> list[query_str]
_KEYWORD_CACHE: dict[int, list[str]] = {}
# Snapshot cache: qid -> MarketSnapshot
_SNAPSHOT_CACHE: dict[int, MarketSnapshot] = {}


def _reset_session_caches() -> None:
    """Clear all per-session caches. Called between tests and at session start."""
    _KALSHI_CACHE.clear()
    _KEYWORD_CACHE.clear()
    _SNAPSHOT_CACHE.clear()


def _get_session() -> aiohttp.ClientSession:
    """Construct a fresh aiohttp session. Patched in tests."""
    return aiohttp.ClientSession()


# ---------------------------------------------------------------------------
# Keyword extraction (S2 / S4 / S5 + union)
# ---------------------------------------------------------------------------


_S4_PROMPT = """Extract the 3-5 most important noun phrases for a prediction-market keyword search from this Metaculus question. Return ONLY a single search query string (no quotes, no lists, no commentary).

Question: {title}

Resolution criteria (first 400 chars): {rc}

Search query:"""

_S5_PROMPT = """From this Metaculus question, extract:
1. The primary entity (person, organization, asset, or event name)
2. The key event or action (what is predicted)
3. The deadline or time window (if any)

Combine into a terse search query (under 12 words). Return ONLY the query, no commentary.

Question: {title}

Resolution criteria (first 400 chars): {rc}

Search query:"""


def _strategy_s2(question_text: str) -> str:
    """Natural-language framing: question_text trimmed at the first '?'."""
    t = (question_text or "").strip()
    i = t.find("?")
    if i > 0:
        t = t[:i]
    return t.strip()


def _clean_llm_query(content: str) -> str:
    """Take the first non-empty line, strip quotes and trailing colons/labels."""
    for line in (content or "").splitlines():
        line = line.strip().strip('"').strip("'")
        if line and not line.lower().startswith(("search query", "query:", "answer")):
            return line[:200]
    return (content or "").strip().strip('"').strip("'")[:200]


class KeywordExtractor:
    """Extracts keyword queries per the configured strategy.

    `s4_s5_union` (default): S4 + S5 in parallel via gpt-5-mini. Union deduped.
    `s5_only`: S5 only (cheaper, 60% hit rate vs 67% for union).
    `simple`: S2 only (no LLM cost, 40% hit rate).
    """

    def __init__(self, strategy: str = "s4_s5_union") -> None:
        if strategy not in PREDICTION_MARKET_KEYWORD_STRATEGY_VALID:
            raise ValueError(
                f"Invalid strategy {strategy!r}; valid: {sorted(PREDICTION_MARKET_KEYWORD_STRATEGY_VALID)}"
            )
        self.strategy = strategy

    async def extract(self, question: Any) -> list[str]:  # noqa: ASYNC910
        qid = getattr(question, "id_of_question", None)
        if qid is not None and qid in _KEYWORD_CACHE:
            return list(_KEYWORD_CACHE[qid])  # noqa: ASYNC910

        question_text = getattr(question, "question_text", "") or ""
        title = getattr(question, "title", "") or question_text
        rc = getattr(question, "resolution_criteria", "") or ""

        queries: list[str] = []
        s2 = _strategy_s2(question_text)

        if self.strategy == "simple":
            if s2:
                queries.append(s2)
        elif self.strategy == "s5_only":
            queries.append(await self._run_llm(_S5_PROMPT, title, rc))
        else:  # s4_s5_union
            s4_task = asyncio.create_task(self._run_llm(_S4_PROMPT, title, rc))
            s5_task = asyncio.create_task(self._run_llm(_S5_PROMPT, title, rc))
            s4, s5 = await asyncio.gather(s4_task, s5_task)
            for q in (s4, s5):
                if q:
                    queries.append(q)

        # Dedup while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for q in queries:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                deduped.append(q)

        if qid is not None:
            _KEYWORD_CACHE[qid] = list(deduped)
        return deduped  # noqa: ASYNC910

    async def _run_llm(self, prompt_template: str, title: str, rc: str) -> str:
        prompt = prompt_template.format(title=title[:400], rc=rc[:400])
        try:
            llm = build_llm_with_openrouter_fallback(
                model=KEYWORD_EXTRACT_MODEL,
                temperature=0.0,
                max_tokens=KEYWORD_EXTRACT_MAX_TOKENS,
                reasoning_effort="low",
                timeout=60,
            )
            content = await llm.invoke(prompt)
        except Exception:
            logger.warning("Keyword extraction LLM call failed", exc_info=True)
            return ""  # noqa: ASYNC910
        return _clean_llm_query(content)

    def queries_for_platform(self, question: Any, base_queries: list[str], platform: str) -> list[str]:
        """Per-platform query augmentation.

        Manifold prefers natural-language S2 framings (G0 finding); we ALWAYS add
        S2 for Manifold on top of whatever the core strategy produced.
        """
        out = list(base_queries)
        if platform == "manifold":
            s2 = _strategy_s2(getattr(question, "question_text", "") or "")
            if s2 and s2.lower() not in {q.lower() for q in out}:
                out.append(s2)
        return out


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------


def _parse_polymarket_matches(payload: Any) -> list[MarketMatch]:
    """Parse Gamma public-search response into MarketMatch objects.

    Gamma returns {events: [...], markets: [...]}. Each event may have nested
    markets with `outcomePrices` (JSON-encoded string or list). Take the first
    outcome's price as implied P(Yes).
    """
    if not isinstance(payload, dict):
        logger.warning("Polymarket returned non-dict payload")
        return []

    out: list[MarketMatch] = []

    def _prob_from_prices(prices: Any) -> float | None:
        if isinstance(prices, str):
            try:
                arr = json.loads(prices)
            except (json.JSONDecodeError, ValueError):
                return None
            if isinstance(arr, list) and arr:
                try:
                    return float(arr[0])
                except (TypeError, ValueError):
                    return None
        if isinstance(prices, list) and prices:
            try:
                return float(prices[0])
            except (TypeError, ValueError):
                return None
        return None

    events = payload.get("events") or []
    for ev in events[:10]:
        title = ev.get("title") or ev.get("question") or ""
        slug = ev.get("slug") or ""
        url = f"https://polymarket.com/event/{slug}" if slug else ""
        description = ev.get("description") or ""
        end_iso = ev.get("endDate") or ev.get("end_date_iso") or ""
        close_time = _parse_iso(end_iso)
        volume = _safe_float(ev.get("volume"))

        implied: float | None = None
        bid: float | None = None
        ask: float | None = None
        vol_24h: float | None = None
        markets = ev.get("markets") or []
        if markets and isinstance(markets[0], dict):
            m0 = markets[0]
            implied = _prob_from_prices(m0.get("outcomePrices"))
            bid = _safe_float(m0.get("bestBid"))
            ask = _safe_float(m0.get("bestAsk"))
            vol_24h = _safe_float(m0.get("volume24hr"))
        spread = (ask - bid) if (bid is not None and ask is not None) else None

        out.append(
            MarketMatch(
                platform="polymarket",
                market_title=title,
                market_url=url,
                implied_prob_yes=implied,
                bid=bid,
                ask=ask,
                spread=spread,
                volume_24h=vol_24h if vol_24h is not None else volume,
                close_time=close_time,
                is_resolved=bool(ev.get("closed")) or bool(ev.get("resolved")),
                match_confidence=0.0,  # filled in later by the orchestrator
                raw_rules=description[:2000],
            )
        )

    # Fallback to top-level markets if events were empty.
    if not out:
        markets = payload.get("markets") or []
        for m in markets[:10]:
            title = m.get("question") or m.get("title") or ""
            slug = m.get("slug") or ""
            url = f"https://polymarket.com/market/{slug}" if slug else ""
            implied = _prob_from_prices(m.get("outcomePrices"))
            out.append(
                MarketMatch(
                    platform="polymarket",
                    market_title=title,
                    market_url=url,
                    implied_prob_yes=implied,
                    bid=_safe_float(m.get("bestBid")),
                    ask=_safe_float(m.get("bestAsk")),
                    spread=None,
                    volume_24h=_safe_float(m.get("volume24hr")),
                    close_time=_parse_iso(m.get("endDate") or ""),
                    is_resolved=bool(m.get("closed")),
                    match_confidence=0.0,
                    raw_rules=(m.get("description") or "")[:2000],
                )
            )

    return out


async def _polymarket_search(session: Any, query: str) -> list[MarketMatch]:
    params = {"q": query, "limit_per_type": "10"}
    for attempt in range(POLYMARKET_MAX_ATTEMPTS):
        try:
            async with session.get(
                POLYMARKET_SEARCH_URL, params=params, timeout=aiohttp.ClientTimeout(total=PLATFORM_HTTP_TIMEOUT)
            ) as resp:
                if resp.status in (403, 429) or resp.status >= 500:
                    backoff = 1.0 * (2**attempt)
                    logger.warning(
                        f"Polymarket HTTP {resp.status} for q={query[:40]!r}; "
                        f"retry {attempt + 1}/{POLYMARKET_MAX_ATTEMPTS} after {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                    continue
                if resp.status != 200:
                    text = (await resp.text())[:200]
                    logger.warning(f"Polymarket HTTP {resp.status} non-retryable: {text}")
                    return []
                payload = await resp.json()
                return _parse_polymarket_matches(payload)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Polymarket transient error q={query[:40]!r}: {e}; retry {attempt + 1}")
            await asyncio.sleep(1.0 * (2**attempt))
    return []  # noqa: ASYNC910


# ---------------------------------------------------------------------------
# Manifold
# ---------------------------------------------------------------------------


def _parse_manifold_matches(payload: Any) -> list[MarketMatch]:
    if not isinstance(payload, list):
        logger.warning("Manifold returned non-list payload")
        return []

    out: list[MarketMatch] = []
    for m in payload[:10]:
        if not isinstance(m, dict):
            continue
        title = m.get("question") or ""
        slug = m.get("slug") or ""
        creator = m.get("creatorUsername") or ""
        url = f"https://manifold.markets/{creator}/{slug}" if slug and creator else (m.get("url") or "")
        prob = _safe_float(m.get("probability"))
        close_ms = m.get("closeTime")
        close_time: datetime | None = None
        if isinstance(close_ms, (int, float)):
            try:
                close_time = datetime.fromtimestamp(float(close_ms) / 1000.0, tz=timezone.utc)
            except (OverflowError, OSError, ValueError):
                close_time = None

        out.append(
            MarketMatch(
                platform="manifold",
                market_title=title,
                market_url=url,
                implied_prob_yes=prob,
                bid=None,
                ask=None,
                spread=None,
                volume_24h=_safe_float(m.get("volume24Hours")),
                close_time=close_time,
                is_resolved=bool(m.get("isResolved")),
                match_confidence=0.0,
                raw_rules=(m.get("textDescription") or "")[:2000],
            )
        )
    return out


async def _manifold_search(session: Any, query: str) -> list[MarketMatch]:
    params = {"term": query, "contractType": "BINARY", "limit": "10"}
    for attempt in range(MANIFOLD_MAX_ATTEMPTS):
        try:
            async with session.get(
                MANIFOLD_SEARCH_URL, params=params, timeout=aiohttp.ClientTimeout(total=PLATFORM_HTTP_TIMEOUT)
            ) as resp:
                if resp.status == 429 or resp.status >= 500:
                    backoff = 1.0 * (2**attempt)
                    await asyncio.sleep(backoff)
                    continue
                if resp.status != 200:
                    text = (await resp.text())[:200]
                    logger.warning(f"Manifold HTTP {resp.status} non-retryable: {text}")
                    return []
                payload = await resp.json()
                return _parse_manifold_matches(payload)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Manifold transient error q={query[:40]!r}: {e}; retry {attempt + 1}")
            await asyncio.sleep(1.0 * (2**attempt))
    return []  # noqa: ASYNC910


# ---------------------------------------------------------------------------
# Kalshi (prefetch events + client-side fuzzy match)
# ---------------------------------------------------------------------------


async def _kalshi_prefetch_events(
    session: Any, event_limit: int = KALSHI_PREFETCH_EVENT_LIMIT, page_sleep_s: float = 1.0
) -> list[dict]:
    """Paginate through open Kalshi events. Returns (possibly empty) list on error.

    Uses the `/events?with_nested_markets=true` endpoint NOT `/markets` --
    per G0, `/markets` is dominated by sports-parlay 'MVE' rows.
    """
    cached = _KALSHI_CACHE.get("events")
    if cached is not None:
        ts, events = cached
        if (time.monotonic() - ts) < KALSHI_CACHE_TTL_S:
            return events  # noqa: ASYNC910

    params = {"status": "open", "limit": "200", "with_nested_markets": "true"}
    all_events: list[dict] = []
    cursor: str | None = None

    while len(all_events) < event_limit:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        try:
            async with session.get(KALSHI_EVENTS_URL, params=p, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    logger.warning("Kalshi 429 during prefetch; stopping pagination early")
                    break
                if resp.status != 200:
                    text = (await resp.text())[:200]
                    logger.warning(f"Kalshi prefetch HTTP {resp.status}: {text}")
                    break
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Kalshi prefetch transient error: {e}")
            break
        except (ValueError, TypeError) as e:
            logger.warning(f"Kalshi prefetch JSON parse error: {e}")
            break

        if not isinstance(data, dict):
            break
        batch = data.get("events") or []
        if not isinstance(batch, list):
            break
        all_events.extend([ev for ev in batch if isinstance(ev, dict)])
        cursor = data.get("cursor") or None
        if not cursor or not batch:
            break
        if page_sleep_s > 0:
            await asyncio.sleep(page_sleep_s)

    _KALSHI_CACHE["events"] = (time.monotonic(), all_events)
    return all_events  # noqa: ASYNC910


def _kalshi_search_local(
    events: list[dict], query: str, top_k: int = 5, min_score: float = KALSHI_MIN_FUZZY_SCORE
) -> list[MarketMatch]:
    q_lower = (query or "").lower()
    scored: list[tuple[float, MarketMatch]] = []

    for ev in events:
        if not isinstance(ev, dict):
            continue
        title = ev.get("title") or ev.get("sub_title") or ""
        if not title:
            continue

        nested = ev.get("markets") or []
        rules_primary = ""
        close_time: datetime | None = None
        yes_bid: float | None = None
        yes_ask: float | None = None
        volume_24h: float | None = None
        is_resolved = False
        if nested and isinstance(nested[0], dict):
            first = nested[0]
            rules_primary = (first.get("rules_primary") or "")[:2000]
            close_time = _parse_iso(first.get("close_time") or "")
            yes_bid = _safe_float(first.get("yes_bid_dollars"))
            yes_ask = _safe_float(first.get("yes_ask_dollars"))
            volume_24h = _safe_float(first.get("volume_24h_fp"))
            is_resolved = (first.get("status") or "").lower() in ("settled", "finalized", "closed")

        title_score = fuzz.token_set_ratio(q_lower, title.lower())
        rules_score = fuzz.token_set_ratio(q_lower, rules_primary.lower()) if rules_primary else 0
        combined = 0.7 * title_score + 0.3 * rules_score
        if combined < min_score:
            continue

        implied: float | None = None
        spread: float | None = None
        if yes_bid is not None and yes_ask is not None:
            implied = (yes_bid + yes_ask) / 2.0
            spread = yes_ask - yes_bid

        event_ticker = ev.get("event_ticker") or ""
        url = f"https://kalshi.com/markets/{event_ticker}" if event_ticker else ""

        scored.append(
            (
                combined,
                MarketMatch(
                    platform="kalshi",
                    market_title=title,
                    market_url=url,
                    implied_prob_yes=implied,
                    bid=yes_bid,
                    ask=yes_ask,
                    spread=spread,
                    volume_24h=volume_24h,
                    close_time=close_time,
                    is_resolved=is_resolved,
                    match_confidence=combined / 100.0,
                    raw_rules=rules_primary[:2000],
                ),
            )
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [pm for _, pm in scored[:top_k]]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def fetch_market_snapshot(
    question: Any,
    *,
    platforms: tuple[str, ...] = ("polymarket", "kalshi", "manifold"),
    max_matches_per_platform: int = 3,
    timeout: float = 5.0,  # noqa: ASYNC109
    as_of: datetime | None = None,
) -> MarketSnapshot:
    """Fan out to all three platforms in parallel, collect matches, apply filters.

    Soft-fails on any error: returns empty MarketSnapshot + WARNING log. A
    broken prediction-market API should never break a forecast run.

    `as_of` (backtest leakage defense): drops matches whose `close_time` is
    less than or equal to as_of. Required in backtest; optional in prod.
    """
    qid = getattr(question, "id_of_question", None)
    cached_snap = _SNAPSHOT_CACHE.get(qid) if qid is not None else None
    if cached_snap is not None:
        return cached_snap  # noqa: ASYNC910

    try:
        snapshot = await asyncio.wait_for(
            _fetch_market_snapshot_impl(
                question,
                platforms=platforms,
                max_matches_per_platform=max_matches_per_platform,
                as_of=as_of,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Prediction-market snapshot TIMEOUT after {timeout}s for qid={qid}")
        return MarketSnapshot(matches=[])  # noqa: ASYNC910
    except Exception:
        logger.warning("Prediction-market snapshot FAILED (soft-fail returning empty)", exc_info=True)
        return MarketSnapshot(matches=[])  # noqa: ASYNC910

    if qid is not None:
        _SNAPSHOT_CACHE[qid] = snapshot
    return snapshot


async def _fetch_market_snapshot_impl(
    question: Any,
    *,
    platforms: tuple[str, ...],
    max_matches_per_platform: int,
    as_of: datetime | None,
) -> MarketSnapshot:
    strategy = os.getenv(PREDICTION_MARKET_KEYWORD_STRATEGY_ENV, "s4_s5_union").lower()
    if strategy not in PREDICTION_MARKET_KEYWORD_STRATEGY_VALID:
        logger.warning(f"Invalid PREDICTION_MARKET_KEYWORD_STRATEGY={strategy!r}, using default")
        strategy = "s4_s5_union"

    extractor = KeywordExtractor(strategy=strategy)
    queries = await extractor.extract(question)
    if not queries:
        logger.info("Keyword extraction produced no queries; returning empty snapshot")
        return MarketSnapshot(matches=[])

    session_cm = _get_session()
    async with session_cm as session:
        all_matches: list[MarketMatch] = []

        async def _poly_for_all_queries() -> list[MarketMatch]:
            tasks = [_polymarket_search(session, q) for q in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return _flatten_results(results, "polymarket")

        async def _manifold_for_all_queries() -> list[MarketMatch]:
            mf_queries = extractor.queries_for_platform(question, queries, "manifold")
            tasks = [_manifold_search(session, q) for q in mf_queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return _flatten_results(results, "manifold")

        async def _kalshi_for_all_queries() -> list[MarketMatch]:
            try:
                events = await _kalshi_prefetch_events(session, event_limit=KALSHI_PREFETCH_EVENT_LIMIT)
            except Exception:
                logger.warning("Kalshi prefetch failed", exc_info=True)
                return []  # noqa: ASYNC910
            merged: list[MarketMatch] = []
            for q in queries:
                merged.extend(_kalshi_search_local(events, q, top_k=max_matches_per_platform + 2))
            return merged

        platform_tasks: list[tuple[str, asyncio.Task]] = []
        if "polymarket" in platforms:
            platform_tasks.append(("polymarket", asyncio.create_task(_poly_for_all_queries())))
        if "manifold" in platforms:
            platform_tasks.append(("manifold", asyncio.create_task(_manifold_for_all_queries())))
        if "kalshi" in platforms:
            platform_tasks.append(("kalshi", asyncio.create_task(_kalshi_for_all_queries())))

        for platform, task in platform_tasks:
            try:
                matches = await task
            except Exception:
                logger.warning(f"Platform {platform} failed (soft-fail)", exc_info=True)
                matches = []
            all_matches.extend(matches)

    # Dedup within-platform by market_url (or title fallback), cap per platform.
    by_platform: dict[str, list[MarketMatch]] = {"polymarket": [], "kalshi": [], "manifold": []}
    seen_urls_per_platform: dict[str, set[str]] = {"polymarket": set(), "kalshi": set(), "manifold": set()}

    for m in all_matches:
        # as_of filter: drop matches that closed at or before as_of.
        if as_of is not None and m.close_time is not None:
            m_close = m.close_time if m.close_time.tzinfo else m.close_time.replace(tzinfo=timezone.utc)
            as_of_tz = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
            if m_close <= as_of_tz:
                continue

        key = m.market_url or m.market_title
        if key in seen_urls_per_platform[m.platform]:
            continue
        seen_urls_per_platform[m.platform].add(key)
        if len(by_platform[m.platform]) < max_matches_per_platform:
            by_platform[m.platform].append(m)

    combined: list[MarketMatch] = []
    for plat in ("polymarket", "kalshi", "manifold"):
        combined.extend(by_platform[plat])

    return MarketSnapshot(matches=combined)


def _flatten_results(results: list[Any], platform: str) -> list[MarketMatch]:
    out: list[MarketMatch] = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning(f"{platform} query task raised: {r}")
            continue
        if isinstance(r, list):
            out.extend(r)
    return out


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def format_snapshot_for_research(snapshot: MarketSnapshot) -> str:
    """Compact markdown block for the research prompt.

    Emits a table + raw-rules section + the NOT-AN-ANCHOR caveat that tells
    the forecaster to verify resolution-criteria alignment before leaning on
    any market's price.
    """
    if not snapshot.matches:
        return ""

    lines: list[str] = []
    lines.append("NOT AN ANCHOR -- verify the resolution criteria match before using any market's price.")
    lines.append("")
    lines.append("| platform | title | prob | vol | close | conf |")
    lines.append("|---|---|---|---|---|---|")
    for m in snapshot.matches:
        prob = f"{m.implied_prob_yes:.2f}" if m.implied_prob_yes is not None else "-"
        vol = f"{m.volume_24h:.0f}" if m.volume_24h is not None else "-"
        close = m.close_time.strftime("%Y-%m-%d") if m.close_time else "-"
        conf = f"{m.match_confidence:.2f}"
        safe_title = (m.market_title or "")[:80].replace("|", "/")
        lines.append(f"| {m.platform} | {safe_title} | {prob} | {vol} | {close} | {conf} |")

    lines.append("")
    lines.append("### Resolution criteria / rules")
    for m in snapshot.matches:
        rules_trunc = (m.raw_rules or "").strip().replace("\n", " ")
        if len(rules_trunc) > RAW_RULES_MAX_CHARS:
            rules_trunc = rules_trunc[:RAW_RULES_MAX_CHARS] + "..."
        link = f" <{m.market_url}>" if m.market_url else ""
        lines.append(f"- **{m.platform}**{link}: {rules_trunc}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ResearchCallable factory (plugged into _select_research_providers)
# ---------------------------------------------------------------------------


def prediction_market_provider() -> ResearchCallable:
    """Factory returning an async research callable for prediction-market data.

    The returned callable takes a question_text (per the ResearchCallable
    contract) but internally constructs a minimal question shim so the
    orchestrator can use its full API. In practice, when called from
    `_run_providers_parallel`, the caller should pass the actual question
    text -- the orchestrator will extract keywords from that text alone.

    Gated on PREDICTION_MARKETS_ENABLED env flag; disabled returns "".
    """

    async def _fetch(question_text: str) -> str:
        if not env_flag_enabled(PREDICTION_MARKETS_ENABLED_ENV):
            return ""  # noqa: ASYNC910

        # Build a minimal question shim from the question_text the orchestrator
        # passes us. Full question object would let us use title +
        # resolution_criteria separately, but at the provider layer we only
        # have question_text.
        class _QuestionShim:
            def __init__(self, text: str) -> None:
                self.id_of_question: int | None = None
                self.question_text = text
                self.title = text
                self.resolution_criteria = ""

        shim = _QuestionShim(question_text)
        snapshot = await fetch_market_snapshot(shim)
        return format_snapshot_for_research(snapshot)

    return _fetch


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_iso(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        # Python 3.11 fromisoformat accepts 'Z' suffix as of 3.11
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
