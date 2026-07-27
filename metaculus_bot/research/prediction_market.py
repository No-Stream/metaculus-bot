"""Prediction-market research provider.

Queries Polymarket + Kalshi + Manifold + PredictIt for markets that resolve on
the same (or adjacent) event as a given Metaculus question, and returns a
`MarketSnapshot` the forecaster can read as a peer cross-check.

Design anchors (from the G0 empirical study, 2026-05-12 -- see
`scratch_docs_and_planning/prediction_market_keyword_extraction_experiment.md`):

- Default keyword extraction is S4 (LLM noun phrases) + S5 (LLM entity + event
  + deadline) run in parallel via gpt-5.4-mini with `max_tokens=800` and
  `reasoning=low`. Hit rate 67% vs 33% for a naive baseline. The 800-token
  budget is load-bearing: gpt-5.4-mini burns 128-512 tokens on invisible
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
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Literal

import aiohttp
import ijson
import litellm.exceptions
from forecasting_tools.data_models.questions import MetaculusQuestion
from rapidfuzz import fuzz

from metaculus_bot.constants import (
    MARKET_RELEVANCE_CONF_MIN,
    MARKET_RELEVANCE_OVERLAP_MIN,
    PREDICTION_MARKET_KEYWORD_BACKOFFS,
    PREDICTION_MARKET_KEYWORD_STRATEGY_ENV,
    PREDICTION_MARKET_KEYWORD_STRATEGY_VALID,
    PREDICTION_MARKET_KEYWORD_WALL_TIMEOUT,
    PREDICTION_MARKET_TIMEOUT,
    PREDICTION_MARKETS_ENABLED_ENV,
    env_flag_enabled,
)
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.llm_configs import PREDICTION_MARKET_KEYWORD_LLM_CONFIG
from metaculus_bot.llm_retry import invoke_with_transient_retry
from metaculus_bot.research.http_fetch import build_session, read_body_capped, read_body_snippet
from metaculus_bot.research.provider_diagnostics import record_provider_detail
from metaculus_bot.research.providers import ResearchCallable
from metaculus_bot.research.raw_log import record_raw_research

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

POLYMARKET_SEARCH_URL = "https://gamma-api.polymarket.com/public-search"
MANIFOLD_SEARCH_URL = "https://api.manifold.markets/v0/search-markets"
KALSHI_EVENTS_URL = "https://api.elections.kalshi.com/trade-api/v2/events"
# Kalshi has no full-text search endpoint; the series list is the sanctioned
# entity->ticker index (one unpaginated call, returns ticker/title/category/tags).
# We match salient question entities against series titles locally, then fetch that
# series' open events by `series_ticker`. Verified against docs.kalshi.com 2026-07-19.
KALSHI_SERIES_URL = "https://api.elections.kalshi.com/trade-api/v2/series"
PREDICTIT_URL = "https://www.predictit.org/api/marketdata/all/"

# Bounded retry-with-backoff for transient 403/429/5xx. The s4_s5_union strategy
# already issues 2 queries per platform; one-and-done retries suffice.
POLYMARKET_MAX_ATTEMPTS = 2
MANIFOLD_MAX_ATTEMPTS = 2
HTTP_RETRY_BACKOFF_SECS = 0.5

# Client-side Kalshi fuzzy-match threshold below which we drop candidates.
KALSHI_MIN_FUZZY_SCORE = 40.0

# Entity-based Kalshi retrieval (closes the recall hole where the exact-title
# market is absent from / drowned in the capped prefetch dump). Salient entities
# extracted from the question title are matched against the Kalshi series list;
# a series clears at KALSHI_ENTITY_SERIES_MIN_SCORE (higher than the general
# fuzzy floor because short entity strings score near-100 against the right
# series title). We fetch at most KALSHI_ENTITY_MAX_SERIES series' open events
# from at most KALSHI_ENTITY_MAX_ENTITIES entities to bound the HTTP fan-out.
KALSHI_ENTITY_SERIES_MIN_SCORE = 80.0
KALSHI_ENTITY_MAX_ENTITIES = 4
KALSHI_ENTITY_MAX_SERIES = 4

# Client-side PredictIt fuzzy-match threshold (mirrors Kalshi; PredictIt has no
# keyword-search endpoint, so we prefetch the full market dump and fuzzy-match).
PREDICTIT_MIN_FUZZY_SCORE = 40.0

# Liquidity / participation signal-label thresholds. Low-volume markets are
# often bot-dominated (roughly sub-$10k), so a "thin" label is a real noise
# warning, not a formality. These cutoffs are a tunable first pass, not
# calibrated values — the "thin" ceiling sits at $5k deliberately conservatively.
LIQUIDITY_THIN_USD = 5_000.0
LIQUIDITY_DEEP_USD = 50_000.0
MANIFOLD_THIN_BETTORS = 20
MANIFOLD_HIGH_BETTORS = 100

# Per-platform search timeout (s). Wrapped in an outer `timeout` in
# fetch_market_snapshot; this is the per-HTTP-call cap.
PLATFORM_HTTP_TIMEOUT = 10.0

# Hard cap on a single response body. Polymarket/Manifold don't paginate
# search responses, so a single payload should fit comfortably under this.
MAX_RESPONSE_BYTES = 10 * 1024 * 1024

# Kalshi events cache TTL.
KALSHI_CACHE_TTL_S = 6 * 60 * 60  # 6h

# PredictIt markets cache TTL (mirrors Kalshi; the /all/ dump is a full snapshot).
PREDICTIT_CACHE_TTL_S = 6 * 60 * 60  # 6h

# Max events to prefetch from Kalshi (G0 used 3k; cap matches).
KALSHI_PREFETCH_EVENT_LIMIT = 3000

# Kalshi /series streaming bounds. The endpoint is UNPAGINATED (no limit/cursor;
# verified against docs.kalshi.com 2026-07-25), so its full-catalogue body only
# grows — on 2026-07-25 it crossed the shared 10 MiB read cap (10.02 MiB) and
# was silently dropped, killing the entity-recall path. We stream-parse it (see
# `_kalshi_fetch_series_streamed`) and retain only the fields entity-matching
# needs, so PEAK MEMORY tracks the retained metadata, not the raw payload, and a
# growing catalogue can't re-trip a fixed size wall. These bound the STREAM:
#  - KALSHI_SERIES_MAX_BYTES is a generous last-resort guard against a runaway /
#    compressed-bomb body; it sits ~6x above the live catalogue so normal growth
#    never trips it, and a breach is LOUD + counted (never a silent drop).
#  - the read timeout covers streaming the whole body (the shared 10s per-call
#    timeout is for small search responses; series is a bulk endpoint). It is a
#    WALL-CLOCK budget for the whole fetch, retry included, so the bounded retry
#    can't push this fetch past what the surrounding PREDICTION_MARKET_TIMEOUT
#    allows the whole snapshot.
KALSHI_SERIES_MAX_BYTES = 64 * 1024 * 1024
KALSHI_SERIES_HTTP_TIMEOUT = 30.0
KALSHI_SERIES_MAX_ATTEMPTS = 2
_KALSHI_SERIES_READ_CHUNK_BYTES = 65536
# Mirrors _http_get_with_backoff's default retryable set (403 = Kalshi's rate-limit
# shape); >= 500 is also retried.
_KALSHI_SERIES_RETRYABLE_STATUSES = frozenset({403, 429, 500, 502, 503, 504})
# ijson structural events, i.e. the ones that carry no scalar `value` to keep.
_IJSON_CONTAINER_EVENTS = frozenset({"start_map", "map_key", "end_map", "start_array", "end_array"})

# Buffer applied to scheduled_resolution_time when the orchestrator derives a
# default `as_of`. Subtracting a day keeps backtest as_of strictly before any
# market that closes alongside the question, defending against same-day leakage.
AS_OF_DEFAULT_BUFFER = timedelta(days=1)

# Raw-rules truncation for formatter output.
RAW_RULES_MAX_CHARS = 200


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MarketMatch:
    platform: Literal["polymarket", "kalshi", "manifold", "predictit"]
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
    # Liquidity / participation fields. Previously received-but-discarded; now
    # parsed so the formatter can label how informative each crowd signal is.
    total_volume: float | None = None
    liquidity: float | None = None
    open_interest: float | None = None
    num_bettors: int | None = None


@dataclass
class MarketSnapshot:
    matches: list[MarketMatch] = field(default_factory=list)
    # Per-source outcome tokens ({source_name: token}) for the provider-diagnostics
    # line: the 4 platforms, plus `kalshi_series` (the entity-index fetch), plus
    # `keywords` / `snapshot` on the whole-provider failure paths (keyword extraction
    # produced nothing; the snapshot timed out or blew up). A token starting with
    # "ok"/"none" is benign; anything else (e.g. "dropped(size_cap)", "error(...)",
    # "partial(1/2)") is a LOST source. `none` means every sub-fetch SUCCEEDED and
    # matched nothing — an outage must never land there, or the published line reads
    # healthy through a blackout. See provider_diagnostics.
    sources: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _FetchTally:
    """How many of a platform's HTTP sub-fetches came back vs. were lost.

    Carried alongside the matches because an upstream outage and a genuine no-match
    both arrive as an empty match list; without these counts the diagnostics token
    cannot tell them apart (see `_platform_source_token`). The unit is one sub-fetch
    at each platform's natural granularity: a keyword query for Polymarket/Manifold,
    a page for the Kalshi events prefetch, the single dump for PredictIt.
    """

    ok: int = 0
    failed: int = 0

    def __add__(self, other: _FetchTally) -> _FetchTally:
        return _FetchTally(self.ok + other.ok, self.failed + other.failed)


def _platform_source_token(matches: list[MarketMatch], tally: _FetchTally) -> str:
    """Classify one platform's outcome as an `ok(N)` / `none` / loss source token.

    `none` is reserved for "every sub-fetch succeeded and nothing matched" — the one
    benign empty outcome `provider_diagnostics._is_lost_source` does not flag. So a
    lost sub-fetch has to produce a loss token even when other sub-fetches returned
    matches: otherwise a total outage reads as a healthy `none`, and a platform that
    lost one of two queries reads as a clean `ok(N)`.
    """
    if tally.failed:
        if tally.ok == 0:
            return "error(all_queries_failed)"
        return f"partial({tally.ok}/{tally.ok + tally.failed})"
    return f"ok({len(matches)})" if matches else "none"


def _liquidity_label(m: MarketMatch) -> str:
    """Label how informative a market's price is, given its liquidity/participation.

    Real-money venues (Polymarket, Kalshi) score on dollar volume / open interest;
    Manifold (play-money) scores on unique bettor count instead. A thin market is
    a noise warning: sub-$10k volume is often bot-dominated, so its price should be
    discounted relative to a deep, actively-traded market. Thresholds are tunable.
    """
    if m.platform == "predictit":
        # PredictIt exposes no volume/liquidity/OI fields in its all-markets dump.
        return "no-liquidity-data"

    if m.platform == "manifold":
        if m.num_bettors is None:
            return "no-liquidity-data"
        if m.num_bettors < MANIFOLD_THIN_BETTORS:
            return "thin"
        if m.num_bettors <= MANIFOLD_HIGH_BETTORS:
            return "decent"
        return "high"

    # Real-money venues: score on the larger of total volume and open interest.
    if m.total_volume is None and m.open_interest is None:
        return "no-liquidity-data"
    score = max(m.total_volume or 0.0, m.open_interest or 0.0)
    if score < LIQUIDITY_THIN_USD:
        return "thin"
    if score <= LIQUIDITY_DEEP_USD:
        return "decent"
    return "deep"


# ---------------------------------------------------------------------------
# Per-session caches (module-scoped; reset via `_reset_session_caches`)
# ---------------------------------------------------------------------------

# Kalshi events cache: (timestamp_monotonic, events_list).
_KALSHI_CACHE: dict[str, tuple[float, list[dict]]] = {}
# PredictIt markets cache: (timestamp_monotonic, markets_list).
_PREDICTIT_CACHE: dict[str, tuple[float, list[dict]]] = {}
# Keyword-extraction cache: qid -> list[query_str].
_KEYWORD_CACHE: dict[int, list[str]] = {}
# Snapshot cache keyed by (qid, as_of_iso). The as_of leg keeps backtest runs
# at different as-of instants from sharing a snapshot computed at one as-of.
_SNAPSHOT_CACHE: dict[tuple[int, str], MarketSnapshot] = {}

# Per-run count of Kalshi /series fetch failures. The series fetch feeds the
# entity-recall path; when it dies (HTTP/transport/parse error, or the generous
# safety ceiling) the provider still returns fuzzy-over-events matches, so the
# failure is otherwise INVISIBLE — no counter, status="ok" (the 2026-07-25
# observability hole: research_provider_failures=0 while the path was dead).
# The orchestrator folds this into alertable_count, so a series path that's dead
# every question reddens CI instead of vanishing. A one-off transient bumps it
# once (an accepted rare false alarm, mirroring gap_fill_v2_error_count).
# Module-level like the caches => accumulates per run; reset between tests.
_KALSHI_SERIES_FETCH_FAILURES: int = 0


def _bump_kalshi_series_failure() -> None:
    global _KALSHI_SERIES_FETCH_FAILURES
    _KALSHI_SERIES_FETCH_FAILURES += 1


def kalshi_series_fetch_failures() -> int:
    """Per-run count of Kalshi /series fetch failures (folded into alertable_count)."""
    return _KALSHI_SERIES_FETCH_FAILURES


def reset_series_degradation_counter() -> None:
    """Zero the series-degradation counter at run start.

    The provider is a stateless callable, so the counter lives at module scope;
    without a run-start reset it would leak across runs sharing one process (and
    across tests, polluting every later alertable_count == 0 assertion). Called
    from forecast_questions alongside reset_pchip_stats — same per-run cadence."""
    global _KALSHI_SERIES_FETCH_FAILURES
    _KALSHI_SERIES_FETCH_FAILURES = 0


# Per-run count of LOST prediction-market SOURCES: one per platform whose status
# token came out a loss (`error(all_queries_failed)` / `partial(...)` / an escaped
# transport error), one per whole-provider failure (snapshot timeout, outer-except),
# and one when keyword extraction produces nothing. That last cause is why the name
# is "source losses" rather than "platform failures" (renamed 2026-07-26): a dead
# keyword extractor silences all four venues without any venue failing, and the old
# name read as "a venue went down". The two causes are distinguished per-source in
# `MarketSnapshot.sources` (`keywords:error(no_queries)` vs `polymarket:error(...)`),
# which rides both the published comment and the schema-v2 research archive.
# Operator decision 2026-07-25: alert on ANY source loss rather than only a total
# blackout — maximum sensitivity, accepting that one flaky venue can redden most
# runs. The provider soft-fails internally, so like the series counter this is the
# only path by which an outage reaches CI. Module-level like the caches =>
# accumulates per run.
_SOURCE_LOSSES: int = 0


def _bump_source_loss() -> None:
    global _SOURCE_LOSSES
    _SOURCE_LOSSES += 1


def prediction_market_source_losses() -> int:
    """Per-run count of lost prediction-market sources (folded into alertable_count)."""
    return _SOURCE_LOSSES


def reset_source_loss_counter() -> None:
    """Zero the source-loss counter at run start (see
    `reset_series_degradation_counter` for why module-scoped counters need this)."""
    global _SOURCE_LOSSES
    _SOURCE_LOSSES = 0


def _reset_session_caches() -> None:
    """Clear all per-session caches. Called between tests and at session start."""
    global _KALSHI_SERIES_FETCH_FAILURES, _SOURCE_LOSSES
    _KALSHI_CACHE.clear()
    _PREDICTIT_CACHE.clear()
    _KEYWORD_CACHE.clear()
    _SNAPSHOT_CACHE.clear()
    _KALSHI_SERIES_FETCH_FAILURES = 0
    _SOURCE_LOSSES = 0


def _get_session() -> aiohttp.ClientSession:
    """Construct a fresh aiohttp session. Patched in tests.

    No headers arg: the JSON APIs get aiohttp's default UA (flipping to a
    browser UA is a separate experiment — see the resolution-source plan).
    """
    return build_session(timeout_s=PLATFORM_HTTP_TIMEOUT)


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
            return line[:200]  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # query length cap, not data sampling
    return (content or "").strip().strip('"').strip("'")[:200]


class KeywordExtractor:
    """Extracts keyword queries per the configured strategy.

    `s4_s5_union` (default): S4 + S5 in parallel via gpt-5.4-mini. Union deduped.
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
            return list(_KEYWORD_CACHE[qid])  # noqa: ASYNC910  # noqa: HARNESS-SCAN-EXEMPT-object-explosion

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
        prompt = prompt_template.format(title=title[:400], rc=rc[:400])  # noqa: HARNESS-SCAN-EXEMPT-subsampling
        # Constructor errors are config bugs (bad model slug, missing API key wiring,
        # etc.) and should crash loudly. Only the .invoke call is expected to face
        # transient LLM errors -- those soft-fall to "" so the snapshot still runs.
        llm = build_llm_with_openrouter_fallback(**PREDICTION_MARKET_KEYWORD_LLM_CONFIG)
        try:
            # The config pins allowed_tries=1, so this wrapper is the only retry
            # layer: it gives the call its own wall cap (previously the snapshot-level
            # wait_for was the sole bound, and a stalled extractor took the whole
            # snapshot down with it) and replaces forecasting-tools' un-gated
            # random.uniform(5, 10) tenacity sleep with one bounded, elapsed-gated
            # backoff that never fires on a deterministic client error.
            content = await invoke_with_transient_retry(
                lambda: llm.invoke(prompt),
                wall_timeout=PREDICTION_MARKET_KEYWORD_WALL_TIMEOUT,
                label="prediction_market_keywords",
                backoffs=PREDICTION_MARKET_KEYWORD_BACKOFFS,
            )
        except (litellm.exceptions.APIError, asyncio.TimeoutError, RuntimeError):
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
# HTTP helper (shared by Polymarket and Manifold)
# ---------------------------------------------------------------------------


async def _read_json_capped(resp: Any, label: str) -> Any | None:
    """Parse a response body as JSON, rejecting responses over MAX_RESPONSE_BYTES.

    Size-capped body read delegated to the shared `read_body_capped` (full
    `resp.read()` — avoids the chunked/brotli truncation trap of
    `resp.content.read(n)`).

    Test stubs that only implement `.json()` are handled via the fallback path.
    Returns None on decode failure or oversized response (caller logs).
    """
    read_method = getattr(resp, "read", None)
    if read_method is not None and callable(read_method):
        raw = await read_body_capped(resp, max_bytes=MAX_RESPONSE_BYTES, label=label)
        if raw is None:
            return None  # noqa: ASYNC910
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as e:
            logger.warning(f"{label} JSON decode failed: {e}")
            return None  # noqa: ASYNC910
    try:
        return await resp.json()
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"{label} JSON decode failed: {e}")
        return None  # noqa: ASYNC910


async def _http_get_with_backoff(
    session: Any,
    url: str,
    params: dict[str, str],
    *,
    max_attempts: int,
    retryable_statuses: Iterable[int] | None = None,
    label: str,
) -> Any | None:
    """GET `url` with `max_attempts` and a single bounded backoff between retries.

    Returns the parsed JSON body on 200, or None on retry exhaustion / non-200.
    Caps the response body at MAX_RESPONSE_BYTES so a runaway upstream can't
    blow up memory. Caps cumulative sleep so we don't exceed PLATFORM_HTTP_TIMEOUT;
    the s4_s5_union strategy already runs 2 queries per platform, so one-and-done
    retries suffice.

    `retryable_statuses` defaults to (403, 429, 500, 502, 503, 504). Statuses
    >= 500 are also treated as retryable.
    """
    retryable: set[int] = set(retryable_statuses or (403, 429, 500, 502, 503, 504))
    cumulative_sleep = 0.0
    timeout = aiohttp.ClientTimeout(total=PLATFORM_HTTP_TIMEOUT, sock_read=PLATFORM_HTTP_TIMEOUT)

    for attempt in range(max_attempts):
        try:
            async with session.get(url, params=params, timeout=timeout) as resp:
                status = resp.status
                if status in retryable or status >= 500:
                    if attempt + 1 >= max_attempts:
                        logger.warning(f"{label} HTTP {status} after {attempt + 1} attempts; giving up")
                        return None
                    # Budget-cap sleep against the per-platform timeout floor.
                    sleep_for = HTTP_RETRY_BACKOFF_SECS
                    if cumulative_sleep + sleep_for + PLATFORM_HTTP_TIMEOUT > PLATFORM_HTTP_TIMEOUT * max_attempts:
                        logger.warning(f"{label} HTTP {status}: sleep budget exhausted; giving up")
                        return None
                    logger.warning(f"{label} HTTP {status}; retry {attempt + 2}/{max_attempts} after {sleep_for:.2f}s")
                    await asyncio.sleep(sleep_for)
                    cumulative_sleep += sleep_for
                    continue
                if status != 200:
                    snippet = await read_body_snippet(resp)
                    logger.warning(f"{label} HTTP {status} non-retryable: {snippet}")
                    return None
                return await _read_json_capped(resp, label)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt + 1 >= max_attempts:
                logger.warning(f"{label} transient error after {attempt + 1} attempts: {e}")
                return None  # noqa: ASYNC910
            sleep_for = HTTP_RETRY_BACKOFF_SECS
            logger.warning(f"{label} transient error: {e}; retry {attempt + 2}/{max_attempts} after {sleep_for:.2f}s")
            await asyncio.sleep(sleep_for)
            cumulative_sleep += sleep_for
    return None  # noqa: ASYNC910


# ---------------------------------------------------------------------------
# Polymarket
# ---------------------------------------------------------------------------


def _parse_polymarket_matches(payload: Any, query: str = "") -> list[MarketMatch]:
    """Parse Gamma public-search response into MarketMatch objects.

    Gamma returns {events: [...], markets: [...]}. Each event may have nested
    markets with `outcomePrices` (JSON-encoded string or list). Take the first
    outcome's price as implied P(Yes).

    `query` is used to compute a fuzzy match-confidence score per row, so
    downstream filtering by confidence works uniformly across platforms.
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

    q_lower = (query or "").lower()

    events = payload.get("events") or []
    for ev in events[:10]:  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # top-N search results cap
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
        total_volume: float | None = None
        liquidity: float | None = None
        open_interest: float | None = None
        markets = ev.get("markets") or []
        if markets and isinstance(markets[0], dict):
            m0 = markets[0]
            implied = _prob_from_prices(m0.get("outcomePrices"))
            bid = _safe_float(m0.get("bestBid"))
            ask = _safe_float(m0.get("bestAsk"))
            vol_24h = _safe_float(m0.get("volume24hr"))
            # volumeNum is Gamma's total (all-time) volume; fall back to the
            # event-level or market-level `volume` when volumeNum is absent.
            total_volume = _safe_float(m0.get("volumeNum"))
            if total_volume is None:
                total_volume = volume if volume is not None else _safe_float(m0.get("volume"))
            liquidity = _safe_float(m0.get("liquidityNum"))
            if liquidity is None:
                liquidity = _safe_float(m0.get("liquidity"))
            open_interest = _safe_float(m0.get("openInterest"))
        else:
            total_volume = volume
        spread = (ask - bid) if (bid is not None and ask is not None) else None

        confidence = fuzz.token_set_ratio(q_lower, title.lower()) / 100.0 if q_lower and title else 0.0

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
                match_confidence=confidence,
                raw_rules=description[:2000],  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # rules-text truncation
                total_volume=total_volume,
                liquidity=liquidity,
                open_interest=open_interest,
            )
        )

    # Fallback to top-level markets if events were empty.
    if not out:
        markets = payload.get("markets") or []
        for m in markets[:10]:  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # top-N search results cap
            title = m.get("question") or m.get("title") or ""
            slug = m.get("slug") or ""
            url = f"https://polymarket.com/market/{slug}" if slug else ""
            implied = _prob_from_prices(m.get("outcomePrices"))
            confidence = fuzz.token_set_ratio(q_lower, title.lower()) / 100.0 if q_lower and title else 0.0
            total_volume = _safe_float(m.get("volumeNum"))
            if total_volume is None:
                total_volume = _safe_float(m.get("volume"))
            liquidity = _safe_float(m.get("liquidityNum"))
            if liquidity is None:
                liquidity = _safe_float(m.get("liquidity"))
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
                    match_confidence=confidence,
                    raw_rules=(m.get("description") or "")[:2000],
                    total_volume=total_volume,
                    liquidity=liquidity,
                    open_interest=_safe_float(m.get("openInterest")),
                )
            )

    return out


async def _polymarket_search(session: Any, query: str) -> list[MarketMatch] | None:
    """Search Polymarket for one query. ``None`` when the fetch itself failed.

    The None-vs-`[]` split is load-bearing: retry-exhausted 503s/429s (the dominant
    outage shape) would otherwise arrive at the caller as an ordinary empty result
    and be published as a benign `none`. `[]` means the search succeeded and parsed
    to nothing.
    """
    payload = await _http_get_with_backoff(
        session,
        POLYMARKET_SEARCH_URL,
        {"q": query, "limit_per_type": "10"},
        max_attempts=POLYMARKET_MAX_ATTEMPTS,
        label=f"Polymarket q={query[:40]!r}",  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # log-label truncation
    )
    if payload is None:
        return None
    return _parse_polymarket_matches(payload, query=query)


# ---------------------------------------------------------------------------
# Manifold
# ---------------------------------------------------------------------------


def _walk_tiptap_text(node: Any) -> list[str]:
    """Recursively collect all `text` string nodes from a TipTap/ProseMirror doc.

    Manifold's rich `description` is a nested `{type, content: [...], text: "..."}`
    document. We depth-first walk `content` arrays and gather leaf `text` strings.
    """
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


def _manifold_rules_text(m: dict) -> str:
    """Resolve a Manifold market's rules text, truncated to 2000 chars.

    Fallback chain: (1) `textDescription` if non-empty (the search endpoint
    usually omits it); (2) the flattened `description` TipTap doc; (3) the
    question title. A `str(...)` dump of the description is a pragmatic fallback
    if the doc shape is unexpected.
    """
    text_description = m.get("textDescription")
    if isinstance(text_description, str) and text_description.strip():
        return text_description[:2000]  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # rules-text truncation

    description = m.get("description")
    if isinstance(description, dict):
        collected = _walk_tiptap_text(description)
        if collected:
            return " ".join(collected)[:2000]
        return str(description)[:2000]
    if isinstance(description, str) and description.strip():
        return description[:2000]  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # rules-text truncation

    return (m.get("question") or "")[:2000]


def _parse_manifold_matches(payload: Any, query: str = "") -> list[MarketMatch]:
    if not isinstance(payload, list):
        logger.warning("Manifold returned non-list payload")
        return []

    q_lower = (query or "").lower()
    out: list[MarketMatch] = []
    for m in payload[:10]:  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # top-N search results cap
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

        confidence = fuzz.token_set_ratio(q_lower, title.lower()) / 100.0 if q_lower and title else 0.0

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
                match_confidence=confidence,
                raw_rules=_manifold_rules_text(m),
                total_volume=_safe_float(m.get("volume")),
                liquidity=_safe_float(m.get("totalLiquidity")),
                num_bettors=_safe_int(m.get("uniqueBettorCount")),
            )
        )
    return out


async def _manifold_search(session: Any, query: str) -> list[MarketMatch] | None:
    """Search Manifold for one query. ``None`` when the fetch itself failed
    (same contract as `_polymarket_search`); `[]` when it parsed to nothing."""
    payload = await _http_get_with_backoff(
        session,
        MANIFOLD_SEARCH_URL,
        {"term": query, "contractType": "BINARY", "limit": "10"},
        max_attempts=MANIFOLD_MAX_ATTEMPTS,
        retryable_statuses=(429, 500, 502, 503, 504),
        label=f"Manifold q={query[:40]!r}",  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # log-label truncation
    )
    if payload is None:
        return None
    return _parse_manifold_matches(payload, query=query)


# ---------------------------------------------------------------------------
# Kalshi (prefetch events + client-side fuzzy match)
# ---------------------------------------------------------------------------


async def _kalshi_prefetch_events(
    session: Any, event_limit: int = KALSHI_PREFETCH_EVENT_LIMIT, page_sleep_s: float = 1.0
) -> tuple[list[dict], _FetchTally]:
    """Paginate through open Kalshi events. Returns the events plus a per-page tally.

    Uses the `/events?with_nested_markets=true` endpoint NOT `/markets` --
    per G0, `/markets` is dominated by sports-parlay 'MVE' rows.

    The tally counts pages, so the caller can tell a total outage (no page came
    back) from an empty catalogue, and a truncated dump (some pages lost) from a
    complete one. A cache hit reports zero fetches — cached data is a success, not a
    degradation.

    Cache is updated INCREMENTALLY after each successful page so a cancelled
    prefetch still warms whatever pages completed. The FINAL write only happens on a
    clean pagination exit (cursor exhausted, or the event limit reached): writing it
    unconditionally pinned an error-truncated — often empty — list for the whole 6h
    TTL, so one transient blip on the first question starved every later question in
    the run.
    """
    cached = _KALSHI_CACHE.get("events")
    if cached is not None:
        ts, events = cached
        if (time.monotonic() - ts) < KALSHI_CACHE_TTL_S:
            return events, _FetchTally()  # noqa: ASYNC910

    params = {"status": "open", "limit": "200", "with_nested_markets": "true"}
    all_events: list[dict] = []
    cursor: str | None = None
    pages_ok = 0
    clean_exit = True

    while len(all_events) < event_limit:
        p = dict(params)
        if cursor:
            p["cursor"] = cursor
        try:
            async with session.get(KALSHI_EVENTS_URL, params=p, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    logger.warning("Kalshi 429 during prefetch; stopping pagination early")
                    clean_exit = False
                    break
                if resp.status != 200:
                    snippet = await read_body_snippet(resp)
                    logger.warning(f"Kalshi prefetch HTTP {resp.status}: {snippet}")
                    clean_exit = False
                    break
                data = await _read_json_capped(resp, "Kalshi prefetch")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Kalshi prefetch transient error: {e}")
            clean_exit = False
            break

        if data is None or not isinstance(data, dict):
            clean_exit = False
            break
        batch = data.get("events") or []
        if not isinstance(batch, list):
            clean_exit = False
            break
        pages_ok += 1
        all_events.extend([ev for ev in batch if isinstance(ev, dict)])
        # Incremental cache write: a cancelled prefetch warms whatever pages
        # made it through so the next call picks up where we left off.
        _KALSHI_CACHE["events"] = (time.monotonic(), list(all_events))
        cursor = data.get("cursor") or None
        if not cursor or not batch:
            break
        if page_sleep_s > 0:
            await asyncio.sleep(page_sleep_s)

    if clean_exit:
        _KALSHI_CACHE["events"] = (time.monotonic(), all_events)
    return all_events, _FetchTally(pages_ok, 0 if clean_exit else 1)  # noqa: ASYNC910


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
        total_volume: float | None = None
        open_interest: float | None = None
        liquidity: float | None = None
        is_resolved = False
        if nested and isinstance(nested[0], dict):
            first = nested[0]
            rules_primary = (first.get("rules_primary") or "")[:2000]
            close_time = _parse_iso(first.get("close_time") or "")
            yes_bid = _safe_float(first.get("yes_bid_dollars"))
            yes_ask = _safe_float(first.get("yes_ask_dollars"))
            volume_24h = _safe_float(first.get("volume_24h_fp"))
            total_volume = _safe_float(first.get("volume"))
            open_interest = _safe_float(first.get("open_interest"))
            liquidity = _safe_float(first.get("liquidity_dollars"))
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
                    raw_rules=rules_primary[:2000],  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # rules-text truncation
                    total_volume=total_volume,
                    open_interest=open_interest,
                    liquidity=liquidity,
                ),
            )
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [pm for _, pm in scored[:top_k]]


# ---------------------------------------------------------------------------
# Kalshi entity-based series retrieval (recall fix on top of fuzzy-over-prefetch)
# ---------------------------------------------------------------------------

# Capitalized tokens that are sentence scaffolding, not entities — dropped when they
# lead or stand alone in a proper-noun run (matched case-insensitively).
_ENTITY_STOPWORDS: frozenset[str] = frozenset(
    """will who what when where why how which whose is are was were do does did can could should
    would may might must shall the a an of in on at to for by with and or if then than as from into
    over under before after during between year years month months day days week weeks
    january february march april june july august september october november december
    """.split()
)

# Quoted spans (film / song / work titles) and word tokens for proper-noun runs.
_QUOTED_RE = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{2,60})[\"“”'‘’]")
_ENTITY_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&.\-]*")


def _extract_title_entities(question: Any) -> list[str]:
    """Deterministically extract salient entities from a question title — no LLM.

    Runs of consecutive proper tokens (Capitalized words or ALL-CAPS acronyms) collapse into
    one entity phrase ("Donald Trump", "BET Awards", "Best Male Athlete"); scaffolding words
    (Will/Who/The/months/...) break runs and are dropped. Quoted spans are captured verbatim.
    Ordered most-specific (longest) first, deduped case-insensitively, capped at
    KALSHI_ENTITY_MAX_ENTITIES.
    """
    title = getattr(question, "title", None) or getattr(question, "question_text", "") or ""
    entities: list[str] = []

    for m in _QUOTED_RE.finditer(title):
        span = m.group(1).strip()
        if span:
            entities.append(span)

    run: list[str] = []
    for tok in _ENTITY_TOKEN_RE.findall(title):
        is_proper = (tok[0].isupper() and any(c.isalpha() for c in tok)) or (tok.isupper() and tok.isalpha())
        if is_proper and tok.lower() not in _ENTITY_STOPWORDS:
            run.append(tok)
            continue
        if run:
            entities.append(" ".join(run))
            run = []
    if run:
        entities.append(" ".join(run))

    seen: set[str] = set()
    out: list[str] = []
    for e in sorted(entities, key=lambda phrase: len(phrase), reverse=True):
        key = e.lower()
        if len(e) >= 3 and key not in seen:
            seen.add(key)
            out.append(e)
    return out[:KALSHI_ENTITY_MAX_ENTITIES]


def _match_entities_to_series(
    entities: list[str], series: list[dict], *, min_score: float = KALSHI_ENTITY_SERIES_MIN_SCORE
) -> list[str]:
    """Fuzzy-match entities against series titles/tags; return distinct series tickers,
    most-confident first, capped at KALSHI_ENTITY_MAX_SERIES.

    token_set_ratio scores a full subset (the entity's tokens all present in the title)
    at ~100, so a short entity like "ESPY" cleanly matches its series title.
    """
    scored: dict[str, float] = {}
    for s in series:
        if not isinstance(s, dict):
            continue
        ticker = s.get("ticker") or ""
        title = (s.get("title") or "").lower()
        if not ticker or not title:
            continue
        tags = s.get("tags") or []
        tags_text = " ".join(str(t) for t in tags).lower() if isinstance(tags, list) else ""
        best = 0.0
        for e in entities:
            el = e.lower()
            score = float(fuzz.token_set_ratio(el, title))
            if tags_text:
                score = max(score, float(fuzz.token_set_ratio(el, tags_text)))
            best = max(best, score)
        if best >= min_score:
            scored[ticker] = max(scored.get(ticker, 0.0), best)
    return [t for t, _ in sorted(scored.items(), key=lambda kv: kv[1], reverse=True)][:KALSHI_ENTITY_MAX_SERIES]


async def _kalshi_fetch_series_attempt(session: Any, *, timeout_s: float) -> tuple[list[dict] | None, str, bool]:
    """One streamed /series request. Returns ``(series, reason, retryable)``.

    Parses at the ijson EVENT level (`ijson.parse_coro` yields prefix-annotated
    events) rather than materializing each series object, for two reasons:

    - Only the four fields entity-matching consumes are ever retained, so peak
      memory tracks the kept metadata (~kilobytes/series) instead of the raw
      payload, and the catalogue can grow without tripping a fixed size wall.
    - The events expose the TOP-LEVEL shape, which item-level extraction cannot
      see: an HTTP 200 carrying ``{"error": "temporarily unavailable"}`` or
      ``{"series": null}`` yields zero items, exactly like a legitimately empty
      catalogue. Without `saw_series_array` that lands in the 6h cache as a valid
      empty index and silently disables entity matching.

    ``retryable`` is True for transport failures and transient statuses, and for a
    missing ``series`` array (the "temporarily unavailable" 200 is a transient
    upstream state, and such a body is tiny). It is False for the size ceiling and
    for malformed JSON, where a second identical request just burns the budget.
    """
    kept: list[dict] = []
    saw_series_array = False

    @ijson.coroutine
    def _collect():  # noqa: ANN202  # ijson push-parser target (untyped ijson coroutine)
        nonlocal saw_series_array
        current: dict[str, Any] | None = None
        while True:
            prefix, event, value = yield
            if prefix == "series":
                if event == "start_array":
                    saw_series_array = True
                continue
            if prefix == "series.item":
                if event == "start_map":
                    current = {"ticker": None, "title": None, "category": None, "tags": []}
                elif event == "end_map":
                    if current is not None and current["ticker"] and current["title"]:
                        kept.append(current)
                    current = None
                continue
            if current is None or event in _IJSON_CONTAINER_EVENTS:
                continue
            if prefix == "series.item.ticker":
                current["ticker"] = value
            elif prefix == "series.item.title":
                current["title"] = value
            elif prefix == "series.item.category":
                current["category"] = value
            elif prefix == "series.item.tags.item":
                current["tags"].append(value)

    timeout = aiohttp.ClientTimeout(total=timeout_s, sock_read=timeout_s)
    total = 0
    try:
        async with session.get(KALSHI_SERIES_URL, params={}, timeout=timeout) as resp:
            if resp.status != 200:
                snippet = await read_body_snippet(resp)
                logger.warning(f"Kalshi series HTTP {resp.status}: {snippet}")
                retryable = resp.status in _KALSHI_SERIES_RETRYABLE_STATUSES or resp.status >= 500
                return None, f"error(http_{resp.status})", retryable
            parser = ijson.parse_coro(_collect())
            try:
                async for chunk in resp.content.iter_chunked(_KALSHI_SERIES_READ_CHUNK_BYTES):
                    total += len(chunk)
                    if total > KALSHI_SERIES_MAX_BYTES:
                        logger.warning(
                            f"Kalshi series body exceeded safety ceiling "
                            f"({total} bytes read > {KALSHI_SERIES_MAX_BYTES}); aborting stream"
                        )
                        return None, "dropped(size_cap)", False
                    parser.send(chunk)
                parser.close()
            except ijson.JSONError as e:
                logger.warning(f"Kalshi series stream parse failed: {type(e).__name__}: {e}")
                return None, "error(parse)", False
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.warning(f"Kalshi series transient error: {type(e).__name__}: {e}")
        return None, f"error({type(e).__name__})", True
    if not saw_series_array:
        logger.warning(
            f"Kalshi series payload carried no top-level 'series' array ({total} bytes); "
            f"treating as a lost index, not an empty catalogue"
        )
        return None, "error(no_series_array)", True
    return kept, "", False


async def _kalshi_fetch_series_streamed(session: Any) -> tuple[list[dict] | None, str]:
    """Stream the Kalshi /series body under a bounded retry, keeping only the four
    fields entity-matching consumes (see `_kalshi_fetch_series_attempt`).

    Retries once on a transient failure — a 503 or a connection reset would
    otherwise drop Kalshi entity recall for the whole question. Each attempt
    re-issues the request with a FRESH parser and accumulator, so a half-consumed
    stream is never resumed. KALSHI_SERIES_HTTP_TIMEOUT is a WALL-CLOCK budget
    shared by all attempts (each attempt gets what's left of it), so adding the
    retry cannot extend the time this fetch takes from the surrounding
    PREDICTION_MARKET_TIMEOUT the whole snapshot runs under.

    Returns ``(series, reason)``: ``(list, "")`` on success (list may be empty for a
    valid-but-empty catalogue), or ``(None, token)`` on failure, where ``token`` is a
    provider-diagnostics source token — ``"dropped(size_cap)"`` for the ceiling,
    ``"error(...)"`` for HTTP / transport / parse / payload-shape failure — so a lost
    series index surfaces per-question in the diagnostics line, not just in the run
    counter.
    """
    deadline = time.monotonic() + KALSHI_SERIES_HTTP_TIMEOUT
    reason = "error(unknown)"
    for attempt in range(KALSHI_SERIES_MAX_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            logger.warning(f"Kalshi series budget exhausted before attempt {attempt + 1}; giving up")
            break
        series, reason, retryable = await _kalshi_fetch_series_attempt(session, timeout_s=remaining)
        if series is not None:
            return series, ""
        if not retryable or attempt + 1 >= KALSHI_SERIES_MAX_ATTEMPTS:
            break
        if deadline - time.monotonic() <= HTTP_RETRY_BACKOFF_SECS:
            logger.warning(f"Kalshi series {reason}: retry budget exhausted; giving up")
            break
        logger.warning(
            f"Kalshi series {reason}; retry {attempt + 2}/{KALSHI_SERIES_MAX_ATTEMPTS} "
            f"after {HTTP_RETRY_BACKOFF_SECS:.2f}s"
        )
        await asyncio.sleep(HTTP_RETRY_BACKOFF_SECS)
    return None, reason


async def _kalshi_prefetch_series(session: Any, *, outcome_sink: dict[str, str] | None = None) -> list[dict]:
    """Fetch the Kalshi series list once (ticker/title/category/tags), cache ~6h.

    Kalshi exposes no full-text search; the series list is the sanctioned
    entity->ticker index. It arrives as one ever-growing unpaginated body, so we
    STREAM-PARSE it (`_kalshi_fetch_series_streamed`) rather than buffering.

    A fetch failure bumps the module degradation counter (so a silently-dead
    series path reddens CI) and is NOT cached, so the next question re-attempts.
    Soft-fails to [] — entity retrieval is best-effort on top of the
    fuzzy-over-prefetch path.

    ``outcome_sink`` (optional) receives a ``{"kalshi_series": token}`` entry — the
    series-index source outcome for the provider-diagnostics line, so a lost index
    (``dropped(size_cap)`` / ``error(...)``) is visible per-question, not only in
    the aggregate run counter. ``ok(N)`` when N series were available (fetched or
    cached), ``none`` for a valid-but-empty catalogue.
    """
    cached = _KALSHI_CACHE.get("series")
    if cached is not None:
        ts, series = cached
        if (time.monotonic() - ts) < KALSHI_CACHE_TTL_S:
            if outcome_sink is not None:
                outcome_sink["kalshi_series"] = f"ok({len(series)})" if series else "none"
            return series  # noqa: ASYNC910

    series, reason = await _kalshi_fetch_series_streamed(session)
    if series is None:
        _bump_kalshi_series_failure()
        if outcome_sink is not None:
            outcome_sink["kalshi_series"] = reason or "error(unknown)"
        return []  # noqa: ASYNC910
    if outcome_sink is not None:
        outcome_sink["kalshi_series"] = f"ok({len(series)})" if series else "none"
    _KALSHI_CACHE["series"] = (time.monotonic(), series)
    return series  # noqa: ASYNC910


async def _kalshi_events_for_series(session: Any, series_ticker: str) -> list[dict] | None:
    """Fetch a single series' open events (with nested markets), cached ~6h per ticker.

    Returns the same event-dict shape `_kalshi_search_local` already parses, or
    ``None`` when the fetch failed upstream (same contract as the other leaf
    fetchers, so the caller's tally sees the loss).
    """
    cache_key = f"events:{series_ticker}"
    cached = _KALSHI_CACHE.get(cache_key)
    if cached is not None:
        ts, events = cached
        if (time.monotonic() - ts) < KALSHI_CACHE_TTL_S:
            return events  # noqa: ASYNC910

    payload = await _http_get_with_backoff(
        session,
        KALSHI_EVENTS_URL,
        {"series_ticker": series_ticker, "status": "open", "with_nested_markets": "true", "limit": "200"},
        max_attempts=2,
        label=f"Kalshi events series={series_ticker}",
    )
    if payload is None:
        return None  # noqa: ASYNC910
    if not isinstance(payload, dict):
        return []  # noqa: ASYNC910
    events = payload.get("events")
    if not isinstance(events, list):
        return []  # noqa: ASYNC910
    events = [ev for ev in events if isinstance(ev, dict)]
    _KALSHI_CACHE[cache_key] = (time.monotonic(), events)
    return events  # noqa: ASYNC910


async def _kalshi_entity_matches(
    session: Any, question: Any, queries: list[str], *, top_k: int, outcome_sink: dict[str, str] | None = None
) -> tuple[list[MarketMatch], _FetchTally]:
    """Entity-targeted Kalshi retrieval, additive to the fuzzy-over-prefetch pass.

    Extracts salient title entities, matches them against the series list, fetches those
    series' open events, and fuzzy-scores them against the queries AND the entities (so a
    market retrieved because its series matched an entity survives even when the LLM query
    text does not). Closes the recall hole where the exact-title market is absent from / drowned
    in the capped prefetch dump. Soft-fails to [] at every boundary.

    The returned tally covers the per-series event fetches only; a lost series INDEX
    is reported separately under the ``kalshi_series`` token, so counting it here too
    would double-report it.

    ``outcome_sink`` is forwarded to `_kalshi_prefetch_series` to record the series-index
    source outcome for the provider-diagnostics line.
    """
    entities = _extract_title_entities(question)
    if not entities:
        return [], _FetchTally()
    series = await _kalshi_prefetch_series(session, outcome_sink=outcome_sink)
    if not series:
        return [], _FetchTally()
    tickers = _match_entities_to_series(entities, series)
    if not tickers:
        return [], _FetchTally()

    event_lists = await asyncio.gather(
        *[_kalshi_events_for_series(session, t) for t in tickers], return_exceptions=True
    )
    events: list[dict] = []
    fetches_ok = 0
    fetches_failed = 0
    for r in event_lists:
        if isinstance(r, list):
            events.extend(r)
            fetches_ok += 1
        elif isinstance(r, Exception):
            logger.warning(f"Kalshi entity events fetch raised: {r}")
            fetches_failed += 1
        else:
            fetches_failed += 1
    tally = _FetchTally(fetches_ok, fetches_failed)
    if not events:
        return [], tally

    matches: list[MarketMatch] = []
    for q in list(queries) + entities:
        matches.extend(_kalshi_search_local(events, q, top_k=top_k))
    return matches, tally


# ---------------------------------------------------------------------------
# PredictIt (prefetch full market dump + client-side fuzzy match)
# ---------------------------------------------------------------------------


async def _predictit_prefetch(session: Any) -> list[dict] | None:
    """Fetch the full PredictIt market dump. ``None`` when the fetch itself failed.

    PredictIt exposes a single unpaginated `/marketdata/all/` endpoint (no auth,
    no query param), so we fetch once, cache for ~6h, and fuzzy-match client-side
    (mirrors the Kalshi prefetch-and-local-match pattern).

    Same None-vs-`[]` contract as the other leaf fetchers: ``None`` is an upstream
    failure the caller must report as a lost source, `[]` is a successful fetch that
    yielded no usable markets.
    """
    cached = _PREDICTIT_CACHE.get("markets")
    if cached is not None:
        ts, markets = cached
        if (time.monotonic() - ts) < PREDICTIT_CACHE_TTL_S:
            return markets  # noqa: ASYNC910

    payload = await _http_get_with_backoff(
        session,
        PREDICTIT_URL,
        {},
        max_attempts=2,
        label="PredictIt prefetch",
    )
    if payload is None:
        return None  # noqa: ASYNC910
    if not isinstance(payload, dict):
        logger.warning("PredictIt returned non-dict payload")
        return []  # noqa: ASYNC910

    markets = payload.get("markets")
    if not isinstance(markets, list):
        logger.warning("PredictIt payload missing 'markets' list")
        return []  # noqa: ASYNC910

    markets = [m for m in markets if isinstance(m, dict)]
    _PREDICTIT_CACHE["markets"] = (time.monotonic(), markets)
    return markets  # noqa: ASYNC910


def _select_predictit_contract(contracts: list, q_lower: str) -> dict:
    """Pick the contract whose name best matches the query.

    A PredictIt market bundles one binary contract per outcome (candidate/party).
    Pricing `contracts[0]` blindly can attach the wrong outcome's price to a
    good market match; instead we fuzzy-match each contract name against the
    query and price the best one. Falls back to the first contract when nothing
    scores (single-contract binaries, or an empty query).
    """
    dict_contracts = [c for c in contracts if isinstance(c, dict)]
    if not dict_contracts:
        return {}
    if len(dict_contracts) == 1 or not q_lower:
        return dict_contracts[0]

    def _contract_score(c: dict) -> float:
        cname = (c.get("name") or c.get("shortName") or "").lower()
        return fuzz.token_set_ratio(q_lower, cname) if cname else 0.0

    return max(dict_contracts, key=_contract_score)


def _predictit_search_local(
    markets: list[dict], query: str, top_k: int = 5, min_score: float = PREDICTIT_MIN_FUZZY_SCORE
) -> list[MarketMatch]:
    """Fuzzy-match PredictIt markets by name/shortName; return top_k MarketMatch.

    PredictIt markets bundle multiple binary contracts (e.g. "which candidate
    wins", one contract per candidate). We emit ONE row per market (to avoid row
    explosion, mirroring Kalshi's single-row-per-event) and price the contract
    whose name best matches the query — so a "Trump wins 2028?" query surfaces
    the Trump contract's price, not whichever contract happens to be first. For a
    single-contract binary that just picks the sole contract. PredictIt carries no
    volume/liquidity/OI fields, so those stay None and the formatter renders
    `no-liquidity-data`.
    """
    q_lower = (query or "").lower()
    scored: list[tuple[float, MarketMatch]] = []

    for market in markets:
        if not isinstance(market, dict):
            continue
        name = market.get("name") or ""
        short_name = market.get("shortName") or ""
        if not name and not short_name:
            continue

        name_score = fuzz.token_set_ratio(q_lower, name.lower()) if name else 0.0
        short_score = fuzz.token_set_ratio(q_lower, short_name.lower()) if short_name else 0.0
        score = max(name_score, short_score)
        if score < min_score:
            continue

        # Boundary validation of external API data (mirrors the payload/markets isinstance
        # checks in _predictit_prefetch): a non-list here would TypeError past the narrow
        # per-platform catch and soft-fail the entire four-venue snapshot.
        contracts = market.get("contracts")
        if not isinstance(contracts, list):
            contracts = []
        contract = _select_predictit_contract(contracts, q_lower)
        contract_name = contract.get("name") or ""

        market_name = name or short_name
        # Disambiguate multi-contract markets by naming the contract we priced.
        if len(contracts) > 1 and contract_name and contract_name != market_name:
            title = f"{market_name} — {contract_name}"
        else:
            title = market_name

        status = (contract.get("status") or "").lower()
        is_resolved = status != "" and status != "open"

        # Price from the live order book when both sides exist (mirrors Kalshi):
        # bestBuyYesCost is the current YES ask; 1 - bestBuyNoCost is the YES bid.
        # lastTradePrice can be stale on thin markets, so it's only the fallback.
        yes_ask = _safe_float(contract.get("bestBuyYesCost"))
        no_ask = _safe_float(contract.get("bestBuyNoCost"))
        yes_bid = 1.0 - no_ask if no_ask is not None else None
        if yes_bid is not None and yes_ask is not None:
            implied = (yes_bid + yes_ask) / 2.0
            spread = yes_ask - yes_bid
        else:
            implied = _safe_float(contract.get("lastTradePrice"))
            spread = None

        scored.append(
            (
                score,
                MarketMatch(
                    platform="predictit",
                    market_title=title,
                    market_url=market.get("url") or "",
                    implied_prob_yes=implied,
                    bid=yes_bid,
                    ask=yes_ask,
                    spread=spread,
                    volume_24h=None,
                    close_time=None,
                    is_resolved=is_resolved,
                    match_confidence=score / 100.0,
                    raw_rules=f"{market_name} — {contract_name}" if contract_name else market_name,
                ),
            )
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    return [pm for _, pm in scored[:top_k]]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _as_of_cache_key(as_of: datetime | None) -> str:
    if as_of is None:
        return "none"
    return (
        as_of.astimezone(timezone.utc).isoformat() if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc).isoformat()
    )


async def fetch_market_snapshot(
    question: Any,
    *,
    platforms: tuple[str, ...] = ("polymarket", "kalshi", "manifold", "predictit"),
    max_matches_per_platform: int = 3,
    timeout: float = 5.0,  # noqa: ASYNC109
    as_of: datetime | None = None,
) -> MarketSnapshot:
    """Fan out to all four platforms in parallel, collect matches, apply filters.

    Soft-fails on any error: returns empty MarketSnapshot + WARNING log. A
    broken prediction-market API should never break a forecast run.

    `as_of` (backtest leakage defense): drops matches whose `close_time` is
    less than or equal to as_of. Required in backtest; optional in prod.
    """
    qid = getattr(question, "id_of_question", None)
    cache_key = (qid, _as_of_cache_key(as_of)) if qid is not None else None
    if cache_key is not None:
        cached_snap = _SNAPSHOT_CACHE.get(cache_key)
        if cached_snap is not None:
            return cached_snap  # noqa: ASYNC910

    # Session lifecycle: create the aiohttp session at the orchestrator level so
    # cleanup happens OUTSIDE the wait_for cancellation boundary. wait_for kills
    # inner work cleanly, then the surrounding context manager runs session.close().
    session_cm = _get_session()
    try:
        async with session_cm as session:
            try:
                snapshot = await asyncio.wait_for(
                    _fetch_market_snapshot_impl(
                        question,
                        session=session,
                        platforms=platforms,
                        max_matches_per_platform=max_matches_per_platform,
                        as_of=as_of,
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"Prediction-market snapshot TIMEOUT after {timeout}s for qid={qid}")
                # A whole-provider failure needs its own loss token: with an empty
                # `sources` map the diagnostics line renders no suffix at all, so a
                # dead snapshot is indistinguishable from one that was never asked for.
                # It also counts toward alertable_count — losing all four venues at
                # once is strictly worse than losing one, which already alerts.
                _bump_source_loss()
                return MarketSnapshot(matches=[], sources={"snapshot": "error(timeout)"})  # noqa: ASYNC910
    except Exception as e:  # HARNESS-SCAN-EXEMPT-broad-except
        # Outer safety net; should not normally fire -- investigate if seen.
        # Re-raise after logging would defeat the soft-fail contract that the
        # rest of the bot depends on, so we swallow + log here. Inner narrow
        # handlers in platform helpers cover the common paths.
        logger.warning("Prediction-market snapshot FAILED (soft-fail returning empty)", exc_info=True)
        _bump_source_loss()
        return MarketSnapshot(matches=[], sources={"snapshot": f"error({type(e).__name__})"})  # noqa: ASYNC910

    if cache_key is not None:
        _SNAPSHOT_CACHE[cache_key] = snapshot
    return snapshot


async def _fetch_market_snapshot_impl(
    question: Any,
    *,
    session: aiohttp.ClientSession,
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
        # No queries means keyword extraction came back empty (its LLM soft-fails to
        # ""), which silences all four platforms — a lost source, not a no-match, so
        # it gets both a loss token and an alertable bump.
        logger.warning(
            "Keyword extraction produced no queries (extractor LLM soft-failed); returning empty snapshot "
            "(alertable). NO venue was queried, so this is a keywords source loss, not a venue outage."
        )
        _bump_source_loss()
        return MarketSnapshot(matches=[], sources={"keywords": "error(no_queries)"})

    all_matches: list[MarketMatch] = []
    # Per-source outcome tokens for the provider-diagnostics line. The platform loop
    # writes each platform's token; the Kalshi closure writes `kalshi_series` (the
    # entity-index fetch) via this same dict as its outcome_sink. Single-threaded
    # asyncio => distinct keys, no race.
    sources: dict[str, str] = {}

    # Each closure returns its matches plus a _FetchTally, so an upstream outage that
    # reaches here as an empty match list is still distinguishable from a genuine
    # no-match (see _platform_source_token).
    async def _poly_for_all_queries() -> tuple[list[MarketMatch], _FetchTally]:
        tasks = [_polymarket_search(session, q) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return _flatten_results(results, "polymarket")

    async def _manifold_for_all_queries() -> tuple[list[MarketMatch], _FetchTally]:
        mf_queries = extractor.queries_for_platform(question, queries, "manifold")
        tasks = [_manifold_search(session, q) for q in mf_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return _flatten_results(results, "manifold")

    async def _kalshi_for_all_queries() -> tuple[list[MarketMatch], _FetchTally]:
        # Inner narrow handlers in _kalshi_prefetch_events are exhaustive
        # (ClientError, TimeoutError, ValueError, TypeError). No outer catch
        # here -- anything escaping is a programming bug we want loud.
        events, prefetch_tally = await _kalshi_prefetch_events(session, event_limit=KALSHI_PREFETCH_EVENT_LIMIT)
        merged: list[MarketMatch] = []
        for q in queries:
            merged.extend(_kalshi_search_local(events, q, top_k=max_matches_per_platform + 2))
        # Entity-targeted retrieval on top of fuzzy-over-prefetch: pulls the series/events
        # matching salient title entities (e.g. an exact ESPY / BET-Awards market) that the
        # capped prefetch dump misses. Additive; soft-fails to [] internally.
        entity_matches, entity_tally = await _kalshi_entity_matches(
            session, question, queries, top_k=max_matches_per_platform + 2, outcome_sink=sources
        )
        merged.extend(entity_matches)
        return merged, prefetch_tally + entity_tally

    async def _predictit_for_all_queries() -> tuple[list[MarketMatch], _FetchTally]:
        # Prefetch the full market dump once, then fuzzy-match each query locally
        # (mirrors the Kalshi prefetch-and-local-match pattern).
        markets = await _predictit_prefetch(session)
        if markets is None:
            return [], _FetchTally(failed=1)
        merged: list[MarketMatch] = []
        for q in queries:
            merged.extend(_predictit_search_local(markets, q, top_k=max_matches_per_platform + 2))
        return merged, _FetchTally(ok=1)

    platform_tasks: list[tuple[str, asyncio.Task]] = []
    if "polymarket" in platforms:
        platform_tasks.append(("polymarket", asyncio.create_task(_poly_for_all_queries())))
    if "manifold" in platforms:
        platform_tasks.append(("manifold", asyncio.create_task(_manifold_for_all_queries())))
    if "kalshi" in platforms:
        platform_tasks.append(("kalshi", asyncio.create_task(_kalshi_for_all_queries())))
    if "predictit" in platforms:
        platform_tasks.append(("predictit", asyncio.create_task(_predictit_for_all_queries())))

    lost_platforms: list[str] = []
    for platform, task in platform_tasks:
        try:
            matches, tally = await task
            sources[platform] = _platform_source_token(matches, tally)
            platform_lost = tally.failed > 0
        except (aiohttp.ClientError, OSError, RuntimeError) as e:
            # Inner platform helpers each call asyncio.gather(..., return_exceptions=True)
            # so coroutines don't propagate. This narrow catch covers residual
            # transport/runtime errors only. AttributeError/TypeError remain bugs.
            logger.warning(f"Platform {platform} failed (soft-fail): {type(e).__name__}: {e}")
            matches = []
            sources[platform] = f"error({type(e).__name__})"
            platform_lost = True
        if platform_lost:
            lost_platforms.append(f"{platform}={sources[platform]}")
            _bump_source_loss()
        all_matches.extend(matches)

    if lost_platforms:
        # One WARN naming every degraded venue: this counter reddens CI, and a red
        # run whose cause isn't named in the log is what teaches people to ignore alerts.
        logger.warning(f"Prediction-market platforms degraded (alertable): {', '.join(lost_platforms)}")

    # Dedup within-platform by market_url (or title fallback), cap per platform.
    by_platform: dict[str, list[MarketMatch]] = {"polymarket": [], "kalshi": [], "manifold": [], "predictit": []}
    seen_urls_per_platform: dict[str, set[str]] = {
        "polymarket": set(),
        "kalshi": set(),
        "manifold": set(),
        "predictit": set(),
    }

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
    for plat in ("polymarket", "kalshi", "manifold", "predictit"):
        combined.extend(by_platform[plat])

    return MarketSnapshot(matches=combined, sources=sources)


def _flatten_results(results: list[Any], platform: str) -> tuple[list[MarketMatch], _FetchTally]:
    """Flatten per-query search results and tally how many sub-queries were lost.

    A leaf search signals upstream failure with ``None`` and a successful-but-empty
    parse with `[]`, so anything that isn't a list — a raised task or a ``None`` —
    counts as a lost sub-query rather than being logged and discarded.
    """
    out: list[MarketMatch] = []
    queries_ok = 0
    queries_failed = 0
    for r in results:
        if isinstance(r, list):
            out.extend(r)
            queries_ok += 1
        elif isinstance(r, Exception):
            logger.warning(f"{platform} query task raised: {r}")
            queries_failed += 1
        else:
            queries_failed += 1
    return out, _FetchTally(queries_ok, queries_failed)


# ---------------------------------------------------------------------------
# Formatter — relevance gate (content-word overlap + matcher conf)
# ---------------------------------------------------------------------------

# Content-word stopwords for the relevance gate. Kept byte-identical to the tuning script
# (scratch/new_analyses_2026-07-18/market_match_precision.py `_overlap`) so the shipped labels
# match the 403-contract grading the thresholds were chosen on.
_RELEVANCE_STOPWORDS: frozenset[str] = frozenset(
    """a an the of in on at to for by with will be is are was were before after during between
    and or not no yes if then than as from into over under above below more less most least
    what which who whom whose when where why how this that these those there here it its
    do does did done have has had having get gets got question market resolve resolves resolved
    resolution against per any all each both other another same different new old first last
    2025 2026 2027 january february march april may june july august september october november december
    """.split()
)


def _relevance_content_words(text: str | None) -> set[str]:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {w for w in words if len(w) >= 3 and w not in _RELEVANCE_STOPWORDS}


def _market_relevance_overlap(question_words: set[str], match: MarketMatch) -> int:
    match_words = _relevance_content_words(match.market_title) | _relevance_content_words(match.raw_rules)
    return len(question_words & match_words)


def _is_likely_relevant(overlap: int, confidence: float) -> bool:
    return overlap >= MARKET_RELEVANCE_OVERLAP_MIN and confidence >= MARKET_RELEVANCE_CONF_MIN


# Shared trailing legend (the table's signal + relevance columns are present in both branches).
_MARKET_SIGNAL_LEGEND = (
    "The `signal` column labels each market's liquidity/participation "
    "(thin/decent/deep or thin/decent/high); the raw total volume and open interest are shown alongside. Treat "
    "deep/high-liquidity markets as a strong anchor and discount thin ones (low volume, few participants) as noisy. "
    "The `relevance` column flags whether the fuzzy match cleared a content-overlap + confidence bar for THIS "
    "question (`likely-relevant`) or did not (`verify-carefully`)."
)

# Strong-evidence framing — used ONLY when >=1 contract is likely-relevant to THIS question.
_MARKET_PREAMBLE_STRONG = (
    "The following prediction markets MAY be relevant — the match below is fuzzy, so verify each market's "
    "resolution criteria, resolution date, and topic against THIS question before weighting. A market whose "
    "criteria and date match this question is extremely strong evidence — anchor on its price. A market on a "
    "related but different event, different date, or different criteria carries proportionally less weight — "
    "name the specific mismatch and discount accordingly; a poorly-matched market may be worth little or "
    "nothing. "
)

# Neutral framing — used when NO contract clears the relevance bar (matches are likely all off-topic).
_MARKET_PREAMBLE_NEUTRAL = (
    "The following prediction markets were fuzzy-matched to this question and may all be off-topic — none "
    "cleared the relevance bar for THIS question, so treat them as leads to verify, not as evidence. Weight a "
    "market only after you confirm its resolution criteria, date, and topic match this question; otherwise "
    "disregard it. "
)


def format_snapshot_for_research(
    snapshot: MarketSnapshot,
    *,
    question_title: str | None = None,
    resolution_criteria: str | None = None,
) -> str:
    """Compact markdown block for the research prompt.

    Emits a table + raw-rules section. Each row carries a `relevance` label
    (`likely-relevant` / `verify-carefully`) from a content-word overlap + matcher-confidence
    bar against THIS question, and NOTHING is dropped. The strong-evidence preamble is used only
    when >=1 contract clears the bar; otherwise a neutral "these are leads to verify, not
    evidence" preamble is used, so an authoritative header never sits on an all-off-topic table.

    Without question context (``question_title`` / ``resolution_criteria`` unset), overlap can't
    be computed, so every row is `verify-carefully` and the neutral preamble is used — the
    conservative default. The production caller always passes question context.
    """
    if not snapshot.matches:
        return ""

    question_words = _relevance_content_words(question_title) | _relevance_content_words(resolution_criteria)
    relevances = [
        _is_likely_relevant(_market_relevance_overlap(question_words, m), m.match_confidence) for m in snapshot.matches
    ]
    any_relevant = any(relevances)

    lines: list[str] = []
    preamble = _MARKET_PREAMBLE_STRONG if any_relevant else _MARKET_PREAMBLE_NEUTRAL
    lines.append(preamble + _MARKET_SIGNAL_LEGEND)
    lines.append("")
    lines.append("| platform | title | prob | total_vol | OI | signal | close | conf | relevance |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for m, is_rel in zip(snapshot.matches, relevances):
        prob = f"{m.implied_prob_yes:.2f}" if m.implied_prob_yes is not None else "-"
        total_vol = f"{m.total_volume:.0f}" if m.total_volume is not None else "-"
        oi = f"{m.open_interest:.0f}" if m.open_interest is not None else "-"
        signal = _liquidity_label(m)
        close = m.close_time.strftime("%Y-%m-%d") if m.close_time else "-"
        conf = f"{m.match_confidence:.2f}"
        relevance = "likely-relevant" if is_rel else "verify-carefully"
        safe_title = (m.market_title or "")[:80].replace("|", "/")
        lines.append(
            f"| {m.platform} | {safe_title} | {prob} | {total_vol} | {oi} | {signal} | {close} | {conf} | {relevance} |"
        )

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


def prediction_market_provider(is_benchmarking: bool = False) -> ResearchCallable:
    """Factory returning an async research callable for prediction-market data.

    The returned callable accepts a `MetaculusQuestion` and uses its full API:
    `id_of_question` for caching, `question_text` / `resolution_criteria` for
    keyword extraction, `scheduled_resolution_time` for backtest leakage defense.

    Gated on PREDICTION_MARKETS_ENABLED env flag; disabled returns "".

    F7: when ``is_benchmarking=True`` the provider hard-disables regardless of
    the env flag. The ``as_of`` filter only drops markets that closed BEFORE
    ``as_of``; still-open markets and markets that closed between ``as_of`` and
    now would leak post-``as_of`` information into a backtest. The benchmarking
    guard is the only safe defense — see CLAUDE.md and the
    ``gemini_search_provider`` / ``native_search_provider`` precedents.
    """

    async def _fetch(question: MetaculusQuestion) -> str:
        if is_benchmarking:
            return ""  # noqa: ASYNC910
        if not env_flag_enabled(PREDICTION_MARKETS_ENABLED_ENV):
            return ""  # noqa: ASYNC910

        scheduled = getattr(question, "scheduled_resolution_time", None)
        if isinstance(scheduled, datetime):
            as_of = scheduled - AS_OF_DEFAULT_BUFFER
        else:
            as_of = datetime.now(timezone.utc)

        snapshot = await fetch_market_snapshot(question, as_of=as_of, timeout=PREDICTION_MARKET_TIMEOUT)
        # Surface per-source outcomes so the orchestrator's diagnostics line shows
        # partial degradation (a live platform while a sub-source silently died).
        # Recorded here at the ResearchCallable boundary so it's keyed to the qid the
        # orchestrator pops; no-op when qid is None.
        record_provider_detail(
            getattr(question, "id_of_question", None),
            "prediction_market",
            {"sources": snapshot.sources},
        )
        record_raw_research(
            qid=getattr(question, "id_of_question", None),
            provider="prediction_market",
            payload=snapshot,
        )
        return format_snapshot_for_research(
            snapshot,
            question_title=getattr(question, "title", None) or getattr(question, "question_text", None),
            resolution_criteria=getattr(question, "resolution_criteria", None),
        )

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


def _safe_int(v: Any) -> int | None:
    f = _safe_float(v)
    # json.loads (used by aiohttp) accepts bare NaN/Infinity literals, and int(nan)/int(inf)
    # raise — a helper named _safe_* must return None on those, not blow up.
    if f is None or not math.isfinite(f):
        return None
    return int(f)


def _parse_iso(s: Any) -> datetime | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        # Python 3.11 fromisoformat accepts 'Z' suffix as of 3.11
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
