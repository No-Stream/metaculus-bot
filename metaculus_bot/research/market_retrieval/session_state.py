"""Per-session caches, per-run degradation counters, the session factory, and the two prefetches.

Split out of the `prediction_market` seam so that module holds only the LLM stages, the snapshot
orchestrator and the provider factory. Everything here is process-scoped state plus the code that
reads or writes it: the two whole-catalogue caches and the two coroutines that fill them, the
snapshot cache, the aiohttp session factory, and the two counters that are the ONLY route by which
a soft-failing provider reddens CI.

The state lives in exactly one place — here — and `prediction_market` re-exports the accessors and
the cache dicts, because the seam is what every consumer and every test reaches for. So a patch or
an assertion against `prediction_market._KALSHI_CACHE` still touches this module's single dict, and
`prediction_market._get_session` is still the patch point the snapshot orchestrator reads.
"""

from __future__ import annotations

import time
from typing import Any

import aiohttp

from metaculus_bot.research.http_fetch import build_session
from metaculus_bot.research.market_retrieval import venues
from metaculus_bot.research.market_retrieval.http import PLATFORM_HTTP_TIMEOUT
from metaculus_bot.research.market_retrieval.types import MarketSnapshot, _FetchTally
from metaculus_bot.research.provider_health import record_catalogue_size

# Catalogue cache TTLs. Both venues are enumerated whole and re-enumerating them per question
# would dominate the snapshot's budget, so the pull is once per process per 6h.
KALSHI_CACHE_TTL_S = 6 * 60 * 60
PREDICTIT_CACHE_TTL_S = 6 * 60 * 60


# ---------------------------------------------------------------------------
# Per-session caches (module-scoped; reset via `_reset_session_caches`)
# ---------------------------------------------------------------------------

# Kalshi projected-events catalogue: (timestamp_monotonic, events_list).
_KALSHI_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
# PredictIt markets cache: (timestamp_monotonic, markets_list).
_PREDICTIT_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
# Snapshot cache keyed by (qid, as_of_iso). The as_of leg keeps backtest runs at different
# as-of instants from sharing a snapshot computed at one as-of; on the provider path as_of is
# None, which is what finally makes this cache hittable (the old `datetime.now(utc)` branch
# changed the key on every call).
_SNAPSHOT_CACHE: dict[tuple[int, str], MarketSnapshot] = {}

# Per-run count of failed Kalshi CATALOGUE pulls. The catalogue is the generation backbone —
# it feeds both the settlement-source join and the fuzzy channel — and the provider soft-fails,
# so without a counter a dead pull is INVISIBLE (the 2026-07-25 observability hole:
# research_provider_failures=0 while a Kalshi path was dead). The orchestrator folds this into
# alertable_count, so a catalogue that dies every question reddens CI. A one-off transient
# bumps it once, an accepted rare false alarm mirroring gap_fill_v2_error_count.
#
# This counter and `_SOURCE_LOSSES` both bump on a failed pull, so one catalogue outage adds 2
# to alertable_count. That is deliberate over-counting rather than a bug: the two counters
# carry different marker fields, and the point of either is that CI goes red.
#
# Module-level like the caches => accumulates per run; reset between tests.
_KALSHI_CATALOGUE_FETCH_FAILURES: int = 0


def _bump_kalshi_catalogue_failure() -> None:
    global _KALSHI_CATALOGUE_FETCH_FAILURES  # noqa: PLW0603  # per-run counter for a stateless provider; see the constant's comment
    _KALSHI_CATALOGUE_FETCH_FAILURES += 1


def kalshi_catalogue_fetch_failures() -> int:
    """Per-run count of failed Kalshi catalogue pulls (folded into alertable_count)."""
    return _KALSHI_CATALOGUE_FETCH_FAILURES


def reset_series_degradation_counter() -> None:
    """Zero the catalogue-degradation counter at run start.

    The provider is a stateless callable, so the counter lives at module scope; without a
    run-start reset it would leak across runs sharing one process (and across tests, polluting
    every later alertable_count == 0 assertion). Called from forecast_questions alongside
    reset_pchip_stats — same per-run cadence. The name is unchanged from when this counter
    tracked the retired /series fetch, because the orchestrator imports it by name."""
    global _KALSHI_CATALOGUE_FETCH_FAILURES  # noqa: PLW0603  # module-scoped counter needs a run-start reset; see docstring
    _KALSHI_CATALOGUE_FETCH_FAILURES = 0


# Per-run count of LOST prediction-market SOURCES: one per venue whose fan-out lost a
# sub-fetch, one per whole-provider failure (snapshot timeout, outer-except), one when the
# query author comes back unusable, and one when the ranking call does. Those last two are why
# this counts SOURCES rather than venues: a dead ranker degrades every venue's contribution
# without any venue failing. The causes are distinguished per-source in
# `MarketSnapshot.sources` (`ranking:error(...)` vs `polymarket:error(...)`), which rides both
# the published comment and the schema-v2 research archive.
#
# Operator decision 2026-07-25: alert on ANY source loss rather than only a total blackout —
# maximum sensitivity, accepting that one flaky venue can redden most runs. The provider
# soft-fails internally, so like the catalogue counter this is the only path by which an outage
# reaches CI. Module-level like the caches => accumulates per run.
_SOURCE_LOSSES: int = 0


def _bump_source_loss() -> None:
    global _SOURCE_LOSSES  # noqa: PLW0603  # per-run counter for a stateless provider; see the constant's comment
    _SOURCE_LOSSES += 1


def prediction_market_source_losses() -> int:
    """Per-run count of lost prediction-market sources (folded into alertable_count)."""
    return _SOURCE_LOSSES


def reset_source_loss_counter() -> None:
    """Zero the source-loss counter at run start (see
    `reset_series_degradation_counter` for why module-scoped counters need this)."""
    global _SOURCE_LOSSES  # noqa: PLW0603  # module-scoped counter needs a run-start reset; see docstring
    _SOURCE_LOSSES = 0


def _reset_session_caches() -> None:
    """Clear all per-session caches. Called between tests and at session start."""
    global _KALSHI_CATALOGUE_FETCH_FAILURES, _SOURCE_LOSSES  # noqa: PLW0603  # per-session caches/counters live at module scope; tests reset them here
    _KALSHI_CACHE.clear()
    _PREDICTIT_CACHE.clear()
    _SNAPSHOT_CACHE.clear()
    _KALSHI_CATALOGUE_FETCH_FAILURES = 0
    _SOURCE_LOSSES = 0


def _get_session() -> aiohttp.ClientSession:
    """Construct a fresh aiohttp session. Patched in tests.

    No headers arg: the JSON APIs get aiohttp's default UA (flipping to a browser UA is a
    separate experiment — see the resolution-source plan).
    """
    return build_session(timeout_s=PLATFORM_HTTP_TIMEOUT)


# ---------------------------------------------------------------------------
# Stage 1a — catalogue prefetches
# ---------------------------------------------------------------------------


async def _kalshi_catalogue(session: Any, *, qid: int | None) -> tuple[list[dict[str, Any]], str]:
    """The complete projected Kalshi open-events catalogue, cached ~6h.

    Returns ``(events, source_token)``; the token is what says whether the pull succeeded, so
    there is nothing for a separate boolean to disagree with. The completeness-gated write below
    is the ONLY writer, deliberately: the read path checks the TTL and nothing else, so any
    incremental warm would pin an error-truncated — often EMPTY — list carrying a fresh
    timestamp, and every later question in the run would then read it back as a healthy
    `ok(N)` with no HTTP and no counter bump. A partial pull is still returned to THIS question,
    which is what keeps a lost page from costing the caller the pages that did arrive.
    """
    cached = _KALSHI_CACHE.get("events")
    if cached is not None:
        timestamp, events = cached
        if (time.monotonic() - timestamp) < KALSHI_CACHE_TTL_S:
            if qid is not None:
                record_catalogue_size(qid=qid, source="kalshi_events", entries=len(events), fetch_ok=True)
            return events, f"ok({len(events)})" if events else "none"

    pull = await venues.kalshi_prefetch_events(session)
    if pull.complete:
        _KALSHI_CACHE["events"] = (time.monotonic(), pull.events)
    else:
        _bump_kalshi_catalogue_failure()

    # A pull that reports SUCCESS and hands pool assembly an empty catalogue is a contradiction
    # the pool size alone cannot show (it looks identical to "the venue had nothing to say"),
    # and an empty catalogue now zeroes the settlement join AND the fuzzy channel.
    if qid is not None:
        record_catalogue_size(qid=qid, source="kalshi_events", entries=len(pull.events), fetch_ok=pull.complete)
    if not pull.complete:
        return pull.events, pull.token or "error(unknown)"
    return pull.events, f"ok({len(pull.events)})" if pull.events else "none"


async def _predictit_universe(session: Any, *, qid: int | None) -> tuple[list[dict[str, Any]], _FetchTally]:
    """PredictIt's whole ~197-market dump, cached ~6h. One unpaginated GET.

    The tally carries the None-vs-`[]` distinction forward: a failed fetch is a lost source,
    while a successful fetch of an empty dump is Signal C's business.
    """
    cached = _PREDICTIT_CACHE.get("markets")
    if cached is not None:
        timestamp, markets = cached
        if (time.monotonic() - timestamp) < PREDICTIT_CACHE_TTL_S:
            if qid is not None:
                record_catalogue_size(qid=qid, source="predictit_markets", entries=len(markets), fetch_ok=True)
            return markets, _FetchTally(ok=1)

    markets = await venues.predictit_prefetch(session)
    if markets is None:
        # A failed fetch is already the source-loss counter's business; recording a catalogue
        # observation here too would double-count one outage (see provider_health Signal C).
        return [], _FetchTally(failed=1)
    if qid is not None:
        record_catalogue_size(qid=qid, source="predictit_markets", entries=len(markets), fetch_ok=True)
    _PREDICTIT_CACHE["markets"] = (time.monotonic(), markets)
    return markets, _FetchTally(ok=1)
