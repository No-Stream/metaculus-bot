"""
Central configuration constants to avoid magic numbers and strings.

These are intentionally minimal and focused on operational tuning knobs that
need to be shared across modules.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Tuple

from metaculus_bot.config import load_environment

# =============================================================================
# TOURNAMENT IDs - UPDATE THESE EACH QUARTER/SEASON
# =============================================================================
# AI Forecasting Benchmark tournament (bot-only competition)
# Update when new season starts: https://www.metaculus.com/project/aib/
TOURNAMENT_ID: str = "spring-aib-2026"  # Spring 2026 AI Benchmarking
TOURNAMENT_END_DATE: str = "2026-05-06"  # Approximate end date for warning/error checks
TOURNAMENT_HARD_STOP_WEEKS: int = 3  # Error out this many weeks after end date

# Metaculus Cup tournament (human + bot competition)
# Update when new cup starts: https://www.metaculus.com/tournament/metaculus-cup/
METACULUS_CUP_ID: str = "metaculus-cup"  # Uses slug, auto-resolves to current cup


class TournamentExpiredError(Exception):
    """Raised when the tournament has ended and the ID needs to be updated."""

    pass


def check_tournament_dates(logger: logging.Logger | None = None) -> None:
    """Check if tournament dates are stale and warn/error accordingly.

    - Warns if current date is past TOURNAMENT_END_DATE
    - Raises TournamentExpiredError if past end date + TOURNAMENT_HARD_STOP_WEEKS

    Call this at bot startup to catch stale tournament IDs.
    """
    import logging as _logging

    log = logger or _logging.getLogger(__name__)

    try:
        end_date = datetime.strptime(TOURNAMENT_END_DATE, "%Y-%m-%d")
    except ValueError:
        log.warning(f"Invalid TOURNAMENT_END_DATE format: {TOURNAMENT_END_DATE}")
        return

    today = datetime.now()
    hard_stop_date = end_date + timedelta(weeks=TOURNAMENT_HARD_STOP_WEEKS)

    if today > hard_stop_date:
        raise TournamentExpiredError(
            f"Tournament '{TOURNAMENT_ID}' ended on {TOURNAMENT_END_DATE} and hard stop "
            f"date ({hard_stop_date.date()}) has passed. Please update TOURNAMENT_ID, "
            f"TOURNAMENT_END_DATE, and TOURNAMENT_HARD_STOP_WEEKS in constants.py for the new season."
        )
    elif today > end_date:
        days_past = (today - end_date).days
        days_until_error = (hard_stop_date - today).days
        log.warning(
            f"⚠️  Tournament '{TOURNAMENT_ID}' likely ended on {TOURNAMENT_END_DATE} "
            f"({days_past} days ago). Update constants.py for the new season! "
            f"Bot will error out in {days_until_error} days."
        )


# Load .env early so ASKNEWS_* values are read correctly at import time in local runs
load_environment()

# Concurrency tuning for research providers (e.g., AskNews, Exa)
# Start conservatively for AskNews; adjust after observing rate limits.
DEFAULT_MAX_CONCURRENT_RESEARCH: int = 1

# Benchmark driver settings
# Default batch size for benchmarking runs
# Keep this modest to balance concurrency and rate limits.
BENCHMARK_BATCH_SIZE: int = 4

# Metaculus comment safety limits
REPORT_SECTION_CHAR_LIMIT: int = 49_999
COMMENT_CHAR_LIMIT: int = 149_999

# Optional environment variable to force research provider selection.
# Accepted values (case-insensitive): "auto", "asknews", "exa", "perplexity", "openrouter"
RESEARCH_PROVIDER_ENV: str = "RESEARCH_PROVIDER"


def env_flag_enabled(env_name: str) -> bool:
    """Return True iff env var is set to "true"/"1"/"yes" (case-insensitive)."""
    return os.getenv(env_name, "").lower() in ("true", "1", "yes")


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


# AskNews provider safety limits (global, across all bots in-process)
# Defaults are conservative for pro plans (1 RPS sustained, 5 RPS burst, 5 concurrency)
ASKNEWS_MAX_CONCURRENCY: int = max(1, _int_env("ASKNEWS_MAX_CONCURRENCY", 1))
# Conservative sustained rate well below pro plan limits (1 RPS sustained)
ASKNEWS_MAX_RPS: float = max(0.1, _float_env("ASKNEWS_MAX_RPS", 0.8))

# Retry tuning for AskNews
ASKNEWS_MAX_TRIES: int = max(1, _int_env("ASKNEWS_MAX_TRIES", 3))
ASKNEWS_BACKOFF_SECS: float = max(0.0, _float_env("ASKNEWS_BACKOFF_SECS", 2.0))
# Hard wall-clock bound around the full AskNews provider (hot+historical+sleeps+retries).
# AskNews's internal retry loop fails fast on non-retryable errors, but a network
# hang is otherwise unbounded; this backstops that case so a stuck AskNews call
# can't hold the whole research phase hostage.
#
# Sizing: each phase (hot + historical) sleeps 10.1s before its first call and
# applies backoff `2.0 * (10 + 3**attempt)` on 429/rate-limit retries — attempt
# 2 ≈ 38s, attempt 3 ≈ 74s. With 3 tries per phase the retry worst case is
# ~110s hot + ~110s historical + ~30s API time ≈ 250s, so 300s leaves ~20%
# headroom above the normal retry envelope while still bounding a genuine hang.
ASKNEWS_WALL_TIMEOUT: int = 300

# --- Forecasting clamps and numeric smoothing ---
# Binary prediction clamp
BINARY_PROB_MIN: float = 0.01
BINARY_PROB_MAX: float = 0.99

# Multiple-choice prediction clamp
MC_PROB_MIN: float = 0.005
MC_PROB_MAX: float = 0.995

# Numeric CDF smoothing and spacing
NUM_VALUE_EPSILON_MULT: float = 1e-9
NUM_SPREAD_DELTA_MULT: float = 1e-6
NUM_MIN_PROB_STEP: float = 5e-5
NUM_MAX_STEP: float = 0.2
NUM_RAMP_K_FACTOR: float = 3.0

# Discrete integer CDF snapping (for "continuous" questions with integer outcomes)
DISCRETE_SNAP_MAX_INTEGERS: int = 200
DISCRETE_SNAP_UNIFORM_MIX: float = 0.0

# --- Conditional Stacking Thresholds ---
# Binary: probability range (max − min) across per-model predictions. Chosen because
# log-odds spread saturates on clamped-extreme models that are often correct,
# conflating "one model is sure" with "ensemble is split."
CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD: float = 0.15
# Multiple choice: max per-option probability spread (max - min across models for worst option).
CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD: float = 0.20
# Numeric: max percentile spread normalized by question range (at 10th/50th/90th percentiles).
CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD: float = 0.15

# --- Native Search Provider ---
# Environment variable names
NATIVE_SEARCH_ENABLED_ENV: str = "NATIVE_SEARCH_ENABLED"
NATIVE_SEARCH_MODEL_ENV: str = "NATIVE_SEARCH_MODEL"
# Default model for native search (without openrouter/ prefix)
NATIVE_SEARCH_DEFAULT_MODEL: str = "x-ai/grok-4.1-fast"
# LLM parameters for native search (lower temp for factual grounding)
NATIVE_SEARCH_TEMPERATURE: float = 0.3
NATIVE_SEARCH_TOP_P: float = 0.9
NATIVE_SEARCH_MAX_TOKENS: int = 16_000
# 4 min. Observed p99 of Grok native search ≈ 48s; 240s leaves ~5x headroom
# without letting a stuck upstream dominate a batch run.
NATIVE_SEARCH_TIMEOUT: int = 240
# Native search web options (passed to OpenRouter plugins)
NATIVE_SEARCH_MAX_RESULTS: int = 20
NATIVE_SEARCH_CONTEXT_SIZE: str = "high"  # "low", "medium", "high"

# --- Gemini Search Provider (Google AI Studio direct SDK) ---
# Uses google-genai SDK with GoogleSearch grounding tool for first-party Google
# Search results (distinct from OpenRouter's Exa-backed :online plugin). Adds a
# genuinely new search index to the ensemble.
GEMINI_SEARCH_ENABLED_ENV: str = "GEMINI_SEARCH_ENABLED"
GEMINI_SEARCH_MODEL_ENV: str = "GEMINI_SEARCH_MODEL"
GOOGLE_API_KEY_ENV: str = "GOOGLE_API_KEY"
# Gemini 3 Flash preview model with grounding support. Requires billing enabled
# on the Google AI Studio project to unlock; falls back to gemini-2.5-flash on
# free tier if needed. Override via GEMINI_SEARCH_MODEL env var.
GEMINI_SEARCH_DEFAULT_MODEL: str = "gemini-3-flash-preview"
# No temperature / top_p / max_tokens overrides — use google-genai SDK defaults.
# Gemini 3 Flash is a thinking model; Google's defaults are tuned for it and
# capping either caused silent truncations in the past.
# 3 min. Observed p99 of Gemini grounded calls (first-pass + gap-fill) ≈ 52s;
# 180s leaves ~3x headroom. Previously 600s, which was enough to sit behind a
# stuck upstream for the full worst-case batch budget.
GEMINI_SEARCH_TIMEOUT: int = 180

# --- Second-pass gap-fill ---
# After first-pass research completes, a cheap analyzer identifies up to
# GAP_FILL_MAX_GAPS factual gaps; each is resolved by a parallel grounded
# Gemini search. Fails soft — forecast proceeds with first-pass research alone
# if any stage errors out.
GAP_FILL_ENABLED_ENV: str = "GAP_FILL_ENABLED"
GAP_FILL_ANALYZER_MODEL: str = "gemini-3-flash-preview"
GAP_FILL_MAX_GAPS: int = 5
# Analyzer call is non-grounded (no Google Search) and should return quickly.
# Use a tight timeout to prevent a single hung analyzer request from holding a
# research concurrency slot for the full grounded-search budget.
GAP_FILL_ANALYZER_TIMEOUT: int = 120
# Skip gap-fill when the first-pass research blob has less than this many
# non-whitespace characters — likely indicates all providers soft-failed and
# gap-fill would just hallucinate gaps or burn quota.
GAP_FILL_MIN_RESEARCH_CHARS: int = 200

# --- Financial Data Provider ---
FINANCIAL_DATA_ENABLED_ENV: str = "FINANCIAL_DATA_ENABLED"
FRED_API_KEY_ENV: str = "FRED_API_KEY"
FINANCIAL_CLASSIFIER_MODEL: str = "openrouter/openai/gpt-5-mini"
FINANCIAL_CLASSIFIER_TIMEOUT: int = 30
FINANCIAL_YFINANCE_LOOKBACK_DAYS: int = 365
FINANCIAL_YFINANCE_RECENT_DAYS: int = 30
FINANCIAL_FRED_LOOKBACK_YEARS: int = 5

# --- Soft deadlines to keep batch wall-clock inside the tournament cron window ---
# Per-forecaster outer deadline wrapped via asyncio.wait_for around each
# _make_prediction call. A single stuck forecaster used to be able to hold a
# question for timeout(480s) * allowed_tries(3) ≈ 24 min; this caps that
# worst case at 10 min, at which point the forecaster is dropped with a loud
# WARNING and the other models carry the ensemble.
FORECASTER_SOFT_DEADLINE: int = 600

# Minimum number of successful base forecasters required to publish a question.
# Below this, the question is skipped entirely rather than publishing a weak
# ensemble. Chosen conservatively: median/stacker aggregation remains meaningful
# with 3/6 inputs; below that we're closer to a single-model opinion.
MIN_FORECASTERS_TO_PUBLISH: int = 3

# Stacker soft deadline. Set slightly above the stacker LLM's litellm timeout
# (480s) so the model's own timeout fires first with a clean exception when
# possible; this wait_for is a final belt-and-suspenders backstop for a wholly
# stuck call. Stacker is configured with allowed_tries=1 in llm_configs.py so
# we only get one try before falling back.
STACKER_SOFT_DEADLINE: int = 500
# Stacker fallback model soft deadline. Tighter because we're already running
# late on the critical path by the time the fallback fires.
STACKER_FALLBACK_SOFT_DEADLINE: int = 300

# --- Benchmark driver tuning ---
HEARTBEAT_INTERVAL: int = 60
FETCH_RETRY_BACKOFFS: list[int] = [5, 15]
# Distribution mix: (binary, numeric, multiple_choice)
TYPE_MIX: Tuple[float, float, float] = (0.5, 0.25, 0.25)
FETCH_PACING_SECONDS: int = 2

# =============================================================================
# BACKTEST SETTINGS
# =============================================================================
BACKTEST_DEFAULT_RESOLVED_AFTER: str = "2025-12-01"
BACKTEST_DEFAULT_TOURNAMENT: str = "fall-aib-2025"
BACKTEST_DEFAULT_MIN_FORECASTERS: int = 40
BACKTEST_OVERFETCH_RATIO: int = 3
LEAKAGE_DETECTOR_MODEL: str = "openrouter/openai/gpt-5-mini"
