"""
Central configuration constants to avoid magic numbers and strings.

These are intentionally minimal and focused on operational tuning knobs that
need to be shared across modules.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

from metaculus_bot.config import load_environment

# =============================================================================
# TOURNAMENT IDs - UPDATE THESE EACH QUARTER/SEASON
# =============================================================================
# AI Forecasting Benchmark tournament (bot-only competition)
# Update when new season starts: https://www.metaculus.com/project/aib/
TOURNAMENT_ID: str = "summer-futureeval-2026"  # Summer 2026 FutureEval Bot Tournament (project ID: 33022)
TOURNAMENT_END_DATE: str = "2026-09-06"  # Formal tournament close date
TOURNAMENT_HARD_STOP_WEEKS: int = 2  # ~2 weeks of wiggle room past close before erroring

# Metaculus Cup tournament (human + bot competition)
# Update when new cup starts: https://www.metaculus.com/tournament/metaculus-cup/
METACULUS_CUP_ID: str = "metaculus-cup"  # Uses slug, auto-resolves to current cup


def gemini_use_donated_openrouter_key() -> bool:
    """Whether OpenRouter Gemini calls should route through the Metaculus-donated key.

    Default is now True: after Metaculus raised the Google rate limits
    (2026-06-16), the donated OpenRouter key (``OAI_ANTH_OPENROUTER_KEY``) serves
    most Gemini models — e.g. ``gemini-3.5-flash`` and ``gemini-3.1-flash-lite``
    both succeed on the donated key (verified by live call this session). Set the
    env var to a false-y value (``"false"`` / ``"0"`` / ``"no"``) to force
    personal-key-only routing.

    KNOWN EXCEPTION: ``gemini-3.1-pro-preview`` (our forecaster slot) is PINNED to
    the personal key — not merely "falls back". It's on the
    ``DONATED_KEY_BLOCKED_GOOGLE_MODELS`` blocklist in ``fallback_openrouter``, so
    ``should_route_via_donated_key`` returns False for it even when this toggle is
    True: no donated attempt, no 429, no personal-key-fallback-counter bump (which
    would otherwise redden CI on every question). That model 429s on the donated
    key because it routes through a free-tier Google AI Studio BYOK key with no
    Pro free tier (quota 0). Temporary workaround pending the Metaculus-side BYOK
    fix; see the ``TODO(gemini-3.1-pro-donated)`` tag on that constant.

    Read at call time (not import) so workflow env changes take effect without
    re-importing.

    Scope: this toggle only affects OpenRouter routing (``fallback_openrouter``).
    The google-genai grounded-search provider has no donated path — it always
    reads the operator's personal GOOGLE_API_KEY.
    """
    return env_flag_enabled(GEMINI_USE_DONATED_OPENROUTER_KEY_ENV, default=True)


class TournamentExpiredError(Exception):
    """Raised when the tournament has ended and the ID needs to be updated."""

    pass


def check_tournament_dates(logger: logging.Logger | None = None) -> None:
    """Check if tournament dates are stale and warn/error accordingly.

    - Warns if current date is past TOURNAMENT_END_DATE
    - Raises TournamentExpiredError if past end date + TOURNAMENT_HARD_STOP_WEEKS

    Call this at bot startup to catch stale tournament IDs.
    """
    log = logger or logging.getLogger(__name__)

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
DEFAULT_MAX_CONCURRENT_RESEARCH: int = 6

# Benchmark driver settings
# Default batch size for benchmarking runs
# Keep this modest to balance concurrency and rate limits.
BENCHMARK_BATCH_SIZE: int = 4

# Metaculus comment safety limits. The published comment has three top-level
# sections (# SUMMARY / # RESEARCH / # FORECASTS), each trimmed to its own
# budget before assembly (see comment.trimming.trim_section). FORECASTS
# (per-model rationales + fenced JSON forecast blocks) gets the largest share
# because it carries the per-model attribution the residual pipeline parses;
# RESEARCH is a lossy fallback-archive re-print; SUMMARY holds the
# parser-critical bullets and is sized well above any realistic bullet block so
# it never clips them. The three caps sum below COMMENT_CHAR_LIMIT; trim_comment
# shrinks RESEARCH first (never bullets or rationales) if the assembled comment
# plus framework overhead still overflows.
FORECASTS_SECTION_CHAR_LIMIT: int = 89_999
RESEARCH_SECTION_CHAR_LIMIT: int = 44_999
SUMMARY_SECTION_CHAR_LIMIT: int = 13_999
COMMENT_CHAR_LIMIT: int = 149_999

# Optional environment variable to force research provider selection.
# Accepted values (case-insensitive): "auto", "asknews", "exa", "perplexity", "openrouter"
RESEARCH_PROVIDER_ENV: str = "RESEARCH_PROVIDER"

# Credential env-var names. Named constants (matching the existing *_ENV
# convention used for GOOGLE_API_KEY_ENV / FRED_API_KEY_ENV) so the literal
# strings aren't duplicated across api_key_utils / fallback_openrouter /
# research_providers / research_orchestrator — that duplication is exactly the
# typo risk the convention exists to prevent. See CLAUDE.md "API keys & secrets"
# for which of these are shared (donated) vs. personal.
OPENROUTER_API_KEY_ENV: str = "OPENROUTER_API_KEY"
OAI_ANTH_OPENROUTER_KEY_ENV: str = "OAI_ANTH_OPENROUTER_KEY"
ASKNEWS_CLIENT_ID_ENV: str = "ASKNEWS_CLIENT_ID"
ASKNEWS_SECRET_ENV: str = "ASKNEWS_SECRET"
EXA_API_KEY_ENV: str = "EXA_API_KEY"
PERPLEXITY_API_KEY_ENV: str = "PERPLEXITY_API_KEY"
METACULUS_TOKEN_ENV: str = "METACULUS_TOKEN"


def env_flag_enabled(env_name: str, *, default: bool = False) -> bool:
    """Return True iff env var is set to "true"/"1"/"yes" (case-insensitive).

    When the env var is unset (or empty string), returns ``default``.
    Explicit "false"/"0"/"no" always returns False, regardless of default.
    """
    raw = os.getenv(env_name, "").lower()
    if raw == "":
        return default
    if raw in ("true", "1", "yes"):
        return True
    if raw in ("false", "0", "no"):
        return False
    return default


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

# --- OpenRouter credit telemetry ---
# End-of-run floor for the DONATED key's remaining balance (limit_remaining).
# Below this, cli.main logs a loud warning and exits non-zero AFTER all
# forecasting/publishing completes — a reminder-to-refill signal, not an abort.
# The floor is meaningless for the personal key (no limit_remaining), so it is
# only checked against the donated key. See metaculus_bot/credit_telemetry.py.
OPENROUTER_CREDIT_FLOOR_USD: float = _float_env("OPENROUTER_CREDIT_FLOOR_USD", 50.0)

# --- Forecasting clamps and numeric smoothing ---
# Binary prediction clamp. Mirrors Preseen-Atlas's clip-only tail protection
# (Atlas publishes `0.96 * estimate + 0.02`; we adopt the clip portion only).
# See scratch_docs_and_planning/atlas_inspired_improvements.md Workstream B.
BINARY_PROB_MIN: float = 0.02
BINARY_PROB_MAX: float = 0.98

# Multiple-choice prediction clamp
MC_PROB_MIN: float = 0.005
MC_PROB_MAX: float = 0.995

# --- Post-hoc Platt calibration of the final published probability ---
# Final-output logistic recalibration following Metaculus's notebook
# "Improving Forecaster Performance via Automated Calibration Adjustment"
# (2026-05-01). Fitted parameters live in metaculus_bot/calibration/params.py
# and are hand-edited after running the fit_platt_cli.
#
# Both deviation caps are HARD absolute caps applied AFTER the smooth
# logistic transform. They cap how far the calibration is allowed to move
# any single probability from the raw aggregation output. The user's stance:
# "tweak, don't massively deviate" — the underlying fit can want a large
# move; the cap prevents us from acting on it. Tune by hand after seeing
# the unconstrained fit.
PLATT_CALIBRATION_ENABLED_ENV: str = "PLATT_CALIBRATION_ENABLED"
PLATT_BINARY_MAX_ABS_DEVIATION: float = 0.10
# MC cap is tighter because the per-option Platt is applied N times per
# question and small per-option drift compounds after renormalization.
PLATT_MC_MAX_ABS_DEVIATION: float = 0.05

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
# Default model for native search (without openrouter/ prefix).
# Critical-path research — this constant covers BOTH the always-on native-search
# provider (every question) and the targeted search on the stacking path.
# Effort stays at the env default LOW (see NATIVE_SEARCH_REASONING_EFFORT_DEFAULT
# below).
# 2026-07-17: sol→terra per blind research-role audit
# (scratch/research_role_audit_2026-07-17/ — terra won native-search role 1st;
# sol 2nd; luna 3rd; verdict "MARGINAL EDGE").
NATIVE_SEARCH_DEFAULT_MODEL: str = "openai/gpt-5.6-terra"
# No temperature / top_p: reasoning models defer to provider defaults; the LLM
# is built with temperature=None so litellm omits the param (see build_native_search_llm).
NATIVE_SEARCH_MAX_TOKENS: int = 16_000
NATIVE_SEARCH_TIMEOUT: int = (
    360  # 2026-05-17: raised 240→360 alongside gpt-5.5 medium-effort migration; see comparison_v3.md
)
# Wall-clock backstop for the native-search provider. NATIVE_SEARCH_TIMEOUT
# above is the litellm per-HTTP-request timeout; it resets across retries
# (allowed_tries default 3 ⇒ worst case ~18 min) and was observed defeated
# entirely on 2026-05-20 by an OpenRouter response that dripped ~700 lines of
# whitespace keep-alive bytes over 8m37s before closing with malformed JSON.
# asyncio.wait_for around llm.invoke gives us a hard wall-clock cap regardless
# of what the underlying HTTP layer does. Slight headroom over the request
# timeout so the cleaner per-request error fires first when possible.
NATIVE_SEARCH_WALL_TIMEOUT: int = 420
# Reasoning effort and verbosity for the OpenAI native-search call.
# Override via env vars NATIVE_SEARCH_REASONING_EFFORT / NATIVE_SEARCH_VERBOSITY.
# Empty string disables passing the kwarg.
NATIVE_SEARCH_REASONING_EFFORT_ENV: str = "NATIVE_SEARCH_REASONING_EFFORT"
# 2026-05-20: dropped medium→low after the OpenRouter whitespace-stream incident
# that consumed 8m37s on a single call. v3 bench (comparison_v3.md) measured
# effort=low at ~50s vs effort=medium at ~230s, so low gives ~4.5× faster
# wall-clock and far more headroom under NATIVE_SEARCH_WALL_TIMEOUT (420s)
# / NATIVE_SEARCH_TIMEOUT (360s). The quality cost of low is now absorbed by
# the tier upgrade to gpt-5.6-sol (smarter model at lower effort). Override via
# NATIVE_SEARCH_REASONING_EFFORT env if a workflow needs medium back. Note:
# this default applies ONLY to the native-search provider —
# DISAGREEMENT_ANALYZER_LLM is also at low (llm_configs.py), all forecasters
# stay at high.
NATIVE_SEARCH_REASONING_EFFORT_DEFAULT: str = "low"
NATIVE_SEARCH_VERBOSITY_ENV: str = "NATIVE_SEARCH_VERBOSITY"
NATIVE_SEARCH_VERBOSITY_DEFAULT: str = "low"
# Native search web options (passed to OpenRouter plugins)
NATIVE_SEARCH_MAX_RESULTS: int = 20
NATIVE_SEARCH_CONTEXT_SIZE: str = "high"  # "low", "medium", "high"

# --- Resolution-Source Fetcher (Tier 1) ---
# Char caps below apply to RAW fetched content only (policy: raw passthrough is
# capped; LLM-emitted research is never truncated).
RESOLUTION_SOURCE_ENABLED_ENV: str = "RESOLUTION_SOURCE_ENABLED"
RESOLUTION_SOURCE_HTTP_TIMEOUT: float = 20.0  # per-request (probe: 0-2s typical; slack for slow gov sites)
RESOLUTION_SOURCE_WALL_TIMEOUT: float = 45.0  # hard cap on the whole provider
RESOLUTION_SOURCE_MAX_URLS: int = 5  # 58 URLs / 40 Qs ≈ 1.45 avg; bounds pathological multi-URL Qs
RESOLUTION_SOURCE_MAX_RESPONSE_BYTES: int = 5 * 1024 * 1024  # CISA KEV JSON ~1.5 MB; 5 MiB headroom
RESOLUTION_SOURCE_PER_URL_MAX_CHARS: int = 6000  # elbow of full-extraction dist (p50=2.2k, p75=5.2k); truncation 48%->21% on 2026-07-09 smoke; ~1.5k tokens/URL
RESOLUTION_SOURCE_TOTAL_MAX_CHARS: int = (
    18000  # headroom so per-URL cap binds (max observed section ~11.1k at 6k/URL); ~4.5k tokens worst case
)
RESOLUTION_SOURCE_JS_WALL_MIN_CHARS: int = 100  # 200-OK with < this extracted text == JS wall (FINDINGS)
RESOLUTION_SOURCE_GLOBAL_CONCURRENCY: int = 5  # TCPConnector limit; per-host serialized separately

# --- Gemini Search Provider (Google AI Studio direct SDK) ---
# Uses google-genai SDK with GoogleSearch grounding tool for first-party Google
# Search results (distinct from OpenRouter's Exa-backed :online plugin). Adds a
# genuinely new search index to the ensemble.
GEMINI_SEARCH_ENABLED_ENV: str = "GEMINI_SEARCH_ENABLED"
GEMINI_SEARCH_MODEL_ENV: str = "GEMINI_SEARCH_MODEL"
# GOOGLE_API_KEY is the operator's personal Google AI Studio key (in CI it's
# stored as ``secrets.GEMINI_API_KEY`` and surfaced as GOOGLE_API_KEY for the
# google-genai SDK). The grounded-search provider always reads this — it has
# no donated/shared-key path because Google AI Studio doesn't offer one.
GOOGLE_API_KEY_ENV: str = "GOOGLE_API_KEY"
# Toggle for OpenRouter Gemini routing only. Controls whether models like
# ``openrouter/google/gemini-3.1-pro-preview`` flow through the Metaculus-
# donated OpenRouter key (``OAI_ANTH_OPENROUTER_KEY``) with paid-key fallback,
# or skip the donated wrapper entirely and route through the operator's
# personal ``OPENROUTER_API_KEY``. Does NOT affect the google-genai grounded
# search provider — that always uses the personal GOOGLE_API_KEY. Default ON
# (2026-06-16): after Metaculus raised the Google rate limits, the donated key
# serves most Gemini models (gemini-3.5-flash, gemini-3.1-flash-lite). The known
# exception is gemini-3.1-pro-preview, which is PINNED to the personal key via the
# DONATED_KEY_BLOCKED_GOOGLE_MODELS blocklist (no donated attempt, no 429) pending
# the Metaculus-side BYOK fix — see TODO(gemini-3.1-pro-donated) in fallback_openrouter.
GEMINI_USE_DONATED_OPENROUTER_KEY_ENV: str = "GEMINI_USE_DONATED_OPENROUTER_KEY"
# Gemini 3 Flash preview model with grounding support. Requires billing enabled
# on the Google AI Studio project to unlock; falls back to gemini-2.5-flash on
# free tier if needed. Override via GEMINI_SEARCH_MODEL env var.
GEMINI_SEARCH_DEFAULT_MODEL: str = "gemini-3-flash-preview"
# No temperature / top_p / max_tokens overrides — use google-genai SDK defaults.
# Gemini 3 Flash is a thinking model; Google's defaults are tuned for it and
# capping either caused silent truncations in the past.
# 6 min. AFC (Automatic Function Calling) can chain up to 10 tool round-trips
# internally (search → model → URL fetch → model → ...), each ~15-20s. A full
# 10-round chain takes 150-200s, so 180s was too tight — observed timeouts on
# legitimate deep-research calls. 360s gives 2x headroom over worst-case AFC.
# Gap-fill runs overlapped with forecaster LLM calls, so higher timeout adds
# zero wall-clock cost. Observed p99 of non-AFC calls ≈ 52s.
GEMINI_SEARCH_TIMEOUT: int = 360

# --- Second-pass gap-fill ---
# After first-pass research completes, a cheap analyzer identifies up to
# GAP_FILL_MAX_GAPS factual gaps; each is resolved by a parallel OpenAI native
# web search (see GAP_FILL_RESOLVER_MODEL below). Fails soft — forecast proceeds
# with first-pass research alone if any stage errors out.
GAP_FILL_ENABLED_ENV: str = "GAP_FILL_ENABLED"
# Non-grounded gap-listing: reads the first-pass research and emits a JSON list
# of up to GAP_FILL_MAX_GAPS factual gaps, under a tight 135s wall cap that
# soft-fails silently on breach — terra-low is the latency-safe choice; the
# task is decomposition, not deep judgment. Grounded search resolution still
# uses google-genai directly via gemini_search_provider — that path needs the
# search index.
GAP_FILL_ANALYZER_MODEL: str = "openrouter/openai/gpt-5.6-terra"
GAP_FILL_MAX_GAPS: int = 5
# Analyzer call is non-grounded (no Google Search) and should return quickly.
# Use a tight timeout to prevent a single hung analyzer request from holding a
# research concurrency slot for the full grounded-search budget.
GAP_FILL_ANALYZER_TIMEOUT: int = 120
# Wall-clock backstop for the analyzer call. Slight headroom over
# GAP_FILL_ANALYZER_TIMEOUT so the cleaner per-request error from litellm fires
# first when possible (auth failure, model-not-found, etc.) — same pattern as
# NATIVE_SEARCH_WALL_TIMEOUT vs NATIVE_SEARCH_TIMEOUT (60s headroom). Without
# this, asyncio.wait_for and the litellm request timeout fire at the exact
# same second and we lose the descriptive error message.
GAP_FILL_ANALYZER_WALL_TIMEOUT: int = 135
# Skip gap-fill when the first-pass research blob has less than this many
# non-whitespace characters — likely indicates all providers soft-failed and
# gap-fill would just hallucinate gaps or burn quota.
GAP_FILL_MIN_RESEARCH_CHARS: int = 200
# 2026-06-25: migrated the per-gap RESOLVER off direct-Google grounded Gemini
# (google-genai, personal GOOGLE_API_KEY) to OpenAI native web search via
# OpenRouter, which bills the Metaculus-donated key. The resolver fanned out up
# to GAP_FILL_MAX_GAPS parallel grounded calls per question — a 5x cost
# multiplier on the personal Google bill, the dominant unwanted spend. The
# single first-pass grounded Gemini call stays on google-genai (operator is fine
# paying for 1 call/question, and it uses url_context which OpenRouter can't
# replicate). No "openrouter/" prefix here — build_native_search_llm adds it.
#
# Agentic single-gap web research whose source-trust judgment lands directly in
# every forecaster prompt. Small per-gap outputs mute sol's latency premium, and
# the 5 workers run in parallel under a 420s cap (latency = slowest call, not
# sum), so effort stays LOW. 2026-07-09 bench: sol-low matched terra-low coverage
# 24/25 in 20% fewer chars and uniquely caught a research-internal error.
GAP_FILL_RESOLVER_MODEL: str = "openai/gpt-5.6-sol"
GAP_FILL_RESOLVER_REASONING_EFFORT: str = "low"

# --- Agentic gap-fill v2 (bounded research loop) ---
# Second-generation gap-fill: a bounded agentic loop (metaculus_bot/research/
# agentic/) that dry-runs the panel's own forecasting template to identify
# fill/verify/resolution targets, then pursues them with search/fetch/read
# tools. Runs CONCURRENTLY with v1 during the overlap window (both flags on);
# soft-fails to "" like v1. See scratch_docs_and_planning/
# agentic_gap_fill_v2_plan.md.
GAP_FILL_V2_ENABLED_ENV: str = "GAP_FILL_V2_ENABLED"
# Driver model + effort picked by the blind 5-arm replay eval 2026-07-17
# (scratch/driver_replay_2026-07-17/blind_judge_report.md): terra-low ranked 1st
# (fetch-verified grounding, best source mix, 30s wall, $0.36/q), terra-medium
# 2nd; sol-low burned budget on near-duplicate searches (5th); sonnet-5 cited
# unfetched URLs (disqualifying for a researcher). All candidates were
# openai/anthropic, so the loop's litellm binding routes via the donated
# OpenRouter key.
GAP_FILL_V2_DRIVER_MODEL: str = os.getenv("GAP_FILL_V2_DRIVER_MODEL") or "openai/gpt-5.6-terra"
GAP_FILL_V2_DRIVER_EFFORT: str = os.getenv("GAP_FILL_V2_DRIVER_EFFORT") or "low"
# read_document backend model on the NATIVE google-genai path (tools.py
# _run_document_read_sync). CAUTION: this id is UNVERIFIED on the native
# AI Studio API until the paid smoke test — the repo's verified-model notes
# ("gemini-3.5-flash works") all refer to the OpenRouter slug route, which maps
# ids differently; the only id verified on the native SDK here is
# GEMINI_SEARCH_DEFAULT_MODEL ("gemini-3-flash-preview"). A wrong id soft-fails
# read_document (model-not-found -> error outcome), silently disabling the
# directed-reading rung.
GAP_FILL_V2_READER_MODEL: str = os.getenv("GAP_FILL_V2_READER_MODEL") or "gemini-3.5-flash"
# Parallel tool calls each count against the cap; steps are where latency
# lives, so batching is encouraged rather than rationed.
GAP_FILL_V2_MAX_TOOL_CALLS: int = _int_env("GAP_FILL_V2_MAX_TOOL_CALLS", 20)
# Hard wall for the whole loop — inside v1's worst-case envelope (analyzer
# 135s + resolver wave 420s ≈ 555s), so running v2 concurrently with v1 adds
# no research-phase wall-clock. The loop is anytime: hitting the deadline
# emits banked findings, never "".
GAP_FILL_V2_WALL_DEADLINE: float = _float_env("GAP_FILL_V2_WALL_DEADLINE", 540.0)
# With less than this many seconds remaining, the harness rejects every tool
# except conclude, forcing the loop to wrap up inside the wall deadline.
GAP_FILL_V2_CONCLUDE_THRESHOLD: float = _float_env("GAP_FILL_V2_CONCLUDE_THRESHOLD", 90.0)
# Below this many extracted chars, the fetch ladder escalates plain HTTP to
# headless-Chromium rendering (JS-wall heuristic; tools.py consumes this).
GAP_FILL_V2_MIN_CONTENT_CHARS: int = _int_env("GAP_FILL_V2_MIN_CONTENT_CHARS", 500)

# --- Financial Data Provider ---
FINANCIAL_DATA_ENABLED_ENV: str = "FINANCIAL_DATA_ENABLED"
FRED_API_KEY_ENV: str = "FRED_API_KEY"
# Binary-ish routing classification (is this a financial/economic question?)
# under a 30s timeout — capability-saturated, so mini stays the cheapest capable tier.
FINANCIAL_CLASSIFIER_MODEL: str = "openrouter/openai/gpt-5.4-mini"
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

# Per-question wall-clock cutoff (58:30 of the 60-min Metaculus close window).
# At deadline, in-flight forecasters are cancelled; we base-combine whatever
# completed (>=MIN_FORECASTERS_TO_PUBLISH) and submit. Remainder budget reserves
# time for stacker-skip + publish (with 20s POST timeouts + 1 retry).
PER_QUESTION_WALL_CLOCK_DEADLINE: int = 3510

# Below this remaining-budget threshold, skip stacking and force fallback_median
# aggregation. Reserves enough time for publish hardening (20s POST timeout + 1
# retry across two POSTs = 80s worst case) plus headroom.
WALL_CLOCK_STACKING_MIN_BUDGET: int = 90

# Per-publish-POST timeout (post_binary/numeric/mc + post_question_comment).
# Stock forecasting-tools uses synchronous `requests.post` with no timeout, so
# a hung server can block the whole batch indefinitely. publish_hardening.py
# wraps each POST on a concurrent.futures.ThreadPoolExecutor with a
# Future.result(timeout=...) cap *and* monkey-patches `requests.post` on the
# forecasting-tools module to inject a request-side socket timeout (so the
# underlying socket actually closes when the server stalls). Retry once on
# timeout / connection error.
PUBLISH_POST_TIMEOUT: int = 20
PUBLISH_POST_RETRIES: int = 1

# Fetch hardening: retry/timeout for question-list GETs to the Metaculus API.
# Stock forecasting-tools issues `requests.get` with no timeout and no retry,
# so a single transient 403/429/5xx anywhere in the question pagination kills
# the whole CI run. Observed 2026-05-19: a CDN/WAF-style 403 (33s stall +
# generic "API only available to authenticated users" body) on a healthy key.
# fetch_hardening.py wraps `_get_questions_from_api` (the single chokepoint
# for every question-list GET) with a request-side socket timeout + bounded
# retry on retryable statuses and connection-level errors.
# Backoff sized for the realistic failure mode: a CDN/WAF edge-node overload
# typically clears in 10-60s, not 1-3s. The observed 2026-05-19 incident had
# a 33s server-side stall before the 403; backoff in the 10-25s range gives
# the edge layer time to recover. Cost of waiting is ~zero (tournament fetch
# is on a 20-minute cron and ~40min total budget); cost of retrying too soon
# is hitting the same wall and burning the run.
FETCH_GET_TIMEOUT: int = 60
FETCH_GET_RETRIES: int = 2
FETCH_GET_BACKOFF_BASE: float = 10.0
FETCH_GET_BACKOFF_JITTER: float = 3.0

# Stacker soft deadline. Set slightly above the stacker LLM's litellm timeout
# (480s) so the model's own timeout fires first with a clean exception when
# possible; this wait_for is a final belt-and-suspenders backstop for a wholly
# stuck call. Stacker is configured with allowed_tries=1 in llm_configs.py so
# we only get one try before falling back.
STACKER_SOFT_DEADLINE: int = 500
# Stacker fallback model soft deadline. Tighter because we're already running
# late on the critical path by the time the fallback fires.
STACKER_FALLBACK_SOFT_DEADLINE: int = 300

# Per-question soft deadline for the disagreement-crux extractor (gpt-5.6-sol low effort).
# Caps the unbounded worst case on the conditional-stacking critical path: without
# this wrapper the analyzer can stall for timeout(300s) * allowed_tries(3) ≈ 15 min.
CRUX_SOFT_DEADLINE: int = 180

# Wall-clock cap for the AskNews summarizer invoke. The summarizer is set
# allowed_tries=1 (llm_configs.py) and wrapped in the broad 30s-gated retry, which
# previously had no wall guard at all. Matches the summarizer's litellm per-request
# timeout (UTILITY_MODEL_CONFIG timeout=300) so the per-attempt cap aligns
# with the underlying request budget; on breach the summarizer soft-fails to the
# raw AskNews articles rather than hanging the question.
SUMMARIZER_WALL_TIMEOUT: int = 300

# --- Benchmark driver tuning ---
HEARTBEAT_INTERVAL: int = 60
FETCH_RETRY_BACKOFFS: list[int] = [5, 15]
# Distribution mix: (binary, numeric, multiple_choice)
TYPE_MIX: tuple[float, float, float] = (0.5, 0.25, 0.25)
FETCH_PACING_SECONDS: int = 2

# =============================================================================
# BACKTEST SETTINGS
# =============================================================================
BACKTEST_DEFAULT_RESOLVED_AFTER: str = "2025-12-01"
BACKTEST_DEFAULT_TOURNAMENT: str = "fall-aib-2025"
BACKTEST_DEFAULT_MIN_FORECASTERS: int = 40
BACKTEST_OVERFETCH_RATIO: int = 3
# Mechanical leakage screen over research text, backtest-only — saturated task,
# mini is the cheapest capable tier.
LEAKAGE_DETECTOR_MODEL: str = "openrouter/openai/gpt-5.4-mini"

# --- Per-type stacking gates ---
# Each question type has an independent enable/disable flag. All three default
# to DISABLED (see the gate in main.py, which reads these via
# env_flag_enabled(..., default=False)). A deploy opts a type back into stacking
# by setting <TYPE>_STACKING_ENABLED=true in its env.
#
# Background: ablation showed the stacker hurts numeric CRPS (median > stack,
# p=0.042); numeric disable is evidence-backed. Binary was a TIE (p=0.496), so
# binary + MC are off as a low-risk default (tie-at-best + compute), UNMEASURED on
# the current stack. TODO: revisit after prod-ish ablation / marker-era resolutions
# (see scratch_docs_and_planning/prod_ish_ablation_plan.md).
BINARY_STACKING_ENABLED_ENV: str = "BINARY_STACKING_ENABLED"
MC_STACKING_ENABLED_ENV: str = "MC_STACKING_ENABLED"
NUMERIC_STACKING_ENABLED_ENV: str = "NUMERIC_STACKING_ENABLED"

# --- Prediction-market provider (Workstream G) ---
# Env-gated. Resolved markets on all three platforms retain their last-trade
# price after resolution — without the ``as_of`` filter in
# ``fetch_market_snapshot``, pulling a market for a resolved Metaculus question
# leaks post-resolution pricing into the rationale, which is why the provider is
# hard-disabled under ``is_benchmarking=True``. ON in all prod workflows as of
# commit 3c12dbe (prod runs with is_benchmarking=False, so the guard doesn't
# suppress it there). The benchmarking guard means the standard ``make
# backtest_*`` gate can't measure its forecasting value — it was validated via
# the manual ``test_bot.yaml`` prod-mode run + opt-in live integration tests
# instead. See atlas_inspired_improvements.md §G.
PREDICTION_MARKETS_ENABLED_ENV: str = "PREDICTION_MARKETS_ENABLED"

# Outer wall-clock timeout for the full prediction-market snapshot (keyword
# extraction + HTTP fan-out to all platforms). Runs inside asyncio.gather
# alongside other research providers, so increasing this does not add
# wall-clock time to the overall research phase.
PREDICTION_MARKET_TIMEOUT: float = float(os.environ.get("PREDICTION_MARKET_TIMEOUT", "30.0"))

# Keyword-extraction strategy for matching Metaculus questions to market
# listings. Default ``s4_s5_union`` is the empirical best on a 15-question
# G0 study (67% hit rate vs 33% naive baseline; see
# scratch_docs_and_planning/prediction_market_keyword_extraction_experiment.md).
# ``s5_only`` is cheaper at 60%; ``simple`` is the cost-floor baseline.
PREDICTION_MARKET_KEYWORD_STRATEGY_ENV: str = "PREDICTION_MARKET_KEYWORD_STRATEGY"
PREDICTION_MARKET_KEYWORD_STRATEGY_VALID: frozenset[str] = frozenset({"s4_s5_union", "s5_only", "simple"})

# --- Time-Series Anchor Provider (Phase B) ---
# Env-gated OFF by default. Renders a deterministic empirical-band anchor grounded
# in the resolution series' OWN history (FRED/yfinance), for numeric questions that
# route cleanly to a known series (via resolution-criteria URLs or a curated title
# registry). No statsforecast / model selection — the Phase-A offline replay
# (scratch/ts_anchor_replay_2026-07-16/synthesis.md) found CV-gated model picks beat
# naive out-of-sample only 43% of the time; the empirical h-step-change band is the
# render. Its value is grounding + SHARPENING our over-wide published low tail
# (cov@10 was 0.02 vs a 0.10 target). Backtest-safe (the FIRST research provider that
# is): live uses as_of=now, is_benchmarking uses question.open_time so series data up
# to resolution IS the answer (NOT scheduled_resolution − buffer), with ALFRED
# vintages at as_of for revising series.
TS_ANCHOR_ENABLED_ENV: str = "TS_ANCHOR_ENABLED"
# Chart-image side-channel: when on (and TS_ANCHOR_ENABLED is also on), the
# provider renders an 800x400 PNG of the anchor (series + P10-P90 band) for
# single-level questions and stashes it per-qid; the forecaster passes it to
# each base model as a vision message. OFF everywhere until the text-vs-image
# A/B (FUTURE.md "TS anchor chart image"). Independent of TS_ANCHOR_ENABLED so
# the text anchor can ship before the (costlier, unvalidated) image does.
TS_ANCHOR_CHART_ENABLED_ENV: str = "TS_ANCHOR_CHART_ENABLED"
# Wall-clock cap on the whole provider (fetch fan-out + render). Fetches run in
# asyncio.to_thread under asyncio.wait_for; a hung endpoint soft-fails to "".
TS_ANCHOR_TIMEOUT: float = float(os.environ.get("TS_ANCHOR_TIMEOUT", "20.0"))
# Per-request HTTP timeout for a single FRED/ALFRED/yfinance fetch.
TS_ANCHOR_HTTP_TIMEOUT: float = 15.0
# History lookback for both the displayed tables and the empirical change/window-max
# distributions. Spread legs use a shorter window (below) to exclude the 2020-04-20
# negative WTI settlement that breaks the strictly-positive log-return construction.
TS_ANCHOR_LOOKBACK_YEARS: int = 15
TS_ANCHOR_SPREAD_LOOKBACK_YEARS: int = 5
# Char budget for the whole rendered section (self-budgeted like resolution_source;
# per-leg render truncates history tables so multi-leg spreads stay bounded).
TS_ANCHOR_SECTION_MAX_CHARS: int = 6000
# History-table lengths per resolution: last-N native-freq observations, weekly
# down-sampled closes (~3 months of trading weeks), monthly down-sampled (~2 years).
TS_ANCHOR_NATIVE_TABLE_ROWS: int = 10
TS_ANCHOR_WEEKLY_TABLE_ROWS: int = 13
TS_ANCHOR_MONTHLY_TABLE_ROWS: int = 24

# --- Research persistence (write path for backtest replay) ---
PERSIST_RESEARCH_ENABLED_ENV: str = "PERSIST_RESEARCH_ENABLED"

# --- Raw research-provider payload logging (durable GHA-artifact tape) ---
# Independent from PERSIST_RESEARCH (which archives the post-summarizer research
# text keyed per question). This captures each provider's RAW return — AskNews
# article dicts per phase, native/gemini raw responses + grounding, prediction-
# market contracts, resolution-source per-URL fetches, gap-fill search results —
# appended as JSONL to a run_logs/ file so the raw evidence behind every forecast
# survives the 90-day artifact window without depending on published comments.
# OFF by default in code (unset env) so tests/local runs never write; the four
# workflow yamls set it ON.
RAW_RESEARCH_LOG_ENABLED_ENV: str = "RAW_RESEARCH_LOG_ENABLED"
# Directory the raw-research JSONL is appended to. Defaults to run_logs/, which
# every workflow tees stdout to and uploads wholesale as an artifact — so the raw
# log rides along with no upload-glob change. Overridable (tests point it at a
# tmp dir) via the RAW_RESEARCH_LOG_DIR env var.
RAW_RESEARCH_LOG_DIR_ENV: str = "RAW_RESEARCH_LOG_DIR"
RAW_RESEARCH_LOG_DIR_DEFAULT: str = "run_logs"
# Per-record serialized-payload cap. A raw AskNews dual-phase pull or a grounded
# Gemini response can be large; beyond this many chars the payload is replaced
# with a bounded truncation marker (preview + original length) so one giant pull
# can't blow up the log file. GHA zips the artifact on upload, so on-disk size is
# the only concern; 200 KB/record is generous headroom for real payloads.
RAW_RESEARCH_MAX_PAYLOAD_CHARS: int = 200_000
