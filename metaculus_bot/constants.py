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

    Default is False (env unset → personal key only; only an explicit ``"true"``
    opts in). The donated OpenRouter key CANNOT serve Gemini: the donated account
    has a free-tier Google AI Studio BYOK key attached, and OpenRouter forces all
    ``google/*`` traffic through that BYOK key first (not overridable per-request).
    ``gemini-3.x-pro`` has no Google free tier, so the BYOK quota is structurally
    0 → every donated-key Gemini call 429s (``is_byok:true`` + ``FreeTier limit:
    0``) and falls back to the personal key anyway. So Gemini routes through the
    operator's personal OPENROUTER_API_KEY directly. This is a Metaculus-account-
    side fix (enable Cloud billing on the BYOK key's GCP project, or remove the
    Google BYOK integration so native OpenRouter Google credits are used); flip
    this default back to True only after that's done.

    Read at call time (not import) so workflow env changes take effect without
    re-importing.

    Scope: this toggle only affects OpenRouter routing (``fallback_openrouter``).
    The google-genai grounded-search provider has no donated path — it always
    reads the operator's personal GOOGLE_API_KEY.
    """
    return os.getenv(GEMINI_USE_DONATED_OPENROUTER_KEY_ENV, "false").strip().lower() == "true"


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
DEFAULT_MAX_CONCURRENT_RESEARCH: int = 6

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
# Default model for native search (without openrouter/ prefix)
# 2026-05-17: migrated from x-ai/grok-4.1-fast (deprecated 2026-05-15 by xAI).
# Initially flipped to gpt-5.4-mini, but v3 bench (scratch/native_search_bench_2026-05-17/comparison_v3.md)
# showed gpt-5.5 with reasoning=medium + verbosity=low fits in ~230s under a
# 360s cap with 130s headroom, and consistently produces materially deeper
# research (Opus rubric: 23/25 vs mini's 15-16/25 across two unrelated
# questions). Donated key currently blocked by data-policy guardrail —
# see FUTURE.md "Resolve OAI_ANTH_OPENROUTER_KEY data-policy block" — calls
# bill to personal OPENROUTER_API_KEY for now (~$0.15/call × 250 Qs =
# ~$40/tournament).
NATIVE_SEARCH_DEFAULT_MODEL: str = "openai/gpt-5.5"
# LLM parameters for native search (lower temp for factual grounding)
NATIVE_SEARCH_TEMPERATURE: float = 0.3
NATIVE_SEARCH_TOP_P: float = 0.9
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
# gpt-5.5 effort=low at ~50s vs effort=medium at ~230s, so low gives ~4.5×
# faster wall-clock and far more headroom under NATIVE_SEARCH_WALL_TIMEOUT (420s)
# / NATIVE_SEARCH_TIMEOUT (360s). Caveat: v3 only sanity-checked low for latency,
# not graded quality (mini at default scored 15/25, gpt-5.5 medium 23/25, low
# unknown). Override via NATIVE_SEARCH_REASONING_EFFORT env if a workflow needs
# medium quality back. Note: this default applies ONLY to the native-search
# provider — DISAGREEMENT_ANALYZER_LLM stays at medium (llm_configs.py:182),
# all forecasters stay at high.
NATIVE_SEARCH_REASONING_EFFORT_DEFAULT: str = "low"
NATIVE_SEARCH_VERBOSITY_ENV: str = "NATIVE_SEARCH_VERBOSITY"
NATIVE_SEARCH_VERBOSITY_DEFAULT: str = "low"
# Native search web options (passed to OpenRouter plugins)
NATIVE_SEARCH_MAX_RESULTS: int = 20
NATIVE_SEARCH_CONTEXT_SIZE: str = "high"  # "low", "medium", "high"

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
# search provider — that always uses the personal GOOGLE_API_KEY. Default off
# because the donated-key Google route has been flaky.
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
# GAP_FILL_MAX_GAPS factual gaps; each is resolved by a parallel grounded
# Gemini search. Fails soft — forecast proceeds with first-pass research alone
# if any stage errors out.
GAP_FILL_ENABLED_ENV: str = "GAP_FILL_ENABLED"
# 2026-05-20: migrated analyzer from gemini-3-flash-preview (google-genai SDK
# direct) to gpt-5.5 effort=low via OpenRouter. The analyzer is non-grounded —
# it reads the first-pass research and emits a JSON list of factual gaps —
# so it doesn't need Google as a search index, and OpenAI-stack consistency
# matches the rest of the auxiliary tier (native_search, disagreement_analyzer
# also at gpt-5.5 low). Grounded search resolution still uses google-genai
# directly via gemini_search_provider — that path needs the search index.
GAP_FILL_ANALYZER_MODEL: str = "openrouter/openai/gpt-5.5"
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

# Per-question soft deadline for the disagreement-crux extractor (gpt-5.5 medium effort).
# Caps the unbounded worst case on the conditional-stacking critical path: without
# this wrapper the analyzer can stall for timeout(300s) * allowed_tries(3) ≈ 15 min.
CRUX_SOFT_DEADLINE: int = 180

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
LEAKAGE_DETECTOR_MODEL: str = "openrouter/openai/gpt-5-mini"

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
# Env-gated so backtests can opt in explicitly. Resolved markets on all three
# platforms retain their last-trade price after resolution — without the
# ``as_of`` filter in ``fetch_market_snapshot``, pulling a market for a
# resolved Metaculus question leaks post-resolution pricing into the
# rationale. Default OFF until smoke + medium backtest validates match
# quality and leakage defense. Flip ON in production workflows after that
# gate. See atlas_inspired_improvements.md §G.
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

# --- Research persistence (write path for backtest replay) ---
PERSIST_RESEARCH_ENABLED_ENV: str = "PERSIST_RESEARCH_ENABLED"
