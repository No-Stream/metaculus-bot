# HARNESS-SCAN-EXEMPT-monolithic-file-loc  # flat constants registry; one home for every knob is the design, a split scatters lookups
"""
Central configuration constants to avoid magic numbers and strings.

These are intentionally minimal and focused on operational tuning knobs that
need to be shared across modules.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, date, datetime, timedelta

from metaculus_bot.config import load_environment
from metaculus_bot.time_utils import _as_utc

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


def check_tournament_dates(logger: logging.Logger | None = None) -> None:
    """Check if tournament dates are stale and warn/error accordingly.

    - Warns if current date is past TOURNAMENT_END_DATE
    - Raises TournamentExpiredError if past end date + TOURNAMENT_HARD_STOP_WEEKS

    Call this at bot startup to catch stale tournament IDs.
    """
    log = logger or logging.getLogger(__name__)

    # Both operands go through _as_utc so the comparison is tz-aware on the same side of
    # the clock. Only the wall-clock reference moves (local -> UTC): the tournament close
    # date is a Metaculus (UTC) date, and prod runs on UTC GitHub Actions runners, so this
    # shifts nothing in prod and at most a few hours of a staleness warning locally.
    try:
        end_date = _as_utc(datetime.strptime(TOURNAMENT_END_DATE, "%Y-%m-%d"))  # noqa: DTZ007  # stamped UTC by _as_utc
    except ValueError:
        log.warning(f"Invalid TOURNAMENT_END_DATE format: {TOURNAMENT_END_DATE}")
        return

    today = _as_utc(datetime.now(UTC))
    hard_stop_date = end_date + timedelta(weeks=TOURNAMENT_HARD_STOP_WEEKS)

    if today > hard_stop_date:
        raise TournamentExpiredError(
            f"Tournament '{TOURNAMENT_ID}' ended on {TOURNAMENT_END_DATE} and hard stop "
            f"date ({hard_stop_date.date()}) has passed. Please update TOURNAMENT_ID, "
            f"TOURNAMENT_END_DATE, and TOURNAMENT_HARD_STOP_WEEKS in constants.py for the new season."
        )
    if today > end_date:
        days_past = (today - end_date).days
        days_until_error = (hard_stop_date - today).days
        log.warning(
            f"⚠️  Tournament '{TOURNAMENT_ID}' likely ended on {TOURNAMENT_END_DATE} "
            f"({days_past} days ago). Update constants.py for the new season! "
            f"Bot will error out in {days_until_error} days."
        )


# --- Fall Metaculus Cup reminder (deliberate, dated time bomb) ---
# The summer season closes on TOURNAMENT_END_DATE (2026-09-06) and the fall cup is the
# one identified next-season lever. The platform object already exists — slug
# `metaculus-cup-fall-2026`, project id 33108, start_date 2026-08-28T12:00:00Z,
# forecasting_end_date 2027-01-01T00:00:00Z (API probe 2026-09-01) — but it held 0 posts,
# the 'Forecast on Metaculus Cup' workflow is disabled_manually, and the bare
# `metaculus-cup` slug now 404s, so METACULUS_CUP_ID above no longer auto-resolves and
# must be pointed at the dated slug (or id 33108) when configuring. The operator expects
# questions ~2026-09-20 and asked for runs to START ERRORING from FALL_CUP_REMINDER_DATE
# as the reminder to configure/enable the cup. Flipping FALL_CUP_CONFIGURED to True is
# the acknowledgment that retires the whole check (and its companion CI time-bomb test
# in tests/test_tournament_dates.py).
#
# Live window on tournament crons is narrow by design: check_tournament_dates raises
# TournamentExpiredError from 2026-09-20 (end date + hard stop) anyway, so there this
# reminder only adds 09-15..09-20 — but it also reddens the cup/minibench crons and any
# manual or test run after, and the CI test keeps failing regardless of run mode.
FALL_CUP_SLUG: str = "metaculus-cup-fall-2026"
FALL_CUP_REMINDER_DATE: str = "2026-09-15"
FALL_CUP_CONFIGURED: bool = False  # flip to True once the fall cup constants + workflow are set up


def fall_cup_reminder_due(today: date | None = None) -> bool:
    """Whether the fall-cup configuration reminder should redden runs.

    False before ``FALL_CUP_REMINDER_DATE`` and always False once the operator flips
    ``FALL_CUP_CONFIGURED``. ``today`` defaults to the system clock read at CALL time,
    same contract as ``credit_alerts_active`` below: tests inject a fixed date, and a
    long-lived process crosses the date without a redeploy.
    """
    if FALL_CUP_CONFIGURED:
        return False
    # Local calendar day is deliberate on an operator-facing dated lever, and prod
    # (UTC runners) sees no difference — same rationale as credit_alerts_active.
    return (today or date.today()) >= date.fromisoformat(FALL_CUP_REMINDER_DATE)  # noqa: DTZ011  # see comment above


def check_fall_cup_reminder(logger: logging.Logger | None = None, today: date | None = None) -> bool:
    """Log the loud FALL_CUP_REMINDER line when due; return whether it fired.

    The caller (cli.main) holds the returned bool and exits non-zero at end of run,
    the same shape as the credit-floor path: forecasting and publishing complete
    normally, and the red exit is purely the reminder signal.
    """
    if not fall_cup_reminder_due(today):
        return False
    log = logger or logging.getLogger(__name__)
    log.error(
        f"FALL_CUP_REMINDER: {FALL_CUP_SLUG} expected to open ~2026-09-20 — enable the "
        f"'Forecast on Metaculus Cup' workflow and set the season constants "
        f"(METACULUS_CUP_ID no longer auto-resolves: the bare 'metaculus-cup' slug 404s, "
        f"point it at {FALL_CUP_SLUG}). Flip FALL_CUP_CONFIGURED=True in constants.py to "
        f"retire this reminder. This run will exit non-zero as the reminder signal."
    )
    return True


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

# Optional override for the --mode test_questions question set. When set to a
# non-empty comma/whitespace-separated list of Metaculus question URLs, the
# test_questions path forecasts exactly those instead of the hardcoded evergreen
# EXAMPLE_QUESTIONS list (see cli.py). Used by the test_bot_basic workflow to run
# a single question end-to-end; unset preserves full test_bot behavior.
TEST_QUESTIONS_OVERRIDE_ENV: str = "TEST_QUESTIONS_OVERRIDE"

# Credential env-var names. Named constants (matching the existing *_ENV
# convention used for GOOGLE_API_KEY_ENV / FRED_API_KEY_ENV) so the literal
# strings aren't duplicated across api_key_utils / fallback_openrouter /
# research_providers / research_orchestrator — that duplication is exactly the
# typo risk the convention exists to prevent. See CLAUDE.md "API keys & secrets"
# for which of these are shared (donated) vs. personal.
OPENROUTER_API_KEY_ENV: str = "OPENROUTER_API_KEY"
OAI_ANTH_OPENROUTER_KEY_ENV: str = "OAI_ANTH_OPENROUTER_KEY"
ASKNEWS_CLIENT_ID_ENV: str = "ASKNEWS_CLIENT_ID"
ASKNEWS_SECRET_ENV: str = "ASKNEWS_SECRET"  # noqa: S105  # env var NAME, not a credential
EXA_API_KEY_ENV: str = "EXA_API_KEY"
PERPLEXITY_API_KEY_ENV: str = "PERPLEXITY_API_KEY"
METACULUS_TOKEN_ENV: str = "METACULUS_TOKEN"  # noqa: S105  # env var NAME, not a credential


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


def _date_env(name: str, default: date) -> date:
    """Parse an ISO ``YYYY-MM-DD`` env var into a date, falling back on garbage."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if raw == "":
        return default
    try:
        return date.fromisoformat(raw)
    except ValueError:
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
# Sizing: each phase (hot + historical) sleeps before its first call and applies
# backoff `ASKNEWS_BACKOFF_SECS * (10 + 3**attempt)` on 429/rate-limit retries, up
# to ASKNEWS_MAX_TRIES attempts per phase (see research/providers.py). This wall
# sits above that whole two-phase retry envelope plus API time, with headroom,
# while still bounding a genuine hang.
ASKNEWS_WALL_TIMEOUT: int = 300

# --- OpenRouter credit telemetry ---
# End-of-run floor for the DONATED key's remaining balance (limit_remaining).
# Below this, cli.main logs a loud warning and exits non-zero AFTER all
# forecasting/publishing completes — a reminder-to-refill signal, not an abort.
# The floor is meaningless for the personal key (no limit_remaining), so it is
# only checked against the donated key. See metaculus_bot/credit_telemetry.py.
OPENROUTER_CREDIT_FLOOR_USD: float = _float_env("OPENROUTER_CREDIT_FLOOR_USD", 1.0)

# Dated suppression of the credit ALERTS (not the logs). The operator is funding
# the rest of the season out of pocket, so an empty donated key is expected
# rather than a defect, and the two paths that turn a credit shortfall into a
# non-zero exit — the floor breach in cli.main and the credit-caused
# donated->personal fallbacks folded into ``alertable`` — must not redden CI
# until this date. The tournament closes on TOURNAMENT_END_DATE (2026-09-06);
# alerting resumes a few days later so a stale suppression can't outlive the
# season. Every CREDIT_* log line, including CREDIT_FLOOR_BREACH, keeps firing
# throughout: only the exit status and the alertable arithmetic change.
#
# Non-credit fallback causes (401 invalid/disabled key, 404 no-allowed-providers,
# 429 rate limit, guardrail/data-policy) stay fully alertable — each of those is
# real breakage, not an expected empty wallet.
CREDIT_ALERT_RESUME_DATE: date = _date_env("OPENROUTER_CREDIT_ALERT_RESUME_DATE", date(2026, 9, 10))


def credit_alerts_active(today: date | None = None) -> bool:
    """Whether credit shortfalls should still exit non-zero.

    False during the suppression window, True from ``CREDIT_ALERT_RESUME_DATE``
    onward. ``today`` defaults to the system clock read at CALL time (not at
    import), so a long-lived process crosses the resume date without a redeploy
    and tests can inject a fixed date instead of depending on the wall clock.
    """

    # Local calendar day is deliberate: this gates a sys.exit(1) and the resume date is an
    # operator-facing lever read in the operator's own calendar day. Prod runs on UTC GitHub
    # Actions runners, so local == UTC there; datetime.now(UTC).date() would only move the
    # boundary in a local dev shell.
    return (today or date.today()) >= CREDIT_ALERT_RESUME_DATE  # noqa: DTZ011  # see comment above


# Prediction-market venues (or prefetch catalogues) whose degradation is KNOWN and
# ACCEPTED, each with a dated resume. Same contract as CREDIT_ALERT_RESUME_DATE
# above: the finding is still logged in full, still rides the PROVIDER_DEGRADATION
# marker, and still names its resume date in the end-of-run summary — only its
# contribution to ``alertable`` is dropped, and only until the date. Dated rather
# than a bare boolean so a stale acceptance cannot outlive the season unnoticed,
# and per-venue rather than global so accepting a dead Manifold does not blind the
# operator to a dead Kalshi.
#
# Ships EMPTY on purpose. Both degradations this machinery was built for (Kalshi's
# blank liquidity labels, Manifold's zero contribution) are being FIXED in the same
# round, so suppressing either would hide the fix's own verification. The mechanism
# exists to give the operator a documented, dated lever instead of reaching for a
# code deletion when a venue is genuinely dead for good.
PROVIDER_DEGRADATION_SUPPRESSED_UNTIL: dict[str, date] = {}


def provider_degradation_alerts_active(venue: str, today: date | None = None) -> bool:
    """Whether ``venue``'s provider-degradation findings should still exit non-zero.

    ``today`` defaults to the system clock read at CALL time (not at import), so a
    resume needs no redeploy and tests can inject a fixed date. A venue with no
    entry is always alertable.
    """
    resume = PROVIDER_DEGRADATION_SUPPRESSED_UNTIL.get(venue)
    # Same contract as credit_alerts_active above: local calendar day is deliberate on an
    # operator-facing dated lever, and prod (UTC runners) sees no difference.
    return resume is None or (today or date.today()) >= resume  # noqa: DTZ011  # see comment above


# --- Forecasting clamps and numeric smoothing ---
# Binary prediction clamp. Mirrors Preseen-Atlas's clip-only tail protection
# (Atlas publishes `0.96 * estimate + 0.02`; we adopt the clip portion only).
# See scratch_docs_and_planning/atlas_inspired_improvements.md Workstream B.
BINARY_PROB_MIN: float = 0.02
BINARY_PROB_MAX: float = 0.98

# The "extreme band" on a binary probability: a member call at or past either edge.
# Nothing here clamps or gates anything — the band only decides which per-member
# EXTREME_CALL telemetry lines get logged (metaculus_bot/extreme_call.py), so that
# the lone-versus-accompanied extreme split is a query instead of a hand
# reconstruction from parsed comments every residual round. Membership is inclusive
# at both edges.
#
# The clamp a single-survivor binary publish goes through is DEFINED as these two
# constants: THIN_PUBLISH_BINARY_FLOOR / THIN_PUBLISH_BINARY_CEIL in this same section
# alias them rather than restating the literals, so the telemetry that measures the
# exposure and the clamp that prices it cannot drift apart. Retuning the band here
# retunes both, which is the intent — retune them together or not at all.
# Evidence for the 0.05/0.95 edges:
# scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md §2 ("The mechanism")
# — 9 lone extreme binary calls, 4 right, at a mean stated confidence of 0.972.
EXTREME_CALL_LOW: float = 0.05
EXTREME_CALL_HIGH: float = 0.95

# Floor on the PUBLISHED binary probability when exactly ONE forecaster survived
# (apply_thin_publish_floor in post_processing.py, wired in
# AggregationPipeline._base_combine on the "single_forecaster" skip reason).
#
# Mechanism, not a fit: the median of an intact ensemble absorbs a member's extreme
# tail call, and median-of-1 supplies no such variance reduction, so the range the
# published value may occupy is NARROWED in exactly that state to price the missing
# aggregation — [0.05, 0.95] sits strictly inside the per-model clamp
# [BINARY_PROB_MIN, BINARY_PROB_MAX] = [0.02, 0.98] the member already passed. It fires
# ONLY on a single-survivor publish — a multi-member median publishes as is, even one
# below 0.05 — and never touches the per-model record (the survivor's declared value
# stays on the comment's summary bullet; only the published aggregate moves).
#
# Evidence, with its honest caveat: the whole measured benefit is ONE question. q44874
# published gemini's lone 0.03 on a YES resolution and took -105.27 spot peer; at
# [0.05, 0.95] it is +51.08 with zero measured cost on the other three archived solo
# binaries (one win, three exact zeros — n=4, one non-zero row). The value 0.05 is
# informed by that question; 0.07 / 0.10 buy more on 44874 but start taxing 44870 and
# 44873, which were right. Bounded downside: publishing 5% where a correct sub-5% call
# would have scored costs ~-3.11 spot peer per instance, against a -105 tail.
# Always-on and global variants were priced and REJECTED (always-on never improves the
# published pre-flip ensemble; global [0.05, 0.95] over 408 binaries is -52.02, 50
# losses to 1 win) — do not widen the trigger. Receipt:
# scratch/residual_2026-08-31/gemini_review/RECOMMENDATION.md §2 (clamp-variant table
# + "A synthesis correction the individual cuts miss") and §3 option "1=".
#
# The edges REUSE the extreme-band constants above by aliasing them, so there is one
# definition of "extreme" serving both the telemetry and the clamp and no pair of
# literals to fall out of step. Retune EXTREME_CALL_LOW / EXTREME_CALL_HIGH to move
# both, or neither.
THIN_PUBLISH_BINARY_FLOOR: float = EXTREME_CALL_LOW
THIN_PUBLISH_BINARY_CEIL: float = EXTREME_CALL_HIGH

# Multiple-choice prediction clamp. Aligned to forecasting-tools 0.2.92's
# PredictedOptionList validator, which unconditionally clamps every option into
# [0.01, 0.99], renormalizes, and raises ValueError when any option moves > 0.05
# from its input. Matching those bounds makes the upstream validator a no-op on
# our already-clamped, sum-1 output, eliminating publish-time ValueError risk on
# many-option ballots (a dominant option + several near-floor options is exactly
# where the upstream renormalize-after-clamp fires the >0.05 raise). See
# clamp_and_renormalize_probs / clamp_and_renormalize_mc, which clamp BEFORE every
# PredictedOptionList construction.
MC_PROB_MIN: float = 0.01
MC_PROB_MAX: float = 0.99

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
# Binary: probability range (max - min) across per-model predictions. Chosen because
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
# above is the litellm per-HTTP-request timeout; it resets across retries, so an
# un-pinned allowed_tries multiplies it (build_native_search_llm pins
# allowed_tries=1 for exactly that reason) and it was observed defeated
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
# wall-clock and far more headroom under NATIVE_SEARCH_WALL_TIMEOUT
# / NATIVE_SEARCH_TIMEOUT. The quality cost of low is now absorbed by
# the model-tier upgrade above (smarter model at lower effort). Override via
# NATIVE_SEARCH_REASONING_EFFORT env if a workflow needs medium back. Note:
# this default applies ONLY to the native-search provider —
# DISAGREEMENT_ANALYZER_LLM is also at low (llm_configs.py), while the
# forecaster slots set their own effort per-instance in llm_configs.py.
NATIVE_SEARCH_REASONING_EFFORT_DEFAULT: str = "low"
NATIVE_SEARCH_VERBOSITY_ENV: str = "NATIVE_SEARCH_VERBOSITY"
NATIVE_SEARCH_VERBOSITY_DEFAULT: str = "low"
# Native search web options (passed to OpenRouter plugins)
NATIVE_SEARCH_MAX_RESULTS: int = 20
NATIVE_SEARCH_CONTEXT_SIZE: str = "high"  # "low", "medium", "high"

# --- Perplexity (fallback research provider; dormant while AskNews wins the ladder) ---
# The model both Perplexity call sites use (research/providers.py's provider factory
# and the orchestrator's AskNews-failure fallback). Single constant because the two
# sites each carried their own literal and silently drifted: providers.py was pinned
# to Perplexity's non-reasoning tier while the orchestrator used the reasoning one. The
# direct-provider route takes the bare slug; the OpenRouter route takes it prefixed,
# which is what ``get_openrouter_api_key`` keys its routing on.
PERPLEXITY_RESEARCH_MODEL: str = "perplexity/sonar-reasoning-pro"
PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER: str = f"openrouter/{PERPLEXITY_RESEARCH_MODEL}"

# Wall-clock cap for the two Perplexity call sites. Both previously had NO wall bound
# at all — unlike native search and resolution-source, neither was ever migrated to
# the gated retry wrapper, so a stalled reasoning-tier call could run as long as
# litellm let it. Sized between GEMINI_SEARCH_TIMEOUT and the research phase's own
# budget: generous enough for a reasoning search, bounded enough that a stall
# can't dominate the phase.
PERPLEXITY_WALL_TIMEOUT: float = 300.0

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
# An extraction at or above the JS-wall floor can still be pure page chrome: a tab list,
# a region selector, a feedback-form blurb, an "about the data" note. Below this many
# extracted chars a 200-OK page is withheld as `no_resolving_content` rather than rendered
# as grading evidence, whatever became of the content — `status_reason` records whether a
# routeless data embed was named (`embed_shell`, Infogram / Flourish / Tableau, see
# `unreadable_data_embed_providers`) or not (`thin_page`). The floor was gated on a named
# provider when it shipped for qids 44554/44556, which withheld one shape of chrome and
# published the other: the 2026-09-01 round found five content-free `success` renders and
# not one of them named a provider.
# Calibrated on the 89 archived resolution_source records, re-checked for the ungated rule
# 2026-09-02: of 68 cited successes, 8 sit under 400 chars and all 8 are chrome — region
# selectors (data.wastewaterscan.org, 127, twice), Kazakh region names (election.gov.kz,
# 385), AP org boilerplate (355), an ABS release-date list with no figure (344), a tracker's
# "about the data" note (262), a feedback-form blurb (camara.leg.br, 157), a
# clinicaltrials.gov data-element pointer (111) — and the SHORTEST archived extraction that
# actually carries the resolving content is 401 (myfloridaelections.com's election-date
# table). So this is the observed elbow, and it stays deliberately below it: a page above
# the floor keeps its text (plus the embed disclosure where one applies), because
# withholding a terse-but-real data table costs more than leaving one shell visible.
RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS: int = 400
# --- Inline chart configs (Highcharts), read straight out of the page we already hold ---
# qid 43949: the resolving IOM page fetched 200 and extracted ~80k chars of incident rows
# and prose carrying none of the resolving figures, because the annual series lives in
# `<div class="charts-highchart" data-chart="{...}">`. `research/resolution_chart_data.py`
# unescapes and json.loads that config — zero LLM calls, no second request. Charts are read
# on EVERY fetched HTML page, not only thin ones, because that page's prose was far above
# the shell floor and a thin-only gate would miss the record the rung exists for.
RESOLUTION_SOURCE_CHART_MAX_CHARTS: int = (
    3  # the resolving chart is ~first in document order (IOM page carries 5), same assumption the Datawrapper hop makes
)
RESOLUTION_SOURCE_CHART_MAX_SERIES: int = 4  # IOM's widest chart is 3 series (Undetermined / Female / Male)
# Points per series, kept from the END (the resolving value is the newest one). 16 keeps a
# full annual series intact — IOM's is 13 points, 2014..2026 — while bounding its 149-point
# monthly sibling to roughly the last year and a half.
RESOLUTION_SOURCE_CHART_MAX_POINTS: int = 16
# Hard cap on the whole rendered block, budgeted out of (never added on top of) the 6,000-char
# per-URL page cap, so chart data can never evict more than a third of a cited page's text.
# Measured: the IOM page's three readable charts render in ~700 chars together.
RESOLUTION_SOURCE_CHART_BLOCK_MAX_CHARS: int = 2000
# Configs examined per page before the scan stops, and the per-config char bound (which also
# bounds the brace scan for the inline-script form). Both exist so a page with hundreds of
# `data-chart` attributes, or one unclosed brace, costs a fixed amount of work.
RESOLUTION_SOURCE_CHART_MAX_CANDIDATES: int = 20
RESOLUTION_SOURCE_CHART_MAX_CONFIG_CHARS: int = 200_000
# --- Datawrapper second hop (Tier 2) ---
# Poll-tracker pages lock their resolving daily series inside Datawrapper
# iframes that trafilatura drops (qids 44858/44841). The hop fetches the
# version-free live dataset (static.dwcdn.net/data/<id>.csv) for charts found
# in a fetched page's raw HTML.
RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS: int = (
    3  # datasets per question; hero/resolving chart is ~always first in document order (Trump tracker carries 5 embeds)
)
# The hop runs as a SECOND network phase after the Tier-1 page gather, inside the same
# 45s provider wall, and the datasets share one CDN host so the per-host politeness
# semaphore serializes them (worst case MAX_CHARTS x the 20s HTTP timeout = 60s > the
# wall). Bounding the hop at whatever wall budget remains — and skipping it below this
# floor — is what keeps a slow CDN tail from cancelling the whole provider and throwing
# away Tier-1 pages that already fetched. The floor admits at least one typical dwcdn
# fetch (a poll CSV is tens of KB off a CDN, sub-second-to-~2s — same probe basis as the
# HTTP timeout's "0-2s typical"); below it the hop cannot land anything and the pages
# are worth more than the attempt.
RESOLUTION_SOURCE_DATAWRAPPER_MIN_HOP_BUDGET_S: float = 3.0
# Wall margin the hop leaves the outer wait_for, so the inner bound fires first and the
# provider returns the pages instead of being cancelled mid-render.
RESOLUTION_SOURCE_DATAWRAPPER_HOP_WALL_MARGIN_S: float = 2.0
# Per-dataset render cap, deliberately WELL under the 6,000-char page cap: a
# middle-truncated daily series keeps ~12 rows at each end per 1,000 chars, so 3,000
# still carries weeks of values around today, and the formatter budgets datasets against
# their own allowance (MAX_CHARTS x this) so a chart's data can never evict the cited
# page text the section exists to serve.
RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS: int = 3000
RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS: float = 30.0  # freshness bound on the dataset's Last-Modified vs fetch time. Live trackers republish at least daily; the stale-route failure class this guards against served 5-14 MONTH old snapshots as HTTP 200 (2026-08-24 verifications). Older/undatable data is withheld (stale_data), never served as live.

# --- Local document text (PDFs read with pypdf, `research/document_text.py`) ---
# Measured 2026-09-03: local pypdf pulled 833,450 chars out of a 6.7 MB 220-page PDF in
# 5.3 s and the passage the research driver was looking for was in it, while the paid
# Gemini url_context read of the same file returned nothing. So a PDF we already hold is
# extracted and passage-selected locally, and a model call is spent only on a document we
# cannot read at all.
DOCUMENT_TEXT_MAX_PAGES: int = 400  # ~2x the 220-page document behind the measurement, so a normal government report reads whole while a 4,000-page appendix dump stays bounded
DOCUMENT_TEXT_MAX_SECONDS: float = 20.0  # ~4x the measured 5.3 s for 220 pages; matches RESOLUTION_SOURCE_HTTP_TIMEOUT, so parsing a document costs no more of the research phase than fetching it did
DOCUMENT_TEXT_PDF_MAX_BYTES: int = (
    40 * 1024 * 1024
)  # ~6x the measured 6.7 MB file; above this the parse is not worth a research phase, and the bytes are refused before pypdf allocates
DOCUMENT_DIGEST_TOP_K: int = 6  # passages per document: 6 x DOCUMENT_DIGEST_WINDOW_CHARS is ~3.6k chars (~900 tokens), the same order as one cited page under RESOLUTION_SOURCE_PER_URL_MAX_CHARS
DOCUMENT_DIGEST_WINDOW_CHARS: int = (
    600  # ~1 paragraph of a report; mirrored as document_text.DEFAULT_WINDOW_CHARS, pinned equal by a test
)
# A document we ALREADY hold whose estimated token count (chars / 4) exceeds this is never
# sent to a paid url_context read — the digest serves it instead. The nine archived documents
# above this bound carried 67% of all reader tokens, and the 833k-char case above is the shape
# that spends most: the paid read of it returned nothing, so the spend bought a null answer.
URL_CONTEXT_SIZE_GATE_TOKENS: int = 100_000

# --- Resolution-source escalation rungs (free ones: meta-refresh hop, local PDF read) ---
# Every rung runs INSIDE the unchanged 45 s provider wall, and the outer `asyncio.wait_for`
# discards every page that already fetched when it fires, so each rung is self-bounding on
# the same pattern as the Datawrapper hop: wall minus elapsed minus a margin, skipped below
# a floor, degrading to whatever the direct route already got.
RESOLUTION_SOURCE_RUNG_WALL_MARGIN_S: float = 2.0  # margin left to the outer wait_for so the rung returns first (the Datawrapper hop keeps its own historically-named twin)
RESOLUTION_SOURCE_META_REFRESH_MIN_BUDGET_S: float = (
    3.0  # the hop is one more page GET; same "0-2 s typical" probe basis as the HTTP timeout
)
# Floor for the LOCAL pypdf parse, which spends CPU rather than network. It doubles as the
# minimum `max_seconds` handed to `extract_pdf_text` (the budget is capped at
# DOCUMENT_TEXT_MAX_SECONDS above it), and 3 s is ~60% of the measured 5.3 s for 220 pages,
# so a short document still reads whole and a long one comes back partial-but-labelled
# rather than not at all.
RESOLUTION_SOURCE_PDF_MIN_BUDGET_S: float = 3.0

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
# Thinking level for the grounded-search call, set EXPLICITLY (operator decision
# 2026-09-03) rather than left at the model's default, which for
# gemini-3-flash-preview is HIGH: 71% of the grounded-search output tokens measured
# in the 2026-09 spend reconstruction were thinking tokens, and this is a retrieval
# + summarise task rather than a reasoning one. Only the LEVEL is set — the
# no-max_tokens rule above still holds, because capping output on a thinking model
# is what caused the silent truncations. NOTE thinking_level is the Gemini 3 knob; the
# 2.5 line takes a thinking_budget instead, so the GEMINI_SEARCH_MODEL escape hatch to
# gemini-2.5-flash mentioned above needs this revisited rather than merely re-pointed.
GEMINI_SEARCH_THINKING_LEVEL: str = "medium"
# Client-side PER-ATTEMPT HTTP timeout (ms) and attempt count (including the first)
# for the grounded-search client. The SDK retries nothing by default, so a fast
# transient — the 503 UNAVAILABLE that killed two production calls — used to lose the
# whole provider; one retry recovers it.
#
# Worst-case arithmetic. The hard bound is UNCHANGED: the outer
# ``asyncio.wait_for(..., GEMINI_SEARCH_TIMEOUT)`` in ``research/gemini_search.py``
# still cancels the whole call at 360s, and it genuinely can (an async coroutine,
# unlike read_document's thread). So the nominal product 350 + <=2 (one jittered retry
# sleep) + 350 never elapses: a first attempt that HANGS eats the window and the outer
# wait_for fires exactly as it does today, while a retry only completes when the first
# attempt failed FAST, which is the recovery case this exists for.
#
# Why the per-attempt cap is not sized so the PRODUCT fits under 360s: that would need
# <=176s per attempt, and the comment above records legitimate AFC chains at 150-200s
# ("180s was too tight — observed timeouts"). Shrinking the per-attempt allowance would
# newly fail calls that succeed today, which is the one thing a timeout change here
# must not do. 350s instead sits just under the outer deadline, so nothing a single
# attempt can do today is cut short.
GEMINI_SEARCH_HTTP_TIMEOUT_MS: int = 350_000
GEMINI_SEARCH_HTTP_ATTEMPTS: int = 2
# Same two settings for gap-fill v2's read_document backend, whose thinking level is a
# tier lower (operator decision 2026-09-03): quoting a fetched document back is the
# least reasoning-heavy of the Gemini calls. The reader's per-attempt TIMEOUT is not a
# constant here because it is DERIVED from the total in-thread budget
# (``_READ_DOCUMENT_HTTP_TIMEOUT_MS`` in ``research/agentic/tool_backends.py``, where the
# arithmetic lives): that call runs under ``asyncio.to_thread``, so its outer wait_for
# cannot cancel it and the retry has to fit INSIDE today's budget rather than beside it.
GAP_FILL_V2_READER_THINKING_LEVEL: str = "low"
GAP_FILL_V2_READER_HTTP_ATTEMPTS: int = 2

# --- Second-pass gap-fill ---
# After first-pass research completes, a cheap analyzer identifies up to
# GAP_FILL_MAX_GAPS factual gaps; each is resolved by a parallel OpenAI native
# web search (see GAP_FILL_RESOLVER_MODEL below). Fails soft — forecast proceeds
# with first-pass research alone if any stage errors out.
GAP_FILL_ENABLED_ENV: str = "GAP_FILL_ENABLED"
# Non-grounded gap-listing: reads the first-pass research and emits a JSON list
# of up to GAP_FILL_MAX_GAPS factual gaps, under the tight
# GAP_FILL_ANALYZER_WALL_TIMEOUT cap, which
# soft-fails silently on breach — terra-low is the latency-safe choice; the
# task is decomposition, not deep judgment. Grounded search resolution still
# uses google-genai directly via gemini_search_provider — that path needs the
# search index.
GAP_FILL_ANALYZER_MODEL: str = "openrouter/openai/gpt-5.6-terra"
# 2026-07-20: 5 → 4. The transcript vibe-analysis (Fable) found no positional
# value cliff, so a fixed cutoff is safe now that the analyzer prompt ranks gaps
# by decision-relevance (see gap_fill_analyzer_prompt — the 4th slot holds the
# least valuable of the kept gaps rather than a random one). 3 was still judged
# unsafe, but the 5th gap is empirically a completeness-stretch: only ~27% of 221
# archived bundles rendered a 5th gap and the observed 5th gaps were
# confirmatory, so 5→4 is near-zero risk. Do NOT go below 4.
GAP_FILL_MAX_GAPS: int = 4
# Analyzer call is non-grounded (no Google Search) and should return quickly.
# Use a tight timeout to prevent a single hung analyzer request from holding a
# research concurrency slot for the full grounded-search budget.
GAP_FILL_ANALYZER_TIMEOUT: int = 120
# Wall-clock backstop for the analyzer call. Slight headroom over
# GAP_FILL_ANALYZER_TIMEOUT so the cleaner per-request error from litellm fires
# first when possible (auth failure, model-not-found, etc.) — same pattern as
# NATIVE_SEARCH_WALL_TIMEOUT vs NATIVE_SEARCH_TIMEOUT. Without this,
# asyncio.wait_for and the litellm request timeout fire at the exact
# same second and we lose the descriptive error message.
GAP_FILL_ANALYZER_WALL_TIMEOUT: int = 135
# Skip gap-fill when the first-pass research blob has less than this many
# non-whitespace characters — likely indicates all providers soft-failed and
# gap-fill would just hallucinate gaps or burn quota.
GAP_FILL_MIN_RESEARCH_CHARS: int = 200
# 2026-06-25: migrated the per-gap RESOLVER off direct-Google grounded Gemini
# (google-genai, personal GOOGLE_API_KEY) to OpenAI native web search via
# OpenRouter, which bills the Metaculus-donated key. The resolver fanned out up
# to GAP_FILL_MAX_GAPS parallel grounded calls per question — a per-gap cost
# multiplier on the personal Google bill, the dominant unwanted spend. The
# single first-pass grounded Gemini call stays on google-genai (operator is fine
# paying for 1 call/question, and it uses url_context which OpenRouter can't
# replicate). No "openrouter/" prefix here — build_native_search_llm adds it.
#
# Agentic single-gap web research whose source-trust judgment lands directly in
# every forecaster prompt. The workers run in parallel under
# NATIVE_SEARCH_WALL_TIMEOUT (latency = slowest call, not sum), so effort stays
# LOW. 2026-07-20: sol → terra. terra
# was preferred or within-noise vs sol across all three 2026-07 blind role audits
# at ~40-50% lower cost, and these searches are ~44% of research spend (17 calls
# in the 2026-07-19 run) — the single biggest research line item, so the cost cut
# is the dominant consideration. (The 2026-07-09 bench had sol-low matching
# terra-low coverage 24/25; the blind audits plus the cost weight flip it.)
GAP_FILL_RESOLVER_MODEL: str = "openai/gpt-5.6-terra"
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
# lives, so batching is encouraged rather than rationed. Raised with the W2
# ambition floor (2026-07-21): v2 runs 41-60s of the GAP_FILL_V2_WALL_DEADLINE
# budget, so the headroom is free — the satisficing problem was ambition, not
# budget, and with the conclude-gate floor in place the extra slots let the driver dig
# deeper on the few decision-relevant gaps instead of stopping early.
GAP_FILL_V2_MAX_TOOL_CALLS: int = _int_env("GAP_FILL_V2_MAX_TOOL_CALLS", 30)
# Hard wall for the whole loop — inside v1's worst-case envelope
# (GAP_FILL_ANALYZER_WALL_TIMEOUT then the resolver wave under
# NATIVE_SEARCH_WALL_TIMEOUT), so running v2 concurrently with v1 adds
# no research-phase wall-clock. The loop is anytime: hitting the deadline
# emits banked findings, never "".
GAP_FILL_V2_WALL_DEADLINE: float = _float_env("GAP_FILL_V2_WALL_DEADLINE", 540.0)
# With less than this many seconds remaining, the harness rejects every tool
# except conclude, forcing the loop to wrap up inside the wall deadline.
GAP_FILL_V2_CONCLUDE_THRESHOLD: float = _float_env("GAP_FILL_V2_CONCLUDE_THRESHOLD", 90.0)
# Below this many extracted chars, the fetch ladder escalates plain HTTP to
# headless-Chromium rendering (JS-wall heuristic; tools.py consumes this).
GAP_FILL_V2_MIN_CONTENT_CHARS: int = _int_env("GAP_FILL_V2_MIN_CONTENT_CHARS", 500)
# Max ranked gaps the driver's set_research_plan tool may register (W1). An
# INDEPENDENT knob from v1's GAP_FILL_MAX_GAPS: v2 gaps are cheap (no dedicated
# per-gap search call — the driver works one shared tool budget), but a focused
# work-list still beats a sprawling one. The gap list is ranked by
# decision-relevance, so the cap drops the least forecast-moving gaps.
GAP_FILL_V2_MAX_GAPS: int = _int_env("GAP_FILL_V2_MAX_GAPS", 4)

# --- Financial Data Provider ---
FINANCIAL_DATA_ENABLED_ENV: str = "FINANCIAL_DATA_ENABLED"
FRED_API_KEY_ENV: str = "FRED_API_KEY"
# Binary-ish routing classification (is this a financial/economic question?)
# under a 30s timeout — capability-saturated, so it rides the cheapest capable
# tier (mini → luna 2026-08-03, when luna's markdown made it cheaper than mini).
FINANCIAL_CLASSIFIER_MODEL: str = "openrouter/openai/gpt-5.6-luna"
FINANCIAL_CLASSIFIER_TIMEOUT: int = 30
# Calendar-day lookback behind every yfinance history() fetch. BOTH paths (live and
# backtest) fetch by explicit start date = as_of - this many days, end-inclusive, so the
# window holds LOOKBACK+1 calendar dates. Never spent as a bare `period="Nd"`: Yahoo's
# chart API reads that custom range as N trading BARS for listed assets but ~N calendar
# DATES for 24/7 ones — one integer under two unit systems, which is how the listed-asset
# backtest margin was never sized at all. Sized so the deepest consumers clear on BOTH
# daily-bar bases, with real headroom:
#   - 365 basis (24/7 markets): the "1y" return needs an observation ≥366 days back, and
#     the 52-week slice wants 365 rows → 391 dates leave ~25 Yahoo gap-days of tolerance
#     (the old 372 left 6, and a persistent one-day BTC-USD hole has been observed live).
#   - 252 basis (exchange-traded): the "1y" return reaches ~365 calendar days back and
#     the 52-week slice wants 252 bars → 391 dates ≈ 265 bars at the worst observed NYSE
#     density (253 bars per 373-date window, measured over three years of real SPY
#     windows), ~12 bars of margin where the old 372 measured margin EXACTLY zero.
FINANCIAL_YFINANCE_LOOKBACK_DAYS: int = 390
FINANCIAL_YFINANCE_RECENT_DAYS: int = 30
# Variance-ratio screen for a vendor-noise-dominated daily series (q44797: USD/SZL's
# 17.8% "volatility" was 79% quote noise on a pegged cross, and all six forecasters sized
# their intervals off it). VR(q) near 1 is a random walk; well below 1 means each day's
# move is largely reversed the next, which is what a thin quote on a fixed cross looks
# like and what cancels over multi-day windows.
#   - LAG 5 (one trading week) because it is the horizon where the two cases separate: the
#     44797 verification (§11) measured VR(5) 0.472 on the noisy series against 0.740 on
#     the clean anchor, while at VR(10) the CLEAN series read 0.208 — no separation left.
#   - FLOOR 0.6 sits between those two, and is calibrated against seeded fixtures in
#     tests/test_timeseries_anchor_provider.py (TestVarianceRatio) rather than against the
#     receipt's own numbers, which came from a differently-parameterised estimator.
#   - MIN_RETURNS 120 because the null standard error of VR(5) is ~sqrt(4.8/n): ~0.20 at
#     n=120 and ~0.40 at n=30, so the 30-row vol window cannot carry this statistic. The
#     provider's own FINANCIAL_YFINANCE_LOOKBACK_DAYS window holds ~265 daily bars, so a
#     normal fetch clears the floor with room; a short/gappy one gets no flag at all.
FINANCIAL_VARIANCE_RATIO_LAG: int = 5
FINANCIAL_VARIANCE_RATIO_FLOOR: float = 0.6
FINANCIAL_VARIANCE_RATIO_MIN_RETURNS: int = 120
# How many recent FRED prints the first-release-vs-current-vintage table covers. Revising
# macro series resolve on the FIRST print (q44944 resolved on first-release Case-Shiller)
# while the levels rendered beside it are today's revised vintage, so the gap between the
# two is a forecastable, signed quantity. Four prints is enough to read a revision
# direction on a monthly series without turning the block into a table nobody reads.
FINANCIAL_FRED_VINTAGE_PRINTS: int = 4
# Cap on how many tickers + FRED series one question may fetch. The identifier list is
# whatever an LLM classifier named plus whatever URL extraction found, and it was
# previously unbounded: each identifier gets its own asyncio.to_thread, all of them
# landing in the process-wide default executor that every other blocking call shares
# (ts_fetch, resolution_source, the agentic fetch ladder, the /auth/key probe). Tasks
# queued behind a saturated pool burn their wait_for budget without executing, so an
# over-eager classification on one question degrades unrelated providers on others. 12 is
# well above any plausible real question (the classifier prompt asks for the resolving
# series, not a sector sweep) while bounding the worst case.
MAX_FINANCIAL_IDENTIFIERS: int = 12

# --- Soft deadlines to keep batch wall-clock inside the tournament cron window ---
# Per-forecaster outer deadline wrapped via asyncio.wait_for around each
# _make_prediction call. A single stuck forecaster used to be able to hold a
# question for REASONING_MODEL_CONFIG's litellm timeout times its allowed_tries
# (llm_configs.py); this caps that worst case, at which point the forecaster is
# dropped with a loud WARNING and the other models carry the ensemble.
FORECASTER_SOFT_DEADLINE: int = 600

# Minimum number of successful base forecasters required to publish a question.
# Below this, the question is skipped entirely rather than publishing a weak
# ensemble.
#
# 2026-07-20: lowered to 1 (was 3 → 2 → 1 over the day). A threshold equal to the
# roster width tolerates ZERO drops, each step below it tolerates one more, and 1
# accepts publishing on a single surviving forecaster.
# The operator accepts a single-forecaster publish: median-of-1 = the
# forecast itself, and exception-driven drops stay CI-visible (counted as
# degradation since 687e113), so a degraded run — even one thinned to a lone
# model — still reddens CI rather than silently withholding the question.
# NOTE: forecaster.py short-circuits the n==1 case before spread computation and
# stacking, because the spread_metrics helpers REQUIRE >=2 predictions and raise
# otherwise; see the single-forecaster guard in _research_and_make_predictions.
MIN_FORECASTERS_TO_PUBLISH: int = 1

# The Metaculus close window each scheduled run has to finish inside: the prod
# crons fire hourly, so a question's whole forecast-and-publish cycle must fit in
# one hour. Named because two deadlines below are sized against it and the
# arithmetic was previously a bare 3600 living only in prose.
METACULUS_CLOSE_WINDOW_SECONDS: int = 3600

# Per-question wall-clock cutoff, sized just inside METACULUS_CLOSE_WINDOW_SECONDS.
# At deadline, in-flight forecasters are cancelled; we base-combine
# whatever completed (>=MIN_FORECASTERS_TO_PUBLISH) and submit. The remainder —
# exactly WALL_CLOCK_STACKING_MIN_BUDGET, not a coincidence — reserves time for
# stacker-skip + publish (see PUBLISH_POST_TIMEOUT / PUBLISH_POST_RETRIES).
# tests/test_llm_retry.py pins both relationships.
PER_QUESTION_WALL_CLOCK_DEADLINE: int = 3510

# Below this remaining-budget threshold, skip stacking and force fallback_median
# aggregation. Sized to clear the publish-hardening worst case — the prediction
# POST plus the comment POST, each up to
# PUBLISH_POST_TIMEOUT * (PUBLISH_POST_RETRIES + 1) — plus headroom.
WALL_CLOCK_STACKING_MIN_BUDGET: int = 90

# --- Close-aware per-question time budget (metaculus_bot/time_budget.py) ---
# PER_QUESTION_WALL_CLOCK_DEADLINE above is sized against the CRON PERIOD, not
# against a question's own deadline, so on its own it lets a question closing in
# 20 minutes spend 58.5. These three size the close-derived budget that bounds it.
#
# Time held back from the budget so the PREDICTION POST can still land: ft's
# _post_question_prediction opens with one _sleep_between_requests (3.5-4.5s)
# before the POST, and publish_hardening bounds that POST at
# PUBLISH_POST_TIMEOUT * (PUBLISH_POST_RETRIES + 1) = 40s. 60 leaves ~15s slack.
# Deliberately SMALLER than WALL_CLOCK_STACKING_MIN_BUDGET, which reserves for
# BOTH POSTs: only the prediction has to beat the close, and a comment posted a
# few seconds late is still accepted.
PUBLISH_RESERVE_SECONDS: int = 60

# Below this effective budget, drop the OPTIONAL research stages (every provider
# but the primary, plus both gap-fill passes) and publish on the fast path.
# Sized at EXACTLY the full pipeline's configured worst case: research 1155s
# (provider phase 600 = AskNews 300 + summarizer 300 sequential inside one
# provider; then gap-fill 555 = analyzer 135 + resolver wave 420)
# + FORECASTER_SOFT_DEADLINE 600 + the publish tail 60 = 1815s. So the rule
# reads: stop running the optional stages once the full pipeline's worst case no
# longer fits the window — the value IS that sum, so there is no band where the
# envelope doesn't fit but the fast path stays off. (SUMMARIZER_WALL_TIMEOUT is
# defined further down this file, which is why the sum is stated rather than
# spelled as an expression.) Measured false-positive cost is zero — 0 of 99
# published triple-era questions had less than 54 minutes of headroom at run
# start (scratch/residual_2026-08-24/time_budget_design.md), and the optional
# stages are worth 84s (gap-fill v2 p50) to 183s (dropping the provider tail at
# its observed max).
TIME_BUDGET_FAST_PATH_THRESHOLD: int = 1815

# Below this budget the question is skipped at INTAKE rather than run on the
# fast path: the minimum viable path is the primary provider (measured worst
# ~110s live; the research phase's half-share of a 300s budget is 150s) plus at
# least one reasoning forecaster (typical completions run 100-300s against
# FORECASTER_SOFT_DEADLINE 600; the other half-share of 300s fits only the
# fastest of them). Below ~5 minutes even that path essentially never lands —
# the fan-out produces 0 valid forecasters and the min-forecasters guard drops
# the question AFTER spending — so the intake skip converts guaranteed-wasted
# spend into an immediate forfeit with a log line naming the close time.
TIME_BUDGET_MIN_VIABLE_S: int = 300

# Fraction of the TOTAL budget granted to the research phase as ONE fixed window
# anchored at the budget's start (research_phase_deadline_s = total*share -
# elapsed), enforced as a deadline on the parallel-provider phase and on each
# gap-fill pass. Fixed rather than a rolling share of remaining: research
# consults the deadline at two sequential points, and re-taking 50% of remaining
# at each compounds to ~75% of the budget — leaving the fan-out under its own
# soft deadline on the close-limited band this budget exists for. The fixed
# window guarantees forecast-and-publish the complementary half, and a slow
# intake spends research's half, not the forecast's. At the static 3510s budget
# the window is ~1755s, well above research's 1155s configured worst case, so it
# never fires on a roomy question; at a close-limited 2400s budget it splits
# 1200 research / 1200 forecast-and-publish.
RESEARCH_PHASE_BUDGET_SHARE: float = 0.5

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

# Stacker soft deadline. Set slightly above the stacker LLM's own litellm timeout
# (REASONING_MODEL_CONFIG in llm_configs.py) so the model's timeout fires first
# with a clean exception when possible;
# this wait_for is a final belt-and-suspenders backstop for a wholly
# stuck call. Stacker is configured with allowed_tries=1 in llm_configs.py so
# we only get one try before falling back.
STACKER_SOFT_DEADLINE: int = 500
# Stacker fallback model soft deadline. Tighter because we're already running
# late on the critical path by the time the fallback fires.
STACKER_FALLBACK_SOFT_DEADLINE: int = 300

# Per-question soft deadline for the disagreement-crux extractor
# (DISAGREEMENT_ANALYZER_LLM, llm_configs.py).
# Caps the worst case on the conditional-stacking critical path: the analyzer's own
# bound is UTILITY_MODEL_CONFIG's litellm timeout per attempt, which is looser than
# this, so without the wrapper a stalled call runs well past the crux's usefulness.
CRUX_SOFT_DEADLINE: int = 180

# Wall-clock cap for the AskNews summarizer invoke. The summarizer is set
# allowed_tries=1 (llm_configs.py) and wrapped in the broad elapsed-gated retry,
# which previously had no wall guard at all. Matches the summarizer's litellm
# per-request timeout (UTILITY_MODEL_CONFIG in llm_configs.py) so the per-attempt
# cap aligns with the underlying request budget;
# on breach the summarizer soft-fails to the
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
# luna is the cheapest capable tier (mini → luna 2026-08-03).
LEAKAGE_DETECTOR_MODEL: str = "openrouter/openai/gpt-5.6-luna"

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

# Outer wall-clock timeout for the full prediction-market snapshot (both LLM stages, the
# catalogue pulls, the venue fan-out and the Manifold detail enrichment). Runs inside
# asyncio.gather alongside the other research providers, so raising it adds no wall-clock time
# to the research phase — for scale, NATIVE_SEARCH_WALL_TIMEOUT is 420, Gemini 360, AskNews 300.
# 150 is the ranked pipeline's 131.5s worst case (table below) plus margin; it was 30.0 under
# the keyword/fuzzy design, which a ~36k-token ranking call and a full catalogue pull do not
# fit inside. `prediction_market.SNAPSHOT_STAGE_BUDGET_S` recomputes that worst case from these
# constants and WARNs loudly at provider init when an env override sits below it, so a stale
# `PREDICTION_MARKET_TIMEOUT=30` in a .env cannot masquerade as a generic snapshot timeout.
PREDICTION_MARKET_TIMEOUT: float = float(os.environ.get("PREDICTION_MARKET_TIMEOUT", "150.0"))

# --- Ranked market retrieval (the two LLM stages and the catalogue pull) ---
# Wall caps and backoff ladders for the snapshot's two new LLM stages, plus the bounds
# on the full Kalshi events catalogue pull. Each stage's worst case is
# ``(len(backoffs) + 1) * wall + sum(backoffs)``, and the SERIAL chain of those worst
# cases has to fit under PREDICTION_MARKET_TIMEOUT. Stages 1a and 1b run concurrently,
# so the chain takes the max of them:
#
#   | stage                                   | cap                    | worst |
#   |-----------------------------------------|------------------------|-------|
#   | 1a Kalshi catalogue (wall)              | 40.0                   |  40   |
#   | 1b query author (concurrent with 1a)    | wall 20, backoffs (1,) |  41   |
#   | 1a PredictIt dump (concurrent)          | 10 x 2 + 0.5           |  20.5 |
#   | 2  venue search                         | 10 x 2 + 0.5           |  20.5 |
#   | 2.5 manifold detail fan-out (wall)      | 10.0                   |  10   |
#   | 4  ranking                              | wall 60, backoffs ()   |  60   |
#   | total  max(41, 40, 20.5) + 20.5 + 10 + 60                       | 131.5 |
#
# The ranker gets NO retry: 36 of 36 measured calls parsed first try, a retry on a
# ~36k-token prompt is expensive latency, and the deterministic pool-order fail-open
# slate is a good fallback sitting right there. Its wall is 60 rather than the
# originally specced 45 because the prompt grew ~50% (full PredictIt universe +
# Manifold enrichment) and prefill scales with it.
MARKET_QUERY_AUTHOR_WALL_TIMEOUT: float = 20.0
MARKET_QUERY_AUTHOR_BACKOFFS: tuple[float, ...] = (1.0,)
MARKET_RANKER_WALL_TIMEOUT: float = 60.0
MARKET_RANKER_BACKOFFS: tuple[float, ...] = ()

# Kalshi catalogue pull: a WALL-CLOCK budget for the whole paginated fetch, retries
# included, so pagination can never push the snapshot past its own timeout. The
# per-page ``aiohttp.ClientTimeout`` sits under this wall, not beside it.
#
# ``KALSHI_PAGE_SLEEP_S`` = 0.25 is MEASURED, on the value itself. Six full cold pulls
# through the real ``kalshi_prefetch_events`` (2026-08-04/05, free unauthenticated GETs,
# ``scratch/market_port_2026-08-04/kalshi_page_sleep_probe.py``, receipts in that
# directory's ``qa_artifacts/``) all completed: 10,083-10,093 events over 51 pages every
# time, ZERO 429s, wall 16.90-25.37s against this 40s cap. Compare the zero-sleep baseline
# it replaced — an HTTP 429 on 2 of 4 pulls, 8-18% of the catalogue lost — and the
# 0.25 value is doing the work it was introduced for. That matters because a 429 is
# deliberately non-retryable for this venue: the pull stops, reports incomplete, and bumps
# BOTH degradation counters, so the sleep value can redden CI on a condition it causes.
#
# The wall decomposes exactly, which is what makes the headroom trustworthy rather than
# lucky: 50 sleeps x 0.25 = 12.50s fixed, plus 4.29-12.77s of actual fetch, residual
# ~0.10s. So the sleep is HALF to THREE-QUARTERS of the elapsed pull and the worst
# observed case used 63% of the wall. Only the run's first question pays it (6h TTL cache),
# and it does not move ``SNAPSHOT_STAGE_BUDGET_S`` (stage 1a already takes the max of the
# author wall, this wall, and the HTTP stage).
#
# Headroom to raise it if a 429 ever returns: 0.3 projects to ~27.8s and 0.4 to ~32.8s at
# the worst measured fetch, so 0.4 is the practical ceiling under this wall — beyond it
# ``attempt_budget`` (which doubles as the per-page HTTP timeout) starves the late pages
# and trades throttle-loss for wall-timeout-loss, bumping the same two counters. Raising
# the wall alongside is the other lever.
#
# One residual observation, benign but worth knowing before re-probing: back-to-back pulls
# 60s apart got monotonically slower (fetch 4.47 -> 8.15 -> 12.77s), and the same cadence
# at 120s gaps stayed flat (4.29 / 4.41 / 7.11s). That looks like soft rate pressure that
# drains, never a 429, and prod pulls the catalogue once per run — so it constrains probe
# design (space pulls out, or a probe's own cadence manufactures the slowdown it reports)
# rather than the constant.
# EVENT_LIMIT is a runaway guard well above the ~10.1k live open events; MAX_PAGES is
# the real bound.
KALSHI_CATALOGUE_WALL_TIMEOUT: float = 40.0
KALSHI_PAGE_SLEEP_S: float = 0.25
KALSHI_PREFETCH_EVENT_LIMIT: int = 20_000
KALSHI_PREFETCH_MAX_PAGES: int = 120

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
# to resolution IS the answer (NOT scheduled_resolution - buffer), with ALFRED
# vintages at as_of for revising series.
TS_ANCHOR_ENABLED_ENV: str = "TS_ANCHOR_ENABLED"
# Chart-image side-channel: when on (and TS_ANCHOR_ENABLED is also on), the
# provider renders a PNG of the anchor (series + P10-P90 band, sized in
# research/ts_chart.py) for single-level questions
# and stashes it per-qid; the forecaster passes it to
# each base model as a vision message. OFF everywhere until the text-vs-image
# A/B (FUTURE.md "TS anchor chart image"). Independent of TS_ANCHOR_ENABLED so
# the text anchor can ship before the (costlier, unvalidated) image does.
# NOTE: matplotlib is a dev-only dependency; under the bot workflows' --no-dev
# install, flipping this on degrades to the text-only anchor with one ERROR log.
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
# How far past an OPEN displayed edge a rendered band may sit before the magnitude backstop
# treats it as a wrong-quantity anchor, measured in multiples of the displayed span. An open
# edge means the outcome genuinely can settle beyond it, so the constraint has to loosen
# there — but treating open as "no constraint at all" disarmed the backstop entirely on the
# ~95% of numeric questions that carry two open bounds.
#
# The measured window is wide, and BOTH ends are pinned by tests so a future tweak has to
# confront the evidence. The rule compares the NEAREST BAND EDGE (not the P50) against the
# range, which is what makes the window wide: every band the anchor has actually published
# overlaps its own range, so it scores 0.00 spans outside and no tolerance below 1.0 can
# suppress it. Meanwhile the wrong-quantity shapes this exists to catch — a percent-unit band
# on a basis-point question — sit 0.63-0.73 outside, so anything at or above ~0.63 stops
# catching them (a 1.0 tolerance catches nothing, which is why the value is well under it).
# Closed edges get no tolerance at all: the outcome cannot settle past them, so the original
# zero-overlap rule stands there unchanged and this whole knob is inert.
TS_ANCHOR_OPEN_BOUND_SPAN_TOLERANCE: float = 0.25

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
