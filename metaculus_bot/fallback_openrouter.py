import asyncio
import logging
import os
import sys
from typing import Any

import openai
from forecasting_tools import GeneralLlm

from metaculus_bot.constants import (
    CREDIT_ALERT_RESUME_DATE,
    OAI_ANTH_OPENROUTER_KEY_ENV,
    OPENROUTER_API_KEY_ENV,
    credit_alerts_active,
    gemini_use_donated_openrouter_key,
)
from metaculus_bot.credit_telemetry import DonatedKeyState, classify_donated_key_state

logger: logging.Logger = logging.getLogger(__name__)


def _record_deprecation_if_matched(model: str, error_msg: str) -> bool:
    """Append to ``_DEPRECATION_ALERTS`` iff the error message looks like a model deprecation.

    Returns True iff matched (and recorded). Match is case-insensitive substring
    against ``_DEPRECATION_PATTERNS``. Designed to be called from any LLM call
    site that observes an exception — the ``FallbackOpenRouterLlm`` wrapper for
    donated-key models, ``_run_providers_parallel`` for plain-GeneralLlm research
    providers (Grok native search), etc. Idempotent within a single recording —
    every distinct error string adds an entry; cli.py only checks ``len > 0``.
    """
    msg_lower = error_msg.lower()
    if any(pattern in msg_lower for pattern in _DEPRECATION_PATTERNS):
        _DEPRECATION_ALERTS.append((model, error_msg))
        return True
    return False


def clear_deprecation_alerts() -> None:
    """Reset the alert list. Used by tests; not for production code."""
    _DEPRECATION_ALERTS.clear()


def check_deprecation_alerts_and_exit() -> None:
    """Post-submission tripwire: log loudly and ``sys.exit(1)`` if any deprecation was seen.

    Called from ``cli.py`` AFTER ``forecast_on_tournament`` / ``forecast_questions``
    completes — so every publishable question is already on Metaculus regardless
    of exit status. Returns silently when the alert list is empty.
    """
    if not _DEPRECATION_ALERTS:
        return
    banner = "=" * 78
    logger.error(banner)
    logger.error("MODEL DEPRECATION DETECTED — %d alert(s) recorded this run", len(_DEPRECATION_ALERTS))
    logger.error("OpenRouter (or another provider) returned a deprecation-shaped error for one or more")
    logger.error("models the bot called. Submission completed via fallbacks, but the model lineup needs")
    logger.error("updating. See metaculus_bot/llm_configs.py and metaculus_bot/constants.py.")
    logger.error(banner)
    for model_slug, error_msg in _DEPRECATION_ALERTS:
        logger.error("  model=%s | error=%s", model_slug, error_msg)
    logger.error(banner)
    sys.exit(1)


# Post-submission deprecation tripwire. When OpenRouter retires a model the
# bot uses (the canonical case: 2026-05-15 deprecation of x-ai/grok-4.1-fast,
# the native-search model, which silently 404'd for ~2 days), we want CI to
# turn red so the operator notices — but NOT to abort mid-run, since the
# remaining ensemble can still publish via fallbacks.
#
# Pattern: any LLM call site that observes an exception calls
# ``_record_deprecation_if_matched(model, str(exc))``. Matches are appended
# here as ``(model_slug, error_msg)`` tuples. After the bot finishes
# submitting all forecasts, ``cli.py`` calls
# ``check_deprecation_alerts_and_exit()``; if anything was recorded, it logs
# loudly and ``sys.exit(1)`` to fail the GitHub Actions job.
_DEPRECATION_ALERTS: list[tuple[str, str]] = []

# High-precision substrings (case-insensitive) that indicate model deprecation.
# Conservative set: false positives turn CI red without justification, which
# is annoying. OpenRouter's deprecation 404s consistently include both
# "deprecated" and "recommends switching to" in the message body, but we
# match either to stay robust against minor copy changes.
_DEPRECATION_PATTERNS: tuple[str, ...] = (
    "deprecated",
    "recommends switching to",
)


# Module-level diagnostic counter for the allowed-providers-404 SUBSET of
# donated->personal fallbacks. Incremented every time the donated key returns a
# "no allowed providers / 404" error and we successfully fall back to the general
# key. This is NOT the alerting input — cli.py folds ``_generic_key_fallback_count``
# (the all-causes total) into ``alertable``; adding this 404 count too would
# double-count events already inside that total. cli.py reads this via
# ``get_donated_404_fallback_count`` only to break it out in the end-of-run log
# line ("... of which donated_404=N"), so a stale allowed-providers list upstream
# is visible without losing the run.
_donated_404_fallback_count: int = 0


def get_donated_404_fallback_count() -> int:
    """Read the module-level counter for donated-key 404 fallback events."""
    return _donated_404_fallback_count


def reset_donated_404_fallback_count() -> None:
    """Reset the counter to zero. Used by tests; not for production code."""
    global _donated_404_fallback_count
    _donated_404_fallback_count = 0


# Module-level counter for EVERY successful donated->personal key fallback,
# regardless of cause (401/402/429/guardrail/404). Distinct from
# ``_donated_404_fallback_count`` (which counts only the allowed-providers 404
# subset). The operator pays for every personal-key fallback, so we want a loud,
# auditable signal whenever the donated key was supposed to cover a call but
# didn't — cli.py folds this into the end-of-run alert so a run that quietly
# leaked spend to the personal key still turns CI red.
_generic_key_fallback_count: int = 0


def get_generic_key_fallback_count() -> int:
    """Read the module-level counter for donated->personal key fallback events (all causes)."""
    return _generic_key_fallback_count


def reset_generic_key_fallback_count() -> None:
    """Reset the counter to zero. Used by tests; not for production code."""
    global _generic_key_fallback_count
    _generic_key_fallback_count = 0


# Module-level counter for the CREDIT-caused (402 / payment required / insufficient
# credit) SUBSET of donated->personal fallbacks — the same subset relationship
# ``_donated_404_fallback_count`` has to the generic total, so it is likewise NOT
# added to ``alertable`` on its own. cli.py subtracts it from the generic total
# while credit alerting is suppressed (constants.credit_alerts_active), because an
# empty donated key is an expected condition during that window rather than
# breakage; after the resume date the subtraction stops and behavior is exactly
# what it was before. Every other cause (401/404/429/guardrail) stays alertable.
_credit_key_fallback_count: int = 0


def get_credit_key_fallback_count() -> int:
    """Read the module-level counter for credit-caused donated->personal fallbacks."""
    return _credit_key_fallback_count


def reset_credit_key_fallback_count() -> None:
    """Reset the counter to zero. Used by tests; not for production code."""
    global _credit_key_fallback_count
    _credit_key_fallback_count = 0


# Providers covered by the Metaculus-donated OpenRouter key
# (``OAI_ANTH_OPENROUTER_KEY``). The donated key has server-side allowed-
# providers preferences locked to this set. Models routed through any other
# provider will 404 on the donated key, so we only prefer the donated key for
# these. The env var name stays ``OAI_ANTH_OPENROUTER_KEY`` for backward
# compatibility with the operator's GitHub secret — adding ``google`` here
# does not require changing the secret name.
DONATED_KEY_PROVIDERS: frozenset[str] = frozenset({"openai", "anthropic", "google"})

# Google models that must NOT route through the donated key even when the
# donated-Gemini toggle is ON. Metaculus's donated OpenRouter account serves
# these via a FREE-TIER Google AI Studio BYOK key, and gemini-3.x-pro has no
# Google free tier (quota 0) → every donated-key call 429s (is_byok:true) and
# falls back to the personal key. Routing them straight to the personal key
# avoids the wasted donated→429→personal round-trip on every call AND the
# CI-red fallback-counter bump it causes (gemini-3.1-pro-preview is a core
# forecaster that runs on every question). Matched by prefix (startswith) so the
# bare GA slug and every suffixed variant (-preview, -preview-customtools,
# OpenRouter :free/route suffixes) are all covered.
#
# TODO(gemini-3.1-pro-donated): gemini-3.1-pro SHOULD work on the donated key.
# The ONLY blocker is Metaculus's free-tier Google BYOK routing. Once Metaculus
# enables Cloud billing on that BYOK key (Tier 1), removes the Google BYOK
# integration so it uses native OpenRouter Google credits, or disables "Always
# use for this provider" on it — REMOVE the matching entry here so Pro rejoins
# the donated subsidy. Re-verify with one live call: the 429 should no longer
# carry is_byok:true + free-tier limit 0. See FUTURE.md "Gemini on the donated
# OpenRouter key".
DONATED_KEY_BLOCKED_GOOGLE_MODELS: frozenset[str] = frozenset({"gemini-3.1-pro"})


def should_route_via_donated_key(model: str) -> bool:
    """Whether ``model`` should prefer the Metaculus-donated key (with paid-key fallback).

    Matches OpenRouter model slugs of the form ``openrouter/<provider>/<model>``
    against ``DONATED_KEY_PROVIDERS``. Returns False for non-OpenRouter slugs
    (e.g. ``perplexity/sonar``) and unrecognized providers (e.g. ``x-ai`` for Grok).

    Special case: Google routing is gated on ``GEMINI_USE_DONATED_OPENROUTER_KEY``,
    which defaults to ON (see ``gemini_use_donated_openrouter_key``). After
    Metaculus raised the Google rate limits (2026-06-16), the donated key serves
    most Gemini models (e.g. ``gemini-3.5-flash``, ``gemini-3.1-flash-lite``).

    EXCEPTION: models matching ``DONATED_KEY_BLOCKED_GOOGLE_MODELS``
    (``gemini-3.1-pro``) are pinned to the personal key — no donated attempt, no
    429, no fallback-counter bump. They run through a free-tier Google AI Studio
    BYOK key on the donated account that has no Pro free tier (limit 0 → 429), so
    a donated attempt would always fail over to personal anyway. The pin is a
    temporary workaround; see the ``TODO(gemini-3.1-pro-donated)`` tag on that
    constant and FUTURE.md — remove the entry once Metaculus fixes the BYOK
    routing. Set the env var to a false-y value to force personal-key-only routing
    for ALL Gemini.
    """
    if not isinstance(model, str):
        return False
    if not model.startswith("openrouter/"):
        return False
    parts = model.split("/")
    if len(parts) < 2:
        return False
    provider = parts[1]
    if provider not in DONATED_KEY_PROVIDERS:
        return False
    if provider == "google":
        if not gemini_use_donated_openrouter_key():
            return False
        model_name = "/".join(parts[2:])
        if any(model_name.startswith(blocked) for blocked in DONATED_KEY_BLOCKED_GOOGLE_MODELS):
            return False
    return True


# OpenRouter reports a breached per-key SPEND CAP as HTTP 403 with this phrase —
# not the 402 its own error docs document. Verified against the 2026-07-26
# production failure, where litellm surfaced it as a bare ``APIError`` (litellm has
# no 403 branch for OpenRouter) whose body carried ``"code":403``. The old negative
# rule below vetoed any message containing "403", so the operator's funded personal
# key was never tried and two of three forecasters plus most of the research stack
# died on a key that was merely empty.
#
# The full phrase is load-bearing. The shorter "limit exceeded" is a substring of
# "rate limit exceeded: free-models-per-day", so it would classify every 429 as an
# empty wallet and silently exempt real rate-limit breakage from alerting for the
# whole suppression window.
KEY_LIMIT_EXCEEDED_CUE = "key limit exceeded"

# Unambiguous "this key is out of money" wording. Safe to match anywhere in the
# message — these are English phrases, not status digits.
_EXPLICIT_CREDIT_PHRASES: tuple[str, ...] = (
    KEY_LIMIT_EXCEEDED_CUE,
    "payment required",
    "insufficient credit",
    "out of credits",
    "insufficient funds",
)

# Signals that the body is a content-moderation refusal rather than a billing
# problem. litellm builds the message as ``APIError: {provider} - {raw body}``, and
# an OpenRouter moderation 403 body carries ``flagged_input``: up to ~100 characters
# of OUR OWN PROMPT replayed back. A forecasting prompt full of dollar figures and
# bill numbers can easily contain the token "402", which would otherwise read as an
# empty wallet — billing the personal key for a call that will refuse again AND
# exempting a real moderation block from alerting.
#
# Word cues only, deliberately: a genuine 402 links to a key hash that has roughly
# a 1-in-11 chance of containing the substring "403", and reading that as moderation
# would break the long-standing 402 fallback.
_MODERATION_CUES: tuple[str, ...] = ("moderation", "forbidden", "flagged_input", "flagged for")


def _is_credit_message(lowercased_msg: str) -> bool:
    """Whether an already-lowercased error message reads as "this key is out of money".

    Single source of truth for the credit-cause classification: the retry
    decision in ``should_retry_with_general_key`` and the credit-subset counter
    in ``FallbackOpenRouterLlm.invoke`` both go through here, so a text-cue edit
    can't make the two disagree about which fallbacks were credit-caused.
    """
    if any(phrase in lowercased_msg for phrase in _EXPLICIT_CREDIT_PHRASES):
        return True
    # Bare status digits are the weak cue — trust them only when nothing says the
    # body is a moderation refusal replaying our own prompt text back at us.
    return "402" in lowercased_msg and not any(cue in lowercased_msg for cue in _MODERATION_CUES)


# Textual cues that stay live regardless of the reported status: English wording, not
# status digits, so they carry no key-hash / prompt-echo risk. They are what classifies
# a statusless exception (a plain ``Exception("401 Unauthorized")``, a non-litellm
# caller) and a provider that words the failure without a recognizable status.
_RATE_LIMIT_TEXT_CUES: tuple[str, ...] = ("too many requests", "rate limit", "rate-limited upstream")
_BAD_CREDENTIAL_TEXT_CUES: tuple[str, ...] = ("unauthorized", "invalid api key", "disabled api key")


def _llm_status_code(exc: Exception) -> int | None:
    """The HTTP status the provider reported, or ``None`` when the exception carries none.

    Scoped to ``openai.APIError`` — the common root of every litellm exception
    (``litellm.exceptions.APIError.__mro__[1] is openai.APIError``) — so a same-named
    attribute on some unrelated exception can never be read as a provider status.
    ``requests``-shaped errors keep their status on ``.response`` and so report ``None``
    here, which is what leaves the textual cues in charge for them.
    """
    if not isinstance(exc, openai.APIError):
        return None
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _is_status(reported_status: int | None, code: int, lowercased_msg: str) -> bool:
    """Whether the failure is HTTP ``code``, preferring the reported status over digits.

    When the provider reported a status, that integer is the ONLY numeric evidence
    consulted. The message is not: litellm formats it as ``APIError: {provider} - {raw
    body}``, and an OpenRouter body carries a 64-hex key hash (~8.8% chance of
    containing one of 401/402/403/429/502/503) plus, on a moderation refusal, up to
    ~100 characters of our own prompt in ``flagged_input``. Matching digits there reads
    coincidences as statuses in both directions — a stray "429" sends a moderation 403
    to the paid key for a call that will refuse again.

    With no status reported, fall back to the substring so plain
    ``Exception("401 Unauthorized")`` callers classify exactly as they always have.
    """
    if reported_status is not None:
        return reported_status == code
    return str(code) in lowercased_msg


def _is_credit_failure(reported_status: int | None, lowercased_msg: str) -> bool:
    """Status-aware view of :func:`_is_credit_message` for the routing decision.

    Two signals classify a failure whose status the provider actually reported: a 402,
    and explicit credit wording. Either alone is sufficient.

    * **402 is unambiguous.** "Payment Required" has no second meaning, and OpenRouter
      reports refusals as 403, so a reported 402 outranks any English word that happens
      to sit in the body. The failure asymmetry agrees: reading a real 402 as a refusal
      strands the ensemble on a dry key, the production bug this exists to fix, while
      reading a hypothetical 402-shaped refusal as credit costs one paid call that
      refuses again.
    * **Explicit wording catches the spend-cap 403**, which OpenRouter returns as "Key
      limit exceeded" despite documenting credit exhaustion as 402. That body can also
      carry generic HTTP "Forbidden" boilerplate, so the wording must win there too.

    No moderation veto is needed here, and adding one would be dead code: with the status
    in hand, a moderation 403 is rejected because 403 is not 402 and carries no credit
    phrase — the poisoning case is closed by READING the status rather than by vetoing on
    words. The veto still earns its place in ``_is_credit_message``, which has only bare
    digits to go on; a statusless exception defers wholly to that function, keeping it the
    single source of truth for text-only classification.

    Nothing here reads a live balance — ``status_code`` is an int already on the
    exception. The ``/auth/key`` probe belongs to ``is_suppressible_credit_error`` and the
    ALERTING decision, never to routing.
    """
    if reported_status is None:
        return _is_credit_message(lowercased_msg)
    return reported_status == 402 or any(phrase in lowercased_msg for phrase in _EXPLICIT_CREDIT_PHRASES)


def is_credit_caused_error(exc: Exception) -> bool:
    """Whether ``exc`` is a credit shortfall (402, spend-cap 403, insufficient credit).

    The public form of ``_is_credit_failure``, so the routing decision in
    ``should_retry_with_general_key`` and the alerting decision in
    ``is_suppressible_credit_error`` answer "was this about money?" the same way. They
    used to disagree: routing became status-aware while this stayed text-only, so a
    terse reported-402 (``APIError(status_code=402, message="wallet empty")``) fell
    back to the paid key without being credit-classified — reddening CI on exactly
    the expected empty wallet the suppression window exists for.
    """
    return _is_credit_failure(_llm_status_code(exc), str(exc).lower())


def is_suppressible_credit_error(exc: Exception) -> bool:
    """Whether ``exc`` is the EXPECTED drained donated key, not some other breakage.

    Only this narrower class is exempt from CI alerting during the suppression
    window (``constants.credit_alerts_active``). The distinction matters because a
    donated key Metaculus revoked, or re-capped to zero, returns the SAME
    "Key limit exceeded" text as one that simply spent its allocation — so the text
    cue alone would have exempted genuine breakage from alerting for six weeks. We
    ask OpenRouter's free, read-only ``/auth/key`` endpoint instead (once per run,
    cached), and every inconclusive answer stays alertable.

    The 402 / insufficient-credit family deliberately skips the probe: it is
    unambiguous, it predates the discriminator, and keeping it probe-free means an
    unreachable balance endpoint cannot change long-standing behavior.

    Note this governs ALERTING only. Fallback ROUTING never consults the probe (see
    ``should_retry_with_general_key``) so a stale or cached balance read can never
    strand the ensemble on a dry key — the failure mode this whole change exists to
    fix.
    """
    if not is_credit_caused_error(exc):
        return False
    if KEY_LIMIT_EXCEEDED_CUE not in str(exc).lower():
        return True
    return classify_donated_key_state() is DonatedKeyState.DRAINED


def should_retry_with_general_key(exc: Exception) -> bool:
    """
    Decide whether a failure likely indicates a key-scoped issue where falling back is appropriate.

    Triggers fallback on:
    - 429 Too Many Requests (rate limit) — donated and personal keys have
      independent BYOK quotas per-provider, so a 429 on the primary key does
      NOT imply the secondary is also throttled. Fall back immediately; the
      SDK already retried internally before raising, so no wrapper-level retry.
    - 401 Unauthorized (invalid/disabled key),
    - 402 Payment Required (insufficient credits),
    - 404 with "no allowed providers" — donated key has server-side
      allowed-providers preferences; a 404 there means the donated key cannot
      route this model, but the general key (no preferences) can. Treated as
      key-scoped so callers fall through to the secondary key.
    - 403 carrying spend-cap wording ("Key limit exceeded"), which is how
      OpenRouter reports a drained per-key budget despite documenting credit
      exhaustion as 402. Classified by ``_is_credit_failure`` UPSTREAM of the 403
      veto below — that ordering is load-bearing.
    - Common text cues for these scenarios.

    Avoids fallback on:
    - 403 Forbidden without credit wording (moderation / permission block, both
      keys would refuse),
    - 502/503 upstream/provider outages (infrastructure, not key-scoped),
    - Plain 404 (missing model), which is why the 404 family is classified by
      TEXT rather than status: the same status covers both a route problem the
      paid key can fix and a model that simply does not exist.

    Numeric detection reads the status the provider reported
    (``_llm_status_code`` / ``_is_status``), never digits in the message. litellm
    formats the message as ``APIError: {provider} - {raw body}``, and an OpenRouter
    body carries a 64-hex key hash plus, on a moderation refusal, up to ~100
    characters of our own prompt — either can contain a number that was never a
    status. Statusless exceptions still classify on text, unchanged.

    Note: direct google-genai SDK 429s (google.genai.errors.ClientError with
    code=429) are out of scope for this wrapper — they don't flow through
    OpenRouter. The gemini search provider handles those separately.
    """
    msg_raw = str(exc)
    # Deprecation tripwire: record the alert before classifying retry behavior.
    # The match is conservative (see _DEPRECATION_PATTERNS) and only records;
    # the actual sys.exit happens later via check_deprecation_alerts_and_exit.
    # We don't have the model slug here — caller-supplied recording in the
    # wrapper's invoke() carries the slug; this is a safety net for any other
    # call site that routes through this predicate.
    _record_deprecation_if_matched("<unknown>", msg_raw)

    # 429 rate-limit: BYOK quotas are per-key, so primary being throttled does
    # NOT imply secondary is also throttled. Fall back immediately — litellm
    # already exhausted its internal retry budget before raising.
    import litellm.exceptions  # noqa: PLC0415  # function-scoped: avoids formatter stripping unused top-level import

    if isinstance(exc, litellm.exceptions.RateLimitError):
        return True

    msg = msg_raw.lower()
    # Authoritative for every NUMERIC branch below; None for statusless exceptions,
    # which then classify on message text exactly as they always have.
    status = _llm_status_code(exc)

    # Belt-and-suspenders detection for 429 edge cases where litellm doesn't raise the
    # typed exception (e.g., class drift, non-standard wrapping).
    if _is_status(status, 429, msg) or any(cue in msg for cue in _RATE_LIMIT_TEXT_CUES):
        return True

    # Positive signals: credentials/credits
    if _is_status(status, 401, msg) or any(cue in msg for cue in _BAD_CREDENTIAL_TEXT_CUES):
        return True
    if _is_credit_failure(status, msg):
        return True
    # Donated-key allowed-providers quirk: the donated key has server-side
    # provider preferences; a model only available via a non-allowed provider
    # returns 404 with "no allowed providers". The general key has no such
    # restriction and routes the same model fine.
    if "no allowed providers" in msg:
        return True
    # Donated-key data-policy / guardrail block (added 2026-05-17 during native
    # search migration). When OpenAI native search is invoked on the donated
    # key, OpenRouter returns 404 with text like:
    #   "No endpoints available matching your guardrail restrictions and data
    #   policy. Configure: https://openrouter.ai/settings/privacy"
    # The donated key has data-collection guardrails set by Metaculus that
    # exclude OpenAI's native-search endpoint. The personal key has no such
    # restriction. Treat as key-scoped so callers fall through to the
    # secondary key automatically — see FUTURE.md "Resolve
    # OAI_ANTH_OPENROUTER_KEY data-policy block".
    if "guardrail" in msg or "data policy" in msg:
        return True

    # Negative signals: do not swap keys for these. Reached only after the credit
    # check above, so a spend-cap 403 has already fallen back by here and what
    # remains is a genuine content/permission refusal the paid key would also get.
    if _is_status(status, 403, msg) or any(cue in msg for cue in _MODERATION_CUES):
        return False
    if _is_status(status, 502, msg) or "bad gateway" in msg:
        return False
    if _is_status(status, 503, msg) or "service unavailable" in msg:
        return False

    # Default: be conservative and do not fallback when unsure
    return False


def _is_donated_404(exc: Exception) -> bool:
    """Whether this exception is the donated-key allowed-providers 404 specifically.

    Used to bump the alerting counter only on this fallback class — not for
    401/402 (those are credit/key issues, not the allowed-providers quirk).
    """
    return "no allowed providers" in str(exc).lower()


def _fallback_alert_note(*, suppressible: bool) -> str:
    """The "what happens to the exit code" clause of the paid-fallback WARNING.

    A suppressible credit fallback during the window does NOT redden CI (see
    ``constants.credit_alerts_active``), so saying it will would mislead whoever
    greps this line. Every other cause still exits non-zero.

    ``suppressible`` is the caller's already-computed
    ``is_suppressible_credit_error`` verdict — it has paid for the donated-key
    probe, and re-deriving it here would either duplicate that HTTP call or key the
    note on the text cue alone and promise a green run a revoked key won't deliver.
    """
    if suppressible and not credit_alerts_active():
        return (
            "Cause is a credit shortfall, so it is NOT counted as alertable until "
            f"{CREDIT_ALERT_RESUME_DATE.isoformat()} (operator is self-funding the season)."
        )
    return "Run will complete, then exit non-zero to alert."


async def record_donated_key_fallback(model: str, exc: Exception) -> None:
    """Count and log ONE donated -> personal-key fallback that is about to happen.

    The shared accounting seam for every donated-first call path: the
    ``FallbackOpenRouterLlm.invoke`` wrapper and gap-fill v2's hand-rolled
    raw-litellm retry in ``research/agentic/llm.py``, which shared the retry
    PREDICATE but not the accounting and so fell over silently — no counter, no
    ``PAID PERSONAL-KEY FALLBACK`` warning, no line in the end-of-run summary —
    despite firing on every question in all four prod workflows. That was the
    ``TODO(unify-fallback-routing)``.

    Every successful donated -> personal fallback means a paid personal-key call
    happened where the free donated key was expected to cover it, so all of them
    are counted and logged loudly: silent personal-key spend must not accumulate
    unnoticed.

    Counting invariant (see the CLAUDE.md credit-suppression note): each event is
    counted exactly ONCE in the generic total, and at most one subset counter
    (credit-caused, or the 404 "no allowed providers" quirk) also claims it. That
    is what lets cli.py compute ``alertable`` as "generic adds, one subset
    subtracts" without drift. Call this only when the fallback will actually be
    attempted — a rejected fallback bills nothing and must not count.

    Async because the probe below must leave the event loop; the counting must not.
    """
    # Only the EXPECTED drained-donated-key subset is exempt from alerting. A key that
    # was revoked or re-capped to zero produces identical "Key limit exceeded" text, so
    # this asks OpenRouter rather than trusting the cue — otherwise the suppression
    # window would have hidden genuine breakage for six weeks.
    #
    # Threaded because on the spend-cap 403 path it reaches
    # ``credit_telemetry.classify_donated_key_state``, which does blocking httpx. Called
    # inline from these coroutines it stalled EVERY concurrently in-flight forecaster and
    # research task for up to ``DONATED_KEY_PROBE_TIMEOUT_S`` (5s), not just the call that
    # hit the 403, eating into per-question soft deadlines. Probing first also keeps the
    # accounting below free of any await.
    suppressible = await asyncio.to_thread(is_suppressible_credit_error, exc)

    # NO await from here down, so the whole accounting block runs to completion on the
    # event loop. That is load-bearing, not incidental: ``+=`` on a module global compiles
    # to LOAD_GLOBAL / INPLACE_ADD / STORE_GLOBAL and is interruptible between bytecodes,
    # so threading this function as a whole (rather than just the probe) would let N
    # forecasters failing on one dry key — the exact 2026-07-26 shape — race the
    # increment, undercount the generic total, and take a degraded run GREEN. That is the
    # failure this whole change exists to prevent.
    global _generic_key_fallback_count
    _generic_key_fallback_count += 1
    if suppressible:
        global _credit_key_fallback_count
        _credit_key_fallback_count += 1
    if _is_donated_404(exc):
        global _donated_404_fallback_count
        _donated_404_fallback_count += 1
        logger.warning(
            "Donated OpenRouter key returned 404 'no allowed providers' for model=%s; "
            "falling back to general (paid personal) key. This means the donated key's "
            "server-side allowed-providers list does not cover this model's upstream "
            "provider. Run will complete, then exit non-zero to alert. error=%s: %s",
            model,
            type(exc).__name__,
            exc,
        )
    else:
        logger.warning(
            "PAID PERSONAL-KEY FALLBACK: donated OpenRouter key failed for model=%s, so this "
            "call billed to the personal OPENROUTER_API_KEY instead of the free donated key. "
            "%s error=%s: %s",
            model,
            _fallback_alert_note(suppressible=suppressible),
            type(exc).__name__,
            exc,
        )


class FallbackOpenRouterLlm(GeneralLlm):
    """A GeneralLlm wrapper that prefers a Metaculus-donated OpenRouter key, falling back to the
    operator's general key on credential/credit/allowed-providers errors. Used for models routed
    through providers covered by the donated key (see ``DONATED_KEY_PROVIDERS``).
    """

    def __init__(
        self,
        *,
        model: str,
        primary_api_key: str | None,
        secondary_api_key: str | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, api_key=primary_api_key, **kwargs)
        self._secondary_llm: GeneralLlm | None = (
            GeneralLlm(model=model, api_key=secondary_api_key, **kwargs) if secondary_api_key else None
        )

    async def invoke(self, prompt: Any, system_prompt: str | None = None) -> str:  # type: ignore[override]
        # system_prompt widened in to match 0.2.92's GeneralLlm.invoke signature
        # (invoke(prompt, system_prompt=None)); threaded through both key paths so a
        # system prompt survives a donated->personal fallback unchanged.
        try:
            return await self._invoke_once_using_primary(prompt, system_prompt)
        except Exception as e:
            # Re-record with the actual model slug. should_retry_with_general_key
            # also calls the matcher with "<unknown>" — duplicates are fine since
            # cli.py only checks list non-empty for the exit decision, but the
            # log is clearer with the slug.
            _record_deprecation_if_matched(self.model, str(e))
            if self._secondary_llm is not None and should_retry_with_general_key(e):
                # ASYNC120 (both awaits): a checkpoint inside `except` can drop the
                # active exception if the task is cancelled mid-await. That's the
                # correct behavior here — on success we return the secondary's
                # output; on cancellation the secondary is cancelled too. The
                # primary's exception is intentionally discarded because the
                # caller asked for a fallback, not a re-raise.
                await record_donated_key_fallback(self.model, e)  # noqa: ASYNC120
                return await self._invoke_once_using_secondary(prompt, system_prompt)  # noqa: ASYNC120
            raise

    async def _invoke_once_using_primary(self, prompt: Any, system_prompt: str | None = None) -> str:
        return await super().invoke(prompt, system_prompt)

    async def _invoke_once_using_secondary(self, prompt: Any, system_prompt: str | None = None) -> str:
        if self._secondary_llm is None:
            raise RuntimeError("No secondary key configured for fallback")
        return await self._secondary_llm.invoke(prompt, system_prompt)


def build_llm_with_openrouter_fallback(model: str, **kwargs: Any) -> GeneralLlm:
    """
    Construct a GeneralLlm that automatically falls back from the Metaculus-donated OpenRouter
    key to the operator's general key for providers covered by the donated key (see
    ``DONATED_KEY_PROVIDERS``). For other models, returns a plain GeneralLlm.
    """
    if should_route_via_donated_key(model):
        special_key = os.getenv(OAI_ANTH_OPENROUTER_KEY_ENV)
        general_key = os.getenv(OPENROUTER_API_KEY_ENV)

        # If both keys exist and are distinct, use the fallback wrapper
        if special_key and general_key and special_key != general_key:
            return FallbackOpenRouterLlm(
                model=model,
                primary_api_key=special_key,
                secondary_api_key=general_key,
                **kwargs,
            )

        # Else fall back to whichever key is available (no runtime fallback possible)
        api_key = special_key or general_key
        return GeneralLlm(model=model, api_key=api_key, **kwargs)

    # OpenRouter models that bypass the donated wrapper: plain GeneralLlm.
    # Covers (a) providers not in DONATED_KEY_PROVIDERS (x-ai, qwen, etc.),
    # (b) Google when GEMINI_USE_DONATED_OPENROUTER_KEY is explicitly off (the
    # default is now ON), and (c) blocklisted Google models
    # (DONATED_KEY_BLOCKED_GOOGLE_MODELS, e.g. gemini-3.1-pro) which are pinned to
    # the personal key even when the toggle is ON.
    # No api_key passed — litellm picks up OPENROUTER_API_KEY from env. This
    # mirrors how Grok-via-OpenRouter has always worked in production.
    return GeneralLlm(model=model, **kwargs)
