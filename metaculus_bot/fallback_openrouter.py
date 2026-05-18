import logging
import os
import sys
from typing import Any

from forecasting_tools import GeneralLlm

from metaculus_bot.constants import gemini_use_donated_openrouter_key

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


# Module-level alerting counter. Incremented every time the donated key returns
# a "no allowed providers / 404" error and we successfully fall back to the
# general key. cli.py reads this via ``get_donated_404_fallback_count`` and
# folds it into ``alertable_count`` so a run that hit this branch completes its
# submissions but exits non-zero, prompting GitHub Actions to email the operator.
# The donated key has server-side allowed-providers preferences set by
# Metaculus; if that list ever changes upstream (or a new model isn't covered),
# we want to know without losing the run.
_donated_404_fallback_count: int = 0


def get_donated_404_fallback_count() -> int:
    """Read the module-level counter for donated-key 404 fallback events."""
    return _donated_404_fallback_count


def reset_donated_404_fallback_count() -> None:
    """Reset the counter to zero. Used by tests; not for production code."""
    global _donated_404_fallback_count
    _donated_404_fallback_count = 0


# Providers covered by the Metaculus-donated OpenRouter key
# (``OAI_ANTH_OPENROUTER_KEY``). The donated key has server-side allowed-
# providers preferences locked to this set. Models routed through any other
# provider will 404 on the donated key, so we only prefer the donated key for
# these. The env var name stays ``OAI_ANTH_OPENROUTER_KEY`` for backward
# compatibility with the operator's GitHub secret — adding ``google`` here
# does not require changing the secret name.
DONATED_KEY_PROVIDERS: frozenset[str] = frozenset({"openai", "anthropic", "google"})


def should_route_via_donated_key(model: str) -> bool:
    """Whether ``model`` should prefer the Metaculus-donated key (with paid-key fallback).

    Matches OpenRouter model slugs of the form ``openrouter/<provider>/<model>``
    against ``DONATED_KEY_PROVIDERS``. Returns False for non-OpenRouter slugs
    (e.g. ``perplexity/sonar``) and unrecognized providers (e.g. ``x-ai`` for Grok).

    Special case: Google routing is gated on ``GEMINI_USE_DONATED_OPENROUTER_KEY``.
    When the donated-key Google route is flaky we want OpenRouter Gemini calls to
    flow through the operator's personal ``OPENROUTER_API_KEY`` only — no donated-
    key preference, no paid-key fallback. Toggle is off by default.
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
    if provider == "google" and not gemini_use_donated_openrouter_key():
        return False
    return True


def should_retry_with_general_key(exc: Exception) -> bool:
    """
    Decide whether a failure likely indicates a key-scoped issue where falling back is appropriate.

    Triggers fallback on:
    - 401 Unauthorized (invalid/disabled key),
    - 402 Payment Required (insufficient credits),
    - 404 with "no allowed providers" — donated key has server-side
      allowed-providers preferences; a 404 there means the donated key cannot
      route this model, but the general key (no preferences) can. Treated as
      key-scoped so callers fall through to the secondary key.
    - Common text cues for these scenarios.

    Avoids fallback on:
    - Plain 403 Forbidden (moderation/blocked, both keys would refuse),
    - 429 Too Many Requests (rate limit, both keys would hit it),
    - 502/503 upstream/provider outages.
    """
    msg_raw = str(exc)
    # Deprecation tripwire: record the alert before classifying retry behavior.
    # The match is conservative (see _DEPRECATION_PATTERNS) and only records;
    # the actual sys.exit happens later via check_deprecation_alerts_and_exit.
    # We don't have the model slug here — caller-supplied recording in the
    # wrapper's invoke() carries the slug; this is a safety net for any other
    # call site that routes through this predicate.
    _record_deprecation_if_matched("<unknown>", msg_raw)
    msg = msg_raw.lower()
    # Positive signals: credentials/credits
    if "401" in msg or "unauthorized" in msg or "invalid api key" in msg or "disabled api key" in msg:
        return True
    if (
        "402" in msg
        or "payment required" in msg
        or "insufficient credit" in msg
        or "out of credits" in msg
        or "insufficient funds" in msg
    ):
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

    # Negative signals: do not swap keys for these
    if "403" in msg or "forbidden" in msg or "moderation" in msg:
        return False
    if "429" in msg or "too many requests" in msg or "rate limit" in msg:
        return False
    if "502" in msg or "bad gateway" in msg:
        return False
    if "503" in msg or "service unavailable" in msg:
        return False

    # Default: be conservative and do not fallback when unsure
    return False


def _is_donated_404(exc: Exception) -> bool:
    """Whether this exception is the donated-key allowed-providers 404 specifically.

    Used to bump the alerting counter only on this fallback class — not for
    401/402 (those are credit/key issues, not the allowed-providers quirk).
    """
    return "no allowed providers" in str(exc).lower()


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
        self._has_warned_once: bool = False

    async def invoke(self, prompt: Any) -> str:  # type: ignore[override]
        try:
            return await self._invoke_once_using_primary(prompt)
        except Exception as e:
            # Re-record with the actual model slug. should_retry_with_general_key
            # also calls the matcher with "<unknown>" — duplicates are fine since
            # cli.py only checks list non-empty for the exit decision, but the
            # log is clearer with the slug.
            _record_deprecation_if_matched(self.model, str(e))
            if self._secondary_llm is not None and should_retry_with_general_key(e):
                # Donated-key allowed-providers 404 is its own bucket: count it
                # so cli.py can exit non-zero (alerting the operator) even
                # though the run completes successfully via the secondary key.
                if _is_donated_404(e):
                    global _donated_404_fallback_count
                    _donated_404_fallback_count += 1
                    logger.warning(
                        "Donated OpenRouter key returned 404 'no allowed providers' for model=%s; "
                        "falling back to general key. This means the donated key's server-side "
                        "allowed-providers list does not cover this model's upstream provider. "
                        "Run will complete, then exit non-zero to alert. error=%s: %s",
                        self.model,
                        type(e).__name__,
                        e,
                    )
                elif not self._has_warned_once:
                    logger.warning(
                        "Primary OpenRouter key failed with credential/credit error; falling back to generic key. "
                        "model=%s; error=%s: %s",
                        self.model,
                        type(e).__name__,
                        e,
                    )
                    self._has_warned_once = True
                # ASYNC120: a checkpoint inside `except` can drop the active
                # exception if the task is cancelled mid-await. That's the
                # correct behavior here — on success we return the secondary's
                # output; on cancellation the secondary is cancelled too. The
                # primary's exception is intentionally discarded because the
                # caller asked for a fallback, not a re-raise.
                return await self._invoke_once_using_secondary(prompt)  # noqa: ASYNC120
            raise

    async def _invoke_once_using_primary(self, prompt: Any) -> str:
        return await super().invoke(prompt)

    async def _invoke_once_using_secondary(self, prompt: Any) -> str:
        if self._secondary_llm is None:
            raise RuntimeError("No secondary key configured for fallback")
        return await self._secondary_llm.invoke(prompt)


def build_llm_with_openrouter_fallback(model: str, **kwargs: Any) -> GeneralLlm:
    """
    Construct a GeneralLlm that automatically falls back from the Metaculus-donated OpenRouter
    key to the operator's general key for providers covered by the donated key (see
    ``DONATED_KEY_PROVIDERS``). For other models, returns a plain GeneralLlm.
    """
    if should_route_via_donated_key(model):
        special_key = os.getenv("OAI_ANTH_OPENROUTER_KEY")
        general_key = os.getenv("OPENROUTER_API_KEY")

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
    # Covers (a) providers not in DONATED_KEY_PROVIDERS (x-ai, qwen, etc.) and
    # (b) Google when GEMINI_USE_DONATED_OPENROUTER_KEY is off (the default).
    # No api_key passed — litellm picks up OPENROUTER_API_KEY from env. This
    # mirrors how Grok-via-OpenRouter has always worked in production.
    return GeneralLlm(model=model, **kwargs)
