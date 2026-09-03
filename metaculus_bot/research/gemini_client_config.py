"""Shared google-genai HTTP configuration for the two native Gemini call sites.

The grounded-search provider (``research/gemini_search.py``) and gap-fill v2's
``read_document`` backend (``research/agentic/tool_backends.py``) are the only places that
call Google's API directly rather than through OpenRouter, and both need the same two
things off ``HttpOptions``: a client-side timeout, and a retry ladder. The SDK ships
NEITHER by default — ``google.genai._api_client.retry_args(None)`` returns
``stop_after_attempt(1)``, so a bare client never retries anything, which is why two
production document reads died outright on a ``503 UNAVAILABLE`` that one retry would
have absorbed.

One builder so those two call sites cannot drift apart on which statuses are worth a
second request. The timeout and attempt count stay with each CALLER, because each is
sized against a different outer deadline (see the arithmetic in ``constants.py`` beside
``GEMINI_SEARCH_HTTP_TIMEOUT_MS`` and in ``tool_backends.py`` beside
``_READ_DOCUMENT_HTTP_TIMEOUT_MS``).
"""

from __future__ import annotations

from google.genai import types as genai_types

# The transient classes worth a second request: rate limiting plus the 5xx family the
# Gemini endpoint returns when a backend is briefly unavailable. No other 4xx belongs
# here — a 400 bad request, a 403 key rejection or a 404 wrong-model-id returns the same
# answer however many times we ask. Note the SDK ALSO retries httpx timeout and connect
# errors unconditionally (``retry_args`` builds that into its ``retry_if_exception`` and
# offers no knob), which is why each caller must size its per-attempt timeout against its
# own worst case rather than assuming only fast failures are retried.
GEMINI_RETRY_HTTP_STATUS_CODES: tuple[int, ...] = (429, 500, 502, 503, 504)
GEMINI_RETRY_INITIAL_DELAY_S: float = 1.0
GEMINI_RETRY_MAX_DELAY_S: float = 8.0
# The SDK leaves tenacity's jitter at its 1.0 default, so each backoff sleep is
# ``min(initial * 2**retry + U(0, 1), max)`` and its worst case adds the full 1.0.
_RETRY_JITTER_WORST_CASE_S: float = 1.0
_RETRY_EXP_BASE: float = 2.0


def gemini_retry_sleep_allowance_s(attempts: int) -> float:
    """Worst-case TOTAL backoff sleep across ``attempts - 1`` retries, in seconds.

    A caller whose outer deadline cannot cancel the call — gap-fill v2's ``read_document``
    runs its client in a ``to_thread`` worker — has to fit the retries inside its existing
    budget, which means subtracting the sleeps before dividing the rest between attempts.
    Kept here beside the delay constants so the arithmetic can never be computed off a
    stale copy of them.
    """
    return sum(
        min(
            GEMINI_RETRY_INITIAL_DELAY_S * _RETRY_EXP_BASE**retry + _RETRY_JITTER_WORST_CASE_S, GEMINI_RETRY_MAX_DELAY_S
        )
        for retry in range(max(attempts - 1, 0))
    )


def gemini_thinking_config(level: str) -> genai_types.ThinkingConfig:
    """``ThinkingConfig`` for a level named as a plain string in ``constants.py``.

    The conversion lives here so the constants can stay strings — nothing else in the repo
    makes a config module import the SDK — and it resolves the name itself rather than
    handing the string to ``ThinkingLevel(...)``, whose ``CaseInSensitiveEnum._missing_``
    answers an unknown value with a ``UserWarning`` and a FABRICATED member: a typo'd
    constant would then travel all the way to the API as a made-up level. Raising here
    names the bad value instead.
    """
    try:
        thinking_level = genai_types.ThinkingLevel[level.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown Gemini thinking level {level!r}") from exc
    return genai_types.ThinkingConfig(thinking_level=thinking_level)


def build_gemini_http_options(*, timeout_ms: int, attempts: int) -> genai_types.HttpOptions:
    """``HttpOptions`` with a per-attempt timeout and a bounded retry ladder.

    ``timeout_ms`` is the SDK's per-REQUEST timeout (milliseconds, handed to httpx), so it
    applies to each attempt separately rather than bounding the elapsed total; ``attempts``
    counts the original request, so ``attempts=2`` means one retry.
    """
    return genai_types.HttpOptions(
        timeout=timeout_ms,
        retry_options=genai_types.HttpRetryOptions(
            attempts=attempts,
            initial_delay=GEMINI_RETRY_INITIAL_DELAY_S,
            max_delay=GEMINI_RETRY_MAX_DELAY_S,
            http_status_codes=list(GEMINI_RETRY_HTTP_STATUS_CODES),
        ),
    )
