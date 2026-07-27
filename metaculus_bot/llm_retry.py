"""Elapsed-gated transient retry for ``allowed_tries=1`` async LLM calls.

Why this exists
---------------
A production LLM call failed with an INSTANT (``Timeout passed=120.0, time
taken=0.001 seconds``) ``litellm.Timeout`` during a concurrent async burst.
litellm 1.80.0 caught an ``httpx.TimeoutException`` from the aiohttp transport
(default since v1.71.x) and re-wrapped it as ``litellm.Timeout``; under
concurrent bursts that transport raises near-instant connection failures
(litellm issue #14895 — see ``scratch_docs_and_planning/transient_retry_fix.md``).
Because the call was configured ``allowed_tries=1`` (forecasting-tools'
``RetryableModel`` tenacity ``stop_after_attempt(1)`` ⇒ zero retries with no
exception predicate), it lost all work with no recovery.

The fix
-------
``invoke_with_transient_retry`` retries ONLY *fast* transient failures and NEVER
retries *slow* ones. The elapsed-time gate is the load-bearing safety
constraint: retrying a real multi-minute stall 3× would be catastrophic. A
failure is retried only when BOTH its type is in ``TRANSIENT_RETRY_EXCEPTIONS``
AND it surfaced in under ``max_elapsed_s`` seconds. A wall-clock
``asyncio.TimeoutError`` (fires at ``wall_timeout`` ≫ ``max_elapsed_s``) and a
genuine 120s ``litellm.Timeout`` are therefore never retried — only the
sub-second blips are.

Composes with the existing ``allowed_tries=1`` configs (the inner tenacity is a
no-op there) and with the stacker's cross-provider fallback design (a slow
stall still falls through to the fallback model rather than being retried here).

Zero-output exemption
---------------------
The elapsed gate has one carve-out. A SLOW failure that returned *no usable
content* — an empty/whitespace HTTP body (litellm ``APIError`` "Unable to get
json response") or an empty completion (forecasting-tools ``RuntimeError`` "LLM
answer is an empty string") — is the single most valuable retry case: the prior
attempt produced nothing, so a re-roll is cheap in EV. ``is_zero_output_failure``
detects it and the loop re-rolls it ONCE, immediately (no backoff), bypassing the
gate. This closed the 2026-07-25 gap where a forecaster's slow whitespace-body
``APIError`` was never retried and the question published on 2 of 3 models. A
genuine ``asyncio.TimeoutError`` is NOT zero-output (the model may have been
mid-generation) so it stays gated. The exemption is ANDed with the retry
predicate, so it only fires on the broad-predicate sites (forecasters, crux,
summarizer); the transient-predicate sites (stacker, research) reject the
zero-output types and are untouched.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import litellm.exceptions
import openai

logger: logging.Logger = logging.getLogger(__name__)

# Fast versions of these are transient connection/server blips worth a cheap
# retry. RateLimitError is deliberately excluded — it's handled upstream by
# FallbackOpenRouterLlm's key-swap and AskNews's own backoff. asyncio.TimeoutError
# is also excluded by design: it only fires when the wall-clock guard trips,
# which is a SLOW failure and must never be retried. Reference the classes via
# ``litellm.exceptions`` (the repo convention, e.g. prediction_market.py) so
# basedpyright sees them as exported (reportPrivateImportUsage).
TRANSIENT_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    litellm.exceptions.Timeout,
    litellm.exceptions.APIConnectionError,
    litellm.exceptions.InternalServerError,
    litellm.exceptions.ServiceUnavailableError,
)

# Clearly-permanent failures the BROAD predicate never retries: re-rolling the
# same call cannot fix a bad key, malformed request, missing model, denied
# permission, unprocessable payload, content-policy block, or over-long context.
# ContentPolicyViolationError / ContextWindowExceededError subclass BadRequestError
# but are listed explicitly for clarity. Everything else (empty-model-response,
# generic parser hiccup, transient blips, asyncio.TimeoutError-by-type) is
# broadly retryable — but the 30s elapsed gate still blocks slow failures.
PERMANENT_NO_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    litellm.exceptions.AuthenticationError,
    litellm.exceptions.BadRequestError,
    litellm.exceptions.NotFoundError,
    litellm.exceptions.PermissionDeniedError,
    litellm.exceptions.UnprocessableEntityError,
    litellm.exceptions.ContentPolicyViolationError,
    litellm.exceptions.ContextWindowExceededError,
)

# Python-internal bug types the BROAD predicate also never retries: these signal a
# code defect (a typo, a None attribute access, a bad index/key, a missing import),
# never a transient API condition. Per the repo's fail-fast policy (CLAUDE.md §2:
# "let unexpected errors crash with clear stack traces") a code bug should surface
# IMMEDIATELY with a clean traceback during debugging — not get retried 3× first.
# NOTE: RuntimeError is deliberately NOT here — forecasting-tools raises
# ``RuntimeError`` on an empty model response (general_llm.py: "LLM answer is an
# empty string ... will probably result in a retry"), which is the single most
# valuable in-invoke retry case. ValueError / AssertionError are left retryable too
# (ambiguous: a parse/validation hiccup is as likely as a bug, and the empty-string
# path's sibling asserts use AssertionError).
PYTHON_BUG_NO_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TypeError,
    AttributeError,
    NameError,
    KeyError,
    IndexError,
    ImportError,
)

# Deterministic HTTP client errors the BROAD predicate never retries: the provider
# rejected the request itself, so an identical retry gets an identical rejection.
# Read off the exception's own ``status_code`` because the two type tuples above
# under-cover this — the canonical gap is an OpenRouter 403, for which litellm has no
# branch in ``_map_openrouter_exception`` and so raises a BARE ``APIError``, the root
# of the tree, matching nothing in ``PERMANENT_NO_RETRY_EXCEPTIONS``.
# (``PermissionDeniedError`` is listed there and reads as though it covers 403, but it
# subclasses a different openai branch that litellm never raises for OpenRouter, so it
# contributes no real coverage.) In the 2026-07-26 run that let a drained-donated-key
# 403 — deterministic, 35ms — pass the 30s elapsed gate, which only screens SLOW
# failures, and the AskNews summarizer burned the full 1s/10s/30s ladder against a key
# that could not succeed. Mostly redundant for 400/401/404/422 (already typed
# permanent above); 402 and 403 are the statuses this actually adds.
# Every status listed here also becomes non-retryable for a ZERO-OUTPUT body arriving
# at it: ``is_zero_output_failure`` recognizes those by message marker and reads no
# status at all, so the carve-out is not status-protected — it survives only because
# the two statuses it actually sees in practice, 200 and 500, are absent here. Keep
# them absent, and weigh that cost before adding a status (a whitespace body served
# with the new status stops being re-rolled).
NON_RETRYABLE_HTTP_STATUS_CODES: frozenset[int] = frozenset({400, 401, 402, 403, 404, 422})

# Universal deadline-safety rule (Round-2): a failure whose own attempt took
# longer than this (seconds) is treated as SLOW and NEVER retried, regardless of
# exception type or predicate. A 5-min reasoning attempt that then times out must
# not spawn another call — that would miss the question submission deadline. 30s
# sits above any genuine transient blip and far below every real per-call timeout
# in the bot (120 / 300 / 360 / 420 / 480 / 500s), cleanly separating the regimes.
TRANSIENT_RETRY_MAX_ELAPSED_S: float = 30.0

# Backoff (seconds) before each retry. len(backoffs) retries ⇒ len(backoffs)+1
# total attempts. Worst-case added latency on an all-fast-fail run is the sum of
# these plus one final attempt up to wall_timeout — bounded.
DEFAULT_TRANSIENT_BACKOFFS: tuple[float, ...] = (1.0, 10.0, 30.0)


def _is_transient_type(exc: BaseException) -> bool:
    """Default retry predicate: the exception's type is a fast transient blip."""
    return isinstance(exc, TRANSIENT_RETRY_EXCEPTIONS)


# Message markers for the empty/whitespace-body case: litellm re-wraps a
# JSONDecodeError on an unparseable HTTP body ("Unable to get json response -
# Expecting value: ...") — the 2026-07-25 production failure where OpenRouter
# returned a 200 with a whitespace-only body for claude-opus-4.8. Matched
# case-insensitively on the message so a phrasing tweak across litellm versions
# still catches the family. The "empty completion" case is handled by RuntimeError
# type + "empty string" (forecasting-tools general_llm.py) in is_zero_output_failure.
_ZERO_OUTPUT_BODY_MARKERS: tuple[str, ...] = (
    "unable to get json response",  # litellm: raw_response.json() failed on empty/whitespace body
    "empty model response",  # defensive: alternate no-content phrasing
)


def is_zero_output_failure(exc: BaseException) -> bool:
    """True when a failure means the provider returned NO usable content.

    Two concrete shapes, both "HTTP success but nothing usable" rather than a real
    timeout or a structured refusal:

    * litellm re-wraps a ``JSONDecodeError`` on an empty/whitespace response body as
      an :class:`~litellm.exceptions.APIError` whose message contains "Unable to get
      json response" (OpenRouter returned a 200 with nothing parseable — the
      2026-07-25 case that dropped a forecaster).
    * forecasting-tools raises ``RuntimeError("LLM answer is an empty string ...")``
      (``general_llm.py``) when the body parsed but the completion string was empty.

    A genuine ``asyncio.TimeoutError`` (the wall guard firing) is deliberately NOT
    zero-output: the model may have been mid-generation when we cut it off, so
    re-rolling it risks the submission deadline — the elapsed gate must keep blocking
    it. This predicate only *exempts a slow failure from the elapsed gate*; it is
    always ANDed with the site's retryability predicate in the retry loop, so a
    permanent error (content-policy block, bad request) can never reach it, and the
    transient-predicate sites (stacker, research providers) — which reject APIError /
    RuntimeError by type — never fire the exemption.
    """
    message = str(exc).lower()
    if isinstance(exc, litellm.exceptions.APIError) and any(marker in message for marker in _ZERO_OUTPUT_BODY_MARKERS):
        return True
    return isinstance(exc, RuntimeError) and "empty string" in message


def llm_status_code(exc: BaseException) -> int | None:
    """The HTTP status an LLM provider reported, or ``None`` when the exception carries none.

    Public so every classifier that needs a status reads the same primitive instead of
    reimplementing the attribute read; today's caller is this module's retry predicate.
    Scoped to ``openai.APIError``, the common root of every litellm
    exception (``litellm.exceptions.APIError.__mro__[1] is openai.APIError``), so a
    same-named attribute on some unrelated exception can never be read as a provider
    status. ``requests``-shaped errors keep their status on ``.response`` and so report
    ``None`` here, which is what leaves textual cues in charge for them — load-bearing,
    because "403 is permanent" is NOT a general truth in this repo: ``fetch_hardening``
    deliberately RETRIES a CDN/WAF 403 on the Metaculus question-list endpoint (the
    2026-05-19 incident).

    Callers read this int rather than grepping the message for digits. litellm formats
    the message as ``f"APIError: {provider} - {body}"`` and an OpenRouter body embeds a
    64-hex key hash (~8.8% chance of containing one of 401/402/403/429/502/503) plus, on
    a moderation refusal, up to ~100 chars of our own prompt in ``flagged_input`` —
    either can make a number appear that was never a status. AskNews is the deliberate
    exception: its SDK raises its own ``asknews_sdk.errors`` classes carrying a ``.code``
    (429000 / 403011) and never subclasses ``openai.APIError``, so this returns ``None``
    for them and a text match is the honest primitive there.
    """
    if not isinstance(exc, openai.APIError):
        return None
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _is_deterministic_client_error(exc: BaseException) -> bool:
    """Whether ``exc`` is an LLM API failure whose HTTP status makes a retry pointless."""
    return llm_status_code(exc) in NON_RETRYABLE_HTTP_STATUS_CODES


def is_broadly_retryable(exc: BaseException) -> bool:
    """Broad retry predicate: retry anything that is not clearly-permanent.

    Used by the ``allowed_tries>=2`` sites (forecasters, crux analyzer, AskNews
    summarizer) which legitimately benefit from retrying things the transient set
    excludes (empty-model-response ``RuntimeError``, a parser-ish hiccup). The 30s
    elapsed gate in the shared loop still blocks slow failures, except a slow
    zero-output failure, which the loop re-rolls once (see ``is_zero_output_failure``).

    Three exclusions are NOT retried: a deterministic 4xx status
    (``NON_RETRYABLE_HTTP_STATUS_CODES``, checked FIRST because litellm's types
    under-cover it), ``PERMANENT_NO_RETRY_EXCEPTIONS`` (litellm permanent API errors)
    and ``PYTHON_BUG_NO_RETRY_EXCEPTIONS`` (code-defect types like
    TypeError/AttributeError) — the last so a bug fails fast with a clean traceback
    instead of being retried 3× during debugging (CLAUDE.md §2 fail-fast).
    """
    if _is_deterministic_client_error(exc):
        return False
    return not isinstance(exc, PERMANENT_NO_RETRY_EXCEPTIONS + PYTHON_BUG_NO_RETRY_EXCEPTIONS)


async def invoke_with_transient_retry(
    make_awaitable: Callable[[], Awaitable[str]],
    *,
    wall_timeout: float,
    label: str,
    backoffs: tuple[float, ...] = DEFAULT_TRANSIENT_BACKOFFS,
    max_elapsed_s: float = TRANSIENT_RETRY_MAX_ELAPSED_S,
    predicate: Callable[[BaseException], bool] | None = None,
) -> str:
    """Invoke an async LLM call with an elapsed-gated retry + wall cap.

    Each attempt wraps ``make_awaitable()`` in ``asyncio.wait_for(..., wall_timeout)``
    so the call is always bounded. On failure, the attempt is retried (predicate
    permitting, and if it is not the last attempt) in one of two ways: a *fast*
    failure (own duration under ``max_elapsed_s``) runs the full backoff ladder; a
    *slow zero-output* failure (``is_zero_output_failure``) is re-rolled exactly ONCE
    with no backoff. Otherwise the exception propagates unchanged.

    The elapsed gate is the universal deadline-safety rule: a slow failure (e.g. a
    5-min reasoning attempt that then times out, or the wall guard's
    ``asyncio.TimeoutError`` firing at ``wall_timeout``) is NEVER retried — the sole
    exception being a slow *zero-output* failure, which produced no content and so
    earns one immediate re-roll (bounded by ``wall_timeout``, capped at one).

    Args:
        make_awaitable: A ZERO-ARG FACTORY returning a FRESH awaitable each call.
            Must not be a single coroutine object — coroutines are single-await,
            so a retry would raise ``RuntimeError`` on a reused one. Pass e.g.
            ``lambda: llm.invoke(prompt)``.
        wall_timeout: Hard per-attempt wall-clock cap (seconds). Mirrors the
            ``asyncio.wait_for`` backstop each call site previously applied.
        label: Short identifier for the call site (e.g. ``"gap_fill_resolver"``),
            included in retry WARNING logs for auditability.
        backoffs: Sleep (seconds) before each retry; ``len(backoffs)+1`` attempts.
        max_elapsed_s: Elapsed-time gate — failures slower than this never retry.
        predicate: Decides whether an exception is retryable. ``None`` (default)
            uses the transient-type check (``_is_transient_type``); supply one to
            replace that check (e.g. ``is_broadly_retryable``). The gate, backoff,
            and wall-clock logic are identical and shared regardless.

    Returns:
        The awaited result of the first successful attempt.

    Raises:
        The exception from the final (or first non-retryable) attempt, unchanged.
    """
    should_retry = predicate if predicate is not None else _is_transient_type
    total_attempts = len(backoffs) + 1
    zero_output_reroll_used = False
    for attempt in range(total_attempts):
        start = time.monotonic()
        # The broad catch is the whole point: we must inspect ANY exception to
        # classify retry-vs-propagate, and every path either re-raises or loops
        # (never swallows) — not a silent failure.
        try:
            return await asyncio.wait_for(make_awaitable(), timeout=wall_timeout)
        except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
            elapsed = time.monotonic() - start
            is_last = attempt == total_attempts - 1
            retryable = should_retry(exc)
            is_fast_retryable = retryable and elapsed < max_elapsed_s
            # Zero-output exemption: a SLOW failure that returned no usable content
            # (empty/whitespace body, empty completion) is re-rolled ONCE with no
            # backoff, bypassing the elapsed gate. The previous attempt produced
            # nothing, so a single re-roll is cheap in EV; capping at one and
            # skipping the backoff keeps it from reintroducing the deadline risk the
            # gate guards against (backoff only helps load/connection blips, never an
            # empty body). Still ANDed with should_retry, so permanent errors and the
            # transient-predicate sites (stacker, research) are untouched.
            take_zero_output_reroll = (
                retryable and not is_fast_retryable and not zero_output_reroll_used and is_zero_output_failure(exc)
            )
            if is_last or not (is_fast_retryable or take_zero_output_reroll):
                raise
            if take_zero_output_reroll:
                zero_output_reroll_used = True
                logger.warning(
                    f"LLM_RETRY[{label}]: slow zero-output failure on attempt {attempt + 1}/{total_attempts} "
                    f"({type(exc).__name__}, {elapsed=:.3f}s >= {max_elapsed_s}s); re-rolling once immediately, "
                    f"no backoff — provider returned no usable content: {exc}"
                )
                continue
            backoff = backoffs[attempt]
            logger.warning(
                f"LLM_RETRY[{label}]: fast retryable failure on attempt {attempt + 1}/{total_attempts} "
                f"({type(exc).__name__}, {elapsed=:.3f}s < {max_elapsed_s}s); retrying after {backoff}s backoff: {exc}"
            )
            await asyncio.sleep(backoff)

    # Unreachable: the final attempt either returns or re-raises above. Present
    # so static analysis sees a definite return/raise on every path.
    raise AssertionError(f"invoke_with_transient_retry[{label}] exhausted loop without returning")


async def invoke_with_broad_retry(
    make_awaitable: Callable[[], Awaitable[str]],
    *,
    wall_timeout: float,
    label: str,
    backoffs: tuple[float, ...] = DEFAULT_TRANSIENT_BACKOFFS,
    max_elapsed_s: float = TRANSIENT_RETRY_MAX_ELAPSED_S,
) -> str:
    """Elapsed-gated retry that retries any non-permanent error (broad predicate).

    Thin wrapper over :func:`invoke_with_transient_retry` with
    ``predicate=is_broadly_retryable`` — same shared loop, gate, backoff, and wall
    cap. For the ``allowed_tries>=2`` sites (forecasters, crux analyzer, AskNews
    summarizer) set to ``allowed_tries=1`` so this gated wrapper is their SOLE
    retry layer, imposing the universal "no retry after 30s" rule that
    forecasting-tools' un-gated tenacity cannot.
    """
    return await invoke_with_transient_retry(
        make_awaitable,
        wall_timeout=wall_timeout,
        label=label,
        backoffs=backoffs,
        max_elapsed_s=max_elapsed_s,
        predicate=is_broadly_retryable,
    )
