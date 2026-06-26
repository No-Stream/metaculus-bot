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
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import litellm.exceptions

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


def is_broadly_retryable(exc: BaseException) -> bool:
    """Broad retry predicate: retry anything that is not clearly-permanent.

    Used by the ``allowed_tries>=2`` sites (forecasters, crux analyzer, AskNews
    summarizer) which legitimately benefit from retrying things the transient set
    excludes (empty-model-response ``RuntimeError``, a parser-ish hiccup). The 30s
    elapsed gate in the shared loop still blocks slow failures regardless.

    Two exclusion sets are NOT retried: ``PERMANENT_NO_RETRY_EXCEPTIONS`` (litellm
    permanent API errors) and ``PYTHON_BUG_NO_RETRY_EXCEPTIONS`` (code-defect types
    like TypeError/AttributeError) — the latter so a bug fails fast with a clean
    traceback instead of being retried 3× during debugging (CLAUDE.md §2 fail-fast).
    """
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
    so the call is always bounded. On failure, the attempt is retried ONLY if it
    is not the last attempt AND the failure is *fast* (its own duration is under
    ``max_elapsed_s``) AND the ``predicate`` accepts the exception. Otherwise the
    exception propagates unchanged.

    The elapsed gate is the universal deadline-safety rule: a slow failure (e.g. a
    5-min reasoning attempt that then times out, or the wall guard's
    ``asyncio.TimeoutError`` firing at ``wall_timeout``) is NEVER retried, no
    matter the type or predicate.

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
            is_fast_retryable = elapsed < max_elapsed_s and should_retry(exc)
            if is_last or not is_fast_retryable:
                raise
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
