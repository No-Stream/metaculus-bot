"""Tests for the elapsed-gated transient-retry helper (``metaculus_bot.llm_retry``).

Background: a production LLM call failed with an INSTANT (``time taken=0.001s``)
``litellm_exc.Timeout`` during a concurrent async burst. Because the call was
``allowed_tries=1`` (forecasting-tools' tenacity ``stop_after_attempt(1)`` ⇒ zero
retries), it lost all work with no recovery.

``invoke_with_transient_retry`` retries ONLY fast/transient failures and NEVER
retries slow ones. The elapsed-time gate is the load-bearing safety constraint:
retrying a real multi-minute timeout would be catastrophic, so a failure that
took longer than ``max_elapsed_s`` is never retried regardless of its type.

All tests are self-contained (no live API). Backoff-sequence tests patch
``asyncio.sleep`` so they run instantly. Awaitable factories are built with
``AsyncMock`` (matching ``tests/test_gap_fill_pass.py`` style) so the source has
no bare ``async def`` helpers for flake8-async to complain about.
"""

import asyncio
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, patch

import litellm.exceptions as litellm_exc
import pytest

from metaculus_bot.llm_retry import (
    DEFAULT_TRANSIENT_BACKOFFS,
    TRANSIENT_RETRY_MAX_ELAPSED_S,
    invoke_with_broad_retry,
    invoke_with_transient_retry,
)

# A wall_timeout comfortably larger than any simulated call duration in these
# tests, so the helper's own asyncio.wait_for never fires unless we want it to.
_BIG_WALL = 1000.0

# Captured BEFORE the autouse fixture patches the module-level asyncio.sleep.
# Tests that need a *real* wall-clock delay (to trip the wall_timeout guard or
# cross a real elapsed gate) use this rather than the patched no-op sleep.
_REAL_SLEEP = asyncio.sleep


def _timeout(msg: str = "Connection timed out.") -> litellm_exc.Timeout:
    return litellm_exc.Timeout(msg, model="m", llm_provider="openrouter")


def _fake_clock(start: float, after: float) -> Callable[[], float]:
    """Return a monotonic() stand-in: first reading ``start``, all later ``after``.

    Robust to however many times the helper (or asyncio internals) sample the
    clock — only the first call is the attempt-start, every subsequent reading
    reports the post-failure time.
    """
    state = {"n": 0}

    def _clock() -> float:
        value = start if state["n"] == 0 else after
        state["n"] += 1
        return value

    return _clock


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch asyncio.sleep inside llm_retry so backoffs don't slow the suite.

    Returned so individual tests can assert call args. The patch targets the
    name as imported in the module under test.
    """
    fake_sleep = AsyncMock()
    monkeypatch.setattr("metaculus_bot.llm_retry.asyncio.sleep", fake_sleep)
    return fake_sleep


@pytest.mark.asyncio
async def test_fast_timeout_retries_then_succeeds() -> None:
    """A fast (sub-elapsed-gate) litellm_exc.Timeout retries and the next call wins."""
    awaitable = AsyncMock(side_effect=[_timeout(), "ok"])

    out = await invoke_with_transient_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="t")

    assert out == "ok"
    assert awaitable.await_count == 2


@pytest.mark.asyncio
async def test_success_on_first_try_calls_once_no_sleep(_no_real_sleep: AsyncMock) -> None:
    """Happy path: one call, no backoff sleep."""
    awaitable = AsyncMock(return_value="ok")

    out = await invoke_with_transient_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="t")

    assert out == "ok"
    assert awaitable.await_count == 1
    _no_real_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_slow_transient_failure_not_retried() -> None:
    """A litellm_exc.Timeout that takes LONGER than max_elapsed_s is NOT retried.

    This is the load-bearing safety gate: a real multi-minute stall surfaces as a
    transient-typed exception but must never be retried. We simulate elapsed time
    via a patched time.monotonic so the test stays instant.
    """
    awaitable = AsyncMock(side_effect=_timeout("real stall"))
    # start reads 0.0; every later reading reports past the gate, so the measured
    # elapsed forces the slow-failure branch regardless of internal clock samples.
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with patch("metaculus_bot.llm_retry.time.monotonic", clock):
        with pytest.raises(litellm_exc.Timeout):
            await invoke_with_transient_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="t")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_asyncio_timeout_error_not_retried() -> None:
    """The wall-clock guard firing (asyncio.TimeoutError) is a SLOW failure — never retried.

    The autouse fixture patches the helper's backoff ``asyncio.sleep``, but the
    helper's ``asyncio.wait_for`` is untouched, so a tiny ``wall_timeout`` against
    a long-running awaitable produces a genuine ``asyncio.TimeoutError``. Because
    that signals the slow-failure path (elapsed ≈ wall_timeout), there must be no
    retry — only one attempt is started.
    """
    calls = {"n": 0}

    async def _hang() -> str:
        # _REAL_SLEEP, not the patched no-op, so the call genuinely outlives the
        # 0.01s wall_timeout and asyncio.wait_for raises TimeoutError.
        await _REAL_SLEEP(5)
        return "never"

    def factory() -> Awaitable[str]:
        calls["n"] += 1
        return _hang()

    with pytest.raises(asyncio.TimeoutError):
        await invoke_with_transient_retry(factory, wall_timeout=0.01, label="t")

    assert calls["n"] == 1


@pytest.mark.asyncio
async def test_non_transient_exception_not_retried() -> None:
    """A non-transient error (AuthenticationError) is re-raised on the first attempt."""
    awaitable = AsyncMock(side_effect=litellm_exc.AuthenticationError("bad key", model="m", llm_provider="openrouter"))

    with pytest.raises(litellm_exc.AuthenticationError):
        await invoke_with_transient_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="t")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_value_error_not_retried() -> None:
    """A plain ValueError is not in the transient set → not retried."""
    awaitable = AsyncMock(side_effect=ValueError("nope"))

    with pytest.raises(ValueError):
        await invoke_with_transient_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="t")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_rate_limit_error_not_retried() -> None:
    """RateLimitError is deliberately excluded (handled by key-swap / AskNews backoff)."""
    awaitable = AsyncMock(side_effect=litellm_exc.RateLimitError("slow down", model="m", llm_provider="openrouter"))

    with pytest.raises(litellm_exc.RateLimitError):
        await invoke_with_transient_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="t")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_exhausting_backoffs_reraises_last_and_calls_n_plus_one_times() -> None:
    """All attempts fail transiently → last exc re-raised, awaitable called len(backoffs)+1 times."""
    awaitable = AsyncMock(
        side_effect=litellm_exc.APIConnectionError(message="blip", model="m", llm_provider="openrouter")
    )

    with pytest.raises(litellm_exc.APIConnectionError):
        await invoke_with_transient_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="t")

    assert awaitable.await_count == len(DEFAULT_TRANSIENT_BACKOFFS) + 1


@pytest.mark.asyncio
async def test_backoff_sequence_in_order(_no_real_sleep: AsyncMock) -> None:
    """Backoff sleeps happen in order: 1.0, 10.0, 30.0 (the default sequence)."""
    awaitable = AsyncMock(
        side_effect=litellm_exc.InternalServerError(message="500", model="m", llm_provider="openrouter")
    )

    with pytest.raises(litellm_exc.InternalServerError):
        await invoke_with_transient_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="t")

    actual = [c.args[0] for c in _no_real_sleep.await_args_list]
    assert actual == list(DEFAULT_TRANSIENT_BACKOFFS)
    assert actual == [1.0, 10.0, 30.0]


@pytest.mark.asyncio
async def test_custom_backoffs_control_attempt_count(_no_real_sleep: AsyncMock) -> None:
    """A custom backoffs tuple sets attempts = len(backoffs)+1 and its own sleep schedule."""
    awaitable = AsyncMock(
        side_effect=litellm_exc.ServiceUnavailableError(message="503", model="m", llm_provider="openrouter")
    )

    with pytest.raises(litellm_exc.ServiceUnavailableError):
        await invoke_with_transient_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="t", backoffs=(0.5, 2.0))

    assert awaitable.await_count == 3  # len((0.5, 2.0)) + 1
    assert [c.args[0] for c in _no_real_sleep.await_args_list] == [0.5, 2.0]


@pytest.mark.asyncio
async def test_elapsed_gate_blocks_retry_even_for_transient_type() -> None:
    """A transient-typed exc whose elapsed exceeds max_elapsed_s is NOT retried.

    Distinct from test_slow_transient_failure_not_retried in that it uses a real
    (short) custom max_elapsed_s and a genuine sleep to cross it, exercising the
    actual time.monotonic measurement rather than a patched clock.
    """

    async def _slow_then_fail() -> str:
        # _REAL_SLEEP so the measured elapsed actually crosses the 0.01s gate;
        # the patched no-op sleep would report elapsed≈0 and wrongly retry.
        await _REAL_SLEEP(0.05)
        raise _timeout("slow blip")

    awaitable = AsyncMock(side_effect=_slow_then_fail)

    with pytest.raises(litellm_exc.Timeout):
        await invoke_with_transient_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="t", max_elapsed_s=0.01)

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_make_awaitable_called_fresh_each_attempt() -> None:
    """The factory must produce a FRESH awaitable per attempt (coroutines are single-await).

    If the helper reused a single coroutine, attempt 2 would raise RuntimeError
    ("cannot reuse already awaited coroutine") instead of retrying cleanly.
    """
    factory_calls = {"n": 0}
    awaitable = AsyncMock(side_effect=[_timeout("blip"), "ok"])

    def factory():
        factory_calls["n"] += 1
        return awaitable()

    out = await invoke_with_transient_retry(factory, wall_timeout=_BIG_WALL, label="t")

    assert out == "ok"
    assert factory_calls["n"] == 2
    assert awaitable.await_count == 2


@pytest.mark.asyncio
async def test_warning_logged_on_each_retry(caplog: pytest.LogCaptureFixture) -> None:
    """Each retry emits a WARNING carrying the label so retries are auditable in logs."""
    awaitable = AsyncMock(side_effect=_timeout("blip"))

    with caplog.at_level("WARNING", logger="metaculus_bot.llm_retry"):
        with pytest.raises(litellm_exc.Timeout):
            await invoke_with_transient_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="my_label")

    retry_warnings = [r for r in caplog.records if "my_label" in r.message]
    # One warning per retry = len(backoffs) (the final attempt does not warn).
    assert len(retry_warnings) == len(DEFAULT_TRANSIENT_BACKOFFS)


def test_elapsed_gate_default_is_30s() -> None:
    """Canary: the universal "no retry after 30s" deadline-safety rule (Round-2).

    The gate measures the failed attempt's own duration; a 5-min reasoning attempt
    that then times out must never spawn another call (it would miss the question
    submission deadline). 30s sits above any genuine transient blip and far below
    every real per-call timeout (120/300/360/420/480/500s).
    """
    assert TRANSIENT_RETRY_MAX_ELAPSED_S == 30.0


# Broad-predicate retry (Round-2 change 3): for the allowed_tries>=2 sites
# (forecasters, crux analyzer, AskNews summarizer) that must keep their valuable
# fast retries (empty-response, parser hiccup) but obey the same 30s gate.
@pytest.mark.asyncio
async def test_broad_retry_retries_generic_error_under_gate() -> None:
    """A generic ValueError (e.g. empty-model-response) under the gate is retried."""
    awaitable = AsyncMock(side_effect=[ValueError("empty response"), "ok"])

    out = await invoke_with_broad_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="b")

    assert out == "ok"
    assert awaitable.await_count == 2


@pytest.mark.asyncio
async def test_broad_retry_does_not_retry_authentication_error() -> None:
    """AuthenticationError is clearly-permanent → never retried, even under the gate."""
    awaitable = AsyncMock(side_effect=litellm_exc.AuthenticationError("bad key", model="m", llm_provider="openrouter"))

    with pytest.raises(litellm_exc.AuthenticationError):
        await invoke_with_broad_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_broad_retry_does_not_retry_bad_request_error() -> None:
    """BadRequestError (and its subclasses, e.g. ContextWindowExceeded) is permanent."""
    awaitable = AsyncMock(side_effect=litellm_exc.BadRequestError("bad", model="m", llm_provider="openrouter"))

    with pytest.raises(litellm_exc.BadRequestError):
        await invoke_with_broad_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == 1


@pytest.mark.parametrize(
    "bug_exc",
    [TypeError("x"), AttributeError("x"), NameError("x"), KeyError("x"), IndexError("x"), ImportError("x")],
)
@pytest.mark.asyncio
async def test_broad_retry_does_not_retry_python_bug_types(bug_exc: Exception) -> None:
    """Python-internal bug types fail fast (CLAUDE.md §2): never retried, surface immediately.

    A TypeError/AttributeError/etc. signals a code defect, never a transient API
    condition; retrying it 3× just delays a clean traceback. Distinct from the
    empty-response RuntimeError, which IS broadly retryable (covered elsewhere).
    """
    awaitable = AsyncMock(side_effect=bug_exc)

    with pytest.raises(type(bug_exc)):
        await invoke_with_broad_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_broad_retry_retries_empty_response_runtimeerror() -> None:
    """RuntimeError (forecasting-tools' empty-model-response signal) IS broadly retried.

    Guards against over-tightening the predicate: RuntimeError must stay retryable
    even though it is a builtin, because it's the canonical valuable in-invoke retry.
    """
    awaitable = AsyncMock(side_effect=[RuntimeError("LLM answer is an empty string"), "ok"])

    out = await invoke_with_broad_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="b")

    assert out == "ok"
    assert awaitable.await_count == 2


@pytest.mark.asyncio
async def test_broad_retry_does_not_retry_slow_failure_of_any_type() -> None:
    """A generic error that takes LONGER than the gate is NOT retried (deadline safety).

    Even though ValueError would normally be broadly-retryable, a slow attempt
    (>30s) must never trigger another call.
    """
    awaitable = AsyncMock(side_effect=ValueError("slow generic failure"))
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with patch("metaculus_bot.llm_retry.time.monotonic", clock):
        with pytest.raises(ValueError):
            await invoke_with_broad_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_broad_retry_does_not_retry_asyncio_timeout() -> None:
    """asyncio.TimeoutError from the wall guard is a slow failure — not retried.

    It is retryable-by-type (not in PERMANENT_NO_RETRY_EXCEPTIONS), but it only
    fires at the wall cap (>> 30s), so the elapsed gate blocks the retry. We model
    the wall-cap firing as an asyncio.TimeoutError raised after the gate elapses
    (deterministic clock), rather than racing a real tiny wait_for timeout.
    """
    awaitable = AsyncMock(side_effect=asyncio.TimeoutError())
    clock = _fake_clock(0.0, TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0)

    with patch("metaculus_bot.llm_retry.time.monotonic", clock):
        with pytest.raises(asyncio.TimeoutError):
            await invoke_with_broad_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == 1


@pytest.mark.asyncio
async def test_broad_retry_shares_backoff_sequence(_no_real_sleep: AsyncMock) -> None:
    """The broad path reuses the same loop/backoff logic (no duplicated machinery)."""
    awaitable = AsyncMock(side_effect=ValueError("always"))

    with pytest.raises(ValueError):
        await invoke_with_broad_retry(lambda: awaitable(), wall_timeout=_BIG_WALL, label="b")

    assert awaitable.await_count == len(DEFAULT_TRANSIENT_BACKOFFS) + 1
    assert [c.args[0] for c in _no_real_sleep.await_args_list] == list(DEFAULT_TRANSIENT_BACKOFFS)


@pytest.mark.asyncio
async def test_predicate_param_overrides_transient_type_check() -> None:
    """Supplying predicate= on invoke_with_transient_retry replaces the type check.

    A ValueError is NOT in the transient type set, but a predicate that accepts it
    makes it retryable — proving predicate supersedes the default isinstance check
    while the shared gate/backoff/wall logic is reused.
    """
    awaitable = AsyncMock(side_effect=[ValueError("blip"), "ok"])

    out = await invoke_with_transient_retry(
        lambda: awaitable(), wall_timeout=_BIG_WALL, label="p", predicate=lambda _exc: True
    )

    assert out == "ok"
    assert awaitable.await_count == 2
