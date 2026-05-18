"""Wall-clock hardening for the synchronous Metaculus publish path.

Stock forecasting-tools makes the four publish POSTs against
``https://www.metaculus.com/api/`` via blocking ``requests.post`` calls with no
timeout (see ``forecasting_tools/helpers/metaculus_api.py``):

- ``MetaculusApi.post_binary_question_prediction``      -> ``requests.post``
- ``MetaculusApi.post_numeric_question_prediction``     -> ``requests.post``
- ``MetaculusApi.post_multiple_choice_question_prediction`` -> ``requests.post``
- ``MetaculusApi.post_question_comment``                -> ``requests.post``

If the Metaculus API hangs mid-tournament, those calls block the asyncio event
loop (they're invoked synchronously from inside the `async def
publish_report_to_metaculus` methods on each report type) and block every other
Q in the batch from publishing. ``apply_publish_hardening()`` monkey-patches
each of those four classmethods at startup so each POST runs on a worker
thread with a ``concurrent.futures.Future.result(timeout=...)`` cap, plus one
retry on timeout / connection error.

We use ``concurrent.futures.ThreadPoolExecutor`` (rather than asyncio.to_thread)
because the patched callsite remains synchronous — calling code is
``MetaculusApi.post_*(...)`` without await — so we can't return a coroutine.

Idempotent: calling ``apply_publish_hardening()`` more than once is a no-op
(checked via a sentinel attribute on ``MetaculusApi``).
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
from typing import Any, Callable

import requests
from forecasting_tools import MetaculusApi

from metaculus_bot.constants import PUBLISH_POST_RETRIES, PUBLISH_POST_TIMEOUT

logger = logging.getLogger(__name__)

_SENTINEL = "_publish_hardening_applied"

# Method names to patch. Each is a synchronous classmethod on MetaculusApi that
# wraps a single requests.post call.
_PATCHED_METHODS: tuple[str, ...] = (
    "post_binary_question_prediction",
    "post_numeric_question_prediction",
    "post_multiple_choice_question_prediction",
    "post_question_comment",
)

# Single-thread executor per patched method is enough — publish calls are
# infrequent and serialized within a single Q's publish_report_to_metaculus().
# A shared pool keeps overhead minimal across the four wrappers.
_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="publish-hardening")
    return _executor


def _wrap_with_timeout_retry(method_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    """Return a sync wrapper that runs ``original`` on a worker thread with timeout + retry."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        last_exc: BaseException | None = None
        attempts = PUBLISH_POST_RETRIES + 1  # initial try + retries
        executor = _get_executor()
        for attempt in range(1, attempts + 1):
            future = executor.submit(original, *args, **kwargs)
            try:
                return future.result(timeout=PUBLISH_POST_TIMEOUT)
            except concurrent.futures.TimeoutError as exc:
                last_exc = exc
                future.cancel()
                logger.warning(
                    "PUBLISH_HARDENING: %s attempt %d/%d timed out after %ds",
                    method_name,
                    attempt,
                    attempts,
                    PUBLISH_POST_TIMEOUT,
                )
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "PUBLISH_HARDENING: %s attempt %d/%d failed (%s: %s)",
                    method_name,
                    attempt,
                    attempts,
                    type(exc).__name__,
                    exc,
                )
        assert last_exc is not None
        raise last_exc

    return wrapper


def apply_publish_hardening() -> None:
    """Patch ``MetaculusApi.post_*`` to add timeout + retry. Idempotent."""
    if getattr(MetaculusApi, _SENTINEL, False):
        return

    for method_name in _PATCHED_METHODS:
        original = getattr(MetaculusApi, method_name)
        wrapped = _wrap_with_timeout_retry(method_name, original)
        setattr(MetaculusApi, method_name, wrapped)

    setattr(MetaculusApi, _SENTINEL, True)
    logger.info(
        "Publish hardening applied: %d MetaculusApi.post_* methods wrapped with %ds timeout + %d retry",
        len(_PATCHED_METHODS),
        PUBLISH_POST_TIMEOUT,
        PUBLISH_POST_RETRIES,
    )
