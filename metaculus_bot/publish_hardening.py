"""Wall-clock hardening for the synchronous Metaculus publish path.

forecasting-tools makes the four publish POSTs against
``https://www.metaculus.com/api/`` via blocking ``requests.post`` calls (see
``forecasting_tools/helpers/metaculus_client.py``):

- ``MetaculusClient.post_binary_question_prediction``          -> ``requests.post``
- ``MetaculusClient.post_numeric_question_prediction``         -> ``requests.post``
- ``MetaculusClient.post_multiple_choice_question_prediction`` -> ``requests.post``
- ``MetaculusClient.post_question_comment``                    -> ``requests.post``

If the Metaculus API hangs mid-tournament, those calls block the asyncio event
loop (they're invoked synchronously from inside the ``async def
publish_report_to_metaculus`` methods on each report type) and block every other
Q in the batch from publishing. ``apply_publish_hardening()`` monkey-patches each
of those four methods at startup with two layers of defense:

1. Request-side socket timeout (primary): for the duration of the wrapped call,
   ``requests.post`` on the metaculus_client module is patched to set
   ``timeout=PUBLISH_POST_TIMEOUT``. This makes the underlying socket actually
   close when the server stalls, so the worker thread terminates instead of
   leaking.

2. ``concurrent.futures.Future.result(timeout=...)`` cap (belt-and-suspenders):
   covers pathological cases where a request might somehow ignore the socket
   timeout (e.g. unbounded DNS resolution before connect). Note that
   ``Future.cancel()`` does NOT interrupt a running thread; without layer (1)
   the worker would keep running until socket close, risking duplicate publishes
   on retry. Layer (1) makes that scenario unreachable.

Each wrapper retries once on timeout / connection error.

**0.2.92 architecture.** Publishing moved from the (now deprecated)
``MetaculusApi`` classmethod shim onto ``MetaculusClient``. ``ForecastBot``
publishes through an *instance* — ``report.publish_report_to_metaculus`` calls
``metaculus_client.post_*(...)`` on ``self.metaculus_client = MetaculusClient()``.
Patching the deprecated ``MetaculusApi`` shim would test green and silently
no-op in prod, so we patch ``MetaculusClient`` at the class level (plain instance
methods): instance method lookup resolves through the class, so a class-level
patch covers every instance the bot builds.

Two 0.2.92 behaviors shape the two layers:

- ``MetaculusClient`` now passes ``timeout=self.timeout`` (default 30s) on the
  POST, so a naive ``setdefault`` timeout injection would be a no-op. Layer (1)
  therefore **overrides** the timeout to the tighter ``PUBLISH_POST_TIMEOUT``
  (20s) for the duration of the wrapped call — this aligns the worker-side socket
  close with the caller-side ``Future.result`` cap (both 20s) so the worker
  thread dies at the cap instead of lingering to the upstream 30s. Overriding is
  safe here: the only caller is ``MetaculusClient`` itself (small publish
  payloads), and a tighter publish ceiling is exactly the intent.
- ``post_question_comment`` and the shared ``_post_question_prediction`` are
  already ``@retry_with_exponential_backoff()`` decorated upstream. For
  ``post_question_comment`` we wrap **beneath** that decorator (via
  ``__wrapped__``) so our retry is the single layer. The three prediction methods
  are undecorated public wrappers that delegate to the decorated
  ``_post_question_prediction``; that inner upstream retry remains one layer
  below ours. Total wall-clock is bounded by the ``Future.result`` cap
  regardless, and duplicate-publish risk on retry is inherent to any
  retry-on-timeout and unchanged by the upgrade.

We use ``concurrent.futures.ThreadPoolExecutor`` (rather than asyncio.to_thread)
because the patched callsite remains synchronous — calling code is
``metaculus_client.post_*(...)`` without await — so we can't return a coroutine.

The wrappers are attached as plain functions (instance methods) so both
class-level and instance-level calls preserve the original ``self``-first calling
convention (0.2.92 uses instance methods, not classmethods).

Idempotent: calling ``apply_publish_hardening()`` more than once is a no-op
(checked via a sentinel attribute on ``MetaculusClient``).
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import logging
from typing import Any, Callable, Iterator

import requests
from forecasting_tools.helpers import metaculus_client as _ft_metaculus_client
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from metaculus_bot.constants import PUBLISH_POST_RETRIES, PUBLISH_POST_TIMEOUT

assert PUBLISH_POST_RETRIES >= 0, "PUBLISH_POST_RETRIES must be non-negative"

logger = logging.getLogger(__name__)

_SENTINEL = "_publish_hardening_applied"

# Method names to patch. Each is a synchronous instance method on MetaculusClient
# that wraps (directly or via _post_question_prediction) a single requests.post.
_PATCHED_METHODS: tuple[str, ...] = (
    "post_binary_question_prediction",
    "post_numeric_question_prediction",
    "post_multiple_choice_question_prediction",
    "post_question_comment",
)

# Single shared executor across the four wrappers. Publish calls are infrequent
# and serialized within a single Q's publish_report_to_metaculus().
_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="publish-hardening")
    return _executor


@contextlib.contextmanager
def _inject_socket_timeout(timeout_s: float) -> Iterator[None]:
    """Patch ``metaculus_client.requests.post`` to force ``timeout=timeout_s``.

    Overrides (not setdefault) the timeout: 0.2.92's ``MetaculusClient`` always
    passes ``timeout=self.timeout`` (30s), so a setdefault would never fire. We
    force our tighter publish ceiling so the socket closes in step with the
    caller-side ``Future.result`` cap.
    """
    original_post = _ft_metaculus_client.requests.post

    @functools.wraps(original_post)
    def post_with_timeout(*args: Any, **kwargs: Any) -> Any:
        kwargs["timeout"] = timeout_s
        return original_post(*args, **kwargs)

    _ft_metaculus_client.requests.post = post_with_timeout
    try:
        yield
    finally:
        _ft_metaculus_client.requests.post = original_post


def _wrap_with_timeout_retry(method_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    """Return a sync wrapper that runs ``original`` on a worker thread with timeout + retry.

    ``original`` is the undecorated MetaculusClient instance method, so the
    wrapper is installed as a plain function (an instance method) and forwards
    ``self`` through ``*args`` transparently. The wrapper layers two timeout
    mechanisms:
    - Request-side: ``requests.post`` on metaculus_client is monkey-patched to
      force ``timeout=PUBLISH_POST_TIMEOUT``, bounding the underlying socket.
    - Caller-side: ``Future.result(timeout=...)`` provides a final ceiling.
    """

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        executor = _get_executor()
        attempts = PUBLISH_POST_RETRIES + 1  # read at call time so tests' monkeypatch is honored

        def _run_with_socket_timeout() -> Any:
            with _inject_socket_timeout(PUBLISH_POST_TIMEOUT):
                return original(*args, **kwargs)

        last_exc: BaseException = RuntimeError(f"PUBLISH_HARDENING: {method_name} loop exited without running")
        for attempt in range(1, attempts + 1):
            future = executor.submit(_run_with_socket_timeout)
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
        raise last_exc

    return wrapper


def apply_publish_hardening() -> None:
    """Patch ``MetaculusClient.post_*`` to add timeout + retry. Idempotent."""
    if getattr(MetaculusClient, _SENTINEL, False):
        return

    for method_name in _PATCHED_METHODS:
        # These are plain instance methods on MetaculusClient (not classmethods),
        # so we install a plain function that Python binds as an instance method.
        # Wrap the *undecorated* original (``__wrapped__``) when present so we sit
        # beneath any upstream @retry_with_exponential_backoff (e.g. on
        # post_question_comment) rather than stacking two retry loops.
        raw = MetaculusClient.__dict__[method_name]
        original_func = getattr(raw, "__wrapped__", raw)
        wrapped = _wrap_with_timeout_retry(method_name, original_func)
        setattr(MetaculusClient, method_name, wrapped)

    setattr(MetaculusClient, _SENTINEL, True)
    logger.info(
        "Publish hardening applied: %d MetaculusClient.post_* methods wrapped with %ds timeout + %d retry",
        len(_PATCHED_METHODS),
        PUBLISH_POST_TIMEOUT,
        PUBLISH_POST_RETRIES,
    )
