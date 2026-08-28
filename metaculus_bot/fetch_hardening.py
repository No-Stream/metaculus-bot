"""Bounded retry + per-request timeout for the Metaculus question-list GET.

forecasting-tools issues ``requests.get`` against
``https://www.metaculus.com/api/posts/`` from ``MetaculusClient`` — a single
transient 403/429/5xx anywhere in the question pagination could kill the whole
CI run. Observed 2026-05-19: a CDN/WAF-style 403 (33s stall + generic "API only
available to authenticated users" body) returned on a healthy key and known-good
tournament; the same key worked seconds before and after.

**0.2.92 architecture.** The live fetch chokepoint moved from the (now
deprecated) ``MetaculusApi`` classmethod shim onto ``MetaculusClient``
(``forecasting_tools/helpers/metaculus_client.py``). ``ForecastBot`` publishes
and fetches through an *instance* — ``self.metaculus_client = MetaculusClient()``
— and reports/fetches call plain **instance methods** on it. Patching the
deprecated ``MetaculusApi`` shim would test green and silently no-op in prod, so
we patch ``MetaculusClient`` at the class level: instance method lookup resolves
through the class, so a class-level patch covers every instance the bot builds
(verified with an instance-identity probe).

Two behaviors in 0.2.92 change how we harden:

- ``MetaculusClient`` passes ``timeout=self.timeout`` (default 30s) on the
  chokepoint GET, so the request already has a socket-timeout ceiling. We accept
  that upstream 30s for the chokepoint (tighter than our historical 60s, and the
  retry layer absorbs a slow-then-403 stall regardless). The socket-timeout
  install below is retained as **defense in depth** — it still bites bare GETs
  that omit a timeout (e.g. ``get_current_user_id``) and any future one.
- ``_get_questions_from_api`` is already ``@retry_with_exponential_backoff()``
  decorated upstream (retries on *any* ``RequestException``, including 401/404/
  422). We wrap **beneath** that decorator (via ``__wrapped__``) so our bounded
  retry is the single authoritative layer with an explicit status policy —
  fail-fast on 401/404/422, retry only 403/429/5xx + transport errors — rather
  than stacking two retry loops.

``apply_fetch_hardening()`` applies both patches (idempotent, sentinel-guarded):

1. Global socket-timeout patch: ``metaculus_client``'s ``requests.get`` is
   replaced once with a wrapper that injects ``timeout=FETCH_GET_TIMEOUT`` if the
   caller didn't supply one. Patched once and left in place — no per-request
   toggle. This dodges a lost-update race that toggling would introduce under any
   future concurrent caller.

2. Bounded retry on ``MetaculusClient._get_questions_from_api`` — the single
   chokepoint for every question-list GET (fed by ``forecast_on_tournament``,
   ``forecast_questions``, and the random/sequential pagination strategies).
   Retries with exponential backoff + jitter on retryable failures:
   ``requests.Timeout``, ``requests.ConnectionError``, and HTTP statuses
   ``{403, 429, 500, 502, 503, 504}``. 403 is included because the observed
   failure was a Cloudflare-style edge-layer 403 with auth-flavored body, not a
   real auth failure. A genuinely missing token raises ``ValueError``
   synchronously from ``_get_auth_headers`` before ever reaching this wrapper, and
   a real 401 (which we do NOT retry) still surfaces immediately.

Unlike publish_hardening, we don't need a ``concurrent.futures`` Future wrapper
here: the fetch path runs once at startup before the asyncio event loop spins up,
so a request-side socket timeout is a sufficient ceiling.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any

import requests
from forecasting_tools.helpers import metaculus_client as _ft_metaculus_client
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from metaculus_bot.constants import (
    FETCH_GET_BACKOFF_BASE,
    FETCH_GET_BACKOFF_JITTER,
    FETCH_GET_RETRIES,
    FETCH_GET_TIMEOUT,
)
from metaculus_bot.http_status import http_status_from_exception, iter_cause_chain

assert FETCH_GET_RETRIES >= 0, "FETCH_GET_RETRIES must be non-negative"
assert FETCH_GET_BACKOFF_BASE >= 0, "FETCH_GET_BACKOFF_BASE must be non-negative"
assert FETCH_GET_BACKOFF_JITTER >= 0, "FETCH_GET_BACKOFF_JITTER must be non-negative"

logger = logging.getLogger(__name__)

_SENTINEL = "_fetch_hardening_applied"

# Method to patch. Single chokepoint that every question-list GET funnels
# through (sequential and random pagination, binary-search count probe).
_PATCHED_METHODS: tuple[str, ...] = ("_get_questions_from_api",)

# HTTP statuses we retry. Excludes 401 (real auth failure — fail fast),
# 400/404/422 (client error — retrying won't help). 403 is included because
# the observed CDN/WAF-style failure surfaces as 403 with an auth-flavored
# body; a genuinely bad token fails earlier in `_get_auth_headers`.
_RETRYABLE_STATUSES: frozenset[int] = frozenset({403, 429, 500, 502, 503, 504})


def _install_get_timeout_default(timeout_s: float) -> None:
    """Patch ``forecasting_tools.helpers.metaculus_client.requests.get`` once globally.

    Wraps the module's ``requests.get`` to inject ``timeout=timeout_s`` when the
    caller doesn't supply one. Idempotent: called from ``apply_fetch_hardening``
    which is itself sentinel-guarded, so no double-wrapping.

    Patched once and left in place rather than toggled per-request — that avoids
    the lost-update race a context-managed patch would have if multiple threads
    ever entered the wrapper simultaneously. In 0.2.92 the question-list
    chokepoint already passes ``timeout=self.timeout`` (30s), so this is
    defense-in-depth: it bites bare GETs (e.g. ``get_current_user_id``) that omit
    a timeout, and is a no-op where one is already supplied.
    """
    original_get = _ft_metaculus_client.requests.get

    @functools.wraps(original_get)
    def get_with_timeout(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", timeout_s)
        return original_get(*args, **kwargs)

    _ft_metaculus_client.requests.get = get_with_timeout


def _is_retryable(exc: BaseException) -> bool:
    """Return True iff the exception represents a transient failure worth retrying.

    The status read is shared with publish_hardening
    (``metaculus_bot.http_status`` — the ``__cause__`` walk over ft's
    message-only ``HTTPError`` re-raise); the POLICY here is deliberately its
    inverse: fetches allow-list retryable statuses and default to no-retry,
    because a failed fetch retries next run for free while a failed publish
    forfeits the question.
    """
    for cause in iter_cause_chain(exc):
        if isinstance(cause, (requests.Timeout, requests.ConnectionError)):
            return True
    status = http_status_from_exception(exc)
    return status in _RETRYABLE_STATUSES


def _summarize_exc(exc: BaseException) -> str:
    """One-line summary for log readability.

    Uses the shared status read (which also recovers "Status code: NNN" from
    ft's message text) so a real Metaculus failure logs "HTTP 405" rather than
    the type name of the message-only re-raise.
    """
    status = http_status_from_exception(exc)
    return f"HTTP {status}" if status is not None else type(exc).__name__


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter. ``attempt`` is 1-indexed (first retry uses attempt=1)."""
    return FETCH_GET_BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(  # noqa: S311  # retry jitter, not cryptography
        0, FETCH_GET_BACKOFF_JITTER
    )


def _wrap_with_retry(method_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    """Return a wrapper that runs ``original`` with bounded retry on transient failures.

    ``original`` is the undecorated ``MetaculusClient`` instance method, so the
    wrapper is installed as a plain function (an instance method) and forwards
    ``self`` through ``*args`` transparently. Per-request socket timeout is
    handled separately by the global patch installed in ``apply_fetch_hardening``;
    this wrapper only owns retry.
    """

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Read at call time so test monkeypatching of FETCH_GET_RETRIES works.
        from metaculus_bot.constants import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # late read: tests patch this constant on the constants module
            FETCH_GET_RETRIES as _retries,
        )

        attempts = _retries + 1

        for attempt in range(1, attempts + 1):
            try:
                return original(*args, **kwargs)
            except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except: retry-then-reraise, re-raised below on non-retryable / last attempt
                if not _is_retryable(exc) or attempt == attempts:
                    raise
                sleep_s = _backoff_seconds(attempt)
                logger.warning(
                    "FETCH_HARDENING: %s attempt %d/%d failed (%s); retrying in %.1fs",
                    method_name,
                    attempt,
                    attempts,
                    _summarize_exc(exc),
                    sleep_s,
                )
                time.sleep(sleep_s)

        # HARNESS-SCAN-EXEMPT-shouldnt-happen-silent-fallback: unreachable while
        # attempts >= 1 — the loop either returns, re-raises, or retries. Spelled out
        # because RET503 wants the fall-through explicit; behavior is unchanged.
        return None

    return wrapper


def apply_fetch_hardening() -> None:
    """Install fetch hardening: global GET timeout default + bounded retry on the question-list path. Idempotent."""
    if getattr(MetaculusClient, _SENTINEL, False):
        return

    # Layer 1: global socket-timeout default on forecasting-tools' requests.get.
    # Done once, not per-request, to avoid the lost-update race a toggling
    # context manager would have under any future concurrent caller.
    _install_get_timeout_default(FETCH_GET_TIMEOUT)

    # Layer 2: bounded retry on the single chokepoint for question-list GETs.
    # These are plain instance methods on MetaculusClient (not classmethods), so
    # we install a plain function that Python binds as an instance method. We
    # wrap the *undecorated* original (``__wrapped__``) to sit beneath upstream's
    # own @retry_with_exponential_backoff — ours is then the single retry layer
    # with an explicit status policy, rather than stacking two loops.
    for method_name in _PATCHED_METHODS:
        raw = MetaculusClient.__dict__[method_name]
        original_func = getattr(raw, "__wrapped__", raw)
        wrapped = _wrap_with_retry(method_name, original_func)
        setattr(MetaculusClient, method_name, wrapped)

    setattr(MetaculusClient, _SENTINEL, True)
    logger.info(
        "Fetch hardening applied: %d MetaculusClient GET method(s) wrapped (%ds timeout, %d retries, exp backoff)",
        len(_PATCHED_METHODS),
        FETCH_GET_TIMEOUT,
        FETCH_GET_RETRIES,
    )
