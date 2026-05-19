"""Bounded retry + per-request timeout for the Metaculus question-list GET.

Stock forecasting-tools issues ``requests.get`` against
``https://www.metaculus.com/api/posts/`` with no timeout and no retry, so a
single transient 403/429/5xx anywhere in the question pagination kills the
whole CI run. Observed 2026-05-19: a CDN/WAF-style 403 (33s stall + generic
"API only available to authenticated users" body) returned on a healthy key
and known-good tournament; the same key worked seconds before and after.

This module applies two patches at startup, both via
``apply_fetch_hardening()`` (idempotent, sentinel-guarded):

1. Global socket-timeout patch: ``forecasting_tools.helpers.metaculus_api``'s
   ``requests.get`` is replaced once with a wrapper that injects
   ``timeout=FETCH_GET_TIMEOUT`` if the caller didn't supply one. Patched
   once and left in place — no per-request toggle. This dodges a lost-update
   race that toggling would introduce under any future concurrent caller,
   and tightens the default for every GET in forecasting-tools that funnels
   through this module (defense in depth).

2. Bounded retry on ``MetaculusApi._get_questions_from_api`` — the single
   chokepoint for every question-list GET (fed by ``forecast_on_tournament``,
   ``forecast_questions``, and the random/sequential pagination strategies).
   Retries with exponential backoff + jitter on retryable failures:
   ``requests.Timeout``, ``requests.ConnectionError``, and HTTP statuses
   ``{403, 429, 500, 502, 503, 504}``. 403 is included because the observed
   failure was a Cloudflare-style edge-layer 403 with auth-flavored body, not
   a real auth failure. A genuinely missing token would have raised
   ``ValueError`` synchronously from ``_get_auth_headers`` before ever
   reaching this wrapper, and a real 401 (which we do NOT retry) would still
   surface immediately.

Unlike publish_hardening, we don't need a ``concurrent.futures`` Future
wrapper here: the fetch path runs once at startup before the asyncio event
loop spins up, so a request-side socket timeout is a sufficient ceiling.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable

import requests
from forecasting_tools import MetaculusApi
from forecasting_tools.helpers import metaculus_api as _ft_metaculus_api

from metaculus_bot.constants import (
    FETCH_GET_BACKOFF_BASE,
    FETCH_GET_BACKOFF_JITTER,
    FETCH_GET_RETRIES,
    FETCH_GET_TIMEOUT,
)

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
    """Patch ``forecasting_tools.helpers.metaculus_api.requests.get`` once globally.

    Wraps the module's ``requests.get`` to inject ``timeout=timeout_s`` when
    the caller doesn't supply one. Idempotent: called from ``apply_fetch_hardening``
    which is itself sentinel-guarded, so no double-wrapping.

    Patched once and left in place rather than toggled per-request — that
    avoids the lost-update race a context-managed patch would have if
    multiple threads ever entered the wrapper simultaneously, and tightens
    the default for every GET in forecasting-tools that flows through this
    module (the bot's call site is single-threaded, but defense in depth).
    """
    original_get = _ft_metaculus_api.requests.get

    @functools.wraps(original_get)
    def get_with_timeout(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", timeout_s)
        return original_get(*args, **kwargs)

    _ft_metaculus_api.requests.get = get_with_timeout


def _is_retryable(exc: BaseException) -> bool:
    """Return True iff the exception represents a transient failure worth retrying.

    Walks the ``__cause__`` chain because forecasting-tools'
    ``raise_for_status_with_additional_info`` re-raises a fresh ``HTTPError``
    with no ``response`` attached, chaining the original via ``raise ... from
    e``. We need to inspect the original to read the status code.
    """
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(cur, requests.HTTPError) and cur.response is not None:
            if cur.response.status_code in _RETRYABLE_STATUSES:
                return True
        cur = cur.__cause__
    return False


def _summarize_exc(exc: BaseException) -> str:
    """One-line summary for log readability.

    Walks ``__cause__`` for the same reason ``_is_retryable`` does: the
    real-world failure goes through ``raise_for_status_with_additional_info``
    which re-raises an unattached ``HTTPError`` with the original chained.
    Without the chain walk, the WARNING line would say "HTTP ?" for every
    real Metaculus failure, defeating the debuggability of the log.
    """
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, requests.HTTPError) and cur.response is not None:
            return f"HTTP {cur.response.status_code}"
        cur = cur.__cause__
    return type(exc).__name__


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter. ``attempt`` is 1-indexed (first retry uses attempt=1)."""
    return FETCH_GET_BACKOFF_BASE * (2 ** (attempt - 1)) + random.uniform(0, FETCH_GET_BACKOFF_JITTER)


def _wrap_with_retry(method_name: str, original: Callable[..., Any]) -> Callable[..., Any]:
    """Return a wrapper that runs ``original`` with bounded retry on transient failures.

    Per-request socket timeout is handled separately by the global patch
    installed in ``apply_fetch_hardening``; this wrapper only owns retry.
    """

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Read at call time so test monkeypatching of FETCH_GET_RETRIES works.
        from metaculus_bot.constants import FETCH_GET_RETRIES as _retries

        attempts = _retries + 1

        for attempt in range(1, attempts + 1):
            try:
                return original(*args, **kwargs)
            except Exception as exc:
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

    return wrapper


def apply_fetch_hardening() -> None:
    """Install fetch hardening: global GET timeout default + bounded retry on the question-list path. Idempotent."""
    if getattr(MetaculusApi, _SENTINEL, False):
        return

    # Layer 1: global socket-timeout default on forecasting-tools' requests.get.
    # Done once, not per-request, to avoid the lost-update race a toggling
    # context manager would have under any future concurrent caller.
    _install_get_timeout_default(FETCH_GET_TIMEOUT)

    # Layer 2: bounded retry on the single chokepoint for question-list GETs.
    for method_name in _PATCHED_METHODS:
        descriptor = MetaculusApi.__dict__[method_name]
        if isinstance(descriptor, classmethod):
            original_func = descriptor.__func__
        else:
            # Defensive: forecasting-tools could change the decorator someday.
            original_func = getattr(MetaculusApi, method_name)

        wrapped = _wrap_with_retry(method_name, original_func)
        setattr(MetaculusApi, method_name, classmethod(wrapped))

    setattr(MetaculusApi, _SENTINEL, True)
    logger.info(
        "Fetch hardening applied: %d MetaculusApi GET method(s) wrapped (%ds timeout, %d retries, exp backoff)",
        len(_PATCHED_METHODS),
        FETCH_GET_TIMEOUT,
        FETCH_GET_RETRIES,
    )
