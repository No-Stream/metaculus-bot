"""Per-run OpenRouter credit-balance telemetry.

Fetches key balances (donated + personal) at run start and end and emits
greppable marker lines — ``CREDIT_BALANCE:`` / ``CREDIT_SPEND:`` — following
the existing marker-log convention (``EXTRACTION_RUNG:``, ``OPEN_BOUND_PILING:``).
All four workflow yamls tee stdout+stderr to a ``run_logs/`` artifact, so these
lines are durably grep-able per run; no extra artifact plumbing is needed.

The end-of-run check also reports whether the DONATED key's remaining balance
(``limit_remaining``) fell below ``OPENROUTER_CREDIT_FLOOR_USD``. cli.main uses
that to exit non-zero AFTER all forecasting/publishing completes — a
reminder-to-refill signal, never an abort — and only while credit alerting is
active (``constants.credit_alerts_active``; suppressed until
``CREDIT_ALERT_RESUME_DATE`` while the operator self-funds the season). The
suppression is purely an exit-status decision made in cli.main: this module
always reports the breach and always logs ``CREDIT_FLOOR_BREACH``.

Field semantics (verified against live /auth/key pulls, 2026-07-17): ``usage``
counts only spend billed as native OpenRouter credits. Spend routed through
BYOK provider integrations (the donated Metaculus key routes nearly everything
this way) lands in ``byok_usage`` instead, so ``usage`` can sit frozen while
real money burns. ``limit_remaining = limit - usage - byok_usage`` (when
``include_byok_in_limit``), making it the only field that reliably tracks
total spend on a limit-bearing key. Per-run spend therefore comes from the
``limit_remaining`` delta when the key reports one, with the ``usage`` delta
as the fallback for uncapped keys (personal: ``limit_remaining`` is null and
spend does land in ``usage``).

This module also owns the DRAINED-vs-REVOKED discriminator
(``classify_donated_key_state``) that ``fallback_openrouter`` consults when a
donated-key call fails with OpenRouter's spend-cap 403. Same endpoint, same
parser, so the "how much is left on the donated key" question has one
implementation.

Telemetry must never fail or block a run: every fetch error is logged as a
WARNING and treated as "unknown", and unknown never triggers the floor exit.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx

from metaculus_bot.check_openrouter_credits import KEY_SPECS, fetch_auth_key
from metaculus_bot.constants import CREDIT_ALERT_RESUME_DATE, OPENROUTER_CREDIT_FLOOR_USD

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeyBalanceSnapshot:
    """One key's balance at a point in time. ``None`` fields = not reported."""

    alias: str
    remaining_usd: float | None  # limit_remaining; None for uncapped keys (personal)
    usage_usd: float | None  # lifetime native-credit usage; excludes BYOK-routed spend


def _as_float(value: Any) -> float | None:
    """Coerce a reported balance field to a usable float, or ``None`` for "not reported".

    Non-finite is rejected along with unparseable, and that is load-bearing rather than
    tidy. ``json.loads`` accepts bare ``NaN`` / ``Infinity`` as an extension, every float
    comparison against NaN is False, and the classification ladder below reads a chain of
    such comparisons — so a NaN balance walked past ``limit <= 0`` and ``remaining > 0``
    into DRAINED, the one state exempt from CI alerting. Failing to "not reported" routes
    it to UNKNOWN instead, which stays red, and keeps a NaN out of the CREDIT_SPEND /
    CREDIT_BALANCE marker lines where it would just be misinformation.
    """
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _run_delta_usd(start: KeyBalanceSnapshot | None, end: KeyBalanceSnapshot) -> float | None:
    """Per-run spend for one key, or None when it can't be computed.

    Prefers the ``limit_remaining`` drop (start - end): on limit-bearing keys
    it is the only field that includes BYOK-routed spend (see module docstring).
    Falls back to the ``usage`` delta for uncapped keys, which report no
    ``limit_remaining`` but do accrue ``usage``.
    """
    if start is None:
        return None
    if start.remaining_usd is not None and end.remaining_usd is not None:
        return start.remaining_usd - end.remaining_usd
    if start.usage_usd is not None and end.usage_usd is not None:
        return end.usage_usd - start.usage_usd
    return None


def _fetch_snapshot(alias: str, phase: str) -> KeyBalanceSnapshot | None:
    """Fetch one key's balance; on ANY failure, warn and return None.

    A missing env var or endpoint hiccup must never fail the run (this is telemetry), so we
    log and continue. The catch is deliberately total rather than a curated tuple: cli.main
    calls ``log_end_and_check_floor`` from a ``finally``, so an escape there replaces
    whatever the run was already raising and takes the whole end-of-run diagnostic surface
    with it (report summary, alertable arithmetic, deprecation tripwire — all downstream).
    A narrow tuple already missed three real shapes: ``FileNotFoundError`` from a stale
    ``SSL_CERT_FILE``, ``httpx.InvalidURL`` (not an ``httpx.HTTPError`` subclass), and the
    ``RuntimeError`` this repo's own autouse network guard raises.
    """
    env_var, _ = KEY_SPECS[alias]
    api_key = os.getenv(env_var)
    if not api_key:
        logger.warning("CREDIT_BALANCE: key=%s phase=%s skipped (env var %s not set)", alias, phase, env_var)
        return None
    try:
        data = fetch_auth_key(api_key)
        # Build the snapshot INSIDE the try: fetch_auth_key returns
        # ``payload.get("data", payload)``, so a 200 whose body carries a
        # non-mapping ``data`` (``{"data": null}`` / ``{"data": [...]}``) yields a
        # non-dict here, and ``data.get(...)`` then raises AttributeError. Keeping
        # the .get() calls under the try means that malformed-but-200 case degrades
        # to a WARNING + None like any other fetch failure, never crashing the run.
        return KeyBalanceSnapshot(
            alias=alias,
            remaining_usd=_as_float(data.get("limit_remaining")),
            usage_usd=_as_float(data.get("usage")),
        )
    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except
        # Broad by design, not defensiveness: the module contract is that telemetry cannot
        # fail a run, and this sits under a cli.main ``finally``. See the docstring.
        logger.warning(
            "CREDIT_BALANCE: key=%s phase=%s fetch failed (%s); continuing without balance telemetry",
            alias,
            phase,
            type(exc).__name__,
        )
        return None


class CreditTelemetry:
    """Start/end balance logging + donated-key floor check for one bot run."""

    def __init__(self, floor_usd: float = OPENROUTER_CREDIT_FLOOR_USD) -> None:
        self._floor_usd = floor_usd
        self._start: dict[str, KeyBalanceSnapshot] = {}

    def log_start(self) -> None:
        for alias in KEY_SPECS:
            snapshot = _fetch_snapshot(alias, phase="start")
            if snapshot is None:
                continue
            self._start[alias] = snapshot
            logger.info(
                "CREDIT_BALANCE: key=%s phase=start remaining=%s usage=%s",
                alias,
                _fmt(snapshot.remaining_usd),
                _fmt(snapshot.usage_usd),
            )

    def log_end_and_check_floor(self) -> bool:
        """Log end balances + per-run spend; return True iff the donated key's
        remaining balance is KNOWN and below the floor (unknown never trips it).

        Spend delta prefers the ``limit_remaining`` drop (start - end): on
        limit-bearing keys it is the only field covering BYOK-routed spend,
        which the donated key routes nearly everything through (``usage`` sat
        frozen at $4.16 across a $3.34 run, 2026-07-17). Uncapped keys report
        no ``limit_remaining``, so they fall back to the ``usage`` delta.
        Caveats: an out-of-band top-up mid-run skews the remaining-based delta
        (rare; per-run spend is indicative anyway), and OpenRouter caches these
        values briefly — don't build on exact figures.
        """
        donated_below_floor = False
        for alias in KEY_SPECS:
            snapshot = _fetch_snapshot(alias, phase="end")
            if snapshot is None:
                continue
            logger.info(
                "CREDIT_BALANCE: key=%s phase=end remaining=%s usage=%s",
                alias,
                _fmt(snapshot.remaining_usd),
                _fmt(snapshot.usage_usd),
            )
            run_delta = _run_delta_usd(self._start.get(alias), snapshot)
            logger.info(
                "CREDIT_SPEND: key=%s run_delta_usd=%s remaining=%s",
                alias,
                _fmt(run_delta),
                _fmt(snapshot.remaining_usd),
            )
            if alias == "donated" and snapshot.remaining_usd is not None and snapshot.remaining_usd < self._floor_usd:
                logger.warning(
                    "CREDIT_FLOOR_BREACH: key=donated remaining=%s floor=%s — donated OpenRouter "
                    "balance needs a top-up; run completed normally. cli.main logs the exit "
                    "decision unless a higher-priority degradation alert exits first.",
                    _fmt(snapshot.remaining_usd),
                    _fmt(self._floor_usd),
                )
                donated_below_floor = True
        return donated_below_floor


# --- Drained-vs-revoked discriminator for the donated key -------------------
#
# OpenRouter reports a breached per-key spend cap as HTTP 403 with the text
# "Key limit exceeded (total limit)", and reports a revoked key as HTTP 401 on
# an LLM call. But a key that Metaculus RE-CAPPED TO ZERO produces the exact
# same 403 text as a key that simply spent its whole allocation, and the
# operator wants opposite CI colors for those: a genuinely drained key is the
# expected state while they self-fund the season (green), a zeroed or revoked
# one is real breakage (red). No amount of text matching can separate them, so
# we ask the free, read-only /auth/key endpoint what the cap actually looks
# like. See ``fallback_openrouter.is_suppressible_credit_error``.


class DonatedKeyState(StrEnum):
    """What ``/auth/key`` says about the donated key, in alerting terms.

    Only ``DRAINED`` is the expected empty wallet. Every other state means the
    "expected empty wallet" explanation does NOT hold, so the run stays alertable
    — including ``UNKNOWN``, which is how every probe failure classifies. Failing
    safe matters more here than being informative: a broken probe must never be
    able to silently turn a red run green.
    """

    DRAINED = "drained"  # positive cap, nothing left — spent its allocation
    ZEROED = "zeroed"  # cap itself is 0 — Metaculus cut us off, never an "empty wallet"
    REVOKED = "revoked"  # key rejected (401/404) — gone, not empty
    FUNDED = "funded"  # money remains, so the failure was not about credit at all
    UNKNOWN = "unknown"  # probe could not answer (no key configured, endpoint error, odd payload)


# Shorter than ``AUTH_KEY_REQUEST_TIMEOUT_S`` by design: this probe can fire mid-run, so it
# must not be able to stall a forecast. The shared ``fetch_auth_key`` default is fine for
# the CLI and the start/end telemetry, which run outside the forecasting window.
#
# Read as PER NETWORK OPERATION, not as a bound on total elapsed time. httpx applies a bare
# float to connect / read / write / pool independently, so a server trickling bytes slower
# than the read timeout resets the clock on every chunk and the call can run many multiples
# of this budget (measured against a local trickling server, a one-second timeout took ten
# seconds to return twenty bytes). The hard total cap lives at the latency-sensitive call
# site (``fallback_openrouter.record_donated_key_fallback`` wraps the probe in
# ``asyncio.wait_for``), because that is the only hop where the promise has to hold.
DONATED_KEY_PROBE_TIMEOUT_S: float = 5.0

# One probe per process, lock-guarded so concurrent callers share one verdict. A run that
# loses every donated-key call would otherwise fire one HTTP request per failure, and
# caching failures matters as much as caching verdicts (a dead endpoint would otherwise
# cost one timeout per failed call). ``None`` means "never probed", which cli renders
# differently from any verdict.
#
# ``threading``, not ``asyncio``: every production caller arrives on an
# ``asyncio.to_thread`` worker (see ``fallback_openrouter.record_donated_key_fallback``),
# so the contention is between real OS threads. The lock is what makes the ONE VERDICT
# part true, which matters more than the one-request part — without it each caller keeps
# its own probe result, so an intermittently failing ``/auth/key`` splits a single
# drained-key incident into some suppressed and some alertable events, and cli then exits
# red on the very condition the suppression window exists for.
_probed_donated_key_state: DonatedKeyState | None = None
_PROBE_LOCK = threading.Lock()


def get_probed_donated_key_state() -> DonatedKeyState | None:
    """The cached verdict, or ``None`` if nothing this run needed to probe."""
    return _probed_donated_key_state


def reset_donated_key_state_cache() -> None:
    """Clear the cached verdict. Used by tests; not for production code."""
    global _probed_donated_key_state
    _probed_donated_key_state = None


def _probe_donated_key_state() -> DonatedKeyState:
    """One ``/auth/key`` read on the donated key, classified.

    Returns UNKNOWN on any failure and logs the exception. UNKNOWN is the fail-safe
    direction: it keeps the run alertable, so a probe that cannot answer never greens a red
    run. ``fallback_openrouter`` guards its own call site too (an escape there would abort
    the fallback it is annotating), but the contract has to hold here for the next caller.
    """
    env_var, _ = KEY_SPECS["donated"]
    api_key = os.getenv(env_var)
    if not api_key:
        # No donated key configured, so there is no donated wallet to be empty.
        # Returning early also keeps this probe free of network I/O in tests that
        # don't stub it.
        return DonatedKeyState.UNKNOWN
    try:
        data = fetch_auth_key(api_key, timeout=DONATED_KEY_PROBE_TIMEOUT_S)
        # Read the fields INSIDE the try for the same reason ``_fetch_snapshot``
        # does: a 200 whose body carries a non-mapping ``data`` makes ``.get``
        # raise AttributeError, and that must degrade to UNKNOWN like any other
        # inconclusive answer.
        limit_usd = _as_float(data.get("limit"))
        remaining_usd = _as_float(data.get("limit_remaining"))
    except httpx.HTTPStatusError as exc:
        # 401 = key rejected, 404 = key no longer exists. Anything else (429 on the
        # balance endpoint, a 5xx) tells us nothing about the wallet.
        if exc.response.status_code in (401, 404):
            return DonatedKeyState.REVOKED
        return DonatedKeyState.UNKNOWN
    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except
        # Broad by design: see the docstring. A curated tuple already missed
        # FileNotFoundError (stale SSL_CERT_FILE), httpx.InvalidURL (not an HTTPError
        # subclass) and RuntimeError (the suite's network guard).
        logger.exception("DONATED_KEY_STATE: /auth/key probe failed; classifying as unknown (stays alertable)")
        return DonatedKeyState.UNKNOWN

    if limit_usd is None or remaining_usd is None:
        # An uncapped key has no cap to exceed, so a spend-cap failure on one is
        # unexplained rather than expected.
        return DonatedKeyState.UNKNOWN
    if limit_usd <= 0:
        return DonatedKeyState.ZEROED
    if remaining_usd > 0:
        return DonatedKeyState.FUNDED
    # OpenRouter clamps ``limit_remaining`` at 0 even when the true arithmetic is
    # negative (live: limit=850, usage=4.39, byok_usage=846.42 → reported 0.00), so
    # ``<= 0`` rather than ``== 0``.
    return DonatedKeyState.DRAINED


def classify_donated_key_state() -> DonatedKeyState:
    """Whether the donated key is genuinely drained, or broken in some other way.

    Blocking HTTP, so callers on the event loop should hand this to a thread. Probes at
    most once per process (lock-guarded, so concurrent callers share one verdict); every
    subsequent call reads the cache.
    """
    global _probed_donated_key_state
    cached = _probed_donated_key_state
    if cached is not None:
        return cached

    with _PROBE_LOCK:
        # Re-check inside the lock: a caller that queued behind the winner must take the
        # winner's verdict rather than probe again with its own.
        cached = _probed_donated_key_state
        if cached is not None:
            return cached

        state = _probe_donated_key_state()
        _probed_donated_key_state = state
        # Logged inside the lock so the marker line appears exactly once per run — N copies
        # of one verdict would read as N separate probes to whoever greps the run log.
        if state is DonatedKeyState.DRAINED:
            logger.info(
                "DONATED_KEY_STATE: state=%s — the donated OpenRouter key spent its whole allocation "
                "with the cap itself intact. Expected while the operator self-funds the season, so "
                "credit-caused personal-key fallbacks are exempt from alerting until %s.",
                state.value,
                CREDIT_ALERT_RESUME_DATE.isoformat(),
            )
        else:
            logger.warning(
                "DONATED_KEY_STATE: state=%s — a credit-shaped donated-key failure that is NOT an "
                "expected drained wallet (zeroed = cap set to 0, revoked = key rejected, funded = the "
                "key still has money so the failure was not about credit, unknown = the probe could "
                "not answer). Personal-key fallbacks stay alertable, so this run will exit non-zero.",
                state.value,
            )
        return state
