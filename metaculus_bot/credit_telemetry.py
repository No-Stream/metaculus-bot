"""Per-run OpenRouter credit-balance telemetry.

Fetches key balances (donated + personal) at run start and end and emits
greppable marker lines — ``CREDIT_BALANCE:`` / ``CREDIT_SPEND:`` — following
the existing marker-log convention (``EXTRACTION_RUNG:``, ``OPEN_BOUND_PILING:``).
All four workflow yamls tee stdout+stderr to a ``run_logs/`` artifact, so these
lines are durably grep-able per run; no extra artifact plumbing is needed.

The end-of-run check also reports whether the DONATED key's remaining balance
(``limit_remaining``) fell below ``OPENROUTER_CREDIT_FLOOR_USD``. cli.main uses
that to exit non-zero AFTER all forecasting/publishing completes — a
reminder-to-refill signal, never an abort.

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

Telemetry must never fail or block a run: every fetch error is logged as a
WARNING and treated as "unknown", and unknown never triggers the floor exit.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

from metaculus_bot.check_openrouter_credits import KEY_SPECS, fetch_auth_key
from metaculus_bot.constants import OPENROUTER_CREDIT_FLOOR_USD

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KeyBalanceSnapshot:
    """One key's balance at a point in time. ``None`` fields = not reported."""

    alias: str
    remaining_usd: float | None  # limit_remaining; None for uncapped keys (personal)
    usage_usd: float | None  # lifetime native-credit usage; excludes BYOK-routed spend


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    """Fetch one key's balance; on any expected failure, warn and return None.

    A missing env var or endpoint hiccup must never fail the run (this is
    telemetry), so we log and continue. AttributeError covers a payload that
    isn't the expected dict shape.
    """
    env_var, _ = KEY_SPECS[alias]
    api_key = os.getenv(env_var)
    if not api_key:
        logger.warning("CREDIT_BALANCE: key=%s phase=%s skipped (env var %s not set)", alias, phase, env_var)
        return None
    try:
        data = fetch_auth_key(api_key)
    except (httpx.HTTPError, ValueError, KeyError, AttributeError) as exc:
        logger.warning(
            "CREDIT_BALANCE: key=%s phase=%s fetch failed (%s); continuing without balance telemetry",
            alias,
            phase,
            type(exc).__name__,
        )
        return None
    return KeyBalanceSnapshot(
        alias=alias,
        remaining_usd=_as_float(data.get("limit_remaining")),
        usage_usd=_as_float(data.get("usage")),
    )


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
                    "balance needs a top-up; run completed normally but will exit non-zero so CI flags it.",
                    _fmt(snapshot.remaining_usd),
                    _fmt(self._floor_usd),
                )
                donated_below_floor = True
        return donated_below_floor
