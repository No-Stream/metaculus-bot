"""Per-run OpenRouter credit-balance telemetry, plus the per-role dollar ledger.

Fetches key balances (donated + personal) at run start and end and emits
greppable marker lines — ``CREDIT_BALANCE:`` / ``CREDIT_SPEND:`` — following
the existing marker-log convention (``EXTRACTION_RUNG:``, ``OPEN_BOUND_PILING:``).
All four workflow yamls tee stdout+stderr to a ``run_logs/`` artifact, so these
lines are durably grep-able per run; no extra artifact plumbing is needed.

The per-key deltas say WHAT a run cost; the ``CREDIT_ROLE_SPEND:`` lines at the
bottom of this module say WHERE it went (forecaster slot, research stage, parser,
...), read off OpenRouter's own per-call usage accounting. See "Per-role dollar
attribution" below.

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

THE PERSONAL KEY'S PER-RUN DELTA IS A LOWER BOUND, AND THE CAUSE IS SETTLEMENT
LAG — NOT BYOK. Worth stating flatly because the BYOK paragraph above is the
wrong explanation for it and misled two separate investigations: on the personal
key ``usage`` genuinely does climb ($154.58 -> $160.24 over 2026-07-20..27), so
nothing is hiding in ``byok_usage``. What happens is that OpenRouter has not
booked the run's spend by the time the end snapshot fires, seconds after the last
call. Measured over ``backtests/telemetry_archive/credit_balance.jsonl``, 178
paired personal-key runs: the within-run deltas summed to $3.31 against $5.66 of
true lifetime-usage growth (58% captured), and 160 of 178 runs reported exactly
$0.00. The missing $2.35 is fully accounted for by the gap between each run's
``phase=end`` usage and the NEXT run's ``phase=start`` usage — $3.31 + $2.35 =
$5.66, exactly, to the cent. The money is late, not lost.

The tightest version of the evidence, restricted to runs that DEMONSTRABLY spent:
of the 25 paired runs carrying at least one ``extraction_rung`` record (a forecast
provably happened, and ``gemini-3.1-pro-preview`` — the slot pinned to the
personal key — produced one in all 25), 7 reported exactly $0.00. That is a 28%
false-zero rate on runs that cannot have been free.
``scripts/reconcile_credit_spend.py`` recovers a real figure for all 7
($0.10-$0.32 each), which is the direct demonstration that the zeros are lag
rather than absence.

There is deliberately no wait-and-re-read here. The earliest CONFIRMED settlement
in the archive is 153s after the end snapshot and the median is ~25 minutes, so a
delay short enough to sit in cli.main's ``finally`` (where telemetry must never
stall a run) is below anything the data can show would work — it would be an
unverifiable guess that also slowed every run. Instead the marker states its own
source (``source=usage_delta_unsettled``) and a sibling
``CREDIT_SPEND_UNSETTLED`` WARNING says the figure is a floor, so a ``0.00`` can
never be misread as "this run was free". The settled per-run number is recovered
after the fact by ``scripts/reconcile_credit_spend.py``, which differences each
run's start usage against its successor's — the only place the lag is actually
observable.

This module also owns the DRAINED-vs-REVOKED discriminator
(``classify_donated_key_state``) that ``fallback_openrouter`` consults when a
donated-key call fails with OpenRouter's spend-cap 403. Same endpoint, same
parser, so the "how much is left on the donated key" question has one
implementation.

Telemetry must never fail or block a run: every fetch error is logged as a
WARNING and treated as "unknown", and unknown never triggers the floor exit.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import httpx
import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

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


# How a CREDIT_SPEND delta was derived, which is what tells a reader how much to
# trust it. Emitted as ``source=`` on the marker line.
SPEND_SOURCE_REMAINING: str = "remaining_delta"
SPEND_SOURCE_USAGE: str = "usage_delta_unsettled"
SPEND_SOURCE_NONE: str = "unavailable"


def _run_delta_usd(start: KeyBalanceSnapshot | None, end: KeyBalanceSnapshot) -> tuple[float | None, str]:
    """Per-run spend for one key plus the SOURCE it came from.

    Returns ``(delta, source)``. The source is the point: the two branches have
    very different trustworthiness and the number alone cannot tell them apart.

    * ``remaining_delta`` (start - end) — a limit-bearing key's ``limit_remaining``
      drop. Reliable: it is the only field covering BYOK-routed spend, which the
      donated key routes nearly everything through.
    * ``usage_delta_unsettled`` (end - start) — the fallback for uncapped keys,
      which report no ``limit_remaining``. Systematically UNDER-reports, because
      ``usage`` lags the run (see ``_run_delta_usd``'s settlement note in the module
      docstring). A ``0.00`` from this branch does NOT mean no spend.
    * ``unavailable`` — no start snapshot, or neither field pair is reported.
    """
    if start is None:
        return None, SPEND_SOURCE_NONE
    if start.remaining_usd is not None and end.remaining_usd is not None:
        return start.remaining_usd - end.remaining_usd, SPEND_SOURCE_REMAINING
    if start.usage_usd is not None and end.usage_usd is not None:
        return end.usage_usd - start.usage_usd, SPEND_SOURCE_USAGE
    return None, SPEND_SOURCE_NONE


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
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except
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

        Every ``CREDIT_SPEND`` line carries ``source=`` naming which branch produced
        it, and the ``usage``-delta branch additionally logs
        ``CREDIT_SPEND_UNSETTLED`` — that branch systematically under-reports
        because of settlement lag (module docstring has the measurements), so the
        number is a floor and a ``0.00`` is not evidence of no spend.

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
            run_delta, spend_source = _run_delta_usd(self._start.get(alias), snapshot)
            logger.info(
                "CREDIT_SPEND: key=%s run_delta_usd=%s remaining=%s source=%s",
                alias,
                _fmt(run_delta),
                _fmt(snapshot.remaining_usd),
                spend_source,
            )
            if spend_source == SPEND_SOURCE_USAGE:
                # Say it inline rather than only in the docs. This is the number
                # watching the operator's monthly cap on a self-funded key, and a
                # bare "0.00" reads as "this run was free" when it actually means
                # "OpenRouter had not settled the spend yet". Measured across 178
                # archived personal-key runs: the within-run deltas summed to 58% of
                # the true growth, and 7 of 25 runs that demonstrably forecast
                # reported exactly 0.00.
                logger.warning(
                    "CREDIT_SPEND_UNSETTLED: key=%s run_delta_usd=%s is a LOWER BOUND — %s reports no "
                    "limit_remaining, so this is a lifetime-usage delta and OpenRouter has typically "
                    "not settled the run's spend by now. Do not read 0.00 as no spend; reconcile "
                    "against the NEXT run's phase=start usage (see scripts/reconcile_credit_spend.py).",
                    alias,
                    _fmt(run_delta),
                    alias,
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
    global _probed_donated_key_state  # noqa: PLW0603  # once-per-process probe cache is the design
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
    global _probed_donated_key_state  # noqa: PLW0603  # once-per-process probe cache is the design
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


# --- Per-role dollar attribution ---------------------------------------------
#
# WHY. The per-key deltas above cannot say which ROLE spent the money, and every cost
# argument in the 2026-08-31 gemini-slot review was blocked on exactly that: the measured
# $0.38-0.41/question could not be split into forecaster vs research vs ranker, so "a 4th
# member costs +33%" stayed an assertion (scratch/residual_2026-08-31/gemini_review/
# RECOMMENDATION.md §3, §4 item 4).
#
# SOURCE OF TRUTH: OpenRouter's own per-call usage accounting, not litellm's price table.
# OpenRouter returns a ``usage`` object on every completion (usage accounting is on by
# default per openrouter.ai/docs/use-cases/usage-accounting; litellm 1.92's OpenRouter
# transformation also sends ``usage: {include: true}`` explicitly) carrying
#   * ``cost``: "The total amount charged to your account" — the credits drawn from the
#     key. Off BYOK routing this is the whole charge; on BYOK routing it is only
#     OpenRouter's platform fee (5% of list price, waived under a monthly allowance).
#   * ``cost_details.upstream_inference_cost``: "The actual cost charged by the upstream
#     AI provider", BYOK requests only, ``0``/``null`` otherwise.
# The donated Metaculus key routes OpenAI/Anthropic/Google through Metaculus's BYOK
# integrations, so on that key nearly everything lands in ``upstream_inference_cost`` —
# which is also what ``/auth/key`` books as ``byok_usage`` and subtracts from
# ``limit_remaining`` (module docstring: a $3.34 run left ``usage`` frozen). The personal
# key is not BYOK, so its whole bill is ``cost``. Summing the two therefore gives one
# number, on either key, that maps onto what ``CREDIT_SPEND`` measures.
#
# WHY A litellm CALLBACK. forecasting-tools' ``GeneralLlm.invoke`` returns only the text;
# the ``TextTokenCostResponse`` it builds keeps litellm's ``response_cost`` hidden param
# (= ``usage.cost`` via litellm's header lift, i.e. ~$0 for every BYOK call) and drops the
# usage object itself. The one seam that still sees the raw ``ModelResponse`` — and so
# ``cost_details`` — is litellm's success callback, which is also how forecasting-tools'
# own ``LitellmCostTracker`` works. ``litellm.Usage`` keeps every extra field the body
# carried as an attribute, so ``response.usage.get("cost_details")`` reads straight off it.
#
# WHY ``metadata=``. ``metadata`` is a litellm-only kwarg: it lands in
# ``litellm_params["metadata"]`` for callbacks and is never forwarded to OpenRouter
# (litellm forwards it to a provider only for OpenAI under ``enable_preview_features``).
# ``GeneralLlm`` passes unknown kwargs through to ``acompletion`` unchanged, so a
# ``metadata=llm_call_metadata(role, key_alias)`` stamped at construction reaches every
# completion that LLM makes. The raw ``acompletion`` path in ``research/agentic/llm.py``
# stamps the same dict per call.
#
# THREADING. The callback runs on the event loop inside litellm's logging worker, and the
# accumulation below has no ``await``, so the ledger needs no lock — the same
# bytecode-atomic argument ``fallback_openrouter.record_donated_key_fallback`` makes for
# the fallback counters. Only ``async_log_success_event`` is implemented: litellm skips
# the sync ``log_success_event`` for ``acompletion`` unless a sync-only callback is
# registered, and implementing both would double-count.

# litellm ``metadata=`` keys the ledger reads back. Distinct from the ``KEY_SPECS``
# aliases below on purpose: these name FIELDS, those name KEYS.
ROLE_METADATA_KEY: str = "role"
KEY_ALIAS_METADATA_KEY: str = "key_alias"

# Which OpenRouter key a completion billed. ``donated`` / ``personal`` are the
# ``KEY_SPECS`` aliases, so ``CREDIT_ROLE_SPEND key=`` joins onto ``CREDIT_SPEND key=``
# and ``CREDIT_BALANCE key=``. ``direct`` is a non-OpenRouter slug (a perplexity/ or
# exa/ model billed to its own provider key, outside this ledger's remit but still
# counted); ``unknown`` is a completion that carried no key tag at all.
DONATED_KEY_ALIAS: str = "donated"
PERSONAL_KEY_ALIAS: str = "personal"
DIRECT_KEY_ALIAS: str = "direct"
UNKNOWN_KEY_ALIAS: str = "unknown"

# A completion nobody tagged: forecasting-tools' own helpers (SmartSearcher), an ablation
# or benchmark harness, or a builder call site that forgot ``role=``. Visible on purpose.
UNTAGGED_ROLE: str = "untagged"

# How long cli.main may wait for litellm's logging worker to deliver the last success
# callbacks before the ledger is logged. Telemetry must never stall the end of a run, and
# the bound is reachable two ways, not one: a wedged worker (a worker loop that dies on
# any non-``CancelledError`` leaves ``queue.join()`` outstanding forever), AND a single
# callback slower than 10s — litellm allows each queued coroutine 20s
# (``LOGGING_WORKER_MAX_TIME_PER_COROUTINE``), twice this window, so a callback litellm
# still considers healthy trips us. Left at 10.0 deliberately: both callbacks we register
# are in-memory arithmetic, so raising it is an unverified retune that would only lengthen
# a pointless wait on a dead worker. Exceeding it costs the last few completions' rows,
# never the run (see ``drain_litellm_callbacks``).
LITELLM_CALLBACK_DRAIN_TIMEOUT_S: float = 10.0


def llm_call_metadata(role: str | None, key_alias: str) -> dict[str, str]:
    """The litellm ``metadata=`` payload that tags every completion for the role ledger.

    Roles in use (descriptive, one per spend line; ``forecaster:<vendor>`` for the
    roster slots, derived from the slug by ``llm_configs.forecaster_role`` so a roster
    swap cannot mislabel a slot): ``forecaster:openai`` / ``forecaster:anthropic`` / ``forecaster:google``,
    ``stacker``, ``stacker_fallback``, ``parser``, ``summarizer``, ``crux_analyzer``,
    ``native_search``, ``targeted_search``, ``gap_fill_analyzer``, ``gap_fill_resolver``,
    ``gap_fill_v2_driver``, ``market_query_author``, ``market_ranker``,
    ``financial_classifier``, ``perplexity_research``. ``role=None`` tags ``untagged`` HERE,
    at construction, so every metaculus_bot-built LLM carries an explicit token and an
    ``untagged`` row in a run log means one builder call site forgot its ``role=``.

    Not on OpenRouter, so never in this ledger: the Gemini grounded-search provider and
    gap-fill v2's ``read_document`` (google-genai on the personal Google AI Studio key),
    AskNews (subscription), Exa (``search_web``).
    """
    return {ROLE_METADATA_KEY: role or UNTAGGED_ROLE, KEY_ALIAS_METADATA_KEY: key_alias}


def plain_llm_key_alias(model: str) -> str:
    """Which key a plain ``GeneralLlm`` (no explicit ``api_key``) bills for ``model``.

    litellm reads ``OPENROUTER_API_KEY`` from the environment for ``openrouter/`` slugs —
    the personal key, since the donated key is only ever passed explicitly. Any other slug
    goes to its own provider's key.
    """
    return PERSONAL_KEY_ALIAS if model.startswith("openrouter/") else DIRECT_KEY_ALIAS


@dataclass
class _RoleSpendAccumulator:
    calls: int = 0
    costed_calls: int = 0
    usd: float = 0.0
    byok_usd: float = 0.0


@dataclass(frozen=True)
class RoleSpendRow:
    """One ``CREDIT_ROLE_SPEND`` line. ``usd`` / ``byok_usd`` are ``None`` when no call
    carried cost data (rendered ``n/a``), never a fabricated zero."""

    role: str
    key_alias: str
    calls: int
    costed_calls: int
    usd: float | None
    byok_usd: float | None


_role_spend: dict[tuple[str, str], _RoleSpendAccumulator] = {}


def record_llm_call_spend(
    role: str, key_alias: str, *, cost_usd: float | None, byok_upstream_usd: float | None
) -> None:
    """Add one successful completion to the ledger.

    ``cost_usd`` is OpenRouter's ``usage.cost`` (credits drawn from the key) and
    ``byok_upstream_usd`` its ``cost_details.upstream_inference_cost`` (the provider's
    charge on a BYOK route). A call with neither is counted but not costed. Synchronous
    and await-free by design — see the THREADING note above.
    """
    accumulator = _role_spend.setdefault((role, key_alias), _RoleSpendAccumulator())
    accumulator.calls += 1
    if cost_usd is None and byok_upstream_usd is None:
        return
    accumulator.costed_calls += 1
    accumulator.usd += (cost_usd or 0.0) + (byok_upstream_usd or 0.0)
    accumulator.byok_usd += byok_upstream_usd or 0.0


def role_spend_rows() -> list[RoleSpendRow]:
    """The ledger as rows, biggest spender first; uncosted rows last, then by role."""
    rows = [
        RoleSpendRow(
            role=role,
            key_alias=key_alias,
            calls=acc.calls,
            costed_calls=acc.costed_calls,
            usd=acc.usd if acc.costed_calls else None,
            byok_usd=acc.byok_usd if acc.costed_calls else None,
        )
        for (role, key_alias), acc in _role_spend.items()
    ]
    return sorted(rows, key=lambda row: (row.usd is None, -(row.usd or 0.0), row.role, row.key_alias))


def reset_role_spend() -> None:
    """Empty the ledger. Used by tests; not for production code."""
    _role_spend.clear()


def _fmt_usd(value: float | None) -> str:
    # Four decimals, not the balance lines' two: per-role figures are fractions of a
    # cent per call (the parser is ~$0.0005/question).
    return "n/a" if value is None else f"{value:.4f}"


def log_role_spend() -> None:
    """Emit one ``CREDIT_ROLE_SPEND`` line per (role, key) beside the ``CREDIT_SPEND`` lines.

    An empty ledger still logs a line, so a run with zero completions is distinguishable
    from one that died before reaching the end-of-run block — but not in the row shape,
    so the harvester cannot mistake it for a row.
    """
    rows = role_spend_rows()
    if not rows:
        logger.info("CREDIT_ROLE_SPEND: no successful LLM completions reached the litellm success callback this run")
        return
    for row in rows:
        logger.info(
            "CREDIT_ROLE_SPEND: role=%s key=%s usd=%s calls=%d costed_calls=%d byok_usd=%s",
            row.role,
            row.key_alias,
            _fmt_usd(row.usd),
            row.calls,
            row.costed_calls,
            _fmt_usd(row.byok_usd),
        )


def _openrouter_usage_cost(response_obj: Any) -> tuple[float | None, float | None]:
    """``(usage.cost, usage.cost_details.upstream_inference_cost)`` off a litellm response,
    each ``None`` when unreported (or non-finite, same rule as ``_as_float``)."""
    usage = getattr(response_obj, "usage", None)
    if usage is None:
        return None, None
    cost_details = usage.get("cost_details") or {}
    return _as_float(usage.get("cost")), _as_float(cost_details.get("upstream_inference_cost"))


class RoleSpendTracker(CustomLogger):
    """litellm success callback feeding the role ledger. Install once via
    :func:`install_role_spend_tracker`."""

    async def async_log_success_event(
        self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        del start_time, end_time  # CustomLogger hook signature; the ledger is not timed
        metadata = (kwargs.get("litellm_params") or {}).get("metadata") or {}
        cost_usd, byok_upstream_usd = _openrouter_usage_cost(response_obj)
        record_llm_call_spend(
            metadata.get(ROLE_METADATA_KEY, UNTAGGED_ROLE),
            metadata.get(KEY_ALIAS_METADATA_KEY, UNKNOWN_KEY_ALIAS),
            cost_usd=cost_usd,
            byok_upstream_usd=byok_upstream_usd,
        )


def install_role_spend_tracker() -> None:
    """Register the tracker with litellm exactly once per process."""
    if any(isinstance(callback, RoleSpendTracker) for callback in litellm.callbacks):
        return
    litellm.callbacks.append(RoleSpendTracker())


async def drain_litellm_callbacks(timeout_s: float = LITELLM_CALLBACK_DRAIN_TIMEOUT_S) -> None:
    """Wait for litellm's logging worker to deliver every pending success callback.

    Must run INSIDE the event loop the completions ran on: the worker's queue is bound to
    that loop and is reset (dropping whatever is queued) when a different loop shows up.
    litellm enqueues each callback from a ``create_task``, so yield to the loop first —
    otherwise ``flush`` can find an empty queue with the enqueue still a tick away — then
    join the queue, bounded so telemetry can never hold the end of a run hostage.

    The bound is caught HERE, not at the call site, so every caller inherits the promise
    this docstring makes. Its one caller awaits this from a ``finally``
    (``cli._forecast_with_callback_drain``) and nothing between there and process exit
    catches, so a raise would discard a fully published run's reports and skip
    ``log_report_summary`` plus the whole degradation/exit block — the q45085 failure
    shape, on a run where every question published — or, on a run that was already
    failing, demote the real forecast error to ``__context__``. ``CancelledError`` is a
    ``BaseException`` and still propagates, so the GHA SIGTERM path is unaffected.
    """
    for _ in range(2):
        await asyncio.sleep(0)
    try:
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=timeout_s)
    except TimeoutError:
        # Distinct marker on purpose: the CREDIT_ROLE_SPEND harvester spec expects
        # role=/key=/usd=/calls= fields, so prose under that prefix would pollute every
        # grep of a run log without ever parsing as a row.
        logger.warning(
            "LITELLM_CALLBACK_DRAIN_TIMEOUT: litellm's logging worker did not deliver its queued "
            "success callbacks within %.1fs; continuing so the run can finish. The CREDIT_ROLE_SPEND "
            "ledger below may under-count this run's last completions.",
            timeout_s,
        )
