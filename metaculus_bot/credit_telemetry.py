"""Per-run OpenRouter credit-balance telemetry, plus the per-role dollar ledger.

Fetches both keys' balances at run start and end and emits greppable
``CREDIT_BALANCE`` / ``CREDIT_SPEND`` / ``CREDIT_FLOOR_BREACH`` markers into the
``run_logs/`` artifact every workflow tees. The per-key deltas say WHAT a run
cost; the ``CREDIT_ROLE_SPEND`` ledger at the bottom of this module says WHERE it
went, off OpenRouter's own per-call usage accounting.

Per-run spend reads the ``limit_remaining`` delta on a limit-bearing key, the only
field covering BYOK-routed spend, and falls back to the ``usage`` delta on an
uncapped one (the personal key). That fallback is a LOWER BOUND, since OpenRouter
has usually not settled the run's spend by the time the end snapshot fires, so a
``0.00`` is not evidence of no spend; there is deliberately no wait-and-re-read,
and ``scripts/reconcile_credit_spend.py`` recovers the settled figure afterwards.
A BYOK route on the personal key (the OpenAI slugs) is a separate blind spot: it
never reaches ``usage`` at all and is visible only on the role ledger.

The end-of-run check also reports whether the DONATED key's ``limit_remaining``
fell below ``OPENROUTER_CREDIT_FLOOR_USD``, an early-warning level and not an
empty tank; cli.main turns a breach into a non-zero exit after forecasting and
publishing finish, never an abort, and only while credit alerting is active. This
module also owns the drained-vs-revoked discriminator
(``classify_donated_key_state``) that ``fallback_openrouter`` consults on
OpenRouter's spend-cap 403. Telemetry must never fail or block a run: every fetch
error logs a WARNING and reads as "unknown", and unknown never trips the floor
exit.

Field semantics and the measurements: docs/operations.md "Credit telemetry and the refill floor".
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import httpx
import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

from metaculus_bot.check_openrouter_credits import KEY_SPECS, fetch_auth_key
from metaculus_bot.constants import (
    CREDIT_ALERT_RESUME_DATE,
    OPENROUTER_CREDIT_FLOOR_USD,
    PROMPT_TOKENS_ALERT_THRESHOLD,
    donated_openrouter_key_enabled,
)

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


# Emitted as ``source=`` on the CREDIT_SPEND line: how the delta was derived, hence how far to trust it.
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
      ``usage`` lags the run (docs/operations.md "Credit telemetry and the refill
      floor"). A ``0.00`` from this branch does NOT mean no spend.
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

    The donated key is skipped outright (one INFO line, no HTTP) while
    ``DONATED_OPENROUTER_KEY_ENABLED`` is off: a Mantic run never routes through that key, so
    its balance is not the run's business and the refill floor downstream must not fire on it.

    A missing env var or endpoint hiccup must never fail the run (this is telemetry), so we
    log and continue. The catch is deliberately total rather than a curated tuple: cli.main
    calls ``log_end_and_check_floor`` from a ``finally``, so an escape there replaces
    whatever the run was already raising and takes the whole end-of-run diagnostic surface
    with it (report summary, alertable arithmetic, deprecation tripwire — all downstream).
    A narrow tuple already missed three real shapes: ``FileNotFoundError`` from a stale
    ``SSL_CERT_FILE``, ``httpx.InvalidURL`` (not an ``httpx.HTTPError`` subclass), and the
    ``RuntimeError`` this repo's own autouse network guard raises.
    """
    if alias == DONATED_KEY_ALIAS and not donated_openrouter_key_enabled():
        logger.info("CREDIT_BALANCE: key=%s phase=%s skipped (donated routing disabled)", alias, phase)
        return None
    env_var, _ = KEY_SPECS[alias]
    api_key = os.getenv(env_var)
    if not api_key:
        logger.warning("CREDIT_BALANCE: key=%s phase=%s skipped (env var %s not set)", alias, phase, env_var)
        return None
    try:
        data = fetch_auth_key(api_key)
        # Built inside the try so a 200 carrying a non-mapping ``data`` degrades like any other fetch failure.
        return KeyBalanceSnapshot(
            alias=alias,
            remaining_usd=_as_float(data.get("limit_remaining")),
            usage_usd=_as_float(data.get("usage")),
        )
    except Exception as exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except: telemetry must never fail a run
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

        The spend delta prefers the ``limit_remaining`` drop (start - end) and falls back
        to the ``usage`` delta on an uncapped key. Every ``CREDIT_SPEND`` line names its
        branch in ``source=``, and the ``usage`` branch also logs
        ``CREDIT_SPEND_UNSETTLED`` because it under-reports. Field semantics, the
        measurements and the top-up and caching caveats: docs/operations.md "Credit
        telemetry and the refill floor".
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
                # Loud, not just a source= field: a bare 0.00 reads as "this run was free".
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
                    "balance is below the early-warning floor, so ask Metaculus for a top-up "
                    "before it runs dry; the key is not necessarily empty and the run completed "
                    "normally. cli.main logs the exit decision unless a higher-priority "
                    "degradation alert exits first.",
                    _fmt(snapshot.remaining_usd),
                    _fmt(self._floor_usd),
                )
                donated_below_floor = True
        return donated_below_floor


# --- Drained vs revoked donated key: docs/operations.md "What a dry donated key actually returns".


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


# Per-operation, not a total elapsed cap: docs/operations.md "What a dry donated key actually returns".
DONATED_KEY_PROBE_TIMEOUT_S: float = 5.0

# One verdict per process, lock-guarded; why: docs/operations.md "What a dry donated key actually returns".
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
        return DonatedKeyState.UNKNOWN
    try:
        data = fetch_auth_key(api_key, timeout=DONATED_KEY_PROBE_TIMEOUT_S)
        # Read inside the try like ``_fetch_snapshot``: a 200 with a non-mapping ``data`` degrades to UNKNOWN.
        limit_usd = _as_float(data.get("limit"))
        remaining_usd = _as_float(data.get("limit_remaining"))
    except httpx.HTTPStatusError as exc:
        # 401 = key rejected, 404 = key gone; any other status says nothing about the wallet.
        if exc.response.status_code in (401, 404):
            return DonatedKeyState.REVOKED
        return DonatedKeyState.UNKNOWN
    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except: a curated tuple already missed three real shapes
        logger.exception("DONATED_KEY_STATE: /auth/key probe failed; classifying as unknown (stays alertable)")
        return DonatedKeyState.UNKNOWN

    if limit_usd is None or remaining_usd is None:
        # An uncapped key has no cap to exceed, so a spend-cap failure on one is unexplained.
        return DonatedKeyState.UNKNOWN
    if limit_usd <= 0:
        return DonatedKeyState.ZEROED
    if remaining_usd > 0:
        return DonatedKeyState.FUNDED
    # OpenRouter clamps ``limit_remaining`` at 0 when the true arithmetic is negative, so drained is <= 0.
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
        # Re-check inside the lock: a caller queued behind the winner takes the winner's verdict.
        cached = _probed_donated_key_state
        if cached is not None:
            return cached

        state = _probe_donated_key_state()
        _probed_donated_key_state = state
        # Logged inside the lock so the marker appears exactly once per run, not once per caller.
        if state is DonatedKeyState.DRAINED:
            logger.info(
                "DONATED_KEY_STATE: state=%s — the donated OpenRouter key spent its whole allocation "
                "with the cap itself intact. Credit-caused personal-key fallbacks are exempt from "
                "alerting only while the dated suppression window is open, i.e. before %s; from that "
                "date on they redden CI like any other fallback.",
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


# --- Per-role dollar attribution. Semantics, the callback seam and threading: docs/operations.md "Per-role spend".

# These name litellm ``metadata=`` FIELDS; the ``KEY_SPECS`` aliases below name KEYS.
ROLE_METADATA_KEY: str = "role"
KEY_ALIAS_METADATA_KEY: str = "key_alias"
# Stamped per call by the one builder that knows its question (the v2 driver); absent on the roster LLMs.
QUESTION_METADATA_KEY: str = "question"

# The ``KEY_SPECS`` aliases verbatim, so ``CREDIT_ROLE_SPEND key=`` joins onto ``CREDIT_SPEND key=``.
DONATED_KEY_ALIAS: str = "donated"
PERSONAL_KEY_ALIAS: str = "personal"
DIRECT_KEY_ALIAS: str = "direct"
UNKNOWN_KEY_ALIAS: str = "unknown"

# A completion nobody tagged, kept visible on purpose rather than folded into another row.
UNTAGGED_ROLE: str = "untagged"

# What trips this bound, and why 10.0 rather than more: docs/operations.md "Per-role spend".
LITELLM_CALLBACK_DRAIN_TIMEOUT_S: float = 10.0


def llm_call_metadata(role: str | None, key_alias: str, *, question_ref: str | None = None) -> dict[str, str]:
    """The litellm ``metadata=`` payload that tags every completion for the role ledger.

    ``role=None`` tags ``untagged`` HERE, at construction, so an ``untagged`` row in a run
    log means one builder call site forgot its ``role=``. ``question_ref`` rides a third key
    when the caller knows it (the v2 driver builds its call per question; the roster
    ``GeneralLlm`` objects are built once per process and cannot) and ``PROMPT_SIZE_ALERT``
    reads it back. The role vocabulary and what never reaches this ledger:
    docs/operations.md "Per-role spend".
    """
    metadata = {ROLE_METADATA_KEY: role or UNTAGGED_ROLE, KEY_ALIAS_METADATA_KEY: key_alias}
    if question_ref is not None:
        metadata[QUESTION_METADATA_KEY] = question_ref
    return metadata


def plain_llm_key_alias(model: str) -> str:
    """Which key a plain ``GeneralLlm`` (no explicit ``api_key``) bills for ``model``.

    litellm reads ``OPENROUTER_API_KEY`` from the environment for ``openrouter/`` slugs —
    the personal key, since the donated key is only ever passed explicitly. Any other slug
    goes to its own provider's key.
    """
    return PERSONAL_KEY_ALIAS if model.startswith("openrouter/") else DIRECT_KEY_ALIAS


@dataclass(frozen=True)
class TokenCounts:
    """Token counts off one completion's ``usage``, summed per row on the ledger.

    ``cached`` is ``prompt_tokens_details.cached_tokens`` (prompt tokens read from the
    provider's prompt cache) and ``reasoning`` is ``completion_tokens_details.reasoning_tokens``
    (hidden reasoning output); each is 0 when the provider reports nothing.
    """

    prompt: int = 0
    completion: int = 0
    cached: int = 0
    reasoning: int = 0

    def __add__(self, other: TokenCounts) -> TokenCounts:
        return TokenCounts(
            prompt=self.prompt + other.prompt,
            completion=self.completion + other.completion,
            cached=self.cached + other.cached,
            reasoning=self.reasoning + other.reasoning,
        )


NO_TOKENS: TokenCounts = TokenCounts()


@dataclass
class _RoleSpendAccumulator:
    calls: int = 0
    costed_calls: int = 0
    byok_calls: int = 0
    usd: float = 0.0
    byok_usd: float = 0.0
    charged_usd: float = 0.0
    tokens: TokenCounts = field(default_factory=TokenCounts)
    max_prompt_tokens: int = 0


@dataclass(frozen=True)
class RoleSpendRow:
    """One ``CREDIT_ROLE_SPEND`` line.

    ``usd`` is the original ``cost + upstream_inference_cost`` sum, kept as emitted since
    2026-09-03; it double counts a non-BYOK call whose upstream cost OpenRouter echoes. Since
    2026-09-09 ``charged_usd`` is the money actually charged (``cost`` plus, on a BYOK call only,
    the upstream cost) and ``byok_calls`` says how many of ``calls`` routed BYOK. The three
    dollar fields are ``None`` when no call carried cost data (rendered ``n/a``), never a
    fabricated zero. ``max_prompt_tokens`` is the largest single prompt among the row's calls,
    the packet-size read that a summed ``tokens.prompt`` hides.
    """

    role: str
    key_alias: str
    calls: int
    costed_calls: int
    usd: float | None
    byok_usd: float | None
    tokens: TokenCounts
    charged_usd: float | None
    byok_calls: int
    max_prompt_tokens: int


_role_spend: dict[tuple[str, str], _RoleSpendAccumulator] = {}


def record_llm_call_spend(
    role: str,
    key_alias: str,
    *,
    cost_usd: float | None,
    byok_upstream_usd: float | None,
    is_byok: bool = False,
    tokens: TokenCounts = NO_TOKENS,
) -> None:
    """Add one successful completion to the ledger.

    ``cost_usd`` is OpenRouter's ``usage.cost`` (what it charged the key's credits) and
    ``byok_upstream_usd`` its ``cost_details.upstream_inference_cost`` (the provider's charge,
    billed to the BYOK account's owner when ``is_byok``). A call with neither is counted but
    not costed; its tokens are summed either way. Synchronous and await-free by design (the
    callback runs on the event loop; docs/operations.md "Per-role spend").
    """
    accumulator = _role_spend.setdefault((role, key_alias), _RoleSpendAccumulator())
    accumulator.calls += 1
    accumulator.byok_calls += is_byok
    accumulator.tokens = accumulator.tokens + tokens
    accumulator.max_prompt_tokens = max(accumulator.max_prompt_tokens, tokens.prompt)
    if cost_usd is None and byok_upstream_usd is None:
        return
    accumulator.costed_calls += 1
    accumulator.usd += (cost_usd or 0.0) + (byok_upstream_usd or 0.0)
    accumulator.byok_usd += byok_upstream_usd or 0.0
    # Off BYOK, OpenRouter echoes the upstream cost beside cost; only the BYOK route charges both payers.
    accumulator.charged_usd += (cost_usd or 0.0) + ((byok_upstream_usd or 0.0) if is_byok else 0.0)


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
            tokens=acc.tokens,
            charged_usd=acc.charged_usd if acc.costed_calls else None,
            byok_calls=acc.byok_calls,
            max_prompt_tokens=acc.max_prompt_tokens,
        )
        for (role, key_alias), acc in _role_spend.items()
    ]
    return sorted(rows, key=lambda row: (row.usd is None, -(row.usd or 0.0), row.role, row.key_alias))


def reset_role_spend() -> None:
    """Empty the ledger. Used by tests; not for production code."""
    _role_spend.clear()


def _fmt_usd(value: float | None) -> str:
    """Render a per-role dollar figure at four decimals, since per-role costs run under a cent per call."""
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
            "CREDIT_ROLE_SPEND: role=%s key=%s usd=%s calls=%d costed_calls=%d byok_usd=%s"
            " prompt_tokens=%d completion_tokens=%d cached_tokens=%d reasoning_tokens=%d"
            " charged_usd=%s byok_calls=%d max_prompt_tokens=%d",
            row.role,
            row.key_alias,
            _fmt_usd(row.usd),
            row.calls,
            row.costed_calls,
            _fmt_usd(row.byok_usd),
            row.tokens.prompt,
            row.tokens.completion,
            row.tokens.cached,
            row.tokens.reasoning,
            _fmt_usd(row.charged_usd),
            row.byok_calls,
            row.max_prompt_tokens,
        )


@dataclass(frozen=True)
class _CallUsage:
    """What one completion's ``usage`` object says about money, routing and tokens."""

    cost_usd: float | None
    byok_upstream_usd: float | None
    is_byok: bool
    tokens: TokenCounts


def _usage_token_counts(usage: Any) -> TokenCounts:
    prompt_details = usage.prompt_tokens_details
    completion_details = usage.completion_tokens_details
    return TokenCounts(
        prompt=usage.prompt_tokens or 0,
        completion=usage.completion_tokens or 0,
        cached=(prompt_details.cached_tokens if prompt_details is not None else None) or 0,
        reasoning=(completion_details.reasoning_tokens if completion_details is not None else None) or 0,
    )


def _openrouter_call_usage(response_obj: Any) -> _CallUsage:
    """Read ``usage.cost``, ``usage.cost_details.upstream_inference_cost``, ``usage.is_byok`` and the
    token counts off a litellm response; each dollar figure is ``None`` when unreported (or
    non-finite, same rule as ``_as_float``), and a response without ``usage`` reads as uncosted."""
    usage = getattr(response_obj, "usage", None)
    if usage is None:
        return _CallUsage(cost_usd=None, byok_upstream_usd=None, is_byok=False, tokens=NO_TOKENS)
    cost_details = usage.get("cost_details") or {}
    return _CallUsage(
        cost_usd=_as_float(usage.get("cost")),
        byok_upstream_usd=_as_float(cost_details.get("upstream_inference_cost")),
        is_byok=usage.get("is_byok") is True,
        tokens=_usage_token_counts(usage),
    )


def _alert_on_oversized_prompt(role: str, question_ref: str | None, prompt_tokens: int) -> None:
    """WARN once per call whose prompt exceeds the threshold; the call is already billed, so this reads, never gates."""
    if prompt_tokens <= PROMPT_TOKENS_ALERT_THRESHOLD:
        return
    logger.warning(
        "PROMPT_SIZE_ALERT: role=%s question=%s prompt_tokens=%d threshold=%d",
        role,
        question_ref or "n/a",
        prompt_tokens,
        PROMPT_TOKENS_ALERT_THRESHOLD,
    )


class RoleSpendTracker(CustomLogger):
    """litellm success callback feeding the role ledger. Install once via
    :func:`install_role_spend_tracker`."""

    async def async_log_success_event(
        self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        del start_time, end_time  # CustomLogger hook signature; the ledger is not timed
        metadata = (kwargs.get("litellm_params") or {}).get("metadata") or {}
        role = metadata.get(ROLE_METADATA_KEY, UNTAGGED_ROLE)
        usage = _openrouter_call_usage(response_obj)
        record_llm_call_spend(
            role,
            metadata.get(KEY_ALIAS_METADATA_KEY, UNKNOWN_KEY_ALIAS),
            cost_usd=usage.cost_usd,
            byok_upstream_usd=usage.byok_upstream_usd,
            is_byok=usage.is_byok,
            tokens=usage.tokens,
        )
        _alert_on_oversized_prompt(role, metadata.get(QUESTION_METADATA_KEY), usage.tokens.prompt)


def install_role_spend_tracker() -> None:
    """Register the tracker with litellm exactly once per process."""
    if any(isinstance(callback, RoleSpendTracker) for callback in litellm.callbacks):
        return
    litellm.callbacks.append(RoleSpendTracker())


async def drain_litellm_callbacks(timeout_s: float = LITELLM_CALLBACK_DRAIN_TIMEOUT_S) -> None:
    """Wait for litellm's logging worker to deliver every pending success callback.

    Must run INSIDE the event loop the completions ran on (the worker's queue is bound to
    it), after yielding twice so the ``create_task`` enqueue has landed, and bounded so
    telemetry can never hold the end of a run hostage. The timeout is swallowed HERE, not
    at the call site: the one caller awaits this from ``cli._forecast_with_callback_drain``'s
    ``finally``, where a raise would discard a fully published run's reports (the q45085
    shape). ``CancelledError`` still propagates. Detail: docs/operations.md "Per-role spend".
    """
    for _ in range(2):
        await asyncio.sleep(0)
    try:
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=timeout_s)
    except TimeoutError:
        # The "within %.1fs" clause is a data contract (scripts/telemetry/markers.py "litellm_callback_drain_timeout").
        logger.warning(
            "LITELLM_CALLBACK_DRAIN_TIMEOUT: litellm's logging worker did not deliver its queued "
            "success callbacks within %.1fs; continuing so the run can finish. The CREDIT_ROLE_SPEND "
            "ledger below may under-count this run's last completions.",
            timeout_s,
        )
