"""Tests for metaculus_bot.credit_telemetry.

Mocks the shared ``fetch_auth_key`` helper so no test touches the network or
needs real keys. Pins:

- start/end CREDIT_BALANCE marker lines + the CREDIT_SPEND run delta,
- the delta source: ``limit_remaining`` drop preferred (covers BYOK-routed
  spend that never hits ``usage`` — the 2026-07-17 frozen-usage bug), with the
  ``usage`` delta as the uncapped-key fallback,
- the donated-key floor check (below → True, at/above → False),
- fetch failures → WARNING + never trip the floor (unknown ≠ low),
- the personal key's missing ``limit_remaining`` (uncapped) rendering as n/a
  and never tripping the floor even when its usage is huge,
- the dated credit-alert suppression predicate (``credit_alerts_active``), which
  gates only the EXIT status in cli.main — the telemetry here is window-agnostic
  and keeps reporting a breach either way — plus the default floor and resume date
  themselves, which are the two levers the operator retunes,
- the drained-vs-revoked donated-key discriminator (``classify_donated_key_state``),
  which decides whether a credit-shaped failure is the EXPECTED empty wallet
  (suppressible) or real breakage (must stay red),
- the per-role dollar ledger behind ``CREDIT_ROLE_SPEND`` (``record_llm_call_spend``,
  ``RoleSpendTracker``): OpenRouter's own per-call ``usage.cost`` /
  ``cost_details.upstream_inference_cost`` reaching the litellm success callback
  tagged with the ``role`` / ``key_alias`` metadata the LLM builders stamp on every
  completion, and a role with no cost data rendering ``usd=n/a`` rather than a
  fabricated zero.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Iterator
from datetime import date, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import litellm
import pytest
from forecasting_tools import GeneralLlm
from forecasting_tools.ai_models import general_llm as ft_general_llm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.types.utils import ModelResponse, Usage

from metaculus_bot.check_openrouter_credits import KEY_SPECS
from metaculus_bot.constants import (
    CREDIT_ALERT_RESUME_DATE,
    OPENROUTER_CREDIT_FLOOR_USD,
    _date_env,
    credit_alerts_active,
)
from metaculus_bot.credit_telemetry import (
    DIRECT_KEY_ALIAS,
    DONATED_KEY_ALIAS,
    DONATED_KEY_PROBE_TIMEOUT_S,
    KEY_ALIAS_METADATA_KEY,
    PERSONAL_KEY_ALIAS,
    ROLE_METADATA_KEY,
    UNKNOWN_KEY_ALIAS,
    UNTAGGED_ROLE,
    CreditTelemetry,
    DonatedKeyState,
    RoleSpendTracker,
    _fetch_snapshot,
    classify_donated_key_state,
    drain_litellm_callbacks,
    get_probed_donated_key_state,
    install_role_spend_tracker,
    llm_call_metadata,
    log_role_spend,
    plain_llm_key_alias,
    record_llm_call_spend,
    reset_donated_key_state_cache,
    reset_role_spend,
    role_spend_rows,
)
from metaculus_bot.fallback_openrouter import FallbackOpenRouterLlm, build_llm_with_openrouter_fallback
from metaculus_bot.llm_configs import (
    DISAGREEMENT_ANALYZER_LLM,
    FORECASTER_LLMS,
    MARKET_QUERY_AUTHOR_LLM_CONFIG,
    MARKET_RANKER_LLM_CONFIG,
    PARSER_LLM,
    STACKER_FALLBACK_LLM,
    STACKER_LLM,
    SUMMARIZER_LLM,
    forecaster_role,
)
from scripts.telemetry.markers import MARKER_SPECS

DONATED_KEY = "sk-or-v1-DONATEDsecretAB12"
PERSONAL_KEY = "sk-or-v1-PERSONALsecretCD34"


def _payload(limit_remaining: float | None, usage: float | None, limit: float | None = None) -> dict[str, Any]:
    return {"label": "test", "limit": limit, "limit_remaining": limit_remaining, "usage": usage}


def _http_status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://openrouter.ai/api/v1/auth/key")
    return httpx.HTTPStatusError(str(status), request=request, response=httpx.Response(status, request=request))


def _set_keys(
    monkeypatch: pytest.MonkeyPatch, *, donated: str | None = DONATED_KEY, personal: str | None = PERSONAL_KEY
) -> None:
    for env_var, value in (("OAI_ANTH_OPENROUTER_KEY", donated), ("OPENROUTER_API_KEY", personal)):
        if value is None:
            monkeypatch.delenv(env_var, raising=False)
        else:
            monkeypatch.setenv(env_var, value)


def _patch_fetch(responses_by_key: dict[str, list[Any]]):
    """Patch fetch_auth_key to pop canned responses per api key.

    Values in each list are either dict payloads or exceptions to raise.
    Successive calls for the same key consume successive entries (start, end).
    """

    def fake_fetch(api_key: str, *, timeout: float | None = None) -> dict[str, Any]:
        del timeout  # callers vary it (the mid-run probe uses a shorter one); irrelevant here
        item = responses_by_key[api_key].pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return patch("metaculus_bot.credit_telemetry.fetch_auth_key", side_effect=fake_fetch)


class TestNormalDeltaLogging:
    def test_start_and_end_emit_markers_and_spend_delta(self, monkeypatch, caplog) -> None:
        _set_keys(monkeypatch)
        responses = {
            DONATED_KEY: [_payload(109.16, 4.16), _payload(107.93, 5.39)],
            # Personal key: no limit_remaining (uncapped); spend comes from usage.
            PERSONAL_KEY: [_payload(None, 23.41), _payload(None, 23.91)],
        }
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            below_floor = telemetry.log_end_and_check_floor()

        assert below_floor is False
        messages = [record.getMessage() for record in caplog.records]
        assert "CREDIT_BALANCE: key=donated phase=start remaining=109.16 usage=4.16" in messages
        assert "CREDIT_BALANCE: key=personal phase=start remaining=n/a usage=23.41" in messages
        assert "CREDIT_BALANCE: key=donated phase=end remaining=107.93 usage=5.39" in messages
        assert "CREDIT_SPEND: key=donated run_delta_usd=1.23 remaining=107.93 source=remaining_delta" in messages
        assert "CREDIT_SPEND: key=personal run_delta_usd=0.50 remaining=n/a source=usage_delta_unsettled" in messages

    def test_missing_start_snapshot_yields_na_delta(self, monkeypatch, caplog) -> None:
        """Start fetch failed → end still logs balances, delta renders n/a."""
        _set_keys(monkeypatch, personal=None)
        responses = {
            DONATED_KEY: [httpx.ConnectError("down"), _payload(90.0, 10.0)],
        }
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            below_floor = telemetry.log_end_and_check_floor()

        assert below_floor is False
        messages = [record.getMessage() for record in caplog.records]
        assert "CREDIT_SPEND: key=donated run_delta_usd=n/a remaining=90.00 source=unavailable" in messages


class TestRunDeltaSource:
    def test_byok_spend_with_frozen_usage_uses_remaining_delta(self, monkeypatch, caplog) -> None:
        """The 2026-07-17 smoke-run bug: donated-key spend routes through BYOK
        integrations, so ``usage`` sits frozen while ``limit_remaining`` drops.
        The delta must come from the remaining drop, not the usage delta (which
        would report 0.00 across a $3.34 run).
        """
        _set_keys(monkeypatch, personal=None)
        responses = {DONATED_KEY: [_payload(95.56, 4.16), _payload(92.22, 4.16)]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            telemetry.log_end_and_check_floor()

        messages = [record.getMessage() for record in caplog.records]
        assert "CREDIT_SPEND: key=donated run_delta_usd=3.34 remaining=92.22 source=remaining_delta" in messages
        assert "run_delta_usd=0.00" not in "\n".join(messages)

    def test_remaining_delta_preferred_over_usage_delta_when_both_move(self, monkeypatch, caplog) -> None:
        """When both fields move, ``limit_remaining`` wins — ``usage`` only sees
        the native-credit slice of the spend.
        """
        _set_keys(monkeypatch, personal=None)
        responses = {DONATED_KEY: [_payload(100.0, 4.0), _payload(95.0, 4.5)]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            telemetry.log_end_and_check_floor()

        messages = [record.getMessage() for record in caplog.records]
        assert "CREDIT_SPEND: key=donated run_delta_usd=5.00 remaining=95.00 source=remaining_delta" in messages

    def test_remaining_missing_at_one_end_falls_back_to_usage(self, monkeypatch, caplog) -> None:
        """A key whose ``limit_remaining`` is only reported at one end (e.g. a
        limit added mid-run) can't diff remaining — fall back to usage.
        """
        _set_keys(monkeypatch, personal=None)
        responses = {DONATED_KEY: [_payload(None, 4.0), _payload(92.0, 6.5)]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            telemetry.log_end_and_check_floor()

        messages = [record.getMessage() for record in caplog.records]
        assert "CREDIT_SPEND: key=donated run_delta_usd=2.50 remaining=92.00 source=usage_delta_unsettled" in messages

    def test_no_usable_fields_yields_na_delta(self, monkeypatch, caplog) -> None:
        _set_keys(monkeypatch, personal=None)
        responses = {DONATED_KEY: [_payload(None, None), _payload(None, None)]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            telemetry.log_end_and_check_floor()

        messages = [record.getMessage() for record in caplog.records]
        assert "CREDIT_SPEND: key=donated run_delta_usd=n/a remaining=n/a source=unavailable" in messages


class TestCreditAlertSuppressionWindow:
    """The dated suppression of credit ALERTING (not of any log line).

    Alerting was suppressed while the donated key was drained and the operator
    self-funded the season, and resumed on ``CREDIT_ALERT_RESUME_DATE``
    (2026-09-03, moved up from 2026-09-10 after the Metaculus grant). Every
    ``today`` here is injected, so both branches keep being exercised however far
    the real clock moves past the resume date.
    """

    def test_resume_date_is_2026_09_03(self) -> None:
        """The hardcoded default is the contract; the env var is only an override.

        Moved up from 2026-09-10 on 2026-09-03, when Metaculus granted $1,500 of
        credits, so a shortfall reddens CI again from that day.
        """
        assert date(2026, 9, 3) == CREDIT_ALERT_RESUME_DATE

    def test_inactive_before_resume_date(self) -> None:
        assert credit_alerts_active(date(2026, 7, 25)) is False
        assert credit_alerts_active(date(2026, 9, 2)) is False

    def test_active_on_and_after_resume_date(self) -> None:
        """Resume day itself counts as active — the window is closed-on-the-right."""
        assert credit_alerts_active(date(2026, 9, 3)) is True
        assert credit_alerts_active(date(2026, 9, 4)) is True
        assert credit_alerts_active(date(2027, 1, 1)) is True

    def test_resume_date_cannot_outlive_the_season(self) -> None:
        """The guard the dated lever exists for: a suppression window left open long
        past the season it was opened for is a stale suppression nobody noticed. The
        current window closed before the tournament even ends, so this holds with room
        to spare; it fails if somebody pushes the resume date out into the next season.
        """
        from metaculus_bot.constants import TOURNAMENT_END_DATE

        assert date.fromisoformat(TOURNAMENT_END_DATE) + timedelta(days=7) >= CREDIT_ALERT_RESUME_DATE

    def test_alerting_is_live_on_the_real_clock(self) -> None:
        """The state the operator asked for on 2026-09-03: no injected date, no env
        override, and credit shortfalls redden CI. Reads the real clock deliberately —
        this is the one assertion that would catch the resume date being pushed back
        into the future by accident.
        """
        assert credit_alerts_active() is True

    def test_today_defaults_to_system_clock_at_call_time(self) -> None:
        """No argument → same answer as passing today's real date explicitly."""
        assert credit_alerts_active() == credit_alerts_active(date.today())

    def test_env_override_parses_iso_date(self, monkeypatch) -> None:
        monkeypatch.setenv("_TEST_RESUME_DATE_XYZ", "2026-10-01")
        assert _date_env("_TEST_RESUME_DATE_XYZ", date(2026, 9, 3)) == date(2026, 10, 1)

    @pytest.mark.parametrize("bad", ["", "   ", "not-a-date", "2026-13-01", "09/03/2026"])
    def test_env_override_falls_back_on_garbage(self, monkeypatch, bad) -> None:
        monkeypatch.setenv("_TEST_RESUME_DATE_XYZ", bad)
        assert _date_env("_TEST_RESUME_DATE_XYZ", date(2026, 9, 3)) == date(2026, 9, 3)

    def test_env_unset_uses_default(self, monkeypatch) -> None:
        monkeypatch.delenv("_TEST_RESUME_DATE_XYZ", raising=False)
        assert _date_env("_TEST_RESUME_DATE_XYZ", date(2026, 9, 3)) == date(2026, 9, 3)

    def test_telemetry_still_reports_breach_during_suppression(self, monkeypatch, caplog) -> None:
        """Suppression lives in cli.main, not here: the telemetry keeps returning
        True and keeps emitting the CREDIT_FLOOR_BREACH WARNING regardless of the
        window, so the run log never loses the fact that the wallet is empty.
        """
        assert credit_alerts_active(date(2026, 7, 25)) is False
        _set_keys(monkeypatch, personal=None)
        responses = {DONATED_KEY: [_payload(5.0, 1.0), _payload(0.25, 6.0)]}
        telemetry = CreditTelemetry(floor_usd=1.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            assert telemetry.log_end_and_check_floor() is True

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("CREDIT_FLOOR_BREACH: key=donated remaining=0.25 floor=1.00" in msg for msg in warnings)


class TestFloorCheck:
    def test_default_floor_warns_while_runway_remains(self, monkeypatch, caplog) -> None:
        """The shipped default is an EARLY WARNING, not an empty tank.

        A donated key with $50 left still has ~125 questions of runway at the measured
        $0.38-0.41 each, and must already trip the reminder: only Metaculus can refill
        this key, so the operator needs lead time to ask. The floor was $1.00 until
        2026-09-03, which fired only once there was nothing left to warn about.
        """
        assert OPENROUTER_CREDIT_FLOOR_USD == 100.0
        _set_keys(monkeypatch, personal=None)
        responses = {DONATED_KEY: [_payload(150.0, 10.0), _payload(50.0, 110.0)]}
        telemetry = CreditTelemetry()  # no argument: the shipped default floor
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            assert telemetry.log_end_and_check_floor() is True

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("CREDIT_FLOOR_BREACH: key=donated remaining=50.00 floor=100.00" in msg for msg in warnings)
        # The wording has to read as a warning to go ask, not as an empty wallet.
        assert any("ask Metaculus for a top-up" in msg for msg in warnings)

    def test_donated_below_floor_returns_true_and_warns(self, monkeypatch, caplog) -> None:
        _set_keys(monkeypatch, personal=None)
        responses = {DONATED_KEY: [_payload(60.0, 1.0), _payload(49.99, 11.0)]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            below_floor = telemetry.log_end_and_check_floor()

        assert below_floor is True
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("CREDIT_FLOOR_BREACH: key=donated remaining=49.99 floor=50.00" in msg for msg in warnings)

    def test_donated_exactly_at_floor_is_not_a_breach(self, monkeypatch) -> None:
        _set_keys(monkeypatch, personal=None)
        responses = {DONATED_KEY: [_payload(60.0, 1.0), _payload(50.0, 11.0)]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses):
            telemetry.log_start()
            assert telemetry.log_end_and_check_floor() is False

    def test_nan_remaining_is_unreported_not_above_floor(self, monkeypatch, caplog) -> None:
        """A non-finite balance must read as "not reported", which silences no reminder.

        ``nan < floor`` is False, so an unguarded NaN silently disabled the refill reminder
        AND rendered ``run_delta_usd=nan`` in CREDIT_SPEND. Coercing it to None instead
        makes both paths say "unknown", which is the truth.
        """
        _set_keys(monkeypatch, personal=None)
        responses = {DONATED_KEY: [_payload(5.0, 1.0), _payload(float("nan"), 6.0)]}
        telemetry = CreditTelemetry(floor_usd=1.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            assert telemetry.log_end_and_check_floor() is False

        messages = [record.getMessage() for record in caplog.records]
        assert "CREDIT_BALANCE: key=donated phase=end remaining=n/a usage=6.00" in messages
        # "Not reported at one end" then routes the delta through the usage fallback, the
        # same path a mid-run limit change takes — so the spend figure survives while the
        # unusable balance renders as n/a.
        assert "CREDIT_SPEND: key=donated run_delta_usd=5.00 remaining=n/a source=usage_delta_unsettled" in messages
        assert "nan" not in "\n".join(messages)

    def test_personal_key_low_remaining_never_trips_floor(self, monkeypatch) -> None:
        """The floor applies to the DONATED key only — a personal key that
        somehow reports a tiny limit_remaining must not trip it.
        """
        _set_keys(monkeypatch, donated=None)
        responses = {PERSONAL_KEY: [_payload(1.0, 100.0), _payload(0.5, 200.0)]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses):
            telemetry.log_start()
            assert telemetry.log_end_and_check_floor() is False

    def test_personal_na_remaining_renders_na_and_no_floor(self, monkeypatch, caplog) -> None:
        """Personal key has no limit_remaining (uncapped): remaining=n/a in both
        markers and no floor trip regardless of usage magnitude.
        """
        _set_keys(monkeypatch, donated=None)
        responses = {PERSONAL_KEY: [_payload(None, 1000.0), _payload(None, 1234.5)]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            assert telemetry.log_end_and_check_floor() is False

        messages = [record.getMessage() for record in caplog.records]
        assert "CREDIT_BALANCE: key=personal phase=end remaining=n/a usage=1234.50" in messages
        assert "CREDIT_SPEND: key=personal run_delta_usd=234.50 remaining=n/a source=usage_delta_unsettled" in messages


class TestUnsettledSpendDisclosure:
    """A usage-delta spend figure must announce that it is a LOWER BOUND.

    The personal key reports no ``limit_remaining``, so its per-run delta comes from
    the lifetime ``usage`` field — which OpenRouter has typically not settled by the
    time the end snapshot fires. Measured over 178 archived personal-key runs, the
    within-run deltas summed to 58% of true growth and 160 read exactly $0.00. Since
    this is the only spend figure the operator has on a self-funded key, a bare
    ``0.00`` must never be readable as "this run was free".
    """

    def test_zero_delta_on_the_usage_branch_is_flagged_unsettled(self, monkeypatch, caplog) -> None:
        # The exact production shape: usage identical at both ends (the settlement
        # window had not closed), which is the case that reads as free-but-wasn't.
        _set_keys(monkeypatch, donated=None)
        responses = {PERSONAL_KEY: [_payload(None, 160.24), _payload(None, 160.24)]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            telemetry.log_end_and_check_floor()

        messages = [record.getMessage() for record in caplog.records]
        assert "CREDIT_SPEND: key=personal run_delta_usd=0.00 remaining=n/a source=usage_delta_unsettled" in messages
        unsettled = [m for m in messages if "CREDIT_SPEND_UNSETTLED:" in m]
        assert unsettled, "a usage-delta spend figure must be disclosed as a lower bound"
        assert "LOWER BOUND" in unsettled[0]
        # Points at the recovery path rather than leaving the reader stuck.
        assert "reconcile_credit_spend" in unsettled[0]

    def test_remaining_branch_is_not_flagged(self, monkeypatch, caplog) -> None:
        """The donated key's ``limit_remaining`` delta is reliable, so it stays quiet.

        Warning on every run would train the operator to ignore the line, which is
        what makes the personal-key case invisible again.
        """
        _set_keys(monkeypatch, personal=None)
        responses = {DONATED_KEY: [_payload(100.0, 4.0), _payload(97.5, 4.0)]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            telemetry.log_end_and_check_floor()

        messages = [record.getMessage() for record in caplog.records]
        assert "CREDIT_SPEND: key=donated run_delta_usd=2.50 remaining=97.50 source=remaining_delta" in messages
        assert not [m for m in messages if "CREDIT_SPEND_UNSETTLED:" in m]


class TestFetchFailureHandling:
    def test_end_fetch_failure_warns_and_does_not_trip_floor(self, monkeypatch, caplog) -> None:
        """Unknown ≠ low: an end-of-run fetch hiccup logs a WARNING and returns
        False even if the start balance was already below the floor.
        """
        _set_keys(monkeypatch, personal=None)
        responses = {DONATED_KEY: [_payload(10.0, 1.0), httpx.ReadTimeout("slow")]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            below_floor = telemetry.log_end_and_check_floor()

        assert below_floor is False
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("key=donated phase=end fetch failed (ReadTimeout)" in msg for msg in warnings)

    def test_http_status_error_is_caught(self, monkeypatch, caplog) -> None:
        _set_keys(monkeypatch, personal=None)
        error = _http_status_error(401)
        responses = {DONATED_KEY: [error, error]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.WARNING, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            assert telemetry.log_end_and_check_floor() is False

        warnings = [r.getMessage() for r in caplog.records]
        assert any("phase=start fetch failed (HTTPStatusError)" in msg for msg in warnings)
        assert any("phase=end fetch failed (HTTPStatusError)" in msg for msg in warnings)

    @pytest.mark.parametrize(
        "boom",
        [
            FileNotFoundError("stale SSL_CERT_FILE"),
            RuntimeError("network egress blocked by the suite guard"),
            httpx.InvalidURL("not a url"),
        ],
    )
    def test_unexpected_fetch_exception_never_escapes_telemetry(self, monkeypatch, caplog, boom) -> None:
        """ "Telemetry must never fail or block a run" has to hold for EVERY exception type.

        The narrow tuple ``(httpx.HTTPError, ValueError, KeyError, AttributeError)`` misses
        real shapes: a stale ``SSL_CERT_FILE`` makes httpx raise ``FileNotFoundError``,
        ``httpx.InvalidURL`` is not an ``httpx.HTTPError`` subclass, and this repo's own
        autouse network guard raises ``RuntimeError``. ``log_end_and_check_floor`` is called
        from a ``finally`` in cli.main, so an escape there replaces whatever the run was
        already raising and destroys the whole end-of-run diagnostic surface — the report
        summary, the alertable arithmetic, and the deprecation tripwire all run after it.
        """
        _set_keys(monkeypatch, personal=None)
        telemetry = CreditTelemetry(floor_usd=50.0)
        with (
            patch("metaculus_bot.credit_telemetry.fetch_auth_key", side_effect=boom),
            caplog.at_level(logging.WARNING, logger="metaculus_bot.credit_telemetry"),
        ):
            telemetry.log_start()
            assert telemetry.log_end_and_check_floor() is False

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(f"phase=start fetch failed ({type(boom).__name__})" in msg for msg in warnings), warnings
        assert any(f"phase=end fetch failed ({type(boom).__name__})" in msg for msg in warnings), warnings

    def test_missing_env_vars_warn_and_skip(self, monkeypatch, caplog) -> None:
        _set_keys(monkeypatch, donated=None, personal=None)
        telemetry = CreditTelemetry(floor_usd=50.0)
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            assert telemetry.log_end_and_check_floor() is False

        warnings = [r.getMessage() for r in caplog.records]
        assert any("key=donated phase=start skipped" in msg for msg in warnings)
        assert any("key=personal phase=start skipped" in msg for msg in warnings)

    def test_keys_never_leak_into_logs(self, monkeypatch, caplog) -> None:
        """Marker lines carry aliases and dollar figures only — never key material."""
        _set_keys(monkeypatch)
        responses = {
            DONATED_KEY: [_payload(109.16, 4.16), _payload(107.93, 5.39)],
            PERSONAL_KEY: [_payload(None, 23.41), _payload(None, 23.91)],
        }
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.DEBUG, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            telemetry.log_end_and_check_floor()

        all_output = "\n".join(record.getMessage() for record in caplog.records)
        assert DONATED_KEY not in all_output
        assert PERSONAL_KEY not in all_output


class TestMalformedButOkPayload:
    """A 200 whose body carries a non-mapping ``data`` (``fetch_auth_key`` returns
    ``payload.get("data", payload)``, so ``{"data": null}`` / ``{"data": [...]}``
    surface a non-dict here). ``_fetch_snapshot`` must degrade to WARNING + None,
    never AttributeError out and abort the run before forecasting."""

    @pytest.mark.parametrize("bad_data", [None, [1, 2, 3], "unexpected", 42])
    def test_non_dict_payload_warns_and_returns_none(self, monkeypatch, caplog, bad_data) -> None:
        _set_keys(monkeypatch, personal=None)
        responses = {DONATED_KEY: [bad_data]}
        with _patch_fetch(responses), caplog.at_level(logging.WARNING, logger="metaculus_bot.credit_telemetry"):
            snapshot = _fetch_snapshot("donated", phase="start")

        assert snapshot is None
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("key=donated phase=start fetch failed (AttributeError)" in msg for msg in warnings)


class TestDonatedKeyStateProbe:
    """The drained-vs-revoked discriminator.

    A dry donated key and a revoked one both surface as the same HTTP 403
    "Key limit exceeded (total limit)" text, but the operator wants opposite CI
    colors: a genuinely drained key is the expected state while they self-fund the
    season (green), while a revoked key or one Metaculus re-capped to zero is real
    breakage (red). Text cannot tell those apart, so we ask OpenRouter's free,
    read-only ``/auth/key`` endpoint.

    Every failure mode classifies as UNKNOWN, and UNKNOWN is NOT suppressible, so
    a broken or slow probe can never silently green a run.
    """

    @pytest.fixture(autouse=True)
    def _clear_probe_cache(self) -> Iterator[None]:
        reset_donated_key_state_cache()
        yield
        reset_donated_key_state_cache()

    def test_drained_key_classifies_as_drained(self, monkeypatch) -> None:
        """The live production shape: $850 cap, nothing left (OpenRouter clamps
        ``limit_remaining`` at 0 even when the true arithmetic is negative).
        """
        _set_keys(monkeypatch, personal=None)
        with _patch_fetch({DONATED_KEY: [_payload(0.0, 4.39, limit=850.0)]}):
            assert classify_donated_key_state() is DonatedKeyState.DRAINED

    def test_negative_remaining_still_drained(self, monkeypatch) -> None:
        """Defensive: if OpenRouter ever stops clamping, over-spend is still drained."""
        _set_keys(monkeypatch, personal=None)
        with _patch_fetch({DONATED_KEY: [_payload(-0.81, 4.39, limit=850.0)]}):
            assert classify_donated_key_state() is DonatedKeyState.DRAINED

    def test_zero_cap_classifies_as_zeroed_not_drained(self, monkeypatch) -> None:
        """A key re-capped to $0 never had an allocation to spend, so this is
        Metaculus cutting us off — real breakage, and it must NOT be suppressed.
        """
        _set_keys(monkeypatch, personal=None)
        with _patch_fetch({DONATED_KEY: [_payload(0.0, 4.39, limit=0.0)]}):
            assert classify_donated_key_state() is DonatedKeyState.ZEROED

    def test_funded_key_classifies_as_funded(self, monkeypatch) -> None:
        """Money remaining means the 403 was not about money at all."""
        _set_keys(monkeypatch, personal=None)
        with _patch_fetch({DONATED_KEY: [_payload(12.50, 4.39, limit=850.0)]}):
            assert classify_donated_key_state() is DonatedKeyState.FUNDED

    @pytest.mark.parametrize("status", [401, 404])
    def test_revoked_or_missing_key_classifies_as_revoked(self, monkeypatch, status: int) -> None:
        _set_keys(monkeypatch, personal=None)
        with _patch_fetch({DONATED_KEY: [_http_status_error(status)]}):
            assert classify_donated_key_state() is DonatedKeyState.REVOKED

    @pytest.mark.parametrize("status", [429, 500, 503])
    def test_other_http_status_classifies_as_unknown(self, monkeypatch, status: int) -> None:
        """A throttled or broken endpoint tells us nothing about the wallet."""
        _set_keys(monkeypatch, personal=None)
        with _patch_fetch({DONATED_KEY: [_http_status_error(status)]}):
            assert classify_donated_key_state() is DonatedKeyState.UNKNOWN

    def test_network_error_classifies_as_unknown(self, monkeypatch) -> None:
        _set_keys(monkeypatch, personal=None)
        with _patch_fetch({DONATED_KEY: [httpx.ReadTimeout("slow")]}):
            assert classify_donated_key_state() is DonatedKeyState.UNKNOWN

    @pytest.mark.parametrize("bad_data", [None, [1, 2, 3], "unexpected", 42])
    def test_malformed_payload_classifies_as_unknown(self, monkeypatch, bad_data) -> None:
        _set_keys(monkeypatch, personal=None)
        with _patch_fetch({DONATED_KEY: [bad_data]}):
            assert classify_donated_key_state() is DonatedKeyState.UNKNOWN

    @pytest.mark.parametrize(
        ("limit", "limit_remaining"),
        [
            (850.0, float("nan")),
            (float("nan"), 0.0),
            (850.0, "NaN"),  # a JSON body can spell it as a string too
            (float("inf"), float("inf")),
            (float("-inf"), float("nan")),
        ],
    )
    def test_non_finite_balance_classifies_as_unknown_not_drained(self, monkeypatch, limit, limit_remaining) -> None:
        """A NaN or infinite balance must fail SAFE (unknown → red), not to DRAINED.

        ``json.loads`` accepts bare ``NaN`` / ``Infinity`` as an extension, so a proxy or a
        malformed upstream body can deliver one. Every float comparison against NaN is
        False, so an unguarded ladder walked straight past ``limit <= 0`` and
        ``remaining > 0`` into DRAINED — the one suppressible state — inverting the
        module's documented fail-safe and letting a broken probe green a red run.
        """
        _set_keys(monkeypatch, personal=None)
        with _patch_fetch({DONATED_KEY: [_payload(limit_remaining, 4.39, limit=limit)]}):
            assert classify_donated_key_state() is DonatedKeyState.UNKNOWN

    @pytest.mark.parametrize(
        "boom",
        [
            FileNotFoundError("stale SSL_CERT_FILE"),
            RuntimeError("network egress blocked by the suite guard"),
            httpx.InvalidURL("not a url"),
        ],
    )
    def test_probe_returns_unknown_for_any_exception_type(self, monkeypatch, boom) -> None:
        """``_probe_donated_key_state`` promises "never raises", so it must catch anything.

        The narrow tuple missed all three of these, and ``fallback_openrouter`` guards only
        its own call site — the next caller would inherit the false contract. UNKNOWN is
        also the right fail-safe: it keeps the run alertable.
        """
        _set_keys(monkeypatch, personal=None)
        with patch("metaculus_bot.credit_telemetry.fetch_auth_key", side_effect=boom):
            assert classify_donated_key_state() is DonatedKeyState.UNKNOWN

    def test_uncapped_key_classifies_as_unknown(self, monkeypatch) -> None:
        """No cap means no cap to exceed; we can't call it drained, so stay red."""
        _set_keys(monkeypatch, personal=None)
        with _patch_fetch({DONATED_KEY: [_payload(None, 23.60, limit=None)]}):
            assert classify_donated_key_state() is DonatedKeyState.UNKNOWN

    def test_missing_env_var_classifies_as_unknown_without_any_fetch(self, monkeypatch) -> None:
        """No donated key configured → no HTTP at all. This is also what keeps the
        probe hermetic in tests that don't stub it: the suite's network guard is
        never even reached.
        """
        _set_keys(monkeypatch, donated=None, personal=None)
        with patch("metaculus_bot.credit_telemetry.fetch_auth_key") as fetch:
            assert classify_donated_key_state() is DonatedKeyState.UNKNOWN
        fetch.assert_not_called()

    def test_verdict_is_cached_for_the_run(self, monkeypatch) -> None:
        """Probe once per run, not once per failed call — a run that loses every
        donated-key call must not fire one HTTP request per failure.
        """
        _set_keys(monkeypatch, personal=None)
        fetch = MagicMock(return_value=_payload(0.0, 4.39, limit=850.0))
        with patch("metaculus_bot.credit_telemetry.fetch_auth_key", fetch):
            assert classify_donated_key_state() is DonatedKeyState.DRAINED
            assert classify_donated_key_state() is DonatedKeyState.DRAINED
            assert classify_donated_key_state() is DonatedKeyState.DRAINED
        assert fetch.call_count == 1

    def test_failed_probe_is_also_cached(self, monkeypatch) -> None:
        """Caching UNKNOWN matters as much as caching a verdict: without it, a dead
        endpoint costs one timeout per failed LLM call.
        """
        _set_keys(monkeypatch, personal=None)
        fetch = MagicMock(side_effect=httpx.ReadTimeout("slow"))
        with patch("metaculus_bot.credit_telemetry.fetch_auth_key", fetch):
            assert classify_donated_key_state() is DonatedKeyState.UNKNOWN
            assert classify_donated_key_state() is DonatedKeyState.UNKNOWN
        assert fetch.call_count == 1

    def test_probe_uses_a_short_timeout(self, monkeypatch) -> None:
        """The probe can run mid-run, so it must not be able to stall a forecast.
        The shared ``fetch_auth_key`` default (15s) is too long for that path.
        """
        assert DONATED_KEY_PROBE_TIMEOUT_S <= 5.0
        _set_keys(monkeypatch, personal=None)
        fetch = MagicMock(return_value=_payload(0.0, 4.39, limit=850.0))
        with patch("metaculus_bot.credit_telemetry.fetch_auth_key", fetch):
            classify_donated_key_state()
        assert fetch.call_args.kwargs["timeout"] == DONATED_KEY_PROBE_TIMEOUT_S

    def test_concurrent_callers_share_one_probe_and_one_verdict(self, monkeypatch) -> None:
        """N donated-key failures arriving at once must produce ONE probe, ONE verdict.

        The 2026-07-26 shape is N-simultaneous-failures-on-one-dry-key (~15 donated-key
        calls per question), and every production caller reaches this through
        ``asyncio.to_thread`` — so the cache is read and written from real worker threads.
        An unsynchronized check-then-set fires one HTTP probe per failure, and the damage
        is not the duplicate calls: each caller keeps ITS OWN probe result, so an
        intermittently failing ``/auth/key`` splits one drained-key incident into some
        suppressed and some alertable events. cli then computes a non-zero ``alertable``
        and exits red on exactly the expected-drained-key condition the suppression window
        exists for, while the end-of-run ``donated_key=`` note reports the last writer's
        verdict and contradicts the exit.
        """
        _set_keys(monkeypatch, personal=None)
        probe_threads: list[int] = []
        record_lock = threading.Lock()

        def slow_fetch(api_key: str, *, timeout: float | None = None) -> dict[str, Any]:
            del api_key, timeout
            with record_lock:
                probe_threads.append(threading.get_ident())
            time.sleep(0.2)  # long enough that unsynchronized callers overlap
            return _payload(0.0, 4.39, limit=850.0)

        fetch = MagicMock(side_effect=slow_fetch)

        async def probe_concurrently() -> list[DonatedKeyState]:
            return list(await asyncio.gather(*(asyncio.to_thread(classify_donated_key_state) for _ in range(12))))

        with patch("metaculus_bot.credit_telemetry.fetch_auth_key", fetch):
            verdicts = asyncio.run(probe_concurrently())

        assert fetch.call_count == 1, f"expected one probe, got {fetch.call_count} on {len(set(probe_threads))} threads"
        assert verdicts == [DonatedKeyState.DRAINED] * 12

    def test_intermittent_probe_failure_cannot_split_the_verdict(self, monkeypatch) -> None:
        """One flaky probe must not make some concurrent callers disagree with the rest.

        With the check-then-set unsynchronized, two of six probes raising produced four
        suppressible events out of six generic ones, so cli's ``alertable`` came out
        non-zero and the run went red on an expected empty wallet. Under the lock only the
        first caller probes, so all six share whatever it found.
        """
        _set_keys(monkeypatch, personal=None)
        call_count = 0
        record_lock = threading.Lock()

        def flaky_fetch(api_key: str, *, timeout: float | None = None) -> dict[str, Any]:
            del api_key, timeout
            nonlocal call_count
            with record_lock:
                call_count += 1
                mine = call_count
            time.sleep(0.2)
            if mine <= 2:
                raise ValueError("intermittent transport failure")
            return _payload(0.0, 4.39, limit=850.0)

        async def probe_concurrently() -> list[DonatedKeyState]:
            return list(await asyncio.gather(*(asyncio.to_thread(classify_donated_key_state) for _ in range(6))))

        with patch("metaculus_bot.credit_telemetry.fetch_auth_key", side_effect=flaky_fetch):
            verdicts = asyncio.run(probe_concurrently())

        assert len(set(verdicts)) == 1, f"concurrent callers disagreed: {verdicts}"
        assert call_count == 1

    def test_state_log_fires_once_across_concurrent_callers(self, monkeypatch, caplog) -> None:
        """The DONATED_KEY_STATE line is emitted under the lock, so it can't duplicate.

        Twelve copies of the same verdict in the run log would read as twelve separate
        probes to whoever greps it.
        """
        _set_keys(monkeypatch, personal=None)

        def slow_fetch(api_key: str, *, timeout: float | None = None) -> dict[str, Any]:
            del api_key, timeout
            time.sleep(0.2)
            return _payload(0.0, 4.39, limit=850.0)

        async def probe_concurrently() -> None:
            await asyncio.gather(*(asyncio.to_thread(classify_donated_key_state) for _ in range(12)))

        with (
            patch("metaculus_bot.credit_telemetry.fetch_auth_key", side_effect=slow_fetch),
            caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"),
        ):
            asyncio.run(probe_concurrently())

        state_lines = [r for r in caplog.records if "DONATED_KEY_STATE:" in r.getMessage()]
        assert len(state_lines) == 1, f"expected one state line, got {len(state_lines)}"

    def test_probed_state_is_none_until_the_probe_runs(self, monkeypatch) -> None:
        """cli renders the verdict only when it was actually established, so
        "never probed" must be distinguishable from every real verdict.
        """
        assert get_probed_donated_key_state() is None
        _set_keys(monkeypatch, personal=None)
        with _patch_fetch({DONATED_KEY: [_payload(0.0, 4.39, limit=850.0)]}):
            classify_donated_key_state()
        assert get_probed_donated_key_state() is DonatedKeyState.DRAINED

    def test_reset_clears_the_cache(self, monkeypatch) -> None:
        _set_keys(monkeypatch, personal=None)
        with _patch_fetch({DONATED_KEY: [_payload(0.0, 4.39, limit=850.0)]}):
            classify_donated_key_state()
        reset_donated_key_state_cache()
        assert get_probed_donated_key_state() is None

    @pytest.mark.parametrize(
        ("responses", "expected_state"),
        [
            ({DONATED_KEY: [_http_status_error(401)]}, "revoked"),
            ({DONATED_KEY: [_payload(0.0, 4.39, limit=0.0)]}, "zeroed"),
        ],
    )
    def test_breakage_states_log_a_warning(self, monkeypatch, caplog, responses, expected_state: str) -> None:
        """A revoked or zeroed donated key is the one outcome the operator has to
        act on, so it gets a WARNING rather than only an exit code.
        """
        _set_keys(monkeypatch, personal=None)
        with _patch_fetch(responses), caplog.at_level(logging.WARNING, logger="metaculus_bot.credit_telemetry"):
            classify_donated_key_state()

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any(f"DONATED_KEY_STATE: state={expected_state}" in msg for msg in warnings), warnings

    def test_drained_state_logs_at_info_not_warning(self, monkeypatch, caplog) -> None:
        """A drained key is expected during the self-funding window; logging it as a
        WARNING would train the operator to ignore the marker that matters.
        """
        _set_keys(monkeypatch, personal=None)
        with (
            _patch_fetch({DONATED_KEY: [_payload(0.0, 4.39, limit=850.0)]}),
            caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"),
        ):
            classify_donated_key_state()

        assert any("DONATED_KEY_STATE: state=drained" in r.getMessage() for r in caplog.records)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- Per-role dollar attribution (CREDIT_ROLE_SPEND) --------------------------


@pytest.fixture
def clean_role_ledger() -> Iterator[None]:
    """Empty ledger before AND after, and no ``RoleSpendTracker`` left in litellm's
    process-global callback lists — the tracker is installed once per process in prod,
    so a test that installs it must not leak it into the rest of the session."""
    reset_role_spend()
    yield
    for callback in list(litellm.callbacks):
        if isinstance(callback, RoleSpendTracker):
            litellm.logging_callback_manager.remove_callback_from_all_lists(callback)
    reset_role_spend()


def _role_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("CREDIT_ROLE_SPEND:")]


def _success_kwargs(metadata: dict[str, str] | None) -> dict[str, Any]:
    """The ``kwargs`` litellm hands a success callback: our tag rides
    ``litellm_params["metadata"]`` (verified against litellm 1.92's function_setup)."""
    return {"model": "openai/gpt-5.6-luna", "litellm_params": {"acompletion": True, "metadata": metadata}}


def _response_with_usage(**usage_fields: Any) -> ModelResponse:
    """A litellm ModelResponse whose usage carries OpenRouter's accounting fields.

    ``litellm.Usage`` keeps every extra constructor kwarg as an attribute (``cost``,
    ``cost_details``), which is exactly how an OpenRouter body's ``usage`` object reaches
    the callback in prod (``convert_dict_to_response`` builds ``Usage(**body["usage"])``).
    """
    return ModelResponse(usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, **usage_fields))


@pytest.mark.usefixtures("clean_role_ledger")
class TestRoleSpendLedger:
    def test_rows_sum_cost_and_byok_upstream_per_role_and_key(self, caplog) -> None:
        # Two donated-key forecaster calls (BYOK: a small OpenRouter fee in ``cost`` plus the
        # provider charge in ``upstream_inference_cost``) and one personal-key call
        # (non-BYOK: the whole charge is ``cost``). Same role, different keys → two rows.
        record_llm_call_spend("forecaster:openai", DONATED_KEY_ALIAS, cost_usd=0.001, byok_upstream_usd=0.12)
        record_llm_call_spend("forecaster:openai", DONATED_KEY_ALIAS, cost_usd=0.002, byok_upstream_usd=0.08)
        record_llm_call_spend("forecaster:openai", PERSONAL_KEY_ALIAS, cost_usd=0.25, byok_upstream_usd=None)

        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_role_spend()

        assert _role_lines(caplog) == [
            "CREDIT_ROLE_SPEND: role=forecaster:openai key=personal usd=0.2500 calls=1 costed_calls=1 byok_usd=0.0000",
            "CREDIT_ROLE_SPEND: role=forecaster:openai key=donated usd=0.2030 calls=2 costed_calls=2 byok_usd=0.2000",
        ]

    def test_uncosted_calls_render_na_not_zero(self, caplog) -> None:
        # A completion that carried no usage.cost is still a call, but its dollars are
        # UNKNOWN. Rendering 0.0000 would read as "this role is free".
        record_llm_call_spend("perplexity_research", DIRECT_KEY_ALIAS, cost_usd=None, byok_upstream_usd=None)
        record_llm_call_spend("perplexity_research", DIRECT_KEY_ALIAS, cost_usd=None, byok_upstream_usd=None)

        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_role_spend()

        assert _role_lines(caplog) == [
            "CREDIT_ROLE_SPEND: role=perplexity_research key=direct usd=n/a calls=2 costed_calls=0 byok_usd=n/a",
        ]

    def test_mixed_costed_and_uncosted_reports_both_counts(self, caplog) -> None:
        # The sum covers only the costed calls, and ``costed_calls < calls`` says so.
        record_llm_call_spend("parser", DONATED_KEY_ALIAS, cost_usd=0.01, byok_upstream_usd=None)
        record_llm_call_spend("parser", DONATED_KEY_ALIAS, cost_usd=None, byok_upstream_usd=None)

        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_role_spend()

        assert _role_lines(caplog) == [
            "CREDIT_ROLE_SPEND: role=parser key=donated usd=0.0100 calls=2 costed_calls=1 byok_usd=0.0000",
        ]

    def test_rows_sort_by_usd_descending_with_uncosted_last(self) -> None:
        record_llm_call_spend("parser", DONATED_KEY_ALIAS, cost_usd=0.01, byok_upstream_usd=None)
        record_llm_call_spend("untagged", UNKNOWN_KEY_ALIAS, cost_usd=None, byok_upstream_usd=None)
        record_llm_call_spend("forecaster:google", PERSONAL_KEY_ALIAS, cost_usd=0.30, byok_upstream_usd=None)
        record_llm_call_spend("summarizer", DONATED_KEY_ALIAS, cost_usd=0.0, byok_upstream_usd=0.05)

        assert [(row.role, row.key_alias) for row in role_spend_rows()] == [
            ("forecaster:google", PERSONAL_KEY_ALIAS),
            ("summarizer", DONATED_KEY_ALIAS),
            ("parser", DONATED_KEY_ALIAS),
            ("untagged", UNKNOWN_KEY_ALIAS),
        ]

    def test_empty_ledger_says_so_without_the_row_shape(self, caplog) -> None:
        # A run with zero completions must still leave a line (silence is indistinguishable
        # from a run that died first), but not one the harvester could mistake for a row.
        with caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            log_role_spend()

        (line,) = _role_lines(caplog)
        assert "role=" not in line
        assert "no successful LLM completions" in line

    def test_key_aliases_are_the_credit_spend_key_names(self) -> None:
        # ``CREDIT_ROLE_SPEND key=`` must join onto ``CREDIT_SPEND key=`` / ``CREDIT_BALANCE
        # key=``, whose vocabulary is KEY_SPECS.
        assert {DONATED_KEY_ALIAS, PERSONAL_KEY_ALIAS} == set(KEY_SPECS)
        assert DIRECT_KEY_ALIAS not in KEY_SPECS
        assert UNKNOWN_KEY_ALIAS not in KEY_SPECS


class TestLlmCallMetadata:
    def test_role_and_key_ride_the_two_metadata_keys(self) -> None:
        assert llm_call_metadata("stacker", DONATED_KEY_ALIAS) == {
            ROLE_METADATA_KEY: "stacker",
            KEY_ALIAS_METADATA_KEY: DONATED_KEY_ALIAS,
        }

    def test_missing_role_is_tagged_untagged_at_construction(self) -> None:
        # Construction, not the callback, owns the default: every metaculus_bot-built LLM
        # carries an explicit role token, so an ``untagged`` row in a run log means a
        # builder call site forgot its ``role=``.
        assert llm_call_metadata(None, PERSONAL_KEY_ALIAS)[ROLE_METADATA_KEY] == UNTAGGED_ROLE

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("openrouter/openai/gpt-5.6-sol", "forecaster:openai"),
            ("openrouter/anthropic/claude-opus-4.8", "forecaster:anthropic"),
            ("openrouter/google/gemini-3.1-pro-preview", "forecaster:google"),
        ],
    )
    def test_forecaster_role_is_the_vendor_slot(self, model: str, expected: str) -> None:
        # Latest-per-vendor roster: the slot outlives any one model, so the role does too.
        assert forecaster_role(model) == expected

    def test_forecaster_role_rejects_a_non_openrouter_slug(self) -> None:
        with pytest.raises(ValueError, match="openrouter/<vendor>/<model>"):
            forecaster_role("perplexity/sonar")

    def test_plain_llm_key_alias(self) -> None:
        # A plain GeneralLlm with no api_key reads OPENROUTER_API_KEY from the environment
        # for openrouter/ slugs; anything else bills a provider-direct key.
        assert plain_llm_key_alias("openrouter/x-ai/grok-4.5") == PERSONAL_KEY_ALIAS
        assert plain_llm_key_alias("perplexity/sonar-reasoning") == DIRECT_KEY_ALIAS


@pytest.mark.usefixtures("clean_role_ledger")
class TestRoleSpendTracker:
    async def test_callback_reads_role_key_and_openrouter_usage_fields(self) -> None:
        tracker = RoleSpendTracker()
        response = _response_with_usage(cost=0.0015, cost_details={"upstream_inference_cost": 0.31})

        await tracker.async_log_success_event(
            _success_kwargs(llm_call_metadata("stacker", DONATED_KEY_ALIAS)), response, None, None
        )

        (row,) = role_spend_rows()
        assert (row.role, row.key_alias, row.calls, row.costed_calls) == ("stacker", DONATED_KEY_ALIAS, 1, 1)
        assert row.usd == pytest.approx(0.3115)
        assert row.byok_usd == pytest.approx(0.31)

    async def test_non_byok_usage_has_no_upstream_component(self) -> None:
        tracker = RoleSpendTracker()
        # OpenRouter sends upstream_inference_cost as null (or omits cost_details) off BYOK.
        response = _response_with_usage(cost=0.02, cost_details={"upstream_inference_cost": None})

        await tracker.async_log_success_event(
            _success_kwargs(llm_call_metadata("parser", PERSONAL_KEY_ALIAS)), response, None, None
        )

        (row,) = role_spend_rows()
        assert (row.usd, row.byok_usd, row.costed_calls) == (pytest.approx(0.02), 0.0, 1)

    async def test_missing_metadata_files_under_untagged_and_unknown_key(self) -> None:
        # Any litellm completion the bot did not build (forecasting-tools' own helpers, an
        # ablation harness) still counts — visibly, under the two sentinel labels.
        tracker = RoleSpendTracker()
        await tracker.async_log_success_event(_success_kwargs(None), _response_with_usage(cost=0.5), None, None)

        (row,) = role_spend_rows()
        assert (row.role, row.key_alias, row.calls) == (UNTAGGED_ROLE, UNKNOWN_KEY_ALIAS, 1)

    async def test_usage_without_cost_is_a_call_but_not_a_costed_call(self) -> None:
        tracker = RoleSpendTracker()
        await tracker.async_log_success_event(
            _success_kwargs(llm_call_metadata("perplexity_research", DIRECT_KEY_ALIAS)),
            _response_with_usage(),
            None,
            None,
        )

        (row,) = role_spend_rows()
        assert (row.calls, row.costed_calls, row.usd) == (1, 0, None)

    async def test_non_finite_cost_is_treated_as_unreported(self) -> None:
        # Same rule as the balance parser: NaN would poison every sum it touched.
        tracker = RoleSpendTracker()
        await tracker.async_log_success_event(
            _success_kwargs(llm_call_metadata("parser", DONATED_KEY_ALIAS)),
            _response_with_usage(cost=float("nan")),
            None,
            None,
        )

        (row,) = role_spend_rows()
        assert (row.calls, row.costed_calls, row.usd) == (1, 0, None)

    def test_install_is_idempotent(self) -> None:
        install_role_spend_tracker()
        install_role_spend_tracker()
        assert sum(isinstance(cb, RoleSpendTracker) for cb in litellm.callbacks) == 1

    async def test_real_litellm_mock_path_delivers_the_role_tag_after_drain(self, monkeypatch) -> None:
        """End to end through forecasting-tools and REAL litellm (network short-circuited by
        ``mock_response``): the builder's ``role=`` reaches the ledger via
        ``metadata`` → ``litellm_params`` → the success callback → the logging worker.

        The drain is load-bearing: litellm enqueues the callback from a ``create_task``,
        so without it the row is not there yet when the awaited call returns."""
        _set_keys(monkeypatch)
        install_role_spend_tracker()
        real_acompletion = litellm.acompletion

        async def mocked_acompletion(**kwargs: Any) -> Any:
            return await real_acompletion(**kwargs, mock_response="ok")

        monkeypatch.setattr(ft_general_llm, "acompletion", mocked_acompletion)
        llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5.6-luna", role="parser", allowed_tries=1)
        assert isinstance(llm, FallbackOpenRouterLlm)

        assert await llm.invoke("hi") == "ok"
        await drain_litellm_callbacks()

        (row,) = role_spend_rows()
        # The mock body carries token counts but no OpenRouter ``usage.cost``, so the call
        # is counted and its dollars stay unknown — the honest shape, not a zero.
        assert (row.role, row.key_alias, row.calls, row.costed_calls, row.usd) == (
            "parser",
            DONATED_KEY_ALIAS,
            1,
            0,
            None,
        )

    async def test_drain_is_bounded_and_a_no_op_with_no_pending_callbacks(self) -> None:
        # cli.main's finally must never stall on telemetry; with nothing queued this
        # returns immediately rather than waiting on a worker that never started.
        await asyncio.wait_for(drain_litellm_callbacks(), timeout=1.0)

    async def test_drain_timeout_warns_and_returns_instead_of_raising(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A flush that never finishes must not turn a published run into a crashed one.

        The timeout is reachable without a bug on our side: litellm allows each queued
        callback 20s (``LOGGING_WORKER_MAX_TIME_PER_COROUTINE``), twice this drain's
        bound, and a worker loop that dies on a non-``CancelledError`` leaves
        ``queue.join()`` outstanding forever. The drain runs from
        ``cli._forecast_with_callback_drain``'s ``finally``, and nothing between there
        and process exit catches — so a raise here discarded a fully published run's
        reports and skipped ``log_report_summary`` plus the whole degradation/exit
        block (the q45085 failure shape), or demoted a real forecast error to
        ``__context__``. It warns and returns; the ledger may under-count.
        """

        async def never_finishes() -> None:
            await asyncio.Event().wait()

        monkeypatch.setattr(GLOBAL_LOGGING_WORKER, "flush", never_finishes)
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.credit_telemetry"):
            await asyncio.wait_for(drain_litellm_callbacks(timeout_s=0.01), timeout=5.0)

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any(msg.startswith("LITELLM_CALLBACK_DRAIN_TIMEOUT:") for msg in warnings), warnings
        # Seam pin: the WARN is a harvested marker as of 2026-09-04, and the archive's only record
        # of WHY a run's CREDIT_ROLE_SPEND rows under-count, so the string this code emits must
        # still parse under the registry regex. ``%.1f`` renders this test's 0.01 bound as 0.0.
        spec = next(s for s in MARKER_SPECS if s.name == "litellm_callback_drain_timeout")
        match = spec.regex.search(caplog.text)
        assert match is not None, warnings
        assert match.group("timeout_s") == "0.0"


class TestProdLlmsAreRoleTagged:
    """Every LLM ``llm_configs`` builds for prod carries its role tag, and the roster slots
    derive theirs from the slug. Roster-agnostic on purpose: a swap must not be able to
    leave a slot booking as ``untagged``."""

    def test_roster_slots_are_tagged_by_vendor(self) -> None:
        assert FORECASTER_LLMS, "roster must be non-empty for this pin to mean anything"
        for llm in FORECASTER_LLMS:
            assert llm.litellm_kwargs["metadata"][ROLE_METADATA_KEY] == forecaster_role(llm.model)

    @pytest.mark.parametrize(
        ("llm", "role"),
        [
            (SUMMARIZER_LLM, "summarizer"),
            (PARSER_LLM, "parser"),
            (STACKER_LLM, "stacker"),
            (STACKER_FALLBACK_LLM, "stacker_fallback"),
            (DISAGREEMENT_ANALYZER_LLM, "crux_analyzer"),
        ],
    )
    def test_support_slots_carry_their_role(self, llm: GeneralLlm, role: str) -> None:
        assert llm.litellm_kwargs["metadata"][ROLE_METADATA_KEY] == role

    def test_market_stage_configs_carry_their_role(self) -> None:
        # Raw dicts fed to build_llm_with_openrouter_fallback(**config) at call time.
        assert MARKET_RANKER_LLM_CONFIG["role"] == "market_ranker"
        assert MARKET_QUERY_AUTHOR_LLM_CONFIG["role"] == "market_query_author"
