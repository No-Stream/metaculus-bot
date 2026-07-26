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
  and keeps reporting a breach either way.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from metaculus_bot.constants import CREDIT_ALERT_RESUME_DATE, _date_env, credit_alerts_active
from metaculus_bot.credit_telemetry import CreditTelemetry, _fetch_snapshot

DONATED_KEY = "sk-or-v1-DONATEDsecretAB12"
PERSONAL_KEY = "sk-or-v1-PERSONALsecretCD34"


def _payload(limit_remaining: float | None, usage: float | None) -> dict[str, Any]:
    return {"label": "test", "limit_remaining": limit_remaining, "usage": usage}


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

    def fake_fetch(api_key: str) -> dict[str, Any]:
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
        assert "CREDIT_SPEND: key=donated run_delta_usd=1.23 remaining=107.93" in messages
        assert "CREDIT_SPEND: key=personal run_delta_usd=0.50 remaining=n/a" in messages

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
        assert "CREDIT_SPEND: key=donated run_delta_usd=n/a remaining=90.00" in messages


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
        assert "CREDIT_SPEND: key=donated run_delta_usd=3.34 remaining=92.22" in messages
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
        assert "CREDIT_SPEND: key=donated run_delta_usd=5.00 remaining=95.00" in messages

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
        assert "CREDIT_SPEND: key=donated run_delta_usd=2.50 remaining=92.00" in messages

    def test_no_usable_fields_yields_na_delta(self, monkeypatch, caplog) -> None:
        _set_keys(monkeypatch, personal=None)
        responses = {DONATED_KEY: [_payload(None, None), _payload(None, None)]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.INFO, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            telemetry.log_end_and_check_floor()

        messages = [record.getMessage() for record in caplog.records]
        assert "CREDIT_SPEND: key=donated run_delta_usd=n/a remaining=n/a" in messages


class TestCreditAlertSuppressionWindow:
    """The dated suppression of credit ALERTING (not of any log line).

    The operator is self-funding the rest of the season, so a drained donated key
    must not redden CI until ``CREDIT_ALERT_RESUME_DATE``. Every ``today`` here is
    injected, so these tests keep asserting the same thing after the real date
    passes 2026-09-10.
    """

    def test_resume_date_is_2026_09_10(self) -> None:
        """The hardcoded default is the contract; the env var is only an override."""
        assert CREDIT_ALERT_RESUME_DATE == date(2026, 9, 10)

    def test_inactive_before_resume_date(self) -> None:
        assert credit_alerts_active(date(2026, 7, 25)) is False
        assert credit_alerts_active(date(2026, 9, 9)) is False

    def test_active_on_and_after_resume_date(self) -> None:
        """Resume day itself counts as active — the window is closed-on-the-right."""
        assert credit_alerts_active(date(2026, 9, 10)) is True
        assert credit_alerts_active(date(2026, 9, 11)) is True
        assert credit_alerts_active(date(2027, 1, 1)) is True

    def test_resume_date_is_after_tournament_close(self) -> None:
        """The suppression must not outlive the season it exists for."""
        from metaculus_bot.constants import TOURNAMENT_END_DATE  # noqa: PLC0415

        assert CREDIT_ALERT_RESUME_DATE > date.fromisoformat(TOURNAMENT_END_DATE)

    def test_today_defaults_to_system_clock_at_call_time(self) -> None:
        """No argument → same answer as passing today's real date explicitly."""
        assert credit_alerts_active() == credit_alerts_active(date.today())

    def test_env_override_parses_iso_date(self, monkeypatch) -> None:
        monkeypatch.setenv("_TEST_RESUME_DATE_XYZ", "2026-10-01")
        assert _date_env("_TEST_RESUME_DATE_XYZ", date(2026, 9, 10)) == date(2026, 10, 1)

    @pytest.mark.parametrize("bad", ["", "   ", "not-a-date", "2026-13-01", "09/10/2026"])
    def test_env_override_falls_back_on_garbage(self, monkeypatch, bad) -> None:
        monkeypatch.setenv("_TEST_RESUME_DATE_XYZ", bad)
        assert _date_env("_TEST_RESUME_DATE_XYZ", date(2026, 9, 10)) == date(2026, 9, 10)

    def test_env_unset_uses_default(self, monkeypatch) -> None:
        monkeypatch.delenv("_TEST_RESUME_DATE_XYZ", raising=False)
        assert _date_env("_TEST_RESUME_DATE_XYZ", date(2026, 9, 10)) == date(2026, 9, 10)

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
        assert "CREDIT_SPEND: key=personal run_delta_usd=234.50 remaining=n/a" in messages


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
        request = httpx.Request("GET", "https://openrouter.ai/api/v1/auth/key")
        error = httpx.HTTPStatusError("401", request=request, response=httpx.Response(401, request=request))
        responses = {DONATED_KEY: [error, error]}
        telemetry = CreditTelemetry(floor_usd=50.0)
        with _patch_fetch(responses), caplog.at_level(logging.WARNING, logger="metaculus_bot.credit_telemetry"):
            telemetry.log_start()
            assert telemetry.log_end_and_check_floor() is False

        warnings = [r.getMessage() for r in caplog.records]
        assert any("phase=start fetch failed (HTTPStatusError)" in msg for msg in warnings)
        assert any("phase=end fetch failed (HTTPStatusError)" in msg for msg in warnings)

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
