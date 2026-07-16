"""Tests for metaculus_bot.credit_telemetry.

Mocks the shared ``fetch_auth_key`` helper so no test touches the network or
needs real keys. Pins:

- start/end CREDIT_BALANCE marker lines + the CREDIT_SPEND run delta,
- the donated-key floor check (below → True, at/above → False),
- fetch failures → WARNING + never trip the floor (unknown ≠ low),
- the personal key's missing ``limit_remaining`` (uncapped) rendering as n/a
  and never tripping the floor even when its usage is huge.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from metaculus_bot.credit_telemetry import CreditTelemetry

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
