"""Tests for metaculus_bot.check_openrouter_credits.

Mocks the HTTP call so this runs without network access. Asserts that the
full API key never appears in stdout, and that exit codes line up with what
callers / Make targets will see.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from metaculus_bot import check_openrouter_credits as cc

DONATED_KEY = "sk-or-v1-DONATEDsecretAB12"
PERSONAL_KEY = "sk-or-v1-PERSONALsecretCD34"


def _ok_response(label: str, usage: float = 12.5) -> httpx.Response:
    payload = {
        "data": {
            "label": label,
            "limit": 500.0,
            "limit_remaining": 500.0 - usage,
            "usage": usage,
            "usage_daily": 1.5,
            "usage_weekly": 4.2,
            "usage_monthly": usage,
            "is_free_tier": False,
            "limit_reset": "monthly",
        }
    }
    return httpx.Response(200, content=json.dumps(payload).encode())


def _err_response(status: int) -> httpx.Response:
    return httpx.Response(status, content=b'{"error": "boom"}')


def _patch_env(monkeypatch: pytest.MonkeyPatch, donated: str | None, personal: str | None) -> None:
    if donated is None:
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
    else:
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", donated)
    if personal is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", personal)


def _patch_load_env():
    """Stop the script from re-loading real `.env` during tests."""
    return patch.object(cc, "load_environment", lambda: None)


def test_redact_short_and_long_keys():
    assert cc._redact("abcd") == "****"
    assert cc._redact("abcdef1234") == "****1234"


def test_both_keys_present_both_succeed(monkeypatch, capsys):
    _patch_env(monkeypatch, DONATED_KEY, PERSONAL_KEY)
    fake = httpx.MockTransport(lambda _req: _ok_response("Metaculus-Donated"))

    with _patch_load_env(), patch.object(httpx, "get", lambda url, **kw: httpx.Client(transport=fake).get(url, **kw)):
        rc = cc.main(["--key", "both"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Donated key (OAI_ANTH_OPENROUTER_KEY)" in captured.out
    assert "Personal key (OPENROUTER_API_KEY)" in captured.out
    # Redacted fingerprints visible
    assert "****AB12" in captured.out
    assert "****CD34" in captured.out
    # Full keys NEVER printed
    assert DONATED_KEY not in captured.out
    assert PERSONAL_KEY not in captured.out
    assert "DONATEDsecret" not in captured.out
    assert "PERSONALsecret" not in captured.out


def test_donated_missing_personal_present_exits_zero(monkeypatch, capsys):
    _patch_env(monkeypatch, None, PERSONAL_KEY)
    fake = httpx.MockTransport(lambda _req: _ok_response("Personal"))

    with _patch_load_env(), patch.object(httpx, "get", lambda url, **kw: httpx.Client(transport=fake).get(url, **kw)):
        rc = cc.main(["--key", "both"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "WARN: env var OAI_ANTH_OPENROUTER_KEY not set" in captured.out
    assert "Personal key (OPENROUTER_API_KEY)" in captured.out
    assert PERSONAL_KEY not in captured.out


def test_both_keys_missing_exits_one(monkeypatch, capsys):
    _patch_env(monkeypatch, None, None)

    with _patch_load_env():
        rc = cc.main(["--key", "both"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "WARN: env var OAI_ANTH_OPENROUTER_KEY not set" in captured.out
    assert "WARN: env var OPENROUTER_API_KEY not set" in captured.out
    assert "ERROR: no keys could be checked." in captured.err


def test_http_401_friendly_error(monkeypatch, capsys):
    _patch_env(monkeypatch, DONATED_KEY, None)
    fake = httpx.MockTransport(lambda _req: _err_response(401))

    with _patch_load_env(), patch.object(httpx, "get", lambda url, **kw: httpx.Client(transport=fake).get(url, **kw)):
        rc = cc.main(["--key", "donated"])

    captured = capsys.readouterr()
    assert rc == 1
    assert "401 Unauthorized" in captured.out
    # Even on error, full key must not leak
    assert DONATED_KEY not in captured.out
    assert DONATED_KEY not in captured.err


def test_only_donated_requested(monkeypatch, capsys):
    _patch_env(monkeypatch, DONATED_KEY, PERSONAL_KEY)
    fake = httpx.MockTransport(lambda _req: _ok_response("Metaculus-Donated"))

    with _patch_load_env(), patch.object(httpx, "get", lambda url, **kw: httpx.Client(transport=fake).get(url, **kw)):
        rc = cc.main(["--key", "donated"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "Donated key (OAI_ANTH_OPENROUTER_KEY)" in captured.out
    assert "Personal key (OPENROUTER_API_KEY)" not in captured.out
    assert PERSONAL_KEY not in captured.out
