"""CLI for checking OpenRouter key balances without leaking the keys.

Hits ``GET https://openrouter.ai/api/v1/auth/key`` for the donated Metaculus key
(``OAI_ANTH_OPENROUTER_KEY``) and the personal key (``OPENROUTER_API_KEY``),
prints labeled balance blocks. The full key is never written to stdout.

Usage:
    python -m metaculus_bot.check_openrouter_credits
    python -m metaculus_bot.check_openrouter_credits --key donated
    python -m metaculus_bot.check_openrouter_credits --key personal
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import httpx

from metaculus_bot.config import load_environment

OPENROUTER_AUTH_KEY_URL = "https://openrouter.ai/api/v1/auth/key"

KEY_SPECS: dict[str, tuple[str, str]] = {
    # alias -> (env_var_name, display_label)
    "donated": ("OAI_ANTH_OPENROUTER_KEY", "Donated key (OAI_ANTH_OPENROUTER_KEY)"),
    "personal": ("OPENROUTER_API_KEY", "Personal key (OPENROUTER_API_KEY)"),
}

logger = logging.getLogger(__name__)


def _redact(secret: str) -> str:
    """Return a safe display string showing only the last 4 characters."""
    if len(secret) <= 4:
        return "****"
    return f"****{secret[-4:]}"


def _format_usd(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _fetch_auth_key(api_key: str) -> dict[str, Any]:
    """Hit /api/v1/auth/key and return the parsed ``data`` payload.

    Raises ``httpx.HTTPStatusError`` on non-2xx responses so the caller can
    surface a friendly message.
    """
    response = httpx.get(
        OPENROUTER_AUTH_KEY_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15.0,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("data", payload)


def _print_block(display_label: str, env_var: str, api_key: str, data: dict[str, Any]) -> None:
    print(f"=== {display_label} ===")
    print(f"Env var:            {env_var}")
    print(f"Key fingerprint:    {_redact(api_key)}")
    print(f"Label:              {data.get('label', 'n/a')}")
    print(f"Limit (USD):        {_format_usd(data.get('limit'))}")
    print(f"Remaining (USD):    {_format_usd(data.get('limit_remaining'))}")
    print(f"Used (USD):         {_format_usd(data.get('usage'))}")
    print(f"Used today (USD):   {_format_usd(data.get('usage_daily'))}")
    print(f"Used this week:     {_format_usd(data.get('usage_weekly'))}")
    print(f"Used this month:    {_format_usd(data.get('usage_monthly'))}")
    print(f"Free tier:          {data.get('is_free_tier', 'n/a')}")
    if data.get("limit_reset"):
        print(f"Limit resets:       {data.get('limit_reset')}")
    print()


def check_key(alias: str) -> bool:
    """Look up + report on one key alias. Returns True iff a balance was printed."""
    env_var, display_label = KEY_SPECS[alias]
    api_key = os.getenv(env_var)
    if not api_key:
        print(f"WARN: env var {env_var} not set, skipping {alias} key.")
        return False

    try:
        data = _fetch_auth_key(api_key)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 401:
            print(f"ERROR: {env_var} rejected (401 Unauthorized) — key likely revoked or invalid.")
        else:
            print(f"ERROR: {env_var} returned HTTP {status} from OpenRouter.")
        return False
    except httpx.HTTPError as exc:
        print(f"ERROR: {env_var} request failed: {type(exc).__name__}")
        return False

    _print_block(display_label, env_var, api_key, data)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check OpenRouter key balances.")
    parser.add_argument(
        "--key",
        choices=("donated", "personal", "both"),
        default="both",
        help="Which key(s) to check (default: both).",
    )
    args = parser.parse_args(argv)

    load_environment()

    aliases = ["donated", "personal"] if args.key == "both" else [args.key]

    any_success = False
    for alias in aliases:
        if check_key(alias):
            any_success = True

    if not any_success:
        print("ERROR: no keys could be checked.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
