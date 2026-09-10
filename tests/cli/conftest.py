"""Package-wide fixtures for the cli suite.

The plain harness (``_cli_main_test_mode`` and friends) lives in ``tests/cli_test_helpers.py``.
"""

from __future__ import annotations

import pytest

from metaculus_bot.credit_telemetry import reset_donated_key_state_cache
from metaculus_bot.fallback_openrouter import (
    reset_credit_key_fallback_count,
    reset_donated_404_fallback_count,
    reset_generic_key_fallback_count,
)
from metaculus_bot.mantic import reset_post_drop_count
from metaculus_bot.research.provider_health import reset_provider_health


@pytest.fixture(autouse=True)
def _reset_fallback_counters() -> None:
    """The fallback counters are process-global (module state in
    fallback_openrouter). Reset all three between tests so cross-test pollution
    can't silently turn an "alertable=0" path into "alertable=1" because a
    prior test bumped a counter.

    The donated-key probe verdict is process-global for the same reason (probe
    once per run), and cli renders it in the end-of-run summary, so it is reset
    here too. Same for the provider-health observation store, which feeds the
    provider-degradation summand of ``alertable_count``.
    """
    reset_generic_key_fallback_count()
    reset_donated_404_fallback_count()
    reset_credit_key_fallback_count()
    reset_donated_key_state_cache()
    reset_provider_health()
    reset_post_drop_count()
