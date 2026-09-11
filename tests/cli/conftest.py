"""Package-wide fixtures for the cli suite.

The plain harness (``_cli_main_test_mode`` and friends) lives in ``tests/cli_test_helpers.py``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from metaculus_bot.credit_telemetry import reset_donated_key_state_cache
from metaculus_bot.fallback_openrouter import (
    reset_credit_key_fallback_count,
    reset_donated_404_fallback_count,
    reset_generic_key_fallback_count,
)
from metaculus_bot.mantic import reset_post_drop_count


def _zero_cli_run_counters() -> None:
    """Zero the process-global counters cli's own exit paths and run summary read."""
    reset_generic_key_fallback_count()
    reset_donated_404_fallback_count()
    reset_credit_key_fallback_count()
    reset_donated_key_state_cache()
    reset_post_drop_count()


@pytest.fixture(autouse=True)
def _reset_fallback_counters() -> Iterator[None]:
    """The three fallback counters, the donated-key probe verdict and the Mantic post-drop
    count are all process-global (module state, probed or counted once per run), and cli
    exits or renders its summary off them, so cross-test pollution turns an "alertable=0"
    path into "alertable=1". Reset on the way out as well as in, because the readers
    outside ``tests/cli/`` are just as exposed. The counters behind ``alertable_count``
    itself, the provider-health store included, are reset for every test by the root
    conftest's ``_isolate_alertable_counters``.
    """
    _zero_cli_run_counters()
    yield
    _zero_cli_run_counters()
