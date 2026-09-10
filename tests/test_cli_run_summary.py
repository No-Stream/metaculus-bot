"""``cli.main`` emits ``CREDIT_RUN_SUMMARY`` with the run's forecast-report count as its denominator.

The summary itself (the ledger folded to dollars per question) is unit tested in
tests/test_credit_role_spend.py; these pin the cli wiring: the line is logged from the same
``finally`` as the role ledger, so a crashed run still reports its money (against zero questions),
and the denominator counts ``ForecastReport`` objects only, never the exceptions returned beside
them. The harness (``_cli_main_test_mode``, ``asyncio_run_stub``) lives in tests/cli_test_helpers.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools import ForecastReport

from metaculus_bot.cli import main as cli_main
from metaculus_bot.credit_telemetry import reset_donated_key_state_cache
from metaculus_bot.fallback_openrouter import (
    reset_credit_key_fallback_count,
    reset_donated_404_fallback_count,
    reset_generic_key_fallback_count,
)
from metaculus_bot.mantic import reset_post_drop_count
from metaculus_bot.research.provider_health import reset_provider_health
from tests.cli_test_helpers import _cli_main_test_mode, asyncio_run_stub


@pytest.fixture(autouse=True)
def _reset_process_global_counters() -> None:
    """The same resets tests/cli/conftest.py applies: the counters are process-global and a
    leftover would turn this module's clean-exit path into a SystemExit."""
    reset_generic_key_fallback_count()
    reset_donated_404_fallback_count()
    reset_credit_key_fallback_count()
    reset_donated_key_state_cache()
    reset_provider_health()
    reset_post_drop_count()


class TestCliRunSummaryWiring:
    def test_denominator_counts_forecast_reports_and_skips_exceptions(self) -> None:
        stub_bot = MagicMock()
        stub_bot.alertable_count = 0
        reports = [MagicMock(spec=ForecastReport), MagicMock(spec=ForecastReport), RuntimeError("one question died")]
        with (
            _cli_main_test_mode(alertable_count=0, stub_bot=stub_bot),
            patch("metaculus_bot.cli.log_run_summary") as log_summary,
            patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(lambda *_a, **_k: reports)),
        ):
            cli_main()
        log_summary.assert_called_once_with(n_questions=2)

    def test_crashed_run_reports_its_money_against_zero_questions(self) -> None:
        def _crash(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("forecasting blew up")

        with (
            _cli_main_test_mode(alertable_count=0),
            patch("metaculus_bot.cli.log_role_spend") as log_roles,
            patch("metaculus_bot.cli.log_run_summary") as log_summary,
            patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_crash)),
            pytest.raises(RuntimeError, match="forecasting blew up"),
        ):
            cli_main()
        log_roles.assert_called_once_with()
        log_summary.assert_called_once_with(n_questions=0)
