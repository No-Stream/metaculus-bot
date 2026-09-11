"""``cli.main`` emits ``CREDIT_RUN_SUMMARY`` with the run's forecast-report count as its denominator.

The summary itself (the ledger folded to dollars per question) is unit tested in
tests/test_credit_role_spend.py; these pin the cli wiring: the line is logged from the same
``finally`` as the role ledger, so a crashed run still reports its money (against zero questions),
and the denominator counts ``ForecastReport`` objects only, never the exceptions returned beside
them.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools import ForecastReport

from metaculus_bot.cli import main as cli_main
from tests.cli_test_helpers import _cli_main_test_mode, asyncio_run_stub


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
