"""Tests for ``metaculus_bot.cli.main`` — specifically the sys.exit wiring that
fires when ``TemplateForecaster.alertable_count > 0``.

Publication already happened inside ``forecast_on_tournament`` by the time cli
checks ``alertable_count``; the non-zero exit is purely so GitHub Actions
marks the run red. That wiring is load-bearing — without it, forecaster drops
and stacker fallback usage go unnoticed.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _patch_cli_main_deps(alertable_count: int) -> tuple[MagicMock, ...]:
    """Return the list of patchers needed to exercise ``cli.main`` in test mode.

    Stubs TemplateForecaster with a MagicMock whose ``alertable_count`` is
    controlled and whose ``forecast_questions`` returns an empty list (so no
    downstream ``log_report_summary`` formatting is needed).
    """
    stub_bot = MagicMock()
    stub_bot.alertable_count = alertable_count
    stub_bot.forecast_questions = AsyncMock(return_value=[])
    stub_bot.forecast_on_tournament = AsyncMock(return_value=[])

    # TemplateForecaster(...) call returns our stub
    tf_patcher = patch("metaculus_bot.cli.TemplateForecaster", return_value=stub_bot)
    # MetaculusApi.get_question_by_url returns a dummy question object; we pass
    # through a Mock so list construction doesn't explode.
    api_patcher = patch("metaculus_bot.cli.MetaculusApi", MagicMock())
    check_patcher = patch("metaculus_bot.cli.check_tournament_dates")
    # Patch log_report_summary: a classmethod on TemplateForecaster that
    # iterates forecast_reports. Our stub returns []; patch the method anyway
    # to keep the test surface small.
    summary_patcher = patch.object(type(stub_bot), "log_report_summary", create=True, return_value=None)

    return tf_patcher, api_patcher, check_patcher, summary_patcher


class TestCliExitStatus:
    def test_alertable_count_zero_returns_normally(self) -> None:
        """Zero degradation events → no SystemExit; main returns normally."""
        tf, api, check, summary = _patch_cli_main_deps(alertable_count=0)

        argv_backup = sys.argv
        sys.argv = ["cli", "--mode", "test_questions"]
        try:
            with tf, api, check, summary:
                from metaculus_bot.cli import main as cli_main

                # Must NOT raise SystemExit.
                cli_main()
        finally:
            sys.argv = argv_backup

    def test_alertable_count_nonzero_triggers_sys_exit_1(self) -> None:
        """Non-zero degradation counter → SystemExit with code 1."""
        tf, api, check, summary = _patch_cli_main_deps(alertable_count=1)

        argv_backup = sys.argv
        sys.argv = ["cli", "--mode", "test_questions"]
        try:
            with tf, api, check, summary:
                from metaculus_bot.cli import main as cli_main

                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
        finally:
            sys.argv = argv_backup

    def test_large_alertable_count_still_exits_with_code_1(self) -> None:
        """Exit code is always 1 regardless of how many events occurred —
        documents that we use exit-code-1 as a binary alert, not as an
        event count.
        """
        tf, api, check, summary = _patch_cli_main_deps(alertable_count=42)

        argv_backup = sys.argv
        sys.argv = ["cli", "--mode", "test_questions"]
        try:
            with tf, api, check, summary:
                from metaculus_bot.cli import main as cli_main

                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
        finally:
            sys.argv = argv_backup
