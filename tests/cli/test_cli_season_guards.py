"""The startup date guards that decide a run's colour: the fall-cup reminder, the stale Mantic
slug, and which mode checks tournament dates at all.

Each verdict is computed at startup and turned into a non-zero exit only AFTER publishing, so
these pin both halves: the run completes, then it goes red. The real ``check_tournament_dates``
is covered in ``tests/test_tournament_dates.py``.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from metaculus_bot.cli import RunMode, _check_tournament_dates
from metaculus_bot.cli import main as cli_main
from metaculus_bot.constants import MANTIC_TOURNAMENT_END_DATE, MANTIC_TOURNAMENT_ID, TournamentExpiredError
from tests.cli_test_helpers import _cli_main_test_mode, _mantic_env


class TestCliFallCupReminderExit:
    """The fall-cup reminder reddens the run the same way the credit floor does.

    The check itself (date gate, FALL_CUP_CONFIGURED flip, log content) is covered in
    test_tournament_dates.py; here the harness pins its verdict and these tests pin
    the exit wiring: run completes and publishes first, exits non-zero after, and the
    run is never stamped clean.
    """

    def test_reminder_fires_run_completes_then_sys_exit_1(self, caplog: pytest.LogCaptureFixture) -> None:
        with (
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
            _cli_main_test_mode(alertable_count=0, fall_cup_reminder=True) as telemetry,
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1
            # Forecasting/telemetry completed before the exit — reminder, not abort.
            telemetry.log_start.assert_called_once()
            telemetry.log_end_and_check_floor.assert_called_once()
        # run_clean is the exact complement of every non-zero exit, so a reminder run must not stamp it.
        assert "Run completed clean" not in caplog.text

    def test_no_reminder_returns_normally(self) -> None:
        with _cli_main_test_mode(alertable_count=0, fall_cup_reminder=False):
            cli_main()


class TestCliManticStaleSlugExit:
    """A Mantic slug past its end date reddens the run the way the fall-cup reminder does.

    ``check_tournament_dates`` warns for TOURNAMENT_HARD_STOP_WEEKS before it raises, and a
    zero-question run is green, so once Preseason 2 closes every scheduled run would stay green
    and silent for two weeks while a Series 2 slug went unforecast (edge review item 5). The
    verdict is held at startup and turned into a non-zero exit AFTER publishing; the shared hard
    stop is untouched, and the Metaculus tournament keeps the warning advisory.
    """

    def test_a_stale_mantic_slug_publishes_first_then_exits_non_zero(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _mantic_env(monkeypatch)
        with (
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
            _cli_main_test_mode(alertable_count=0, mode="mantic", tournament_stale=True) as telemetry,
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1
            telemetry.log_start.assert_called_once()
            telemetry.log_end_and_check_floor.assert_called_once()
        assert "Run completed clean" not in caplog.text
        assert "re-point MANTIC_TOURNAMENT_ID" in caplog.text

    def test_a_current_mantic_slug_returns_normally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mantic_env(monkeypatch)
        with _cli_main_test_mode(alertable_count=0, mode="mantic", tournament_stale=False):
            cli_main()

    def test_a_stale_metaculus_tournament_stays_advisory(self) -> None:
        with _cli_main_test_mode(alertable_count=0, mode="tournament", tournament_stale=True):
            cli_main()


class TestTournamentDateCheck:
    """The stale-slug check runs once at startup, per mode, and only the Mantic verdict is held.

    ``check_tournament_dates`` warns for TOURNAMENT_HARD_STOP_WEEKS before it raises, and a
    zero-question run is green, so on Mantic the fortnight between the two was a silent forfeit
    of every Series 2 question (edge review item 5). The Metaculus tournament keeps the warning
    advisory: its questions are open for weeks. ``TestCliManticStaleSlugExit`` pins what main
    does with the verdict.
    """

    def test_mantic_mode_checks_the_mantic_dates_and_returns_the_verdict(self) -> None:
        with patch("metaculus_bot.cli.check_tournament_dates", return_value=True) as check_dates:
            assert _check_tournament_dates("mantic") is True
        check_dates.assert_called_once_with(
            logging.getLogger("metaculus_bot.cli"),
            tournament_id=MANTIC_TOURNAMENT_ID,
            end_date_str=MANTIC_TOURNAMENT_END_DATE,
        )

    def test_tournament_mode_checks_the_metaculus_dates_but_stays_advisory(self) -> None:
        with patch("metaculus_bot.cli.check_tournament_dates", return_value=True) as check_dates:
            assert _check_tournament_dates("tournament") is False
        check_dates.assert_called_once_with(logging.getLogger("metaculus_bot.cli"))

    @pytest.mark.parametrize("run_mode", ["minibench", "quarterly_cup", "metaculus_cup", "test_questions"])
    def test_undated_modes_check_nothing(self, run_mode: RunMode) -> None:
        with patch("metaculus_bot.cli.check_tournament_dates") as check_dates:
            assert _check_tournament_dates(run_mode) is False
        check_dates.assert_not_called()

    def test_the_shared_hard_stop_still_raises_before_the_forecaster_is_built(self) -> None:
        forecaster_class = MagicMock()
        with (
            _cli_main_test_mode(alertable_count=0, mode="tournament"),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli.check_tournament_dates", side_effect=TournamentExpiredError("stale")),
            pytest.raises(TournamentExpiredError),
        ):
            cli_main()
        forecaster_class.assert_not_called()
