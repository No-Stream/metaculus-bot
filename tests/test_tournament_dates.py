"""Tests for tournament date validation.

Includes a test that will FAIL if run after the tournament end date + grace period,
forcing developers to update TOURNAMENT_ID and related constants for the new season.
"""

import logging
from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from metaculus_bot import constants
from metaculus_bot.constants import (
    FALL_CUP_REMINDER_DATE,
    FALL_CUP_SLUG,
    TOURNAMENT_END_DATE,
    TOURNAMENT_HARD_STOP_WEEKS,
    TOURNAMENT_ID,
    TournamentExpiredError,
    check_fall_cup_reminder,
    check_tournament_dates,
    fall_cup_reminder_due,
)


class TestTournamentDateCheck:
    """Unit tests for check_tournament_dates function."""

    def test_no_warning_during_active_tournament(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning when tournament is still active."""
        # Use a date well before the end date
        end_date = datetime.strptime(TOURNAMENT_END_DATE, "%Y-%m-%d")
        fake_now = end_date - timedelta(days=30)

        with patch("metaculus_bot.constants.datetime") as mock_dt:
            mock_dt.strptime = datetime.strptime
            mock_dt.now.return_value = fake_now
            check_tournament_dates()

        assert "ended" not in caplog.text.lower()
        assert "update" not in caplog.text.lower()

    def test_warning_after_end_date(self, caplog: pytest.LogCaptureFixture) -> None:
        """Warning is logged when past end date but before hard stop."""
        end_date = datetime.strptime(TOURNAMENT_END_DATE, "%Y-%m-%d")
        fake_now = end_date + timedelta(days=7)  # 1 week past end

        with patch("metaculus_bot.constants.datetime") as mock_dt:
            mock_dt.strptime = datetime.strptime
            mock_dt.now.return_value = fake_now
            check_tournament_dates()

        assert TOURNAMENT_ID in caplog.text
        assert "ended" in caplog.text.lower() or "update" in caplog.text.lower()

    def test_error_after_hard_stop(self) -> None:
        """TournamentExpiredError raised when past hard stop date."""
        end_date = datetime.strptime(TOURNAMENT_END_DATE, "%Y-%m-%d")
        hard_stop = end_date + timedelta(weeks=TOURNAMENT_HARD_STOP_WEEKS)
        fake_now = hard_stop + timedelta(days=1)

        with patch("metaculus_bot.constants.datetime") as mock_dt:
            mock_dt.strptime = datetime.strptime
            mock_dt.now.return_value = fake_now
            with pytest.raises(TournamentExpiredError) as exc_info:
                check_tournament_dates()

        assert TOURNAMENT_ID in str(exc_info.value)
        assert "update" in str(exc_info.value).lower()

    def test_invalid_date_format_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Invalid date format logs warning but doesn't crash."""
        with patch("metaculus_bot.constants.TOURNAMENT_END_DATE", "not-a-date"):
            check_tournament_dates()

        assert "invalid" in caplog.text.lower()


class TestTournamentConfigFreshness:
    """
    Tests that FAIL if tournament config is stale.

    These tests use real dates (no mocking) to catch stale configs in CI.
    If these tests start failing, it's time to update constants.py for the new season!
    """

    def test_tournament_not_expired(self) -> None:
        """
        FAILS if the tournament hard stop date has passed.

        If this test fails, update these in metaculus_bot/constants.py:
        - TOURNAMENT_ID
        - TOURNAMENT_END_DATE
        - TOURNAMENT_HARD_STOP_WEEKS (if needed)

        Then update this test's error message with the new season info.
        """
        end_date = datetime.strptime(TOURNAMENT_END_DATE, "%Y-%m-%d")
        hard_stop_date = end_date + timedelta(weeks=TOURNAMENT_HARD_STOP_WEEKS)
        today = datetime.now()

        assert today <= hard_stop_date, (
            f"\n\n"
            f"{'=' * 70}\n"
            f"TOURNAMENT CONFIG IS STALE - ACTION REQUIRED\n"
            f"{'=' * 70}\n"
            f"Tournament '{TOURNAMENT_ID}' ended on {TOURNAMENT_END_DATE}.\n"
            f"Hard stop date ({hard_stop_date.date()}) has passed.\n"
            f"\n"
            f"Please update metaculus_bot/constants.py with the new season:\n"
            f"  - TOURNAMENT_ID (e.g., 'summer-aib-2026' or 'fall-aib-2026')\n"
            f"  - TOURNAMENT_END_DATE (approximate end date)\n"
            f"\n"
            f"Check https://www.metaculus.com/project/aib/ for current tournament info.\n"
            f"{'=' * 70}\n"
        )

    def test_tournament_end_date_is_valid_format(self) -> None:
        """TOURNAMENT_END_DATE should be a valid YYYY-MM-DD date."""
        try:
            parsed = datetime.strptime(TOURNAMENT_END_DATE, "%Y-%m-%d")
        except ValueError:
            pytest.fail(f"TOURNAMENT_END_DATE '{TOURNAMENT_END_DATE}' is not valid YYYY-MM-DD format")

        assert parsed.year >= 2025, f"TOURNAMENT_END_DATE year seems wrong: {parsed.year}"

    def test_tournament_id_looks_reasonable(self) -> None:
        """TOURNAMENT_ID should follow expected naming pattern."""
        assert TOURNAMENT_ID, "TOURNAMENT_ID should not be empty"
        assert isinstance(TOURNAMENT_ID, str), "TOURNAMENT_ID should be a string"
        # Should be either a slug like "spring-aib-2026" or numeric ID
        is_slug = "-" in TOURNAMENT_ID or TOURNAMENT_ID.isalpha()
        is_numeric = TOURNAMENT_ID.isdigit()
        assert is_slug or is_numeric, (
            f"TOURNAMENT_ID '{TOURNAMENT_ID}' doesn't look like a valid tournament ID. "
            f"Expected slug (e.g., 'spring-aib-2026') or numeric ID."
        )


class TestFallCupReminderGate:
    """The date gate and the FALL_CUP_CONFIGURED flag-off path, on injected dates.

    These stay green forever (no wall clock); the real-clock time bomb lives in
    TestFallCupReminderTimeBomb below.
    """

    REMINDER_DATE = date.fromisoformat(FALL_CUP_REMINDER_DATE)

    def test_not_due_before_the_reminder_date(self) -> None:
        assert not fall_cup_reminder_due(today=self.REMINDER_DATE - timedelta(days=1))

    def test_due_on_and_after_the_reminder_date(self) -> None:
        assert fall_cup_reminder_due(today=self.REMINDER_DATE)
        assert fall_cup_reminder_due(today=self.REMINDER_DATE + timedelta(days=90))

    def test_configured_flip_silences_it_even_after_the_date(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Flipping FALL_CUP_CONFIGURED is the operator's acknowledgment: it retires the
        reminder on any date, which is what makes the whole check easy to decommission."""
        monkeypatch.setattr(constants, "FALL_CUP_CONFIGURED", True)
        assert not fall_cup_reminder_due(today=self.REMINDER_DATE)
        assert not fall_cup_reminder_due(today=self.REMINDER_DATE + timedelta(days=365))

    def test_check_logs_the_loud_error_and_reports_fired(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR):
            assert check_fall_cup_reminder(today=self.REMINDER_DATE) is True
        assert "FALL_CUP_REMINDER" in caplog.text
        assert FALL_CUP_SLUG in caplog.text
        assert "FALL_CUP_CONFIGURED" in caplog.text  # the message must name its own off-switch

    def test_check_is_silent_before_the_date(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR):
            assert check_fall_cup_reminder(today=self.REMINDER_DATE - timedelta(days=1)) is False
        assert "FALL_CUP_REMINDER" not in caplog.text

    def test_check_is_silent_once_configured(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(constants, "FALL_CUP_CONFIGURED", True)
        with caplog.at_level(logging.ERROR):
            assert check_fall_cup_reminder(today=self.REMINDER_DATE) is False
        assert "FALL_CUP_REMINDER" not in caplog.text

    def test_reminder_date_is_valid_iso_format(self) -> None:
        assert date.fromisoformat(FALL_CUP_REMINDER_DATE).year == 2026


class TestFallCupReminderTimeBomb:
    """DELIBERATE TIME BOMB — this test is SUPPOSED to start failing on 2026-09-15.

    Operator-requested reminder (2026-08-31): the fall Metaculus Cup
    (`metaculus-cup-fall-2026`, project id 33108) is expected to open ~2026-09-20, the
    'Forecast on Metaculus Cup' workflow is disabled_manually, and the bare
    `metaculus-cup` slug that METACULUS_CUP_ID relies on now 404s. This test reads the
    REAL clock on purpose, exactly like test_tournament_not_expired above — do NOT
    "fix" it as flaky by mocking the date. Silence it by doing the configuration it
    is reminding you about, then flipping FALL_CUP_CONFIGURED = True in
    metaculus_bot/constants.py.
    """

    def test_fall_cup_is_configured_before_the_fall_season(self) -> None:
        assert not fall_cup_reminder_due(), (
            f"\n\n"
            f"{'=' * 70}\n"
            f"FALL METACULUS CUP IS NOT CONFIGURED - ACTION REQUIRED\n"
            f"{'=' * 70}\n"
            f"It is on/after {FALL_CUP_REMINDER_DATE} and FALL_CUP_CONFIGURED is False.\n"
            f"The fall cup ('{FALL_CUP_SLUG}', project id 33108) is expected to open\n"
            f"~2026-09-20. To silence this DELIBERATE reminder (it is not flaky):\n"
            f"  1. Point METACULUS_CUP_ID in metaculus_bot/constants.py at\n"
            f"     '{FALL_CUP_SLUG}' (the bare 'metaculus-cup' slug 404s now).\n"
            f"  2. Re-enable the 'Forecast on Metaculus Cup' workflow\n"
            f"     (.github/workflows/run_bot_on_metaculus_cup.yaml, disabled_manually).\n"
            f"  3. Update the season constants (TOURNAMENT_ID / TOURNAMENT_END_DATE)\n"
            f"     if the fall AIB season is also known by then.\n"
            f"  4. Flip FALL_CUP_CONFIGURED = True in metaculus_bot/constants.py.\n"
            f"{'=' * 70}\n"
        )
