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
    METACULUS_CUP_ID,
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

    These stay green forever (no wall clock). The reminder ships DISCHARGED — the fall cup
    was configured on 2026-09-03 and ``FALL_CUP_CONFIGURED`` is True — so every test of the
    date gate itself has to re-arm the flag first, or it would pass for the wrong reason
    (the flag short-circuits before the date is ever compared). ``rearmed`` is that
    re-arm, and it is also what the next cup season's operator does for real.
    """

    REMINDER_DATE = date.fromisoformat(FALL_CUP_REMINDER_DATE)

    @pytest.fixture
    def rearmed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(constants, "FALL_CUP_CONFIGURED", False)

    @pytest.mark.usefixtures("rearmed")
    def test_not_due_before_the_reminder_date(self) -> None:
        assert not fall_cup_reminder_due(today=self.REMINDER_DATE - timedelta(days=1))

    @pytest.mark.usefixtures("rearmed")
    def test_due_on_and_after_the_reminder_date(self) -> None:
        assert fall_cup_reminder_due(today=self.REMINDER_DATE)
        assert fall_cup_reminder_due(today=self.REMINDER_DATE + timedelta(days=90))

    def test_shipped_default_is_silent_on_every_date(self) -> None:
        """FALL_CUP_CONFIGURED is the operator's acknowledgment: it retires the reminder on
        any date, which is what makes the whole check easy to decommission — and, since
        2026-09-03, what keeps the discharged time bomb below from re-arming itself."""
        assert not fall_cup_reminder_due(today=self.REMINDER_DATE)
        assert not fall_cup_reminder_due(today=self.REMINDER_DATE + timedelta(days=365))

    @pytest.mark.usefixtures("rearmed")
    def test_check_logs_the_loud_error_and_reports_fired(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR):
            assert check_fall_cup_reminder(today=self.REMINDER_DATE) is True
        assert "FALL_CUP_REMINDER" in caplog.text
        assert FALL_CUP_SLUG in caplog.text
        assert "FALL_CUP_CONFIGURED" in caplog.text  # the message must name its own off-switch

    @pytest.mark.usefixtures("rearmed")
    def test_check_is_silent_before_the_date(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR):
            assert check_fall_cup_reminder(today=self.REMINDER_DATE - timedelta(days=1)) is False
        assert "FALL_CUP_REMINDER" not in caplog.text

    def test_check_is_silent_once_configured(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.ERROR):
            assert check_fall_cup_reminder(today=self.REMINDER_DATE) is False
        assert "FALL_CUP_REMINDER" not in caplog.text

    def test_reminder_date_is_valid_iso_format(self) -> None:
        assert date.fromisoformat(FALL_CUP_REMINDER_DATE).year == 2026


class TestFallCupStaysConfigured:
    """The 2026-09-15 time bomb, DISCHARGED on 2026-09-03 and inverted into a pin.

    It used to read the real clock and start failing on FALL_CUP_REMINDER_DATE, as the
    reminder that the fall Metaculus Cup needed configuring. It got configured: Metaculus
    granted $1,500 of API credits for the fall season, METACULUS_CUP_ID now names the dated
    fall slug (project 33108, API-verified), and the cup workflow runs the tournament's
    hourly split cron. So the same real-clock assertion now passes forever, and these tests
    keep it that way — they fail if someone re-points the cup at the retired undated slug,
    or re-arms the reminder without meaning to.

    Re-arming for the NEXT cup season is deliberate and legitimate (see constants.py): set
    FALL_CUP_CONFIGURED = False and re-date FALL_CUP_REMINDER_DATE. Doing that turns this
    class red on/after that date, which is exactly the reminder working again.
    """

    def test_the_reminder_is_discharged(self) -> None:
        # Reads the REAL clock on purpose, exactly like test_tournament_not_expired above —
        # do NOT "fix" it as flaky by mocking the date. Deliberately asserts the DUE verdict
        # rather than the flag, so re-arming for a future season stays green until that
        # season's reminder date actually arrives.
        assert not fall_cup_reminder_due(), (
            f"\n\n"
            f"{'=' * 70}\n"
            f"METACULUS CUP IS NOT CONFIGURED - ACTION REQUIRED\n"
            f"{'=' * 70}\n"
            f"It is on/after {FALL_CUP_REMINDER_DATE} and FALL_CUP_CONFIGURED is False.\n"
            f"This is a DELIBERATE reminder, not a flaky test. To silence it:\n"
            f"  1. Point METACULUS_CUP_ID in metaculus_bot/constants.py at the new\n"
            f"     season's DATED slug (currently '{FALL_CUP_SLUG}'; the undated\n"
            f"     'metaculus-cup' slug is rejected with HTTP 400 and no auto-resolving\n"
            f"     spelling exists).\n"
            f"  2. Enable the 'Forecast on Metaculus Cup' workflow on GitHub\n"
            f"     (.github/workflows/run_bot_on_metaculus_cup.yaml).\n"
            f"  3. Update the season constants (TOURNAMENT_ID / TOURNAMENT_END_DATE)\n"
            f"     if the successor bot tournament is known by then.\n"
            f"  4. Flip FALL_CUP_CONFIGURED = True in metaculus_bot/constants.py.\n"
            f"{'=' * 70}\n"
        )

    def test_the_cup_id_is_a_dated_season_slug(self) -> None:
        # The undated slug is the specific failure this whole block exists for: Metaculus
        # answers HTTP 400 for it, so a cup run under it finds no questions and forfeits the
        # season silently.
        assert METACULUS_CUP_ID != "metaculus-cup"
        assert METACULUS_CUP_ID == FALL_CUP_SLUG, "FALL_CUP_SLUG must stay an alias of the configured cup"
        assert METACULUS_CUP_ID == "metaculus-cup-fall-2026", (
            "Fall 2026 cup, API-verified 2026-09-03 as project 33108 "
            "(/api/projects/tournaments/metaculus-cup-fall-2026/). Update this pin when the "
            "cup rolls to the next season."
        )
