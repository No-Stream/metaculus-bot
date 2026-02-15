"""Tests for tournament date validation.

Includes a test that will FAIL if run after the tournament end date + grace period,
forcing developers to update TOURNAMENT_ID and related constants for the new season.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from metaculus_bot.constants import (
    TOURNAMENT_END_DATE,
    TOURNAMENT_HARD_STOP_WEEKS,
    TOURNAMENT_ID,
    TournamentExpiredError,
    check_tournament_dates,
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
