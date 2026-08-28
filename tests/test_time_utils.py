"""Tests for the shared datetime helpers (metaculus_bot/time_utils.py).

``parse_iso_utc`` is the ONE parser behind every era-bucketing read of
``bot_comment_created_at`` (``width_monitor.assign_era`` and
``analysis.max_step_clamp_screen`` both call it, replacing two copies that could
drift). Era attribution decides which config a resolved question is scored
against, so the contract that matters is: every ISO shape the archive carries
resolves to the same instant in UTC, and anything unparseable reads as None —
never as a silently-defaulted datetime that would file the record into the wrong
era.
"""

from datetime import UTC, datetime

import pytest

from metaculus_bot.time_utils import parse_iso_utc

EXPECTED = datetime(2026, 7, 21, 17, 7, 37, tzinfo=UTC)


class TestParseIsoUtc:
    @pytest.mark.parametrize(
        "raw",
        [
            "2026-07-21T17:07:37Z",  # Metaculus API / comment shape
            "2026-07-21T17:07:37+00:00",  # explicit zero offset
            "2026-07-21T10:07:37-07:00",  # same instant, local offset
            "2026-07-21T17:07:37",  # naive: assumed UTC, not local
        ],
    )
    def test_every_archived_shape_reads_the_same_instant(self, raw: str):
        parsed = parse_iso_utc(raw)
        assert parsed == EXPECTED
        assert parsed is not None
        assert parsed.tzinfo is not None

    def test_subsecond_precision_survives(self):
        assert parse_iso_utc("2026-07-21T17:07:37.123456Z") == EXPECTED.replace(microsecond=123456)

    @pytest.mark.parametrize("raw", [None, "", "not-a-date", "2026-13-45T99:00:00Z", "07/21/2026"])
    def test_absent_or_unparseable_reads_none(self, raw: str | None):
        assert parse_iso_utc(raw) is None
