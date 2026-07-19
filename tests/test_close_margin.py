"""Tests for per-question CLOSE_MARGIN marker formatting (metaculus_bot/close_margin.py).

The marker is emitted at submission time in forecaster.py and harvested by
scripts/telemetry/markers.py. These tests pin the emitted format so a producer-side
change breaks loudly (same source-of-truth stance as tests/test_telemetry_markers.py).
"""

from datetime import datetime, timezone

from forecasting_tools.data_models.questions import BinaryQuestion

from metaculus_bot.close_margin import format_close_margin_marker


def _question(*, close_time, open_time, qid=44620) -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will X happen by the deadline?",
        id_of_question=qid,
        close_time=close_time,
        open_time=open_time,
    )


class TestFormatCloseMarginMarker:
    def test_full_marker_naive_utc_times(self):
        # Naive UTC datetimes (how the framework parses Metaculus API timestamps).
        q = _question(
            close_time=datetime(2026, 7, 20, 0, 0, 0),
            open_time=datetime(2026, 7, 10, 0, 0, 0),  # 10-day (864000s) window
        )
        submitted = datetime(2026, 7, 19, 13, 50, 0, tzinfo=timezone.utc)  # 36600s before close
        marker = format_close_margin_marker(q, submitted)
        assert marker == (
            "CLOSE_MARGIN: question=44620 close_time=2026-07-20T00:00:00+00:00 "
            "submitted_at=2026-07-19T13:50:00+00:00 window_s=864000 margin_s=36600 margin_frac=0.0424"
        )

    def test_missing_close_time_skips(self):
        q = _question(close_time=None, open_time=datetime(2026, 7, 10, tzinfo=timezone.utc))
        assert format_close_margin_marker(q, datetime.now(timezone.utc)) is None

    def test_missing_open_time_renders_na_window_and_frac(self):
        q = _question(close_time=datetime(2026, 7, 20, 0, 0, 0, tzinfo=timezone.utc), open_time=None)
        submitted = datetime(2026, 7, 19, 0, 0, 0, tzinfo=timezone.utc)  # exactly 1 day = 86400s
        marker = format_close_margin_marker(q, submitted)
        assert marker == (
            "CLOSE_MARGIN: question=44620 close_time=2026-07-20T00:00:00+00:00 "
            "submitted_at=2026-07-19T00:00:00+00:00 window_s=n/a margin_s=86400 margin_frac=n/a"
        )

    def test_nonpositive_window_renders_na_frac(self):
        # Degenerate: open_time == close_time (window 0) — no division, frac is n/a but margin still emitted.
        moment = datetime(2026, 7, 20, tzinfo=timezone.utc)
        q = _question(close_time=moment, open_time=moment)
        submitted = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)
        marker = format_close_margin_marker(q, submitted)
        assert marker is not None
        assert "window_s=n/a" in marker
        assert "margin_frac=n/a" in marker
        assert "margin_s=43200" in marker

    def test_missed_close_gives_negative_margin(self):
        # Submitted AFTER close (a missed deadline) — margin_s and margin_frac go negative.
        q = _question(
            close_time=datetime(2026, 7, 19, 0, 0, 0),
            open_time=datetime(2026, 7, 9, 0, 0, 0),  # 864000s window
        )
        submitted = datetime(2026, 7, 19, 1, 0, 0, tzinfo=timezone.utc)  # 1h past close
        marker = format_close_margin_marker(q, submitted)
        assert marker is not None
        assert "margin_s=-3600" in marker
        assert "margin_frac=-0.0042" in marker

    def test_naive_and_aware_submitted_time_agree(self):
        # A naive submitted_at is treated as UTC, matching an explicitly-UTC one.
        q = _question(
            close_time=datetime(2026, 7, 20, 0, 0, 0),
            open_time=datetime(2026, 7, 10, 0, 0, 0),
        )
        naive = datetime(2026, 7, 19, 13, 50, 0)
        aware = datetime(2026, 7, 19, 13, 50, 0, tzinfo=timezone.utc)
        assert format_close_margin_marker(q, naive) == format_close_margin_marker(q, aware)
