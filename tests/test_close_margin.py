"""Tests for per-question CLOSE_MARGIN marker formatting (metaculus_bot/close_margin.py).

The marker is emitted at submission time in forecaster.py and harvested by
scripts/telemetry/markers.py. These tests pin the emitted format so a producer-side
change breaks loudly (same source-of-truth stance as tests/test_telemetry_markers.py).
"""

import logging
from datetime import UTC, datetime
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import ForecastBot, GeneralLlm, MetaculusQuestion
from forecasting_tools.data_models.questions import BinaryQuestion

from metaculus_bot.close_margin import format_close_margin_marker
from metaculus_bot.forecaster import TemplateForecaster


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
        submitted = datetime(2026, 7, 19, 13, 50, 0, tzinfo=UTC)  # 36600s before close
        marker = format_close_margin_marker(q, submitted)
        assert marker == (
            "CLOSE_MARGIN: question=44620 close_time=2026-07-20T00:00:00+00:00 "
            "submitted_at=2026-07-19T13:50:00+00:00 window_s=864000 margin_s=36600 margin_frac=0.0424"
        )

    def test_missing_close_time_skips(self):
        q = _question(close_time=None, open_time=datetime(2026, 7, 10, tzinfo=UTC))
        assert format_close_margin_marker(q, datetime.now(UTC)) is None

    def test_missing_open_time_renders_na_window_and_frac(self):
        q = _question(close_time=datetime(2026, 7, 20, 0, 0, 0, tzinfo=UTC), open_time=None)
        submitted = datetime(2026, 7, 19, 0, 0, 0, tzinfo=UTC)  # exactly 1 day = 86400s
        marker = format_close_margin_marker(q, submitted)
        assert marker == (
            "CLOSE_MARGIN: question=44620 close_time=2026-07-20T00:00:00+00:00 "
            "submitted_at=2026-07-19T00:00:00+00:00 window_s=n/a margin_s=86400 margin_frac=n/a"
        )

    def test_nonpositive_window_renders_na_frac(self):
        # Degenerate: open_time == close_time (window 0) — no division, frac is n/a but margin still emitted.
        moment = datetime(2026, 7, 20, tzinfo=UTC)
        q = _question(close_time=moment, open_time=moment)
        submitted = datetime(2026, 7, 19, 12, 0, 0, tzinfo=UTC)
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
        submitted = datetime(2026, 7, 19, 1, 0, 0, tzinfo=UTC)  # 1h past close
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
        aware = datetime(2026, 7, 19, 13, 50, 0, tzinfo=UTC)
        assert format_close_margin_marker(q, naive) == format_close_margin_marker(q, aware)


def _mock_forecaster_llm() -> MagicMock:
    llm = MagicMock(spec=GeneralLlm)
    llm.model = "mock_model"
    llm.invoke = AsyncMock(return_value="mock reasoning")
    return llm


def _make_bot(*, publish: bool) -> TemplateForecaster:
    """A minimal 2-forecaster bot; only ``publish_reports_to_metaculus`` matters for the seam."""
    llm = _mock_forecaster_llm()
    llms_config = {
        "forecasters": [llm, llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    return TemplateForecaster(
        llms=cast("dict[str, str | GeneralLlm]", llms_config),
        publish_reports_to_metaculus=publish,
        min_forecasters_to_publish=1,
    )


class TestRunIndividualQuestionEmitsMarker:
    """The forecaster.py emit seam: ``_run_individual_question`` calls ``super()`` then logs
    the CLOSE_MARGIN marker only when publishing (submission time) AND only when the formatter
    returns a line. ``format_close_margin_marker`` itself is covered above; here we pin the
    gating + logging around it, with the parent method and the formatter both patched.
    """

    @pytest.mark.asyncio
    async def test_marker_logged_when_publishing_and_marker_present(self, caplog):
        bot = _make_bot(publish=True)
        question = MagicMock(spec=MetaculusQuestion)
        sentinel_report = object()
        marker_line = "CLOSE_MARGIN: question=42 close_time=2026-07-20T00:00:00+00:00 margin_s=100"
        with (
            patch.object(ForecastBot, "_run_individual_question", new=AsyncMock(return_value=sentinel_report)),
            patch("metaculus_bot.forecaster.format_close_margin_marker", return_value=marker_line) as fmt,
            caplog.at_level(logging.INFO, logger="metaculus_bot.forecaster"),
        ):
            result = await bot._run_individual_question(question)

        assert result is sentinel_report, "the base report must pass through unchanged"
        fmt.assert_called_once()
        assert any(marker_line in rec.getMessage() for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_no_marker_when_publishing_disabled(self, caplog):
        bot = _make_bot(publish=False)
        question = MagicMock(spec=MetaculusQuestion)
        sentinel_report = object()
        with (
            patch.object(ForecastBot, "_run_individual_question", new=AsyncMock(return_value=sentinel_report)),
            patch("metaculus_bot.forecaster.format_close_margin_marker", return_value="CLOSE_MARGIN: ...") as fmt,
            caplog.at_level(logging.INFO, logger="metaculus_bot.forecaster"),
        ):
            result = await bot._run_individual_question(question)

        assert result is sentinel_report
        # gated behind publish_reports_to_metaculus, so the formatter is never even called
        fmt.assert_not_called()
        assert not any("CLOSE_MARGIN" in rec.getMessage() for rec in caplog.records)

    @pytest.mark.asyncio
    async def test_no_marker_logged_when_formatter_returns_none(self, caplog):
        bot = _make_bot(publish=True)
        question = MagicMock(spec=MetaculusQuestion)
        sentinel_report = object()
        with (
            patch.object(ForecastBot, "_run_individual_question", new=AsyncMock(return_value=sentinel_report)),
            patch("metaculus_bot.forecaster.format_close_margin_marker", return_value=None) as fmt,
            caplog.at_level(logging.INFO, logger="metaculus_bot.forecaster"),
        ):
            result = await bot._run_individual_question(question)

        assert result is sentinel_report
        fmt.assert_called_once()
        assert not any("CLOSE_MARGIN" in rec.getMessage() for rec in caplog.records)
