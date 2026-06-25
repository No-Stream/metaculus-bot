"""Tests for the ablation-time mid-window monkey-patch context managers.

The bot's prompts include `_forecasting_window_str(question)`, which calls
`datetime.now()` to interpolate "Today: ..." into every forecasting prompt.
For ablation backtests on resolved 2026 questions, the negative "from now"
deltas tell the forecaster the question already resolved — a leakage vector.

This module patches `_forecasting_window_str` (and the
`{datetime.now().year}` token in `gap_fill_analyzer_prompt`) for the
duration of a context manager scoped to one specific question.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import cast

import pytest
from forecasting_tools import NumericQuestion

from metaculus_bot import prompts as prompts_module
from metaculus_bot.ablation.window_patch import (
    compute_mid_window_today,
    patched_gap_fill_year_for_question,
    patched_window_and_year_for_question,
    patched_window_for_question,
)


def _question(
    *,
    open_time: datetime = datetime(2026, 1, 1),
    scheduled_resolution_time: datetime = datetime(2026, 5, 1),
    id_of_question: int | None = 12345,
) -> SimpleNamespace:
    """Minimal question stub with the attributes _forecasting_window_str reads."""
    return SimpleNamespace(
        open_time=open_time,
        scheduled_resolution_time=scheduled_resolution_time,
        id_of_question=id_of_question,
    )


# ---------------------------------------------------------------------------
# compute_mid_window_today
# ---------------------------------------------------------------------------


class TestComputeMidWindowToday:
    def test_returns_midpoint_between_open_and_resolution(self) -> None:
        question = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
        )
        # Jan 1 → May 1 = 120 days; midpoint = Jan 1 + 60d = Mar 2.
        assert compute_mid_window_today(question) == datetime(2026, 3, 2)

    def test_asserts_on_missing_open_time(self) -> None:
        question = _question()
        question.open_time = None
        with pytest.raises(AssertionError):
            compute_mid_window_today(question)

    def test_asserts_on_missing_scheduled_resolution_time(self) -> None:
        question = _question()
        question.scheduled_resolution_time = None
        with pytest.raises(AssertionError):
            compute_mid_window_today(question)


# ---------------------------------------------------------------------------
# patched_window_for_question
# ---------------------------------------------------------------------------


class TestPatchedWindowForQuestion:
    def test_replaces_today_with_mid_window_date(self) -> None:
        question = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
        )
        with patched_window_for_question(question):
            output = prompts_module._forecasting_window_str(cast(NumericQuestion, question))

        # Mid-window of Jan 1 → May 1 is Mar 2.
        assert "Today: 2026-03-02" in output
        today_str = datetime.now().strftime("%Y-%m-%d")
        assert f"Today: {today_str}" not in output

    def test_restored_on_normal_exit(self) -> None:
        question = _question()
        before = prompts_module._forecasting_window_str
        with patched_window_for_question(question):
            assert prompts_module._forecasting_window_str is not before
        after = prompts_module._forecasting_window_str
        assert after is before

    def test_restored_on_exception(self) -> None:
        question = _question()
        before = prompts_module._forecasting_window_str

        with pytest.raises(RuntimeError, match="boom"):
            with patched_window_for_question(question):
                raise RuntimeError("boom")

        assert prompts_module._forecasting_window_str is before

    def test_falls_through_for_other_question(self) -> None:
        question_a = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
            id_of_question=11,
        )
        question_b = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
            id_of_question=22,
        )

        with patched_window_for_question(question_a):
            output_b = prompts_module._forecasting_window_str(cast(NumericQuestion, question_b))

        today_str = datetime.now().strftime("%Y-%m-%d")
        assert f"Today: {today_str}" in output_b

    def test_falls_through_for_other_question_when_id_is_none(self) -> None:
        """Synthetic test fixtures may have id_of_question=None;
        identity-by-id falls back to id() so distinct objects don't collide."""
        question_a = _question(id_of_question=None)
        question_b = _question(id_of_question=None)

        with patched_window_for_question(question_a):
            output_b = prompts_module._forecasting_window_str(cast(NumericQuestion, question_b))

        today_str = datetime.now().strftime("%Y-%m-%d")
        assert f"Today: {today_str}" in output_b

    def test_output_structure_matches_original(self) -> None:
        question = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
        )
        with patched_window_for_question(question):
            output = prompts_module._forecasting_window_str(cast(NumericQuestion, question))

        assert output.startswith("Today: ")
        assert "Question opened: 2026-01-01" in output
        assert "Scheduled to resolve: 2026-05-01" in output
        assert "Forecasting window: open date" in output
        assert "BEFORE the open date" in output
        assert "days ago" in output
        assert "days from now" in output

    def test_re_entrancy_raises(self) -> None:
        question = _question()
        with patched_window_for_question(question):
            with pytest.raises(RuntimeError, match="already active"):
                with patched_window_for_question(question):
                    pass

    def test_re_entrancy_does_not_corrupt_outer_restore(self) -> None:
        """If an inner enter raises RuntimeError, the outer cleanup must still run."""
        question = _question()
        before = prompts_module._forecasting_window_str

        with patched_window_for_question(question):
            with pytest.raises(RuntimeError):
                with patched_window_for_question(question):
                    pass

        assert prompts_module._forecasting_window_str is before


# ---------------------------------------------------------------------------
# patched_gap_fill_year_for_question
# ---------------------------------------------------------------------------


class TestPatchedGapFillYearForQuestion:
    def test_replaces_current_year_in_gap_fill_prompt(self) -> None:
        question = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
        )
        current_year = str(datetime.now().year)

        with patched_gap_fill_year_for_question(question):
            prompt = prompts_module.gap_fill_analyzer_prompt(
                question_text="Q?",
                resolution_criteria="rc",
                fine_print="fp",
                first_pass_research="fpr",
                is_benchmarking=False,
            )

        # The "no <YEAR> data" leak must NOT contain the actual current year.
        assert f"no {current_year} data" not in prompt
        # Should be replaced with year before the question's resolution year.
        assert "no 2025 data" in prompt

    def test_restored_on_normal_exit(self) -> None:
        question = _question()
        before = prompts_module.gap_fill_analyzer_prompt
        with patched_gap_fill_year_for_question(question):
            assert prompts_module.gap_fill_analyzer_prompt is not before
        assert prompts_module.gap_fill_analyzer_prompt is before

    def test_restored_on_exception(self) -> None:
        question = _question()
        before = prompts_module.gap_fill_analyzer_prompt
        with pytest.raises(RuntimeError, match="boom"):
            with patched_gap_fill_year_for_question(question):
                raise RuntimeError("boom")
        assert prompts_module.gap_fill_analyzer_prompt is before

    def test_threads_through_to_real_prompt_function(self) -> None:
        """The wrapper must still produce a useful prompt — sanity-check
        that the question text and other args still appear."""
        question = _question()
        with patched_gap_fill_year_for_question(question):
            prompt = prompts_module.gap_fill_analyzer_prompt(
                question_text="Will BTC hit 200k?",
                resolution_criteria="rc",
                fine_print="fp",
                first_pass_research="fpr",
                is_benchmarking=False,
            )
        assert "Will BTC hit 200k?" in prompt


# ---------------------------------------------------------------------------
# patched_window_and_year_for_question
# ---------------------------------------------------------------------------


class TestPatchedWindowAndYearForQuestion:
    def test_applies_both_patches(self) -> None:
        question = _question(
            open_time=datetime(2026, 1, 1),
            scheduled_resolution_time=datetime(2026, 5, 1),
        )
        current_year = str(datetime.now().year)

        with patched_window_and_year_for_question(question):
            window = prompts_module._forecasting_window_str(cast(NumericQuestion, question))
            gap_prompt = prompts_module.gap_fill_analyzer_prompt(
                question_text="Q?",
                resolution_criteria="rc",
                fine_print="fp",
                first_pass_research="fpr",
                is_benchmarking=False,
            )

        assert "Today: 2026-03-02" in window
        assert f"no {current_year} data" not in gap_prompt
        assert "no 2025 data" in gap_prompt

    def test_restores_both_on_exit(self) -> None:
        question = _question()
        before_window = prompts_module._forecasting_window_str
        before_gap = prompts_module.gap_fill_analyzer_prompt

        with patched_window_and_year_for_question(question):
            pass

        assert prompts_module._forecasting_window_str is before_window
        assert prompts_module.gap_fill_analyzer_prompt is before_gap

    def test_restores_both_on_exception(self) -> None:
        question = _question()
        before_window = prompts_module._forecasting_window_str
        before_gap = prompts_module.gap_fill_analyzer_prompt

        with pytest.raises(RuntimeError, match="boom"):
            with patched_window_and_year_for_question(question):
                raise RuntimeError("boom")

        assert prompts_module._forecasting_window_str is before_window
        assert prompts_module.gap_fill_analyzer_prompt is before_gap
