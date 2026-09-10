"""Tests for scripts/cost_report.py, the standing cost-per-question instrument.

Synthetic archive rows shaped exactly like ``backtests/telemetry_archive/*.jsonl``: role-ledger rows
in both the pre-token (before 2026-09-09) and full shapes, ``FORECASTERS_SURVIVED`` lines for the
runs that predate ``CREDIT_RUN_SUMMARY``, and one run summary. The arithmetic under test is the
denominator handling: which marker counts a run's questions, which runs a role divides by, and that
token figures never divide tokens the archive did not record.
"""

from __future__ import annotations

from datetime import date

import pytest

from scripts.cost_report import (
    QUESTIONS_FROM_SUMMARY,
    QUESTIONS_FROM_SURVIVED,
    QuestionCount,
    question_counts,
    render_report,
    role_costs,
    rows_since,
    run_costs,
    weekly_trend,
)


def _role_row(run_id: str, run_date: str, role: str, *, key: str = "donated", calls: int = 4, **fields) -> dict:
    """One ``credit_role_spend`` record; pass token fields to get the 2026-09-09 shape."""
    return {
        "marker": "credit_role_spend",
        "run_id": run_id,
        "workflow": "tournament",
        "run_date": run_date,
        "role": role,
        "key": key,
        "calls": calls,
        "costed_calls": calls,
        **fields,
    }


def _survived(run_id: str, qid: int) -> dict:
    return {"marker": "forecasters_survived", "run_id": run_id, "qid": qid, "seq": qid}


def _summary(run_id: str, n_questions: int) -> dict:
    return {"marker": "credit_run_summary", "run_id": run_id, "n_questions": n_questions}


# Run A predates every 2026-09-09 field: usd only, questions from FORECASTERS_SURVIVED.
OLD_RUN = [
    _role_row("A", "2026-09-03T16:07:12Z", "forecaster:openai", usd=1.00, byok_usd=1.00),
    _role_row("A", "2026-09-03T16:07:12Z", "gap_fill_v2_driver", usd=0.60, byok_usd=0.60, calls=40),
]
# Run B carries charged_usd, the token tail and max_prompt_tokens, and a CREDIT_RUN_SUMMARY.
NEW_RUN = [
    _role_row(
        "B",
        "2026-09-10T02:00:00Z",
        "forecaster:openai",
        usd=1.20,
        byok_usd=1.20,
        charged_usd=1.20,
        prompt_tokens=68_000,
        completion_tokens=9_000,
        cached_tokens=0,
        reasoning_tokens=8_000,
        max_prompt_tokens=17_000,
    ),
    _role_row(
        "B",
        "2026-09-10T02:00:00Z",
        "gap_fill_v2_driver",
        usd=0.40,
        byok_usd=0.40,
        charged_usd=0.40,
        calls=40,
        prompt_tokens=300_000,
        completion_tokens=3_600,
        cached_tokens=261_000,
        reasoning_tokens=2_400,
        max_prompt_tokens=41_176,
    ),
    _role_row("B", "2026-09-10T02:00:00Z", "perplexity_research", key="direct", calls=1, costed_calls=0),
]
SURVIVED = [_survived("A", 101), _survived("A", 102), _survived("A", 101), _survived("B", 201)]
SUMMARIES = [_summary("B", 4)]


class TestQuestionCounts:
    def test_summary_wins_over_survived_lines_and_survived_counts_distinct_questions(self) -> None:
        counts = question_counts(SUMMARIES, SURVIVED)
        # Run A: two distinct qids across three lines (one duplicate); run B: the summary's count, not its one line.
        assert counts["A"] == QuestionCount(2, QUESTIONS_FROM_SURVIVED)
        assert counts["B"] == QuestionCount(4, QUESTIONS_FROM_SUMMARY)

    def test_window_filter_compares_the_run_date_prefix(self) -> None:
        rows = rows_since(OLD_RUN + NEW_RUN, date(2026, 9, 4))
        assert {row["run_id"] for row in rows} == {"B"}


class TestRunCosts:
    def test_per_run_dollars_over_the_counted_questions(self) -> None:
        runs = run_costs(OLD_RUN + NEW_RUN, question_counts(SUMMARIES, SURVIVED))
        assert [run.run_id for run in runs] == ["A", "B"]
        run_a, run_b = runs
        assert run_a.charged_usd == pytest.approx(1.60)
        assert run_a.usd_per_question == pytest.approx(0.80)
        assert run_b.charged_usd == pytest.approx(1.60)
        assert run_b.usd_per_question == pytest.approx(0.40)

    def test_run_without_any_question_marker_reports_no_rate(self) -> None:
        (run,) = run_costs(OLD_RUN, {})
        assert run.questions is None
        assert run.charged_usd == pytest.approx(1.60)
        assert run.usd_per_question is None


class TestRoleCosts:
    def test_dollars_divide_by_every_counted_run_and_tokens_by_the_token_runs_only(self) -> None:
        roles = {role.role: role for role in role_costs(OLD_RUN + NEW_RUN, question_counts(SUMMARIES, SURVIVED))}

        forecaster = roles["forecaster:openai"]
        # Dollars: $2.20 over the 6 questions of both runs.
        assert forecaster.n_questions == 6
        assert forecaster.usd_per_question == pytest.approx(2.20 / 6)
        # Tokens: only run B recorded them, so they divide by ITS 4 questions, not by 6.
        assert forecaster.token_questions == 4
        assert forecaster.prompt_tokens_per_question == pytest.approx(17_000)
        assert forecaster.completion_tokens_per_question == pytest.approx(2_250)
        assert forecaster.reasoning_tokens_per_question == pytest.approx(2_000)
        assert forecaster.cached_share == pytest.approx(0.0)
        assert forecaster.max_prompt_tokens == 17_000

        driver = roles["gap_fill_v2_driver"]
        assert driver.usd_per_question == pytest.approx(1.00 / 6)
        assert driver.cached_share == pytest.approx(0.87)
        assert driver.max_prompt_tokens == 41_176

    def test_uncosted_role_reads_none_not_zero_and_sorts_last(self) -> None:
        roles = role_costs(OLD_RUN + NEW_RUN, question_counts(SUMMARIES, SURVIVED))
        assert [role.role for role in roles] == ["forecaster:openai", "gap_fill_v2_driver", "perplexity_research"]
        perplexity = roles[-1]
        assert perplexity.charged_usd is None
        assert perplexity.usd_per_question is None
        assert perplexity.prompt_tokens is None
        assert perplexity.max_prompt_tokens is None

    def test_a_run_without_a_question_count_is_excluded_from_the_role_denominators(self) -> None:
        """Its dollars without its questions would inflate every per-question figure."""
        counts = question_counts(SUMMARIES, [record for record in SURVIVED if record["run_id"] != "A"])
        roles = {role.role: role for role in role_costs(OLD_RUN + NEW_RUN, counts)}
        assert roles["forecaster:openai"].n_questions == 4
        assert roles["forecaster:openai"].charged_usd == pytest.approx(1.20)


class TestWeeklyTrend:
    def test_medians_split_at_seven_days_and_report_run_counts(self) -> None:
        runs = run_costs(OLD_RUN + NEW_RUN, question_counts(SUMMARIES, SURVIVED))
        this_week, prior_week = weekly_trend(runs, today=date(2026, 9, 10))
        assert (this_week.median_usd_per_question, this_week.n_runs) == (pytest.approx(0.40), 1)
        assert (prior_week.median_usd_per_question, prior_week.n_runs) == (pytest.approx(0.80), 1)

    def test_empty_window_reads_none(self) -> None:
        runs = run_costs(OLD_RUN, question_counts([], SURVIVED))
        this_week, prior_week = weekly_trend(runs, today=date(2026, 10, 1))
        assert (this_week.median_usd_per_question, this_week.n_runs) == (None, 0)
        assert (prior_week.median_usd_per_question, prior_week.n_runs) == (None, 0)


class TestRenderReport:
    def test_report_carries_the_run_table_the_role_table_and_the_trend(self) -> None:
        counts = question_counts(SUMMARIES, SURVIVED)
        rows = OLD_RUN + NEW_RUN
        runs = run_costs(rows, counts)
        report = render_report(
            runs, role_costs(rows, counts), weekly_trend(runs, date(2026, 9, 10)), days=30, since=date(2026, 8, 11)
        )
        assert "2 runs, 6 questions, $3.20 charged" in report
        # The survived-counted run is starred; the summary-counted one is not.
        assert "2026-09-03T16:07  A            tournament              2*    1.6000      0.8000" in report
        assert "2026-09-10T02:00  B            tournament               4    1.6000      0.4000" in report
        assert "per role, over 6 questions:" in report
        assert "gap_fill_v2_driver           0.1667       75,000     87%        900          600      41,176" in report
        assert "perplexity_research             n/a          n/a     n/a        n/a          n/a         n/a" in report
        assert "total (costed roles)         0.5333" in report
        assert "trend: median $/question this week 0.4000 (n=1) vs prior week 0.8000 (n=1)" in report
