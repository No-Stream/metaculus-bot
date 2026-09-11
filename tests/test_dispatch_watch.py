"""Tests for the trigger-delivery watch (scripts/dispatch_watch.py).

The bot's hourly runs are triggered two ways: GitHub's own ``schedule`` crons, which deliver
about a fifth of this repository's firings, and an external dispatcher calling the
``workflow_dispatch`` API. Either can stop silently, so the watch tabulates both per UTC day
against what the cron entries and the dispatcher cadence say should have arrived. Every test
drives the pure functions on a fake run list and fake workflow text; the ``gh`` call is not
exercised.
"""

from datetime import UTC, date, datetime
from pathlib import Path

from scripts import dispatch_watch
from scripts.dispatch_watch import (
    WorkflowSpec,
    load_bot_workflows,
    parse_workflow_yaml,
    render_report,
    tabulate_delivery,
)

FAKE_WORKFLOW_YAML = """\
# Comments above the name, as the Mantic workflow has.
name: Forecast on Fake tournament

on:
  workflow_dispatch:
  schedule:
    # Make sure to skip already forecasted questions!
    - cron: "5 * * * *" # every hour at :05
    - cron: '25 * * * *'

jobs:
  forecast:
    steps:
      - name: Checkout
        uses: actions/checkout@v4
"""

FAKE_SPEC = WorkflowSpec(
    stem="run_bot_on_fake", display_name="Forecast on Fake tournament", cron_entries=("5 * * * *", "25 * * * *")
)
NOW = datetime(2026, 9, 9, 10, 30, tzinfo=UTC)


def _runs(day: str, event: str, count: int, *, conclusion: str = "success", workflow: str = FAKE_SPEC.display_name):
    return [
        {
            "workflowName": workflow,
            "event": event,
            "status": "completed" if conclusion else "in_progress",
            "conclusion": conclusion,
            "createdAt": f"{day}T{i % 24:02d}:{(i * 7) % 60:02d}:00Z",
        }
        for i in range(count)
    ]


class TestWorkflowParsing:
    def test_display_name_and_cron_entries_come_off_the_yaml_text(self):
        spec = parse_workflow_yaml("run_bot_on_fake", FAKE_WORKFLOW_YAML)
        assert spec == FAKE_SPEC

    def test_a_workflow_without_crons_has_no_entries(self):
        spec = parse_workflow_yaml("run_bot_on_manual", "name: Manual only\non:\n  workflow_dispatch:\n")
        assert spec.cron_entries == ()

    def test_the_real_bot_workflows_parse(self):
        specs = load_bot_workflows()
        by_stem = {spec.stem: spec for spec in specs}
        assert {"run_bot_on_tournament", "run_bot_on_metaculus_cup", "run_bot_on_mantic"} <= set(by_stem)
        for spec in specs:
            assert spec.display_name.startswith("Forecast on"), spec
            assert all(len(entry.split()) == 5 for entry in spec.cron_entries), spec

    def test_a_missing_workflows_dir_yields_nothing(self, tmp_path: Path):
        assert load_bot_workflows(tmp_path / "absent") == []


class TestTabulation:
    def _runs(self):
        return (
            _runs("2026-09-07", "schedule", 30)
            + _runs("2026-09-07", "workflow_dispatch", 48)
            + _runs("2026-09-08", "schedule", 10)
            + _runs("2026-09-09", "schedule", 12)
            + _runs("2026-09-09", "workflow_dispatch", 19)
            + _runs("2026-09-09", "workflow_dispatch", 1, conclusion="failure")
            + _runs("2026-09-09", "push", 3, workflow="CI")
            + _runs("2026-09-09", "schedule", 5, workflow="Forecast on Metaculus Cup")
            + _runs("2026-09-01", "schedule", 40)
        )

    def _table(self):
        return tabulate_delivery(self._runs(), FAKE_SPEC, now=NOW, days=3, expected_dispatch_per_hour=2)

    def test_one_row_per_utc_day_in_the_window_oldest_first(self):
        assert [row.day for row in self._table()] == [date(2026, 9, 7), date(2026, 9, 8), date(2026, 9, 9)]

    def test_counts_split_by_event_and_conclusion(self):
        rows = {row.day: row for row in self._table()}
        assert rows[date(2026, 9, 7)].scheduled == {"success": 30}
        assert rows[date(2026, 9, 7)].dispatched == {"success": 48}
        assert rows[date(2026, 9, 8)].scheduled == {"success": 10}
        assert rows[date(2026, 9, 8)].dispatched == {}
        assert rows[date(2026, 9, 9)].dispatched == {"success": 19, "failure": 1}

    def test_other_workflows_events_and_days_are_ignored(self):
        rows = {row.day: row for row in self._table()}
        assert rows[date(2026, 9, 9)].scheduled == {"success": 12}
        assert sum(row.scheduled_total for row in rows.values()) == 52

    def test_expectations_come_from_the_cron_entries_and_the_dispatch_cadence(self):
        rows = {row.day: row for row in self._table()}
        assert (rows[date(2026, 9, 7)].expected_scheduled, rows[date(2026, 9, 7)].expected_dispatched) == (48, 48)

    def test_today_is_prorated_to_the_completed_hours(self):
        today = {row.day: row for row in self._table()}[date(2026, 9, 9)]
        assert today.hours_elapsed == 10
        assert (today.expected_scheduled, today.expected_dispatched) == (20, 20)
        assert today.flags() == []

    def test_a_day_below_expectation_is_flagged_on_both_axes(self):
        rows = {row.day: row for row in self._table()}
        assert rows[date(2026, 9, 7)].flags() == []
        assert rows[date(2026, 9, 8)].flags() == [
            "dispatch 0/48 below expectation",
            "schedule 10/48 (21%) delivered, below 50%",
        ]

    def test_an_in_progress_run_counts_under_its_status(self):
        rows = tabulate_delivery(
            _runs("2026-09-09", "workflow_dispatch", 1, conclusion=""),
            FAKE_SPEC,
            now=NOW,
            days=1,
            expected_dispatch_per_hour=0,
        )
        assert rows[0].dispatched == {"in_progress": 1}

    def test_no_cron_entries_means_no_schedule_flag(self):
        manual = WorkflowSpec(stem="run_bot_on_manual", display_name="Manual", cron_entries=())
        rows = tabulate_delivery([], manual, now=NOW, days=1, expected_dispatch_per_hour=0)
        assert rows[0].expected_scheduled == 0
        assert rows[0].flags() == []


class TestRendering:
    def test_the_report_names_each_workflow_and_its_flags(self):
        runs = TestTabulation()._runs()
        table = tabulate_delivery(runs, FAKE_SPEC, now=NOW, days=3, expected_dispatch_per_hour=2)
        text = render_report([(FAKE_SPEC, table)], now=NOW, expected_dispatch_per_hour=2)
        assert "=== run_bot_on_fake (Forecast on Fake tournament) ===" in text
        assert "2 cron entries" in text
        assert "2026-09-08" in text
        assert "dispatch 0/48 below expectation" in text
        assert "success=19 failure=1" in text

    def test_a_workflow_with_no_runs_gets_one_line_not_a_table(self):
        idle = WorkflowSpec(stem="run_bot_on_minibench", display_name="Minibench", cron_entries=("8 * * * *",))
        table = tabulate_delivery([], idle, now=NOW, days=3, expected_dispatch_per_hour=2)
        text = render_report([(idle, table)], now=NOW, expected_dispatch_per_hour=2)
        assert "no runs in the window (workflow disabled, or never triggered)" in text
        assert "2026-09-08" not in text

    def test_truncation_is_reported_when_the_oldest_run_is_inside_the_window(self):
        runs = _runs("2026-09-08", "schedule", 5)
        assert dispatch_watch.truncation_warning(runs, limit=5, now=NOW, days=3) is not None
        assert dispatch_watch.truncation_warning(runs, limit=1000, now=NOW, days=3) is None
        old = _runs("2026-09-01", "schedule", 5)
        assert dispatch_watch.truncation_warning(old, limit=5, now=NOW, days=3) is None
