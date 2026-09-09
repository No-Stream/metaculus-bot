"""Trigger-delivery watch over the bot workflows' GitHub Actions runs.

GitHub delivers about a fifth of this repository's scheduled cron firings (docs/operations.md
"Scheduling reliability"), so an external dispatcher (cron-job.org calling the
``workflow_dispatch`` API, twice an hour per workflow) is taking over the hourly trigger. Either
trigger can stop silently: the crons through GitHub's load shedding, the dispatcher through an
expired token or a paused job. This report reads ONE ``gh run list`` and prints, per bot workflow
and per UTC day, how many ``schedule`` and ``workflow_dispatch`` runs arrived and how they
concluded, next to what the ``- cron:`` entries in ``.github/workflows`` and the dispatcher
cadence say should have, and flags a day whose dispatched runs fall short or whose cron delivery
is below half. Today is prorated to its completed hours.

Read-only and free: one authenticated GitHub API read through ``gh``, no dispatch, no LLM, no
publish. ``--repo`` is pinned because ``origin`` here is the fork and ``upstream`` the Metaculus
template, so a bare ``gh`` targets the wrong repository.

Usage:
    uv run python scripts/dispatch_watch.py
    uv run python scripts/dispatch_watch.py --days 14 --expected-dispatch-per-hour 2
    make dispatch_watch
    make dispatch_watch ARGS="--days 3"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO = "No-Stream/metaculus-bot"
WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"
BOT_WORKFLOW_GLOB = "run_bot_on_*.yaml"
RUN_LIST_FIELDS = "workflowName,event,status,conclusion,createdAt"
DEFAULT_RUN_LIST_LIMIT = 1000
DEFAULT_DAYS = 7
DEFAULT_EXPECTED_DISPATCH_PER_HOUR = 2
SCHEDULE_DELIVERY_FLOOR = 0.5
HOURS_PER_DAY = 24
EVENT_SCHEDULE = "schedule"
EVENT_DISPATCH = "workflow_dispatch"

# Column-0 `name:` only, so a step's indented `- name:` never matches.
_WORKFLOW_NAME_RE = re.compile(r"^name:\s*(.+?)\s*$", re.MULTILINE)
_CRON_ENTRY_RE = re.compile(r"""^\s*-\s*cron:\s*["']([^"']+)["']""", re.MULTILINE)


@dataclass(frozen=True)
class WorkflowSpec:
    stem: str
    display_name: str  # the `name:` field, which is what `gh run list` reports as workflowName
    cron_entries: tuple[str, ...]


@dataclass(frozen=True)
class DayDelivery:
    day: date
    hours_elapsed: int
    scheduled: dict[str, int]  # conclusion (or status while running) -> runs
    dispatched: dict[str, int]
    expected_scheduled: int
    expected_dispatched: int

    @property
    def scheduled_total(self) -> int:
        return sum(self.scheduled.values())

    @property
    def dispatched_total(self) -> int:
        return sum(self.dispatched.values())

    def flags(self) -> list[str]:
        flags: list[str] = []
        if self.dispatched_total < self.expected_dispatched:
            flags.append(f"dispatch {self.dispatched_total}/{self.expected_dispatched} below expectation")
        if self.expected_scheduled:
            share = self.scheduled_total / self.expected_scheduled
            if share < SCHEDULE_DELIVERY_FLOOR:
                flags.append(
                    f"schedule {self.scheduled_total}/{self.expected_scheduled} ({share:.0%}) delivered, "
                    f"below {SCHEDULE_DELIVERY_FLOOR:.0%}"
                )
        return flags


def parse_workflow_yaml(stem: str, text: str) -> WorkflowSpec:
    """Display name and cron entries off the raw workflow text. Pure."""
    name_match = _WORKFLOW_NAME_RE.search(text)
    display_name = name_match.group(1).strip("\"'") if name_match else stem
    return WorkflowSpec(stem=stem, display_name=display_name, cron_entries=tuple(_CRON_ENTRY_RE.findall(text)))


def load_bot_workflows(workflows_dir: Path = WORKFLOWS_DIR) -> list[WorkflowSpec]:
    return [parse_workflow_yaml(path.stem, path.read_text()) for path in sorted(workflows_dir.glob(BOT_WORKFLOW_GLOB))]


def window_days(now: datetime, days: int) -> list[date]:
    """The last ``days`` UTC dates, oldest first, today included."""
    today = now.astimezone(UTC).date()
    return [today - timedelta(days=offset) for offset in range(days - 1, -1, -1)]


def _created_at(run: Mapping[str, Any]) -> datetime:
    return datetime.fromisoformat(run["createdAt"]).astimezone(UTC)


def tabulate_delivery(
    runs: Iterable[Mapping[str, Any]],
    spec: WorkflowSpec,
    *,
    now: datetime,
    days: int,
    expected_dispatch_per_hour: int,
) -> list[DayDelivery]:
    """One row per UTC day in the window for this workflow's schedule and dispatch runs. Pure."""
    by_day: dict[date, dict[str, Counter[str]]] = {
        day: {EVENT_SCHEDULE: Counter(), EVENT_DISPATCH: Counter()} for day in window_days(now, days)
    }
    for run in runs:
        counters = by_day.get(_created_at(run).date())
        if counters is None or run["workflowName"] != spec.display_name or run["event"] not in counters:
            continue
        counters[run["event"]][run["conclusion"] or run["status"]] += 1
    today = now.astimezone(UTC).date()
    rows: list[DayDelivery] = []
    for day, counters in by_day.items():
        hours = now.astimezone(UTC).hour if day == today else HOURS_PER_DAY
        rows.append(
            DayDelivery(
                day=day,
                hours_elapsed=hours,
                scheduled=dict(counters[EVENT_SCHEDULE]),
                dispatched=dict(counters[EVENT_DISPATCH]),
                expected_scheduled=len(spec.cron_entries) * hours,
                expected_dispatched=expected_dispatch_per_hour * hours,
            )
        )
    return rows


def truncation_warning(runs: Sequence[Mapping[str, Any]], *, limit: int, now: datetime, days: int) -> str | None:
    """A warning when the run list hit its cap before reaching the start of the window."""
    if not runs or len(runs) < limit:
        return None
    oldest = min(_created_at(run) for run in runs)
    window_start = datetime.combine(window_days(now, days)[0], datetime.min.time(), tzinfo=UTC)
    if oldest <= window_start:
        return None
    return (
        f"gh run list returned the {limit}-run cap and its oldest run ({oldest.isoformat()}) is inside the "
        f"window, so earlier days undercount; pass --limit above {limit}"
    )


def _conclusions(counts: Mapping[str, int]) -> str:
    return " ".join(f"{name}={n}" for name, n in sorted(counts.items(), key=lambda item: -item[1])) or "-"


def render_report(
    delivery_by_workflow: Sequence[tuple[WorkflowSpec, Sequence[DayDelivery]]],
    *,
    now: datetime,
    expected_dispatch_per_hour: int,
) -> str:
    """Render the watch as text. Pure: no clock read, no IO."""
    lines = [
        f"Trigger-delivery watch for {REPO}, per UTC day, as of {now.isoformat()}. "
        "Today is prorated to its completed hours."
    ]
    for spec, rows in delivery_by_workflow:
        lines.append("")
        lines.append(f"=== {spec.stem} ({spec.display_name}) ===")
        minutes = ", ".join(entry.split()[0] for entry in spec.cron_entries) or "none"
        lines.append(
            f"  {len(spec.cron_entries)} cron entries (minute {minutes}), so "
            f"{len(spec.cron_entries) * HOURS_PER_DAY} scheduled runs/day expected; "
            f"dispatcher {expected_dispatch_per_hour}/h, so {expected_dispatch_per_hour * HOURS_PER_DAY}/day"
        )
        if not any(row.scheduled_total or row.dispatched_total for row in rows):
            lines.append("  no runs in the window (workflow disabled, or never triggered)")
            continue
        lines.append(f"  {'day':<12}{'sched':>6}{'exp':>5}  {'disp':>5}{'exp':>5}  conclusions (schedule | dispatch)")
        for row in rows:
            flags = f"  <- {'; '.join(row.flags())}" if row.flags() else ""
            lines.append(
                f"  {row.day.isoformat():<12}{row.scheduled_total:>6}{row.expected_scheduled:>5}  "
                f"{row.dispatched_total:>5}{row.expected_dispatched:>5}  "
                f"{_conclusions(row.scheduled)} | {_conclusions(row.dispatched)}{flags}"
            )
    return "\n".join(lines)


def fetch_runs(limit: int) -> list[dict[str, Any]]:
    cmd = ["gh", "run", "list", "--repo", REPO, "--limit", str(limit), "--json", RUN_LIST_FIELDS]
    # S603: fixed `gh run list` argv, no shell; the only interpolation is the integer --limit.
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        logger.error(f"gh run list failed: {result.stderr.strip()}")
        sys.exit(1)
    return json.loads(result.stdout)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Per-UTC-day delivery of the bot workflows' schedule and workflow_dispatch triggers."
    )
    parser.add_argument(
        "--days", type=int, default=DEFAULT_DAYS, help="UTC days to report, today included (default: %(default)s)"
    )
    parser.add_argument(
        "--expected-dispatch-per-hour",
        type=int,
        default=DEFAULT_EXPECTED_DISPATCH_PER_HOUR,
        help="Dispatcher firings per workflow per hour (default: %(default)s); 0 turns the dispatch flag off",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_RUN_LIST_LIMIT,
        help="Runs to fetch with gh run list (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    specs = load_bot_workflows()
    if not specs:
        parser.error(f"no {BOT_WORKFLOW_GLOB} under {WORKFLOWS_DIR}")
    now = datetime.now(UTC)
    runs = fetch_runs(args.limit)
    warning = truncation_warning(runs, limit=args.limit, now=now, days=args.days)
    if warning is not None:
        logger.warning(warning)
    tables = [
        (
            spec,
            tabulate_delivery(
                runs, spec, now=now, days=args.days, expected_dispatch_per_hour=args.expected_dispatch_per_hour
            ),
        )
        for spec in specs
    ]
    print(render_report(tables, now=now, expected_dispatch_per_hour=args.expected_dispatch_per_hour))


if __name__ == "__main__":
    main()
