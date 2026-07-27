"""Guards on the scheduled archive-sync job's failure semantics.

These assert on the Makefile recipe, the launchd wrapper, and the plist rather than on
Python behavior, because that is where the 2026-06/07 data-loss bug lived: every
scheduled run from 2026-06-28 through 2026-07-26 failed and nothing noticed. The three
compounding causes, one test class each:

1. ``sync_all`` ran the Metaculus-comment backfill first and let its failure abort the
   recipe under ``set -euo pipefail``, so a failure in the half whose data never expires
   killed the half that GitHub deletes at 90 days.
2. ``run_sync.sh`` invoked make with no network-readiness wait, and launchd runs a
   missed job at the next wake — before Wi-Fi re-associates.
3. Nothing surfaced the failure: no sentinel, no notification, and the plist woke once a
   week so one bad wake cost a full week.

They are cheap string assertions on files that no test otherwise reads, which is exactly
why the regression was invisible for six weeks.
"""

from __future__ import annotations

import plistlib
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MAKEFILE = _REPO_ROOT / "Makefile"
_SYNC_DIR = _REPO_ROOT / "scripts" / "research_sync"
_RUN_SYNC = _SYNC_DIR / "run_sync.sh"
_PLIST = _SYNC_DIR / "com.metaculusbot.research-sync.plist"


def _sync_all_recipe() -> list[str]:
    """The command lines of the Makefile's ``sync_all`` recipe, in order.

    A recipe body is the tab-indented block following the target line, so we slice from
    ``sync_all:`` to the first line that is neither tab-indented nor blank.
    """
    lines = _MAKEFILE.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("sync_all:"))
    body: list[str] = []
    for line in lines[start + 1 :]:
        if not line.startswith("\t"):
            if line.strip() == "":
                continue
            break
        body.append(line.lstrip("\t"))
    return body


class TestSyncAllOrdersTheExpiringHalfSafely:
    """The GHA pull must run even when the Metaculus backfill fails."""

    def test_backfill_failure_is_non_fatal(self) -> None:
        # The `-` prefix is the fix: make reports the failure as "(ignored)" and keeps
        # going. Without it, `set -e` in run_sync.sh aborts before any artifact
        # downloads. Verified end-to-end: a `-false` recipe line exits 0 and runs the
        # rest of the recipe.
        backfill_lines = [line for line in _sync_all_recipe() if "backfill_research_from_comments.py" in line]
        assert len(backfill_lines) == 1, f"expected exactly one backfill line, got {backfill_lines}"
        assert backfill_lines[0].startswith("-"), (
            "sync_all's backfill line must be `-`-prefixed (non-fatal). Metaculus comments "
            "never expire; GHA artifacts are deleted at 90 days, so a backfill failure must "
            "not abort the GHA pull. See tests/test_research_sync_job.py docstring."
        )

    def test_backfill_still_precedes_the_gha_driver(self) -> None:
        # Ordering is load-bearing for a different reason than the `-` is: sync_all.py's
        # research build loads comments_backfill.jsonl, and there is exactly ONE build,
        # so the backfill must write that file first or its records miss this run.
        recipe = _sync_all_recipe()
        backfill_at = next(i for i, line in enumerate(recipe) if "backfill_research_from_comments.py" in line)
        driver_at = next(i for i, line in enumerate(recipe) if "scripts/sync_all.py" in line)
        assert backfill_at < driver_at, (
            "the comment backfill must precede scripts/sync_all.py so its "
            "comments_backfill.jsonl is on disk for the single research build"
        )


class TestRunSyncWaitsForTheNetwork:
    """launchd fires at the next wake, when the network is typically still down."""

    def test_network_preflight_runs_before_make(self) -> None:
        script = _RUN_SYNC.read_text()
        probe_at = script.find("if ! wait_for_network")
        # Anchor on the INVOCATION, not the bare string: `make sync_all` also appears in
        # the file's header comment, which would make the ordering assertion vacuous.
        make_at = script.find("if ! make sync_all")
        assert probe_at != -1, "run_sync.sh must call a network-readiness preflight"
        assert make_at != -1, "run_sync.sh must still invoke `make sync_all`"
        assert probe_at < make_at, "the network preflight must run BEFORE `make sync_all`"

    def test_preflight_is_bounded_and_actually_probes(self) -> None:
        script = _RUN_SYNC.read_text()
        # A bounded loop, not an unbounded wait: a job that hangs forever is its own
        # silent failure. And the probe must be a real request, so a permanently-offline
        # wake reports rather than proceeding into a doomed make.
        assert "curl -sf -m 5 https://api.github.com/" in script
        tries = re.search(r"NETWORK_WAIT_TRIES=(\d+)", script)
        sleep_s = re.search(r"NETWORK_WAIT_SLEEP_S=(\d+)", script)
        assert tries is not None and sleep_s is not None, "the wait must be bounded by explicit constants"
        assert 0 < int(tries.group(1)) * int(sleep_s.group(1)) <= 900, "bounded wait should cap out within ~15 min"


class TestSyncFailureIsVisible:
    """Six weeks of staleness happened because a failure produced no signal."""

    def test_failure_writes_a_sentinel_and_notifies(self) -> None:
        script = _RUN_SYNC.read_text()
        assert "LAST_SYNC_FAILED" in script, "a failed run must leave a greppable sentinel file"
        assert "osascript" in script and "display notification" in script, (
            "a failed run must fire a macOS notification — the dated logfile alone is what nobody read for six weeks"
        )

    def test_sentinel_is_cleared_only_on_a_green_run(self) -> None:
        script = _RUN_SYNC.read_text()
        # A sentinel that outlives its failure is as misleading as none at all.
        assert 'rm -f "${FAILURE_SENTINEL}"' in script
        clear_at = script.find('rm -f "${FAILURE_SENTINEL}"')
        make_at = script.find("if ! make sync_all")
        assert make_at < clear_at, "the sentinel must be cleared AFTER a successful make, not before"

    def test_failure_exits_non_zero(self) -> None:
        script = _RUN_SYNC.read_text()
        assert script.count("exit 1") >= 2, "both the preflight and the make failure paths must exit non-zero"


class TestPlistSurvivesOneBadWake:
    def test_two_wakes_per_week(self) -> None:
        with _PLIST.open("rb") as handle:
            plist = plistlib.load(handle)
        intervals = plist["StartCalendarInterval"]
        assert isinstance(intervals, list), (
            "StartCalendarInterval must be a LIST of wake times — a single dict means one bad "
            "wake costs a whole week against a 90-day retention window"
        )
        assert len(intervals) >= 2
        assert {interval["Weekday"] for interval in intervals} == {0, 3}, "expect Sun + Wed wakes"

    def test_throttle_interval_prevents_a_relaunch_loop(self) -> None:
        with _PLIST.open("rb") as handle:
            plist = plistlib.load(handle)
        # run_sync.sh now exits non-zero on failure, and launchd relaunches a failed job
        # promptly by default. Throttle so a network outage gets a real retry, not a loop.
        assert plist["ThrottleInterval"] >= 600

    def test_plist_points_at_the_wrapper_this_suite_checks(self) -> None:
        with _PLIST.open("rb") as handle:
            plist = plistlib.load(handle)
        # Otherwise every assertion above could pass while launchd runs something else.
        assert plist["ProgramArguments"] == [str(_RUN_SYNC)]
