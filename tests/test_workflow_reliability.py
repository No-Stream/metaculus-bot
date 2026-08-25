"""Guards that a hung workflow step cannot hold a concurrency group for hours.

On 2026-08-19 the Azure Ubuntu mirror failed over and ``apt-get update`` hung inside
``playwright install --with-deps`` on four tournament runs. The step carried
``continue-on-error: true``, which covers a step that FAILS and not one that HANGS, and
no step-level ``timeout-minutes``, so each hang ran to the job-level backstop of 300
minutes while holding ``concurrency.group: ${{ github.workflow }}``. GitHub allows at
most one PENDING run per group and cancels the existing pending one when a new run
arrives, so 54 of the day's 68 fires were evicted, the group was held 18 of 24 hours,
and q45374 and q45375 opened and closed without a forecast.

The fix has two halves and this module pins both, because either alone leaves the hole:

1. Every step that shells out or touches the network declares its own
   ``timeout-minutes``. A step cap kills the process, lets ``continue-on-error`` do what
   it was already there for, and — where the step is not allowed to fail — reports a
   ``failure`` rather than the ``cancelled`` a job-level timeout reports, which is
   indistinguishable from a concurrency eviction.
2. Every job declares ``timeout-minutes`` at all, so nothing inherits GitHub's 360.

The numbers are sized in scratch/residual_2026-08-24/workflow_reliability_audit.md off
measured step durations (scratch/residual_2026-08-24/track_c_step_timings.json), with
one hard floor asserted here: the ``Run bot`` cap must clear the bot's OWN per-question
contract in constants.py, or the workflow silently truncates a run in the second it was
about to publish — the forfeit these caps exist to prevent. ``test_bot_basic.yaml``
shipped exactly that inversion, a 60-minute job cap over a 3600-second contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from metaculus_bot.constants import (
    METACULUS_CLOSE_WINDOW_SECONDS,
    PER_QUESTION_WALL_CLOCK_DEADLINE,
    WALL_CLOCK_STACKING_MIN_BUDGET,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"

# Repo-relative posix paths only: an assertion on a path derived from the developer's
# checkout passes locally by construction and fails on the first CI run.
_ALL_WORKFLOWS = sorted(p.relative_to(_REPO_ROOT).as_posix() for p in _WORKFLOW_DIR.glob("*.y*ml"))
_BOT_WORKFLOWS = sorted(p.relative_to(_REPO_ROOT).as_posix() for p in _WORKFLOW_DIR.glob("*bot*.y*ml"))

# Steps that provably cannot hang, so a cap would be noise: no network, no subprocess.
_UNCAPPED_BY_DESIGN = {"Warn if Playwright install failed"}

# Worst healthy duration measured over 200 successful runs per step, in seconds
# (track_c_step_timings.json). Caps must clear these by the multiple below, or a slow
# runner day starts killing healthy steps.
_MEASURED_WORST_SECONDS = {
    "Check out repository": 4,
    "Install dependencies": 22,
    "Install Playwright Chromium": 55,
    "Upload research outputs": 2,
}
_MIN_HEADROOM_OVER_MEASURED = 5

# A hang must cost far less than the 180-minute window a tournament question is open
# for; 300 (the pre-fix value) is longer than the whole window.
_MAX_TOLERABLE_JOB_CAP_MINUTES = 90


def _workflow(rel_path: str) -> dict[str, Any]:
    return yaml.safe_load((_REPO_ROOT / rel_path).read_text())


def _steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for job in workflow["jobs"].values() for step in job.get("steps", [])]


def _named_step(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [step for step in _steps(workflow) if step.get("name") == name]
    assert len(matches) == 1, f"expected exactly one {name!r} step, got {len(matches)}"
    return matches[0]


def _step_label(step: dict[str, Any]) -> str:
    return step.get("name") or step.get("uses", "<unnamed>")


class TestEveryJobIsCapped:
    """No job may inherit GitHub's 360-minute default, in any workflow."""

    def test_the_workflow_set_is_what_we_think_it_is(self) -> None:
        # Pinned so a NEW workflow has to satisfy the invariants below rather than
        # silently escape them. .y*ml because both spellings are live (claude.yml).
        assert _ALL_WORKFLOWS == [
            ".github/workflows/ci.yaml",
            ".github/workflows/claude.yml",
            ".github/workflows/run_bot_on_metaculus_cup.yaml",
            ".github/workflows/run_bot_on_minibench.yaml",
            ".github/workflows/run_bot_on_tournament.yaml",
            ".github/workflows/test_bot.yaml",
            ".github/workflows/test_bot_basic.yaml",
        ]

    @pytest.mark.parametrize("rel_path", _ALL_WORKFLOWS)
    def test_workflow_parses(self, rel_path: str) -> None:
        # A yaml GitHub cannot parse is a workflow that silently never runs, and for the
        # three cron bot workflows that is indistinguishable from cron starvation.
        assert _workflow(rel_path)["jobs"], f"{rel_path} declares no jobs"

    @pytest.mark.parametrize("rel_path", _ALL_WORKFLOWS)
    def test_every_job_declares_a_timeout(self, rel_path: str) -> None:
        for job_name, job in _workflow(rel_path)["jobs"].items():
            assert isinstance(job.get("timeout-minutes"), int), (
                f"{rel_path}:{job_name} has no timeout-minutes, so it inherits GitHub's 360-minute "
                "default — six hours of a held concurrency group per hang"
            )


class TestBotWorkflowStepsAreCapped:
    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_every_step_that_can_hang_declares_a_timeout(self, rel_path: str) -> None:
        for step in _steps(_workflow(rel_path)):
            label = _step_label(step)
            if label in _UNCAPPED_BY_DESIGN:
                continue
            assert isinstance(step.get("timeout-minutes"), int), (
                f"{rel_path}: step {label!r} shells out or hits the network with no timeout-minutes. "
                "The job cap is not a substitute: it fires late and reports `cancelled`, which reads "
                "identically to a concurrency eviction in the run list"
            )

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_step_caps_clear_the_measured_worst_case(self, rel_path: str) -> None:
        for name, worst_seconds in _MEASURED_WORST_SECONDS.items():
            cap_seconds = _named_step(_workflow(rel_path), name)["timeout-minutes"] * 60
            assert cap_seconds >= worst_seconds * _MIN_HEADROOM_OVER_MEASURED, (
                f"{rel_path}: {name!r} capped at {cap_seconds}s but the worst of 200 healthy runs is "
                f"{worst_seconds}s; keep {_MIN_HEADROOM_OVER_MEASURED}x headroom so a slow runner day "
                "cannot kill a healthy step"
            )

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_playwright_step_is_both_capped_and_allowed_to_fail(self, rel_path: str) -> None:
        # Either half alone leaves the 2026-08-19 hole: continue-on-error cannot see a
        # hang, and a cap without continue-on-error would turn a benign apt blip into a
        # lost question instead of a degraded rendered-fetch rung.
        step = _named_step(_workflow(rel_path), "Install Playwright Chromium")
        assert step.get("continue-on-error") is True, (
            f"{rel_path}: a Chromium install failure must not fail the run — gap-fill v2 degrades to plain fetch"
        )
        assert isinstance(step.get("timeout-minutes"), int), (
            f"{rel_path}: continue-on-error does not cover a HANG; this step needs its own cap"
        )


class TestRunBotCapRespectsTheBotsOwnContract:
    """The workflow must not cut the bot off before its own deadline machinery fires."""

    def test_the_contract_is_still_the_one_hour_cycle(self) -> None:
        # If this ever changes, the caps below have to move with it — that is the point of
        # deriving them from the constants instead of hardcoding 60.
        assert PER_QUESTION_WALL_CLOCK_DEADLINE + WALL_CLOCK_STACKING_MIN_BUDGET == METACULUS_CLOSE_WINDOW_SECONDS

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_run_bot_cap_exceeds_the_per_question_contract(self, rel_path: str) -> None:
        cap_seconds = _named_step(_workflow(rel_path), "Run bot")["timeout-minutes"] * 60
        assert cap_seconds > METACULUS_CLOSE_WINDOW_SECONDS, (
            f"{rel_path}: 'Run bot' capped at {cap_seconds}s, but a question may legitimately use "
            f"PER_QUESTION_WALL_CLOCK_DEADLINE ({PER_QUESTION_WALL_CLOCK_DEADLINE}s) and then publish "
            f"inside WALL_CLOCK_STACKING_MIN_BUDGET ({WALL_CLOCK_STACKING_MIN_BUDGET}s). Questions run "
            "concurrently, so this bound does not scale with the batch size — a tighter cap kills a "
            "slow-but-recovering run in the second it was about to publish"
        )

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_job_cap_sits_above_the_run_bot_cap(self, rel_path: str) -> None:
        workflow = _workflow(rel_path)
        run_bot_cap = _named_step(workflow, "Run bot")["timeout-minutes"]
        for job_name, job in workflow["jobs"].items():
            assert job["timeout-minutes"] > run_bot_cap, (
                f"{rel_path}:{job_name} job cap ({job['timeout-minutes']}m) does not clear the 'Run bot' "
                f"step cap ({run_bot_cap}m), so the job cap fires first and the run ends as `cancelled` "
                "with no artifact upload instead of a legible step failure"
            )

    @pytest.mark.parametrize("rel_path", _BOT_WORKFLOWS)
    def test_job_cap_is_far_shorter_than_a_question_window(self, rel_path: str) -> None:
        for job_name, job in _workflow(rel_path)["jobs"].items():
            assert job["timeout-minutes"] <= _MAX_TOLERABLE_JOB_CAP_MINUTES, (
                f"{rel_path}:{job_name} job cap is {job['timeout-minutes']}m; a hang holds the "
                "concurrency group for that long, and a tournament question is only open for 180m"
            )

    def test_bot_job_caps_do_not_drift_apart(self) -> None:
        # The five workflows are near-identical by design, and drift is how the unsafe cap
        # hid: test_bot_basic sat at 60 while the other four sat at 300, so the one file
        # whose cap was BELOW the bot's own contract looked like the conservative one.
        caps = {
            rel_path: sorted({job["timeout-minutes"] for job in _workflow(rel_path)["jobs"].values()})
            for rel_path in _BOT_WORKFLOWS
        }
        distinct = {tuple(value) for value in caps.values()}
        assert len(distinct) == 1, f"bot workflow job caps disagree: {caps}"
