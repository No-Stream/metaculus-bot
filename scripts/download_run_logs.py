"""Download GHA run-log artifacts and harvest telemetry markers into a durable archive.

The bot's four workflows tee stdout+stderr to ``run_logs/`` and upload it:

* the three PROD workflows (tournament / metaculus_cup / minibench) bundle ``run_logs/``
  INSIDE the ``research-<run_id>`` artifact (alongside ``research_outputs/``);
* ``test_bot`` uploads ``run_logs/`` as a SEPARATE ``logs-<run_id>`` artifact.

So harvesting run logs means pulling BOTH artifact families and reading the ``*.log``
files under ``run_logs/``. Enumeration + download run through the shared
``scripts.gha_artifacts`` core (paginated, no-cap artifact enumeration): enumerate every
live artifact, filter to the two prefixes, download each, parse the logs via
:mod:`scripts.telemetry.markers`, and merge into ``backtests/telemetry_archive/``
(replace-by-run, idempotent).

WHY THIS MUST RUN REGULARLY: GHA deletes artifacts after 90 days (``retention-days: 90``
on every upload step). The local telemetry archive is the only durable copy of the
run-log markers — the same silent-loss risk that motivates the research-archive sync.
Read-only + free (GitHub API only); no paid LLM/research calls, no publishing.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.gha_artifacts import GH_API_TIMEOUT_S, download_run_dirs, select_run_artifacts, split_by_family
from scripts.telemetry.archive import HarvestedRun, merge_and_write
from scripts.telemetry.markers import parse_log_text

logger = logging.getLogger(__name__)

# Run logs live in these two artifact families (see module docstring): prod runs bundle
# them in ``research-*``; test_bot uploads a separate ``logs-*``.
RUN_LOG_ARTIFACT_PREFIXES: tuple[str, ...] = ("research-", "logs-")

DEFAULT_REPO = "No-Stream/metaculus-bot"
DEFAULT_ARCHIVE_DIR = "backtests/telemetry_archive"


def workflow_slug_from_path(path: str) -> str:
    """Map a workflow file path to a short slug (``run_bot_on_tournament.yaml`` -> ``tournament``)."""
    stem = Path(path).name.removesuffix(".yaml").removesuffix(".yml")
    return stem.removeprefix("run_bot_on_")


def build_workflow_map(repo: str, since_days: int = 120) -> dict[int, str]:
    """Best-effort ``{run_id: workflow_slug}`` map via the workflow-runs endpoint.

    The artifacts endpoint doesn't carry the workflow name, so we enumerate runs to
    attribute each artifact to its exact workflow (tournament vs cup vs minibench vs
    test_bot). The enumeration is BOUNDED to runs created in the last ``since_days``
    (default 120, comfortably past the 90-day artifact retention) via the API's
    ``created`` filter — an unbounded ``--paginate`` walks the repo's ENTIRE run history
    (thousands of runs) and stalls for minutes on a nicety. On any failure we return
    ``{}`` and callers fall back to prefix inference; the map is manifest-bucketing
    convenience, never load-bearing for the markers.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).strftime("%Y-%m-%d")
    cmd = [
        "gh",
        "api",
        "--paginate",
        # -X GET is required: gh api sends -f fields as a POST body by default (which
        # 404s on this GET-only endpoint); -X GET makes them query params instead.
        "-X",
        "GET",
        f"/repos/{repo}/actions/runs",
        "-f",
        "per_page=100",
        "-f",
        f"created=>={cutoff}",
        "--jq",
        ".workflow_runs[] | {id, path}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=GH_API_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        logger.warning(f"workflow-runs enumeration timed out ({GH_API_TIMEOUT_S}s); using prefix inference only")
        return {}
    if result.returncode != 0:
        logger.warning(f"workflow-runs enumeration failed ({result.stderr.strip()}); using prefix inference only")
        return {}

    workflow_map: dict[int, str] = {}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        run_id = obj.get("id")
        path = obj.get("path")
        if run_id is not None and path:
            workflow_map[int(run_id)] = workflow_slug_from_path(path)
    return workflow_map


def infer_workflow(artifact_name: str, run_id: int, workflow_map: dict[int, str]) -> str:
    """Resolve a run's workflow: exact from the map, else infer from the artifact prefix."""
    mapped = workflow_map.get(run_id)
    if mapped:
        return mapped
    if artifact_name.startswith("logs-"):
        return "test_bot"
    return "unknown"


def filter_run_log_artifacts(artifacts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split artifacts into (live, expired) run-log artifacts (research-* / logs-*)."""
    return split_by_family(artifacts, RUN_LOG_ARTIFACT_PREFIXES)


def harvest_run_logs_from_dir(
    run_dir: Path,
    *,
    run_id: str,
    workflow: str,
    artifact: str,
    run_date: str,
) -> HarvestedRun | None:
    """Parse every ``*.log`` under ``run_dir`` into one :class:`HarvestedRun`.

    Returns ``None`` when the artifact carried no ``*.log`` files (e.g. a ``research-*``
    artifact with ``research_outputs/`` but no ``run_logs/``). Multiple log files are
    concatenated in filename order and parsed once, so ``seq`` stays contiguous across
    the run.
    """
    log_files = sorted(Path(run_dir).glob("**/*.log"))
    if not log_files:
        return None

    names = [p.name for p in log_files]
    joined = "\n".join(p.read_text(errors="replace") for p in log_files)
    records = parse_log_text(
        joined,
        run_id=run_id,
        workflow=workflow,
        artifact=artifact,
        run_date=run_date,
        log_file=names[0] if len(names) == 1 else "; ".join(names),
    )
    return HarvestedRun(
        run_id=run_id,
        workflow=workflow,
        artifact=artifact,
        run_date=run_date,
        log_files=names,
        records=records,
    )


def download_and_harvest(
    repo: str, since_days: int, archive_dir: Path
) -> tuple[dict[str, int], list[HarvestedRun], int]:
    """Enumerate + download every live run-log artifact, harvest markers, merge into the archive.

    Enumeration + download go through the shared core; this function contributes only the
    run-log-specific harvest (``harvest_run_logs_from_dir``) and merge. Returns
    ``(per_marker_totals, harvested_runs, expired_count)``.
    """
    selection = select_run_artifacts(
        repo, family_prefixes=RUN_LOG_ARTIFACT_PREFIXES, since_days=since_days, family_label="run-log"
    )

    workflow_map = build_workflow_map(repo)
    logger.info(f"Resolved {len(workflow_map)} run->workflow mappings")

    runs: list[HarvestedRun] = []
    for run_id, art, run_dir in download_run_dirs(
        selection, repo, tmp_prefix="run_logs_dl_", progress_noun="run-log artifacts"
    ):
        name = art.get("name", "")
        harvested = harvest_run_logs_from_dir(
            run_dir,
            run_id=str(run_id),
            workflow=infer_workflow(name, run_id, workflow_map),
            artifact=name,
            run_date=art.get("created_at", ""),
        )
        if harvested is not None:
            runs.append(harvested)

    totals = merge_and_write(archive_dir, runs)
    return totals, runs, len(selection.expired)


# Markers that live in the PUBLISHED comment, not stdout/stderr (see markers.py
# docstring) — the framework logs only "Posted comment on post N", never the body,
# so these ~never appear in run logs. A zero count here is expected, not a miss.
_COMMENT_ONLY_MARKERS: frozenset[str] = frozenset(
    {"stacker_outcome", "tools_used", "anchor_overshoot_pp", "clause_product_divergence_pp"}
)


def _report(totals: dict[str, int], runs: list[HarvestedRun], expired_count: int) -> None:
    dates = sorted(r.run_date for r in runs if r.run_date)
    coverage = f"{dates[0]} .. {dates[-1]}" if dates else "(none)"
    logger.info("=" * 60)
    logger.info(f"Telemetry harvest complete: {len(runs)} run(s) with logs, {expired_count} expired/lost")
    logger.info(f"Run-date coverage: {coverage}")
    logger.info("Per-marker archive totals:")
    for marker, count in sorted(totals.items()):
        logger.info(f"  {marker:32s} {count}")

    # A marker reads 0 for one of two reasons, NEITHER of which is a parse failure:
    #  (1) it's a comment-only marker (never logged), or
    #  (2) its emitting code isn't on the branch the scheduled prod runs execute from
    #      yet (scheduled workflows run the DEFAULT branch — so a marker added on a
    #      feature branch stays 0 in the logs until that branch merges to main and a
    #      run executes). To disambiguate a real regression from expected-zero, grep a
    #      recent run_logs/*.log for the marker token AND confirm the emitter is on main.
    zero_log_markers = sorted(m for m, c in totals.items() if c == 0 and m not in _COMMENT_ONLY_MARKERS)
    if zero_log_markers:
        logger.info(
            "NOTE: %d log-marker(s) at 0 (%s). Expected if the emitter isn't on the prod branch (main) yet; "
            "verify by grepping a recent log for the token before treating as a regression.",
            len(zero_log_markers),
            ", ".join(zero_log_markers),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GHA run-log artifacts and harvest telemetry markers.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo (owner/name)")
    parser.add_argument(
        "--since-days",
        type=int,
        default=0,
        help="Optional post-filter: only artifacts created within N days (0 = every live artifact).",
    )
    parser.add_argument("--archive-dir", default=DEFAULT_ARCHIVE_DIR, help="Where to write the telemetry archive")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    totals, runs, expired_count = download_and_harvest(
        repo=args.repo, since_days=args.since_days, archive_dir=Path(args.archive_dir)
    )
    _report(totals, runs, expired_count)
    logger.info(f"Archive written to {args.archive_dir}")


if __name__ == "__main__":
    sys.exit(main())
