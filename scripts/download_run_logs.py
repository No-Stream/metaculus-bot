"""Download GHA run-log artifacts and harvest telemetry markers into a durable archive.

The bot's four workflows tee stdout+stderr to ``run_logs/`` and upload it:

* the three PROD workflows (tournament / metaculus_cup / minibench) bundle ``run_logs/``
  INSIDE the ``research-<run_id>`` artifact (alongside ``research_outputs/``);
* ``test_bot`` uploads ``run_logs/`` as a SEPARATE ``logs-<run_id>`` artifact.

So harvesting run logs means pulling BOTH artifact families and reading the ``*.log``
files under ``run_logs/``. This mirrors ``scripts/download_research.py`` (and reuses its
paginated, no-cap artifact enumeration): enumerate every live artifact, filter to the
two prefixes, download each, parse the logs via :mod:`scripts.telemetry.markers`, and
merge into ``backtests/telemetry_archive/`` (replace-by-run, idempotent).

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
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Reuse the research downloader's paginated artifact enumeration + gh preflight; both
# are workflow-agnostic (``list_research_artifacts`` lists EVERY artifact in the repo).
from scripts.download_research import _parse_created_at, list_research_artifacts, verify_gh_cli
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


def build_workflow_map(repo: str) -> dict[int, str]:
    """Best-effort ``{run_id: workflow_slug}`` map via the workflow-runs endpoint.

    The artifacts endpoint doesn't carry the workflow name, so we enumerate runs once
    to attribute each artifact to its exact workflow (tournament vs cup vs minibench vs
    test_bot). On any failure we return ``{}`` and callers fall back to prefix inference —
    the map is a nicety for manifest bucketing, never load-bearing for the markers.
    """
    cmd = [
        "gh",
        "api",
        "--paginate",
        f"/repos/{repo}/actions/runs?per_page=100",
        "--jq",
        ".workflow_runs[] | {id, path}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
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
    run_log = [a for a in artifacts if str(a.get("name", "")).startswith(RUN_LOG_ARTIFACT_PREFIXES)]
    live = [a for a in run_log if not a.get("expired")]
    expired = [a for a in run_log if a.get("expired")]
    return live, expired


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


def _download_artifact_to(run_id: int, repo: str, artifact_name: str, dest_dir: Path) -> Path | None:
    """Download one artifact into ``dest_dir/<run_id>``; return the dir or None on failure."""
    run_dir = dest_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["gh", "run", "download", str(run_id), "--repo", repo, "--name", artifact_name, "--dir", str(run_dir)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        logger.warning(f"Failed to download {artifact_name} (run {run_id}): {result.stderr.strip()}")
        return None
    return run_dir


def download_and_harvest(
    repo: str, since_days: int, archive_dir: Path
) -> tuple[dict[str, int], list[HarvestedRun], int]:
    """Enumerate + download every live run-log artifact, harvest markers, merge into the archive.

    Returns ``(per_marker_totals, harvested_runs, expired_count)``.
    """
    verify_gh_cli()

    all_artifacts = list_research_artifacts(repo)
    live, expired = filter_run_log_artifacts(all_artifacts)
    logger.info(
        f"Artifacts endpoint returned {len(all_artifacts)} total, "
        f"{len(live)} live + {len(expired)} expired run-log artifacts"
    )

    if expired:
        logger.warning(f"{len(expired)} run-log artifact(s) are EXPIRED and UNRECOVERABLE (past 90-day retention):")
        for art in sorted(expired, key=lambda a: a.get("created_at", "")):
            logger.warning(f"  LOST: {art.get('name')} (created_at={art.get('created_at')})")
    else:
        logger.info("No expired run-log artifacts — nothing lost to the 90-day window.")

    if since_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        before = len(live)
        live = [a for a in live if (_parse_created_at(a.get("created_at", "")) or cutoff) >= cutoff]
        logger.info(f"--since-days={since_days} post-filter: {len(live)}/{before} live artifacts within window")

    workflow_map = build_workflow_map(repo)
    logger.info(f"Resolved {len(workflow_map)} run->workflow mappings")

    runs: list[HarvestedRun] = []
    with tempfile.TemporaryDirectory(prefix="run_logs_dl_") as tmpdir:
        tmp_path = Path(tmpdir)
        for idx, art in enumerate(sorted(live, key=lambda a: a.get("created_at", "")), 1):
            run_id = art.get("run_id")
            name = art.get("name", "")
            if run_id is None:
                logger.warning(f"Live artifact {name} has no workflow_run id, skipping")
                continue
            run_dir = _download_artifact_to(run_id, repo, name, tmp_path)
            if run_dir is None:
                continue
            harvested = harvest_run_logs_from_dir(
                run_dir,
                run_id=str(run_id),
                workflow=infer_workflow(name, run_id, workflow_map),
                artifact=name,
                run_date=art.get("created_at", ""),
            )
            if harvested is not None:
                runs.append(harvested)
            if idx % 25 == 0:
                print(f"  processed {idx}/{len(live)} artifacts, {len(runs)} with logs", flush=True)

    totals = merge_and_write(archive_dir, runs)
    return totals, runs, len(expired)


def _report(totals: dict[str, int], runs: list[HarvestedRun], expired_count: int) -> None:
    dates = sorted(r.run_date for r in runs if r.run_date)
    coverage = f"{dates[0]} .. {dates[-1]}" if dates else "(none)"
    logger.info("=" * 60)
    logger.info(f"Telemetry harvest complete: {len(runs)} run(s) with logs, {expired_count} expired/lost")
    logger.info(f"Run-date coverage: {coverage}")
    logger.info("Per-marker archive totals:")
    for marker, count in sorted(totals.items()):
        logger.info(f"  {marker:32s} {count}")


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
