"""Download GHA run-log artifacts and harvest telemetry markers into a durable archive.

All five bot workflows tee stdout+stderr to ``run_logs/`` and upload it:

* every one of them (the three PROD workflows tournament / metaculus_cup / minibench, plus
  ``test_bot`` and ``test_bot_basic``) bundles ``run_logs/`` INSIDE the
  ``research-<run_id>`` artifact, alongside ``research_outputs/``;
* the two test workflows uploaded a SEPARATE ``logs-<run_id>`` artifact until 2026-08-03,
  and those stay on GHA until their 90-day retention expires.

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

from scripts.gha_artifacts import (
    GH_API_TIMEOUT_S,
    add_store_arguments,
    persisted_run_dirs,
    select_artifacts,
    split_by_family,
)
from scripts.telemetry.archive import HarvestedRun, load_run_manifest, merge_and_write
from scripts.telemetry.markers import parse_log_text

logger = logging.getLogger(__name__)

# Run logs live in these two artifact families (see module docstring). Every bot
# workflow now bundles them in ``research-*``; ``logs-*`` is the pre-2026-08-03 name the
# two test workflows used, kept here because those artifacts stay on GHA until their
# 90-day retention expires.
RUN_LOG_ARTIFACT_PREFIXES: tuple[str, ...] = ("research-", "logs-")

DEFAULT_REPO = "No-Stream/metaculus-bot"
DEFAULT_ARCHIVE_DIR = "backtests/telemetry_archive"

# Bound on the workflow-runs enumeration's ``created`` filter. Generous relative to the
# 90-day artifact retention, but GitHub's 1000-item pagination cap almost always
# dominates it in practice (see build_workflow_map).
_RUNS_ENUMERATION_WINDOW_DAYS = 120


def workflow_slug_from_path(path: str) -> str:
    """Map a workflow file path to a short slug (``run_bot_on_tournament.yaml`` -> ``tournament``)."""
    stem = Path(path).name.removesuffix(".yaml").removesuffix(".yml")
    return stem.removeprefix("run_bot_on_")


def build_workflow_map(repo: str) -> dict[int, str]:
    """Best-effort ``{run_id: workflow_slug}`` map via the workflow-runs endpoint.

    The artifacts endpoint doesn't carry the workflow name, so we enumerate runs to
    attribute each artifact to its exact workflow (tournament vs cup vs minibench vs
    test_bot). The ``created`` filter bounds the walk to the last
    ``_RUNS_ENUMERATION_WINDOW_DAYS`` (an unbounded ``--paginate`` walks the repo's
    ENTIRE run history and stalls for minutes on a nicety), but the effective bound is
    tighter: GitHub caps ``created``-filtered pagination at 1000 items, so this map
    never covers the full window — always resolve through :func:`resolve_workflow_map`.
    On any failure we return ``{}`` and the resolver degrades to the archive-derived
    map alone; the map is manifest-bucketing convenience, never load-bearing for the
    markers.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=_RUNS_ENUMERATION_WINDOW_DAYS)).strftime("%Y-%m-%d")
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
        logger.warning(
            f"workflow-runs enumeration timed out ({GH_API_TIMEOUT_S}s); falling back to the archive-derived map"
        )
        return {}
    if result.returncode != 0:
        logger.warning(
            f"workflow-runs enumeration failed ({result.stderr.strip()}); falling back to the archive-derived map"
        )
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


def workflow_map_from_archive(archive_dir: Path) -> dict[int, str]:
    """Rebuild ``{run_id: workflow_slug}`` from the telemetry archive's own run manifest.

    The under-layer of :func:`resolve_workflow_map` on BOTH paths: offline
    (``--from-store``) it is the whole map, online it fills the runs GitHub's window no
    longer returns. ``unknown`` entries are skipped so they never displace a later
    exact resolution.
    """
    mapping: dict[int, str] = {}
    for record in load_run_manifest(Path(archive_dir)):
        run_id = str(record.get("run_id", ""))
        workflow = str(record.get("workflow", ""))
        if run_id.isdigit() and workflow and workflow != "unknown":
            mapping[int(run_id)] = workflow
    return mapping


def resolve_workflow_map(repo: str, archive_dir: Path, *, from_store: bool = False) -> dict[int, str]:
    """``{run_id: workflow_slug}`` for a harvest: GitHub's fresh window layered OVER the archive.

    ``build_workflow_map``'s enumeration returns at most 1000 runs (GitHub caps
    ``created``-filtered pagination), so any run older than ~15 days is absent from the
    fresh map. ``infer_workflow`` reads those as ``unknown``, and the replace-by-run
    merge then writes that over the correct slug the archive already recorded. Merging
    the archive's own manifest UNDERNEATH the fresh map keeps that attribution: GitHub
    wins wherever both know a run (it is the exact source), the archive fills every run
    the window no longer returns. Offline (``from_store``) there is no network, so the
    archive map is the whole answer.
    """
    archive_map = workflow_map_from_archive(archive_dir)
    if from_store:
        logger.info(f"Offline harvest: recovered {len(archive_map)} run->workflow mappings from {archive_dir}")
        return archive_map
    fresh_map = build_workflow_map(repo)
    merged = {**archive_map, **fresh_map}
    logger.info(
        f"Workflow map: {len(fresh_map)} run->workflow mapping(s) from GitHub "
        f"+ {len(merged) - len(fresh_map)} archive-only (total {len(merged)})"
    )
    return merged


def infer_workflow(artifact_name: str, run_id: int, workflow_map: dict[int, str]) -> str:
    """Resolve a run's workflow: exact from the map, else infer from the artifact prefix.

    The prefix rung only identifies test runs that predate the 2026-08-03 rename to
    ``research-*``; a newer test run whose id is missing from the map (enumeration
    failed, or the run is older than the map's window) reads ``unknown``, since the
    artifact name no longer distinguishes it from a prod run. Bucketing convenience
    only — the markers themselves never depend on it.
    """
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
    repo: str,
    since_days: int,
    archive_dir: Path,
    *,
    store_dir: Path | str | None = None,
    from_store: bool = False,
) -> tuple[dict[str, int], list[HarvestedRun], int]:
    """Persist every live run-log artifact, harvest markers from the store, merge into the archive.

    Enumeration + persistence go through the shared core; this function contributes only
    the run-log-specific harvest (``harvest_run_logs_from_dir``) and merge. With
    ``from_store=True`` nothing is downloaded and the workflow map comes off the archive.
    Returns ``(per_marker_totals, harvested_runs, expired_count)``.
    """
    selection = select_artifacts(
        repo,
        family_prefixes=RUN_LOG_ARTIFACT_PREFIXES,
        since_days=since_days,
        family_label="run-log",
        store_dir=store_dir,
        from_store=from_store,
    )

    workflow_map = resolve_workflow_map(repo, archive_dir, from_store=from_store)

    runs: list[HarvestedRun] = []
    for run_id, art, run_dir in persisted_run_dirs(
        selection, repo, store_dir=store_dir, from_store=from_store, progress_noun="run-log artifacts"
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
    add_store_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    totals, runs, expired_count = download_and_harvest(
        repo=args.repo,
        since_days=args.since_days,
        archive_dir=Path(args.archive_dir),
        store_dir=args.store_dir,
        from_store=args.from_store,
    )
    _report(totals, runs, expired_count)
    logger.info(f"Archive written to {args.archive_dir}")


if __name__ == "__main__":
    sys.exit(main())
