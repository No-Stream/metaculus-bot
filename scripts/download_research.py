"""Download research artifacts from GHA and merge with backfill into a local archive.

Enumerates recent GitHub Actions runs across ALL bot run-workflows, downloads the
research JSONL artifacts each run uploads (artifact name `research-<run_id>`),
combines them with existing backfill data, and writes a queryable local archive:

  backtests/research_archive/
    latest/<qid>.json      # most recent research per question
    by_qid/<qid>.jsonl     # all versions per question (newest-first)
    manifest.json          # index: {qid: {latest_timestamp, versions_count, providers}}

WHY THIS MUST RUN REGULARLY
---------------------------
GHA uploads each run's `research_outputs/` artifact with `retention-days: 90`
(see .github/workflows/run_bot_on_{tournament,metaculus_cup,minibench}.yaml).
After 90 days the artifact is deleted from GitHub FOREVER, so this local archive
is the only durable copy. The puller is manual (`make sync_research`); schedule it
(see scripts/research_sync/) so artifacts are captured well inside the 90-day window.

COVERAGE STRATEGY
-----------------
The tournament workflow runs on a 3x/hour cron (~72 runs/day, ~6.5k runs over the
90-day window). A fixed `--limit` would silently miss the older majority of the
window once the cron has produced more runs than the limit, so by default we PAGE
through `gh run list` (sorted newest-first) and keep checking until runs fall
outside `--since-days` (default 90, matching retention). `--limit` is the per-page
size; `--max-runs` is a hard safety cap. Pass `--since-days 0` to disable the date
window and fall back to a single `--limit`-sized page (legacy behavior).

All three run-workflows are swept by default (tournament + metaculus_cup +
minibench); pass `--workflow` (repeatable) to restrict. Run ids are deduped, so a
run that somehow appears under multiple workflows is only downloaded once.
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# All bot run-workflows that upload `research-<run_id>` artifacts. Keep in sync with
# .github/workflows/run_bot_on_*.yaml.
DEFAULT_WORKFLOWS: list[str] = [
    "run_bot_on_tournament.yaml",
    "run_bot_on_metaculus_cup.yaml",
    "run_bot_on_minibench.yaml",
]

# GitHub's `gh run list --limit` page-size ceiling.
GH_RUN_LIST_PAGE_MAX = 1000


def verify_gh_cli() -> None:
    """Ensure gh CLI is installed and authenticated."""
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        logger.error("gh CLI not found. Install from https://cli.github.com/")
        sys.exit(1)
    except subprocess.CalledProcessError:
        logger.error("gh CLI returned an error. Check authentication with 'gh auth status'.")
        sys.exit(1)

    result = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"gh CLI not authenticated: {result.stderr.strip()}")
        sys.exit(1)


def _parse_created_at(value: str) -> datetime | None:
    """Parse a GH `createdAt` ISO-8601 timestamp (e.g. '2026-05-22T14:00:00Z')."""
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def list_runs_with_artifacts(workflow: str, limit: int, repo: str) -> list[dict]:
    """List recent workflow runs (newest-first) that might have artifacts."""
    cmd = [
        "gh",
        "run",
        "list",
        "--repo",
        repo,
        "--workflow",
        workflow,
        "--limit",
        str(limit),
        "--json",
        "databaseId,createdAt",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"gh run list failed for {workflow}: {result.stderr.strip()}")
        sys.exit(1)
    return json.loads(result.stdout)


def list_runs_in_window(workflow: str, repo: str, since_days: int, page_size: int, max_runs: int) -> list[dict]:
    """List all runs for a workflow created within the last `since_days` days.

    `gh run list` returns the most-recent runs first but has no createdAt filter
    and caps a single call at GH_RUN_LIST_PAGE_MAX. When `since_days <= 0` we make a
    single `page_size`-sized call (legacy behavior). Otherwise we page in chunks of
    `page_size`, stopping once a chunk's oldest run falls outside the window or once
    `max_runs` is reached. Runs without artifacts are common; the goal here is to
    *check* every run inside the retention window, not to assume they all have one.
    """
    if since_days <= 0:
        return list_runs_with_artifacts(workflow, page_size, repo)

    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    page_size = min(page_size, GH_RUN_LIST_PAGE_MAX)

    in_window: list[dict] = []
    seen_ids: set[int] = set()
    page = 0
    while len(in_window) < max_runs:
        page += 1
        batch = list_runs_with_artifacts(workflow, page_size, repo)
        if not batch:
            break

        new_in_window = 0
        oldest_outside_window = False
        for run in batch:
            run_id = run.get("databaseId")
            if run_id is None or run_id in seen_ids:
                continue
            created = _parse_created_at(run.get("createdAt", ""))
            if created is not None and created < cutoff:
                oldest_outside_window = True
                continue
            seen_ids.add(run_id)
            in_window.append(run)
            new_in_window += 1

        # gh run list has no cursor, so a fixed page_size can only return the newest
        # page_size runs. If the page filled and every run was still inside the window,
        # we've hit gh's per-call ceiling and cannot page deeper — surface it loudly.
        if oldest_outside_window or new_in_window == 0 or len(batch) < page_size:
            if oldest_outside_window:
                logger.info(f"  {workflow}: reached the {since_days}-day cutoff after checking {len(batch)} runs.")
            elif len(batch) < page_size:
                logger.info(f"  {workflow}: exhausted run history ({len(batch)} runs total).")
            else:
                logger.warning(
                    f"  {workflow}: {len(in_window)} runs all fall inside the {since_days}-day window but "
                    f"`gh run list` is capped at {page_size}/call with no pagination cursor — older runs in the "
                    f"window may be UNCHECKED. Run this puller more frequently, or lower --since-days."
                )
            break

    return in_window[:max_runs]


def download_artifact(run_id: int, repo: str, dest_dir: Path) -> list[Path]:
    """Try to download research artifact for a run. Returns list of downloaded JSONL files.

    Many runs won't have artifacts (zero-prediction runs, old runs with expired artifacts).
    Returns empty list on failure — this is expected and not an error.
    """
    artifact_name = f"research-{run_id}"
    run_dir = dest_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gh",
        "run",
        "download",
        str(run_id),
        "--repo",
        repo,
        "--name",
        artifact_name,
        "--dir",
        str(run_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return []

    return list(run_dir.glob("**/*.jsonl"))


def load_jsonl_records(path: Path) -> list[dict]:
    """Load all records from a JSONL file, skipping malformed lines."""
    records = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Malformed JSON at {path}:{line_num}, skipping")
    return records


def load_backfill(backfill_dir: Path) -> list[dict]:
    """Load all JSONL records from the backfill directory."""
    if not backfill_dir.exists():
        logger.info(f"Backfill directory does not exist: {backfill_dir}")
        return []

    records = []
    for jsonl_file in sorted(backfill_dir.glob("*.jsonl")):
        file_records = load_jsonl_records(jsonl_file)
        records.extend(file_records)
        logger.debug(f"Loaded {len(file_records)} records from {jsonl_file.name}")

    logger.info(f"Loaded {len(records)} total backfill records from {backfill_dir}")
    return records


def deduplicate_records(records: list[dict]) -> list[dict]:
    """Deduplicate by (qid, run_id), keeping the record with the latest timestamp."""
    by_key: dict[tuple[int, str], dict] = {}
    for record in records:
        qid = record.get("qid")
        run_id = record.get("run_id", "")
        if qid is None:
            continue
        key = (qid, str(run_id))
        existing = by_key.get(key)
        if existing is None or record.get("timestamp", "") > existing.get("timestamp", ""):
            by_key[key] = record
    return list(by_key.values())


def download_all_workflows(
    workflows: list[str],
    repo: str,
    since_days: int,
    page_size: int,
    max_runs: int,
) -> list[dict]:
    """Sweep every workflow for artifact-bearing runs inside the retention window.

    Run ids are deduped across workflows so an artifact is downloaded at most once.
    Logs per-workflow coverage (runs checked / artifacts found / records added) so a
    stale or short pull is visible in the output.
    """
    verify_gh_cli()

    all_records: list[dict] = []
    downloaded_run_ids: set[int] = set()

    with tempfile.TemporaryDirectory(prefix="research_dl_") as tmpdir:
        tmp_path = Path(tmpdir)

        for workflow in workflows:
            runs = list_runs_in_window(workflow, repo, since_days, page_size, max_runs)
            logger.info(f"[{workflow}] checking {len(runs)} runs for artifacts")

            artifacts_found = 0
            records_added = 0
            for idx, run in enumerate(runs, 1):
                run_id = run["databaseId"]
                if run_id in downloaded_run_ids:
                    continue
                downloaded_run_ids.add(run_id)

                downloaded_files = download_artifact(run_id, repo, tmp_path)
                if downloaded_files:
                    artifacts_found += 1
                    for jsonl_file in downloaded_files:
                        new_records = load_jsonl_records(jsonl_file)
                        records_added += len(new_records)
                        all_records.extend(new_records)

                if idx % 50 == 0 or downloaded_files:
                    print(
                        f"  [{workflow}] {artifacts_found} artifacts in {idx}/{len(runs)} runs",
                        flush=True,
                    )

            logger.info(
                f"[{workflow}] checked {len(runs)} runs, {artifacts_found} had artifacts, {records_added} records added"
            )

    logger.info(f"Download phase complete: {len(all_records)} records across {len(workflows)} workflow(s)")
    return all_records


def build_archive(records: list[dict], output_dir: Path) -> None:
    """Write the merged archive: latest/, by_qid/, manifest.json."""
    latest_dir = output_dir / "latest"
    by_qid_dir = output_dir / "by_qid"
    latest_dir.mkdir(parents=True, exist_ok=True)
    by_qid_dir.mkdir(parents=True, exist_ok=True)

    # Group by qid
    by_qid: dict[int, list[dict]] = {}
    for record in records:
        qid = record.get("qid")
        if qid is None:
            continue
        by_qid.setdefault(qid, []).append(record)

    # Sort each group newest-first
    for qid in by_qid:
        by_qid[qid].sort(key=lambda r: r.get("timestamp", ""), reverse=True)

    manifest: dict[str, dict] = {}

    for qid, qid_records in sorted(by_qid.items()):
        # Write latest/<qid>.json
        latest_record = qid_records[0]
        latest_path = latest_dir / f"{qid}.json"
        with open(latest_path, "w") as f:
            json.dump(latest_record, f, indent=2)

        # Write by_qid/<qid>.jsonl (newest-first)
        by_qid_path = by_qid_dir / f"{qid}.jsonl"
        with open(by_qid_path, "w") as f:
            for record in qid_records:
                f.write(json.dumps(record) + "\n")

        # Collect all providers seen across all versions
        all_providers: set[str] = set()
        for record in qid_records:
            all_providers.update(record.get("providers_used", []))

        manifest[str(qid)] = {
            "latest_timestamp": latest_record.get("timestamp", ""),
            "versions_count": len(qid_records),
            "providers": sorted(all_providers),
        }

    # Write manifest.json
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    logger.info(f"Archive built: {len(by_qid)} questions, {len(records)} total records")


def main():
    parser = argparse.ArgumentParser(description="Download research artifacts from GHA and merge with backfill.")
    parser.add_argument(
        "--workflow",
        action="append",
        dest="workflows",
        default=None,
        help=(
            f"Workflow filename to sweep. Repeatable. Defaults to all bot run-workflows: {', '.join(DEFAULT_WORKFLOWS)}"
        ),
    )
    parser.add_argument(
        "--since-days",
        type=int,
        default=90,
        help=(
            "Check all runs created within this many days (matches GHA's 90-day artifact "
            "retention). Pages until runs fall outside the window. Set to 0 to disable the "
            "window and use a single --limit-sized page (legacy behavior)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=GH_RUN_LIST_PAGE_MAX,
        help=f"Per-page size for `gh run list` (capped at {GH_RUN_LIST_PAGE_MAX}).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=5000,
        help="Hard safety cap on runs checked per workflow.",
    )
    parser.add_argument("--repo", default="No-Stream/metaculus-bot", help="GitHub repo")
    parser.add_argument(
        "--backfill-dir",
        default="backtests/research_archive/backfill",
        help="Where backfill JSONL lives",
    )
    parser.add_argument(
        "--output-dir",
        default="backtests/research_archive",
        help="Where to write the merged archive",
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip artifact download, only rebuild from backfill"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    workflows = args.workflows if args.workflows else DEFAULT_WORKFLOWS

    output_dir = Path(args.output_dir)
    backfill_dir = Path(args.backfill_dir)

    all_records: list[dict] = []

    # --- Phase 1: Download artifacts from GHA (all workflows) ---
    if not args.skip_download:
        all_records.extend(
            download_all_workflows(
                workflows=workflows,
                repo=args.repo,
                since_days=args.since_days,
                page_size=args.limit,
                max_runs=args.max_runs,
            )
        )

    # --- Phase 2: Load backfill ---
    backfill_records = load_backfill(backfill_dir)
    all_records.extend(backfill_records)

    # --- Phase 3: Deduplicate and build archive ---
    if not all_records:
        logger.warning("No records found (no artifacts downloaded and no backfill data). Nothing to build.")
        return

    deduplicated = deduplicate_records(all_records)
    logger.info(f"After deduplication: {len(deduplicated)} unique records (from {len(all_records)} total)")

    build_archive(deduplicated, output_dir)
    logger.info(f"Archive written to {output_dir}")


if __name__ == "__main__":
    main()
