"""Download research artifacts from GHA and merge with backfill into a local archive.

Enumerates recent GitHub Actions runs, downloads research JSONL artifacts,
combines them with existing backfill data, and writes a queryable local archive:

  backtests/research_archive/
    latest/<qid>.json      # most recent research per question
    by_qid/<qid>.jsonl     # all versions per question (newest-first)
    manifest.json          # index: {qid: {latest_timestamp, versions_count, providers}}
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


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


def list_runs_with_artifacts(workflow: str, limit: int, repo: str) -> list[dict]:
    """List recent workflow runs that might have artifacts."""
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
        logger.error(f"gh run list failed: {result.stderr.strip()}")
        sys.exit(1)
    return json.loads(result.stdout)


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
    parser.add_argument("--workflow", default="run_bot_on_tournament.yaml", help="Workflow filename")
    parser.add_argument("--limit", type=int, default=200, help="Max runs to check for artifacts")
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

    output_dir = Path(args.output_dir)
    backfill_dir = Path(args.backfill_dir)

    all_records: list[dict] = []

    # --- Phase 1: Download artifacts from GHA ---
    if not args.skip_download:
        verify_gh_cli()

        runs = list_runs_with_artifacts(args.workflow, args.limit, args.repo)
        logger.info(f"Found {len(runs)} runs to check for artifacts")

        with tempfile.TemporaryDirectory(prefix="research_dl_") as tmpdir:
            tmp_path = Path(tmpdir)
            artifacts_found = 0

            for idx, run in enumerate(runs, 1):
                run_id = run["databaseId"]
                downloaded_files = download_artifact(run_id, args.repo, tmp_path)
                if downloaded_files:
                    artifacts_found += 1
                    for jsonl_file in downloaded_files:
                        all_records.extend(load_jsonl_records(jsonl_file))

                # Progress on every 20th run or when artifacts are found
                if idx % 20 == 0 or downloaded_files:
                    print(
                        f"  Downloading artifacts... ({artifacts_found}/{idx} runs had artifacts)",
                        flush=True,
                    )

            logger.info(f"Downloaded {artifacts_found}/{len(runs)} runs with artifacts ({len(all_records)} records)")

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
