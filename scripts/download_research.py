"""Download research artifacts from GHA and merge with backfill into a local archive.

Enumerates EVERY research artifact in the repo via GitHub's artifacts REST endpoint,
downloads the research JSONL each bot run uploads (artifact name `research-<run_id>`),
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
Enumeration + download run through the shared ``scripts.gha_artifacts`` core, which
lists every artifact via the AUTHORITATIVE, COMPLETE paginated REST endpoint (no
1000-result cap, unlike `gh run list`). Every bot run's artifact is named
`research-<run_id>` regardless of which run-workflow (tournament / metaculus_cup /
minibench) produced it, so filtering to the `research-` prefix captures everything.
Expired artifacts are unrecoverable; the core logs them loudly so the operator knows
exactly what (if anything) was lost.

`--since-days` is an OPTIONAL post-filter on each artifact's `created_at`. The DEFAULT
is no window (pull every live artifact), since the endpoint already returns everything.
"""

import argparse
import json
import logging
from pathlib import Path

# Enumeration + download run through the shared core.
from scripts.gha_artifacts import download_run_dirs, select_run_artifacts
from scripts.telemetry.jsonl import load_jsonl_records

logger = logging.getLogger(__name__)

# Artifacts whose name starts with this prefix are bot research uploads. Every bot
# run-workflow (tournament / metaculus_cup / minibench) uploads `research-<run_id>`,
# so this single prefix captures all of them via the (workflow-agnostic) artifacts API.
RESEARCH_ARTIFACT_PREFIX = "research-"

# The raw research-provider payload log (metaculus_bot.research.raw_log) rides in
# run_logs/ INSIDE the same research-* artifact as research_outputs/. Its records
# (qid/provider/phase/payload) are a different shape from the per-question research
# records and would corrupt the archive's (qid, run_id) dedup, so the main-archive
# glob must skip these files. scripts/download_raw_research.py archives them separately.
RAW_RESEARCH_LOG_PREFIX = "raw_research_"

# Archive layout defaults, shared with scripts/sync_all.py so the single-pass driver
# writes the same locations the standalone script does.
DEFAULT_OUTPUT_DIR = "backtests/research_archive"
DEFAULT_BACKFILL_DIR = "backtests/research_archive/backfill"


def research_jsonl_files(run_dir: Path) -> list[Path]:
    """List the per-question research JSONL under ``run_dir``, excluding raw-research logs."""
    return [p for p in run_dir.glob("**/*.jsonl") if not p.name.startswith(RAW_RESEARCH_LOG_PREFIX)]


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


def download_research_artifacts(repo: str, since_days: int) -> list[dict]:
    """Download every LIVE research artifact in the repo and return their JSONL records.

    Delegates enumeration + download to the shared core (``select_run_artifacts`` +
    ``download_run_dirs``), then reads the per-question research JSONL from each
    downloaded run dir (excluding the raw-research logs that ride alongside). Logs how
    many artifacts downloaded, records added, and how many were EXPIRED/lost.

    `since_days <= 0` (the default) disables the window and pulls every live artifact.
    """
    selection = select_run_artifacts(
        repo, family_prefixes=(RESEARCH_ARTIFACT_PREFIX,), since_days=since_days, family_label="research"
    )
    logger.info(f"Downloading {len(selection.by_run)} live research artifact(s)...")

    all_records: list[dict] = []
    downloaded = 0
    records_added = 0
    for _run_id, _art, run_dir in download_run_dirs(
        selection, repo, tmp_prefix="research_dl_", progress_noun="research artifacts"
    ):
        jsonl_files = research_jsonl_files(run_dir)
        if jsonl_files:
            downloaded += 1
            for jsonl_file in jsonl_files:
                new_records = load_jsonl_records(jsonl_file)
                records_added += len(new_records)
                all_records.extend(new_records)

    logger.info(
        f"Download phase complete: {downloaded}/{len(selection.by_run)} artifacts downloaded, "
        f"{records_added} records added ({len(selection.expired)} expired/lost)"
    )
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
        "--since-days",
        type=int,
        default=0,
        help=(
            "Optional post-filter: only download artifacts created within this many days. "
            "Default 0 = no window (pull EVERY live artifact). The artifacts endpoint already "
            "returns everything inside the 90-day retention window with no cap, so a window is "
            "rarely needed — use it only to scope a targeted re-pull."
        ),
    )
    parser.add_argument("--repo", default="No-Stream/metaculus-bot", help="GitHub repo")
    parser.add_argument(
        "--backfill-dir",
        default=DEFAULT_BACKFILL_DIR,
        help="Where backfill JSONL lives",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
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

    # --- Phase 1: Download artifacts from GHA (complete artifacts endpoint) ---
    if not args.skip_download:
        all_records.extend(download_research_artifacts(repo=args.repo, since_days=args.since_days))

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
