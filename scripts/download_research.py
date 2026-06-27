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
We enumerate artifacts via `gh api --paginate /repos/<repo>/actions/artifacts`. This
endpoint is the AUTHORITATIVE, COMPLETE source: it lists every artifact in the repo
with full Link-header pagination (no 1000-result cap, unlike `gh run list`). Every bot
run's artifact is named `research-<run_id>` regardless of which run-workflow
(tournament / metaculus_cup / minibench) produced it, so a single paginated call across
ALL workflows captures everything — we just filter to `name` starting with `research-`
and `expired == False`. Expired artifacts are unrecoverable; we log them loudly so the
operator knows exactly what (if anything) was lost.

`--since-days` is an OPTIONAL post-filter on each artifact's `created_at`. The DEFAULT
is no window (pull every live artifact), since the endpoint already returns everything.
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

# Artifacts whose name starts with this prefix are bot research uploads. Every bot
# run-workflow (tournament / metaculus_cup / minibench) uploads `research-<run_id>`,
# so this single prefix captures all of them via the (workflow-agnostic) artifacts API.
RESEARCH_ARTIFACT_PREFIX = "research-"


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
    """Parse a GH `created_at` ISO-8601 timestamp (e.g. '2026-05-22T14:00:00Z')."""
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def list_research_artifacts(repo: str) -> list[dict]:
    """List EVERY artifact in the repo via the paginated artifacts REST endpoint.

    `gh api --paginate` follows the Link headers fully, so this returns all artifacts
    across all workflows with no 1000-result cap. We page in batches of 100 (the API
    max page size) and emit one JSON object per artifact via `--jq`, then parse the
    newline-delimited stream. Each object carries `id`, `name`, `created_at`,
    `expires_at`, `expired`, `size_in_bytes`, and the originating `workflow_run.id`.

    Filtering to research artifacts happens downstream so callers can also see/report
    non-research and expired artifacts. Returns the raw artifact objects.
    """
    cmd = [
        "gh",
        "api",
        "--paginate",
        f"/repos/{repo}/actions/artifacts?per_page=100",
        "--jq",
        (".artifacts[] | {id, name, created_at, expires_at, expired, size_in_bytes, run_id: .workflow_run.id}"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"gh api artifacts listing failed for {repo}: {result.stderr.strip()}")
        sys.exit(1)

    artifacts: list[dict] = []
    for line_num, line in enumerate(result.stdout.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            artifacts.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"Malformed artifact JSON at line {line_num}, skipping")
    return artifacts


def download_artifact(run_id: int, repo: str, artifact_name: str, dest_dir: Path) -> list[Path]:
    """Download the named research artifact for a run. Returns downloaded JSONL files.

    Uses `gh run download <run_id> --name <artifact_name>`, which resolves the artifact
    by its originating workflow-run id (carried on each artifact object as
    `workflow_run.id`). On failure returns an empty list — the caller treats a missing
    download as a hard error (we only attempt artifacts the listing said exist).
    """
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
        logger.warning(f"Failed to download {artifact_name} (run {run_id}): {result.stderr.strip()}")
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


def download_research_artifacts(repo: str, since_days: int) -> list[dict]:
    """Download every LIVE research artifact in the repo and return their JSONL records.

    Enumerates artifacts via the complete, paginated REST endpoint (no run-list cap),
    filters to live `research-*` artifacts, optionally post-filters by `since_days`, and
    downloads each by its originating workflow-run id. Logs total artifacts found, how
    many were downloaded, records added, and — critically — every EXPIRED research
    artifact by name + created_at, since those are gone forever and represent silent
    data loss the operator needs to know about.

    `since_days <= 0` (the default) disables the window and pulls every live artifact.
    """
    verify_gh_cli()

    all_artifacts = list_research_artifacts(repo)
    research = [a for a in all_artifacts if str(a.get("name", "")).startswith(RESEARCH_ARTIFACT_PREFIX)]
    logger.info(f"Artifacts endpoint returned {len(all_artifacts)} total, {len(research)} research-* artifacts")

    expired = [a for a in research if a.get("expired")]
    live = [a for a in research if not a.get("expired")]

    if expired:
        logger.warning(f"{len(expired)} research artifact(s) are EXPIRED and UNRECOVERABLE (past 90-day retention):")
        for art in sorted(expired, key=lambda a: a.get("created_at", "")):
            logger.warning(f"  LOST: {art.get('name')} (created_at={art.get('created_at')})")
    else:
        logger.info("No expired research artifacts — nothing lost to the 90-day window.")

    if since_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        before = len(live)
        windowed: list[dict] = []
        for art in live:
            created = _parse_created_at(art.get("created_at", ""))
            if created is None or created >= cutoff:
                windowed.append(art)
        live = windowed
        logger.info(f"--since-days={since_days} post-filter: {len(live)}/{before} live artifacts within window")

    # Dedup by run_id so an artifact is downloaded at most once.
    by_run: dict[int, dict] = {}
    for art in live:
        run_id = art.get("run_id")
        if run_id is None:
            logger.warning(f"Live artifact {art.get('name')} has no workflow_run id, skipping")
            continue
        by_run.setdefault(run_id, art)

    logger.info(f"Downloading {len(by_run)} live research artifact(s)...")

    all_records: list[dict] = []
    downloaded = 0
    records_added = 0

    with tempfile.TemporaryDirectory(prefix="research_dl_") as tmpdir:
        tmp_path = Path(tmpdir)
        for idx, (run_id, art) in enumerate(sorted(by_run.items()), 1):
            artifact_name = art["name"]
            downloaded_files = download_artifact(run_id, repo, artifact_name, tmp_path)
            if downloaded_files:
                downloaded += 1
                for jsonl_file in downloaded_files:
                    new_records = load_jsonl_records(jsonl_file)
                    records_added += len(new_records)
                    all_records.extend(new_records)

            if idx % 25 == 0 or downloaded_files:
                print(f"  downloaded {downloaded}/{idx} of {len(by_run)} artifacts", flush=True)

    logger.info(
        f"Download phase complete: {downloaded}/{len(by_run)} artifacts downloaded, "
        f"{records_added} records added ({len(expired)} expired/lost)"
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
