"""Download GHA artifacts and archive the raw research-provider payload logs.

metaculus_bot.research.raw_log appends each provider's RAW return (AskNews article
dicts per phase, native/Gemini raw responses, prediction-market contracts,
resolution-source fetches, gap-fill search results) to ``run_logs/raw_research_<run_id>.jsonl``.
The four workflows upload ``run_logs/`` — bundled inside ``research-<run_id>`` for the
three prod workflows, as a standalone ``logs-<run_id>`` for test_bot — so, like the
telemetry harvest, this pulls BOTH artifact families.

Archive layout under ``backtests/research_archive/raw/`` (gitignored via ``backtests/``):

    <run_id>.jsonl   # one file per bot run, mirroring that run's raw_research log

IDEMPOTENCY — REPLACE-BY-RUN: a run's uploaded log is immutable, so re-harvesting
overwrites ``<run_id>.jsonl`` byte-identically (records deduped within a run on
``(qid, provider, phase)``, keeping the latest ``fetched_at``). Runs absent from a
harvest (expired artifact) are left untouched. Re-running neither dupes nor loses data.

WHY THIS MUST RUN REGULARLY: GHA deletes artifacts after 90 days. This local archive
is the only durable copy of the raw evidence behind each forecast. Read-only + free
(GitHub API only); no paid LLM/research calls, no publishing. Wrapped by
``make sync_raw_research`` and chained into ``make sync_all``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Reuse the paginated enumeration + gh preflight (download_research) and the
# both-prefix filter + per-artifact download (download_run_logs). Both are
# workflow-agnostic; raw research rides in the same artifacts as the run logs.
from scripts.download_research import _parse_created_at, list_research_artifacts, verify_gh_cli
from scripts.download_run_logs import _download_artifact_to, filter_run_log_artifacts

logger = logging.getLogger(__name__)

DEFAULT_REPO = "No-Stream/metaculus-bot"
DEFAULT_ARCHIVE_DIR = "backtests/research_archive/raw"

# raw_research_<run_id>.jsonl — the run_id is the GITHUB_RUN_ID stamped at write time
# (or "local"), which equals the artifact's originating workflow_run id.
RAW_LOG_FILENAME_RE = re.compile(r"^raw_research_(?P<run_id>.+)\.jsonl$")


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line_num, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning(f"Malformed JSON at {path}:{line_num}, skipping")
    return records


def harvest_raw_logs_from_dir(run_dir: Path) -> dict[str, list[dict]]:
    """Return ``{run_id: [records]}`` from every ``raw_research_*.jsonl`` under ``run_dir``.

    ``run_id`` comes from the filename (the records themselves don't carry it). A dir
    with no raw-research logs yields ``{}`` — expected for an artifact that predates
    the feature or ran with RAW_RESEARCH_LOG disabled.
    """
    out: dict[str, list[dict]] = {}
    for path in sorted(Path(run_dir).glob("**/raw_research_*.jsonl")):
        match = RAW_LOG_FILENAME_RE.match(path.name)
        if match is None:
            continue
        out.setdefault(match.group("run_id"), []).extend(_read_jsonl(path))
    return out


def _dedup_run_records(records: list[dict]) -> list[dict]:
    """Dedup one run's records on ``(qid, provider, phase)``, keeping the latest fetched_at."""
    by_key: dict[tuple, dict] = {}
    for record in records:
        key = (record.get("qid"), record.get("provider"), record.get("phase"))
        existing = by_key.get(key)
        if existing is None or record.get("fetched_at", "") >= existing.get("fetched_at", ""):
            by_key[key] = record
    return sorted(
        by_key.values(),
        key=lambda r: (str(r.get("qid")), str(r.get("provider")), str(r.get("phase")), str(r.get("fetched_at"))),
    )


def merge_and_write(archive_dir: Path, harvested: dict[str, list[dict]]) -> dict[str, int]:
    """Write ``<archive_dir>/<run_id>.jsonl`` for each harvested run (replace-by-run).

    Returns ``{run_id: record_count}`` after dedup. Runs not in ``harvested`` are left
    on disk untouched.
    """
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)

    totals: dict[str, int] = {}
    for run_id, records in harvested.items():
        deduped = _dedup_run_records(records)
        out_path = archive_dir / f"{run_id}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for record in deduped:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        totals[run_id] = len(deduped)
    return totals


def download_and_harvest(repo: str, since_days: int, archive_dir: Path) -> tuple[dict[str, int], int]:
    """Enumerate + download every live run-log artifact, harvest raw logs, merge into the archive.

    Returns ``(per_run_totals, expired_count)``.
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

    # Dedup by run_id so a pagination-duplicated artifact is downloaded at most once.
    by_run: dict[int, dict] = {}
    for art in live:
        run_id = art.get("run_id")
        if run_id is None:
            logger.warning(f"Live artifact {art.get('name')} has no workflow_run id, skipping")
            continue
        by_run.setdefault(run_id, art)

    harvested: dict[str, list[dict]] = {}
    with tempfile.TemporaryDirectory(prefix="raw_research_dl_") as tmpdir:
        tmp_path = Path(tmpdir)
        for idx, (run_id, art) in enumerate(sorted(by_run.items(), key=lambda kv: kv[1].get("created_at", "")), 1):
            run_dir = _download_artifact_to(run_id, repo, art.get("name", ""), tmp_path)
            if run_dir is None:
                continue
            for harvested_run_id, records in harvest_raw_logs_from_dir(run_dir).items():
                harvested.setdefault(harvested_run_id, []).extend(records)
            if idx % 25 == 0:
                print(f"  processed {idx}/{len(by_run)} artifacts, {len(harvested)} runs with raw logs", flush=True)

    totals = merge_and_write(archive_dir, harvested)
    return totals, len(expired)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GHA artifacts and archive raw research-provider logs.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo (owner/name)")
    parser.add_argument(
        "--since-days",
        type=int,
        default=0,
        help="Optional post-filter: only artifacts created within N days (0 = every live artifact).",
    )
    parser.add_argument("--archive-dir", default=DEFAULT_ARCHIVE_DIR, help="Where to write the raw-research archive")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    totals, expired_count = download_and_harvest(
        repo=args.repo, since_days=args.since_days, archive_dir=Path(args.archive_dir)
    )
    total_records = sum(totals.values())
    logger.info("=" * 60)
    logger.info(
        f"Raw-research harvest complete: {len(totals)} run(s) with raw logs, "
        f"{total_records} records archived, {expired_count} expired/lost"
    )
    logger.info(f"Archive written to {args.archive_dir}")


if __name__ == "__main__":
    sys.exit(main())
