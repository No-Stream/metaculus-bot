"""Shared GHA-artifact enumeration + download core for the archive-sync scripts.

``download_research`` / ``download_run_logs`` / ``download_raw_research`` all do the
same six things: (1) enumerate EVERY repo artifact via the paginated REST endpoint,
(2) filter to an artifact family by name prefix, (3) report expired (unrecoverable)
artifacts loudly, (4) window by ``--since-days``, (5) dedup by originating
workflow-run id, and (6) download each unique artifact once into a temp dir. This
module is the single implementation of that skeleton.

Two consumers use it:

* the three standalone scripts, each over its own family (research-* for the
  research archive; research-* + logs-* for the run-log + raw-research archives);
* ``scripts/sync_all.py``, which enumerates + downloads ONCE over the UNION family
  and runs all three harvests over the shared downloaded dirs — so a ``make sync_all``
  does ~100 downloads instead of ~300 (three scripts each re-downloading overlapping
  families into their own temp dirs).

The download path is READ-ONLY and FREE — it hits only the GitHub API, no LLM /
research calls, no publishing.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# subprocess.run timeouts (seconds). Bounding both keeps one slow/hung `gh` call from
# stalling a scheduled pull: a per-artifact download that times out is skipped (the
# other artifacts still process), and the artifacts enumeration can't block forever.
GH_API_TIMEOUT_S = 120
ARTIFACT_DOWNLOAD_TIMEOUT_S = 180


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

    Filtering to a specific artifact family happens downstream (see ``split_by_family``)
    so callers can also see/report non-family and expired artifacts. Returns the raw
    artifact objects.
    """
    cmd = [
        "gh",
        "api",
        "--paginate",
        f"/repos/{repo}/actions/artifacts?per_page=100",
        "--jq",
        (".artifacts[] | {id, name, created_at, expires_at, expired, size_in_bytes, run_id: .workflow_run.id}"),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=GH_API_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        logger.error(f"gh api artifacts listing timed out ({GH_API_TIMEOUT_S}s) for {repo}")
        sys.exit(1)
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


def split_by_family(artifacts: list[dict], prefixes: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    """Split artifacts whose name starts with any of ``prefixes`` into ``(live, expired)``."""
    family = [a for a in artifacts if str(a.get("name", "")).startswith(prefixes)]
    live = [a for a in family if not a.get("expired")]
    expired = [a for a in family if a.get("expired")]
    return live, expired


def report_expired(expired: list[dict], family_label: str) -> None:
    """Log every expired artifact by name + created_at (a loud, operator-watched LOST line).

    Expired artifacts are past the 90-day retention and gone from GitHub forever, so this
    is the only signal that data was silently lost. The ``LOST: <name> (created_at=...)``
    format is load-bearing — operators grep for it.
    """
    if expired:
        logger.warning(
            f"{len(expired)} {family_label} artifact(s) are EXPIRED and UNRECOVERABLE (past 90-day retention):"
        )
        for art in sorted(expired, key=lambda a: a.get("created_at", "")):
            logger.warning(f"  LOST: {art.get('name')} (created_at={art.get('created_at')})")
    else:
        logger.info(f"No expired {family_label} artifacts — nothing lost to the 90-day window.")


def _dedup_by_run(live: list[dict]) -> dict[int, dict]:
    """Keep one artifact per originating workflow-run id (first-wins on the pagination stream)."""
    by_run: dict[int, dict] = {}
    for art in live:
        run_id = art.get("run_id")
        if run_id is None:
            logger.warning(f"Live artifact {art.get('name')} has no workflow_run id, skipping")
            continue
        by_run.setdefault(run_id, art)
    return by_run


@dataclass
class ArtifactSelection:
    """Result of ``select_run_artifacts``: the artifacts to download + the expired ones.

    ``by_run`` maps run_id -> artifact (deduped, live, windowed). ``expired`` is the
    family's expired artifacts, retained so the caller can report the count in its
    summary (they're already logged loudly by ``report_expired``).
    """

    by_run: dict[int, dict]
    expired: list[dict]
    total_artifacts: int


def select_run_artifacts(
    repo: str, *, family_prefixes: tuple[str, ...], since_days: int, family_label: str
) -> ArtifactSelection:
    """Enumerate once, filter to the family, report expired, window by ``since_days``, dedup by run_id.

    ``since_days <= 0`` disables the window and selects every live artifact (the endpoint
    already returns everything inside the 90-day retention window with no cap).
    """
    verify_gh_cli()

    all_artifacts = list_research_artifacts(repo)
    live, expired = split_by_family(all_artifacts, family_prefixes)
    logger.info(
        f"Artifacts endpoint returned {len(all_artifacts)} total, "
        f"{len(live)} live + {len(expired)} expired {family_label} artifact(s)"
    )
    report_expired(expired, family_label)

    if since_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        before = len(live)
        live = [a for a in live if (_parse_created_at(a.get("created_at", "")) or cutoff) >= cutoff]
        logger.info(f"--since-days={since_days} post-filter: {len(live)}/{before} live artifacts within window")

    return ArtifactSelection(by_run=_dedup_by_run(live), expired=expired, total_artifacts=len(all_artifacts))


def _download_artifact_to(run_id: int, repo: str, artifact_name: str, dest_dir: Path) -> Path | None:
    """Download one artifact into ``dest_dir/<run_id>``; return the dir or None on failure.

    Resolves the artifact by its originating workflow-run id (carried on each artifact
    object as ``workflow_run.id``). A timeout or non-zero exit is logged and returns None
    so the download loop can skip this run and keep going — one hung `gh` never sinks the pull.
    """
    run_dir = dest_dir / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["gh", "run", "download", str(run_id), "--repo", repo, "--name", artifact_name, "--dir", str(run_dir)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=ARTIFACT_DOWNLOAD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        logger.warning(
            f"Timed out ({ARTIFACT_DOWNLOAD_TIMEOUT_S}s) downloading {artifact_name} (run {run_id}); skipping"
        )
        return None
    if result.returncode != 0:
        logger.warning(f"Failed to download {artifact_name} (run {run_id}): {result.stderr.strip()}")
        return None
    return run_dir


def download_run_dirs(
    selection: ArtifactSelection, repo: str, *, tmp_prefix: str, progress_noun: str = "artifacts"
) -> Iterator[tuple[int, dict, Path]]:
    """Download each unique artifact into ONE temp dir, yielding ``(run_id, artifact, run_dir)``.

    A per-artifact download that fails (returns None) is skipped so one hung `gh` never
    aborts the pull. The temp dir lives for the life of the generator, so the consumer must
    harvest each ``run_dir`` INSIDE the iteration (before the next artifact / generator close)
    — every consumer here does exactly that. Progress is printed every 25 downloads so a
    long pull isn't silent.
    """
    by_run = selection.by_run
    total = len(by_run)
    with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdir:
        tmp_path = Path(tmpdir)
        for idx, (run_id, art) in enumerate(sorted(by_run.items(), key=lambda kv: kv[1].get("created_at", "")), 1):
            run_dir = _download_artifact_to(run_id, repo, art.get("name", ""), tmp_path)
            if run_dir is None:
                continue
            yield run_id, art, run_dir
            if idx % 25 == 0:
                print(f"  processed {idx}/{total} {progress_noun}", flush=True)
