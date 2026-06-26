"""Verify the local research archive captures EVERY live GHA research artifact.

After `make sync_research`, run this to prove the local archive at
`backtests/research_archive/` reflects every LIVE `research-*` artifact GitHub still
holds — and to surface, by name + created_at, any EXPIRED artifact whose contents are
gone forever (the unrecoverable loss). It is the QA gate for "maximal completeness".

WHAT IT CHECKS
--------------
1. Enumerate every artifact via the complete, paginated artifacts REST endpoint
   (`gh api --paginate /repos/<repo>/actions/artifacts`) — the same authoritative
   source the puller uses. Split into live `research-*` vs expired `research-*`.
2. Load the rebuilt `manifest.json` and the per-question version files
   (`by_qid/<qid>.jsonl`), collecting the set of `run_id`s the archive represents.
3. For each live artifact, confirm its originating `workflow_run.id` is present among
   the archive's recorded run_ids. A live artifact whose run_id is absent is a GAP
   (its research was not captured) — reported explicitly.
4. Print a clear PASS / FAIL: "all N live artifacts represented in archive", or the
   exact misses, plus any expired (lost-forever) artifacts.

This pull is READ-ONLY and FREE — it hits only the GitHub API, no LLM/research calls,
no publishing — so it is safe to run any time and is NOT subject to the cost gate.

CAVEAT
------
Some runs upload an artifact but produce ZERO research records (e.g. a run that
forecast no questions, or a run whose JSONL had no parseable rows). Such a run_id will
never appear in the archive even though its artifact downloaded fine. We download and
parse each flagged artifact's JSONL to distinguish a genuine GAP (records exist on
GitHub but are missing locally) from an empty artifact (nothing to capture). Empty
artifacts are reported separately and do NOT fail the check.
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

# Reuse the puller's authoritative enumeration + download + parse helpers so the QA
# uses the exact same code paths as the sync itself.
from scripts.download_research import (
    RESEARCH_ARTIFACT_PREFIX,
    download_artifact,
    list_research_artifacts,
    load_jsonl_records,
    verify_gh_cli,
)

logger = logging.getLogger(__name__)


def archived_run_ids(output_dir: Path) -> set[str]:
    """Collect every run_id represented in the rebuilt archive's by_qid/ files.

    The manifest itself doesn't carry run_ids, but each `by_qid/<qid>.jsonl` version
    line does. We read them all and return the set of run_ids as strings (matching how
    records store `run_id`).
    """
    by_qid_dir = output_dir / "by_qid"
    if not by_qid_dir.exists():
        logger.error(f"Archive has no by_qid/ directory at {by_qid_dir} — run `make sync_research` first.")
        sys.exit(1)

    run_ids: set[str] = set()
    for jsonl_file in by_qid_dir.glob("*.jsonl"):
        for record in load_jsonl_records(jsonl_file):
            run_id = record.get("run_id")
            if run_id is not None and str(run_id) != "":
                run_ids.add(str(run_id))
    return run_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the research archive captures every live GHA artifact.")
    parser.add_argument("--repo", default="No-Stream/metaculus-bot", help="GitHub repo")
    parser.add_argument(
        "--output-dir",
        default="backtests/research_archive",
        help="Archive root (contains manifest.json + by_qid/).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    output_dir = Path(args.output_dir)

    verify_gh_cli()

    all_artifacts = list_research_artifacts(args.repo)
    research = [a for a in all_artifacts if str(a.get("name", "")).startswith(RESEARCH_ARTIFACT_PREFIX)]
    live = [a for a in research if not a.get("expired")]
    expired = [a for a in research if a.get("expired")]

    logger.info(f"GitHub: {len(research)} research-* artifacts ({len(live)} live, {len(expired)} expired)")

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"No manifest at {manifest_path} — run `make sync_research` first.")
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())
    archived_ids = archived_run_ids(output_dir)
    logger.info(f"Archive: {len(manifest)} questions, {len(archived_ids)} distinct run_ids across by_qid/ versions")

    # A live artifact is "represented" if its workflow_run id appears among archived run_ids.
    missing: list[dict] = []
    for art in live:
        run_id = art.get("run_id")
        if run_id is None or str(run_id) not in archived_ids:
            missing.append(art)

    # Distinguish genuine gaps (records exist on GitHub but not locally) from empty
    # artifacts (the run produced no research records, so nothing to capture).
    empty_artifacts: list[dict] = []
    genuine_gaps: list[dict] = []
    if missing:
        logger.info(f"Probing {len(missing)} unrepresented live artifact(s) to classify gap vs empty...")
        with tempfile.TemporaryDirectory(prefix="verify_dl_") as tmpdir:
            tmp_path = Path(tmpdir)
            for art in missing:
                run_id = art.get("run_id")
                if run_id is None:
                    genuine_gaps.append(art)
                    continue
                files = download_artifact(run_id, args.repo, art["name"], tmp_path)
                has_records = any(load_jsonl_records(f) for f in files)
                (genuine_gaps if has_records else empty_artifacts).append(art)

    represented = len(live) - len(missing)
    print("\n" + "=" * 72)
    print("RESEARCH ARCHIVE COMPLETENESS CHECK")
    print("=" * 72)
    print(f"Live research artifacts on GitHub : {len(live)}")
    print(f"Represented in local archive       : {represented}")
    print(f"Empty artifacts (no records, OK)   : {len(empty_artifacts)}")
    print(f"Genuine gaps (records NOT captured): {len(genuine_gaps)}")
    print(f"Expired (unrecoverable, lost)      : {len(expired)}")

    if expired:
        print("\nEXPIRED / LOST FOREVER (past 90-day retention):")
        for art in sorted(expired, key=lambda a: a.get("created_at", "")):
            print(f"  LOST: {art.get('name')} (created_at={art.get('created_at')})")

    if empty_artifacts:
        print("\nEmpty live artifacts (downloaded fine but held no research records):")
        for art in sorted(empty_artifacts, key=lambda a: a.get("created_at", "")):
            print(f"  EMPTY: {art.get('name')} (created_at={art.get('created_at')})")

    if genuine_gaps:
        print("\nGAPS — live artifacts with research records NOT in the archive:")
        for art in sorted(genuine_gaps, key=lambda a: a.get("created_at", "")):
            print(f"  GAP: {art.get('name')} (run_id={art.get('run_id')}, created_at={art.get('created_at')})")
        print("\nFAIL: archive is missing capturable research from the artifacts above.")
        print("=" * 72)
        sys.exit(1)

    print(f"\nPASS: all {represented + len(empty_artifacts)} live artifacts represented in archive")
    print(f"      ({represented} with records, {len(empty_artifacts)} legitimately empty).")
    if expired:
        print(f"NOTE: {len(expired)} artifact(s) already expired before any pull — see LOST list above.")
    print("=" * 72)


if __name__ == "__main__":
    main()
