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
2. Confirm the PERSISTED STORE holds every live artifact. The store
   (`backtests/gha_artifact_store/`) is the durable copy; GHA is a 90-day staging area,
   so an artifact GitHub still has but the store does not is data one clock-tick from
   being unrecoverable. This is the "verify by count" step: a store gap FAILS.
3. Load the rebuilt `manifest.json` and the per-question version files
   (`by_qid/<qid>.jsonl`), collecting the set of `run_id`s the archive represents.
4. For each live artifact, confirm its originating `workflow_run.id` is present among
   the archive's recorded run_ids. A live artifact whose run_id is absent is a GAP
   (its research was not captured) — reported explicitly.
5. Confirm the MERGE stage promoted what it captured: every question holding an artifact
   record must be served by one in `latest/`. Presence in `by_qid/` is not the same as
   being the record a backtest replays, and for months it wasn't — a raw-string timestamp
   sort handed `latest/` to the lossy comment reconstruction on 255 of 256 dual-sourced
   questions while this check reported PASS throughout, because it only ever looked at
   `by_qid/`. That blind spot is why the bug survived so long.
6. Print a clear PASS / FAIL: "all N live artifacts represented in archive", or the
   exact misses, plus any expired (lost-forever) artifacts.

This pull is READ-ONLY and FREE — it hits only the GitHub API, no LLM/research calls,
no publishing — so it is safe to run any time and is NOT subject to the cost gate.

CAVEAT — MOST ARTIFACTS LEGITIMATELY HOLD NO RESEARCH
-----------------------------------------------------
A run uploads an artifact even when it forecast no questions, and since every workflow
began bundling `run_logs/` inside `research-*` such a run still produces an artifact —
one carrying logs and no `research_outputs/` at all. That is the ordinary case, not an
anomaly: 632 of the 859 artifacts on disk are exactly this, which is why the archive
holds artifact records from 227 runs rather than 859. Such a run_id can never appear in
the archive, so each flagged artifact's JSONL is parsed to tell a genuine GAP (records
exist but are missing locally) from an empty artifact (nothing to capture). Empty
artifacts are reported separately and do NOT fail the check. The parse reads the
PERSISTED copy when there is one and only downloads what the store lacks.
"""

import argparse
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Reuse the puller's authoritative enumeration + download + parse helpers so the QA
# uses the exact same code paths as the sync itself.
from scripts.download_research import (
    RESEARCH_ARTIFACT_PREFIX,
    SOURCE_ARTIFACT,
    load_jsonl_records,
    research_jsonl_files,
)
from scripts.gha_artifacts import (
    DEFAULT_STORE_DIR,
    _download_artifact_to,
    is_persisted,
    list_research_artifacts,
    store_run_dir,
    verify_gh_cli,
)

logger = logging.getLogger(__name__)

# How many examples each capped list names before collapsing the rest into a count. The
# summary counts are always exact; these lists exist to name a few instances.
_SAMPLE_PRINT_LIMIT = 20


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


def unpersisted_artifacts(live: list[dict], store_dir: Path) -> list[dict]:
    """Live artifacts GitHub still holds that the persisted store does NOT.

    Every one is research whose only copy is on a 90-day clock. Run `make sync_all` to
    close the gap; a handful right after a fresh bot run just means the sync hasn't run
    since (this check is meant to run right after one).
    """
    return [art for art in live if not is_persisted(store_dir, str(art.get("name", "")))]


def classify_gap_or_empty(
    missing: list[dict], repo: str, store_dir: Path, download_dir: Path
) -> tuple[list[dict], list[dict]]:
    """Split unrepresented artifacts into ``(genuine_gaps, empty_artifacts)``.

    Reads the PERSISTED copy whenever the store has one (no network, no re-extraction)
    and downloads only what it lacks — so on a healthy store this classification is
    entirely offline.
    """
    genuine_gaps: list[dict] = []
    empty_artifacts: list[dict] = []
    for art in missing:
        name = str(art.get("name", ""))
        run_id = art.get("run_id")
        if is_persisted(store_dir, name):
            run_dir = store_run_dir(store_dir, name)
        elif run_id is None:
            genuine_gaps.append(art)
            continue
        else:
            run_dir = _download_artifact_to(run_id, repo, name, download_dir)
        files = research_jsonl_files(run_dir) if run_dir is not None else []
        has_records = any(load_jsonl_records(f) for f in files)
        (genuine_gaps if has_records else empty_artifacts).append(art)
    return genuine_gaps, empty_artifacts


def _print_artifact_sample(header: str, artifacts: list[dict], label: str, limit: int = _SAMPLE_PRINT_LIMIT) -> None:
    """Print an oldest-first artifact list, capped, with the remainder counted.

    The counts in the summary block above are always exact; these lists exist to name
    examples. Capping matters because the ordinary case is now LARGE — 633 of the 860 live
    artifacts hold no research — and an uncapped dump buried the PASS/FAIL verdict under
    hundreds of expected lines.
    """
    print(f"\n{header}")
    ordered = sorted(artifacts, key=lambda a: a.get("created_at", ""))
    for art in ordered[:limit]:  # HARNESS-SCAN-EXEMPT-subsampling
        print(f"  {label}: {art.get('name')} (created_at={art.get('created_at')})")
    if len(ordered) > limit:
        print(f"  ... and {len(ordered) - limit} more")


def unpromoted_artifact_questions(manifest: dict) -> list[str]:
    """Questions that HAVE an artifact record but whose latest/ is served by a lesser source.

    The manifest's ``sources`` lists every source class in that question's ``by_qid/``
    history and ``latest_source`` names the one that won. An artifact in the history that
    did not win means the merge stage demoted the authoritative capture — the exact failure
    the precedence fix addressed, and one that leaves the archive looking complete while a
    replay reads trimmed comment text.
    """
    return sorted(
        qid
        for qid, entry in manifest.items()
        if SOURCE_ARTIFACT in entry.get("sources", []) and entry.get("latest_source") != SOURCE_ARTIFACT
    )


@dataclass
class CompletenessFindings:
    """Everything the check measured, gathered before any of it is printed."""

    live: list[dict]
    expired: list[dict]
    unpersisted: list[dict]
    missing: list[dict]
    genuine_gaps: list[dict]
    empty_artifacts: list[dict]
    unpromoted: list[str]
    manifest: dict

    @property
    def represented(self) -> int:
        """Live artifacts whose originating run appears in the archive."""
        return len(self.live) - len(self.missing)

    @property
    def failed(self) -> bool:
        """The three FAIL conditions. An EMPTY artifact is deliberately not one of them."""
        return bool(self.genuine_gaps or self.unpromoted or self.unpersisted)


def load_manifest(output_dir: Path) -> dict:
    """Read the rebuilt archive's manifest, exiting loudly when the sync hasn't run."""
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"No manifest at {manifest_path} — run `make sync_research` first.")
        sys.exit(1)
    return json.loads(manifest_path.read_text())


def unrepresented_artifacts(live: list[dict], archived_ids: set[str]) -> list[dict]:
    """Live artifacts whose ``workflow_run.id`` is absent from the archive's recorded run_ids.

    "Represented" is run-id presence only, so an artifact that legitimately carried no
    research lands here too — ``classify_gap_or_empty`` is what tells the two apart.
    """
    return [art for art in live if art.get("run_id") is None or str(art.get("run_id")) not in archived_ids]


def gather_findings(repo: str, output_dir: Path, store_dir: Path) -> CompletenessFindings:
    """Measure GitHub's live artifact set against the persisted store and the merged archive."""
    verify_gh_cli()

    all_artifacts = list_research_artifacts(repo)
    research = [a for a in all_artifacts if str(a.get("name", "")).startswith(RESEARCH_ARTIFACT_PREFIX)]
    live = [a for a in research if not a.get("expired")]
    expired = [a for a in research if a.get("expired")]

    logger.info(f"GitHub: {len(research)} research-* artifacts ({len(live)} live, {len(expired)} expired)")

    unpersisted = unpersisted_artifacts(live, store_dir)

    manifest = load_manifest(output_dir)
    archived_ids = archived_run_ids(output_dir)
    logger.info(f"Archive: {len(manifest)} questions, {len(archived_ids)} distinct run_ids across by_qid/ versions")

    missing = unrepresented_artifacts(live, archived_ids)

    # Distinguish genuine gaps (records exist on GitHub but not locally) from empty
    # artifacts (the run produced no research records, so nothing to capture).
    empty_artifacts: list[dict] = []
    genuine_gaps: list[dict] = []
    if missing:
        logger.info(f"Classifying {len(missing)} unrepresented live artifact(s) as gap vs empty...")
        with tempfile.TemporaryDirectory(prefix="verify_dl_") as tmpdir:
            genuine_gaps, empty_artifacts = classify_gap_or_empty(missing, repo, store_dir, Path(tmpdir))

    return CompletenessFindings(
        live=live,
        expired=expired,
        unpersisted=unpersisted,
        missing=missing,
        genuine_gaps=genuine_gaps,
        empty_artifacts=empty_artifacts,
        unpromoted=unpromoted_artifact_questions(manifest),
        manifest=manifest,
    )


def _print_counts(findings: CompletenessFindings) -> None:
    print("\n" + "=" * 72)
    print("RESEARCH ARCHIVE COMPLETENESS CHECK")
    print("=" * 72)
    print(f"Live research artifacts on GitHub : {len(findings.live)}")
    print(f"Persisted in the local store       : {len(findings.live) - len(findings.unpersisted)}")
    print(f"NOT persisted (at 90-day risk)     : {len(findings.unpersisted)}")
    print(f"Represented in local archive       : {findings.represented}")
    print(f"Empty artifacts (no records, OK)   : {len(findings.empty_artifacts)}")
    print(f"Genuine gaps (records NOT captured): {len(findings.genuine_gaps)}")
    print(f"Expired (unrecoverable, lost)      : {len(findings.expired)}")
    print(f"Captured but not promoted to latest: {len(findings.unpromoted)}")


def _print_expired(expired: list[dict]) -> None:
    """Name every expired artifact, deliberately UNCAPPED unlike the sampled lists.

    This is the one section reporting permanent data loss, so the operator needs every name.
    """
    if not expired:
        return
    print("\nEXPIRED / LOST FOREVER (past 90-day retention):")
    for art in sorted(expired, key=lambda a: a.get("created_at", "")):
        print(f"  LOST: {art.get('name')} (created_at={art.get('created_at')})")


def _print_unpromoted(unpromoted: list[str], manifest: dict) -> None:
    if not unpromoted:
        return
    print("\nNOT PROMOTED — questions with an artifact record that latest/ does not serve:")
    # Report truncation, not analysis: the count above is exact and the tail is named.
    for qid in unpromoted[:_SAMPLE_PRINT_LIMIT]:  # HARNESS-SCAN-EXEMPT-subsampling
        print(f"  DEMOTED: qid={qid} (latest_source={manifest[qid].get('latest_source')})")
    if len(unpromoted) > _SAMPLE_PRINT_LIMIT:
        print(f"  ... and {len(unpromoted) - _SAMPLE_PRINT_LIMIT} more")


def _print_failures(findings: CompletenessFindings, store_dir: Path) -> None:
    """Name each FAIL condition that fired; every one of them exits non-zero."""
    if findings.unpersisted:
        print(
            f"\nFAIL: {len(findings.unpersisted)} live artifact(s) are not in the local store "
            f"({store_dir}) — run `make sync_all` to grab them before they expire."
        )
    if findings.genuine_gaps:
        print("\nGAPS — live artifacts with research records NOT in the archive:")
        for art in sorted(findings.genuine_gaps, key=lambda a: a.get("created_at", "")):
            print(f"  GAP: {art.get('name')} (run_id={art.get('run_id')}, created_at={art.get('created_at')})")
        print("\nFAIL: archive is missing capturable research from the artifacts above.")
    if findings.unpromoted:
        print("\nFAIL: the merge stage demoted captured artifact research on the questions above.")


def _print_pass(findings: CompletenessFindings) -> None:
    print(f"\nPASS: all {findings.represented + len(findings.empty_artifacts)} live artifacts represented in archive")
    print(f"      ({findings.represented} with records, {len(findings.empty_artifacts)} legitimately empty).")
    if findings.expired:
        print(f"NOTE: {len(findings.expired)} artifact(s) already expired before any pull — see LOST list above.")


def print_report(findings: CompletenessFindings, store_dir: Path) -> None:
    """Print the whole verdict block; ``findings.failed`` carries the exit decision."""
    _print_counts(findings)
    _print_expired(findings.expired)
    if findings.empty_artifacts:
        _print_artifact_sample(
            "Empty live artifacts (harvested fine but held no research records):",
            findings.empty_artifacts,
            "EMPTY",
        )
    _print_unpromoted(findings.unpromoted, findings.manifest)
    if findings.unpersisted:
        _print_artifact_sample(
            "NOT PERSISTED — live artifacts whose only copy is still on GitHub's 90-day clock:",
            findings.unpersisted,
            "UNPERSISTED",
        )
    if findings.failed:
        _print_failures(findings, store_dir)
    else:
        _print_pass(findings)
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the research archive captures every live GHA artifact.")
    parser.add_argument("--repo", default="No-Stream/metaculus-bot", help="GitHub repo")
    parser.add_argument(
        "--output-dir",
        default="backtests/research_archive",
        help="Archive root (contains manifest.json + by_qid/).",
    )
    parser.add_argument(
        "--store-dir",
        default=DEFAULT_STORE_DIR,
        help="The persisted artifact store to check for coverage.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    store_dir = Path(args.store_dir)

    findings = gather_findings(args.repo, Path(args.output_dir), store_dir)
    print_report(findings, store_dir)
    if findings.failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
