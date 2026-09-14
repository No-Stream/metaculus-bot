"""Coverage audit of a completed residual-round pull: which resolved posts produced no record, and why.

Free and offline: it reads the pull's checkpoint (the raw post payloads it fetched) and the records it
emitted, plus the prior round's same pair, and makes no API call. Playbook Phase 2's "spot-check
re-pull stability" step, with the coverage and diff checks the per-round scripts grew around it.

Five checks:

1. Coverage: every resolved post the pull fetched must either carry an emitted record or classify
   benign -- annulled, or a post a prior round already saw produce no record. Anything else is
   INVESTIGATE.
2. Group posts: any sub-question of a multi-question post with a real resolution and no record.
3. Diff versus the prior round: the new cohort is what the round is about, written out as JSON.
4. Re-pull stability: every platform score on an overlapping record must reproduce exactly.
   Metaculus re-resolves in place without moving any timestamp, so a moved score means a table in
   the prior round's write-up is silently stale (``docs/performance_analysis.md``).
5. Per-model parse parity: how many overlapping records carry parsed per-model forecasts and comment
   text before and after. A collector change that quietly stops parsing comments shows up here.

"Known from a prior round" is DERIVED from the prior pull, not listed: a post the prior checkpoint
fetched and the prior records skipped already carries a verdict in that round's write-up, while one
the prior pull never fetched reads INVESTIGATE (how post 44950 surfaced on 2026-09-09).

Usage:
    uv run python scripts/verify_pull.py --records scratch/residual_2026-09-09/perf_summer-futureeval-2026.json \
        --prior-records scratch/residual_2026-09-01/perf_summer-futureeval-2026.json
    uv run python scripts/verify_pull.py --records <path> --prior-records <path> --output /tmp/new_since_prior.json
    make verify_pull ARGS="--records <path> --prior-records <path>"
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from metaculus_bot.performance_analysis.collector import questions_on_post
from scripts.supply_probe_platforms import bot_forecast_state

logger = logging.getLogger(__name__)

RecordKey = tuple[int, int]

# A resolution that legitimately produces no scoreable record.
ANNULLED_RESOLUTIONS: frozenset[Any] = frozenset({"annulled", "ambiguous"})

VERDICT_ANNULLED = "benign (annulled)"
VERDICT_KNOWN_RECORDLESS = "known from a prior round"
VERDICT_INVESTIGATE = "INVESTIGATE (new recordless post)"

# The tournament ranks on spot peer; the others ride along so a re-pull moving any of them is visible.
STABILITY_FIELDS: tuple[str, ...] = (
    "spot_peer_score",
    "peer_score",
    "baseline_score",
    "spot_baseline_score",
    "coverage",
)
# Emptiness here means the pull stopped parsing the bot's comment, not that the bot published less.
PARITY_FIELDS: tuple[str, ...] = (
    "per_model_forecasts",
    "per_base_model_forecasts",
    "per_model_numeric_percentiles",
    "comment_text",
)
SCORE_TOLERANCE = 1e-9
DEFAULT_MAX_WORST_ROWS = 12
# The naming `scratch/residual_*/resilient_pull.py` gives its checkpoint, so one --records path locates both.
CHECKPOINT_SUFFIX = "_checkpoint.json"
TITLE_WIDTH = 55
WORST_TITLE_WIDTH = 50


@dataclass(frozen=True)
class RecordlessPost:
    """A resolved post the pull fetched and emitted no record for, with its classification."""

    post_id: int
    verdict: str
    status: str | None
    question_types: tuple[str, ...]
    resolutions: tuple[Any, ...]
    forecast_states: tuple[str, ...]
    comment_count: int | None
    title: str


@dataclass(frozen=True)
class MissingSubQuestion:
    """A resolved member of a covered group post that produced no record of its own."""

    post_id: int
    question_id: Any
    resolution: Any
    question_type: str | None
    forecast_state: str
    title: str


@dataclass(frozen=True)
class CoverageAudit:
    checkpoint_posts: int
    records: int
    record_posts: int
    group_posts: int
    recordless: tuple[RecordlessPost, ...]
    missing_sub_questions: tuple[MissingSubQuestion, ...]

    @property
    def investigate(self) -> tuple[int, ...]:
        return tuple(post.post_id for post in self.recordless if post.verdict == VERDICT_INVESTIGATE)


@dataclass(frozen=True)
class PriorDiff:
    prior_records: int
    new_records: int
    added: tuple[dict[str, Any], ...]
    lost: tuple[RecordKey, ...]
    added_types: Mapping[str, int]
    without_spot_peer: int
    resolution_set_days: Mapping[str, int]
    actual_resolve_days: Mapping[str, int]


@dataclass(frozen=True)
class StabilityMismatch:
    question_id: int
    field: str
    prior: float | None
    now: float | None


@dataclass(frozen=True)
class StabilityAudit:
    overlap: int
    expected_overlap: int
    checked: Mapping[str, int]
    mismatches: tuple[StabilityMismatch, ...]

    @property
    def mismatches_by_field(self) -> Mapping[str, int]:
        return Counter(mismatch.field for mismatch in self.mismatches)


@dataclass(frozen=True)
class ParityRow:
    field: str
    prior_nonempty: int
    now_nonempty: int
    lost: tuple[int, ...]


def checkpoint_path_for(records_path: Path) -> Path:
    """The checkpoint the pull wrote beside its records file."""
    return records_path.with_name(f"{records_path.stem}{CHECKPOINT_SUFFIX}")


def platform_score(record: Mapping[str, Any], field: str) -> float | None:
    """One of the record's Metaculus scores, or None where the pull recorded none."""
    scores = record.get("metaculus_scores")
    return scores.get(field) if isinstance(scores, Mapping) else None


def record_key(record: Mapping[str, Any]) -> RecordKey:
    return (record["question_id"], record["post_id"])


def recordless_post_ids(checkpoint: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> set[int]:
    """Post ids the pull fetched that emitted no record."""
    scored = {record["post_id"] for record in records}
    return {int(post_id) for post_id in checkpoint if int(post_id) not in scored}


def _verdict(resolutions: Sequence[Any], *, known: bool) -> str:
    if resolutions and all(resolution in ANNULLED_RESOLUTIONS for resolution in resolutions):
        return VERDICT_ANNULLED
    return VERDICT_KNOWN_RECORDLESS if known else VERDICT_INVESTIGATE


def _recordless_post(post: Mapping[str, Any], post_id: int, *, known: bool) -> RecordlessPost:
    questions = questions_on_post(post)
    resolutions = tuple(question.get("resolution") for question in questions)
    return RecordlessPost(
        post_id=post_id,
        verdict=_verdict(resolutions, known=known),
        status=post.get("status"),
        question_types=tuple(str(question.get("type")) for question in questions),
        resolutions=resolutions,
        forecast_states=tuple(bot_forecast_state(question) for question in questions),
        comment_count=post.get("comment_count"),
        title=str(post.get("title") or ""),
    )


def _missing_sub_questions(
    checkpoint: Mapping[str, Any], record_post_ids: set[Any], record_question_ids: set[Any]
) -> tuple[MissingSubQuestion, ...]:
    """Resolved members of group posts that DID emit records; whole-post drops are check 1's job."""
    rows: list[MissingSubQuestion] = []
    for post_id_str, post in checkpoint.items():
        post_id = int(post_id_str)
        questions = questions_on_post(post)
        if post_id not in record_post_ids or len(questions) < 2:
            continue
        for question in questions:
            resolution = question.get("resolution")
            if question.get("id") in record_question_ids or resolution is None or resolution in ANNULLED_RESOLUTIONS:
                continue
            rows.append(
                MissingSubQuestion(
                    post_id=post_id,
                    question_id=question.get("id"),
                    resolution=resolution,
                    question_type=question.get("type"),
                    forecast_state=bot_forecast_state(question),
                    title=str(question.get("title") or ""),
                )
            )
    return tuple(rows)


def audit_coverage(
    checkpoint: Mapping[str, Any], records: Sequence[Mapping[str, Any]], *, prior_recordless: frozenset[int]
) -> CoverageAudit:
    """Classify every fetched post that produced no record. Pure."""
    record_post_ids = {record["post_id"] for record in records}
    record_question_ids = {record["question_id"] for record in records}
    return CoverageAudit(
        checkpoint_posts=len(checkpoint),
        records=len(records),
        record_posts=len(record_post_ids),
        group_posts=sum(1 for post in checkpoint.values() if len(questions_on_post(post)) > 1),
        recordless=tuple(
            _recordless_post(checkpoint[str(post_id)], post_id, known=post_id in prior_recordless)
            for post_id in sorted(recordless_post_ids(checkpoint, records))
        ),
        missing_sub_questions=_missing_sub_questions(checkpoint, record_post_ids, record_question_ids),
    )


def _new_record_entry(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    return {
        "question_id": record["question_id"],
        "post_id": record["post_id"],
        "type": record["type"],
        "title": record.get("title"),
        "spot_peer_score": platform_score(record, "spot_peer_score"),
        "peer_score": platform_score(record, "peer_score"),
        "coverage": platform_score(record, "coverage"),
        "bot_comment_created_at": record.get("bot_comment_created_at"),
        "open_time": metadata.get("open_time"),
        "scheduled_resolve_time": metadata.get("scheduled_resolve_time"),
        "actual_resolve_time": metadata.get("actual_resolve_time"),
        "resolution_set_time": metadata.get("resolution_set_time"),
    }


def _day_histogram(entries: Sequence[Mapping[str, Any]], field: str) -> Mapping[str, int]:
    return dict(sorted(Counter(str(entry[field])[:10] for entry in entries).items()))


def diff_against_prior(records: Sequence[Mapping[str, Any]], prior_records: Sequence[Mapping[str, Any]]) -> PriorDiff:
    """The cohort this round is about, plus any prior record the re-pull lost. Pure."""
    prior_keys = {record_key(record) for record in prior_records}
    new_keys = {record_key(record) for record in records}
    added = tuple(
        _new_record_entry(record)
        for record in sorted((record for record in records if record_key(record) not in prior_keys), key=record_key)
    )
    return PriorDiff(
        prior_records=len(prior_records),
        new_records=len(records),
        added=added,
        lost=tuple(sorted(prior_keys - new_keys)),
        added_types=dict(Counter(entry["type"] for entry in added)),
        without_spot_peer=sum(1 for entry in added if entry["spot_peer_score"] is None),
        resolution_set_days=_day_histogram(added, "resolution_set_time"),
        actual_resolve_days=_day_histogram(added, "actual_resolve_time"),
    )


def audit_stability(records: Sequence[Mapping[str, Any]], prior_records: Sequence[Mapping[str, Any]]) -> StabilityAudit:
    """Compare every platform score on the overlapping records against the prior pull's. Pure."""
    prior_by_key = {record_key(record): record for record in prior_records}
    checked: Counter[str] = Counter()
    mismatches: list[StabilityMismatch] = []
    overlap = 0
    for record in records:
        prior = prior_by_key.get(record_key(record))
        if prior is None:
            continue
        overlap += 1
        for field in STABILITY_FIELDS:
            before, after = platform_score(prior, field), platform_score(record, field)
            if before is None and after is None:
                continue
            checked[field] += 1
            if before is None or after is None or abs(before - after) > SCORE_TOLERANCE:
                mismatches.append(StabilityMismatch(record["question_id"], field, before, after))
    return StabilityAudit(overlap, len(prior_records), dict(checked), tuple(mismatches))


def audit_parse_parity(
    records: Sequence[Mapping[str, Any]], prior_records: Sequence[Mapping[str, Any]]
) -> tuple[ParityRow, ...]:
    """Per-field counts of parsed comment data on the overlapping records, before and after. Pure."""
    prior_by_key = {record_key(record): record for record in prior_records}
    overlapping = [(prior, record) for record in records if (prior := prior_by_key.get(record_key(record))) is not None]
    rows: list[ParityRow] = []
    for field in PARITY_FIELDS:
        rows.append(
            ParityRow(
                field=field,
                prior_nonempty=sum(1 for prior, _ in overlapping if prior.get(field)),
                now_nonempty=sum(1 for _, record in overlapping if record.get(field)),
                lost=tuple(
                    record["question_id"] for prior, record in overlapping if prior.get(field) and not record.get(field)
                ),
            )
        )
    return tuple(rows)


def _render_coverage(coverage: CoverageAudit, *, prior_pull_known: bool) -> list[str]:
    lines = [
        "=== 1. coverage: resolved posts vs emitted records ===",
        f"  resolved posts fetched (checkpoint): {coverage.checkpoint_posts}",
        f"  records emitted: {coverage.records}  (distinct posts: {coverage.record_posts})",
        f"  group posts (>1 sub-question) among fetched: {coverage.group_posts}",
        f"  posts with no emitted record: {len(coverage.recordless)}",
    ]
    for post in coverage.recordless:
        lines.append(
            f"    post {post.post_id}: {post.verdict} status={post.status} types={list(post.question_types)} "
            f"resolutions={list(post.resolutions)} forecast={list(post.forecast_states)} "
            f"comment_count={post.comment_count} :: {post.title[:TITLE_WIDTH]}"
        )
    lines.append(f"  -> {len(coverage.investigate)} post(s) need investigation: {list(coverage.investigate)}")
    if not prior_pull_known:
        lines.append(
            "  no prior pull to compare against, so every non-annulled recordless post reads INVESTIGATE; "
            "pass --prior-records and its checkpoint to inherit the prior round's classifications"
        )
    return lines


def _render_sub_questions(coverage: CoverageAudit) -> list[str]:
    lines = ["", "=== 2. group posts: sub-questions without a record ==="]
    for row in coverage.missing_sub_questions:
        lines.append(
            f"    post {row.post_id} question {row.question_id}: resolution={row.resolution!r} "
            f"type={row.question_type} forecast={row.forecast_state} :: {row.title[:TITLE_WIDTH]}"
        )
    lines.append(f"  sub-questions resolved-but-recordless inside covered posts: {len(coverage.missing_sub_questions)}")
    return lines


def _score_cell(value: float | None) -> str:
    """A score column that survives a record scored on one convention and not the other."""
    return f"{value:>8.2f}" if value is not None else f"{'-':>8}"


def _render_worst(diff: PriorDiff, max_rows: int) -> list[str]:
    scored = sorted(
        (entry for entry in diff.added if entry["spot_peer_score"] is not None),
        key=lambda entry: entry["spot_peer_score"],
    )
    lines = ["  worst new SPOT peer scores (coverage-scaled peer as labelled secondary):"]
    for entry in scored[:max_rows]:
        lines.append(
            f"    spot {_score_cell(entry['spot_peer_score'])}  peer {_score_cell(entry['peer_score'])}  "
            f"q{entry['question_id']} post {entry['post_id']} [{entry['type']}] "
            f"{str(entry['title'])[:WORST_TITLE_WIDTH]}"
        )
    hidden = max(0, len(scored) - max_rows)
    if hidden:
        lines.append(f"    +{hidden} more scored new records (raise --max-worst-rows to see them)")
    return lines


def _render_diff(diff: PriorDiff, max_rows: int) -> list[str]:
    lines = [
        "",
        "=== 3. diff vs the prior round ===",
        f"  prior records: {diff.prior_records}   this pull: {diff.new_records}",
        f"  NEW since prior: {len(diff.added)}",
        f"  new-record types: {dict(diff.added_types)}",
        f"  prior records MISSING from this pull: {len(diff.lost)} -> {list(diff.lost)}",
        f"  new records lacking spot_peer_score: {diff.without_spot_peer}",
        "  resolution_set_time day histogram for the new records:",
    ]
    lines.extend(f"    {day}: {count}" for day, count in diff.resolution_set_days.items())
    lines.append("  actual_resolve_time day histogram for the new records:")
    lines.extend(f"    {day}: {count}" for day, count in diff.actual_resolve_days.items())
    lines.extend(_render_worst(diff, max_rows))
    return lines


def _render_stability(stability: StabilityAudit) -> list[str]:
    lines = ["", "=== 4. re-pull stability: overlapping platform scores must reproduce ==="]
    for mismatch in stability.mismatches:
        lines.append(
            f"    MISMATCH q{mismatch.question_id} {mismatch.field}: prior={mismatch.prior} now={mismatch.now}"
        )
    lines.append(f"  overlapping records: {stability.overlap} (prior pull held {stability.expected_overlap})")
    by_field = stability.mismatches_by_field
    for field in STABILITY_FIELDS:
        lines.append(f"  {field:<20} checked={stability.checked.get(field, 0):>4}  mismatches={by_field.get(field, 0)}")
    lines.append(f"  TOTAL mismatches across fields: {len(stability.mismatches)}")
    if stability.mismatches:
        lines.append(
            "  a moved score means Metaculus re-resolved in place: re-read those questions and correct the "
            "prior round's tables before quoting them"
        )
    return lines


def _render_parity(parity: Sequence[ParityRow]) -> list[str]:
    lines = ["", "=== 5. parsed comment data on the overlapping records, before and after ==="]
    for row in parity:
        lines.append(
            f"  {row.field:<32} prior_nonempty={row.prior_nonempty:>4} now_nonempty={row.now_nonempty:>4} "
            f"lost={len(row.lost)}"
        )
        if row.lost:
            lines.append(f"    lost on: {list(row.lost)}")
    return lines


def render_report(
    coverage: CoverageAudit,
    diff: PriorDiff | None,
    stability: StabilityAudit | None,
    parity: Sequence[ParityRow] | None,
    *,
    max_worst_rows: int = DEFAULT_MAX_WORST_ROWS,
    prior_pull_known: bool = True,
) -> str:
    """Render the audit as text. Pure: no IO, no clock read."""
    lines = _render_coverage(coverage, prior_pull_known=prior_pull_known)
    lines.extend(_render_sub_questions(coverage))
    if diff is not None:
        lines.extend(_render_diff(diff, max_worst_rows))
    if stability is not None:
        lines.extend(_render_stability(stability))
    if parity is not None:
        lines.extend(_render_parity(parity))
    return "\n".join(lines)


def _load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def _resolve_checkpoint(explicit: str | None, records_path: Path, parser: argparse.ArgumentParser) -> Path:
    path = Path(explicit) if explicit else checkpoint_path_for(records_path)
    if not path.exists():
        parser.error(f"checkpoint {path} does not exist; pass --checkpoint explicitly")
    return path


def _prior_recordless(prior_records_path: Path, explicit_checkpoint: str | None) -> frozenset[int]:
    """Post ids the prior pull fetched and skipped, which that round's write-up classified."""
    path = Path(explicit_checkpoint) if explicit_checkpoint else checkpoint_path_for(prior_records_path)
    if not path.exists():
        logger.warning(
            f"prior checkpoint {path} does not exist: no post can classify as already-known, so every "
            f"non-annulled recordless post will read INVESTIGATE"
        )
        return frozenset()
    known = recordless_post_ids(_load_json(path), _load_json(prior_records_path))
    logger.info(f"prior pull {path.parent.name} fetched {len(known)} post(s) that emitted no record")
    return frozenset(known)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline coverage audit of a residual-round pull: recordless posts, prior-round diff, "
        "re-pull score stability and comment-parse parity."
    )
    parser.add_argument("--records", required=True, help="This round's emitted perf records JSON")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=f"The pull's raw post payloads (default: the --records path with {CHECKPOINT_SUFFIX})",
    )
    parser.add_argument(
        "--prior-records",
        default=None,
        help="The prior round's records JSON; without it checks 3-5 are skipped and nothing classifies as known",
    )
    parser.add_argument(
        "--prior-checkpoint",
        default=None,
        help=f"The prior pull's payloads (default: the --prior-records path with {CHECKPOINT_SUFFIX})",
    )
    parser.add_argument("--output", default=None, help="Optional path to write the new-since-prior records as JSON")
    parser.add_argument(
        "--max-worst-rows",
        type=int,
        default=DEFAULT_MAX_WORST_ROWS,
        help="New records listed in the worst-spot-peer preview (default: %(default)s)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    records_path = Path(args.records)
    checkpoint = _load_json(_resolve_checkpoint(args.checkpoint, records_path, parser))
    records = _load_json(records_path)

    prior_records_path = Path(args.prior_records) if args.prior_records else None
    prior_recordless = (
        _prior_recordless(prior_records_path, args.prior_checkpoint) if prior_records_path else frozenset()
    )
    coverage = audit_coverage(checkpoint, records, prior_recordless=prior_recordless)

    diff = stability = parity = None
    if prior_records_path is not None:
        prior_records = _load_json(prior_records_path)
        diff = diff_against_prior(records, prior_records)
        stability = audit_stability(records, prior_records)
        parity = audit_parse_parity(records, prior_records)

    print(
        render_report(
            coverage,
            diff,
            stability,
            parity,
            max_worst_rows=args.max_worst_rows,
            prior_pull_known=bool(prior_recordless),
        )
    )

    if args.output and diff is not None:
        Path(args.output).write_text(json.dumps(list(diff.added), indent=2, default=str))
        logger.info(f"Wrote {len(diff.added)} new-since-prior record(s) to {args.output}")
    elif args.output:
        logger.warning("--output needs --prior-records: there is no new-since-prior cohort without a prior pull")


if __name__ == "__main__":
    main()
