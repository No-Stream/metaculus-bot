"""The four files a residual round writes, and the console report that goes with them.

``perf_all_tagged.json`` is the dataset every downstream lane reads; ``new_since_prior.json`` is
the slimmed view the rank and dossier lanes take; ``counts_by_era.json`` carries the era grids,
the boundary table and the round's stability and healing audit; ``degraded_cohort.json`` carries
the dry-key cohort's trail. Key order inside each payload is load-bearing, because rounds are
compared byte-for-byte. What each block means: ``docs/performance_analysis.md``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from metaculus_bot.performance_analysis.eras import (
    BOUNDARIES_UTC,
    ERAS,
    FINE_SUBERAS,
    POST_TIME_BUDGET_FINE,
    TRIPLE_SUBERAS,
)
from metaculus_bot.performance_analysis.platform_scores import peer_score, spot_peer_score
from metaculus_bot.performance_analysis.round_dataset import (
    DEGRADED_FULL_QUESTION_IDS,
    DEGRADED_PARTIAL_QUESTION_IDS,
    KNOWN_BUG_QUESTION_IDS,
    RecordKey,
    RoundDataset,
    is_scored,
    record_key,
    sorted_question_ids,
)
from metaculus_bot.performance_analysis.scaling import NUMERIC_TYPES
from metaculus_bot.time_utils import parse_iso_utc

logger: logging.Logger = logging.getLogger(__name__)

# The grid's type axis. "?" holds a record whose type is missing, so it cannot vanish from a count.
GRID_TYPES: tuple[str, ...] = ("binary", "multiple_choice", "numeric", "discrete", "?")
SCORED_ERAS: tuple[str, ...] = ("pre_flip", "post_flip", "triple_era")
# Below this the current era's claims need an effective-n and cluster-structure caveat.
SMALL_TRIPLE_ERA_N = 60
WORST_ROWS = 15
BEST_ROWS = 8


@dataclass(frozen=True)
class Grids:
    """The scored/total counts every era table in ``counts_by_era.json`` is rendered from."""

    era: dict[tuple[str, str], Counter]
    triple_subera: dict[tuple[str, str], Counter]
    triple_subera_fine: dict[tuple[str, str], Counter]
    per_tournament: dict[tuple[str, str, str], Counter]
    submit_range: dict[str, list]


def count_grids(records: list[dict]) -> Grids:
    era_grid: dict[tuple[str, str], Counter] = defaultdict(Counter)
    subera_grid: dict[tuple[str, str], Counter] = defaultdict(Counter)
    fine_grid: dict[tuple[str, str], Counter] = defaultdict(Counter)
    slug_grid: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    submit_range: dict[str, list] = defaultdict(lambda: [None, None])
    for record in records:
        slug, era = record["source_tournament"], record["config_era"]
        qtype = record.get("type") or "?"
        scored = is_scored(record)
        for grid, row in (
            (era_grid, era),
            (subera_grid, record.get("triple_subera") or "n/a"),
            (fine_grid, record.get("triple_subera_fine") or "n/a"),
        ):
            grid[(row, qtype)]["total"] += 1
            if scored:
                grid[(row, qtype)]["scored"] += 1
        slug_grid[(slug, era, qtype)]["total"] += 1
        if scored:
            slug_grid[(slug, era, qtype)]["scored"] += 1
        submitted = parse_iso_utc(record.get("bot_comment_created_at"))
        if submitted is not None:
            low, high = submit_range[slug]
            submit_range[slug] = [
                submitted if low is None or submitted < low else low,
                submitted if high is None or submitted > high else high,
            ]
    return Grids(
        era=era_grid,
        triple_subera=subera_grid,
        triple_subera_fine=fine_grid,
        per_tournament=slug_grid,
        submit_range=submit_range,
    )


def per_model_summary(record: dict) -> dict:
    """Compact per-model view: which models forecast, and their raw parsed values."""
    per_model = record.get("per_model_forecasts") or {}
    numeric = record.get("per_model_numeric_percentiles") or {}
    return {
        "n_models": len(set(per_model) | set(numeric)),
        "models": sorted(set(per_model) | set(numeric)),
        "forecasters_used": record.get("forecasters_used"),
        "forecasters_configured": record.get("forecasters_configured"),
        "values": dict(per_model),
        "has_numeric_percentiles": sorted(numeric),
    }


def slim(record: dict, prior_index: dict[RecordKey, dict]) -> dict:
    """The rank and dossier lanes' view of a record: scores, tags, per-model values, prior view."""
    scores = record.get("metaculus_scores") or {}
    metadata = record.get("metadata") or {}
    spot, weight = scores.get("spot_peer_score"), record.get("question_weight")
    return {
        "question_id": record.get("question_id"),
        "post_id": record.get("post_id"),
        "title": record.get("title"),
        "type": record.get("type"),
        "config_era": record.get("config_era"),
        "triple_subera": record.get("triple_subera"),
        "triple_subera_fine": record.get("triple_subera_fine"),
        "ft_unfreeze_side": record.get("ft_unfreeze_side"),
        "post_linters_merge": record.get("post_linters_merge"),
        "source_tournament": record.get("source_tournament"),
        "degraded_run": record.get("degraded_run"),
        "partial_degraded": record.get("partial_degraded"),
        "known_bug": record.get("known_bug"),
        "bot_comment_created_at": record.get("bot_comment_created_at"),
        "resolution_raw": record.get("resolution_raw"),
        "resolution_parsed": record.get("resolution_parsed"),
        "our_prob_yes": record.get("our_prob_yes"),
        "options": record.get("options"),
        "our_forecast_values": record.get("our_forecast_values"),
        "open_lower_bound": record.get("open_lower_bound"),
        "open_upper_bound": record.get("open_upper_bound"),
        "scaling": record.get("scaling"),
        "log_score": record.get("log_score"),
        "mc_log_score": record.get("mc_log_score"),
        "numeric_log_score": record.get("numeric_log_score"),
        "brier_score": record.get("brier_score"),
        "peer_score": scores.get("peer_score"),
        "spot_peer_score": spot,
        "baseline_score": scores.get("baseline_score"),
        "spot_baseline_score": scores.get("spot_baseline_score"),
        "coverage": scores.get("coverage"),
        "question_weight": weight,
        "weighted_spot_peer": None if spot is None or weight is None else spot * weight,
        "was_stacked": record.get("was_stacked"),
        "stacker_outcome": record.get("stacker_outcome"),
        "stacker_skip_reason": record.get("stacker_skip_reason"),
        "per_model": per_model_summary(record),
        "nr_forecasters_crowd": metadata.get("nr_forecasters"),
        "resolution_set_time": metadata.get("resolution_set_time"),
        "actual_resolve_time": metadata.get("actual_resolve_time"),
        "scheduled_resolve_time": metadata.get("scheduled_resolve_time"),
        "is_new_since_prior": record.get("is_new_since_prior"),
        "newly_scored": record.get("newly_scored"),
        "rescored_fields": record.get("rescored_fields"),
        "rescored_fields_this_round": record.get("rescored_fields_this_round"),
        "platform_rescored": record.get("platform_rescored"),
        "platform_rescored_this_round": record.get("platform_rescored_this_round"),
        "platform_rescored_pull_tag": record.get("platform_rescored_pull_tag"),
        "prior_round_view": prior_index.get(record_key(record)),
    }


def _mean_median(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return sum(ordered) / n, median


def atomic_dump(payload: object, path: Path, indent: int | None) -> None:
    """Write JSON to a temp file and rename it, so a concurrent reader never sees a partial file."""
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=indent, default=str)
    os.replace(tmp, path)


def _log_grid(title: str, grid: dict, row_keys: tuple[str, ...], label: str) -> None:
    logger.info(f"=== {title} ===")
    for row in row_keys:
        totals = {qtype: grid[(row, qtype)]["total"] for qtype in GRID_TYPES}
        if not any(totals.values()):
            logger.info(f"  {label}={row}: (empty, zero records)")
            continue
        logger.info(f"  {label}={row}:")
        for qtype in GRID_TYPES:
            if totals[qtype] > 0:
                counts = grid[(row, qtype)]
                logger.info(f"      {qtype:<16} scored={counts['scored']:>4}  total={counts['total']:>4}")


def _score_row(record: dict) -> str:
    tags = []
    if record["degraded_run"]:
        tags.append("DEGRADED_FULL_1of3")
    if record["partial_degraded"]:
        tags.append("PARTIAL_2of3")
    if record["known_bug"]:
        tags.append("KNOWN_BUG")
    weight, spot = record.get("question_weight"), spot_peer_score(record)
    weighted = "" if weight is None or spot is None else f" w={weight:.2f} wspot={spot * weight:>8.2f}"
    return (
        f"  spot={spot:>8.2f} peer={peer_score(record):>8.2f}{weighted}  qid={record.get('question_id')} "
        f"pid={record.get('post_id')} era={record['config_era']:<10} type={record.get('type'):<15} "
        f"{'[' + ','.join(tags) + ']' if tags else '':<22} :: {(record.get('title') or '')[:60]}"
    )


def _log_novelty(records: list[dict], new_union_keys: set[RecordKey], prior_label: str) -> None:
    new_records = [r for r in records if r["is_new_since_prior"]]
    newly_scored = [r for r in records if r["newly_scored"]]
    logger.info(f"  is_new_since_prior (absent from the {prior_label} file): {len(new_records)}")
    logger.info(f"  newly_scored (scored now, not scored on {prior_label}): {len(newly_scored)}")
    logger.info(f"  union (new OR newly scored): {len(new_union_keys)}")
    logger.info(f"  new by type: {Counter(r.get('type') for r in new_records).most_common()}")
    logger.info(f"  new by era: {Counter(r.get('config_era') for r in new_records).most_common()}")
    logger.info(
        f"  new by triple_subera_fine: {Counter(r.get('triple_subera_fine') for r in new_records).most_common()}"
    )


def _log_cohort_hits(records: list[dict], flagged: list[dict]) -> None:
    logger.info("=== degraded / partial-degraded records present in the resolved dataset ===")
    for record in sorted(flagged, key=lambda r: r.get("bot_comment_created_at") or ""):
        kind = "FULL_1of3" if record["degraded_run"] else "PARTIAL_2of3"
        logger.info(
            f"  [{kind}] qid={record.get('question_id')} pid={record.get('post_id')} type={record.get('type')} "
            f"scored={is_scored(record)} spot={spot_peer_score(record)} peer={peer_score(record)} "
            f"new={record['is_new_since_prior']} submitted={record.get('bot_comment_created_at')} "
            f":: {(record.get('title') or '')[:55]}"
        )
    if not flagged:
        logger.info("  none of the degraded-run questions have resolved yet")

    logger.info(f"=== known_bug cohort present (imported KNOWN_BUG_QIDS = {sorted(KNOWN_BUG_QUESTION_IDS)}) ===")
    for record in [r for r in records if r["known_bug"]]:
        logger.info(
            f"  qid={record.get('question_id')} era={record['config_era']} type={record.get('type')} "
            f"spot={spot_peer_score(record)} weight={record.get('question_weight')} "
            f"new={record['is_new_since_prior']} :: {(record.get('title') or '')[:55]}"
        )


def _log_grids(grids: Grids) -> None:
    logger.info("=== per-tournament submit-time ranges ===")
    for slug in sorted(grids.submit_range):
        low, high = grids.submit_range[slug]
        logger.info(f"  {slug}: {low} .. {high}")

    logger.info("=== per-tournament x era x type (scored / total) ===")
    for slug in sorted({key[0] for key in grids.per_tournament}):
        logger.info(f"  {slug}:")
        for era in ERAS:
            present = [t for t in GRID_TYPES if grids.per_tournament[(slug, era, t)]["total"] > 0]
            if not present:
                continue
            logger.info(f"    {era}:")
            for qtype in present:
                counts = grids.per_tournament[(slug, era, qtype)]
                logger.info(f"        {qtype:<16} scored={counts['scored']:>4}  total={counts['total']:>4}")

    _log_grid("ERA x TYPE rollup (all tournaments)", grids.era, ERAS, "era")
    _log_grid("TRIPLE SUB-ERA x TYPE", grids.triple_subera, TRIPLE_SUBERAS, "subera")
    _log_grid("TRIPLE FINE SUB-ERA x TYPE", grids.triple_subera_fine, FINE_SUBERAS, "fine")

    logger.info("=== LOAD-BEARING: per-type SCORED counts by era ===")
    for era in SCORED_ERAS:
        row = {qtype: grids.era[(era, qtype)]["scored"] for qtype in GRID_TYPES}
        numeric_n = sum(row[qtype] for qtype in NUMERIC_TYPES)
        logger.info(
            f"  {era}: binary={row['binary']}  mc={row['multiple_choice']}  numeric={row['numeric']}  "
            f"discrete={row['discrete']}  numeric+discrete={numeric_n}"
        )


@dataclass(frozen=True)
class TripleEraCut:
    """The current-era cohorts every headline claim is quoted against."""

    total: list[dict]
    scored: list[dict]
    clean: list[dict]
    clean_excl_partial: list[dict]
    clean_numeric: list[dict]
    post_time_budget_scored: list[dict]
    newest_submission: str


def triple_era_cut(records: list[dict]) -> TripleEraCut:
    triple = [r for r in records if r["config_era"] == "triple_era"]
    scored = [r for r in triple if is_scored(r)]
    clean = [r for r in scored if not r["degraded_run"] and not r["known_bug"]]
    return TripleEraCut(
        total=triple,
        scored=scored,
        clean=clean,
        clean_excl_partial=[r for r in clean if not r["partial_degraded"]],
        clean_numeric=[r for r in clean if (r.get("type") or "") in NUMERIC_TYPES],
        post_time_budget_scored=[r for r in scored if r.get("triple_subera_fine") in POST_TIME_BUDGET_FINE],
        newest_submission=max((r.get("bot_comment_created_at") or "") for r in records),
    )


def _log_triple_era(cut: TripleEraCut) -> None:
    logger.info(
        f"  triple_era: total={len(cut.total)} scored={len(cut.scored)} clean_scored={len(cut.clean)} "
        f"clean_scored_excl_partial={len(cut.clean_excl_partial)} clean_numeric={len(cut.clean_numeric)}"
    )
    logger.info(f"  triple_subera (scored): {Counter(r.get('triple_subera') for r in cut.scored).most_common()}")
    logger.info(
        f"  triple_subera_fine (scored): {Counter(r.get('triple_subera_fine') for r in cut.scored).most_common()}"
    )
    logger.info(
        f"  time_budget-or-later sub-eras scored n={len(cut.post_time_budget_scored)}; newest bot submission "
        f"in the whole dataset: {cut.newest_submission}"
    )
    if cut.post_time_budget_scored:
        logger.info(
            "  a dossier on one of these MAY cite the time-budget-or-later config as live; everything "
            "in an earlier sub-era is counterfactual-only:"
        )
        for record in sorted(cut.post_time_budget_scored, key=lambda r: r.get("bot_comment_created_at") or ""):
            logger.info(
                f"      qid={record.get('question_id')} pid={record.get('post_id')} type={record.get('type')} "
                f"fine={record.get('triple_subera_fine')} submitted={record.get('bot_comment_created_at')} "
                f"spot={spot_peer_score(record)} peer={peer_score(record)} :: {(record.get('title') or '')[:55]}"
            )
    else:
        logger.info(
            "  every resolved forecast predates the time-budget merge, so those features stay "
            "counterfactual-only, and an empty bucket is the honest answer"
        )
    if len(cut.clean) < SMALL_TRIPLE_ERA_N:
        logger.info(
            f"  NOTE: triple_era clean scored n={len(cut.clean)}, below {SMALL_TRIPLE_ERA_N}. Quote effective "
            "n and cluster structure with every current-era claim."
        )


def _log_stacker_sanity(records: list[dict]) -> None:
    stacked_late = [
        r
        for r in records
        if r["config_era"] == "triple_era"
        and isinstance(r.get("stacker_outcome"), str)
        and "stack" in r["stacker_outcome"].lower()
        and "no" not in r["stacker_outcome"].lower()
    ]
    if stacked_late:
        logger.warning(f"{len(stacked_late)} triple_era records show a STACKED outcome, which is unexpected:")
        for record in stacked_late:
            logger.warning(
                f"      qid={record.get('question_id')} stacker_outcome={record.get('stacker_outcome')!r} "
                f":: {(record.get('title') or '')[:55]}"
            )
    else:
        logger.info(
            "  stacker sanity: no triple_era record shows a stacked outcome, as expected; the six "
            "pre-2026-05-30 post_flip stacked records are known and legitimate."
        )


def new_wave_summary(records: list[dict], new_union_keys: set[RecordKey]) -> tuple[list[dict], dict[str, dict]]:
    """The new wave ranked on SPOT peer, with coverage-scaled peer as the labelled secondary."""
    new_union = [r for r in records if record_key(r) in new_union_keys and spot_peer_score(r) is not None]
    n_no_spot = sum(
        1
        for r in records
        if record_key(r) in new_union_keys and spot_peer_score(r) is None and peer_score(r) is not None
    )
    logger.info(f"  new records with peer but NO spot peer (would fall to the peer-only tier): {n_no_spot}")
    new_union.sort(key=lambda r: spot_peer_score(r) or 0.0)  # list pre-filtered to a non-None spot peer

    logger.info(f"=== worst {WORST_ROWS} NEW records by SPOT peer (coverage-scaled peer as secondary) ===")
    for record in new_union[:WORST_ROWS]:
        logger.info(_score_row(record))
    logger.info(f"=== best {BEST_ROWS} NEW records by SPOT peer ===")
    for record in list(reversed(new_union))[:BEST_ROWS]:
        logger.info(_score_row(record))

    ex_cohort = [r for r in new_union if not r["degraded_run"] and not r["partial_degraded"] and not r["known_bug"]]
    summary: dict[str, dict] = {}
    for label, rows in (
        ("all", new_union),
        ("strict (no degraded/partial/known_bug)", ex_cohort),
        ("new post_flip only", [r for r in new_union if r["config_era"] == "post_flip"]),
        ("new triple_era only", [r for r in new_union if r["config_era"] == "triple_era"]),
        ("new pre_flip only", [r for r in new_union if r["config_era"] == "pre_flip"]),
    ):
        if not rows:
            continue
        spots = [spot for row in rows if (spot := spot_peer_score(row)) is not None]
        peers = [peer for row in rows if (peer := peer_score(row)) is not None]
        wspots = [
            spot * row["question_weight"]
            for row in rows
            if row.get("question_weight") is not None and (spot := spot_peer_score(row)) is not None
        ]
        spot_mean, spot_median = _mean_median(spots)
        peer_mean, peer_median = _mean_median(peers) if peers else (float("nan"), float("nan"))
        weighted_mean = _mean_median(wspots)[0] if wspots else float("nan")
        logger.info(
            f"  new-wave {label}: n={len(rows)}  SPOT mean={spot_mean:+.2f} median={spot_median:+.2f} "
            f"min={min(spots):+.2f} max={max(spots):+.2f} n_neg={sum(1 for v in spots if v < 0)}"
            f"  |  peer (secondary) mean={peer_mean:+.2f} median={peer_median:+.2f}"
            f"  |  weight-adjusted spot mean={weighted_mean:+.2f}"
        )
        summary[label] = {
            "n": len(rows),
            "spot_mean": spot_mean,
            "spot_median": spot_median,
            "spot_min": min(spots),
            "spot_max": max(spots),
            "spot_n_negative": sum(1 for v in spots if v < 0),
            "peer_mean": peer_mean,
            "peer_median": peer_median,
            "weighted_spot_mean": weighted_mean,
            "weighted_spot_sum": sum(wspots) if wspots else None,
        }
    return new_union, summary


def counts_payload(
    dataset: RoundDataset,
    *,
    grids: Grids,
    cut: TripleEraCut,
    new_union_keys: set[RecordKey],
    wave_summary: dict[str, dict],
) -> dict:
    """Everything ``counts_by_era.json`` records about this round, in its published key order."""
    records = dataset.records
    weighted = [r for r in records if r["question_weight"] is not None]
    downweighted = [r for r in weighted if r["question_weight"] < 1.0]
    pull = dataset.pull_payload
    return {
        "generated": datetime.now(UTC).isoformat(),
        "boundaries_utc": BOUNDARIES_UTC,
        "era_by_type": {
            era: {t: dict(grids.era[(era, t)]) for t in GRID_TYPES if grids.era[(era, t)]["total"] > 0} for era in ERAS
        },
        "triple_subera_by_type": {
            sub: {
                t: dict(grids.triple_subera[(sub, t)]) for t in GRID_TYPES if grids.triple_subera[(sub, t)]["total"] > 0
            }
            for sub in TRIPLE_SUBERAS
        },
        "triple_subera_fine_by_type": {
            sub: {
                t: dict(grids.triple_subera_fine[(sub, t)])
                for t in GRID_TYPES
                if grids.triple_subera_fine[(sub, t)]["total"] > 0
            }
            for sub in FINE_SUBERAS
        },
        "per_tournament": {
            f"{slug}|{era}|{qtype}": dict(counts) for (slug, era, qtype), counts in sorted(grids.per_tournament.items())
        },
        "totals": {
            "records": len(records),
            "scored": sum(1 for r in records if is_scored(r)),
            "new_since_prior": sum(1 for r in records if r["is_new_since_prior"]),
            "newly_scored": sum(1 for r in records if r["newly_scored"]),
            "new_union": len(new_union_keys),
            "triple_era_total": len(cut.total),
            "triple_era_scored": len(cut.scored),
            "triple_era_clean_scored": len(cut.clean),
            "triple_era_clean_scored_excl_partial": len(cut.clean_excl_partial),
            "triple_era_clean_numeric": len(cut.clean_numeric),
            "post_time_budget_subera_scored": len(cut.post_time_budget_scored),
            "newest_bot_comment_created_at": cut.newest_submission,
        },
        "repull_stability": {
            "bot_log_score_drift": len(dataset.bot_score_drift),
            "metaculus_platform_score_drift_fields": len(dataset.platform_score_drift),
            "bot_log_score_drift_detail": dataset.bot_score_drift,
            "platform_drift_detail": dataset.platform_score_drift,
            "pull_side_diff": {
                "compared": pull.get("compared"),
                "rescored_records": pull.get("rescored_records"),
                "unmatched_no_prior_counterpart": pull.get("unmatched_no_prior_counterpart"),
                "changed_fields": len(pull.get("changes") or []),
                "tag_distribution": pull.get("tag_distribution"),
            },
            "cumulative_platform_rescored_question_ids": sorted_question_ids(
                r for r in records if r.get("platform_rescored")
            ),
        },
        "question_weights": {
            "note": (
                "The tournament leaderboard totals spot_peer_score * question_weight. Weights are keyed "
                "by question id in question_weights.json and stamped on every summer record; records "
                "from other tournaments carry None because no weight was pulled for them."
            ),
            "summer_records_weighted": len(weighted),
            "summer_records_downweighted": len(downweighted),
            "downweight_distribution": {
                str(weight): n for weight, n in Counter(r["question_weight"] for r in downweighted).most_common()
            },
        },
        "new_wave_summary_spot_primary": wave_summary,
        "score_healing": {
            "note": (
                "collector.rescore_records recomputes scores from stored inputs; it heals the "
                "pre-grid_zero_point numeric_log_score values baked into reused baselines."
            ),
            "records_changed_this_round": len(dataset.healing_changes),
            "detail_this_round": dataset.healing_changes,
            "cumulative_rescored_question_ids": sorted_question_ids(r for r in records if r.get("rescored_fields")),
            "provenance_note": (
                "rescored_fields is cumulative across rounds (carried from the prior tagged file, "
                "unioned with this round's deltas); rescored_fields_this_round is this round only. "
                "platform_rescored follows the same three-field shape for Metaculus-side changes, and "
                "platform_rescored_pull_tag carries rescore_diff's separate ternary."
            ),
        },
    }


def degraded_payload(dataset: RoundDataset, flagged: list[dict]) -> dict:
    degraded = dataset.degraded
    return {
        "runs": degraded.runs,
        "telemetry_degraded_question_ids": sorted(degraded.telemetry_question_ids),
        "canonical_full_degraded_question_ids": sorted(DEGRADED_FULL_QUESTION_IDS),
        "canonical_partial_degraded_question_ids": sorted(DEGRADED_PARTIAL_QUESTION_IDS),
        "union_used_for_tagging": sorted(degraded.union_question_ids),
        "resolved_hits": [slim(r, dataset.prior_index) for r in flagged],
    }


def write_round_outputs(dataset: RoundDataset) -> None:
    """Log the round's report and write the four files, all under the round's own directory."""
    spec, records = dataset.spec, dataset.records
    new_union_keys = {record_key(r) for r in records if r["is_new_since_prior"] or r["newly_scored"]}

    _log_novelty(records, new_union_keys, spec.prior_label)
    flagged = [r for r in records if r["degraded_run"] or r["partial_degraded"]]
    _log_cohort_hits(records, flagged)

    grids = count_grids(records)
    _log_grids(grids)
    cut = triple_era_cut(records)
    _log_triple_era(cut)
    _log_stacker_sanity(records)
    _, wave_summary = new_wave_summary(records, new_union_keys)

    tagged_path = spec.round_dir / "perf_all_tagged.json"
    atomic_dump(records, tagged_path, indent=None)
    logger.info(f"wrote {len(records)} tagged records -> {tagged_path}")

    new_out = [
        slim(r, dataset.prior_index) for r in records if record_key(r) in new_union_keys or r.get("platform_rescored")
    ]
    n_carried = sum(
        1 for r in new_out if r["platform_rescored"] and not r["is_new_since_prior"] and not r["newly_scored"]
    )
    logger.info(
        f"  (new_since_prior.json also carries {n_carried} record(s) with cumulative platform_rescored "
        "provenance that are NOT new; check the flags)"
    )
    new_out.sort(key=lambda r: r.get("bot_comment_created_at") or "")
    new_path = spec.round_dir / "new_since_prior.json"
    atomic_dump(new_out, new_path, indent=1)
    logger.info(f"wrote {len(new_out)} new/newly-scored/rescored records -> {new_path}")

    counts_path = spec.round_dir / "counts_by_era.json"
    payload = counts_payload(dataset, grids=grids, cut=cut, new_union_keys=new_union_keys, wave_summary=wave_summary)
    atomic_dump(payload, counts_path, indent=1)
    logger.info(f"wrote era x type counts -> {counts_path}")

    audit_path = spec.round_dir / "degraded_cohort.json"
    atomic_dump(degraded_payload(dataset, flagged), audit_path, indent=1)
    logger.info(f"wrote degraded-cohort audit -> {audit_path}")
