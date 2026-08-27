"""CLI orchestrator for the probabilistic-tools ablation benchmark.

Wave-4 integration layer. Pulls together every Wave 1-3 module under one
argparse-driven entry point. The behavioral spec lives in
``scratch_docs_and_planning/`` (Atlas-inspired ablation plan). Highlights:

* Stage-by-stage pipeline: fetch → research → screen → forecast → stack → pdf → median → score.
* Per-stage disk cache: every artifact lives under ``backtests/ablation/`` and a
  re-run is a no-op for cached qids.
* Smoke→expand semantics: re-running with larger ``--num-*`` counts fetches the
  delta only; the manifest is append-extendable.
* ``--qa-research`` halts after the screen stage, dumps a markdown QA file.
* ``--stages`` selects a subset (default all); ``--force-stages`` re-runs the
  listed stages bypassing their cache reads.
* Each stage reads cache where prerequisites permit; missing prerequisites for
  ``--stages score`` exit non-zero with a clear message.

Module-level imports are deliberate: tests monkeypatch the wave-1/2/3 entry
points at ``metaculus_bot.ablation.cli.<name>``. Keep the imports below stable.

That patch surface is what fixes this module's boundary. A monkeypatch of
``metaculus_bot.ablation.cli.screen_batch`` only reaches a call site in *this*
module, so every stage that invokes one of those entry points stays here, alongside
``run_ablation``, the per-stage narration, and the argparse entry point. The stages
that call nothing patchable, and every supporting layer, live next door:

* ``ablation.cli_args`` — argparse, the stage token list + force cascade, the
  rate-limit dial, and the ``_StagePlan`` / ``_ArmStage`` plan types.
* ``ablation.manifest_serde`` — qids-manifest serde and question rehydration.
* ``ablation.run_state`` — ``WorkingSet`` / ``SpendReport``, their cache hydration,
  the cached-vs-fresh partition helpers, and the ``--qids`` filters.
* ``ablation.reporting`` — the end-of-run spend block and the ``--qa-research`` dump.
* ``ablation.deterministic_arms`` — the mean / median / PDF arms (no LLM calls).
* ``ablation.score_stage`` — arm label sets, report shims, paired scoring, artifacts.
* ``ablation.stacker_selection`` — which stacker model and cache slug a run uses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from metaculus_bot.ablation.cache import AblationCache, atomic_write_text, model_slug_to_filename
from metaculus_bot.ablation.cli_args import (
    _ARM_STAGES,
    DEFAULT_CACHE_DIR,
    DEFAULT_TOURNAMENTS,
    STAGES,  # noqa: F401  # re-export: tests read the stage vocabulary off this module
    _build_parser,
    _expand_forced_stages,
    _rate_limit_mode_kwargs,
    _StagePlan,
)
from metaculus_bot.ablation.deterministic_arms import _stage_pdf, _stage_simple_agg

# ``get_lineup`` was function-scoped in ``_stage_forecast`` so a partial edit couldn't leave
# the formatter stripping it as unused. Top-level is safe: this module already imports
# ``forecaster_lineup``, and nothing patches ``get_lineup`` at its own module, so the
# binding cannot go stale under a monkeypatch.
from metaculus_bot.ablation.forecaster_lineup import FREE_FORECASTER_MODELS, get_lineup
from metaculus_bot.ablation.forecasters import run_forecasters_batch
from metaculus_bot.ablation.leakage_screen import _EMPTY_BLOB_RESPONSE, screen_batch
from metaculus_bot.ablation.manifest_serde import (
    _build_manifest_entry,
    _build_question_shim_from_manifest_entry,
    _deserialize_ground_truth,
    _serialize_ground_truth,  # noqa: F401  # re-export: manifest-schema tests import it from here
    _serialize_question_metadata,  # noqa: F401  # re-export: manifest-schema tests import it from here
)
from metaculus_bot.ablation.prune import DEFAULT_BATCH_SIZE as PRUNE_DEFAULT_BATCH_SIZE
from metaculus_bot.ablation.prune import run_prune_for_qids
from metaculus_bot.ablation.qa_iterate import (
    IterateOutcome,
    read_manual_rejects,
    render_qa_summary,
    run_qa_iterate_batch,
    serialize_outcome,
    write_manual_rejects,
)
from metaculus_bot.ablation.reporting import _print_spend_report, _stage_qa_research_dump
from metaculus_bot.ablation.research import run_gemini_research_for_qids
from metaculus_bot.ablation.run_stacker import (
    ABLATION_MIN_FORECASTERS,
    ARM_MEAN,
    ARM_MEDIAN,
    ARM_PDF,
    run_stacker_batch,
)
from metaculus_bot.ablation.run_state import (
    SpendReport,
    WorkingSet,
    _apply_qids_filter,
    _hydrate_working_set_from_cache,
    _partition_screen_cache,
    _partition_stacker_cache,
    _qids_with_any_arm_payload,
    _record_stacker_spend,
)
from metaculus_bot.ablation.score_stage import (
    _build_report_shim,  # noqa: F401  # re-export: report-shim tests import it from here
    _stage_score,
)
from metaculus_bot.ablation.stacker_selection import _active_stacker_slug, _stacker_batch_kwargs
from metaculus_bot.aiohttp_cleanup import enable_aiohttp_session_autoclose
from metaculus_bot.api_preflight import verify_metaculus_api_identity
from metaculus_bot.backtest.question_prep import (
    BacktestQuestionSet,
    fetch_resolved_questions_stratified,
)
from metaculus_bot.backtest.scoring import GroundTruth

logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Question loaders
# ---------------------------------------------------------------------------


async def load_questions_by_qids(qids: list[int]) -> tuple[list[Any], dict[int, GroundTruth]]:
    """Fetch a fixed list of qids individually. Tests monkey-patch this.

    Production path: not implemented yet (real Metaculus per-qid fetch). The
    ablation runner only uses ``--qids`` for cached re-runs, where the manifest
    already has the data; tests cover the bypass-fetch path.
    """
    await asyncio.sleep(0)
    raise NotImplementedError(
        "load_questions_by_qids: production path needs MetaculusApi.get_question_by_post_id wiring; "
        "tests monkey-patch this entry point."
    )


# ---------------------------------------------------------------------------
# Fetch stage
# ---------------------------------------------------------------------------


async def _stage_fetch(args: argparse.Namespace, cache: AblationCache, working: WorkingSet) -> None:
    """Populate the working set with questions + ground truths + manifest.

    Behavior:
    - ``args.qids`` provided → call ``load_questions_by_qids``; do not fetch from tournaments.
    - Else → read existing manifest; fetch ADDITIONAL questions per type if requested counts
      exceed existing counts, excluding already-known qids.
    """
    await asyncio.sleep(0)
    if args.qids:
        await _fetch_explicit_qids(args, cache, working)
    else:
        await _fetch_tournament_delta(args, cache, working)


async def _fetch_explicit_qids(args: argparse.Namespace, cache: AblationCache, working: WorkingSet) -> None:
    """Hydrate ``args.qids`` from the manifest, fetching only the ones missing from it."""
    existing = cache.read_qids_manifest()
    # If every qid already in manifest, no fetch needed — load shims from manifest.
    if all(qid in existing for qid in args.qids):
        for qid in args.qids:
            entry = existing[qid]
            question = _build_question_shim_from_manifest_entry(qid, entry)
            working.questions[qid] = question
            working.ground_truths[qid] = _deserialize_ground_truth(entry["ground_truth"])
        return

    # Some qids missing from manifest: fetch them via the per-qid loader.
    questions, ground_truths = await load_questions_by_qids(args.qids)
    new_entries: dict[int, dict] = {}
    tournament = args.tournaments[0] if args.tournaments else DEFAULT_TOURNAMENTS[0]
    for question in questions:
        qid = question.id_of_question
        gt = ground_truths[qid]
        working.questions[qid] = question
        working.ground_truths[qid] = gt
        new_entries[qid] = _build_manifest_entry(question, gt, tournament)
    cache.append_qids_manifest(new_entries)


async def _fetch_tournament_delta(args: argparse.Namespace, cache: AblationCache, working: WorkingSet) -> None:
    """Tournament fetch with append-extend semantics.

    Hydrates every already-known qid from the manifest, then fetches only the per-type
    shortfall against the requested counts, dropping anything the manifest already has.
    """
    await asyncio.sleep(0)
    existing = cache.read_qids_manifest()
    existing_per_type = {"binary": 0, "multiple_choice": 0, "numeric": 0}
    for entry in existing.values():
        # Manifest entries always have ``type``; direct subscript surfaces drift.
        qtype = entry["type"]
        if qtype in existing_per_type:
            existing_per_type[qtype] += 1
        # Hydrate working set from manifest for already-known qids.
        qid = int(entry["ground_truth"]["question_id"])
        question = _build_question_shim_from_manifest_entry(qid, entry)
        working.questions[qid] = question
        working.ground_truths[qid] = _deserialize_ground_truth(entry["ground_truth"])

    delta_binary = max(0, args.num_binary - existing_per_type["binary"])
    delta_mc = max(0, args.num_multiple_choice - existing_per_type["multiple_choice"])
    delta_numeric = max(0, args.num_numeric - existing_per_type["numeric"])

    if delta_binary == 0 and delta_mc == 0 and delta_numeric == 0:
        logger.info(
            "fetch | manifest already saturated for requested counts | per-type=%s",
            existing_per_type,
        )
        return

    logger.info(
        "fetch | requesting delta | binary=%d mc=%d numeric=%d (existing=%s)",
        delta_binary,
        delta_mc,
        delta_numeric,
        existing_per_type,
    )

    # Confirm the host is the real Metaculus before the token-sending fetch.
    verify_metaculus_api_identity()

    question_set: BacktestQuestionSet = await fetch_resolved_questions_stratified(
        num_binary=delta_binary,
        num_multiple_choice=delta_mc,
        num_numeric=delta_numeric,
        resolved_after=args.resolved_after,
        resolved_before=args.resolved_before,
        tournaments=args.tournaments,
    )

    # Drop any qids already in the manifest (the fetcher doesn't have an exclude-list arg).
    # ``fetch_resolved_questions_stratified`` guarantees ``ground_truths`` covers every
    # qid in ``questions`` — direct subscript surfaces a fetcher invariant violation.
    new_entries = {}
    tournament = args.tournaments[0] if args.tournaments else DEFAULT_TOURNAMENTS[0]
    for question in question_set.questions:
        qid = question.id_of_question
        assert qid is not None
        if qid in existing:
            continue
        gt = question_set.ground_truths[qid]
        working.questions[qid] = question
        working.ground_truths[qid] = gt
        new_entries[qid] = _build_manifest_entry(question, gt, tournament)

    if new_entries:
        cache.append_qids_manifest(new_entries)


# ---------------------------------------------------------------------------
# Research stage
# ---------------------------------------------------------------------------


async def _stage_research(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    force: bool,
    spend: SpendReport,
) -> None:
    """Populate working.research_blobs from cache + live runner.

    Spend counters: a qid cached at stage entry (and not forced) increments
    ``cached_research_hits``. A fresh run for a qid increments
    ``gemini_research_calls`` (whether or not it succeeded — the API was hit)
    and, on success, adds ``meta["gap_count"]`` to ``gemini_gap_fill_calls``
    when ``meta["gap_fill_used"]`` is True.
    """
    await asyncio.sleep(0)
    qids = sorted(working.questions.keys())
    cached_blobs: dict[int, str] = {}
    if not force:
        for qid in qids:
            cached = cache.read_research(qid)
            if cached is not None:
                cached_blobs[qid] = cached[0]
                spend.cached_research_hits += 1

    questions_to_fetch = [working.questions[qid] for qid in qids if qid not in cached_blobs]

    fresh_results: dict[int, tuple[str, dict] | None] = {}
    if questions_to_fetch:
        fresh_results = await run_gemini_research_for_qids(
            questions_to_fetch,
            cache,
            gap_fill_max_gaps=args.gap_fill_max_gaps,
            is_benchmarking=True,
            force=force,
            concurrency=args.concurrency,
            gemini_model=args.gemini_model,
            enable_gap_fill=args.gap_fill,
        )

    for qid, blob in cached_blobs.items():
        working.research_blobs[qid] = blob
    for qid, result in fresh_results.items():
        spend.gemini_research_calls += 1
        if result is None:
            logger.warning("research | qid=%d failed; dropping from downstream", qid)
            continue
        blob, meta = result
        if meta.get("gap_fill_used"):
            spend.gemini_gap_fill_calls += int(meta.get("gap_count", 0))
        working.research_blobs[qid] = blob


# ---------------------------------------------------------------------------
# Prune stage
# ---------------------------------------------------------------------------


async def _stage_prune(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    force: bool,
    spend: SpendReport,
) -> None:
    """Run the redactor over ``working.research_blobs``; replace with sanitized blobs.

    After this stage, ``working.research_blobs[qid]`` holds the SANITIZED blob
    (the screen / forecast / stack stages always operate on sanitized blobs).
    Validation failures (subagent emitted a sanitized blob still containing the
    ground truth) drop the qid from ``working.research_blobs`` entirely; that
    qid then never reaches the screen or any downstream stage.
    """
    await asyncio.sleep(0)
    qids = sorted(working.research_blobs.keys())
    if not qids:
        return

    triples: list[tuple[Any, GroundTruth, str]] = []
    pre_cached_qids: set[int] = set()
    for qid in qids:
        question = working.questions[qid]
        gt = working.ground_truths[qid]
        raw_blob = working.research_blobs[qid]
        if not force and cache.has_pruned_research(qid):
            pre_cached_qids.add(qid)
        triples.append((question, gt, raw_blob))

    spend.cached_prune_hits += len(pre_cached_qids)

    results = await run_prune_for_qids(triples, cache, force=force, batch_size=args.prune_batch_size)

    new_blobs: dict[int, str] = {}
    new_metas: dict[int, dict] = {}
    for qid, result in results.items():
        if result is None:
            spend.prune_validation_failures += 1
            logger.warning("prune | qid=%d failed; dropping from downstream", qid)
            continue
        sanitized_blob, meta = result
        new_blobs[qid] = sanitized_blob
        new_metas[qid] = meta

    # Count fresh redactor invocations: one per qid that was NOT pre-cached.
    # One invocation per batch (default 10 qids per batch).
    fresh_qids = [qid for qid in qids if qid not in pre_cached_qids]
    if fresh_qids:
        batches = (len(fresh_qids) + args.prune_batch_size - 1) // args.prune_batch_size
        spend.redactor_invocations += batches

    working.research_blobs = new_blobs
    working.prune_metas = new_metas


# ---------------------------------------------------------------------------
# Screen stage
# ---------------------------------------------------------------------------


async def _stage_screen(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    force: bool,
    spend: SpendReport,
) -> None:
    """Run leakage screen on research blobs; drop leaked qids.

    Spend counters: cached + non-forced verdicts bump ``cached_screen_hits``.
    Fresh verdicts bump ``leakage_detector_calls`` UNLESS the verdict's
    ``detector_response`` equals the empty-blob sentinel — that path is a
    short-circuit in ``screen_research_blob`` that never hits the LLM.
    """
    await asyncio.sleep(0)
    questions_with_research = [working.questions[qid] for qid in working.research_blobs if qid in working.questions]
    ground_truths = {qid: working.ground_truths[qid] for qid in working.research_blobs if qid in working.ground_truths}
    research_blobs = dict(working.research_blobs)

    cached_verdicts, qids_needing_screen = _partition_screen_cache(cache, research_blobs, force=force, spend=spend)

    fresh_verdicts: dict[int, dict] = {}
    if qids_needing_screen:
        questions_subset = [q for q in questions_with_research if q.id_of_question in qids_needing_screen]
        gts_subset = {qid: gt for qid, gt in ground_truths.items() if qid in qids_needing_screen}
        blobs_subset = {qid: blob for qid, blob in research_blobs.items() if qid in qids_needing_screen}
        _clean_qs, _clean_gts, fresh_verdicts = await screen_batch(
            questions_subset,
            gts_subset,
            blobs_subset,
            cache,
            force=force,
            concurrency=args.concurrency,
        )

    for verdict in fresh_verdicts.values():
        # Verdict schema is fixed (leakage_screen._build_verdict);
        # ``detector_response`` is always present.
        if verdict["detector_response"] == _EMPTY_BLOB_RESPONSE:
            continue
        spend.leakage_detector_calls += 1

    all_verdicts = {**cached_verdicts, **fresh_verdicts}
    working.leakage_verdicts = all_verdicts

    # Drop leaked qids from the working set's downstream-stage data.
    # Verdict dicts come from ``leakage_screen._build_verdict`` with a fixed
    # schema; ``is_leaked`` is always present.
    for qid, verdict in all_verdicts.items():
        if verdict["is_leaked"]:
            working.research_blobs.pop(qid, None)


# ---------------------------------------------------------------------------
# QA-iterate stage
# ---------------------------------------------------------------------------


def _qa_iterate_paths(cache: AblationCache, *, timestamp: str) -> tuple[Path, Path]:
    """Return (summary_path, manual_rejects_path) for a qa_iterate run."""
    summary_path = cache.root / f"qa_summary_{timestamp}.md"
    manual_rejects_path = cache.root / "manual_rejects.json"
    return summary_path, manual_rejects_path


async def _stage_qa_iterate(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    force: bool = False,
) -> tuple[dict[int, IterateOutcome], Path | None]:
    """Iterate-until-clean QA over surviving qids.

    Returns ``(outcomes, summary_path)``. ``summary_path`` is ``None`` only when
    mode='skip' or there are zero surviving qids. Otherwise the summary is
    always written, regardless of mode.

    When ``force`` is True (passed via ``--force-stages qa_iterate``), the
    pre-existing ``manual_rejects.json`` is archived to
    ``manual_rejects.bak.<timestamp>.json`` instead of being honored, so the
    operator can re-run the verifier on previously-rejected qids after fixing
    upstream issues (e.g. redactor over-aggression).
    """
    await asyncio.sleep(0)
    if args.qa_iterate_mode == "skip":
        logger.info("stage=qa_iterate SKIPPED (mode=skip)")
        return {}, None

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    summary_path, manual_rejects_path = _qa_iterate_paths(cache, timestamp=timestamp)

    if force and manual_rejects_path.exists():
        # Auto-archive prior rejects on --force-stages qa_iterate. Without this,
        # forcing the stage is a silent no-op (the file is read at stage start
        # and pre-rejected qids are popped from working.research_blobs before
        # the verifier runs).
        backup_path = manual_rejects_path.parent / f"manual_rejects.bak.{timestamp}.json"
        manual_rejects_path.rename(backup_path)
        logger.info("--force-stages qa_iterate: archived prior rejects to %s", backup_path)
        existing_rejects = {}
    else:
        existing_rejects = read_manual_rejects(manual_rejects_path)
    for qid in list(existing_rejects.keys()):
        working.research_blobs.pop(qid, None)

    surviving_qids = sorted(working.research_blobs.keys())
    if not surviving_qids:
        logger.info("stage=qa_iterate no surviving qids after honoring manual_rejects.json")
        return {}, None

    inputs: dict[int, dict[str, Any]] = {}
    for qid in surviving_qids:
        inputs[qid] = {
            "question": working.questions[qid],
            "ground_truth": working.ground_truths[qid],
            "current_blob": working.research_blobs[qid],
            "screen_verdict": working.leakage_verdicts.get(
                qid,
                {"is_leaked": False, "detector_response": "", "detector_model": "", "detector_failed": False},
            ),
        }

    outcomes = await run_qa_iterate_batch(
        inputs,
        cache=cache,
        max_iterations=args.qa_iterate_max_iterations,
        leakage_threshold=args.qa_iterate_leakage_threshold,
        forecastability_threshold=args.qa_iterate_forecastability_threshold,
        concurrency=args.concurrency,
    )

    render_qa_summary(outcomes, summary_path)
    write_manual_rejects(list(outcomes.values()), manual_rejects_path)

    # Per-qid qa_reports: one structured JSON per qid recording final_status,
    # iterations, and verifier_scores. The plan
    # (``scratch_docs_and_planning/ablation_phase_a3_plan.md:290``) specifies
    # these alongside the aggregate summary; the audit at
    # ``backtests/ablation/audit_smoke_20260515.md:243-263`` confirmed they
    # were missing from the smoke run.
    qa_reports_dir = cache.root / "qa_reports"
    for qid, outcome in outcomes.items():
        report_path = qa_reports_dir / f"{qid}.json"
        atomic_write_text(report_path, json.dumps(serialize_outcome(outcome), indent=2, default=str))
        logger.info("qa_iterate | qid=%d report written: %s", qid, report_path)

    rejected_qids = [qid for qid, outcome in outcomes.items() if outcome.final_status != "clean"]
    for qid in rejected_qids:
        working.research_blobs.pop(qid, None)

    n_clean = sum(1 for o in outcomes.values() if o.final_status == "clean")
    logger.info(
        "stage=qa_iterate DONE | clean=%d rejected=%d summary=%s",
        n_clean,
        len(rejected_qids),
        summary_path,
    )
    return outcomes, summary_path


# ---------------------------------------------------------------------------
# Forecast stage
# ---------------------------------------------------------------------------


async def _stage_forecast(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    force: bool,
    spend: SpendReport,
) -> None:
    """Populate working.forecaster_payloads via cache + run_forecasters_batch.

    Spend counters work at the (qid, model_slug) cell level. Per qid we
    pre-snapshot the slugs already on disk; cells in the snapshot AND not
    forced count as ``cached_forecaster_hits``. Cells outside the snapshot
    (or all cells under force) count as ``forecaster_llm_calls``. Cell
    granularity uses the active lineup's model list so the count doesn't
    drift if the orchestrator's coarse all-or-nothing cache check changes
    shape later.
    """
    await asyncio.sleep(0)
    qids = sorted(working.research_blobs.keys())

    lineup_name: str = getattr(args, "lineup", "free")
    forecaster_llms, lineup_models = get_lineup(lineup_name)

    cached_per_qid: dict[int, dict[str, dict]] = {}
    pre_snapshot_slugs: dict[int, set[str]] = {}
    for qid in qids:
        # m2: filter the on-disk listing to the CURRENT lineup so obsolete
        # files don't pollute cache hit accounting or downstream stacker calls.
        on_disk = cache.list_forecaster_outputs(qid, lineup_filter=lineup_models)
        pre_snapshot_slugs[qid] = set(on_disk.keys())
        if force or not on_disk:
            continue
        # Task #23: content-aware cache hit. A forecaster cache that contains
        # only error payloads (prediction_value=None or non-empty errors) would
        # cause the stacker stage to permanently cache "insufficient_forecasters"
        # and skip the qid at score time. Re-run if the surviving count is below
        # the stacker's downstream threshold.
        n_valid = sum(1 for p in on_disk.values() if p.get("prediction_value") is not None and not p.get("errors"))
        if n_valid < ABLATION_MIN_FORECASTERS:
            logger.info(
                "forecast | qid=%d cache has %d valid forecasters (< %d threshold); re-running",
                qid,
                n_valid,
                ABLATION_MIN_FORECASTERS,
            )
            continue
        cached_per_qid[qid] = on_disk

    needs_run = [qid for qid in qids if qid not in cached_per_qid]
    questions_with_research = [(working.questions[qid], working.research_blobs[qid]) for qid in needs_run]

    fresh_results: dict[int, dict[str, dict]] = {}
    if questions_with_research:
        rate_limit_kwargs = _rate_limit_mode_kwargs(args.rate_limit_mode)
        fresh_results = await run_forecasters_batch(
            questions_with_research,
            cache,
            forecaster_llms=forecaster_llms,
            force=force,
            per_question_concurrency=args.concurrency,
            per_forecaster_concurrency=rate_limit_kwargs["per_forecaster_concurrency"],
            max_retries=rate_limit_kwargs["max_retries"],
        )

    for qid in qids:
        cached_slugs = pre_snapshot_slugs[qid] if not force else set()
        for model in lineup_models:
            slug = model_slug_to_filename(model)
            if slug in cached_slugs:
                spend.cached_forecaster_hits += 1
            else:
                spend.forecaster_llm_calls += 1

    for qid, payloads in cached_per_qid.items():
        working.forecaster_payloads[qid] = payloads
    for qid, payloads in fresh_results.items():
        working.forecaster_payloads[qid] = payloads


# ---------------------------------------------------------------------------
# Stacker stage
# ---------------------------------------------------------------------------


async def _stage_llm_stacker(
    arm: str,
    args: argparse.Namespace,
    working: WorkingSet,
    cache: AblationCache,
    *,
    force: bool,
    spend: SpendReport,
) -> None:
    """LLM stacker dispatch (stack/stack_aug). Uses run_stacker_batch."""
    await asyncio.sleep(0)
    qids = sorted(working.forecaster_payloads.keys())

    # Per-stacker cache keying: stack/stack_aug payloads are slugged by the active
    # stacker so a swap never clobbers another stacker's results.
    stacker_slug = _active_stacker_slug(args)

    cached_payloads, needs_run = _partition_stacker_cache(
        cache, qids, arm, stacker_slug=stacker_slug, force=force, spend=spend
    )

    fresh_results: dict[int, dict] = {}
    if needs_run:
        qid_to_data = {
            qid: {
                "question": working.questions[qid],
                "research": working.research_blobs.get(qid, ""),
                "forecaster_payloads": working.forecaster_payloads[qid],
            }
            for qid in needs_run
        }
        fresh_results = await run_stacker_batch(
            qid_to_data,
            arm,
            cache,
            **_stacker_batch_kwargs(args, stacker_slug=stacker_slug, force=force),
        )

    _record_stacker_spend(fresh_results, arm, spend)
    working.stacker_payloads[arm] = {**cached_payloads, **fresh_results}


async def _stage_stack(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    arm: str,
    force: bool,
    spend: SpendReport,
) -> None:
    """Dispatcher for stacker arms. Routes to per-type implementation."""
    if arm in (ARM_MEDIAN, ARM_MEAN):
        await _stage_simple_agg(arm, working, cache, force=force, spend=spend)
    elif arm == ARM_PDF:
        await _stage_pdf(working, cache, force, spend)
    else:
        await _stage_llm_stacker(arm, args, working, cache, force=force, spend=spend)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


async def _run_score_only(
    args: argparse.Namespace, cache: AblationCache, working: WorkingSet, spend: SpendReport
) -> int:
    """``--stages score``: hydrate every artifact from disk and score, without spending.

    Returns the process exit code — 2 when no qid has any arm payload at all, since
    there is nothing to score and the operator asked for exactly that.
    """
    await _hydrate_working_set_from_cache(cache, working, spend=spend, stacker_slug=_active_stacker_slug(args))
    _apply_qids_filter(working, args.qids)
    if not _qids_with_any_arm_payload(working):
        arm_counts = {arm: len(p) for arm, p in working.stacker_payloads.items()}
        logger.error(
            "Cannot run 'score': zero qids have any stacker outputs %s.",
            arm_counts,
        )
        print("ERROR: --stages score: zero qids have any arm payloads.")
        return 2
    summary_path = _stage_score(args, cache, working)
    _print_spend_report(spend, working, summary_path)
    return 0


async def _run_research_stages(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    spend: SpendReport,
    plan: _StagePlan,
) -> None:
    """Run the requested research / prune / screen stages in order."""
    if plan.wants("research"):
        n = len(working.questions)
        # Gemini grounded search ~30s/qid, parallelism = args.concurrency.
        est_seconds = max(30, n * 30 // max(1, args.concurrency))
        logger.info("stage=research START | est wall-clock ~%d min (n=%d)", est_seconds // 60 + 1, n)
        await _stage_research(args, cache, working, force=plan.is_forced("research"), spend=spend)
        logger.info("stage=research DONE | qids_with_blob=%d", len(working.research_blobs))
        await asyncio.sleep(plan.sleep_seconds)

    if plan.wants("prune"):
        n = len(working.research_blobs)
        # Redactor: ~30s/batch × ceil(n/batch_size).
        n_batches = (n + PRUNE_DEFAULT_BATCH_SIZE - 1) // max(1, PRUNE_DEFAULT_BATCH_SIZE) if n else 0
        est_seconds = max(30, n_batches * 30)
        logger.info(
            "stage=prune START | est wall-clock ~%d min (n=%d, batches=%d)", est_seconds // 60 + 1, n, n_batches
        )
        await _stage_prune(args, cache, working, force=plan.is_forced("prune"), spend=spend)
        logger.info(
            "stage=prune DONE | qids_with_sanitized_blob=%d | validation_failures=%d",
            len(working.research_blobs),
            spend.prune_validation_failures,
        )
        await asyncio.sleep(plan.sleep_seconds)

    if plan.wants("screen"):
        n = len(working.research_blobs)
        est_seconds = max(15, n * 10 // max(1, args.concurrency))
        logger.info("stage=screen START | est wall-clock ~%d min (n=%d)", est_seconds // 60 + 1, n)
        await _stage_screen(args, cache, working, force=plan.is_forced("screen"), spend=spend)
        # Verdict dicts have a fixed schema; ``is_leaked`` always present.
        n_leaked = sum(1 for v in working.leakage_verdicts.values() if v["is_leaked"])
        logger.info("stage=screen DONE | leaked=%d clean=%d", n_leaked, len(working.research_blobs))
        await asyncio.sleep(plan.sleep_seconds)


async def _run_qa_iterate_stage(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    plan: _StagePlan,
) -> None:
    """Run the QA-iterate stage, raising in halt mode so the operator reviews first.

    The inter-stage sleep happens BEFORE the halt-mode raise so the pause symmetry
    matches advisory mode and the operator's preferred backoff is honored even when
    halting right after the verifier batch. Halt is strict — it always blocks, even on
    a fully-clean batch, so the QA summary is read before any forecast spend.
    """
    logger.info("stage=qa_iterate START mode=%s", args.qa_iterate_mode)
    outcomes, qa_summary_path = await _stage_qa_iterate(args, cache, working, force=plan.is_forced("qa_iterate"))
    n_clean = sum(1 for o in outcomes.values() if o.final_status == "clean")
    n_rejected = len(outcomes) - n_clean
    await asyncio.sleep(plan.sleep_seconds)
    if args.qa_iterate_mode == "halt" and qa_summary_path is not None:
        raise RuntimeError(
            f"QA iteration halted: {n_rejected} rejects + {n_clean} clean qids. "
            f"Review {qa_summary_path}. To resume after review:\n"
            f"  1. (Optional) edit {cache.root}/manual_rejects.json to override rejects.\n"
            f"  2. Run: --stages forecast,stack,stack_aug,pdf,median,score (note: this skips qa_iterate; "
            f"manual_rejects is only consulted when qa_iterate is in --stages)."
        )


async def _run_forecast_stage(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    spend: SpendReport,
    plan: _StagePlan,
) -> None:
    """Run the forecaster fan-out stage."""
    n = len(working.research_blobs)
    n_forecasters = len(FREE_FORECASTER_MODELS)
    rl_kwargs = _rate_limit_mode_kwargs(args.rate_limit_mode)
    per_forecaster_concurrency = max(1, rl_kwargs["per_forecaster_concurrency"])
    # Forecaster: ~30s/call serially per question, parallelism =
    # (per_question_concurrency × per_forecaster_concurrency).
    per_question_seconds = (n_forecasters * 30) // per_forecaster_concurrency
    global_parallel = max(1, args.concurrency)
    est_seconds = max(60, n * per_question_seconds // global_parallel)
    logger.info(
        "stage=forecast START | est wall-clock ~%d min (n=%d × n_forecasters=%d / "
        "per_forecaster_concurrency=%d / question_concurrency=%d)",
        est_seconds // 60 + 1,
        n,
        n_forecasters,
        per_forecaster_concurrency,
        global_parallel,
    )
    await _stage_forecast(args, cache, working, force=plan.is_forced("forecast"), spend=spend)
    logger.info("stage=forecast DONE | qids=%d", len(working.forecaster_payloads))
    await asyncio.sleep(plan.sleep_seconds)


async def _run_arm_stages(
    args: argparse.Namespace,
    cache: AblationCache,
    working: WorkingSet,
    *,
    spend: SpendReport,
    plan: _StagePlan,
) -> None:
    """Run each requested aggregation arm, in ``_ARM_STAGES`` order."""
    for stage in _ARM_STAGES:
        if not plan.wants(stage.name):
            continue
        n = len(working.forecaster_payloads)
        if stage.deterministic_note is None:
            est_seconds = max(30, n * 30 // max(1, args.concurrency))
            logger.info("stage=%s START | est wall-clock ~%d min (n=%d)", stage.name, est_seconds // 60 + 1, n)
        else:
            logger.info("stage=%s START | est wall-clock ~1 min (n=%d, %s)", stage.name, n, stage.deterministic_note)
        await _stage_stack(args, cache, working, arm=stage.arm, force=plan.is_forced(stage.name), spend=spend)
        logger.info("stage=%s DONE | qids=%d", stage.name, len(working.stacker_payloads.get(stage.report_arm, {})))
        if stage.deterministic_note is None:
            await asyncio.sleep(plan.sleep_seconds)


def _run_score_stage(args: argparse.Namespace, cache: AblationCache, working: WorkingSet) -> Path | None:
    """Score the run, or return None (after a WARN) when no arm produced any payload."""
    logger.info("stage=score START")
    if not _qids_with_any_arm_payload(working):
        arm_counts = {arm: len(p) for arm, p in working.stacker_payloads.items()}
        logger.warning("score | no qids have any arm payloads %s; skipping", arm_counts)
        return None
    summary_path = _stage_score(args, cache, working)
    logger.info("stage=score DONE | summary=%s", summary_path)
    return summary_path


async def run_ablation(args: argparse.Namespace) -> int:
    """Top-level orchestrator. Returns process exit code (0 OK, 1 partial, 2 fatal config)."""
    if args.rate_limit_mode == "patient" and args.concurrency > 1:
        # patient mode promises "concurrency=1" but --concurrency is the
        # per-question knob (default 4) and stacks multiplicatively with
        # per_forecaster_concurrency. Clamping here keeps the doc-comment
        # honest at 50q+ where free-tier 429s correlate across providers.
        logger.warning(
            "rate_limit_mode=patient overrides --concurrency=%d to 1 (avoids 4-way "
            "free-tier flooding; pass --rate-limit-mode gentle for parallel question fan-out)",
            args.concurrency,
        )
        args.concurrency = 1

    cache = AblationCache(args.cache_dir)
    working = WorkingSet()
    spend = SpendReport()

    forced_explicit = set(args.force_stages)
    forced = _expand_forced_stages(forced_explicit)
    if forced != forced_explicit:
        logger.info("forced stages (after cascade): %s", sorted(forced))
    plan = _StagePlan(requested=set(args.stages), forced=forced, sleep_seconds=args.per_question_sleep)

    if plan.requested == {"score"}:
        return await _run_score_only(args, cache, working, spend)

    # Full pipeline (or any subset that includes upstream stages).
    if plan.wants("fetch"):
        logger.info("stage=fetch START")
        await _stage_fetch(args, cache, working)
        logger.info("stage=fetch DONE | qids=%d", len(working.questions))
    else:
        await _hydrate_working_set_from_cache(cache, working, stacker_slug=_active_stacker_slug(args))
        _apply_qids_filter(working, args.qids)

    await _run_research_stages(args, cache, working, spend=spend, plan=plan)

    if args.qa_research:
        qa_path = _stage_qa_research_dump(args, cache, working)
        print(f"QA research dump written to: {qa_path}")
        _print_spend_report(spend, working, summary_path=None)
        return 0

    if plan.wants("qa_iterate"):
        await _run_qa_iterate_stage(args, cache, working, plan=plan)

    if plan.wants("forecast"):
        await _run_forecast_stage(args, cache, working, spend=spend, plan=plan)

    await _run_arm_stages(args, cache, working, spend=spend, plan=plan)

    summary_path = _run_score_stage(args, cache, working) if plan.wants("score") else None
    _print_spend_report(spend, working, summary_path)
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _configure_logging(args: argparse.Namespace, cache_dir: Path) -> Path:
    """Configure logging to write to both console and a per-invocation file.

    The audit at ``backtests/ablation/audit_smoke_20260515.md`` flagged a
    38-line smoke log because Python's root logger defaults to WARNING and no
    ``logging.basicConfig`` was called anywhere in ``metaculus_bot/ablation/``.
    This wires up file-archived INFO logging so every stage's per-qid
    diagnostics survive on disk for review.

    ``force=True`` replaces any pre-existing handlers (e.g., ones that other
    modules attached during import) so the configured level actually takes
    effect.
    """
    logs_dir = cache_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    # astimezone() attaches the local zone without shifting the wall clock, so the
    # run-log filename stays local-time (what the operator greps) and tz-aware.
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{timestamp}.log"

    level = getattr(logging, args.log_level.upper())

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(DEFAULT_CACHE_DIR)
    log_path = _configure_logging(args, cache_dir)
    logger.info("ablation run starting; log file: %s", log_path)
    # Suppress aiohttp "Unclosed client session" warnings from litellm's
    # cached HTTP transport (1-hour TTL) — same fix used by backtest.py and
    # community_benchmark.py. Once-per-process noise, no FD leak.
    enable_aiohttp_session_autoclose()
    return asyncio.run(run_ablation(args))


if __name__ == "__main__":
    sys.exit(main())
