"""The two mutable records every ablation stage threads through, and how they load.

``SpendReport`` counts what a run spent and what it reused; ``WorkingSet`` holds the
per-qid artifacts (questions, ground truths, research blobs, verdicts, per-forecaster
and per-arm payloads) that flow from one stage to the next. Both are constructed once
per CLI run in ``ablation.cli.run_ablation``.

The helpers here are the read paths that populate them without spending:

* ``_partition_*_cache`` — split a stage's qids into reusable cached artifacts vs. the
  ones needing a fresh call, bumping the matching ``cached_*_hits`` counter;
* ``_record_stacker_spend`` — count stacker + parser calls off an arm's fresh results;
* ``_hydrate_*`` — rebuild the whole working set from disk for ``--stages score`` and
  for any subset that skips ``fetch``;
* ``_filter_working_set_to_qids`` / ``_apply_qids_filter`` — narrow it to ``--qids``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.leakage_screen import _research_blob_sha
from metaculus_bot.ablation.manifest_serde import (
    _build_question_shim_from_manifest_entry,
    _deserialize_ground_truth,
)
from metaculus_bot.ablation.run_pdf import ARM_PDF_MIN1, ARM_PDF_MIN2
from metaculus_bot.ablation.run_stacker import ARM_MEAN, ARM_MEDIAN, ARM_PDF, ARM_STACK, ARM_STACK_AUG
from metaculus_bot.backtest.scoring import GroundTruth

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class SpendReport:
    """Tracks API-call counts and cache hits across a single CLI run.

    Counters are populated by each stage by snapshotting cache state before the
    stage runs and diffing against post-stage cache state. The orchestrator
    (``run_ablation``) constructs one ``SpendReport`` per CLI run and threads
    it through every stage; ``_print_spend_report`` renders it at the end.
    """

    gemini_research_calls: int = 0
    gemini_gap_fill_calls: int = 0
    leakage_detector_calls: int = 0
    forecaster_llm_calls: int = 0
    stacker_llm_calls_stack: int = 0
    stacker_llm_calls_stack_aug: int = 0
    parser_llm_calls: int = 0
    redactor_invocations: int = 0
    cached_research_hits: int = 0
    cached_prune_hits: int = 0
    cached_screen_hits: int = 0
    cached_forecaster_hits: int = 0
    cached_stacker_hits: dict[str, int] = field(default_factory=dict)
    fallback_stacker_stack: int = 0
    fallback_stacker_stack_aug: int = 0
    prune_validation_failures: int = 0


@dataclass
class WorkingSet:
    """In-memory state shared across stages.

    ``research_blobs`` holds RAW blobs after the research stage and SANITIZED
    blobs after the prune stage. Raw blobs always remain on disk under
    ``research/<qid>.md``; sanitized blobs land in ``research_pruned/<qid>.md``.
    Downstream stages (screen, forecast, stack) always see sanitized blobs.
    ``prune_metas`` carries per-qid redactor metadata for the QA dump.
    """

    questions: dict[int, Any] = field(default_factory=dict)
    ground_truths: dict[int, GroundTruth] = field(default_factory=dict)
    research_blobs: dict[int, str] = field(default_factory=dict)
    prune_metas: dict[int, dict] = field(default_factory=dict)
    leakage_verdicts: dict[int, dict] = field(default_factory=dict)
    forecaster_payloads: dict[int, dict[str, dict]] = field(default_factory=dict)
    stacker_payloads: dict[str, dict[int, dict]] = field(default_factory=dict)


def _partition_screen_cache(
    cache: AblationCache,
    research_blobs: dict[int, str],
    *,
    force: bool,
    spend: SpendReport,
) -> tuple[dict[int, dict], list[int]]:
    """Split the qids into (reusable cached verdicts, qids needing a fresh screen).

    A cached verdict only counts when its ``research_blob_sha`` still matches the blob
    on hand: a pre-C3 entry carrying no sha, or a stale sha after a re-prune, re-screens.
    Bumps ``spend.cached_screen_hits`` for each reuse.
    """
    if force:
        return {}, list(research_blobs.keys())

    cached_verdicts: dict[int, dict] = {}
    qids_needing_screen: list[int] = []
    for qid, blob in research_blobs.items():
        cached = cache.read_leakage_screen(qid)
        if cached is not None and cached.get("research_blob_sha") == _research_blob_sha(blob):
            cached_verdicts[qid] = cached
            spend.cached_screen_hits += 1
        else:
            if cached is not None:
                logger.info("screen | qid=%d cache stale (blob changed); re-screening", qid)
            qids_needing_screen.append(qid)
    return cached_verdicts, qids_needing_screen


def _partition_stacker_cache(
    cache: AblationCache,
    qids: list[int],
    arm: str,
    *,
    stacker_slug: str | None,
    force: bool,
    spend: SpendReport,
) -> tuple[dict[int, dict], list[int]]:
    """Split the qids into (cached arm payloads, qids needing a stacker run)."""
    if force:
        return {}, list(qids)

    cached_payloads: dict[int, dict] = {}
    needs_run: list[int] = []
    for qid in qids:
        cached = cache.read_stacker_output(qid=qid, arm=arm, stacker_slug=stacker_slug)
        if cached is not None:
            cached_payloads[qid] = cached
            spend.cached_stacker_hits[arm] = spend.cached_stacker_hits.get(arm, 0) + 1
        else:
            needs_run.append(qid)
    return cached_payloads, needs_run


def _record_stacker_spend(fresh_results: dict[int, dict], arm: str, spend: SpendReport) -> None:
    """Count stacker + parser LLM calls and fallback usage from this arm's fresh results."""
    is_arm_a = arm == ARM_STACK
    for payload in fresh_results.values():
        used = payload["stacker_model_used"]
        if used in ("primary", "fallback"):
            if is_arm_a:
                spend.stacker_llm_calls_stack += 1
            else:
                spend.stacker_llm_calls_stack_aug += 1
            spend.parser_llm_calls += 1
        if used == "fallback":
            if is_arm_a:
                spend.fallback_stacker_stack += 1
            else:
                spend.fallback_stacker_stack_aug += 1


def _hydrate_research_artifacts(
    cache: AblationCache,
    working: WorkingSet,
    qid: int,
    *,
    spend: SpendReport | None,
) -> None:
    """Load one qid's research + pruned-research artifacts into the working set.

    Only the SANITIZED blob flows downstream — the raw blob stays on disk for QA-dump
    inspection only. A qid whose prune failed (no cached pruned blob) must NOT be
    eligible for forecast or later stages even when those stages are requested without
    ``--stages prune``; the prune stage drops such qids from ``research_blobs``, and
    leaving them out here mirrors that, so re-running downstream stages from cache
    produces the same working set.
    """
    if cache.read_research(qid) is not None and spend is not None:
        spend.cached_research_hits += 1
    cached_pruned = cache.read_pruned_research(qid)
    if cached_pruned is not None:
        working.research_blobs[qid] = cached_pruned[0]
        working.prune_metas[qid] = cached_pruned[1]
        if spend is not None:
            spend.cached_prune_hits += 1


def _hydrate_screen_and_forecasters(
    cache: AblationCache,
    working: WorkingSet,
    qid: int,
    *,
    spend: SpendReport | None,
) -> None:
    """Load one qid's leakage verdict + per-model forecaster payloads."""
    verdict = cache.read_leakage_screen(qid)
    if verdict is not None:
        working.leakage_verdicts[qid] = verdict
        if spend is not None:
            spend.cached_screen_hits += 1
    forecaster_payloads = cache.list_forecaster_outputs(qid)
    if forecaster_payloads:
        working.forecaster_payloads[qid] = forecaster_payloads
        if spend is not None:
            spend.cached_forecaster_hits += len(forecaster_payloads)


def _hydrate_stacker_arms(
    cache: AblationCache,
    working: WorkingSet,
    qid: int,
    *,
    spend: SpendReport | None,
    stacker_slug: str | None,
) -> None:
    """Load one qid's payload for every arm that has one on disk."""
    for arm in (ARM_STACK, ARM_STACK_AUG, ARM_PDF, ARM_PDF_MIN1, ARM_PDF_MIN2, ARM_MEDIAN, ARM_MEAN):
        # Only the LLM-stacker arms are slugged; deterministic arms stay shared.
        arm_slug = stacker_slug if arm in (ARM_STACK, ARM_STACK_AUG) else None
        payload = cache.read_stacker_output(qid=qid, arm=arm, stacker_slug=arm_slug)
        if payload is not None:
            working.stacker_payloads.setdefault(arm, {})[qid] = payload
            if spend is not None:
                spend.cached_stacker_hits[arm] = spend.cached_stacker_hits.get(arm, 0) + 1


async def _hydrate_working_set_from_cache(
    cache: AblationCache,
    working: WorkingSet,
    spend: SpendReport | None = None,
    stacker_slug: str | None = None,
) -> None:
    """For score-only paths: load every artifact from disk.

    When ``spend`` is supplied (score-only path), bumps the relevant
    ``cached_*_hits`` fields so the spend report reflects what was loaded
    rather than reading as a hard zero across the board.

    ``stacker_slug`` is applied ONLY to the LLM-stacker arms (stack / stack_aug)
    so the score-only path reads the active run's stacker outputs; deterministic
    arms (pdf / median / mean) read with ``stacker_slug=None`` (shared, unslugged).
    The in-memory ``working.stacker_payloads`` keys stay the plain arm name —
    only the on-disk filename carries the slug — so scoring/summary code is
    untouched.
    """
    await asyncio.sleep(0)
    for qid_raw, entry in cache.read_qids_manifest().items():
        qid = int(qid_raw)
        working.questions[qid] = _build_question_shim_from_manifest_entry(qid, entry)
        working.ground_truths[qid] = _deserialize_ground_truth(entry["ground_truth"])
        _hydrate_research_artifacts(cache, working, qid, spend=spend)
        _hydrate_screen_and_forecasters(cache, working, qid, spend=spend)
        _hydrate_stacker_arms(cache, working, qid, spend=spend, stacker_slug=stacker_slug)


def _filter_working_set_to_qids(working: WorkingSet, qids: list[int]) -> set[int]:
    """Restrict every per-qid attribute of ``working`` to entries in ``qids``.

    Returns the set of requested qids that were NOT found in the manifest
    (so the caller can log them). Without this filter, --qids X --stages
    stack would silently fan out the stacker over the full manifest.
    """
    requested = set(qids)
    for attr in (
        "questions",
        "ground_truths",
        "research_blobs",
        "prune_metas",
        "leakage_verdicts",
        "forecaster_payloads",
    ):
        existing = getattr(working, attr)
        filtered = {qid: v for qid, v in existing.items() if qid in requested}
        setattr(working, attr, filtered)
    # Filter each arm's payload dict within stacker_payloads.
    for arm_key in list(working.stacker_payloads.keys()):
        working.stacker_payloads[arm_key] = {
            qid: v for qid, v in working.stacker_payloads[arm_key].items() if qid in requested
        }
    return requested - set(working.questions.keys())


def _qids_with_any_arm_payload(working: WorkingSet) -> set[int]:
    """Union of qids carrying a payload for ANY arm.

    Per-comparison N means scoring only needs SOME qids with >= 2 arm payloads, so the
    gate is "any arm at all", not the intersection across arms.
    """
    all_arm_qids: set[int] = set()
    for arm_payloads in working.stacker_payloads.values():
        all_arm_qids.update(arm_payloads.keys())
    return all_arm_qids


def _apply_qids_filter(working: WorkingSet, qids: list[int] | None) -> None:
    """Restrict the working set to ``qids``, logging any the manifest didn't have."""
    if not qids:
        return
    missing = _filter_working_set_to_qids(working, qids)
    if missing:
        logger.error("--qids filter: %d qids not in working set: %s", len(missing), sorted(missing))
