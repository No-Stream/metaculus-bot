"""The ablation's deterministic aggregation arms: mean, median, and structured-math PDF.

Three arms that make zero LLM calls, so unlike the ``stack`` / ``stack_aug`` arms they
carry no retry budget, no fallback chain and no stacker slug — they read the cached
per-forecaster payloads and aggregate. ``ablation.cli._stage_stack`` dispatches here for
``ARM_MEAN`` / ``ARM_MEDIAN`` / ``ARM_PDF`` and keeps the LLM arm itself, whose
``run_stacker_batch`` call has to stay patchable at ``ablation.cli``.

The PDF stage runs three sub-arms per qid — ``pdf_min1`` and ``pdf_min2`` are scored, and
``pdf_min1_mean`` is computed for diagnostic completeness only. Each sub-arm reads and
writes its own cache entry under its own label, and ``run_pdf_for_qid`` is the single
cache authority for all three: the stage keys nothing on the parent ``ARM_PDF``. A stage
that read the parent key would hand one payload to two arms and skip the per-arm reads,
which is exactly what happened on cache dirs holding legacy ``arm_pdf.json`` files.

``ARM_PDF_MIN1_MEAN`` used to be a function-scoped import in ``ablation.cli`` so a partial
edit couldn't leave the formatter stripping it as unused. It is top-level here because the
rest of the module already imports ``run_pdf``, and nothing patches it at its own module,
so a top-level binding cannot go stale under a monkeypatch.
"""

from __future__ import annotations

import asyncio

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.run_pdf import (
    ARM_PDF_MIN1,
    ARM_PDF_MIN1_MEAN,
    ARM_PDF_MIN2,
    run_pdf_for_qid,
)
from metaculus_bot.ablation.run_simple_agg import run_mean_for_qid, run_median_for_qid
from metaculus_bot.ablation.run_stacker import ARM_MEDIAN
from metaculus_bot.ablation.run_state import SpendReport, WorkingSet


async def _stage_simple_agg(
    arm: str,
    working: WorkingSet,
    cache: AblationCache,
    *,
    force: bool,
    spend: SpendReport,
) -> None:
    """Sequential deterministic aggregation (mean/median). No LLM calls."""
    await asyncio.sleep(0)
    qids = sorted(working.forecaster_payloads.keys())

    cached_payloads: dict[int, dict] = {}
    needs_run: list[int] = []
    for qid in qids:
        if not force:
            cached = cache.read_stacker_output(qid=qid, arm=arm)
            if cached is not None:
                cached_payloads[qid] = cached
                spend.cached_stacker_hits[arm] = spend.cached_stacker_hits.get(arm, 0) + 1
                continue
        needs_run.append(qid)

    fresh_results: dict[int, dict] = {}
    if needs_run:
        run_fn = run_median_for_qid if arm == ARM_MEDIAN else run_mean_for_qid
        for qid in needs_run:
            fresh_results[qid] = await run_fn(
                qid=qid,
                question=working.questions[qid],
                forecaster_payloads=working.forecaster_payloads[qid],
                cache=cache,
                force=force,
            )

    working.stacker_payloads[arm] = {**cached_payloads, **fresh_results}


async def _stage_pdf(
    working: WorkingSet,
    cache: AblationCache,
    force: bool,
    spend: SpendReport,
) -> None:
    """Deterministic structured-math aggregation for all PDF sub-arms.

    Every qid goes through ``run_pdf_for_qid`` once per sub-arm; that function owns the
    per-label cache read, so a cached sub-arm costs one file read and nothing here is
    keyed on the parent ``ARM_PDF``.
    """
    await asyncio.sleep(0)
    qids = sorted(working.forecaster_payloads.keys())

    results_min1: dict[int, dict] = {}
    results_min2: dict[int, dict] = {}
    for qid in qids:
        question = working.questions[qid]
        forecaster_payloads = working.forecaster_payloads[qid]
        results_min1[qid] = await run_pdf_for_qid(
            qid=qid,
            question=question,
            forecaster_payloads=forecaster_payloads,
            cache=cache,
            force=force,
            spend=spend,
            min_forecasters=1,
            arm_label=ARM_PDF_MIN1,
        )
        results_min2[qid] = await run_pdf_for_qid(
            qid=qid,
            question=question,
            forecaster_payloads=forecaster_payloads,
            cache=cache,
            force=force,
            spend=spend,
            min_forecasters=2,
            arm_label=ARM_PDF_MIN2,
        )
        # pdf_min1_mean is computed for diagnostic completeness but not scored.
        await run_pdf_for_qid(
            qid=qid,
            question=question,
            forecaster_payloads=forecaster_payloads,
            cache=cache,
            force=force,
            spend=spend,
            min_forecasters=1,
            arm_label=ARM_PDF_MIN1_MEAN,
            aggregation="mean",
        )

    working.stacker_payloads[ARM_PDF_MIN1] = results_min1
    working.stacker_payloads[ARM_PDF_MIN2] = results_min2
