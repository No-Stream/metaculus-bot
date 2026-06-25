"""Gemini-only research for the probabilistic-tools ablation benchmark.

The bot's production ``run_research`` falls through to an empty stub when the
AskNews/Exa/Perplexity/OpenRouter primary providers are all unconfigured —
which short-circuits Gemini entirely. This module bypasses that by calling
``gemini_search_provider`` directly, then a bounded second-pass via
``run_gap_fill_pass``, and persists the concatenated blob in
``AblationCache``.

Concatenation format and gap-fill threshold mirror ``main.run_research`` so
downstream code paths see the same shape they'd see in production.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from forecasting_tools import MetaculusQuestion

from metaculus_bot import constants
from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.window_patch import patched_gap_fill_year_for_question
from metaculus_bot.constants import (
    GAP_FILL_MIN_RESEARCH_CHARS,
    GEMINI_SEARCH_DEFAULT_MODEL,
    GEMINI_SEARCH_MODEL_ENV,
)
from metaculus_bot.research import targeted
from metaculus_bot.research.gemini_search import gemini_search_provider
from metaculus_bot.research.targeted import run_gap_fill_pass

__all__ = ["run_gemini_only_research", "run_gemini_research_for_qids"]

logger: logging.Logger = logging.getLogger(__name__)

_GAP_FILL_HEADER = "\n\n---\n\n## Targeted Gap-Fill (second pass)\n\n"


@contextmanager
def _patched_gap_fill_max_gaps(value: int) -> Iterator[None]:
    """Override ``GAP_FILL_MAX_GAPS`` in both ``constants`` and ``research.targeted``.

    ``research.targeted`` does ``from metaculus_bot.constants import GAP_FILL_MAX_GAPS``
    at import time, binding the integer in its own namespace; patching only
    ``constants`` would leave the analyzer call seeing the old value. Patch both
    and restore both in ``finally`` so tests don't leak state across the suite.
    """
    original_constants = constants.GAP_FILL_MAX_GAPS
    original_tr = targeted.GAP_FILL_MAX_GAPS
    constants.GAP_FILL_MAX_GAPS = value
    targeted.GAP_FILL_MAX_GAPS = value
    try:
        yield
    finally:
        constants.GAP_FILL_MAX_GAPS = original_constants
        targeted.GAP_FILL_MAX_GAPS = original_tr


@contextmanager
def _patched_gemini_search_model(model_slug: str | None) -> Iterator[None]:
    """Override the ``GEMINI_SEARCH_MODEL`` env var for the duration of a call.

    ``gemini_search_provider`` reads ``GEMINI_SEARCH_MODEL`` at request time via
    ``os.getenv``; the ablation CLI's ``--gemini-model`` flag is canonical, so we
    patch the env var while the provider runs and restore the original value in
    ``finally`` regardless of outcome. ``None`` is a no-op so the existing
    cache-hit path doesn't disturb the env.
    """
    if model_slug is None:
        yield
        return
    sentinel = object()
    original = os.environ.get(GEMINI_SEARCH_MODEL_ENV, sentinel)
    os.environ[GEMINI_SEARCH_MODEL_ENV] = model_slug
    try:
        yield
    finally:
        if original is sentinel:
            os.environ.pop(GEMINI_SEARCH_MODEL_ENV, None)
        else:
            assert isinstance(original, str)
            os.environ[GEMINI_SEARCH_MODEL_ENV] = original


def _count_gap_sections(addendum: str) -> int:
    """Count ``### Gap N:`` section headers in the gap-fill addendum.

    ``run_gap_fill_pass`` emits one ``### Gap <idx>: <text>`` per resolved gap.
    Counting headers is the only way to recover the actual gap count from the
    string output without re-running the analyzer.
    """
    return sum(1 for line in addendum.splitlines() if line.startswith("### Gap "))


def _build_meta(
    *,
    first_pass: str,
    gap_fill: str,
    gap_fill_used: bool,
    gap_fill_enabled: bool,
    gap_fill_max_gaps: int,
    is_benchmarking: bool,
    gemini_model: str,
) -> dict:
    return {
        "gemini_search_used": True,
        "gap_fill_used": gap_fill_used,
        "gap_fill_enabled": gap_fill_enabled,
        "gap_count": _count_gap_sections(gap_fill) if gap_fill_used else 0,
        "first_pass_chars": len(first_pass),
        "gap_fill_chars": len(gap_fill),
        "researched_at": datetime.now().isoformat(),
        "gemini_model": gemini_model,
        "gap_fill_max_gaps": gap_fill_max_gaps,
        "is_benchmarking": is_benchmarking,
    }


async def run_gemini_only_research(
    question: MetaculusQuestion,
    cache: AblationCache,
    *,
    gap_fill_max_gaps: int = 3,
    is_benchmarking: bool = True,
    force: bool = False,
    gemini_model: str | None = None,
    enable_gap_fill: bool = True,
) -> tuple[str, dict]:
    """Run Gemini grounded search + bounded gap-fill, cached on disk.

    On cache hit (and ``force=False``) returns immediately without API calls.
    On primary Gemini failure, re-raises (caller decides whether to drop the qid).
    On gap-fill failure, soft-fails: caches and returns the first-pass blob alone
    with ``gap_fill_used=False`` (matches production semantics).

    When ``gemini_model`` is set, ``GEMINI_SEARCH_MODEL`` env var is overridden
    for the duration of the provider call. The CLI flag is canonical: a shell
    ``GEMINI_SEARCH_MODEL`` setting cannot leak through.

    When ``enable_gap_fill=False``, the second-pass gap-fill stage is skipped
    entirely (no LLM call, no addendum). Meta records ``gap_fill_enabled``
    alongside ``gap_fill_used`` so cached blobs are self-describing.
    """
    qid = question.id_of_question
    assert qid is not None, "MetaculusQuestion must have id_of_question set"

    if not force:
        cached = cache.read_research(qid)
        if cached is not None:
            logger.info(f"Cache HIT for qid {qid}")
            await asyncio.sleep(0)
            return cached

    logger.info(f"Cache MISS for qid {qid}, fetching...")

    effective_model = gemini_model or GEMINI_SEARCH_DEFAULT_MODEL

    with _patched_gemini_search_model(gemini_model):
        provider = gemini_search_provider(is_benchmarking=is_benchmarking)
        first_pass = await provider(question)

    gap_fill_blob = ""
    gap_fill_used = False

    if enable_gap_fill and len(first_pass) >= GAP_FILL_MIN_RESEARCH_CHARS:
        try:
            with (
                patched_gap_fill_year_for_question(question),
                _patched_gap_fill_max_gaps(gap_fill_max_gaps),
                _patched_gemini_search_model(gemini_model),
            ):
                gap_fill_blob = await run_gap_fill_pass(
                    question,
                    first_pass,
                    is_benchmarking=is_benchmarking,
                )
            gap_fill_used = bool(gap_fill_blob)
            logger.info(f"Gap-fill returned {len(gap_fill_blob)} chars for qid {qid}")
        except Exception as exc:
            logger.warning(f"Gap-fill failed for qid {qid}: {exc}", exc_info=True)
            gap_fill_blob = ""
            gap_fill_used = False

    if gap_fill_blob:
        blob = f"{first_pass}{_GAP_FILL_HEADER}{gap_fill_blob}"
    else:
        blob = first_pass

    meta = _build_meta(
        first_pass=first_pass,
        gap_fill=gap_fill_blob,
        gap_fill_used=gap_fill_used,
        gap_fill_enabled=enable_gap_fill,
        gap_fill_max_gaps=gap_fill_max_gaps,
        is_benchmarking=is_benchmarking,
        gemini_model=effective_model,
    )

    cache.write_research(qid, blob, meta)
    return blob, meta


async def run_gemini_research_for_qids(
    questions: list[MetaculusQuestion],
    cache: AblationCache,
    *,
    gap_fill_max_gaps: int = 3,
    is_benchmarking: bool = True,
    force: bool = False,
    concurrency: int = 4,
    gemini_model: str | None = None,
    enable_gap_fill: bool = True,
) -> dict[int, tuple[str, dict] | None]:
    """Run ``run_gemini_only_research`` per question under a semaphore.

    Per-question failures cache nothing for that qid and surface as ``None`` in
    the result dict; other questions still complete.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(question: MetaculusQuestion) -> tuple[int, tuple[str, dict] | None]:
        qid = question.id_of_question
        assert qid is not None, "MetaculusQuestion must have id_of_question set"
        async with semaphore:
            try:
                result = await run_gemini_only_research(
                    question,
                    cache,
                    gap_fill_max_gaps=gap_fill_max_gaps,
                    is_benchmarking=is_benchmarking,
                    force=force,
                    gemini_model=gemini_model,
                    enable_gap_fill=enable_gap_fill,
                )
            except Exception as exc:
                logger.warning(f"Research failed for qid {qid}: {exc}", exc_info=True)
                return qid, None
            return qid, result

    tasks = [_one(q) for q in questions]
    completed = await asyncio.gather(*tasks)
    return dict(completed)
