"""Targeted research pipeline for conditional stacking.

When base forecaster models disagree significantly, this module:
1. Extracts the crux of disagreement using a cheap analyzer model.
2. Runs a targeted web search via Grok native search to resolve it.

Also provides ``run_gap_fill_pass`` — an always-on second-pass that runs after
first-pass research. It identifies factual gaps in the first pass and resolves
each via a parallel grounded Gemini search.
"""

import asyncio
import json
import logging
from typing import Any

import httpx
from forecasting_tools import GeneralLlm, MetaculusQuestion
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from metaculus_bot.constants import (
    GAP_FILL_ANALYZER_MODEL,
    GAP_FILL_ANALYZER_TIMEOUT,
    GAP_FILL_MAX_GAPS,
)
from metaculus_bot.gemini_search_provider import build_gemini_client, invoke_gemini_grounded
from metaculus_bot.prompts import (
    disagreement_crux_prompt,
    gap_fill_analyzer_prompt,
    gap_fill_search_prompt,
    targeted_search_prompt,
)
from metaculus_bot.research_providers import build_native_search_llm
from metaculus_bot.structured_output_schema import extract_first_balanced_braces, extract_json_block

__all__ = [
    "extract_disagreement_crux",
    "run_gap_fill_pass",
    "run_targeted_search",
]

logger: logging.Logger = logging.getLogger(__name__)

# Exceptions that trigger a soft-fail (return "") from run_gap_fill_pass.
_GAP_FILL_SOFT_FAIL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    ValueError,  # missing GOOGLE_API_KEY from build_gemini_client
    genai_errors.APIError,  # covers ClientError + ServerError from the SDK
    httpx.HTTPError,  # raw httpx network errors (SDK uses httpx underneath)
    ConnectionError,
    OSError,  # DNS / socket failures not wrapped by httpx
)


async def extract_disagreement_crux(
    analyzer_llm: GeneralLlm,
    question_text: str,
    base_prediction_texts: list[str],
) -> str:
    """Identify the core factual disagreement across base forecaster analyses.

    Args:
        analyzer_llm: A cheap model (e.g. GPT-5-mini) used for extraction only.
        question_text: The full question text being forecasted.
        base_prediction_texts: Reasoning texts from base models (already stripped of model tags).

    Returns:
        A short string describing the factual crux of disagreement.
    """
    prompt = disagreement_crux_prompt(question_text, base_prediction_texts)
    logger.info(f"Extracting disagreement crux from {len(base_prediction_texts)} forecaster analyses")
    crux = await analyzer_llm.invoke(prompt)
    logger.info(f"Disagreement crux extracted: {len(crux)} chars")
    return crux


async def run_targeted_search(crux: str, question_text: str, *, is_benchmarking: bool = False) -> str:
    """Run a targeted web search to resolve a specific factual disagreement.

    Uses Grok with native search via OpenRouter to find current, authoritative
    information about the identified crux.

    Args:
        crux: The factual question(s) driving forecaster disagreement.
        question_text: The full question text being forecasted.
        is_benchmarking: If True, excludes prediction market data to avoid data leakage.

    Returns:
        Search results with inline citations addressing the crux.
    """
    llm = build_native_search_llm()
    prompt = targeted_search_prompt(crux, question_text, is_benchmarking=is_benchmarking)
    logger.info(f"Running targeted search via {llm.model} for crux: {crux[:100]}...")
    result = await llm.invoke(prompt)
    logger.info(f"Targeted search complete: {len(result)} chars")
    return result


# ---------------------------------------------------------------------------
# Second-pass gap-fill
# ---------------------------------------------------------------------------


def _parse_gap_list(raw: str, *, max_gaps: int | None = None) -> list[dict[str, str]]:
    """Extract the gap list from the analyzer's JSON output.

    Robust to light markdown wrapping (```json``` fences) and trailing commentary.
    Returns [] on any parse failure — callers should soft-fail.

    If ``max_gaps`` is provided, the returned list is clipped to that length.
    This lives in the parser (rather than the caller) so the "no more than N"
    contract can be unit-tested without mocking out the analyzer call.
    """
    if not raw or not raw.strip():
        return []

    # Prefer fenced blocks (canonical extractor in structured_output_schema);
    # fall back to a string-literal-aware balanced-brace scan for unfenced
    # payloads with trailing commentary. Both helpers live in one module so
    # the brace-scanner is fixed in one place.
    fenced = extract_json_block(raw)
    if fenced is not None:
        stripped = fenced
    else:
        stripped = extract_first_balanced_braces(raw) or raw.strip()

    try:
        data: Any = json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(f"GapFill: could not parse analyzer JSON ({type(exc).__name__}): {exc}; raw[:200]={raw[:200]!r}")
        return []

    if not isinstance(data, dict):
        logger.warning(f"GapFill: analyzer output was not a dict, got {type(data).__name__}")
        return []

    gaps_raw = data.get("gaps", [])
    if not isinstance(gaps_raw, list):
        logger.warning(f"GapFill: 'gaps' field was not a list, got {type(gaps_raw).__name__}")
        return []

    gaps: list[dict[str, str]] = []
    for item in gaps_raw:
        if not isinstance(item, dict):
            continue
        gap_text = str(item.get("gap", "")).strip()
        search_query = str(item.get("search_query", "") or gap_text).strip()
        why_matters = str(item.get("why_matters", "")).strip()
        if not gap_text or not search_query:
            continue
        gaps.append({"gap": gap_text, "search_query": search_query, "why_matters": why_matters})

    if max_gaps is not None:
        return gaps[:max_gaps]
    return gaps


async def _run_analyzer(
    question: MetaculusQuestion,
    first_pass_research: str,
    *,
    is_benchmarking: bool,
) -> list[dict[str, str]]:
    """Call the analyzer Gemini model (no grounding) to identify gaps.

    Uses `response_mime_type="application/json"` so the model returns valid JSON
    directly rather than prose-wrapped JSON, which makes `_parse_gap_list` more
    reliable and documents the "no grounding" contract explicitly.
    """
    client = build_gemini_client()
    prompt = gap_fill_analyzer_prompt(
        question_text=question.question_text,
        resolution_criteria=getattr(question, "resolution_criteria", None),
        fine_print=getattr(question, "fine_print", None),
        first_pass_research=first_pass_research,
        is_benchmarking=is_benchmarking,
        max_gaps=GAP_FILL_MAX_GAPS,
    )
    config = genai_types.GenerateContentConfig(response_mime_type="application/json")
    logger.info(f"GapFill: calling analyzer {GAP_FILL_ANALYZER_MODEL} for gap identification")
    response = await asyncio.wait_for(
        client.aio.models.generate_content(model=GAP_FILL_ANALYZER_MODEL, contents=prompt, config=config),
        timeout=GAP_FILL_ANALYZER_TIMEOUT,
    )
    raw_text: str = getattr(response, "text", "") or ""
    gaps = _parse_gap_list(raw_text, max_gaps=GAP_FILL_MAX_GAPS)
    logger.info(f"GapFill: analyzer returned {len(gaps)} gap(s)")
    return gaps


async def _resolve_single_gap(
    gap: dict[str, str],
    question_text: str,
    *,
    is_benchmarking: bool,
) -> str:
    """Run one grounded Gemini search for a single gap.

    Raises on SDK errors — the caller uses ``asyncio.gather(..., return_exceptions=True)``
    so one failure doesn't kill the rest.
    """
    prompt = gap_fill_search_prompt(
        gap=gap["gap"],
        search_query=gap["search_query"],
        question_text=question_text,
        is_benchmarking=is_benchmarking,
    )
    return await invoke_gemini_grounded(prompt)


async def run_gap_fill_pass(
    question: MetaculusQuestion,
    first_pass_research: str,
    *,
    is_benchmarking: bool = False,
) -> str:
    """Identify and resolve factual gaps in first-pass research.

    Two-stage flow:
    1. Analyzer call (Gemini 3 Flash, no grounding) → JSON list of up to
       ``GAP_FILL_MAX_GAPS`` gaps.
    2. Parallel grounded Gemini searches, one per gap, via ``asyncio.gather``.

    Never raises. Returns "" on any upstream failure (missing API key, timeout,
    SDK error, network error), logging type + message. This is a deliberate
    blanket soft-fail because the gap-fill pass is an optional enrichment layer
    and a forecast with only first-pass research is strictly better than no
    forecast at all; the `research.strip()` guard at the call site already
    ensures we never swallow a first-pass failure here.
    """
    gaps: list[dict[str, str]] = []
    try:
        gaps = await _run_analyzer(question, first_pass_research, is_benchmarking=is_benchmarking)
    except _GAP_FILL_SOFT_FAIL_EXCEPTIONS as exc:
        logger.warning(f"GapFill: analyzer failed ({type(exc).__name__}): {exc}")

    if not gaps:
        # A soft-fail here is already an async no-op; give the scheduler a
        # checkpoint so flake8-async (ASYNC910) is satisfied on every path.
        await asyncio.sleep(0)
        return ""

    search_tasks = [_resolve_single_gap(g, question.question_text, is_benchmarking=is_benchmarking) for g in gaps]
    # return_exceptions=True captures per-gap failures so one SDK error
    # doesn't take the whole addendum down.
    results = await asyncio.gather(*search_tasks, return_exceptions=True)

    sections: list[str] = []
    for idx, (gap, res) in enumerate(zip(gaps, results), start=1):
        if isinstance(res, BaseException):
            logger.warning(f"GapFill: gap #{idx} search failed ({type(res).__name__}): {res}")
            continue
        result_text = res
        if not result_text or not result_text.strip():
            continue
        why = gap.get("why_matters", "").strip()
        why_line = f"_Why it matters: {why}_\n\n" if why else ""
        sections.append(f"### Gap {idx}: {gap['gap']}\n\n{why_line}{result_text}")

    if not sections:
        return ""

    logger.info(f"GapFill: produced addendum from {len(sections)}/{len(gaps)} gap resolutions")
    return "\n\n".join(sections)
