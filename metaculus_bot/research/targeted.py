"""Targeted research pipeline for conditional stacking.

When base forecaster models disagree significantly, this module:
1. Extracts the crux of disagreement using a cheap analyzer model.
2. Runs a targeted web search via OpenAI native search to resolve it.

Also provides ``run_gap_fill_pass`` — an always-on second-pass that runs after
first-pass research. It identifies factual gaps in the first pass and resolves
each via a parallel OpenAI native web search (OpenRouter, donated-key billed).
"""

import asyncio
import json
import logging
from typing import Any

from forecasting_tools import GeneralLlm, MetaculusQuestion

from metaculus_bot.constants import (
    CRUX_SOFT_DEADLINE,
    GAP_FILL_ANALYZER_MODEL,
    GAP_FILL_ANALYZER_TIMEOUT,
    GAP_FILL_ANALYZER_WALL_TIMEOUT,
    GAP_FILL_MAX_GAPS,
    GAP_FILL_RESOLVER_MODEL,
    GAP_FILL_RESOLVER_REASONING_EFFORT,
    NATIVE_SEARCH_WALL_TIMEOUT,
)
from metaculus_bot.llm_retry import invoke_with_broad_retry, invoke_with_transient_retry
from metaculus_bot.prompts import (
    disagreement_crux_prompt,
    gap_fill_analyzer_prompt,
    gap_fill_search_prompt,
    targeted_search_prompt,
)
from metaculus_bot.research.providers import build_native_search_llm
from metaculus_bot.research.raw_log import record_raw_research
from metaculus_bot.structured_output_schema import extract_first_balanced_braces, extract_json_block

__all__ = [
    "extract_disagreement_crux",
    "run_gap_fill_pass",
    "run_targeted_search",
]

logger: logging.Logger = logging.getLogger(__name__)

# Broad by design; the pass soft-fails to "" and CancelledError escapes. See docs/research.md "v1 implementation notes".
_GAP_FILL_SOFT_FAIL_EXCEPTIONS: tuple[type[BaseException], ...] = (Exception,)


async def extract_disagreement_crux(
    analyzer_llm: GeneralLlm,
    question_text: str,
    base_prediction_texts: list[str],
) -> str:
    """Identify the core factual disagreement across base forecaster analyses.

    Args:
        analyzer_llm: A cheap, low-effort model used for extraction only (the
            DISAGREEMENT_ANALYZER_LLM slot in llm_configs.py).
        question_text: The full question text being forecasted.
        base_prediction_texts: Reasoning texts from base models (already stripped of model tags).

    Returns:
        A short string describing the factual crux of disagreement.
    """
    prompt = disagreement_crux_prompt(question_text, base_prediction_texts)
    logger.info(f"Extracting disagreement crux from {len(base_prediction_texts)} forecaster analyses")
    # A 30s-gated retry on an allowed_tries=1 analyzer. See docs/research.md "v1 implementation notes".
    crux = await invoke_with_broad_retry(
        lambda: analyzer_llm.invoke(prompt), wall_timeout=CRUX_SOFT_DEADLINE, label="disagreement_crux"
    )
    logger.info(f"Disagreement crux extracted: {len(crux)} chars")
    return crux


async def run_targeted_search(crux: str, question_text: str, *, is_benchmarking: bool = False) -> str:
    """Run a targeted web search to resolve a specific factual disagreement.

    Uses OpenAI native web search via OpenRouter (`build_native_search_llm`) to
    find current, authoritative information about the identified crux.

    Args:
        crux: The factual question(s) driving forecaster disagreement.
        question_text: The full question text being forecasted.
        is_benchmarking: If True, excludes prediction market data to avoid data leakage.

    Returns:
        Search results with inline citations addressing the crux.
    """
    llm = build_native_search_llm(role="targeted_search")
    prompt = targeted_search_prompt(crux, question_text, is_benchmarking=is_benchmarking)
    logger.info(
        f"Running targeted search via {llm.model} for crux: "
        f"{crux[:100]}..."  # HARNESS-SCAN-EXEMPT-subsampling: a log-line preview, not a data reduction
    )
    # The wall is the hard cap; litellm's per-request timeout is not. See docs/research.md "v1 implementation notes".
    result = await invoke_with_transient_retry(
        lambda: llm.invoke(prompt), wall_timeout=NATIVE_SEARCH_WALL_TIMEOUT, label="targeted_search"
    )
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

    # Fenced first, then a balanced-brace scan for trailing prose. See docs/research.md "v1 implementation notes".
    fenced = extract_json_block(raw)
    stripped = fenced if fenced is not None else extract_first_balanced_braces(raw) or raw.strip()

    try:
        data: Any = json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            f"GapFill: could not parse analyzer JSON ({type(exc).__name__}): {exc}; "
            f"raw[:200]={raw[:200]!r}"  # HARNESS-SCAN-EXEMPT-subsampling: a log-line preview, not a data reduction
        )
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
    """Call the analyzer LLM (no grounding) to identify gaps.

    Runs gpt-5.6-terra at low effort via OpenRouter (with donated-key
    fallback). The analyzer is non-grounded so it doesn't need Google's search
    index; the task is gap decomposition (not deep judgment) under a tight
    soft-fail wall cap, so terra-low is the latency-safe tier.

    Without google-genai's response_mime_type=application/json, the prompt asks
    for ```json fenced output and _parse_gap_list handles fence stripping +
    balanced-brace fallback for trailing commentary.
    """
    from metaculus_bot.fallback_openrouter import (  # noqa: PLC0415  # late import: tests patch this at its source module
        build_llm_with_openrouter_fallback,
    )

    llm = build_llm_with_openrouter_fallback(
        model=GAP_FILL_ANALYZER_MODEL,
        role="gap_fill_analyzer",
        reasoning={"effort": "low"},
        # Reasoning models take the provider default. See docs/research.md "v1 implementation notes".
        temperature=None,
        timeout=GAP_FILL_ANALYZER_TIMEOUT,
        allowed_tries=1,
    )
    prompt = gap_fill_analyzer_prompt(
        question_text=question.question_text,
        resolution_criteria=question.resolution_criteria,
        fine_print=question.fine_print,
        first_pass_research=first_pass_research,
        is_benchmarking=is_benchmarking,
        max_gaps=GAP_FILL_MAX_GAPS,
        # A "no coverage of candidate X" gap needs the ballot (q44952). See docs/research.md "v1 implementation notes".
        options=getattr(question, "options", None),
    )
    logger.info(f"GapFill: calling analyzer {GAP_FILL_ANALYZER_MODEL} for gap identification")
    # The wall has headroom over the per-request timeout: 135s vs 120s. See docs/research.md "v1 implementation notes".
    raw_text = await invoke_with_transient_retry(
        lambda: llm.invoke(prompt), wall_timeout=GAP_FILL_ANALYZER_WALL_TIMEOUT, label="gap_fill_analyzer"
    )
    gaps = _parse_gap_list(raw_text, max_gaps=GAP_FILL_MAX_GAPS)
    logger.info(f"GapFill: analyzer returned {len(gaps)} gap(s)")
    return gaps


async def _resolve_single_gap(
    gap: dict[str, str],
    question: MetaculusQuestion,
    *,
    is_benchmarking: bool,
) -> str:
    """Run one OpenAI native web search (via OpenRouter) for a single gap.

    Migrated 2026-06-25 off direct-Google grounded Gemini (google-genai, personal
    GOOGLE_API_KEY) to native search on the Metaculus-donated key — this is the
    dominant cost-saving change since the resolver fans out up to GAP_FILL_MAX_GAPS
    calls per question. Runs GAP_FILL_RESOLVER_MODEL at GAP_FILL_RESOLVER_REASONING_EFFORT
    (the same low as the main native_search provider): the workers run in parallel, so
    latency is the slowest call, not the sum.

    Raises on SDK/OpenRouter errors — the caller uses ``asyncio.gather(..., return_exceptions=True)``
    so one failure doesn't kill the rest.
    """
    prompt = gap_fill_search_prompt(
        gap=gap["gap"],
        search_query=gap["search_query"],
        question_text=question.question_text,
        resolution_criteria=question.resolution_criteria,
        fine_print=question.fine_print,
        is_benchmarking=is_benchmarking,
    )
    llm = build_native_search_llm(
        GAP_FILL_RESOLVER_MODEL, reasoning_effort=GAP_FILL_RESOLVER_REASONING_EFFORT, role="gap_fill_resolver"
    )
    # The same shared wall as native_search; a hard cap either way. See docs/research.md "v1 implementation notes".
    return await invoke_with_transient_retry(
        lambda: llm.invoke(prompt), wall_timeout=NATIVE_SEARCH_WALL_TIMEOUT, label="gap_fill_resolver"
    )


async def run_gap_fill_pass(
    question: MetaculusQuestion,
    first_pass_research: str,
    *,
    is_benchmarking: bool = False,
) -> str:
    """Identify and resolve factual gaps in first-pass research.

    Two-stage flow:
    1. Analyzer call (GAP_FILL_ANALYZER_MODEL at low effort via OpenRouter, no grounding) →
       JSON list of up to ``GAP_FILL_MAX_GAPS`` gaps.
    2. Parallel OpenAI native web searches (GAP_FILL_RESOLVER_MODEL at
       GAP_FILL_RESOLVER_REASONING_EFFORT via OpenRouter), one per gap, via ``asyncio.gather``.

    Never raises. Returns "" on any upstream failure (missing API key, timeout,
    SDK error, network error), logging type + message. This is a deliberate
    blanket soft-fail because the gap-fill pass is an optional enrichment layer
    and a forecast with only first-pass research is strictly better than no
    forecast at all; the `research.strip()` guard at the call site already
    ensures we never swallow a first-pass failure here.
    """
    gaps: list[dict[str, str]] = []
    qid = getattr(question, "id_of_question", None)
    try:
        gaps = await _run_analyzer(question, first_pass_research, is_benchmarking=is_benchmarking)
    except _GAP_FILL_SOFT_FAIL_EXCEPTIONS as exc:
        # A dead analyzer looks exactly like a question with no gaps. See docs/research.md "v1 implementation notes".
        logger.warning(f"GAP_FILL_ANALYZER_FAILED: question={qid} error={type(exc).__name__} detail={exc}")

    if not gaps:
        # A scheduler checkpoint on the no-op path, for ASYNC910. See docs/research.md "v1 implementation notes".
        await asyncio.sleep(0)
        return ""

    search_tasks = [_resolve_single_gap(g, question, is_benchmarking=is_benchmarking) for g in gaps]
    # One SDK error must not take the whole addendum down. See docs/research.md "v1 implementation notes".
    results = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Exceptions serialize to their str via the logger's encoder. See docs/research.md "v1 implementation notes".
    record_raw_research(
        qid=getattr(question, "id_of_question", None),
        provider="gap_fill",
        payload={"gaps": gaps, "results": results},
    )

    sections: list[str] = []
    for idx, (gap, res) in enumerate(zip(gaps, results, strict=True), start=1):
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
