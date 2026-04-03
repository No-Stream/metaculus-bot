"""Targeted research pipeline for conditional stacking.

When base forecaster models disagree significantly, this module:
1. Extracts the crux of disagreement using a cheap analyzer model.
2. Runs a targeted web search via Grok native search to resolve it.
"""

import logging

from forecasting_tools import GeneralLlm

from metaculus_bot.prompts import disagreement_crux_prompt, targeted_search_prompt
from metaculus_bot.research_providers import build_native_search_llm

__all__ = ["extract_disagreement_crux", "run_targeted_search"]

logger: logging.Logger = logging.getLogger(__name__)


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
