"""The AskNews-only summarizer pass, and its soft-fail back to the raw articles.

Split out of ``orchestrator.py``: AskNews is the one provider that returns raw article
markdown rather than LLM-written prose, so it is the one provider whose output gets a
second LLM pass before it reaches a forecaster. Everything that pass needs — the
prompt call, the narrow transient-exception set it is allowed to degrade on, and the
per-source loss token a degradation records — lives here.

``summarize_asknews`` RETURNS its soft-fail reason rather than counting it, because the
counter the end-of-run degradation line reads lives on the orchestrator; the thin
``ResearchOrchestrator._summarize_asknews`` wrapper does the bump (and stays the class
that tests patch).
"""

import asyncio
import logging

import openai
from forecasting_tools import GeneralLlm
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import SUMMARIZER_WALL_TIMEOUT
from metaculus_bot.llm_retry import invoke_with_broad_retry
from metaculus_bot.prompts import SUMMARIZER_SOFT_FAIL_BANNER, asknews_summarizer_prompt
from metaculus_bot.research.provider_diagnostics import record_provider_detail

logger = logging.getLogger(__name__)

# Summarizer failures that legitimately soft-fail to the raw AskNews articles:
# transient LLM-provider hiccups (``openai.APIError`` is the common base for
# litellm's connection/timeout/rate-limit/service-unavailable wrappers) and
# asyncio timeouts. Anything outside this set — a prompt-construction bug, an
# AttributeError from a refactor, a credential-routing regression — is a real
# bug and must propagate rather than silently degrade every forecast's research.
_SUMMARIZER_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    openai.APIError,
)


def _degraded_to_raw_articles(question: MetaculusQuestion, research: str, reason: str) -> str:
    """Return the raw articles under a visible banner, recording the per-source loss.

    Three destinations, because the loss was invisible in all three before
    2026-07-26: the forecaster sees the banner in its research bundle, CI sees
    ``summarizer_failures`` in the end-of-run degradation line (the caller bumps that
    counter off the reason this function's caller returns), and the published comment /
    research archive see a ``summarizer`` source loss on the AskNews diagnostics line
    (whose ``status`` is computed from POST-summarizer text and therefore still reads
    ``ok``).
    """
    record_provider_detail(
        getattr(question, "id_of_question", None),
        "asknews",
        {"sources": {"summarizer": f"error({reason})"}},
    )
    return f"{SUMMARIZER_SOFT_FAIL_BANNER}\n\n{research}"


async def summarize_asknews(
    question: MetaculusQuestion, research: str, *, summarizer_llm: GeneralLlm
) -> tuple[str, str | None]:
    """Compress raw AskNews article markdown into an analyst briefing.

    Returns ``(text, soft_fail_reason)``, the reason being ``None`` on every path
    that did not degrade — including the empty-input path, where nothing was lost.
    The caller owns the counter, so the reason has to ride back rather than be
    counted here.

    Only AskNews output flows here — it's the one provider that returns raw
    article text rather than LLM-written prose. Soft-fails to the raw input
    (under a banner, see _degraded_to_raw_articles) so a summarizer hiccup never
    drops the news entirely.
    """
    if not research.strip():
        return research, None
    # Real API-fetched questions always populate open_time; a missing value
    # means broken upstream data and the forecaster prompts assert on it
    # anyway (_forecasting_window_str), so fail loudly here too.
    assert question.open_time is not None, "question.open_time is required for window-stamping"
    # Prompt text lives in prompts.py (asknews_summarizer_prompt) so it shares
    # the source-tier tag vocabulary with web_research_prompt.
    prompt = asknews_summarizer_prompt(
        question_text=question.question_text,
        resolution_criteria=question.resolution_criteria or "",
        fine_print=question.fine_print or "",
        open_date=question.open_time.strftime("%Y-%m-%d"),
        research=research,
        # The MC ballot (None on other types): the relevance screen needs the candidate
        # names to judge which articles bear on the resolution (q44952).
        options=getattr(question, "options", None),
    )
    try:
        # Broad retry under the TRANSIENT_RETRY_MAX_ELAPSED_S elapsed gate
        # (SUMMARIZER_LLM is allowed_tries=1 in llm_configs.py): recovers a fast
        # blip / empty-response while obeying the universal "don't retry a slow
        # failure" deadline rule. Adds the wall-clock cap this call previously
        # lacked. A slow/permanent failure still propagates to the soft-fail
        # below (raw AskNews articles).
        summary = await invoke_with_broad_retry(
            lambda: summarizer_llm.invoke(prompt),
            wall_timeout=SUMMARIZER_WALL_TIMEOUT,
            label="asknews_summarizer",
        )
    except _SUMMARIZER_TRANSIENT_EXCEPTIONS as exc:
        logger.warning(
            "AskNews summarization failed (%s); using raw articles under a degradation banner",
            type(exc).__name__,
        )
        reason = type(exc).__name__
        return _degraded_to_raw_articles(question, research, reason), reason
    if not summary.strip():
        logger.warning("AskNews summarization returned blank output; using raw articles under a banner")
        return _degraded_to_raw_articles(question, research, "blank_output"), "blank_output"
    return summary, None
