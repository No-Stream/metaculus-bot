"""The AskNews-only summarizer pass, and its soft-fail back to the raw articles.

Split out of ``orchestrator.py``: AskNews is the one provider that returns raw article
markdown rather than LLM-written prose, so it is the one provider whose output gets a
second LLM pass before it reaches a forecaster. Everything that pass needs — the
prompt call, the narrow transient-exception set it is allowed to degrade on, and the
three-destination accounting of a degradation — lives here.

``ResearchOrchestrator`` mixes in ``AskNewsSummarization``, so
``ResearchOrchestrator._summarize_asknews`` stays patchable through the class and the
``summarizer_failure_count`` the degradation line reads stays on the orchestrator.
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


class AskNewsSummarization:
    """Mixin: compresses raw AskNews articles into an analyst briefing.

    The two members below are DECLARATIONS for the type checker, not defaults:
    ``ResearchOrchestrator`` owns the summarizer LLM and the soft-fail counter, and
    this mixin is never instantiated on its own.
    """

    _summarizer_llm: GeneralLlm
    summarizer_failure_count: int

    def _degraded_to_raw_articles(self, question: MetaculusQuestion, research: str, reason: str) -> str:
        """Return the raw articles under a visible banner, counting the soft-fail.

        Three destinations, because the loss was invisible in all three before
        2026-07-26: the forecaster sees the banner in its research bundle, CI sees
        ``summarizer_failures`` in the end-of-run degradation line, and the published
        comment / research archive see a ``summarizer`` source loss on the AskNews
        diagnostics line (whose ``status`` is computed from POST-summarizer text and
        therefore still reads ``ok``).
        """
        self.summarizer_failure_count += 1
        record_provider_detail(
            getattr(question, "id_of_question", None),
            "asknews",
            {"sources": {"summarizer": f"error({reason})"}},
        )
        return f"{SUMMARIZER_SOFT_FAIL_BANNER}\n\n{research}"

    async def _summarize_asknews(self, question: MetaculusQuestion, research: str) -> str:
        """Compress raw AskNews article markdown into an analyst briefing.

        Only AskNews output flows here — it's the one provider that returns raw
        article text rather than LLM-written prose. Soft-fails to the raw input
        (under a banner, see _degraded_to_raw_articles) so a summarizer hiccup never
        drops the news entirely.
        """
        if not research.strip():
            return research
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
                lambda: self._summarizer_llm.invoke(prompt),
                wall_timeout=SUMMARIZER_WALL_TIMEOUT,
                label="asknews_summarizer",
            )
        except _SUMMARIZER_TRANSIENT_EXCEPTIONS as exc:
            logger.warning(
                "AskNews summarization failed (%s); using raw articles under a degradation banner",
                type(exc).__name__,
            )
            return self._degraded_to_raw_articles(question, research, type(exc).__name__)
        if not summary.strip():
            logger.warning("AskNews summarization returned blank output; using raw articles under a banner")
            return self._degraded_to_raw_articles(question, research, "blank_output")
        return summary
