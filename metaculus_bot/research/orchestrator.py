"""Research orchestration extracted from TemplateForecaster.

Encapsulates provider selection, parallel execution, caching, gap-fill, and
fallback logic. The TemplateForecaster delegates run_research to an instance of
this class. AskNews output is summarized into an analyst briefing inline (it's
the only provider that returns raw article text rather than LLM-written prose);
all other providers pass through raw.
"""

import asyncio
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict

import openai
from forecasting_tools import GeneralLlm, SmartSearcher, clean_indents
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.api_key_utils import get_openrouter_api_key
from metaculus_bot.constants import (
    DEFAULT_MAX_CONCURRENT_RESEARCH,
    EXA_API_KEY_ENV,
    FINANCIAL_DATA_ENABLED_ENV,
    GAP_FILL_ENABLED_ENV,
    GAP_FILL_MIN_RESEARCH_CHARS,
    GAP_FILL_V2_ENABLED_ENV,
    GEMINI_SEARCH_ENABLED_ENV,
    GEMINI_SEARCH_MODEL_ENV,
    NATIVE_SEARCH_ENABLED_ENV,
    NATIVE_SEARCH_MODEL_ENV,
    OPENROUTER_API_KEY_ENV,
    PERPLEXITY_API_KEY_ENV,
    PREDICTION_MARKETS_ENABLED_ENV,
    RESOLUTION_SOURCE_ENABLED_ENV,
    SUMMARIZER_WALL_TIMEOUT,
    env_flag_enabled,
)
from metaculus_bot.llm_retry import invoke_with_broad_retry
from metaculus_bot.prompts import asknews_summarizer_prompt
from metaculus_bot.research.provider_diagnostics import (
    SUCCEEDED_STATUSES,
    ProviderResult,
    format_provider_diagnostics_block,
)
from metaculus_bot.research.providers import (
    ResearchCallable,
    choose_provider_with_name,
    native_search_provider,
)

_PROVIDER_ERROR_MESSAGE_MAX_CHARS = 300

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

_LEADING_HEADING_RE = re.compile(r"^(#{1,2})(?=\s|$)", re.MULTILINE)


def _demote_inner_headings(text: str) -> str:
    """Shift any in-body h1/h2 heading down by two levels (h1→h3, h2→h4).

    Provider headers are h2 (``_provider_header``). If an LLM-written body emits
    its own h1/h2 (e.g. ``# Historical Context``), it sits at/above the provider
    header and breaks the framework's ``report_sections_to_markdown``
    renormalization, which degrades to the ugly ``[Hashtag]`` fallback. Demoting
    keeps every provider header the minimum-level section.
    """
    return _LEADING_HEADING_RE.sub(lambda m: "##" + m.group(1), text)


class ResearchOrchestrator:
    """Manages research provider selection, parallel execution, caching, and gap-fill."""

    def __init__(
        self,
        *,
        default_llm: GeneralLlm,
        summarizer_llm: GeneralLlm,
        custom_provider: ResearchCallable | None = None,
        research_cache: dict[int, str] | None = None,
        is_benchmarking: bool = False,
        allow_research_fallback: bool = True,
        max_concurrent_research: int = DEFAULT_MAX_CONCURRENT_RESEARCH,
        research_sink: Callable[..., None] | None = None,
    ) -> None:
        self._default_llm = default_llm
        self._summarizer_llm = summarizer_llm
        self._custom_provider = custom_provider
        self._research_cache = research_cache
        self._is_benchmarking = is_benchmarking
        self._allow_research_fallback = allow_research_fallback
        self._concurrency_limiter = asyncio.Semaphore(max_concurrent_research)
        self._research_sink = research_sink
        self.timeout_count: int = 0

    async def run_research(self, question: MetaculusQuestion) -> str:
        cache_key, cached = self._lookup_research_cache(question)
        if cached is not None:
            logger.info(f"Using cached research for question {cache_key}")
            return cached

        async with self._concurrency_limiter:
            cache_key, cached = self._lookup_research_cache(question)
            if cached is not None:
                logger.info(f"Using cached research for question {cache_key} (double-check)")
                return cached

            providers = self._select_research_providers()
            provider_names = [name for _, name in providers]
            logger.info(f"Using research providers: {provider_names}")

            research, provider_results = await self._run_providers_parallel(question, providers)

            # Gap-fill v1 and v2 both consume the pre-gap-fill bundle and run
            # CONCURRENTLY in one gather (plan doc §2: research-phase wall-clock
            # is max(v1, v2), not the sum — v2's 540s deadline fits inside v1's
            # worst-case envelope only under this parallelism). Consequence: the
            # v2 driver's brief sees the bundle WITHOUT v1's addendum. v2's
            # section appends after v1's, both before the diagnostics block.
            gap_fill_v1_active = (
                env_flag_enabled(GAP_FILL_ENABLED_ENV) and len(research.strip()) >= GAP_FILL_MIN_RESEARCH_CHARS
            )
            gap_fill_v2_active = env_flag_enabled(GAP_FILL_V2_ENABLED_ENV)
            gap_fill_v2_payload: dict | None = None

            if gap_fill_v1_active or gap_fill_v2_active:
                try:
                    from metaculus_bot.research.agentic_gap_fill import (
                        run_gap_fill_v2,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
                    )
                    from metaculus_bot.research.targeted import (
                        run_gap_fill_pass,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
                    )

                    def _capture_gap_fill_v2(payload: dict) -> None:
                        nonlocal gap_fill_v2_payload
                        gap_fill_v2_payload = payload

                    async def _run_v1() -> str:
                        if not gap_fill_v1_active:
                            return ""
                        return await run_gap_fill_pass(question, research, is_benchmarking=self._is_benchmarking)

                    async def _run_v2() -> str:
                        if not gap_fill_v2_active:
                            return ""
                        return await run_gap_fill_v2(
                            question,
                            research,
                            is_benchmarking=self._is_benchmarking,
                            archive_sink=_capture_gap_fill_v2,
                        )

                    addendum, v2_findings = await asyncio.gather(_run_v1(), _run_v2())
                except Exception:  # HARNESS-SCAN-EXEMPT-broad-except — gap-fill is optional; a failure (import error, unhandled raise) must never kill the forecast
                    logger.exception("Gap-fill stage failed; proceeding without it")
                    addendum, v2_findings = "", ""
                if addendum:
                    research = f"{research}\n\n---\n\n## Targeted Gap-Fill (second pass)\n\n{addendum}"
                if v2_findings:
                    # v2_findings carries its own "## Agentic Research Findings"
                    # header (render_findings) — distinct from v1's section.
                    research = f"{research}\n\n---\n\n{v2_findings}"

            gap_fill_used = "## Targeted Gap-Fill (second pass)" in research

            # Block ordering in the returned text (and therefore the comment):
            # providers -> gap-fill -> provider diagnostics. Appended last so the
            # diagnostics never perturb the gap-fill threshold/detection above.
            diagnostics_block = format_provider_diagnostics_block(provider_results)
            if diagnostics_block:
                research = f"{research}\n\n{diagnostics_block}"

            self._store_research_cache(cache_key, research)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")

            if self._research_sink is not None:
                qid = getattr(question, "id_of_question", None)
                if qid is not None:
                    try:
                        # provider_results is the authoritative per-provider outcome;
                        # providers_used is kept only for legacy archive readers.
                        self._research_sink(
                            qid=qid,
                            page_url=question.page_url,
                            question_text=question.question_text,
                            research_text=research,
                            providers_used=provider_names,
                            gap_fill_used=gap_fill_used,
                            provider_results=[asdict(r) for r in provider_results],
                            providers_attempted=provider_names,
                            providers_succeeded=[r.name for r in provider_results if r.status in SUCCEEDED_STATUSES],
                            gap_fill_v2=gap_fill_v2_payload,
                        )
                    except (
                        Exception
                    ):  # HARNESS-SCAN-EXEMPT-broad-except — archive write is best-effort; never blocks the forecast
                        logger.exception("Research sink failed for qid=%d; continuing", qid)

            return research

    async def _summarize_asknews(self, question: MetaculusQuestion, research: str) -> str:
        """Compress raw AskNews article markdown into an analyst briefing.

        Only AskNews output flows here — it's the one provider that returns raw
        article text rather than LLM-written prose. Soft-fails to the raw input
        so a summarizer hiccup never drops the news entirely.
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
        )
        try:
            # Broad, 30s-gated retry (SUMMARIZER_LLM is allowed_tries=1 in
            # llm_configs.py): recovers a fast blip / empty-response while obeying
            # the universal "no retry after 30s" deadline rule. Adds the wall-clock
            # cap this call previously lacked. A slow/permanent failure still
            # propagates to the soft-fail below (raw AskNews articles).
            summary = await invoke_with_broad_retry(
                lambda: self._summarizer_llm.invoke(prompt),
                wall_timeout=SUMMARIZER_WALL_TIMEOUT,
                label="asknews_summarizer",
            )
        except _SUMMARIZER_TRANSIENT_EXCEPTIONS as exc:
            logger.warning("AskNews summarization failed (%s); using raw articles", type(exc).__name__)
            return research
        if not summary.strip():
            logger.warning("AskNews summarization returned blank output; using raw articles")
            return research
        return summary

    def _lookup_research_cache(self, question: MetaculusQuestion) -> tuple[int | None, str | None]:
        cache_key = getattr(question, "id_of_question", None)
        if not self._is_benchmarking or self._research_cache is None or cache_key is None:
            return cache_key, None
        return cache_key, self._research_cache.get(cache_key)

    def _store_research_cache(self, cache_key: int | None, research: str) -> None:
        if not self._is_benchmarking or self._research_cache is None or cache_key is None:
            return
        self._research_cache[cache_key] = research
        logger.info(f"Cached research for question {cache_key}")

    def _select_research_provider(self) -> tuple[ResearchCallable, str]:
        if self._custom_provider is not None:
            return self._custom_provider, "custom"

        provider, provider_name = choose_provider_with_name(
            self._default_llm,
            exa_callback=self._call_exa_smart_searcher,
            perplexity_callback=self._call_perplexity,
            openrouter_callback=self._call_perplexity_openrouter,
            is_benchmarking=self._is_benchmarking,
        )
        return provider, provider_name

    def _select_research_providers(self) -> list[tuple[ResearchCallable, str]]:
        providers: list[tuple[ResearchCallable, str]] = []

        primary, primary_name = self._select_research_provider()
        if primary_name != "none":
            providers.append((primary, primary_name))

        if env_flag_enabled(NATIVE_SEARCH_ENABLED_ENV):
            model = os.getenv(NATIVE_SEARCH_MODEL_ENV)
            providers.append(
                (
                    native_search_provider(model, is_benchmarking=self._is_benchmarking),
                    "native_search",
                )
            )

        if env_flag_enabled(GEMINI_SEARCH_ENABLED_ENV):
            from metaculus_bot.research.gemini_search import (
                gemini_search_provider,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
            )

            gemini_model = os.getenv(GEMINI_SEARCH_MODEL_ENV)
            providers.append(
                (
                    gemini_search_provider(gemini_model, is_benchmarking=self._is_benchmarking),
                    "gemini_search",
                )
            )

        if env_flag_enabled(FINANCIAL_DATA_ENABLED_ENV):
            from metaculus_bot.research.financial_data import (
                financial_data_provider,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
            )

            providers.append((financial_data_provider(), "financial_data"))

        if env_flag_enabled(PREDICTION_MARKETS_ENABLED_ENV):
            from metaculus_bot.research.prediction_market import (
                prediction_market_provider,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
            )

            providers.append((prediction_market_provider(is_benchmarking=self._is_benchmarking), "prediction_market"))

        if env_flag_enabled(RESOLUTION_SOURCE_ENABLED_ENV):
            from metaculus_bot.research.resolution_source import (
                resolution_source_provider,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
            )

            providers.append((resolution_source_provider(is_benchmarking=self._is_benchmarking), "resolution_source"))

        if not providers:

            async def _empty(_: MetaculusQuestion) -> str:
                return ""

            providers.append((_empty, "none"))

        return providers

    async def _run_providers_parallel(
        self,
        question: MetaculusQuestion,
        providers: list[tuple[ResearchCallable, str]],
    ) -> tuple[str, list[ProviderResult]]:
        from metaculus_bot.research.providers import (
            is_asknews_subscription_error,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
        )

        async def _run_one(provider: ResearchCallable, name: str) -> tuple[str, ProviderResult]:
            started = time.monotonic()
            try:
                used_fallback = False
                if name == "asknews" and self._allow_research_fallback:
                    raw, used_fallback = await self._fetch_research_with_fallback(question, provider, name)
                else:
                    raw = await provider(question)
                # AskNews returns raw article markdown (no LLM prose); summarize it
                # into an analyst briefing. Every other provider already emits
                # LLM-written prose (native search, Gemini, Perplexity, Exa) or
                # deterministic tables (financial, prediction markets), so they
                # pass through raw — no lossy second-pass summarization. When
                # AskNews fails and we fall back to Perplexity/Exa, that fallback
                # is already prose, so skip summarization too.
                if name == "asknews" and not used_fallback:
                    raw = await self._summarize_asknews(question, raw)
                latency_ms = int((time.monotonic() - started) * 1000)
                has_output = bool(raw and raw.strip())
                if not has_output:
                    status = "empty"
                elif used_fallback:
                    status = "fallback"
                else:
                    status = "ok"
                result = ProviderResult(
                    name=name,
                    status=status,
                    chars=len(raw) if has_output else 0,
                    latency_ms=latency_ms,
                )
                return (raw, result)
            except Exception as e:  # HARNESS-SCAN-EXEMPT-broad-except — converted to a ProviderResult(status=errored/inactive); one provider failing never kills the research phase
                latency_ms = int((time.monotonic() - started) * 1000)
                if name == "asknews" and is_asknews_subscription_error(e):
                    status = "inactive"
                    logger.info(
                        "Research provider %s inactive (expected off-season): %s: %s",
                        name,
                        type(e).__name__,
                        e,
                    )
                else:
                    status = "errored"
                    self.timeout_count += 1
                    logger.warning(f"Research provider {name} failed ({type(e).__name__}): {e}")
                    from metaculus_bot.fallback_openrouter import (  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
                        _record_deprecation_if_matched,
                    )

                    _record_deprecation_if_matched(f"<provider:{name}>", str(e))
                result = ProviderResult(
                    name=name,
                    status=status,
                    chars=0,
                    latency_ms=latency_ms,
                    error_type=type(e).__name__,
                    error_message=str(e)[:_PROVIDER_ERROR_MESSAGE_MAX_CHARS],
                )
                return ("", result)

        tasks = [_run_one(p, n) for p, n in providers]
        results: list[tuple[str, ProviderResult]] = await asyncio.gather(*tasks)

        combined_parts = []
        provider_results: list[ProviderResult] = []
        for raw, provider_result in results:
            provider_results.append(provider_result)
            if raw and raw.strip():
                header = self._provider_header(provider_result.name)
                combined_parts.append(f"{header}\n{_demote_inner_headings(raw)}")

        combined = "\n\n---\n\n".join(combined_parts) if combined_parts else ""
        return combined, provider_results

    @staticmethod
    def _provider_header(name: str) -> str:
        headers = {
            "asknews": "## News Articles (AskNews)",
            "native_search": "## Web Research (Native Search)",
            "gemini_search": "## Web Research (Google Search via Gemini)",
            "financial_data": "## Financial & Economic Data",
            "prediction_market": "## Prediction Market Snapshot",
            "resolution_source": "## Resolution Source Snapshot",
            "exa": "## Web Research (Exa)",
            "perplexity": "## Web Research (Perplexity)",
            "openrouter": "## Web Research (OpenRouter)",
            "custom": "## Research (Custom)",
        }
        return headers.get(name, f"## Research ({name})")

    async def _fetch_research_with_fallback(
        self,
        question: MetaculusQuestion,
        provider: ResearchCallable,
        provider_name: str,
    ) -> tuple[str, bool]:
        """Return (research_text, used_fallback).

        used_fallback is True when the primary (AskNews) failed and a prose
        fallback provider (Perplexity/Exa) supplied the result instead — the
        caller uses this to skip AskNews summarization on already-prose output.
        """
        try:
            return (await provider(question), False)
        except Exception as exc:
            if self._allow_research_fallback and provider_name == "asknews":
                logger.warning(f"Primary research provider '{provider_name}' failed with {type(exc).__name__}: {exc}")
                fallback = await self._attempt_research_fallback(question.question_text)
                if fallback is not None:
                    return (fallback, True)
            raise

    async def _attempt_research_fallback(self, question_text: str) -> str | None:
        # Ordering intentionally differs from the primary selector
        # (choose_provider_with_name: AskNews -> Exa -> Perplexity -> OpenRouter).
        # This fallback only fires when AskNews (always the primary in prod) has
        # already failed, so AskNews is excluded. Among the remaining options we
        # prefer the Perplexity-via-OpenRouter route first (cheap, prose-returning,
        # routed through the donated-key wrapper), then direct Perplexity, then
        # Exa last (SmartSearcher spins up its own multi-search/LLM loop, the most
        # expensive path). The primary selector orders by index quality, not cost,
        # which is why the two lists diverge by design.
        try:
            if os.getenv(OPENROUTER_API_KEY_ENV):
                logger.info("Falling back to openrouter/perplexity for research")
                return await self._call_perplexity(question_text, use_open_router=True)
            if os.getenv(PERPLEXITY_API_KEY_ENV):
                logger.info("Falling back to Perplexity for research")
                return await self._call_perplexity(question_text, use_open_router=False)
            if os.getenv(EXA_API_KEY_ENV):
                logger.info("Falling back to Exa search for research")
                return await self._call_exa_smart_searcher(question_text)
        except Exception as fallback_exc:  # HARNESS-SCAN-EXEMPT-broad-except — best-effort fallback; logs and returns None so the primary error propagates
            logger.warning(f"Fallback research provider also failed: {type(fallback_exc).__name__}: {fallback_exc}")
        return None

    async def _call_perplexity(self, question: MetaculusQuestion | str, use_open_router: bool = True) -> str:
        question_text = question.question_text if isinstance(question, MetaculusQuestion) else question

        prediction_markets_instruction = (
            ""
            if self._is_benchmarking
            else (
                "In addition to news, briefly research prediction markets that are relevant to the question. "
                "(If there are no relevant prediction markets, simply skip reporting on this and "
                "DO NOT speculate what they would say.)"
            )
        )

        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            {prediction_markets_instruction}
            You DO NOT produce forecasts yourself; you must provide ALL relevant data to the superforecaster so they can make an expert judgment.

            Question:
            {question_text}
            """
        )
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning-pro"
        else:
            model_name = "perplexity/sonar-reasoning-pro"
        model = GeneralLlm(
            model=model_name,
            # temperature=None (not omitted): GeneralLlm injects temperature=0 otherwise;
            # defer to provider defaults. No top_p.
            temperature=None,
            api_key=get_openrouter_api_key(model_name) if model_name.startswith("openrouter/") else None,
        )
        return await model.invoke(prompt)

    async def _call_perplexity_openrouter(self, question: MetaculusQuestion) -> str:
        return await self._call_perplexity(question, use_open_router=True)

    async def _call_exa_smart_searcher(self, question: MetaculusQuestion | str) -> str:
        question_text = question.question_text if isinstance(question, MetaculusQuestion) else question
        searcher = SmartSearcher(
            # temperature ignored when model is a preconfigured GeneralLlm; None
            # keeps litellm from applying a sampling param on the fallback str path.
            model=self._default_llm,
            temperature=None,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question_text}"
        )
        return await searcher.invoke(prompt)
