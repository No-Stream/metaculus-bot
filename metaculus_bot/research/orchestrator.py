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
    PERPLEXITY_WALL_TIMEOUT,
    PREDICTION_MARKETS_ENABLED_ENV,
    RESOLUTION_SOURCE_ENABLED_ENV,
    SUMMARIZER_WALL_TIMEOUT,
    TS_ANCHOR_ENABLED_ENV,
    env_flag_enabled,
)
from metaculus_bot.llm_retry import invoke_with_broad_retry, invoke_with_transient_retry
from metaculus_bot.prompts import (
    SUMMARIZER_SOFT_FAIL_BANNER,
    TS_ANCHOR_SECTION_HEADER,
    asknews_summarizer_prompt,
)
from metaculus_bot.research.provider_diagnostics import (
    SUCCEEDED_STATUSES,
    ProviderResult,
    format_provider_diagnostics_block,
    pop_provider_detail,
    record_provider_detail,
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
        # Comment-bound provider-diagnostics blocks, keyed by qid. run_research
        # returns forecaster-clean text; TemplateForecaster pops the block via
        # pop_provider_diagnostics when assembling the published comment.
        self._comment_diagnostics: dict[int, str] = {}
        # Per-run count of research-provider calls that FAILED — any exception, not
        # just timeouts (the generic failure branch in _run_one never inspects the
        # exception type). Excludes the expected off-season AskNews subscription
        # error, which reports status="inactive" and is not alertable.
        self.provider_failure_count: int = 0
        # Per-run count of AskNews summarizer soft-fails (transient LLM error or
        # blank output), each of which ships raw unscreened articles in place of the
        # analyst briefing. Alertable by operator decision 2026-07-26 on quality
        # grounds: provider status is computed from POST-summarizer text, so a
        # permanently dead summarizer otherwise degrades every briefing while
        # AskNews keeps reporting status="ok".
        self.summarizer_failure_count: int = 0
        # Genuine gap-fill-v2 CRASHES (not idle "driver found nothing" runs, not
        # deadline hits). Mirrors provider_failure_count: surfaced to the forecaster as
        # _gap_fill_v2_error_count and folded into alertable_count so a dead v2
        # feature reddens CI. A dead-on-arrival bug (the fastapi eager-import
        # defect) bumps this on EVERY question -> CI reddens immediately; a
        # one-off transient provider 500 bumps it once -> an accepted rare false
        # alarm (investigating that beats silently missing a dead feature). See
        # run_research for the three mutually-exclusive bump points.
        self.gap_fill_v2_error_count: int = 0

    @property
    def prediction_market_degraded_count(self) -> int:
        """Per-run Kalshi /series fetch failures, read from the prediction-market
        module counter and folded into the forecaster's alertable_count.

        The prediction-market provider soft-fails internally (a dead Kalshi
        series path still returns fuzzy-over-events matches), so this sub-path
        failure never raises and never bumps provider_failure_count. Reading the
        module counter here is the only way it reddens CI (the 2026-07-25 hole where
        research_provider_failures=0 while the path was dead)."""
        from metaculus_bot.research.prediction_market import (
            kalshi_series_fetch_failures,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
        )

        return kalshi_series_fetch_failures()

    @property
    def prediction_market_source_loss_count(self) -> int:
        """Per-run count of LOST prediction-market sources, read from the module
        counter and folded into alertable_count.

        A "source" is anything the snapshot depends on: one per venue whose
        query/prefetch fan-out lost a sub-fetch, one per whole-provider failure, and
        one when the keyword extractor produces nothing (which silences all four
        venues without any venue failing). That last cause is why this counts
        sources rather than venues — a dead extractor loses every venue's data
        without any venue going down. The distinguishing detail is durable
        per-source in ``MarketSnapshot.sources`` (``keywords:error(no_queries)`` vs
        ``polymarket:error(...)``), which rides the published comment and the
        schema-v2 research archive; this scalar deliberately stays one number.

        Operator decision 2026-07-25: alert on ANY source loss, not only a total
        blackout. The provider soft-fails every venue internally, so without this the
        forecasters silently run on zero market data while CI stays green."""
        from metaculus_bot.research.prediction_market import (
            prediction_market_source_losses,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
        )

        return prediction_market_source_losses()

    def reset_run_degradation_counters(self) -> None:
        """Zero per-run degradation counters at run start (called by
        forecast_questions alongside reset_pchip_stats). The prediction-market
        series and source-loss counters are module globals — resetting them here keeps
        them clean per-run metrics instead of leaking across runs/tests that share a
        process. The orchestrator's own instance counters are fresh per bot, so they
        need no reset here."""
        from metaculus_bot.research.prediction_market import (  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
            reset_series_degradation_counter,
            reset_source_loss_counter,
        )

        reset_series_degradation_counter()
        reset_source_loss_counter()

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

            research, provider_results, asknews_raw = await self._run_providers_parallel(question, providers)

            # Gap-fill v1 and v2 both consume the pre-gap-fill bundle and run
            # CONCURRENTLY in one gather (plan doc §2: research-phase wall-clock
            # is max(v1, v2), not the sum — v2's 540s deadline fits inside v1's
            # worst-case envelope only under this parallelism). Consequence: the
            # v2 driver's brief sees the bundle WITHOUT v1's addendum. v2's
            # section appends after v1's.
            gap_fill_v1_active = (
                env_flag_enabled(GAP_FILL_ENABLED_ENV) and len(research.strip()) >= GAP_FILL_MIN_RESEARCH_CHARS
            )
            gap_fill_v2_active = env_flag_enabled(GAP_FILL_V2_ENABLED_ENV)
            gap_fill_v2_payload: dict | None = None

            if gap_fill_v1_active or gap_fill_v2_active:
                # v1 and v2 import + run inside their own guards so each pass
                # degrades independently: a v2 code defect (import error in the
                # agentic package, unhandled raise) must never zero v1's
                # addendum in prod, and vice versa. The single gather keeps the
                # research-phase wall-clock at max(v1, v2), not the sum.
                def _capture_gap_fill_v2(payload: dict) -> None:
                    nonlocal gap_fill_v2_payload
                    gap_fill_v2_payload = payload

                async def _run_v1() -> str:
                    if not gap_fill_v1_active:
                        return ""
                    try:
                        from metaculus_bot.research.targeted import (
                            run_gap_fill_pass,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
                        )

                        return await run_gap_fill_pass(question, research, is_benchmarking=self._is_benchmarking)
                    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except — gap-fill is optional; a failure (import error, unhandled raise) must never kill the forecast
                        logger.exception("Gap-fill v1 stage failed; proceeding without it")
                        return ""

                def _count_gap_fill_v2_error(_exc: BaseException) -> None:
                    # Bumped ONLY on a genuine v2 crash, never on an idle
                    # "found nothing" run or a deadline hit. Three
                    # mutually-exclusive crash paths, one bump each (no
                    # double-count): (a) the loop-internal soft-fail — detected
                    # post-gather via the archive payload's telemetry["error"];
                    # (b) this seam's construction-error soft-fail — via
                    # run_gap_fill_v2's on_error callback; (c) the import/escape
                    # error in _run_v2's except — counted directly there. (a) and
                    # (b) are exclusive because (b)'s error means the loop never
                    # ran (no payload), and (c) is exclusive of both because the
                    # seam swallows all Exception, so nothing escapes it once
                    # construction succeeds.
                    self.gap_fill_v2_error_count += 1

                async def _run_v2() -> str:
                    if not gap_fill_v2_active:
                        return ""
                    try:
                        from metaculus_bot.research.agentic_gap_fill import (
                            run_gap_fill_v2,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
                        )

                        return await run_gap_fill_v2(
                            question,
                            research,
                            is_benchmarking=self._is_benchmarking,
                            archive_sink=_capture_gap_fill_v2,
                            on_error=_count_gap_fill_v2_error,
                        )
                    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except — gap-fill is optional; a failure (import error, unhandled raise) must never kill the forecast
                        logger.exception("Gap-fill v2 stage failed; proceeding without it")
                        # Path (c): an import failure (or any escape past the
                        # seam's own soft-fail) is a crash — count it directly
                        # here since no payload/on_error fires on this path.
                        self.gap_fill_v2_error_count += 1
                        return ""

                addendum, v2_findings = await asyncio.gather(_run_v1(), _run_v2())
                # Path (a): the loop ran but hit its catch-all soft-fail. The
                # loop swallows the crash and returns findings normally, so the
                # only crash signal is the stamped telemetry["error"] on the
                # archive payload. Checked here (not in _run_v2) so it can't
                # double-count with the on_error/except paths above — those
                # produce no payload with a non-None telemetry error.
                if gap_fill_v2_payload is not None:
                    v2_telemetry = gap_fill_v2_payload.get("telemetry")
                    if isinstance(v2_telemetry, dict) and v2_telemetry.get("error") is not None:
                        self.gap_fill_v2_error_count += 1
                if addendum:
                    research = f"{research}\n\n---\n\n## Targeted Gap-Fill (second pass)\n\n{addendum}"
                if v2_findings:
                    # v2_findings carries its own "## Agentic Research Findings"
                    # header (render_findings) — distinct from v1's section.
                    research = f"{research}\n\n---\n\n{v2_findings}"

            gap_fill_used = "## Targeted Gap-Fill (second pass)" in research

            # Diagnostics seam: the block is deliberately NOT appended to the
            # returned research — forecasters (and the gap-fill v2 driver brief)
            # consume that text verbatim and must never see it. It still reaches
            # its three destinations: (a) the INFO log line just below, (b) the
            # research archive via the sink's provider_diagnostics_block kwarg,
            # and (c) the published comment — stashed per-qid here and popped by
            # TemplateForecaster.pop_provider_diagnostics at comment-build time.
            diagnostics_block = format_provider_diagnostics_block(provider_results)
            qid = getattr(question, "id_of_question", None)
            if diagnostics_block:
                logger.info(f"Provider diagnostics for URL {question.page_url}:\n{diagnostics_block}")
                if qid is not None:
                    self._comment_diagnostics[qid] = diagnostics_block

            self._store_research_cache(cache_key, research)
            logger.info(f"Found Research for URL {question.page_url}:\n{research}")

            if self._research_sink is not None:
                if qid is not None:
                    try:
                        # provider_results is the authoritative per-provider outcome;
                        # providers_used is kept only for legacy archive readers.
                        self._research_sink(
                            qid=qid,
                            post_id=getattr(question, "id_of_post", None),
                            page_url=question.page_url,
                            question_text=question.question_text,
                            research_text=research,
                            providers_used=provider_names,
                            gap_fill_used=gap_fill_used,
                            provider_results=[asdict(r) for r in provider_results],
                            providers_attempted=provider_names,
                            providers_succeeded=[r.name for r in provider_results if r.status in SUCCEEDED_STATUSES],
                            gap_fill_v2=gap_fill_v2_payload,
                            provider_diagnostics_block=diagnostics_block,
                            asknews_raw=asknews_raw,
                        )
                    except (
                        Exception
                    ):  # HARNESS-SCAN-EXEMPT-broad-except — archive write is best-effort; never blocks the forecast
                        logger.exception("Research sink failed for qid=%d; continuing", qid)

            return research

    def pop_provider_diagnostics(self, qid: int | None) -> str:
        """Return-and-clear the comment-bound provider-diagnostics block for a question.

        The other half of the diagnostics seam in ``run_research``: the block is
        withheld from the forecaster-facing research text, and the forecaster pops
        it here when assembling ``research_report`` (the published comment).
        Popping keeps the stash from growing across a batch. Returns ``""`` when
        no diagnostics were recorded for the qid.
        """
        if qid is None:
            return ""
        return self._comment_diagnostics.pop(qid, "")

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
            logger.warning(
                "AskNews summarization failed (%s); using raw articles under a degradation banner",
                type(exc).__name__,
            )
            return self._degraded_to_raw_articles(question, research, type(exc).__name__)
        if not summary.strip():
            logger.warning("AskNews summarization returned blank output; using raw articles under a banner")
            return self._degraded_to_raw_articles(question, research, "blank_output")
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
            # Each rung gets the vendor its env var pays for. Binding the bare
            # ``_call_perplexity`` here would hand priority 3 the method's
            # OpenRouter-first default — deliberate on the AskNews-fallback path
            # (_attempt_research_fallback prefers the cheap route), wrong here,
            # where it collapses the ladder's two Perplexity rungs into one and
            # passes api_key=None whenever only PERPLEXITY_API_KEY is set.
            perplexity_callback=self._call_perplexity_direct,
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

            providers.append((financial_data_provider(is_benchmarking=self._is_benchmarking), "financial_data"))

        if env_flag_enabled(TS_ANCHOR_ENABLED_ENV):
            from metaculus_bot.research.timeseries_anchor import (
                timeseries_anchor_provider,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
            )

            providers.append((timeseries_anchor_provider(is_benchmarking=self._is_benchmarking), "timeseries_anchor"))

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
    ) -> tuple[str, list[ProviderResult], str]:
        from metaculus_bot.research.providers import (
            is_asknews_subscription_error,  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
        )

        # Raw pre-summarization AskNews article text, captured for the research
        # archive (2026-07-18 audit hygiene: the archive otherwise stores only the
        # post-summarization briefing, so FETCH-vs-SUMMARIZE attribution and
        # summarizer replays required fresh paid pulls). Empty when AskNews didn't
        # run, errored, or fell back to already-prose providers.
        asknews_raw_holder: dict[str, str] = {}

        async def _run_one(provider: ResearchCallable, name: str) -> tuple[str, ProviderResult]:
            started = time.monotonic()
            # A multi-source provider records its per-source outcome into the
            # (qid, provider) registry during the call; drain it here so partial
            # upstream loss (e.g. Kalshi dropped over the size cap) rides into
            # ProviderResult.details instead of vanishing behind a healthy `ok`.
            qid = getattr(question, "id_of_question", None)
            try:
                fallback_provider: str | None = None
                if name == "asknews" and self._allow_research_fallback:
                    raw, fallback_provider = await self._fetch_research_with_fallback(question, provider, name)
                else:
                    raw = await provider(question)
                used_fallback = fallback_provider is not None
                # AskNews returns raw article markdown (no LLM prose); summarize it
                # into an analyst briefing. Every other provider already emits
                # LLM-written prose (native search, Gemini, Perplexity, Exa) or
                # deterministic tables (financial, prediction markets), so they
                # pass through raw — no lossy second-pass summarization. When
                # AskNews fails and we fall back to Perplexity/Exa, that fallback
                # is already prose, so skip summarization too.
                if name == "asknews" and not used_fallback:
                    if raw and raw.strip():
                        asknews_raw_holder["text"] = raw
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
                    details=pop_provider_detail(qid, name),
                    fallback_provider=fallback_provider,
                )
                return (raw, result)
            except Exception as e:  # HARNESS-SCAN-EXEMPT-broad-except — converted to a ProviderResult(status=errored/inactive); one provider failing never kills the research phase
                # Drain-and-discard any partial detail the provider recorded before
                # raising: an errored result carries the error, not source detail,
                # and a stale entry must not leak into a later same-key call.
                pop_provider_detail(qid, name)
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
                    self.provider_failure_count += 1
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
                # Label the section with whoever actually produced the text. On a
                # fallback that is NOT provider_result.name (which keeps the primary's
                # identity for the diagnostics line) — rendering Perplexity prose under
                # "## News Articles (AskNews)" mislabelled the source in the published
                # comment and in the archive.
                header = self._provider_header(provider_result.fallback_provider or provider_result.name)
                combined_parts.append(f"{header}\n{_demote_inner_headings(raw)}")

        combined = "\n\n---\n\n".join(combined_parts) if combined_parts else ""
        return combined, provider_results, asknews_raw_holder.get("text", "")

    @staticmethod
    def _provider_header(name: str) -> str:
        headers = {
            "asknews": "## News Articles (AskNews)",
            "native_search": "## Web Research (Native Search)",
            "gemini_search": "## Web Research (Google Search via Gemini)",
            "financial_data": "## Financial & Economic Data",
            "timeseries_anchor": TS_ANCHOR_SECTION_HEADER,
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
    ) -> tuple[str, str | None]:
        """Return ``(research_text, fallback_provider_name)``.

        ``fallback_provider_name`` is None on the normal path and otherwise names the
        vendor that actually answered ("openrouter" / "perplexity" / "exa"). The caller
        uses it for two things: to skip AskNews summarization on already-prose output, and
        to label the research section with the source that produced it.

        A vendor swap on the PRIMARY provider is real degradation, not a success: a
        different index, a different recency profile, and — because the fallback is already
        prose — no AskNews summarizer pass, so the briefing loses the per-article relevance
        gate, [PRE-WINDOW] labeling, and recency reordering that the 2026-07-18 audit made
        load-bearing. It used to bump no counter and record no detail, so it read as
        healthy. We record a per-source loss token here so the diagnostics line and the
        schema-v2 archive carry it. Deliberately NOT a new alertable counter: folding one
        into alertable_count changes what CI treats as red, which is the operator's call.
        """
        try:
            return (await provider(question), None)
        except Exception as exc:
            if self._allow_research_fallback and provider_name == "asknews":
                logger.warning(f"Primary research provider '{provider_name}' failed with {type(exc).__name__}: {exc}")
                fallback_text, fallback_name = await self._attempt_research_fallback(question.question_text)
                if fallback_text is not None:
                    record_provider_detail(
                        getattr(question, "id_of_question", None),
                        provider_name,
                        {
                            "sources": {
                                provider_name: f"error({type(exc).__name__})",
                                "fallback": f"ok({fallback_name})",
                            }
                        },
                    )
                    return (fallback_text, fallback_name)
            raise

    async def _attempt_research_fallback(self, question_text: str) -> tuple[str | None, str | None]:
        """Return ``(research_text, provider_name)``, or ``(None, None)`` if no rung answered.

        The provider name rides back so the caller can label the section with the vendor
        that actually answered; rendering it as AskNews mislabeled the source in the
        published comment and in the archive.
        """
        # Ordering intentionally differs from the primary selector
        # (choose_provider_with_name: AskNews -> Exa -> Perplexity -> OpenRouter).
        # This fallback only fires when AskNews (always the primary in prod) has
        # already failed, so AskNews is excluded. Among the remaining options we
        # prefer the Perplexity-via-OpenRouter route first (cheap, prose-returning,
        # routed through the donated-key wrapper), then direct Perplexity, then
        # Exa last (SmartSearcher spins up its own multi-search/LLM loop, the most
        # expensive path). The primary selector orders by index quality, not cost,
        # which is why the two lists diverge by design.
        #
        # The names returned here are the same keys ``_provider_header`` maps, so the
        # section header follows the vendor automatically.
        try:
            if os.getenv(OPENROUTER_API_KEY_ENV):
                logger.info("Falling back to openrouter/perplexity for research")
                return (await self._call_perplexity(question_text, use_open_router=True), "openrouter")
            if os.getenv(PERPLEXITY_API_KEY_ENV):
                logger.info("Falling back to Perplexity for research")
                return (await self._call_perplexity(question_text, use_open_router=False), "perplexity")
            if os.getenv(EXA_API_KEY_ENV):
                logger.info("Falling back to Exa search for research")
                return (await self._call_exa_smart_searcher(question_text), "exa")
        except Exception as fallback_exc:  # HARNESS-SCAN-EXEMPT-broad-except — best-effort fallback; logs and returns None so the primary error propagates
            logger.warning(f"Fallback research provider also failed: {type(fallback_exc).__name__}: {fallback_exc}")
        return (None, None)

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
            # temperature=None defers to provider defaults; redundant on ft 0.2.92
            # (GeneralLlm ctor default is already None). No top_p.
            temperature=None,
            api_key=get_openrouter_api_key(model_name) if model_name.startswith("openrouter/") else None,
            # allowed_tries=1 hands the retry budget to the gated wrapper below; left
            # unpinned it inherited forecasting-tools' default of 2 with an un-gated
            # random.uniform(5, 10) tenacity sleep.
            allowed_tries=1,
        )
        return await invoke_with_transient_retry(
            lambda: model.invoke(prompt),
            wall_timeout=PERPLEXITY_WALL_TIMEOUT,
            label="perplexity_research",
        )

    async def _call_perplexity_openrouter(self, question: MetaculusQuestion) -> str:
        return await self._call_perplexity(question, use_open_router=True)

    async def _call_perplexity_direct(self, question: MetaculusQuestion) -> str:
        return await self._call_perplexity(question, use_open_router=False)

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
