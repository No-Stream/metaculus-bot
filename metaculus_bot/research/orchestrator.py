"""Research orchestration extracted from TemplateForecaster.

Encapsulates provider selection, the primary provider's fallback ladder, caching, the
diagnostics seam, and the per-question time budget. The TemplateForecaster delegates
run_research to an instance of this class.

The stages that are self-contained live in sibling modules and are mixed back into
``ResearchOrchestrator``, so every call form and patch surface is unchanged:

* ``provider_fanout`` — running the selected providers under the research-phase deadline.
* ``section_format`` — provider headers and heading levels in the assembled bundle.
* ``asknews_summarization`` — the AskNews-only summarizer pass (it's the one provider
  that returns raw article text rather than LLM-written prose; all others pass through).
* ``gap_fill_stages`` — the two optional gap-fill passes and their budget accounting.
* ``degradation_views`` — read-only views onto the research side's module counters.
"""

import asyncio
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict

from forecasting_tools import GeneralLlm, SmartSearcher, clean_indents
from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.api_key_utils import get_openrouter_api_key
from metaculus_bot.constants import (
    DEFAULT_MAX_CONCURRENT_RESEARCH,
    EXA_API_KEY_ENV,
    FINANCIAL_DATA_ENABLED_ENV,
    GEMINI_SEARCH_ENABLED_ENV,
    GEMINI_SEARCH_MODEL_ENV,
    NATIVE_SEARCH_ENABLED_ENV,
    NATIVE_SEARCH_MODEL_ENV,
    OPENROUTER_API_KEY_ENV,
    PERPLEXITY_API_KEY_ENV,
    PERPLEXITY_RESEARCH_MODEL,
    PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER,
    PERPLEXITY_WALL_TIMEOUT,
    PREDICTION_MARKETS_ENABLED_ENV,
    RESOLUTION_SOURCE_ENABLED_ENV,
    TS_ANCHOR_ENABLED_ENV,
    env_flag_enabled,
)
from metaculus_bot.fallback_openrouter import _record_deprecation_if_matched
from metaculus_bot.llm_retry import invoke_with_transient_retry
from metaculus_bot.research.asknews_summarization import AskNewsSummarization
from metaculus_bot.research.degradation_views import ResearchDegradationViews
from metaculus_bot.research.gap_fill_stages import GapFillStages
from metaculus_bot.research.provider_diagnostics import (
    SUCCEEDED_STATUSES,
    ProviderResult,
    format_provider_diagnostics_block,
    pop_provider_detail,
    record_provider_detail,
)
from metaculus_bot.research.provider_fanout import ProviderFanout, _empty_provider
from metaculus_bot.research.providers import (
    ResearchCallable,
    choose_provider_with_name,
    is_asknews_subscription_error,
    native_search_provider,
)
from metaculus_bot.research.section_format import ResearchSectionFormatting, _demote_inner_headings
from metaculus_bot.time_budget import QuestionTimeBudget

# ``_demote_inner_headings`` moved to section_format but is still imported from this
# module path by callers outside the package; the re-export keeps that working (and
# keeps the auto-formatter from stripping an otherwise-unused import).
__all__ = ["ResearchOrchestrator", "_demote_inner_headings"]

_PROVIDER_ERROR_MESSAGE_MAX_CHARS = 300

logger = logging.getLogger(__name__)


class ResearchOrchestrator(
    ProviderFanout,
    ResearchSectionFormatting,
    AskNewsSummarization,
    GapFillStages,
    ResearchDegradationViews,
):
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
        # Per-QUESTION count of research thinned by the time budget OFF the fast
        # path: a provider cancelled at the research-phase deadline, or gap-fill
        # cut/skipped for budget, on a question whose window was wide enough that
        # fast_path never fired. The fast path has its own alertable counter
        # (_time_budget_fast_path_count, forecaster-side); without this one the
        # band just above the threshold — where the research window can still sit
        # under research's configured worst case — degraded silently and the
        # end-of-run census read all-clear. Deduplicated per question (the seen
        # set), so a question losing a provider AND both gap-fill passes counts
        # once, and fast-path questions are excluded so nothing double-charges.
        self.research_budget_cut_count: int = 0
        self._research_budget_cut_seen: set[object] = set()

    def _record_research_budget_cut(self, question: MetaculusQuestion, *, fast_path: bool) -> None:
        """Count one question's budget-driven research degradation, once, off the fast path."""
        if fast_path:
            return
        key: object = getattr(question, "id_of_question", None) or id(question)
        if key in self._research_budget_cut_seen:
            return
        self._research_budget_cut_seen.add(key)
        self.research_budget_cut_count += 1

    async def run_research(self, question: MetaculusQuestion, time_budget: QuestionTimeBudget | None = None) -> str:
        """Build one question's research bundle, inside its time budget if it has one.

        ``time_budget`` is None for every caller that isn't the per-question
        pipeline (tests, the research-only tooling), and then the phase runs
        unbounded exactly as it did before. When present it does two things: on a
        thin window (``fast_path``) the OPTIONAL stages are not run at all, and in
        every case the phase is bounded by its share of the remaining budget so it
        cannot spend the time the forecast fan-out and the prediction POST need.
        """
        cache_key, cached = self._lookup_research_cache(question)
        if cached is not None:
            logger.info(f"Using cached research for question {cache_key}")
            return cached

        async with self._concurrency_limiter:
            cache_key, cached = self._lookup_research_cache(question)
            if cached is not None:
                logger.info(f"Using cached research for question {cache_key} (double-check)")
                return cached

            fast_path = time_budget is not None and time_budget.fast_path
            providers = self._select_research_providers(fast_path=fast_path)
            provider_names = [name for _, name in providers]
            logger.info(f"Using research providers: {provider_names}")

            research, provider_results, asknews_raw = await self._run_providers_parallel(
                question, providers, time_budget=time_budget
            )
            if any(pr.status == "deadline" for pr in provider_results):
                # A provider cancelled at the research-phase deadline is budget-driven
                # degradation; off the fast path nothing else counts it (see
                # _record_research_budget_cut).
                self._record_research_budget_cut(question, fast_path=fast_path)

            research, gap_fill_v2_payload = await self._run_gap_fill_passes(
                question, research, fast_path=fast_path, time_budget=time_budget
            )

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

            if self._research_sink is not None and qid is not None:
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

    def _select_research_providers(self, fast_path: bool = False) -> list[tuple[ResearchCallable, str]]:
        """Assemble the enabled providers for one question.

        ``fast_path`` is the time-budget thin-window mode: drop the two SLOW search
        providers (native_search, gemini_search) and keep everything else. The
        optional providers all run CONCURRENTLY with the primary, whose own worst
        case (AskNews 300 s + summarizer 300 s, sequential inside one provider) is
        the phase's longest configured pole — so dropping the cheap hard-capped
        providers (resolution_source 45 s, prediction_market 150 s, ts_anchor 20 s,
        financial classifier 30 s) cannot shorten the phase and only discards the
        resolution ground truth. What the fast path CAN shed is the measured tail:
        native_search is the phase's slowest provider on 51.5% of questions and
        reached 292 s against the primary's 110 s measured worst case
        (scratch/residual_2026-08-24/time_budget_design.md). Anything still
        straggling past the research window is cancelled by
        ``_await_providers_within_deadline`` with its partial bundle kept.
        """
        providers: list[tuple[ResearchCallable, str]] = []

        primary, primary_name = self._select_research_provider()
        if primary_name != "none":
            providers.append((primary, primary_name))

        if not fast_path and env_flag_enabled(NATIVE_SEARCH_ENABLED_ENV):
            model = os.getenv(NATIVE_SEARCH_MODEL_ENV)
            providers.append(
                (
                    native_search_provider(model, is_benchmarking=self._is_benchmarking),
                    "native_search",
                )
            )

        if not fast_path and env_flag_enabled(GEMINI_SEARCH_ENABLED_ENV):
            from metaculus_bot.research.gemini_search import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # gated google-genai provider
                gemini_search_provider,
            )

            gemini_model = os.getenv(GEMINI_SEARCH_MODEL_ENV)
            providers.append(
                (
                    gemini_search_provider(gemini_model, is_benchmarking=self._is_benchmarking),
                    "gemini_search",
                )
            )

        if env_flag_enabled(FINANCIAL_DATA_ENABLED_ENV):
            from metaculus_bot.research.financial_data import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # gated pandas/yfinance/fredapi provider
                financial_data_provider,
            )

            providers.append((financial_data_provider(is_benchmarking=self._is_benchmarking), "financial_data"))

        if env_flag_enabled(TS_ANCHOR_ENABLED_ENV):
            from metaculus_bot.research.timeseries_anchor import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # gated numpy/pandas provider
                timeseries_anchor_provider,
            )

            providers.append((timeseries_anchor_provider(is_benchmarking=self._is_benchmarking), "timeseries_anchor"))

        if env_flag_enabled(PREDICTION_MARKETS_ENABLED_ENV):
            from metaculus_bot.research.prediction_market import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # gated rapidfuzz/aiohttp provider
                prediction_market_provider,
            )

            providers.append((prediction_market_provider(is_benchmarking=self._is_benchmarking), "prediction_market"))

        if env_flag_enabled(RESOLUTION_SOURCE_ENABLED_ENV):
            from metaculus_bot.research.resolution_source import (  # noqa: PLC0415  # HARNESS-SCAN-EXEMPT-function-level-import  # gated aiohttp/trafilatura provider
                resolution_source_provider,
            )

            providers.append((resolution_source_provider(is_benchmarking=self._is_benchmarking), "resolution_source"))

        if not providers:
            providers.append((_empty_provider, "none"))

        return providers

    def _failed_provider_result(self, name: str, exc: Exception, latency_ms: int) -> ProviderResult:
        """Classify a provider that raised: ``inactive`` for expected off-season AskNews, else ``errored``.

        Only ``errored`` bumps ``provider_failure_count`` (which reddens CI) and
        feeds the deprecation matcher; an inactive subscription is a known
        off-season state, not degradation.
        """
        if name == "asknews" and is_asknews_subscription_error(exc):
            status = "inactive"
            logger.info(
                "Research provider %s inactive (expected off-season): %s: %s",
                name,
                type(exc).__name__,
                exc,
            )
        else:
            status = "errored"
            self.provider_failure_count += 1
            logger.warning(f"Research provider {name} failed ({type(exc).__name__}): {exc}")
            _record_deprecation_if_matched(f"<provider:{name}>", str(exc))
        return ProviderResult(
            name=name,
            status=status,
            chars=0,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
            error_message=str(exc)[:_PROVIDER_ERROR_MESSAGE_MAX_CHARS],
        )

    async def _run_providers_parallel(
        self,
        question: MetaculusQuestion,
        providers: list[tuple[ResearchCallable, str]],
        time_budget: QuestionTimeBudget | None = None,
    ) -> tuple[str, list[ProviderResult], str]:
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
                if name == "asknews" and not used_fallback and raw and raw.strip():
                    # Empty raw skips the summarizer entirely: there is nothing to brief
                    # from, and asking anyway spends a call to get either a refusal or an
                    # invented briefing (the summarizer prompt has no no-data escape).
                    # AskNews already recorded an `articles: empty(no_articles)` loss token,
                    # so the `empty` status below stays distinguishable from a skipped run.
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
            except asyncio.CancelledError:
                # A deadline-cancelled provider must drain its registry entry too:
                # CancelledError is a BaseException and would otherwise skip both
                # drain paths, leaving exactly the stale same-key entry the except
                # below exists to prevent. Re-raised so the caller still records
                # the cancellation as status="deadline".
                pop_provider_detail(qid, name)
                raise
            except Exception as e:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except — converted to a ProviderResult(status=errored/inactive); one provider failing never kills the research phase
                # Drain-and-discard any partial detail the provider recorded before
                # raising: an errored result carries the error, not source detail,
                # and a stale entry must not leak into a later same-key call.
                pop_provider_detail(qid, name)
                latency_ms = int((time.monotonic() - started) * 1000)
                return ("", self._failed_provider_result(name, e, latency_ms))

        results = await self._await_providers_within_deadline(providers, _run_one, time_budget)
        combined, provider_results = self._assemble_provider_sections(results)
        return combined, provider_results, asknews_raw_holder.get("text", "")

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
        except Exception as fallback_exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except — best-effort fallback; logs and returns None so the primary error propagates
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
        model_name = PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER if use_open_router else PERPLEXITY_RESEARCH_MODEL
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
