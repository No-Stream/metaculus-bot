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
from collections.abc import Callable, Coroutine
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
    PERPLEXITY_RESEARCH_MODEL,
    PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER,
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
    is_asknews_subscription_error,
    native_search_provider,
)
from metaculus_bot.time_budget import QuestionTimeBudget

_PROVIDER_ERROR_MESSAGE_MAX_CHARS = 300

logger = logging.getLogger(__name__)


async def _empty_provider(_: MetaculusQuestion) -> str:
    """Stand-in provider for a selection that produced nothing to run."""
    return ""


def _remaining_research_phase_s(time_budget: QuestionTimeBudget | None) -> float | None:
    """Seconds the research phase may still spend, or None for unbounded.

    ``None`` is both "this caller has no budget" and the value ``asyncio.wait`` /
    ``asyncio.wait_for`` already take to mean "no timeout", so it passes straight
    through to them without a sentinel translation.
    """
    if time_budget is None:
        return None
    return time_budget.research_phase_deadline_s()


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

    @property
    def prediction_market_degraded_count(self) -> int:
        """Per-run Kalshi CATALOGUE fetch failures, read from the prediction-market
        module counter and folded into the forecaster's alertable_count.

        The prediction-market provider soft-fails internally (a lost catalogue pull still
        returns whatever the venue-search channel found), so this sub-path failure never
        raises and never bumps provider_failure_count. Reading the module counter here is
        the only way it reddens CI (the 2026-07-25 hole where
        research_provider_failures=0 while the path was dead). The property and marker
        names predate the ranked pipeline, where the counter moved from the retired /series
        index to the events catalogue — a strictly more load-bearing thing, since the
        catalogue feeds both the settlement-source join and the fuzzy channel."""
        from metaculus_bot.research.prediction_market import (  # noqa: PLC0415  # optional market deps stay off cold path
            kalshi_catalogue_fetch_failures,
        )

        return kalshi_catalogue_fetch_failures()

    @property
    def prediction_market_source_loss_count(self) -> int:
        """Per-run count of LOST prediction-market sources, read from the module
        counter and folded into alertable_count.

        A "source" is anything the snapshot depends on: one per venue whose
        search/prefetch fan-out lost a sub-fetch, one per whole-provider failure, and
        one each when the query author or the RANKING call comes back unusable. Those
        last two are why this counts sources rather than venues — a dead ranker
        degrades every venue's contribution without any venue going down. The
        distinguishing detail is durable per-source in ``MarketSnapshot.sources``
        (``ranking:error(...)`` vs ``polymarket:error(...)``), which rides the
        published comment and the schema-v2 research archive; this scalar
        deliberately stays one number.

        Operator decision 2026-07-25: alert on ANY source loss, not only a total
        blackout. The provider soft-fails every venue internally, so without this the
        forecasters silently run on zero market data while CI stays green."""
        from metaculus_bot.research.prediction_market import (  # noqa: PLC0415  # optional market deps stay off cold path
            prediction_market_source_losses,
        )

        return prediction_market_source_losses()

    @property
    def provider_degradation_count(self) -> int:
        """Per-run count of ALERTABLE provider-degradation findings, folded into
        alertable_count.

        One finding per (signal, venue), over the two signals provider_health
        defines: a declared liquidity field dead across 100% of the pool rows a
        venue produced (``market_field_contract``), or a prefetch reporting success
        while returning an empty catalogue (``catalogue_empty``). Each is a
        100%-of-denominator conjunction over the whole run, so a single question
        with no matching market stays silent — the denominators are a venue's own
        pool rows and a catalogue's own size, never questions-in-a-run (prod runs
        carry 1-2 questions, so a rate over those IS a per-question flag).

        The first is the signal that would have caught Kalshi's liquidity labels
        blank on 100% of rows for weeks in prod while every counter read zero; the
        second closes its blind spot, since a catalogue that silently empties out
        looks to it like a venue with nothing to say. A third rule (Signal B, a
        venue contributing nothing while >=2 siblings answered) was deleted
        2026-08-04 as unsound under ranked retrieval — see provider_health's module
        docstring and FUTURE.md; the surviving cross-run intent is unjudgeable
        inside one question.

        Suppressed findings are excluded here but still logged in full and still
        ride the PROVIDER_DEGRADATION marker (see
        constants.provider_degradation_alerts_active)."""
        from metaculus_bot.research.provider_health import (  # noqa: PLC0415  # lazy counter import
            provider_degradation_count,
        )

        return provider_degradation_count()

    def log_provider_degradation_summary(self) -> None:
        """Emit the per-run PROVIDER_DEGRADATION marker + one WARN per finding.

        Called from forecast_questions after publishing completes, alongside the
        other end-of-run summaries. Fires even at zero findings — a measured zero is
        a positive statement of provider health, the same reasoning behind
        FORECASTERS_SURVIVED existing next to FORECASTER_DROPS."""
        from metaculus_bot.research.provider_health import (  # noqa: PLC0415  # lazy counter import
            log_provider_degradation_summary,
        )

        log_provider_degradation_summary()

    def reset_run_degradation_counters(self) -> None:
        """Zero per-run degradation counters at run start (called by
        forecast_questions alongside reset_pchip_stats). The prediction-market
        series and source-loss counters, and the provider-health observation store,
        are module globals — resetting them here keeps them clean per-run metrics
        instead of leaking across runs/tests that share a process. The
        orchestrator's own instance counters are fresh per bot, so they need no
        reset here."""
        from metaculus_bot.research.prediction_market import (  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
            reset_series_degradation_counter,
            reset_source_loss_counter,
        )
        from metaculus_bot.research.provider_health import (  # noqa: PLC0415  # lazy counter import
            reset_provider_health,
        )

        reset_series_degradation_counter()
        reset_source_loss_counter()
        reset_provider_health()

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

            # Gap-fill v1 and v2 both consume the pre-gap-fill bundle and run
            # CONCURRENTLY in one gather (plan doc §2: research-phase wall-clock
            # is max(v1, v2), not the sum — v2's GAP_FILL_V2_WALL_DEADLINE fits
            # inside v1's worst-case envelope only under this parallelism).
            # Consequence: the v2 driver's brief sees the bundle WITHOUT v1's
            # addendum. v2's section appends after v1's.
            #
            # Both are OPTIONAL, and they are the research phase's largest optional
            # cost: v1's configured worst case is 555s (analyzer 135 + resolver wave
            # 420) and v2 measures 84s at p50 / 293s at its observed max. So the
            # fast path drops both — that is where the time for a thin window comes
            # from, far more than provider selection.
            gap_fill_budget_s = _remaining_research_phase_s(time_budget)
            skip_optional_gap_fill = fast_path or (gap_fill_budget_s is not None and gap_fill_budget_s <= 0.0)
            gap_fill_v1_active = (
                env_flag_enabled(GAP_FILL_ENABLED_ENV)
                and not skip_optional_gap_fill
                and len(research.strip()) >= GAP_FILL_MIN_RESEARCH_CHARS
            )
            gap_fill_v2_active = env_flag_enabled(GAP_FILL_V2_ENABLED_ENV) and not skip_optional_gap_fill
            if skip_optional_gap_fill and (
                env_flag_enabled(GAP_FILL_ENABLED_ENV) or env_flag_enabled(GAP_FILL_V2_ENABLED_ENV)
            ):
                logger.warning(
                    "GAP_FILL_SKIPPED_FOR_BUDGET: question=%s fast_path=%s research_phase_remaining=%s",
                    getattr(question, "id_of_question", None),
                    str(fast_path).lower(),
                    "n/a" if gap_fill_budget_s is None else f"{gap_fill_budget_s:.0f}s",
                )
                self._record_research_budget_cut(question, fast_path=fast_path)
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
                        from metaculus_bot.research.targeted import (  # noqa: PLC0415  # import stays inside failure guard
                            run_gap_fill_pass,
                        )

                        # Bounded by whatever the research phase has left, so a pass
                        # that overruns its own internal deadlines still cannot spend
                        # the forecast's time.
                        return await asyncio.wait_for(
                            run_gap_fill_pass(question, research, is_benchmarking=self._is_benchmarking),
                            timeout=_remaining_research_phase_s(time_budget),
                        )
                    except TimeoutError:
                        # Its own branch (like v2's below) because it is not a failure:
                        # falling into the generic except would log a traceback under
                        # "stage failed" for a deliberate budget cut.
                        logger.warning(
                            "GAP_FILL_V1_CUT_FOR_BUDGET: question=%s; research phase ran out of budget",
                            getattr(question, "id_of_question", None),
                        )
                        self._record_research_budget_cut(question, fast_path=fast_path)
                        return ""
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
                        from metaculus_bot.research.agentic_gap_fill import (  # noqa: PLC0415  # import stays inside failure guard
                            run_gap_fill_v2,
                        )

                        # Same research-phase bound as v1 above, on top of v2's own
                        # GAP_FILL_V2_WALL_DEADLINE (which measures as never binding:
                        # 0 of 103 triple-era records report deadline_hit).
                        return await asyncio.wait_for(
                            run_gap_fill_v2(
                                question,
                                research,
                                is_benchmarking=self._is_benchmarking,
                                archive_sink=_capture_gap_fill_v2,
                                on_error=_count_gap_fill_v2_error,
                            ),
                            timeout=_remaining_research_phase_s(time_budget),
                        )
                    except TimeoutError:
                        # NOT a v2 crash: we cut it to protect the prediction POST, so
                        # this must not bump gap_fill_v2_error_count (which exists to
                        # redden CI on a dead v2 feature) — the budget decision is
                        # alertable via research_budget_cut_count (fast-path questions
                        # never reach here; gap-fill is skipped upstream for them).
                        logger.warning(
                            "GAP_FILL_V2_CUT_FOR_BUDGET: question=%s; research phase ran out of budget",
                            getattr(question, "id_of_question", None),
                        )
                        self._record_research_budget_cut(question, fast_path=fast_path)
                        return ""
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
            from metaculus_bot.research.gemini_search import (  # noqa: PLC0415  # gated google-genai provider
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
            from metaculus_bot.research.financial_data import (  # noqa: PLC0415  # gated pandas/yfinance/fredapi provider
                financial_data_provider,
            )

            providers.append((financial_data_provider(is_benchmarking=self._is_benchmarking), "financial_data"))

        if env_flag_enabled(TS_ANCHOR_ENABLED_ENV):
            from metaculus_bot.research.timeseries_anchor import (  # noqa: PLC0415  # gated numpy/pandas provider
                timeseries_anchor_provider,
            )

            providers.append((timeseries_anchor_provider(is_benchmarking=self._is_benchmarking), "timeseries_anchor"))

        if env_flag_enabled(PREDICTION_MARKETS_ENABLED_ENV):
            from metaculus_bot.research.prediction_market import (  # noqa: PLC0415  # gated rapidfuzz/aiohttp provider
                prediction_market_provider,
            )

            providers.append((prediction_market_provider(is_benchmarking=self._is_benchmarking), "prediction_market"))

        if env_flag_enabled(RESOLUTION_SOURCE_ENABLED_ENV):
            from metaculus_bot.research.resolution_source import (  # noqa: PLC0415  # gated aiohttp/trafilatura provider
                resolution_source_provider,
            )

            providers.append((resolution_source_provider(is_benchmarking=self._is_benchmarking), "resolution_source"))

        if not providers:
            providers.append((_empty_provider, "none"))

        return providers

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

        results = await self._await_providers_within_deadline(providers, _run_one, time_budget)

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
    async def _await_providers_within_deadline(
        providers: list[tuple[ResearchCallable, str]],
        run_one: Callable[[ResearchCallable, str], Coroutine[object, object, tuple[str, ProviderResult]]],
        time_budget: QuestionTimeBudget | None,
    ) -> list[tuple[str, ProviderResult]]:
        """Run every provider concurrently, cancelling any still running at the deadline.

        Replaces a bare ``asyncio.gather``, which had no outer bound at all: each
        provider carries its own wall timeout, but the PHASE carried none, so one
        provider whose internal timeout failed to fire could hold the question past
        its close with nothing to stop it. Stragglers are cancelled and recorded as
        ``status="deadline"``, so the partial bundle is used rather than lost and the
        cut providers are named in the diagnostics block and the research archive.

        A cancelled provider does NOT bump ``provider_failure_count``: it did not
        fail, we stopped it. The budget decision is alertable instead through
        ``research_budget_cut_count`` (bumped by the caller when any result comes
        back ``status="deadline"`` off the fast path; fast-path questions are
        already counted by the forecaster's ``time_budget_fast_path`` counter).
        Keeping the two apart is what lets ``research_provider_failures`` keep
        meaning "a provider broke".

        With no budget (every caller outside the per-question pipeline) the wait is
        unbounded and behavior is identical to the old gather.
        """
        if not providers:
            # ``asyncio.gather()`` returned [] on an empty list; ``asyncio.wait(set())``
            # raises. Selection always yields at least the "none" stub, so this is the
            # direct-call path only.
            return []

        tasks = [asyncio.create_task(run_one(provider, name), name=f"research:{name}") for provider, name in providers]
        task_name = {task: name for task, (_, name) in zip(tasks, providers, strict=True)}

        deadline_s = time_budget.research_phase_deadline_s() if time_budget is not None else None
        _done, pending = await asyncio.wait(tasks, timeout=deadline_s, return_when=asyncio.ALL_COMPLETED)
        for task in pending:
            task.cancel()
        if pending:
            # Let the cancellations land so no "task was destroyed but it is pending"
            # warning escapes into the run log.
            await asyncio.wait(pending, timeout=2.0)
            logger.warning(
                "RESEARCH_PHASE_DEADLINE: cancelled %d/%d providers after %.0fs (%s)",
                len(pending),
                len(tasks),
                deadline_s or 0.0,
                ",".join(sorted(task_name.get(task, "unknown") for task in pending)),
            )

        # Provider order is the section order in the research bundle and the row order
        # in the diagnostics block, so rebuild it from `providers` rather than from
        # asyncio.wait's unordered sets.
        results: list[tuple[str, ProviderResult]] = []
        for task, (_, name) in zip(tasks, providers, strict=True):
            if task in pending:
                # Latency IS the deadline for a cancelled provider: every task starts
                # at phase start, so one still running when the deadline lands ran for
                # exactly that long.
                results.append(
                    (
                        "",
                        ProviderResult(
                            name=name,
                            status="deadline",
                            chars=0,
                            latency_ms=round((deadline_s or 0.0) * 1000),
                        ),
                    )
                )
                continue
            exc = task.exception()
            if exc is not None:
                # _run_one converts every provider exception into a ProviderResult, so
                # reaching here means the wrapper itself broke — a bug, not a provider
                # failure, and it must not be swallowed into a fake result.
                raise exc
            results.append(task.result())
        return results

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
