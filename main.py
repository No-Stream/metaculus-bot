import asyncio
import logging
import random
import time
from collections import defaultdict
from typing import Any, Coroutine, Sequence, cast

from exceptiongroup import ExceptionGroup
from forecasting_tools import (  # AskNewsSearcher,
    BinaryQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.forecast_report import ForecastReport, ResearchWithPredictions
from forecasting_tools.data_models.questions import DateQuestion

from metaculus_bot import stacking as stacking
from metaculus_bot.aggregation_strategies import (
    AggregationStrategy,
    combine_binary_predictions,
    combine_multiple_choice_predictions,
    combine_numeric_predictions,
)
from metaculus_bot.comment_trimming import trim_section
from metaculus_bot.config import load_environment
from metaculus_bot.constants import (
    BINARY_STACKING_ENABLED_ENV,
    CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD,
    CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD,
    CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD,
    CRUX_SOFT_DEADLINE,
    DEFAULT_MAX_CONCURRENT_RESEARCH,
    FORECASTER_SOFT_DEADLINE,
    MC_STACKING_ENABLED_ENV,
    MIN_FORECASTERS_TO_PUBLISH,
    NUMERIC_STACKING_ENABLED_ENV,
    PER_QUESTION_WALL_CLOCK_DEADLINE,
    STACKER_FALLBACK_SOFT_DEADLINE,
    STACKER_SOFT_DEADLINE,
    WALL_CLOCK_STACKING_MIN_BUDGET,
    env_flag_enabled,
)
from metaculus_bot.llm_setup import prepare_llm_config
from metaculus_bot.numeric_diagnostics import log_final_prediction
from metaculus_bot.numeric_pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric_utils import bound_messages
from metaculus_bot.numeric_validation import detect_unit_mismatch
from metaculus_bot.pchip_processing import log_pchip_summary, reset_pchip_stats
from metaculus_bot.performance_analysis.parsing import (
    annotate_forecaster_bullets_with_models,
    extract_model_display_name_from_reasoning,
)
from metaculus_bot.research_orchestrator import ResearchOrchestrator
from metaculus_bot.research_providers import (
    ResearchCallable,
)
from metaculus_bot.spread_metrics import compute_spread
from metaculus_bot.targeted_research import extract_disagreement_crux, run_targeted_search

# Probabilistic-tools wiring (Workstream C activation). The feature flag is
# re-exported under a descriptive local alias so call sites read clearly.
from metaculus_bot.utils.logging_utils import CompactLoggingForecastBot

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

load_environment()


class TemplateForecaster(CompactLoggingForecastBot):
    def __init__(
        self,
        *,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
        use_research_summary_to_forecast: bool = False,
        publish_reports_to_metaculus: bool = False,
        folder_to_save_reports_to: str | None = None,
        skip_previously_forecasted_questions: bool = False,
        llms: dict[str, str | GeneralLlm] | None = None,
        aggregation_strategy: AggregationStrategy = AggregationStrategy.MEAN,
        research_provider: ResearchCallable | None = None,
        max_questions_per_run: int | None = 10,
        is_benchmarking: bool = False,
        max_concurrent_research: int = DEFAULT_MAX_CONCURRENT_RESEARCH,
        allow_research_fallback: bool = True,
        research_cache: dict[int, str] | None = None,
        stacking_fallback_on_failure: bool = True,
        stacking_randomize_order: bool = True,
        stacking_spread_thresholds: dict[str, float] | None = None,
        min_forecasters_to_publish: int | None = None,
    ) -> None:
        if not isinstance(aggregation_strategy, AggregationStrategy):
            raise ValueError(f"aggregation_strategy must be an AggregationStrategy enum, got {aggregation_strategy}")
        self.aggregation_strategy: AggregationStrategy = aggregation_strategy

        setup = prepare_llm_config(
            llms=llms,
            aggregation_strategy=self.aggregation_strategy,
            predictions_per_report=predictions_per_research_report,
        )

        self._forecaster_llms: list[GeneralLlm] = setup.forecaster_llms
        self._stacker_llm: GeneralLlm | None = setup.stacker_llm
        self._analyzer_llm: GeneralLlm | None = setup.analyzer_llm
        normalized_llms: dict[str, str | GeneralLlm] = setup.normalized_llms
        predictions_per_research_report = setup.predictions_per_report

        self._custom_research_provider: ResearchCallable | None = research_provider
        self.research_provider: ResearchCallable | None = research_provider  # For framework config access
        if max_questions_per_run is not None and max_questions_per_run <= 0:
            raise ValueError("max_questions_per_run must be a positive integer if provided")
        self.max_questions_per_run: int | None = max_questions_per_run
        self.is_benchmarking: bool = is_benchmarking
        self.allow_research_fallback: bool = allow_research_fallback
        self.research_cache: dict[int, str] | None = research_cache
        # Resolve effective min-forecaster threshold. Defaults to the module
        # constant for production; tests/benchmarks can override (e.g. tests
        # typically use 2-model ensembles and would otherwise always fail the
        # guard).
        self.min_forecasters_to_publish: int = (
            min_forecasters_to_publish if min_forecasters_to_publish is not None else MIN_FORECASTERS_TO_PUBLISH
        )
        if self.min_forecasters_to_publish <= 0:
            raise ValueError(
                f"min_forecasters_to_publish must be a positive integer, got {self.min_forecasters_to_publish}"
            )
        if self.min_forecasters_to_publish > len(self._forecaster_llms):
            logger.warning(
                "min_forecasters_to_publish=%d exceeds configured forecaster count=%d; "
                "every question will fail the guard and skip publication",
                self.min_forecasters_to_publish,
                len(self._forecaster_llms),
            )
        self.stacking_fallback_on_failure: bool = stacking_fallback_on_failure
        self.stacking_randomize_order: bool = stacking_randomize_order
        # Per-question storage for stacker meta-analysis reasoning text
        self._stack_meta_reasoning: dict[int, str] = {}
        # Per-question outcome for the stacker pipeline. One of:
        #   "primary"          — primary stacker LLM produced the value
        #   "fallback_llm"     — primary failed, fallback stacker LLM succeeded
        #   "fallback_median"  — both stacker LLMs failed; MEDIAN aggregation used
        #   "skipped"          — conditional-stacking spread <= threshold
        # Must be set on the path that actually produced the aggregated value
        # (not at branch entry), so median-fallback isn't mislabeled as stacked.
        self._stacker_outcome: dict[int, str] = {}
        # Diagnostics + state for STACKING base aggregation behavior
        # Tracks per-question expectation that the base aggregator will be called to combine
        # per-research-report, already-stacked outputs.
        self._stack_expected_base_combine: set[int] = set()
        # Counters for expected vs unexpected base-combine calls during STACKING
        self._stacking_expected_combine_count: int = 0
        self._stacking_unexpected_combine_count: int = 0
        self._stacking_fallback_count: int = 0

        # --- Alerting counters (consumed by cli.py to decide sys.exit status) ---
        # Forecasters dropped by the per-call soft deadline (see FORECASTER_SOFT_DEADLINE).
        self._forecasters_dropped_count: int = 0
        # Questions that couldn't be published because fewer than
        # MIN_FORECASTERS_TO_PUBLISH base forecasters succeeded.
        self._questions_failed_to_publish: int = 0
        # Primary stacker failed (timeout or exception). Counts regardless of
        # whether fallback eventually succeeded.
        self._stacker_primary_failed_count: int = 0
        # Stacker fallback model was invoked.
        self._stacker_fallback_used_count: int = 0
        # Stacker fallback also failed; median aggregation used.
        self._stacker_fallback_failed_count: int = 0
        # Research provider failures other than AskNews subscription-inactive
        # (which is expected in off-season). Backed by self._research.timeout_count.

        # Per-question votes from each LLM on whether outcomes are discrete integers
        self._discrete_integer_votes: defaultdict[int, list[bool]] = defaultdict(list)
        # Conditional stacking thresholds (overridable per question type)
        _valid_threshold_keys = {"binary", "mc", "numeric"}
        if stacking_spread_thresholds is not None:
            unknown_keys = set(stacking_spread_thresholds) - _valid_threshold_keys
            if unknown_keys:
                raise ValueError(
                    f"Unknown stacking_spread_thresholds keys: {unknown_keys}. Valid keys: {_valid_threshold_keys}"
                )
        self._stacking_spread_thresholds: dict[str, float] = {
            "binary": CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD,
            "mc": CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD,
            "numeric": CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD,
        } | (stacking_spread_thresholds or {})
        self._conditional_stacking_triggered_count: int = 0
        self._conditional_stacking_skipped_count: int = 0
        self._conditional_stacking_crux_failures: int = 0
        self._conditional_stacking_search_failures: int = 0

        if max_concurrent_research <= 0:
            raise ValueError("max_concurrent_research must be a positive integer")
        # Persist for framework config introspection and logging
        self.max_concurrent_research: int = max_concurrent_research
        # Instance-level semaphore to avoid cross-instance throttling
        self._concurrency_limiter: asyncio.Semaphore = asyncio.Semaphore(max_concurrent_research)

        super().__init__(
            research_reports_per_question=research_reports_per_question,
            predictions_per_research_report=predictions_per_research_report,
            use_research_summary_to_forecast=use_research_summary_to_forecast,
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            llms=normalized_llms,  # type: ignore[arg-type]  # dict value type lacks None but parent expects Optional
        )

        self._research = ResearchOrchestrator(
            default_llm=self.get_llm("default", "llm"),
            summarizer_llm=self.get_llm("summarizer", "llm"),
            custom_provider=research_provider,
            research_cache=research_cache,
            is_benchmarking=is_benchmarking,
            allow_research_fallback=allow_research_fallback,
            max_concurrent_research=max_concurrent_research,
        )

        # Log ensemble + aggregation configuration once on init
        num_models = len(self._forecaster_llms) if self._forecaster_llms else 1
        logger.info(
            "Ensemble configured: %s model(s) | Aggregation: %s",
            num_models,
            self.aggregation_strategy.value,
        )
        if self.aggregation_strategy == AggregationStrategy.STACKING:
            stacker_name = self._stacker_llm.model if self._stacker_llm else "<missing>"
            base_models = [m.model for m in self._forecaster_llms]
            short_list = base_models if len(base_models) <= 6 else base_models[:6] + ["..."]
            logger.info(
                "STACKING config | stacker=%s | base_forecasters(%d)=%s | final_outputs_per_question=1",
                stacker_name,
                len(base_models),
                short_list,
            )
        elif self.aggregation_strategy == AggregationStrategy.CONDITIONAL_STACKING:
            stacker_name = self._stacker_llm.model if self._stacker_llm else "<missing>"
            analyzer_name = self._analyzer_llm.model if self._analyzer_llm else "<missing>"
            base_models = [m.model for m in self._forecaster_llms]
            short_list = base_models if len(base_models) <= 6 else base_models[:6] + ["..."]
            logger.info(
                "CONDITIONAL_STACKING config | stacker=%s | analyzer=%s | base_forecasters(%d)=%s | thresholds=%s",
                stacker_name,
                analyzer_name,
                len(base_models),
                short_list,
                self._stacking_spread_thresholds,
            )

    def _get_threshold_for_question(self, question: MetaculusQuestion) -> float:
        """Return the spread threshold for the given question type."""
        if isinstance(question, BinaryQuestion):
            return self._stacking_spread_thresholds["binary"]
        if isinstance(question, MultipleChoiceQuestion):
            return self._stacking_spread_thresholds["mc"]
        if isinstance(question, NumericQuestion):
            return self._stacking_spread_thresholds["numeric"]
        raise ValueError(f"No spread threshold for question type: {type(question).__name__}")

    def _register_expected_base_combine(self, question: MetaculusQuestion) -> None:
        """Register that the framework's base aggregator should expect a combine call for this question.

        Relies on the assertion at the top of ``_research_and_make_predictions`` that
        ``question.id_of_question is not None`` — so that upstream stacking state-dict
        ops (``self._stacker_outcome[qid]``, ``self._stack_meta_reasoning[qid]``)
        and this set's key stay consistent. A silent fallback to ``id(question)`` here
        would let the keys desync if upstream keying ever changed.
        """
        qid = question.id_of_question
        assert qid is not None, "_register_expected_base_combine requires question.id_of_question"
        self._stack_expected_base_combine.add(qid)

    async def forecast_questions(
        self,
        questions: Sequence[MetaculusQuestion],
        return_exceptions: bool = False,
    ) -> list[ForecastReport] | list[ForecastReport | BaseException]:
        # Apply skip filter first (mirrors base class behavior) so we cap unforecasted items
        if self.skip_previously_forecasted_questions:
            unforecasted_questions = [q for q in questions if not q.already_forecasted]
            if len(questions) != len(unforecasted_questions):
                logger.info(f"Skipping {len(questions) - len(unforecasted_questions)} previously forecasted questions")
            questions = unforecasted_questions

        # Enforce max questions per run safety cap
        if self.max_questions_per_run is not None and len(questions) > self.max_questions_per_run:
            logger.info(f"Limiting to first {self.max_questions_per_run} questions out of {len(questions)}")
            questions = list(questions)[: self.max_questions_per_run]

        # Log question processing info with progress
        if questions:
            bot_name = getattr(self, "name", "Bot")
            logger.info(f"📊 {bot_name}: Processing {len(questions)} questions...")

        reset_pchip_stats()

        results = await super().forecast_questions(questions, return_exceptions)

        log_pchip_summary()

        if self.aggregation_strategy == AggregationStrategy.CONDITIONAL_STACKING:
            logger.info(
                "Conditional stacking summary: triggered=%d, skipped=%d, crux_failures=%d, search_failures=%d",
                self._conditional_stacking_triggered_count,
                self._conditional_stacking_skipped_count,
                self._conditional_stacking_crux_failures,
                self._conditional_stacking_search_failures,
            )

        # Loud end-of-run degradation summary. Any non-zero counter here means
        # something got dropped, the stacker fell back, or a research provider
        # failed — all states where CI (cli.py) should exit non-zero so we get
        # paged, but every publishable question has already been published.
        logger.info(
            "Degradation counters: forecasters_dropped=%d, questions_failed_to_publish=%d, "
            "stacker_primary_failed=%d, stacker_fallback_used=%d, stacker_fallback_failed=%d, "
            "research_provider_timeouts=%d",
            self._forecasters_dropped_count,
            self._questions_failed_to_publish,
            self._stacker_primary_failed_count,
            self._stacker_fallback_used_count,
            self._stacker_fallback_failed_count,
            self._research_provider_timeout_count,
        )

        return results

    @property
    def _research_provider_timeout_count(self) -> int:
        return self._research.timeout_count

    @_research_provider_timeout_count.setter
    def _research_provider_timeout_count(self, value: int) -> None:
        self._research.timeout_count = value

    @property
    def alertable_count(self) -> int:
        """Sum of counters whose non-zero value should page us.

        Consumed by cli.py to decide whether to sys.exit(1) after all
        publications complete. Any individual non-zero counter is enough to
        trip the alert; the sum is just a convenient single number.
        """
        return (
            self._forecasters_dropped_count
            + self._questions_failed_to_publish
            + self._stacker_primary_failed_count
            + self._stacker_fallback_used_count
            + self._stacker_fallback_failed_count
            + self._research_provider_timeout_count
        )

    async def _run_stacking(
        self,
        question: MetaculusQuestion,
        research: str,
        reasoned_predictions: list[ReasonedPrediction[PredictionTypes]],
        stacker_llm_override: GeneralLlm | None = None,
        aggregated_tool_output: str | None = None,
    ) -> PredictionTypes:
        """Run stacking to aggregate multiple model predictions using a meta-model.

        ``stacker_llm_override`` lets the aggregation layer invoke a fallback
        stacker (e.g. gpt-5.5) without swapping ``self._stacker_llm`` — keeping
        fallback behavior local to the call site and avoiding state mutation
        that could race with concurrent questions.

        ``aggregated_tool_output`` is the optional markdown block produced
        upstream by ``build_cross_model_aggregation``; threaded into
        ``run_stacking_*`` so the stacker sees deterministic cross-model
        math at the top of its prompt.
        """
        if stacker_llm_override is not None:
            stacker_llm = stacker_llm_override
        else:
            if self._stacker_llm is None:
                raise ValueError("No stacker LLM configured")
            stacker_llm = self._stacker_llm

        page_url = question.page_url or "<unknown>"
        qid = question.id_of_question
        assert qid is not None, "_run_stacking requires question.id_of_question (upstream guarantees this)"

        # Strip model names from reasoning and prepare base predictions
        base_predictions = [stacking.strip_model_tag(pred.reasoning) for pred in reasoned_predictions]

        # Optionally randomize order to avoid position bias
        if self.stacking_randomize_order:
            combined = list(zip(base_predictions, reasoned_predictions))
            random.shuffle(combined)
            base_predictions = [bp for bp, _ in combined]
            reasoned_predictions = [rp for _, rp in combined]

        # Generate appropriate stacking call based on question type
        if isinstance(question, BinaryQuestion):
            value, meta_text = await stacking.run_stacking_binary(
                stacker_llm,
                self.get_llm("parser", "llm"),
                question,
                research,
                base_predictions,
                aggregated_tool_output=aggregated_tool_output,
            )
            self._log_llm_output(stacker_llm, qid, meta_text)
            self._stack_meta_reasoning[qid] = meta_text
            logger.info(f"Stacked binary prediction for {page_url}: {value}")
            return value
        elif isinstance(question, MultipleChoiceQuestion):
            pol, meta_text = await stacking.run_stacking_mc(
                stacker_llm,
                self.get_llm("parser", "llm"),
                question,
                research,
                base_predictions,
                aggregated_tool_output=aggregated_tool_output,
            )
            self._log_llm_output(stacker_llm, qid, meta_text)
            self._stack_meta_reasoning[qid] = meta_text
            logger.info(f"Stacked multiple choice prediction for {page_url}: {pol}")
            return pol
        elif isinstance(question, NumericQuestion):
            upper_msg, lower_msg = bound_messages(question)
            perc_list, meta_text = await stacking.run_stacking_numeric(
                stacker_llm,
                self.get_llm("parser", "llm"),
                question,
                research,
                base_predictions,
                lower_msg,
                upper_msg,
                aggregated_tool_output=aggregated_tool_output,
            )
            self._log_llm_output(stacker_llm, qid, meta_text)
            self._stack_meta_reasoning[qid] = meta_text

            # Use same validation and processing logic as base numeric forecasting
            percentile_list, zero_point = sanitize_percentiles(list(perc_list), question)

            # question is narrowed to NumericQuestion by the elif, but the type checker
            # only sees MetaculusQuestion from the method signature
            mismatch, reason = detect_unit_mismatch(percentile_list, question)  # type: ignore[arg-type]
            if mismatch:
                from metaculus_bot.exceptions import UnitMismatchError

                logger.error(
                    f"Unit mismatch likely for Q {qid} | URL {page_url} | reason={reason}. Withholding prediction."
                )
                raise UnitMismatchError(
                    f"Unit mismatch likely; {reason}. Values: {[float(p.value) for p in percentile_list]}"
                )

            prediction = build_numeric_distribution(percentile_list, question, zero_point)
            log_final_prediction(prediction, question)
            logger.info(f"Stacked numeric prediction for {page_url}")
            return prediction
        else:
            raise ValueError(f"Unsupported question type for stacking: {type(question)}")

    async def run_research(self, question: MetaculusQuestion) -> str:
        return await self._research.run_research(question)

    async def summarize_research(self, question: MetaculusQuestion, research: str) -> str:
        return await self._research.summarize_research(question, research)

    def _select_research_providers(self) -> list[tuple[ResearchCallable, str]]:
        return self._research._select_research_providers()

    async def _run_providers_parallel(
        self,
        question: MetaculusQuestion,
        providers: list[tuple[ResearchCallable, str]],
    ) -> str:
        return await self._research._run_providers_parallel(question, providers)

    def _select_research_provider(self) -> tuple[ResearchCallable, str]:
        return self._research._select_research_provider()

    async def _call_perplexity(self, question: MetaculusQuestion | str, use_open_router: bool = True) -> str:
        return await self._research._call_perplexity(question, use_open_router=use_open_router)

    async def _call_exa_smart_searcher(self, question: MetaculusQuestion | str) -> str:
        return await self._research._call_exa_smart_searcher(question)

    # Override _research_and_make_predictions to support multiple LLMs
    def _remaining_budget_seconds(self, start_time: float) -> float:
        """Return remaining per-Q wall-clock budget in seconds (can go negative).

        ``start_time`` is captured at the top of ``_research_and_make_predictions``
        and represents this question's processing-start tick. Compared against
        ``PER_QUESTION_WALL_CLOCK_DEADLINE`` (58:30 of the 60-min Metaculus close
        window).
        """
        return PER_QUESTION_WALL_CLOCK_DEADLINE - (time.time() - start_time)

    async def _gather_predictions_with_wall_clock(
        self,
        coros: list[Coroutine[Any, Any, ReasonedPrediction[Any]]],
        qid_for_log: int,
        per_q_start: float,
    ) -> tuple[list[ReasonedPrediction[PredictionTypes]], list[str], ExceptionGroup | None]:
        """Run forecaster coroutines concurrently with a wall-clock cap.

        Differs from the parent ``_gather_results_and_exceptions`` in two ways:
        - Pending tasks at deadline are cancelled (parent's ``asyncio.gather``
          can't cancel mid-flight).
        - Drops counter is bumped on cancellation so end-of-run alerting
          surfaces the abort.

        Mirrors ``_gather_results_and_exceptions`` return shape so callers
        treat it identically. Tests can patch this method directly to inject
        a synthetic prediction list without spinning up real tasks.
        """
        tasks = [asyncio.create_task(coro, name=f"forecaster:{idx}:q{qid_for_log}") for idx, coro in enumerate(coros)]
        n_total = len(tasks)
        remaining = self._remaining_budget_seconds(per_q_start)
        wait_timeout = max(0.0, remaining)
        done_set, pending_set = await asyncio.wait(tasks, timeout=wait_timeout, return_when=asyncio.ALL_COMPLETED)
        if pending_set:
            for pending in pending_set:
                pending.cancel()
            # Give cancelled tasks a chance to clean up so we don't leak warnings.
            await asyncio.wait(pending_set, timeout=2.0)
            self._forecasters_dropped_count += len(pending_set)
            logger.warning(
                "WALLCLOCK_ABORT: qid=%s elapsed=%.1fs forecasters_completed=%d/%d cancelled=%d remaining_budget=%.1fs",
                qid_for_log,
                time.time() - per_q_start,
                len(done_set),
                n_total,
                len(pending_set),
                remaining,
            )

        # Sort by task name (stable across runs) since asyncio.wait returns sets.
        done_sorted = sorted(done_set, key=lambda t: t.get_name())
        valid_predictions: list[ReasonedPrediction[PredictionTypes]] = []
        errors: list[str] = []
        exceptions: list[BaseException] = []
        for task in done_sorted:
            exc = task.exception()
            if exc is None:
                valid_predictions.append(cast(ReasonedPrediction[PredictionTypes], task.result()))
            else:
                errors.append(f"{type(exc).__name__}: {exc}")
                exceptions.append(exc)
        exception_group: ExceptionGroup | None = ExceptionGroup(f"Errors: {errors}", exceptions) if exceptions else None
        return valid_predictions, errors, exception_group

    async def _research_and_make_predictions(
        self,
        question: MetaculusQuestion,
    ) -> ResearchWithPredictions[PredictionTypes]:
        # Call the parent class's method if no specific forecaster LLMs are provided
        if not self._forecaster_llms:
            return await super()._research_and_make_predictions(question)

        assert question.id_of_question is not None, "id_of_question must not be None for stacking state-dict keying"

        # Per-Q wall-clock cutoff: research, fan-out, aggregation, and publish
        # all share the same budget. Recorded as early as possible so we don't
        # overshoot from research-time alone.
        per_q_start = time.time()

        notepad = await self._get_notepad(question)
        notepad.total_research_reports_attempted += 1
        research = await self.run_research(question)

        # Only call summarizer if we plan to use the summary for forecasting
        if self.use_research_summary_to_forecast:
            summary_report = await self.summarize_research(question, research)
            research_to_use = summary_report
        else:
            summary_report = research  # Use raw research for reporting compatibility
            research_to_use = research

        qid_for_log = question.id_of_question
        tasks = cast(
            list[Coroutine[Any, Any, ReasonedPrediction[Any]]],
            [
                self._forecaster_with_soft_deadline(question, research_to_use, llm_instance, qid_for_log)
                for llm_instance in self._forecaster_llms
            ],
        )
        (
            valid_predictions,
            errors,
            exception_group,
        ) = await self._gather_predictions_with_wall_clock(tasks, qid_for_log, per_q_start)
        if errors:
            logger.warning(f"Encountered errors while predicting: {errors}")

        # Min-forecasters guard: below self.min_forecasters_to_publish, the
        # ensemble is too degraded to publish. Increment counter for end-of-run
        # alerting and raise so this question is skipped (but other batch
        # questions and publication continue). See cli.py for exit-status wiring.
        n_valid = len(valid_predictions)
        if n_valid < self.min_forecasters_to_publish:
            self._questions_failed_to_publish += 1
            msg = (
                f"Only {n_valid}/{len(self._forecaster_llms)} forecasters succeeded for Q {qid_for_log} "
                f"(need >= {self.min_forecasters_to_publish}); skipping publication."
            )
            logger.error(msg)
            if exception_group is not None:
                self._reraise_exception_with_prepended_message(exception_group, msg)
            raise RuntimeError(msg)
        # Stacking budget gate. If we've burned through the per-Q wall-clock
        # budget (e.g. research stalled, fan-out used most of the budget),
        # skip the stacker LLM entirely and force the MEDIAN fallback. Typical
        # publish is ~1s; the WALL_CLOCK_STACKING_MIN_BUDGET (90s) floor leaves
        # headroom for sustained slowness on a single POST. The 160s worst case
        # (4 POSTs * 20s * (1 + 1 retry)) requires multi-POST stalling, which
        # is recovered by skip_stacking_for_budget already.
        skip_stacking_for_budget = (
            self.aggregation_strategy in (AggregationStrategy.STACKING, AggregationStrategy.CONDITIONAL_STACKING)
            and self._remaining_budget_seconds(per_q_start) < WALL_CLOCK_STACKING_MIN_BUDGET
        )
        if skip_stacking_for_budget:
            # F15: the base-combine re-entry uses MEAN under STACKING and
            # MEDIAN under CONDITIONAL_STACKING (see _aggregate_predictions:
            # base_combine_strategy at main.py:1310-1314). The marker must
            # match the actual aggregation method so residual analysis cuts
            # bucket the two paths correctly.
            budget_skip_outcome = (
                "fallback_median"
                if self.aggregation_strategy == AggregationStrategy.CONDITIONAL_STACKING
                else "fallback_mean"
            )
            logger.warning(
                "WALLCLOCK_ABORT: skipping stacking for Q %s; remaining=%.1fs < %ds; forcing %s fallback",
                qid_for_log,
                self._remaining_budget_seconds(per_q_start),
                WALL_CLOCK_STACKING_MIN_BUDGET,
                budget_skip_outcome,
            )
            self._stacker_outcome[question.id_of_question] = budget_skip_outcome
            # Register so parent's _aggregate_predictions (which will run with
            # reasoned_predictions=None) takes the expected base-combine path
            # and doesn't log "Unexpected STACKING combine".
            self._register_expected_base_combine(question)

        # If using stacking, aggregate the predictions here
        if self.aggregation_strategy == AggregationStrategy.STACKING and not skip_stacking_for_budget:
            if getattr(self, "research_reports_per_question", 1) != 1:
                logger.warning(
                    "STACKING configured with research_reports_per_question=%s; final results will average per-report stacked outputs by mean.",
                    getattr(self, "research_reports_per_question", 1),
                )
            prediction_values = [pred.prediction_value for pred in valid_predictions]
            # Probabilistic tools: deterministic cross-model math runs once
            # per question and rides at the top of the stacker prompt. No-ops
            # when PROBABILISTIC_TOOLS_ENABLED is unset.
            from metaculus_bot.tool_runner import (
                build_cross_model_aggregation,  # noqa: PLC0415  # function-scoped: see AGENTS.md
            )

            aggregated_tool_output = (
                build_cross_model_aggregation(
                    question=question,
                    rationales=[p.reasoning for p in valid_predictions],
                    prediction_values=prediction_values,
                )
                or None
            )
            aggregated_value = await self._aggregate_predictions(
                prediction_values,
                question,
                research=research_to_use,
                reasoned_predictions=valid_predictions,
                aggregated_tool_output=aggregated_tool_output,
            )
            # Create a single aggregated prediction, preserving the stacker meta-analysis
            # AND the base model reasonings (so residual analysis can recover per-model
            # attribution even when stacking overrode the base-aggregation output).
            meta_text = self._stack_meta_reasoning.pop(
                question.id_of_question,
                "Stacked prediction aggregated from multiple models",
            )
            combined_reasoning = stacking.combine_stacker_and_base_reasoning(meta_text, valid_predictions)
            aggregated_prediction = ReasonedPrediction(prediction_value=aggregated_value, reasoning=combined_reasoning)
            self._register_expected_base_combine(question)
            # _stacker_outcome is populated by _aggregate_predictions on the path
            # that actually produced aggregated_value; do not set it here.
            return ResearchWithPredictions(
                research_report=research,
                summary_report=summary_report,
                errors=errors,
                predictions=[aggregated_prediction],
            )
        elif self.aggregation_strategy == AggregationStrategy.CONDITIONAL_STACKING and not skip_stacking_for_budget:
            prediction_values = [pred.prediction_value for pred in valid_predictions]
            spread = compute_spread(question, prediction_values)
            threshold = self._get_threshold_for_question(question)

            # Per-question-type stacking gates. All three default to ENABLED. Set
            # <TYPE>_STACKING_ENABLED=false in deploy env to bypass stacking for that
            # type (forces median/skipped path).
            spread_exceeds_threshold = spread > threshold
            type_to_stacking_env = {
                BinaryQuestion: BINARY_STACKING_ENABLED_ENV,
                MultipleChoiceQuestion: MC_STACKING_ENABLED_ENV,
                NumericQuestion: NUMERIC_STACKING_ENABLED_ENV,
            }
            for q_type, env_name in type_to_stacking_env.items():
                if isinstance(question, q_type):
                    if not env_flag_enabled(env_name, default=True):
                        spread_exceeds_threshold = False
                    break

            if spread_exceeds_threshold:
                self._conditional_stacking_triggered_count += 1
                logger.info(
                    "Conditional stacking TRIGGERED: spread=%.3f > threshold=%.3f for question %s",
                    spread,
                    threshold,
                    question.id_of_question,
                )

                if self._stacker_llm is None:
                    raise ValueError("CONDITIONAL_STACKING requires a stacker LLM to be configured")
                if self._analyzer_llm is None:
                    raise ValueError("CONDITIONAL_STACKING requires an analyzer LLM to be configured")

                # 1. Extract the crux of disagreement under a soft deadline.
                # Without the wait_for the call's worst case is litellm timeout
                # (300s) * allowed_tries (3) ≈ 15 min on the critical path; the
                # CRUX_SOFT_DEADLINE caps that at 180s.
                base_texts = [stacking.strip_model_tag(pred.reasoning) for pred in valid_predictions]
                try:
                    crux = await asyncio.wait_for(
                        extract_disagreement_crux(
                            self._analyzer_llm,
                            question.question_text,
                            base_texts,
                        ),
                        timeout=CRUX_SOFT_DEADLINE,
                    )
                except asyncio.TimeoutError:
                    self._conditional_stacking_crux_failures += 1
                    logger.warning(
                        "CRUX_SOFT_DEADLINE: crux extraction exceeded %ds for Q %s; skipping targeted research",
                        CRUX_SOFT_DEADLINE,
                        question.id_of_question,
                    )
                    crux = ""
                except Exception:
                    self._conditional_stacking_crux_failures += 1
                    logger.exception("Disagreement crux extraction failed, skipping targeted research")
                    crux = ""

                # 2. Run targeted research if crux was extracted
                targeted_research_text = ""
                if crux:
                    try:
                        targeted_research_text = await run_targeted_search(
                            crux, question.question_text, is_benchmarking=self.is_benchmarking
                        )
                    except Exception:
                        self._conditional_stacking_search_failures += 1
                        logger.exception("Targeted search failed, proceeding with base research only")

                # 3. Combine research
                if targeted_research_text:
                    combined_research = (
                        f"{research_to_use}\n\n"
                        f"## Targeted Research (addressing model disagreement)\n"
                        f"{targeted_research_text}"
                    )
                else:
                    combined_research = research_to_use

                # 4. Run stacking
                # Cross-model aggregation for tool-augmented runs (no-op when
                # PROBABILISTIC_TOOLS_ENABLED is unset).
                from metaculus_bot.tool_runner import (
                    build_cross_model_aggregation,  # noqa: PLC0415  # function-scoped: see AGENTS.md
                )

                aggregated_tool_output = (
                    build_cross_model_aggregation(
                        question=question,
                        rationales=[p.reasoning for p in valid_predictions],
                        prediction_values=prediction_values,
                    )
                    or None
                )
                aggregated_value = await self._aggregate_predictions(
                    prediction_values,
                    question,
                    research=combined_research,
                    reasoned_predictions=valid_predictions,
                    aggregated_tool_output=aggregated_tool_output,
                )
                meta_text = self._stack_meta_reasoning.pop(
                    question.id_of_question,
                    "Conditional stacking: aggregated from multiple models after high-disagreement detected",
                )
                combined_reasoning = stacking.combine_stacker_and_base_reasoning(meta_text, valid_predictions)
                aggregated_prediction = ReasonedPrediction(
                    prediction_value=aggregated_value, reasoning=combined_reasoning
                )
                self._register_expected_base_combine(question)
                # _stacker_outcome is populated by _aggregate_predictions on the
                # path that actually produced aggregated_value.
                #
                # research_report must be combined_research so the
                # ## Targeted Research (addressing model disagreement) header
                # reaches the published comment. Note: when
                # use_research_summary_to_forecast=True, combined_research is
                # built from summary_report (research_to_use), so the comment
                # will show summary + targeted_research; the regular STACKING
                # path above instead publishes raw research.
                return ResearchWithPredictions(
                    research_report=combined_research,
                    summary_report=summary_report,
                    errors=errors,
                    predictions=[aggregated_prediction],
                )
            else:
                self._conditional_stacking_skipped_count += 1
                logger.info(
                    "Conditional stacking SKIPPED: spread=%.3f <= threshold=%.3f for question %s",
                    spread,
                    threshold,
                    question.id_of_question,
                )
                self._register_expected_base_combine(question)
                self._stacker_outcome[question.id_of_question] = "skipped"
                return ResearchWithPredictions(
                    research_report=research,
                    summary_report=summary_report,
                    errors=errors,
                    predictions=valid_predictions,
                )

        # Catch-all: non-stacking strategy, OR stacking strategy whose budget
        # gate forced fallback_median above. In both cases we return the raw
        # valid_predictions and let the parent class's per-Q aggregator combine
        # them. For the skip case, _stacker_outcome was already set to
        # "fallback_median" upstream so the comment-marker reflects reality.
        return ResearchWithPredictions(
            research_report=research,
            summary_report=summary_report,
            errors=errors,
            predictions=valid_predictions,
        )

    @classmethod
    def _format_and_expand_research_summary(
        cls,
        report_number: int,
        report_type: type[ForecastReport],
        predicted_research: ResearchWithPredictions,
    ) -> str:
        text = super()._format_and_expand_research_summary(report_number, report_type, predicted_research)
        # Inject model name into summary bullets so per-model attribution survives comment trimming.
        model_names_by_index: dict[int, str] = {}
        for j, forecast in enumerate(predicted_research.predictions):
            name = extract_model_display_name_from_reasoning(forecast.reasoning)
            if name is not None:
                model_names_by_index[j + 1] = name
        text = annotate_forecaster_bullets_with_models(text, model_names_by_index)
        return trim_section(text, f"report_{report_number}_summary")

    @classmethod
    def _format_main_research(
        cls,
        report_number: int,
        predicted_research: ResearchWithPredictions,
    ) -> str:
        text = super()._format_main_research(report_number, predicted_research)
        return trim_section(text, f"report_{report_number}_research")

    def _format_forecaster_rationales(
        self,
        report_number: int,
        collection: ResearchWithPredictions,
    ) -> str:
        text = super()._format_forecaster_rationales(report_number, collection).lstrip()
        return trim_section(text, f"report_{report_number}_rationales")

    def _create_unified_explanation(
        self,
        question: MetaculusQuestion,
        research_prediction_collections: list[ResearchWithPredictions],
        aggregated_prediction: PredictionTypes,
        final_cost: float,
        time_spent_in_minutes: float,
    ) -> str:
        from metaculus_bot.comment_formatting import build_unified_explanation

        base_text = super()._create_unified_explanation(
            question,
            research_prediction_collections,
            aggregated_prediction,
            final_cost,
            time_spent_in_minutes,
        )
        qid = question.id_of_question
        stacker_outcome = self._stacker_outcome.pop(qid, None) if qid is not None else None
        return build_unified_explanation(base_text, question, self.aggregation_strategy, stacker_outcome)

    async def _forecaster_with_soft_deadline(
        self,
        question: MetaculusQuestion,
        research: str,
        llm: GeneralLlm,
        qid: int | None,
    ) -> ReasonedPrediction[PredictionTypes]:
        """Run a single forecaster with FORECASTER_SOFT_DEADLINE.

        Why the deadline: a single stuck forecaster used to be able to hold a
        question for litellm_timeout(480s) * allowed_tries(3) ≈ 24 min. This
        caps each forecaster at FORECASTER_SOFT_DEADLINE (10 min).

        On timeout: bumps _forecasters_dropped_count, logs a loud WARNING
        identifying the model + question, and re-raises TimeoutError so the
        caller's _gather_results_and_exceptions treats it like any other failed
        forecaster (dropped from the ensemble; the other models in the ensemble
        carry the question).
        """
        start = time.monotonic()
        try:
            return await asyncio.wait_for(
                self._make_prediction(question, research, llm),
                timeout=FORECASTER_SOFT_DEADLINE,
            )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            self._forecasters_dropped_count += 1
            logger.warning(
                "SOFT_DEADLINE: forecaster model=%s qid=%s exceeded %ds (elapsed=%.0fs); dropping this forecaster",
                llm.model,
                qid,
                FORECASTER_SOFT_DEADLINE,
                elapsed,
            )
            raise

    async def _make_prediction(
        self,
        question: MetaculusQuestion,
        research: str,
        llm_to_use: GeneralLlm | None = None,
    ) -> ReasonedPrediction[PredictionTypes]:
        notepad = await self._get_notepad(question)
        notepad.total_predictions_attempted += 1

        actual_llm = llm_to_use if llm_to_use else self.get_llm("default", "llm")

        if isinstance(question, BinaryQuestion):

            def forecast_function(q, r, llm):
                return self._run_forecast_on_binary(q, r, llm)
        elif isinstance(question, MultipleChoiceQuestion):

            def forecast_function(q, r, llm):
                return self._run_forecast_on_multiple_choice(q, r, llm)
        elif isinstance(question, NumericQuestion):

            def forecast_function(q, r, llm):
                return self._run_forecast_on_numeric(q, r, llm)
        elif isinstance(question, DateQuestion):
            raise NotImplementedError("Date questions not supported yet")
        else:
            raise ValueError(f"Unknown question type: {type(question)}")

        prediction = await forecast_function(question, research, actual_llm)
        # Load-bearing: performance_analysis.parsing pulls per-model attribution from this "Model:" prefix.
        prediction.reasoning = f"Model: {actual_llm.model}\n\n{prediction.reasoning}"

        # Probabilistic-tools activation: run deterministic math tools over
        # the forecaster's structured JSON block (see tool_runner.py). The
        # tool runner no-ops when PROBABILISTIC_TOOLS_ENABLED is off or no
        # block was emitted; callers don't gate.
        from metaculus_bot.tool_runner import (
            run_tools_for_forecaster,  # noqa: PLC0415  # function-scoped: see AGENTS.md
        )

        computed_md = run_tools_for_forecaster(
            question=question,
            rationale=prediction.reasoning,
            forecaster_id=actual_llm.model,
        )
        if computed_md:
            prediction.reasoning = f"{prediction.reasoning}\n\n## Computed quantities\n{computed_md}"
        # Each branch returns a specific ReasonedPrediction[T] but the signature
        # requires ReasonedPrediction[PredictionTypes]; framework has the same pattern
        return prediction  # type: ignore[return-value]

    async def _aggregate_predictions(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
        research: str | None = None,
        reasoned_predictions: list[ReasonedPrediction[PredictionTypes]] | None = None,
        aggregated_tool_output: str | None = None,
    ) -> PredictionTypes:
        if not predictions:
            raise ValueError("Cannot aggregate empty list of predictions")

        # Base aggregator calls when using STACKING.
        # If the base class calls aggregation after we've already stacked per research-report,
        # there will be no reasoned_predictions/research context provided here.
        # Treat this as a base-combine. Distinguish expected vs unexpected for logging.
        if (
            self.aggregation_strategy in (AggregationStrategy.STACKING, AggregationStrategy.CONDITIONAL_STACKING)
            and reasoned_predictions is None
            and research is None
        ):
            qkey = question.id_of_question

            expected = qkey in self._stack_expected_base_combine
            if expected:
                self._stack_expected_base_combine.discard(qkey)
                self._stacking_expected_combine_count += 1
            else:
                self._stacking_unexpected_combine_count += 1

            # Single pre-stacked prediction – return as-is
            if len(predictions) == 1:
                if expected:
                    logger.info("STACKING base combine: single pre-stacked output; returning as-is")
                else:
                    logger.warning(
                        "Unexpected STACKING combine: single input without stacking context; returning as-is"
                    )
                return predictions[0]
            # Multiple predictions – combine them. CONDITIONAL_STACKING uses MEDIAN (its low-spread
            # skip path returns all individual predictions); regular STACKING uses MEAN.
            base_combine_strategy = (
                AggregationStrategy.MEDIAN
                if self.aggregation_strategy == AggregationStrategy.CONDITIONAL_STACKING
                else AggregationStrategy.MEAN
            )
            strategy_name = base_combine_strategy.value
            if expected:
                logger.info(
                    "STACKING base combine: %d pre-stacked outputs; aggregating by %s for final output",
                    len(predictions),
                    strategy_name,
                )
            else:
                logger.warning(
                    "Unexpected STACKING combine: %d inputs without stacking context; aggregating by %s",
                    len(predictions),
                    strategy_name,
                )
            first = predictions[0]
            # F16: under CONDITIONAL_STACKING the multi-input re-entry case is the
            # low-spread skip path, where `predictions` are RAW per-forecaster
            # outputs (not yet Platt-calibrated). Apply Platt to the combined
            # value so low-spread vs. high-spread questions have symmetric
            # calibration treatment. STACKING's multi-input case is per-research-
            # report stacker outputs that were already calibrated by the fresh-
            # aggregation path that produced them — re-applying would double-
            # apply, so leave that branch alone.
            apply_platt_after_combine = self.aggregation_strategy == AggregationStrategy.CONDITIONAL_STACKING
            # In the branches below, isinstance narrows `first` but the checker can't
            # narrow the full `predictions` list or know that combine_* returns a
            # PredictionTypes member.  The return-value ignores are safe because each
            # concrete type (float, PredictedOptionList, NumericDistribution) IS a
            # member of PredictionTypes.
            if isinstance(first, (int, float)):
                values = [float(p) for p in predictions if isinstance(p, (int, float))]
                result = combine_binary_predictions(values, base_combine_strategy)
                logger.info("STACKING base combine: binary %s of %s = %.3f", strategy_name, values, result)
                if apply_platt_after_combine:
                    return self._apply_platt_calibration(result, question)  # type: ignore[arg-type]
                return result  # type: ignore[return-value]
            if isinstance(first, PredictedOptionList):
                mc_preds = [p for p in predictions if isinstance(p, PredictedOptionList)]
                aggregated = combine_multiple_choice_predictions(mc_preds, base_combine_strategy)
                summary = {o.option_name: round(o.probability, 4) for o in aggregated.predicted_options}
                logger.info("STACKING base combine: MC %s aggregation | %s", strategy_name, summary)
                if apply_platt_after_combine:
                    return self._apply_platt_calibration(aggregated, question)  # type: ignore[arg-type]
                return aggregated  # type: ignore[return-value]
            if isinstance(first, NumericDistribution) and isinstance(question, NumericQuestion):
                numeric_preds = [p for p in predictions if isinstance(p, NumericDistribution)]
                aggregated = await combine_numeric_predictions(numeric_preds, question, base_combine_strategy)
                logger.info(
                    "STACKING base combine: numeric %s aggregation | CDF points=%d",
                    strategy_name,
                    len(getattr(aggregated, "cdf", [])),
                )
                snapped = self._maybe_snap_to_integers(aggregated, question)
                if apply_platt_after_combine:
                    return self._apply_platt_calibration(snapped, question)  # type: ignore[arg-type]
                return snapped  # type: ignore[return-value]
            raise ValueError(f"Unsupported prediction type for STACKING base combine: {type(first)}")

        # Handle stacking strategy
        if self.aggregation_strategy in (AggregationStrategy.STACKING, AggregationStrategy.CONDITIONAL_STACKING):
            if self._stacker_llm is None:
                raise ValueError("STACKING aggregation strategy requires a stacker LLM to be configured")
            if reasoned_predictions is None:
                raise ValueError("STACKING aggregation strategy requires reasoned predictions")
            if research is None:
                raise ValueError("STACKING aggregation strategy requires research context")

            # Stacker fallback chain:
            #   1. Primary stacker (opus-4.7) under STACKER_SOFT_DEADLINE.
            #   2. On failure/timeout: secondary stacker (gpt-5.5, different
            #      provider so Anthropic-API thrash doesn't take us down) under
            #      STACKER_FALLBACK_SOFT_DEADLINE.
            #   3. On both failing: MEDIAN aggregation (not MEAN — MEAN on
            #      disagreeing opus/gpt/gemini distributions can produce a weird
            #      multi-modal numeric output; MEDIAN is robust).
            #
            # stacking_fallback_on_failure=False (used in benchmarks) preserves
            # hard-fail behavior end-to-end so benchmark runs surface stacker
            # regressions loudly.
            # CancelledError is BaseException in 3.11+ — intentionally not caught here.
            # _research_and_make_predictions asserts id_of_question is not None
            # before this is reachable, so the qid is guaranteed populated.
            qid_for_outcome = question.id_of_question
            assert qid_for_outcome is not None
            try:
                stacked = await asyncio.wait_for(
                    self._run_stacking(
                        question,
                        research,
                        reasoned_predictions,
                        aggregated_tool_output=aggregated_tool_output,
                    ),
                    timeout=STACKER_SOFT_DEADLINE,
                )
                self._stacker_outcome[qid_for_outcome] = "primary"
                return self._apply_platt_calibration(self._maybe_snap_to_integers(stacked, question), question)
            except Exception as primary_exc:
                if not self.stacking_fallback_on_failure:
                    raise

                self._stacker_primary_failed_count += 1
                logger.warning(
                    "STACKER_PRIMARY_FAILED: primary stacker failed on Q %s (%s: %s); trying fallback model",
                    question.id_of_question,
                    type(primary_exc).__name__,
                    primary_exc,
                )

                # Try fallback stacker LLM (lazy import to avoid circular deps)
                from metaculus_bot.llm_configs import STACKER_FALLBACK_LLM

                try:
                    self._stacker_fallback_used_count += 1
                    stacked = await asyncio.wait_for(
                        self._run_stacking(
                            question,
                            research,
                            reasoned_predictions,
                            stacker_llm_override=STACKER_FALLBACK_LLM,
                            aggregated_tool_output=aggregated_tool_output,
                        ),
                        timeout=STACKER_FALLBACK_SOFT_DEADLINE,
                    )
                    logger.info(
                        "STACKER_FALLBACK_SUCCEEDED: fallback stacker succeeded on Q %s",
                        question.id_of_question,
                    )
                    self._stacker_outcome[qid_for_outcome] = "fallback_llm"
                    return self._apply_platt_calibration(self._maybe_snap_to_integers(stacked, question), question)
                except Exception as fallback_exc:
                    self._stacker_fallback_failed_count += 1
                    self._stacking_fallback_count += 1
                    logger.error(
                        "STACKER_FALLBACK_FAILED: fallback stacker also failed on Q %s (%s: %s); "
                        "falling back to MEDIAN aggregation",
                        question.id_of_question,
                        type(fallback_exc).__name__,
                        fallback_exc,
                    )
                    self._stacker_outcome[qid_for_outcome] = "fallback_median"
                    # Direct per-type MEDIAN dispatch. We deliberately do NOT
                    # mutate self.aggregation_strategy here: forecast_questions
                    # runs questions concurrently via asyncio.gather on the same
                    # bot instance, so a mutation during the await would let
                    # another concurrent question observe the wrong strategy
                    # and mis-route its dispatch.
                    first_prediction = predictions[0]
                    if isinstance(first_prediction, (int, float)):
                        float_preds = [float(p) for p in predictions if isinstance(p, (int, float))]
                        return self._apply_platt_calibration(
                            combine_binary_predictions(float_preds, AggregationStrategy.MEDIAN),  # type: ignore[arg-type]  # float is a PredictionTypes member
                            question,
                        )
                    if isinstance(first_prediction, NumericDistribution) and isinstance(question, NumericQuestion):
                        numeric_preds = [p for p in predictions if isinstance(p, NumericDistribution)]
                        median_numeric = await combine_numeric_predictions(
                            numeric_preds, question, AggregationStrategy.MEDIAN
                        )
                        return self._apply_platt_calibration(
                            self._maybe_snap_to_integers(median_numeric, question),  # type: ignore[arg-type]  # NumericDistribution is a PredictionTypes member
                            question,
                        )
                    if isinstance(first_prediction, PredictedOptionList):
                        mc_preds = [p for p in predictions if isinstance(p, PredictedOptionList)]
                        return self._apply_platt_calibration(
                            combine_multiple_choice_predictions(mc_preds, AggregationStrategy.MEDIAN),  # type: ignore[arg-type]  # PredictedOptionList is a PredictionTypes member
                            question,
                        )
                    raise ValueError(
                        f"Unknown prediction type for MEDIAN fallback: {type(first_prediction)}"
                    ) from fallback_exc

        # High-level aggregation log for clarity
        qtype = (
            "binary"
            if isinstance(predictions[0], (int, float))
            else (
                "numeric"
                if isinstance(predictions[0], NumericDistribution)
                else (
                    "multiple-choice"
                    if isinstance(predictions[0], PredictedOptionList)
                    else type(predictions[0]).__name__
                )
            )
        )
        logger.info("Aggregating %s predictions with %s", qtype, self.aggregation_strategy.value)

        # CONDITIONAL_STACKING uses MEDIAN for the low-spread (no-stack) path
        effective_strategy = (
            AggregationStrategy.MEDIAN
            if self.aggregation_strategy == AggregationStrategy.CONDITIONAL_STACKING
            else self.aggregation_strategy
        )

        # Binary aggregation - strategy-based dispatch
        # Same return-value pattern as stacking branch above: each concrete type IS a
        # PredictionTypes member but the checker can't prove it through isinstance on first.
        first_prediction = predictions[0]
        if isinstance(first_prediction, (int, float)):
            float_preds = [float(p) for p in predictions if isinstance(p, (int, float))]
            result = combine_binary_predictions(float_preds, effective_strategy)
            if effective_strategy == AggregationStrategy.MEAN:
                logger.info(
                    "Binary question ensembling: mean of %s = %.3f (rounded)",
                    float_preds,
                    result,
                )
            elif effective_strategy == AggregationStrategy.MEDIAN:
                logger.info(
                    "Binary question ensembling: median of %s = %.3f",
                    float_preds,
                    result,
                )
            else:
                logger.info(
                    "Binary question ensembling: %s of %s = %.3f",
                    effective_strategy.value,
                    float_preds,
                    result,
                )
            return self._apply_platt_calibration(result, question)  # type: ignore[arg-type]  # float is a PredictionTypes member

        if isinstance(first_prediction, NumericDistribution) and isinstance(question, NumericQuestion):
            numeric_preds = [p for p in predictions if isinstance(p, NumericDistribution)]
            aggregated = await combine_numeric_predictions(
                numeric_preds,
                question,
                effective_strategy,
            )
            lb = getattr(question, "lower_bound", None)
            ub = getattr(question, "upper_bound", None)
            logger.info(
                "Numeric aggregation=%s | preserved bounds [%s, %s] | CDF points=%d",
                effective_strategy.value,
                lb,
                ub,
                len(getattr(aggregated, "cdf", [])),
            )
            return self._apply_platt_calibration(
                self._maybe_snap_to_integers(aggregated, question),  # type: ignore[arg-type]  # NumericDistribution is a PredictionTypes member
                question,
            )

        # Multiple choice aggregation - strategy-based dispatch
        if isinstance(first_prediction, PredictedOptionList):
            mc_preds = [p for p in predictions if isinstance(p, PredictedOptionList)]
            aggregated = combine_multiple_choice_predictions(mc_preds, effective_strategy)
            summary = {o.option_name: round(o.probability, 4) for o in aggregated.predicted_options}
            logger.info(
                "MC %s aggregation; renormalized to 1.0 | %s",
                effective_strategy.value,
                summary,
            )
            return self._apply_platt_calibration(aggregated, question)  # type: ignore[arg-type]  # PredictedOptionList is a PredictionTypes member

        # Fallback for unexpected prediction types
        raise ValueError(f"Unknown prediction type for aggregation: {type(predictions[0])}")

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[float]:
        from metaculus_bot.forecaster_runners import run_binary_forecast

        return await run_binary_forecast(question, research, llm_to_use, self.get_llm("parser", "llm"))

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[PredictedOptionList]:
        from metaculus_bot.forecaster_runners import run_mc_forecast

        return await run_mc_forecast(question, research, llm_to_use, self.get_llm("parser", "llm"))

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[NumericDistribution]:
        from metaculus_bot.forecaster_runners import run_numeric_forecast

        prediction, discrete_vote = await run_numeric_forecast(
            question, research, llm_to_use, self.get_llm("parser", "llm")
        )
        qid = question.id_of_question
        if qid is not None and discrete_vote is not None:
            self._discrete_integer_votes[qid].append(discrete_vote)
        return prediction

    def _apply_platt_calibration(self, prediction: PredictionTypes, question: MetaculusQuestion) -> PredictionTypes:
        from metaculus_bot.calibration import BINARY_PLATT_PARAMS, MC_PLATT_PARAMS
        from metaculus_bot.post_processing import apply_platt_calibration

        return apply_platt_calibration(prediction, question, BINARY_PLATT_PARAMS, MC_PLATT_PARAMS)

    def _maybe_snap_to_integers(self, prediction: PredictionTypes, question: MetaculusQuestion) -> PredictionTypes:
        from metaculus_bot.post_processing import maybe_snap_to_integers

        if not isinstance(prediction, NumericDistribution) or not isinstance(question, NumericQuestion):
            return prediction
        qid = question.id_of_question
        if qid is None:
            return prediction
        votes = self._discrete_integer_votes.pop(qid, [])
        return maybe_snap_to_integers(prediction, question, votes)

    def _log_llm_output(self, llm_to_use: GeneralLlm, question_id: int | None, reasoning: str) -> None:
        model_name = llm_to_use.model
        logger.info(
            f"""
\n\n
========================================
LLM OUTPUT | Model: {model_name} | Question: {question_id} | Length: {len(reasoning)} chars
========================================
{reasoning}
========================================
END LLM OUTPUT | {model_name}
========================================
\n\n
"""
        )


if __name__ == "__main__":
    from metaculus_bot.cli import main as cli_main

    cli_main()
