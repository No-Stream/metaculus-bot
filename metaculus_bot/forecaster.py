import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Callable, Coroutine, Sequence, cast

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
from metaculus_bot.aggregation_pipeline import AggregationPipeline
from metaculus_bot.aggregation_strategies import (
    AggregationStrategy,
)
from metaculus_bot.comment.trimming import trim_section
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
    WALL_CLOCK_STACKING_MIN_BUDGET,
    env_flag_enabled,
)
from metaculus_bot.llm_setup import prepare_llm_config
from metaculus_bot.numeric.pchip_processing import log_pchip_summary, reset_pchip_stats
from metaculus_bot.performance_analysis.parsing import (
    annotate_forecaster_bullets_with_models,
    extract_model_display_name_from_reasoning,
)
from metaculus_bot.research.orchestrator import ResearchOrchestrator
from metaculus_bot.research.providers import (
    ResearchCallable,
)
from metaculus_bot.research.targeted import extract_disagreement_crux, run_targeted_search
from metaculus_bot.spread_metrics import compute_spread

# Probabilistic-tools wiring (Workstream C activation). The feature flag is
# re-exported under a descriptive local alias so call sites read clearly.
from metaculus_bot.utils.logging_utils import CompactLoggingForecastBot

logger = logging.getLogger(__name__)

load_environment()


class TemplateForecaster(CompactLoggingForecastBot):
    def __init__(
        self,
        *,
        research_reports_per_question: int = 1,
        predictions_per_research_report: int = 1,
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
        research_sink: Any | None = None,
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
        self.__stacker_llm: GeneralLlm | None = setup.stacker_llm
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

        # Aggregation pipeline owns stacking state, counters, and dispatch
        self._pipeline = AggregationPipeline(
            strategy=self.aggregation_strategy,
            stacker_llm=self._stacker_llm,
            parser_llm=GeneralLlm(model="placeholder"),  # replaced after super().__init__
            stacking_fallback_on_failure=stacking_fallback_on_failure,
            stacking_randomize_order=stacking_randomize_order,
            stacking_spread_thresholds=self._stacking_spread_thresholds,
            discrete_integer_votes=self._discrete_integer_votes,
        )

        # --- Alerting counters (consumed by cli.py to decide sys.exit status) ---
        self._forecasters_dropped_count: int = 0
        self._questions_failed_to_publish: int = 0

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
            publish_reports_to_metaculus=publish_reports_to_metaculus,
            folder_to_save_reports_to=folder_to_save_reports_to,
            skip_previously_forecasted_questions=skip_previously_forecasted_questions,
            llms=normalized_llms,  # type: ignore[arg-type]  # dict value type lacks None but parent expects Optional
        )

        # Now that super().__init__ has run, resolve the parser LLM and wire the
        # stacking function so that mocking bot._run_stacking flows through.
        # Use a lambda with dynamic attribute lookup so mock.patch replaces propagate.
        self._pipeline.parser_llm = self.get_llm("parser", "llm")
        self._pipeline.run_stacking_fn = lambda *args, **kwargs: self._run_stacking(*args, **kwargs)

        self._research = ResearchOrchestrator(
            default_llm=self.get_llm("default", "llm"),
            summarizer_llm=self.get_llm("summarizer", "llm"),
            custom_provider=research_provider,
            research_cache=research_cache,
            is_benchmarking=is_benchmarking,
            allow_research_fallback=allow_research_fallback,
            max_concurrent_research=max_concurrent_research,
            research_sink=research_sink,
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
        return self._pipeline.get_threshold_for_question(question)

    def _register_expected_base_combine(self, question: MetaculusQuestion) -> None:
        self._pipeline.register_expected_base_combine(question)

    @property
    def _stacker_llm(self) -> GeneralLlm | None:
        return self.__stacker_llm

    @_stacker_llm.setter
    def _stacker_llm(self, value: GeneralLlm | None) -> None:
        self.__stacker_llm = value
        if hasattr(self, "_pipeline"):
            self._pipeline.stacker_llm = value

    @property
    def _stack_meta_reasoning(self) -> dict[int, str]:
        return self._pipeline.meta_reasoning

    @property
    def _stacker_outcome(self) -> dict[int, str]:
        return self._pipeline.outcomes

    @property
    def _stack_expected_base_combine(self) -> set[int]:
        return self._pipeline.expected_base_combines

    @property
    def _stacking_expected_combine_count(self) -> int:
        return self._pipeline.counters.stacking_expected_combine_count

    @_stacking_expected_combine_count.setter
    def _stacking_expected_combine_count(self, value: int) -> None:
        self._pipeline.counters.stacking_expected_combine_count = value

    @property
    def _stacking_unexpected_combine_count(self) -> int:
        return self._pipeline.counters.stacking_unexpected_combine_count

    @_stacking_unexpected_combine_count.setter
    def _stacking_unexpected_combine_count(self, value: int) -> None:
        self._pipeline.counters.stacking_unexpected_combine_count = value

    @property
    def _stacking_fallback_count(self) -> int:
        return self._pipeline.counters.stacking_fallback_count

    @_stacking_fallback_count.setter
    def _stacking_fallback_count(self, value: int) -> None:
        self._pipeline.counters.stacking_fallback_count = value

    @property
    def _stacker_primary_failed_count(self) -> int:
        return self._pipeline.counters.stacker_primary_failed_count

    @_stacker_primary_failed_count.setter
    def _stacker_primary_failed_count(self, value: int) -> None:
        self._pipeline.counters.stacker_primary_failed_count = value

    @property
    def _stacker_fallback_used_count(self) -> int:
        return self._pipeline.counters.stacker_fallback_used_count

    @_stacker_fallback_used_count.setter
    def _stacker_fallback_used_count(self, value: int) -> None:
        self._pipeline.counters.stacker_fallback_used_count = value

    @property
    def _stacker_fallback_failed_count(self) -> int:
        return self._pipeline.counters.stacker_fallback_failed_count

    @_stacker_fallback_failed_count.setter
    def _stacker_fallback_failed_count(self, value: int) -> None:
        self._pipeline.counters.stacker_fallback_failed_count = value

    async def forecast_questions(  # pyright: ignore[reportIncompatibleMethodOverride]  # matches base's broadest @overload; base declares Literal overloads we deliberately don't replicate
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
        return await self._pipeline.run_stacking(
            question,
            research,
            reasoned_predictions,
            stacker_llm_override=stacker_llm_override,
            aggregated_tool_output=aggregated_tool_output,
        )

    async def run_research(self, question: MetaculusQuestion) -> str:
        return await self._research.run_research(question)

    def _select_research_providers(self) -> list[tuple[ResearchCallable, str]]:
        return self._research._select_research_providers()

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
        exception_group: ExceptionGroup | None = (
            ExceptionGroup(f"Errors: {errors}", cast(list[Exception], exceptions)) if exceptions else None
        )
        return valid_predictions, errors, exception_group

    async def _finalize_stacked_prediction(
        self,
        question: MetaculusQuestion,
        valid_predictions: list[ReasonedPrediction[PredictionTypes]],
        research_for_stacking: str,
        research_report: str,
        summary_report: str,
        errors: list[str],
        default_meta_reasoning: str,
    ) -> ResearchWithPredictions[PredictionTypes]:
        """Run the stacker and package the single aggregated prediction.

        Shared by the STACKING and the CONDITIONAL_STACKING-triggered branches:
        both compute the deterministic cross-model math, invoke
        ``_aggregate_predictions`` (which runs the stacker LLM), then preserve
        the stacker meta-analysis alongside the base-model reasonings so
        residual analysis can recover per-model attribution even when stacking
        overrode the base aggregation.

        The two branches differ only in which research text feeds the stacker
        (``research_for_stacking``), which text is surfaced in the published
        comment (``research_report``), and the meta-reasoning fallback string.
        ``_stacker_outcome`` is populated by ``_aggregate_predictions`` on the
        path that actually produced ``aggregated_value``; it is not set here.
        """
        prediction_values = [pred.prediction_value for pred in valid_predictions]
        # Probabilistic tools: deterministic cross-model math runs once per
        # question and rides at the top of the stacker prompt. No-ops when
        # PROBABILISTIC_TOOLS_ENABLED is unset.
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
            research=research_for_stacking,
            reasoned_predictions=valid_predictions,
            aggregated_tool_output=aggregated_tool_output,
        )
        assert question.id_of_question is not None
        meta_text = self._stack_meta_reasoning.pop(question.id_of_question, default_meta_reasoning)
        combined_reasoning = stacking.combine_stacker_and_base_reasoning(meta_text, valid_predictions)
        aggregated_prediction = ReasonedPrediction(prediction_value=aggregated_value, reasoning=combined_reasoning)
        self._register_expected_base_combine(question)
        return ResearchWithPredictions(
            research_report=research_report,
            summary_report=summary_report,
            errors=errors,
            predictions=[aggregated_prediction],
        )

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

        # A stub, not the full corpus: the framework embeds summary_report under
        # "### Research Summary" and research_report under "# RESEARCH", so
        # setting both to `research` duplicated it and bloated the comment past
        # the char limit. The "### Research Summary" heading is emitted
        # regardless of body, so the trim anchor and parser markers survive.
        summary_report = "_Full research in the RESEARCH section below._"
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
            # MEDIAN under CONDITIONAL_STACKING (see base_combine_strategy in
            # aggregation_pipeline.py:226-231). The marker must match the
            # actual aggregation method so residual analysis cuts bucket the
            # two paths correctly.
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
            return await self._finalize_stacked_prediction(
                question,
                valid_predictions,
                research_for_stacking=research_to_use,
                research_report=research,
                summary_report=summary_report,
                errors=errors,
                default_meta_reasoning="Stacked prediction aggregated from multiple models",
            )
        elif self.aggregation_strategy == AggregationStrategy.CONDITIONAL_STACKING and not skip_stacking_for_budget:
            prediction_values = [pred.prediction_value for pred in valid_predictions]
            spread = compute_spread(question, prediction_values)
            threshold = self._get_threshold_for_question(question)

            # Per-question-type stacking gates. All three default to DISABLED. Set
            # <TYPE>_STACKING_ENABLED=true in deploy env to opt a type back into
            # stacking; otherwise the stacker is bypassed (forces median/skipped path).
            spread_exceeds_threshold = spread > threshold
            type_stacking_disabled = False
            type_to_stacking_env = {
                BinaryQuestion: BINARY_STACKING_ENABLED_ENV,
                MultipleChoiceQuestion: MC_STACKING_ENABLED_ENV,
                NumericQuestion: NUMERIC_STACKING_ENABLED_ENV,
            }
            for q_type, env_name in type_to_stacking_env.items():
                if isinstance(question, q_type):
                    if spread_exceeds_threshold and not env_flag_enabled(env_name, default=False):
                        # Disagreement was high enough to trigger stacking, but the
                        # per-type gate is off, so we deliberately bypass it.
                        spread_exceeds_threshold = False
                        type_stacking_disabled = True
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

                # 4. Run stacking.
                #
                # research_report must be combined_research so the
                # ## Targeted Research (addressing model disagreement) header
                # reaches the published comment.
                return await self._finalize_stacked_prediction(
                    question,
                    valid_predictions,
                    research_for_stacking=combined_research,
                    research_report=combined_research,
                    summary_report=summary_report,
                    errors=errors,
                    default_meta_reasoning=(
                        "Conditional stacking: aggregated from multiple models after high-disagreement detected"
                    ),
                )
            else:
                self._conditional_stacking_skipped_count += 1
                if type_stacking_disabled:
                    logger.info(
                        "Conditional stacking SKIPPED: stacking disabled for this question type "
                        "(spread=%.3f, threshold=%.3f) for question %s",
                        spread,
                        threshold,
                        question.id_of_question,
                    )
                else:
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
        from metaculus_bot.comment.formatting import build_unified_explanation

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

        forecast_function: Callable[..., Coroutine[Any, Any, ReasonedPrediction[Any]]]
        if isinstance(question, BinaryQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_binary(q, r, llm)  # noqa: E731
        elif isinstance(question, MultipleChoiceQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_multiple_choice(q, r, llm)  # noqa: E731
        elif isinstance(question, NumericQuestion):
            forecast_function = lambda q, r, llm: self._run_forecast_on_numeric(q, r, llm)  # noqa: E731
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
        return await self._pipeline.aggregate(
            predictions, question, research, reasoned_predictions, aggregated_tool_output
        )

    async def _run_forecast_on_binary(  # pyright: ignore[reportIncompatibleMethodOverride]  # extra llm_to_use param: ensemble fan-out passes a specific LLM per call
        self, question: BinaryQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[float]:
        from metaculus_bot.forecaster_runners import run_binary_forecast

        return await run_binary_forecast(question, research, llm_to_use, self.get_llm("parser", "llm"))

    async def _run_forecast_on_multiple_choice(  # pyright: ignore[reportIncompatibleMethodOverride]  # extra llm_to_use param: ensemble fan-out passes a specific LLM per call
        self, question: MultipleChoiceQuestion, research: str, llm_to_use: GeneralLlm
    ) -> ReasonedPrediction[PredictedOptionList]:
        from metaculus_bot.forecaster_runners import run_mc_forecast

        return await run_mc_forecast(question, research, llm_to_use, self.get_llm("parser", "llm"))

    async def _run_forecast_on_numeric(  # pyright: ignore[reportIncompatibleMethodOverride]  # extra llm_to_use param: ensemble fan-out passes a specific LLM per call
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
