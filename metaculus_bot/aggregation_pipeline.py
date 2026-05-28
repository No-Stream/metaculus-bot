"""Aggregation pipeline: stacking fallback chain and simple combine logic.

Extracted from main.py to keep TemplateForecaster focused on orchestration.
Owns all per-question stacking state (outcomes, meta reasoning, counters).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from forecasting_tools import (
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

from metaculus_bot import stacking
from metaculus_bot.aggregation_strategies import (
    AggregationStrategy,
    combine_binary_predictions,
    combine_multiple_choice_predictions,
    combine_numeric_predictions,
)
from metaculus_bot.constants import STACKER_FALLBACK_SOFT_DEADLINE, STACKER_SOFT_DEADLINE
from metaculus_bot.numeric_diagnostics import log_final_prediction
from metaculus_bot.numeric_pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric_utils import bound_messages
from metaculus_bot.numeric_validation import detect_unit_mismatch

logger = logging.getLogger(__name__)


@dataclass
class AggregationCounters:
    stacking_expected_combine_count: int = 0
    stacking_unexpected_combine_count: int = 0
    stacking_fallback_count: int = 0
    stacker_primary_failed_count: int = 0
    stacker_fallback_used_count: int = 0
    stacker_fallback_failed_count: int = 0


RunStackingFn = Callable[..., Awaitable[PredictionTypes]]


@dataclass
class AggregationPipeline:
    strategy: AggregationStrategy
    stacker_llm: GeneralLlm | None
    parser_llm: GeneralLlm
    stacking_fallback_on_failure: bool = True
    stacking_randomize_order: bool = True
    stacking_spread_thresholds: dict[str, float] = field(default_factory=dict)
    discrete_integer_votes: defaultdict[int, list[bool]] = field(default_factory=lambda: defaultdict(list))
    run_stacking_fn: RunStackingFn | None = None

    # Per-question state
    meta_reasoning: dict[int, str] = field(default_factory=dict)
    outcomes: dict[int, str] = field(default_factory=dict)
    expected_base_combines: set[int] = field(default_factory=set)
    counters: AggregationCounters = field(default_factory=AggregationCounters)

    def get_threshold_for_question(self, question: MetaculusQuestion) -> float:
        if isinstance(question, BinaryQuestion):
            return self.stacking_spread_thresholds["binary"]
        if isinstance(question, MultipleChoiceQuestion):
            return self.stacking_spread_thresholds["mc"]
        if isinstance(question, NumericQuestion):
            return self.stacking_spread_thresholds["numeric"]
        raise ValueError(f"No spread threshold for question type: {type(question).__name__}")

    def register_expected_base_combine(self, question: MetaculusQuestion) -> None:
        qid = question.id_of_question
        assert qid is not None, "register_expected_base_combine requires question.id_of_question"
        self.expected_base_combines.add(qid)

    async def run_stacking(
        self,
        question: MetaculusQuestion,
        research: str,
        reasoned_predictions: list[ReasonedPrediction[PredictionTypes]],
        stacker_llm_override: GeneralLlm | None = None,
        aggregated_tool_output: str | None = None,
    ) -> PredictionTypes:
        """Dispatch stacker LLM per question type."""
        if stacker_llm_override is not None:
            stacker_llm = stacker_llm_override
        else:
            if self.stacker_llm is None:
                raise ValueError("STACKING aggregation strategy requires a stacker LLM to be configured")
            stacker_llm = self.stacker_llm

        page_url = question.page_url or "<unknown>"
        qid = question.id_of_question
        assert qid is not None, "run_stacking requires question.id_of_question"

        base_predictions = [stacking.strip_model_tag(pred.reasoning) for pred in reasoned_predictions]

        if self.stacking_randomize_order:
            combined = list(zip(base_predictions, reasoned_predictions))
            random.shuffle(combined)
            base_predictions = [bp for bp, _ in combined]
            reasoned_predictions = [rp for _, rp in combined]

        if isinstance(question, BinaryQuestion):
            value, meta_text = await stacking.run_stacking_binary(
                stacker_llm,
                self.parser_llm,
                question,
                research,
                base_predictions,
                aggregated_tool_output=aggregated_tool_output,
            )
            self.meta_reasoning[qid] = meta_text
            logger.info(f"Stacked binary prediction for {page_url}: {value}")
            return value
        elif isinstance(question, MultipleChoiceQuestion):
            pol, meta_text = await stacking.run_stacking_mc(
                stacker_llm,
                self.parser_llm,
                question,
                research,
                base_predictions,
                aggregated_tool_output=aggregated_tool_output,
            )
            self.meta_reasoning[qid] = meta_text
            logger.info(f"Stacked multiple choice prediction for {page_url}: {pol}")
            return pol
        elif isinstance(question, NumericQuestion):
            upper_msg, lower_msg = bound_messages(question)
            perc_list, meta_text = await stacking.run_stacking_numeric(
                stacker_llm,
                self.parser_llm,
                question,
                research,
                base_predictions,
                lower_msg,
                upper_msg,
                aggregated_tool_output=aggregated_tool_output,
            )
            self.meta_reasoning[qid] = meta_text

            percentile_list, zero_point = sanitize_percentiles(list(perc_list), question)

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

    async def aggregate(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
        research: str | None = None,
        reasoned_predictions: list[ReasonedPrediction[PredictionTypes]] | None = None,
        aggregated_tool_output: str | None = None,
    ) -> PredictionTypes:
        """Full aggregation: stacking fallback chain OR simple combine."""
        if not predictions:
            raise ValueError("Cannot aggregate empty list of predictions")

        # Base-combine re-entry: parent class calls aggregate after stacking already happened
        if (
            self.strategy in (AggregationStrategy.STACKING, AggregationStrategy.CONDITIONAL_STACKING)
            and reasoned_predictions is None
            and research is None
        ):
            return await self._base_combine(predictions, question)

        # Stacking path
        if self.strategy in (AggregationStrategy.STACKING, AggregationStrategy.CONDITIONAL_STACKING):
            return await self._stacking_aggregate(
                predictions, question, research, reasoned_predictions, aggregated_tool_output
            )

        # Simple MEAN/MEDIAN path
        return await self._simple_aggregate(predictions, question)

    async def _base_combine(  # noqa: ASYNC910 - awaits only on numeric path
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        qkey = question.id_of_question

        expected = qkey in self.expected_base_combines
        if expected:
            self.expected_base_combines.discard(qkey)
            self.counters.stacking_expected_combine_count += 1
        else:
            self.counters.stacking_unexpected_combine_count += 1

        if len(predictions) == 1:
            if expected:
                logger.info("STACKING base combine: single pre-stacked output; returning as-is")
            else:
                logger.warning("Unexpected STACKING combine: single input without stacking context; returning as-is")
            return predictions[0]

        # CONDITIONAL_STACKING uses MEDIAN; regular STACKING uses MEAN
        base_combine_strategy = (
            AggregationStrategy.MEDIAN
            if self.strategy == AggregationStrategy.CONDITIONAL_STACKING
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

        apply_platt_after_combine = self.strategy == AggregationStrategy.CONDITIONAL_STACKING

        first = predictions[0]
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

    def _get_stacking_fn(self) -> Callable[..., Awaitable[PredictionTypes]]:
        if self.run_stacking_fn is not None:
            return self.run_stacking_fn
        return self.run_stacking

    async def _stacking_aggregate(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
        research: str | None,
        reasoned_predictions: list[ReasonedPrediction[PredictionTypes]] | None,
        aggregated_tool_output: str | None,
    ) -> PredictionTypes:
        if self.stacker_llm is None:
            raise ValueError("STACKING aggregation strategy requires a stacker LLM to be configured")
        if reasoned_predictions is None:
            raise ValueError("STACKING aggregation strategy requires reasoned predictions")
        if research is None:
            raise ValueError("STACKING aggregation strategy requires research context")

        qid_for_outcome = question.id_of_question
        assert qid_for_outcome is not None

        stacking_fn = self._get_stacking_fn()

        try:
            stacked = await asyncio.wait_for(
                stacking_fn(
                    question,
                    research,
                    reasoned_predictions,
                    aggregated_tool_output=aggregated_tool_output,
                ),
                timeout=STACKER_SOFT_DEADLINE,
            )
            self.outcomes[qid_for_outcome] = "primary"
            return self._apply_platt_calibration(self._maybe_snap_to_integers(stacked, question), question)
        except Exception as primary_exc:
            if not self.stacking_fallback_on_failure:
                raise

            self.counters.stacker_primary_failed_count += 1
            logger.warning(
                "STACKER_PRIMARY_FAILED: primary stacker failed on Q %s (%s: %s); trying fallback model",
                question.id_of_question,
                type(primary_exc).__name__,
                primary_exc,
            )

            from metaculus_bot.llm_configs import STACKER_FALLBACK_LLM

            try:
                self.counters.stacker_fallback_used_count += 1
                stacked = await asyncio.wait_for(
                    stacking_fn(
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
                self.outcomes[qid_for_outcome] = "fallback_llm"
                return self._apply_platt_calibration(self._maybe_snap_to_integers(stacked, question), question)
            except Exception as fallback_exc:
                self.counters.stacker_fallback_failed_count += 1
                self.counters.stacking_fallback_count += 1
                logger.error(
                    "STACKER_FALLBACK_FAILED: fallback stacker also failed on Q %s (%s: %s); "
                    "falling back to MEDIAN aggregation",
                    question.id_of_question,
                    type(fallback_exc).__name__,
                    fallback_exc,
                )
                self.outcomes[qid_for_outcome] = "fallback_median"
                return await self._median_fallback(predictions, question)

    async def _median_fallback(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        first_prediction = predictions[0]
        if isinstance(first_prediction, (int, float)):
            float_preds = [float(p) for p in predictions if isinstance(p, (int, float))]
            return self._apply_platt_calibration(
                combine_binary_predictions(float_preds, AggregationStrategy.MEDIAN),  # type: ignore[arg-type]
                question,
            )
        if isinstance(first_prediction, NumericDistribution) and isinstance(question, NumericQuestion):
            numeric_preds = [p for p in predictions if isinstance(p, NumericDistribution)]
            median_numeric = await combine_numeric_predictions(numeric_preds, question, AggregationStrategy.MEDIAN)
            return self._apply_platt_calibration(
                self._maybe_snap_to_integers(median_numeric, question),  # type: ignore[arg-type]
                question,
            )
        if isinstance(first_prediction, PredictedOptionList):
            mc_preds = [p for p in predictions if isinstance(p, PredictedOptionList)]
            return self._apply_platt_calibration(
                combine_multiple_choice_predictions(mc_preds, AggregationStrategy.MEDIAN),  # type: ignore[arg-type]
                question,
            )
        raise ValueError(f"Unknown prediction type for MEDIAN fallback: {type(first_prediction)}")

    async def _simple_aggregate(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
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
        logger.info("Aggregating %s predictions with %s", qtype, self.strategy.value)

        effective_strategy = (
            AggregationStrategy.MEDIAN if self.strategy == AggregationStrategy.CONDITIONAL_STACKING else self.strategy
        )

        first_prediction = predictions[0]
        if isinstance(first_prediction, (int, float)):
            float_preds = [float(p) for p in predictions if isinstance(p, (int, float))]
            result = combine_binary_predictions(float_preds, effective_strategy)
            if effective_strategy == AggregationStrategy.MEAN:
                logger.info("Binary question ensembling: mean of %s = %.3f (rounded)", float_preds, result)
            elif effective_strategy == AggregationStrategy.MEDIAN:
                logger.info("Binary question ensembling: median of %s = %.3f", float_preds, result)
            else:
                logger.info(
                    "Binary question ensembling: %s of %s = %.3f", effective_strategy.value, float_preds, result
                )
            return self._apply_platt_calibration(result, question)  # type: ignore[arg-type]

        if isinstance(first_prediction, NumericDistribution) and isinstance(question, NumericQuestion):
            numeric_preds = [p for p in predictions if isinstance(p, NumericDistribution)]
            aggregated = await combine_numeric_predictions(numeric_preds, question, effective_strategy)
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
                self._maybe_snap_to_integers(aggregated, question),  # type: ignore[arg-type]
                question,
            )

        if isinstance(first_prediction, PredictedOptionList):
            mc_preds = [p for p in predictions if isinstance(p, PredictedOptionList)]
            aggregated = combine_multiple_choice_predictions(mc_preds, effective_strategy)
            summary = {o.option_name: round(o.probability, 4) for o in aggregated.predicted_options}
            logger.info("MC %s aggregation; renormalized to 1.0 | %s", effective_strategy.value, summary)
            return self._apply_platt_calibration(aggregated, question)  # type: ignore[arg-type]

        raise ValueError(f"Unknown prediction type for aggregation: {type(predictions[0])}")

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
        votes = self.discrete_integer_votes.pop(qid, [])
        return maybe_snap_to_integers(prediction, question, votes)
