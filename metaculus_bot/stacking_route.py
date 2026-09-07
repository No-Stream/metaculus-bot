"""Post-fan-out routing: given the surviving forecasts, decide how to aggregate.

Everything here happens AFTER the forecaster fan-out has returned and the
min-forecasters guard has passed, and before the framework's per-question
aggregator runs. The decision tree, in order:

1. **Single survivor** — skip spread and stacking entirely (the spread helpers
   require >=2 predictions and raise otherwise).
2. **Wall-clock budget** — too little budget left to afford the stacker LLM, so
   force the base-combine fallback.
3. **STACKING** — always stack.
4. **CONDITIONAL_STACKING** — stack only when the base models disagree past the
   per-question-type threshold AND that type's stacking gate is on; a triggered
   stack first extracts the disagreement crux and runs targeted research.
5. Anything else (a non-stacking strategy) — hand the raw predictions back.

The aggregation pipeline owns the strategy, stacker configuration, per-question
maps, and counters that routing reads and updates.
"""

import asyncio
import logging
import math

from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    ReasonedPrediction,
)
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.forecast_report import ResearchWithPredictions

from metaculus_bot import stacking
from metaculus_bot.aggregation_pipeline import AggregationPipeline
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import (
    BINARY_STACKING_ENABLED_ENV,
    CRUX_SOFT_DEADLINE,
    MC_STACKING_ENABLED_ENV,
    NATIVE_SEARCH_WALL_TIMEOUT,
    NUMERIC_STACKING_ENABLED_ENV,
    STACKER_FALLBACK_SOFT_DEADLINE,
    STACKER_SOFT_DEADLINE,
    WALL_CLOCK_STACKING_MIN_BUDGET,
    env_flag_enabled,
)
from metaculus_bot.research.targeted import extract_disagreement_crux, run_targeted_search
from metaculus_bot.spread_metrics import compute_spread
from metaculus_bot.time_budget import QuestionTimeBudget
from metaculus_bot.tool_runner import build_cross_model_aggregation

logger = logging.getLogger(__name__)

_STACKING_STRATEGIES = (AggregationStrategy.STACKING, AggregationStrategy.CONDITIONAL_STACKING)

# Per-question-type stacking gates. All three default to DISABLED. Set
# <TYPE>_STACKING_ENABLED=true in deploy env to opt a type back into stacking;
# otherwise the stacker is bypassed (forces the median/skipped path). Binary and
# MC are matched before NumericQuestion because DiscreteQuestion subclasses the
# latter and must read the numeric gate.
_STACKING_ENV_BY_QUESTION_TYPE: dict[type[MetaculusQuestion], str] = {
    BinaryQuestion: BINARY_STACKING_ENABLED_ENV,
    MultipleChoiceQuestion: MC_STACKING_ENABLED_ENV,
    NumericQuestion: NUMERIC_STACKING_ENABLED_ENV,
}


def _with_diagnostics(text: str, diagnostics_block: str | None) -> str:
    """Re-append the provider-diagnostics block to comment-bound research text.

    ``run_research`` returns forecaster-clean text (diagnostics withheld so they
    never reach forecaster prompts, the stacker, or the gap-fill v2 driver brief),
    but they must still reach the published comment.
    """
    return f"{text}\n\n{diagnostics_block}" if diagnostics_block else text


def _type_gate_enabled(question: MetaculusQuestion) -> bool:
    """Whether this question type's ``<TYPE>_STACKING_ENABLED`` flag is set.

    A question matching none of the three types has no gate to fail, so it stays
    eligible for stacking.
    """
    for question_type, env_name in _STACKING_ENV_BY_QUESTION_TYPE.items():
        if isinstance(question, question_type):
            return env_flag_enabled(env_name, default=False)
    return True


def _enrichment_timeout(soft_deadline_s: float, time_budget: QuestionTimeBudget) -> float:
    """Clamp an enrichment stage's soft deadline to what the budget still affords.

    Leaves ``WALL_CLOCK_STACKING_MIN_BUDGET`` behind for the publish, so even a
    stage that runs to this clamp cannot consume the POST's reserve. Floored at a
    positive value rather than 0 — ``asyncio.wait_for(coro, 0)`` cancels before the
    coroutine gets a first step, which would report a stage failure that never
    happened.

    Where this actually binds: on a CLOSE-LIMITED budget it is a no-op by
    construction, because ``_skip_stacking_for_budget``'s raised floor already
    sums both enrichment bounds (via ``_stacking_budget_required_s``) and refuses
    the path without the whole worst case in hand. It bites on the STATIC budget,
    where the gate's floor stays at 90 s: there a crux extraction entered with
    e.g. 100 s remaining is clamped to 10 s instead of running its full soft
    deadline into the publish reserve — a deliberate tightening of the
    pre-budget behavior on that path.
    """
    affordable = time_budget.remaining_s() - WALL_CLOCK_STACKING_MIN_BUDGET
    return max(1.0, min(float(soft_deadline_s), affordable))


def _stacking_budget_required_s(pipeline: AggregationPipeline) -> float:
    """Budget the stacking path can consume in the worst case, in seconds.

    The ladder's own soft deadlines, summed: primary stacker, then the
    different-vendor fallback. Under CONDITIONAL_STACKING a triggered stack first
    extracts the disagreement crux and runs a targeted search, so those two bounds
    join the sum — computed conservatively (before the spread is known), because the
    gate has to decide whether the path is affordable before it starts walking it.
    """
    required = STACKER_SOFT_DEADLINE + STACKER_FALLBACK_SOFT_DEADLINE
    if pipeline.strategy == AggregationStrategy.CONDITIONAL_STACKING:
        required += CRUX_SOFT_DEADLINE + NATIVE_SEARCH_WALL_TIMEOUT
    return float(required)


def _skip_stacking_for_budget(
    pipeline: AggregationPipeline,
    question: MetaculusQuestion,
    qid: int,
    time_budget: QuestionTimeBudget,
) -> bool:
    """Force the base-combine fallback when the per-Q wall clock is nearly spent.

    If we've burned through the per-Q budget (research stalled, or the fan-out used
    most of it), skip the stacker LLM entirely. Typical publish is ~1s; the
    ``WALL_CLOCK_STACKING_MIN_BUDGET`` floor leaves headroom for sustained slowness
    on a single POST. The full worst case (both POSTs stalling for
    ``PUBLISH_POST_TIMEOUT * (PUBLISH_POST_RETRIES + 1)``) requires multi-POST
    stalling, which this skip already recovers.

    On a CLOSE-LIMITED budget the floor rises to cover the stacking path's own worst
    case (``_stacking_budget_required_s``), because the two overruns cost different
    things. Against the static deadline an overrun costs nothing — the deadline is
    sized against the cron period and the question's close is hours away — so the
    90 s floor is exactly right and stays unchanged. Against a close-limited budget
    an overrun forfeits the question, and the 90 s floor would happily start a
    stacker that can legitimately run 800 s.
    """
    if pipeline.strategy not in _STACKING_STRATEGIES:
        return False
    remaining = time_budget.remaining_s()
    floor = WALL_CLOCK_STACKING_MIN_BUDGET
    if time_budget.close_limited:
        floor += _stacking_budget_required_s(pipeline)
    if remaining >= floor:
        return False

    # F15: the base-combine re-entry uses MEAN under STACKING and MEDIAN under
    # CONDITIONAL_STACKING (see ``base_combine_strategy`` in
    # aggregation_pipeline.py). The marker must match the actual aggregation
    # method so residual analysis cuts bucket the two paths correctly.
    budget_skip_outcome = (
        "fallback_median" if pipeline.strategy == AggregationStrategy.CONDITIONAL_STACKING else "fallback_mean"
    )
    logger.warning(
        # The floor ACTUALLY applied, not the static constant: on a close-limited
        # budget it includes the stacking path's own worst case, and this WARN is
        # the only record of why the stacker was skipped.
        "WALLCLOCK_ABORT: skipping stacking for Q %s; remaining=%.1fs < %.0fs; forcing %s fallback",
        qid,
        remaining,
        floor,
        budget_skip_outcome,
    )
    pipeline.outcomes[qid] = budget_skip_outcome
    # Same skip-reason + counter treatment as the other skip paths
    # (single_forecaster / config_off / spread_below_threshold): without them a
    # residual cut keyed on STACKER_SKIP_REASON silently misses this bucket. The
    # conditional-stacking tally only exists under CONDITIONAL_STACKING.
    pipeline.skip_reasons[qid] = "wall_clock_budget"
    if pipeline.strategy == AggregationStrategy.CONDITIONAL_STACKING:
        pipeline.counters.conditional_stacking_skipped_count += 1
    # Register so the pipeline's aggregate step (which will run with
    # reasoned_predictions=None) takes the expected base-combine path and doesn't
    # log "Unexpected STACKING combine".
    pipeline.register_expected_base_combine(question)
    return True


async def _targeted_research_for_crux(
    pipeline: AggregationPipeline,
    question: MetaculusQuestion,
    *,
    analyzer_llm: GeneralLlm,
    is_benchmarking: bool,
    valid_predictions: list[ReasonedPrediction[PredictionTypes]],
    time_budget: QuestionTimeBudget,
) -> str:
    """Extract what the base models actually disagree about, then research it.

    Returns the targeted-research text, or ``""`` if either stage soft-failed (both
    failures are counted so the end-of-run summary reports them; a triggered stack
    proceeds on base research alone). The broad excepts are deliberate: targeted
    research is an ENRICHMENT of the stacker prompt, so any failure to obtain it
    must degrade to stacking on base research rather than lose the question. Both
    are counted and logged with a traceback, and the caller's stack still runs.

    ``analyzer_llm`` is passed in already narrowed — the caller raises when it is
    unconfigured, before any research spend.

    Both stages are additionally clamped to what the question's remaining budget
    can afford, so an enrichment stage can never spend the time the stacker (and
    behind it, the prediction POST) still needs. On a close-limited budget the
    clamp is a no-op by construction — ``_skip_stacking_for_budget``'s raised
    floor already refused this path without the whole worst case in hand — so
    what the clamp really guards is the STATIC budget's thin tail, where the
    gate's 90 s floor admits a stage whose full soft deadline would overrun the
    publish reserve (see ``_enrichment_timeout``).
    """
    # Crux extraction under a soft deadline: without the wait_for the only bound is
    # the analyzer LLM's own litellm timeout (UTILITY_MODEL_CONFIG in
    # llm_configs.py), which is looser than CRUX_SOFT_DEADLINE on this critical path.
    base_texts = [stacking.strip_model_tag(pred.reasoning) for pred in valid_predictions]
    crux_timeout = _enrichment_timeout(CRUX_SOFT_DEADLINE, time_budget)
    try:
        crux = await asyncio.wait_for(
            extract_disagreement_crux(analyzer_llm, question.question_text, base_texts),
            timeout=crux_timeout,
        )
    except TimeoutError:
        pipeline.counters.conditional_stacking_crux_failures += 1
        logger.warning(
            "CRUX_SOFT_DEADLINE: crux extraction exceeded %.0fs for Q %s; skipping targeted research",
            crux_timeout,
            question.id_of_question,
        )
        return ""
    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except  # enrichment-only; degrade to base research
        pipeline.counters.conditional_stacking_crux_failures += 1
        logger.exception("Disagreement crux extraction failed, skipping targeted research")
        return ""

    if not crux:
        return ""
    try:
        return await asyncio.wait_for(
            run_targeted_search(crux, question.question_text, is_benchmarking=is_benchmarking),
            timeout=_enrichment_timeout(NATIVE_SEARCH_WALL_TIMEOUT, time_budget),
        )
    except Exception:  # HARNESS-SCAN-EXEMPT-broad-except  # enrichment-only; degrade to base research
        pipeline.counters.conditional_stacking_search_failures += 1
        logger.exception("Targeted search failed, proceeding with base research only")
        return ""


def _conditional_stacking_verdict(
    pipeline: AggregationPipeline,
    question: MetaculusQuestion,
    valid_predictions: list[ReasonedPrediction[PredictionTypes]],
) -> tuple[float, float, str | None]:
    """Measure disagreement and decide whether the stacker fires.

    Returns ``(spread, threshold, skip_reason)`` where ``skip_reason`` is None iff the
    stacker should run. Deriving the reason HERE, once, is what keeps the stacker
    outcome / ``STACKER_SKIP_REASON`` pair from drifting when a fourth cause
    lands: both markers read this one value.
    """
    spread = compute_spread(question, [pred.prediction_value for pred in valid_predictions])
    threshold = pipeline.get_threshold_for_question(question)

    # An UNMEASURABLE spread reports inf (spread_metrics' SPREAD_UNDEFINED: a
    # non-positive normalizing denominator), and inf > threshold would fire the stacker —
    # crux extraction, a targeted search and a stacker call — off no measurement at all.
    # It used to report 0.0 and skip while the marker claimed the models AGREED, which is
    # the affirmative reading this whole finding is about. Skipping is the conservative
    # route (MEDIAN, no spend), and the reason names what actually happened so residual
    # analysis can find these questions.
    if math.isinf(spread):
        return spread, threshold, "spread_undefined"
    if spread <= threshold:
        return spread, threshold, "spread_below_threshold"
    # Disagreement was high enough to trigger stacking, but the per-type gate is
    # off, so we deliberately bypass it.
    if not _type_gate_enabled(question):
        return spread, threshold, "config_off"
    return spread, threshold, None


def _record_conditional_skip(
    pipeline: AggregationPipeline,
    question: MetaculusQuestion,
    qid: int,
    *,
    skip_reason: str,
    spread: float,
    threshold: float,
) -> None:
    """Log the skip, bump the counter, and stamp both stacker markers.

    "skipped_config_off" (spread exceeded the threshold but the per-type gate was off) vs
    plain "skipped" keeps the suppression reason durable in the published marker instead of
    requiring git archaeology over workflow-yaml flag history. The skip_reason companion
    restates it in the one field shared with the single-forecaster path, so a consumer can
    read STACKER_SKIP_REASON alone.
    """
    pipeline.counters.conditional_stacking_skipped_count += 1
    if skip_reason == "spread_undefined":
        logger.warning(
            "Conditional stacking SKIPPED: spread was UNMEASURABLE for question %s "
            "(see the SPREAD_UNDEFINED warning above); routing to MEDIAN without a "
            "disagreement measurement",
            qid,
        )
    elif skip_reason == "config_off":
        logger.info(
            "Conditional stacking SKIPPED: stacking disabled for this question type "
            "(spread=%.3f, threshold=%.3f) for question %s",
            spread,
            threshold,
            qid,
        )
    else:
        logger.info(
            "Conditional stacking SKIPPED: spread=%.3f <= threshold=%.3f for question %s",
            spread,
            threshold,
            qid,
        )
    pipeline.register_expected_base_combine(question)
    pipeline.outcomes[qid] = "skipped_config_off" if skip_reason == "config_off" else "skipped"
    pipeline.skip_reasons[qid] = skip_reason


async def _finalize_stacked_prediction(
    pipeline: AggregationPipeline,
    question: MetaculusQuestion,
    valid_predictions: list[ReasonedPrediction[PredictionTypes]],
    *,
    research_for_stacking: str,
    research_report: str,
    summary_report: str,
    errors: list[str],
    default_meta_reasoning: str,
) -> ResearchWithPredictions[PredictionTypes]:
    """Run the stacker and package its value with the base-model reasoning."""
    prediction_values = [pred.prediction_value for pred in valid_predictions]
    aggregated_tool_output = (
        build_cross_model_aggregation(
            question=question,
            rationales=[prediction.reasoning for prediction in valid_predictions],
            prediction_values=prediction_values,
        )
        or None
    )
    aggregated_value = await pipeline.stack_predictions(
        prediction_values,
        question,
        research=research_for_stacking,
        reasoned_predictions=valid_predictions,
        aggregated_tool_output=aggregated_tool_output,
    )
    qid = question.id_of_question
    if qid is None:
        raise ValueError("Question must have id_of_question to finalize stacked prediction")
    meta_text = pipeline.meta_reasoning.pop(qid, default_meta_reasoning)
    combined_reasoning = stacking.combine_stacker_and_base_reasoning(meta_text, valid_predictions)
    aggregated_prediction = ReasonedPrediction(prediction_value=aggregated_value, reasoning=combined_reasoning)
    pipeline.register_expected_base_combine(question)
    return ResearchWithPredictions(
        research_report=research_report,
        summary_report=summary_report,
        errors=errors,
        predictions=[aggregated_prediction],
    )


async def route_after_forecasts(
    pipeline: AggregationPipeline,
    *,
    analyzer_llm: GeneralLlm | None,
    is_benchmarking: bool,
    research_reports_per_question: int,
    question: MetaculusQuestion,
    qid: int,
    valid_predictions: list[ReasonedPrediction[PredictionTypes]],
    errors: list[str],
    research: str,
    summary_report: str,
    diagnostics_block: str | None,
    time_budget: QuestionTimeBudget,
) -> ResearchWithPredictions[PredictionTypes]:
    """Pick the aggregation path for one question's surviving forecasts.

    ``research`` is the forecaster-clean text; the comment-bound copy gets the
    diagnostics block re-appended here. ``qid`` is passed rather than re-derived so
    the dict keys stay a plain int (the caller has already asserted it is not None).
    """
    comment_research = _with_diagnostics(research, diagnostics_block)

    def base_predictions_collection() -> ResearchWithPredictions[PredictionTypes]:
        return ResearchWithPredictions(
            research_report=comment_research,
            summary_report=summary_report,
            errors=errors,
            predictions=valid_predictions,
        )

    # Single-forecaster short-circuit. When MIN_FORECASTERS_TO_PUBLISH permits it, a
    # question can survive on one forecaster, but the spread metrics (compute_spread
    # and the per-type helpers in spread_metrics.py) REQUIRE >=2 predictions and
    # raise otherwise, and stacking a lone base model is meaningless. So when only
    # one forecaster survived we skip spread + stacking entirely and hand the single
    # prediction to the aggregator, whose base_combine returns it as-is
    # (snap-to-integers applied for discrete numerics). Placed before the budget gate
    # and the per-strategy branches so it short-circuits every stacking path — which
    # is also why this branch has to bump its OWN skip counter: the two increment
    # sites below are unreachable from here. The STACKER_OUTCOME marker stays
    # "skipped" (stable for every parser of the legacy value); the skip REASON rides
    # the additive STACKER_SKIP_REASON marker, because this path computes no spread
    # at all and would otherwise publish identically to a spread-below-threshold skip
    # (q44870 was the first resolved instance of that ambiguity).
    if len(valid_predictions) == 1 and pipeline.strategy in _STACKING_STRATEGIES:
        pipeline.counters.conditional_stacking_skipped_single_forecaster_count += 1
        logger.info(
            "Conditional stacking SKIPPED: single forecaster survived for Q %s; "
            "skipping spread + stacking, aggregating the lone prediction",
            qid,
        )
        pipeline.register_expected_base_combine(question)
        pipeline.outcomes[qid] = "skipped"
        pipeline.skip_reasons[qid] = "single_forecaster"
        return base_predictions_collection()

    skip_stacking_for_budget = _skip_stacking_for_budget(pipeline, question, qid, time_budget)

    if pipeline.strategy == AggregationStrategy.STACKING and not skip_stacking_for_budget:
        if research_reports_per_question != 1:
            logger.warning(
                "STACKING configured with research_reports_per_question=%s; final results will average "
                "per-report stacked outputs by mean.",
                research_reports_per_question,
            )
        return await _finalize_stacked_prediction(
            pipeline,
            question,
            valid_predictions,
            research_for_stacking=research,
            research_report=comment_research,
            summary_report=summary_report,
            errors=errors,
            default_meta_reasoning="Stacked prediction aggregated from multiple models",
        )

    if pipeline.strategy == AggregationStrategy.CONDITIONAL_STACKING and not skip_stacking_for_budget:
        spread, threshold, skip_reason = _conditional_stacking_verdict(pipeline, question, valid_predictions)

        if skip_reason is None:
            pipeline.counters.conditional_stacking_triggered_count += 1
            logger.info(
                "Conditional stacking TRIGGERED: spread=%.3f > threshold=%.3f for question %s",
                spread,
                threshold,
                qid,
            )

            if pipeline.stacker_llm is None:
                raise ValueError("CONDITIONAL_STACKING requires a stacker LLM to be configured")
            if analyzer_llm is None:
                raise ValueError("CONDITIONAL_STACKING requires an analyzer LLM to be configured")

            targeted_research_text = await _targeted_research_for_crux(
                pipeline,
                question,
                analyzer_llm=analyzer_llm,
                is_benchmarking=is_benchmarking,
                valid_predictions=valid_predictions,
                time_budget=time_budget,
            )
            if targeted_research_text:
                combined_research = (
                    f"{research}\n\n## Targeted Research (addressing model disagreement)\n{targeted_research_text}"
                )
            else:
                combined_research = research

            # research_report must be the combined text so the
            # "## Targeted Research (addressing model disagreement)" header reaches
            # the published comment.
            return await _finalize_stacked_prediction(
                pipeline,
                question,
                valid_predictions,
                research_for_stacking=combined_research,
                research_report=_with_diagnostics(combined_research, diagnostics_block),
                summary_report=summary_report,
                errors=errors,
                default_meta_reasoning=(
                    "Conditional stacking: aggregated from multiple models after high-disagreement detected"
                ),
            )

        _record_conditional_skip(
            pipeline,
            question,
            qid,
            skip_reason=skip_reason,
            spread=spread,
            threshold=threshold,
        )
        return base_predictions_collection()

    # Catch-all: a non-stacking strategy, OR a stacking strategy whose budget gate
    # forced the fallback above. In both cases we return the raw valid_predictions
    # and let the pipeline's per-Q aggregator combine them. For the skip case,
    # the stacker outcome was already set upstream so the comment marker reflects reality.
    return base_predictions_collection()
