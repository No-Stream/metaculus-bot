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

``bot`` is threaded through rather than closed over because the routing writes
run-level state that other stages read: ``_stacker_outcome`` (published in the
comment marker and asserted by the conditional-stacking tests), the three
conditional-stacking counters that feed the end-of-run summary, and the
expected-base-combine registry that keeps the pipeline from logging an
"Unexpected STACKING combine".
"""

import asyncio
import logging
from typing import TYPE_CHECKING

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
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import (
    BINARY_STACKING_ENABLED_ENV,
    CRUX_SOFT_DEADLINE,
    MC_STACKING_ENABLED_ENV,
    NUMERIC_STACKING_ENABLED_ENV,
    WALL_CLOCK_STACKING_MIN_BUDGET,
    env_flag_enabled,
)
from metaculus_bot.research.targeted import extract_disagreement_crux, run_targeted_search
from metaculus_bot.spread_metrics import compute_spread

if TYPE_CHECKING:
    from metaculus_bot.forecaster import TemplateForecaster

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


def _skip_stacking_for_budget(
    bot: "TemplateForecaster", question: MetaculusQuestion, qid: int, per_q_start: float
) -> bool:
    """Force the base-combine fallback when the per-Q wall clock is nearly spent.

    If we've burned through the per-Q budget (research stalled, or the fan-out used
    most of it), skip the stacker LLM entirely. Typical publish is ~1s; the
    ``WALL_CLOCK_STACKING_MIN_BUDGET`` floor leaves headroom for sustained slowness
    on a single POST. The full worst case (both POSTs stalling for
    ``PUBLISH_POST_TIMEOUT * (PUBLISH_POST_RETRIES + 1)``) requires multi-POST
    stalling, which this skip already recovers.
    """
    if bot.aggregation_strategy not in _STACKING_STRATEGIES:
        return False
    remaining = bot._remaining_budget_seconds(per_q_start)
    if remaining >= WALL_CLOCK_STACKING_MIN_BUDGET:
        return False

    # F15: the base-combine re-entry uses MEAN under STACKING and MEDIAN under
    # CONDITIONAL_STACKING (see ``base_combine_strategy`` in
    # aggregation_pipeline.py). The marker must match the actual aggregation
    # method so residual analysis cuts bucket the two paths correctly.
    budget_skip_outcome = (
        "fallback_median" if bot.aggregation_strategy == AggregationStrategy.CONDITIONAL_STACKING else "fallback_mean"
    )
    logger.warning(
        "WALLCLOCK_ABORT: skipping stacking for Q %s; remaining=%.1fs < %ds; forcing %s fallback",
        qid,
        remaining,
        WALL_CLOCK_STACKING_MIN_BUDGET,
        budget_skip_outcome,
    )
    bot._stacker_outcome[qid] = budget_skip_outcome
    # Register so the pipeline's aggregate step (which will run with
    # reasoned_predictions=None) takes the expected base-combine path and doesn't
    # log "Unexpected STACKING combine".
    bot._register_expected_base_combine(question)
    return True


async def _targeted_research_for_crux(
    bot: "TemplateForecaster",
    question: MetaculusQuestion,
    analyzer_llm: GeneralLlm,
    valid_predictions: list[ReasonedPrediction[PredictionTypes]],
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
    """
    # Crux extraction under a soft deadline: without the wait_for the only bound is
    # the analyzer LLM's own litellm timeout (UTILITY_MODEL_CONFIG in
    # llm_configs.py), which is looser than CRUX_SOFT_DEADLINE on this critical path.
    base_texts = [stacking.strip_model_tag(pred.reasoning) for pred in valid_predictions]
    try:
        crux = await asyncio.wait_for(
            extract_disagreement_crux(analyzer_llm, question.question_text, base_texts),
            timeout=CRUX_SOFT_DEADLINE,
        )
    except asyncio.TimeoutError:
        bot._conditional_stacking_crux_failures += 1
        logger.warning(
            "CRUX_SOFT_DEADLINE: crux extraction exceeded %ds for Q %s; skipping targeted research",
            CRUX_SOFT_DEADLINE,
            question.id_of_question,
        )
        return ""
    except Exception:  # noqa: HARNESS-SCAN-EXEMPT-broad-except  # enrichment-only; degrade to base research
        bot._conditional_stacking_crux_failures += 1
        logger.exception("Disagreement crux extraction failed, skipping targeted research")
        return ""

    if not crux:
        return ""
    try:
        return await run_targeted_search(crux, question.question_text, is_benchmarking=bot.is_benchmarking)
    except Exception:  # noqa: HARNESS-SCAN-EXEMPT-broad-except  # enrichment-only; degrade to base research
        bot._conditional_stacking_search_failures += 1
        logger.exception("Targeted search failed, proceeding with base research only")
        return ""


async def route_after_forecasts(
    bot: "TemplateForecaster",
    *,
    question: MetaculusQuestion,
    qid: int,
    valid_predictions: list[ReasonedPrediction[PredictionTypes]],
    errors: list[str],
    research: str,
    summary_report: str,
    diagnostics_block: str | None,
    per_q_start: float,
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
    # prediction to the aggregator, whose _base_combine returns it as-is
    # (snap-to-integers applied for discrete numerics). Placed before the budget gate
    # and the per-strategy branches so it short-circuits every stacking path — which
    # is also why this branch has to bump its OWN skip counter: the two increment
    # sites below are unreachable from here. The _stacker_outcome marker is "skipped"
    # (stacking was skipped, non-stacked aggregation); the distinct log line and
    # counter record the single-forecaster reason.
    if len(valid_predictions) == 1 and bot.aggregation_strategy in _STACKING_STRATEGIES:
        bot._conditional_stacking_skipped_single_forecaster_count += 1
        logger.info(
            "Conditional stacking SKIPPED: single forecaster survived for Q %s; "
            "skipping spread + stacking, aggregating the lone prediction",
            qid,
        )
        bot._register_expected_base_combine(question)
        bot._stacker_outcome[qid] = "skipped"
        return base_predictions_collection()

    skip_stacking_for_budget = _skip_stacking_for_budget(bot, question, qid, per_q_start)

    if bot.aggregation_strategy == AggregationStrategy.STACKING and not skip_stacking_for_budget:
        if bot.research_reports_per_question != 1:
            logger.warning(
                "STACKING configured with research_reports_per_question=%s; final results will average "
                "per-report stacked outputs by mean.",
                bot.research_reports_per_question,
            )
        return await bot._finalize_stacked_prediction(
            question,
            valid_predictions,
            research_for_stacking=research,
            research_report=comment_research,
            summary_report=summary_report,
            errors=errors,
            default_meta_reasoning="Stacked prediction aggregated from multiple models",
        )

    if bot.aggregation_strategy == AggregationStrategy.CONDITIONAL_STACKING and not skip_stacking_for_budget:
        spread = compute_spread(question, [pred.prediction_value for pred in valid_predictions])
        threshold = bot._get_threshold_for_question(question)

        spread_exceeds_threshold = spread > threshold
        # Disagreement was high enough to trigger stacking, but the per-type gate is
        # off, so we deliberately bypass it.
        type_stacking_disabled = spread_exceeds_threshold and not _type_gate_enabled(question)
        if type_stacking_disabled:
            spread_exceeds_threshold = False

        if spread_exceeds_threshold:
            bot._conditional_stacking_triggered_count += 1
            logger.info(
                "Conditional stacking TRIGGERED: spread=%.3f > threshold=%.3f for question %s",
                spread,
                threshold,
                qid,
            )

            analyzer_llm = bot._analyzer_llm
            if bot._stacker_llm is None:
                raise ValueError("CONDITIONAL_STACKING requires a stacker LLM to be configured")
            if analyzer_llm is None:
                raise ValueError("CONDITIONAL_STACKING requires an analyzer LLM to be configured")

            targeted_research_text = await _targeted_research_for_crux(bot, question, analyzer_llm, valid_predictions)
            if targeted_research_text:
                combined_research = (
                    f"{research}\n\n## Targeted Research (addressing model disagreement)\n{targeted_research_text}"
                )
            else:
                combined_research = research

            # research_report must be the combined text so the
            # "## Targeted Research (addressing model disagreement)" header reaches
            # the published comment.
            return await bot._finalize_stacked_prediction(
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

        bot._conditional_stacking_skipped_count += 1
        if type_stacking_disabled:
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
        bot._register_expected_base_combine(question)
        # "skipped_config_off" (spread exceeded the threshold but the per-type gate
        # was off) vs plain "skipped" (spread at/below threshold) — keeps the
        # suppression reason durable in the published marker instead of requiring git
        # archaeology over workflow-yaml flag history.
        bot._stacker_outcome[qid] = "skipped_config_off" if type_stacking_disabled else "skipped"
        return base_predictions_collection()

    # Catch-all: a non-stacking strategy, OR a stacking strategy whose budget gate
    # forced the fallback above. In both cases we return the raw valid_predictions
    # and let the pipeline's per-Q aggregator combine them. For the skip case
    # _stacker_outcome was already set upstream so the comment marker reflects reality.
    return base_predictions_collection()
