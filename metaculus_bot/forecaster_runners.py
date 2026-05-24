"""Per-type forecasting functions extracted from TemplateForecaster.

Each function takes a question, research context, a forecaster LLM (for
generation), and a parser LLM (for structured extraction), then returns
the appropriate ReasonedPrediction.

These are stateless — the caller is responsible for storing any side-effects
(like discrete integer votes for numeric questions).
"""

from __future__ import annotations

import logging

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
    structure_output,
)
from forecasting_tools.data_models.numeric_report import Percentile
from pydantic import ValidationError

from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN
from metaculus_bot.discrete_snap import OutcomeTypeResult
from metaculus_bot.exceptions import UnitMismatchError
from metaculus_bot.mc_processing import build_mc_prediction
from metaculus_bot.numeric_config import STANDARD_PERCENTILES
from metaculus_bot.numeric_diagnostics import log_final_prediction
from metaculus_bot.numeric_format_router import detect_numeric_format, route_numeric_output
from metaculus_bot.numeric_pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric_utils import bound_messages, clamp_and_renormalize_mc
from metaculus_bot.numeric_validation import detect_unit_mismatch
from metaculus_bot.pchip_processing import create_pchip_numeric_distribution
from metaculus_bot.prompts import binary_prompt, multiple_choice_prompt, numeric_prompt
from metaculus_bot.simple_types import OptionProbability

logger = logging.getLogger(__name__)


def _log_llm_output(model_name: str, question_id: int | None, reasoning: str) -> None:
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


async def run_binary_forecast(
    question: BinaryQuestion,
    research: str,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
) -> ReasonedPrediction[float]:
    prompt = binary_prompt(question, research)
    reasoning = await forecaster_llm.invoke(prompt)
    _log_llm_output(forecaster_llm.model, question.id_of_question, reasoning)

    binary_parse_instructions = (
        "Return a single JSON object only. Set `prediction_in_decimal` strictly as a decimal in [0,1] "
        "(e.g., 0.17 for 17%). If the text contains 'Probability: NN%' or 'NN %', set `prediction_in_decimal` to NN/100. "
        "Do not return percentages, strings, or any extra fields."
    )
    binary_prediction: BinaryPrediction = await structure_output(
        reasoning,
        BinaryPrediction,
        model=parser_llm,
        additional_instructions=binary_parse_instructions,
    )
    decimal_pred = max(
        BINARY_PROB_MIN,
        min(BINARY_PROB_MAX, binary_prediction.prediction_in_decimal),
    )

    logger.info(f"Forecasted URL {question.page_url} with prediction: {decimal_pred}")
    return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)


async def run_mc_forecast(
    question: MultipleChoiceQuestion,
    research: str,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
) -> ReasonedPrediction[PredictedOptionList]:
    prompt = multiple_choice_prompt(question, research)
    reasoning = await forecaster_llm.invoke(prompt)
    _log_llm_output(forecaster_llm.model, question.id_of_question, reasoning)

    parsing_instructions = clean_indents(
        f"""
        Output a JSON array of objects with exactly these two keys per item: `option_name` (string) and `probability` (decimal in [0,1]).
        Use option names exactly from this list (case-insensitive match is OK, but prefer canonical spelling):
        {question.options}
        Do not include any options beyond this list. If the source text prefixes with words like 'Option A:' remove the prefix.
        Ensure the probabilities approximately sum to 1.0; slight floating-point drift is OK.
        """
    )

    try:
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=parser_llm,
            additional_instructions=parsing_instructions,
        )
        try:
            predicted_option_list = clamp_and_renormalize_mc(predicted_option_list)
        except ValueError as e:
            logger.warning(f"MC clamp/renormalize failed, using raw predictions: {e}")
    except (ValidationError, ValueError) as exc:
        logger.warning(f"Primary MC parse failed: {exc}")
        raw_options: list[OptionProbability] = await structure_output(
            text_to_structure=reasoning,
            output_type=list[OptionProbability],
            model=parser_llm,
            additional_instructions=parsing_instructions,
        )
        predicted_option_list = build_mc_prediction(raw_options, list(question.options))

    logger.info(f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}")
    return ReasonedPrediction(prediction_value=predicted_option_list, reasoning=reasoning)


async def run_numeric_forecast(
    question: NumericQuestion,
    research: str,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
) -> tuple[ReasonedPrediction[NumericDistribution], bool | None]:
    """Run a numeric forecast and return (prediction, discrete_vote).

    The caller is responsible for storing the discrete_vote in
    _discrete_integer_votes if needed.
    """
    upper_bound_message, lower_bound_message = bound_messages(question)
    prompt = numeric_prompt(question, research, lower_bound_message, upper_bound_message)
    reasoning = await forecaster_llm.invoke(prompt)

    _log_llm_output(forecaster_llm.model, question.id_of_question, reasoning)

    qid = question.id_of_question
    discrete_vote: bool | None = None
    try:
        outcome_result: OutcomeTypeResult = await structure_output(
            reasoning,
            OutcomeTypeResult,
            model=parser_llm,
            additional_instructions=(
                "The forecaster classified whether this question's resolution values are discrete "
                "integers (OUTCOME_TYPE: DISCRETE) or continuous real numbers (OUTCOME_TYPE: CONTINUOUS). "
                "Return is_discrete_integer=true if the forecaster said DISCRETE, false if CONTINUOUS."
            ),
        )
        discrete_vote = outcome_result.is_discrete_integer
    except (ValidationError, ValueError) as e:
        logger.warning("Failed to parse OUTCOME_TYPE for Q %s | model=%s: %s", qid, forecaster_llm.model, e)

    if qid is not None:
        if discrete_vote is True:
            vote_label = "DISCRETE"
        elif discrete_vote is False:
            vote_label = "CONTINUOUS"
        else:
            vote_label = "PARSE_FAILED"
        logger.info(
            "Discrete vote for Q %s | model=%s | vote=%s",
            qid,
            forecaster_llm.model,
            vote_label,
        )

    unit_str = getattr(question, "unit_of_measure", None) or "base unit"
    parse_notes = (
        (
            "Return exactly these 11 percentiles and no others: 2.5,5,10,20,40,50,60,80,90,95,97.5. "
            "Do not include 0 or 100. Use keys 'percentile' (decimal in [0,1]) and 'value' (float). "
            f"Values must be in the base unit '{unit_str}' and within [{{lower}}, {{upper}}]. "
            "If your text uses B/M/k, convert numerically to base unit (e.g., 350B → 350000000000). No suffixes."
        )
        .replace("{lower}", str(getattr(question, "lower_bound", 0)))
        .replace("{upper}", str(getattr(question, "upper_bound", 0)))
    )

    percentile_list: list[Percentile] | None
    try:
        percentile_list = await structure_output(
            reasoning,
            list[Percentile],
            model=parser_llm,
            additional_instructions=parse_notes,
        )
    except (ValidationError, ValueError) as e:
        detected = detect_numeric_format(reasoning)
        if detected in ("mixture", "both"):
            logger.info(
                "Numeric percentile parser found no percentile lines for Q %s | model=%s: %s "
                "(rationale carries mixture_components — using mixture branch)",
                qid,
                forecaster_llm.model,
                e,
            )
            percentile_list = None
        else:
            raise

    routed = route_numeric_output(
        rationale=reasoning,
        declared_percentiles=percentile_list,
        question=question,
    )
    logger.info(
        "numeric_format=%s for Q %s | model=%s | mixture_components=%s",
        routed.format,
        qid,
        forecaster_llm.model,
        (len(routed.mixture.components) if routed.mixture is not None else 0),
    )

    if routed.mixture is not None:
        mixture_cdf_values: list[float] = [float(p.percentile) for p in routed.cdf_percentiles]
        mixture_declared: list[Percentile] = []
        for target_pct in STANDARD_PERCENTILES:
            hit = next(
                (p for p in routed.cdf_percentiles if p.percentile >= target_pct),
                routed.cdf_percentiles[-1],
            )
            mixture_declared.append(Percentile(percentile=target_pct, value=float(hit.value)))

        prediction = create_pchip_numeric_distribution(
            mixture_cdf_values,
            mixture_declared,
            question,
            zero_point=None,
        )
        log_final_prediction(prediction, question)
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning), discrete_vote

    assert routed.declared_percentiles is not None, (
        "route_numeric_output returned a non-mixture result without declared_percentiles; "
        "this is a router bug — mixture-fallback should have raised ValueError instead."
    )
    sanitized_percentiles, zero_point = sanitize_percentiles(routed.declared_percentiles, question)

    prediction = build_numeric_distribution(sanitized_percentiles, question, zero_point)

    mismatch, reason = detect_unit_mismatch(sanitized_percentiles, question)
    if mismatch:
        logger.error(
            f"Unit mismatch likely for Q {getattr(question, 'id_of_question', 'N/A')} | "
            f"URL {getattr(question, 'page_url', '<unknown>')} | reason={reason}. Withholding prediction."
        )
        raise UnitMismatchError(
            f"Unit mismatch likely; {reason}. Values: {[float(p.value) for p in sanitized_percentiles]}"
        )

    log_final_prediction(prediction, question)
    return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning), discrete_vote
