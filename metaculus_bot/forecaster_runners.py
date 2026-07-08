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
)
from forecasting_tools.data_models.numeric_report import Percentile
from pydantic import ValidationError

from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN, FORECASTER_SOFT_DEADLINE
from metaculus_bot.exceptions import UnitMismatchError
from metaculus_bot.llm_retry import invoke_with_broad_retry
from metaculus_bot.mc_processing import build_mc_prediction
from metaculus_bot.numeric.config import EXPECTED_PERCENTILE_COUNT, STANDARD_PERCENTILES_CSV
from metaculus_bot.numeric.diagnostics import log_final_prediction
from metaculus_bot.numeric.discrete_snap import OutcomeTypeResult
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric.utils import bound_messages, clamp_and_renormalize_mc
from metaculus_bot.numeric.validation import detect_unit_mismatch
from metaculus_bot.numeric_format_router import route_numeric_output
from metaculus_bot.prompts import binary_prompt, multiple_choice_prompt, numeric_prompt
from metaculus_bot.simple_types import OptionProbability
from metaculus_bot.structured_parse import parse_structured

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


def build_parse_notes(question: NumericQuestion) -> str:
    """Build the parser LLM's extraction instructions for a numeric question.

    The parser's only job is to EXTRACT the forecaster's declared percentile values
    and convert unit suffixes (350B -> 350000000000). It must NOT interpret or
    constrain them. For a CLOSED bound the outcome genuinely cannot cross it, so a
    hard sanity note ("at or above/below {bound}") is fine. For an OPEN bound the
    displayed bound is only the bottom/top of the shown range — the outcome can
    resolve outside it — so the parser must preserve out-of-range values verbatim
    and never clamp them into range. The downstream sanitize layer
    (numeric/bounds_clamping.py) already gates its clamps on open/closed, so this is
    the one remaining place that must respect that distinction.
    """
    unit_str = question.unit_of_measure or "base unit"

    if question.open_lower_bound:
        lower_note = (
            f"The lower bound {question.lower_bound} is only the bottom of the displayed range; the outcome can "
            f"resolve below it. If the forecaster's text states a value below {question.lower_bound}, extract that "
            "value verbatim — never clamp or round it up into range."
        )
    else:
        lower_note = f"Values are at or above the lower bound {question.lower_bound}."

    if question.open_upper_bound:
        upper_note = (
            f"The upper bound {question.upper_bound} is only the top of the displayed range; the outcome can "
            f"resolve above it. If the forecaster's text states a value above {question.upper_bound}, extract that "
            "value verbatim — never clamp or round it down into range."
        )
    else:
        upper_note = f"Values are at or below the upper bound {question.upper_bound}."

    return (
        f"Return exactly these {EXPECTED_PERCENTILE_COUNT} percentiles and no others: {STANDARD_PERCENTILES_CSV}. "
        "Do not include 0 or 100. Use keys 'percentile' (decimal in [0,1]) and 'value' (float). "
        f"Values must be in the base unit '{unit_str}'. The displayed range is [{question.lower_bound}, "
        f"{question.upper_bound}] — use it only to infer scale, not as a constraint. "
        f"{lower_note} {upper_note} "
        "If your text uses B/M/k, convert numerically to base unit (e.g., 350B → 350000000000). No suffixes."
    )


async def run_binary_forecast(
    question: BinaryQuestion,
    research: str,
    forecaster_llm: GeneralLlm,
    parser_llm: GeneralLlm,
) -> ReasonedPrediction[float]:
    prompt = binary_prompt(question, research)
    # Broad, 30s-gated retry (forecaster instances are allowed_tries=1 in
    # llm_configs.py): recovers a fast blip / empty-response while obeying the
    # universal "no retry after 30s" deadline rule. wall_timeout mirrors the outer
    # FORECASTER_SOFT_DEADLINE that _forecaster_with_soft_deadline already enforces.
    reasoning = await invoke_with_broad_retry(
        lambda: forecaster_llm.invoke(prompt), wall_timeout=FORECASTER_SOFT_DEADLINE, label="forecaster_binary"
    )
    _log_llm_output(forecaster_llm.model, question.id_of_question, reasoning)

    binary_parse_instructions = (
        "Return a single JSON object only. Set `prediction_in_decimal` strictly as a decimal in [0,1] "
        "(e.g., 0.17 for 17%). If the text contains 'Probability: NN%' or 'NN %', set `prediction_in_decimal` to NN/100. "
        "Do not return percentages, strings, or any extra fields."
    )
    binary_prediction: BinaryPrediction = await parse_structured(
        reasoning,
        BinaryPrediction,
        parser_llm,
        prompt_notes=binary_parse_instructions,
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
    # Broad, 30s-gated retry — see run_binary_forecast for the rationale.
    reasoning = await invoke_with_broad_retry(
        lambda: forecaster_llm.invoke(prompt), wall_timeout=FORECASTER_SOFT_DEADLINE, label="forecaster_mc"
    )
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
        predicted_option_list: PredictedOptionList = await parse_structured(
            reasoning,
            PredictedOptionList,
            parser_llm,
            prompt_notes=parsing_instructions,
        )
        try:
            predicted_option_list = clamp_and_renormalize_mc(predicted_option_list)
        except ValueError as e:
            logger.warning(f"MC clamp/renormalize failed, using raw predictions: {e}")
    except (ValidationError, ValueError) as exc:
        logger.warning(f"Primary MC parse failed: {exc}")
        raw_options: list[OptionProbability] = await parse_structured(
            reasoning,
            list[OptionProbability],
            parser_llm,
            prompt_notes=parsing_instructions,
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
    # Broad, 30s-gated retry — see run_binary_forecast for the rationale.
    reasoning = await invoke_with_broad_retry(
        lambda: forecaster_llm.invoke(prompt), wall_timeout=FORECASTER_SOFT_DEADLINE, label="forecaster_numeric"
    )

    _log_llm_output(forecaster_llm.model, question.id_of_question, reasoning)

    qid = question.id_of_question
    discrete_vote: bool | None = None

    # C3: try to read outcome_type from the structured JSON block first (saves
    # a parser LLM call). Fall back to the OutcomeTypeResult parser call when
    # the block is missing or doesn't declare outcome_type.
    from metaculus_bot.structured_output_schema import (  # noqa: PLC0415  # function-scoped: see AGENTS.md  # noqa: HARNESS-SCAN-EXEMPT-function-level-import
        NumericStructured,
        parse_structured_block,
    )

    block = parse_structured_block(reasoning, "numeric")
    if isinstance(block, NumericStructured) and block.outcome_type is not None:
        discrete_vote = block.outcome_type == "discrete_integer"
    else:
        try:
            outcome_result: OutcomeTypeResult = await parse_structured(
                reasoning,
                OutcomeTypeResult,
                parser_llm,
                prompt_notes=(
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

    parse_notes = build_parse_notes(question)

    percentile_list: list[Percentile] | None
    try:
        percentile_list = await parse_structured(
            reasoning,
            list[Percentile],
            parser_llm,
            prompt_notes=parse_notes,
        )
    except (ValidationError, ValueError):
        # Parser couldn't extract percentile lines - the router's F5 fallback
        # will try to lift declared_percentiles from the JSON block instead.
        percentile_list = None

    routed = route_numeric_output(
        rationale=reasoning,
        declared_percentiles=percentile_list,
        question=question,
    )
    logger.info(
        "numeric_format=%s for Q %s | model=%s",
        routed.format,
        qid,
        forecaster_llm.model,
    )

    assert routed.declared_percentiles is not None, (
        "route_numeric_output returned without declared_percentiles; "
        "this is a router bug — should have raised ValueError instead."
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
