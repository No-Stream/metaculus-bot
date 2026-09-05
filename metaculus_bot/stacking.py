from __future__ import annotations

import logging
from collections.abc import Sequence

from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.comment.markers import STACKED_BASE_REASONING_HEADER, STACKER_META_ANALYSIS_HEADER
from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN, STACKER_SOFT_DEADLINE
from metaculus_bot.forecaster_runners import build_parse_notes
from metaculus_bot.llm_retry import invoke_with_transient_retry
from metaculus_bot.member_forecast import MEMBER_FORECAST_ROLE_STACKER, format_member_forecast_marker, option_vector
from metaculus_bot.numeric.utils import clamp_and_renormalize_mc
from metaculus_bot.prompts import stacking_binary_prompt, stacking_multiple_choice_prompt, stacking_numeric_prompt
from metaculus_bot.value_extraction import extract_binary, extract_mc, extract_numeric

logger: logging.Logger = logging.getLogger(__name__)


def strip_model_tag(text: str) -> str:
    """Remove a leading "Model: ...\n\n" tag if present.

    This normalizes base-model reasoning snippets before feeding them to the stacker.
    """
    if text.startswith("Model: "):
        parts = text.split("\n", 2)
        if len(parts) >= 3 and parts[1] == "":
            return parts[2]
    return text


def combine_stacker_and_base_reasoning(
    meta_text: str,
    base_predictions: Sequence[ReasonedPrediction],
) -> str:
    """Build the single 'Forecaster 1' reasoning block for a stacked question.

    When stacking fires, the framework collapses all base predictions into one
    ``ReasonedPrediction``. To keep the base models' reasoning visible in the
    published comment (and recoverable by the residual-analysis collector),
    we fold them below the stacker's meta-analysis. Each base reasoning is
    already prefixed with ``Model: ...`` (see ``_make_prediction`` in
    ``main.py``), so downstream parsers can still attribute each block.
    """
    sections = [
        STACKER_META_ANALYSIS_HEADER,
        "",
        meta_text,
        "",
        STACKED_BASE_REASONING_HEADER,
        "",
    ]
    for pred in base_predictions:
        sections.append(pred.reasoning)
        sections.append("")
    return "\n".join(sections)


async def run_stacking_binary(
    stacker_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    question: BinaryQuestion,
    *,
    research: str,
    base_texts: Sequence[str],
    aggregated_tool_output: str | None = None,
    stacker_wall_timeout: float = STACKER_SOFT_DEADLINE,
) -> tuple[float, str]:
    """Invoke the stacker for a binary question and parse to a decimal probability.

    Returns (prediction_in_decimal, meta_reasoning_text).

    ``aggregated_tool_output``: optional markdown from
    ``metaculus_bot.tool_runner.build_cross_model_aggregation``; injected at
    the top of the stacker prompt. Empty / None → no section emitted.

    ``stacker_wall_timeout``: hard wall-clock cap for the stacker invoke, passed
    by the pipeline (STACKER_SOFT_DEADLINE primary / STACKER_FALLBACK_SOFT_DEADLINE
    fallback). The invoke is wrapped in the elapsed-gated transient retry so an
    instant aiohttp blip (litellm #14895) on this allowed_tries=1 stacker recovers,
    while a slow stall propagates to engage the pipeline's fallback chain.
    """
    prompt = stacking_binary_prompt(
        question,
        research,
        list(base_texts),
        aggregated_tool_output=aggregated_tool_output,
    )
    meta_reasoning = await invoke_with_transient_retry(
        lambda: stacker_llm.invoke(prompt), wall_timeout=stacker_wall_timeout, label="stacker"
    )

    parse_instructions = (
        "Return a single JSON object only. Set `prediction_in_decimal` strictly as a decimal in [0,1] "
        "(e.g., 0.17 for 17%). If the text contains 'Probability: NN%' or 'NN %', set `prediction_in_decimal` to NN/100. "
        "Do not return percentages, strings, or any extra fields."
    )
    outcome = await extract_binary(
        meta_reasoning,
        parser_llm,
        prompt_notes=parse_instructions,
        question_id=question.id_of_question,
        model_name=stacker_llm.model,
    )
    decimal_pred = max(BINARY_PROB_MIN, min(BINARY_PROB_MAX, outcome.value))
    logger.info(
        format_member_forecast_marker(
            question_id=question.id_of_question,
            model=stacker_llm.model,
            role=MEMBER_FORECAST_ROLE_STACKER,
            qtype="binary",
            raw=outcome.value,
            published=decimal_pred,
        )
    )
    return decimal_pred, meta_reasoning


async def run_stacking_mc(
    stacker_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    question: MultipleChoiceQuestion,
    *,
    research: str,
    base_texts: Sequence[str],
    aggregated_tool_output: str | None = None,
    stacker_wall_timeout: float = STACKER_SOFT_DEADLINE,
) -> tuple[PredictedOptionList, str]:
    """Invoke the stacker for a multiple choice question and parse options.

    Returns (PredictedOptionList, meta_reasoning_text). See
    ``run_stacking_binary`` for ``aggregated_tool_output`` and
    ``stacker_wall_timeout`` semantics.
    """
    prompt = stacking_multiple_choice_prompt(
        question,
        research,
        list(base_texts),
        aggregated_tool_output=aggregated_tool_output,
    )
    meta_reasoning = await invoke_with_transient_retry(
        lambda: stacker_llm.invoke(prompt), wall_timeout=stacker_wall_timeout, label="stacker"
    )

    parsing_instructions = (
        "Output a JSON array of objects with exactly these two keys per item: `option_name` (string) and "
        "`probability` (decimal in [0,1]). Use option names exactly from this list (case-insensitive accepted):\n"
        f"{question.options}\nDo not include any other options."
    )
    outcome = await extract_mc(
        meta_reasoning,
        list(question.options),
        parser_llm,
        prompt_notes=parsing_instructions,
        question_id=question.id_of_question,
        model_name=stacker_llm.model,
    )
    predicted_option_list = outcome.value.option_list
    try:
        predicted_option_list = clamp_and_renormalize_mc(predicted_option_list)
    except ValueError as e:
        logger.warning(f"MC clamp/renormalize failed: {e}")
    logger.info(
        format_member_forecast_marker(
            question_id=question.id_of_question,
            model=stacker_llm.model,
            role=MEMBER_FORECAST_ROLE_STACKER,
            qtype="multiple_choice",
            raw=outcome.value.declared_probs,  # the list is clamped on construction; see McForecast
            published=option_vector(predicted_option_list),
        )
    )
    return predicted_option_list, meta_reasoning


async def run_stacking_numeric(
    stacker_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    question: NumericQuestion,
    *,
    research: str,
    base_texts: Sequence[str],
    lower_bound_message: str,
    upper_bound_message: str,
    aggregated_tool_output: str | None = None,
    stacker_wall_timeout: float = STACKER_SOFT_DEADLINE,
) -> tuple[list[Percentile], str]:
    """Invoke the stacker for a numeric question and parse percentiles.

    Returns (declared_percentiles, meta_reasoning_text). The caller should perform
    numeric validation, jitter/clamping, and CDF construction. See
    ``run_stacking_binary`` for ``aggregated_tool_output`` and
    ``stacker_wall_timeout`` semantics.
    """
    prompt = stacking_numeric_prompt(
        question,
        research,
        list(base_texts),
        lower_bound_message=lower_bound_message,
        upper_bound_message=upper_bound_message,
        aggregated_tool_output=aggregated_tool_output,
    )
    meta_reasoning = await invoke_with_transient_retry(
        lambda: stacker_llm.invoke(prompt), wall_timeout=stacker_wall_timeout, label="stacker"
    )

    parse_notes = build_parse_notes(question)
    outcome = await extract_numeric(
        meta_reasoning,
        parser_llm,
        prompt_notes=parse_notes,
        question_id=question.id_of_question,
        model_name=stacker_llm.model,
    )
    return outcome.value, meta_reasoning
