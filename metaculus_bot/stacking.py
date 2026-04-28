from __future__ import annotations

import logging
from collections.abc import Sequence

from forecasting_tools import (
    BinaryPrediction,
    BinaryQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
    structure_output,
)
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.comment_markers import STACKED_BASE_REASONING_HEADER, STACKER_META_ANALYSIS_HEADER
from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN
from metaculus_bot.mc_processing import build_mc_prediction
from metaculus_bot.numeric_utils import clamp_and_renormalize_mc
from metaculus_bot.prompts import stacking_binary_prompt, stacking_multiple_choice_prompt, stacking_numeric_prompt
from metaculus_bot.simple_types import OptionProbability

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
    research: str,
    base_texts: Sequence[str],
) -> tuple[float, str]:
    """Invoke the stacker for a binary question and parse to a decimal probability.

    Returns (prediction_in_decimal, meta_reasoning_text).
    """
    prompt = stacking_binary_prompt(question, research, list(base_texts))
    meta_reasoning = await stacker_llm.invoke(prompt)

    parse_instructions = (
        "Return a single JSON object only. Set `prediction_in_decimal` strictly as a decimal in [0,1] "
        "(e.g., 0.17 for 17%). If the text contains 'Probability: NN%' or 'NN %', set `prediction_in_decimal` to NN/100. "
        "Do not return percentages, strings, or any extra fields."
    )
    binary_prediction: BinaryPrediction = await structure_output(
        meta_reasoning,
        BinaryPrediction,
        model=parser_llm,
        additional_instructions=parse_instructions,
    )
    decimal_pred = max(BINARY_PROB_MIN, min(BINARY_PROB_MAX, binary_prediction.prediction_in_decimal))
    return decimal_pred, meta_reasoning


async def run_stacking_mc(
    stacker_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    question: MultipleChoiceQuestion,
    research: str,
    base_texts: Sequence[str],
) -> tuple[PredictedOptionList, str]:
    """Invoke the stacker for a multiple choice question and parse options.

    Returns (PredictedOptionList, meta_reasoning_text).
    """
    prompt = stacking_multiple_choice_prompt(question, research, list(base_texts))
    meta_reasoning = await stacker_llm.invoke(prompt)

    # Try strict PredictedOptionList first (compatibility) then tolerant fallback
    try:
        parsing_instructions = (
            "Output a JSON array of objects with exactly these two keys per item: `option_name` (string) and "
            "`probability` (decimal in [0,1]). Use option names exactly from this list (case-insensitive accepted):\n"
            f"{question.options}\nDo not include any other options. Remove prefixes like 'Option X:' if present."
        )
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=meta_reasoning,
            output_type=PredictedOptionList,
            model=parser_llm,
            additional_instructions=parsing_instructions,
        )

        try:
            predicted_option_list = clamp_and_renormalize_mc(predicted_option_list)
        except ValueError as e:
            logger.warning(f"MC clamp/renormalize failed: {e}")
    except Exception as e:
        logger.warning(f"Primary MC structured parse failed: {e}")
        raw_options: list[OptionProbability] = await structure_output(
            text_to_structure=meta_reasoning,
            output_type=list[OptionProbability],
            model=parser_llm,
            additional_instructions=parsing_instructions,
        )
        predicted_option_list = build_mc_prediction(raw_options, list(question.options))
    return predicted_option_list, meta_reasoning


async def run_stacking_numeric(
    stacker_llm: GeneralLlm,
    parser_llm: GeneralLlm,
    question: NumericQuestion,
    research: str,
    base_texts: Sequence[str],
    lower_bound_message: str,
    upper_bound_message: str,
) -> tuple[list[Percentile], str]:
    """Invoke the stacker for a numeric question and parse percentiles.

    Returns (declared_percentiles, meta_reasoning_text). The caller should perform
    numeric validation, jitter/clamping, and CDF construction.
    """
    prompt = stacking_numeric_prompt(question, research, list(base_texts), lower_bound_message, upper_bound_message)
    meta_reasoning = await stacker_llm.invoke(prompt)

    unit_str = question.unit_of_measure or "base unit"
    parse_notes = (
        (
            "Return exactly these 11 percentiles and no others: 2.5,5,10,20,40,50,60,80,90,95,97.5. "
            "Do not include 0 or 100. Use keys 'percentile' (decimal in [0,1]) and 'value' (float). "
            f"Values must be in the base unit '{unit_str}' and within [{{lower}}, {{upper}}]. "
            "If your text uses B/M/k, convert numerically to base unit (e.g., 350B → 350000000000). No suffixes."
        )
        .replace("{lower}", str(question.lower_bound))
        .replace("{upper}", str(question.upper_bound))
    )
    percentile_list: list[Percentile] = await structure_output(
        meta_reasoning, list[Percentile], model=parser_llm, additional_instructions=parse_notes
    )
    return percentile_list, meta_reasoning
