"""Structured parser with strict json_schema and framework fallback.

Primary path: invoke a constrained LLM with response_format set to a strict
json_schema wrapper, plus provider.require_parameters=true to prevent silent
schema drops on OpenRouter.

Fallback: on any failure (validation, refusal, truncation, provider error),
fall back to forecasting_tools.structure_output (today's exact behavior).
"""

from __future__ import annotations

import logging
from typing import get_args, get_origin

from forecasting_tools import GeneralLlm, structure_output
from forecasting_tools.data_models.numeric_report import Percentile
from pydantic import BaseModel

from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.simple_types import OptionProbability

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wrapper models for list types (response_format requires a single BaseModel)
# ---------------------------------------------------------------------------


class PercentileListWrapper(BaseModel):
    """Wrapper for list[Percentile] to satisfy json_schema response_format."""

    percentiles: list[Percentile]


class OptionProbabilityListWrapper(BaseModel):
    """Wrapper for list[OptionProbability] to satisfy json_schema response_format."""

    options: list[OptionProbability]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_wrapper_type(output_type: type) -> type[BaseModel] | None:
    """Return a wrapper BaseModel if output_type is a list[X], else None."""
    origin = get_origin(output_type)
    if origin is list:
        args = get_args(output_type)
        if args and len(args) == 1:
            item_type = args[0]
            if item_type is Percentile:
                return PercentileListWrapper
            if item_type is OptionProbability:
                return OptionProbabilityListWrapper
    return None


def _build_constrained_llm(response_format_model: type[BaseModel], parser_model: str) -> GeneralLlm:
    """Build a parser LLM with strict json_schema response_format.

    Uses the same donated-key fallback chain as the production PARSER_LLM.
    The extra_body provider.require_parameters=true ensures OpenRouter rejects
    the request rather than silently dropping the schema.

    ``allowed_tries=1`` + ``timeout=90`` bounds the constrained primary so the
    ``structure_output`` fallback always has budget within the 600s forecaster
    soft deadline; without that cap a stuck primary can consume the entire
    deadline and cancel the coroutine before the fallback runs (F1).
    """
    return build_llm_with_openrouter_fallback(
        parser_model,
        # temperature=None: 0.2.92's GeneralLlm ctor already defaults temperature to
        # None (it was a hard 0 pre-0.2.92), so this is now redundant-but-explicit —
        # kept to pin provider-default sampling against a future default flip. reasoning
        # models defer to provider defaults. No top_p.
        temperature=None,
        max_tokens=32_000,
        stream=False,
        timeout=90,
        allowed_tries=1,
        reasoning={"effort": "low"},
        response_format=response_format_model,
        extra_body={"provider": {"require_parameters": True}},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def parse_structured[T](
    text: str,
    output_type: type[T],
    parser_llm: GeneralLlm,
    *,
    prompt_notes: str = "",
) -> T:
    """Parse text into a pydantic model using strict json_schema, falling back to structure_output.

    Parameters
    ----------
    text:
        The forecaster reasoning to extract structured data from.
    output_type:
        The target pydantic BaseModel (or list[BaseModel] generic).
    parser_llm:
        The parser LLM instance (used only on the fallback path via structure_output).
    prompt_notes:
        Additional extraction instructions (e.g. build_parse_notes for numeric).
    """
    # Determine if we need a wrapper (list types)
    wrapper_type = _get_wrapper_type(output_type)
    schema_model: type[BaseModel] = wrapper_type if wrapper_type is not None else output_type  # type: ignore[assignment]

    # --- Primary path: constrained json_schema ---
    try:
        constrained_llm = _build_constrained_llm(schema_model, parser_llm.model)

        # Build the extraction prompt (simpler than structure_output's — the schema
        # is enforced by the model's constrained decoding, so we just need the text
        # + instructions).
        prompt_parts = [
            "Extract the structured data from the text below.",
        ]
        if prompt_notes:
            prompt_parts.append(f"\nInstructions: {prompt_notes}")
        prompt_parts.append(f"\n\nText:\n{text}")
        prompt = "\n".join(prompt_parts)

        raw_response = await constrained_llm.invoke(prompt)

        # Parse the constrained JSON response
        if wrapper_type is not None:
            wrapper_instance = wrapper_type.model_validate_json(raw_response)
            # Unwrap to the list contents
            if wrapper_type is PercentileListWrapper:
                return wrapper_instance.percentiles  # type: ignore[return-value]
            if wrapper_type is OptionProbabilityListWrapper:
                return wrapper_instance.options  # type: ignore[return-value]
        else:
            return schema_model.model_validate_json(raw_response)  # type: ignore[return-value]

    except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # intentional: catch-all → graceful fallback
        logger.info(
            "Constrained parse failed (%s: %s); falling back to structure_output",
            type(exc).__name__,
            str(exc)[:200],
        )

    # --- Fallback path: today's exact behavior ---
    return await structure_output(
        text_to_structure=text,
        output_type=output_type,
        model=parser_llm,
        additional_instructions=prompt_notes or None,
    )
