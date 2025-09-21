"""Helpers for normalising LLM configuration dictionaries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from forecasting_tools import GeneralLlm

from metaculus_bot.aggregation_strategies import AggregationStrategy

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ForecasterSetup:
    """Container describing the prepared LLM configuration for the forecaster."""

    normalized_llms: Dict[str, Any]
    forecaster_llms: List[GeneralLlm]
    stacker_llm: GeneralLlm | None
    predictions_per_report: int


def prepare_llm_config(
    *,
    llms: Dict[str, Any] | None,
    aggregation_strategy: AggregationStrategy,
    predictions_per_report: int,
) -> ForecasterSetup:
    """Normalise `llms` dict and extract forecaster/stacker models.

    Parameters
    ----------
    llms
        Mapping of LLM roles to configuration supplied by the caller.
    aggregation_strategy
        Current aggregation strategy; influences how defaults are patched.
    predictions_per_report
        Base `predictions_per_research_report` value requested by caller.
    """

    if llms is None:
        raise ValueError("Either 'forecasters' or a 'default' LLM must be provided.")

    normalized_llms: Dict[str, Any] = dict(llms)

    forecaster_llms: List[GeneralLlm] = []
    effective_predictions = predictions_per_report

    if "forecasters" in normalized_llms:
        value = normalized_llms["forecasters"]
        if isinstance(value, list) and all(isinstance(x, GeneralLlm) for x in value):
            if value:
                forecaster_llms = list(value)
                normalized_llms["default"] = forecaster_llms[0]
                effective_predictions = len(forecaster_llms)
        else:
            logger.warning("'forecasters' key in llms must be a list of GeneralLlm objects.")
        normalized_llms.pop("forecasters", None)

    stacker_llm: GeneralLlm | None = None
    if "stacker" in normalized_llms:
        value = normalized_llms["stacker"]
        if isinstance(value, GeneralLlm):
            stacker_llm = value
        else:
            logger.warning("'stacker' key in llms must be a GeneralLlm object.")
        normalized_llms.pop("stacker", None)

    required_keys = {"default", "parser", "researcher", "summarizer"}
    missing = sorted(k for k in required_keys if k not in normalized_llms)
    if missing:
        raise ValueError(f"Missing required LLM purposes: {', '.join(missing)}. Provide these in the 'llms' config.")

    if aggregation_strategy == AggregationStrategy.STACKING and stacker_llm and forecaster_llms:
        normalized_llms["default"] = stacker_llm

    return ForecasterSetup(
        normalized_llms=normalized_llms,
        forecaster_llms=forecaster_llms,
        stacker_llm=stacker_llm,
        predictions_per_report=effective_predictions,
    )


__all__ = ["ForecasterSetup", "prepare_llm_config"]
