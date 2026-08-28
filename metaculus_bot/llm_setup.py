"""Helpers for normalising LLM configuration dictionaries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from forecasting_tools import GeneralLlm

from metaculus_bot.aggregation_strategies import AggregationStrategy

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ForecasterSetup:
    """Container describing the prepared LLM configuration for the forecaster."""

    normalized_llms: dict[str, Any]
    forecaster_llms: list[GeneralLlm]
    stacker_llm: GeneralLlm | None
    analyzer_llm: GeneralLlm | None
    predictions_per_report: int


def _pop_forecaster_llms(normalized_llms: dict[str, Any]) -> list[GeneralLlm]:
    """Pop the ``forecasters`` roster off the llms dict.

    Returns ``[]`` when the key is absent, holds an empty list, or holds the wrong type.
    An empty list is accepted SILENTLY (the caller falls back to its defaults); only a
    wrong type warrants the warning.
    """
    if "forecasters" not in normalized_llms:
        return []
    value = normalized_llms.pop("forecasters")
    if isinstance(value, list) and all(isinstance(x, GeneralLlm) for x in value):
        return list(value)
    logger.warning("'forecasters' key in llms must be a list of GeneralLlm objects.")
    return []


def _pop_single_llm(normalized_llms: dict[str, Any], key: str) -> GeneralLlm | None:
    """Pop a single-model role (``stacker`` / ``analyzer``) off the llms dict.

    Returns None when the key is absent or holds something other than a GeneralLlm; the
    key is removed either way so it never reaches the framework's own llms mapping.
    """
    if key not in normalized_llms:
        return None
    value = normalized_llms.pop(key)
    if isinstance(value, GeneralLlm):
        return value
    logger.warning("'%s' key in llms must be a GeneralLlm object.", key)
    return None


def prepare_llm_config(
    *,
    llms: dict[str, Any] | None,
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

    normalized_llms: dict[str, Any] = dict(llms)

    forecaster_llms = _pop_forecaster_llms(normalized_llms)
    effective_predictions = predictions_per_report
    if forecaster_llms:
        normalized_llms["default"] = forecaster_llms[0]
        effective_predictions = len(forecaster_llms)

    stacker_llm = _pop_single_llm(normalized_llms, "stacker")
    analyzer_llm = _pop_single_llm(normalized_llms, "analyzer")

    required_keys = {"default", "parser", "researcher", "summarizer"}
    missing = sorted(k for k in required_keys if k not in normalized_llms)
    if missing:
        raise ValueError(f"Missing required LLM purposes: {', '.join(missing)}. Provide these in the 'llms' config.")

    if (
        aggregation_strategy in (AggregationStrategy.STACKING, AggregationStrategy.CONDITIONAL_STACKING)
        and stacker_llm
        and forecaster_llms
    ):
        normalized_llms["default"] = stacker_llm

    if aggregation_strategy == AggregationStrategy.CONDITIONAL_STACKING and analyzer_llm is None:
        raise ValueError("CONDITIONAL_STACKING requires an 'analyzer' LLM in the llms config")

    return ForecasterSetup(
        normalized_llms=normalized_llms,
        forecaster_llms=forecaster_llms,
        stacker_llm=stacker_llm,
        analyzer_llm=analyzer_llm,
        predictions_per_report=effective_predictions,
    )


__all__ = ["ForecasterSetup", "prepare_llm_config"]
