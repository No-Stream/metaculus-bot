"""Mean-aggregation runner: deterministic baseline for the probabilistic-tools ablation.

Bypasses the stacker LLM entirely. For each question, deserializes the per-forecaster
``prediction_value`` payloads already cached on disk and aggregates deterministically
by question type:

* Binary: mean of probabilities, then ``[BINARY_PROB_MIN, BINARY_PROB_MAX]`` clamp.
* Multiple choice: option-wise mean across forecasters per option (preserving the
  question's option order), then ``clamp_and_renormalize_mc`` for sum-to-1.
* Numeric: ``aggregate_numeric(predictions, question, method="mean")`` —
  the same pointwise-CDF aggregator the production pipeline uses, in mean mode.

The output payload is structurally identical to ``run_stacker_for_arm``'s success
payload (same set of top-level keys, same ``stacker_prediction`` schema), so
``_build_report_shim`` and the scoring pipeline consume it without changes.
``stacker_model_used="simple_aggregation"`` is a sentinel literal used only for
the confounder section of the summary.

Cost: zero LLM calls. Wall-clock: O(milliseconds per qid).

Surviving-forecaster filter is single-sourced from
``metaculus_bot.ablation.run_stacker._surviving_forecasters`` so ARM_STACK/ARM_STACK_AUG/ARM_MEAN
all start from an identical surviving set. Min-forecasters threshold mirrors
``ABLATION_MIN_FORECASTERS`` (= 2) for the same reason.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from datetime import datetime
from typing import Any

from forecasting_tools import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.forecasters import (
    deserialize_prediction_value,
    question_type_for_serialization,
    serialize_prediction_value,
)
from metaculus_bot.ablation.run_stacker import (
    ABLATION_MIN_FORECASTERS,
    ARM_MEAN,
    _surviving_forecasters,
)
from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN, MC_PROB_MAX, MC_PROB_MIN
from metaculus_bot.numeric_utils import aggregate_numeric, clamp_and_renormalize_mc

logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["run_mean_for_qid"]

# Sentinel literal stored in ``stacker_model_used`` for ARM_MEAN payloads. The
# downstream report shim doesn't branch on this value — it's purely informational
# for the confounder section of the summary.
SIMPLE_AGGREGATION_LABEL = "simple_aggregation"


def _aggregate_binary(deserialized: list[float]) -> float:
    """Mean of binary probabilities, clamped to [BINARY_PROB_MIN, BINARY_PROB_MAX]."""
    mean_prob = statistics.mean(deserialized)
    return max(BINARY_PROB_MIN, min(BINARY_PROB_MAX, float(mean_prob)))


def _aggregate_mc(
    deserialized: list[PredictedOptionList],
    question: MultipleChoiceQuestion,
) -> PredictedOptionList:
    """Option-wise mean across forecasters, then clamp + renormalize.

    Preserves the question's option order. Each forecaster's PredictedOptionList
    carries the same option_name set; we group probabilities by option_name and
    take the mean per option.
    """
    option_order = list(question.options)
    per_option_values: dict[str, list[float]] = {name: [] for name in option_order}
    for predicted in deserialized:
        for opt in predicted.predicted_options:
            if opt.option_name in per_option_values:
                per_option_values[opt.option_name].append(float(opt.probability))

    raw_probs = {name: float(statistics.mean(per_option_values[name])) for name in option_order}
    clamped = {name: max(MC_PROB_MIN, min(MC_PROB_MAX, p)) for name, p in raw_probs.items()}
    total = sum(clamped.values())
    normalized = {name: (p / total) for name, p in clamped.items()} if total > 0 else clamped
    aggregated_options = [PredictedOption(option_name=name, probability=normalized[name]) for name in option_order]
    aggregated_list = PredictedOptionList(predicted_options=aggregated_options)
    return clamp_and_renormalize_mc(aggregated_list)


async def run_mean_for_qid(
    *,
    qid: int,
    question: MetaculusQuestion,
    forecaster_payloads: dict[str, dict],
    cache: AblationCache,
    force: bool = False,
) -> dict:
    """Run the ARM_MEAN simple-aggregation baseline for one question. Cached per ``(qid, ARM_MEAN)``.

    Steps:
    1. Cache check (``read_stacker_output(qid, ARM_MEAN)``). Hit + not ``force`` returns as-is.
    2. Filter to surviving forecasters via ``_surviving_forecasters`` from run_stacker
       (same filter ARM_STACK / ARM_STACK_AUG use — drops ``prediction_value=None``, errors,
       NaN/inf). Need >= ``ABLATION_MIN_FORECASTERS`` (2).
    3. Deserialize each surviving ``prediction_value`` to its native type.
    4. Aggregate deterministically by question type (mean).
    5. Serialize back to the on-disk schema and write the payload to cache.
    """
    if not force:
        cached = cache.read_stacker_output(qid=qid, arm=ARM_MEAN)
        if cached is not None:
            await asyncio.sleep(0)
            return cached

    surviving = _surviving_forecasters(forecaster_payloads)
    if len(surviving) < ABLATION_MIN_FORECASTERS:
        error_payload = {
            "success": False,
            "arm": ARM_MEAN,
            "reason": "insufficient_forecasters",
            "stacker_prediction": None,
            "stacker_meta_reasoning": "",
            "computed_quantities": {},
            "cross_model_aggregation": "",
            "stacker_model_used": SIMPLE_AGGREGATION_LABEL,
            "n_forecasters_used": len(surviving),
            "ran_at": datetime.now().isoformat(),
            "tools_enabled_at_runtime": False,
            "errors": [],
        }
        cache.write_stacker_output(qid=qid, arm=ARM_MEAN, payload=error_payload)
        await asyncio.sleep(0)
        return error_payload

    deserialized = [
        deserialize_prediction_value(payload["prediction_value"], question) for payload in surviving.values()
    ]

    if isinstance(question, BinaryQuestion):
        aggregated: Any = _aggregate_binary([float(v) for v in deserialized])
    elif isinstance(question, MultipleChoiceQuestion):
        aggregated = _aggregate_mc(deserialized, question)
    elif isinstance(question, NumericQuestion):
        aggregated = await aggregate_numeric(deserialized, question, method="mean")
    else:
        raise ValueError(f"Unsupported question type for ARM_MEAN: {type(question).__name__}")

    serialized_prediction = serialize_prediction_value(aggregated, question_type_for_serialization(question))

    success_payload = {
        "success": True,
        "arm": ARM_MEAN,
        "stacker_prediction": serialized_prediction,
        "stacker_meta_reasoning": "",
        "computed_quantities": {},
        "cross_model_aggregation": "",
        "stacker_model_used": SIMPLE_AGGREGATION_LABEL,
        "n_forecasters_used": len(surviving),
        "ran_at": datetime.now().isoformat(),
        "tools_enabled_at_runtime": False,
        "errors": [],
    }
    cache.write_stacker_output(qid=qid, arm=ARM_MEAN, payload=success_payload)
    # Cooperative yield so flake8-async ASYNC910 sees a checkpoint on the
    # success-path return. Binary and MC branches are sync; only the numeric
    # branch contains an await.
    await asyncio.sleep(0)
    return success_payload
