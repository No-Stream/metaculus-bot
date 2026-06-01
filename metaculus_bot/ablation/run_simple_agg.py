"""Unified simple-aggregation runners for the probabilistic-tools ablation.

Both mean and median arms bypass the stacker LLM entirely: they deserialize
per-forecaster ``prediction_value`` payloads and aggregate deterministically
by question type. The output payload is structurally identical to
``run_stacker_for_arm``'s success payload so downstream consumers need no
branching.

Surviving-forecaster filter is single-sourced from
``metaculus_bot.ablation.run_stacker._surviving_forecasters`` so all arms
start from an identical surviving set.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from forecasting_tools import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
)

from metaculus_bot.ablation.aggregation_primitives import aggregate_binary, aggregate_mc
from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.forecasters import (
    deserialize_prediction_value,
    question_type_for_serialization,
    serialize_prediction_value,
)
from metaculus_bot.ablation.run_stacker import (
    ABLATION_MIN_FORECASTERS,
    ARM_MEAN,
    ARM_MEDIAN,
    _surviving_forecasters,
)
from metaculus_bot.ablation.stage_payload import make_error_payload, make_success_payload
from metaculus_bot.numeric.utils import aggregate_numeric

logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["run_mean_for_qid", "run_median_for_qid", "SIMPLE_AGGREGATION_LABEL"]

SIMPLE_AGGREGATION_LABEL = "simple_aggregation"


def _accumulate_mc_options(
    deserialized: list[PredictedOptionList], question: MultipleChoiceQuestion
) -> tuple[dict[str, list[float]], list[str]]:
    """Convert PredictedOptionList items into the per-option-values format aggregate_mc expects."""
    option_order = list(question.options)
    per_option_values: dict[str, list[float]] = {name: [] for name in option_order}
    for predicted in deserialized:
        for opt in predicted.predicted_options:
            if opt.option_name in per_option_values:
                per_option_values[opt.option_name].append(float(opt.probability))
    return per_option_values, option_order


async def _run_simple_agg_for_qid(
    *,
    method: Literal["mean", "median"],
    arm: str,
    qid: int,
    question: MetaculusQuestion,
    forecaster_payloads: dict[str, dict],
    cache: AblationCache,
    force: bool = False,
) -> dict:
    """Shared logic for deterministic simple-aggregation arms (mean / median)."""
    if not force:
        cached = cache.read_stacker_output(qid=qid, arm=arm)
        if cached is not None:
            await asyncio.sleep(0)
            return cached

    surviving = _surviving_forecasters(forecaster_payloads)
    if len(surviving) < ABLATION_MIN_FORECASTERS:
        error_payload = make_error_payload(
            arm=arm,
            reason="insufficient_forecasters",
            model_used=SIMPLE_AGGREGATION_LABEL,
            n_forecasters=len(surviving),
        )
        cache.write_stacker_output(qid=qid, arm=arm, payload=error_payload)
        await asyncio.sleep(0)
        return error_payload

    deserialized = [
        deserialize_prediction_value(payload["prediction_value"], question) for payload in surviving.values()
    ]

    if isinstance(question, BinaryQuestion):
        aggregated: Any = aggregate_binary([float(v) for v in deserialized], method=method)
    elif isinstance(question, MultipleChoiceQuestion):
        per_option_values, option_order = _accumulate_mc_options(deserialized, question)
        aggregated = aggregate_mc(per_option_values, option_order, method=method)
    elif isinstance(question, NumericQuestion):
        aggregated = await aggregate_numeric(deserialized, question, method=method)
    else:
        raise ValueError(f"Unsupported question type for {arm}: {type(question).__name__}")

    serialized_prediction = serialize_prediction_value(aggregated, question_type_for_serialization(question))

    success_payload = make_success_payload(
        arm=arm,
        prediction=serialized_prediction,
        model_used=SIMPLE_AGGREGATION_LABEL,
        n_forecasters=len(surviving),
    )
    cache.write_stacker_output(qid=qid, arm=arm, payload=success_payload)
    # Cooperative yield (ASYNC910 compliance): binary and MC branches are sync;
    # only the numeric branch contains an await above.
    await asyncio.sleep(0)
    return success_payload


async def run_mean_for_qid(
    *,
    qid: int,
    question: MetaculusQuestion,
    forecaster_payloads: dict[str, dict],
    cache: AblationCache,
    force: bool = False,
) -> dict:
    """Run the ARM_MEAN simple-aggregation baseline for one question."""
    return await _run_simple_agg_for_qid(
        method="mean",
        arm=ARM_MEAN,
        qid=qid,
        question=question,
        forecaster_payloads=forecaster_payloads,
        cache=cache,
        force=force,
    )


async def run_median_for_qid(
    *,
    qid: int,
    question: MetaculusQuestion,
    forecaster_payloads: dict[str, dict],
    cache: AblationCache,
    force: bool = False,
) -> dict:
    """Run the ARM_MEDIAN simple-aggregation baseline for one question."""
    return await _run_simple_agg_for_qid(
        method="median",
        arm=ARM_MEDIAN,
        qid=qid,
        question=question,
        forecaster_payloads=forecaster_payloads,
        cache=cache,
        force=force,
    )
