"""Replay record shapes and their hydration from the ablation cache (zero-API).

One responsibility: turning on-disk ablation artifacts — the qids manifest plus the cached
per-forecaster outputs — into the typed per-question records the replay aggregation configs
consume. Split out of ``ablation.offline_replay`` so the loader (which reuses the live arms'
survivor filter, question shim, prediction deserializer and ground-truth deserializer, and
therefore carries their import chain) can be read and tested apart from the pure aggregation
and scoring math.

Loading is pure disk reads; nothing here calls a forecaster or a provider. The load-bearing
zero-API enforcement stays ``offline_replay.no_network`` — wrap the call site in it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from forecasting_tools import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import OutOfBoundsResolution

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.forecasters import deserialize_prediction_value
from metaculus_bot.ablation.manifest_serde import (
    _build_question_shim_from_manifest_entry,
    _deserialize_ground_truth,
)
from metaculus_bot.ablation.run_stacker import ABLATION_MIN_FORECASTERS, _surviving_forecasters
from metaculus_bot.backtest.scoring import GroundTruth, _canonicalize_mc_option
from metaculus_bot.probabilistic_tools.binary_pooling import reconstruct_p_math
from metaculus_bot.structured_output_schema import BinaryStructured, parse_structured_block

logger: logging.Logger = logging.getLogger(__name__)


# Per-question replay records


@dataclass(frozen=True)
class BinaryRecord:
    """One replayable binary question: per-forecaster p_model + reconstructed p_math + outcome."""

    qid: int
    question: BinaryQuestion
    outcome: bool
    p_models: list[float]
    p_maths: list[float]  # reconstructed via reconstruct_p_math; may be empty if no blocks parse
    # Per-forecaster model slug (``payload["model"]``, e.g. openrouter/anthropic/claude-opus-4.8),
    # aligned index-for-index with ``p_models`` — the join key for per-(qid, model) coherence weights.
    models: tuple[str, ...] = ()


@dataclass(frozen=True)
class MCRecord:
    """One replayable MC question: per-forecaster option-prob vectors + the correct option index."""

    qid: int
    question: MultipleChoiceQuestion
    option_order: list[str]
    correct_option_index: int
    option_vectors: list[dict[str, float]]
    # Per-forecaster model slug, aligned index-for-index with ``option_vectors``.
    models: tuple[str, ...] = ()


@dataclass(frozen=True)
class NumericRecord:
    """One replayable numeric question: per-forecaster 201-point CDFs + the resolution value."""

    qid: int
    question: NumericQuestion
    resolution_value: float  # OutOfBoundsResolution already mapped to slightly-out-of-bounds value
    cdfs: list[list[Percentile]]
    # Per-forecaster model slug, aligned index-for-index with ``cdfs``.
    models: tuple[str, ...] = ()


@dataclass
class ReplayDataset:
    """All replayable questions grouped by type."""

    binary: list[BinaryRecord] = field(default_factory=list)
    mc: list[MCRecord] = field(default_factory=list)
    numeric: list[NumericRecord] = field(default_factory=list)


# Data loading (zero-API)


def _resolution_to_float(resolution: Any, question: NumericQuestion) -> float:
    """Map a numeric ground-truth resolution to a float, handling OutOfBoundsResolution.

    Mirrors ``numeric_log_score_from_report``: an out-of-bounds resolution maps just
    past the corresponding bound so the PMF-bucket index lands in the boundary bucket.
    """
    if isinstance(resolution, OutOfBoundsResolution):
        if resolution == OutOfBoundsResolution.BELOW_LOWER_BOUND:
            return float(question.lower_bound) - 1.0
        if resolution == OutOfBoundsResolution.ABOVE_UPPER_BOUND:
            return float(question.upper_bound) + 1.0
        raise ValueError(f"Unknown OutOfBoundsResolution: {resolution}")
    return float(resolution)


def _reconstruct_p_math_from_block(block: BinaryStructured) -> float | None:
    """Reconstruct p_math for one binary forecaster via ``reconstruct_p_math``.

    Anchor selection mirrors the run_pdf binary cascade (Bayes > prior_blend), but routes
    through the W2 primitive: base-rate counts when present (beta-binomial posterior mean),
    else the stated prior. Evidence items shift the anchor in logit space. Returns None when
    the block carries no usable anchor (no base_rate AND no prior) so the forecaster is
    simply omitted from the p_math aggregate.
    """
    if block.base_rate is not None:
        return reconstruct_p_math(
            base_prob=0.0,  # ignored when base_rate_counts is supplied
            evidence_items=list(block.evidence),
            base_rate_counts=(block.base_rate.k, block.base_rate.n),
        )
    if block.prior is not None:
        return reconstruct_p_math(base_prob=block.prior.prob, evidence_items=list(block.evidence))
    return None


def load_replay_dataset(cache: AblationCache, *, min_forecasters: int = ABLATION_MIN_FORECASTERS) -> ReplayDataset:
    """Load every replayable question from the cache into a typed :class:`ReplayDataset`.

    For each qid present in BOTH the forecaster-outputs directory AND the qids manifest:
    build the question shim + ground truth, take the surviving forecasters (same filter
    the live arms use), deserialize each survivor's prediction, and (for binary) reconstruct
    p_math from the structured block. Questions with fewer than ``min_forecasters`` survivors
    are skipped (matching the live min-forecasters guard).

    Pure disk reads — no network. Wrap the call site in ``offline_replay.no_network`` for the
    belt-and-suspenders zero-API guarantee.
    """
    manifest = cache.read_qids_manifest()
    dataset = ReplayDataset()

    for qid in sorted(manifest.keys()):
        forecaster_payloads = cache.list_forecaster_outputs(qid)
        if not forecaster_payloads:
            continue
        surviving = _surviving_forecasters(forecaster_payloads)
        if len(surviving) < min_forecasters:
            logger.debug("qid=%s skipped: %d survivors < %d", qid, len(surviving), min_forecasters)
            continue

        entry = manifest[qid]
        question = _build_question_shim_from_manifest_entry(qid, entry)
        ground_truth = _deserialize_ground_truth(entry["ground_truth"])

        if isinstance(question, BinaryQuestion):
            dataset.binary.append(_build_binary_record(qid, question, ground_truth, surviving))
        elif isinstance(question, MultipleChoiceQuestion):
            record = _build_mc_record(qid, question, ground_truth, surviving)
            if record is not None:
                dataset.mc.append(record)
        elif isinstance(question, NumericQuestion):
            dataset.numeric.append(_build_numeric_record(qid, question, ground_truth, surviving))
        else:
            raise ValueError(f"Unsupported question type for qid {qid}: {type(question).__name__}")

    logger.info(
        "loaded replay dataset: %d binary / %d mc / %d numeric",
        len(dataset.binary),
        len(dataset.mc),
        len(dataset.numeric),
    )
    return dataset


def _build_binary_record(
    qid: int, question: BinaryQuestion, ground_truth: GroundTruth, surviving: dict[str, dict]
) -> BinaryRecord:
    outcome = ground_truth.resolution
    if not isinstance(outcome, bool):
        raise ValueError(f"qid {qid}: binary resolution must be bool, got {type(outcome).__name__}")
    p_models: list[float] = []
    p_maths: list[float] = []
    models: list[str] = []
    for payload in surviving.values():
        p_models.append(float(deserialize_prediction_value(payload["prediction_value"], question)))
        models.append(str(payload["model"]))
        block = parse_structured_block(payload.get("reasoning", ""), "binary")
        if isinstance(block, BinaryStructured):
            p_math = _reconstruct_p_math_from_block(block)
            if p_math is not None and np.isfinite(p_math):
                p_maths.append(p_math)
    return BinaryRecord(
        qid=qid, question=question, outcome=outcome, p_models=p_models, p_maths=p_maths, models=tuple(models)
    )


def _build_mc_record(
    qid: int, question: MultipleChoiceQuestion, ground_truth: GroundTruth, surviving: dict[str, dict]
) -> MCRecord | None:
    option_order = list(question.options)
    correct = ground_truth.resolution
    correct_index = _mc_correct_index(option_order, correct)
    if correct_index is None:
        logger.warning("qid=%s: MC correct option %r not in options %s; skipping", qid, correct, option_order)
        return None

    option_vectors: list[dict[str, float]] = []
    models: list[str] = []
    for payload in surviving.values():
        predicted = deserialize_prediction_value(payload["prediction_value"], question)
        if not isinstance(predicted, PredictedOptionList):
            raise TypeError(f"qid {qid}: expected PredictedOptionList, got {type(predicted).__name__}")
        vec = dict.fromkeys(option_order, 0.0)
        for opt in predicted.predicted_options:
            if opt.option_name in vec:
                vec[opt.option_name] = float(opt.probability)
        option_vectors.append(vec)
        models.append(str(payload["model"]))
    return MCRecord(
        qid=qid,
        question=question,
        option_order=option_order,
        correct_option_index=correct_index,
        option_vectors=option_vectors,
        models=tuple(models),
    )


def _mc_correct_index(option_order: list[str], correct: Any) -> int | None:
    """Locate the correct option's index, with a canonical-numeric-form fallback.

    Mirrors ``mc_log_score_from_report``: resolution strings sometimes arrive float-formatted
    ('2.0') while options are integer-formatted ('2'), so canonicalize both sides on a miss.
    """
    correct_str = str(correct)
    if correct_str in option_order:
        return option_order.index(correct_str)

    canonical_correct = _canonicalize_mc_option(correct_str)
    canonical_options = [_canonicalize_mc_option(o) for o in option_order]
    if canonical_correct in canonical_options:
        return canonical_options.index(canonical_correct)
    return None


def _build_numeric_record(
    qid: int, question: NumericQuestion, ground_truth: GroundTruth, surviving: dict[str, dict]
) -> NumericRecord:
    resolution_value = _resolution_to_float(ground_truth.resolution, question)
    cdfs: list[list[Percentile]] = []
    models: list[str] = []
    for payload in surviving.values():
        distribution = deserialize_prediction_value(payload["prediction_value"], question)
        # The PchipNumericDistribution exposes its constraint-enforced 201-point CDF via .cdf.
        cdfs.append(list(distribution.cdf))
        models.append(str(payload["model"]))
    return NumericRecord(qid=qid, question=question, resolution_value=resolution_value, cdfs=cdfs, models=tuple(models))
