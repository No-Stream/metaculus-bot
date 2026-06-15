"""Offline replay + iterated-k-fold CV harness for rigorous-PDF aggregation (W5).

Replays the cached per-forecaster predictions for an ablation backtest through the
W1-W3 aggregation primitives ENTIRELY OFFLINE — zero API / network calls — and scores
each candidate aggregation config against real Metaculus resolutions on the tournament
metric (Metaculus-style log score, including saturation blowups). The output is a
per-type comparison of each config vs. the median baseline, under iterated k-fold
cross-validation so we can see whether an edge is stable or evaporates across resamples.

The whole point is correctness: a bug here produces a wrong conclusion about which
aggregation strategy wins. So the data load reuses the EXACT same survivor filter,
deserializer, question shim, and ground-truth deserializer that the live ablation arms
use, and scoring goes straight through the pure ``scoring_common`` primitives.

Zero-API guarantee
------------------
The only inputs are on-disk cache files (forecaster outputs + qids manifest) read via
:class:`AblationCache`. Nothing here *calls* the forecaster, a research/LLM provider, or
``main.py``: we consume cached predictions + ground truth and run pure aggregation math.

Note on import vs. call: the question-shim + ground-truth + deserializer helpers we reuse
live in modules (``ablation.cli`` / ``ablation.forecasters``) that transitively *import*
``metaculus_bot.forecaster`` at module load time. Importing a module is not a network
call — instantiating a forecaster and calling ``.forecast()`` would be. So the load-bearing
enforcement is :func:`no_network` — a context manager that monkeypatches ``socket`` so any
outbound connection during replay raises immediately, making a live call impossible by
construction. Run the whole replay inside ``with no_network():`` and a stray provider call
crashes instead of silently spending credits.

Candidate configs (what we are comparing), per type
---------------------------------------------------
* BINARY: ``median_baseline`` (median of per-forecaster probs — the incumbent), fixed-w
  logit shrinkage ``pool_binary(median(p_model), median(p_math), w)`` for w in {0, .1, .25,
  .5}, and a divergence-gated ``adaptive_weight`` config. ``p_math`` per forecaster is
  reconstructed from its structured block via ``reconstruct_p_math``. Also reports the
  overconfidence measurement |logit(median p_math) - logit(median p_model)| per question.
* MC: ``median_baseline`` (per-option median, renormalized — current behavior), geometric
  ``pool_mc``, and ``pool_mc`` + Dirichlet smoothing at a couple of concentration values.
* NUMERIC: ``median_baseline`` (vertical CDF-median — the incumbent), ``mean_baseline``
  (vertical mean), ``vincentize(mean)``, ``vincentize(median)``, ``log_pool``, plus a small
  tail-floor sweep wrapping the vertical-mean baseline.
"""

from __future__ import annotations

import logging
import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

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
from metaculus_bot.ablation.cli import _build_question_shim_from_manifest_entry, _deserialize_ground_truth
from metaculus_bot.ablation.forecasters import deserialize_prediction_value
from metaculus_bot.ablation.run_stacker import ABLATION_MIN_FORECASTERS, _surviving_forecasters
from metaculus_bot.backtest.scoring import GroundTruth, _canonicalize_mc_option, numeric_crps
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.prob_math_utils import clamp_prob, logit
from metaculus_bot.probabilistic_tools.binary_pooling import (
    adaptive_weight,
    overconfidence_divergence,
    pool_binary,
    reconstruct_p_math,
)
from metaculus_bot.probabilistic_tools.mc_pooling import pool_mc
from metaculus_bot.probabilistic_tools.pdf_pooling import (
    _cdf_probs,
    apply_tail_floor,
    log_pool_cdfs,
    vincentize_cdfs,
)
from metaculus_bot.scoring_common import binary_log_score, brier_score, mc_log_score, numeric_log_score
from metaculus_bot.structured_output_schema import BinaryStructured, parse_structured_block

logger: logging.Logger = logging.getLogger(__name__)

QuestionType = Literal["binary", "multiple_choice", "numeric"]

MEDIAN_BASELINE = "median_baseline"


# Per-question replay records


@dataclass(frozen=True)
class BinaryRecord:
    """One replayable binary question: per-forecaster p_model + reconstructed p_math + outcome."""

    qid: int
    question: BinaryQuestion
    outcome: bool
    p_models: list[float]
    p_maths: list[float]  # reconstructed via reconstruct_p_math; may be empty if no blocks parse


@dataclass(frozen=True)
class MCRecord:
    """One replayable MC question: per-forecaster option-prob vectors + the correct option index."""

    qid: int
    question: MultipleChoiceQuestion
    option_order: list[str]
    correct_option_index: int
    option_vectors: list[dict[str, float]]


@dataclass(frozen=True)
class NumericRecord:
    """One replayable numeric question: per-forecaster 201-point CDFs + the resolution value."""

    qid: int
    question: NumericQuestion
    resolution_value: float  # OutOfBoundsResolution already mapped to slightly-out-of-bounds value
    cdfs: list[list[Percentile]]


@dataclass
class ReplayDataset:
    """All replayable questions grouped by type."""

    binary: list[BinaryRecord] = field(default_factory=list)
    mc: list[MCRecord] = field(default_factory=list)
    numeric: list[NumericRecord] = field(default_factory=list)


# Zero-API guard


class NetworkAccessDuringReplayError(RuntimeError):
    """Raised when the offline replay attempts any outbound network connection."""


@contextmanager
def no_network() -> Iterator[None]:
    """Block outbound network for the duration of the block (real zero-API enforcement).

    Monkeypatches ``socket.getaddrinfo`` — the universal DNS-resolution chokepoint every
    outbound HTTP/LLM/provider call passes through to resolve a hostname — to raise
    :class:`NetworkAccessDuringReplayError`. Run the whole replay under this so a stray
    forecaster / research / provider call crashes loudly instead of silently spending API
    credits. Resolving ``localhost`` / a literal IP is allowed so local tooling still works.

    This is the load-bearing guarantee — far stronger than an import-graph check, because the
    helper modules we reuse legitimately *import* the forecaster at load time without ever
    *calling* it.
    """
    real_getaddrinfo = socket.getaddrinfo

    def _blocked_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host in (None, "localhost", "127.0.0.1", "::1"):
            return real_getaddrinfo(host, *args, **kwargs)
        raise NetworkAccessDuringReplayError(
            f"offline replay attempted to resolve host {host!r}; this must be zero-API"
        )

    # setattr (not direct assignment) so the type checker doesn't flag the deliberate
    # module-attribute swap as an incompatible reassignment of the stdlib signature.
    setattr(socket, "getaddrinfo", _blocked_getaddrinfo)
    try:
        yield
    finally:
        setattr(socket, "getaddrinfo", real_getaddrinfo)


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

    Pure disk reads — no network. Wrap the call site in :func:`no_network` for the
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
    for payload in surviving.values():
        p_models.append(float(deserialize_prediction_value(payload["prediction_value"], question)))
        block = parse_structured_block(payload.get("reasoning", ""), "binary")
        if isinstance(block, BinaryStructured):
            p_math = _reconstruct_p_math_from_block(block)
            if p_math is not None and np.isfinite(p_math):
                p_maths.append(p_math)
    return BinaryRecord(qid=qid, question=question, outcome=outcome, p_models=p_models, p_maths=p_maths)


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
    for payload in surviving.values():
        predicted = deserialize_prediction_value(payload["prediction_value"], question)
        if not isinstance(predicted, PredictedOptionList):
            raise TypeError(f"qid {qid}: expected PredictedOptionList, got {type(predicted).__name__}")
        vec = {name: 0.0 for name in option_order}
        for opt in predicted.predicted_options:
            if opt.option_name in vec:
                vec[opt.option_name] = float(opt.probability)
        option_vectors.append(vec)
    return MCRecord(
        qid=qid,
        question=question,
        option_order=option_order,
        correct_option_index=correct_index,
        option_vectors=option_vectors,
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
    for payload in surviving.values():
        distribution = deserialize_prediction_value(payload["prediction_value"], question)
        # The PchipNumericDistribution exposes its constraint-enforced 201-point CDF via .cdf.
        cdfs.append(list(distribution.cdf))
    return NumericRecord(qid=qid, question=question, resolution_value=resolution_value, cdfs=cdfs)


# Aggregation configs
#
# A config is a callable that takes a per-question record and returns the aggregated
# prediction in the shape the corresponding scorer wants:
#   binary  -> float probability
#   mc      -> list[float] aligned to record.option_order
#   numeric -> list[float] 201-point CDF probabilities

BinaryConfig = Callable[[BinaryRecord], float]
MCConfig = Callable[[MCRecord], list[float]]
NumericConfig = Callable[[NumericRecord], list[float]]


# --- Binary ---------------------------------------------------------------

# Fixed shrinkage weights swept for binary pool_binary(median p_model, median p_math, w).
BINARY_SHRINKAGE_WEIGHTS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)
# Adaptive-weight knobs (divergence-gated). Threshold 0 = shrink whenever the two estimates
# disagree at all; slope/cap are the W2 defaults.
ADAPTIVE_THRESHOLD: float = 0.0
ADAPTIVE_SLOPE: float = 0.25
ADAPTIVE_MAX_WEIGHT: float = 0.5


def _binary_median_p_model(record: BinaryRecord) -> float:
    return float(np.median(record.p_models))


def _binary_median_p_math(record: BinaryRecord) -> float | None:
    if not record.p_maths:
        return None
    return float(np.median(record.p_maths))


def _make_binary_shrinkage_config(w: float) -> BinaryConfig:
    def config(record: BinaryRecord) -> float:
        p_model = _binary_median_p_model(record)
        p_math = _binary_median_p_math(record)
        if p_math is None:
            return p_model  # no structured math available -> fall back to median p_model
        return pool_binary(p_model, p_math, w)

    return config


def _binary_adaptive_config(record: BinaryRecord) -> float:
    p_model = _binary_median_p_model(record)
    p_math = _binary_median_p_math(record)
    if p_math is None:
        return p_model
    divergence = overconfidence_divergence(p_model, p_math)
    w = adaptive_weight(divergence, threshold=ADAPTIVE_THRESHOLD, slope=ADAPTIVE_SLOPE, max_weight=ADAPTIVE_MAX_WEIGHT)
    return pool_binary(p_model, p_math, w)


def build_binary_configs() -> dict[str, BinaryConfig]:
    """Binary candidate configs keyed by name. ``median_baseline`` is the incumbent."""
    configs: dict[str, BinaryConfig] = {MEDIAN_BASELINE: _binary_median_p_model}
    for w in BINARY_SHRINKAGE_WEIGHTS:
        if w == 0.0:
            continue  # w=0 is identical to median_baseline; skip the redundant arm
        configs[f"shrink_w{w:g}"] = _make_binary_shrinkage_config(w)
    configs["shrink_adaptive"] = _binary_adaptive_config
    return configs


def binary_overconfidence(record: BinaryRecord) -> float | None:
    """|logit(median p_math) - logit(median p_model)| for one question, or None if no p_math.

    The empirical gate for whether binary shrinkage is even warranted: if this is ~0 across
    questions, p_math and p_model agree and shrinkage is a no-op.
    """
    p_math = _binary_median_p_math(record)
    if p_math is None:
        return None
    return abs(logit(clamp_prob(p_math)) - logit(clamp_prob(_binary_median_p_model(record))))


# --- MC -------------------------------------------------------------------

MC_DIRICHLET_CONCENTRATIONS: tuple[float, ...] = (10.0, 50.0)


def _mc_vector_to_list(vec: dict[str, float], option_order: list[str]) -> list[float]:
    return [vec[name] for name in option_order]


def _mc_median_baseline(record: MCRecord) -> list[float]:
    """Per-option median across forecasters, renormalized to sum 1 (current behavior)."""
    matrix = np.array([[vec[name] for name in record.option_order] for vec in record.option_vectors], dtype=float)
    medians = np.median(matrix, axis=0)
    total = float(medians.sum())
    if total <= 0:
        raise ValueError(f"qid {record.qid}: MC median produced non-positive total {total}")
    return list(medians / total)


def _make_mc_pool_config(concentration: float | None) -> MCConfig:
    def config(record: MCRecord) -> list[float]:
        pooled = pool_mc(record.option_vectors, concentration=concentration)
        return _mc_vector_to_list(pooled, record.option_order)

    return config


def build_mc_configs() -> dict[str, MCConfig]:
    """MC candidate configs keyed by name. ``median_baseline`` is the incumbent."""
    configs: dict[str, MCConfig] = {
        MEDIAN_BASELINE: _mc_median_baseline,
        "pool_mc": _make_mc_pool_config(None),
    }
    for c in MC_DIRICHLET_CONCENTRATIONS:
        configs[f"pool_mc_dir{c:g}"] = _make_mc_pool_config(c)
    return configs


# --- Numeric --------------------------------------------------------------

NUMERIC_TAIL_FLOORS: tuple[float, ...] = (1e-3, 5e-3)


def _numeric_vertical(record: NumericRecord, method: Literal["mean", "median"]) -> list[float]:
    """Vertical (pointwise) mean/median of the per-forecaster CDF probabilities — the incumbent."""
    prob_arrays = np.array([_cdf_probs(cdf) for cdf in record.cdfs], dtype=float)
    if method == "mean":
        agg = np.mean(prob_arrays, axis=0)
    else:
        agg = np.median(prob_arrays, axis=0)
    agg = np.clip(agg, 0.0, 1.0)
    agg = np.maximum.accumulate(agg)
    return list(map(float, agg))


def _numeric_median_baseline(record: NumericRecord) -> list[float]:
    return _numeric_vertical(record, "median")


def _numeric_mean_baseline(record: NumericRecord) -> list[float]:
    return _numeric_vertical(record, "mean")


def _make_vincentize_config(method: Literal["mean", "median"]) -> NumericConfig:
    def config(record: NumericRecord) -> list[float]:
        pooled = vincentize_cdfs(record.cdfs, record.question, method=method)
        return [p.percentile for p in pooled]

    return config


def _numeric_log_pool(record: NumericRecord) -> list[float]:
    pooled = log_pool_cdfs(record.cdfs, record.question)
    return [p.percentile for p in pooled]


def _make_tail_floor_config(floor_eps: float) -> NumericConfig:
    """Vertical-mean baseline wrapped with apply_tail_floor at ``floor_eps`` (anti-saturation)."""

    def config(record: NumericRecord) -> list[float]:
        mean_cdf = _numeric_mean_baseline(record)
        return apply_tail_floor(mean_cdf, record.question, floor_eps=floor_eps)

    return config


def build_numeric_configs() -> dict[str, NumericConfig]:
    """Numeric candidate configs keyed by name. ``median_baseline`` is the incumbent."""
    configs: dict[str, NumericConfig] = {
        MEDIAN_BASELINE: _numeric_median_baseline,
        "mean_baseline": _numeric_mean_baseline,
        "vincentize_mean": _make_vincentize_config("mean"),
        "vincentize_median": _make_vincentize_config("median"),
        "log_pool": _numeric_log_pool,
    }
    for floor in NUMERIC_TAIL_FLOORS:
        configs[f"mean_tailfloor{floor:g}"] = _make_tail_floor_config(floor)
    return configs


# Scoring (Route A — pure primitives)
#
# PRIMARY metric for every type is the Metaculus-style log score (higher = better),
# INCLUDING saturation blowups. Secondary diagnostics: Brier (binary) and CRPS (numeric).
# Saturation = the per-question primary log score falling below SATURATION_THRESHOLD.

# A binary/MC log score of -100 happens at the prob clamp (p ~ 1e-4, log2(p) ~ -13.3 →
# 100*(log2(1e-4)+1) ~ -1230 for binary). We define "saturated" as a deeply negative
# per-question score; -200 captures genuine blowups (confidently-wrong) without flagging
# routine misses. Numeric uses 50*ln(pmf/baseline); a near-empty bucket (~1e-15) gives a
# very negative score. Same -200 threshold.
SATURATION_THRESHOLD: float = -200.0


def score_binary(record: BinaryRecord, predicted_prob: float) -> tuple[float, float]:
    """(primary log score, secondary brier) for a binary aggregate prediction."""
    return binary_log_score(predicted_prob, record.outcome), brier_score(predicted_prob, record.outcome)


def score_mc(record: MCRecord, predicted_probs: list[float]) -> float:
    """Primary MC log score for an aggregate option-prob vector."""
    return mc_log_score(predicted_probs, record.correct_option_index)


def score_numeric(record: NumericRecord, cdf_values: list[float]) -> tuple[float, float]:
    """(primary log score, secondary CRPS) for a numeric aggregate CDF."""
    q = record.question
    log_score = numeric_log_score(
        cdf_values,
        record.resolution_value,
        float(q.lower_bound),
        float(q.upper_bound),
        bool(q.open_lower_bound),
        bool(q.open_upper_bound),
        float(q.zero_point) if q.zero_point is not None else None,
    )
    # CRPS x-values must be the SAME grid the production CDF lives on: geometric for
    # zero_point (log-scaled) questions, linear otherwise. A linear grid here would
    # mis-locate the CDF mass for zero_point questions and bias CRPS.
    zero_point = float(q.zero_point) if q.zero_point is not None else None
    x_values = list(build_cdf_value_grid(float(q.lower_bound), float(q.upper_bound), zero_point, len(cdf_values)))
    # CRPS needs an in-range resolution; clamp the out-of-bounds sentinel back to the grid.
    crps_resolution = min(max(record.resolution_value, x_values[0]), x_values[-1])
    crps = numeric_crps(x_values, cdf_values, crps_resolution)
    return log_score, crps


# Per-question scoring across all configs


def score_all_binary(records: list[BinaryRecord], configs: dict[str, BinaryConfig]) -> dict[str, np.ndarray]:
    """Per-config array of per-question primary log scores, aligned to ``records`` order."""
    return {
        name: np.array([score_binary(r, config(r))[0] for r in records], dtype=float)
        for name, config in configs.items()
    }


def score_all_mc(records: list[MCRecord], configs: dict[str, MCConfig]) -> dict[str, np.ndarray]:
    return {name: np.array([score_mc(r, config(r)) for r in records], dtype=float) for name, config in configs.items()}


def score_all_numeric(records: list[NumericRecord], configs: dict[str, NumericConfig]) -> dict[str, np.ndarray]:
    return {
        name: np.array([score_numeric(r, config(r))[0] for r in records], dtype=float)
        for name, config in configs.items()
    }


# Iterated k-fold cross-validation


@dataclass(frozen=True)
class ConfigCVResult:
    """CV summary for one config vs. the median baseline.

    ``mean_log_score`` / ``std_log_score`` are over the held-out-fold mean log score across
    all (iteration, fold) resamples. ``delta_vs_median_mean`` / ``delta_vs_median_std`` are
    the PAIRED held-out delta (config minus median, per question, averaged within each fold)
    summarized across resamples. ``full_data_log_score`` is the plain mean over all questions
    (no resampling) for a headline number.

    Selection-bias caveat: the harness reuses the SAME data to fit the CV bands and to pick
    the best-delta config (no nested CV, no multiplicity correction across the 4-6 candidates
    per type), so the winning config's reported band is conditional-on-having-won and is
    optimistically biased toward whichever config came out ahead — read it as edge-STABILITY
    across resamples, not as an unbiased estimate of the selected winner's true edge.
    """

    name: str
    full_data_log_score: float
    mean_log_score: float
    std_log_score: float
    delta_vs_median_mean: float
    delta_vs_median_std: float
    n_resamples: int


def _make_folds(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Partition indices [0, n) into ``k`` shuffled folds (sizes differ by at most 1)."""
    indices = rng.permutation(n)
    return [np.asarray(fold) for fold in np.array_split(indices, k)]


def iterated_kfold_cv(
    per_config_scores: dict[str, np.ndarray],
    *,
    baseline: str = MEDIAN_BASELINE,
    k: int = 5,
    iterations: int = 10,
) -> dict[str, ConfigCVResult]:
    """Iterated k-fold CV over per-question scores. Reports held-out fold means + paired deltas.

    ``per_config_scores[name]`` is the per-question primary-log-score array (same question
    order across configs). For each of ``iterations`` iterations we shuffle (seeded by the
    iteration index for reproducibility) and split into ``k`` folds; each fold's held-out
    questions give one resample. We collect, per config, the fold-mean log score and the
    fold-mean paired delta vs. the baseline. This is a variance-estimation / honesty
    mechanism — with this little data it shows whether an edge is stable, not which config
    to pick.

    Returns a per-config :class:`ConfigCVResult`.
    """
    names = list(per_config_scores.values())
    n = len(names[0]) if names else 0
    for arr in per_config_scores.values():
        if len(arr) != n:
            raise ValueError("all configs must have the same number of per-question scores")
    if baseline not in per_config_scores:
        raise ValueError(f"baseline {baseline!r} not among configs {list(per_config_scores)}")

    config_names = list(per_config_scores.keys())
    fold_means: dict[str, list[float]] = {name: [] for name in config_names}
    fold_deltas: dict[str, list[float]] = {name: [] for name in config_names}
    baseline_scores = per_config_scores[baseline]

    if n == 0:
        return {
            name: ConfigCVResult(name, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 0)
            for name in config_names
        }

    effective_k = min(k, n)
    for it in range(iterations):
        rng = np.random.default_rng(it)
        folds = _make_folds(n, effective_k, rng)
        for fold in folds:
            if fold.size == 0:
                continue
            for name in config_names:
                fold_means[name].append(float(np.mean(per_config_scores[name][fold])))
                fold_deltas[name].append(float(np.mean(per_config_scores[name][fold] - baseline_scores[fold])))

    results: dict[str, ConfigCVResult] = {}
    for name in config_names:
        means = np.array(fold_means[name], dtype=float)
        deltas = np.array(fold_deltas[name], dtype=float)
        results[name] = ConfigCVResult(
            name=name,
            full_data_log_score=float(np.mean(per_config_scores[name])),
            mean_log_score=float(np.mean(means)),
            std_log_score=float(np.std(means)),
            delta_vs_median_mean=float(np.mean(deltas)),
            delta_vs_median_std=float(np.std(deltas)),
            n_resamples=int(means.size),
        )
    return results


def count_saturation_events(scores: np.ndarray) -> int:
    """Number of questions whose primary log score is a saturation blowup (< threshold)."""
    return int(np.sum(np.asarray(scores, dtype=float) < SATURATION_THRESHOLD))


# A config whose per-question scores barely vary is producing near-identical predictions on
# every question — i.e. it has collapsed to a (near-)uniform / constant distribution that
# ignores the data. Its "score" is then an artifact of the metric's baseline, not a real
# aggregation edge. We flag such configs so they aren't mistaken for winners. The threshold
# is in primary-log-score units; on the prod set a genuine numeric aggregator's per-question
# scores have std ~60-90, while the uniform-collapse tail-floor config has std ~1.1 (its
# scores cluster on the 2-3 boundary-baseline values). 5.0 separates the two regimes with a
# wide margin.
DEGENERATE_SCORE_STD: float = 5.0


def is_degenerate_config(scores: np.ndarray) -> bool:
    """True if a config's per-question scores barely vary (collapsed to a constant prediction).

    Catches the tail-floor failure mode where ``floor_eps`` is large enough relative to the
    201-point grid (floor_eps * 200 ~ 1.0) that flooring every step forces a uniform CDF: every
    question then scores the same small constant regardless of where it resolved, which can
    look like a "win" against a baseline that takes occasional blowups. That is not a real
    improvement and must not drive the re-bench decision.
    """
    arr = np.asarray(scores, dtype=float)
    return bool(arr.size >= 2 and np.std(arr) < DEGENERATE_SCORE_STD)
