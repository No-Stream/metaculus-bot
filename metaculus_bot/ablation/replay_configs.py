"""Candidate aggregation configs the offline replay compares, per question type.

One responsibility: the catalogue of aggregation arms — each a callable from one replay
record to the aggregate prediction its scorer wants — plus the sweep constants that define
the arms and the per-(qid, model) weight lookup the coherence arms consume. Split out of
``ablation.offline_replay`` so adding or re-tuning an arm touches only this module, and the
whole catalogue can be exercised against synthetic records without the cache loader or the
scoring/CV machinery.

Pure math over already-loaded records: nothing here reads disk or the network.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from metaculus_bot.ablation.replay_dataset import BinaryRecord, MCRecord, NumericRecord
from metaculus_bot.ablation.weighted_quantiles import weighted_cdf_median, weighted_quantile
from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN
from metaculus_bot.prob_math_utils import clamp_prob, logit
from metaculus_bot.probabilistic_tools.aggregation import log_pool
from metaculus_bot.probabilistic_tools.binary_pooling import (
    adaptive_weight,
    overconfidence_divergence,
    pool_binary,
)
from metaculus_bot.probabilistic_tools.mc_pooling import pool_mc
from metaculus_bot.probabilistic_tools.pdf_pooling import (
    _cdf_probs,
    apply_tail_floor,
    log_pool_cdfs,
    vincentize_cdfs,
)

MEDIAN_BASELINE = "median_baseline"
COHERENCE_SOFTWEIGHT = "coherence_softweight"


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

# Tighter probability clamps swept against the prod incumbent. The cached per-forecaster
# binary probs are ALREADY clamped to [BINARY_PROB_MIN, BINARY_PROB_MAX] = [0.02, 0.98] at
# bench time, so ``median_baseline`` (median of those probs) is effectively the [0.02, 0.98]
# arm: the comparison is tighter-clamp vs. the existing 0.02 clamp. The hypothesis is that a
# tighter clamp bounds the worst-case unbounded log-loss tail on confident-wrong calls (the
# binary "blowups") at a small, asymmetric cost on confident-correct calls. This is a pure
# ceiling/floor on the AGGREGATE probability — it only caps the extremes, it does NOT move
# the middle (unlike shrink-toward-p_math, which is a different hypothesis).
#
# Each bound is (low, high). The after-median clamp is the primary arm; a per-forecaster
# (clamp-each-model-before-the-median) variant is added as a secondary curiosity — with the
# probs already pinned at [0.02, 0.98], the two diverge only when an extreme value survives
# the median, which is exactly the saturation case we care about.
BINARY_CLAMP_BOUNDS: tuple[tuple[float, float], ...] = ((0.05, 0.95), (0.10, 0.90))


def _binary_median_p_model(record: BinaryRecord) -> float:
    return float(np.median(record.p_models))


def _binary_median_p_math(record: BinaryRecord) -> float | None:
    if not record.p_maths:
        return None
    return float(np.median(record.p_maths))


def _binary_geo_odds(record: BinaryRecord) -> float:
    """Geometric-mean-of-odds pool of the per-forecaster probs, clamped to [0.02, 0.98].

    ``log_pool`` = ``sigmoid(mean(logit(p)))``, the normalized geometric-mean-of-odds for
    binary. MEDIAN is invariant to the logit transform, so this is the only mean-type pool that
    can differ from ``median_baseline``: it sharpens toward the tails when forecasters agree and
    stays near the median when they don't. The clamp matches the incumbent's [0.02, 0.98] bounds
    for a fair head-to-head — a no-op on the already-clamped cache, kept for parity with the live
    per-model clamp. ``logit`` self-clamps at 1e-4, so log_pool is safe against 0/1 inputs.
    """
    return min(max(log_pool(record.p_models), BINARY_PROB_MIN), BINARY_PROB_MAX)


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


def _make_binary_clamp_after_median_config(low: float, high: float) -> BinaryConfig:
    """Median the per-forecaster probs (same as ``median_baseline``), then clamp to [low, high].

    This is the PRIMARY clamp arm: the aggregate prediction is computed exactly as the
    incumbent does, then a pure ceiling/floor caps the extremes. It only touches the tails;
    a mid-range median (e.g. 0.5) passes through unchanged.
    """

    def config(record: BinaryRecord) -> float:
        return min(max(_binary_median_p_model(record), low), high)

    return config


def _make_binary_clamp_before_median_config(low: float, high: float) -> BinaryConfig:
    """Clamp each forecaster's prob to [low, high] FIRST, then take the median (secondary arm).

    Differs from the after-median clamp only when an out-of-[low, high] value would otherwise
    survive the median — exactly the confident-extreme case the clamp targets. With the cached
    probs already pinned at [0.02, 0.98], clamping before vs. after the median diverges only on
    questions where a tighter bound bites a majority of the forecasters.
    """

    def config(record: BinaryRecord) -> float:
        clamped = [min(max(p, low), high) for p in record.p_models]
        return float(np.median(clamped))

    return config


def _clamp_config_suffix(low: float, high: float) -> str:
    """Compact name suffix like ``05_95`` / ``10_90`` from the [low, high] bounds."""
    return f"{round(low * 100):02d}_{round(high * 100):02d}"


# Per-(qid, model) coherence-weight lookup shared by all three type builders. The
# VALUES are computed by the caller (the scratch coherence harness) under strict
# era-blocking — hyperparameters come only from fall-aib-2025, never random folds
# across eras — so this module stays a thin, hyperparameter-agnostic consumer of
# externally-derived weights. ``model`` is the raw slug (``payload["model"]``,
# e.g. openrouter/anthropic/claude-opus-4.8), matching ``Record.models``.
WeightLookup = dict[tuple[int, str], float]


def _record_weights(qid: int, models: tuple[str, ...], weights_by_qid_model: WeightLookup) -> list[float] | None:
    """Weight vector aligned to ``models`` order, or ``None`` if any model is unmapped.

    A ``None`` return signals the caller to fall back to equal weights (the median
    baseline) for the whole question — so a missing weight never silently biases the
    combine. With complete per-(qid, model) weights (z=0 imputed for models with no
    coherence signal) this fallback is never hit in practice.
    """
    ws: list[float] = []
    for m in models:
        w = weights_by_qid_model.get((int(qid), m))
        if w is None:
            return None
        ws.append(float(w))
    return ws


def _make_binary_coherence_config(weights_by_qid_model: WeightLookup) -> BinaryConfig:
    """Weighted median of the per-forecaster probs; equal-weight fallback == median_baseline."""

    def config(record: BinaryRecord) -> float:
        ws = _record_weights(record.qid, record.models, weights_by_qid_model)
        if ws is None:
            return _binary_median_p_model(record)
        return weighted_quantile(record.p_models, ws, 0.5)

    return config


def build_binary_configs(weights_by_qid_model: WeightLookup | None = None) -> dict[str, BinaryConfig]:
    """Binary candidate configs keyed by name. ``median_baseline`` is the incumbent.

    The cached per-forecaster probs are already [0.02, 0.98]-clamped at bench time, so
    ``median_baseline`` is effectively the ``clamp_02_98`` arm; the clamp configs below test
    whether a TIGHTER ceiling/floor (after-median primary, per-forecaster-before-median
    secondary) bounds the saturation tail enough to win on log score.

    ``geo_odds`` is the geometric-mean-of-odds pool (``sigmoid(mean(logit(p)))``, clamped to
    [0.02, 0.98]). MEDIAN is invariant to the logit transform, so this is the only mean-type
    pool that can differ from ``median_baseline`` — it sharpens toward the tails when the
    forecasters agree, tests whether that sharpening helps or hurts the ensemble log score.

    When ``weights_by_qid_model`` is supplied, a ``coherence_softweight`` arm is added:
    the weighted median of the per-forecaster probs (equal weights reproduce
    ``median_baseline``). Weights are pre-computed by the caller under era-blocking.
    """
    configs: dict[str, BinaryConfig] = {MEDIAN_BASELINE: _binary_median_p_model}
    configs["geo_odds"] = _binary_geo_odds
    for w in BINARY_SHRINKAGE_WEIGHTS:
        if w == 0.0:
            continue  # w=0 is identical to median_baseline; skip the redundant arm
        configs[f"shrink_w{w:g}"] = _make_binary_shrinkage_config(w)
    configs["shrink_adaptive"] = _binary_adaptive_config
    for low, high in BINARY_CLAMP_BOUNDS:
        suffix = _clamp_config_suffix(low, high)
        configs[f"clamp_{suffix}"] = _make_binary_clamp_after_median_config(low, high)
        configs[f"clamp_{suffix}_premedian"] = _make_binary_clamp_before_median_config(low, high)
    if weights_by_qid_model is not None:
        configs[COHERENCE_SOFTWEIGHT] = _make_binary_coherence_config(weights_by_qid_model)
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


def _make_mc_coherence_config(weights_by_qid_model: WeightLookup) -> MCConfig:
    """Per-option weighted median, renormalized; equal-weight fallback == median_baseline."""

    def config(record: MCRecord) -> list[float]:
        ws = _record_weights(record.qid, record.models, weights_by_qid_model)
        if ws is None:
            return _mc_median_baseline(record)
        matrix = np.array(
            [[vec[name] for name in record.option_order] for vec in record.option_vectors], dtype=float
        )  # (M, K)
        combined = np.array([weighted_quantile(matrix[:, k], ws, 0.5) for k in range(matrix.shape[1])], dtype=float)
        total = float(combined.sum())
        if total <= 0:
            raise ValueError(f"qid {record.qid}: MC weighted median produced non-positive total {total}")
        return list(combined / total)

    return config


def build_mc_configs(weights_by_qid_model: WeightLookup | None = None) -> dict[str, MCConfig]:
    """MC candidate configs keyed by name. ``median_baseline`` is the incumbent.

    When ``weights_by_qid_model`` is supplied, a ``coherence_softweight`` arm is added
    (per-option weighted median + renormalize; equal weights reproduce ``median_baseline``).
    """
    configs: dict[str, MCConfig] = {
        MEDIAN_BASELINE: _mc_median_baseline,
        "pool_mc": _make_mc_pool_config(None),
    }
    for c in MC_DIRICHLET_CONCENTRATIONS:
        configs[f"pool_mc_dir{c:g}"] = _make_mc_pool_config(c)
    if weights_by_qid_model is not None:
        configs[COHERENCE_SOFTWEIGHT] = _make_mc_coherence_config(weights_by_qid_model)
    return configs


# --- Numeric --------------------------------------------------------------

NUMERIC_TAIL_FLOORS: tuple[float, ...] = (1e-3, 5e-3)


def _numeric_vertical(record: NumericRecord, method: Literal["mean", "median"]) -> list[float]:
    """Vertical (pointwise) mean/median of the per-forecaster CDF probabilities — the incumbent."""
    prob_arrays = np.array([_cdf_probs(cdf) for cdf in record.cdfs], dtype=float)
    agg = np.mean(prob_arrays, axis=0) if method == "mean" else np.median(prob_arrays, axis=0)
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


def _make_numeric_coherence_config(weights_by_qid_model: WeightLookup) -> NumericConfig:
    """Weighted vertical CDF median; equal-weight fallback == median_baseline.

    Mirrors the incumbent vertical-median combine but weights the per-grid-point
    quantile by the per-forecaster coherence weight — the drop-in for the current
    groupby-median at ``numeric/utils.py``. All per-model CDFs share the question
    value grid, so the vertical (pointwise) combine is well-defined.
    """

    def config(record: NumericRecord) -> list[float]:
        ws = _record_weights(record.qid, record.models, weights_by_qid_model)
        if ws is None:
            return _numeric_vertical(record, "median")
        prob_matrix = np.array([_cdf_probs(cdf) for cdf in record.cdfs], dtype=float)
        return weighted_cdf_median(prob_matrix, ws)

    return config


def build_numeric_configs(weights_by_qid_model: WeightLookup | None = None) -> dict[str, NumericConfig]:
    """Numeric candidate configs keyed by name. ``median_baseline`` is the incumbent.

    When ``weights_by_qid_model`` is supplied, a ``coherence_softweight`` arm is added
    (weighted vertical CDF median; equal weights reproduce ``median_baseline``).
    """
    configs: dict[str, NumericConfig] = {
        MEDIAN_BASELINE: _numeric_median_baseline,
        "mean_baseline": _numeric_mean_baseline,
        "vincentize_mean": _make_vincentize_config("mean"),
        "vincentize_median": _make_vincentize_config("median"),
        "log_pool": _numeric_log_pool,
    }
    for floor in NUMERIC_TAIL_FLOORS:
        configs[f"mean_tailfloor{floor:g}"] = _make_tail_floor_config(floor)
    if weights_by_qid_model is not None:
        configs[COHERENCE_SOFTWEIGHT] = _make_numeric_coherence_config(weights_by_qid_model)
    return configs
