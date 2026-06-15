"""Multiple-choice ensemble pooling on the probability simplex.

The production MC "pdf" aggregation path is a no-op: it copies each forecaster's
option-probability vector unchanged and then takes the per-option MEDIAN, so the
ensemble result is identical to plain median aggregation. That throws away the
correct simplex operation.

``pool_mc`` replaces it with GEOMETRIC-mean (log-space) pooling: the product of
forecaster probabilities per option, renormalized to sum to 1. Geometric pooling
is the natural aggregation on a probability simplex — it penalizes confident
disagreement harder than the arithmetic mean (a single near-zero vote drags the
pooled probability down multiplicatively), which is the behavior we want for an
ensemble of independent forecasters.

Contrast: ``probabilistic_tools.aggregation.linear_pool_options`` does ARITHMETIC
(weighted-mean) pooling. This module is the geometric counterpart.
"""

from __future__ import annotations

import logging

import numpy as np

from metaculus_bot.prob_math_utils import PROB_CLAMP_EPS, clamp_prob
from metaculus_bot.probabilistic_tools.mc_discrete import dirichlet_with_other

logger = logging.getLogger(__name__)


def _validate_option_vectors(option_prob_vectors: list[dict[str, float]]) -> list[str]:
    """Validate shared keys, [0,1] range, and ~1.0 normalization. Return ordered keys."""
    if not option_prob_vectors:
        raise ValueError("option_prob_vectors must be non-empty")

    first_keys = set(option_prob_vectors[0].keys())
    if not first_keys:
        raise ValueError("option_prob_vectors entries must be non-empty dicts")

    for i, vec in enumerate(option_prob_vectors):
        if set(vec.keys()) != first_keys:
            raise ValueError(f"option vector {i} has keys {set(vec.keys())}, expected {first_keys}")
        total = 0.0
        for k, v in vec.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"option vector {i} probability for {k!r} not in [0,1] (got {v})")
            total += v
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"option vector {i} sum {total} not within 0.02 of 1.0")

    return list(option_prob_vectors[0].keys())


def _smooth_toward_uniform(pooled: dict[str, float], concentration: float) -> dict[str, float]:
    """Symmetric-Dirichlet smoothing: blend the pooled vector toward uniform.

    Posterior mean of a symmetric Dirichlet(1) prior given ``concentration``
    pseudo-observations distributed as ``pooled``:

        smoothed_k = (concentration * pooled_k + 1) / (concentration + n_options)

    concentration -> inf recovers the unsmoothed pool; concentration -> 0 pulls
    every option toward the uniform 1/n. ``dirichlet_with_other`` is the existing
    helper for posterior option means + CIs; we evaluate it on the smoothed vector
    to reuse its validation (positive-mean / normalization checks) and to keep MC
    aggregation routed through one Dirichlet code path.
    """
    if concentration <= 0:
        raise ValueError(f"concentration must be > 0 (got {concentration})")

    keys = list(pooled.keys())
    n_options = len(keys)
    denom = concentration + n_options
    blended = {k: (concentration * pooled[k] + 1.0) / denom for k in keys}

    blend_total = sum(blended.values())
    normalized_blend = {k: v / blend_total for k, v in blended.items()}

    # alpha_k = concentration * mean_k cancels in alpha_k/alpha_total, so .mean
    # equals normalized_blend; the call enforces the simplex/positivity invariants.
    dirichlet = dirichlet_with_other(normalized_blend, other_mass=None, concentration=concentration)
    means = {k: dirichlet[k].mean for k in keys}
    mean_total = sum(means.values())
    return {k: v / mean_total for k, v in means.items()}


def pool_mc(
    option_prob_vectors: list[dict[str, float]],
    concentration: float | None = None,
) -> dict[str, float]:
    """Geometric-mean pool of per-forecaster option-probability vectors.

    Args:
        option_prob_vectors: One dict {option_name -> probability} per forecaster.
            All dicts must share the same key set and each sum to ~1.0 (tol 0.02).
        concentration: If provided, apply symmetric-Dirichlet smoothing toward the
            uniform distribution after pooling. Lower concentration smooths harder;
            ``None`` (default) applies no smoothing.

    Returns:
        A single {option_name -> probability} dict that sums to 1.0.
    """
    keys = _validate_option_vectors(option_prob_vectors)
    n_forecasters = len(option_prob_vectors)

    # Geometric mean per option in log space (floor zeros so log stays finite).
    log_sums = {k: 0.0 for k in keys}
    for vec in option_prob_vectors:
        for k in keys:
            log_sums[k] += np.log(clamp_prob(vec[k], eps=PROB_CLAMP_EPS))

    geo_means = {k: float(np.exp(log_sums[k] / n_forecasters)) for k in keys}
    geo_total = sum(geo_means.values())
    pooled = {k: v / geo_total for k, v in geo_means.items()}

    if concentration is None:
        return pooled

    return _smooth_toward_uniform(pooled, concentration)
