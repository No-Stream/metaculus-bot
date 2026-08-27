"""Weighted-quantile aggregation primitives (coherence-weighting offline study).

One responsibility: the shared weighted combine — a Hazen-midpoint weighted quantile over
a value vector, and its vectorized per-grid-point form over a per-model CDF matrix. Split
out of ``ablation.offline_replay`` because these are pure numpy math with no ablation-cache
or forecasting-tools dependency, so the scratch coherence re-aggregation harness (which
works off comment-derived prod ensembles, not the ablation cache) can import the IDENTICAL
operator without pulling in the replay loader's import chain.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# One shared weighted operator so the scratch coherence re-aggregation harness
# (comment-derived prod ensemble) and the ablation replay use the IDENTICAL
# combine. Any arm's offset from the median baseline is then attributable to the
# WEIGHTS, not to a different operator. We use the Hazen (midpoint) plotting
# position ``p_i = S_i - w_i/2`` on the normalized cumulative weight ``S``: at
# equal weights this is ``(i-0.5)/n`` for every n, which the ``q=0.5`` linear
# interpolation reads off as EXACTLY ``np.median`` (odd n → the middle value;
# even n → the two-middle average). It is symmetric in the weights (unlike the
# type-7 ``S_{i-1}/(1-w_last)`` form, whose largest-value weight enters only via
# the denominator, so up-weighting the top model would not move the median).
# The combine is only ever taken at q=0.5 here. On even-M questions the median
# reproduction is machine-precision, not literal bit-identity: ``np.median``
# forms ``(a+b)/2`` while interpolation forms ``a + 0.5*(b-a)`` — a last-ULP
# difference with zero score impact (verified <2e-15 in tests).


def weighted_quantile(values: Any, weights: Any, q: float = 0.5) -> float:
    """Hazen (midpoint) weighted quantile of ``values`` at level ``q``.

    ``values`` / ``weights`` are 1-D, equal length; ``weights`` are non-negative
    and not all zero. Sorted ascending by value, sorted point ``i`` is placed at
    plotting position ``p_i = S_i - w_i/2`` (normalized cumulative-weight
    midpoint). At equal weights ``p_i = (i-0.5)/n``, so ``q=0.5`` reduces to
    ``np.median`` for every n. ``q`` outside ``[p_1, p_n]`` clamps to the nearest
    extreme value (``np.interp`` behavior) — the sensible "all weight on one tail"
    limit.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.shape != w.shape or v.ndim != 1:
        raise ValueError(f"values/weights must be equal-length 1-D arrays, got {v.shape} / {w.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    order = np.argsort(v, kind="stable")
    vs = v[order]
    ws = w[order] / total
    plotting = np.cumsum(ws) - ws / 2.0
    return float(np.interp(q, plotting, vs))


def _normalized_model_weights(weights: Any, n_models: int) -> np.ndarray:
    """Per-model weights normalized to sum 1, defaulting to uniform when ``weights`` is None."""
    if weights is None:
        return np.full(n_models, 1.0 / n_models, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n_models,):
        raise ValueError(f"weights shape {w.shape} must be ({n_models},)")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return w / total


def _interp_at_median_per_column(plotting: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Vectorized ``np.interp(0.5, plotting[:, g], vs[:, g])`` over every column ``g``.

    Reproduces ``np.interp``'s endpoint-clamp semantics: ``q`` at or below the first
    plotting position yields ``vs[0]``, and ``q`` above the last yields ``vs[-1]``.
    Both inputs are ``(n_models, n_grid)`` with ``plotting`` ascending down each column.
    """
    cols = np.arange(plotting.shape[1])
    reaches_half = plotting >= 0.5
    idx_upper = reaches_half.argmax(axis=0)  # first row reaching 0.5 (0 if none reach it)
    none_reach = ~reaches_half.any(axis=0)  # 0.5 above the whole column -> clamp to top
    idx_lower = np.maximum(idx_upper - 1, 0)
    p_lo = plotting[idx_lower, cols]
    span = plotting[idx_upper, cols] - p_lo
    v_lo = vs[idx_lower, cols]
    t = np.where(span > 0, (0.5 - p_lo) / np.where(span > 0, span, 1.0), 0.0)
    out = np.where(idx_upper == 0, vs[0, cols], v_lo + t * (vs[idx_upper, cols] - v_lo))
    return np.where(none_reach, vs[-1, cols], out)


def weighted_cdf_median(prob_matrix: np.ndarray, weights: Any) -> list[float]:
    """Weighted vertical median of a per-model CDF-probability matrix.

    ``prob_matrix`` is ``(n_models, n_grid)`` — each row one model's CDF
    probabilities on the SHARED question value grid. ``weights`` is ``(n_models,)``
    (``None`` → equal). At each grid column the ``q=0.5`` weighted quantile of the
    per-model probabilities is taken (fully vectorized: sort each column, apply the
    per-column plotting positions, interpolate), then clipped to [0,1] and made
    monotone via ``np.maximum.accumulate`` — the exact post-step of
    ``_numeric_vertical`` so equal weights reproduce ``_numeric_median_baseline``.
    """
    mat = np.asarray(prob_matrix, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"prob_matrix must be 2-D (n_models, n_grid), got {mat.shape}")
    n_models = mat.shape[0]
    w = _normalized_model_weights(weights, n_models)

    order = np.argsort(mat, axis=0, kind="stable")  # (M, G)
    vs = np.take_along_axis(mat, order, axis=0)  # sorted probs per column
    ws = w[order]  # weights reordered per column (M, G); each column already sums to 1
    plotting = np.cumsum(ws, axis=0) - ws / 2.0  # Hazen midpoint positions (M, G)

    out = _interp_at_median_per_column(plotting, vs)
    out = np.clip(out, 0.0, 1.0)
    out = np.maximum.accumulate(out)
    return list(map(float, out))
