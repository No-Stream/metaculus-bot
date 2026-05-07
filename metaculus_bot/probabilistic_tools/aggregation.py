from __future__ import annotations

import logging

import numpy as np

from metaculus_bot.probabilistic_tools._numeric_helpers import logit, resolve_weights, sigmoid

logger = logging.getLogger(__name__)


def _validate_probs(probs: list[float]) -> None:
    if not probs:
        raise ValueError("probs must be non-empty")
    for p in probs:
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"all probs must be in [0, 1] (got {p})")


def linear_pool(probs: list[float], weights: list[float] | None = None) -> float:
    _validate_probs(probs)
    w = resolve_weights(weights, len(probs))
    arr = np.array(probs, dtype=float)
    return float(np.sum(w * arr) / np.sum(w))


def log_pool(probs: list[float], weights: list[float] | None = None) -> float:
    """Binary log-linear pool over {p, 1-p}: sigmoid(weighted_mean(logit(p))).

    For binary probabilities this is equivalent to the normalized geometric-
    mean-of-odds pool. NOT generalizable to multi-category — use
    linear_pool_options for categorical distributions, where the arithmetic
    vs. geometric pool distinction matters.
    """
    _validate_probs(probs)
    w = resolve_weights(weights, len(probs))
    logits = np.array([logit(p) for p in probs], dtype=float)
    avg_logit = float(np.sum(w * logits) / np.sum(w))
    return sigmoid(avg_logit)


def satopaa_extremize(
    probs: list[float],
    alpha: float = 2.5,
    weights: list[float] | None = None,
) -> float:
    """Satopää log-odds extremizer: sigmoid(alpha * weighted_mean(logit(p))).

    alpha must be >= 1. alpha=1.0 reduces to the binary log-pool; alpha > 1
    extremizes toward 0/1. We intentionally forbid alpha < 1 (the
    anti-extremization regime) — callers wanting to pull forecasts toward
    0.5 should use linear_pool instead.
    """
    _validate_probs(probs)
    if alpha < 1:
        raise ValueError(f"alpha must be >= 1 (got {alpha})")
    w = resolve_weights(weights, len(probs))
    logits = np.array([logit(p) for p in probs], dtype=float)
    avg_logit = float(np.sum(w * logits) / np.sum(w))
    return sigmoid(alpha * avg_logit)


def inverse_variance_pool(means: list[float], variances: list[float]) -> tuple[float, float]:
    if not means:
        raise ValueError("means must be non-empty")
    if len(means) != len(variances):
        raise ValueError(f"means length {len(means)} must match variances length {len(variances)}")
    for v in variances:
        if v <= 0:
            raise ValueError(f"all variances must be > 0 (got {v})")
    w = np.array([1.0 / v for v in variances], dtype=float)
    mu = np.array(means, dtype=float)
    pooled_mean = float(np.sum(w * mu) / np.sum(w))
    pooled_variance = float(1.0 / np.sum(w))
    return pooled_mean, pooled_variance


def linear_pool_options(
    option_prob_lists: list[dict[str, float]],
    weights: list[float] | None = None,
) -> dict[str, float]:
    """Weighted linear pool across multiple categorical distributions.

    Each ``option_prob_lists`` entry is a dict {option_name -> probability}
    that sums to ~1.0 (tolerance 0.02). All dicts must share the same key
    set. Returns a single dict mapping option_name to the weight-averaged
    probability (no Dirichlet Monte Carlo is performed — the name reflects
    the actual behavior).
    """
    if not option_prob_lists:
        raise ValueError("option_prob_lists must be non-empty")

    first_keys = set(option_prob_lists[0].keys())
    if not first_keys:
        raise ValueError("option_prob_lists entries must be non-empty dicts")
    for i, d in enumerate(option_prob_lists):
        if set(d.keys()) != first_keys:
            raise ValueError(f"option dict {i} has keys {set(d.keys())}, expected {first_keys}")
        total = 0.0
        for k, v in d.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"option dict {i} probability for {k!r} not in [0,1] (got {v})")
            total += v
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"option dict {i} sum {total} not within 0.02 of 1.0")

    w = resolve_weights(weights, len(option_prob_lists))
    w_sum = float(w.sum())

    ordered_keys = list(option_prob_lists[0].keys())
    result: dict[str, float] = {}
    for key in ordered_keys:
        vals = np.array([d[key] for d in option_prob_lists], dtype=float)
        result[key] = float(np.sum(w * vals) / w_sum)
    return result
