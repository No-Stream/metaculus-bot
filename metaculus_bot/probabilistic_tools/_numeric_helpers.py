"""Shared numerical helpers for the probabilistic_tools package.

Package-private (underscore) module — do not import from outside
``metaculus_bot.probabilistic_tools``.

The logit/sigmoid/clamp primitives live in ``metaculus_bot.prob_math_utils``
and are re-exported here (with ``clamp_prob`` aliased as ``clamp01``) so
existing internal imports keep working without changes.
"""

from __future__ import annotations

import numpy as np

from metaculus_bot.prob_math_utils import clamp_prob as clamp01
from metaculus_bot.prob_math_utils import logit, sigmoid

__all__ = ["clamp01", "logit", "resolve_weights", "sigmoid", "validate_option_prob_dicts"]


def validate_option_prob_dicts(vectors: list[dict[str, float]]) -> list[str]:
    """Validate a list of option-probability vectors on the simplex.

    Each entry is a dict {option_name -> probability}. All dicts must be
    non-empty, share the first dict's key set, have every value in [0, 1],
    and sum to ~1.0 (tolerance 0.02). Returns the ordered keys of the first
    dict (insertion order preserved). Raises ``ValueError`` on any violation.
    """
    if not vectors:
        raise ValueError("option vectors must be non-empty")

    first_keys = set(vectors[0].keys())
    if not first_keys:
        raise ValueError("option vector entries must be non-empty dicts")

    for i, vec in enumerate(vectors):
        if set(vec.keys()) != first_keys:
            raise ValueError(f"option vector {i} has keys {set(vec.keys())}, expected {first_keys}")
        total = 0.0
        for k, v in vec.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"option vector {i} probability for {k!r} not in [0,1] (got {v})")
            total += v
        if abs(total - 1.0) > 0.02:
            raise ValueError(f"option vector {i} sum {total} not within 0.02 of 1.0")

    return list(vectors[0].keys())


def resolve_weights(weights: list[float] | None, n: int) -> np.ndarray:
    """Validate and normalize a weights argument against an expected length.

    Returns a numpy array of length ``n``. When ``weights`` is ``None`` this
    is an unweighted call and the result is ``np.ones(n)``. Negative weights
    or a zero-sum weight vector raise ``ValueError``.
    """
    if weights is None:
        return np.ones(n, dtype=float)
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} must match probs length {n}")
    for wi in weights:
        if wi < 0:
            raise ValueError(f"weights must be >= 0 (got {wi})")
    w = np.array(weights, dtype=float)
    if float(w.sum()) <= 0:
        raise ValueError("weights must sum to > 0")
    return w
