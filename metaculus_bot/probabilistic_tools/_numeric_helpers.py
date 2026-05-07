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

__all__ = ["clamp01", "logit", "resolve_weights", "sigmoid"]


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
