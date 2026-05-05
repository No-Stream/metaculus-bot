"""Shared logit/sigmoid/clamp utilities.

Canonical epsilon: 1e-4. Used by both live scoring (scoring_common) and
the dormant probability toolkit. If you need a DIFFERENT epsilon for a
specific call site, pass it explicitly via the ``eps`` argument — don't
add a second module-level constant.
"""

from __future__ import annotations

import math

PROB_CLAMP_EPS: float = 1e-4


def clamp_prob(p: float, eps: float = PROB_CLAMP_EPS) -> float:
    """Clamp a probability to [eps, 1-eps] to keep logit/log finite."""
    return min(max(p, eps), 1.0 - eps)


def logit(p: float, eps: float = PROB_CLAMP_EPS) -> float:
    """Log-odds of a probability, clamped to avoid ±inf."""
    p_c = clamp_prob(p, eps)
    return math.log(p_c / (1.0 - p_c))


def sigmoid(x: float) -> float:
    """Numerically-stable sigmoid — no clamping needed on the input side."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)
