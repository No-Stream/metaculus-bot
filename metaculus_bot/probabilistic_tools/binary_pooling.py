"""Binary calibration-shrinkage pooling of P_model with P_math.

CRITICAL FRAMING: this is NOT a prior -> posterior Bayesian update. Both
``p_model`` (the forecaster's stated all-considered posterior) and ``p_math``
(a probability reconstructed from the forecaster's own structured base-rate +
evidence math) estimate the SAME quantity P(event), two different ways.
Pooling them in logit space is CALIBRATION SHRINKAGE between two noisy
estimates of the same target. It is only justified if the model posterior is
MEASURABLY overconfident vs. the structured reconstruction on our data, which
is why ``overconfidence_divergence`` lives here too — the offline harness
measures the gap before any weight is committed to.

Nothing in this module runs in the live forecasting path. It is a reusable,
offline-only primitive for the rigorous-PDF aggregation ablation.
"""

from __future__ import annotations

import math
from typing import Any

from metaculus_bot.prob_math_utils import clamp_prob, logit, sigmoid
from metaculus_bot.probabilistic_tools.base_rate import beta_binomial_update

# Strength -> likelihood-ratio mapping, kept identical to the ad-hoc table at
# metaculus_bot/ablation/run_pdf.py:78 (_STRENGTH_TO_LR). Re-defined here rather
# than imported so this module does not depend on the ablation harness; the
# numbers must stay in lockstep with run_pdf.
_STRENGTH_TO_LR: dict[str, float] = {
    "weak": 1.5,
    "moderate": 3.0,
    "strong": 10.0,
}

# Base-prob clamp used before taking log-odds. Matches the inline [0.001, 0.999]
# guard in run_pdf.py:110-113 — deliberately wider than PROB_CLAMP_EPS so the
# reconstruction reproduces the ablation baseline exactly.
_BASE_PROB_FLOOR: float = 0.001
_BASE_PROB_CEIL: float = 0.999


def _apply_evidence_lr(base_prob: float, evidence_items: list[Any]) -> float:
    """Sequential log-odds accumulation from evidence items.

    Faithful re-implementation of run_pdf.py:106-124. Evidence items are
    duck-typed: each must expose ``.direction`` ("up"/"down"/"neutral"),
    ``.strength`` (key into ``_STRENGTH_TO_LR``), and ``.likelihood_ratio``
    (an explicit positive float that overrides the strength mapping, or None).
    """
    if not evidence_items:
        return base_prob
    anchored = min(max(base_prob, _BASE_PROB_FLOOR), _BASE_PROB_CEIL)
    log_odds = math.log(anchored / (1.0 - anchored))
    for item in evidence_items:
        if item.direction == "neutral":
            continue
        lr = _STRENGTH_TO_LR.get(item.strength, 1.0)
        if item.likelihood_ratio is not None and item.likelihood_ratio > 0:
            lr = item.likelihood_ratio
        if item.direction == "down":
            lr = 1.0 / lr
        log_odds += math.log(lr)
    return 1.0 / (1.0 + math.exp(-log_odds))


def reconstruct_p_math(
    base_prob: float,
    evidence_items: list[Any],
    *,
    base_rate_counts: tuple[int, int] | None = None,
) -> float:
    """Reconstruct a probability from a forecaster's structured base + evidence.

    The anchor is ``base_prob`` unless ``base_rate_counts=(k, n)`` is supplied,
    in which case the anchor becomes the beta-binomial posterior mean for
    ``k`` successes in ``n`` trials (reusing ``beta_binomial_update``). Evidence
    items then shift the anchor in logit space via ``_apply_evidence_lr``.

    To avoid double-counting, the anchor is the OUTSIDE-VIEW base rate; the
    evidence items carry all inside-view adjustments. Don't pass an
    all-considered posterior as ``base_prob`` and then re-apply its evidence.
    """
    anchor = base_prob
    if base_rate_counts is not None:
        k, n = base_rate_counts
        anchor = beta_binomial_update(k=k, n=n).posterior_mean
    return _apply_evidence_lr(anchor, evidence_items)


def pool_binary(p_model: float, p_math: float, w: float) -> float:
    """Logit-space pool: logit(out) = w*logit(p_math) + (1-w)*logit(p_model).

    ``w`` is the weight placed on the structured reconstruction. ``w=0`` returns
    ``p_model`` unchanged, ``w=1`` returns ``p_math``. Inputs are clamped with
    ``clamp_prob`` so 0/1 endpoints stay finite (symmetric clamp -> a 0/1 pair
    at ``w=0.5`` collapses to 0.5).
    """
    if not (0.0 <= w <= 1.0):
        raise ValueError(f"w must be in [0, 1] (got {w})")
    logit_model = logit(clamp_prob(p_model))
    logit_math = logit(clamp_prob(p_math))
    return sigmoid(w * logit_math + (1.0 - w) * logit_model)


def overconfidence_divergence(p_model: float, p_math: float) -> float:
    """Per-question divergence signal: |logit(p_math) - logit(p_model)|.

    Symmetric, zero iff the two estimates agree, and the natural gate for
    ``adaptive_weight``. Measured offline to decide whether shrinkage is even
    warranted before any ``w > 0`` is committed.
    """
    return abs(logit(clamp_prob(p_math)) - logit(clamp_prob(p_model)))


def adaptive_weight(
    divergence: float,
    *,
    threshold: float = 0.0,
    slope: float = 0.25,
    max_weight: float = 0.5,
) -> float:
    """Divergence-gated shrinkage weight for ``pool_binary``.

    Returns 0 until ``divergence`` exceeds ``threshold`` (no shrinkage when the
    two estimates already agree), then grows linearly at ``slope`` per unit of
    excess logit-divergence, capped at ``max_weight``. Monotone non-decreasing
    in ``divergence``. Knobs are intentionally minimal — the offline harness
    sweeps them; this just gives a clean shape to sweep over.
    """
    if not (0.0 <= max_weight <= 1.0):
        raise ValueError(f"max_weight must be in [0, 1] (got {max_weight})")
    excess = divergence - threshold
    if excess <= 0.0:
        return 0.0
    return min(slope * excess, max_weight)
