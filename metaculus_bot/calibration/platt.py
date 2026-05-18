"""Platt scaling (logistic recalibration) for final binary / MC probabilities.

The fit is the standard 2-parameter logistic recalibration in log-odds space::

    logit(p_adj) = bias + slope * logit(p_raw)

Equivalently, ``p_adj = sigmoid(bias + slope * logit(p_raw))``. This is the
same parameterization used by the Metaculus notebook "Improving Forecaster
Performance via Automated Calibration Adjustment" (2026-05-01); the only
strategy in that benchmark that produced a statistically significant
improvement on both binary and MC questions.

Conventions in this module:

* ``logit`` / ``sigmoid`` / ``clamp_prob`` are reused from
  ``metaculus_bot.prob_math_utils`` so the canonical 1e-4 epsilon is the
  single source of truth for log-odds clamping.
* Fits refuse on degenerate input (single-class, near-zero or negative slope).
  This matches the article's noted failure modes and avoids silently shipping
  a recalibration that would invert or annihilate the forecaster's signal.
* Apply functions enforce a hard absolute-deviation cap (``max_abs_deviation``)
  on top of the smooth logistic transform. The cap is the load-bearing safety
  rail for "tweak, don't massively deviate" — the underlying fit is allowed
  to want a large move; the cap prevents us from acting on it.

The MC path applies the same single fitted curve to each option's probability
independently, then clamps and renormalizes via the existing
``clamp_and_renormalize_mc`` utility. This matches the article's one-vs-rest
decomposition.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from forecasting_tools.data_models.multiple_choice_report import PredictedOptionList
from scipy.optimize import minimize

from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN
from metaculus_bot.numeric_utils import clamp_and_renormalize_mc
from metaculus_bot.prob_math_utils import PROB_CLAMP_EPS, clamp_prob, logit, sigmoid

logger: logging.Logger = logging.getLogger(__name__)

# Slopes below this threshold are treated as "uninformative forecaster" and
# refused. The 0.05 floor is judgment: a slope of 0.05 means the forecaster's
# log-odds carry almost no signal; a fit at that level is closer to "always
# predict the base rate" than a calibration adjustment.
MIN_INFORMATIVE_SLOPE: float = 0.05


@dataclass(frozen=True)
class PlattParams:
    """Two parameters of a Platt recalibration fit.

    The transform is ``p_adj = sigmoid(bias + slope * logit(p_raw))``.

    * ``bias = 0, slope = 1`` is the identity (no recalibration).
    * ``slope > 1`` makes the forecaster more confident (pushes toward 0/1).
    * ``slope < 1`` makes the forecaster less confident (pulls toward 0.5).
    * ``bias != 0`` shifts the whole curve up or down in log-odds space.
    """

    bias: float
    slope: float

    @classmethod
    def identity(cls) -> "PlattParams":
        return cls(bias=0.0, slope=1.0)

    def is_identity(self) -> bool:
        return self.bias == 0.0 and self.slope == 1.0


def fit_platt(
    raw_probs: Sequence[float],
    outcomes: Sequence[bool],
    *,
    eps: float = PROB_CLAMP_EPS,
) -> PlattParams:
    """Fit ``PlattParams`` by minimizing negative log-likelihood.

    Refuses to fit (raises ``ValueError``) when:

    * fewer than 5 observations are provided (too noisy to be meaningful);
    * all outcomes are the same class (logistic regression is undefined);
    * the fit's recovered slope is non-positive (forecaster is anti-correlated
      with outcomes — applying this would invert the forecast);
    * the fit's recovered slope is below ``MIN_INFORMATIVE_SLOPE`` (forecaster
      log-odds carry essentially no signal; applying this collapses everything
      toward a single bias-driven probability).

    The optimizer is ``scipy.optimize.minimize`` with the L-BFGS-B method —
    same precedent used by ``probabilistic_tools/distributions.py`` and
    ``probabilistic_tools/mixtures.py``. Two parameters, smooth convex loss
    (logistic regression NLL is convex), so a single starting point suffices.
    """
    if len(raw_probs) != len(outcomes):
        raise ValueError(f"raw_probs and outcomes length mismatch: {len(raw_probs)} vs {len(outcomes)}")
    if len(raw_probs) < 5:
        raise ValueError(f"Too few observations to fit Platt: n={len(raw_probs)} < 5")

    y = np.asarray([1.0 if o else 0.0 for o in outcomes], dtype=np.float64)
    if y.sum() == 0.0 or y.sum() == len(y):
        raise ValueError("Cannot fit Platt on single-class training data (all yes or all no)")

    # Pre-compute logits with the canonical epsilon so the fit objective is
    # finite even when the input contains 0s or 1s.
    raw_logits = np.asarray([logit(float(p), eps=eps) for p in raw_probs], dtype=np.float64)

    def _nll(params: np.ndarray) -> float:
        bias, slope = float(params[0]), float(params[1])
        z = bias + slope * raw_logits
        # Numerically stable log(sigmoid(z)) and log(1 - sigmoid(z)) via
        # log1p(exp(-|z|)) trick. softplus(x) = log1p(exp(x)).
        # log_sigmoid(z) = -softplus(-z); log(1 - sigmoid(z)) = -softplus(z).
        log_sig = -np.logaddexp(0.0, -z)
        log_one_minus_sig = -np.logaddexp(0.0, z)
        return float(-(y * log_sig + (1.0 - y) * log_one_minus_sig).sum())

    result = minimize(_nll, x0=np.array([0.0, 1.0]), method="L-BFGS-B")
    if not result.success:
        raise ValueError(f"Platt fit did not converge: {result.message!r}")

    bias_hat, slope_hat = float(result.x[0]), float(result.x[1])

    if slope_hat <= 0.0:
        raise ValueError(
            f"Platt fit produced non-positive slope ({slope_hat:.4f}); "
            "forecaster appears anti-correlated with outcomes. Refusing to apply."
        )
    if slope_hat < MIN_INFORMATIVE_SLOPE:
        raise ValueError(
            f"Platt fit produced slope {slope_hat:.4f} < {MIN_INFORMATIVE_SLOPE}; "
            "forecaster log-odds carry essentially no signal. Refusing to apply."
        )

    return PlattParams(bias=bias_hat, slope=slope_hat)


def apply_binary_platt(
    p_raw: float,
    params: PlattParams,
    *,
    max_abs_deviation: float | None = None,
) -> float:
    """Apply a Platt recalibration to a single binary probability.

    Pipeline:

    1. ``z = bias + slope * logit(p_raw)`` then ``p_adj = sigmoid(z)``.
    2. If ``max_abs_deviation`` is set, clip ``p_adj`` to within
       ``[p_raw - max_abs_deviation, p_raw + max_abs_deviation]``.
    3. Final clamp to ``[BINARY_PROB_MIN, BINARY_PROB_MAX]``.

    The cap is the user's "tweak, don't massively deviate" safety rail.
    Identity params are a fast no-op (still goes through the binary clamp at
    the tail, matching every other binary code path in the bot).
    """
    if not math.isfinite(p_raw):
        raise ValueError(f"p_raw must be finite; got {p_raw!r}")

    if params.is_identity() and max_abs_deviation is None:
        return max(BINARY_PROB_MIN, min(BINARY_PROB_MAX, p_raw))

    p_clamped = clamp_prob(p_raw)
    z = params.bias + params.slope * logit(p_clamped)
    p_adj = sigmoid(z)

    if max_abs_deviation is not None:
        if not (max_abs_deviation >= 0.0):
            raise ValueError(f"max_abs_deviation must be a finite non-negative number; got {max_abs_deviation!r}")
        lower = p_raw - max_abs_deviation
        upper = p_raw + max_abs_deviation
        p_adj = max(lower, min(upper, p_adj))

    return max(BINARY_PROB_MIN, min(BINARY_PROB_MAX, p_adj))


def apply_mc_platt(
    predicted_option_list: PredictedOptionList,
    params: PlattParams,
    *,
    max_abs_deviation: float | None = None,
) -> PredictedOptionList:
    """Apply a Platt recalibration to each MC option, then renormalize.

    The same single fitted curve is applied to each option's probability
    via the binary helper (one-vs-rest decomposition matching the article).
    After the per-option transforms, ``clamp_and_renormalize_mc`` enforces
    ``[MC_PROB_MIN, MC_PROB_MAX]`` and rescales to sum to 1.

    Returns the same ``PredictedOptionList`` for convenience (option
    probabilities are mutated in place, mirroring ``clamp_and_renormalize_mc``).
    """
    if params.is_identity() and max_abs_deviation is None:
        return clamp_and_renormalize_mc(predicted_option_list)

    for option in predicted_option_list.predicted_options:
        option.probability = apply_binary_platt(
            float(option.probability),
            params,
            max_abs_deviation=max_abs_deviation,
        )

    return clamp_and_renormalize_mc(predicted_option_list)
