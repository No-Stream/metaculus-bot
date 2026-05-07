from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from metaculus_bot.probabilistic_tools._numeric_helpers import logit, resolve_weights, sigmoid

logger = logging.getLogger(__name__)

# Default strength (pseudo-count total α+β) for an informative Beta prior
# centered on ``prior_mean``. 5 pseudo-counts is a mildly informative prior
# that moves with the stated outside view but lets modest data (n>=10) shift
# the posterior noticeably. Tune via backtest.
DEFAULT_INFORMATIVE_PRIOR_STRENGTH: float = 5.0


@dataclass(frozen=True)
class BetaBinomialResult:
    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    ci_80_low: float
    ci_80_high: float
    ci_90_low: float
    ci_90_high: float


def beta_binomial_update(
    k: int,
    n: int,
    prior_mean: float = 0.5,
    prior_strength: float = 1.0,
) -> BetaBinomialResult:
    """Bayesian update on a Beta-binomial model with informative prior.

    The prior is Beta(α, β) with ``α = prior_strength * prior_mean`` and
    ``β = prior_strength * (1 - prior_mean)``. Defaults
    ``prior_mean=0.5, prior_strength=1.0`` give a weakly informative prior
    (roughly Jeffreys' strength with uniform center).

    To pass a forecaster's outside-view as the prior, use
    ``prior_mean=<stated probability>`` with
    ``prior_strength=DEFAULT_INFORMATIVE_PRIOR_STRENGTH``.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0 (got {n})")
    if k < 0 or k > n:
        raise ValueError(f"k must satisfy 0 <= k <= n (got k={k}, n={n})")
    if not (0.0 < prior_mean < 1.0):
        raise ValueError(f"prior_mean must be in (0, 1) (got {prior_mean})")
    if prior_strength <= 0:
        raise ValueError(f"prior_strength must be > 0 (got {prior_strength})")

    alpha_prior = prior_strength * prior_mean
    beta_prior = prior_strength * (1.0 - prior_mean)
    a = alpha_prior + k
    b = beta_prior + n - k
    mean = a / (a + b)
    ci_80 = stats.beta.ppf([0.10, 0.90], a, b)
    ci_90 = stats.beta.ppf([0.05, 0.95], a, b)
    return BetaBinomialResult(
        posterior_alpha=float(a),
        posterior_beta=float(b),
        posterior_mean=float(mean),
        ci_80_low=float(ci_80[0]),
        ci_80_high=float(ci_80[1]),
        ci_90_low=float(ci_90[0]),
        ci_90_high=float(ci_90[1]),
    )


def laplace_rule_of_succession(k: int, n: int) -> float:
    if n < 0:
        raise ValueError(f"n must be >= 0 (got {n})")
    if k < 0 or k > n:
        raise ValueError(f"k must satisfy 0 <= k <= n (got k={k}, n={n})")
    return (k + 1) / (n + 2)


def base_rate_blend(
    rates: list[float],
    weights: list[float] | None = None,
    method: Literal["linear", "log_odds", "inverse_variance"] = "linear",
    variances: list[float] | None = None,
) -> float:
    if not rates:
        raise ValueError("rates must be non-empty")
    for r in rates:
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"all rates must be in [0, 1] (got {r})")

    if method == "inverse_variance":
        if variances is None:
            raise ValueError("variances must be provided when method='inverse_variance'")
        if weights is not None:
            raise ValueError(
                "base_rate_blend(method='inverse_variance') uses variances for weighting; "
                "pass either weights OR variances, not both"
            )
        if len(variances) != len(rates):
            raise ValueError(f"variances length {len(variances)} must match rates length {len(rates)}")
        for v in variances:
            if v <= 0:
                raise ValueError(f"all variances must be > 0 (got {v})")
        inv = np.array([1.0 / v for v in variances])
        arr = np.array(rates, dtype=float)
        return float(np.sum(inv * arr) / np.sum(inv))

    w = resolve_weights(weights, len(rates))

    if method == "linear":
        arr = np.array(rates, dtype=float)
        return float(np.sum(w * arr) / np.sum(w))
    if method == "log_odds":
        logits = np.array([logit(r) for r in rates], dtype=float)
        avg = float(np.sum(w * logits) / np.sum(w))
        return sigmoid(avg)

    raise ValueError(f"unknown method: {method}")


def implied_likelihood_ratio(prior_prob: float, posterior_prob: float) -> float:
    """Return (posterior_odds / prior_odds).

    Raises ``ValueError`` when either probability is at the boundary (0 or 1);
    the likelihood ratio is genuinely undefined there (0/0 or ∞/anything).
    Callers that want a graceful degradation (e.g., consistency checks) should
    catch ValueError and branch on "LR undefined".
    """
    if not (0.0 < prior_prob < 1.0):
        raise ValueError(f"prior_prob must be in (0, 1) (got {prior_prob})")
    if not (0.0 < posterior_prob < 1.0):
        raise ValueError(f"posterior_prob must be in (0, 1) (got {posterior_prob})")
    prior_odds = prior_prob / (1.0 - prior_prob)
    post_odds = posterior_prob / (1.0 - posterior_prob)
    return float(post_odds / prior_odds)


def bayes_from_likelihoods(
    prior_prob: float,
    p_e_given_h: float,
    p_e_given_not_h: float,
) -> float:
    if not (0.0 < prior_prob < 1.0):
        raise ValueError(f"prior_prob must be in (0, 1) (got {prior_prob})")
    if not (0.0 <= p_e_given_h <= 1.0):
        raise ValueError(f"p_e_given_h must be in [0, 1] (got {p_e_given_h})")
    if not (0.0 <= p_e_given_not_h <= 1.0):
        raise ValueError(f"p_e_given_not_h must be in [0, 1] (got {p_e_given_not_h})")
    if p_e_given_h == 0.0 and p_e_given_not_h == 0.0:
        raise ValueError("p_e_given_h and p_e_given_not_h cannot both be zero")
    num = p_e_given_h * prior_prob
    denom = num + p_e_given_not_h * (1.0 - prior_prob)
    return float(num / denom)
