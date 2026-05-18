from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SurvivalResult:
    unconditional_prob: float
    conditional_prob_given_no_event_yet: float


@dataclass(frozen=True)
class GammaFit:
    shape: float
    scale: float
    mean: float
    variance: float


def prob_event_before(
    hazard_rate: float,
    elapsed_fraction: float,
    remaining_fraction: float,
    window_length: float = 1.0,
) -> SurvivalResult:
    """Constant-hazard survival probability over a forecast window.

    ``hazard_rate`` is a rate per unit time (e.g., 0.25 events/day).
    ``window_length`` is the full window length in the SAME units as
    ``hazard_rate`` (e.g., 30 when the rate is per-day and the window is 30
    days). Units cancel.

    ``elapsed_fraction`` and ``remaining_fraction`` are the fractions of
    ``window_length`` that have already passed and that remain at forecast
    time, respectively. They should sum to ~1.0.

    Returns unconditional P(event in full window) and conditional P(event
    in remaining | none yet).
    """
    if hazard_rate < 0:
        raise ValueError(f"hazard_rate must be >= 0 (got {hazard_rate})")
    if not (0.0 <= elapsed_fraction <= 1.0):
        raise ValueError(f"elapsed_fraction must be in [0, 1] (got {elapsed_fraction})")
    if not (0.0 <= remaining_fraction <= 1.0):
        raise ValueError(f"remaining_fraction must be in [0, 1] (got {remaining_fraction})")
    if window_length <= 0:
        raise ValueError(f"window_length must be > 0 (got {window_length})")

    lam_scaled = hazard_rate * window_length
    total_frac = elapsed_fraction + remaining_fraction
    unconditional_prob = 1.0 - math.exp(-lam_scaled * total_frac)
    conditional_prob = 1.0 - math.exp(-lam_scaled * remaining_fraction)
    return SurvivalResult(
        unconditional_prob=float(unconditional_prob),
        conditional_prob_given_no_event_yet=float(conditional_prob),
    )


def weibull_prob_event_before(scale: float, shape: float, t: float) -> float:
    if scale <= 0:
        raise ValueError(f"scale must be > 0 (got {scale})")
    if shape <= 0:
        raise ValueError(f"shape must be > 0 (got {shape})")
    if t < 0:
        raise ValueError(f"t must be >= 0 (got {t})")
    return float(1.0 - math.exp(-((t / scale) ** shape)))


def poisson_at_least_one(lambda_t: float) -> float:
    if lambda_t < 0:
        raise ValueError(f"lambda_t must be >= 0 (got {lambda_t})")
    return float(1.0 - math.exp(-lambda_t))


def base_rate_to_hazard(
    k: int,
    n: int,
    years_per_ref_period: float = 1.0,
    prior_rate: float | None = None,
    prior_strength: float = 1.0,
) -> float:
    """Posterior-mean hazard rate (events per year) from observed k/n.

    Uses a Gamma conjugate prior on the rate λ:
      - When ``prior_rate`` is None: Jeffreys-ish Gamma(0.5, 0) — improper
        but with well-defined posterior mean (0.5 + k) / (n·years) whenever
        data are present.
      - When ``prior_rate`` is provided: Gamma(α, β) with
        ``α = prior_strength * prior_rate`` and ``β = prior_strength``, so
        the prior mean is ``prior_rate`` with strength ``prior_strength``
        pseudo-exposure-units (in the same units as ``n * years_per_ref_period``).

    Posterior mean is ``(α + k) / (β + n·years_per_ref_period)``.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0 (got {k})")
    if n < 1:
        raise ValueError(f"n must be >= 1 (got {n})")
    if years_per_ref_period <= 0:
        raise ValueError(f"years_per_ref_period must be > 0 (got {years_per_ref_period})")
    if prior_rate is not None and prior_rate < 0:
        raise ValueError(f"prior_rate must be >= 0 when provided (got {prior_rate})")
    if prior_strength <= 0:
        raise ValueError(f"prior_strength must be > 0 (got {prior_strength})")

    exposure = n * years_per_ref_period
    if prior_rate is None:
        alpha_prior = 0.5
        beta_prior = 0.0
    else:
        alpha_prior = prior_strength * prior_rate
        beta_prior = prior_strength
    return float((alpha_prior + k) / (beta_prior + exposure))


def fit_gamma_from_gaps(
    observed_gaps: Sequence[float],
    *,
    method: Literal["mom", "mle"] = "mom",
) -> GammaFit:
    """Fit a Gamma distribution to historical inter-arrival gaps.

    ``method='mom'`` (default) uses moment matching: ``k = mean**2 / variance``
    and ``θ = variance / mean`` against the population variance (``ddof=0``).
    ``method='mle'`` uses ``scipy.stats.gamma.fit(observed_gaps, floc=0)``.

    Requires at least two observations with a positive mean AND positive
    variance. A constant sequence raises ``ValueError`` because the MoM system
    is singular and MLE's shape estimate diverges.
    """
    if len(observed_gaps) < 2:
        raise ValueError(f"observed_gaps must have at least 2 entries (got {len(observed_gaps)})")
    gaps = np.asarray(observed_gaps, dtype=float)
    if np.any(gaps <= 0):
        raise ValueError("observed_gaps must all be > 0")
    mean = float(np.mean(gaps))
    variance = float(np.var(gaps, ddof=0))
    if mean <= 0:
        raise ValueError(f"mean of observed_gaps must be > 0 (got {mean})")
    if variance <= 0:
        raise ValueError(f"variance of observed_gaps must be > 0 (got {variance})")

    if method == "mom":
        shape = mean * mean / variance
        scale = variance / mean
    elif method == "mle":
        shape, _loc, scale = stats.gamma.fit(gaps, floc=0)
    else:
        raise ValueError(f"method must be 'mom' or 'mle' (got {method!r})")

    return GammaFit(
        shape=float(shape),
        scale=float(scale),
        mean=float(shape * scale),
        variance=float(shape * scale * scale),
    )


def gamma_prob_event_before(
    fit: GammaFit,
    *,
    elapsed: float,
    remaining: float,
) -> SurvivalResult:
    """P(next event by ``t = elapsed + remaining``) under Gamma(k, θ).

    Returns both the unconditional CDF at ``t`` and the conditional
    ``(F(t) - F(elapsed)) / (1 - F(elapsed))`` given no event has occurred in
    ``[0, elapsed]``. When ``elapsed <= 0`` the conditional equals the
    unconditional. When ``1 - F(elapsed)`` underflows to 0 (the event has
    almost surely already happened), the conditional is clamped to 1.0.
    """
    if elapsed < 0:
        raise ValueError(f"elapsed must be >= 0 (got {elapsed})")
    if remaining <= 0:
        raise ValueError(f"remaining must be > 0 (got {remaining})")

    total = elapsed + remaining
    unconditional = float(stats.gamma.cdf(total, a=fit.shape, scale=fit.scale))
    if elapsed <= 0:
        conditional = unconditional
    else:
        f_elapsed = float(stats.gamma.cdf(elapsed, a=fit.shape, scale=fit.scale))
        survival_elapsed = 1.0 - f_elapsed
        if survival_elapsed <= 0.0:
            conditional = 1.0
        else:
            # Clamp to [0, 1]: deep-tail FP error in scipy.gamma.cdf can let
            # the raw division leak just outside the unit interval.
            conditional = max(0.0, min(1.0, (unconditional - f_elapsed) / survival_elapsed))
    return SurvivalResult(
        unconditional_prob=max(0.0, min(1.0, unconditional)),
        conditional_prob_given_no_event_yet=float(conditional),
    )


def weibull_prob_event_before_conditional(
    scale: float,
    shape: float,
    *,
    elapsed: float,
    remaining: float,
) -> SurvivalResult:
    """Weibull CDF over the full window plus conditional-on-survival split.

    ``F(t) = 1 - exp(-(t/scale)**shape)``. Semantics match
    ``gamma_prob_event_before``: unconditional CDF at ``t = elapsed + remaining``
    and conditional ``(F(t) - F(elapsed)) / (1 - F(elapsed))``.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0 (got {scale})")
    if shape <= 0:
        raise ValueError(f"shape must be > 0 (got {shape})")
    if elapsed < 0:
        raise ValueError(f"elapsed must be >= 0 (got {elapsed})")
    if remaining <= 0:
        raise ValueError(f"remaining must be > 0 (got {remaining})")

    total = elapsed + remaining
    unconditional = 1.0 - math.exp(-((total / scale) ** shape))
    if elapsed <= 0:
        conditional = unconditional
    else:
        f_elapsed = 1.0 - math.exp(-((elapsed / scale) ** shape))
        survival_elapsed = 1.0 - f_elapsed
        if survival_elapsed <= 0.0:
            conditional = 1.0
        else:
            # Clamp to [0, 1]: deep-tail FP error in the survival math can let
            # the raw division leak just outside the unit interval.
            conditional = max(0.0, min(1.0, (unconditional - f_elapsed) / survival_elapsed))
    return SurvivalResult(
        unconditional_prob=max(0.0, min(1.0, float(unconditional))),
        conditional_prob_given_no_event_yet=float(conditional),
    )
