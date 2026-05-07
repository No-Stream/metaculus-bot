from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SurvivalResult:
    unconditional_prob: float
    conditional_prob_given_no_event_yet: float


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
