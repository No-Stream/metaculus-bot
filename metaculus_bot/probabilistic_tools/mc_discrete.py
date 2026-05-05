from __future__ import annotations

import logging
from dataclasses import dataclass

from scipy import stats

from metaculus_bot.numeric_config import STANDARD_PERCENTILES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DirichletCI:
    mean: float
    ci_80_low: float
    ci_80_high: float


def dirichlet_with_other(
    option_probs: dict[str, float],
    other_mass: float | None,
    concentration: float = 10.0,
) -> dict[str, DirichletCI]:
    if not option_probs:
        raise ValueError("option_probs must be non-empty")
    if concentration <= 0:
        raise ValueError(f"concentration must be > 0 (got {concentration})")

    total = 0.0
    for k, v in option_probs.items():
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"option_probs[{k!r}]={v} not in [0,1]")
        total += v

    if other_mass is not None:
        if not (0.0 <= other_mass <= 1.0):
            raise ValueError(f"other_mass must be in [0,1] (got {other_mass})")
        total += other_mass

    if abs(total - 1.0) > 0.02:
        raise ValueError(f"option_probs (+ other_mass if present) must sum to 1.0 ± 0.02 (got {total})")

    # Build the alpha vector for the full Dirichlet with the means as given.
    means_full = dict(option_probs)
    if other_mass is not None:
        means_full["__OTHER__"] = other_mass

    alphas = {k: concentration * m for k, m in means_full.items()}
    alpha_total = sum(alphas.values())

    result: dict[str, DirichletCI] = {}
    for key, alpha_k in alphas.items():
        if alpha_k <= 0:
            raise ValueError(
                f"dirichlet_with_other: computed alpha_k={alpha_k} for key={key!r}; "
                "all option means (and other_mass if set) must be > 0 to produce a valid Dirichlet"
            )
        beta_k = alpha_total - alpha_k
        mean = float(alpha_k / alpha_total)
        lo = float(stats.beta.ppf(0.10, alpha_k, beta_k))
        hi = float(stats.beta.ppf(0.90, alpha_k, beta_k))
        public_key = "Other" if key == "__OTHER__" else key
        result[public_key] = DirichletCI(mean=mean, ci_80_low=lo, ci_80_high=hi)
    return result


def _canonical_percentiles_from_rv(rv) -> dict[float, float]:
    return {float(q): float(rv.ppf(q)) for q in STANDARD_PERCENTILES}


def negative_binomial_percentiles(mean: float, overdispersion_factor: float) -> dict[float, float]:
    """Return the 11 canonical percentiles of a Negative Binomial(mean, phi).

    Special case: when ``mean == 0`` the distribution collapses to a point mass
    at 0, so every canonical percentile is returned as 0.0.
    """
    if mean < 0:
        raise ValueError(f"mean must be >= 0 (got {mean})")
    if overdispersion_factor <= 1.0:
        raise ValueError(
            f"overdispersion_factor must be > 1 (got {overdispersion_factor}); "
            "use poisson_at_least_one for the Poisson limit"
        )
    if mean == 0:
        return {float(q): 0.0 for q in STANDARD_PERCENTILES}
    # scipy.stats.nbinom uses (n, p) where variance = mean / p = mean * phi ⇒ p = 1/phi
    p = 1.0 / overdispersion_factor
    r = mean / (overdispersion_factor - 1.0)
    rv = stats.nbinom(r, p)
    return _canonical_percentiles_from_rv(rv)


def poisson_percentiles(mean: float) -> dict[float, float]:
    if mean < 0:
        raise ValueError(f"mean must be >= 0 (got {mean})")
    rv = stats.poisson(mean)
    return _canonical_percentiles_from_rv(rv)


def beta_binomial_ceiling_percentiles(
    mean: float,
    ceiling: int,
    concentration: float = 10.0,
) -> dict[float, float]:
    if ceiling < 1:
        raise ValueError(f"ceiling must be >= 1 (got {ceiling})")
    if not (0.0 <= mean <= ceiling):
        raise ValueError(f"mean must be in [0, ceiling={ceiling}] (got {mean})")
    if concentration <= 0:
        raise ValueError(f"concentration must be > 0 (got {concentration})")
    p = mean / ceiling
    # Avoid degenerate alpha/beta=0 at edges
    p_eff = min(max(p, 1e-9), 1.0 - 1e-9)
    alpha = concentration * p_eff
    beta_ = concentration * (1.0 - p_eff)
    rv = stats.betabinom(ceiling, alpha, beta_)
    return _canonical_percentiles_from_rv(rv)


__all__ = [
    "DirichletCI",
    "dirichlet_with_other",
    "negative_binomial_percentiles",
    "poisson_percentiles",
    "beta_binomial_ceiling_percentiles",
]
