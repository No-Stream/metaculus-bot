from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy import optimize, stats

from metaculus_bot.numeric_config import STANDARD_PERCENTILES
from metaculus_bot.pchip_cdf import generate_pchip_cdf

CdfFn = Callable[[np.ndarray, float, float], np.ndarray]


@dataclass(frozen=True)
class NormalFit:
    mu: float
    sigma: float
    method: str


@dataclass(frozen=True)
class LognormalFit:
    mu: float
    sigma: float
    method: str


@dataclass(frozen=True)
class StudentTFit:
    loc: float
    scale: float
    df: float
    method: str


@dataclass(frozen=True)
class TailMassResult:
    prob_below_min: float
    prob_above_max: float
    interior_mass: float


FitType = Union[NormalFit, LognormalFit, StudentTFit]


def _validate_percentile_dict(percentile_values: dict[float, float]) -> list[tuple[float, float]]:
    if not percentile_values:
        raise ValueError("percentile_values must be non-empty")
    if len(percentile_values) < 2:
        raise ValueError(f"need at least 2 percentile points (got {len(percentile_values)})")
    items: list[tuple[float, float]] = []
    for p, v in percentile_values.items():
        p_f = float(p)
        v_f = float(v)
        if not (0.0 < p_f < 1.0):
            raise ValueError(f"percentile keys must be in (0, 1) (got {p_f})")
        if not math.isfinite(v_f):
            raise ValueError(f"percentile value must be finite (got {v_f})")
        items.append((p_f, v_f))
    items.sort(key=lambda x: x[0])
    return items


# Z-gap between P10 and P90 in a standard normal, for the sigma initial guess.
_P10_P90_Z_GAP: float = 2.5631


def _fit_cdf_lsq(
    cdf_fn: CdfFn,
    items: list[tuple[float, float]],
    initial_loc: float,
    initial_scale: float,
    label: str,
) -> tuple[float, float]:
    """Least-squares fit of a location-scale distribution via Nelder-Mead on
    the CDF-difference objective. Shared scaffolding for the normal and
    Student-t fits — only ``cdf_fn`` (and the initial guess) differ.

    ``cdf_fn(values, loc, scale) -> cdf_at_values``. ``label`` is used in the
    error message on non-convergence (and to disambiguate the Student-t df).
    Raises ``ValueError`` if Nelder-Mead does not converge or the recovered
    scale is non-positive.
    """
    probs = np.array([p for p, _ in items])
    vals = np.array([v for _, v in items])

    def objective(theta: np.ndarray) -> float:
        loc, log_scale = theta
        scale = math.exp(log_scale)
        cdf_vals = cdf_fn(vals, loc, scale)
        return float(np.sum((cdf_vals - probs) ** 2))

    x0 = np.array([initial_loc, math.log(max(initial_scale, 1e-9))])
    result = optimize.minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000},
    )
    if not result.success:
        raise ValueError(
            f"Nelder-Mead {label} fit did not converge (msg={getattr(result, 'message', '?')!r}); "
            f"initial guess loc={initial_loc:.6g}, scale={initial_scale:.6g} over {len(items)} percentiles"
        )
    loc, log_scale = result.x
    scale = math.exp(log_scale)
    if scale <= 0:
        raise ValueError(f"fitted scale must be > 0 (got {scale}) for {label} fit")
    return float(loc), float(scale)


def _fit_normal_lsq(items: list[tuple[float, float]], initial_mu: float, initial_sigma: float) -> tuple[float, float]:
    return _fit_cdf_lsq(
        lambda v, mu, s: stats.norm.cdf(v, loc=mu, scale=s),
        items,
        initial_mu,
        initial_sigma,
        "normal",
    )


def _initial_normal_guess(items: list[tuple[float, float]]) -> tuple[float, float]:
    probs = np.array([p for p, _ in items])
    vals = np.array([v for _, v in items])
    mu_guess = float(np.interp(0.5, probs, vals))
    # Prefer P10/P90 directly when available: sigma ≈ (P90 - P10) / z_gap.
    probs_set = {float(p) for p, _ in items}
    if 0.1 in probs_set and 0.9 in probs_set:
        p10 = next(v for p, v in items if p == 0.1)
        p90 = next(v for p, v in items if p == 0.9)
        sigma_guess = max((p90 - p10) / _P10_P90_Z_GAP, 1e-6)
        return mu_guess, sigma_guess
    # Fallback: interpolate at ±1σ under the normal (0.1587 / 0.8413).
    lo = float(np.interp(0.1587, probs, vals))
    hi = float(np.interp(0.8413, probs, vals))
    sigma_guess = max((hi - lo) / 2.0, 1e-6)
    return mu_guess, sigma_guess


def fit_normal_from_percentiles(percentile_values: dict[float, float]) -> NormalFit:
    items = _validate_percentile_dict(percentile_values)
    mu0, sigma0 = _initial_normal_guess(items)
    mu, sigma = _fit_normal_lsq(items, mu0, sigma0)
    if sigma <= 0:
        raise ValueError(f"fitted sigma must be > 0 (got {sigma})")
    return NormalFit(mu=mu, sigma=sigma, method="least_squares_cdf")


def fit_lognormal_from_percentiles(percentile_values: dict[float, float]) -> LognormalFit:
    items = _validate_percentile_dict(percentile_values)
    for _, v in items:
        if v <= 0:
            raise ValueError(f"all percentile values must be > 0 for lognormal (got {v})")
    log_items = [(p, math.log(v)) for p, v in items]
    mu0, sigma0 = _initial_normal_guess(log_items)
    mu, sigma = _fit_normal_lsq(log_items, mu0, sigma0)
    if sigma <= 0:
        raise ValueError(f"fitted sigma must be > 0 (got {sigma})")
    return LognormalFit(mu=mu, sigma=sigma, method="least_squares_cdf_logspace")


def fit_student_t_from_percentiles(percentile_values: dict[float, float], df: float = 5.0) -> StudentTFit:
    if df <= 0:
        raise ValueError(f"df must be > 0 (got {df})")
    items = _validate_percentile_dict(percentile_values)
    loc0, scale0 = _initial_normal_guess(items)

    loc, scale = _fit_cdf_lsq(
        lambda v, loc, s: stats.t.cdf(v, df=df, loc=loc, scale=s),
        items,
        loc0,
        scale0,
        f"student_t(df={df:.3g})",
    )
    return StudentTFit(loc=loc, scale=scale, df=float(df), method="least_squares_cdf")


def eval_cdf(fit: FitType, x: float) -> float:
    if isinstance(fit, NormalFit):
        return float(stats.norm.cdf(x, loc=fit.mu, scale=fit.sigma))
    if isinstance(fit, LognormalFit):
        if x <= 0:
            return 0.0
        return float(stats.norm.cdf(math.log(x), loc=fit.mu, scale=fit.sigma))
    if isinstance(fit, StudentTFit):
        return float(stats.t.cdf(x, df=fit.df, loc=fit.loc, scale=fit.scale))
    raise ValueError(f"unknown fit type: {type(fit).__name__}")


def _eval_ppf(fit: FitType, q: float) -> float:
    if isinstance(fit, NormalFit):
        return float(stats.norm.ppf(q, loc=fit.mu, scale=fit.sigma))
    if isinstance(fit, LognormalFit):
        return float(math.exp(stats.norm.ppf(q, loc=fit.mu, scale=fit.sigma)))
    if isinstance(fit, StudentTFit):
        return float(stats.t.ppf(q, df=fit.df, loc=fit.loc, scale=fit.scale))
    raise ValueError(f"unknown fit type: {type(fit).__name__}")


def out_of_bounds_mass(
    fit: FitType,
    lower_bound: float | None,
    upper_bound: float | None,
) -> TailMassResult:
    if lower_bound is not None and upper_bound is not None and not (lower_bound < upper_bound):
        raise ValueError(f"lower_bound {lower_bound} must be < upper_bound {upper_bound}")

    if lower_bound is None:
        prob_below = 0.0
    else:
        prob_below = eval_cdf(fit, lower_bound)

    if upper_bound is None:
        prob_above = 0.0
    else:
        prob_above = 1.0 - eval_cdf(fit, upper_bound)

    interior = max(0.0, 1.0 - prob_below - prob_above)
    return TailMassResult(
        prob_below_min=float(prob_below),
        prob_above_max=float(prob_above),
        interior_mass=float(interior),
    )


def cdf_at_threshold(fit: FitType, threshold: float) -> float:
    return eval_cdf(fit, threshold)


def fit_to_11_percentiles(fit: FitType) -> dict[float, float]:
    return {float(q): float(_eval_ppf(fit, q)) for q in STANDARD_PERCENTILES}


def percentiles_to_metaculus_cdf(
    percentile_values: dict[float, float],
    lower_bound: float,
    upper_bound: float,
    open_lower: bool,
    open_upper: bool,
    zero_point: float | None = None,
) -> list[float]:
    if not percentile_values:
        raise ValueError("percentile_values must be non-empty")
    if upper_bound <= lower_bound:
        raise ValueError(f"upper_bound {upper_bound} must be > lower_bound {lower_bound}")

    converted: dict[float, float] = {}
    for p, v in percentile_values.items():
        p_f = float(p)
        if not (0.0 < p_f < 1.0):
            raise ValueError(f"percentile keys must be in (0, 1) (got {p_f})")
        converted[p_f * 100.0] = float(v)

    cdf, _ = generate_pchip_cdf(
        percentile_values=converted,
        open_upper_bound=open_upper,
        open_lower_bound=open_lower,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        zero_point=zero_point,
    )
    return cdf
