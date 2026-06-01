"""Mixture-of-normals distributions for numeric forecasting.

Each scenario the LLM reasons about — 'underperform / baseline / breakout' —
is a normal component with (weight, mean, sd). The mixture CDF is the weighted
sum of component CDFs; sampling percentiles is numeric (grid-based).

Used by:
- tool_runner._run_numeric_tools for the "Mixture-of-normals" markdown block.
- Workstream E's numeric format router for building 201-point Metaculus CDFs
  from LLM-emitted mixture blocks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion
from scipy import optimize
from scipy.stats import norm

from metaculus_bot.numeric.config import MIN_CDF_PROB_STEP, PCHIP_CDF_POINTS
from metaculus_bot.numeric.pchip_cdf import enforce_min_steps, safe_cdf_bounds


@dataclass(frozen=True)
class MixtureComponent:
    weight: float
    mean: float
    sd: float


@dataclass(frozen=True)
class MixtureOfNormals:
    """Immutable mixture of normals. Weights are normalized in __post_init__.

    After construction, sum(components[i].weight for all i) == 1.0 within
    floating-point precision. Raises ValueError if any weight < 0, any sd <= 0,
    no components, or total weight <= 0.
    """

    components: tuple[MixtureComponent, ...]

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("MixtureOfNormals requires at least one component")
        for i, c in enumerate(self.components):
            if c.weight < 0:
                raise ValueError(f"component {i} has negative weight {c.weight}")
            if c.sd <= 0:
                raise ValueError(f"component {i} has non-positive sd {c.sd}")
        total = sum(c.weight for c in self.components)
        if total <= 0:
            raise ValueError(f"total weight must be > 0, got {total}")
        normalized = tuple(MixtureComponent(weight=c.weight / total, mean=c.mean, sd=c.sd) for c in self.components)
        object.__setattr__(self, "components", normalized)


def mixture_cdf(mix: MixtureOfNormals, grid: np.ndarray) -> np.ndarray:
    """Analytic mixture CDF: sum_i weight_i * Phi((x - mean_i) / sd_i).

    grid must be 1-D, non-empty, finite. Returns array of the same shape, each
    entry in [0, 1], non-decreasing along the grid.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError(f"grid must be 1-D, got shape {grid.shape}")
    if grid.size == 0:
        raise ValueError("grid must be non-empty")
    if not np.all(np.isfinite(grid)):
        raise ValueError("grid must contain only finite values")

    result = np.zeros_like(grid, dtype=float)
    for c in mix.components:
        result += c.weight * norm.cdf(grid, loc=c.mean, scale=c.sd)
    # Clip to [0, 1] to guard against float drift.
    return np.clip(result, 0.0, 1.0)


_MIN_FIT_SD: float = 1e-9


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    e = np.exp(shifted)
    return e / e.sum()


def _initial_params_from_percentiles(
    keys: np.ndarray, values: np.ndarray, n_components: int, rng: np.random.Generator
) -> np.ndarray:
    """Initial parameters: equal weights, means spread across the declared range,
    sds scaled to roughly a third of the range."""
    lo = float(values.min())
    hi = float(values.max())
    spread = max(hi - lo, 1.0)
    if n_components == 1:
        means = np.array([float(np.median(values))])
    else:
        # Spread means uniformly across the declared value range; small jitter for seed determinism.
        means = np.linspace(lo, hi, n_components) + rng.normal(scale=spread * 0.01, size=n_components)
    sigma_init = max(spread / max(3.0 * n_components, 1.0), 1e-6)
    log_sds = np.full(n_components, float(np.log(sigma_init)))
    logits = np.zeros(n_components)
    # Parameter vector: [logits (n), means (n), log_sds (n)] — total 3n.
    return np.concatenate([logits, means, log_sds])


def _unpack(theta: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = theta[:n_components]
    means = theta[n_components : 2 * n_components]
    log_sds = theta[2 * n_components : 3 * n_components]
    weights = _softmax(logits)
    sds = np.exp(log_sds)
    return weights, means, sds


def _mixture_cdf_values(weights: np.ndarray, means: np.ndarray, sds: np.ndarray, values: np.ndarray) -> np.ndarray:
    # Shape: (n_components, n_values)
    per_component = weights[:, None] * norm.cdf(values[None, :], loc=means[:, None], scale=sds[:, None])
    return per_component.sum(axis=0)


def _fallback_single_normal(
    values: np.ndarray, keys: np.ndarray, percentiles: Mapping[float, float]
) -> MixtureOfNormals:
    """Fallback from failed optimization: single-normal best guess."""
    # Prefer median + IQR / 1.349
    median_val: float | None = None
    p25_val: float | None = None
    p75_val: float | None = None
    for p, v in percentiles.items():
        if abs(p - 0.5) < 1e-9:
            median_val = float(v)
        if abs(p - 0.25) < 1e-9:
            p25_val = float(v)
        if abs(p - 0.75) < 1e-9:
            p75_val = float(v)
    if median_val is not None and p25_val is not None and p75_val is not None:
        iqr = p75_val - p25_val
        if iqr > 0:
            sd = iqr / 1.349
            return MixtureOfNormals((MixtureComponent(weight=1.0, mean=median_val, sd=sd),))
    # Secondary fallback: mean + std of values (ddof=0), with tiny-positive floor.
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std <= 0:
        std = max(abs(mean) * 0.1, 1e-6)
    return MixtureOfNormals((MixtureComponent(weight=1.0, mean=mean, sd=std),))


_MULTISTART_SEED_COUNT: int = 5


def _fit_mixture_one_seed(
    keys: np.ndarray,
    values: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[MixtureOfNormals | None, float]:
    """Run a single-seed L-BFGS-B fit and return (mixture, sse).

    Returns ``(None, +inf)`` when the optimizer fails to converge or yields
    a sub-threshold sd. The caller picks the best across multiple seeds.
    """
    rng = np.random.default_rng(seed)

    def objective(theta: np.ndarray) -> float:
        weights, means, sds = _unpack(theta, n_components)
        sds_safe = np.clip(sds, _MIN_FIT_SD, None)
        cdf_vals = _mixture_cdf_values(weights, means, sds_safe, values)
        return float(np.sum((cdf_vals - keys) ** 2))

    x0 = _initial_params_from_percentiles(keys, values, n_components, rng)
    result = optimize.minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-8},
    )
    if not result.success:
        return None, float("inf")

    weights, means, sds = _unpack(result.x, n_components)
    if np.any(sds < _MIN_FIT_SD):
        return None, float("inf")

    comps = tuple(MixtureComponent(weight=float(w), mean=float(m), sd=float(s)) for w, m, s in zip(weights, means, sds))
    return MixtureOfNormals(comps), float(result.fun)


def fit_mixture_from_percentiles(
    percentiles: Mapping[float, float],
    *,
    n_components: int = 3,
    seed: int = 0,
) -> MixtureOfNormals:
    """Fit a mixture to declared percentiles via multi-start constrained LSQ.

    Optimization over (softmax-logits, means, log-sds). Minimize SSE between
    mixture CDF at percentile values and the probability levels. Runs the
    optimizer with ``seed, seed+1, ..., seed+4`` initial conditions and keeps
    the result with the lowest SSE — bimodal/multi-modal generators are
    sensitive to where L-BFGS-B starts, and a single seed sometimes lands
    in a poor local minimum.

    Fallback: if every seed's run fails to converge or yields any sd <
    ``_MIN_FIT_SD``, return a single-normal MixtureOfNormals with
    mean=median, sd=IQR/1.349. If that also fails (no valid p=0.5, p=0.25,
    p=0.75 in keys, or IQR <= 0), use mean=mean-of-values, sd=std-of-values
    with ddof=0.

    Requires 2 <= n_components <= 4. Raises ValueError on out-of-range.
    Deterministic given the same inputs + seed.
    """
    if not (2 <= n_components <= 4):
        raise ValueError(f"n_components must be in [2, 4], got {n_components}")
    if not percentiles:
        raise ValueError("percentiles must be non-empty")

    sorted_items = sorted(percentiles.items())
    keys = np.array([p for p, _ in sorted_items], dtype=float)
    values = np.array([v for _, v in sorted_items], dtype=float)

    # Degenerate: all values equal → fallback.
    if float(values.max() - values.min()) <= 0:
        return _fallback_single_normal(values, keys, percentiles)

    best_mix: MixtureOfNormals | None = None
    best_sse: float = float("inf")
    for offset in range(_MULTISTART_SEED_COUNT):
        candidate, sse = _fit_mixture_one_seed(keys, values, n_components, seed + offset)
        if candidate is not None and sse < best_sse:
            best_mix = candidate
            best_sse = sse

    if best_mix is None:
        return _fallback_single_normal(values, keys, percentiles)
    return best_mix


def percentiles_to_metaculus_cdf_via_mixture(
    mix: MixtureOfNormals,
    question: NumericQuestion,
    *,
    num_points: int = PCHIP_CDF_POINTS,
) -> list[Percentile]:
    """Build a Metaculus CDF from a mixture, constraint-enforced.

    1. Build a uniform grid of ``num_points`` on [question.lower_bound,
       question.upper_bound].
    2. Evaluate mixture_cdf on the grid.
    3. Clip to bound conditions: open -> [0.001, 0.999]; closed -> [0.0, 1.0]
       with endpoint-pinning.
    4. Enforce strictly non-decreasing + min-step via pchip_cdf helpers.
    5. Return list[Percentile] matching forecasting_tools convention:
       [Percentile(value=x_i, percentile=F(x_i)), ...]
    """
    lower = float(question.lower_bound)
    upper = float(question.upper_bound)
    if upper <= lower:
        raise ValueError(f"upper_bound {upper} must be > lower_bound {lower}")

    open_lower = bool(question.open_lower_bound)
    open_upper = bool(question.open_upper_bound)

    grid = np.linspace(lower, upper, num_points)
    cdf = mixture_cdf(mix, grid)

    # Pre-clip to open-bound limits so safe_cdf_bounds's trailing
    # ``np.maximum.accumulate`` cannot push the upper endpoint back above 0.999
    # or the lower endpoint below 0.001 via neighbor values.
    hi_cap = 0.999 if open_upper else 1.0
    lo_cap = 0.001 if open_lower else 0.0
    cdf = np.clip(cdf, lo_cap, hi_cap)

    # Pin endpoints for closed bounds before constraint enforcement.
    if not open_lower:
        cdf[0] = 0.0
    if not open_upper:
        cdf[-1] = 1.0

    # Ensure monotonic accumulation.
    cdf = np.maximum.accumulate(cdf)

    # Enforce min-step iteratively, with backward pass to handle CDFs that
    # saturate before the grid endpoint (forward+backward sweep).
    cdf = enforce_min_steps(cdf, MIN_CDF_PROB_STEP, upper_cap=hi_cap, lower_cap=lo_cap)

    # Final monotonicity safety + apply boundary constraints + max-step redistribution.
    cdf = np.maximum.accumulate(cdf)
    cdf = safe_cdf_bounds(cdf, open_lower, open_upper)

    return [Percentile(value=float(grid[i]), percentile=float(cdf[i])) for i in range(num_points)]
