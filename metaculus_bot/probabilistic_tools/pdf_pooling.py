"""Offline numeric-CDF aggregation primitives: Vincentization, tail-floor, log-pool.

Production numeric aggregation (``metaculus_bot/ablation/run_pdf.py``,
``metaculus_bot/numeric/utils.py``) averages each forecaster's CDF *vertically* — i.e.
it averages F(x) at every shared grid value. On a disagreeing (bimodal) ensemble that
smears two sharp distributions into one wide low-information CDF, which over-disperses the
tails and bleeds natural-log score. These functions are the rigorous alternatives a later
offline harness scores against median; none of them is wired into the live path.

All three operate on per-forecaster 201-point CDFs represented as ``list[Percentile]``
(``.value`` = x on a shared ``np.linspace(lower, upper, 201)`` grid, ``.percentile`` =
cumulative probability F(x) in [0, 1]) and emit a constraint-enforced ``list[Percentile]``
on the question's own grid. Constraint enforcement reuses the same
``safe_cdf_bounds`` / ``enforce_min_steps`` layer that
``percentiles_to_metaculus_cdf_via_mixture`` uses, so outputs satisfy the Metaculus
CDF rules (201 points, min-step, open/closed bounds, monotone).

- ``vincentize_cdfs``  — quantile averaging (preserves sharpness; the headline primitive).
- ``apply_tail_floor`` — guarantee >= floor_eps PMF mass in boundary buckets (anti-saturation).
- ``log_pool_cdfs``    — geometric (log) pooling of densities (a third aggregation option).
"""

from __future__ import annotations

import numpy as np
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric.config import MIN_CDF_PROB_STEP, PCHIP_CDF_POINTS
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid, enforce_min_steps, safe_cdf_bounds

# Probability grid for inverting a CDF to a quantile function. Dense enough that linear
# interpolation back onto the value grid does not itself introduce visible bias.
_QUANTILE_GRID_POINTS: int = 401


def _question_grid(question: NumericQuestion, num_points: int) -> np.ndarray:
    """Value grid for projecting a pooled CDF back onto the question's x-axis.

    Mirrors the production CDF grid (``pchip_cdf.build_cdf_value_grid``): linear for
    linear-scaled questions, geometric when ``question.zero_point`` is set (log-scaled).
    The geometric branch is load-bearing for zero_point questions — the Metaculus scorer
    buckets resolutions on the geometric grid, so a pooled CDF projected onto a linear grid
    would be scored against the wrong buckets.
    """
    lower = float(question.lower_bound)
    upper = float(question.upper_bound)
    if upper <= lower:
        raise ValueError(f"upper_bound {upper} must be > lower_bound {lower}")
    zero_point = float(question.zero_point) if question.zero_point is not None else None
    return build_cdf_value_grid(lower, upper, zero_point, num_points)


def _cdf_probs(cdf: list[Percentile]) -> np.ndarray:
    """Extract the cumulative-probability array from a forecaster CDF, sorted by value."""
    pairs = sorted(((float(p.value), float(p.percentile)) for p in cdf), key=lambda t: t[0])
    return np.array([p for _, p in pairs], dtype=float)


def _cdf_values(cdf: list[Percentile]) -> np.ndarray:
    pairs = sorted(((float(p.value), float(p.percentile)) for p in cdf), key=lambda t: t[0])
    return np.array([v for v, _ in pairs], dtype=float)


def _finalize_cdf(
    cdf: np.ndarray,
    grid: np.ndarray,
    question: NumericQuestion,
    *,
    min_step: float = MIN_CDF_PROB_STEP,
) -> list[Percentile]:
    """Apply the shared Metaculus constraint layer and emit list[Percentile] on ``grid``.

    Mirrors the closing steps of ``percentiles_to_metaculus_cdf_via_mixture``: pre-clip to
    open-bound caps, pin closed endpoints, accumulate, enforce min-step (forward+backward
    sweep), then ``safe_cdf_bounds`` for the max-step redistribution + final bounds.
    """
    open_lower = bool(question.open_lower_bound)
    open_upper = bool(question.open_upper_bound)

    hi_cap = 0.999 if open_upper else 1.0
    lo_cap = 0.001 if open_lower else 0.0

    cdf = np.clip(np.asarray(cdf, dtype=float), lo_cap, hi_cap)
    if not open_lower:
        cdf[0] = 0.0
    if not open_upper:
        cdf[-1] = 1.0

    cdf = np.maximum.accumulate(cdf)
    cdf = enforce_min_steps(cdf, min_step, upper_cap=hi_cap, lower_cap=lo_cap)
    cdf = np.maximum.accumulate(cdf)
    cdf = safe_cdf_bounds(cdf, open_lower, open_upper)

    return [Percentile(value=float(grid[i]), percentile=float(cdf[i])) for i in range(len(grid))]


def vincentize_cdfs(
    cdfs: list[list[Percentile]],
    question: NumericQuestion,
    method: str = "mean",
) -> list[Percentile]:
    """Vincentize (quantile-average) per-forecaster CDFs into one constraint-valid CDF.

    Each input is one forecaster's CDF (``list[Percentile]``). Each is inverted to a
    quantile function Q(p) = F^{-1}(p) on a shared dense probability grid, the VALUES are
    averaged at each probability level (``mean`` or ``median``), and the averaged quantile
    function is re-projected onto the question's ``num_points``-point value grid as a CDF.

    Quantile averaging preserves sharpness: a bimodal ensemble (one forecaster low, one
    high) yields a quantile function whose central spread tracks the average component
    location, rather than the smeared low-information F(x) that vertical averaging produces.
    """
    if method not in ("mean", "median"):
        raise ValueError(f"method must be 'mean' or 'median', got {method!r}")
    if not cdfs:
        raise ValueError("cdfs must be non-empty")

    grid = _question_grid(question, PCHIP_CDF_POINTS)
    prob_levels = np.linspace(0.0, 1.0, _QUANTILE_GRID_POINTS)

    # Invert each CDF to a quantile function on the shared probability grid. np.interp needs
    # strictly increasing x (= probability); jitter ties so the inversion is well-defined.
    quantile_curves: list[np.ndarray] = []
    for cdf in cdfs:
        probs = _cdf_probs(cdf)
        vals = _cdf_values(cdf)
        probs_mono = np.maximum.accumulate(probs)
        eps = np.arange(probs_mono.size) * 1e-12
        probs_strict = probs_mono + eps
        quantile_curves.append(np.interp(prob_levels, probs_strict, vals))

    stacked = np.vstack(quantile_curves)
    averaged_quantiles = np.mean(stacked, axis=0) if method == "mean" else np.median(stacked, axis=0)
    averaged_quantiles = np.maximum.accumulate(averaged_quantiles)

    # Re-project the averaged quantile function back to a CDF on the question grid:
    # F(x) = p where Q(p) = x, i.e. interpolate p against the (value -> prob) inverse.
    # Quantile averaging represents the upper tail as F(upper) = max_i(cdf_i[-1]) by
    # construction (the highest forecaster's reach into the range), which _finalize_cdf
    # then caps to <= 0.999 for open-upper questions — a defensible property of the
    # quantile method, not the dropped-bucket asymmetry log_pool_cdfs corrects.
    cdf_on_grid = np.interp(grid, averaged_quantiles, prob_levels)

    return _finalize_cdf(cdf_on_grid, grid, question)


def apply_tail_floor(
    cdf_values: list[float] | np.ndarray,
    question: NumericQuestion,
    floor_eps: float,
) -> list[float]:
    """Guarantee >= ``floor_eps`` PMF mass in every bucket (anti-saturation), re-normalized.

    Operates on a probability array (a 201-point CDF). The PMF is the per-bucket mass:
    ``diff(cdf)`` for interior buckets, plus the implicit boundary masses (``cdf[0]`` below
    the grid and ``1 - cdf[-1]`` above it) when the bound is open. We floor every step to
    ``floor_eps``, then hand off to the shared constraint layer which re-pins endpoints,
    re-enforces monotonicity + min-step, and redistributes any max-step overflow.

    ``floor_eps`` is a small tunable knob; values near or below ``MIN_CDF_PROB_STEP`` are a
    no-op relative to the existing min-step floor.
    """
    if floor_eps < 0:
        raise ValueError(f"floor_eps must be >= 0, got {floor_eps}")

    grid = _question_grid(question, len(cdf_values))
    arr = np.clip(np.asarray(cdf_values, dtype=float), 0.0, 1.0)
    arr = np.maximum.accumulate(arr)

    # Floor every interior step to at least floor_eps. Lifting low steps pushes the whole
    # tail of the CDF up; the constraint layer renormalizes and re-pins the endpoints.
    floored = enforce_min_steps(arr, max(floor_eps, MIN_CDF_PROB_STEP), upper_cap=1.0, lower_cap=0.0)

    finalized = _finalize_cdf(floored, grid, question)
    return [p.percentile for p in finalized]


def log_pool_cdfs(
    cdfs: list[list[Percentile]],
    question: NumericQuestion,
    weights: list[float] | None = None,
) -> list[Percentile]:
    """Geometric (log) pool of per-forecaster densities into one constraint-valid CDF.

    Differentiate each CDF to a per-bucket PMF, take the weighted geometric mean across
    forecasters (equivalent to a weighted average in log-density space), renormalize to
    sum 1, then re-integrate (cumsum) back to a CDF. Geometric pooling concentrates mass
    where forecasters agree — sharper than the arithmetic (vertical) average — so it is the
    natural "confident-consensus" counterpart to Vincentization's quantile averaging.

    ``weights`` (optional) are per-forecaster; they are normalized to sum 1. Defaults to
    equal weights.
    """
    if not cdfs:
        raise ValueError("cdfs must be non-empty")

    n = len(cdfs)
    if weights is None:
        w = np.full(n, 1.0 / n, dtype=float)
    else:
        if len(weights) != n:
            raise ValueError(f"weights length {len(weights)} must match number of cdfs {n}")
        w_arr = np.asarray(weights, dtype=float)
        if np.any(w_arr < 0):
            raise ValueError("weights must be non-negative")
        total = float(w_arr.sum())
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        w = w_arr / total

    grid = _question_grid(question, PCHIP_CDF_POINTS)

    # Per-forecaster PMF over the full 202 buckets a 201-point CDF decomposes into: the
    # below-lower boundary mass (cdf[0]), 200 interior masses (diff(cdf)), and the above-upper
    # boundary mass (1 - cdf[-1]). Keeping BOTH boundary buckets is what makes the pool
    # symmetric — dropping the upper bucket (and renormalizing) would silently discard the
    # open-upper tail mass forecasters assign above the grid. Tiny epsilon keeps log-space
    # pooling finite at empty buckets.
    n_buckets = len(grid) + 1
    log_density_acc = np.zeros(n_buckets, dtype=float)
    pmf_eps = 1e-12
    for cdf, weight in zip(cdfs, w):
        probs = np.maximum.accumulate(_cdf_probs(cdf))
        pmf = np.empty(n_buckets, dtype=float)
        pmf[0] = probs[0]  # below-lower boundary bucket
        pmf[1:-1] = np.diff(probs)  # interior buckets
        pmf[-1] = 1.0 - probs[-1]  # above-upper boundary bucket
        pmf = np.clip(pmf, 0.0, None) + pmf_eps
        log_density_acc += weight * np.log(pmf)

    pooled_pmf = np.exp(log_density_acc)
    pooled_pmf /= pooled_pmf.sum()

    # Re-integrate only the in-range portion (below-lower + interior = the first len(grid)
    # buckets). The pooled above-upper bucket (pooled_pmf[-1]) stays out of the CDF, so the
    # final value is cdf[-1] = 1 - that mass < 1 for open-upper questions with real tail mass.
    # _finalize_cdf applies the open/closed-bound caps (open-upper -> <= 0.999; closed-upper
    # -> pinned to 1.0, since the above bucket is ~eps when every input has cdf[-1] == 1).
    pooled_cdf = np.cumsum(pooled_pmf[: len(grid)])

    return _finalize_cdf(pooled_cdf, grid, question)
