"""
Configuration constants for numeric forecasting pipeline.

Extracted from main.py to centralize magic numbers and make them more maintainable.
These constants control various aspects of the numeric prediction processing pipeline.
"""

from __future__ import annotations

import math

from forecasting_tools.data_models.questions import DiscreteQuestion, NumericQuestion

from metaculus_bot.constants import NUM_MAX_STEP, PLATFORM_MANTIC
from metaculus_bot.question_platform import question_platform

# --- Percentile Processing Constants ---

# Decimals in [0, 1]; P1 and P99 are the tail anchors that let a forecaster place mass beyond an open bound.
STANDARD_PERCENTILES: list[float] = [
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.40,
    0.50,
    0.60,
    0.80,
    0.90,
    0.95,
    0.975,
    0.99,
]

EXPECTED_PERCENTILE_COUNT: int = len(STANDARD_PERCENTILES)

# The one source of the percentile label CSV for prompts and validation errors; never restate the list elsewhere.
STANDARD_PERCENTILES_CSV: str = ",".join(f"{p * 100:g}" for p in STANDARD_PERCENTILES)

MIN_PERCENTILES_REQUIRED: int = 3

# --- PCHIP CDF Configuration ---

# The standard grid; ``cdf_size`` is a non-optional int defaulting to this, so no reader normalises an absent value.
PCHIP_CDF_POINTS: int = 201

# The 201-grid cap by name for the piling threshold and the residual analysis; live limits come from grid_step_constraints.
MAX_CDF_PROB_STEP: float = NUM_MAX_STEP


def grid_step_constraints(num_points: int) -> tuple[float, float]:
    """Return ``(min_step, max_step)`` for a ``num_points``-point CDF grid, as the server enforces them.

    The server's per-bin rules scale with the bin count ``inbound = num_points - 1`` and are
    checked against a PMF rounded to 9 decimals, so both limits are the 9-decimal values that
    survive that rounding: the min step is the server's own ``round(0.01 / inbound, 9)``; the
    max step is the largest 9-decimal value not exceeding ``0.2 * 200 / inbound``, clamped at
    1.0, because a bin clipped to the raw cap rounds above it wherever the cap is not 9-decimal
    exact (450 bins: 0.0888... rounds to 0.088888889 and the submission is rejected). There is
    deliberately no floor at the 201-grid min step: it was a no-op below 200 bins and 2.25x to
    10x stricter than the server above them. At 201 points this is exactly
    ``(NUM_MIN_PROB_STEP, NUM_MAX_STEP)``; a coarse grid relaxes the cap, a fine grid tightens
    both. Detail: ``docs/numeric_pipeline.md``, "Server-side constraints".
    """
    inbound = max(1, num_points - 1)
    min_step = round(0.01 / inbound, 9)
    max_step = min(1.0, math.floor(0.2 * 200.0 / inbound * 1e9) / 1e9)
    return min_step, max_step


def grid_bin_width(lower_bound: float, upper_bound: float, num_points: int) -> float:
    """Width of one bin of a ``num_points``-point linear CDF grid over ``[lower, upper]``.

    ``(upper - lower) / (num_points - 1)``: the platform derives a discrete question's
    ``range_min`` / ``range_max`` as the nominal bounds pushed out by half a step, so on an
    integer-count question this is exactly 1.0 and each bin is centred on its integer. It is
    the grid step the cluster spreader keeps a plateau inside of (``cluster_processing``) and
    the one the discrete-snap skip reports (``discrete_snap``). Linear only: on a ``zero_point``
    grid the bins are geometric and this is the mean width.
    """
    return (upper_bound - lower_bound) / (num_points - 1)


def grid_is_outcome_space(question: NumericQuestion) -> bool:
    """True when the question's published bins are its outcome space and nothing downstream reshapes them.

    Two shapes qualify: a natively discrete question (``DiscreteQuestion``: every Metaculus
    discrete question and every Mantic quantitative question, whose wire type the Mantic client
    rewrites to ``discrete``) and any non-201 grid. On both, the vote-gated discrete snap never
    runs (``discrete_snap`` skips on exactly this predicate), so the cluster spreader's plateau
    inside its bin is the final word; on the 201-point continuous grid the count-like spread is
    a pre-processing step the snap can re-concentrate, so that grid stays byte-identical.
    ``cdf_size`` alone cannot carry the distinction: a 200-bin Mantic discrete question has
    ``cdf_size == 201``. The ablation harness's replay caveat: ``docs/numeric_pipeline.md``, Step 7.
    """
    return isinstance(question, DiscreteQuestion) or question.cdf_size != PCHIP_CDF_POINTS


# --- Per-bin elicitation on enumerable grids ---

# A month of daily bins: the natural coarse Series 2 date shape, and 29% of Series 1 discrete grids sit at or below it.
PMF_ELICITATION_MAX_BINS: int = 31

# Mantic only until one season shows the per-bin declaration is faithful; adding PLATFORM_METACULUS is the whole switch.
PMF_ELICITATION_PLATFORMS: frozenset[str] = frozenset({PLATFORM_MANTIC})

# Added to every non-zero cell floor of a per-bin build so the server's 9-decimal PMF rounding can never land a cell under it.
PMF_FLOOR_MARGIN: float = 1e-9

# The server's minimum mass beyond an OPEN bound: ``cdf[0] >= 0.001`` and ``cdf[-1] <= 1 - 0.001``.
OPEN_TAIL_MIN_MASS: float = 0.001


def elicit_per_bin(question: NumericQuestion) -> bool:
    """True when ``question`` is forecast as one probability per bin instead of as percentiles.

    Per bin because on a coarse grid the bins ARE the outcome space: 13 percentile anchors cannot
    say "zero on this bin", and PCHIP spreads mass onto bins the criteria exclude (question 651:
    three weekend days in a 12-day trading-day window, a quarter of the mass, -14.4 baseline
    points). Above ``PMF_ELICITATION_MAX_BINS`` the per-bin ask grows long and noisy and the
    13-anchor curve is the better instrument. Mantic-only because the ask is unproven live and on
    Metaculus it would move about half of all discrete questions (141 of 300 sampled have 31 bins
    or fewer), a config-era change of its own.

    ``cdf_size - 1`` is the bin count (``inbound_outcome_count + 1`` on every real question). False
    by construction on the 201-point continuous grid (not an outcome-space grid) and on a 200-bin
    Mantic discrete question (far above the threshold), so both keep the percentile path
    byte-identical. A date question is gated on its epoch view, which carries ``page_url``.
    """
    return (
        question_platform(question) in PMF_ELICITATION_PLATFORMS
        and grid_is_outcome_space(question)
        and (question.cdf_size - 1) <= PMF_ELICITATION_MAX_BINS
    )


# Higher = more aggressive smoothing
CDF_RAMP_K_FACTOR: float = 3.0

# --- Cluster Detection and Spreading Constants ---

# Relative tolerance below which two values are considered an identical cluster
CLUSTER_DETECTION_RTOL: float = 1e-9

CLUSTER_DETECTION_ATOL: float = 1e-12

CLUSTER_SPREAD_BASE_DELTA: float = 1e-6

# Spacing below which a distribution is treated as "count-like" (integer-adjacent)
COUNT_LIKE_THRESHOLD: float = 0.1

COUNT_LIKE_DELTA_MULTIPLIER: float = 1.0

# --- Jitter and Validation Constants ---

STRICT_ORDERING_EPSILON: float = 1e-12

MAX_JITTER_ITERATIONS: int = 10

JITTER_CONVERGENCE_TOL: float = 1e-10

# --- Boundary Handling Constants ---

# As a fraction of the question range
BOUNDARY_SAFETY_MARGIN: float = 0.01

MIN_BOUNDARY_DISTANCE: float = 1e-9


def minimum_separation(range_size: float) -> float:
    """The smallest gap the sanitizer opens on a ``range_size``-wide axis.

    ``max(MIN_BOUNDARY_DISTANCE * range_size, STRICT_ORDERING_EPSILON)``: the standoff a clamped
    value keeps inside a closed bound (``bounds_clamping``), the standoff the 201-point spreader
    keeps inside a closed bound (``cluster_processing``; the outcome-space spreader instead
    translates its plateau to sit on the bound), and the gap the jitter and strict-ordering passes
    open between equal neighbours. One formula so those passes agree on what "just inside" and
    "just above" mean and never fight over a value one of them placed.
    """
    return max(MIN_BOUNDARY_DISTANCE * range_size, STRICT_ORDERING_EPSILON)


# --- Diagnostic and Logging Thresholds ---

LARGE_CORRECTION_THRESHOLD: float = 0.1

MAX_DIAGNOSTIC_PERCENTILES: int = 5

EXTREME_STEP_THRESHOLD: float = NUM_MAX_STEP * 0.9

# Terminal-bin mass that flags open-bound piling: fires on the two observed crammers (0.20, 0.126), not the four correct handlers (<= 0.073).
OPEN_BOUND_PILING_THRESHOLD: float = 0.10

# --- PCHIP Fallback Configuration ---

MAX_PCHIP_ATTEMPTS: int = 3

PCHIP_RETRY_BACKOFF: float = 1.5

# Seconds
PCHIP_BASE_RETRY_DELAY: float = 0.1

# --- Validation Tolerances ---

PERCENTILE_ORDER_TOLERANCE: float = 1e-10

BOUND_VALIDATION_TOLERANCE: float = 1e-8

MAX_PERCENTILE_RELATIVE_ERROR: float = 1e-6

# --- Tail Widening (identity-pass defaults; configurable) ---

# Enable/disable transform-space tail widening of declared percentiles before CDF generation
TAIL_WIDENING_ENABLE: bool = True

# Deepest-tail stretch factor; 1.0 is the identity pass, since 1.25 moved PIT std away from ideal in every segment (docs, Step 4).
TAIL_WIDEN_K_TAIL: float = 1.0

# Widening starts this far from the median: none for p in [0.3, 0.7], full at p <= 0.1 or p >= 0.9.
TAIL_WIDEN_TAIL_START: float = 0.2

# Tail spans at least this multiple of the adjacent inner span; 0 disables the floor, which never bound in 2026 data.
TAIL_WIDEN_SPAN_FLOOR_GAMMA: float = 0.0
