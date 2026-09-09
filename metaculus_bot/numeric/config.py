"""
Configuration constants for numeric forecasting pipeline.

Extracted from main.py to centralize magic numbers and make them more maintainable.
These constants control various aspects of the numeric prediction processing pipeline.
"""

from __future__ import annotations

import math

from forecasting_tools.data_models.questions import DiscreteQuestion, NumericQuestion

from metaculus_bot.constants import NUM_MAX_STEP

# --- Percentile Processing Constants ---

# Expressed as decimals in [0,1]. P1 (0.01) and P99 (0.99) give forecasters finer tail
# anchors so they can express probability mass below an open lower bound / above an open
# upper bound (the Minions & Monsters miss).
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

# Human-readable label CSV (percentile * 100, e.g. "1,2.5,5,...,97.5,99"). The single
# source for the label string embedded in prompts and validation errors — never hardcode
# the list elsewhere so it stays in lockstep with STANDARD_PERCENTILES.
STANDARD_PERCENTILES_CSV: str = ",".join(f"{p * 100:g}" for p in STANDARD_PERCENTILES)

MIN_PERCENTILES_REQUIRED: int = 3

# --- PCHIP CDF Configuration ---

# The standard grid. ``NumericQuestion.cdf_size`` (and ``DateQuestion.cdf_size``) is a
# non-optional int on the forecasting-tools model, defaulting to this, so no reader in the
# pipeline normalises an absent value: a test double that wants the standard grid passes it.
PCHIP_CDF_POINTS: int = 201

# The 201-grid max step by name: the reference the open-bound piling threshold is calibrated
# against (numeric/diagnostics.py) and the pre-scaling cap the residual analysis reads back
# (performance_analysis/analysis.py). Every live grid limit comes from grid_step_constraints.
MAX_CDF_PROB_STEP: float = NUM_MAX_STEP


def grid_step_constraints(num_points: int) -> tuple[float, float]:
    """Return ``(min_step, max_step)`` for a ``num_points``-point CDF grid.

    The server's per-bin rules (``questions/serializers/common.py`` in the open-source
    Metaculus backend, which Mantic forked with the same constants) scale with the bin
    count ``inbound = num_points - 1``, and the server checks them against a PMF it has
    rounded to 9 decimals (``np.round(np.diff(cdf), 9)``), so both limits here are the
    9-decimal values that survive that rounding:

    * min step ``round(0.01 / inbound, 9)``, the server's own rounded floor.
    * max step: the largest 9-decimal value not exceeding ``0.2 * 200 / inbound``, clamped
      at ``1.0`` (a probability step can never exceed 1.0). The server compares the ROUNDED
      pmf against the UNROUNDED cap, and the max-step repair clips over-cap bins to exactly
      this value, so wherever the cap is not 9-decimal exact a bin clipped to the raw cap
      rounds above it and the submission is rejected (450 bins: 0.0888... rounds to
      0.088888889). Flooring is a no-op wherever the cap is 9-decimal exact, which covers
      every grid of 41 points or fewer and the 51, 101, 201 and 2,001-point grids.

    There is deliberately no floor at the 201-grid min step. Such a floor was a no-op for
    ``inbound <= 200`` and stricter than the server above it (2.25x at 450 bins, 10x at
    2,000), which forced a uniform mixture several times larger than the server requires
    into the tails of every fine-grid forecast. At the standard 201-point grid this returns
    exactly ``(NUM_MIN_PROB_STEP, NUM_MAX_STEP)``, the constants the 201-point builders
    default to. On a coarse discrete grid (``num_points < 201``) the max step relaxes above
    0.2 (1.0 at ``num_points=9``), which lets a small-count distribution keep its mass
    concentrated on the low integers instead of being clipped to the 201-grid cap; on a
    finer grid both limits tighten.
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
    discrete question and every Mantic quantitative question, whose wire type the Mantic
    client rewrites to ``discrete``) and any non-201 grid. On both, the vote-gated discrete
    snap never runs (``discrete_snap`` skips on exactly this predicate), so what the
    sanitizer's cluster spreader publishes is the final word and a plateau is kept inside
    its bin (``cluster_processing``). On the 201-point continuous grid of a
    ``NumericQuestion`` the count-like unit spread is a pre-processing step the snap can
    re-concentrate afterwards, so that grid stays byte-identical. ``cdf_size`` alone cannot
    carry the distinction: a 200-bin Mantic discrete question has ``cdf_size == 201``.

    The ablation harness rehydrates archived questions as plain ``NumericQuestion`` with the
    recorded ``cdf_size``, so its replay of a 200-bin discrete question would take the
    continuous branch here. That divergence bites only on Mantic 200-bin discrete questions,
    which do not exist in the Metaculus archives the harness replays.
    """
    return isinstance(question, DiscreteQuestion) or question.cdf_size != PCHIP_CDF_POINTS


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
    or spread value keeps inside a closed bound (``bounds_clamping``, ``cluster_processing``) and
    the gap the jitter and strict-ordering passes open between equal neighbours. One formula so
    those passes agree on what "just inside" and "just above" mean and never fight over a value
    one of them placed.
    """
    return max(MIN_BOUNDARY_DISTANCE * range_size, STRICT_ORDERING_EPSILON)


# --- Diagnostic and Logging Thresholds ---

LARGE_CORRECTION_THRESHOLD: float = 0.1

MAX_DIAGNOSTIC_PERCENTILES: int = 5

EXTREME_STEP_THRESHOLD: float = NUM_MAX_STEP * 0.9

# Top/bottom-bin mass at or above this fraction, with no percentile placed beyond the open
# edge, flags open-bound percentile piling (models treating an open edge as a hard cap).
# n=1 calibration: fires on the two observed crammers (0.20 and 0.126 top-bin mass) but not
# the four correct handlers (all <= 0.073). K is intentionally tunable as more data arrives.
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

# Tail widening stretch factor applied in transformed space around the median in tails.
# e.g., 1.25 means 25% stretch at the deepest tails, ramping to 0% near the center.
# Default is 1.0 (identity pass, no widening) per the 2026-05-12 empirical calibration
# on 43 resolved numerics: k_tail=1.0 produced PIT std closest to the uniform ideal
# (0.289) in every segment; k_tail=1.25 moved away from ideal in every segment. See
# scratch_docs_and_planning/tail_widening_empirical_calibration.md.
TAIL_WIDEN_K_TAIL: float = 1.0

# Tail start region (fraction of percentile distance from median where widening begins)
# Example: 0.2 means no widening for p in [0.3, 0.7], linearly ramp to full widening by p<=0.1 or p>=0.9
TAIL_WIDEN_TAIL_START: float = 0.2

# Span floor gamma to ensure tail spans are at least gamma times adjacent inner spans.
# Applies to (p05 - p02.5) vs (p10 - p05) and (p97.5 - p95) vs (p95 - p90).
# Floor enforcement (tail_widening.py:171/178) is gated on `> 0`. Default disabled
# because in all 2026 data the floor never bound (see
# scratch_docs_and_planning/tail_widening_empirical_calibration.md section 3).
# Setting this to any positive value re-enables the existing floor enforcement —
# kept configurable for future models with unusually sharp declared tails.
TAIL_WIDEN_SPAN_FLOOR_GAMMA: float = 0.0
