"""
Configuration constants for numeric forecasting pipeline.

Extracted from main.py to centralize magic numbers and make them more maintainable.
These constants control various aspects of the numeric prediction processing pipeline.
"""

from __future__ import annotations

from metaculus_bot.constants import NUM_MAX_STEP

# --- Percentile Processing Constants ---

EXPECTED_PERCENTILE_COUNT: int = 11

# Expressed as decimals in [0,1]
STANDARD_PERCENTILES: list[float] = [
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
]

MIN_PERCENTILES_REQUIRED: int = 3

# --- PCHIP CDF Configuration ---

PCHIP_CDF_POINTS: int = 201

MIN_CDF_PROB_STEP: float = 5e-5

MAX_CDF_PROB_STEP: float = NUM_MAX_STEP

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

# --- Diagnostic and Logging Thresholds ---

LARGE_CORRECTION_THRESHOLD: float = 0.1

MAX_DIAGNOSTIC_PERCENTILES: int = 5

EXTREME_STEP_THRESHOLD: float = NUM_MAX_STEP * 0.9

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
