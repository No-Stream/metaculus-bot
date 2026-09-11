"""Shared pure scoring functions used by both backtest and performance_analysis modules.

All functions are pure with no side effects and have no dependency on forecasting_tools.
"""

import math

from metaculus_bot.prob_math_utils import PROB_CLAMP_EPS, clamp_prob

PROB_CLAMP_MIN: float = PROB_CLAMP_EPS
PROB_CLAMP_MAX: float = 1.0 - PROB_CLAMP_EPS
BOUNDARY_BASELINE: float = 0.05

# The platform's own halved family (``QUESTION_CONTINUOUS_TYPES``); "discrete" is CONTINUOUS there, hence UNHALVED, not DISCRETE, for the other name.
CONTINUOUS_QUESTION_TYPES: frozenset[str] = frozenset({"numeric", "discrete", "date"})
UNHALVED_QUESTION_TYPES: frozenset[str] = frozenset({"binary", "multiple_choice"})
CONTINUOUS_PEER_DIVISOR: float = 2.0

__all__ = [
    "BOUNDARY_BASELINE",
    "CONTINUOUS_PEER_DIVISOR",
    "CONTINUOUS_QUESTION_TYPES",
    "PROB_CLAMP_MAX",
    "PROB_CLAMP_MIN",
    "UNHALVED_QUESTION_TYPES",
    "baseline_to_peer_factor",
    "binary_log_score",
    "brier_score",
    "clamp_prob",
    "mc_log_score",
    "numeric_log_score",
    "resolution_to_bucket_index",
    "spot_peer_delta",
]


def brier_score(predicted_prob: float, outcome: bool) -> float:
    """Brier score: (clamp(p) - y)^2. Lower is better."""
    p = clamp_prob(predicted_prob)
    y = 1.0 if outcome else 0.0
    return (p - y) ** 2


def binary_log_score(predicted_prob: float, outcome: bool) -> float:
    """Metaculus-style log score for binary questions.

    Formula: 100 * (y * (log2(p) + 1) + (1 - y) * (log2(1 - p) + 1))
    Higher is better. Uniform prediction (0.5) scores 0.
    """
    p = clamp_prob(predicted_prob)
    y = 1.0 if outcome else 0.0
    return 100.0 * (y * (math.log2(p) + 1.0) + (1.0 - y) * (math.log2(1.0 - p) + 1.0))


def resolution_to_bucket_index(
    resolution: float,
    lower_bound: float,
    upper_bound: float,
    *,
    n_inbound: int,
    zero_point: float | None = None,
) -> int:
    """Map a numeric resolution value to a PMF bucket index.

    Replicates Metaculus backend's unscaled_location_to_bucket_index.
    Returns bucket in [0, n_inbound+1] where 0 = below-lower-bound, n_inbound+1 = above-upper-bound.
    """
    total_range = upper_bound - lower_bound
    if total_range <= 0:
        raise ValueError(f"Invalid bounds: lower={lower_bound}, upper={upper_bound}")

    if zero_point is not None:
        deriv_ratio = (upper_bound - zero_point) / (lower_bound - zero_point)
        scaled_offset = (resolution - lower_bound) * (deriv_ratio - 1) + total_range
        if scaled_offset <= 0:
            return 0
        unscaled = math.log(scaled_offset / total_range) / math.log(deriv_ratio)
    else:
        unscaled = (resolution - lower_bound) / total_range

    if unscaled < 0:
        return 0
    if unscaled > 1:
        return n_inbound + 1
    if unscaled == 1.0:
        return n_inbound
    return max(int(unscaled * n_inbound + 1 - 1e-10), 1)


def numeric_log_score(
    cdf_values: list[float],
    resolution: float,
    lower_bound: float,
    upper_bound: float,
    *,
    open_lower_bound: bool,
    open_upper_bound: bool,
    zero_point: float | None = None,
) -> float:
    """Metaculus-style PMF-bucket log score for numeric questions.

    Formula: 50 * ln(pmf[resolution_bucket] / baseline)
    Higher is better. Uniform prediction scores 0.

    CDF is converted to a PMF with len(cdf)+1 buckets (boundary + interior).
    The resolution maps to one bucket; the score is the log of the PMF mass
    in that bucket relative to a uniform baseline.
    """
    n_cdf = len(cdf_values)
    if n_cdf < 2:
        raise ValueError(f"Need at least 2 CDF values, got {n_cdf}")

    n_inbound = n_cdf - 1  # 200 for standard 201-point CDF

    pmf = [cdf_values[0]]
    for i in range(1, n_cdf):
        pmf.append(cdf_values[i] - cdf_values[i - 1])
    pmf.append(1.0 - cdf_values[-1])

    bucket = resolution_to_bucket_index(
        resolution, lower_bound, upper_bound, n_inbound=n_inbound, zero_point=zero_point
    )

    n_open_bounds = int(open_lower_bound) + int(open_upper_bound)
    if bucket in (0, len(pmf) - 1):
        baseline = BOUNDARY_BASELINE
    else:
        baseline = (1.0 - BOUNDARY_BASELINE * n_open_bounds) / n_inbound

    pmf_value = max(pmf[bucket], 1e-15)
    return 50.0 * math.log(pmf_value / baseline)


def mc_log_score(predicted_probs: list[float], correct_option_index: int) -> float:
    """Metaculus-style log score for multiple-choice questions.

    Formula: 100 * (log2(clamp(p_correct)) / log2(K) + 1)
    Higher is better. Uniform prediction scores 0.
    """
    k = len(predicted_probs)
    if k < 2:
        raise ValueError(f"Need at least 2 options, got {k}")
    if correct_option_index < 0 or correct_option_index >= k:
        raise ValueError(f"correct_option_index {correct_option_index} out of range [0, {k})")

    p_correct = clamp_prob(predicted_probs[correct_option_index])
    return 100.0 * (math.log2(p_correct) / math.log2(k) + 1.0)


def baseline_to_peer_factor(question_type: str, *, n_options: int | None = None) -> float:
    """Multiply a difference of two baseline log scores by this to read it in spot-peer points.

    The companion of :func:`spot_peer_delta` for callers holding scores rather than probabilities.
    :func:`binary_log_score` and :func:`mc_log_score` are log-base-K, so their differences are in
    ``log_K`` units and reach the platform's natural-log peer points times ``ln K``;
    :func:`numeric_log_score` already returns the halved ``50 ln`` form, so a continuous difference is
    a peer delta as it stands. Raises on an unrecognized type or a multiple-choice call without its
    option count, for the same reason ``spot_peer_delta`` does.
    """
    if question_type in CONTINUOUS_QUESTION_TYPES:
        return 1.0
    if question_type == "binary":
        return math.log(2.0)
    if question_type == "multiple_choice":
        if n_options is None or n_options < 2:
            raise ValueError(f"a multiple-choice score delta needs its option count, got {n_options!r}")
        return math.log(n_options)
    raise ValueError(
        f"unrecognized {question_type=}; expected one of {sorted(CONTINUOUS_QUESTION_TYPES | UNHALVED_QUESTION_TYPES)}"
    )


def spot_peer_delta(*, old_prob: float, new_prob: float, question_type: str) -> float:
    """Spot-peer points gained by moving OUR mass on the resolving outcome old -> new.

    ``100 * ln(new/old)``, halved for a :data:`CONTINUOUS_QUESTION_TYPES` question: the platform's
    spot peer is ``100 * (N/(N-1)) * ln(p/gmp)`` and the crowd mean includes us, so changing only our
    forecast leaves no crowd term. The halving is the easiest thing in the codebase to apply twice
    (:func:`numeric_log_score` already carries it) and the log-base-K baseline scores need ``ln K`` to
    reach peer points (:func:`baseline_to_peer_factor`); both traps, their receipts and the platform
    source are in docs/performance_analysis.md "Price a counterfactual with spot_peer_delta".

    Raises ``ValueError`` on a non-positive probability (a zero mass on the resolving outcome is a
    caller-side pmf extraction bug, not a -inf score) or an unrecognized question type (silently
    taking the un-halved branch is the bug this guards).
    """
    if question_type in CONTINUOUS_QUESTION_TYPES:
        divisor = CONTINUOUS_PEER_DIVISOR
    elif question_type in UNHALVED_QUESTION_TYPES:
        divisor = 1.0
    else:
        raise ValueError(
            f"unrecognized {question_type=}; expected one of "
            f"{sorted(CONTINUOUS_QUESTION_TYPES | UNHALVED_QUESTION_TYPES)}"
        )
    if old_prob <= 0.0 or new_prob <= 0.0:
        raise ValueError(f"spot peer delta needs positive probabilities, got {old_prob=} {new_prob=}")
    return 100.0 * math.log(new_prob / old_prob) / divisor
