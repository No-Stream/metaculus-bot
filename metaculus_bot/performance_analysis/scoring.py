"""Scoring functions for performance analysis.

Re-exports shared pure scoring functions from metaculus_bot.scoring_common
so that existing imports (e.g. ``from metaculus_bot.performance_analysis.scoring import ...``)
continue to work without modification.

``spot_peer_delta`` rides along for the same reason: a residual round prices its
counterfactuals from this module, and the halving of a continuous peer delta is the one
piece of the convention rounds have got wrong (see its docstring).
"""

from metaculus_bot.scoring_common import (  # noqa: F401
    BOUNDARY_BASELINE,
    CONTINUOUS_PEER_DIVISOR,
    CONTINUOUS_QUESTION_TYPES,
    PROB_CLAMP_MAX,
    PROB_CLAMP_MIN,
    binary_log_score,
    brier_score,
    clamp_prob,
    mc_log_score,
    numeric_log_score,
    resolution_to_bucket_index,
    spot_peer_delta,
)
