"""
Tests pinning the values of key forecasting clamp constants.

These pin the *specific bounds* separately from the clamp-logic tests, so a
future hyperparameter change is a single-point edit with a clear test to
update and reason about.
"""

from metaculus_bot.constants import (
    BINARY_PROB_MAX,
    BINARY_PROB_MIN,
    MC_PROB_MAX,
    MC_PROB_MIN,
)


class TestBinaryClampBounds:
    """Pin binary clamp bounds at [0.02, 0.98]."""

    def test_binary_prob_min_is_0_02(self):
        """BINARY_PROB_MIN == 0.02.

        Adopted from Preseen-Atlas (top bot on spring-AIB-2026), whose comments
        publish `submitted = 0.96 * model_estimate + 0.02` in every forecast —
        equivalent to clipping to [0.02, 0.98]. We're taking the clip (tail
        protection via log-loss reasoning) without the linear shrink.

        See: scratch_docs_and_planning/atlas_inspired_improvements.md (Workstream B).
        """
        assert BINARY_PROB_MIN == 0.02

    def test_binary_prob_max_is_0_98(self):
        """BINARY_PROB_MAX == 0.98.

        See `test_binary_prob_min_is_0_02` for rationale (Atlas-inspired
        clip-only adoption).
        """
        assert BINARY_PROB_MAX == 0.98


class TestMCClampBoundsUnchanged:
    """MC clamp bounds stay at [0.005, 0.995] — Workstream B only touches binary."""

    def test_mc_prob_min_is_0_005(self):
        assert MC_PROB_MIN == 0.005

    def test_mc_prob_max_is_0_995(self):
        assert MC_PROB_MAX == 0.995
