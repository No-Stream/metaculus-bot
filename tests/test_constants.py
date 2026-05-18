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
    NATIVE_SEARCH_DEFAULT_MODEL,
    NATIVE_SEARCH_REASONING_EFFORT_DEFAULT,
    NATIVE_SEARCH_TIMEOUT,
    NATIVE_SEARCH_VERBOSITY_DEFAULT,
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


class TestNativeSearchDefaults:
    """Pin OpenAI native-search defaults to the W-A bench winner.

    These four constants together are the configuration that came out on top in
    ``scratch/native_search_bench_2026-05-17/comparison_v3.md`` — gpt-5.5 with
    medium reasoning effort + low verbosity, fitting under a 360s cap with
    materially deeper research than the gpt-5.4-mini fallback. Don't change
    these without rerunning the bench; rotating to a new model on a hunch
    silently regresses research quality on every question.
    """

    def test_native_search_default_model_is_gpt_5_5(self):
        """Locks the default OpenRouter model to ``openai/gpt-5.5``."""
        assert NATIVE_SEARCH_DEFAULT_MODEL == "openai/gpt-5.5"

    def test_native_search_reasoning_effort_default_is_medium(self):
        """Medium effort is the bench winner — high burned budget without
        improving rubric scores; low produced shallower research."""
        assert NATIVE_SEARCH_REASONING_EFFORT_DEFAULT == "medium"

    def test_native_search_verbosity_default_is_low(self):
        """Low verbosity keeps the response tight without losing substance."""
        assert NATIVE_SEARCH_VERBOSITY_DEFAULT == "low"

    def test_native_search_timeout_is_360s(self):
        """360s cap leaves ~130s headroom on top of observed p99 (~230s)."""
        assert NATIVE_SEARCH_TIMEOUT == 360
