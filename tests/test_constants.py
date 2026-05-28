"""
Tests pinning the values of key forecasting clamp constants.

These pin the *specific bounds* separately from the clamp-logic tests, so a
future hyperparameter change is a single-point edit with a clear test to
update and reason about.
"""

from metaculus_bot.constants import (
    BINARY_PROB_MAX,
    BINARY_PROB_MIN,
    BINARY_STACKING_ENABLED_ENV,
    MC_PROB_MAX,
    MC_PROB_MIN,
    MC_STACKING_ENABLED_ENV,
    NATIVE_SEARCH_DEFAULT_MODEL,
    NATIVE_SEARCH_REASONING_EFFORT_DEFAULT,
    NATIVE_SEARCH_TIMEOUT,
    NATIVE_SEARCH_VERBOSITY_DEFAULT,
    NUMERIC_STACKING_ENABLED_ENV,
    env_flag_enabled,
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
    """Pin OpenAI native-search defaults.

    Model + verbosity locked from ``scratch/native_search_bench_2026-05-17/comparison_v3.md``.
    Reasoning effort dropped medium→low on 2026-05-20 after an OpenRouter
    whitespace-stream incident consumed 8m37s on a single call; v3 bench
    measured low at ~50s vs medium at ~230s, so low gives ~4.5× more headroom
    under the wall-clock cap. Quality at low is not graded by the v3 bench
    (sanity-check only) — rerun the bench with a graded `low` arm if you
    want to revert.
    """

    def test_native_search_default_model_is_gpt_5_5(self):
        """Locks the default OpenRouter model to ``openai/gpt-5.5``."""
        assert NATIVE_SEARCH_DEFAULT_MODEL == "openai/gpt-5.5"

    def test_native_search_reasoning_effort_default_is_low(self):
        """Low effort gives ~4.5× faster wall-clock vs medium on the v3 bench
        (~50s vs ~230s), keeping us well clear of NATIVE_SEARCH_WALL_TIMEOUT
        (420s) after the 2026-05-20 OpenRouter whitespace-stream incident.
        Override via NATIVE_SEARCH_REASONING_EFFORT env if a workflow needs
        medium quality back."""
        assert NATIVE_SEARCH_REASONING_EFFORT_DEFAULT == "low"

    def test_native_search_verbosity_default_is_low(self):
        """Low verbosity keeps the response tight without losing substance."""
        assert NATIVE_SEARCH_VERBOSITY_DEFAULT == "low"

    def test_native_search_timeout_is_360s(self):
        """360s cap leaves ~130s headroom on top of observed p99 (~230s)."""
        assert NATIVE_SEARCH_TIMEOUT == 360


class TestEnvFlagEnabledDefaultKwarg:
    """Tests for the ``default`` keyword argument on ``env_flag_enabled``."""

    def test_unset_env_returns_default_true(self, monkeypatch):
        """When env var is unset, returns the provided default (True)."""
        monkeypatch.delenv("_TEST_FLAG_NONEXISTENT_XYZ", raising=False)
        assert env_flag_enabled("_TEST_FLAG_NONEXISTENT_XYZ", default=True) is True

    def test_unset_env_returns_default_false(self, monkeypatch):
        """When env var is unset and default=False, returns False."""
        monkeypatch.delenv("_TEST_FLAG_NONEXISTENT_XYZ", raising=False)
        assert env_flag_enabled("_TEST_FLAG_NONEXISTENT_XYZ", default=False) is False

    def test_unset_env_returns_false_when_no_default_specified(self, monkeypatch):
        """Backward compat: no default kwarg means default=False."""
        monkeypatch.delenv("_TEST_FLAG_NONEXISTENT_XYZ", raising=False)
        assert env_flag_enabled("_TEST_FLAG_NONEXISTENT_XYZ") is False

    def test_explicit_false_overrides_default_true(self, monkeypatch):
        """Explicit 'false' always returns False regardless of default."""
        monkeypatch.setenv("_TEST_FLAG_XYZ", "false")
        assert env_flag_enabled("_TEST_FLAG_XYZ", default=True) is False

    def test_explicit_true_overrides_default_false(self, monkeypatch):
        """Explicit 'true' always returns True regardless of default."""
        monkeypatch.setenv("_TEST_FLAG_XYZ", "true")
        assert env_flag_enabled("_TEST_FLAG_XYZ", default=False) is True

    def test_explicit_zero_overrides_default_true(self, monkeypatch):
        """'0' is falsy regardless of default."""
        monkeypatch.setenv("_TEST_FLAG_XYZ", "0")
        assert env_flag_enabled("_TEST_FLAG_XYZ", default=True) is False

    def test_explicit_one_overrides_default_false(self, monkeypatch):
        """'1' is truthy regardless of default."""
        monkeypatch.setenv("_TEST_FLAG_XYZ", "1")
        assert env_flag_enabled("_TEST_FLAG_XYZ", default=False) is True

    def test_empty_string_treated_as_unset(self, monkeypatch):
        """Empty string env var treated same as unset — returns default."""
        monkeypatch.setenv("_TEST_FLAG_XYZ", "")
        assert env_flag_enabled("_TEST_FLAG_XYZ", default=True) is True

    def test_unrecognized_value_returns_default(self, monkeypatch):
        """Garbage value falls through to default."""
        monkeypatch.setenv("_TEST_FLAG_XYZ", "maybe")
        assert env_flag_enabled("_TEST_FLAG_XYZ", default=True) is True


class TestPerTypeStackingEnvVarNames:
    """Pin the env var name constants for per-type stacking gates."""

    def test_binary_stacking_enabled_env_name(self):
        assert BINARY_STACKING_ENABLED_ENV == "BINARY_STACKING_ENABLED"

    def test_mc_stacking_enabled_env_name(self):
        assert MC_STACKING_ENABLED_ENV == "MC_STACKING_ENABLED"

    def test_numeric_stacking_enabled_env_name(self):
        assert NUMERIC_STACKING_ENABLED_ENV == "NUMERIC_STACKING_ENABLED"
