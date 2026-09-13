"""Tests for which clamp was in force at publish time, and for the clamp primitive itself.

The sweep prices a counterfactual against the clamp that was live when the comment was posted, so
the boundary lookup and the ``lo``/``hi`` extension of the live MC clamp are pinned together.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from metaculus_bot.constants import MC_PROB_MAX, MC_PROB_MIN
from metaculus_bot.mc_processing import clamp_and_renormalize_probs
from metaculus_bot.performance_analysis.analysis import FT_0292_MERGED_AT, WIDENING_FLIP_MERGED_AT
from metaculus_bot.performance_analysis.clip_threshold_sweep import in_force_bounds
from metaculus_bot.performance_analysis.width_monitor import WIDENING_FLIP
from tests.clip_threshold_fakes import AFTER_WIDENING, AS_OF, BEFORE_WIDENING, one_binary


class TestInForceClampLookup:
    """The clamp in force is looked up from the bot-comment timestamp."""

    def test_binary_boundary(self):
        assert in_force_bounds("binary", WIDENING_FLIP_MERGED_AT - timedelta(seconds=1)) == (0.01, 0.99)
        assert in_force_bounds("binary", WIDENING_FLIP_MERGED_AT) == (0.02, 0.98)
        assert in_force_bounds("binary", WIDENING_FLIP_MERGED_AT + timedelta(days=30)) == (0.02, 0.98)

    def test_mc_boundary(self):
        assert in_force_bounds("multiple_choice", FT_0292_MERGED_AT - timedelta(seconds=1)) == (0.005, 0.995)
        assert in_force_bounds("multiple_choice", FT_0292_MERGED_AT) == (0.01, 0.99)

    def test_no_timestamp_falls_back_to_the_widest_regime(self):
        """Undatable records get the LOOSEST historical clamp, the assumption that claims the least.

        A censoring claim has to know which floor was live, and an undated record cannot say.
        """
        assert in_force_bounds("binary", None) == (0.01, 0.99)
        assert in_force_bounds("multiple_choice", None) == (0.005, 0.995)

    def test_record_carries_the_bounds_its_timestamp_implies(self):
        assert one_binary(question_id=50, p_yes=0.5, resolution=True, created_at=BEFORE_WIDENING).in_force_lo == 0.01
        assert one_binary(question_id=51, p_yes=0.5, resolution=True, created_at=AFTER_WIDENING).in_force_lo == 0.02

    def test_unknown_question_type_raises(self):
        with pytest.raises(KeyError):
            in_force_bounds("numeric", AS_OF)

    def test_width_monitor_alias_is_the_shared_constant(self):
        assert WIDENING_FLIP is WIDENING_FLIP_MERGED_AT


class TestMcClampSignatureExtension:
    """The ``lo``/``hi`` kwargs are a pure extension of the live clamp."""

    def test_defaults_reproduce_the_module_globals(self):
        vector = [0.984, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002]
        assert clamp_and_renormalize_probs(vector) == clamp_and_renormalize_probs(
            vector, lo=MC_PROB_MIN, hi=MC_PROB_MAX
        )
        clamped = clamp_and_renormalize_probs(vector)
        assert sum(clamped) == pytest.approx(1.0, abs=1e-9)
        assert all(MC_PROB_MIN <= p <= MC_PROB_MAX for p in clamped)

    def test_default_still_reads_the_module_global_at_call_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The monkeypatch surface the existing suite relies on must survive the extension.

        A literal ``lo=MC_PROB_MIN`` default would bind at import, so patching the module
        global would silently stop working; the sentinel keeps the late lookup.
        """
        monkeypatch.setattr("metaculus_bot.mc_processing.MC_PROB_MIN", 0.20)
        clamped = clamp_and_renormalize_probs([0.90, 0.05, 0.05])
        assert min(clamped) >= 0.20 - 1e-12

    def test_explicit_floor_holds_on_a_four_option_vector(self):
        clamped = clamp_and_renormalize_probs([0.90, 0.06, 0.03, 0.01], lo=0.05, hi=0.95)
        assert sum(clamped) == pytest.approx(1.0, abs=1e-9)
        assert all(p >= 0.05 - 1e-12 for p in clamped)
        assert all(p <= 0.95 + 1e-12 for p in clamped)

    def test_explicit_floor_repairs_a_dominant_option(self):
        clamped = clamp_and_renormalize_probs([0.97, 0.01, 0.01, 0.01], lo=0.05, hi=0.99)
        assert sum(clamped) == pytest.approx(1.0, abs=1e-9)
        assert min(clamped) == pytest.approx(0.05, abs=1e-9)

    def test_exactly_feasible_floor_returns_the_uniform_vector(self):
        """``n * lo == 1.0`` has exactly one in-bounds sum-1 solution, the uniform vector, and
        the strict degenerate guard lets the normal path converge to it. The old ``>=`` sent this
        case down the degenerate branch and returned sub-floor values, which priced the sweep's
        MC c = 0.10 cell on the archive's two 10-option ballots at an effective floor of ~0.07."""
        vector = [0.30, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.03, 0.02]
        clamped = clamp_and_renormalize_probs(vector, lo=0.10, hi=0.90)
        assert sum(clamped) == pytest.approx(1.0, abs=1e-9)
        assert all(p == pytest.approx(0.10, abs=1e-9) for p in clamped)

    def test_infeasible_floor_takes_the_documented_degenerate_fallback(self):
        """Eleven options at a 0.10 floor already exceed 1, so no in-bounds sum-1 vector exists;
        the fallback keeps the sum at 1 and its minimum sits BELOW the requested floor."""
        vector = [0.30, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01]
        clamped = clamp_and_renormalize_probs(vector, lo=0.10, hi=0.90)
        assert sum(clamped) == pytest.approx(1.0, abs=1e-9)
        assert min(clamped) < 0.10

    def test_binding_ceiling_pins_the_leader_and_rescales_the_rest(self):
        """The ``hi`` kwarg reaches the repair loop: a 0.5 ceiling pins the 0.9 leader and the
        freed mass rescales the other two, so an implementation reading ``MC_PROB_MAX`` instead
        of ``hi`` cannot pass."""
        clamped = clamp_and_renormalize_probs([0.9, 0.05, 0.05], lo=0.01, hi=0.5)
        assert clamped == pytest.approx([0.5, 0.25, 0.25], abs=1e-12)
        assert max(clamped) <= 0.5 + 1e-12
        assert sum(clamped) == pytest.approx(1.0, abs=1e-9)
