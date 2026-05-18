"""Tests for Platt scaling (logistic recalibration) of binary / MC probabilities.

Covers:

* ``PlattParams`` dataclass invariants (identity, frozen).
* ``apply_binary_platt`` transform + cap + final clamp.
* ``apply_mc_platt`` per-option transform + clamp-and-renormalize.
* ``fit_platt`` validation, recovery on synthetic data, and refusal modes.

Tests use real ``scipy.optimize`` fits (no mocking) — synthetic data is
seeded for determinism. The synthetic-recovery tolerance is ``±0.2`` for
both bias and slope on n=2000; tightening below that fails on the
random-Bernoulli noise floor for this sample size.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from forecasting_tools.data_models.multiple_choice_report import PredictedOption, PredictedOptionList

from metaculus_bot.calibration.platt import (
    MIN_INFORMATIVE_SLOPE,
    PlattParams,
    apply_binary_platt,
    apply_mc_platt,
    fit_platt,
)
from metaculus_bot.constants import BINARY_PROB_MAX, BINARY_PROB_MIN, MC_PROB_MAX, MC_PROB_MIN


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


class TestPlattParams:
    def test_identity_is_zero_one(self):
        ident = PlattParams.identity()
        assert ident.bias == 0.0
        assert ident.slope == 1.0

    def test_is_identity_true_for_identity(self):
        assert PlattParams.identity().is_identity() is True

    def test_is_identity_false_for_nonzero_bias(self):
        assert PlattParams(bias=0.1, slope=1.0).is_identity() is False

    def test_is_identity_false_for_nonunit_slope(self):
        assert PlattParams(bias=0.0, slope=1.1).is_identity() is False

    def test_frozen_assignment_raises(self):
        params = PlattParams.identity()
        with pytest.raises(FrozenInstanceError):
            params.bias = 0.5  # ty: ignore[invalid-assignment]  # type: ignore[misc]


class TestApplyBinaryPlatt:
    @pytest.mark.parametrize("p", [0.02, 0.1, 0.5, 0.9, 0.98])
    def test_identity_round_trip_in_bounds(self, p: float):
        # Identity + no cap → fast path returns p (after the trivial clamp).
        assert apply_binary_platt(p, PlattParams.identity()) == pytest.approx(p, abs=1e-12)

    def test_identity_below_clamp_lifts_to_min(self):
        assert apply_binary_platt(0.001, PlattParams.identity()) == pytest.approx(BINARY_PROB_MIN, abs=1e-12)

    def test_identity_above_clamp_pulls_to_max(self):
        assert apply_binary_platt(0.999, PlattParams.identity()) == pytest.approx(BINARY_PROB_MAX, abs=1e-12)

    def test_non_identity_no_cap_matches_sigmoid_logit(self):
        # apply with bias=0, slope=2 should produce sigmoid(2 * logit(p)).
        params = PlattParams(bias=0.0, slope=2.0)
        got = apply_binary_platt(0.7, params)
        expected = _sigmoid(2.0 * _logit(0.7))
        assert got == pytest.approx(expected, abs=1e-6)
        # Sanity check: expected ≈ 0.8448.
        assert got == pytest.approx(0.8448, abs=1e-3)

    def test_slope_two_at_half_stays_at_half(self):
        # logit(0.5)=0, slope * 0 = 0, sigmoid(0)=0.5.
        got = apply_binary_platt(0.5, PlattParams(bias=0.0, slope=2.0))
        assert got == pytest.approx(0.5, abs=1e-12)

    def test_cap_binds_upward(self):
        # bias=2 → uncapped sigmoid(2) ≈ 0.881; cap at +0.1 from p=0.5 gives 0.6.
        got = apply_binary_platt(0.5, PlattParams(bias=2.0, slope=1.0), max_abs_deviation=0.1)
        assert got == pytest.approx(0.6, abs=1e-12)

    def test_cap_binds_downward(self):
        got = apply_binary_platt(0.5, PlattParams(bias=-2.0, slope=1.0), max_abs_deviation=0.1)
        assert got == pytest.approx(0.4, abs=1e-12)

    def test_cap_does_not_bind_when_loose(self):
        # bias=0.1, slope=1, p=0.5 → sigmoid(0.1) ≈ 0.5249. Cap of 0.5 is well past that.
        got = apply_binary_platt(0.5, PlattParams(bias=0.1, slope=1.0), max_abs_deviation=0.5)
        expected = _sigmoid(0.1)
        assert got == pytest.approx(expected, abs=1e-6)

    def test_cap_zero_means_raw_passes_through(self):
        # max_abs_deviation=0 forces p_adj into [p_raw, p_raw] → exactly p_raw, then binary clamp.
        got = apply_binary_platt(0.5, PlattParams(bias=2.0, slope=1.0), max_abs_deviation=0.0)
        assert got == pytest.approx(0.5, abs=1e-12)

    def test_negative_cap_raises(self):
        with pytest.raises(ValueError, match="max_abs_deviation"):
            apply_binary_platt(0.5, PlattParams(bias=0.0, slope=1.0), max_abs_deviation=-0.01)

    def test_nan_p_raw_raises(self):
        with pytest.raises(ValueError, match="p_raw must be finite"):
            apply_binary_platt(math.nan, PlattParams.identity())

    def test_inf_p_raw_raises(self):
        with pytest.raises(ValueError, match="p_raw must be finite"):
            apply_binary_platt(math.inf, PlattParams.identity())

    def test_nan_cap_raises(self):
        with pytest.raises(ValueError, match="max_abs_deviation"):
            apply_binary_platt(0.5, PlattParams.identity(), max_abs_deviation=math.nan)

    def test_final_clamp_respected_with_loose_cap(self):
        # bias=10, slope=1, p=0.5 → sigmoid(10) ≈ 0.99995. Cap=1.0 doesn't bind.
        # Final clamp must pin to BINARY_PROB_MAX = 0.98.
        got = apply_binary_platt(0.5, PlattParams(bias=10.0, slope=1.0), max_abs_deviation=1.0)
        assert got == pytest.approx(BINARY_PROB_MAX, abs=1e-12)


class TestApplyMCPlatt:
    @staticmethod
    def _make_options(probs: list[tuple[str, float]]) -> PredictedOptionList:
        return PredictedOptionList(
            predicted_options=[PredictedOption(option_name=name, probability=p) for name, p in probs]
        )

    def test_identity_preserves_ordering_and_renormalizes(self):
        opts = self._make_options([("A", 0.5), ("B", 0.3), ("C", 0.2)])
        result = apply_mc_platt(opts, PlattParams.identity())
        names = [o.option_name for o in result.predicted_options]
        assert names == ["A", "B", "C"]
        total = sum(o.probability for o in result.predicted_options)
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_two_option_matches_per_option_binary_then_renormalize(self):
        params = PlattParams(bias=0.5, slope=1.5)
        raw_a, raw_b = 0.6, 0.4
        opts = self._make_options([("A", raw_a), ("B", raw_b)])
        result = apply_mc_platt(opts, params)

        # Reference: apply binary helper to each option, then re-normalize+clamp via MC bounds.
        ref_a = apply_binary_platt(raw_a, params)
        ref_b = apply_binary_platt(raw_b, params)
        # Mirror clamp_and_renormalize_mc: clamp to MC bounds, then divide by sum.
        ref_a_clamped = max(MC_PROB_MIN, min(MC_PROB_MAX, ref_a))
        ref_b_clamped = max(MC_PROB_MIN, min(MC_PROB_MAX, ref_b))
        total = ref_a_clamped + ref_b_clamped
        expected_a = ref_a_clamped / total
        expected_b = ref_b_clamped / total

        got_a = result.predicted_options[0].probability
        got_b = result.predicted_options[1].probability
        assert got_a == pytest.approx(expected_a, abs=1e-9)
        assert got_b == pytest.approx(expected_b, abs=1e-9)

    def test_four_option_matches_per_option_binary_then_renormalize(self):
        params = PlattParams(bias=-0.3, slope=1.2)
        raw = [("A", 0.4), ("B", 0.3), ("C", 0.2), ("D", 0.1)]
        opts = self._make_options(raw)
        result = apply_mc_platt(opts, params)

        # Build expected option-by-option.
        per_option = [apply_binary_platt(p, params) for _, p in raw]
        clamped = [max(MC_PROB_MIN, min(MC_PROB_MAX, q)) for q in per_option]
        total = sum(clamped)
        expected = [q / total for q in clamped]

        for option, exp in zip(result.predicted_options, expected):
            assert option.probability == pytest.approx(exp, abs=1e-9)
        # Sum to 1 invariant.
        assert sum(o.probability for o in result.predicted_options) == pytest.approx(1.0, abs=1e-9)

    def test_returns_same_object_reference(self):
        opts = self._make_options([("A", 0.5), ("B", 0.5)])
        result = apply_mc_platt(opts, PlattParams.identity())
        assert result is opts

    def test_sum_to_one_invariant_with_non_identity_and_cap(self):
        params = PlattParams(bias=1.5, slope=1.4)
        opts = self._make_options([("A", 0.25), ("B", 0.25), ("C", 0.25), ("D", 0.25)])
        result = apply_mc_platt(opts, params, max_abs_deviation=0.05)
        total = sum(o.probability for o in result.predicted_options)
        assert total == pytest.approx(1.0, abs=1e-9)


class TestFitPlatt:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            fit_platt([0.1, 0.2, 0.3, 0.4, 0.5], [True, False, True, False])

    def test_too_few_observations_raises(self):
        with pytest.raises(ValueError, match="Too few"):
            fit_platt([0.1, 0.5, 0.9, 0.5], [True, False, True, False])

    def test_all_true_outcomes_raises(self):
        with pytest.raises(ValueError, match="single-class"):
            fit_platt([0.1, 0.2, 0.3, 0.4, 0.5], [True, True, True, True, True])

    def test_all_false_outcomes_raises(self):
        with pytest.raises(ValueError, match="single-class"):
            fit_platt([0.1, 0.2, 0.3, 0.4, 0.5], [False, False, False, False, False])

    def test_synthetic_recovery(self):
        # Generate from a known affine in log-odds space and check the fit recovers it.
        # Tolerance ±0.2: at n=2000 with U(0.05, 0.95) raw probs and Bernoulli sampling,
        # the fitted (bias, slope) lands within roughly ±0.1 of the truth in practice;
        # ±0.2 is a comfortable noise-floor band that doesn't flake.
        rng = np.random.default_rng(42)
        n = 2000
        true_bias = 0.5
        true_slope = 1.5

        raw_probs = rng.uniform(0.05, 0.95, size=n)
        true_logits = true_bias + true_slope * np.log(raw_probs / (1.0 - raw_probs))
        true_probs = 1.0 / (1.0 + np.exp(-true_logits))
        outcomes = (rng.uniform(size=n) < true_probs).tolist()

        params = fit_platt(raw_probs.tolist(), outcomes)
        assert params.bias == pytest.approx(true_bias, abs=0.2)
        assert params.slope == pytest.approx(true_slope, abs=0.2)

    def test_well_calibrated_recovers_identity(self):
        # If outcomes are sampled from the raw probs themselves (well-calibrated), the
        # MLE should land near (bias=0, slope=1). Tolerance 0.3 — Bernoulli noise at n=2000
        # routinely shifts each parameter by up to ~0.15 even when the truth is identity.
        rng = np.random.default_rng(123)
        n = 2000
        raw_probs = rng.uniform(0.05, 0.95, size=n)
        outcomes = (rng.uniform(size=n) < raw_probs).tolist()

        params = fit_platt(raw_probs.tolist(), outcomes)
        assert abs(params.bias) < 0.3
        assert abs(params.slope - 1.0) < 0.3

    def test_anti_correlated_forecaster_raises(self):
        # Outcomes are inverted relative to the predicted probability — fit slope should be
        # non-positive and trigger the "non-positive slope" or "anti-correlated" guard.
        rng = np.random.default_rng(7)
        n = 200
        raw_probs = rng.uniform(0.05, 0.95, size=n)
        outcomes = (raw_probs < 0.5).tolist()
        with pytest.raises(ValueError, match="non-positive slope|anti-correlated"):
            fit_platt(raw_probs.tolist(), outcomes)

    def test_uninformative_forecaster_raises_below_slope_threshold(self):
        # Probs are uniform in (0.05, 0.95) and outcomes are coin flips independent of them.
        # The fitted slope sits near zero — either non-positive (anti-correlated) or below
        # MIN_INFORMATIVE_SLOPE; either way fit_platt must refuse.
        rng = np.random.default_rng(99)
        n = 500
        raw_probs = rng.uniform(0.05, 0.95, size=n)
        outcomes = (rng.uniform(size=n) < 0.5).tolist()
        with pytest.raises(ValueError, match=str(MIN_INFORMATIVE_SLOPE) + "|non-positive slope|no signal"):
            fit_platt(raw_probs.tolist(), outcomes)
