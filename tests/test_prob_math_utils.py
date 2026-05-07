"""Tests for the shared logit/sigmoid/clamp utilities."""

from __future__ import annotations

import math

import pytest

from metaculus_bot.prob_math_utils import PROB_CLAMP_EPS, clamp_prob, logit, sigmoid


class TestClampProb:
    def test_clamps_zero_to_eps(self):
        assert clamp_prob(0.0) == PROB_CLAMP_EPS

    def test_clamps_one_to_one_minus_eps(self):
        assert clamp_prob(1.0) == 1.0 - PROB_CLAMP_EPS

    def test_mid_probability_unchanged(self):
        assert clamp_prob(0.5) == 0.5

    def test_custom_eps_tighter(self):
        assert clamp_prob(0.0, eps=1e-6) == 1e-6

    def test_negative_input_clamped_to_eps(self):
        assert clamp_prob(-0.25) == PROB_CLAMP_EPS


class TestLogit:
    def test_half_is_zero(self):
        assert logit(0.5) == 0.0

    def test_boundary_zero_is_finite_and_large_negative(self):
        y = logit(0.0)
        assert math.isfinite(y)
        assert y < -5.0

    def test_boundary_one_is_finite_and_large_positive(self):
        y = logit(1.0)
        assert math.isfinite(y)
        assert y > 5.0


class TestSigmoid:
    def test_zero_is_half(self):
        assert sigmoid(0.0) == 0.5

    def test_large_positive_does_not_overflow(self):
        got = sigmoid(1000.0)
        assert math.isfinite(got)
        assert got == pytest.approx(1.0, abs=1e-9)

    def test_large_negative_does_not_overflow(self):
        got = sigmoid(-1000.0)
        assert math.isfinite(got)
        assert got == pytest.approx(0.0, abs=1e-9)


class TestRoundTrip:
    @pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
    def test_sigmoid_of_logit_roundtrips(self, p: float):
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-9)
