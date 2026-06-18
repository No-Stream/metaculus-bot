from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from metaculus_bot.prob_math_utils import logit, sigmoid
from metaculus_bot.probabilistic_tools.base_rate import beta_binomial_update
from metaculus_bot.probabilistic_tools.binary_pooling import (
    adaptive_weight,
    overconfidence_divergence,
    pool_binary,
    reconstruct_p_math,
)


@dataclass
class _Evidence:
    """Duck-typed stand-in for the structured-block evidence items.

    Mirrors the attributes consumed by run_pdf.py:_apply_evidence_lr:
    .direction ("up"/"down"/"neutral"), .strength ("weak"/"moderate"/"strong"),
    .likelihood_ratio (float | None — explicit override).
    """

    direction: str
    strength: str
    likelihood_ratio: float | None = None


class TestPoolBinary:
    def test_w_zero_returns_p_model(self):
        assert pool_binary(p_model=0.3, p_math=0.8, w=0.0) == pytest.approx(0.3)

    def test_w_one_returns_p_math(self):
        assert pool_binary(p_model=0.3, p_math=0.8, w=1.0) == pytest.approx(0.8)

    def test_w_half_is_sigmoid_of_mean_logit(self):
        p_model, p_math = 0.3, 0.8
        expected = sigmoid(0.5 * logit(p_math) + 0.5 * logit(p_model))
        assert pool_binary(p_model=p_model, p_math=p_math, w=0.5) == pytest.approx(expected)

    def test_general_weight_is_logit_linear_combination(self):
        p_model, p_math, w = 0.4, 0.65, 0.3
        expected = sigmoid(w * logit(p_math) + (1.0 - w) * logit(p_model))
        assert pool_binary(p_model=p_model, p_math=p_math, w=w) == pytest.approx(expected)

    def test_equal_inputs_return_that_value(self):
        assert pool_binary(p_model=0.42, p_math=0.42, w=0.5) == pytest.approx(0.42)

    def test_extreme_inputs_do_not_produce_inf_or_nan(self):
        result = pool_binary(p_model=0.0, p_math=1.0, w=0.5)
        assert math.isfinite(result)
        # Clamped logits are symmetric, so the midpoint sits at 0.5.
        assert result == pytest.approx(0.5)

    def test_extreme_p_model_zero_stays_finite(self):
        result = pool_binary(p_model=0.0, p_math=0.0, w=0.7)
        assert math.isfinite(result)
        assert 0.0 < result < 1.0

    def test_rejects_weight_below_zero(self):
        with pytest.raises(ValueError):
            pool_binary(p_model=0.3, p_math=0.8, w=-0.1)

    def test_rejects_weight_above_one(self):
        with pytest.raises(ValueError):
            pool_binary(p_model=0.3, p_math=0.8, w=1.1)


class TestOverconfidenceDivergence:
    def test_zero_when_equal(self):
        assert overconfidence_divergence(p_model=0.7, p_math=0.7) == pytest.approx(0.0)

    def test_symmetric(self):
        d1 = overconfidence_divergence(p_model=0.2, p_math=0.6)
        d2 = overconfidence_divergence(p_model=0.6, p_math=0.2)
        assert d1 == pytest.approx(d2)

    def test_is_absolute_logit_gap(self):
        expected = abs(logit(0.9) - logit(0.5))
        assert overconfidence_divergence(p_model=0.5, p_math=0.9) == pytest.approx(expected)

    def test_extreme_inputs_finite(self):
        assert math.isfinite(overconfidence_divergence(p_model=0.0, p_math=1.0))


class TestReconstructPMath:
    def test_no_evidence_returns_base(self):
        assert reconstruct_p_math(base_prob=0.42, evidence_items=[]) == pytest.approx(0.42)

    def test_synthetic_evidence_matches_hand_computed_log_odds(self):
        # base 0.3; one up/moderate (LR=3.0); one down/weak (LR=1.5 -> 1/1.5).
        # log_odds = log(0.3/0.7) + log(3.0) + log(1/1.5)
        #          = log(0.3/0.7 * 3.0/1.5) = log(0.857142857) -> p = 0.461538461
        evidence = [
            _Evidence(direction="up", strength="moderate"),
            _Evidence(direction="down", strength="weak"),
        ]
        result = reconstruct_p_math(base_prob=0.3, evidence_items=evidence)
        assert result == pytest.approx(0.461538461, abs=1e-6)

    def test_neutral_evidence_is_skipped(self):
        evidence = [_Evidence(direction="neutral", strength="strong")]
        assert reconstruct_p_math(base_prob=0.3, evidence_items=evidence) == pytest.approx(0.3)

    def test_explicit_likelihood_ratio_overrides_strength(self):
        # Explicit LR=4.0 should win over the "weak"->1.5 strength mapping.
        evidence = [_Evidence(direction="up", strength="weak", likelihood_ratio=4.0)]
        log_odds = math.log(0.3 / 0.7) + math.log(4.0)
        expected = 1.0 / (1.0 + math.exp(-log_odds))
        assert reconstruct_p_math(base_prob=0.3, evidence_items=evidence) == pytest.approx(expected)

    def test_down_direction_inverts_likelihood_ratio(self):
        evidence = [_Evidence(direction="down", strength="strong")]  # LR=10 -> 1/10
        log_odds = math.log(0.3 / 0.7) + math.log(1.0 / 10.0)
        expected = 1.0 / (1.0 + math.exp(-log_odds))
        assert reconstruct_p_math(base_prob=0.3, evidence_items=evidence) == pytest.approx(expected)

    def test_extreme_base_prob_clamped_not_inf(self):
        # base 0.0 is clamped to 0.001 before taking log-odds (matches run_pdf).
        evidence = [_Evidence(direction="up", strength="moderate")]
        result = reconstruct_p_math(base_prob=0.0, evidence_items=evidence)
        assert math.isfinite(result)
        log_odds = math.log(0.001 / 0.999) + math.log(3.0)
        expected = 1.0 / (1.0 + math.exp(-log_odds))
        assert result == pytest.approx(expected)

    def test_base_rate_counts_anchor_via_beta_binomial(self):
        # When base_rate_counts=(k, n) is supplied, the anchor is the
        # beta-binomial posterior mean rather than the raw base_prob.
        posterior = beta_binomial_update(k=3, n=12).posterior_mean
        evidence = [_Evidence(direction="up", strength="moderate")]
        expected = reconstruct_p_math(base_prob=posterior, evidence_items=evidence)
        result = reconstruct_p_math(base_prob=0.5, evidence_items=evidence, base_rate_counts=(3, 12))
        assert result == pytest.approx(expected)


class TestAdaptiveWeight:
    def test_zero_divergence_gives_zero_weight(self):
        assert adaptive_weight(0.0) == pytest.approx(0.0)

    def test_monotone_nondecreasing_in_divergence(self):
        weights = [adaptive_weight(d) for d in (0.0, 0.5, 1.0, 2.0, 4.0)]
        assert all(b >= a for a, b in zip(weights, weights[1:]))

    def test_clamped_to_max_weight(self):
        assert adaptive_weight(1e6, max_weight=0.5) == pytest.approx(0.5)

    def test_below_threshold_is_zero(self):
        assert adaptive_weight(0.3, threshold=1.0) == pytest.approx(0.0)

    def test_rejects_max_weight_out_of_range(self):
        with pytest.raises(ValueError):
            adaptive_weight(1.0, max_weight=1.5)
