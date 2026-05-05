from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from metaculus_bot.numeric_config import STANDARD_PERCENTILES
from metaculus_bot.probabilistic_tools import (
    DEFAULT_INFORMATIVE_PRIOR_STRENGTH,
    BetaBinomialResult,
    ConsistencyResult,
    LognormalFit,
    NormalFit,
    StudentTFit,
    SurvivalResult,
    TailMassResult,
    base_rate_blend,
    base_rate_to_hazard,
    bayes_from_likelihoods,
    beta_binomial_ceiling_percentiles,
    beta_binomial_update,
    cdf_at_threshold,
    dirichlet_with_other,
    fit_lognormal_from_percentiles,
    fit_normal_from_percentiles,
    fit_student_t_from_percentiles,
    fit_to_11_percentiles,
    implied_likelihood_ratio,
    inverse_variance_pool,
    laplace_rule_of_succession,
    linear_pool,
    linear_pool_options,
    log_pool,
    negative_binomial_percentiles,
    out_of_bounds_mass,
    percentile_family_consistency,
    percentile_monotonicity_check,
    percentiles_to_metaculus_cdf,
    poisson_at_least_one,
    poisson_percentiles,
    prob_event_before,
    satopaa_extremize,
    stated_base_rate_consistency,
    weibull_prob_event_before,
)


class TestBetaBinomialUpdate:
    def test_default_weakly_informative_prior_posterior_mean(self):
        # Default prior_mean=0.5, prior_strength=1.0 → α=β=0.5.
        # Posterior: α=0.5+3=3.5, β=0.5+9=9.5, mean=3.5/13.
        r = beta_binomial_update(k=3, n=12)
        assert isinstance(r, BetaBinomialResult)
        assert r.posterior_alpha == pytest.approx(3.5)
        assert r.posterior_beta == pytest.approx(9.5)
        assert r.posterior_mean == pytest.approx(3.5 / 13.0, rel=1e-9)

    def test_informative_prior_pulls_toward_prior_mean(self):
        # Stated outside view 0.15 with strength 5 → α=0.75, β=4.25.
        # k/n = 3/12, posterior α=3.75, β=13.25, mean ≈ 0.2206.
        r = beta_binomial_update(k=3, n=12, prior_mean=0.15, prior_strength=DEFAULT_INFORMATIVE_PRIOR_STRENGTH)
        assert r.posterior_alpha == pytest.approx(0.75 + 3.0)
        assert r.posterior_beta == pytest.approx(4.25 + 9.0)
        assert r.posterior_mean == pytest.approx((0.75 + 3.0) / (5.0 + 12.0), rel=1e-9)

    def test_n_equal_zero_returns_prior_mean(self):
        # With no data, posterior mean equals prior mean.
        r = beta_binomial_update(k=0, n=0, prior_mean=0.4, prior_strength=5.0)
        assert r.posterior_alpha == pytest.approx(2.0)
        assert r.posterior_beta == pytest.approx(3.0)
        assert r.posterior_mean == pytest.approx(0.4)

    def test_ci_ordering(self):
        r = beta_binomial_update(k=5, n=20)
        assert r.ci_90_low <= r.ci_80_low <= r.posterior_mean <= r.ci_80_high <= r.ci_90_high

    def test_ci_matches_scipy(self):
        r = beta_binomial_update(k=7, n=15, prior_mean=0.5, prior_strength=2.0)
        a, b = r.posterior_alpha, r.posterior_beta
        assert r.ci_80_low == pytest.approx(stats.beta.ppf(0.10, a, b), rel=1e-10)
        assert r.ci_80_high == pytest.approx(stats.beta.ppf(0.90, a, b), rel=1e-10)
        assert r.ci_90_low == pytest.approx(stats.beta.ppf(0.05, a, b), rel=1e-10)
        assert r.ci_90_high == pytest.approx(stats.beta.ppf(0.95, a, b), rel=1e-10)

    def test_invalid_k_gt_n_raises(self):
        with pytest.raises(ValueError, match="k"):
            beta_binomial_update(k=5, n=3)

    def test_invalid_negative_n_raises(self):
        with pytest.raises(ValueError, match="n"):
            beta_binomial_update(k=0, n=-1)

    def test_invalid_negative_k_raises(self):
        with pytest.raises(ValueError, match="k"):
            beta_binomial_update(k=-1, n=5)

    def test_invalid_prior_mean_zero_raises(self):
        with pytest.raises(ValueError, match="prior_mean"):
            beta_binomial_update(k=1, n=2, prior_mean=0.0)

    def test_invalid_prior_mean_one_raises(self):
        with pytest.raises(ValueError, match="prior_mean"):
            beta_binomial_update(k=1, n=2, prior_mean=1.0)

    def test_invalid_prior_strength_raises(self):
        with pytest.raises(ValueError, match="prior_strength"):
            beta_binomial_update(k=1, n=2, prior_strength=-0.5)


class TestLaplaceRuleOfSuccession:
    def test_zero_of_ten(self):
        assert laplace_rule_of_succession(0, 10) == pytest.approx(1.0 / 12.0)

    def test_all_successes(self):
        assert laplace_rule_of_succession(5, 5) == pytest.approx(6.0 / 7.0)

    def test_zero_zero(self):
        assert laplace_rule_of_succession(0, 0) == pytest.approx(0.5)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            laplace_rule_of_succession(5, 3)
        with pytest.raises(ValueError):
            laplace_rule_of_succession(-1, 10)
        with pytest.raises(ValueError):
            laplace_rule_of_succession(0, -1)


class TestBaseRateBlend:
    def test_linear_equal_weights(self):
        assert base_rate_blend([0.2, 0.4, 0.6]) == pytest.approx(0.4)

    def test_linear_custom_weights(self):
        got = base_rate_blend([0.1, 0.9], weights=[1.0, 3.0])
        assert got == pytest.approx(0.7)

    def test_log_odds(self):
        rates = [0.2, 0.5, 0.8]
        got = base_rate_blend(rates, method="log_odds")
        logits = [math.log(p / (1 - p)) for p in rates]
        expected = 1.0 / (1.0 + math.exp(-sum(logits) / len(logits)))
        assert got == pytest.approx(expected)

    def test_inverse_variance(self):
        got = base_rate_blend(
            [0.2, 0.8],
            method="inverse_variance",
            variances=[0.01, 0.04],
        )
        w = [1 / 0.01, 1 / 0.04]
        expected = (w[0] * 0.2 + w[1] * 0.8) / sum(w)
        assert got == pytest.approx(expected)

    def test_inverse_variance_missing_variances_raises(self):
        with pytest.raises(ValueError, match="variances"):
            base_rate_blend([0.2, 0.3], method="inverse_variance")

    def test_inverse_variance_with_weights_raises(self):
        # F10: passing both weights and variances should error rather than
        # silently dropping the weights.
        with pytest.raises(ValueError, match="either weights OR variances"):
            base_rate_blend(
                [0.1, 0.2],
                method="inverse_variance",
                weights=[1.0, 1.0],
                variances=[0.01, 0.01],
            )

    def test_rate_out_of_range_raises(self):
        with pytest.raises(ValueError, match="rates"):
            base_rate_blend([0.2, 1.5])

    def test_weights_length_mismatch(self):
        with pytest.raises(ValueError, match="length"):
            base_rate_blend([0.2, 0.3], weights=[1.0])

    def test_weights_sum_zero_raises(self):
        with pytest.raises(ValueError, match="sum"):
            base_rate_blend([0.2, 0.3], weights=[0.0, 0.0])

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match=">="):
            base_rate_blend([0.2, 0.3], weights=[-1.0, 2.0])

    def test_empty_rates_raises(self):
        with pytest.raises(ValueError):
            base_rate_blend([])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            base_rate_blend([0.2, 0.3], method="nope")  # ty: ignore[invalid-argument-type]  # type: ignore[arg-type]


class TestImpliedLikelihoodRatio:
    def test_known_case(self):
        assert implied_likelihood_ratio(0.1, 0.5) == pytest.approx(9.0, rel=1e-9)

    def test_no_change_is_one(self):
        assert implied_likelihood_ratio(0.3, 0.3) == pytest.approx(1.0, rel=1e-9)

    def test_posterior_less_than_prior(self):
        assert implied_likelihood_ratio(0.5, 0.1) == pytest.approx(1.0 / 9.0, rel=1e-9)

    def test_prior_zero_raises(self):
        # LR genuinely undefined when prior is at the 0/1 boundary.
        with pytest.raises(ValueError, match="prior_prob"):
            implied_likelihood_ratio(0.0, 0.5)

    def test_prior_one_raises(self):
        with pytest.raises(ValueError, match="prior_prob"):
            implied_likelihood_ratio(1.0, 0.5)

    def test_posterior_zero_raises(self):
        with pytest.raises(ValueError, match="posterior_prob"):
            implied_likelihood_ratio(0.5, 0.0)

    def test_posterior_one_raises(self):
        with pytest.raises(ValueError, match="posterior_prob"):
            implied_likelihood_ratio(0.5, 1.0)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            implied_likelihood_ratio(-0.1, 0.5)
        with pytest.raises(ValueError):
            implied_likelihood_ratio(0.1, 1.1)


class TestBayesFromLikelihoods:
    def test_known_case(self):
        got = bayes_from_likelihoods(0.1, 0.8, 0.2)
        expected = (0.8 * 0.1) / (0.8 * 0.1 + 0.2 * 0.9)
        assert got == pytest.approx(expected)

    def test_equal_likelihoods_no_update(self):
        assert bayes_from_likelihoods(0.3, 0.5, 0.5) == pytest.approx(0.3)

    def test_both_likelihoods_zero_raises(self):
        with pytest.raises(ValueError):
            bayes_from_likelihoods(0.3, 0.0, 0.0)

    def test_prior_at_zero_boundary_raises(self):
        # F11: prior=0 is a degenerate boundary — reject rather than returning 0.
        with pytest.raises(ValueError, match="prior_prob"):
            bayes_from_likelihoods(0.0, 0.5, 0.5)

    def test_prior_at_one_boundary_raises(self):
        # F11: prior=1 with p_e_given_h=0 silently returned 0 before the fix,
        # masking a contradictory-evidence scenario. Now must raise.
        with pytest.raises(ValueError, match="prior_prob"):
            bayes_from_likelihoods(1.0, 0.1, 0.1)

    def test_invalid_probs_raise(self):
        with pytest.raises(ValueError):
            bayes_from_likelihoods(-0.1, 0.5, 0.5)
        with pytest.raises(ValueError):
            bayes_from_likelihoods(0.5, 1.1, 0.5)
        with pytest.raises(ValueError):
            bayes_from_likelihoods(0.5, 0.5, -0.2)


class TestProbEventBefore:
    def test_conditional_known_value(self):
        r = prob_event_before(hazard_rate=0.25, elapsed_fraction=0.33, remaining_fraction=0.67)
        assert isinstance(r, SurvivalResult)
        assert r.conditional_prob_given_no_event_yet == pytest.approx(1.0 - math.exp(-0.25 * 0.67), rel=1e-10)

    def test_elapsed_window_prob_implied_by_total_minus_remaining(self):
        # We no longer expose elapsed_window_prob on SurvivalResult; verify the
        # identity P(elapsed) = P(total) - P(remaining) + P(total)*P(remaining)
        # holds via the two surfaced probabilities. For a constant-hazard
        # process, P(elapsed) = 1 - exp(-λ * elapsed_fraction).
        r = prob_event_before(hazard_rate=0.5, elapsed_fraction=0.4, remaining_fraction=0.6)
        # Derive elapsed-window prob: S(elapsed) = (1 - P(total)) / (1 - P(remaining))
        # so P(elapsed) = 1 - S(elapsed).
        s_elapsed = (1.0 - r.unconditional_prob) / (1.0 - r.conditional_prob_given_no_event_yet)
        p_elapsed = 1.0 - s_elapsed
        assert p_elapsed == pytest.approx(1.0 - math.exp(-0.5 * 0.4), rel=1e-10)

    def test_unconditional_matches_total(self):
        r = prob_event_before(hazard_rate=0.3, elapsed_fraction=0.5, remaining_fraction=0.5)
        assert r.unconditional_prob == pytest.approx(1.0 - math.exp(-0.3 * 1.0))

    def test_unit_duration_scaling(self):
        r = prob_event_before(
            hazard_rate=1.0,
            elapsed_fraction=0.0,
            remaining_fraction=1.0,
            window_length=0.25,
        )
        assert r.conditional_prob_given_no_event_yet == pytest.approx(1.0 - math.exp(-0.25))

    def test_hazard_units_lockdown_day_rate_30day_window(self):
        # Contract: units cancel. rate 0.25/day over 30-day window, 67%
        # remaining → P(event in remaining | none yet) = 1 - exp(-0.25 * 30 * 0.67)
        # ≈ 0.993. Previously the runner converted 30 days → years (~0.082)
        # giving a ~3-order-of-magnitude wrong hazard exposure.
        r = prob_event_before(
            hazard_rate=0.25,
            elapsed_fraction=0.33,
            remaining_fraction=0.67,
            window_length=30.0,
        )
        expected = 1.0 - math.exp(-0.25 * 30.0 * 0.67)
        assert r.conditional_prob_given_no_event_yet == pytest.approx(expected, rel=1e-10)
        # Sanity check on the exposure scale — pre-fix this would have been
        # order 0.04, orders of magnitude off.
        assert r.conditional_prob_given_no_event_yet > 0.95

    def test_zero_hazard(self):
        r = prob_event_before(0.0, 0.5, 0.5)
        assert r.unconditional_prob == 0.0
        assert r.conditional_prob_given_no_event_yet == 0.0

    def test_invalid_hazard_raises(self):
        with pytest.raises(ValueError):
            prob_event_before(-1.0, 0.5, 0.5)

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            prob_event_before(0.5, 1.5, 0.5)
        with pytest.raises(ValueError):
            prob_event_before(0.5, 0.5, -0.1)

    def test_invalid_unit_duration(self):
        with pytest.raises(ValueError):
            prob_event_before(0.5, 0.5, 0.5, window_length=0)


class TestWeibullProbEventBefore:
    def test_matches_scipy(self):
        scale, shape, t = 10.0, 1.5, 5.0
        got = weibull_prob_event_before(scale=scale, shape=shape, t=t)
        expected = stats.weibull_min.cdf(t, c=shape, scale=scale)
        assert got == pytest.approx(expected, rel=1e-10)

    def test_shape_one_reduces_to_exponential(self):
        got = weibull_prob_event_before(scale=2.0, shape=1.0, t=1.0)
        assert got == pytest.approx(1.0 - math.exp(-0.5))

    def test_t_zero_is_zero(self):
        assert weibull_prob_event_before(scale=2.0, shape=1.5, t=0.0) == 0.0

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            weibull_prob_event_before(scale=0, shape=1.0, t=1.0)
        with pytest.raises(ValueError):
            weibull_prob_event_before(scale=1.0, shape=-1.0, t=1.0)
        with pytest.raises(ValueError):
            weibull_prob_event_before(scale=1.0, shape=1.0, t=-1.0)


class TestPoissonAtLeastOne:
    def test_known_value(self):
        assert poisson_at_least_one(0.5) == pytest.approx(1.0 - math.exp(-0.5))

    def test_zero(self):
        assert poisson_at_least_one(0.0) == 0.0

    def test_large(self):
        assert poisson_at_least_one(20.0) == pytest.approx(1.0 - math.exp(-20.0))

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            poisson_at_least_one(-0.1)


class TestBaseRateToHazard:
    def test_jeffreys_default_posterior_mean(self):
        # Default Jeffreys-ish Gamma(0.5, 0): posterior mean = (0.5 + k) / (n * years).
        assert base_rate_to_hazard(k=3, n=12, years_per_ref_period=1.0) == pytest.approx(3.5 / 12.0)

    def test_scaling_with_ref_period(self):
        assert base_rate_to_hazard(k=3, n=12, years_per_ref_period=2.0) == pytest.approx(3.5 / 24.0)

    def test_zero_events_jeffreys_nonzero(self):
        # Jeffreys-ish gives 0.5/n instead of zero — rare-event uncertainty stays visible.
        assert base_rate_to_hazard(k=0, n=10) == pytest.approx(0.05)

    def test_informative_prior_pulls_toward_prior_rate(self):
        # Prior rate 0.5/yr with strength 10 → α=5.0, β=10.
        # Data 3/12 years → posterior mean = (5 + 3) / (10 + 12) = 8/22.
        got = base_rate_to_hazard(k=3, n=12, prior_rate=0.5, prior_strength=10.0)
        assert got == pytest.approx(8.0 / 22.0)

    def test_informative_prior_no_data_returns_prior_rate(self):
        got = base_rate_to_hazard(k=0, n=1, prior_rate=0.3, prior_strength=1000.0)
        # With huge prior strength, posterior ≈ prior_rate.
        assert got == pytest.approx(0.3, rel=1e-2)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            base_rate_to_hazard(k=-1, n=10)
        with pytest.raises(ValueError):
            base_rate_to_hazard(k=0, n=0)
        with pytest.raises(ValueError):
            base_rate_to_hazard(k=1, n=5, years_per_ref_period=0.0)
        with pytest.raises(ValueError):
            base_rate_to_hazard(k=1, n=5, prior_rate=-0.1)
        with pytest.raises(ValueError):
            base_rate_to_hazard(k=1, n=5, prior_strength=0.0)


class TestLinearPool:
    def test_equal_weights_matches_mean(self):
        probs = [0.1, 0.3, 0.5, 0.7]
        assert linear_pool(probs) == pytest.approx(float(np.mean(probs)))

    def test_custom_weights(self):
        got = linear_pool([0.2, 0.8], weights=[1.0, 3.0])
        assert got == pytest.approx((0.2 + 3 * 0.8) / 4)

    def test_invalid_prob_raises(self):
        with pytest.raises(ValueError):
            linear_pool([0.2, 1.5])

    def test_invalid_weight_length(self):
        with pytest.raises(ValueError):
            linear_pool([0.2, 0.3], weights=[1.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            linear_pool([])


class TestLogPool:
    def test_known_case_equal_weights(self):
        probs = [0.2, 0.5, 0.8]
        logits = [math.log(p / (1 - p)) for p in probs]
        expected = 1.0 / (1.0 + math.exp(-sum(logits) / len(logits)))
        assert log_pool(probs) == pytest.approx(expected, rel=1e-12)

    def test_symmetric_around_half(self):
        assert log_pool([0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_weighted(self):
        probs = [0.2, 0.8]
        w = [1.0, 3.0]
        logits = [math.log(p / (1 - p)) for p in probs]
        avg = (w[0] * logits[0] + w[1] * logits[1]) / sum(w)
        expected = 1.0 / (1.0 + math.exp(-avg))
        assert log_pool(probs, weights=w) == pytest.approx(expected)

    def test_clamps_extreme_probs(self):
        got = log_pool([0.0, 1.0])
        assert 0.0 < got < 1.0

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            log_pool([0.2, -0.1])


class TestSatopaaExtremize:
    def test_extremizes_above_max(self):
        probs = [0.7, 0.8, 0.9]
        got = satopaa_extremize(probs, alpha=2.5)
        assert got > max(probs)

    def test_alpha_one_equals_log_pool(self):
        probs = [0.2, 0.5, 0.8]
        assert satopaa_extremize(probs, alpha=1.0) == pytest.approx(log_pool(probs))

    def test_symmetric_neutral(self):
        got = satopaa_extremize([0.5, 0.5, 0.5], alpha=2.5)
        assert got == pytest.approx(0.5)

    def test_extremizes_below_min_for_low_probs(self):
        probs = [0.1, 0.2, 0.3]
        got = satopaa_extremize(probs, alpha=2.5)
        assert got < min(probs)

    def test_alpha_below_one_raises(self):
        with pytest.raises(ValueError):
            satopaa_extremize([0.3, 0.5], alpha=0.5)


class TestInverseVariancePool:
    def test_known_case(self):
        means = [10.0, 12.0]
        variances = [1.0, 4.0]
        m, v = inverse_variance_pool(means, variances)
        w = [1.0, 0.25]
        expected_m = (w[0] * 10 + w[1] * 12) / sum(w)
        expected_v = 1.0 / sum(w)
        assert m == pytest.approx(expected_m)
        assert v == pytest.approx(expected_v)

    def test_equal_variance_collapses_to_mean(self):
        m, _ = inverse_variance_pool([1.0, 3.0, 5.0], [2.0, 2.0, 2.0])
        assert m == pytest.approx(3.0)

    def test_variance_reduces_with_more_estimates(self):
        _, v2 = inverse_variance_pool([1.0, 2.0], [1.0, 1.0])
        _, v3 = inverse_variance_pool([1.0, 2.0, 3.0], [1.0, 1.0, 1.0])
        assert v3 < v2

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            inverse_variance_pool([1.0, 2.0], [0.0, 1.0])
        with pytest.raises(ValueError):
            inverse_variance_pool([1.0], [1.0, 1.0])
        with pytest.raises(ValueError):
            inverse_variance_pool([], [])


class TestLinearPoolOptions:
    def test_uniform_pool(self):
        dicts = [{"A": 0.5, "B": 0.3, "C": 0.2}, {"A": 0.3, "B": 0.5, "C": 0.2}]
        got = linear_pool_options(dicts)
        assert set(got.keys()) == {"A", "B", "C"}
        assert got["A"] == pytest.approx(0.4)
        assert got["B"] == pytest.approx(0.4)
        assert got["C"] == pytest.approx(0.2)

    def test_weighted(self):
        dicts = [{"A": 1.0, "B": 0.0}, {"A": 0.0, "B": 1.0}]
        got = linear_pool_options(dicts, weights=[3.0, 1.0])
        assert got["A"] == pytest.approx(0.75)
        assert got["B"] == pytest.approx(0.25)

    def test_mismatched_keys_raises(self):
        with pytest.raises(ValueError, match="keys"):
            linear_pool_options([{"A": 1.0}, {"B": 1.0}])

    def test_sums_not_normalized_raises(self):
        with pytest.raises(ValueError, match="sum"):
            linear_pool_options([{"A": 0.3, "B": 0.3}])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            linear_pool_options([])


class TestFitNormalFromPercentiles:
    def test_round_trip(self):
        mu, sigma = 5.0, 2.0
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        pvs = {q: float(stats.norm.ppf(q, loc=mu, scale=sigma)) for q in quantiles}
        fit = fit_normal_from_percentiles(pvs)
        assert isinstance(fit, NormalFit)
        assert fit.mu == pytest.approx(mu, rel=1e-3)
        assert fit.sigma == pytest.approx(sigma, rel=1e-3)

    def test_few_points_still_fits(self):
        pvs = {0.1: -1.28, 0.5: 0.0, 0.9: 1.28}
        fit = fit_normal_from_percentiles(pvs)
        assert fit.mu == pytest.approx(0.0, abs=0.05)
        assert fit.sigma == pytest.approx(1.0, rel=0.05)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            fit_normal_from_percentiles({})

    def test_single_point_raises(self):
        with pytest.raises(ValueError):
            fit_normal_from_percentiles({0.5: 1.0})

    def test_invalid_percentile_key_raises(self):
        with pytest.raises(ValueError):
            fit_normal_from_percentiles({0.0: 1.0, 0.5: 2.0})
        with pytest.raises(ValueError):
            fit_normal_from_percentiles({0.5: 1.0, 1.0: 2.0})


class TestFitLognormalFromPercentiles:
    def test_round_trip(self):
        mu, sigma = 1.5, 0.5
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        pvs = {q: float(math.exp(stats.norm.ppf(q, loc=mu, scale=sigma))) for q in quantiles}
        fit = fit_lognormal_from_percentiles(pvs)
        assert isinstance(fit, LognormalFit)
        assert fit.mu == pytest.approx(mu, rel=1e-2)
        assert fit.sigma == pytest.approx(sigma, rel=1e-2)

    def test_non_positive_value_raises(self):
        with pytest.raises(ValueError, match="lognormal"):
            fit_lognormal_from_percentiles({0.1: 1.0, 0.5: 0.0, 0.9: 10.0})


class TestFitStudentTFromPercentiles:
    def test_round_trip(self):
        loc, scale, df = 0.0, 1.0, 5.0
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        pvs = {q: float(stats.t.ppf(q, df=df, loc=loc, scale=scale)) for q in quantiles}
        fit = fit_student_t_from_percentiles(pvs, df=df)
        assert isinstance(fit, StudentTFit)
        assert fit.loc == pytest.approx(loc, abs=0.05)
        assert fit.scale == pytest.approx(scale, rel=0.05)
        assert fit.df == pytest.approx(df)

    def test_invalid_df_raises(self):
        with pytest.raises(ValueError):
            fit_student_t_from_percentiles({0.25: -1.0, 0.75: 1.0}, df=0)


class TestOutOfBoundsMass:
    def test_normal_interior(self):
        fit = NormalFit(mu=10.0, sigma=2.0, method="least_squares_cdf")
        r = out_of_bounds_mass(fit, lower_bound=5.0, upper_bound=15.0)
        assert isinstance(r, TailMassResult)
        expected_interior = float(stats.norm.cdf(15.0, 10.0, 2.0) - stats.norm.cdf(5.0, 10.0, 2.0))
        assert r.interior_mass == pytest.approx(expected_interior)
        assert r.prob_below_min == pytest.approx(stats.norm.cdf(5.0, 10.0, 2.0))
        assert r.prob_above_max == pytest.approx(1.0 - stats.norm.cdf(15.0, 10.0, 2.0))
        assert r.interior_mass == pytest.approx(0.9876, abs=5e-4)

    def test_lognormal_bounds_transformed(self):
        fit = LognormalFit(mu=0.0, sigma=1.0, method="least_squares_cdf_logspace")
        r = out_of_bounds_mass(fit, lower_bound=0.5, upper_bound=3.0)
        expected_below = stats.norm.cdf(math.log(0.5), 0.0, 1.0)
        expected_above = 1.0 - stats.norm.cdf(math.log(3.0), 0.0, 1.0)
        assert r.prob_below_min == pytest.approx(expected_below)
        assert r.prob_above_max == pytest.approx(expected_above)

    def test_lognormal_lower_bound_nonpositive_clamps_to_zero(self):
        fit = LognormalFit(mu=0.0, sigma=1.0, method="least_squares_cdf_logspace")
        r = out_of_bounds_mass(fit, lower_bound=0.0, upper_bound=None)
        assert r.prob_below_min == 0.0

    def test_none_bounds(self):
        fit = NormalFit(mu=0.0, sigma=1.0, method="m")
        r = out_of_bounds_mass(fit, lower_bound=None, upper_bound=None)
        assert r.prob_below_min == 0.0
        assert r.prob_above_max == 0.0
        assert r.interior_mass == 1.0

    def test_bounds_ordering_validation(self):
        fit = NormalFit(mu=0.0, sigma=1.0, method="m")
        with pytest.raises(ValueError):
            out_of_bounds_mass(fit, lower_bound=5.0, upper_bound=1.0)

    def test_only_upper_bound(self):
        # Open-lower / closed-upper shape — common for bounded-above counts.
        fit = NormalFit(mu=0.0, sigma=1.0, method="m")
        r = out_of_bounds_mass(fit, lower_bound=None, upper_bound=2.0)
        assert r.prob_below_min == 0.0
        expected_above = 1.0 - float(stats.norm.cdf(2.0, 0.0, 1.0))
        assert r.prob_above_max == pytest.approx(expected_above)
        assert r.interior_mass == pytest.approx(1.0 - expected_above)

    def test_only_lower_bound(self):
        # Closed-lower / open-upper shape — common for bounded-below durations.
        fit = NormalFit(mu=0.0, sigma=1.0, method="m")
        r = out_of_bounds_mass(fit, lower_bound=-2.0, upper_bound=None)
        assert r.prob_above_max == 0.0
        expected_below = float(stats.norm.cdf(-2.0, 0.0, 1.0))
        assert r.prob_below_min == pytest.approx(expected_below)
        assert r.interior_mass == pytest.approx(1.0 - expected_below)


class TestCdfAtThreshold:
    def test_normal(self):
        fit = NormalFit(mu=0.0, sigma=1.0, method="m")
        assert cdf_at_threshold(fit, 0.0) == pytest.approx(0.5)
        assert cdf_at_threshold(fit, 1.96) == pytest.approx(0.975, abs=1e-3)

    def test_lognormal(self):
        fit = LognormalFit(mu=0.0, sigma=1.0, method="m")
        got = cdf_at_threshold(fit, 1.0)
        assert got == pytest.approx(0.5)

    def test_student_t(self):
        fit = StudentTFit(loc=0.0, scale=1.0, df=5.0, method="m")
        got = cdf_at_threshold(fit, 0.0)
        assert got == pytest.approx(0.5)


class TestFitTo11Percentiles:
    def test_normal(self):
        fit = NormalFit(mu=0.0, sigma=1.0, method="m")
        out = fit_to_11_percentiles(fit)
        assert set(out.keys()) == set(STANDARD_PERCENTILES)
        for q in STANDARD_PERCENTILES:
            assert out[q] == pytest.approx(float(stats.norm.ppf(q)), rel=1e-9, abs=1e-9)

    def test_monotone(self):
        fit = LognormalFit(mu=1.0, sigma=0.5, method="m")
        out = fit_to_11_percentiles(fit)
        values = [out[q] for q in sorted(out.keys())]
        assert all(a < b for a, b in zip(values, values[1:]))


class TestPercentilesToMetaculusCdf:
    def test_smoke(self):
        cdf = percentiles_to_metaculus_cdf(
            percentile_values={0.1: 5.0, 0.5: 10.0, 0.9: 15.0},
            lower_bound=0.0,
            upper_bound=20.0,
            open_lower=False,
            open_upper=False,
        )
        assert isinstance(cdf, list)
        assert len(cdf) == 201
        assert cdf[0] == pytest.approx(0.0, abs=1e-9)
        assert cdf[-1] == pytest.approx(1.0, abs=1e-9)
        diffs = np.diff(np.array(cdf))
        assert np.all(diffs > 0)

    def test_open_bounds_respected(self):
        cdf = percentiles_to_metaculus_cdf(
            percentile_values={0.1: 5.0, 0.5: 10.0, 0.9: 15.0},
            lower_bound=0.0,
            upper_bound=20.0,
            open_lower=True,
            open_upper=True,
        )
        assert cdf[0] >= 0.001
        assert cdf[-1] <= 0.999

    def test_invalid_bound_raises(self):
        with pytest.raises(ValueError):
            percentiles_to_metaculus_cdf(
                percentile_values={0.5: 5.0},
                lower_bound=10.0,
                upper_bound=0.0,
                open_lower=False,
                open_upper=False,
            )

    def test_invalid_percentile_key_raises(self):
        with pytest.raises(ValueError):
            percentiles_to_metaculus_cdf(
                percentile_values={1.5: 5.0, 0.5: 10.0},
                lower_bound=0.0,
                upper_bound=20.0,
                open_lower=False,
                open_upper=False,
            )


class TestDirichletWithOther:
    def test_means_recovered(self):
        out = dirichlet_with_other({"A": 0.5, "B": 0.3, "C": 0.2}, other_mass=None, concentration=20.0)
        assert set(out.keys()) == {"A", "B", "C"}
        for k, expected in [("A", 0.5), ("B", 0.3), ("C", 0.2)]:
            assert out[k].mean == pytest.approx(expected, rel=1e-9)

    def test_ci_non_degenerate(self):
        out = dirichlet_with_other({"A": 0.5, "B": 0.5}, other_mass=None, concentration=10.0)
        for v in out.values():
            assert v.ci_80_low < v.mean < v.ci_80_high

    def test_higher_concentration_narrower_ci(self):
        low = dirichlet_with_other({"A": 0.5, "B": 0.5}, other_mass=None, concentration=2.0)
        high = dirichlet_with_other({"A": 0.5, "B": 0.5}, other_mass=None, concentration=200.0)
        low_w = low["A"].ci_80_high - low["A"].ci_80_low
        high_w = high["A"].ci_80_high - high["A"].ci_80_low
        assert high_w < low_w

    def test_with_other(self):
        out = dirichlet_with_other({"A": 0.3, "B": 0.3}, other_mass=0.4, concentration=10.0)
        assert "Other" in out
        assert out["Other"].mean == pytest.approx(0.4, rel=1e-9)

    def test_sum_violation_raises(self):
        with pytest.raises(ValueError, match="sum"):
            dirichlet_with_other({"A": 0.2, "B": 0.2}, other_mass=0.2)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            dirichlet_with_other({}, other_mass=None)

    def test_invalid_concentration(self):
        with pytest.raises(ValueError):
            dirichlet_with_other({"A": 1.0}, other_mass=None, concentration=0.0)


class TestNegativeBinomialPercentiles:
    def test_canonical_keys(self):
        out = negative_binomial_percentiles(mean=10.0, overdispersion_factor=2.0)
        assert set(out.keys()) == set(STANDARD_PERCENTILES)

    def test_median_matches_scipy(self):
        # Reconstruct the (r, p) parameterization used by
        # ``negative_binomial_percentiles`` and assert that the reported 0.5
        # percentile matches scipy.stats.nbinom.ppf(0.5) exactly (integer PPF,
        # so exact equality is appropriate).
        mean, phi = 50.0, 1.5
        p = 1.0 / phi
        r = mean / (phi - 1.0)
        out = negative_binomial_percentiles(mean=mean, overdispersion_factor=phi)
        expected_median = float(stats.nbinom(r, p).ppf(0.5))
        assert out[0.5] == pytest.approx(expected_median)

    def test_monotone(self):
        out = negative_binomial_percentiles(mean=8.0, overdispersion_factor=3.0)
        vs = [out[q] for q in sorted(out.keys())]
        assert all(a <= b for a, b in zip(vs, vs[1:]))

    def test_phi_one_raises(self):
        with pytest.raises(ValueError, match="overdispersion"):
            negative_binomial_percentiles(mean=5.0, overdispersion_factor=1.0)

    def test_negative_mean_raises(self):
        with pytest.raises(ValueError):
            negative_binomial_percentiles(mean=-1.0, overdispersion_factor=2.0)


class TestPoissonPercentiles:
    def test_canonical_keys(self):
        out = poisson_percentiles(mean=5.0)
        assert set(out.keys()) == set(STANDARD_PERCENTILES)

    def test_median_near_mean(self):
        out = poisson_percentiles(mean=20.0)
        assert abs(out[0.5] - 20.0) <= 2

    def test_negative_mean_raises(self):
        with pytest.raises(ValueError):
            poisson_percentiles(mean=-0.5)


class TestBetaBinomialCeilingPercentiles:
    def test_mean_recovered_roughly(self):
        out = beta_binomial_ceiling_percentiles(mean=5.0, ceiling=10, concentration=50.0)
        assert abs(out[0.5] - 5.0) <= 1.5

    def test_max_not_exceeding_ceiling(self):
        out = beta_binomial_ceiling_percentiles(mean=3.0, ceiling=10, concentration=10.0)
        for v in out.values():
            assert v <= 10

    def test_invalid_mean_range(self):
        with pytest.raises(ValueError):
            beta_binomial_ceiling_percentiles(mean=11.0, ceiling=10)

    def test_invalid_ceiling(self):
        with pytest.raises(ValueError):
            beta_binomial_ceiling_percentiles(mean=0.0, ceiling=0)

    def test_invalid_concentration(self):
        with pytest.raises(ValueError):
            beta_binomial_ceiling_percentiles(mean=5.0, ceiling=10, concentration=0.0)


class TestPercentileFamilyConsistency:
    def test_lognormal_declared_normal_flags(self):
        # Generate from lognormal, claim normal → best fit is lognormal → flag.
        mu, sigma = 2.0, 0.8
        declared = {
            q: float(math.exp(stats.norm.ppf(q, loc=mu, scale=sigma)))
            for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        }
        res = percentile_family_consistency(declared, claimed_family="normal")
        assert isinstance(res, ConsistencyResult)
        assert res.details["best_fit_family"] == "lognormal"
        assert res.flag is True
        # Ratio should be > 2
        ratio = res.details["sse_ratio_claimed_over_best"]
        assert ratio > 2.0

    def test_normal_declared_normal_no_flag(self):
        declared = {q: float(stats.norm.ppf(q, loc=0.0, scale=1.0)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
        res = percentile_family_consistency(declared, claimed_family="normal")
        assert res.flag is False

    def test_claimed_none_never_flags(self):
        declared = {q: float(math.exp(stats.norm.ppf(q))) for q in [0.1, 0.5, 0.9]}
        res = percentile_family_consistency(declared, claimed_family=None)
        assert res.flag is False

    def test_unfittable_family_no_flag(self):
        declared = {q: float(stats.norm.ppf(q)) for q in [0.1, 0.5, 0.9]}
        res = percentile_family_consistency(declared, claimed_family="skew_normal")
        assert res.flag is False

    def test_student_t_df_threaded_through_to_fit(self):
        # Percentiles from a Student-t with df=2 (heavy tails). With df=2 the
        # student_t fit should win; with a different (wrong) df it should lose.
        df_true = 2.0
        declared = {
            q: float(stats.t.ppf(q, df=df_true, loc=0.0, scale=1.0)) for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        }
        res_correct_df = percentile_family_consistency(declared, claimed_family=None, student_t_df=df_true)
        # Student-t should be a much better fit than normal for df=2 heavy tails.
        sse = res_correct_df.details["sse_by_family"]
        assert sse["student_t"] <= sse["normal"]

    def test_invalid_student_t_df_raises(self):
        declared = {0.1: -1.0, 0.5: 0.0, 0.9: 1.0}
        with pytest.raises(ValueError, match="student_t_df"):
            percentile_family_consistency(declared, claimed_family=None, student_t_df=0.0)

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            percentile_family_consistency({}, claimed_family="normal")
        with pytest.raises(ValueError):
            percentile_family_consistency({0.5: 1.0}, claimed_family="normal")
        with pytest.raises(ValueError):
            percentile_family_consistency({1.5: 1.0, 0.5: 2.0}, claimed_family="normal")


class TestStatedBaseRateConsistency:
    def test_weak_evidence_big_move_flags(self):
        # prior 0.2 → posterior 0.9 — LR far greater than 3
        res = stated_base_rate_consistency(0.2, 0.9, "weak")
        assert res.flag is True
        assert "weak" in (res.flag_reason or "")

    def test_moderate_evidence_small_move_no_flag(self):
        res = stated_base_rate_consistency(0.2, 0.25, "moderate")
        assert res.flag is False

    def test_none_evidence_just_under_log125_no_flag(self):
        # LR just under log(1.25) ≈ 0.2231. Prior 0.5 → posterior 0.55 gives
        # LR = (0.55/0.45) / 1 = 1.2222, log(LR) ≈ 0.2007 < 0.2231 → no flag.
        res = stated_base_rate_consistency(0.5, 0.55, "none")
        assert res.flag is False
        assert res.details["implied_lr"] == pytest.approx(1.2222, rel=1e-3)

    def test_none_evidence_over_log125_flags(self):
        # Prior 0.5 → posterior 0.58 gives LR ≈ 1.381, log(LR) ≈ 0.3228 > 0.2231 → flag.
        res = stated_base_rate_consistency(0.5, 0.58, "none")
        assert res.flag is True

    def test_none_evidence_exactly_identity_no_flag(self):
        res = stated_base_rate_consistency(0.3, 0.3, "none")
        assert res.flag is False

    def test_boundary_prior_does_not_raise(self):
        # Degenerate prior (0 or 1) → unflagged result with LR=None.
        res = stated_base_rate_consistency(0.0, 0.5, "weak")
        assert res.flag is False
        assert res.details["implied_lr"] is None
        assert "boundary" in (res.flag_reason or "")

    def test_strong_never_flags(self):
        res = stated_base_rate_consistency(0.01, 0.99, "strong")
        assert res.flag is False

    def test_moderate_big_jump_flags(self):
        res = stated_base_rate_consistency(0.001, 0.99, "moderate")
        assert res.flag is True

    def test_invalid_strength_raises(self):
        with pytest.raises(ValueError):
            stated_base_rate_consistency(0.5, 0.5, "super")  # ty: ignore[invalid-argument-type]  # type: ignore[arg-type]


class TestPercentileMonotonicity:
    def test_monotone_no_flag(self):
        declared = {0.1: 1.0, 0.5: 5.0, 0.9: 10.0}
        res = percentile_monotonicity_check(declared)
        assert res.flag is False
        assert res.details["first_violation"] is None

    def test_violation_flags(self):
        declared = {0.1: 1.0, 0.5: 0.5, 0.9: 10.0}
        res = percentile_monotonicity_check(declared)
        assert res.flag is True
        first = res.details["first_violation"]
        assert first == (0.5, 1.0, 0.5)

    def test_equal_values_flag(self):
        declared = {0.1: 1.0, 0.5: 1.0, 0.9: 2.0}
        res = percentile_monotonicity_check(declared)
        assert res.flag is True

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            percentile_monotonicity_check({})


# ===========================================================================
# Branch-edge coverage — consistency, Dirichlet, NB (F15-F19)
# ===========================================================================


class TestConsistencyBranchEdges:
    def test_near_zero_sse_best_produces_huge_ratio(self):
        # Exact normal CDF-implied percentiles → normal SSE is ~0 (Nelder-Mead
        # doesn't land on literal 0.0; the ratio is finite-but-huge). Claim
        # lognormal, which fits much worse. Assert the flag fires either via
        # the sse_best==0 guard branch (→ math.inf) OR via a large finite ratio.
        mu, sigma = 5.0, 2.0
        declared = {
            q: float(stats.norm.ppf(q, loc=mu, scale=sigma)) for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
        }
        res = percentile_family_consistency(declared, claimed_family="lognormal")
        sse = res.details["sse_by_family"]
        assert sse["normal"] == pytest.approx(0.0, abs=1e-8)
        ratio = res.details["sse_ratio_claimed_over_best"]
        assert ratio == math.inf or ratio > 1e6
        assert res.flag is True

    def test_lognormal_skipped_when_any_non_positive_value(self):
        # Mixed-sign declared values → lognormal cannot fit (must be > 0).
        declared = {0.1: -5.0, 0.5: 0.0, 0.9: 5.0}
        res = percentile_family_consistency(declared, claimed_family="normal")
        sse = res.details["sse_by_family"]
        assert sse["lognormal"] == math.inf
        assert res.details["best_fit_family"] in {"normal", "student_t"}

    def test_fits_by_family_populated_on_success(self):
        # Clean normal percentiles → all three families should produce a valid
        # cached fit object in details["fits_by_family"].
        declared = {q: float(stats.norm.ppf(q, loc=5.0, scale=2.0)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]}
        res = percentile_family_consistency(declared, claimed_family="normal")
        fits = res.details["fits_by_family"]
        assert isinstance(fits["normal"], NormalFit)
        assert isinstance(fits["lognormal"], LognormalFit)
        assert isinstance(fits["student_t"], StudentTFit)

    def test_fits_by_family_missing_on_raise(self):
        # Non-positive declared values → lognormal fit raises ValueError and is
        # absent from fits_by_family, while sse_by_family still records math.inf.
        declared = {0.1: -5.0, 0.5: 0.0, 0.9: 5.0}
        res = percentile_family_consistency(declared, claimed_family="normal")
        fits = res.details["fits_by_family"]
        assert "lognormal" not in fits
        assert res.details["sse_by_family"]["lognormal"] == math.inf
        # normal and student_t should still succeed on mixed-sign data.
        assert isinstance(fits["normal"], NormalFit)
        assert isinstance(fits["student_t"], StudentTFit)


class TestDirichletDegenerateBetaBranches:
    def test_alpha_k_zero_raises(self):
        # option_probs with mass 0 for a key → α=0 → fail-fast ValueError.
        # Configuration: {"A": 0.0} + other_mass=1.0 sums to 1.0. α_A = 0.
        with pytest.raises(ValueError, match="alpha_k"):
            dirichlet_with_other({"A": 0.0}, other_mass=1.0, concentration=10.0)

    def test_other_mass_zero_raises(self):
        # {"A": 1.0} + other_mass=0.0 → α for __OTHER__ is 0 → ValueError.
        with pytest.raises(ValueError, match="alpha_k"):
            dirichlet_with_other({"A": 1.0}, other_mass=0.0, concentration=10.0)


class TestNegativeBinomialMeanZero:
    def test_nb_mean_zero_returns_all_zeros(self):
        # mean=0 is now handled as a point mass at 0: every canonical
        # percentile is exactly 0.0.
        out = negative_binomial_percentiles(mean=0.0, overdispersion_factor=2.0)
        assert set(out.keys()) == set(STANDARD_PERCENTILES)
        for v in out.values():
            assert v == 0.0


class TestNelderMeadNonConvergenceRaises:
    @pytest.fixture
    def forced_failure_monkeypatch(self, monkeypatch):
        """Force scipy.optimize.minimize (as imported by distributions.py) to
        return a non-success result so each fit function raises."""
        from scipy.optimize import OptimizeResult

        from metaculus_bot.probabilistic_tools import distributions as dist_mod

        def _forced_fail(objective, x0, **kwargs):
            return OptimizeResult(success=False, message="forced", x=x0, fun=float("inf"), nit=0)

        monkeypatch.setattr(dist_mod.optimize, "minimize", _forced_fail)
        return monkeypatch

    def test_fit_normal_raises_on_non_convergence(self, forced_failure_monkeypatch):
        pvs = {q: float(stats.norm.ppf(q)) for q in [0.1, 0.5, 0.9]}
        with pytest.raises(ValueError, match="did not converge"):
            fit_normal_from_percentiles(pvs)

    def test_fit_lognormal_raises_on_non_convergence(self, forced_failure_monkeypatch):
        pvs = {q: float(math.exp(stats.norm.ppf(q))) for q in [0.1, 0.5, 0.9]}
        with pytest.raises(ValueError, match="did not converge"):
            fit_lognormal_from_percentiles(pvs)

    def test_fit_student_t_raises_on_non_convergence(self, forced_failure_monkeypatch):
        pvs = {q: float(stats.t.ppf(q, df=5.0)) for q in [0.1, 0.5, 0.9]}
        with pytest.raises(ValueError, match="did not converge"):
            fit_student_t_from_percentiles(pvs, df=5.0)


# ===========================================================================
# Distribution roundtrip — B.2 property-style tests
# ===========================================================================


class TestDistributionRoundtrip:
    @pytest.mark.parametrize("mu", [-3.0, 0.0, 5.0])
    @pytest.mark.parametrize("sigma", [0.1, 1.0, 10.0])
    def test_normal_roundtrip(self, mu: float, sigma: float):
        declared = {q: float(stats.norm.ppf(q, loc=mu, scale=sigma)) for q in STANDARD_PERCENTILES}
        fit = fit_normal_from_percentiles(declared)
        assert fit.mu == pytest.approx(mu, abs=0.01)
        assert fit.sigma == pytest.approx(sigma, rel=0.01)

    @pytest.mark.parametrize("mu", [-1.0, 0.0, 2.0])
    @pytest.mark.parametrize("sigma", [0.25, 0.75, 1.5])
    def test_lognormal_roundtrip(self, mu: float, sigma: float):
        # Lognormal parameters are in log-space (mu, sigma). Generate the
        # implied percentile values, then fit — recovered mu/sigma should match.
        declared = {q: float(math.exp(stats.norm.ppf(q, loc=mu, scale=sigma))) for q in STANDARD_PERCENTILES}
        fit = fit_lognormal_from_percentiles(declared)
        assert fit.mu == pytest.approx(mu, abs=0.01)
        assert fit.sigma == pytest.approx(sigma, rel=0.01)

    @pytest.mark.parametrize("loc", [0.0, 10.0])
    @pytest.mark.parametrize("scale", [1.0, 5.0])
    def test_student_t_roundtrip(self, loc: float, scale: float):
        df = 5.0
        declared = {q: float(stats.t.ppf(q, df=df, loc=loc, scale=scale)) for q in STANDARD_PERCENTILES}
        fit = fit_student_t_from_percentiles(declared, df=df)
        assert fit.loc == pytest.approx(loc, abs=0.02)
        assert fit.scale == pytest.approx(scale, rel=0.02)
        assert fit.df == pytest.approx(df)


# ===========================================================================
# Pooling identities — Satopaa α=1 equals log pool; weight rescaling
# ===========================================================================


class TestPoolingIdentities:
    @pytest.mark.parametrize(
        "probs",
        [
            [0.1, 0.5, 0.9],
            [0.2, 0.3, 0.4, 0.6],
            [0.05, 0.95],
        ],
    )
    def test_satopaa_alpha_one_equals_log_pool(self, probs: list[float]):
        assert satopaa_extremize(probs, alpha=1.0) == pytest.approx(log_pool(probs), rel=1e-12)

    @pytest.mark.parametrize("pool_fn", [linear_pool, log_pool])
    def test_pool_invariant_under_weight_rescaling(self, pool_fn):
        probs = [0.2, 0.5, 0.8]
        weights_small = [1.0, 2.0, 3.0]
        weights_large = [10.0, 20.0, 30.0]
        assert pool_fn(probs, weights=weights_small) == pytest.approx(pool_fn(probs, weights=weights_large), rel=1e-12)


# ===========================================================================
# Beta-binomial posterior mean closed form
# ===========================================================================


class TestBetaBinomialClosedForm:
    @pytest.mark.parametrize(
        "k,n,prior_mean,prior_strength",
        [
            (0, 10, 0.1, 1.0),
            (5, 20, 0.3, 2.0),
            (15, 30, 0.5, 5.0),
            (0, 0, 0.4, 10.0),
            (2, 3, 0.8, 1.0),
        ],
    )
    def test_posterior_mean_closed_form(self, k: int, n: int, prior_mean: float, prior_strength: float):
        r = beta_binomial_update(k=k, n=n, prior_mean=prior_mean, prior_strength=prior_strength)
        expected = (prior_strength * prior_mean + k) / (prior_strength + n)
        assert r.posterior_mean == pytest.approx(expected, rel=1e-9)


class TestNumericHelpersResolveWeights:
    def test_resolve_weights_none_returns_ones(self):
        from metaculus_bot.probabilistic_tools._numeric_helpers import resolve_weights

        w = resolve_weights(None, 4)
        assert w.tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_resolve_weights_length_mismatch_raises(self):
        from metaculus_bot.probabilistic_tools._numeric_helpers import resolve_weights

        with pytest.raises(ValueError, match="weights length"):
            resolve_weights([0.5, 0.5], 3)

    def test_resolve_weights_negative_raises(self):
        from metaculus_bot.probabilistic_tools._numeric_helpers import resolve_weights

        with pytest.raises(ValueError, match=">= 0"):
            resolve_weights([1.0, -0.5, 0.5], 3)

    def test_resolve_weights_zero_sum_raises(self):
        from metaculus_bot.probabilistic_tools._numeric_helpers import resolve_weights

        with pytest.raises(ValueError, match="sum to > 0"):
            resolve_weights([0.0, 0.0], 2)

    def test_resolve_weights_preserves_magnitude(self):
        # Weights are returned as-is (not pre-normalized); callers divide by sum.
        from metaculus_bot.probabilistic_tools._numeric_helpers import resolve_weights

        w = resolve_weights([2.0, 4.0, 6.0], 3)
        assert w.tolist() == [2.0, 4.0, 6.0]


class TestEvalCdfExported:
    def test_eval_cdf_importable_and_callable(self):
        from metaculus_bot.probabilistic_tools.distributions import eval_cdf

        fit = NormalFit(mu=0.0, sigma=1.0, method="test")
        # Standard normal CDF at 0 is 0.5.
        assert eval_cdf(fit, 0.0) == pytest.approx(0.5, abs=1e-12)

    def test_eval_cdf_lognormal_nonpositive_returns_zero(self):
        # LognormalFit support excludes x <= 0; eval_cdf returns 0.0 for
        # negative/zero inputs (NOT raise) — preserved semantics from _eval_cdf.
        from metaculus_bot.probabilistic_tools.distributions import eval_cdf

        fit = LognormalFit(mu=0.0, sigma=1.0, method="test")
        assert eval_cdf(fit, -1.0) == 0.0
        assert eval_cdf(fit, 0.0) == 0.0
