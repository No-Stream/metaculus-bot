"""
Probabilistic / statistical tools for forecaster-declared quantities.

Pure functions that consume declarations the forecaster made in its
structured output block (priors, base rates, percentiles, hazard
assumptions, etc.) and return ground-truth probability math: Beta-binomial
posteriors, survival probabilities, pooled probabilities, fitted
distributions, out-of-bounds mass, consistency flags.

Dormant surface — imported by ``metaculus_bot.tool_runner`` but not wired
into prompts or the runtime pipeline yet.

Error conventions
-----------------
All public functions in this package raise ``ValueError`` on invalid input
(out-of-range probabilities, malformed percentile dicts, incompatible
argument combinations). Optimizer-based fits (``fit_normal_from_percentiles``,
``fit_lognormal_from_percentiles``, ``fit_student_t_from_percentiles``)
raise ``ValueError`` on non-convergence rather than silently returning a
degraded fit — callers should wrap these in ``try/except ValueError`` and
skip the tool output on failure. No function in this package returns
``None`` or ``inf`` as an error signal; those are reserved for legitimate
results (e.g., ``_sse_for_fit`` may return ``math.inf`` when a
lognormal fit's support excludes a declared percentile, which is a valid
"best-fit-is-not-lognormal" signal, not an error).

Export taxonomy (see ``scratch_docs_and_planning/probabilistic_tools_activation.md``
for the full activation plan):

- **Currently dispatched by tool_runner**: ``beta_binomial_update``,
  ``base_rate_blend``, ``implied_likelihood_ratio``, ``linear_pool``,
  ``log_pool``, ``satopaa_extremize``, ``linear_pool_options``,
  ``prob_event_before``, ``percentile_family_consistency``,
  ``percentile_monotonicity_check``, ``stated_base_rate_consistency``,
  ``fit_normal_from_percentiles``, ``fit_lognormal_from_percentiles``,
  ``fit_student_t_from_percentiles``, ``out_of_bounds_mass``,
  ``cdf_at_threshold``, ``dirichlet_with_other``, plus the associated
  dataclasses (``BetaBinomialResult``, ``ConsistencyResult``,
  ``TailMassResult``, ``NormalFit``/``LognormalFit``/``StudentTFit``,
  ``DirichletCI``, ``SurvivalResult``) and ``DEFAULT_INFORMATIVE_PRIOR_STRENGTH``.
- **Library helpers for future wiring (phase-3)**: ``bayes_from_likelihoods``,
  ``laplace_rule_of_succession``, ``inverse_variance_pool``,
  ``weibull_prob_event_before``, ``poisson_at_least_one``,
  ``base_rate_to_hazard``, ``fit_to_11_percentiles``,
  ``percentiles_to_metaculus_cdf``, ``negative_binomial_percentiles``,
  ``poisson_percentiles``, ``beta_binomial_ceiling_percentiles``,
  ``FitType``.
"""

from __future__ import annotations

from metaculus_bot.probabilistic_tools.aggregation import (
    inverse_variance_pool,
    linear_pool,
    linear_pool_options,
    log_pool,
    satopaa_extremize,
)
from metaculus_bot.probabilistic_tools.base_rate import (
    DEFAULT_INFORMATIVE_PRIOR_STRENGTH,
    BetaBinomialResult,
    base_rate_blend,
    bayes_from_likelihoods,
    beta_binomial_update,
    implied_likelihood_ratio,
    laplace_rule_of_succession,
)
from metaculus_bot.probabilistic_tools.consistency import (
    ConsistencyResult,
    percentile_family_consistency,
    percentile_monotonicity_check,
    stated_base_rate_consistency,
)
from metaculus_bot.probabilistic_tools.distributions import (
    FitType,
    LognormalFit,
    NormalFit,
    StudentTFit,
    TailMassResult,
    cdf_at_threshold,
    fit_lognormal_from_percentiles,
    fit_normal_from_percentiles,
    fit_student_t_from_percentiles,
    fit_to_11_percentiles,
    out_of_bounds_mass,
    percentiles_to_metaculus_cdf,
)
from metaculus_bot.probabilistic_tools.mc_discrete import (
    DirichletCI,
    beta_binomial_ceiling_percentiles,
    dirichlet_with_other,
    negative_binomial_percentiles,
    poisson_percentiles,
)
from metaculus_bot.probabilistic_tools.survival import (
    SurvivalResult,
    base_rate_to_hazard,
    poisson_at_least_one,
    prob_event_before,
    weibull_prob_event_before,
)

__all__ = [
    "BetaBinomialResult",
    "ConsistencyResult",
    "DEFAULT_INFORMATIVE_PRIOR_STRENGTH",
    "DirichletCI",
    "FitType",
    "LognormalFit",
    "NormalFit",
    "StudentTFit",
    "SurvivalResult",
    "TailMassResult",
    "base_rate_blend",
    "base_rate_to_hazard",
    "bayes_from_likelihoods",
    "beta_binomial_ceiling_percentiles",
    "beta_binomial_update",
    "cdf_at_threshold",
    "dirichlet_with_other",
    "fit_lognormal_from_percentiles",
    "fit_normal_from_percentiles",
    "fit_student_t_from_percentiles",
    "fit_to_11_percentiles",
    "implied_likelihood_ratio",
    "inverse_variance_pool",
    "laplace_rule_of_succession",
    "linear_pool",
    "linear_pool_options",
    "log_pool",
    "negative_binomial_percentiles",
    "out_of_bounds_mass",
    "percentile_family_consistency",
    "percentile_monotonicity_check",
    "percentiles_to_metaculus_cdf",
    "poisson_at_least_one",
    "poisson_percentiles",
    "prob_event_before",
    "satopaa_extremize",
    "stated_base_rate_consistency",
    "weibull_prob_event_before",
]
