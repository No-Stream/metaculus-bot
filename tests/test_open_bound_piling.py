"""WARN detector for open-bound percentile piling.

On an OPEN bound the displayed edge is not a hard cap: mass beyond it is expressed by
placing declared percentiles past the edge. When a model instead crams the terminal CDF
bin AND keeps every declared percentile inside the range, it is treating the open edge as a
hard limit — the prompt-contradiction bug. ``log_open_bound_piling_diagnostics`` WARNs on
that pattern. It must NOT fire when the model correctly placed a percentile beyond the edge,
when the terminal-bin mass is a thin genuine tail, or when the bound is closed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric.diagnostics import log_open_bound_piling_diagnostics

_N = 201  # continuous CDF point count (value grid resolution for these stubs)


def _make_question(*, open_upper=False, open_lower=False, lower=0.0, upper=100.0) -> NumericQuestion:
    return cast(
        NumericQuestion,
        SimpleNamespace(
            open_upper_bound=open_upper,
            open_lower_bound=open_lower,
            upper_bound=upper,
            lower_bound=lower,
            id_of_question=1234,
            page_url="https://example.com/q/1234",
        ),
    )


def _make_prediction(
    cdf_probs: list[float],
    declared_values: list[float],
    *,
    lower: float = 0.0,
    upper: float = 100.0,
) -> NumericDistribution:
    """Stub exposing ``.cdf`` (Percentile list on a uniform value grid) + ``.declared_percentiles``.

    ``cdf_probs`` are the cumulative probabilities along an evenly-spaced value grid;
    ``declared_values`` are the model's declared percentile VALUES (percentile labels are
    irrelevant to the detector, which only reads ``.value`` from the declared set).
    """
    x_vals = np.linspace(lower, upper, len(cdf_probs))
    cdf = [Percentile(percentile=float(p), value=float(v)) for v, p in zip(x_vals, cdf_probs)]
    declared = [Percentile(percentile=0.5, value=float(v)) for v in declared_values]
    return cast(
        NumericDistribution,
        SimpleNamespace(cdf=cdf, declared_percentiles=declared),
    )


def _cdf_with_top_bin_mass(top_bin_mass: float) -> list[float]:
    """Monotone CDF over ``_N`` points whose final step equals ``top_bin_mass``."""
    body = np.linspace(0.0, 1.0 - top_bin_mass, _N - 1)
    return [*[float(v) for v in body], 1.0]


def _cdf_with_bottom_bin_mass(bottom_bin_mass: float) -> list[float]:
    """Monotone CDF over ``_N`` points whose first step equals ``bottom_bin_mass``."""
    tail = np.linspace(bottom_bin_mass, 1.0, _N - 1)
    return [0.0, *[float(v) for v in tail]]


def test_open_upper_crammed_fires(caplog):
    """Crammed terminal bin (0.20) with every declared percentile <= upper → WARN."""
    q = _make_question(open_upper=True, upper=100.0)
    # All declared percentiles sit inside the range (max declared 95 <= 100).
    pred = _make_prediction(_cdf_with_top_bin_mass(0.20), [10.0, 50.0, 95.0])

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model")

    assert any("OPEN_BOUND_PILING" in r.message for r in caplog.records)


def test_open_upper_correct_handler_no_fire(caplog):
    """P99 value ABOVE the open ceiling → max_declared > upper → no fire, even if bin is heavy."""
    q = _make_question(open_upper=True, upper=100.0)
    # Model placed its top percentile at 150, beyond the open ceiling of 100.
    pred = _make_prediction(_cdf_with_top_bin_mass(0.20), [10.0, 50.0, 150.0])

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model")

    assert not any("OPEN_BOUND_PILING" in r.message for r in caplog.records)


def test_open_upper_thin_tail_no_fire(caplog):
    """Terminal-bin mass 0.05 (< K=0.10) is a thin genuine tail → no fire."""
    q = _make_question(open_upper=True, upper=100.0)
    pred = _make_prediction(_cdf_with_top_bin_mass(0.05), [10.0, 50.0, 95.0])

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model")

    assert not any("OPEN_BOUND_PILING" in r.message for r in caplog.records)


def test_closed_bound_early_returns_no_fire(caplog):
    """Closed bounds are outside the detector's remit → early-return, no fire."""
    q = _make_question(open_upper=False, open_lower=False, upper=100.0)
    pred = _make_prediction(_cdf_with_top_bin_mass(0.20), [10.0, 50.0, 95.0])

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model")

    assert not any("OPEN_BOUND_PILING" in r.message for r in caplog.records)


def test_open_lower_crammed_fires(caplog):
    """Symmetric lower-branch cram: bottom bin 0.20, min declared >= lower → WARN on 'lower'."""
    q = _make_question(open_lower=True, lower=0.0, upper=100.0)
    pred = _make_prediction(_cdf_with_bottom_bin_mass(0.20), [5.0, 50.0, 90.0])

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model")

    records = [r.message for r in caplog.records if "OPEN_BOUND_PILING" in r.message]
    assert records
    assert any("bound=lower" in m for m in records)


def test_open_upper_well_behaved_continuous_no_false_positive(caplog):
    """A smooth continuous open-upper distribution (small terminal bin) → no false positive."""
    q = _make_question(open_upper=True, lower=0.0, upper=100.0)
    # Uniform CDF: every step is 1/(N-1) ≈ 0.005, well under K.
    uniform = [float(v) for v in np.linspace(0.0, 1.0, _N)]
    pred = _make_prediction(uniform, [10.0, 50.0, 90.0])

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model")

    assert not any("OPEN_BOUND_PILING" in r.message for r in caplog.records)


def test_observed_crammer_0126_fires_correct_handler_0073_does_not(caplog):
    """K=0.10 boundary sanity: the 0.126 crammer fires, a 0.073 correct handler does not."""
    q = _make_question(open_upper=True, upper=100.0)

    crammer = _make_prediction(_cdf_with_top_bin_mass(0.126), [10.0, 50.0, 95.0])
    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(crammer, q, "crammer")
    assert any("OPEN_BOUND_PILING" in r.message for r in caplog.records)

    good = _make_prediction(_cdf_with_top_bin_mass(0.073), [10.0, 50.0, 95.0])
    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(good, q, "good")
    assert not any("OPEN_BOUND_PILING" in r.message for r in caplog.records)
