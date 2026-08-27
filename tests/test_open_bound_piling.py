"""WARN detector for open-bound percentile piling.

On an OPEN bound the displayed edge is not a hard cap: mass beyond it is expressed by
placing declared percentiles past the edge. When a model instead crams the terminal CDF
bin AND keeps every declared percentile inside the range, it is treating the open edge as a
hard limit — the prompt-contradiction bug. ``log_open_bound_piling_diagnostics`` WARNs on
that pattern. It must NOT fire when the model correctly placed a percentile beyond the edge,
when the terminal-bin mass is a thin genuine tail, or when the bound is closed.

The declared percentiles are passed to the detector explicitly (the model-declared,
sanitized values) rather than read from ``prediction.declared_percentiles``: on discrete
questions ``build_numeric_distribution`` overwrites that field with a resampled grid pinned
to the raw bounds, which false-fired on correct handlers (regression test at the bottom).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric.config import OPEN_BOUND_PILING_THRESHOLD
from metaculus_bot.numeric.diagnostics import log_open_bound_piling_diagnostics
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles

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


def _make_prediction(cdf_probs: list[float], *, lower: float = 0.0, upper: float = 100.0) -> NumericDistribution:
    """Stub exposing ``.cdf`` as a Percentile list on a uniform value grid."""
    x_vals = np.linspace(lower, upper, len(cdf_probs))
    cdf = [Percentile(percentile=float(p), value=float(v)) for v, p in zip(x_vals, cdf_probs, strict=False)]
    return cast(NumericDistribution, SimpleNamespace(cdf=cdf))


def _declared(values: list[float]) -> list[Percentile]:
    """Model-declared percentile VALUES (percentile labels are irrelevant to the detector)."""
    return [Percentile(percentile=0.5, value=float(v)) for v in values]


def _cdf_with_top_bin_mass(top_bin_mass: float) -> list[float]:
    """Monotone CDF over ``_N`` points whose final step equals ``top_bin_mass``."""
    body = np.linspace(0.0, 1.0 - top_bin_mass, _N - 1)
    return [*[float(v) for v in body], 1.0]


def _cdf_with_bottom_bin_mass(bottom_bin_mass: float) -> list[float]:
    """Monotone CDF over ``_N`` points whose first step equals ``bottom_bin_mass``."""
    tail = np.linspace(bottom_bin_mass, 1.0, _N - 1)
    return [0.0, *[float(v) for v in tail]]


def _piling_records(caplog) -> list[str]:
    return [r.message for r in caplog.records if "OPEN_BOUND_PILING" in r.message]


def test_open_upper_crammed_fires(caplog):
    """Crammed terminal bin (0.20) with every declared percentile <= upper → exactly one WARN."""
    q = _make_question(open_upper=True, upper=100.0)
    pred = _make_prediction(_cdf_with_top_bin_mass(0.20))

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model", _declared([10.0, 50.0, 95.0]))

    records = _piling_records(caplog)
    assert len(records) == 1
    msg = records[0]
    assert "question=1234" in msg
    assert "model=test-model" in msg
    assert "bound=upper" in msg
    assert "bin_mass=0.200" in msg
    assert "declared_edge=95" in msg
    assert "bound_value=100" in msg


def test_open_upper_correct_handler_no_fire(caplog):
    """P99 value ABOVE the open ceiling → max declared > upper → no fire, even if bin is heavy."""
    q = _make_question(open_upper=True, upper=100.0)
    pred = _make_prediction(_cdf_with_top_bin_mass(0.20))

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model", _declared([10.0, 50.0, 150.0]))

    assert not _piling_records(caplog)


def test_open_upper_thin_tail_no_fire(caplog):
    """Terminal-bin mass 0.05 (< K=0.10) is a thin genuine tail → no fire."""
    q = _make_question(open_upper=True, upper=100.0)
    pred = _make_prediction(_cdf_with_top_bin_mass(0.05))

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model", _declared([10.0, 50.0, 95.0]))

    assert not _piling_records(caplog)


def test_bin_mass_exactly_at_threshold_fires(caplog):
    """The gate is >= threshold: bin mass exactly equal to K must fire (regression guard for >= → >).

    The threshold kwarg is set to the exact float the detector will compute as the top-bin
    mass, so the comparison is a true equality (0.10 itself isn't representable in binary).
    """
    q = _make_question(open_upper=True, upper=100.0)
    pred = _make_prediction(_cdf_with_top_bin_mass(OPEN_BOUND_PILING_THRESHOLD))
    exact_mass = pred.cdf[-1].percentile - pred.cdf[-2].percentile

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model", _declared([10.0, 50.0, 95.0]), threshold=exact_mass)

    assert len(_piling_records(caplog)) == 1


def test_closed_bound_early_returns_no_fire(caplog):
    """Closed bounds are outside the detector's remit → early-return, no fire."""
    q = _make_question(open_upper=False, open_lower=False, upper=100.0)
    pred = _make_prediction(_cdf_with_top_bin_mass(0.20))

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model", _declared([10.0, 50.0, 95.0]))

    assert not _piling_records(caplog)


def test_guard_branches_no_fire(caplog):
    """Missing/short CDF or empty declared percentiles → early-return, no fire, no crash."""
    q = _make_question(open_upper=True, upper=100.0)
    declared = _declared([10.0, 50.0, 95.0])

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(_make_prediction([1.0]), q, "test-model", declared)  # len(cdf) < 2
    log_open_bound_piling_diagnostics(
        cast(NumericDistribution, SimpleNamespace(cdf=None)), q, "test-model", declared
    )  # cdf missing
    log_open_bound_piling_diagnostics(_make_prediction(_cdf_with_top_bin_mass(0.20)), q, "test-model", [])  # no decls

    assert not _piling_records(caplog)


def test_open_lower_crammed_fires(caplog):
    """Symmetric lower-branch cram: bottom bin 0.20, min declared >= lower → WARN on 'lower'."""
    q = _make_question(open_lower=True, lower=0.0, upper=100.0)
    pred = _make_prediction(_cdf_with_bottom_bin_mass(0.20))

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model", _declared([5.0, 50.0, 90.0]))

    records = _piling_records(caplog)
    assert len(records) == 1
    assert "bound=lower" in records[0]
    assert "declared_edge=5" in records[0]


def test_both_bounds_open_crammed_fires_both_branches(caplog):
    """Both bounds open and both terminal bins crammed → one WARN per branch."""
    q = _make_question(open_upper=True, open_lower=True, lower=0.0, upper=100.0)
    # CDF with 0.15 mass in each terminal bin and the rest spread across the body.
    body = np.linspace(0.15, 0.85, _N - 2)
    probs = [0.0, *[float(v) for v in body], 1.0]
    pred = _make_prediction(probs)

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model", _declared([5.0, 50.0, 95.0]))

    records = _piling_records(caplog)
    assert len(records) == 2
    assert any("bound=upper" in m for m in records)
    assert any("bound=lower" in m for m in records)


def test_open_upper_well_behaved_continuous_no_false_positive(caplog):
    """A smooth continuous open-upper distribution (small terminal bin) → no false positive."""
    q = _make_question(open_upper=True, lower=0.0, upper=100.0)
    # Uniform CDF: every step is 1/(N-1) ≈ 0.005, well under K.
    uniform = [float(v) for v in np.linspace(0.0, 1.0, _N)]
    pred = _make_prediction(uniform)

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(pred, q, "test-model", _declared([10.0, 50.0, 90.0]))

    assert not _piling_records(caplog)


def test_observed_crammer_0126_fires_correct_handler_0073_does_not(caplog):
    """K=0.10 boundary sanity: the 0.126 crammer fires, a 0.073 correct handler does not."""
    q = _make_question(open_upper=True, upper=100.0)
    declared_inside = _declared([10.0, 50.0, 95.0])

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(_make_prediction(_cdf_with_top_bin_mass(0.126)), q, "crammer", declared_inside)
    assert len(_piling_records(caplog)) == 1

    caplog.clear()
    log_open_bound_piling_diagnostics(_make_prediction(_cdf_with_top_bin_mass(0.073)), q, "good", declared_inside)
    assert not _piling_records(caplog)


# ---------------------------------------------------------------------------
# Discrete-question regression: the detector must read the MODEL-DECLARED
# percentiles, not prediction.declared_percentiles. On discrete questions
# build_numeric_distribution resamples and overwrites declared_percentiles with
# a grid on [lower_bound, upper_bound], pinning max declared at exactly the raw
# bound — reading it false-fired on models that correctly placed P99 above the
# open ceiling (live repro on a Q38195-class question, 2026-07-11).
# ---------------------------------------------------------------------------

_STANDARD_PS = [0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975, 0.99]


def _discrete_open_upper_question() -> NumericQuestion:
    return NumericQuestion(
        id_of_question=38195,
        id_of_post=38195,
        page_url="https://example.com/q/38195",
        question_text="How many events?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=-0.5,
        upper_bound=7.5,
        open_lower_bound=False,
        open_upper_bound=True,
        unit_of_measure="events",
        zero_point=None,
        cdf_size=9,
    )


def _run_discrete_detector(values: list[float], caplog) -> list[str]:
    """Drive values through the REAL pipeline (sanitize → build) then the detector."""
    q = _discrete_open_upper_question()
    pcts = [Percentile(percentile=p, value=v) for p, v in zip(_STANDARD_PS, values, strict=False)]
    sanitized, zero_point = sanitize_percentiles(pcts, q)
    prediction = build_numeric_distribution(sanitized, q, zero_point)

    caplog.clear()
    caplog.set_level("WARNING")
    log_open_bound_piling_diagnostics(prediction, q, "test-model", sanitized)
    return _piling_records(caplog)


def test_discrete_correct_handler_with_mass_above_ceiling_no_fire(caplog):
    """Discrete open-upper, heavy top bin, but P95/P97.5/P99 placed ABOVE the ceiling → no fire.

    Before the fix this false-fired: the resampled prediction.declared_percentiles maxed out
    at exactly upper_bound (7.5), which the upper branch read as "nothing beyond the ceiling".
    """
    records = _run_discrete_detector(
        [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 5.5, 6.0, 7.0, 7.4, 8.0, 9.0, 12.0],
        caplog,
    )
    assert not records


def test_discrete_crammer_all_percentiles_inside_fires(caplog):
    """Discrete open-upper, heavy top bin, EVERY declared percentile inside the range → fires."""
    records = _run_discrete_detector(
        [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 5.5, 6.0, 6.8, 7.1, 7.3, 7.4, 7.45],
        caplog,
    )
    assert len(records) == 1
    assert "bound=upper" in records[0]
