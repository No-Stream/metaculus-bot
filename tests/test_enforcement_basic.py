import types
from typing import cast

from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric.cluster_processing import compute_cluster_parameters
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.pipeline import sanitize_percentiles
from metaculus_bot.numeric.validation import detect_unit_mismatch


def _fake_question(lower: float, upper: float) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        lower_bound=lower,
        upper_bound=upper,
        open_lower_bound=False,
        open_upper_bound=False,
        id_of_question=123,
        page_url="https://example/q/123",
    )


def _numeric_question(*, lower: float, upper: float, cdf_size: int = 201, open_upper: bool = True) -> NumericQuestion:
    return NumericQuestion(
        question_text="How many?",
        id_of_question=38880,
        id_of_post=38880,
        page_url="https://www.metaculus.com/questions/38880/",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=lower,
        upper_bound=upper,
        open_lower_bound=False,
        open_upper_bound=open_upper,
        zero_point=None,
        unit_of_measure="",
        cdf_size=cdf_size,
    )


def _declared_13(values: list[float]) -> list[Percentile]:
    assert len(values) == len(STANDARD_PERCENTILES) == 13
    return [Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, values, strict=True)]


def test_unit_mismatch_detector_flags_tiny_span():
    # Range is large (1e9); values are nearly identical and tiny → mismatch
    q = _fake_question(0.0, 1_000_000_000.0)
    plist = [Percentile(percentile=p, value=1.0) for p in [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.95]]

    mismatch, reason = detect_unit_mismatch(plist, cast(NumericQuestion, q))
    assert mismatch is True
    assert "span_ratio" in reason or "near-duplicate" in reason or "tiny" in reason


def test_cluster_parameters_use_span_based_delta():
    # Small range (1e6) but substantial span (1e5) should yield spread >= 0.02 * span
    range_size = 1_000_000.0
    span = 100_000.0
    count_like = True

    _value_eps, base_delta, spread_delta = compute_cluster_parameters(range_size, count_like, span)

    # Base delta is range * 1e-6 = 1.0; span-based is 2000.0
    assert base_delta >= 1.0
    assert spread_delta >= 2000.0


def test_concentrated_integer_declaration_passes_guard():
    """Regression: a faithful concentrated low-count declaration (P5..P50=1) must not be
    withheld as a unit mismatch.

    ``sanitize_percentiles`` separates the equal integer values by a jitter epsilon
    (~MIN_BOUNDARY_DISTANCE * range), leaving a ~8e-9 relative gap. The old fixed
    ``min_step_ratio < 1e-8`` near-duplicate rule tripped on that jitter artifact and
    silently dropped the forecaster even though the CDF is faithful. Covers both the
    discrete grid (cdf_size < 201) and the small-range continuous grid (cdf_size == 201),
    since the false positive reproduced on both.
    """
    declared = _declared_13([0, 0, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7])
    for cdf_size in (9, 201):
        question = _numeric_question(lower=-0.5, upper=7.5, cdf_size=cdf_size, open_upper=True)
        sanitized, _zp = sanitize_percentiles(declared, question)
        mismatch, reason = detect_unit_mismatch(sanitized, question)
        assert mismatch is False, f"cdf_size={cdf_size}: false-positive unit mismatch ({reason})"


def test_genuine_unit_error_still_trips_on_discrete():
    """Complement to the regression above: an actual order-of-magnitude scale error on a
    small-count discrete question must still be caught. Values 1000x below the range
    collapse the span, which the (untouched) span/magnitude ratios flag."""
    question = _numeric_question(lower=0.0, upper=10_000.0, cdf_size=9, open_upper=False)
    # Every value ~1000x too small (declared in the wrong unit) → tiny span & magnitude.
    declared = _declared_13([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013])
    mismatch, reason = detect_unit_mismatch(declared, question)
    assert mismatch is True
    assert "span" in reason or "tiny" in reason or "near-duplicate" in reason
