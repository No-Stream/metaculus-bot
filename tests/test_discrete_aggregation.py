import logging

import numpy as np
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.numeric.utils import _postprocess_ensemble_cdf, aggregate_numeric


def test_discrete_mean_aggregation_cdf_size_and_min_step():
    """Mean aggregation returns exactly cdf_size values with required min step for discrete."""
    question = NumericQuestion(
        id_of_question=123,
        id_of_post=123,
        page_url="https://example.com/q/123",
        question_text="Discrete count question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=-0.5,
        upper_bound=7.5,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
        cdf_size=9,
    )

    # Three simple input distributions with modestly different shapes
    decl_a = [
        Percentile(value=0.0, percentile=0.10),
        Percentile(value=3.0, percentile=0.50),
        Percentile(value=7.0, percentile=0.90),
    ]
    decl_b = [
        Percentile(value=1.0, percentile=0.20),
        Percentile(value=4.0, percentile=0.55),
        Percentile(value=6.5, percentile=0.85),
    ]
    decl_c = [
        Percentile(value=0.5, percentile=0.15),
        Percentile(value=3.5, percentile=0.48),
        Percentile(value=7.2, percentile=0.92),
    ]

    dist_a = NumericDistribution(declared_percentiles=decl_a, **question.model_dump())
    dist_b = NumericDistribution(declared_percentiles=decl_b, **question.model_dump())
    dist_c = NumericDistribution(declared_percentiles=decl_c, **question.model_dump())

    agg = aggregate_numeric([dist_a, dist_b, dist_c], question, "mean")

    cdf = agg.cdf
    probs = np.array([p.percentile for p in cdf], dtype=float)
    values = np.array([p.value for p in cdf], dtype=float)

    assert len(cdf) == question.cdf_size
    required_min_step = 0.01 / (question.cdf_size - 1)
    diffs = np.diff(probs)
    assert np.all(diffs >= required_min_step - 1e-12)
    # Endpoints pinned for closed bounds
    assert abs(probs[0] - 0.0) <= 1e-12
    assert abs(probs[-1] - 1.0) <= 1e-12
    # Value axis is the evenly spaced discrete grid
    expected_values = np.linspace(question.lower_bound, question.upper_bound, question.cdf_size)
    assert np.allclose(values, expected_values)


def test_discrete_median_aggregation_open_upper():
    """Median aggregation handles open upper bound and discrete min step."""
    question = NumericQuestion(
        id_of_question=456,
        id_of_post=456,
        page_url="https://example.com/q/456",
        question_text="Discrete count question (open upper)",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=-0.5,
        upper_bound=7.5,
        open_lower_bound=False,
        open_upper_bound=True,
        unit_of_measure="",
        zero_point=None,
        cdf_size=9,
    )

    decl_a = [
        Percentile(value=0.0, percentile=0.05),
        Percentile(value=3.0, percentile=0.45),
        Percentile(value=7.0, percentile=0.88),
    ]
    decl_b = [
        Percentile(value=1.0, percentile=0.18),
        Percentile(value=4.0, percentile=0.52),
        Percentile(value=6.8, percentile=0.87),
    ]
    dist_a = NumericDistribution(declared_percentiles=decl_a, **question.model_dump())
    dist_b = NumericDistribution(declared_percentiles=decl_b, **question.model_dump())

    agg = aggregate_numeric([dist_a, dist_b], question, "median")

    cdf = agg.cdf
    probs = np.array([p.percentile for p in cdf], dtype=float)
    values = np.array([p.value for p in cdf], dtype=float)

    assert len(cdf) == question.cdf_size
    required_min_step = 0.01 / (question.cdf_size - 1)
    diffs = np.diff(probs)
    assert np.all(diffs >= required_min_step - 1e-12)
    # Endpoint semantics: closed lower = 0.0, open upper ≤ 0.999
    assert abs(probs[0] - 0.0) <= 1e-12
    assert probs[-1] <= 0.999 + 1e-12
    expected_values = np.linspace(question.lower_bound, question.upper_bound, question.cdf_size)
    assert np.allclose(values, expected_values)


def _discrete_open_upper_question() -> NumericQuestion:
    return NumericQuestion(
        id_of_question=38880,
        id_of_post=38880,
        page_url="https://example.com/q/38880",
        question_text="Discrete low-count question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=-0.5,
        upper_bound=7.5,
        open_lower_bound=False,
        open_upper_bound=True,
        unit_of_measure="",
        zero_point=None,
        cdf_size=9,
    )


def test_ensemble_discrete_resample_does_not_clip_concentrated_low_count():
    """Regression: the discrete-resample branch of _postprocess_ensemble_cdf must not clip P(0) to 0.2.

    When the aggregated CDF arrives on a finer grid than the question's cdf_size
    (``len(x_vals) != cdf_size``), the branch resamples via ``generate_pchip_cdf``.
    On a 9-point grid the server's max-step is 0.2*200/8 = 5.0 (vacuous), so a
    low-count consensus (most mass on 0) must keep P(0) well above the old
    201-grid 0.2 cap this branch used to inherit.
    """
    question = _discrete_open_upper_question()

    # Concentrated-low aggregated CDF on the 201-point grid (mismatched length ->
    # is_discrete=True -> discrete-resample branch). ~32% of mass on 0.
    x = np.linspace(question.lower_bound, question.upper_bound, 201)
    below_one = np.minimum(1.0, np.maximum(0.0, (x + 0.5) / 1.0))
    above_one = np.maximum(0.0, (x - 0.5) / 7.0)
    p = np.maximum.accumulate(np.clip(0.32 * below_one + 0.66 * above_one, 0.0, 0.999))

    dist = _postprocess_ensemble_cdf(x, p, question, "median")
    probs = np.array([pp.percentile for pp in dist.cdf], dtype=float)

    assert len(probs) == question.cdf_size
    p_zero = probs[1] - probs[0]
    assert p_zero > 0.25, f"P(0)={p_zero} clipped to the 0.2 cap"
    diffs = np.diff(probs)
    server_max_step = min(1.0, 0.2 * 200.0 / (question.cdf_size - 1))
    assert np.all(diffs <= server_max_step + 1e-9)
    assert np.all(diffs >= 0.01 / (question.cdf_size - 1) - 1e-12)


def test_ensemble_median_ramp_does_not_overflow_above_one():
    """Regression: the continuous branch's min-step ramp must not push interior CDF > 1.0.

    On a concentrated low-count discrete question the aggregated median CDF has
    sub-min-step gaps near the top bins. The ramp (``p_vals + linspace(...)``) lifts
    every point, and ``_pin_endpoints`` only fixes ``[0]`` and ``[-1]`` — so an interior
    value could land above 1.0 (e.g. 1.0026) and crash ``Percentile`` validation
    (``percentile <= 1``), dropping the whole question. The aggregated CDF must instead
    be routed through ``safe_cdf_bounds`` and emerge as a valid submission.
    """
    for open_upper in (False, True):
        question = _discrete_open_upper_question() if open_upper else _closed_discrete_question()
        x = np.linspace(question.lower_bound, question.upper_bound, question.cdf_size)
        # Valid monotonic CDF in [0, 1] with sub-min-step gaps near the top bins that force
        # the ramp branch (the exact shape the 2026-07-20 audit used to trigger the crash).
        p = np.array([0.0, 0.6, 0.9, 0.97, 0.99, 0.994, 0.997, 0.9993, 1.0])
        assert bool(np.all(np.diff(p) > 0))
        assert p.min() >= 0.0
        assert p.max() <= 1.0
        assert float(np.diff(p).min()) < 0.01 / (question.cdf_size - 1), "test setup must trigger the ramp"

        dist = _postprocess_ensemble_cdf(x, p.copy(), question, "median")
        probs = np.array([pp.percentile for pp in dist.cdf], dtype=float)

        assert len(probs) == question.cdf_size
        assert probs.min() >= 0.0, f"interior overflow: max={probs.max()}"
        assert probs.max() <= 1.0 + 1e-12, f"interior overflow: max={probs.max()}"
        diffs = np.diff(probs)
        assert np.all(diffs > 0.0), "CDF must be strictly increasing"
        min_step = 0.01 / (question.cdf_size - 1)
        max_step = min(1.0, 0.2 * 200.0 / (question.cdf_size - 1))
        assert np.all(diffs >= min_step - 1e-12), f"min-step violation: {diffs.min()}"
        assert np.all(diffs <= max_step + 1e-9), f"max-step violation: {diffs.max()}"
        if open_upper:
            assert probs[-1] <= 0.999 + 1e-12
        else:
            assert abs(probs[-1] - 1.0) <= 1e-9
        assert abs(probs[0] - 0.0) <= 1e-9


def _closed_discrete_question() -> NumericQuestion:
    return NumericQuestion(
        id_of_question=38881,
        id_of_post=38881,
        page_url="https://example.com/q/38881",
        question_text="Discrete low-count question (closed upper)",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=-0.5,
        upper_bound=7.5,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
        cdf_size=9,
    )


def test_ensemble_aligned_discrete_grid_preserves_shape():
    """When per-model CDFs already sit on the cdf_size grid, aggregation keeps the shape.

    This is the branch that actually fires in prod (per-model CDFs are pre-resampled
    to cdf_size by build_numeric_distribution), so len(x_vals) == cdf_size and the
    continuous branch runs — it must not clip a concentrated low-count consensus.
    """
    question = _discrete_open_upper_question()
    decls = [
        [
            Percentile(value=0.0, percentile=0.30),
            Percentile(value=2.0, percentile=0.75),
            Percentile(value=5.0, percentile=0.97),
        ],
        [
            Percentile(value=0.0, percentile=0.28),
            Percentile(value=2.0, percentile=0.72),
            Percentile(value=5.0, percentile=0.96),
        ],
    ]
    dists = [NumericDistribution(declared_percentiles=d, **question.model_dump()) for d in decls]

    for method in ("mean", "median"):
        agg = aggregate_numeric(dists, question, method)
        probs = np.array([p.percentile for p in agg.cdf], dtype=float)
        assert probs[1] - probs[0] > 0.25, f"{method}: P(0) unexpectedly clipped"


def _fine_grid_question() -> NumericQuestion:
    return NumericQuestion(
        id_of_question=45065,
        id_of_post=44916,
        page_url="https://example.com/q/44916",
        question_text="Continuous question on the standard grid",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
        cdf_size=201,
    )


def test_ensemble_clip_marker_names_the_stage_and_the_question(caplog):
    """The continuous-branch CDF_MAXSTEP_CLIP must carry ensemble_median and the question id.

    The question id is the telemetry archive's join key, so an unlabeled ensemble
    clip is unjoinable. The clip is NOT reachable through public aggregate_numeric —
    a pointwise median of already-capped member CDFs is itself capped — so the
    private helper is the honest producer to pin.
    """
    question = _fine_grid_question()
    x = np.linspace(question.lower_bound, question.upper_bound, 201)
    p = 1.0 / (1.0 + np.exp(-(x - 50.0) / 0.2))
    p[0], p[-1] = 0.0, 1.0
    p = np.maximum.accumulate(p)

    with caplog.at_level(logging.WARNING, logger="metaculus_bot.numeric.pchip_cdf"):
        _postprocess_ensemble_cdf(x, p, question, "median")

    markers = [r.getMessage() for r in caplog.records if "CDF_MAXSTEP_CLIP:" in r.getMessage()]
    assert len(markers) == 1
    assert f"question={question.id_of_question}" in markers[0]
    assert "model=ensemble_median" in markers[0]
