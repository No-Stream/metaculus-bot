"""Regression tests for the discrete-grid max-step bug.

`generate_pchip_cdf` used to hard-code the 201-point-grid max-step constant
(`NUM_MAX_STEP = 0.2`) inside `safe_cdf_bounds`, even when resampling a
per-model CDF onto a coarse discrete grid (`cdf_size < 201`). On a 9-point grid
the server's own max-step is `0.2 * 200 / (cdf_size - 1) = 5.0` (effectively
unconstrained), so the 0.2 cap wrongly clipped every integer's probability to
20% and shoved the excess mass onto higher integers — a systematic upward shift
on small-count questions (e.g. Q38880).

These tests pin the corrected behaviour: the max-step passed to the CDF builder
scales with the grid, so a concentrated low-count distribution keeps its mass on
the low integers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.constants import NUM_MAX_STEP
from metaculus_bot.numeric.config import grid_step_constraints
from metaculus_bot.numeric.pchip_cdf import generate_pchip_cdf
from metaculus_bot.numeric.pchip_processing import generate_pchip_cdf_with_smoothing
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles

# grok's actual concentrated declared percentiles for Q38880 (count 0-7,
# open upper), padded to the 13 canonical percentiles. P(0) faithful ~0.3.
_CONCENTRATED_LOW_COUNT: list[tuple[float, float]] = [
    (0.01, 0.02),
    (0.025, 0.05),
    (0.05, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.40, 0.65),
    (0.50, 0.90),
    (0.60, 1.20),
    (0.80, 2.20),
    (0.90, 3.20),
    (0.95, 4.20),
    (0.975, 5.20),
    (0.99, 6.60),
]


def _discrete_count_question(**overrides) -> NumericQuestion:
    base: dict[str, Any] = dict(
        id_of_question=38880,
        id_of_post=38880,
        page_url="https://example.com/q/38880",
        question_text="How many X will happen? (count 0-7)",
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
    base.update(overrides)
    return NumericQuestion(**base)


class TestGridStepConstraints:
    def test_201_grid_matches_legacy_constants(self):
        min_step, max_step = grid_step_constraints(201)
        assert min_step == pytest.approx(5e-5)
        assert max_step == pytest.approx(NUM_MAX_STEP)  # 0.2 unchanged on the standard grid

    def test_coarse_grid_relaxes_max_step(self):
        # cdf_size=9 -> inbound=8 -> server max-step 0.2*200/8 = 5.0, clamped to 1.0.
        min_step, max_step = grid_step_constraints(9)
        assert min_step == pytest.approx(0.01 / 8)
        assert max_step == pytest.approx(1.0)

    def test_fine_grid_tightens_max_step(self):
        # cdf_size=401 -> inbound=400 -> server max-step 0.2*200/400 = 0.1.
        min_step, max_step = grid_step_constraints(401)
        assert min_step == pytest.approx(5e-5)  # floored at NUM_MIN_PROB_STEP
        assert max_step == pytest.approx(0.1)


class TestGeneratePchipCdfMaxStep:
    def test_coarse_grid_does_not_clip_concentrated_mass(self):
        """A concentrated low-count distribution keeps P(0) > 0.25 on a 9-pt grid."""
        pv = {20.0: 0.30, 40.0: 0.65, 50.0: 0.90, 80.0: 2.20, 90.0: 3.20, 99.0: 6.60}
        min_step, max_step = grid_step_constraints(9)
        cdf, _ = generate_pchip_cdf(
            percentile_values=pv,
            open_upper_bound=True,
            open_lower_bound=False,
            upper_bound=7.5,
            lower_bound=-0.5,
            min_step=min_step,
            max_step=max_step,
            num_points=9,
        )
        cdf = np.array(cdf)
        # cdf[0] == F(-0.5) (closed lower -> 0), cdf[1] == F(0.5) == P(0).
        assert cdf[0] == pytest.approx(0.0, abs=1e-9)
        p_zero = cdf[1] - cdf[0]
        assert p_zero > 0.25, f"P(0)={p_zero} was clipped (faithful ~0.3)"

    def test_201_default_still_caps_at_020(self):
        """Adversarial spike on the 201 grid is still redistributed to <= 0.2 (default)."""
        pv = {float(p): v for p, v in [(10, 5.0), (50, 5.5), (90, 6.0)]}
        cdf, _ = generate_pchip_cdf(
            percentile_values=pv,
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=100.0,
            lower_bound=0.0,
            num_points=201,
        )
        steps = np.diff(np.array(cdf))
        assert float(np.max(steps)) <= NUM_MAX_STEP + 1e-6


class TestBuildNumericDistributionDiscrete:
    def test_q38880_repro_no_low_bin_clip(self):
        """build_numeric_distribution must not clip the low integers to 0.2 on cdf_size=9."""
        question = _discrete_count_question()
        percentiles = [Percentile(percentile=p, value=v) for p, v in _CONCENTRATED_LOW_COUNT]
        sanitized, zero_point = sanitize_percentiles(percentiles, question)
        prediction = build_numeric_distribution(sanitized, question, zero_point)

        cdf = prediction.cdf
        assert len(cdf) == question.cdf_size
        probs = np.array([p.percentile for p in cdf], dtype=float)

        # P(0) = F(0.5) - F(-0.5); closed lower pins F(-0.5)=0.
        p_zero = probs[1] - probs[0]
        assert p_zero > 0.25, f"P(0)={p_zero} clipped to the 0.2 cap"

        # Server constraints for the 9-point grid still hold.
        diffs = np.diff(probs)
        required_min_step = 0.01 / (question.cdf_size - 1)
        server_max_step = min(1.0, 0.2 * 200.0 / (question.cdf_size - 1))
        assert np.all(diffs >= required_min_step - 1e-12)
        assert np.all(diffs <= server_max_step + 1e-9)
        assert probs[0] == pytest.approx(0.0, abs=1e-9)  # closed lower
        assert probs[-1] <= 0.999 + 1e-9  # open upper

    def test_resample_tracks_the_201_grid_shape(self):
        """The 9-point resample interpolates the full 201-point CDF at the grid points."""
        question = _discrete_count_question()
        percentiles = [Percentile(percentile=p, value=v) for p, v in _CONCENTRATED_LOW_COUNT]
        sanitized, zero_point = sanitize_percentiles(percentiles, question)

        # Full-resolution reference CDF from the same percentiles.
        ref_cdf, _, _ = generate_pchip_cdf_with_smoothing(sanitized, question, zero_point)
        ref_x = np.linspace(question.lower_bound, question.upper_bound, len(ref_cdf))

        prediction = build_numeric_distribution(sanitized, question, zero_point)
        probs = np.array([p.percentile for p in prediction.cdf], dtype=float)
        grid_x = np.array([p.value for p in prediction.cdf], dtype=float)

        ref_at_grid = np.interp(grid_x, ref_x, ref_cdf)
        assert np.allclose(probs, ref_at_grid, atol=0.03)
