"""Regression tests for the two discrete-grid max-step bugs.

**Wrong cap (2026-06).** `generate_pchip_cdf` used to hard-code the 201-point-grid
max-step constant (`NUM_MAX_STEP = 0.2`) inside `safe_cdf_bounds`, even when resampling a
per-model CDF onto a coarse discrete grid (`cdf_size < 201`). On a 9-point grid the
server's own max-step is `0.2 * 200 / (cdf_size - 1) = 5.0` (effectively unconstrained),
so the 0.2 cap wrongly clipped every integer's probability to 20% and shoved the excess
mass onto higher integers — a systematic upward shift on small-count questions (e.g.
Q38880). `grid_step_constraints` now scales the cap with the grid.

**Wrong destination for the excess (2026-08).** On a FINE grid the 0.2 cap is the
platform's own and correctly binds, but `_redistribute_excess_probability` handed the
clipped excess out in proportion to each bin's SLACK — near-uniform on a grid whose other
bins are near-empty. q45065 published 47% of its mass above 35 deaths where all three
forecasters had declared ~2%. Packing is nearest-first now
(`_pack_excess_nearest_first`), and the two regimes are pinned against each other here:
a coarse grid must not redistribute at all, a fine grid must keep the excess adjacent.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.constants import NUM_MAX_STEP, NUM_MIN_PROB_STEP
from metaculus_bot.numeric.config import grid_step_constraints
from metaculus_bot.numeric.discrete_snap import snap_distribution_to_integers
from metaculus_bot.numeric.pchip_cdf import generate_pchip_cdf
from metaculus_bot.numeric.pchip_processing import (
    create_fallback_numeric_distribution,
    generate_pchip_cdf_with_smoothing,
)
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric.utils import aggregate_numeric

_PCHIP_LOGGER = "metaculus_bot.numeric.pchip_cdf"

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


# Same question shape, but with a tight spike at 3 (P40-P60 within +-0.01). On the
# STANDARD 201-point grid this declaration breaches the 0.2 cap (one ~0.04-wide bin
# holds >20% of mass), while the published 9-point CDF has a 1.0 cap and never clips —
# so any clip marker from building it on the cdf_size=9 question is a phantom.
_SPIKED_LOW_COUNT: list[tuple[float, float]] = [
    (0.01, 0.30),
    (0.025, 0.80),
    (0.05, 1.30),
    (0.10, 1.90),
    (0.20, 2.50),
    (0.40, 2.99),
    (0.50, 3.00),
    (0.60, 3.01),
    (0.80, 3.60),
    (0.90, 4.30),
    (0.95, 5.00),
    (0.975, 5.70),
    (0.99, 6.60),
]


def _clip_markers(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "CDF_MAXSTEP_CLIP:" in r.getMessage()]


def _discrete_count_question(**overrides) -> NumericQuestion:
    base: dict[str, Any] = {
        "id_of_question": 38880,
        "id_of_post": 38880,
        "page_url": "https://example.com/q/38880",
        "question_text": "How many X will happen? (count 0-7)",
        "background_info": "",
        "resolution_criteria": "",
        "fine_print": "",
        "published_time": None,
        "close_time": None,
        "lower_bound": -0.5,
        "upper_bound": 7.5,
        "open_lower_bound": False,
        "open_upper_bound": True,
        "unit_of_measure": "",
        "zero_point": None,
        "cdf_size": 9,
    }
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

    def test_coarse_grid_never_reaches_the_max_step_repair(self, caplog):
        """The relaxed coarse-grid cap means the repair does not fire at all there.

        This is the guard that keeps the two regimes from crossing: nothing about
        nearest-first packing may reintroduce clipping on a grid whose server cap is 1.0.
        The declaration is deliberately tight enough to clip on the standard 201-point
        grid, so a provisional fine-grid build leaking a phantom marker fails here too.
        """
        question = _discrete_count_question()
        percentiles = [Percentile(percentile=p, value=v) for p, v in _SPIKED_LOW_COUNT]
        sanitized, zero_point = sanitize_percentiles(percentiles, question)

        with caplog.at_level(logging.WARNING, logger=_PCHIP_LOGGER):
            build_numeric_distribution(sanitized, question, zero_point, model_name="some/forecaster")

        assert not _clip_markers(caplog)


# claude-opus-4.8's actual declared percentiles for q45065 (post 44916, "how many deaths
# will the U.S. government officially report", resolved 14.0). All three triple-era
# forecasters declared a near point-mass on 14; opus put 0.718 in the (13.5, 14.5] bin,
# which the platform's 0.2 cap cannot hold. Receipts: residual round q45065_capbug.md.
_Q45065_OPUS_PERCENTILES: list[tuple[float, float]] = [
    (0.01, 12.4),
    (0.025, 12.9),
    (0.05, 13.2),
    (0.10, 13.5),
    (0.20, 13.8),
    (0.40, 13.96),
    (0.50, 14.0),
    (0.60, 14.08),
    (0.80, 14.4),
    (0.90, 15.0),
    (0.95, 16.2),
    (0.975, 18.0),
    (0.99, 20.5),
]

# (13.5, 14.5] on linspace(9.5, 209.5, 201), i.e. the integer count 14 that resolved.
_Q45065_RESOLUTION_BIN = 4


def _q45065_question(**overrides) -> NumericQuestion:
    """201-point grid over [9.5, 209.5] — one bin per integer count 10..209, both bounds open."""
    base: dict[str, Any] = {
        "id_of_question": 45065,
        "id_of_post": 44916,
        "page_url": "https://www.metaculus.com/questions/44916/",
        "question_text": "How many deaths will the U.S. government officially report?",
        "background_info": "",
        "resolution_criteria": "",
        "fine_print": "",
        "published_time": None,
        "close_time": None,
        "lower_bound": 9.5,
        "upper_bound": 209.5,
        "open_lower_bound": True,
        "open_upper_bound": True,
        "unit_of_measure": "deaths",
        "zero_point": None,
        "cdf_size": 201,
    }
    base.update(overrides)
    return NumericQuestion(**base)


class TestQ45065NearestFirstPacking:
    """The fine-grid half of the family: the cap is right, its old destination was not.

    Under slack-proportional redistribution opus's clipped 0.518 spread near-uniformly over
    ~198 bins, so the published ensemble asserted a 47% chance of 35+ deaths against the
    forecasters' own ~2%. The realized score barely noticed (the answer landed in the
    declared bin) but a one-bin miss priced at +50 to +200 spot-peer points, which is what
    these pins protect.

    Those figures were +100 to +400 until 2026-09-02. The replay script they came from
    doubled every continuous peer delta, and Metaculus HALVES a continuous peer score --
    receipt in ``scratch/residual_2026-09-01/DOSSIER_SYNTHESIS.md`` section 7.2, convention
    pinned in ``tests/test_peer_delta_convention.py``.
    """

    _MODEL: ClassVar[str] = "openrouter/anthropic/claude-opus-4.8"

    def _build(self) -> np.ndarray:
        question = _q45065_question()
        declared = [Percentile(percentile=p, value=v) for p, v in _Q45065_OPUS_PERCENTILES]
        sanitized, zero_point = sanitize_percentiles(declared, question, model_name=self._MODEL)
        prediction = build_numeric_distribution(sanitized, question, zero_point, model_name=self._MODEL)
        probs = np.asarray([p.percentile for p in prediction.get_cdf()], dtype=float)
        assert len(probs) == 201
        return np.diff(probs)

    def test_resolving_bin_sits_exactly_at_the_platform_cap(self):
        steps = self._build()
        assert float(steps[_Q45065_RESOLUTION_BIN]) == pytest.approx(NUM_MAX_STEP, abs=1e-9)

    def test_declared_spike_stays_within_two_bins_of_the_resolution(self):
        lo, hi = _Q45065_RESOLUTION_BIN - 2, _Q45065_RESOLUTION_BIN + 3
        near = float(self._build()[lo:hi].sum())
        # Declared ~0.74 in this window; slack-proportional redistribution left ~0.43.
        assert near >= 0.60, f"only {near:.4f} of mass within +-2 bins of the resolution"

    def test_far_tail_no_longer_carries_the_displaced_mass(self):
        steps = self._build()
        far = float(steps[_Q45065_RESOLUTION_BIN + 21 :].sum())
        # 0.458 of opus's clipped mass used to land here (>= 35 deaths); declared ~0.02.
        assert far <= 0.05, f"{far:.4f} of mass landed 21+ bins above the resolution"

    def test_submission_constraints_hold(self):
        steps = self._build()
        assert float(steps.max()) <= NUM_MAX_STEP + 1e-12
        assert float(steps.min()) >= NUM_MIN_PROB_STEP - 1e-12
        assert np.all(steps >= 0.0)

    def test_published_composition_keeps_the_mass_where_it_was_declared(self):
        """The ensemble and snap stages must not undo the nearest-first packing.

        The 47%-above-35-deaths figure was measured on the PUBLISHED forecast —
        per-model build -> aggregate_numeric(median) -> snap_distribution_to_integers
        — and both later stages re-enter safe_cdf_bounds. Median of one is that
        forecast; the point is the two extra safe_cdf_bounds passes, not the combine.
        """
        question = _q45065_question()
        declared = [Percentile(percentile=p, value=v) for p, v in _Q45065_OPUS_PERCENTILES]
        sanitized, zero_point = sanitize_percentiles(declared, question, model_name=self._MODEL)
        per_model = build_numeric_distribution(sanitized, question, zero_point, model_name=self._MODEL)

        aggregated = aggregate_numeric([per_model], question, "median")
        snapped = snap_distribution_to_integers(aggregated, question)
        assert snapped is not None
        probs = np.asarray([p.percentile for p in snapped.get_cdf()], dtype=float)
        steps = np.diff(probs)

        near = float(steps[_Q45065_RESOLUTION_BIN - 2 : _Q45065_RESOLUTION_BIN + 3].sum())
        far = float(steps[_Q45065_RESOLUTION_BIN + 21 :].sum())
        assert near >= 0.60, f"only {near:.4f} of published mass within +-2 bins of the resolution"
        assert far <= 0.05, f"{far:.4f} of published mass landed 21+ bins above the resolution"
        assert float(steps.max()) <= NUM_MAX_STEP + 1e-12
        assert float(steps.min()) >= NUM_MIN_PROB_STEP - 1e-12

    def test_marker_names_the_forecaster_and_where_the_mass_went(self, caplog):
        question = _q45065_question()
        declared = [Percentile(percentile=p, value=v) for p, v in _Q45065_OPUS_PERCENTILES]
        sanitized, zero_point = sanitize_percentiles(declared, question, model_name=self._MODEL)

        with caplog.at_level(logging.WARNING, logger=_PCHIP_LOGGER):
            build_numeric_distribution(sanitized, question, zero_point, model_name=self._MODEL)

        markers = _clip_markers(caplog)
        assert len(markers) == 1
        assert f"question={question.id_of_question}" in markers[0]
        assert f"model={self._MODEL}" in markers[0]
        assert "over_cap_bins=1" in markers[0]
        # The displacement fields are what make the packing policy auditable from the log.
        assert "max_offset_bins=2" in markers[0]

    def test_ordinary_declaration_emits_no_marker(self, caplog):
        """A declaration wide enough to clear the cap must leave the log clean.

        Without this the WARN would read as routine and stop being a signal.
        """
        question = _q45065_question()
        wide = [(p, 20.0 + 150.0 * p) for p, _v in _Q45065_OPUS_PERCENTILES]
        declared = [Percentile(percentile=p, value=v) for p, v in wide]
        sanitized, zero_point = sanitize_percentiles(declared, question, model_name=self._MODEL)

        with caplog.at_level(logging.WARNING, logger=_PCHIP_LOGGER):
            prediction = build_numeric_distribution(sanitized, question, zero_point, model_name=self._MODEL)

        steps = np.diff(np.asarray([p.percentile for p in prediction.get_cdf()], dtype=float))
        assert float(steps.max()) < NUM_MAX_STEP
        assert not _clip_markers(caplog)


class TestFallbackDistributionMemoization:
    """BoundSafeNumericDistribution (the ft-fallback wrapper) memoizes get_cdf()."""

    def test_fallback_path_emits_one_marker_per_build_not_per_read(self, caplog):
        """One clip emits ONE marker, however many times the CDF is read.

        Un-memoized, every accessor read (validate + diagnostics + aggregate +
        publish) re-ran safe_cdf_bounds and re-emitted the marker, so fallback-path
        clips counted 3-4x against the PCHIP path's once in the telemetry archive.
        The declaration spans the range (upstream's builder rejects flat tails) but
        piles 60% of its mass inside one 0.5-wide bin, forcing an over-cap step.
        """
        question = _q45065_question(lower_bound=0.0, upper_bound=100.0, open_lower_bound=False, open_upper_bound=True)
        spiked = [
            (0.01, 1.0),
            (0.025, 3.0),
            (0.05, 6.0),
            (0.10, 12.0),
            (0.20, 50.02),
            (0.40, 50.06),
            (0.50, 50.10),
            (0.60, 50.14),
            (0.80, 50.20),
            (0.90, 75.0),
            (0.95, 88.0),
            (0.975, 94.0),
            (0.99, 99.0),
        ]
        declared = [Percentile(percentile=p, value=v) for p, v in spiked]
        fallback = create_fallback_numeric_distribution(declared, question, None, model_name="some/forecaster")

        with caplog.at_level(logging.WARNING, logger=_PCHIP_LOGGER):
            first = fallback.get_cdf()
            second = fallback.get_cdf()

        assert second is first
        assert len(_clip_markers(caplog)) == 1
