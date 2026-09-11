"""Fine-grid (``cdf_size > 201``) behaviour of the numeric CDF pipeline.

Mantic quantitative questions run up to 2,000 bins (2,001-point CDFs) and a live preseason
question has ``inbound_outcome_count = 450`` (451 points). Metaculus grids are 201 points or
fewer, so nothing above 201 had ever been exercised. Three things are pinned here.

1. ``grid_step_constraints`` reproduces the server's per-bin rules exactly, with no floor:
   ``round(0.01 / N, 9)`` for the min step and, for the max step, the largest 9-decimal value
   not exceeding ``0.2 * 200 / N`` (``N = cdf_size - 1``). The retired ``5e-5`` min-step floor
   was a no-op for ``N <= 200`` and STRICTER than the server above it: at 450 bins it demanded
   ``5e-5`` per step where the server demands ``2.22e-5``, forcing ~2.25% of total mass into a
   uniform mixture instead of 1%; at 2,000 bins it forced 10% instead of 1%. The max step is
   floored to 9 decimals because the server rounds the PMF to 9 decimals and compares it against
   the UNROUNDED cap: at 450 bins the raw cap 0.0888... rounds up to 0.088888889, so a bin the
   max-step repair had clipped to exactly the raw cap was rejected with HTTP 400.
2. The per-model build (``sanitize_percentiles`` -> ``build_numeric_distribution``, the
   runner path) and the ensemble aggregation (``aggregate_numeric``, the aggregation-pipeline
   path) publish a CDF at 451 and 2,001 points that the server accepts after its own
   rounding, and whose open-bound outer tails carry only the mass the server's min-step
   forces, below what the old floor forced. This holds for broad forecasts and for forecasts
   tight enough that the max-step cap binds on every member.
3. A log-scaled (``zero_point``) question on a fine grid keeps its geometric value axis. The
   server maps the submitted probabilities positionally onto ITS bin edges, geometric on a
   ``zero_point`` question, so probabilities built on a linear axis would publish a different
   distribution from the one the forecasters declared.

The server check is ``assert_server_accepts_cdf`` in ``tests/pipeline_test_helpers.py``,
replicated from the open-source Metaculus backend (``questions/serializers/common.py``), which
Mantic forked with identical constants.
"""

from __future__ import annotations

import logging
from typing import ClassVar, NamedTuple

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import NumericQuestion
from scipy.stats import norm

from metaculus_bot.constants import NUM_MIN_PROB_STEP
from metaculus_bot.numeric.config import (
    EXPECTED_PERCENTILE_COUNT,
    MAX_CDF_PROB_STEP,
    PCHIP_CDF_POINTS,
    STANDARD_PERCENTILES,
    grid_step_constraints,
)
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric.utils import aggregate_numeric
from tests.pipeline_test_helpers import assert_server_accepts_cdf, server_max_step, server_min_step


class TestGridStepConstraintsMatchServer:
    @pytest.mark.parametrize("num_points", [9, 13, 51, 201, 451, 2001])
    def test_min_step_is_the_server_formula(self, num_points: int) -> None:
        assert grid_step_constraints(num_points)[0] == server_min_step(num_points - 1)

    @pytest.mark.parametrize("num_points", [9, 13, 51, 61, 201, 451, 2001])
    def test_max_step_is_the_largest_9_decimal_value_under_the_server_cap(self, num_points: int) -> None:
        """The server compares ``round(pmf, 9)`` against the UNROUNDED cap ``0.2 * 200 / N``.

        A bin clipped to a cap that is not 9-decimal exact can therefore round above it, so the
        max step is the largest 9-decimal value not exceeding the cap: equal to the cap wherever
        the cap is 9-decimal exact (9, 13, 51, 201 and 2,001 points), one unit of the ninth
        decimal under it elsewhere (61 and 451 points).
        """
        cap = min(1.0, server_max_step(num_points - 1))
        max_step = grid_step_constraints(num_points)[1]
        assert max_step <= cap
        assert round(max_step, 9) == max_step, "the server's rounding must not be able to lift it"
        assert max_step + 1e-9 > cap, "no 9-decimal value fits between the max step and the cap"

    def test_451_raw_cap_rounds_above_itself_and_the_floored_step_does_not(self) -> None:
        """The live Preseason 2 bitcoin grid: 450 bins, cap 0.0888...

        ``np.round(0.0888..., 9) = 0.088888889 > cap``, so the max-step repair's clip to exactly
        the raw cap was rejected by the server; the floored step survives the same rounding.
        """
        cap = server_max_step(450)
        assert np.round(cap, 9) > cap
        max_step = grid_step_constraints(451)[1]
        assert max_step == 0.088888888
        assert np.round(max_step, 9) <= cap

    def test_min_step_carries_the_server_rounding(self) -> None:
        """The server compares ``round(diff, 9) >= round(0.01 / N, 9)``.

        An unrounded ``0.01 / 450`` (2.2222222222e-5) would be a stricter floor than the
        server's 2.2222e-5, so the rounding is part of the contract.
        """
        assert grid_step_constraints(451)[0] == 2.2222e-05
        assert grid_step_constraints(13)[0] == 0.000833333
        assert grid_step_constraints(2001)[0] == 5e-06

    @pytest.mark.parametrize("num_points", [451, 2001])
    def test_fine_grid_min_step_is_below_the_201_grid_value(self, num_points: int) -> None:
        """No floor: a fine grid demands LESS per bin than the 201-point grid, as the server does."""
        min_step, max_step = grid_step_constraints(num_points)
        assert min_step < NUM_MIN_PROB_STEP
        assert max_step < MAX_CDF_PROB_STEP

    def test_standard_grid_is_unchanged(self) -> None:
        assert grid_step_constraints(PCHIP_CDF_POINTS) == (NUM_MIN_PROB_STEP, MAX_CDF_PROB_STEP) == (5e-5, 0.2)


class _GridSpec(NamedTuple):
    """One fine-grid question plus the ``(mean, sd)`` of three forecasters that broadly agree.

    Every declared percentile sits inside the displayed range and well clear of both edges,
    so on an open side the outermost bins carry no declared mass.
    """

    cdf_size: int
    lower_bound: float
    upper_bound: float
    members: tuple[tuple[float, float], ...]


# The live Mantic preseason bitcoin question: 450 bins over $54,950 to $99,950.
_BITCOIN_451 = _GridSpec(451, 54_950.0, 99_950.0, ((78_000.0, 6_000.0), (80_000.0, 7_000.0), (76_500.0, 5_500.0)))
# The same grid with forecasters tight enough that the max-step cap binds: a normal with sd $250
# puts ~16% of its mass in one $100 bin where the 450-bin cap allows 8.9%, so every member is clipped.
_BITCOIN_451_TIGHT = _GridSpec(451, 54_950.0, 99_950.0, ((78_000.0, 250.0), (78_100.0, 260.0), (77_900.0, 240.0)))
# A 2,000-bin count over [0, 2000], the largest grid Mantic issues.
_COUNT_2001 = _GridSpec(2001, 0.0, 2000.0, ((1_000.0, 120.0), (1_040.0, 140.0), (960.0, 110.0)))

_GRID_CASES = [
    pytest.param(_BITCOIN_451, True, True, id="bitcoin-451-open-open"),
    pytest.param(_BITCOIN_451, False, False, id="bitcoin-451-closed-closed"),
    pytest.param(_BITCOIN_451_TIGHT, True, True, id="bitcoin-451-tight-open-open"),
    pytest.param(_BITCOIN_451_TIGHT, False, False, id="bitcoin-451-tight-closed-closed"),
    pytest.param(_COUNT_2001, True, True, id="count-2001-open-open"),
    pytest.param(_COUNT_2001, False, False, id="count-2001-closed-closed"),
    pytest.param(_COUNT_2001, False, True, id="count-2001-closed-open"),
]


def _question(
    cdf_size: int,
    lower_bound: float,
    upper_bound: float,
    *,
    open_lower: bool,
    open_upper: bool,
    zero_point: float | None,
) -> NumericQuestion:
    return NumericQuestion(
        id_of_question=650,
        id_of_post=650,
        page_url="https://competitions.mantic.com/questions/650",
        question_text="Fine-grid quantitative question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        open_lower_bound=open_lower,
        open_upper_bound=open_upper,
        unit_of_measure="",
        zero_point=zero_point,
        cdf_size=cdf_size,
    )


def _linear_question(spec: _GridSpec, *, open_lower: bool, open_upper: bool) -> NumericQuestion:
    return _question(
        spec.cdf_size, spec.lower_bound, spec.upper_bound, open_lower=open_lower, open_upper=open_upper, zero_point=None
    )


def _normal_declaration(mean: float, sd: float) -> list[Percentile]:
    """The 13 standard percentiles of a normal forecast, as a forecaster would declare them."""
    return [Percentile(percentile=p, value=float(norm.ppf(p, loc=mean, scale=sd))) for p in STANDARD_PERCENTILES]


def _build_member(declaration: list[Percentile], question: NumericQuestion, model_name: str) -> NumericDistribution:
    """Run one forecaster's declaration through the runner path: sanitize, then build."""
    sanitized, zero_point = sanitize_percentiles(declaration, question, model_name=model_name)
    return build_numeric_distribution(sanitized, question, zero_point, model_name=model_name)


def _build_members(spec: _GridSpec, question: NumericQuestion) -> list[NumericDistribution]:
    return [
        _build_member(_normal_declaration(mean, sd), question, f"member-{index}")
        for index, (mean, sd) in enumerate(spec.members)
    ]


def _probs(distribution: NumericDistribution) -> np.ndarray:
    return np.asarray([p.percentile for p in distribution.get_cdf()], dtype=float)


def _values(distribution: NumericDistribution) -> np.ndarray:
    return np.asarray([p.value for p in distribution.get_cdf()], dtype=float)


def _outer_tail_masses(probs: np.ndarray, fraction: float = 0.05) -> tuple[int, float, float]:
    """(bin count, low-side mass, high-side mass) of the outermost ``fraction`` of bins per side."""
    n_bins = len(probs) - 1
    n_outer = round(fraction * n_bins)
    return n_outer, float(probs[n_outer] - probs[0]), float(probs[-1] - probs[-1 - n_outer])


@pytest.mark.parametrize(("spec", "open_lower", "open_upper"), _GRID_CASES)
class TestFineGridPublishPath:
    def test_each_member_cdf_is_server_valid(self, spec: _GridSpec, open_lower: bool, open_upper: bool) -> None:
        question = _linear_question(spec, open_lower=open_lower, open_upper=open_upper)
        expected_grid = np.linspace(spec.lower_bound, spec.upper_bound, spec.cdf_size)
        for member in _build_members(spec, question):
            assert_server_accepts_cdf(
                _probs(member), cdf_size=spec.cdf_size, open_lower=open_lower, open_upper=open_upper
            )
            np.testing.assert_allclose(_values(member), expected_grid, rtol=0, atol=1e-9)

    @pytest.mark.parametrize("method", ["median", "mean"])
    def test_aggregate_is_server_valid(self, spec: _GridSpec, open_lower: bool, open_upper: bool, method: str) -> None:
        question = _linear_question(spec, open_lower=open_lower, open_upper=open_upper)
        aggregated = aggregate_numeric(_build_members(spec, question), question, method)
        probs = _probs(aggregated)
        assert_server_accepts_cdf(probs, cdf_size=spec.cdf_size, open_lower=open_lower, open_upper=open_upper)

        grid_values = np.linspace(spec.lower_bound, spec.upper_bound, spec.cdf_size)
        aggregate_median = float(np.interp(0.5, probs, grid_values))
        member_medians = sorted(mean for mean, _ in spec.members)
        assert member_medians[0] <= aggregate_median <= member_medians[-1]

    def test_open_tails_carry_only_the_server_floor(self, spec: _GridSpec, open_lower: bool, open_upper: bool) -> None:
        """On an open side the outermost 5% of bins hold no declared mass, so their mass is the floor.

        The retired ``5e-5`` floor would have forced at least ``n_outer * 5e-5`` there (twice the
        server's requirement at 450 bins, ten times at 2,000). The bound is computed from the grid,
        not hardcoded, so it tracks whatever grid the case uses.
        """
        if not (open_lower or open_upper):
            pytest.skip("both bounds closed: PCHIP anchors the CDF at the bound, so the tails carry declared mass")
        question = _linear_question(spec, open_lower=open_lower, open_upper=open_upper)
        probs = _probs(aggregate_numeric(_build_members(spec, question), question, "median"))
        n_outer, low_mass, high_mass = _outer_tail_masses(probs)
        server_min_step = grid_step_constraints(spec.cdf_size)[0]
        old_floor_forced = n_outer * NUM_MIN_PROB_STEP
        new_floor_forced = n_outer * server_min_step
        assert old_floor_forced > new_floor_forced

        if open_lower:
            assert new_floor_forced - 1e-9 <= low_mass < old_floor_forced, (
                f"low tail mass {low_mass} not in [{new_floor_forced}, {old_floor_forced})"
            )
        if open_upper:
            assert new_floor_forced - 1e-9 <= high_mass < old_floor_forced, (
                f"high tail mass {high_mass} not in [{new_floor_forced}, {old_floor_forced})"
            )


@pytest.mark.parametrize("open_bounds", [True, False], ids=["open-open", "closed-closed"])
def test_tight_ensemble_clips_to_exactly_the_cap_and_still_publishes(open_bounds: bool, caplog) -> None:
    """The max-step repair clips every over-cap bin to exactly ``max_step``.

    That puts the clipped bin on the edge of what the server accepts, which is where the raw
    (unfloored) 450-bin cap was rejected. Asserting the clip fired keeps the case from going
    vacuous if the forecasters are ever widened.
    """
    question = _linear_question(_BITCOIN_451_TIGHT, open_lower=open_bounds, open_upper=open_bounds)
    max_step = grid_step_constraints(_BITCOIN_451_TIGHT.cdf_size)[1]

    with caplog.at_level(logging.WARNING, logger="metaculus_bot.numeric.pchip_cdf"):
        members = _build_members(_BITCOIN_451_TIGHT, question)
    clip_markers = [r.getMessage() for r in caplog.records if "CDF_MAXSTEP_CLIP:" in r.getMessage()]
    assert len(clip_markers) == len(_BITCOIN_451_TIGHT.members)

    for member in members:
        assert np.diff(_probs(member)).max() == pytest.approx(max_step, abs=1e-12)
    for distribution in (*members, aggregate_numeric(members, question, "median")):
        assert_server_accepts_cdf(
            _probs(distribution), cdf_size=_BITCOIN_451_TIGHT.cdf_size, open_lower=open_bounds, open_upper=open_bounds
        )


class TestLogScaledFineGridKeepsTheGeometricAxis:
    """A ``zero_point`` question on a non-201 grid publishes against the geometric axis it declares.

    Until 2026-09 every non-201 grid forced ``zero_point=None``, so the probabilities were built on
    a LINEAR axis and then mapped by the server onto its GEOMETRIC bins: on this [1, 1e6] question
    a forecast centred on 1,000 read back as a median near 1.2. Metaculus never reached the branch
    (its log-scaled questions are always 201 points); Mantic's fine grids do.
    """

    _LOWER, _UPPER, _ZERO_POINT, _CDF_SIZE = 1.0, 1e6, 0.0, 451
    # Log-normal forecasters centred near 1,000 with a 0.3-decade spread: P1 about 200, P99 about 5,000.
    _MEMBERS = ((3.0, 0.3), (3.05, 0.3), (2.95, 0.3))

    def _question(self, *, open_bounds: bool) -> NumericQuestion:
        return _question(
            self._CDF_SIZE,
            self._LOWER,
            self._UPPER,
            open_lower=open_bounds,
            open_upper=open_bounds,
            zero_point=self._ZERO_POINT,
        )

    def _members(self, question: NumericQuestion) -> list[NumericDistribution]:
        members = []
        for index, (log10_mean, log10_sd) in enumerate(self._MEMBERS):
            declaration = [
                Percentile(percentile=p, value=float(10 ** norm.ppf(p, loc=log10_mean, scale=log10_sd)))
                for p in STANDARD_PERCENTILES
            ]
            members.append(_build_member(declaration, question, f"member-{index}"))
        return members

    def test_sanitize_keeps_the_zero_point_on_a_fine_grid(self) -> None:
        question = self._question(open_bounds=False)
        declaration = [Percentile(percentile=p, value=1000.0 * (1 + p)) for p in STANDARD_PERCENTILES]
        _, zero_point = sanitize_percentiles(declaration, question, model_name="member")
        assert zero_point == self._ZERO_POINT

    @pytest.mark.parametrize("open_bounds", [False, True], ids=["closed-closed", "open-open"])
    def test_members_and_aggregate_publish_on_the_geometric_axis(self, open_bounds: bool) -> None:
        question = self._question(open_bounds=open_bounds)
        geometric = build_cdf_value_grid(self._LOWER, self._UPPER, self._ZERO_POINT, self._CDF_SIZE)
        members = self._members(question)
        aggregate = aggregate_numeric(members, question, "median")

        for distribution in (*members, aggregate):
            probs = _probs(distribution)
            np.testing.assert_allclose(_values(distribution), geometric, rtol=1e-12, atol=0)
            declared_values = np.asarray([p.value for p in distribution.declared_percentiles], dtype=float)
            np.testing.assert_allclose(declared_values, geometric, rtol=1e-12, atol=0)
            assert_server_accepts_cdf(probs, cdf_size=self._CDF_SIZE, open_lower=open_bounds, open_upper=open_bounds)
            median = float(np.interp(0.5, probs, geometric))
            assert 800.0 <= median <= 1250.0, f"median {median} is not where the forecasters put it"


class TestAllMassBeyondABoundStillBuilds:
    """A declaration that puts essentially all of its mass beyond an open bound builds and publishes.

    Mantic scores an out-of-range resolution against the mass the CDF leaves beyond the bound
    (``50 * ln(tail / 0.05)``), so a forecaster that follows the bound instruction and places all
    thirteen percentiles past a ceiling it believes is too low is stating the highest-scoring shape
    the platform has (+148.8 baseline points for a 0.98 tail). The in-range CDF is then exactly the
    min-step ramp, whose required range equals its available range to within float epsilon. The
    untoleranced rebuild trigger fired on that epsilon (a step 1e-18 short) and the untoleranced
    range check inside the rebuild then refused a range 1e-16 short, so every non-201 grid raised
    and dropped the member, and the 201-point grid fell through to the forecasting-tools builder,
    which failed on the same input. Because the trigger is a property of the question, agreeing
    forecasters failed together and the question published nothing (Mantic edge-case review,
    2026-09, rank 3). The trigger and the range check now share the tolerance the post-check and
    final assertion always had.

    The range is the live Preseason 2 bitcoin question's; the 15-point grid is the coarse extreme,
    2,001 the largest grid Mantic issues, and 201 the grid whose failure took the fallback route.
    """

    _LOWER, _UPPER = 54_950.0, 99_950.0
    _SPAN = _UPPER - _LOWER
    _ABOVE_CEILING = tuple(np.linspace(_UPPER + 0.01 * _SPAN, _UPPER + _SPAN, EXPECTED_PERCENTILE_COUNT))
    _BELOW_FLOOR = tuple(np.linspace(_LOWER - _SPAN, _LOWER - 0.01 * _SPAN, EXPECTED_PERCENTILE_COUNT))
    # Every declared percentile sits beyond the bound, so the mass beyond it is at least 1 - P1
    # less the uniform mixture the min-step forces into the interior.
    _MIN_OUT_OF_RANGE_MASS = 0.97

    _SHAPES: ClassVar[list] = [
        pytest.param(_ABOVE_CEILING, True, True, id="above-ceiling-open-floor"),
        pytest.param(_ABOVE_CEILING, False, True, id="above-ceiling-closed-floor"),
        pytest.param(_BELOW_FLOOR, True, True, id="below-floor-open-ceiling"),
        pytest.param(_BELOW_FLOOR, True, False, id="below-floor-closed-ceiling"),
    ]

    @pytest.mark.parametrize("cdf_size", [15, PCHIP_CDF_POINTS, 451, 2001])
    @pytest.mark.parametrize(("values", "open_lower", "open_upper"), _SHAPES)
    def test_member_builds_and_the_server_accepts_it(
        self, cdf_size: int, values: tuple[float, ...], open_lower: bool, open_upper: bool, caplog
    ) -> None:
        question = _question(
            cdf_size, self._LOWER, self._UPPER, open_lower=open_lower, open_upper=open_upper, zero_point=None
        )
        declaration = [Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, values, strict=True)]

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.numeric"):
            member = _build_member(declaration, question, "member")

        # Built by the PCHIP path on the question's own grid, not rescued by the forecasting-tools
        # fallback: ``_pchip_cdf_values`` is the marker the pipeline itself keys on.
        assert hasattr(member, "_pchip_cdf_values")
        probs = _probs(member)
        assert_server_accepts_cdf(probs, cdf_size=cdf_size, open_lower=open_lower, open_upper=open_upper)
        beyond_the_bound = float(probs[0]) if values is self._BELOW_FLOOR else float(1.0 - probs[-1])
        assert beyond_the_bound >= self._MIN_OUT_OF_RANGE_MASS, f"only {beyond_the_bound:.4f} left beyond the bound"
        messages = [record.getMessage() for record in caplog.records]
        assert not any("PCHIP minimum step enforcement required" in m for m in messages), (
            "the epsilon ramp is not a repair"
        )
        assert not any("PCHIP_FALLBACK" in m or "fallback" in m.lower() for m in messages), messages
