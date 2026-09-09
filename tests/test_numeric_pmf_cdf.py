"""The per-bin PMF to CDF builder (``metaculus_bot/numeric/pmf_cdf.py``).

On an enumerable Mantic grid a forecaster declares one probability per bin (plus ``below_range`` /
``above_range`` where a bound is open), the platform's own ``N + 2`` PMF shape. The builder
normalises that vector, blends it toward the server's exact per-cell floors with the smallest
weight that makes every cell legal, assembles the CDF, runs the shared ``safe_cdf_bounds``, pins
closed bounds, and fails SHUT through ``validate_grid_cdf`` before wrapping the result in the same
distribution object the discrete percentile path produces. Three layers are pinned here: the server's
``continuous_cdf`` rules on the output (five grid sizes by three bound shapes, then every coarse
grid Mantic has ever published), the floor-blend arithmetic (a certain bin keeps at least 0.988, a
zero bin lands at exactly the platform minimum), and the guards (mass in a closed tail is refused,
every server rule ``validate_grid_cdf`` replicates raises).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from forecasting_tools.data_models.questions import DiscreteQuestion, NumericQuestion

from metaculus_bot.constants import MANTIC_SITE_URL
from metaculus_bot.numeric.config import OPEN_TAIL_MIN_MASS, PMF_FLOOR_MARGIN, elicit_per_bin, grid_step_constraints
from metaculus_bot.numeric.date_axis import EpochDateQuestion, as_epoch_question
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid, safe_cdf_bounds
from metaculus_bot.numeric.pmf_cdf import build_pmf_distribution, published_pmf, validate_grid_cdf
from metaculus_bot.numeric.pmf_grid import pmf_grid
from metaculus_bot.numeric.utils import aggregate_numeric
from tests.mantic_fakes import load_preseason_date_question
from tests.pipeline_test_helpers import assert_server_accepts_cdf, server_min_step

GRIDS_PATH = Path(__file__).parent / "data" / "mantic_cdf_grids_2026_09_08.json"

# The plan's oracle set: five grid sizes across the elicitable range by the three bound shapes Mantic uses.
ORACLE_BIN_COUNTS = (3, 4, 12, 21, 31)
ORACLE_BOUND_SHAPES = (
    pytest.param(False, False, id="closed-closed"),
    pytest.param(False, True, id="closed-open"),
    pytest.param(True, True, id="open-open"),
)


def _corpus_coarse_grids() -> list[tuple[int, bool, bool]]:
    """Every distinct ``(bins, open_lower, open_upper)`` Mantic has published at or below the per-bin threshold."""
    with GRIDS_PATH.open() as f:
        grids = json.load(f)["grids"]
    return [
        (grid["inbound_outcome_count"], grid["open_lower_bound"], grid["open_upper_bound"])
        for grid in grids
        if grid["inbound_outcome_count"] <= 31
    ]


def _count_question(
    n_bins: int, *, open_lower: bool, open_upper: bool, zero_point: float | None = None
) -> NumericQuestion:
    """A Mantic count question on ``n_bins`` integer-centred bins, the modal coarse quantity shape."""
    fields = {
        "id_of_question": 900_000 + n_bins,
        "id_of_post": 900_000 + n_bins,
        "page_url": f"{MANTIC_SITE_URL}/questions/{900_000 + n_bins}/",
        "question_text": "How many?",
        "background_info": "",
        "resolution_criteria": "",
        "fine_print": "",
        "published_time": None,
        "close_time": None,
        "lower_bound": -0.5,
        "upper_bound": n_bins - 0.5,
        "open_lower_bound": open_lower,
        "open_upper_bound": open_upper,
        "unit_of_measure": "",
        "zero_point": zero_point,
        "cdf_size": n_bins + 1,
    }
    if zero_point is not None:
        fields["lower_bound"] = 1.0
        fields["upper_bound"] = 1000.0
        return NumericQuestion(**fields)
    return DiscreteQuestion(**fields)


# --- The declared PMF shapes, as ``N + 2`` vectors with 0.0 in a closed tail's slot ---


def _one_hot(index_of: Callable[[int], int]) -> Callable[[int, bool, bool], list[float]]:
    def declare(n_bins: int, open_lower: bool, open_upper: bool) -> list[float]:
        pmf = [0.0] * (n_bins + 2)
        pmf[1 + index_of(n_bins)] = 1.0
        return pmf

    return declare


def _uniform(n_bins: int, open_lower: bool, open_upper: bool) -> list[float]:
    return [0.0, *([1.0 / n_bins] * n_bins), 0.0]


def _half_half(n_bins: int, open_lower: bool, open_upper: bool) -> list[float]:
    pmf = [0.0] * (n_bins + 2)
    pmf[1] = 0.5
    pmf[n_bins] = 0.5
    return pmf


def _all_in_open_tail(n_bins: int, open_lower: bool, open_upper: bool) -> list[float]:
    """Every gram beyond an open bound; the shape a member certain of an out-of-range outcome declares."""
    pmf = [0.0] * (n_bins + 2)
    pmf[0 if open_lower else -1] = 1.0
    return pmf


SHAPES: dict[str, Callable[[int, bool, bool], list[float]]] = {
    "one_hot_first": _one_hot(lambda n: 0),
    "one_hot_middle": _one_hot(lambda n: n // 2),
    "one_hot_last": _one_hot(lambda n: n - 1),
    "uniform": _uniform,
    "half_half": _half_half,
    "all_in_open_tail": _all_in_open_tail,
}


def _declared(shape: str, n_bins: int, open_lower: bool, open_upper: bool) -> list[float] | None:
    """The shape's declaration on this grid, or None where the shape needs an open bound the grid lacks."""
    if shape == "all_in_open_tail" and not (open_lower or open_upper):
        return None
    return SHAPES[shape](n_bins, open_lower, open_upper)


def _heights(prediction) -> np.ndarray:
    return np.asarray([point.percentile for point in prediction.get_cdf()], dtype=float)


def _assert_legal_on_the_grid(prediction, question: NumericQuestion) -> None:
    assert_server_accepts_cdf(
        _heights(prediction),
        cdf_size=question.cdf_size,
        open_lower=question.open_lower_bound,
        open_upper=question.open_upper_bound,
    )


class TestTheServerAcceptsEveryBuild:
    """The platform's ``continuous_cdf`` validator, replicated, on the output of every shape and grid."""

    @pytest.mark.parametrize("shape", sorted(SHAPES))
    @pytest.mark.parametrize(("open_lower", "open_upper"), ORACLE_BOUND_SHAPES)
    @pytest.mark.parametrize("n_bins", ORACLE_BIN_COUNTS)
    def test_the_oracle_grids(self, n_bins: int, open_lower: bool, open_upper: bool, shape: str) -> None:
        declared = _declared(shape, n_bins, open_lower, open_upper)
        if declared is None:
            pytest.skip("needs an open bound")
        question = _count_question(n_bins, open_lower=open_lower, open_upper=open_upper)
        prediction = build_pmf_distribution(declared, question)
        _assert_legal_on_the_grid(prediction, question)

    @pytest.mark.parametrize("shape", sorted(SHAPES))
    @pytest.mark.parametrize(("n_bins", "open_lower", "open_upper"), _corpus_coarse_grids())
    def test_every_coarse_grid_mantic_has_published(
        self, n_bins: int, open_lower: bool, open_upper: bool, shape: str
    ) -> None:
        declared = _declared(shape, n_bins, open_lower, open_upper)
        if declared is None:
            pytest.skip("needs an open bound")
        question = _count_question(n_bins, open_lower=open_lower, open_upper=open_upper)
        assert elicit_per_bin(question), "the corpus filter and the gate must agree on what is coarse"
        prediction = build_pmf_distribution(declared, question)
        _assert_legal_on_the_grid(prediction, question)
        assert sum(published_pmf(prediction)) == pytest.approx(1.0, abs=1e-12)

    def test_the_corpus_has_coarse_grids_of_every_bound_shape(self) -> None:
        grids = _corpus_coarse_grids()
        assert len(grids) >= 30
        assert {(low, high) for _, low, high in grids} == {(False, False), (False, True), (True, True)}


class TestTheFloorBlend:
    """The smallest move that makes every cell legal: a zero bin lands on the floor, a certain bin keeps the rest."""

    @pytest.mark.parametrize(("open_lower", "open_upper"), ORACLE_BOUND_SHAPES)
    @pytest.mark.parametrize("n_bins", ORACLE_BIN_COUNTS)
    def test_a_certain_bin_keeps_at_least_0_988(self, n_bins: int, open_lower: bool, open_upper: bool) -> None:
        question = _count_question(n_bins, open_lower=open_lower, open_upper=open_upper)
        prediction = build_pmf_distribution(SHAPES["one_hot_middle"](n_bins, open_lower, open_upper), question)
        assert published_pmf(prediction)[1 + n_bins // 2] >= 0.988

    @pytest.mark.parametrize(("open_lower", "open_upper"), ORACLE_BOUND_SHAPES)
    @pytest.mark.parametrize("n_bins", ORACLE_BIN_COUNTS)
    def test_a_zero_bin_lands_exactly_on_the_platform_minimum_plus_the_margin(
        self, n_bins: int, open_lower: bool, open_upper: bool
    ) -> None:
        question = _count_question(n_bins, open_lower=open_lower, open_upper=open_upper)
        prediction = build_pmf_distribution(SHAPES["one_hot_middle"](n_bins, open_lower, open_upper), question)
        published = published_pmf(prediction)
        min_step, _ = grid_step_constraints(question.cdf_size)
        assert min_step == server_min_step(n_bins)
        zero_bins = [published[1 + k] for k in range(n_bins) if k != n_bins // 2]
        assert zero_bins == pytest.approx([min_step + PMF_FLOOR_MARGIN] * len(zero_bins), abs=1e-14)
        if open_lower:
            assert published[0] == pytest.approx(OPEN_TAIL_MIN_MASS + PMF_FLOOR_MARGIN, abs=1e-14)
        if open_upper:
            assert published[-1] == pytest.approx(OPEN_TAIL_MIN_MASS + PMF_FLOOR_MARGIN, abs=1e-14)

    def test_the_open_tail_minimum_is_the_one_safe_cdf_bounds_pins_to(self) -> None:
        """``safe_cdf_bounds`` still carries the platform's 0.001 as a literal; the named constant must agree with it."""
        pinned = safe_cdf_bounds(np.linspace(0.0, 1.0, 13), True, True, min_step=0.0008, max_step=1.0)
        assert pinned[0] == OPEN_TAIL_MIN_MASS
        assert pinned[-1] == 1.0 - OPEN_TAIL_MIN_MASS

    def test_a_closed_tail_receives_nothing(self) -> None:
        question = _count_question(12, open_lower=False, open_upper=False)
        published = published_pmf(build_pmf_distribution(SHAPES["one_hot_middle"](12, False, False), question))
        assert published[0] == 0.0
        assert published[-1] == 0.0

    def test_a_legal_declaration_is_returned_unchanged(self) -> None:
        """Nothing deficient, so the blend weight is zero and the declaration IS the published PMF."""
        question = _count_question(4, open_lower=False, open_upper=True)
        declared = [0.0, 0.4, 0.3, 0.2, 0.05, 0.05]
        assert published_pmf(build_pmf_distribution(declared, question)) == pytest.approx(declared, abs=1e-12)

    def test_the_blend_is_idempotent(self) -> None:
        """Feeding a published PMF back in changes nothing: every cell is already at or above its floor."""
        question = _count_question(21, open_lower=False, open_upper=True)
        first = build_pmf_distribution(SHAPES["one_hot_first"](21, False, True), question)
        second = build_pmf_distribution(published_pmf(first), question)
        assert _heights(second) == pytest.approx(_heights(first), abs=1e-12)

    def test_an_unnormalised_declaration_is_normalised_first(self) -> None:
        """The ladder tolerates a sum within 2% of 1.0; the builder divides it out before anything else."""
        question = _count_question(4, open_lower=False, open_upper=False)
        published = published_pmf(build_pmf_distribution([0.0, 0.5, 0.3, 0.2, 0.02, 0.0], question))
        assert sum(published) == pytest.approx(1.0, abs=1e-12)
        assert published[1:5] == pytest.approx([0.5, 0.3, 0.2, 0.02], rel=0.02)

    def test_the_mass_moved_is_bounded_by_the_floor_total(self) -> None:
        """The blend weight is at most the sum of the floors, about 0.012, so the declaration survives nearly intact."""
        question = _count_question(31, open_lower=True, open_upper=True)
        declared = [0.0, *([0.0] * 15), 0.6, 0.4, *([0.0] * 14), 0.0]
        published = np.asarray(published_pmf(build_pmf_distribution(declared, question)))
        assert 0.5 * np.abs(published - np.asarray(declared)).sum() <= 0.0125


class TestQuestion651:
    """The motivating case: twelve trading-window days, both bounds closed, three weekend bins that cannot resolve."""

    @pytest.fixture
    def view(self) -> EpochDateQuestion:
        return as_epoch_question(load_preseason_date_question())

    def test_a_member_certain_of_the_16th_puts_at_least_0_98_in_its_bin(self, view: EpochDateQuestion) -> None:
        grid = pmf_grid(view)
        assert grid.labels[8] == "2026-09-16"
        declared = [0.0] * 14
        declared[1 + 8] = 1.0
        prediction = build_pmf_distribution(declared, view)
        published = published_pmf(prediction)
        assert published[1 + 8] >= 0.98
        _assert_legal_on_the_grid(prediction, view)

    def test_the_weekend_bins_carry_exactly_the_floor(self, view: EpochDateQuestion) -> None:
        grid = pmf_grid(view)
        weekend = [grid.labels.index(day) for day in ("2026-09-12", "2026-09-13", "2026-09-19")]
        declared = [0.0] * 14
        for trading_day in (0, 1, 2, 3, 6, 7, 8, 9, 10):
            declared[1 + trading_day] = 1.0 / 9
        published = published_pmf(build_pmf_distribution(declared, view))
        min_step, _ = grid_step_constraints(view.cdf_size)
        for index in weekend:
            assert published[1 + index] == pytest.approx(min_step + PMF_FLOOR_MARGIN, abs=1e-14)

    def test_the_wrap_carries_the_date_grid(self, view: EpochDateQuestion) -> None:
        declared = [0.0, *([1.0 / 12] * 12), 0.0]
        prediction = build_pmf_distribution(declared, view)
        assert prediction.is_date is True
        assert prediction.cdf_size == 13
        assert prediction.zero_point is None
        edges = build_cdf_value_grid(view.lower_bound, view.upper_bound, None, view.cdf_size)
        assert [point.value for point in prediction.declared_percentiles] == pytest.approx(list(edges))
        assert [point.percentile for point in prediction.declared_percentiles] == pytest.approx(
            list(_heights(prediction))
        )
        assert prediction.declared_percentiles[0].percentile == 0.0
        assert prediction.declared_percentiles[-1].percentile == 1.0


class TestTheWrap:
    def test_a_quantity_question_is_not_a_date(self) -> None:
        question = _count_question(21, open_lower=False, open_upper=True)
        prediction = build_pmf_distribution(SHAPES["uniform"](21, False, True), question)
        assert prediction.is_date is False
        assert prediction.cdf_size == 22
        assert len(prediction.declared_percentiles) == 22
        assert prediction.open_upper_bound is True
        assert prediction.open_lower_bound is False

    def test_a_zero_point_grid_keeps_its_geometric_axis(self) -> None:
        question = _count_question(10, open_lower=False, open_upper=False, zero_point=0.0)
        prediction = build_pmf_distribution(SHAPES["uniform"](10, False, False), question)
        assert prediction.zero_point == 0.0
        values = [point.value for point in prediction.declared_percentiles]
        assert values == pytest.approx(list(build_cdf_value_grid(1.0, 1000.0, 0.0, 11)))
        assert values[1] != pytest.approx(1.0 + 999.0 / 10)

    def test_published_pmf_is_the_platforms_n_plus_2_shape(self) -> None:
        question = _count_question(12, open_lower=True, open_upper=True)
        prediction = build_pmf_distribution(SHAPES["half_half"](12, True, True), question)
        published = published_pmf(prediction)
        heights = _heights(prediction)
        assert len(published) == 14
        assert published[0] == heights[0]
        assert published[-1] == pytest.approx(1.0 - heights[-1])
        assert published[1:-1] == pytest.approx(list(np.diff(heights)))
        assert sum(published) == pytest.approx(1.0, abs=1e-12)

    def test_get_cdf_and_declared_percentiles_agree(self) -> None:
        question = _count_question(4, open_lower=False, open_upper=True)
        prediction = build_pmf_distribution(SHAPES["one_hot_last"](4, False, True), question)
        assert [p.percentile for p in prediction.declared_percentiles] == pytest.approx(list(_heights(prediction)))


class TestTheBuilderRefuses:
    """The public module boundary fails shut on a declaration the ladder should never have produced."""

    def test_mass_in_a_closed_lower_tail(self) -> None:
        """The review reproduced 0.297 of a closed lower tail landing in bin 1 when the pin came after the sum."""
        question = _count_question(12, open_lower=False, open_upper=False)
        declared = [0.3, *([0.7 / 12] * 12), 0.0]
        with pytest.raises(ValueError, match="closed lower bound"):
            build_pmf_distribution(declared, question)

    def test_mass_in_a_closed_upper_tail(self) -> None:
        question = _count_question(12, open_lower=True, open_upper=False)
        declared = [0.1, *([0.6 / 12] * 12), 0.3]
        with pytest.raises(ValueError, match="closed upper bound"):
            build_pmf_distribution(declared, question)

    def test_mass_in_an_open_tail_is_the_forecast(self) -> None:
        question = _count_question(12, open_lower=False, open_upper=True)
        declared = [0.0, *([0.7 / 12] * 12), 0.3]
        published = published_pmf(build_pmf_distribution(declared, question))
        assert published[-1] == pytest.approx(0.3, abs=1e-3)

    def test_a_vector_of_the_wrong_length(self) -> None:
        question = _count_question(12, open_lower=False, open_upper=False)
        with pytest.raises(ValueError, match="14"):
            build_pmf_distribution([0.0, *([1.0 / 12] * 12)], question)

    @pytest.mark.parametrize("bad", [float("nan"), -0.1, float("inf")])
    def test_a_non_finite_or_negative_mass(self, bad: float) -> None:
        question = _count_question(4, open_lower=False, open_upper=True)
        with pytest.raises(ValueError, match=r"finite|negative"):
            build_pmf_distribution([0.0, 0.5, bad, 0.3, 0.1, 0.1], question)

    def test_an_all_zero_vector(self) -> None:
        question = _count_question(4, open_lower=False, open_upper=True)
        with pytest.raises(ValueError, match="sum"):
            build_pmf_distribution([0.0] * 6, question)


class TestValidateGridCdf:
    """The production twin of ``assert_server_accepts_cdf``: every rule the server enforces raises here."""

    @staticmethod
    def _legal(n_bins: int, *, open_lower: bool, open_upper: bool) -> np.ndarray:
        question = _count_question(n_bins, open_lower=open_lower, open_upper=open_upper)
        return _heights(build_pmf_distribution(SHAPES["uniform"](n_bins, open_lower, open_upper), question))

    def test_a_legal_cdf_passes(self) -> None:
        cdf = self._legal(12, open_lower=False, open_upper=True)
        validate_grid_cdf(cdf, cdf_size=13, open_lower=False, open_upper=True)

    def test_the_wrong_length_raises(self) -> None:
        cdf = self._legal(12, open_lower=False, open_upper=False)
        with pytest.raises(RuntimeError, match="length"):
            validate_grid_cdf(cdf[:-1], cdf_size=13, open_lower=False, open_upper=False)

    def test_a_step_under_the_floor_raises(self) -> None:
        cdf = self._legal(12, open_lower=False, open_upper=False)
        cdf[6] = cdf[5]
        with pytest.raises(RuntimeError, match="min step"):
            validate_grid_cdf(cdf, cdf_size=13, open_lower=False, open_upper=False)

    def test_a_closed_bound_not_pinned_raises(self) -> None:
        cdf = self._legal(12, open_lower=False, open_upper=False)
        cdf[0] = 1e-6
        with pytest.raises(RuntimeError, match="closed lower bound"):
            validate_grid_cdf(cdf, cdf_size=13, open_lower=False, open_upper=False)
        cdf = self._legal(12, open_lower=False, open_upper=False)
        cdf[-1] = 0.9999999
        with pytest.raises(RuntimeError, match="closed upper bound"):
            validate_grid_cdf(cdf, cdf_size=13, open_lower=False, open_upper=False)

    def test_an_open_bound_inside_the_pin_raises(self) -> None:
        cdf = self._legal(12, open_lower=True, open_upper=True)
        cdf[0] = 0.0005
        with pytest.raises(RuntimeError, match="open lower bound"):
            validate_grid_cdf(cdf, cdf_size=13, open_lower=True, open_upper=True)
        cdf = self._legal(12, open_lower=True, open_upper=True)
        cdf[-1] = 0.9995
        with pytest.raises(RuntimeError, match="open upper bound"):
            validate_grid_cdf(cdf, cdf_size=13, open_lower=True, open_upper=True)

    def test_a_max_step_breach_raises(self) -> None:
        """The cap binds only above 40 bins (``0.2 * 200 / N < 1``): 50 bins cap each step at 0.8."""
        n_bins = 50
        _, max_step = grid_step_constraints(n_bins + 1)
        assert max_step == 0.8
        min_step, _ = grid_step_constraints(n_bins + 1)
        steps = np.full(n_bins, min_step)
        steps[25] = 1.0 - min_step * (n_bins - 1)
        cdf = np.concatenate([[0.0], np.cumsum(steps)])
        cdf[-1] = 1.0
        with pytest.raises(RuntimeError, match="max step"):
            validate_grid_cdf(cdf, cdf_size=n_bins + 1, open_lower=False, open_upper=False)

    def test_a_nan_raises(self) -> None:
        cdf = self._legal(12, open_lower=False, open_upper=False)
        cdf[3] = float("nan")
        with pytest.raises(RuntimeError, match="NaN"):
            validate_grid_cdf(cdf, cdf_size=13, open_lower=False, open_upper=False)

    def test_the_server_rounding_is_replicated_not_tightened(self) -> None:
        """A step one part in 1e12 under the floor rounds onto it at 9 decimals, so the server accepts it and so do we."""
        cdf = self._legal(12, open_lower=False, open_upper=False)
        min_step, _ = grid_step_constraints(13)
        steps = np.diff(cdf)
        steps[6] = min_step - 1e-12
        steps[7] += 1e-12
        cdf = np.concatenate([[0.0], np.cumsum(steps)])
        cdf[-1] = 1.0
        validate_grid_cdf(cdf, cdf_size=13, open_lower=False, open_upper=False)


class TestTheLinearPoolOfPerBinMembers:
    """The mean of three sharp members' CDFs is the mixture PMF, and the server accepts it on every oracle grid."""

    @pytest.mark.parametrize(("open_lower", "open_upper"), ORACLE_BOUND_SHAPES)
    @pytest.mark.parametrize("n_bins", ORACLE_BIN_COUNTS)
    def test_three_disagreeing_members_pool_to_a_third_each(
        self, n_bins: int, open_lower: bool, open_upper: bool
    ) -> None:
        question = _count_question(n_bins, open_lower=open_lower, open_upper=open_upper)
        believed = sorted({0, n_bins // 2, n_bins - 1})
        members = [
            build_pmf_distribution(_one_hot(lambda n, k=k: k)(n_bins, open_lower, open_upper), question)
            for k in believed
        ]
        pooled = aggregate_numeric(members, question, "mean")
        _assert_legal_on_the_grid(pooled, question)
        published = published_pmf(pooled)
        for k in believed:
            assert 1.0 / len(believed) - 0.04 <= published[1 + k] <= 1.0 / len(believed) + 0.04
