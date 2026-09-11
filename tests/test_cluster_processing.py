"""
Unit tests for cluster processing utilities.

Tests for cluster detection and spreading functions extracted from main.py.
"""

from itertools import pairwise
from types import SimpleNamespace
from typing import ClassVar, cast

import pytest
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.constants import NUM_SPREAD_DELTA_MULT
from metaculus_bot.numeric.cluster_processing import (
    apply_cluster_spreading,
    apply_jitter_for_duplicates,
    compute_cluster_parameters,
    detect_count_like_pattern,
    ensure_strictly_increasing_bounded,
    is_degenerate_cluster,
)
from metaculus_bot.numeric.config import (
    EXPECTED_PERCENTILE_COUNT,
    STANDARD_PERCENTILES,
    grid_bin_width,
    minimum_separation,
)
from metaculus_bot.numeric.validation import detect_unit_mismatch


def _make_question(open_upper=False, open_lower=False, lower=0.0, upper=100.0) -> NumericQuestion:
    return cast(
        NumericQuestion,
        SimpleNamespace(
            open_upper_bound=open_upper,
            open_lower_bound=open_lower,
            upper_bound=upper,
            lower_bound=lower,
            cdf_size=201,
            id_of_question=999,
        ),
    )


class TestClusterProcessing:
    """Test cluster detection and processing functions."""

    def test_detect_count_like_pattern_true(self):
        """Test detection of count-like patterns."""
        values = [10.0, 11.0, 12.0, 13.0]  # All integers
        assert detect_count_like_pattern(values) is True

    def test_detect_count_like_pattern_false(self):
        """Test non-count-like patterns."""
        values = [10.5, 11.7, 12.3, 13.8]  # Not near integers
        assert detect_count_like_pattern(values) is False

    def test_detect_count_like_pattern_near_integers(self):
        """Test values very close to integers."""
        values = [10.000001, 11.0, 12.000002, 13.0]  # Close to integers
        assert detect_count_like_pattern(values) is True

    def test_detect_count_like_pattern_empty(self):
        """Test empty values list."""
        assert detect_count_like_pattern([]) is False

    def test_detect_count_like_pattern_exception_handling(self):
        """Test exception handling in count-like detection."""
        # This should handle any unexpected values gracefully
        values = [float("nan"), 10.0]
        assert detect_count_like_pattern(values) is False

    def test_compute_cluster_parameters_normal(self):
        """Test cluster parameter computation for normal case."""
        range_size = 100.0
        count_like = False

        value_eps, base_delta, spread_delta = compute_cluster_parameters(range_size, count_like)

        assert value_eps > 0
        assert base_delta > 0
        assert spread_delta > 0
        # For non-count-like, spread_delta should equal base_delta
        assert spread_delta == base_delta

    def test_compute_cluster_parameters_count_like(self):
        """Test cluster parameter computation for count-like case."""
        range_size = 100.0
        count_like = True

        value_eps, base_delta, spread_delta = compute_cluster_parameters(range_size, count_like)

        assert value_eps > 0
        assert base_delta > 0
        assert spread_delta > 0
        # For count-like, spread_delta should be at least 1.0
        assert spread_delta >= 1.0

    def test_apply_cluster_spreading_no_clusters(self):
        """Test cluster spreading when no clusters exist."""
        values = [10.0, 20.0, 30.0, 40.0]  # Well-separated values
        question = _make_question()

        result, clusters_applied = apply_cluster_spreading(
            values.copy(), question, value_eps=1e-9, spread_delta=1e-6, range_size=100.0
        )

        assert clusters_applied == 0
        assert result == values  # Should be unchanged

    def test_apply_cluster_spreading_with_cluster(self):
        """Test cluster spreading with actual clusters."""
        values = [10.0, 20.0, 20.0, 20.0, 30.0]  # Cluster in the middle
        question = _make_question()

        result, clusters_applied = apply_cluster_spreading(
            values.copy(), question, value_eps=1e-6, spread_delta=0.01, range_size=100.0
        )

        assert clusters_applied == 1
        cluster_values = result[1:4]
        assert len(set(cluster_values)) == 3
        assert all(a < b for a, b in pairwise(cluster_values))

    def test_apply_cluster_spreading_boundary_constraints(self):
        """Test cluster spreading respects boundary constraints.

        The cluster sits at the closed lower bound but is PARTIAL (an unclustered
        neighbour follows), which is the shape the spreader exists for. It used to
        use a whole-set collapse, which no longer spreads at all — see
        ``test_whole_set_collapse_is_not_spread``.
        """
        values = [0.0, 0.0, 0.0, 50.0]  # Cluster at lower bound, plus a real neighbour
        question = _make_question(open_lower=False, lower=0.0, upper=100.0)

        result, clusters_applied = apply_cluster_spreading(
            values.copy(), question, value_eps=1e-6, spread_delta=1.0, range_size=100.0
        )

        assert clusters_applied == 1
        # All values should be above the lower bound
        assert all(v > question.lower_bound for v in result)

    def test_whole_set_collapse_is_not_spread(self):
        """A point mass gets NO fabricated width (H2, 2026-08-25).

        Every value inside one epsilon cluster means the model declared no
        distribution at all. Spreading it invented the width — on a count-like
        question, a full unit per position — and that invented span was also what
        let the degenerate declaration pass ``detect_unit_mismatch``. The spreader
        now leaves it alone and reports 0 clusters; the minimum separation a CDF
        needs comes from the jitter / strict-ordering passes instead.
        """
        values = [42.0] * 13
        question = _make_question(lower=0.0, upper=100.0)

        result, clusters_applied = apply_cluster_spreading(
            values.copy(), question, value_eps=1e-7, spread_delta=1.0, range_size=100.0
        )

        assert clusters_applied == 0
        assert result == values

    def test_near_equal_whole_set_collapse_is_not_spread(self):
        """Epsilon-chained (not exactly equal) whole-set collapse counts as a point mass."""
        values = [42.0 + i * 5e-8 for i in range(13)]
        question = _make_question(lower=0.0, upper=100.0)

        result, clusters_applied = apply_cluster_spreading(
            values.copy(), question, value_eps=1e-7, spread_delta=1.0, range_size=100.0
        )

        assert clusters_applied == 0
        assert result == values


class TestIsDegenerateCluster:
    """``is_degenerate_cluster`` — the point-mass predicate shared by the spreader
    and the pipeline's ``NUMERIC_DEGENERATE_DECLARATION`` marker."""

    def test_identical_values(self):
        assert is_degenerate_cluster([7.0] * 13, 1e-7) is True

    def test_epsilon_chained_values(self):
        assert is_degenerate_cluster([7.0 + i * 9e-8 for i in range(5)], 1e-7) is True

    def test_one_real_gap_is_not_degenerate(self):
        assert is_degenerate_cluster([7.0, 7.0, 7.0, 9.0], 1e-7) is False

    def test_single_value_is_not_degenerate(self):
        """Nothing to compare, and a 1-element set never reaches the spreader."""
        assert is_degenerate_cluster([7.0], 1e-7) is False
        assert is_degenerate_cluster([], 1e-7) is False

    def test_apply_jitter_for_duplicates(self):
        """Test jitter application for duplicate values."""
        values = [10.0, 10.0, 30.0]  # Duplicate at start
        percentiles = [
            Percentile(percentile=0.10, value=10.0),
            Percentile(percentile=0.20, value=10.0),
            Percentile(percentile=0.30, value=30.0),
        ]
        question = _make_question()

        result = apply_jitter_for_duplicates(values.copy(), question, 100.0, percentiles)

        # Should be strictly increasing
        assert all(a < b for a, b in pairwise(result))

    def test_apply_jitter_for_duplicates_boundary_respect(self):
        """Test jitter respects boundaries."""
        values = [99.0, 99.0]  # Near upper bound
        percentiles = [
            Percentile(percentile=0.90, value=99.0),
            Percentile(percentile=0.95, value=99.0),
        ]
        question = _make_question(open_upper=False, upper=100.0)

        result = apply_jitter_for_duplicates(values.copy(), question, 100.0, percentiles)

        # Should be strictly increasing and within bounds
        assert all(a < b for a, b in pairwise(result))
        assert all(v <= question.upper_bound for v in result)

    def test_ensure_strictly_increasing_bounded_left_to_right(self):
        """Test left-to-right strictly increasing enforcement."""
        values = [10.0, 9.0, 30.0]  # Second value is smaller
        question = _make_question()

        result = ensure_strictly_increasing_bounded(values.copy(), question, 100.0)

        # Should be strictly increasing
        assert all(a < b for a, b in pairwise(result))

    def test_ensure_strictly_increasing_bounded_right_to_left(self):
        """Test right-to-left adjustment for boundary cases."""
        values = [98.0, 99.0, 99.0]  # Cluster near upper bound
        question = _make_question(open_upper=False, upper=100.0)

        result = ensure_strictly_increasing_bounded(values.copy(), question, 100.0)

        # Should be strictly increasing and within bounds
        assert all(a < b for a, b in pairwise(result))
        assert all(v <= question.upper_bound for v in result)

    def test_ensure_strictly_increasing_bounded_respects_boundaries(self):
        """Test that boundary enforcement respects open/closed bounds."""
        values = [1.0, 1.0]  # Duplicates near lower bound
        question = _make_question(open_lower=False, lower=0.0, upper=100.0)

        result = ensure_strictly_increasing_bounded(values.copy(), question, 100.0)

        # Should be strictly increasing and above lower bound
        assert all(a < b for a, b in pairwise(result))
        assert all(v >= question.lower_bound for v in result)


class TestApplyClusterSpreadingGoldenOutputs:
    """Exact-output pins over the spreader's whole shape space.

    ``apply_cluster_spreading`` walks the value list with a hand-rolled index cursor,
    grows each epsilon cluster, spreads it symmetrically around its mean, then applies a
    shift-up against the previous neighbour and a compression against the next one. The
    property tests above cover the individual rules; these goldens pin the composition
    (multiple clusters in one pass, a cluster at each bound, a chained near-equal run)
    so the walk can be restructured without moving a published value. Captured from the
    implementation as of 2026-08-26 as exact reprs, and compared bit for bit.
    """

    SHAPES: ClassVar[dict[str, list[float]]] = {
        "mid_cluster": [10.0, 20.0, 20.0, 20.0, 30.0],
        "two_clusters": [1.0, 1.0, 5.0, 9.0, 9.0, 9.0, 9.0, 40.0],
        "at_lower": [0.0, 0.0, 0.0, 50.0],
        "trailing": [1.0, 4.0, 10.0, 10.0],
        "tight_gap": [10.0, 20.0, 20.0, 20.000001, 20.0001, 60.0],
        "near_upper": [10.0, 99.9, 99.9, 99.9, 99.95],
    }

    # (shape, open_lower, open_upper) -> (expected values, expected clusters_applied)
    GOLDEN: ClassVar[dict[tuple[str, bool, bool], tuple[list[float], int]]] = {
        ("mid_cluster", False, False): ([10.0, 19.0, 20.0, 21.0, 30.0], 1),
        ("mid_cluster", True, True): ([10.0, 19.0, 20.0, 21.0, 30.0], 1),
        ("two_clusters", False, False): ([0.5, 1.5, 5.0, 7.5, 8.5, 9.5, 10.5, 40.0], 2),
        ("two_clusters", True, True): ([0.5, 1.5, 5.0, 7.5, 8.5, 9.5, 10.5, 40.0], 2),
        ("at_lower", False, False): ([1.0000000000000001e-07, 1.0000000000000001e-07, 1.0, 50.0], 1),
        ("at_lower", True, True): ([-1.0, 0.0, 1.0, 50.0], 1),
        ("trailing", False, False): ([1.0, 4.0, 9.5, 10.5], 1),
        ("trailing", True, True): ([1.0, 4.0, 9.5, 10.5], 1),
        ("tight_gap", False, False): ([10.0, 19.5, 19.7500005, 20.000001, 20.0001, 60.0], 1),
        ("tight_gap", True, True): ([10.0, 19.5, 19.7500005, 20.000001, 20.0001, 60.0], 1),
        ("near_upper", False, False): (
            [10.0, 98.90000000000002, 99.25000000000001, 99.60000000000001, 99.95],
            1,
        ),
        ("near_upper", True, True): (
            [10.0, 98.90000000000002, 99.25000000000001, 99.60000000000001, 99.95],
            1,
        ),
    }

    @pytest.mark.parametrize(("open_lower", "open_upper"), [(False, False), (True, True)])
    @pytest.mark.parametrize("shape_name", list(SHAPES))
    def test_golden_output(self, shape_name: str, open_lower: bool, open_upper: bool):
        values = list(self.SHAPES[shape_name])
        question = _make_question(open_lower=open_lower, open_upper=open_upper)

        result, clusters_applied = apply_cluster_spreading(
            values,
            question,
            value_eps=1e-6,
            spread_delta=1.0,
            range_size=100.0,
        )

        expected_values, expected_clusters = self.GOLDEN[shape_name, open_lower, open_upper]
        assert clusters_applied == expected_clusters
        assert result == expected_values

    def test_spreading_mutates_the_caller_list_in_place(self):
        """The spreader returns the SAME list object it was handed; the pipeline relies
        on the return value, but a caller reusing its own list must see the mutation."""
        values = [10.0, 20.0, 20.0, 20.0, 30.0]
        question = _make_question()

        result, _ = apply_cluster_spreading(values, question, value_eps=1e-6, spread_delta=1.0, range_size=100.0)

        assert result is values


class TestDiscreteGridPlateauCap:
    """On a discrete grid a plateau's whole spread stays inside the bin it names.

    The bin width is ``(upper - lower) / (cdf_size - 1)`` and the plateau is centred on the
    declared value, so its values span at most ``value +- width / 2``, which on an
    integer-centred grid is exactly the bin. The 201-point continuous grid is exempt: the
    golden pins above run at ``cdf_size=201`` and are unchanged.
    """

    def _discrete_question(self, cdf_size: int, lower: float, upper: float) -> NumericQuestion:
        return cast(
            NumericQuestion,
            SimpleNamespace(
                open_upper_bound=False,
                open_lower_bound=False,
                upper_bound=upper,
                lower_bound=lower,
                cdf_size=cdf_size,
                id_of_question=253,
            ),
        )

    def test_three_bin_plateau_spans_at_most_one_bin(self):
        """Mantic post 253: bins centred on 0, 1, 2; P1..P90 = 0, P95 = P97.5 = 1, P99 = 2."""
        values = [0.0] * 10 + [1.0, 1.0, 2.0]
        question = self._discrete_question(4, -0.5, 2.5)

        result, clusters_applied = apply_cluster_spreading(
            values, question, value_eps=1e-9, spread_delta=1.0, range_size=3.0
        )

        assert clusters_applied == 2
        assert all(-0.5 <= v <= 0.5 for v in result[:10]), result[:10]
        assert all(0.5 <= v <= 1.5 + 1e-9 for v in result[10:12]), result[10:12]
        assert all(a < b for a, b in pairwise(result[:10]))

    def test_per_position_spread_is_the_bin_width_over_the_plateau_length(self):
        """The cap only tightens a plateau whose full spread would exceed its bin: uncapped the
        three-value plateau is [19, 20, 21] (the "mid_cluster" golden), a 2.0 spread, which fits a
        2.5-wide bin unchanged and is halved by a 1.0-wide one."""
        values = [10.0, 20.0, 20.0, 20.0, 30.0]
        question = self._discrete_question(41, 0.0, 100.0)  # bin width 2.5

        result, _ = apply_cluster_spreading(values, question, value_eps=1e-6, spread_delta=1.0, range_size=100.0)

        assert result == pytest.approx([10.0, 19.0, 20.0, 21.0, 30.0])

        question = self._discrete_question(201, 0.0, 100.0)  # bin width 0.5 but the 201 grid is exempt
        result, _ = apply_cluster_spreading(
            [10.0, 20.0, 20.0, 20.0, 30.0], question, value_eps=1e-6, spread_delta=1.0, range_size=100.0
        )
        assert result == pytest.approx([10.0, 19.0, 20.0, 21.0, 30.0])

        question = self._discrete_question(101, 0.0, 100.0)  # bin width 1.0 caps the 2.0 spread
        result, _ = apply_cluster_spreading(
            [10.0, 20.0, 20.0, 20.0, 30.0], question, value_eps=1e-6, spread_delta=1.0, range_size=100.0
        )
        assert result == pytest.approx([10.0, 19.5, 20.0, 20.5, 30.0])

    def test_a_whole_set_collapse_is_spread_inside_its_bin_where_the_bins_are_the_outcome_space(self):
        """Thirteen identical values on twelve one-day bins: "100% on this day" is fully
        expressible there, so the set is spread like any other plateau, under the one-bin cap,
        and reaches the unit-mismatch guard with a real span instead of being withheld. The
        201-grid contrast is ``test_whole_set_collapse_is_not_spread``: there the collapse is
        left alone and the guard withholds it."""
        day = 86_400.0
        noon_of_day_8 = 8 * day + day / 2
        values = [noon_of_day_8] * 13
        range_size = 12 * day
        value_eps, _base_delta, spread_delta = compute_cluster_parameters(
            range_size, detect_count_like_pattern(values), span=0.0
        )
        question = self._discrete_question(13, 0.0, range_size)

        result, clusters_applied = apply_cluster_spreading(
            values, question, value_eps=value_eps, spread_delta=spread_delta, range_size=range_size
        )

        assert clusters_applied == 1
        assert all(a < b for a, b in pairwise(result))
        assert 8 * day < result[0]
        assert result[-1] < 9 * day
        assert result[-1] - result[0] == pytest.approx(12 * spread_delta)
        assert result[-1] - result[0] <= day


class TestCollapseOnABoundOfAnOutcomeSpaceGrid:
    """A whole-set collapse ON a bound is translated into the range, whichever kind of bound.

    The symmetric spread was bounded only at CLOSED edges. On an OPEN edge half the values
    crossed outside the range: post 651's grid with 13 declarations at the open lower bound
    spread to [-6.2 s, +6.2 s] around it, and the build published ``cdf[0] == 0.5``, a coin
    flip on "before the window" invented from a declaration that named nothing before it. On
    a CLOSED edge the clamp folded the spread to half its width, a span ratio of 6e-6 that
    ``detect_unit_mismatch`` withheld at its 1e-5 threshold: the very drop the outcome-space
    carve-out exists to remove (codex second-opinion review, 2026-09). The bound value
    itself buckets into the terminal bin, so the faithful reading is "all mass in that bin":
    the plateau is shifted, not clipped, until it lies within the range, keeping its full
    12-step span under the one-bin cap. Interior collapses and collapses at a bin CENTRE
    were fine at every grid size and are byte-identical.
    """

    GRID_SIZES: ClassVar[tuple[int, ...]] = (3, 13, 22, 451, 2001)
    RANGE_SIZE: ClassVar[float] = 12 * 86_400.0  # post 651's twelve days, in epoch seconds

    def _question(self, cdf_size: int, *, open_lower: bool, open_upper: bool) -> NumericQuestion:
        return cast(
            NumericQuestion,
            SimpleNamespace(
                open_upper_bound=open_upper,
                open_lower_bound=open_lower,
                upper_bound=self.RANGE_SIZE,
                lower_bound=0.0,
                cdf_size=cdf_size,
                id_of_question=651,
            ),
        )

    @pytest.mark.parametrize("cdf_size", GRID_SIZES)
    @pytest.mark.parametrize("bound_kind", ["closed", "open"])
    @pytest.mark.parametrize("edge", ["lower", "upper"])
    def test_the_plateau_keeps_its_full_span_inside_the_terminal_bin(
        self, edge: str, bound_kind: str, cdf_size: int
    ) -> None:
        is_open = bound_kind == "open"
        question = self._question(
            cdf_size, open_lower=is_open and edge == "lower", open_upper=is_open and edge == "upper"
        )
        bound = question.lower_bound if edge == "lower" else question.upper_bound
        values = [bound] * EXPECTED_PERCENTILE_COUNT
        value_eps, _base_delta, spread_delta = compute_cluster_parameters(
            self.RANGE_SIZE, detect_count_like_pattern(values), span=0.0
        )
        bin_width = grid_bin_width(question.lower_bound, question.upper_bound, cdf_size)

        result, clusters_applied = apply_cluster_spreading(
            values, question, value_eps=value_eps, spread_delta=spread_delta, range_size=self.RANGE_SIZE
        )

        assert clusters_applied == 1
        assert all(a < b for a, b in pairwise(result))
        assert question.lower_bound <= result[0]
        assert result[-1] <= question.upper_bound
        if edge == "lower":
            assert result[-1] <= question.lower_bound + bin_width
        else:
            assert question.upper_bound - bin_width <= result[0]
        # Shifted, not clipped: the span is the one an interior collapse gets under the one-bin cap.
        per_position = min(spread_delta, bin_width / (EXPECTED_PERCENTILE_COUNT - 1))
        assert result[-1] - result[0] == pytest.approx((EXPECTED_PERCENTILE_COUNT - 1) * per_position)
        # 12 * NUM_SPREAD_DELTA_MULT = 1.2e-5 of the range, a 20% margin over the guard's 1e-5.
        span_ratio = (result[-1] - result[0]) / self.RANGE_SIZE
        assert span_ratio == pytest.approx((EXPECTED_PERCENTILE_COUNT - 1) * NUM_SPREAD_DELTA_MULT)
        percentiles = [Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, result, strict=True)]
        mismatch, reason = detect_unit_mismatch(percentiles, question)
        assert not mismatch, reason

    @pytest.mark.parametrize("bound_kind", ["closed", "open"])
    def test_the_plateau_starts_exactly_on_the_bound_it_was_declared_on(self, bound_kind: str) -> None:
        """Shifted to the bound, not to a standoff inside it: a declared P1 = lower already lands
        there and builds the same CDF (closed: cdf[0] = 0; open: the structural 0.01), and
        keeping the plateau where it is unless it crosses a bound is what leaves a bin-centre
        plateau whose cap binds (post 253) byte-identical."""
        question = self._question(13, open_lower=bound_kind == "open", open_upper=False)
        values = [question.lower_bound] * EXPECTED_PERCENTILE_COUNT
        value_eps, _base_delta, spread_delta = compute_cluster_parameters(
            self.RANGE_SIZE, detect_count_like_pattern(values), span=0.0
        )

        result, _ = apply_cluster_spreading(
            values, question, value_eps=value_eps, spread_delta=spread_delta, range_size=self.RANGE_SIZE
        )

        assert result[0] == question.lower_bound
        assert result[1] == pytest.approx(question.lower_bound + spread_delta)


class TestWholeSetCollapseDecidesOnTheDeclaredValue:
    """The whole-set collapse is placed by its DECLARED value, the median element of the sorted
    set, compared exactly against the bounds.

    ``np.mean([1.3] * 13)`` is ``1.3000000000000003``, which read a collapse exactly ON an open
    upper bound of 1.3 as beyond it, skipped the translation and published 7 of 13 values past
    the bound (codex re-check 3, 2026-09, on grids 3 to 2001); ``values[0]`` placed an
    epsilon-CHAIN of near-equal values off its centre by the chain's width (re-check 4). On the
    bound or inside, the full span goes inside the range; strictly beyond an OPEN bound the
    spread stays symmetric about the declared value; strictly beyond a CLOSED bound the plateau
    starts at it, so the clamp that runs after the spreader judges the declared distance. A
    whole-set collapse has no neighbours, so the shift-up / compress repairs never re-space it.
    """

    GRID_SIZES: ClassVar[tuple[int, ...]] = (3, 13, 22, 451, 2001)
    RANGE: ClassVar[tuple[float, float]] = (0.0, 12.0)

    def _full_span(self, cdf_size: int, spread_delta: float = 1.0) -> float:
        """The whole-set collapse's span on ``RANGE``: 12 steps of the capped per-position spread."""
        lower, upper = self.RANGE
        return (EXPECTED_PERCENTILE_COUNT - 1) * min(
            spread_delta, grid_bin_width(lower, upper, cdf_size) / (EXPECTED_PERCENTILE_COUNT - 1)
        )

    def _spread(self, values: list[float], question: NumericQuestion, *, spread_delta: float = 1.0) -> list[float]:
        result, clusters_applied = apply_cluster_spreading(
            values,
            question,
            value_eps=1e-6,
            spread_delta=spread_delta,
            range_size=question.upper_bound - question.lower_bound,
        )
        assert clusters_applied == 1
        return result

    def _question(
        self, lower: float, upper: float, cdf_size: int, *, open_lower: bool, open_upper: bool
    ) -> NumericQuestion:
        return cast(
            NumericQuestion,
            SimpleNamespace(
                open_upper_bound=open_upper,
                open_lower_bound=open_lower,
                upper_bound=upper,
                lower_bound=lower,
                cdf_size=cdf_size,
                id_of_question=1,
            ),
        )

    def _collapse(
        self, declared: float, question: NumericQuestion, *, value_eps: float, spread_delta: float
    ) -> list[float]:
        values = [declared] * EXPECTED_PERCENTILE_COUNT
        result, clusters_applied = apply_cluster_spreading(
            values,
            question,
            value_eps=value_eps,
            spread_delta=spread_delta,
            range_size=question.upper_bound - question.lower_bound,
        )
        assert clusters_applied == 1
        return result

    @pytest.mark.parametrize("cdf_size", GRID_SIZES)
    @pytest.mark.parametrize("bound_kind", ["closed", "open"])
    @pytest.mark.parametrize(("edge", "lower", "upper"), [("lower", 1.1, 2.3), ("upper", 0.1, 1.3)])
    def test_exactly_on_a_float_fragile_bound_stays_inside(
        self, edge: str, lower: float, upper: float, bound_kind: str, cdf_size: int
    ) -> None:
        is_open = bound_kind == "open"
        question = self._question(
            lower, upper, cdf_size, open_lower=is_open and edge == "lower", open_upper=is_open and edge == "upper"
        )
        declared = lower if edge == "lower" else upper
        value_eps, _base_delta, spread_delta = compute_cluster_parameters(upper - lower, count_like=False, span=0.0)
        per_position = min(spread_delta, grid_bin_width(lower, upper, cdf_size) / (EXPECTED_PERCENTILE_COUNT - 1))

        result = self._collapse(declared, question, value_eps=value_eps, spread_delta=spread_delta)

        assert all(lower <= v <= upper for v in result)
        assert all(a < b for a, b in pairwise(result))
        assert (result[0] if edge == "lower" else result[-1]) == declared
        assert result[-1] - result[0] == pytest.approx((EXPECTED_PERCENTILE_COUNT - 1) * per_position)

    @pytest.mark.parametrize("cdf_size", GRID_SIZES)
    @pytest.mark.parametrize("declared", [-0.25, 12.25], ids=["below_open_lower", "above_open_upper"])
    def test_strictly_beyond_an_open_bound_spreads_symmetrically_about_the_declared_value(
        self, declared: float, cdf_size: int
    ) -> None:
        question = self._question(*self.RANGE, cdf_size, open_lower=True, open_upper=True)
        half_span = self._full_span(cdf_size) / 2

        result = self._collapse(declared, question, value_eps=1e-6, spread_delta=1.0)

        assert sum(result) / len(result) == pytest.approx(declared)
        assert (result[0], result[-1]) == pytest.approx((declared - half_span, declared + half_span))

    @pytest.mark.parametrize("cdf_size", GRID_SIZES)
    @pytest.mark.parametrize("declared", [-0.25, 12.25], ids=["below_closed_lower", "above_closed_upper"])
    def test_strictly_beyond_a_closed_bound_starts_at_the_declared_value(self, declared: float, cdf_size: int) -> None:
        question = self._question(*self.RANGE, cdf_size, open_lower=False, open_upper=False)
        span = self._full_span(cdf_size)
        below = declared < question.lower_bound

        result = self._collapse(declared, question, value_eps=1e-6, spread_delta=1.0)

        expected_edges = (declared, declared + span) if below else (declared - span, declared)
        assert (result[0], result[-1]) == pytest.approx(expected_edges)
        assert (result[0] if below else result[-1]) == declared

    @pytest.mark.parametrize("cdf_size", GRID_SIZES)
    @pytest.mark.parametrize("bound_kind", ["closed", "open"])
    @pytest.mark.parametrize("bins_inside", [0.5, 1.0], ids=["half_a_bin_inside", "one_bin_inside"])
    @pytest.mark.parametrize("edge", ["lower", "upper"])
    def test_inside_the_range_near_a_bound_is_centred_on_the_declared_value(
        self, edge: str, bins_inside: float, bound_kind: str, cdf_size: int
    ) -> None:
        """Half a bin inside (the terminal bin's centre) is where the capped span first fits without
        a translation; from there inward the collapse is symmetric about its declared value."""
        lower, upper = self.RANGE
        is_open = bound_kind == "open"
        question = self._question(
            lower, upper, cdf_size, open_lower=is_open and edge == "lower", open_upper=is_open and edge == "upper"
        )
        offset = bins_inside * grid_bin_width(lower, upper, cdf_size)
        declared = lower + offset if edge == "lower" else upper - offset
        half_span = self._full_span(cdf_size) / 2

        result = self._collapse(declared, question, value_eps=1e-6, spread_delta=1.0)

        assert all(lower <= v <= upper for v in result)
        assert all(a < b for a, b in pairwise(result))
        assert (result[0], result[-1]) == pytest.approx((declared - half_span, declared + half_span))

    def test_an_epsilon_chain_dipping_below_an_open_bound_is_placed_by_its_median(self) -> None:
        """Thirteen near-equal values chained within ``value_eps`` are one collapse
        (``is_degenerate_cluster``). Placed by ``values[0]`` this chain was read as beyond the open
        bound and left straddling it; its median sits exactly on the bound, so it goes inside."""
        question = self._question(*self.RANGE, 13, open_lower=True, open_upper=False)
        chain = [-3e-7 + i * 5e-8 for i in range(EXPECTED_PERCENTILE_COUNT)]
        assert sorted(chain)[EXPECTED_PERCENTILE_COUNT // 2] == question.lower_bound

        result = self._spread(chain, question)

        assert (result[0], result[-1]) == (question.lower_bound, question.lower_bound + self._full_span(13))
        assert all(a < b for a, b in pairwise(result))

    def test_an_epsilon_chain_inside_the_range_is_centred_on_its_median(self) -> None:
        question = self._question(*self.RANGE, 13, open_lower=True, open_upper=True)
        chain = [6.0 + i * 5e-8 for i in range(EXPECTED_PERCENTILE_COUNT)]
        median = sorted(chain)[EXPECTED_PERCENTILE_COUNT // 2]

        result = self._spread(chain, question)

        assert sum(result) / len(result) == pytest.approx(median, abs=1e-9)
        assert result[-1] - result[0] == pytest.approx(self._full_span(13))
        assert all(a < b for a, b in pairwise(result))

    def test_the_codex_epsilon_chain_repro_lands_inside_with_its_full_span(self) -> None:
        """``[i * 5e-8 for i in range(13)]`` on an open-lower [0, 12] grid (codex re-check 4): the
        median 3e-7 is inside, the symmetric span crosses the bound, so the translated plateau is
        ``[0.0, 1.0]`` under the narrowed rule; the pre-branch straddle was never the intent."""
        question = self._question(*self.RANGE, 13, open_lower=True, open_upper=False)

        result = self._spread([i * 5e-8 for i in range(EXPECTED_PERCENTILE_COUNT)], question)

        assert (result[0], result[-1]) == (0.0, 1.0)
        assert all(a < b for a, b in pairwise(result))

    def test_no_neighbour_repair_re_spaces_the_collapse(self) -> None:
        """No preceding or following value exists, so every gap is exactly the capped per-position
        spread and the translated end sits exactly on the bound."""
        question = self._question(0.0, 12.0, 13, open_lower=True, open_upper=False)

        result = self._collapse(12.0, question, value_eps=1e-6, spread_delta=1.0)

        assert result[-1] == 12.0
        assert all(b - a == pytest.approx(1 / (EXPECTED_PERCENTILE_COUNT - 1)) for a, b in pairwise(result))


class TestPartialPlateauAtABoundKeepsThePreBranchBehaviour:
    """A PARTIAL plateau at a bound gets the pre-branch spread: symmetric under the one-bin cap,
    clamped at a closed edge, then the shift-up / compress repairs against its neighbours.

    Rounds one and two of the edge-collapse fix (2026-09) translated partial plateaus too, and
    each codex re-check found a new hole (a plateau on an open bound misclassified by
    ``np.mean`` rounding, the repairs dragging a beyond-open plateau inside, a ceiling
    re-spacing collapsing to duplicates). The review finding was only ever about the WHOLE-SET
    collapse, so partial plateaus were reverted to the behaviour benchmarked on Metaculus,
    pinned here on a 13-point [0, 12] grid with a unit count-like spread (one bin, one unit):
    on an OPEN bound a partial plateau may straddle it, and one declared beyond an open bound
    keeps its symmetric spread where its neighbours leave room.
    """

    def _question(self, *, open_lower: bool, open_upper: bool) -> NumericQuestion:
        return cast(
            NumericQuestion,
            SimpleNamespace(
                open_upper_bound=open_upper,
                open_lower_bound=open_lower,
                upper_bound=12.0,
                lower_bound=0.0,
                cdf_size=13,
                id_of_question=1,
            ),
        )

    def _spread(self, values: list[float], question: NumericQuestion) -> list[float]:
        result, _ = apply_cluster_spreading(values, question, value_eps=1e-6, spread_delta=1.0, range_size=12.0)
        return result

    def test_a_plateau_on_an_open_upper_bound_straddles_it_after_the_shift_up(self) -> None:
        question = self._question(open_lower=False, open_upper=True)

        result = self._spread([0.0, 6.0, 11.8] + [12.0] * 10, question)

        assert result[:3] == [0.0, 6.0, 11.8]
        assert result[3] == pytest.approx(11.8 + 1e-6)
        assert result[-1] == pytest.approx(12.8 + 1e-6)
        assert all(b - a == pytest.approx(1 / 9) for a, b in pairwise(result[3:]))

    def test_a_plateau_on_an_open_lower_bound_straddles_it_after_the_compress(self) -> None:
        question = self._question(open_lower=True, open_upper=False)

        result = self._spread([0.0] * 10 + [0.2, 6.0, 12.0], question)

        assert result[10:] == [0.2, 6.0, 12.0]
        assert result[0] == pytest.approx(-0.5)
        assert result[9] == pytest.approx(0.13)
        assert all(a < b for a, b in pairwise(result))

    def test_a_plateau_on_a_closed_lower_bound_is_clamped_then_compressed_below_its_successor(self) -> None:
        question = self._question(open_lower=False, open_upper=False)

        result = self._spread([0.0] * 10 + [0.2, 6.0, 12.0], question)

        assert result[0] == pytest.approx(minimum_separation(12.0))
        assert result[9] == pytest.approx(0.18)
        assert all(a < b for a, b in pairwise(result))

    @pytest.mark.parametrize(
        ("values", "plateau", "expected_edges"),
        [
            ([-0.25] * 10 + [1.0, 6.0, 12.0], slice(0, 10), (-0.75, 0.25)),
            ([0.0, 6.0, 11.0] + [12.25] * 10, slice(3, 13), (11.75, 12.75)),
        ],
        ids=["below_open_lower", "above_open_upper"],
    )
    def test_a_plateau_declared_beyond_an_open_bound_keeps_its_symmetric_spread(
        self, values: list[float], plateau: slice, expected_edges: tuple[float, float]
    ) -> None:
        question = self._question(open_lower=True, open_upper=True)

        spread = self._spread(list(values), question)[plateau]

        assert sum(spread) / len(spread) == pytest.approx(values[plateau][0])
        assert (spread[0], spread[-1]) == pytest.approx(expected_edges)

    def test_a_plateau_declared_beyond_a_closed_lower_bound_is_folded_onto_the_standoff(self) -> None:
        """The pre-branch closed-edge clamp: what the symmetric spread puts outside a closed bound
        lands at the ``minimum_separation`` standoff, so the pipeline's tolerance clamp sees
        nothing outside."""
        question = self._question(open_lower=False, open_upper=False)

        result = self._spread([-0.25] * 10 + [1.0, 6.0, 12.0], question)

        assert result[:7] == pytest.approx([minimum_separation(12.0)] * 7)
        assert result[7:10] == pytest.approx([-0.75 + 7 / 9, -0.75 + 8 / 9, 0.25])

    def test_a_plateau_declared_beyond_a_closed_upper_bound_is_folded_onto_the_standoff(self) -> None:
        question = self._question(open_lower=False, open_upper=False)

        result = self._spread([0.0, 6.0, 11.0] + [12.25] * 10, question)

        assert result[3:6] == pytest.approx([11.75, 11.75 + 1 / 9, 11.75 + 2 / 9])
        assert result[6:] == pytest.approx([12.0 - minimum_separation(12.0)] * 7)
