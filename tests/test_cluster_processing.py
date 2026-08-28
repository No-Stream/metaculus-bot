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

from metaculus_bot.numeric.cluster_processing import (
    apply_cluster_spreading,
    apply_jitter_for_duplicates,
    compute_cluster_parameters,
    detect_count_like_pattern,
    ensure_strictly_increasing_bounded,
    is_degenerate_cluster,
)


def _make_question(open_upper=False, open_lower=False, lower=0.0, upper=100.0) -> NumericQuestion:
    return cast(
        NumericQuestion,
        SimpleNamespace(
            open_upper_bound=open_upper,
            open_lower_bound=open_lower,
            upper_bound=upper,
            lower_bound=lower,
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
        # Check that clustered values are now different
        cluster_values = result[1:4]  # The cluster positions
        assert len(set(cluster_values)) == 3  # All different now
        assert all(a < b for a, b in pairwise(cluster_values))  # Strictly increasing

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
        # Nothing to compare, and a 1-element set never reaches the spreader.
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
    implementation as of 2026-08-26.
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
        for got, want in zip(result, expected_values, strict=True):
            assert got == pytest.approx(want, rel=1e-12, abs=1e-15)

    def test_spreading_mutates_the_caller_list_in_place(self):
        """The spreader returns the SAME list object it was handed; the pipeline relies
        on the return value, but a caller reusing its own list must see the mutation."""
        values = [10.0, 20.0, 20.0, 20.0, 30.0]
        question = _make_question()

        result, _ = apply_cluster_spreading(values, question, value_eps=1e-6, spread_delta=1.0, range_size=100.0)

        assert result is values
