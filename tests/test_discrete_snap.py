"""Comprehensive tests for discrete integer CDF snapping.

Tests cover:
  - Core snapping algorithm (step shape, PMF conservation, monotonicity, min/max step)
  - Boundary handling (open/closed)
  - Edge cases (high-PMF integer, many integers, single integer, log-scaled)
  - Vote parsing and majority logic
  - Round-trip validation against Metaculus constraints
"""

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import NumericQuestion

from metaculus_bot.backtest.scoring import numeric_log_score
from metaculus_bot.constants import DISCRETE_SNAP_MAX_INTEGERS, NUM_MAX_STEP, NUM_MIN_PROB_STEP
from metaculus_bot.discrete_snap import (
    OutcomeTypeResult,
    majority_votes_discrete,
    snap_cdf_to_integers,
    snap_distribution_to_integers,
)


def _make_question(
    lower_bound: float = 0.0,
    upper_bound: float = 10.0,
    open_lower: bool = False,
    open_upper: bool = False,
    zero_point: float | None = None,
    cdf_size: int = 201,
) -> NumericQuestion:
    return NumericQuestion(
        id_of_question=999,
        id_of_post=999,
        page_url="https://example.com/q/999",
        question_text="Test numeric question",
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


def _make_smooth_cdf(lower_bound: float, upper_bound: float, center: float, spread: float) -> list[float]:
    """Generate a smooth sigmoid-like 201-point CDF centered at `center`."""
    x = np.linspace(lower_bound, upper_bound, 201)
    # Logistic CDF
    cdf = 1.0 / (1.0 + np.exp(-(x - center) / spread))
    # Pin endpoints for closed bounds
    cdf[0] = 0.0
    cdf[-1] = 1.0
    # Ensure strictly increasing
    min_step = NUM_MIN_PROB_STEP * 1.1
    for i in range(1, len(cdf)):
        if cdf[i] - cdf[i - 1] < min_step:
            cdf[i] = cdf[i - 1] + min_step
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[-1] = 1.0
    return cdf.tolist()


# =============================================================================
# Core Algorithm Tests
# =============================================================================


class TestSnapCdfToIntegers:
    def test_basic_step_shape(self):
        """Snapped CDF concentrates more mass at integer grid points than smooth CDF."""
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        result = snap_cdf_to_integers(cdf, 0.0, 10.0, False, False)
        assert result is not None
        assert len(result) == 201

        # The snapped CDF should have higher variance in step sizes than the smooth CDF
        # (large steps at integers, small steps between them)
        smooth_steps = np.diff(cdf)
        snapped_steps = np.diff(result)
        assert np.std(snapped_steps) > np.std(smooth_steps)

        # The max step in snapped should be larger (concentrated at integers)
        assert np.max(snapped_steps) > np.max(smooth_steps) * 0.8

    def test_pmf_conservation(self):
        """Total probability mass should be conserved after snapping."""
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        result = snap_cdf_to_integers(cdf, 0.0, 10.0, False, False)
        assert result is not None
        # Total mass = CDF[-1] - CDF[0]
        original_mass = cdf[-1] - cdf[0]
        snapped_mass = result[-1] - result[0]
        assert abs(snapped_mass - original_mass) < 0.05  # Allow small deviation from redistribution

    def test_strictly_increasing(self):
        """Every adjacent pair must differ by at least min_step."""
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        result = snap_cdf_to_integers(cdf, 0.0, 10.0, False, False)
        assert result is not None
        diffs = np.diff(result)
        assert np.all(diffs >= NUM_MIN_PROB_STEP - 1e-10)

    def test_min_step_compliance(self):
        """Min step must be >= 5e-5 (Metaculus API requirement)."""
        cdf = _make_smooth_cdf(0.0, 20.0, center=10.0, spread=3.0)
        result = snap_cdf_to_integers(cdf, 0.0, 20.0, False, False)
        assert result is not None
        min_diff = float(np.min(np.diff(result)))
        assert min_diff >= NUM_MIN_PROB_STEP - 1e-10

    def test_max_step_compliance(self):
        """Max step must be <= 0.2 (Metaculus API requirement)."""
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        result = snap_cdf_to_integers(cdf, 0.0, 10.0, False, False)
        assert result is not None
        max_diff = float(np.max(np.diff(result)))
        assert max_diff <= NUM_MAX_STEP + 1e-10

    def test_values_in_unit_interval(self):
        """All CDF values must be in [0, 1]."""
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        result = snap_cdf_to_integers(cdf, 0.0, 10.0, False, False)
        assert result is not None
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_wider_range(self):
        """Snapping works for a wider range like [0, 150] with 151 integers."""
        cdf = _make_smooth_cdf(0.0, 150.0, center=50.0, spread=20.0)
        result = snap_cdf_to_integers(cdf, 0.0, 150.0, False, False)
        assert result is not None
        assert len(result) == 201
        diffs = np.diff(result)
        assert np.all(diffs >= NUM_MIN_PROB_STEP - 1e-10)
        assert np.all(diffs <= NUM_MAX_STEP + 1e-10)


# =============================================================================
# Boundary Handling Tests
# =============================================================================


class TestBoundaryHandling:
    def test_closed_lower_bound(self):
        """Closed lower bound → CDF[0] == 0.0."""
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        result = snap_cdf_to_integers(cdf, 0.0, 10.0, open_lower_bound=False, open_upper_bound=False)
        assert result is not None
        assert abs(result[0] - 0.0) < 1e-10

    def test_closed_upper_bound(self):
        """Closed upper bound → CDF[-1] == 1.0."""
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        result = snap_cdf_to_integers(cdf, 0.0, 10.0, open_lower_bound=False, open_upper_bound=False)
        assert result is not None
        assert abs(result[-1] - 1.0) < 1e-10

    def test_open_lower_bound(self):
        """Open lower bound → CDF[0] >= 0.001."""
        # Start with open-bound CDF
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        cdf[0] = 0.005  # Open lower
        result = snap_cdf_to_integers(cdf, 0.0, 10.0, open_lower_bound=True, open_upper_bound=False)
        assert result is not None
        assert result[0] >= 0.001

    def test_open_upper_bound(self):
        """Open upper bound → CDF[-1] <= 0.999."""
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        cdf[-1] = 0.995  # Open upper
        result = snap_cdf_to_integers(cdf, 0.0, 10.0, open_lower_bound=False, open_upper_bound=True)
        assert result is not None
        assert result[-1] <= 0.999


def create_pchip_distribution_from_cdf(
    cdf: list[float], question: NumericQuestion, zero_point: float | None = None
) -> NumericDistribution:
    from metaculus_bot.pchip_processing import create_pchip_numeric_distribution

    return create_pchip_numeric_distribution(
        pchip_cdf=cdf,
        percentile_list=[Percentile(value=float(question.lower_bound + question.upper_bound) / 2, percentile=0.5)],
        question=question,
        zero_point=zero_point,
    )


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    def test_high_probability_integer_max_step_redistribution(self):
        """When one integer has PMF > 0.2, max-step redistribution kicks in."""
        # Create a CDF with very concentrated mass around integer 5
        x = np.linspace(0.0, 10.0, 201)
        cdf = 1.0 / (1.0 + np.exp(-(x - 5.0) / 0.3))  # Very narrow spread
        cdf[0] = 0.0
        cdf[-1] = 1.0
        # Ensure min step
        for i in range(1, len(cdf)):
            if cdf[i] - cdf[i - 1] < NUM_MIN_PROB_STEP * 1.1:
                cdf[i] = cdf[i - 1] + NUM_MIN_PROB_STEP * 1.1
        cdf = np.clip(cdf, 0.0, 1.0)
        cdf[-1] = 1.0

        result = snap_cdf_to_integers(cdf.tolist(), 0.0, 10.0, False, False)
        assert result is not None
        max_diff = float(np.max(np.diff(result)))
        assert max_diff <= NUM_MAX_STEP + 1e-10

    def test_too_many_integers_returns_none(self):
        """More than DISCRETE_SNAP_MAX_INTEGERS → skip snapping."""
        n_ints = DISCRETE_SNAP_MAX_INTEGERS + 50
        cdf = _make_smooth_cdf(0.0, float(n_ints), center=float(n_ints / 2), spread=float(n_ints / 5))
        result = snap_cdf_to_integers(cdf, 0.0, float(n_ints), False, False)
        assert result is None

    def test_already_discrete_question_skipped(self):
        """Question with cdf_size != 201 (already discrete) → skip."""
        question = _make_question(lower_bound=-0.5, upper_bound=7.5, cdf_size=9)
        dist = NumericDistribution(
            declared_percentiles=[Percentile(value=3.0, percentile=0.5)],
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=7.5,
            lower_bound=-0.5,
            zero_point=None,
            cdf_size=9,
        )
        result = snap_distribution_to_integers(dist, question)
        assert result is None

    def test_log_scaled_small_range(self):
        """Log-scaled question with small integer range → snapping still works."""
        # Create a question with zero_point but few integers
        question = _make_question(lower_bound=0.0, upper_bound=20.0, zero_point=0.0)
        cdf = _make_smooth_cdf(0.0, 20.0, center=10.0, spread=3.0)
        dist_kwargs = question.model_dump()
        # Remove fields that aren't in NumericDistribution
        for field in [
            "id_of_question",
            "id_of_post",
            "page_url",
            "question_text",
            "background_info",
            "resolution_criteria",
            "fine_print",
            "published_time",
            "close_time",
            "unit_of_measure",
            "question_type",
            "nominal_upper_bound",
            "nominal_lower_bound",
        ]:
            dist_kwargs.pop(field, None)
        from metaculus_bot.pchip_processing import create_pchip_numeric_distribution

        dist = create_pchip_numeric_distribution(
            pchip_cdf=cdf,
            percentile_list=[Percentile(value=10.0, percentile=0.5)],
            question=question,
            zero_point=0.0,
        )
        result = snap_distribution_to_integers(dist, question)
        # 21 integers in [0, 20], should snap
        assert result is not None

    def test_log_scaled_huge_range_skipped(self):
        """Log-scaled question with huge integer range → skip."""
        question = _make_question(lower_bound=0.0, upper_bound=10000.0, zero_point=0.0)
        cdf = _make_smooth_cdf(0.0, 10000.0, center=5000.0, spread=1000.0)
        dist = create_pchip_distribution_from_cdf(cdf, question, zero_point=0.0)
        result = snap_distribution_to_integers(dist, question)
        assert result is None

    def test_single_integer_in_range(self):
        """Edge case: only one integer in range."""
        cdf = _make_smooth_cdf(0.5, 1.5, center=1.0, spread=0.2)
        result = snap_cdf_to_integers(cdf, 0.5, 1.5, False, False)
        assert result is not None
        assert len(result) == 201
        diffs = np.diff(result)
        assert np.all(diffs >= NUM_MIN_PROB_STEP - 1e-10)

    def test_no_integers_in_range(self):
        """Edge case: no integers in range (e.g., [0.1, 0.9])."""
        cdf = _make_smooth_cdf(0.1, 0.9, center=0.5, spread=0.1)
        result = snap_cdf_to_integers(cdf, 0.1, 0.9, False, False)
        assert result is None


# =============================================================================
# Vote Parsing Tests
# =============================================================================


class TestOutcomeTypeResult:
    def test_model_discrete(self):
        result = OutcomeTypeResult(is_discrete_integer=True)
        assert result.is_discrete_integer is True

    def test_model_continuous(self):
        result = OutcomeTypeResult(is_discrete_integer=False)
        assert result.is_discrete_integer is False

    def test_model_from_dict(self):
        result = OutcomeTypeResult.model_validate({"is_discrete_integer": True})
        assert result.is_discrete_integer is True


class TestMajorityVote:
    def test_majority_discrete(self):
        assert majority_votes_discrete([True, True, False]) is True

    def test_majority_continuous(self):
        assert majority_votes_discrete([False, False, True]) is False

    def test_tie_is_not_majority(self):
        assert majority_votes_discrete([True, False]) is False

    def test_unanimous_discrete(self):
        assert majority_votes_discrete([True, True, True]) is True

    def test_empty_votes(self):
        assert majority_votes_discrete([]) is False

    def test_single_vote_discrete(self):
        assert majority_votes_discrete([True]) is True

    def test_single_vote_continuous(self):
        assert majority_votes_discrete([False]) is False


# =============================================================================
# Round-Trip Validation Tests
# =============================================================================


class TestRoundTripValidation:
    """Verify snapped CDFs pass all Metaculus constraint checks."""

    def _assert_metaculus_constraints(self, cdf: list[float], open_lower: bool, open_upper: bool):
        """Check all server-side constraints from Metaculus backend."""
        assert len(cdf) == 201
        assert all(0.0 <= v <= 1.0 for v in cdf)

        diffs = np.diff(cdf)
        min_diff = float(np.min(diffs))
        max_diff = float(np.max(diffs))

        # Min step: 0.01 / 200 = 5e-5
        assert min_diff >= 5e-5 - 1e-10, f"min_diff={min_diff} < 5e-5"
        # Max step: 0.2
        assert max_diff <= 0.2 + 1e-10, f"max_diff={max_diff} > 0.2"
        # Boundary constraints
        if not open_lower:
            assert abs(cdf[0]) < 1e-10, f"closed lower bound: cdf[0]={cdf[0]}"
        else:
            assert cdf[0] >= 0.001, f"open lower bound: cdf[0]={cdf[0]} < 0.001"
        if not open_upper:
            assert abs(cdf[-1] - 1.0) < 1e-10, f"closed upper bound: cdf[-1]={cdf[-1]}"
        else:
            assert cdf[-1] <= 0.999, f"open upper bound: cdf[-1]={cdf[-1]} > 0.999"

    def test_roundtrip_closed_bounds(self):
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        result = snap_cdf_to_integers(cdf, 0.0, 10.0, False, False)
        assert result is not None
        self._assert_metaculus_constraints(result, False, False)

    def test_roundtrip_open_bounds(self):
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        cdf[0] = 0.005
        cdf[-1] = 0.995
        result = snap_cdf_to_integers(cdf, 0.0, 10.0, True, True)
        assert result is not None
        self._assert_metaculus_constraints(result, True, True)

    def test_roundtrip_concentrated_distribution(self):
        """Concentrated distribution (narrow spread) still passes all constraints."""
        x = np.linspace(0.0, 10.0, 201)
        cdf = 1.0 / (1.0 + np.exp(-(x - 5.0) / 0.5))
        cdf[0] = 0.0
        cdf[-1] = 1.0
        for i in range(1, len(cdf)):
            if cdf[i] - cdf[i - 1] < NUM_MIN_PROB_STEP * 1.1:
                cdf[i] = cdf[i - 1] + NUM_MIN_PROB_STEP * 1.1
        cdf = np.clip(cdf, 0.0, 1.0)
        cdf[-1] = 1.0

        result = snap_cdf_to_integers(cdf.tolist(), 0.0, 10.0, False, False)
        assert result is not None
        self._assert_metaculus_constraints(result, False, False)

    def test_roundtrip_wide_range(self):
        """Range [0, 150] with 151 integers passes constraints."""
        cdf = _make_smooth_cdf(0.0, 150.0, center=75.0, spread=25.0)
        result = snap_cdf_to_integers(cdf, 0.0, 150.0, False, False)
        assert result is not None
        self._assert_metaculus_constraints(result, False, False)

    def test_roundtrip_framework_validation(self):
        """Snapped CDF passes the framework's _validate_pchip_cdf()."""
        from metaculus_bot.pchip_processing import _validate_pchip_cdf

        question = _make_question(lower_bound=0.0, upper_bound=10.0)
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        result = snap_cdf_to_integers(cdf, 0.0, 10.0, False, False)
        assert result is not None
        _validate_pchip_cdf(result, question)


# =============================================================================
# Scoring Tests — Metaculus log score impact of snapping
# =============================================================================


def _log_score(cdf: list[float], resolution: float, lb: float, ub: float) -> float:
    """Shorthand for numeric_log_score with closed bounds."""
    return numeric_log_score(cdf, resolution, lb, ub, open_lower_bound=False, open_upper_bound=False)


class TestDiscreteSnapScoring:
    """Metaculus log score impact of discrete snapping.

    With steps placed at integer positions k, the snapped CDF concentrates
    mass in exactly the scored PMF bucket for integer resolutions. This means
    snapping improves log scores for integer resolutions (the common case for
    discrete-integer questions). Non-integer resolutions still score worse
    since mass is concentrated at integers, not between them.
    """

    @pytest.mark.parametrize("resolution", range(1, 10))
    def test_snapped_beats_smooth_for_integer_resolution(self, resolution: int):
        """Snapping improves log score for integer resolutions (step alignment at k)."""
        cdf_smooth = _make_smooth_cdf(0.0, 10.0, center=float(resolution), spread=2.0)
        cdf_snapped = snap_cdf_to_integers(cdf_smooth, 0.0, 10.0, False, False)
        assert cdf_snapped is not None
        score_smooth = _log_score(cdf_smooth, float(resolution), 0.0, 10.0)
        score_snapped = _log_score(cdf_snapped, float(resolution), 0.0, 10.0)
        assert np.isfinite(score_snapped)
        assert score_snapped > score_smooth

    def test_snapped_scores_finite_for_noninteger(self):
        """For non-integer resolution, snapping produces a finite (but worse) score."""
        resolution = 5.3
        cdf_smooth = _make_smooth_cdf(0.0, 10.0, center=resolution, spread=2.0)
        cdf_snapped = snap_cdf_to_integers(cdf_smooth, 0.0, 10.0, False, False)
        assert cdf_snapped is not None
        score_smooth = _log_score(cdf_smooth, resolution, 0.0, 10.0)
        score_snapped = _log_score(cdf_snapped, resolution, 0.0, 10.0)
        assert np.isfinite(score_snapped)
        assert score_snapped < score_smooth

    def test_concentrated_distribution_snapped_is_finite(self):
        """Narrow spread with max-step redistribution still produces finite scores."""
        cdf_smooth = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=0.5)
        cdf_snapped = snap_cdf_to_integers(cdf_smooth, 0.0, 10.0, False, False)
        assert cdf_snapped is not None
        score_snapped = _log_score(cdf_snapped, 5.0, 0.0, 10.0)
        assert np.isfinite(score_snapped)

    def test_lower_bound_integer_closed_bound(self):
        """Resolution at lower bound (0 on [0,10], closed) improves with snapping."""
        cdf_smooth = _make_smooth_cdf(0.0, 10.0, center=0.0, spread=2.0)
        cdf_snapped = snap_cdf_to_integers(cdf_smooth, 0.0, 10.0, False, False)
        assert cdf_snapped is not None
        assert abs(cdf_snapped[0] - 0.0) < 1e-10
        score_snapped = _log_score(cdf_snapped, 0.0, 0.0, 10.0)
        assert np.isfinite(score_snapped)


# =============================================================================
# Distribution-Level Tests
# =============================================================================


class TestSnapDistributionToIntegers:
    def test_returns_distribution(self):
        question = _make_question(lower_bound=0.0, upper_bound=10.0)
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        dist = create_pchip_distribution_from_cdf(cdf, question)
        result = snap_distribution_to_integers(dist, question)
        assert result is not None
        assert isinstance(result, NumericDistribution)
        assert len(result.cdf) == 201

    def test_skips_discrete_question(self):
        question = _make_question(lower_bound=-0.5, upper_bound=7.5, cdf_size=9)
        dist = NumericDistribution(
            declared_percentiles=[Percentile(value=3.0, percentile=0.5)],
            open_upper_bound=False,
            open_lower_bound=False,
            upper_bound=7.5,
            lower_bound=-0.5,
            zero_point=None,
            cdf_size=9,
        )
        assert snap_distribution_to_integers(dist, question) is None


# =============================================================================
# Integration Test: _maybe_snap_to_integers
# =============================================================================


class TestMaybeSnapIntegration:
    """Test the vote-collection → majority-check → snap-application flow
    through TemplateForecaster._maybe_snap_to_integers."""

    def _make_bot(self):
        from forecasting_tools import GeneralLlm

        from main import TemplateForecaster

        llms_config = {
            "default": GeneralLlm(model="test_default"),
            "summarizer": "mock_summarizer_model",
            "parser": "mock_parser_model",
            "researcher": "mock_researcher_model",
        }
        return TemplateForecaster(llms=llms_config)

    def test_majority_discrete_snaps_distribution(self):
        """When majority of votes are DISCRETE, snapping is applied and the returned distribution differs."""
        bot = self._make_bot()
        question = _make_question(lower_bound=0.0, upper_bound=10.0)
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        distribution = create_pchip_distribution_from_cdf(cdf, question)

        qid = question.id_of_question
        bot._discrete_integer_votes[qid] = [True, True, False]

        result = bot._maybe_snap_to_integers(distribution, question)

        assert result is not distribution
        assert isinstance(result, NumericDistribution)
        assert len(result.cdf) == 201

    def test_majority_continuous_returns_original(self):
        """When majority of votes are CONTINUOUS, the original distribution object is returned unchanged."""
        bot = self._make_bot()
        question = _make_question(lower_bound=0.0, upper_bound=10.0)
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        distribution = create_pchip_distribution_from_cdf(cdf, question)

        qid = question.id_of_question
        bot._discrete_integer_votes[qid] = [False, False, True]

        result = bot._maybe_snap_to_integers(distribution, question)

        assert result is distribution

    def test_no_votes_returns_original(self):
        """When no votes exist for the question, the original distribution is returned."""
        bot = self._make_bot()
        question = _make_question(lower_bound=0.0, upper_bound=10.0)
        cdf = _make_smooth_cdf(0.0, 10.0, center=5.0, spread=2.0)
        distribution = create_pchip_distribution_from_cdf(cdf, question)

        result = bot._maybe_snap_to_integers(distribution, question)

        assert result is distribution
