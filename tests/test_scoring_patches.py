"""
Tests for the scoring patches module.

Tests the monkey patching and scoring logic for mixed question types.
"""

import math
from unittest.mock import Mock

import numpy as np
import pytest

from metaculus_bot.scoring_patches import (
    apply_scoring_patches,
    calculate_multiple_choice_baseline_score,
    calculate_numeric_baseline_score,
    extract_multiple_choice_probabilities,
    extract_numeric_percentiles,
    get_scoring_path_stats,
    reset_scoring_path_stats,
)


class TestMultipleChoiceExtraction:
    """Test multiple choice probability extraction."""

    def test_extract_mc_probabilities_success(self):
        """Test successful extraction of MC probabilities."""
        option1 = Mock()
        option1.option_name = "Option A"
        option1.probability = 0.3

        option2 = Mock()
        option2.option_name = "Option B"
        option2.probability = 0.7

        prediction = Mock()
        prediction.predicted_options = [option2, option1]  # Unsorted

        probs, option_names = extract_multiple_choice_probabilities(prediction)

        # Should be sorted by option_name: A, B
        assert probs == [0.3, 0.7]
        assert option_names == ["Option A", "Option B"]

    def test_extract_mc_probabilities_empty(self):
        """Test extraction with empty or invalid data."""
        prediction = Mock()
        prediction.predicted_options = None

        probs, option_names = extract_multiple_choice_probabilities(prediction)
        assert probs == []
        assert option_names == []

        # Test with empty list
        prediction.predicted_options = []
        probs, option_names = extract_multiple_choice_probabilities(prediction)
        assert probs == []
        assert option_names == []


class TestNumericExtraction:
    """Test numeric percentile extraction."""

    def test_extract_numeric_percentiles_success(self):
        """Test successful extraction of numeric percentiles."""
        p1 = Mock()
        p1.percentile = 10
        p1.value = 100

        p2 = Mock()
        p2.percentile = 90
        p2.value = 1000

        prediction = Mock()
        prediction.declared_percentiles = [p1, p2]

        percentiles = extract_numeric_percentiles(prediction)
        assert percentiles == [(10, 100), (90, 1000)]

    def test_extract_numeric_percentiles_empty(self):
        """Test extraction with empty or invalid data."""
        prediction = Mock()
        prediction.declared_percentiles = None

        percentiles = extract_numeric_percentiles(prediction)
        assert percentiles == []


class TestMultipleChoiceScoring:
    """Test multiple choice baseline scoring."""

    def test_mc_scoring_success(self):
        """Test successful MC scoring."""
        # Create mock question
        question = Mock()
        question.id_of_question = 123
        question.num_predictions = 15

        # Create mock prediction with 3 options
        option1 = Mock()
        option1.option_name = "A"
        option1.probability = 0.1

        option2 = Mock()
        option2.option_name = "B"
        option2.probability = 0.8

        option3 = Mock()
        option3.option_name = "C"
        option3.probability = 0.1

        prediction = Mock()
        prediction.predicted_options = [option1, option2, option3]

        # Provide community CP aligned to options
        question.options = ["A", "B", "C"]
        question.api_json = {
            "question": {
                "type": "multiple_choice",
                "options": ["A", "B", "C"],
                "aggregations": {"recency_weighted": {"latest": {"forecast_values": [0.1, 0.8, 0.1]}}},
            }
        }

        # Create mock report
        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_multiple_choice_baseline_score(report)

        # Score should be calculated and finite
        assert score is not None
        assert math.isfinite(score)
        assert isinstance(score, float)

        # Bound matches the sibling per-type assertions below (-200..100), not the
        # 5x-wider -500..500 this used to carry. The MC formula caps at +100 by
        # construction, so the old upper bound was unreachable by any input; measured
        # value for these fixed literals is 41.83.
        assert -200 <= score <= 100, f"MC baseline score off the Metaculus-like scale: {score}"

    def test_mc_scoring_unreadable_report_scores_none(self):
        """An unusable report scores None rather than raising out of the scorer.

        ``report.prediction`` is left as a bare Mock, so ``predicted_options`` is a Mock
        too and sorting it raises TypeError inside the scorer's outer boundary. That
        boundary is what this pins: ``expected_baseline_score`` is read from a property
        while a benchmark is being assembled, so a report it cannot read must degrade to
        None. (This test used to be titled "insufficient_predictions" and set
        ``num_predictions = 5``, asserting a community-count gate that 7fe4afe deleted in
        2025-08 — it has been passing for this unrelated reason ever since.)
        """
        question = Mock()
        question.id_of_question = 123

        report = Mock()
        report.question = question

        score = calculate_multiple_choice_baseline_score(report)
        assert score is None

    def test_mc_scoring_no_bot_probs(self):
        """Test MC scoring when bot probabilities cannot be extracted."""
        question = Mock()
        question.id_of_question = 123
        question.num_predictions = 15

        prediction = Mock()
        prediction.predicted_options = None  # No options

        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_multiple_choice_baseline_score(report)
        assert score is None


class TestNumericScoring:
    """Test numeric baseline scoring with PDF approach."""

    def test_numeric_scoring_success(self):
        """Test successful numeric scoring with community benchmark approach."""
        # Create mock question with bounds
        question = Mock()
        question.id_of_question = 456
        question.num_predictions = 20
        question.lower_bound = 0.0
        question.upper_bound = 100.0

        # Create mock percentiles (enough for PDF estimation)
        percentiles = []
        for p, v in [(10, 10), (20, 20), (40, 40), (60, 60), (80, 80), (90, 90)]:
            mock_p = Mock()
            mock_p.percentile = p
            mock_p.value = v
            percentiles.append(mock_p)

        prediction = Mock()
        prediction.declared_percentiles = percentiles

        # Provide community CDF (uniform for simplicity)
        question.api_json = {
            "question": {
                "aggregations": {
                    "recency_weighted": {"latest": {"forecast_values": np.linspace(0.0, 1.0, 201).tolist()}}
                }
            }
        }

        # Create mock report
        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_numeric_baseline_score(report)

        # Score should be calculated and finite
        assert score is not None
        assert math.isfinite(score)
        assert isinstance(score, float)

        # Score should be in new expanded range with fixed normalization ln(10)
        # Range should be similar to MC scores: roughly [-100, +20]
        assert -200 <= score <= 100

    def test_numeric_scoring_pmf_path(self):
        """Test numeric scoring via PMF path when both model and community CDFs exist."""
        # Create mock question with bounds
        question = Mock()
        question.id_of_question = 789
        question.lower_bound = 0.0
        question.upper_bound = 100.0
        # Provide community CDF (uniform for simplicity)
        community_cdf = np.linspace(0.0, 1.0, 201).tolist()
        question.api_json = {
            "question": {"aggregations": {"recency_weighted": {"latest": {"forecast_values": community_cdf}}}}
        }

        # Create model CDF as a list of objects with .percentile
        class P:
            def __init__(self, percentile):
                self.percentile = percentile

        model_cdf = [P(p) for p in np.linspace(0.0, 1.0, 201).tolist()]

        prediction = Mock()
        prediction.cdf = model_cdf

        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_numeric_baseline_score(report)

        assert score is not None
        assert math.isfinite(score)
        assert isinstance(score, float)
        # Score should be in realistic community benchmark range
        # With fixed normalization, scores should be in MC-like range
        assert -200 <= score <= 100

    def test_numeric_scoring_insufficient_percentiles(self):
        """Test numeric scoring with insufficient percentiles."""
        question = Mock()
        question.id_of_question = 456
        question.num_predictions = 20

        # Only 2 percentiles - insufficient for PDF estimation
        p1 = Mock()
        p1.percentile = 10
        p1.value = 100

        p2 = Mock()
        p2.percentile = 90
        p2.value = 200

        prediction = Mock()
        prediction.declared_percentiles = [p1, p2]

        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_numeric_baseline_score(report)
        assert score is None

    def test_numeric_scoring_unreadable_report_scores_none(self):
        """An unusable report scores None through the percentile fallback, not a raise.

        With ``report.prediction`` a bare Mock there is no model CDF and no community CDF,
        so scoring degrades to the declared-percentile fallback, whose own boundary catches
        the TypeError from ``len(Mock())`` and returns None. Same history as the MC sibling
        above: this asserted a community-count gate deleted in 7fe4afe.
        """
        question = Mock()
        question.id_of_question = 456

        report = Mock()
        report.question = question

        score = calculate_numeric_baseline_score(report)
        assert score is None


class _Percentile:
    """Minimal stand-in for the ``.percentile``-bearing objects a model CDF is made of."""

    def __init__(self, percentile):
        self.percentile = percentile


class TestNumericSuccessCountersByPath:
    """Each numeric scoring path counts its OWN success.

    ``_calculate_relative_numeric_score`` is shared by the PMF path and the declared-
    percentile fallback, so the success bump cannot live inside it: it used to bump
    ``numeric_pmf_successes`` for both, which let pmf_successes exceed pmf_attempts and left
    ``numeric_fallback_successes`` at zero forever even though the run-log summary prints it.
    """

    @staticmethod
    def _uniform_community_api_json(n_points):
        return {
            "question": {
                "aggregations": {
                    "recency_weighted": {"latest": {"forecast_values": np.linspace(0.0, 1.0, n_points).tolist()}}
                }
            }
        }

    def test_pmf_path_counts_a_pmf_success_only(self):
        question = Mock()
        question.id_of_question = 4321
        question.lower_bound = 0.0
        question.upper_bound = 100.0
        question.api_json = self._uniform_community_api_json(11)

        prediction = Mock()
        prediction.cdf = [_Percentile(i / 10.0) for i in range(11)]

        report = Mock()
        report.question = question
        report.prediction = prediction

        reset_scoring_path_stats()
        assert calculate_numeric_baseline_score(report) is not None

        stats = get_scoring_path_stats()
        assert stats["numeric_pmf_attempts"] == 1
        assert stats["numeric_pmf_successes"] == 1
        assert stats["numeric_fallback_attempts"] == 0
        assert stats["numeric_fallback_successes"] == 0

    def test_fallback_path_counts_a_fallback_success_only(self):
        question = Mock()
        question.id_of_question = 4322
        question.lower_bound = 0.0
        question.upper_bound = 100.0
        # No community aggregations, so the scorer degrades to the declared percentiles.
        question.api_json = {"question": {"aggregations": {}}}

        declared = [_Percentile(p) for p in (5.0, 20.0, 40.0, 50.0, 60.0, 80.0, 95.0)]
        prediction = Mock()
        prediction.cdf = None
        prediction.declared_percentiles = declared

        report = Mock()
        report.question = question
        report.prediction = prediction

        reset_scoring_path_stats()
        assert calculate_numeric_baseline_score(report) is not None

        stats = get_scoring_path_stats()
        assert stats["numeric_fallback_attempts"] == 1
        assert stats["numeric_fallback_successes"] == 1
        assert stats["numeric_pmf_attempts"] == 0
        assert stats["numeric_pmf_successes"] == 0


class TestRelativeNumericScoring:
    """Test relative numeric scoring against community distribution (community benchmark context)."""

    def test_relative_scoring_with_community_cdf(self):
        """Test relative numeric scoring using community CDF as expectation weights."""
        # Create mock question with bounds [0, 100]
        question = Mock()
        question.id_of_question = 999
        question.lower_bound = 0.0
        question.upper_bound = 100.0

        # Mock uniform community CDF
        question.api_json = {
            "question": {
                "aggregations": {
                    "recency_weighted": {
                        "latest": {
                            "forecast_values": np.linspace(0.0, 1.0, 201).tolist()  # Uniform community
                        }
                    }
                }
            }
        }

        # Create bot prediction with concentration around middle (better than uniform)
        class P:
            def __init__(self, percentile):
                self.percentile = percentile

        # Create CDF that's more concentrated in middle than uniform
        model_cdf = []
        for i in range(201):
            p = i / 200.0
            # Sigmoid-like concentration
            cdf_val = 1.0 / (1.0 + math.exp(-8 * (p - 0.5)))
            model_cdf.append(P(cdf_val))

        prediction = Mock()
        prediction.cdf = model_cdf

        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_numeric_baseline_score(report)

        # Should get a finite score; with calibrated normalization, concentrated vs uniform
        # may be more negative than earlier bands, but should remain bounded.
        assert score is not None
        assert isinstance(score, float)
        assert -230 <= score <= 120

    def test_relative_scoring_fallback_to_percentiles(self):
        """Test fallback to percentiles when CDF unavailable."""
        question = Mock()
        question.id_of_question = 998
        question.lower_bound = 0.0
        question.upper_bound = 100.0

        # No community CDF - will fall back to uniform community
        question.api_json = {"question": {"aggregations": {}}}

        # Create bot prediction using declared percentiles
        percentiles = []

        class P:
            def __init__(self, percentile, value):
                self.percentile = percentile
                self.value = value

        # Bot has tight distribution around 50 (better than uniform)
        values = [30, 45, 48, 50, 52, 55, 70]
        percs = [0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95]
        for p, v in zip(percs, values, strict=True):
            percentiles.append(P(p * 100, v))

        prediction = Mock()
        prediction.declared_percentiles = percentiles
        # No CDF available - will trigger fallback

        report = Mock()
        report.question = question
        report.prediction = prediction

        score = calculate_numeric_baseline_score(report)

        # Should still get a reasonable score using fallback method
        assert score is not None
        assert isinstance(score, float)
        # With fixed normalization, should be in MC-like range
        assert -200 <= score <= 100

    def test_scoring_consistency_with_binary_mc(self):
        """Test that numeric scores are in similar range as binary/MC."""
        question = Mock()
        question.id_of_question = 997
        question.lower_bound = 0.0
        question.upper_bound = 100.0

        # Uniform community CDF
        question.api_json = {
            "question": {
                "aggregations": {
                    "recency_weighted": {"latest": {"forecast_values": np.linspace(0.0, 1.0, 11).tolist()}}
                }
            }
        }

        # Uniform bot CDF for comparison
        class P:
            def __init__(self, percentile):
                self.percentile = percentile

        uniform_bot_cdf = [P(i / 10.0) for i in range(11)]

        prediction = Mock()
        prediction.cdf = uniform_bot_cdf

        report = Mock()
        report.question = question
        report.prediction = prediction

        numeric_score = calculate_numeric_baseline_score(report)

        # Compare to binary scoring for similar "neutral" prediction
        # Binary scoring formula is 100.0 * (c * (log2(p) + 1.0) + (1.0 - c) * (log2(1.0 - p) + 1.0))
        c, p = 0.5, 0.5  # Both 50% - neutral
        binary_score = 100.0 * (c * (math.log2(p) + 1.0) + (1.0 - c) * (math.log2(1.0 - p) + 1.0))

        # Should be in similar range (both around 0 for neutral predictions)
        assert numeric_score is not None
        assert isinstance(numeric_score, float)

        # Both should be relatively close to each other for neutral predictions
        score_diff = abs(numeric_score - binary_score)
        assert score_diff < 150  # Should be within reasonable range of each other


class TestScoreScaling:
    """Test that scores are on similar scales across question types."""

    def test_score_scales_comparable(self):
        """Test that binary, MC, and numeric scores are on comparable scales."""
        # This is an integration test to verify score scaling

        # Mock binary score (known good scale from existing implementation)
        # Binary formula: 100.0 * (c * (log2(p) + 1.0) + (1.0 - c) * (log2(1.0 - p) + 1.0))
        c, p = 0.7, 0.6  # Community 70%, bot 60%
        binary_score = 100.0 * (c * (math.log2(p) + 1.0) + (1.0 - c) * (math.log2(1.0 - p) + 1.0))

        # Create MC and numeric scores using our functions
        mc_question = Mock()
        mc_question.id_of_question = 123
        mc_question.num_predictions = 15

        mc_option1 = Mock()
        mc_option1.option_name = "A"
        mc_option1.probability = 0.6

        mc_option2 = Mock()
        mc_option2.option_name = "B"
        mc_option2.probability = 0.4

        mc_prediction = Mock()
        mc_prediction.predicted_options = [mc_option1, mc_option2]

        mc_question.options = ["A", "B"]
        mc_question.api_json = {
            "question": {
                "type": "multiple_choice",
                "options": ["A", "B"],
                "aggregations": {"recency_weighted": {"latest": {"forecast_values": [0.6, 0.4]}}},
            }
        }

        mc_report = Mock()
        mc_report.question = mc_question
        mc_report.prediction = mc_prediction

        mc_score = calculate_multiple_choice_baseline_score(mc_report)

        # Create numeric score
        numeric_question = Mock()
        numeric_question.id_of_question = 456
        numeric_question.num_predictions = 20

        numeric_percentiles = []
        for p, v in [(10, 100), (20, 150), (40, 200), (60, 300), (80, 500), (90, 800)]:
            mock_p = Mock()
            mock_p.percentile = p
            mock_p.value = v
            numeric_percentiles.append(mock_p)

        numeric_prediction = Mock()
        numeric_prediction.declared_percentiles = numeric_percentiles

        numeric_question.api_json = {
            "question": {
                "aggregations": {
                    "recency_weighted": {"latest": {"forecast_values": np.linspace(0.0, 1.0, 201).tolist()}}
                }
            }
        }

        numeric_report = Mock()
        numeric_report.question = numeric_question
        numeric_report.prediction = numeric_prediction

        numeric_score = calculate_numeric_baseline_score(numeric_report)

        # ``binary_score`` is the hand-computed reference above, not production output —
        # only mc_score and numeric_score come from the functions under test. That matters:
        # the previous version of this block computed
        # ``max(abs(binary_score), abs(mc_score), abs(numeric_score)) > 0``, which the
        # 8.7546 constant satisfies unconditionally, and bounded every score by
        # ``-500 <= score <= 500`` — 12x wider than the +100 the MC formula can reach, so
        # unfalsifiable by any input. Both sat inside
        # ``if mc_score is not None and numeric_score is not None``, and numeric_score IS
        # None for these inputs, so the whole block never executed at all.
        #
        # Bounds match the sibling per-type tests (-200..100), which 4c1db3a re-tightened
        # from the same 5x-wide band after finding the missing ln(10) normalization.
        assert mc_score is not None, "MC baseline scoring returned no score for a complete report"
        assert math.isfinite(mc_score)
        assert -200 <= mc_score <= 100, f"MC score off the Metaculus-like scale: {mc_score}"

        # The scaling claim, actually asserted: the MC score must land within an order of
        # magnitude of the hand-computed binary reference for an equivalent-confidence
        # forecast. This is what the old "within factor of 10" comment described and never
        # checked. Both are single-digit positive for these inputs (8.75 vs 2.90).
        assert 0.1 <= mc_score / binary_score <= 10.0, (
            f"MC and binary scores are not within a factor of 10: {mc_score=} {binary_score=}"
        )

        # numeric_score is None here (these mock percentiles produce no community
        # comparison), so assert that explicitly rather than guarding on it: a change that
        # starts returning a value should fail loudly and get a real bound, not slip
        # through a None-check. Per-type numeric scoring is covered by
        # tests/test_numeric_scoring_sanity.py and the -200..100 sibling assertions above.
        assert numeric_score is None


class TestMonkeyPatching:
    """Test monkey patching functionality."""

    def test_apply_scoring_patches_installs_all_three_patches(self):
        """Each patch target actually carries our replacement afterwards.

        Asserting on the installed attributes rather than merely "didn't raise":
        every one of ``patch_multiple_choice_scoring`` / ``patch_numeric_scoring``
        / ``patch_error_handling`` swallows its own ImportError and Exception and
        only logs, so ``apply_scoring_patches`` returns cleanly even when all
        three silently no-op. A raises-nothing check therefore cannot fail and
        cannot distinguish "patched" from "every patch bailed".

        Identity of the replacement function is the assertion substrate rather
        than its behavior (covered by the calculate_* tests above): what this
        test uniquely guards is that the monkey-patch reached the class.
        """
        from forecasting_tools.data_models.forecast_report import ForecastReport
        from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
        from forecasting_tools.data_models.numeric_report import NumericReport

        apply_scoring_patches()

        mc_patch = MultipleChoiceReport.__dict__["expected_baseline_score"]
        assert isinstance(mc_patch, property)
        assert mc_patch.fget is not None
        assert mc_patch.fget.__name__ == "expected_baseline_score_mc"

        numeric_patch = NumericReport.__dict__["expected_baseline_score"]
        assert isinstance(numeric_patch, property)
        assert numeric_patch.fget is not None
        assert numeric_patch.fget.__name__ == "expected_baseline_score_numeric"

        avg_patch = ForecastReport.__dict__["calculate_average_expected_baseline_score"]
        assert isinstance(avg_patch, staticmethod)
        assert avg_patch.__func__.__name__ == "calculate_average_expected_baseline_score_fixed"


if __name__ == "__main__":
    pytest.main([__file__])
