"""Tests for spread_metrics module — measures forecaster disagreement."""

import math

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericDistribution, NumericQuestion
from forecasting_tools.data_models.multiple_choice_report import PredictedOption, PredictedOptionList
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.constants import (
    CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD,
    CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD,
    CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD,
)
from metaculus_bot.spread_metrics import (
    binary_log_odds_spread,
    binary_prob_range_spread,
    compute_spread,
    mc_max_option_spread,
    numeric_percentile_spread,
)

# ===========================================================================
# binary_prob_range_spread (the active trigger metric)
# ===========================================================================


class TestBinaryProbRangeSpread:
    def test_moderate_disagreement(self):
        assert binary_prob_range_spread([0.50, 0.68]) == pytest.approx(0.18)

    def test_tail_disagreement(self):
        assert binary_prob_range_spread([0.01, 0.19]) == pytest.approx(0.18)

    def test_all_same(self):
        assert binary_prob_range_spread([0.5, 0.5, 0.5]) == 0.0

    def test_six_model_ensemble(self):
        assert binary_prob_range_spread([0.10, 0.15, 0.25, 0.30, 0.40, 0.55]) == pytest.approx(0.45)

    def test_single_prediction_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            binary_prob_range_spread([0.5])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            binary_prob_range_spread([])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_binary_question(**overrides) -> BinaryQuestion:
    defaults = dict(
        question_text="Will it rain?",
        id_of_question=1,
        page_url="https://example.com/q/1",
        background_info="",
        resolution_criteria="",
        fine_print="",
    )
    defaults.update(overrides)
    return BinaryQuestion(**defaults)


def _make_mc_question(**overrides) -> MultipleChoiceQuestion:
    defaults = dict(
        question_text="What color?",
        options=["Red", "Blue", "Green"],
        id_of_question=2,
        page_url="https://example.com/q/2",
        background_info="",
        resolution_criteria="",
        fine_print="",
    )
    defaults.update(overrides)
    return MultipleChoiceQuestion(**defaults)


def _make_numeric_question(**overrides) -> NumericQuestion:
    defaults = dict(
        question_text="How many?",
        id_of_question=3,
        page_url="https://example.com/q/3",
        background_info="",
        resolution_criteria="",
        fine_print="",
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=False,
        open_upper_bound=False,
    )
    defaults.update(overrides)
    return NumericQuestion(**defaults)


STANDARD_PERCENTILE_VALUES = [2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5]


def _make_percentile_list(predicted_values: list[float]) -> list[Percentile]:
    """Build a standard 11-percentile list from predicted values at each percentile."""
    assert len(predicted_values) == len(STANDARD_PERCENTILE_VALUES)
    return [
        Percentile(percentile=pct / 100.0, value=val) for pct, val in zip(STANDARD_PERCENTILE_VALUES, predicted_values)
    ]


# ===========================================================================
# binary_log_odds_spread
# ===========================================================================


class TestBinaryLogOddsSpread:
    def test_moderate_disagreement(self):
        """50% vs 68% -- modest spread."""
        spread = binary_log_odds_spread([0.50, 0.68])
        expected = abs(math.log(0.68 / 0.32) - math.log(0.50 / 0.50))
        assert spread == pytest.approx(expected, abs=0.01)
        assert spread == pytest.approx(0.75, abs=0.05)

    def test_tail_disagreement(self):
        """1% vs 19% -- large spread despite similar absolute gap."""
        spread = binary_log_odds_spread([0.01, 0.19])
        assert spread == pytest.approx(3.15, abs=0.1)

    def test_upper_tail_disagreement(self):
        """80% vs 95% -- notable spread."""
        spread = binary_log_odds_spread([0.80, 0.95])
        expected = abs(math.log(0.95 / 0.05) - math.log(0.80 / 0.20))
        assert spread == pytest.approx(expected, abs=0.01)
        assert spread == pytest.approx(1.55, abs=0.2)

    def test_all_same(self):
        """All models agree -- spread is zero."""
        assert binary_log_odds_spread([0.5, 0.5, 0.5]) == 0.0

    def test_extreme_values_clamped(self):
        """Values at the boundary should not raise (clamping works)."""
        spread = binary_log_odds_spread([0.001, 0.999])
        assert spread > 0.0
        assert math.isfinite(spread)

    def test_single_prediction_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            binary_log_odds_spread([0.5])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            binary_log_odds_spread([])


# ===========================================================================
# mc_max_option_spread
# ===========================================================================


class TestMcMaxOptionSpread:
    def test_two_models_one_option_disagreement(self):
        """Two models, 20pp spread on one option."""
        pred1 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.50),
                PredictedOption(option_name="B", probability=0.50),
            ]
        )
        pred2 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.70),
                PredictedOption(option_name="B", probability=0.30),
            ]
        )
        spread = mc_max_option_spread([pred1, pred2])
        assert spread == pytest.approx(0.20, abs=0.001)

    def test_three_models_agreement(self):
        """All three models agree -- spread near zero."""
        pred = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.50),
                PredictedOption(option_name="B", probability=0.50),
            ]
        )
        spread = mc_max_option_spread([pred, pred, pred])
        assert spread == pytest.approx(0.0, abs=0.001)

    def test_disagreement_on_one_option_only(self):
        """Disagreement only on option C; A and B agree."""
        pred1 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.40),
                PredictedOption(option_name="B", probability=0.40),
                PredictedOption(option_name="C", probability=0.20),
            ]
        )
        pred2 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.40),
                PredictedOption(option_name="B", probability=0.25),
                PredictedOption(option_name="C", probability=0.35),
            ]
        )
        spread = mc_max_option_spread([pred1, pred2])
        # max option spread: A=0, B=0.15, C=0.15 -> 0.15
        assert spread == pytest.approx(0.15, abs=0.001)

    def test_single_prediction_raises(self):
        pred = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.50),
                PredictedOption(option_name="B", probability=0.50),
            ]
        )
        with pytest.raises(ValueError, match="at least 2"):
            mc_max_option_spread([pred])

    def test_mismatched_options_raises(self):
        """Predictions with different option sets should raise ValueError."""
        pred1 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.50),
                PredictedOption(option_name="B", probability=0.50),
            ]
        )
        pred2 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="A", probability=0.60),
                PredictedOption(option_name="C", probability=0.40),
            ]
        )
        with pytest.raises(ValueError, match="mc_max_option_spread"):
            mc_max_option_spread([pred1, pred2])


# ===========================================================================
# numeric_percentile_spread
# ===========================================================================


class TestNumericPercentileSpread:
    def test_two_models_closed_bounds(self):
        """Two models with different medians on [0, 100]."""
        # Model 1: centered at 40
        model1 = _make_percentile_list([10, 15, 20, 25, 35, 40, 45, 55, 60, 65, 70])
        # Model 2: centered at 60
        model2 = _make_percentile_list([30, 35, 40, 45, 55, 60, 65, 75, 80, 85, 90])
        question = _make_numeric_question()

        spread = numeric_percentile_spread([model1, model2], question)
        # At index 2 (10th pct): |40-20|/100 = 0.20
        # At index 5 (50th pct): |60-40|/100 = 0.20
        # At index 8 (90th pct): |80-60|/100 = 0.20
        assert spread == pytest.approx(0.20, abs=0.01)

    def test_open_ended_bounds_uses_iqr_fallback(self):
        """Open-ended lower bound -- falls back to IQR denominator."""
        model1 = _make_percentile_list([10, 15, 20, 25, 35, 40, 45, 55, 60, 65, 70])
        model2 = _make_percentile_list([30, 35, 40, 45, 55, 60, 65, 75, 80, 85, 90])
        question = _make_numeric_question(open_lower_bound=True, open_upper_bound=True)

        spread = numeric_percentile_spread([model1, model2], question)
        # p90 values (index 8): model1=60, model2=80 -> median=70
        # p10 values (index 2): model1=20, model2=40 -> median=30
        # IQR denominator = 70 - 30 = 40
        # raw spread at all key pcts = 20; normalized = 20/40 = 0.5
        assert spread == pytest.approx(0.5, abs=0.02)

    def test_all_models_agree(self):
        """All models produce same percentiles -- spread is zero."""
        model = _make_percentile_list([10, 15, 20, 25, 35, 40, 45, 55, 60, 65, 70])
        question = _make_numeric_question()

        spread = numeric_percentile_spread([model, model, model], question)
        assert spread == pytest.approx(0.0, abs=0.001)

    def test_single_prediction_raises(self):
        model = _make_percentile_list([10, 15, 20, 25, 35, 40, 45, 55, 60, 65, 70])
        question = _make_numeric_question()

        with pytest.raises(ValueError, match="at least 2"):
            numeric_percentile_spread([model], question)

    def test_short_percentile_list_raises(self):
        """Percentile lists with fewer than 9 elements should raise ValueError."""
        short_model = _make_percentile_list([10, 15, 20, 25, 35, 40, 45, 55, 60, 65, 70])[:5]
        full_model = _make_percentile_list([10, 15, 20, 25, 35, 40, 45, 55, 60, 65, 70])
        question = _make_numeric_question()

        with pytest.raises(ValueError, match="at least 9 percentiles"):
            numeric_percentile_spread([short_model, full_model], question)


# ===========================================================================
# compute_spread (dispatcher)
# ===========================================================================


class TestComputeSpread:
    def test_binary_dispatch(self):
        question = _make_binary_question()
        spread = compute_spread(question, [0.50, 0.68])
        assert spread == pytest.approx(binary_prob_range_spread([0.50, 0.68]))

    def test_binary_dispatch_uses_prob_range_not_log_odds(self):
        # Regression: dispatcher must return prob-range, not log-odds.
        question = _make_binary_question()
        spread = compute_spread(question, [0.01, 0.19])
        # prob-range = 0.18; log-odds ≈ 3.15 — these must differ noticeably
        assert spread == pytest.approx(0.18, abs=0.01)
        assert spread != pytest.approx(binary_log_odds_spread([0.01, 0.19]), abs=0.5)

    def test_mc_dispatch(self):
        question = _make_mc_question()
        pred1 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Red", probability=0.50),
                PredictedOption(option_name="Blue", probability=0.30),
                PredictedOption(option_name="Green", probability=0.20),
            ]
        )
        pred2 = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Red", probability=0.70),
                PredictedOption(option_name="Blue", probability=0.20),
                PredictedOption(option_name="Green", probability=0.10),
            ]
        )
        spread = compute_spread(question, [pred1, pred2])
        assert spread == pytest.approx(mc_max_option_spread([pred1, pred2]))

    def test_numeric_dispatch(self):
        question = _make_numeric_question()
        pcts1 = _make_percentile_list([10, 15, 20, 25, 35, 40, 45, 55, 60, 65, 70])
        pcts2 = _make_percentile_list([30, 35, 40, 45, 55, 60, 65, 75, 80, 85, 90])
        dist_args = {
            "lower_bound": question.lower_bound,
            "upper_bound": question.upper_bound,
            "open_lower_bound": question.open_lower_bound,
            "open_upper_bound": question.open_upper_bound,
            "zero_point": question.zero_point,
        }
        dist1 = NumericDistribution(declared_percentiles=pcts1, **dist_args)
        dist2 = NumericDistribution(declared_percentiles=pcts2, **dist_args)
        spread = compute_spread(question, [dist1, dist2])
        assert spread == pytest.approx(numeric_percentile_spread([pcts1, pcts2], question))

    def test_unknown_type_raises(self):
        from unittest.mock import Mock

        unknown_question = Mock()
        with pytest.raises(ValueError, match="Unsupported question type"):
            compute_spread(unknown_question, [0.5, 0.5])


# ===========================================================================
# Constants are exported
# ===========================================================================


class TestConstants:
    def test_threshold_values(self):
        assert CONDITIONAL_STACKING_BINARY_PROB_RANGE_THRESHOLD == 0.15
        assert CONDITIONAL_STACKING_MC_MAX_OPTION_THRESHOLD == 0.20
        assert CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD == 0.15
