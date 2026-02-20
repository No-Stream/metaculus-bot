"""Tests for backtest scoring functions."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from metaculus_bot.backtest.scoring import (
    GroundTruth,
    QuestionScore,
    binary_log_score,
    brier_score,
    mc_log_score,
    numeric_crps,
    numeric_log_score,
    score_report,
)

# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------


def test_brier_perfect_prediction_true():
    assert brier_score(1.0, True) == pytest.approx(0.0, abs=1e-3)


def test_brier_perfect_prediction_false():
    assert brier_score(0.0, False) == pytest.approx(0.0, abs=1e-3)


def test_brier_worst_prediction():
    assert brier_score(0.0, True) == pytest.approx(1.0, abs=1e-3)


def test_brier_fifty_fifty():
    assert brier_score(0.5, True) == 0.25


# ---------------------------------------------------------------------------
# Binary log score
# ---------------------------------------------------------------------------


def test_log_score_fifty_fifty():
    assert binary_log_score(0.5, True) == pytest.approx(0.0, abs=0.1)


def test_log_score_perfect_true():
    assert binary_log_score(0.99, True) > 0


def test_log_score_bad_prediction():
    assert binary_log_score(0.01, True) < 0


def test_log_score_symmetry():
    assert binary_log_score(0.9, True) == pytest.approx(binary_log_score(0.1, False), abs=0.1)


# ---------------------------------------------------------------------------
# Numeric CRPS
# ---------------------------------------------------------------------------


def test_crps_perfect_step_function():
    x_values = [0, 4.99, 5.0, 5.01, 10]
    cdf_values = [0, 0, 0.5, 1, 1]
    crps = numeric_crps(x_values, cdf_values, resolution=5.0)
    # Near-perfect CDF concentrated at resolution — CRPS should be small
    assert crps < 0.5


def test_crps_uniform_cdf():
    x_values = list(range(11))
    cdf_values = [i / 10 for i in range(11)]
    crps = numeric_crps(x_values, cdf_values, resolution=5.0)
    assert 0 < crps < 1


def test_crps_wrong_side():
    x_values = list(range(11))
    wrong_cdf = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    crps_wrong = numeric_crps(x_values, wrong_cdf, resolution=0.0)

    uniform_cdf = [i / 10 for i in range(11)]
    crps_uniform = numeric_crps(x_values, uniform_cdf, resolution=0.0)

    assert crps_wrong > crps_uniform


def test_crps_invalid_inputs():
    with pytest.raises(ValueError):
        numeric_crps([1.0], [0.5], resolution=1.0)

    with pytest.raises(ValueError):
        numeric_crps([1.0, 2.0, 3.0], [0.0, 1.0], resolution=2.0)


# ---------------------------------------------------------------------------
# Numeric log score (Metaculus PMF-bucket)
# ---------------------------------------------------------------------------


def test_numeric_log_score_uniform_cdf_scores_zero():
    """Uniform CDF → each interior PMF bin = 1/N = baseline → score ≈ 0."""
    n = 201
    cdf = [i / (n - 1) for i in range(n)]  # uniform: 0.0, 0.005, 0.01, ..., 1.0
    score = numeric_log_score(
        cdf, resolution=5.0, lower_bound=0.0, upper_bound=10.0, open_lower_bound=False, open_upper_bound=False
    )
    assert score == pytest.approx(0.0, abs=0.5)


def test_numeric_log_score_concentrated_positive():
    """CDF with steep transition at resolution → high PMF in that bucket → positive score."""
    x = np.linspace(0.0, 10.0, 201)
    cdf = (1.0 / (1.0 + np.exp(-(x - 5.0) / 0.3))).tolist()
    cdf[0] = 0.0
    cdf[-1] = 1.0
    score = numeric_log_score(
        cdf, resolution=5.0, lower_bound=0.0, upper_bound=10.0, open_lower_bound=False, open_upper_bound=False
    )
    assert score > 0


def test_numeric_log_score_far_from_resolution_negative():
    """CDF concentrated far from resolution → low PMF in resolution bucket → negative score."""
    x = np.linspace(0.0, 10.0, 201)
    cdf = (1.0 / (1.0 + np.exp(-(x - 2.0) / 0.3))).tolist()
    cdf[0] = 0.0
    cdf[-1] = 1.0
    score = numeric_log_score(
        cdf, resolution=8.0, lower_bound=0.0, upper_bound=10.0, open_lower_bound=False, open_upper_bound=False
    )
    assert score < 0


def test_numeric_log_score_open_bounds_different_baseline():
    """Open bounds lower the interior baseline → same CDF scores higher for interior resolutions."""
    n = 201
    x = np.linspace(0.0, 10.0, n)
    cdf = (1.0 / (1.0 + np.exp(-(x - 5.0) / 2.0))).tolist()
    cdf[0] = 0.001
    cdf[-1] = 0.999

    score_closed = numeric_log_score(
        cdf, resolution=5.0, lower_bound=0.0, upper_bound=10.0, open_lower_bound=False, open_upper_bound=False
    )
    score_open = numeric_log_score(
        cdf, resolution=5.0, lower_bound=0.0, upper_bound=10.0, open_lower_bound=True, open_upper_bound=True
    )
    # Open bounds → lower baseline → higher score for same PMF
    assert score_open > score_closed


def test_numeric_log_score_better_prediction_scores_higher():
    """A CDF centered closer to resolution should score higher than one farther away."""
    x = np.linspace(0.0, 10.0, 201)

    cdf_close = (1.0 / (1.0 + np.exp(-(x - 7.0) / 1.0))).tolist()
    cdf_close[0] = 0.0
    cdf_close[-1] = 1.0

    cdf_far = (1.0 / (1.0 + np.exp(-(x - 3.0) / 1.0))).tolist()
    cdf_far[0] = 0.0
    cdf_far[-1] = 1.0

    score_close = numeric_log_score(
        cdf_close, resolution=7.0, lower_bound=0.0, upper_bound=10.0, open_lower_bound=False, open_upper_bound=False
    )
    score_far = numeric_log_score(
        cdf_far, resolution=7.0, lower_bound=0.0, upper_bound=10.0, open_lower_bound=False, open_upper_bound=False
    )
    assert score_close > score_far


# ---------------------------------------------------------------------------
# MC log score
# ---------------------------------------------------------------------------


def test_mc_log_score_uniform():
    assert mc_log_score([0.25, 0.25, 0.25, 0.25], 0) == pytest.approx(0.0, abs=0.1)


def test_mc_log_score_perfect():
    score = mc_log_score([1.0, 0.0, 0.0], 0)
    assert score == pytest.approx(100.0, abs=2.0)


def test_mc_log_score_wrong():
    assert mc_log_score([0.0, 0.0, 1.0], 0) < -50


def test_mc_log_score_invalid_k():
    with pytest.raises(ValueError):
        mc_log_score([1.0], 0)


# ---------------------------------------------------------------------------
# score_report dispatcher
# ---------------------------------------------------------------------------


def test_score_report_binary():
    from forecasting_tools.data_models.binary_report import BinaryReport

    mock_report = MagicMock(spec=BinaryReport)
    mock_report.prediction = 0.7

    gt = GroundTruth(
        question_id=1,
        question_type="binary",
        resolution=True,
        resolution_string="yes",
        community_prediction=0.6,
        actual_resolution_time=None,
        question_text="Test?",
    )

    scores = score_report(mock_report, gt)
    assert len(scores) == 2
    metric_names = {s.metric_name for s in scores}
    assert metric_names == {"brier", "log_score"}
    for s in scores:
        assert isinstance(s, QuestionScore)
        assert s.question_id == 1
        assert s.question_type == "binary"


def test_score_report_numeric():
    from forecasting_tools.data_models.numeric_report import NumericReport

    mock_report = MagicMock(spec=NumericReport)

    # Build mock CDF points — 201 uniform CDF values
    n = 201
    mock_cdf_points = []
    for i in range(n):
        point = MagicMock()
        point.value = float(i) / (n - 1) * 10.0
        point.percentile = float(i) / (n - 1)
        mock_cdf_points.append(point)

    mock_prediction = MagicMock()
    mock_prediction.cdf = mock_cdf_points
    mock_report.prediction = mock_prediction

    mock_question = MagicMock()
    mock_question.lower_bound = 0.0
    mock_question.upper_bound = 10.0
    mock_question.open_lower_bound = False
    mock_question.open_upper_bound = False
    mock_question.zero_point = None
    mock_report.question = mock_question

    gt = GroundTruth(
        question_id=10,
        question_type="numeric",
        resolution=5.0,
        resolution_string="5.0",
        community_prediction=None,
        actual_resolution_time=None,
        question_text="How many?",
    )

    scores = score_report(mock_report, gt)
    assert len(scores) == 1
    assert scores[0].metric_name == "numeric_log_score"
    assert scores[0].question_type == "numeric"
    # Uniform CDF → score ≈ 0
    assert abs(scores[0].bot_score) < 1.0


def test_score_report_returns_empty_for_canceled():
    from forecasting_tools.data_models.binary_report import BinaryReport

    mock_report = MagicMock(spec=BinaryReport)
    mock_report.prediction = 0.7

    gt = GroundTruth(
        question_id=2,
        question_type="binary",
        resolution="annulled",
        resolution_string="annulled",
        community_prediction=0.5,
        actual_resolution_time=None,
        question_text="Canceled?",
    )

    scores = score_report(mock_report, gt)
    assert scores == []
