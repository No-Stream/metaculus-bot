"""Tests for backtest scoring functions."""

from unittest.mock import MagicMock

import pytest

from metaculus_bot.backtest.scoring import (
    GroundTruth,
    QuestionScore,
    binary_log_score,
    brier_score,
    mc_log_score,
    numeric_crps,
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
