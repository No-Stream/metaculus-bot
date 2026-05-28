"""Unit tests for metaculus_bot.post_processing — extracted Platt and integer-snap logic."""

from __future__ import annotations

from copy import deepcopy
from unittest.mock import MagicMock

import numpy as np
import pytest
from forecasting_tools import BinaryQuestion, NumericQuestion
from forecasting_tools.data_models.multiple_choice_report import PredictedOption, PredictedOptionList
from forecasting_tools.data_models.numeric_report import NumericDistribution

from metaculus_bot.calibration import PlattParams, apply_binary_platt, apply_mc_platt
from metaculus_bot.constants import PLATT_BINARY_MAX_ABS_DEVIATION, PLATT_MC_MAX_ABS_DEVIATION
from metaculus_bot.post_processing import apply_platt_calibration, maybe_snap_to_integers

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binary_question(qid: int = 100) -> MagicMock:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = qid
    return q


def _mc_pred(probs: list[float]) -> PredictedOptionList:
    names = ["A", "B", "C", "D"][: len(probs)]
    return PredictedOptionList(
        predicted_options=[PredictedOption(option_name=n, probability=p) for n, p in zip(names, probs)]
    )


def _numeric_question(qid: int = 200) -> MagicMock:
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = qid
    q.lower_bound = 0.0
    q.upper_bound = 100.0
    q.open_lower_bound = True
    q.open_upper_bound = True
    q.cdf_size = 201
    return q


# ---------------------------------------------------------------------------
# apply_platt_calibration
# ---------------------------------------------------------------------------


class TestApplyPlattCalibration:
    def test_disabled_returns_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PLATT_CALIBRATION_ENABLED", raising=False)
        params = PlattParams(bias=0.5, slope=1.5)
        result = apply_platt_calibration(0.6, _binary_question(), params, params)
        assert result == 0.6

    def test_identity_params_returns_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
        identity = PlattParams.identity()
        result = apply_platt_calibration(0.6, _binary_question(), identity, identity)
        assert result == 0.6

    def test_binary_applies_platt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
        params = PlattParams(bias=0.5, slope=1.5)
        mc_params = PlattParams.identity()
        raw = 0.65
        expected = apply_binary_platt(raw, params, max_abs_deviation=PLATT_BINARY_MAX_ABS_DEVIATION)
        result = apply_platt_calibration(raw, _binary_question(), params, mc_params)
        assert result == expected
        assert result != raw

    def test_mc_applies_platt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
        binary_params = PlattParams.identity()
        mc_params = PlattParams(bias=0.3, slope=1.4)
        pred = _mc_pred([0.50, 0.30, 0.15, 0.05])
        ref = deepcopy(pred)
        apply_mc_platt(ref, mc_params, max_abs_deviation=PLATT_MC_MAX_ABS_DEVIATION)

        q = MagicMock()
        q.id_of_question = 999
        result = apply_platt_calibration(pred, q, binary_params, mc_params)
        assert isinstance(result, PredictedOptionList)
        for got, exp in zip(result.predicted_options, ref.predicted_options):
            assert got.probability == pytest.approx(exp.probability)

    def test_numeric_distribution_passes_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PLATT_CALIBRATION_ENABLED", "true")
        params = PlattParams(bias=0.5, slope=1.5)
        dist = MagicMock(spec=NumericDistribution)
        result = apply_platt_calibration(dist, _numeric_question(), params, params)
        assert result is dist


# ---------------------------------------------------------------------------
# maybe_snap_to_integers
# ---------------------------------------------------------------------------


class TestMaybeSnapToIntegers:
    def test_non_numeric_returns_unchanged(self) -> None:
        result = maybe_snap_to_integers(0.5, _binary_question(), [True, True, True])
        assert result == 0.5

    def test_no_votes_returns_unchanged(self) -> None:
        q = _numeric_question()
        dist = MagicMock(spec=NumericDistribution)
        result = maybe_snap_to_integers(dist, q, [])
        assert result is dist

    def test_majority_continuous_returns_unchanged(self) -> None:
        q = _numeric_question()
        dist = MagicMock(spec=NumericDistribution)
        result = maybe_snap_to_integers(dist, q, [False, False, True])
        assert result is dist

    def test_majority_discrete_snaps(self) -> None:
        q = _numeric_question()
        q.lower_bound = 0.0
        q.upper_bound = 10.0
        q.open_lower_bound = True
        q.open_upper_bound = True
        q.cdf_size = 201
        q.zero_point = None

        cdf_values = np.linspace(0.001, 0.999, 201).tolist()
        dist = MagicMock(spec=NumericDistribution)
        dist._pchip_cdf_values = cdf_values
        dist.declared_percentiles = []

        votes = [True, True, True, False, False]
        result = maybe_snap_to_integers(dist, q, votes)
        # Should return something different from the input (snapped)
        assert result is not dist
