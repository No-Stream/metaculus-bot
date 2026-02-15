"""Tests for backtest analysis module: aggregation, reporting, and data export."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from metaculus_bot.backtest.analysis import BacktestResult
from metaculus_bot.backtest.scoring import GroundTruth, QuestionScore


class TestAggregateScores:
    def test_single_metric_single_score(self):
        from metaculus_bot.backtest.analysis import aggregate_scores

        scores = [QuestionScore(1, "binary", 0.25, 0.16, "brier")]
        result = aggregate_scores(scores)

        assert "brier" in result
        assert result["brier"]["bot_mean"] == pytest.approx(0.25)
        assert result["brier"]["bot_std"] == pytest.approx(0.0)
        assert result["brier"]["community_mean"] == pytest.approx(0.16)
        assert result["brier"]["bot_minus_community"] == pytest.approx(0.09)
        assert result["brier"]["n"] == 1

    def test_multiple_scores_same_metric(self):
        from metaculus_bot.backtest.analysis import aggregate_scores

        scores = [
            QuestionScore(1, "binary", 0.10, 0.05, "brier"),
            QuestionScore(2, "binary", 0.30, 0.15, "brier"),
            QuestionScore(3, "binary", 0.20, 0.10, "brier"),
        ]
        result = aggregate_scores(scores)

        assert result["brier"]["bot_mean"] == pytest.approx(0.20)
        expected_std = float(np.std([0.10, 0.30, 0.20]))
        assert result["brier"]["bot_std"] == pytest.approx(expected_std)
        assert result["brier"]["community_mean"] == pytest.approx(0.10)
        assert result["brier"]["bot_minus_community"] == pytest.approx(0.10)
        assert result["brier"]["n"] == 3

    def test_multiple_metrics(self):
        from metaculus_bot.backtest.analysis import aggregate_scores

        scores = [
            QuestionScore(1, "binary", 0.25, 0.16, "brier"),
            QuestionScore(1, "binary", 50.0, 60.0, "log_score"),
            QuestionScore(2, "binary", 0.15, 0.10, "brier"),
            QuestionScore(2, "binary", 70.0, 80.0, "log_score"),
        ]
        result = aggregate_scores(scores)

        assert set(result.keys()) == {"brier", "log_score"}
        assert result["brier"]["n"] == 2
        assert result["log_score"]["n"] == 2
        assert result["brier"]["bot_mean"] == pytest.approx(0.20)
        assert result["log_score"]["bot_mean"] == pytest.approx(60.0)

    def test_community_scores_with_none(self):
        from metaculus_bot.backtest.analysis import aggregate_scores

        scores = [
            QuestionScore(1, "numeric", 0.05, None, "crps"),
            QuestionScore(2, "numeric", 0.10, None, "crps"),
        ]
        result = aggregate_scores(scores)

        assert result["crps"]["community_mean"] is None
        assert result["crps"]["bot_minus_community"] is None

    def test_mixed_community_some_none(self):
        from metaculus_bot.backtest.analysis import aggregate_scores

        scores = [
            QuestionScore(1, "binary", 0.25, 0.16, "brier"),
            QuestionScore(2, "binary", 0.15, None, "brier"),
            QuestionScore(3, "binary", 0.20, 0.10, "brier"),
        ]
        result = aggregate_scores(scores)

        assert result["brier"]["community_mean"] == pytest.approx(0.13)
        assert result["brier"]["bot_mean"] == pytest.approx(0.20)

    def test_empty_scores(self):
        from metaculus_bot.backtest.analysis import aggregate_scores

        result = aggregate_scores([])
        assert result == {}


class TestBacktestResult:
    def test_dataclass_fields(self):
        scores = [QuestionScore(1, "binary", 0.25, 0.16, "brier")]
        result = BacktestResult(
            bot_name="test_bot",
            scores=scores,
            num_questions=10,
            num_scored=8,
            num_failed=2,
        )
        assert result.bot_name == "test_bot"
        assert result.num_questions == 10
        assert result.num_scored == 8
        assert result.num_failed == 2
        assert len(result.scores) == 1


class TestGenerateBacktestReport:
    def _make_result(self, bot_name: str = "test_bot") -> BacktestResult:
        scores = [
            QuestionScore(1, "binary", 0.25, 0.16, "brier"),
            QuestionScore(1, "binary", 50.0, 60.0, "log_score"),
            QuestionScore(2, "binary", 0.15, 0.10, "brier"),
            QuestionScore(2, "binary", 70.0, 80.0, "log_score"),
            QuestionScore(3, "numeric", 0.05, None, "crps"),
        ]
        return BacktestResult(
            bot_name=bot_name,
            scores=scores,
            num_questions=5,
            num_scored=3,
            num_failed=2,
        )

    def test_report_contains_header(self):
        from metaculus_bot.backtest.analysis import generate_backtest_report

        result = self._make_result()
        report = generate_backtest_report([result])

        assert "# Backtest Report" in report
        assert "Generated:" in report

    def test_report_contains_bot_name(self):
        from metaculus_bot.backtest.analysis import generate_backtest_report

        result = self._make_result(bot_name="my_fancy_bot")
        report = generate_backtest_report([result])

        assert "my_fancy_bot" in report

    def test_report_contains_score_table(self):
        from metaculus_bot.backtest.analysis import generate_backtest_report

        result = self._make_result()
        report = generate_backtest_report([result])

        assert "| Metric" in report
        assert "brier" in report.lower()
        assert "log_score" in report.lower()

    def test_report_contains_scored_info(self):
        from metaculus_bot.backtest.analysis import generate_backtest_report

        result = self._make_result()
        report = generate_backtest_report([result])

        assert "3/5" in report
        assert "failed: 2" in report

    def test_report_with_question_set_metadata(self):
        from metaculus_bot.backtest.analysis import generate_backtest_report

        result = self._make_result()
        question_set = Mock()
        question_set.fetch_metadata = {
            "type_distribution": {"BinaryQuestion": 3, "NumericQuestion": 2},
            "total_clean": 5,
        }
        report = generate_backtest_report([result], question_set=question_set)

        assert "BinaryQuestion" in report
        assert "NumericQuestion" in report

    def test_report_contains_per_type_breakdown(self):
        from metaculus_bot.backtest.analysis import generate_backtest_report

        result = self._make_result()
        report = generate_backtest_report([result])

        assert "Per-Type Breakdown" in report
        assert "binary" in report.lower()
        assert "numeric" in report.lower()

    def test_report_contains_notes_about_community(self):
        from metaculus_bot.backtest.analysis import generate_backtest_report

        result = self._make_result()
        report = generate_backtest_report([result])

        assert "community" in report.lower()
        assert "convergence" in report.lower()

    def test_report_writes_to_file(self, tmp_path):
        from metaculus_bot.backtest.analysis import generate_backtest_report

        result = self._make_result()
        output_file = str(tmp_path / "subdir" / "report.md")
        report = generate_backtest_report([result], output_path=output_file)

        written = Path(output_file).read_text()
        assert written == report
        assert "# Backtest Report" in written

    def test_report_multiple_bots(self):
        from metaculus_bot.backtest.analysis import generate_backtest_report

        result_a = self._make_result(bot_name="bot_alpha")
        result_b = self._make_result(bot_name="bot_beta")
        report = generate_backtest_report([result_a, result_b])

        assert "bot_alpha" in report
        assert "bot_beta" in report
        assert "2 bots evaluated" in report


class TestSaveBacktestData:
    def _make_ground_truths(self) -> dict[int, GroundTruth]:
        return {
            1: GroundTruth(
                question_id=1,
                question_type="binary",
                resolution=True,
                resolution_string="Yes",
                community_prediction=0.75,
                actual_resolution_time=datetime(2025, 6, 15, 12, 0, 0),
                question_text="Will X happen?",
                page_url="https://metaculus.com/questions/1",
            ),
            2: GroundTruth(
                question_id=2,
                question_type="numeric",
                resolution=42.5,
                resolution_string="42.5",
                community_prediction=[0.1, 0.3, 0.5, 0.7, 0.9],
                actual_resolution_time=None,
                question_text="What will Y be?",
            ),
        }

    def _make_result(self) -> BacktestResult:
        scores = [
            QuestionScore(1, "binary", 0.25, 0.16, "brier"),
            QuestionScore(2, "numeric", 0.05, None, "crps"),
        ]
        return BacktestResult(
            bot_name="test_bot",
            scores=scores,
            num_questions=3,
            num_scored=2,
            num_failed=1,
        )

    def test_saves_json_file(self, tmp_path):
        from metaculus_bot.backtest.analysis import save_backtest_data

        question_set = Mock()
        question_set.fetch_metadata = {"tournament": "test_tourney"}
        question_set.ground_truths = self._make_ground_truths()

        result = self._make_result()
        output_dir = str(tmp_path / "output")
        filepath = save_backtest_data(question_set, [result], output_dir)

        assert Path(filepath).exists()
        assert filepath.startswith(output_dir)
        assert filepath.endswith(".json")

    def test_json_structure(self, tmp_path):
        from metaculus_bot.backtest.analysis import save_backtest_data

        question_set = Mock()
        question_set.fetch_metadata = {"tournament": "test_tourney"}
        question_set.ground_truths = self._make_ground_truths()

        result = self._make_result()
        filepath = save_backtest_data(question_set, [result], str(tmp_path))

        data = json.loads(Path(filepath).read_text())

        assert "timestamp" in data
        assert "fetch_metadata" in data
        assert data["fetch_metadata"]["tournament"] == "test_tourney"
        assert "ground_truths" in data
        assert "results" in data

    def test_ground_truth_serialization(self, tmp_path):
        from metaculus_bot.backtest.analysis import save_backtest_data

        question_set = Mock()
        question_set.fetch_metadata = {}
        question_set.ground_truths = self._make_ground_truths()

        result = self._make_result()
        filepath = save_backtest_data(question_set, [result], str(tmp_path))

        data = json.loads(Path(filepath).read_text())

        gt1 = data["ground_truths"]["1"]
        assert gt1["question_id"] == 1
        assert gt1["question_type"] == "binary"
        assert gt1["resolution"] == "True"
        assert gt1["actual_resolution_time"] == "2025-06-15T12:00:00"

        gt2 = data["ground_truths"]["2"]
        assert gt2["actual_resolution_time"] is None
        assert gt2["community_prediction"] == [0.1, 0.3, 0.5, 0.7, 0.9]

    def test_results_serialization(self, tmp_path):
        from metaculus_bot.backtest.analysis import save_backtest_data

        question_set = Mock()
        question_set.fetch_metadata = {}
        question_set.ground_truths = {}

        result = self._make_result()
        filepath = save_backtest_data(question_set, [result], str(tmp_path))

        data = json.loads(Path(filepath).read_text())

        r = data["results"][0]
        assert r["bot_name"] == "test_bot"
        assert r["num_questions"] == 3
        assert r["num_scored"] == 2
        assert r["num_failed"] == 1
        assert len(r["scores"]) == 2
        assert "aggregated" in r

    def test_creates_output_dir(self, tmp_path):
        from metaculus_bot.backtest.analysis import save_backtest_data

        question_set = Mock()
        question_set.fetch_metadata = {}
        question_set.ground_truths = {}

        result = self._make_result()
        nested_dir = str(tmp_path / "a" / "b" / "c")
        filepath = save_backtest_data(question_set, [result], nested_dir)

        assert Path(filepath).exists()
        assert Path(nested_dir).is_dir()
