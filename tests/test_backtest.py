"""Smoke tests for backtest.py CLI entry point."""

import argparse
import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest


def test_import_backtest():
    """Module can be imported without errors."""
    import backtest  # noqa: F401


def test_cli_argument_parsing():
    """CLI argument parsing with different combinations."""
    from backtest import _build_parser

    parser = _build_parser()

    args = parser.parse_args([])
    assert args.num_questions == 20
    assert args.include_models is None
    assert args.exclude_models is None

    args = parser.parse_args(["--num-questions", "50"])
    assert args.num_questions == 50

    args = parser.parse_args(["--include-models", "gpt", "qwen"])
    assert args.include_models == ["gpt", "qwen"]

    args = parser.parse_args(["--exclude-models", "grok"])
    assert args.exclude_models == ["grok"]

    args = parser.parse_args(["--resolved-after", "2025-06-01"])
    assert args.resolved_after == "2025-06-01"

    args = parser.parse_args(["--tournament", "my-tournament"])
    assert args.tournament == "my-tournament"


def test_cli_defaults_use_constants():
    """Default CLI values come from constants module."""
    from backtest import _build_parser
    from metaculus_bot.constants import BACKTEST_DEFAULT_RESOLVED_AFTER, BACKTEST_DEFAULT_TOURNAMENT

    parser = _build_parser()
    args = parser.parse_args([])
    assert args.resolved_after == BACKTEST_DEFAULT_RESOLVED_AFTER
    assert args.tournament == BACKTEST_DEFAULT_TOURNAMENT


def test_model_filtering_include():
    """Include filter keeps only matching bots."""
    from backtest import _filter_bots

    bot_a = Mock()
    bot_a.name = "gpt-5.1"
    bot_b = Mock()
    bot_b.name = "qwen3-235b"
    bot_c = Mock()
    bot_c.name = "deepseek-3.2"

    filtered = _filter_bots([bot_a, bot_b, bot_c], include_models=["gpt", "qwen"], exclude_models=None)
    assert len(filtered) == 2
    assert filtered[0].name == "gpt-5.1"
    assert filtered[1].name == "qwen3-235b"


def test_model_filtering_exclude():
    """Exclude filter removes matching bots."""
    from backtest import _filter_bots

    bot_a = Mock()
    bot_a.name = "gpt-5.1"
    bot_b = Mock()
    bot_b.name = "qwen3-235b"
    bot_c = Mock()
    bot_c.name = "deepseek-3.2"

    filtered = _filter_bots([bot_a, bot_b, bot_c], include_models=None, exclude_models=["deepseek"])
    assert len(filtered) == 2
    assert filtered[0].name == "gpt-5.1"
    assert filtered[1].name == "qwen3-235b"


def test_model_filtering_no_filters():
    """No filters returns all bots."""
    from backtest import _filter_bots

    bot_a = Mock()
    bot_a.name = "gpt-5.1"

    filtered = _filter_bots([bot_a], include_models=None, exclude_models=None)
    assert len(filtered) == 1


def test_model_filtering_empty_result_raises():
    """Filtering out all bots raises ValueError."""
    from backtest import _filter_bots

    bot_a = Mock()
    bot_a.name = "gpt-5.1"

    with pytest.raises(ValueError, match="No bots remaining"):
        _filter_bots([bot_a], include_models=["nonexistent"], exclude_models=None)


@patch("backtest.save_backtest_data")
@patch("backtest.generate_backtest_report")
@patch("backtest.score_report")
@patch("backtest.Benchmarker")
@patch("backtest.MonetaryCostManager")
@patch("backtest.typeguard.check_type", side_effect=lambda val, _type: val)
@patch("backtest.create_individual_bots")
@patch("backtest.apply_scoring_patches")
@patch("backtest.screen_research_for_leakage")
@patch("backtest.fetch_resolved_questions")
def test_run_backtest_full_flow(
    mock_fetch,
    mock_leakage,
    mock_patches,
    mock_create_bots,
    mock_check_type,
    mock_cost_manager,
    mock_benchmarker_class,
    mock_score_report,
    mock_gen_report,
    mock_save_data,
):
    """Full backtest flow with all external calls mocked."""
    import backtest
    from metaculus_bot.backtest.question_prep import BacktestQuestionSet
    from metaculus_bot.backtest.scoring import GroundTruth, QuestionScore

    mock_question = Mock()
    mock_question.id_of_question = 123
    mock_question.question_text = "Will X happen?"

    mock_gt = GroundTruth(
        question_id=123,
        question_type="binary",
        resolution=True,
        resolution_string="Yes",
        community_prediction=0.7,
        actual_resolution_time=None,
        question_text="Will X happen?",
    )

    question_set = BacktestQuestionSet(
        questions=[mock_question],
        ground_truths={123: mock_gt},
    )
    mock_fetch.return_value = question_set

    mock_leakage.return_value = ([mock_question], {123: mock_gt}, {})

    mock_bot = Mock()
    mock_bot.name = "test-bot"
    mock_create_bots.return_value = [mock_bot]

    mock_cost_mgr_instance = Mock()
    mock_cost_mgr_instance.__enter__ = Mock(return_value=mock_cost_mgr_instance)
    mock_cost_mgr_instance.__exit__ = Mock(return_value=None)
    mock_cost_mgr_instance.current_usage = "$0.01"
    mock_cost_manager.return_value = mock_cost_mgr_instance

    mock_benchmark_result = Mock()
    mock_benchmark_result.name = "test-bot"
    mock_report = Mock()
    mock_report.question = mock_question
    mock_report.question.id_of_question = 123
    mock_benchmark_result.forecast_reports = [mock_report]

    mock_benchmarker = Mock()
    mock_benchmarker.run_benchmark = AsyncMock(return_value=[mock_benchmark_result])
    mock_benchmarker_class.return_value = mock_benchmarker

    mock_score_report.return_value = [
        QuestionScore(
            question_id=123, question_type="binary", bot_score=0.09, community_score=0.09, metric_name="brier"
        )
    ]

    mock_gen_report.return_value = "# Backtest Report\n..."
    mock_save_data.return_value = "backtests/data.json"

    args = argparse.Namespace(
        num_questions=1,
        resolved_after="2025-12-01",
        tournament="fall-aib-2025",
        include_models=None,
        exclude_models=None,
    )

    asyncio.run(backtest.run_backtest(args))

    mock_fetch.assert_called_once()
    mock_leakage.assert_called_once()
    mock_create_bots.assert_called_once()
    mock_patches.assert_called_once()
    mock_benchmarker.run_benchmark.assert_called_once()
    mock_score_report.assert_called_once()
    mock_gen_report.assert_called_once()
    mock_save_data.assert_called_once()
