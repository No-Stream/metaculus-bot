"""Tests for backtest question preparation."""

import pytest
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    QuestionState,
)

from metaculus_bot.backtest.question_prep import _extract_ground_truth, _prepare_question_for_backtest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binary_question_yes() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will it rain tomorrow?",
        id_of_question=1,
        resolution_string="yes",
        community_prediction_at_access_time=0.7,
        state=QuestionState.RESOLVED,
        api_json={},
    )


@pytest.fixture
def binary_question_no() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will it snow?",
        id_of_question=2,
        resolution_string="no",
        community_prediction_at_access_time=0.3,
        state=QuestionState.RESOLVED,
        api_json={},
    )


@pytest.fixture
def numeric_question() -> NumericQuestion:
    return NumericQuestion(
        question_text="How many widgets?",
        id_of_question=3,
        resolution_string="42.5",
        state=QuestionState.RESOLVED,
        lower_bound=0.0,
        upper_bound=100.0,
        open_upper_bound=False,
        open_lower_bound=False,
        api_json={},
    )


@pytest.fixture
def mc_question() -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which option?",
        id_of_question=4,
        resolution_string="Option A",
        state=QuestionState.RESOLVED,
        options=["Option A", "Option B", "Option C"],
        api_json={},
    )


@pytest.fixture
def canceled_question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will this be annulled?",
        id_of_question=5,
        resolution_string="annulled",
        community_prediction_at_access_time=0.5,
        state=QuestionState.RESOLVED,
        api_json={},
    )


# ---------------------------------------------------------------------------
# _extract_ground_truth tests
# ---------------------------------------------------------------------------


def test_extract_binary_true(binary_question_yes: BinaryQuestion):
    gt = _extract_ground_truth(binary_question_yes)
    assert gt is not None
    assert gt.resolution is True
    assert gt.question_type == "binary"
    assert gt.question_id == 1


def test_extract_binary_false(binary_question_no: BinaryQuestion):
    gt = _extract_ground_truth(binary_question_no)
    assert gt is not None
    assert gt.resolution is False
    assert gt.question_type == "binary"


def test_extract_numeric(numeric_question: NumericQuestion):
    gt = _extract_ground_truth(numeric_question)
    assert gt is not None
    assert gt.resolution == 42.5
    assert gt.question_type == "numeric"


def test_extract_mc(mc_question: MultipleChoiceQuestion):
    gt = _extract_ground_truth(mc_question)
    assert gt is not None
    assert gt.resolution == "Option A"
    assert gt.question_type == "multiple_choice"


def test_extract_canceled_returns_none(canceled_question: BinaryQuestion):
    gt = _extract_ground_truth(canceled_question)
    assert gt is None


# ---------------------------------------------------------------------------
# _prepare_question_for_backtest tests
# ---------------------------------------------------------------------------


def test_prepare_sets_state_open(binary_question_yes: BinaryQuestion):
    clean = _prepare_question_for_backtest(binary_question_yes)
    assert clean.state == QuestionState.OPEN


def test_prepare_clears_resolution(binary_question_yes: BinaryQuestion):
    clean = _prepare_question_for_backtest(binary_question_yes)
    assert clean.resolution_string is None
    assert clean.actual_resolution_time is None


def test_prepare_clears_background_info(binary_question_yes: BinaryQuestion):
    binary_question_yes.background_info = "Some background info"
    clean = _prepare_question_for_backtest(binary_question_yes)
    assert clean.background_info is None


def test_prepare_is_deep_copy(binary_question_yes: BinaryQuestion):
    clean = _prepare_question_for_backtest(binary_question_yes)
    clean.question_text = "MODIFIED"
    assert binary_question_yes.question_text == "Will it rain tomorrow?"
