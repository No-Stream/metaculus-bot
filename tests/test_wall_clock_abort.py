"""Tests for the per-question wall-clock abort + publish hardening (task #13).

The bot uses `PER_QUESTION_WALL_CLOCK_DEADLINE` (3510s) to bound a single
question's research + fan-out + aggregation + publish budget. At deadline,
in-flight forecasters get cancelled, completed predictions ride a base-combine
median, and stacking is skipped if remaining budget falls below
`WALL_CLOCK_STACKING_MIN_BUDGET`.

Publish hardening is a separate concern: the four `MetaculusApi.post_*`
classmethods are monkey-patched to add `PUBLISH_POST_TIMEOUT` (20s) per call
plus `PUBLISH_POST_RETRIES` (1) on timeout / connection errors. See
`metaculus_bot/publish_hardening.py`.
"""

import asyncio
import concurrent.futures
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests
from forecasting_tools import BinaryQuestion, GeneralLlm, ReasonedPrediction

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy


def _stub_open_time() -> datetime:
    return datetime.now() - timedelta(days=30)


def _stub_resolve_time() -> datetime:
    return datetime.now() + timedelta(days=365)


@pytest.fixture
def mock_general_llm() -> MagicMock:
    mock_llm = MagicMock(spec=GeneralLlm)
    mock_llm.model = "mock_model"
    mock_llm.invoke = AsyncMock(return_value="mock reasoning")
    return mock_llm


@pytest.fixture
def mock_binary_question() -> MagicMock:
    question = MagicMock(spec=BinaryQuestion)
    question.page_url = "http://example.com/q"
    question.question_text = "Will X happen?"
    question.background_info = ""
    question.resolution_criteria = ""
    question.fine_print = ""
    question.unit_of_measure = ""
    question.id_of_question = 999
    question.open_time = _stub_open_time()
    question.scheduled_resolution_time = _stub_resolve_time()
    return question


def _make_bot(mock_llm: MagicMock, *, n_forecasters: int = 4, **kwargs) -> TemplateForecaster:
    llms_config = {
        "forecasters": [mock_llm] * n_forecasters,
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    return TemplateForecaster(llms=llms_config, **kwargs)


# ---------------------------------------------------------------------------
# Wall-clock fan-out abort
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_abort_with_three_or_more_forecasters_publishes(monkeypatch, mock_binary_question, mock_general_llm):
    """Wall-clock cap fires; at least 3 forecasters returned in time → publish path."""
    monkeypatch.setattr("main.PER_QUESTION_WALL_CLOCK_DEADLINE", 0.2)
    monkeypatch.setattr("main.WALL_CLOCK_STACKING_MIN_BUDGET", 0.0)

    bot = _make_bot(mock_general_llm, n_forecasters=4, min_forecasters_to_publish=3)
    bot._get_notepad = AsyncMock(return_value=MagicMock(total_research_reports_attempted=0))
    bot.run_research = AsyncMock(return_value="research")

    # 3 fast, 1 slow (will be cancelled).
    call_count = {"n": 0}

    async def mixed(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 3:
            return ReasonedPrediction(prediction_value=0.5, reasoning="ok")
        await asyncio.sleep(10)
        return ReasonedPrediction(prediction_value=0.5, reasoning="never")

    bot._forecaster_with_soft_deadline = mixed

    # The min-forecasters guard would normally pass with 3/4; the abort then
    # carries through to a non-stacking path (MEAN). Since the bot was
    # initialized with default (MEAN), aggregation uses the parent class
    # behavior and we just verify that publication doesn't fail.
    result = await bot._research_and_make_predictions(mock_binary_question)
    assert result is not None
    assert len(result.predictions) >= 1
    # 1 forecaster was cancelled.
    assert bot._forecasters_dropped_count == 1


@pytest.mark.asyncio
async def test_abort_with_fewer_than_min_forecasters_skips_publish(monkeypatch, mock_binary_question, mock_general_llm):
    """Wall-clock cap fires with only 1/4 forecasters returned → guard raises, counter bumps."""
    monkeypatch.setattr("main.PER_QUESTION_WALL_CLOCK_DEADLINE", 0.1)

    bot = _make_bot(mock_general_llm, n_forecasters=4, min_forecasters_to_publish=3)
    bot._get_notepad = AsyncMock(return_value=MagicMock(total_research_reports_attempted=0))
    bot.run_research = AsyncMock(return_value="research")

    call_count = {"n": 0}

    async def mixed(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ReasonedPrediction(prediction_value=0.5, reasoning="ok")
        await asyncio.sleep(10)
        return ReasonedPrediction(prediction_value=0.5, reasoning="never")

    bot._forecaster_with_soft_deadline = mixed

    assert bot._questions_failed_to_publish == 0
    with pytest.raises((RuntimeError, ExceptionGroup)):  # noqa: F821  # 3.11+ builtin
        await bot._research_and_make_predictions(mock_binary_question)
    assert bot._questions_failed_to_publish == 1
    # 3 forecasters were cancelled.
    assert bot._forecasters_dropped_count == 3


@pytest.mark.asyncio
async def test_tight_budget_skips_stacking_forces_fallback_median(monkeypatch, mock_binary_question, mock_general_llm):
    """Even with all forecasters returning, sub-90s budget forces fallback_median."""
    monkeypatch.setattr("main.PER_QUESTION_WALL_CLOCK_DEADLINE", 0.5)
    monkeypatch.setattr("main.WALL_CLOCK_STACKING_MIN_BUDGET", 1_000_000)  # always trips

    llms_config = {
        "forecasters": [mock_general_llm] * 3,
        "stacker": mock_general_llm,
        "analyzer": mock_general_llm,
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(
        llms=llms_config,
        min_forecasters_to_publish=2,
        aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
    )
    bot._get_notepad = AsyncMock(return_value=MagicMock(total_research_reports_attempted=0))
    bot.run_research = AsyncMock(return_value="research")

    async def fast(*args, **kwargs):
        return ReasonedPrediction(prediction_value=0.6, reasoning="ok")

    bot._forecaster_with_soft_deadline = fast

    # _aggregate_predictions invocation in non-stacking path should set
    # _stacker_outcome to "fallback_median" via our gate. Our skip path sets
    # outcome=fallback_median *before* _aggregate_predictions runs.
    await bot._research_and_make_predictions(mock_binary_question)
    assert bot._stacker_outcome.get(mock_binary_question.id_of_question) == "fallback_median"


# ---------------------------------------------------------------------------
# Publish hardening
# ---------------------------------------------------------------------------


def test_publish_hardening_retries_on_timeout_and_succeeds(monkeypatch):
    """First call times out, retry succeeds: net effect is success, one warning."""
    from metaculus_bot import publish_hardening

    # Reset sentinel + use small budgets.
    monkeypatch.setattr("metaculus_bot.publish_hardening.PUBLISH_POST_TIMEOUT", 0.05)
    monkeypatch.setattr("metaculus_bot.publish_hardening.PUBLISH_POST_RETRIES", 1)

    n_calls = {"n": 0}

    def fake_post(*args, **kwargs):
        n_calls["n"] += 1
        if n_calls["n"] == 1:
            time.sleep(0.5)  # exceeds 0.05s timeout
        return None

    wrapped = publish_hardening._wrap_with_timeout_retry("fake", fake_post)
    # Should succeed on retry (returns None).
    assert wrapped("dummy") is None
    assert n_calls["n"] == 2  # 1 attempt timed out, 1 retry succeeded


def test_publish_hardening_gives_up_after_retry_exhausted(monkeypatch):
    """Both attempts time out: outermost call raises TimeoutError."""
    from metaculus_bot import publish_hardening

    monkeypatch.setattr("metaculus_bot.publish_hardening.PUBLISH_POST_TIMEOUT", 0.05)
    monkeypatch.setattr("metaculus_bot.publish_hardening.PUBLISH_POST_RETRIES", 1)

    n_calls = {"n": 0}

    def fake_post(*args, **kwargs):
        n_calls["n"] += 1
        time.sleep(0.5)
        return None

    wrapped = publish_hardening._wrap_with_timeout_retry("fake", fake_post)
    with pytest.raises(concurrent.futures.TimeoutError):
        wrapped("dummy")
    assert n_calls["n"] == 2  # initial + 1 retry


def test_publish_hardening_retries_on_request_exception(monkeypatch):
    """Connection error: retried, succeeds on 2nd attempt."""
    from metaculus_bot import publish_hardening

    monkeypatch.setattr("metaculus_bot.publish_hardening.PUBLISH_POST_RETRIES", 1)

    n_calls = {"n": 0}

    def fake_post(*args, **kwargs):
        n_calls["n"] += 1
        if n_calls["n"] == 1:
            raise requests.ConnectionError("network down")
        return None

    wrapped = publish_hardening._wrap_with_timeout_retry("fake", fake_post)
    assert wrapped("dummy") is None
    assert n_calls["n"] == 2


def test_publish_hardening_idempotent(monkeypatch):
    """apply_publish_hardening is a no-op the second time."""
    from forecasting_tools import MetaculusApi

    from metaculus_bot import publish_hardening

    # Capture original methods so we can restore them.
    originals = {name: getattr(MetaculusApi, name) for name in publish_hardening._PATCHED_METHODS}
    # Reset sentinel for a clean start.
    if hasattr(MetaculusApi, publish_hardening._SENTINEL):
        delattr(MetaculusApi, publish_hardening._SENTINEL)
        for name, fn in originals.items():
            setattr(MetaculusApi, name, fn)

    publish_hardening.apply_publish_hardening()
    after_first = {name: getattr(MetaculusApi, name) for name in publish_hardening._PATCHED_METHODS}
    publish_hardening.apply_publish_hardening()
    after_second = {name: getattr(MetaculusApi, name) for name in publish_hardening._PATCHED_METHODS}
    # The second call must NOT re-wrap (otherwise we'd see a new wrapper object).
    for name in publish_hardening._PATCHED_METHODS:
        assert after_first[name] is after_second[name]
