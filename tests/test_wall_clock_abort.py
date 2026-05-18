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


@pytest.mark.asyncio
async def test_tight_budget_under_stacking_forces_fallback_mean(monkeypatch, mock_binary_question, mock_general_llm):
    """F15: STACKING budget-skip path must set outcome=fallback_mean (not fallback_median).

    The base-combine re-entry under AggregationStrategy.STACKING uses MEAN
    (main.py:1308-1314). Setting outcome=fallback_median there mislabels the
    aggregation method in the comment marker, contaminating downstream
    residual-analysis cuts. After the F15 fix, the budget-skip path must pick
    fallback_mean for STACKING and fallback_median for CONDITIONAL_STACKING.
    """
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
        aggregation_strategy=AggregationStrategy.STACKING,
    )
    bot._get_notepad = AsyncMock(return_value=MagicMock(total_research_reports_attempted=0))
    bot.run_research = AsyncMock(return_value="research")

    async def fast(*args, **kwargs):
        return ReasonedPrediction(prediction_value=0.6, reasoning="ok")

    bot._forecaster_with_soft_deadline = fast

    await bot._research_and_make_predictions(mock_binary_question)
    assert bot._stacker_outcome.get(mock_binary_question.id_of_question) == "fallback_mean"


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

    # Use monkeypatch to capture + restore each method (and the sentinel) so the
    # real MetaculusApi class is unpatched at test exit. monkeypatch.setattr
    # records originals before the test runs and restores them after.
    for name in publish_hardening._PATCHED_METHODS:
        # The descriptor in __dict__ is the classmethod wrapper; setattr accepts
        # the same shape on restore, so capturing via getattr is fine.
        monkeypatch.setattr(MetaculusApi, name, MetaculusApi.__dict__[name])
    # Sentinel: pytest's monkeypatch will delete this attr if it didn't exist
    # before, or restore the prior value if it did.
    if hasattr(MetaculusApi, publish_hardening._SENTINEL):
        monkeypatch.setattr(MetaculusApi, publish_hardening._SENTINEL, False)
        delattr(MetaculusApi, publish_hardening._SENTINEL)
    else:
        monkeypatch.setattr(MetaculusApi, publish_hardening._SENTINEL, False, raising=False)
        delattr(MetaculusApi, publish_hardening._SENTINEL)

    publish_hardening.apply_publish_hardening()
    # Compare the underlying function objects in __dict__ — bound classmethod
    # views from getattr() are fresh each access so identity-via-getattr is
    # unreliable. The wrapper function inside the classmethod descriptor IS
    # identity-stable across calls.
    after_first = {name: MetaculusApi.__dict__[name].__func__ for name in publish_hardening._PATCHED_METHODS}
    publish_hardening.apply_publish_hardening()
    after_second = {name: MetaculusApi.__dict__[name].__func__ for name in publish_hardening._PATCHED_METHODS}
    for name in publish_hardening._PATCHED_METHODS:
        assert after_first[name] is after_second[name]


def test_publish_hardening_supports_class_and_instance_calls(monkeypatch):
    """F18 regression: hardened post_* must be callable from class AND instance.

    Original code used ``setattr(MetaculusApi, name, wrapper)`` where ``wrapper``
    was a plain function — class-level calls worked but instance-level calls
    passed ``self`` as the first positional arg, breaking the signature. Fix
    re-wraps as ``classmethod`` so both calling conventions work.
    """
    from forecasting_tools import MetaculusApi

    from metaculus_bot import publish_hardening

    # Snapshot + restore via monkeypatch.
    for name in publish_hardening._PATCHED_METHODS:
        monkeypatch.setattr(MetaculusApi, name, MetaculusApi.__dict__[name])
    if hasattr(MetaculusApi, publish_hardening._SENTINEL):
        monkeypatch.setattr(MetaculusApi, publish_hardening._SENTINEL, False)
        delattr(MetaculusApi, publish_hardening._SENTINEL)

    # Stub _post_question_prediction so we don't hit the network. It's the
    # underlying call inside post_binary_question_prediction.
    calls: list[tuple[int, dict]] = []

    def fake_post_question_prediction(cls, question_id, payload):
        calls.append((question_id, payload))

    monkeypatch.setattr(MetaculusApi, "_post_question_prediction", classmethod(fake_post_question_prediction))

    publish_hardening.apply_publish_hardening()

    # Class-level call must work.
    MetaculusApi.post_binary_question_prediction(question_id=1, prediction_in_decimal=0.5)
    # Instance-level call must ALSO work without TypeError.
    MetaculusApi().post_binary_question_prediction(question_id=2, prediction_in_decimal=0.6)

    assert len(calls) == 2
    assert calls[0][0] == 1
    assert calls[1][0] == 2


def test_publish_hardening_injects_socket_timeout(monkeypatch):
    """F17 regression: a hung requests.post must be bounded by injected ``timeout=``.

    Without the request-side socket timeout, ``Future.cancel()`` is a no-op once
    the worker thread is inside ``requests.post``, so a hung POST silently runs
    until the underlying socket times out (no caller-side bound). With the fix,
    the wrapper monkey-patches ``requests.post`` on the forecasting-tools
    module to inject ``timeout=PUBLISH_POST_TIMEOUT`` whenever the caller didn't
    pass one, so the socket-level timeout fires.

    This test verifies that the injection happens by capturing kwargs passed to
    a fake ``requests.post``.
    """
    from forecasting_tools.helpers import metaculus_api as ft_metaculus_api

    from metaculus_bot import publish_hardening

    monkeypatch.setattr("metaculus_bot.publish_hardening.PUBLISH_POST_TIMEOUT", 0.5)
    monkeypatch.setattr("metaculus_bot.publish_hardening.PUBLISH_POST_RETRIES", 0)

    captured_kwargs: list[dict] = []

    class FakeResponse:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {}

    def fake_post(*args, **kwargs):
        captured_kwargs.append(dict(kwargs))
        return FakeResponse()

    monkeypatch.setattr(ft_metaculus_api.requests, "post", fake_post)

    # Build a minimal "post"-like fn that calls requests.post on the patched
    # module, the same way MetaculusApi's internals do.
    def caller_that_uses_module_requests(*args, **kwargs):
        return ft_metaculus_api.requests.post("http://example.invalid", json={"k": "v"})

    wrapped = publish_hardening._wrap_with_timeout_retry("fake", caller_that_uses_module_requests)
    wrapped()

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0].get("timeout") == 0.5, (
        f"timeout kwarg should have been injected by _inject_socket_timeout; got {captured_kwargs[0]}"
    )

    # And after the wrapper exits, requests.post must be restored to fake_post
    # (NOT the timeout-injecting wrapper). Verify by calling directly without
    # a timeout kwarg and confirming none was injected.
    captured_kwargs.clear()
    ft_metaculus_api.requests.post("http://example.invalid", json={"k": "v"})
    assert "timeout" not in captured_kwargs[0], (
        f"timeout injection should NOT leak past the wrapper; got {captured_kwargs[0]}"
    )


def test_publish_hardening_bounds_hung_request_via_socket_timeout(monkeypatch):
    """F17: a requests.post that hangs forever must be bounded by injected timeout.

    Simulates a server-stalled POST by having fake_post raise requests.Timeout
    when the timeout kwarg is passed (the real behavior of urllib3 when the
    socket timeout fires). Without F17, no timeout is passed and the call
    would hang. With F17, the wrapper raises after the timeout-induced error.
    """
    from forecasting_tools.helpers import metaculus_api as ft_metaculus_api

    from metaculus_bot import publish_hardening

    monkeypatch.setattr("metaculus_bot.publish_hardening.PUBLISH_POST_TIMEOUT", 0.5)
    monkeypatch.setattr("metaculus_bot.publish_hardening.PUBLISH_POST_RETRIES", 0)

    def fake_hung_post(*args, **kwargs):
        # urllib3 / requests raises requests.Timeout when the socket-level
        # timeout fires. Simulate that here, gated on the injected kwarg.
        if "timeout" in kwargs:
            raise requests.Timeout("simulated socket timeout")
        # If no timeout is passed, simulate an unbounded hang. Sleep longer
        # than any reasonable test would tolerate — if F17 were broken this
        # branch would be reached and the test would fail by timeout.
        time.sleep(60)

    monkeypatch.setattr(ft_metaculus_api.requests, "post", fake_hung_post)

    def caller(*args, **kwargs):
        return ft_metaculus_api.requests.post("http://example.invalid", json={})

    wrapped = publish_hardening._wrap_with_timeout_retry("fake", caller)
    start = time.monotonic()
    with pytest.raises(requests.Timeout):
        wrapped()
    elapsed = time.monotonic() - start
    # Must complete promptly (well under PUBLISH_POST_TIMEOUT + 1s); the
    # injected timeout makes fake_hung_post raise immediately.
    assert elapsed < 1.5, f"hung-post test took {elapsed:.2f}s, indicating socket timeout NOT injected"
