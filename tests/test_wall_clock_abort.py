"""Tests for the per-question wall-clock abort + publish hardening (task #13).

The bot uses `PER_QUESTION_WALL_CLOCK_DEADLINE` (3510s) to bound a single
question's research + fan-out + aggregation + publish budget. At deadline,
in-flight forecasters get cancelled, completed predictions ride a base-combine
median, and stacking is skipped if remaining budget falls below
`WALL_CLOCK_STACKING_MIN_BUDGET`.

Publish hardening is a separate concern: the shared prediction helper
`MetaculusClient._post_question_prediction` (the single POST that all three
public `post_*_question_prediction` wrappers delegate to) and
`post_question_comment` are monkey-patched to add `PUBLISH_POST_TIMEOUT` (20s)
per call plus `PUBLISH_POST_RETRIES` (1) on timeout / connection errors. Both
are wrapped *beneath* their upstream `@retry_with_exponential_backoff` so ours
is the single retry layer on each path. See `metaculus_bot/publish_hardening.py`.
"""

import asyncio
import concurrent.futures
import functools
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests
from forecasting_tools import BinaryQuestion, GeneralLlm, ReasonedPrediction
from forecasting_tools.data_models.data_organizer import PredictionTypes

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.publish_hardening import reset_publish_attempt_failures


def _stub_open_time() -> datetime:
    return datetime.now() - timedelta(days=30)


def _stub_resolve_time() -> datetime:
    return datetime.now() + timedelta(days=365)


@pytest.fixture(autouse=True)
def _reset_publish_attempt_failures():
    """The retry-exhaustion tests below bump publish_hardening's module-global
    counter; without a reset it leaks into whichever test file runs next and
    makes the suite order-dependent."""
    reset_publish_attempt_failures()
    yield
    reset_publish_attempt_failures()


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
    monkeypatch.setattr("metaculus_bot.forecaster.PER_QUESTION_WALL_CLOCK_DEADLINE", 0.2)
    monkeypatch.setattr("metaculus_bot.stacking_route.WALL_CLOCK_STACKING_MIN_BUDGET", 0.0)

    bot = _make_bot(mock_general_llm, n_forecasters=4, min_forecasters_to_publish=3)
    bot._get_notepad = AsyncMock(return_value=MagicMock(total_research_reports_attempted=0))
    bot.run_research = AsyncMock(return_value="research")

    # 3 fast, 1 slow (will be cancelled).
    call_count = {"n": 0}

    async def mixed(*args, **kwargs) -> ReasonedPrediction[PredictionTypes]:
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
    monkeypatch.setattr("metaculus_bot.forecaster.PER_QUESTION_WALL_CLOCK_DEADLINE", 0.1)

    bot = _make_bot(mock_general_llm, n_forecasters=4, min_forecasters_to_publish=3)
    bot._get_notepad = AsyncMock(return_value=MagicMock(total_research_reports_attempted=0))
    bot.run_research = AsyncMock(return_value="research")

    call_count = {"n": 0}

    async def mixed(*args, **kwargs) -> ReasonedPrediction[PredictionTypes]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ReasonedPrediction(prediction_value=0.5, reasoning="ok")
        await asyncio.sleep(10)
        return ReasonedPrediction(prediction_value=0.5, reasoning="never")

    bot._forecaster_with_soft_deadline = mixed

    assert bot._questions_failed_to_publish == 0
    with pytest.raises((RuntimeError, ExceptionGroup)):  # 3.11+ builtin
        await bot._research_and_make_predictions(mock_binary_question)
    assert bot._questions_failed_to_publish == 1
    # 3 forecasters were cancelled.
    assert bot._forecasters_dropped_count == 3


@pytest.mark.asyncio
async def test_tight_budget_skips_stacking_forces_fallback_median(monkeypatch, mock_binary_question, mock_general_llm):
    """Even with all forecasters returning, sub-90s budget forces fallback_median."""
    monkeypatch.setattr("metaculus_bot.forecaster.PER_QUESTION_WALL_CLOCK_DEADLINE", 0.5)
    monkeypatch.setattr("metaculus_bot.stacking_route.WALL_CLOCK_STACKING_MIN_BUDGET", 1_000_000)  # always trips

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

    async def fast(*args, **kwargs) -> ReasonedPrediction[PredictionTypes]:
        return ReasonedPrediction(prediction_value=0.6, reasoning="ok")

    bot._forecaster_with_soft_deadline = fast

    # _aggregate_predictions invocation in non-stacking path should set
    # pipeline outcomes to "fallback_median" via our gate. Our skip path sets
    # outcome=fallback_median *before* _aggregate_predictions runs.
    await bot._research_and_make_predictions(mock_binary_question)
    assert bot._pipeline.outcomes.get(mock_binary_question.id_of_question) == "fallback_median"
    # The budget skip records the same skip-reason + counter treatment as its
    # sibling skip paths, so a STACKER_SKIP_REASON cut cannot miss this bucket.
    assert bot._pipeline.skip_reasons.get(mock_binary_question.id_of_question) == "wall_clock_budget"
    assert bot._pipeline.counters.conditional_stacking_skipped_count == 1


@pytest.mark.asyncio
async def test_tight_budget_under_stacking_forces_fallback_mean(monkeypatch, mock_binary_question, mock_general_llm):
    """F15: STACKING budget-skip path must set outcome=fallback_mean (not fallback_median).

    The base-combine re-entry under AggregationStrategy.STACKING uses MEAN
    (main.py:1308-1314). Setting outcome=fallback_median there mislabels the
    aggregation method in the comment marker, contaminating downstream
    residual-analysis cuts. After the F15 fix, the budget-skip path must pick
    fallback_mean for STACKING and fallback_median for CONDITIONAL_STACKING.
    """
    monkeypatch.setattr("metaculus_bot.forecaster.PER_QUESTION_WALL_CLOCK_DEADLINE", 0.5)
    monkeypatch.setattr("metaculus_bot.stacking_route.WALL_CLOCK_STACKING_MIN_BUDGET", 1_000_000)  # always trips

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

    async def fast(*args, **kwargs) -> ReasonedPrediction[PredictionTypes]:
        return ReasonedPrediction(prediction_value=0.6, reasoning="ok")

    bot._forecaster_with_soft_deadline = fast

    await bot._research_and_make_predictions(mock_binary_question)
    assert bot._pipeline.outcomes.get(mock_binary_question.id_of_question) == "fallback_mean"


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
        return

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
        return

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
        return

    wrapped = publish_hardening._wrap_with_timeout_retry("fake", fake_post)
    assert wrapped("dummy") is None
    assert n_calls["n"] == 2


def test_publish_hardening_idempotent(monkeypatch):
    """apply_publish_hardening is a no-op the second time."""
    from forecasting_tools.helpers.metaculus_client import MetaculusClient

    from metaculus_bot import publish_hardening

    # Use monkeypatch to capture + restore each method (and the sentinel) so the
    # real MetaculusClient class is unpatched at test exit. monkeypatch.setattr
    # records originals before the test runs and restores them after.
    for name in publish_hardening._PATCHED_METHODS:
        monkeypatch.setattr(MetaculusClient, name, MetaculusClient.__dict__[name])
    # Sentinel: pytest's monkeypatch will delete this attr if it didn't exist
    # before, or restore the prior value if it did.
    if hasattr(MetaculusClient, publish_hardening._SENTINEL):
        monkeypatch.setattr(MetaculusClient, publish_hardening._SENTINEL, False)
        delattr(MetaculusClient, publish_hardening._SENTINEL)
    else:
        monkeypatch.setattr(MetaculusClient, publish_hardening._SENTINEL, False, raising=False)
        delattr(MetaculusClient, publish_hardening._SENTINEL)

    publish_hardening.apply_publish_hardening()
    # 0.2.92's post_* are plain instance methods, so the wrapper is a plain
    # function in __dict__ (identity-stable across calls) — no classmethod
    # descriptor unwrap needed.
    after_first = {name: MetaculusClient.__dict__[name] for name in publish_hardening._PATCHED_METHODS}
    publish_hardening.apply_publish_hardening()
    after_second = {name: MetaculusClient.__dict__[name] for name in publish_hardening._PATCHED_METHODS}
    for name in publish_hardening._PATCHED_METHODS:
        assert after_first[name] is after_second[name]


def test_publish_hardening_patches_shared_prediction_helper_not_public_wrappers(monkeypatch):
    """0.2.92 seam: hardening wraps the shared private helper; public wrappers stay pristine and delegate.

    The three public ``post_*_question_prediction`` methods are *undecorated*
    wrappers that only validate input and delegate to the
    ``@retry_with_exponential_backoff()``-decorated ``_post_question_prediction``.
    Patching the public wrappers (an earlier version) left that inner upstream
    retry stacked *beneath* ours — the double-retry bug this fix removes. So
    hardening patches ``_post_question_prediction`` instead: the public wrappers
    are left untouched and route into the hardened helper. The wrapper must be a
    plain function bound as an instance method (not a classmethod — that would
    bind ``self`` to ``cls`` and break the bot's instance calls), so two distinct
    clients both route through it with ``self`` bound correctly.
    """
    from forecasting_tools.helpers.metaculus_client import MetaculusClient

    from metaculus_bot import publish_hardening

    # Pin the fix: the patch table targets the shared private helper, not the
    # public prediction wrappers. This is the crux of the double-retry fix.
    assert "_post_question_prediction" in publish_hardening._PATCHED_METHODS
    assert "post_binary_question_prediction" not in publish_hardening._PATCHED_METHODS

    # Snapshot + restore via monkeypatch.
    for name in publish_hardening._PATCHED_METHODS:
        monkeypatch.setattr(MetaculusClient, name, MetaculusClient.__dict__[name])
    if hasattr(MetaculusClient, publish_hardening._SENTINEL):
        monkeypatch.setattr(MetaculusClient, publish_hardening._SENTINEL, False)
        delattr(MetaculusClient, publish_hardening._SENTINEL)
    else:
        monkeypatch.setattr(MetaculusClient, publish_hardening._SENTINEL, False, raising=False)
        delattr(MetaculusClient, publish_hardening._SENTINEL)

    # The public wrapper is never patched; capture it to prove hardening leaves it be.
    public_wrapper_before = MetaculusClient.__dict__["post_binary_question_prediction"]

    # Stub the shared private helper (what hardening now wraps) so we don't hit
    # the network. It's the single POST every public prediction wrapper delegates to.
    calls: list[tuple[int, dict]] = []

    def fake_post_question_prediction(self, question_id, payload):
        calls.append((question_id, payload))

    monkeypatch.setattr(MetaculusClient, "_post_question_prediction", fake_post_question_prediction)

    publish_hardening.apply_publish_hardening()

    # After hardening: the private helper is our wrapper; the public wrapper is untouched.
    assert MetaculusClient.__dict__["_post_question_prediction"] is not fake_post_question_prediction
    assert MetaculusClient.__dict__["post_binary_question_prediction"] is public_wrapper_before

    # Two distinct instances (the bot constructs its own) must both route through
    # the pristine public wrapper -> hardened helper without a self/cls binding error.
    MetaculusClient().post_binary_question_prediction(question_id=1, prediction_in_decimal=0.5)
    MetaculusClient().post_binary_question_prediction(question_id=2, prediction_in_decimal=0.6)

    assert len(calls) == 2
    assert calls[0][0] == 1
    assert calls[1][0] == 2


def test_publish_hardening_unwraps_upstream_retry_so_it_is_the_single_layer(monkeypatch):
    """Regression: the ``__wrapped__`` unwrap strips the upstream retry so ours is the ONLY layer.

    Before the fix, our wrapper sat on top of the still-decorated
    ``_post_question_prediction``, so a stall triggered BOTH retry loops: the
    inner upstream one (up to 4 attempts, sleeping min(delay*jitter, 75s) each)
    ran to completion inside a worker that our outer ``Future.result`` had already
    abandoned. Here we install a fake upstream helper with its own retry loop and
    a ``functools.wraps``-style ``__wrapped__`` pointing at the raw inner. After
    ``apply_publish_hardening`` unwraps ``__wrapped__``, the inner is called
    exactly once per OUTER attempt — proving the upstream retry loop was stripped
    (otherwise the inner would fire outer_attempts * upstream_attempts times).
    """
    from forecasting_tools.helpers.metaculus_client import MetaculusClient

    from metaculus_bot import publish_hardening

    monkeypatch.setattr("metaculus_bot.publish_hardening.PUBLISH_POST_TIMEOUT", 0.5)
    monkeypatch.setattr("metaculus_bot.publish_hardening.PUBLISH_POST_RETRIES", 1)

    # Snapshot + restore the patched methods and clear the sentinel.
    for name in publish_hardening._PATCHED_METHODS:
        monkeypatch.setattr(MetaculusClient, name, MetaculusClient.__dict__[name])
    if hasattr(MetaculusClient, publish_hardening._SENTINEL):
        monkeypatch.setattr(MetaculusClient, publish_hardening._SENTINEL, False)
        delattr(MetaculusClient, publish_hardening._SENTINEL)
    else:
        monkeypatch.setattr(MetaculusClient, publish_hardening._SENTINEL, False, raising=False)
        delattr(MetaculusClient, publish_hardening._SENTINEL)

    inner_calls = {"n": 0}
    upstream_attempts = 4  # mimic @retry_with_exponential_backoff (max_retries=3 -> 4 tries)

    def raw_inner(self, question_id, payload):
        inner_calls["n"] += 1
        raise requests.Timeout("simulated socket-close stall")

    # A stand-in for the upstream @retry_with_exponential_backoff-decorated helper:
    # its own retry loop over raw_inner, plus __wrapped__ = raw_inner (what wraps sets).
    @functools.wraps(raw_inner)
    def fake_decorated(self, question_id, payload):
        last_exc: BaseException = RuntimeError("never ran")
        for _ in range(upstream_attempts):
            try:
                return raw_inner(self, question_id, payload)
            except requests.RequestException as exc:
                last_exc = exc
        raise last_exc

    assert fake_decorated.__wrapped__ is raw_inner  # functools.wraps set the unwrap seam

    monkeypatch.setattr(MetaculusClient, "_post_question_prediction", fake_decorated)
    publish_hardening.apply_publish_hardening()

    outer_attempts = publish_hardening.PUBLISH_POST_RETRIES + 1  # == 2
    with pytest.raises(requests.Timeout):
        MetaculusClient()._post_question_prediction(1, {"probability_yes": 0.5})

    # Single-layer invariant: raw_inner fires once per OUTER attempt. If the unwrap
    # had failed, our wrapper would have wrapped fake_decorated and each outer
    # attempt would spin the inner loop -> outer_attempts * upstream_attempts (8).
    assert inner_calls["n"] == outer_attempts, (
        f"expected {outer_attempts} inner calls (single retry layer), got {inner_calls['n']}; "
        "the upstream retry was NOT stripped via __wrapped__"
    )
    assert inner_calls["n"] != outer_attempts * upstream_attempts, "double-retry stack present"


def test_publish_hardening_forces_socket_timeout(monkeypatch):
    """F17 regression: a hung requests.post must be bounded by a forced ``timeout=``.

    Without the request-side socket timeout, ``Future.cancel()`` is a no-op once
    the worker thread is inside ``requests.post``, so a hung POST silently runs
    until the underlying socket times out (no caller-side bound). So hardening
    patches ``requests.post`` on the metaculus_client module to force
    ``timeout=PUBLISH_POST_TIMEOUT``, OVERRIDING upstream's own 30s timeout so the
    socket closes in step with the caller-side Future cap.

    The override is installed ONCE by ``apply_publish_hardening`` and left in
    place, matching ``fetch_hardening``'s GET twin. It used to be a per-call
    context manager, which was only correct under strict LIFO nesting — and the
    timeout-and-retry path violates that by construction, since ``future.cancel()``
    returns False on a running future, so a timed-out orphan and its retry are
    both inside the context manager at once. The leak is covered directly in
    ``tests/test_publish_hardening_concurrency.py``; here we pin the behavior that
    matters to a POST: the tighter timeout wins over the one upstream supplies.
    """
    from forecasting_tools.helpers import metaculus_client as ft_metaculus_client

    from metaculus_bot import publish_hardening

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

    monkeypatch.setattr(ft_metaculus_client.requests, "post", fake_post)
    publish_hardening._install_post_timeout_override(0.5)
    try:
        # Call the way MetaculusClient's internals do on 0.2.92: WITH its own
        # timeout=self.timeout (30s). Ours must override it, not defer to it.
        ft_metaculus_client.requests.post(
            "https://www.metaculus.com/api/questions/forecast/", json={"k": "v"}, timeout=30
        )
        # A non-Metaculus POST must keep its own timeout: metaculus_client.requests IS
        # the global requests module, so an unscoped permanent override would also
        # re-time exa_py / litellm / huggingface POSTs (see the helper's docstring).
        ft_metaculus_client.requests.post("https://api.exa.ai/search", json={"k": "v"}, timeout=120)
    finally:
        monkeypatch.undo()

    assert len(captured_kwargs) == 2
    assert captured_kwargs[0].get("timeout") == 0.5, (
        f"the tighter publish timeout must OVERRIDE upstream's own; got {captured_kwargs[0]}"
    )
    assert captured_kwargs[1].get("timeout") == 120, (
        f"a non-Metaculus POST must keep its own timeout; got {captured_kwargs[1]}"
    )


def test_publish_hardening_bounds_hung_request_via_socket_timeout(monkeypatch):
    """F17: a requests.post that hangs forever must be bounded by the forced timeout.

    Simulates a server-stalled POST by having fake_post raise requests.Timeout
    when the timeout kwarg is present (the real behavior of urllib3 when the
    socket timeout fires) and hang otherwise. Without the forced timeout no bound
    reaches the socket, ``Future.cancel()`` is a no-op on the running worker, and
    the POST runs until the server gives up; with it, the wrapper surfaces the
    timeout-induced error promptly.

    The forced timeout is now installed once by ``apply_publish_hardening`` rather
    than per call (see ``_install_post_timeout_override``), so this drives the
    install and then the wrapper, in the order production does.
    """
    from forecasting_tools.helpers import metaculus_client as ft_metaculus_client

    from metaculus_bot import publish_hardening

    monkeypatch.setattr("metaculus_bot.publish_hardening.PUBLISH_POST_TIMEOUT", 0.5)
    monkeypatch.setattr("metaculus_bot.publish_hardening.PUBLISH_POST_RETRIES", 0)

    def fake_hung_post(*args, **kwargs):
        # urllib3 / requests raises requests.Timeout when the socket-level
        # timeout fires. Simulate that here, gated on the forced kwarg.
        if "timeout" in kwargs:
            raise requests.Timeout("simulated socket timeout")
        # With no timeout, simulate an unbounded hang. Sleep longer than any
        # reasonable test would tolerate — if the override were broken this branch
        # would be reached and the elapsed assertion below would fail.
        time.sleep(60)

    real_post = ft_metaculus_client.requests.post
    monkeypatch.setattr(ft_metaculus_client.requests, "post", fake_hung_post)
    publish_hardening._install_post_timeout_override(0.5)

    def caller(*args, **kwargs):
        return ft_metaculus_client.requests.post("https://www.metaculus.com/api/questions/forecast/", json={})

    wrapped = publish_hardening._wrap_with_timeout_retry("fake", caller)
    start = time.monotonic()
    try:
        with pytest.raises(requests.Timeout):
            wrapped()
    finally:
        ft_metaculus_client.requests.post = real_post
    elapsed = time.monotonic() - start
    # Must complete promptly (well under PUBLISH_POST_TIMEOUT + 1s); the forced
    # timeout makes fake_hung_post raise immediately.
    assert elapsed < 1.5, f"hung-post test took {elapsed:.2f}s, indicating socket timeout NOT forced"
