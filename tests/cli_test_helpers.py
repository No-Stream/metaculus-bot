"""Shared harness for the ``tests/cli`` package: the cli-under-test stubs and its date pins.

Plain functions and context managers, at the tests root like ``tests/resolution_source_fakes.py``
and ``tests/pipeline_test_helpers.py``; the package's autouse counter reset lives in
``tests/cli/conftest.py``.
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import GeneralLlm

from metaculus_bot.aggregation_pipeline import AggregationPipeline
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import (
    CREDIT_ALERT_RESUME_DATE,
    DONATED_OPENROUTER_KEY_ENABLED_ENV,
    MANTIC_TOKEN_ENV,
    credit_alerts_active,
)
from metaculus_bot.forecaster import TemplateForecaster
from metaculus_bot.research.orchestrator import ResearchOrchestrator

# Injected rather than read from the clock, so both suppression branches keep running forever.
DURING_SUPPRESSION = date(2026, 7, 25)
ON_RESUME_DATE = CREDIT_ALERT_RESUME_DATE
AFTER_RESUME_DATE = date(2026, 10, 1)

# Provider-degradation suppression has nowhere to inject a date, so pin its branches beyond the clock's reach.
PERMANENTLY_FUTURE_RESUME = date(2099, 1, 1)
PERMANENTLY_PAST_RESUME = date(2000, 1, 1)


def asyncio_run_stub(side_effect):
    """A ``metaculus_bot.cli.asyncio.run`` stand-in that closes the coroutine it is given.

    ``cli.main`` calls ``asyncio.run(template_bot.forecast_questions(...))``, so the
    coroutine is constructed by the inner call and handed to ``asyncio.run``, which
    owns it. Patching ``asyncio.run`` with a bare ``side_effect`` drops it, and since
    ``forecast_questions`` is an ``AsyncMock`` on the stub bot, the dropped object is a
    real coroutine: it is later garbage-collected unawaited and emits ``RuntimeWarning:
    coroutine ... was never awaited``, attributed to whichever unrelated test happened
    to trigger the collection.

    Closing it honors the ownership contract without executing it, then defers to
    ``side_effect`` for the behavior each test is actually pinning (crash, or return a
    report list).
    """

    def _close_then_apply(*args: object, **kwargs: object):
        for arg in args:
            if inspect.iscoroutine(arg):
                arg.close()
        return side_effect(*args, **kwargs)

    return _close_then_apply


@contextmanager
def _cli_main_test_mode(
    alertable_count: int,
    *,
    donated_below_floor: bool = False,
    fall_cup_reminder: bool = False,
    tournament_stale: bool = False,
    today: date | None = None,
    stub_bot: MagicMock | None = None,
    forecaster_class: MagicMock | None = None,
    mode: str = "test_questions",
    only_posts: str | None = None,
) -> Iterator[MagicMock]:
    """Run ``cli.main`` with every external dependency stubbed; yields the CreditTelemetry stub.

    TemplateForecaster becomes a MagicMock with the given ``alertable_count`` whose
    ``forecast_questions`` returns []; CreditTelemetry is stubbed so no test reaches the real
    OpenRouter balance endpoint, with ``donated_below_floor`` as its floor verdict; argv is pinned.

    ``today`` pins the credit-suppression window through cli's own ``credit_alerts_active``
    reference, and ``None`` leaves the real clock (the production path). ``stub_bot`` replaces the
    whole bot, for ``alertable_count`` COMPUTED through the real property chain (see
    ``_bot_with_real_alertable_count``); ``forecaster_class`` (see ``_forecaster_class``) replaces
    the CLASS mock, for tests reading the constructor kwargs cli passed, and its ``return_value``
    is then the bot. Either makes the ``alertable_count`` argument moot. ``mode`` (default
    ``test_questions``, the cheapest path through ``_question_source``) and ``only_posts`` go onto
    argv; ``fall_cup_reminder`` and ``tournament_stale`` pin verdicts read off the prod clock.
    """
    if forecaster_class is None:
        if stub_bot is None:
            stub_bot = MagicMock()
            stub_bot.alertable_count = alertable_count
        forecaster_class = MagicMock(return_value=stub_bot)
    bot_stub = forecaster_class.return_value
    bot_stub.forecast_questions = AsyncMock(return_value=[])
    bot_stub.forecast_on_tournament = AsyncMock(return_value=[])

    stub_telemetry = MagicMock()
    stub_telemetry.log_end_and_check_floor.return_value = donated_below_floor

    # Re-bind cli's own reference so only the injected date decides the window (the real function when None).
    pinned_clock = patch(
        "metaculus_bot.cli.credit_alerts_active",
        credit_alerts_active if today is None else lambda: credit_alerts_active(today),
    )

    argv_backup = sys.argv
    sys.argv = ["cli", "--mode", mode] + ([] if only_posts is None else ["--only-posts", only_posts])
    try:
        with (
            pinned_clock,
            # TemplateForecaster(...) call returns our stub
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            # ``get_question_by_url`` hands back a Mock so list construction doesn't explode.
            patch("metaculus_bot.cli.MetaculusApi", MagicMock()),
            # The real hardening permanently mutates MetaculusClient, leaking into every later test in the session.
            patch("metaculus_bot.cli.apply_publish_hardening"),
            patch("metaculus_bot.cli.apply_fetch_hardening"),
            patch("metaculus_bot.cli.check_tournament_dates", return_value=tournament_stale),
            # Left unpinned, FALL_CUP_REMINDER_DATE would flip this whole suite red off the real clock.
            patch("metaculus_bot.cli.check_fall_cup_reminder", return_value=fall_cup_reminder),
            # Both preflights make a real unauthenticated GET to the platform host; stub them to stay hermetic.
            patch("metaculus_bot.cli.verify_metaculus_api_identity"),
            patch("metaculus_bot.cli.verify_api_identity"),
            # The Mantic tournament preflight is an authenticated GET on the real client main builds.
            patch("metaculus_bot.cli.preflight_mantic_tournaments"),
            # A classmethod that iterates forecast_reports; our stub returns [], so keep the surface small.
            patch.object(type(bot_stub), "log_report_summary", create=True, return_value=None),
            patch("metaculus_bot.cli.CreditTelemetry", return_value=stub_telemetry),
            # The real install leaks a RoleSpendTracker into litellm's process-global callbacks for the session.
            patch("metaculus_bot.cli.install_role_spend_tracker"),
        ):
            yield stub_telemetry
    finally:
        sys.argv = argv_backup


def _forecaster_class() -> MagicMock:
    """A ``TemplateForecaster`` class stub whose constructor kwargs stay inspectable.

    Hand it to ``_cli_main_test_mode`` as ``forecaster_class`` when a test reads what cli built and
    passed in (``research_sink``, ``metaculus_client``) or pins when the class was called at all.
    ``alertable_count`` is a real int because a run reaching the end of ``main`` compares it.
    """
    forecaster_class = MagicMock()
    forecaster_class.return_value.alertable_count = 0
    return forecaster_class


_FAKE_MANTIC_TOKEN = "m" * 40


def _mantic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment a Mantic run requires: the donated-key switch off and a (fake) Mantic token.

    Set explicitly rather than inherited, because the operator's ``.env`` (loaded at import by
    ``constants``) may carry a real ``MANTIC_TOKEN`` and never carries the switch.
    """
    monkeypatch.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, "false")
    monkeypatch.setenv(MANTIC_TOKEN_ENV, _FAKE_MANTIC_TOKEN)


@contextmanager
def _configure_process_stubs() -> Iterator[dict[str, MagicMock]]:
    """``_configure_process`` with its process-global side effects stubbed: the hardening patches
    (they mutate MetaculusClient for the whole session) and both identity preflights (real GETs)."""
    with (
        patch("metaculus_bot.cli.apply_publish_hardening") as publish_hardening,
        patch("metaculus_bot.cli.apply_fetch_hardening") as fetch_hardening,
        patch("metaculus_bot.cli.verify_api_identity") as verify_api,
        patch("metaculus_bot.cli.verify_metaculus_api_identity") as verify_metaculus,
    ):
        yield {
            "publish_hardening": publish_hardening,
            "fetch_hardening": fetch_hardening,
            "verify_api_identity": verify_api,
            "verify_metaculus_api_identity": verify_metaculus,
        }


def _open_question(post_id: int) -> MagicMock:
    """A fetched open question, as far as the ``--only-posts`` filter reads one."""
    question = MagicMock()
    question.id_of_post = post_id
    return question


def _filterable_bot(*open_post_ids: int) -> MagicMock:
    """A bot whose client holds ``open_post_ids`` open and whose two forecast entry points are spies."""
    bot = MagicMock()
    bot.metaculus_client.get_all_open_questions_from_tournament.return_value = [
        _open_question(post_id) for post_id in open_post_ids
    ]
    bot.forecast_questions = AsyncMock(return_value=["report"])
    bot.forecast_on_tournament = AsyncMock(return_value=["report"])
    return bot


class _RealAlertableCountBot(MagicMock):
    """A cli stub whose ``alertable_count`` is COMPUTED, not pinned to a literal.

    The provider-degradation summand has to travel the whole real chain — the observation store,
    ``ResearchOrchestrator.provider_degradation_count``, ``TemplateForecaster._provider_degradation_count``,
    ``alertable_count``, then the ``sys.exit`` in cli — or the test proves only that cli exits on a number
    the test handed it, which is how a broken summand ships green.

    A dedicated SUBCLASS rather than assignments onto ``type(mock)``, which for a ``MagicMock`` instance
    is ``MagicMock`` itself and would leak the properties into every mock in the session.

    EVERY adapter property used by the snapshot has to be listed below (aggregation counters come from
    the real pipeline the fixture installs): a missing owner leaves a ``MagicMock`` in the sum, so
    ``alertable_count`` stops being an int and every exit-code test in ``test_cli_exit_status.py``
    and ``test_cli_provider_degradation.py`` (both under ``tests/cli/``) fails at once.
    """

    alertable_count = TemplateForecaster.alertable_count
    _degradation_snapshot = TemplateForecaster._degradation_snapshot
    _research_provider_failure_count = TemplateForecaster._research_provider_failure_count
    _summarizer_failure_count = TemplateForecaster._summarizer_failure_count
    _gap_fill_v1_error_count = TemplateForecaster._gap_fill_v1_error_count
    _gap_fill_v2_error_count = TemplateForecaster._gap_fill_v2_error_count
    _prediction_market_degraded_count = TemplateForecaster._prediction_market_degraded_count
    _prediction_market_source_loss_count = TemplateForecaster._prediction_market_source_loss_count
    _provider_degradation_count = TemplateForecaster._provider_degradation_count
    _publish_attempt_failures = TemplateForecaster._publish_attempt_failures
    _publish_skipped_closed_count = TemplateForecaster._publish_skipped_closed_count


def _bot_with_real_alertable_count() -> _RealAlertableCountBot:
    """Build the computed-``alertable_count`` stub with a REAL orchestrator attached.

    Only what cli touches is real: the orchestrator supplies the provider-degradation
    and prediction-market properties, and the bot-side counters start at zero so the
    provider-degradation summand is the only thing that can move the total.
    """
    stub_bot = _RealAlertableCountBot()
    stub_bot._research = ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())
    stub_bot._pipeline = AggregationPipeline(
        strategy=AggregationStrategy.MEAN,
        stacker_llm=None,
        parser_llm=GeneralLlm(model="test-model", temperature=0.0),
    )
    for counter in (
        "_forecasters_dropped_count",
        "_questions_failed_to_publish",
        "_time_budget_fast_path_count",
    ):
        setattr(stub_bot, counter, 0)
    return stub_bot
