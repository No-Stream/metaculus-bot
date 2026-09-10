"""Tests for ``metaculus_bot.cli.main`` — specifically the sys.exit wiring that
fires when ``TemplateForecaster.alertable_count > 0``, when the donated
OpenRouter key fell back to the operator's personal (paid) key during the run,
or when the donated key's remaining balance ended the run below the
early-warning floor (``OPENROUTER_CREDIT_FLOOR_USD``).

The fallback counter folded into ``alertable`` is ``_generic_key_fallback_count``
— it counts EVERY donated->personal fallback (all causes: 401/402/429/guardrail/
404). ``_donated_404_fallback_count`` (allowed-providers 404) and
``_credit_key_fallback_count`` (402 / insufficient credit) are two disjoint
subsets of that total, broken out in the log line for diagnostics but NOT
separately added to ``alertable`` (that would double-count events already inside
the generic total).

Credit alerting is gated on ``CREDIT_ALERT_RESUME_DATE`` (2026-09-03, the day
Metaculus granted credits again). Before that date the floor breach did not exit
non-zero and the credit-caused fallbacks were subtracted back out of
``alertable``; every other fallback cause (401 / 404 / 429 / guardrail) always kept
its full weight. Tests inject the window state via ``credit_alerts_active`` rather
than the wall clock, so both sides stay covered now that the real date has passed.

Publication already happened inside ``forecast_on_tournament`` by the time cli
checks alertable state; the non-zero exit is purely so GitHub Actions marks
the run red. That wiring is load-bearing — without it, forecaster drops,
stacker fallback usage, silent personal-key spend, and a draining donated
balance all go unnoticed.
"""

from __future__ import annotations

import inspect
import json
import logging
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, ClassVar, get_args
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from forecasting_tools import GeneralLlm, MetaculusApi

import metaculus_bot.fallback_openrouter as fb_module
import metaculus_bot.research.prediction_market as pmp
from metaculus_bot import mantic as mantic_module
from metaculus_bot.aggregation_pipeline import AggregationPipeline
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.api_preflight import ApiIdentityError
from metaculus_bot.cli import (
    CliArgs,
    RunMode,
    _assert_personal_keys_only,
    _check_tournament_dates,
    _configure_process,
    _forecast_with_callback_drain,
    _parse_cli_args,
    _run_forecasts,
    persisted_platform,
    persisted_tournament_id,
)
from metaculus_bot.cli import main as cli_main
from metaculus_bot.constants import (
    CREDIT_ALERT_RESUME_DATE,
    DONATED_OPENROUTER_KEY_ENABLED_ENV,
    MANTIC_API_BASE_URL,
    MANTIC_TOKEN_ENV,
    MANTIC_TOURNAMENT_END_DATE,
    MANTIC_TOURNAMENT_ID,
    METACULUS_CUP_ID,
    PERSIST_RESEARCH_ENABLED_ENV,
    PROVIDER_DEGRADATION_SUPPRESSED_UNTIL,
    TOURNAMENT_ID,
    TournamentExpiredError,
    credit_alerts_active,
)
from metaculus_bot.credit_telemetry import DonatedKeyState, RoleSpendTracker, reset_donated_key_state_cache
from metaculus_bot.fallback_openrouter import (
    reset_credit_key_fallback_count,
    reset_donated_404_fallback_count,
    reset_generic_key_fallback_count,
)
from metaculus_bot.forecaster import TemplateForecaster
from metaculus_bot.mantic import ManticClient, reset_post_drop_count
from metaculus_bot.research.orchestrator import ResearchOrchestrator
from metaculus_bot.research.provider_health import (
    VENUE_EXPECTED_LIQUIDITY_FIELDS,
    VenueObservation,
    record_venue_observation,
    reset_provider_health,
)
from scripts.telemetry.markers import MARKER_SPECS

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


@pytest.fixture(autouse=True)
def _reset_fallback_counters() -> None:
    """The fallback counters are process-global (module state in
    fallback_openrouter). Reset all three between tests so cross-test pollution
    can't silently turn an "alertable=0" path into "alertable=1" because a
    prior test bumped a counter.

    The donated-key probe verdict is process-global for the same reason (probe
    once per run), and cli renders it in the end-of-run summary, so it is reset
    here too. Same for the provider-health observation store, which feeds the
    provider-degradation summand of ``alertable_count``.
    """
    reset_generic_key_fallback_count()
    reset_donated_404_fallback_count()
    reset_credit_key_fallback_count()
    reset_donated_key_state_cache()
    reset_provider_health()
    reset_post_drop_count()


@contextmanager
def _cli_main_test_mode(
    alertable_count: int,
    *,
    donated_below_floor: bool = False,
    fall_cup_reminder: bool = False,
    tournament_stale: bool = False,
    today: date | None = None,
    stub_bot: MagicMock | None = None,
    mode: str = "test_questions",
    only_posts: str | None = None,
) -> Iterator[MagicMock]:
    """Run ``cli.main`` with every external dependency stubbed; yields the CreditTelemetry stub.

    TemplateForecaster becomes a MagicMock with the given ``alertable_count`` whose
    ``forecast_questions`` returns []; CreditTelemetry is stubbed so no test reaches the real
    OpenRouter balance endpoint, with ``donated_below_floor`` as its floor verdict; argv is pinned.

    ``today`` pins the credit-suppression window through cli's own ``credit_alerts_active``
    reference, and ``None`` leaves the real clock (the production path). ``stub_bot`` replaces the
    whole bot, for tests needing ``alertable_count`` COMPUTED through the real property chain (see
    ``_bot_with_real_alertable_count``), and makes the ``alertable_count`` argument moot. ``mode``
    (default ``test_questions``, the cheapest path through ``_question_source``) and ``only_posts``
    go onto argv; ``fall_cup_reminder`` and ``tournament_stale`` pin verdicts read off the prod clock.
    """
    if stub_bot is None:
        stub_bot = MagicMock()
        stub_bot.alertable_count = alertable_count
    stub_bot.forecast_questions = AsyncMock(return_value=[])
    stub_bot.forecast_on_tournament = AsyncMock(return_value=[])

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
            patch("metaculus_bot.cli.TemplateForecaster", return_value=stub_bot),
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
            patch.object(type(stub_bot), "log_report_summary", create=True, return_value=None),
            patch("metaculus_bot.cli.CreditTelemetry", return_value=stub_telemetry),
            # The real install leaks a RoleSpendTracker into litellm's process-global callbacks for the session.
            patch("metaculus_bot.cli.install_role_spend_tracker"),
        ):
            yield stub_telemetry
    finally:
        sys.argv = argv_backup


class TestCliExitStatus:
    def test_alertable_count_zero_returns_normally(self) -> None:
        """Zero degradation events → no SystemExit; main returns normally."""
        with _cli_main_test_mode(alertable_count=0):
            # Must NOT raise SystemExit.
            cli_main()

    def test_alertable_count_nonzero_triggers_sys_exit_1(self) -> None:
        """Non-zero degradation counter → SystemExit with code 1."""
        with _cli_main_test_mode(alertable_count=1):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1

    def test_large_alertable_count_still_exits_with_code_1(self) -> None:
        """Exit code is always 1 regardless of how many events occurred —
        documents that we use exit-code-1 as a binary alert, not as an
        event count.
        """
        with _cli_main_test_mode(alertable_count=42):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1

    def test_generic_key_fallback_alone_triggers_sys_exit_1(self) -> None:
        """The donated->personal key fallback counter is folded into alertable.

        Even when the bot's own ``alertable_count`` is 0, a single fallback to
        the personal (paid) key during the run must still trigger a non-zero
        exit. The semantics: the run completed all submissions successfully
        (via the paid key), but a call that should have hit the free donated
        key billed to the operator instead, and the operator deserves an email.
        """
        # cli.main reads this generic (non-404) fallback AFTER the forecast returns.
        fb_module._generic_key_fallback_count = 1
        try:
            with _cli_main_test_mode(alertable_count=0):
                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
        finally:
            # The autouse fixture resets too; explicit here because this path bypasses normal flow.
            fb_module._generic_key_fallback_count = 0

    def test_donated_404_fallback_triggers_sys_exit_without_double_counting(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 404 fallback bumps BOTH counters (the wrapper's real behavior), but
        ``alertable`` adds only the generic total — the 404 subset is NOT added
        again. With bot alertable 0 and one 404 fallback, alertable must be 1
        (not 2), and a single fallback still triggers the non-zero exit.

        The exit code alone can't distinguish the correct (alertable==1) from the
        double-count bug (alertable==2): cli.main does an unconditional
        ``sys.exit(1)`` whenever ``alertable > 0``. So we assert against the
        WARNING log line, whose first ``%d`` is the rendered ``alertable`` count —
        "with 1 alertable" under correct wiring, "with 2 alertable" under the
        regression. This is the only test that actually pins the no-double-count
        invariant (the diff's headline correctness claim).
        """
        # Mirror FallbackOpenRouterLlm.invoke: a 404 bumps the generic counter AND the 404 subset.
        fb_module._generic_key_fallback_count = 1
        fb_module._donated_404_fallback_count = 1
        try:
            with (
                _cli_main_test_mode(alertable_count=0),
                caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
                # Pins alertable == 1 (not 2): the count is the first %d in the end-of-run warning.
                assert any("with 1 alertable" in record.getMessage() for record in caplog.records), (
                    f"expected 'with 1 alertable' in warnings; got: {[r.getMessage() for r in caplog.records]}"
                )
        finally:
            fb_module._generic_key_fallback_count = 0
            fb_module._donated_404_fallback_count = 0

    def test_no_fallback_with_bot_alertable_zero_returns_normally(self) -> None:
        """Both bot alertable_count == 0 AND fallback counters == 0 → no SystemExit.

        Pins the conjunction: the autouse fixture resets both counters,
        and main returns normally when nothing was alertable.
        """
        with _cli_main_test_mode(alertable_count=0):
            # Must NOT raise SystemExit.
            cli_main()

    def test_a_dropped_mantic_post_alone_triggers_sys_exit_1(self, caplog: pytest.LogCaptureFixture) -> None:
        """The Mantic parse-drop counter (mantic.py) is folded into ``alertable`` like the key fallback.

        The framework's per-post loop swallows a parse failure as a warning, so the counter is the
        only thing that turns a forfeited post into a red run. The breakdown names the term so a
        reader can see why ``alertable`` is 1 with bot=0 and no key fallback; the counter itself is
        read through ``get_post_drop_count`` because ``_configure_process`` resets it at startup,
        before the fetch that bumps it in prod.
        """
        with (
            _cli_main_test_mode(alertable_count=0),
            patch("metaculus_bot.cli.get_post_drop_count", return_value=1),
            caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1
        [summary] = [r.getMessage() for r in caplog.records if "alertable degradation event" in r.getMessage()]
        assert "with 1 alertable" in summary
        assert summary.endswith("credit=0, mantic_post_drops=1); exiting non-zero so CI marks this run red."), summary

    def test_the_mantic_drop_term_is_absent_when_nothing_was_dropped(self, caplog: pytest.LogCaptureFixture) -> None:
        """Rendered only when it applies, so the registry's optional group and the reader agree."""
        with _cli_main_test_mode(alertable_count=0), caplog.at_level(logging.INFO, logger="metaculus_bot.cli"):
            cli_main()
        assert "mantic_post_drops" not in caplog.text


class TestCliCreditFloor:
    """End-of-run donated-key credit-floor wiring in cli.main.

    The floor check itself (thresholds, n/a handling, fetch failures) is unit
    tested in test_credit_telemetry.py; these tests pin the cli wiring — that
    the boolean returned by ``log_end_and_check_floor`` drives the exit code,
    and that both telemetry phases run even when forecasting crashes.

    The breach→exit link is now gated on the credit-alert window, so every test
    that asserts an exit pins ``today`` on or past the resume date. The
    suppressed side lives in ``TestCliCreditAlertSuppression``.
    """

    def test_below_floor_triggers_sys_exit_1(self) -> None:
        """Donated balance below floor → run completes, then SystemExit(1)."""
        with _cli_main_test_mode(alertable_count=0, donated_below_floor=True, today=AFTER_RESUME_DATE) as telemetry:
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1
            telemetry.log_start.assert_called_once()
            telemetry.log_end_and_check_floor.assert_called_once()

    def test_above_floor_returns_normally(self) -> None:
        """Healthy balance → telemetry logs both phases, no SystemExit."""
        with _cli_main_test_mode(alertable_count=0, donated_below_floor=False, today=AFTER_RESUME_DATE) as telemetry:
            cli_main()
            telemetry.log_start.assert_called_once()
            telemetry.log_end_and_check_floor.assert_called_once()

    def test_end_telemetry_runs_when_forecasting_crashes(self) -> None:
        """The end-of-run fetch is in a finally: a crashed run still logs its
        spend (the original exception propagates, not a floor SystemExit).
        """
        with _cli_main_test_mode(alertable_count=0, donated_below_floor=True, today=AFTER_RESUME_DATE) as telemetry:

            def _crash(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("forecasting blew up")

            with (
                patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_crash)),
                pytest.raises(RuntimeError, match="forecasting blew up"),
            ):
                cli_main()
            telemetry.log_start.assert_called_once()
            telemetry.log_end_and_check_floor.assert_called_once()


class TestCliRoleSpendWiring:
    """cli.main installs the CREDIT_ROLE_SPEND tracker before the first completion and logs
    the ledger from the same ``finally`` as the balance telemetry, so a crashed run still
    reports where its money went. The ledger itself is unit tested in
    test_credit_telemetry.py; these pin the wiring.
    """

    def test_tracker_installed_and_ledger_logged_on_a_clean_run(self) -> None:
        with (
            _cli_main_test_mode(alertable_count=0),
            patch("metaculus_bot.cli.install_role_spend_tracker") as install,
            patch("metaculus_bot.cli.log_role_spend") as log_roles,
        ):
            cli_main()
        install.assert_called_once_with()
        log_roles.assert_called_once_with()

    def test_ledger_logged_when_forecasting_crashes(self) -> None:
        def _crash(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("forecasting blew up")

        with (
            _cli_main_test_mode(alertable_count=0),
            patch("metaculus_bot.cli.log_role_spend") as log_roles,
            patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_crash)),
            pytest.raises(RuntimeError, match="forecasting blew up"),
        ):
            cli_main()
        log_roles.assert_called_once_with()

    def test_driving_cli_main_leaves_no_tracker_in_litellms_globals(self) -> None:
        """The harness must not leak the process-global callback the real install adds.

        ``install_role_spend_tracker`` is stubbed in ``_cli_main_test_mode`` for exactly
        this reason; without the stub a run of this suite left a live RoleSpendTracker
        registered for every later test in the session.
        """
        before = sum(isinstance(cb, RoleSpendTracker) for cb in litellm.callbacks)
        with _cli_main_test_mode(alertable_count=0):
            cli_main()
        assert sum(isinstance(cb, RoleSpendTracker) for cb in litellm.callbacks) == before

    async def test_forecast_wrapper_drains_callbacks_after_the_forecast(self) -> None:
        """The drain has to run INSIDE the forecast loop (the litellm logging worker's queue is
        bound to it), after the forecast, and still run when the forecast raises."""
        drained = AsyncMock()

        async def _forecast() -> list[Any]:
            return ["report"]

        with patch("metaculus_bot.cli.drain_litellm_callbacks", drained):
            assert await _forecast_with_callback_drain(_forecast) == ["report"]
        drained.assert_awaited_once_with()

        async def _boom() -> list[Any]:
            raise RuntimeError("forecasting blew up")

        drained.reset_mock()
        with patch("metaculus_bot.cli.drain_litellm_callbacks", drained), pytest.raises(RuntimeError):
            await _forecast_with_callback_drain(_boom)
        drained.assert_awaited_once_with()

    @pytest.mark.parametrize("run_mode", get_args(RunMode))
    def test_every_run_mode_forecasts_through_the_callback_drain(self, run_mode: RunMode) -> None:
        """Every mode goes through the one ``asyncio.run`` + drain in ``_run_forecasts``.

        The drain used to be applied per mode, four times over, so a mode added without it
        would still forecast and publish normally while silently reporting no per-role
        spend. ``_question_source`` now hands back a factory and cannot run a loop of its
        own, which is what this pins: the drain is awaited exactly once per mode.
        """
        bot = MagicMock()
        bot.forecast_questions = AsyncMock(return_value=["report"])
        bot.forecast_on_tournament = AsyncMock(return_value=["report"])
        drained = AsyncMock()

        with (
            patch("metaculus_bot.cli.MetaculusApi", MagicMock()),
            patch("metaculus_bot.cli.drain_litellm_callbacks", drained),
        ):
            assert _run_forecasts(bot, run_mode) == ["report"]

        drained.assert_awaited_once_with()

    def test_an_unknown_run_mode_raises_before_any_spend(self) -> None:
        """The invalid-mode guard has to fire while resolving the source, i.e. before the
        loop starts and before any question is fetched."""
        with pytest.raises(ValueError, match="Invalid run mode"):
            _run_forecasts(MagicMock(), "not_a_mode")  # type: ignore[arg-type]

    def test_a_forecast_failure_propagates_out_of_run_forecasts_unchanged(self) -> None:
        """A forecast exception keeps its own type through the consolidated dispatch, and the
        drain still runs — main's finally and the emit-then-raise block depend on both.

        Drives the real ``asyncio.run`` and the real wrapper (only the drain is stubbed), so
        this exercises the propagation path rather than a patched ``asyncio.run``.
        """
        bot = MagicMock()
        bot.forecast_on_tournament = AsyncMock(side_effect=RuntimeError("forecasting blew up"))
        drained = AsyncMock()

        with (
            patch("metaculus_bot.cli.drain_litellm_callbacks", drained),
            pytest.raises(RuntimeError, match="forecasting blew up"),
        ):
            _run_forecasts(bot, "tournament")

        drained.assert_awaited_once_with()


class TestCliFallCupReminderExit:
    """The fall-cup reminder reddens the run the same way the credit floor does.

    The check itself (date gate, FALL_CUP_CONFIGURED flip, log content) is covered in
    test_tournament_dates.py; here the harness pins its verdict and these tests pin
    the exit wiring: run completes and publishes first, exits non-zero after, and the
    run is never stamped clean.
    """

    def test_reminder_fires_run_completes_then_sys_exit_1(self, caplog: pytest.LogCaptureFixture) -> None:
        with (
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
            _cli_main_test_mode(alertable_count=0, fall_cup_reminder=True) as telemetry,
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1
            # Forecasting/telemetry completed before the exit — reminder, not abort.
            telemetry.log_start.assert_called_once()
            telemetry.log_end_and_check_floor.assert_called_once()
        # run_clean is the exact complement of every non-zero exit, so a reminder run must not stamp it.
        assert "Run completed clean" not in caplog.text

    def test_no_reminder_returns_normally(self) -> None:
        with _cli_main_test_mode(alertable_count=0, fall_cup_reminder=False):
            cli_main()


class TestCliManticStaleSlugExit:
    """A Mantic slug past its end date reddens the run the way the fall-cup reminder does.

    ``check_tournament_dates`` warns for TOURNAMENT_HARD_STOP_WEEKS before it raises, and a
    zero-question run is green, so once Preseason 2 closes every scheduled run would stay green
    and silent for two weeks while a Series 2 slug went unforecast (edge review item 5). The
    verdict is held at startup and turned into a non-zero exit AFTER publishing; the shared hard
    stop is untouched, and the Metaculus tournament keeps the warning advisory.
    """

    def test_a_stale_mantic_slug_publishes_first_then_exits_non_zero(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        _mantic_env(monkeypatch)
        with (
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
            _cli_main_test_mode(alertable_count=0, mode="mantic", tournament_stale=True) as telemetry,
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1
            telemetry.log_start.assert_called_once()
            telemetry.log_end_and_check_floor.assert_called_once()
        assert "Run completed clean" not in caplog.text
        assert "re-point MANTIC_TOURNAMENT_ID" in caplog.text

    def test_a_current_mantic_slug_returns_normally(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mantic_env(monkeypatch)
        with _cli_main_test_mode(alertable_count=0, mode="mantic", tournament_stale=False):
            cli_main()

    def test_a_stale_metaculus_tournament_stays_advisory(self) -> None:
        with _cli_main_test_mode(alertable_count=0, mode="tournament", tournament_stale=True):
            cli_main()


class TestCliResearchFlush:
    """The research batch reaches disk on BOTH exit paths.

    Records accumulate in memory for the whole run and are written once at the end, so
    before the flush moved inside the ``finally`` any exception escaping ``asyncio.run``
    — an OSError, the invalid-run-mode ValueError, a ``timeout-minutes`` SIGTERM —
    discarded every question's research. A 40-question tournament run that died on the
    last question archived nothing, and since GHA deletes artifacts at 90 days that hole
    is permanent. The same ``finally`` already protected the credit telemetry, which is
    what made the omission easy to miss.

    Both tests drive the REAL ``ResearchPersistenceWriter`` through the sink cli hands the
    forecaster, so they cover the whole write path (sink -> accumulate -> flush -> JSONL)
    rather than asserting that a mock got called.
    """

    @staticmethod
    def _forecaster_class() -> MagicMock:
        """A ``TemplateForecaster`` class stub that also exposes the constructor kwargs.

        The helper's own patch discards its mock, and these tests need the
        ``research_sink`` cli built and passed in. ``alertable_count`` is pinned to a real
        int because the normal-path test runs off the end of ``main``, into the
        ``alertable > 0`` comparison.
        """
        forecaster_class = MagicMock()
        forecaster_class.return_value.alertable_count = 0
        return forecaster_class

    @staticmethod
    def _record_two(sink: Callable[..., None]) -> None:
        """Record two questions' research through cli's own sink callback."""
        for qid in (43613, 50001):
            sink(
                qid=qid,
                page_url=f"https://www.metaculus.com/questions/{qid}/",
                question_text=f"Question {qid}?",
                research_text=f"## News Articles (AskNews)\nResearch for {qid}.",
                providers_used=["asknews"],
                gap_fill_used=False,
            )

    @staticmethod
    def _flushed_records(tmp_path: Path) -> list[dict]:
        """Every record in the JSONL the writer flushed into ``research_outputs/``."""
        written = sorted((tmp_path / "research_outputs").glob("research_*.jsonl"))
        assert len(written) == 1, f"expected exactly one flushed JSONL, got {written}"
        return [json.loads(line) for line in written[0].read_text().strip().splitlines()]

    def test_flush_runs_when_the_forecast_loop_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PERSIST_RESEARCH_ENABLED_ENV, "true")
        monkeypatch.chdir(tmp_path)  # writer.flush() writes research_outputs/ under CWD

        forecaster_class = self._forecaster_class()

        def _record_then_crash(*_args: object, **_kwargs: object) -> None:
            """Two questions researched, then the run dies before returning: the shape that lost the batch."""
            self._record_two(forecaster_class.call_args.kwargs["research_sink"])
            raise RuntimeError("forecast loop blew up")

        with (
            _cli_main_test_mode(alertable_count=0),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_record_then_crash)),
            pytest.raises(RuntimeError, match="forecast loop blew up"),
        ):
            # The original exception must still propagate: the flush is a rescue, not a swallow.
            cli_main()

        assert [r["qid"] for r in self._flushed_records(tmp_path)] == [43613, 50001]

    def test_flush_still_runs_on_the_normal_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PERSIST_RESEARCH_ENABLED_ENV, "true")
        monkeypatch.chdir(tmp_path)

        forecaster_class = self._forecaster_class()

        def _record_then_return(*_args: object, **_kwargs: object) -> list[object]:
            self._record_two(forecaster_class.call_args.kwargs["research_sink"])
            return []

        with (
            _cli_main_test_mode(alertable_count=0),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_record_then_return)),
        ):
            cli_main()

        assert [r["qid"] for r in self._flushed_records(tmp_path)] == [43613, 50001]

    def test_nothing_is_written_when_the_flag_is_off(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """No writer, no sink: the forecaster is handed None, so the finally-block flush stays guarded."""
        monkeypatch.delenv(PERSIST_RESEARCH_ENABLED_ENV, raising=False)
        monkeypatch.chdir(tmp_path)

        def _crash(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("boom")

        forecaster_class = self._forecaster_class()
        with (
            _cli_main_test_mode(alertable_count=0),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_crash)),
            pytest.raises(RuntimeError, match="boom"),
        ):
            cli_main()

        assert forecaster_class.call_args.kwargs["research_sink"] is None
        assert not (tmp_path / "research_outputs").exists()


class TestPersistedTournamentId:
    """The research archive's ``tournament_id`` label follows the RUN MODE.

    Until 2026-09-03 cli stamped ``TOURNAMENT_ID`` on every record whatever mode was
    running, so enabling the Metaculus Cup workflow would have filed cup questions under
    the bot tournament's slug — inside its config-era buckets and inside the supply probe's
    per-slug rows. That is silent data corruption, not a cosmetic label: nothing else on the
    record says which competition a question came from, since ``run_mode`` names the
    pipeline rather than the object.
    """

    EXPECTED_LABEL: ClassVar[dict[str, str]] = {
        "tournament": TOURNAMENT_ID,
        "minibench": str(MetaculusApi.CURRENT_MINIBENCH_ID),
        "quarterly_cup": METACULUS_CUP_ID,
        "metaculus_cup": METACULUS_CUP_ID,
        "mantic": MANTIC_TOURNAMENT_ID,
        # No label fits the evergreen set; retained so the archive's existing test-run records stay comparable.
        "test_questions": TOURNAMENT_ID,
    }

    def test_every_run_mode_has_a_decided_label(self) -> None:
        """Derived from RunMode, so a mode added without a decision fails here instead of inheriting a slug."""
        assert set(get_args(RunMode)) == set(self.EXPECTED_LABEL)

    @pytest.mark.parametrize(("run_mode", "expected"), sorted(EXPECTED_LABEL.items()))
    def test_label_per_run_mode(self, run_mode: RunMode, expected: str) -> None:
        assert persisted_tournament_id(run_mode) == expected

    def test_the_competitions_do_not_share_a_label(self) -> None:
        """The bot tournament, the cup and the Mantic tournament must stay distinguishable in the archive."""
        labels = {persisted_tournament_id(run_mode) for run_mode in ("tournament", "metaculus_cup", "mantic")}
        assert len(labels) == 3, labels

    def test_an_unknown_mode_raises_rather_than_mislabelling(self) -> None:
        with pytest.raises(ValueError, match="Invalid run mode"):
            persisted_tournament_id("world_cup")  # type: ignore[arg-type]  # deliberately outside RunMode

    def test_a_cup_run_archives_its_records_under_the_cup_slug(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End to end through cli's own writer, not just the helper: mode -> label -> JSONL."""
        monkeypatch.setenv(PERSIST_RESEARCH_ENABLED_ENV, "true")
        monkeypatch.chdir(tmp_path)  # writer.flush() writes research_outputs/ under CWD

        forecaster_class = MagicMock()
        forecaster_class.return_value.alertable_count = 0

        def _record_then_return(*_args: object, **_kwargs: object) -> list[object]:
            forecaster_class.call_args.kwargs["research_sink"](
                qid=45500,
                page_url="https://www.metaculus.com/questions/45500/",
                question_text="A fall cup question?",
                research_text="## News Articles (AskNews)\nResearch for 45500.",
                providers_used=["asknews"],
                gap_fill_used=False,
            )
            return []

        with (
            _cli_main_test_mode(alertable_count=0, mode="metaculus_cup"),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_record_then_return)),
        ):
            cli_main()

        written = sorted((tmp_path / "research_outputs").glob("research_*.jsonl"))
        assert len(written) == 1, f"expected exactly one flushed JSONL, got {written}"
        records = [json.loads(line) for line in written[0].read_text().strip().splitlines()]
        assert [(r["run_mode"], r["tournament_id"], r["platform"]) for r in records] == [
            ("metaculus_cup", METACULUS_CUP_ID, "metaculus")
        ]

    def test_a_mantic_run_archives_its_records_under_the_mantic_slug_and_platform(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same end-to-end shape for the Mantic mode: the Mantic slug, and ``platform`` set to
        ``mantic`` so a post id in the 600s is never read as a Metaculus id."""
        _mantic_env(monkeypatch)
        monkeypatch.setenv(PERSIST_RESEARCH_ENABLED_ENV, "true")
        monkeypatch.chdir(tmp_path)

        forecaster_class = MagicMock()
        forecaster_class.return_value.alertable_count = 0

        def _record_then_return(*_args: object, **_kwargs: object) -> list[object]:
            forecaster_class.call_args.kwargs["research_sink"](
                qid=650,
                page_url="https://competitions.mantic.com/questions/650/",
                question_text="What will the price of bitcoin be?",
                research_text="## News Articles (AskNews)\nResearch for 650.",
                providers_used=["asknews"],
                gap_fill_used=False,
            )
            return []

        with (
            _cli_main_test_mode(alertable_count=0, mode="mantic"),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_record_then_return)),
        ):
            cli_main()

        written = sorted((tmp_path / "research_outputs").glob("research_*.jsonl"))
        assert len(written) == 1, f"expected exactly one flushed JSONL, got {written}"
        records = [json.loads(line) for line in written[0].read_text().strip().splitlines()]
        assert [(r["run_mode"], r["tournament_id"], r["platform"]) for r in records] == [
            ("mantic", MANTIC_TOURNAMENT_ID, "mantic")
        ]


class TestPersistedPlatform:
    """The archive's additive ``platform`` field: which question platform a record's ids belong to.

    Filenames are not namespaced and the archive groups on the bare qid, so this field is what
    separates the two platforms' records and what an analysis keyed on bare post ids must
    filter on; the id gap today runs from the Mantic counter (~650) to 14333, the next
    Metaculus key.
    """

    def test_mantic_mode_is_the_only_mantic_platform(self) -> None:
        assert persisted_platform("mantic") == "mantic"
        for run_mode in set(get_args(RunMode)) - {"mantic"}:
            assert persisted_platform(run_mode) == "metaculus", run_mode


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


class TestAssertPersonalKeysOnly:
    """The fail-shut guard for Mantic runs. Metaculus donated ``OAI_ANTH_OPENROUTER_KEY`` for its own
    tournaments, so a run that forecasts for Mantic may spend only personal keys; the switch has to
    be off in the environment before the process starts, because the roster freezes its OpenRouter
    key at import (``llm_configs``) and ``main.py`` imports it before ``cli.main`` runs."""

    @pytest.mark.parametrize("switch", [None, "true", "1", ""])
    def test_raises_naming_the_switch_unless_it_reads_false(
        self, switch: str | None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if switch is None:
            monkeypatch.delenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, raising=False)
        else:
            monkeypatch.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, switch)
        with pytest.raises(RuntimeError, match=DONATED_OPENROUTER_KEY_ENABLED_ENV):
            _assert_personal_keys_only()

    def test_passes_when_the_switch_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, "false")
        _assert_personal_keys_only()


class TestConfigureProcess:
    """The run mode decides which identity preflight runs. A Mantic run vets the Mantic API host and
    never contacts metaculus.com (it must not depend on Metaculus DNS health), after failing shut on
    the donated-key switch; every Metaculus mode is unchanged. The hardening patches are mode-blind."""

    METACULUS_MODES: ClassVar[list[str]] = sorted(set(get_args(RunMode)) - {"mantic"})

    @pytest.mark.parametrize("run_mode", METACULUS_MODES)
    def test_metaculus_modes_preflight_metaculus_only(self, run_mode: RunMode, monkeypatch: pytest.MonkeyPatch) -> None:
        """The switch's default (donated key on) IS the Metaculus production state, so it must not raise."""
        monkeypatch.delenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, raising=False)
        with _configure_process_stubs() as stubs:
            _configure_process(run_mode)
        stubs["verify_metaculus_api_identity"].assert_called_once_with()
        stubs["verify_api_identity"].assert_not_called()

    def test_mantic_mode_preflights_the_mantic_host_and_never_metaculus(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mantic_env(monkeypatch)
        with _configure_process_stubs() as stubs:
            _configure_process("mantic")
        stubs["verify_api_identity"].assert_called_once_with(MANTIC_API_BASE_URL)
        stubs["verify_metaculus_api_identity"].assert_not_called()

    def test_mantic_mode_fails_shut_before_any_preflight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, raising=False)
        with (
            _configure_process_stubs() as stubs,
            pytest.raises(RuntimeError, match=DONATED_OPENROUTER_KEY_ENABLED_ENV),
        ):
            _configure_process("mantic")
        stubs["verify_api_identity"].assert_not_called()
        stubs["verify_metaculus_api_identity"].assert_not_called()

    @pytest.mark.parametrize("run_mode", get_args(RunMode))
    def test_hardening_is_installed_in_every_mode(self, run_mode: RunMode, monkeypatch: pytest.MonkeyPatch) -> None:
        _mantic_env(monkeypatch)
        with _configure_process_stubs() as stubs:
            _configure_process(run_mode)
        stubs["publish_hardening"].assert_called_once_with()
        stubs["fetch_hardening"].assert_called_once_with()

    @pytest.mark.parametrize("run_mode", get_args(RunMode))
    def test_the_mantic_parse_drop_counter_is_reset_at_startup(
        self, run_mode: RunMode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reset here rather than in forecast_questions, whose resets run AFTER the fetch that bumps it."""
        _mantic_env(monkeypatch)
        mantic_module._post_drop_count = 3
        with _configure_process_stubs():
            _configure_process(run_mode)
        assert mantic_module.get_post_drop_count() == 0


class TestManticQuestionSource:
    """``--mode mantic`` mirrors the tournament mode over the Mantic slug: the re-spend guard on and
    the forecast over ``MANTIC_TOURNAMENT_ID``. The stale-date check runs at startup (next class)."""

    def test_mantic_mode_forecasts_the_mantic_tournament(self) -> None:
        bot = MagicMock()
        bot.skip_previously_forecasted_questions = False
        bot.forecast_on_tournament = AsyncMock(return_value=["report"])

        with patch("metaculus_bot.cli.drain_litellm_callbacks", AsyncMock()):
            assert _run_forecasts(bot, "mantic") == ["report"]

        bot.forecast_on_tournament.assert_awaited_once_with(MANTIC_TOURNAMENT_ID, return_exceptions=True)
        assert bot.skip_previously_forecasted_questions is True


class TestTournamentDateCheck:
    """The stale-slug check runs once at startup, per mode, and only the Mantic verdict is held.

    ``check_tournament_dates`` warns for TOURNAMENT_HARD_STOP_WEEKS before it raises, and a
    zero-question run is green, so on Mantic the fortnight between the two was a silent forfeit
    of every Series 2 question (edge review item 5). The Metaculus tournament keeps the warning
    advisory: its questions are open for weeks. ``TestCliManticStaleSlugExit`` pins what main
    does with the verdict.
    """

    def test_mantic_mode_checks_the_mantic_dates_and_returns_the_verdict(self) -> None:
        with patch("metaculus_bot.cli.check_tournament_dates", return_value=True) as check_dates:
            assert _check_tournament_dates("mantic") is True
        check_dates.assert_called_once_with(
            logging.getLogger("metaculus_bot.cli"),
            tournament_id=MANTIC_TOURNAMENT_ID,
            end_date_str=MANTIC_TOURNAMENT_END_DATE,
        )

    def test_tournament_mode_checks_the_metaculus_dates_but_stays_advisory(self) -> None:
        with patch("metaculus_bot.cli.check_tournament_dates", return_value=True) as check_dates:
            assert _check_tournament_dates("tournament") is False
        check_dates.assert_called_once_with(logging.getLogger("metaculus_bot.cli"))

    @pytest.mark.parametrize("run_mode", ["minibench", "quarterly_cup", "metaculus_cup", "test_questions"])
    def test_undated_modes_check_nothing(self, run_mode: RunMode) -> None:
        with patch("metaculus_bot.cli.check_tournament_dates") as check_dates:
            assert _check_tournament_dates(run_mode) is False
        check_dates.assert_not_called()

    def test_the_shared_hard_stop_still_raises_before_the_forecaster_is_built(self) -> None:
        forecaster_class = MagicMock()
        with (
            _cli_main_test_mode(alertable_count=0, mode="tournament"),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli.check_tournament_dates", side_effect=TournamentExpiredError("stale")),
            pytest.raises(TournamentExpiredError),
        ):
            cli_main()
        forecaster_class.assert_not_called()


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


class TestOnlyPostsFilter:
    """``--only-posts`` narrows a tournament-shaped run to the listed post ids: the one-question
    paid smoke run. The question-list fetch is the one ``forecast_on_tournament`` makes, on the
    same injected client, and only the matching questions reach ``forecast_questions``. Without
    the flag every mode still forecasts through ``forecast_on_tournament`` exactly as before.
    """

    TOURNAMENT_SLUGS: ClassVar[dict[str, int | str]] = {
        "tournament": TOURNAMENT_ID,
        "minibench": MetaculusApi.CURRENT_MINIBENCH_ID,
        "quarterly_cup": METACULUS_CUP_ID,
        "metaculus_cup": METACULUS_CUP_ID,
        "mantic": MANTIC_TOURNAMENT_ID,
    }

    def test_every_tournament_shaped_mode_is_covered(self) -> None:
        """Derived from RunMode, so a new tournament-shaped mode fails here until it is listed."""
        assert set(self.TOURNAMENT_SLUGS) == set(get_args(RunMode)) - {"test_questions"}

    @pytest.mark.parametrize("run_mode", sorted(TOURNAMENT_SLUGS))
    def test_only_the_requested_posts_reach_forecast_questions(
        self, run_mode: RunMode, caplog: pytest.LogCaptureFixture
    ) -> None:
        bot = _filterable_bot(648, 650, 651, 652)
        open_questions = bot.metaculus_client.get_all_open_questions_from_tournament.return_value

        with (
            patch("metaculus_bot.cli.check_tournament_dates"),
            patch("metaculus_bot.cli.drain_litellm_callbacks", AsyncMock()),
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
        ):
            assert _run_forecasts(bot, run_mode, only_posts=frozenset({652, 650})) == ["report"]

        bot.metaculus_client.get_all_open_questions_from_tournament.assert_called_once_with(
            self.TOURNAMENT_SLUGS[run_mode]
        )
        # Fetch order, not request order, and nothing but the two asked for.
        bot.forecast_questions.assert_awaited_once_with([open_questions[1], open_questions[3]], return_exceptions=True)
        bot.forecast_on_tournament.assert_not_awaited()
        # The re-spend guard is pinned on exactly as in an unfiltered run.
        assert bot.skip_previously_forecasted_questions is True
        assert "ONLY_POSTS: requested=650,652 matched=650,652 dropped=2" in caplog.messages

    def test_the_marker_line_is_the_one_the_harvester_reads(self, caplog: pytest.LogCaptureFixture) -> None:
        """Seam pin: the archive keys off the exact spelling (scripts/telemetry/markers.py)."""
        bot = _filterable_bot(649, 650, 651)

        with (
            patch("metaculus_bot.cli.check_tournament_dates"),
            patch("metaculus_bot.cli.drain_litellm_callbacks", AsyncMock()),
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
        ):
            _run_forecasts(bot, "mantic", only_posts=frozenset({650}))

        marker_line = next(message for message in caplog.messages if message.startswith("ONLY_POSTS:"))
        spec = next(s for s in MARKER_SPECS if s.name == "only_posts")
        match = spec.regex.search(marker_line)
        assert match is not None, marker_line
        assert match.group("requested") == "650"
        assert match.group("matched") == "650"
        assert match.group("dropped") == "2"

    def test_an_unmatched_filter_warns_and_forecasts_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        """No match forecasts nothing, never the whole tournament, and says so at WARNING."""
        bot = _filterable_bot(650)

        with (
            patch("metaculus_bot.cli.check_tournament_dates"),
            patch("metaculus_bot.cli.drain_litellm_callbacks", AsyncMock()),
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
        ):
            assert _run_forecasts(bot, "mantic", only_posts=frozenset({999})) == []

        bot.forecast_questions.assert_not_awaited()
        bot.forecast_on_tournament.assert_not_awaited()
        assert "ONLY_POSTS: requested=999 matched=none dropped=1" in caplog.messages
        warning = next(record for record in caplog.records if record.levelno == logging.WARNING)
        assert "--only-posts matched none of the 1 open question(s)" in warning.getMessage()

    @pytest.mark.parametrize("run_mode", sorted(TOURNAMENT_SLUGS))
    def test_without_the_filter_every_mode_still_forecasts_on_tournament(self, run_mode: RunMode) -> None:
        bot = _filterable_bot(650)

        with (
            patch("metaculus_bot.cli.check_tournament_dates"),
            patch("metaculus_bot.cli.drain_litellm_callbacks", AsyncMock()),
        ):
            assert _run_forecasts(bot, run_mode) == ["report"]

        bot.forecast_on_tournament.assert_awaited_once_with(self.TOURNAMENT_SLUGS[run_mode], return_exceptions=True)
        bot.metaculus_client.get_all_open_questions_from_tournament.assert_not_called()
        bot.forecast_questions.assert_not_awaited()

    def test_the_flag_reaches_the_filter_through_main(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The whole argv path: ``--only-posts 650`` on a mantic run forecasts post 650 alone."""
        _mantic_env(monkeypatch)
        stub_bot = _filterable_bot(649, 650, 651)
        stub_bot.alertable_count = 0
        open_questions = stub_bot.metaculus_client.get_all_open_questions_from_tournament.return_value

        with (
            _cli_main_test_mode(alertable_count=0, stub_bot=stub_bot, mode="mantic", only_posts="650"),
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
        ):
            cli_main()

        stub_bot.metaculus_client.get_all_open_questions_from_tournament.assert_called_once_with(MANTIC_TOURNAMENT_ID)
        stub_bot.forecast_questions.assert_awaited_once_with([open_questions[1]], return_exceptions=True)
        stub_bot.forecast_on_tournament.assert_not_awaited()
        assert "ONLY_POSTS: requested=650 matched=650 dropped=2" in caplog.messages

    def test_parses_a_comma_separated_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["cli", "--mode", "mantic", "--only-posts", "650,651"])
        assert _parse_cli_args() == CliArgs(run_mode="mantic", only_posts=frozenset({650, 651}))

    def test_defaults_to_no_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["cli"])
        assert _parse_cli_args() == CliArgs(run_mode="tournament", only_posts=None)

    def test_a_non_integer_id_is_a_usage_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["cli", "--mode", "mantic", "--only-posts", "650,abc"])
        with pytest.raises(SystemExit) as exc_info:
            _parse_cli_args()
        assert exc_info.value.code == 2
        assert "comma-separated integer post ids" in capsys.readouterr().err

    def test_the_filter_is_refused_for_test_questions(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The evergreen set is not a tournament's open questions, so a filter there would be a
        silent no-op on a paid run; the parser refuses it instead."""
        monkeypatch.setattr(sys, "argv", ["cli", "--mode", "test_questions", "--only-posts", "650"])
        with pytest.raises(SystemExit) as exc_info:
            _parse_cli_args()
        assert exc_info.value.code == 2
        assert "--only-posts" in capsys.readouterr().err


class TestManticClientWiring:
    """``main`` hands the framework a ``ManticClient`` in mantic mode and nothing (so the framework
    builds its default Metaculus client) otherwise. The client is built only after the fail-shut
    check and the identity preflight, so the Mantic token is not even read before the host is vetted."""

    @staticmethod
    def _forecaster_class() -> MagicMock:
        """A ``TemplateForecaster`` class stub that keeps the constructor kwargs inspectable (the
        harness's own patch discards its mock) and whose instance forecasts nothing."""
        forecaster_class = MagicMock()
        forecaster_class.return_value.alertable_count = 0
        forecaster_class.return_value.forecast_on_tournament = AsyncMock(return_value=[])
        forecaster_class.return_value.forecast_questions = AsyncMock(return_value=[])
        return forecaster_class

    def test_mantic_mode_injects_a_mantic_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _mantic_env(monkeypatch)
        forecaster_class = self._forecaster_class()
        with (
            _cli_main_test_mode(alertable_count=0, mode="mantic"),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
        ):
            cli_main()
        client = forecaster_class.call_args.kwargs["metaculus_client"]
        assert isinstance(client, ManticClient)
        assert client.base_url == MANTIC_API_BASE_URL

    @pytest.mark.parametrize("run_mode", sorted(set(get_args(RunMode)) - {"mantic"}))
    def test_metaculus_modes_leave_the_framework_default_client(self, run_mode: RunMode) -> None:
        forecaster_class = self._forecaster_class()
        with (
            _cli_main_test_mode(alertable_count=0, mode=run_mode),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
        ):
            cli_main()
        assert forecaster_class.call_args.kwargs["metaculus_client"] is None

    def test_fail_shut_runs_before_the_token_is_read_or_the_host_vetted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The switch at its default: the guard raises, so no preflight GET, token read, forecaster or spend."""
        monkeypatch.delenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, raising=False)
        monkeypatch.setenv(MANTIC_TOKEN_ENV, _FAKE_MANTIC_TOKEN)
        forecaster_class = self._forecaster_class()
        with (
            _cli_main_test_mode(alertable_count=0, mode="mantic"),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli.build_mantic_client") as build_client,
            patch("metaculus_bot.cli.verify_api_identity") as verify_api,
            pytest.raises(RuntimeError, match=DONATED_OPENROUTER_KEY_ENABLED_ENV),
        ):
            cli_main()
        build_client.assert_not_called()
        verify_api.assert_not_called()
        forecaster_class.assert_not_called()

    def test_the_client_is_built_after_the_guard_and_the_preflight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ordered, not just "each was called": guard, then preflight, then the client (which reads
        the token), then the forecaster. Two independent ``assert_called`` checks would pass with
        the token read before the host was vetted."""
        _mantic_env(monkeypatch)
        manager = MagicMock()
        forecaster_class = self._forecaster_class()
        with (
            _cli_main_test_mode(alertable_count=0, mode="mantic"),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli._assert_personal_keys_only") as guard,
            patch("metaculus_bot.cli.verify_api_identity") as verify_api,
            patch("metaculus_bot.cli.build_mantic_client") as build_client,
        ):
            manager.attach_mock(guard, "guard")
            manager.attach_mock(verify_api, "verify_api_identity")
            manager.attach_mock(build_client, "build_mantic_client")
            manager.attach_mock(forecaster_class, "TemplateForecaster")
            cli_main()

        call_names = [name for name, _, _ in manager.mock_calls]
        order = [
            call_names.index(name)
            for name in ("guard", "verify_api_identity", "build_mantic_client", "TemplateForecaster")
        ]
        assert order == sorted(order), call_names

    def test_the_tournament_preflight_runs_on_the_built_client_before_the_forecaster(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ordered like the test above: client built (token read), then the authenticated tournament
        GET on THAT client, then the forecaster. Nothing has been spent when the preflight raises."""
        _mantic_env(monkeypatch)
        manager = MagicMock()
        forecaster_class = self._forecaster_class()
        with (
            _cli_main_test_mode(alertable_count=0, mode="mantic"),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli.build_mantic_client") as build_client,
            patch("metaculus_bot.cli.preflight_mantic_tournaments") as preflight,
        ):
            manager.attach_mock(build_client, "build_mantic_client")
            manager.attach_mock(preflight, "preflight_mantic_tournaments")
            manager.attach_mock(forecaster_class, "TemplateForecaster")
            cli_main()

        call_names = [name for name, _, _ in manager.mock_calls]
        order = [
            call_names.index(name)
            for name in ("build_mantic_client", "preflight_mantic_tournaments", "TemplateForecaster")
        ]
        assert order == sorted(order), call_names
        preflight.assert_called_once_with(build_client.return_value, MANTIC_TOURNAMENT_ID)

    def test_a_failed_tournament_preflight_stops_before_the_forecaster_is_built(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _mantic_env(monkeypatch)
        forecaster_class = self._forecaster_class()
        with (
            _cli_main_test_mode(alertable_count=0, mode="mantic"),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            patch("metaculus_bot.cli.preflight_mantic_tournaments", side_effect=ApiIdentityError("viewer")),
            pytest.raises(ApiIdentityError, match="viewer"),
        ):
            cli_main()
        forecaster_class.assert_not_called()

    @pytest.mark.parametrize("run_mode", sorted(set(get_args(RunMode)) - {"mantic"}))
    def test_metaculus_modes_never_run_the_tournament_preflight(self, run_mode: RunMode) -> None:
        with (
            _cli_main_test_mode(alertable_count=0, mode=run_mode),
            patch("metaculus_bot.cli.preflight_mantic_tournaments") as preflight,
        ):
            cli_main()
        preflight.assert_not_called()


class TestCliCreditAlertSuppression:
    """The dated credit-alert suppression, both paths.

    Path 1 is the floor breach; path 2 is the credit-caused donated->personal
    fallback folded into ``alertable``. Both must stop reddening CI until
    ``CREDIT_ALERT_RESUME_DATE``, and both must behave exactly as before once the
    date passes. Nothing about the logs changes — only the exit status and the
    alertable arithmetic.
    """

    def test_floor_breach_during_suppression_does_not_exit(self, caplog: pytest.LogCaptureFixture) -> None:
        """Path 1, suppressed: the run finishes green, and the log explains why
        so a reader who greps CREDIT_FLOOR_BREACH isn't left guessing.
        """
        with (
            _cli_main_test_mode(alertable_count=0, donated_below_floor=True, today=DURING_SUPPRESSION) as telemetry,
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
        ):
            # Must NOT raise SystemExit.
            cli_main()
            telemetry.log_end_and_check_floor.assert_called_once()

        messages = [record.getMessage() for record in caplog.records]
        expected = f"credit alerting is suppressed until {CREDIT_ALERT_RESUME_DATE.isoformat()}"
        assert any(expected in msg for msg in messages), messages

    def test_floor_breach_with_no_injected_date_exits_non_zero(self) -> None:
        """The live state since 2026-09-03: the real constant, the real clock, no
        injection — a donated-key floor breach reddens CI again. This is the one test
        here that would fail if the resume date were pushed back into the future.
        """
        assert credit_alerts_active() is True
        with _cli_main_test_mode(alertable_count=0, donated_below_floor=True, today=None):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1

    def test_floor_breach_on_resume_date_exits_non_zero(self) -> None:
        """The window is closed-on-the-right: the resume day itself alerts."""
        with _cli_main_test_mode(alertable_count=0, donated_below_floor=True, today=ON_RESUME_DATE):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1

    def test_floor_breach_after_resume_date_exits_non_zero(self) -> None:
        with _cli_main_test_mode(alertable_count=0, donated_below_floor=True, today=AFTER_RESUME_DATE):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1

    def test_credit_fallback_during_suppression_does_not_exit(self) -> None:
        """Path 2, suppressed: a 402 fallback bumps both the generic total and the
        credit subset (mirroring the wrapper), and the subtraction takes
        ``alertable`` back to 0 — the empty-wallet case the operator exempted.
        """
        fb_module._generic_key_fallback_count = 1
        fb_module._credit_key_fallback_count = 1
        try:
            with _cli_main_test_mode(alertable_count=0, today=DURING_SUPPRESSION):
                # Must NOT raise SystemExit.
                cli_main()
        finally:
            fb_module._generic_key_fallback_count = 0
            fb_module._credit_key_fallback_count = 0

    def test_credit_fallback_after_resume_date_exits_non_zero(self, caplog: pytest.LogCaptureFixture) -> None:
        """Same state, past the resume date → the pre-suppression behavior, and the
        summary drops the suppression clause rather than reporting "0 suppressed
        until <a date in the past>".
        """
        fb_module._generic_key_fallback_count = 1
        fb_module._credit_key_fallback_count = 1
        try:
            with (
                _cli_main_test_mode(alertable_count=0, today=AFTER_RESUME_DATE),
                caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
                messages = [record.getMessage() for record in caplog.records]
                assert any("with 1 alertable" in msg for msg in messages), messages
                assert any("credit=1);" in msg for msg in messages), messages
                assert not any("suppressed until" in msg for msg in messages), messages
        finally:
            fb_module._generic_key_fallback_count = 0
            fb_module._credit_key_fallback_count = 0

    @pytest.mark.parametrize("cause", ["401", "429"])
    def test_non_credit_fallback_still_alertable_during_suppression(
        self, cause: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The regression that matters most: the suppression must not swallow real
        breakage. A 401 (invalid/disabled key) or 429 (rate limit) bumps only the
        generic counter, so nothing is subtracted and the run still exits non-zero.

        ``cause`` is only a label — both errors land in the same counter; the
        wrapper-side classification is pinned in test_fallback_openrouter.py.
        """
        fb_module._generic_key_fallback_count = 1
        try:
            with (
                _cli_main_test_mode(alertable_count=0, today=DURING_SUPPRESSION),
                caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
                assert any("with 1 alertable" in record.getMessage() for record in caplog.records), (
                    f"{cause}: expected 'with 1 alertable'; got {[r.getMessage() for r in caplog.records]}"
                )
        finally:
            fb_module._generic_key_fallback_count = 0

    def test_donated_404_still_alertable_and_counted_once_during_suppression(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The double-counting trap. A 404 fallback bumps the generic total and the
        404 subset; the credit subset stays 0, so nothing is subtracted and
        ``alertable`` is exactly 1 — not 0 (over-subtracted) and not 2 (added
        twice). The rendered count in the WARNING is the only way to see this.
        """
        fb_module._generic_key_fallback_count = 1
        fb_module._donated_404_fallback_count = 1
        try:
            with (
                _cli_main_test_mode(alertable_count=0, today=DURING_SUPPRESSION),
                caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
                assert any("with 1 alertable" in record.getMessage() for record in caplog.records), (
                    f"expected 'with 1 alertable'; got {[r.getMessage() for r in caplog.records]}"
                )
        finally:
            fb_module._generic_key_fallback_count = 0
            fb_module._donated_404_fallback_count = 0

    def test_mixed_causes_subtract_only_the_credit_share(self, caplog: pytest.LogCaptureFixture) -> None:
        """One 402 plus one 404 in the same run: generic=2, credit=1, donated_404=1.
        Only the credit event is exempt, so alertable is 1 and the run still exits
        non-zero on the 404's account.
        """
        fb_module._generic_key_fallback_count = 2
        fb_module._credit_key_fallback_count = 1
        fb_module._donated_404_fallback_count = 1
        try:
            with (
                _cli_main_test_mode(alertable_count=0, today=DURING_SUPPRESSION),
                caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
                messages = [record.getMessage() for record in caplog.records]
                assert any("with 1 alertable" in msg for msg in messages), messages
                # The breakdown stays informative: both subsets and the suppressed share are rendered.
                assert any("donated_404=1, credit=1 with 1 credit event(s) suppressed" in msg for msg in messages), (
                    messages
                )
        finally:
            fb_module._generic_key_fallback_count = 0
            fb_module._credit_key_fallback_count = 0
            fb_module._donated_404_fallback_count = 0

    def test_drained_donated_key_alone_exits_zero(self, caplog: pytest.LogCaptureFixture) -> None:
        """The full shape of the 2026-07-26 run, once the fix lands. Both credit
        paths fire together — every donated-key call fell back to the personal key
        AND the end-of-run balance is under the refill floor — and because the probe
        confirmed the key is genuinely drained, the whole run is green.

        This is the outcome the operator asked for: while they self-fund the season,
        an empty donated wallet is bookkeeping, not breakage.

        Green is exactly the shape that most needs a written record, so the run must
        still explain itself: the same breakdown the red path logs (rendered at INFO),
        carrying the probe verdict, plus the floor-breach explanation. Without the
        summary on this branch, the run this whole change set was built for — every
        donated call falling back, so the credit subset cancels the entire generic
        total — would leave no trace of either the degradation or the verdict.
        """
        fb_module._generic_key_fallback_count = 7
        fb_module._credit_key_fallback_count = 7
        try:
            with (
                _cli_main_test_mode(alertable_count=0, donated_below_floor=True, today=DURING_SUPPRESSION),
                patch("metaculus_bot.cli.get_probed_donated_key_state", return_value=DonatedKeyState.DRAINED),
                caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
            ):
                # Must NOT raise SystemExit.
                cli_main()

            messages = [record.getMessage() for record in caplog.records]
            summary = [msg for msg in messages if "alertable degradation event" in msg]
            assert summary, messages
            assert "with 0 alertable" in summary[0], summary
            assert "personal_key_fallback=7" in summary[0], summary
            resume = CREDIT_ALERT_RESUME_DATE.isoformat()
            assert f"credit=7 with 7 credit event(s) suppressed until {resume}" in summary[0], summary
            assert "donated_key=drained" in summary[0], summary
            # The floor-breach explanation is a separate concern and still lands.
            assert any(f"credit alerting is suppressed until {resume}" in msg for msg in messages), messages
        finally:
            fb_module._generic_key_fallback_count = 0
            fb_module._credit_key_fallback_count = 0

    def test_revoked_donated_key_exits_non_zero_during_suppression(self, caplog: pytest.LogCaptureFixture) -> None:
        """The regression the discriminator exists to prevent. Same error text as the
        drained run, but the probe found the key revoked, so the wrapper left the
        credit subset at zero: nothing is subtracted and CI goes red.

        Without the probe, "Key limit exceeded" alone would have exempted a revoked
        or re-capped-to-zero donated key from alerting for six weeks.
        """
        fb_module._generic_key_fallback_count = 7
        fb_module._credit_key_fallback_count = 0
        try:
            with (
                _cli_main_test_mode(alertable_count=0, donated_below_floor=False, today=DURING_SUPPRESSION),
                patch("metaculus_bot.cli.get_probed_donated_key_state", return_value=DonatedKeyState.REVOKED),
                caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
                messages = [record.getMessage() for record in caplog.records]
                assert any("with 7 alertable" in msg for msg in messages), messages
                # The verdict says why nothing was suppressed on a run full of credit-shaped failures.
                assert any("donated_key=revoked" in msg for msg in messages), messages
        finally:
            fb_module._generic_key_fallback_count = 0

    def test_clean_run_logs_an_all_clear_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        """A fully clean run states so, under a distinguishable "clean" phrase.

        This REVERSES the earlier pinned behavior (a clean run logged nothing, so the
        line's presence would stay a signal rather than boilerplate); the operator
        overturned that on 2026-08-25. Silence is indistinguishable from a run that
        died before reaching the summary block, and once the donated key is refilled
        the clean shape becomes the common one — so the archive's per-run census
        would lose precisely the runs that went well.
        """
        with (
            _cli_main_test_mode(alertable_count=0, today=DURING_SUPPRESSION),
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
        ):
            cli_main()

        messages = [record.getMessage() for record in caplog.records]
        summary = [msg for msg in messages if "alertable degradation event" in msg]
        assert len(summary) == 1, messages
        # Pin the phrase: the harvester tells this run from a degraded one whose counters read zero.
        assert summary[0].startswith("Run completed clean with 0 alertable degradation event(s)"), summary
        assert "bot=0, personal_key_fallback=0 of which donated_404=0, credit=0" in summary[0], summary
        # Nothing probed the donated key, so the verdict clause stays absent.
        assert "donated_key=" not in summary[0], summary
        # Seam pin: the harvester must stamp ``outcome=clean``, since an all-zero record alone is ambiguous.
        spec = next(s for s in MARKER_SPECS if s.name == "run_alertable_summary")
        match = spec.regex.search(summary[0])
        assert match is not None, summary
        assert match.group("outcome") == "clean"
        assert match.group("alertable") == "0"

    def test_clean_run_after_the_resume_date_drops_the_suppression_clause(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The shape prod will actually emit once the donated key is refilled: no
        suppression clause, since alerting is live again. This is the run the census
        was about to lose, so pin that it both fires and still harvests."""
        with (
            _cli_main_test_mode(alertable_count=0, today=AFTER_RESUME_DATE),
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
        ):
            cli_main()

        summary = [msg for msg in caplog.messages if "alertable degradation event" in msg]
        assert len(summary) == 1, caplog.messages
        assert summary[0].startswith("Run completed clean with 0 alertable"), summary
        assert "suppressed until" not in summary[0], summary
        spec = next(s for s in MARKER_SPECS if s.name == "run_alertable_summary")
        match = spec.regex.search(summary[0])
        assert match is not None, summary
        assert match.group("outcome") == "clean"
        assert match.group("suppressed_credit") is None

    def test_a_floor_breach_run_after_resume_is_not_labelled_clean(self, caplog: pytest.LogCaptureFixture) -> None:
        """run_clean must be the exact complement of every non-zero exit path,
        including the credit-floor breach that exits AFTER the summary line: a run
        about to go red must not first stamp the archive with the clean token."""
        with (
            _cli_main_test_mode(alertable_count=0, donated_below_floor=True, today=AFTER_RESUME_DATE),
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1

        summary = [msg for msg in caplog.messages if "alertable degradation event" in msg]
        assert len(summary) == 1, caplog.messages
        assert not summary[0].startswith("Run completed clean"), summary
        spec = next(s for s in MARKER_SPECS if s.name == "run_alertable_summary")
        match = spec.regex.search(summary[0])
        assert match is not None, summary
        assert match.group("outcome") is None

    def test_a_deprecation_alert_run_is_not_labelled_clean(self, caplog: pytest.LogCaptureFixture) -> None:
        """Same complement rule for the post-summary deprecation tripwire."""
        fb_module._DEPRECATION_ALERTS.append(("openrouter/x-ai/grok-4.1-fast", "deprecated"))
        try:
            with (
                _cli_main_test_mode(alertable_count=0, today=DURING_SUPPRESSION),
                caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1

            summary = [msg for msg in caplog.messages if "alertable degradation event" in msg]
            assert len(summary) == 1, caplog.messages
            assert not summary[0].startswith("Run completed clean"), summary
        finally:
            fb_module.clear_deprecation_alerts()

    def test_a_degraded_run_is_never_labelled_clean(self, caplog: pytest.LogCaptureFixture) -> None:
        """The "clean" phrase is load-bearing telemetry, so it must not leak onto a
        run that fell back. One suppressed credit fallback exits zero, which is the
        nearest neighbour of the clean path and the easiest one to mislabel."""
        fb_module._generic_key_fallback_count = 1
        fb_module._credit_key_fallback_count = 1
        try:
            with (
                _cli_main_test_mode(alertable_count=0, today=DURING_SUPPRESSION),
                caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
            ):
                cli_main()

            summary = [msg for msg in caplog.messages if "alertable degradation event" in msg]
            assert len(summary) == 1, caplog.messages
            assert summary[0].startswith("Run completed with 0 alertable"), summary
            assert "clean" not in summary[0], summary
        finally:
            fb_module._generic_key_fallback_count = 0
            fb_module._credit_key_fallback_count = 0

    def test_probe_verdict_is_rendered_on_partially_suppressed_red_run(self, caplog: pytest.LogCaptureFixture) -> None:
        """Partial suppression: two fallbacks, one of them credit-caused, so one
        event survives the subtraction and the run is red. The verdict still rides
        the summary — a reader has to be able to tell that the suppressed share was
        exempted because the key was genuinely drained. (The fully-suppressed green
        counterpart is ``test_drained_donated_key_alone_exits_zero``.)
        """
        fb_module._generic_key_fallback_count = 2
        fb_module._credit_key_fallback_count = 1
        try:
            with (
                _cli_main_test_mode(alertable_count=0, today=DURING_SUPPRESSION),
                patch("metaculus_bot.cli.get_probed_donated_key_state", return_value=DonatedKeyState.DRAINED),
                caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"),
            ):
                with pytest.raises(SystemExit):
                    cli_main()
                messages = [record.getMessage() for record in caplog.records]
                assert any("donated_key=drained" in msg for msg in messages), messages
        finally:
            fb_module._generic_key_fallback_count = 0
            fb_module._credit_key_fallback_count = 0

    def test_unprobed_run_omits_the_verdict_clause(self, caplog: pytest.LogCaptureFixture) -> None:
        """No key-limit failure means no probe ran, and "unknown" would read as a
        failed probe rather than "never needed one" — so the clause is omitted.
        """
        fb_module._generic_key_fallback_count = 1
        try:
            with (
                _cli_main_test_mode(alertable_count=0, today=DURING_SUPPRESSION),
                caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"),
            ):
                with pytest.raises(SystemExit):
                    cli_main()
                messages = [record.getMessage() for record in caplog.records]
                assert not any("donated_key=" in msg for msg in messages), messages
        finally:
            fb_module._generic_key_fallback_count = 0

    def test_bot_degradation_still_red_on_a_drained_key_run(self) -> None:
        """A drained donated key must not launder real bot-side degradation into a
        green run: the subtraction is scoped to the credit subset only.
        """
        fb_module._generic_key_fallback_count = 7
        fb_module._credit_key_fallback_count = 7
        try:
            with (
                _cli_main_test_mode(alertable_count=2, donated_below_floor=True, today=DURING_SUPPRESSION),
                patch("metaculus_bot.cli.get_probed_donated_key_state", return_value=DonatedKeyState.DRAINED),
            ):
                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
        finally:
            fb_module._generic_key_fallback_count = 0
            fb_module._credit_key_fallback_count = 0

    def test_bot_alertable_survives_credit_suppression(self) -> None:
        """The subtraction is scoped to the credit subset — a bot-side degradation
        (forecaster drop, stacker fallback) still exits non-zero mid-window even
        if a credit fallback happened in the same run.
        """
        fb_module._generic_key_fallback_count = 1
        fb_module._credit_key_fallback_count = 1
        try:
            with _cli_main_test_mode(alertable_count=3, today=DURING_SUPPRESSION):
                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
        finally:
            fb_module._generic_key_fallback_count = 0
            fb_module._credit_key_fallback_count = 0


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
    ``alertable_count`` stops being an int and every exit-code test in this file fails at once.
    """

    alertable_count = TemplateForecaster.alertable_count
    _degradation_snapshot = TemplateForecaster._degradation_snapshot
    _research_provider_failure_count = TemplateForecaster._research_provider_failure_count
    _summarizer_failure_count = TemplateForecaster._summarizer_failure_count
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


def _observe_venue(venue: str, *, candidates: int, rows: int, fields: frozenset[str]) -> None:
    record_venue_observation(
        VenueObservation(
            qid=45082,
            venue=venue,
            candidates_pre_filter=candidates,
            rows_post_filter=rows,
            liquidity_fields_present=fields,
        )
    )


class TestCliProviderDegradationExit:
    """Provider degradation reaches the exit code, and publishing is untouched.

    The operator's ask: if a provider doesn't populate properly, still submit the
    forecast, but exit non-zero so it doesn't take a residual round weeks later to
    surface. Both halves are load-bearing, and the second is the invariant that must
    never regress — the exit lives in cli AFTER the forecasting call returns, so every
    publishable question is already on Metaculus by the time the status is decided.
    """

    @staticmethod
    def _degrade_kalshi_liquidity_fields() -> None:
        """Record the shape the Kalshi defect produced: three rows with both declared
        liquidity fields absent, so every row renders ``no-liquidity-data``."""
        _observe_venue("kalshi", candidates=3, rows=3, fields=frozenset())

    def test_one_finding_exits_non_zero(self) -> None:
        """The end-to-end wiring, with alertable_count computed rather than pinned."""
        bot = _bot_with_real_alertable_count()
        self._degrade_kalshi_liquidity_fields()
        assert bot.alertable_count == 1

        with _cli_main_test_mode(alertable_count=0, stub_bot=bot, today=AFTER_RESUME_DATE):
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1

    def test_no_findings_returns_normally(self) -> None:
        """A healthy run stays green: the store is empty, so nothing is evaluable."""
        bot = _bot_with_real_alertable_count()
        assert bot.alertable_count == 0

        with _cli_main_test_mode(alertable_count=0, stub_bot=bot, today=AFTER_RESUME_DATE):
            # Must NOT raise SystemExit.
            cli_main()

    def test_a_market_less_run_stays_green(self) -> None:
        """THE false-positive test, at the exit-code level. Every venue returned zero
        rows and zero candidates because the run's one open question is about
        something no prediction market covers. That is normal operation and must not
        redden CI — an alert the operator learns to ignore is worse than silence.
        """
        bot = _bot_with_real_alertable_count()
        for venue in ("polymarket", "kalshi", "manifold", "predictit"):
            _observe_venue(venue, candidates=0, rows=0, fields=frozenset())
        assert bot.alertable_count == 0

        with _cli_main_test_mode(alertable_count=0, stub_bot=bot, today=AFTER_RESUME_DATE):
            # Must NOT raise SystemExit.
            cli_main()

    def test_publishing_completes_before_the_exit(self) -> None:
        """The sacrosanct invariant. Forecasting — which publishes per question, deep
        inside the call — has to have finished, and the report summary has to have been
        logged, before the SystemExit propagates. A degradation alert that suppressed a
        publication would be strictly worse than the silence it replaces.

        Asserted as an ORDERED event log rather than two independent "was it called"
        checks: the ordering is the whole invariant, and two ``assert_called_once``
        calls would pass just as happily if the exit came first.

        ``log_report_summary`` is invoked as ``TemplateForecaster.log_report_summary``
        on the CLASS, so it lands on the class mock cli holds, not on the bot instance.
        This test re-patches that name to keep a handle on it — the helper's own patch
        discards it.
        """
        bot = _bot_with_real_alertable_count()
        self._degrade_kalshi_liquidity_fields()
        events: list[str] = []

        forecaster_class = MagicMock(return_value=bot)
        forecaster_class.log_report_summary.side_effect = lambda *a, **k: events.append("report_summary")

        with _cli_main_test_mode(alertable_count=0, stub_bot=bot, today=AFTER_RESUME_DATE):
            # Set INSIDE the context: the helper installs its own forecast stub on entry.
            async def _record_forecast(*_args: object, **_kwargs: object) -> list[object]:
                events.append("forecast")
                return []

            bot.forecast_questions = AsyncMock(side_effect=_record_forecast)
            with patch("metaculus_bot.cli.TemplateForecaster", forecaster_class):
                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1

        events.append("exit")
        assert events == ["forecast", "report_summary", "exit"]

    def test_suppression_keeps_the_run_green_and_still_logs_the_finding(self, caplog: pytest.LogCaptureFixture) -> None:
        """A dated per-venue acceptance drops the finding out of ``alertable`` while
        keeping every log line, following ``credit_alerts_active``'s contract.

        ``alertable_count`` is a plain property with nowhere to inject a date, so the
        window is pinned through the DICT instead of the clock: a resume date
        permanently in the future is inside the window, one permanently in the past is
        past it. Both branches therefore keep running forever without patching
        ``date.today``.
        """
        bot = _bot_with_real_alertable_count()
        for venue in ("kalshi", "predictit", "polymarket"):
            _observe_venue(venue, candidates=3, rows=3, fields=frozenset(VENUE_EXPECTED_LIQUIDITY_FIELDS[venue]))
        # Manifold's declared `num_bettors` absent from every row: one finding, on the venue under test.
        _observe_venue("manifold", candidates=3, rows=3, fields=frozenset())

        with patch.dict(PROVIDER_DEGRADATION_SUPPRESSED_UNTIL, {"manifold": PERMANENTLY_FUTURE_RESUME}):
            assert bot.alertable_count == 0
            with (
                _cli_main_test_mode(alertable_count=0, stub_bot=bot, today=AFTER_RESUME_DATE),
                caplog.at_level(logging.INFO, logger="metaculus_bot.research.provider_health"),
            ):
                # Must NOT raise SystemExit.
                cli_main()
                # The helper stubs out the REAL forecast_questions that emits the marker, so drive the seam here.
                bot._research.log_provider_degradation_summary()

            messages = [record.getMessage() for record in caplog.records]
            marker = next(msg for msg in messages if msg.startswith("PROVIDER_DEGRADATION:"))
            assert "findings=1 alertable=0 suppressed=1" in marker
            assert f"suppressed until {PERMANENTLY_FUTURE_RESUME.isoformat()}" in marker
            assert "run stays green" in marker

        with patch.dict(PROVIDER_DEGRADATION_SUPPRESSED_UNTIL, {"manifold": PERMANENTLY_PAST_RESUME}):
            # Past the resume date the same state is alertable again: a stale acceptance cannot outlive it.
            assert bot.alertable_count == 1

    def test_a_snapshot_timeout_is_not_double_counted(self) -> None:
        """A whole-provider timeout bumps ``prediction_market_source_losses`` and
        records NO venue observations, so provider-degradation stays 0 and the run
        reports one event rather than two. The exit code can't distinguish 1 from 2, so
        assert the counters directly — the same reasoning as
        ``test_donated_404_fallback_triggers_sys_exit_without_double_counting``.
        """
        bot = _bot_with_real_alertable_count()
        pmp._bump_source_loss()
        try:
            assert bot._prediction_market_source_loss_count == 1
            assert bot._provider_degradation_count == 0
            assert bot.alertable_count == 1
        finally:
            pmp.reset_source_loss_counter()


class TestAlertableSummarySurvivesForecastFailure:
    """Emit-then-raise on a raising ``log_report_summary`` (q45085's shape).

    ``compact_log_report_summary`` deliberately re-raises when any report is an
    exception, so a failed question reddens CI under ``return_exceptions=True`` —
    but that call used to sit ABOVE the alertable block, so the one run that most
    needed a summary record left none: q45085's publish failure (2026-08-03) is
    the single forecasting run since 2026-07-26 with no ``run_alertable_summary``
    line in the archive. The invariant: the breakdown line is emitted, THEN the
    original exception propagates. Never a swallow — CI must stay red.
    """

    def test_breakdown_emitted_then_failure_reraised(self, caplog: pytest.LogCaptureFixture) -> None:
        bot = _bot_with_real_alertable_count()
        forecaster_class = MagicMock(return_value=bot)
        forecaster_class.log_report_summary.side_effect = RuntimeError("1 errors occurred while forecasting")

        with (
            _cli_main_test_mode(alertable_count=0, stub_bot=bot, today=AFTER_RESUME_DATE),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"),
            pytest.raises(RuntimeError, match="errors occurred while forecasting"),
        ):
            cli_main()

        breakdown_lines = [m for m in caplog.messages if m.startswith("Run completed with")]
        assert len(breakdown_lines) == 1
        assert "re-raising the forecasting failure" in breakdown_lines[0]
        # All three counters read zero, but the run lost a question, so it must not carry the all-clear phrase.
        assert "clean" not in breakdown_lines[0]

    def test_failure_outranks_the_alertable_exit_and_keeps_the_count(self, caplog: pytest.LogCaptureFixture) -> None:
        """Both red states at once: the exception (with its traceback) is the red
        signal rather than ``SystemExit``, and the emitted breakdown still records
        the positive alertable count instead of losing it to the crash."""
        bot = _bot_with_real_alertable_count()
        bot._forecasters_dropped_count = 3
        forecaster_class = MagicMock(return_value=bot)
        forecaster_class.log_report_summary.side_effect = RuntimeError("2 errors occurred while forecasting")

        with (
            _cli_main_test_mode(alertable_count=0, stub_bot=bot, today=AFTER_RESUME_DATE),
            patch("metaculus_bot.cli.TemplateForecaster", forecaster_class),
            caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"),
            pytest.raises(RuntimeError),
        ):
            cli_main()

        breakdown = next(m for m in caplog.messages if m.startswith("Run completed with"))
        assert breakdown.startswith("Run completed with 3 alertable")
