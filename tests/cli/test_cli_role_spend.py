"""cli.main installs the CREDIT_ROLE_SPEND tracker before the first completion, logs the ledger
from the same ``finally`` as the balance telemetry, and drains litellm's callbacks inside the
forecast loop, so a crashed run still reports where its money went.

The ledger itself is unit tested in ``tests/test_credit_role_spend.py``; these pin the wiring,
including the single ``asyncio.run`` in ``_run_forecasts`` that every run mode goes through.
"""

from __future__ import annotations

from typing import Any, get_args
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest

from metaculus_bot.cli import RunMode, _forecast_with_callback_drain, _run_forecasts
from metaculus_bot.cli import main as cli_main
from metaculus_bot.credit_telemetry import RoleSpendTracker
from tests.cli_test_helpers import _cli_main_test_mode, asyncio_run_stub


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
