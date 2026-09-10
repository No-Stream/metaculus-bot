"""Provider degradation reaches cli's exit code without touching publication.

``alertable_count`` is COMPUTED through the real property chain here rather than pinned to a
literal, which is what makes the summand's own wiring observable; the stub that does it is
``_bot_with_real_alertable_count``.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import metaculus_bot.research.prediction_market as pmp
from metaculus_bot.cli import main as cli_main
from metaculus_bot.constants import PROVIDER_DEGRADATION_SUPPRESSED_UNTIL
from metaculus_bot.research.provider_health import VENUE_EXPECTED_LIQUIDITY_FIELDS
from tests.cli_test_helpers import (
    AFTER_RESUME_DATE,
    PERMANENTLY_FUTURE_RESUME,
    PERMANENTLY_PAST_RESUME,
    _bot_with_real_alertable_count,
    _cli_main_test_mode,
    _observe_venue,
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
