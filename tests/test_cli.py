"""Tests for ``metaculus_bot.cli.main`` — specifically the sys.exit wiring that
fires when ``TemplateForecaster.alertable_count > 0``, when the donated
OpenRouter key fell back to the operator's personal (paid) key during the run,
or when the donated key's remaining balance ended the run below the refill
floor (``OPENROUTER_CREDIT_FLOOR_USD``).

The fallback counter folded into ``alertable`` is ``_generic_key_fallback_count``
— it counts EVERY donated->personal fallback (all causes: 401/402/429/guardrail/
404). ``_donated_404_fallback_count`` is the allowed-providers-404 subset, broken
out in the log line for diagnostics but NOT separately added to ``alertable``
(that would double-count the 404 events already inside the generic total).

Publication already happened inside ``forecast_on_tournament`` by the time cli
checks alertable state; the non-zero exit is purely so GitHub Actions marks
the run red. That wiring is load-bearing — without it, forecaster drops,
stacker fallback usage, silent personal-key spend, and a draining donated
balance all go unnoticed.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metaculus_bot.cli import main as cli_main
from metaculus_bot.fallback_openrouter import (
    reset_donated_404_fallback_count,
    reset_generic_key_fallback_count,
)


@pytest.fixture(autouse=True)
def _reset_fallback_counters() -> None:
    """The fallback counters are process-global (module state in
    fallback_openrouter). Reset both between tests so cross-test pollution
    can't silently turn an "alertable=0" path into "alertable=1" because a
    prior test bumped a counter.
    """
    reset_generic_key_fallback_count()
    reset_donated_404_fallback_count()


@contextmanager
def _cli_main_test_mode(alertable_count: int, *, donated_below_floor: bool = False) -> Iterator[MagicMock]:
    """Run ``cli.main`` with all external dependencies stubbed; yields the
    CreditTelemetry stub for call assertions.

    Stubs TemplateForecaster with a MagicMock whose ``alertable_count`` is
    controlled and whose ``forecast_questions`` returns an empty list (so no
    downstream ``log_report_summary`` formatting is needed). Also stubs
    CreditTelemetry so tests never hit the real OpenRouter balance endpoint
    (a local ``.env`` would otherwise supply real keys); its floor-check
    result is controlled via ``donated_below_floor``. sys.argv is pinned to
    test_questions mode and restored afterwards.
    """
    stub_bot = MagicMock()
    stub_bot.alertable_count = alertable_count
    stub_bot.forecast_questions = AsyncMock(return_value=[])
    stub_bot.forecast_on_tournament = AsyncMock(return_value=[])

    stub_telemetry = MagicMock()
    stub_telemetry.log_end_and_check_floor.return_value = donated_below_floor

    argv_backup = sys.argv
    sys.argv = ["cli", "--mode", "test_questions"]
    try:
        with (
            # TemplateForecaster(...) call returns our stub
            patch("metaculus_bot.cli.TemplateForecaster", return_value=stub_bot),
            # MetaculusApi.get_question_by_url returns a dummy question object; we
            # pass through a Mock so list construction doesn't explode.
            patch("metaculus_bot.cli.MetaculusApi", MagicMock()),
            # cli.main() applies fetch/publish hardening at startup, which globally
            # and permanently mutates MetaculusClient (patches post_*/_get_questions_from_api,
            # sets a sentinel). Left un-stubbed those mutations leak into every later
            # test in the session — a randomly-ordered run then poisons the publish/fetch
            # seam tests' un-hardened negative controls. Stub them: this test pins the
            # exit-status wiring, not the hardening install (covered by its own tests).
            patch("metaculus_bot.cli.apply_publish_hardening"),
            patch("metaculus_bot.cli.apply_fetch_hardening"),
            patch("metaculus_bot.cli.check_tournament_dates"),
            # Patch log_report_summary: a classmethod on TemplateForecaster that
            # iterates forecast_reports. Our stub returns []; patch the method
            # anyway to keep the test surface small.
            patch.object(type(stub_bot), "log_report_summary", create=True, return_value=None),
            patch("metaculus_bot.cli.CreditTelemetry", return_value=stub_telemetry),
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
        # Simulate the wrapper having fired a generic (non-404) donated->personal
        # fallback during the run. cli.main reads this AFTER forecast returns.
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

        fb_module._generic_key_fallback_count = 1
        try:
            with _cli_main_test_mode(alertable_count=0):
                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
        finally:
            # autouse fixture already resets, but be explicit on the path
            # that bypasses normal flow.
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
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

        # Mirror FallbackOpenRouterLlm.invoke: a 404 fallback bumps the generic
        # counter AND the 404 subset.
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
                # Pins alertable == 1 (not 2): the count is the first %d in the
                # end-of-run warning. A double-count regression renders "with 2".
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


class TestCliCreditFloor:
    """End-of-run donated-key credit-floor wiring in cli.main.

    The floor check itself (thresholds, n/a handling, fetch failures) is unit
    tested in test_credit_telemetry.py; these tests pin the cli wiring — that
    the boolean returned by ``log_end_and_check_floor`` drives the exit code,
    and that both telemetry phases run even when forecasting crashes.
    """

    def test_below_floor_triggers_sys_exit_1(self) -> None:
        """Donated balance below floor → run completes, then SystemExit(1)."""
        with _cli_main_test_mode(alertable_count=0, donated_below_floor=True) as telemetry:
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1
            telemetry.log_start.assert_called_once()
            telemetry.log_end_and_check_floor.assert_called_once()

    def test_above_floor_returns_normally(self) -> None:
        """Healthy balance → telemetry logs both phases, no SystemExit."""
        with _cli_main_test_mode(alertable_count=0, donated_below_floor=False) as telemetry:
            cli_main()
            telemetry.log_start.assert_called_once()
            telemetry.log_end_and_check_floor.assert_called_once()

    def test_end_telemetry_runs_when_forecasting_crashes(self) -> None:
        """The end-of-run fetch is in a finally: a crashed run still logs its
        spend (the original exception propagates, not a floor SystemExit).
        """
        with _cli_main_test_mode(alertable_count=0, donated_below_floor=True) as telemetry:
            with patch(
                "metaculus_bot.cli.asyncio.run",
                side_effect=RuntimeError("forecasting blew up"),
            ):
                with pytest.raises(RuntimeError, match="forecasting blew up"):
                    cli_main()
            telemetry.log_start.assert_called_once()
            telemetry.log_end_and_check_floor.assert_called_once()
