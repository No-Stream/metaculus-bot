"""Tests for ``metaculus_bot.cli.main`` — specifically the sys.exit wiring that
fires when ``TemplateForecaster.alertable_count > 0`` OR when the donated
OpenRouter key fell back to the operator's personal (paid) key during the run.

The fallback counter folded into ``alertable`` is ``_generic_key_fallback_count``
— it counts EVERY donated->personal fallback (all causes: 401/402/429/guardrail/
404). ``_donated_404_fallback_count`` is the allowed-providers-404 subset, broken
out in the log line for diagnostics but NOT separately added to ``alertable``
(that would double-count the 404 events already inside the generic total).

Publication already happened inside ``forecast_on_tournament`` by the time cli
checks alertable state; the non-zero exit is purely so GitHub Actions marks
the run red. That wiring is load-bearing — without it, forecaster drops,
stacker fallback usage, and silent personal-key spend go unnoticed.
"""

from __future__ import annotations

import logging
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def _patch_cli_main_deps(alertable_count: int) -> tuple[Any, ...]:
    """Return the list of patchers needed to exercise ``cli.main`` in test mode.

    Stubs TemplateForecaster with a MagicMock whose ``alertable_count`` is
    controlled and whose ``forecast_questions`` returns an empty list (so no
    downstream ``log_report_summary`` formatting is needed).
    """
    stub_bot = MagicMock()
    stub_bot.alertable_count = alertable_count
    stub_bot.forecast_questions = AsyncMock(return_value=[])
    stub_bot.forecast_on_tournament = AsyncMock(return_value=[])

    # TemplateForecaster(...) call returns our stub
    tf_patcher = patch("metaculus_bot.cli.TemplateForecaster", return_value=stub_bot)
    # MetaculusApi.get_question_by_url returns a dummy question object; we pass
    # through a Mock so list construction doesn't explode.
    api_patcher = patch("metaculus_bot.cli.MetaculusApi", MagicMock())
    check_patcher = patch("metaculus_bot.cli.check_tournament_dates")
    # Patch log_report_summary: a classmethod on TemplateForecaster that
    # iterates forecast_reports. Our stub returns []; patch the method anyway
    # to keep the test surface small.
    summary_patcher = patch.object(type(stub_bot), "log_report_summary", create=True, return_value=None)

    return tf_patcher, api_patcher, check_patcher, summary_patcher


class TestCliExitStatus:
    def test_alertable_count_zero_returns_normally(self) -> None:
        """Zero degradation events → no SystemExit; main returns normally."""
        tf, api, check, summary = _patch_cli_main_deps(alertable_count=0)

        argv_backup = sys.argv
        sys.argv = ["cli", "--mode", "test_questions"]
        try:
            with tf, api, check, summary:
                from metaculus_bot.cli import main as cli_main

                # Must NOT raise SystemExit.
                cli_main()
        finally:
            sys.argv = argv_backup

    def test_alertable_count_nonzero_triggers_sys_exit_1(self) -> None:
        """Non-zero degradation counter → SystemExit with code 1."""
        tf, api, check, summary = _patch_cli_main_deps(alertable_count=1)

        argv_backup = sys.argv
        sys.argv = ["cli", "--mode", "test_questions"]
        try:
            with tf, api, check, summary:
                from metaculus_bot.cli import main as cli_main

                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
        finally:
            sys.argv = argv_backup

    def test_large_alertable_count_still_exits_with_code_1(self) -> None:
        """Exit code is always 1 regardless of how many events occurred —
        documents that we use exit-code-1 as a binary alert, not as an
        event count.
        """
        tf, api, check, summary = _patch_cli_main_deps(alertable_count=42)

        argv_backup = sys.argv
        sys.argv = ["cli", "--mode", "test_questions"]
        try:
            with tf, api, check, summary:
                from metaculus_bot.cli import main as cli_main

                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
        finally:
            sys.argv = argv_backup

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
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415

        fb_module._generic_key_fallback_count = 1

        tf, api, check, summary = _patch_cli_main_deps(alertable_count=0)

        argv_backup = sys.argv
        sys.argv = ["cli", "--mode", "test_questions"]
        try:
            with tf, api, check, summary:
                from metaculus_bot.cli import main as cli_main  # noqa: PLC0415

                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
        finally:
            sys.argv = argv_backup
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
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415

        # Mirror FallbackOpenRouterLlm.invoke: a 404 fallback bumps the generic
        # counter AND the 404 subset.
        fb_module._generic_key_fallback_count = 1
        fb_module._donated_404_fallback_count = 1

        tf, api, check, summary = _patch_cli_main_deps(alertable_count=0)

        argv_backup = sys.argv
        sys.argv = ["cli", "--mode", "test_questions"]
        try:
            with tf, api, check, summary, caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"):
                from metaculus_bot.cli import main as cli_main  # noqa: PLC0415

                with pytest.raises(SystemExit) as exc_info:
                    cli_main()
                assert exc_info.value.code == 1
                # Pins alertable == 1 (not 2): the count is the first %d in the
                # end-of-run warning. A double-count regression renders "with 2".
                assert any("with 1 alertable" in record.getMessage() for record in caplog.records), (
                    f"expected 'with 1 alertable' in warnings; got: {[r.getMessage() for r in caplog.records]}"
                )
        finally:
            sys.argv = argv_backup
            fb_module._generic_key_fallback_count = 0
            fb_module._donated_404_fallback_count = 0

    def test_no_fallback_with_bot_alertable_zero_returns_normally(self) -> None:
        """Both bot alertable_count == 0 AND fallback counters == 0 → no SystemExit.

        Pins the conjunction: the autouse fixture resets both counters,
        and main returns normally when nothing was alertable.
        """
        tf, api, check, summary = _patch_cli_main_deps(alertable_count=0)

        argv_backup = sys.argv
        sys.argv = ["cli", "--mode", "test_questions"]
        try:
            with tf, api, check, summary:
                from metaculus_bot.cli import main as cli_main  # noqa: PLC0415

                # Must NOT raise SystemExit.
                cli_main()
        finally:
            sys.argv = argv_backup
