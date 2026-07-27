"""Tests for ``metaculus_bot.cli.main`` — specifically the sys.exit wiring that
fires when ``TemplateForecaster.alertable_count > 0``, when the donated
OpenRouter key fell back to the operator's personal (paid) key during the run,
or when the donated key's remaining balance ended the run below the refill
floor (``OPENROUTER_CREDIT_FLOOR_USD``).

The fallback counter folded into ``alertable`` is ``_generic_key_fallback_count``
— it counts EVERY donated->personal fallback (all causes: 401/402/429/guardrail/
404). ``_donated_404_fallback_count`` (allowed-providers 404) and
``_credit_key_fallback_count`` (402 / insufficient credit) are two disjoint
subsets of that total, broken out in the log line for diagnostics but NOT
separately added to ``alertable`` (that would double-count events already inside
the generic total).

Credit alerting is suppressed until ``CREDIT_ALERT_RESUME_DATE`` (2026-09-10)
because the operator is self-funding the rest of the season, so a drained donated
key is expected rather than broken. During the window the floor breach does not
exit non-zero and the credit-caused fallbacks are subtracted back out of
``alertable``; every other fallback cause (401 / 404 / 429 / guardrail) keeps its
full weight. Tests inject the window state via ``credit_alerts_active`` rather
than the wall clock, so they keep testing both sides after the real date passes.

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
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metaculus_bot.cli import main as cli_main
from metaculus_bot.constants import CREDIT_ALERT_RESUME_DATE, credit_alerts_active
from metaculus_bot.credit_telemetry import DonatedKeyState, reset_donated_key_state_cache
from metaculus_bot.fallback_openrouter import (
    reset_credit_key_fallback_count,
    reset_donated_404_fallback_count,
    reset_generic_key_fallback_count,
)

# Dates on either side of the suppression boundary. Injected instead of read from
# the clock so these tests keep exercising both branches forever.
DURING_SUPPRESSION = date(2026, 7, 25)
ON_RESUME_DATE = CREDIT_ALERT_RESUME_DATE
AFTER_RESUME_DATE = date(2026, 10, 1)


@pytest.fixture(autouse=True)
def _reset_fallback_counters() -> None:
    """The fallback counters are process-global (module state in
    fallback_openrouter). Reset all three between tests so cross-test pollution
    can't silently turn an "alertable=0" path into "alertable=1" because a
    prior test bumped a counter.

    The donated-key probe verdict is process-global for the same reason (probe
    once per run), and cli renders it in the end-of-run summary, so it is reset
    here too.
    """
    reset_generic_key_fallback_count()
    reset_donated_404_fallback_count()
    reset_credit_key_fallback_count()
    reset_donated_key_state_cache()


@contextmanager
def _cli_main_test_mode(
    alertable_count: int,
    *,
    donated_below_floor: bool = False,
    today: date | None = None,
) -> Iterator[MagicMock]:
    """Run ``cli.main`` with all external dependencies stubbed; yields the
    CreditTelemetry stub for call assertions.

    Stubs TemplateForecaster with a MagicMock whose ``alertable_count`` is
    controlled and whose ``forecast_questions`` returns an empty list (so no
    downstream ``log_report_summary`` formatting is needed). Also stubs
    CreditTelemetry so tests never hit the real OpenRouter balance endpoint
    (a local ``.env`` would otherwise supply real keys); its floor-check
    result is controlled via ``donated_below_floor``. sys.argv is pinned to
    test_questions mode and restored afterwards.

    ``today`` pins the credit-suppression window: cli reads it through
    ``credit_alerts_active``, which we re-bind to evaluate against the injected
    date. ``None`` leaves the real system clock in place (the production path),
    which is what the tests that don't care about credit state want.
    """
    stub_bot = MagicMock()
    stub_bot.alertable_count = alertable_count
    stub_bot.forecast_questions = AsyncMock(return_value=[])
    stub_bot.forecast_on_tournament = AsyncMock(return_value=[])

    stub_telemetry = MagicMock()
    stub_telemetry.log_end_and_check_floor.return_value = donated_below_floor

    # Re-bind cli's own reference so only the injected date decides the window;
    # patching with the real function when today is None keeps one `with` shape.
    pinned_clock = patch(
        "metaculus_bot.cli.credit_alerts_active",
        credit_alerts_active if today is None else lambda: credit_alerts_active(today),
    )

    argv_backup = sys.argv
    sys.argv = ["cli", "--mode", "test_questions"]
    try:
        with (
            pinned_clock,
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
            with patch(
                "metaculus_bot.cli.asyncio.run",
                side_effect=RuntimeError("forecasting blew up"),
            ):
                with pytest.raises(RuntimeError, match="forecasting blew up"):
                    cli_main()
            telemetry.log_start.assert_called_once()
            telemetry.log_end_and_check_floor.assert_called_once()


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
        assert any("credit alerting is suppressed until 2026-09-10" in msg for msg in messages), messages

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
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

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
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

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
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

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
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

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
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

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
                # The breakdown stays informative: both subsets and the suppressed
                # share are rendered for whoever greps this line.
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
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

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
            assert "credit=7 with 7 credit event(s) suppressed until 2026-09-10" in summary[0], summary
            assert "donated_key=drained" in summary[0], summary
            # The floor-breach explanation is a separate concern and still lands.
            assert any("credit alerting is suppressed until 2026-09-10" in msg for msg in messages), messages
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
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

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
                # The summary names the verdict so a reader knows why nothing was
                # suppressed on a run full of credit-shaped failures.
                assert any("donated_key=revoked" in msg for msg in messages), messages
        finally:
            fb_module._generic_key_fallback_count = 0

    def test_clean_run_logs_no_degradation_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        """The green-path summary is conditioned on a fallback having happened, not
        on the exit status: a run where nothing degraded says nothing, so the line's
        presence in a log remains a signal rather than boilerplate.
        """
        with (
            _cli_main_test_mode(alertable_count=0, today=DURING_SUPPRESSION),
            caplog.at_level(logging.INFO, logger="metaculus_bot.cli"),
        ):
            cli_main()

        messages = [record.getMessage() for record in caplog.records]
        assert not any("alertable degradation event" in msg for msg in messages), messages

    def test_probe_verdict_is_rendered_on_partially_suppressed_red_run(self, caplog: pytest.LogCaptureFixture) -> None:
        """Partial suppression: two fallbacks, one of them credit-caused, so one
        event survives the subtraction and the run is red. The verdict still rides
        the summary — a reader has to be able to tell that the suppressed share was
        exempted because the key was genuinely drained. (The fully-suppressed green
        counterpart is ``test_drained_donated_key_alone_exits_zero``.)
        """
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

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
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

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
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

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
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

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
