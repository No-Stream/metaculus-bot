"""Tests for the donated-OpenRouter-key credit wiring in ``metaculus_bot.cli.main``: the
end-of-run balance floor (``OPENROUTER_CREDIT_FLOOR_USD``) and the dated suppression of credit
alerts. The per-role spend ledger is in ``test_cli_role_spend.py``.

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
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

import metaculus_bot.fallback_openrouter as fb_module
from metaculus_bot.cli import main as cli_main
from metaculus_bot.constants import CREDIT_ALERT_RESUME_DATE, credit_alerts_active
from metaculus_bot.credit_telemetry import DonatedKeyState
from scripts.telemetry.markers import MARKER_SPECS
from tests.cli_test_helpers import (
    AFTER_RESUME_DATE,
    DURING_SUPPRESSION,
    ON_RESUME_DATE,
    _cli_main_test_mode,
    asyncio_run_stub,
)


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
