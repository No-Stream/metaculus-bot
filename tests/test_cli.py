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

import inspect
import json
import logging
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from metaculus_bot.cli import main as cli_main
from metaculus_bot.constants import (
    CREDIT_ALERT_RESUME_DATE,
    PERSIST_RESEARCH_ENABLED_ENV,
    PROVIDER_DEGRADATION_SUPPRESSED_UNTIL,
    credit_alerts_active,
)
from metaculus_bot.credit_telemetry import DonatedKeyState, reset_donated_key_state_cache
from metaculus_bot.fallback_openrouter import (
    reset_credit_key_fallback_count,
    reset_donated_404_fallback_count,
    reset_generic_key_fallback_count,
)
from metaculus_bot.forecaster import TemplateForecaster
from metaculus_bot.research.provider_health import (
    VENUE_EXPECTED_LIQUIDITY_FIELDS,
    VenueObservation,
    record_venue_observation,
    reset_provider_health,
)
from scripts.telemetry.markers import MARKER_SPECS

# Dates on either side of the suppression boundary. Injected instead of read from
# the clock so these tests keep exercising both branches forever.
DURING_SUPPRESSION = date(2026, 7, 25)
ON_RESUME_DATE = CREDIT_ALERT_RESUME_DATE
AFTER_RESUME_DATE = date(2026, 10, 1)

# Provider-degradation suppression takes no injected date on the property path
# (``alertable_count`` is a plain sum), so its two branches are pinned by choosing
# resume dates that can never fall on the wrong side of the real clock.
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


@contextmanager
def _cli_main_test_mode(
    alertable_count: int,
    *,
    donated_below_floor: bool = False,
    today: date | None = None,
    stub_bot: MagicMock | None = None,
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

    ``stub_bot`` overrides the whole bot object, for tests that need
    ``alertable_count`` COMPUTED through the real property chain rather than
    pinned to a literal (see ``_bot_with_real_alertable_count``). ``alertable_count``
    is ignored when it is supplied.
    """
    if stub_bot is None:
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
            # The API identity preflight makes a real unauthenticated GET to
            # metaculus.com; stub it so these exit-status/telemetry tests stay
            # hermetic (its own behavior is covered in test_api_preflight.py).
            patch("metaculus_bot.cli.verify_metaculus_api_identity"),
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

            def _crash(*_args: object, **_kwargs: object) -> None:
                raise RuntimeError("forecasting blew up")

            with patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_crash)):
                with pytest.raises(RuntimeError, match="forecasting blew up"):
                    cli_main()
            telemetry.log_start.assert_called_once()
            telemetry.log_end_and_check_floor.assert_called_once()


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
            # Two questions researched, then the run dies before returning — the shape
            # that used to lose the whole batch.
            self._record_two(forecaster_class.call_args.kwargs["research_sink"])
            raise RuntimeError("forecast loop blew up")

        with _cli_main_test_mode(alertable_count=0):
            with patch("metaculus_bot.cli.TemplateForecaster", forecaster_class):
                with patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_record_then_crash)):
                    # The original exception must still propagate: the flush is a rescue,
                    # not a swallow.
                    with pytest.raises(RuntimeError, match="forecast loop blew up"):
                        cli_main()

        assert [r["qid"] for r in self._flushed_records(tmp_path)] == [43613, 50001]

    def test_flush_still_runs_on_the_normal_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(PERSIST_RESEARCH_ENABLED_ENV, "true")
        monkeypatch.chdir(tmp_path)

        forecaster_class = self._forecaster_class()

        def _record_then_return(*_args: object, **_kwargs: object) -> list[object]:
            self._record_two(forecaster_class.call_args.kwargs["research_sink"])
            return []

        with _cli_main_test_mode(alertable_count=0):
            with patch("metaculus_bot.cli.TemplateForecaster", forecaster_class):
                with patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_record_then_return)):
                    cli_main()

        assert [r["qid"] for r in self._flushed_records(tmp_path)] == [43613, 50001]

    def test_nothing_is_written_when_the_flag_is_off(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # No writer, no sink: the forecaster is handed None and the run leaves no
        # research_outputs/ at all. Pins that the finally-block flush is guarded.
        monkeypatch.delenv(PERSIST_RESEARCH_ENABLED_ENV, raising=False)
        monkeypatch.chdir(tmp_path)

        def _crash(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("boom")

        forecaster_class = self._forecaster_class()
        with _cli_main_test_mode(alertable_count=0):
            with patch("metaculus_bot.cli.TemplateForecaster", forecaster_class):
                with patch("metaculus_bot.cli.asyncio.run", side_effect=asyncio_run_stub(_crash)):
                    with pytest.raises(RuntimeError, match="boom"):
                        cli_main()

        assert forecaster_class.call_args.kwargs["research_sink"] is None
        assert not (tmp_path / "research_outputs").exists()


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
        # The whole point is that the harvester can tell this run apart from a
        # degraded one whose counters happen to read zero, so pin the phrase.
        assert summary[0].startswith("Run completed clean with 0 alertable degradation event(s)"), summary
        assert "bot=0, personal_key_fallback=0 of which donated_404=0, credit=0" in summary[0], summary
        # Nothing probed the donated key, so the verdict clause stays absent.
        assert "donated_key=" not in summary[0], summary
        # Seam pin: the harvester must recognise the line this code actually emits,
        # and stamp it ``outcome=clean`` — an all-zero record alone is ambiguous
        # (a run that lost a question reads all zeros too).
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

    def test_a_degraded_run_is_never_labelled_clean(self, caplog: pytest.LogCaptureFixture) -> None:
        """The "clean" phrase is load-bearing telemetry, so it must not leak onto a
        run that fell back. One suppressed credit fallback exits zero, which is the
        nearest neighbour of the clean path and the easiest one to mislabel."""
        import metaculus_bot.fallback_openrouter as fb_module  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

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


class _RealAlertableCountBot(MagicMock):
    """A cli stub whose ``alertable_count`` is COMPUTED, not pinned to a literal.

    The provider-degradation summand has to travel the whole real chain — module
    observation store, then ``ResearchOrchestrator.provider_degradation_count``, then
    ``TemplateForecaster._provider_degradation_count``, then ``alertable_count``, then
    the ``sys.exit`` in cli — or the test proves only that cli exits on a number the
    test handed it. A plain ``MagicMock`` attribute set to an int proves exactly that,
    which is how a broken summand ships green.

    A dedicated SUBCLASS rather than assignments onto ``type(mock)``: for a
    ``MagicMock`` instance that expression is ``MagicMock`` itself, so binding the
    properties there would mutate the class for every mock in the session and leak
    into unrelated tests.

    EVERY property-backed summand of ``alertable_total`` has to be listed below. A
    missing one leaves a ``MagicMock`` in the sum, so ``alertable_count`` stops being an
    int and every test in this file that reads the exit code or the summary line fails
    at once — loud, but only if you know to look here.
    """

    alertable_count = TemplateForecaster.alertable_count
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
    from metaculus_bot.research.orchestrator import (  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
        ResearchOrchestrator,
    )

    stub_bot = _RealAlertableCountBot()
    stub_bot._research = ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())
    for counter in (
        "_forecasters_dropped_count",
        "_questions_failed_to_publish",
        "_stacker_primary_failed_count",
        "_stacker_fallback_used_count",
        "_stacker_fallback_failed_count",
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
            # Set INSIDE the context: the helper installs its own forecast stub while
            # entering, so an assignment made beforehand is silently clobbered.
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
        # Manifold's declared field (`num_bettors`) absent from every pool row: one
        # `market_field_contract` finding on the venue whose acceptance is under test.
        _observe_venue("manifold", candidates=3, rows=3, fields=frozenset())

        with patch.dict(PROVIDER_DEGRADATION_SUPPRESSED_UNTIL, {"manifold": PERMANENTLY_FUTURE_RESUME}):
            assert bot.alertable_count == 0
            with (
                _cli_main_test_mode(alertable_count=0, stub_bot=bot, today=AFTER_RESUME_DATE),
                caplog.at_level(logging.INFO, logger="metaculus_bot.research.provider_health"),
            ):
                # Must NOT raise SystemExit.
                cli_main()
                # The marker is emitted by the REAL forecast_questions, which this
                # helper stubs out (it pins cli's exit wiring, not the forecast loop),
                # so drive the orchestrator seam cli's bot exposes. That the forecaster
                # calls it per run is pinned in test_template_forecaster.py.
                bot._research.log_provider_degradation_summary()

            messages = [record.getMessage() for record in caplog.records]
            marker = next(msg for msg in messages if msg.startswith("PROVIDER_DEGRADATION:"))
            assert "findings=1 alertable=0 suppressed=1" in marker
            assert f"suppressed until {PERMANENTLY_FUTURE_RESUME.isoformat()}" in marker
            assert "run stays green" in marker

        with patch.dict(PROVIDER_DEGRADATION_SUPPRESSED_UNTIL, {"manifold": PERMANENTLY_PAST_RESUME}):
            # Past the resume date, the same state is alertable again — a stale
            # acceptance cannot outlive its date unnoticed.
            assert bot.alertable_count == 1

    def test_a_snapshot_timeout_is_not_double_counted(self) -> None:
        """A whole-provider timeout bumps ``prediction_market_source_losses`` and
        records NO venue observations, so provider-degradation stays 0 and the run
        reports one event rather than two. The exit code can't distinguish 1 from 2, so
        assert the counters directly — the same reasoning as
        ``test_donated_404_fallback_triggers_sys_exit_without_double_counting``.
        """
        import metaculus_bot.research.prediction_market as pmp  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import

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

        with _cli_main_test_mode(alertable_count=0, stub_bot=bot, today=AFTER_RESUME_DATE):
            with patch("metaculus_bot.cli.TemplateForecaster", forecaster_class):
                with caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"):
                    with pytest.raises(RuntimeError, match="errors occurred while forecasting"):
                        cli_main()

        breakdown_lines = [m for m in caplog.messages if m.startswith("Run completed with")]
        assert len(breakdown_lines) == 1
        assert "re-raising the forecasting failure" in breakdown_lines[0]
        # All three counters read zero on this run, but it lost a question — so it
        # must NOT pick up the all-clear phrase that a genuinely clean run carries.
        assert "clean" not in breakdown_lines[0]

    def test_failure_outranks_the_alertable_exit_and_keeps_the_count(self, caplog: pytest.LogCaptureFixture) -> None:
        """Both red states at once: the exception (with its traceback) is the red
        signal rather than ``SystemExit``, and the emitted breakdown still records
        the positive alertable count instead of losing it to the crash."""
        bot = _bot_with_real_alertable_count()
        bot._forecasters_dropped_count = 3
        forecaster_class = MagicMock(return_value=bot)
        forecaster_class.log_report_summary.side_effect = RuntimeError("2 errors occurred while forecasting")

        with _cli_main_test_mode(alertable_count=0, stub_bot=bot, today=AFTER_RESUME_DATE):
            with patch("metaculus_bot.cli.TemplateForecaster", forecaster_class):
                with caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"):
                    with pytest.raises(RuntimeError):
                        cli_main()

        breakdown = next(m for m in caplog.messages if m.startswith("Run completed with"))
        assert breakdown.startswith("Run completed with 3 alertable")
