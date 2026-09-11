"""Tests for ``metaculus_bot.cli.main`` — the ``sys.exit`` wiring that fires when
``TemplateForecaster.alertable_count > 0``, when the donated OpenRouter key fell back to the
operator's personal (paid) key during the run, or when a Mantic post was dropped, plus the
emit-then-raise ordering that keeps the end-of-run breakdown on a run whose forecast failed.

Publication already happened inside ``forecast_on_tournament`` by the time cli
checks alertable state; the non-zero exit is purely so GitHub Actions marks
the run red. That wiring is load-bearing — without it, forecaster drops,
stacker fallback usage, silent personal-key spend, and a draining donated
balance all go unnoticed.

The fallback-counter arithmetic and the dated credit-alert window are pinned in
``test_cli_credit_alerts.py``.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import metaculus_bot.fallback_openrouter as fb_module
from metaculus_bot.cli import main as cli_main
from tests.cli_test_helpers import AFTER_RESUME_DATE, _bot_with_real_alertable_count, _cli_main_test_mode


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

    def test_real_v1_schema_failure_reaches_cli_exit_after_forecasting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A malformed analyzer response reaches the real v1 stage and CLI path."""
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.delenv("GAP_FILL_V2_ENABLED", raising=False)

        bot = _bot_with_real_alertable_count()
        bot._research._custom_provider = AsyncMock(return_value="first-pass research " * 20)
        question = MagicMock()
        question.id_of_question = 45123
        question.page_url = "https://example.com/questions/45123"
        question.question_text = "Will X happen?"
        question.resolution_criteria = "Resolves on X."
        question.fine_print = ""
        question.options = None
        analyzer = MagicMock(
            invoke=AsyncMock(
                return_value=json.dumps(
                    {
                        "gaps": [
                            {
                                "gap": f"Missing fact {index}",
                                "search_query": f"official fact {index}",
                                "why_matters": "Changes the forecast",
                            }
                            for index in range(4)
                        ]
                    }
                )
            )
        )
        events: list[str] = []
        forecaster_class = MagicMock(return_value=bot)
        forecaster_class.log_report_summary.side_effect = lambda *a, **k: events.append("report_summary")

        with (
            patch("metaculus_bot.fallback_openrouter.build_llm_with_openrouter_fallback", return_value=analyzer),
            patch(
                "metaculus_bot.research.targeted.build_native_search_llm",
            ) as resolver_builder,
            _cli_main_test_mode(
                alertable_count=0,
                forecaster_class=forecaster_class,
                today=AFTER_RESUME_DATE,
            ),
        ):

            async def _record_forecast(*_args: object, **_kwargs: object) -> list[object]:
                await bot._research.run_research(question)
                events.append("forecast")
                return []

            bot.forecast_questions = AsyncMock(side_effect=_record_forecast)
            with pytest.raises(SystemExit) as exc_info:
                cli_main()
            assert exc_info.value.code == 1

        assert bot._research.gap_fill_v1_error_count == 1
        assert bot.alertable_count == 1
        assert analyzer.invoke.await_count == 1
        resolver_builder.assert_not_called()
        assert events == ["forecast", "report_summary"]

    def test_real_v1_empty_analysis_stays_green(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A valid empty analyzer list follows the same real path without alerting."""
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.delenv("GAP_FILL_V2_ENABLED", raising=False)

        bot = _bot_with_real_alertable_count()
        bot._research._custom_provider = AsyncMock(return_value="first-pass research " * 20)
        question = MagicMock()
        question.id_of_question = 45124
        question.page_url = "https://example.com/questions/45124"
        question.question_text = "Will X happen?"
        question.resolution_criteria = "Resolves on X."
        question.fine_print = ""
        question.options = None
        analyzer = MagicMock(invoke=AsyncMock(return_value=json.dumps({"gaps": []})))

        with (
            patch("metaculus_bot.fallback_openrouter.build_llm_with_openrouter_fallback", return_value=analyzer),
            patch(
                "metaculus_bot.research.targeted.build_native_search_llm",
            ) as resolver_builder,
            _cli_main_test_mode(alertable_count=0, stub_bot=bot, today=AFTER_RESUME_DATE),
        ):

            async def _record_forecast(*_args: object, **_kwargs: object) -> list[object]:
                await bot._research.run_research(question)
                return []

            bot.forecast_questions = AsyncMock(side_effect=_record_forecast)
            cli_main()

        assert bot._research.gap_fill_v1_error_count == 0
        assert bot.alertable_count == 0
        assert analyzer.invoke.await_count == 1
        resolver_builder.assert_not_called()

    def test_the_mantic_drop_term_is_absent_when_nothing_was_dropped(self, caplog: pytest.LogCaptureFixture) -> None:
        """Rendered only when it applies, so the registry's optional group and the reader agree."""
        with _cli_main_test_mode(alertable_count=0), caplog.at_level(logging.INFO, logger="metaculus_bot.cli"):
            cli_main()
        assert "mantic_post_drops" not in caplog.text


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
            _cli_main_test_mode(alertable_count=0, forecaster_class=forecaster_class, today=AFTER_RESUME_DATE),
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
            _cli_main_test_mode(alertable_count=0, forecaster_class=forecaster_class, today=AFTER_RESUME_DATE),
            caplog.at_level(logging.WARNING, logger="metaculus_bot.cli"),
            pytest.raises(RuntimeError),
        ):
            cli_main()

        breakdown = next(m for m in caplog.messages if m.startswith("Run completed with"))
        assert breakdown.startswith("Run completed with 3 alertable")
