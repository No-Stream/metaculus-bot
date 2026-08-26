import argparse
import asyncio

# ruff: noqa: F401
import logging
import os
import sys
from typing import Any, Literal

from forecasting_tools import MetaculusApi

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.api_preflight import verify_metaculus_api_identity
from metaculus_bot.constants import (
    CREDIT_ALERT_RESUME_DATE,
    METACULUS_CUP_ID,
    PERSIST_RESEARCH_ENABLED_ENV,
    TEST_QUESTIONS_OVERRIDE_ENV,
    TOURNAMENT_ID,
    check_tournament_dates,
    credit_alerts_active,
    env_flag_enabled,
)
from metaculus_bot.credit_telemetry import CreditTelemetry, get_probed_donated_key_state
from metaculus_bot.fallback_openrouter import (
    check_deprecation_alerts_and_exit,
    get_credit_key_fallback_count,
    get_donated_404_fallback_count,
    get_generic_key_fallback_count,
)
from metaculus_bot.fetch_hardening import apply_fetch_hardening
from metaculus_bot.forecaster import TemplateForecaster
from metaculus_bot.llm_configs import (
    DISAGREEMENT_ANALYZER_LLM,
    FORECASTER_LLMS,
    PARSER_LLM,
    RESEARCHER_LLM,
    STACKER_LLM,
    SUMMARIZER_LLM,
)
from metaculus_bot.publish_hardening import apply_publish_hardening

logger = logging.getLogger(__name__)


def main() -> None:
    """Command-line entry-point for running the TemplateForecaster.

    This code was moved verbatim from the bottom of main.py so external behaviour
    (e.g. GitHub Actions invoking `python main.py`) remains identical.  The only
    difference is that main.py now delegates to this function.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    # Forecaster module logs at DEBUG for full per-question tracing; the
    # openai-agents logger is noisy at INFO so pin it to ERROR. Configured here
    # (the runtime entry point) rather than at module import so test imports
    # and library consumers don't inherit these global level mutations.
    logging.getLogger("metaculus_bot.forecaster").setLevel(logging.DEBUG)
    logging.getLogger("openai.agents").setLevel(logging.ERROR)

    # Wrap MetaculusClient publish POSTs with timeout + retry. See
    # metaculus_bot/publish_hardening.py for rationale (a single hung POST
    # blocks the whole batch; we bound it tighter than the upstream default).
    apply_publish_hardening()

    # Wrap MetaculusClient question-list GET with bounded retry. See
    # metaculus_bot/fetch_hardening.py for rationale (a single transient
    # 403/429/5xx would otherwise kill the whole run).
    apply_fetch_hardening()

    # One-shot, unauthenticated identity check before any mode sends the token.
    # See metaculus_bot/api_preflight.py (DNS-parking incident): aborts non-zero
    # if www.metaculus.com isn't answered by the real API, so we never leak
    # METACULUS_TOKEN to a hijacked host.
    verify_metaculus_api_identity()

    parser = argparse.ArgumentParser(description="Run the Q1TemplateBot forecasting system")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "minibench", "quarterly_cup", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "minibench", "quarterly_cup", "metaculus_cup", "test_questions"] = args.mode

    # Wire research persistence if enabled (production GHA runs set this env var)
    research_writer = None
    research_sink = None
    if env_flag_enabled(PERSIST_RESEARCH_ENABLED_ENV):
        from metaculus_bot.research.persistence import (  # noqa: PLC0415, HARNESS-SCAN-EXEMPT-function-level-import
            ResearchPersistenceWriter,
        )

        research_writer = ResearchPersistenceWriter(
            run_mode=run_mode,
            tournament_id=str(TOURNAMENT_ID),
            run_id=os.environ.get("GITHUB_RUN_ID", "local"),
        )
        research_sink = research_writer.record

    # "forecasters" holds a list[GeneralLlm]; the helper slots hold single GeneralLlm
    # values. The parent ForecastBot.__init__ annotates llms as dict[str, str | GeneralLlm],
    # which (being invariant) cannot express the list value, so annotate the heterogeneous
    # dict as dict[str, Any]. prepare_llm_config consumes the "forecasters" list at runtime.
    llms: dict[str, Any] = {
        "forecasters": FORECASTER_LLMS,
        "stacker": STACKER_LLM,
        "analyzer": DISAGREEMENT_ANALYZER_LLM,
        "summarizer": SUMMARIZER_LLM,
        "parser": PARSER_LLM,
        "researcher": RESEARCHER_LLM,
    }
    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # Ignored when 'forecasters' present
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
        research_sink=research_sink,
        llms=llms,
    )

    # Credit-balance telemetry: CREDIT_BALANCE/CREDIT_SPEND marker lines land in
    # the run_logs/ artifact via the workflows' stdout tee, making per-run spend
    # on the shared donated key durably grep-able. The end fetch runs in a
    # finally so a crashed run still logs its spend; the floor check result is
    # consumed AFTER forecasting/publishing below (reminder signal, not abort).
    credit_telemetry = CreditTelemetry()
    credit_telemetry.log_start()
    donated_below_floor = False
    try:
        if run_mode == "tournament":
            check_tournament_dates(logging.getLogger(__name__))  # Warn/error if tournament dates are stale
            # to not risk explosive spend, we won't update preds
            template_bot.skip_previously_forecasted_questions = True
            forecast_reports = asyncio.run(template_bot.forecast_on_tournament(TOURNAMENT_ID, return_exceptions=True))
        elif run_mode == "minibench":
            # to not risk explosive spend, we won't update preds
            template_bot.skip_previously_forecasted_questions = True
            forecast_reports = asyncio.run(
                template_bot.forecast_on_tournament(MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True)
            )
        elif run_mode in ("quarterly_cup", "metaculus_cup"):
            # The metaculus cup is a good way to test the bot's performance on regularly open questions
            # to not risk explosive spend, we won't update preds
            template_bot.skip_previously_forecasted_questions = True
            forecast_reports = asyncio.run(
                template_bot.forecast_on_tournament(METACULUS_CUP_ID, return_exceptions=True)
            )
        elif run_mode == "test_questions":
            # Example questions are a good way to test the bot's performance on a single question
            EXAMPLE_QUESTIONS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
                "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
                # "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
                "https://www.metaculus.com/questions/20683/which-ai-world/",  # Scott Aaronson's five AI worlds
                "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",  # Number of US Labor Strikes Due to AI in 2029 - Discrete
            ]
            template_bot.skip_previously_forecasted_questions = (
                False  # obviously, we need to rerun test q predictions to test them :)
            )
            # Optional override (test_bot_basic workflow): a comma/whitespace-
            # separated list of Metaculus URLs to forecast instead of the full
            # evergreen set. Unset -> the hardcoded EXAMPLE_QUESTIONS above.
            override_urls = os.environ.get(TEST_QUESTIONS_OVERRIDE_ENV, "").replace(",", " ").split()
            question_urls = override_urls or EXAMPLE_QUESTIONS
            if override_urls:
                logger.info(
                    "TEST_QUESTIONS_OVERRIDE set: forecasting %d override question(s) instead of the evergreen set",
                    len(override_urls),
                )
            questions = [MetaculusApi.get_question_by_url(url) for url in question_urls]
            forecast_reports = asyncio.run(template_bot.forecast_questions(questions, return_exceptions=True))
        else:
            raise ValueError(f"Invalid run mode: {run_mode}")
    finally:
        donated_below_floor = credit_telemetry.log_end_and_check_floor()
        # Flush inside the finally: records accumulate in memory for the whole run,
        # so an exception escaping asyncio.run (an OSError, the invalid-run-mode
        # ValueError above, a KeyboardInterrupt, the 300-minute timeout-minutes
        # SIGTERM) would otherwise discard every question's research — a 40-question
        # run that dies on the last question would archive nothing. The workflows'
        # upload step is `if: always()`, so a crashed run's partial batch still
        # reaches the GHA artifact.
        if research_writer is not None:
            research_writer.flush()

    # The report summary RAISES by design when any report is an exception
    # (compact_log_report_summary re-raises so a failed question reddens CI under
    # return_exceptions=True) — but it used to sit ABOVE the alertable block, so
    # the one run that most needed a summary record left none: q45085's publish
    # failure (2026-08-03) propagated here, ``alertable`` was never computed, and
    # that run is the single forecasting run since 2026-07-26 with no
    # run_alertable_summary line in the archive. Emit-then-raise: hold the error,
    # emit the breakdown below on this path too, then re-raise — the original
    # exception keeps its traceback and CI stays exactly as red as before.
    report_summary_error: Exception | None = None
    try:
        TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore
    except Exception as exc:  # noqa: HARNESS-SCAN-EXEMPT-broad-except  # held only until the breakdown is emitted, then re-raised below
        report_summary_error = exc

    # Alert on degraded runs. Publication has already happened inside
    # forecast_on_tournament / forecast_questions above, so every Q that met
    # MIN_FORECASTERS_TO_PUBLISH is on Metaculus regardless of exit status.
    # Non-zero exit here just triggers the GitHub Actions red-check alert so
    # the operator knows to investigate (forecaster drops, stacker fallback
    # usage, research provider failures, etc. — see
    # `forecaster.py` `alertable_count`).
    bot_alertable = template_bot.alertable_count
    # Donated->personal key fallback: counted in fallback_openrouter at the
    # wrapper level (process-global, since the wrapper has no link back to the
    # bot). Each fallback was successful — the run completed using the paid
    # personal key — but a call that should have hit the free donated key
    # billed to the operator instead, so the operator should investigate.
    # ``generic_fallback`` counts ALL fallback causes (401/402/429/guardrail/
    # 404); ``donated_404`` and ``credit_fallback`` are two disjoint subsets of
    # that total, broken out for diagnostics. Add only ``generic_fallback`` to
    # ``alertable`` — adding either subset too would double-count events already
    # inside that total.
    #
    # Credit suppression (until CREDIT_ALERT_RESUME_DATE): the operator is
    # self-funding the rest of the season, so an empty donated key is expected
    # and its fallbacks are SUBTRACTED back out of the total. Every other cause
    # keeps its full weight, because 401/404/429/guardrail each mean real
    # breakage. Each event is still counted exactly once: generic adds it, and at
    # most one subset subtracts it.
    #
    # ``credit_fallback`` counts only the SUPPRESSIBLE credit subset — the
    # donated key genuinely drained. A key that was revoked or re-capped to zero
    # returns the same "Key limit exceeded" text but is classified separately by
    # ``fallback_openrouter.is_suppressible_credit_error`` (which probes
    # /auth/key), so it stays inside the generic total and keeps this run red.
    alerts_active = credit_alerts_active()
    generic_fallback = get_generic_key_fallback_count()
    donated_404 = get_donated_404_fallback_count()
    credit_fallback = get_credit_key_fallback_count()
    suppressed_credit_fallback = 0 if alerts_active else credit_fallback
    alertable = bot_alertable + generic_fallback - suppressed_credit_fallback

    suppression_note = (
        ""
        if alerts_active
        else f" with {suppressed_credit_fallback} credit event(s) suppressed until "
        f"{CREDIT_ALERT_RESUME_DATE.isoformat()}"
    )
    # Only rendered when a spend-cap failure actually made the wrapper probe the
    # donated key. Omitted otherwise, because "unknown" would read as a failed
    # probe rather than "no run this shape ever needed one".
    probed_donated_key_state = get_probed_donated_key_state()
    donated_key_note = "" if probed_donated_key_state is None else f", donated_key={probed_donated_key_state.value}"
    # One breakdown, EVERY path — degraded, suppressed-green, crashed, and fully
    # clean. The green paths need it as much as the red one: when every donated-key
    # call fell back and the credit subset cancels the whole generic total,
    # ``alertable`` is 0 — the exact shape of the 2026-07-26 drained-key run — and
    # gating this line on the exit status would leave that run's degradation and
    # probe verdict entirely unrecorded.
    #
    # A fully clean run says so explicitly, under a distinguishable "clean" phrase
    # that harvests as the same run_alertable_summary marker. It used to emit
    # NOTHING, so that the line's presence would stay a signal rather than
    # boilerplate; the operator OVERTURNED that on 2026-08-25. The reason: silence
    # is not distinguishable from a run that died before reaching this block, and
    # once the donated key is refilled (past CREDIT_ALERT_RESUME_DATE) the clean
    # shape becomes the COMMON one, so the archive's per-run census would lose
    # exactly the runs that went well. During the drained-key window the question
    # was moot — every run fell back at least once, and 0 of the 73 archived
    # records are the clean shape.
    #
    # A raising ``log_report_summary`` is never "clean" no matter what the counters
    # read: that run lost a question. Its counters can legitimately be all-zero
    # (q45085's shape), which is why the phrase, not the fields, is what marks a run
    # clean.
    run_clean = report_summary_error is None and alertable <= 0 and generic_fallback <= 0
    completion_phrase = "Run completed clean with" if run_clean else "Run completed with"
    breakdown = (
        f"{completion_phrase} {alertable} alertable degradation event(s) "
        f"(bot={bot_alertable}, personal_key_fallback={generic_fallback} of which "
        f"donated_404={donated_404}, credit={credit_fallback}{suppression_note}{donated_key_note});"
    )
    if report_summary_error is not None:
        # Emit-then-raise, never swallow: the breakdown line above is the record
        # the archive needs, and re-raising (rather than sys.exit) preserves the
        # forecasting failure's traceback in the log. Takes precedence over the
        # alertable exit below because the exception is the richer red signal.
        logger.warning("%s re-raising the forecasting failure so CI marks this run red.", breakdown)
        raise report_summary_error
    if alertable > 0:
        logger.warning("%s exiting non-zero so CI marks this run red.", breakdown)
        sys.exit(1)
    if generic_fallback > 0:
        # Reachable only under suppression with every fallback credit-caused (the
        # subtraction can't otherwise reach zero from a positive total), so state
        # that rather than leaving a reader to derive it from the arithmetic.
        logger.info("%s every fallback was a suppressed credit event, so this run stays green.", breakdown)
    else:
        # The all-clear census line (see ``run_clean`` above).
        logger.info("%s nothing degraded, so this run stays green.", breakdown)

    # Donated-key balance below the refill floor (CREDIT_FLOOR_BREACH warning
    # already logged by credit_telemetry). The run completed and published
    # normally; exiting non-zero here is purely the reminder-to-refill signal —
    # and it is suppressed until CREDIT_ALERT_RESUME_DATE, since a drained
    # donated key is the expected state while the operator self-funds. The INFO
    # line keeps the log self-explanatory: a reader who sees the breach WARNING
    # but a green run should not have to guess why.
    if donated_below_floor:
        if alerts_active:
            sys.exit(1)
        logger.info(
            "Donated-key credit floor breached, but credit alerting is suppressed until %s "
            "(operator is self-funding the rest of the season), so this run exits zero.",
            CREDIT_ALERT_RESUME_DATE.isoformat(),
        )

    # Post-submission deprecation tripwire. Runs LAST so submission has fully
    # completed (and so other alertable conditions exit first with their own
    # log lines). When OpenRouter retires a model the bot uses (e.g. the
    # 2026-05-15 x-ai/grok-4.1-fast deprecation that silently 404'd for ~2
    # days), this prints a loud banner + sys.exit(1) so GitHub Actions turns
    # red. Returns silently when no deprecation was observed.
    check_deprecation_alerts_and_exit()


if __name__ == "__main__":
    main()
