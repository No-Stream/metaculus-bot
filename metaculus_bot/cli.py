import argparse
import asyncio

# ruff: noqa: F401
import logging
import os
import sys
from typing import Literal

from forecasting_tools import MetaculusApi

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import (
    METACULUS_CUP_ID,
    PERSIST_RESEARCH_ENABLED_ENV,
    TOURNAMENT_ID,
    check_tournament_dates,
    env_flag_enabled,
)
from metaculus_bot.fallback_openrouter import (
    check_deprecation_alerts_and_exit,
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

    # Wrap MetaculusApi publish POSTs with timeout + retry. See
    # metaculus_bot/publish_hardening.py for rationale (stock requests.post
    # has no timeout; a single hung POST blocks the whole batch).
    apply_publish_hardening()

    # Wrap MetaculusApi question-list GET with timeout + bounded retry. See
    # metaculus_bot/fetch_hardening.py for rationale (stock requests.get has
    # no timeout/retry; a single transient 403/429/5xx kills the whole run).
    apply_fetch_hardening()

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
        from metaculus_bot.research.persistence import ResearchPersistenceWriter  # noqa: PLC0415

        research_writer = ResearchPersistenceWriter(
            run_mode=run_mode,
            tournament_id=str(TOURNAMENT_ID),
            run_id=os.environ.get("GITHUB_RUN_ID", "local"),
        )
        research_sink = research_writer.record

    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,  # Ignored when 'forecasters' present
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING,
        research_sink=research_sink,
        llms={
            "forecasters": FORECASTER_LLMS,
            "stacker": STACKER_LLM,
            "analyzer": DISAGREEMENT_ANALYZER_LLM,
            "summarizer": SUMMARIZER_LLM,
            "parser": PARSER_LLM,
            "researcher": RESEARCHER_LLM,
        },
    )

    if run_mode == "tournament":
        check_tournament_dates(logging.getLogger(__name__))  # Warn/error if tournament dates are stale
        template_bot.skip_previously_forecasted_questions = True  # to not risk explosive spend, we won't update preds
        forecast_reports = asyncio.run(template_bot.forecast_on_tournament(TOURNAMENT_ID, return_exceptions=True))
    elif run_mode == "minibench":
        template_bot.skip_previously_forecasted_questions = True  # to not risk explosive spend, we won't update preds
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True)
        )
    elif run_mode in ("quarterly_cup", "metaculus_cup"):
        # The metaculus cup is a good way to test the bot's performance on regularly open questions
        template_bot.skip_previously_forecasted_questions = True  # to not risk explosive spend, we won't update preds
        forecast_reports = asyncio.run(template_bot.forecast_on_tournament(METACULUS_CUP_ID, return_exceptions=True))
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
        questions = [MetaculusApi.get_question_by_url(url) for url in EXAMPLE_QUESTIONS]
        forecast_reports = asyncio.run(template_bot.forecast_questions(questions, return_exceptions=True))
    else:
        raise ValueError(f"Invalid run mode: {run_mode}")

    if research_writer is not None:
        research_writer.flush()

    TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore

    # Alert on degraded runs. Publication has already happened inside
    # forecast_on_tournament / forecast_questions above, so every Q that met
    # MIN_FORECASTERS_TO_PUBLISH is on Metaculus regardless of exit status.
    # Non-zero exit here just triggers the GitHub Actions red-check alert so
    # the operator knows to investigate (forecaster drops, stacker fallback
    # usage, research provider timeouts, etc. — see main.py `alertable_count`).
    bot_alertable = template_bot.alertable_count
    # Donated->personal key fallback: counted in fallback_openrouter at the
    # wrapper level (process-global, since the wrapper has no link back to the
    # bot). Each fallback was successful — the run completed using the paid
    # personal key — but a call that should have hit the free donated key
    # billed to the operator instead, so the operator should investigate.
    # ``generic_fallback`` counts ALL fallback causes (401/402/429/guardrail/
    # 404); ``donated_404`` is the allowed-providers-404 subset of that total,
    # broken out for diagnostics. Add only ``generic_fallback`` to ``alertable``
    # — adding ``donated_404`` too would double-count the 404 subset.
    generic_fallback = get_generic_key_fallback_count()
    donated_404 = get_donated_404_fallback_count()
    alertable = bot_alertable + generic_fallback
    if alertable > 0:
        logger.warning(
            "Run completed with %d alertable degradation event(s) "
            "(bot=%d, personal_key_fallback=%d of which donated_404=%d); "
            "exiting non-zero so CI marks this run red.",
            alertable,
            bot_alertable,
            generic_fallback,
            donated_404,
        )
        sys.exit(1)

    # Post-submission deprecation tripwire. Runs LAST so submission has fully
    # completed (and so other alertable conditions exit first with their own
    # log lines). When OpenRouter retires a model the bot uses (e.g. the
    # 2026-05-15 x-ai/grok-4.1-fast deprecation that silently 404'd for ~2
    # days), this prints a loud banner + sys.exit(1) so GitHub Actions turns
    # red. Returns silently when no deprecation was observed.
    check_deprecation_alerts_and_exit()


if __name__ == "__main__":
    main()
