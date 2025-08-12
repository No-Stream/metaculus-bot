from __future__ import annotations

import argparse
import asyncio
import atexit
import logging
import sys
import weakref
from datetime import datetime, timedelta
from typing import Literal

import aiohttp
import typeguard
from dotenv import load_dotenv
from forecasting_tools import (
    ApiFilter,
    Benchmarker,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MonetaryCostManager,
    run_benchmark_streamlit_page,
)

from main import TemplateForecaster
from metaculus_bot.constants import BENCHMARK_BATCH_SIZE
from metaculus_bot.llm_configs import FORECASTER_LLMS, PARSER_LLM, RESEARCHER_LLM, SUMMARIZER_LLM

logger = logging.getLogger(__name__)

load_dotenv()
load_dotenv(".env.local", override=True)


# Quick mitigation for occasional "Unclosed client session" warnings from aiohttp when
# using EXA search under high concurrency. Tracks sessions and closes them at process exit.
def _enable_aiohttp_session_autoclose() -> None:
    open_sessions: "weakref.WeakSet[aiohttp.ClientSession]" = weakref.WeakSet()
    original_init = aiohttp.ClientSession.__init__

    def tracking_init(self: aiohttp.ClientSession, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        open_sessions.add(self)

    aiohttp.ClientSession.__init__ = tracking_init  # type: ignore[assignment]

    def _close_open_sessions() -> None:
        to_close = [s for s in list(open_sessions) if not s.closed]
        if not to_close:
            return
        logger.debug(f"Closing {len(to_close)} lingering aiohttp sessions at exit")

        async def _close_all() -> None:
            for s in to_close:
                try:
                    await s.close()
                except Exception as e:  # pragma: no cover - best-effort cleanup
                    logger.debug(f"Error closing aiohttp session at exit: {e}")

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.create_task(_close_all())
        else:
            try:
                asyncio.run(_close_all())
            except RuntimeError:
                new_loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(new_loop)
                    new_loop.run_until_complete(_close_all())
                finally:
                    new_loop.close()

    atexit.register(_close_open_sessions)


_enable_aiohttp_session_autoclose()


async def benchmark_forecast_bot(mode: str, number_of_questions: int = 2) -> None:
    """
    Run a benchmark that compares your forecasts against the community prediction.
    Ideally 100+ questions for meaningful error bars, but can use e.g. just a few for smoke testing or 30 for a quick run.
    """
    # TODO: make sure this is ok w/ the max predictions at once cost safety controls we have in place

    if mode == "display":
        run_benchmark_streamlit_page()
        return
    elif mode == "run":
        questions = MetaculusApi.get_benchmark_questions(number_of_questions)
    elif mode == "custom":
        # Below is an example of getting custom questions
        one_year_from_now = datetime.now() + timedelta(days=365)
        api_filter = ApiFilter(
            allowed_statuses=["open"],
            allowed_types=["binary"],
            num_forecasters_gte=40,
            scheduled_resolve_time_lt=one_year_from_now,
            includes_bots_in_aggregates=False,
            community_prediction_exists=True,
        )
        questions = await MetaculusApi.get_questions_matching_filter(
            api_filter,
            num_questions=number_of_questions,
            randomly_sample=True,
        )
        for question in questions:
            question.background_info = None  # Test ability to find new information
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Shared configuration for all benchmark bots
    BENCHMARK_BOT_CONFIG = {
        "research_reports_per_question": 1,
        "predictions_per_research_report": 1,  # Ignored when forecasters present
        "use_research_summary_to_forecast": False,
        "publish_reports_to_metaculus": False,  # Don't publish during benchmarking
        "folder_to_save_reports_to": None,
        "skip_previously_forecasted_questions": False,
        "numeric_aggregation_method": "mean",
        "research_provider": None,  # Use default provider selection
        "max_questions_per_run": None,  # No limit for benchmarking
        "is_benchmarking": True,  # Exclude prediction markets to avoid data leakage
        "allow_research_fallback": False,  # Ensure AskNews runs; do not fallback in benchmark
    }
    MODEL_CONFIG = {
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": 8000,  # Prevent truncation issues with reasoning models
        "stream": False,
        "timeout": 180,
        "allowed_tries": 3,
    }
    DEFAULT_HELPER_LLMS = {
        "summarizer": SUMMARIZER_LLM,
        "parser": PARSER_LLM,
        "researcher": RESEARCHER_LLM,
    }

    with MonetaryCostManager() as cost_manager:
        # Keep benchmark and bot research concurrency aligned
        batch_size = BENCHMARK_BATCH_SIZE
        bots = [
            # Full ensemble bot using production configuration
            # TemplateForecaster(
            #     **BENCHMARK_BOT_CONFIG,
            #     llms={
            #         "forecasters": FORECASTER_LLMS,  # Our current ensemble approach
            #         "summarizer": SUMMARIZER_LLM,
            #         "parser": PARSER_LLM,
            #         "researcher": RESEARCHER_LLM,
            #     },
            # ),
            TemplateForecaster(
                **BENCHMARK_BOT_CONFIG,
                llms={
                    "forecasters": [
                        GeneralLlm(
                            model="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
                            **MODEL_CONFIG,
                        )
                    ],
                    **DEFAULT_HELPER_LLMS,
                },
                max_concurrent_research=batch_size,
            ),
            # TemplateForecaster(
            #     **BENCHMARK_BOT_CONFIG,
            #     llms={
            #         "forecasters": [
            #             GeneralLlm(
            #                 model="openrouter/z-ai/glm-4.5",
            #                 **MODEL_CONFIG,
            #             )
            #         ],
            #         **DEFAULT_HELPER_LLMS,
            #     },
            #     max_concurrent_research=batch_size,
            # ),
            # TemplateForecaster(
            #     **BENCHMARK_BOT_CONFIG,
            #     llms={
            #         "forecasters": [
            #             GeneralLlm(
            #                 model="openrouter/deepseek/deepseek-r1-0528",
            #                 **MODEL_CONFIG,
            #             )
            #         ],
            #         **DEFAULT_HELPER_LLMS,
            #     },
            #     max_concurrent_research=batch_size,
            # ),
            # TemplateForecaster(
            #     **BENCHMARK_BOT_CONFIG,
            #     llms={
            #         "forecasters": [
            #             GeneralLlm(
            #                 model="openrouter/anthropic/claude-sonnet-4",
            #                 reasoning={"max_tokens": 4000},
            #                 **MODEL_CONFIG,
            #             )
            #         ],
            #         **DEFAULT_HELPER_LLMS,
            #     },
            #     max_concurrent_research=batch_size,
            # ),
            # TemplateForecaster(
            #     **BENCHMARK_BOT_CONFIG,
            #     llms={
            #         "forecasters": [
            #             GeneralLlm(
            #                 model="openrouter/openai/gpt-5",
            #                 reasoning_effort="high",
            #                 **MODEL_CONFIG,
            #             )
            #         ],
            #         **DEFAULT_HELPER_LLMS,
            #     },
            #     max_concurrent_research=batch_size,
            # ),
            # TemplateForecaster(
            #     **BENCHMARK_BOT_CONFIG,
            #     llms={
            #         "forecasters": [
            #             GeneralLlm(
            #                 model="openrouter/google/gemini-2.5-pro",
            #                 reasoning={"max_tokens": 8000},
            #                 **MODEL_CONFIG,
            #             )
            #         ],
            #         **DEFAULT_HELPER_LLMS,
            #     },
            # ),
            # TemplateForecaster(
            #     **BENCHMARK_BOT_CONFIG,
            #     llms={
            #         "forecasters": [
            #             GeneralLlm(
            #                 model="openrouter/openai/o3",
            #                 reasoning_effort="high",
            #                 **MODEL_CONFIG,
            #             )
            #         ],
            #         **DEFAULT_HELPER_LLMS,
            #     },
            # ),
            # TemplateForecaster(
            #     **BENCHMARK_BOT_CONFIG,
            #     llms={
            #         "forecasters": [
            #             GeneralLlm(
            #                 model="openrouter/x-ai/grok-4",
            #                 reasoning={"effort": "high"},
            #                 **MODEL_CONFIG,
            #             )
            #         ],
            #         **DEFAULT_HELPER_LLMS,
            #     },
            #     max_concurrent_research=batch_size,
            # ),
        ]
        bots = typeguard.check_type(bots, list[ForecastBot])

        # Log progress info
        total_predictions = len(bots) * len(questions)
        logger.info(
            f"🚀 Starting benchmark: {len(bots)} bots × {len(questions)} questions = {total_predictions} total predictions"
        )

        benchmarks = await Benchmarker(
            questions_to_use=questions,
            forecast_bots=bots,
            file_path_to_save_reports="benchmarks/",
            concurrent_question_batch_size=batch_size,
        ).run_benchmark()
        try:
            for i, benchmark in enumerate(benchmarks):
                logger.info(f"Benchmark {i+1} of {len(benchmarks)}: {benchmark.name}")
                logger.info(
                    f"- Final Metaculus Baseline Score: {benchmark.average_expected_baseline_score:.4f} (based on log score, 0=always predict same, https://www.metaculus.com/help/scores-faq/#baseline-score )"
                )
                logger.info(f"- Total Cost: {benchmark.total_cost:.2f}")
                logger.info(f"- Time taken: {benchmark.time_taken_in_minutes:.4f}")
        except ValueError as ve:
            # Provide clearer guidance when no reports exist (likely research provider failures)
            raise RuntimeError(
                "Benchmark produced no forecast reports. Likely all research calls failed due to AskNews API issues. "
                "Check AskNews credentials, account status, and API permissions. "
                "Fallback is disabled for benchmarks by design."
            ) from ve
        logger.info(f"Total Cost: {cost_manager.current_usage}")

        # Perform correlation analysis if we have multiple models
        if len(benchmarks) > 1:
            from metaculus_bot.correlation_analysis import CorrelationAnalyzer

            analyzer = CorrelationAnalyzer()
            analyzer.add_benchmark_results(benchmarks)

            # Generate and log correlation report
            report = analyzer.generate_correlation_report("benchmarks/correlation_analysis.md")
            logger.info("\n" + "=" * 50)
            logger.info("CORRELATION ANALYSIS")
            logger.info("=" * 50)
            logger.info(report)

            # Log optimal ensembles for easy reference
            optimal_ensembles = analyzer.find_optimal_ensembles(max_ensemble_size=6, max_cost_per_question=1.0)
            if optimal_ensembles:
                logger.info(f"\nTop 3 Recommended Ensembles (Cost ≤ $0.50/question):")
                for i, ensemble in enumerate(optimal_ensembles[:3], 1):
                    models = " + ".join(ensemble.model_names)
                    logger.info(
                        f"{i}. {models} | Score: {ensemble.avg_performance:.2f} | "
                        f"Cost: ${ensemble.avg_cost:.3f} | Diversity: {ensemble.diversity_score:.3f}"
                    )
        else:
            logger.info("Skipping correlation analysis (need multiple models)")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f"benchmarks/log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"),
        ],
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark a list of bots")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "custom", "display"],
        default="display",
        help="Specify the run mode (default: display)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=2,
        help="Number of questions to benchmark (default: 2)",
    )
    args = parser.parse_args()
    mode: Literal["run", "custom", "display"] = args.mode
    asyncio.run(benchmark_forecast_bot(mode, args.num_questions))
