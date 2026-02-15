"""
DEPRECATED: Community prediction benchmark.

Scores bot predictions against the Metaculus community prediction as a proxy for ground truth.
Metaculus removed the ``aggregations`` field from their list API, so
``community_prediction_at_access_time`` is now always None for newly-fetched questions.
The ``expected_baseline_score`` metric is therefore broken for new runs.

Prefer ``backtest.py`` which scores against actual question resolutions.
"""

import argparse
import asyncio
import atexit
import logging
import os
import random
import sys
import time
import weakref
from datetime import datetime, timedelta
from typing import Literal

import aiohttp
import typeguard
from forecasting_tools import (
    ApiFilter,
    Benchmarker,
    ForecastBot,
    MetaculusApi,
    MonetaryCostManager,
    run_benchmark_streamlit_page,
)
from tqdm import tqdm

from metaculus_bot.benchmark.bot_factory import (
    BENCHMARK_BOT_CONFIG,
    DEFAULT_HELPER_LLMS,
    INDIVIDUAL_MODEL_SPECS,
    STACKING_MODEL_SPECS,
    create_individual_bots,
    create_stacking_bots,
)
from metaculus_bot.benchmark.heartbeat import install_benchmarker_heartbeat
from metaculus_bot.benchmark.logging import log_benchmarker_headline_note, log_bot_lineup, log_stacking_summaries
from metaculus_bot.benchmark.logging_setup import configure_benchmark_logging
from metaculus_bot.config import load_environment
from metaculus_bot.constants import (
    BENCHMARK_BATCH_SIZE,
    FETCH_PACING_SECONDS,
    FETCH_RETRY_BACKOFFS,
    HEARTBEAT_INTERVAL,
    TYPE_MIX,
)
from metaculus_bot.scoring_patches import (
    apply_scoring_patches,
    log_score_scale_validation,
    log_scoring_path_stats,
    reset_scoring_path_stats,
)

logger = logging.getLogger(__name__)

load_environment()


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


# Global progress tracking state
_progress_state = {
    "total_predictions": 0,
    "start_time": 0,
    "completed_batches": 0,
    "total_batches": 0,
    "pbar": None,
}


install_benchmarker_heartbeat(HEARTBEAT_INTERVAL, _progress_state)


async def _get_mixed_question_types(total_questions: int, one_year_from_now: datetime) -> list:
    """Get mixed question types with 50/25/25 distribution (binary/numeric/multiple-choice).

    Reliability enhancements:
    - Add 2 retries with 5s then 15s backoff on transient fetch errors
    - Sleep 2s between type fetches to reduce burstiness
    - Fail fast with a clear error if an expected type cannot be fetched
    """

    # Calculate counts for each type (50/25/25 distribution)
    binary_count = int(total_questions * TYPE_MIX[0])
    numeric_count = int(total_questions * TYPE_MIX[1])
    mc_count = total_questions - binary_count - numeric_count  # Remainder goes to MC

    logger.info(f"Fetching mixed questions: {binary_count} binary, {numeric_count} numeric, {mc_count} multiple-choice")

    # Base filter settings for all question types
    base_filter_kwargs = {
        "allowed_statuses": ["open"],
        "num_forecasters_gte": 40,
        "scheduled_resolve_time_lt": one_year_from_now,
        "includes_bots_in_aggregates": False,
        "open_time_gt": datetime.now() - timedelta(days=90),
    }

    all_questions = []

    # Helper: fetch with retries and backoff
    async def _fetch_type_with_retries(question_type: str, count: int) -> list:
        import http.client

        from requests import exceptions as req_exc  # type: ignore
        from urllib3 import exceptions as ul3_exc  # type: ignore

        # Build filter per type
        filter_kwargs = base_filter_kwargs.copy()

        # For numeric questions, include discrete types as well
        if question_type == "numeric":
            allowed_types = ["numeric", "discrete"]
        else:
            allowed_types = [question_type]

        api_filter = ApiFilter(allowed_types=allowed_types, **filter_kwargs)

        def _is_retryable_error(err: Exception) -> bool:
            retryables = (
                req_exc.ConnectionError,
                req_exc.Timeout,
                ul3_exc.ProtocolError,
                http.client.RemoteDisconnected,
            )
            if isinstance(err, retryables):
                return True
            # Best-effort string check for common transient statuses when wrapped
            msg = str(err).lower()
            return any(tok in msg for tok in ["429", "too many requests", "502", "503", "504", "timeout"])  # type: ignore[return-value]

        attempts = 0
        backoffs = list(FETCH_RETRY_BACKOFFS)  # seconds
        while True:
            try:
                logger.info(f"🔍 Attempt {attempts + 1}: fetching {count} {question_type} questions...")
                sys.stdout.flush()
                questions = await MetaculusApi.get_questions_matching_filter(
                    api_filter,
                    num_questions=count,
                    randomly_sample=False,
                )
                if not questions:
                    raise RuntimeError("API returned 0 questions")
                return questions
            except Exception as e:  # Retry on transient errors, otherwise raise
                if attempts < 2 and _is_retryable_error(e):
                    sleep_s = backoffs[attempts] if attempts < len(backoffs) else backoffs[-1]
                    logger.warning(
                        f"Retryable error fetching {question_type} questions (attempt {attempts + 1}/3): {e}. "
                        f"Backing off {sleep_s}s before retry."
                    )
                    sys.stdout.flush()
                    await asyncio.sleep(sleep_s)
                    attempts += 1
                    continue
                # Final failure or non-retryable
                logger.error(f"❌ Failed to fetch {question_type} questions: {e}")
                import traceback

                logger.error(f"Full traceback: {traceback.format_exc()}")
                sys.stdout.flush()
                raise RuntimeError(
                    f"Aborting benchmark: unable to fetch {question_type} questions after {attempts + 1} attempts"
                ) from e

    # Fetch each question type separately with pacing and validation
    types_and_counts = [
        ("binary", binary_count),
        ("numeric", numeric_count),
        ("multiple_choice", mc_count),
    ]
    for i, (question_type, count) in enumerate(types_and_counts, 1):
        if count <= 0:
            continue
        logger.info(f"[{i}/3] Fetching {count} {question_type} questions...")
        sys.stdout.flush()
        questions = await _fetch_type_with_retries(question_type, count)
        logger.info(f"✅ Successfully fetched {len(questions)} {question_type} questions")
        if questions:
            logger.info(f"📋 Sample {question_type} question: {questions[0].question_text[:100]}...")
        all_questions.extend(questions)
        sys.stdout.flush()

        # Intentional pacing between types
        if i < len(types_and_counts):
            await asyncio.sleep(FETCH_PACING_SECONDS)

    # Shuffle to avoid clustering by type
    random.shuffle(all_questions)

    # Clear background_info for all questions (to test ability to find new information)
    for question in all_questions:
        question.background_info = None

    # Log final distribution
    type_counts = {}
    for q in all_questions:
        q_type = type(q).__name__
        type_counts[q_type] = type_counts.get(q_type, 0) + 1

    logger.info(f"Final mixed question distribution: {type_counts}")
    return all_questions


async def benchmark_forecast_bot(
    mode: str,
    number_of_questions: int = 2,
    mixed_types: bool = False,
    include_models: list[str] | None = None,
    exclude_models: list[str] | None = None,
) -> None:
    """
    DEPRECATED: Run a benchmark that compares your forecasts against the community prediction.
    Prefer backtest.py which scores against actual question resolutions.
    """
    logger.warning(
        "community_benchmark.py is deprecated. Metaculus removed the aggregations field "
        "from their list API, so community_prediction_at_access_time is always None for "
        "newly-fetched questions and expected_baseline_score is unreliable. "
        "Use backtest.py instead."
    )

    if mode == "display":
        run_benchmark_streamlit_page()
        return
    elif mode == "run":
        api_filter = ApiFilter(
            allowed_statuses=["open"],
            allowed_types=["binary"],
            num_forecasters_gte=30,
            includes_bots_in_aggregates=False,
            open_time_gt=datetime.now() - timedelta(days=90),
        )
        questions = await MetaculusApi.get_questions_matching_filter(
            api_filter,
            num_questions=number_of_questions,
            randomly_sample=False,
        )
    elif mode == "custom":
        # Below is an example of getting custom questions
        one_year_from_now = datetime.now() + timedelta(days=365)

        if mixed_types:
            # Get mixed question types with 50/25/25 distribution
            questions = await _get_mixed_question_types(number_of_questions, one_year_from_now)
        else:
            # Original binary-only approach
            api_filter = ApiFilter(
                allowed_statuses=["open"],
                allowed_types=["binary"],
                num_forecasters_gte=40,
                scheduled_resolve_time_lt=one_year_from_now,
                includes_bots_in_aggregates=False,
                open_time_gt=datetime.now() - timedelta(days=90),
            )
            questions = await MetaculusApi.get_questions_matching_filter(
                api_filter,
                num_questions=number_of_questions,
                randomly_sample=False,
            )

        for question in questions:
            question.background_info = None  # Test ability to find new information
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Apply scoring patches for mixed question types and reset counters
    apply_scoring_patches()
    reset_scoring_path_stats()

    with MonetaryCostManager() as cost_manager:
        # Keep benchmark and bot research concurrency aligned
        batch_size = BENCHMARK_BATCH_SIZE

        # Shared research cache for all bots to avoid duplicate API calls
        research_cache: dict[int, str] = {}
        individual_specs = INDIVIDUAL_MODEL_SPECS
        base_forecasters = [spec["forecaster"] for spec in individual_specs]
        if len(base_forecasters) < 2:
            logger.warning(
                "STACKING configuration: fewer than 2 base forecasters (%d). Stacking quality may suffer.",
                len(base_forecasters),
            )

        stacking_specs = STACKING_MODEL_SPECS

        bots = create_individual_bots(
            individual_specs,
            DEFAULT_HELPER_LLMS,
            BENCHMARK_BOT_CONFIG,
            batch_size=batch_size,
            research_cache=research_cache,
        )

        stacking_bots = create_stacking_bots(
            stacking_specs,
            list(base_forecasters),
            DEFAULT_HELPER_LLMS,
            BENCHMARK_BOT_CONFIG,
            batch_size=batch_size,
            research_cache=research_cache,
        )

        bots.extend(stacking_bots)

        logger.info(
            f"Created {len(bots)} total bots for benchmarking: {len(individual_specs)} individual models + {len(stacking_specs)} stacking models. "
            f"Traditional ensembles will be generated post-hoc by correlation analysis."
        )
        bots = typeguard.check_type(bots, list[ForecastBot])

        # Log progress info
        total_predictions = len(bots) * len(questions)
        logger.info(
            f"🚀 Starting benchmark: {len(bots)} bots x {len(questions)} questions = {total_predictions} total predictions"
        )
        sys.stdout.flush()  # Ensure this critical message appears immediately

        # Initialize progress tracking
        _progress_state.update(
            {
                "total_predictions": total_predictions,
                "start_time": time.time(),
                "completed_batches": 0,
                "total_batches": len(bots),  # Each bot runs as a separate "batch"
                "pbar": tqdm(total=total_predictions, desc="Forecasting", unit="predictions"),
            }
        )

        # Pre-run per-bot overview for clarity
        try:
            log_bot_lineup(bots)
        except Exception:
            pass

        logger.info("📊 Entering Benchmarker.run_benchmark() - this may take a while...")
        sys.stdout.flush()

        benchmarks = await Benchmarker(
            questions_to_use=questions,
            forecast_bots=bots,
            file_path_to_save_reports="benchmarks/",
            concurrent_question_batch_size=batch_size,
        ).run_benchmark()

        # Close progress bar
        if _progress_state["pbar"] is not None:
            _progress_state["pbar"].close()
            _progress_state["pbar"] = None

        logger.info("✅ Benchmarker.run_benchmark() completed, processing results...")
        sys.stdout.flush()
        try:
            for i, benchmark in enumerate(benchmarks):
                logger.info(f"Benchmark {i + 1} of {len(benchmarks)}: {benchmark.name}")
                logger.info(
                    f"- Final Metaculus Baseline Score: {benchmark.average_expected_baseline_score:.4f} (based on log score, 0=always predict same, https://www.metaculus.com/help/scores-faq/#baseline-score )"
                )
                logger.info(f"- Total Cost: {benchmark.total_cost:.2f}")
                logger.info(f"- Time taken: {benchmark.time_taken_in_minutes:.4f}")
            log_benchmarker_headline_note()
        except ValueError as ve:
            # Provide clearer guidance when no reports exist (likely research provider failures)
            raise RuntimeError(
                "Benchmark produced no forecast reports.Fallback is disabled for benchmarks by design."
            ) from ve
        logger.info(f"Total Cost: {cost_manager.current_usage}")

        # Log score scale validation for mixed question types
        log_score_scale_validation(benchmarks)

        # Summarize scoring path usage and flag if fallbacks dominate
        log_scoring_path_stats()

        # TODO: refactor out this logic, jank to have here.
        # Perform correlation analysis if we have multiple models
        if len(benchmarks) > 1:
            from metaculus_bot.correlation_analysis import CorrelationAnalyzer

            analyzer = CorrelationAnalyzer()
            analyzer.add_benchmark_results(benchmarks)

            # Optional model filtering prior to report/ensembles
            if include_models and exclude_models:
                logger.warning("Both include and exclude provided; include takes precedence, excludes still applied.")
            if include_models or exclude_models:
                summary = analyzer.filter_models_inplace(include=include_models, exclude=exclude_models)
                try:
                    logger.info("Model filters applied:")
                    if include_models:
                        logger.info(f"  include tokens: {include_models}")
                        if summary.get("unmatched_includes"):
                            logger.info(f"  unmatched include tokens: {summary['unmatched_includes']}")
                    if exclude_models:
                        logger.info(f"  exclude tokens: {exclude_models}")
                        if summary.get("unmatched_excludes"):
                            logger.info(f"  unmatched exclude tokens: {summary['unmatched_excludes']}")
                    logger.info(f"  remaining models: {analyzer.get_model_names()}")
                except Exception:
                    pass

            # Generate and log correlation report
            report = analyzer.generate_correlation_report("benchmarks/correlation_analysis.md")
            logger.info("\n" + "=" * 50)
            logger.info("CORRELATION ANALYSIS")
            logger.info("=" * 50)
            logger.info(report)

            # Generate all possible ensemble combinations with different aggregation strategies
            logger.info("\n" + "=" * 50)
            logger.info("ENSEMBLE GENERATION (Post-hoc)")
            logger.info("=" * 50)
            optimal_ensembles = analyzer.find_optimal_ensembles(max_ensemble_size=6, max_cost_per_question=1.0)
            if optimal_ensembles:
                logger.info(
                    f"Generated {len(optimal_ensembles)} ensemble combinations from {len(benchmarks)} individual models"
                )
                logger.info("\nTop 10 Recommended Ensembles (Both Aggregation Strategies, Cost ≤ $1.0/question):")
                for i, ensemble in enumerate(optimal_ensembles[:10], 1):
                    models = " + ".join(ensemble.model_names)
                    logger.info(f"{i}. {models} ({ensemble.aggregation_strategy.upper()})")
                    logger.info(
                        f"   Score: {ensemble.avg_performance:.2f} | "
                        f"Cost: ${ensemble.avg_cost:.3f} | "
                        f"Diversity: {ensemble.diversity_score:.3f} | "
                        f"Overall: {ensemble.ensemble_score:.3f}"
                    )

                logger.info(
                    f"\n💡 Use 'python analyze_correlations.py benchmarks/' to explore all {len(optimal_ensembles)} ensemble combinations"
                )
            else:
                logger.info("No viable ensemble combinations found within cost constraints")
        else:
            logger.info("Skipping correlation analysis (need multiple models)")

        # Summarize any STACKING fallbacks and guard triggers encountered
        try:
            log_stacking_summaries(stacking_bots)
        except Exception:
            pass


if __name__ == "__main__":
    # Force unbuffered output for real-time logging in long-running processes
    os.environ["PYTHONUNBUFFERED"] = "1"

    configure_benchmark_logging()

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
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Use mixed question types with 50/25/25 distribution (binary/numeric/multiple-choice)",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=None,
        help=("Exclude models by substring match (case-insensitive). Example: --exclude-models grok-4 gemini-2.5-pro"),
    )
    parser.add_argument(
        "--include-models",
        nargs="*",
        default=None,
        help=(
            "Only include models matching these substrings (case-insensitive). "
            "Mutually exclusive with --exclude-models."
        ),
    )
    args = parser.parse_args()
    mode: Literal["run", "custom", "display"] = args.mode
    asyncio.run(
        benchmark_forecast_bot(
            mode,
            args.num_questions,
            args.mixed,
            include_models=args.include_models,
            exclude_models=args.exclude_models,
        )
    )
