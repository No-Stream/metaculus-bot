"""Ground truth backtesting: score bot predictions against actual resolved outcomes."""

import argparse
import asyncio
import atexit
import logging
import os
import sys
import time
import weakref
from typing import Any

import aiohttp
import typeguard
from forecasting_tools import Benchmarker, ForecastBot, MonetaryCostManager
from tqdm import tqdm

from metaculus_bot.backtest.analysis import (
    BacktestResult,
    aggregate_scores,
    generate_backtest_report,
    save_backtest_data,
)
from metaculus_bot.backtest.leakage import screen_research_for_leakage
from metaculus_bot.backtest.question_prep import BacktestQuestionSet, fetch_resolved_questions
from metaculus_bot.backtest.scoring import QuestionScore, score_report
from metaculus_bot.benchmark.bot_factory import (
    BENCHMARK_BOT_CONFIG,
    DEFAULT_HELPER_LLMS,
    INDIVIDUAL_MODEL_SPECS,
    create_individual_bots,
)
from metaculus_bot.benchmark.heartbeat import install_benchmarker_heartbeat
from metaculus_bot.benchmark.logging_setup import configure_benchmark_logging
from metaculus_bot.config import load_environment
from metaculus_bot.constants import (
    BACKTEST_DEFAULT_RESOLVED_AFTER,
    BACKTEST_DEFAULT_TOURNAMENT,
    BENCHMARK_BATCH_SIZE,
    HEARTBEAT_INTERVAL,
)
from metaculus_bot.scoring_patches import apply_scoring_patches

logger: logging.Logger = logging.getLogger(__name__)

load_environment()


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


_progress_state: dict[str, Any] = {
    "total_predictions": 0,
    "start_time": 0,
    "completed_batches": 0,
    "total_batches": 0,
    "pbar": None,
}


install_benchmarker_heartbeat(HEARTBEAT_INTERVAL, _progress_state)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ground truth backtesting against resolved questions")
    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="Number of resolved questions to backtest (default: 20)",
    )
    parser.add_argument(
        "--resolved-after",
        type=str,
        default=BACKTEST_DEFAULT_RESOLVED_AFTER,
        help=f"Only use questions resolved after this date (default: {BACKTEST_DEFAULT_RESOLVED_AFTER})",
    )
    parser.add_argument(
        "--tournament",
        type=str,
        default=BACKTEST_DEFAULT_TOURNAMENT,
        help=f"Tournament slug to fetch resolved questions from (default: {BACKTEST_DEFAULT_TOURNAMENT})",
    )
    parser.add_argument(
        "--include-models",
        nargs="*",
        default=None,
        help="Only include models matching these substrings (case-insensitive)",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="*",
        default=None,
        help="Exclude models by substring match (case-insensitive)",
    )
    return parser


def _filter_bots(
    bots: list[ForecastBot],
    include_models: list[str] | None,
    exclude_models: list[str] | None,
) -> list[ForecastBot]:
    """Filter bots by include/exclude substring matching on bot name."""
    filtered = list(bots)

    if include_models:
        filtered = [b for b in filtered if any(token.lower() in b.name.lower() for token in include_models)]

    if exclude_models:
        filtered = [b for b in filtered if not any(token.lower() in b.name.lower() for token in exclude_models)]

    if not filtered:
        available_names = [b.name for b in bots]
        raise ValueError(
            f"No bots remaining after model filtering. "
            f"Available: {available_names}, include={include_models}, exclude={exclude_models}"
        )

    logger.info(f"Model filtering: {len(bots)} -> {len(filtered)} bots: {[b.name for b in filtered]}")
    return filtered


async def run_backtest(args: argparse.Namespace) -> None:
    """Run the full backtest pipeline."""
    # 1. Fetch resolved questions and extract ground truths
    logger.info(
        f"Fetching {args.num_questions} resolved questions "
        f"(tournament={args.tournament}, resolved_after={args.resolved_after})"
    )
    sys.stdout.flush()

    question_set: BacktestQuestionSet = await fetch_resolved_questions(
        total_questions=args.num_questions,
        resolved_after=args.resolved_after,
        tournament=args.tournament,
    )

    logger.info(
        f"Prepared {len(question_set.questions)} questions with {len(question_set.ground_truths)} ground truths"
    )

    # 2. Leakage pre-screening
    clean_questions, clean_ground_truths, research_cache = await screen_research_for_leakage(
        question_set.questions,
        question_set.ground_truths,
    )
    question_set.questions = clean_questions
    question_set.ground_truths = clean_ground_truths
    question_set.research_cache = research_cache

    logger.info(f"After leakage screening: {len(clean_questions)} clean questions")

    # 3. Create bots
    bots: list[ForecastBot] = create_individual_bots(
        INDIVIDUAL_MODEL_SPECS,
        DEFAULT_HELPER_LLMS,
        BENCHMARK_BOT_CONFIG,
        batch_size=BENCHMARK_BATCH_SIZE,
        research_cache=research_cache,
    )
    bots = typeguard.check_type(bots, list[ForecastBot])

    bots = _filter_bots(bots, args.include_models, args.exclude_models)

    # 4. Apply scoring patches for mixed question types
    apply_scoring_patches()

    with MonetaryCostManager() as cost_manager:
        # 5. Run Benchmarker
        total_predictions = len(bots) * len(clean_questions)
        logger.info(
            f"Starting backtest: {len(bots)} bots x {len(clean_questions)} questions "
            f"= {total_predictions} total predictions"
        )
        sys.stdout.flush()

        _progress_state.update(
            {
                "total_predictions": total_predictions,
                "start_time": time.time(),
                "completed_batches": 0,
                "total_batches": len(bots),
                "pbar": tqdm(total=total_predictions, desc="Backtesting", unit="predictions"),
            }
        )

        benchmarks = await Benchmarker(
            questions_to_use=clean_questions,
            forecast_bots=bots,
            file_path_to_save_reports="backtests/",
            concurrent_question_batch_size=BENCHMARK_BATCH_SIZE,
        ).run_benchmark()

        if _progress_state["pbar"] is not None:
            _progress_state["pbar"].close()
            _progress_state["pbar"] = None

        logger.info("Benchmarker completed, scoring against ground truth...")
        sys.stdout.flush()

        # 6. Score each bot's reports against ground truth
        results: list[BacktestResult] = []
        for benchmark in benchmarks:
            bot_scores: list[QuestionScore] = []
            num_scored = 0
            num_failed = 0

            for report in benchmark.forecast_reports:
                qid = report.question.id_of_question
                if qid not in clean_ground_truths:
                    num_failed += 1
                    logger.warning(f"No ground truth for question {qid}, skipping")
                    continue

                report_scores = score_report(report, clean_ground_truths[qid])
                if report_scores:
                    bot_scores.extend(report_scores)
                    num_scored += 1
                else:
                    num_failed += 1

            result = BacktestResult(
                bot_name=benchmark.name,
                scores=bot_scores,
                num_questions=len(benchmark.forecast_reports),
                num_scored=num_scored,
                num_failed=num_failed,
            )
            results.append(result)

            aggregated = aggregate_scores(bot_scores)
            logger.info(f"Bot '{benchmark.name}': scored={num_scored}, failed={num_failed}")
            for metric_name, agg in aggregated.items():
                community_str = ""
                if agg.get("community_mean") is not None:
                    community_str = f" | Community: {agg['community_mean']:.4f}"
                logger.info(f"  {metric_name}: Bot mean = {agg['bot_mean']:.4f} (n={agg['n']}){community_str}")

        # 7. Generate report and save data
        report_text = generate_backtest_report(results, question_set, output_path="backtests/backtest_report.md")
        save_backtest_data(question_set, results, output_dir="backtests")

        logger.info(f"\nTotal Cost: {cost_manager.current_usage}")
        logger.info("\n" + "=" * 60)
        logger.info("BACKTEST REPORT")
        logger.info("=" * 60)
        logger.info(report_text)


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"

    configure_benchmark_logging(log_dir="backtests")

    parser = _build_parser()
    args = parser.parse_args()

    asyncio.run(run_backtest(args))
