"""CLI entry point for performance analysis.

Usage:
    python -m metaculus_bot.performance_analysis [--tournament SLUG] [--output PATH] [--cached PATH]
"""

import argparse
import logging
import sys

from metaculus_bot.performance_analysis.analysis import generate_report
from metaculus_bot.performance_analysis.collector import build_performance_dataset, load_dataset, save_dataset

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "scratch/performance_data.json"
DEFAULT_TOURNAMENT = "spring-aib-2026"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Metaculus bot performance analysis")
    parser.add_argument("--tournament", default=DEFAULT_TOURNAMENT, help="Tournament slug (default: %(default)s)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSON path (default: %(default)s)")
    parser.add_argument("--cached", default=None, help="Load from cached JSON instead of fetching from API")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)

    if args.cached:
        logger.info(f"Loading cached dataset from {args.cached}")
        data = load_dataset(args.cached)
    else:
        data = build_performance_dataset(tournament=args.tournament)
        save_dataset(data, args.output)

    report = generate_report(data)
    print(report)


if __name__ == "__main__":
    main()
