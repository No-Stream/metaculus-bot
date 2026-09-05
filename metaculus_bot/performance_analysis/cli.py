"""CLI entry point for performance analysis.

Usage:
    python -m metaculus_bot.performance_analysis [--tournament SLUG] [--output PATH] [--cached PATH]
                                                 [--prior PATH]
"""

import argparse
import logging
import sys

from metaculus_bot.api_preflight import verify_metaculus_api_identity
from metaculus_bot.performance_analysis.analysis import generate_report
from metaculus_bot.performance_analysis.collector import build_performance_dataset, load_dataset, save_dataset
from metaculus_bot.performance_analysis.rescore_diff import diff_platform_rescores, render_rescore_summary

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "scratch/performance_data.json"
DEFAULT_TOURNAMENT = "spring-aib-2026"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Metaculus bot performance analysis")
    parser.add_argument("--tournament", default=DEFAULT_TOURNAMENT, help="Tournament slug (default: %(default)s)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output JSON path (default: %(default)s)")
    parser.add_argument("--cached", default=None, help="Load from cached JSON instead of fetching from API")
    parser.add_argument(
        "--prior",
        default=None,
        help=(
            "A previous round's dataset JSON. Every question present in both is diffed on its "
            "resolution and its Metaculus scores, and anything Metaculus changed in place is "
            "tagged platform_rescored and logged as a PLATFORM_RESCORED warning."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)

    prior = load_dataset(args.prior) if args.prior else None

    if args.cached:
        logger.info(f"Loading cached dataset from {args.cached}")
        data = load_dataset(args.cached)
        # The cached path skips build_performance_dataset entirely, so the diff runs here.
        # Diffing a cached pull against an older one is a legitimate offline check — it is
        # how a re-resolution gets caught after the fact, without a second API pull.
        if prior is not None:
            diff_platform_rescores(prior, data)
    else:
        # The live pull sends METACULUS_TOKEN to the API; confirm the host is
        # the real Metaculus first (DNS-parking incident — see
        # metaculus_bot/api_preflight.py). Skipped for --cached (disk read).
        verify_metaculus_api_identity()
        data = build_performance_dataset(tournament=args.tournament, prior_records=prior)
        save_dataset(data, args.output)

    if prior is not None:
        print("\n".join(render_rescore_summary(data)))

    report = generate_report(data)
    print(report)


if __name__ == "__main__":
    main()
