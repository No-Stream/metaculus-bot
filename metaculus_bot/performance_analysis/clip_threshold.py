"""Counterfactual clip-threshold sweep for binary and MC publishes (READ-ONLY, free, offline).

Reprices every resolved binary and MC publish under a grid of candidate clip floors and
reports each in spot-peer points. Nothing here touches a live pipeline constant and nothing
spends: the pass reads a cached performance dataset and computes. What every section of the
output means, why a looser clip is censored rather than measured, the window vocabulary and
the standing result: docs/performance_analysis.md "The clip-threshold sweep".

This module is the CLI. The model and math live in ``clip_threshold_sweep``, the windows in
``clip_threshold_windows``, the selection-aware readings in ``clip_threshold_selection``, the
derived tables in ``clip_threshold_tables`` and the markdown in ``clip_threshold_report``; a
script or test imports each name from the module that defines it. Run it as
``uv run python -m metaculus_bot.performance_analysis.clip_threshold --cached <path>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime

from metaculus_bot.performance_analysis.clip_threshold_report import render_report
from metaculus_bot.performance_analysis.clip_threshold_tables import compute_report
from metaculus_bot.performance_analysis.clip_threshold_windows import LOOKBACK_DAYS
from metaculus_bot.performance_analysis.cohorts import (
    EXCLUSION_COHORTS,
    KNOWN_BUG_SHORTHAND,
    parse_exclude_qids,
)
from metaculus_bot.performance_analysis.collector import load_dataset
from metaculus_bot.time_utils import parse_iso_utc

logger: logging.Logger = logging.getLogger(__name__)


def _parse_as_of(raw: str | None) -> datetime:
    """The instant ``last_90d`` counts back from; the UTC clock when the flag is absent."""
    if raw is None:
        return datetime.now(UTC)
    parsed = parse_iso_utc(raw)
    if parsed is None:
        raise ValueError(f"--as-of {raw!r} is not an ISO-8601 timestamp")
    return parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Counterfactual clip-threshold sweep over resolved binary/MC publishes (read-only, offline)"
    )
    parser.add_argument(
        "--cached",
        required=True,
        help=(
            "Path to a cached performance dataset JSON (list of records), normally the current "
            "round's perf_all_tagged.json. Required: a default naming one round keeps resolving "
            "after that round is superseded, so the sweep would silently measure a stale dataset."
        ),
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help=(
            f"ISO-8601 instant the last_{LOOKBACK_DAYS}d window is measured back from, echoed in "
            "the header. Default: the UTC clock at run time."
        ),
    )
    parser.add_argument("--output-json", default=None, help="Optional path to also write every number as JSON.")
    parser.add_argument(
        "--exclude-qids",
        default="",
        help=(
            "Comma-separated question ids to drop before the sweep (the count is rendered in the "
            "header so the exclusion is visible). Each cohort shorthand below composes with "
            "explicit ids: "
            + "; ".join(f"'{name}' = {','.join(sorted(ids))}" for name, ids in sorted(EXCLUSION_COHORTS.items()))
            + f". So '{KNOWN_BUG_SHORTHAND},43800' excludes that cohort AND 43800. Default: exclude nothing."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)

    as_of = _parse_as_of(args.as_of)
    exclude_qids = parse_exclude_qids(args.exclude_qids)
    data = load_dataset(args.cached)
    report = compute_report(data, dataset_path=args.cached, as_of=as_of, exclude_qids=exclude_qids)

    # Logging is pinned to stderr above so this report can be piped on its own.
    print(render_report(report))  # noqa: T201

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Wrote the clip sweep to {args.output_json}")


if __name__ == "__main__":
    main()
