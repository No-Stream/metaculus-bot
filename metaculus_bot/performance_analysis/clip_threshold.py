"""Counterfactual clip-threshold sweep for binary and MC publishes (READ-ONLY, free, offline).

**What this answers.** The bot clamps every published probability: binary per-model forecasts
into ``[BINARY_PROB_MIN, BINARY_PROB_MAX]`` and MC option vectors into
``[MC_PROB_MIN, MC_PROB_MAX]`` (then renormalised). Both floors are judgment calls nobody has
priced. This pass reprices every resolved binary and MC question the bot has forecast under a
grid of candidate floors ``c`` and reports what each would have been worth in SPOT PEER
points, the quantity the tournament ranks on. Moving only OUR mass on the resolving outcome
changes spot peer by ``100 * ln(new / old)`` for these two un-halved types
(:func:`metaculus_bot.scoring_common.spot_peer_delta`; never hand-roll that conversion, the
continuous-question halving is the easiest thing here to apply twice).

**How to read the output, in the order it prints.**

0. *The header's live-regime line.* Whether the clamp now in force has bound ANY publish since
   it went live. If it has not, the loosening question is unmeasurable for the live config and
   every tightening row is priced on records the live clamp never touched.
1. *Extreme-bin calibration.* Of the publishes that priced an outcome at 1-2%, how often did
   that outcome happen, beside how often the bot's OWN prices said it would? This is the
   direct evidence, and every sweep number below is a re-expression of it.
2. *The in-window argmax, the out-of-bag check, and the nesting table.* ``sum_delta`` is the
   whole window's spot-peer gain from moving to floor ``c``; ``mean_delta`` divides by the
   window's n, so it reads as points per question forecast. The argmax is a CHOICE over the
   grid, so the out-of-bag table refits it on resamples and scores it on the records left
   out, which is what the choice is actually worth. The nesting table shows that the
   ``all`` / ``last_N`` / dated windows re-count one set of records; the ``era_*`` windows
   are the disjoint slices.
3. *The floor-only sweep per window, each with an insurance view.* A floor is insurance
   against the sub-``c`` band being under-priced. Spot peer is proper, so a clip costs a
   calibrated forecaster ``E[sum] own p`` regardless, and pays only if the moved records
   resolve on the clipped side more often than the ``break-even r*`` rate; the Jeffreys
   interval on the observed rate says whether that is even open.
4. *Ceiling-only and symmetric.* The high side is nearly empty in the archive, so these are
   compact; they exist so nobody has to infer them from the floor table.
5. *Loosening bounds.* A record published exactly AT its in-force floor destroyed the raw
   member value, so a LOOSER candidate is unobservable on it. Those rows carry bounds, not
   estimates, and the member-level bound (``cen_m``, ``at_c (members)``) reaches the records
   where a clamped member sat in a median position under a publish that was itself above
   the floor. The section also prices the live floor on the records published under the
   older clamp: the one floor comparison the archive measures rather than bounds.
6. *Out-of-sample carry.* A clip level is a fitted calibration layer, and this repo's
   standing rule (AGENTS.md era-bucketing) is that a fitted layer ships only after an
   out-of-sample era test. Each suffix window's ``c_star`` is fitted on the records OLDER
   than the window and then evaluated inside it. A fit that moves nothing in its own
   complement is flagged, because its carry is vacuous rather than a pass.
7. *Per-model replay cross-check.* The honesty check on the shortcut in (3): clamping the
   PUBLISHED median is exact only because clamping is monotone and the median of an ODD
   number of members is an order statistic. With an even member count the published value
   averages the two middle members, so the two paths can differ; the cross-check replays
   members through the clamp under the aggregator each record was actually published with,
   reports the gap, states the sign of the error, and counts the records whose stacking
   route the counterfactual would in fact have changed.
8. *Single-survivor publishes.* The live single-forecaster publish floor fires on exactly one
   cohort; this prices it on those records and no others.

**Censoring, the one thing that cannot be measured.** A candidate BELOW a record's in-force
floor cannot be priced when the published value sits at that floor: the raw member value was
erased by the clamp live at the time and could have been anywhere below. ``sum_delta_lower``
assumes every censored raw value was exactly the floor (nothing moves); ``sum_delta_upper``
assumes every one was at or below ``c`` (all move the whole way). Those two are not a bracket
when the signs disagree, so ``bracket_lo``/``bracket_hi`` carry the identified set with each
record contributing its own best and worst case. A record published strictly ABOVE its floor
with no clamped member in a median position is unaffected by a looser clip and contributes
exactly 0 everywhere.

The in-force clamp is looked up per record from ``bot_comment_created_at`` against the merge
timestamps in ``analysis.py`` (``WIDENING_FLIP_MERGED_AT`` for binary, ``FT_0292_MERGED_AT``
for MC): merge-to-main committer dates, never authoring dates, because prod runs from main.

Nothing here touches a live pipeline constant and nothing here spends: the pass reads a
cached performance dataset and computes. The model and math live in
``clip_threshold_sweep``, the windows in ``clip_threshold_windows``, the selection-aware
readings in ``clip_threshold_selection``, the derived tables in ``clip_threshold_tables`` and
the markdown in ``clip_threshold_report``; this module is the CLI, and a script or test
imports each name from the module that defines it. Run it as
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
        default="scratch/residual_2026-09-01/perf_all_tagged.json",
        help="Path to a cached performance dataset JSON (list of records). Default: %(default)s",
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
