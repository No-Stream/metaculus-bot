"""Weekly close-margin watch view over the CLOSE_MARGIN telemetry archive.

Reads the ``close_margin.jsonl`` records harvested by ``make sync_telemetry`` and
prints, per ISO submission week, the p50 / p10 / min of ``margin_frac`` (fraction of
each question's open→close window still remaining when we submitted) plus a count of
questions under the red line. The red line — default 30% of window remaining — is the
2026-07-18 latency/completeness audit's recommended watch threshold: a shrinking
margin is the early symptom of GitHub Actions cron starvation pushing submissions
toward the close deadline, and a missed close forfeits the whole spot score.

Read-only + free (reads a local JSONL, no network, no LLM calls). This is a table
printer, not a dashboard: the aggregation logic lives in small pure functions
(``week_key`` / ``percentile`` / ``summarize_weeks``) so it is unit-testable, and
``main`` only renders.

Usage:
    uv run python scripts/close_margin_watch.py [--archive-dir DIR] [--red-line 0.30]
    make close_margin_watch                       # same, default archive dir
    make close_margin_watch ARGS="--red-line 0.5" # tighter red line
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from scripts.telemetry.archive import load_marker_records

logger = logging.getLogger(__name__)

DEFAULT_ARCHIVE_DIR = "backtests/telemetry_archive"
DEFAULT_RED_LINE = 0.30


def week_key(iso_ts: object) -> str | None:
    """Map a ``submitted_at`` ISO timestamp to its ISO-week label (``2026-W29``), or None."""
    if not isinstance(iso_ts, str):
        return None
    try:
        moment = datetime.fromisoformat(iso_ts)
    except ValueError:
        return None
    cal = moment.isocalendar()
    return f"{cal.year}-W{cal.week:02d}"


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy default method) for ``q`` in [0, 1].

    Sorts a copy, so callers needn't pre-sort. Raises on an empty sequence — the
    caller guarantees a non-empty bucket.
    """
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = q * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (rank - low) * (ordered[high] - ordered[low])


@dataclass(frozen=True)
class WeekSummary:
    """One ISO week's close-margin distribution + red-line breach count."""

    week: str
    n: int
    p50: float
    p10: float
    minimum: float
    n_below_red: int


def summarize_weeks(records: list[dict], red_line: float) -> tuple[list[WeekSummary], list[dict], int]:
    """Bucket CLOSE_MARGIN records by submission week and compute per-week margin stats.

    Returns ``(week_summaries, below_red_records, skipped)``. Records with a null
    ``margin_frac`` (open_time was missing, so the window fraction is unknown) or an
    unparseable ``submitted_at`` are skipped and counted — they carry no frac to
    percentile or compare against the red line.
    """
    by_week: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    skipped = 0
    for rec in records:
        frac = rec.get("margin_frac")
        week = week_key(rec.get("submitted_at"))
        if not isinstance(frac, (int, float)) or week is None:
            skipped += 1
            continue
        by_week[week].append((float(frac), rec))

    summaries: list[WeekSummary] = []
    below_red: list[dict] = []
    for week in sorted(by_week):
        fracs = [
            f for f, _ in by_week[week]
        ]  # HARNESS-SCAN-EXEMPT-object-explosion  # small per-week bucket, not a DataFrame
        breaches = [
            rec for f, rec in by_week[week] if f < red_line
        ]  # HARNESS-SCAN-EXEMPT-object-explosion  # small per-week bucket, not a DataFrame
        below_red.extend(breaches)
        summaries.append(
            WeekSummary(
                week=week,
                n=len(fracs),
                p50=percentile(fracs, 0.50),
                p10=percentile(fracs, 0.10),
                minimum=min(fracs),
                n_below_red=len(breaches),
            )
        )
    return summaries, below_red, skipped


def _render(summaries: list[WeekSummary], below_red: list[dict], skipped: int, red_line: float) -> None:
    print(f"CLOSE_MARGIN weekly watch (red line: margin_frac < {red_line:.2f} of window remaining)")
    print(f"{'week':10s} {'n':>4s} {'p50':>8s} {'p10':>8s} {'min':>8s} {'below_red':>10s}")
    for s in summaries:
        alarm = "  <-- p10 below red line" if s.p10 < red_line else ""
        print(f"{s.week:10s} {s.n:>4d} {s.p50:>8.3f} {s.p10:>8.3f} {s.minimum:>8.3f} {s.n_below_red:>10d}{alarm}")

    if below_red:
        print(f"\nQuestions under the red line ({len(below_red)}), worst first:")
        print(f"{'week':10s} {'qid':>10s} {'margin_frac':>12s} {'margin_h':>10s}")
        for rec in sorted(below_red, key=lambda r: r.get("margin_frac", 0.0)):
            margin_s = rec.get("margin_s")
            margin_h = f"{margin_s / 3600:.1f}" if isinstance(margin_s, (int, float)) else "n/a"
            print(
                f"{week_key(rec.get('submitted_at')) or 'n/a':10s} "
                f"{rec.get('qid')!s:>10s} {rec.get('margin_frac'):>12.4f} {margin_h:>10s}"
            )
    else:
        print("\nNo questions under the red line.")

    if skipped:
        print(f"\nNote: {skipped} record(s) skipped (missing window fraction or unparseable submit time).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly close-margin watch over the CLOSE_MARGIN telemetry archive.")
    parser.add_argument("--archive-dir", default=DEFAULT_ARCHIVE_DIR, help="Telemetry archive dir")
    parser.add_argument(
        "--red-line",
        type=float,
        default=DEFAULT_RED_LINE,
        help="Flag questions/weeks whose margin_frac falls below this fraction of the window (default 0.30).",
    )
    parser.add_argument("--output", default=None, help="Optional path to dump the weekly summary as JSON.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    records = load_marker_records(Path(args.archive_dir), "close_margin")
    if not records:
        print(f"No close_margin records in {args.archive_dir} — run `make sync_telemetry` first (or none logged yet).")
        return

    summaries, below_red, skipped = summarize_weeks(records, args.red_line)
    _render(summaries, below_red, skipped, args.red_line)

    if args.output:
        Path(args.output).write_text(
            json.dumps({"red_line": args.red_line, "weeks": [asdict(s) for s in summaries]}, indent=2)
        )
        print(f"\nWrote weekly summary to {args.output}")


if __name__ == "__main__":
    sys.exit(main())
