"""Reconcile per-run OpenRouter spend against the telemetry archive.

Why this exists
---------------
``CREDIT_SPEND: key=personal`` is a LOWER BOUND, and often reads ``0.00`` on a run
that spent real money. The personal key reports no ``limit_remaining``, so the
per-run delta falls back to the lifetime ``usage`` field — and OpenRouter has
typically not settled a run's spend by the time the end snapshot fires, seconds
after the last call.

Measured over ``backtests/telemetry_archive/credit_balance.jsonl`` (178 paired
personal-key runs, 2026-07-20 to 2026-07-27): within-run deltas summed to $3.31
against $5.66 of true lifetime-usage growth, so the marker captured 58%. The
missing $2.35 is fully recovered by the gap between each run's ``phase=end`` usage
and the NEXT run's ``phase=start`` usage — $3.31 + $2.35 = $5.66 exactly. The money
is not lost, just late.

So the honest per-run figure is not obtainable at end-of-run. It IS obtainable
afterwards, which is what this script computes: for each run, the settled spend is
the next observation's usage minus this run's start usage. The final run in the
archive has no successor, so its spend is reported as still-unsettled rather than
guessed at.

Free and offline: reads only the local archive. Run ``make sync_all`` first if you
want the archive current (that pull is also free).

Usage
-----
    uv run python scripts/reconcile_credit_spend.py
    uv run python scripts/reconcile_credit_spend.py --key donated --since 2026-07-20
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stderr)
logger: logging.Logger = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
ARCHIVE_PATH: Path = REPO_ROOT / "backtests" / "telemetry_archive" / "credit_balance.jsonl"


@dataclass(frozen=True)
class RunSpend:
    """One run's spend, reconciled where possible."""

    run_id: str
    run_date: str
    workflow: str
    within_run_usd: float | None
    settled_usd: float | None
    is_final: bool

    @property
    def lagged_usd(self) -> float | None:
        """Spend that landed AFTER the end snapshot — the part the marker missed."""
        if self.settled_usd is None or self.within_run_usd is None:
            return None
        return self.settled_usd - self.within_run_usd


def _paired_snapshots(records: list[dict], key: str) -> list[tuple[str, dict, dict]]:
    """Return ``(run_id, start, end)`` for runs with BOTH snapshots, oldest first."""
    by_run: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in records:
        if record.get("key") != key:
            continue
        phase = record.get("phase")
        if phase in ("start", "end"):
            by_run[record["run_id"]][phase] = record
    paired = [
        (run_id, phases["start"], phases["end"])
        for run_id, phases in by_run.items()
        if "start" in phases and "end" in phases
    ]
    # Order by the START snapshot's own timestamp: run_date is the workflow's
    # dispatch time, which can tie across concurrently-dispatched runs.
    return sorted(paired, key=lambda triple: (triple[1].get("line_ts") or "", triple[0]))


def reconcile(records: list[dict], key: str) -> list[RunSpend]:
    """Per-run spend for ``key``, settled against each run's successor.

    The successor's ``phase=start`` usage is the first observation taken after the
    settlement window had time to close, so ``next_start - this_start`` is the
    run's spend as OpenRouter eventually booked it.
    """
    paired = _paired_snapshots(records, key)
    out: list[RunSpend] = []
    for index, (run_id, start, end) in enumerate(paired):
        start_usage, end_usage = start.get("usage"), end.get("usage")
        within = None if start_usage is None or end_usage is None else end_usage - start_usage

        settled: float | None = None
        is_final = index == len(paired) - 1
        if not is_final and start_usage is not None:
            next_start_usage = paired[index + 1][1].get("usage")
            if next_start_usage is not None:
                settled = next_start_usage - start_usage

        out.append(
            RunSpend(
                run_id=run_id,
                run_date=str(start.get("run_date") or ""),
                workflow=str(start.get("workflow") or "?"),
                within_run_usd=within,
                settled_usd=settled,
                is_final=is_final,
            )
        )
    return out


def _fmt(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:6.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH, help="credit_balance.jsonl")
    parser.add_argument("--key", default="personal", choices=("personal", "donated"), help="which key")
    parser.add_argument("--since", default="", help="only runs whose run_date starts at/after this (YYYY-MM-DD)")
    args = parser.parse_args()

    if not args.archive.exists():
        raise SystemExit(f"archive not found at {args.archive}; run `make sync_all` (free) to populate it")

    with args.archive.open() as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    rows = [row for row in reconcile(records, args.key) if row.run_date >= args.since]
    if not rows:
        raise SystemExit(f"no paired {args.key}-key runs in {args.archive}")

    print(f"{'run_date':17} {'run_id':12} {'workflow':16} {'marker':>6} {'settled':>7} {'lagged':>7}")
    for row in rows:
        note = "  <- unsettled (no successor yet)" if row.is_final else ""
        print(
            f"{row.run_date[:16]:17} {row.run_id:12} {row.workflow:16} "
            f"{_fmt(row.within_run_usd)} {_fmt(row.settled_usd)} {_fmt(row.lagged_usd)}{note}"
        )

    marker_total = sum(row.within_run_usd or 0.0 for row in rows)
    settled_total = sum(row.settled_usd or 0.0 for row in rows if row.settled_usd is not None)
    print(f"\nmarker-reported total: ${marker_total:.2f}")
    print(f"settled total:         ${settled_total:.2f}")
    if settled_total > 0:
        print(f"marker captured:       {marker_total / settled_total:.0%} of settled spend")
    zeros = sum(1 for row in rows if row.within_run_usd == 0.0)
    print(f"runs whose marker read exactly 0.00: {zeros}/{len(rows)}")


if __name__ == "__main__":
    main()
