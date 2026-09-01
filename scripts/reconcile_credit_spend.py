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

``--roles`` adds the per-role ledger (``credit_role_spend.jsonl``, the
``CREDIT_ROLE_SPEND`` marker, since the 2026-09 bundle): each run's role-ledger
total for the same key beside its settled spend — the two measure the same money
from opposite ends (OpenRouter's per-call usage accounting vs. the key's booked
balance), so their ratio is the ledger's own coverage check — plus a per-(role, key)
table over the selected runs, which is the decomposition every cost argument used
to lack. Both printed coverage ratios cover the SETTLED runs only, since the trailing
run's settled spend is unknown and every figure states the run set it sums.

Usage
-----
    uv run python scripts/reconcile_credit_spend.py
    uv run python scripts/reconcile_credit_spend.py --key donated --since 2026-07-20
    uv run python scripts/reconcile_credit_spend.py --roles --since 2026-09-20
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
ROLE_ARCHIVE_FILENAME: str = "credit_role_spend.jsonl"


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


@dataclass(frozen=True)
class SettledCohort:
    """The runs whose spend OpenRouter has booked, and their total.

    Carried around rather than passed as a bare float because it is the denominator of every
    coverage percentage the script prints, and each numerator has to be summed over the SAME
    runs. The trailing run has no successor, so its settled spend is unknown; counting its
    marker or its role rows against a total that cannot include it reads over 100%.
    """

    run_ids: frozenset[str]
    usd: float

    @property
    def label(self) -> str:
        """How a printed ratio names its own run set, so no total is labelled just ``total``."""
        return f"the {len(self.run_ids)} runs with a successor"


@dataclass(frozen=True)
class RunRoleSpend:
    """One run's role-ledger total on one key. ``usd`` is None when no row carried cost."""

    run_id: str
    rows: int
    costed_rows: int
    usd: float | None


@dataclass(frozen=True)
class RoleTotal:
    """One (role, key) line summed over the selected runs."""

    role: str
    key: str
    calls: int
    costed_calls: int
    usd: float | None


def role_spend_by_run(role_records: list[dict], key: str) -> dict[str, RunRoleSpend]:
    """Sum each run's ``credit_role_spend`` rows for ``key``.

    Comparable with :func:`reconcile`'s ``settled_usd`` for the same key: both are that key's
    spend on that run, one read per call from OpenRouter's usage accounting, the other from
    the key's booked balance. A row whose ``usd`` is None (no cost data) is counted but not
    summed, and a run with only such rows reports ``usd=None`` rather than a false zero.
    """
    by_run: dict[str, list[dict]] = defaultdict(list)
    for record in role_records:
        if record.get("key") == key:
            by_run[record["run_id"]].append(record)
    out: dict[str, RunRoleSpend] = {}
    for run_id, rows in by_run.items():
        costed = [row["usd"] for row in rows if row.get("usd") is not None]
        out[run_id] = RunRoleSpend(
            run_id=run_id, rows=len(rows), costed_rows=len(costed), usd=sum(costed) if costed else None
        )
    return out


def aggregate_roles(role_records: list[dict], run_ids: set[str] | None = None) -> list[RoleTotal]:
    """Per-(role, key) totals over ``run_ids`` (all runs when None), biggest spender first,
    rows with no cost data last."""
    calls: dict[tuple[str, str], int] = defaultdict(int)
    costed_calls: dict[tuple[str, str], int] = defaultdict(int)
    usd: dict[tuple[str, str], float | None] = {}
    for record in role_records:
        if run_ids is not None and record["run_id"] not in run_ids:
            continue
        line = (record["role"], record["key"])
        calls[line] += record["calls"]
        costed_calls[line] += record["costed_calls"]
        row_usd = record.get("usd")
        if row_usd is not None:
            usd[line] = (usd.get(line) or 0.0) + row_usd
        else:
            usd.setdefault(line, None)
    totals = [
        RoleTotal(
            role=role, key=key, calls=calls[(role, key)], costed_calls=costed_calls[(role, key)], usd=usd[(role, key)]
        )
        for role, key in calls
    ]
    return sorted(totals, key=lambda total: (total.usd is None, -(total.usd or 0.0), total.role, total.key))


def _fmt(value: float | None) -> str:
    return "   n/a" if value is None else f"{value:6.2f}"


def _fmt4(value: float | None) -> str:
    return "     n/a" if value is None else f"{value:8.4f}"


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _print_role_table(role_records: list[dict], run_ids: set[str]) -> None:
    """The per-(role, key) decomposition of the selected runs.

    The table spans both keys — a cost argument wants the whole ledger, not one key's half —
    so its grand total and the ``share`` column read against it say so. The key-filtered total
    printed just above it is a different number over a different set.
    """
    totals = aggregate_roles(role_records, run_ids)
    if not totals:
        print("\nno credit_role_spend rows for the selected runs")
        return
    grand = sum(total.usd or 0.0 for total in totals)
    print(f"\n{'role':24} {'key':9} {'calls':>6} {'costed':>6} {'usd':>8} {'share':>6}")
    for total in totals:
        share = "   n/a" if total.usd is None or grand <= 0 else f"{total.usd / grand:6.0%}"
        print(f"{total.role:24} {total.key:9} {total.calls:6d} {total.costed_calls:6d} {_fmt4(total.usd)} {share}")
    print(f"{'total (all keys, costed)':24} {'':9} {'':6} {'':6} {_fmt4(grand)}")


def _load_role_ledger(balance_archive: Path) -> list[dict]:
    """The ``credit_role_spend`` records that live beside the balance archive, or [] with a
    warning while the file does not exist yet."""
    role_archive = balance_archive.with_name(ROLE_ARCHIVE_FILENAME)
    if not role_archive.exists():
        logger.warning(
            "%s not found beside %s — the CREDIT_ROLE_SPEND marker ships with the 2026-09 bundle, so it "
            "appears after the first prod run on that code plus a `make sync_telemetry`",
            ROLE_ARCHIVE_FILENAME,
            balance_archive,
        )
        return []
    return _load_jsonl(role_archive)


def _print_run_table(rows: list[RunSpend], role_totals: dict[str, RunRoleSpend] | None) -> None:
    """One line per run; the ``roles`` column (role-ledger total on the same key) only with --roles."""
    roles_header = "" if role_totals is None else f" {'roles':>8}"
    print(f"{'run_date':17} {'run_id':12} {'workflow':16} {'marker':>6} {'settled':>7} {'lagged':>7}{roles_header}")
    for row in rows:
        note = "  <- unsettled (no successor yet)" if row.is_final else ""
        roles_cell = ""
        if role_totals is not None:
            run_roles = role_totals.get(row.run_id)
            roles_cell = f" {_fmt4(None if run_roles is None else run_roles.usd)}"
        print(
            f"{row.run_date[:16]:17} {row.run_id:12} {row.workflow:16} "
            f"{_fmt(row.within_run_usd)} {_fmt(row.settled_usd)} {_fmt(row.lagged_usd)}{roles_cell}{note}"
        )


def _print_key_totals(rows: list[RunSpend]) -> SettledCohort:
    """Marker-vs-settled totals for the key; returns the settled cohort for the role summary."""
    settled_rows = [row for row in rows if row.settled_usd is not None]
    cohort = SettledCohort(
        run_ids=frozenset(row.run_id for row in settled_rows),
        usd=sum(row.settled_usd for row in settled_rows if row.settled_usd is not None),
    )
    marker_total = sum(row.within_run_usd or 0.0 for row in rows)
    print(f"\nmarker-reported total: ${marker_total:.2f}  (all {len(rows)} runs)")
    print(f"settled total:         ${cohort.usd:.2f}  ({cohort.label})")
    if cohort.usd > 0:
        # Numerator restricted to the settled cohort, because the denominator is: summing the
        # markers of runs whose settled spend is unknown against a total that cannot include
        # them overstates capture, and reads over 100% whenever the trailing run spent much.
        marker_on_settled = sum(row.within_run_usd or 0.0 for row in settled_rows)
        print(f"marker captured:       {marker_on_settled / cohort.usd:.0%} of settled spend ({cohort.label})")
    zeros = sum(1 for row in rows if row.within_run_usd == 0.0)
    print(f"runs whose marker read exactly 0.00: {zeros}/{len(rows)}")
    return cohort


def _print_role_summary(
    role_records: list[dict],
    role_totals: dict[str, RunRoleSpend],
    *,
    rows: list[RunSpend],
    key: str,
    settled: SettledCohort,
) -> None:
    """The role ledger read against the settled delta, then decomposed per (role, key).

    Three figures over three different sets sit next to each other here, so each names its
    own: this key's ledger total over every selected run, the coverage ratio over the settled
    cohort alone (the ratio's denominator excludes the trailing run, so its numerator must
    too), and the table's grand total, which spans BOTH keys because its rows do.
    """
    selected_run_ids = {row.run_id for row in rows}
    roles_total = sum(run.usd or 0.0 for run_id, run in role_totals.items() if run_id in selected_run_ids)
    print(f"role-ledger total ({key}, all {len(rows)} runs): ${roles_total:.4f}")
    if settled.usd > 0:
        roles_on_settled = sum(run.usd or 0.0 for run_id, run in role_totals.items() if run_id in settled.run_ids)
        print(f"role ledger covers:    {roles_on_settled / settled.usd:.0%} of settled spend ({settled.label})")
    _print_role_table(role_records, selected_run_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH, help="credit_balance.jsonl")
    parser.add_argument("--key", default="personal", choices=("personal", "donated"), help="which key")
    parser.add_argument("--since", default="", help="only runs whose run_date starts at/after this (YYYY-MM-DD)")
    parser.add_argument(
        "--roles",
        action="store_true",
        help=f"also read {ROLE_ARCHIVE_FILENAME} beside --archive: per-run role-ledger totals and a per-role table",
    )
    args = parser.parse_args()

    if not args.archive.exists():
        raise SystemExit(f"archive not found at {args.archive}; run `make sync_all` (free) to populate it")

    rows = [row for row in reconcile(_load_jsonl(args.archive), args.key) if row.run_date >= args.since]
    if not rows:
        raise SystemExit(f"no paired {args.key}-key runs in {args.archive}")

    role_records = _load_role_ledger(args.archive) if args.roles else []
    role_totals = role_spend_by_run(role_records, args.key) if role_records else None

    _print_run_table(rows, role_totals)
    settled = _print_key_totals(rows)
    if role_totals is not None:
        _print_role_summary(role_records, role_totals, rows=rows, key=args.key, settled=settled)


if __name__ == "__main__":
    main()
