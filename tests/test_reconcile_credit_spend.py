"""Tests for scripts/reconcile_credit_spend.py.

The script exists because `CREDIT_SPEND: key=personal` is a lower bound: the
personal key reports no `limit_remaining`, so its per-run delta comes from the
lifetime `usage` field, and OpenRouter has typically not settled the run's spend by
the time the end snapshot fires. The settled figure is recoverable only afterwards,
by differencing a run's start usage against its SUCCESSOR's start usage.

These tests pin that arithmetic on synthetic records shaped exactly like
`backtests/telemetry_archive/credit_balance.jsonl`.
"""

from __future__ import annotations

import re

import pytest

from scripts import reconcile_credit_spend as rcs
from scripts.reconcile_credit_spend import RunSpend, aggregate_roles, reconcile, role_spend_by_run


def _snapshot(run_id: str, phase: str, usage: float | None, *, ts: str, key: str = "personal") -> dict:
    """One credit_balance archive record."""
    return {
        "marker": "credit_balance",
        "run_id": run_id,
        "workflow": "tournament",
        "run_date": f"{ts}Z",
        "line_ts": ts,
        "key": key,
        "phase": phase,
        "remaining": None,  # personal key is uncapped — the whole reason for the lag
        "usage": usage,
    }


def _run(run_id: str, start_usage: float, end_usage: float, *, hour: int) -> list[dict]:
    return [
        _snapshot(run_id, "start", start_usage, ts=f"2026-07-25T{hour:02d}:00:00"),
        _snapshot(run_id, "end", end_usage, ts=f"2026-07-25T{hour:02d}:10:00"),
    ]


class TestSettledSpendRecovery:
    def test_lagged_spend_is_recovered_from_the_next_run(self) -> None:
        # Run A reports 0.00 within-run (usage flat across it) but the next run's
        # start shows usage 0.30 higher — that 0.30 is A's real spend, booked late.
        records = _run("A", 100.00, 100.00, hour=1) + _run("B", 100.30, 100.30, hour=2)
        rows = reconcile(records, "personal")

        assert [row.run_id for row in rows] == ["A", "B"]
        run_a = rows[0]
        assert run_a.within_run_usd == pytest.approx(0.0)
        assert run_a.settled_usd == pytest.approx(0.30)
        assert run_a.lagged_usd == pytest.approx(0.30)

    def test_partially_settled_run_keeps_both_halves(self) -> None:
        # Some spend lands inside the run, the rest after. Both must be accounted.
        records = _run("A", 100.00, 100.10, hour=1) + _run("B", 100.35, 100.35, hour=2)
        rows = reconcile(records, "personal")
        assert rows[0].within_run_usd == pytest.approx(0.10)
        assert rows[0].settled_usd == pytest.approx(0.35)
        assert rows[0].lagged_usd == pytest.approx(0.25)

    def test_final_run_is_reported_unsettled_not_guessed(self) -> None:
        # No successor means no observation after the settlement window. Reporting a
        # number here would be fabricating one, so settled stays None.
        records = _run("A", 100.00, 100.00, hour=1) + _run("B", 100.30, 100.30, hour=2)
        rows = reconcile(records, "personal")
        final = rows[-1]
        assert final.is_final is True
        assert final.settled_usd is None
        assert final.lagged_usd is None

    def test_settled_totals_close_the_gap_the_marker_left(self) -> None:
        """The archive-level invariant: within-run + lagged == total growth.

        This is the property that proved the diagnosis (settlement lag, not BYOK):
        on the real archive $3.31 + $2.35 == $5.66 exactly. Pinned here in miniature
        so a change to the differencing can't silently break the identity.
        """
        records = (
            _run("A", 100.00, 100.05, hour=1) + _run("B", 100.20, 100.25, hour=2) + _run("C", 100.60, 100.60, hour=3)
        )
        rows = reconcile(records, "personal")
        settled = [row for row in rows if row.settled_usd is not None]
        assert sum(row.settled_usd or 0.0 for row in settled) == pytest.approx(0.60)
        # Every settled row's own halves add up.
        for row in settled:
            assert row.settled_usd == pytest.approx((row.within_run_usd or 0.0) + (row.lagged_usd or 0.0))

    def test_runs_are_ordered_by_snapshot_time_not_dict_order(self) -> None:
        # Successor identification depends on ordering, and archive files are not
        # guaranteed chronological (concurrent workflow dispatches interleave).
        late = _run("LATE", 100.30, 100.30, hour=5)
        early = _run("EARLY", 100.00, 100.00, hour=1)
        rows = reconcile(late + early, "personal")
        assert [row.run_id for row in rows] == ["EARLY", "LATE"]
        assert rows[0].settled_usd == pytest.approx(0.30)

    def test_unpaired_and_other_key_records_are_ignored(self) -> None:
        records = [
            *_run("A", 100.00, 100.00, hour=1),
            _snapshot("ORPHAN", "start", 999.0, ts="2026-07-25T02:00:00"),  # no end
            _snapshot("D", "start", 5.0, ts="2026-07-25T03:00:00", key="donated"),
            _snapshot("D", "end", 6.0, ts="2026-07-25T03:10:00", key="donated"),
            *_run("B", 100.30, 100.30, hour=4),
        ]
        rows = reconcile(records, "personal")
        assert [row.run_id for row in rows] == ["A", "B"]

    def test_missing_usage_yields_none_rather_than_a_wrong_number(self) -> None:
        # A fetch failure records usage=None. Treating it as 0 would invent spend.
        records = _run("A", 100.00, 100.00, hour=1)
        records[0]["usage"] = None
        records += _run("B", 100.30, 100.30, hour=2)
        rows = reconcile(records, "personal")
        assert rows[0].within_run_usd is None
        assert rows[0].settled_usd is None


def _role_row(
    run_id: str, role: str, key: str, *, usd: float | None, calls: int, costed_calls: int | None = None
) -> dict:
    """One credit_role_spend archive record (the harvested CREDIT_ROLE_SPEND line)."""
    return {
        "marker": "credit_role_spend",
        "run_id": run_id,
        "role": role,
        "key": key,
        "usd": usd,
        "calls": calls,
        "costed_calls": calls if costed_calls is None else costed_calls,
        "byok_usd": None if usd is None else 0.0,
    }


class TestRoleLedgerReconciliation:
    """The per-role ledger measures the same money as the settled delta from the other end
    (OpenRouter's per-call accounting vs. the key's booked balance), so the script has to sum
    it per run ON THE SAME KEY and keep "no cost data" distinct from zero."""

    def test_per_run_total_is_summed_for_the_requested_key_only(self) -> None:
        records = [
            _role_row("A", "forecaster:google", "personal", usd=0.25, calls=1),
            _role_row("A", "parser", "personal", usd=0.0012, calls=3),
            _role_row("A", "forecaster:openai", "donated", usd=0.40, calls=1),  # other key
            _role_row("B", "forecaster:google", "personal", usd=0.30, calls=1),
        ]
        by_run = role_spend_by_run(records, "personal")
        assert by_run["A"].usd == pytest.approx(0.2512)
        assert (by_run["A"].rows, by_run["A"].costed_rows) == (2, 2)
        assert by_run["B"].usd == pytest.approx(0.30)

    def test_run_with_only_uncosted_rows_reports_none_not_zero(self) -> None:
        records = [_role_row("A", "perplexity_research", "personal", usd=None, calls=2, costed_calls=0)]
        run_a = role_spend_by_run(records, "personal")["A"]
        assert run_a.usd is None
        assert (run_a.rows, run_a.costed_rows) == (1, 0)

    def test_aggregate_orders_by_usd_with_uncosted_last_and_respects_the_run_filter(self) -> None:
        records = [
            _role_row("A", "parser", "donated", usd=0.01, calls=3),
            _role_row("A", "forecaster:openai", "donated", usd=0.40, calls=1),
            _role_row("B", "forecaster:openai", "donated", usd=0.35, calls=1),
            _role_row("B", "untagged", "unknown", usd=None, calls=1, costed_calls=0),
            _role_row("EXCLUDED", "forecaster:openai", "donated", usd=9.0, calls=1),
        ]
        totals = aggregate_roles(records, {"A", "B"})
        assert [(t.role, t.key) for t in totals] == [
            ("forecaster:openai", "donated"),
            ("parser", "donated"),
            ("untagged", "unknown"),
        ]
        assert totals[0].usd == pytest.approx(0.75)
        assert (totals[0].calls, totals[0].costed_calls) == (2, 2)
        assert totals[2].usd is None

    def test_aggregate_without_a_filter_covers_every_run(self) -> None:
        records = [
            _role_row("A", "parser", "donated", usd=0.01, calls=1),
            _role_row("B", "parser", "donated", usd=0.02, calls=1),
        ]
        (total,) = aggregate_roles(records)
        assert total.usd == pytest.approx(0.03)
        assert total.calls == 2


class TestPrintedTotalsAndCoverage:
    """The `--roles` display arithmetic, where two adjacent "totals" and a coverage ratio can
    be read as the same quantity when they are not.

    The fixture is built so the pre-fix code printed ratios ABOVE 100%: the trailing run has
    no successor, so its settled spend is unknown and the settled total cannot include it,
    while the marker sum and the role-ledger sum both did. That is a display bug, but the
    display is the whole product here, and a coverage figure over 100% reads as "the ledger
    found money the balance did not" rather than "these two numbers cover different runs".
    """

    def _rows(self) -> list[RunSpend]:
        """Three runs: A and B settle against their successors at 0.30 each, C is final.

        C's own within-run marker is large (1.00), which is what used to inflate every ratio
        whose denominator had to leave C out.
        """
        records = (
            _run("A", 100.00, 100.10, hour=1) + _run("B", 100.30, 100.40, hour=2) + _run("C", 100.60, 101.60, hour=3)
        )
        return reconcile(records, "personal")

    def _role_records(self) -> list[dict]:
        """Personal-key rows on all three runs, plus one donated row so the table's grand
        total is a different number from the personal key's."""
        return [
            _role_row("A", "forecaster:openai", "personal", usd=0.30, calls=2),
            _role_row("B", "forecaster:openai", "personal", usd=0.30, calls=2),
            _role_row("C", "forecaster:openai", "personal", usd=0.90, calls=2),
            _role_row("B", "parser", "donated", usd=0.05, calls=4),
        ]

    def test_settled_totals_name_their_run_set_and_the_ratio_uses_it(self, capsys) -> None:
        """Capture is 0.20 of marker spend over the 2 settled runs, not 1.20 over all three."""
        rows = self._rows()

        cohort = rcs._print_key_totals(rows)
        out = capsys.readouterr().out

        assert cohort.run_ids == {"A", "B"}, "the final run has no successor, so it is not settled"
        assert cohort.usd == pytest.approx(0.60)
        assert "marker-reported total: $1.20  (all 3 runs)" in out
        assert "settled total:         $0.60  (the 2 runs with a successor)" in out
        assert "marker captured:       33% of settled spend (the 2 runs with a successor)" in out

    def test_role_coverage_shares_the_settled_run_set_with_its_denominator(self, capsys) -> None:
        """The ledger total over every selected run stays visible — it is the honest per-key
        figure — but the ratio beside it is 0.60/0.60, not 1.50/0.60."""
        rows = self._rows()
        role_records = self._role_records()
        role_totals = role_spend_by_run(role_records, "personal")

        cohort = rcs._print_key_totals(rows)
        capsys.readouterr()
        rcs._print_role_summary(role_records, role_totals, rows=rows, key="personal", settled=cohort)
        out = capsys.readouterr().out

        assert "role-ledger total (personal, all 3 runs): $1.5000" in out
        assert "role ledger covers:    100% of settled spend (the 2 runs with a successor)" in out

    def test_no_printed_coverage_percentage_exceeds_100(self, capsys) -> None:
        """The regression guard in one line: every ratio's numerator and denominator are summed
        over the same runs, so nothing in this fixture can read over 100%."""
        rows = self._rows()
        role_records = self._role_records()

        cohort = rcs._print_key_totals(rows)
        rcs._print_role_summary(
            role_records,
            role_spend_by_run(role_records, "personal"),
            rows=rows,
            key="personal",
            settled=cohort,
        )
        out = capsys.readouterr().out

        percentages = [int(match) for match in re.findall(r"(\d+)% of settled spend", out)]
        assert percentages, "the coverage lines must have printed"
        assert max(percentages) <= 100, out

    def test_the_role_table_total_is_labelled_as_spanning_both_keys(self, capsys) -> None:
        """It sits one line under a key-filtered total and its `share` column is a fraction of
        it, so two lines reading plain "total" would be two different numbers: the donated
        parser row belongs in the grand total and not in the personal key's."""
        role_records = self._role_records()

        rcs._print_role_table(role_records, {"A", "B", "C"})
        out = capsys.readouterr().out

        assert "total (all keys, costed)" in out
        assert "1.5500" in out, "the grand total spans the donated row too"
