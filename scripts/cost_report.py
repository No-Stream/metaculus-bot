"""Cost per question over the telemetry archive: per run, per role, and a weekly trend.

The documented per-question figure stood five-fold wrong for two months because the only
instrument anyone quoted was a per-key lower bound, and the role ledger printed per run with no
question denominator (``scratch/cost_pass_2026-09-09/COST_PASS.md``, section 6). This script is the
standing instrument. Over the last ``--days`` (default 30) of ``backtests/telemetry_archive/`` it
prints, per run, the questions forecast, the dollars charged and the dollars per question; per role,
the dollars, prompt tokens, output and reasoning tokens per question, the prompt-cache share and the
largest single prompt; and one trend line, the median dollars per question this week against the
prior week.

Denominators. A run's question count is its ``CREDIT_RUN_SUMMARY`` line where one exists (runs from
2026-09-09 on) and otherwise the number of its ``FORECASTERS_SURVIVED`` lines, one per question that
reached aggregation; the per-run table marks which. Per-role dollars divide that role's total over
the selected runs by the questions of every selected run that carries a role ledger, so a role that
runs on some questions only is averaged over all of them. Per-role token figures divide by the
questions of the runs whose rows carry token fields (2026-09-09 on) and read ``n/a`` before that.
A run with a ledger but no question count is listed and excluded from the per-role table.

Dollars are each row's ``charged_usd``, the money actually charged, with ``usd`` as the fallback on
rows archived before that field (``scripts/reconcile_credit_spend.row_charged_usd``). Free and
offline: it reads only the local archive, which ``make sync_telemetry`` refreshes.

Usage
-----
    uv run python scripts/cost_report.py
    uv run python scripts/cost_report.py --days 7
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import median

from scripts.reconcile_credit_spend import row_charged_usd
from scripts.telemetry.archive import load_marker_records

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
ARCHIVE_DIR: Path = REPO_ROOT / "backtests" / "telemetry_archive"

QUESTIONS_FROM_SUMMARY: str = "summary"
QUESTIONS_FROM_SURVIVED: str = "survived"


@dataclass(frozen=True)
class QuestionCount:
    """How many questions a run forecast, and which marker said so."""

    n: int
    source: str


@dataclass(frozen=True)
class RunCost:
    """One run's ledger total over its question count; ``questions`` is None when no marker counted them."""

    run_id: str
    run_date: str
    workflow: str
    questions: QuestionCount | None
    charged_usd: float | None

    @property
    def usd_per_question(self) -> float | None:
        if self.charged_usd is None or self.questions is None or self.questions.n == 0:
            return None
        return self.charged_usd / self.questions.n


@dataclass
class _RoleAccumulator:
    calls: int = 0
    costed_rows: int = 0
    charged_usd: float = 0.0
    token_rows: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    token_run_ids: set[str] = field(default_factory=set)
    max_prompt_tokens: int | None = None


@dataclass(frozen=True)
class RoleCost:
    """One role's totals over the selected runs, with the two denominators its per-question figures use.

    ``n_questions`` is the questions of every selected run with a ledger; ``token_questions`` is the
    questions of the runs whose rows carry token fields, so a token figure never divides tokens the
    archive did not record by questions it did.
    """

    role: str
    calls: int
    charged_usd: float | None
    n_questions: int
    token_questions: int
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    reasoning_tokens: int | None
    max_prompt_tokens: int | None

    @property
    def usd_per_question(self) -> float | None:
        return None if self.charged_usd is None or self.n_questions == 0 else self.charged_usd / self.n_questions

    def _per_token_question(self, count: int | None) -> float | None:
        return None if count is None or self.token_questions == 0 else count / self.token_questions

    @property
    def prompt_tokens_per_question(self) -> float | None:
        return self._per_token_question(self.prompt_tokens)

    @property
    def completion_tokens_per_question(self) -> float | None:
        return self._per_token_question(self.completion_tokens)

    @property
    def reasoning_tokens_per_question(self) -> float | None:
        return self._per_token_question(self.reasoning_tokens)

    @property
    def cached_share(self) -> float | None:
        if self.prompt_tokens is None or self.cached_tokens is None or self.prompt_tokens == 0:
            return None
        return self.cached_tokens / self.prompt_tokens


def question_counts(summaries: list[dict], survived: list[dict]) -> dict[str, QuestionCount]:
    """Per run: the ``CREDIT_RUN_SUMMARY`` count when the run has one, else its ``FORECASTERS_SURVIVED`` lines."""
    survived_by_run: dict[str, set] = defaultdict(set)
    for record in survived:
        survived_by_run[record["run_id"]].add(record.get("qid", record.get("seq")))
    counts = {run_id: QuestionCount(len(qids), QUESTIONS_FROM_SURVIVED) for run_id, qids in survived_by_run.items()}
    for record in summaries:
        counts[record["run_id"]] = QuestionCount(int(record["n_questions"]), QUESTIONS_FROM_SUMMARY)
    return counts


def rows_since(records: list[dict], since: date) -> list[dict]:
    """The records whose ``run_date`` falls on or after ``since`` (ISO ``run_date`` compares as text)."""
    floor = since.isoformat()
    return [record for record in records if str(record.get("run_date", ""))[:10] >= floor]


def run_costs(role_rows: list[dict], counts: dict[str, QuestionCount]) -> list[RunCost]:
    """One line per run that has ledger rows, oldest first."""
    by_run: dict[str, list[dict]] = defaultdict(list)
    for record in role_rows:
        by_run[record["run_id"]].append(record)
    runs = []
    for run_id, rows in by_run.items():
        charged = [charged for charged in map(row_charged_usd, rows) if charged is not None]
        runs.append(
            RunCost(
                run_id=run_id,
                run_date=str(rows[0].get("run_date") or ""),
                workflow=str(rows[0].get("workflow") or "?"),
                questions=counts.get(run_id),
                charged_usd=sum(charged) if charged else None,
            )
        )
    return sorted(runs, key=lambda run: (run.run_date, run.run_id))


def role_costs(role_rows: list[dict], counts: dict[str, QuestionCount]) -> list[RoleCost]:
    """Per role over the runs with a question count, biggest spender first, uncosted roles last."""
    counted_runs = {run_id for run_id in {record["run_id"] for record in role_rows} if run_id in counts}
    n_questions = sum(counts[run_id].n for run_id in counted_runs)
    accumulators: dict[str, _RoleAccumulator] = defaultdict(_RoleAccumulator)
    for record in role_rows:
        if record["run_id"] not in counted_runs:
            continue
        acc = accumulators[record["role"]]
        acc.calls += int(record.get("calls") or 0)
        charged = row_charged_usd(record)
        if charged is not None:
            acc.costed_rows += 1
            acc.charged_usd += charged
        if record.get("prompt_tokens") is not None:
            acc.token_rows += 1
            acc.token_run_ids.add(record["run_id"])
            acc.prompt_tokens += int(record["prompt_tokens"])
            acc.completion_tokens += int(record.get("completion_tokens") or 0)
            acc.cached_tokens += int(record.get("cached_tokens") or 0)
            acc.reasoning_tokens += int(record.get("reasoning_tokens") or 0)
        if record.get("max_prompt_tokens") is not None:
            acc.max_prompt_tokens = max(acc.max_prompt_tokens or 0, int(record["max_prompt_tokens"]))
    roles = [
        RoleCost(
            role=role,
            calls=acc.calls,
            charged_usd=acc.charged_usd if acc.costed_rows else None,
            n_questions=n_questions,
            token_questions=sum(counts[run_id].n for run_id in acc.token_run_ids),
            prompt_tokens=acc.prompt_tokens if acc.token_rows else None,
            completion_tokens=acc.completion_tokens if acc.token_rows else None,
            cached_tokens=acc.cached_tokens if acc.token_rows else None,
            reasoning_tokens=acc.reasoning_tokens if acc.token_rows else None,
            max_prompt_tokens=acc.max_prompt_tokens,
        )
        for role, acc in accumulators.items()
    ]
    return sorted(roles, key=lambda role: (role.charged_usd is None, -(role.charged_usd or 0.0), role.role))


@dataclass(frozen=True)
class WeekMedian:
    """Median dollars per question over the runs of one seven-day window, and how many runs it covers."""

    median_usd_per_question: float | None
    n_runs: int


def weekly_trend(runs: list[RunCost], today: date) -> tuple[WeekMedian, WeekMedian]:
    """``(this_week, prior_week)``: runs dated within the last 7 days, and within the 7 before those."""

    def window(start: date, end: date) -> WeekMedian:
        rates = [
            run.usd_per_question
            for run in runs
            if start.isoformat() <= run.run_date[:10] < end.isoformat() and run.usd_per_question is not None
        ]
        return WeekMedian(median(rates) if rates else None, len(rates))

    week = timedelta(days=7)
    tomorrow = today + timedelta(days=1)
    return window(tomorrow - week, tomorrow), window(tomorrow - 2 * week, tomorrow - week)


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _tokens(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _run_lines(runs: list[RunCost]) -> list[str]:
    lines = [f"{'run_date':17} {'run_id':12} {'workflow':16} {'questions':>9} {'charged':>9} {'$/question':>11}"]
    for run in runs:
        questions = "n/a" if run.questions is None else str(run.questions.n)
        marker = "" if run.questions is None or run.questions.source == QUESTIONS_FROM_SUMMARY else "*"
        lines.append(
            f"{run.run_date[:16]:17} {run.run_id:12} {run.workflow:16} {questions + marker:>9} "
            f"{_usd(run.charged_usd):>9} {_usd(run.usd_per_question):>11}"
        )
    lines.append("* questions counted from FORECASTERS_SURVIVED lines (the run predates CREDIT_RUN_SUMMARY)")
    return lines


def _role_lines(roles: list[RoleCost]) -> list[str]:
    if not roles:
        return ["no role ledger rows on a run with a question count"]
    header = (
        f"{'role':24} {'$/question':>10} {'prompt_tok/q':>12} {'cached':>7} {'out_tok/q':>10} "
        f"{'reason_tok/q':>12} {'max_prompt':>11}"
    )
    lines = [f"per role, over {roles[0].n_questions} questions:", header]
    for role in roles:
        lines.append(
            f"{role.role:24} {_usd(role.usd_per_question):>10} {_tokens(role.prompt_tokens_per_question):>12} "
            f"{_share(role.cached_share):>7} {_tokens(role.completion_tokens_per_question):>10} "
            f"{_tokens(role.reasoning_tokens_per_question):>12} {_tokens(role.max_prompt_tokens):>11}"
        )
    total = sum(role.charged_usd or 0.0 for role in roles)
    per_question = None if roles[0].n_questions == 0 else total / roles[0].n_questions
    lines.append(f"{'total (costed roles)':24} {_usd(per_question):>10}")
    lines.append("tokens read n/a on rows archived before the token fields (2026-09-09)")
    return lines


def _trend_line(this_week: WeekMedian, prior_week: WeekMedian) -> str:
    return (
        f"trend: median $/question this week {_usd(this_week.median_usd_per_question)} (n={this_week.n_runs}) "
        f"vs prior week {_usd(prior_week.median_usd_per_question)} (n={prior_week.n_runs})"
    )


def render_report(
    runs: list[RunCost], roles: list[RoleCost], trend: tuple[WeekMedian, WeekMedian], *, days: int, since: date
) -> str:
    total_questions = sum(run.questions.n for run in runs if run.questions is not None)
    total_charged = sum(run.charged_usd or 0.0 for run in runs)
    lines = [
        f"Cost per question, telemetry archive, last {days} days (runs since {since.isoformat()}): "
        f"{len(runs)} runs, {total_questions} questions, ${total_charged:.2f} charged",
        *_run_lines(runs),
        "",
        *_role_lines(roles),
        "",
        _trend_line(*trend),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR, help="telemetry archive directory")
    parser.add_argument("--days", type=int, default=30, help="window in days, ending today (default 30)")
    args = parser.parse_args()

    if not args.archive_dir.exists():
        raise SystemExit(f"archive not found at {args.archive_dir}; run `make sync_telemetry` (free) to populate it")

    today = datetime.now(UTC).date()
    since = today - timedelta(days=args.days)
    role_rows = rows_since(load_marker_records(args.archive_dir, "credit_role_spend"), since)
    if not role_rows:
        raise SystemExit(f"no CREDIT_ROLE_SPEND rows since {since.isoformat()} in {args.archive_dir}")
    counts = question_counts(
        load_marker_records(args.archive_dir, "credit_run_summary"),
        load_marker_records(args.archive_dir, "forecasters_survived"),
    )
    runs = run_costs(role_rows, counts)
    print(render_report(runs, role_costs(role_rows, counts), weekly_trend(runs, today), days=args.days, since=since))


if __name__ == "__main__":
    main()
