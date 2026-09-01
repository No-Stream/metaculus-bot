"""Question-supply probe over Metaculus post statuses, INCLUDING ``closed``.

Why this exists: two consecutive residual rounds' supply projections missed, both for the
same reason. A question that has closed to forecasting but has not resolved yet sits at
post status ``closed``, and each round's probe queried only ``statuses=resolved`` and
``statuses=open``, so those questions were invisible. On 2026-08-31 the summer tournament
held 178 posts at ``closed``; 26 of them were the frozen-triple checkpoint cohort the
projection was about, and 16 of those were already past their own
``scheduled_resolve_time`` (worst 17.1 days). Both probes were scratch scripts, so the fix
kept getting re-lost — hence a tracked utility with tests.

What it reports per slug: posts and questions at each requested status, and the backlog of
UNRESOLVED questions already past their own ``scheduled_resolve_time`` with the worst
overdue margin. The backlog is the number that tells a supply projection whether questions
are late on Metaculus's side (nothing we can do) rather than missing from our pull.

Read-only and free: it hits only the Metaculus posts list — no LLM call, no research
provider, no publish — so it sits outside the repo's cost gate.

Two API facts it is built around, both learned by the scratch probes it replaces:

* The tournament-filtered posts list gives no usable total, so paging stops on the first
  short page rather than trusting ``count``/``next``.
* The endpoint rate-limits aggressively right after a full performance pull, so every
  request carries a bounded 429 retry and pages are spaced.

A slug that errors (the bare ``metaculus-cup`` slug 404s today — see the fall-cup note in
``metaculus_bot/constants.py``) is reported as an error row and the survey continues, so
one dead slug cannot hide the live ones. That also makes this the cheapest way to watch
for the fall cup opening questions: the ``metaculus-cup-fall-2026`` row goes from zero
posts to non-zero on the day it does.

Usage:
    uv run python scripts/supply_probe.py
    uv run python scripts/supply_probe.py --slugs summer-futureeval-2026 --statuses open closed
    make supply_probe
    make supply_probe ARGS="--slugs metaculus-cup-fall-2026 --output /tmp/supply.json"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from forecasting_tools import MetaculusApi
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from metaculus_bot.api_preflight import verify_metaculus_api_identity
from metaculus_bot.config import load_environment
from metaculus_bot.constants import FALL_CUP_SLUG, METACULUS_CUP_ID, TOURNAMENT_ID

# The scoring pull's own post-unwrapping, shared rather than re-derived: both read the same
# posts list, and a probe that counted questions differently from the pull it exists to
# project would be answering a subtly different question.
from metaculus_bot.performance_analysis.collector import questions_on_post
from metaculus_bot.time_utils import _as_utc, parse_iso_utc

logger = logging.getLogger(__name__)

# Read off the client rather than hardcoded, so the host this probe sends the token to is
# the same host `verify_metaculus_api_identity` vetted (it derives its preflight URL the same
# way). Both honor a METACULUS_API_BASE_URL override, including one set in a .env file: this
# assignment runs after the imports above, and importing `metaculus_bot.constants` is what
# loads .env / .env.local, while the preflight resolves its own URL per call for the same
# reason (see `api_preflight.preflight_url`).
POSTS_URL = f"{MetaculusClient().base_url}/posts/"

# The three statuses the receipts exercised. `closed` is the whole point of the utility;
# pass --statuses to ask about any other status the API accepts.
DEFAULT_STATUSES: tuple[str, ...] = ("open", "closed", "resolved")

# The repo's own season slugs, deduplicated so re-pointing METACULUS_CUP_ID at the dated
# fall slug collapses two rows into one instead of probing it twice. Minibench comes from
# forecasting-tools, the same spelling cli.py forecasts on.
DEFAULT_SLUGS: tuple[str, ...] = tuple(
    dict.fromkeys([TOURNAMENT_ID, METACULUS_CUP_ID, FALL_CUP_SLUG, MetaculusApi.CURRENT_MINIBENCH_ID])
)

PAGE_SIZE = 100
MAX_PAGES = 40  # 4,000 posts per status — an order of magnitude above any slug we probe
REQUEST_SPACING_SECS = 1.0
REQUEST_TIMEOUT_SECS = 45
MAX_RETRIES = 6
RETRY_BACKOFF_SECS = 6.0
SECONDS_PER_DAY = 86_400.0
DEFAULT_MAX_BACKLOG_ROWS = 20


@dataclass(frozen=True)
class QuestionRow:
    """One forecastable question, tagged with the post status it was paged under."""

    question_id: int
    post_id: int | None
    post_status: str
    question_type: str | None
    title: str
    scheduled_resolve_time: str | None
    is_resolved: bool


@dataclass(frozen=True)
class StatusCount:
    status: str
    posts: int
    questions: int


@dataclass(frozen=True)
class BacklogRow:
    """An unresolved question already past its own scheduled resolve time."""

    question_id: int
    post_id: int | None
    post_status: str
    question_type: str | None
    title: str
    scheduled_resolve_time: str
    overdue_days: float


@dataclass(frozen=True)
class SlugSupply:
    """One slug's supply census, or the error that stopped it."""

    slug: str
    status_counts: tuple[StatusCount, ...] = ()
    total_posts: int = 0
    total_questions: int = 0
    backlog: tuple[BacklogRow, ...] = ()
    unresolved_without_schedule: int = 0
    resolved_within_unresolved_posts: int = 0
    error: str | None = None

    @property
    def worst_overdue_days(self) -> float:
        """Overdue margin of the worst backlog question; 0.0 when nothing is overdue."""
        return self.backlog[0].overdue_days if self.backlog else 0.0


def _question_is_resolved(question: Mapping[str, Any], post_status: str) -> bool:
    """Whether THIS question has resolved, independent of its post's status.

    A group post's status is the post's: its members resolve on their own schedules, so
    reading post status alone counts already-resolved members of a ``closed`` group as
    backlog. ``resolution`` is compared against None rather than tested for truthiness —
    a count question can resolve to 0, and a truthiness test would file that as pending.
    """
    if post_status == "resolved":
        return True
    return question.get("actual_resolve_time") is not None or question.get("resolution") is not None


def question_rows(posts_by_status: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[QuestionRow]:
    """Flatten the per-status post pages into one row per (question, status) pairing."""
    rows: list[QuestionRow] = []
    for status, posts in posts_by_status.items():
        for post in posts:
            title = str(post.get("title") or "")
            for question in questions_on_post(post):
                rows.append(
                    QuestionRow(
                        question_id=question["id"],
                        post_id=post.get("id"),
                        post_status=status,
                        question_type=question.get("type"),
                        title=title,
                        scheduled_resolve_time=question.get("scheduled_resolve_time"),
                        is_resolved=_question_is_resolved(question, status),
                    )
                )
    return rows


def _first_per_question_id(rows: Iterable[QuestionRow]) -> list[QuestionRow]:
    """One row per question id, keeping the first — a question can page under two statuses."""
    seen: set[int] = set()
    unique: list[QuestionRow] = []
    for row in rows:
        if row.question_id in seen:
            continue
        seen.add(row.question_id)
        unique.append(row)
    return unique


def _backlog_rows(rows: Sequence[QuestionRow], now: datetime) -> tuple[tuple[BacklogRow, ...], int]:
    """Overdue rows (worst first) plus the count whose schedule was absent or unreadable.

    An unreadable schedule is DISCLOSED rather than imputed: a question with no resolve
    date is not evidence of an on-time question.
    """
    overdue: list[BacklogRow] = []
    without_schedule = 0
    for row in rows:
        scheduled = parse_iso_utc(row.scheduled_resolve_time)
        if scheduled is None:
            without_schedule += 1
            continue
        overdue_days = (now - scheduled).total_seconds() / SECONDS_PER_DAY
        if overdue_days <= 0:
            continue
        overdue.append(
            BacklogRow(
                question_id=row.question_id,
                post_id=row.post_id,
                post_status=row.post_status,
                question_type=row.question_type,
                title=row.title,
                scheduled_resolve_time=str(row.scheduled_resolve_time),
                overdue_days=overdue_days,
            )
        )
    return tuple(sorted(overdue, key=lambda r: -r.overdue_days)), without_schedule


def summarize_slug_supply(
    slug: str,
    posts_by_status: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    now: datetime,
) -> SlugSupply:
    """Partition one slug's paged posts by status and compute its resolution backlog.

    Pure: the caller supplies the pages and the clock. Per-status counts report what the
    API returned for that status; the totals count each post and question once, because a
    post that resolves mid-probe can be paged under both ``closed`` and ``resolved``.
    """
    rows = question_rows(posts_by_status)
    resolved_ids = {row.question_id for row in rows if row.is_resolved}
    unresolved = [row for row in _first_per_question_id(rows) if row.question_id not in resolved_ids]
    # Scheduled times parse to tz-aware UTC, so a naive `now` from an analysis script would
    # make the overdue subtraction raise instead of answering.
    backlog, without_schedule = _backlog_rows(unresolved, _as_utc(now))

    return SlugSupply(
        slug=slug,
        status_counts=tuple(
            StatusCount(
                status=status,
                posts=len(posts),
                questions=sum(len(questions_on_post(post)) for post in posts),
            )
            for status, posts in posts_by_status.items()
        ),
        total_posts=len({post.get("id") for posts in posts_by_status.values() for post in posts}),
        total_questions=len({row.question_id for row in rows}),
        backlog=backlog,
        unresolved_without_schedule=without_schedule,
        resolved_within_unresolved_posts=len(
            {row.question_id for row in rows if row.is_resolved and row.post_status != "resolved"}
        ),
    )


def _get_json(params: dict[str, str | int], token: str) -> dict:
    """GET the posts list with a bounded, 429-aware retry.

    Local rather than reusing ``performance_analysis.collector``'s helper: that one is
    scoped to the scoring pull (three retries, and a ``RuntimeError`` when they run out),
    while this probe pages several slugs in one pass and soft-fails per slug — so an
    exhausted retry has to arrive as a ``requests`` exception for the per-slug handler.
    """
    headers = {"Authorization": f"Token {token}"}
    for attempt in range(MAX_RETRIES):
        response = requests.get(POSTS_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECS)
        if response.status_code == 429 and attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF_SECS * (attempt + 1)
            logger.warning(f"Rate limited (429); retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise requests.HTTPError(f"429 rate limit: retries exhausted after {MAX_RETRIES} attempts")


def fetch_posts_by_status(slug: str, statuses: Sequence[str], token: str) -> dict[str, list[dict]]:
    """Page every requested status for one slug. Raises ``requests.RequestException``.

    Stops on the first short page: the scratch probes this replaces found the
    tournament-filtered list serving no usable total, so page length is the only
    end-of-results signal we trust. ``MAX_PAGES`` bounds the walk.
    """
    posts_by_status: dict[str, list[dict]] = {}
    for status in statuses:
        posts: list[dict] = []
        for page in range(MAX_PAGES):
            data = _get_json(
                {"tournaments": slug, "statuses": status, "limit": PAGE_SIZE, "offset": page * PAGE_SIZE},
                token,
            )
            results = data.get("results") or []
            posts.extend(results)
            if len(results) < PAGE_SIZE:
                break
            time.sleep(REQUEST_SPACING_SECS)
        else:
            logger.warning(f"{slug} statuses={status}: hit MAX_PAGES={MAX_PAGES}; counts are a lower bound")
        logger.info(f"{slug} statuses={status}: {len(posts)} posts")
        posts_by_status[status] = posts
    return posts_by_status


def probe_slugs(
    slugs: Sequence[str],
    statuses: Sequence[str],
    token: str,
    *,
    now: datetime,
) -> list[SlugSupply]:
    """Survey every slug, soft-failing per slug so one dead slug reports as an error row.

    Scoped to ``requests.RequestException`` (transport, HTTP status and JSON-decode
    failures of the call). A survey over several slugs expects some to be dead — the bare
    ``metaculus-cup`` slug 404s today — and aborting the whole run on the first would hide
    the live ones. Anything that is not a request failure is a contract break and crashes.
    """
    supplies: list[SlugSupply] = []
    for slug in slugs:
        try:
            posts_by_status = fetch_posts_by_status(slug, statuses, token)
        except requests.RequestException as exc:
            logger.warning(f"{slug}: supply probe failed ({exc})")
            supplies.append(SlugSupply(slug=slug, error=str(exc)))
            continue
        supplies.append(summarize_slug_supply(slug, posts_by_status, now=now))
    return supplies


def _render_backlog(supply: SlugSupply, max_rows: int) -> list[str]:
    lines: list[str] = []
    if not supply.backlog:
        lines.append("  Unresolved past scheduled_resolve_time: 0")
    else:
        lines.append(
            f"  Unresolved past scheduled_resolve_time: {len(supply.backlog)} "
            f"(worst {supply.worst_overdue_days:.1f} days overdue)"
        )
        lines.append(f"    {'qid':>8} {'post':>8} {'status':<9} {'overdue_d':>9}  {'scheduled':<17} title")
        for row in supply.backlog[:max_rows]:
            lines.append(
                f"    {row.question_id:>8} {row.post_id!s:>8} {row.post_status:<9} {row.overdue_days:>9.1f}  "
                f"{row.scheduled_resolve_time[:16]:<17} {row.title[:60]}"
            )
        hidden = max(0, len(supply.backlog) - max_rows)
        if hidden:
            lines.append(f"    +{hidden} more overdue (raise --max-backlog-rows to see them)")
    if supply.unresolved_without_schedule:
        lines.append(f"  Unresolved with no readable scheduled_resolve_time: {supply.unresolved_without_schedule}")
    if supply.resolved_within_unresolved_posts:
        lines.append(
            f"  Questions already resolved inside non-resolved posts: {supply.resolved_within_unresolved_posts}"
        )
    return lines


def render_report(
    supplies: Sequence[SlugSupply],
    *,
    now: datetime,
    max_backlog_rows: int = DEFAULT_MAX_BACKLOG_ROWS,
) -> str:
    """Render the survey as text. Pure — no clock read, no IO."""
    lines = [
        "Metaculus question-supply probe. Post status `closed` means closed to forecasting but NOT yet resolved.",
        f"as of {now.isoformat()}",
    ]
    for supply in supplies:
        lines.append("")
        lines.append(f"=== {supply.slug} ===")
        if supply.error is not None:
            lines.append(f"  ERROR: {supply.error} (no counts for this slug)")
            continue
        lines.append(f"  {'status':<12}{'posts':>8}{'questions':>11}")
        for count in supply.status_counts:
            lines.append(f"  {count.status:<12}{count.posts:>8}{count.questions:>11}")
        lines.append(f"  {'total':<12}{supply.total_posts:>8}{supply.total_questions:>11}")
        lines.extend(_render_backlog(supply, max_backlog_rows))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Report Metaculus question supply per tournament slug, counting `closed` posts."
    )
    parser.add_argument(
        "--slugs",
        nargs="+",
        default=list(DEFAULT_SLUGS),
        help=f"Tournament slugs to probe (default: {' '.join(DEFAULT_SLUGS)})",
    )
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=list(DEFAULT_STATUSES),
        help=f"Post statuses to count (default: {' '.join(DEFAULT_STATUSES)})",
    )
    parser.add_argument(
        "--max-backlog-rows",
        type=int,
        default=DEFAULT_MAX_BACKLOG_ROWS,
        help="Overdue questions listed per slug (default: %(default)s)",
    )
    parser.add_argument("--output", default=None, help="Optional path to dump the census as JSON.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    load_environment()
    token = os.environ.get("METACULUS_TOKEN")
    if not token:
        parser.error("METACULUS_TOKEN is not set (put it in .env or the environment)")

    # Confirm the host is the real Metaculus before the token goes out (DNS-parking
    # incident — see metaculus_bot/api_preflight.py).
    verify_metaculus_api_identity()

    now = datetime.now(UTC)
    supplies = probe_slugs(args.slugs, tuple(args.statuses), token, now=now)
    print(render_report(supplies, now=now, max_backlog_rows=args.max_backlog_rows))

    if args.output:
        payload = {"generated_at": now.isoformat(), "slugs": [asdict(supply) for supply in supplies]}
        Path(args.output).write_text(json.dumps(payload, indent=2))
        logger.info(f"Wrote supply census to {args.output}")


if __name__ == "__main__":
    main()
