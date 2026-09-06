"""Question-supply probe over Metaculus post statuses, INCLUDING ``closed``.

Why this exists: two consecutive residual rounds' supply projections missed, both for the
same reason. A question that has closed to forecasting but has not resolved yet sits at
post status ``closed``, and each round's probe queried only ``statuses=resolved`` and
``statuses=open``, so those questions were invisible. On 2026-08-31 the summer tournament
held 178 posts at ``closed``; 26 of them were the frozen-triple checkpoint cohort the
projection was about, and 16 of those were already past their own
``scheduled_resolve_time`` (worst 17.1 days). Both probes were scratch scripts, so the fix
kept getting re-lost — hence a tracked utility with tests.

What it reports per slug: posts and questions at each requested status, the backlog of
UNRESOLVED questions already past their own ``scheduled_resolve_time`` with the worst
overdue margin, and the FORFEIT sweep described next. The backlog is the number that tells
a supply projection whether questions are late on Metaculus's side (nothing we can do)
rather than missing from our pull.

**The forfeit sweep** lists every question on a ``closed`` or ``resolved`` post that the
bot never forecast at all, with its open/close window. This is a supply question, not a
scoring one: a forfeited question never reaches the performance dataset (the collector drops
a question with no ``my_forecasts.latest``), so a sweep that starts from questions the bot
intook cannot see one. The 2026-09-01 residual round found the triple era had lost SIX
questions to delivery where the prior sweep saw one — q44801 to a cron gap, q45085 to a
late submit against a 12:00 close, q45093 / q45374 / q45375 to cancelled runs, q45216 to a
retroactive close — which is why this belongs in the weekly read rather than in a round's
scratch scripts. Only ``closed`` and ``resolved`` posts count: an open question the bot has
not forecast YET is not a forfeit.

Resolving "did we forecast this" needs ``my_forecasts``, which the posts LIST payload does
not reliably carry (the scoring pull fetches every post individually for exactly that
reason), so the sweep reads the list payload where the key is there and issues one
per-post detail GET where it is not. Questions whose state stays unreadable are counted
and disclosed as ``unknown`` rather than filed as forfeits — under-reporting a forfeit is
recoverable, calling a forecast question forfeited is not.

Read-only and free: it hits only the Metaculus posts list and post detail — no LLM call, no
research provider, no publish — so it sits outside the repo's cost gate.

Two API facts it is built around, both learned by the scratch probes it replaces:

* The tournament-filtered posts list gives no usable total, so paging stops on the first
  short page rather than trusting ``count``/``next``.
* The endpoint rate-limits aggressively right after a full performance pull, so every
  request carries a bounded 429 retry and pages are spaced.

A slug that errors (the bare ``metaculus-cup`` slug returns 400 today — see the fall-cup note in
``metaculus_bot/constants.py``) is reported as an error row and the survey continues, so
one dead slug cannot hide the live ones. That also makes this the cheapest way to watch
for the fall cup opening questions: the ``metaculus-cup-fall-2026`` row goes from zero
posts to non-zero on the day it does.

Usage:
    uv run python scripts/supply_probe.py
    uv run python scripts/supply_probe.py --slugs fall-futureeval-2026 --statuses open closed
    uv run python scripts/supply_probe.py --no-forfeits          # counts only, no detail GETs
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
from metaculus_bot.performance_analysis.collector import FETCH_DELAY_SECS, questions_on_post
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

# The forfeit sweep's scope. An OPEN question the bot has not forecast yet is not a forfeit,
# so only posts whose forecasting window has shut are candidates.
FORFEIT_STATUSES: tuple[str, ...] = ("closed", "resolved")

# The repo's own season slugs, deduplicated so re-pointing METACULUS_CUP_ID at the dated
# fall slug collapses two rows into one instead of probing it twice. Minibench comes from
# forecasting-tools, the same spelling cli.py forecasts on.
DEFAULT_SLUGS: tuple[str, ...] = tuple(
    dict.fromkeys([TOURNAMENT_ID, METACULUS_CUP_ID, FALL_CUP_SLUG, MetaculusApi.CURRENT_MINIBENCH_ID])
)

PAGE_SIZE = 100
MAX_PAGES = 40  # 4,000 posts per status — an order of magnitude above any slug we probe
REQUEST_SPACING_SECS = 1.0
# Per-post detail GETs are spaced tighter than page GETs: the forfeit sweep can issue a few
# hundred of them for a full season. Imported from the scoring pull rather than copied, so the
# two read-only Metaculus walkers cannot drift into different politeness.
DETAIL_REQUEST_SPACING_SECS = FETCH_DELAY_SECS
# How often the sweep says where it is. Every GET is a DEBUG line; at 0.5 s spacing a
# 25-post cadence puts an INFO line on the console about every 13 seconds, which is often
# enough to tell a slow sweep from a wedged one without burying the slug's own summary.
DETAIL_PROGRESS_EVERY = 25
REQUEST_TIMEOUT_SECS = 45
MAX_RETRIES = 6
RETRY_BACKOFF_SECS = 6.0
SECONDS_PER_DAY = 86_400.0
SECONDS_PER_HOUR = 3_600.0
DEFAULT_MAX_BACKLOG_ROWS = 20
DEFAULT_MAX_FORFEIT_ROWS = 20

# What ``bot_forecast_state`` can answer. UNKNOWN is a measurement failure, not a forfeit.
FORECAST_PRESENT = "forecast"
FORECAST_ABSENT = "no_forecast"
FORECAST_UNKNOWN = "unknown"


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
    open_time: str | None = None
    close_time: str | None = None
    # One of FORECAST_PRESENT / FORECAST_ABSENT / FORECAST_UNKNOWN. UNKNOWN on any payload
    # that carried no readable ``my_forecasts`` — a list page the sweep did not enrich, or a
    # detail page that answered with a null block.
    forecast_state: str = FORECAST_UNKNOWN


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
class ForfeitRow:
    """A question whose forecasting window shut without the bot ever forecasting it."""

    question_id: int
    post_id: int | None
    post_status: str
    question_type: str | None
    title: str
    open_time: str | None
    close_time: str | None
    window_hours: float | None
    is_resolved: bool


@dataclass(frozen=True)
class ForecastStateCounts:
    """How the forfeit-eligible questions split on "did we forecast this".

    ``unknown`` is disclosed rather than folded into either arm: it means the payload never
    answered, which is not the same fact as a forfeit. A slug where ``with_forecast`` is 0
    while ``without_forecast`` is large is far more likely a non-bot ``METACULUS_TOKEN`` than
    a total forfeit, and the split is what makes that readable.
    """

    with_forecast: int = 0
    without_forecast: int = 0
    unknown: int = 0

    @property
    def total(self) -> int:
        return self.with_forecast + self.without_forecast + self.unknown


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
    forfeits: tuple[ForfeitRow, ...] = ()
    forecast_states: ForecastStateCounts = ForecastStateCounts()
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


def bot_forecast_state(question: Mapping[str, Any]) -> str:
    """Whether the token's own user forecast THIS question, per its ``my_forecasts`` block.

    Three answers, because "the payload says we did not forecast it" and "the payload does
    not say" are different facts and only the first is a forfeit. A list-page question dict
    carries no ``my_forecasts`` at all, so it answers UNKNOWN until the sweep enriches it
    from a per-post detail GET.

    ``history`` is the authoritative emptiness test (the operator's own read of the API), but
    a non-empty ``latest`` also counts as present: this must never call a real forecast a
    forfeit, and the scoring collector keys on ``latest``.
    """
    if "my_forecasts" not in question:
        return FORECAST_UNKNOWN
    my_forecasts = question.get("my_forecasts")
    if not isinstance(my_forecasts, Mapping):
        # Present but null/scalar: the block carried no answer, so neither do we.
        return FORECAST_UNKNOWN
    if my_forecasts.get("history") or my_forecasts.get("latest"):
        return FORECAST_PRESENT
    return FORECAST_ABSENT


def _close_time(question: Mapping[str, Any]) -> str | None:
    """When forecasting actually shut, preferring the realized close over the scheduled one."""
    return question.get("actual_close_time") or question.get("scheduled_close_time")


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
                        open_time=question.get("open_time"),
                        close_time=_close_time(question),
                        forecast_state=bot_forecast_state(question),
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


def _window_hours(row: QuestionRow) -> float | None:
    """Length of the forecasting window in hours, or None when either end is unreadable."""
    opened, closed = parse_iso_utc(row.open_time), parse_iso_utc(row.close_time)
    if opened is None or closed is None:
        return None
    return (closed - opened).total_seconds() / SECONDS_PER_HOUR


def _forfeit_rows(rows: Sequence[QuestionRow]) -> tuple[tuple[ForfeitRow, ...], ForecastStateCounts]:
    """Forfeited questions (newest window first) plus the forecast-state split behind them.

    Newest first because a weekly read is about what we just lost; the window length rides
    each row instead of ordering it, since a short window and a stale one are different
    diagnoses and only one of them is urgent.
    """
    # Deduped, because a post that resolves mid-probe pages under both `closed` and
    # `resolved` and would otherwise be counted (and listed) twice. Every copy of a post
    # carries the same enrichment, so which one survives changes only the reported status.
    eligible = _first_per_question_id(row for row in rows if row.post_status in FORFEIT_STATUSES)
    counts = ForecastStateCounts(
        with_forecast=sum(1 for row in eligible if row.forecast_state == FORECAST_PRESENT),
        without_forecast=sum(1 for row in eligible if row.forecast_state == FORECAST_ABSENT),
        unknown=sum(1 for row in eligible if row.forecast_state == FORECAST_UNKNOWN),
    )
    forfeits = [
        ForfeitRow(
            question_id=row.question_id,
            post_id=row.post_id,
            post_status=row.post_status,
            question_type=row.question_type,
            title=row.title,
            open_time=row.open_time,
            close_time=row.close_time,
            window_hours=_window_hours(row),
            is_resolved=row.is_resolved,
        )
        for row in eligible
        if row.forecast_state == FORECAST_ABSENT
    ]
    forfeits.sort(key=lambda row: (row.open_time or "", row.question_id), reverse=True)
    return tuple(forfeits), counts


def summarize_slug_supply(
    slug: str,
    posts_by_status: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    now: datetime,
) -> SlugSupply:
    """Partition one slug's paged posts by status, then compute its backlog and forfeits.

    Pure: the caller supplies the pages and the clock. Per-status counts report what the
    API returned for that status; the totals count each post and question once, because a
    post that resolves mid-probe can be paged under both ``closed`` and ``resolved``.

    The forfeit sweep reads whatever ``my_forecasts`` the supplied payloads carry. On raw
    list pages that is nothing, so every eligible question comes back ``unknown`` and the
    forfeit list is empty — call :func:`resolve_bot_forecasts` on the pages first (as
    :func:`probe_slugs` does) to get an answer.
    """
    rows = question_rows(posts_by_status)
    resolved_ids = {row.question_id for row in rows if row.is_resolved}
    unresolved = [row for row in _first_per_question_id(rows) if row.question_id not in resolved_ids]
    # Scheduled times parse to tz-aware UTC, so a naive `now` from an analysis script would
    # make the overdue subtraction raise instead of answering.
    backlog, without_schedule = _backlog_rows(unresolved, _as_utc(now))
    forfeits, forecast_states = _forfeit_rows(rows)

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
        forfeits=forfeits,
        forecast_states=forecast_states,
    )


def _get_json(params: dict[str, str | int], token: str, *, url: str = POSTS_URL) -> dict:
    """GET a posts endpoint with a bounded, 429-aware retry.

    ``url`` defaults to the posts LIST; the forfeit sweep passes a single post's detail URL
    through the same retry, since both endpoints share the rate limiter that motivated it.

    Local rather than reusing ``performance_analysis.collector``'s helper: that one is
    scoped to the scoring pull (three retries, and a ``RuntimeError`` when they run out),
    while this probe pages several slugs in one pass and soft-fails per slug — so an
    exhausted retry has to arrive as a ``requests`` exception for the per-slug handler.

    The exhausted 429 breaks out and raises the descriptive error below. It used to fall
    through to ``raise_for_status`` on the last attempt, which made that raise unreachable
    and reported six rate-limited attempts as one unlucky request.
    """
    headers = {"Authorization": f"Token {token}"}
    for attempt in range(MAX_RETRIES):
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECS)
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        if attempt == MAX_RETRIES - 1:
            break
        wait = RETRY_BACKOFF_SECS * (attempt + 1)
        logger.warning(f"Rate limited (429); retrying in {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
        time.sleep(wait)
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


def _posts_needing_detail(posts_by_status: Mapping[str, Sequence[dict]]) -> dict[object, str]:
    """Post ids on a forfeit-eligible status whose questions do not answer ``my_forecasts``.

    Maps id -> the status it was first seen under, purely for the log line. A post whose
    questions all already carry a readable block costs no request. Keys are typed ``object``
    because they come straight off untyped JSON; the None ones are dropped here, which is
    what lets the caller use them as lookup keys without re-checking.
    """
    needed: dict[object, str] = {}
    for status in FORFEIT_STATUSES:
        for post in posts_by_status.get(status) or []:
            post_id = post.get("id")
            if post_id is None or post_id in needed:
                continue
            questions = questions_on_post(post)
            if questions and any(bot_forecast_state(q) == FORECAST_UNKNOWN for q in questions):
                needed[post_id] = status
    return needed


def resolve_bot_forecasts(posts_by_status: dict[str, list[dict]], token: str, *, slug: str | None = None) -> int:
    """Fill in ``my_forecasts`` on forfeit-eligible posts, in place. Returns fetches issued.

    One detail GET per post that needs one, and the fetched payload replaces that post under
    EVERY status it was paged under, so the two copies of a post that resolved mid-probe
    cannot disagree about whether we forecast it.

    ``slug`` only labels the log lines. The sweep spends minutes issuing spaced GETs and used
    to say nothing while it did, so a run that had wedged looked exactly like one that was
    working; it now reports progress every ``DETAIL_PROGRESS_EVERY`` posts, with one DEBUG
    line per GET for a per-URL trace.

    A post whose detail GET fails is left as it was, which reads through as ``unknown``
    rather than as a forfeit. The exception is swallowed per post on purpose: the sweep is a
    supplement to the counts, and one unreachable post must not cost the slug its census.
    Raises nothing; an exhausted retry on EVERY post shows up as a large ``unknown`` count.
    """
    needed = _posts_needing_detail(posts_by_status)
    if not needed:
        return 0
    total = len(needed)
    label = f"{slug} forfeit sweep" if slug else "forfeit sweep"
    logger.info(f"{label}: fetching my_forecasts detail for {total} post(s)")

    fetched: dict[object, dict] = {}
    for index, (post_id, status) in enumerate(needed.items()):
        logger.debug(f"{label}: detail GET post {post_id} ({status}), {index + 1}/{total}")
        try:
            fetched[post_id] = _get_json({}, token, url=f"{POSTS_URL}{post_id}/")
        except requests.RequestException as exc:
            logger.warning(f"{label}: post {post_id} ({status}) detail fetch failed ({exc}); state stays unknown")
        done = index + 1
        if done % DETAIL_PROGRESS_EVERY == 0 and done < total:
            logger.info(f"{label}: {done}/{total} detail GETs done ({len(fetched)} answered)")
        if index < total - 1:
            time.sleep(DETAIL_REQUEST_SPACING_SECS)
    logger.info(f"{label}: {total}/{total} detail GETs done ({len(fetched)} answered)")

    for status, posts in posts_by_status.items():
        posts_by_status[status] = [fetched.get(post.get("id"), post) for post in posts]
    return len(fetched)


def probe_slugs(
    slugs: Sequence[str],
    statuses: Sequence[str],
    token: str,
    *,
    now: datetime,
    resolve_forfeits: bool = False,
) -> list[SlugSupply]:
    """Survey every slug, soft-failing per slug so one dead slug reports as an error row.

    Scoped to ``requests.RequestException`` (transport, HTTP status and JSON-decode
    failures of the call). A survey over several slugs expects some to be dead — the bare
    ``metaculus-cup`` slug returns 400 today — and aborting the whole run on the first would hide
    the live ones. Anything that is not a request failure is a contract break and crashes.

    ``resolve_forfeits`` costs one detail GET per closed/resolved post that the list page did
    not already answer for, which is a few hundred requests over a full season. It defaults
    OFF so a caller that only wants the status counts pays nothing; the CLI turns it ON
    (``--no-forfeits`` to opt out), because a forfeit is the thing the weekly read exists to
    catch.
    """
    supplies: list[SlugSupply] = []
    for slug in slugs:
        try:
            posts_by_status = fetch_posts_by_status(slug, statuses, token)
            if resolve_forfeits:
                resolve_bot_forecasts(posts_by_status, token, slug=slug)
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


def _render_forfeits(supply: SlugSupply, max_rows: int) -> list[str]:
    """The forfeit block: what the bot never forecast, and the state split behind the count."""
    states = supply.forecast_states
    if states.total == 0:
        return []
    lines = [
        f"  Closed/resolved questions never forecast by the bot: {len(supply.forfeits)} "
        f"(of {states.total}; forecast {states.with_forecast}, unknown {states.unknown})"
    ]
    if states.unknown == states.total:
        lines.append("    my_forecasts was unreadable on every one — run without --no-forfeits to resolve it")
        return lines
    if states.with_forecast == 0 and states.without_forecast:
        lines.append(
            "    !!! no question on this slug carries a bot forecast. Check that METACULUS_TOKEN is "
            "the bot's own token before reading these as forfeits."
        )
    if not supply.forfeits:
        return lines
    lines.append(f"    {'qid':>8} {'post':>8} {'status':<9} {'window_h':>8}  {'opened':<17} title")
    for row in supply.forfeits[:max_rows]:
        window = f"{row.window_hours:.1f}" if row.window_hours is not None else "n/a"
        opened = (row.open_time or "unknown")[:16]
        lines.append(
            f"    {row.question_id:>8} {row.post_id!s:>8} {row.post_status:<9} {window:>8}  "
            f"{opened:<17} {row.title[:60]}"
        )
    hidden = max(0, len(supply.forfeits) - max_rows)
    if hidden:
        lines.append(f"    +{hidden} more forfeited (raise --max-forfeit-rows to see them)")
    return lines


def render_report(
    supplies: Sequence[SlugSupply],
    *,
    now: datetime,
    max_backlog_rows: int = DEFAULT_MAX_BACKLOG_ROWS,
    max_forfeit_rows: int = DEFAULT_MAX_FORFEIT_ROWS,
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
        lines.extend(_render_forfeits(supply, max_forfeit_rows))
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
    parser.add_argument(
        "--max-forfeit-rows",
        type=int,
        default=DEFAULT_MAX_FORFEIT_ROWS,
        help="Never-forecast questions listed per slug (default: %(default)s)",
    )
    parser.add_argument(
        "--no-forfeits",
        dest="forfeits",
        action="store_false",
        help=(
            "Skip the forfeit sweep. It costs one extra read-only GET per closed/resolved post "
            "whose list page did not already carry my_forecasts; skipping leaves every question's "
            "state reported as unknown."
        ),
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
    supplies = probe_slugs(args.slugs, tuple(args.statuses), token, now=now, resolve_forfeits=args.forfeits)
    print(
        render_report(
            supplies,
            now=now,
            max_backlog_rows=args.max_backlog_rows,
            max_forfeit_rows=args.max_forfeit_rows,
        )
    )

    if args.output:
        payload = {"generated_at": now.isoformat(), "slugs": [asdict(supply) for supply in supplies]}
        Path(args.output).write_text(json.dumps(payload, indent=2))
        logger.info(f"Wrote supply census to {args.output}")


if __name__ == "__main__":
    main()
