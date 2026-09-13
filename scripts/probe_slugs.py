"""Season-slug probe: which tournament slugs exist, which the repo is configured for, and which are stale.

Read-only and free: one authenticated GET of ``/api/projects/tournaments/<slug>/`` per candidate slug,
no LLM, research or publish call. The project object is the authoritative existence check the
season-start checklist asks for by hand -- the tournaments LIST omits any project whose ``visibility``
is ``unlisted``, which is exactly the state a new season sits in before its first question, so absence
from that list is not evidence a project does not exist (``docs/operations.md``).

Candidates are the slugs the repo's own constants point at, plus the season-successor spellings for the
current and following seasons, generated rather than hand-listed so a tracked file needs no per-round
edit. Two questions it answers: is a configured slug stale or gone (every run under it silently finds
no question), and has the next season's project appeared under a spelling the constants do not name yet.

Post and question COUNTS are ``scripts/supply_probe.py``'s job; this probe deliberately does not page
posts. Run ``make supply_probe`` for the census once a slug here reads live.

Usage:
    uv run python scripts/probe_slugs.py
    uv run python scripts/probe_slugs.py --seasons-ahead 2 --slugs market-pulse-26q2
    make probe_slugs
    make probe_slugs ARGS="--output /tmp/slugs.json"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import requests

from metaculus_bot.api_preflight import verify_api_identity
from metaculus_bot.config import load_environment
from metaculus_bot.time_utils import parse_iso_utc
from scripts.supply_probe import REQUEST_SPACING_SECS, get_json
from scripts.supply_probe_platforms import DEFAULT_SLUGS, METACULUS_PROBE

logger = logging.getLogger(__name__)

SEASONS: tuple[str, ...] = ("winter", "spring", "summer", "fall")
_SEASON_INDEX_BY_MONTH: dict[int, int] = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}

# Every spelling a Metaculus bot season has used or plausibly could; a rename surfaces as a live row.
SLUG_TEMPLATES: tuple[str, ...] = (
    "{season}-futureeval-{year}",
    "futureeval-{season}-{year}",
    "{season}-aib-{year}",
    "aib-{season}-{year}",
    "metaculus-cup-{season}-{year}",
)
DEFAULT_SEASONS_AHEAD = 1
# The only status that means "this slug names no project"; anything else leaves existence unknown.
NOT_FOUND_STATUS = 404
NAME_WIDTH = 34


@dataclass(frozen=True)
class ProjectRow:
    """One candidate slug's project object, or the reason the probe could not read it.

    ``absent`` is a definite 404 and ``error`` without it is a measurement failure, because "no such
    season" and "the probe could not tell" license different actions and only the first is news.
    """

    slug: str
    configured: bool
    project_id: int | None = None
    name: str | None = None
    visibility: str | None = None
    questions_count: int | None = None
    forecasting_end_date: str | None = None
    absent: bool = False
    error: str | None = None

    @property
    def exists(self) -> bool:
        return self.project_id is not None

    def forecasting_closed(self, now: datetime) -> bool:
        """Whether the season's forecasting window has already shut."""
        end = parse_iso_utc(self.forecasting_end_date)
        return end is not None and end < now

    def flags(self, now: datetime) -> list[str]:
        if self.configured and self.absent:
            return ["CONFIGURED BUT ABSENT: every run under this slug finds no question"]
        if not self.exists:
            return [f"existence unknown ({self.error})"] if self.error and not self.absent else []
        if self.configured and self.forecasting_closed(now):
            return [f"forecasting closed {self.forecasting_end_date}: re-point the constant at the next season"]
        if not self.configured and not self.forecasting_closed(now):
            return ["LIVE AND UNCONFIGURED: a season the repo's constants do not name"]
        return []


def season_of(day: date) -> tuple[str, int]:
    """The season a date falls in, with the year its slug carries.

    A winter season opening in December carries the FOLLOWING year, which is why the 2026-09 rounds
    probed ``winter-futureeval-2027``.
    """
    year = day.year + 1 if day.month == 12 else day.year
    return SEASONS[_SEASON_INDEX_BY_MONTH[day.month]], year


def next_season(season: str, year: int) -> tuple[str, int]:
    index = SEASONS.index(season) + 1
    return SEASONS[index % len(SEASONS)], year + 1 if index >= len(SEASONS) else year


def candidate_slugs(
    day: date, *, seasons_ahead: int = DEFAULT_SEASONS_AHEAD, extra: Sequence[str] = ()
) -> tuple[str, ...]:
    """The configured slugs first, then every template spelling for this season and the next few. Pure."""
    seasons = [season_of(day)]
    for _ in range(seasons_ahead):
        seasons.append(next_season(*seasons[-1]))
    generated = [template.format(season=season, year=year) for season, year in seasons for template in SLUG_TEMPLATES]
    return tuple(dict.fromkeys([*DEFAULT_SLUGS, *generated, *extra]))


def fetch_project(slug: str, token: str, *, base_url: str) -> ProjectRow:
    """Read one slug's project object. Soft-fails: an unreadable slug becomes a row, never an abort."""
    configured = slug in DEFAULT_SLUGS
    try:
        project = get_json({}, token, url=f"{base_url}/projects/tournaments/{slug}/")
    except requests.RequestException as exc:
        status = exc.response.status_code if exc.response is not None else None
        return ProjectRow(
            slug=slug,
            configured=configured,
            absent=status == NOT_FOUND_STATUS,
            error=f"HTTP {status}" if status else str(exc),
        )
    return ProjectRow(
        slug=slug,
        configured=configured,
        project_id=project.get("id"),
        name=project.get("name"),
        visibility=project.get("visibility"),
        questions_count=project.get("questions_count"),
        forecasting_end_date=project.get("forecasting_end_date"),
    )


def probe_projects(slugs: Sequence[str], token: str, *, base_url: str) -> list[ProjectRow]:
    """Read every candidate slug, spaced like the other read-only walkers."""
    rows: list[ProjectRow] = []
    for index, slug in enumerate(slugs):
        row = fetch_project(slug, token, base_url=base_url)
        rows.append(row)
        logger.info(f"{slug}: {f'project {row.project_id}' if row.exists else row.error}")
        if index < len(slugs) - 1:
            time.sleep(REQUEST_SPACING_SECS)
    return rows


def _row_line(row: ProjectRow) -> str:
    state = "live" if row.exists else ("absent" if row.absent else "unknown")
    return (
        f"  {row.slug:<30} {'yes' if row.configured else '-':<4} {state:<8} "
        f"{row.project_id!s:>7} {row.questions_count!s:>7}  {(row.visibility or '-'):<9} "
        f"{(row.forecasting_end_date or '-')[:10]:<11} {(row.name or row.error or '-')[:NAME_WIDTH]}"
    )


def render_report(rows: Sequence[ProjectRow], *, now: datetime) -> str:
    """Render the probe as text. Pure: no clock read, no IO."""
    lines = [
        f"Metaculus season-slug probe as of {now.isoformat()}. `cfg` marks a slug the repo's constants "
        "point at; post counts are `make supply_probe`.",
        f"  {'slug':<30} {'cfg':<4} {'state':<8} {'id':>7} {'quest':>7}  {'visible':<9} {'fcast_end':<11} name",
    ]
    lines.extend(_row_line(row) for row in rows)
    findings = [(row, flag) for row in rows for flag in row.flags(now)]
    lines.append("")
    if not findings:
        lines.append("  nothing to act on: every configured slug is live and open, and no unconfigured season is.")
        return "\n".join(lines)
    lines.append("  findings:")
    lines.extend(f"    {row.slug}: {flag}" for row, flag in findings)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Probe the repo's configured tournament slugs and the season-successor spellings for "
        "existence, visibility and whether their forecasting window is still open."
    )
    parser.add_argument(
        "--seasons-ahead",
        type=int,
        default=DEFAULT_SEASONS_AHEAD,
        help="Seasons past the current one to generate candidate spellings for (default: %(default)s)",
    )
    parser.add_argument("--slugs", nargs="+", default=(), help="Extra slugs to probe alongside the generated set")
    parser.add_argument("--output", default=None, help="Optional path to dump the rows as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    load_environment()
    token = os.environ.get(METACULUS_PROBE.token_env)
    if not token:
        parser.error(f"{METACULUS_PROBE.token_env} is not set (put it in .env or the environment)")

    # The host is vetted before any token goes out (DNS-parking incident; see metaculus_bot/api_preflight.py).
    verify_api_identity(METACULUS_PROBE.base_url)

    now = datetime.now(UTC)
    slugs = candidate_slugs(now.date(), seasons_ahead=args.seasons_ahead, extra=args.slugs)
    rows = probe_projects(slugs, token, base_url=METACULUS_PROBE.base_url)
    print(render_report(rows, now=now))

    if args.output:
        payload = {"generated_at": now.isoformat(), "rows": [asdict(row) for row in rows]}
        Path(args.output).write_text(json.dumps(payload, indent=2))
        logger.info(f"Wrote {len(rows)} slug row(s) to {args.output}")


if __name__ == "__main__":
    main()
