"""Shared datetime helpers.

Kept minimal: the single UTC-normalization helper below is used by
``close_margin``, ``prompts``, and ``backtest.question_prep`` — hoisted here so
the naive-vs-aware handling lives in one place rather than being duplicated.
"""

from __future__ import annotations

from datetime import UTC, datetime


def _as_utc(moment: datetime) -> datetime:
    """Normalize a possibly-naive datetime to tz-aware UTC.

    forecasting-tools 0.2.92 parses Metaculus API timestamps as tz-aware UTC
    (``pendulum.parse`` plus an ``add_timezone_to_dates`` validator), but some
    call sites still hand in naive datetimes. Assuming naive == UTC keeps
    subtractions against ``datetime.now(timezone.utc)`` from raising
    ``TypeError`` on mixed naive/aware operands: a tz-aware value is converted,
    a naive one is stamped UTC.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def parse_iso_utc(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp (``Z`` or offset form) to tz-aware UTC.

    None on an absent or unparseable value. The one parser behind every
    era-bucketing read of ``bot_comment_created_at`` — a second copy that
    diverged (one accepting a format the other rejects) would let two modules
    file the same record into different config eras.
    """
    if not raw:
        return None
    try:
        moment = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return _as_utc(moment)
