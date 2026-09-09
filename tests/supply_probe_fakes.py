"""Dict builders shaped like the Metaculus posts-list payload, shared by the supply-probe tests.

A single-question post carries ``question``, a group post carries ``group_of_questions.questions``,
a notebook-shaped post carries neither, and every question carries ``scheduled_resolve_time`` /
``actual_resolve_time`` / ``resolution`` plus its open and close times. ``tests/test_supply_probe.py``
(the Metaculus mode) and ``tests/test_supply_probe_mantic.py`` (the Mantic mode, which adds the
public spot-time snapshot on top of this shape) both build from here so the two files cannot drift
into disagreeing about one payload.

Not named ``test_*`` on purpose: pytest imports it without collecting it. Nothing here opens a socket.
"""

from __future__ import annotations

from datetime import UTC, datetime

NOW = datetime(2026, 8, 31, 2, 24, tzinfo=UTC)


def _question(
    qid: int,
    *,
    scheduled: str | None = "2026-09-30T00:00:00Z",
    actual: str | None = None,
    resolution: object = None,
    qtype: str = "numeric",
    forecast: bool | None = None,
    open_time: str | None = "2026-07-20T03:00:00Z",
    close_time: str | None = "2026-07-20T06:00:00Z",
) -> dict:
    """One question dict.

    ``forecast`` models the three states the forfeit sweep distinguishes: None omits
    ``my_forecasts`` entirely (a posts-LIST page read without ``with_cp=true``, or without a
    token, which is what the sweep's detail GETs exist for), True carries a forecast, False
    carries the empty block a never-forecast question shows under the bot's own token.
    """
    question = {
        "id": qid,
        "type": qtype,
        "scheduled_resolve_time": scheduled,
        "actual_resolve_time": actual,
        "resolution": resolution,
        "open_time": open_time,
        "actual_close_time": close_time,
    }
    if forecast is True:
        question["my_forecasts"] = {"latest": {"forecast_values": [0.4, 0.6]}, "history": [{"id": 1}]}
    elif forecast is False:
        question["my_forecasts"] = {"latest": None, "history": []}
    return question


def _post(post_id: int, question: dict, *, title: str = "Some question?") -> dict:
    return {"id": post_id, "title": title, "question": question}


def _group_post(post_id: int, questions: list[dict], *, title: str = "A group?") -> dict:
    return {"id": post_id, "title": title, "group_of_questions": {"questions": questions}}
