"""The recorded Mantic preseason payload and its identity, shared by the Mantic test modules.

``tests/data/mantic_preseason2_posts_2026_09_08.json`` is the authenticated
``GET /api/posts/?tournaments=preseason-2`` response from the 2026-09-08 live probe against
competitions.mantic.com: four posts, one per question type, every ``my_forecasts`` block carrying an
empty history. ``tests/test_mantic_client.py`` parses it post by post and ``tests/test_mantic_e2e.py``
serves it from a fake transport under a full run; before this module each declared its own copy of
the fixture path and the four post ids, which is how two files come to disagree about one payload.

``tests/data/mantic_series1_date_post_500_2026_09_08.json`` is the second date fixture: post 500 from
the unauthenticated 2026-09-08 pull of every public Mantic date post, a RESOLVED Series 1 question in
the modal legacy shape the preseason fixture lacks (200 uniform bins whose edges fall at arbitrary
times of day, ``date_granularity`` empty, closed lower bound, OPEN upper bound, resolved
``above_upper_bound`` like half of all Series 1 date questions). Trimmed once: the ten competitor CDFs
under ``aggregations.recency_weighted.score_data.disagreement_forecasts`` are dropped (56 KB nothing
here reads); everything else, including the community's last CDF, is the wire payload verbatim. It
carries no ``my_forecasts`` block because the pull was unauthenticated.

Not named ``test_*`` on purpose: pytest imports it without collecting it, and consumers bind what
they need by import. Nothing here opens a socket.

:func:`with_prior_forecast` is the one derived shape both consumers need: a copy of a post whose
question the bot's own user has already forecast. The fixture cannot supply that state (the probe
ran before the bot ever forecast on Mantic), yet it is the state of nearly every scheduled run once
the hourly cron has fired once, and the framework's skip filter reads exactly this field.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from forecasting_tools.data_models.questions import DateQuestion

PRESEASON_FIXTURE_PATH = Path(__file__).parent / "data" / "mantic_preseason2_posts_2026_09_08.json"
LEGACY_DATE_FIXTURE_PATH = Path(__file__).parent / "data" / "mantic_series1_date_post_500_2026_09_08.json"

BINARY_POST_ID = 648
MULTIPLE_CHOICE_POST_ID = 649
DISCRETE_POST_ID = 650
DATE_POST_ID = 651
PRESEASON_POST_IDS = (BINARY_POST_ID, MULTIPLE_CHOICE_POST_ID, DISCRETE_POST_ID, DATE_POST_ID)
LEGACY_DATE_POST_ID = 500

# Shaped like a platform user id; the real bot account's id is not in the fixture and nothing reads it.
_FAKE_BOT_USER_ID = 9001
# The previous scheduled run: the bot's workflow fires hourly, so a prior forecast is about that old.
_PRIOR_FORECAST_AGE = timedelta(hours=1)


def load_preseason_posts() -> list[dict[str, Any]]:
    """The probe's four posts, freshly parsed, so a caller may reshape its copy freely."""
    with PRESEASON_FIXTURE_PATH.open() as f:
        payload = json.load(f)
    return copy.deepcopy(payload["results"])


def load_preseason_post(post_id: int) -> dict[str, Any]:
    """One of the probe's four posts by id."""
    return next(post for post in load_preseason_posts() if post["id"] == post_id)


def load_legacy_date_post() -> dict[str, Any]:
    """Post 500, the legacy 200-bin closed-lower / open-upper date question, freshly parsed."""
    with LEGACY_DATE_FIXTURE_PATH.open() as f:
        payload = json.load(f)
    return copy.deepcopy(payload["results"][0])


def load_preseason_date_question() -> DateQuestion:
    """Post 651 as the ``DateQuestion`` the framework parses: closed 12-bin, day granularity."""
    return DateQuestion.from_metaculus_api_json(load_preseason_post(DATE_POST_ID))


def load_legacy_date_question() -> DateQuestion:
    """Post 500 as the ``DateQuestion`` the framework parses: 200 bins, closed lower, open upper."""
    return DateQuestion.from_metaculus_api_json(load_legacy_date_post())


def with_prior_forecast(post: dict[str, Any], forecast_values: Sequence[float]) -> dict[str, Any]:
    """A deep copy of ``post`` whose question the bot's own user forecast once, an hour ago.

    The entry follows the platform's ``MyForecastSerializer`` (``questions/serializers/common.py`` in
    the open-source Metaculus backend, which Mantic forks): ``history`` holds every forecast the
    authenticated user made and ``latest`` the standing one, both in the same shape, with unix
    timestamps for the times and ``forecast_values`` as ``[1 - p, p]`` for a binary question, the
    per-option list for multiple choice and the CDF for a continuous one. The three interval fields
    are the platform's quartile summaries of a CDF and stay ``None`` here, as they do on the wire for
    a binary or multiple-choice forecast; the bot reads none of them.
    """
    forecast = copy.deepcopy(post)
    question = forecast["question"]
    entry = {
        "question_id": question["id"],
        "author_id": _FAKE_BOT_USER_ID,
        "start_time": (datetime.now(UTC) - _PRIOR_FORECAST_AGE).timestamp(),
        "end_time": None,
        "forecast_values": list(forecast_values),
        "interval_lower_bounds": None,
        "centers": None,
        "interval_upper_bounds": None,
        "distribution_input": None,
    }
    question["my_forecasts"] = {
        "history": [entry],
        "latest": copy.deepcopy(entry),
        "score_data": {},
    }
    return forecast
