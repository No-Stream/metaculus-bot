import itertools
import logging
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import MagicMock

import pytest
from forecasting_tools import BinaryQuestion, MetaculusQuestion
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot

from main import TemplateForecaster

_POST_IDS = itertools.count(start=7001)


def _supported_question(done: bool = False, closes_in: timedelta | None = None) -> MetaculusQuestion:
    """A stand-in question that survives forecast_questions' unsupported-type gate.

    forecast_questions drops anything that is not Binary/MC/Numeric up front, so a
    bare object no longer reaches the cap/skip logic. A spec'd MagicMock passes
    ``isinstance(q, BinaryQuestion)`` while letting us set ``already_forecasted``
    for the skip filter.

    ``close_time``, ``id_of_post`` and ``page_url`` have to be set explicitly because
    ``spec=BinaryQuestion`` does not expose Pydantic field names as class attributes:
    forecast_questions sorts on the first (tightest close first, so the cap keeps the most urgent
    questions) and names the other two, as the post id and the platform, for every question the
    cap drops.
    """
    q = MagicMock(spec=BinaryQuestion)
    q.already_forecasted = done
    q.close_time = datetime.now(UTC) + (closes_in if closes_in is not None else timedelta(days=1))
    q.id_of_post = next(_POST_IDS)
    q.page_url = f"https://www.metaculus.com/questions/{q.id_of_post}/"
    return cast(MetaculusQuestion, q)


@pytest.mark.asyncio
async def test_cap_applied_after_skip(monkeypatch):
    # Create 15 questions, first 5 already forecasted -> 10 unforecasted remain
    questions = [_supported_question(True) for _ in range(5)] + [_supported_question(False) for _ in range(10)]

    captured = []

    async def stub_forecast_questions(self, questions_arg, return_exceptions=False):
        captured.append(len(questions_arg))
        return [MagicMock() for _ in range(len(questions_arg))]

    # Patch base class method to observe how many questions are forwarded after capping

    monkeypatch.setattr(ForecastBot, "forecast_questions", stub_forecast_questions, raising=True)

    bot = TemplateForecaster(
        llms={
            "default": "mock",
            "summarizer": "mock_sum",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
        },
        max_questions_per_run=10,
    )
    bot.skip_previously_forecasted_questions = True

    results = await bot.forecast_questions(cast(list[MetaculusQuestion], questions))

    assert captured == [10]
    assert len(results) == 10


@pytest.mark.asyncio
async def test_cap_limits_to_10(monkeypatch, caplog):
    # 12 unforecasted questions -> expect 10 due to cap
    questions = [_supported_question() for _ in range(12)]

    captured = []

    async def stub_forecast_questions(self, questions_arg, return_exceptions=False):
        captured.append(len(questions_arg))
        return [MagicMock() for _ in range(len(questions_arg))]

    monkeypatch.setattr(ForecastBot, "forecast_questions", stub_forecast_questions, raising=True)

    bot = TemplateForecaster(
        llms={
            "default": "mock",
            "summarizer": "mock_sum",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
        }
    )  # default cap = 10
    bot.skip_previously_forecasted_questions = False

    with caplog.at_level(logging.WARNING, logger="metaculus_bot.forecaster"):
        results = await bot.forecast_questions(cast(list[MetaculusQuestion], questions))

    assert captured == [10]
    assert len(results) == 10
    # The cap forfeits real questions, so it is a registered WARNING marker that names the posts
    # left behind (the two latest-closing ones: the stubs close in creation order and the sort
    # keeps the soonest-closing ten). Verbatim, since the archive keys off this spelling.
    dropped_ids = [q.id_of_post for q in questions[-2:]]
    forfeit_lines = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.getMessage().startswith("QUESTION_CAP_FORFEIT:")
    ]
    assert forfeit_lines == [
        f"QUESTION_CAP_FORFEIT: platform=metaculus cap=10 total=12 dropped=2 posts={dropped_ids[0]},{dropped_ids[1]}"
    ]


@pytest.mark.asyncio
async def test_no_cap_when_below_limit(monkeypatch):
    # 7 unforecasted questions -> should pass through unchanged
    questions = [_supported_question() for _ in range(7)]

    captured = []

    async def stub_forecast_questions(self, questions_arg, return_exceptions=False):
        captured.append(len(questions_arg))
        return [MagicMock() for _ in range(len(questions_arg))]

    monkeypatch.setattr(ForecastBot, "forecast_questions", stub_forecast_questions, raising=True)

    bot = TemplateForecaster(
        llms={
            "default": "mock",
            "summarizer": "mock_sum",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
        }
    )  # default cap = 10

    results = await bot.forecast_questions(cast(list[MetaculusQuestion], questions))

    assert captured == [7]
    assert len(results) == 7
