import itertools
import logging
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from forecasting_tools import BinaryQuestion, MetaculusQuestion
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot

from main import TemplateForecaster

_POST_IDS = itertools.count(start=7001)

# The authenticated `with_cp=true` list-page shape, observed live 2026-09-09; `history` fills on the first forecast.
FRESH_MY_FORECASTS: dict[str, Any] = {"history": [], "latest": None, "score_data": {}}
PRIOR_MY_FORECASTS: dict[str, Any] = {
    "history": [{"start_time": 1.0, "forecast_values": [0.3, 0.7]}],
    "latest": {"start_time": 1.0},
    "score_data": {},
}

UNREADABLE_MARKER = "SKIP_GUARD_UNREADABLE:"


def _supported_question(
    done: bool = False,
    closes_in: timedelta | None = None,
    api_json: dict[str, Any] | None = None,
) -> MetaculusQuestion:
    """A stand-in question that survives forecast_questions' unsupported-type gate.

    forecast_questions drops anything that is not Binary/MC/Numeric up front, so a
    bare object no longer reaches the cap/skip logic. A spec'd MagicMock passes
    ``isinstance(q, BinaryQuestion)`` while letting us set ``already_forecasted``
    for the skip filter.

    ``close_time``, ``id_of_post``, ``page_url`` and ``api_json`` have to be set explicitly because
    ``spec=BinaryQuestion`` does not expose Pydantic field names as class attributes:
    forecast_questions sorts on the first (tightest close first, so the cap keeps the most urgent
    questions), names the next two, as the post id and the platform, for every question the
    cap drops, and reads ``my_forecasts`` off the last for the skip guard's fail-shut leg. The
    default payload is the readable list-page shape, with a prior forecast when ``done``.
    """
    q = MagicMock(spec=BinaryQuestion)
    q.already_forecasted = done
    q.close_time = datetime.now(UTC) + (closes_in if closes_in is not None else timedelta(days=1))
    q.id_of_post = next(_POST_IDS)
    q.id_of_question = q.id_of_post + 100_000
    q.page_url = f"https://www.metaculus.com/questions/{q.id_of_post}/"
    if api_json is None:
        api_json = {"question": {"my_forecasts": PRIOR_MY_FORECASTS if done else FRESH_MY_FORECASTS}}
    q.api_json = api_json
    return cast(MetaculusQuestion, q)


def _bot(**kwargs) -> TemplateForecaster:
    return TemplateForecaster(
        llms={
            "default": "mock",
            "summarizer": "mock_sum",
            "parser": "mock_parser",
            "researcher": "mock_researcher",
        },
        **kwargs,
    )


def _capture_forwarded(monkeypatch) -> list[list[MetaculusQuestion]]:
    """Patch the base class method to observe which questions are forwarded after filtering."""
    forwarded: list[list[MetaculusQuestion]] = []

    async def stub_forecast_questions(self, questions_arg, return_exceptions=False):
        forwarded.append(list(questions_arg))
        return [MagicMock() for _ in range(len(questions_arg))]

    monkeypatch.setattr(ForecastBot, "forecast_questions", stub_forecast_questions, raising=True)
    return forwarded


@pytest.mark.asyncio
async def test_cap_applied_after_skip(monkeypatch):
    """15 questions, the first 5 already forecast: 10 unforecasted remain and all fit under the cap."""
    questions = [_supported_question(True) for _ in range(5)] + [_supported_question(False) for _ in range(10)]
    forwarded = _capture_forwarded(monkeypatch)

    bot = _bot(max_questions_per_run=10)
    bot.skip_previously_forecasted_questions = True

    results = await bot.forecast_questions(cast(list[MetaculusQuestion], questions))

    assert [len(batch) for batch in forwarded] == [10]
    assert len(results) == 10


@pytest.mark.asyncio
async def test_cap_limits_to_10(monkeypatch, caplog):
    """12 unforecasted questions: the default cap forwards 10 and marks the two it forfeits."""
    questions = [_supported_question() for _ in range(12)]
    forwarded = _capture_forwarded(monkeypatch)

    bot = _bot()  # default cap = 10
    bot.skip_previously_forecasted_questions = False

    with caplog.at_level(logging.WARNING, logger="metaculus_bot.forecaster"):
        results = await bot.forecast_questions(cast(list[MetaculusQuestion], questions))

    assert [len(batch) for batch in forwarded] == [10]
    assert len(results) == 10
    # Verbatim, since the archive keys off this spelling; the stubs close in creation order, so the last two are forfeit.
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
    """7 unforecasted questions pass through the cap unchanged."""
    questions = [_supported_question() for _ in range(7)]
    forwarded = _capture_forwarded(monkeypatch)

    bot = _bot()  # default cap = 10

    results = await bot.forecast_questions(cast(list[MetaculusQuestion], questions))

    assert [len(batch) for batch in forwarded] == [7]
    assert len(results) == 7


class TestSkipGuardFailsShut:
    """The re-spend guard drops a question whose ``my_forecasts`` field it cannot read.

    The framework derives ``already_forecasted`` inside a blanket except that answers False, so a
    payload with no ``my_forecasts`` (a list GET without ``with_cp=true``, an unauthenticated Mantic
    read, an API change) reads as never forecast and every hourly run would re-forecast and
    re-publish the whole tournament. With the guard on, such a question is not eligible at all.
    """

    def _unreadable_marker_lines(self, caplog: pytest.LogCaptureFixture) -> list[str]:
        return [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and r.getMessage().startswith(UNREADABLE_MARKER)
        ]

    @pytest.mark.asyncio
    async def test_a_payload_without_my_forecasts_is_dropped_and_marked(self, monkeypatch, caplog):
        readable = _supported_question()
        unreadable = _supported_question(api_json={"question": {"type": "binary"}})
        forwarded = _capture_forwarded(monkeypatch)
        bot = _bot()
        bot.skip_previously_forecasted_questions = True

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.forecaster"):
            await bot.forecast_questions([readable, unreadable])

        assert forwarded == [[readable]]
        # Verbatim: the archive keys off this spelling (scripts/telemetry/markers.py).
        assert self._unreadable_marker_lines(caplog) == [
            f"SKIP_GUARD_UNREADABLE: question={unreadable.id_of_question} post_id={unreadable.id_of_post} "
            "platform=metaculus reason=my_forecasts_missing"
        ]
        assert "Dropped 1 question(s) with no readable my_forecasts field; the skip guard fails shut" in (
            caplog.messages
        )

    @pytest.mark.asyncio
    async def test_a_null_my_forecasts_is_dropped_too(self, monkeypatch, caplog):
        unreadable = _supported_question(api_json={"question": {"my_forecasts": None}})
        forwarded = _capture_forwarded(monkeypatch)
        bot = _bot()
        bot.skip_previously_forecasted_questions = True

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.forecaster"):
            await bot.forecast_questions([unreadable])

        assert forwarded == [[]]
        assert len(self._unreadable_marker_lines(caplog)) == 1

    @pytest.mark.asyncio
    async def test_a_payload_without_a_question_block_is_dropped_too(self, monkeypatch, caplog):
        unreadable = _supported_question(api_json={})
        forwarded = _capture_forwarded(monkeypatch)
        bot = _bot()
        bot.skip_previously_forecasted_questions = True

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.forecaster"):
            await bot.forecast_questions([unreadable])

        assert forwarded == [[]]
        assert len(self._unreadable_marker_lines(caplog)) == 1

    @pytest.mark.asyncio
    async def test_an_empty_history_stays_eligible(self, monkeypatch, caplog):
        """The steady state on a fresh question: the field is present, nothing forecast yet."""
        fresh = _supported_question(api_json={"question": {"my_forecasts": FRESH_MY_FORECASTS}})
        forwarded = _capture_forwarded(monkeypatch)
        bot = _bot()
        bot.skip_previously_forecasted_questions = True

        with caplog.at_level(logging.INFO, logger="metaculus_bot.forecaster"):
            await bot.forecast_questions([fresh])

        assert forwarded == [[fresh]]
        assert self._unreadable_marker_lines(caplog) == []

    @pytest.mark.asyncio
    async def test_a_prior_forecast_is_still_skipped_as_before(self, monkeypatch, caplog):
        done = _supported_question(done=True)
        fresh = _supported_question()
        forwarded = _capture_forwarded(monkeypatch)
        bot = _bot()
        bot.skip_previously_forecasted_questions = True

        with caplog.at_level(logging.INFO, logger="metaculus_bot.forecaster"):
            await bot.forecast_questions([done, fresh])

        assert forwarded == [[fresh]]
        assert "Skipping 1 previously forecasted questions" in caplog.messages
        assert self._unreadable_marker_lines(caplog) == []

    @pytest.mark.asyncio
    async def test_guard_off_forwards_an_unreadable_payload(self, monkeypatch, caplog):
        """test_questions mode runs with the guard off and re-forecasts on purpose."""
        unreadable = _supported_question(api_json={"question": {}})
        forwarded = _capture_forwarded(monkeypatch)
        bot = _bot()
        bot.skip_previously_forecasted_questions = False

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.forecaster"):
            await bot.forecast_questions([unreadable])

        assert forwarded == [[unreadable]]
        assert self._unreadable_marker_lines(caplog) == []
