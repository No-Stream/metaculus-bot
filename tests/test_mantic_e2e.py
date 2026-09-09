"""End-to-end proof of the Mantic run mode: tournament fetch through publish, HTTP mocked.

This is the ONLY proof of the injected-client path before a paid live run against
competitions.mantic.com, so it drives the real chain rather than a stand-in for it:
``TemplateForecaster.forecast_on_tournament`` -> the framework's ``forecast_questions`` ->
``_run_individual_question`` -> ``report.publish_report_to_metaculus(metaculus_client=...)``, with a
:class:`~metaculus_bot.mantic.ManticClient` injected exactly as ``cli.main`` injects it in mantic
mode. Both class-level hardening patches are installed first, because that is what production
installs in ``cli._configure_process`` and the publish path runs through them.

Only two things are faked. The HTTP transport (``HTTPAdapter.send``) answers the Mantic API shape,
so the real ``requests.Session`` still prepares every request and the headers, query string and body
that would go on the wire are observable; and ``GeneralLlm.invoke`` returns canned rationales
carrying the fenced ```json STRUCTURED FORECAST block the extraction ladder reads, so the forecast
values are deterministic. Research is stubbed at ``run_research``, so no provider runs.

What the run proves, per assertion group below: the four preseason questions are fetched through the
Mantic client with no ``forecast_type`` parameter (the difference that made the framework's default
fetch return zero quantitative questions); all four are forecast, the 12-bin day-granularity date
question included (on its epoch-seconds view, published as a 13-value CDF whose mass lands in the
calendar day's bin, with the comment rendering dates); every prediction and comment POST goes to
the Mantic API with the payload shape that platform validates, including a 451-point CDF the
server's own rules accept; and nothing in the run contacts metaculus.com.

A second run, over the same posts with the bot's own forecast already standing on one of them, proves
the already-forecast skip that the first run cannot: the workflow fires hourly, so once it has run
once that is the state of nearly every question, and a skip that silently failed would re-spend the
whole ensemble every hour. That run must make no forecast POST, no comment POST and no forecaster
call for the forecast question while the others still publish.

The fixture (``tests/data/mantic_preseason2_posts_2026_09_08.json``) is the authenticated
``GET /api/posts/?tournaments=preseason-2`` response from the 2026-09-08 live probe (its path, post
ids and the already-forecast derivation live in ``tests/mantic_fakes.py``). Its questions close
2026-09-20, which becomes the past: the loaded copy is re-dated into the future here, because the
intake time budget skips a question whose close leaves no room and the publish gate skips a question
that has closed, both against the real clock.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import requests
from forecasting_tools import GeneralLlm
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
from forecasting_tools.data_models.numeric_report import DateReport, DiscreteReport
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.helpers import metaculus_client as ft_client
from forecasting_tools.helpers.metaculus_client import MetaculusClient
from requests.adapters import HTTPAdapter
from scipy.stats import norm

from metaculus_bot import fetch_hardening, publish_hardening
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import (
    BINARY_PROB_MAX,
    BINARY_PROB_MIN,
    DONATED_OPENROUTER_KEY_ENABLED_ENV,
    MANTIC_API_BASE_URL,
    MANTIC_OUT_OF_RANGE_TAIL_FLOOR,
    MANTIC_SITE_URL,
    MANTIC_TOKEN_ENV,
    MANTIC_TOURNAMENT_ID,
    MC_PROB_MAX,
    MC_PROB_MIN,
)
from metaculus_bot.mantic import ManticClient
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.time_budget import QuestionTimeBudget
from tests.mantic_fakes import (
    BINARY_POST_ID,
    DATE_POST_ID,
    DISCRETE_POST_ID,
    MULTIPLE_CHOICE_POST_ID,
    load_preseason_posts,
    with_prior_forecast,
)
from tests.pipeline_test_helpers import assert_server_accepts_cdf, make_e2e_bot

pytestmark = pytest.mark.e2e

_FAKE_TOKEN = "m" * 40
_EXPECTED_AUTH = f"Token {_FAKE_TOKEN}"

# The second run: the bot's forecast already stands on the binary question, leaving three fresh ones.
_PRIOR_FORECAST_POST_ID = BINARY_POST_ID
_STILL_FRESH_POST_IDS = frozenset({MULTIPLE_CHOICE_POST_ID, DISCRETE_POST_ID, DATE_POST_ID})

# Derived from the base URL, so the routing paths and the asserted prefixes cannot disagree.
_POSTS_URL = f"{MANTIC_API_BASE_URL}/posts/"
_FORECAST_URL = f"{MANTIC_API_BASE_URL}/questions/forecast/"
_COMMENT_URL = f"{MANTIC_API_BASE_URL}/comments/create/"
_POSTS_PATH = urlparse(_POSTS_URL).path
_FORECAST_PATH = urlparse(_FORECAST_URL).path
_COMMENT_PATH = urlparse(_COMMENT_URL).path

# Far enough ahead that the time budget is the static one and the publish gate a no-op, forever.
_CLOSE_OFFSET = timedelta(days=30)
_OPEN_OFFSET = timedelta(days=5)

_RESEARCH_TEXT = (
    "## Research Summary\n\nStubbed research for the Mantic end-to-end test: no provider ran, and "
    "no forecast in this run depends on the contents of this section.\n"
)

# Per-forecaster declarations; every triple's spread sits below its stacking threshold, so the median publishes.
_BINARY_MEMBER_PROBS: tuple[float, ...] = (0.20, 0.22, 0.25)
_BINARY_MEDIAN = 0.22
# What the platform reports as standing after the previous run published that median: ``[1 - p, p]``.
_BINARY_PRIOR_FORECAST_VALUES = (1.0 - _BINARY_MEDIAN, _BINARY_MEDIAN)
# In the fixture's option order: a hike above 25bp, a 25bp hike, a hold, a cut.
_MC_MEMBER_PROBS: tuple[tuple[float, ...], ...] = (
    (0.05, 0.10, 0.55, 0.30),
    (0.05, 0.12, 0.52, 0.31),
    (0.05, 0.08, 0.58, 0.29),
)
_MC_MODAL_OPTION_INDEX = 2
_BITCOIN_MEMBER_NORMALS: tuple[tuple[float, float], ...] = (
    (78_000.0, 6_000.0),
    (78_800.0, 6_200.0),
    (77_300.0, 5_800.0),
)
_BITCOIN_MEDIAN_RANGE = (77_000.0, 79_000.0)
# Every forecaster is certain the largest move lands on 2026-09-16 and spreads its percentiles
# inside that UTC day, each over a slightly different window of hours. On the platform's grid that
# day is bin 8, ``cdf[9] - cdf[8]`` over (2026-09-16T00:00Z, 2026-09-17T00:00Z].
_DATE_CERTAIN_DAY = datetime(2026, 9, 16, tzinfo=UTC)
_DATE_MEMBER_HOUR_WINDOWS: tuple[tuple[int, int], ...] = ((2, 22), (3, 21), (1, 23))
_DATE_CERTAIN_BIN = 8

_FORECASTS_PER_QUESTION = 3
_FORECAST_QUESTION_COUNT = 4


# ---------------------------------------------------------------------------
# The fixture, re-dated into the future
# ---------------------------------------------------------------------------


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _future_dated_posts() -> list[dict[str, Any]]:
    """The probe's four posts with their open and close timestamps moved around today.

    Both levels are rewritten: the framework reads the close and resolve times off the QUESTION
    json, while the post-level copies are what a reader of this fixture would compare against.
    """
    posts = load_preseason_posts()
    now = datetime.now(UTC)
    close_iso = _iso(now + _CLOSE_OFFSET)
    open_iso = _iso(now - _OPEN_OFFSET)
    for post in posts:
        for holder in (post, post["question"]):
            for key in ("scheduled_close_time", "scheduled_resolve_time", "cp_reveal_time"):
                if key in holder:
                    holder[key] = close_iso
            if "open_time" in holder:
                holder["open_time"] = open_iso
    return posts


# ---------------------------------------------------------------------------
# Canned forecaster rationales
# ---------------------------------------------------------------------------


def _percentile_key(percentile: float) -> str:
    return f"{percentile:g}"


def _structured_block(payload: dict[str, Any]) -> str:
    return f"```json\n{json.dumps(payload)}\n```\n"


def _binary_reasoning(posterior_prob: float) -> str:
    return (
        "## Analysis\n\nSeptember is the climatological peak of the Atlantic season, and the "
        "current basin state is the dominant term.\n\n"
        f"{_structured_block({'question_type': 'binary', 'posterior_prob': posterior_prob})}"
    )


def _mc_reasoning(options: Sequence[str], probs: Sequence[float]) -> str:
    option_probs = dict(zip(options, probs, strict=True))
    return (
        "## Analysis\n\nThe futures-implied path and the last statement both point at a hold, with "
        "a cut as the live alternative.\n\n"
        f"{_structured_block({'question_type': 'multiple_choice', 'option_probs': option_probs})}"
    )


def _numeric_reasoning(mean: float, sd: float) -> str:
    percentiles = {_percentile_key(p): round(float(norm.ppf(p, loc=mean, scale=sd)), 2) for p in STANDARD_PERCENTILES}
    payload = {
        "question_type": "numeric",
        "declared_percentiles": percentiles,
        "outcome_type": "continuous",
    }
    return (
        "## Analysis\n\nSpot, realized volatility over the window, and the eleven daily "
        "resolutions this question averages.\n\n"
        f"{_structured_block(payload)}"
    )


def _date_reasoning(start_hour: int, end_hour: int) -> str:
    """A forecaster certain of 2026-09-16, its 13 percentiles spread over that day's given hours.

    The block is the ``DateStructured`` shape: ISO-8601 strings keyed by percentile, no
    ``outcome_type``. Timestamps rather than bare dates so the three members differ.
    """
    start = _DATE_CERTAIN_DAY + timedelta(hours=start_hour)
    span = timedelta(hours=end_hour - start_hour)
    count = len(STANDARD_PERCENTILES)
    percentiles = {
        _percentile_key(p): _iso(start + span * index / (count - 1)) for index, p in enumerate(STANDARD_PERCENTILES)
    }
    payload = {"question_type": "date", "declared_percentiles": percentiles}
    return (
        "## Analysis\n\nRealised volatility clusters around the CPI print and the FOMC decision; the "
        "session after the FOMC statement is the modal largest move.\n\n"
        f"{_structured_block(payload)}"
    )


def _canned_responses(posts: list[dict[str, Any]]) -> dict[str, list[str]]:
    """One rationale per forecaster per question, keyed by the question title the prompt carries."""
    questions = {post["id"]: post["question"] for post in posts}
    mc_options: list[str] = questions[MULTIPLE_CHOICE_POST_ID]["options"]
    return {
        questions[BINARY_POST_ID]["title"]: [_binary_reasoning(prob) for prob in _BINARY_MEMBER_PROBS],
        questions[MULTIPLE_CHOICE_POST_ID]["title"]: [_mc_reasoning(mc_options, probs) for probs in _MC_MEMBER_PROBS],
        questions[DISCRETE_POST_ID]["title"]: [_numeric_reasoning(mean, sd) for mean, sd in _BITCOIN_MEMBER_NORMALS],
        questions[DATE_POST_ID]["title"]: [_date_reasoning(start, end) for start, end in _DATE_MEMBER_HOUR_WINDOWS],
    }


@dataclass
class _LlmCallLog:
    """Which canned rationale each ``GeneralLlm.invoke`` served, and anything unrouted."""

    forecaster_calls: list[str] = field(default_factory=list)
    unexpected_prompts: list[str] = field(default_factory=list)


def _install_llm_stub(mp: pytest.MonkeyPatch, responses_by_title: dict[str, list[str]]) -> _LlmCallLog:
    """Serve the canned rationales off ``GeneralLlm.invoke``, and refuse any other LLM call.

    Routing is by question title, so an invocation that is NOT a base-forecaster prompt (a parser
    fallback because the ladder's block rung failed, or a stacker call the spread should have
    skipped) has nowhere to go and raises instead of being quietly answered.
    """
    log = _LlmCallLog()
    served: dict[str, int] = dict.fromkeys(responses_by_title, 0)

    async def invoke(self: GeneralLlm, prompt: Any, system_prompt: str | None = None) -> str:
        text = prompt if isinstance(prompt, str) else str(prompt)
        for title, responses in responses_by_title.items():
            if title in text:
                index = served[title]
                served[title] = index + 1
                log.forecaster_calls.append(title)
                return responses[index % len(responses)]
        log.unexpected_prompts.append(text[:400])
        raise AssertionError("unrouted LLM call: not a base-forecaster prompt for any preseason question")

    mp.setattr(GeneralLlm, "invoke", invoke)
    return log


# ---------------------------------------------------------------------------
# The fake Mantic transport
# ---------------------------------------------------------------------------

# Mantic's ``forecast_type`` vocabulary, honoured by the fake so asserting the parameter's absence has teeth.
_MANTIC_FILTER_TYPE = {
    "binary": "binary",
    "multiple_choice": "multiple_choice",
    "date": "date",
    "discrete": "quantitative",
    "quantitative": "quantitative",
}


@dataclass(frozen=True)
class _RecordedRequest:
    method: str
    url: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: Any
    send_kwargs: dict[str, Any]


def _json_response(request: requests.PreparedRequest, status: int, payload: Any) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload).encode()
    response.encoding = "utf-8"
    response.url = request.url or ""
    response.request = request
    return response


def _visible_posts(posts: list[dict[str, Any]], query: dict[str, list[str]]) -> list[dict[str, Any]]:
    """The page of ``posts`` a real list endpoint would answer: type-filtered, then ``offset``/``limit``-sliced.

    The Mantic client walks offsets until an EMPTY page (its ceiling makes the framework keep
    paging, since Mantic's ``next`` link is unreliable), so the fake must run out of posts the
    way the platform does or the framework reads the same four posts twice and refuses them as
    duplicates.
    """
    requested_types = query.get("forecast_type")
    if requested_types:
        posts = [post for post in posts if _MANTIC_FILTER_TYPE[post["question"]["type"]] in requested_types]
    offset = int(query.get("offset", ["0"])[0])
    limit = int(query.get("limit", [str(len(posts))])[0])
    return posts[offset : offset + limit]


def _install_fake_transport(mp: pytest.MonkeyPatch, posts: list[dict[str, Any]]) -> list[_RecordedRequest]:
    """Answer the three Mantic endpoints at the transport, recording every prepared request."""
    recorded: list[_RecordedRequest] = []

    def fake_send(self: HTTPAdapter, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        parsed = urlparse(request.url or "")
        query = parse_qs(parsed.query)
        body = json.loads(request.body) if request.body else None
        recorded.append(
            _RecordedRequest(
                method=request.method or "",
                url=request.url or "",
                path=parsed.path,
                query=query,
                headers=dict(request.headers),
                body=body,
                send_kwargs=dict(kwargs),
            )
        )
        if request.method == "GET" and parsed.path == _POSTS_PATH:
            results = _visible_posts(posts, query)
            return _json_response(request, 200, {"next": None, "previous": None, "results": results})
        if request.method == "POST" and parsed.path == _FORECAST_PATH:
            return _json_response(request, 201, [])
        if request.method == "POST" and parsed.path == _COMMENT_PATH:
            return _json_response(request, 201, {})
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    mp.setattr(HTTPAdapter, "send", fake_send)
    return recorded


# ---------------------------------------------------------------------------
# Process-global state: hardening and log capture
# ---------------------------------------------------------------------------


def _apply_hardening_with_restore(mp: pytest.MonkeyPatch) -> None:
    """Install both hardening patch sets the way ``cli._configure_process`` does, reversibly.

    Every patched name is snapshotted through monkeypatch first (the idiom in
    tests/test_ft_upgrade_seams.py), because these mutate the framework's classes and its
    ``requests`` module for the whole process: left installed, they would change what the seam
    tests' own negative controls observe. Both sentinels are cleared so this run always exercises
    a fresh install rather than silently inheriting an earlier test's.
    """
    for method_name in (*publish_hardening._PATCHED_METHODS, *fetch_hardening._PATCHED_METHODS):
        mp.setattr(MetaculusClient, method_name, MetaculusClient.__dict__[method_name])
    for report_type in publish_hardening._PATCHED_REPORT_TYPES:
        publish_method = publish_hardening._PUBLISH_METHOD
        mp.setattr(report_type, publish_method, report_type.__dict__[publish_method])
    # Setting each sentinel before deleting it is what makes the deletion reversible.
    for sentinel in (publish_hardening._SENTINEL, fetch_hardening._SENTINEL):
        mp.setattr(MetaculusClient, sentinel, False, raising=False)
        delattr(MetaculusClient, sentinel)
    # Both hardening layers replace the module-level requests.post / requests.get permanently.
    mp.setattr(ft_client.requests, "post", ft_client.requests.post)
    mp.setattr(ft_client.requests, "get", ft_client.requests.get)

    publish_hardening.apply_publish_hardening()
    fetch_hardening.apply_fetch_hardening()


class _RecordingHandler(logging.Handler):
    def __init__(self, records: list[logging.LogRecord]) -> None:
        super().__init__(level=logging.DEBUG)
        self._records = records

    def emit(self, record: logging.LogRecord) -> None:
        self._records.append(record)


@contextmanager
def _captured_bot_logs() -> Iterator[list[logging.LogRecord]]:
    """Capture ``metaculus_bot`` records for the whole run (caplog is function-scoped)."""
    logger = logging.getLogger("metaculus_bot")
    records: list[logging.LogRecord] = []
    handler = _RecordingHandler(records)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ManticRun:
    reports: Sequence[ForecastReport | BaseException]
    requests: list[_RecordedRequest]
    llm_calls: _LlmCallLog
    records: list[logging.LogRecord]
    posts: list[dict[str, Any]]

    @property
    def log_text(self) -> str:
        return "\n".join(record.getMessage() for record in self.records)

    def posts_requests(self) -> list[_RecordedRequest]:
        return [r for r in self.requests if r.path == _POSTS_PATH]

    def forecast_posts(self) -> list[_RecordedRequest]:
        return [r for r in self.requests if r.path == _FORECAST_PATH]

    def comment_posts(self) -> list[_RecordedRequest]:
        return [r for r in self.requests if r.path == _COMMENT_PATH]

    def posted_question_ids(self) -> set[int]:
        return {r.body[0]["question"] for r in self.forecast_posts()}

    def commented_post_ids(self) -> set[int]:
        return {r.body["on_post"] for r in self.comment_posts()}

    def mantic_question_marker_lines(self) -> list[str]:
        return self.marker_lines("MANTIC_QUESTION:")

    def marker_lines(self, prefix: str) -> list[str]:
        return [line for line in self.log_text.splitlines() if line.startswith(prefix)]

    def forecast_payload(self, question_id: int) -> dict[str, Any]:
        payloads = [r.body[0] for r in self.forecast_posts() if r.body[0]["question"] == question_id]
        assert len(payloads) == 1, f"expected exactly one forecast POST for question {question_id}, got {payloads}"
        return payloads[0]

    def comment_payload(self, post_id: int) -> dict[str, Any]:
        payloads = [r.body for r in self.comment_posts() if r.body["on_post"] == post_id]
        assert len(payloads) == 1, f"expected exactly one comment POST for post {post_id}, got {len(payloads)}"
        return payloads[0]

    def report_for(self, post_id: int) -> ForecastReport:
        matches = [r for r in self.reports if isinstance(r, ForecastReport) and r.question.id_of_post == post_id]
        assert len(matches) == 1, f"expected one report for post {post_id}, got {matches}"
        return matches[0]


async def _stub_research(question: MetaculusQuestion, time_budget: QuestionTimeBudget | None = None) -> str:
    """Stand in for ``TemplateForecaster.run_research`` with its real signature: no provider runs."""
    return _RESEARCH_TEXT


def _run_mantic_mode(posts: list[dict[str, Any]]) -> Iterator[_ManticRun]:
    """One full mantic-mode run over ``posts``; the module-scoped fixtures below each wrap one.

    Synchronous so each fixture runs the pipeline a single time (and installs and removes the
    process-global patches once). The framework's tournament fetch calls ``asyncio.run`` inside the
    running loop, which works because forecasting-tools applies nest_asyncio at import — the same
    nesting production relies on.

    The bot is built in ``cli.main``'s shape for a mantic run, with two deliberate notes:
    ``is_benchmarking=False`` keeps the publish path real, and ``min_forecasters_to_publish`` is the
    full roster (production's floor is 1) so a forecaster lost to a stub defect fails the test.
    """
    with pytest.MonkeyPatch.context() as mp:
        # The environment a Mantic run requires, though the client below is handed its token.
        mp.setenv(DONATED_OPENROUTER_KEY_ENABLED_ENV, "false")
        mp.setenv(MANTIC_TOKEN_ENV, _FAKE_TOKEN)
        # Every framework request sleeps 3.5-4.5s first; seven requests would be half a minute.
        mp.setattr(MetaculusClient, "_sleep_between_requests", lambda self: None)

        _apply_hardening_with_restore(mp)
        recorded = _install_fake_transport(mp, posts)
        llm_calls = _install_llm_stub(mp, _canned_responses(posts))

        bot = make_e2e_bot(
            AggregationStrategy.CONDITIONAL_STACKING,
            n_forecasters=_FORECASTS_PER_QUESTION,
            publish_reports_to_metaculus=True,
            is_benchmarking=False,
            skip_previously_forecasted_questions=True,
            min_forecasters_to_publish=_FORECASTS_PER_QUESTION,
            metaculus_client=ManticClient(token=_FAKE_TOKEN),
        )
        mp.setattr(bot, "run_research", _stub_research)

        with _captured_bot_logs() as records:
            reports = asyncio.run(bot.forecast_on_tournament(MANTIC_TOURNAMENT_ID, return_exceptions=True))

        yield _ManticRun(
            reports=list(reports),
            requests=recorded,
            llm_calls=llm_calls,
            records=records,
            posts=posts,
        )


@pytest.fixture(scope="module")
def mantic_run() -> Iterator[_ManticRun]:
    """The preseason's first run: every question fresh, every supported one forecast and published."""
    yield from _run_mantic_mode(_future_dated_posts())


@pytest.fixture(scope="module")
def mantic_run_after_prior_forecast() -> Iterator[_ManticRun]:
    """The hourly workflow's steady state: the bot's own forecast already stands on one question.

    The binary question carries a prior forecast of this run's own median, the way the platform
    reports it once the previous scheduled run has published; the other three posts are the same
    re-dated copies the first run sees.
    """
    posts = [
        with_prior_forecast(post, forecast_values=_BINARY_PRIOR_FORECAST_VALUES)
        if post["id"] == _PRIOR_FORECAST_POST_ID
        else post
        for post in _future_dated_posts()
    ]
    yield from _run_mantic_mode(posts)


@pytest.fixture
def mantic_posts_transport(monkeypatch: pytest.MonkeyPatch) -> list[_RecordedRequest]:
    """The same fake transport, for the fetch-only comparison between the two clients."""
    monkeypatch.setattr(MetaculusClient, "_sleep_between_requests", lambda self: None)
    return _install_fake_transport(monkeypatch, _future_dated_posts())


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


class TestTheFetchGoesThroughTheManticClient:
    def test_the_fetch_walks_pages_until_an_empty_one(self, mantic_run: _ManticRun) -> None:
        """Two question-list GETs: the first page carries all four posts, the second is empty and
        stops the walk. The client asks for a ceiling rather than trusting Mantic's ``next`` link."""
        gets = mantic_run.posts_requests()
        assert len(gets) == 2, f"expected a full page then an empty one, got {[r.url for r in gets]}"
        for recorded in gets:
            assert recorded.url.startswith(_POSTS_URL)
        first_offset = int(gets[0].query["offset"][0])
        second_offset = int(gets[1].query["offset"][0])
        assert first_offset == 0
        assert second_offset == int(gets[0].query["limit"][0])

    def test_the_get_asks_for_the_open_questions_of_the_preseason_tournament(self, mantic_run: _ManticRun) -> None:
        query = mantic_run.posts_requests()[0].query
        assert query["tournaments"] == [MANTIC_TOURNAMENT_ID]
        assert query["statuses"] == ["open"]

    def test_the_get_omits_the_forecast_type_parameter(self, mantic_run: _ManticRun) -> None:
        """Mantic's vocabulary for this filter is ``quantitative``, so the framework's default value
        loses the 450-bin question entirely."""
        assert "forecast_type" not in mantic_run.posts_requests()[0].query

    def test_every_request_carries_the_mantic_bot_token(self, mantic_run: _ManticRun) -> None:
        for recorded in mantic_run.requests:
            assert recorded.headers.get("Authorization") == _EXPECTED_AUTH, (
                f"{recorded.method} {recorded.url} carried {recorded.headers.get('Authorization')!r}"
            )

    def test_all_four_preseason_questions_were_parsed(self, mantic_run: _ManticRun) -> None:
        """One MANTIC_QUESTION marker per parsed question, so this counts what the client returned
        rather than what the bot went on to forecast."""
        marker_lines = mantic_run.mantic_question_marker_lines()
        assert len(marker_lines) == len(mantic_run.posts)
        for post_id in (BINARY_POST_ID, MULTIPLE_CHOICE_POST_ID, DISCRETE_POST_ID, DATE_POST_ID):
            assert any(f"post={post_id} " in line for line in marker_lines)


class TestTypeRouting:
    def test_four_questions_produced_reports_one_per_type(self, mantic_run: _ManticRun) -> None:
        assert not [r for r in mantic_run.reports if isinstance(r, BaseException)], mantic_run.reports
        assert len(mantic_run.reports) == _FORECAST_QUESTION_COUNT
        assert isinstance(mantic_run.report_for(BINARY_POST_ID), BinaryReport)
        assert isinstance(mantic_run.report_for(MULTIPLE_CHOICE_POST_ID), MultipleChoiceReport)
        assert isinstance(mantic_run.report_for(DISCRETE_POST_ID), DiscreteReport)
        assert isinstance(mantic_run.report_for(DATE_POST_ID), DateReport)

    def test_no_question_was_dropped_as_unsupported(self, mantic_run: _ManticRun) -> None:
        """Until 2026-09-08 the date question was dropped here with a WARNING; Mantic's pool is 41%
        date questions, so the guard now admits it and only a conditional question would trip it."""
        warnings = [
            record.getMessage()
            for record in mantic_run.records
            if record.levelno >= logging.WARNING and "unsupported" in record.getMessage()
        ]
        assert warnings == []

    def test_the_date_report_is_published_off_the_epoch_axis(self, mantic_run: _ManticRun) -> None:
        report = mantic_run.report_for(DATE_POST_ID)
        assert isinstance(report, DateReport)
        assert report.prediction.is_date is True
        assert report.prediction.lower_bound == report.question.lower_bound.timestamp()
        assert report.prediction.upper_bound == report.question.upper_bound.timestamp()


class TestThePreviouslyForecastQuestionIsSkipped:
    """The hourly workflow's steady state, and the direction with power.

    With every fixture history empty, ``already_forecasted`` is False whether ``my_forecasts`` is
    read correctly or is missing entirely (the framework derives it inside a blanket except that
    answers False), so the first run cannot tell a working skip from a no-op one. Only a question
    that already carries the bot's forecast can, and after the first scheduled run of the preseason
    window that is nearly every question on every run: a skip that fails silently re-spends the
    whole ensemble, hourly, on questions the bot has already answered.
    """

    def test_the_skip_is_logged(self, mantic_run_after_prior_forecast: _ManticRun) -> None:
        assert "Skipping 1 previously forecasted questions" in mantic_run_after_prior_forecast.log_text

    def test_nothing_is_posted_for_the_forecast_question(self, mantic_run_after_prior_forecast: _ManticRun) -> None:
        assert _PRIOR_FORECAST_POST_ID not in mantic_run_after_prior_forecast.posted_question_ids()
        assert _PRIOR_FORECAST_POST_ID not in mantic_run_after_prior_forecast.commented_post_ids()

    def test_no_forecaster_ran_on_it(self, mantic_run_after_prior_forecast: _ManticRun) -> None:
        """The skip must happen before the fan-out: no LLM spend, not just no publish."""
        run = mantic_run_after_prior_forecast
        skipped_title = next(post["question"]["title"] for post in run.posts if post["id"] == _PRIOR_FORECAST_POST_ID)
        assert skipped_title not in run.llm_calls.forecaster_calls
        assert len(run.llm_calls.forecaster_calls) == _FORECASTS_PER_QUESTION * len(_STILL_FRESH_POST_IDS)
        assert run.llm_calls.unexpected_prompts == []

    def test_the_other_questions_still_publish(self, mantic_run_after_prior_forecast: _ManticRun) -> None:
        run = mantic_run_after_prior_forecast
        assert not [r for r in run.reports if isinstance(r, BaseException)], run.reports
        reported = {r.question.id_of_post for r in run.reports if isinstance(r, ForecastReport)}
        assert reported == _STILL_FRESH_POST_IDS
        assert run.posted_question_ids() == _STILL_FRESH_POST_IDS
        assert run.commented_post_ids() == _STILL_FRESH_POST_IDS

    def test_the_skipped_question_was_fetched_and_parsed(self, mantic_run_after_prior_forecast: _ManticRun) -> None:
        """It is the bot's filter that drops it, downstream of a client that parsed it: this run
        still sees all four posts, so the skip cannot be a fetch that quietly lost one."""
        run = mantic_run_after_prior_forecast
        marker_lines = run.mantic_question_marker_lines()
        assert len(marker_lines) == len(run.posts)
        assert any(f"post={_PRIOR_FORECAST_POST_ID} " in line for line in marker_lines)


class TestPublishedPayloads:
    def test_one_forecast_and_one_comment_post_per_forecast_question(self, mantic_run: _ManticRun) -> None:
        assert len(mantic_run.forecast_posts()) == _FORECAST_QUESTION_COUNT
        assert len(mantic_run.comment_posts()) == _FORECAST_QUESTION_COUNT
        for recorded in mantic_run.forecast_posts():
            assert isinstance(recorded.body, list)
            assert len(recorded.body) == 1, "the platform takes a one-element list per forecast POST"
            assert recorded.body[0]["source"] == "api"

    def test_binary_payload_is_the_median_probability(self, mantic_run: _ManticRun) -> None:
        payload = mantic_run.forecast_payload(BINARY_POST_ID)
        probability = payload["probability_yes"]
        assert probability == pytest.approx(_BINARY_MEDIAN)
        assert BINARY_PROB_MIN <= probability <= BINARY_PROB_MAX
        # The platform's own floor and ceiling on a submitted probability.
        assert 0.001 <= probability <= 0.999

    def test_multiple_choice_payload_covers_every_option_and_sums_to_one(self, mantic_run: _ManticRun) -> None:
        options: list[str] = next(
            post["question"]["options"] for post in mantic_run.posts if post["id"] == MULTIPLE_CHOICE_POST_ID
        )
        payload = mantic_run.forecast_payload(MULTIPLE_CHOICE_POST_ID)
        per_category: dict[str, float] = payload["probability_yes_per_category"]
        assert list(per_category) == options
        assert sum(per_category.values()) == pytest.approx(1.0, abs=1e-6)
        for probability in per_category.values():
            assert MC_PROB_MIN <= probability <= MC_PROB_MAX
            assert 0.001 <= probability <= 0.999
        # The aggregate keeps the ensemble's modal option rather than flattening it.
        assert max(per_category, key=lambda option: per_category[option]) == options[_MC_MODAL_OPTION_INDEX]

    def test_discrete_payload_is_a_451_point_cdf_the_server_accepts(self, mantic_run: _ManticRun) -> None:
        question_json = next(post["question"] for post in mantic_run.posts if post["id"] == DISCRETE_POST_ID)
        scaling = question_json["scaling"]
        cdf_size = scaling["inbound_outcome_count"] + 1
        payload = mantic_run.forecast_payload(DISCRETE_POST_ID)
        cdf = payload["continuous_cdf"]

        assert len(cdf) == cdf_size
        probs = np.asarray(cdf, dtype=float)
        assert np.all(np.diff(probs) >= 0.0), "the CDF must be non-decreasing"
        assert_server_accepts_cdf(
            probs,
            cdf_size=cdf_size,
            open_lower=scaling["open_lower_bound"],
            open_upper=scaling["open_upper_bound"],
        )

    def test_the_discrete_cdf_is_centred_where_the_ensemble_put_it(self, mantic_run: _ManticRun) -> None:
        question_json = next(post["question"] for post in mantic_run.posts if post["id"] == DISCRETE_POST_ID)
        scaling = question_json["scaling"]
        cdf = np.asarray(mantic_run.forecast_payload(DISCRETE_POST_ID)["continuous_cdf"], dtype=float)
        grid = np.linspace(scaling["range_min"], scaling["range_max"], len(cdf))
        median_value = float(grid[int(np.searchsorted(cdf, 0.5))])
        low, high = _BITCOIN_MEDIAN_RANGE
        assert low <= median_value <= high, f"published median {median_value} outside {_BITCOIN_MEDIAN_RANGE}"

    def test_date_payload_is_a_13_point_cdf_the_server_accepts(self, mantic_run: _ManticRun) -> None:
        """Both of post 651's bounds are closed, so the CDF is pinned to exactly 0.0 and 1.0 at its
        ends, and its 12 bins must each clear the server's ``0.01 / 12`` minimum step."""
        question_json = next(post["question"] for post in mantic_run.posts if post["id"] == DATE_POST_ID)
        scaling = question_json["scaling"]
        cdf_size = scaling["inbound_outcome_count"] + 1
        assert cdf_size == 13
        cdf = mantic_run.forecast_payload(DATE_POST_ID)["continuous_cdf"]

        assert len(cdf) == cdf_size
        probs = np.asarray(cdf, dtype=float)
        assert probs[0] == 0.0
        assert probs[-1] == 1.0
        assert np.all(np.diff(probs) >= 0.0), "the CDF must be non-decreasing"
        assert_server_accepts_cdf(
            probs,
            cdf_size=cdf_size,
            open_lower=scaling["open_lower_bound"],
            open_upper=scaling["open_upper_bound"],
        )

    def test_a_forecaster_certain_of_september_16_puts_the_mass_in_bin_8(self, mantic_run: _ManticRun) -> None:
        """The oracle for the bin convention: calendar day D = range_min + k days is bin k, whose mass
        is ``cdf[k + 1] - cdf[k]`` over ``(edge_k, edge_{k+1}]``, and 2026-09-16 is k = 8. Every member
        put its whole day inside that bin, so the published median must too."""
        question_json = next(post["question"] for post in mantic_run.posts if post["id"] == DATE_POST_ID)
        edges = question_json["scaling"]["continuous_range"]
        assert edges[_DATE_CERTAIN_BIN] == _iso(_DATE_CERTAIN_DAY)
        assert edges[_DATE_CERTAIN_BIN + 1] == _iso(_DATE_CERTAIN_DAY + timedelta(days=1))

        probs = np.asarray(mantic_run.forecast_payload(DATE_POST_ID)["continuous_cdf"], dtype=float)
        bin_masses = np.diff(probs)
        assert int(np.argmax(bin_masses)) == _DATE_CERTAIN_BIN
        assert bin_masses[_DATE_CERTAIN_BIN] > 0.9
        # The other eleven bins hold only the server's minimum step each.
        others = np.delete(bin_masses, _DATE_CERTAIN_BIN)
        assert np.all(others < 0.01)

    def test_the_date_comment_renders_dates_not_epoch_seconds(self, mantic_run: _ManticRun) -> None:
        text: str = mantic_run.comment_payload(DATE_POST_ID)["text"]
        assert "2026-09-16" in text
        # An epoch second in 2026 is a ten-digit number starting 17; none may reach the reader.
        assert re.search(r"\b17\d{8}(?:\.\d+)?\b", text) is None, text[:2000]

    @pytest.mark.parametrize("post_id", [BINARY_POST_ID, MULTIPLE_CHOICE_POST_ID, DISCRETE_POST_ID, DATE_POST_ID])
    def test_each_comment_targets_its_own_post(self, mantic_run: _ManticRun, post_id: int) -> None:
        payload = mantic_run.comment_payload(post_id)
        assert payload["on_post"] == post_id
        # The framework's comment body opens with the SUMMARY heading (after one leading newline).
        assert payload["text"].lstrip().startswith("# SUMMARY")
        assert payload["is_private"] is True
        assert payload["included_forecast"] is True


class TestDateTelemetry:
    """The date question is counted as its own type in the run log, on the epoch axis."""

    def test_each_member_leaves_a_member_forecast_line_with_qtype_date(self, mantic_run: _ManticRun) -> None:
        lines = [line for line in mantic_run.marker_lines("MEMBER_FORECAST:") if f"question={DATE_POST_ID} " in line]
        assert len(lines) == _FORECASTS_PER_QUESTION
        for line in lines:
            assert " role=member qtype=date " in line
            # Both bounds closed: no mass outside the range, and the fields close the line.
            assert line.endswith(" oor_low=0.000000 oor_high=0.000000")
            published = json.loads(line.split(" published=", 1)[1].split(" ", 1)[0])
            assert len(published) == len(STANDARD_PERCENTILES)
            # The values are epoch seconds inside 2026-09-16 UTC.
            day_start = _DATE_CERTAIN_DAY.timestamp()
            assert all(day_start < value < day_start + 86_400 for _, value in published)

    def test_the_aggregate_marker_names_the_date_grid(self, mantic_run: _ManticRun) -> None:
        # Both of post 651's bounds are closed, so the tail floor has nothing to move: raw and
        # published tails are zero and the floor reads zero.
        lines = [line for line in mantic_run.marker_lines("NUMERIC_AGGREGATE:") if f"question={DATE_POST_ID} " in line]
        assert lines == [
            f"NUMERIC_AGGREGATE: question={DATE_POST_ID} qtype=date cdf_size=13 oor_low=0.000000 oor_high=0.000000 "
            "oor_low_raw=0.000000 oor_high_raw=0.000000 tail_floor=0.000000"
        ]

    def test_the_discrete_question_has_its_aggregate_marker_too(self, mantic_run: _ManticRun) -> None:
        lines = [
            line for line in mantic_run.marker_lines("NUMERIC_AGGREGATE:") if f"question={DISCRETE_POST_ID} " in line
        ]
        assert len(lines) == 1
        assert " qtype=numeric cdf_size=451 " in lines[0]

    def test_the_mantic_question_marker_still_names_the_date_question(self, mantic_run: _ManticRun) -> None:
        (line,) = [line for line in mantic_run.mantic_question_marker_lines() if f"post={DATE_POST_ID} " in line]
        assert "type=date" in line
        assert "cdf_size=13" in line
        assert "date_granularity=day" in line


class TestTheOutOfRangeTailFloor:
    """The published discrete aggregate carries the Mantic tail floor beyond each open bound.

    Post 650 has both bounds open and every member's thirteen percentiles inside the range, so
    the aggregate's own tails are the structural 1% and the floor raises each to
    ``MANTIC_OUT_OF_RANGE_TAIL_FLOOR``. The payload the server receives is what is checked, so
    the floor is proven on the wire and not only at the seam.
    """

    def test_the_published_discrete_cdf_carries_the_floor_beyond_each_open_bound(self, mantic_run: _ManticRun) -> None:
        question_json = next(post["question"] for post in mantic_run.posts if post["id"] == DISCRETE_POST_ID)
        scaling = question_json["scaling"]
        assert scaling["open_lower_bound"] is True
        assert scaling["open_upper_bound"] is True
        probs = np.asarray(mantic_run.forecast_payload(DISCRETE_POST_ID)["continuous_cdf"], dtype=float)
        assert probs[0] == MANTIC_OUT_OF_RANGE_TAIL_FLOOR
        assert 1.0 - probs[-1] == pytest.approx(MANTIC_OUT_OF_RANGE_TAIL_FLOOR)

    def test_the_discrete_aggregate_marker_records_raw_and_published_tails(self, mantic_run: _ManticRun) -> None:
        lines = [
            line for line in mantic_run.marker_lines("NUMERIC_AGGREGATE:") if f"question={DISCRETE_POST_ID} " in line
        ]
        assert lines == [
            f"NUMERIC_AGGREGATE: question={DISCRETE_POST_ID} qtype=numeric cdf_size=451 oor_low=0.050000 "
            "oor_high=0.050000 oor_low_raw=0.010000 oor_high_raw=0.010000 tail_floor=0.050000"
        ]

    def test_the_members_keep_their_own_tails(self, mantic_run: _ManticRun) -> None:
        lines = [
            line for line in mantic_run.marker_lines("MEMBER_FORECAST:") if f"question={DISCRETE_POST_ID} " in line
        ]
        assert len(lines) == _FORECASTS_PER_QUESTION
        for line in lines:
            assert line.endswith(" oor_low=0.010000 oor_high=0.010000")


class TestNothingReachesMetaculus:
    def test_every_request_went_to_the_mantic_api(self, mantic_run: _ManticRun) -> None:
        assert mantic_run.requests, "the run made no requests at all"
        for recorded in mantic_run.requests:
            assert recorded.url.startswith(f"{MANTIC_API_BASE_URL}/"), recorded.url

    def test_no_request_mentions_metaculus(self, mantic_run: _ManticRun) -> None:
        for recorded in mantic_run.requests:
            assert "metaculus.com" not in recorded.url, recorded.url

    def test_the_reports_carry_mantic_page_urls(self, mantic_run: _ManticRun) -> None:
        for post_id in (BINARY_POST_ID, MULTIPLE_CHOICE_POST_ID, DISCRETE_POST_ID, DATE_POST_ID):
            page_url = mantic_run.report_for(post_id).question.page_url
            assert page_url is not None
            assert page_url.startswith(f"{MANTIC_SITE_URL}/questions/"), page_url


class TestPublishHardeningIsOnThePath:
    def test_every_publish_post_carries_the_forced_timeout(self, mantic_run: _ManticRun) -> None:
        publish_requests = mantic_run.forecast_posts() + mantic_run.comment_posts()
        assert len(publish_requests) == 2 * _FORECAST_QUESTION_COUNT
        for recorded in publish_requests:
            assert recorded.send_kwargs.get("timeout") == publish_hardening.PUBLISH_POST_TIMEOUT, (
                f"{recorded.path} carried timeout={recorded.send_kwargs.get('timeout')!r}"
            )

    def test_the_question_list_get_is_bounded_too(self, mantic_run: _ManticRun) -> None:
        assert mantic_run.posts_requests()[0].send_kwargs.get("timeout") is not None


class TestTheEnsembleFannedOut:
    def test_three_forecasters_ran_on_each_question(self, mantic_run: _ManticRun) -> None:
        calls = mantic_run.llm_calls.forecaster_calls
        assert len(calls) == _FORECASTS_PER_QUESTION * _FORECAST_QUESTION_COUNT
        assert {calls.count(title) for title in set(calls)} == {_FORECASTS_PER_QUESTION}

    def test_no_parser_or_stacker_call_was_needed(self, mantic_run: _ManticRun) -> None:
        """Every value came off the structured block (extraction rung 1) and every spread sat below
        its stacking threshold, so the twelve forecaster calls were the only LLM calls."""
        assert mantic_run.llm_calls.unexpected_prompts == []


class TestTheFrameworkDefaultTypeFilterLosesTheQuantitativeQuestion:
    """The negative control for the fetch fix, against the same fake transport.

    An unmodified ``MetaculusClient`` pointed at the Mantic base URL sends
    ``forecast_type=binary,numeric,...``; Mantic answers that with none of its quantitative
    questions, so the 450-bin bitcoin question never reaches the bot at all. ``ManticClient`` omits
    the parameter and gets it.
    """

    def test_the_framework_default_drops_post_650(self, mantic_posts_transport: list[_RecordedRequest]) -> None:
        plain_client = MetaculusClient(base_url=MANTIC_API_BASE_URL, token=_FAKE_TOKEN)
        fetched = plain_client.get_all_open_questions_from_tournament(MANTIC_TOURNAMENT_ID)
        post_ids = {question.id_of_post for question in fetched}
        assert DISCRETE_POST_ID not in post_ids
        assert BINARY_POST_ID in post_ids
        assert "forecast_type" in mantic_posts_transport[0].query

    def test_the_mantic_client_keeps_post_650(self, mantic_posts_transport: list[_RecordedRequest]) -> None:
        fetched = ManticClient(token=_FAKE_TOKEN).get_all_open_questions_from_tournament(MANTIC_TOURNAMENT_ID)
        post_ids = {question.id_of_post for question in fetched}
        assert DISCRETE_POST_ID in post_ids
        assert post_ids == {BINARY_POST_ID, MULTIPLE_CHOICE_POST_ID, DISCRETE_POST_ID, DATE_POST_ID}
        assert "forecast_type" not in mantic_posts_transport[0].query
