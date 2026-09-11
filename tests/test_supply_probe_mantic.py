"""Tests for the supply probe's Mantic mode (``scripts/supply_probe.py --platform mantic``).

Mantic's Crucible (competitions.mantic.com) is a Metaculus fork whose read endpoints are public,
so the probe runs without a token there. ``with_cp=true`` rides every list GET, token or not: it is
what puts a RESOLVED question's public spot-time snapshot on the list page
(``question.aggregations.recency_weighted.score_data.disagreement_forecasts.forecasts[]``, one entry
per competitor keyed by ``author_id``), and under a token it also puts ``my_forecasts`` there, which
is the only way to classify a closed-but-unresolved question. Without the flag the list page carries
``score_data: {}`` on every post and nothing classifies, which is why the two recorded pages below are
the same two posts read both ways. The instrument the mode adds is the miss rate per UTC release
hour, which is what decides the cron-cadence question: GitHub delivers about a fifth of this
repository's scheduled firings and a Series 2 window is sixty minutes long.

Fixtures are the Metaculus builders from ``tests/supply_probe_fakes.py`` plus the snapshot, and two
recorded Series 1 list pages (unauthenticated, ``limit=2``, with and without ``with_cp=true``). No
live API: every test drives the pure functions or monkeypatches ``requests.get``, and the autouse
egress guard would raise on any real connect.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pytest

from metaculus_bot.constants import (
    MANTIC_API_BASE_URL,
    MANTIC_BOT_USER_ID,
    MANTIC_TOKEN_ENV,
    MANTIC_TOURNAMENT_ID,
    PLATFORM_MANTIC,
    PLATFORM_METACULUS,
)
from scripts import supply_probe
from scripts.supply_probe import (
    ReleaseHourRow,
    SlugSupply,
    WindowMinutes,
    fetch_posts_by_status,
    render_report,
    summarize_slug_supply,
)
from scripts.supply_probe_platforms import (
    DEFAULT_SLUGS,
    FORECAST_ABSENT,
    FORECAST_PRESENT,
    FORECAST_UNKNOWN,
    MANTIC_PROBE,
    METACULUS_PROBE,
    PLATFORM_PROBES,
    POSTS_URL,
    mantic_forecast_state,
)
from tests.http_fakes import json_response
from tests.supply_probe_fakes import NOW, _post, _question

MANTIC_NOW = datetime(2026, 9, 9, 18, 0, tzinfo=UTC)
# Unauthenticated GET /api/posts/?tournaments=series-1&statuses=resolved&limit=2, read with and without with_cp=true.
_DATA = Path(__file__).parent / "data"
SERIES1_LIST_WITH_CP_FIXTURE = _DATA / "mantic_series1_resolved_list_with_cp_public_2026_09_09.json"
SERIES1_LIST_WITHOUT_WITH_CP_FIXTURE = _DATA / "mantic_series1_resolved_list_without_with_cp_public_2026_09_09.json"


def _snapshot(authors: Sequence[int] | None) -> dict:
    """``score_data`` as the public wire carries it: null ``disagreement_forecasts`` on a
    closed-but-unresolved post, a (possibly empty) ``forecasts`` list once scored."""
    if authors is None:
        return {"disagreement_forecasts": None}
    return {
        "disagreement_forecasts": {
            "forecast_count": len(authors),
            "spot_scoring_time": 1_757_430_000.0,
            "forecasts": [
                {
                    "author_id": author,
                    "author_username": f"user-{author}",
                    "start_time": 1_757_426_400.0,
                    "end_time": None,
                    "values": [0.4, 0.6],
                    "pmf": [0.0, 0.4, 0.6],
                }
                for author in authors
            ],
        }
    }


def _mantic_question(
    qid: int,
    *,
    snapshot_authors: Sequence[int] | None = None,
    forecast: bool | None = None,
    open_time: str | None = "2026-09-09T14:00:00Z",
    close_time: str | None = "2026-09-09T15:00:00Z",
    resolved: bool = False,
    score_data: dict | None = None,
) -> dict:
    """A Mantic question dict: the Metaculus shape plus the public spot-time snapshot.

    ``snapshot_authors`` lists the author ids the snapshot names (an empty list is the
    resolved-but-not-yet-scored shape; None is the closed-but-unresolved null). ``forecast``
    is the same three-state ``my_forecasts`` control as :func:`_question`, and only a token
    produces it on Mantic. ``score_data`` overrides the whole block for the open-post ``{}``.
    """
    question = _question(
        qid,
        qtype="binary",
        forecast=forecast,
        open_time=open_time,
        close_time=close_time,
        actual="2026-09-09T16:00:00Z" if resolved else None,
        resolution="yes" if resolved else None,
    )
    question["scheduled_close_time"] = close_time
    block = _snapshot(snapshot_authors) if score_data is None else score_data
    question["aggregations"] = {"recency_weighted": {"score_data": block}}
    return question


def _mantic_summary(posts_by_status: dict) -> SlugSupply:
    return summarize_slug_supply("series-2", posts_by_status, now=MANTIC_NOW, platform=MANTIC_PROBE)


class TestManticForecastState:
    """The Mantic three-state read: the token's own block when present, else the platform's
    public spot-time snapshot, else unknown. The snapshot is what the platform scores, so a
    non-empty one without the bot's id is a forfeit; an empty or null one is the pre-scoring
    state and says nothing."""

    def _state(self, question: dict, *, bot_user_id: int = MANTIC_BOT_USER_ID) -> str:
        return mantic_forecast_state(question, bot_user_id=bot_user_id)

    def test_the_token_block_wins_over_the_snapshot(self):
        forecast_by_token = _mantic_question(1, forecast=True, snapshot_authors=[13])
        absent_by_token = _mantic_question(2, forecast=False, snapshot_authors=[MANTIC_BOT_USER_ID])

        assert self._state(forecast_by_token) == FORECAST_PRESENT
        assert self._state(absent_by_token) == FORECAST_ABSENT

    def test_a_snapshot_naming_the_bot_is_a_forecast(self):
        question = _mantic_question(3, snapshot_authors=[13, MANTIC_BOT_USER_ID, 22], resolved=True)
        assert self._state(question) == FORECAST_PRESENT

    def test_a_snapshot_without_the_bot_is_a_forfeit(self):
        question = _mantic_question(4, snapshot_authors=[13, 22], resolved=True)
        assert self._state(question) == FORECAST_ABSENT

    def test_no_snapshot_and_no_token_block_is_unknown(self):
        """The closed-but-unresolved wire shape: ``disagreement_forecasts`` is null."""
        assert self._state(_mantic_question(5)) == FORECAST_UNKNOWN

    def test_an_empty_snapshot_list_is_unknown_not_a_forfeit(self):
        question = _mantic_question(6, snapshot_authors=[], resolved=True)
        assert self._state(question) == FORECAST_UNKNOWN

    def test_an_open_posts_empty_score_data_is_unknown(self):
        assert self._state(_mantic_question(7, score_data={})) == FORECAST_UNKNOWN

    def test_a_null_token_block_falls_through_to_the_snapshot(self):
        question = _mantic_question(8, snapshot_authors=[MANTIC_BOT_USER_ID], resolved=True) | {"my_forecasts": None}
        assert self._state(question) == FORECAST_PRESENT

    def test_the_snapshot_is_read_against_the_configured_bot_id(self):
        question = _mantic_question(9, snapshot_authors=[13], resolved=True)

        assert self._state(question, bot_user_id=13) == FORECAST_PRESENT
        assert self._state(question, bot_user_id=MANTIC_BOT_USER_ID) == FORECAST_ABSENT

    def test_the_platform_probe_binds_the_repo_bot_id(self):
        question = _mantic_question(10, snapshot_authors=[MANTIC_BOT_USER_ID], resolved=True)
        assert MANTIC_PROBE.forecast_state(question) == FORECAST_PRESENT
        assert MANTIC_PROBE.bot_user_id == MANTIC_BOT_USER_ID


class TestReleaseHourTable:
    """The cadence instrument: forfeit-eligible questions grouped by the UTC hour they opened,
    with the miss rate over the DECIDED ones, plus the realized window length so the
    60-minute assumption is checked by the same run."""

    def _posts(self) -> dict:
        return {
            "resolved": [
                _post(1401, _mantic_question(141, snapshot_authors=[MANTIC_BOT_USER_ID, 13], resolved=True)),
                _post(1402, _mantic_question(142, snapshot_authors=[13], resolved=True)),
            ],
            "closed": [
                _post(1501, _mantic_question(151, open_time="2026-09-09T15:00:00Z", close_time="2026-09-09T16:00:00Z"))
            ],
            "open": [
                _post(1601, _mantic_question(161, open_time="2026-09-09T17:00:00Z", close_time="2026-09-09T18:00:00Z"))
            ],
        }

    def test_rows_group_eligible_questions_by_utc_open_hour(self):
        supply = _mantic_summary(self._posts())

        assert supply.by_release_hour == (
            ReleaseHourRow(hour_utc=14, questions=2, forecast=1, no_forecast=1, unknown=0, miss_rate=0.5),
            ReleaseHourRow(hour_utc=15, questions=1, forecast=0, no_forecast=0, unknown=1, miss_rate=None),
        )

    def test_the_window_line_reads_the_open_to_close_minutes(self):
        supply = _mantic_summary(self._posts())

        assert supply.window_minutes == WindowMinutes(shortest=60.0, median=60.0, longest=60.0, questions=3)

    def test_the_window_distribution_spans_the_eligible_questions(self):
        posts = self._posts()
        posts["closed"].append(
            _post(1502, _mantic_question(152, open_time="2026-09-08T12:00:00Z", close_time="2026-09-09T18:00:00Z"))
        )

        supply = _mantic_summary(posts)

        assert supply.window_minutes == WindowMinutes(shortest=60.0, median=60.0, longest=1800.0, questions=4)

    def test_render_shows_one_line_per_hour_a_total_and_the_window(self):
        text = render_report([_mantic_summary(self._posts())], now=MANTIC_NOW, platform=MANTIC_PROBE)

        table = [line.split() for line in text.splitlines() if line.startswith("    ") and "hour" not in line]
        assert ["14", "2", "1", "1", "0", "50.0%"] in table
        assert ["15", "1", "0", "0", "1", "-"] in table
        assert ["total", "3", "1", "1", "1", "50.0%"] in table
        assert "window open-to-close, minutes: min 60 / median 60 / max 60 (over 3 questions)" in text

    def test_the_json_dump_carries_the_same_numbers(self):
        payload = json.loads(json.dumps(asdict(_mantic_summary(self._posts()))))

        assert payload["by_release_hour"] == [
            {"hour_utc": 14, "questions": 2, "forecast": 1, "no_forecast": 1, "unknown": 0, "miss_rate": 0.5},
            {"hour_utc": 15, "questions": 1, "forecast": 0, "no_forecast": 0, "unknown": 1, "miss_rate": None},
        ]
        assert payload["window_minutes"] == {"shortest": 60.0, "median": 60.0, "longest": 60.0, "questions": 3}

    def test_a_question_paged_under_two_statuses_is_counted_once(self):
        post = _post(1701, _mantic_question(171, snapshot_authors=[13], resolved=True))
        supply = _mantic_summary({"closed": [post], "resolved": [post]})

        assert supply.by_release_hour == (
            ReleaseHourRow(hour_utc=14, questions=1, forecast=0, no_forecast=1, unknown=0, miss_rate=1.0),
        )
        assert supply.window_minutes == WindowMinutes(shortest=60.0, median=60.0, longest=60.0, questions=1)

    def test_an_unreadable_open_time_is_its_own_bucket_and_leaves_the_window(self):
        supply = _mantic_summary({"closed": [_post(1801, _mantic_question(181, open_time=None))]})

        assert supply.by_release_hour == (
            ReleaseHourRow(hour_utc=None, questions=1, forecast=0, no_forecast=0, unknown=1, miss_rate=None),
        )
        assert supply.window_minutes is None
        text = render_report([supply], now=MANTIC_NOW, platform=MANTIC_PROBE)
        assert ["n/a", "1", "0", "0", "1", "-"] in [line.split() for line in text.splitlines()]

    def test_nothing_eligible_means_no_table(self):
        supply = _mantic_summary({"open": [_post(1901, _mantic_question(191))]})

        assert supply.by_release_hour == ()
        assert supply.window_minutes is None
        assert "release hour" not in render_report([supply], now=MANTIC_NOW, platform=MANTIC_PROBE)

    def test_the_metaculus_mode_gets_the_same_table(self):
        """One instrument, both platforms: a Metaculus forfeit against a 12:00 close is the same
        loss, and the default-platform path computes the table from its own token read."""
        by_status = {
            "closed": [
                _post(2001, _question(201, forecast=False, open_time="2026-08-03T12:00:00Z")),
                _post(2002, _question(202, forecast=True, open_time="2026-08-03T12:30:00Z")),
            ]
        }

        supply = summarize_slug_supply("summer-futureeval-2026", by_status, now=NOW)

        assert supply.by_release_hour == (
            ReleaseHourRow(hour_utc=12, questions=2, forecast=1, no_forecast=1, unknown=0, miss_rate=0.5),
        )
        assert "Miss rate by UTC release hour" in render_report([supply], now=NOW)


class TestManticPaging:
    """Mantic populates ``next`` even on the last page and reports ``count`` as null, so the
    short-page stop is the only end signal — and the probe never reads either field."""

    def _install_pages(self, monkeypatch, pages: list[list[dict]]):
        seen: list[dict] = []

        def _fake_get(params, token, *, url=POSTS_URL):
            seen.append({"url": url, "params": dict(params), "token": token})
            index = params["offset"] // supply_probe.PAGE_SIZE
            results = pages[index] if index < len(pages) else []
            return {"results": results, "next": f"{url}?offset={(index + 1) * supply_probe.PAGE_SIZE}", "count": None}

        monkeypatch.setattr(supply_probe, "_get_json", _fake_get)
        monkeypatch.setattr(supply_probe.time, "sleep", lambda _s: None)
        return seen

    def test_a_populated_next_on_the_short_page_does_not_prolong_the_walk(self, monkeypatch):
        full = [_post(3000 + i, _mantic_question(3000 + i, snapshot_authors=[13], resolved=True)) for i in range(100)]
        short = [_post(3100 + i, _mantic_question(3100 + i, snapshot_authors=[13], resolved=True)) for i in range(20)]
        seen = self._install_pages(monkeypatch, [full, short, []])

        by_status = fetch_posts_by_status("series-1", ("resolved",), None, platform=MANTIC_PROBE)

        assert len(by_status["resolved"]) == 120
        assert [call["params"]["offset"] for call in seen] == [0, 100]
        assert {call["url"] for call in seen} == {MANTIC_PROBE.posts_url}

    def test_without_a_token_the_list_params_still_include_with_cp(self, monkeypatch):
        """``with_cp=true`` is what puts the public snapshot on the list page, so it rides every page
        GET whether or not a token is present; the token adds only ``my_forecasts``."""
        seen = self._install_pages(monkeypatch, [[]])

        fetch_posts_by_status("series-1", ("closed",), None, platform=MANTIC_PROBE)

        assert seen[0]["params"] == {
            "tournaments": "series-1",
            "statuses": "closed",
            "limit": 100,
            "offset": 0,
            "with_cp": "true",
        }
        assert seen[0]["token"] is None

    def test_with_a_token_the_list_pages_ask_for_my_forecasts(self, monkeypatch):
        seen = self._install_pages(monkeypatch, [[]])

        fetch_posts_by_status("series-1", ("closed",), "personal-token", platform=MANTIC_PROBE)

        assert seen[0]["params"]["with_cp"] == "true"
        assert "forecaster_id" not in seen[0]["params"]
        assert seen[0]["token"] == "personal-token"

    def test_the_metaculus_platform_sends_with_cp_too(self, monkeypatch):
        """On Metaculus ``with_cp=true`` puts the token's ``my_forecasts`` on the list page, which is
        what lets the forfeit sweep skip its per-post detail GETs."""
        seen = self._install_pages(monkeypatch, [[]])

        fetch_posts_by_status("summer-futureeval-2026", ("closed",), "token")

        assert seen[0]["params"] == {
            "tournaments": "summer-futureeval-2026",
            "statuses": "closed",
            "limit": 100,
            "offset": 0,
            "with_cp": "true",
        }
        assert seen[0]["url"] == POSTS_URL


class TestGetJsonWithoutAToken:
    def test_no_authorization_header_is_sent_when_there_is_no_token(self, monkeypatch):
        calls: list[dict] = []

        def _fake_get(url, *, headers, params, timeout):
            calls.append({"url": url, "headers": headers})
            return json_response({"results": []}, url=url)

        monkeypatch.setattr(supply_probe.requests, "get", _fake_get)

        supply_probe._get_json({"limit": 1}, None, url=MANTIC_PROBE.posts_url)

        assert calls == [{"url": MANTIC_PROBE.posts_url, "headers": {}}]


class TestManticMain:
    """``--platform mantic`` end to end over a fake ``requests.get``: the Mantic host, the Mantic
    preflight, an OPTIONAL token, ``with_cp`` on every list GET, never ``forecaster_id``, no
    detail GETs, and the platform in the JSON dump."""

    def _install(self, monkeypatch, *, token: str | None, results_by_status: dict[str, list[dict]]):
        calls: list[dict] = []
        preflights: list[str] = []

        def _fake_get(url, *, headers, params, timeout):
            calls.append({"url": url, "headers": dict(headers), "params": dict(params)})
            results = results_by_status.get(str(params.get("statuses")), [])
            page = {"results": results, "next": f"{url}?offset=100", "previous": None, "count": None}
            return json_response(page, url=url)

        monkeypatch.setattr(supply_probe.requests, "get", _fake_get)
        monkeypatch.setattr(supply_probe.time, "sleep", lambda _s: None)
        monkeypatch.setattr(supply_probe, "load_environment", lambda: None)
        monkeypatch.setattr(supply_probe, "verify_api_identity", preflights.append)
        monkeypatch.setattr(
            supply_probe,
            "resolve_bot_forecasts",
            lambda *_a, **_k: pytest.fail("mantic mode answers off the list page and issues no detail GET"),
        )
        monkeypatch.delenv("METACULUS_TOKEN", raising=False)
        if token is None:
            monkeypatch.delenv(MANTIC_TOKEN_ENV, raising=False)
        else:
            monkeypatch.setenv(MANTIC_TOKEN_ENV, token)
        return calls, preflights

    def _results(self) -> dict[str, list[dict]]:
        return {
            "resolved": [
                _post(4001, _mantic_question(401, snapshot_authors=[13], resolved=True), title="Will X happen?"),
                _post(4002, _mantic_question(402, snapshot_authors=[MANTIC_BOT_USER_ID, 13], resolved=True)),
            ],
            "closed": [_post(4003, _mantic_question(403, open_time="2026-09-09T15:00:00Z"))],
        }

    def test_without_a_token_the_probe_runs_public_only(self, monkeypatch, capsys):
        calls, preflights = self._install(monkeypatch, token=None, results_by_status=self._results())
        monkeypatch.setattr("sys.argv", ["supply_probe", "--platform", "mantic"])

        supply_probe.main()
        out = capsys.readouterr().out

        assert preflights == [MANTIC_API_BASE_URL]
        assert {call["url"] for call in calls} == {f"{MANTIC_API_BASE_URL}/posts/"}
        assert all(call["headers"] == {} for call in calls)
        assert all(call["params"]["with_cp"] == "true" and "forecaster_id" not in call["params"] for call in calls)
        assert {call["params"]["tournaments"] for call in calls} == {MANTIC_TOURNAMENT_ID}
        assert out.startswith("Mantic question-supply probe.")
        assert "never forecast by the bot: 1 (of 3; forecast 1, unknown 1)" in out
        assert "Will X happen?" in out
        assert "Miss rate by UTC release hour" in out

    def test_a_missing_token_is_logged_not_fatal(self, monkeypatch, capsys, caplog):
        self._install(monkeypatch, token=None, results_by_status={})
        monkeypatch.setattr("sys.argv", ["supply_probe", "--platform", "mantic", "--slugs", "series-2"])

        with caplog.at_level(logging.INFO, logger=supply_probe.logger.name):
            supply_probe.main()
        capsys.readouterr()

        messages = [record.getMessage() for record in caplog.records]
        assert any(MANTIC_TOKEN_ENV in message and "unknown" in message for message in messages)

    def test_with_a_token_every_list_page_asks_for_my_forecasts(self, monkeypatch, capsys, caplog):
        calls, _preflights = self._install(monkeypatch, token="personal-secret", results_by_status=self._results())
        monkeypatch.setattr("sys.argv", ["supply_probe", "--platform", "mantic", "--slugs", "series-2"])

        with caplog.at_level(logging.DEBUG):
            supply_probe.main()
        out = capsys.readouterr().out

        assert calls, "the fake must have been paged"
        assert all(call["params"]["with_cp"] == "true" for call in calls)
        assert all("forecaster_id" not in call["params"] for call in calls)
        assert all(call["headers"] == {"Authorization": "Token personal-secret"} for call in calls)
        assert "personal-secret" not in out
        assert all("personal-secret" not in record.getMessage() for record in caplog.records)

    def test_the_json_dump_names_the_platform_and_the_bot_id(self, monkeypatch, tmp_path, capsys):
        self._install(monkeypatch, token=None, results_by_status=self._results())
        dump = tmp_path / "mantic_supply.json"
        monkeypatch.setattr("sys.argv", ["supply_probe", "--platform", "mantic", "--output", str(dump)])

        supply_probe.main()
        capsys.readouterr()

        payload = json.loads(dump.read_text())
        assert payload["platform"] == PLATFORM_MANTIC
        assert payload["bot_user_id"] == MANTIC_BOT_USER_ID
        slug = payload["slugs"][0]
        assert slug["slug"] == MANTIC_TOURNAMENT_ID
        assert [row["hour_utc"] for row in slug["by_release_hour"]] == [14, 15]
        assert slug["window_minutes"]["median"] == 60.0

    def test_all_unknown_points_at_the_token_not_the_forfeit_flag(self, monkeypatch, capsys):
        results = {"closed": [_post(4101, _mantic_question(411))]}
        self._install(monkeypatch, token=None, results_by_status=results)
        monkeypatch.setattr("sys.argv", ["supply_probe", "--platform", "mantic", "--slugs", "series-2"])

        supply_probe.main()
        out = capsys.readouterr().out

        assert MANTIC_TOKEN_ENV in out
        assert "--no-forfeits" not in out

    def test_a_slug_with_no_bot_forecast_names_the_mantic_identity(self, monkeypatch, capsys):
        results = {"resolved": [_post(4201, _mantic_question(421, snapshot_authors=[13], resolved=True))]}
        self._install(monkeypatch, token=None, results_by_status=results)
        monkeypatch.setattr("sys.argv", ["supply_probe", "--platform", "mantic", "--slugs", "series-1"])

        supply_probe.main()
        out = capsys.readouterr().out

        assert f"MANTIC_BOT_USER_ID ({MANTIC_BOT_USER_ID})" in out
        assert "METACULUS_TOKEN" not in out


class TestMetaculusMode:
    """The default platform: the Metaculus posts URL, the Metaculus base URL vetted by the
    preflight, a REQUIRED token in the header, the five list params (``with_cp=true`` puts the
    token's ``my_forecasts`` on the list page) and nothing else."""

    def _install(self, monkeypatch):
        calls: list[dict] = []
        preflights: list[str] = []

        def _fake_get(url, *, headers, params, timeout):
            calls.append({"url": url, "headers": dict(headers), "params": dict(params)})
            return json_response({"results": [_post(5001, _question(501, forecast=True))]}, url=url)

        monkeypatch.setattr(supply_probe.requests, "get", _fake_get)
        monkeypatch.setattr(supply_probe.time, "sleep", lambda _s: None)
        monkeypatch.setattr(supply_probe, "load_environment", lambda: None)
        monkeypatch.setattr(supply_probe, "verify_api_identity", preflights.append)
        monkeypatch.setenv("METACULUS_TOKEN", "metaculus-token")
        return calls, preflights

    @pytest.mark.parametrize("platform_argv", [(), ("--platform", PLATFORM_METACULUS)])
    def test_the_metaculus_list_request_is_the_authenticated_five_param_get(self, monkeypatch, capsys, platform_argv):
        calls, preflights = self._install(monkeypatch)
        monkeypatch.setattr("sys.argv", ["supply_probe", *platform_argv, "--slugs", "slug", "--statuses", "closed"])

        supply_probe.main()
        out = capsys.readouterr().out

        assert preflights == [METACULUS_PROBE.base_url]
        assert calls == [
            {
                "url": POSTS_URL,
                "headers": {"Authorization": "Token metaculus-token"},
                "params": {"tournaments": "slug", "statuses": "closed", "limit": 100, "offset": 0, "with_cp": "true"},
            }
        ]
        assert out.startswith("Metaculus question-supply probe.")

    def test_the_json_dump_names_the_metaculus_platform(self, monkeypatch, tmp_path, capsys):
        self._install(monkeypatch)
        dump = tmp_path / "supply.json"
        monkeypatch.setattr(
            "sys.argv", ["supply_probe", "--slugs", "slug", "--statuses", "closed", "--output", str(dump)]
        )

        supply_probe.main()
        capsys.readouterr()

        payload = json.loads(dump.read_text())
        assert payload["platform"] == PLATFORM_METACULUS
        assert payload["bot_user_id"] is None

    def test_a_mantic_token_alone_does_not_satisfy_the_metaculus_mode(self, monkeypatch):
        self._install(monkeypatch)
        monkeypatch.delenv("METACULUS_TOKEN", raising=False)
        monkeypatch.setenv(MANTIC_TOKEN_ENV, "personal-token")
        monkeypatch.setattr("sys.argv", ["supply_probe", "--slugs", "slug"])

        with pytest.raises(SystemExit):
            supply_probe.main()


class TestManticDefaults:
    def test_the_platform_table_covers_both_platforms(self):
        assert PLATFORM_PROBES.keys() == {PLATFORM_METACULUS, PLATFORM_MANTIC}
        assert PLATFORM_PROBES[PLATFORM_METACULUS] is METACULUS_PROBE
        assert PLATFORM_PROBES[PLATFORM_MANTIC] is MANTIC_PROBE

    def test_the_mantic_probe_url_shares_the_host_its_preflight_vets(self):
        """Same promise as the Metaculus pin: the vetted base URL is the one the token goes to."""
        assert MANTIC_PROBE.base_url == MANTIC_API_BASE_URL
        assert MANTIC_PROBE.posts_url == f"{MANTIC_API_BASE_URL}/posts/"
        assert MANTIC_PROBE.posts_url.startswith("https://competitions.mantic.com/api/")

    def test_the_mantic_defaults_come_from_the_repo_constants(self):
        assert MANTIC_PROBE.default_slugs == (MANTIC_TOURNAMENT_ID,)
        assert MANTIC_PROBE.token_env == MANTIC_TOKEN_ENV
        assert MANTIC_PROBE.token_required is False
        assert METACULUS_PROBE.token_required is True
        assert METACULUS_PROBE.default_slugs == DEFAULT_SLUGS


@pytest.fixture(scope="module")
def series1_with_cp_page() -> dict:
    return json.loads(SERIES1_LIST_WITH_CP_FIXTURE.read_text())


@pytest.fixture(scope="module")
def series1_without_with_cp_page() -> dict:
    return json.loads(SERIES1_LIST_WITHOUT_WITH_CP_FIXTURE.read_text())


def _score_data(post: dict) -> dict:
    return post["question"]["aggregations"]["recency_weighted"]["score_data"]


def _snapshot_authors(post: dict) -> set[int]:
    return {entry["author_id"] for entry in _score_data(post)["disagreement_forecasts"]["forecasts"]}


class TestRecordedManticListPages:
    """The two recorded payloads: the same two resolved Series 1 posts, read unauthenticated with
    and without ``with_cp=true``. The first is the request the probe issues and carries the public
    snapshot; the second is what the list GET returns without the flag, and it classifies nothing,
    which is the shape the tokenless mode read until the flag rode every page."""

    def test_the_with_cp_page_carries_the_snapshot_and_no_token_block(self, series1_with_cp_page):
        posts = series1_with_cp_page["results"]

        assert [post["id"] for post in posts] == [645, 643]
        assert all(post["status"] == "resolved" for post in posts)
        assert all("my_forecasts" not in post["question"] for post in posts), "the recording was unauthenticated"
        assert [len(_snapshot_authors(post)) for post in posts] == [10, 9]
        assert all(MANTIC_BOT_USER_ID not in _snapshot_authors(post) for post in posts)

    def test_the_snapshot_names_one_competitor_fewer_than_nr_forecasters(self, series1_with_cp_page):
        """The gap the report header discloses: on 507 of the 524 snapshot-bearing posts in the
        2026-09-08 public corpus the snapshot is short by exactly one, these two included."""
        posts = series1_with_cp_page["results"]

        assert [post["nr_forecasters"] - len(_snapshot_authors(post)) for post in posts] == [1, 1]

    def test_the_bot_reads_as_absent_and_a_named_competitor_as_present(self, series1_with_cp_page):
        post = series1_with_cp_page["results"][0]
        a_competitor = next(iter(_snapshot_authors(post)))

        assert mantic_forecast_state(post["question"], bot_user_id=MANTIC_BOT_USER_ID) == FORECAST_ABSENT
        assert mantic_forecast_state(post["question"], bot_user_id=a_competitor) == FORECAST_PRESENT

    def test_the_page_without_with_cp_is_the_same_posts_with_empty_score_data(
        self, series1_with_cp_page, series1_without_with_cp_page
    ):
        posts = series1_without_with_cp_page["results"]

        assert [post["id"] for post in posts] == [post["id"] for post in series1_with_cp_page["results"]]
        assert all(post["status"] == "resolved" for post in posts)
        assert all(_score_data(post) == {} for post in posts)
        assert all("my_forecasts" not in post["question"] for post in posts)

    def test_the_with_cp_page_summarizes_as_two_forfeits_in_the_23_utc_hour(self, series1_with_cp_page):
        supply = summarize_slug_supply(
            "series-1", {"resolved": series1_with_cp_page["results"]}, now=MANTIC_NOW, platform=MANTIC_PROBE
        )

        assert [row.question_id for row in supply.forfeits] == [645, 643]
        assert supply.forecast_states == supply_probe.ForecastStateCounts(with_forecast=0, without_forecast=2)
        assert supply.by_release_hour == (
            ReleaseHourRow(hour_utc=23, questions=2, forecast=0, no_forecast=2, unknown=0, miss_rate=1.0),
        )
        assert supply.window_minutes == WindowMinutes(shortest=60.0, median=60.0, longest=60.0, questions=2)
        assert supply.backlog == ()

    def test_the_page_without_with_cp_reads_every_question_unknown(self, series1_without_with_cp_page):
        """Why ``with_cp=true`` rides every list GET: without it the same resolved posts classify as
        nothing, and the report says so instead of printing zero forfeits."""
        supply = summarize_slug_supply(
            "series-1", {"resolved": series1_without_with_cp_page["results"]}, now=MANTIC_NOW, platform=MANTIC_PROBE
        )

        assert supply.forfeits == ()
        assert supply.forecast_states == supply_probe.ForecastStateCounts(unknown=2)
        assert supply.by_release_hour == (
            ReleaseHourRow(hour_utc=23, questions=2, forecast=0, no_forecast=0, unknown=2, miss_rate=None),
        )
        assert MANTIC_PROBE.all_unknown_hint in render_report([supply], now=MANTIC_NOW, platform=MANTIC_PROBE)
