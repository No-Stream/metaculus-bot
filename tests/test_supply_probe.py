"""Tests for the tracked question-supply probe (scripts/supply_probe.py).

Fixtures are shaped like the Metaculus posts-list payload the probe pages: a
single-question post carries ``question``, a group post carries
``group_of_questions.questions``, a notebook-shaped post carries neither, and every
question carries ``scheduled_resolve_time`` / ``actual_resolve_time`` / ``resolution``.

The situation they model is the one this utility exists for. On 2026-08-31 the summer
tournament held 178 posts at status ``closed`` (closed to forecasting, not yet
resolved), 26 of them the frozen-triple checkpoint cohort, 16 of those already past their
own ``scheduled_resolve_time`` and the worst by 17.1 days — none of which two consecutive
rounds' supply projections could see, because their scratch probes queried only
``statuses=resolved`` and ``statuses=open``. The exact-numbers replay of that census
against these functions is a separate offline check (see the round-census validation under
``scratch/next_season_bundle_2026-09/19c/``); what is pinned here is the behavior.

No live API: every test drives the pure functions directly or monkeypatches the probe's
single HTTP entry point, and the autouse egress guard in tests/conftest.py would raise on
any real connect anyway.
"""

import json
from datetime import UTC, datetime

import pytest
import requests
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from metaculus_bot import api_preflight
from metaculus_bot.constants import FALL_CUP_SLUG, METACULUS_CUP_ID, TOURNAMENT_ID
from metaculus_bot.performance_analysis.collector import questions_on_post
from scripts import supply_probe
from scripts.supply_probe import (
    SlugSupply,
    fetch_posts_by_status,
    probe_slugs,
    render_report,
    summarize_slug_supply,
)

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
    ``my_forecasts`` entirely (a raw posts-LIST page, which is why the sweep needs detail
    GETs), True carries a forecast, False carries the empty block a never-forecast question
    shows under the bot's own token.
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


class TestQuestionsOnPost:
    """The shared unwrapping, which lives in ``performance_analysis.collector`` — the probe
    reads the same posts list the scoring pull does, so both count questions one way. These
    cases were written against the probe's own former copy and are the only coverage the
    helper has, so they stay here rather than moving with it."""

    def test_single_question_post(self):
        post = _post(101, _question(1))
        assert [q["id"] for q in questions_on_post(post)] == [1]

    def test_group_post_returns_every_member(self):
        post = _group_post(102, [_question(2), _question(3)])
        assert [q["id"] for q in questions_on_post(post)] == [2, 3]

    def test_post_with_no_questions_is_empty(self):
        # Notebook-shaped posts live in tournaments too; they carry no forecastable question.
        assert questions_on_post({"id": 103, "title": "A notebook"}) == []


class TestStatusPartition:
    """The whole point of the utility: `closed` is counted, not skipped."""

    def _posts_by_status(self):
        return {
            "open": [_post(201, _question(11))],
            "closed": [
                _post(202, _question(12, scheduled="2026-08-14T00:00:00Z")),
                _group_post(203, [_question(13), _question(14)]),
            ],
            "resolved": [_post(204, _question(15, actual="2026-08-20T00:00:00Z", resolution="123"))],
        }

    def test_posts_and_questions_counted_per_status(self):
        supply = summarize_slug_supply("summer-futureeval-2026", self._posts_by_status(), now=NOW)

        counts = {c.status: (c.posts, c.questions) for c in supply.status_counts}
        assert counts == {"open": (1, 1), "closed": (2, 3), "resolved": (1, 1)}

    def test_status_order_follows_the_payload(self):
        supply = summarize_slug_supply("slug", self._posts_by_status(), now=NOW)
        assert [c.status for c in supply.status_counts] == ["open", "closed", "resolved"]

    def test_totals_dedup_posts_and_questions_seen_under_two_statuses(self):
        # A post that resolves mid-probe can be paged under both `closed` and `resolved`;
        # the per-status rows report what the API said, the totals count each post once.
        by_status = self._posts_by_status()
        by_status["resolved"].append(_post(202, _question(12, actual="2026-08-30T00:00:00Z", resolution="9")))

        supply = summarize_slug_supply("slug", by_status, now=NOW)

        assert supply.total_posts == 4
        assert supply.total_questions == 5

    def test_error_free_summary_carries_no_error(self):
        assert summarize_slug_supply("slug", self._posts_by_status(), now=NOW).error is None


class TestBacklog:
    """Unresolved questions past their own scheduled_resolve_time, worst first."""

    def test_overdue_closed_questions_are_the_backlog(self):
        by_status = {
            "closed": [
                _post(301, _question(21, scheduled="2026-08-14T00:00:00Z"), title="Drilling rigs?"),
                _post(302, _question(22, scheduled="2026-08-28T00:00:00Z")),
                _post(303, _question(23, scheduled="2026-09-30T00:00:00Z")),  # not yet due
            ]
        }

        supply = summarize_slug_supply("slug", by_status, now=NOW)

        assert [row.question_id for row in supply.backlog] == [21, 22]
        assert supply.backlog[0].overdue_days == pytest.approx(17.1)
        assert supply.backlog[1].overdue_days == pytest.approx(3.1)
        assert supply.worst_overdue_days == pytest.approx(17.1)
        assert supply.backlog[0].post_status == "closed"
        assert supply.backlog[0].title == "Drilling rigs?"

    def test_open_questions_past_schedule_are_tagged_not_dropped(self):
        by_status = {"open": [_post(304, _question(24, scheduled="2026-08-25T00:00:00Z"))]}

        supply = summarize_slug_supply("slug", by_status, now=NOW)

        assert [(row.question_id, row.post_status) for row in supply.backlog] == [(24, "open")]

    def test_resolved_questions_never_enter_the_backlog(self):
        by_status = {
            "resolved": [_post(305, _question(25, scheduled="2026-08-01T00:00:00Z", actual="2026-08-02T00:00:00Z"))]
        }

        assert summarize_slug_supply("slug", by_status, now=NOW).backlog == ()

    def test_question_resolved_inside_a_closed_group_post_is_not_backlog(self):
        # A group post's status is the POST's; individual members resolve on their own
        # schedules, so counting every member of a `closed` post as unresolved over-counts.
        by_status = {
            "closed": [
                _group_post(
                    306,
                    [
                        _question(26, scheduled="2026-08-14T00:00:00Z", resolution="4.5"),
                        _question(27, scheduled="2026-08-14T00:00:00Z"),
                    ],
                )
            ]
        }

        supply = summarize_slug_supply("slug", by_status, now=NOW)

        assert [row.question_id for row in supply.backlog] == [27]
        assert supply.resolved_within_unresolved_posts == 1

    def test_zero_resolution_counts_as_resolved(self):
        # A numeric/count question can resolve to 0; a truthiness test would read that
        # as unresolved and file it as backlog.
        by_status = {"closed": [_post(307, _question(28, scheduled="2026-08-14T00:00:00Z", resolution=0))]}

        assert summarize_slug_supply("slug", by_status, now=NOW).backlog == ()

    def test_resolved_elsewhere_in_the_probe_wins_over_a_stale_closed_page(self):
        by_status = {
            "closed": [_post(308, _question(29, scheduled="2026-08-14T00:00:00Z"))],
            "resolved": [_post(308, _question(29, scheduled="2026-08-14T00:00:00Z", actual="2026-08-30T00:00:00Z"))],
        }

        assert summarize_slug_supply("slug", by_status, now=NOW).backlog == ()

    def test_missing_and_unparseable_schedules_are_disclosed_not_imputed(self):
        by_status = {
            "closed": [
                _post(309, _question(30, scheduled=None)),
                _post(310, _question(31, scheduled="not a timestamp")),
                _post(311, _question(32, scheduled="2026-08-14T00:00:00Z")),
            ]
        }

        supply = summarize_slug_supply("slug", by_status, now=NOW)

        assert [row.question_id for row in supply.backlog] == [32]
        assert supply.unresolved_without_schedule == 2

    def test_naive_now_is_read_as_utc(self):
        # Analysis scripts hand in bare datetimes; scheduled times parse to tz-aware UTC,
        # so a naive clock has to be normalized rather than raise on the subtraction.
        by_status = {"closed": [_post(313, _question(34, scheduled="2026-08-14T00:00:00Z"))]}

        naive = summarize_slug_supply("slug", by_status, now=NOW.replace(tzinfo=None))

        assert naive.worst_overdue_days == pytest.approx(17.1)

    def test_no_backlog_leaves_worst_overdue_at_zero(self):
        by_status = {"open": [_post(312, _question(33))]}
        supply = summarize_slug_supply("slug", by_status, now=NOW)

        assert supply.backlog == ()
        assert supply.worst_overdue_days == 0.0


class TestBotForecastState:
    """The three-state read of ``my_forecasts``. UNKNOWN is a measurement failure, not a
    forfeit, and conflating the two is how a sweep reports a whole season as forfeited."""

    def test_a_list_page_question_answers_unknown(self):
        assert supply_probe.bot_forecast_state(_question(1)) == supply_probe.FORECAST_UNKNOWN

    def test_an_empty_block_is_a_forfeit(self):
        assert supply_probe.bot_forecast_state(_question(1, forecast=False)) == supply_probe.FORECAST_ABSENT

    def test_a_populated_block_is_a_forecast(self):
        assert supply_probe.bot_forecast_state(_question(1, forecast=True)) == supply_probe.FORECAST_PRESENT

    def test_a_latest_without_a_history_still_counts_as_a_forecast(self):
        # Never call a real forecast a forfeit: the scoring collector keys on `latest`, so a
        # payload carrying it has a forecast whatever `history` looks like.
        question = _question(1) | {"my_forecasts": {"latest": {"forecast_values": [0.5, 0.5]}, "history": []}}
        assert supply_probe.bot_forecast_state(question) == supply_probe.FORECAST_PRESENT

    def test_a_null_block_answers_unknown_not_forfeit(self):
        question = _question(1) | {"my_forecasts": None}
        assert supply_probe.bot_forecast_state(question) == supply_probe.FORECAST_UNKNOWN


class TestForfeitSweep:
    """Which questions the sweep files as forfeited, from payloads that already answer."""

    def _summary(self, posts_by_status) -> SlugSupply:
        return summarize_slug_supply("summer-futureeval-2026", posts_by_status, now=NOW)

    def test_closed_and_resolved_unforecast_questions_are_forfeits(self):
        supply = self._summary(
            {
                "closed": [_post(901, _question(91, forecast=False))],
                "resolved": [_post(902, _question(92, actual="2026-08-01T00:00:00Z", resolution="3", forecast=False))],
            }
        )

        assert {row.question_id for row in supply.forfeits} == {91, 92}
        assert supply.forecast_states == supply_probe.ForecastStateCounts(
            with_forecast=0, without_forecast=2, unknown=0
        )

    def test_an_open_unforecast_question_is_not_a_forfeit(self):
        """An open question we have not forecast YET is supply, not loss."""
        supply = self._summary({"open": [_post(903, _question(93, forecast=False))]})

        assert supply.forfeits == ()
        assert supply.forecast_states.total == 0

    def test_a_forecast_question_is_counted_but_not_listed(self):
        supply = self._summary({"closed": [_post(904, _question(94, forecast=True))]})

        assert supply.forfeits == ()
        assert supply.forecast_states.with_forecast == 1

    def test_an_unknown_state_is_disclosed_rather_than_filed_as_a_forfeit(self):
        supply = self._summary({"closed": [_post(905, _question(95))]})

        assert supply.forfeits == ()
        assert supply.forecast_states == supply_probe.ForecastStateCounts(
            with_forecast=0, without_forecast=0, unknown=1
        )

    def test_group_members_are_swept_individually(self):
        supply = self._summary(
            {"closed": [_group_post(906, [_question(96, forecast=False), _question(97, forecast=True)])]}
        )

        assert [row.question_id for row in supply.forfeits] == [96]
        assert supply.forecast_states.with_forecast == 1

    def test_the_window_length_is_reported_in_hours(self):
        supply = self._summary(
            {
                "closed": [
                    _post(
                        907,
                        _question(
                            98,
                            forecast=False,
                            open_time="2026-07-20T03:00:00Z",
                            close_time="2026-07-20T06:00:00Z",
                        ),
                    )
                ]
            }
        )

        assert supply.forfeits[0].window_hours == pytest.approx(3.0)

    def test_an_unreadable_window_is_none_not_zero(self):
        supply = self._summary({"closed": [_post(908, _question(99, forecast=False, open_time=None))]})

        assert supply.forfeits[0].window_hours is None

    def test_forfeits_are_ordered_newest_window_first(self):
        supply = self._summary(
            {
                "closed": [
                    _post(910, _question(100, forecast=False, open_time="2026-07-01T00:00:00Z")),
                    _post(911, _question(101, forecast=False, open_time="2026-08-15T00:00:00Z")),
                ]
            }
        )

        assert [row.question_id for row in supply.forfeits] == [101, 100]

    def test_a_post_paged_under_two_statuses_is_counted_once(self):
        """A post that resolves mid-probe pages under both `closed` and `resolved`."""
        post = _post(912, _question(102, forecast=False))
        supply = self._summary({"closed": [post], "resolved": [post]})

        assert [row.question_id for row in supply.forfeits] == [102]
        assert supply.forecast_states.total == 1


class TestResolveBotForecasts:
    """The one place the sweep spends requests, and what it refuses to spend them on."""

    def _install_details(self, monkeypatch, details_by_post_id, *, failing_ids=()):
        seen: list[str] = []
        sleeps: list[float] = []

        def _fake_get(params, token, *, url=supply_probe.POSTS_URL):
            seen.append(url)
            post_id = int(url.rstrip("/").rsplit("/", 1)[-1])
            if post_id in failing_ids:
                raise requests.HTTPError(f"429 rate limit on {post_id}")
            return details_by_post_id[post_id]

        monkeypatch.setattr(supply_probe, "_get_json", _fake_get)
        monkeypatch.setattr(supply_probe.time, "sleep", sleeps.append)
        return seen, sleeps

    def test_one_detail_get_per_unanswered_post_and_the_payload_is_substituted(self, monkeypatch):
        by_status = {"closed": [_post(920, _question(110))]}
        detail = _post(920, _question(110, forecast=False))
        seen, _sleeps = self._install_details(monkeypatch, {920: detail})

        fetched = supply_probe.resolve_bot_forecasts(by_status, "token")

        assert fetched == 1
        assert seen == [f"{supply_probe.POSTS_URL}920/"]
        supply = summarize_slug_supply("slug", by_status, now=NOW)
        assert [row.question_id for row in supply.forfeits] == [110]

    def test_a_post_that_already_answers_costs_no_request(self, monkeypatch):
        by_status = {"closed": [_post(921, _question(111, forecast=True))]}
        seen, _sleeps = self._install_details(monkeypatch, {})

        assert supply_probe.resolve_bot_forecasts(by_status, "token") == 0
        assert seen == []

    def test_open_posts_are_never_fetched(self, monkeypatch):
        by_status = {"open": [_post(922, _question(112))]}
        seen, _sleeps = self._install_details(monkeypatch, {})

        assert supply_probe.resolve_bot_forecasts(by_status, "token") == 0
        assert seen == []

    def test_a_post_paged_twice_is_fetched_once_and_substituted_in_both_lists(self, monkeypatch):
        post = _post(923, _question(113))
        by_status = {"closed": [post], "resolved": [post]}
        detail = _post(923, _question(113, forecast=False))
        seen, _sleeps = self._install_details(monkeypatch, {923: detail})

        assert supply_probe.resolve_bot_forecasts(by_status, "token") == 1
        assert seen == [f"{supply_probe.POSTS_URL}923/"]
        assert all(
            supply_probe.bot_forecast_state(questions_on_post(posts[0])[0]) == supply_probe.FORECAST_ABSENT
            for posts in by_status.values()
        )

    def test_requests_are_spaced_between_posts_but_not_after_the_last(self, monkeypatch):
        by_status = {"closed": [_post(924, _question(114)), _post(925, _question(115))]}
        details = {924: _post(924, _question(114, forecast=False)), 925: _post(925, _question(115, forecast=True))}
        _seen, sleeps = self._install_details(monkeypatch, details)

        supply_probe.resolve_bot_forecasts(by_status, "token")

        assert sleeps == [supply_probe.DETAIL_REQUEST_SPACING_SECS]

    def test_a_failed_detail_fetch_leaves_the_state_unknown_and_does_not_raise(self, monkeypatch):
        """One unreachable post must not cost the slug its census, and must never be filed as
        a forfeit on the strength of a payload we could not read."""
        by_status = {"closed": [_post(926, _question(116)), _post(927, _question(117))]}
        details = {927: _post(927, _question(117, forecast=False))}
        self._install_details(monkeypatch, details, failing_ids=(926,))

        fetched = supply_probe.resolve_bot_forecasts(by_status, "token")

        assert fetched == 1
        supply = summarize_slug_supply("slug", by_status, now=NOW)
        assert [row.question_id for row in supply.forfeits] == [117]
        assert supply.forecast_states.unknown == 1

    def test_probe_slugs_does_not_resolve_forfeits_unless_asked(self, monkeypatch):
        monkeypatch.setattr(
            supply_probe,
            "fetch_posts_by_status",
            lambda slug, statuses, token: {"closed": [_post(928, _question(118))]},
        )
        monkeypatch.setattr(
            supply_probe,
            "resolve_bot_forecasts",
            lambda *_a, **_k: pytest.fail("probe_slugs must not spend requests by default"),
        )

        supplies = probe_slugs(["slug"], ("closed",), "token", now=NOW)

        assert supplies[0].forecast_states.unknown == 1


class TestRenderReport:
    def test_report_names_the_closed_count_and_the_worst_overdue(self):
        by_status = {
            "open": [_post(401, _question(41))],
            "closed": [_post(402, _question(42, scheduled="2026-08-14T00:00:00Z"), title="Drilling rigs?")],
            "resolved": [_post(403, _question(43, actual="2026-08-01T00:00:00Z"))],
        }
        supply = summarize_slug_supply("summer-futureeval-2026", by_status, now=NOW)

        text = render_report([supply], now=NOW)

        assert "summer-futureeval-2026" in text
        assert "closed" in text
        assert "17.1" in text
        assert "Drilling rigs?" in text

    def test_report_caps_the_backlog_table_and_names_the_remainder(self):
        by_status = {"closed": [_post(500 + i, _question(50 + i, scheduled="2026-08-14T00:00:00Z")) for i in range(5)]}
        supply = summarize_slug_supply("slug", by_status, now=NOW)

        text = render_report([supply], now=NOW, max_backlog_rows=2)

        assert "+3 more" in text

    def test_report_renders_an_error_row_without_counts(self):
        failed = SlugSupply(slug="metaculus-cup", error="HTTP 404 (slug not found)")

        text = render_report([failed], now=NOW)

        assert "metaculus-cup" in text
        assert "HTTP 404" in text

    def test_report_names_the_forfeits_with_their_window(self):
        by_status = {
            "closed": [
                _post(
                    404,
                    _question(
                        44,
                        forecast=False,
                        open_time="2026-08-03T12:00:00Z",
                        close_time="2026-08-03T15:00:00Z",
                    ),
                    title="Late submit?",
                )
            ],
            "resolved": [_post(405, _question(45, actual="2026-08-01T00:00:00Z", forecast=True))],
        }
        supply = summarize_slug_supply("slug", by_status, now=NOW)

        text = render_report([supply], now=NOW)

        assert "never forecast by the bot: 1 (of 2; forecast 1, unknown 0)" in text
        assert "Late submit?" in text
        assert "3.0" in text

    def test_report_says_nothing_about_forfeits_when_nothing_was_eligible(self):
        """An all-open slug has no forfeit-eligible question, so the block is absent rather
        than reporting a zero that reads as a clean sweep."""
        supply = summarize_slug_supply("slug", {"open": [_post(406, _question(46))]}, now=NOW)

        assert "never forecast" not in render_report([supply], now=NOW)

    def test_report_points_at_the_flag_when_every_state_is_unknown(self):
        supply = summarize_slug_supply("slug", {"closed": [_post(407, _question(47))]}, now=NOW)

        assert "--no-forfeits" in render_report([supply], now=NOW)

    def test_report_flags_a_slug_where_nothing_carries_a_forecast(self):
        """Far likelier a non-bot METACULUS_TOKEN than a total forfeit, and the report has to
        say so rather than publish a 100% forfeit rate."""
        by_status = {"closed": [_post(408 + i, _question(48 + i, forecast=False)) for i in range(3)]}
        supply = summarize_slug_supply("slug", by_status, now=NOW)

        assert "METACULUS_TOKEN is the bot's own token" in render_report([supply], now=NOW)

    def test_report_caps_the_forfeit_table_and_names_the_remainder(self):
        by_status = {"closed": [_post(420 + i, _question(60 + i, forecast=False)) for i in range(5)]}
        supply = summarize_slug_supply("slug", by_status, now=NOW)

        text = render_report([supply], now=NOW, max_forfeit_rows=2)

        assert "+3 more forfeited" in text


class TestRateLimitRetry:
    """``_get_json``'s 429 handling, driven over a fake ``requests.get``.

    Every other test in this file monkeypatches ``_get_json`` away, so nothing exercised the
    retry itself — and this is the one path the probe was built around: the posts endpoint
    rate-limits hard right after a full performance pull, and an unretried 429 turns a live
    slug into an error row that reads exactly like a dead one.
    """

    def _install_responses(self, monkeypatch, statuses: list[int], payload: dict | None = None):
        """Serve one canned response per call; collect the calls and the sleeps."""
        calls: list[dict] = []
        sleeps: list[float] = []
        served = iter(statuses)

        def _fake_get(url, *, headers, params, timeout):
            calls.append({"url": url, "headers": headers, "params": dict(params), "timeout": timeout})
            response = requests.Response()
            response.status_code = next(served)
            response.url = url
            response.encoding = "utf-8"
            response._content = json.dumps(payload if payload is not None else {"results": []}).encode()
            return response

        monkeypatch.setattr(supply_probe.requests, "get", _fake_get)
        monkeypatch.setattr(supply_probe.time, "sleep", sleeps.append)
        return calls, sleeps

    def test_a_429_is_retried_and_the_next_attempt_is_returned(self, monkeypatch):
        payload = {"results": [_post(801, _question(81))]}
        calls, sleeps = self._install_responses(monkeypatch, [429, 200], payload=payload)

        data = supply_probe._get_json({"tournaments": "slug", "statuses": "closed"}, "token")

        assert data == payload
        assert len(calls) == 2, "the rate-limited attempt must be retried, not surfaced"
        assert sleeps == [supply_probe.RETRY_BACKOFF_SECS]
        assert calls[0]["headers"] == {"Authorization": "Token token"}
        assert calls[0]["timeout"] == supply_probe.REQUEST_TIMEOUT_SECS

    def test_the_backoff_grows_one_multiple_per_attempt_and_none_follows_the_last(self, monkeypatch):
        # Linear, not exponential, and deliberately so: the endpoint recovers in seconds and
        # the probe pages several slugs. The absent trailing sleep is the point of the
        # arithmetic — waiting after the final attempt would delay a failure nobody retries.
        calls, sleeps = self._install_responses(monkeypatch, [429] * supply_probe.MAX_RETRIES)

        with pytest.raises(requests.RequestException):
            supply_probe._get_json({"tournaments": "slug", "statuses": "closed"}, "token")

        assert len(calls) == supply_probe.MAX_RETRIES
        assert sleeps == [
            supply_probe.RETRY_BACKOFF_SECS * (attempt + 1) for attempt in range(supply_probe.MAX_RETRIES - 1)
        ]

    def test_exhausted_retries_raise_a_requests_exception_naming_the_cause(self, monkeypatch):
        # requests.RequestException specifically, because that is the only class probe_slugs
        # catches: anything else aborts the whole survey instead of filing one error row.
        self._install_responses(monkeypatch, [429] * supply_probe.MAX_RETRIES)

        with pytest.raises(requests.RequestException) as excinfo:
            supply_probe._get_json({"tournaments": "slug", "statuses": "closed"}, "token")

        assert "429" in str(excinfo.value)
        assert "retries exhausted" in str(excinfo.value), (
            "the exhausted path must say six attempts were rate-limited, not report one unlucky request"
        )

    def test_a_non_429_error_status_is_not_retried(self, monkeypatch):
        # A 404 slug is the expected case here (the bare `metaculus-cup` slug 404s today), and
        # retrying it six times with backoff would stall the survey on every dead slug.
        calls, sleeps = self._install_responses(monkeypatch, [404])

        with pytest.raises(requests.HTTPError):
            supply_probe._get_json({"tournaments": "metaculus-cup", "statuses": "closed"}, "token")

        assert len(calls) == 1
        assert sleeps == []

    def test_a_rate_limited_slug_becomes_an_error_row_and_the_survey_continues(self, monkeypatch):
        """The end-to-end reason the exception class matters: exhaustion inside a slug has to
        surface as that slug's error row, with every later slug still surveyed."""
        payload = {"results": [_post(802, _question(82, scheduled="2026-08-14T00:00:00Z"))]}
        first_slug_statuses = [429] * supply_probe.MAX_RETRIES
        self._install_responses(monkeypatch, [*first_slug_statuses, 200], payload=payload)

        supplies = probe_slugs(["rate-limited-slug", "summer-futureeval-2026"], ("closed",), "token", now=NOW)

        assert [supply.slug for supply in supplies] == ["rate-limited-slug", "summer-futureeval-2026"]
        assert supplies[0].error is not None
        assert "retries exhausted" in supplies[0].error
        assert supplies[0].status_counts == ()
        assert supplies[1].error is None
        assert [row.question_id for row in supplies[1].backlog] == [82]


class TestFetchPaging:
    """Paging stops on a short page: the tournament-filtered list gives no usable count."""

    def _install_pages(self, monkeypatch, pages_by_status):
        seen: list[dict] = []

        def _fake_get(params, token):
            seen.append(dict(params))
            status = params["statuses"]
            index = params["offset"] // supply_probe.PAGE_SIZE
            pages = pages_by_status.get(status, [])
            return {"results": pages[index] if index < len(pages) else []}

        monkeypatch.setattr(supply_probe, "_get_json", _fake_get)
        monkeypatch.setattr(supply_probe.time, "sleep", lambda _s: None)
        return seen

    def test_full_page_is_followed_by_the_next_offset(self, monkeypatch):
        full = [_post(1000 + i, _question(1000 + i)) for i in range(supply_probe.PAGE_SIZE)]
        seen = self._install_pages(monkeypatch, {"closed": [full, [_post(2000, _question(2000))]]})

        by_status = fetch_posts_by_status("slug", ("closed",), "token")

        assert len(by_status["closed"]) == supply_probe.PAGE_SIZE + 1
        assert [p["offset"] for p in seen] == [0, supply_probe.PAGE_SIZE]

    def test_every_requested_status_is_paged(self, monkeypatch):
        seen = self._install_pages(
            monkeypatch,
            {"open": [[_post(1, _question(1))]], "closed": [[_post(2, _question(2))]], "resolved": [[]]},
        )

        by_status = fetch_posts_by_status("slug", ("open", "closed", "resolved"), "token")

        assert {k: len(v) for k, v in by_status.items()} == {"open": 1, "closed": 1, "resolved": 0}
        assert [p["statuses"] for p in seen] == ["open", "closed", "resolved"]
        assert {p["tournaments"] for p in seen} == {"slug"}

    def test_paging_is_bounded(self, monkeypatch):
        # An endpoint that never returns a short page must not page forever.
        def _always_full(params, token):
            offset = params["offset"]
            return {"results": [_post(3000 + offset, _question(3000 + offset))] * supply_probe.PAGE_SIZE}

        monkeypatch.setattr(supply_probe, "_get_json", _always_full)
        monkeypatch.setattr(supply_probe.time, "sleep", lambda _s: None)

        by_status = fetch_posts_by_status("slug", ("closed",), "token")

        assert len(by_status["closed"]) == supply_probe.PAGE_SIZE * supply_probe.MAX_PAGES


class TestProbeSlugsSoftFailsPerSlug:
    def test_one_dead_slug_does_not_hide_the_live_ones(self, monkeypatch):
        def _fake_fetch(slug, statuses, token):
            if slug == "metaculus-cup":
                raise requests.HTTPError("404 Client Error: Not Found")
            return {"closed": [_post(601, _question(61, scheduled="2026-08-14T00:00:00Z"))]}

        monkeypatch.setattr(supply_probe, "fetch_posts_by_status", _fake_fetch)

        supplies = probe_slugs(["metaculus-cup", "summer-futureeval-2026"], ("closed",), "token", now=NOW)

        assert [s.slug for s in supplies] == ["metaculus-cup", "summer-futureeval-2026"]
        assert supplies[0].error is not None
        assert "404" in supplies[0].error
        assert supplies[0].status_counts == ()
        assert supplies[1].error is None
        assert [row.question_id for row in supplies[1].backlog] == [61]


class TestMain:
    """End-to-end CLI: argument parsing, preflight ordering, rendering, the JSON dump.

    ``main`` reads the real clock, so the overdue arithmetic is pinned in ``TestBacklog``
    against an injected ``now``; the assertions here are on shape and wiring.
    """

    def _run(self, monkeypatch, *, extra_argv=()) -> None:
        posts = {
            "open": [_post(701, _question(71, forecast=True))],
            "closed": [
                _post(
                    702,
                    _question(72, scheduled="2026-08-14T00:00:00Z", forecast=False),
                    title="Drilling rigs?",
                )
            ],
            "resolved": [_post(703, _question(73, actual="2026-08-01T00:00:00Z", resolution="7", forecast=True))],
        }
        monkeypatch.setattr(supply_probe, "fetch_posts_by_status", lambda slug, statuses, token: posts)
        monkeypatch.setattr(supply_probe, "verify_metaculus_api_identity", lambda: None)
        monkeypatch.setenv("METACULUS_TOKEN", "token")
        monkeypatch.setattr("sys.argv", ["supply_probe", "--slugs", "summer-futureeval-2026", *extra_argv])

        supply_probe.main()

    def test_main_prints_the_partition(self, monkeypatch, capsys):
        self._run(monkeypatch)
        out = capsys.readouterr().out

        assert "summer-futureeval-2026" in out
        assert "closed" in out
        assert "Unresolved past scheduled_resolve_time: 1" in out
        assert "Drilling rigs?" in out

    def test_main_sweeps_forfeits_by_default(self, monkeypatch, capsys):
        """The list payloads here already carry ``my_forecasts``, so the default sweep runs
        for real and issues no request — which is also the cheap-path assertion."""
        self._run(monkeypatch)
        out = capsys.readouterr().out

        assert "Closed/resolved questions never forecast by the bot: 1 (of 2; forecast 1, unknown 0)" in out

    def test_no_forfeits_skips_the_resolver_entirely(self, monkeypatch, capsys):
        monkeypatch.setattr(
            supply_probe,
            "resolve_bot_forecasts",
            lambda *_a, **_k: pytest.fail("--no-forfeits must not resolve forecasts"),
        )

        self._run(monkeypatch, extra_argv=["--no-forfeits"])
        out = capsys.readouterr().out

        # The flag only skips the RESOLVER; these fixtures already answer from their list
        # payload, so the count still lands. On real list pages it would read all-unknown.
        assert "never forecast by the bot: 1" in out

    def test_main_writes_the_json_dump(self, monkeypatch, tmp_path, capsys):
        dump = tmp_path / "supply.json"
        self._run(monkeypatch, extra_argv=["--output", str(dump)])
        capsys.readouterr()

        payload = json.loads(dump.read_text())
        assert payload["slugs"][0]["slug"] == "summer-futureeval-2026"
        assert payload["slugs"][0]["backlog"][0]["question_id"] == 72
        assert {c["status"]: c["questions"] for c in payload["slugs"][0]["status_counts"]} == {
            "open": 1,
            "closed": 1,
            "resolved": 1,
        }

    def test_missing_token_exits_before_any_request(self, monkeypatch):
        monkeypatch.delenv("METACULUS_TOKEN", raising=False)
        monkeypatch.setattr(
            supply_probe, "verify_metaculus_api_identity", lambda: pytest.fail("preflight ran without a token")
        )
        monkeypatch.setattr("sys.argv", ["supply_probe", "--slugs", "slug"])

        with pytest.raises(SystemExit):
            supply_probe.main()

    def test_identity_preflight_runs_before_the_token_pull(self, monkeypatch, capsys):
        calls: list[str] = []
        # forecast= is set so the default forfeit sweep answers off this payload and issues no
        # detail GET; this test is about ordering, not about the sweep.
        posts = {"closed": [_post(801, _question(81, forecast=True))]}

        def _fetch(slug, statuses, token):
            calls.append("fetch")
            return posts

        monkeypatch.setattr(supply_probe, "verify_metaculus_api_identity", lambda: calls.append("preflight"))
        monkeypatch.setattr(supply_probe, "fetch_posts_by_status", _fetch)
        monkeypatch.setenv("METACULUS_TOKEN", "token")
        monkeypatch.setattr("sys.argv", ["supply_probe", "--slugs", "slug", "--statuses", "closed"])

        supply_probe.main()
        capsys.readouterr()

        assert calls == ["preflight", "fetch"]


class TestDefaults:
    def test_default_slugs_come_from_the_repo_constants(self):
        assert TOURNAMENT_ID in supply_probe.DEFAULT_SLUGS
        assert FALL_CUP_SLUG in supply_probe.DEFAULT_SLUGS
        assert METACULUS_CUP_ID in supply_probe.DEFAULT_SLUGS
        assert len(set(supply_probe.DEFAULT_SLUGS)) == len(supply_probe.DEFAULT_SLUGS)

    def test_closed_is_a_default_status(self):
        assert "closed" in supply_probe.DEFAULT_STATUSES

    def test_probe_url_shares_the_host_the_preflight_vets(self):
        # The identity guard's promise is that the vetted host is the host the token goes
        # to, so a hardcoded probe URL would quietly break it under a base-URL override.
        assert api_preflight.preflight_url().startswith(supply_probe.POSTS_URL)

    def test_an_override_loaded_from_a_dotenv_file_moves_both_urls_together(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failure this pair is built to exclude: vetting one host and sending the token
        to another. It was live until 2026-09 — `api_preflight` bound its URL as a module
        constant, so an override arriving with the .env files (nothing loads `.env.local` at
        import time) moved POSTS_URL and left the vetted host at www.metaculus.com. Simulated
        here by setting the variable AFTER both modules are imported, which is exactly what a
        late dotenv load looks like from the modules' point of view."""
        monkeypatch.setenv("METACULUS_API_BASE_URL", "https://staging.example.invalid/api")
        posts_url = f"{MetaculusClient().base_url}/posts/"

        assert posts_url.startswith("https://staging.example.invalid/api")
        assert api_preflight.preflight_url().startswith(posts_url)
