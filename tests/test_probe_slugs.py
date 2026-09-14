"""Tests for the season-slug probe (scripts/probe_slugs.py).

The season arithmetic and the flag rules are pure and driven directly. The one network seam, the
project GET, is monkeypatched at ``probe_slugs.get_json`` (the helper it borrows from the supply
probe), so nothing here opens a socket and no request spacing is actually slept.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
import requests

from scripts import probe_slugs
from scripts.probe_slugs import (
    ProjectRow,
    candidate_slugs,
    fetch_project,
    next_season,
    probe_projects,
    render_report,
    season_of,
)
from scripts.supply_probe_platforms import DEFAULT_SLUGS

NOW = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)
BASE_URL = "https://www.metaculus.com/api"
CONFIGURED = DEFAULT_SLUGS[0]


def _project(slug: str, *, questions: int = 42, end: str = "2027-01-06T00:00:00Z", visibility: str = "normal") -> dict:
    return {
        "id": 33121,
        "name": f"Project for {slug}",
        "visibility": visibility,
        "questions_count": questions,
        "forecasting_end_date": end,
    }


def _http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"HTTP {status}", response=response)


class TestSeasonArithmetic:
    @pytest.mark.parametrize(
        ("day", "expected"),
        [
            (date(2026, 9, 12), ("fall", 2026)),
            (date(2026, 11, 30), ("fall", 2026)),
            (date(2026, 12, 5), ("winter", 2027)),
            (date(2027, 1, 20), ("winter", 2027)),
            (date(2027, 2, 28), ("winter", 2027)),
            (date(2027, 3, 1), ("spring", 2027)),
            (date(2026, 7, 1), ("summer", 2026)),
        ],
    )
    def test_a_december_winter_carries_the_following_year(self, day: date, expected: tuple[str, int]):
        assert season_of(day) == expected

    def test_the_successor_rolls_the_year_only_past_fall(self):
        assert next_season("summer", 2026) == ("fall", 2026)
        assert next_season("fall", 2026) == ("winter", 2027)
        assert next_season("winter", 2027) == ("spring", 2027)


class TestCandidates:
    def test_the_configured_slugs_come_first_and_nothing_repeats(self):
        slugs = candidate_slugs(date(2026, 9, 12))

        assert slugs[: len(DEFAULT_SLUGS)] == DEFAULT_SLUGS
        assert len(set(slugs)) == len(slugs)

    def test_this_season_and_the_next_are_generated_under_every_spelling(self):
        slugs = candidate_slugs(date(2026, 9, 12))

        assert "futureeval-fall-2026" in slugs
        assert "fall-aib-2026" in slugs
        assert "winter-futureeval-2027" in slugs
        assert "metaculus-cup-winter-2027" in slugs
        assert "spring-futureeval-2027" not in slugs

    def test_extra_slugs_ride_along_and_the_lookahead_is_adjustable(self):
        slugs = candidate_slugs(date(2026, 9, 12), seasons_ahead=2, extra=["market-pulse-26q2"])

        assert "spring-futureeval-2027" in slugs
        assert slugs[-1] == "market-pulse-26q2"

    def test_no_lookahead_generates_only_this_season(self):
        slugs = candidate_slugs(date(2026, 9, 12), seasons_ahead=0)
        assert "winter-futureeval-2027" not in slugs
        assert "fall-futureeval-2026" in slugs


class TestFetch:
    def test_a_live_project_carries_its_metadata_and_configured_flag(self, monkeypatch):
        monkeypatch.setattr(probe_slugs, "get_json", lambda *_, **__: _project(CONFIGURED))

        row = fetch_project(CONFIGURED, "token", base_url=BASE_URL)

        assert row.exists
        assert row.configured
        assert row.project_id == 33121
        assert row.questions_count == 42
        assert row.visibility == "normal"
        assert row.error is None

    def test_the_project_route_is_the_one_the_checklist_names(self, monkeypatch):
        seen: list[str] = []

        def _fake_get(params, token, *, url):
            seen.append(url)
            return _project("winter-futureeval-2027")

        monkeypatch.setattr(probe_slugs, "get_json", _fake_get)
        fetch_project("winter-futureeval-2027", "token", base_url=BASE_URL)

        assert seen == [f"{BASE_URL}/projects/tournaments/winter-futureeval-2027/"]

    def test_a_404_is_absence_and_any_other_status_leaves_existence_unknown(self, monkeypatch):
        def _raise(status):
            def _fake_get(*_, **__):
                raise _http_error(status)

            return _fake_get

        monkeypatch.setattr(probe_slugs, "get_json", _raise(404))
        absent = fetch_project("aib-winter-2027", "token", base_url=BASE_URL)
        assert absent.absent
        assert not absent.exists
        assert absent.error == "HTTP 404"

        monkeypatch.setattr(probe_slugs, "get_json", _raise(500))
        unknown = fetch_project("aib-winter-2027", "token", base_url=BASE_URL)
        assert not unknown.absent
        assert unknown.error == "HTTP 500"

    def test_a_transport_failure_reports_itself_rather_than_absence(self, monkeypatch):
        def _fake_get(*_, **__):
            raise requests.ConnectTimeout("timed out")

        monkeypatch.setattr(probe_slugs, "get_json", _fake_get)
        row = fetch_project("fall-aib-2026", "token", base_url=BASE_URL)

        assert not row.absent
        assert "timed out" in str(row.error)

    def test_every_candidate_produces_one_row(self, monkeypatch):
        monkeypatch.setattr(probe_slugs, "get_json", lambda *_, **__: _project("any"))
        monkeypatch.setattr(probe_slugs.time, "sleep", lambda _: None)

        rows = probe_projects(["a", "b", "c"], "token", base_url=BASE_URL)

        assert [row.slug for row in rows] == ["a", "b", "c"]


class TestFlags:
    def test_a_configured_slug_that_does_not_exist_is_the_loud_one(self):
        row = ProjectRow(slug=CONFIGURED, configured=True, absent=True, error="HTTP 404")
        assert row.flags(NOW) == ["CONFIGURED BUT ABSENT: every run under this slug finds no question"]

    def test_a_configured_season_past_its_forecasting_window_asks_for_a_re_point(self):
        row = ProjectRow(
            slug=CONFIGURED, configured=True, project_id=33022, forecasting_end_date="2026-09-01T00:00:00Z"
        )
        (flag,) = row.flags(NOW)
        assert "forecasting closed 2026-09-01T00:00:00Z" in flag

    def test_an_open_configured_season_is_quiet(self):
        row = ProjectRow(
            slug=CONFIGURED, configured=True, project_id=33121, forecasting_end_date="2027-01-06T00:00:00Z"
        )
        assert row.flags(NOW) == []

    def test_a_live_unconfigured_season_is_the_successor_signal(self):
        row = ProjectRow(
            slug="winter-futureeval-2027",
            configured=False,
            project_id=33200,
            questions_count=0,
            visibility="unlisted",
            forecasting_end_date="2027-04-01T00:00:00Z",
        )
        assert row.flags(NOW) == ["LIVE AND UNCONFIGURED: a season the repo's constants do not name"]

    def test_a_finished_unconfigured_season_and_a_plainly_absent_candidate_say_nothing(self):
        finished = ProjectRow(
            slug="summer-futureeval-2026",
            configured=False,
            project_id=33022,
            forecasting_end_date="2026-09-01T00:00:00Z",
        )
        assert finished.flags(NOW) == []
        assert ProjectRow(slug="aib-winter-2027", configured=False, absent=True, error="HTTP 404").flags(NOW) == []

    def test_an_unreadable_candidate_says_so_rather_than_nothing(self):
        row = ProjectRow(slug="aib-winter-2027", configured=False, error="HTTP 500")
        assert row.flags(NOW) == ["existence unknown (HTTP 500)"]


class TestRendering:
    def test_a_clean_probe_says_there_is_nothing_to_act_on(self):
        rows = [
            ProjectRow(slug=CONFIGURED, configured=True, project_id=33121, forecasting_end_date="2027-01-06T00:00:00Z"),
            ProjectRow(slug="aib-winter-2027", configured=False, absent=True, error="HTTP 404"),
        ]
        report = render_report(rows, now=NOW)

        assert "nothing to act on" in report
        assert "absent" in report
        assert "findings:" not in report

    def test_findings_name_the_slug_and_point_at_the_supply_probe_for_counts(self):
        rows = [
            ProjectRow(
                slug="winter-futureeval-2027",
                configured=False,
                project_id=33200,
                forecasting_end_date="2027-04-01T00:00:00Z",
            )
        ]
        report = render_report(rows, now=NOW)

        assert "winter-futureeval-2027: LIVE AND UNCONFIGURED" in report
        assert "make supply_probe" in report
