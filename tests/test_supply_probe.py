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
) -> dict:
    return {
        "id": qid,
        "type": qtype,
        "scheduled_resolve_time": scheduled,
        "actual_resolve_time": actual,
        "resolution": resolution,
    }


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
            "open": [_post(701, _question(71))],
            "closed": [_post(702, _question(72, scheduled="2026-08-14T00:00:00Z"), title="Drilling rigs?")],
            "resolved": [_post(703, _question(73, actual="2026-08-01T00:00:00Z", resolution="7"))],
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
        posts = {"closed": [_post(801, _question(81))]}

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
