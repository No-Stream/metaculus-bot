"""Tier-2 Datawrapper hop tests (`resolution_source.py` + the `http_fetch` route).

The hop exists because poll trackers lock their resolving daily series inside
Datawrapper iframes (qids 44858 / 44841, 2026-08-24 dossiers): trafilatura
drops the embeds, and the page-pinned versioned dataset route serves
months-stale snapshots as HTTP 200. These tests pin the load-bearing
properties:

- the hop fetches ONLY the version-free live route
  (``static.dwcdn.net/data/<id>.csv``); the naive
  ``datawrapper.dwcdn.net/<id>/<version>/dataset.csv`` form is NEVER requested,
- a dataset that fails the Last-Modified freshness guard is withheld
  (``stale_data``) instead of being served as live,
- CSV truncation keeps the MOST RECENT rows (the tail), not the head.

Network plumbing (FakeSession / FakeResponse) is shared with the Tier-1 suite.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import MagicMock

import aiohttp
import pytest

from metaculus_bot.research import resolution_source
from metaculus_bot.research.http_fetch import DatawrapperChartRef
from metaculus_bot.research.provider_diagnostics import pop_provider_detail
from metaculus_bot.research.resolution_source import (
    _fetch_datawrapper_dataset,
    _truncate_csv_middle,
    _truncate_with_marker,
    fetch_resolution_sources,
    format_resolution_sections,
    resolution_source_provider,
)
from tests.test_resolution_source_provider import FakeResponse, FakeSession

PAGE_URL = "https://tracker.example.com/polls"
CHART_ID = "T3st1"
DATASET_URL = f"https://static.dwcdn.net/data/{CHART_ID}.csv"


@pytest.fixture(autouse=True)
def _stub_public_dns(monkeypatch):
    """Same rationale as the Tier-1 suite: test hostnames have no real DNS, so
    the SSRF preflight would classify everything ``ssrf_blocked`` without a
    public-IP stub (autouse fixtures don't cross module boundaries)."""

    def _sync_ainfo(host, port, *args, **kwargs):
        del host, port, args, kwargs
        return [(0, 0, 0, "", ("8.8.8.8", 0))]

    monkeypatch.setattr(resolution_source.socket, "getaddrinfo", _sync_ainfo)


def _http_date(dt: datetime) -> str:
    return format_datetime(dt, usegmt=True)


def _fresh_last_modified() -> str:
    return _http_date(datetime.now(timezone.utc) - timedelta(hours=2))


def _stale_last_modified() -> str:
    bound = resolution_source.RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS
    return _http_date(datetime.now(timezone.utc) - timedelta(days=bound + 15))


def _tracker_page_html(*chart_ids: str) -> bytes:
    """A tracker-shaped page: real article prose (so trafilatura succeeds)
    plus one Substack-escaped Datawrapper embed per chart id, each pinning a
    STALE version — the exact shape of the natesilver.net trackers."""
    embeds = "".join(
        f'<div id="datawrapper-iframe" data-attrs="{{&quot;url&quot;:&quot;'
        f"https://datawrapper.dwcdn.net/{cid}/11/&quot;,"
        f'&quot;height&quot;:527,&quot;title&quot;:&quot;Tracker chart {cid}&quot;}}"></div>'
        for cid in chart_ids
    )
    return (
        "<!doctype html><html><head><title>Poll tracker</title></head><body>"
        "<article><h1>The latest on the tracker</h1>"
        "<p>Updated recently. As of today, just 33 percent of Americans support "
        "the conflict, while about 59 percent oppose it. The chart below is "
        "updated whenever new qualifying polls are released, and the modeled "
        "average weights each pollster by sample size and track record.</p>"
        f"{embeds}"
        "<p>Methodology: polls are adjusted for house effects and recency; the "
        "downloadable data under the chart takes precedence over the prose.</p>"
        "</article></body></html>"
    ).encode("utf-8")


def _csv_body(n_rows: int) -> str:
    lines = ["modeldate,approve,disapprove"]
    lines.extend(f"day-{i:04d},38.5,57.9" for i in range(n_rows))
    return "\n".join(lines) + "\n"


def _csv_response(
    body: str,
    *,
    last_modified: str | None,
    content_type: str = "text/csv",
) -> FakeResponse:
    headers = {"Last-Modified": last_modified} if last_modified is not None else {}
    return FakeResponse(200, body=body.encode("utf-8"), content_type=content_type, headers=headers)


class TestTruncateCsvMiddle:
    """Truncation keeps the header + BOTH ends of the rows. The newest rows —
    the resolution-relevant ones — sit at the END of the tracker model-average
    series but at the START of the poll-input tables on the same pages
    (observed live on both natesilver.net trackers, 2026-08-25), so keeping
    both ends is what makes the newest rows survive regardless of ordering."""

    def test_under_cap_is_identity(self):
        text = _csv_body(5)
        assert _truncate_csv_middle(text, 10_000, DATASET_URL) == text

    def test_keeps_header_and_both_ends(self):
        text = _csv_body(500)
        out = _truncate_csv_middle(text, 800, DATASET_URL)
        assert len(out) <= 800
        assert out.startswith("modeldate,approve,disapprove")
        assert "day-0499,38.5,57.9" in out  # ascending series: newest row survives
        assert "day-0000,38.5,57.9" in out  # descending series: its newest row survives too
        assert "day-0250" not in out  # the middle is what gets cut
        assert "middle rows omitted" in out
        assert DATASET_URL in out

    def test_omitted_count_is_accurate(self):
        text = _csv_body(100)
        out = _truncate_csv_middle(text, 400, DATASET_URL)
        kept = sum(1 for line in out.split("\n") if line.startswith("day-"))
        assert f"[... {100 - kept} middle rows omitted" in out

    def test_single_long_line_degrades_to_head_truncation(self):
        text = "x" * 5000
        out = _truncate_csv_middle(text, 300, DATASET_URL)
        assert len(out) <= 300
        assert out.startswith("xxx")

    def test_tiny_cap_degrades_without_exceeding(self):
        text = _csv_body(50)
        out = _truncate_csv_middle(text, 60, DATASET_URL)
        assert len(out) <= 60

    def test_rows_too_long_for_the_budget_degrade_to_head_truncation(self):
        # A CSV whose individual rows each exceed the row budget (wide poll
        # tables with dozens of pollster columns): no row fits either end, so
        # the both-ends path has nothing to keep and falls back to plain head
        # truncation rather than emitting a header-plus-marker stub with no data.
        text = "date,value\n" + "\n".join("x" * 200 for _ in range(5)) + "\n"
        out = _truncate_csv_middle(text, 150, DATASET_URL)
        assert len(out) <= 150
        assert out.startswith("date,value")
        assert "middle rows omitted" not in out
        assert "truncated at 150 chars" in out

    @pytest.mark.parametrize("cap", [40, 90, 150, 300, 800, 2_000, 6_500])
    @pytest.mark.parametrize("n_rows", [4, 17, 500])
    def test_cap_invariant_holds_across_shapes(self, cap: int, n_rows: int):
        """The documented invariant (``len(return) <= cap``) across every branch:
        the reserved marker width is computed at its worst case, so a drift in
        that arithmetic is the way this function starts overrunning the per-URL
        budget it exists to respect."""
        out = _truncate_csv_middle(_csv_body(n_rows), cap, DATASET_URL)
        assert len(out) <= cap

    @pytest.mark.parametrize("cap", [0, -1, -400])
    def test_a_nonpositive_cap_yields_nothing_rather_than_the_whole_text(self, cap: int):
        """A zero-or-negative budget renders nothing.

        The caller's arithmetic (section allowance minus a lead line minus a very long parent
        URL) can go negative, and ``text[:cap]`` on a negative cap returns nearly the WHOLE
        text while the documented invariant claims a bound — so a zero-budget slot would have
        rendered a full page.
        """
        assert _truncate_csv_middle(_csv_body(500), cap, DATASET_URL) == ""
        assert _truncate_with_marker("x" * 5_000, cap, DATASET_URL) == ""


class TestDatawrapperHop:
    async def test_page_plus_live_dataset_happy_path(self, monkeypatch):
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(20), last_modified=_fresh_last_modified()),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL])

        assert [r.status for r in results] == ["success", "success"]
        page, dataset = results
        # Dataset result sits directly after its parent page.
        assert dataset.url == DATASET_URL
        assert dataset.chart_id == CHART_ID
        assert dataset.parent_url == PAGE_URL
        assert dataset.data_last_modified is not None
        # Provenance line names the chart, its title, the parent page, and the
        # publish timestamp; the CSV rows follow.
        assert f"Datawrapper chart {CHART_ID}" in dataset.text
        assert f"Tracker chart {CHART_ID}" in dataset.text
        assert PAGE_URL in dataset.text
        assert "Dataset published" in dataset.text
        assert "day-0019,38.5,57.9" in dataset.text
        # The page result carries the discovered chart refs.
        assert [c.chart_id for c in page.datawrapper_charts] == [CHART_ID]

    async def test_naive_versioned_route_is_never_requested(self, monkeypatch):
        """THE route pin. The page HTML pins `datawrapper.dwcdn.net/<id>/11/`;
        appending `/dataset.csv` to that returns an HTTP-200 snapshot that was
        5-14 MONTHS stale on the two real trackers (2026-08-24 verifications).
        Only the version-free static route may ever be fetched."""
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(20), last_modified=_fresh_last_modified()),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        await fetch_resolution_sources([PAGE_URL])

        dataset_requests = [u for u in session.requested if u != PAGE_URL]
        assert dataset_requests == [DATASET_URL]
        for url in session.requested:
            assert "datawrapper.dwcdn.net" not in url, f"naive versioned host requested: {url}"
            assert "dataset.csv" not in url, f"naive dataset.csv route requested: {url}"
            assert "/11/" not in url, f"page-pinned version requested: {url}"

    async def test_stale_dataset_is_withheld_not_served(self, monkeypatch):
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(20), last_modified=_stale_last_modified()),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL])

        dataset = results[1]
        assert dataset.status == "stale_data"
        assert dataset.text == ""  # never rendered — stale-as-live is the failure this guards
        assert dataset.data_last_modified is not None
        # The formatter surfaces the withholding on the datasets' OWN line — a
        # chart CSV is not a cited resolution source, so it must not ride the
        # "cited resolution source(s) could not be fetched" wording — and the
        # (fresh) page content still renders.
        out = format_resolution_sections(results, datetime.now(timezone.utc))
        assert "[1 embedded chart dataset(s) not served (stale_data)" in out
        assert "could not be fetched" not in out
        assert "day-0019" not in out
        assert "updated whenever new qualifying polls" in out

    async def test_missing_last_modified_is_withheld(self, monkeypatch):
        # No Last-Modified header -> freshness unverifiable -> withheld
        # (serve live or nothing; a dataset of unknown age must not pose as live).
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(20), last_modified=None),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL])
        assert results[1].status == "stale_data"
        assert results[1].text == ""
        assert results[1].data_last_modified is None

    async def test_malformed_last_modified_is_withheld(self, monkeypatch):
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(20), last_modified="not a date"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL])
        assert results[1].status == "stale_data"
        assert results[1].text == ""

    async def test_a_future_dated_dataset_is_withheld_not_treated_as_freshest(self, monkeypatch):
        """The freshness check is two-sided. A ``Last-Modified`` in the future means a broken
        clock or a misparse on one side, which makes it unusable as the publication date the
        lead asserts — the one-sided check passed any future date as maximally fresh."""
        tomorrow = _http_date(datetime.now(timezone.utc) + timedelta(days=1))
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(20), last_modified=tomorrow),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL])

        assert results[1].status == "stale_data"
        assert results[1].text == ""

    async def test_a_dataset_inside_the_clock_skew_tolerance_is_still_served(self, monkeypatch):
        # Ordinary CDN/host clock skew must not cost us a live dataset, so the future side
        # carries a small tolerance rather than a hard `> now` bound.
        skewed = _http_date(datetime.now(timezone.utc) + timedelta(minutes=30))
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(20), last_modified=skewed),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL])

        assert results[1].status == "success"

    async def test_dataset_just_inside_bound_is_served(self, monkeypatch):
        bound = resolution_source.RESOLUTION_SOURCE_DATAWRAPPER_MAX_AGE_DAYS
        recent = _http_date(datetime.now(timezone.utc) - timedelta(days=bound - 1))
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(20), last_modified=recent),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL])
        assert results[1].status == "success"

    async def test_js_walled_page_still_hops(self, monkeypatch):
        # A JS-walled tracker (SPA shell) exposes its embeds in raw HTML even
        # though extraction fails — the hop can rescue the data the page hides.
        shell = (
            '<!doctype html><html><body><div id="root"></div>'
            '<div data-attrs="{&quot;url&quot;:&quot;'
            f"https://datawrapper.dwcdn.net/{CHART_ID}/11/&quot;,"
            '&quot;title&quot;:&quot;Walled chart&quot;}"></div></body></html>'
        ).encode("utf-8")
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=shell, content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(20), last_modified=_fresh_last_modified()),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL])
        assert [r.status for r in results] == ["js_wall", "success"]
        out = format_resolution_sections(results, datetime.now(timezone.utc))
        assert "day-0019,38.5,57.9" in out
        assert "js_wall" in out  # the walled page is still reported as unreachable

    async def test_chart_cap_bounds_dataset_fetches(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_DATAWRAPPER_MAX_CHARTS", 2)
        page = _tracker_page_html("Aaaa1", "Bbbb2", "Cccc3", "Dddd4")
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=page, content_type="text/html"),
                "https://static.dwcdn.net/data/": _csv_response(_csv_body(5), last_modified=_fresh_last_modified()),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL])

        dataset_urls = [u for u in session.requested if u.startswith("https://static.dwcdn.net/")]
        # Document order, capped at 2 — the hero/resolving chart comes first.
        assert dataset_urls == [
            "https://static.dwcdn.net/data/Aaaa1.csv",
            "https://static.dwcdn.net/data/Bbbb2.csv",
        ]
        assert len(results) == 3  # page + 2 datasets
        # Per-host politeness holds on the CDN host too.
        assert session.host_peak["static.dwcdn.net"] == 1

    async def test_same_chart_on_two_pages_fetched_once(self, monkeypatch):
        other_page = "https://other.example.com/mirror"
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                other_page: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(5), last_modified=_fresh_last_modified()),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL, other_page])

        assert [u for u in session.requested if u == DATASET_URL] == [DATASET_URL]
        assert len(results) == 3  # 2 pages + 1 dataset

    async def test_dataset_404_maps_to_not_found(self, monkeypatch):
        # Publishers can disable "Get the data"; the CDN then 403/404s. Fail
        # visibly, never substitute another route.
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: FakeResponse(404, body=b"", content_type="text/plain"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL])
        assert results[1].status == "not_found"
        assert results[1].chart_id == CHART_ID

    async def test_dataset_client_error_maps_to_error(self, monkeypatch):
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: aiohttp.ClientError("boom"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL])
        assert results[1].status == "error"
        assert results[1].text == ""

    async def test_long_dataset_keeps_most_recent_rows(self, monkeypatch):
        # Datasets truncate against their OWN cap, not the page cap — the dataset
        # allowance is what keeps a chart's rows from evicting cited page text.
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS", 900)
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(500), last_modified=_fresh_last_modified()),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL])
        dataset = results[1]
        assert dataset.status == "success"
        assert len(dataset.text) <= 900
        # Both ends survive so the newest rows are kept whether the series is
        # ascending (tracker averages) or descending (poll tables); only the
        # middle is dropped.
        assert "day-0499,38.5,57.9" in dataset.text
        assert "day-0000,38.5,57.9" in dataset.text
        assert "day-0250" not in dataset.text
        assert "middle rows omitted" in dataset.text

    async def test_no_charts_no_extra_fetches(self, monkeypatch):
        plain = (
            b"<!doctype html><html><body><article><p>"
            b"A perfectly ordinary resolution source page with plenty of prose "
            b"content and no embedded interactive charts of any kind at all."
            b"</p></article></body></html>"
        )
        session = FakeSession({PAGE_URL: FakeResponse(200, body=plain, content_type="text/html")})
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL])
        assert len(results) == 1
        assert session.requested == [PAGE_URL]

    async def test_two_pages_each_keep_their_own_dataset_adjacent(self, monkeypatch):
        """Interleaving is per-parent, not append-at-the-end: each dataset renders
        directly after the page that embeds it, so a forecaster reading the section
        (and the total-budget trimming that walks it in order) never has to guess
        which page a CSV belongs to."""
        other_page = "https://mirror.example.com/tracker"
        other_chart = "Zz9Yy"
        other_dataset = f"https://static.dwcdn.net/data/{other_chart}.csv"
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                other_page: FakeResponse(200, body=_tracker_page_html(other_chart), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(5), last_modified=_fresh_last_modified()),
                other_dataset: _csv_response(_csv_body(5), last_modified=_fresh_last_modified()),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL, other_page])

        assert [r.url for r in results] == [PAGE_URL, DATASET_URL, other_page, other_dataset]
        assert [r.parent_url for r in results] == [None, PAGE_URL, None, other_page]

    async def test_wall_clock_timeout_drains_a_hanging_dataset_fetch(self, monkeypatch):
        """The F5 teardown guard, now that dataset tasks join the cancel list: when
        the hop's budget expires mid-fetch, the in-flight dataset request must settle
        BEFORE the session closes. Closing the session first is what yanks transports
        out from under live requests (aiohttp then logs transport-closed tracebacks
        and can leak connections), so the event ordering — not just the final state —
        is the invariant.

        Since the hop gained its own wall bound, a hanging dataset no longer takes
        the whole provider down: the inner timeout fires first and the provider
        serves the Tier-1 page it already fetched — the pre-bound behavior (`out ==
        ""`, every fetched page discarded) is exactly the regression the bound
        removed. The wall/margin/floor are scaled down together, preserving the
        production relationship (margin > 0 is what makes the inner bound fire
        before the outer wall) at test speed; the skip branch is tested separately."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_WALL_TIMEOUT", 0.4)
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_DATAWRAPPER_HOP_WALL_MARGIN_S", 0.3)
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_DATAWRAPPER_MIN_HOP_BUDGET_S", 0.0)
        events: list[str] = []

        class _HangingResponse(FakeResponse):
            async def read(self) -> bytes:
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    events.append("dataset-settled")
                    raise
                raise AssertionError("unreachable: the wall-clock timeout should cancel this read")

        class _EventSession(FakeSession):
            async def close(self) -> None:
                events.append("session-closed")
                await super().close()

        session = _EventSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _HangingResponse(200, body=b"", content_type="text/csv"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        q = _mock_question(f"Resolves per the tracker at {PAGE_URL}.")
        out = await resolution_source_provider(is_benchmarking=False)(q)

        # The hop's own bound fired, so Tier-1 survives the hanging CDN fetch.
        assert f"### {PAGE_URL}" in out
        assert DATASET_URL not in out
        assert DATASET_URL in session.requested  # the hop did start
        assert events == ["dataset-settled", "session-closed"]
        assert session.host_inflight["static.dwcdn.net"] == 0  # the request's context manager exited
        assert session.closed

    async def test_hop_is_skipped_when_the_wall_budget_is_nearly_spent(self, monkeypatch, caplog):
        """Below the hop floor there is no room for even one typical CDN fetch, so
        the hop must not start at all — the pages in hand are worth more than an
        attempt that cannot land and could push the provider past its wall."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_WALL_TIMEOUT", 0.05)
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(5), last_modified=_fresh_last_modified()),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        with caplog.at_level(logging.WARNING):
            q = _mock_question(f"Resolves per the tracker at {PAGE_URL}.")
            out = await resolution_source_provider(is_benchmarking=False)(q)

        assert f"### {PAGE_URL}" in out
        assert DATASET_URL not in session.requested  # never started
        assert any("skipping the datawrapper hop" in message for message in caplog.messages)


class TestDatawrapperHopFailureModes:
    """Per-status behavior of the hop itself, driven directly so each failure
    mode is pinned in isolation. Every one must return an empty ``text`` while
    keeping the chart provenance, so a failed hop shows up in the diagnostics
    ``lost=`` segment instead of vanishing."""

    @staticmethod
    def _chart() -> DatawrapperChartRef:
        return DatawrapperChartRef(chart_id=CHART_ID, title="Tracker chart")

    async def test_cdn_host_resolving_private_is_ssrf_blocked_before_any_request(self, monkeypatch):
        """The hop constructs its own URL, so it gets the SAME preflight as a cited
        page — no CDN allowlist exemption. A poisoned/rebinding answer for
        static.dwcdn.net must be refused before a request is issued."""

        def _private_ainfo(host, port, *args, **kwargs):
            del host, port, args, kwargs
            return [(0, 0, 0, "", ("127.0.0.1", 0))]

        monkeypatch.setattr(resolution_source.socket, "getaddrinfo", _private_ainfo)
        session = FakeSession({})  # no handlers: any request would raise

        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})

        assert result.status == "ssrf_blocked"
        assert result.text == ""
        assert result.http_status is None
        assert session.requested == []
        assert result.chart_id == CHART_ID
        assert result.parent_url == PAGE_URL

    @pytest.mark.parametrize("status", [403, 406, 429])
    async def test_refused_dataset_maps_to_blocked(self, status: int):
        """Publishers can disable "Get the data"; the CDN then refuses or throttles.
        Distinguish that from a missing chart so the loss token names the cause."""
        session = FakeSession({DATASET_URL: FakeResponse(status, body=b"", content_type="text/html")})
        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})
        assert result.status == "blocked"
        assert result.text == ""
        assert result.http_status == status

    async def test_gone_maps_to_not_found(self):
        session = FakeSession({DATASET_URL: FakeResponse(410, body=b"", content_type="text/html")})
        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})
        assert result.status == "not_found"

    async def test_server_error_maps_to_error(self):
        session = FakeSession({DATASET_URL: FakeResponse(503, body=b"", content_type="text/html")})
        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})
        assert result.status == "error"
        assert result.http_status == 503

    async def test_redirect_is_not_followed_and_maps_to_error(self):
        """Redirects are unexpected on this CDN, and following one is how a hop
        would silently land on some other host's CSV. `allow_redirects=False`
        means a 3xx surfaces as an error rather than a fetched dataset."""
        session = FakeSession(
            {
                DATASET_URL: FakeResponse(
                    302,
                    body=b"",
                    content_type="text/html",
                    headers={"Location": "https://elsewhere.example.com/other.csv"},
                )
            }
        )
        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})
        assert result.status == "error"
        assert result.http_status == 302
        assert session.requested == [DATASET_URL]  # the Location was never fetched

    async def test_oversize_dataset_is_dropped_as_error_not_served(self, monkeypatch):
        """The body cap fires BEFORE the freshness check, so a huge CSV is an
        `error` (nothing readable) rather than a truncated half-dataset presented
        as the live series."""
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_MAX_RESPONSE_BYTES", 100)
        session = FakeSession(
            {DATASET_URL: _csv_response(_csv_body(200), last_modified=_fresh_last_modified())},
        )
        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})
        assert result.status == "error"
        assert result.text == ""
        assert result.chart_id == CHART_ID

    async def test_timeout_maps_to_error(self):
        session = FakeSession({DATASET_URL: asyncio.TimeoutError()})
        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})
        assert result.status == "error"
        assert result.http_status is None

    @pytest.mark.parametrize("body", ["", "   \n\n", "﻿"])
    async def test_an_empty_dataset_body_is_withheld_rather_than_stamped_live(self, body: str):
        """The lead asserts `Live "Get the data" dataset … Dataset published <ts>` off the
        Last-Modified header alone. With no content check, an empty CDN body rendered that
        authoritative freshness claim over nothing at all — structurally the same defect as a venue
        quoting a manufactured price for an empty book."""
        session = FakeSession({DATASET_URL: _csv_response(body, last_modified=_fresh_last_modified())})

        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})

        assert result.status == "empty_body"
        assert result.text == ""
        assert result.chart_id == CHART_ID  # provenance survives, so the loss is attributable

    @pytest.mark.parametrize(
        "body",
        [
            "<!DOCTYPE html>\n<html><head><title>404, not found</title></head>\n<body>gone</body>\n",
            "modeldate,approve,disapprove\n",
            "no delimiters here\njust prose\n",
        ],
    )
    async def test_a_body_that_is_not_row_shaped_is_withheld(self, body: str):
        """A soft-404 page, a header with no rows, and prose. None of them is the chart's live
        series, so none may be served under the liveness lead. The HTML case is why markup is
        rejected outright: an error page carries a comma easily enough to pass a delimiter test."""
        session = FakeSession({DATASET_URL: _csv_response(body, last_modified=_fresh_last_modified())})

        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})

        assert result.status == "unsupported_type"
        assert result.text == ""

    async def test_the_content_check_outranks_the_freshness_verdict(self):
        """Order matters for the DIAGNOSTICS, not just the render: `stale_data` maps to the benign
        `none` token (the freshness guard working as designed), so an empty body reported under it
        would borrow that amnesty and hide a real CDN failure."""
        session = FakeSession({DATASET_URL: _csv_response("", last_modified=_stale_last_modified())})

        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})

        assert result.status == "empty_body"

    async def test_a_bomless_utf16_dataset_is_refused_rather_than_served_as_mojibake(self):
        """`errors="replace"` turned an oddly-encoded CSV into `0<?>.<?>4<?>2<?>` and served it as
        the live resolving series. Excel-exported poll tables are exactly the UTF-16 shape."""
        body = "date,value\n2026-08-01,0.42\n".encode("utf-16-le")
        session = FakeSession(
            {
                DATASET_URL: FakeResponse(
                    200,
                    body=body,
                    content_type="text/csv",
                    headers={"Last-Modified": _fresh_last_modified()},
                )
            }
        )

        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})

        assert result.status == "unsupported_type"

    async def test_a_bomd_utf16_dataset_decodes_and_is_served(self):
        """The other half: honouring the BOM means the same body is USABLE rather than lost."""
        body = "date,value\n2026-08-01,0.42\n"
        session = FakeSession(
            {
                DATASET_URL: FakeResponse(
                    200,
                    body=body.encode("utf-16"),
                    content_type="text/csv",
                    headers={"Last-Modified": _fresh_last_modified()},
                )
            }
        )

        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})

        assert result.status == "success"
        assert "2026-08-01,0.42" in result.text

    async def test_octet_stream_content_type_is_still_served(self):
        """Content-Type is deliberately not gated: the CDN labels the same CSV
        bytes `application/octet-stream` on some routes, and gating on it would
        throw away the live dataset for a header cosmetic."""
        session = FakeSession(
            {
                DATASET_URL: _csv_response(
                    _csv_body(5), last_modified=_fresh_last_modified(), content_type="application/octet-stream"
                )
            }
        )
        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})
        assert result.status == "success"
        assert "day-0004,38.5,57.9" in result.text


def _poll_table_csv(n_rows: int, *, tagged: bool) -> str:
    """A poll-input table shaped like the live VUUVz dataset (2026-08-26 receipts).

    Each pollster cell is a styled anchor whose inner text is the pollster name — the shape that
    made 69% of that CSV's 33k chars tag markup. ``tagged=False`` is the same table with the
    markup already gone, i.e. what the strip should leave behind.
    """
    lines = ["Dates,Pollster,Sample,Approve,Disapprove"]
    for i in range(n_rows):
        name = f"Pollster {i:03d}"
        cell = (
            f"<a href='https://pollster{i:03d}.example.com/august-2026-national-poll/'"
            f"style='color:#000000; text-decoration: underline;'target='_blank' "
            f"rel='nofollow noopener'>{name}</a>"
            if tagged
            else name
        )
        lines.append(f'"8/{(i % 28) + 1}, 2026",{cell},"1,000 LV",36.4,49.2')
    return "\n".join(lines) + "\n"


class TestDatasetMarkupStripping:
    """The Tier-2 budget fix: strip markup BEFORE truncation so the char budget buys poll rows.

    On exactly the poll-tracker questions the hop was built for, two-thirds of the visible
    evidence was being spent on `<a href=… style=…>` wrappers around the pollster names.
    """

    @staticmethod
    def _chart() -> DatawrapperChartRef:
        return DatawrapperChartRef(chart_id=CHART_ID, title="Polls included in our average")

    @staticmethod
    def _kept_rows(text: str) -> int:
        return sum(1 for line in text.split("\n") if line.startswith('"8/'))

    async def test_the_pollster_names_survive_and_the_markup_does_not(self):
        session = FakeSession(
            {DATASET_URL: _csv_response(_poll_table_csv(5, tagged=True), last_modified=_fresh_last_modified())}
        )

        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})

        assert result.status == "success"
        assert "Pollster 000" in result.text and "Pollster 004" in result.text
        assert "<a " not in result.text
        assert "style=" not in result.text
        assert "nofollow" not in result.text

    async def test_more_rows_survive_the_same_budget_once_the_markup_is_gone(self, monkeypatch):
        """The measured claim, pinned: at the live run's actual 2,853-char budget, 9 rows survived
        with tags against 30 with them stripped. Here the same comparison runs against the
        untouched truncator, so a regression that stopped stripping shows up as fewer rows."""
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_DATAWRAPPER_PER_DATASET_MAX_CHARS", 3_000)
        tagged = _poll_table_csv(60, tagged=True)
        session = FakeSession({DATASET_URL: _csv_response(tagged, last_modified=_fresh_last_modified())})

        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})

        naive_kept = self._kept_rows(_truncate_csv_middle(tagged.strip(), 2_800, DATASET_URL))
        stripped_kept = self._kept_rows(result.text)
        assert stripped_kept > naive_kept, f"{stripped_kept} rows vs {naive_kept} with markup"
        assert stripped_kept >= 2 * naive_kept, "the whole point is a multiple, not a rounding win"
        assert len(result.text) <= 3_000

    async def test_a_numeric_dataset_is_untouched_by_the_strip(self):
        """The two numeric tracker CSVs on the same pages contain zero `<` characters, so the strip
        must be a provable no-op there rather than a transformation that merely looks harmless."""
        session = FakeSession({DATASET_URL: _csv_response(_csv_body(20), last_modified=_fresh_last_modified())})

        result = await _fetch_datawrapper_dataset(session, self._chart(), PAGE_URL, {})

        assert result.text.endswith(_csv_body(20).strip())


def _mock_question(criteria: str) -> MagicMock:
    q = MagicMock()
    q.resolution_criteria = criteria
    q.fine_print = ""
    q.id_of_question = 998
    q.question_text = "tracker question"
    q.page_url = "https://metaculus.com/q/998"
    return q


class TestProviderEndToEnd:
    async def test_dataset_section_rendered_through_provider(self, monkeypatch):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(20), last_modified=_fresh_last_modified()),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        q = _mock_question(f"Resolves per the tracker at {PAGE_URL} — the CSV under the chart.")
        out = await resolution_source_provider(is_benchmarking=False)(q)

        assert f"### {PAGE_URL}" in out
        assert f"### {DATASET_URL}" in out
        assert "day-0019,38.5,57.9" in out
        assert f"Tracker chart {CHART_ID}" in out
        # Diagnostics record both the page and the hop outcome. The dataset is
        # keyed by chart id, not netloc: every dataset shares one CDN host, so
        # domain keys would collapse multiple charts into `static.dwcdn.net#N`.
        sources = pop_provider_detail(q.id_of_question, "resolution_source")["sources"]
        assert sources["tracker.example.com"] == "ok"
        assert sources[f"datawrapper:{CHART_ID}"] == "ok"

    async def test_withheld_stale_dataset_reads_benign_in_diagnostics(self, monkeypatch):
        """A `stale_data` withhold is the freshness guard WORKING — refusing to serve
        months-old data as live — not a lost cited source. It maps to the benign
        `none` token (reached, contributed nothing) under its chart-id key, so the
        diagnostics `lost=` segment stays reserved for genuine losses; the forecaster
        still sees the withholding via the rendered dataset note."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(20), last_modified=_stale_last_modified()),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        q = _mock_question(f"Resolves per the tracker at {PAGE_URL}.")
        out = await resolution_source_provider(is_benchmarking=False)(q)

        assert "day-0019" not in out  # the stale CSV never reaches a forecaster
        assert "[1 embedded chart dataset(s) not served (stale_data)" in out
        sources = pop_provider_detail(q.id_of_question, "resolution_source")["sources"]
        assert sources["tracker.example.com"] == "ok"
        assert sources[f"datawrapper:{CHART_ID}"] == "none"

    async def test_a_failed_hop_is_still_a_visible_loss_in_diagnostics(self, monkeypatch):
        """The benign mapping is stale_data-only: a genuinely failed hop (CDN error)
        keeps its verbatim loss token — that is real signal about the CDN."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: aiohttp.ClientError("boom"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        q = _mock_question(f"Resolves per the tracker at {PAGE_URL}.")
        await resolution_source_provider(is_benchmarking=False)(q)

        sources = pop_provider_detail(q.id_of_question, "resolution_source")["sources"]
        assert sources[f"datawrapper:{CHART_ID}"] == "error"

    async def test_datasets_cannot_evict_cited_page_text(self, monkeypatch):
        """The partitioned section budget: datasets draw on their OWN allowance, so
        even a page-budget squeeze cannot be caused by chart rows — the second page's
        text renders whole alongside a full-size dataset."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        other_page = "https://mirror.example.com/tracker-two"
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                other_page: FakeResponse(200, body=_tracker_page_html("Zz9Yy"), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(500), last_modified=_fresh_last_modified()),
                "https://static.dwcdn.net/data/Zz9Yy.csv": _csv_response(
                    _csv_body(500), last_modified=_fresh_last_modified()
                ),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources([PAGE_URL, other_page])
        out = format_resolution_sections(results, datetime.now(timezone.utc))

        # Both cited pages render despite two full-size datasets interleaved
        # before/between them in the walk order.
        assert f"### {PAGE_URL}" in out
        assert f"### {other_page}" in out
        assert "additional source(s) omitted" not in out

    async def test_benchmarking_guard_covers_the_hop(self, monkeypatch):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {
                PAGE_URL: FakeResponse(200, body=_tracker_page_html(CHART_ID), content_type="text/html"),
                DATASET_URL: _csv_response(_csv_body(20), last_modified=_fresh_last_modified()),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        q = _mock_question(f"Resolves per the tracker at {PAGE_URL}.")
        out = await resolution_source_provider(is_benchmarking=True)(q)
        assert out == ""
        assert session.requested == []  # leakage guard fires before any fetch
