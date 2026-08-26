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
        # The formatter surfaces the withholding as a clear status, and the
        # (fresh) page content still renders.
        out = format_resolution_sections(results, datetime.now(timezone.utc))
        assert "static.dwcdn.net: stale_data" in out
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
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_PER_URL_MAX_CHARS", 900)
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
        the provider's wall-clock timeout cancels the fetch mid-hop, the in-flight
        dataset request must settle BEFORE the session closes. Closing the session
        first is what yanks transports out from under live requests (aiohttp then
        logs transport-closed tracebacks and can leak connections), so the event
        ordering — not just the final state — is the invariant."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_WALL_TIMEOUT", 0.05)
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

        assert out == ""  # provider soft-fails on the wall-clock timeout
        assert DATASET_URL in session.requested  # the hop did start
        assert events == ["dataset-settled", "session-closed"]
        assert session.host_inflight["static.dwcdn.net"] == 0  # the request's context manager exited
        assert session.closed


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
        # Diagnostics record both the page and the hop outcome.
        sources = pop_provider_detail(q.id_of_question, "resolution_source")["sources"]
        assert sources["tracker.example.com"] == "ok"
        assert sources["static.dwcdn.net"] == "ok"

    async def test_withheld_stale_dataset_surfaces_in_diagnostics(self, monkeypatch):
        """A withheld dataset must be a VISIBLE loss, not a silent one: the hop's
        `stale_data` verdict rides into the per-URL diagnostics map as its own
        token, so the provider block reads "we found the chart and refused its
        data" rather than looking fully healthy."""
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
        sources = pop_provider_detail(q.id_of_question, "resolution_source")["sources"]
        assert sources["tracker.example.com"] == "ok"
        assert sources["static.dwcdn.net"] == "stale_data"

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
