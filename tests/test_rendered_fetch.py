"""The shared headless-Chromium transport: its gates, its bounds, and the JSON it harvests.

The transport's SSRF half (DNS pinning, the per-request route guard) is pinned by
``tests/test_agentic_tools.py``, which drove it before it moved out of ``agentic/tools.py`` and
still owns those cases. What lives here is what the move ADDED and what the transport owns for
both callers: the XHR harvest and its bounds, the two render memos and their scoping, the
navigation budget recomputed once the gates are held, the DOM ceiling, the main-frame status,
the browser-context hardening, and the run-scoped state reset.

Nothing here launches a browser. Playwright is faked through ``sys.modules`` by the one shared
object graph in ``tests/playwright_fakes.py``, which the agentic suite drives too. Its fake page
fires its response handlers the way pyee does — call, do not await — so a listener that hands
back a coroutine leaves a detached task behind exactly as the real ``Page.on`` would, and the
harvest tests exercise that interleaving rather than a one-handler-at-a-time serialisation that
would make every race look sound.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import WebSocketRoute

from metaculus_bot.research import rendered_fetch, resolution_source
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.derived_api import DerivedEndpoint, derived_api_lead, largest_json
from metaculus_bot.research.rendered_fetch import HarvestedJson, RenderedPage
from metaculus_bot.research.resolution_fetch_result import FetchResult
from metaculus_bot.research.resolution_source import FetchContext
from scripts.telemetry.markers import MARKER_SPECS
from tests.playwright_fakes import (
    FakePage,
    FakeResponse,
    FakeWebSocketRoute,
    Faults,
    PlaywrightError,
    install_fake_playwright,
)

_PAGE_URL = "https://dashboard.example.com/senate"
_DOM = "<!doctype html><html><head><title>Dashboard</title></head><body><p>rendered</p></body></html>"
_EMPTY_DOM = "<!doctype html><html><head></head><body></body></html>"
_TIER1_SCOPE = "resolution_source"
_V2_SCOPE = "gap_fill_v2"
_RENDER_TAIL_MS = rendered_fetch.RENDER_SETTLE_MS + rendered_fetch.RENDER_DOM_READ_TIMEOUT_MS


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Run-scoped transport state around every test, plus the DNS stub ``tests/resolution_source/``
    has and this module lacked. Neither suite-level guard covers ``socket.getaddrinfo``
    (``_block_network_egress`` patches connect, ``_block_native_egress`` patches the browser and
    curl entry points), and the Tier-1 rung's re-vet of a redirected landing resolves its
    hostname through ``is_public_http_url``; without the stub that lookup leaves the process,
    and on a resolver that returns nothing for ``*.example.com`` it comes back ``ssrf_blocked``,
    so a test would assert the decline branch while looking green."""

    def _public_dns(host, port, *args, **kwargs):
        del host, port, args, kwargs
        return [(0, 0, 0, "", ("8.8.8.8", 0))]

    monkeypatch.setattr(resolution_source.socket, "getaddrinfo", _public_dns)
    rendered_fetch.reset_render_state()
    FakeResponse.reset_read_tracking()
    yield
    rendered_fetch.reset_render_state()


async def _render(
    monkeypatch: pytest.MonkeyPatch,
    page: FakePage,
    *,
    harvest_json: bool = True,
    goto_timeout_ms: int = 10_000,
    deadline_monotonic_s: float | None = None,
):
    install_fake_playwright(monkeypatch, page)
    return await rendered_fetch.render_page(
        _PAGE_URL,
        memo_scope=_TIER1_SCOPE,
        host_gate=asyncio.Semaphore(1),
        goto_timeout_ms=goto_timeout_ms,
        deadline_monotonic_s=deadline_monotonic_s,
        harvest_json=harvest_json,
    )


def _json_body(rows: int = 40) -> bytes:
    return b'{"series":[' + b'{"v":1},' * rows + b"]}"


class TestJsonHarvest:
    """A JavaScript dashboard's numbers arrive over XHR and are in its HTML at no wait
    condition, so the render records the page's own JSON as the derived-feed rung's input."""

    async def test_a_same_origin_json_body_is_harvested(self, monkeypatch):
        body = b'{"series":[' + b'{"date":"2026-09-01","value":47.2},' * 20 + b"]}"
        page = FakePage([FakeResponse(f"{_PAGE_URL}/api/series", content_type="application/json", body=body)])

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert [harvested.url for harvested in rendered.json_responses] == [f"{_PAGE_URL}/api/series"]

    async def test_harvesting_is_off_unless_the_caller_asks(self, monkeypatch):
        """The bodies buffer inside the render task alongside a 100-300 MB browser, so only a
        caller with a use for a derived feed pays for them."""
        page = FakePage([FakeResponse(f"{_PAGE_URL}/api/series", content_type="application/json", body=_json_body())])

        rendered = await _render(monkeypatch, page, harvest_json=False)

        assert rendered is not None
        assert rendered.json_responses == ()

    async def test_a_cross_origin_response_is_not_harvested(self, monkeypatch):
        """A stranger's JSON must never become the cited page's content."""
        page = FakePage(
            [FakeResponse("https://ads.tracker.test/beacon.json", content_type="application/json", body=_json_body())]
        )

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert rendered.json_responses == ()

    async def test_a_non_json_response_is_not_harvested(self, monkeypatch):
        body = b"x" * 4000
        page = FakePage([FakeResponse(f"{_PAGE_URL}/app.js", content_type="application/javascript", body=body)])

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert rendered.json_responses == ()

    async def test_a_tiny_json_body_is_not_harvested(self, monkeypatch):
        """Below the floor a JSON body is a ping, a feature flag or an empty envelope."""
        page = FakePage([FakeResponse(f"{_PAGE_URL}/api/flags", content_type="application/json", body=b'{"ok":true}')])

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert rendered.json_responses == ()

    async def test_an_oversized_json_body_is_not_harvested(self, monkeypatch):
        body = b"[" + b"1," * rendered_fetch.HARVEST_MAX_BODY_BYTES + b"]"
        page = FakePage([FakeResponse(f"{_PAGE_URL}/api/all", content_type="application/json", body=body)])

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert rendered.json_responses == ()

    async def test_a_body_declared_over_the_cap_is_never_read(self, monkeypatch):
        """``Response.body()`` materialises the whole body (base64 over the driver pipe, then
        decoded) before any size test could run on it, so the declared size is the only bound
        that can keep a 60 MB GeoJSON out of memory. The post-read test stays as the backstop
        for absent, compressed or lying headers."""
        response = FakeResponse(
            f"{_PAGE_URL}/api/geo",
            content_type="application/json",
            body=_json_body(),
            declared_length=rendered_fetch.HARVEST_MAX_BODY_BYTES + 1,
        )

        rendered = await _render(monkeypatch, FakePage([response]))

        assert rendered is not None
        assert rendered.json_responses == ()
        assert response.body_reads == 0

    async def test_a_body_still_in_flight_when_goto_returns_is_harvested(self, monkeypatch):
        """Every ``page.on`` firing is its own task and nothing used to join them, so a body that
        arrived a beat late was appended after the snapshot and then lost at teardown — exactly
        the derived-API rung's payload, and the miss stuck because the caller memoised it."""
        response = FakeResponse(
            f"{_PAGE_URL}/api/series", content_type="application/json", body=_json_body(), body_delay_s=0.05
        )

        rendered = await _render(monkeypatch, FakePage([response]))

        assert rendered is not None
        assert [harvested.url for harvested in rendered.json_responses] == [f"{_PAGE_URL}/api/series"]

    async def test_a_body_read_that_outlives_the_dom_read_bound_is_dropped_without_delaying_the_render(
        self, monkeypatch
    ):
        """The drain runs INSIDE the DOM-read bound, never after it: the harvest is opportunistic
        and may not lengthen a render whose real product is the DOM."""
        monkeypatch.setattr(rendered_fetch, "RENDER_DOM_READ_TIMEOUT_MS", 100)
        response = FakeResponse(
            f"{_PAGE_URL}/api/slow", content_type="application/json", body=_json_body(), body_delay_s=5.0
        )

        started = time.monotonic()
        rendered = await _render(monkeypatch, FakePage([response]))

        assert time.monotonic() - started < 1.0
        assert rendered is not None
        assert "rendered" in rendered.html
        assert rendered.json_responses == ()
        assert response.body_read_cancelled is True

    async def test_the_response_count_is_capped_before_the_sixth_body_is_read(self, monkeypatch):
        """Every response event fires before any body is read, so a count check made only in the
        listener bounds nothing; the reads are serialised and the count is checked again under
        that gate, so the cap is binding on the READS, which is the memory bound, not just on
        what is appended afterwards."""
        responses = [
            FakeResponse(f"{_PAGE_URL}/api/{index}", content_type="application/json", body=_json_body())
            for index in range(rendered_fetch.HARVEST_MAX_RESPONSES + 4)
        ]

        rendered = await _render(monkeypatch, FakePage(responses))

        assert rendered is not None
        assert len(rendered.json_responses) == rendered_fetch.HARVEST_MAX_RESPONSES
        assert sum(response.body_reads for response in responses) == rendered_fetch.HARVEST_MAX_RESPONSES

    async def test_at_most_one_body_is_buffered_at_a_time(self, monkeypatch):
        """``Response.body()`` materialises the whole body (base64 over the driver pipe, then
        decoded, about 2.3x the body at peak), and a dashboard's response events all fire before
        any body comes back. Read together, four undeclared 30 MB layers sat beside a 100-300 MB
        browser twice over on a 7 GB runner; read one at a time, peak harvest memory is one body."""
        responses = [
            FakeResponse(
                f"{_PAGE_URL}/api/{index}", content_type="application/json", body=_json_body(), body_delay_s=0.01
            )
            for index in range(4)
        ]

        rendered = await _render(monkeypatch, FakePage(responses))

        assert rendered is not None
        assert len(rendered.json_responses) == 4
        assert all(response.body_reads == 1 for response in responses)
        assert FakeResponse.peak_in_flight == 1

    async def test_a_response_that_fails_the_screens_spawns_no_read_task(self):
        """The host, content-type and declared-length screens run in the SYNC listener, so a
        page's hundreds of subresources (scripts, images, beacons) never become tasks at all;
        only a response that will actually be read does."""
        harvest = rendered_fetch._JsonHarvest(page_host="dashboard.example.com", playwright_error=PlaywrightError)
        screened_out = [
            FakeResponse("https://ads.tracker.test/beacon.json", content_type="application/json", body=_json_body()),
            FakeResponse(f"{_PAGE_URL}/app.js", content_type="application/javascript", body=b"x" * 4000),
            FakeResponse(
                f"{_PAGE_URL}/api/geo",
                content_type="application/json",
                body=_json_body(),
                declared_length=rendered_fetch.HARVEST_MAX_BODY_BYTES + 1,
            ),
        ]
        for response in screened_out:
            harvest.on_response(response)
        assert harvest._pending == set()

        harvest.on_response(FakeResponse(f"{_PAGE_URL}/api/series", content_type="application/json", body=_json_body()))
        assert len(harvest._pending) == 1
        harvest.cancel_pending()
        await asyncio.sleep(0)
        assert all(response.body_reads == 0 for response in screened_out)

    async def test_a_body_read_that_races_teardown_is_dropped_not_raised(self, monkeypatch):
        """Opportunistic discovery attached to a render whose real product is the DOM: a body
        we could not read must never be able to fail the render."""
        page = FakePage(
            [
                FakeResponse(f"{_PAGE_URL}/api/gone", content_type="application/json", body=_json_body(), raises=True),
                FakeResponse(f"{_PAGE_URL}/api/series", content_type="application/json", body=_json_body()),
            ]
        )

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert [harvested.url for harvested in rendered.json_responses] == [f"{_PAGE_URL}/api/series"]
        assert "rendered" in rendered.html


class TestJsonContentType:
    """The one JSON vocabulary for the harvest, the derived-feed reuse gate and the 200-response
    router: a feed one half of the ladder discovers must not be `unsupported_type` to another."""

    @pytest.mark.parametrize(
        ("content_type", "expected"),
        [
            ("application/json", True),
            ("application/json; charset=utf-8", True),
            ("text/json", True),
            ("application/geo+json", True),
            ("application/vnd.api+json", True),
            ("application/ld+json", True),
            ("text/html", False),
            ("application/javascript", False),
            ("text/plain", False),
            ("", False),
        ],
    )
    def test_the_vocabulary(self, content_type, expected):
        assert rendered_fetch.is_json_content_type(content_type) is expected


class TestHarvestableHost:
    """Same publisher by registrable domain (the vendored public-suffix list), or an allow-listed
    CDN. ``www.<x>`` page plus ``api.<x>`` / ``data.<x>`` feed is the ordinary dashboard shape,
    and the copernicus dashboard is the sibling-subdomain shape live QA found."""

    @pytest.mark.parametrize(
        ("response_host", "page_host", "expected"),
        [
            ("x.gov", "x.gov", True),
            ("api.x.gov", "www.x.gov", True),
            ("data.bls.gov", "www.bls.gov", True),
            ("api.x.gov", "x.gov", True),
            ("x.gov", "www.x.gov", True),
            ("api2.effis.emergency.copernicus.eu", "forest-fire.emergency.copernicus.eu", True),
            ("data.abs.gov.au", "abs.gov.au", True),
            ("static.dwcdn.net", "tracker.example.com", True),
            ("evil.test", "x.gov", False),
            ("a.co.uk", "b.co.uk", False),
            ("other.gov.au", "abs.gov.au", False),
            # A bare public suffix is nobody's publisher, so a stranger's site on it never matches.
            ("github.io", "x.github.io", False),
            # IP literals have no registrable domain; the PSL would collapse both to `113.9`.
            ("198.51.113.9", "203.0.113.9", False),
            ("203.0.113.9", "203.0.113.9", True),
            ("", "x.gov", False),
            ("x.gov", "", False),
        ],
    )
    def test_the_host_rule(self, response_host, page_host, expected):
        assert rendered_fetch._harvestable_json_host(response_host, page_host) is expected


class TestDerivedFeedSelection:
    """Size is the only signal available without knowing a dashboard's schema: a page fetches
    its config, its flags and its data, and the data is the big one."""

    def test_the_largest_body_wins(self):
        small = HarvestedJson(url="https://x.gov/flags", body=b"x" * 300)
        big = HarvestedJson(url="https://x.gov/series", body=b"x" * 9000)
        assert largest_json([small, big]) is big

    def test_a_tie_resolves_to_document_order(self):
        first = HarvestedJson(url="https://x.gov/a", body=b"x" * 500)
        second = HarvestedJson(url="https://x.gov/b", body=b"x" * 500)
        assert largest_json([first, second]) is first

    def test_nothing_harvested_is_none(self):
        assert largest_json([]) is None


class TestDerivedFeedLead:
    """The lead is what makes a served feed checkable, so the two provenances read differently."""

    def test_a_feed_from_this_pages_own_render(self):
        endpoint = DerivedEndpoint(endpoint_url="https://x.gov/api/series", discovered_on="https://x.gov/senate")
        lead = derived_api_lead(endpoint, "https://x.gov/senate")
        assert "https://x.gov/api/series" in lead
        assert "DIFFERENT page" not in lead

    def test_a_feed_reused_from_another_page_says_so(self):
        endpoint = DerivedEndpoint(endpoint_url="https://x.gov/api/series", discovered_on="https://x.gov/senate")
        lead = derived_api_lead(endpoint, "https://x.gov/house")
        assert "DIFFERENT page" in lead
        assert "https://x.gov/senate" in lead
        assert "check that it covers the quantity" in lead


class TestTheRenderMemos:
    """Two memos, both written by the callers and keyed by (scope, url), so neither fetch path's
    negative answers the other's question. The two paths mean different things by "rendered to
    nothing": gap-fill v2 writes it on bare trafilatura emptiness, Tier-1 only after the ARIA
    rewrite, the inline-chart read AND the XHR-harvest fallback all failed, so a v2 miss used to
    switch off a strictly richer Tier-1 attempt on the same URL."""

    async def test_one_paths_no_text_memo_does_not_answer_the_others(self, monkeypatch):
        rendered_fetch.note_rendered_no_text(_PAGE_URL, memo_scope=_V2_SCOPE)
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_V2_SCOPE) is True
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        chromium = install_fake_playwright(monkeypatch, FakePage([]))

        declined = await rendered_fetch.render_page(_PAGE_URL, memo_scope=_V2_SCOPE, host_gate=asyncio.Semaphore(1))
        rendered = await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))

        assert declined is None
        assert rendered is not None
        assert len(chromium.launch_args) == 1

    async def test_a_render_the_transport_cut_off_is_memoised_and_raises_again_without_launching(self, monkeypatch):
        """A second question citing a page that already ran out the clock must record
        ``render_timeout`` again, not ``renderer_unavailable`` — the count the operator reads as
        the Chromium install having failed. So the transport re-raises the memoised timeout
        instead of folding it into the ``None`` every other decline shares."""
        monkeypatch.setattr(rendered_fetch, "RENDER_DOM_READ_TIMEOUT_MS", 50)
        chromium = install_fake_playwright(monkeypatch, FakePage([], content_hangs=True))
        with pytest.raises(TimeoutError):
            await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))
        assert rendered_fetch.render_timed_out(_PAGE_URL, memo_scope=_TIER1_SCOPE) is True
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert len(chromium.launch_args) == 1

        with pytest.raises(TimeoutError):
            await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))

        assert len(chromium.launch_args) == 1
        # The other path's clock is its own.
        assert rendered_fetch.render_timed_out(_PAGE_URL, memo_scope=_V2_SCOPE) is False

    async def test_a_cut_in_the_queue_is_not_memoised(self, monkeypatch):
        """The caller's outer bound also fires while the render is still waiting on the two
        gates, which says nothing about the page. Only the transport knows a browser ran, so only
        the transport writes the timed-out memo — a queue cut leaves the URL live for the next
        question."""
        chromium = install_fake_playwright(monkeypatch, FakePage([]))
        gate = rendered_fetch._RENDERED_FETCH_GLOBAL_SEMAPHORE
        for _ in range(rendered_fetch.RENDER_LAUNCH_CAP):
            await gate.acquire()

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)),
                timeout=0.1,
            )

        assert chromium.launch_args == []
        assert rendered_fetch.render_timed_out(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        for _ in range(rendered_fetch.RENDER_LAUNCH_CAP):
            gate.release()
        rendered = await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))
        assert rendered is not None

    def test_the_memo_is_bounded(self):
        for index in range(rendered_fetch._RENDER_MEMO_MAX_ENTRIES + 5):
            rendered_fetch.note_rendered_no_text(f"https://x.example/{index}", memo_scope=_TIER1_SCOPE)
        assert rendered_fetch.rendered_to_nothing("https://x.example/0", memo_scope=_TIER1_SCOPE) is False
        assert (
            rendered_fetch.rendered_to_nothing(
                f"https://x.example/{rendered_fetch._RENDER_MEMO_MAX_ENTRIES + 4}", memo_scope=_TIER1_SCOPE
            )
            is True
        )

    async def test_a_v2_render_to_nothing_does_not_suppress_tier_1s_richer_attempt(self, monkeypatch):
        """Through the two real callers, not the memo functions: gap-fill v2's ``fetch`` reads an
        empty DOM and memoises under its own scope; the Tier-1 rung on the same URL must still
        launch — its classification can rescue the page on chart data or the harvested feed alone —
        and only then memoise under ITS scope."""
        chromium = install_fake_playwright(monkeypatch, FakePage([], html=_EMPTY_DOM))

        v2_result = await agentic_tools._try_rendered_fetch(_PAGE_URL)

        assert v2_result is not None
        assert v2_result.status == "error"
        assert v2_result.method == "rendered"
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_V2_SCOPE) is True
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert len(chromium.launch_args) == 1

        direct = FetchResult(url=_PAGE_URL, status="js_wall", text="", http_status=200, content_type="text/html")
        ctx = FetchContext()
        tier1_result = await resolution_source._rendered_rung(_PAGE_URL, direct, {}, ctx)

        assert len(chromium.launch_args) == 2
        assert tier1_result is None
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is True
        assert [attempt.skipped_reason for attempt in ctx.rungs] == [""]


class TestTheNavigationBudgetAfterTheGates:
    """The render queues on two unbounded acquires — the caller's loop-wide per-host gate, shared
    by every concurrent question, and the process-global launch cap shared with gap-fill v2 —
    and used to navigate on a budget measured BEFORE either. The Tier-1 rung's outer ``wait_for``
    bounds the overrun, but a browser was still launched with no time left, and the goto budget
    left no room for the settle and the DOM read that follow it, so the salvage-after-goto-timeout
    population the rung exists for was cut off by the outer bound at every ordinary budget."""

    async def test_a_deadline_already_spent_declines_without_launching(self, monkeypatch, caplog):
        """Its own exception rather than the shared ``None``: the caller's wall budget ran out in
        the queue, which is neither a missing browser nor a render that was cut off, and it must
        be recorded as the first of those three and not the other two."""
        page = FakePage([])
        chromium = install_fake_playwright(monkeypatch, page)
        with (
            caplog.at_level(logging.WARNING, logger="metaculus_bot.research.rendered_fetch"),
            pytest.raises(rendered_fetch.RenderBudgetExpired),
        ):
            await rendered_fetch.render_page(
                _PAGE_URL,
                memo_scope=_TIER1_SCOPE,
                host_gate=asyncio.Semaphore(1),
                deadline_monotonic_s=time.monotonic() + 1.0,
            )

        assert chromium.launch_args == []
        assert page.goto_calls == []
        assert rendered_fetch._PLAYWRIGHT_WARNED is False
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert rendered_fetch.render_timed_out(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert [message for message in caplog.messages if "declined after the gates" in message]

    async def test_the_tier_1_rung_records_a_post_gate_decline_as_its_wall_budget(self, monkeypatch):
        async def _expired(url: str, **_kwargs: Any) -> None:
            await asyncio.sleep(0)
            raise rendered_fetch.RenderBudgetExpired(f"under 5000ms left for {url}")

        monkeypatch.setattr(resolution_source, "render_page", _expired)
        direct = FetchResult(url=_PAGE_URL, status="js_wall", text="", http_status=200, content_type="text/html")
        ctx = FetchContext()

        result = await resolution_source._rendered_rung(_PAGE_URL, direct, {}, ctx)

        assert result is None
        assert [attempt.skipped_reason for attempt in ctx.rungs] == ["wall_budget"]
        assert rendered_fetch._PLAYWRIGHT_WARNED is False

    async def test_the_goto_budget_is_recomputed_once_the_gates_are_held(self, monkeypatch):
        """Both launch slots are held for 0.3 s while the render queues behind them; the goto
        it then runs must be measured from AFTER the wait, not from the figure the caller
        computed before it."""
        page = FakePage([])
        install_fake_playwright(monkeypatch, page)
        gate = rendered_fetch._RENDERED_FETCH_GLOBAL_SEMAPHORE
        for _ in range(rendered_fetch.RENDER_LAUNCH_CAP):
            await gate.acquire()
        deadline_s = 15.0
        started = time.monotonic()
        render = asyncio.create_task(
            rendered_fetch.render_page(
                _PAGE_URL,
                memo_scope=_TIER1_SCOPE,
                host_gate=asyncio.Semaphore(1),
                goto_timeout_ms=20_000,
                deadline_monotonic_s=started + deadline_s,
            )
        )
        await asyncio.sleep(0.3)
        for _ in range(rendered_fetch.RENDER_LAUNCH_CAP):
            gate.release()

        rendered = await render

        assert rendered is not None
        (call,) = page.goto_calls
        assert call["timeout"] < 20_000
        assert rendered_fetch.RENDER_MIN_GOTO_MS <= call["timeout"] <= int(deadline_s * 1000) - 300 - _RENDER_TAIL_MS

    async def test_the_goto_leaves_room_for_the_settle_and_the_dom_read(self, monkeypatch):
        """Tier-1 passes a 20 s budget as an 18 s goto ceiling plus the deadline. A goto that runs
        its budget out and is then salvaged needs the settle AND the DOM read to finish inside
        that same 20 s, so the transport sizes the goto off the deadline less both."""
        page = FakePage([], goto_raises=PlaywrightError("Timeout exceeded"))
        deadline_s = 20.0
        started = time.monotonic()

        rendered = await _render(monkeypatch, page, goto_timeout_ms=18_000, deadline_monotonic_s=started + deadline_s)

        assert rendered is not None
        assert "rendered" in rendered.html
        (call,) = page.goto_calls
        assert call["timeout"] + _RENDER_TAIL_MS <= int(deadline_s * 1000)
        assert call["timeout"] >= rendered_fetch.RENDER_MIN_GOTO_MS

    @staticmethod
    def _scale_the_tier_1_shape_down(monkeypatch: pytest.MonkeyPatch) -> None:
        """The Tier-1 rung's constants at a tenth of their size, so a test can run the whole
        goto / settle / read / drain / teardown chain with the clock on in under two seconds. The
        exit reserve keeps its production RELATION to the teardown bound (that bound plus an
        allowance for the launch and the driver stop); only the magnitudes shrink."""
        monkeypatch.setattr(rendered_fetch, "RENDER_SETTLE_MS", 0)
        monkeypatch.setattr(rendered_fetch, "RENDER_DOM_READ_TIMEOUT_MS", 500)
        monkeypatch.setattr(rendered_fetch, "RENDER_POST_GOTO_TAIL_MS", 500)
        monkeypatch.setattr(rendered_fetch, "RENDER_MIN_GOTO_MS", 100)
        monkeypatch.setattr(rendered_fetch, "RENDER_TEARDOWN_TIMEOUT_MS", 300)
        monkeypatch.setattr(rendered_fetch, "RENDER_EXIT_RESERVE_MS", 300 + 500)

    async def test_a_salvaged_dom_comes_back_inside_the_callers_deadline_over_a_pending_body_read(self, monkeypatch):
        """The Tier-1 rung's shape end to end, with the clock running: the rung's outer
        ``wait_for`` sits at the budget, the deadline handed to the transport sits the exit
        reserve before it, the navigation budget is computed BEFORE the launch, and the goto runs
        that budget out. The DOM read's fixed bound then lands past the transport's deadline by
        the launch time, and a same-publisher body still in flight used to hold the harvest drain
        there and on into the reserve, so the outer bound fired and discarded a DOM the transport
        had already read, billed as ``render_timeout``. The drain is clamped to the transport's
        deadline instead; the DOM read keeps its own bound.
        """
        self._scale_the_tier_1_shape_down(monkeypatch)
        pending = FakeResponse(
            f"{_PAGE_URL}/api/poll", content_type="application/json", body=_json_body(), body_delay_s=10.0
        )
        page = FakePage([pending], goto_raises=PlaywrightError("Timeout exceeded"), goto_runs_its_budget_out=True)
        install_fake_playwright(monkeypatch, page, launch_delay_s=0.1)
        budget_s = 2.0

        rendered = await asyncio.wait_for(
            rendered_fetch.render_page(
                _PAGE_URL,
                memo_scope=_TIER1_SCOPE,
                host_gate=asyncio.Semaphore(1),
                goto_timeout_ms=int(budget_s * 1000),
                deadline_monotonic_s=time.monotonic() + budget_s - rendered_fetch.RENDER_EXIT_RESERVE_MS / 1000,
                harvest_json=True,
            ),
            timeout=budget_s,
        )

        assert rendered is not None
        assert "rendered" in rendered.html
        assert rendered.json_responses == ()
        assert pending.body_read_cancelled is True
        assert page.teardown == ["unroute_all", "context.close", "browser.close"]

    async def test_a_render_that_runs_every_bound_out_still_hands_its_dom_back_inside_the_rungs_bound(
        self, monkeypatch, caplog
    ):
        """The worst case the exit reserve exists for, under the rung's own ``wait_for``: the goto
        consumes its whole recomputed budget, a harvested body is still in flight at the
        transport's deadline, and the browser then wedges on ``context.close``. Three separate
        2 s teardown bounds let that teardown run 6 s past the outer cut (``wait_for`` cancels
        the render and then AWAITS its finallys), which tripped the provider's 45 s wall and
        discarded every page the question had fetched. With one shared bound and the reserve
        subtracted from the deadline, the DOM comes back before the cut and the wedged close is
        left to the driver stop.
        """
        self._scale_the_tier_1_shape_down(monkeypatch)
        pending = FakeResponse(
            f"{_PAGE_URL}/api/poll", content_type="application/json", body=_json_body(), body_delay_s=10.0
        )
        page = FakePage([pending], goto_raises=PlaywrightError("Timeout exceeded"), goto_runs_its_budget_out=True)
        install_fake_playwright(monkeypatch, page, faults=Faults(close_hangs=True), launch_delay_s=0.1)
        budget_s = 2.0

        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.rendered_fetch"):
            rendered = await asyncio.wait_for(
                rendered_fetch.render_page(
                    _PAGE_URL,
                    memo_scope=_TIER1_SCOPE,
                    host_gate=asyncio.Semaphore(1),
                    goto_timeout_ms=int(budget_s * 1000),
                    deadline_monotonic_s=time.monotonic() + budget_s - rendered_fetch.RENDER_EXIT_RESERVE_MS / 1000,
                    harvest_json=True,
                ),
                timeout=budget_s,
            )
        elapsed = time.monotonic() - started

        assert rendered is not None
        assert "rendered" in rendered.html
        assert rendered.json_responses == ()
        assert pending.body_read_cancelled is True
        # The drain ran to the transport's deadline (1.2 s) and the wedged close to the shared
        # teardown bound (0.3 s); the browser close had nothing left and went to the driver stop.
        assert 1.45 <= elapsed < budget_s
        assert page.teardown == ["unroute_all", "context.close"]
        left_to_the_driver = [message for message in caplog.messages if "leaving it to the driver stop" in message]
        assert len(left_to_the_driver) == 2

    async def test_a_call_without_a_deadline_keeps_the_callers_goto_budget(self, monkeypatch):
        """Gap-fill v2's shape: its own ceilings bound the call, so the transport has nothing to
        recompute against and the caller's figure stands."""
        page = FakePage([])

        rendered = await _render(monkeypatch, page, goto_timeout_ms=10_000)

        assert rendered is not None
        assert page.goto_calls[0]["timeout"] == 10_000

    async def test_the_tier_1_rung_passes_its_wall_deadline_and_scope(self, monkeypatch):
        calls: list[dict[str, Any]] = []

        async def _recording_render(url: str, **kwargs: Any) -> None:
            calls.append({"url": url, "called_at": time.monotonic(), **kwargs})
            await asyncio.sleep(0)

        monkeypatch.setattr(resolution_source, "render_page", _recording_render)
        monkeypatch.setattr(FetchContext, "rung_budget_s", lambda self: 20.0)
        direct = FetchResult(url=_PAGE_URL, status="js_wall", text="", http_status=200, content_type="text/html")

        await resolution_source._rendered_rung(_PAGE_URL, direct, {}, FetchContext())

        (call,) = calls
        assert call["memo_scope"] == _TIER1_SCOPE
        assert call["harvest_json"] is True
        # The deadline sits the exit reserve before the rung's own 20 s bound, so the transport's
        # teardown and driver stop land inside that bound rather than being cancelled by it.
        reserve_s = rendered_fetch.RENDER_EXIT_RESERVE_MS / 1000
        assert 19.5 - reserve_s < call["deadline_monotonic_s"] - call["called_at"] <= 20.0 - reserve_s


class TestTheDomCeiling:
    """The rendered DOM was the one body path that escaped the 5 MiB ceiling every aiohttp fetch
    enforces, and the Tier-1 caller then copies it three more times before trafilatura parses it
    into a tree several times bigger. The transport declines above a named ceiling, measured in
    characters so the check does not itself make the copy it exists to prevent."""

    async def test_a_dom_over_the_ceiling_is_declined_and_the_browser_torn_down(self, monkeypatch, caplog):
        """Its own exception rather than the shared ``None``: the page RENDERED, so the caller must
        be able to count it apart from a browser that is missing, and it is not memoised, because
        "rendered to nothing" would be false."""
        monkeypatch.setattr(rendered_fetch, "RENDERED_DOM_MAX_CHARS", 100)
        page = FakePage([], html="<html><body>" + "x" * 200 + "</body></html>")

        with (
            caplog.at_level(logging.WARNING, logger="metaculus_bot.research.rendered_fetch"),
            pytest.raises(rendered_fetch.RenderDomOverCeiling),
        ):
            await _render(monkeypatch, page)

        assert page.teardown == ["unroute_all", "context.close", "browser.close"]
        assert rendered_fetch._PLAYWRIGHT_WARNED is False
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert rendered_fetch.render_timed_out(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert [message for message in caplog.messages if "over the" in message and "ceiling" in message]

    async def test_the_tier_1_rung_records_a_dom_over_the_ceiling_as_its_own_skip(self, monkeypatch):
        """Folded into ``renderer_unavailable`` it pointed triage at the Playwright install."""

        async def _too_large(url: str, **_kwargs: Any) -> None:
            await asyncio.sleep(0)
            raise rendered_fetch.RenderDomOverCeiling(f"the rendered DOM of {url} is over the ceiling")

        monkeypatch.setattr(resolution_source, "render_page", _too_large)
        direct = FetchResult(url=_PAGE_URL, status="js_wall", text="", http_status=200, content_type="text/html")
        ctx = FetchContext()

        result = await resolution_source._rendered_rung(_PAGE_URL, direct, {}, ctx)

        assert result is None
        assert [attempt.skipped_reason for attempt in ctx.rungs] == ["render_dom_too_large"]
        direct.rung_attempts = list(ctx.rungs)
        counts = resolution_source._rung_counts([direct])
        assert counts["render_dom_too_large_skips"] == 1
        assert counts["renderer_unavailable_skips"] == 0
        assert counts["rendered_attempts"] == 0
        assert rendered_fetch._PLAYWRIGHT_WARNED is False

    async def test_a_dom_at_the_ceiling_is_read(self, monkeypatch):
        html = "<html><body>" + "x" * 50 + "</body></html>"
        monkeypatch.setattr(rendered_fetch, "RENDERED_DOM_MAX_CHARS", len(html))

        rendered = await _render(monkeypatch, FakePage([], html=html))

        assert rendered is not None
        assert rendered.html == html


class TestTheBrowserContext:
    async def test_service_workers_are_blocked(self, monkeypatch):
        """``browser_context.route`` does not intercept requests a service worker makes, so a
        worker could dial past the SSRF route guard; Playwright's own remedy is to block them
        whenever interception is in use."""
        page = FakePage([])

        await _render(monkeypatch, page)

        assert page.context_kwargs["service_workers"] == "block"
        assert page.context_kwargs["user_agent"]


class TestTheWebSocketChannel:
    """``context.route`` never sees a WebSocket handshake: HTTP interception is the CDP ``Fetch``
    domain, sockets surface only on the report-only ``Network.webSocket*`` events, and an
    IP-literal target such as ``ws://127.0.0.1`` never consults the resolver the DNS pin
    rewrites. Playwright's separate ``route_web_socket`` API is the one hook, and a routed socket
    dials nothing unless its handler calls ``connect_to_server()``, so a handler that does nothing
    closes the channel. What the fake can show is the registration and the handler's behaviour;
    Chromium actually refusing the handshake is not observable here, because the suite blocks
    every real launch."""

    async def test_the_block_is_registered_on_the_context_before_the_page_exists(self, monkeypatch):
        """Only sockets created after the registration are routed, so a page opened first could
        open one before the block existed."""
        page = FakePage([])

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert page.web_socket_patterns == ["**/*"]
        assert page.web_socket_handler is rendered_fetch._block_web_socket
        assert page.setup_events.index("route_web_socket") < page.setup_events.index("new_page")

    def test_the_block_never_connects_and_returns_normally(self, caplog):
        """The handler runs on a task Playwright creates, where a raise is a detached-listener
        traceback (the 2026-07-25 storm), and ``connect_to_server()`` is the handshake being
        refused; the one thing it does is name the host it refused, at INFO, because ``cli.py``
        configures the root logger at INFO and this line is the only record that a page's
        socket-fed content was withheld from the render."""
        socket = FakeWebSocketRoute("ws://user:secret@127.0.0.1:8080/feed?token=abc")

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.rendered_fetch"):
            result = rendered_fetch._block_web_socket(socket)

        assert result is None
        assert socket.connect_calls == 0
        assert socket.close_calls == []
        (record,) = [record for record in caplog.records if "WebSocket" in record.getMessage()]
        assert record.levelno == logging.INFO
        message = record.getMessage()
        assert "127.0.0.1:8080" in message
        assert "secret" not in message
        assert "token" not in message

    @pytest.mark.parametrize("url", ["", "not a url", "ws://[::1/broken", "wss://x.example"])
    def test_the_block_cannot_raise_on_an_odd_url(self, url):
        """``urlparse`` raises on an unbalanced IPv6 bracket; the handler must not, whatever the
        page hands it."""
        socket = FakeWebSocketRoute(url)
        assert rendered_fetch._block_web_socket(socket) is None
        assert socket.connect_calls == 0

    async def test_a_playwright_error_at_the_registration_takes_the_pre_page_path(self, monkeypatch, caplog):
        """One more driver call before ``new_page``: not wrapped, so a Playwright-class error there
        latches the once-per-run warning like a failed launch, and the browser is still closed."""
        page = FakePage([])
        install_fake_playwright(
            monkeypatch, page, faults=Faults(route_web_socket_error=PlaywrightError("Target closed"))
        )

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.rendered_fetch"):
            rendered = await rendered_fetch.render_page(
                _PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)
            )

        assert rendered is None
        assert page.goto_calls == []
        assert rendered_fetch._PLAYWRIGHT_WARNED is True
        # The failure lands before the page's own try/finally, so there is no HTTP guard drain to
        # run, and the two closes still happen.
        assert page.teardown == ["context.close", "browser.close"]
        assert [message for message in caplog.messages if "rung unavailable" in message]


class TestTheSharedPlaywrightFake:
    """The one fake object graph every render test drives has to honour the two transport contracts
    a test cannot see it breaking: the DNS pin names the host the transport asked for, as the real
    resolver does, and the WebSocket double accepts exactly the calls the real one does."""

    async def test_the_default_pin_is_the_requested_host(self, monkeypatch):
        """With a fixed default pin, a render of any other host took the off-host refusal instead of
        rendering, so a test asserting a decline passed while the mechanism it pinned never ran."""
        page = FakePage([])
        chromium = install_fake_playwright(monkeypatch, page)

        rendered = await rendered_fetch.render_page(
            "https://other.example.org/page", memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)
        )

        assert rendered is not None
        assert page.content_reads == 1
        assert chromium.launch_args == [["--host-resolver-rules=MAP other.example.org 93.184.216.34"]]

    async def test_an_explicit_pin_still_lets_a_test_pin_the_wrong_host(self, monkeypatch):
        page = FakePage([])
        install_fake_playwright(monkeypatch, page, pinned=("elsewhere.example.net", "93.184.216.34"))

        with pytest.raises(rendered_fetch.RenderOffHost):
            await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))

        assert page.content_reads == 0

    async def test_the_default_pin_declines_a_host_the_real_resolver_would(self, monkeypatch):
        """A unicode hostname is unpinnable for real (Chromium canonicalises it before matching the
        MAP rule), so the default derives eligibility the same way rather than pinning it anyway."""
        page = FakePage([])
        chromium = install_fake_playwright(monkeypatch, page)

        rendered = await rendered_fetch.render_page(
            "https://münchen.example/x", memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)
        )

        assert rendered is None
        assert chromium.launch_args == []

    def test_the_socket_double_is_keyword_only_like_the_real_close(self):
        """A handler rewritten to ``close(1008, "blocked")`` would pass the suite and raise
        ``TypeError`` inside a Playwright-dispatched task in the one path the suite never runs, so
        the double must refuse exactly the call the real driver refuses."""
        for close in (FakeWebSocketRoute.close, WebSocketRoute.close):
            signature = inspect.signature(close)
            for name in ("code", "reason"):
                assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            with pytest.raises(TypeError):
                signature.bind(None, 1008, "blocked")


class TestDnsPinEligibility:
    """Chromium matches ``--host-resolver-rules`` against the canonical (punycode) hostname, so a
    pattern built from a unicode or trailing-dot host is accepted and matches nothing: the pin
    goes inert and the rebinding window it closes re-opens. Fail closed instead of reproducing
    Chromium's canonicalisation."""

    @pytest.mark.parametrize(
        "url",
        ["https://münchen.example/x", "https://straße.example/x", "https://example.com./x"],
    )
    def test_a_host_chromium_would_canonicalise_is_not_pinnable(self, url):
        assert rendered_fetch._pinnable_url_host(url) is None

    def test_an_already_punycoded_host_is_pinnable(self):
        assert rendered_fetch._pinnable_url_host("https://xn--mnchen-3ya.example/x") == "xn--mnchen-3ya.example"

    async def test_a_unicode_host_never_launches(self, monkeypatch, caplog):
        real_resolve = rendered_fetch._resolve_pinned_host
        chromium = install_fake_playwright(monkeypatch, FakePage([]))
        monkeypatch.setattr(rendered_fetch, "_resolve_pinned_host", real_resolve)

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.rendered_fetch"):
            rendered = await rendered_fetch.render_page(
                "https://münchen.example/x", memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)
            )

        assert rendered is None
        assert chromium.launch_args == []
        assert [message for message in caplog.messages if "not pinnable" in message]


class TestTheMainFrameStatus:
    """The browser's own main-frame HTTP status rides on the page, so the Tier-1 rung can refuse
    to read a browser-targeted 403 or 429 interstitial as content. The direct GET answered 200
    for that URL, so a non-200 from the browser is the edge telling the browser apart."""

    _CHALLENGE = (
        "<!doctype html><html><head><title>Just a moment</title></head><body><article>"
        "<h1>Checking your browser before accessing the tracker</h1>"
        "<p>This process is automatic. Your browser will redirect to your requested content shortly. "
        "Please allow up to five seconds. If you are on a personal connection you can run an "
        "anti-virus scan on your device to make sure it is not infected with malware. If you are at "
        "an office or shared network you can ask the network administrator to run a scan across the "
        "network looking for misconfigured or infected devices. Performance and security by an edge "
        "provider. Ray ID 8a1b2c3d4e5f6789. Your IP has been recorded for this request.</p>"
        "</article></body></html>"
    )

    async def test_the_status_rides_on_the_rendered_page(self, monkeypatch):
        rendered = await _render(monkeypatch, FakePage([], status=403))
        assert rendered is not None
        assert rendered.http_status == 403

    async def test_a_salvaged_dom_carries_no_status(self, monkeypatch):
        """No response object survives a goto timeout, which is the salvage path."""
        rendered = await _render(monkeypatch, FakePage([], goto_raises=PlaywrightError("Timeout exceeded")))
        assert rendered is not None
        assert rendered.http_status is None
        assert rendered.content_type == ""

    async def test_the_tier_1_rung_does_not_publish_a_browser_targeted_error_page(self, monkeypatch, caplog):
        async def _blocked_render(url: str, **_kwargs: Any) -> RenderedPage:
            await asyncio.sleep(0)
            return RenderedPage(url=url, content_type="text/html", html=self._CHALLENGE, http_status=403)

        monkeypatch.setattr(resolution_source, "render_page", _blocked_render)
        direct = FetchResult(url=_PAGE_URL, status="js_wall", text="", http_status=200, content_type="text/html")
        ctx = FetchContext()

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.resolution_source"):
            result = await resolution_source._rendered_rung(_PAGE_URL, direct, {}, ctx)

        assert result is None
        # A 403 or 429 is retryable, so the URL is not memoised as rendered-to-nothing.
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert [message for message in caplog.messages if "403" in message]
        # Its own skip token, so the refusal is not counted as a render that read chrome.
        assert [attempt.skipped_reason for attempt in ctx.rungs] == ["render_non_200"]

    async def test_a_200_from_the_browser_still_classifies(self, monkeypatch):
        async def _ok_render(url: str, **_kwargs: Any) -> RenderedPage:
            await asyncio.sleep(0)
            return RenderedPage(url=url, content_type="text/html", html=self._CHALLENGE, http_status=200)

        monkeypatch.setattr(resolution_source, "render_page", _ok_render)
        direct = FetchResult(url=_PAGE_URL, status="js_wall", text="", http_status=200, content_type="text/html")

        result = await resolution_source._rendered_rung(_PAGE_URL, direct, {}, FetchContext())

        assert result is not None
        assert result.status == "success"


class TestTheLandingHost:
    """A server-side redirect hop is dialed with no check of ours: the driver constructs a Route
    only for a request with no ``redirectedFrom``, so a 302 to a private address was followed,
    rendered, and handed back attached to the cited URL. The DNS pin covers ONE hostname, so a
    main frame that landed anywhere else was reached through Chromium's own resolver, and its DOM
    is refused before it is read, or discarded unpublished when the navigation committed during the
    read. ``page.url`` is read after the settle on both paths, because on the salvage path (the goto
    raised, no response object) it is the only source of the landing, and again after the read."""

    _OFF_HOST = "https://internal.example.net/admin/secrets"

    async def test_an_off_host_landing_is_refused_before_the_dom_is_read(self, monkeypatch, caplog):
        page = FakePage([], land_on=self._OFF_HOST)

        with (
            caplog.at_level(logging.WARNING, logger="metaculus_bot.research.rendered_fetch"),
            pytest.raises(rendered_fetch.RenderOffHost) as raised,
        ):
            await _render(monkeypatch, page)

        assert page.content_reads == 0
        assert raised.value.requested_url == _PAGE_URL
        assert raised.value.final_url == self._OFF_HOST
        assert raised.value.pinned_host == "dashboard.example.com"
        assert page.teardown == ["unroute_all", "context.close", "browser.close"]
        assert rendered_fetch._PLAYWRIGHT_WARNED is False
        # The page rendered, so "rendered to nothing" would be false, and no clock ran out.
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert rendered_fetch.render_timed_out(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        # The refusal is the only per-event record of a page sending the browser off the pin, so it
        # is a registered marker (scripts/telemetry/markers.py) rather than free text that expires
        # with the GitHub Actions logs, and the spelling is pinned HERE too so the emitter and the
        # spec are checked against the same bytes. Hostnames only: the landing URL's path can carry
        # a session token or a credential.
        (message,) = [message for message in caplog.messages if "RENDERED_FETCH_OFF_HOST" in message]
        assert (
            message == "RENDERED_FETCH_OFF_HOST: scope=resolution_source pinned_host=dashboard.example.com "
            "landed_host=internal.example.net"
        )
        assert "/admin" not in message
        spec = next(s for s in MARKER_SPECS if s.name == "rendered_fetch_off_host")
        match = spec.regex.search(message)
        assert match is not None
        assert match.group("scope") == _TIER1_SCOPE
        assert match.group("pinned_host") == "dashboard.example.com"
        assert match.group("landed_host") == "internal.example.net"

    async def test_an_ip_literal_landing_is_off_host(self, monkeypatch):
        """The IMDS shape: the hostname compare refuses a literal like any other stranger."""
        page = FakePage([], land_on="http://169.254.169.254/latest/meta-data/")

        with pytest.raises(rendered_fetch.RenderOffHost):
            await _render(monkeypatch, page)

        assert page.content_reads == 0

    async def test_a_same_host_landing_on_another_path_or_scheme_is_read(self, monkeypatch):
        """Only the HOST is pinned: a canonical-path or scheme hop on the same host is the ordinary
        shape, and the landing rides on the page for the caller."""
        landed = "http://dashboard.example.com/senate/2026?tab=polls"
        page = FakePage([], land_on=landed)

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert rendered.url == _PAGE_URL
        assert rendered.final_url == landed
        assert page.content_reads == 1

    @pytest.mark.parametrize(
        ("final_url", "expected"),
        [
            ("https://dashboard.example.com/senate/2026/", "https://dashboard.example.com/senate/2026/"),
            # A navigation that never committed names no document, so the requested URL stands.
            ("about:blank", _PAGE_URL),
            ("", _PAGE_URL),
        ],
    )
    def test_the_document_url_is_the_landing_when_one_committed(self, final_url, expected):
        """The base both callers classify and resolve links against: the document the DOM came
        from, never a no-document landing."""
        page = RenderedPage(url=_PAGE_URL, content_type="text/html", html=_DOM, final_url=final_url)
        assert page.document_url == expected

    async def test_a_landing_that_differs_only_in_case_is_the_same_host(self, monkeypatch):
        page = FakePage([], land_on="https://Dashboard.Example.COM/senate")

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert rendered.final_url == "https://Dashboard.Example.COM/senate"

    async def test_a_navigation_that_never_committed_falls_through_to_the_dom_read(self, monkeypatch):
        """A genuine navigation failure leaves the page on ``about:blank``, which is nobody's host:
        the DOM read proceeds and yields the empty document it always did."""
        page = FakePage([], goto_raises=PlaywrightError("net::ERR_NAME_NOT_RESOLVED"), html=_EMPTY_DOM)

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert rendered.final_url == "about:blank"
        assert rendered.html == _EMPTY_DOM
        assert page.content_reads == 1

    async def test_chromiums_own_error_document_after_a_failed_navigation_is_refused(self, monkeypatch, caplog):
        """Live QA (2026-09-04) against real Chromium: a goto that failed with ``net::ERR_UNSAFE_PORT``,
        and one that failed with ``net::ERR_CONNECTION_REFUSED`` on a redirect to a loopback target,
        both left ``page.url`` at ``chrome-error://chromewebdata/`` rather than ``about:blank``. A
        helper that allowed every non-http(s) scheme read that document and handed back an empty
        page carrying that URL; fail-shut, it is off the pinned host like any other stranger."""
        page = FakePage(
            [], goto_raises=PlaywrightError("net::ERR_UNSAFE_PORT"), land_on="chrome-error://chromewebdata/"
        )

        with (
            caplog.at_level(logging.WARNING, logger="metaculus_bot.research.rendered_fetch"),
            pytest.raises(rendered_fetch.RenderOffHost) as raised,
        ):
            await _render(monkeypatch, page)

        assert page.content_reads == 0
        assert raised.value.final_url == "chrome-error://chromewebdata/"
        (message,) = [message for message in caplog.messages if "RENDERED_FETCH_OFF_HOST" in message]
        assert "landed_host=chromewebdata" in message

    @pytest.mark.parametrize(
        "landed",
        [
            "data:text/html,<p>built by the page</p>",
            "file:///etc/hostname",
            "blob:https://dashboard.example.com/3f1c9a2e",
            "chrome-error://chromewebdata/",
        ],
    )
    async def test_a_landing_on_a_scheme_that_is_not_the_pinned_host_is_refused(self, monkeypatch, landed):
        """The guard answers "is this document from the host we vetted and pinned?", and fails SHUT:
        every scheme it does not name is refused rather than read, so a scheme Chromium adds later
        cannot slip through an allowlist nobody updated."""
        page = FakePage([], land_on=landed)

        with pytest.raises(rendered_fetch.RenderOffHost):
            await _render(monkeypatch, page)

        assert page.content_reads == 0

    @pytest.mark.parametrize(
        ("final_url", "expected"),
        [
            ("", False),
            ("about:blank", False),
            ("about:srcdoc", False),
            ("ABOUT:BLANK", False),
            ("https://dashboard.example.com/senate", False),
            ("http://Dashboard.Example.COM:8443/x", False),
            ("https://other.example.com/", True),
            ("http://169.254.169.254/latest/meta-data/", True),
            # An http(s) URL with no hostname at all matches no pin.
            ("https:///no-host", True),
            ("data:text/html,x", True),
            ("chrome-error://chromewebdata/", True),
        ],
    )
    def test_the_no_document_landings_are_the_only_allowlist(self, final_url, expected):
        assert rendered_fetch._landed_off_host(final_url, "dashboard.example.com") is expected

    async def test_a_navigation_that_commits_during_the_dom_read_is_discarded_unpublished(self, monkeypatch, caplog):
        """The window between the pre-read check and the read. ``page.url`` is a client-side cache
        updated by the driver's ``navigated`` event, and ``page.content()`` is a driver round trip
        evaluated in whatever document is current when the driver handles it, so a main frame whose
        navigation commits in that window hands back the OTHER host's DOM with the pre-read check
        already passed. The driver's pipe is ordered, so the commit's ``navigated`` event lands
        before the content reply and the post-read ``page.url`` reflects it: the check runs again
        after the read, and the DOM that was read is thrown away rather than returned."""
        internal_markup = "<html><body><p>ami-id: ami-0abc</p></body></html>"

        class _NavigatesDuringTheRead(FakePage):
            async def content(self) -> str:
                html = await super().content()
                self.url = "http://169.254.169.254/latest/meta-data/"
                del html
                return internal_markup

        page = _NavigatesDuringTheRead([])

        with (
            caplog.at_level(logging.WARNING, logger="metaculus_bot.research.rendered_fetch"),
            pytest.raises(rendered_fetch.RenderOffHost) as raised,
        ):
            await _render(monkeypatch, page)

        assert page.content_reads == 1
        assert raised.value.final_url == "http://169.254.169.254/latest/meta-data/"
        assert raised.value.pinned_host == "dashboard.example.com"
        assert page.teardown == ["unroute_all", "context.close", "browser.close"]
        (message,) = [message for message in caplog.messages if "RENDERED_FETCH_OFF_HOST" in message]
        assert "landed_host=169.254.169.254" in message
        assert not [message for message in caplog.messages if "ami-0abc" in message]

    async def test_the_final_url_is_the_post_read_landing(self, monkeypatch):
        """The URL the page carries names the document whose DOM was read, so it is the second
        read of ``page.url``, not the one taken before ``page.content()``."""
        landed = "https://dashboard.example.com/senate/2026/"

        class _HopsOnTheSameHostDuringTheRead(FakePage):
            async def content(self) -> str:
                html = await super().content()
                self.url = landed
                return html

        rendered = await _render(monkeypatch, _HopsOnTheSameHostDuringTheRead([]))

        assert rendered is not None
        assert rendered.final_url == landed

    async def test_the_salvage_path_applies_the_check_too(self, monkeypatch):
        """A timed-out goto may already have landed on the redirect target (4 of the replay's 10
        render rescues came through this path); ``response`` is None there, so ``page.url`` is
        the only source of the landing, and it is checked before the salvage read."""
        page = FakePage([], goto_raises=PlaywrightTimeoutError("Timeout 33000ms exceeded."), land_on=self._OFF_HOST)

        with pytest.raises(rendered_fetch.RenderOffHost):
            await _render(monkeypatch, page)

        assert page.content_reads == 0
        assert page.settles == [rendered_fetch.RENDER_SETTLE_MS]

    async def test_a_salvaged_same_host_landing_carries_its_final_url(self, monkeypatch):
        landed = f"{_PAGE_URL}/2026"
        page = FakePage([], goto_raises=PlaywrightTimeoutError("Timeout 33000ms exceeded."), land_on=landed)

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert rendered.http_status is None
        assert rendered.final_url == landed

    async def test_the_tier_1_rung_renders_a_redirected_same_publisher_landing(self, monkeypatch):
        """The rung re-vets ``direct.url`` when it differs from the cited URL, and that re-vet
        resolves the hostname; through the module's DNS stub it is public and the render proceeds
        on the landing URL. Kept in this module so the stub is proven load-bearing here."""
        calls: list[str] = []

        async def _recording_render(url: str, **_kwargs: Any) -> None:
            calls.append(url)
            await asyncio.sleep(0)

        monkeypatch.setattr(resolution_source, "render_page", _recording_render)
        landed = "https://www.dashboard.example.com/senate"
        direct = FetchResult(url=landed, status="js_wall", text="", http_status=200, content_type="text/html")

        result = await resolution_source._rendered_rung(_PAGE_URL, direct, {}, FetchContext())

        assert result is None
        assert calls == [landed]

    async def test_the_tier_1_rung_records_an_off_host_landing_as_its_own_skip(self, monkeypatch):
        """Folded into ``renderer_unavailable`` it would point triage at the Playwright install;
        folded into ``None`` it would be invisible. The direct result stands."""

        async def _off_host(url: str, **_kwargs: Any) -> None:
            await asyncio.sleep(0)
            raise rendered_fetch.RenderOffHost(
                requested_url=url, final_url="http://10.0.0.8/status", pinned_host="dashboard.example.com"
            )

        monkeypatch.setattr(resolution_source, "render_page", _off_host)
        direct = FetchResult(url=_PAGE_URL, status="js_wall", text="", http_status=200, content_type="text/html")
        ctx = FetchContext()

        result = await resolution_source._rendered_rung(_PAGE_URL, direct, {}, ctx)

        assert result is None
        assert [attempt.skipped_reason for attempt in ctx.rungs] == ["render_off_host"]
        direct.rung_attempts = list(ctx.rungs)
        counts = resolution_source._rung_counts([direct])
        assert counts["render_off_host_skips"] == 1
        assert counts["renderer_unavailable_skips"] == 0
        assert counts["rendered_attempts"] == 0
        assert rendered_fetch._PLAYWRIGHT_WARNED is False
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False


class TestTeardown:
    """Teardown may neither wedge the render nor replace its exception. ``BrowserContext.close``
    (Playwright 1.61) does not swallow a target-closed error the way ``Browser.close`` does, and
    the next protocol call re-raises any error a detached listener stored on the connection, so
    an unguarded close in a ``finally`` replaced the cut-off unwinding through it and read as
    "the renderer is unavailable"; and it awaits its closed-future with no bound of its own."""

    async def test_a_teardown_error_does_not_replace_the_cut_off_and_still_closes_the_browser(self, monkeypatch):
        monkeypatch.setattr(rendered_fetch, "RENDER_DOM_READ_TIMEOUT_MS", 50)
        page = FakePage([], content_hangs=True)
        install_fake_playwright(
            monkeypatch,
            page,
            faults=Faults(close_error=PlaywrightError("Target page, context or browser has been closed")),
        )

        with pytest.raises(rendered_fetch.RenderTimeout):
            await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))

        assert page.teardown == ["unroute_all", "context.close", "browser.close"]
        assert rendered_fetch._PLAYWRIGHT_WARNED is False

    async def test_a_wedged_close_is_bounded_and_the_read_dom_is_still_returned(self, monkeypatch, caplog):
        monkeypatch.setattr(rendered_fetch, "RENDER_TEARDOWN_TIMEOUT_MS", 50)
        page = FakePage([])
        install_fake_playwright(monkeypatch, page, faults=Faults(close_hangs=True))

        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.rendered_fetch"):
            rendered = await rendered_fetch.render_page(
                _PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)
            )

        assert time.monotonic() - started < 1.0
        assert rendered is not None
        assert "rendered" in rendered.html
        # The wedged context close spent the whole shared budget, so the browser close is not
        # attempted on its own clock; the driver stop kills the browser either way.
        assert page.teardown == ["unroute_all", "context.close"]
        assert [message for message in caplog.messages if "did not finish" in message]

    async def test_the_teardown_steps_share_one_bound(self, monkeypatch):
        """Three separately bounded steps let a wedged browser hold a render for three bounds after
        its DOM was read, past the Tier-1 rung's cut. One budget, started by the first step that
        runs, caps the whole exit at one bound whatever the browser does."""
        monkeypatch.setattr(rendered_fetch, "RENDER_TEARDOWN_TIMEOUT_MS", 200)
        page = FakePage([])
        install_fake_playwright(monkeypatch, page, faults=Faults(close_hangs=True, browser_close_hangs=True))

        started = time.monotonic()
        rendered = await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))
        elapsed = time.monotonic() - started

        assert rendered is not None
        # One bound (200 ms, truncated to whole milliseconds), not two: the second wedged close
        # got nothing and went to the driver stop.
        assert 0.18 <= elapsed < 0.4
        assert page.teardown == ["unroute_all", "context.close"]

    async def test_the_teardown_budget_starts_with_the_first_step_not_the_launch(self, monkeypatch):
        """Computed before the launch, the budget would be spent by the time teardown runs and
        every step would be abandoned to the driver stop, even on a healthy browser."""
        monkeypatch.setattr(rendered_fetch, "RENDER_TEARDOWN_TIMEOUT_MS", 100)
        monkeypatch.setattr(rendered_fetch, "RENDER_MIN_GOTO_MS", 100)
        page = FakePage([], goto_runs_its_budget_out=True)

        rendered = await _render(monkeypatch, page, goto_timeout_ms=300)

        assert rendered is not None
        assert page.teardown == ["unroute_all", "context.close", "browser.close"]

    async def test_the_timed_out_memo_survives_a_callers_cut_during_teardown(self, monkeypatch):
        """The DOM read fires ``RenderTimeout``, the wedged close then holds the unwinding
        finallys, and the caller's own ``wait_for`` cuts in during that teardown. The cancellation
        REPLACES the propagating ``RenderTimeout``, so a handler around the call never sees it;
        the memo is written at the raise site so the cut-off URL is still remembered for the run."""
        monkeypatch.setattr(rendered_fetch, "RENDER_DOM_READ_TIMEOUT_MS", 50)
        monkeypatch.setattr(rendered_fetch, "RENDER_TEARDOWN_TIMEOUT_MS", 5_000)
        page = FakePage([], content_hangs=True)
        install_fake_playwright(monkeypatch, page, faults=Faults(close_hangs=True))

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)),
                timeout=0.3,
            )

        assert page.teardown[:2] == ["unroute_all", "context.close"]
        assert rendered_fetch.render_timed_out(_PAGE_URL, memo_scope=_TIER1_SCOPE) is True
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False

    async def test_a_failure_before_the_page_exists_still_closes_the_browser(self, monkeypatch, caplog):
        page = FakePage([])
        install_fake_playwright(monkeypatch, page, faults=Faults(new_context_error=RuntimeError("no context")))

        with caplog.at_level(logging.ERROR, logger="metaculus_bot.research.rendered_fetch"):
            rendered = await rendered_fetch.render_page(
                _PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)
            )

        assert rendered is None
        assert page.teardown == ["browser.close"]
        assert [record for record in caplog.records if record.exc_info is not None]


class TestRunScopedState:
    async def test_the_launch_cap_is_the_named_constant(self):
        gate = rendered_fetch._RENDERED_FETCH_GLOBAL_SEMAPHORE
        for _ in range(rendered_fetch.RENDER_LAUNCH_CAP):
            assert not gate.locked()
            await gate.acquire()
        assert gate.locked()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(gate.acquire(), timeout=0.05)


class TestTheDomReadIsBounded:
    """P3-1 (live QA, 2026-09-03). On ogimet.com the goto timed out at 33 s as designed, the settle
    ran, and then ``page.content()`` blocked for a further 40 s before Playwright gave up ("the
    page is navigating and changing the content"). Nothing bounded that read, so the render ran
    76 s, the Tier-1 provider's 45 s wall fired, and every page the question had already fetched
    was discarded. A DOM that has not answered in ``RENDER_DOM_READ_TIMEOUT_MS`` is one that keeps
    navigating, and the transport now says so with a ``TimeoutError`` the callers record as their
    own reason — it is not a browser that is missing or broken.
    """

    async def _render_a_hanging_dom(self, monkeypatch: pytest.MonkeyPatch) -> FakePage:
        monkeypatch.setattr(rendered_fetch, "RENDER_DOM_READ_TIMEOUT_MS", 50)
        page = FakePage([], content_hangs=True)
        install_fake_playwright(monkeypatch, page)
        with pytest.raises(TimeoutError):
            await rendered_fetch.render_page(
                _PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1), goto_timeout_ms=10_000
            )
        return page

    async def test_a_dom_read_that_hangs_is_cut_off_at_the_bound(self, monkeypatch):
        started = time.monotonic()
        await self._render_a_hanging_dom(monkeypatch)
        assert time.monotonic() - started < 2.0

    async def test_a_timed_out_read_still_tears_the_browser_down_and_frees_the_launch_slot(self, monkeypatch):
        """The 76 s render also held one of the two process-global Chromium slots throughout."""
        page = await self._render_a_hanging_dom(monkeypatch)
        assert page.teardown == ["unroute_all", "context.close", "browser.close"]
        gate = rendered_fetch._RENDERED_FETCH_GLOBAL_SEMAPHORE
        for _ in range(rendered_fetch.RENDER_LAUNCH_CAP):
            await asyncio.wait_for(gate.acquire(), timeout=0.5)

    async def test_a_timeout_is_not_the_renderer_being_unavailable(self, monkeypatch, caplog):
        """The once-per-process latch used to trip on the timeout, so a real Chromium outage
        later in the same run logged nothing."""
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.rendered_fetch"):
            await self._render_a_hanging_dom(monkeypatch)
        assert rendered_fetch._PLAYWRIGHT_WARNED is False
        messages = [record.getMessage() for record in caplog.records]
        assert not [message for message in messages if "rung unavailable" in message]
        assert [message for message in messages if "timed out" in message]

    async def test_the_cut_off_is_the_transports_own_exception(self, monkeypatch):
        """A subclass of the builtin, so both callers' ``except TimeoutError`` still catch it,
        while an OS-level ``TimeoutError`` (also a subclass, via ``OSError``) from anywhere under
        the render is NOT a cut-off render and lands in the logged boundary instead."""
        monkeypatch.setattr(rendered_fetch, "RENDER_DOM_READ_TIMEOUT_MS", 50)
        install_fake_playwright(monkeypatch, FakePage([], content_hangs=True))
        with pytest.raises(rendered_fetch.RenderTimeout):
            await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))

    async def test_an_os_timeout_under_the_render_is_a_logged_decline_not_a_cut_off(self, monkeypatch, caplog):
        """The classification, pinned: the builtin ``TimeoutError`` that is not a ``RenderTimeout``
        lands in the logged boundary with its own traceback, latches nothing, memoises nothing, and
        is never described as the DOM read being cut off. Asserting only ``None`` plus some
        ``exc_info`` record let an ``AttributeError`` raised one line earlier keep this green."""
        page = FakePage([])
        install_fake_playwright(monkeypatch, page, faults=Faults(new_page_error=TimeoutError("[Errno 60] ETIMEDOUT")))

        with caplog.at_level(logging.DEBUG, logger="metaculus_bot.research.rendered_fetch"):
            rendered = await rendered_fetch.render_page(
                _PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)
            )

        assert rendered is None
        (record,) = [record for record in caplog.records if record.exc_info is not None]
        assert record.levelno == logging.ERROR
        assert record.exc_info[0] is TimeoutError
        assert "failed unexpectedly" in record.getMessage()
        assert not [message for message in caplog.messages if "timed out reading the DOM" in message]
        assert rendered_fetch._PLAYWRIGHT_WARNED is False
        assert rendered_fetch.render_timed_out(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert page.teardown == ["context.close", "browser.close"]

    async def test_a_prompt_dom_read_is_untouched_by_the_bound(self, monkeypatch):
        monkeypatch.setattr(rendered_fetch, "RENDER_DOM_READ_TIMEOUT_MS", 50)
        rendered = await _render(monkeypatch, FakePage([]), harvest_json=False)
        assert rendered is not None
        assert "rendered" in rendered.html


class TestTheFailureBoundary:
    """Every failure out of the browser used to go through one once-per-process warn latch, so a
    bug in the render path logged one line per RUN and then went silent (a test-containment agent
    hit exactly that on 2026-09-03). The boundary still swallows — a raise here propagates out of
    the Tier-1 ``gather`` and cancels the question's other pages — but it now tells the kinds
    apart: a timeout raises for the caller to record, Playwright's own errors keep the latch, and
    anything else logs a full traceback every time it happens.
    """

    async def _render_twice(self, monkeypatch: pytest.MonkeyPatch, caplog, error: BaseException) -> None:
        install_fake_playwright(monkeypatch, FakePage([]), faults=Faults(new_page_error=error))
        with caplog.at_level(logging.DEBUG, logger="metaculus_bot.research.rendered_fetch"):
            first = await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))
            second = await rendered_fetch.render_page(
                _PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)
            )
        assert first is None
        assert second is None

    async def test_a_playwright_error_declines_and_warns_once(self, monkeypatch, caplog):
        await self._render_twice(monkeypatch, caplog, PlaywrightError("Browser closed unexpectedly"))
        assert rendered_fetch._PLAYWRIGHT_WARNED is True
        unavailable = [record for record in caplog.records if "rung unavailable" in record.getMessage()]
        assert len(unavailable) == 1
        assert unavailable[0].levelno == logging.WARNING

    async def test_an_unexpected_error_declines_with_a_traceback_every_time_and_never_latches(
        self, monkeypatch, caplog
    ):
        await self._render_twice(monkeypatch, caplog, RuntimeError("a bug in the render path"))
        assert rendered_fetch._PLAYWRIGHT_WARNED is False
        errors = [record for record in caplog.records if record.levelno == logging.ERROR]
        assert len(errors) == 2
        assert all(record.exc_info is not None and record.exc_info[0] is RuntimeError for record in errors)
        assert not [record for record in caplog.records if "rung unavailable" in record.getMessage()]
