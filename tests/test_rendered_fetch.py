"""The shared headless-Chromium transport: its gates, its bounds, and the JSON it harvests.

The transport's SSRF half (DNS pinning, the per-request route guard) is pinned by
``tests/test_agentic_tools.py``, which drove it before it moved out of ``agentic/tools.py`` and
still owns those cases. What lives here is what the move ADDED and what the transport owns for
both callers: the XHR harvest and its bounds, the two render memos and their scoping, the
navigation budget recomputed once the gates are held, the DOM ceiling, the main-frame status,
the browser-context hardening, and the run-scoped state reset.

Nothing here launches a browser. Playwright is faked through ``sys.modules`` the same way the
agentic suite fakes it. The fake page fires its response handlers the way pyee does — call, do
not await — so a listener that hands back a coroutine leaves a detached task behind exactly as
the real ``Page.on`` would, and the harvest tests exercise that interleaving rather than a
one-handler-at-a-time serialisation that would make every race look sound.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from metaculus_bot.research import rendered_fetch, resolution_source
from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.derived_api import DerivedEndpoint, derived_api_lead, largest_json
from metaculus_bot.research.rendered_fetch import HarvestedJson, RenderedPage
from metaculus_bot.research.resolution_fetch_result import FetchResult
from metaculus_bot.research.resolution_source import FetchContext

_PAGE_URL = "https://dashboard.example.com/senate"
_DOM = "<!doctype html><html><head><title>Dashboard</title></head><body><p>rendered</p></body></html>"
_EMPTY_DOM = "<!doctype html><html><head></head><body></body></html>"
_TIER1_SCOPE = "resolution_source"
_V2_SCOPE = "gap_fill_v2"
_RENDER_TAIL_MS = rendered_fetch.RENDER_SETTLE_MS + rendered_fetch.RENDER_DOM_READ_TIMEOUT_MS


class _PlaywrightError(Exception):
    """Stands in for playwright.async_api.Error."""


@pytest.fixture(autouse=True)
def _reset_state():
    rendered_fetch.reset_render_state()
    yield
    rendered_fetch.reset_render_state()


class _FakeResponse:
    """One response event. ``body()`` always yields once, because Playwright's is a round trip to
    the driver: every handler that passed the count check is suspended in it at the same time,
    which is the interleaving the post-read re-check exists for."""

    def __init__(
        self,
        url: str,
        *,
        content_type: str,
        body: bytes,
        raises: bool = False,
        body_delay_s: float = 0.0,
        declared_length: int | None = None,
    ) -> None:
        self.url = url
        self.headers = {"content-type": content_type}
        if declared_length is not None:
            self.headers["content-length"] = str(declared_length)
        self._body = body
        self._raises = raises
        self._body_delay_s = body_delay_s
        self.body_reads = 0
        self.body_read_cancelled = False

    async def body(self) -> bytes:
        self.body_reads += 1
        try:
            await asyncio.sleep(self._body_delay_s)
        except asyncio.CancelledError:
            self.body_read_cancelled = True
            raise
        if self._raises:
            raise _PlaywrightError("target closed")
        return self._body


class _FakePage:
    """A page that replays a fixed list of response events during ``goto``.

    ``content_hangs`` is the ogimet shape (P3-1): a DOM read that never answers because the page
    keeps navigating. ``goto_raises`` is the salvage shape: the navigation times out with the DOM
    already rendered. ``teardown`` records the close sequence the context and browser ran, so a
    test can assert the browser was still torn down after a failure mid-render.
    """

    def __init__(
        self,
        responses: list[_FakeResponse],
        *,
        html: str = _DOM,
        content_hangs: bool = False,
        goto_raises: BaseException | None = None,
        status: int = 200,
    ) -> None:
        self._responses = responses
        self._html = html
        self._content_hangs = content_hangs
        self._goto_raises = goto_raises
        self._status = status
        self._handlers: list[Any] = []
        self.detached_handler_tasks: list[asyncio.Future[Any]] = []
        self.goto_calls: list[dict[str, Any]] = []
        self.context_kwargs: dict[str, Any] = {}
        self.teardown: list[str] = []

    def on(self, event: str, handler: Any) -> None:
        assert event == "response"
        self._handlers.append(handler)

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> Any:  # noqa: ASYNC109  # Playwright's own signature; this stands in for it
        self.goto_calls.append({"url": url, "wait_until": wait_until, "timeout": timeout})
        # pyee's dispatch: call the listener; a coroutine comes back wrapped in ensure_future and
        # is never awaited by anyone, so it is still pending when goto returns.
        for response in self._responses:
            for handler in self._handlers:
                result = handler(response)
                if asyncio.iscoroutine(result):
                    self.detached_handler_tasks.append(asyncio.ensure_future(result))
        if self._goto_raises is not None:
            raise self._goto_raises
        return SimpleNamespace(headers={"content-type": "text/html"}, status=self._status)

    async def wait_for_timeout(self, ms: int) -> None:
        del ms

    async def content(self) -> str:
        if self._content_hangs:
            await asyncio.Event().wait()
        # The real read is a round trip; yielding here lets the detached handler tasks run
        # between goto and the snapshot, which is where the un-joined harvest lost bodies.
        await asyncio.sleep(0)
        return self._html


@dataclass
class _Faults:
    """Where the fake browser misbehaves. ``new_page_error`` / ``new_context_error`` fail the
    render INSIDE the gates, which is the failure-boundary shape the warn latch was written for;
    ``close_error`` and ``close_hangs`` misbehave in teardown, where the real
    ``BrowserContext.close`` neither swallows a target-closed error nor bounds its wait."""

    new_page_error: BaseException | None = None
    new_context_error: BaseException | None = None
    close_error: BaseException | None = None
    close_hangs: bool = False


class _FakeContext:
    def __init__(self, page: _FakePage, faults: _Faults) -> None:
        self._page = page
        self._faults = faults

    async def route(self, pattern: str, handler: Any) -> None:
        del pattern, handler

    async def new_page(self) -> _FakePage:
        if self._faults.new_page_error is not None:
            raise self._faults.new_page_error
        return self._page

    async def unroute_all(self, *, behavior: str) -> None:
        del behavior
        self._page.teardown.append("unroute_all")

    async def close(self) -> None:
        self._page.teardown.append("context.close")
        if self._faults.close_hangs:
            await asyncio.Event().wait()
        if self._faults.close_error is not None:
            raise self._faults.close_error


class _FakeBrowser:
    def __init__(self, page: _FakePage, faults: _Faults) -> None:
        self._page = page
        self._faults = faults

    async def new_context(self, **kwargs: Any) -> _FakeContext:
        self._page.context_kwargs = kwargs
        if self._faults.new_context_error is not None:
            raise self._faults.new_context_error
        return _FakeContext(self._page, self._faults)

    async def close(self) -> None:
        self._page.teardown.append("browser.close")


def _install_fake_playwright(
    monkeypatch: pytest.MonkeyPatch, page: _FakePage, *, faults: _Faults | None = None
) -> list[list[str]]:
    """Wire the fakes in; ``faults`` says where the browser should misbehave (see :class:`_Faults`)."""
    launch_args: list[list[str]] = []
    browser_faults = faults or _Faults()

    class _FakeChromium:
        async def launch(self, *, headless: bool, args: list[str] | None = None) -> _FakeBrowser:
            del headless
            launch_args.append(list(args or []))
            return _FakeBrowser(page, browser_faults)

    class _FakePlaywrightManager:
        chromium = _FakeChromium()

        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *_exc: Any) -> None:
            return None

    monkeypatch.setitem(
        sys.modules,
        "playwright.async_api",
        SimpleNamespace(async_playwright=_FakePlaywrightManager, Error=_PlaywrightError),
    )
    monkeypatch.setattr(
        rendered_fetch,
        "_resolve_pinned_host",
        _async_return(("dashboard.example.com", "93.184.216.34")),
    )
    return launch_args


def _async_return(value: Any):
    async def _call(*_args: Any, **_kwargs: Any) -> Any:
        await asyncio.sleep(0)
        return value

    return _call


async def _render(
    monkeypatch: pytest.MonkeyPatch,
    page: _FakePage,
    *,
    harvest_json: bool = True,
    goto_timeout_ms: int = 10_000,
    deadline_monotonic_s: float | None = None,
):
    _install_fake_playwright(monkeypatch, page)
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
        page = _FakePage([_FakeResponse(f"{_PAGE_URL}/api/series", content_type="application/json", body=body)])

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert [harvested.url for harvested in rendered.json_responses] == [f"{_PAGE_URL}/api/series"]

    async def test_harvesting_is_off_unless_the_caller_asks(self, monkeypatch):
        """The bodies buffer inside the render task alongside a 100-300 MB browser, so only a
        caller with a use for a derived feed pays for them."""
        page = _FakePage([_FakeResponse(f"{_PAGE_URL}/api/series", content_type="application/json", body=_json_body())])

        rendered = await _render(monkeypatch, page, harvest_json=False)

        assert rendered is not None
        assert rendered.json_responses == ()

    async def test_a_cross_origin_response_is_not_harvested(self, monkeypatch):
        """A stranger's JSON must never become the cited page's content."""
        page = _FakePage(
            [_FakeResponse("https://ads.tracker.test/beacon.json", content_type="application/json", body=_json_body())]
        )

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert rendered.json_responses == ()

    async def test_a_non_json_response_is_not_harvested(self, monkeypatch):
        body = b"x" * 4000
        page = _FakePage([_FakeResponse(f"{_PAGE_URL}/app.js", content_type="application/javascript", body=body)])

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert rendered.json_responses == ()

    async def test_a_tiny_json_body_is_not_harvested(self, monkeypatch):
        """Below the floor a JSON body is a ping, a feature flag or an empty envelope."""
        page = _FakePage(
            [_FakeResponse(f"{_PAGE_URL}/api/flags", content_type="application/json", body=b'{"ok":true}')]
        )

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert rendered.json_responses == ()

    async def test_an_oversized_json_body_is_not_harvested(self, monkeypatch):
        body = b"[" + b"1," * rendered_fetch.HARVEST_MAX_BODY_BYTES + b"]"
        page = _FakePage([_FakeResponse(f"{_PAGE_URL}/api/all", content_type="application/json", body=body)])

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert rendered.json_responses == ()

    async def test_a_body_declared_over_the_cap_is_never_read(self, monkeypatch):
        """``Response.body()`` materialises the whole body (base64 over the driver pipe, then
        decoded) before any size test could run on it, so the declared size is the only bound
        that can keep a 60 MB GeoJSON out of memory. The post-read test stays as the backstop
        for absent, compressed or lying headers."""
        response = _FakeResponse(
            f"{_PAGE_URL}/api/geo",
            content_type="application/json",
            body=_json_body(),
            declared_length=rendered_fetch.HARVEST_MAX_BODY_BYTES + 1,
        )

        rendered = await _render(monkeypatch, _FakePage([response]))

        assert rendered is not None
        assert rendered.json_responses == ()
        assert response.body_reads == 0

    async def test_a_body_still_in_flight_when_goto_returns_is_harvested(self, monkeypatch):
        """Every ``page.on`` firing is its own task and nothing used to join them, so a body that
        arrived a beat late was appended after the snapshot and then lost at teardown — exactly
        the derived-API rung's payload, and the miss stuck because the caller memoised it."""
        response = _FakeResponse(
            f"{_PAGE_URL}/api/series", content_type="application/json", body=_json_body(), body_delay_s=0.05
        )

        rendered = await _render(monkeypatch, _FakePage([response]))

        assert rendered is not None
        assert [harvested.url for harvested in rendered.json_responses] == [f"{_PAGE_URL}/api/series"]

    async def test_a_body_read_that_outlives_the_dom_read_bound_is_dropped_without_delaying_the_render(
        self, monkeypatch
    ):
        """The drain runs INSIDE the DOM-read bound, never after it: the harvest is opportunistic
        and may not lengthen a render whose real product is the DOM."""
        monkeypatch.setattr(rendered_fetch, "RENDER_DOM_READ_TIMEOUT_MS", 100)
        response = _FakeResponse(
            f"{_PAGE_URL}/api/slow", content_type="application/json", body=_json_body(), body_delay_s=5.0
        )

        started = time.monotonic()
        rendered = await _render(monkeypatch, _FakePage([response]))

        assert time.monotonic() - started < 1.0
        assert rendered is not None
        assert "rendered" in rendered.html
        assert rendered.json_responses == ()
        assert response.body_read_cancelled is True

    async def test_the_response_count_is_capped_when_every_read_is_in_flight_at_once(self, monkeypatch):
        """All N handlers pass the count check together and are then all suspended in ``body()``,
        so a check made only BEFORE the read bounds nothing."""
        page = _FakePage(
            [
                _FakeResponse(f"{_PAGE_URL}/api/{index}", content_type="application/json", body=_json_body())
                for index in range(rendered_fetch.HARVEST_MAX_RESPONSES + 4)
            ]
        )

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert len(rendered.json_responses) == rendered_fetch.HARVEST_MAX_RESPONSES

    async def test_a_body_read_that_races_teardown_is_dropped_not_raised(self, monkeypatch):
        """Opportunistic discovery attached to a render whose real product is the DOM: a body
        we could not read must never be able to fail the render."""
        page = _FakePage(
            [
                _FakeResponse(f"{_PAGE_URL}/api/gone", content_type="application/json", body=_json_body(), raises=True),
                _FakeResponse(f"{_PAGE_URL}/api/series", content_type="application/json", body=_json_body()),
            ]
        )

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert [harvested.url for harvested in rendered.json_responses] == [f"{_PAGE_URL}/api/series"]
        assert "rendered" in rendered.html


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
        launch_args = _install_fake_playwright(monkeypatch, _FakePage([]))

        declined = await rendered_fetch.render_page(_PAGE_URL, memo_scope=_V2_SCOPE, host_gate=asyncio.Semaphore(1))
        rendered = await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))

        assert declined is None
        assert rendered is not None
        assert len(launch_args) == 1

    async def test_a_render_the_transport_cut_off_is_memoised_and_raises_again_without_launching(self, monkeypatch):
        """A second question citing a page that already ran out the clock must record
        ``render_timeout`` again, not ``renderer_unavailable`` — the count the operator reads as
        the Chromium install having failed. So the transport re-raises the memoised timeout
        instead of folding it into the ``None`` every other decline shares."""
        monkeypatch.setattr(rendered_fetch, "RENDER_DOM_READ_TIMEOUT_MS", 50)
        launch_args = _install_fake_playwright(monkeypatch, _FakePage([], content_hangs=True))
        with pytest.raises(TimeoutError):
            await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))
        assert rendered_fetch.render_timed_out(_PAGE_URL, memo_scope=_TIER1_SCOPE) is True
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert len(launch_args) == 1

        with pytest.raises(TimeoutError):
            await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))

        assert len(launch_args) == 1
        # The other path's clock is its own.
        assert rendered_fetch.render_timed_out(_PAGE_URL, memo_scope=_V2_SCOPE) is False

    async def test_a_cut_in_the_queue_is_not_memoised(self, monkeypatch):
        """The caller's outer bound also fires while the render is still waiting on the two
        gates, which says nothing about the page. Only the transport knows a browser ran, so only
        the transport writes the timed-out memo — a queue cut leaves the URL live for the next
        question."""
        launch_args = _install_fake_playwright(monkeypatch, _FakePage([]))
        gate = rendered_fetch._RENDERED_FETCH_GLOBAL_SEMAPHORE
        for _ in range(rendered_fetch.RENDER_LAUNCH_CAP):
            await gate.acquire()

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)),
                timeout=0.1,
            )

        assert launch_args == []
        assert rendered_fetch.render_timed_out(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        for _ in range(rendered_fetch.RENDER_LAUNCH_CAP):
            gate.release()
        rendered = await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))
        assert rendered is not None

    async def test_the_memo_is_bounded(self):
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
        launch_args = _install_fake_playwright(monkeypatch, _FakePage([], html=_EMPTY_DOM))

        v2_result = await agentic_tools._try_rendered_fetch(_PAGE_URL)

        assert v2_result is not None
        assert v2_result.status == "error"
        assert v2_result.method == "rendered"
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_V2_SCOPE) is True
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert len(launch_args) == 1

        direct = FetchResult(url=_PAGE_URL, status="js_wall", text="", http_status=200, content_type="text/html")
        ctx = FetchContext()
        tier1_result = await resolution_source._rendered_rung(_PAGE_URL, direct, {}, ctx)

        assert len(launch_args) == 2
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
        page = _FakePage([])
        launch_args = _install_fake_playwright(monkeypatch, page)
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

        assert launch_args == []
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
        page = _FakePage([])
        _install_fake_playwright(monkeypatch, page)
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
        page = _FakePage([], goto_raises=_PlaywrightError("Timeout exceeded"))
        deadline_s = 20.0
        started = time.monotonic()

        rendered = await _render(monkeypatch, page, goto_timeout_ms=18_000, deadline_monotonic_s=started + deadline_s)

        assert rendered is not None
        assert "rendered" in rendered.html
        (call,) = page.goto_calls
        assert call["timeout"] + _RENDER_TAIL_MS <= int(deadline_s * 1000)
        assert call["timeout"] >= rendered_fetch.RENDER_MIN_GOTO_MS

    async def test_a_call_without_a_deadline_keeps_the_callers_goto_budget(self, monkeypatch):
        """Gap-fill v2's shape: its own ceilings bound the call, so the transport has nothing to
        recompute against and the caller's figure stands."""
        page = _FakePage([])

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
        assert 19.5 < call["deadline_monotonic_s"] - call["called_at"] <= 20.0


class TestTheDomCeiling:
    """The rendered DOM was the one body path that escaped the 5 MiB ceiling every aiohttp fetch
    enforces, and the Tier-1 caller then copies it three more times before trafilatura parses it
    into a tree several times bigger. The transport declines above a named ceiling, measured in
    characters so the check does not itself make the copy it exists to prevent."""

    async def test_a_dom_over_the_ceiling_is_declined_and_the_browser_torn_down(self, monkeypatch, caplog):
        monkeypatch.setattr(rendered_fetch, "RENDERED_DOM_MAX_CHARS", 100)
        page = _FakePage([], html="<html><body>" + "x" * 200 + "</body></html>")

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.rendered_fetch"):
            rendered = await _render(monkeypatch, page)

        assert rendered is None
        assert page.teardown == ["unroute_all", "context.close", "browser.close"]
        assert rendered_fetch._PLAYWRIGHT_WARNED is False
        assert rendered_fetch.rendered_to_nothing(_PAGE_URL, memo_scope=_TIER1_SCOPE) is False
        assert [message for message in caplog.messages if "over the" in message and "ceiling" in message]

    async def test_a_dom_at_the_ceiling_is_read(self, monkeypatch):
        html = "<html><body>" + "x" * 50 + "</body></html>"
        monkeypatch.setattr(rendered_fetch, "RENDERED_DOM_MAX_CHARS", len(html))

        rendered = await _render(monkeypatch, _FakePage([], html=html))

        assert rendered is not None
        assert rendered.html == html


class TestTheBrowserContext:
    async def test_service_workers_are_blocked(self, monkeypatch):
        """``browser_context.route`` does not intercept requests a service worker makes, so a
        worker could dial past the SSRF route guard; Playwright's own remedy is to block them
        whenever interception is in use."""
        page = _FakePage([])

        await _render(monkeypatch, page)

        assert page.context_kwargs["service_workers"] == "block"
        assert page.context_kwargs["user_agent"]


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
        launch_args = _install_fake_playwright(monkeypatch, _FakePage([]))
        monkeypatch.setattr(rendered_fetch, "_resolve_pinned_host", real_resolve)

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.rendered_fetch"):
            rendered = await rendered_fetch.render_page(
                "https://münchen.example/x", memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)
            )

        assert rendered is None
        assert launch_args == []
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
        rendered = await _render(monkeypatch, _FakePage([], status=403))
        assert rendered is not None
        assert rendered.http_status == 403

    async def test_a_salvaged_dom_carries_no_status(self, monkeypatch):
        """No response object survives a goto timeout, which is the salvage path."""
        rendered = await _render(monkeypatch, _FakePage([], goto_raises=_PlaywrightError("Timeout exceeded")))
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

    async def test_a_200_from_the_browser_still_classifies(self, monkeypatch):
        async def _ok_render(url: str, **_kwargs: Any) -> RenderedPage:
            await asyncio.sleep(0)
            return RenderedPage(url=url, content_type="text/html", html=self._CHALLENGE, http_status=200)

        monkeypatch.setattr(resolution_source, "render_page", _ok_render)
        direct = FetchResult(url=_PAGE_URL, status="js_wall", text="", http_status=200, content_type="text/html")

        result = await resolution_source._rendered_rung(_PAGE_URL, direct, {}, FetchContext())

        assert result is not None
        assert result.status == "success"


class TestTeardown:
    """Teardown may neither wedge the render nor replace its exception. ``BrowserContext.close``
    (Playwright 1.61) does not swallow a target-closed error the way ``Browser.close`` does, and
    the next protocol call re-raises any error a detached listener stored on the connection, so
    an unguarded close in a ``finally`` replaced the cut-off unwinding through it and read as
    "the renderer is unavailable"; and it awaits its closed-future with no bound of its own."""

    async def test_a_teardown_error_does_not_replace_the_cut_off_and_still_closes_the_browser(self, monkeypatch):
        monkeypatch.setattr(rendered_fetch, "RENDER_DOM_READ_TIMEOUT_MS", 50)
        page = _FakePage([], content_hangs=True)
        _install_fake_playwright(
            monkeypatch,
            page,
            faults=_Faults(close_error=_PlaywrightError("Target page, context or browser has been closed")),
        )

        with pytest.raises(rendered_fetch.RenderTimeout):
            await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))

        assert page.teardown == ["unroute_all", "context.close", "browser.close"]
        assert rendered_fetch._PLAYWRIGHT_WARNED is False

    async def test_a_wedged_close_is_bounded_and_the_read_dom_is_still_returned(self, monkeypatch, caplog):
        monkeypatch.setattr(rendered_fetch, "RENDER_TEARDOWN_TIMEOUT_MS", 50)
        page = _FakePage([])
        _install_fake_playwright(monkeypatch, page, faults=_Faults(close_hangs=True))

        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.rendered_fetch"):
            rendered = await rendered_fetch.render_page(
                _PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)
            )

        assert time.monotonic() - started < 1.0
        assert rendered is not None
        assert "rendered" in rendered.html
        assert page.teardown == ["unroute_all", "context.close", "browser.close"]
        assert [message for message in caplog.messages if "did not finish" in message]

    async def test_a_failure_before_the_page_exists_still_closes_the_browser(self, monkeypatch, caplog):
        page = _FakePage([])
        _install_fake_playwright(monkeypatch, page, faults=_Faults(new_context_error=RuntimeError("no context")))

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


class TestTheDomReadIsBounded:
    """P3-1 (live QA, 2026-09-03). On ogimet.com the goto timed out at 33 s as designed, the settle
    ran, and then ``page.content()`` blocked for a further 40 s before Playwright gave up ("the
    page is navigating and changing the content"). Nothing bounded that read, so the render ran
    76 s, the Tier-1 provider's 45 s wall fired, and every page the question had already fetched
    was discarded. A DOM that has not answered in ``RENDER_DOM_READ_TIMEOUT_MS`` is one that keeps
    navigating, and the transport now says so with a ``TimeoutError`` the callers record as their
    own reason — it is not a browser that is missing or broken.
    """

    async def _render_a_hanging_dom(self, monkeypatch: pytest.MonkeyPatch) -> _FakePage:
        monkeypatch.setattr(rendered_fetch, "RENDER_DOM_READ_TIMEOUT_MS", 50)
        page = _FakePage([], content_hangs=True)
        _install_fake_playwright(monkeypatch, page)
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
        _install_fake_playwright(monkeypatch, _FakePage([], content_hangs=True))
        with pytest.raises(rendered_fetch.RenderTimeout):
            await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))

    async def test_an_os_timeout_under_the_render_is_a_logged_decline_not_a_cut_off(self, monkeypatch, caplog):
        _install_fake_playwright(
            monkeypatch, _FakePage([]), faults=_Faults(new_page_error=TimeoutError("[Errno 60] ETIMEDOUT"))
        )
        with caplog.at_level(logging.ERROR, logger="metaculus_bot.research.rendered_fetch"):
            rendered = await rendered_fetch.render_page(
                _PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)
            )
        assert rendered is None
        assert [record for record in caplog.records if record.exc_info is not None]

    async def test_a_prompt_dom_read_is_untouched_by_the_bound(self, monkeypatch):
        monkeypatch.setattr(rendered_fetch, "RENDER_DOM_READ_TIMEOUT_MS", 50)
        rendered = await _render(monkeypatch, _FakePage([]), harvest_json=False)
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
        _install_fake_playwright(monkeypatch, _FakePage([]), faults=_Faults(new_page_error=error))
        with caplog.at_level(logging.DEBUG, logger="metaculus_bot.research.rendered_fetch"):
            first = await rendered_fetch.render_page(_PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1))
            second = await rendered_fetch.render_page(
                _PAGE_URL, memo_scope=_TIER1_SCOPE, host_gate=asyncio.Semaphore(1)
            )
        assert first is None
        assert second is None

    async def test_a_playwright_error_declines_and_warns_once(self, monkeypatch, caplog):
        await self._render_twice(monkeypatch, caplog, _PlaywrightError("Browser closed unexpectedly"))
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
