"""The shared headless-Chromium transport: its gates, and the JSON it harvests during a render.

The transport's SSRF half (DNS pinning, the per-request route guard) is pinned by
``tests/test_agentic_tools.py``, which drove it before it moved out of ``agentic/tools.py`` and
still owns those cases. What lives here is what the move ADDED: the XHR harvest, its bounds,
and the run-scoped state reset.

Nothing here launches a browser. Playwright is faked through ``sys.modules`` the same way the
agentic suite fakes it, so the harvest is exercised against a fake page that emits response
events. The same fakes drive the transport's two bounds — the DOM-read cap and the failure
boundary around the whole render — which are the transport's own and so are pinned here.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest

from metaculus_bot.research import rendered_fetch
from metaculus_bot.research.derived_api import DerivedEndpoint, derived_api_lead, largest_json
from metaculus_bot.research.rendered_fetch import HarvestedJson

_PAGE_URL = "https://dashboard.example.com/senate"
_DOM = "<!doctype html><html><head><title>Dashboard</title></head><body><p>rendered</p></body></html>"


class _PlaywrightError(Exception):
    """Stands in for playwright.async_api.Error."""


@pytest.fixture(autouse=True)
def _reset_state():
    rendered_fetch.reset_render_state()
    yield
    rendered_fetch.reset_render_state()


class _FakeResponse:
    def __init__(self, url: str, *, content_type: str, body: bytes, raises: bool = False) -> None:
        self.url = url
        self.headers = {"content-type": content_type}
        self._body = body
        self._raises = raises

    async def body(self) -> bytes:
        if self._raises:
            raise _PlaywrightError("target closed")
        return self._body


class _FakePage:
    """A page that replays a fixed list of response events during ``goto``.

    ``content_hangs`` is the ogimet shape (P3-1): a DOM read that never answers because the page
    keeps navigating. ``teardown`` records the close sequence the context and browser ran, so a
    test can assert the browser was still torn down after a failure mid-render.
    """

    def __init__(self, responses: list[_FakeResponse], *, html: str = _DOM, content_hangs: bool = False) -> None:
        self._responses = responses
        self._html = html
        self._content_hangs = content_hangs
        self._handlers: list[Any] = []
        self.teardown: list[str] = []

    def on(self, event: str, handler: Any) -> None:
        assert event == "response"
        self._handlers.append(handler)

    async def goto(self, url: str, *, wait_until: str, timeout: int) -> Any:  # noqa: ASYNC109  # Playwright's own signature; this stands in for it
        del url, wait_until, timeout
        for response in self._responses:
            for handler in self._handlers:
                await handler(response)
        return SimpleNamespace(headers={"content-type": "text/html"})

    async def wait_for_timeout(self, ms: int) -> None:
        del ms

    async def content(self) -> str:
        if self._content_hangs:
            await asyncio.Event().wait()
        return self._html


class _FakeContext:
    def __init__(self, page: _FakePage, *, new_page_error: BaseException | None = None) -> None:
        self._page = page
        self._new_page_error = new_page_error

    async def route(self, pattern: str, handler: Any) -> None:
        del pattern, handler

    async def new_page(self) -> _FakePage:
        if self._new_page_error is not None:
            raise self._new_page_error
        return self._page

    async def unroute_all(self, *, behavior: str) -> None:
        del behavior
        self._page.teardown.append("unroute_all")

    async def close(self) -> None:
        self._page.teardown.append("context.close")


class _FakeBrowser:
    def __init__(self, page: _FakePage, *, new_page_error: BaseException | None = None) -> None:
        self._page = page
        self._new_page_error = new_page_error

    async def new_context(self, **_kwargs: Any) -> _FakeContext:
        return _FakeContext(self._page, new_page_error=self._new_page_error)

    async def close(self) -> None:
        self._page.teardown.append("browser.close")


def _install_fake_playwright(
    monkeypatch: pytest.MonkeyPatch, page: _FakePage, *, new_page_error: BaseException | None = None
) -> list[list[str]]:
    """Wire the fakes in. ``new_page_error`` makes the browser fail INSIDE the render, after the
    launch gates are held, which is the failure-boundary shape the warn latch was written for."""
    launch_args: list[list[str]] = []

    class _FakeChromium:
        async def launch(self, *, headless: bool, args: list[str] | None = None) -> _FakeBrowser:
            del headless
            launch_args.append(list(args or []))
            return _FakeBrowser(page, new_page_error=new_page_error)

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


async def _render(monkeypatch: pytest.MonkeyPatch, page: _FakePage, *, harvest_json: bool = True):
    _install_fake_playwright(monkeypatch, page)
    return await rendered_fetch.render_page(
        _PAGE_URL, host_gate=asyncio.Semaphore(1), goto_timeout_ms=10_000, harvest_json=harvest_json
    )


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
        body = b'{"series":[' + b'{"v":1},' * 40 + b"]}"
        page = _FakePage([_FakeResponse(f"{_PAGE_URL}/api/series", content_type="application/json", body=body)])

        rendered = await _render(monkeypatch, page, harvest_json=False)

        assert rendered is not None
        assert rendered.json_responses == ()

    async def test_a_cross_origin_response_is_not_harvested(self, monkeypatch):
        """A stranger's JSON must never become the cited page's content."""
        body = b'{"data":[' + b'{"v":1},' * 40 + b"]}"
        page = _FakePage(
            [_FakeResponse("https://ads.tracker.test/beacon.json", content_type="application/json", body=body)]
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

    async def test_the_response_count_is_capped(self, monkeypatch):
        body = b'{"series":[' + b'{"v":1},' * 40 + b"]}"
        page = _FakePage(
            [
                _FakeResponse(f"{_PAGE_URL}/api/{index}", content_type="application/json", body=body)
                for index in range(rendered_fetch.HARVEST_MAX_RESPONSES + 4)
            ]
        )

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert len(rendered.json_responses) == rendered_fetch.HARVEST_MAX_RESPONSES

    async def test_a_body_read_that_races_teardown_is_dropped_not_raised(self, monkeypatch):
        """Opportunistic discovery attached to a render whose real product is the DOM: a body
        we could not read must never be able to fail the render."""
        body = b'{"series":[' + b'{"v":1},' * 40 + b"]}"
        page = _FakePage(
            [
                _FakeResponse(f"{_PAGE_URL}/api/gone", content_type="application/json", body=body, raises=True),
                _FakeResponse(f"{_PAGE_URL}/api/series", content_type="application/json", body=body),
            ]
        )

        rendered = await _render(monkeypatch, page)

        assert rendered is not None
        assert [harvested.url for harvested in rendered.json_responses] == [f"{_PAGE_URL}/api/series"]
        assert "rendered" in rendered.html


class TestHarvestableHost:
    """Same origin, either direction of the subdomain relation, or an allow-listed CDN."""

    @pytest.mark.parametrize(
        ("response_host", "page_host", "expected"),
        [
            ("x.gov", "x.gov", True),
            ("api.x.gov", "www.x.gov", False),
            ("api.x.gov", "x.gov", True),
            ("x.gov", "www.x.gov", True),
            ("static.dwcdn.net", "tracker.example.com", True),
            ("evil.test", "x.gov", False),
            ("a.co.uk", "b.co.uk", False),
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
            await rendered_fetch.render_page(_PAGE_URL, host_gate=asyncio.Semaphore(1), goto_timeout_ms=10_000)
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
        for _ in range(2):
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
        _install_fake_playwright(monkeypatch, _FakePage([]), new_page_error=error)
        with caplog.at_level(logging.DEBUG, logger="metaculus_bot.research.rendered_fetch"):
            first = await rendered_fetch.render_page(_PAGE_URL, host_gate=asyncio.Semaphore(1))
            second = await rendered_fetch.render_page(_PAGE_URL, host_gate=asyncio.Semaphore(1))
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
