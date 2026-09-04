"""The shared headless-Chromium transport: its gates, and the JSON it harvests during a render.

The transport's SSRF half (DNS pinning, the per-request route guard) is pinned by
``tests/test_agentic_tools.py``, which drove it before it moved out of ``agentic/tools.py`` and
still owns those cases. What lives here is what the move ADDED: the XHR harvest, its bounds,
and the run-scoped state reset.

Nothing here launches a browser. Playwright is faked through ``sys.modules`` the same way the
agentic suite fakes it, so the harvest is exercised against a fake page that emits response
events.
"""

from __future__ import annotations

import asyncio
import sys
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
    """A page that replays a fixed list of response events during ``goto``."""

    def __init__(self, responses: list[_FakeResponse], *, html: str = _DOM) -> None:
        self._responses = responses
        self._html = html
        self._handlers: list[Any] = []

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
        return self._html


class _FakeContext:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    async def route(self, pattern: str, handler: Any) -> None:
        del pattern, handler

    async def new_page(self) -> _FakePage:
        return self._page

    async def unroute_all(self, *, behavior: str) -> None:
        del behavior

    async def close(self) -> None:
        return None


class _FakeBrowser:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    async def new_context(self, **_kwargs: Any) -> _FakeContext:
        return _FakeContext(self._page)

    async def close(self) -> None:
        return None


def _install_fake_playwright(monkeypatch: pytest.MonkeyPatch, page: _FakePage) -> list[list[str]]:
    launch_args: list[list[str]] = []

    class _FakeChromium:
        async def launch(self, *, headless: bool, args: list[str] | None = None) -> _FakeBrowser:
            del headless
            launch_args.append(list(args or []))
            return _FakeBrowser(page)

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
