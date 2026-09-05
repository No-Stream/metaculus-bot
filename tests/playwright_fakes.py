"""One fake Playwright object graph for every test that drives ``rendered_fetch.render_page``.

The transport is faked through ``sys.modules["playwright.async_api"]`` rather than by patching
``render_page``: these tests exercise the launch, the route guard, the navigation budget, the
DOM read, the XHR harvest and the teardown, so the double has to honour the slice of
Playwright's API the transport touches. Nothing here launches a browser, and the suite's
autouse ``_block_native_egress`` refuses a real launch anyway.

One copy here so they can't drift. Before this module the graph was hand-copied nine times
across ``tests/test_agentic_tools.py``, and the transport change that put ``http_status`` on
``RenderedPage`` cost seven byte-identical ``status=200`` edits, one per copy, while the single
parameterised page in ``tests/test_rendered_fetch.py`` absorbed it with one default. A test
with bespoke behaviour (a launch that blocks on a barrier, a goto that drives the route guard,
a manager that records when the driver started) SUBCLASSES the class it needs and hands the
instance to :func:`install_fake_playwright` rather than growing a hook parameter.

The error class is Playwright's real ``Error``: the transport catches whatever class the fake
module exposes as ``Error``, and tests that raise Playwright's real ``TimeoutError`` (a subclass
of that ``Error``) from ``goto`` need the salvage path to recognise it.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from playwright.async_api import Error as PlaywrightError

from metaculus_bot.research import rendered_fetch

__all__ = [
    "PIN_THE_REQUESTED_HOST",
    "FakeBrowser",
    "FakeChromium",
    "FakeContext",
    "FakePage",
    "FakePlaywrightManager",
    "FakeResponse",
    "FakeWebSocketRoute",
    "Faults",
    "PlaywrightError",
    "install_fake_playwright",
]

DEFAULT_DOM = "<!doctype html><html><head><title>Dashboard</title></head><body><p>rendered</p></body></html>"
# The one public address every default pin resolves to. Which HOST is pinned is derived from the
# URL under test (see `install_fake_playwright`), never fixed here.
DEFAULT_PIN_IP = "93.184.216.34"


class _PinTheRequestedHost:
    """The ``pinned`` default: pin whatever host the transport asks for, as the real resolver does."""


PIN_THE_REQUESTED_HOST = _PinTheRequestedHost()


class FakeResponse:
    """One response event. ``body()`` always yields once, because Playwright's is a round trip to
    the driver: every read task that reaches it would be suspended in it at the same time, which
    is the interleaving the harvest's serialised read exists to prevent. ``peak_in_flight`` records
    how many bodies were being read at once across every instance, which is the memory claim."""

    in_flight = 0
    peak_in_flight = 0

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

    @classmethod
    def reset_read_tracking(cls) -> None:
        cls.in_flight = 0
        cls.peak_in_flight = 0

    async def body(self) -> bytes:
        self.body_reads += 1
        FakeResponse.in_flight += 1
        FakeResponse.peak_in_flight = max(FakeResponse.peak_in_flight, FakeResponse.in_flight)
        try:
            await asyncio.sleep(self._body_delay_s)
        except asyncio.CancelledError:
            self.body_read_cancelled = True
            raise
        finally:
            FakeResponse.in_flight -= 1
        if self._raises:
            raise PlaywrightError("target closed")
        return self._body


class FakePage:
    """A page that replays a fixed list of response events during ``goto``.

    ``content_hangs`` is the ogimet shape (P3-1): a DOM read that never answers because the page
    keeps navigating. ``content_raises`` is the driver's other answer to that page: in Playwright
    1.61 ``Frame.content()`` evaluates ``outerHTML`` once and, when the page navigates
    mid-evaluate, raises its plain ``Error("Unable to retrieve content because the page is
    navigating and changing the content.")`` rather than a ``TimeoutError``. ``goto_raises`` is the
    salvage shape: the navigation times out with the DOM already rendered.
    ``goto_runs_its_budget_out`` puts the clock on that shape: the navigation sleeps for its whole
    ``timeout`` before it raises or returns, as a real goto timeout does, so a test can measure
    what the transport spends AFTER the budget is gone. ``status`` is the main-frame response's
    HTTP status.

    ``land_on`` is where the main frame ends up, when that is not the URL ``goto`` was given: the
    server-side redirect hop Playwright follows without consulting any route handler. ``url``
    mirrors Playwright's own ``Page.url``: ``about:blank`` until a navigation commits, then the
    landing URL. A goto that raises leaves it at ``about:blank`` (the navigation never committed)
    unless the test set ``land_on``, which is the salvage shape where a timed-out navigation had
    already landed on a redirect target.

    Everything the transport does to the page and its context is recorded on the page, because
    the page is the one object a test holds: ``goto_calls``, ``settles`` (each
    ``wait_for_timeout``), ``content_reads`` (each ``page.content()``), ``context_kwargs`` (the
    real ``new_context`` options the transport passed) and ``unknown_context_kwargs`` (any keyword
    that is not one, which the real ``Browser.new_context`` would refuse with ``TypeError`` and
    which a render must therefore leave empty), ``route_patterns`` and ``route_handler`` (what
    ``context.route``
    registered), ``web_socket_patterns`` and ``web_socket_handler`` (what ``route_web_socket``
    registered), ``setup_events`` (the order the context was guarded and the page opened in),
    ``unroute_behavior``, and ``teardown`` (the close sequence the context and browser ran, so a
    test can assert the browser was still torn down after a failure mid-render).
    """

    def __init__(
        self,
        responses: list[FakeResponse] | None = None,
        *,
        html: str = DEFAULT_DOM,
        content_hangs: bool = False,
        content_raises: BaseException | None = None,
        goto_raises: BaseException | None = None,
        goto_runs_its_budget_out: bool = False,
        status: int = 200,
        land_on: str | None = None,
    ) -> None:
        self._responses = responses or []
        self._html = html
        self._content_hangs = content_hangs
        self._content_raises = content_raises
        self._goto_raises = goto_raises
        self._goto_runs_its_budget_out = goto_runs_its_budget_out
        self._status = status
        self._land_on = land_on
        self._handlers: list[Any] = []
        self.url = "about:blank"
        self.detached_handler_tasks: list[asyncio.Future[Any]] = []
        self.goto_calls: list[dict[str, Any]] = []
        self.settles: list[float] = []
        self.content_reads = 0
        self.context_kwargs: dict[str, Any] = {}
        self.unknown_context_kwargs: dict[str, Any] = {}
        self.route_patterns: list[str] = []
        self.route_handler: Any = None
        self.web_socket_patterns: list[str] = []
        self.web_socket_handler: Any = None
        self.setup_events: list[str] = []
        self.unroute_behavior: str | None = None
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
        if self._goto_runs_its_budget_out:
            await asyncio.sleep(timeout / 1000)
        if self._goto_raises is not None:
            if self._land_on is not None:
                self.url = self._land_on
            raise self._goto_raises
        self.url = self._land_on or url
        return SimpleNamespace(headers={"content-type": "text/html"}, status=self._status, url=self.url)

    async def wait_for_timeout(self, ms: float) -> None:
        self.settles.append(ms)

    async def content(self) -> str:
        self.content_reads += 1
        if self._content_hangs:
            await asyncio.Event().wait()
        # The real read is a round trip; yielding here lets the detached handler tasks run
        # between goto and the snapshot, which is where the un-joined harvest lost bodies.
        await asyncio.sleep(0)
        if self._content_raises is not None:
            raise self._content_raises
        return self._html


@dataclass
class Faults:
    """Where the fake browser misbehaves. ``new_page_error`` / ``new_context_error`` fail the
    render INSIDE the gates, which is the failure-boundary shape the warn latch was written for;
    ``close_error``, ``close_hangs`` and ``browser_close_hangs`` misbehave in teardown, where the
    real ``BrowserContext.close`` neither swallows a target-closed error nor bounds its wait."""

    new_page_error: BaseException | None = None
    new_context_error: BaseException | None = None
    route_web_socket_error: BaseException | None = None
    close_error: BaseException | None = None
    close_hangs: bool = False
    browser_close_hangs: bool = False


class FakeWebSocketRoute:
    """One routed WebSocket handshake, as Playwright hands it to a ``route_web_socket`` handler.

    A routed socket dials the server only if the handler calls ``connect_to_server()``, so the
    count of those calls is the whole claim a blocking handler makes. ``close`` takes ``code`` and
    ``reason`` KEYWORD-ONLY, exactly as Playwright 1.61's ``WebSocketRoute.close`` does, so a
    handler that reached for it with positional arguments fails here rather than only inside a
    Playwright-dispatched task in production, the one path the suite never runs.
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self.connect_calls = 0
        self.close_calls: list[tuple[int | None, str | None]] = []

    def connect_to_server(self) -> FakeWebSocketRoute:
        self.connect_calls += 1
        return self

    async def close(self, *, code: int | None = None, reason: str | None = None) -> None:
        self.close_calls.append((code, reason))


class FakeContext:
    def __init__(self, page: FakePage, faults: Faults) -> None:
        self._page = page
        self._faults = faults

    async def route(self, pattern: str, handler: Any) -> None:
        self._page.route_patterns.append(pattern)
        self._page.route_handler = handler
        self._page.setup_events.append("route")

    async def route_web_socket(self, pattern: str, handler: Any) -> None:
        self._page.setup_events.append("route_web_socket")
        if self._faults.route_web_socket_error is not None:
            raise self._faults.route_web_socket_error
        self._page.web_socket_patterns.append(pattern)
        self._page.web_socket_handler = handler

    async def new_page(self) -> FakePage:
        self._page.setup_events.append("new_page")
        if self._faults.new_page_error is not None:
            raise self._faults.new_page_error
        return self._page

    async def unroute_all(self, *, behavior: str | None = None) -> None:
        self._page.unroute_behavior = behavior
        self._page.teardown.append("unroute_all")

    async def close(self) -> None:
        self._page.teardown.append("context.close")
        if self._faults.close_hangs:
            await asyncio.Event().wait()
        if self._faults.close_error is not None:
            raise self._faults.close_error


class FakeBrowser:
    """``new_context`` names the options the transport passes, spelled as Playwright 1.61's
    ``Browser.new_context`` spells them, and parks anything else in ``unknown_context_kwargs``
    rather than accepting it silently: a ``**kwargs`` double kept every assertion green under a
    misspelled option (``service_worker="block"``) that the real browser would refuse and that
    would have quietly stopped blocking service workers."""

    def __init__(self, page: FakePage, faults: Faults) -> None:
        self._page = page
        self._faults = faults

    async def new_context(
        self,
        *,
        user_agent: str | None = None,
        extra_http_headers: dict[str, str] | None = None,
        service_workers: str | None = None,
        **unknown: Any,
    ) -> FakeContext:
        self._page.context_kwargs = {
            "user_agent": user_agent,
            "extra_http_headers": extra_http_headers,
            "service_workers": service_workers,
        }
        self._page.unknown_context_kwargs = unknown
        if self._faults.new_context_error is not None:
            raise self._faults.new_context_error
        return FakeContext(self._page, self._faults)

    async def close(self) -> None:
        self._page.teardown.append("browser.close")
        if self._faults.browser_close_hangs:
            await asyncio.Event().wait()


class FakeChromium:
    """The launcher. ``launch_delay_s`` is what the launch costs on the clock (0.3 s warm to
    several seconds cold on the real Chromium), spent AFTER the transport has recomputed its
    navigation budget. Every launch's ``args`` and ``headless`` are recorded."""

    def __init__(self, page: FakePage, faults: Faults | None = None, *, launch_delay_s: float = 0.0) -> None:
        self._page = page
        self._faults = faults or Faults()
        self._launch_delay_s = launch_delay_s
        self.launch_args: list[list[str]] = []
        self.headless: list[bool] = []

    async def launch(self, *, headless: bool, args: list[str] | None = None) -> FakeBrowser:
        self.launch_args.append(list(args or []))
        self.headless.append(headless)
        await asyncio.sleep(self._launch_delay_s)
        return FakeBrowser(self._page, self._faults)


class FakePlaywrightManager:
    """What ``async_playwright()`` returns: an async context manager exposing ``chromium``."""

    def __init__(self, chromium: FakeChromium) -> None:
        self.chromium = chromium

    async def __aenter__(self) -> FakePlaywrightManager:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        return None


def _async_return(value: Any):
    async def _call(*_args: Any, **_kwargs: Any) -> Any:
        await asyncio.sleep(0)
        return value

    return _call


async def _pin_the_requested_host(url: str) -> tuple[str, str] | None:
    """The default DNS pin: the host the transport asked for, at ``DEFAULT_PIN_IP``.

    Eligibility comes from the transport's own ``pinnable_url_host``, so a URL the real resolver
    would refuse to pin (a unicode or trailing-dot host, userinfo, a non-http(s) scheme) declines
    here too rather than being pinned anyway.
    """
    await asyncio.sleep(0)
    host = rendered_fetch.pinnable_url_host(url)
    return None if host is None else (host, DEFAULT_PIN_IP)


def install_fake_playwright(
    monkeypatch: pytest.MonkeyPatch,
    page: FakePage,
    *,
    faults: Faults | None = None,
    launch_delay_s: float = 0.0,
    pinned: tuple[str, str] | None | _PinTheRequestedHost = PIN_THE_REQUESTED_HOST,
    chromium: FakeChromium | None = None,
    manager_cls: type[FakePlaywrightManager] = FakePlaywrightManager,
) -> FakeChromium:
    """Wire the fakes in and return the launcher, whose ``launch_args`` say what was launched.

    ``faults`` says where the browser should misbehave (see :class:`Faults`). ``pinned`` is what
    the transport's DNS pin resolves to. The pinned host MUST equal the rendered URL's host or the
    transport refuses the DOM as an off-host landing, so the default,
    :data:`PIN_THE_REQUESTED_HOST`, derives the host from whatever URL the transport asks for and
    pins it to ``DEFAULT_PIN_IP``, which is what the real resolver does; with a fixed default, a
    render test on any other host silently took the refusal path while the mechanism it meant to
    pin never ran. Pass an explicit ``(host, vetted_ip)`` only to pin a host that deliberately
    does NOT match the URL, and ``None`` to make the host unpinnable, so the transport declines
    before any launch. A bespoke ``chromium`` (a subclass with its own ``launch``) or
    ``manager_cls`` (a subclass with its own ``__aenter__``) replaces the default of that one
    piece; everything else stays shared.
    """
    launcher = chromium or FakeChromium(page, faults, launch_delay_s=launch_delay_s)
    manager = manager_cls(launcher)
    monkeypatch.setitem(
        sys.modules,
        "playwright.async_api",
        SimpleNamespace(async_playwright=lambda: manager, Error=PlaywrightError),
    )
    resolve = _pin_the_requested_host if isinstance(pinned, _PinTheRequestedHost) else _async_return(pinned)
    monkeypatch.setattr(rendered_fetch, "resolve_pinned_host", resolve)
    return launcher
