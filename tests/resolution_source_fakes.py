"""Fake aiohttp session and HTML page builders for the resolution-source tests.

Copied and extended from ``tests/market_retrieval_fakes.py``. Extensions versus the
prediction-market template:

* ``FakeResponse`` gets an async ``.read()`` returning bytes, a ``.content.iter_chunked``
  stream (what ``read_body_capped`` consumes), and a ``headers`` dict exposing
  Content-Type, which the resolution-source fetcher branches on.
* ``FakeSession`` tracks per-host in-flight counts, so the per-netloc-Semaphore
  serialization guarantee can be asserted.

Holds no pytest fixtures on purpose: the fixtures that wrap these builders live in
``tests/resolution_source/conftest.py``, so the autouse DNS stub covers every module in
that package. Same division as ``tests/ablation/``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from html import escape as html_escape
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlparse

from metaculus_bot.research import resolution_source


class _FakeContent:
    """Stub for `resp.content`: streams the body via `iter_chunked`.

    Delegates to the owning FakeResponse's `.read()` so tests that monkeypatch
    or override `read()` (slow-read serialization probe, UnreadableResponse's
    body-must-not-be-read assertion) keep working against the streaming path.
    """

    def __init__(self, resp: FakeResponse):
        self._resp = resp

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        body = await self._resp.read()
        for i in range(0, len(body), n):
            yield body[i : i + n]


class FakeResponse:
    def __init__(
        self,
        status: int,
        *,
        body: bytes = b"",
        content_type: str = "text/html; charset=utf-8",
        text: str | None = None,
        headers: Mapping[str, str] | None = None,
    ):
        self.status = status
        self._body = body
        # Extra headers (notably `Location` for redirect tests) override
        # Content-Type when the same key is provided.
        merged: dict[str, str] = {"Content-Type": content_type}
        if headers:
            merged.update(headers)
        self.headers = merged
        self._text = text if text is not None else body.decode("utf-8", errors="replace")
        self.content = _FakeContent(self)

    async def read(self) -> bytes:
        return self._body

    async def text(self) -> str:
        return self._text

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        return None


# A handler is either a ready-made FakeResponse, a callable returning one, or a
# callable/exception raiser. Using Any keeps the raise-on-get case (see the
# timeout test) simple.
_Handler = Any
_Handlers = Mapping[str, _Handler | list[_Handler]]


class FakeSession:
    """aiohttp.ClientSession replacement dispatching GET requests to per-URL handlers.

    Handlers keyed by URL prefix. Also tracks concurrent in-flight requests
    per netloc for the per-host-serialization assertion.
    """

    def __init__(self, handlers: _Handlers):
        self._handlers: dict[str, list[_Handler]] = {
            k: (v if isinstance(v, list) else [v]) for k, v in handlers.items()
        }
        self._call_counts: dict[str, int] = dict.fromkeys(handlers, 0)
        self.closed = False
        # host_inflight[host] = current concurrent count; peak observed value
        # captured in host_peak[host]. Provider must keep per-host peak == 1.
        self.host_inflight: dict[str, int] = {}
        self.host_peak: dict[str, int] = {}
        # Every URL requested, in order — lets tests pin which routes were
        # (and, critically for the Datawrapper hop, were NOT) fetched.
        self.requested: list[str] = []

    def get(self, url: str, **_kwargs: Any) -> _TrackingResponse:
        self.requested.append(url)
        for prefix, handler_list in self._handlers.items():
            if url.startswith(prefix):
                idx = min(self._call_counts[prefix], len(handler_list) - 1)
                self._call_counts[prefix] += 1
                handler = handler_list[idx]
                inner = handler(url) if callable(handler) and not isinstance(handler, FakeResponse) else handler
                host = urlparse(url).netloc
                return _TrackingResponse(inner, host, self)
        raise AssertionError(f"no handler for URL {url}")

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> FakeSession:
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()


class _TrackingResponse:
    """Wraps a FakeResponse; on __aenter__ increments the host's in-flight
    counter (updating the observed peak), decrements on __aexit__. Supports
    handlers that ARE exceptions (raised on __aenter__) — used to simulate
    aiohttp/asyncio errors."""

    def __init__(self, inner: Any, host: str, session: FakeSession):
        self._inner = inner
        self._host = host
        self._session = session

    async def __aenter__(self) -> Any:
        cur = self._session.host_inflight.get(self._host, 0) + 1
        self._session.host_inflight[self._host] = cur
        peak = self._session.host_peak.get(self._host, 0)
        if cur > peak:
            self._session.host_peak[self._host] = cur
        if isinstance(self._inner, BaseException):
            raise self._inner
        if hasattr(self._inner, "__aenter__"):
            return await self._inner.__aenter__()
        return self._inner

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            if hasattr(self._inner, "__aexit__") and not isinstance(self._inner, BaseException):
                await self._inner.__aexit__(exc_type, exc, tb)
        finally:
            cur = self._session.host_inflight.get(self._host, 0) - 1
            self._session.host_inflight[self._host] = max(cur, 0)


# Infogram's own published embed code (container div + async loader + credit
# block), the shape racetothewh.com/senate/26 used on qids 44554/44556.
_INFOGRAM_EMBED_MARKUP = (
    '<div class="infogram-embed" data-id="_/vs9b6iAeARko8cuwH51x" data-type="interactive" '
    'data-title="NE - Osborn v. Ricketts"></div>'
    '<script>!function(e,i,n,s){var t="InfogramEmbeds";}(document,0,"infogram-async",'
    '"https://e.infogram.com/js/dist/embed-loader-min.js");</script>'
    '<div style="padding:8px 0;text-align:center">'
    '<a href="https://infogram.com/vs9b6iAeARko8cuwH51x">NE - Osborn v. Ricketts</a><br>'
    '<a href="https://infogram.com" rel="nofollow">Infogram</a></div>'
)


def _embed_shell_page(embed_markup: str) -> bytes:
    """A page that is nothing but chrome around one embed.

    Trafilatura extracts 167 chars from this (heading + the one caption sentence):
    ABOVE the 100-char JS-wall floor, so before `no_resolving_content` existed it
    published as an unqualified `success` with no numbers in it.
    """
    return (
        "<!doctype html><html><head><title>NE Senate polling average</title></head><body>"
        "<nav>Home | Senate | House | Governors</nav>"
        "<article><h1>Nebraska Senate polling average</h1>"
        f"{embed_markup}"
        "<p>The chart above updates whenever a new qualifying poll is released.</p>"
        "</article><footer>&copy; 2026</footer></body></html>"
    ).encode()


def _prose_page(paragraph: str) -> bytes:
    """An article page whose extraction is exactly ``paragraph`` — used to sit either
    side of the chrome floor without depending on how trafilatura treats chrome."""
    return (
        "<!doctype html><html><head><title>Report</title></head><body>"
        "<nav>Home | About</nav>"
        f"<article><p>{paragraph}</p></article>"
        "<footer>&copy; 2026</footer></body></html>"
    ).encode()


def _escape_config(config: dict[str, Any]) -> str:
    """A chart config as it appears inside a double-quoted ``data-chart`` attribute."""
    return html_escape(json.dumps(config), quote=True)


# The real q43949 markup, shortened: Drupal's Charts module renders the config into a
# `data-chart` attribute on the container div, HTML-escaped. Categories and series name
# are verbatim from the 2026-05-24 Wayback snapshot; the annual data is its last three
# points (2024 / 2025 / 2026), 2026 = 1,240 being the pre-forecast live count.
_IOM_CHART_CONFIG: dict[str, Any] = {
    "chart": {"type": "column"},
    "title": {"text": ""},
    "xAxis": [{"categories": ["2024", "2025", "2026"]}],
    "series": [{"name": "Total Number of Dead and Missing", "data": [2573, 2185, 1240]}],
}

_IOM_PROSE = (
    "Migration across the Mediterranean: context in brief. Routes shift with weather, "
    "patrol posture and departure conditions, and the figures below are compiled from "
    "survivor testimony, coast guard reports and press accounts. " + "Context continues. " * 10
)


def _iom_shaped_page(prose: str = _IOM_PROSE, config: dict[str, Any] | None = None) -> bytes:
    """Prose plus one inline Highcharts config, the shape trafilatura drops entirely."""
    payload = _escape_config(_IOM_CHART_CONFIG if config is None else config)
    return (
        "<!doctype html><html><head><title>Missing Migrants</title></head><body>"
        f"<article><h1>Mediterranean</h1><p>{prose}</p>"
        f'<div data-drupal-selector-chart="dead-and-missing" class="charts-highchart chart" data-chart="{payload}">'
        "</div></article></body></html>"
    ).encode()


def _mid_band_chart_page() -> bytes:
    """An IOM-shaped chart page whose extraction lands strictly BETWEEN the JS-wall floor
    and the chrome floor.

    That band is where a chart rescue's counterfactual is `thin_page`, and it is where 7 of
    the 8 archived sub-400 chrome extractions sit. The bare-prose fixture
    (`prose="Mediterranean."`) extracts 43 chars, which is `js_wall` territory, and
    `_IOM_PROSE` extracts 425, only 25 chars above the chrome floor — so neither of them
    exercises the middle.
    """
    js_wall_floor = resolution_source.RESOLUTION_SOURCE_JS_WALL_MIN_CHARS
    chrome_floor = resolution_source.RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS
    # Trafilatura emits the h1 and then the paragraph twice, so the extraction is
    # 13 + 6n chars for an `"ab " * n` paragraph. The tests that use this measure the
    # result rather than trusting that arithmetic.
    return _iom_shaped_page(prose="ab " * ((js_wall_floor + chrome_floor) // 2 // 6))


def _mock_question(*, resolution_criteria: str = "", fine_print: str = "") -> MagicMock:
    q = MagicMock()
    q.resolution_criteria = resolution_criteria
    q.fine_print = fine_print
    q.id_of_question = 999
    q.question_text = "test question"
    q.page_url = "https://metaculus.com/q/999"
    return q
