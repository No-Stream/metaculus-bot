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

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from datetime import datetime
from html import escape as html_escape
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlparse

from metaculus_bot.constants import GOOGLE_API_KEY_ENV, RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV
from metaculus_bot.research import resolution_source
from metaculus_bot.research.rendered_fetch import RenderedPage


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
        # Per-request kwargs, parallel to `requested` — the per-hop ClientTimeout the
        # fetcher derives from the remaining wall budget is only observable here.
        self.get_kwargs: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _TrackingResponse:
        self.requested.append(url)
        self.get_kwargs.append(dict(kwargs))
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
# The bare Infogram div: the class and data-id an ``unreadable_data_embed_providers`` scan
# keys on, with none of the loader chrome. Kept as its own constant rather than sliced out of
# the full markup below, because the loader's visible anchor text ("NE - Osborn v. Ricketts",
# "Infogram") is extractable, and a page fixture measured against a character floor must not
# silently gain it.
_INFOGRAM_EMBED_DIV = '<div class="infogram-embed" data-id="_/vs9b6iAeARko8cuwH51x" data-type="interactive"></div>'

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


# Real-capture fixtures live beside the tests as files rather than inline strings: their
# VALUE is that nobody tidied the whitespace or the tag soup, which is exactly what an
# inline literal invites. Resolved from this file's own location so the path holds
# wherever the checkout sits (a developer-absolute path is green locally and red in CI).
_TEST_DATA_DIR = Path(__file__).parent / "data"

# CDC's cyclosporiasis stat block: a real table built out of `<div role="table">`.
CDC_ARIA_STAT_BLOCK_PATH = _TEST_DATA_DIR / "cdc_aria_stat_block.html"


def cdc_aria_stat_block_page() -> bytes:
    return CDC_ARIA_STAT_BLOCK_PATH.read_bytes()


def cp1252_aria_stat_block_page() -> bytes:
    """A windows-1252 page declaring its charset ONLY in a `<meta charset>`, with an ARIA table.

    The combination is what defeats the ARIA rewrite: `decode_text_body` honours a BOM and
    the HTTP header's charset and cannot see a meta declaration, so our decode replaces
    every accented byte with U+FFFD — while trafilatura reading the raw bytes finds the
    declaration and returns the accents. Mostly ASCII on purpose, so the undecodable ratio
    lands far below the shared refuse-the-body bound, which is the whole point: keying the
    rewrite's gate on that bound admitted this page and shipped the mojibake.
    """
    return (
        '<!doctype html><html><head><meta charset="windows-1252">'
        "<title>Bilan hebdomadaire</title></head><body>"
        "<article><h1>Résumé de l'éclosion au Québec</h1>"
        "<p>Résumé de l'éclosion au Québec, préparé par André Maître pour le Comité de "
        "surveillance. Les données ci-dessous couvrent la période du 1er mai au 24 août 2026 "
        "et proviennent des laboratoires provinciaux, qui les transmettent chaque semaine au "
        "registre national des maladies à déclaration obligatoire.</p>"
        '<div role="table"><div role="row"><div role="rowheader">Cas confirmés</div>'
        '<div role="cell">17 180</div></div>'
        '<div role="row"><div role="rowheader">Hospitalisations</div>'
        '<div role="cell">922</div></div>'
        '<div role="row"><div role="rowheader">Décès</div><div role="cell">2</div></div></div>'
        "<p>Les chiffres sont révisés à mesure que les enquêtes épidémiologiques progressent, "
        "de sorte que le total des cas confirmés peut augmenter après la publication.</p>"
        "</article></body></html>"
    ).encode("windows-1252")


def _meta_refresh_stub(target: str) -> bytes:
    """The cdc.gov shape: a ~300-byte 200 whose only content is the refresh tag.

    Deliberately under the chrome floor with nothing else in it, because that is what
    makes the direct read a `js_wall` and the hop the only way to the page.
    """
    return (
        "<!doctype html><html><head><title>Redirecting</title>"
        f'<meta http-equiv="refresh" content="0; url={target}">'
        "</head><body></body></html>"
    ).encode()


# --- Escalation-ladder shared builders (moved from the escalation test module when it was
# split by rung; imported by the per-rung modules plus the dispatch and route-caveats modules). ---
_URL = "https://tracker.example.com/senate"
_FEED_URL = "https://tracker.example.com/api/series"

# A body under RESOLUTION_SOURCE_JS_WALL_MIN_CHARS extracts to nothing: the direct route
# calls that `js_wall`, which is the rendered rung's primary trigger population.
_JS_SHELL = b'<!doctype html><html><body><div id="root"></div><script src="/app.js"></script></body></html>'


# Deliberately well ABOVE RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS (400) once extracted: the
# rendered DOM has to clear the same chrome floor a directly-fetched page does, so a fixture
# sized just under it would test the floor rather than the rung.
_RENDERED_PROSE = (
    "The Nebraska Senate polling average stands at 47.2 percent for Osborn and 45.8 percent "
    "for Ricketts as of September 2, 2026, across the eleven qualifying polls released since "
    "the primary. The average is recomputed whenever a new qualifying poll is published, and "
    "the trend over the last three weeks has been a narrowing of the gap by roughly a point. "
    "Polls are weighted by sample size, recency and the pollster's historic accuracy, and "
    "partisan-sponsored surveys are included at half weight with an adjustment for house "
    "effects. The eleven polls in the current average were fielded between August 4 and "
    "September 1 and carry sample sizes between 480 and 1,320 likely voters."
)


def _rendered(html: str, *, content_type: str = "text/html") -> RenderedPage:
    return RenderedPage(url=_URL, content_type=content_type, html=html)


def _rendered_document(inner: str) -> RenderedPage:
    """A rendered DOM shaped like a real document.

    Trafilatura discards a bare ``<article>`` with no head element ("discarding data"), so a
    fixture that skipped the head would test the extractor's give-up path rather than the
    rung. What a browser hands us is always a whole document, so the fixture is one too.
    """
    return _rendered(
        "<!doctype html><html><head><title>Nebraska Senate polling average</title></head><body>"
        f"<nav>Home | Senate</nav><article>{inner}</article><footer>&copy; 2026</footer></body></html>"
    )


def _fake_render(page: RenderedPage | None, calls: list[dict[str, object]]):
    async def _render(
        url: str,
        *,
        memo_scope: str,
        host_gate,
        goto_timeout_ms: int,
        deadline_monotonic_s: float | None = None,
        harvest_json: bool = False,
    ):
        calls.append({"url": url, "goto_timeout_ms": goto_timeout_ms, "harvest_json": harvest_json})
        del memo_scope, host_gate, deadline_monotonic_s
        # A real yield point, so the fake schedules like the browser rung it replaces and a
        # test that races two escalations sees the same interleaving the transport would give.
        await asyncio.sleep(0)
        return page

    return _render


def _snapshot_url(page_url: str, *, captured: datetime) -> str:
    """The final URL the archive redirects a snapshot request to (live-verified shape)."""
    return f"https://web.archive.org/web/{captured.strftime('%Y%m%d%H%M%S')}id_/{page_url}"


# --- Paid url_context rung scaffolding (shared by the four modules that arm the rung) ---
# Written five times before this lived here, which is how one copy came to re-declare `_URL` as
# its own literal. The rung is the only one that spends money, so what "armed" means has to be
# one thing every module agrees on.

_ROBOTS_URL = "https://tracker.example.com/robots.txt"
ROBOTS_ALLOW_ALL = b"User-agent: *\nAllow: /\n"

_PAID_READ_ANSWER = (
    "The Bureau of Labor Statistics work stoppages page reports 12 major work stoppages "
    "beginning in 2026 through August, per the table dated 2026-08-28."
)


def paid_reader(
    *,
    text: str | None = None,
    retrievals: int = 1,
    statuses: list[str] | None = None,
    raises: type[BaseException] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """A stand-in for ``run_url_context_read``, plus the kwargs the rung handed it.

    The kwargs are the point of recording them: ``timeout_ms`` and ``attempts`` are the only
    bounds that ever return a worker the outer ``wait_for`` cannot cancel.
    """
    calls: list[dict[str, Any]] = []

    def _read(url: str, ask: str, **kwargs: Any) -> tuple[str, int, list[str]]:
        calls.append({"url": url, "ask": ask, **kwargs})
        if raises is not None:
            raise raises("the client ceiling fired")
        return (
            _PAID_READ_ANSWER if text is None else text,
            retrievals,
            statuses or ["URL_RETRIEVAL_STATUS_SUCCESS"],
        )

    return _read, calls


def arm_paid_rung(monkeypatch: Any, reader: Any, *, budget_s: float | None = None) -> None:
    """Open the flag, the key and the reader, so the only closed gate is the one under test.

    Env NAMES come from ``constants.py`` rather than being spelled again here, so a rename cannot
    leave a test arming a flag nobody reads. The robots cache is deliberately NOT reset here: the
    package conftest's autouse fixture drops it around every test, and a second reset inside the
    arming helper is what used to hide that.
    """
    monkeypatch.setenv(RESOLUTION_SOURCE_URL_CONTEXT_ENABLED_ENV, "true")
    monkeypatch.setenv(GOOGLE_API_KEY_ENV, "key")
    monkeypatch.setattr(resolution_source, "run_url_context_read", reader)
    if budget_s is not None:
        monkeypatch.setattr(resolution_source.FetchContext, "rung_budget_s", lambda self: budget_s)


def refused_page_with_robots(*, robots: bytes = ROBOTS_ALLOW_ALL, extra: _Handlers | None = None) -> FakeSession:
    """The paid rung's standard fixture: the cited page refuses us and robots.txt answers.

    ``extra`` is merged LAST, so a test that needs the pre-check answered with something other
    than a plain-text policy (a PDF, a response that never returns) overrides ``_ROBOTS_URL``
    there rather than rebuilding the pair.
    """
    return FakeSession(
        {
            _URL: FakeResponse(403, body=b"denied", content_type="text/html"),
            _ROBOTS_URL: FakeResponse(200, body=robots, content_type="text/plain"),
            **(extra or {}),
        }
    )
