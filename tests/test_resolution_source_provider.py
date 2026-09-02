"""Tests for the Tier-1 resolution-source fetcher (`resolution_source.py`).

Structured in three layers, mirroring the module:

- Pure helpers (URL extraction, skip predicates, formatter, JS-wall heuristic).
- Network layer: `_fetch_one` / `fetch_resolution_sources` under a FakeSession
  (patched via the module's `_get_session`).
- Factory / gating: `resolution_source_provider()` env-flag + benchmarking guard.

Real trafilatura runs on a fixed article-shaped HTML fixture so the success
path exercises extraction end-to-end.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket as _socket
import time
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime
from html import escape as html_escape
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlparse

import aiohttp
import pytest

from metaculus_bot.research import resolution_chart_data, resolution_source
from metaculus_bot.research.http_fetch import FilteringResolver
from metaculus_bot.research.provider_diagnostics import pop_provider_detail
from metaculus_bot.research.resolution_chart_data import CHART_DATA_LEAD, render_inline_chart_data
from metaculus_bot.research.resolution_source import (
    FetchResult,
    _fetch_one,
    _fetch_result_sources,
    _unreadable_embed_disclosure,
    extract_source_urls,
    fetch_resolution_sources,
    format_resolution_sections,
    is_fred_url,
    is_metaculus_self_ref,
    is_yahoo_ticker_url,
    looks_like_csv_rows,
    looks_like_js_wall,
    looks_like_page_chrome,
    resolution_source_provider,
    select_fetchable_urls,
    strip_html_tags,
    strip_markdown_escapes,
    vacuous_body_status,
)

# ---------------------------------------------------------------------------
# Fake aiohttp session (copied + extended from tests/market_retrieval_fakes.py).
# Extensions vs. the prediction-market template:
#   * FakeResponse gets an async .read() -> bytes, a `.content.iter_chunked`
#     stream (what `read_body_capped` consumes), and a `headers` dict
#     exposing Content-Type (the resolution-source fetcher branches on it).
#   * FakeSession tracks per-host in-flight counts so the per-netloc-Semaphore
#     serialization guarantee can be asserted.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_public_dns(monkeypatch):
    """Every test hostname in this file uses ``*.example.com`` — an RFC-2606
    reserved TLD with no real DNS. Without a stub, the SSRF guard's
    ``getaddrinfo`` call raises ``gaierror`` and every fetch becomes
    ``ssrf_blocked``. Return a public IP by default; tests that need private-
    IP behavior monkeypatch ``getaddrinfo`` again inside the test body (that
    later patch wins).
    """

    def _sync_ainfo(host, port, *args, **kwargs):
        del host, port, args, kwargs
        return [(0, 0, 0, "", ("8.8.8.8", 0))]

    monkeypatch.setattr(resolution_source.socket, "getaddrinfo", _sync_ainfo)


@pytest.fixture
def article_html() -> bytes:
    """A ~2 KB article-shaped HTML fixture. Trafilatura should extract the
    <article> body while dropping nav/footer chrome.
    """
    return (
        b"<!doctype html><html><head><title>Sample Report</title></head><body>"
        b"<nav>Home | About | Contact</nav>"
        b"<article><h1>Latest CPI Reading</h1>"
        b"<p>The Bureau of Labor Statistics reported a 3.2 percent annual "
        b"increase in the Consumer Price Index for the twelve months ending "
        b"in September 2026. Core CPI, which excludes food and energy, rose "
        b"3.4 percent over the same period. Housing costs contributed the "
        b"largest share of the monthly increase, while used-car prices fell "
        b"slightly.</p>"
        b"<p>Analysts had projected a 3.3 percent headline reading. The "
        b"lower-than-expected result was welcomed by markets and reinforced "
        b"expectations that the Federal Reserve would hold rates steady at "
        b"its next meeting.</p></article>"
        b"<footer>&copy; 2026 Example News</footer></body></html>"
    )


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


@pytest.fixture
def infogram_shell_html() -> bytes:
    return _embed_shell_page(_INFOGRAM_EMBED_MARKUP)


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


@pytest.fixture
def tracker_with_infogram_html() -> bytes:
    """The 44554 shape: real forecast prose (581 extracted chars) around the embed
    that holds the resolving polling average. The prose is worth keeping; what it
    does NOT contain is any polling number."""
    return (
        "<!doctype html><html><head><title>The 2026 Senate Forecast</title></head><body>"
        "<article><h1>The 2026 Senate Forecast</h1>"
        "<p>The forecast predicts the outcome of every Senate race in 2026 using a data-driven "
        "model that factors in the latest polling, historic trends, candidate quality, and "
        "fundraising. Every day, we simulate the election 50,000 times to get the best projection "
        "we can on how likely each party is to win the majority.</p>"
        f"{_INFOGRAM_EMBED_MARKUP}"
        "<p>Background: after a successful 2024 cycle, Republicans hold a 53-47 advantage, and "
        "Democrats need to flip four seats to take a 51-49 majority. Their best offensive "
        "opportunities are Maine and North Carolina, with Ohio and Alaska also competitive.</p>"
        "</article></body></html>"
    ).encode()


def _mock_question(*, resolution_criteria: str = "", fine_print: str = "") -> MagicMock:
    q = MagicMock()
    q.resolution_criteria = resolution_criteria
    q.fine_print = fine_print
    q.id_of_question = 999
    q.question_text = "test question"
    q.page_url = "https://metaculus.com/q/999"
    return q


# ---------------------------------------------------------------------------
# 1. Pure helpers: URL extraction, markdown unescape, dedup, skip predicates
# ---------------------------------------------------------------------------


class TestStripMarkdownEscapes:
    def test_underscore_and_dot(self):
        assert strip_markdown_escapes(r"https://example\.com/foo\_bar") == "https://example.com/foo_bar"

    def test_no_escapes_is_identity(self):
        assert strip_markdown_escapes("https://a.com/x") == "https://a.com/x"

    def test_escaped_ampersand_hash_paren(self):
        # The regex covers _ & . - # ( ). Verify the covered set:
        assert strip_markdown_escapes(r"a\#b\(c\)d\-e") == "a#b(c)d-e"


class TestExtractSourceUrls:
    def test_markdown_link_extraction(self):
        text = "See [BLS report](https://www.bls.gov/cpi/) for details."
        assert extract_source_urls(text) == ["https://www.bls.gov/cpi/"]

    def test_bare_url_extraction(self):
        text = "Source: https://fred.stlouisfed.org/series/DGS10 as reported."
        assert extract_source_urls(text) == ["https://fred.stlouisfed.org/series/DGS10"]

    def test_trailing_punctuation_stripped(self):
        text = "See https://example.com/foo, and also https://example.com/bar."
        urls = extract_source_urls(text)
        assert urls == ["https://example.com/foo", "https://example.com/bar"]

    def test_backslash_escapes_unescaped(self):
        text = r"See [report](https://example\.com/foo\_bar)"
        assert extract_source_urls(text) == ["https://example.com/foo_bar"]

    def test_dedup_preserves_order(self):
        text = "First https://a.example.com/x then [link](https://b.example.com/y) again https://a.example.com/x."
        urls = extract_source_urls(text)
        assert urls == ["https://a.example.com/x", "https://b.example.com/y"]

    def test_http_and_https_only(self):
        text = "ftp://old.example.com/x and gopher://x.com and https://ok.com/z"
        assert extract_source_urls(text) == ["https://ok.com/z"]

    def test_dedup_collapses_bare_host_and_trailing_slash(self):
        # Real questions cite both root-page forms (2026-07-09 smoke test, Q41581:
        # childmortality.org vs childmortality.org/) — one fetch slot, not two.
        text = "See https://x.org and also https://x.org/ for data."
        assert extract_source_urls(text) == ["https://x.org"]

    def test_dedup_ignores_fragment(self):
        # Fragments are never sent over HTTP — URLs differing only by fragment
        # are the same fetch and must not burn two fetch slots. First-seen wins.
        text = "See https://x.org/page#section-a and https://x.org/page#section-b for data."
        assert extract_source_urls(text) == ["https://x.org/page#section-a"]

    def test_no_cap_in_extraction(self, monkeypatch):
        # The cap moved to `select_fetchable_urls` (F2 fix). `extract_source_urls`
        # now returns the FULL deduped list so the skip-filter can drop
        # self-refs/FRED/Yahoo before the cap fires — a run of leading self-refs
        # was starving real sources out of the fetch budget.
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_MAX_URLS", 3)
        text = " ".join(f"https://example{i}.com/x" for i in range(10))
        urls = resolution_source.extract_source_urls(text)
        assert len(urls) == 10
        assert urls[0] == "https://example0.com/x"
        assert urls[-1] == "https://example9.com/x"


class TestSkipPredicates:
    def test_is_metaculus_self_ref(self):
        assert is_metaculus_self_ref("https://metaculus.com/q/12345") is True
        assert is_metaculus_self_ref("https://www.metaculus.com/questions/12345") is True
        assert is_metaculus_self_ref("https://example.com/metaculus-fan") is False

    def test_is_metaculus_self_ref_port_and_userinfo_do_not_bypass(self):
        # .hostname strips port + userinfo; .netloc would have kept them and let
        # these slip past the exact-host / suffix checks.
        assert is_metaculus_self_ref("https://www.metaculus.com:443/questions/12345") is True
        assert is_metaculus_self_ref("https://metaculus.com:8080/q/1") is True
        assert is_metaculus_self_ref("https://user@metaculus.com/q/1") is True
        assert is_metaculus_self_ref("https://sub.metaculus.com/page") is True
        # A host that merely contains the string is not a self-ref.
        assert is_metaculus_self_ref("https://notmetaculus.com/x") is False

    def test_is_fred_url(self):
        assert is_fred_url("https://fred.stlouisfed.org/series/DGS10") is True
        assert is_fred_url("https://stlouisfed.org/other") is False
        # Port must not bypass (.hostname fix).
        assert is_fred_url("https://fred.stlouisfed.org:443/series/DGS10") is True

    def test_is_yahoo_ticker_url(self):
        assert is_yahoo_ticker_url("https://finance.yahoo.com/quote/AAPL") is True
        assert is_yahoo_ticker_url("https://finance.yahoo.com/quote/BTC-USD/history") is True
        # Generic Yahoo articles are still fetchable — only /quote/ URLs are yfinance-served.
        assert is_yahoo_ticker_url("https://finance.yahoo.com/news/some-article") is False
        # Port must not bypass (.hostname fix).
        assert is_yahoo_ticker_url("https://finance.yahoo.com:443/quote/AAPL") is True


class TestSelectFetchableUrls:
    def test_none_fields_are_safe(self):
        assert select_fetchable_urls(None, None) == []
        assert select_fetchable_urls("", "") == []

    def test_drops_self_ref_fred_yahoo_ticker(self):
        criteria = (
            "See https://metaculus.com/q/1 and https://fred.stlouisfed.org/series/DGS10 "
            "and https://finance.yahoo.com/quote/AAPL — but also https://www.bls.gov/cpi/."
        )
        urls = select_fetchable_urls(criteria, "")
        assert urls == ["https://www.bls.gov/cpi/"]

    def test_combines_criteria_and_fine_print(self):
        urls = select_fetchable_urls(
            "See https://a.example.com/x",
            "Details at https://b.example.com/y",
        )
        assert set(urls) == {"https://a.example.com/x", "https://b.example.com/y"}

    def test_cap_applied_after_skip_filter(self, monkeypatch):
        # F2 regression: cap must apply AFTER dropping self-refs/FRED/Yahoo, or
        # a run of leading self-refs starves the real source out of the fetch
        # budget. With MAX_URLS=1 and 5 leading self-refs, the one real source
        # must survive.
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_MAX_URLS", 1)
        criteria = (
            "See https://metaculus.com/q/1 and https://metaculus.com/q/2 "
            "and https://metaculus.com/q/3 and https://metaculus.com/q/4 "
            "and https://metaculus.com/q/5 — resolution source: https://www.bls.gov/cpi/."
        )
        urls = select_fetchable_urls(criteria, "")
        assert urls == ["https://www.bls.gov/cpi/"]

    def test_cap_bounds_result_length(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_MAX_URLS", 3)
        criteria = " ".join(f"https://example{i}.com/x" for i in range(10))
        urls = select_fetchable_urls(criteria, "")
        assert len(urls) == 3
        assert urls == [
            "https://example0.com/x",
            "https://example1.com/x",
            "https://example2.com/x",
        ]


class TestLooksLikeJsWall:
    def test_short_text_flagged(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_JS_WALL_MIN_CHARS", 100)
        assert resolution_source.looks_like_js_wall("only a few chars") is True

    def test_long_text_not_flagged(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_JS_WALL_MIN_CHARS", 20)
        assert resolution_source.looks_like_js_wall("x" * 30) is False

    def test_whitespace_only_flagged(self):
        assert looks_like_js_wall("       \n\n   ") is True


class TestStripHtmlTags:
    """Markup stripping for the RAW-body branches. Two properties matter: real tags go, and
    inequality signs in a data cell are NOT tags. The naive `</?[A-Za-z][^>]*>` form fails the
    second — it eats `x <a and y > b` down to `x  b`."""

    _VUUVZ_ROW = (
        '"8/16 - 8/17, 2026@@24335",'
        "<a href='https://emersoncollegepolling.com/august-2026-national-poll/'"
        "style='color:#000000; text-decoration: underline;'target='_blank' rel='nofollow noopener'>"
        "Emerson College</a>,"
        '"1,000 LV@@1000",1.108478,36.4,49.2,-12.8'
    )

    def test_a_live_poll_table_row_keeps_the_pollster_and_loses_the_markup(self):
        """The measured shape from the live VUUVz dataset (2026-08-26 receipts): a styled anchor
        per pollster row, 69% of that CSV's 33k chars being tag markup. The pollster name IS the
        content, so it stays and the tags go — 248 chars down to 84."""
        out = strip_html_tags(self._VUUVZ_ROW)

        assert "Emerson College" in out
        assert "<a " not in out
        assert "</a>" not in out
        assert "style=" not in out
        assert len(out) < len(self._VUUVZ_ROW) / 2.5

    @pytest.mark.parametrize(
        "cell",
        ["a < 5, b > 3", "x <a and y > b", "1 < 2 and 3 > 2", "temp < -40 or > 40"],
    )
    def test_inequalities_in_a_data_cell_are_untouched(self, cell: str):
        """`<a and y >` is why the tag NAME is an allow-list and an attribute region must contain
        an `=`: without both halves this eats real numeric data out of a dataset."""
        assert strip_html_tags(cell) == cell

    def test_a_body_with_no_angle_brackets_is_byte_identical(self):
        """The numeric tracker CSVs (1mU3g / kSCt4) contain zero `<` characters, so the strip must
        be a provable no-op there rather than merely a small one."""
        csv = "modeldate,approve,disapprove\n8/25/2026,36.41889,55.62032\n"
        assert strip_html_tags(csv) == csv

    def test_a_bare_link_cell_keeps_its_href_as_the_content(self):
        """An anchor with empty inner text carries its content in the href, so dropping the tag
        outright would delete the cell."""
        assert strip_html_tags("source,<a href='https://x.test/report'></a>") == "source,https://x.test/report"

    def test_an_unlisted_tag_name_is_left_alone(self):
        """The allow-list is closed: `<body>`/`<script>` never appear in a CSV cell, and matching
        every `<word>` is what makes the inequality cases above fail."""
        assert strip_html_tags("<body class='x'>hi</body>") == "<body class='x'>hi</body>"

    def test_a_pathological_no_close_tag_body_strips_in_linear_time(self):
        """The tag-body alternation must reach its first `=` exactly one way. With `[^<>]*` on
        both sides of the `=`, a body holding one `<b ` lookalike followed by an angle-bracket-free
        run of URL cells (query-string `=` signs, no closing `>`) backtracks quadratically: 3.4s at
        200 KiB measured, ~35 minutes at the 5 MiB response cap — synchronously on the event loop,
        wedging the sibling fetches past every wall timeout. The linear form is sub-millisecond
        here, so the 1s bound has three orders of magnitude of slack on either side."""
        body = "x <b " + ("url=https://example.test/p?q=1&r=2, " * 6000)
        start = time.perf_counter()
        out = strip_html_tags(body)
        elapsed = time.perf_counter() - start
        assert out == body, "`<b ` with no closing `>` names no tag — the body must be untouched"
        assert elapsed < 1.0, f"quadratic backtracking regression: {elapsed:.2f}s on a ~220 KiB body"


class TestLooksLikeCsvRows:
    """The precondition for the Tier-2 lead's `Dataset published <ts>` liveness claim."""

    def test_a_header_plus_a_row_is_a_dataset(self):
        assert looks_like_csv_rows("date,value\n2026-08-01,0.42\n") is True

    def test_a_header_alone_is_not(self):
        assert looks_like_csv_rows("date,value\n") is False

    def test_a_delimiterless_header_is_not(self):
        assert looks_like_csv_rows("Not Found\nThe requested chart is unavailable\n") is False

    def test_an_html_error_page_is_not_even_when_it_carries_commas(self):
        """A soft-404 page passes a bare delimiter test easily, which is why markup is rejected
        outright on the first non-blank line."""
        body = "<!DOCTYPE html>\n<html><head><title>404, not found</title></head>\n<body>gone</body>\n"
        assert looks_like_csv_rows(body) is False

    def test_tab_and_semicolon_delimiters_count(self):
        assert looks_like_csv_rows("date\tvalue\n2026-08-01\t0.42\n") is True
        assert looks_like_csv_rows("date;value\n2026-08-01;0.42\n") is True


class TestVacuousBodyStatus:
    """The one place "does this 200 body carry information?" is decided."""

    def test_content_returns_none(self):
        assert vacuous_body_status("date,value\n2026-08-01,0.42\n", 0.0, require_csv_rows=True) is None

    @pytest.mark.parametrize("body", ["", "   ", "\n\n\t"])
    def test_an_empty_or_whitespace_body_is_empty_body(self, body: str):
        assert vacuous_body_status(body, 0.0, require_csv_rows=False) == "empty_body"

    def test_an_undecodable_body_is_unsupported_type(self):
        assert vacuous_body_status("d\x00a\x00t\x00e\x00", 0.5, require_csv_rows=False) == "unsupported_type"

    def test_the_row_shape_requirement_is_dataset_only(self):
        """A cited JSON or plain-text page has no row shape to satisfy; only a dataset claiming to
        be a live series does."""
        assert vacuous_body_status('{"cve": "x"}', 0.0, require_csv_rows=False) is None
        assert vacuous_body_status('{"cve": "x"}', 0.0, require_csv_rows=True) == "unsupported_type"


class TestFetchResultInvariant:
    def test_a_success_with_blank_text_cannot_be_constructed(self):
        """The invariant the field comment always stated and nothing enforced. An empty 200 body
        shipped as `success` rendered an empty section under the "primary grading evidence"
        caveat, suppressed the all-failed notice for its siblings, and reported `ok` to provider
        diagnostics — so a future edit that reintroduces it should crash, not publish a hole."""
        with pytest.raises(ValueError, match="blank text"):
            FetchResult(url="https://x.test/a", status="success", text="   ", http_status=200, content_type="text/csv")

    def test_a_failure_with_blank_text_is_the_normal_case(self):
        assert FetchResult(url="https://x.test/a", status="empty_body", text="", http_status=200, content_type=None)


class TestFormatResolutionSections:
    def test_empty_results_returns_empty_string(self):
        assert format_resolution_sections([], datetime(2026, 7, 9, tzinfo=UTC)) == ""

    def test_all_failed_renders_unreachable_notice(self):
        # URLs were attempted but every fetch failed — surface it instead of
        # staying silent (the qid 44211 miss: the resolving CBP page 403'd and
        # nobody in the pipeline learned it was unreachable).
        results = [
            FetchResult(
                url="https://a.com/x",
                status="blocked",
                text="",
                http_status=403,
                content_type=None,
            ),
            FetchResult(
                url="https://b.com/y",
                status="js_wall",
                text="",
                http_status=200,
                content_type="text/html",
            ),
        ]
        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))
        assert out  # no longer empty
        assert "2 resolution source(s) yielded no usable content" in out
        assert "a.com: blocked" in out
        assert "b.com: js_wall" in out
        assert "nothing from the cited resolving page(s) is in this bundle; weight other evidence accordingly" in out
        # Body only — the orchestrator prepends the "## Resolution Source Snapshot" header.
        assert "## Resolution Source Snapshot" not in out

    def test_an_empty_body_result_no_longer_suppresses_the_all_failed_notice(self):
        """The render half of the empty-200 defect. While an empty body counted as `success`, ONE
        such result put the section on the success path: it rendered an empty `### <url>` block
        under the primary-grading-evidence caveat, and the all-failed "yielded no usable content" notice
        — the whole point of which is to tell the forecaster to weight other evidence — was
        withheld for the sibling URLs that genuinely failed."""
        results = [
            FetchResult(
                url="https://empty.example.com/x",
                status="empty_body",
                text="",
                http_status=200,
                content_type="application/json",
            ),
            FetchResult(url="https://bad.com/y", status="blocked", text="", http_status=403, content_type=None),
        ]

        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))

        assert "2 resolution source(s) yielded no usable content" in out
        assert "empty.example.com: empty_body" in out
        assert "nothing from the cited resolving page(s) is in this bundle" in out
        assert "primary grading evidence" not in out
        assert "### https://empty.example.com/x" not in out

    def test_partial_success_appends_failure_note(self):
        # Some sources fetched, some failed: keep the success content and append
        # a terse note naming the unreachable ones.
        results = [
            FetchResult(
                url="https://ok.com/data",
                status="success",
                text="the reading is 3.2%",
                http_status=200,
                content_type="text/html",
            ),
            FetchResult(
                url="https://bad.com/y",
                status="blocked",
                text="",
                http_status=403,
                content_type=None,
            ),
        ]
        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))
        # Success content still rendered.
        assert "### https://ok.com/data" in out
        assert "the reading is 3.2%" in out
        assert "primary grading evidence" in out
        # Terse note about the failed source appended.
        assert "bad.com: blocked" in out
        assert "other cited resolution source(s) yielded no usable content" in out
        # The success path must not carry the all-failed sentence.
        assert "nothing from the cited resolving page(s) is in this bundle" not in out

    def test_success_rendering_includes_url_and_date(self):
        results = [
            FetchResult(
                url="https://www.bls.gov/cpi/",
                status="success",
                text="CPI rose 3.2% over the past 12 months.",
                http_status=200,
                content_type="text/html",
            ),
        ]
        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))
        assert "primary grading evidence" in out
        assert "### https://www.bls.gov/cpi/" in out
        assert "fetched 2026-07-09" in out
        assert "CPI rose 3.2% over the past 12 months." in out

    def test_total_budget_trims_later_sections(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_TOTAL_MAX_CHARS", 400)
        results = [
            FetchResult(
                url=f"https://example.com/{i}",
                status="success",
                text="A" * 300,
                http_status=200,
                content_type="text/html",
            )
            for i in range(4)
        ]
        out = resolution_source.format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))
        # First section fits; later ones must be trimmed or dropped.
        assert "https://example.com/0" in out
        # We should NOT see all four full 300-char blocks packed together.
        assert out.count("A" * 300) <= 2

    def test_a_budget_trim_leaves_a_visible_truncation_marker(self, monkeypatch):
        """The aggregate trim goes through the marker-emitting truncator, not a bare slice.

        A bare slice cut mid-sentence and could eat the per-URL ``[truncated at N chars ...]``
        marker the fetch already appended at the end — so an already-truncated page rendered
        as complete. Reachable on prod constants (5 x 6000 per-URL against an 18000 total).
        """
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_TOTAL_MAX_CHARS", 400)
        results = [
            FetchResult(
                url="https://example.com/long",
                status="success",
                text="B" * 5000 + "\n[truncated at 5000 chars — full source at https://example.com/long]",
                http_status=200,
                content_type="text/html",
            )
        ]

        out = resolution_source.format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))

        # The section is cut, and the cut says so rather than ending mid-body.
        assert "[truncated at 400 chars — full source at https://example.com/long]" in out
        assert "B" * 5000 not in out

    def test_dropped_sections_note_appended(self, monkeypatch):
        # Tighten TOTAL cap so at least one section is dropped entirely: cap=300,
        # 4 sources of 300 chars each — first section fills the budget, 3 dropped.
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_TOTAL_MAX_CHARS", 300)
        results = [
            FetchResult(
                url=f"https://example.com/{i}",
                status="success",
                text="A" * 300,
                http_status=200,
                content_type="text/html",
            )
            for i in range(4)
        ]
        out = resolution_source.format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))
        # The dropped-section note must appear, naming the dropped count.
        assert "additional source(s) omitted — section budget" in out
        assert "3 additional" in out

    def test_no_drop_note_when_all_sections_fit(self):
        # All sections fit -> no trailing "omitted" note.
        results = [
            FetchResult(
                url="https://x.example.com/a",
                status="success",
                text="short body",
                http_status=200,
                content_type="text/html",
            ),
            FetchResult(
                url="https://x.example.com/b",
                status="success",
                text="another short body",
                http_status=200,
                content_type="text/html",
            ),
        ]
        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))
        assert "omitted" not in out


# ---------------------------------------------------------------------------
# 2. Network layer: _fetch_one branches + fetch_resolution_sources per-host serialization
# ---------------------------------------------------------------------------


class TestFetchOne:
    async def test_success_html_extracts_and_truncates(self, article_html, monkeypatch):
        # Tighten the per-URL cap so we can also verify truncation lands.
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_PER_URL_MAX_CHARS", 200)
        session = FakeSession(
            {"https://news.example.com/report": FakeResponse(200, body=article_html, content_type="text/html")}
        )
        result = await _fetch_one(session, "https://news.example.com/report", {})
        assert result.status == "success"
        assert result.http_status == 200
        # Real trafilatura ran on the article — a known substring survives.
        assert "Bureau of Labor Statistics" in result.text
        # Per-URL truncation was applied.
        assert len(result.text) <= 200

    async def test_html_truncation_appends_marker(self, article_html, monkeypatch):
        # Live run analysis (2026-07-10): the per-URL cap truncates mid-sentence
        # with no marker so forecasters can't tell the snapshot is partial.
        # When truncation fires, a marker line naming the cap and URL must
        # appear, and total text length must remain bounded by the cap.
        cap = 200
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_PER_URL_MAX_CHARS", cap)
        session = FakeSession(
            {"https://news.example.com/report": FakeResponse(200, body=article_html, content_type="text/html")}
        )
        result = await _fetch_one(session, "https://news.example.com/report", {})
        assert result.status == "success"
        assert f"[truncated at {cap} chars — full source at https://news.example.com/report]" in result.text
        assert len(result.text) <= cap

    async def test_no_truncation_marker_when_fits_under_cap(self, article_html, monkeypatch):
        # Extraction fits entirely under the cap -> NO marker appended.
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_PER_URL_MAX_CHARS", 100_000)
        session = FakeSession(
            {"https://news.example.com/report": FakeResponse(200, body=article_html, content_type="text/html")}
        )
        result = await _fetch_one(session, "https://news.example.com/report", {})
        assert result.status == "success"
        assert "truncated at" not in result.text

    async def test_403_maps_to_blocked(self):
        session = FakeSession({"https://blocked.example.com/x": FakeResponse(403, body=b"nope")})
        result = await _fetch_one(session, "https://blocked.example.com/x", {})
        assert result.status == "blocked"
        assert result.http_status == 403
        assert result.text == ""

    async def test_404_maps_to_not_found(self):
        session = FakeSession({"https://gone.example.com/x": FakeResponse(404)})
        result = await _fetch_one(session, "https://gone.example.com/x", {})
        assert result.status == "not_found"
        assert result.http_status == 404

    async def test_js_wall_short_html_flagged(self):
        # 200 OK but the extracted text is short: js_wall.
        tiny = b"<!doctype html><html><body><div id='root'></div></body></html>"
        session = FakeSession({"https://spa.example.com/x": FakeResponse(200, body=tiny, content_type="text/html")})
        result = await _fetch_one(session, "https://spa.example.com/x", {})
        assert result.status == "js_wall"
        assert result.http_status == 200
        assert result.text == ""

    async def test_oversize_body_is_dropped(self, monkeypatch):
        # Force a 100-byte cap; the body exceeds it.
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_MAX_RESPONSE_BYTES", 100)
        oversized = b"<html><body>" + b"A" * 500 + b"</body></html>"
        session = FakeSession(
            {"https://big.example.com/x": FakeResponse(200, body=oversized, content_type="text/html")}
        )
        result = await _fetch_one(session, "https://big.example.com/x", {})
        # read_body_capped returns None -> we mark as error (no readable body).
        assert result.status == "error"
        assert result.text == ""

    async def test_timeout_maps_to_error(self):
        session = FakeSession({"https://slow.example.com/x": TimeoutError()})
        result = await _fetch_one(session, "https://slow.example.com/x", {})
        assert result.status == "error"
        assert result.http_status is None

    async def test_client_error_maps_to_error(self):
        session = FakeSession({"https://broken.example.com/x": aiohttp.ClientError("boom")})
        result = await _fetch_one(session, "https://broken.example.com/x", {})
        assert result.status == "error"
        assert result.http_status is None

    async def test_json_content_type_returns_raw_truncated(self, monkeypatch):
        cap = 200
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_PER_URL_MAX_CHARS", cap)
        payload = b'{"vulnerabilities":[{"cveID":"CVE-2026-0001","description":"' + b"x" * 500 + b'"}]}'
        session = FakeSession(
            {"https://json.example.com/kev": FakeResponse(200, body=payload, content_type="application/json")}
        )
        result = await _fetch_one(session, "https://json.example.com/kev", {})
        assert result.status == "success"
        assert result.content_type is not None
        assert "json" in result.content_type
        assert result.text.startswith('{"vulnerabilities')
        # Truncated -> marker appears, total bounded by cap.
        assert f"[truncated at {cap} chars — full source at https://json.example.com/kev]" in result.text
        assert len(result.text) <= cap

    @pytest.mark.parametrize(
        ("body", "content_type"),
        [
            (b"", "application/json"),
            (b"   \n\t\n ", "application/json"),
            (b"", "text/csv"),
            (b"\n\n", "text/plain"),
        ],
    )
    async def test_a_200_with_an_empty_body_is_not_a_success(self, body: bytes, content_type: str):
        """`read_body_capped` returns `b""` for an empty body and the only guard was `is None`, so
        an empty 200 shipped as `success` with `text=""`. Three things followed: an empty `###
        <url>` section rendered under the "primary grading evidence" caveat, that one result
        suppressed the all-failed "yielded no usable content" notice for every OTHER failed URL, and
        provider diagnostics reported `ok` — indistinguishable from a real fetch."""
        session = FakeSession({"https://empty.example.com/x": FakeResponse(200, body=body, content_type=content_type)})

        result = await _fetch_one(session, "https://empty.example.com/x", {})

        assert result.status == "empty_body"
        assert result.text == ""
        assert result.http_status == 200

    async def test_a_declared_charset_body_decodes_instead_of_mojibaking(self):
        """`charset=` was parsed for ROUTING and then ignored for decoding, so a Windows-1252 CSV
        rendered as grading evidence with replacement characters where its punctuation had been."""
        body = "Pollster,Approve\nO’Brien Research,44\n".encode("windows-1252")  # noqa: RUF001  # cp1252 fixture
        session = FakeSession(
            {
                "https://poll.example.com/d.csv": FakeResponse(
                    200, body=body, content_type="text/csv; charset=windows-1252"
                )
            }
        )

        result = await _fetch_one(session, "https://poll.example.com/d.csv", {})

        assert result.status == "success"
        assert "O’Brien Research" in result.text  # noqa: RUF001  # cp1252 fixture
        assert "�" not in result.text

    async def test_an_undecodable_body_is_refused_rather_than_rendered_as_mojibake(self):
        """BOM-less UTF-16 — the shape a replacement-char count alone cannot see, since every
        second byte decodes as a valid NUL. `0<?>.<?>4<?>2<?>` type-checks as text and used to
        reach the forecaster under the primary-grading-evidence caveat."""
        body = "date,value\n2026-08-01,0.42\n".encode("utf-16-le")
        session = FakeSession({"https://odd.example.com/d.csv": FakeResponse(200, body=body, content_type="text/csv")})

        result = await _fetch_one(session, "https://odd.example.com/d.csv", {})

        assert result.status == "unsupported_type"
        assert result.text == ""

    async def test_html_markup_inside_a_csv_cell_is_stripped_on_the_text_branch(self):
        """The Tier-1 half of the Datawrapper budget fix: the same class of input (a delimited
        table whose cells carry styled anchors) reaches this branch whenever a source serves its
        data as CSV directly, and the per-URL char budget should buy rows rather than markup."""
        body = b"Dates,Pollster,Approve\n8/16,<a href='https://x.test/p' style='color:#000'>Emerson College</a>,36.4\n"
        session = FakeSession(
            {"https://poll.example.com/rows.csv": FakeResponse(200, body=body, content_type="text/csv")}
        )

        result = await _fetch_one(session, "https://poll.example.com/rows.csv", {})

        assert result.status == "success"
        assert "Emerson College" in result.text
        assert "<a " not in result.text
        assert "style=" not in result.text

    async def test_json_bodies_keep_their_angle_brackets(self):
        """A JSON body's angle brackets sit inside string values that ARE the data, so the strip is
        confined to the text branches."""
        body = b'{"note": "value <a and b > c", "n": 1}'
        session = FakeSession(
            {"https://api.example.com/v": FakeResponse(200, body=body, content_type="application/json")}
        )

        result = await _fetch_one(session, "https://api.example.com/v", {})

        assert result.status == "success"
        assert result.text == body.decode("utf-8")

    async def test_pdf_content_type_is_unsupported(self):
        # PDF: body is NEVER read (per the plan). A read() that raises verifies that.
        class UnreadableResponse(FakeResponse):
            async def read(self) -> bytes:
                raise AssertionError("body must not be read for unsupported types")

        session = FakeSession(
            {"https://pdf.example.com/doc": UnreadableResponse(200, body=b"", content_type="application/pdf")}
        )
        result = await _fetch_one(session, "https://pdf.example.com/doc", {})
        assert result.status == "unsupported_type"
        assert result.text == ""

    async def test_missing_content_type_is_unsupported_type(self):
        # INTENDED limitation (F13): a 200 OK served without a Content-Type
        # header matches no routing prefix and is classified unsupported_type,
        # body unread — real resolution sources always send Content-Type, and
        # we deliberately don't content-sniff unknown bodies.
        resp = FakeResponse(200, body=b"<html><body>hello there</body></html>")
        del resp.headers["Content-Type"]
        session = FakeSession({"https://noct.example.com/x": resp})
        result = await _fetch_one(session, "https://noct.example.com/x", {})
        assert result.status == "unsupported_type"
        assert result.content_type is None
        assert result.text == ""


class TestEmbedShapedPages:
    """qids 44554/44556: a tracker page returned HTTP 200, extracted forecast background,
    and reported `success` while the resolving Nebraska polling average sat in two Infogram
    iframes trafilatura drops. The section published under "primary grading evidence" with
    zero polling numbers in it, byte-identical across three questions, and nothing anywhere
    said so. Two outcomes now, split by how much page text came back."""

    async def test_an_embed_shell_200_is_no_resolving_content(self, infogram_shell_html):
        session = FakeSession({"https://tracker.example.com/senate": FakeResponse(200, body=infogram_shell_html)})

        result = await _fetch_one(session, "https://tracker.example.com/senate", {})

        assert result.status == "no_resolving_content"
        assert result.status_reason == "embed_shell"
        assert result.http_status == 200  # the fetch itself succeeded; the CONTENT did not arrive
        assert result.text == ""
        assert result.unreadable_embeds == ["infogram"]

    async def test_the_same_thin_page_without_an_embed_is_withheld_as_a_thin_page(self):
        """The FLOOR is the discriminator; the embed only says where the content went.

        This test asserted the opposite when the verdict shipped — identical chrome with
        the embed markup swapped for an inert div extracted the same 167 chars and still
        published. The 2026-09-01 round found five content-free `success` renders and not
        one of them named a provider (q45088's 127-char SPA tab list, q45215's 385 chars
        of Kazakh region names), so the gate was withholding one shape of chrome and
        publishing the other.
        """
        session = FakeSession(
            {"https://plain.example.com/p": FakeResponse(200, body=_embed_shell_page("<div>chart goes here</div>"))}
        )

        result = await _fetch_one(session, "https://plain.example.com/p", {})

        assert result.status == "no_resolving_content"
        assert result.status_reason == "thin_page"
        assert result.http_status == 200  # the fetch itself succeeded; the CONTENT did not arrive
        assert result.text == ""
        assert result.unreadable_embeds == []

    async def test_prose_plus_an_unreadable_embed_keeps_the_prose_and_discloses_the_gap(
        self, tracker_with_infogram_html
    ):
        """The 44554 page itself: 2.9k chars of real background around the embed. Withholding
        it would throw away readable evidence, so the text stays and the section says plainly
        that the embedded figures are not in it — the caveat above it claims primary grading
        evidence, so an unqualified success overstated what was retrieved."""
        session = FakeSession(
            {"https://tracker.example.com/senate/26": FakeResponse(200, body=tracker_with_infogram_html)}
        )

        result = await _fetch_one(session, "https://tracker.example.com/senate/26", {})

        assert result.status == "success"
        assert "simulate the election 50,000 times" in result.text
        assert result.unreadable_embeds == ["infogram"]
        assert "infogram embed(s) that this fetch cannot read" in result.text
        # The note LEADS the page text, and says "below" because of it — as a trailer
        # a head-preserving trim deleted it (see the aggregate-trim test below).
        assert result.text.startswith("[This page displays data through infogram")
        assert "NOT in the page text below" in result.text

    async def test_the_disclosure_is_budgeted_inside_the_per_url_cap(self, tracker_with_infogram_html, monkeypatch):
        # The note is budgeted out of the cap (like the Tier-2 dataset lead), never added
        # on top of it, so the per-URL bound the section budget relies on still holds.
        cap = 500
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_PER_URL_MAX_CHARS", cap)
        session = FakeSession({"https://t.example.com/p": FakeResponse(200, body=tracker_with_infogram_html)})

        result = await _fetch_one(session, "https://t.example.com/p", {})

        assert result.status == "success"
        assert len(result.text) <= cap
        assert "NOT in the page text below" in result.text
        # The page text is what the cap truncates; the note is not the thing cut.
        body_cap = cap - len(_unreadable_embed_disclosure(["infogram"])) - 2
        assert f"[truncated at {body_cap} chars" in result.text

    def test_the_disclosure_survives_the_aggregate_section_budget_trim(self):
        """F6: the note used to TRAIL the page text, and every truncator here preserves the
        HEAD, so the aggregate cut in `_budgeted_success_sections` deleted it outright — the
        page then rendered under the "primary grading evidence" caption with no mention of the
        unreadable embed at all, which is the q44554/44556 failure the disclosure exists to
        prevent. Sizes are derived from the prod constants so the scenario stays a REACHABLE
        one: earlier full-size pages spend most of the total, and the embed page lands last.
        """
        per_url = resolution_source.RESOLUTION_SOURCE_PER_URL_MAX_CHARS
        total = resolution_source.RESOLUTION_SOURCE_TOTAL_MAX_CHARS
        leftover = per_url // 2  # what the embed page is left to render in
        spend = total - leftover
        filler_sizes = [per_url] * (spend // per_url)
        if spend % per_url:
            filler_sizes.append(spend % per_url)
        # Reachable on prod constants, which is what makes this a regression rather
        # than a hypothetical: the pages fit inside RESOLUTION_SOURCE_MAX_URLS.
        assert len(filler_sizes) + 1 <= resolution_source.RESOLUTION_SOURCE_MAX_URLS

        fillers = [
            FetchResult(
                url=f"https://p{i}.example.com/x",
                status="success",
                text="F" * size,
                http_status=200,
                content_type="text/html",
            )
            for i, size in enumerate(filler_sizes)
        ]
        embed_text = resolution_source._page_text_with_leads(
            "lorem ipsum " * (per_url // 2), "https://tracker.example.com/senate", ["infogram"]
        )
        embed = FetchResult(
            url="https://tracker.example.com/senate",
            status="success",
            text=embed_text,
            http_status=200,
            content_type="text/html",
            unreadable_embeds=["infogram"],
        )

        out = format_resolution_sections([*fillers, embed], datetime(2026, 9, 1, tzinfo=UTC))

        # The trim really fired — otherwise this test would pass for the wrong reason.
        assert embed_text not in out
        assert "infogram embed(s) that this fetch cannot read" in out
        assert "NOT in the page text below" in out
        # And it leads its own section, immediately under the heading.
        assert "### https://tracker.example.com/senate\n(fetched 2026-09-01)\n\n[This page displays" in out

    async def test_an_ordinary_article_carries_no_disclosure(self, article_html):
        session = FakeSession({"https://news.example.com/report": FakeResponse(200, body=article_html)})

        result = await _fetch_one(session, "https://news.example.com/report", {})

        assert result.status == "success"
        assert result.unreadable_embeds == []
        assert "cannot read" not in result.text

    def test_the_chrome_floor_sits_above_the_js_wall_floor(self):
        # Both floors read module globals so tests can retune them; the ordering is what
        # keeps `js_wall` its own population instead of a subset the chrome floor swallowed.
        assert looks_like_page_chrome("x" * 300) is True
        assert looks_like_js_wall("x" * 300) is False
        assert looks_like_page_chrome("x" * 500) is False

    def test_a_no_resolving_content_result_is_a_loss_token_not_ok(self):
        """The diagnostics half: as `success` this reported `ok`, so the provider block read
        fully healthy on a question whose only cited source handed back no numbers."""
        results = [
            FetchResult(
                url="https://tracker.example.com/senate",
                status="no_resolving_content",
                text="",
                http_status=200,
                content_type="text/html",
                unreadable_embeds=["infogram"],
            )
        ]

        assert _fetch_result_sources(results) == {"tracker.example.com": "no_resolving_content"}

    def test_a_no_resolving_content_page_is_not_rendered_as_grading_evidence(self):
        results = [
            FetchResult(
                url="https://tracker.example.com/senate",
                status="no_resolving_content",
                text="",
                http_status=200,
                content_type="text/html",
                unreadable_embeds=["infogram"],
            )
        ]

        out = format_resolution_sections(results, datetime(2026, 9, 1, tzinfo=UTC))

        assert "### https://tracker.example.com/senate" not in out
        assert "tracker.example.com: no_resolving_content" in out
        assert "weight other evidence accordingly" in out

    def test_a_blank_no_resolving_content_result_constructs(self):
        # The success-implies-content guard must not fire on the new status: it is a
        # FAILURE status and its text is empty by construction.
        assert FetchResult(
            url="https://t.example.com/p",
            status="no_resolving_content",
            text="",
            http_status=200,
            content_type="text/html",
        )

    async def test_a_page_just_above_the_chrome_floor_still_publishes(self):
        """The elbow, from both sides. The archive census puts the shortest extraction
        that carries the resolving content at exactly 401 chars
        (myfloridaelections.com's election-date table), so the floor has to withhold at
        399 and publish at 401 or it is throwing away terse-but-real data tables."""
        floor = resolution_source.RESOLUTION_SOURCE_EMBED_SHELL_MAX_CHARS
        # Trafilatura keeps the <article> paragraph verbatim, so the extraction length is
        # the paragraph length; "ab " * n is a word-shaped filler it does not collapse.
        above = _prose_page("ab " * ((floor + 40) // 3))
        below = _prose_page("ab " * ((floor - 40) // 3))
        session = FakeSession(
            {"https://a.example.com/p": FakeResponse(200, body=above)}
            | {"https://b.example.com/p": FakeResponse(200, body=below)}
        )

        long_result = await _fetch_one(session, "https://a.example.com/p", {})
        short_result = await _fetch_one(session, "https://b.example.com/p", {})

        assert len(long_result.text.strip()) >= floor
        assert long_result.status == "success"
        assert short_result.status == "no_resolving_content"
        assert short_result.status_reason == "thin_page"


class TestInlineChartData:
    """qid 43949 (IOM Missing Migrants). The resolving page fetches 200 through the repo's
    own Tier-1 path and trafilatura extracts ~80k chars of incident rows and prose carrying
    none of `1342` / `Total Dead and Missing` / `2026`, because the annual series lives in a
    `data-chart` attribute. A Wayback snapshot 25 days BEFORE that forecast carries the same
    markup with 2026 = 1,240; the published forecast sat ~340 above the true level."""

    async def test_a_prose_page_gets_its_chart_series_rendered_and_stays_a_success(self):
        session = FakeSession({"https://iom.example.com/med": FakeResponse(200, body=_iom_shaped_page())})

        result = await _fetch_one(session, "https://iom.example.com/med", {})

        assert result.status == "success"
        # The resolving figure, which the prose does not carry.
        assert "2026=1240" in result.text
        assert "Total Number of Dead and Missing: 2024=2573, 2025=2185, 2026=1240" in result.text
        # The chart block LEADS, so it is the last thing any downstream trim reaches.
        assert result.text.startswith(CHART_DATA_LEAD)
        # And the page's own prose is still there under it.
        assert "context in brief" in result.text

    async def test_a_chart_only_page_is_rescued_from_the_chrome_floor(self):
        """The two fixes meet here: without the chart rung this page is 60 chars of chrome
        and gets withheld as `thin_page`; with it, the numbers we recovered ARE the content."""
        session = FakeSession(
            {"https://iom.example.com/bare": FakeResponse(200, body=_iom_shaped_page(prose="Mediterranean."))}
        )

        result = await _fetch_one(session, "https://iom.example.com/bare", {})

        assert result.status == "success"
        assert result.status_reason is None
        assert "2026=1240" in result.text

    async def test_a_malformed_chart_payload_is_ignored_rather_than_raising(self):
        # A truncated attribute (`{"series":[{"name":}]`) is what a mid-response cut or a
        # non-JSON JS literal looks like. It must cost the page nothing: the prose still
        # publishes, with no chart block and no exception out of the provider.
        body = (
            "<!doctype html><html><body><article><h1>Counts</h1><p>"
            "Background prose long enough to clear the chrome floor on its own. " * 8 + "</p>"
            '<div class="charts-highchart" data-chart="{&quot;series&quot;:[{&quot;name&quot;:}]"></div>'
            "</article></body></html>"
        ).encode()
        session = FakeSession({"https://broken.example.com/p": FakeResponse(200, body=body)})

        result = await _fetch_one(session, "https://broken.example.com/p", {})

        assert result.status == "success"
        assert CHART_DATA_LEAD not in result.text
        assert "Background prose long enough" in result.text

    def test_a_config_with_no_parseable_series_renders_nothing(self):
        for html_text in (
            "",
            "<div>no charts here at all</div>",
            '<div data-chart="{}"></div>',
            '<div data-chart="{&quot;series&quot;:[]}"></div>',
            # A series whose data is callbacks / labels rather than numbers.
            '<div data-chart="{&quot;series&quot;:[{&quot;data&quot;:[&quot;n/a&quot;,null]}]}"></div>',
            # Valid JSON that is not an object.
            '<div data-chart="[1,2,3]"></div>',
        ):
            assert render_inline_chart_data(html_text) == ""

    def test_the_script_call_form_is_read_when_its_argument_is_json(self):
        html_text = (
            "<script>Highcharts.chart('container', "
            '{"title":{"text":"Weekly rate"},'
            '"xAxis":{"categories":["W1","W2"]},'
            '"series":[{"name":"Rate","data":[1.5,2.25]}]}'
            ");</script>"
        )

        out = render_inline_chart_data(html_text)

        assert "Chart 1 — Weekly rate" in out
        assert "Rate: W1=1.5, W2=2.25" in out

    def test_a_brace_inside_a_string_does_not_close_the_config_early(self):
        html_text = (
            '<script>new Highcharts.Chart({"title":{"text":"Deaths {2014-2026}"},'
            '"series":[{"name":"Total","data":[7]}]});</script>'
        )

        assert "Total: 7" in render_inline_chart_data(html_text)

    def test_point_object_and_pair_shapes_carry_their_own_labels(self):
        html_text = (
            '<div data-chart="'
            + _escape_config(
                {
                    "series": [
                        {"name": "Named", "data": [{"name": "Jan", "y": 4}, {"name": "Feb", "y": 5}]},
                        {"name": "Paired", "data": [["Q1", 10], ["Q2", 11.5]]},
                    ]
                }
            )
            + '"></div>'
        )

        out = render_inline_chart_data(html_text)

        assert "Named: Jan=4, Feb=5" in out
        assert "Paired: Q1=10, Q2=11.5" in out

    def test_a_declared_datetime_axis_renders_dates_not_epoch_millis(self):
        # Highcharts defines a datetime axis in ms since the epoch, UTC. Without the
        # conversion a tracker's own daily series renders `1756771200000=42`, which is
        # the shape most likely to matter rendered as noise.
        html_text = (
            '<div data-chart="'
            + _escape_config(
                {
                    "xAxis": {"type": "datetime"},
                    "series": [{"name": "Daily", "data": [[1788220800000, 41], [1788307200000, 42]]}],
                }
            )
            + '"></div>'
        )

        assert "Daily: 2026-09-01=41, 2026-09-02=42" in render_inline_chart_data(html_text)

    def test_a_numeric_x_axis_without_the_datetime_declaration_is_left_alone(self):
        # The conversion is keyed on the axis's own declaration, never on the magnitude
        # of the x values, so a chart plotting a large quantity on x is not re-dated.
        html_text = '<div data-chart="' + _escape_config({"series": [{"data": [[1788220800000, 41]]}]}) + '"></div>'

        out = render_inline_chart_data(html_text)

        assert "1788220800000=41" in out
        assert "2026-" not in out

    def test_long_series_keep_the_newest_points_and_say_so(self):
        # The resolving value is the newest one, so the window is taken from the END.
        n = resolution_chart_data.RESOLUTION_SOURCE_CHART_MAX_POINTS + 5
        html_text = (
            '<div data-chart="'
            + _escape_config(
                {
                    "xAxis": [{"categories": [f"m{i}" for i in range(n)]}],
                    "series": [{"name": "Monthly", "data": list(range(n))}],
                }
            )
            + '"></div>'
        )

        out = render_inline_chart_data(html_text)

        assert f"(last {resolution_chart_data.RESOLUTION_SOURCE_CHART_MAX_POINTS} of {n} points)" in out
        assert f"m{n - 1}={n - 1}" in out
        assert "m0=0" not in out

    def test_the_block_is_bounded_and_drops_whole_charts(self):
        # A half-rendered row reads like a complete series, so charts are dropped whole
        # and the omitted count is stated. The bound has to hold including that note.
        big = _escape_config(
            {
                "xAxis": [{"categories": [f"category-label-{i}" for i in range(16)]}],
                "series": [{"name": f"series-{s}", "data": [1000000 + i for i in range(16)]} for s in range(4)],
            }
        )
        html_text = "".join(f'<div data-chart="{big}"></div>' for _ in range(4))

        out = render_inline_chart_data(html_text)

        assert len(out) <= resolution_chart_data.RESOLUTION_SOURCE_CHART_BLOCK_MAX_CHARS
        assert "further chart(s) on this page omitted — chart-data budget" in out

    def test_at_most_max_charts_are_rendered(self):
        one = _escape_config({"series": [{"name": "S", "data": [1]}]})
        html_text = "".join(f'<div data-chart="{one}"></div>' for _ in range(8))

        out = render_inline_chart_data(html_text)

        assert out.count("\nChart ") == resolution_chart_data.RESOLUTION_SOURCE_CHART_MAX_CHARTS

    async def test_the_chart_block_is_budgeted_inside_the_per_url_cap(self, monkeypatch):
        # Same rule as the embed disclosure: leads come OUT of the per-URL cap, never on
        # top of it, so the aggregate section budget's arithmetic still holds.
        cap = 400
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_PER_URL_MAX_CHARS", cap)
        session = FakeSession({"https://iom.example.com/med": FakeResponse(200, body=_iom_shaped_page())})

        result = await _fetch_one(session, "https://iom.example.com/med", {})

        assert result.status == "success"
        assert len(result.text) <= cap


class TestResolutionSourceFetchMarker:
    """Item 19d: per-URL outcomes as ONE harvested marker line (`resolution_source_fetch`).

    The outcomes used to live only in free-text logs and the comment's diagnostics block,
    so "cdc.gov is 0 successes in 1,069 fetch records" meant re-scraping GHA logs that
    expire at 90 days.
    """

    async def test_one_line_per_fetched_url_with_status_and_http_code(self, article_html, monkeypatch, caplog):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {
                "https://www.bls.gov/cpi/": FakeResponse(200, body=article_html, content_type="text/html"),
                "https://cbp.gov/data": FakeResponse(403, body=b"", content_type="text/html"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://www.bls.gov/cpi/ and https://cbp.gov/data")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        lines = [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")]
        assert lines == [
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://www.bls.gov/cpi/ status=ok http=200 embeds=none",
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://cbp.gov/data status=blocked http=403 embeds=none",
        ]

    async def test_the_marker_names_the_unreadable_embed_providers(
        self, tracker_with_infogram_html, monkeypatch, caplog
    ):
        # The whole point on the 44554 shape: the fetch is a legitimate `success`, so the
        # only thing that makes the missing numbers queryable is this field.
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {"https://www.racetothewh.com/senate/26": FakeResponse(200, body=tracker_with_infogram_html)}
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://www.racetothewh.com/senate/26")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://www.racetothewh.com/senate/26 "
            "status=ok http=200 embeds=infogram"
        ]

    async def test_the_marker_names_which_rule_withheld_the_page(self, infogram_shell_html, monkeypatch, caplog):
        """`no_resolving_content` has two rules behind it and the status alone cannot say
        which fired. `reason` is what keeps the embed-gated population (queryable since
        2026-08) separable from the thin-page one the ungated floor added."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {
                "https://tracker.example.com/senate": FakeResponse(200, body=infogram_shell_html),
                "https://data.example.com/": FakeResponse(200, body=_embed_shell_page("<div>tabs</div>")),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://tracker.example.com/senate and https://data.example.com/")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://tracker.example.com/senate "
            "status=no_resolving_content http=200 embeds=infogram reason=embed_shell",
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://data.example.com/ "
            "status=no_resolving_content http=200 embeds=none reason=thin_page",
        ]

    async def test_a_fetch_that_never_got_a_response_reports_http_n_a(self, monkeypatch, caplog):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession({"https://slow.example.com/x": TimeoutError()})
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://slow.example.com/x")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert [m for m in caplog.messages if m.startswith("RESOLUTION_SOURCE_FETCH:")] == [
            "RESOLUTION_SOURCE_FETCH: question=999 url=https://slow.example.com/x status=error http=n/a embeds=none"
        ]

    async def test_no_fetch_is_logged_twice(self, article_html, monkeypatch, caplog):
        """One outcome line per fetch, in one format. The free-text
        `resolution_source fetched <netloc> (<status>)` lines the marker supersedes were
        deleted rather than left beside it; what remains are REASON lines (a decode score,
        an unread content-type, an SSRF rejection) that carry what the marker cannot."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {
                "https://www.bls.gov/cpi/": FakeResponse(200, body=article_html, content_type="text/html"),
                "https://cbp.gov/data": FakeResponse(403, body=b"", content_type="text/html"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(resolution_criteria="See https://www.bls.gov/cpi/ and https://cbp.gov/data")

        with caplog.at_level("INFO", logger="metaculus_bot.research.resolution_source"):
            await resolution_source_provider(is_benchmarking=False)(q)

        assert not [m for m in caplog.messages if "resolution_source fetched" in m]


class TestFetchResolutionSources:
    async def test_per_host_serialization(self, article_html, monkeypatch):
        """Two URLs on the same host must never fetch concurrently, while
        distinct hosts may. We track per-host in-flight counts in FakeSession
        and assert peak == 1 for the shared host."""

        # Slow the two same-host reads so their windows would overlap without the semaphore.
        original_read = FakeResponse.read
        slow_hosts_seen: dict[str, int] = {}

        async def slow_read(self: FakeResponse) -> bytes:
            host_probe = "same-host"  # marker for diagnostic only
            slow_hosts_seen[host_probe] = slow_hosts_seen.get(host_probe, 0) + 1
            # A microscopic sleep gives the event loop a chance to schedule the
            # second same-host coroutine — the semaphore must hold it back.
            await asyncio.sleep(0.01)
            return await original_read(self)  # type: ignore[misc]

        monkeypatch.setattr(FakeResponse, "read", slow_read)

        # Provide two URLs on the SAME host (must serialize) and one on a DIFFERENT host.
        session = FakeSession(
            {
                "https://a.example.com/one": FakeResponse(200, body=article_html, content_type="text/html"),
                "https://a.example.com/two": FakeResponse(200, body=article_html, content_type="text/html"),
                "https://b.example.com/three": FakeResponse(200, body=article_html, content_type="text/html"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources(
            [
                "https://a.example.com/one",
                "https://a.example.com/two",
                "https://b.example.com/three",
            ],
        )
        assert len(results) == 3
        # Same-host peak must be exactly 1 — that's the semaphore's guarantee.
        assert session.host_peak["a.example.com"] == 1
        # Other host observed at most 1 concurrent request (only one URL scheduled to it).
        assert session.host_peak.get("b.example.com", 0) == 1
        # Session was closed.
        assert session.closed is True

    async def test_redirect_convergence_serializes_on_final_host(self, article_html, monkeypatch):
        """F15 regression: two chains starting on DISTINCT hosts that both
        redirect to the SAME final host must serialize there. Keying the
        semaphore on the original URL's netloc (the old bug) gives each task
        its own semaphore, so the shared final host sees concurrency 2."""

        class SlowReadResponse(FakeResponse):
            async def read(self) -> bytes:
                # Keep the final-host GET context open long enough for the
                # other task's GET to arrive — without per-hop semaphores the
                # two windows overlap and host_peak records 2.
                await asyncio.sleep(0.01)
                return self._body

        session = FakeSession(
            {
                "https://a.example.com/one": FakeResponse(302, headers={"Location": "https://c.example.com/final"}),
                "https://b.example.com/two": FakeResponse(302, headers={"Location": "https://c.example.com/final"}),
                "https://c.example.com/final": SlowReadResponse(200, body=article_html, content_type="text/html"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await fetch_resolution_sources(
            ["https://a.example.com/one", "https://b.example.com/two"],
        )
        assert [r.status for r in results] == ["success", "success"]
        # The politeness guarantee holds at the CONVERGED host, not just the
        # original ones: never more than one in-flight request to c.example.com.
        assert session.host_peak["c.example.com"] == 1

    async def test_redirect_revisiting_initial_host_does_not_deadlock(self, article_html, monkeypatch):
        """A→B→A chain: strict per-hop acquire/release must never re-acquire a
        semaphore the task still holds (asyncio semaphores are not reentrant).
        wait_for turns a reentrancy regression into a fast TimeoutError
        instead of hanging the suite."""
        session = FakeSession(
            {
                "https://a.example.com/start": FakeResponse(302, headers={"Location": "https://b.example.com/mid"}),
                "https://b.example.com/mid": FakeResponse(302, headers={"Location": "https://a.example.com/final"}),
                "https://a.example.com/final": FakeResponse(200, body=article_html, content_type="text/html"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        results = await asyncio.wait_for(
            fetch_resolution_sources(["https://a.example.com/start"]),
            timeout=5.0,
        )
        assert len(results) == 1
        assert results[0].status == "success"
        assert results[0].url == "https://a.example.com/final"

    async def test_unexpected_error_cancels_and_drains_an_in_flight_sibling(self, monkeypatch):
        """The other half of the F5 teardown guard (the wall-clock-cancellation half
        is pinned in the Datawrapper suite): when one task dies on an exception the
        fetcher does NOT catch, the gather re-raises immediately and its still-running
        siblings must be cancelled and drained BEFORE the session closes. Closing
        first is what yanks transports out from under live requests."""
        events: list[str] = []

        class _HangingResponse(FakeResponse):
            async def read(self) -> bytes:
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    events.append("sibling-settled")
                    raise
                raise AssertionError("unreachable: the sibling should be cancelled")

        class _EventSession(FakeSession):
            async def close(self) -> None:
                events.append("session-closed")
                await super().close()

        session = _EventSession(
            {
                "https://slow.example.com/x": _HangingResponse(200, body=b"", content_type="text/html"),
                # RuntimeError is outside the (ClientError, TimeoutError) the fetcher
                # handles, so it propagates out of the gather.
                "https://broken.example.com/y": RuntimeError("driver blew up mid-fetch"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        with pytest.raises(RuntimeError, match="driver blew up mid-fetch"):
            await asyncio.wait_for(
                fetch_resolution_sources(["https://slow.example.com/x", "https://broken.example.com/y"]),
                timeout=5.0,
            )

        assert events == ["sibling-settled", "session-closed"]
        assert session.host_inflight["slow.example.com"] == 0
        assert session.closed is True


# ---------------------------------------------------------------------------
# 3. Factory + gating
# ---------------------------------------------------------------------------


class TestResolutionSourceProvider:
    async def test_flag_off_returns_empty(self, monkeypatch):
        monkeypatch.delenv("RESOLUTION_SOURCE_ENABLED", raising=False)
        provider = resolution_source_provider(is_benchmarking=False)
        out = await provider(_mock_question(resolution_criteria="See https://www.bls.gov/cpi/"))
        assert out == ""

    async def test_benchmarking_hard_disables_even_with_flag_on(self, monkeypatch):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        provider = resolution_source_provider(is_benchmarking=True)
        out = await provider(_mock_question(resolution_criteria="See https://www.bls.gov/cpi/"))
        assert out == ""

    async def test_no_urls_returns_empty(self, monkeypatch):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        provider = resolution_source_provider(is_benchmarking=False)
        out = await provider(_mock_question(resolution_criteria="No URLs here at all."))
        assert out == ""

    async def test_happy_path_end_to_end(self, article_html, monkeypatch):
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")

        session = FakeSession(
            {"https://www.bls.gov/cpi/": FakeResponse(200, body=article_html, content_type="text/html")}
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        provider = resolution_source_provider(is_benchmarking=False)
        out = await provider(_mock_question(resolution_criteria="See https://www.bls.gov/cpi/ for the reading."))
        assert "### https://www.bls.gov/cpi/" in out
        assert "Bureau of Labor Statistics" in out
        assert session.closed is True

    async def test_all_fetches_fail_surfaces_notice_end_to_end(self, monkeypatch):
        # Non-benchmarking: a 403 on the sole resolution URL must surface the
        # unreachable notice through the full provider path (feeds the header).
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession({"https://www.bls.gov/cpi/": FakeResponse(403, body=b"", content_type="text/html")})
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        provider = resolution_source_provider(is_benchmarking=False)
        out = await provider(_mock_question(resolution_criteria="See https://www.bls.gov/cpi/ for the reading."))
        assert "www.bls.gov: blocked" in out
        assert "nothing from the cited resolving page(s) is in this bundle; weight other evidence accordingly" in out

    async def test_benchmarking_disables_even_when_all_fetches_fail(self, monkeypatch):
        # The leakage guard fires before format_resolution_sections, so the new
        # all-failed notice must NOT leak into a benchmarking run.
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession({"https://www.bls.gov/cpi/": FakeResponse(403, body=b"", content_type="text/html")})
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        provider = resolution_source_provider(is_benchmarking=True)
        out = await provider(_mock_question(resolution_criteria="See https://www.bls.gov/cpi/ for the reading."))
        assert out == ""

    async def test_records_partial_fetch_detail_for_diagnostics(self, article_html, monkeypatch):
        """A partial fetch (one URL ok, one blocked) records a per-source token map so the
        diagnostics line can surface the loss even though the provider status stays `ok`."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {
                "https://www.bls.gov/cpi/": FakeResponse(200, body=article_html, content_type="text/html"),
                "https://cbp.gov/data": FakeResponse(403, body=b"", content_type="text/html"),
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        q = _mock_question(resolution_criteria="See https://www.bls.gov/cpi/ and https://cbp.gov/data")
        await resolution_source_provider(is_benchmarking=False)(q)

        sources = pop_provider_detail(q.id_of_question, "resolution_source")["sources"]
        assert sources["www.bls.gov"] == "ok"  # a fetched URL normalizes to "ok"
        assert sources["cbp.gov"] == "blocked"  # the failure keeps its FetchStatus token

    async def test_records_all_ok_detail_when_every_url_fetches(self, article_html, monkeypatch):
        """A fully-healthy fetch records every source as `ok` — the formatter renders no
        degradation suffix, so a clean resolution_source reads unchanged."""
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        session = FakeSession(
            {"https://www.bls.gov/cpi/": FakeResponse(200, body=article_html, content_type="text/html")}
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)

        q = _mock_question(resolution_criteria="See https://www.bls.gov/cpi/ for the reading.")
        await resolution_source_provider(is_benchmarking=False)(q)

        assert pop_provider_detail(q.id_of_question, "resolution_source")["sources"] == {"www.bls.gov": "ok"}

    def test_duplicate_domains_keep_both_outcomes(self):
        """Two URLs on the SAME domain are common (a stats site's index + data page).
        The source map is keyed by domain, so without the `#N` suffix the second URL's
        outcome silently overwrites the first — a blocked page would vanish behind a
        sibling that fetched fine, and the diagnostics line would read healthy."""
        results = [
            FetchResult(url="https://www.bls.gov/cpi/", status="success", text="x", http_status=200, content_type=None),
            FetchResult(url="https://www.bls.gov/ppi/", status="blocked", text="", http_status=403, content_type=None),
            FetchResult(url="https://www.bls.gov/ces/", status="js_wall", text="", http_status=200, content_type=None),
        ]

        sources = _fetch_result_sources(results)

        assert len(sources) == 3, f"one outcome was overwritten: {sources}"
        assert sorted(sources.values()) == ["blocked", "js_wall", "ok"]
        assert sources["www.bls.gov"] == "ok"
        assert sources["www.bls.gov#2"] == "blocked"
        assert sources["www.bls.gov#3"] == "js_wall"

    def test_an_empty_body_is_a_loss_token_not_ok(self):
        """The diagnostics half of the empty-200 defect: it reported `ok`, so the block read fully
        healthy on a question whose only cited source returned nothing."""
        results = [
            FetchResult(url="https://x.test/a", status="empty_body", text="", http_status=200, content_type="text/csv")
        ]

        assert _fetch_result_sources(results) == {"x.test": "empty_body"}


# ---------------------------------------------------------------------------
# 4. SSRF guard: is_public_http_url + _fetch_one integration + redirect hardening
# ---------------------------------------------------------------------------


# Fake getaddrinfo results: (family, type, proto, canonname, sockaddr) tuples.
# aiohttp/socket only cares about sockaddr[0] (the IP string) for our guard.
def _addrinfo(ip: str) -> tuple:
    return (0, 0, 0, "", (ip, 0))


class TestIsPublicHttpUrl:
    """Unit tests for the SSRF guard's URL-safety predicate.

    Runs against the async helper because DNS resolution must be awaitable
    off the event loop. Pure (non-DNS) rejections short-circuit before any
    resolver call — verified by never patching getaddrinfo in those cases.
    """

    async def test_rejects_non_http_scheme(self):
        assert await resolution_source.is_public_http_url("ftp://example.com/x") is False
        assert await resolution_source.is_public_http_url("file:///etc/passwd") is False
        assert await resolution_source.is_public_http_url("javascript:alert(1)") is False

    async def test_rejects_userinfo(self):
        # `https://trusted@169.254.169.254/` — the userinfo pretends to be a
        # trusted host in casual reading but the request goes to the IMDS.
        assert await resolution_source.is_public_http_url("https://trusted@169.254.169.254/") is False
        assert await resolution_source.is_public_http_url("https://user:pass@example.com/x") is False

    async def test_rejects_ipv4_link_local(self):
        # AWS IMDS lives at 169.254.169.254 — the canonical SSRF target.
        assert await resolution_source.is_public_http_url("http://169.254.169.254/latest/meta-data/") is False

    async def test_rejects_ipv4_loopback(self):
        assert await resolution_source.is_public_http_url("http://127.0.0.1/") is False
        assert await resolution_source.is_public_http_url("http://127.0.0.1:8000/admin") is False

    async def test_rejects_ipv4_private_ranges(self):
        assert await resolution_source.is_public_http_url("http://10.0.0.5/") is False
        assert await resolution_source.is_public_http_url("http://192.168.1.1/") is False
        assert await resolution_source.is_public_http_url("http://172.16.0.1/") is False

    async def test_rejects_bracketed_ipv6_loopback(self):
        assert await resolution_source.is_public_http_url("http://[::1]/") is False

    async def test_rejects_bracketed_ipv6_link_local(self):
        assert await resolution_source.is_public_http_url("http://[fe80::1]/") is False

    async def test_accepts_public_ipv4_literal(self, monkeypatch):
        # A public IP literal should NOT trigger DNS resolution — the ip_address
        # branch decides. Patch getaddrinfo to fail loudly to prove that.
        def _fail(*_args, **_kwargs):
            raise AssertionError("getaddrinfo must not be called for IP literals")

        monkeypatch.setattr(resolution_source.socket, "getaddrinfo", _fail)
        assert await resolution_source.is_public_http_url("http://8.8.8.8/") is True

    async def test_rejects_hostname_resolving_to_private(self, monkeypatch):
        # Patch getaddrinfo to return a private IP for the hostname.
        async def _fake_ainfo(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002  # mirrors socket.getaddrinfo
            del host, port, family, type, proto, flags
            return [_addrinfo("10.0.0.5")]

        # is_public_http_url should call asyncio.to_thread(socket.getaddrinfo, ...);
        # patch socket.getaddrinfo (the sync version) with a plain callable that
        # returns the same shape.
        def _sync_ainfo(host, port, *args, **kwargs):
            del host, port, args, kwargs
            return [_addrinfo("10.0.0.5")]

        monkeypatch.setattr(resolution_source.socket, "getaddrinfo", _sync_ainfo)
        assert await resolution_source.is_public_http_url("https://malicious.example.com/x") is False

    async def test_rejects_hostname_where_any_address_is_private(self, monkeypatch):
        # If ANY resolved address is private, reject — protects against DNS
        # rebinding-style multi-answer attacks (public + private).
        def _sync_ainfo(host, port, *args, **kwargs):
            del host, port, args, kwargs
            return [_addrinfo("8.8.8.8"), _addrinfo("127.0.0.1")]

        monkeypatch.setattr(resolution_source.socket, "getaddrinfo", _sync_ainfo)
        assert await resolution_source.is_public_http_url("https://mixed.example.com/") is False

    async def test_accepts_hostname_resolving_to_public(self, monkeypatch):
        def _sync_ainfo(host, port, *args, **kwargs):
            del host, port, args, kwargs
            return [_addrinfo("8.8.8.8")]

        monkeypatch.setattr(resolution_source.socket, "getaddrinfo", _sync_ainfo)
        assert await resolution_source.is_public_http_url("https://google.example.com/") is True

    async def test_rejects_on_dns_failure(self, monkeypatch):
        # DNS failure -> treat as unfetchable (would fail the fetch anyway).
        # We reject at the guard so the caller uniformly emits ssrf_blocked.

        def _sync_ainfo(host, port, *args, **kwargs):
            del host, port, args, kwargs
            raise _socket.gaierror("nodename nor servname provided")

        monkeypatch.setattr(resolution_source.socket, "getaddrinfo", _sync_ainfo)
        assert await resolution_source.is_public_http_url("https://nxdomain.example.com/") is False


class TestFetchOneSsrf:
    async def test_direct_fetch_of_link_local_is_ssrf_blocked(self):
        # Even if a broken handler is registered, the guard must reject before
        # session.get is ever called. Use a session with NO handlers to prove it.
        session = FakeSession({})
        result = await _fetch_one(session, "http://169.254.169.254/latest/meta-data/", {})
        assert result.status == "ssrf_blocked"
        assert result.text == ""
        # http_status is None (no request ever made).
        assert result.http_status is None

    async def test_direct_fetch_of_userinfo_url_is_ssrf_blocked(self):
        session = FakeSession({})
        result = await _fetch_one(session, "https://trusted@169.254.169.254/", {})
        assert result.status == "ssrf_blocked"
        assert result.http_status is None

    async def test_redirect_to_private_ip_is_ssrf_blocked(self, article_html):
        # 302 redirect from a public URL to the IMDS. Public URL passes the
        # guard, session.get returns 302 with Location, the guard re-runs on
        # the Location and rejects -> ssrf_blocked.
        session = FakeSession(
            {
                "https://redirect.example.com/x": FakeResponse(
                    302,
                    body=b"",
                    content_type="text/html",
                    headers={"Location": "http://169.254.169.254/latest/meta-data/"},
                ),
            }
        )
        result = await _fetch_one(session, "https://redirect.example.com/x", {})
        assert result.status == "ssrf_blocked"

    async def test_redirect_to_metaculus_is_blocked(self):
        # 302 from a public URL to metaculus.com. The SSRF guard passes (metaculus
        # is public) but the self-ref check stops the hop — no GET of metaculus.com
        # (FakeSession has no handler for it, so a follow would raise), status blocked.
        session = FakeSession(
            {
                "https://redirect.example.com/x": FakeResponse(
                    302,
                    body=b"",
                    content_type="text/html",
                    headers={"Location": "https://www.metaculus.com/questions/12345/"},
                ),
            }
        )
        result = await _fetch_one(session, "https://redirect.example.com/x", {})
        assert result.status == "blocked"
        assert result.url == "https://www.metaculus.com/questions/12345/"

    async def test_single_redirect_to_public_page_succeeds(self, article_html):
        # 302 from one public URL to another public URL — the loop follows,
        # extracts the final HTML, and returns success.
        session = FakeSession(
            {
                "https://start.example.com/x": FakeResponse(
                    302,
                    body=b"",
                    content_type="text/html",
                    headers={"Location": "https://final.example.com/report"},
                ),
                "https://final.example.com/report": FakeResponse(200, body=article_html, content_type="text/html"),
            }
        )
        result = await _fetch_one(session, "https://start.example.com/x", {})
        assert result.status == "success"
        # Final URL wins in the returned URL field so the section header points
        # readers at the actual page fetched, not the redirect stub.
        assert result.url == "https://final.example.com/report"
        assert "Bureau of Labor Statistics" in result.text

    async def test_redirect_chain_exceeding_max_hops_is_error_or_blocked(self):
        # Build a 7-step chain — the fetcher's 5-hop cap should trip.
        handlers: dict[str, Any] = {}
        for i in range(7):
            handlers[f"https://hop{i}.example.com/"] = FakeResponse(
                302,
                body=b"",
                content_type="text/html",
                headers={"Location": f"https://hop{i + 1}.example.com/"},
            )
        # Final target (never reached) — kept so no missing-handler AssertionError.
        handlers["https://hop7.example.com/"] = FakeResponse(200, body=b"<html><body>ok</body></html>")
        session = FakeSession(handlers)
        result = await _fetch_one(session, "https://hop0.example.com/", {})
        # Runaway redirect chain — reject conservatively. Either classification
        # is acceptable; the point is we don't follow past the cap or return success.
        assert result.status in ("error", "ssrf_blocked")
        assert result.text == ""

    async def test_redirect_missing_location_header_is_error(self):
        # 301 without a Location: header — malformed, treat as error.
        session = FakeSession(
            {
                "https://noloc.example.com/x": FakeResponse(
                    301,
                    body=b"",
                    content_type="text/html",
                    headers={},  # explicit: no Location
                ),
            }
        )
        result = await _fetch_one(session, "https://noloc.example.com/x", {})
        assert result.status == "error"
        assert result.text == ""


class TestGetSessionUsesFilteringResolver:
    """The actual DNS-rebinding trust boundary lives at aiohttp's connect-time
    DNS lookup: _get_session must plumb a FilteringResolver seeded with
    _ip_is_disallowed into the TCPConnector, so aiohttp only ever dials IPs
    that pass the same predicate as the preflight guard."""

    async def test_connector_is_wired_to_filtering_resolver(self):
        session_cm = resolution_source._get_session()
        try:
            # session_cm is an aiohttp.ClientSession (build_session returns
            # the session directly, not an async context manager wrapper).
            connector = session_cm.connector
            assert connector is not None
            resolver = getattr(connector, "_resolver", None)
            assert isinstance(resolver, FilteringResolver), (
                f"expected FilteringResolver on the connector, got {type(resolver).__name__}"
            )
            # And that resolver's predicate is our SSRF disallowlist.
            assert resolver._disallow is resolution_source._ip_is_disallowed
        finally:
            await session_cm.close()

    async def test_filtering_resolver_rejects_cgnat_shared_range(self):
        # R5 addition: is_global covers CGNAT (100.64/10) that isn't in the
        # explicit predicate list. Direct spot-check on _ip_is_disallowed.

        cgnat = ipaddress.ip_address("100.64.0.1")
        assert resolution_source._ip_is_disallowed(cgnat) is True
        # And a legit public address still passes.
        public = ipaddress.ip_address("8.8.8.8")
        assert resolution_source._ip_is_disallowed(public) is False
