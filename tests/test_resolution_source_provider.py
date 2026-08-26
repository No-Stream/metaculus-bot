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
import socket as _socket
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock
from urllib.parse import urlparse

import aiohttp
import pytest

from metaculus_bot.research import resolution_source
from metaculus_bot.research.http_fetch import FilteringResolver
from metaculus_bot.research.provider_diagnostics import pop_provider_detail
from metaculus_bot.research.resolution_source import (
    FetchResult,
    _fetch_one,
    _fetch_result_sources,
    extract_source_urls,
    fetch_resolution_sources,
    format_resolution_sections,
    is_fred_url,
    is_metaculus_self_ref,
    is_yahoo_ticker_url,
    looks_like_js_wall,
    resolution_source_provider,
    select_fetchable_urls,
    strip_markdown_escapes,
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

    def __init__(self, resp: "FakeResponse"):
        self._resp = resp

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:  # noqa: ASYNC900
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
        return self._body  # noqa: ASYNC910

    async def text(self) -> str:
        return self._text  # noqa: ASYNC910

    async def __aenter__(self) -> "FakeResponse":
        return self  # noqa: ASYNC910

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        return None  # noqa: ASYNC910


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
        self._call_counts: dict[str, int] = {k: 0 for k in handlers}
        self.closed = False
        # host_inflight[host] = current concurrent count; peak observed value
        # captured in host_peak[host]. Provider must keep per-host peak == 1.
        self.host_inflight: dict[str, int] = {}
        self.host_peak: dict[str, int] = {}
        # Every URL requested, in order — lets tests pin which routes were
        # (and, critically for the Datawrapper hop, were NOT) fetched.
        self.requested: list[str] = []

    def get(self, url: str, **_kwargs: Any) -> "_TrackingResponse":
        self.requested.append(url)
        for prefix, handler_list in self._handlers.items():
            if url.startswith(prefix):
                idx = min(self._call_counts[prefix], len(handler_list) - 1)
                self._call_counts[prefix] += 1
                handler = handler_list[idx]
                if callable(handler) and not isinstance(handler, FakeResponse):
                    inner = handler(url)
                else:
                    inner = handler
                host = urlparse(url).netloc
                return _TrackingResponse(inner, host, self)
        raise AssertionError(f"no handler for URL {url}")

    async def close(self) -> None:  # noqa: ASYNC910
        self.closed = True

    async def __aenter__(self) -> "FakeSession":
        return self  # noqa: ASYNC910

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


class TestFormatResolutionSections:
    def test_empty_results_returns_empty_string(self):
        assert format_resolution_sections([], datetime(2026, 7, 9, tzinfo=timezone.utc)) == ""

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
        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=timezone.utc))
        assert out  # no longer empty
        assert "2 resolution source(s) could not be fetched" in out
        assert "a.com: blocked" in out
        assert "b.com: js_wall" in out
        assert "the resolving page was unreachable; weight other evidence accordingly" in out
        # Body only — the orchestrator prepends the "## Resolution Source Snapshot" header.
        assert "## Resolution Source Snapshot" not in out

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
        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=timezone.utc))
        # Success content still rendered.
        assert "### https://ok.com/data" in out
        assert "the reading is 3.2%" in out
        assert "primary grading evidence" in out
        # Terse note about the failed source appended.
        assert "bad.com: blocked" in out
        assert "could not be fetched" in out
        # The success path must not carry the all-failed sentence.
        assert "the resolving page was unreachable" not in out

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
        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=timezone.utc))
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
        out = resolution_source.format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=timezone.utc))
        # First section fits; later ones must be trimmed or dropped.
        assert "https://example.com/0" in out
        # We should NOT see all four full 300-char blocks packed together.
        assert out.count("A" * 300) <= 2

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
        out = resolution_source.format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=timezone.utc))
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
        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=timezone.utc))
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
        session = FakeSession({"https://slow.example.com/x": asyncio.TimeoutError()})
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
        assert result.content_type is not None and "json" in result.content_type
        assert result.text.startswith('{"vulnerabilities')
        # Truncated -> marker appears, total bounded by cap.
        assert f"[truncated at {cap} chars — full source at https://json.example.com/kev]" in result.text
        assert len(result.text) <= cap

    async def test_pdf_content_type_is_unsupported(self):
        # PDF: body is NEVER read (per the plan). A read() that raises verifies that.
        class UnreadableResponse(FakeResponse):
            async def read(self) -> bytes:  # noqa: ASYNC910
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
        assert "the resolving page was unreachable; weight other evidence accordingly" in out

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
        async def _fake_ainfo(host, port, family=0, type=0, proto=0, flags=0):
            del host, port, family, type, proto, flags  # noqa: A001,A002
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
