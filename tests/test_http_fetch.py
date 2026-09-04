"""Tests for the shared aiohttp HTTP utilities (`metaculus_bot/research/http_fetch.py`).

Covers:
- `BROWSER_HEADERS` completeness (Safari-like UA + Accept / Accept-Language / Accept-Encoding)
- `build_session` config plumbing (ClientTimeout total+sock_read, TCPConnector limit, headers, resolver)
- `read_body_capped` under/at/over the byte cap (over -> None + WARNING), including
  multi-chunk streaming and mid-stream abort without consuming the remaining stream
- `FilteringResolver` filters private IPs, raises OSError when everything is filtered,
  and delegates `close()` to its inner resolver.
- Datawrapper embed detection (`extract_datawrapper_charts`) on real-shaped
  tracker HTML, the live-data URL builder, and Last-Modified parsing.
- Routeless data-embed detection (`unreadable_data_embed_providers`) on each
  provider's own published embed snippet.
- `decode_text_body`: BOM-then-declared-charset precedence, and the undecodable-char
  score that lets a caller refuse mojibake instead of rendering it as evidence.

No real network calls: sessions are constructed, inspected, and closed.
"""

from __future__ import annotations

import codecs
import ipaddress
import logging
import socket
import ssl
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import certifi
import pytest
from aiohttp.abc import AbstractResolver

from metaculus_bot.research.http_fetch import (
    BROWSER_HEADERS,
    MAX_UNDECODABLE_CHAR_RATIO,
    FilteringResolver,
    build_session,
    datawrapper_live_data_url,
    decode_text_body,
    extract_datawrapper_charts,
    parse_http_last_modified,
    read_body_capped,
    undecodable_char_ratio,
    unreadable_data_embed_providers,
)


def _resolve_entry(ip: str, host: str = "example.com") -> dict[str, Any]:
    """Return a dict matching aiohttp.abc.ResolveResult (TypedDict)."""
    return {
        "hostname": host,
        "host": ip,
        "port": 0,
        "family": int(socket.AF_INET if ":" not in ip else socket.AF_INET6),
        "proto": 0,
        "flags": 0,
    }


class FakeInnerResolver(AbstractResolver):
    """Inner resolver that returns fixed entries and tracks close()."""

    def __init__(self, entries: list[dict[str, Any]]):
        self._entries = entries
        self.close_called = False

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[Any]:
        del host, port, family
        return list(self._entries)

    async def close(self) -> None:
        self.close_called = True


def _all_disallowed(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    del ip
    return True


def _reject_private(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return ip.is_private or ip.is_loopback


class FakeStreamContent:
    """Stub for `resp.content`: yields pre-set chunks and counts how many were consumed."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks
        self.chunks_consumed = 0

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        del n  # chunk boundaries are dictated by the test's pre-set chunks
        for chunk in self._chunks:
            self.chunks_consumed += 1
            yield chunk


class FakeStreamResponse:
    """Minimal stub exposing the `.content.iter_chunked` surface `read_body_capped` consumes."""

    def __init__(self, chunks: list[bytes]):
        self.content = FakeStreamContent(chunks)


def _certifi_context() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=certifi.where())


class TestBrowserHeaders:
    def test_contains_required_keys(self):
        for key in ("User-Agent", "Accept", "Accept-Language", "Accept-Encoding"):
            assert key in BROWSER_HEADERS, f"missing header {key}"

    def test_user_agent_is_safari_like(self):
        ua = BROWSER_HEADERS["User-Agent"]
        assert "Safari" in ua
        assert "Mozilla/5.0" in ua

    def test_accept_negotiates_html(self):
        assert "text/html" in BROWSER_HEADERS["Accept"]

    def test_accept_encoding_excludes_brotli(self):
        # aiohttp's brotli decoder isn't a project dep (HAS_BROTLI=False), so
        # advertising `br` would have Brotli-preferring servers return content
        # aiohttp can't decode -> ClientResponseError -> silent source drop.
        # We advertise only gzip/deflate.
        assert BROWSER_HEADERS["Accept-Encoding"] == "gzip, deflate"
        assert "br" not in BROWSER_HEADERS["Accept-Encoding"]


class TestBuildSession:
    async def test_configures_timeout_connector_and_headers(self):
        async with build_session(timeout_s=12.5, connector_limit=7, headers=BROWSER_HEADERS) as session:
            assert session.timeout.total == 12.5
            assert session.timeout.sock_read == 12.5
            assert session.connector is not None
            assert session.connector.limit == 7
            assert session.headers["User-Agent"] == BROWSER_HEADERS["User-Agent"]
            assert session.headers["Accept-Language"] == BROWSER_HEADERS["Accept-Language"]

    async def test_defaults_connector_limit_20_and_no_default_headers(self):
        async with build_session(timeout_s=5.0) as session:
            assert session.timeout.total == 5.0
            assert session.timeout.sock_read == 5.0
            assert session.connector is not None
            assert session.connector.limit == 20
            # No headers arg -> no session-level User-Agent (aiohttp adds its own
            # at request time; prediction_market relies on this no-header default).
            assert "User-Agent" not in session.headers

    async def test_tls_trust_is_pinned_to_certifis_bundle(self):
        """Which sources are reachable must not depend on the machine's CA store: trade.gov
        failed CERTIFICATE_VERIFY_FAILED against the default store on 2026-09-03 and
        succeeded against certifi's, and a source lost that way reads as a dead host."""
        expected = {(cert.get("serialNumber"), cert.get("issuer")) for cert in _certifi_context().get_ca_certs()}

        async with build_session(timeout_s=5.0) as session:
            assert session.connector is not None
            # aiohttp keeps the connector's TLS config on the private `_ssl`; peeking is the
            # only way to assert the wiring without opening a real connection.
            context = getattr(session.connector, "_ssl", None)
            assert isinstance(context, ssl.SSLContext)
            assert {(cert.get("serialNumber"), cert.get("issuer")) for cert in context.get_ca_certs()} == expected
            assert expected, "a context loading no CA at all would verify nothing"

    async def test_header_size_caps_are_raised_above_aiohttps_default(self):
        """who.int (8,765 B) and visitwales.com (9,697 B) send a Content-Security-Policy
        header over aiohttp's 8,190-byte default, and the response is rejected before any
        body is read — arriving as `error http=None`, same as a host that never answered."""
        async with build_session(timeout_s=5.0) as session:
            assert session._max_line_size == 65536
            assert session._max_field_size == 65536

    async def test_resolver_kwarg_flows_into_connector(self):
        # build_session must plumb the resolver into TCPConnector so aiohttp's
        # connect-time DNS lookup goes through the caller's predicate.
        inner = FakeInnerResolver([_resolve_entry("8.8.8.8")])
        resolver = FilteringResolver(disallow=_reject_private, inner=inner)
        async with build_session(timeout_s=5.0, resolver=resolver) as session:
            assert session.connector is not None
            # aiohttp stores the resolver on TCPConnector as `_resolver`. Its
            # a private attribute — we peek at it as the only reliable way to
            # assert wiring without opening a real connection.
            assert getattr(session.connector, "_resolver", None) is resolver


class TestReadBodyCapped:
    async def test_returns_body_under_cap(self):
        body = b"x" * 100
        assert await read_body_capped(FakeStreamResponse([body]), max_bytes=200, label="under") == body

    async def test_returns_body_at_cap_boundary(self):
        body = b"x" * 200
        assert await read_body_capped(FakeStreamResponse([body]), max_bytes=200, label="at-cap") == body

    async def test_over_cap_returns_none_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = await read_body_capped(FakeStreamResponse([b"x" * 201]), max_bytes=200, label="mylabel")
        assert result is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "mylabel" in r.getMessage()]
        assert warnings, "expected a WARNING mentioning the label for the oversized body"

    async def test_multi_chunk_body_under_cap_is_joined(self):
        chunks = [b"alpha-", b"beta-", b"gamma"]
        result = await read_body_capped(FakeStreamResponse(chunks), max_bytes=200, label="multi")
        assert result == b"alpha-beta-gamma"

    async def test_over_cap_mid_stream_aborts_without_draining(self, caplog):
        # Cap 100, 50-byte chunks: chunk 2 lands exactly AT the cap (kept),
        # chunk 3 crosses it — the loop must bail there (bounding peak memory)
        # and never pull chunk 4 off the stream.
        resp = FakeStreamResponse([b"a" * 50, b"b" * 50, b"c" * 50, b"d" * 50])
        with caplog.at_level(logging.WARNING):
            result = await read_body_capped(resp, max_bytes=100, label="midstream")
        assert result is None
        assert resp.content.chunks_consumed == 3
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "midstream" in r.getMessage()]
        assert warnings, "expected a WARNING mentioning the label for the mid-stream over-cap abort"


class TestFilteringResolver:
    """Direct unit tests for the connect-time DNS-vetting resolver."""

    async def test_all_public_addresses_pass_through(self):
        inner = FakeInnerResolver([_resolve_entry("8.8.8.8"), _resolve_entry("1.1.1.1")])
        r = FilteringResolver(disallow=_reject_private, inner=inner)
        got = await r.resolve("example.com")
        assert [entry["host"] for entry in got] == ["8.8.8.8", "1.1.1.1"]

    async def test_private_addresses_filtered(self):
        # Public + private mixed answer (classic DNS-rebinding-lite payload).
        # The private survivor must be dropped; the public one kept.
        inner = FakeInnerResolver([_resolve_entry("8.8.8.8"), _resolve_entry("10.0.0.5")])
        r = FilteringResolver(disallow=_reject_private, inner=inner)
        got = await r.resolve("mixed.example.com")
        assert [entry["host"] for entry in got] == ["8.8.8.8"]

    async def test_all_disallowed_raises_os_error(self):
        # If every address is filtered out, mirror getaddrinfo semantics and
        # raise OSError so the fetch layer's existing except-clause catches it.
        inner = FakeInnerResolver([_resolve_entry("127.0.0.1"), _resolve_entry("10.0.0.5")])
        r = FilteringResolver(disallow=_all_disallowed, inner=inner)
        with pytest.raises(OSError, match="all resolved addresses disallowed"):
            await r.resolve("evil.example.com")

    async def test_close_delegates_to_inner(self):
        inner = FakeInnerResolver([_resolve_entry("8.8.8.8")])
        r = FilteringResolver(disallow=_reject_private, inner=inner)
        assert inner.close_called is False
        await r.close()
        assert inner.close_called is True


# ---------------------------------------------------------------------------
# Datawrapper embed detection (resolution-source Tier-2 hop)
# ---------------------------------------------------------------------------

# Miniature of the REAL natesilver.net Iran-war tracker markup (qid 44858):
# Substack wraps each Datawrapper embed in an HTML-escaped JSON `data-attrs`
# whose `url` pins a STALE version (`/1mU3g/11/` — the live chart was at
# v2570 when this shape was captured, 2026-08-25) and whose `title` follows
# two S3 thumbnail URLs. Structure preserved; prose trimmed.
_IRAN_TRACKER_SHAPED_HTML = (
    "<p><span>Polling on the war remains negative.</span></p>"
    '<div id="datawrapper-iframe" data-attrs="{&quot;url&quot;:&quot;https://datawrapper.dwcdn.net/1mU3g/11/&quot;,'
    "&quot;thumbnail_url&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/aaa_1220x708.png&quot;,"
    "&quot;thumbnail_url_full&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/bbb_1220x1076.png&quot;,"
    "&quot;height&quot;:527,&quot;title&quot;:&quot;Do Americans support or oppose the Iran War?&quot;,"
    '&quot;description&quot;:&quot;&quot;}"></div>'
    "<p>Methodology notes follow.</p>"
    '<div id="datawrapper-iframe" data-attrs="{&quot;url&quot;:&quot;https://datawrapper.dwcdn.net/VUUVz/7/&quot;,'
    "&quot;thumbnail_url&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/ccc_1220x994.png&quot;,"
    "&quot;height&quot;:673,&quot;title&quot;:&quot;Polls included in our Iran War support average&quot;,"
    '&quot;description&quot;:&quot;&quot;}"></div>'
)

# Miniature of the Trump-approval tracker (qid 44841): five charts, the
# resolving one (`kSCt4`) first in document order, one title carrying an
# apostrophe (the shape that truncates a naive quote-terminated regex).
_TRUMP_TRACKER_SHAPED_HTML = (
    '<div data-attrs="{&quot;url&quot;:&quot;https://datawrapper.dwcdn.net/kSCt4/349/&quot;,'
    "&quot;thumbnail_url&quot;:&quot;https://substack-post-media.s3.amazonaws.com/public/images/ddd_1260x660.png&quot;,"
    '&quot;height&quot;:478,&quot;title&quot;:&quot;Do Americans approve or disapprove of Donald Trump?&quot;}"></div>'
    '<div data-attrs="{&quot;url&quot;:&quot;https://datawrapper.dwcdn.net/vknzT/269/&quot;,'
    '&quot;height&quot;:827,&quot;title&quot;:&quot;Polls included in our average&quot;}"></div>'
    '<div data-attrs="{&quot;url&quot;:&quot;https://datawrapper.dwcdn.net/RFXsV/73/&quot;,'
    '&quot;height&quot;:448,&quot;title&quot;:&quot;Trump&#8217;s net approval on the issues&quot;}"></div>'
)


class TestExtractDatawrapperCharts:
    def test_iran_tracker_shape_ids_titles_document_order(self):
        charts = extract_datawrapper_charts(_IRAN_TRACKER_SHAPED_HTML)
        assert [c.chart_id for c in charts] == ["1mU3g", "VUUVz"]
        assert charts[0].title == "Do Americans support or oppose the Iran War?"
        assert charts[1].title == "Polls included in our Iran War support average"

    def test_trump_tracker_shape_resolving_chart_first(self):
        charts = extract_datawrapper_charts(_TRUMP_TRACKER_SHAPED_HTML)
        assert [c.chart_id for c in charts] == ["kSCt4", "vknzT", "RFXsV"]
        assert charts[0].title == "Do Americans approve or disapprove of Donald Trump?"

    def test_title_with_apostrophe_survives(self):
        # "Trump's net approval …" — a quote-class terminator would cut at the
        # apostrophe and label the chart just "Trump" (observed on the live page).
        charts = extract_datawrapper_charts(_TRUMP_TRACKER_SHAPED_HTML)
        assert charts[2].title == "Trump’s net approval on the issues"  # noqa: RUF001  # pins the live page's own curly apostrophe

    def test_plain_iframe_embed_with_title_attribute(self):
        # Datawrapper's own responsive embed: title= sits BEFORE src in the tag.
        html = (
            '<iframe title="Weekly jobless claims" aria-label="Interactive line chart" '
            'id="datawrapper-chart-Ab9x2" src="https://datawrapper.dwcdn.net/Ab9x2/4/" '
            'scrolling="no" frameborder="0"></iframe>'
        )
        charts = extract_datawrapper_charts(html)
        assert len(charts) == 1
        assert charts[0].chart_id == "Ab9x2"
        assert charts[0].title == "Weekly jobless claims"

    def test_a_title_on_a_neighbouring_element_is_not_borrowed(self):
        """The proximity window alone let an unrelated `title=` render as the chart's
        identity in the Tier-2 lead — and the lead names the chart it is serving data for,
        so a borrowed title is a false claim about which series is being read. The title
        has to sit inside the URL's own tag."""
        html = (
            '<a href="#" title="Share on X">share</a>'
            '<script defer src="https://datawrapper.dwcdn.net/Xy12Z/embed.js"></script>'
        )
        charts = extract_datawrapper_charts(html)

        assert [c.chart_id for c in charts] == ["Xy12Z"]
        assert charts[0].title is None

    def test_a_forward_json_title_on_a_later_element_is_not_borrowed(self):
        # Same rule in the forward direction: a `"title":` blob that belongs to the NEXT
        # element must not attach to this chart just because it fell inside the window.
        html = (
            '<script defer src="https://datawrapper.dwcdn.net/Xy12Z/embed.js"></script>'
            '<div data-attrs="{&quot;title&quot;:&quot;Some unrelated card&quot;}"></div>'
        )
        charts = extract_datawrapper_charts(html)

        assert [c.chart_id for c in charts] == ["Xy12Z"]
        assert charts[0].title is None

    def test_embed_script_form_yields_id_without_title(self):
        html = '<script defer src="https://datawrapper.dwcdn.net/Xy12Z/embed.js" charset="utf-8"></script>'
        charts = extract_datawrapper_charts(html)
        assert [c.chart_id for c in charts] == ["Xy12Z"]
        assert charts[0].title is None

    def test_json_escaped_slashes_match(self):
        # Embed URLs inside plain (non-HTML-escaped) JSON carry `\/` slashes.
        html = '{"url":"https:\\/\\/datawrapper.dwcdn.net\\/Qw3rT\\/12\\/","title":"Escaped chart"}'
        charts = extract_datawrapper_charts(html)
        assert [c.chart_id for c in charts] == ["Qw3rT"]
        assert charts[0].title == "Escaped chart"

    def test_static_data_url_form_matches(self):
        html = '<a href="https://static.dwcdn.net/data/kSCt4.csv">Get the data</a>'
        assert [c.chart_id for c in extract_datawrapper_charts(html)] == ["kSCt4"]

    def test_dedup_first_seen_wins(self):
        html = (
            '<iframe src="https://datawrapper.dwcdn.net/1mU3g/11/"></iframe>'
            '<a href="https://static.dwcdn.net/data/1mU3g.csv">data</a>'
        )
        assert [c.chart_id for c in extract_datawrapper_charts(html)] == ["1mU3g"]

    def test_longer_path_segments_do_not_match(self):
        # Chart ids are exactly 5 alnum chars; asset paths must not match.
        html = '<script src="https://datawrapper.dwcdn.net/plugins/render.js"></script>'
        assert extract_datawrapper_charts(html) == []

    def test_titleless_chart_does_not_steal_neighbours_title(self):
        # First embed has no title; the second (within the forward window)
        # does. The window is bounded at the next embed, so chart one stays
        # untitled instead of inheriting chart two's title.
        html = (
            '<iframe src="https://datawrapper.dwcdn.net/AAAA1/3/"></iframe>'
            '<div data-attrs="{&quot;url&quot;:&quot;https://datawrapper.dwcdn.net/BBBB2/5/&quot;,'
            '&quot;title&quot;:&quot;Chart two title&quot;}"></div>'
        )
        charts = extract_datawrapper_charts(html)
        assert [c.chart_id for c in charts] == ["AAAA1", "BBBB2"]
        assert charts[0].title is None
        assert charts[1].title == "Chart two title"

    def test_titleless_chart_does_not_steal_a_preceding_iframe_title(self):
        # Mirror image of the test above, for the BACKWARD half of the scan: the
        # first embed carries an iframe `title=` attribute, the second (embed.js
        # form) carries none and sits well inside the 300-char backward window.
        # The backward scan is clamped at the previous embed's URL, so chart two
        # stays untitled instead of being labelled with chart one's title — a
        # mislabelled dataset is worse than an unlabelled one, since the lead
        # line the forecaster reads names the chart.
        html = (
            '<iframe title="First chart" src="https://datawrapper.dwcdn.net/AAAA1/3/"></iframe>'
            '<script defer src="https://datawrapper.dwcdn.net/BBBB2/embed.js"></script>'
        )
        charts = extract_datawrapper_charts(html)
        assert [c.chart_id for c in charts] == ["AAAA1", "BBBB2"]
        assert charts[0].title == "First chart"
        assert charts[1].title is None

    def test_single_quoted_iframe_title_attribute(self):
        # The attribute form alternates on quote style; single-quoted markup is
        # common in hand-written embeds and must resolve to the same title.
        html = "<iframe title='Weekly jobless claims' src='https://datawrapper.dwcdn.net/Ab9x2/4/'></iframe>"
        charts = extract_datawrapper_charts(html)
        assert [c.chart_id for c in charts] == ["Ab9x2"]
        assert charts[0].title == "Weekly jobless claims"

    def test_json_title_beyond_the_forward_window_is_not_attached(self):
        # The forward window brackets the observed Substack layout (~380 chars
        # of thumbnail URLs between the embed URL and its title). A `"title"`
        # far past that belongs to some other page structure, so the chart stays
        # untitled rather than borrowing it.
        filler = "&quot;description&quot;:&quot;" + ("x" * 800) + "&quot;,"
        html = (
            '<div data-attrs="{&quot;url&quot;:&quot;https://datawrapper.dwcdn.net/Cccc3/9/&quot;,'
            f"{filler}"
            '&quot;title&quot;:&quot;Too far away to be this chart&quot;}"></div>'
        )
        charts = extract_datawrapper_charts(html)
        assert [c.chart_id for c in charts] == ["Cccc3"]
        assert charts[0].title is None

    def test_no_embeds_returns_empty(self):
        assert extract_datawrapper_charts("<html><body>No charts here.</body></html>") == []
        assert extract_datawrapper_charts("") == []


# Each snippet below is the provider's OWN published embed code (Infogram's
# embed-code generator, Flourish's developer docs, Tableau's "Writing Embed
# Code" help page), trimmed of the minified loader body. They are what a page
# author pastes in, so they are the shapes the scan has to recognize.
_INFOGRAM_EMBED_SNIPPET = (
    '<div class="infogram-embed" data-id="_/vs9b6iAeARko8cuwH51x" data-type="interactive" '
    'data-title="NE - Osborn v. Ricketts"></div>'
    '<script>!function(e,i,n,s){var t="InfogramEmbeds";}(document,0,"infogram-async",'
    '"https://e.infogram.com/js/dist/embed-loader-min.js");</script>'
)
_FLOURISH_EMBED_SNIPPET = (
    '<div class="flourish-embed flourish-chart" data-src="visualisation/4853699">'
    '<script src="https://public.flourish.studio/resources/embed.js"></script></div>'
)
_TABLEAU_V1_EMBED_SNIPPET = (
    "<script type='text/javascript' src='https://public.tableau.com/javascripts/api/viz_v1.js'></script>"
    "<div class='tableauPlaceholder' style='width:800; height:600;'>"
    "<object class='tableauViz' width='800' height='600' style='display:none;'></object></div>"
)
_TABLEAU_V3_EMBED_SNIPPET = (
    '<script type="module" src="https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js">'
    '</script><tableau-viz id="tableauViz" src="https://public.tableau.com/views/wb/view"></tableau-viz>'
)


class TestUnreadableDataEmbedProviders:
    """The routeless half of embed detection: providers whose numbers we cannot reach.

    qids 44554/44556 — a tracker page served 2.9k chars of forecast background over
    HTTP 200 while the resolving polling average sat in two Infogram iframes, and the
    fetch reported an unqualified success. Naming the provider is what lets the caller
    withhold an embed-only page or disclose the gap on a page that also carried prose.
    """

    def test_infogram_embed_snippet(self):
        assert unreadable_data_embed_providers(_INFOGRAM_EMBED_SNIPPET) == ["infogram"]

    def test_flourish_embed_snippet(self):
        assert unreadable_data_embed_providers(_FLOURISH_EMBED_SNIPPET) == ["flourish"]

    def test_tableau_v1_and_v3_embed_snippets(self):
        assert unreadable_data_embed_providers(_TABLEAU_V1_EMBED_SNIPPET) == ["tableau"]
        assert unreadable_data_embed_providers(_TABLEAU_V3_EMBED_SNIPPET) == ["tableau"]

    def test_infogram_iframe_form_matches_on_the_host(self):
        # The iframe variant carries no `infogram-embed` class — only the host.
        html = '<iframe src="https://e.infogram.com/_/vs9b6iAeARko8cuwH51x?src=embed"></iframe>'
        assert unreadable_data_embed_providers(html) == ["infogram"]

    def test_json_escaped_embed_url_matches(self):
        # Same tolerance the Datawrapper id regex carries: a `data-attrs` JSON blob
        # escapes its slashes.
        html = r'{"url":"https:\/\/public.flourish.studio\/visualisation\/4853699\/"}'
        assert unreadable_data_embed_providers(html) == ["flourish"]

    def test_datawrapper_is_not_reported(self):
        # Datawrapper HAS a route (the Tier-2 live-dataset hop), so its embeds are not
        # unreadable and its outcome rides that hop's own FetchStatus. Reporting it here
        # would relabel every js-walled tracker the hop already rescues.
        assert unreadable_data_embed_providers(_IRAN_TRACKER_SHAPED_HTML) == []

    def test_multiple_providers_in_document_order(self):
        html = f"<body><p>intro</p>{_TABLEAU_V1_EMBED_SNIPPET}<p>mid</p>{_INFOGRAM_EMBED_SNIPPET}</body>"
        assert unreadable_data_embed_providers(html) == ["tableau", "infogram"]

    def test_one_entry_per_provider_however_many_embeds(self):
        html = _INFOGRAM_EMBED_SNIPPET + _INFOGRAM_EMBED_SNIPPET
        assert unreadable_data_embed_providers(html) == ["infogram"]

    def test_prose_naming_a_provider_is_not_an_embed(self):
        # The disclosure this feeds is forecaster-facing ("the figures are NOT in the
        # page text below"), so a page that merely CREDITS a tool in prose must not trip it.
        html = "<p>The chart was built with Infogram and Tableau by our data team.</p>"
        assert unreadable_data_embed_providers(html) == []

    def test_no_embeds_returns_empty(self):
        assert unreadable_data_embed_providers("<html><body>Plain prose.</body></html>") == []
        assert unreadable_data_embed_providers("") == []


class TestDatawrapperLiveDataUrl:
    def test_builds_version_free_static_route(self):
        # The one dataset route that serves LIVE data. The versioned
        # `datawrapper.dwcdn.net/<id>/<ver>/dataset.csv` form pinned in page
        # HTML served 5- and 14-month-stale snapshots on the two real trackers
        # (2026-08-24 verifications) and must never be constructed.
        assert datawrapper_live_data_url("1mU3g") == "https://static.dwcdn.net/data/1mU3g.csv"
        assert datawrapper_live_data_url("kSCt4") == "https://static.dwcdn.net/data/kSCt4.csv"

    @pytest.mark.parametrize(
        "bad_id",
        ["", "abcd", "abcdef", "ab/c1", "../..", "a b c", "1mU3g\n", "1mU3g/2053"],
    )
    def test_rejects_non_chart_id_shapes(self, bad_id: str):
        with pytest.raises(ValueError, match="not a Datawrapper chart id"):
            datawrapper_live_data_url(bad_id)


class TestParseHttpLastModified:
    def test_parses_rfc7231_to_aware_utc(self):
        parsed = parse_http_last_modified("Tue, 25 Aug 2026 19:00:51 GMT")
        assert parsed == datetime(2026, 8, 25, 19, 0, 51, tzinfo=UTC)
        assert parsed is not None
        assert parsed.tzinfo is not None

    def test_malformed_returns_none(self):
        assert parse_http_last_modified("not a date") is None
        assert parse_http_last_modified("") is None

    def test_timezoneless_value_is_stamped_utc(self):
        # RFC 5322's "-0000" means "no timezone information", and
        # `parsedate_to_datetime` returns a NAIVE datetime for it. The freshness
        # guard subtracts this from an aware `now`, which raises TypeError on a
        # naive operand — so the UTC stamp is what keeps an odd CDN header
        # producing a `stale_data` verdict instead of a crashed hop.
        parsed = parse_http_last_modified("Tue, 25 Aug 2026 19:00:51 -0000")
        assert parsed == datetime(2026, 8, 25, 19, 0, 51, tzinfo=UTC)
        assert parsed is not None
        assert parsed.tzinfo is not None


class TestDecodeTextBody:
    """Raw bodies decode by the response's OWN self-description, then get scored.

    The retired blanket `body.decode("utf-8", errors="replace")` turned a UTF-16 or
    Windows-1252 body into mojibake that still type-checked as text, and the
    resolution-source fetcher then rendered `0<?>.<?>4<?>2<?>` to a forecaster as
    primary grading evidence with `status="success"`.
    """

    def test_a_bom_wins_over_everything_and_is_consumed(self):
        """Excel exports CSV as BOM'd UTF-16, which is exactly the poll-table shape the
        Datawrapper hop fetches. The BOM must decide the codec AND not survive into the text."""
        body = codecs.BOM_UTF16_LE + "date,value\n2026-08-01,0.42\n".encode("utf-16-le")

        text, ratio = decode_text_body(body, "text/csv")

        assert text == "date,value\n2026-08-01,0.42\n"
        assert ratio == 0.0

    def test_a_utf8_bom_is_stripped_rather_than_read_as_content(self):
        body = codecs.BOM_UTF8 + b"date,value\n2026-08-01,0.42\n"

        text, ratio = decode_text_body(body, "text/csv")

        assert text.startswith("date,value")
        assert ratio == 0.0

    def test_a_declared_charset_is_honoured(self):
        """`charset=` was parsed for ROUTING and then ignored for decoding, so a Windows-1252
        pollster name lost its apostrophe to a replacement char."""
        body = "Pollster,Approve\nO’Brien Research,44\n".encode("windows-1252")  # noqa: RUF001  # cp1252 byte 0x92 is what this test decodes

        text, ratio = decode_text_body(body, "text/csv; charset=windows-1252")

        assert "O’Brien Research" in text  # noqa: RUF001  # cp1252 byte 0x92 is what this test decodes
        assert ratio == 0.0

    def test_a_quoted_charset_is_honoured(self):
        body = "café,1\n".encode("latin-1")

        text, _ratio = decode_text_body(body, 'text/csv; charset="latin-1"')

        assert text == "café,1\n"

    def test_an_unknown_charset_falls_back_to_utf8_rather_than_raising(self):
        """A typo'd charset label is not a reason to lose the body."""
        text, ratio = decode_text_body(b"date,value\n", "text/csv; charset=utf-8000")

        assert text == "date,value\n"
        assert ratio == 0.0

    def test_a_bomless_utf16_body_scores_as_undecodable(self):
        """The half a replacement-char count alone cannot see: every second byte of BOM-less
        UTF-16 ASCII is a NUL, which is a VALID UTF-8 character, so the naive decode produced
        zero replacement chars and passed for text."""
        body = "date,value\n2026-08-01,0.42\n".encode("utf-16-le")

        _text, ratio = decode_text_body(body, "text/csv")

        assert ratio > MAX_UNDECODABLE_CHAR_RATIO

    def test_a_utf16_bom_read_as_utf8_would_score_as_undecodable(self):
        """The other shape: had the BOM sniff not fired, the ratio is the backstop."""
        body = codecs.BOM_UTF16_LE + "date,value\n".encode("utf-16-le")

        assert undecodable_char_ratio(body.decode("utf-8", errors="replace")) > MAX_UNDECODABLE_CHAR_RATIO

    def test_text_with_a_single_bad_byte_stays_usable(self):
        """The threshold has to sit far above real text's dirt: a page carrying one mis-encoded
        smart quote is evidence we want, not a body we failed to decode. This is why the bound is
        0.10 rather than something tight — a 37-char line with one bad byte already scores 0.027."""
        body = "Pollster,Approve\nO’Brien Research,44\n".encode("windows-1252")  # noqa: RUF001  # cp1252 byte 0x92 is what this test decodes

        _text, ratio = decode_text_body(body, "text/csv")

        assert 0.0 < ratio <= MAX_UNDECODABLE_CHAR_RATIO

    def test_an_empty_body_scores_zero_rather_than_dividing_by_zero(self):
        assert decode_text_body(b"", "text/csv") == ("", 0.0)
