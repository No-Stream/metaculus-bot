"""Provider factory, env-flag gating, and the SSRF guard.

The two outermost layers of the Tier-1 resolution-source fetcher: whether the provider is
built at all (env flag plus the benchmarking leakage guard), and what the SSRF guard admits
before any body is read. Split out of the old single ``test_resolution_source_provider.py``;
see ``test_resolution_source_helpers.py`` for the layer map.
"""

from __future__ import annotations

import ipaddress
import socket as _socket
from typing import Any

from metaculus_bot.research import resolution_source
from metaculus_bot.research.http_fetch import FilteringResolver
from metaculus_bot.research.provider_diagnostics import _counts_suffix, pop_provider_detail
from metaculus_bot.research.resolution_source import (
    FetchResult,
    _fetch_one,
    _fetch_result_sources,
    resolution_source_provider,
)
from tests.resolution_source_fakes import (
    FakeResponse,
    FakeSession,
    _mock_question,
)
from tests.test_document_text import build_text_pdf


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

    async def test_the_questions_criteria_reach_a_cited_documents_passage_selection(self, monkeypatch):
        """Nothing else pins that the question's own text gets as far as BM25.

        `fetch_resolution_sources` defaults `query=""`, every PDF test drives `_fetch_one`
        with a hand-built FetchContext, and no test references `_document_query` — so a
        dropped or misspelled kwarg here type-checks, passes the whole suite, and renders
        every cited document's header and outline with the "no passage matched" sentence and
        none of the resolving figures. The vocabulary below appears ONLY in the resolution
        criteria: not in the question title, not in the document's front matter, and not in
        the decoy pages, so this fails if the query is empty OR title-only.
        """
        monkeypatch.setenv("RESOLUTION_SOURCE_ENABLED", "true")
        pages = [
            ["Annual Surveillance Report", "Contents: methods, tables, appendix, acknowledgements"],
            *[[f"Chapter {n}: vector control programs and municipal drainage works."] for n in range(1, 7)],
            ["Laboratory-confirmed cyclosporiasis hospitalizations reached 922 during the reporting period."],
        ]
        session = FakeSession(
            {
                "https://cdc.example.com/report.pdf": FakeResponse(
                    200, body=build_text_pdf(pages), content_type="application/pdf"
                )
            }
        )
        monkeypatch.setattr(resolution_source, "_get_session", lambda: session)
        q = _mock_question(
            resolution_criteria=(
                "Resolves per the laboratory-confirmed cyclosporiasis hospitalizations "
                "reported at https://cdc.example.com/report.pdf"
            )
        )

        out = await resolution_source_provider(is_benchmarking=False)(q)

        assert "922" in out
        assert "No passage in this document matched the query" not in out
        assert "municipal drainage works" not in out, "non-vacuity: the decoys are ranked out, not all included"

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

        detail = pop_provider_detail(q.id_of_question, "resolution_source")
        assert detail["sources"] == {"www.bls.gov": "ok"}
        # Every rung count is zero and STAYS in the detail: a zero renders nothing in the
        # diagnostics line (so a healthy provider's line is byte-identical to what it was
        # before the ladder existed) while the archive keeps it, which is what makes "the
        # rung ran and never fired" distinguishable from "this record predates the rung".
        assert detail["counts"] == {
            "meta_refresh_hops": 0,
            "pdf_documents_read": 0,
            "rung_budget_skips": 0,
            "pdf_contention_skips": 0,
        }
        assert _counts_suffix(detail) == ""

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
