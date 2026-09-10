"""Tests for the SEC EDGAR client (`metaculus_bot/research/sec_edgar.py`).

Every request runs through the real aiohttp session the client builds, against a loopback
`aiohttp.test_utils.TestServer` that serves the recorded fixtures under `tests/data/sec_edgar/`
and records what arrived (path, query, headers, arrival time). Loopback is the one host the
suite's egress guard allows, so the fair-access header, the byte cap and the request spacing are
all asserted on the wire rather than on a fake.

Fixtures were recorded from the live endpoints on 2026-09-09 and trimmed to a few entries each.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from metaculus_bot.constants import SEC_EDGAR_CONTACT_EMAIL_ENV
from metaculus_bot.research import sec_edgar
from metaculus_bot.research.http_fetch import reset_host_semaphores
from metaculus_bot.research.sec_edgar import (
    RequestSpacer,
    SecEdgarContactUnsetError,
    SecEdgarError,
    company_facts,
    company_submissions,
    edgar_session,
    filing_document,
    filing_document_url,
    frame,
    full_text_search,
    pad_cik,
    sec_edgar_user_agent,
    ticker_to_cik,
)

_FIXTURES = Path(__file__).parent / "data" / "sec_edgar"
_CONTACT = "bot-operator@example.com"
_EXPECTED_UA = f"metaculus-bot {_CONTACT}"

_TEN_K_HTML = b"<html><body><h1>UBER TECHNOLOGIES, INC. FORM 10-K</h1><p>Revenue $50.0 billion.</p></body></html>"


def _fixture(name: str) -> Any:
    return json.loads((_FIXTURES / name).read_text())


@dataclass
class RecordedRequest:
    path: str
    query: dict[str, str]
    user_agent: str
    accept_encoding: str
    at: float


@dataclass
class EdgarStub:
    """The loopback stand-in for www.sec.gov, data.sec.gov and efts.sec.gov at once."""

    requests: list[RecordedRequest] = field(default_factory=list)
    status_overrides: dict[str, int] = field(default_factory=dict)

    async def handle(self, request: web.Request) -> web.Response:
        await asyncio.sleep(0)
        self.requests.append(
            RecordedRequest(
                path=request.path,
                query=dict(request.query),
                user_agent=request.headers.get("User-Agent", ""),
                accept_encoding=request.headers.get("Accept-Encoding", ""),
                at=time.monotonic(),
            )
        )
        if request.path in self.status_overrides:
            return web.Response(status=self.status_overrides[request.path], text="Request Rate Threshold Exceeded")
        if request.path == "/files/company_tickers.json":
            return web.json_response(_fixture("company_tickers.json"))
        if request.path == "/submissions/CIK0000320193.json":
            return web.json_response(_fixture("submissions_CIK0000320193.json"))
        if request.path == "/api/xbrl/companyfacts/CIK0000019617.json":
            return web.json_response(_fixture("companyfacts_CIK0000019617.json"))
        if request.path == "/api/xbrl/frames/us-gaap/Revenues/USD/CY2025.json":
            return web.json_response(_fixture("frames_us-gaap_Revenues_USD_CY2025.json"))
        if request.path == "/LATEST/search-index":
            return web.json_response(_fixture("full_text_search_delivery_hero.json"))
        if request.path.startswith("/Archives/edgar/data/"):
            return web.Response(body=_TEN_K_HTML, content_type="text/html")
        return web.Response(status=404, text="not found")


@pytest.fixture
async def edgar(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[EdgarStub]:
    """A running loopback EDGAR with the client pointed at it, a contact email set, and fast spacing."""
    stub = EdgarStub()
    app = web.Application()
    app.router.add_get("/{tail:.*}", stub.handle)
    server = TestServer(app)
    await server.start_server()
    base = str(server.make_url("")).rstrip("/")
    monkeypatch.setattr(sec_edgar, "WWW_BASE_URL", base)
    monkeypatch.setattr(sec_edgar, "DATA_BASE_URL", base)
    monkeypatch.setattr(sec_edgar, "FULL_TEXT_SEARCH_BASE_URL", base)
    monkeypatch.setattr(sec_edgar, "SEC_EDGAR_MAX_REQUESTS_PER_SECOND", 500.0)
    monkeypatch.setenv(SEC_EDGAR_CONTACT_EMAIL_ENV, _CONTACT)
    sec_edgar.reset_request_spacer()
    reset_host_semaphores()
    try:
        yield stub
    finally:
        await server.close()
        sec_edgar.reset_request_spacer()
        reset_host_semaphores()


class TestPadCik:
    @pytest.mark.parametrize("cik", [320193, "320193", "0000320193", " 320193 "])
    def test_pads_to_ten_digits(self, cik: int | str):
        assert pad_cik(cik) == "0000320193"

    @pytest.mark.parametrize("bad", ["AAPL", "", "12345678901", "32-0193"])
    def test_rejects_non_ciks(self, bad: str):
        with pytest.raises(ValueError, match="not a CIK"):
            pad_cik(bad)

    def test_document_url_uses_the_unpadded_cik_and_dashless_accession(self):
        assert filing_document_url(1543151, "0001543151-26-000015", "uber-20251231.htm") == (
            "https://www.sec.gov/Archives/edgar/data/1543151/000154315126000015/uber-20251231.htm"
        )


class TestFairAccessUserAgent:
    def test_user_agent_is_identity_then_contact(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(SEC_EDGAR_CONTACT_EMAIL_ENV, _CONTACT)
        assert sec_edgar_user_agent() == _EXPECTED_UA

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_unset_contact_fails_shut(self, monkeypatch: pytest.MonkeyPatch, raw: str | None):
        if raw is None:
            monkeypatch.delenv(SEC_EDGAR_CONTACT_EMAIL_ENV, raising=False)
        else:
            monkeypatch.setenv(SEC_EDGAR_CONTACT_EMAIL_ENV, raw)
        with pytest.raises(SecEdgarContactUnsetError, match=SEC_EDGAR_CONTACT_EMAIL_ENV):
            sec_edgar_user_agent()

    async def test_session_refuses_to_open_without_a_contact(self, edgar: EdgarStub, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(SEC_EDGAR_CONTACT_EMAIL_ENV)
        with pytest.raises(SecEdgarContactUnsetError):
            async with edgar_session():
                pytest.fail("the session opened without a contact email")
        assert edgar.requests == []

    async def test_every_request_carries_the_fair_access_headers(self, edgar: EdgarStub):
        async with edgar_session() as session:
            submissions = await company_submissions(session, "aapl")
            await company_facts(session, 19617)
            await frame(session, "Revenues", "USD", "CY2025")
            await full_text_search(session, '"Delivery Hero"', forms=["8-K"])
            await filing_document(session, submissions.filings_of_form("10-K")[0].primary_document_url)

        assert len(edgar.requests) == 6, [request.path for request in edgar.requests]
        assert {request.user_agent for request in edgar.requests} == {_EXPECTED_UA}
        assert {request.accept_encoding for request in edgar.requests} == {"gzip, deflate"}


class TestRequestSpacer:
    async def test_concurrent_waiters_are_spaced_by_the_interval(self):
        spacer = RequestSpacer(min_interval_s=0.05)
        starts: list[float] = []

        async def one() -> None:
            await spacer.wait()
            starts.append(time.monotonic())

        await asyncio.gather(*(one() for _ in range(5)))

        gaps = [later - earlier for earlier, later in pairwise(sorted(starts))]
        assert len(gaps) == 4
        assert min(gaps) >= 0.045, gaps

    async def test_the_shared_spacer_paces_at_the_requests_per_second_constant(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sec_edgar, "SEC_EDGAR_MAX_REQUESTS_PER_SECOND", 10.0)
        sec_edgar.reset_request_spacer()
        try:
            spacer = sec_edgar.request_spacer()
            assert sec_edgar.request_spacer() is spacer, "one spacer per loop, shared by every caller"
            await spacer.wait()
            first = time.monotonic()
            await spacer.wait()
            assert time.monotonic() - first >= 0.095
        finally:
            sec_edgar.reset_request_spacer()

    async def test_requests_on_the_wire_are_spaced(self, edgar: EdgarStub, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sec_edgar, "SEC_EDGAR_MAX_REQUESTS_PER_SECOND", 20.0)
        sec_edgar.reset_request_spacer()
        url = filing_document_url(1543151, "0001543151-26-000015", "uber-20251231.htm")
        async with edgar_session() as session:
            await asyncio.gather(*(filing_document(session, url) for _ in range(4)))

        arrivals = sorted(request.at for request in edgar.requests)
        gaps = [later - earlier for earlier, later in pairwise(arrivals)]
        assert len(gaps) == 3
        assert min(gaps) >= 0.045, gaps


class TestTickerMap:
    async def test_looks_up_a_ticker_case_insensitively(self, edgar: EdgarStub):
        async with edgar_session() as session:
            assert await ticker_to_cik(session, "aapl") == 320193
            assert await ticker_to_cik(session, "BRK-B") == 1067983
        assert [request.path for request in edgar.requests] == ["/files/company_tickers.json"] * 2

    async def test_unknown_ticker_raises(self, edgar: EdgarStub):
        async with edgar_session() as session:
            with pytest.raises(SecEdgarError, match="not in SEC's company_tickers"):
                await ticker_to_cik(session, "NOPE")


class TestCompanySubmissions:
    async def test_parses_the_recent_filings_block(self, edgar: EdgarStub):
        async with edgar_session() as session:
            submissions = await company_submissions(session, 320193)

        assert submissions.cik == 320193
        assert submissions.name == "Apple Inc."
        assert submissions.tickers == ("AAPL",)
        assert submissions.exchanges == ("Nasdaq",)
        assert submissions.fiscal_year_end == "0926"
        assert [filing.form for filing in submissions.filings] == ["4", "10-K", "10-Q"]

        ten_k = submissions.filings_of_form("10-K")[0]
        assert ten_k.accession_number == "0000320193-25-000079"
        assert ten_k.filing_date == date(2025, 10, 31)
        assert ten_k.report_date == date(2025, 9, 27)
        assert ten_k.primary_document == "aapl-20250927.htm"
        assert ten_k.primary_document_description == "10-K"
        assert ten_k.is_inline_xbrl is True
        assert ten_k.primary_document_url.endswith("/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm")

    async def test_an_empty_report_date_is_none(self, edgar: EdgarStub):
        async with edgar_session() as session:
            submissions = await company_submissions(session, "0000320193")
        form_4 = submissions.filings[0]
        assert form_4.form == "4"
        assert form_4.report_date == date(2026, 9, 1)
        assert form_4.is_inline_xbrl is False

    async def test_a_ticker_resolves_through_the_ticker_map_first(self, edgar: EdgarStub):
        async with edgar_session() as session:
            submissions = await company_submissions(session, "AAPL")
        assert submissions.cik == 320193
        assert [request.path for request in edgar.requests] == [
            "/files/company_tickers.json",
            "/submissions/CIK0000320193.json",
        ]

    async def test_a_non_200_raises_with_the_status(self, edgar: EdgarStub):
        edgar.status_overrides["/submissions/CIK0000320193.json"] = 403
        async with edgar_session() as session:
            with pytest.raises(SecEdgarError, match="HTTP 403") as excinfo:
                await company_submissions(session, 320193)
        assert "Request Rate Threshold Exceeded" in str(excinfo.value)


class TestCompanyFacts:
    async def test_values_for_a_concept_and_unit_oldest_first(self, edgar: EdgarStub):
        async with edgar_session() as session:
            facts = await company_facts(session, "19617")

        assert facts.cik == 19617
        assert facts.entity_name == "JPMORGAN CHASE & CO"
        revenues = facts.values("Revenues", "USD")
        assert [fact.end for fact in revenues] == [date(2024, 12, 31), date(2024, 12, 31), date(2025, 12, 31)]
        first_print, restated, latest = revenues
        assert (first_print.filed, first_print.frame) == (date(2025, 2, 14), None)
        assert (restated.filed, restated.frame) == (date(2026, 2, 13), "CY2024")
        assert first_print.value == restated.value == 177_556_000_000
        assert latest.value == 182_447_000_000
        assert latest.start == date(2025, 1, 1)
        assert latest.fiscal_year == 2025
        assert latest.fiscal_period == "FY"
        assert latest.form == "10-K"
        assert latest.filed == date(2026, 2, 13)
        assert latest.accession_number == "0001628280-26-008131"
        assert latest.frame == "CY2025"

    async def test_an_instant_concept_in_another_taxonomy(self, edgar: EdgarStub):
        async with edgar_session() as session:
            facts = await company_facts(session, 19617)
        shares = facts.values("EntityCommonStockSharesOutstanding", "shares", taxonomy="dei")
        assert [fact.value for fact in shares] == [2_679_511_418, 2_658_186_195]
        assert shares[-1].start is None
        assert shares[-1].frame == "CY2026Q2I"

    async def test_a_concept_the_filer_never_reported_is_empty(self, edgar: EdgarStub):
        async with edgar_session() as session:
            facts = await company_facts(session, 19617)
        assert facts.values("Revenues", "EUR") == ()
        assert facts.values("NoSuchConcept", "USD") == ()
        assert facts.values("Revenues", "USD", taxonomy="ifrs-full") == ()


class TestFrame:
    async def test_parses_the_cross_filer_frame(self, edgar: EdgarStub):
        async with edgar_session() as session:
            result = await frame(session, "Revenues", "USD", "CY2025")

        assert (result.taxonomy, result.concept, result.unit, result.period) == ("us-gaap", "Revenues", "USD", "CY2025")
        assert result.label == "Revenues"
        assert [value.cik for value in result.values] == [2098, 2969]
        first = result.values[0]
        assert first.entity_name == "ACME UNITED CORP"
        assert first.location == "US-CT"
        assert first.value == 196_541_816
        assert (first.start, first.end) == (date(2025, 1, 1), date(2025, 12, 31))
        assert first.accession_number == "0001193125-26-102079"
        assert edgar.requests[0].path == "/api/xbrl/frames/us-gaap/Revenues/USD/CY2025.json"

    @pytest.mark.parametrize("bad", ["2025", "CY25", "CY2025Q5", "CY2025QI", "FY2025"])
    async def test_rejects_periods_outside_the_documented_shapes(self, edgar: EdgarStub, bad: str):
        async with edgar_session() as session:
            with pytest.raises(ValueError, match="frames period"):
                await frame(session, "Revenues", "USD", bad)
        assert edgar.requests == []


class TestFullTextSearch:
    async def test_sends_the_documented_query_parameters(self, edgar: EdgarStub):
        async with edgar_session() as session:
            await full_text_search(
                session,
                '"Delivery Hero"',
                date_from=date(2026, 1, 1),
                date_to=date(2026, 9, 1),
                forms=["8-K", "10-K"],
            )
        assert edgar.requests[0].path == "/LATEST/search-index"
        assert edgar.requests[0].query == {
            "q": '"Delivery Hero"',
            "dateRange": "custom",
            "startdt": "2026-01-01",
            "enddt": "2026-09-01",
            "forms": "8-K,10-K",
        }

    async def test_omits_the_date_and_form_filters_when_not_asked(self, edgar: EdgarStub):
        async with edgar_session() as session:
            await full_text_search(session, "spin-off")
        assert edgar.requests[0].query == {"q": "spin-off"}

    async def test_a_single_bound_still_switches_to_a_custom_range(self, edgar: EdgarStub):
        async with edgar_session() as session:
            await full_text_search(session, "spin-off", date_from=date(2026, 6, 1))
        assert edgar.requests[0].query == {"q": "spin-off", "dateRange": "custom", "startdt": "2026-06-01"}

    async def test_parses_hits_into_fetchable_documents(self, edgar: EdgarStub):
        async with edgar_session() as session:
            results = await full_text_search(session, '"Delivery Hero"')

        assert results.total == 15
        assert len(results.hits) == 3
        exhibit = results.hits[0]
        assert exhibit.accession_number == "0001552781-26-000382"
        assert exhibit.filename == "e26302_ex2-1.htm"
        assert exhibit.ciks == (1543151,)
        assert exhibit.display_names == ("Uber Technologies, Inc  (UBER)  (CIK 0001543151)",)
        assert exhibit.form == "8-K"
        assert exhibit.file_date == date(2026, 7, 16)
        assert exhibit.period_ending == date(2026, 7, 16)
        assert exhibit.file_type == "EX-2.1"
        assert exhibit.file_description is None
        assert exhibit.document_url.endswith("/Archives/edgar/data/1543151/000155278126000382/e26302_ex2-1.htm")
        assert {hit.form for hit in results.hits} == {"8-K", "10-K"}


class TestFilingDocument:
    async def test_fetches_the_body_and_content_type(self, edgar: EdgarStub):
        url = filing_document_url(1543151, "0001543151-26-000015", "uber-20251231.htm")
        async with edgar_session() as session:
            document = await filing_document(session, url)
        assert document.body == _TEN_K_HTML
        assert document.content_type.startswith("text/html")
        assert document.url == url
        assert edgar.requests[0].path == "/Archives/edgar/data/1543151/000154315126000015/uber-20251231.htm"

    async def test_a_body_over_the_cap_is_refused(self, edgar: EdgarStub, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sec_edgar, "SEC_EDGAR_MAX_RESPONSE_BYTES", len(_TEN_K_HTML) - 1)
        url = filing_document_url(1543151, "0001543151-26-000015", "uber-20251231.htm")
        async with edgar_session() as session:
            with pytest.raises(SecEdgarError, match="exceeds"):
                await filing_document(session, url)

    async def test_a_body_at_the_cap_is_served(self, edgar: EdgarStub, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sec_edgar, "SEC_EDGAR_MAX_RESPONSE_BYTES", len(_TEN_K_HTML))
        url = filing_document_url(1543151, "0001543151-26-000015", "uber-20251231.htm")
        async with edgar_session() as session:
            document = await filing_document(session, url)
        assert document.body == _TEN_K_HTML

    async def test_refuses_a_url_off_the_edgar_hosts_before_dialing(self, edgar: EdgarStub):
        async with edgar_session() as session:
            with pytest.raises(ValueError, match="not on an EDGAR host"):
                await filing_document(session, "https://www.trueup.io/layoffs")
        assert edgar.requests == []

    async def test_a_403_surfaces_as_an_error_not_a_retry(self, edgar: EdgarStub):
        path = "/Archives/edgar/data/1543151/000154315126000015/uber-20251231.htm"
        edgar.status_overrides[path] = 403
        url = filing_document_url(1543151, "0001543151-26-000015", "uber-20251231.htm")
        async with edgar_session() as session:
            with pytest.raises(SecEdgarError, match="HTTP 403"):
                await filing_document(session, url)
        assert [request.path for request in edgar.requests] == [path]
