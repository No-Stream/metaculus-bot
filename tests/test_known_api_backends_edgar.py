"""The SEC EDGAR known-API backend: declines without a contact, else reads the JSON API / filing.

The load-bearing behaviour is the decline: with ``SEC_EDGAR_CONTACT_EMAIL`` unset the backend
returns ``None`` so the fetch ladder falls through to the ordinary page fetch. Every EDGAR call is
faked (the session, the submissions read and the filing read are patched on ``sec_edgar``).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date

import pytest

from metaculus_bot.constants import SEC_EDGAR_CONTACT_EMAIL_ENV
from metaculus_bot.research import sec_edgar
from metaculus_bot.research.known_api import backends
from metaculus_bot.research.known_api.translate import KnownApiCall

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _fake_session(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def _session():
        yield object()

    monkeypatch.setattr(sec_edgar, "edgar_session", _session)


def _submissions_call() -> KnownApiCall:
    return KnownApiCall(
        kind="edgar",
        edgar_kind="company_submissions",
        edgar_arg="0000320193",
        canonical_url="https://www.sec.gov/cgi-bin/browse-edgar?CIK=0000320193",
    )


def _document_call() -> KnownApiCall:
    url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl.htm"
    return KnownApiCall(kind="edgar", edgar_kind="filing_document", edgar_url=url, canonical_url=url)


class TestDecline:
    async def test_declines_when_contact_unset(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv(SEC_EDGAR_CONTACT_EMAIL_ENV, raising=False)

        assert await backends.edgar(_submissions_call()) is None


class TestCompanySubmissions:
    async def test_renders_a_filings_table(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(SEC_EDGAR_CONTACT_EMAIL_ENV, "bot ops@example.com")
        filing = sec_edgar.Filing(
            cik=320193,
            accession_number="0000320193-24-000123",
            form="10-K",
            filing_date=date(2024, 11, 1),
            report_date=date(2024, 9, 28),
            primary_document="aapl-20240928.htm",
            primary_document_description="10-K",
            is_inline_xbrl=True,
        )
        submissions = sec_edgar.CompanySubmissions(
            cik=320193,
            name="Apple Inc.",
            tickers=("AAPL",),
            exchanges=("Nasdaq",),
            fiscal_year_end="0928",
            filings=(filing,),
        )

        async def _fake(session, cik_or_ticker):
            return submissions

        monkeypatch.setattr(sec_edgar, "company_submissions", _fake)

        result = await backends.edgar(_submissions_call())

        assert result is not None
        assert result.status == "ok"
        assert "Apple Inc." in result.content_markdown
        assert "10-K" in result.content_markdown
        assert filing.primary_document_url in result.content_markdown


class TestFilingDocument:
    async def test_html_body_is_extracted(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(SEC_EDGAR_CONTACT_EMAIL_ENV, "bot ops@example.com")
        html = (
            b"<html><body><article><h1>8-K</h1>"
            b"<p>Delivery Hero received BaFin approval for the acquisition on 2026-01-05.</p>"
            b"</article></body></html>"
        )
        doc = sec_edgar.FilingDocument(url=_document_call().edgar_url, content_type="text/html", body=html)

        async def _fake(session, url):
            return doc

        monkeypatch.setattr(sec_edgar, "filing_document", _fake)

        result = await backends.edgar(_document_call())

        assert result is not None
        assert result.status == "ok"
        assert "BaFin approval" in result.content_markdown

    async def test_edgar_error_is_reported(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(SEC_EDGAR_CONTACT_EMAIL_ENV, "bot ops@example.com")

        async def _raise(session, url):
            raise sec_edgar.SecEdgarError("EDGAR answered HTTP 404 for the filing")

        monkeypatch.setattr(sec_edgar, "filing_document", _raise)

        result = await backends.edgar(_document_call())

        assert result is not None
        assert result.status == "error"
        assert "404" in result.content_markdown
