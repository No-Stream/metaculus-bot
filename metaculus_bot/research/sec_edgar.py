"""SEC EDGAR through its public JSON APIs, under the fair-access policy, instead of scraping sec.gov pages.

sec.gov answers the bot's page fetches 403: 12 blocked events over 5 questions in the archived gap-fill
v2 transcripts (2026-09-09 fetch-gap inventory), because EDGAR's fair-access policy wants a declared
User-Agent naming the caller and a contact address rather than a browser fingerprint, and the same
facts are served as JSON by data.sec.gov anyway. This module is the client: filing history by CIK or
ticker (submissions), XBRL company facts and cross-filer frames, EDGAR full-text search, and the
primary document of a filing. Standalone; no research ladder calls it yet.

Fair access as implemented here: every request carries ``User-Agent: <identity> <contact email>``
with the email read from ``SEC_EDGAR_CONTACT_EMAIL`` when the session opens, and no session opens
without it, so an anonymous User-Agent is never sent. Request starts are spaced process-wide to
``SEC_EDGAR_MAX_REQUESTS_PER_SECOND`` across all three EDGAR hosts and serialized per host through
the politeness gate in ``http_fetch``; bodies stream through ``read_body_capped`` under
``SEC_EDGAR_MAX_RESPONSE_BYTES``. No retries: a 403 here is a policy verdict a retry cannot change
and a 429 is the ceiling itself. The endpoint shapes, the 10 requests per second ceiling and the
User-Agent format were read from SEC's own documentation on 2026-09-09; docs/research.md
"SEC EDGAR client" carries the receipts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urlparse

import aiohttp

from metaculus_bot.constants import (
    RESOLUTION_SOURCE_HTTP_TIMEOUT,
    SEC_EDGAR_CONTACT_EMAIL_ENV,
    SEC_EDGAR_MAX_REQUESTS_PER_SECOND,
    SEC_EDGAR_MAX_RESPONSE_BYTES,
    SEC_EDGAR_USER_AGENT_TEMPLATE,
)
from metaculus_bot.research.http_fetch import (
    REDIRECT_STATUSES,
    build_session,
    host_semaphores,
    read_body_capped,
    read_body_snippet,
    semaphore_for_host,
)

logger = logging.getLogger(__name__)

# Read at call time by every URL builder, so the tests can point the client at a loopback server.
WWW_BASE_URL = "https://www.sec.gov"
DATA_BASE_URL = "https://data.sec.gov"
FULL_TEXT_SEARCH_BASE_URL = "https://efts.sec.gov"

# SEC's fair-access sample headers name exactly this pair; aiohttp's own default would add br and zstd.
_ACCEPT_ENCODING = "gzip, deflate"

_FRAME_PERIOD_RE = re.compile(r"CY\d{4}(?:Q[1-4]I?)?\Z")

# EDGAR full-text search indexes filings from 2001 on; the lower bound a one-sided range is completed with.
FULL_TEXT_SEARCH_EARLIEST_DATE = date(2001, 1, 1)


class SecEdgarError(RuntimeError):
    """An EDGAR request that did not yield what was asked for: a non-200, an oversized body, an unknown ticker."""


class SecEdgarContactUnsetError(SecEdgarError):
    """``SEC_EDGAR_CONTACT_EMAIL`` is unset, so no fair-access User-Agent can be formed and nothing is dialed."""


def sec_edgar_user_agent() -> str:
    """The fair-access User-Agent, read from the environment at call time; raises when the contact is unset."""
    contact_email = os.getenv(SEC_EDGAR_CONTACT_EMAIL_ENV, "").strip()
    if not contact_email:
        raise SecEdgarContactUnsetError(
            f"{SEC_EDGAR_CONTACT_EMAIL_ENV} is unset. SEC's fair-access policy requires a User-Agent that names "
            "a contact address, so the EDGAR client refuses to send any request without one."
        )
    return SEC_EDGAR_USER_AGENT_TEMPLATE.format(contact_email=contact_email)


def pad_cik(cik: int | str) -> str:
    """The ten-digit, zero-padded Central Index Key every data.sec.gov path wants."""
    digits = str(cik).strip()
    if not digits.isdigit() or len(digits) > 10:
        raise ValueError(f"not a CIK: {cik!r}")
    return digits.zfill(10)


def filing_document_url(cik: int, accession_number: str, filename: str) -> str:
    """The Archives URL of one document in a filing: the unpadded CIK, the dashless accession number, the filename."""
    return f"{WWW_BASE_URL}/Archives/edgar/data/{cik}/{accession_number.replace('-', '')}/{filename}"


# ---------------------------------------------------------------------------
# Politeness: request spacing under the fair-access ceiling
# ---------------------------------------------------------------------------


class RequestSpacer:
    """Spaces request starts at least ``min_interval_s`` apart for every caller sharing the instance."""

    def __init__(self, min_interval_s: float) -> None:
        self._min_interval_s = min_interval_s
        self._lock = asyncio.Lock()
        self._next_start = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if now < self._next_start:
                await asyncio.sleep(self._next_start - now)
                now = time.monotonic()
            self._next_start = now + self._min_interval_s


# Loop-scoped like http_fetch's gates: an asyncio.Lock binds to the first loop that contends on it.
_SPACER: RequestSpacer | None = None
_SPACER_LOOP: asyncio.AbstractEventLoop | None = None


def request_spacer() -> RequestSpacer:
    """The process-wide spacer for the running event loop, shared by every EDGAR host and caller."""
    global _SPACER, _SPACER_LOOP  # noqa: PLW0603  # module-level cache of the loop's spacer
    loop = asyncio.get_running_loop()
    if _SPACER is None or loop is not _SPACER_LOOP:
        _SPACER = RequestSpacer(1.0 / SEC_EDGAR_MAX_REQUESTS_PER_SECOND)
        _SPACER_LOOP = loop
    return _SPACER


def reset_request_spacer() -> None:
    """Drop the cached spacer. For tests, so one test's spacing cannot delay another's."""
    global _SPACER, _SPACER_LOOP  # noqa: PLW0603  # paired with request_spacer's cache
    _SPACER = None
    _SPACER_LOOP = None


# ---------------------------------------------------------------------------
# Session and the one bounded GET every endpoint goes through
# ---------------------------------------------------------------------------


@asynccontextmanager
async def edgar_session() -> AsyncIterator[aiohttp.ClientSession]:
    """An aiohttp session carrying the fair-access headers on every request.

    Raises :class:`SecEdgarContactUnsetError` before anything is opened when the contact email is
    unset, which is the fail-shut half of the fair-access rule.
    """
    headers = {"User-Agent": sec_edgar_user_agent(), "Accept-Encoding": _ACCEPT_ENCODING}
    async with build_session(timeout_s=RESOLUTION_SOURCE_HTTP_TIMEOUT, headers=headers) as session:
        yield session


async def _get_bytes(
    session: aiohttp.ClientSession, url: str, params: dict[str, str] | None = None
) -> tuple[bytes, str, str]:
    """One spaced, host-gated, byte-capped GET: ``(body, content_type, final_url)``; raises on anything else.

    Redirects are refused rather than followed: every URL here is a documented endpoint or an
    Archives document, so a 3xx is unexpected, and following it would dial a hop the spacer, the host
    gate and the EDGAR host check never saw while still carrying the fair-access User-Agent.
    """
    async with semaphore_for_host(url, host_semaphores()):
        await request_spacer().wait()
        async with session.get(url, params=params, allow_redirects=False) as resp:
            if resp.status in REDIRECT_STATUSES:
                location = resp.headers.get("Location")
                raise SecEdgarError(
                    f"EDGAR redirected {url} to {location!r}; this client dials documented endpoints only"
                )
            if resp.status != 200:
                snippet = await read_body_snippet(resp)
                raise SecEdgarError(f"EDGAR answered HTTP {resp.status} for {url}: {snippet!r}")
            body = await read_body_capped(resp, max_bytes=SEC_EDGAR_MAX_RESPONSE_BYTES, label="sec_edgar")
            if body is None:
                raise SecEdgarError(f"EDGAR body for {url} exceeds {SEC_EDGAR_MAX_RESPONSE_BYTES} bytes")
            return body, resp.headers.get("Content-Type") or "", str(resp.url)


async def _get_json(session: aiohttp.ClientSession, url: str, params: dict[str, str] | None = None) -> Any:
    body, _content_type, _final_url = await _get_bytes(session, url, params)
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SecEdgarError(f"EDGAR body for {url} is not JSON: {exc}") from exc


# ---------------------------------------------------------------------------
# Ticker map and submissions (filing history)
# ---------------------------------------------------------------------------


async def ticker_to_cik(session: aiohttp.ClientSession, ticker: str) -> int:
    """The CIK behind an exchange ticker, from SEC's own ticker file; raises when the ticker is not listed."""
    wanted = ticker.strip().upper()
    table = await _get_json(session, f"{WWW_BASE_URL}/files/company_tickers.json")
    for entry in table.values():
        if entry["ticker"] == wanted:
            return int(entry["cik_str"])
    raise SecEdgarError(f"ticker {ticker!r} is not in SEC's company_tickers.json")


@dataclass(frozen=True, slots=True)
class Filing:
    """One row of a filer's submission history."""

    cik: int
    accession_number: str
    form: str
    filing_date: date
    report_date: date | None
    primary_document: str
    primary_document_description: str
    is_inline_xbrl: bool

    @property
    def primary_document_url(self) -> str:
        return filing_document_url(self.cik, self.accession_number, self.primary_document)


@dataclass(frozen=True, slots=True)
class CompanySubmissions:
    """A filer's identity and its most recent filings, newest first as SEC serves them."""

    cik: int
    name: str
    tickers: tuple[str, ...]
    exchanges: tuple[str, ...]
    fiscal_year_end: str
    filings: tuple[Filing, ...]

    def filings_of_form(self, *forms: str) -> tuple[Filing, ...]:
        wanted = frozenset(forms)
        return tuple(filing for filing in self.filings if filing.form in wanted)


def _optional_date(value: str) -> date | None:
    return date.fromisoformat(value) if value else None


def _parse_submissions(payload: dict[str, Any]) -> CompanySubmissions:
    cik = int(payload["cik"])
    recent = payload["filings"]["recent"]
    columns = zip(
        recent["accessionNumber"],
        recent["form"],
        recent["filingDate"],
        recent["reportDate"],
        recent["primaryDocument"],
        recent["primaryDocDescription"],
        recent["isInlineXBRL"],
        strict=True,
    )
    filings = tuple(
        Filing(
            cik=cik,
            accession_number=accession_number,
            form=form,
            filing_date=date.fromisoformat(filing_date),
            report_date=_optional_date(report_date),
            primary_document=primary_document,
            primary_document_description=description,
            is_inline_xbrl=bool(inline_xbrl),
        )
        for accession_number, form, filing_date, report_date, primary_document, description, inline_xbrl in columns
    )
    return CompanySubmissions(
        cik=cik,
        name=payload["name"],
        tickers=tuple(payload["tickers"]),
        exchanges=tuple(payload["exchanges"]),
        fiscal_year_end=payload["fiscalYearEnd"],
        filings=filings,
    )


async def company_submissions(session: aiohttp.ClientSession, cik_or_ticker: int | str) -> CompanySubmissions:
    """A filer's submission history by CIK, or by ticker through SEC's ticker file (one extra request)."""
    key = str(cik_or_ticker).strip()
    cik = int(key) if key.isdigit() else await ticker_to_cik(session, key)
    payload = await _get_json(session, f"{DATA_BASE_URL}/submissions/CIK{pad_cik(cik)}.json")
    return _parse_submissions(payload)


# ---------------------------------------------------------------------------
# XBRL: company facts and frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FactValue:
    """One reported XBRL fact: the value, the period it covers, and the filing it came from."""

    value: float
    end: date
    start: date | None
    fiscal_year: int | None
    fiscal_period: str | None
    form: str
    filed: date
    accession_number: str
    frame: str | None


def _fact_value(raw: dict[str, Any]) -> FactValue:
    return FactValue(
        value=raw["val"],
        end=date.fromisoformat(raw["end"]),
        start=_optional_date(raw.get("start", "")),
        fiscal_year=raw["fy"],
        fiscal_period=raw["fp"],
        form=raw["form"],
        filed=date.fromisoformat(raw["filed"]),
        accession_number=raw["accn"],
        frame=raw.get("frame"),
    )


@dataclass(frozen=True, slots=True)
class CompanyFacts:
    """Every XBRL fact a filer has reported, keyed taxonomy -> concept -> unit, as SEC serves it."""

    cik: int
    entity_name: str
    facts: dict[str, Any]

    def values(self, concept: str, unit: str, *, taxonomy: str = "us-gaap") -> tuple[FactValue, ...]:
        """The reported values of one concept in one unit, oldest period first; empty when the filer never reported it."""
        concepts = self.facts.get(taxonomy, {})
        units = concepts.get(concept, {}).get("units", {})
        values = [_fact_value(raw) for raw in units.get(unit, ())]
        return tuple(sorted(values, key=lambda fact: (fact.end, fact.filed)))


async def company_facts(session: aiohttp.ClientSession, cik: int | str) -> CompanyFacts:
    """Every XBRL fact for one filer. Large filers run several megabytes; the byte cap is sized for that."""
    payload = await _get_json(session, f"{DATA_BASE_URL}/api/xbrl/companyfacts/CIK{pad_cik(cik)}.json")
    return CompanyFacts(cik=int(payload["cik"]), entity_name=payload["entityName"], facts=payload["facts"])


@dataclass(frozen=True, slots=True)
class FrameValue:
    """One filer's value of a concept in a calendar frame."""

    cik: int
    entity_name: str
    location: str
    value: float
    end: date
    start: date | None
    accession_number: str


@dataclass(frozen=True, slots=True)
class Frame:
    """One concept in one unit across every filer that reported it for one calendar period."""

    taxonomy: str
    concept: str
    unit: str
    period: str
    label: str
    values: tuple[FrameValue, ...]


async def frame(
    session: aiohttp.ClientSession, concept: str, unit: str, period: str, *, taxonomy: str = "us-gaap"
) -> Frame:
    """Cross-filer values for ``period``: ``CY2025`` (annual), ``CY2025Q2`` (quarterly), ``CY2025Q2I`` (instant)."""
    if not _FRAME_PERIOD_RE.match(period):
        raise ValueError(f"not a frames period (CY####, CY####Q#, CY####Q#I): {period!r}")
    payload = await _get_json(session, f"{DATA_BASE_URL}/api/xbrl/frames/{taxonomy}/{concept}/{unit}/{period}.json")
    values = tuple(
        FrameValue(
            cik=int(raw["cik"]),
            entity_name=raw["entityName"],
            location=raw["loc"],
            value=raw["val"],
            end=date.fromisoformat(raw["end"]),
            start=_optional_date(raw.get("start", "")),
            accession_number=raw["accn"],
        )
        for raw in payload["data"]
    )
    return Frame(
        taxonomy=payload["taxonomy"],
        concept=payload["tag"],
        unit=payload["uom"],
        period=payload["ccp"],
        label=payload["label"],
        values=values,
    )


# ---------------------------------------------------------------------------
# Full-text search
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One document matched by EDGAR full-text search, with enough to fetch it."""

    accession_number: str
    filename: str
    ciks: tuple[int, ...]
    display_names: tuple[str, ...]
    form: str
    file_date: date
    period_ending: date | None
    file_type: str
    file_description: str | None

    @property
    def document_url(self) -> str:
        return filing_document_url(self.ciks[0], self.accession_number, self.filename)


@dataclass(frozen=True, slots=True)
class SearchResults:
    total: int
    hits: tuple[SearchHit, ...]


def _search_hit(raw: dict[str, Any]) -> SearchHit:
    source = raw["_source"]
    _adsh, filename = raw["_id"].split(":", 1)
    return SearchHit(
        accession_number=source["adsh"],
        filename=filename,
        ciks=tuple(int(cik) for cik in source["ciks"]),
        display_names=tuple(source["display_names"]),
        form=source["form"],
        file_date=date.fromisoformat(source["file_date"]),
        period_ending=_optional_date(source.get("period_ending", "")),
        file_type=source["file_type"],
        file_description=source["file_description"],
    )


async def full_text_search(
    session: aiohttp.ClientSession,
    query: str,
    *,
    date_from: date | None = None,
    date_to: date | None = None,
    forms: str | Sequence[str] = (),
) -> SearchResults:
    """EDGAR full-text search over filings since 2001; ``forms`` filters on the root form type (``10-K``, ``8-K``).

    A one-sided date range is completed with the index's earliest date or today, because the server
    silently drops the whole ``file_date`` filter when either bound is missing (probed 2026-09-09).
    """
    if isinstance(forms, str):
        forms = (forms,)
    params = {"q": query}
    if date_from is not None or date_to is not None:
        params["dateRange"] = "custom"
        params["startdt"] = (date_from or FULL_TEXT_SEARCH_EARLIEST_DATE).isoformat()
        params["enddt"] = (date_to or datetime.now(tz=UTC).date()).isoformat()
    if forms:
        params["forms"] = ",".join(forms)
    payload = await _get_json(session, f"{FULL_TEXT_SEARCH_BASE_URL}/LATEST/search-index", params)
    hits = payload["hits"]
    return SearchResults(total=int(hits["total"]["value"]), hits=tuple(_search_hit(raw) for raw in hits["hits"]))


# ---------------------------------------------------------------------------
# Filing documents
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FilingDocument:
    """The raw bytes of one filing document; decoding and extraction are the caller's ladder's job."""

    url: str
    content_type: str
    body: bytes


def _edgar_hostnames() -> frozenset[str]:
    return frozenset(urlparse(base).hostname or "" for base in (WWW_BASE_URL, DATA_BASE_URL, FULL_TEXT_SEARCH_BASE_URL))


def is_edgar_url(url: str) -> bool:
    """Whether ``url`` is an HTTP(S) URL on one of the EDGAR hosts this client dials; hostnames compare case-insensitively."""
    parsed = urlparse(url)
    return parsed.scheme.lower() in ("http", "https") and (parsed.hostname or "") in _edgar_hostnames()


async def filing_document(session: aiohttp.ClientSession, url: str) -> FilingDocument:
    """Fetch one document from the EDGAR archive under the byte cap. Only EDGAR hosts: the User-Agent is theirs."""
    if not is_edgar_url(url):
        raise ValueError(f"{url} is not on an EDGAR host; this client dials sec.gov only")
    body, content_type, final_url = await _get_bytes(session, url)
    return FilingDocument(url=final_url, content_type=content_type, body=body)
