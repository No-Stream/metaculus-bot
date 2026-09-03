"""Reading a document the fetch ladder already holds, instead of paying a model to read it.

The gap-fill v2 ladder used to classify a PDF as ``document_needed`` from its Content-Type
alone, before a single byte was read, and hand it to a paid Gemini ``url_context`` call. Over
the 2026 summer season that was 191 reader calls, nine documents over 100k tokens carried 67%
of all the tokens retrieved, and on the one document where both routes were tried local pypdf
pulled 833,450 chars in 5.3 s while the paid read returned nothing at all. So the order is now
acquire the bytes, extract the text locally, select passages locally, and spend a reader call
only on a document we genuinely cannot read.

This module owns the PDF rung and the digest rendering that sit between the ladder spine in
``tools.py`` and the pure text machinery in ``research/document_text.py``:

* :func:`pdf_fetch_result` — bytes in, a :class:`PlainFetchResult` out, plus the run-scoped
  cache entry that keeps a second look at the same URL from re-parsing it.
* :func:`digest_held` — the passage digest for a document we hold, page-wise for a PDF and
  flat for an HTML page, rendered in one shape either way.
* :func:`exceeds_url_context_size_gate` — the hard floor on the one paid call in the ladder.
* :func:`log_local_document_read` — the ``AGENTIC_FETCH_LOCAL_DOC`` telemetry marker.

It deliberately does NOT import ``tools``: the dependency runs one way (``tools`` →
``local_document`` → ``document_text``), so the extraction and the digest stay testable
without standing up the ladder, its aiohttp session or its Chromium rung.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass

from metaculus_bot.constants import (
    DOCUMENT_TEXT_MAX_PAGES,
    DOCUMENT_TEXT_MAX_SECONDS,
    DOCUMENT_TEXT_PDF_MAX_BYTES,
    URL_CONTEXT_SIZE_GATE_TOKENS,
)
from metaculus_bot.research.agentic.fetch_outcomes import (
    PlainFetchResult,
    _document_needed_result,
)
from metaculus_bot.research.document_text import (
    DocumentDigest,
    PdfText,
    digest_pdf,
    digest_text,
    extract_pdf_text,
    has_text_layer,
    joined_page_text,
)

logger = logging.getLogger(__name__)

# ``ToolOutcome.method`` values this rung produces. Both map to the ``fetched`` verification
# tier in ``provenance._METHOD_TO_TIER`` — we decoded the bytes the host served, which is a
# stronger claim than a model's paraphrase of them — so both names are load-bearing and are
# pinned by the tier tests rather than spelled inline at their call sites.
PDF_LOCAL_METHOD = "pdf_local"
DIGEST_LOCAL_METHOD = "digest_local"
# NOT tiered: nothing was read. Its own name rather than a bare "error" so the driver's
# outcome, and the archived run log, say WHY the paid reader was skipped as well.
OVERSIZE_DOCUMENT_METHOD = "oversize_document"

# chars / 4, the estimator the season's reader sizing was measured with.
_CHARS_PER_TOKEN_ESTIMATE = 4

# Extracted documents held for the rest of the run, so ``start_char`` pagination and a later
# ``read_document`` on the same URL neither refetch nor re-parse. Small on purpose: an entry is
# per-page text of a document up to DOCUMENT_TEXT_PDF_MAX_BYTES, and only the TEXT is kept —
# the body itself is dropped as soon as extraction returns.
_DOCUMENT_CACHE_MAX_ENTRIES = 20
_DOCUMENT_CACHE: OrderedDict[str, PdfText] = OrderedDict()

# Process-global cap on concurrent PDF parses, for the same reason the rendered rung caps
# Chromium launches: pypdf decodes every content stream, so a parse is both CPU-bound and
# holds its body for the duration. The driver's parallel_tool_calls can request several
# fetches in one step and six questions run concurrently under the orchestrator, so an
# unbounded fan-out is 6xN bodies of up to 40 MiB plus their parse arenas — a MemoryError no
# soft-fail boundary can catch. Two slots covers a real burst while bounding the peak.
_EXTRACT_GLOBAL_SEMAPHORE = asyncio.Semaphore(2)


@dataclass(frozen=True, slots=True)
class HeldDocument:
    """What the free ladder holds for one URL: its text, its page structure, or neither.

    ``text`` is the whole document as one string (a PDF's pages joined, or a page's extracted
    main text) — the pagination and digest source. ``pdf`` is present only when we parsed a
    PDF, and carries the page offsets that make a digest's ``[p.N]`` labels exact; it is set
    even for a scan, where ``text`` is empty, because "we looked locally and there is no text
    layer" is exactly what tells a later call to stop trying for free. ``oversize`` means the
    body was refused before parsing, which is a reason NOT to escalate rather than a reason to.
    """

    text: str = ""
    pdf: PdfText | None = None
    oversize: bool = False

    @property
    def has_text(self) -> bool:
        return bool(self.text.strip())


def held_pdf(pdf: PdfText) -> HeldDocument:
    """The held form of a parsed document: its joined text plus the page structure.

    A scan comes back with page structure and no text, which is the shape that tells a caller
    the free route is exhausted rather than untried.
    """
    return HeldDocument(text=joined_page_text(pdf)[0].strip(), pdf=pdf)


def cached_document(url: str) -> PdfText | None:
    """The parsed document held for ``url`` this run, or None."""
    pdf = _DOCUMENT_CACHE.get(url)
    if pdf is not None:
        _DOCUMENT_CACHE.move_to_end(url)
    return pdf


def cache_document(url: str, pdf: PdfText) -> None:
    """Hold ``pdf`` for ``url`` for the rest of the run, evicting the least recently used."""
    _DOCUMENT_CACHE[url] = pdf
    _DOCUMENT_CACHE.move_to_end(url)
    while len(_DOCUMENT_CACHE) > _DOCUMENT_CACHE_MAX_ENTRIES:
        _DOCUMENT_CACHE.popitem(last=False)


def clear_document_cache() -> None:
    """Drop every held document. Run-scoped state, so the suite resets it per test."""
    _DOCUMENT_CACHE.clear()


async def pdf_fetch_result(body: bytes, *, url: str, content_type: str) -> PlainFetchResult:
    """Extract ``body``'s text locally and shape it as a ladder result.

    A text layer comes back as ``pdf_local`` carrying the WHOLE joined text, so the fetch
    handler's existing window/cache path paginates a 220-page report exactly as it paginates
    a long HTML page. No text layer — a scan — comes back as ``document_needed``, which is
    the paid reader's one remaining job on this rung; either way the parse is cached, so the
    escalation costs no second request and no second parse.

    Never raises: :func:`extract_pdf_text` reports a mangled document through
    ``unreadable_reason`` instead, and that also lands as ``document_needed``.
    """
    async with _EXTRACT_GLOBAL_SEMAPHORE:
        # CPU-bound (pypdf parses and decodes every content stream), so it must not run on the
        # event loop. A caller whose own budget expires first cancels this coroutine but not
        # the worker thread, which finishes and drops its result — bounded by max_seconds.
        pdf = await asyncio.to_thread(
            extract_pdf_text, body, max_pages=DOCUMENT_TEXT_MAX_PAGES, max_seconds=DOCUMENT_TEXT_MAX_SECONDS
        )
    cache_document(url, pdf)
    if not has_text_layer(pdf):
        return _document_needed_result(url, content_type)
    text, _page_breaks = joined_page_text(pdf)
    return PlainFetchResult(
        status="ok",
        method=PDF_LOCAL_METHOD,
        text=text.strip(),
        links=[],
        url=url,
        content_type=content_type or None,
    )


_OVERSIZE_DOCUMENT_MSG = (
    "Document too large to read: {url} is a document over the {mib} MiB local-read cap, so "
    "nothing was read from it and no model read was attempted either — a document this size "
    "costs more to have a model retrieve than any answer it could return is worth. Find a "
    "smaller source, or a page that summarises this one."
)


def oversize_result(url: str, content_type: str) -> PlainFetchResult:
    """Terminal result for a document body refused before parsing, on its own method name."""
    return PlainFetchResult(
        status="error",
        method=OVERSIZE_DOCUMENT_METHOD,
        text=oversize_message(url),
        links=[],
        url=url,
        content_type=content_type or None,
    )


def oversize_message(url: str) -> str:
    return _OVERSIZE_DOCUMENT_MSG.format(url=url, mib=DOCUMENT_TEXT_PDF_MAX_BYTES // (1024 * 1024))


def digest_held(held: HeldDocument, *, ask: str, top_k: int, max_chars: int, source_url: str) -> DocumentDigest:
    """The passage digest for a document we hold, page-wise where we have pages.

    One entry point for both shapes so the driver reads one format whether the ask landed on
    a PDF or an HTML page; the PDF branch is the richer one (outline, per-passage page
    numbers) and is preferred whenever a parse is in hand.
    """
    if held.pdf is not None:
        return digest_pdf(held.pdf, query=ask, top_k=top_k, max_chars=max_chars, source_url=source_url)
    return digest_text(held.text, query=ask, top_k=top_k, max_chars=max_chars, source_url=source_url)


def exceeds_url_context_size_gate(text: str) -> bool:
    """True when text we ALREADY hold is too big to be worth a paid ``url_context`` read.

    A hard floor on the one paid call in this rung rather than a live branch: the ladder above
    serves any URL whose text it holds from the local digest, so a document this size should
    never reach the reader at all. It is enforced anyway because the failure it prevents is
    the season's worst reader spend — the nine archived documents past this bound carried 67%
    of all tokens the reader retrieved, and the largest of them returned nothing for the money.
    """
    return len(text) // _CHARS_PER_TOKEN_ESTIMATE > URL_CONTEXT_SIZE_GATE_TOKENS


def log_local_document_read(url: str, *, method: str, chars: int, pages: int | None, passages: int | None) -> None:
    """Emit ``AGENTIC_FETCH_LOCAL_DOC``: one line per document served without a model call.

    ``chars`` is the local text we HELD, not the window or block handed to the driver, so the
    figure is comparable across both methods and against the size gate above. ``pages`` is
    ``n/a`` for a page with no page structure and ``passages`` is ``n/a`` for a ``pdf_local``
    fetch, which serves the text itself and selects nothing.
    """
    logger.info(
        f"AGENTIC_FETCH_LOCAL_DOC: url={url} method={method} chars={chars} "
        f"pages={'n/a' if pages is None else pages} passages={'n/a' if passages is None else passages}"
    )
