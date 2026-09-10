"""The documents one run has already parsed, keyed by URL: the fetch ladder's side channel.

A parsed :class:`~metaculus_bot.research.document_text.PdfText` never rides the ``FetchResult``
the research archive serialises, because its ``pages`` field is the whole document (833,450
characters on the receipt file) against a 200,000-character archive cap, so one such field would
truncate the whole question's archived payload. It travels here instead, and the gap-fill v2 loop
reads it back after the fetch for the page offsets a digest's ``[p.N]`` labels need.

Run-scoped rather than per question, so a second look at the same URL neither refetches nor
reparses it. Small on purpose: an entry is the per-page TEXT of a document up to
``DOCUMENT_TEXT_PDF_MAX_BYTES``, and the body itself is dropped as soon as extraction returns.
"""

from __future__ import annotations

from collections import OrderedDict

from metaculus_bot.research.document_text import PdfText

_DOCUMENT_CACHE_MAX_ENTRIES = 20
_DOCUMENT_CACHE: OrderedDict[str, PdfText] = OrderedDict()


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
