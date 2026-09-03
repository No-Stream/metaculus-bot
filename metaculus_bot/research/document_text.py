"""Deterministic PDF text extraction and lexical passage selection.

Stdlib and pypdf only, with no I/O of its own: bytes in, structured text out. Both
consumers — the Tier-1 resolution-source fetcher, which drops every PDF unread, and the
gap-fill v2 research loop, which sends every PDF to a paid Gemini ``url_context`` read —
already hold the bytes by the time they need the text, so acquiring them is the caller's
job and keeping this module free of network and model calls is what lets either one use
it without inheriting the other's dependencies.

Why deterministic extraction comes first: measured 2026-09-03, local pypdf pulled 833,450
chars out of a 6.7 MB 220-page PDF in 5.3 s and the passage the driver was looking for was
in it, while the paid reader returned nothing for the same file. So the order is acquire
the text locally, select the relevant passages locally, and spend a model call only on a
document we genuinely cannot read (``unreadable_reason``, or an empty
``has_text_layer``, which together separate "we could not parse this" from "this is a scan
with no text layer at all").

``is_pdf_body`` re-implements the ``%PDF-`` half of the private ``_body_is_document`` in
``research/agentic/fetch_outcomes.py`` rather than importing it: this module is the shared
foundation the agentic loop calls, so an import in that direction would invert the
dependency. The one-line magic check is cheaper to duplicate than the inversion is to live
with, and the two are pinned against each other in ``tests/test_document_text.py``.

Extraction is CPU-bound (pypdf parses and decodes every content stream), so an async
caller must run ``extract_pdf_text`` in a thread — ``asyncio.to_thread`` — never inline on
the event loop.
"""

from __future__ import annotations

import io
from bisect import bisect_right
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import log
from time import monotonic
from typing import Literal

from pypdf import PdfReader
from pypdf.errors import DependencyError, PyPdfError
from pypdf.generic import Destination

# Everything the reader and page-text paths raise on a mangled PDF, enumerated rather than
# blanket-caught. pypdf's own tree hangs off PyPdfError (PdfReadError, PdfStreamError,
# EmptyFileError, FileNotDecryptedError, LimitReachedError, ParseError), but DependencyError
# sits outside it, and the parser also raises plain stdlib types from inside the same paths
# (counted in pypdf 6.16.2 across _reader / _page / _doc_common / generic / _utils / filters:
# ValueError x23, TypeError x12, NotImplementedError x6, KeyError x5, IndexError x3, and the
# two Unicode errors under UnicodeError). The handful of bare `raise Exception` sites in
# pypdf are deliberately NOT in here: those are pypdf bugs and swallowing one would hide a
# real defect. Every guarded block below wraps exactly one pypdf call with none of this
# module's own logic inside it, so a KeyError from our code still crashes the way it should.
_PDF_ERRORS: tuple[type[Exception], ...] = (
    PyPdfError,
    DependencyError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    NotImplementedError,
    UnicodeError,
)

_PDF_MAGIC = b"%PDF-"

# A page carrying fewer than this many non-whitespace chars is running-header noise (a page
# number, a footer stamp), not a text layer, so a scan whose OCR-free pages emit a stray
# glyph still reads as "no text layer" rather than as a readable document.
TEXT_LAYER_MIN_CHARS = 40

# Okapi BM25 at its standard parameters, hand-rolled because the whole retrieval step is
# ~40 lines and a search dependency for that is not worth the supply-chain surface.
BM25_K1 = 1.5
BM25_B = 0.75

# Mirrors DOCUMENT_DIGEST_WINDOW_CHARS in constants.py, which is the knob callers pass;
# the two are pinned equal by a test so the default can never drift from the configured one.
DEFAULT_WINDOW_CHARS = 600

# Outline entries the digest renders before collapsing the rest into a count. A 220-page
# report's bookmark tree runs to hundreds of entries, which would crowd out the passages
# the digest exists to carry.
DIGEST_MAX_OUTLINE_ENTRIES = 25

UnreadableReason = Literal["", "encrypted", "malformed"]
TruncationCause = Literal["", "pages", "seconds"]


@dataclass(frozen=True)
class PdfText:
    """Per-page text of one PDF, plus what stopped the read.

    ``pages`` holds one entry per page actually read, stripped of surrounding whitespace,
    with ``""`` for a page that carries no text layer (or whose own content stream failed to
    decode — a partial read keeps the pages that worked). ``pages_read`` is therefore
    ``len(pages)`` and ``page_count`` is what the document declares, so
    ``pages_read < page_count`` plus ``truncated_by`` says which bound bit.

    ``unreadable_reason`` distinguishes the two ways a document yields nothing: ``"scanned"``
    is not one of its values because a scan parses fine and is reported as pages present and
    all empty (``has_text_layer`` is False), while ``"encrypted"`` / ``"malformed"`` mean we
    never got a page list at all.
    """

    page_count: int
    pages_read: int
    pages: tuple[str, ...]
    truncated_by: TruncationCause
    outline: tuple[tuple[str, int], ...]
    unreadable_reason: UnreadableReason = ""


@dataclass(frozen=True)
class Passage:
    """One scored window of a document, addressed by char offset into the joined text.

    ``text`` is exactly ``source[start:end]``. ``page`` is the 1-based page the passage sits
    on, or None when the caller supplied no page offsets; where they were supplied a window
    never straddles a page break, so the number is exact rather than a best guess.
    """

    score: float
    start: int
    end: int
    text: str
    page: int | None


def is_pdf_body(body: bytes) -> bool:
    """True when these bytes are a PDF, by the ``%PDF-`` header the format mandates.

    Tolerates leading whitespace, which some servers prepend, the way pypdf's own header
    scan does.
    """
    return body.lstrip().startswith(_PDF_MAGIC)


def extract_pdf_text(body: bytes, *, max_pages: int, max_seconds: float) -> PdfText:
    """Per-page text of ``body``, bounded by page count and wall-clock, never raising.

    Both bounds are the caller's: a resolution source we cannot read in ``max_seconds``
    costs the research phase more than it is worth, and ``max_pages`` bounds the pathological
    case (a 4,000-page appendix dump) independently of how fast the machine is. A bad PDF
    comes back as a ``PdfText`` carrying ``unreadable_reason`` rather than an exception,
    because both callers are soft-failing research providers and a malformed source document
    must never take a forecast run down.

    CPU-bound; run it in a thread from async code.
    """
    reader, reason = _open_reader(body)
    if reader is None:
        return PdfText(page_count=0, pages_read=0, pages=(), truncated_by="", outline=(), unreadable_reason=reason)

    page_count = _page_count(reader)
    if page_count is None:
        return PdfText(page_count=0, pages_read=0, pages=(), truncated_by="", outline=(), unreadable_reason="malformed")

    outline = _read_outline(reader)
    pages, truncated_by = _read_pages(reader, page_count=page_count, max_pages=max_pages, max_seconds=max_seconds)
    return PdfText(
        page_count=page_count,
        pages_read=len(pages),
        pages=pages,
        truncated_by=truncated_by,
        outline=outline,
        unreadable_reason="",
    )


def has_text_layer(pdf: PdfText) -> bool:
    """True when at least one read page carries real text rather than header noise.

    False on a scan, which is the signal a caller uses to escalate to a paid document read:
    that is the one case where a model is the only way to get the content.
    """
    return any(_visible_char_count(page) >= TEXT_LAYER_MIN_CHARS for page in pdf.pages)


def joined_page_text(pdf: PdfText) -> tuple[str, tuple[int, ...]]:
    """The read pages as one string, plus each page's start offset in it.

    Pages join on a blank line so a page boundary is also a paragraph boundary for
    ``select_passages``, and the returned offsets are what its ``page_breaks`` argument
    expects, so page attribution never has to be reconstructed by a caller.
    """
    breaks: list[int] = []
    cursor = 0
    for page in pdf.pages:
        breaks.append(cursor)
        cursor += len(page) + 2  # the "\n\n" separator below
    return "\n\n".join(pdf.pages), tuple(breaks)


def select_passages(
    text: str,
    query: str,
    *,
    top_k: int,
    window_chars: int = DEFAULT_WINDOW_CHARS,
    page_breaks: Sequence[int] | None = None,
) -> list[Passage]:
    """The ``top_k`` windows of ``text`` most relevant to ``query``, by BM25, highest first.

    Deterministic and lexical on purpose: the selection step decides which few hundred words
    of a 220-page document a forecaster or the research driver sees, and a model call there
    would put an unauditable choice on the critical path of every PDF. Ties break on
    position, earliest first, so re-running on the same inputs returns the same passages in
    the same order.

    Returns ``[]`` when no window contains any query term — an empty result means "this
    document does not discuss what you asked", which is information, where handing back the
    first ``top_k`` windows would dress the front matter up as an answer.

    Supplying ``page_breaks`` also makes a page boundary a hard window boundary, which is what
    keeps ``Passage.page`` exact: without it, two short pages pack into one window and the
    passage is labelled with the page it starts on even when the match is a page later. That
    costs a phrase straddling a page break, which BM25 then scores as two partial windows —
    the right trade, since a ``[p.N]`` label a forecaster may cite has to be true.
    """
    # Sorted once here so the windowing and the page labels read the same order, whatever a
    # caller hands in; `joined_page_text` already returns them ascending.
    breaks = tuple(sorted(page_breaks)) if page_breaks else None
    spans = _window_spans(text, window_chars, page_breaks=breaks)
    if not spans or top_k <= 0:
        return []

    query_tokens = _unique_tokens(query)
    if not query_tokens:
        return []

    window_tokens = [_tokenise(text[start:end]) for start, end in spans]
    scores = _bm25_scores(window_tokens, query_tokens)
    ranked = sorted(
        ((score, spans[i]) for i, score in enumerate(scores) if score > 0.0),
        key=lambda item: (-item[0], item[1][0]),
    )
    return [
        Passage(
            score=score,
            start=start,
            end=end,
            text=text[start:end],
            page=_page_for_offset(start, breaks),
        )
        for score, (start, end) in ranked[:top_k]
    ]


def render_document_digest(pdf: PdfText, *, query: str, top_k: int, max_chars: int, source_url: str) -> str:
    """The forecaster/driver-facing block for one document: what it is, then what it says.

    Deterministic and I/O-free, so the same ``PdfText`` always renders the same block. The
    header states what was read and what was not, because a digest of the first 400 pages of
    a 900-page document that does not say so is a silent partial read.

    The no-text-layer branch fires on a document with no text AT ALL, not on a failed
    ``has_text_layer`` — that function's 40-char floor is the caller's escalate-to-a-paid-read
    signal, and reusing it here would withhold the whole content of a one-line PDF ("Q3
    unemployment rate: 4.1%") as if it were a scan. Terse-but-real text is rendered; a scan
    whose only text is page numbers simply matches no query and says so.
    """
    sections = [_digest_header(pdf, source_url=source_url)]
    if pdf.unreadable_reason:
        sections.append(f"The document could not be parsed ({pdf.unreadable_reason}); no text was extracted.")
        return _truncate_digest("\n\n".join(sections), max_chars)

    sections.extend(_digest_outline(pdf.outline))
    text, page_breaks = joined_page_text(pdf)
    if not text.strip():
        sections.append("No extractable text layer: the pages carry images rather than text.")
        return _truncate_digest("\n\n".join(sections), max_chars)

    passages = select_passages(text, query, top_k=top_k, page_breaks=page_breaks)
    sections.append(_digest_passages(passages, query=query))
    return _truncate_digest("\n\n".join(sections), max_chars)


# --- PDF reading -------------------------------------------------------------------------


def _open_reader(body: bytes) -> tuple[PdfReader | None, UnreadableReason]:
    """A reader for ``body``, or None plus the reason it is unreadable."""
    try:
        reader = PdfReader(io.BytesIO(body))
    except _PDF_ERRORS:
        return None, "malformed"
    if not reader.is_encrypted:
        return reader, ""
    # An empty user password is the common "encrypted for print restrictions only" case and
    # reads normally; a real password means the bytes stay opaque to us.
    try:
        decrypted = bool(reader.decrypt(""))
    except _PDF_ERRORS:
        decrypted = False
    return (reader, "") if decrypted else (None, "encrypted")


def _page_count(reader: PdfReader) -> int | None:
    """The declared page count, or None when the page tree itself does not parse."""
    try:
        return len(reader.pages)
    except _PDF_ERRORS:
        return None


def _read_pages(
    reader: PdfReader, *, page_count: int, max_pages: int, max_seconds: float
) -> tuple[tuple[str, ...], TruncationCause]:
    """Text of the first ``max_pages`` pages, stopping early once ``max_seconds`` elapses.

    The clock is checked AFTER each page so a budget too small for even one page still
    returns that page: forward progress beats an empty result, and the caller learns the
    budget bound anyway from ``truncated_by``.
    """
    deadline = monotonic() + max_seconds
    limit = min(page_count, max_pages)
    pages: list[str] = []
    for index in range(limit):
        pages.append(_page_text(reader, index))
        if monotonic() >= deadline and index + 1 < page_count:
            return tuple(pages), "seconds"
    return tuple(pages), "pages" if limit < page_count else ""


def _page_text(reader: PdfReader, index: int) -> str:
    """One page's text, or ``""`` when that page has no text layer or fails to decode.

    A page whose content stream is corrupt is reported the same way as a page with no text:
    the whole point of a per-page tuple is that one bad page does not cost the other 219.
    """
    try:
        return reader.pages[index].extract_text().strip()
    except _PDF_ERRORS:
        return ""


def _read_outline(reader: PdfReader) -> tuple[tuple[str, int], ...]:
    """The bookmark tree flattened to (title, 1-based page) in document order.

    Nesting is flattened rather than indented because the digest uses the outline to say what
    is in the document and on which page, and a caller that wants the hierarchy can read the
    PDF itself. An outline whose destinations do not resolve — a linearised file whose page
    objects moved, a tree pypdf cannot walk — yields ``()``: a partial outline with silently
    wrong page numbers is worse than none, since the digest's own page labels come from the
    text offsets and would then contradict it.
    """
    try:
        return tuple(_walk_outline(reader, reader.outline))
    except _PDF_ERRORS:
        return ()


def _walk_outline(reader: PdfReader, items: Sequence[object]) -> list[tuple[str, int]]:
    """Depth-first walk of pypdf's nested list of Destinations.

    An untitled entry is skipped: the digest renders the outline as "title (p.N)", and a
    bookmark with no title carries no information a reader can act on.
    """
    entries: list[tuple[str, int]] = []
    for item in items:
        if isinstance(item, list):
            entries.extend(_walk_outline(reader, item))
            continue
        if not isinstance(item, Destination):
            continue
        title = (item.title or "").strip()
        page_index = reader.get_destination_page_number(item)
        if title and page_index is not None and page_index >= 0:
            entries.append((title, page_index + 1))
    return entries


def _visible_char_count(page: str) -> int:
    return sum(1 for char in page if not char.isspace())


# --- Passage selection -------------------------------------------------------------------


def _tokenise(text: str) -> list[str]:
    """Lowercase alphanumeric runs. Punctuation and case carry no retrieval signal here.

    Alphanumeric in the Unicode sense, not ASCII: resolution sources include non-English
    pages, and an ASCII-only class would shred "Ürgüp" into two meaningless fragments and
    then fail to match the same word in the query.
    """
    tokens: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _unique_tokens(query: str) -> list[str]:
    """Query terms, deduplicated in first-appearance order.

    A repeated term in a short query is emphasis a user did not mean to encode as a weight,
    and dedup keeps the score independent of how the ask was phrased.
    """
    return list(dict.fromkeys(_tokenise(query)))


def _bm25_scores(window_tokens: list[list[str]], query_tokens: list[str]) -> list[float]:
    """Okapi BM25 of every window against the query terms."""
    n_windows = len(window_tokens)
    lengths = [len(tokens) for tokens in window_tokens]
    total_length = sum(lengths)
    if not total_length:
        return [0.0] * n_windows
    avg_length = total_length / n_windows

    counters = [Counter(tokens) for tokens in window_tokens]
    idf = {
        term: log(1 + (n_windows - df + 0.5) / (df + 0.5))
        for term, df in ((term, sum(1 for c in counters if term in c)) for term in query_tokens)
    }
    return [
        _window_score(counter, length, query_tokens=query_tokens, idf=idf, avg_length=avg_length)
        for counter, length in zip(counters, lengths, strict=True)
    ]


def _window_score(
    counter: Counter[str],
    length: int,
    *,
    query_tokens: list[str],
    idf: dict[str, float],
    avg_length: float,
) -> float:
    score = 0.0
    for term in query_tokens:
        freq = counter.get(term, 0)
        if not freq:
            continue  # a term absent from this window contributes nothing, absent everywhere or not
        denominator = freq + BM25_K1 * (1 - BM25_B + BM25_B * length / avg_length)
        score += idf[term] * freq * (BM25_K1 + 1) / denominator
    return score


def _window_spans(text: str, window_chars: int, *, page_breaks: Sequence[int] | None = None) -> list[tuple[int, int]]:
    """Char spans covering ``text``, never crossing a page break, in document order."""
    if window_chars <= 0:
        return []
    spans: list[tuple[int, int]] = []
    for start, end in _page_segments(text, page_breaks):
        spans.extend(_segment_windows(text, start, end, window_chars))
    return spans


def _page_segments(text: str, page_breaks: Sequence[int] | None) -> list[tuple[int, int]]:
    """``text`` cut at the page starts, or one segment when the caller gave none."""
    if not page_breaks:
        return [(0, len(text))]
    bounds = sorted({0, len(text), *(offset for offset in page_breaks if 0 < offset < len(text))})
    return list(pairwise(bounds))


def _segment_windows(text: str, start: int, end: int, window_chars: int) -> list[tuple[int, int]]:
    """One page's spans, split on paragraph boundaries where it has them.

    Packing consecutive short paragraphs up to ``window_chars`` keeps a scored window big
    enough to carry context, and a paragraph longer than the budget (a wall-of-text page, or
    a PDF whose extraction produced no blank lines at all) falls back to fixed windows cut at
    whitespace.
    """
    spans: list[tuple[int, int]] = []
    pending: tuple[int, int] | None = None
    for para_start, para_end in _paragraph_spans(text, start, end):
        if para_end - para_start > window_chars:
            if pending is not None:
                spans.append(pending)
                pending = None
            spans.extend(_split_long_span(text, para_start, para_end, window_chars))
        elif pending is None:
            pending = (para_start, para_end)
        elif para_end - pending[0] <= window_chars:
            pending = (pending[0], para_end)
        else:
            spans.append(pending)
            pending = (para_start, para_end)
    if pending is not None:
        spans.append(pending)
    return spans


def _paragraph_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Non-empty paragraph spans within ``[start, end)``, whitespace-trimmed, in order."""
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        break_at = _next_blank_line(text, cursor, end)
        stop = end if break_at is None else break_at[0]
        trimmed = _trim_span(text, cursor, stop)
        if trimmed is not None:
            spans.append(trimmed)
        if break_at is None:
            break
        cursor = break_at[1]
    return spans


def _next_blank_line(text: str, start: int, length: int) -> tuple[int, int] | None:
    """Span of the next run of whitespace containing two or more newlines, or None."""
    cursor = start
    while cursor < length:
        if text[cursor] != "\n":
            cursor += 1
            continue
        run_end = cursor
        newlines = 0
        while run_end < length and text[run_end].isspace():
            if text[run_end] == "\n":
                newlines += 1
            run_end += 1
        if newlines >= 2:
            return cursor, run_end
        cursor = max(run_end, cursor + 1)
    return None


def _trim_span(text: str, start: int, end: int) -> tuple[int, int] | None:
    """``(start, end)`` narrowed past surrounding whitespace, or None when it holds none."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def _split_long_span(text: str, start: int, end: int, window_chars: int) -> list[tuple[int, int]]:
    """A too-long paragraph cut into ``window_chars`` windows, at whitespace where possible."""
    spans: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + window_chars, end)
        if stop < end:
            pivot = _last_space(text, cursor + (window_chars * 3) // 4, stop)
            if pivot > cursor:
                stop = pivot
        trimmed = _trim_span(text, cursor, stop)
        if trimmed is not None:
            spans.append(trimmed)
        cursor = stop if stop > cursor else cursor + 1
    return spans


def _last_space(text: str, low: int, high: int) -> int:
    """Index of the last whitespace char in ``[low, high)``, or -1 — the preferred cut point."""
    for index in range(high - 1, low - 1, -1):
        if text[index].isspace():
            return index
    return -1


def _page_for_offset(offset: int, page_breaks: Sequence[int] | None) -> int | None:
    """The 1-based page containing ``offset``, given ascending page-start offsets."""
    if not page_breaks:
        return None
    return max(1, bisect_right(page_breaks, offset))


# --- Digest rendering --------------------------------------------------------------------


def _digest_header(pdf: PdfText, *, source_url: str) -> str:
    chars = sum(len(page) for page in pdf.pages)
    note = {
        "pages": f"; stopped at the {pdf.pages_read}-page read cap",
        "seconds": f"; stopped after {pdf.pages_read} pages on the extraction time budget",
    }.get(pdf.truncated_by, "")
    return (
        f"Document: {source_url}\n{pdf.page_count} pages, {pdf.pages_read} read, {chars} chars of text extracted{note}"
    )


def _digest_outline(outline: tuple[tuple[str, int], ...]) -> list[str]:
    if not outline:
        return []
    shown = outline[:DIGEST_MAX_OUTLINE_ENTRIES]
    lines = ["Outline:"] + [f"  {title} (p.{page})" for title, page in shown]
    if len(outline) > len(shown):
        lines.append(f"  ... {len(outline) - len(shown)} further outline entries")
    return ["\n".join(lines)]


def _digest_passages(passages: list[Passage], *, query: str) -> str:
    if not passages:
        return f"No passage in this document matched the query: {query}"
    header = f"Most relevant passages for: {query}"
    body = [f"[p.{p.page}] {p.text}" if p.page is not None else f"[p.?] {p.text}" for p in passages]
    return "\n\n".join([header, *body])


def _truncate_digest(block: str, max_chars: int) -> str:
    """``block`` bounded to ``max_chars``, with the cut disclosed in the text.

    The marker is never dropped to make room, even when ``max_chars`` is smaller than the
    marker itself: a block that silently ends mid-passage reads as a complete document
    digest, which is exactly the failure the marker exists to prevent.
    """
    if len(block) <= max_chars:
        return block
    marker = f"[digest truncated at {max_chars} chars]"
    keep = max(0, max_chars - len(marker) - 1)  # -1 for the newline joining the two
    kept = block[:keep].rstrip()
    return f"{kept}\n{marker}" if kept else marker
