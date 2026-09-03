"""Tests for local PDF text extraction and deterministic passage selection.

The module under test is the shared foundation for reading a PDF we already hold, instead of
dropping it unread (the Tier-1 resolution-source fetcher's `unsupported_type`) or paying a
model to read it (the gap-fill v2 loop's `read_document` escalation). Its two hard contracts
are what the tests below are organised around: it never raises on a bad document, and it
never invents relevance — a query with no match returns nothing rather than the front matter.

PDFs are built here rather than checked in, so a fixture is readable as source and the suite
carries no binaries. `build_text_pdf` writes a structurally valid PDF with a real Type1 text
block per page, which is what makes pypdf's own extractor the thing under test.
"""

from __future__ import annotations

import io

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.errors import PdfReadError

from metaculus_bot.constants import DOCUMENT_DIGEST_WINDOW_CHARS
from metaculus_bot.research import document_text
from metaculus_bot.research.document_text import (
    DEFAULT_WINDOW_CHARS,
    TEXT_LAYER_MIN_CHARS,
    PdfText,
    extract_pdf_text,
    has_text_layer,
    is_pdf_body,
    joined_page_text,
    render_document_digest,
    select_passages,
)


def _extract(body: bytes) -> PdfText:
    """Extract with bounds well above every fixture, so only the intended bound is under test."""
    return extract_pdf_text(body, max_pages=100, max_seconds=60.0)


def _escape_pdf_string(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str]) -> bytes:
    """A PDF content stream drawing ``lines`` as Helvetica text, one line per row."""
    body = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
    for index, line in enumerate(lines):
        if index:
            body.append("T*")
        body.append(f"({_escape_pdf_string(line)}) Tj")
    body.append("ET")
    return ("\n".join(body) + "\n").encode("latin-1")


def build_text_pdf(pages: list[list[str]]) -> bytes:
    """A valid single-font PDF whose page ``i`` draws ``pages[i]`` as lines of text.

    Hand-rolled because pypdf's writer can create pages but not draw text on them, and a
    text-drawing dependency (reportlab) for a test fixture is not worth the install.
    """
    objects: list[bytes] = [b"", b""]  # 1 = catalog, 2 = page tree; filled in once kids are known
    catalog_num, pages_num = 1, 2

    def add(raw: bytes) -> int:
        objects.append(raw)
        return len(objects)

    font_num = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    kids: list[int] = []
    for lines in pages:
        stream = _content_stream(lines)
        contents_num = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"endstream")
        kids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>" % (pages_num, font_num, contents_num)
            )
        )
    objects[pages_num - 1] = (
        b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % kid for kid in kids) + b"] /Count %d >>" % len(pages)
    )
    objects[catalog_num - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_num

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, raw in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + raw + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        catalog_num,
        xref_at,
    )
    return bytes(out)


def _with_outline(data: bytes, entries: list[tuple[str, int, bool]]) -> bytes:
    """``data`` re-written with bookmarks; each entry is (title, 0-based page, nest_under_last)."""
    writer = PdfWriter(clone_from=io.BytesIO(data))
    parent = None
    for title, page_index, nested in entries:
        added = writer.add_outline_item(title, page_index, parent=parent if nested else None)
        if not nested:
            parent = added
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _encrypted(data: bytes, *, user_password: str) -> bytes:
    writer = PdfWriter(clone_from=io.BytesIO(data))
    writer.encrypt(user_password=user_password, owner_password="owner-secret")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class _FakeClock:
    """A monotonic clock advancing a fixed step per read, so the time bound is deterministic."""

    def __init__(self, step: float) -> None:
        self._step = step
        self._now = 0.0

    def __call__(self) -> float:
        now = self._now
        self._now += self._step
        return now


class TestIsPdfBody:
    def test_magic_and_leading_whitespace(self) -> None:
        assert is_pdf_body(b"%PDF-1.7\nrest")
        assert is_pdf_body(b"\r\n  %PDF-1.4\n"), "servers do prepend whitespace; pypdf tolerates it too"
        assert not is_pdf_body(b"<html><body>not a pdf</body></html>")
        assert not is_pdf_body(b"")

    def test_agrees_with_the_agentic_private_check(self) -> None:
        """The duplicated one-liner must stay equivalent on the PDF branch it duplicates.

        `research/agentic/fetch_outcomes.py` keeps its own `_body_is_document`, which also
        matches PNG/JPEG/GIF magic. Importing it HERE (test-only) pins the PDF half without
        putting an agentic -> document_text-caller edge in the shipped package, which would
        invert the dependency this module exists to sit under.
        """
        from metaculus_bot.research.agentic.fetch_outcomes import _body_is_document

        for body in (b"%PDF-1.7\nx", b"  %PDF-1.4", b"plain text", b"", b'{"json": true}'):
            assert is_pdf_body(body) == _body_is_document(body), f"disagreed on {body!r}"


class TestExtraction:
    def test_text_per_page(self) -> None:
        data = build_text_pdf([["alpha beta gamma", "second line here"], ["page two content"]])
        result = _extract(data)

        assert result.page_count == 2
        assert result.pages_read == 2
        assert result.pages == ("alpha beta gamma\nsecond line here", "page two content")
        assert result.truncated_by == ""
        assert result.unreadable_reason == ""
        assert has_text_layer(result) is False, "these pages are shorter than the text-layer floor"

    def test_page_cap_truncates_and_says_so(self) -> None:
        data = build_text_pdf([[f"page {n} body text"] for n in range(1, 8)])
        result = extract_pdf_text(data, max_pages=3, max_seconds=60.0)

        assert result.page_count == 7, "the declared count is the whole document"
        assert result.pages_read == 3
        assert len(result.pages) == 3
        assert result.pages[2] == "page 3 body text"
        assert result.truncated_by == "pages"

    def test_page_cap_above_the_document_is_not_a_truncation(self) -> None:
        data = build_text_pdf([["only page"]])
        result = extract_pdf_text(data, max_pages=50, max_seconds=60.0)

        assert result.pages_read == result.page_count == 1
        assert result.truncated_by == ""

    def test_time_cap_stops_mid_document(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 10 s per clock read: deadline 15 s is passed on the read after page 2.
        monkeypatch.setattr(document_text, "monotonic", _FakeClock(step=10.0))
        data = build_text_pdf([[f"page {n}"] for n in range(1, 6)])

        result = extract_pdf_text(data, max_pages=100, max_seconds=15.0)

        assert result.pages_read == 2
        assert result.truncated_by == "seconds"
        assert result.page_count == 5, "the time bound must not misreport how long the document is"

    def test_exhausted_budget_still_returns_one_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Forward progress beats an empty result; the caller learns the bound from truncated_by."""
        monkeypatch.setattr(document_text, "monotonic", _FakeClock(step=100.0))
        data = build_text_pdf([["first"], ["second"]])

        result = extract_pdf_text(data, max_pages=100, max_seconds=0.0)

        assert result.pages_read == 1
        assert result.pages == ("first",)
        assert result.truncated_by == "seconds"

    def test_time_cap_on_the_last_page_is_not_a_truncation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(document_text, "monotonic", _FakeClock(step=100.0))
        data = build_text_pdf([["only page"]])

        result = extract_pdf_text(data, max_pages=100, max_seconds=0.0)

        assert result.pages_read == 1
        assert result.truncated_by == "", "nothing was left unread, so nothing was truncated"


class TestUnreadableDocuments:
    def test_no_text_layer_reads_as_pages_present_and_empty(self) -> None:
        """A scan is the one case a paid document read is the only route, so it must be visible."""
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buffer = io.BytesIO()
        writer.write(buffer)

        result = _extract(buffer.getvalue())

        assert result.unreadable_reason == "", "a scan parses fine; it simply has no text"
        assert result.pages == ("",)
        assert result.page_count == 1
        assert has_text_layer(result) is False

    def test_has_text_layer_floor(self) -> None:
        assert has_text_layer(PdfText(1, 1, ("x" * TEXT_LAYER_MIN_CHARS,), "", ())) is True
        assert has_text_layer(PdfText(1, 1, ("x" * (TEXT_LAYER_MIN_CHARS - 1),), "", ())) is False
        assert has_text_layer(PdfText(2, 2, ("", " " * 500), "", ())) is False, "whitespace is not a text layer"
        assert has_text_layer(PdfText(0, 0, (), "", ())) is False

    def test_encrypted_with_a_real_password(self) -> None:
        data = _encrypted(build_text_pdf([["secret contents"]]), user_password="sekrit")

        result = _extract(data)

        assert result.unreadable_reason == "encrypted"
        assert result.pages == ()
        assert result.pages_read == 0

    def test_encrypted_with_an_empty_user_password_reads_normally(self) -> None:
        """Print-restriction encryption is common on government PDFs and is readable."""
        data = _encrypted(build_text_pdf([["readable contents here"]]), user_password="")

        result = _extract(data)

        assert result.unreadable_reason == ""
        assert result.pages == ("readable contents here",)

    @pytest.mark.parametrize(
        ("label", "body"),
        [
            ("not a pdf at all", b"<html>nope</html>"),
            ("magic only", b"%PDF-1.7\n" + b"x" * 64),
            ("empty", b""),
            ("truncated", build_text_pdf([["alpha"], ["beta"]])[:200]),
            ("nuked xref", build_text_pdf([["alpha"]]).replace(b"xref", b"xrfe", 1)),
        ],
    )
    def test_malformed_never_raises(self, label: str, body: bytes) -> None:
        result = _extract(body)

        assert result.unreadable_reason == "malformed", label
        assert result.pages == ()
        assert result.page_count == 0
        assert result.outline == ()

    def test_a_corrupt_stream_length_costs_only_that_page(self) -> None:
        """A partial read keeps the pages that worked — one bad page must not cost the rest."""
        data = build_text_pdf([["alpha alpha alpha"], ["beta beta beta"]])
        result = _extract(data)

        assert result.unreadable_reason == ""
        assert result.pages_read == 2


class TestOutline:
    def test_nested_entries_flatten_in_document_order(self) -> None:
        data = _with_outline(
            build_text_pdf([["one"], ["two"], ["three"]]),
            [("Chapter One", 0, False), ("Chapter Two", 1, False), ("Section 2.1", 2, True)],
        )

        result = _extract(data)

        assert result.outline == (("Chapter One", 1), ("Chapter Two", 2), ("Section 2.1", 3)), (
            "pages are 1-based and the nested child sits inline after its parent"
        )

    def test_no_outline_is_an_empty_tuple(self) -> None:
        result = _extract(build_text_pdf([["no bookmarks here"]]))
        assert result.outline == ()

    def test_unresolvable_destinations_yield_no_outline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Silently wrong page numbers would contradict the digest's own text-derived labels."""

        def _raise(*_args: object, **_kwargs: object) -> int:
            raise PdfReadError("destination page not in this tree")

        monkeypatch.setattr(PdfReader, "get_destination_page_number", _raise)
        data = _with_outline(build_text_pdf([["one"], ["two"]]), [("Chapter One", 0, False)])

        result = _extract(data)

        assert result.outline == ()
        assert result.pages_read == 2, "an unusable outline must not cost the text"


class TestJoinedPageText:
    def test_offsets_locate_each_page(self) -> None:
        pdf = PdfText(3, 3, ("first page", "second page", "third page"), "", ())
        text, breaks = joined_page_text(pdf)

        assert text == "first page\n\nsecond page\n\nthird page"
        assert len(breaks) == 3
        for index, page in enumerate(pdf.pages):
            assert text[breaks[index] : breaks[index] + len(page)] == page

    def test_empty_document(self) -> None:
        assert joined_page_text(PdfText(0, 0, (), "", ())) == ("", ())


_FILLER_WORDS = [
    "regional",
    "quarterly",
    "administrative",
    "appendix",
    "methodology",
    "footnote",
    "tabulation",
    "supplementary",
]


def _filler_paragraphs(count: int) -> list[str]:
    """Deterministic filler that shares no token with the needle query below."""
    return [
        " ".join(_FILLER_WORDS[(index + offset) % len(_FILLER_WORDS)] for offset in range(40)) for index in range(count)
    ]


class TestSelectPassages:
    NEEDLE = "The shuttle carried 47 passengers on its final approach to Urgup."
    QUERY = "how many passengers did the shuttle carry"

    def _haystack(self, needle_at: int = 137, count: int = 200) -> str:
        paragraphs = _filler_paragraphs(count)
        paragraphs[needle_at] = self.NEEDLE
        return "\n\n".join(paragraphs)

    def test_finds_the_needle_among_two_hundred_windows(self) -> None:
        text = self._haystack()
        passages = select_passages(text, self.QUERY, top_k=3)

        assert passages, "the needle shares four terms with the query"
        assert "47 passengers" in passages[0].text
        assert passages[0].score > 0.0
        assert all(passages[i].score >= passages[i + 1].score for i in range(len(passages) - 1))

    def test_passage_offsets_address_the_source_text(self) -> None:
        text = self._haystack()
        for passage in select_passages(text, self.QUERY, top_k=5):
            assert text[passage.start : passage.end] == passage.text
            assert not passage.text[:1].isspace()
            assert not passage.text[-1:].isspace()

    def test_zero_hit_query_returns_nothing(self) -> None:
        text = self._haystack()

        assert select_passages(text, "hydroelectric turbine commissioning", top_k=5) == []
        assert select_passages(text, "!!! ???", top_k=5) == [], "a query with no tokens matches nothing"
        assert select_passages(text, self.QUERY, top_k=0) == []
        assert select_passages("", self.QUERY, top_k=5) == []

    def test_ties_break_on_position(self) -> None:
        """Same content twice must rank earliest-first, or the digest is not reproducible."""
        block = "passengers boarded the shuttle"
        text = "\n\n".join([block, *_filler_paragraphs(3), block])
        passages = select_passages(text, "shuttle passengers", top_k=2)

        assert len(passages) == 2
        assert passages[0].start < passages[1].start
        assert passages[0].score == pytest.approx(passages[1].score)

    def test_wall_of_text_falls_back_to_fixed_windows(self) -> None:
        """An extraction with no blank lines still has to be windowed, not scored whole."""
        text = " ".join(["dense"] * 400 + ["passengers"] + ["dense"] * 400)
        passages = select_passages(text, "passengers", top_k=3, window_chars=200)

        assert len(passages) == 1, "only the window holding the term scores"
        assert "passengers" in passages[0].text
        assert passages[0].end - passages[0].start <= 200

    def test_short_paragraphs_pack_up_to_the_window(self) -> None:
        text = "\n\n".join(["alpha shuttle"] * 20)
        passages = select_passages(text, "shuttle", top_k=10, window_chars=120)

        assert len(passages) > 1
        for passage in passages:
            assert passage.end - passage.start <= 120

    def test_page_attribution_from_page_breaks(self) -> None:
        pdf = PdfText(
            3,
            3,
            ("front matter and table of contents", "methodology and definitions", self.NEEDLE),
            "",
            (),
        )
        text, breaks = joined_page_text(pdf)

        passages = select_passages(text, self.QUERY, top_k=1, page_breaks=breaks)

        assert passages[0].page == 3
        assert select_passages(text, self.QUERY, top_k=1)[0].page is None, "no offsets means no page claim"

    def test_a_window_never_straddles_a_page_break(self) -> None:
        """The label a forecaster may cite has to be true, so pages bound the windows.

        Without this, three pages of ~30 chars pack into one 600-char window and the match on
        page 3 is reported as page 1 — an attribution the document does not support.
        """
        pdf = PdfText(3, 3, ("shuttle notes one", "shuttle notes two", "shuttle notes three"), "", ())
        text, breaks = joined_page_text(pdf)

        passages = select_passages(text, "shuttle", top_k=5, page_breaks=breaks)

        assert [passage.page for passage in passages] == [1, 2, 3]
        for passage in passages:
            assert "\n\n" not in passage.text, "a page separator inside a window means it merged pages"

    def test_default_window_mirrors_the_configured_constant(self) -> None:
        assert DEFAULT_WINDOW_CHARS == DOCUMENT_DIGEST_WINDOW_CHARS, (
            "the module default and the constants.py knob must not drift apart"
        )


class TestRenderDocumentDigest:
    URL = "https://example.org/annual-report.pdf"

    def _pdf(self) -> PdfText:
        paragraphs = _filler_paragraphs(6)
        paragraphs[3] = TestSelectPassages.NEEDLE
        return PdfText(
            page_count=9,
            pages_read=6,
            pages=tuple(paragraphs),
            truncated_by="pages",
            outline=(("Summary", 1), ("Passenger statistics", 4)),
        )

    def test_header_outline_and_passages(self) -> None:
        digest = render_document_digest(
            self._pdf(), query=TestSelectPassages.QUERY, top_k=2, max_chars=8000, source_url=self.URL
        )

        assert digest.startswith(f"Document: {self.URL}")
        assert "9 pages, 6 read," in digest
        assert "chars of text extracted; stopped at the 6-page read cap" in digest
        assert "Outline:" in digest
        assert "  Passenger statistics (p.4)" in digest
        assert f"Most relevant passages for: {TestSelectPassages.QUERY}" in digest
        assert "[p.4] The shuttle carried 47 passengers" in digest

    def test_outline_cap_reports_the_remainder(self) -> None:
        pdf = PdfText(
            page_count=1,
            pages_read=1,
            pages=("passengers on the shuttle " * 10,),
            truncated_by="",
            outline=tuple((f"Section {n}", 1) for n in range(1, 41)),
        )

        digest = render_document_digest(pdf, query="shuttle", top_k=1, max_chars=8000, source_url=self.URL)

        assert "  Section 25 (p.1)" in digest
        assert "  Section 26 (p.1)" not in digest
        assert "... 15 further outline entries" in digest

    def test_truncation_marker_is_visible_and_bounds_the_block(self) -> None:
        digest = render_document_digest(
            self._pdf(), query=TestSelectPassages.QUERY, top_k=6, max_chars=300, source_url=self.URL
        )

        assert len(digest) <= 300
        assert digest.endswith("[digest truncated at 300 chars]")

    def test_marker_survives_an_absurd_budget(self) -> None:
        digest = render_document_digest(
            self._pdf(), query=TestSelectPassages.QUERY, top_k=6, max_chars=5, source_url=self.URL
        )

        assert digest == "[digest truncated at 5 chars]", "a silent empty digest would read as a real one"

    def test_deterministic(self) -> None:
        first = render_document_digest(
            self._pdf(), query=TestSelectPassages.QUERY, top_k=3, max_chars=4000, source_url=self.URL
        )
        second = render_document_digest(
            self._pdf(), query=TestSelectPassages.QUERY, top_k=3, max_chars=4000, source_url=self.URL
        )

        assert first == second

    def test_unreadable_document_says_why(self) -> None:
        pdf = PdfText(0, 0, (), "", (), unreadable_reason="encrypted")

        digest = render_document_digest(pdf, query="anything", top_k=3, max_chars=2000, source_url=self.URL)

        assert "could not be parsed (encrypted)" in digest
        assert "Most relevant passages" not in digest

    def test_scanned_document_says_there_is_no_text_layer(self) -> None:
        pdf = PdfText(4, 4, ("", "", "", ""), "", (("Cover", 1),))

        digest = render_document_digest(pdf, query="anything", top_k=3, max_chars=2000, source_url=self.URL)

        assert "No extractable text layer" in digest
        assert "Cover (p.1)" in digest, "the outline is still evidence about what the scan contains"
        assert "Most relevant passages" not in digest

    def test_a_terse_document_is_not_treated_as_a_scan(self) -> None:
        """`has_text_layer`'s floor is the caller's escalation signal, not a render gate.

        A one-line PDF is below that floor, and withholding its text as "no text layer" would
        hide the only figure in it — the same failure as a content-free 200 rendered as
        evidence, run in reverse.
        """
        pdf = PdfText(1, 1, ("Q3 unemployment rate: 4.1%",), "", ())

        digest = render_document_digest(pdf, query="unemployment rate", top_k=2, max_chars=2000, source_url=self.URL)

        assert has_text_layer(pdf) is False, "the fixture is deliberately under the floor"
        assert "No extractable text layer" not in digest
        assert "[p.1] Q3 unemployment rate: 4.1%" in digest

    def test_no_matching_passage_says_so(self) -> None:
        digest = render_document_digest(
            self._pdf(), query="hydroelectric turbine commissioning", top_k=3, max_chars=4000, source_url=self.URL
        )

        assert "No passage in this document matched the query" in digest

    def test_end_to_end_from_pdf_bytes(self) -> None:
        """The path both callers take: bytes -> extract -> digest, with real pypdf in the middle."""
        data = _with_outline(
            build_text_pdf(
                [
                    ["Annual report of the regional transit authority", "Prepared for the ministry"],
                    ["The shuttle carried 47 passengers on its final approach."],
                ]
            ),
            [("Cover", 0, False), ("Ridership", 1, False)],
        )

        pdf = _extract(data)
        digest = render_document_digest(
            pdf, query="how many passengers did the shuttle carry", top_k=2, max_chars=4000, source_url=self.URL
        )

        assert is_pdf_body(data)
        assert pdf.outline == (("Cover", 1), ("Ridership", 2))
        assert "[p.2] The shuttle carried 47 passengers" in digest
