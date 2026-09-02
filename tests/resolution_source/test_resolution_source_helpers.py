"""Pure helpers of the Tier-1 resolution-source fetcher: URL extraction to formatter.

One of three modules split out of the old single ``test_resolution_source_provider.py``,
which had crossed 2,000 lines. The split follows that file's own three layers: helpers
here, the network layer in ``test_resolution_source_fetch.py``, and the provider factory
plus the SSRF guard in ``test_resolution_source_provider_gating.py``.

Real trafilatura runs on a fixed article-shaped HTML fixture, so the success path exercises
extraction end to end.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from metaculus_bot.research import resolution_source
from metaculus_bot.research.resolution_source import (
    FetchResult,
    extract_source_urls,
    format_resolution_sections,
    is_fred_url,
    is_metaculus_self_ref,
    is_yahoo_ticker_url,
    looks_like_csv_rows,
    looks_like_js_wall,
    select_fetchable_urls,
    strip_html_tags,
    strip_markdown_escapes,
    vacuous_body_status,
)


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


class TestStripHtmlTags:
    """Markup stripping for the RAW-body branches. Two properties matter: real tags go, and
    inequality signs in a data cell are NOT tags. The naive `</?[A-Za-z][^>]*>` form fails the
    second — it eats `x <a and y > b` down to `x  b`."""

    _VUUVZ_ROW = (
        '"8/16 - 8/17, 2026@@24335",'
        "<a href='https://emersoncollegepolling.com/august-2026-national-poll/'"
        "style='color:#000000; text-decoration: underline;'target='_blank' rel='nofollow noopener'>"
        "Emerson College</a>,"
        '"1,000 LV@@1000",1.108478,36.4,49.2,-12.8'
    )

    def test_a_live_poll_table_row_keeps_the_pollster_and_loses_the_markup(self):
        """The measured shape from the live VUUVz dataset (2026-08-26 receipts): a styled anchor
        per pollster row, 69% of that CSV's 33k chars being tag markup. The pollster name IS the
        content, so it stays and the tags go — 248 chars down to 84."""
        out = strip_html_tags(self._VUUVZ_ROW)

        assert "Emerson College" in out
        assert "<a " not in out
        assert "</a>" not in out
        assert "style=" not in out
        assert len(out) < len(self._VUUVZ_ROW) / 2.5

    @pytest.mark.parametrize(
        "cell",
        ["a < 5, b > 3", "x <a and y > b", "1 < 2 and 3 > 2", "temp < -40 or > 40"],
    )
    def test_inequalities_in_a_data_cell_are_untouched(self, cell: str):
        """`<a and y >` is why the tag NAME is an allow-list and an attribute region must contain
        an `=`: without both halves this eats real numeric data out of a dataset."""
        assert strip_html_tags(cell) == cell

    def test_a_body_with_no_angle_brackets_is_byte_identical(self):
        """The numeric tracker CSVs (1mU3g / kSCt4) contain zero `<` characters, so the strip must
        be a provable no-op there rather than merely a small one."""
        csv = "modeldate,approve,disapprove\n8/25/2026,36.41889,55.62032\n"
        assert strip_html_tags(csv) == csv

    def test_a_bare_link_cell_keeps_its_href_as_the_content(self):
        """An anchor with empty inner text carries its content in the href, so dropping the tag
        outright would delete the cell."""
        assert strip_html_tags("source,<a href='https://x.test/report'></a>") == "source,https://x.test/report"

    def test_an_unlisted_tag_name_is_left_alone(self):
        """The allow-list is closed: `<body>`/`<script>` never appear in a CSV cell, and matching
        every `<word>` is what makes the inequality cases above fail."""
        assert strip_html_tags("<body class='x'>hi</body>") == "<body class='x'>hi</body>"

    def test_a_pathological_no_close_tag_body_strips_in_linear_time(self):
        """The tag-body alternation must reach its first `=` exactly one way. With `[^<>]*` on
        both sides of the `=`, a body holding one `<b ` lookalike followed by an angle-bracket-free
        run of URL cells (query-string `=` signs, no closing `>`) backtracks quadratically: 3.4s at
        200 KiB measured, ~35 minutes at the 5 MiB response cap — synchronously on the event loop,
        wedging the sibling fetches past every wall timeout. The linear form is sub-millisecond
        here, so the 1s bound has three orders of magnitude of slack on either side."""
        body = "x <b " + ("url=https://example.test/p?q=1&r=2, " * 6000)
        start = time.perf_counter()
        out = strip_html_tags(body)
        elapsed = time.perf_counter() - start
        assert out == body, "`<b ` with no closing `>` names no tag — the body must be untouched"
        assert elapsed < 1.0, f"quadratic backtracking regression: {elapsed:.2f}s on a ~220 KiB body"


class TestLooksLikeCsvRows:
    """The precondition for the Tier-2 lead's `Dataset published <ts>` liveness claim."""

    def test_a_header_plus_a_row_is_a_dataset(self):
        assert looks_like_csv_rows("date,value\n2026-08-01,0.42\n") is True

    def test_a_header_alone_is_not(self):
        assert looks_like_csv_rows("date,value\n") is False

    def test_a_delimiterless_header_is_not(self):
        assert looks_like_csv_rows("Not Found\nThe requested chart is unavailable\n") is False

    def test_an_html_error_page_is_not_even_when_it_carries_commas(self):
        """A soft-404 page passes a bare delimiter test easily, which is why markup is rejected
        outright on the first non-blank line."""
        body = "<!DOCTYPE html>\n<html><head><title>404, not found</title></head>\n<body>gone</body>\n"
        assert looks_like_csv_rows(body) is False

    def test_tab_and_semicolon_delimiters_count(self):
        assert looks_like_csv_rows("date\tvalue\n2026-08-01\t0.42\n") is True
        assert looks_like_csv_rows("date;value\n2026-08-01;0.42\n") is True


class TestVacuousBodyStatus:
    """The one place "does this 200 body carry information?" is decided."""

    def test_content_returns_none(self):
        assert vacuous_body_status("date,value\n2026-08-01,0.42\n", 0.0, require_csv_rows=True) is None

    @pytest.mark.parametrize("body", ["", "   ", "\n\n\t"])
    def test_an_empty_or_whitespace_body_is_empty_body(self, body: str):
        assert vacuous_body_status(body, 0.0, require_csv_rows=False) == "empty_body"

    def test_an_undecodable_body_is_unsupported_type(self):
        assert vacuous_body_status("d\x00a\x00t\x00e\x00", 0.5, require_csv_rows=False) == "unsupported_type"

    def test_the_row_shape_requirement_is_dataset_only(self):
        """A cited JSON or plain-text page has no row shape to satisfy; only a dataset claiming to
        be a live series does."""
        assert vacuous_body_status('{"cve": "x"}', 0.0, require_csv_rows=False) is None
        assert vacuous_body_status('{"cve": "x"}', 0.0, require_csv_rows=True) == "unsupported_type"


class TestFetchResultInvariant:
    def test_a_success_with_blank_text_cannot_be_constructed(self):
        """The invariant the field comment always stated and nothing enforced. An empty 200 body
        shipped as `success` rendered an empty section under the "primary grading evidence"
        caveat, suppressed the all-failed notice for its siblings, and reported `ok` to provider
        diagnostics — so a future edit that reintroduces it should crash, not publish a hole."""
        with pytest.raises(ValueError, match="blank text"):
            FetchResult(url="https://x.test/a", status="success", text="   ", http_status=200, content_type="text/csv")

    def test_a_failure_with_blank_text_is_the_normal_case(self):
        assert FetchResult(url="https://x.test/a", status="empty_body", text="", http_status=200, content_type=None)


class TestFormatResolutionSections:
    def test_empty_results_returns_empty_string(self):
        assert format_resolution_sections([], datetime(2026, 7, 9, tzinfo=UTC)) == ""

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
        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))
        assert out  # no longer empty
        assert "2 resolution source(s) yielded no usable content" in out
        assert "a.com: blocked" in out
        assert "b.com: js_wall" in out
        assert "nothing from the cited resolving page(s) is in this bundle; weight other evidence accordingly" in out
        # Body only — the orchestrator prepends the "## Resolution Source Snapshot" header.
        assert "## Resolution Source Snapshot" not in out

    def test_an_empty_body_result_no_longer_suppresses_the_all_failed_notice(self):
        """The render half of the empty-200 defect. While an empty body counted as `success`, ONE
        such result put the section on the success path: it rendered an empty `### <url>` block
        under the primary-grading-evidence caveat, and the all-failed "yielded no usable content" notice
        — the whole point of which is to tell the forecaster to weight other evidence — was
        withheld for the sibling URLs that genuinely failed."""
        results = [
            FetchResult(
                url="https://empty.example.com/x",
                status="empty_body",
                text="",
                http_status=200,
                content_type="application/json",
            ),
            FetchResult(url="https://bad.com/y", status="blocked", text="", http_status=403, content_type=None),
        ]

        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))

        assert "2 resolution source(s) yielded no usable content" in out
        assert "empty.example.com: empty_body" in out
        assert "nothing from the cited resolving page(s) is in this bundle" in out
        assert "primary grading evidence" not in out
        assert "### https://empty.example.com/x" not in out

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
        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))
        # Success content still rendered.
        assert "### https://ok.com/data" in out
        assert "the reading is 3.2%" in out
        assert "primary grading evidence" in out
        # Terse note about the failed source appended.
        assert "bad.com: blocked" in out
        assert "other cited resolution source(s) yielded no usable content" in out
        # The success path must not carry the all-failed sentence.
        assert "nothing from the cited resolving page(s) is in this bundle" not in out

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
        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))
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
        out = resolution_source.format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))
        # First section fits; later ones must be trimmed or dropped.
        assert "https://example.com/0" in out
        # We should NOT see all four full 300-char blocks packed together.
        assert out.count("A" * 300) <= 2

    def test_a_budget_trim_leaves_a_visible_truncation_marker(self, monkeypatch):
        """The aggregate trim goes through the marker-emitting truncator, not a bare slice.

        A bare slice cut mid-sentence and could eat the per-URL ``[truncated at N chars ...]``
        marker the fetch already appended at the end — so an already-truncated page rendered
        as complete. Reachable on prod constants (5 x 6000 per-URL against an 18000 total).
        """
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_TOTAL_MAX_CHARS", 400)
        results = [
            FetchResult(
                url="https://example.com/long",
                status="success",
                text="B" * 5000 + "\n[truncated at 5000 chars — full source at https://example.com/long]",
                http_status=200,
                content_type="text/html",
            )
        ]

        out = resolution_source.format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))

        # The section is cut, and the cut says so rather than ending mid-body.
        assert "[truncated at 400 chars — full source at https://example.com/long]" in out
        assert "B" * 5000 not in out

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
        out = resolution_source.format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))
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
        out = format_resolution_sections(results, datetime(2026, 7, 9, tzinfo=UTC))
        assert "omitted" not in out
