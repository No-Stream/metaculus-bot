"""The forecaster-facing route caveats every non-direct rung contributes to the section."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import get_args

import pytest

from metaculus_bot.research import resolution_source
from metaculus_bot.research.resolution_fetch_result import ROUTE_CAVEATS, FetchResult, FetchRoute
from metaculus_bot.research.resolution_source import (
    format_resolution_sections,
)
from tests.resolution_source_fakes import (
    _URL,
)


def _success(route: FetchRoute, url: str = _URL, text: str = "body text") -> FetchResult:
    return FetchResult(url=url, status="success", text=text, http_status=200, content_type="text/html", route=route)


class TestRouteCaveats:
    """A forecaster reading a rescued section has to be able to tell how it was obtained, since
    the heading above it calls the whole snapshot primary grading evidence."""

    _AT = datetime(2026, 9, 4, tzinfo=UTC)

    def test_an_all_direct_question_renders_byte_identically(self):
        """The overwhelming majority of questions, and the case a diff against the archive has
        to keep clean: a route that adds nothing is unrepresentable in the mapping."""
        rendered = format_resolution_sections([_success("direct")], self._AT)

        assert rendered.splitlines()[0] == (
            "Snapshot of the cited resolution source(s) as of 2026-09-04 — primary grading evidence."
        )
        assert rendered.splitlines()[1] == ""
        for caveat in ROUTE_CAVEATS.values():
            assert caveat not in rendered

    def test_every_route_token_except_direct_has_a_caveat(self):
        """The completeness guard, and the reason `impersonate` carries a sentence for a rung
        that has not shipped: a future route token added to `FetchRoute` without a caveat would
        otherwise render rescued content with no disclosure at all, silently."""
        routes = set(get_args(FetchRoute))

        assert routes - {"direct"} == set(ROUTE_CAVEATS)
        assert "direct" not in ROUTE_CAVEATS

    @pytest.mark.parametrize("route", sorted(ROUTE_CAVEATS))
    def test_every_non_direct_route_says_how_its_section_was_obtained(self, route):
        rendered = format_resolution_sections([_success(route)], self._AT)

        assert ROUTE_CAVEATS[route] in rendered

    def test_one_sentence_per_route_present_in_stable_order(self):
        results = [
            _success("wayback", "https://a.example.com/p"),
            _success("rendered", "https://b.example.com/p"),
            _success("wayback", "https://c.example.com/p"),
        ]

        rendered = format_resolution_sections(results, self._AT)

        # Deduped, and ordered by the mapping rather than by fetch order.
        assert rendered.count(ROUTE_CAVEATS["wayback"]) == 1
        assert rendered.index(ROUTE_CAVEATS["rendered"]) < rendered.index(ROUTE_CAVEATS["wayback"])

    def test_a_section_dropped_by_the_budget_adds_no_caveat(self, monkeypatch):
        """The caveat describes an artifact a forecaster can SEE. Computed over every success, it
        told forecasters a section below was rendered in a browser when the aggregate budget had
        already dropped that section (reproduced on prod constants: 5 x 6000 per-URL pages
        against an 18000 total, with the rendered page cited last). The first page fills the
        whole total, so the second lands on a zero remainder, under the section floor."""
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_TOTAL_MAX_CHARS", 400)
        first = FetchResult(
            url="https://a.example.com/p", status="success", text="x" * 600, http_status=200, content_type="text/html"
        )
        rendered_last = _success("rendered", "https://b.example.com/p")

        rendered = format_resolution_sections([first, rendered_last], self._AT)

        assert "[1 additional source(s) omitted — section budget]" in rendered
        assert "### https://b.example.com/p" not in rendered
        assert ROUTE_CAVEATS["rendered"] not in rendered

    def test_a_kept_section_keeps_its_caveat_when_a_sibling_is_dropped(self, monkeypatch):
        monkeypatch.setattr(resolution_source, "RESOLUTION_SOURCE_TOTAL_MAX_CHARS", 400)
        rendered_first = _success("rendered", "https://b.example.com/p", text="y" * 400)
        dropped = FetchResult(
            url="https://a.example.com/p", status="success", text="x" * 600, http_status=200, content_type="text/html"
        )

        rendered = format_resolution_sections([rendered_first, dropped], self._AT)

        assert "### https://b.example.com/p" in rendered
        assert ROUTE_CAVEATS["rendered"] in rendered
        assert "[1 additional source(s) omitted — section budget]" in rendered
        assert "### https://a.example.com/p" not in rendered

    def test_a_failed_rung_adds_no_caveat(self):
        """A caveat describes an artifact a forecaster can see; a rung that fired and failed left
        the direct outcome, which the failure notice already names."""
        failed = FetchResult(
            url=_URL, status="js_wall", text="", http_status=200, content_type="text/html", route="rendered"
        )

        rendered = format_resolution_sections([_success("direct", "https://ok.example.com/p"), failed], self._AT)

        assert ROUTE_CAVEATS["rendered"] not in rendered
