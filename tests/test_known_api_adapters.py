"""The two adapters: a known-API result to the loop's ToolOutcome and to the fetcher's FetchResult.

The adapters are the whole coupling to the two callers; the wiring step adds the ``known_api``
method tier and the ``known_api`` FetchRoute member. Until then the route is passed in.
"""

from __future__ import annotations

import pytest

from metaculus_bot.research.known_api import adapters
from metaculus_bot.research.known_api.result import KnownApiResult


def _ok() -> KnownApiResult:
    return KnownApiResult(
        status="ok",
        content_markdown="### DGS30\nLatest: 4.85",
        source_url="https://fred.stlouisfed.org/series/DGS30",
        links=["https://fred.stlouisfed.org/series/DGS30"],
    )


class TestToToolOutcome:
    def test_ok_maps_to_a_known_api_outcome(self):
        outcome = adapters.to_tool_outcome(_ok())
        assert outcome.method == "known_api"
        assert outcome.status == "ok"
        assert "DGS30" in outcome.content_markdown
        assert outcome.links == ["https://fred.stlouisfed.org/series/DGS30"]

    @pytest.mark.parametrize("status", ["empty", "not_found", "error"])
    def test_failure_statuses_pass_through(self, status: str):
        result = KnownApiResult(status=status, content_markdown="nope", source_url="")  # type: ignore[arg-type]
        outcome = adapters.to_tool_outcome(result)
        assert outcome.status == status
        assert outcome.method == "known_api"


class TestToFetchResult:
    def test_ok_maps_to_success(self):
        fetch = adapters.to_fetch_result(_ok(), url="https://fred.stlouisfed.org/series/DGS30")
        assert fetch.status == "success"
        assert fetch.text == "### DGS30\nLatest: 4.85"
        assert fetch.route == "known_api"
        assert fetch.url == "https://fred.stlouisfed.org/series/DGS30"

    def test_not_found_maps_without_text(self):
        result = KnownApiResult(status="not_found", content_markdown="no such series", source_url="")
        fetch = adapters.to_fetch_result(result, url="https://fred.stlouisfed.org/series/NOPE")
        assert fetch.status == "not_found"
        assert fetch.text == ""

    def test_empty_maps_to_no_resolving_content(self):
        result = KnownApiResult(status="empty", content_markdown="", source_url="")
        fetch = adapters.to_fetch_result(result, url="https://x")
        assert fetch.status == "no_resolving_content"

    def test_route_is_a_parameter_until_the_enum_gains_the_member(self):
        fetch = adapters.to_fetch_result(_ok(), url="https://x", route="direct")
        assert fetch.route == "direct"
