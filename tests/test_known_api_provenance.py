"""Verification tiers for deterministic known-API tool results."""

from metaculus_bot.research.agentic.provenance import _harvest_verification_tiers
from metaculus_bot.research.agentic.types import ToolOutcome


def _known_api_outcome() -> ToolOutcome:
    return ToolOutcome(
        content_markdown=(
            "### DGS10\n"
            "Source: https://fred.stlouisfed.org/series/DGS10\n"
            "Mentioned only in prose: https://example.com/unread"
        ),
        links=[
            "https://fred.stlouisfed.org/series/DGS10",
            "javascript:alert(1)",
        ],
        method="known_api",
        status="ok",
    )


def test_explicit_known_api_tool_tiers_only_backend_owned_links() -> None:
    assert _harvest_verification_tiers("fred_series", {"series_id": "DGS10"}, _known_api_outcome()) == {
        "https://fred.stlouisfed.org/series/DGS10": "fetched"
    }


def test_fetch_via_known_api_still_tiers_only_the_requested_url() -> None:
    requested = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"

    assert _harvest_verification_tiers("fetch", {"url": requested}, _known_api_outcome()) == {requested: "fetched"}
