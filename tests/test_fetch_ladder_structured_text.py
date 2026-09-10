from __future__ import annotations

import pytest

from metaculus_bot.research.fetch_ladder import classify
from metaculus_bot.research.fetch_ladder.context import LadderContext
from metaculus_bot.research.fetch_ladder.policy import GAP_FILL_FETCH_POLICY, RESOLUTION_SOURCE_POLICY, LadderPolicy
from metaculus_bot.research.resolution_fetch_result import FetchResult

_TREASURY_ATOM_FEED = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Daily Treasury Par Yield Curve Rates</title>
  <entry>
    <content type="application/xml">
      <properties>
        <Date>2026-09-09T00:00:00</Date>
        <BC_10YEAR>4.08</BC_10YEAR>
      </properties>
    </content>
  </entry>
</feed>
"""


@pytest.mark.parametrize(
    "policy",
    [RESOLUTION_SOURCE_POLICY, GAP_FILL_FETCH_POLICY],
    ids=["resolution_source", "gap_fill_v2"],
)
@pytest.mark.parametrize("content_type", ["text/xml", "application/xml", "application/atom+xml"])
def test_both_caller_policies_route_xml_media_types_as_text(policy: LadderPolicy, content_type: str) -> None:
    assert policy.verdict.body_route(content_type, _TREASURY_ATOM_FEED.encode()) == "text"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy",
    [RESOLUTION_SOURCE_POLICY, GAP_FILL_FETCH_POLICY],
    ids=["resolution_source", "gap_fill_v2"],
)
async def test_treasury_xml_is_returned_verbatim_without_html_tag_stripping(policy: LadderPolicy) -> None:
    result = await classify._classify_body(
        _TREASURY_ATOM_FEED.encode(),
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml",
        "application/atom+xml; charset=utf-8",
        LadderContext(policy=policy),
        http_status=200,
    )

    assert isinstance(result, FetchResult)
    assert result.status == "success"
    assert result.text == _TREASURY_ATOM_FEED
    assert "<BC_10YEAR>4.08</BC_10YEAR>" in result.text
