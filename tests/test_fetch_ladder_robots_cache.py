"""Robots pre-check behavior when the shared fetch ladder already holds the file."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from metaculus_bot.research.agentic import tools as agentic_tools
from metaculus_bot.research.fetch_ladder import run_cache
from metaculus_bot.research.robots_policy import reset_robots_cache, robots_txt_url

_DOCUMENT_URL = "https://example.com/private/report.pdf"
_ROBOTS_BODY = "User-agent: Google-Extended\nDisallow: /private/"


@pytest.fixture(autouse=True)
def _clear_robots_policy_cache() -> Iterator[None]:
    reset_robots_cache()
    yield
    reset_robots_cache()


@pytest.mark.asyncio
async def test_cached_robots_txt_still_blocks_the_paid_reader() -> None:
    robots_url = robots_txt_url(_DOCUMENT_URL)
    run_cache.put(
        robots_url,
        run_cache.TextRead(
            url=robots_url,
            text=_ROBOTS_BODY,
            http_status=200,
            content_type="text/plain",
        ),
        route="direct",
    )

    assert await agentic_tools._fetch_robots_txt(robots_url) == _ROBOTS_BODY
    assert await agentic_tools._url_context_robots_skip(_DOCUMENT_URL) is True
