"""Opt-in integration tests for the prediction-market provider.

These hit real APIs (Polymarket Gamma, Kalshi, Manifold). Skipped in CI by
default; run locally with `pytest -m integration tests/test_prediction_market_integration.py`
when validating after a schema drift or initial implementation.

Each test uses a simple, high-signal query and asserts that the response
parses. Skips gracefully on transient network errors (aiohttp.ClientError,
asyncio.TimeoutError) so a 5xx blip or slow upstream doesn't break a dev
loop. Genuine schema breaks — pydantic validation errors, AttributeError on
missing fields, etc. — are NOT caught and will fail loud, which is the point
of the test.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock

import aiohttp
import pytest

pytestmark = pytest.mark.integration


def _build_integration_question() -> MagicMock:
    q = MagicMock()
    q.id_of_question = 99999
    q.question_text = "Will Donald Trump remain US president through 2026?"
    q.title = "Will Donald Trump remain US president through 2026?"
    q.short_title = "Trump president 2026"
    q.resolution_criteria = "Resolves Yes if Donald Trump is still in office on December 31, 2026."
    q.page_url = "https://metaculus.com/q/99999"
    return q


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason="set RUN_INTEGRATION_TESTS=1 to enable")
async def test_polymarket_real_search_returns_parseable_response():
    from metaculus_bot.prediction_market_provider import _polymarket_search

    async with aiohttp.ClientSession() as session:
        try:
            matches = await _polymarket_search(session, "Trump president 2026")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            pytest.skip(f"Polymarket transient error: {e}")

    if not matches:
        pytest.skip("No matches from Polymarket (transient or zero-result)")

    m0 = matches[0]
    assert m0.platform == "polymarket"
    assert m0.market_title
    assert m0.implied_prob_yes is not None
    assert 0.0 <= m0.implied_prob_yes <= 1.0


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason="set RUN_INTEGRATION_TESTS=1 to enable")
async def test_manifold_real_search_returns_parseable_response():
    from metaculus_bot.prediction_market_provider import _manifold_search

    async with aiohttp.ClientSession() as session:
        try:
            matches = await _manifold_search(session, "Trump president 2026")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            pytest.skip(f"Manifold transient error: {e}")

    if not matches:
        pytest.skip("No matches from Manifold (transient or zero-result)")

    m0 = matches[0]
    assert m0.platform == "manifold"
    assert m0.market_title
    assert m0.implied_prob_yes is not None
    assert 0.0 <= m0.implied_prob_yes <= 1.0


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason="set RUN_INTEGRATION_TESTS=1 to enable")
async def test_kalshi_real_prefetch_and_search_returns_parseable_response():
    from metaculus_bot.prediction_market_provider import _kalshi_prefetch_events, _kalshi_search_local

    async with aiohttp.ClientSession() as session:
        try:
            events = await _kalshi_prefetch_events(session, event_limit=500, page_sleep_s=1.0)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            pytest.skip(f"Kalshi prefetch transient error: {e}")

    if not events:
        pytest.skip("Kalshi prefetch returned no events (transient)")

    matches = _kalshi_search_local(events, "Trump president 2026", top_k=3, min_score=30.0)
    # It's OK if matches is empty -- Kalshi coverage is sparse for some topics.
    for m in matches:
        assert m.platform == "kalshi"
        if m.implied_prob_yes is not None:
            assert 0.0 <= m.implied_prob_yes <= 1.0


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason="set RUN_INTEGRATION_TESTS=1 to enable")
async def test_full_orchestrator_against_real_apis():  # noqa: ASYNC910
    from metaculus_bot import prediction_market_provider as pmp

    q = _build_integration_question()
    pmp._reset_session_caches()

    try:
        snapshot = await pmp.fetch_market_snapshot(q, timeout=30.0)
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        pytest.skip(f"Orchestrator transient error: {e}")

    # Zero matches is acceptable -- skip rather than fail on a sparse-topic day.
    if not snapshot.matches:
        pytest.skip("No matches across any platform (transient or zero-result)")

    for m in snapshot.matches:
        assert m.platform in ("polymarket", "kalshi", "manifold")
        assert m.market_title
