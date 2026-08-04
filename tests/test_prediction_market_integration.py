"""Opt-in integration tests for the prediction-market provider.

These hit real APIs (Polymarket Gamma, Kalshi, Manifold, PredictIt). All four are
FREE, unauthenticated, public read endpoints — no key, no spend — so they are outside
the repo's cost gate. They stay opt-in on `RUN_INTEGRATION_TESTS` only because a
network round-trip does not belong in a dev loop, not because they cost anything.

Run them with:

    RUN_INTEGRATION_TESTS=1 uv run pytest -m integration tests/test_prediction_market_integration.py

`allow_network` is REQUIRED here and is what makes the env gate real. The autouse
`_block_network_egress` guard in `tests/conftest.py` blocks every non-localhost connect
for any test carrying neither `allow_network` nor `live`; without the marker these tests
died on a blocked socket even with the env var set, so the whole file was unrunnable and
its assertions were decoration. That is the same failure shape this file exists to catch
— a check that looks live and cannot fire — so the marker is load-bearing, not hygiene.

Each test uses a simple, high-signal query and asserts that the response parses. Skips
gracefully on transient network errors (aiohttp.ClientError, asyncio.TimeoutError) so a
5xx blip or slow upstream doesn't break a dev loop. Genuine schema breaks — a renamed
field, a moved nesting level, a changed search contract — are NOT caught and fail loud,
which is the point.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock

import aiohttp
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.allow_network]

# Spelled out so a run log states WHY this is off rather than just that it is: these
# calls are free (four public unauthenticated read endpoints), so the gate is about
# keeping a network round-trip out of the dev loop and out of CI's failure surface,
# NOT about cost or credentials. Schema-drift coverage against the venues' real
# payloads is enforced unconditionally by tests/test_prediction_market_provider.py
# against the committed verbatim payload trims in tests/data/, so what this file
# uniquely adds is "the live API still answers this shape TODAY".
_SKIP_REASON = (
    "opt-in live-API check: set RUN_INTEGRATION_TESTS=1 to enable. Free (public "
    "unauthenticated endpoints) but network-dependent, so it is off by default; parser "
    "coverage against committed real payloads runs unconditionally in "
    "tests/test_prediction_market_provider.py."
)


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
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
async def test_polymarket_real_search_returns_parseable_response():
    from metaculus_bot.research.prediction_market import _polymarket_search

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
    assert any(m.total_volume is not None for m in matches), "Polymarket parsed no volume on any match"
    # The D1 sibling: Gamma carries `openInterest` at the EVENT level, and reading it off the
    # nested market left the OI column blank on 142/142 archived rows. Asserted separately
    # from volume because volume was never the broken leg — a volume-only check passes on the
    # defect. Not `all()`: Gamma genuinely omits openInterest on some events, so the contract
    # is "the event-level read reaches at least one match", which is what a market-level-only
    # parser fails.
    assert any(m.open_interest is not None for m in matches), (
        "Polymarket parsed no open interest on any match — the event-level openInterest read regressed"
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
async def test_manifold_real_search_returns_parseable_response():
    from metaculus_bot.research.prediction_market import _manifold_search

    async with aiohttp.ClientSession() as session:
        try:
            matches = await _manifold_search(session, "Trump president 2026")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            pytest.skip(f"Manifold transient error: {e}")

    # No skip-on-empty here, deliberately. This test used to
    # `pytest.skip("No matches from Manifold (transient or zero-result)")`, which is how
    # a venue that had returned zero rows for 17 straight days stayed invisible: the one
    # live check treated the failure mode as a reason not to run. "Trump president 2026"
    # is 3 content tokens, comfortably inside Manifold's strict-conjunction limit, so an
    # empty result here is a real defect.
    assert matches, "Manifold returned nothing for a 3-token query — the search contract changed"

    m0 = matches[0]
    assert m0.platform == "manifold"
    assert m0.market_title
    assert m0.implied_prob_yes is not None
    assert 0.0 <= m0.implied_prob_yes <= 1.0


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
async def test_manifold_search_term_is_still_a_strict_conjunction():
    """Tripwire on the UPSTREAM ASSUMPTION the relaxation ladder rests on.

    Manifold documents `term` only as "The search query in question", so the conjunction
    semantics are measured, not specified. If upstream ever moves to ranked search, this
    test tells us the ladder is now unnecessary rather than letting it silently over-fit.

    The discriminator is order-invariance plus one-absent-token sensitivity: appending a
    token that appears in no market zeroes the result, while reordering present tokens does
    not. A relevance floor would behave the opposite way.
    """
    from metaculus_bot.research.prediction_market import _manifold_search

    async with aiohttp.ClientSession() as session:
        try:
            baseline = await _manifold_search(session, "gas prices")
            with_impossible_token = await _manifold_search(session, "gas prices zzzqqqxyz")
            reordered = await _manifold_search(session, "prices gas")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            pytest.skip(f"Manifold transient error: {e}")

    if baseline is None or with_impossible_token is None or reordered is None:
        pytest.skip("Manifold fetch failed (transient)")

    assert baseline, "'gas prices' matched nothing — Manifold's corpus or search changed"
    assert with_impossible_token == [], (
        "Appending an impossible token no longer zeroes the result: Manifold's `term` is no "
        "longer a strict conjunction, so manifold_relaxation_terms may now be unnecessary."
    )
    assert reordered, "Reordering present tokens changed the result — semantics are not set intersection"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
async def test_kalshi_real_prefetch_and_search_returns_parseable_response():
    from metaculus_bot.research.prediction_market import _kalshi_prefetch_events, _kalshi_search_local

    async with aiohttp.ClientSession() as session:
        try:
            events, _tally = await _kalshi_prefetch_events(session, event_limit=500, page_sleep_s=1.0)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            pytest.skip(f"Kalshi prefetch transient error: {e}")

    if not events:
        pytest.skip("Kalshi prefetch returned no events (transient)")

    # Liquidity-field contract, checked against the LIVE payload rather than a fixture.
    # This is the check that was missing: the previous version asserted only `platform`
    # and a probability range, so a Kalshi field rename (or a parser reading a name that
    # never existed) passed here indefinitely. Every open Kalshi market carries volume and
    # open interest — 1,504/1,504 on 2026-08-03 — so a venue-wide absence is a real break.
    nested = [mkt for ev in events for mkt in (ev.get("markets") or []) if isinstance(mkt, dict)]
    assert nested, "Kalshi events carried no nested markets"
    assert any("volume_fp" in mkt for mkt in nested), "Kalshi no longer publishes volume_fp — parser needs updating"
    assert any("open_interest_fp" in mkt for mkt in nested), (
        "Kalshi no longer publishes open_interest_fp — parser needs updating"
    )

    matches = _kalshi_search_local(events, "Trump president 2026", top_k=3, min_score=30.0)
    # It's OK if matches is empty -- Kalshi coverage is sparse for some topics.
    for m in matches:
        assert m.platform == "kalshi"
        if m.implied_prob_yes is not None:
            assert 0.0 <= m.implied_prob_yes <= 1.0
        # A real-money venue must yield a real-money figure; `no-liquidity-data` on a live
        # Kalshi row is the D1 defect recurring.
        assert m.total_volume is not None or m.open_interest is not None, (
            f"Kalshi match {m.market_title!r} parsed no liquidity — field names drifted again"
        )


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
async def test_predictit_real_prefetch_and_search_returns_parseable_response():
    """PredictIt's /marketdata/all/ is free + no-auth. Assert the schema
    (top-level `markets` list, contracts carrying `lastTradePrice`) and that a
    US-politics query fuzzy-matches. Schema breaks fail loud; transient errors
    and sparse-topic zero-results skip."""
    from metaculus_bot.research.prediction_market import _predictit_prefetch, _predictit_search_local

    async with aiohttp.ClientSession() as session:
        try:
            markets = await _predictit_prefetch(session)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            pytest.skip(f"PredictIt prefetch transient error: {e}")

    if not markets:
        pytest.skip("PredictIt prefetch returned no markets (transient)")

    # Schema contract: every market carries a name and a contract list; contracts
    # carry lastTradePrice. Fail loud on a schema break.
    sample = markets[0]
    assert isinstance(sample, dict)
    assert sample.get("name") or sample.get("shortName")
    contracts = sample.get("contracts")
    assert isinstance(contracts, list) and contracts
    assert "lastTradePrice" in contracts[0]

    matches = _predictit_search_local(markets, "Trump president 2026", top_k=3, min_score=30.0)
    for m in matches:
        assert m.platform == "predictit"
        assert m.market_title
        if m.implied_prob_yes is not None:
            assert 0.0 <= m.implied_prob_yes <= 1.0
        # PredictIt exposes no liquidity fields.
        assert m.total_volume is None
        assert m.open_interest is None


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
async def test_full_orchestrator_against_real_apis():  # noqa: ASYNC910
    from metaculus_bot.research import prediction_market as pmp

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
        assert m.platform in ("polymarket", "kalshi", "manifold", "predictit")
        assert m.market_title
