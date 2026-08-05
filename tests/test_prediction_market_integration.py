"""Opt-in integration tests for the prediction-market provider.

These hit real APIs (Polymarket Gamma, Kalshi, Manifold, PredictIt). All four are
FREE, unauthenticated, public read endpoints — no key, no spend — so they are outside
the repo's cost gate. They stay opt-in on `RUN_INTEGRATION_TESTS` only because a
network round-trip does not belong in a dev loop, not because they cost anything.

**The two LLM stages are MOCKED even here, and that is a cost-gate requirement rather
than convenience.** The full-pipeline test drives the real venues but patches
`_invoke_market_llm`, so no test in this file can ever bill an OpenRouter key. A live
ranking call is a priced step the operator fires deliberately, not something a test
turns on by setting an env var.

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
import json
import os
from unittest.mock import MagicMock, patch

import aiohttp
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.allow_network]

# Canned completions for the two LLM stages, in the shapes their parsers accept. The ranker
# array is empty on purpose: `[]` is a VALID answer (it is the adaptive-width mechanism), it
# needs no knowledge of how many candidates today's live catalogues produced, and it exercises
# the parse-and-apply path without pretending to be a judgment.
_CANNED_QUERY_AUTHOR = json.dumps({"synonyms": ["POTUS", "White House"], "framings": ["Trump stays president"]})
_CANNED_RANKING = "[]"


async def _canned_market_llm(config: dict, prompt: str, **_kwargs: object) -> str:
    """Stand-in for `_invoke_market_llm`, routing on the prompt each stage builds."""
    return _CANNED_RANKING if "Rank the candidates by EVIDENTIAL VALUE" in prompt else _CANNED_QUERY_AUTHOR


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
    from metaculus_bot.research.market_retrieval.venues import polymarket_search

    async with aiohttp.ClientSession() as session:
        try:
            matches = await polymarket_search(session, "Trump president 2026", width=10)
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
    from metaculus_bot.research.market_retrieval.venues import manifold_search

    async with aiohttp.ClientSession() as session:
        try:
            matches = await manifold_search(session, "Trump president 2026", width=10)
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
    from metaculus_bot.research.market_retrieval.venues import manifold_search

    async with aiohttp.ClientSession() as session:
        try:
            baseline = await manifold_search(session, "gas prices", width=10)
            with_impossible_token = await manifold_search(session, "gas prices zzzqqqxyz", width=10)
            reordered = await manifold_search(session, "prices gas", width=10)
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
    from metaculus_bot.research.market_retrieval.venues import kalshi_event_match, kalshi_prefetch_events

    async with aiohttp.ClientSession() as session:
        try:
            pull = await kalshi_prefetch_events(session, event_limit=500, max_pages=3)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            pytest.skip(f"Kalshi prefetch transient error: {e}")

    if not pull.events:
        pytest.skip("Kalshi prefetch returned no events (transient)")

    # Liquidity-field contract, checked against the LIVE payload rather than a fixture.
    # This is the check that was missing: the previous version asserted only `platform`
    # and a probability range, so a Kalshi field rename (or a parser reading a name that
    # never existed) passed here indefinitely. Every open Kalshi market carries volume and
    # open interest — 1,504/1,504 on 2026-08-03 — so a venue-wide absence is a real break.
    #
    # The pull PROJECTS each page down as it streams, so a field missing here means either
    # Kalshi renamed it or the projection stopped retaining it — both are the same defect from
    # the parser's point of view, and both blank the rendered `signal` column.
    nested = [mkt for ev in pull.events for mkt in (ev.get("markets") or []) if isinstance(mkt, dict)]
    assert nested, "Kalshi events carried no nested markets"
    assert any(mkt.get("volume_fp") is not None for mkt in nested), (
        "Kalshi volume_fp reached no projected market — either upstream renamed it or the projection dropped it"
    )
    assert any(mkt.get("open_interest_fp") is not None for mkt in nested), (
        "Kalshi open_interest_fp reached no projected market — upstream rename or a dropped projection field"
    )

    rows = [kalshi_event_match(ev, match_confidence=1.0, channel="universe_fuzzy") for ev in pull.events]
    matches = [row for row in rows if row is not None]
    assert matches, "no candidate row survived from a live catalogue page"
    for m in matches[:20]:
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
    every market becomes a candidate row. Schema breaks fail loud; transient errors skip."""
    from metaculus_bot.research.market_retrieval.venues import predictit_market_match, predictit_prefetch

    async with aiohttp.ClientSession() as session:
        try:
            markets = await predictit_prefetch(session)
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

    rows = [predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy") for market in markets]
    matches = [row for row in rows if row is not None]
    assert matches, "the whole PredictIt universe must reach the pool — there is no pre-filter"
    for m in matches[:20]:
        assert m.platform == "predictit"
        assert m.market_title
        # No price: a design that hands the whole universe to a ranker has no per-question query
        # to select a contract on, so pricing an arbitrary one would misreport the market.
        assert m.implied_prob_yes is None
        # PredictIt exposes no liquidity fields.
        assert m.total_volume is None
        assert m.open_interest is None


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("RUN_INTEGRATION_TESTS"), reason=_SKIP_REASON)
async def test_full_orchestrator_against_real_apis():  # noqa: ASYNC910
    from metaculus_bot.research import prediction_market as pmp

    q = _build_integration_question()
    pmp._reset_session_caches()

    # The LLM stages are patched, NOT enabled: see the module docstring. This test's job is the
    # live venue surface and the pool it produces, and the canned ranking returns `[]` so no
    # rows render — which is why the assertions below are about the POOL, read out of the
    # MARKET_RANKING telemetry line, rather than about rendered matches.
    with patch.object(pmp, "_invoke_market_llm", _canned_market_llm):
        try:
            snapshot = await pmp.fetch_market_snapshot(q, timeout=pmp.PREDICTION_MARKET_TIMEOUT)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            pytest.skip(f"Orchestrator transient error: {e}")

    # Every source must have reported, and none of the four venues may be a loss: this is the
    # live check that the seven-key diagnostics dict still describes a working pipeline.
    assert set(snapshot.sources) == {
        "polymarket",
        "manifold",
        "kalshi",
        "predictit",
        "manifold_detail",
        "query_author",
        "ranking",
    }, snapshot.sources
    for venue in ("polymarket", "manifold", "kalshi", "predictit"):
        assert snapshot.sources[venue].startswith(("ok", "none")), f"{venue} lost a sub-fetch: {snapshot.sources}"
    assert snapshot.sources["ranking"] == "ok(0)", snapshot.sources

    for m in snapshot.matches:
        assert m.platform in ("polymarket", "kalshi", "manifold", "predictit")
        assert m.market_title
