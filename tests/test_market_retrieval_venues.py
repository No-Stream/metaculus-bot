"""The four venue fetch/parse paths, at the four decisions the ranked pipeline rests on.

Each of these is a place where a plausible-looking simplification stays green while quietly
costing recall or misreporting a row:

- The Kalshi ``/events`` stream projects each page down to exactly the retained fields, and a
  200 carrying ``{"events": null}`` reads as a LOST catalogue rather than an empty one — the
  failure that would otherwise pin an empty index for a 6h TTL.
- An event's ``close_time`` is the MAX over its nested markets and it is resolved only when
  ALL of them are. Reading ``nested[0]`` instead misclassifies ~305 live event families as
  RESOLVED on the frozen universe, and both fields are a rendered column and a ranker input.
- PredictIt derives ``close_time`` and ``is_resolved`` from EVERY contract, through a date
  guard that survives the ``NA``/``N/A`` sentinels the dump ships.
- Polymarket sends ``events_status=active``; nothing downstream filters on close date any
  more, so without it a search puts mostly-closed events straight into the pool.

Payload fixtures are real captures, deliberately: the 2026-07-12 Kalshi liquidity regression
shipped because a hand-written fixture was authored from the same wrong belief as the parser,
so the pair agreed and the suite stayed green.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from metaculus_bot.research.market_retrieval import venues
from metaculus_bot.research.market_retrieval.http import (
    flatten_results,
    parse_iso,
    parse_iso_guarded,
    safe_float,
)
from metaculus_bot.research.market_retrieval.types import MarketMatch, _liquidity_label
from tests.test_prediction_market_provider import FakeResponse, FakeSession

_PAYLOADS_PATH = Path(__file__).parent / "data" / "prediction_market_venue_payloads.json"
_MULTI_CLOSE_PATH = Path(__file__).parent / "data" / "kalshi_multi_close_event_2026_08_04.json"


@pytest.fixture(scope="module")
def captured_payloads() -> dict[str, Any]:
    return json.loads(_PAYLOADS_PATH.read_text())


@pytest.fixture(scope="module")
def multi_close_page() -> dict[str, Any]:
    return json.loads(_MULTI_CLOSE_PATH.read_text())


def _event(page: dict[str, Any], ticker: str) -> dict[str, Any]:
    for event in page["events"]:
        if event.get("event_ticker") == ticker:
            return copy.deepcopy(event)
    raise AssertionError(f"fixture carries no event {ticker!r}")


class TestKalshiEventProjection:
    """The streamed ``/events`` projection: what it keeps, and what it refuses to cache."""

    @pytest.mark.asyncio
    async def test_projection_retains_exactly_the_listed_fields_per_tier(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """A raw market object carries ~38 keys and the catalogue is held for 6h, so the
        projection is the memory bound. This asserts the DROP, not just the keep — which is why
        the fixture is a raw page rather than a pre-projected one.

        Both TIERS are pinned. Only `close_time` and `status` are read across nested markets, so
        every market after the first keeps just those two; 87.6% of the retained price/rules
        fields were dead weight held for the full TTL. Widening the tail silently would double the
        cache again, and narrowing the head would break the price/liquidity legs, which read
        `nested[0]`. The fixture's 4-market event is what makes the two tiers distinguishable.
        """
        raw_market_keys = set(multi_close_page["events"][0]["markets"][0])
        assert len(raw_market_keys) > 20, "fixture should be a RAW page, or this proves nothing"
        assert any(len(event["markets"]) > 1 for event in multi_close_page["events"]), (
            "fixture needs a multi-market event, or the tail tier is never exercised"
        )
        assert set(venues.KALSHI_NESTED_TAIL_FIELDS) < set(venues.KALSHI_MARKET_FIELDS)

        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload=multi_close_page)})
        pull = await venues.kalshi_prefetch_events(session)

        assert pull.token == ""
        assert pull.complete is True
        assert len(pull.events) == len(multi_close_page["events"])
        for event in pull.events:
            assert set(event) == {*venues.KALSHI_EVENT_FIELDS, "settlement_sources", "markets"}
            for index, market in enumerate(event["markets"]):
                expected = venues.KALSHI_MARKET_FIELDS if index == 0 else venues.KALSHI_NESTED_TAIL_FIELDS
                assert set(market) == set(expected), f"{event['event_ticker']} market {index}"

    @pytest.mark.asyncio
    async def test_settlement_sources_are_projected_from_the_event_level(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """``settlement_sources`` lives on the EVENT; every nested market's copy is null, so a
        projection reading the market level would ship a permanently empty join key."""
        raw = _event(multi_close_page, "USCLIMATE")
        assert raw["settlement_sources"], "fixture must carry event-level sources"
        assert all(market.get("settlement_sources") is None for market in raw["markets"])

        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload=multi_close_page)})
        pull = await venues.kalshi_prefetch_events(session)

        projected = next(event for event in pull.events if event["event_ticker"] == "USCLIMATE")
        assert projected["settlement_sources"] == [
            {"name": source["name"], "url": source["url"]} for source in raw["settlement_sources"]
        ]

    @pytest.mark.asyncio
    async def test_a_200_carrying_a_null_events_array_is_a_lost_catalogue(self) -> None:
        """The failure this check exists for: a 200 whose body has no ``events`` array yields
        zero items exactly like an empty exchange, and caching that pins an empty index for
        6h and silently disables both Kalshi channels."""
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload={"events": None, "cursor": ""})})
        pull = await venues.kalshi_prefetch_events(session)

        assert pull.events == []
        assert pull.token == "error(no_events_array)"
        assert pull.complete is False, "an incomplete pull must never be cached for the TTL"

    @pytest.mark.asyncio
    async def test_a_genuinely_empty_catalogue_is_distinguishable_from_a_lost_one(self) -> None:
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload={"events": [], "cursor": ""})})
        pull = await venues.kalshi_prefetch_events(session)

        assert (pull.events, pull.token, pull.complete) == ([], "", True)

    @pytest.mark.asyncio
    async def test_a_rate_limited_page_stops_pagination_without_retrying(self) -> None:
        """Re-asking a rate limiter 0.5s later is a second violation, not a retry, so a 429
        stops the pull dead and refuses to cache — otherwise a throttled exchange gets pinned
        as a short catalogue for the 6h TTL."""
        session = FakeSession(
            {venues.KALSHI_EVENTS_URL: [FakeResponse(429, text="slow down"), FakeResponse(200, payload={"events": []})]}
        )
        pull = await venues.kalshi_prefetch_events(session)

        assert pull.events == []
        assert pull.token == "error(http_429)"
        assert pull.complete is False
        assert pull.tally.failed == 1
        assert session._call_counts[venues.KALSHI_EVENTS_URL] == 1, "a 429 must not be re-issued"

    @pytest.mark.asyncio
    async def test_a_page_over_the_byte_ceiling_is_dropped_and_not_retried(
        self, monkeypatch: pytest.MonkeyPatch, multi_close_page: dict[str, Any]
    ) -> None:
        """The last-resort guard against a runaway or compressed-bomb body, which had no
        behavioural test at all — neutering it (`if False`) left the whole suite green.

        Non-retryability is half the invariant: a retryable size cap doubles requests against a
        rate-limited exchange for a body that will be just as oversized the second time. The
        other half is `complete=False`, since the seam caches on exactly that flag and a
        `complete=True` here would pin a truncated catalogue for the 6h TTL. The real 64 MB
        ceiling is impractical to materialize, so the constant is patched down.
        """
        monkeypatch.setattr(venues, "KALSHI_PAGE_MAX_BYTES", 32)
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload=multi_close_page)})

        pull = await venues.kalshi_prefetch_events(session)

        assert pull.token == "dropped(size_cap)"
        assert pull.complete is False
        assert pull.events == []
        assert session._call_counts[venues.KALSHI_EVENTS_URL] == 1, "an oversized body must not be re-requested"

    @pytest.mark.asyncio
    async def test_a_body_that_dies_mid_stream_is_a_lost_page_not_an_empty_one(self) -> None:
        """The worst of the untested branches. Mutated to return a clean EMPTY page instead of
        `error(parse)`, a garbled body pins an empty events list for the 6h TTL with `token="none"`
        and the degradation counter NOT bumped — the exact failure the sibling `{"events": null}`
        test exists to prevent, and it never reddens CI.

        The truncated body fires the parse branch at the default ceiling, and it is the first and
        only call site of the `raw_content` hook `FakeResponse` was built for.
        """
        truncated = b'{"events": [{"event_ticker": "A", "title": "x"'
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, raw_content=truncated)})

        pull = await venues.kalshi_prefetch_events(session)

        assert pull.token == "error(parse)"
        assert pull.complete is False
        assert pull.events == []
        assert session._call_counts[venues.KALSHI_EVENTS_URL] == 1, "a malformed body must not be re-requested"

    @pytest.mark.asyncio
    async def test_a_retry_that_would_not_fit_the_wall_is_not_taken(self, caplog: pytest.LogCaptureFixture) -> None:
        """The retry-budget guard: a 503 IS retryable, but sleeping the backoff would consume a
        wall that has almost nothing left, so the pull gives up and reports the transport error
        it actually saw.

        Anchored on THIS guard rather than the loop-top deadline check, which is behaviourally
        shadowed by the per-attempt `attempt_budget <= 0.0` branch — deleting the loop-top check
        leaves token, complete, tally and call count byte-identical, so a test written against it
        would pass on its own mutant. Delete the guard under test instead and the pull sleeps
        0.5s and reports `error(wall_timeout)`, which is what the token assertion catches.
        """
        session = FakeSession(
            {venues.KALSHI_EVENTS_URL: [FakeResponse(503, text="down"), FakeResponse(503, text="still down")]}
        )

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.market_retrieval.venues"):
            pull = await venues.kalshi_prefetch_events(session, wall_timeout=0.01)

        assert pull.token == "error(http_503)", "the wall must not relabel the transport failure"
        assert pull.complete is False
        assert session._call_counts[venues.KALSHI_EVENTS_URL] == 1
        messages = "\n".join(caplog.messages)
        assert "retry budget exhausted" in messages
        assert "retry 2/" not in messages, "no backoff may be taken when the wall cannot fit it"

    @pytest.mark.asyncio
    async def test_pagination_follows_the_cursor_and_accumulates_every_page(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        first = {"events": multi_close_page["events"][:1], "cursor": "page2"}
        second = {"events": multi_close_page["events"][1:], "cursor": ""}
        session = FakeSession(
            {venues.KALSHI_EVENTS_URL: [FakeResponse(200, payload=first), FakeResponse(200, payload=second)]}
        )
        pull = await venues.kalshi_prefetch_events(session)

        assert [event["event_ticker"] for event in pull.events] == ["KXMARRIAGESTYLESKRAVITZ-HSZK", "USCLIMATE"]
        assert pull.tally.ok == 2
        assert pull.complete is True

    @pytest.mark.asyncio
    async def test_the_page_bound_stops_a_runaway_cursor_and_reports_it_incomplete(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """A cursor that never empties must terminate on the page bound, not spin — and the
        catalogue it leaves is TRUNCATED, so the pull is not complete. Reporting `complete=True`
        here would pin a short universe for the 6h TTL behind a green `ok(n)`, which is reachable
        with no code change at all: Kalshi halving its effective page size puts ~38% of the
        universe past the 120-page bound."""
        forever = {"events": multi_close_page["events"][:1], "cursor": "always-more"}
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload=forever)})
        # page_sleep_s=0.0: this test is about the BOUND, not the throttle courtesy the real
        # 0.25s default buys, and three real sleeps per bound test is dead suite time.
        pull = await venues.kalshi_prefetch_events(session, max_pages=3, page_sleep_s=0.0)

        assert pull.tally.ok == 3
        assert len(pull.events) == 3
        assert pull.complete is False
        assert pull.token == "dropped(runaway_bound)"

    @pytest.mark.asyncio
    async def test_the_event_limit_also_reports_the_pull_incomplete(self, multi_close_page: dict[str, Any]) -> None:
        """The sibling bound. Growth past `event_limit` truncates the catalogue exactly as the
        page bound does, and the two must not disagree on what `complete` means."""
        forever = {"events": multi_close_page["events"][:1], "cursor": "always-more"}
        session = FakeSession({venues.KALSHI_EVENTS_URL: FakeResponse(200, payload=forever)})
        pull = await venues.kalshi_prefetch_events(session, event_limit=2, page_sleep_s=0.0)

        assert len(pull.events) == 2
        assert pull.complete is False
        assert pull.token == "dropped(runaway_bound)"


class TestKalshiEventDerivations:
    """The two derivations that moved off ``nested[0]``."""

    def test_close_time_is_the_max_over_nested_markets(self, multi_close_page: dict[str, Any]) -> None:
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        nested_closes = [market["close_time"] for market in event["markets"]]
        assert len(set(nested_closes)) == 4, "fixture must have DIFFERING closes or this proves nothing"

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.close_time is not None
        assert match.close_time.strftime("%Y-%m-%d") == "2028-01-01"
        assert match.close_time.strftime("%Y-%m-%d") != nested_closes[0][:10]

    def test_close_time_survives_a_nested_market_missing_its_zulu_suffix(
        self, multi_close_page: dict[str, Any]
    ) -> None:
        """The mixed-awareness trap, at the site that actually compares two datetimes.

        `fromisoformat` returns an AWARE datetime for `...Z` and a NAIVE one for a bare
        timestamp, and `max()` over the two raises `TypeError: can't compare offset-naive and
        offset-aware`. It happens inside `to_thread` with no guard, so it reaches the
        snapshot-level net and zeroes all four venues for the question — and the offending event
        stays in the 6h cache, so every later question repeats it. `parse_iso` normalizes at the
        boundary, which covers all four venue parsers at once.
        """
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        latest = max(market["close_time"] for market in event["markets"])
        event["markets"][0]["close_time"] = event["markets"][0]["close_time"].replace("Z", "")

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.close_time is not None
        assert match.close_time.strftime("%Y-%m-%d") == latest[:10]

    def test_an_event_is_resolved_only_when_every_nested_market_is(self, multi_close_page: dict[str, Any]) -> None:
        """``nested[0]`` is ``finalized`` here while the other three are ``active``, so the old
        read reports a live event family RESOLVED — measured on ~305 of them."""
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        assert event["markets"][0]["status"] == "finalized"
        assert {market["status"] for market in event["markets"][1:]} == {"active"}

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.is_resolved is False

    def test_an_all_settled_event_is_resolved(self, multi_close_page: dict[str, Any]) -> None:
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        for market in event["markets"]:
            market["status"] = "settled"

        match = venues.kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.is_resolved is True

    def test_liquidity_and_prices_stay_on_the_first_nested_market(self, multi_close_page: dict[str, Any]) -> None:
        """Deliberately NOT aggregated: the legs describe one tradeable contract, and the
        divergence from the max-over-nested close time is what this fixture pins."""
        event = _event(multi_close_page, "KXMARRIAGESTYLESKRAVITZ-HSZK")
        first = event["markets"][0]
        expected_volume, expected_oi = venues.kalshi_usd_liquidity(first)

        match = venues.kalshi_event_match(event, match_confidence=0.5, channel="universe_fuzzy")

        assert match is not None
        assert match.total_volume == expected_volume
        assert match.open_interest == expected_oi
        assert match.bid == safe_float(first["yes_bid_dollars"])
        assert match.ask == safe_float(first["yes_ask_dollars"])

    def test_the_row_carries_the_ranked_retrieval_fields(self, multi_close_page: dict[str, Any]) -> None:
        event = _event(multi_close_page, "USCLIMATE")

        match = venues.kalshi_event_match(event, match_confidence=71.5, channel="settlement_join")

        assert match is not None
        assert match.venue_market_id == "USCLIMATE"
        assert match.sub_title == event["sub_title"]
        assert [source.url for source in match.settlement_sources] == [
            source["url"] for source in event["settlement_sources"]
        ]
        assert match.match_confidence == 71.5
        assert match.retrieval_channel == "settlement_join"

    def test_an_event_without_a_title_yields_no_row(self) -> None:
        assert venues.kalshi_event_match({"event_ticker": "X"}, match_confidence=1.0, channel="x") is None

    def test_kalshi_usd_liquidity_uses_the_captured_payload_units(self, captured_payloads: dict[str, Any]) -> None:
        """Volume scales by trade price and open interest by notional value. Reading the bare
        ``volume`` / ``open_interest`` names — which the spec does not define — blanked the
        label on 100% of Kalshi rows once already."""
        market = captured_payloads["kalshi_events"]["events"][0]["markets"][0]
        volume_usd, oi_usd = venues.kalshi_usd_liquidity(market)

        volume_fp = safe_float(market["volume_fp"])
        oi_fp = safe_float(market["open_interest_fp"])
        notional = safe_float(market["notional_value_dollars"]) or 1.0
        assert volume_fp is not None and oi_fp is not None
        assert oi_usd == pytest.approx(oi_fp * notional)
        assert volume_usd is not None and volume_usd <= volume_fp, "volume converts at a sub-dollar price"

    def test_absent_counts_read_as_no_data_rather_than_zero(self) -> None:
        assert venues.kalshi_usd_liquidity({}) == (None, None)
        assert venues.kalshi_usd_liquidity({"volume_fp": "0", "last_price_dollars": "0.5"})[0] == 0.0


class TestPolymarket:
    @pytest.mark.asyncio
    async def test_the_search_sends_events_status_active(self, captured_payloads: dict[str, Any]) -> None:
        """Nothing downstream filters on close date, so this parameter is the only thing keeping
        resolved markets out of the pool: ``Ethereum`` returns 7 closed events out of 10 without
        it."""
        seen: dict[str, Any] = {}

        def handler(params: dict[str, Any]) -> FakeResponse:
            seen.update(params)
            return FakeResponse(200, payload=captured_payloads["polymarket_search"])

        session = FakeSession({venues.POLYMARKET_SEARCH_URL: handler})

        await venues.polymarket_search(session, "ethereum", width=60)

        assert seen["events_status"] == "active"
        assert seen["q"] == "ethereum"

    def test_the_event_branch_reads_open_interest(self, captured_payloads: dict[str, Any]) -> None:
        """Gamma carries ``openInterest`` on the EVENT (5/5 live events vs 0/42 nested), which
        is why reading it only at the market level blanked the OI column on every archived row."""
        payload = copy.deepcopy(captured_payloads["polymarket_search"])
        payload["events"][0]["openInterest"] = 4321.0
        payload["events"][0]["markets"][0].pop("openInterest", None)

        rows = venues.parse_polymarket_matches(payload, width=60)

        assert rows[0].open_interest == 4321.0
        assert rows[0].platform == "polymarket"

    def test_the_markets_fallback_branch_also_reads_open_interest(self) -> None:
        payload = {
            "events": [],
            "markets": [
                {
                    "question": "Will X happen?",
                    "slug": "will-x-happen",
                    "openInterest": 999.0,
                    "volumeNum": 12345.0,
                    "description": "rules text",
                    "closed": True,
                }
            ],
        }

        rows = venues.parse_polymarket_matches(payload, width=60)

        assert len(rows) == 1
        assert rows[0].open_interest == 999.0
        assert rows[0].total_volume == 12345.0
        assert rows[0].is_resolved is True
        assert rows[0].venue_market_id == "will-x-happen"

    def test_the_score_is_the_inverted_venue_rank(self, captured_payloads: dict[str, Any]) -> None:
        rows = venues.parse_polymarket_matches(captured_payloads["polymarket_search"], width=60)

        assert [row.match_confidence for row in rows] == [100.0 - rank for rank in range(len(rows))]
        assert all(row.retrieval_channel == "venue_search" for row in rows)

    def test_width_bounds_the_parse(self, captured_payloads: dict[str, Any]) -> None:
        payload = copy.deepcopy(captured_payloads["polymarket_search"])
        payload["events"] = payload["events"] * 5

        assert len(venues.parse_polymarket_matches(payload, width=3)) == 3

    def test_a_non_dict_payload_parses_to_nothing(self) -> None:
        assert venues.parse_polymarket_matches(["not", "a", "dict"], width=60) == []


class TestManifold:
    def test_close_and_liquidity_come_off_each_search_row(self, captured_payloads: dict[str, Any]) -> None:
        """The bake-off measured blank close/liquidity on 52 of 94 Manifold rows because it
        looked each row up in a cached universe it was never in. Everything is read off the
        search row itself now."""
        payload = captured_payloads["manifold_search"]
        rows = venues.parse_manifold_matches(payload, width=60)

        assert rows, "fixture must carry Manifold search rows"
        for row, raw in zip(rows, payload):
            assert row.num_bettors == raw.get("uniqueBettorCount")
            assert row.total_volume == safe_float(raw.get("volume"))
            if isinstance(raw.get("closeTime"), (int, float)):
                assert row.close_time is not None
            assert row.venue_market_id == str(raw["id"])

    def test_the_score_is_the_inverted_venue_rank(self, captured_payloads: dict[str, Any]) -> None:
        rows = venues.parse_manifold_matches(captured_payloads["manifold_search"], width=60)

        assert [row.match_confidence for row in rows] == [100.0 - rank for rank in range(len(rows))]

    def test_the_caller_owns_the_row_width(self) -> None:
        """Width is the caller's, never a slice inside the parser: a ``payload[:10]`` here would
        make "width 60" mean 10 per query, with nothing to see at the call site."""
        payload = [{"id": f"m{i}", "question": f"Q{i}", "probability": 0.5} for i in range(25)]

        assert len(venues.parse_manifold_matches(payload, width=25)) == 25
        assert len(venues.parse_manifold_matches(payload, width=4)) == 4

    @pytest.mark.asyncio
    async def test_the_search_keeps_the_binary_contract_type(self, captured_payloads: dict[str, Any]) -> None:
        """Retained with eyes open: a known ~30% recall ceiling, accepted for this port because
        multi-outcome rows have no render path yet."""
        seen: dict[str, Any] = {}

        def handler(params: dict[str, Any]) -> FakeResponse:
            seen.update(params)
            return FakeResponse(200, payload=captured_payloads["manifold_search"])

        session = FakeSession({venues.MANIFOLD_SEARCH_URL: handler})

        await venues.manifold_search(session, "gas prices", width=60)

        assert seen["contractType"] == "BINARY"
        assert seen["term"] == "gas prices"

    def test_rules_text_does_not_fall_back_to_the_title(self) -> None:
        """A title-as-rules row spends ranker tokens repeating its own title; a blank is the
        honest signal that enrichment has not filled it in."""
        assert venues.manifold_rules_text({"question": "Will X happen?"}) == ""

    def test_rules_text_flattens_a_tiptap_description(self) -> None:
        doc = {"type": "doc", "content": [{"type": "p", "content": [{"text": "resolves per"}, {"text": "NOAA"}]}]}

        assert venues.manifold_rules_text({"description": doc}) == "resolves per NOAA"

    @pytest.mark.asyncio
    async def test_a_detail_fetch_returns_none_on_a_non_dict_body(self) -> None:
        session = FakeSession({venues.MANIFOLD_MARKET_URL: FakeResponse(200, payload=["not", "a", "dict"])})

        assert await venues.manifold_market_detail(session, "abc") is None


class TestScalarCoercions:
    """The coercions every venue parser sits on, at the boundary where they are enforced."""

    def test_parse_iso_returns_an_aware_datetime_for_both_shapes(self) -> None:
        """The `Z` form and a bare timestamp must come back mutually comparable, or a `max()`
        over an event's nested closes raises and takes the whole snapshot down."""
        aware = parse_iso("2026-11-01T00:00:00Z")
        naive_input = parse_iso("2026-11-01T00:00:00")

        assert aware is not None and naive_input is not None
        assert aware.tzinfo is not None
        assert naive_input.tzinfo is not None
        assert naive_input == aware, "a naive value is TREATED as UTC, not shifted"
        assert max([aware, naive_input]) == aware

    def test_parse_iso_does_not_shift_the_rendered_wall_clock(self) -> None:
        """Attaching a tzinfo must leave every `%Y-%m-%d` render byte-identical — that is what
        makes the normalization safe to do at the boundary rather than per comparison site."""
        parsed = parse_iso("2026-11-01T23:30:00")

        assert parsed is not None
        assert parsed.strftime("%Y-%m-%d") == "2026-11-01"

    def test_a_date_only_string_is_also_aware(self) -> None:
        """The shape `parse_iso_guarded` falls back to, which is how PredictIt rows reach the
        pool naive today."""
        parsed = parse_iso("2026-11-01")

        assert parsed is not None and parsed.tzinfo is not None

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_safe_float_rejects_non_finite_values(self, literal: str) -> None:
        """`json.loads` accepts these bare literals, and NaN defeats every comparison in
        `_liquidity_label` — so a row whose volume arrived as NaN would fall through to the
        strongest label and render `signal=deep`, presenting missing data to a forecaster as the
        best possible liquidity evidence. None renders `no-liquidity-data` instead."""
        value = json.loads(literal)

        assert safe_float(value) is None
        assert _liquidity_label(_row_with_volume(value)) == "no-liquidity-data"

    def test_safe_float_keeps_a_real_zero(self) -> None:
        """The distinction the liquidity signal rests on: absent is None, a genuinely
        zero-volume brand-new market is 0.0 and renders `thin`."""
        assert safe_float(0) == 0.0
        assert safe_float("0.0000") == 0.0


def _row_with_volume(volume: Any) -> MarketMatch:
    return MarketMatch(
        platform="kalshi",
        market_title="M",
        market_url="https://kalshi.com/markets/M",
        implied_prob_yes=None,
        bid=None,
        ask=None,
        spread=None,
        volume_24h=None,
        close_time=None,
        is_resolved=False,
        match_confidence=1.0,
        raw_rules="",
        total_volume=safe_float(volume),
        open_interest=None,
    )


class TestPredictIt:
    @pytest.mark.parametrize("sentinel", ["NA", "N/A", None, "", "not-a-date", "2026-11", "20261103"])
    def test_a_value_with_no_calendar_date_prefix_yields_no_close_time(self, sentinel: Any) -> None:
        """The prefix guard is what keeps PredictIt's ``NA``/``N/A`` no-close sentinels — and any
        other non-date the dump might ship — from reaching the parser at all."""
        assert parse_iso_guarded(sentinel) is None

    def test_a_seven_digit_fraction_keeps_its_FULL_timestamp(self) -> None:
        """``fromisoformat`` has accepted arbitrary fraction lengths since 3.11, so this shape
        parses whole rather than degrading to the date — asserting only ``%Y-%m-%d`` here would
        hold for BOTH outcomes and so pin neither.

        Pinned because the fallback is lossy on purpose: if a future guard tightened the tail and
        pushed this shape down to the date rung, `close_time` on every fractional PredictIt row
        would silently jump backwards by up to a day.
        """
        parsed = parse_iso_guarded("2026-11-03T23:59:59.1234567")

        assert parsed == datetime(2026, 11, 3, 23, 59, 59, 123456, tzinfo=timezone.utc)

    @pytest.mark.parametrize(
        "unparseable_tail",
        [
            "2026-11-03T23:59:59 EST",  # a named zone, which fromisoformat has never accepted
            "2026-11-03T25:00:00",  # hour out of range
            "2026-11-03T23:59:60",  # leap second
            "2026-11-03T23:59:59.",  # a truncated fraction
        ],
    )
    def test_a_valid_date_with_an_unparseable_tail_degrades_to_the_date(self, unparseable_tail: str) -> None:
        """The rung the guard exists for, and the only one that can actually fire: a real
        calendar date whose TAIL `fromisoformat` refuses. Truncating to the date keeps the row's
        close window — all the ranker reads — where returning None would render `closes:` blank
        and losing the guard would raise inside `to_thread` and zero all four venues."""
        parsed = parse_iso_guarded(unparseable_tail)

        assert parsed == datetime(2026, 11, 3, tzinfo=timezone.utc)

    def test_a_plain_timestamp_keeps_its_time_of_day(self) -> None:
        assert parse_iso_guarded("2026-11-03T23:59:59") == datetime(2026, 11, 3, 23, 59, 59, tzinfo=timezone.utc)

    def test_the_row_carries_the_parsed_close_time(self, captured_payloads: dict[str, Any]) -> None:
        market = copy.deepcopy(captured_payloads["predictit_all"]["markets"][0])
        market["contracts"][0]["dateEnd"] = "2026-12-31T23:59:59.9999999"

        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.close_time is not None
        assert match.close_time.strftime("%Y-%m-%d") == "2026-12-31"

    def test_contract_names_become_the_rules_text(self, captured_payloads: dict[str, Any]) -> None:
        """A PredictIt market's own title is often just "Which party will win ...", so the
        contract names are most of its semantic content."""
        market = copy.deepcopy(captured_payloads["predictit_all"]["markets"][0])
        market["contracts"] = [{"name": f"Candidate {i}"} for i in range(12)]

        names = venues.predictit_contract_names(market)
        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert names.count(",") == venues.PREDICTIT_CONTRACTS_RENDERED - 1
        assert match is not None
        assert match.raw_rules == f"contracts: {names}"

    def test_no_price_is_reported(self, captured_payloads: dict[str, Any]) -> None:
        """The whole universe reaches the ranker, so there is no query to select a contract
        with, and pricing an arbitrary one would misreport the market."""
        market = copy.deepcopy(captured_payloads["predictit_all"]["markets"][0])

        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert (match.implied_prob_yes, match.bid, match.ask, match.spread) == (None, None, None, None)
        assert match.venue_market_id == str(market["id"])

    def test_a_nameless_market_yields_no_row(self) -> None:
        assert venues.predictit_market_match({"id": 7}, match_confidence=1.0, channel="x") is None

    def test_a_market_whose_every_contract_is_closed_is_resolved(self) -> None:
        market = {"id": 1, "name": "M", "contracts": [{"name": "c", "status": "Closed"}]}

        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None and match.is_resolved is True

    @pytest.mark.parametrize("order", [("Closed", "Open"), ("Open", "Closed")])
    def test_one_open_contract_keeps_the_market_live_in_either_order(self, order: tuple[str, str]) -> None:
        """The `nested[0]` bug this file already fixes for Kalshi. Reading the first contract made
        the verdict depend on the order untrusted external JSON arrived in — the SAME market read
        RESOLVED one way and open the other — and the ranker prompt tells the model a RESOLVED
        price is a realized outcome rather than a forecast, so it deprioritizes a live market."""
        market = {
            "id": 1,
            "name": "M",
            "contracts": [
                {"name": "a", "status": order[0], "dateEnd": "2026-07-01T00:00:00"},
                {"name": "b", "status": order[1], "dateEnd": "2026-12-31T00:00:00"},
            ],
        }

        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.is_resolved is False
        assert match.close_time is not None
        assert match.close_time.strftime("%Y-%m-%d") == "2026-12-31", "close_time is the max over contracts"

    def test_a_contracts_block_holding_no_usable_entry_is_not_resolved(self) -> None:
        """The empty-sequence trap: `all()` over nothing is True, so folding the dict filter
        inside the conjunction would report a market with no readable contract as SETTLED."""
        market = {"id": 1, "name": "M", "contracts": ["nonsense", 7, None]}

        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None
        assert match.is_resolved is False
        assert match.close_time is None

    def test_a_contract_with_no_status_reads_as_open(self) -> None:
        """A missing field is not evidence of settlement."""
        market = {"id": 1, "name": "M", "contracts": [{"name": "c"}]}

        match = venues.predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy")

        assert match is not None and match.is_resolved is False

    @pytest.mark.asyncio
    async def test_prefetch_keeps_the_none_versus_empty_contract(self) -> None:
        lost = FakeSession({venues.PREDICTIT_URL: [FakeResponse(503, text="down"), FakeResponse(503, text="down")]})
        assert await venues.predictit_prefetch(lost) is None

        shapeless = FakeSession({venues.PREDICTIT_URL: FakeResponse(200, payload={"nope": []})})
        assert await venues.predictit_prefetch(shapeless) == []


class TestVenueRetryBudget:
    """Each buffered venue gets a SECOND attempt, pinned on a literal 2.

    The sibling assertion in the provider suite compares the call count to
    ``*_MAX_ATTEMPTS`` itself, which is tautological: dropping all three constants to 1
    leaves it — and the whole suite — green while silently retiring every venue retry. A
    transient blip on a single query then costs that query's rows outright.
    """

    @pytest.mark.parametrize(
        ("url", "fetch", "attempts", "empty_body"),
        [
            (
                venues.MANIFOLD_SEARCH_URL,
                lambda session: venues.manifold_search(session, "q", width=60),
                venues.MANIFOLD_MAX_ATTEMPTS,
                [],
            ),
            (
                venues.POLYMARKET_SEARCH_URL,
                lambda session: venues.polymarket_search(session, "q", width=60),
                venues.POLYMARKET_MAX_ATTEMPTS,
                {"events": [], "markets": []},
            ),
            (venues.PREDICTIT_URL, venues.predictit_prefetch, venues.PREDICTIT_MAX_ATTEMPTS, {"markets": []}),
        ],
    )
    @pytest.mark.asyncio
    async def test_a_transient_first_attempt_is_retried_and_the_second_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, url: str, fetch: Any, attempts: int, empty_body: Any
    ) -> None:
        monkeypatch.setattr("metaculus_bot.research.market_retrieval.http.HTTP_RETRY_BACKOFF_SECS", 0.0)
        session = FakeSession({url: [FakeResponse(503, text="down"), FakeResponse(200, payload=empty_body)]})

        assert await fetch(session) == [], "the second attempt's body must reach the parser"
        assert session._call_counts[url] == 2, "a transient first attempt must be retried"
        assert attempts >= 2, "the constant must permit the retry this test just took"


class TestBufferedBodyCap:
    """``MAX_RESPONSE_BYTES``, driven through the three venues that buffer a whole body.

    Previously unreachable from any test: ``read_json_capped`` chose between the capped stream
    and a bare ``resp.json()`` on whether the response exposed ``.read``, and the fake did not,
    so every stubbed venue took the UNCAPPED branch — production's memory ceiling was decided
    by what a test double implemented, and neutering the cap left the suite green.
    """

    @pytest.mark.parametrize(
        ("url", "fetch"),
        [
            (venues.MANIFOLD_SEARCH_URL, lambda session: venues.manifold_search(session, "q", width=60)),
            (venues.POLYMARKET_SEARCH_URL, lambda session: venues.polymarket_search(session, "q", width=60)),
            (venues.PREDICTIT_URL, venues.predictit_prefetch),
        ],
    )
    @pytest.mark.asyncio
    async def test_an_oversized_body_reads_as_a_failed_fetch_not_an_empty_one(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, url: str, fetch: Any
    ) -> None:
        """Every buffered venue sits on the one cap, and a breach must return ``None``: as ``[]``
        it would publish as a benign ``none`` token and read as a venue whose index found
        nothing. The 10 MiB ceiling is impractical to materialize, so it is patched down."""
        monkeypatch.setattr("metaculus_bot.research.market_retrieval.http.MAX_RESPONSE_BYTES", 64)
        session = FakeSession({url: FakeResponse(200, payload=[{"id": "m", "question": "x" * 500}])})

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.http_fetch"):
            assert await fetch(session) is None

        assert "response too large" in "\n".join(caplog.messages)

    @pytest.mark.asyncio
    async def test_a_body_under_the_cap_parses_whole(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The other half of the cap: it must not truncate a legitimate body. A capped read that
        returned its partial buffer instead of None would hand `json.loads` a half object and
        surface as an unexplained decode failure on the largest real payloads."""
        payload = [{"id": f"m{index}", "question": f"Q{index}", "probability": 0.5} for index in range(40)]
        monkeypatch.setattr(
            "metaculus_bot.research.market_retrieval.http.MAX_RESPONSE_BYTES", len(json.dumps(payload).encode())
        )
        session = FakeSession({venues.MANIFOLD_SEARCH_URL: FakeResponse(200, payload=payload)})

        rows = await venues.manifold_search(session, "q", width=60)

        assert rows is not None
        assert [row.venue_market_id for row in rows] == [str(entry["id"]) for entry in payload]


def _manifold_row(title: str) -> MarketMatch:
    return MarketMatch(
        platform="manifold",
        market_title=title,
        market_url="",
        implied_prob_yes=None,
        bid=None,
        ask=None,
        spread=None,
        volume_24h=None,
        close_time=None,
        is_resolved=False,
        match_confidence=1.0,
        raw_rules="",
    )


class TestPerQueryFailureIsolation:
    """``None`` is an upstream failure, ``[]`` a search that parsed to nothing. Collapsing the
    two publishes a total outage as a benign ``none``."""

    def test_a_lost_query_is_tallied_without_losing_its_siblings(self) -> None:
        results = [[_manifold_row("a")], None, [_manifold_row("b")], ValueError("boom")]
        rows, tally = flatten_results(results, "manifold")

        assert [row.market_title for row in rows] == ["a", "b"]
        assert (tally.ok, tally.failed) == (2, 2)

    def test_an_all_empty_fan_out_is_a_clean_zero(self) -> None:
        rows, tally = flatten_results([[], []], "manifold")

        assert rows == []
        assert (tally.ok, tally.failed) == (2, 0)
