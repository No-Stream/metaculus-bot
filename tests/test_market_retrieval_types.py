"""``MarketMatch``'s construction shape and its archive contract.

``MarketMatch`` is ADDITIVE-ONLY, and three consumers are why: ``raw_log`` archives the whole
snapshot through ``dataclasses.asdict`` under an envelope whose ``schema_version`` is shared
across every provider (so a removed field changes the archive shape with no version to bump),
``provider_health`` reads the liquidity fields by ``getattr`` name, and several test sites
construct the row with twelve positional arguments. These tests pin all three: the positional
prefix, the defaults on every field added since, and that the whole row survives ``asdict``
into JSON.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone

from metaculus_bot.research.market_retrieval.types import MarketMatch, MarketSnapshot, SettlementSource

_CLOSE = datetime(2026, 8, 31, tzinfo=timezone.utc)


def _positional_row() -> MarketMatch:
    """The row exactly as the venue parsers and six test sites build it: twelve positional args."""
    return MarketMatch(
        "kalshi",
        "Australia unemployment rate above 4.2% in July",
        "https://kalshi.com/markets/kxauunemp",
        0.62,
        0.60,
        0.64,
        0.04,
        1_200.0,
        _CLOSE,
        False,
        1.0,
        "Settles on the ABS seasonally adjusted rate.",
    )


class TestMarketMatchShape:
    def test_the_twelve_positional_arguments_still_bind_the_same_fields(self):
        """Six test sites construct the row positionally, so the prefix order is a contract."""
        match = _positional_row()
        assert match.platform == "kalshi"
        assert match.market_title == "Australia unemployment rate above 4.2% in July"
        assert match.market_url == "https://kalshi.com/markets/kxauunemp"
        assert (match.implied_prob_yes, match.bid, match.ask, match.spread) == (0.62, 0.60, 0.64, 0.04)
        assert (match.volume_24h, match.close_time, match.is_resolved) == (1_200.0, _CLOSE, False)
        assert match.match_confidence == 1.0
        assert match.raw_rules == "Settles on the ABS seasonally adjusted rate."

    def test_every_field_added_since_defaults_to_absent(self):
        """A row built by a venue parser that knows nothing about ranking must still be valid;
        the ranker fills the relation fields later, and the archive must read an unranked row
        as unranked rather than as rank 0 of tier ''."""
        match = _positional_row()
        assert (match.total_volume, match.liquidity, match.open_interest, match.num_bettors) == (None,) * 4
        assert match.venue_market_id == ""
        assert match.relation_tier == ""
        assert match.relevance_label == ""
        assert match.rank is None
        assert match.retrieval_channel == ""
        assert match.sub_title == ""
        assert match.settlement_sources == ()
        assert match.top_answers == ()


class TestArchiveRoundTrip:
    """``raw_log`` serializes the snapshot with ``dataclasses.asdict`` then JSON. Anything on the
    row that ``asdict`` cannot walk, or that JSON cannot encode, breaks the raw-research archive
    for the whole provider rather than for one field."""

    def test_asdict_walks_nested_settlement_sources(self):
        match = _positional_row()
        match.venue_market_id = "KXAUUNEMP-26JUL"
        match.relation_tier = "same_quantity_same_date"
        match.relevance_label = "near-identical: same ABS series, same month"
        match.rank = 0
        match.retrieval_channel = "settlement_join"
        match.sub_title = "Above 4.2%"
        match.settlement_sources = (
            SettlementSource(name="Australian Bureau of Statistics", url="https://www.abs.gov.au/"),
        )

        as_dict = dataclasses.asdict(MarketSnapshot(matches=[match], sources={"kalshi": "ok(1)"}))
        row = as_dict["matches"][0]
        # `asdict` preserves the container type, so a tuple field stays a tuple; JSON encodes it
        # as an array either way (pinned in the round-trip test below).
        assert row["settlement_sources"] == (
            {"name": "Australian Bureau of Statistics", "url": "https://www.abs.gov.au/"},
        )
        assert row["rank"] == 0
        assert row["relation_tier"] == "same_quantity_same_date"
        assert as_dict["sources"] == {"kalshi": "ok(1)"}

    def test_the_archived_row_is_json_encodable_once_datetimes_are_stringified(self):
        match = _positional_row()
        match.settlement_sources = (SettlementSource(name="BLS", url="https://www.bls.gov/cpi/"),)
        encoded = json.loads(json.dumps(dataclasses.asdict(MarketSnapshot(matches=[match])), default=str))
        assert encoded["matches"][0]["close_time"].startswith("2026-08-31")
        assert encoded["matches"][0]["settlement_sources"] == [{"name": "BLS", "url": "https://www.bls.gov/cpi/"}]
        assert encoded["sources"] == {}

    def test_multi_outcome_answers_survive_asdict_and_json_as_pairs(self):
        """A multi-outcome row's answers are its ONLY price, so losing them in the archive would
        make every such row read as priceless in residual analysis. Nested plain tuples, so
        `asdict` leaves them alone and JSON writes them as arrays of two."""
        match = _positional_row()
        match.platform = "manifold"
        match.implied_prob_yes = None  # what Manifold publishes for a multi-outcome market
        match.top_answers = (("Over $4.60", 0.4992), ("Over $4.65", 0.3723))

        as_dict = dataclasses.asdict(MarketSnapshot(matches=[match]))
        assert as_dict["matches"][0]["top_answers"] == (("Over $4.60", 0.4992), ("Over $4.65", 0.3723))

        encoded = json.loads(json.dumps(as_dict, default=str))
        assert encoded["matches"][0]["top_answers"] == [["Over $4.60", 0.4992], ["Over $4.65", 0.3723]]
        assert encoded["matches"][0]["implied_prob_yes"] is None
