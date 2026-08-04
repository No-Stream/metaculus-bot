"""The settlement-source provenance join: host normalization, registrable domains, the index.

This is the port's only genuinely new retrieval channel, so it carries the port's
highest-value tests. The mechanism: a Metaculus question names the source that settles it
(a BLS release page, a NATO press page), and every open Kalshi event carries
``settlement_sources[].url``, so the two join on provenance rather than on words.

The two failure modes these tests exist to prevent, both measured during the bake-off:

- **Naive last-two-labels domain reduction.** ``abs.gov.au`` would collapse to ``gov.au``,
  which unions every Australian government event into any question citing any Australian
  agency. The public suffix list is the published answer, and ``gov.au`` being *itself* a
  public suffix is the case that separates the two algorithms.
- **Self-referential settlement sources.** Kalshi points 986 events' placeholder sources at
  ``https://kalshi.com/``, so joining on that domain unions a tenth of the exchange into
  every question that happens to link Kalshi.

The Kalshi fixture is the committed verbatim trim (``prediction_market_venue_payloads.json``),
which already carries real EVENT-level ``settlement_sources`` — including the kalshi.com
self-reference — and null copies at the market level. That null/non-null split is exactly
what the index must read correctly, so it is asserted rather than assumed.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from metaculus_bot.research.market_retrieval.settlement_join import (
    SELF_REFERENCE_DOMAINS,
    normalize_host,
    question_domains,
    registrable_domain,
    settlement_domain_index,
)

_DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def kalshi_events() -> list[dict[str, Any]]:
    """The committed venue-payload trim's Kalshi events (2 events, both with settlement sources)."""
    payload = json.loads((_DATA / "prediction_market_venue_payloads.json").read_text())
    events = payload["kalshi_events"]["events"]
    assert events, "fixture lost its Kalshi events — regenerate it from a live payload"
    return events


class TestNormalizeHost:
    def test_lowercases_and_strips_www(self):
        assert normalize_host("https://WWW.BLS.gov/news.release/empsit.htm") == "bls.gov"

    def test_strips_numbered_www_prefix(self):
        """``www150.statcan.gc.ca`` is real in the Kalshi settlement-source payload."""
        assert normalize_host("https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1") == "statcan.gc.ca"

    def test_drops_port_and_userinfo(self):
        """`.hostname`, not `.netloc` — a port or userinfo must not survive into the host."""
        assert normalize_host("https://user:pw@data.bls.gov:8443/timeseries") == "data.bls.gov"

    def test_returns_none_without_a_host(self):
        assert normalize_host("not a url at all") is None
        assert normalize_host("") is None


class TestRegistrableDomain:
    def test_collapses_a_subdomain_to_its_publisher(self):
        """``data.bls.gov`` and ``www.bls.gov`` are the same publisher; ``bls.gov`` is where they meet."""
        assert registrable_domain("data.bls.gov") == "bls.gov"
        assert registrable_domain("bls.gov") == "bls.gov"

    def test_keeps_all_three_labels_when_the_two_label_suffix_is_public(self):
        """The wildcard-suffix trap: ``gov.au`` is itself a public suffix, so the registrable
        domain of ``abs.gov.au`` is the WHOLE host. Naive last-two-labels would return
        ``gov.au`` and union every Australian government market into the pool."""
        assert registrable_domain("abs.gov.au") == "abs.gov.au"
        assert registrable_domain("www150.statcan.gc.ca") == "statcan.gc.ca"

    def test_returns_none_for_a_bare_public_suffix(self):
        """Nothing is registered under a bare suffix, so there is no publisher to join on."""
        assert registrable_domain("gov.au") is None
        assert registrable_domain("com") is None
        assert registrable_domain("gc.ca") is None

    def test_applies_wildcard_rules(self):
        """``*.compute.amazonaws.com`` makes every direct child a suffix in its own right, so
        the registrable domain is one label deeper than the host's own parent."""
        assert registrable_domain("host.us-east-1.compute.amazonaws.com") == "host.us-east-1.compute.amazonaws.com"
        assert registrable_domain("anything.kawasaki.jp") is None

    def test_applies_exception_rules(self):
        """``!city.kawasaki.jp`` carves one host back out of ``*.kawasaki.jp``: the prevailing
        exception rule drops its leftmost label, so ``kawasaki.jp`` is the suffix here and
        ``city.kawasaki.jp`` is registrable — unlike its wildcard siblings just above."""
        assert registrable_domain("city.kawasaki.jp") == "city.kawasaki.jp"
        assert registrable_domain("sub.city.kawasaki.jp") == "city.kawasaki.jp"

    def test_treats_an_unknown_tld_as_a_one_label_suffix(self):
        assert registrable_domain("internal.example.nosuchtld") == "example.nosuchtld"


class TestQuestionDomains:
    def test_extracts_the_publisher_of_a_resolution_source(self):
        text = (
            "This resolves per the seasonally adjusted U-3 rate published at "
            "https://data.bls.gov/timeseries/LNS14000000 and cross-checked against "
            "[the release](https://www.bls.gov/news.release/empsit.htm)."
        )
        assert question_domains(text) == {"bls.gov"}

    def test_drops_both_self_reference_domains(self):
        """A question citing Kalshi or Metaculus names the VENUE, not a settlement series."""
        text = (
            "See the sister market at https://kalshi.com/markets/kxelonmars and the "
            "companion question at https://www.metaculus.com/questions/44796/."
        )
        assert question_domains(text) == set()

    def test_a_port_or_userinfo_cannot_smuggle_a_host_past_the_self_reference_check(self):
        text = (
            "https://user@kalshi.com:8443/markets/kxu3 "
            "https://metaculus.com:443/questions/44796/ "
            "https://subdomain.metaculus.com/questions/1/"
        )
        assert question_domains(text) == set()

    def test_returns_an_empty_set_for_text_with_no_urls(self):
        assert question_domains("Resolves per the official announcement.") == set()
        assert question_domains("") == set()

    def test_self_reference_domains_are_the_two_measured_ones(self):
        assert SELF_REFERENCE_DOMAINS == frozenset({"kalshi.com", "metaculus.com"})


class TestSettlementDomainIndex:
    def test_indexes_event_level_settlement_sources(self, kalshi_events):
        index = settlement_domain_index(kalshi_events)
        assert index["theguardian.com"][0]["event_ticker"] == "KXELONMARS-99"
        assert index["nato.int"][0]["event_ticker"] == "KXNEXTNATOSECGEN-99"

    def test_drops_the_kalshi_self_reference(self, kalshi_events):
        """The fixture's Elon-Mars event carries a real ``https://kalshi.com/`` placeholder
        source alongside The Guardian; only the Guardian leg may reach the index."""
        raw_urls = [src["url"] for ev in kalshi_events for src in (ev.get("settlement_sources") or [])]
        assert any("kalshi.com" in url for url in raw_urls), "fixture lost its self-reference source"
        assert "kalshi.com" not in settlement_domain_index(kalshi_events)

    def test_reads_the_event_level_not_the_market_level(self, kalshi_events):
        """``settlement_sources`` lives on the EVENT; every nested market's copy is null.
        An index built off nested markets would be permanently empty."""
        for event in kalshi_events:
            for market in event.get("markets") or []:
                assert market.get("settlement_sources") is None

        market_level_only = copy.deepcopy(kalshi_events)
        for event in market_level_only:
            sources = event.pop("settlement_sources")
            for market in event.get("markets") or []:
                market["settlement_sources"] = sources
        assert settlement_domain_index(market_level_only) == {}

    def test_skips_events_without_a_ticker_and_dedupes_within_a_domain(self):
        events = [
            {
                "event_ticker": "KXCPI-26",
                "settlement_sources": [
                    {"name": "BLS", "url": "https://www.bls.gov/cpi/"},
                    {"name": "BLS data", "url": "https://data.bls.gov/timeseries/CUUR0000SA0"},
                ],
            },
            {"settlement_sources": [{"name": "BLS", "url": "https://www.bls.gov/cpi/"}]},
        ]
        index = settlement_domain_index(events)
        assert [ev["event_ticker"] for ev in index["bls.gov"]] == ["KXCPI-26"]

    def test_tolerates_missing_malformed_and_hostless_sources(self):
        events = [
            {"event_ticker": "A"},
            {"event_ticker": "B", "settlement_sources": None},
            {"event_ticker": "C", "settlement_sources": ["https://www.bls.gov/cpi/"]},
            {"event_ticker": "D", "settlement_sources": [{"name": "no url"}]},
            {"event_ticker": "E", "settlement_sources": [{"name": "bare suffix", "url": "https://gov.au"}]},
        ]
        assert settlement_domain_index(events) == {}

    def test_returns_an_empty_index_for_an_empty_catalogue(self):
        assert settlement_domain_index([]) == {}
