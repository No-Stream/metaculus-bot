"""Tests for the per-run provider-degradation signals (publish, then redden CI).

Two real degradations ran in prod for weeks while every counter read zero: Kalshi's
liquidity labels blank on 100% of rows since 2026-07-12, and Manifold contributing
zero rows since before 2026-07-17. Both were invisible because a degraded provider
is byte-identical to a healthy one in every channel the bot has. These tests pin
the three signals that close that gap, and — more importantly — pin the cases that
must stay SILENT, because an alert that fires on a legitimately market-less
tournament day is worse than no alert at all.

The field-contract tests run against ``tests/data/prediction_market_venue_payloads.json``,
real payloads captured off the venues' free public APIs. That is deliberate: the
2026-07-12 Kalshi regression shipped precisely because the unit fixture was
hand-written from the same wrong belief as the parser, so the pair agreed and the
suite stayed green. A test whose fixture mirrors the bug can never catch it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from metaculus_bot.constants import (
    PROVIDER_DEGRADATION_SUPPRESSED_UNTIL,
    provider_degradation_alerts_active,
)
from metaculus_bot.research.market_retrieval.generation import RETRIEVAL_WIDTH
from metaculus_bot.research.market_retrieval.venues import (
    kalshi_event_match,
    parse_manifold_matches,
    parse_polymarket_matches,
    predictit_market_match,
)
from metaculus_bot.research.orchestrator import ResearchOrchestrator
from metaculus_bot.research.prediction_market import MarketMatch, _liquidity_label
from metaculus_bot.research.provider_health import (
    SIGNAL_CATALOGUE_EMPTY,
    SIGNAL_MARKET_FIELD_CONTRACT,
    VENUE_EXPECTED_LIQUIDITY_FIELDS,
    VenueObservation,
    log_provider_degradation_summary,
    provider_degradation_count,
    provider_degradation_findings,
    record_catalogue_size,
    record_venue_observation,
    reset_provider_health,
)

_PAYLOADS_PATH = Path(__file__).parent / "data" / "prediction_market_venue_payloads.json"
_MULTI_CLOSE_PATH = Path(__file__).parent / "data" / "kalshi_multi_close_event_2026_08_04.json"
_FAMILY_LIQUIDITY_PATH = Path(__file__).parent / "data" / "kalshi_family_liquidity_2026_08_05.json"
_MANIFOLD_MULTI_OUTCOME_PATH = Path(__file__).parent / "data" / "manifold_multi_outcome_2026_08_05.json"

DURING_SUPPRESSION = date(2026, 8, 3)
AFTER_RESUME_DATE = date(2027, 1, 1)


@pytest.fixture(autouse=True)
def _reset_health() -> Iterator[None]:
    """The observation store is module state (the provider is a stateless callable
    with no handle back to the bot), so a leaked observation would poison every
    later "no findings" assertion. Mirrors the reset that
    ``reset_run_degradation_counters`` performs per run."""
    reset_provider_health()
    yield
    reset_provider_health()


@pytest.fixture(scope="module")
def captured_payloads() -> dict:
    return json.loads(_PAYLOADS_PATH.read_text())


def test_the_captured_payload_fixture_is_committed_and_complete() -> None:
    """The fixture must reach CI, and every venue's capture must be in it.

    A repo-wide ``*.json`` ignore rule made this file untracked when it was first
    written, which is the worst possible failure mode: it exists locally, so the
    fixture-backed tests pass on the developer's machine by construction, and in CI
    the whole module errors on a missing file. Asserting on tracked-ness is not
    possible from inside pytest, but asserting the file is present and carries all
    four venues is — and in CI that assertion IS the tracked-ness check.
    """
    assert _PAYLOADS_PATH.exists(), (
        f"{_PAYLOADS_PATH} missing. If it exists locally but not in CI, check .gitignore — "
        "the blanket *.json rule needs the !tests/data/*.json negation."
    )
    payload = json.loads(_PAYLOADS_PATH.read_text())
    assert set(payload) >= {"kalshi_events", "polymarket_search", "manifold_search", "predictit_all"}
    assert payload["_provenance"]["endpoints"].keys() == {"kalshi", "polymarket", "manifold", "predictit"}


def test_the_multi_close_kalshi_fixture_is_committed_and_still_divergent() -> None:
    """Same tracked-ness check as above, for the fixture the ranked port added.

    It also asserts the property the fixture EXISTS for, because a capture that lost its
    divergence would leave the tests that read it green while proving nothing: the event must
    carry differing nested ``close_time`` values AND a first nested market whose status
    disagrees with its siblings', which is what separates the max-over-nested / all-resolved
    derivations from the old ``nested[0]`` read.
    """
    assert _MULTI_CLOSE_PATH.exists(), (
        f"{_MULTI_CLOSE_PATH} missing. If it exists locally but not in CI, check .gitignore — "
        "the blanket *.json rule needs the !tests/data/*.json negation."
    )
    payload = json.loads(_MULTI_CLOSE_PATH.read_text())
    assert set(payload) >= {"events", "_provenance"}
    assert payload["_provenance"]["endpoints"].keys() == {"kalshi"}

    divergent = next(event for event in payload["events"] if event["event_ticker"] == "KXMARRIAGESTYLESKRAVITZ-HSZK")
    nested = divergent["markets"]
    assert len({market["close_time"] for market in nested}) == len(nested) > 1
    assert nested[0]["status"] not in {market["status"] for market in nested[1:]}
    assert divergent["settlement_sources"], "the event-level settlement sources are read by the join"
    assert all(market.get("settlement_sources") is None for market in nested)


def test_the_family_liquidity_kalshi_fixture_is_committed_and_still_straddles_its_boundaries() -> None:
    """Same tracked-ness check, for the fixture the family-scope liquidity change added.

    Both of its events were captured because they sit ON a boundary, and a capture that drifted off
    one would leave the tests reading it green while proving nothing. ``KXGOVWINS-27JAN01`` has to
    keep all three strikes OPEN and keep its first strike in a different liquidity bucket from the
    family sum — that straddle is the whole reason a live capture was needed rather than a hand-built
    dict, and only 2 frozen-universe events offer it. ``KXNETANYAHUPARDON-26`` has to keep exactly
    one strike live and that strike must not be ``nested[0]``, which is what makes a positional price
    read demonstrably wrong rather than merely unprincipled.
    """
    assert _FAMILY_LIQUIDITY_PATH.exists(), (
        f"{_FAMILY_LIQUIDITY_PATH} missing. If it exists locally but not in CI, check .gitignore — "
        "the blanket *.json rule needs the !tests/data/*.json negation."
    )
    payload = json.loads(_FAMILY_LIQUIDITY_PATH.read_text())
    assert set(payload) >= {"events", "_provenance"}
    assert payload["_provenance"]["endpoints"].keys() == {"kalshi"}

    straddle = next(event for event in payload["events"] if event["event_ticker"] == "KXGOVWINS-27JAN01")
    assert [market["status"] for market in straddle["markets"]] == ["active"] * 3
    family = kalshi_event_match(straddle, match_confidence=1.0, channel="universe_fuzzy")
    assert family is not None
    first_strike_only = kalshi_event_match(
        {**straddle, "markets": straddle["markets"][:1]}, match_confidence=1.0, channel="universe_fuzzy"
    )
    assert first_strike_only is not None
    assert _liquidity_label(first_strike_only) != _liquidity_label(family), (
        "the fixture no longer straddles a liquidity-label boundary, so the tests reading it "
        "can no longer tell the family sum from its first strike"
    )

    collapsed = next(event for event in payload["events"] if event["event_ticker"] == "KXNETANYAHUPARDON-26")
    live = [market for market in collapsed["markets"] if market["status"] == "active"]
    assert len(live) == 1 and live[0] is not collapsed["markets"][0], (
        "the fixture's collapsed family must keep its ONE live strike out of position 0"
    )


def test_the_manifold_multi_outcome_fixture_is_committed_and_still_splits_search_from_detail() -> None:
    """Same tracked-ness check, for the fixture the ``contractType=ALL`` flip added.

    The property it exists for is a DISAGREEMENT between two endpoints, so both halves have to
    keep holding or the tests reading it go vacuous. The search half must keep multi-outcome rows
    that carry no probability and no ``answers`` key — that absence is why a detail GET is the only
    render path. The detail half must keep an ``answers`` array on the multi-outcome markets and
    keep NONE on the BINARY control, which is what makes the array a real outcome-type
    discriminator rather than a convention. The tie in the MULTIPLE_CHOICE ladder is asserted where
    it is read, in ``test_tied_leading_answers_keep_the_arrays_own_order``.
    """
    assert _MANIFOLD_MULTI_OUTCOME_PATH.exists(), (
        f"{_MANIFOLD_MULTI_OUTCOME_PATH} missing. If it exists locally but not in CI, check .gitignore — "
        "the blanket *.json rule needs the !tests/data/*.json negation."
    )
    payload = json.loads(_MANIFOLD_MULTI_OUTCOME_PATH.read_text())
    assert set(payload) >= {"search_all", "detail_multiple_choice", "detail_multi_numeric", "detail_binary"}
    assert payload["_provenance"]["endpoints"].keys() == {"manifold"}

    multi = [row for row in payload["search_all"] if row["outcomeType"] != "BINARY"]
    assert len(multi) >= 2 and [row["outcomeType"] for row in payload["search_all"]].count("BINARY") >= 1, (
        "the search capture must keep BOTH shapes, or it cannot show what the flip admits"
    )
    for row in multi:
        assert row.get("probability") is None and "answers" not in row
        assert row.get("uniqueBettorCount") is not None, "the liquidity label must stay measurable on these rows"

    for key in ("detail_multiple_choice", "detail_multi_numeric"):
        answers = payload[key]["answers"]
        assert len(answers) >= 3, f"{key} must keep enough answers to exercise the cap"
        assert all({"text", "probability"} <= set(answer) for answer in answers)
    assert "answers" not in payload["detail_binary"], "the BINARY control is the whole discriminator"
    assert payload["detail_binary"]["probability"] is not None


def _observe(
    venue: str,
    *,
    qid: int = 1,
    candidates: int = 3,
    rows: int = 3,
    fields_present: frozenset[str] | None = None,
) -> None:
    """Record one venue observation, defaulting to a healthy shape."""
    if fields_present is None:
        fields_present = frozenset(VENUE_EXPECTED_LIQUIDITY_FIELDS[venue])
    record_venue_observation(
        VenueObservation(
            qid=qid,
            venue=venue,
            candidates_pre_filter=candidates,
            rows_post_filter=rows,
            liquidity_fields_present=fields_present,
        )
    )


def _kalshi_rows(events: list[dict]) -> list[MarketMatch]:
    """A candidate row per catalogue event. No query and no floor.

    Ranked retrieval builds a row for EVERY event and lets the ranker select, so these tests
    read the same unfiltered rows production records field presence over — which is the point:
    Signal A measures PARSER health, so it must be immune to which rows the ranker happened to
    keep. A helper that took a query would be measuring a filter that no longer exists.
    """
    rows = [kalshi_event_match(event, match_confidence=1.0, channel="universe_fuzzy") for event in events]
    return [row for row in rows if row is not None]


def _predictit_rows(markets: list[dict]) -> list[MarketMatch]:
    rows = [predictit_market_match(market, match_confidence=1.0, channel="universe_fuzzy") for market in markets]
    return [row for row in rows if row is not None]


def _fields_present(rows: list[MarketMatch], venue: str) -> frozenset[str]:
    """Which declared liquidity fields parsed non-None on >=1 row.

    Byte-identical logic to the recording site in ``prediction_market._record_venue_health``,
    so these tests measure exactly what production records — including that it reads the venue's
    POOL rows rather than the ranked ones.
    """
    return frozenset(
        name for name in VENUE_EXPECTED_LIQUIDITY_FIELDS[venue] if any(getattr(row, name) is not None for row in rows)
    )


# Signal A — market_field_contract


class TestSignalMarketFieldContract:
    """A declared liquidity field dead across 100% of a venue's POOL rows.

    The threshold is 100%-of-rows rather than a fraction because absent and zero
    are ALREADY distinct in the code: ``_liquidity_label`` renders
    ``no-liquidity-data`` if and only if the parsed value is None, while a
    brand-new zero-volume market parses to 0.0 and renders ``thin``. So there is no
    noise floor to threshold against, and three rows is a conclusive denominator.

    The denominator is the POOL, not the render. Both halves of the rule read
    ``candidates_pre_filter``, so a broken parser is equally visible whether or not
    the ranker kept any of that venue's rows.
    """

    def test_the_signal_agrees_with_the_rendered_label_on_real_kalshi_rows(self, captured_payloads: dict) -> None:
        """The signal must never disagree with the ``signal`` column a forecaster reads.

        That equivalence is the whole reason presence is computed off the parsed
        ``MarketMatch`` rather than the upstream payload, and it is what makes the
        signal true-positive on the live Kalshi defect without needing to know
        anything about field NAMES: whenever every row renders ``no-liquidity-data``,
        a finding must exist, and whenever a row renders a real label, there must be
        none.

        Deliberately builder-agnostic. Pinning the field names here would give the same
        answer as the parser by construction — exactly the hand-written-fixture trap
        that let the wrong names live for three weeks — and would have to be rewritten
        the moment the field-name fix lands. This assertion holds identically before
        and after, and it FAILS if the two sides ever drift apart.
        """
        rows = _kalshi_rows(captured_payloads["kalshi_events"]["events"])
        assert rows, "captured payload must yield at least one Kalshi row"

        _observe("kalshi", rows=len(rows), fields_present=_fields_present(rows, "kalshi"))
        every_row_blank = all(_liquidity_label(row) == "no-liquidity-data" for row in rows)
        fired = bool([f for f in provider_degradation_findings() if f.signal == SIGNAL_MARKET_FIELD_CONTRACT])
        assert fired == every_row_blank

    def test_the_signal_agrees_with_the_rendered_label_on_real_polymarket_rows(self, captured_payloads: dict) -> None:
        """The same equivalence on Polymarket, where the live defect is narrower: only
        ``open_interest`` is dead (Gamma carries ``openInterest`` at the EVENT level and
        the builder reads it off the nested market), while ``total_volume`` populates.

        So the LABEL stays informative — ``_liquidity_label`` scores on whichever field
        survived — and the rendered ``OI`` column reads ``-`` on every row. A per-FIELD
        rule catches that; a rule keyed on the label going blank would not. Hence the
        assertion here is per-field rather than the label equivalence above.
        """
        rows = parse_polymarket_matches(captured_payloads["polymarket_search"], width=RETRIEVAL_WIDTH["polymarket"])
        assert rows, "captured payload must yield at least one Polymarket row"

        _observe("polymarket", rows=len(rows), fields_present=_fields_present(rows, "polymarket"))
        dead_fields = {
            name
            for name in VENUE_EXPECTED_LIQUIDITY_FIELDS["polymarket"]
            if all(getattr(row, name) is None for row in rows)
        }
        findings = [f for f in provider_degradation_findings() if f.signal == SIGNAL_MARKET_FIELD_CONTRACT]
        if dead_fields:
            assert len(findings) == 1
            assert set(str(findings[0].detail["fields"]).split(",")) == dead_fields
        else:
            assert findings == []

    def test_manifold_captured_payload_produces_no_finding(self, captured_payloads: dict) -> None:
        """Manifold's upstream names are the ones the builder reads, so this venue is
        healthy today and must stay silent — the control that shows Signal A is not
        simply firing on every venue."""
        rows = parse_manifold_matches(captured_payloads["manifold_search"], width=RETRIEVAL_WIDTH["manifold"])
        assert rows, "captured payload must yield at least one Manifold row"

        _observe("manifold", rows=len(rows), fields_present=_fields_present(rows, "manifold"))
        assert provider_degradation_findings() == []

    def test_predictit_declares_no_fields_and_populates_none(self, captured_payloads: dict) -> None:
        """Half of the declared-fields map pin, on the venue whose entry is empty.

        PredictIt's ``/marketdata/all/`` dump carries no volume, liquidity, or
        open-interest field anywhere, so its blank ``signal`` column is honest. If a
        future builder started populating one of these from the real payload, the map
        would stay silent about a field whose death should alert — this is where that
        shows up.
        """
        rows = _predictit_rows(captured_payloads["predictit_all"]["markets"])
        assert rows, "captured payload must yield at least one PredictIt row"
        assert VENUE_EXPECTED_LIQUIDITY_FIELDS["predictit"] == ()
        for row in rows:
            assert row.total_volume is None
            assert row.open_interest is None
            assert row.num_bettors is None

    def test_every_declared_field_is_populated_from_the_captured_payloads(self, captured_payloads: dict) -> None:
        """The map pin, on all three venues that declare fields.

        The map is the design's one piece of DECLARED knowledge, and it cuts both ways. A
        builder that stops reading a field leaves the map claiming the field is supplied,
        and the signal then alerts forever on a fiction. Equally, a map that under-declares
        leaves a real field unmonitored — the D1 shape.

        This is also the assertion that would have caught D1 itself: on the pre-fix builder
        Kalshi populated neither field and Polymarket's ``open_interest`` was dead, so this
        test fails on all three counts against the pre-fix code and passes after.
        """
        parsed = {
            "kalshi": _kalshi_rows(captured_payloads["kalshi_events"]["events"]),
            "polymarket": parse_polymarket_matches(
                captured_payloads["polymarket_search"], width=RETRIEVAL_WIDTH["polymarket"]
            ),
            "manifold": parse_manifold_matches(captured_payloads["manifold_search"], width=RETRIEVAL_WIDTH["manifold"]),
        }
        for venue, rows in parsed.items():
            assert rows, f"captured payload must yield at least one {venue} row"
            assert _fields_present(rows, venue) == set(VENUE_EXPECTED_LIQUIDITY_FIELDS[venue]), (
                f"{venue}: declared liquidity fields disagree with what the builder populates"
            )

    def test_the_map_covers_exactly_the_four_venues(self) -> None:
        """A venue absent from the map is silently un-monitored, which is the failure
        mode this whole signal exists to end. ``generation.VENUE_ORDER`` is the authority on
        which venues exist."""
        assert set(VENUE_EXPECTED_LIQUIDITY_FIELDS) == {"polymarket", "kalshi", "manifold", "predictit"}

    def test_predictit_all_none_rows_are_exempt(self) -> None:
        """The honest-blank case. PredictIt genuinely exposes no liquidity fields, so
        its 141/141 archived blanks are a fact about PredictIt, not a defect."""
        _observe("predictit", rows=3, fields_present=frozenset())
        assert provider_degradation_findings() == []

    def test_a_dead_field_fires_once_per_venue(self) -> None:
        """Both Kalshi fields dead across three rows is ONE defect, not six events:
        the count carries no diagnostic weight, the log line carries all of it."""
        _observe("kalshi", rows=3, fields_present=frozenset())
        findings = provider_degradation_findings()

        assert len(findings) == 1
        assert findings[0].signal == SIGNAL_MARKET_FIELD_CONTRACT
        assert findings[0].venue == "kalshi"
        assert findings[0].detail["fields"] == "total_volume,open_interest"
        assert findings[0].detail["pool_rows"] == 3
        assert provider_degradation_count() == 1

    def test_zero_is_not_absent(self) -> None:
        """The distinction the whole signal rests on. A brand-new market with
        genuinely zero volume parses to 0.0, renders ``thin``, and must not alert —
        only an ABSENT (None) field means the upstream key vanished."""
        row = MarketMatch(
            platform="kalshi",
            market_title="brand new market",
            market_url="https://kalshi.com/markets/NEW",
            implied_prob_yes=0.5,
            bid=0.49,
            ask=0.51,
            spread=0.02,
            volume_24h=0.0,
            close_time=None,
            is_resolved=False,
            match_confidence=0.9,
            raw_rules="",
            total_volume=0.0,
            open_interest=0.0,
        )
        assert _liquidity_label(row) == "thin"

        _observe("kalshi", rows=1, fields_present=_fields_present([row], "kalshi"))
        assert provider_degradation_findings() == []

    def test_a_venue_with_an_empty_pool_is_never_evaluated(self) -> None:
        """Why a market-less question stays silent under Signal A: the rule only looks
        at venues that put at least one candidate in the pool. This is the whole
        false-positive defence, and it is a statement about the POOL — nothing about
        the render can silence the rule."""
        _observe("kalshi", candidates=0, rows=0, fields_present=frozenset())
        assert provider_degradation_findings() == []

    def test_a_dead_field_fires_even_when_the_ranker_rendered_nothing(self) -> None:
        """The no-false-negative case, and the regression this gate exists for.

        A zero-row render is routine under ranked selection: ``RENDER_BUDGET`` is a
        global ceiling of 8 across four venues against pool widths of 100/60/60/~197,
        and the bake-off's own diagnostics put a venue at zero rendered rows on 42% of
        question-runs for kalshi. Gating on the rendered count therefore let the SAME
        dead parser alert or stay silent purely on the model's pick — which is the
        2026-07-12 Kalshi hole (labels blank on 100% of rows for weeks) reopened.
        """
        _observe("kalshi", candidates=40, rows=0, fields_present=frozenset())
        findings = provider_degradation_findings()

        assert len(findings) == 1
        assert findings[0].signal == SIGNAL_MARKET_FIELD_CONTRACT
        assert findings[0].venue == "kalshi"
        assert findings[0].detail["pool_rows"] == 40
        assert provider_degradation_count() == 1

    def test_an_enumerable_venue_with_a_healthy_catalogue_always_reports_candidates(self) -> None:
        """The enumerable venues no longer depend on a query at all — their whole
        catalogue enters the pool — so a healthy catalogue ALWAYS yields candidates.

        That is what makes Signal A's pool gate safe for Kalshi and PredictIt: their
        denominator is a fact about the catalogue rather than about the ranker, so a
        broken parser on either is always evaluated. A refactor that reintroduced a
        per-query filter on an enumerable venue would break this and could silence the
        signal on the two venues whose only other alarm (Signal C) fires solely on an
        EMPTY catalogue.
        """
        payload = json.loads(_PAYLOADS_PATH.read_text())
        kalshi = _kalshi_rows(payload["kalshi_events"]["events"])
        predictit = _predictit_rows(payload["predictit_all"]["markets"])

        assert len(kalshi) == len(
            [e for e in payload["kalshi_events"]["events"] if e.get("title") or e.get("sub_title")]
        )
        assert len(predictit) == len(payload["predictit_all"]["markets"])
        for venue, rows in (("kalshi", kalshi), ("predictit", predictit)):
            assert rows, f"{venue}: a healthy catalogue must always reach the pool"
            _observe(venue, candidates=len(rows), rows=0, fields_present=_fields_present(rows, venue))

        # Zero RENDERED rows on both, and the signal still reads the real parser state:
        # these captured payloads populate every declared field, so nothing fires.
        assert provider_degradation_findings() == []

    def test_presence_on_one_question_of_two_does_not_fire(self) -> None:
        """The 100%-vs-any distinction, at the granularity where it is decidable.

        Presence already collapses across a question's rows at the recording site (a
        field counts as present if ANY row carries it), so within one question there
        is no any/all difference to test. Across questions there is: one question
        whose Kalshi rows carried ``total_volume`` PROVES the upstream key exists, so
        the other question's blanks are a property of those particular markets, not a
        contract break. A rule requiring presence on every question would alert here
        — and would alert on any venue whose thinly-traded markets happen to omit a
        field on one question of a two-question run.
        """
        _observe("kalshi", qid=1, rows=3, fields_present=frozenset({"total_volume", "open_interest"}))
        _observe("kalshi", qid=2, rows=3, fields_present=frozenset())
        assert provider_degradation_findings() == []

    def test_partial_presence_within_a_question_does_not_fire(self) -> None:
        """One row carrying the field is enough to prove the upstream key exists."""
        record_venue_observation(
            VenueObservation(
                qid=1,
                venue="kalshi",
                candidates_pre_filter=3,
                rows_post_filter=3,
                liquidity_fields_present=frozenset({"total_volume", "open_interest"}),
            )
        )
        assert provider_degradation_findings() == []

    def test_one_dead_field_of_two_still_fires(self) -> None:
        """Polymarket's real shape pre-fix: ``total_volume`` populated on every row
        while ``open_interest`` was None on all 41. A per-field rule catches that; a
        rule keyed on "the label went blank" would not, because
        ``_liquidity_label`` falls back to whichever field survived."""
        _observe("polymarket", rows=3, fields_present=frozenset({"total_volume"}))
        findings = provider_degradation_findings()

        assert len(findings) == 1
        assert findings[0].detail["fields"] == "open_interest"


# Signal C — catalogue_empty


class TestSignalCatalogueEmpty:
    """A prefetch reported SUCCESS and handed the local matcher an empty catalogue.

    Kalshi and PredictIt enumerate a prefetched catalogue into the pool, so their fetch tally
    reports success even when the catalogue is empty and there is nothing to enumerate. Signals
    A and B both read rows, so this is the one class they share a blind spot on — and under
    ranked retrieval it is the ONLY alarm those two venues have, since their
    ``candidates_pre_filter`` is a whole catalogue and therefore never zero when healthy. An
    empty Kalshi catalogue now zeroes the settlement-source join AND the fuzzy channel, so the
    signal guards more than it used to.
    """

    def test_successful_empty_prefetch_fires(self) -> None:
        record_catalogue_size(qid=1, source="kalshi_events", entries=0, fetch_ok=True)
        findings = provider_degradation_findings()

        assert len(findings) == 1
        assert findings[0].signal == SIGNAL_CATALOGUE_EMPTY
        assert findings[0].venue == "kalshi_events"
        assert provider_degradation_count() == 1

    def test_failed_prefetch_does_not_double_count(self) -> None:
        """Pins the no-double-count boundary. A FAILED fetch is already alertable via
        ``prediction_market_source_losses``; counting it here too would report one
        outage as two, and "check the response parsing" is the wrong remedy for a
        503."""
        record_catalogue_size(qid=1, source="kalshi_events", entries=0, fetch_ok=False)
        assert provider_degradation_findings() == []

    def test_non_empty_catalogue_is_silent(self) -> None:
        record_catalogue_size(qid=1, source="kalshi_events", entries=12_370, fetch_ok=True)
        record_catalogue_size(qid=1, source="predictit_markets", entries=8, fetch_ok=True)
        assert provider_degradation_findings() == []

    def test_one_populated_question_clears_the_run(self) -> None:
        """The whole-run conjunction: one question whose prefetch came back populated
        proves the catalogue is reachable, so a single empty read is a blip, not a dead
        matcher."""
        record_catalogue_size(qid=1, source="kalshi_events", entries=0, fetch_ok=True)
        record_catalogue_size(qid=2, source="kalshi_events", entries=12_370, fetch_ok=True)
        assert provider_degradation_findings() == []


# Reset, suppression, and the log line


class TestPerRunReset:
    def test_reset_clears_findings(self) -> None:
        """Without a per-run reset the observations leak across runs sharing a
        process and across tests, poisoning every later ``alertable == 0``
        assertion — the footgun ``reset_source_loss_counter`` already exists for."""
        _observe("kalshi", rows=3, fields_present=frozenset())
        assert provider_degradation_findings()

        reset_provider_health()
        assert provider_degradation_findings() == []
        assert provider_degradation_count() == 0

    def test_orchestrator_reset_clears_provider_health(self) -> None:
        """The wiring, not just the function: ``reset_run_degradation_counters`` is
        what ``forecast_questions`` calls, so the reset has to be reachable from
        there."""
        orch = ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())
        _observe("kalshi", rows=3, fields_present=frozenset())
        assert orch.provider_degradation_count == 1

        orch.reset_run_degradation_counters()
        assert orch.provider_degradation_count == 0


class TestSuppression:
    """The dated per-venue suppression, following ``credit_alerts_active``'s contract:
    reads the clock at CALL time, accepts an injected ``today``, suppresses the EXIT
    CODE only, and is dated rather than a bare boolean so it cannot outlive the
    season unnoticed.
    """

    def test_ships_empty(self) -> None:
        """Both degradations this machinery was built for are being fixed in the same
        round, so suppressing either would hide the fix's own verification."""
        assert PROVIDER_DEGRADATION_SUPPRESSED_UNTIL == {}

    def test_unlisted_venue_is_always_alertable(self) -> None:
        assert provider_degradation_alerts_active("kalshi", DURING_SUPPRESSION) is True

    def test_inside_the_window_the_finding_is_logged_but_not_counted(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:

        monkeypatch.setitem(PROVIDER_DEGRADATION_SUPPRESSED_UNTIL, "manifold", date(2026, 9, 10))
        _observe("kalshi")
        _observe("predictit")
        _observe("polymarket")
        _observe("manifold", fields_present=frozenset())

        findings = provider_degradation_findings(DURING_SUPPRESSION)
        assert len(findings) == 1
        assert findings[0].is_alertable is False
        assert provider_degradation_count(DURING_SUPPRESSION) == 0

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.provider_health"):
            log_provider_degradation_summary(DURING_SUPPRESSION)
        messages = [record.getMessage() for record in caplog.records]

        # Visible, not hidden: the marker shows the arithmetic and names the resume
        # date, and the per-finding line still fires. A run reading alertable=0
        # alongside real degradation is exactly the shape that most needs a written
        # record — the drained-donated-key incident is the precedent.
        marker = next(msg for msg in messages if msg.startswith("PROVIDER_DEGRADATION:"))
        assert "findings=1 alertable=0 suppressed=1" in marker
        assert "manifold:market_field_contract suppressed until 2026-09-10" in marker
        assert "run stays green" in marker
        assert any("PROVIDER DEGRADATION (suppressed until 2026-09-10)" in msg for msg in messages)

    def test_past_the_resume_date_it_alerts_again(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both dates injected, never read from the clock, so both branches keep
        running after the real date passes."""
        monkeypatch.setitem(PROVIDER_DEGRADATION_SUPPRESSED_UNTIL, "manifold", date(2026, 9, 10))
        _observe("kalshi")
        _observe("predictit")
        _observe("polymarket")
        _observe("manifold", fields_present=frozenset())

        assert provider_degradation_count(AFTER_RESUME_DATE) == 1

    def test_suppressing_one_venue_does_not_blind_the_others(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Per-venue rather than global: accepting a dead Manifold must not hide a
        dead Kalshi."""
        monkeypatch.setitem(PROVIDER_DEGRADATION_SUPPRESSED_UNTIL, "manifold", date(2026, 9, 10))
        _observe("kalshi", rows=3, fields_present=frozenset())
        _observe("predictit")
        _observe("polymarket")
        _observe("manifold", fields_present=frozenset())

        findings = provider_degradation_findings(DURING_SUPPRESSION)
        assert {(f.venue, f.is_alertable) for f in findings} == {("kalshi", True), ("manifold", False)}
        assert provider_degradation_count(DURING_SUPPRESSION) == 1


class TestSummaryLine:
    def test_marker_fires_at_zero(self, caplog: pytest.LogCaptureFixture) -> None:
        """A measured zero is a positive statement of provider health — the same
        reasoning that put FORECASTERS_SURVIVED next to FORECASTER_DROPS. Without
        this line, "no degradation" and "the check never ran" look identical."""

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.provider_health"):
            log_provider_degradation_summary()

        marker = next(msg for msg in caplog.messages if msg.startswith("PROVIDER_DEGRADATION:"))
        assert "findings=0 alertable=0 suppressed=0" in marker
        assert "detail=[]" in marker

    def test_marker_detail_round_trips_through_json(self, caplog: pytest.LogCaptureFixture) -> None:

        _observe("kalshi", rows=3, fields_present=frozenset())
        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.provider_health"):
            log_provider_degradation_summary()

        marker = next(msg for msg in caplog.messages if msg.startswith("PROVIDER_DEGRADATION:"))
        payload = json.loads(marker.split("detail=", 1)[1])
        assert payload == [
            {
                "signal": SIGNAL_MARKET_FIELD_CONTRACT,
                "venue": "kalshi",
                "questions": 1,
                "fields": "total_volume,open_interest",
                "pool_rows": 3,
            }
        ]

    def test_per_finding_warn_names_the_remedy(self, caplog: pytest.LogCaptureFixture) -> None:
        """D1's real cost was three weeks of nobody knowing what to do about a blank
        column, so the WARN the operator reads first has to name the likely cause and
        the file to open."""

        _observe("kalshi", rows=3, fields_present=frozenset())
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.provider_health"):
            log_provider_degradation_summary()

        warning = next(msg for msg in caplog.messages if msg.startswith("PROVIDER DEGRADATION"))
        assert "(alertable)" in warning
        assert "total_volume,open_interest" in warning
        assert "prediction_market.py" in warning
