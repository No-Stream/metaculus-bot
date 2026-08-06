"""What a prediction-market run reports about itself: per-source tokens, and provider health.

Two reporting seams that share one question — is a degraded run visible?

- `MarketSnapshot.sources`, the seven per-source tokens the orchestrator drains into the
  provider-diagnostics line, plus every path that turns one into an alertable loss,
- the numbers a real fetch hands the provider-health degradation rules, which under ranked
  selection come from two different places (field presence and `candidates_pre_filter` from the
  venue's POOL rows, `rows_post_filter` from the RENDERED ones).

Fakes, payload fixtures and the `handlers()` baseline live in `tests/market_retrieval_fakes.py`.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from metaculus_bot.research import prediction_market as pmp
from metaculus_bot.research.prediction_market import _reset_session_caches
from metaculus_bot.research.provider_diagnostics import _partial_loss_suffix, pop_provider_detail
from metaculus_bot.research.provider_health import (
    VENUE_EXPECTED_LIQUIDITY_FIELDS,
    provider_degradation_findings,
    recorded_observations,
    reset_provider_health,
)
from tests import market_retrieval_fakes as _fakes
from tests.market_retrieval_fakes import CANDIDATE_LINE_RE as _CANDIDATE_LINE_RE
from tests.market_retrieval_fakes import KALSHI_EVENTS_URL as _KALSHI_EVENTS_URL
from tests.market_retrieval_fakes import MANIFOLD_DETAIL_URL as _MANIFOLD_DETAIL_URL
from tests.market_retrieval_fakes import MANIFOLD_SEARCH_URL as _MANIFOLD_SEARCH_URL
from tests.market_retrieval_fakes import POLY_URL as _POLY_URL
from tests.market_retrieval_fakes import PREDICTIT_URL as _PREDICTIT_URL
from tests.market_retrieval_fakes import FakeResponse, FakeSession
from tests.market_retrieval_fakes import fetch_snapshot as _fetch
from tests.market_retrieval_fakes import handlers as _handlers
from tests.market_retrieval_fakes import market_llm as _market_llm
from tests.market_retrieval_fakes import no_backoff as _no_backoff
from tests.market_retrieval_fakes import rank_one_per_venue as _rank_one_per_venue

# Bound by assignment rather than imported — see the note in tests/test_prediction_market_transport.py.
reset_provider_caches = _fakes.reset_provider_caches
mock_question = _fakes.mock_question
polymarket_payload = _fakes.polymarket_payload
manifold_payload = _fakes.manifold_payload
kalshi_events_payload = _fakes.kalshi_events_payload
predictit_payload = _fakes.predictit_payload

# The seven keys a complete run reports. Pinned as a set here because the diagnostics line, the
# archive and the source-loss counter all read this dict, and a silently-dropped key is a source
# nobody is watching any more.
_ALL_SOURCES = frozenset(
    {"polymarket", "manifold", "kalshi", "predictit", "manifold_detail", "query_author", "ranking"}
)


class TestSnapshotSourceDiagnostics:
    """The seven per-source tokens on `MarketSnapshot.sources`.

    Drained by the orchestrator into the provider-diagnostics line, so PARTIAL degradation — one
    dead stage while the rest of the pipeline runs — is visible per question rather than only in
    an aggregate counter. A token starting with "ok"/"none" is benign; everything else is a lost
    source that bumps the alertable counter.
    """

    @pytest.fixture(autouse=True)
    def _no_retry_backoff(self, monkeypatch):
        """These tests drive 503 retry-exhaustion paths; the real 0.5s backoff between attempts
        would dominate their runtime."""
        _no_backoff(monkeypatch)

    @pytest.mark.asyncio
    async def test_a_healthy_run_reports_all_seven_sources_and_no_loss(self, mock_question, kalshi_events_payload):
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        assert set(snapshot.sources) == _ALL_SOURCES
        assert all(token.startswith(("ok", "none")) for token in snapshot.sources.values()), snapshot.sources
        # No lost source -> the diagnostics suffix stays empty (byte-identical to a healthy line).
        assert _partial_loss_suffix({"sources": snapshot.sources}) == ""
        assert pmp.prediction_market_source_losses() == 0

    @pytest.mark.asyncio
    async def test_all_live_but_empty_stays_benign_none(self, mock_question):
        """The other direction: live venues with nothing to say must ALL read `none` with an
        empty suffix, so the token vocabulary never cries outage on a healthy no-match question.

        The two enumerable venues are the exception and are correct to be: their catalogues
        populated, so they report `ok(N)` on the candidates they contributed even though the
        ranker kept none.
        """
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, {"events": [], "cursor": ""}),
                _PREDICTIT_URL: FakeResponse(200, {"markets": []}),
            }
        )

        snapshot = await _fetch(mock_question, handlers)

        assert snapshot.sources == {
            "polymarket": "none",
            "manifold": "none",
            "kalshi": "none",
            "predictit": "none",
            "manifold_detail": "none",
            "query_author": "ok(3)",
            "ranking": "none",
        }
        assert _partial_loss_suffix({"sources": snapshot.sources}) == ""
        assert pmp.prediction_market_source_losses() == 0

    @pytest.mark.asyncio
    async def test_a_lost_catalogue_is_a_venue_loss_and_bumps_the_degraded_counter(self, mock_question):
        """A failed catalogue pull bumps BOTH counters, which is deliberate over-counting rather
        than a bug: the two carry different marker fields, and the catalogue is now the
        generation backbone — it feeds the settlement-source join AND the fuzzy channel, so its
        loss is worse than it used to be, not better."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(503, text="service unavailable")})

        snapshot = await _fetch(mock_question, handlers)

        assert snapshot.sources["kalshi"] == "error(http_503)"
        assert "lost=kalshi:error(http_503)" in _partial_loss_suffix({"sources": snapshot.sources})
        assert pmp.kalshi_catalogue_fetch_failures() == 1
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_a_rate_limited_catalogue_reports_incomplete_without_a_second_violation(self, mock_question):
        """Re-asking a rate limiter 0.5s later is not a retry, it is a second violation. So a 429
        stops pagination dead and reports the pull incomplete, which is also what keeps a
        rate-limited exchange from being cached as a short catalogue for 6h."""
        session = FakeSession(_handlers(**{_KALSHI_EVENTS_URL: FakeResponse(429, text="slow down")}))

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", _market_llm()),
            patch.object(pmp, "_get_session", lambda: session),
        ):
            snapshot = await pmp.fetch_market_snapshot(mock_question, timeout=5.0)

        assert session._call_counts[_KALSHI_EVENTS_URL] == 1
        assert snapshot.sources["kalshi"] == "error(http_429)"
        assert "events" not in pmp._KALSHI_CACHE

    @pytest.mark.asyncio
    async def test_a_shapeless_200_reads_as_a_LOSS_at_every_venue_and_caches_nothing(
        self, mock_question, kalshi_events_payload
    ):
        """A 200 whose top-level shape changed is a loss at the seam, not a benign `none`.

        The transport already converts every non-200 and every undecodable body to `None`, so
        this shape is exactly "the endpoint still answers, but not with its documented contract".
        Parsing it to `[]` made each venue publish `none` and bump nothing, and on PredictIt it
        was worse than a lost signal: `_predictit_universe` caches a successful dump for 6h, so
        one malformed response pinned an EMPTY universe as healthy and every later question read
        it back with no HTTP at all. The cache assertion is the load-bearing half.
        """
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
                _POLY_URL: FakeResponse(200, ["not", "an", "object"]),
                _MANIFOLD_SEARCH_URL: FakeResponse(200, {"markets": []}),
                _PREDICTIT_URL: FakeResponse(200, {"nope": []}),
            }
        )
        # Only the provider-health class carries a reset fixture, and the catalogue assertion
        # below reads the module-scoped store, so this test resets it itself rather than reading
        # an earlier test's healthy PredictIt observation back as its own.
        reset_provider_health()

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        assert snapshot.sources["polymarket"] == "error(all_queries_failed)"
        assert snapshot.sources["manifold"] == "error(all_queries_failed)"
        assert snapshot.sources["predictit"] == "error(all_queries_failed)"
        assert "markets" not in pmp._PREDICTIT_CACHE, "a malformed dump must not be pinned as a healthy universe"
        assert pmp.prediction_market_source_losses() == 3
        # Signal C must not ALSO fire: a failed fetch is the source-loss counter's business, and
        # counting it here would report one outage twice.
        _, catalogues = recorded_observations()
        assert "predictit_markets" not in {observation.source for observation in catalogues}

    @pytest.mark.asyncio
    async def test_a_dead_query_author_is_an_additive_loss_not_a_silenced_pipeline(
        self, mock_question, kalshi_events_payload
    ):
        """The author's output is ADDITIVE, so its failure costs no recall — the deterministic
        query set still runs and the venues still answer. It is still reported, because a
        permanently dead author would otherwise be invisible, which is the exact failure class
        the degradation counters exist for."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        snapshot = await _fetch(
            mock_question, handlers, author="I could not help with that.", ranking=_rank_one_per_venue
        )

        assert snapshot.sources["query_author"] == "error(unusable)"
        assert "lost=query_author:error(unusable)" in _partial_loss_suffix({"sources": snapshot.sources})
        assert pmp.prediction_market_source_losses() == 1
        # The pipeline ran anyway: venues answered and rows rendered.
        assert snapshot.matches
        assert snapshot.sources["kalshi"].startswith("ok(")

    @pytest.mark.asyncio
    async def test_a_venue_outage_is_a_loss_and_the_healthy_venues_still_land(
        self, mock_question, manifold_payload, kalshi_events_payload
    ):
        """A 503'd venue reads as a LOSS, never as `none`."""
        handlers = _handlers(
            **{
                _POLY_URL: FakeResponse(503, text="service unavailable"),
                _MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload),
                _KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
            }
        )

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        assert snapshot.sources["polymarket"] == "error(all_queries_failed)"
        assert "lost=polymarket:" in _partial_loss_suffix({"sources": snapshot.sources})
        assert "manifold" in {row.platform for row in snapshot.matches}
        assert snapshot.sources["manifold"].startswith("ok(")
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_one_lost_query_of_several_reads_as_partial(self, mock_question, manifold_payload):
        """The venue fan-out issues every deduped query, so losing one of them used to publish a
        clean `ok(N)` off the survivors."""
        seen: list[str] = []

        def _manifold(params: dict[str, Any]) -> FakeResponse:
            term = params.get("term") or ""
            seen.append(term)
            if term == seen[0]:
                return FakeResponse(503, text="service unavailable")
            return FakeResponse(200, manifold_payload)

        handlers = _handlers(**{_MANIFOLD_SEARCH_URL: _manifold})

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        assert len(seen) > 1, "the venue must issue every deduped query, not one"
        assert snapshot.sources["manifold"].startswith("partial("), snapshot.sources["manifold"]
        assert "lost=manifold:partial(" in _partial_loss_suffix({"sources": snapshot.sources})
        assert "manifold" in {row.platform for row in snapshot.matches}, "the surviving queries' rows still land"
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_a_total_blackout_bumps_once_per_lost_source(self, mock_question):
        """Every venue down. Each is its own loss — the published diagnostics line must not read
        healthy through a blackout, and one bump for four dead venues would understate it."""
        handlers = {
            url: FakeResponse(503, text="service unavailable")
            for url in (_POLY_URL, _MANIFOLD_SEARCH_URL, _MANIFOLD_DETAIL_URL, _KALSHI_EVENTS_URL, _PREDICTIT_URL)
        }

        snapshot = await _fetch(mock_question, handlers)

        suffix = _partial_loss_suffix({"sources": snapshot.sources})
        for venue in ("polymarket", "manifold", "predictit"):
            assert snapshot.sources[venue] == "error(all_queries_failed)", venue
            assert f"{venue}:error(all_queries_failed)" in suffix, venue
        assert snapshot.sources["kalshi"] == "error(http_503)"
        assert pmp.prediction_market_source_losses() == 4

    @pytest.mark.asyncio
    async def test_manifold_enrichment_reports_none_when_there_is_nothing_to_enrich(
        self, mock_question, kalshi_events_payload
    ):
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        assert snapshot.sources["manifold_detail"] == "none"
        assert pmp.prediction_market_source_losses() == 0

    @pytest.mark.asyncio
    async def test_manifold_enrichment_fills_the_rules_text_the_search_listing_omits(
        self, mock_question, manifold_payload
    ):
        """Without this every Manifold candidate reaches the ranker title-only, and the prompt's
        stated "single most reliable cue" is the settlement/rules text. It must run BEFORE the
        prompt is built, because it mutates the pool rows in place and `apply_picks` copies
        them."""
        detail = {"textDescription": "Resolves YES on an FAA-confirmed orbital insertion."}
        handlers = _handlers(
            **{
                _MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload),
                _MANIFOLD_DETAIL_URL: FakeResponse(200, detail),
            }
        )
        captured: list[str] = []

        def _capture(prompt: str) -> str:
            captured.append(prompt)
            return _rank_one_per_venue(prompt)

        snapshot = await _fetch(mock_question, handlers, ranking=_capture)

        assert snapshot.sources["manifold_detail"].startswith("ok(")
        assert "FAA-confirmed orbital insertion" in captured[0]
        assert pmp.prediction_market_source_losses() == 0

    @pytest.mark.asyncio
    async def test_a_totally_lost_enrichment_fan_out_is_the_only_reported_case(self, mock_question, manifold_payload):
        """A lost detail GET costs rules text, never recall, so only a TOTAL loss is reported: a
        partial fan-out has nothing actionable to alert on."""
        handlers = _handlers(
            **{
                _MANIFOLD_SEARCH_URL: FakeResponse(200, manifold_payload),
                _MANIFOLD_DETAIL_URL: FakeResponse(503, text="service unavailable"),
            }
        )

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        assert snapshot.sources["manifold_detail"] == "error(all_details_failed)"
        assert snapshot.sources["manifold"].startswith("ok("), "the SEARCH was healthy; only the detail GET died"
        assert pmp.prediction_market_source_losses() == 1

    @pytest.mark.asyncio
    async def test_the_provider_records_the_sources_map_into_the_registry(
        self, monkeypatch, mock_question, kalshi_events_payload
    ):
        """The `_fetch` ResearchCallable records the map keyed by (qid, 'prediction_market') for
        the orchestrator to drain into the comment-only diagnostics block."""
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "true")
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(503, text="service unavailable")})

        with (
            patch.object(pmp, "build_llm_with_openrouter_fallback", _market_llm()),
            patch.object(pmp, "_get_session", lambda: FakeSession(handlers)),
        ):
            await pmp.prediction_market_provider()(mock_question)

        detail = pop_provider_detail(mock_question.id_of_question, "prediction_market")
        assert set(detail.get("sources", {})) == _ALL_SOURCES
        assert detail["sources"]["kalshi"] == "error(http_503)"


class TestProviderHealthRecording:
    """The recording seam: what a real snapshot fetch hands the degradation rules.

    The rules themselves are tested in `tests/test_provider_health.py`. These tests pin the
    numbers this module feeds them, which is where the risk actually lives — and under ranked
    selection the two counts come from DIFFERENT places, which is the whole subtlety: field
    presence and `candidates_pre_filter` from the venue's POOL rows, `rows_post_filter` from the
    RENDERED ones.
    """

    @pytest.fixture(autouse=True)
    def _reset_health(self):
        reset_provider_health()
        yield
        reset_provider_health()

    @pytest.mark.asyncio
    async def test_field_presence_is_measured_over_the_pool_not_the_ranked_rows(
        self, mock_question, kalshi_events_payload, polymarket_payload
    ):
        """The load-bearing choice, and the reason Signal A does not fire on the ranker's
        judgment.

        The ranker keeps only Kalshi here, so Polymarket renders zero rows. Measured over the
        RENDERED rows, Polymarket's declared fields would read 100% dead and redden CI on a
        legitimate 1-row ranking. Signal A exists to catch a PARSER whose field names went
        stale, so it must be immune to selection — which is what reading the pool gives it.
        """
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
                _POLY_URL: FakeResponse(200, polymarket_payload),
            }
        )

        def _kalshi_only(prompt: str) -> str:
            index = int(_CANDIDATE_LINE_RE.search(prompt).group(1))  # type: ignore[union-attr]
            return json.dumps([{"i": index, "tier": "same_quantity_same_date", "why": "same event"}])

        snapshot = await _fetch(mock_question, handlers, ranking=_kalshi_only)

        assert {row.platform for row in snapshot.matches} == {"kalshi"}
        observed = {obs.venue: obs for obs in recorded_observations()[0]}
        assert observed["polymarket"].rows_post_filter == 0
        assert observed["polymarket"].candidates_pre_filter > 0
        assert set(observed["polymarket"].liquidity_fields_present) == set(
            VENUE_EXPECTED_LIQUIDITY_FIELDS["polymarket"]
        )
        assert provider_degradation_findings() == []

    @pytest.mark.asyncio
    async def test_rendered_counts_come_from_the_ranked_rows(self, mock_question, kalshi_events_payload):
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})

        snapshot = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        rendered = [row for row in snapshot.matches if row.platform == "kalshi"]
        observed = next(obs for obs in recorded_observations()[0] if obs.venue == "kalshi")
        assert observed.rows_post_filter == len(rendered)
        # Asserted here and in `test_recording_does_not_alter_the_snapshot` so a future signal
        # cannot start firing on this healthy shape unnoticed: both tests previously left two
        # live alertable findings and passed anyway, which is how the deleted Signal B's false
        # positive reached review with the suite demonstrating it.
        assert provider_degradation_findings() == []

    @pytest.mark.asyncio
    async def test_a_market_less_question_records_zero_rows_and_finds_nothing(self, mock_question):
        """The false-positive case that matters most, through the real fetch path: every venue
        answered, the ranker kept nothing, and NOTHING is alertable.

        Note the asymmetry, which is a fact about ranked retrieval rather than a quirk of this
        fixture. The two SEARCH venues sit at zero candidates, because their index had no hit, so
        Signal A never evaluates them. The two ENUMERABLE venues cannot be at zero: their whole
        catalogue enters the pool, so a healthy catalogue always yields candidates and their
        parsers are always checked — while `record_catalogue_size` (Signal C) is the only thing
        that can fire on an EMPTY catalogue for them.
        """
        await _fetch(mock_question, _handlers())

        observed = {obs.venue: obs for obs in recorded_observations()[0]}
        assert set(observed) == {"polymarket", "manifold", "kalshi", "predictit"}
        assert all(obs.rows_post_filter == 0 for obs in observed.values())
        assert observed["polymarket"].candidates_pre_filter == 0
        assert observed["manifold"].candidates_pre_filter == 0
        assert observed["kalshi"].candidates_pre_filter > 0
        assert observed["predictit"].candidates_pre_filter > 0
        assert provider_degradation_findings() == []

    @pytest.mark.asyncio
    async def test_a_lost_venue_records_no_observation(self, mock_question, monkeypatch):
        """A venue whose fan-out lost a sub-fetch is already alertable via
        `prediction_market_source_losses`, so it is not observed here — counting the same outage
        twice would report one failure as two, and "check the query construction" is the wrong
        remedy for a 503."""
        _no_backoff(monkeypatch)
        handlers = _handlers(**{_MANIFOLD_SEARCH_URL: FakeResponse(503, text="service unavailable")})

        await _fetch(mock_question, handlers)

        assert "manifold" not in {obs.venue for obs in recorded_observations()[0]}
        assert pmp.prediction_market_source_losses() >= 1
        assert provider_degradation_findings() == []

    @pytest.mark.asyncio
    async def test_both_catalogue_sizes_are_recorded_with_their_fetch_outcome(
        self, mock_question, kalshi_events_payload, predictit_payload
    ):
        """Signal C's input, and the only health signal the enumerable venues have left."""
        handlers = _handlers(
            **{
                _KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload),
                _PREDICTIT_URL: FakeResponse(200, predictit_payload),
            }
        )

        await _fetch(mock_question, handlers)

        catalogues = {obs.source: obs for obs in recorded_observations()[1]}
        assert catalogues["kalshi_events"].entries == len(kalshi_events_payload["events"])
        assert catalogues["kalshi_events"].fetch_ok is True
        assert catalogues["predictit_markets"].entries == len(predictit_payload["markets"])
        assert catalogues["predictit_markets"].fetch_ok is True

    @pytest.mark.asyncio
    async def test_a_failed_pull_records_the_failure_not_a_phantom_empty_catalogue(self, mock_question, monkeypatch):
        """A 503'd catalogue must record `fetch_ok=False`, so Signal C stays silent and the
        source-loss counter remains the sole reporter."""
        _no_backoff(monkeypatch)
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(503, text="service unavailable")})

        await _fetch(mock_question, handlers)

        catalogues = {obs.source: obs for obs in recorded_observations()[1]}
        assert catalogues["kalshi_events"].fetch_ok is False
        assert provider_degradation_findings() == []

    @pytest.mark.asyncio
    async def test_a_question_with_no_id_records_nothing(self, mock_question):
        """Every observation is keyed on the question, so an id-less question records nothing —
        matching `record_provider_detail` / `record_raw_research`."""
        mock_question.id_of_question = None

        await _fetch(mock_question, _handlers())

        assert recorded_observations() == ((), ())

    @pytest.mark.asyncio
    async def test_recording_does_not_alter_the_snapshot(self, mock_question, kalshi_events_payload):
        """Recording is a pure module-state write on the research path: same rows, same tokens,
        with or without observations already present."""
        handlers = _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)})
        first = await _fetch(mock_question, handlers, ranking=_rank_one_per_venue)

        _reset_session_caches()
        reset_provider_health()
        second = await _fetch(
            mock_question,
            _handlers(**{_KALSHI_EVENTS_URL: FakeResponse(200, kalshi_events_payload)}),
            ranking=_rank_one_per_venue,
        )

        assert [(row.platform, row.market_title) for row in first.matches] == [
            (row.platform, row.market_title) for row in second.matches
        ]
        assert first.sources == second.sources
        assert provider_degradation_findings() == [], "recording a healthy run must leave nothing alertable"
