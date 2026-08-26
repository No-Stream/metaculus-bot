# noqa: HARNESS-SCAN-EXEMPT-monolithic-file-loc  # one class per MARKER_SPECS entry; a split fragments the registry
"""Tests for the run-log telemetry marker parser (scripts/telemetry/markers.py).

Each example line below is copied from the format string in the emitting code
(the source of truth), so a producer-side change to a marker shape breaks these
tests loudly instead of silently dropping records from the archive:

* EXTRACTION_RUNG   -> metaculus_bot/value_extraction.py:_log_extraction
* GAP_FILL_V2       -> metaculus_bot/research/agentic/loop.py:_log_completion
* GHOST_PRE[_JSON]  -> metaculus_bot/research/agentic/loop.py:_set_research_plan_tool
* GHOST_FORECAST    -> metaculus_bot/research/agentic/loop.py:_run_ghost_phase
* OPEN_BOUND_PILING -> metaculus_bot/numeric/diagnostics.py:log_open_bound_piling_diagnostics
* CLOSE_MARGIN       -> metaculus_bot/close_margin.py:format_close_margin_marker
* MARKET_RANKING    -> metaculus_bot/research/prediction_market.py:_log_ranking_telemetry
* CREDIT_BALANCE/SPEND/FLOOR_BREACH -> metaculus_bot/credit_telemetry.py
* STACKER_OUTCOME/TOOLS_USED/ANCHOR_OVERSHOOT_PP/CLAUSE_PRODUCT_DIVERGENCE_PP
  -> metaculus_bot/comment/markers.py (HTML-comment markers; see module docstring
     in markers.py for why they rarely appear in run logs).
"""

import json

import pytest

from scripts.telemetry.markers import (
    MARKER_SPECS,
    coerce_value,
    parse_log_text,
    qid_from_ref,
)

# Prod cli.py log format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
PFX = "2026-07-17 14:23:01,123 - metaculus_bot.x - INFO - "
PFX_WARN = "2026-07-17 14:30:00,456 - metaculus_bot.x - WARNING - "

EXTRACTION_RUNG_LINE = (
    PFX + "EXTRACTION_RUNG: question=12345 model=openai/gpt-5.6-sol qtype=binary rung=block block_present=True"
)
GAP_FILL_V2_LINE = (
    "2026-07-21 14:25:10,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GAP_FILL_V2: model=openai/gpt-5.6-terra "
    "steps=7 tool_calls=9 searches=4 fetches=3 rendered=1 reads=2 dup_tool_calls=0 deadline_hit=False "
    "concluded_early=True wall_s=312.44 findings=5 pending_leads=1 lint_rejections=0 "
    "provenance_rejections=1 quote_mismatch_warnings=2 plan_gaps=3 plan_skipped=False "
    "conclude_gate_rejections=1 error=None"
)
# A crashed v2 run: byte-identical counters to a legitimate idle run EXCEPT the
# error= field carries repr(exc). This is the whole point of the field — the
# fastapi eager-import defect emitted steps=0 tool_calls=0 findings=0 (identical
# to "driver found nothing") and only error= makes the crash greppable.
GAP_FILL_V2_CRASHED_LINE = (
    "2026-07-23 14:25:10,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GAP_FILL_V2: model=openai/gpt-5.6-terra "
    "steps=0 tool_calls=0 searches=0 fetches=0 rendered=0 reads=0 dup_tool_calls=0 deadline_hit=False "
    "concluded_early=False wall_s=0.12 findings=0 pending_leads=0 lint_rejections=0 "
    "provenance_rejections=0 quote_mismatch_warnings=0 plan_gaps=0 plan_skipped=False "
    "conclude_gate_rejections=0 error=APIConnectionError(\"No module named 'fastapi'\")"
)
# Pre-2026-07-21 completion format (ends at lint_rejections). Replace-by-run
# re-harvesting replays old logs, so the parser must keep accepting this shape.
GAP_FILL_V2_LEGACY_LINE = (
    "2026-07-17 14:25:10,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GAP_FILL_V2: model=openai/gpt-5.6-terra "
    "steps=7 tool_calls=9 searches=4 fetches=3 rendered=1 reads=2 dup_tool_calls=0 deadline_hit=False "
    "concluded_early=True wall_s=312.44 findings=5 pending_leads=1 lint_rejections=0"
)
GHOST_PRE_LINE = (
    "2026-07-21 14:25:09,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GHOST_PRE: gaps=3 sensitive_assumptions=2"
)
GHOST_PRE_JSON_LINE = (
    "2026-07-21 14:25:09,001 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GHOST_PRE_JSON: "
    '{"qtype":"binary","prob":0.35}'
)
GHOST_FORECAST_LINE = (
    "2026-07-17 14:25:11,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GHOST_FORECAST: qtype=binary summary=posterior_prob=0.4200"
)
GHOST_FORECAST_JSON_LINE = (
    "2026-07-17 14:25:11,001 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GHOST_FORECAST_JSON: "
    '{"qtype":"binary","prob":0.42}'
)
GHOST_FORECAST_JSON_NUMERIC_LINE = (
    "2026-07-17 14:25:11,002 - metaculus_bot.research.agentic.loop - INFO - "
    "question=12 GHOST_FORECAST_JSON: "
    '{"qtype":"numeric","declared_percentiles":{"0.1":10.0,"0.5":20.5,"0.9":30.0},"median":20.5}'
)
OPEN_BOUND_PILING_LINE = (
    PFX_WARN + "OPEN_BOUND_PILING: question=51000 model=gemini-3.1-pro-preview bound=upper "
    "bin_mass=0.153 declared_edge=1000 bound_value=1000"
)
# Copied from metaculus_bot/close_margin.py:format_close_margin_marker output.
CLOSE_MARGIN_LINE = (
    PFX + "CLOSE_MARGIN: question=44620 close_time=2026-07-20T00:00:00+00:00 "
    "submitted_at=2026-07-19T13:50:00+00:00 window_s=864000 margin_s=36600 margin_frac=0.0424"
)
CLOSE_MARGIN_NA_LINE = (
    PFX + "CLOSE_MARGIN: question=44620 close_time=2026-07-20T00:00:00+00:00 "
    "submitted_at=2026-07-19T00:00:00+00:00 window_s=n/a margin_s=86400 margin_frac=n/a"
)
# Captured by calling metaculus_bot/research/prediction_market.py:_log_ranking_telemetry
# under the prod log format, so these are emitted bytes rather than a transcription. The
# four shapes are the four the emitter can produce: a normal ranked slate, an empty pool,
# a fail-open, and a row whose pool index could not be recovered.
MARKET_RANKING_RANKED_LINE = (
    PFX + "MARKET_RANKING: question=44620 pool=4 outcome=ranked rows=2 prompt_chars=36412 "
    "rendered=polymarket:2@0,kalshi:0@1"
)
MARKET_RANKING_EMPTY_LINE = (
    PFX + "MARKET_RANKING: question=44620 pool=0 outcome=empty rows=0 prompt_chars=0 rendered=none"
)
MARKET_CHILD_RENDER_LINE = (
    PFX + "MARKET_CHILD_RENDER: question=45363 families=6 full_rows=14 ladder_rows=6 outcomes=158 "
    "named=71 collapsed=87 withheld=3 max_stage=5 ladder_chars=1368"
)
MARKET_RANKING_FAILOPEN_LINE = (
    PFX + "MARKET_RANKING: question=None pool=4 outcome=failopen rows=2 prompt_chars=36412 "
    "rendered=polymarket:2@0,kalshi:0@1"
)
MARKET_RANKING_UNTRACEABLE_INDEX_LINE = (
    PFX + "MARKET_RANKING: question=44620 pool=4 outcome=ranked rows=1 prompt_chars=36412 rendered=manifold:-1@0"
)
TS_ANCHOR_ROUTE_ROUTED_LINE = PFX + "TS_ANCHOR_ROUTE: question=45401 decision=routed series=PAYEMS step=kw_single"
TS_ANCHOR_ROUTE_GATE_SKIP_LINE = (
    PFX + "TS_ANCHOR_ROUTE: question=45367 decision=skipped series=PAYEMS step=kw_derivation_gate"
)
TS_ANCHOR_ROUTE_NO_HIT_LINE = (
    PFX + "TS_ANCHOR_ROUTE: question=45193 decision=skipped series=none step=kw_no_keyword_hit"
)
TS_ANCHOR_ROUTE_SPREAD_LINE = PFX + "TS_ANCHOR_ROUTE: question=44700 decision=routed series=CL=F/^GSPC step=url_spread"
# Copied from the two emitters sharing the shape: financial_data.py:_fetch_yfinance_data
# and ts_render.py:_render_single (same stale_latest_age_days estimator behind both).
FINANCIAL_STALE_LATEST_YFINANCE_LINE = (
    PFX_WARN + "FINANCIAL_STALE_LATEST: surface=financial_data symbol=TEST age_d=3 cadence=calendar-day"
)
FINANCIAL_STALE_LATEST_TS_ANCHOR_LINE = (
    PFX_WARN + "FINANCIAL_STALE_LATEST: surface=ts_anchor symbol=^DEAD age_d=9 cadence=trading-day"
)
CREDIT_BALANCE_LINE = PFX + "CREDIT_BALANCE: key=donated phase=start remaining=123.45 usage=4.16"
CREDIT_BALANCE_SKIP_LINE = (
    PFX_WARN + "CREDIT_BALANCE: key=personal phase=start skipped (env var OPENROUTER_API_KEY not set)"
)
# Pre-2026-07-27 shape: no source= field. Kept verbatim because re-harvesting
# replays these older logs, and they must still parse.
CREDIT_SPEND_LINE = PFX + "CREDIT_SPEND: key=donated run_delta_usd=3.34 remaining=120.11"
CREDIT_SPEND_NA_LINE = PFX + "CREDIT_SPEND: key=personal run_delta_usd=n/a remaining=n/a"
# Current shape, verbatim from credit_telemetry.log_end_and_check_floor.
CREDIT_SPEND_REMAINING_SOURCE_LINE = (
    PFX + "CREDIT_SPEND: key=donated run_delta_usd=3.34 remaining=120.11 source=remaining_delta"
)
CREDIT_SPEND_UNSETTLED_SOURCE_LINE = (
    PFX + "CREDIT_SPEND: key=personal run_delta_usd=0.00 remaining=n/a source=usage_delta_unsettled"
)
CREDIT_FLOOR_BREACH_LINE = (
    PFX_WARN + "CREDIT_FLOOR_BREACH: key=donated remaining=45.00 floor=50.00 — donated OpenRouter "
    "balance needs a top-up; run completed normally. cli.main logs the resulting "
    "exit decision (non-zero unless credit alerting is currently suppressed)."
)

_META = {
    "run_id": "999",
    "workflow": "tournament",
    "artifact": "research-999",
    "run_date": "2026-07-17T14:00:00Z",
    "log_file": "run.log",
}


def _parse_one(line: str) -> dict:
    """Parse a single line, assert exactly one record came out, return it."""
    harvested = parse_log_text(line + "\n", **_META)
    records = [r for recs in harvested.values() for r in recs]
    assert len(records) == 1, f"expected 1 record, got {records}"
    return records[0]


class TestCoerceValue:
    def test_bools(self):
        assert coerce_value("True") is True
        assert coerce_value("False") is False

    def test_none_sentinels(self):
        assert coerce_value("None") is None
        assert coerce_value("n/a") is None

    def test_ints_and_floats(self):
        assert coerce_value("7") == 7
        assert isinstance(coerce_value("7"), int)
        assert coerce_value("312.44") == 312.44
        assert coerce_value("-4.0") == -4.0

    def test_strings_stay_strings(self):
        assert coerce_value("upper") == "upper"
        assert coerce_value("openai/gpt-5.6-sol") == "openai/gpt-5.6-sol"
        assert coerce_value("binary") == "binary"


class TestQidFromRef:
    def test_url(self):
        assert qid_from_ref("https://www.metaculus.com/questions/38975/") == 38975

    def test_bare_int(self):
        assert qid_from_ref("12345") == 12345

    def test_none_sentinel(self):
        assert qid_from_ref("None") is None
        assert qid_from_ref(None) is None


class TestExtractionRung:
    def test_fields(self):
        rec = _parse_one(EXTRACTION_RUNG_LINE)
        assert rec["marker"] == "extraction_rung"
        assert rec["question"] == "12345"
        assert rec["qid"] == 12345
        assert rec["model"] == "openai/gpt-5.6-sol"
        assert rec["qtype"] == "binary"
        assert rec["rung"] == "block"
        assert rec["block_present"] is True

    def test_qid_kind_is_question_id(self):
        # EXTRACTION_RUNG logs question.id_of_question, so its records live in the
        # QUESTION-id space — the tag a residual join uses to translate correctly.
        assert _parse_one(EXTRACTION_RUNG_LINE)["qid_kind"] == "question_id"

    def test_line_timestamp_parsed(self):
        rec = _parse_one(EXTRACTION_RUNG_LINE)
        assert rec["line_ts"].startswith("2026-07-17T14:23:01")

    def test_run_metadata_attached(self):
        rec = _parse_one(EXTRACTION_RUNG_LINE)
        assert rec["run_id"] == "999"
        assert rec["workflow"] == "tournament"
        assert rec["artifact"] == "research-999"

    def test_verbatim_real_prod_line(self):
        # Grounding: this line is copied byte-for-byte from a real prod tournament run
        # log (run 29633926137, 2026-07-18) — not reconstructed. Guards against the
        # regexes drifting from the actual emitted format.
        real = (
            "2026-07-18 06:30:01,112 - metaculus_bot.value_extraction - INFO - "
            "EXTRACTION_RUNG: question=44620 model=openrouter/x-ai/grok-4.5 qtype=binary rung=block block_present=True"
        )
        rec = _parse_one(real)
        assert rec["qid"] == 44620
        assert rec["model"] == "openrouter/x-ai/grok-4.5"
        assert rec["qtype"] == "binary"
        assert rec["rung"] == "block"
        assert rec["block_present"] is True

    def test_llm_salvage_rung(self):
        line = PFX + "EXTRACTION_RUNG: question=None model=grok-4.5 qtype=numeric rung=llm block_present=False"
        rec = _parse_one(line)
        assert rec["rung"] == "llm"
        assert rec["qid"] is None
        assert rec["block_present"] is False


class TestGapFillV2:
    def test_fields(self):
        rec = _parse_one(GAP_FILL_V2_LINE)
        assert rec["marker"] == "gap_fill_v2"
        assert rec["qid"] == 38975
        # GAP_FILL_V2's question= comes from question.page_url -> a POST id.
        assert rec["qid_kind"] == "post_id"
        assert rec["model"] == "openai/gpt-5.6-terra"
        assert rec["steps"] == 7
        assert rec["tool_calls"] == 9
        assert rec["searches"] == 4
        assert rec["fetches"] == 3
        assert rec["rendered"] == 1
        assert rec["reads"] == 2
        assert rec["dup_tool_calls"] == 0
        assert rec["deadline_hit"] is False
        assert rec["concluded_early"] is True
        assert rec["wall_s"] == 312.44
        assert rec["findings"] == 5
        assert rec["pending_leads"] == 1
        assert rec["lint_rejections"] == 0
        # The five loop counters added 2026-07-21 (see loop.py:_log_completion).
        assert rec["provenance_rejections"] == 1
        assert rec["quote_mismatch_warnings"] == 2
        assert rec["plan_gaps"] == 3
        assert rec["plan_skipped"] is False
        assert rec["conclude_gate_rejections"] == 1
        # error= (added 2026-07-23) coerces "None" to Python None on a healthy run.
        assert rec["error"] is None

    def test_crashed_run_carries_error_repr(self):
        # The distinguishing signal: a v2 crash and a legitimate idle run emit
        # byte-identical counters; only error= tells them apart (the fastapi
        # eager-import defect was silently dead precisely because this was missing).
        rec = _parse_one(GAP_FILL_V2_CRASHED_LINE)
        assert rec["marker"] == "gap_fill_v2"
        assert rec["steps"] == 0
        assert rec["tool_calls"] == 0
        assert rec["findings"] == 0
        # repr(exc) is preserved verbatim (spaces and all) — it is not coerced away.
        assert rec["error"] == "APIConnectionError(\"No module named 'fastapi'\")"

    def test_legacy_line_without_new_counters_still_harvests(self):
        # Old-format lines (pre-2026-07-21) must keep parsing on re-harvest; the
        # five new counter fields plus error come through as None, not a dropped record.
        rec = _parse_one(GAP_FILL_V2_LEGACY_LINE)
        assert rec["marker"] == "gap_fill_v2"
        assert rec["qid"] == 38975
        assert rec["lint_rejections"] == 0
        assert rec["provenance_rejections"] is None
        assert rec["quote_mismatch_warnings"] is None
        assert rec["plan_gaps"] is None
        assert rec["plan_skipped"] is None
        assert rec["conclude_gate_rejections"] is None
        assert rec["error"] is None


class TestGhostPre:
    def test_fields(self):
        rec = _parse_one(GHOST_PRE_LINE)
        assert rec["marker"] == "ghost_pre"
        # question= comes from log_prefix (question.page_url) -> a POST id.
        assert rec["qid"] == 38975
        assert rec["qid_kind"] == "post_id"
        assert rec["gaps"] == 3
        assert rec["sensitive_assumptions"] == 2

    def test_json_payload_round_trips(self):
        rec = _parse_one(GHOST_PRE_JSON_LINE)
        assert rec["marker"] == "ghost_pre_json"
        assert rec["qid"] == 38975
        assert rec["qid_kind"] == "post_id"
        # forecast_json stays a raw string (never coerced) so the scorer can json.loads it.
        assert json.loads(rec["forecast_json"]) == {"qtype": "binary", "prob": 0.35}

    def test_does_not_collide_with_ghost_forecast_pair(self):
        # GHOST_PRE: / GHOST_PRE_JSON: / GHOST_FORECAST: / GHOST_FORECAST_JSON: are
        # four distinct tokens — each line must harvest as exactly its own marker
        # under the one-marker-per-line break.
        assert _parse_one(GHOST_PRE_LINE)["marker"] == "ghost_pre"
        assert _parse_one(GHOST_PRE_JSON_LINE)["marker"] == "ghost_pre_json"
        assert _parse_one(GHOST_FORECAST_LINE)["marker"] == "ghost_forecast"
        assert _parse_one(GHOST_FORECAST_JSON_LINE)["marker"] == "ghost_forecast_json"


class TestGhostForecast:
    def test_binary(self):
        rec = _parse_one(GHOST_FORECAST_LINE)
        assert rec["marker"] == "ghost_forecast"
        assert rec["qid"] == 38975
        assert rec["qtype"] == "binary"
        assert rec["summary"] == "posterior_prob=0.4200"

    def test_multiple_choice_summary(self):
        line = (
            "2026-07-17 14:25:11,000 - metaculus_bot.research.agentic.loop - INFO - "
            "question=12 GHOST_FORECAST: qtype=multiple_choice summary=Blue=0.300, Red=0.700"
        )
        rec = _parse_one(line)
        assert rec["qtype"] == "multiple_choice"
        assert rec["summary"] == "Blue=0.300, Red=0.700"

    def test_numeric_median_only(self):
        line = (
            "2026-07-17 14:25:11,000 - metaculus_bot.research.agentic.loop - INFO - "
            "question=12 GHOST_FORECAST: qtype=numeric summary=median=42.5"
        )
        rec = _parse_one(line)
        assert rec["qtype"] == "numeric"
        assert rec["summary"] == "median=42.5"


class TestGhostForecastJson:
    def test_binary_payload_round_trips(self):
        rec = _parse_one(GHOST_FORECAST_JSON_LINE)
        assert rec["marker"] == "ghost_forecast_json"
        # qid is carried via the log_prefix leading group, exactly like GHOST_FORECAST.
        assert rec["qid"] == 38975
        # forecast_json stays a raw string (never coerced) so the scorer can json.loads it.
        assert json.loads(rec["forecast_json"]) == {"qtype": "binary", "prob": 0.42}

    def test_numeric_payload_carries_full_percentiles(self):
        rec = _parse_one(GHOST_FORECAST_JSON_NUMERIC_LINE)
        assert rec["marker"] == "ghost_forecast_json"
        assert rec["qid"] == 12
        payload = json.loads(rec["forecast_json"])
        assert payload == {
            "qtype": "numeric",
            "declared_percentiles": {"0.1": 10.0, "0.5": 20.5, "0.9": 30.0},
            "median": 20.5,
        }

    def test_does_not_collide_with_legacy_ghost_forecast(self):
        # A legacy GHOST_FORECAST line must NOT be mis-harvested as ghost_forecast_json,
        # and a GHOST_FORECAST_JSON line must NOT match the legacy spec — the two tokens
        # are mutually exclusive under the one-marker-per-line break.
        assert _parse_one(GHOST_FORECAST_LINE)["marker"] == "ghost_forecast"
        assert _parse_one(GHOST_FORECAST_JSON_LINE)["marker"] == "ghost_forecast_json"


class TestOpenBoundPiling:
    def test_fields(self):
        rec = _parse_one(OPEN_BOUND_PILING_LINE)
        assert rec["marker"] == "open_bound_piling"
        assert rec["qid"] == 51000
        assert rec["model"] == "gemini-3.1-pro-preview"
        assert rec["bound"] == "upper"
        assert rec["bin_mass"] == 0.153
        assert rec["declared_edge"] == 1000
        assert rec["bound_value"] == 1000


class TestCloseMargin:
    def test_full_fields(self):
        rec = _parse_one(CLOSE_MARGIN_LINE)
        assert rec["marker"] == "close_margin"
        assert rec["question"] == "44620"
        assert rec["qid"] == 44620
        # ISO timestamps stay strings (float() fails, so coerce_value leaves them be).
        assert rec["close_time"] == "2026-07-20T00:00:00+00:00"
        assert rec["submitted_at"] == "2026-07-19T13:50:00+00:00"
        assert rec["window_s"] == 864000
        assert isinstance(rec["window_s"], int)
        assert rec["margin_s"] == 36600
        assert rec["margin_frac"] == 0.0424

    def test_na_window_and_frac(self):
        rec = _parse_one(CLOSE_MARGIN_NA_LINE)
        assert rec["window_s"] is None
        assert rec["margin_frac"] is None
        assert rec["margin_s"] == 86400


class TestMarketRanking:
    """The ranked-retrieval port's own post-ship instrument, so it has to reach the archive.

    `rendered`'s pool indices are what answer the two questions the port left open (ranker
    attention decay down a ~400-candidate prompt, and whether Manifold detail enrichment
    changes the picks); `prompt_chars` is the free prod distribution against the prompt
    ceiling. Run logs leave GHA at 90 days, so anything unharvested is unanswerable later.
    """

    def test_ranked_run_fields(self):
        rec = _parse_one(MARKET_RANKING_RANKED_LINE)
        assert rec["marker"] == "market_ranking"
        assert rec["pool"] == 4
        assert rec["outcome"] == "ranked"
        assert rec["rows"] == 2
        assert rec["prompt_chars"] == 36412
        # The venue:pool_index@rank list survives whole — the indices are the instrument.
        assert rec["rendered"] == "polymarket:2@0,kalshi:0@1"

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(MARKET_RANKING_RANKED_LINE)
        # prediction_market.py emits question.id_of_question, not the post id.
        assert rec["qid"] == 44620
        assert rec["qid_kind"] == "question_id"

    def test_empty_pool_run_is_distinguishable_from_a_declining_ranker(self):
        # `empty` means the pool had nothing to rank (a retrieval story), while a ranker
        # that legitimately returned zero rows reads `outcome=ranked rows=0` — the two
        # must not collapse, since only the first implicates the venues.
        rec = _parse_one(MARKET_RANKING_EMPTY_LINE)
        assert rec["outcome"] == "empty"
        assert rec["pool"] == 0
        assert rec["rows"] == 0
        assert rec["prompt_chars"] == 0
        # "none" is a _NONE_SENTINELS value, so an unrendered slate reads as None.
        assert rec["rendered"] is None

    def test_failopen_run_and_absent_qid(self):
        # A fail-open renders the head of the ranker's own input, so its rows are still
        # worth measuring; qid is Optional at the call site and renders as "None".
        rec = _parse_one(MARKET_RANKING_FAILOPEN_LINE)
        assert rec["outcome"] == "failopen"
        assert rec["rows"] == 2
        assert rec["qid"] is None

    def test_untraceable_pool_index_sentinel_survives(self):
        # -1 means the rendered row could not be matched back to a pool entry, which is a
        # defect in the index recovery rather than a real position — it must stay visible
        # in the archive rather than coercing into the index distribution as a 0.
        rec = _parse_one(MARKET_RANKING_UNTRACEABLE_INDEX_LINE)
        assert rec["rendered"] == "manifold:-1@0"


class TestMarketChildRender:
    """The multi-outcome render's own instrument, and the reason it is a SEPARATE marker.

    `market_ranking`'s regex is not end-anchored, so appending fields to that line would have
    re-cut a spec other work touches; a new marker keeps the harvester change purely additive.

    `withheld` is what the line exists for. The Kalshi no-price spread threshold that blanks an
    empty book is calibrated on eleven fixture strikes, so its prod incidence has to be a query
    rather than a guess, and the same field counts the Polymarket placeholder legs and Manifold
    untouched priors. Run logs leave GHA at 90 days, so an unharvested line is unanswerable later.
    """

    def test_the_fields_survive_harvesting(self):
        rec = _parse_one(MARKET_CHILD_RENDER_LINE)
        assert rec["marker"] == "market_child_render"
        assert rec["families"] == 6
        assert rec["full_rows"] == 14
        assert rec["ladder_rows"] == 6
        assert rec["outcomes"] == 158
        assert rec["withheld"] == 3
        assert rec["max_stage"] == 5
        assert rec["ladder_chars"] == 1368

    def test_the_completeness_invariant_is_checkable_from_the_archive(self):
        """`named + collapsed == outcomes` is what the render guarantees, so a harvested line where
        they disagree is a render bug rather than a tuning signal — which only works if both halves
        reach the archive as numbers."""
        rec = _parse_one(MARKET_CHILD_RENDER_LINE)

        assert rec["named"] + rec["collapsed"] == rec["outcomes"]

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(MARKET_CHILD_RENDER_LINE)
        # prediction_market.py emits question.id_of_question, matching its MARKET_RANKING sibling.
        assert rec["qid"] == 45363
        assert rec["qid_kind"] == "question_id"


# The ranker fail-open WARN (research/prediction_market.py:_rank_pool), verbatim from the
# f-string there. `reason=shape_regression` is the one the finding is about: before
# 2026-08-25 that case reported `ok(0)`, i.e. it rendered the deliberate "we reviewed the
# markets and none bore on the question" sentence on a path where the ranker's answer was
# unreadable. `detail=` holds the exception's str, which carries spaces and a repr.
MARKET_RANKING_DEGRADED_SHAPE_LINE = (
    PFX_WARN + "MARKET_RANKING_DEGRADED: question=44620 pool=17 reason=shape_regression "
    "detail=falling back to retrieval order; 3 entries yielded no usable pick (renamed index key, "
    "or every index outside a pool of 17); first={'index': 4, 'tier': 'weak'}"
)
MARKET_RANKING_DEGRADED_UNREADABLE_LINE = (
    PFX_WARN + "MARKET_RANKING_DEGRADED: question=None pool=4 reason=unreadable "
    "detail=falling back to retrieval order; empty completion"
)


class TestMarketRankingDegraded:
    """The sibling that says WHY a question rendered retrieval order.

    `market_ranking`'s `outcome=failopen` records that a fail-open happened; only this line
    separates "the model emitted something that is not a ranking array" from "our own
    prompt/parser contract broke", and the second is the regression that used to arrive as
    `ok(0)`. Run logs leave GHA at 90 days, so an archive holding one line and not the other
    cannot answer which failure a degraded question hit.
    """

    def test_shape_regression_fields(self):
        rec = _parse_one(MARKET_RANKING_DEGRADED_SHAPE_LINE)
        assert rec["marker"] == "market_ranking_degraded"
        assert rec["pool"] == 17
        assert rec["reason"] == "shape_regression"

    def test_detail_free_text_survives_verbatim(self):
        # detail holds repr(exc): spaces, parentheses, quotes, a dict repr. It belongs to
        # _RAW_FIELDS so it is never coerced, and the regex is not end-anchored so a
        # terser future form still harvests.
        rec = _parse_one(MARKET_RANKING_DEGRADED_SHAPE_LINE)
        assert "renamed index key" in rec["detail"]
        assert rec["detail"].endswith("first={'index': 4, 'tier': 'weak'}")

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(MARKET_RANKING_DEGRADED_SHAPE_LINE)
        # prediction_market.py emits question.id_of_question, matching its two siblings.
        assert rec["qid"] == 44620
        assert rec["qid_kind"] == "question_id"

    def test_unreadable_reason_and_absent_qid(self):
        rec = _parse_one(MARKET_RANKING_DEGRADED_UNREADABLE_LINE)
        assert rec["reason"] == "unreadable"
        assert rec["qid"] is None

    def test_does_not_collide_with_the_market_ranking_spec(self):
        # MARKET_RANKING_DEGRADED contains MARKET_RANKING as a prefix, and market_ranking's
        # spec sits EARLIER in MARKER_SPECS — under the one-marker-per-line break, a
        # colon-less prefix match there would have swallowed every degraded line.
        harvested = parse_log_text(
            MARKET_RANKING_DEGRADED_SHAPE_LINE + "\n" + MARKET_RANKING_RANKED_LINE + "\n", **_META
        )
        assert len(harvested["market_ranking_degraded"]) == 1
        assert len(harvested["market_ranking"]) == 1


# Verbatim emitted bytes from metaculus_bot/numeric/pipeline.py (captured under the prod log
# format), metaculus_bot/numeric/utils.py, and metaculus_bot/spread_metrics.py. All three
# lines carry trailing em-dash prose, so none of the specs may be end-anchored.
NUMERIC_DEGENERATE_DECLARATION_LINE = (
    "2026-08-25 23:07:17,044 - metaculus_bot.numeric.pipeline - WARNING - "
    "NUMERIC_DEGENERATE_DECLARATION: question=77 model=openrouter/openai/gpt-5.6-sol n_unique=1 "
    "span=0 value_eps=1e-07 spread_applied=false"
)
NUMERIC_DEGENERATE_DECLARATION_UNLABELLED_LINE = (
    "2026-08-25 23:07:17,044 - metaculus_bot.numeric.pipeline - WARNING - "
    "NUMERIC_DEGENERATE_DECLARATION: question=77 model=unknown n_unique=1 "
    "span=1.5e-06 value_eps=1e-07 spread_applied=false"
)
NUMERIC_AGGREGATE_GRID_MISMATCH_LINE = (
    "2026-08-25 23:07:17,044 - metaculus_bot.numeric.utils - WARNING - "
    "NUMERIC_AGGREGATE_GRID_MISMATCH: question=44620 model_index=2 got_points=201 expected_points=11 "
    "— resampling in cdf-location space before aggregation"
)
SPREAD_UNDEFINED_LINE = (
    "2026-08-25 23:07:17,044 - metaculus_bot.spread_metrics - WARNING - "
    "SPREAD_UNDEFINED: question=45363 qtype=numeric denominator=-0 models=3 — key-percentile spread "
    "is unmeasurable (non-positive denominator); reporting inf so it cannot read as agreement"
)


class TestNumericDegenerateDeclaration:
    """A per-forecaster fabrication-ATTEMPT rate, which is why it needs a spec.

    A point-mass declaration is no longer cluster-spread, so the unit-mismatch guard sees the
    model's own zero span and withholds the forecaster. The drop itself lands in
    FORECASTER_DROPS as an UnitMismatchError; only this line names the cause, and its
    predecessor (`Cluster spread applied`) was never harvested — which is exactly why the
    finding's prod incidence was unanswerable from the archive.
    """

    def test_fields(self):
        rec = _parse_one(NUMERIC_DEGENERATE_DECLARATION_LINE)
        assert rec["marker"] == "numeric_degenerate_declaration"
        assert rec["model"] == "openrouter/openai/gpt-5.6-sol"
        assert rec["n_unique"] == 1
        assert rec["span"] == 0
        assert rec["value_eps"] == pytest.approx(1e-07)
        assert rec["spread_applied"] is False

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(NUMERIC_DEGENERATE_DECLARATION_LINE)
        assert rec["qid"] == 77
        assert rec["qid_kind"] == "question_id"

    def test_unlabelled_model_stays_a_readable_string(self):
        # "unknown" is what the line carries when a caller doesn't pass model_name. All
        # three production callers now do (8cccdaa), so it survives for historical lines
        # and any future caller that forgets. It is NOT in _NONE_SENTINELS, so it must
        # survive as a string — a None here would be indistinguishable from a missing field.
        rec = _parse_one(NUMERIC_DEGENERATE_DECLARATION_UNLABELLED_LINE)
        assert rec["model"] == "unknown"
        # %.6g renders a sub-epsilon span in exponent form; it must reach the archive as a
        # number, since the span is what the unit-mismatch guard then judges.
        assert rec["span"] == pytest.approx(1.5e-06)


class TestNumericAggregateGridMismatch:
    """Expect zero records in prod; a nonzero count means a model's CDF length drifted.

    Worth harvesting because the predecessor defect was invisible: group-by-VALUE
    aggregation medianed over a rotating SUBSET of the ensemble whenever an ft-fallback
    distribution mixed with PCHIP ones, and nothing recorded the partial membership.
    """

    def test_fields(self):
        rec = _parse_one(NUMERIC_AGGREGATE_GRID_MISMATCH_LINE)
        assert rec["marker"] == "numeric_aggregate_grid_mismatch"
        assert rec["model_index"] == 2
        assert rec["got_points"] == 201
        assert rec["expected_points"] == 11

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(NUMERIC_AGGREGATE_GRID_MISMATCH_LINE)
        assert rec["qid"] == 44620
        assert rec["qid_kind"] == "question_id"


class TestSpreadUndefined:
    def test_fields(self):
        rec = _parse_one(SPREAD_UNDEFINED_LINE)
        assert rec["marker"] == "spread_undefined"
        assert rec["qtype"] == "numeric"
        assert rec["models"] == 3
        # %.6g of a negative zero denominator renders "-0"; it must read as the number it
        # is rather than falling through to a string.
        assert rec["denominator"] == 0

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(SPREAD_UNDEFINED_LINE)
        assert rec["qid"] == 45363
        assert rec["qid_kind"] == "question_id"

    def test_positive_denominator_variant_of_the_same_shape_harvests(self):
        # The guard is `denominator <= 0`, so a plain 0 is the common case; qtype is a
        # captured field rather than a literal so a future binary/MC variant needs no
        # spec change.
        rec = _parse_one(
            PFX_WARN + "SPREAD_UNDEFINED: question=1 qtype=numeric denominator=0 models=2 — key-percentile spread "
            "is unmeasurable (non-positive denominator); reporting inf so it cannot read as agreement"
        )
        assert rec["denominator"] == 0
        assert rec["models"] == 2


class TestTsAnchorRoute:
    """The routing marker that made anchor coverage a query instead of an offline re-run.

    route_question used to log only the ambiguous/guard branches: 27 of the triple era's 30
    route-level misses were the silent `kw_no_keyword_hit` return and left no line in 1,800
    persisted run logs. Every decision now emits one line, and losing it from the archive
    would put the next coverage audit back to reconstructing routes offline.
    """

    def test_routed_fields(self):
        rec = _parse_one(TS_ANCHOR_ROUTE_ROUTED_LINE)
        assert rec["marker"] == "ts_anchor_route"
        assert rec["decision"] == "routed"
        assert rec["series"] == "PAYEMS"
        assert rec["step"] == "kw_single"

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(TS_ANCHOR_ROUTE_ROUTED_LINE)
        assert rec["qid"] == 45401
        assert rec["qid_kind"] == "question_id"

    def test_derivation_gate_skip_names_the_refusing_entry(self):
        # The q45401 defect class: title keywords hit, the quantity gate refused. Naming
        # the series is what makes the marker actionable — a bare "skipped" would collapse
        # this back into the no-keyword miss it was previously indistinguishable from.
        rec = _parse_one(TS_ANCHOR_ROUTE_GATE_SKIP_LINE)
        assert rec["decision"] == "skipped"
        assert rec["series"] == "PAYEMS"
        assert rec["step"] == "kw_derivation_gate"

    def test_a_plain_keyword_miss_reads_none_series(self):
        rec = _parse_one(TS_ANCHOR_ROUTE_NO_HIT_LINE)
        # "none" is a _NONE_SENTINELS value, so a series-less skip reads as None.
        assert rec["series"] is None
        assert rec["step"] == "kw_no_keyword_hit"

    def test_a_spread_ref_survives_whole(self):
        rec = _parse_one(TS_ANCHOR_ROUTE_SPREAD_LINE)
        assert rec["series"] == "CL=F/^GSPC"
        assert rec["step"] == "url_spread"


class TestFinancialStaleLatest:
    """The stale-"latest" disclosure, one spec for both emitting surfaces.

    Informational, not alertable: the render already tells the forecaster to treat the
    value as stale. Harvesting it is what turns "how often does each surface serve a
    stale anchor value" into a query — run logs expire from GHA at 90 days.
    """

    def test_yfinance_surface_fields(self):
        rec = _parse_one(FINANCIAL_STALE_LATEST_YFINANCE_LINE)
        assert rec["marker"] == "financial_stale_latest"
        assert rec["surface"] == "financial_data"
        assert rec["symbol"] == "TEST"
        assert rec["age_d"] == 3
        assert rec["cadence"] == "calendar-day"

    def test_ts_anchor_surface_fields(self):
        # A caret-prefixed Yahoo index symbol must survive as the string it is.
        rec = _parse_one(FINANCIAL_STALE_LATEST_TS_ANCHOR_LINE)
        assert rec["surface"] == "ts_anchor"
        assert rec["symbol"] == "^DEAD"
        assert rec["age_d"] == 9
        assert rec["cadence"] == "trading-day"

    def test_no_question_ref(self):
        # Per-identifier, not per-question (one question can fire several), so the
        # record carries no qid at all — same shape as the credit markers.
        rec = _parse_one(FINANCIAL_STALE_LATEST_YFINANCE_LINE)
        assert "qid" not in rec
        assert "qid_kind" not in rec


class TestCredit:
    def test_balance(self):
        rec = _parse_one(CREDIT_BALANCE_LINE)
        assert rec["marker"] == "credit_balance"
        assert rec["key"] == "donated"
        assert rec["phase"] == "start"
        assert rec["remaining"] == 123.45
        assert rec["usage"] == 4.16

    def test_balance_skip_line_has_no_balance(self):
        rec = _parse_one(CREDIT_BALANCE_SKIP_LINE)
        assert rec["key"] == "personal"
        assert rec["phase"] == "start"
        assert rec["remaining"] is None
        assert rec["usage"] is None

    def test_spend(self):
        rec = _parse_one(CREDIT_SPEND_LINE)
        assert rec["marker"] == "credit_spend"
        assert rec["key"] == "donated"
        assert rec["run_delta_usd"] == 3.34
        assert rec["remaining"] == 120.11

    def test_spend_na(self):
        rec = _parse_one(CREDIT_SPEND_NA_LINE)
        assert rec["run_delta_usd"] is None
        assert rec["remaining"] is None

    def test_spend_source_is_captured(self):
        rec = _parse_one(CREDIT_SPEND_REMAINING_SOURCE_LINE)
        assert rec["source"] == "remaining_delta"
        assert rec["run_delta_usd"] == 3.34

    def test_unsettled_zero_is_distinguishable_from_a_real_zero(self):
        # The whole point of source=. A 0.00 from the usage branch means "OpenRouter
        # had not settled yet", NOT "this run was free" — and the delta alone cannot
        # carry that distinction, so the archive has to.
        rec = _parse_one(CREDIT_SPEND_UNSETTLED_SOURCE_LINE)
        assert rec["run_delta_usd"] == 0.0
        assert rec["source"] == "usage_delta_unsettled"

    def test_pre_field_lines_parse_with_source_none(self):
        # Back-compat: a re-harvest of an older log must not drop the record. None
        # reads correctly as "this run predates the field", which is distinct from
        # any of the three real source values.
        rec = _parse_one(CREDIT_SPEND_LINE)
        assert rec.get("source") is None

    def test_floor_breach(self):
        rec = _parse_one(CREDIT_FLOOR_BREACH_LINE)
        assert rec["marker"] == "credit_floor_breach"
        assert rec["key"] == "donated"
        assert rec["remaining"] == 45.00
        assert rec["floor"] == 50.00


class TestHtmlCommentMarkers:
    def test_stacker_outcome(self):
        rec = _parse_one("<!-- STACKER_OUTCOME=primary -->")
        assert rec["marker"] == "stacker_outcome"
        assert rec["outcome"] == "primary"

    def test_stacker_outcome_skipped_config_off(self):
        # Longer literal must win over its "skipped" prefix in the alternation.
        rec = _parse_one("<!-- STACKER_OUTCOME=skipped_config_off -->")
        assert rec["marker"] == "stacker_outcome"
        assert rec["outcome"] == "skipped_config_off"

    def test_stacker_skip_reason(self):
        # The additive skip-reason companion: the reason the plain "skipped"
        # outcome can't express (single-forecaster skips compute no spread at all).
        for reason in ("single_forecaster", "spread_below_threshold", "config_off"):
            rec = _parse_one(f"<!-- STACKER_SKIP_REASON={reason} -->")
            assert rec["marker"] == "stacker_skip_reason"
            assert rec["reason"] == reason

    def test_stacker_skip_reason_does_not_collide_with_stacker_outcome(self):
        # One marker per line: a comment tail carries both markers on separate
        # lines, and each line must harvest as exactly its own marker.
        harvested = parse_log_text(
            "<!-- STACKER_OUTCOME=skipped -->\n<!-- STACKER_SKIP_REASON=single_forecaster -->\n",
            **_META,
        )
        assert [r["outcome"] for r in harvested["stacker_outcome"]] == ["skipped"]
        assert [r["reason"] for r in harvested["stacker_skip_reason"]] == ["single_forecaster"]

    def test_tools_used(self):
        rec = _parse_one("<!-- TOOLS_USED=false -->")
        assert rec["marker"] == "tools_used"
        assert rec["value"] is False

    def test_anchor_overshoot(self):
        rec = _parse_one("<!-- ANCHOR_OVERSHOOT_PP=+16.2 -->")
        assert rec["marker"] == "anchor_overshoot_pp"
        assert rec["pp"] == 16.2

    def test_clause_divergence(self):
        rec = _parse_one("<!-- CLAUSE_PRODUCT_DIVERGENCE_PP=-4.0 -->")
        assert rec["marker"] == "clause_product_divergence_pp"
        assert rec["pp"] == -4.0


class TestQidKindAcrossMarkers:
    """qid_kind names the id space of each marker's ``question`` ref, so a residual
    join can translate a query rather than silently dropping the other-keyed records.
    """

    def test_post_id_markers(self):
        assert _parse_one(GHOST_PRE_LINE)["qid_kind"] == "post_id"
        assert _parse_one(GHOST_PRE_JSON_LINE)["qid_kind"] == "post_id"
        assert _parse_one(GHOST_FORECAST_LINE)["qid_kind"] == "post_id"
        assert _parse_one(GHOST_FORECAST_JSON_LINE)["qid_kind"] == "post_id"

    def test_question_id_markers(self):
        assert _parse_one(OPEN_BOUND_PILING_LINE)["qid_kind"] == "question_id"
        assert _parse_one(CLOSE_MARGIN_LINE)["qid_kind"] == "question_id"

    def test_credit_markers_have_no_qid_kind(self):
        # No ``question`` ref -> no id space -> the record carries neither qid nor qid_kind.
        rec = _parse_one(CREDIT_SPEND_LINE)
        assert "qid_kind" not in rec
        assert "qid" not in rec

    def test_divergent_question_recovered_by_both_id_forms(self):
        # The real 38880/38195 divergence: EXTRACTION_RUNG carries the QUESTION id
        # (38195), GAP_FILL_V2 carries the POST id (38880), same question. A per-marker
        # grep on one id would miss the other; qid_kind tags each so a join can unify
        # them (see tests/test_id_mapping.py::TestMarkerRecordsForQuestion).
        extraction = PFX + (
            "EXTRACTION_RUNG: question=38195 model=openai/gpt-5.6-sol qtype=numeric rung=block block_present=True"
        )
        gap_fill = (
            "2026-07-19 06:30:00,000 - metaculus_bot.research.agentic.loop - INFO - "
            "question=https://www.metaculus.com/questions/38880/ GAP_FILL_V2: model=openai/gpt-5.6-terra "
            "steps=7 tool_calls=9 searches=4 fetches=3 rendered=1 reads=2 dup_tool_calls=0 deadline_hit=False "
            "concluded_early=True wall_s=312.44 findings=5 pending_leads=1 lint_rejections=0"
        )
        harvested = parse_log_text(extraction + "\n" + gap_fill + "\n", **_META)
        er = harvested["extraction_rung"][0]
        gf = harvested["gap_fill_v2"][0]
        assert (er["qid"], er["qid_kind"]) == (38195, "question_id")
        assert (gf["qid"], gf["qid_kind"]) == (38880, "post_id")


class TestParseLogText:
    def test_multiple_markers_and_seq(self):
        text = "\n".join(
            [
                EXTRACTION_RUNG_LINE,
                "some unrelated log line - INFO - nothing here",
                GAP_FILL_V2_LINE,
                EXTRACTION_RUNG_LINE,  # a second extraction line -> seq 1
                CREDIT_SPEND_LINE,
            ]
        )
        harvested = parse_log_text(text, **_META)
        assert len(harvested["extraction_rung"]) == 2
        assert [r["seq"] for r in harvested["extraction_rung"]] == [
            0,
            1,
        ]  # HARNESS-SCAN-EXEMPT-object-explosion  # small list of dicts, not a DataFrame
        assert len(harvested["gap_fill_v2"]) == 1
        assert len(harvested["credit_spend"]) == 1

    def test_noise_only_yields_nothing(self):
        harvested = parse_log_text("just some logs\nno markers at all\n", **_META)
        assert all(len(v) == 0 for v in harvested.values())

    def test_every_spec_has_a_filename_stem(self):
        # Guards the archive layout: one JSONL file per marker type.
        stems = {
            spec.name for spec in MARKER_SPECS
        }  # HARNESS-SCAN-EXEMPT-object-explosion  # list of dataclasses, not a DataFrame column
        assert "extraction_rung" in stems
        assert "ghost_forecast" in stems
        assert "credit_balance" in stems
        assert len(stems) == len(MARKER_SPECS), "marker names must be unique (one file per type)"


# Example lines copied from the emitting format string
# (metaculus_bot/drop_telemetry.py:emit_drop_telemetry) — the source of
# truth, so a producer-side shape change breaks these loudly.
FORECASTER_DROPS_LINE = (
    PFX + "FORECASTER_DROPS: total=3 systematic=openrouter/anthropic/claude-opus-4.8 "
    'detail={"openrouter/anthropic/claude-opus-4.8":{"zero_output":2},'
    '"openrouter/google/gemini-3.1-pro-preview":{"timeout_soft_deadline":1}}'
)
FORECASTER_DROPS_CLEAN_LINE = PFX + "FORECASTER_DROPS: total=0 systematic=none detail={}"


class TestForecasterDrops:
    def test_fields(self):
        rec = _parse_one(FORECASTER_DROPS_LINE)
        assert rec["marker"] == "forecaster_drops"
        assert rec["total"] == 3
        # A '/'-laden OpenRouter slug survives the systematic field intact.
        assert rec["systematic"] == "openrouter/anthropic/claude-opus-4.8"
        # Per-run summary: no per-question ref, so no qid space is stamped.
        assert "qid" not in rec
        assert rec["qid_kind"] is None if "qid_kind" in rec else True

    def test_detail_json_round_trips(self):
        rec = _parse_one(FORECASTER_DROPS_LINE)
        # detail stays a raw string (never coerced) so residual analysis can json.loads it;
        # the nested model->cause->count survives slugs with slashes and dots.
        assert json.loads(rec["detail"]) == {
            "openrouter/anthropic/claude-opus-4.8": {"zero_output": 2},
            "openrouter/google/gemini-3.1-pro-preview": {"timeout_soft_deadline": 1},
        }

    def test_clean_run_line_parses_with_zero_total(self):
        rec = _parse_one(FORECASTER_DROPS_CLEAN_LINE)
        assert rec["total"] == 0
        assert rec["systematic"] is None  # "none" sentinel coerces to None
        assert json.loads(rec["detail"]) == {}


# Verbatim from metaculus_bot/forecaster.py:_research_and_make_predictions — the
# positive per-question counterpart to FORECASTER_DROPS above.
FORECASTERS_SURVIVED_FULL_LINE = (
    PFX + "FORECASTERS_SURVIVED: question=70002 survived=3/3 models=claude-opus-4.8,gemini-3.1-pro-preview,gpt-5.6-sol"
)
FORECASTERS_SURVIVED_DEGRADED_LINE = PFX + "FORECASTERS_SURVIVED: question=14333 survived=1/3 models=gpt-5.6-sol"
FORECASTERS_SURVIVED_UNKNOWN_LINE = PFX + "FORECASTERS_SURVIVED: question=14333 survived=2/3 models=unknown"


class TestForecastersSurvived:
    def test_full_ensemble_fields(self):
        rec = _parse_one(FORECASTERS_SURVIVED_FULL_LINE)
        assert rec["marker"] == "forecasters_survived"
        assert rec["survived"] == 3
        assert rec["configured"] == 3
        assert rec["models"] == "claude-opus-4.8,gemini-3.1-pro-preview,gpt-5.6-sol"

    def test_question_ref_is_stamped_in_the_question_id_space(self):
        # forecaster.py emits question.id_of_question, not the post id — a residual
        # join has to know which space to translate into.
        rec = _parse_one(FORECASTERS_SURVIVED_FULL_LINE)
        assert rec["qid"] == 70002
        assert rec["qid_kind"] == "question_id"

    def test_degraded_run_is_distinguishable_from_a_full_one(self):
        # The whole reason the marker exists: at a low MIN_FORECASTERS_TO_PUBLISH a
        # 1-of-3 publish exits zero, so the archive must be able to tell it apart.
        rec = _parse_one(FORECASTERS_SURVIVED_DEGRADED_LINE)
        assert rec["survived"] == 1
        assert rec["configured"] == 3
        assert rec["survived"] < rec["configured"]
        assert rec["models"] == "gpt-5.6-sol"

    def test_unknown_models_sentinel_survives_as_a_string(self):
        # "unknown" is the fallback when no prediction carried a Model: prefix. It is
        # NOT in _NONE_SENTINELS, so it must stay a readable string rather than None
        # — a None here would be indistinguishable from a missing field.
        rec = _parse_one(FORECASTERS_SURVIVED_UNKNOWN_LINE)
        assert rec["models"] == "unknown"


# The FORECASTERS_USED ensemble-size marker is an HTML comment injected into the
# published comment (metaculus_bot/comment/markers.py); its durable home is the
# comment, but the run-log parser carries a spec too (same as STACKER_OUTCOME /
# TOOLS_USED) so it stays complete if a comment body is ever logged.
FORECASTERS_USED_LINE = PFX + "<!-- FORECASTERS_USED=2/3 -->"


class TestForecastersUsed:
    def test_fields(self):
        rec = _parse_one(FORECASTERS_USED_LINE)
        assert rec["marker"] == "forecasters_used"
        assert rec["used"] == 2
        assert rec["configured"] == 3


# The per-run degradation summary — the single line that decides CI color, since
# cli.py exits non-zero whenever alertable_count is positive. Copied from the
# format string in metaculus_bot/degradation_counters.py
# (format_degradation_summary), the source of
# truth. Without a spec here the archive held no record of the counter that reddens
# every run: the 2026-07-26 research_provider_timeouts -> research_provider_failures
# rename would have been invisible to a replay.
DEGRADATION_COUNTERS_LINE = (
    PFX + "Degradation counters: forecasters_dropped=2, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_failures=1, summarizer_failures=3, gap_fill_v2_errors=0, "
    "prediction_market_degraded=0, prediction_market_source_losses=4, provider_degradation=1, "
    "publish_attempt_failures=1, publish_skipped_closed=2, time_budget_fast_path=3, "
    "research_budget_cuts=5"
)
# The shape emitted before the off-fast-path budget-cut counter shipped: ends at
# time_budget_fast_path. Same optional-group rationale as every tail before it.
DEGRADATION_COUNTERS_PRE_BUDGET_CUT_LINE = (
    PFX + "Degradation counters: forecasters_dropped=2, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_failures=1, summarizer_failures=3, gap_fill_v2_errors=0, "
    "prediction_market_degraded=0, prediction_market_source_losses=4, provider_degradation=1, "
    "publish_attempt_failures=1, publish_skipped_closed=2, time_budget_fast_path=3"
)
# The shape emitted before the time-budget counter shipped: ends at
# publish_skipped_closed. Same optional-group rationale as every tail before it.
DEGRADATION_COUNTERS_NO_BUDGET_TAIL_LINE = (
    PFX + "Degradation counters: forecasters_dropped=2, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_failures=1, summarizer_failures=3, gap_fill_v2_errors=0, "
    "prediction_market_degraded=0, prediction_market_source_losses=4, provider_degradation=1, "
    "publish_attempt_failures=1, publish_skipped_closed=2"
)
# The 2026-08-25-and-earlier shape: ends at publish_attempt_failures, i.e. no
# publish_skipped_closed tail (the close-time gate's counter). Same optional-group
# rationale as its predecessors — the regex is $-anchored, so a mandatory tail would
# drop every record before the gate shipped.
DEGRADATION_COUNTERS_NO_SKIP_TAIL_LINE = (
    PFX + "Degradation counters: forecasters_dropped=2, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_failures=1, summarizer_failures=3, gap_fill_v2_errors=0, "
    "prediction_market_degraded=0, prediction_market_source_losses=4, provider_degradation=1, "
    "publish_attempt_failures=1"
)
# The 2026-08-24-and-earlier shape: ends at provider_degradation, i.e. no
# publish_attempt_failures tail. Every archived record until that date has this
# shape, so the newest key is optional-group wrapped like its predecessors.
DEGRADATION_COUNTERS_NO_PUBLISH_TAIL_LINE = (
    PFX + "Degradation counters: forecasters_dropped=2, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_failures=1, summarizer_failures=3, gap_fill_v2_errors=0, "
    "prediction_market_degraded=0, prediction_market_source_losses=4, provider_degradation=1"
)
# The shape every one of the 290 archived records carries: the same keys as the
# line above but ending at prediction_market_source_losses, i.e. no
# provider_degradation tail. Replace-by-run re-harvesting replays these logs, so a
# MANDATORY new group would drop all 290 records wholesale on the next sync.
DEGRADATION_COUNTERS_NO_PROVIDER_TAIL_LINE = (
    PFX + "Degradation counters: forecasters_dropped=0, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_failures=0, summarizer_failures=0, gap_fill_v2_errors=0, "
    "prediction_market_degraded=0, prediction_market_source_losses=0"
)
# Pre-rename shape (research_provider_timeouts, no summarizer_failures, and
# prediction_market_platform_failures as the trailing key). Replace-by-run
# re-harvesting replays these old logs, so the trailing keys are optional-group
# wrapped — a mandatory tail would drop every pre-rename record wholesale rather
# than harvesting the counters it does carry (same rationale as gap_fill_v2).
DEGRADATION_COUNTERS_LEGACY_LINE = (
    PFX + "Degradation counters: forecasters_dropped=0, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_timeouts=5, gap_fill_v2_errors=0, prediction_market_degraded=1"
)


class TestDegradationCounters:
    def test_all_fifteen_current_keys_parse(self):
        rec = _parse_one(DEGRADATION_COUNTERS_LINE)
        assert rec["marker"] == "degradation_counters"
        assert rec["research_budget_cuts"] == 5
        assert rec["time_budget_fast_path"] == 3
        assert rec["publish_skipped_closed"] == 2
        assert rec["forecasters_dropped"] == 2
        assert rec["questions_failed_to_publish"] == 0

    def test_pre_budget_cut_line_still_harvests_everything_else(self):
        rec = _parse_one(DEGRADATION_COUNTERS_PRE_BUDGET_CUT_LINE)
        assert rec["time_budget_fast_path"] == 3
        assert "research_budget_cuts" not in rec
        assert rec["stacker_primary_failed"] == 0
        assert rec["stacker_fallback_used"] == 0
        assert rec["stacker_fallback_failed"] == 0
        assert rec["research_provider_failures"] == 1
        assert rec["summarizer_failures"] == 3
        assert rec["gap_fill_v2_errors"] == 0
        assert rec["prediction_market_degraded"] == 0
        assert rec["prediction_market_source_losses"] == 4
        assert rec["provider_degradation"] == 1
        assert rec["publish_attempt_failures"] == 1

    def test_line_without_the_budget_tail_still_harvests_everything_else(self):
        """Every record archived before the time-budget counter shipped ends at
        publish_skipped_closed, and that now-lazy group must still capture its full
        value there rather than handing a digit to backtracking."""
        rec = _parse_one(DEGRADATION_COUNTERS_NO_BUDGET_TAIL_LINE)
        assert rec["marker"] == "degradation_counters"
        assert rec["publish_skipped_closed"] == 2
        assert rec["publish_attempt_failures"] == 1
        # Absent must read as "this era didn't emit it", never as a measured zero.
        assert "time_budget_fast_path" not in rec

    def test_line_without_the_skip_tail_still_harvests_everything_else(self):
        """Every record archived before 2026-08-25 ends at publish_attempt_failures,
        and that now-lazy group must still capture its full value there rather than
        handing a digit to backtracking, while the newest key reads as absent."""
        rec = _parse_one(DEGRADATION_COUNTERS_NO_SKIP_TAIL_LINE)
        assert rec["marker"] == "degradation_counters"
        assert rec["publish_attempt_failures"] == 1
        assert rec["provider_degradation"] == 1
        # Absent must read as "this era didn't emit it", never as a measured zero.
        assert "publish_skipped_closed" not in rec
        assert "time_budget_fast_path" not in rec

    def test_line_without_the_publish_tail_still_harvests_everything_else(self):
        """Every record archived before 2026-08-24 ends at provider_degradation, and
        the lazy provider_degradation group must still capture its full value there
        (not hand a digit to backtracking) while the newer keys read as absent."""
        rec = _parse_one(DEGRADATION_COUNTERS_NO_PUBLISH_TAIL_LINE)
        assert rec["marker"] == "degradation_counters"
        assert rec["provider_degradation"] == 1
        assert rec["prediction_market_source_losses"] == 4
        # Absent must read as "this era didn't emit it", never as a measured zero.
        assert "publish_attempt_failures" not in rec
        assert "publish_skipped_closed" not in rec
        assert "time_budget_fast_path" not in rec

    def test_line_without_the_provider_degradation_tail_still_harvests_everything_else(self):
        """The load-bearing back-compat case. All 290 archived records end at
        prediction_market_source_losses, so the new tail has to be optional-group
        wrapped — a mandatory group would drop every one of them on the next
        replace-by-run re-harvest rather than harvesting the ten counters it carries.
        """
        rec = _parse_one(DEGRADATION_COUNTERS_NO_PROVIDER_TAIL_LINE)
        assert rec["marker"] == "degradation_counters"
        assert rec["forecasters_dropped"] == 0
        assert rec["research_provider_failures"] == 0
        assert rec["summarizer_failures"] == 0
        assert rec["gap_fill_v2_errors"] == 0
        assert rec["prediction_market_degraded"] == 0
        assert rec["prediction_market_source_losses"] == 0
        # Absent must read as "this era didn't emit it", never as a measured zero.
        assert "provider_degradation" not in rec
        assert "publish_attempt_failures" not in rec
        assert "publish_skipped_closed" not in rec
        assert "time_budget_fast_path" not in rec

    def test_per_run_summary_carries_no_question_ref(self):
        rec = _parse_one(DEGRADATION_COUNTERS_LINE)
        # Aggregates a whole run, so there is no id space to stamp.
        assert "qid" not in rec

    def test_pre_rename_line_still_harvests_its_leading_counters(self):
        # The pre-rename keys it shares with today's line must still come through;
        # the renamed/added ones are absent rather than dropping the whole record.
        rec = _parse_one(DEGRADATION_COUNTERS_LEGACY_LINE)
        assert rec["marker"] == "degradation_counters"
        assert rec["forecasters_dropped"] == 0
        assert rec["stacker_fallback_failed"] == 0
        assert rec["research_provider_timeouts"] == 5
        assert rec["gap_fill_v2_errors"] == 0
        assert rec["prediction_market_degraded"] == 1
        # Keys that did not exist pre-rename are ABSENT from the record, not 0 —
        # absent must not read as "measured zero" in the archive.
        assert "research_provider_failures" not in rec
        assert "summarizer_failures" not in rec
        assert "prediction_market_source_losses" not in rec

    def test_a_future_counter_harvests_with_no_spec_change(self):
        # The whole point of the tokenized tail: appending a key to
        # format_degradation_summary must never again require a markers.py edit.
        line = (
            "2026-08-25 12:00:00,000 - metaculus_bot.forecaster - INFO - "
            "Degradation counters: forecasters_dropped=0, some_future_counter=7"
        )
        rec = _parse_one(line)
        assert rec["forecasters_dropped"] == 0
        assert rec["some_future_counter"] == 7


class TestTimeBudgetLoudMarkers:
    """The four budget WARNs docs/operations.md tells the operator to grep. Without
    specs they vanished at the 90-day GHA log expiry; the mid-phase gap-fill cut on
    a non-fast-path question was recoverable from nothing else."""

    def test_time_budget_fast_path_roundtrip(self):
        line = (
            "2026-08-25 12:00:00,000 - metaculus_bot.forecaster - WARNING - "
            "TIME_BUDGET_FAST_PATH: qid=45085 budget=1140s close_time=2026-08-25 13:00:00+00:00; "
            "dropping the slow search providers and gap-fill to protect the prediction POST"
        )
        rec = _parse_one(line)
        assert rec["marker"] == "time_budget_fast_path"
        assert rec["qid"] == 45085
        assert rec["qid_kind"] == "question_id"
        assert rec["budget_s"] == pytest.approx(1140.0)
        assert rec["close_time"] == "2026-08-25 13:00:00+00:00"

    def test_research_phase_deadline_roundtrip(self):
        line = (
            "2026-08-25 12:00:00,000 - metaculus_bot.research.orchestrator - WARNING - "
            "RESEARCH_PHASE_DEADLINE: cancelled 2/6 providers after 570s (gemini_search,native_search)"
        )
        rec = _parse_one(line)
        assert rec["marker"] == "research_phase_deadline"
        assert rec["cancelled"] == 2
        assert rec["total"] == 6
        assert rec["deadline_s"] == pytest.approx(570.0)
        assert rec["providers"] == "gemini_search,native_search"
        # No question ref on this line; attribution lives in provider_results.
        assert "qid" not in rec

    def test_gap_fill_skipped_for_budget_roundtrip(self):
        line = (
            "2026-08-25 12:00:00,000 - metaculus_bot.research.orchestrator - WARNING - "
            "GAP_FILL_SKIPPED_FOR_BUDGET: question=45085 fast_path=true research_phase_remaining=n/a"
        )
        rec = _parse_one(line)
        assert rec["marker"] == "gap_fill_skipped_for_budget"
        assert rec["qid"] == 45085
        assert rec["fast_path"] is True
        assert rec["research_phase_remaining"] is None  # "n/a" coerces to None

    def test_gap_fill_cut_for_budget_roundtrip_both_passes(self):
        for gap_fill_pass in ("V1", "V2"):
            line = (
                "2026-08-25 12:00:00,000 - metaculus_bot.research.orchestrator - WARNING - "
                f"GAP_FILL_{gap_fill_pass}_CUT_FOR_BUDGET: question=45085; research phase ran out of budget"
            )
            rec = _parse_one(line)
            assert rec["marker"] == "gap_fill_cut_for_budget"
            assert rec["qid"] == 45085
            assert rec["gap_fill_pass"] == gap_fill_pass


# The per-run provider-degradation summary (metaculus_bot/research/provider_health.py
# log_provider_degradation_summary), the marker counterpart to the
# provider_degradation counter above. Two shapes: findings, and the healthy zero
# (emitted anyway, so "no provider degraded" is recorded rather than absent).
PROVIDER_DEGRADATION_LINE = (
    PFX + "PROVIDER_DEGRADATION: run=30784152530 findings=2 alertable=2 suppressed=0 "
    'detail=[{"signal":"market_field_contract","venue":"kalshi","questions":1,'
    '"fields":"total_volume,open_interest","pool_rows":40},'
    '{"signal":"catalogue_empty","venue":"predictit_markets","questions":1,"entries":0,"fetch_ok":true}]'
)
PROVIDER_DEGRADATION_CLEAN_LINE = PFX + "PROVIDER_DEGRADATION: run=local findings=0 alertable=0 suppressed=0 detail=[]"
# Current shape (2026-08-24): the observation denominators that let a reader tell a
# measured zero (venues_observed=4 pool_rows=404) from a vacuous one (a run that
# forecast nothing and evaluated nothing — 96% of the archived records).
PROVIDER_DEGRADATION_DENOMINATED_LINE = (
    PFX + "PROVIDER_DEGRADATION: run=32300000000 findings=0 alertable=0 suppressed=0 "
    "venues_observed=4 catalogues_observed=2 pool_rows=404 detail=[]"
)
PROVIDER_DEGRADATION_SUPPRESSED_LINE = (
    PFX + "PROVIDER_DEGRADATION: run=local findings=1 alertable=0 suppressed=1 "
    'detail=[{"signal":"market_field_contract","venue":"manifold","questions":1,"fields":"num_bettors",'
    '"pool_rows":12,"suppressed_until":"2026-09-10"}] '
    "(manifold:market_field_contract suppressed until 2026-09-10); run stays green on those."
)


class TestProviderDegradation:
    def test_fields(self):
        rec = _parse_one(PROVIDER_DEGRADATION_LINE)
        assert rec["marker"] == "provider_degradation"
        # A GHA run id is integer-looking so coerce_value makes it an int, while the
        # local sentinel stays a str (below). The archive's own ``run_id`` metadata
        # field is the join key either way; this group is the in-line cross-check.
        assert rec["run"] == 30784152530
        assert rec["findings"] == 2
        assert rec["alertable"] == 2
        assert rec["suppressed"] == 0

    def test_local_run_sentinel_stays_a_string(self):
        assert _parse_one(PROVIDER_DEGRADATION_CLEAN_LINE)["run"] == "local"

    def test_detail_json_round_trips(self):
        """``detail`` is a JSON array captured verbatim (it belongs to _RAW_FIELDS
        beside FORECASTER_DROPS' detail): venue and field names are delimiter-hostile
        and residual analysis json.loads it, so coercion would mangle it."""
        rec = _parse_one(PROVIDER_DEGRADATION_LINE)
        payload = json.loads(rec["detail"])
        assert [entry["signal"] for entry in payload] == ["market_field_contract", "catalogue_empty"]
        assert payload[0]["fields"] == "total_volume,open_interest"
        assert payload[1]["venue"] == "predictit_markets"

    def test_clean_run_parses_with_an_empty_detail_array(self):
        """A measured zero is signal, so the marker fires on healthy runs too and the
        parser has to accept the empty array rather than skipping the line."""
        rec = _parse_one(PROVIDER_DEGRADATION_CLEAN_LINE)
        assert rec["findings"] == 0
        assert json.loads(rec["detail"]) == []

    def test_suppressed_run_keeps_its_arithmetic_and_resume_date(self):
        """The suppression clause is free text AFTER the JSON, so ``detail`` must stop
        at the closing bracket instead of swallowing it."""
        rec = _parse_one(PROVIDER_DEGRADATION_SUPPRESSED_LINE)
        assert rec["findings"] == 1
        assert rec["alertable"] == 0
        assert rec["suppressed"] == 1
        assert json.loads(rec["detail"])[0]["suppressed_until"] == "2026-09-10"

    def test_observation_denominators_parse(self):
        rec = _parse_one(PROVIDER_DEGRADATION_DENOMINATED_LINE)
        assert rec["findings"] == 0
        assert rec["venues_observed"] == 4
        assert rec["catalogues_observed"] == 2
        assert rec["pool_rows"] == 404

    def test_pre_denominator_lines_read_absent_not_zero(self):
        """All ~1039 archived lines predate the denominators; on a re-harvest they
        must keep parsing, with the new fields None — a vacuous zero must never be
        promoted into a measured one."""
        for line in (PROVIDER_DEGRADATION_LINE, PROVIDER_DEGRADATION_CLEAN_LINE, PROVIDER_DEGRADATION_SUPPRESSED_LINE):
            rec = _parse_one(line)
            assert rec["venues_observed"] is None
            assert rec["catalogues_observed"] is None
            assert rec["pool_rows"] is None

    def test_per_run_summary_carries_no_question_ref(self):
        rec = _parse_one(PROVIDER_DEGRADATION_LINE)
        assert "qid" not in rec


# The per-CALL donated->personal key fallback WARN (fallback_openrouter.py
# _log_fallback). Its counters already ride degradation_counters and the cli
# summary, but only this line names WHICH model fell back — the difference between
# one flaky Gemini call and every forecaster running on the operator's paid key.
PAID_FALLBACK_LINE = (
    PFX_WARN + "PAID PERSONAL-KEY FALLBACK: donated OpenRouter key failed for model=openai/gpt-5.6-sol, "
    "so this call billed to the personal OPENROUTER_API_KEY instead of the free donated key. "
    "Run will complete, then exit non-zero to alert. error=APIError: litellm.APIError: "
    'OpenrouterException - {"error":{"message":"Key limit exceeded (total limit)","code":403}}'
)
PAID_FALLBACK_SUPPRESSED_LINE = (
    PFX_WARN + "PAID PERSONAL-KEY FALLBACK: donated OpenRouter key failed for model=anthropic/claude-opus-4.8, "
    "so this call billed to the personal OPENROUTER_API_KEY instead of the free donated key. "
    "Cause is a credit shortfall, so it is NOT counted as alertable until 2026-09-10 "
    "(operator is self-funding the season). error=APIError: insufficient credit"
)


class TestPaidPersonalKeyFallback:
    def test_model_and_error_are_captured(self):
        rec = _parse_one(PAID_FALLBACK_LINE)
        assert rec["marker"] == "paid_personal_key_fallback"
        assert rec["model"] == "openai/gpt-5.6-sol"
        assert rec["error_type"] == "APIError"
        # ``error`` holds the exception's str, which carries the 403 spend-cap phrase
        # that distinguishes a drained key from a moderation refusal.
        assert "Key limit exceeded" in rec["error"]

    def test_suppressed_variant_parses_the_same(self):
        """The alert-note clause between the model and the error differs by cause, so
        the spec must not depend on its wording."""
        rec = _parse_one(PAID_FALLBACK_SUPPRESSED_LINE)
        assert rec["model"] == "anthropic/claude-opus-4.8"
        assert rec["error"] == "insufficient credit"

    def test_the_404_variant_is_not_captured_as_this_marker(self):
        """The 404 no-allowed-providers branch logs a DIFFERENT line with no
        ``PAID PERSONAL-KEY FALLBACK`` token, and it means something else (the
        donated key's provider list doesn't cover the model, not a spend problem),
        so it must not be harvested here."""
        line = (
            PFX_WARN + "Donated OpenRouter key returned 404 'no allowed providers' for model=x-ai/grok-4.5; "
            "falling back to general (paid personal) key. error=APIError: 404"
        )
        harvested = parse_log_text(line + "\n", **_META)
        assert harvested["paid_personal_key_fallback"] == []


# The per-attempt publish-failure WARN (publish_hardening.py _wrap_with_timeout_retry).
# Two emitted shapes; the failed one is copied from q45085's real 405-closed run
# (2026-08-03), the incident that showed a publish failure left no harvestable trace.
PUBLISH_HARDENING_TIMEOUT_LINE = (
    PFX_WARN + "PUBLISH_HARDENING: _post_question_prediction attempt 1/2 timed out after 20s"
)
PUBLISH_HARDENING_FAILED_LINE = (
    PFX_WARN + "PUBLISH_HARDENING: _post_question_prediction attempt 2/2 failed "
    "(HTTPError: Error while posting prediction: Status code: 405. "
    'Response: {"error":"Question 45085 is already closed to forecasting !"})'
)


class TestPublishHardening:
    def test_timeout_shape(self):
        rec = _parse_one(PUBLISH_HARDENING_TIMEOUT_LINE)
        assert rec["marker"] == "publish_hardening"
        assert rec["method"] == "_post_question_prediction"
        assert rec["attempt"] == 1
        assert rec["attempts"] == 2
        assert rec["timeout_s"] == 20
        # Exactly one branch populates per record.
        assert rec["error_type"] is None
        assert rec["error"] is None

    def test_exception_shape_captures_class_and_message(self):
        rec = _parse_one(PUBLISH_HARDENING_FAILED_LINE)
        assert rec["method"] == "_post_question_prediction"
        assert rec["attempt"] == 2
        assert rec["attempts"] == 2
        assert rec["error_type"] == "HTTPError"
        assert "405" in rec["error"]
        assert rec["timeout_s"] is None

    def test_comment_post_method_parses_too(self):
        rec = _parse_one(PFX_WARN + "PUBLISH_HARDENING: post_question_comment attempt 1/2 timed out after 20s")
        assert rec["method"] == "post_question_comment"

    def test_other_publish_hardening_strings_are_not_harvested(self):
        """The module reuses the PUBLISH_HARDENING prefix in its seam-moved
        AttributeErrors and its loop-exited RuntimeError; only the per-attempt
        failure WARNs carry the ``attempt N/M`` clause, so only those harvest."""
        non_attempt_lines = [
            PFX + "Publish hardening applied: 2 MetaculusClient.post_* methods wrapped with 20s timeout + 1 retry",
            PFX_WARN + "PUBLISH_HARDENING: MetaculusClient defines no '_post_question_prediction' to patch. "
            "The forecasting-tools publish seam moved or was renamed; repoint _PATCHED_METHODS.",
            PFX_WARN + "PUBLISH_HARDENING: _post_question_prediction loop exited without running",
        ]
        for line in non_attempt_lines:
            harvested = parse_log_text(line + "\n", **_META)
            assert harvested["publish_hardening"] == [], line

    def test_per_call_marker_carries_no_question_ref(self):
        # The wrapper sees only the POST, so there is no id space to stamp.
        rec = _parse_one(PUBLISH_HARDENING_TIMEOUT_LINE)
        assert "qid" not in rec

    def test_the_new_not_retrying_line_is_not_harvested_as_an_attempt(self):
        """The non-retryable-4xx WARN shares the prefix but carries no ``attempt N/M``
        clause on purpose — folding it into the line above would have broken the
        attempt spec's anchored shape."""
        line = (
            PFX_WARN + "PUBLISH_HARDENING: _post_question_prediction not retrying status 405 "
            "— a second identical POST cannot succeed"
        )
        assert parse_log_text(line + "\n", **_META)["publish_hardening"] == []


# The per-question pre-publish skip WARN (publish_gate.py skip_publish_if_closed).
# Values are q45085's real numbers: fetched at 11:59:38Z against a 12:00:00Z close,
# publish reached at 12:05:06Z.
PUBLISH_SKIPPED_CLOSED_LINE = (
    PFX_WARN + "PUBLISH_SKIPPED_CLOSED: question=45085 reason=close_time_passed "
    "close_time=2026-08-03T12:00:00+00:00 now=2026-08-03T12:05:06+00:00 overdue_s=306 state=open"
)


class TestPublishSkippedClosed:
    def test_close_time_passed_shape(self):
        rec = _parse_one(PUBLISH_SKIPPED_CLOSED_LINE)
        assert rec["marker"] == "publish_skipped_closed"
        assert rec["reason"] == "close_time_passed"
        assert rec["close_time"] == "2026-08-03T12:00:00+00:00"
        assert rec["now"] == "2026-08-03T12:05:06+00:00"
        assert rec["overdue_s"] == 306
        assert rec["state"] == "open"

    def test_question_ref_is_stamped_in_the_question_id_space(self):
        # publish_gate emits question.id_of_question, so a residual join must not read
        # it as a post id (the two share one integer space).
        rec = _parse_one(PUBLISH_SKIPPED_CLOSED_LINE)
        assert rec["qid"] == 45085
        assert rec["qid_kind"] == "question_id"

    def test_state_closed_shape_with_absent_close_time(self):
        rec = _parse_one(
            PFX_WARN + "PUBLISH_SKIPPED_CLOSED: question=45093 reason=state_closed "
            "close_time=n/a now=2026-08-06T09:00:00+00:00 overdue_s=n/a state=resolved"
        )
        assert rec["reason"] == "state_closed"
        # n/a must read as absent, never as a measured zero overdue.
        assert rec["close_time"] is None
        assert rec["overdue_s"] is None

    def test_negative_overdue_parses(self):
        # An early admin close leaves close_time in the future, so overdue is negative.
        rec = _parse_one(
            PFX_WARN + "PUBLISH_SKIPPED_CLOSED: question=45093 reason=state_closed "
            "close_time=2026-08-07T00:00:00+00:00 now=2026-08-06T09:00:00+00:00 "
            "overdue_s=-54000 state=closed"
        )
        assert rec["overdue_s"] == -54000


# The per-question budget grant INFO (time_budget.py). Emitted for EVERY question,
# which is the point: CLOSE_MARGIN fires only after a SUCCESSFUL submission, so it is
# censored on exactly the thin-window questions the budget exists for. Values below
# are q45085's real close time against a 20-minutes-out fetch.
TIME_BUDGET_THIN_LINE = (
    PFX + "TIME_BUDGET: question=45085 budget_s=1140 close_time=2026-08-03T12:00:00+00:00 "
    "close_limited=true fast_path=true"
)
TIME_BUDGET_ROOMY_LINE = (
    PFX + "TIME_BUDGET: question=44870 budget_s=3510 close_time=2026-07-24T15:00:00+00:00 "
    "close_limited=false fast_path=false"
)


class TestTimeBudget:
    def test_thin_window_shape(self):
        rec = _parse_one(TIME_BUDGET_THIN_LINE)
        assert rec["marker"] == "time_budget"
        assert rec["budget_s"] == 1140
        assert rec["close_time"] == "2026-08-03T12:00:00+00:00"
        assert rec["close_limited"] is True
        assert rec["fast_path"] is True

    def test_roomy_window_shape_reads_the_static_budget(self):
        """The uncensored denominator: a roomy question emits this line too, so a later
        round can measure how often a window is actually thin."""
        rec = _parse_one(TIME_BUDGET_ROOMY_LINE)
        assert rec["budget_s"] == 3510
        assert rec["close_limited"] is False
        assert rec["fast_path"] is False

    def test_absent_close_time_reads_as_none(self):
        rec = _parse_one(
            PFX + "TIME_BUDGET: question=14333 budget_s=3510 close_time=n/a close_limited=false fast_path=false"
        )
        # n/a must read as absent, never as a parsed timestamp or a zero.
        assert rec["close_time"] is None

    def test_question_ref_is_stamped_in_the_question_id_space(self):
        # time_budget emits question.id_of_question, so a residual join must not read it
        # as a post id (the two share one integer space).
        rec = _parse_one(TIME_BUDGET_THIN_LINE)
        assert rec["qid"] == 45085
        assert rec["qid_kind"] == "question_id"


# The end-of-run alertable breakdown (cli.py). Emitted on BOTH exit paths — the
# fully-suppressed green run is exactly the one that would otherwise leave no
# record, since the credit subset can cancel the whole generic total and read
# alertable=0 alongside real degradation.
RUN_ALERTABLE_RED_LINE = (
    PFX_WARN + "Run completed with 3 alertable degradation event(s) (bot=2, personal_key_fallback=1 "
    "of which donated_404=1, credit=0); exiting non-zero so CI marks this run red."
)
RUN_ALERTABLE_SUPPRESSED_LINE = (
    PFX + "Run completed with 0 alertable degradation event(s) (bot=0, personal_key_fallback=7 "
    "of which donated_404=0, credit=7 with 7 credit event(s) suppressed until 2026-09-10, "
    "donated_key=drained); every fallback was a suppressed credit event, so this run stays green."
)
# The 2026-08-25 addition: a run where nothing degraded at all. It logged no line
# before, so the census counted only degraded runs — which the drained-key window
# hid, since every run in it fell back at least once.
RUN_ALERTABLE_CLEAN_LINE = (
    PFX + "Run completed clean with 0 alertable degradation event(s) (bot=0, personal_key_fallback=0 "
    "of which donated_404=0, credit=0 with 0 credit event(s) suppressed until 2026-09-10); "
    "nothing degraded, so this run stays green."
)


class TestRunAlertableSummary:
    def test_red_run_fields(self):
        rec = _parse_one(RUN_ALERTABLE_RED_LINE)
        assert rec["marker"] == "run_alertable_summary"
        assert rec["alertable"] == 3
        assert rec["bot"] == 2
        assert rec["personal_key_fallback"] == 1
        assert rec["donated_404"] == 1
        assert rec["credit"] == 0
        # No suppression mid-clause and no probe ran, so both are absent rather than
        # zero — "never needed a probe" must not read as "the probe said unknown".
        assert rec["suppressed_credit"] is None
        assert rec["donated_key"] is None
        # A degraded line carries no phrase marker; ``outcome`` is only ever "clean".
        assert rec["outcome"] is None

    def test_suppressed_green_run_carries_the_probe_verdict(self):
        """The shape the drained-donated-key incident produced: alertable=0 with seven
        real fallbacks. The verdict is what tells a reader why nothing was counted."""
        rec = _parse_one(RUN_ALERTABLE_SUPPRESSED_LINE)
        assert rec["alertable"] == 0
        assert rec["personal_key_fallback"] == 7
        assert rec["credit"] == 7
        assert rec["suppressed_credit"] == 7
        assert rec["resume_date"] == "2026-09-10"
        assert rec["donated_key"] == "drained"
        # Pre-2026-08-25 records carry no phrase marker, and neither does any
        # degraded line since — ``outcome`` is what says "clean", so it must stay
        # absent here rather than defaulting to it.
        assert rec["outcome"] is None

    def test_clean_run_is_harvested_and_flagged(self):
        """The all-clear shape harvests as the same marker, distinguishable by
        ``outcome`` rather than by its all-zero fields — a run that lost a question
        also reads all zeros (q45085's shape) and keeps the plain phrase."""
        rec = _parse_one(RUN_ALERTABLE_CLEAN_LINE)
        assert rec["marker"] == "run_alertable_summary"
        assert rec["outcome"] == "clean"
        assert rec["alertable"] == 0
        assert rec["bot"] == 0
        assert rec["personal_key_fallback"] == 0
        assert rec["donated_404"] == 0
        assert rec["credit"] == 0
        assert rec["suppressed_credit"] == 0
        assert rec["donated_key"] is None


# Gemini ungrounded-suppression WARN (metaculus_bot/research/gemini_search.py
# _format_grounded_response). The section is suppressed and "" returned, which the
# orchestrator records as status="empty" — not alertable and not otherwise counted,
# so the archive is the only way to measure how often grounding silently produced
# nothing.
GEMINI_UNGROUNDED_LINE = PFX_WARN + "GEMINI_UNGROUNDED_SUPPRESSED: question=38195 model=gemini-3.5-flash queries=3"


class TestGeminiUngroundedSuppressed:
    def test_fields(self):
        rec = _parse_one(GEMINI_UNGROUNDED_LINE)
        assert rec["marker"] == "gemini_ungrounded_suppressed"
        assert rec["model"] == "gemini-3.5-flash"
        assert rec["queries"] == 3

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(GEMINI_UNGROUNDED_LINE)
        # gemini_search.py passes question.id_of_question, not the post id.
        assert rec["qid"] == 38195
        assert rec["qid_kind"] == "question_id"

    def test_absent_qid_coerces_to_none(self):
        # qid is Optional at the call site; "None" renders into the line verbatim.
        rec = _parse_one(PFX_WARN + "GEMINI_UNGROUNDED_SUPPRESSED: question=None model=gemini-3.5-flash queries=0")
        assert rec["qid"] is None
        assert rec["queries"] == 0


# read_document's twin of the WARN above (metaculus_bot/research/agentic/tools.py): Gemini's
# url_context tool retrieved nothing, so the "fetched" tier is withheld rather than granting a
# parametric-recall answer the authority to supersede the briefing for every forecaster.
AGENTIC_DOCUMENT_UNGROUNDED_LINE = (
    PFX_WARN + "AGENTIC_DOCUMENT_UNGROUNDED_SUPPRESSED: url=https://example.com/filing.pdf"
)


class TestAgenticDocumentUngroundedSuppressed:
    def test_fields(self):
        rec = _parse_one(AGENTIC_DOCUMENT_UNGROUNDED_LINE)
        assert rec["marker"] == "agentic_document_ungrounded_suppressed"
        assert rec["url"] == "https://example.com/filing.pdf"

    def test_does_not_collide_with_the_gemini_search_marker(self):
        # Both markers end in UNGROUNDED_SUPPRESSED; each spec must claim only its own
        # line or the archive would double-count one of them.
        assert _parse_one(GEMINI_UNGROUNDED_LINE)["marker"] == "gemini_ungrounded_suppressed"
        assert _parse_one(AGENTIC_DOCUMENT_UNGROUNDED_LINE)["marker"] == "agentic_document_ungrounded_suppressed"


# Gap-fill v1 analyzer death (metaculus_bot/research/targeted.py). The analyzer gates the whole
# pass, so its failure zeroes the addendum and reads exactly like a question with no gaps —
# and gap-fill has no ProviderResult to carry a `lost=` token, so this marker is the signal.
GAP_FILL_ANALYZER_FAILED_LINE = (
    PFX_WARN + "GAP_FILL_ANALYZER_FAILED: question=44912 error=APIError detail=404 model not found"
)


class TestGapFillAnalyzerFailed:
    def test_fields(self):
        rec = _parse_one(GAP_FILL_ANALYZER_FAILED_LINE)
        assert rec["marker"] == "gap_fill_analyzer_failed"
        assert rec["error"] == "APIError"
        # detail holds the exception str, which contains spaces — it must capture to EOL.
        assert rec["detail"] == "404 model not found"

    def test_question_ref_is_a_question_id(self):
        rec = _parse_one(GAP_FILL_ANALYZER_FAILED_LINE)
        assert rec["qid"] == 44912
        assert rec["qid_kind"] == "question_id"

    def test_detail_is_optional(self):
        # Keeps older lines (and any future terser form) parseable rather than dropped.
        rec = _parse_one(PFX_WARN + "GAP_FILL_ANALYZER_FAILED: question=None error=TimeoutError")
        assert rec["marker"] == "gap_fill_analyzer_failed"
        assert rec["qid"] is None
        assert rec["error"] == "TimeoutError"
