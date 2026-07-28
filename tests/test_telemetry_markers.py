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
* CREDIT_BALANCE/SPEND/FLOOR_BREACH -> metaculus_bot/credit_telemetry.py
* STACKER_OUTCOME/TOOLS_USED/ANCHOR_OVERSHOOT_PP/CLAUSE_PRODUCT_DIVERGENCE_PP
  -> metaculus_bot/comment/markers.py (HTML-comment markers; see module docstring
     in markers.py for why they rarely appear in run logs).
"""

import json

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
CREDIT_BALANCE_LINE = PFX + "CREDIT_BALANCE: key=donated phase=start remaining=123.45 usage=4.16"
CREDIT_BALANCE_SKIP_LINE = (
    PFX_WARN + "CREDIT_BALANCE: key=personal phase=start skipped (env var OPENROUTER_API_KEY not set)"
)
CREDIT_SPEND_LINE = PFX + "CREDIT_SPEND: key=donated run_delta_usd=3.34 remaining=120.11"
CREDIT_SPEND_NA_LINE = PFX + "CREDIT_SPEND: key=personal run_delta_usd=n/a remaining=n/a"
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
# (metaculus_bot/forecaster.py:_emit_forecaster_drop_telemetry) — the source of
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
# format string in metaculus_bot/forecaster.py (forecast_questions), the source of
# truth. Without a spec here the archive held no record of the counter that reddens
# every run: the 2026-07-26 research_provider_timeouts -> research_provider_failures
# rename would have been invisible to a replay.
DEGRADATION_COUNTERS_LINE = (
    PFX + "Degradation counters: forecasters_dropped=2, questions_failed_to_publish=0, "
    "stacker_primary_failed=0, stacker_fallback_used=0, stacker_fallback_failed=0, "
    "research_provider_failures=1, summarizer_failures=3, gap_fill_v2_errors=0, "
    "prediction_market_degraded=0, prediction_market_source_losses=4"
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
    def test_all_ten_current_keys_parse(self):
        rec = _parse_one(DEGRADATION_COUNTERS_LINE)
        assert rec["marker"] == "degradation_counters"
        assert rec["forecasters_dropped"] == 2
        assert rec["questions_failed_to_publish"] == 0
        assert rec["stacker_primary_failed"] == 0
        assert rec["stacker_fallback_used"] == 0
        assert rec["stacker_fallback_failed"] == 0
        assert rec["research_provider_failures"] == 1
        assert rec["summarizer_failures"] == 3
        assert rec["gap_fill_v2_errors"] == 0
        assert rec["prediction_market_degraded"] == 0
        assert rec["prediction_market_source_losses"] == 4

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
        # Keys that did not exist pre-rename coerce to None, not 0 — absent must not
        # read as "measured zero" in the archive.
        assert rec["research_provider_failures"] is None
        assert rec["summarizer_failures"] is None
        assert rec["prediction_market_source_losses"] is None


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
