"""Tests for the run-log telemetry marker parser (scripts/telemetry/markers.py).

Each example line below is copied from the format string in the emitting code
(the source of truth), so a producer-side change to a marker shape breaks these
tests loudly instead of silently dropping records from the archive:

* EXTRACTION_RUNG   -> metaculus_bot/value_extraction.py:_log_extraction
* GAP_FILL_V2       -> metaculus_bot/research/agentic/loop.py:_log_completion
* GHOST_FORECAST    -> metaculus_bot/research/agentic/loop.py:_run_ghost_phase
* OPEN_BOUND_PILING -> metaculus_bot/numeric/diagnostics.py:log_open_bound_piling_diagnostics
* CREDIT_BALANCE/SPEND/FLOOR_BREACH -> metaculus_bot/credit_telemetry.py
* STACKER_OUTCOME/TOOLS_USED/ANCHOR_OVERSHOOT_PP/CLAUSE_PRODUCT_DIVERGENCE_PP
  -> metaculus_bot/comment/markers.py (HTML-comment markers; see module docstring
     in markers.py for why they rarely appear in run logs).
"""

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
    "2026-07-17 14:25:10,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GAP_FILL_V2: model=openai/gpt-5.6-terra "
    "steps=7 tool_calls=9 searches=4 fetches=3 rendered=1 reads=2 dup_tool_calls=0 deadline_hit=False "
    "concluded_early=True wall_s=312.44 findings=5 pending_leads=1 lint_rejections=0"
)
GHOST_FORECAST_LINE = (
    "2026-07-17 14:25:11,000 - metaculus_bot.research.agentic.loop - INFO - "
    "question=https://www.metaculus.com/questions/38975/ GHOST_FORECAST: qtype=binary summary=posterior_prob=0.4200"
)
OPEN_BOUND_PILING_LINE = (
    PFX_WARN + "OPEN_BOUND_PILING: question=51000 model=gemini-3.1-pro-preview bound=upper "
    "bin_mass=0.153 declared_edge=1000 bound_value=1000"
)
CREDIT_BALANCE_LINE = PFX + "CREDIT_BALANCE: key=donated phase=start remaining=123.45 usage=4.16"
CREDIT_BALANCE_SKIP_LINE = (
    PFX_WARN + "CREDIT_BALANCE: key=personal phase=start skipped (env var OPENROUTER_API_KEY not set)"
)
CREDIT_SPEND_LINE = PFX + "CREDIT_SPEND: key=donated run_delta_usd=3.34 remaining=120.11"
CREDIT_SPEND_NA_LINE = PFX + "CREDIT_SPEND: key=personal run_delta_usd=n/a remaining=n/a"
CREDIT_FLOOR_BREACH_LINE = (
    PFX_WARN + "CREDIT_FLOOR_BREACH: key=donated remaining=45.00 floor=50.00 — donated OpenRouter "
    "balance needs a top-up; run completed normally but will exit non-zero so CI flags it."
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

    def test_line_timestamp_parsed(self):
        rec = _parse_one(EXTRACTION_RUNG_LINE)
        assert rec["line_ts"].startswith("2026-07-17T14:23:01")

    def test_run_metadata_attached(self):
        rec = _parse_one(EXTRACTION_RUNG_LINE)
        assert rec["run_id"] == "999"
        assert rec["workflow"] == "tournament"
        assert rec["artifact"] == "research-999"

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
