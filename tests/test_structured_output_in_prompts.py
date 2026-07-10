"""Golden-ish checks that forecaster prompts carry the structured-block schema instruction.

Each forecaster prompt (binary / MC / numeric) must instruct the model to
emit a fenced ``json`` block describing its final forecast in
machine-readable form, BEFORE the trailing answer line (Option A ordering
from scratch_docs_and_planning/probabilistic_tools_activation.md §56-89).

The schema fields checked per question type are the ones tool_runner.py
actually consumes — adding a new optional field here should not break the
test, but removing a required field (posterior_prob / declared_percentiles
/ option_probs) should.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from metaculus_bot.prompts import binary_prompt, multiple_choice_prompt, numeric_prompt
from tests.conftest import make_mock_numeric_question

# All three prompts call _forecasting_window_str(question) which asserts on
# open_time / scheduled_resolution_time. Populate in every fixture.
_OPEN = datetime.now() - timedelta(days=30)
_RESOLVE = datetime.now() + timedelta(days=365)


def _make_binary_q() -> BinaryQuestion:
    q = MagicMock(spec=BinaryQuestion)
    q.question_text = "Will X happen?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


def _make_mc_q() -> MultipleChoiceQuestion:
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.question_text = "Which color?"
    q.options = ["Red", "Blue", "Green"]
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


def _make_numeric_q() -> NumericQuestion:
    q = make_mock_numeric_question()
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


class TestBinaryPromptSchemaInstruction:
    def test_contains_structured_forecast_header(self):
        prompt = binary_prompt(_make_binary_q(), research="R")
        assert "STRUCTURED FORECAST" in prompt

    def test_contains_json_fence_and_binary_schema_fields(self):
        prompt = binary_prompt(_make_binary_q(), research="R")
        assert '"question_type"' in prompt
        assert '"binary"' in prompt
        # Required output field
        assert "posterior_prob" in prompt

    def test_tier2_fields_not_demanded_in_schema(self):
        """Tier-2 scaffold fields are no longer demanded in the JSON schema block (C2)."""
        prompt = binary_prompt(_make_binary_q(), research="R")
        # The schema example in the JSON block should NOT contain tier-2 fields.
        # They may still appear elsewhere in the analysis template (e.g. "base rate"
        # as a prose concept), so we check only the STRUCTURED FORECAST section.
        structured_section = prompt[prompt.find("STRUCTURED FORECAST") :]
        for field in ("prior", "base_rate", "hazard", "evidence", "scenarios"):
            assert f'"{field}"' not in structured_section, (
                f"tier-2 field {field!r} should not be demanded in the STRUCTURED FORECAST schema block"
            )

    def test_schema_block_precedes_answer_line(self):
        # Critical ordering constraint: JSON block must appear BEFORE the
        # final "Probability: ZZ%" line so the parser picks the right text.
        prompt = binary_prompt(_make_binary_q(), research="R")
        schema_idx = prompt.find('"question_type"')
        answer_idx = prompt.find('"Probability: ZZ%"')
        assert schema_idx >= 0, "schema block missing"
        assert answer_idx >= 0, "answer line missing"
        assert schema_idx < answer_idx, "JSON schema must come before the final answer line (Option A ordering)"

    def test_telemetry_fields_documented_in_binary_schema(self):
        """Anchor + clause telemetry fields (2026-07-08) are shown in the schema
        example AND carry fill instructions, so forecasters populate them."""
        prompt = binary_prompt(_make_binary_q(), research="R")
        structured_section = prompt[prompt.find("STRUCTURED FORECAST") :]
        assert '"base_rate_anchor"' in structured_section
        assert '"criteria_clauses"' in structured_section
        # Fill instructions reference where the values come from.
        assert "outside-view base-rate range" in structured_section
        assert "conjunctive criteria pricing table" in structured_section.lower()


class TestMultipleChoicePromptSchemaInstruction:
    def test_contains_mc_schema_fields(self):
        prompt = multiple_choice_prompt(_make_mc_q(), research="R")
        assert "STRUCTURED FORECAST" in prompt
        assert '"multiple_choice"' in prompt
        assert "option_probs" in prompt
        # Optional tier-1 fields still shown in the example
        for field in ("other_mass", "concentration"):
            assert field in prompt, f"missing optional field {field!r} in MC schema"

    def test_tier2_fields_not_demanded_in_mc_schema(self):
        """Tier-2 scaffold fields are no longer demanded in the MC JSON schema block (C2)."""
        prompt = multiple_choice_prompt(_make_mc_q(), research="R")
        structured_section = prompt[prompt.find("STRUCTURED FORECAST") :]
        for field in ("prior", "base_rate", "hazard", "evidence", "scenarios"):
            assert f'"{field}"' not in structured_section, (
                f"tier-2 field {field!r} should not be demanded in the MC STRUCTURED FORECAST schema block"
            )

    def test_schema_block_precedes_option_answer_lines(self):
        prompt = multiple_choice_prompt(_make_mc_q(), research="R")
        schema_idx = prompt.find('"question_type"')
        # The trailing answer block now interpolates real option names rather
        # than literal Option_A placeholders, so we anchor on the first real
        # option from the test fixture (`_make_mc_q` uses ["Red","Blue","Green"]).
        answer_idx = prompt.find("Red: NN%")
        assert schema_idx >= 0
        assert answer_idx >= 0
        assert schema_idx < answer_idx, "JSON schema must come before final per-option answer lines (Option A ordering)"


class TestNumericPromptSchemaInstruction:
    def test_contains_numeric_schema_fields(self):
        prompt = numeric_prompt(_make_numeric_q(), research="R", lower_bound_message="", upper_bound_message="")
        assert "STRUCTURED FORECAST" in prompt
        assert '"numeric"' in prompt
        assert "declared_percentiles" in prompt

    def test_outcome_type_in_numeric_schema(self):
        """C3: outcome_type field is documented in the numeric JSON schema block."""
        prompt = numeric_prompt(_make_numeric_q(), research="R", lower_bound_message="", upper_bound_message="")
        structured_section = prompt[prompt.find("STRUCTURED FORECAST") :]
        assert "outcome_type" in structured_section
        assert "discrete_integer" in structured_section
        assert "continuous" in structured_section

    def test_tier2_fields_not_demanded_in_numeric_schema(self):
        """Tier-2 scaffold fields are no longer demanded in the numeric JSON schema block (C2)."""
        prompt = numeric_prompt(_make_numeric_q(), research="R", lower_bound_message="", upper_bound_message="")
        structured_section = prompt[prompt.find("STRUCTURED FORECAST") :]
        for field in ("prior", "base_rate", "hazard", "evidence", "scenarios"):
            assert f'"{field}"' not in structured_section, (
                f"tier-2 field {field!r} should not be demanded in the numeric STRUCTURED FORECAST schema block"
            )

    def test_numeric_percentiles_match_trailing_lines_note_present(self):
        # Reminder that JSON declared_percentiles should reflect the trailing
        # "Percentile X: ..." lines — see activation doc §253-261.
        prompt = numeric_prompt(_make_numeric_q(), research="R", lower_bound_message="", upper_bound_message="")
        assert "match your final Percentile" in prompt or "match your final percentile" in prompt.lower(), (
            "numeric prompt should note that JSON percentiles should match the trailing Percentile lines"
        )

    def test_numeric_schema_example_carries_all_thirteen_percentile_keys(self):
        """The declared_percentiles example must show all 13 standard percentiles as
        fractional keys ("0.01".."0.99" — the format the F5 fallback lifts into
        Percentile objects). A 3-key block is unsalvageable in exactly the
        parser-miss scenario the fallback exists for: sanitize_percentiles
        hard-requires the full 13-set."""
        prompt = numeric_prompt(_make_numeric_q(), research="R", lower_bound_message="", upper_bound_message="")
        structured_section = prompt[prompt.find("STRUCTURED FORECAST") :]
        for key in ("0.01", "0.025", "0.05", "0.1", "0.2", "0.4", "0.5", "0.6", "0.8", "0.9", "0.95", "0.975", "0.99"):
            assert f'"{key}"' in structured_section, f"missing percentile key {key!r} in declared_percentiles example"
        assert "all 13" in structured_section, "note must state the full 13-percentile requirement"
        assert "at least" not in structured_section, "stale 'at least {0.1, 0.5, 0.9}' wording must be gone"

    def test_schema_block_precedes_percentile_answer_lines(self):
        prompt = numeric_prompt(_make_numeric_q(), research="R", lower_bound_message="", upper_bound_message="")
        schema_idx = prompt.find('"question_type"')
        # Last percentile example — must come after the JSON block.
        answer_idx = prompt.find("Percentile 97.5:")
        assert schema_idx >= 0
        assert answer_idx >= 0
        assert schema_idx < answer_idx, "JSON schema must come before the final Percentile lines (Option A ordering)"
