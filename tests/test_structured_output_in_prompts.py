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
    q = MagicMock(spec=NumericQuestion)
    q.question_text = "What will X be?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.unit_of_measure = "USD"
    q.lower_bound = 0.0
    q.upper_bound = 100.0
    q.open_lower_bound = False
    q.open_upper_bound = False
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
        # At least one optional field documented
        for field in ("prior", "base_rate", "hazard", "evidence", "scenarios"):
            assert field in prompt, f"missing optional field {field!r} in binary schema"

    def test_schema_block_precedes_answer_line(self):
        # Critical ordering constraint: JSON block must appear BEFORE the
        # final "Probability: ZZ%" line so the parser picks the right text.
        prompt = binary_prompt(_make_binary_q(), research="R")
        schema_idx = prompt.find('"question_type"')
        answer_idx = prompt.find('"Probability: ZZ%"')
        assert schema_idx >= 0, "schema block missing"
        assert answer_idx >= 0, "answer line missing"
        assert schema_idx < answer_idx, "JSON schema must come before the final answer line (Option A ordering)"


class TestMultipleChoicePromptSchemaInstruction:
    def test_contains_mc_schema_fields(self):
        prompt = multiple_choice_prompt(_make_mc_q(), research="R")
        assert "STRUCTURED FORECAST" in prompt
        assert '"multiple_choice"' in prompt
        assert "option_probs" in prompt
        for field in ("other_mass", "concentration"):
            assert field in prompt, f"missing optional field {field!r} in MC schema"

    def test_schema_block_precedes_option_answer_lines(self):
        prompt = multiple_choice_prompt(_make_mc_q(), research="R")
        schema_idx = prompt.find('"question_type"')
        answer_idx = prompt.find("Option_A: NN%")
        assert schema_idx >= 0
        assert answer_idx >= 0
        assert schema_idx < answer_idx, "JSON schema must come before final Option_A: NN% lines (Option A ordering)"


class TestNumericPromptSchemaInstruction:
    def test_contains_numeric_schema_fields(self):
        prompt = numeric_prompt(_make_numeric_q(), research="R", lower_bound_message="", upper_bound_message="")
        assert "STRUCTURED FORECAST" in prompt
        assert '"numeric"' in prompt
        assert "declared_percentiles" in prompt
        for field in ("distribution_family_hint", "student_t_df", "tails", "scenarios"):
            assert field in prompt, f"missing optional field {field!r} in numeric schema"

    def test_numeric_percentiles_match_trailing_lines_note_present(self):
        # Reminder that JSON declared_percentiles should reflect the trailing
        # "Percentile X: ..." lines — see activation doc §253-261.
        prompt = numeric_prompt(_make_numeric_q(), research="R", lower_bound_message="", upper_bound_message="")
        assert "match your final Percentile" in prompt or "match your final percentile" in prompt.lower(), (
            "numeric prompt should note that JSON percentiles should match the trailing Percentile lines"
        )

    def test_mixture_components_not_in_numeric_schema_v1(self):
        # Workstream D will add mixture_components. In Workstream C we must
        # NOT include it yet (schema extension is out of scope). A TODO comment
        # flagging future mixture support is expected.
        prompt = numeric_prompt(_make_numeric_q(), research="R", lower_bound_message="", upper_bound_message="")
        # mixture_components as a schema field must not appear.
        assert '"mixture_components"' not in prompt, (
            "mixture_components is a Workstream D addition — must not ship in Workstream C"
        )

    def test_schema_block_precedes_percentile_answer_lines(self):
        prompt = numeric_prompt(_make_numeric_q(), research="R", lower_bound_message="", upper_bound_message="")
        schema_idx = prompt.find('"question_type"')
        # Last percentile example — must come after the JSON block.
        answer_idx = prompt.find("Percentile 97.5:")
        assert schema_idx >= 0
        assert answer_idx >= 0
        assert schema_idx < answer_idx, "JSON schema must come before the final Percentile lines (Option A ordering)"
