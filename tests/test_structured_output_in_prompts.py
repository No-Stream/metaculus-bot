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

    def test_schema_block_is_last_forecast_surface(self):
        # Critical ordering constraint (post-block-only refactor): the JSON
        # block is the LAST forecast surface — the old "Probability: ZZ%" prose
        # line is gone. Only the schema block carries a machine-readable
        # forecast. The prompt closes with the "write nothing after it"
        # instruction attached to the block section.
        prompt = binary_prompt(_make_binary_q(), research="R")
        assert '"question_type"' in prompt, "schema block missing"
        assert '"Probability: ZZ%"' not in prompt, "trailing prose answer line must be gone"
        assert "Probability: ZZ%" not in prompt, "trailing prose answer line must be gone"
        # The final section header before the tail should be STRUCTURED FORECAST.
        assert "STRUCTURED FORECAST" in prompt
        assert prompt.rstrip().endswith("Write nothing after it."), "prompt must end with the block-is-last instruction"

    def test_retired_telemetry_fields_are_not_asked_for(self):
        """The anchor + clause telemetry slots (prompted 2026-07-08, retired 2026-09-02)
        must stay out of the binary block. Both only re-keyed prose the template already
        forces — the Phase 1 base rate and the step-5b clause table — and their only
        reader was telemetry behind ``PROBABILISTIC_TOOLS_ENABLED``, off in every prod
        workflow. The schema keeps the fields tolerant for archived comments, so nothing
        but this test stops the prompt half coming back."""
        prompt = binary_prompt(_make_binary_q(), research="R")
        structured_section = prompt[prompt.find("STRUCTURED FORECAST") :]
        assert '"base_rate_anchor"' not in structured_section
        assert '"criteria_clauses"' not in structured_section
        assert "outside-view base-rate range" not in structured_section
        assert "conjunctive criteria pricing table" not in structured_section.lower()


class TestMultipleChoicePromptSchemaInstruction:
    def test_contains_mc_schema_fields(self):
        prompt = multiple_choice_prompt(_make_mc_q(), research="R")
        assert "STRUCTURED FORECAST" in prompt
        assert '"multiple_choice"' in prompt
        assert "option_probs" in prompt

    def test_retired_dirichlet_inputs_are_not_asked_for(self):
        """``other_mass`` and ``concentration`` were inputs to a Dirichlet tool that has
        been dormant behind ``PROBABILISTIC_TOOLS_ENABLED`` since it shipped, and neither
        improves a ballot: the option set is exhaustive so ``option_probs`` sums to 1
        regardless, and 7 of 19 archived ``concentration`` fills just echoed the example's
        20.0. Retired 2026-09-02 — asking for ``concentration`` also cost q45189 a rung-1
        parse. The schema keeps both tolerant for archived comments."""
        prompt = multiple_choice_prompt(_make_mc_q(), research="R")
        structured_section = prompt[prompt.find("STRUCTURED FORECAST") :]
        assert "other_mass" not in structured_section
        assert "concentration" not in structured_section

    def test_tier2_fields_not_demanded_in_mc_schema(self):
        """Tier-2 scaffold fields are no longer demanded in the MC JSON schema block (C2)."""
        prompt = multiple_choice_prompt(_make_mc_q(), research="R")
        structured_section = prompt[prompt.find("STRUCTURED FORECAST") :]
        for field in ("prior", "base_rate", "hazard", "evidence", "scenarios"):
            assert f'"{field}"' not in structured_section, (
                f"tier-2 field {field!r} should not be demanded in the MC STRUCTURED FORECAST schema block"
            )

    def test_schema_block_is_last_forecast_surface(self):
        # Post-refactor: the trailing "Red: NN%" prose lines are gone. The
        # JSON block's `option_probs` (keyed by real option names) is the
        # only per-option forecast surface, and the block closes the prompt.
        prompt = multiple_choice_prompt(_make_mc_q(), research="R")
        assert '"question_type"' in prompt, "schema block missing"
        assert "Red: NN%" not in prompt, "trailing prose per-option lines must be gone"
        assert "Blue: NN%" not in prompt
        assert "Green: NN%" not in prompt
        # Real option names must still appear inside the JSON block itself.
        structured_section = prompt[prompt.find("STRUCTURED FORECAST") :]
        for opt in ("Red", "Blue", "Green"):
            assert f'"{opt}"' in structured_section, f"option {opt!r} missing from option_probs JSON example"
        assert prompt.rstrip().endswith("Write nothing after it."), "prompt must end with the block-is-last instruction"


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

    def test_numeric_prompt_declares_block_as_only_forecast_source(self):
        # Post-refactor: no trailing "Percentile X: ..." prose lines exist —
        # the declared_percentiles block is the ONLY forecast surface. The
        # numeric prompt says so explicitly so the model doesn't split its
        # forecast across two places.
        prompt = numeric_prompt(_make_numeric_q(), research="R", lower_bound_message="", upper_bound_message="")
        structured_section = prompt[prompt.find("STRUCTURED FORECAST") :]
        lowered = structured_section.lower()
        assert "only source of your forecast" in lowered or "only authoritative source" in lowered, (
            "numeric prompt should state that the JSON block is the sole forecast surface"
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

    def test_schema_block_is_last_forecast_surface(self):
        # Post-refactor: the trailing "Percentile X: [value]" prose lines are
        # gone. The declared_percentiles JSON block is the only forecast
        # surface, and the prompt closes with the block-is-last instruction.
        prompt = numeric_prompt(_make_numeric_q(), research="R", lower_bound_message="", upper_bound_message="")
        assert '"question_type"' in prompt, "schema block missing"
        assert "Percentile 97.5:" not in prompt, "trailing Percentile prose lines must be gone"
        assert "Percentile 1:" not in prompt, "trailing Percentile prose lines must be gone"
        assert "Percentile 99:" not in prompt, "trailing Percentile prose lines must be gone"
        assert "STRUCTURED FORECAST" in prompt
        assert prompt.rstrip().endswith("Write nothing after it."), "prompt must end with the block-is-last instruction"
