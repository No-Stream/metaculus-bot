"""Tests for Workstream E's numeric prompt OPTION A / OPTION B branching.

The numeric_prompt must offer the LLM a binary choice:
- OPTION A — emit the trailing ``Percentile X: ...`` lines (default).
- OPTION B — emit the JSON ``mixture_components`` field.

Per user steer 2026-05-12: the LLM picks ONE. If both are emitted, the
parser uses mixture and warns. The prompt should say so explicitly.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from forecasting_tools import NumericQuestion

from metaculus_bot.prompts import numeric_prompt


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
    q.open_time = datetime.now() - timedelta(days=30)
    q.scheduled_resolution_time = datetime.now() + timedelta(days=365)
    return q


class TestNumericPromptOptionABranching:
    def test_prompt_describes_option_a_percentiles(self) -> None:
        prompt = numeric_prompt(
            _make_numeric_q(),
            research="R",
            lower_bound_message="",
            upper_bound_message="",
        )
        assert "OPTION A" in prompt
        # The default fallback is percentiles.
        assert "PERCENTILES" in prompt.upper()

    def test_prompt_describes_option_b_mixture(self) -> None:
        prompt = numeric_prompt(
            _make_numeric_q(),
            research="R",
            lower_bound_message="",
            upper_bound_message="",
        )
        assert "OPTION B" in prompt
        # OPTION B is the mixture-of-normals path; it must reference the
        # structured-block field name so the LLM knows where to emit it.
        assert "mixture_components" in prompt

    def test_prompt_keeps_trailing_percentile_example(self) -> None:
        # The trailing "Percentile 97.5:" example is the canonical Option A
        # template; Workstream E must NOT remove it.
        prompt = numeric_prompt(
            _make_numeric_q(),
            research="R",
            lower_bound_message="",
            upper_bound_message="",
        )
        assert "Percentile 97.5:" in prompt

    def test_prompt_documents_pick_one_rule(self) -> None:
        # Per user steer: "Pick one. If both are emitted, mixture wins."
        # The prompt should communicate this clearly enough that someone
        # auditing the prompt can see the rule.
        prompt = numeric_prompt(
            _make_numeric_q(),
            research="R",
            lower_bound_message="",
            upper_bound_message="",
        )
        lower = prompt.lower()
        # We accept either "pick one" / "use one" / "ignore" framing — any of
        # these makes the tie-breaker explicit.
        assert any(
            phrase in lower for phrase in ("pick one", "pick exactly one", "use one", "ignore", "mixture and ignore")
        ), "prompt must spell out the pick-one (mixture-wins) rule"

    def test_option_b_appears_after_structured_block_instruction(self) -> None:
        # The structured JSON block instruction comes from Workstream C; the
        # OUTPUT FORMAT branching block belongs near the trailing template,
        # so OPTION B should appear after the "STRUCTURED FORECAST" section
        # (because mixture_components lives in that JSON block, but the
        # branching choice is the LAST instruction the LLM reads).
        prompt = numeric_prompt(
            _make_numeric_q(),
            research="R",
            lower_bound_message="",
            upper_bound_message="",
        )
        structured_idx = prompt.find("STRUCTURED FORECAST")
        option_a_idx = prompt.find("OPTION A")
        option_b_idx = prompt.find("OPTION B")
        assert structured_idx >= 0
        assert option_a_idx >= 0
        assert option_b_idx >= 0
        assert structured_idx < option_a_idx
        assert option_a_idx < option_b_idx
