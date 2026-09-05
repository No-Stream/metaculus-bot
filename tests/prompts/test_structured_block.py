"""The STRUCTURED FORECAST block: its schema instruction, its example JSON, and the option
and percentile sets the prompts interpolate into it.

This block is the only authoritative source of a forecast, so a malformed example teaches the
model to emit something the deterministic first rung of the extraction ladder cannot read. The
example block is therefore parsed as real JSON here rather than substring-matched.
"""

import json
import re
from collections.abc import Callable

import pytest

from metaculus_bot.prompts import (
    binary_prompt,
    multiple_choice_prompt,
    numeric_prompt,
    stacking_binary_prompt,
    stacking_multiple_choice_prompt,
    stacking_numeric_prompt,
)
from tests.prompt_builders import (
    _binary_q,
    _extract_last_json_block,
    _mc_q,
    _numeric_q,
)


class TestMcPromptInterpolatesRealOptionNames:
    """Strict parsers (e.g. gemma-4-31b-it) refuse to map literal ``Option_A``
    placeholders onto real option names in the allowed-list — they correctly
    emit ``<<NOT_FOUND>>`` because the prompt example does not contain anything
    semantically tied to the question's actual options.

    Fix: the STRUCTURED FORECAST JSON block in both ``multiple_choice_prompt``
    and ``stacking_multiple_choice_prompt`` must use the REAL option names as
    JSON keys so the LLM emits text the parser can directly recognize.

    Post-refactor: the trailing prose "{opt}: NN%" answer lines are gone;
    ``option_probs`` in the JSON block is the sole forecast surface.
    """

    def test_stacking_mc_prompt_emits_real_option_names_in_json_block(self) -> None:
        q = _mc_q()
        q.options = ["Apple", "Banana", "Cherry"]

        result = stacking_multiple_choice_prompt(q, research="r", base_predictions=["a1", "a2"])

        structured_section = result[result.find("STRUCTURED FORECAST") :]
        for opt in ("Apple", "Banana", "Cherry"):
            assert f'"{opt}"' in structured_section, f"option {opt!r} missing from option_probs JSON example"
        # Trailing prose per-option lines must be gone.
        assert "Apple: NN%" not in result
        assert "Banana: NN%" not in result
        assert "Cherry: NN%" not in result

    def test_stacking_mc_prompt_drops_literal_option_a_b_placeholders(self) -> None:
        q = _mc_q()
        q.options = ["Apple", "Banana", "Cherry"]

        result = stacking_multiple_choice_prompt(q, research="r", base_predictions=["a1", "a2"])

        assert '"Option_A"' not in result
        assert '"Option_B"' not in result
        assert '"Option_N"' not in result
        # Also the old prose placeholders.
        assert "Option_A: NN%" not in result
        assert "Option_B: NN%" not in result
        assert "Option_N: NN%" not in result

    def test_multiple_choice_prompt_emits_real_option_names_in_json_block(self) -> None:
        q = _mc_q()
        q.options = ["Apple", "Banana", "Cherry"]

        result = multiple_choice_prompt(q, research="r")

        structured_section = result[result.find("STRUCTURED FORECAST") :]
        for opt in ("Apple", "Banana", "Cherry"):
            assert f'"{opt}"' in structured_section, f"option {opt!r} missing from option_probs JSON example"
        assert "Apple: NN%" not in result
        assert "Banana: NN%" not in result
        assert "Cherry: NN%" not in result

    def test_multiple_choice_prompt_drops_literal_option_a_b_placeholders(self) -> None:
        q = _mc_q()
        q.options = ["Apple", "Banana", "Cherry"]

        result = multiple_choice_prompt(q, research="r")

        # No literal "Option_A" placeholders anywhere — not in the JSON block
        # (real option names go there), not in prose (prose forecast lines gone).
        assert '"Option_A"' not in result
        assert '"Option_B"' not in result
        assert '"Option_N"' not in result
        assert "Option_A: NN%" not in result
        assert "Option_B: NN%" not in result
        assert "Option_N: NN%" not in result

    def test_stacking_mc_prompt_preserves_options_in_order_in_json_block(self) -> None:
        """The JSON-block ``option_probs`` example must list options in the same
        order as ``question.options`` — a strict parser matching on positional
        alignment depends on that ordering."""
        q = _mc_q()
        q.options = ["Manufacturing PMI higher", "Services PMI higher", "Equal"]

        result = stacking_multiple_choice_prompt(q, research="r", base_predictions=["a1", "a2"])

        idx_mfg = result.find('"Manufacturing PMI higher"')
        idx_svc = result.find('"Services PMI higher"')
        idx_eq = result.find('"Equal"')
        assert idx_mfg >= 0
        assert idx_svc >= 0
        assert idx_eq >= 0
        assert idx_mfg < idx_svc < idx_eq


class TestMcExampleBlockEscaping:
    """The MC example block is the forecaster's authoritative template, so it has to stay
    valid JSON for option names carrying quotes or backslashes — the escaping is done by
    ``_option_probs_example``, and a break there is invisible to substring assertions."""

    @pytest.mark.parametrize("options", [["Red", "Blue"], ['He said "yes"', r"C:\Windows"]])
    def test_mc_example_block_stays_valid_json(self, options: list[str]) -> None:
        q = _mc_q()
        q.options = options
        prompt = multiple_choice_prompt(q, research="r")
        body = re.findall(r"```json\s*\n(.*?)\n\s*```", prompt, re.DOTALL)[-1]
        parsed = json.loads(body)
        assert list(parsed["option_probs"].keys()) == options


class TestNumericPromptThirteenPercentiles:
    """The numeric prompts must document all 13 standard percentile keys in
    the STRUCTURED FORECAST JSON schema example (P1 first, P99 last) and
    never tell the model to emit exactly 11.

    Post-refactor: the ONLY forecast surface is ``declared_percentiles`` in the
    JSON block — the old trailing "Percentile X: [value]" prose lines are gone.
    We assert on the JSON-key form ("0.01" .. "0.99")."""

    _PERCENTILE_KEYS = (
        "0.01",
        "0.025",
        "0.05",
        "0.1",
        "0.2",
        "0.4",
        "0.5",
        "0.6",
        "0.8",
        "0.9",
        "0.95",
        "0.975",
        "0.99",
    )

    def test_numeric_prompt_json_block_has_all_thirteen_keys_in_order(self) -> None:
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        structured_section = result[result.find("STRUCTURED FORECAST") :]
        indices = []
        for key in self._PERCENTILE_KEYS:
            token = f'"{key}"'
            assert token in structured_section, f"missing percentile key {token} in declared_percentiles example"
            indices.append(structured_section.find(token))
        # Keys appear in the declared order: 0.01 < 0.025 < ... < 0.99.
        assert indices == sorted(indices), f"percentile keys out of order: {indices}"
        # No trailing prose "Percentile 1: [value]" block anywhere.
        assert "Percentile 1:" not in result
        assert "Percentile 99:" not in result

    def test_numeric_prompt_says_13_not_11(self) -> None:
        result = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        lowered = " ".join(result.lower().split())
        assert "all 13 percentiles" in lowered or "all 13 standard" in lowered
        assert "13 standard percentiles" in lowered
        assert "11 percentiles" not in lowered
        assert "11 standard percentiles" not in lowered

    def test_stacking_numeric_prompt_says_13_not_11(self) -> None:
        result = stacking_numeric_prompt(
            _numeric_q(),
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message="lbm",
            upper_bound_message="ubm",
        )
        lowered = " ".join(result.lower().split())
        assert "all 13 percentiles" in lowered or "all 13 standard" in lowered
        assert "11 percentiles" not in lowered

    def test_stacking_numeric_prompt_json_block_has_all_thirteen_keys_in_order(self) -> None:
        result = stacking_numeric_prompt(
            _numeric_q(),
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message="lbm",
            upper_bound_message="ubm",
        )
        structured_section = result[result.find("STRUCTURED FORECAST") :]
        indices = []
        for key in self._PERCENTILE_KEYS:
            token = f'"{key}"'
            assert token in structured_section, (
                f"missing percentile key {token} in stacking declared_percentiles example"
            )
            indices.append(structured_section.find(token))
        assert indices == sorted(indices), f"stacking percentile keys out of order: {indices}"
        # No trailing prose "Percentile 1: [value]" block.
        assert "Percentile 1:" not in result
        assert "Percentile 99:" not in result


class TestOptionProbsExampleJsonValidity:
    """The MC schema example is the forecaster's authoritative template — it must
    be VALID JSON for any real option names, including ones carrying quotes,
    backslashes, or newlines (F2). A naive f-string concat emitted invalid JSON
    for those and silently taught the model a broken schema."""

    @pytest.mark.parametrize(
        "options",
        [
            ['He said "yes"', r"C:\Windows", "Option C"],
            ["Line\nbreak", "Plain"],
        ],
    )
    def test_mc_prompt_block_example_parses_for_special_char_options(self, options: list[str]) -> None:
        q = _mc_q()
        q.options = options
        prompt = multiple_choice_prompt(q, research="r")
        body = _extract_last_json_block(prompt)
        parsed = json.loads(body)
        assert list(parsed["option_probs"].keys()) == options

    def test_stacking_mc_prompt_block_example_parses_for_special_char_options(self) -> None:
        options = ['He said "yes"', r"C:\Windows", "Option C"]
        q = _mc_q()
        q.options = options
        prompt = stacking_multiple_choice_prompt(q, research="r", base_predictions=["a1"])
        body = _extract_last_json_block(prompt)
        parsed = json.loads(body)
        assert list(parsed["option_probs"].keys()) == options

    def test_example_probs_valid_for_large_option_count(self) -> None:
        options = [f"Bucket {i}" for i in range(12)]
        q = _mc_q()
        q.options = options
        prompt = multiple_choice_prompt(q, research="r")
        parsed = json.loads(_extract_last_json_block(prompt))
        probs = list(parsed["option_probs"].values())
        assert sum(probs) == pytest.approx(1.0, abs=0.02)
        assert all(0.0 < p < 1.0 for p in probs)


_EXAMPLE_BLOCK_BUILDERS = [
    pytest.param(lambda: binary_prompt(_binary_q(), research="r"), id="binary"),
    pytest.param(lambda: multiple_choice_prompt(_mc_q(), research="r"), id="multiple_choice"),
    pytest.param(
        lambda: numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm"),
        id="numeric",
    ),
    pytest.param(
        lambda: stacking_binary_prompt(_binary_q(), research="r", base_predictions=["a1", "a2"]),
        id="stacking_binary",
    ),
    pytest.param(
        lambda: stacking_multiple_choice_prompt(_mc_q(), research="r", base_predictions=["a1", "a2"]),
        id="stacking_multiple_choice",
    ),
    pytest.param(
        lambda: stacking_numeric_prompt(
            _numeric_q(),
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message="lbm",
            upper_bound_message="ubm",
        ),
        id="stacking_numeric",
    ),
]


class TestStructuredForecastExampleBlocks:
    """Every builder's STRUCTURED FORECAST example must PARSE, and must not re-grow a
    key the 2026-09-02 de-bloat retired.

    These examples are static literals, so the only thing that breaks them is a source
    edit — and until this test existed most of them were guarded by substring checks only,
    which a dropped comma sails straight past. Dropping the comma before a field in the
    binary block once left the whole suite green. Parsing all six closes that gap in one
    place, and the retired-key check means re-adding a post-hoc telemetry slot to any
    example has to come through here (the reasoning behind the removals is in
    ``scratch/schema_bloat_audit_2026-09-02.md``: the block is written after the forecast
    is fixed, so a slot in it cannot scaffold reasoning).
    """

    # Retired by the schema de-bloat: each was read only by dormant telemetry, and every one
    # of them asked the model for post-hoc admin rather than for its forecast.
    _RETIRED_KEYS = (
        "remaining_window_days",
        "base_rate_anchor",
        "criteria_clauses",
        "other_mass",
        "concentration",
    )

    @pytest.mark.parametrize("build_prompt", _EXAMPLE_BLOCK_BUILDERS)
    def test_example_block_parses(self, build_prompt: Callable[[], str]) -> None:
        parsed = json.loads(_extract_last_json_block(build_prompt()))
        assert parsed["question_type"] in {"binary", "multiple_choice", "numeric"}

    @pytest.mark.parametrize("build_prompt", _EXAMPLE_BLOCK_BUILDERS)
    def test_example_block_carries_no_retired_key(self, build_prompt: Callable[[], str]) -> None:
        parsed = json.loads(_extract_last_json_block(build_prompt()))
        assert not [key for key in self._RETIRED_KEYS if key in parsed]

    def test_only_the_base_numeric_prompt_asks_for_outcome_type(self) -> None:
        """``outcome_type`` gates discrete snapping and saves a parser call, so the BASE
        numeric prompt keeps it. The stacker's vote is never read — the discrete decision is
        the base members' majority — so asking the stacker for it was pure admin, dropped
        2026-09-02."""
        base = numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")
        stacking = stacking_numeric_prompt(
            _numeric_q(),
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message="lbm",
            upper_bound_message="ubm",
        )
        assert json.loads(_extract_last_json_block(base))["outcome_type"] == "continuous"
        assert "outcome_type" not in json.loads(_extract_last_json_block(stacking))
        # The stacking prompt still DESCRIBES the field where it names what the base
        # members' own blocks carry; what went is its own schema instruction.
        stacking_schema = stacking[stacking.rfind("STRUCTURED FORECAST") :]
        assert "outcome_type" not in stacking_schema

    def test_the_thirteen_percentile_requirement_is_stated_once_per_numeric_prompt(self) -> None:
        """It was printed in the schema header line and again a few lines below in the
        Notes, in both numeric prompts. The header keeps it (it is the definition)."""
        for build in (
            lambda: numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm"),
            lambda: stacking_numeric_prompt(
                _numeric_q(),
                research="r",
                base_predictions=["a1", "a2"],
                lower_bound_message="lbm",
                upper_bound_message="ubm",
            ),
        ):
            schema_section = build()
            schema_section = schema_section[schema_section.rfind("STRUCTURED FORECAST") :]
            assert schema_section.count("MUST contain all") == 1
