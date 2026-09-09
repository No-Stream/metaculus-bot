"""Tests for the block extraction and parsing helpers in ``structured_output_schema`` —
``extract_json_block`` and its candidate ranking, the two balanced-brace scanners,
``parse_structured_block`` / ``parse_structured_payload`` with their telemetry
strip-and-retry recovery, and the block size cap.

Separate from ``tests/test_structured_output_schema.py``, which covers the pydantic block
models and their validators: these tests are about how a forecaster's raw rationale text
becomes a block, not about what a block is allowed to contain.
"""

from __future__ import annotations

import json
import logging

import pytest

from metaculus_bot.structured_output_schema import (
    _MAX_STRUCTURED_BLOCK_BYTES,
    BinaryStructured,
    DiscreteCountStructured,
    MultipleChoiceStructured,
    NumericStructured,
    extract_first_balanced_braces,
    extract_json_block,
    extract_json_block_candidates,
    iter_balanced_braces,
    parse_structured_block,
    parse_structured_payload,
)
from metaculus_bot.tool_runner import _aggregate_binary_lines, _parse_all_blocks

# Why: a plugin load registers the shared fixture chain without module-level names that shadow test parameters.
pytest_plugins = ["tests.structured_block_fixtures"]

# ===========================================================================
# extract_json_block
# ===========================================================================


class TestExtractJsonBlock:
    def test_fenced_json_block_returned_trimmed(self) -> None:
        text = 'Some text\n```json\n{"question_type": "binary", "posterior_prob": 0.5}\n```\ntail'
        body = extract_json_block(text)
        assert body is not None
        assert body.startswith("{")
        assert body.endswith("}")
        assert '"question_type"' in body

    def test_returns_last_fenced_block(self) -> None:
        text = (
            "intro\n"
            '```json\n{"question_type": "binary", "posterior_prob": 0.1}\n```\n'
            "middle\n"
            '```json\n{"question_type": "binary", "posterior_prob": 0.9}\n```\n'
        )
        body = extract_json_block(text)
        assert body is not None
        assert '"posterior_prob": 0.9' in body
        assert '"posterior_prob": 0.1' not in body

    def test_no_block_returns_none(self) -> None:
        assert extract_json_block("Plain prose with no fence.") is None

    def test_empty_input_returns_none(self) -> None:
        assert extract_json_block("") is None

    def test_unclosed_fence_returns_none(self) -> None:
        text = '```json\n{"question_type": "binary"}\n'
        assert extract_json_block(text) is None

    def test_case_insensitive_json_tag(self) -> None:
        text = '```JSON\n{"question_type": "binary", "posterior_prob": 0.5}\n```'
        body = extract_json_block(text)
        assert body is not None
        assert '"posterior_prob"' in body

    def test_mixed_case_json_tag(self) -> None:
        text = '```Json\n{"question_type": "binary", "posterior_prob": 0.5}\n```'
        body = extract_json_block(text)
        assert body is not None

    def test_whitespace_around_tag(self) -> None:
        text = '```   json   \n{"question_type": "binary", "posterior_prob": 0.5}\n```'
        body = extract_json_block(text)
        assert body is not None
        assert '"question_type"' in body

    def test_untagged_fence_with_json_object_body(self) -> None:
        text = '```\n{"question_type": "binary", "posterior_prob": 0.5}\n```'
        body = extract_json_block(text)
        assert body is not None
        assert '"question_type"' in body

    def test_untagged_fence_with_non_json_body_returns_none(self) -> None:
        text = "```\nplain prose body\n```"
        assert extract_json_block(text) is None

    def test_prefers_tagged_over_untagged(self) -> None:
        """An untagged fence with JSON-like content is ignored when a tagged json block exists."""
        text = '```\n{"untagged": true}\n```\nsome text\n```json\n{"tagged": true}\n```\n'
        body = extract_json_block(text)
        assert body is not None
        assert '"tagged": true' in body
        assert "untagged" not in body

    def test_empty_body_ignored(self) -> None:
        text = "```json\n\n```"
        assert extract_json_block(text) is None


class TestExtractFirstBalancedBraces:
    """Cover the string-literal-aware balanced-brace extractor shared by
    ``_parse_gap_list`` (unfenced JSON fallback). Naive brace-counting silently
    truncates JSON that contains braces inside string values — this helper
    must not do that."""

    def test_simple_object(self) -> None:
        assert extract_first_balanced_braces('{"a": 1}') == '{"a": 1}'

    def test_returns_none_on_no_braces(self) -> None:
        assert extract_first_balanced_braces("plain prose") is None

    def test_returns_none_on_empty_input(self) -> None:
        assert extract_first_balanced_braces("") is None

    def test_object_with_prefix_and_suffix_prose(self) -> None:
        text = 'Here is the output:\n{"gap": "g"}\n\nHope that helps!'
        assert extract_first_balanced_braces(text) == '{"gap": "g"}'

    def test_brace_inside_string_value_not_counted(self) -> None:
        """The crux of F11: a naive brace counter would close the object at the ``}`` inside the string."""
        text = '{"foo": "has a } brace", "b": 1}'
        assert extract_first_balanced_braces(text) == text

    def test_opening_brace_inside_string_value_not_counted(self) -> None:
        text = '{"foo": "has a { brace", "b": 1}'
        assert extract_first_balanced_braces(text) == text

    def test_mixed_braces_in_string_values(self) -> None:
        text = '{"a": "has } and { chars", "b": 1}'
        assert extract_first_balanced_braces(text) == text

    def test_escaped_quote_inside_string(self) -> None:
        """An escaped quote does not exit the string, so the ``}`` that follows is still inside it."""
        text = '{"a": "quote \\" then } brace", "b": 1}'
        assert extract_first_balanced_braces(text) == text

    def test_escaped_backslash_then_quote_exits_string(self) -> None:
        """An escaped backslash lets the next quote exit the string, so the final ``}`` is still found."""
        text = '{"a": "trailing slash \\\\", "b": 1}'
        assert extract_first_balanced_braces(text) == text

    def test_nested_objects(self) -> None:
        text = '{"outer": {"inner": 1}, "k": "v"}'
        assert extract_first_balanced_braces(text) == text

    def test_returns_first_balanced_block_only(self) -> None:
        """A trailing second object is not part of the first balanced block."""
        text = '{"first": 1} then {"second": 2}'
        assert extract_first_balanced_braces(text) == '{"first": 1}'

    def test_unbalanced_returns_none(self) -> None:
        assert extract_first_balanced_braces('{"a": 1') is None


class TestExtractJsonBlockCandidates:
    """The candidate ranking behind validity-aware selection: tagged before
    untagged, last-by-position first within a tier, empty bodies skipped."""

    def test_empty_input_returns_empty_list(self) -> None:
        assert extract_json_block_candidates("") == []

    def test_no_fence_returns_empty_list(self) -> None:
        assert extract_json_block_candidates("plain prose, no fence") == []

    def test_tagged_ranked_last_by_position_first(self) -> None:
        text = '```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```\n'
        assert extract_json_block_candidates(text) == ['{"b": 2}', '{"a": 1}']

    def test_tagged_ranked_ahead_of_untagged(self) -> None:
        """An untagged fence appearing LATER in the text still ranks below the tagged one."""
        text = '```json\n{"tagged": 1}\n```\n```\n{"untagged": 2}\n```\n'
        assert extract_json_block_candidates(text) == ['{"tagged": 1}', '{"untagged": 2}']

    def test_empty_bodied_fence_skipped(self) -> None:
        text = '```json\n\n```\n```json\n{"a": 1}\n```\n'
        assert extract_json_block_candidates(text) == ['{"a": 1}']

    def test_extract_json_block_returns_first_candidate(self) -> None:
        text = '```json\n{"a": 1}\n```\n```json\n{"b": 2}\n```\n'
        assert extract_json_block(text) == '{"b": 2}'
        assert extract_json_block(text) == extract_json_block_candidates(text)[0]


class TestIterBalancedBraces:
    """iter_balanced_braces yields EVERY top-level balanced block; the repair
    rung iterates them so a junk leading blob doesn't block a valid later one."""

    def test_yields_multiple_top_level_blocks(self) -> None:
        text = 'junk {"first": 1} middle {"second": 2} tail'
        assert list(iter_balanced_braces(text)) == ['{"first": 1}', '{"second": 2}']

    def test_no_braces_yields_nothing(self) -> None:
        assert list(iter_balanced_braces("plain prose")) == []

    def test_stops_after_unbalanced_run(self) -> None:
        """The first blob closes and the second run is unbalanced, so nothing further is yielded."""
        text = '{"ok": 1} then {"unbalanced": '
        assert list(iter_balanced_braces(text)) == ['{"ok": 1}']

    def test_brace_inside_string_not_counted_across_blocks(self) -> None:
        text = '{"a": "has } brace"} and {"b": 2}'
        assert list(iter_balanced_braces(text)) == ['{"a": "has } brace"}', '{"b": 2}']

    def test_first_of_iter_matches_extract_first_balanced_braces(self) -> None:
        text = '{"first": 1} then {"second": 2}'
        assert next(iter_balanced_braces(text)) == extract_first_balanced_braces(text)


# ===========================================================================
# parse_structured_block
# ===========================================================================


class TestParseStructuredBlock:
    def test_valid_binary_rationale(self) -> None:
        payload = {"question_type": "binary", "posterior_prob": 0.35}
        rationale = f"My thinking...\n```json\n{json.dumps(payload)}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.35)

    def test_valid_numeric_rationale(self) -> None:
        payload = {
            "question_type": "numeric",
            "declared_percentiles": {"0.1": 1.0, "0.5": 5.0, "0.9": 9.0},
        }
        rationale = f"Analysis...\n```json\n{json.dumps(payload)}\n```"
        result = parse_structured_block(rationale, "numeric")
        assert isinstance(result, NumericStructured)
        assert result.declared_percentiles is not None
        assert result.declared_percentiles[0.5] == pytest.approx(5.0)

    def test_valid_mc_rationale(self) -> None:
        payload = {
            "question_type": "multiple_choice",
            "option_probs": {"A": 0.6, "B": 0.4},
        }
        rationale = f"```json\n{json.dumps(payload)}\n```"
        result = parse_structured_block(rationale, "multiple_choice")
        assert isinstance(result, MultipleChoiceStructured)
        assert result.option_probs == {"A": 0.6, "B": 0.4}

    def test_q45189_zero_concentration_ballot_parses(self) -> None:
        """The exact archived shape that used to fall to the LLM salvage rung.

        q45189 (2026-08-31): gemini-3.1-pro wrote ``other_mass`` and ``concentration`` both
        at 0.0 beside three option probabilities summing to 1.00. The old
        ``concentration > 0`` validator rejected the block, ``json_repair`` cannot alter
        valid JSON, and MC has no telemetry strip-and-retry, so the ballot was re-read by
        the parser LLM. Both retired fields read leniently now, so the ballot parses here
        and the unusable declarations read as absent rather than as zero.
        """
        payload = {
            "question_type": "multiple_choice",
            "option_probs": {"A": 0.5, "B": 0.3, "C": 0.2},
            "other_mass": 0.0,
            "concentration": 0.0,
        }
        rationale = f"Analysis...\n```json\n{json.dumps(payload)}\n```"
        result = parse_structured_block(rationale, "multiple_choice")
        assert isinstance(result, MultipleChoiceStructured)
        assert result.option_probs == {"A": 0.5, "B": 0.3, "C": 0.2}
        assert result.concentration is None
        assert result.other_mass == pytest.approx(0.0)

    def test_discrete_count_class_still_constructable(self) -> None:
        """Discrete-count dispatch is phase-3, but the class stays constructable for prompts and wiring."""
        d = DiscreteCountStructured(question_type="discrete_count", mean_estimate=2.0, dispersion="poisson")
        assert d.mean_estimate == pytest.approx(2.0)

    def test_no_block_returns_none_and_info_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        rationale = "Prose with no JSON block at all."
        with caplog.at_level(logging.INFO, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        # A0b: lifted from DEBUG to INFO so block-reliability is visible in run logs
        assert any(
            record.levelno == logging.INFO and "No JSON block found" in record.message for record in caplog.records
        )
        # Scoped to our own loggers: caplog.records spans every logger propagating to root.
        our_warnings = [
            r for r in caplog.records if r.levelno >= logging.WARNING and r.name.startswith("metaculus_bot")
        ]
        assert not our_warnings, [r.getMessage() for r in our_warnings]

    def test_malformed_json_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        rationale = "```json\n{this is not valid json\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        assert any("Malformed JSON" in record.message for record in caplog.records)
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_missing_required_field_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """A payload missing posterior_prob returns None and warns."""
        payload = {"question_type": "binary"}
        rationale = f"```json\n{json.dumps(payload)}\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        assert any("failed validation" in record.message for record in caplog.records)
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_question_type_mismatch_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        payload = {"question_type": "binary", "posterior_prob": 0.5}
        rationale = f"```json\n{json.dumps(payload)}\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "numeric")
        assert result is None
        assert any("question_type mismatch" in record.message for record in caplog.records)
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_missing_question_type_in_payload_injected(self) -> None:
        payload = {"posterior_prob": 0.42}
        rationale = f"```json\n{json.dumps(payload)}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.question_type == "binary"
        assert result.posterior_prob == pytest.approx(0.42)

    def test_an_explicit_null_question_type_is_a_mismatch_not_an_omission(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Only an ABSENT ``question_type`` is the caller's to fill in; ``null`` is a wrong literal like any other."""
        rationale = '```json\n{"question_type": null, "posterior_prob": 0.42}\n```'
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        assert any("question_type mismatch" in record.message for record in caplog.records)

    def test_a_repeated_key_is_refused_and_named_in_the_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """The default decoder keeps the last of two equal keys, which would publish ``0.8`` for a block that
        declared both ``0.2`` and ``0.8``; the repeated key fails the decode instead."""
        rationale = '```json\n{"question_type": "binary", "posterior_prob": 0.2, "posterior_prob": 0.8}\n```'
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        assert any("duplicate key 'posterior_prob'" in record.message for record in caplog.records)

    def test_a_boolean_posterior_is_refused_rather_than_read_as_certainty(self) -> None:
        payload = '{"question_type": "binary", "posterior_prob": true}'
        assert parse_structured_payload(payload, "binary", log_failures=False) is None
        payload = '{"question_type": "binary", "posterior_prob": 1}'
        parsed = parse_structured_payload(payload, "binary", log_failures=False)
        assert isinstance(parsed, BinaryStructured)
        assert parsed.posterior_prob == 1.0

    def test_json_array_payload_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        rationale = "```json\n[1, 2, 3]\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        assert any("must decode to a JSON object" in record.message for record in caplog.records)

    def test_roundtrip_binary(self, valid_binary_block: BinaryStructured) -> None:
        dumped = valid_binary_block.model_dump_json()
        rationale = f"Reasoning here.\n```json\n{dumped}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.model_dump() == valid_binary_block.model_dump()

    def test_roundtrip_numeric(self, valid_numeric_block: NumericStructured) -> None:
        dumped = valid_numeric_block.model_dump_json()
        rationale = f"```json\n{dumped}\n```"
        result = parse_structured_block(rationale, "numeric")
        assert isinstance(result, NumericStructured)
        assert result.model_dump() == valid_numeric_block.model_dump()

    def test_roundtrip_mc(self, valid_mc_block: MultipleChoiceStructured) -> None:
        dumped = valid_mc_block.model_dump_json()
        rationale = f"```json\n{dumped}\n```"
        result = parse_structured_block(rationale, "multiple_choice")
        assert isinstance(result, MultipleChoiceStructured)
        assert result.model_dump() == valid_mc_block.model_dump()

    def test_discrete_count_roundtrip_via_model(self, valid_discrete_block: DiscreteCountStructured) -> None:
        """Discrete-count skips parse_structured_block (phase-3), so the class must round-trip on its own."""
        dumped = valid_discrete_block.model_dump_json()
        loaded = DiscreteCountStructured.model_validate_json(dumped)
        assert loaded.model_dump() == valid_discrete_block.model_dump()


class TestValidityAwareBlockSelection:
    """Selection keeps the last block that VALIDATES, not the last by position.

    Regression: a trailing schema-recap / example block (the model echoing the
    STRUCTURED FORECAST schema after its real forecast) used to shadow a valid
    earlier block, because ``extract_json_block`` returned the last block
    unconditionally and ``parse_structured_block`` validated only that one. A
    model swap could surface this without warning, and a recap block with
    DIFFERENT numbers would publish the wrong forecast. Selection now walks all
    candidates and keeps the first that validates for the requested type.
    """

    def test_valid_block_then_malformed_trailing_block_selects_valid(self) -> None:
        """A valid forecast block followed by a malformed schema-recap selects the valid one.

        Adapted from the original repro, which used ``prediction_in_decimal``: not a
        BinaryStructured field, so both of its blocks were invalid.
        """
        text = (
            "reasoning here\n"
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            "Note, the schema looks like:\n"
            '```json\n{"question_type": "binary", "posterior_prob": <your value>}\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_valid_block_then_schema_invalid_trailing_selects_valid(self) -> None:
        """A trailing block that parses but fails validation loses to the earlier valid one."""
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            '```json\n{"question_type": "binary", "posterior_prob": 1.5}\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_valid_block_then_wrong_qtype_trailing_selects_valid(self) -> None:
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            '```json\n{"question_type": "numeric", "declared_percentiles": '
            '{"0.1": 1.0, "0.5": 5.0, "0.9": 9.0}}\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_two_valid_blocks_last_by_position_wins(self) -> None:
        """Among VALID blocks the last by position wins, because the prompt asks for the forecast last."""
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.1}\n```\n'
            '```json\n{"question_type": "binary", "posterior_prob": 0.9}\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.9)

    def test_only_malformed_block_returns_none_and_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """The honest-failure path is unchanged: no valid candidate returns None at WARNING."""
        text = "```json\n{this is not valid json\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(text, "binary")
        assert result is None
        assert any(record.levelno == logging.WARNING for record in caplog.records)

    def test_valid_untagged_recovered_when_tagged_all_invalid(self) -> None:
        """Tagged blocks are tried first, and a valid untagged fence is recovered when none validate.

        Previously any tagged block at all suppressed the untagged ones.
        """
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": <bad>}\n```\n'
            '```\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_valid_tagged_outranks_valid_untagged(self) -> None:
        """A valid tagged block beats a valid untagged one even when the untagged appears later."""
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            '```\n{"question_type": "binary", "posterior_prob": 0.9}\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_skip_then_recover_logs_info_not_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            '```json\n{"question_type": "binary", "posterior_prob": <your value>}\n```\n'
        )
        with caplog.at_level(logging.INFO, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        # A skipped-then-recovered block is an INFO signal, never a WARNING: extraction succeeded.
        assert any(record.levelno == logging.INFO and "skip" in record.message.lower() for record in caplog.records)
        # Scoped to our own loggers: caplog.records spans every logger propagating to root.
        our_warnings = [
            r for r in caplog.records if r.levelno >= logging.WARNING and r.name.startswith("metaculus_bot")
        ]
        assert not our_warnings, [r.getMessage() for r in our_warnings]

    def test_truncated_closed_final_block_skipped_for_valid(self) -> None:
        """A block truncated at the token limit with its fence still closed loses to the earlier valid one."""
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            '```json\n{"question_type": "binary", "posterior_pr\n```\n'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_truncated_unclosed_final_fence_ignored(self) -> None:
        """An unclosed final fence never matches the fence pattern, so it cannot shadow the valid block."""
        text = (
            '```json\n{"question_type": "binary", "posterior_prob": 0.42}\n```\n'
            '```json\n{"question_type": "binary", "posterior_pr'
        )
        result = parse_structured_block(text, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)

    def test_numeric_valid_then_malformed_trailing_selects_valid(self) -> None:
        text = (
            '```json\n{"question_type": "numeric", "declared_percentiles": '
            '{"0.1": 1.0, "0.5": 5.0, "0.9": 9.0}}\n```\n'
            '```json\n{"question_type": "numeric", "declared_percentiles": '
            '{"0.1": <a>, "0.5": <b>, "0.9": <c>}}\n```\n'
        )
        result = parse_structured_block(text, "numeric")
        assert isinstance(result, NumericStructured)
        assert result.declared_percentiles is not None
        assert result.declared_percentiles[0.5] == pytest.approx(5.0)

    def test_mc_valid_then_malformed_trailing_selects_valid(self) -> None:
        text = (
            '```json\n{"question_type": "multiple_choice", "option_probs": {"A": 0.6, "B": 0.4}}\n```\n'
            '```json\n{"question_type": "multiple_choice", "option_probs": {"A": <x>, "B": <y>}}\n```\n'
        )
        result = parse_structured_block(text, "multiple_choice")
        assert isinstance(result, MultipleChoiceStructured)
        assert result.option_probs == {"A": 0.6, "B": 0.4}


# ===========================================================================
# Strip-and-retry recovery for malformed binary telemetry
# ===========================================================================


class TestBinaryTelemetryStripAndRetry:
    """Strip-and-retry recovery for malformed BINARY telemetry (2026-07-08).

    Contract: a good ``posterior_prob`` (and other core fields) must survive
    a malformed ``base_rate_anchor`` / ``criteria_clauses`` value — dropping
    the entire block on a pure telemetry formatting bug would silently shift
    stacker input via the cross-model aggregation path in ``tool_runner``.

    Fail-fast on core fields is preserved: a bad ``posterior_prob`` must
    still return None even if telemetry is well-formed.
    """

    def test_criteria_clauses_null_recovers_core_block(self, caplog: pytest.LogCaptureFixture) -> None:
        """A ``criteria_clauses: null`` block warns and keeps its core binary fields.

        The prompt says to omit the key without a conjunctive breakdown, but models emit ``null``
        instead, and dropping the whole block lost the base-rate blend and the prior/posterior
        contributions with it.
        """
        rationale = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "prior": {"prob": 0.2, "source": "20yr base rate"},
                    "base_rate": {"k": 4, "n": 20, "ref_class": "past 20 yrs"},
                    "posterior_prob": 0.35,
                    "criteria_clauses": None,
                }
            )
            + "\n```"
        )
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.35)
        # Core telemetry-adjacent fields preserved.
        assert result.prior is not None
        assert result.prior.prob == pytest.approx(0.2)
        assert result.base_rate is not None
        assert result.base_rate.k == 4
        # Telemetry defaults after strip-and-retry.
        assert result.base_rate_anchor is None
        assert result.criteria_clauses == []
        # WARNING logged so this recovery is visible in run logs.
        assert any(
            "malformed telemetry fields" in rec.message and "criteria_clauses" in rec.message for rec in caplog.records
        )

    def test_reversed_anchor_recovers_core_block(self, caplog: pytest.LogCaptureFixture) -> None:
        """A reversed ``base_rate_anchor`` warns and keeps the core binary fields.

        BaseRateAnchor's ordering validator rejects ``{low: 0.6, high: 0.2}``; same recovery
        contract as ``criteria_clauses: null``.
        """
        rationale = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "posterior_prob": 0.42,
                    "base_rate_anchor": {"low": 0.6, "high": 0.2},
                }
            )
            + "\n```"
        )
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)
        assert result.base_rate_anchor is None
        assert any(
            "malformed telemetry fields" in rec.message and "base_rate_anchor" in rec.message for rec in caplog.records
        )

    def test_both_telemetry_fields_malformed_recovers(self, caplog: pytest.LogCaptureFixture) -> None:
        """Both telemetry keys present and malformed: strip both, keep the core block."""
        rationale = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "posterior_prob": 0.28,
                    "base_rate_anchor": {"low": 0.9, "high": 0.1},
                    "criteria_clauses": None,
                }
            )
            + "\n```"
        )
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.28)
        assert result.base_rate_anchor is None
        assert result.criteria_clauses == []
        message_text = " ".join(rec.message for rec in caplog.records)
        assert "base_rate_anchor" in message_text
        assert "criteria_clauses" in message_text

    def test_bad_core_field_still_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        """Strip-and-retry must not rescue a bad core field: a posterior_prob of 1.5 still drops the block."""
        rationale = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "posterior_prob": 1.5,
                    "base_rate_anchor": {"low": 0.15, "high": 0.35},
                    "criteria_clauses": [{"name": "clause", "prob": 0.5}],
                }
            )
            + "\n```"
        )
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        # Original failed-validation warning still fires; no recovery warning.
        assert any("failed validation" in rec.message for rec in caplog.records)
        assert not any("malformed telemetry fields" in rec.message for rec in caplog.records)

    def test_bad_core_and_bad_telemetry_still_none(self) -> None:
        """Bad core plus bad telemetry: neither retry variant validates, so the result is None."""
        rationale = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "posterior_prob": 1.5,  # out of range
                    "base_rate_anchor": {"low": 0.9, "high": 0.1},  # reversed
                }
            )
            + "\n```"
        )
        result = parse_structured_block(rationale, "binary")
        assert result is None

    def test_recovered_block_feeds_cross_model_aggregation(self) -> None:
        """A forecaster whose only validation error is malformed telemetry still contributes its base rate.

        Calls ``_parse_all_blocks`` and ``_aggregate_binary_lines`` directly so the test does not
        depend on feature-flag env state.
        """
        good = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "base_rate": {"k": 2, "n": 10, "ref_class": "ref"},
                    "posterior_prob": 0.25,
                }
            )
            + "\n```"
        )
        # Previously dropped entirely on the null criteria_clauses; strip-and-retry keeps it.
        recovered = (
            "```json\n"
            + json.dumps(
                {
                    "question_type": "binary",
                    "base_rate": {"k": 6, "n": 10, "ref_class": "ref"},
                    "posterior_prob": 0.55,
                    "criteria_clauses": None,
                }
            )
            + "\n```"
        )
        blocks = _parse_all_blocks([good, recovered], "binary")
        assert len(blocks) == 2  # both survive — invariant restored
        lines = _aggregate_binary_lines([0.25, 0.55], [b for b in blocks if isinstance(b, BinaryStructured)])
        blend_line = next((line for line in lines if "Blended base rate" in line), None)
        assert blend_line is not None
        # Blend should reflect BOTH forecasters (n=2), not just the well-formed one.
        assert "2 forecasters" in blend_line


# ===========================================================================
# Realistic rationale fixture
# ===========================================================================


REALISTIC_BINARY_RATIONALE = """\
Question: Will Country X's inflation rate exceed 5% by year-end?

Relevant base rate: Over the past 20 years, Country X has had inflation above 5%
in 4 out of 20 full years, giving a rough prior of 20%.

Recent signals:
- The central bank raised rates by 75bp in the last two meetings, which pushes DOWN
- Food prices (30% of CPI basket) have surged 8% YoY, which pushes UP
- Wage growth accelerating to 6% nominal, pushing UP
- Energy subsidies extended through Q4, pushing DOWN

Weighting the evidence, I think this is above the base rate but below 50%. The
rate hikes are lagging; inflation pressure is real but partially offset by policy.

Probability: 35%

```json
{
    "question_type": "binary",
    "prior": {"prob": 0.20, "source": "20-year base rate for Country X"},
    "base_rate": {"k": 4, "n": 20, "ref_class": "past 20 annual CPI readings"},
    "posterior_prob": 0.35
}
```
"""


class TestRealisticRationale:
    def test_extract_picks_json_block(self) -> None:
        body = extract_json_block(REALISTIC_BINARY_RATIONALE)
        assert body is not None
        assert '"question_type": "binary"' in body
        # The "Probability: 35%" line should NOT leak into the extracted body.
        assert "Probability: 35%" not in body
        # Body should be parseable as JSON.
        parsed = json.loads(body)
        assert parsed["posterior_prob"] == pytest.approx(0.35)

    def test_parse_structured_block_from_realistic_rationale(self) -> None:
        result = parse_structured_block(REALISTIC_BINARY_RATIONALE, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.35)
        assert result.prior is not None
        assert result.prior.prob == pytest.approx(0.20)
        assert result.base_rate is not None
        assert result.base_rate.k == 4
        assert result.base_rate.n == 20


# ===========================================================================
# Schema robustness — deep nesting, size cap, unicode, fence edge cases
# ===========================================================================


class TestSchemaRobustness:
    def test_deeply_nested_json_parses_without_crash(self) -> None:
        """The extractor handles deeply nested JSON without blowing up Python's parser.

        The nested dict rides on an unknown key, which ``extra="forbid"`` would reject, so only
        extraction and the preserved depth are under test.
        """
        nested: dict[str, object] = {"leaf": 1}
        for _ in range(100):
            nested = {"next": nested}
        outer = {"question_type": "binary", "posterior_prob": 0.3, "nested_payload": nested}
        rationale = f"```json\n{json.dumps(outer)}\n```"
        body = extract_json_block(rationale)
        assert body is not None
        parsed = json.loads(body)
        cursor = parsed["nested_payload"]
        depth = 0
        while isinstance(cursor, dict) and "next" in cursor:
            cursor = cursor["next"]
            depth += 1
        assert depth == 100

    def test_size_cap_rejects_huge_well_formed_block(self, caplog: pytest.LogCaptureFixture) -> None:
        """The size cap fires before pydantic validation on a payload padded past it.

        The padding rides on an unknown field, which ``extra="forbid"`` would reject, so a size-cap
        warning proves the cap ran first.
        """
        huge_body = {"question_type": "binary", "posterior_prob": 0.5, "padding": "x" * 250_000}
        rationale = f"```json\n{json.dumps(huge_body)}\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        assert any("size cap" in rec.message for rec in caplog.records)

    def test_unicode_field_values_parse(self) -> None:
        """Non-ASCII characters in string values round-trip through the parser."""
        payload = {
            "question_type": "binary",
            "prior": {"prob": 0.3, "source": "日本の基準"},
            "posterior_prob": 0.4,
        }
        rationale = f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.prior is not None
        assert result.prior.source == "日本の基準"

    def test_unicode_emoji_option_keys(self) -> None:
        payload = {
            "question_type": "multiple_choice",
            "option_probs": {"🔴 red": 0.5, "🔵 blue": 0.5},
        }
        rationale = f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
        result = parse_structured_block(rationale, "multiple_choice")
        assert isinstance(result, MultipleChoiceStructured)
        assert "🔴 red" in result.option_probs
        assert "🔵 blue" in result.option_probs

    def test_multiple_fenced_blocks_last_wins(self) -> None:
        """Two valid blocks with different posteriors: the extractor returns the last one."""
        first = {"question_type": "binary", "posterior_prob": 0.1}
        last = {"question_type": "binary", "posterior_prob": 0.9}
        rationale = f"Draft:\n```json\n{json.dumps(first)}\n```\nRevision:\n```json\n{json.dumps(last)}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.9)

    def test_untagged_fence_with_json_body_parses(self) -> None:
        """An untagged fence whose body starts with ``{`` still matches, as the fallback to tagged."""
        payload = {"question_type": "binary", "posterior_prob": 0.42}
        rationale = f"```\n{json.dumps(payload)}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)
        assert result.posterior_prob == pytest.approx(0.42)


class TestSizeCapBoundary:
    """Boundary coverage for the _MAX_STRUCTURED_BLOCK_BYTES guard."""

    def _padded_binary_payload(self, padding_size: int) -> str:
        """Pad ``ref_class``, which has no max_length, to hit the size cap without tripping extra="forbid"."""
        payload = {
            "question_type": "binary",
            "posterior_prob": 0.5,
            "base_rate": {"k": 1, "n": 10, "ref_class": "x" * padding_size},
        }
        return json.dumps(payload)

    def test_just_below_cap_parses_ok(self) -> None:
        """Pad to land just under the cap, leaving about 1KB of slack for JSON overhead."""
        padding = _MAX_STRUCTURED_BLOCK_BYTES - 1000
        raw = self._padded_binary_payload(padding)
        assert len(raw) < _MAX_STRUCTURED_BLOCK_BYTES
        rationale = f"```json\n{raw}\n```"
        result = parse_structured_block(rationale, "binary")
        assert isinstance(result, BinaryStructured)

    def test_just_over_cap_rejected(self, caplog: pytest.LogCaptureFixture) -> None:
        padding = _MAX_STRUCTURED_BLOCK_BYTES + 100
        raw = self._padded_binary_payload(padding)
        assert len(raw) > _MAX_STRUCTURED_BLOCK_BYTES
        rationale = f"```json\n{raw}\n```"
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.structured_output_schema"):
            result = parse_structured_block(rationale, "binary")
        assert result is None
        assert any("size cap" in rec.message for rec in caplog.records)
