"""Tests for metaculus_bot.performance_analysis.parsing.

The parser maps `*Forecaster N*: value` bullets in the summary section of a
Metaculus bot comment to the model that actually produced each forecast.
Ground truth for the index->model mapping comes from `Model: openrouter/...`
lines inside the `## R1: Forecaster N Reasoning` blocks further down in the
comment, injected by main.py when it wraps each prediction's reasoning.

Attribution must never silently mislabel — if the Model: lines are missing
or malformed, the parser should return an anonymized fallback (Forecaster N
keys), never guess from a configured roster.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from metaculus_bot.comment_markers import STACKED_BASE_REASONING_HEADER, STACKER_META_ANALYSIS_HEADER
from metaculus_bot.performance_analysis.parsing import (
    _iter_per_model_blocks,
    annotate_forecaster_bullets_with_models,
    extract_model_display_name_from_reasoning,
    parse_forecaster_model_map,
    parse_per_model_forecasts,
    parse_per_model_numeric_percentiles,
    parse_per_model_reasoning_text,
    parse_resolution,
    parse_stacked_marker,
)
from metaculus_bot.stacking import combine_stacker_and_base_reasoning

# ---------------------------------------------------------------------------
# parse_stacked_marker
# ---------------------------------------------------------------------------


class TestParseStackedMarker:
    def test_true_marker(self):
        assert parse_stacked_marker("# SUMMARY\n...\n<!-- STACKED=true -->\n") is True

    def test_false_marker(self):
        assert parse_stacked_marker("# SUMMARY\n...\n<!-- STACKED=false -->\n") is False

    def test_absent_marker_returns_none(self):
        assert parse_stacked_marker("# SUMMARY\nNo marker here\n") is None

    def test_case_insensitive(self):
        assert parse_stacked_marker("<!-- stacked=TRUE -->") is True
        assert parse_stacked_marker("<!-- STACKED=FALSE -->") is False

    def test_whitespace_tolerant(self):
        assert parse_stacked_marker("<!--   STACKED=true   -->") is True


# ---------------------------------------------------------------------------
# parse_per_model_numeric_percentiles
# ---------------------------------------------------------------------------


class TestParsePerModelNumericPercentiles:
    def test_extracts_percentile_blocks(self):
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "analysis...\n\n"
            "Percentile 2.5: 100.0\n"
            "Percentile 50: 150.0\n"
            "Percentile 97.5: 200.0\n\n"
            "## R1: Forecaster 2 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "more analysis...\n\n"
            "Percentile 2.5: 110.0\n"
            "Percentile 50: 160.0\n"
            "Percentile 97.5: 210.0\n"
        )
        result = parse_per_model_numeric_percentiles(comment)
        assert result["gpt-5.5"] == [(2.5, 100.0), (50.0, 150.0), (97.5, 200.0)]
        assert result["claude-opus-4.7"] == [(2.5, 110.0), (50.0, 160.0), (97.5, 210.0)]

    def test_section_without_percentiles_skipped(self):
        # Stacker meta-block with no distribution should not appear in result
        comment = (
            "## R1: Forecaster 1 Reasoning\nModel: openrouter/openai/gpt-5.5\n\nJust text reasoning, no distribution.\n"
        )
        assert parse_per_model_numeric_percentiles(comment) == {}

    def test_missing_model_line_uses_anonymized_key(self):
        comment = "## R1: Forecaster 1 Reasoning\nNo Model line.\n\nPercentile 50: 42.0\n"
        assert parse_per_model_numeric_percentiles(comment) == {"Forecaster 1": [(50.0, 42.0)]}

    def test_empty_comment(self):
        assert parse_per_model_numeric_percentiles("") == {}


# ---------------------------------------------------------------------------
# Forward-fix: inline model name in summary bullets
# ---------------------------------------------------------------------------


class TestExtractModelDisplayNameFromReasoning:
    def test_happy_path(self):
        reasoning = "Model: openrouter/openai/gpt-5.5\n\nActual reasoning text."
        assert extract_model_display_name_from_reasoning(reasoning) == "gpt-5.5"

    def test_whitespace_variations(self):
        reasoning = "Model:   openrouter/anthropic/claude-opus-4.7  \n\nreasoning."
        assert extract_model_display_name_from_reasoning(reasoning) == "claude-opus-4.7"

    def test_no_prefix_returns_none(self):
        assert extract_model_display_name_from_reasoning("Just reasoning, no prefix.") is None

    def test_empty_value_returns_none(self):
        assert extract_model_display_name_from_reasoning("Model:   \n\nreasoning.") is None

    def test_only_prefix(self):
        assert extract_model_display_name_from_reasoning("Model: openrouter/openai/gpt-5.5") == "gpt-5.5"


class TestAnnotateForecasterBulletsWithModels:
    def test_adds_model_name_to_each_bullet(self):
        text = "### Forecasts\n*Forecaster 1*: 73.0%\n*Forecaster 2*: 75.0%\n"
        annotated = annotate_forecaster_bullets_with_models(text, {1: "gpt-5.5", 2: "claude-opus-4.7"})
        assert "*Forecaster 1 (gpt-5.5)*: 73.0%" in annotated
        assert "*Forecaster 2 (claude-opus-4.7)*: 75.0%" in annotated

    def test_idempotent_when_already_annotated(self):
        text = "*Forecaster 1 (gpt-5.5)*: 73.0%\n"
        annotated = annotate_forecaster_bullets_with_models(text, {1: "gpt-5.5"})
        assert annotated == text

    def test_indices_not_in_map_left_unchanged(self):
        text = "*Forecaster 1*: 73.0%\n*Forecaster 2*: 75.0%\n"
        annotated = annotate_forecaster_bullets_with_models(text, {1: "gpt-5.5"})
        assert "*Forecaster 1 (gpt-5.5)*: 73.0%" in annotated
        assert "*Forecaster 2*: 75.0%" in annotated

    def test_round_trip_parses_back_to_model_names(self):
        # The full round-trip: annotate a summary-only text, then parse it back.
        # Simulates the comment-trimmed case where the rationales section is
        # cut off but the summary survives.
        text = (
            "## Report 1 Summary\n### Forecasts\n*Forecaster 1*: 73.0%\n*Forecaster 2*: 75.0%\n[... trimmed for length]"
        )
        annotated = annotate_forecaster_bullets_with_models(text, {1: "gpt-5.5", 2: "claude-opus-4.7"})
        parsed = parse_per_model_forecasts(annotated)
        assert parsed == {"gpt-5.5": "73.0%", "claude-opus-4.7": "75.0%"}

    def test_survives_comment_trim_path(self):
        # End-to-end: annotated summary + trimmed rationale section still
        # yields correct per-model attribution.
        summary_only = (
            "*Forecaster 1 (gpt-5.5)*: 73.0%\n*Forecaster 2 (claude-opus-4.7)*: 75.0%\n[... trimmed for length]"
        )
        parsed = parse_per_model_forecasts(summary_only)
        assert parsed == {"gpt-5.5": "73.0%", "claude-opus-4.7": "75.0%"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_comment(bullets: list[tuple[int, str]], rationales: list[tuple[int, str]]) -> str:
    """Build a minimal bot-comment-shaped string.

    bullets:   list of (idx, value) for the Report 1 Summary section
    rationales: list of (idx, model) for the R1 Forecaster N Reasoning blocks
    """
    bullet_lines = "\n".join(f"*Forecaster {i}*: {v}" for i, v in bullets)
    rationale_sections = "\n\n".join(
        f"## R1: Forecaster {i} Reasoning\nModel: {m}\n\nreasoning text here." for i, m in rationales
    )
    return f"""# SUMMARY
*Question*: ?

## Report 1 Summary
### Forecasts
{bullet_lines}


### Research Summary
Some research text.

================================================================================
FORECAST SECTION:

{rationale_sections}
"""


def _build_base_pred(model_path: str, extra_body: str) -> Mock:
    """Build a Mock ReasonedPrediction-shaped object with the bot's Model: prefix.

    main.TemplateForecaster._make_prediction injects the prefix on real predictions;
    we replicate that here so combine_stacker_and_base_reasoning produces a realistic
    shape.
    """
    pred = Mock()
    pred.reasoning = f"Model: {model_path}\n\n{extra_body}"
    return pred


def _build_stacked_comment(
    combined_reasoning: str,
    stacker_bullet_name: str = "stacker-gpt",
    bullet_value: str = "72.0%",
) -> str:
    """Wrap a stacker-combined body in a plausible Metaculus comment skeleton.

    The production flow produces exactly ONE summary bullet for stacked questions
    (the aggregate value) and ONE R1 header whose body is the combined reasoning
    emitted by combine_stacker_and_base_reasoning.
    """
    return f"""# SUMMARY
*Question*: Will stacking improve calibration?

## Report 1 Summary
### Forecasts
*Forecaster 1 ({stacker_bullet_name})*: {bullet_value}

### Research Summary
Some research.

================================================================================
FORECAST SECTION:

## R1: Forecaster 1 Reasoning
{combined_reasoning}

<!-- STACKED=true -->
"""


# ---------------------------------------------------------------------------
# parse_forecaster_model_map
# ---------------------------------------------------------------------------


class TestParseForecasterModelMap:
    def test_standard_five_model_ensemble(self):
        comment = _build_comment(
            bullets=[(1, "73.0%"), (2, "75.0%"), (3, "85.0%"), (4, "92.0%"), (5, "88.0%")],
            rationales=[
                (1, "openrouter/openai/gpt-5.2"),
                (2, "openrouter/openai/gpt-5.1"),
                (3, "openrouter/anthropic/claude-4.6-opus"),
                (4, "openrouter/anthropic/claude-opus-4.5"),
                (5, "openrouter/google/gemini-3.1-pro-preview"),
            ],
        )
        result = parse_forecaster_model_map(comment)
        assert result == {
            1: "gpt-5.2",
            2: "gpt-5.1",
            3: "claude-4.6-opus",
            4: "claude-opus-4.5",
            5: "gemini-3.1-pro-preview",
        }

    def test_model_names_with_dots_and_hyphens(self):
        comment = _build_comment(
            bullets=[(1, "50%")],
            rationales=[(1, "openrouter/openai/gpt-5.5")],
        )
        assert parse_forecaster_model_map(comment) == {1: "gpt-5.5"}

    def test_long_multi_dot_name(self):
        comment = _build_comment(
            bullets=[(1, "50%")],
            rationales=[(1, "openrouter/google/gemini-3.1-pro-preview")],
        )
        assert parse_forecaster_model_map(comment) == {1: "gemini-3.1-pro-preview"}

    def test_ignores_report_2_rationales(self):
        # Only R1 matters — R2 is a separate research report; attribution should
        # come from R1.
        comment = _build_comment(
            bullets=[(1, "60%")],
            rationales=[(1, "openrouter/openai/gpt-5.2")],
        )
        # Append a bogus R2 with a different model
        comment += "\n\n## R2: Forecaster 1 Reasoning\nModel: openrouter/openai/gpt-4\n\nother."
        assert parse_forecaster_model_map(comment) == {1: "gpt-5.2"}

    def test_missing_model_lines_returns_empty(self):
        comment = _build_comment(
            bullets=[(1, "60%")],
            rationales=[],
        )
        assert parse_forecaster_model_map(comment) == {}

    def test_malformed_model_line_skipped(self):
        # Line says "Model:" but no value — must not crash, must return empty
        comment = """## R1: Forecaster 1 Reasoning
Model:

## R1: Forecaster 2 Reasoning
Model: openrouter/openai/gpt-5.2

"""
        # Forecaster 1 is malformed; Forecaster 2 is fine
        result = parse_forecaster_model_map(comment)
        assert result == {2: "gpt-5.2"}

    def test_stray_model_line_in_research_ignored(self):
        # A Model: line appearing outside a Forecaster Reasoning header (e.g.,
        # research quoting a model name) must not confuse attribution.
        comment = (
            "### Research Summary\n"
            "Model: openrouter/openai/gpt-99-fake\n"
            "This was cited in research.\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.2\n\n"
            "real reasoning.\n"
        )
        assert parse_forecaster_model_map(comment) == {1: "gpt-5.2"}

    def test_whitespace_variations(self):
        # Tabs, extra spaces, trailing whitespace on Model line must still parse.
        comment = "## R1: Forecaster 1 Reasoning\n  Model:   openrouter/openai/gpt-5.2   \n\nreasoning\n"
        assert parse_forecaster_model_map(comment) == {1: "gpt-5.2"}

    def test_empty_comment(self):
        assert parse_forecaster_model_map("") == {}


# ---------------------------------------------------------------------------
# parse_per_model_forecasts — integration of model map + bullets
# ---------------------------------------------------------------------------


class TestParsePerModelForecasts:
    def test_happy_path_keys_by_model_name(self):
        comment = _build_comment(
            bullets=[(1, "73.0%"), (2, "75.0%"), (3, "85.0%")],
            rationales=[
                (1, "openrouter/openai/gpt-5.2"),
                (2, "openrouter/openai/gpt-5.1"),
                (3, "openrouter/anthropic/claude-4.6-opus"),
            ],
        )
        result = parse_per_model_forecasts(comment)
        assert result == {
            "gpt-5.2": "73.0%",
            "gpt-5.1": "75.0%",
            "claude-4.6-opus": "85.0%",
        }

    def test_bullets_without_rationales_falls_back_to_anonymized(self):
        # Pre-injection comments (if any exist) or future breakage — don't
        # guess, return anonymized keys so analysis can still aggregate
        # overall but per-model numbers are correctly absent.
        comment = """## Report 1 Summary
### Forecasts
*Forecaster 1*: 73.0%
*Forecaster 2*: 75.0%
"""
        result = parse_per_model_forecasts(comment)
        assert result == {"Forecaster 1": "73.0%", "Forecaster 2": "75.0%"}

    def test_partial_model_lines_mixes_named_and_anonymized(self):
        # Forecaster 1 has Model: line, Forecaster 2 does not.
        comment = """## Report 1 Summary
### Forecasts
*Forecaster 1*: 73.0%
*Forecaster 2*: 75.0%

## R1: Forecaster 1 Reasoning
Model: openrouter/openai/gpt-5.2

reasoning.
"""
        result = parse_per_model_forecasts(comment)
        assert result == {"gpt-5.2": "73.0%", "Forecaster 2": "75.0%"}

    def test_empty_comment(self):
        assert parse_per_model_forecasts("") == {}

    def test_explicit_model_names_override_is_supported(self):
        # Back-compat: callers can pass model_names to force a specific roster
        # (useful for tests or external pipelines). When passed, skip the
        # Model:-line lookup.
        comment = """*Forecaster 1*: 73.0%
*Forecaster 2*: 75.0%
"""
        result = parse_per_model_forecasts(comment, model_names=["alpha", "beta"])
        assert result == {"alpha": "73.0%", "beta": "75.0%"}


# ---------------------------------------------------------------------------
# Stacked-comment parsing — per-base-model recovery from the combined R1 body.
# ---------------------------------------------------------------------------


class TestParsePerModelNumericPercentilesStacked:
    """Per-base-model percentile recovery from stacker-combined comments.

    When stacking fires, all base predictions collapse into one ReasonedPrediction,
    so there's only one R1 header; the base reasonings are folded into its body
    under the "## Base Model Reasoning (inputs to stacker)" delimiter. The parser
    must split on that delimiter and attribute percentiles per base model.
    """

    def test_three_base_models_each_with_percentiles(self):
        base_names = [
            "openrouter/openai/gpt-5.5",
            "openrouter/anthropic/claude-opus-4.7",
            "openrouter/google/gemini-3.1-pro-preview",
        ]
        base_preds = []
        for i, path in enumerate(base_names):
            body = (
                f"analysis for model {i}\n\n"
                f"Percentile 10: {100 + 10 * i}\n"
                f"Percentile 25: {125 + 10 * i}\n"
                f"Percentile 50: {150 + 10 * i}\n"
                f"Percentile 75: {175 + 10 * i}\n"
                f"Percentile 90: {200 + 10 * i}"
            )
            base_preds.append(_build_base_pred(path, body))
        meta_text = "My stacker analysis: considered the three inputs, leaned toward median-ish."
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined)

        result = parse_per_model_numeric_percentiles(comment)
        # Three per-base-model entries — one per sub-block.
        assert set(result.keys()) == {"gpt-5.5", "claude-opus-4.7", "gemini-3.1-pro-preview"}
        assert result["gpt-5.5"] == [(10.0, 100.0), (25.0, 125.0), (50.0, 150.0), (75.0, 175.0), (90.0, 200.0)]
        assert result["claude-opus-4.7"] == [(10.0, 110.0), (25.0, 135.0), (50.0, 160.0), (75.0, 185.0), (90.0, 210.0)]
        assert result["gemini-3.1-pro-preview"] == [
            (10.0, 120.0),
            (25.0, 145.0),
            (50.0, 170.0),
            (75.0, 195.0),
            (90.0, 220.0),
        ]

    def test_stacker_meta_percentiles_included_under_stacker_name(self):
        # Some stackers explicitly restate the final distribution in prose.
        # Those percentile lines should be captured under the stacker's model
        # name (the R1 header's Model: line, or "Forecaster 1" fallback when
        # absent).
        base_pred = _build_base_pred(
            "openrouter/openai/gpt-5.5",
            "base prose\n\nPercentile 50: 100",
        )
        meta_text = "Final stacker distribution:\nPercentile 10: 90\nPercentile 50: 110\nPercentile 90: 140"
        combined = combine_stacker_and_base_reasoning(meta_text, [base_pred])
        comment = _build_stacked_comment(combined)

        result = parse_per_model_numeric_percentiles(comment)
        # Base model's single percentile line
        assert result["gpt-5.5"] == [(50.0, 100.0)]
        # Stacker's prose percentiles — the R1 body has no Model: line (the
        # aggregated_prediction's reasoning starts with ## Stacker Meta-Analysis),
        # so the stacker falls back to "Forecaster 1".
        assert result["Forecaster 1"] == [(10.0, 90.0), (50.0, 110.0), (90.0, 140.0)]

    def test_non_stacked_comment_matches_existing_behavior(self):
        # Sanity: when the delimiter is absent, the stacker branch must not
        # trigger, and per-R1-header attribution behaves as before.
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "analysis...\n\nPercentile 50: 42\n\n"
            "## R1: Forecaster 2 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "analysis...\n\nPercentile 50: 55\n"
        )
        assert parse_per_model_numeric_percentiles(comment) == {
            "gpt-5.5": [(50.0, 42.0)],
            "claude-opus-4.7": [(50.0, 55.0)],
        }

    def test_mixed_r1_blocks_stacked_and_unstacked(self):
        # Unusual (cross-report) shape: one R1 block is stacker-combined, another
        # is a plain per-forecaster section. Each should be handled on its own
        # terms without cross-contamination.
        base_pred = _build_base_pred(
            "openrouter/openai/gpt-5.5",
            "base prose\n\nPercentile 50: 200",
        )
        combined = combine_stacker_and_base_reasoning("stacker meta", [base_pred])
        comment = (
            f"## R1: Forecaster 1 Reasoning\n{combined}\n"
            "## R1: Forecaster 2 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "plain analysis\n\nPercentile 50: 300\n"
        )
        result = parse_per_model_numeric_percentiles(comment)
        # Stacked block contributes the base model's name
        assert result["gpt-5.5"] == [(50.0, 200.0)]
        # Non-stacked block keyed by its own Model: line
        assert result["claude-opus-4.7"] == [(50.0, 300.0)]
        # Fallback "Forecaster N" should NOT appear — the stacker meta-text
        # has no percentile lines so it's not attributed.
        assert "Forecaster 1" not in result


class TestParsePerModelReasoningTextStacked:
    """Per-base-model prose recovery from stacker-combined comments."""

    def test_returns_entry_per_base_model_plus_stacker(self):
        base_names = [
            "openrouter/openai/gpt-5.5",
            "openrouter/anthropic/claude-opus-4.7",
            "openrouter/google/gemini-3.1-pro-preview",
        ]
        base_preds = [_build_base_pred(name, f"reasoning body for model {i}") for i, name in enumerate(base_names)]
        meta_text = "My stacker meta-analysis prose."
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined)

        result = parse_per_model_reasoning_text(comment)
        # 3 base models + stacker = 4 entries. Stacker's R1 body has no Model:
        # line, so it falls back to "Forecaster 1".
        assert set(result.keys()) == {"gpt-5.5", "claude-opus-4.7", "gemini-3.1-pro-preview", "Forecaster 1"}
        for i, name in enumerate(["gpt-5.5", "claude-opus-4.7", "gemini-3.1-pro-preview"]):
            assert f"reasoning body for model {i}" in result[name]
        assert "stacker meta-analysis prose" in result["Forecaster 1"].lower()

    def test_non_stacked_comment_matches_existing_behavior(self):
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "plain reasoning A.\n\n"
            "## R1: Forecaster 2 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "plain reasoning B.\n"
        )
        result = parse_per_model_reasoning_text(comment)
        assert "plain reasoning A" in result["gpt-5.5"]
        assert "plain reasoning B" in result["claude-opus-4.7"]
        assert set(result.keys()) == {"gpt-5.5", "claude-opus-4.7"}


class TestParsePerModelForecastsStacked:
    """parse_per_model_forecasts only captures the summary bullet.

    For stacked comments the bot publishes ONE summary bullet (the stacker's
    aggregate), not one per base model. So this function correctly returns
    a single-entry dict — the per-base-model values are never written to the
    summary and are recoverable only from the reasoning prose via
    parse_per_model_reasoning_text / parse_per_model_numeric_percentiles.
    Documenting this behavior explicitly so a future reader doesn't misread
    the single-entry return as a bug.
    """

    def test_stacked_comment_returns_single_entry_for_stacker(self):
        base_preds = [
            _build_base_pred("openrouter/openai/gpt-5.5", "base A"),
            _build_base_pred("openrouter/anthropic/claude-opus-4.7", "base B"),
        ]
        combined = combine_stacker_and_base_reasoning("stacker prose", base_preds)
        comment = _build_stacked_comment(combined, stacker_bullet_name="stacker-model", bullet_value="72.0%")

        result = parse_per_model_forecasts(comment)
        # Exactly one entry — the stacker's aggregate. This is BY DESIGN: the
        # bot does not publish per-base-model bullets when stacking fires.
        assert result == {"stacker-model": "72.0%"}


class TestStackedCommentEndToEnd:
    """End-to-end: synthesize a stacker-combined comment and run every parser."""

    def test_full_parse_pipeline(self):
        base_names = [
            "openrouter/openai/gpt-5.5",
            "openrouter/anthropic/claude-opus-4.7",
        ]
        base_preds = []
        for i, path in enumerate(base_names):
            body = f"reasoning body {i}\n\nPercentile 50: {150 + 10 * i}\nPercentile 90: {200 + 10 * i}"
            base_preds.append(_build_base_pred(path, body))
        meta_text = "Stacker: I averaged the medians with a slight rightward lean."
        combined = combine_stacker_and_base_reasoning(meta_text, base_preds)
        comment = _build_stacked_comment(combined, stacker_bullet_name="stacker-gpt-5.5", bullet_value="165.0")

        # parse_stacked_marker — trivially True
        assert parse_stacked_marker(comment) is True

        # parse_per_model_forecasts — only the stacker's aggregate
        forecasts = parse_per_model_forecasts(comment)
        assert forecasts == {"stacker-gpt-5.5": "165.0"}

        # parse_per_model_numeric_percentiles — per-base-model distributions
        percentiles = parse_per_model_numeric_percentiles(comment)
        assert percentiles["gpt-5.5"] == [(50.0, 150.0), (90.0, 200.0)]
        assert percentiles["claude-opus-4.7"] == [(50.0, 160.0), (90.0, 210.0)]

        # parse_per_model_reasoning_text — per-base-model prose plus stacker's
        reasoning = parse_per_model_reasoning_text(comment)
        assert "reasoning body 0" in reasoning["gpt-5.5"]
        assert "reasoning body 1" in reasoning["claude-opus-4.7"]
        # Stacker's meta text — no Model: line in combined body, so key falls
        # back to "Forecaster 1".
        assert "averaged the medians" in reasoning["Forecaster 1"]


# ---------------------------------------------------------------------------
# Real-data regression test
# ---------------------------------------------------------------------------


_PERF_DATA_PATH = Path(__file__).parent.parent / "scratch" / "performance_data.json"


@pytest.mark.skipif(not _PERF_DATA_PATH.exists(), reason="performance_data.json not checked in")
class TestRealDataRegression:
    """Parse stored Metaculus comments and check attributions against real models.

    Uses whatever roster the checked-in fixture was collected under. This test
    does not assume a specific set of models — it just asserts that every
    parsed record's per-model keys look like real model names, not
    index-anonymized `Forecaster N`.
    """

    @pytest.fixture(scope="class")
    def records(self):
        with open(_PERF_DATA_PATH) as f:
            return json.load(f)

    def test_most_march_records_parse_to_named_models(self, records):
        # Some comments hit the Metaculus comment char limit and get trimmed
        # mid-rationale, erasing a Model: line. Anonymized fallback is the
        # CORRECT behavior in those cases — better than silently mislabeling.
        # We require >=90% of records to parse cleanly; the remainder must be
        # accounted for by trim-induced missing Model: lines, not a parser
        # regression.
        total = 0
        fully_parsed = 0
        trimmed_bad = []
        parser_bad = []
        for rec in records:
            comment = rec.get("comment_text") or ""
            if not comment:
                continue
            total += 1
            parsed = parse_per_model_forecasts(comment)
            bad_keys = [k for k in parsed if k.startswith("Forecaster ")]
            if not bad_keys:
                fully_parsed += 1
            elif "[... trimmed for length]" in comment:
                trimmed_bad.append(rec["post_id"])
            else:
                parser_bad.append((rec["post_id"], bad_keys))
        assert not parser_bad, f"Non-trim-related parse failures: {parser_bad[:10]}"
        assert fully_parsed / total >= 0.90, (
            f"Only {fully_parsed}/{total} records parsed cleanly ({len(trimmed_bad)} due to comment trimming)"
        )

    def test_known_sample_post_matches_expected_models(self, records):
        # post 42631 (Oscar winner question) is in the March cohort — sampled
        # in plan exploration and known to use the old roster.
        sample = next((r for r in records if r["post_id"] == 42631), None)
        if sample is None:
            pytest.skip("sample post 42631 not in data")
        assert sample is not None  # narrows for the type checker (pytest.skip raises)
        if "openrouter/openai/gpt-5.2" not in sample["comment_text"]:
            pytest.skip("post 42631 no longer from March roster; refresh test fixture or delete")
        parsed = parse_per_model_forecasts(sample["comment_text"])
        expected_values = {
            "gpt-5.2": "56.0%",
            "gpt-5.1": "57.0%",
            "claude-4.6-opus": "52.0%",
            "claude-opus-4.5": "52.0%",
            "gemini-3.1-pro-preview": "58.0%",
        }
        assert parsed == expected_values


# ---------------------------------------------------------------------------
# parse_resolution — unchanged behavior, but worth a guard
# ---------------------------------------------------------------------------


class TestParseResolution:
    def test_binary_yes(self):
        assert parse_resolution("yes", "binary") == (True, False)

    def test_binary_no(self):
        assert parse_resolution("no", "binary") == (False, False)

    def test_annulled(self):
        _, skip = parse_resolution("annulled", "binary")
        assert skip is True

    def test_numeric_float(self):
        val, skip = parse_resolution("42.5", "numeric")
        assert val == 42.5 and skip is False

    def test_numeric_above_upper(self):
        val, skip = parse_resolution("above_upper_bound", "numeric")
        assert val == "above_upper_bound" and skip is False


# ---------------------------------------------------------------------------
# Shared-constant producer/consumer round-trip. The stacker-combined comment
# format is emitted by metaculus_bot.stacking.combine_stacker_and_base_reasoning
# and re-parsed by the performance-analysis pipeline. If these ever drift, the
# collector silently loses per-base-model attribution on stacked questions, so
# guard the shared header literals explicitly.
# ---------------------------------------------------------------------------


class TestStackedHeaderConstantRoundTrip:
    def test_combine_output_contains_shared_constants(self):
        base_pred = _build_base_pred("openrouter/openai/gpt-5.5", "base prose")
        combined = combine_stacker_and_base_reasoning("stacker meta prose", [base_pred])
        assert STACKER_META_ANALYSIS_HEADER in combined
        assert STACKED_BASE_REASONING_HEADER in combined

    def test_parser_splits_on_shared_constant(self):
        # Verify parser ingests the same literal the producer emits — no drift.
        base_pred = _build_base_pred("openrouter/openai/gpt-5.5", "base prose\n\nPercentile 50: 42")
        combined = combine_stacker_and_base_reasoning("stacker meta", [base_pred])
        comment = _build_stacked_comment(combined)
        parsed = parse_per_model_numeric_percentiles(comment)
        assert parsed["gpt-5.5"] == [(50.0, 42.0)]


# ---------------------------------------------------------------------------
# _iter_per_model_blocks: centralizes the R1-walking boilerplate shared by
# parse_per_model_numeric_percentiles and parse_per_model_reasoning_text. Keep
# a minimal shape check so a future refactor doesn't silently change attribution.
# ---------------------------------------------------------------------------


class TestIterPerModelBlocks:
    def test_plain_body_yields_single_non_stacker_entry_per_r1(self):
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "prose A\n\n"
            "## R1: Forecaster 2 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "prose B\n"
        )
        blocks = list(_iter_per_model_blocks(comment))
        assert len(blocks) == 2
        names = [b[0] for b in blocks]
        is_meta_flags = [b[2] for b in blocks]
        assert names == ["gpt-5.5", "claude-opus-4.7"]
        assert is_meta_flags == [False, False]
        # Bodies must have the Model: prefix stripped.
        assert "prose A" in blocks[0][1]
        assert not blocks[0][1].lstrip().startswith("Model:")

    def test_stacker_combined_body_yields_meta_then_bases(self):
        base_preds = [
            _build_base_pred("openrouter/openai/gpt-5.5", "base A prose"),
            _build_base_pred("openrouter/anthropic/claude-opus-4.7", "base B prose"),
        ]
        combined = combine_stacker_and_base_reasoning("stacker meta prose", base_preds)
        comment = _build_stacked_comment(combined)
        blocks = list(_iter_per_model_blocks(comment))
        # 1 stacker-meta entry + 2 base entries
        assert len(blocks) == 3
        assert blocks[0][2] is True  # stacker meta
        assert blocks[1][2] is False and blocks[2][2] is False
        assert "stacker meta prose" in blocks[0][1]
        assert blocks[1][0] == "gpt-5.5"
        assert blocks[2][0] == "claude-opus-4.7"
