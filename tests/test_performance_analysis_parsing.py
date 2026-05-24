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
    _parse_probability,
    annotate_forecaster_bullets_with_models,
    detect_historical_stacker_signature,
    extract_model_display_name_from_reasoning,
    parse_forecaster_model_map,
    parse_inferred_stacker_outcome,
    parse_per_model_forecasts,
    parse_per_model_numeric_percentiles,
    parse_per_model_reasoning_text,
    parse_resolution,
    parse_stacked_marker,
    parse_stacker_outcome_marker,
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
# parse_stacker_outcome_marker (new tri-state marker)
# ---------------------------------------------------------------------------


class TestParseStackerOutcomeMarker:
    def test_primary_marker(self):
        assert parse_stacker_outcome_marker("...\n<!-- STACKER_OUTCOME=primary -->\n") == "primary"

    def test_fallback_llm_marker(self):
        assert parse_stacker_outcome_marker("...\n<!-- STACKER_OUTCOME=fallback_llm -->\n") == "fallback_llm"

    def test_fallback_median_marker(self):
        # The load-bearing case: pre-fix, this path silently emitted STACKED=true.
        assert parse_stacker_outcome_marker("...\n<!-- STACKER_OUTCOME=fallback_median -->\n") == "fallback_median"

    def test_skipped_marker(self):
        assert parse_stacker_outcome_marker("...\n<!-- STACKER_OUTCOME=skipped -->\n") == "skipped"

    def test_absent_marker_returns_none(self):
        assert parse_stacker_outcome_marker("# SUMMARY\nNo marker here\n") is None

    def test_case_insensitive(self):
        assert parse_stacker_outcome_marker("<!-- stacker_outcome=PRIMARY -->") == "primary"

    def test_whitespace_tolerant(self):
        assert parse_stacker_outcome_marker("<!--  STACKER_OUTCOME=fallback_llm  -->") == "fallback_llm"


# ---------------------------------------------------------------------------
# detect_historical_stacker_signature (body shape, no marker)
# ---------------------------------------------------------------------------


class TestDetectHistoricalStackerSignature:
    """Detection of pre-marker stacked comments (April 2026 vintage and earlier).

    The historical signature is `## R1: Forecaster 1 Reasoning` immediately
    followed by `## (Stacker )?Meta-Analysis`. This shape is produced by the
    stacking commit (2026-04-02, c6d1ab3) and survived the rename in
    95c4fff (2026-04-27). Comments published before either commit landed
    don't carry the signature; comments where Forecaster 1's body is
    ordinary model reasoning don't carry it either.
    """

    def test_historical_meta_analysis_header(self):
        # Pre-rename: `## Meta-Analysis` (no "Stacker" prefix).
        comment = (
            "# SUMMARY\nstuff\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.5\n\n"
            "## Meta-Analysis\n\n"
            "Synthesis of 6 base models below.\n"
        )
        assert detect_historical_stacker_signature(comment) is True

    def test_current_stacker_meta_analysis_header(self):
        # Post-rename: `## Stacker Meta-Analysis`.
        comment = (
            "# SUMMARY\nstuff\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "## Stacker Meta-Analysis\n\n"
            "Synthesis of 6 base models below.\n"
        )
        assert detect_historical_stacker_signature(comment) is True

    def test_h1_meta_analysis_and_synthesis_header(self):
        # Earliest stacker variant emitted ``# Meta-Analysis and Synthesis``
        # as an H1 heading. Observed in April 2026 records of the
        # spring-aib-2026 dataset (e.g., post 43155).
        comment = (
            "# SUMMARY\nstuff\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.4\n\n"
            "# Meta-Analysis and Synthesis\n"
            "## Model Agreement Analysis\n"
            "..."
        )
        assert detect_historical_stacker_signature(comment) is True

    def test_no_meta_header_in_first_section_returns_false(self):
        # Plain stacking-skipped or non-stacking comment: Forecaster 1 has its
        # own analysis structure, not a meta-analysis.
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "Probability: 72%\nReasoning: ...\n\n"
            "## R1: Forecaster 2 Reasoning\nProbability: 75%\n"
        )
        assert detect_historical_stacker_signature(comment) is False

    def test_meta_analysis_inside_individual_forecaster_body_returns_false(self):
        # A model's own reasoning structure may include a "## Meta-Analysis"
        # heading several paragraphs in — that's not the stacker shape; the
        # FIRST heading after the section header is what matters.
        comment = (
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "## Source Analysis\nstuff\n\n"
            "## Meta-Analysis\nlocal sub-section\n"
        )
        assert detect_historical_stacker_signature(comment) is False

    def test_no_forecaster_1_section_returns_false(self):
        assert detect_historical_stacker_signature("# SUMMARY\nno forecasts here\n") is False

    def test_whitespace_tolerant(self):
        comment = "## R1: Forecaster 1 Reasoning\n\n\nModel:openrouter/openai/gpt-5.5\n## Stacker Meta-Analysis\n"
        assert detect_historical_stacker_signature(comment) is True


# ---------------------------------------------------------------------------
# parse_inferred_stacker_outcome (combined: marker → legacy → historical-body)
# ---------------------------------------------------------------------------


class TestParseInferredStackerOutcome:
    """End-to-end stacker-outcome detection across all three sources of signal.

    Verifies that the combined parser handles every combination a real dataset
    can throw at it: post-fix tri-state markers (preferred), legacy STACKED=
    markers (one-round back-compat), and pre-marker historical body shapes.
    """

    def test_marker_outcome_preferred_when_present(self):
        comment = (
            "## R1: Forecaster 1 Reasoning\nModel: x\n\n## Meta-Analysis\nstuff\n"
            "<!-- STACKER_OUTCOME=fallback_llm -->\n"
            "<!-- STACKED=true -->\n"
        )
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "fallback_llm"
        assert source == "marker_outcome"

    def test_marker_outcome_fallback_median_distinguished_from_legacy(self):
        # The bug fix: fallback_median should be readable distinctly, not
        # collapsed to legacy STACKED=true (the pre-fix behavior).
        comment = "stuff\n<!-- STACKER_OUTCOME=fallback_median -->\n"
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "fallback_median"
        assert source == "marker_outcome"

    def test_legacy_marker_true_maps_to_primary(self):
        # Legacy STACKED=true cannot distinguish primary from fallback_llm,
        # so it conservatively maps to primary (the most common case).
        comment = "stuff\n<!-- STACKED=true -->\n"
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "primary"
        assert source == "marker_legacy"

    def test_legacy_marker_false_maps_to_skipped(self):
        comment = "stuff\n<!-- STACKED=false -->\n"
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "skipped"
        assert source == "marker_legacy"

    def test_historical_body_with_no_marker_infers_primary(self):
        comment = (
            "# SUMMARY\nstuff\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.5\n\n"
            "## Meta-Analysis\nSynthesis of base models.\n"
        )
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "primary"
        assert source == "historical_body"

    def test_no_signal_returns_none(self):
        comment = "## R1: Forecaster 1 Reasoning\nModel: openrouter/openai/gpt-5.5\n\nProbability: 72%\n"
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome is None
        assert source == "none"

    def test_marker_outcome_takes_precedence_over_historical_body(self):
        # Belt-and-suspenders: even if both signals are present (marker added
        # to a comment that already has the historical body shape), the
        # explicit marker wins.
        comment = (
            "## R1: Forecaster 1 Reasoning\nModel: x\n## Stacker Meta-Analysis\nbody\n"
            "<!-- STACKER_OUTCOME=fallback_median -->\n"
        )
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "fallback_median"
        assert source == "marker_outcome"

    def test_legacy_marker_takes_precedence_over_historical_body(self):
        # Precedence ordering is marker_outcome > marker_legacy > historical_body.
        # If a comment has both STACKED=false (legacy "skipped") and a historical
        # body shape that would otherwise be inferred as primary, the legacy
        # marker wins. Source must reflect this so downstream filters can tell
        # marker-derived outcomes from body-shape-derived ones.
        comment = "## R1: Forecaster 1 Reasoning\nModel: x\n## Stacker Meta-Analysis\nbody\n<!-- STACKED=false -->\n"
        outcome, source = parse_inferred_stacker_outcome(comment)
        assert outcome == "skipped"
        assert source == "marker_legacy"


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


# ---------------------------------------------------------------------------
# F10 — _FORECASTER_RE must only match summary-section bullets, not stray
# ``*Forecaster N*: value`` strings appearing inside reasoning prose. Without
# this, an LLM that quotes a forecaster's value back ("...*Forecaster 3*: 50%
# would have been wrong") gets parsed as if it were a real bullet, contaminating
# rank/disagreement metrics.
# ---------------------------------------------------------------------------


class TestForecasterReBoundedToSummarySection:
    def test_stray_forecaster_pattern_in_reasoning_not_parsed(self):
        # A bullet-shaped string appearing INSIDE Forecaster 1's reasoning must
        # NOT be picked up as a real summary bullet. Only the bullets in
        # ``## Report 1 Summary`` count.
        comment = _build_comment(
            bullets=[(1, "73.0%"), (2, "75.0%")],
            rationales=[
                (1, "openrouter/openai/gpt-5.5"),
                (2, "openrouter/anthropic/claude-opus-4.7"),
            ],
        )
        # Inject a stray "bullet" inside the rationale prose - this should be
        # ignored. Without anchoring this becomes a third forecaster.
        comment = comment.replace(
            "reasoning text here.",
            "reasoning text here. Note: *Forecaster 99*: 12.0% would have been wrong.",
            1,
        )
        result = parse_per_model_forecasts(comment)
        # Only the two real summary bullets — not "Forecaster 99".
        assert set(result.keys()) == {"gpt-5.5", "claude-opus-4.7"}
        assert "Forecaster 99" not in result

    def test_real_summary_bullets_still_parsed(self):
        # Belt-and-suspenders: confirm anchoring doesn't break the normal path.
        comment = _build_comment(
            bullets=[(1, "73.0%"), (2, "75.0%"), (3, "80.0%")],
            rationales=[
                (1, "openrouter/openai/gpt-5.5"),
                (2, "openrouter/anthropic/claude-opus-4.7"),
                (3, "openrouter/google/gemini-3.1-pro-preview"),
            ],
        )
        result = parse_per_model_forecasts(comment)
        assert result == {"gpt-5.5": "73.0%", "claude-opus-4.7": "75.0%", "gemini-3.1-pro-preview": "80.0%"}

    def test_no_summary_marker_falls_back_with_warning(self, caplog):
        # When the comment has no ``### Research Summary`` marker (truncated /
        # malformed), the parser must still be able to read summary bullets if
        # they exist BEFORE any stray reasoning text — and emit a warning so
        # operators notice. We accept the legacy unanchored behavior in this
        # path because there's no clean section to bound to.
        import logging

        comment = "*Forecaster 1*: 73.0%\n*Forecaster 2*: 75.0%\n"
        with caplog.at_level(logging.WARNING):
            result = parse_per_model_forecasts(comment)
        assert result == {"Forecaster 1": "73.0%", "Forecaster 2": "75.0%"}

    def test_bullets_in_research_summary_section_still_excluded(self):
        # Edge case: research prose between ``### Research Summary`` and the
        # rationale section sometimes quotes forecaster-shaped strings. The
        # cleanest split is on the section-end marker (Research Summary or the
        # forecast section divider).
        comment = _build_comment(
            bullets=[(1, "73.0%")],
            rationales=[(1, "openrouter/openai/gpt-5.5")],
        )
        # Inject a stray bullet shape inside the research summary section.
        comment = comment.replace(
            "Some research text.",
            "Some research text.\nQuote: *Forecaster 99*: 99% (this is a quote, not a bullet).",
        )
        result = parse_per_model_forecasts(comment)
        assert "Forecaster 99" not in result
        assert result == {"gpt-5.5": "73.0%"}


# ---------------------------------------------------------------------------
# F11 — _parse_probability must reject implausible values that would be silent
# parse errors. Currently ``"1.5"`` becomes 0.015 (almost certainly wrong),
# which silently enters Brier/spread calcs. Tighten the heuristic so:
#   - Explicit "%" → divide by 100 always
#   - Bare value in [0, 1] → leave alone
#   - Bare value in [1.5, 100] → treat as percentage, divide by 100
#   - Bare value in (1, 1.5) → ambiguous, reject (None) - was a probability
#     >1.0 (impossible) but too small to be a reasonable percentage
#   - Bare value > 100 → reject (None)
# ---------------------------------------------------------------------------


class TestParseProbabilityHeuristic:
    def test_decimal_probability_unchanged(self):
        assert _parse_probability("0.72") == pytest.approx(0.72)

    def test_explicit_percentage(self):
        assert _parse_probability("72.0%") == pytest.approx(0.72)

    def test_explicit_percentage_with_value_below_1_5_still_works(self):
        # ``"1.0%"`` is unambiguously 1% because of the explicit % sign.
        assert _parse_probability("1.0%") == pytest.approx(0.01)

    def test_bare_value_above_threshold_treated_as_percentage(self):
        # ``"72"`` (no %) is almost certainly the bot's missing-% form.
        assert _parse_probability("72") == pytest.approx(0.72)

    def test_bare_value_at_threshold_lower_bound(self):
        # The lower bound for "treat as percentage" — 1.5 is unambiguous.
        assert _parse_probability("1.5") == pytest.approx(0.015)

    def test_bare_value_in_ambiguous_range_rejected(self):
        # ``"1.2"`` (no %) is ambiguous: too high to be a valid probability,
        # too low to confidently be a percentage. Reject as parse error.
        assert _parse_probability("1.2") is None

    def test_bare_value_just_above_one_rejected(self):
        # ``"1.01"`` shouldn't silently become 0.0101 — that's almost certainly
        # a parse error (a real bullet would be 0.01 or 1%).
        assert _parse_probability("1.01") is None

    def test_bare_value_above_one_hundred_rejected(self):
        # ``"150"`` could be a percentage (impossible) or some other quantity.
        # Either way, not a valid probability.
        assert _parse_probability("150") is None

    def test_negative_value_rejected(self):
        assert _parse_probability("-0.5") is None

    def test_unparseable_string_returns_none(self):
        assert _parse_probability("not a number") is None
        assert _parse_probability("") is None

    def test_zero_and_one_boundaries_accepted(self):
        assert _parse_probability("0") == pytest.approx(0.0)
        assert _parse_probability("0.0") == pytest.approx(0.0)
        assert _parse_probability("1") == pytest.approx(1.0)
        assert _parse_probability("1.0") == pytest.approx(1.0)

    def test_one_hundred_percent_accepted(self):
        assert _parse_probability("100%") == pytest.approx(1.0)
        assert _parse_probability("100") == pytest.approx(1.0)

    def test_zero_percent_accepted(self):
        # 0% explicit is a real corner: divide by 100 still yields 0.
        assert _parse_probability("0%") == pytest.approx(0.0)

    def test_warning_emitted_when_skipping_ambiguous(self, caplog):
        # WARNING-level log when we drop an out-of-range value — operators
        # should be able to grep for these to spot upstream parse drift.
        import logging

        with caplog.at_level(logging.WARNING):
            assert _parse_probability("1.2") is None
        # We don't strictly require a specific message format, just that
        # something was logged at WARNING when we dropped a parseable but
        # implausible value.
        assert any("1.2" in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# F12 — fall-aib-2025 fixture: the binary Platt fit was unstable between
# fall-aib-2025 and spring-aib-2026 (slope drift > 0.3). The Platt plan
# documented an >=80% parse-rate stability gate. Lock in the parser's
# performance on a representative fall-aib-2025 comment so we catch any
# regression that would silently invalidate the fit.
# ---------------------------------------------------------------------------


# Synthesized from a real fall-aib-2025 binary comment shape (post 41137-era):
# 6-model ensemble with the older roster (gpt-5.1, o3, claude-sonnet-4.5,
# grok-4.1-fast, qwen3-235b-a22b-thinking-2507, kimi-k2-0905). Structurally
# identical to a current-vintage comment, just with different model paths
# and slightly older summary header content.
FALL_AIB_2025_FIXTURE = """# SUMMARY
*Question*: Will a sixth contentious case be opened at the International Court of Justice in 2025?
*Final Prediction*: 21.0%
*Total Cost*: $0.0775
*Time Spent*: 8.64 minutes


## Report 1 Summary
### Forecasts
*Forecaster 1*: 25.0%
*Forecaster 2*: 33.0%
*Forecaster 3*: 22.0%
*Forecaster 4*: 10.0%
*Forecaster 5*: 20.0%
*Forecaster 6*: 10.0%


### Research Summary
Brief research summary covering recent ICJ news.

================================================================================
FORECAST SECTION:

## R1: Forecaster 1 Reasoning
Model: openrouter/openai/gpt-5.1

Analysis text. Final probability: 25%

## R1: Forecaster 2 Reasoning
Model: openrouter/openai/o3

Analysis text. Final probability: 33%

## R1: Forecaster 3 Reasoning
Model: openrouter/anthropic/claude-sonnet-4.5

Analysis text. Final probability: 22%

## R1: Forecaster 4 Reasoning
Model: openrouter/x-ai/grok-4.1-fast

Analysis text. Final probability: 10%

## R1: Forecaster 5 Reasoning
Model: openrouter/qwen/qwen3-235b-a22b-thinking-2507

Analysis text. Final probability: 20%

## R1: Forecaster 6 Reasoning
Model: openrouter/moonshotai/kimi-k2-0905

Analysis text. Final probability: 10%

<!-- STACKED=false -->
"""


class TestFallAib2025Fixture:
    def test_parse_rate_meets_eighty_percent_threshold(self):
        # All 6 bullets must resolve to NAMED model keys (no anonymized
        # ``Forecaster N`` placeholders). Parse rate = 6/6 = 100%, well
        # above the 80% gate.
        result = parse_per_model_forecasts(FALL_AIB_2025_FIXTURE)
        named_keys = [k for k in result if not k.startswith("Forecaster ")]
        total = len(result)
        assert total == 6, f"Expected 6 bullets, parsed {total}"
        parse_rate = len(named_keys) / total
        assert parse_rate >= 0.80, f"Parse rate {parse_rate:.0%} below 80% gate"

    def test_old_roster_model_names_extracted(self):
        # Specific check: the older roster names must come through cleanly.
        result = parse_per_model_forecasts(FALL_AIB_2025_FIXTURE)
        expected_models = {
            "gpt-5.1",
            "o3",
            "claude-sonnet-4.5",
            "grok-4.1-fast",
            "qwen3-235b-a22b-thinking-2507",
            "kimi-k2-0905",
        }
        assert set(result.keys()) == expected_models

    def test_values_attributed_correctly(self):
        result = parse_per_model_forecasts(FALL_AIB_2025_FIXTURE)
        assert result["gpt-5.1"] == "25.0%"
        assert result["o3"] == "33.0%"
        assert result["claude-sonnet-4.5"] == "22.0%"
        assert result["grok-4.1-fast"] == "10.0%"
        assert result["qwen3-235b-a22b-thinking-2507"] == "20.0%"
        assert result["kimi-k2-0905"] == "10.0%"

    def test_probabilities_parse_to_valid_range(self):
        # The full Platt fit pipeline runs _parse_probability on each value;
        # confirm none of them get dropped by the F11 heuristic tightening.
        result = parse_per_model_forecasts(FALL_AIB_2025_FIXTURE)
        parsed = {k: _parse_probability(v) for k, v in result.items()}
        for model, prob in parsed.items():
            assert prob is not None, f"{model} dropped by parser"
            assert 0.0 <= prob <= 1.0

    def test_legacy_stacked_marker_parsed(self):
        # Fall-aib-2025 comments use the legacy STACKED= marker. Confirm
        # the marker reader still picks it up.
        assert parse_stacked_marker(FALL_AIB_2025_FIXTURE) is False


# ---------------------------------------------------------------------------
# parse_per_model_mc_option_probs — MC full option vector extraction
# ---------------------------------------------------------------------------


class TestParsePerModelMcOptionProbs:
    """Extracts per-forecaster option probability vectors from MC comment bullets.

    The bug: parse_per_model_forecasts only captures the first line after
    ``*Forecaster N*:``, which for MC is just the first option (e.g.
    ``- Option A: 40.0%``). This parser captures ALL option lines per
    forecaster and returns ``{model: {option: probability}}``.
    """

    def test_basic_mc_two_forecasters(self):
        from metaculus_bot.performance_analysis.parsing import parse_per_model_mc_option_probs

        comment = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: \n"
            "- Option A: 40.0%\n"
            "- Option B: 30.0%\n"
            "- Option C: 20.0%\n"
            "- Option D: 10.0%\n"
            "\n"
            "*Forecaster 2 (claude-opus-4.7)*: \n"
            "- Option A: 35.0%\n"
            "- Option B: 25.0%\n"
            "- Option C: 25.0%\n"
            "- Option D: 15.0%\n"
            "\n"
            "### Research Summary\n"
            "Some research here\n"
        )
        result = parse_per_model_mc_option_probs(comment)
        assert result == {
            "gpt-5.5": {"Option A": 0.40, "Option B": 0.30, "Option C": 0.20, "Option D": 0.10},
            "claude-opus-4.7": {"Option A": 0.35, "Option B": 0.25, "Option C": 0.25, "Option D": 0.15},
        }

    def test_returns_empty_for_binary_comment(self):
        from metaculus_bot.performance_analysis.parsing import parse_per_model_mc_option_probs

        comment = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: 72.0%\n"
            "*Forecaster 2 (claude-opus-4.7)*: 68.0%\n"
            "\n"
            "### Research Summary\n"
            "Some research here\n"
        )
        result = parse_per_model_mc_option_probs(comment)
        assert result == {}

    def test_empty_comment(self):
        from metaculus_bot.performance_analysis.parsing import parse_per_model_mc_option_probs

        assert parse_per_model_mc_option_probs("") == {}

    def test_options_with_special_characters_in_name(self):
        from metaculus_bot.performance_analysis.parsing import parse_per_model_mc_option_probs

        comment = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: \n"
            "- Yes (>50%): 60.0%\n"
            "- No (<=50%): 30.0%\n"
            "- Ambiguous / unclear: 10.0%\n"
            "\n"
            "### Research Summary\n"
        )
        result = parse_per_model_mc_option_probs(comment)
        assert result == {
            "gpt-5.5": {"Yes (>50%)": 0.60, "No (<=50%)": 0.30, "Ambiguous / unclear": 0.10},
        }

    def test_model_map_from_rationales(self):
        """Attribution uses Model: lines from R1 sections when inline name is absent."""
        from metaculus_bot.performance_analysis.parsing import parse_per_model_mc_option_probs

        comment = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1*: \n"
            "- Yes: 70.0%\n"
            "- No: 30.0%\n"
            "\n"
            "### Research Summary\n"
            "stuff\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/openai/gpt-5.5\n\n"
            "reasoning here\n"
        )
        result = parse_per_model_mc_option_probs(comment)
        assert result == {"gpt-5.5": {"Yes": 0.70, "No": 0.30}}

    def test_probabilities_sum_approximately_one(self):
        """Sanity: parsed probabilities should sum to ~1.0 for well-formed comments."""
        from metaculus_bot.performance_analysis.parsing import parse_per_model_mc_option_probs

        comment = (
            "## Report 1 Summary\n"
            "### Forecasts\n"
            "*Forecaster 1 (gpt-5.5)*: \n"
            "- A: 25.0%\n"
            "- B: 25.0%\n"
            "- C: 25.0%\n"
            "- D: 25.0%\n"
            "\n"
            "### Research Summary\n"
        )
        result = parse_per_model_mc_option_probs(comment)
        total = sum(result["gpt-5.5"].values())
        assert abs(total - 1.0) < 0.01
