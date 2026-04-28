"""Tests for comment_trimming.

The structured trim path (preserve summary head + tail) protects
residual-analysis parsers that extract per-forecaster predictions and the
STACKED marker from bot comments. Breaking this silently would degrade every
future performance-analysis pull, so the tests here combine unit coverage,
end-to-end coverage against realistic comment structure, and an explicit
parser-compatibility check that runs the live parsing.py regexes.
"""

import math

import pytest

from metaculus_bot.comment_trimming import TRIM_NOTICE, TrimConfig, trim_comment, trim_section
from metaculus_bot.constants import COMMENT_CHAR_LIMIT, REPORT_SECTION_CHAR_LIMIT

# ---------------------------------------------------------------------------
# Legacy section trim (unchanged behavior — keeps header, uses tail)
# ---------------------------------------------------------------------------


def test_trim_section_preserves_header_and_tail() -> None:
    header = "## Report 1 Summary"
    body_length = REPORT_SECTION_CHAR_LIMIT + 512
    body = "A" * body_length
    original = f"{header}\n{body}"

    trimmed = trim_section(original, "unit-test-section")

    assert trimmed.splitlines()[0] == header
    assert TRIM_NOTICE in trimmed.splitlines()[1]
    assert len(trimmed) == REPORT_SECTION_CHAR_LIMIT

    available = REPORT_SECTION_CHAR_LIMIT - len(header) - len(TRIM_NOTICE) - 2
    assert available > 0
    expected_tail = body[-available:]
    assert trimmed.endswith(expected_tail)


def test_trim_section_noop_when_within_limit() -> None:
    original = "## Report 1 Summary\nshort body"
    trimmed = trim_section(original, "unit-test-noop")
    assert trimmed == original


# ---------------------------------------------------------------------------
# Fallback: tail-only trim when the summary marker is absent
# ---------------------------------------------------------------------------


def test_trim_final_comment_falls_back_to_tail_when_no_marker() -> None:
    # Payload without "### Research Summary" must take the fallback path.
    payload = "PREFIX\n" + ("0123456789" * math.ceil((COMMENT_CHAR_LIMIT + 250) / 10))
    trimmed = trim_comment(payload)

    assert trimmed.startswith(TRIM_NOTICE)
    tail_length = COMMENT_CHAR_LIMIT - len(TRIM_NOTICE) - 1
    assert tail_length > 0
    assert len(trimmed) == COMMENT_CHAR_LIMIT
    assert trimmed.endswith(payload[-tail_length:])


def test_trim_final_comment_noop_when_short() -> None:
    payload = "Concise explanation"
    trimmed = trim_comment(payload)
    assert trimmed == payload


# ---------------------------------------------------------------------------
# Structured trim (new path): preserves summary + tail, drops middle
# ---------------------------------------------------------------------------


def _build_realistic_comment(
    *,
    summary_bullets: list[tuple[int, str, str]],  # (idx, model, value)
    rationale_filler_per_model: int,
    num_rationales: int | None = None,
    research_filler: int = 0,
    stacked: bool | None = None,
) -> str:
    """Build a bot comment that mirrors the real production structure.

    Structure matches main.TemplateForecaster._create_unified_explanation:
      # SUMMARY
      ## Report 1 Summary
      ### Forecasts
        *Forecaster N*: value  (one bullet per ensemble member)
      ### Research Summary
        ...research body...
      ## R1: Forecaster N Reasoning
        Model: openrouter/.../<name>
        ...reasoning body...
      <!-- STACKED=<bool> -->   (optional, only for stacking strategies)
    """
    if num_rationales is None:
        num_rationales = len(summary_bullets)

    lines = [
        "# SUMMARY",
        "*Question*: will X?",
        "",
        "## Report 1 Summary",
        "### Forecasts",
    ]
    for idx, _, value in summary_bullets:
        lines.append(f"*Forecaster {idx}*: {value}")
    lines.append("")
    lines.append("### Research Summary")
    if research_filler:
        lines.append("research_token " * research_filler)
    lines.append("")
    for idx, model, _ in summary_bullets[:num_rationales]:
        lines.append(f"## R1: Forecaster {idx} Reasoning")
        lines.append(f"Model: openrouter/provider/{model}")
        lines.append("")
        lines.append("reasoning_token " * rationale_filler_per_model)
        lines.append("")
    if stacked is True:
        lines.append("<!-- STACKED=true -->")
    elif stacked is False:
        lines.append("<!-- STACKED=false -->")
    return "\n".join(lines)


SUMMARY_BULLETS = [
    (1, "gpt-5.2", "73.0%"),
    (2, "gpt-5.4", "68.0%"),
    (3, "claude-opus-4.6", "71.0%"),
    (4, "claude-opus-4.5", "70.0%"),
    (5, "gemini-3.1-pro-preview", "65.0%"),
    (6, "grok-4.1-fast", "74.0%"),
]


class TestStructuredTrim:
    def test_noop_when_within_limit(self) -> None:
        comment = _build_realistic_comment(
            summary_bullets=SUMMARY_BULLETS,
            rationale_filler_per_model=100,
            stacked=True,
        )
        assert len(comment) < COMMENT_CHAR_LIMIT
        assert trim_comment(comment) == comment

    def test_preserves_summary_and_tail_when_overflowing(self) -> None:
        # Inflate the research section so the comment exceeds the limit.
        comment = _build_realistic_comment(
            summary_bullets=SUMMARY_BULLETS,
            rationale_filler_per_model=800,
            research_filler=25_000,  # ~375k chars of filler, forces overflow
            stacked=True,
        )
        assert len(comment) > COMMENT_CHAR_LIMIT, "test precondition: must overflow"

        trimmed = trim_comment(comment)

        # Every summary bullet must survive.
        for idx, _, value in SUMMARY_BULLETS:
            assert f"*Forecaster {idx}*: {value}" in trimmed, f"Forecaster {idx} summary bullet lost"

        # The trim notice must be present exactly once (signalling the middle gap).
        assert trimmed.count(TRIM_NOTICE) == 1

        # The STACKED marker must survive (it's at the tail).
        assert "<!-- STACKED=true -->" in trimmed

        # Total length must respect the comment limit.
        assert len(trimmed) <= COMMENT_CHAR_LIMIT

    def test_stacked_false_marker_preserved(self) -> None:
        comment = _build_realistic_comment(
            summary_bullets=SUMMARY_BULLETS,
            rationale_filler_per_model=800,
            research_filler=25_000,
            stacked=False,
        )
        trimmed = trim_comment(comment)
        assert "<!-- STACKED=false -->" in trimmed

    def test_missing_stacked_marker_ok(self) -> None:
        # Non-stacking strategies emit no marker; trim must still succeed.
        comment = _build_realistic_comment(
            summary_bullets=SUMMARY_BULLETS,
            rationale_filler_per_model=800,
            research_filler=25_000,
            stacked=None,
        )
        assert len(comment) > COMMENT_CHAR_LIMIT
        trimmed = trim_comment(comment)
        assert TRIM_NOTICE in trimmed
        # All summary bullets still present.
        for idx, _, value in SUMMARY_BULLETS:
            assert f"*Forecaster {idx}*: {value}" in trimmed

    def test_head_over_budget_falls_back_to_tail_only(self) -> None:
        # Absurdly-long summary head — e.g. a runaway model dump before
        # Research Summary. The head exceeds head_budget, so we fall back
        # to the plain tail-only trim path rather than blow the limit.
        cfg = TrimConfig(head_budget=500)  # tight head cap to force fallback
        bloated_summary = "*Forecaster 1*: 50%\n" + ("x" * 10_000)
        comment = (
            f"# SUMMARY\n{bloated_summary}\n### Research Summary\n" + ("body " * 40_000) + "\n<!-- STACKED=true -->\n"
        )
        assert len(comment) > cfg.comment_limit
        trimmed = trim_comment(comment, config=cfg)
        # Fallback path starts with the notice, keeps the final tail.
        assert trimmed.startswith(TRIM_NOTICE)
        assert len(trimmed) <= cfg.comment_limit

    def test_middle_gets_dropped_not_head_or_tail(self) -> None:
        # Build a comment where the middle contains a unique marker that
        # must NOT appear in the trimmed output, while head and tail
        # markers MUST appear.
        head_sentinel = "HEAD_SENTINEL_XYZ"
        middle_sentinel = "MIDDLE_SENTINEL_XYZ"
        tail_sentinel = "TAIL_SENTINEL_XYZ"
        comment = (
            f"# SUMMARY\n*Forecaster 1*: 50%\n{head_sentinel}\n"
            f"### Research Summary\n"
            + middle_sentinel
            + "\n"
            + ("padding " * 40_000)
            + "\n"
            + tail_sentinel
            + "\n<!-- STACKED=true -->\n"
        )
        assert len(comment) > COMMENT_CHAR_LIMIT
        trimmed = trim_comment(comment)
        assert head_sentinel in trimmed
        assert tail_sentinel in trimmed
        assert middle_sentinel not in trimmed


# ---------------------------------------------------------------------------
# End-to-end: trimmed comments remain parseable by residual-analysis regexes
# ---------------------------------------------------------------------------


class TestParserCompatibility:
    """Guard against silent breakage of the parsers that consume bot comments.

    If a trim reduces what these parsers can extract, historical performance
    analyses become inconsistent with live ones, which was the specific
    motivation for keeping the summary head intact.
    """

    def _comment(self, *, stacked: bool | None = True) -> str:
        return _build_realistic_comment(
            summary_bullets=SUMMARY_BULLETS,
            rationale_filler_per_model=600,
            research_filler=25_000,
            stacked=stacked,
        )

    def test_parse_per_model_forecasts_survives_trim(self) -> None:
        from metaculus_bot.performance_analysis.parsing import parse_per_model_forecasts

        comment = self._comment()
        assert len(comment) > COMMENT_CHAR_LIMIT
        trimmed = trim_comment(comment)

        # Original extraction — baseline
        original_parsed = parse_per_model_forecasts(comment)
        trimmed_parsed = parse_per_model_forecasts(trimmed)

        assert original_parsed, "precondition: parser must find something in the original"
        # Trim must not lose any model's summary forecast.
        assert trimmed_parsed == original_parsed, (
            f"Trim broke per-model parsing. orig={original_parsed} trimmed={trimmed_parsed}"
        )

    def test_parse_stacked_marker_survives_trim(self) -> None:
        from metaculus_bot.performance_analysis.parsing import parse_stacked_marker

        for flag in (True, False):
            comment = self._comment(stacked=flag)
            assert len(comment) > COMMENT_CHAR_LIMIT
            trimmed = trim_comment(comment)
            assert parse_stacked_marker(trimmed) is flag, f"STACKED={flag} marker lost during trim"

    def test_parse_forecaster_model_map_at_least_partial(self) -> None:
        # This one's more permissive: the R1 rationale blocks sit in the
        # middle of the comment and may be dropped by trimming. We don't
        # require ALL of them to survive — just that whatever survives
        # uses real model names (not anonymized fallbacks). Losing model
        # attribution silently would be a regression; missing some
        # rationales entirely is acceptable and already handled by
        # anonymized fallback in the parser.
        from metaculus_bot.performance_analysis.parsing import parse_forecaster_model_map

        comment = self._comment()
        trimmed = trim_comment(comment)
        # The tail-preserving trim is likely to retain the LAST few
        # rationales. Whatever it retains must have valid model names.
        trimmed_map = parse_forecaster_model_map(trimmed)
        for idx, name in trimmed_map.items():
            assert name and not name.startswith("Forecaster"), f"Forecaster {idx} attribution degraded to '{name}'"


# ---------------------------------------------------------------------------
# Regression guards on real historical comments
# ---------------------------------------------------------------------------


class TestAgainstRealHistoricalData:
    """Run the trim on actual comments from scratch/analysis_2026-04/.

    These comments are the exact strings the bot emitted and Metaculus stored;
    trimming them shouldn't produce behavior meaningfully different from what
    production emitted. This catches environment-specific edge cases that
    synthetic comments might miss.
    """

    def _load_real_comments(self) -> list[str]:
        """Return only the string comment_text values from the Q2 dataset."""
        import json
        from pathlib import Path

        path = Path(__file__).parent.parent / "scratch" / "analysis_2026-04" / "performance_data.json"
        if not path.exists():
            pytest.skip(f"real data not available at {path}")
        with path.open() as f:
            records = json.load(f)
        return [r["comment_text"] for r in records if isinstance(r.get("comment_text"), str)]

    def test_no_real_comment_gets_worse_parsing_after_re_trim(self) -> None:
        """Re-trimming an already-trimmed comment must not make things worse.

        The stored comments were trimmed by production before being saved.
        Passing them through trim_comment again should be a no-op (they're
        all ≤ limit). But even if a future change altered the limit, the
        parser output on the re-trimmed version should not REGRESS vs. the
        stored version.
        """
        from metaculus_bot.performance_analysis.parsing import (
            parse_forecaster_model_map,
            parse_per_model_forecasts,
            parse_stacked_marker,
        )

        comments = self._load_real_comments()
        assert comments, "precondition: some real comments must exist"

        for text in comments:
            if len(text) > COMMENT_CHAR_LIMIT:
                pytest.fail(f"Stored comment longer than limit ({len(text)} > {COMMENT_CHAR_LIMIT}) — data corrupt?")

            trimmed = trim_comment(text)
            assert trimmed == text, "trim_comment must be a no-op on comments already within the limit"

            # Parsers behave identically (trivially, since no-op).
            assert parse_per_model_forecasts(trimmed) == parse_per_model_forecasts(text)
            assert parse_stacked_marker(trimmed) == parse_stacked_marker(text)
            assert parse_forecaster_model_map(trimmed) == parse_forecaster_model_map(text)

    def test_inflated_real_comment_still_parses_after_trim(self) -> None:
        """Take a real comment, inflate it past the limit, and confirm the
        trim preserves everything the parsers need."""
        from metaculus_bot.performance_analysis.parsing import (
            parse_per_model_forecasts,
            parse_stacked_marker,
        )

        comments = self._load_real_comments()

        # The Q2 dataset always contains multiple healthy large comments with
        # the expected structure. If this filter returns nothing, it's a real
        # regression (data format change, trim limit change, or dataset
        # corruption) — not a test-environment issue — so assert.
        original = next(
            (
                c
                for c in comments
                if len(c) > 90_000
                and "[... trimmed for length]" not in c
                and "### Research Summary" in c
                and "*Forecaster 1*:" in c
            ),
            None,
        )
        assert original is not None, (
            "No healthy ≥90k-char comment with expected structure found in Q2 dataset. "
            "If the dataset is intact, this indicates a structural regression in the comment format."
        )

        original_forecasts = parse_per_model_forecasts(original)
        original_stacked = parse_stacked_marker(original)
        assert original_forecasts, "precondition: original must have parseable forecasts"

        # Inflate the research section until the comment overflows
        marker = "### Research Summary"
        idx = original.find(marker)
        assert idx >= 0
        inflated = (
            original[: idx + len(marker)] + "\n" + ("filler_token " * 15_000) + "\n" + original[idx + len(marker) :]
        )
        assert len(inflated) > COMMENT_CHAR_LIMIT, "precondition: inflated must overflow"

        trimmed = trim_comment(inflated)
        assert len(trimmed) <= COMMENT_CHAR_LIMIT

        # Summary forecasts must round-trip unchanged.
        assert parse_per_model_forecasts(trimmed) == original_forecasts
        # STACKED marker behavior must match (tail is preserved either way).
        assert parse_stacked_marker(trimmed) == original_stacked
