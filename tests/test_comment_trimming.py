"""Tests for comment.trimming.

The structured trim path (preserve summary head + tail) protects
residual-analysis parsers that extract per-forecaster predictions and the
STACKED marker from bot comments. Breaking this silently would degrade every
future performance-analysis pull, so the tests here combine unit coverage,
end-to-end coverage against realistic comment structure, and an explicit
parser-compatibility check that runs the live parsing.py regexes.
"""

import math

import pytest

from metaculus_bot.comment.trimming import TRIM_NOTICE, TrimConfig, trim_comment, trim_section
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
    # Payload with neither "### Research Summary" nor # RESEARCH / # FORECASTS
    # structure must take the last-resort header-preserving fallback. This is
    # the exact crash scenario from the 2026-06-05 log: a "# SUMMARY ..." blob
    # with no recognizable sections that overflows the limit. The fallback must
    # preserve the leading "#" line so the framework's
    # validate_explanation_starts_with_hash validator never rejects it.
    header = "# SUMMARY"
    payload = header + "\n" + ("0123456789" * math.ceil((COMMENT_CHAR_LIMIT + 250) / 10))
    trimmed = trim_comment(payload)

    assert trimmed.lstrip().startswith("#"), "output must remain '#'-leading (validator invariant)"
    assert trimmed.startswith(header)
    assert TRIM_NOTICE in trimmed
    assert len(trimmed) == COMMENT_CHAR_LIMIT
    # The tail of the original payload must survive after header + notice.
    available = COMMENT_CHAR_LIMIT - len(header) - len(TRIM_NOTICE) - 2
    assert available > 0
    assert trimmed.endswith(payload[-available:])


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

    def test_head_over_budget_falls_back_to_header_and_tail(self) -> None:
        # Absurdly-long summary head — e.g. a runaway model dump before
        # Research Summary — with no # RESEARCH/# FORECASTS structure. The
        # summary_and_tail path bails (head exceeds head_budget), so we fall
        # through to the header-preserving last-resort trim. It must keep the
        # leading "# SUMMARY" line (validator invariant) rather than prepend
        # the notice and drop the header.
        cfg = TrimConfig(head_budget=500)  # tight head cap to force fallback
        bloated_summary = "*Forecaster 1*: 50%\n" + ("x" * 10_000)
        comment = (
            f"# SUMMARY\n{bloated_summary}\n### Research Summary\n" + ("body " * 40_000) + "\n<!-- STACKED=true -->\n"
        )
        assert len(comment) > cfg.comment_limit
        trimmed = trim_comment(comment, config=cfg)
        assert trimmed.startswith("# SUMMARY"), "header-preserving fallback must keep the leading '#' line"
        assert TRIM_NOTICE in trimmed
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
# Section-aware research-first trim (primary strategy)
# ---------------------------------------------------------------------------


def _build_sectioned_comment(
    *,
    summary_body: str,
    research_body: str,
    forecasts_body: str,
    trailing_markers: str = "<!-- STACKED=true -->\n<!-- TOOLS_USED=false -->",
) -> str:
    """Mirror the framework's unified comment: # SUMMARY / # RESEARCH / # FORECASTS.

    Matches forecast_bot.py:538-550 — the comment always opens with # SUMMARY
    and carries # RESEARCH and # FORECASTS as top-level (h1) sections, with the
    residual-analysis markers trailing after the FORECASTS rationales.
    """
    return (
        "# SUMMARY\n"
        "*Question*: will X?\n\n"
        "## Report 1 Summary\n"
        "### Forecasts\n"
        f"{summary_body}\n\n"
        "### Research Summary\n"
        "_Full research in the RESEARCH section below._\n\n"
        "# RESEARCH\n"
        "## Report 1 Research\n"
        f"{research_body}\n\n"
        "# FORECASTS\n"
        f"{forecasts_body}\n"
        f"{trailing_markers}\n"
    )


class TestResearchFirstTrim:
    """The primary strategy: shrink # RESEARCH before SUMMARY/FORECASTS."""

    def test_research_shrunk_first_summary_and_forecasts_survive(self) -> None:
        summary = "\n".join(f"*Forecaster {i}*: {60 + i}.0%" for i in range(1, 7))
        # Sentinel sits at the END of the research body — research is summary-
        # style and front-loaded, so the trim keeps the front and drops the
        # tail, taking this sentinel with it.
        research = ("research_token " * 30_000) + " RESEARCH_SENTINEL"  # forces overflow
        forecasts = "FORECASTS_SENTINEL\n" + "\n".join(
            f"## R1: Forecaster {i} Reasoning\nModel: openrouter/provider/m{i}\nrationale body" for i in range(1, 7)
        )
        comment = _build_sectioned_comment(summary_body=summary, research_body=research, forecasts_body=forecasts)
        assert len(comment) > COMMENT_CHAR_LIMIT, "precondition: must overflow"

        trimmed = trim_comment(comment)

        assert len(trimmed) <= COMMENT_CHAR_LIMIT
        assert trimmed.lstrip().startswith("#")
        # SUMMARY survives whole: every bullet + the Research Summary marker.
        assert "### Research Summary" in trimmed
        for i in range(1, 7):
            assert f"*Forecaster {i}*: {60 + i}.0%" in trimmed
        # FORECASTS survives whole.
        assert "FORECASTS_SENTINEL" in trimmed
        for i in range(1, 7):
            assert f"## R1: Forecaster {i} Reasoning" in trimmed
        # Research middle is sacrificed.
        assert "RESEARCH_SENTINEL" not in trimmed
        assert TRIM_NOTICE in trimmed
        # Trailing residual-analysis markers survive.
        assert "<!-- STACKED=true -->" in trimmed
        assert "<!-- TOOLS_USED=false -->" in trimmed

    def test_summary_and_tail_fallback_when_head_plus_tail_overflow(self) -> None:
        # When head + tail alone overflow (massive FORECASTS body), research-
        # first bails (research_budget < 0) and the summary-and-tail path takes
        # over. The trim must still produce a valid '#'-leading string within
        # the limit, preserving the SUMMARY head and the trailing markers.
        summary = "\n".join(f"*Forecaster {i}*: {60 + i}.0%" for i in range(1, 7))
        research = "research " * 5_000
        forecasts = "FORECASTS_HEAD\n" + ("fcast_token " * 30_000)  # huge tail
        comment = _build_sectioned_comment(summary_body=summary, research_body=research, forecasts_body=forecasts)
        assert len(comment) > COMMENT_CHAR_LIMIT

        trimmed = trim_comment(comment)

        assert len(trimmed) <= COMMENT_CHAR_LIMIT
        assert trimmed.lstrip().startswith("#")
        # SUMMARY is preserved even in this extreme case.
        assert "### Research Summary" in trimmed
        for i in range(1, 7):
            assert f"*Forecaster {i}*: {60 + i}.0%" in trimmed
        # Research body is gone entirely.
        assert "research research" not in trimmed
        assert TRIM_NOTICE in trimmed
        # Trailing residual-analysis markers (at the very end) survive.
        assert "<!-- STACKED=true -->" in trimmed
        assert "<!-- TOOLS_USED=false -->" in trimmed


# ---------------------------------------------------------------------------
# Leading-'#' invariant: for any '#'-leading input (the framework's contract —
# every comment opens with "# SUMMARY"), trim_comment's output also starts with
# '#', so validate_explanation_starts_with_hash can never reject it. This is the
# contract that makes the 2026-06-05 crash structurally impossible.
# ---------------------------------------------------------------------------


def _invariant_shapes() -> list[str]:
    big = COMMENT_CHAR_LIMIT + 50_000
    return [
        # Sectioned, with Research Summary marker, just over the limit.
        _build_sectioned_comment(
            summary_body="*Forecaster 1*: 50%",
            research_body="r " * (big // 2),
            forecasts_body="rationale",
        ),
        # Sectioned, no trailing markers.
        _build_sectioned_comment(
            summary_body="*Forecaster 1*: 50%",
            research_body="r " * (big // 2),
            forecasts_body="rationale",
            trailing_markers="",
        ),
        # Has ### Research Summary but no # RESEARCH/# FORECASTS structure.
        "# SUMMARY\n*Forecaster 1*: 50%\n### Research Summary\n" + ("body " * (big // 5)),
        # The exact crash shape: # SUMMARY blob, no markers, overflowing.
        "# SUMMARY\n" + ("x" * big),
        # Leading "\n# SUMMARY..." with no sections — the framework's
        # clean_indents output starts with a newline, so the last-resort trim
        # must skip it and keep the "#" header line.
        "\n# SUMMARY\n" + ("x" * big),
        # Degenerate single-line '# x' that somehow overflows.
        "# " + ("y" * big),
        # No leading newline structure at all (still starts with #).
        "#" + ("z" * big),
    ]


@pytest.mark.parametrize("shape", _invariant_shapes())
def test_trim_comment_output_always_starts_with_hash(shape: str) -> None:
    out = trim_comment(shape)
    assert out.lstrip().startswith("#"), f"invariant violated for shape starting {shape[:40]!r}"
    assert len(out) <= COMMENT_CHAR_LIMIT


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

    def test_every_real_comment_inflated_holds_invariant_and_parses(self) -> None:
        """Replay over ALL stored comments, not just one hand-picked example.

        For every real comment in the Q2 dataset: inflate its research section
        past the limit, trim, and assert the trimmed output (a) stays within the
        limit, (b) starts with '#' (the validator invariant — the exact contract
        whose violation crashed Q578/Q20683), and (c) round-trips the per-model
        forecasts and STACKED marker that residual analysis depends on. This is
        the closest thing to a production replay without spending API credits.
        """
        from metaculus_bot.performance_analysis.parsing import (
            parse_per_model_forecasts,
            parse_stacked_marker,
        )

        comments = self._load_real_comments()
        assert comments, "precondition: some real comments must exist"

        marker = "### Research Summary"
        checked = 0
        for original in comments:
            # Only comments with the structured marker are inflatable in a way
            # that mirrors production; skip the rare ones without it (already
            # covered by the synthetic no-marker invariant test).
            idx = original.find(marker)
            if idx < 0:
                continue

            original_forecasts = parse_per_model_forecasts(original)
            original_stacked = parse_stacked_marker(original)

            inflated = (
                original[: idx + len(marker)] + "\n" + ("filler_token " * 15_000) + "\n" + original[idx + len(marker) :]
            )
            assert len(inflated) > COMMENT_CHAR_LIMIT, "precondition: inflated must overflow"

            trimmed = trim_comment(inflated)

            assert len(trimmed) <= COMMENT_CHAR_LIMIT, f"trim exceeded limit ({len(trimmed)})"
            assert trimmed.lstrip().startswith("#"), "validator invariant violated on a real comment"
            # Per-model forecasts in the SUMMARY head must survive unchanged.
            if original_forecasts:
                assert parse_per_model_forecasts(trimmed) == original_forecasts
            # STACKED marker (tail) must survive unchanged.
            assert parse_stacked_marker(trimmed) == original_stacked
            checked += 1

        assert checked >= 50, f"expected to exercise dozens of real comments, only hit {checked}"
