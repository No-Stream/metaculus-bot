"""Tests for build_parse_notes — the numeric percentile parser instruction builder.

The parser's only job is to EXTRACT numbers from the forecaster's free text (and
convert unit suffixes like 350B → base unit). On an OPEN bound the displayed range
is only a suggestion; the outcome can resolve outside it, so the parser must
preserve out-of-range values verbatim and NEVER clamp them into range. On a CLOSED
bound the outcome genuinely cannot exceed the limit, so a hard sanity note is fine.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from forecasting_tools import NumericQuestion

from metaculus_bot.forecaster_runners import build_parse_notes


def _numeric_question(
    *,
    lower: float,
    upper: float,
    open_lower: bool,
    open_upper: bool,
    unit: str = "USD",
) -> NumericQuestion:
    q = MagicMock(spec=NumericQuestion)
    q.lower_bound = lower
    q.upper_bound = upper
    q.open_lower_bound = open_lower
    q.open_upper_bound = open_upper
    q.unit_of_measure = unit
    return q


class TestBuildParseNotes:
    def test_parse_notes_open_lower_no_clamp(self):
        """Open lower bound: no clamp clause, no keep->=-lower note; permits and preserves below-range values."""
        q = _numeric_question(lower=75_000_000, upper=150_000_000, open_lower=True, open_upper=False)
        notes = build_parse_notes(q)

        # No clamp-into-range instruction.
        assert "within [" not in notes
        # No "keep values at or above the lower bound" instruction for an open lower bound.
        assert "at or above" not in notes
        # Explicitly permits below-range values and demands verbatim extraction.
        assert "resolve below" in notes
        assert "verbatim" in notes
        assert "never clamp" in notes.lower()

    def test_parse_notes_open_upper_no_clamp(self):
        """Open upper bound: symmetric — no clamp, permits and preserves above-range values."""
        q = _numeric_question(lower=75_000_000, upper=150_000_000, open_lower=False, open_upper=True)
        notes = build_parse_notes(q)

        assert "within [" not in notes
        # No "keep values at or below the upper bound" instruction for an open upper bound.
        assert "at or below" not in notes
        assert "resolve above" in notes
        assert "verbatim" in notes
        assert "never clamp" in notes.lower()

    def test_parse_notes_closed_bounds_still_constrain(self):
        """Fully closed bounds: retain hard bound language on both sides, no verbatim/out-of-range clause."""
        q = _numeric_question(lower=75_000_000, upper=150_000_000, open_lower=False, open_upper=False)
        notes = build_parse_notes(q)

        assert "at or above" in notes
        assert "at or below" in notes
        assert "resolve below" not in notes
        assert "resolve above" not in notes

    def test_parse_notes_open_both_bounds(self):
        """Both bounds open: permits out-of-range values on both sides, no hard bound language."""
        q = _numeric_question(lower=75_000_000, upper=150_000_000, open_lower=True, open_upper=True)
        notes = build_parse_notes(q)

        assert "resolve below" in notes
        assert "resolve above" in notes
        assert "at or above" not in notes
        assert "at or below" not in notes
        assert "within [" not in notes

    def test_parse_notes_includes_unit_and_conversion(self):
        """Base-unit and B/M/k conversion instructions are present regardless of bound openness."""
        for open_lower, open_upper in [(False, False), (True, False), (False, True), (True, True)]:
            q = _numeric_question(
                lower=75_000_000,
                upper=150_000_000,
                open_lower=open_lower,
                open_upper=open_upper,
                unit="widgets",
            )
            notes = build_parse_notes(q)

            assert "widgets" in notes
            assert "base unit" in notes
            assert "350B" in notes
            assert "350000000000" in notes
