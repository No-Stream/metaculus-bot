"""``date_prompt``: the continuous template on the calendar axis.

A date question is a numeric question on the epoch-seconds axis, so the prompt is the numeric
template with its axis-specific slots swapped (docs/prompts.md: shared rules stated once, as named
constants both prompts render). These pins establish that the shared rules DID carry over, that
every numeric-only sentence (units, unit conversion, ``outcome_type``) is gone, and that the
STRUCTURED FORECAST example teaches ISO-8601 strings the extraction ladder actually reads.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.date_axis import as_epoch_question, parse_iso_utc
from metaculus_bot.prompts import _SOFT_CLOCK_RULE, date_prompt
from metaculus_bot.structured_output_schema import parse_structured_payload
from tests.prompt_builders import (
    _date_prompt_text,
    _date_q,
    _extract_last_json_block,
    _flat,
    _numeric_prompt_text,
    _open_upper_date_q,
)


class TestSharedRulesCarryOver:
    """Every rule the numeric prompt states must reach the date prompt through the same constant."""

    _SHARED = (
        "status-quo derivation",
        "open and unresolved as of",
        "post-open event",
        "no qualifying event has yet occurred inside the window",
        "resolution-metric echo (named-series questions only)",
        "proximity to the primary record",
        "[unverified attribution]",
        "read a null search result as a null search result",
        "how many events of a kind occur in a period",
        "forecastability and width",
        "match your interval width to what your reasoning actually supports",
        "keep your extreme tails (p1 and p99) wide enough",
        "anchor on your math",
        "small delta check",
        "bait-and-switch check",
        "consistency line: which percentile corresponds to the status quo or trend",
        "this block is the only authoritative source of your forecast",
    )

    def test_every_shared_rule_lands_once_in_both_prompts(self) -> None:
        date_flat = _flat(_date_prompt_text())
        numeric_flat = _flat(_numeric_prompt_text())
        for phrase in self._SHARED:
            assert date_flat.count(phrase) == numeric_flat.count(phrase) >= 1, phrase

    def test_the_date_prompt_ends_on_the_block_last_instruction(self) -> None:
        assert _date_prompt_text().rstrip().endswith("Write nothing after it.")

    def test_the_soft_clock_rule_ships_to_the_date_prompt(self) -> None:
        """A "when will X happen" question with an announced target is the announced-but-unbound shape
        the rule was measured on, and the mass on the target date IS the timing term it prices; it
        sits in the reference-class step after the count-in-period rule, as in binary and MC."""
        prompt = _date_prompt_text()
        flat = _flat(prompt)
        assert flat.count(_flat(_SOFT_CLOCK_RULE)) == 1
        count_at = prompt.index("For questions asking how many events")
        soft_clock_at = prompt.index("A target date the responsible actor has not bound itself to")
        assert prompt.index("(2) Outside view") < count_at < soft_clock_at < prompt.index("(3) Timeframe and dynamics")
        assert "measured record of meeting" not in _flat(_numeric_prompt_text())


class TestDateSpecificBlocks:
    def test_dates_and_bounds_replaces_units_and_bounds(self) -> None:
        prompt = _date_prompt_text()
        flat = _flat(prompt)
        assert "── dates & bounds ──" in flat
        assert "displayed range: [2026-09-08, 2026-09-19]" in flat
        assert 'a calendar date "yyyy-mm-dd", which means that whole utc day' in flat
        assert (
            'dates after an open upper bound are how you say "this does not happen within the displayed window"' in flat
        )
        # The numeric-only sentences are gone: units, unit conversion, the units final check.
        assert "units & bounds" not in flat
        assert "base units for output values" not in flat
        assert "350b" not in flat
        assert "suggestive of units" not in flat
        assert "units: what are the units of the output values" not in flat
        assert "1788825600" not in prompt, "an epoch float leaked into the date prompt"

    def test_bound_messages_arrive_date_rendered(self) -> None:
        q = _open_upper_date_q()
        prompt = _date_prompt_text(q)
        assert "the outcome can resolve above 2027-03-01T00:00:00Z" in prompt
        assert "1804204800" not in prompt

    def test_status_quo_is_the_not_yet_happened_reading(self) -> None:
        flat = _flat(_date_prompt_text())
        assert "on what date would it resolve?" in flat
        assert (
            "open means x has not happened yet, so the status quo is that it does not happen within the displayed window"
            in flat
        )
        assert "every date you place earlier is a claim that something changes" in flat
        assert flat.count("most recent authoritative measurement") == 1
        assert "what value would it resolve at" not in flat

    def test_tails_forecastability_and_final_check_are_date_worded(self) -> None:
        flat = _flat(_date_prompt_text())
        assert "coherent pathway for an unusually early date" in flat
        assert "coherent pathway for an unusually late date, including not within the displayed window" in flat
        assert "decide how forecastable this date is from current information" in flat
        assert "an event on a binding clock" in flat
        assert 'put the "not within the window" mass beyond an open upper bound' in flat
        assert "calendar check: does every date you output fall on a day the resolution criteria allow" in flat
        assert "unusually low results" not in flat
        assert "center on the current value" not in flat

    def test_no_outcome_type_step_and_final_checks_renumber(self) -> None:
        prompt = _date_prompt_text()
        assert "outcome_type" not in prompt
        assert "Outcome type:" not in prompt
        assert "(9) Final checks" in prompt
        assert "(10)" not in prompt
        numeric = _numeric_prompt_text()
        assert "(9) Outcome type" in numeric
        assert "(10) Final checks" in numeric


class TestDateSchemaBlock:
    _PERCENTILE_KEYS = tuple(f"{p:g}" for p in STANDARD_PERCENTILES)

    def test_example_block_parses_as_a_date_block_the_ladder_reads(self) -> None:
        body = _extract_last_json_block(_date_prompt_text())
        parsed = json.loads(body)
        assert parsed["question_type"] == "date"
        assert set(parsed) == {"question_type", "declared_percentiles"}
        assert list(parsed["declared_percentiles"]) == list(self._PERCENTILE_KEYS)
        block = parse_structured_payload(body, "date")
        assert block is not None, "the example the prompt teaches must pass the block rung's own schema"

    def test_example_values_are_calendar_dates_spanning_the_displayed_range(self) -> None:
        """On a day-granularity grid the example shows ``YYYY-MM-DD`` strings from the displayed lower
        bound to the displayed upper bound, non-decreasing, so the model sees the exact form its grid
        expects rather than an illustrative value from another calendar."""
        parsed = json.loads(_extract_last_json_block(_date_prompt_text()))
        values = list(parsed["declared_percentiles"].values())
        assert values[0] == "2026-09-08"
        assert values[-1] == "2026-09-19"
        assert all(len(v) == len("YYYY-MM-DD") for v in values)
        parsed_dates = [parse_iso_utc(v) for v in values]
        assert parsed_dates == sorted(parsed_dates)

    def test_legacy_fine_grid_example_shows_utc_timestamps(self) -> None:
        parsed = json.loads(_extract_last_json_block(_date_prompt_text(_open_upper_date_q())))
        values = list(parsed["declared_percentiles"].values())
        assert values[0] == "2026-09-01T00:00:00Z"
        assert values[-1] == "2027-03-01T00:00:00Z"
        assert all(v.endswith("Z") for v in values)

    def test_schema_notes_allow_ties_and_forbid_bare_years(self) -> None:
        flat = _flat(_date_prompt_text())
        schema = flat[flat.rfind("structured forecast") :]
        assert "must contain all 13 standard" in schema
        assert schema.count("must contain all") == 1
        assert "non-decreasing across percentiles" in schema
        assert "a repeated date is allowed where your mass concentrates on one day, a decrease is not" in schema
        assert "a bare year, a month, or a number is rejected" in schema
        assert "strictly increasing" not in schema, "13 strictly increasing dates cannot fit a 12-bin grid"
        assert "outcome_type" not in schema
        assert "no scientific notation" not in schema


class TestDatePromptAcceptsEitherView:
    def test_the_date_question_and_its_epoch_view_render_identically(self) -> None:
        q = _date_q()
        view = as_epoch_question(q)
        assert date_prompt(q, "r", "l", "u") == date_prompt(view, "r", "l", "u")

    def test_bound_and_window_dates_do_not_depend_on_the_host_timezone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Every rendered date comes from a tz-aware UTC value; the host clock never enters."""
        import time

        monkeypatch.setenv("TZ", "America/Los_Angeles")
        time.tzset()
        try:
            prompt = _date_prompt_text()
        finally:
            monkeypatch.delenv("TZ")
            time.tzset()
        assert "Displayed range: [2026-09-08, 2026-09-19]" in prompt
        assert datetime.now(UTC).strftime("%Y-%m-%d") in prompt
