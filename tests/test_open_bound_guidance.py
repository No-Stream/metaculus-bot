"""W4: the open-bound guidance must MECHANICALLY teach out-of-bound mass expression.

Percentiles are the single authoritative channel now that the `tails` field is gone.
The prior guidance ("place your lower percentiles at or below {lower}") was too vague —
models kept percentiles in-range and the below-open-bound miss recurred. The strengthened
guidance must: (1) state percentiles are the ONLY channel (no separate tail field),
(2) give the concrete rule that to put N% of mass below an open floor you place that
fraction of percentiles below it — e.g. a ~75%-below belief puts the MEDIAN below the
floor and only upper percentiles inside/above, (3) tell the model not to pile percentiles
at the boundary. Closed bounds must NOT carry this language.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from forecasting_tools import NumericQuestion

from metaculus_bot.numeric.utils import bound_messages
from metaculus_bot.prompts import numeric_prompt, stacking_numeric_prompt

_OPEN = datetime.now() - timedelta(days=30)
_RESOLVE = datetime.now() + timedelta(days=365)


def _open_question(*, open_lower: bool, open_upper: bool) -> NumericQuestion:
    return NumericQuestion(
        id_of_question=91,
        id_of_post=91,
        page_url="https://example.com/q/91",
        question_text="What will the gross be?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
        lower_bound=75_000_000.0,
        upper_bound=150_000_000.0,
        open_lower_bound=open_lower,
        open_upper_bound=open_upper,
        unit_of_measure="$",
        zero_point=None,
    )


class TestBoundMessagesConcreteGuidance:
    def test_open_lower_carries_concrete_percentile_placement_guidance(self):
        _upper, lower = bound_messages(_open_question(open_lower=True, open_upper=True))
        low = " ".join(lower.lower().split())
        # Percentiles are the only channel: mass beyond the range is expressed through them,
        # not through the removed scalar tail-mass field (so the copy must not resurrect it).
        assert "only way" in low or "only channel" in low
        assert "including mass beyond the displayed range" in low
        assert "separate field" not in low
        assert "tail mass" not in low
        # Concrete mechanic: to put mass below the floor, place that fraction of percentiles below it,
        # and a majority-below belief pushes the MEDIAN below the floor.
        assert "median" in low
        assert "below" in low
        # Do not pile percentiles at the boundary.
        assert "do not pile" in low or "don't pile" in low

    def test_open_upper_carries_symmetric_concrete_guidance(self):
        upper, _lower = bound_messages(_open_question(open_lower=True, open_upper=True))
        up = " ".join(upper.lower().split())
        assert "only way" in up or "only channel" in up
        assert "including mass beyond the displayed range" in up
        assert "separate field" not in up
        assert "tail mass" not in up
        assert "median" in up
        assert "above" in up
        assert "do not pile" in up or "don't pile" in up

    def test_closed_bounds_have_no_concrete_out_of_bound_guidance(self):
        upper, lower = bound_messages(_open_question(open_lower=False, open_upper=False))
        for msg in (upper.lower(), lower.lower()):
            collapsed = " ".join(msg.split())
            assert "only way" not in collapsed
            assert "only channel" not in collapsed
            assert "including mass beyond the displayed range" not in collapsed
            assert "do not pile" not in collapsed
            assert "don't pile" not in collapsed
            # Closed messages keep the hard-limit framing.
        assert "can not be higher" in upper
        assert "can not be lower" in lower

    def test_no_language_implies_a_scalar_tails_field(self):
        """The deleted `tails`/tail-mass field must not be referenced as a place to put mass."""
        upper, lower = bound_messages(_open_question(open_lower=True, open_upper=True))
        for msg in (upper.lower(), lower.lower()):
            # A "tail mass field" / "tails field" would imply the removed scalar channel.
            assert "tail-mass field" not in msg
            assert "tails field" not in msg


class TestNumericPromptOpenLowerGuidance:
    def test_numeric_prompt_open_lower_has_median_below_floor_instruction(self):
        """The assembled numeric prompt for an open-lower question surfaces the concrete
        'median below the floor' style instruction (threaded via the lower-bound message)."""
        question = _open_question(open_lower=True, open_upper=False)
        _upper_msg, lower_msg = bound_messages(question)
        prompt = numeric_prompt(question, research="r", lower_bound_message=lower_msg, upper_bound_message="")
        collapsed = " ".join(prompt.lower().split())
        assert "median" in collapsed
        assert "below" in collapsed
        assert "only way" in collapsed or "only channel" in collapsed


def _discrete_question() -> NumericQuestion:
    """Q38195-class discrete count: raw bounds [-0.5, 7.5], cdf_size 9 (0..7 → 8 values + 1).

    Half-step derivation yields nominal (displayed) bounds [0.0, 7.0].
    """
    return NumericQuestion(
        id_of_question=38195,
        id_of_post=38195,
        page_url="https://example.com/q/38195",
        question_text="How many events?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        open_time=_OPEN,
        scheduled_resolution_time=_RESOLVE,
        lower_bound=-0.5,
        upper_bound=7.5,
        open_lower_bound=False,
        open_upper_bound=True,
        unit_of_measure="events",
        zero_point=None,
        cdf_size=9,
    )


class TestNumericPromptDisplaysNominalBounds:
    """De-contradiction fix: the prompt must render the nominal/displayed bound (not the raw
    scaling bound) and must not carry hard-cap phrasing that overrides the open-bound notes."""

    def test_numeric_prompt_renders_nominal_range_not_raw(self):
        q = _discrete_question()
        upper_msg, lower_msg = bound_messages(q)
        prompt = numeric_prompt(q, research="r", lower_bound_message=lower_msg, upper_bound_message=upper_msg)
        assert "Displayed range (in base units): [0.0, 7.0]" in prompt
        assert "[-0.5, 7.5]" not in prompt

    def test_numeric_prompt_drops_hard_cap_phrasing(self):
        q = _discrete_question()
        upper_msg, lower_msg = bound_messages(q)
        prompt = numeric_prompt(q, research="r", lower_bound_message=lower_msg, upper_bound_message=upper_msg)
        assert "(must follow)" not in prompt
        assert "Respect the explicit bounds" not in prompt
        assert "Allowed range" not in prompt

    def test_stacking_numeric_prompt_renders_nominal_range_not_raw(self):
        q = _discrete_question()
        upper_msg, lower_msg = bound_messages(q)
        prompt = stacking_numeric_prompt(
            q,
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message=lower_msg,
            upper_bound_message=upper_msg,
        )
        assert "Displayed range (base units): [0.0, 7.0]" in prompt
        assert "[-0.5, 7.5]" not in prompt
        assert "(must follow)" not in prompt
        assert "Allowed range" not in prompt

    def test_numeric_prompt_carries_replacement_open_bound_guidance(self):
        """Positive assertions on the NEW guidance (not just absence of the old): the
        closed/open semantics live ONCE, in the Units & Bounds bullet (plus the per-side bound
        messages interpolated below it); the step-8 restatement that used to carry them was
        retired as a duplicate, so the pins moved here rather than being deleted."""
        q = _discrete_question()
        upper_msg, lower_msg = bound_messages(q)
        prompt = numeric_prompt(q, research="r", lower_bound_message=lower_msg, upper_bound_message=upper_msg)
        collapsed = " ".join(prompt.split())
        assert "Keep them within a closed bound (the outcome cannot cross it)" in collapsed
        assert (
            "an open bound is only the displayed range, so a percentile may sit at or beyond it when warranted"
            in collapsed
        )
        assert "displayed range is suggestive of units" in collapsed
        # The retired step-8 duplicates (the stacking prompt keeps its own copy, untouched).
        assert "the displayed edge is NOT a hard limit" not in prompt
        assert "For a closed bound, no percentile may cross it." not in prompt
        assert "Think in ranges, not single points" not in prompt
        assert "Ensure strictly increasing percentiles" not in prompt
        assert "Avoid scientific notation" not in prompt
        # The schema Notes stay the one copy of the mechanical format rules.
        assert collapsed.count("no scientific notation") == 1
        assert collapsed.count("strictly increasing") == 1

    def test_numeric_prompt_forecastability_and_width_is_one_template_step(self):
        """The old Step-9b asked for an output line ``FORECASTABILITY: HIGH/MEDIUM/LOW`` that
        nothing parsed, and the preamble carried a separate 567-char calibration paragraph. The
        operator's intent is real and now lives in ONE template step, "Forecastability and
        width": decide whether the quantity is forecastable from current information or close to
        a random walk, and if near-unforecastable centre on the current value with a width taken
        from realized variability. It absorbs the calibration paragraph's kernel, keeps the
        even-handed wording the 2026-07 width audit settled on (no directional push toward wide
        or narrow), and asks for no output line; the preamble shrinks to a pointer."""
        q = _discrete_question()
        upper_msg, lower_msg = bound_messages(q)
        prompt = numeric_prompt(q, research="r", lower_bound_message=lower_msg, upper_bound_message=upper_msg)
        collapsed = " ".join(prompt.split())
        assert "Forecastability and width" in collapsed
        # No output line, and no classification block: nothing downstream ever read either.
        # These are the strings step 9b actually carried at 7e7d449.
        assert "FORECASTABILITY:" not in prompt
        assert "FORECASTABILITY: HIGH" not in prompt
        assert "Forecastability classification" not in prompt
        assert "How inherently predictable is this quantity" not in prompt
        # The decision the step asks for, and what to do at the unforecastable end.
        assert "how forecastable this quantity is from current information" in collapsed
        assert "close to a random walk" in collapsed
        assert "centre on the current value" in collapsed
        assert "realized variability" in collapsed
        assert "do not expect movement you cannot source" in collapsed
        # The calibration paragraph's even-handed kernel, stated once.
        assert collapsed.count("Match your interval width to what your reasoning actually supports") == 1
        assert collapsed.count("generic disposition") == 1
        assert "Calibration guidance:" not in collapsed
        # Both sides of the log-score asymmetry, so neither direction is a standing push.
        # Pinned as the whole clause: bare "narrow" / "wide" also occur in the Scoring Rule
        # paragraph and the tails line, so those passed with this step deleted.
        assert (
            "Log score punishes a narrow interval that misses far more than a wide one that covers, "
            "but a wide interval on a predictable quantity also bleeds points" in collapsed
        )
        assert "take the width from its realized variability over comparable windows" in collapsed
        # The tails line survives inside the same step.
        assert (
            "Keep your extreme tails (P1 and P99) wide enough to cover unknown unknowns you can actually name"
            in collapsed
        )
        # The removed directional pushes must not resurface (Step-9b widening + Step-7 narrowing).
        assert "large fraction of the displayed range" not in prompt
        assert "as narrow as the historical data justifies" not in prompt
        assert "you are losing points" not in prompt
        assert "hedge audit" not in prompt.lower()
        assert "Forecastability check:" not in prompt

    def test_stacking_numeric_prompt_carries_replacement_open_bound_guidance(self):
        q = _discrete_question()
        upper_msg, lower_msg = bound_messages(q)
        prompt = stacking_numeric_prompt(
            q,
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message=lower_msg,
            upper_bound_message=upper_msg,
        )
        assert "the displayed edge is NOT a hard limit" in prompt
        assert "For a closed bound, no percentile may cross it." in prompt
