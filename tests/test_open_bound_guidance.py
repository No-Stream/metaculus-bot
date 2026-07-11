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
from metaculus_bot.prompts import numeric_prompt

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
        upper, lower = bound_messages(_open_question(open_lower=True, open_upper=True))
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
        upper, lower = bound_messages(_open_question(open_lower=True, open_upper=True))
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
            assert "only way" not in collapsed and "only channel" not in collapsed
            assert "including mass beyond the displayed range" not in collapsed
            assert "do not pile" not in collapsed and "don't pile" not in collapsed
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
