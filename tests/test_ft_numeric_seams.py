"""Seam-pinning tests for the forecasting-tools numeric/MC data-model contracts
that the 0.2.54 -> 0.2.92 upgrade is known to touch (FUTURE.md Workstream B).

All tests are green on the currently-installed 0.2.54. They fall into two
kinds:

Pin 1 (``TestMcSubOnePercentProbabilityIsPublished`` — a DECISION PIN):
``clamp_and_renormalize_mc`` floors option probabilities at ``MC_PROB_MIN =
0.005``, so an option landing in ``[0.005, 0.01)`` survives our per-model clamp,
the ``aggregation_strategies`` median combine, and the report publish path
sub-0.01 on 0.2.54. At 0.2.92 the upstream ``PredictedOptionList`` validator
dropped its ``abs(sum - 1) < 0.0001`` early return and unconditionally clamps
every option into ``[0.01, 0.99]`` on construction, so a 0.007 option would be
bumped to 0.01. That is a *conscious* semantics decision at upgrade time (keep
our 0.005 floor or adopt upstream's 0.01), not a silent drift — hence the pin is
tagged as a DECISION PIN rather than an invariant we intend to preserve.

Pin 2 (``TestNumericDistributionAcceptsBeyondRangePercentiles`` — an INVARIANT
PIN): our numeric pipeline expresses out-of-bound probability mass by placing
declared percentiles beyond the displayed range (the Minions & Monsters /
Toy Story convention). This pins that upstream's plain ``NumericDistribution``
(a) constructs from that beyond-range ``sanitize_percentiles`` output without
raising, (b) does not rewrite ``declared_percentiles`` in its validators, and
(c) still exposes a ``.cdf`` that ``aggregate_numeric`` consumes to reproduce the
declared distribution. At 0.2.92 the strict validators (``_check_too_far_from_bounds``,
``_check_and_update_repeating_values``) can reject or mutate this convention;
the fix there is to thread ``strict_validation=False``. If any of these go red at
HEAD that thread is required, so the pin is the signal, not a silent break.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from forecasting_tools import (
    MetaculusApi,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
)
from forecasting_tools.data_models.multiple_choice_report import (
    MultipleChoiceReport,
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.aggregation_strategies import AggregationStrategy, combine_multiple_choice_predictions
from metaculus_bot.constants import MC_PROB_MIN
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.pipeline import sanitize_percentiles
from metaculus_bot.numeric.utils import aggregate_numeric, clamp_and_renormalize_mc

_MC_OPTIONS = ["A", "B", "C"]


def _pol(a_prob: float, *, b_share: float = 0.6) -> PredictedOptionList:
    """A 3-option PredictedOptionList summing to 1, with option ``A`` at ``a_prob``.

    The remainder is split between B/C so construction always passes the upstream
    sum-close-to-1 validator; only ``A``'s magnitude is under test.
    """
    rest = 1.0 - a_prob
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=a_prob),
            PredictedOption(option_name="B", probability=rest * b_share),
            PredictedOption(option_name="C", probability=rest * (1.0 - b_share)),
        ]
    )


# In-range 13-percentile forecast on closed [0, 100]; used for the aggregate seam.
_IN_RANGE_VALUES = [5, 8, 12, 18, 28, 42, 50, 58, 72, 82, 88, 92, 96]
# A materially different in-range forecast, for the non-vacuity control below.
_IN_RANGE_VALUES_ALT = [2, 4, 7, 11, 16, 30, 40, 50, 64, 76, 84, 90, 95]

# Cribbed from tests/test_thirteen_percentile_e2e.py. Open bounds [75M, 150M].
# Minions & Monsters: P1..P60 sit BELOW the open floor (mass below lower bound).
_MINIONS_VALUES = [30e6, 35e6, 40e6, 45e6, 52e6, 63e6, 68e6, 74e6, 90e6, 110e6, 130e6, 145e6, 160e6]
# Toy Story 5: P40..P99 sit ABOVE the open ceiling (mass above upper bound).
_TOY_STORY_VALUES = [90e6, 100e6, 110e6, 125e6, 140e6, 158e6, 165e6, 172e6, 185e6, 200e6, 220e6, 240e6, 260e6]


def _declared(values: Sequence[float]) -> list[Percentile]:
    return [Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, values)]


class TestMcSubOnePercentProbabilityIsPublished:
    """DECISION PIN: an MC option in [0.005, 0.01) survives our aggregation and publishes sub-0.01.

    # DECISION PIN: encodes MC_PROB_MIN=0.005 semantics; will be consciously
    # revised at ft 0.2.92 where upstream clamps [0.01,0.99].
    """

    @pytest.fixture
    def mc_question(self) -> MultipleChoiceQuestion:
        return MultipleChoiceQuestion(
            question_text="Which?",
            id_of_question=103,
            id_of_post=103,
            page_url="https://www.metaculus.com/questions/103/",
            background_info="",
            resolution_criteria="",
            fine_print="",
            options=_MC_OPTIONS,
        )

    def _capture_mc_publish(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Mock the two publish classmethods and return a container for the captured MC payload."""
        captured: dict[str, Any] = {}

        def fake_post_mc(question_id: int, options_with_probabilities: dict[str, float]) -> Any:
            captured["question_id"] = question_id
            captured["options"] = dict(options_with_probabilities)
            return MagicMock()

        monkeypatch.setattr(MetaculusApi, "post_multiple_choice_question_prediction", fake_post_mc)
        monkeypatch.setattr(MetaculusApi, "post_question_comment", MagicMock())
        return captured

    def test_per_model_clamp_preserves_sub_one_percent_option(self) -> None:
        """clamp_and_renormalize_mc floors at 0.005, so a 0.007 option is left in the [0.005, 0.01) band."""
        clamped = clamp_and_renormalize_mc(_pol(0.007))
        a_prob = clamped.predicted_options[0].probability
        assert MC_PROB_MIN <= a_prob < 0.01, f"per-model clamp must keep 0.007 in [0.005, 0.01); got {a_prob}"

    def test_median_combine_then_publish_emits_sub_one_percent(
        self, mc_question: MultipleChoiceQuestion, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Full prod MC path: per-model clamp -> aggregation_strategies MEDIAN -> publish, all sub-0.01."""
        per_model = [clamp_and_renormalize_mc(_pol(a)) for a in (0.006, 0.007, 0.008)]
        aggregated = combine_multiple_choice_predictions(per_model, AggregationStrategy.MEDIAN)

        # The aggregated object already carries the sub-0.01 mass on 0.2.54 (no re-clamp to 0.01).
        agg_a = aggregated.predicted_options[0].probability
        assert agg_a < 0.01, f"median combine must not clamp 0.007 up to 0.01 on 0.2.54; got {agg_a}"

        captured = self._capture_mc_publish(monkeypatch)
        report = MultipleChoiceReport(question=mc_question, prediction=aggregated, explanation="# Seam test")
        asyncio.run(report.publish_report_to_metaculus())

        assert captured["question_id"] == mc_question.id_of_question
        published_a = captured["options"]["A"]
        assert published_a < 0.01, f"published MC probability must stay sub-0.01 on 0.2.54; got {published_a}"
        # Sanity: it is the actual small mass, not a dropped-to-zero option.
        assert published_a > 0.0

    def test_floor_is_half_a_percent_not_one_percent(self) -> None:
        """Non-vacuity: a below-floor option is raised toward 0.005 (our floor), NOT to upstream's 0.01.

        If the clamp floor were 0.01 (the 0.2.92 upstream semantics), a 0.001 input
        would land >= 0.01 and this assertion would fail — that discrimination is
        what makes the sub-0.01 pin above meaningful rather than accidental.
        """
        clamped = clamp_and_renormalize_mc(_pol(0.001))
        a_prob = clamped.predicted_options[0].probability
        assert a_prob > 0.001, "the clamp must actually raise a below-floor option (proves it fires)"
        assert a_prob < 0.01, f"our 0.005 floor keeps a floored option sub-0.01; upstream's 0.01 would not ({a_prob})"

    def test_in_bounds_option_is_untouched_by_clamp(self) -> None:
        """Non-vacuity: a comfortably in-bounds option is not perturbed, so the clamp only bites the tail."""
        clamped = clamp_and_renormalize_mc(_pol(0.4))
        assert clamped.predicted_options[0].probability == pytest.approx(0.4)


class TestNumericDistributionAcceptsBeyondRangePercentiles:
    """INVARIANT PIN: upstream accepts our beyond-range percentile convention and does not rewrite it."""

    @pytest.fixture
    def open_question(self) -> NumericQuestion:
        return NumericQuestion(
            question_text="What will the gross be?",
            id_of_question=4242,
            id_of_post=4242,
            page_url="https://www.metaculus.com/questions/4242/",
            background_info="",
            resolution_criteria="",
            fine_print="",
            lower_bound=75_000_000.0,
            upper_bound=150_000_000.0,
            open_lower_bound=True,
            open_upper_bound=True,
            zero_point=None,
            unit_of_measure="USD",
            cdf_size=201,
        )

    @pytest.fixture
    def closed_question(self) -> NumericQuestion:
        return NumericQuestion(
            question_text="What will the value be?",
            id_of_question=4243,
            id_of_post=4243,
            page_url="https://www.metaculus.com/questions/4243/",
            background_info="",
            resolution_criteria="",
            fine_print="",
            lower_bound=0.0,
            upper_bound=100.0,
            open_lower_bound=False,
            open_upper_bound=False,
            zero_point=None,
            unit_of_measure="units",
            cdf_size=201,
        )

    @pytest.mark.parametrize(
        ("fixture_name", "values"),
        [("minions", _MINIONS_VALUES), ("toy_story", _TOY_STORY_VALUES)],
    )
    def test_beyond_range_construction_succeeds_without_mutating_declared(
        self, fixture_name: str, values: list[float], open_question: NumericQuestion
    ) -> None:
        """Construction from beyond-range sanitize output must succeed and leave declared_percentiles verbatim."""
        del fixture_name  # only present so the two cases show distinct test ids
        sanitized, _zero_point = sanitize_percentiles(_declared(values), open_question)
        # Precondition: this really is a beyond-range case (some declared mass sits outside the displayed range),
        # otherwise the pin would not exercise the beyond-range validator behavior at all.
        assert (
            min(p.value for p in sanitized) < open_question.lower_bound
            or max(p.value for p in sanitized) > open_question.upper_bound
        )

        distribution = NumericDistribution.from_question(sanitized, open_question)

        # Pin: this is the plain upstream model, not our PchipNumericDistribution subclass.
        assert type(distribution) is NumericDistribution
        # Pin: validators accept the beyond-range convention and do not rewrite our percentiles.
        assert [p.value for p in distribution.declared_percentiles] == [p.value for p in sanitized]
        assert [p.percentile for p in distribution.declared_percentiles] == [p.percentile for p in sanitized]

    def test_minions_cdf_carries_large_below_floor_mass(self, open_question: NumericQuestion) -> None:
        """The consumed .cdf must express the below-open-floor belief (cdf[0] > 0.5)."""
        sanitized, _ = sanitize_percentiles(_declared(_MINIONS_VALUES), open_question)
        cdf = NumericDistribution.from_question(sanitized, open_question).cdf
        assert len(cdf) == 201
        assert cdf[0].percentile > 0.5, f"expected large below-floor mass at cdf[0], got {cdf[0].percentile}"

    def test_toy_story_cdf_carries_large_above_ceiling_mass(self, open_question: NumericQuestion) -> None:
        """The consumed .cdf must express the above-open-ceiling belief (cdf[-1] well under 0.5)."""
        sanitized, _ = sanitize_percentiles(_declared(_TOY_STORY_VALUES), open_question)
        cdf = NumericDistribution.from_question(sanitized, open_question).cdf
        assert len(cdf) == 201
        assert cdf[-1].percentile < 0.5, f"expected large above-ceiling mass, got cdf[-1]={cdf[-1].percentile}"

    def test_aggregate_numeric_consumes_plain_cdf_and_reproduces_declared(
        self, closed_question: NumericQuestion
    ) -> None:
        """aggregate_numeric must read .cdf off a plain NumericDistribution and reproduce it for a single input.

        For a single prediction the median-of-one is identity, so the aggregated
        distribution's CDF must equal the plain distribution's own .cdf. This pins
        that aggregate_numeric still consumes the upstream .cdf property — at 0.2.92
        that property is deprecated in favor of get_cdf(), so if the read seam moves
        this equality (or the read itself) breaks.
        """
        sanitized, _ = sanitize_percentiles(_declared(_IN_RANGE_VALUES), closed_question)
        plain = NumericDistribution.from_question(sanitized, closed_question)
        assert type(plain) is NumericDistribution

        plain_cdf = plain.cdf
        aggregated_cdf = aggregate_numeric([plain], closed_question, method="median").cdf

        assert len(aggregated_cdf) == len(plain_cdf) == 201
        # Value axis is the same lower..upper grid; a pandas groupby round-trip introduces
        # float epsilon (~1e-14), so compare with a tolerance rather than exact equality.
        assert [p.value for p in aggregated_cdf] == pytest.approx([p.value for p in plain_cdf], abs=1e-6)
        # The probability (CDF height) axis is the load-bearing part — it must round-trip exactly.
        assert [p.percentile for p in aggregated_cdf] == pytest.approx([p.percentile for p in plain_cdf], abs=1e-9)

    def test_aggregate_numeric_is_input_sensitive(self, closed_question: NumericQuestion) -> None:
        """Non-vacuity control: a different input yields a different aggregated CDF.

        Proves the equality pinned above is a real round-trip through the .cdf read,
        not an artifact of aggregate_numeric ignoring its input.
        """
        base_sanitized, _ = sanitize_percentiles(_declared(_IN_RANGE_VALUES), closed_question)
        alt_sanitized, _ = sanitize_percentiles(_declared(_IN_RANGE_VALUES_ALT), closed_question)

        base_probs = np.array(
            [
                p.percentile
                for p in aggregate_numeric(
                    [NumericDistribution.from_question(base_sanitized, closed_question)],
                    closed_question,
                    method="median",
                ).cdf
            ]
        )
        alt_probs = np.array(
            [
                p.percentile
                for p in aggregate_numeric(
                    [NumericDistribution.from_question(alt_sanitized, closed_question)],
                    closed_question,
                    method="median",
                ).cdf
            ]
        )

        assert float(np.max(np.abs(base_probs - alt_probs))) > 0.01

    def test_upstream_validator_is_active_negative_control(self, open_question: NumericQuestion) -> None:
        """Non-vacuity: the upstream validator DOES reject genuinely-invalid percentiles.

        A decreasing percentile sequence must raise, proving the validator runs — so
        the "beyond-range is accepted" pin above reflects real tolerance, not a
        disabled validator.
        """
        with pytest.raises(ValueError):
            NumericDistribution(
                declared_percentiles=[
                    Percentile(percentile=0.5, value=10.0),
                    Percentile(percentile=0.4, value=20.0),
                ],
                open_upper_bound=open_question.open_upper_bound,
                open_lower_bound=open_question.open_lower_bound,
                upper_bound=open_question.upper_bound,
                lower_bound=open_question.lower_bound,
                zero_point=None,
                cdf_size=201,
            )
