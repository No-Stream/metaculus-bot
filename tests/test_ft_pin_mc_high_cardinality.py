"""Pin test for the forecasting-tools 0.2.54 -> 0.2.92 upgrade: high-cardinality MC.

Companion to ``tests/test_ft_upgrade_seams.py`` and ``tests/test_ft_numeric_seams.py``
(FUTURE.md Workstream B). Green on the currently-installed 0.2.54; it defines
"done" for the multiple-choice clamp seam at the upgrade.

The seam: at 0.2.92 upstream's ``PredictedOptionList`` constructor drops its
``abs(sum - 1) < 0.0001`` early return and *unconditionally* clamps every option
into ``[0.01, 0.99]``, renormalizes, and raises ``ValueError`` when any option
moves more than ``0.05``. A high-cardinality ballot (>=6 options) with one
dominant option and several near-floor options is exactly where that raise
fires: the near-floor options each jump to ``0.01``, inflating the sum enough
that renormalization drags the dominant option down by >0.05.

Two pins:

Pin 1 (``TestHighCardinalityMcSurvivesRealPipeline`` -- an INVARIANT PIN):
drives the *real* prod MC path -- ``extract_mc`` (deterministic block rung, no
parser-LLM call) -> per-model ``clamp_and_renormalize_mc`` -> ``MEDIAN`` combine
via ``aggregation_strategies`` -- on a 7-option dominant+near-floor ballot and
asserts the resulting ``PredictedOptionList`` respects OUR
``[MC_PROB_MIN, MC_PROB_MAX]`` clamp bounds. This proves no raw/unclamped list
reaches the publish boundary; at 0.2.92 the fix must keep our clamp ahead of
upstream's stricter constructor so this stays green.

Pin 2 (``TestRawSubFloorConstructionPreservedOn0254`` -- a DECISION PIN):
documents that a *raw* ``PredictedOptionList`` built directly from sub-floor
(0.002) options survives construction verbatim on 0.2.54 (via the early return),
with a worked example of the dominant+eight-near-floor shape that fires the
0.2.92 raise. This is the semantics the upgrade changes, so it is a conscious
revision at 0.2.92, not a silent drift -- the W4 fix agent revises it.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOption,
    PredictedOptionList,
)

from metaculus_bot.aggregation_strategies import AggregationStrategy, combine_multiple_choice_predictions
from metaculus_bot.constants import MC_PROB_MAX, MC_PROB_MIN
from metaculus_bot.numeric.utils import clamp_and_renormalize_mc
from metaculus_bot.value_extraction import extract_mc

# A 7-option ballot: one dominant option plus six near-floor options. ">=6
# options" is the cardinality the seam calls out. The near-floor share (0.006)
# sits in the [MC_PROB_MIN, 0.01) band -- at/above OUR 0.005 floor, but BELOW
# upstream 0.2.92's 0.01 floor -- which is precisely where the two clamp regimes
# diverge.
_DOMINANT = "Alpha"
_NEAR_FLOOR = ["Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf"]
_OPTIONS = [_DOMINANT, *_NEAR_FLOOR]


def _block_rationale(option_probs: dict[str, float]) -> str:
    """A forecaster rationale whose LAST content is a schema-valid MC STRUCTURED FORECAST block.

    Built so ``extract_mc`` resolves on rung 1 (deterministic block parse) with
    zero parser-LLM calls: the block keys map case-insensitively onto ``_OPTIONS``
    and ``option_probs`` sums to 1.0, so ``parse_structured_block`` ->
    ``_make_mc_from_block`` -> ``_validate_mc`` all pass without touching the
    network. This mirrors how production forecasts arrive (a fenced ```json block
    appended to the rationale).
    """
    payload = {"question_type": "multiple_choice", "option_probs": option_probs}
    return "Free-text reasoning about the options.\n\n```json\n" + json.dumps(payload) + "\n```\n"


def _near_floor_ballot(floor_share: float) -> dict[str, float]:
    """Ballot with each near-floor option at ``floor_share``; the dominant option takes the remainder.

    By construction the probabilities sum to exactly 1.0, so the block parser's
    renormalize-to-1.0 step is an identity and the near-floor magnitudes reach
    the clamp untouched.
    """
    dominant = 1.0 - floor_share * len(_NEAR_FLOOR)
    return {_DOMINANT: dominant, **{name: floor_share for name in _NEAR_FLOOR}}


class TestHighCardinalityMcSurvivesRealPipeline:
    """INVARIANT PIN: a >=6-option near-floor ballot survives extract -> clamp -> aggregate, bounded.

    On 0.2.54 the whole prod path yields a valid ``PredictedOptionList`` whose
    every option respects OUR ``[MC_PROB_MIN, MC_PROB_MAX]`` bounds, so no
    raw/unclamped list escapes to publish. At 0.2.92 the upgrade must keep this
    green by threading our clamp ahead of upstream's stricter constructor.

    The near-floor share is 0.006 rather than a truly sub-``MC_PROB_MIN`` value
    on purpose: at/above our 0.005 floor the per-model clamp is an identity and
    the ballot already sums to 1.0, so renormalization is exact and the bounds
    assertion is not confounded by the renormalize-below-floor drift a sub-floor
    input would introduce (a large dominant option divides several floored
    siblings just under 0.005). That drift is a separate, known property of
    ``clamp_and_renormalize_mc`` and is not what this seam pins.
    """

    @pytest.fixture
    def parser_llm(self) -> MagicMock:
        # Rung 1 (block) must satisfy every extraction with zero parser calls. If
        # the ladder ever fell through to the LLM rung, parse_structured would
        # invoke this and the side_effect would surface it -- but the load-bearing
        # proof is the ``outcome.rung == "block"`` assertion below.
        llm = MagicMock()
        llm.invoke.side_effect = AssertionError("parser LLM must not be called; the block rung must win")
        return llm

    def _extract_and_clamp(self, option_probs: dict[str, float], parser_llm: MagicMock) -> PredictedOptionList:
        """Run the per-model prod path: extract_mc (block rung) -> clamp_and_renormalize_mc."""
        outcome = asyncio.run(
            extract_mc(
                _block_rationale(option_probs),
                list(_OPTIONS),
                parser_llm,
                question_id=4242,
                model_name="pin-model",
            )
        )
        assert outcome.rung == "block", f"expected the deterministic block rung, got {outcome.rung!r}"
        assert outcome.block_present is True
        return clamp_and_renormalize_mc(outcome.value)

    def test_per_model_extract_and_clamp_is_bounded(self, parser_llm: MagicMock) -> None:
        """One model's extract+clamp keeps all options in [MC_PROB_MIN, MC_PROB_MAX] for the near-floor ballot."""
        pol = self._extract_and_clamp(_near_floor_ballot(0.006), parser_llm)

        assert {o.option_name for o in pol.predicted_options} == set(_OPTIONS)
        assert sum(o.probability for o in pol.predicted_options) == pytest.approx(1.0)
        for option in pol.predicted_options:
            assert MC_PROB_MIN <= option.probability <= MC_PROB_MAX, (
                f"option {option.option_name!r} probability {option.probability} outside [{MC_PROB_MIN}, {MC_PROB_MAX}]"
            )

        # Non-vacuity: the near-floor options really do sit in the [MC_PROB_MIN, 0.01)
        # band -- below upstream 0.2.92's 0.01 floor -- so the seam is exercised, not
        # dodged by accidentally comfortable probabilities.
        near_floor_probs = [o.probability for o in pol.predicted_options if o.option_name in _NEAR_FLOOR]
        assert near_floor_probs, "expected the near-floor options to be present"
        assert all(MC_PROB_MIN <= p < 0.01 for p in near_floor_probs), (
            f"near-floor options must land in [MC_PROB_MIN, 0.01); got {near_floor_probs}"
        )

    def test_full_path_extract_clamp_median_is_bounded(self, parser_llm: MagicMock) -> None:
        """Full prod MC path across three models: extract -> per-model clamp -> MEDIAN combine, all bounded."""
        per_model = [self._extract_and_clamp(_near_floor_ballot(share), parser_llm) for share in (0.006, 0.007, 0.005)]

        aggregated = combine_multiple_choice_predictions(per_model, AggregationStrategy.MEDIAN)

        assert {o.option_name for o in aggregated.predicted_options} == set(_OPTIONS)
        assert sum(o.probability for o in aggregated.predicted_options) == pytest.approx(1.0)
        for option in aggregated.predicted_options:
            assert MC_PROB_MIN <= option.probability <= MC_PROB_MAX, (
                f"aggregated option {option.option_name!r} probability {option.probability} outside "
                f"[{MC_PROB_MIN}, {MC_PROB_MAX}]"
            )

        dominant_prob = next(o.probability for o in aggregated.predicted_options if o.option_name == _DOMINANT)
        near_floor_probs = [o.probability for o in aggregated.predicted_options if o.option_name in _NEAR_FLOOR]
        # The aggregate is still a dominant-plus-near-floor shape (not smeared into
        # a uniform), so the pin covers the shape the 0.2.92 raise targets.
        assert dominant_prob > 0.9, f"dominant option should remain dominant after MEDIAN combine; got {dominant_prob}"
        assert all(MC_PROB_MIN <= p < 0.01 for p in near_floor_probs), (
            f"aggregated near-floor options must stay in [MC_PROB_MIN, 0.01); got {near_floor_probs}"
        )


class TestRawSubFloorConstructionPreservedOn0254:
    """DECISION PIN: raw PredictedOptionList construction preserves sub-floor options verbatim on 0.2.54.

    # DECISION PIN: encodes 0.2.54 PredictedOptionList construction semantics
    # -- when abs(sum - 1) < 0.0001 the constructor returns the options verbatim
    # with NO clamp -- so a raw list with sub-floor (0.002) options is preserved.
    # At ft 0.2.92 this is consciously revised: the constructor drops that early
    # return, unconditionally clamps every option into [0.01, 0.99], renormalizes,
    # and raises ValueError when any option moves > 0.05. For the exact shape below
    # (one dominant 0.984 option + eight near-floor 0.002 options), the eight clamp
    # 0.002 -> 0.01, inflating the sum to 1.064, which drags the dominant
    # 0.984 -> ~0.925 (a ~0.059 move) and fires the raise. The W4 fix agent revises
    # this test to the 0.2.92 semantics and confirms our pipeline clamps to
    # [MC_PROB_MIN, MC_PROB_MAX] BEFORE any such raw construction can reach upstream.
    """

    def test_sub_floor_options_survive_construction_verbatim(self) -> None:
        """A raw sub-floor ballot summing to exactly 1.0 is preserved verbatim on 0.2.54 (the early-return path)."""
        dominant = 0.984
        sub_floor = 0.002
        near_floor_names = [f"opt{i}" for i in range(8)]  # 8 near-floor options -> the >=6-option raise regime

        pol = PredictedOptionList(
            predicted_options=[
                PredictedOption(option_name="Dominant", probability=dominant),
                *[PredictedOption(option_name=name, probability=sub_floor) for name in near_floor_names],
            ]
        )

        by_name = {o.option_name: o.probability for o in pol.predicted_options}
        # Precondition for the early return: the ballot sums to (float-)exactly 1.0.
        assert sum(by_name.values()) == pytest.approx(1.0, abs=1e-9)
        # 0.2.54: options are kept verbatim -- the 0.002 sub-floor probabilities are
        # NOT clamped up to 0.01. This raw/unclamped list is exactly what our
        # pipeline must never hand upstream once 0.2.92 lands.
        assert by_name["Dominant"] == pytest.approx(dominant)
        for name in near_floor_names:
            assert by_name[name] == pytest.approx(sub_floor)
            assert by_name[name] < 0.01, (
                "sub-floor option is below upstream 0.2.92's 0.01 floor, yet preserved on 0.2.54"
            )
