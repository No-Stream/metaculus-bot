"""A concentrated declaration on a small discrete grid publishes its mass on the bin it named.

Half of Mantic's quantitative questions are discrete, 132 of them with a grid step of exactly 1.0
and 18 with thirteen bins or fewer. The prompt asks for thirteen strictly increasing percentile
values, which no integer sequence can supply on a three-bin grid, so the model repeats integers,
and the count-like cluster spreader used to separate that plateau by a full unit per position.
The grid points of a discrete question are its bin EDGES (``range_min = nominal_min - step / 2``),
so a plateau at integer k spread past k +- 0.5 handed the mass the forecaster put on k to the
neighbouring bins. Run through the repo's own pipeline on real corpus questions (Mantic edge-case
review, 2026-09, rank 7), a declared 90% on the first bin of post 253 (three bins) published
0.5584, 80% on post 430 (four bins) published 0.4965, and 70% on post 643 (twenty-one bins)
published 0.4337 with 0.4302 on the neighbour.

The spreader now caps a plateau's TOTAL spread at one bin width wherever the published bins are
the outcome space (``grid_is_outcome_space``: a natively discrete question or a non-201 grid, the
shapes the vote-gated discrete snap never reshapes afterwards), so the plateau stays inside its
bin and the CDF read at the bin edge is the plateau's top percentile. What a declaration pins is
an interval: ``P90 = 0`` and ``P95 = 1`` say ``F(0.5)`` lies in [0.90, 0.95), and the published
mass is that interval's lower edge less the uniform mixture the min-step forces in. The 201-point
continuous grid of a ``NumericQuestion`` is untouched (``tests/test_cluster_processing`` pins its
outputs): there the unit spread is pre-processing the snap can re-concentrate.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import DiscreteQuestion, NumericQuestion

from metaculus_bot.numeric.config import STANDARD_PERCENTILES, grid_bin_width, grid_step_constraints
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from tests.pipeline_test_helpers import assert_server_accepts_cdf

# The uniform mixture that pre-satisfies the min-step moves at most ``alpha`` of the mass, and
# ``alpha = 1.1 * min_step * cdf_size / range`` is below 0.015 on every grid here.
_UNIFORM_MIXTURE_SLACK = 0.02


class _CorpusCase(NamedTuple):
    post_id: int
    cdf_size: int
    lower_bound: float
    upper_bound: float
    open_upper: bool
    # The thirteen declared values, one per STANDARD_PERCENTILES label.
    declared: tuple[float, ...]
    # The bin the plateau names and the percentile interval the declaration pins on its upper edge.
    plateau_bin: int
    plateau_bottom: float
    plateau_top: float
    # What the uncapped spreader published on that bin, from the review's reproduction.
    published_before: float


# "How many counts in James Comey's April 2026 indictment will remain active": bins 0, 1, 2.
_POST_253 = _CorpusCase(253, 4, -0.5, 2.5, False, (0.0,) * 10 + (1.0, 1.0, 2.0), 0, 0.0, 0.90, 0.5584)

_CASES = [
    pytest.param(_POST_253, id="post-253-three-bins-90pct-on-bin-0"),
    # "How many Starship integrated flight tests ... successfully complete": bins 0..3, open ceiling.
    pytest.param(
        _CorpusCase(430, 5, -0.5, 3.5, True, (0.0,) * 9 + (1.0, 1.0, 2.0, 3.0), 0, 0.0, 0.80, 0.4965),
        id="post-430-four-bins-80pct-on-bin-0",
    ),
    # "How many public releases will U.S. Central Command publish": bins 0..20, open ceiling.
    pytest.param(
        _CorpusCase(643, 22, -0.5, 20.5, True, (0.0,) * 8 + (1.0, 1.0, 2.0, 3.0, 4.0), 0, 0.0, 0.60, 0.4337),
        id="post-643-twenty-one-bins-60pct-on-bin-0",
    ),
    # "How many votes in favour will the DRC 1533 sanctions renewal receive": bins 12..16.
    pytest.param(
        _CorpusCase(507, 6, 11.5, 16.5, False, (12.0,) + (13.0,) * 10 + (14.0, 15.0), 1, 0.025, 0.95, 0.7098),
        id="post-507-five-bins-92pct-on-bin-13",
    ),
]


def _question(case: _CorpusCase) -> DiscreteQuestion:
    return DiscreteQuestion(
        id_of_question=case.post_id,
        id_of_post=case.post_id,
        page_url=f"https://competitions.mantic.com/questions/{case.post_id}",
        question_text="Discrete corpus question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=case.lower_bound,
        upper_bound=case.upper_bound,
        open_lower_bound=False,
        open_upper_bound=case.open_upper,
        unit_of_measure="",
        zero_point=None,
        cdf_size=case.cdf_size,
    )


def _declaration(case: _CorpusCase) -> list[Percentile]:
    return [Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, case.declared, strict=True)]


@pytest.mark.parametrize("case", _CASES)
class TestPlateauStaysInsideItsBin:
    def test_sanitized_plateau_never_leaves_the_bin(self, case: _CorpusCase) -> None:
        question = _question(case)
        sanitized, _ = sanitize_percentiles(_declaration(case), question, model_name="member")
        bin_width = grid_bin_width(case.lower_bound, case.upper_bound, case.cdf_size)
        assert bin_width == 1.0
        plateau_value = case.declared[len(case.declared) // 2]  # the median label sits on the plateau
        plateau = [p.value for p, v in zip(sanitized, case.declared, strict=True) if v == plateau_value]
        assert min(plateau) >= plateau_value - bin_width / 2 - 1e-9
        assert max(plateau) <= plateau_value + bin_width / 2 + 1e-9

    def test_published_mass_is_the_declared_interval_lower_edge(self, case: _CorpusCase) -> None:
        question = _question(case)
        sanitized, zero_point = sanitize_percentiles(_declaration(case), question, model_name="member")
        member = build_numeric_distribution(sanitized, question, zero_point, model_name="member")
        probs = np.asarray([p.percentile for p in member.get_cdf()], dtype=float)
        assert_server_accepts_cdf(probs, cdf_size=case.cdf_size, open_lower=False, open_upper=case.open_upper)

        pmf = np.diff(probs)
        published = float(pmf[case.plateau_bin])
        declared_floor = case.plateau_top - case.plateau_bottom
        assert published >= declared_floor - _UNIFORM_MIXTURE_SLACK, f"{published:.4f} on bin {case.plateau_bin}"
        assert published > case.published_before + 0.15, "the cap must move the mass back onto the bin"


def test_the_ninety_percent_intent_publishes_at_least_0_85() -> None:
    """The review's headline number: 24 baseline points on post 253 under today's formula."""
    question = _question(_POST_253)
    sanitized, zero_point = sanitize_percentiles(_declaration(_POST_253), question, model_name="member")
    probs = np.asarray([p.percentile for p in build_numeric_distribution(sanitized, question, zero_point).get_cdf()])
    assert float(np.diff(probs)[0]) >= 0.85


class TestTheCapIsKeyedOnTheQuestionTypeNotOnCdfSize:
    """A 200-bin Mantic discrete question has ``cdf_size == 201`` and is still capped.

    Thirteen Series 1 corpus questions have this shape. Its bins are the outcome space just as a
    three-bin grid's are, so the plateau stays inside bin 0; the same declaration on a
    ``NumericQuestion`` of the same grid keeps the unit spread, because there the vote-gated snap
    can still re-concentrate the mass. On this grid the platform's own per-bin cap (0.2 at 200
    bins) then binds, so the published bin-0 mass is the cap either way; the spreader's contract
    is visible in the sanitized values, not in the PMF.
    """

    _LOWER, _UPPER, _CDF_SIZE = -0.5, 199.5, 201
    _DECLARED = (0.0,) * 10 + (1.0, 1.0, 2.0)

    def _question(self, question_type: type[NumericQuestion]) -> NumericQuestion:
        return question_type(
            id_of_question=999,
            id_of_post=999,
            page_url="https://competitions.mantic.com/questions/999",
            question_text="200-bin count question",
            background_info="",
            resolution_criteria="",
            fine_print="",
            published_time=None,
            close_time=None,
            lower_bound=self._LOWER,
            upper_bound=self._UPPER,
            open_lower_bound=False,
            open_upper_bound=True,
            unit_of_measure="",
            zero_point=None,
            cdf_size=self._CDF_SIZE,
        )

    def _sanitized_plateau(self, question: NumericQuestion) -> list[float]:
        declaration = [
            Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, self._DECLARED, strict=True)
        ]
        sanitized, _ = sanitize_percentiles(declaration, question, model_name="member")
        return [p.value for p in sanitized[:10]]

    def test_discrete_question_plateau_stays_in_bin_zero(self) -> None:
        plateau = self._sanitized_plateau(self._question(DiscreteQuestion))
        assert min(plateau) >= -0.5 - 1e-9
        assert max(plateau) <= 0.5 + 1e-9

    def test_continuous_question_on_the_same_grid_keeps_the_unit_spread(self) -> None:
        plateau = self._sanitized_plateau(self._question(NumericQuestion))
        assert max(plateau) > 0.5, "the 201-point continuous grid must be byte-identical to before"

    def test_discrete_question_publishes_the_platform_cap_on_bin_zero(self) -> None:
        question = self._question(DiscreteQuestion)
        declaration = [
            Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, self._DECLARED, strict=True)
        ]
        sanitized, zero_point = sanitize_percentiles(declaration, question, model_name="member")
        probs = np.asarray(
            [p.percentile for p in build_numeric_distribution(sanitized, question, zero_point).get_cdf()]
        )
        assert_server_accepts_cdf(probs, cdf_size=self._CDF_SIZE, open_lower=False, open_upper=True)
        assert float(np.diff(probs)[0]) == pytest.approx(grid_step_constraints(self._CDF_SIZE)[1], abs=1e-12)
