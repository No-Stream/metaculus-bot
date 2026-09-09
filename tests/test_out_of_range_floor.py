"""The Mantic-only out-of-range tail floor (``metaculus_bot/numeric/out_of_range_floor.py``).

Mantic scores an outcome beyond an OPEN bound against a fixed 5% reference (``50 * ln(mass / 0.05)``):
5% there scores 0, the structural 1% this pipeline publishes when every percentile sits inside the
range scores -80.5. Mantic's writers set the ranges and half its date questions resolved above the
displayed range in Series 1, so the PUBLISHED aggregate of a Mantic numeric, discrete or date question
gets each open tail raised to ``MANTIC_OUT_OF_RANGE_TAIL_FLOOR``; Metaculus aggregates stay
byte-identical and per-member forecasts are never touched.

Three layers are pinned here: the pure array function (its semantics and the server's CDF rules on
its output), the same rules on every distinct grid Mantic has ever published (the recorded corpus,
distilled into ``tests/data/mantic_cdf_grids_2026_09_08.json``), and the forecaster seam that applies
it to the published aggregate and records raw and published tails on ``NUMERIC_AGGREGATE``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from forecasting_tools import NumericDistribution, NumericQuestion
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import DateQuestion
from scipy.stats import norm

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import MANTIC_OUT_OF_RANGE_TAIL_FLOOR, MANTIC_SITE_URL, PLATFORM_MANTIC, PLATFORM_METACULUS
from metaculus_bot.numeric.config import STANDARD_PERCENTILES, grid_step_constraints
from metaculus_bot.numeric.date_axis import as_epoch_question, to_epoch
from metaculus_bot.numeric.out_of_range_floor import (
    FlooredCdf,
    floor_out_of_range_tails,
    floor_published_tails,
    tail_floor_for_platform,
)
from metaculus_bot.numeric.pchip_cdf import safe_cdf_bounds
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric.utils import aggregate_numeric
from metaculus_bot.question_platform import question_platform
from tests.pipeline_test_helpers import (
    assert_server_accepts_cdf,
    make_e2e_bot,
    make_real_date_question,
    make_real_numeric_question,
)

FLOOR = MANTIC_OUT_OF_RANGE_TAIL_FLOOR
GRIDS_PATH = Path(__file__).parent / "data" / "mantic_cdf_grids_2026_09_08.json"


def _legal_cdf(raw_heights: np.ndarray, *, open_lower: bool, open_upper: bool) -> np.ndarray:
    """Make raw heights server-legal the way the ensemble post-processing does: endpoint pins, then
    ``safe_cdf_bounds`` with the grid's own step limits; asserts the server accepts the result."""
    heights = np.maximum.accumulate(np.clip(np.asarray(raw_heights, dtype=float), 0.0, 1.0))
    heights[0] = max(heights[0], 0.001) if open_lower else 0.0
    heights[-1] = min(heights[-1], 0.999) if open_upper else 1.0
    min_step, max_step = grid_step_constraints(len(heights))
    legal = safe_cdf_bounds(heights, open_lower, open_upper, min_step=min_step, max_step=max_step)
    assert_server_accepts_cdf(legal, cdf_size=len(legal), open_lower=open_lower, open_upper=open_upper)
    return legal


def _normal_cdf(n_points: int, *, centre: float, sd: float) -> np.ndarray:
    """A normal CDF sampled on the unit grid: ``centre`` and ``sd`` are fractions of the range."""
    return norm.cdf(np.linspace(0.0, 1.0, n_points), loc=centre, scale=sd)


SHAPES: dict[str, Callable[[int], np.ndarray]] = {
    "concentrated_middle": lambda n: _normal_cdf(n, centre=0.5, sd=0.03),
    "piled_on_lower_edge": lambda n: _normal_cdf(n, centre=0.05, sd=0.02),
    "piled_on_upper_edge": lambda n: _normal_cdf(n, centre=0.95, sd=0.02),
    "flat": lambda n: np.linspace(0.0, 1.0, n),
    "fat_both_tails": lambda n: _normal_cdf(n, centre=0.5, sd=0.6),
    "fat_upper_tail_only": lambda n: _normal_cdf(n, centre=0.9, sd=0.3),
}
"""Every aggregate shape a Mantic question can publish, up to the grid, as raw heights on the unit grid."""


def _shape(name: str, n_points: int) -> np.ndarray:
    return SHAPES[name](n_points)


def _assert_floored_result_is_legal(result: FlooredCdf, before: np.ndarray, *, open_lower: bool, open_upper: bool):
    after = result.cdf
    assert len(after) == len(before)
    assert np.all(np.diff(after) >= 0.0), "the floored CDF must stay non-decreasing"
    assert_server_accepts_cdf(after, cdf_size=len(after), open_lower=open_lower, open_upper=open_upper)

    raw_low, raw_high = float(before[0]), 1.0 - float(before[-1])
    assert result.raw == (raw_low, raw_high)
    assert result.published == (float(after[0]), 1.0 - float(after[-1]))

    # Per side: an open tail under the floor is raised to exactly the floor, otherwise left alone.
    if open_lower:
        assert after[0] == (FLOOR if raw_low < FLOOR else before[0])
    else:
        assert after[0] == 0.0 == before[0]
    if open_upper:
        assert after[-1] == ((1.0 - FLOOR) if raw_high < FLOOR else before[-1])
    else:
        assert after[-1] == 1.0 == before[-1]
    assert result.floored == (after[0] != before[0] or after[-1] != before[-1])


class TestFloorSemantics:
    """The array function on the standard 201-point grid, one rule per test."""

    def test_both_open_tails_under_the_floor_are_raised_to_exactly_the_floor(self) -> None:
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=True, open_upper=True)
        assert (before[0], before[-1]) == (0.001, 0.999), "the structural 1% tails are the input"

        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)

        _assert_floored_result_is_legal(result, before, open_lower=True, open_upper=True)
        assert result.floored is True
        assert result.published == (FLOOR, 1.0 - (1.0 - FLOOR))
        assert result.published[1] == pytest.approx(FLOOR)

    def test_the_interior_is_rescaled_not_shifted_so_the_median_stays_put(self) -> None:
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=True, open_upper=True)
        after = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR).cdf
        # An affine map of [0.001, 0.999] onto [0.05, 0.95] fixes 0.5; the min-step re-lift of the tail
        # bins then moves the crossing by a few thousandths of a bin, nothing a reader could see.
        grid = np.arange(len(before), dtype=float)
        crossing_before = float(np.interp(0.5, before, grid))
        crossing_after = float(np.interp(0.5, after, grid))
        assert crossing_before == pytest.approx(100.0)
        assert abs(crossing_after - crossing_before) < 0.01

    def test_an_open_lower_and_closed_upper_bound_floors_the_low_side_only(self) -> None:
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=True, open_upper=False)
        result = floor_out_of_range_tails(before, open_lower=True, open_upper=False, floor=FLOOR)
        _assert_floored_result_is_legal(result, before, open_lower=True, open_upper=False)
        assert result.cdf[0] == FLOOR
        assert result.cdf[-1] == 1.0

    def test_a_closed_lower_and_open_upper_bound_floors_the_high_side_only(self) -> None:
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=False, open_upper=True)
        result = floor_out_of_range_tails(before, open_lower=False, open_upper=True, floor=FLOOR)
        _assert_floored_result_is_legal(result, before, open_lower=False, open_upper=True)
        assert result.cdf[0] == 0.0
        assert result.cdf[-1] == 1.0 - FLOOR

    def test_two_closed_bounds_return_the_input_unchanged(self) -> None:
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=False, open_upper=False)
        result = floor_out_of_range_tails(before, open_lower=False, open_upper=False, floor=FLOOR)
        assert result.floored is False
        assert np.array_equal(result.cdf, before)
        assert result.raw == result.published == (0.0, 0.0)

    def test_tails_already_at_or_above_the_floor_are_never_reduced(self) -> None:
        before = _legal_cdf(_shape("fat_both_tails", 201), open_lower=True, open_upper=True)
        assert before[0] > FLOOR
        assert 1.0 - before[-1] > FLOOR
        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)
        assert result.floored is False
        assert np.array_equal(result.cdf, before)

    def test_one_thin_and_one_fat_tail_floors_only_the_thin_side(self) -> None:
        before = _legal_cdf(_shape("fat_upper_tail_only", 201), open_lower=True, open_upper=True)
        assert before[0] < FLOOR < 1.0 - before[-1]
        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)
        _assert_floored_result_is_legal(result, before, open_lower=True, open_upper=True)
        assert result.cdf[0] == FLOOR
        assert result.cdf[-1] == before[-1], "the fat side keeps its exact mass"

    def test_a_flat_cdf_floors_both_sides_and_stays_flat_inside(self) -> None:
        before = _legal_cdf(_shape("flat", 201), open_lower=True, open_upper=True)
        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)
        _assert_floored_result_is_legal(result, before, open_lower=True, open_upper=True)
        interior_steps = np.diff(result.cdf)[1:-1]
        assert interior_steps.max() - interior_steps.min() < 1e-9, "an affine map keeps a uniform interior uniform"

    def test_a_zero_floor_is_a_no_op_on_any_shape(self) -> None:
        for name in SHAPES:
            before = _legal_cdf(_shape(name, 201), open_lower=True, open_upper=True)
            result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=0.0)
            assert result.floored is False, name
            assert np.array_equal(result.cdf, before), name

    def test_the_input_array_is_not_mutated(self) -> None:
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=True, open_upper=True)
        snapshot = before.copy()
        floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)
        assert np.array_equal(before, snapshot)

    def test_the_min_step_is_re_enforced_because_the_shrink_alone_breaks_it(self) -> None:
        """A concentrated aggregate carries long runs of bins sitting EXACTLY at the server's minimum
        step (the uniform-mixture tails of the PCHIP build). The affine shrink scales every step by
        the same factor under one, so those bins land below the minimum and the server would reject
        the submission; the floor re-runs the pipeline's min-step sweep, which is what makes the
        result legal. Pinned by showing the naive shrink fails where the function passes."""
        before = _legal_cdf(_shape("concentrated_middle", 201), open_lower=True, open_upper=True)
        min_step, _ = grid_step_constraints(len(before))
        assert np.isclose(np.diff(before), min_step).sum() > 100, "the input carries min-step runs"

        scale = (1.0 - 2.0 * FLOOR) / (before[-1] - before[0])
        naive = FLOOR + (before - before[0]) * scale
        with pytest.raises(AssertionError, match="step below server min"):
            assert_server_accepts_cdf(naive, cdf_size=len(naive), open_lower=True, open_upper=True)

        result = floor_out_of_range_tails(before, open_lower=True, open_upper=True, floor=FLOOR)
        _assert_floored_result_is_legal(result, before, open_lower=True, open_upper=True)


def _corpus_grids() -> list[tuple[int, bool, bool, int]]:
    payload = json.loads(GRIDS_PATH.read_text(encoding="utf-8"))
    return [
        (g["inbound_outcome_count"] + 1, g["open_lower_bound"], g["open_upper_bound"], g["questions"])
        for g in payload["grids"]
    ]


class TestEveryManticGrid:
    """The server rules hold on every distinct grid Mantic has published, for every shape.

    The fixture holds each ``(inbound_outcome_count, open_lower_bound, open_upper_bound)`` triple
    among the 524 continuous questions of the recorded 2026-09-08 corpus: 88 grids from 3 to 450
    bins, both-open, one-open and both-closed. The floor acts on heights alone, so the value axis
    is not part of a grid's identity.
    """

    def test_the_fixture_covers_the_corpus(self) -> None:
        grids = _corpus_grids()
        assert len(grids) == 88
        assert sum(questions for *_, questions in grids) == 524
        assert {(lo, hi) for _, lo, hi, _ in grids} == {(True, True), (False, True), (True, False), (False, False)}
        assert (min(n for n, *_ in grids), max(n for n, *_ in grids)) == (4, 451)

    @pytest.mark.parametrize("shape_name", sorted(SHAPES))
    def test_the_floored_cdf_passes_the_server_on_every_grid(self, shape_name: str) -> None:
        for cdf_size, open_lower, open_upper, _ in _corpus_grids():
            before = _legal_cdf(_shape(shape_name, cdf_size), open_lower=open_lower, open_upper=open_upper)
            result = floor_out_of_range_tails(before, open_lower=open_lower, open_upper=open_upper, floor=FLOOR)
            try:
                _assert_floored_result_is_legal(result, before, open_lower=open_lower, open_upper=open_upper)
            except AssertionError as exc:
                raise AssertionError(f"{shape_name} on cdf_size={cdf_size} open=({open_lower},{open_upper})") from exc

    def test_a_thin_tailed_shape_ends_with_the_floor_beyond_every_open_bound(self) -> None:
        for cdf_size, open_lower, open_upper, _ in _corpus_grids():
            before = _legal_cdf(_shape("concentrated_middle", cdf_size), open_lower=open_lower, open_upper=open_upper)
            after = floor_out_of_range_tails(before, open_lower=open_lower, open_upper=open_upper, floor=FLOOR).cdf
            if open_lower:
                assert after[0] >= FLOOR, (cdf_size, open_lower, open_upper)
            if open_upper:
                assert 1.0 - after[-1] >= FLOOR - 1e-12, (cdf_size, open_lower, open_upper)


_MANTIC_URL = f"{MANTIC_SITE_URL}/questions/2001/"


def _inside_range_members(
    question: NumericQuestion, centres: tuple[float, ...], sd: float
) -> list[NumericDistribution]:
    """Three members whose thirteen percentiles all sit inside the range: the structural 1% tails."""
    members: list[NumericDistribution] = []
    for centre in centres:
        declared = [
            Percentile(percentile=p, value=float(norm.ppf(p, loc=centre, scale=sd))) for p in STANDARD_PERCENTILES
        ]
        sanitized, zero_point = sanitize_percentiles(declared, question, model_name="test-model")
        members.append(build_numeric_distribution(sanitized, question, zero_point, model_name="test-model"))
    return members


def _heights(distribution: NumericDistribution) -> np.ndarray:
    return np.asarray([p.percentile for p in distribution.get_cdf()], dtype=float)


def _aggregate_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("NUMERIC_AGGREGATE:")]


def _fields(line: str) -> dict[str, str]:
    return dict(token.split("=", 1) for token in line.split(": ", 1)[1].split(" "))


class TestPlatformGate:
    def test_mantic_gets_the_floor_and_metaculus_gets_none(self) -> None:
        mantic = make_real_numeric_question(open_lower_bound=True)
        mantic.page_url = _MANTIC_URL
        assert question_platform(mantic) == PLATFORM_MANTIC
        assert tail_floor_for_platform(mantic) == FLOOR

        metaculus = make_real_numeric_question(open_lower_bound=True)
        assert question_platform(metaculus) == PLATFORM_METACULUS
        assert tail_floor_for_platform(metaculus) == 0.0

    def test_the_date_adapter_keeps_the_platform(self) -> None:
        date_question = make_real_date_question(open_upper_bound=True)
        assert tail_floor_for_platform(date_question) == FLOOR
        assert tail_floor_for_platform(as_epoch_question(date_question)) == FLOOR


class TestTheForecasterSeam:
    """``TemplateForecaster._aggregate_predictions``, the one seam every aggregation path returns through."""

    @pytest.mark.asyncio
    async def test_a_mantic_numeric_aggregate_publishes_the_floor_beyond_each_open_bound(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        question = make_real_numeric_question(open_lower_bound=True, open_upper_bound=True)
        question.page_url = _MANTIC_URL
        members = _inside_range_members(question, centres=(10.0, 10.4, 9.7), sd=1.0)
        member_heights_before = [_heights(m) for m in members]

        bot = make_e2e_bot(AggregationStrategy.MEDIAN)
        aggregated = await bot._aggregate_predictions(cast(list[PredictionTypes], members), question)

        assert isinstance(aggregated, NumericDistribution)
        published = _heights(aggregated)
        assert published[0] >= FLOOR
        assert 1.0 - published[-1] >= FLOOR - 1e-12
        assert_server_accepts_cdf(published, cdf_size=len(published), open_lower=True, open_upper=True)
        assert (aggregated.open_lower_bound, aggregated.open_upper_bound) == (True, True)
        assert (aggregated.lower_bound, aggregated.upper_bound) == (question.lower_bound, question.upper_bound)
        assert [p.value for p in aggregated.declared_percentiles] == [p.value for p in aggregated.get_cdf()]
        assert [p.percentile for p in aggregated.declared_percentiles] == list(map(float, published))
        for before, member in zip(member_heights_before, members, strict=True):
            assert np.array_equal(before, _heights(member))

        (line,) = _aggregate_lines(caplog)
        fields = _fields(line)
        assert fields["qtype"] == "numeric"
        assert fields["cdf_size"] == "201"
        assert (fields["oor_low_raw"], fields["oor_high_raw"]) == ("0.010000", "0.010000")
        assert (fields["oor_low"], fields["oor_high"]) == ("0.050000", "0.050000")
        assert fields["tail_floor"] == "0.050000"

    @pytest.mark.asyncio
    async def test_a_metaculus_aggregate_is_byte_identical_to_the_combiner_output(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        question = make_real_numeric_question(open_lower_bound=True, open_upper_bound=True)
        assert question_platform(question) == PLATFORM_METACULUS
        members = _inside_range_members(question, centres=(10.0, 10.4, 9.7), sd=1.0)
        expected = _heights(aggregate_numeric(members, question, method="median"))

        bot = make_e2e_bot(AggregationStrategy.MEDIAN)
        aggregated = await bot._aggregate_predictions(cast(list[PredictionTypes], members), question)

        assert isinstance(aggregated, NumericDistribution)
        assert np.array_equal(_heights(aggregated), expected)
        assert (expected[0], expected[-1]) == pytest.approx((0.01, 0.99)), "the structural 1% tails, unfloored"
        (line,) = _aggregate_lines(caplog)
        fields = _fields(line)
        assert fields["oor_low"] == fields["oor_low_raw"] == "0.010000"
        assert fields["oor_high"] == fields["oor_high_raw"] == "0.010000"
        assert fields["tail_floor"] == "0.000000"

    def test_the_same_object_comes_back_when_nothing_is_floored(self) -> None:
        question = make_real_numeric_question(open_lower_bound=True, open_upper_bound=True)
        members = _inside_range_members(question, centres=(10.0, 10.4, 9.7), sd=1.0)
        aggregated = aggregate_numeric(members, question, method="median")

        outcome = floor_published_tails(aggregated, question)

        assert outcome.distribution is aggregated
        assert outcome.floor == 0.0
        assert outcome.raw == outcome.published
        assert outcome.raw == pytest.approx((0.01, 0.01))
        assert outcome.cdf_size == 201

    @pytest.mark.asyncio
    async def test_a_mantic_date_aggregate_floors_its_open_upper_bound_and_stays_a_date(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        question: DateQuestion = make_real_date_question(open_lower_bound=False, open_upper_bound=True)
        epoch_question = as_epoch_question(question)
        day = datetime(2026, 9, 16, tzinfo=UTC)
        centres = tuple(to_epoch(day + timedelta(hours=h)) for h in (12, 13, 11))
        members = _inside_range_members(epoch_question, centres=centres, sd=4 * 3600.0)

        bot = make_e2e_bot(AggregationStrategy.MEDIAN)
        aggregated = await bot._aggregate_predictions(cast(list[PredictionTypes], members), question)

        assert isinstance(aggregated, NumericDistribution)
        assert aggregated.is_date is True, "the rebuilt aggregate must still render as dates in the comment"
        published = _heights(aggregated)
        assert len(published) == 13
        assert published[0] == 0.0, "a closed lower bound is never moved"
        assert 1.0 - published[-1] >= FLOOR - 1e-12
        assert_server_accepts_cdf(published, cdf_size=13, open_lower=False, open_upper=True)
        assert int(np.argmax(np.diff(published))) == 8
        (line,) = _aggregate_lines(caplog)
        fields = _fields(line)
        assert (fields["qtype"], fields["cdf_size"]) == ("date", "13")
        assert fields["oor_low"] == fields["oor_low_raw"] == "0.000000"
        assert (fields["oor_high_raw"], fields["oor_high"]) == ("0.010000", "0.050000")
        assert fields["tail_floor"] == "0.050000"
