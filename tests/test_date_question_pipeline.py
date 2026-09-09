"""A date question through the forecast pipeline, stage by stage, with the LLM stubbed.

The end-to-end proof lives in ``tests/test_mantic_e2e.py``; this module pins each seam a
``DateQuestion`` crosses on the way, so a regression names its stage. The question object stays a
``DateQuestion`` end to end and the numeric math runs on its epoch-seconds view
(``metaculus_bot/numeric/date_axis.py``), so the seams are: the runner (prompt and parse notes
render dates, the extracted epoch percentiles take the guarded numeric build, the member marker says
``qtype=date``), the routing sites that used to raise on anything but the three classic types
(threshold, spread, combine, the stacking gate), the forecaster's dispatch and its aggregate marker,
and the gap-fill v2 brief and ghost forecast.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from forecasting_tools import GeneralLlm, NumericDistribution, ReasonedPrediction
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.numeric_report import NumericReport, Percentile
from forecasting_tools.data_models.questions import DateQuestion

from metaculus_bot.aggregation_pipeline import AggregationPipeline
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import (
    CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD,
    NUMERIC_STACKING_ENABLED_ENV,
)
from metaculus_bot.exceptions import UnitMismatchError
from metaculus_bot.forecaster_runners import (
    _build_guarded_numeric_distribution,
    build_date_parse_notes,
    run_date_forecast,
)
from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.date_axis import as_epoch_question, parse_forecast_date, to_epoch
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.prompts import MARKET_SNAPSHOT_SECTION_HEADER
from metaculus_bot.research.agentic.driver_prompt import _question_header, _template_skeleton
from metaculus_bot.research.agentic.loop import _summarize_ghost
from metaculus_bot.spread_metrics import compute_spread
from metaculus_bot.stacking_route import _conditional_stacking_verdict, _type_gate_enabled
from tests.mantic_fakes import load_legacy_date_question, load_preseason_date_question
from tests.pipeline_test_helpers import assert_server_accepts_cdf, make_e2e_bot, make_real_date_question

_DAY = timedelta(days=1)
_EPOCH_FLOAT_PATTERN = "17"  # every epoch second in 2026 starts with these digits
_TEN_DIGIT_EPOCH = re.compile(r"\b1\d{9}\b")


def _iso_z(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _percentile_key(percentile: float) -> str:
    return f"{percentile:g}"


def _spread_over(start: datetime, end: datetime) -> dict[str, str]:
    """The 13 standard percentiles spaced evenly between two instants, as ISO strings."""
    span = (end - start).total_seconds()
    count = len(STANDARD_PERCENTILES)
    return {
        _percentile_key(p): _iso_z(start + timedelta(seconds=span * index / (count - 1)))
        for index, p in enumerate(STANDARD_PERCENTILES)
    }


def _date_block(declared: dict[str, str]) -> str:
    return "```json\n" + json.dumps({"question_type": "date", "declared_percentiles": declared}) + "\n```\n"


def _date_reasoning(declared: dict[str, str]) -> str:
    return f"## Analysis\n\nThe realised-volatility clustering points at one session.\n\n{_date_block(declared)}"


def _member(question: DateQuestion, declared: dict[str, str]) -> NumericDistribution:
    """One forecaster's built distribution, through the real sanitize and build on the epoch view."""
    epoch = as_epoch_question(question)
    percentiles = [
        Percentile(percentile=float(key), value=to_epoch(parse_forecast_date(value))) for key, value in declared.items()
    ]
    sanitized, zero_point = sanitize_percentiles(percentiles, epoch, model_name="test-model")
    return build_numeric_distribution(sanitized, epoch, zero_point, model_name="test-model")


def _cdf_heights(distribution: NumericDistribution) -> np.ndarray:
    return np.asarray([p.percentile for p in distribution.get_cdf()], dtype=float)


@pytest.fixture
def q651() -> DateQuestion:
    return load_preseason_date_question()


@pytest.fixture
def q500() -> DateQuestion:
    return load_legacy_date_question()


@pytest.fixture
def september_16_members(q651: DateQuestion) -> list[NumericDistribution]:
    """Three forecasters all certain the answer is 2026-09-16, each spread inside that day."""
    day = datetime(2026, 9, 16, tzinfo=UTC)
    windows = ((2, 22), (3, 21), (1, 23))
    return [
        _member(q651, _spread_over(day + timedelta(hours=start), day + timedelta(hours=end))) for start, end in windows
    ]


@pytest.fixture
def test_llm() -> GeneralLlm:
    return GeneralLlm(model="test-model", temperature=0.0)


class TestTheRunner:
    @pytest.mark.asyncio
    async def test_run_date_forecast_builds_a_date_distribution_off_the_block(
        self, q651: DateQuestion, test_llm: GeneralLlm, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The whole runner with only the LLM stubbed: the prompt reads as dates, the block rung
        extracts the ISO dates, and the guarded numeric build runs on the epoch view."""
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        day = datetime(2026, 9, 16, tzinfo=UTC)
        reasoning = _date_reasoning(_spread_over(day + timedelta(hours=2), day + timedelta(hours=22)))
        invoke = AsyncMock(return_value=reasoning)
        with (
            patch.object(test_llm, "invoke", new=invoke),
            patch("metaculus_bot.forecaster_runners.parse_structured", new=AsyncMock()) as parser,
        ):
            result = await run_date_forecast(q651, "research", test_llm, test_llm)

        prompt = invoke.call_args.args[0]
        assert "2026-09-19" in prompt
        assert "2026-09-08" in prompt
        assert "1789776000" not in prompt
        parser.assert_not_called()  # block rung read the value; no discrete vote, no salvage

        prediction = result.prediction_value
        assert isinstance(prediction, NumericDistribution)
        assert prediction.is_date is True
        heights = _cdf_heights(prediction)
        assert len(heights) == 13
        assert heights[0] == 0.0
        assert heights[-1] == 1.0
        assert int(np.argmax(np.diff(heights))) == 8

        (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("MEMBER_FORECAST:")]
        assert " qtype=date " in line
        assert line.endswith(" oor_low=0.000000 oor_high=0.000000")
        assert f"question={q651.id_of_question} " in line

    def test_parse_notes_read_as_dates_and_forbid_ambiguous_forms(self, q651: DateQuestion, q500: DateQuestion) -> None:
        closed = build_date_parse_notes(as_epoch_question(q651))
        assert "at or after the lower bound 2026-09-08" in closed
        assert "at or before the upper bound 2026-09-20" in closed
        assert "Never return a bare year, a year-month or a number" in closed
        assert "'YYYY-MM-DD'" in closed
        assert _EPOCH_FLOAT_PATTERN + "8" not in closed  # no epoch float leaked into the notes

        open_upper = build_date_parse_notes(as_epoch_question(q500))
        assert "only the end of the displayed range" in open_upper
        assert "may not happen inside the window" in open_upper
        assert "extract that date verbatim" in open_upper
        assert "at or after the lower bound 2026-06-17T15:00:00Z" in open_upper

        # Neither recorded payload has an open lower bound, so that branch is pinned on a built question.
        open_lower = build_date_parse_notes(
            as_epoch_question(make_real_date_question(open_lower_bound=True, date_granularity=""))
        )
        assert "The lower bound 2026-09-08T00:00:00Z is only the start of the displayed range" in open_lower
        assert "states a date before 2026-09-08T00:00:00Z, extract that date verbatim" in open_lower
        assert "never move it later into range" in open_lower
        assert "at or before the upper bound 2026-09-20T00:00:00Z" in open_lower
        assert _TEN_DIGIT_EPOCH.search(open_lower) is None

    def test_mass_after_an_open_upper_bound_means_not_by_then(
        self, q500: DateQuestion, test_llm: GeneralLlm, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The modal Mantic date shape: a forecaster who puts 40% of the mass after the window's
        end is saying the event probably does not happen by then. That mass must survive the
        guarded build as ``1 - cdf[-1]`` and show up on the member line."""
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        epoch = as_epoch_question(q500)
        start = datetime(2026, 7, 1, tzinfo=UTC)
        after_window = q500.upper_bound + timedelta(days=60)
        declared = _spread_over(start, after_window)
        percentiles = [
            Percentile(percentile=float(key), value=to_epoch(parse_forecast_date(value)))
            for key, value in declared.items()
        ]

        prediction = _build_guarded_numeric_distribution(percentiles, epoch, test_llm)

        assert prediction.is_date is True
        heights = _cdf_heights(prediction)
        assert len(heights) == 201
        assert heights[0] == 0.0  # closed lower bound
        assert 0.3 < 1.0 - heights[-1] <= 0.999
        (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("MEMBER_FORECAST:")]
        assert " qtype=date " in line
        oor_high = float(line.rsplit("oor_high=", 1)[1])
        assert oor_high == pytest.approx(1.0 - heights[-1], abs=1e-6)

    def test_every_percentile_on_one_day_publishes_with_the_mass_in_that_day(
        self, q651: DateQuestion, test_llm: GeneralLlm, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The date prompt invites this shape and twelve one-day bins can express it. Before the
        outcome-space carve-out in ``apply_cluster_spreading`` the thirteen identical values
        reached the unit-mismatch guard with only the jitter epsilon between them and the member
        was dropped; the gap between "dropped" and "96% on the right day" was one differing
        percentile."""
        caplog.set_level(logging.WARNING, logger="metaculus_bot.numeric.pipeline")
        epoch = as_epoch_question(q651)
        september_16 = to_epoch(parse_forecast_date("2026-09-16"))
        percentiles = [Percentile(percentile=p, value=september_16) for p in STANDARD_PERCENTILES]

        prediction = _build_guarded_numeric_distribution(percentiles, epoch, test_llm)

        heights = _cdf_heights(prediction)
        mass = np.diff(heights)
        assert len(heights) == 13
        assert int(np.argmax(mass)) == 8
        assert mass[8] > 0.95
        assert_server_accepts_cdf(heights, cdf_size=13, open_lower=False, open_upper=False)
        (line,) = [
            r.getMessage() for r in caplog.records if r.getMessage().startswith("NUMERIC_DEGENERATE_DECLARATION:")
        ]
        assert " n_unique=1 " in line
        assert line.endswith(" spread_applied=true")

    @pytest.mark.parametrize(
        ("open_edge", "terminal_bin"),
        [("lower", 0), ("upper", 11)],
        ids=["open_lower", "open_upper"],
    )
    def test_every_percentile_on_an_open_bound_publishes_with_the_mass_in_the_terminal_bin(
        self, open_edge: str, terminal_bin: int, test_llm: GeneralLlm
    ) -> None:
        """A forecaster who writes the bound's own timestamp at all 13 percentiles (the parse
        notes invite "at or after the lower bound 2026-09-08") has put everything in the terminal
        bin, which is where the bound value buckets. The symmetric spread used to put half the
        values past an OPEN bound and the build published ``cdf[0] == 0.5``: a coin flip on
        "before the window" invented from a declaration that named nothing before it (codex
        second-opinion review, 2026-09)."""
        question = make_real_date_question(open_lower_bound=open_edge == "lower", open_upper_bound=open_edge == "upper")
        epoch = as_epoch_question(question)
        bound = epoch.lower_bound if open_edge == "lower" else epoch.upper_bound
        percentiles = [Percentile(percentile=p, value=bound) for p in STANDARD_PERCENTILES]

        prediction = _build_guarded_numeric_distribution(percentiles, epoch, test_llm)

        heights = _cdf_heights(prediction)
        mass = np.diff(heights)
        assert len(heights) == 13
        out_of_range = heights[0] if open_edge == "lower" else 1.0 - heights[-1]
        assert out_of_range < 0.05, out_of_range  # the structural 0.01 of a P1 on the edge, never the invented 0.5
        assert int(np.argmax(mass)) == terminal_bin
        assert mass[terminal_bin] > 0.9
        assert_server_accepts_cdf(
            heights, cdf_size=13, open_lower=open_edge == "lower", open_upper=open_edge == "upper"
        )

    def test_every_percentile_one_day_before_a_closed_lower_bound_is_clamped_in_then_withheld_as_a_point_mass(
        self, test_llm: GeneralLlm, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A whole-set collapse exactly one day before a CLOSED lower bound, the tolerance edge of
        the clamp on a one-day grid. The spreader starts the plateau at the declared value, so the
        clamp, which runs after it, judges the declared distance and accepts it, folding all 13
        values onto the bound; the unit-mismatch guard then withholds that point mass, exactly as
        it did before the branch. No member has ever published from this shape."""
        caplog.set_level(logging.INFO, logger="metaculus_bot.numeric.bounds_clamping")
        epoch = as_epoch_question(make_real_date_question())
        percentiles = [Percentile(percentile=p, value=epoch.lower_bound - 86_400.0) for p in STANDARD_PERCENTILES]

        with pytest.raises(UnitMismatchError, match="tiny span"):
            _build_guarded_numeric_distribution(percentiles, epoch, test_llm)

        clamped = [r for r in caplog.records if r.getMessage().startswith("Clamped lower for Q")]
        assert len(clamped) == len(STANDARD_PERCENTILES)

    def test_every_percentile_two_days_before_a_closed_lower_bound_is_dropped_as_a_scale_error(
        self, test_llm: GeneralLlm
    ) -> None:
        """Beyond the clamp's tolerance the whole-set collapse still raises: starting the plateau at
        the declared value, rather than pulling it inside the range, is what keeps a scale error
        visible to the clamp."""
        epoch = as_epoch_question(make_real_date_question())
        percentiles = [Percentile(percentile=p, value=epoch.lower_bound - 2 * 86_400.0) for p in STANDARD_PERCENTILES]

        with pytest.raises(ValueError, match="too far below lower bound"):
            _build_guarded_numeric_distribution(percentiles, epoch, test_llm)

    def test_a_plateau_declared_before_an_open_lower_bound_keeps_its_below_range_mass(
        self, test_llm: GeneralLlm
    ) -> None:
        """P1..P90 six hours before the window opens, on an open lower bound: a real "probably
        before the window" declaration. The count-like spread caps the plateau at one bin, a
        full day centred six hours out, so 57% of it lies below the bound; the round-one
        translation rule pulled every plateau inside the range and published the structural 1%
        instead (codex re-check, 2026-09). A partial plateau with room from its neighbours keeps
        its symmetric spread under the pre-branch rule too."""
        epoch = as_epoch_question(make_real_date_question(open_lower_bound=True))
        day = 86_400.0
        values = [epoch.lower_bound - 0.25 * day] * 10 + [
            epoch.lower_bound + day,
            epoch.lower_bound + 6 * day,
            epoch.upper_bound,
        ]
        percentiles = [Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, values, strict=True)]

        prediction = _build_guarded_numeric_distribution(percentiles, epoch, test_llm)

        heights = _cdf_heights(prediction)
        assert heights[0] == pytest.approx(0.57, abs=0.01), heights[0]
        assert_server_accepts_cdf(heights, cdf_size=13, open_lower=True, open_upper=False)

    def test_a_plateau_one_day_before_a_closed_lower_bound_is_clamped_in_not_dropped(
        self, test_llm: GeneralLlm
    ) -> None:
        """P1..P90 exactly one day before a CLOSED lower bound, the clamp's tolerance edge on a
        one-day grid, then three in-range percentiles. The spreader's own closed-edge clamp folds
        the part of the symmetric spread that lies below the bound onto the standoff before the
        pipeline's tolerance clamp runs, so nothing reaches that clamp outside its tolerance and
        the member publishes with its mass in the first bin."""
        epoch = as_epoch_question(make_real_date_question())
        day = 86_400.0
        values = [epoch.lower_bound - day] * 10 + [
            epoch.lower_bound + day,
            epoch.lower_bound + 6 * day,
            epoch.upper_bound - 0.5 * day,
        ]
        percentiles = [Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, values, strict=True)]

        prediction = _build_guarded_numeric_distribution(percentiles, epoch, test_llm)

        heights = _cdf_heights(prediction)
        mass = np.diff(heights)
        assert heights[0] == 0.0
        assert mass[0] > 0.85, mass[0]
        assert_server_accepts_cdf(heights, cdf_size=13, open_lower=False, open_upper=False)

    def test_the_fallback_distribution_of_a_201_grid_date_question_still_renders_dates(
        self, q500: DateQuestion, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 201-grid date question (post 500's shape) takes the PCHIP path and lands in
        ``create_fallback_numeric_distribution`` on any build failure. ``is_date`` has to ride
        that path too, or the published comment renders ten-digit epoch floats on exactly the
        questions whose PCHIP build already failed."""
        epoch = as_epoch_question(q500)
        declared = _spread_over(datetime(2026, 7, 1, tzinfo=UTC), q500.upper_bound - timedelta(days=5))
        percentiles = [
            Percentile(percentile=float(key), value=to_epoch(parse_forecast_date(value)))
            for key, value in declared.items()
        ]
        sanitized, zero_point = sanitize_percentiles(percentiles, epoch)

        def _pchip_fails(*_args, **_kwargs):
            raise ValueError("forced PCHIP failure")

        monkeypatch.setattr("metaculus_bot.numeric.pipeline.generate_pchip_cdf_with_smoothing", _pchip_fails)

        prediction = build_numeric_distribution(sanitized, epoch, zero_point)

        assert type(prediction).__name__ == "BoundSafeNumericDistribution"
        assert prediction.is_date is True
        readable = NumericReport.make_readable_prediction(prediction)
        assert "2026-07-" in readable
        assert " UTC" in readable
        assert _TEN_DIGIT_EPOCH.search(readable) is None
        heights = _cdf_heights(prediction)
        assert_server_accepts_cdf(heights, cdf_size=201, open_lower=False, open_upper=True)


class TestRoutingSites:
    def test_the_threshold_is_the_numeric_one(self, q651: DateQuestion) -> None:
        pipeline = AggregationPipeline(
            strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_llm=None,
            parser_llm=GeneralLlm(model="test-model"),
            stacking_spread_thresholds={"binary": 0.1, "mc": 0.2, "numeric": 0.3},
        )
        assert pipeline.get_threshold_for_question(q651) == 0.3

    def test_the_spread_is_measured_on_the_epoch_axis(
        self, q651: DateQuestion, september_16_members: list[NumericDistribution]
    ) -> None:
        spread = compute_spread(q651, september_16_members)
        assert 0.0 <= spread < CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD

    def test_the_median_combine_keeps_the_date_axis(
        self, q651: DateQuestion, september_16_members: list[NumericDistribution]
    ) -> None:
        pipeline = AggregationPipeline(
            strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_llm=None,
            parser_llm=GeneralLlm(model="test-model"),
        )
        combined = pipeline._combine_by_type(
            cast(list[PredictionTypes], september_16_members), q651, AggregationStrategy.MEDIAN, error_context="test"
        )
        assert isinstance(combined, NumericDistribution)
        assert combined.is_date is True
        heights = _cdf_heights(combined)
        assert len(heights) == 13
        assert heights[0] == 0.0
        assert heights[-1] == 1.0
        assert int(np.argmax(np.diff(heights))) == 8
        assert_server_accepts_cdf(heights, cdf_size=13, open_lower=False, open_upper=False)

    def test_a_date_question_reads_the_numeric_stacking_gate(
        self, q651: DateQuestion, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every flag is off in prod; a date question must not be the one type that stacks anyway."""
        monkeypatch.delenv(NUMERIC_STACKING_ENABLED_ENV, raising=False)
        assert _type_gate_enabled(q651) is False
        monkeypatch.setenv(NUMERIC_STACKING_ENABLED_ENV, "true")
        assert _type_gate_enabled(q651) is True

    def test_the_conditional_verdict_skips_on_agreement_without_raising(
        self, q651: DateQuestion, september_16_members: list[NumericDistribution]
    ) -> None:
        pipeline = AggregationPipeline(
            strategy=AggregationStrategy.CONDITIONAL_STACKING,
            stacker_llm=None,
            parser_llm=GeneralLlm(model="test-model"),
            stacking_spread_thresholds={"numeric": CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD},
        )
        reasoned = [
            ReasonedPrediction(prediction_value=member, reasoning="Model: m\n\nr") for member in september_16_members
        ]
        spread, threshold, skip_reason = _conditional_stacking_verdict(
            pipeline, q651, cast(list[ReasonedPrediction[PredictionTypes]], reasoned)
        )
        assert threshold == CONDITIONAL_STACKING_NUMERIC_NORMALIZED_THRESHOLD
        assert spread <= threshold
        assert skip_reason == "spread_below_threshold"


class TestTheForecaster:
    @pytest.mark.asyncio
    async def test_make_prediction_dispatches_to_the_date_runner(
        self, q651: DateQuestion, test_llm: GeneralLlm
    ) -> None:
        bot = make_e2e_bot(AggregationStrategy.CONDITIONAL_STACKING)
        stub = AsyncMock(return_value=ReasonedPrediction(prediction_value=0.0, reasoning="date reasoning"))
        with (
            patch.object(type(bot), "_run_forecast_on_date", new=stub),
            patch.object(type(bot), "_get_notepad", new=AsyncMock(return_value=MagicMock())),
        ):
            result = await bot._make_prediction(q651, "research", test_llm)
        stub.assert_awaited_once_with(q651, "research", test_llm, None)
        assert result.reasoning.startswith(f"Model: {test_llm.model}\n\n")

    @pytest.mark.asyncio
    async def test_the_aggregate_marker_names_the_date_grid(
        self, q651: DateQuestion, september_16_members: list[NumericDistribution], caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.INFO, logger="metaculus_bot")
        bot = make_e2e_bot(AggregationStrategy.MEDIAN)
        aggregated = await bot._aggregate_predictions(cast(list[PredictionTypes], september_16_members), q651)
        assert isinstance(aggregated, NumericDistribution)
        assert aggregated.is_date is True
        (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("NUMERIC_AGGREGATE:")]
        assert line.startswith(
            "NUMERIC_AGGREGATE: question=651 qtype=date cdf_size=13 oor_low=0.000000 oor_high=0.000000"
        )


class TestGapFillV2:
    def test_the_brief_header_and_skeleton_render_dates(self, q651: DateQuestion) -> None:
        header = _question_header(q651)
        assert "Type: date (UTC)" in header
        assert "Displayed range: [2026-09-08, 2026-09-19]" in header
        assert "lower bound closed, upper bound closed" in header

        skeleton = _template_skeleton(q651)
        assert MARKET_SNAPSHOT_SECTION_HEADER in skeleton
        assert "2026-09-19" in skeleton
        assert "1789776000" not in skeleton

    def test_the_open_upper_legacy_shape_renders_timestamps(self, q500: DateQuestion) -> None:
        header = _question_header(q500)
        assert "Displayed range: [2026-06-17T15:00:00Z, 2026-08-12T12:00:00Z]" in header
        assert "lower bound closed, upper bound open" in header

    def test_the_ghost_summary_reads_a_date_block_in_epoch_seconds(self) -> None:
        day = datetime(2026, 9, 16, tzinfo=UTC)
        declared = _spread_over(day + timedelta(hours=2), day + timedelta(hours=22))
        qtype, summary, forecast = _summarize_ghost(_date_block(declared))
        assert qtype == "date"
        assert forecast is not None
        assert forecast["qtype"] == "date"
        assert forecast["median"] == to_epoch(parse_forecast_date(declared["0.5"]))
        assert summary == f"median={declared['0.5']}"
        assert set(forecast["declared_percentiles"]) == set(STANDARD_PERCENTILES)
