"""The epoch-seconds view of a date question (``metaculus_bot/numeric/date_axis.py``).

Pinned against the two recorded Mantic date payloads (``tests/mantic_fakes.py``): post 651, the
12-bin day-granularity question with both bounds closed, and post 500, the legacy 200-bin
closed-lower / open-upper shape. The load-bearing facts, each with its own test: the adapter is a
real ``NumericQuestion`` whose bounds are the ``.timestamp()`` of the datetime bounds; nominal
bounds come off the wire (Mantic's ``nominal_max`` is the LAST bin's left edge, one bin below
``range_max``, which the repo's half-step derivation would get wrong); the repo's CDF value grid
reproduces the platform's ``continuous_range`` exactly, so a CDF built here is bucketed the way the
platform buckets it; a date-only declaration lands at noon UTC, inside its day's right-closed bin;
and none of it depends on the host's timezone.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import DateQuestion, DiscreteQuestion, MultipleChoiceQuestion

from metaculus_bot.numeric.bounds_clamping import calculate_bounds_buffer, clamp_values_to_bounds
from metaculus_bot.numeric.date_axis import (
    DATE_GRANULARITIES,
    EpochDateQuestion,
    as_epoch_question,
    format_epoch,
    numeric_qtype,
    numeric_view,
    parse_iso_utc,
    to_epoch,
)
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.numeric.utils import bound_messages, nominal_bounds
from metaculus_bot.question_types import question_type_of
from tests.mantic_fakes import (
    DATE_POST_ID,
    DISCRETE_POST_ID,
    load_legacy_date_post,
    load_preseason_post,
)
from tests.pipeline_test_helpers import make_real_date_question, make_real_numeric_question

_DAY = 86_400.0

# Post 651's wire scaling, as recorded 2026-09-08: 12 one-day bins from 2026-09-08 to 2026-09-20.
_Q651_RANGE_MIN = datetime(2026, 9, 8, tzinfo=UTC)
_Q651_RANGE_MAX = datetime(2026, 9, 20, tzinfo=UTC)
_Q651_NOMINAL_MAX = datetime(2026, 9, 19, tzinfo=UTC)
_Q651_CDF_SIZE = 13

# Post 500's wire scaling: 2026-06-17T15:00Z to 2026-08-12T12:00Z, 200 bins, open upper bound.
_Q500_RANGE_MIN = datetime(2026, 6, 17, 15, tzinfo=UTC)
_Q500_RANGE_MAX = datetime(2026, 8, 12, 12, tzinfo=UTC)


@pytest.fixture
def q651() -> DateQuestion:
    return DateQuestion.from_metaculus_api_json(load_preseason_post(DATE_POST_ID))


@pytest.fixture
def q500() -> DateQuestion:
    return DateQuestion.from_metaculus_api_json(load_legacy_date_post())


def _continuous_range_epochs(post: dict) -> np.ndarray:
    return np.asarray(
        [to_epoch(datetime.fromisoformat(edge)) for edge in post["question"]["scaling"]["continuous_range"]]
    )


class TestTheAdapterIsTheDateQuestionOnTheEpochAxis:
    def test_it_is_a_real_numeric_question_with_epoch_bounds(self, q651: DateQuestion) -> None:
        epoch = as_epoch_question(q651)
        assert isinstance(epoch, EpochDateQuestion)
        assert isinstance(epoch, type(make_real_numeric_question()))
        assert epoch.lower_bound == _Q651_RANGE_MIN.timestamp()
        assert epoch.upper_bound == _Q651_RANGE_MAX.timestamp()
        assert epoch.upper_bound - epoch.lower_bound == pytest.approx(12 * _DAY)

    def test_the_bound_flags_grid_and_identity_are_copied(self, q651: DateQuestion) -> None:
        epoch = as_epoch_question(q651)
        assert (epoch.open_lower_bound, epoch.open_upper_bound) == (False, False)
        assert epoch.zero_point is None
        assert epoch.cdf_size == _Q651_CDF_SIZE
        assert epoch.id_of_question == q651.id_of_question == 651
        assert epoch.id_of_post == q651.id_of_post
        assert epoch.page_url == q651.page_url
        assert epoch.question_text == q651.question_text
        assert epoch.resolution_criteria == q651.resolution_criteria
        assert epoch.fine_print == q651.fine_print
        assert epoch.background_info == q651.background_info
        assert epoch.close_time == q651.close_time
        assert epoch.scheduled_resolution_time == q651.scheduled_resolution_time
        assert epoch.open_time == q651.open_time
        assert epoch.unit_of_measure == q651.unit_of_measure
        assert epoch.api_json == q651.api_json

    def test_nominal_bounds_come_off_the_wire_not_from_the_half_step_rule(self, q651: DateQuestion) -> None:
        """Mantic labels a day-granularity bin by its LEFT edge: ``nominal_max`` is the last answer
        date, a whole bin below ``range_max``. The half-step derivation would report noon of the
        19th and noon of the 8th, neither of which is a date on the platform's grid."""
        epoch = as_epoch_question(q651)
        nominal_upper = epoch.nominal_upper_bound
        assert nominal_upper is not None
        assert epoch.nominal_lower_bound == _Q651_RANGE_MIN.timestamp()
        assert nominal_upper == _Q651_NOMINAL_MAX.timestamp()
        assert epoch.upper_bound - nominal_upper == pytest.approx(_DAY)
        upper, lower = nominal_bounds(epoch)
        assert (upper, lower) == (_Q651_NOMINAL_MAX.timestamp(), _Q651_RANGE_MIN.timestamp())

    def test_granularity_is_read_from_the_payload(self, q651: DateQuestion, q500: DateQuestion) -> None:
        assert as_epoch_question(q651).date_granularity == "day"
        assert as_epoch_question(q500).date_granularity == ""

    def test_legacy_shape_has_nominal_equal_to_range_and_an_open_upper_bound(self, q500: DateQuestion) -> None:
        epoch = as_epoch_question(q500)
        assert (epoch.open_lower_bound, epoch.open_upper_bound) == (False, True)
        assert epoch.cdf_size == 201
        assert (epoch.lower_bound, epoch.upper_bound) == (_Q500_RANGE_MIN.timestamp(), _Q500_RANGE_MAX.timestamp())
        assert (epoch.nominal_lower_bound, epoch.nominal_upper_bound) == (epoch.lower_bound, epoch.upper_bound)

    def test_a_question_built_without_api_json_falls_back_to_the_range_bounds(self) -> None:
        question = make_real_date_question(date_granularity="", with_scaling=False)
        epoch = as_epoch_question(question)
        assert epoch.nominal_lower_bound == epoch.lower_bound == to_epoch(question.lower_bound)
        assert epoch.nominal_upper_bound == epoch.upper_bound == to_epoch(question.upper_bound)
        assert epoch.date_granularity == ""

    def test_an_unknown_granularity_raises_instead_of_being_forecast(self) -> None:
        """``month`` is in Mantic's rules doc but not in the API, and calendar-month bins are not
        expressible in the platform's uniform scaling; a grid whose bins mean something new must
        fail here, not publish."""
        question = make_real_date_question(date_granularity="month")
        with pytest.raises(ValueError, match="unknown date_granularity 'month'"):
            as_epoch_question(question)
        assert {"", "day", "week"} == DATE_GRANULARITIES


class TestTheGridMatchesThePlatform:
    """``build_cdf_value_grid`` on the adapter's bounds IS the platform's ``continuous_range``: the
    server derives its edges from ``linspace(0, 1, N + 1)`` through the same scaling, so a CDF
    evaluated here is bucketed exactly as it will be scored. Pinned on all three recorded grids."""

    def test_the_twelve_day_grid_is_exact(self, q651: DateQuestion) -> None:
        epoch = as_epoch_question(q651)
        grid = build_cdf_value_grid(epoch.lower_bound, epoch.upper_bound, epoch.zero_point, epoch.cdf_size)
        expected = _continuous_range_epochs(load_preseason_post(DATE_POST_ID))
        assert len(expected) == _Q651_CDF_SIZE
        np.testing.assert_allclose(grid, expected, rtol=0, atol=1e-6)
        # Every edge is a UTC midnight, one day apart: calendar day D = range_min + k days is bin k.
        assert np.all(np.diff(grid) == _DAY)
        assert all(datetime.fromtimestamp(edge, tz=UTC).hour == 0 for edge in grid)

    def test_the_legacy_201_point_grid_is_exact(self, q500: DateQuestion) -> None:
        epoch = as_epoch_question(q500)
        grid = build_cdf_value_grid(epoch.lower_bound, epoch.upper_bound, epoch.zero_point, epoch.cdf_size)
        expected = _continuous_range_epochs(load_legacy_date_post())
        assert len(expected) == 201
        np.testing.assert_allclose(grid, expected, rtol=0, atol=1e-6)
        # The legacy shape: uniform in epoch seconds but never aligned to a calendar day.
        assert np.allclose(np.diff(grid), np.diff(grid)[0])
        assert not all(datetime.fromtimestamp(edge, tz=UTC).hour == 0 for edge in grid)

    def test_the_451_point_quantitative_grid_is_exact(self) -> None:
        post = load_preseason_post(DISCRETE_POST_ID)
        question = DiscreteQuestion.from_metaculus_api_json(post)
        grid = build_cdf_value_grid(question.lower_bound, question.upper_bound, question.zero_point, question.cdf_size)
        expected = np.asarray(post["question"]["scaling"]["continuous_range"], dtype=float)
        assert len(expected) == 451
        np.testing.assert_allclose(grid, expected, rtol=0, atol=1e-6)


class TestParsingIsStrictUtcAndLandsInsideTheDay:
    def test_a_date_only_value_is_noon_utc(self) -> None:
        moment = parse_iso_utc("2026-09-16")
        assert moment == datetime(2026, 9, 16, 12, tzinfo=UTC)
        assert moment.tzinfo is UTC

    def test_noon_sits_strictly_inside_the_platform_bin_for_that_day(self, q651: DateQuestion) -> None:
        """Bin k of post 651 is ``(edge_k, edge_{k+1}]``; 2026-09-16 is k = 8. A midnight reading
        would sit ON edge 8, where the platform's 1e-10 bucket fudge decides the bin."""
        epoch = as_epoch_question(q651)
        grid = build_cdf_value_grid(epoch.lower_bound, epoch.upper_bound, None, epoch.cdf_size)
        value = to_epoch(parse_iso_utc("2026-09-16"))
        assert grid[8] < value < grid[9]
        assert value - grid[8] == pytest.approx(_DAY / 2)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("2026-09-16T00:00:00Z", datetime(2026, 9, 16, tzinfo=UTC)),
            ("2026-09-16T09:30:00+02:00", datetime(2026, 9, 16, 7, 30, tzinfo=UTC)),
            ("2026-09-16T23:59:59", datetime(2026, 9, 16, 23, 59, 59, tzinfo=UTC)),
            ("2026-09-16 15:00", datetime(2026, 9, 16, 15, tzinfo=UTC)),
            ("2026-09-16T15:00:00.250Z", datetime(2026, 9, 16, 15, 0, 0, 250_000, tzinfo=UTC)),
        ],
    )
    def test_timestamps_are_honoured_and_a_naive_one_is_utc(self, text: str, expected: datetime) -> None:
        assert parse_iso_utc(text) == expected

    @pytest.mark.parametrize(
        "text",
        ["2027", "2027-06", "20270601", "2027-W10", "June 1 2027", "06/01/2027", "2027-06-1", "", "soon"],
    )
    def test_anything_looser_than_iso_is_rejected(self, text: str) -> None:
        with pytest.raises(ValueError, match="not a strict ISO-8601"):
            parse_iso_utc(text)

    def test_the_epoch_does_not_depend_on_the_host_timezone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """forecasting-tools' own date template calls ``.timestamp()`` on a naive datetime, an
        8-hour error on a Pacific-time host. Ours reads a naive value as UTC on every host."""
        expected = datetime(2026, 9, 16, 12, tzinfo=UTC).timestamp()
        original_tz = os.environ.get("TZ")
        try:
            for zone in ("America/Los_Angeles", "Asia/Tokyo", "UTC"):
                monkeypatch.setenv("TZ", zone)
                time.tzset()
                assert to_epoch(parse_iso_utc("2026-09-16")) == expected
                assert to_epoch(datetime(2026, 9, 16, 12)) == expected
                # The framework's naive call is the trap this guards against.
                naive_local = datetime(2026, 9, 16, 12).timestamp()
                assert (naive_local == expected) == (zone == "UTC")
        finally:
            if original_tz is None:
                monkeypatch.delenv("TZ", raising=False)
            else:
                monkeypatch.setenv("TZ", original_tz)
            time.tzset()

    def test_to_epoch_converts_an_aware_non_utc_value(self) -> None:
        aware = datetime(2026, 9, 16, 12, tzinfo=UTC).astimezone(datetime.now().astimezone().tzinfo)
        assert to_epoch(aware) == datetime(2026, 9, 16, 12, tzinfo=UTC).timestamp()


class TestRenderingForHumans:
    def test_day_granularity_renders_calendar_dates(self) -> None:
        assert format_epoch(_Q651_NOMINAL_MAX.timestamp(), "day") == "2026-09-19"
        assert format_epoch(_Q651_NOMINAL_MAX.timestamp(), "week") == "2026-09-19"

    def test_legacy_granularity_renders_the_full_utc_timestamp(self) -> None:
        assert format_epoch(_Q500_RANGE_MAX.timestamp(), "") == "2026-08-12T12:00:00Z"

    def test_bound_messages_read_as_dates_on_the_adapter(self, q651: DateQuestion, q500: DateQuestion) -> None:
        upper_msg, lower_msg = bound_messages(as_epoch_question(q651))
        assert "2026-09-19" in upper_msg
        assert "closed" in upper_msg
        assert "2026-09-08" in lower_msg
        assert "closed" in lower_msg
        assert "1789776000" not in upper_msg

        upper_msg, lower_msg = bound_messages(as_epoch_question(q500))
        assert "2026-08-12T12:00:00Z" in upper_msg
        assert "open" in upper_msg
        assert "2026-06-17T15:00:00Z" in lower_msg
        assert "closed" in lower_msg

    def test_bound_messages_on_a_plain_numeric_question_are_unchanged(self) -> None:
        upper_msg, lower_msg = bound_messages(make_real_numeric_question(lower_bound=0.0, upper_bound=20.0))
        assert "20.0" in upper_msg
        assert "0.0" in lower_msg


class TestNumericViewAndTypeDispatch:
    def test_identity_for_a_numeric_question(self) -> None:
        question = make_real_numeric_question()
        assert numeric_view(question) is question

    def test_identity_for_the_adapter_itself(self, q651: DateQuestion) -> None:
        epoch = as_epoch_question(q651)
        assert numeric_view(epoch) is epoch

    def test_adapter_for_a_date_question(self, q651: DateQuestion) -> None:
        view = numeric_view(q651)
        assert isinstance(view, EpochDateQuestion)
        assert view.lower_bound == _Q651_RANGE_MIN.timestamp()

    def test_type_error_for_anything_else(self) -> None:
        question = MultipleChoiceQuestion(question_text="which?", options=["a", "b"])
        with pytest.raises(TypeError, match="has no numeric view"):
            numeric_view(question)

    def test_question_type_of_names_the_real_date_question(self, q651: DateQuestion) -> None:
        assert question_type_of(q651) == "date"
        # The adapter IS a NumericQuestion to the leaf module; its emitters ask numeric_qtype.
        assert question_type_of(as_epoch_question(q651)) == "numeric"

    def test_numeric_qtype_tells_the_adapter_apart(self, q651: DateQuestion) -> None:
        assert numeric_qtype(as_epoch_question(q651)) == "date"
        assert numeric_qtype(make_real_numeric_question()) == "numeric"


class TestTheClosedBoundClampOnADateAxis:
    """One day outside a closed bound of a 12-day question clamps; ten days outside still raises."""

    def test_one_day_before_a_closed_lower_bound_clamps(self, q651: DateQuestion) -> None:
        epoch = as_epoch_question(q651)
        buffer = calculate_bounds_buffer(epoch)
        assert buffer == pytest.approx(_DAY)
        day_before = to_epoch(_Q651_RANGE_MIN - timedelta(days=1))
        percentiles = _percentiles([day_before, epoch.lower_bound + 2 * _DAY, epoch.lower_bound + 5 * _DAY])
        values, corrected = clamp_values_to_bounds([p.value for p in percentiles], percentiles, epoch, buffer)
        assert corrected
        assert values[0] == epoch.lower_bound + buffer
        assert values[1:] == [epoch.lower_bound + 2 * _DAY, epoch.lower_bound + 5 * _DAY]

    def test_ten_bins_outside_still_raises(self, q651: DateQuestion) -> None:
        epoch = as_epoch_question(q651)
        ten_days_after = to_epoch(_Q651_RANGE_MAX + timedelta(days=10))
        percentiles = _percentiles([epoch.lower_bound + 2 * _DAY, epoch.lower_bound + 5 * _DAY, ten_days_after])
        with pytest.raises(ValueError, match="too far above upper bound"):
            clamp_values_to_bounds([p.value for p in percentiles], percentiles, epoch, calculate_bounds_buffer(epoch))


def _percentiles(values: list[float]) -> list[Percentile]:
    labels = [0.1, 0.5, 0.9]
    return [Percentile(percentile=label, value=value) for label, value in zip(labels, values, strict=True)]
