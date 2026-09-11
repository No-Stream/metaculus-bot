"""End-to-end proof of the Mantic run mode: tournament fetch through publish, HTTP mocked.

This is the ONLY proof of the injected-client path before a paid live run against
competitions.mantic.com, so it drives the real chain rather than a stand-in for it; the harness (the
fake transport, the LLM stub, the hardening install, the re-dated fixture posts and the canned
declarations) is ``tests/mantic_e2e_harness.py``.

What the run proves, per assertion group below: the five questions are fetched through the Mantic
client with no ``forecast_type`` parameter (the difference that made the framework's default fetch
return zero quantitative questions); all five are forecast and published to the Mantic API with the
payload shape that platform validates, including a 451-point CDF the server's own rules accept; the
12-bin date question is published on its epoch-seconds view as a 13-value CDF with the comment
rendering dates; and nothing in the run contacts metaculus.com.

Two of the five take the PER-BIN path (``numeric.config.elicit_per_bin``, a Mantic grid of 31 bins or
fewer). The date question's members agree on one day and put 0 on the three weekend days, which
proves those bins publish at exactly the platform minimum. Post 643's members (21 count bins, open
ceiling) are certain of ``1``, ``3`` and ``above_range`` respectively, the aggregation cliff: the
pointwise median of three sharp members is the middle member outright, so per-bin members are pooled
by the pointwise MEAN of their CDFs and each believed cell publishes about a third. The 450-bin
question stays on percentiles and on the median, which the ``method=`` field pins.

A second run, with the bot's own forecast already standing on one question, proves the
already-forecast skip the first run cannot: the workflow fires hourly, so after its first run that is
the state of nearly every question, and a skip that failed silently would re-spend the whole ensemble
every hour. That run makes no forecast POST, no comment POST and no forecaster call for that question
while the others still publish.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from datetime import timedelta

import numpy as np
import pytest
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport
from forecasting_tools.data_models.numeric_report import DateReport, DiscreteReport
from forecasting_tools.helpers.metaculus_client import MetaculusClient

from metaculus_bot import publish_hardening
from metaculus_bot.constants import (
    BINARY_PROB_MAX,
    BINARY_PROB_MIN,
    MANTIC_API_BASE_URL,
    MANTIC_OUT_OF_RANGE_TAIL_FLOOR,
    MANTIC_SITE_URL,
    MANTIC_TOURNAMENT_ID,
    MC_PROB_MAX,
    MC_PROB_MIN,
)
from metaculus_bot.mantic import ManticClient
from metaculus_bot.numeric.config import grid_step_constraints
from tests.mantic_e2e_harness import (
    ALL_POST_IDS,
    BINARY_MEDIAN,
    BINARY_PRIOR_FORECAST_VALUES,
    BITCOIN_MEDIAN_RANGE,
    BITCOIN_MEMBER_NORMALS_BELOW_THE_RANGE,
    COUNT_BELIEVED_BINS,
    COUNT_LABELS,
    DATE_CERTAIN_BIN,
    DATE_CERTAIN_DAY,
    DATE_CERTAIN_LABEL,
    DATE_LABELS,
    DATE_MEMBER_CERTAINTIES,
    DATE_WEEKEND_BINS,
    DATE_WEEKEND_LABELS,
    EXPECTED_AUTH,
    FAKE_TOKEN,
    FORECAST_QUESTION_COUNT,
    FORECASTS_PER_QUESTION,
    MC_MODAL_OPTION_INDEX,
    POSTS_URL,
    PRIOR_FORECAST_POST_ID,
    STILL_FRESH_POST_IDS,
    ManticRun,
    RecordedRequest,
    future_dated_posts,
    install_fake_transport,
    iso,
    run_mantic_mode,
)
from tests.mantic_fakes import (
    BINARY_POST_ID,
    COARSE_DISCRETE_POST_ID,
    DATE_POST_ID,
    DISCRETE_POST_ID,
    MULTIPLE_CHOICE_POST_ID,
    with_prior_forecast,
)
from tests.pipeline_test_helpers import assert_server_accepts_cdf, server_min_step

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def mantic_run() -> Iterator[ManticRun]:
    """The preseason's first run: every question fresh, every supported one forecast and published."""
    yield from run_mantic_mode(future_dated_posts())


@pytest.fixture(scope="module")
def mantic_run_after_prior_forecast() -> Iterator[ManticRun]:
    """The hourly workflow's steady state: the bot's own forecast already stands on one question.

    The binary question carries a prior forecast of this run's own median, the way the platform
    reports it once the previous scheduled run has published; the other three posts are the same
    re-dated copies the first run sees.
    """
    posts = [
        with_prior_forecast(post, forecast_values=BINARY_PRIOR_FORECAST_VALUES)
        if post["id"] == PRIOR_FORECAST_POST_ID
        else post
        for post in future_dated_posts()
    ]
    yield from run_mantic_mode(posts)


@pytest.fixture(scope="module")
def mantic_run_with_members_below_the_range() -> Iterator[ManticRun]:
    """The same fresh run with the discrete question's members piled below its range: the aggregate's
    lower tail is fat, so the Mantic floor can raise the upper tail only as far as the interior allows."""
    yield from run_mantic_mode(future_dated_posts(), discrete_normals=BITCOIN_MEMBER_NORMALS_BELOW_THE_RANGE)


@pytest.fixture
def mantic_posts_transport(monkeypatch: pytest.MonkeyPatch) -> list[RecordedRequest]:
    """The same fake transport, for the fetch-only comparison between the two clients."""
    monkeypatch.setattr(MetaculusClient, "_sleep_between_requests", lambda self: None)
    return install_fake_transport(monkeypatch, future_dated_posts())


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


class TestTheFetchGoesThroughTheManticClient:
    def test_the_fetch_walks_pages_until_an_empty_one(self, mantic_run: ManticRun) -> None:
        """Two question-list GETs: the first page carries all four posts, the second is empty and
        stops the walk. The client asks for a ceiling rather than trusting Mantic's ``next`` link."""
        gets = mantic_run.posts_requests()
        assert len(gets) == 2, f"expected a full page then an empty one, got {[r.url for r in gets]}"
        for recorded in gets:
            assert recorded.url.startswith(POSTS_URL)
        first_offset = int(gets[0].query["offset"][0])
        second_offset = int(gets[1].query["offset"][0])
        assert first_offset == 0
        assert second_offset == int(gets[0].query["limit"][0])

    def test_the_get_asks_for_the_open_questions_of_the_preseason_tournament(self, mantic_run: ManticRun) -> None:
        query = mantic_run.posts_requests()[0].query
        assert query["tournaments"] == [MANTIC_TOURNAMENT_ID]
        assert query["statuses"] == ["open"]

    def test_the_get_omits_the_forecast_type_parameter(self, mantic_run: ManticRun) -> None:
        """Mantic's vocabulary for this filter is ``quantitative``, so the framework's default value
        loses the 450-bin question entirely."""
        assert "forecast_type" not in mantic_run.posts_requests()[0].query

    def test_every_request_carries_the_mantic_bot_token(self, mantic_run: ManticRun) -> None:
        for recorded in mantic_run.requests:
            assert recorded.headers.get("Authorization") == EXPECTED_AUTH, (
                f"{recorded.method} {recorded.url} carried {recorded.headers.get('Authorization')!r}"
            )

    def test_all_five_questions_were_parsed(self, mantic_run: ManticRun) -> None:
        """One MANTIC_QUESTION marker per parsed question, so this counts what the client returned
        rather than what the bot went on to forecast."""
        marker_lines = mantic_run.mantic_question_marker_lines()
        assert len(marker_lines) == len(mantic_run.posts)
        for post_id in ALL_POST_IDS:
            assert any(f"post={post_id} " in line for line in marker_lines)


class TestTypeRouting:
    def test_five_questions_produced_reports_of_their_own_type(self, mantic_run: ManticRun) -> None:
        assert not [r for r in mantic_run.reports if isinstance(r, BaseException)], mantic_run.reports
        assert len(mantic_run.reports) == FORECAST_QUESTION_COUNT
        assert isinstance(mantic_run.report_for(BINARY_POST_ID), BinaryReport)
        assert isinstance(mantic_run.report_for(MULTIPLE_CHOICE_POST_ID), MultipleChoiceReport)
        assert isinstance(mantic_run.report_for(DISCRETE_POST_ID), DiscreteReport)
        assert isinstance(mantic_run.report_for(DATE_POST_ID), DateReport)
        assert isinstance(mantic_run.report_for(COARSE_DISCRETE_POST_ID), DiscreteReport)

    def test_no_question_was_dropped_as_unsupported(self, mantic_run: ManticRun) -> None:
        """Until 2026-09-08 the date question was dropped here with a WARNING; Mantic's pool is 41%
        date questions, so the guard now admits it and only a conditional question would trip it."""
        warnings = [
            record.getMessage()
            for record in mantic_run.records
            if record.levelno >= logging.WARNING and "unsupported" in record.getMessage()
        ]
        assert warnings == []

    def test_the_date_report_is_published_off_the_epoch_axis(self, mantic_run: ManticRun) -> None:
        report = mantic_run.report_for(DATE_POST_ID)
        assert isinstance(report, DateReport)
        assert report.prediction.is_date is True
        assert report.prediction.lower_bound == report.question.lower_bound.timestamp()
        assert report.prediction.upper_bound == report.question.upper_bound.timestamp()


class TestThePreviouslyForecastQuestionIsSkipped:
    """The hourly workflow's steady state, and the direction with power.

    With every fixture history empty, ``already_forecasted`` is False whether ``my_forecasts`` is
    read correctly or is missing entirely (the framework derives it inside a blanket except that
    answers False), so the first run cannot tell a working skip from a no-op one. Only a question
    that already carries the bot's forecast can, and after the first scheduled run of the preseason
    window that is nearly every question on every run: a skip that fails silently re-spends the
    whole ensemble, hourly, on questions the bot has already answered.
    """

    def test_the_skip_is_logged(self, mantic_run_after_prior_forecast: ManticRun) -> None:
        assert "Skipping 1 previously forecasted questions" in mantic_run_after_prior_forecast.log_text

    def test_nothing_is_posted_for_the_forecast_question(self, mantic_run_after_prior_forecast: ManticRun) -> None:
        assert PRIOR_FORECAST_POST_ID not in mantic_run_after_prior_forecast.posted_question_ids()
        assert PRIOR_FORECAST_POST_ID not in mantic_run_after_prior_forecast.commented_post_ids()

    def test_no_forecaster_ran_on_it(self, mantic_run_after_prior_forecast: ManticRun) -> None:
        """The skip must happen before the fan-out: no LLM spend, not just no publish."""
        run = mantic_run_after_prior_forecast
        skipped_title = next(post["question"]["title"] for post in run.posts if post["id"] == PRIOR_FORECAST_POST_ID)
        assert skipped_title not in run.llm_calls.forecaster_calls
        assert len(run.llm_calls.forecaster_calls) == FORECASTS_PER_QUESTION * len(STILL_FRESH_POST_IDS)
        assert run.llm_calls.unexpected_prompts == []

    def test_the_other_questions_still_publish(self, mantic_run_after_prior_forecast: ManticRun) -> None:
        run = mantic_run_after_prior_forecast
        assert not [r for r in run.reports if isinstance(r, BaseException)], run.reports
        reported = {r.question.id_of_post for r in run.reports if isinstance(r, ForecastReport)}
        assert reported == STILL_FRESH_POST_IDS
        assert run.posted_question_ids() == STILL_FRESH_POST_IDS
        assert run.commented_post_ids() == STILL_FRESH_POST_IDS

    def test_the_skipped_question_was_fetched_and_parsed(self, mantic_run_after_prior_forecast: ManticRun) -> None:
        """It is the bot's filter that drops it, downstream of a client that parsed it: this run
        still sees all four posts, so the skip cannot be a fetch that quietly lost one."""
        run = mantic_run_after_prior_forecast
        marker_lines = run.mantic_question_marker_lines()
        assert len(marker_lines) == len(run.posts)
        assert any(f"post={PRIOR_FORECAST_POST_ID} " in line for line in marker_lines)


class TestPublishedPayloads:
    def test_one_forecast_and_one_comment_post_per_forecast_question(self, mantic_run: ManticRun) -> None:
        assert len(mantic_run.forecast_posts()) == FORECAST_QUESTION_COUNT
        assert len(mantic_run.comment_posts()) == FORECAST_QUESTION_COUNT
        for recorded in mantic_run.forecast_posts():
            assert isinstance(recorded.body, list)
            assert len(recorded.body) == 1, "the platform takes a one-element list per forecast POST"
            assert recorded.body[0]["source"] == "api"

    def test_binary_payload_is_the_median_probability(self, mantic_run: ManticRun) -> None:
        payload = mantic_run.forecast_payload(BINARY_POST_ID)
        probability = payload["probability_yes"]
        assert probability == pytest.approx(BINARY_MEDIAN)
        assert BINARY_PROB_MIN <= probability <= BINARY_PROB_MAX
        # The platform's own floor and ceiling on a submitted probability.
        assert 0.001 <= probability <= 0.999

    def test_multiple_choice_payload_covers_every_option_and_sums_to_one(self, mantic_run: ManticRun) -> None:
        options: list[str] = next(
            post["question"]["options"] for post in mantic_run.posts if post["id"] == MULTIPLE_CHOICE_POST_ID
        )
        payload = mantic_run.forecast_payload(MULTIPLE_CHOICE_POST_ID)
        per_category: dict[str, float] = payload["probability_yes_per_category"]
        assert list(per_category) == options
        assert sum(per_category.values()) == pytest.approx(1.0, abs=1e-6)
        for probability in per_category.values():
            assert MC_PROB_MIN <= probability <= MC_PROB_MAX
            assert 0.001 <= probability <= 0.999
        # The aggregate keeps the ensemble's modal option rather than flattening it.
        assert max(per_category, key=lambda option: per_category[option]) == options[MC_MODAL_OPTION_INDEX]

    def test_discrete_payload_is_a_451_point_cdf_the_server_accepts(self, mantic_run: ManticRun) -> None:
        question_json = next(post["question"] for post in mantic_run.posts if post["id"] == DISCRETE_POST_ID)
        scaling = question_json["scaling"]
        cdf_size = scaling["inbound_outcome_count"] + 1
        payload = mantic_run.forecast_payload(DISCRETE_POST_ID)
        cdf = payload["continuous_cdf"]

        assert len(cdf) == cdf_size
        probs = np.asarray(cdf, dtype=float)
        assert np.all(np.diff(probs) >= 0.0), "the CDF must be non-decreasing"
        assert_server_accepts_cdf(
            probs,
            cdf_size=cdf_size,
            open_lower=scaling["open_lower_bound"],
            open_upper=scaling["open_upper_bound"],
        )

    def test_the_discrete_cdf_is_centred_where_the_ensemble_put_it(self, mantic_run: ManticRun) -> None:
        question_json = next(post["question"] for post in mantic_run.posts if post["id"] == DISCRETE_POST_ID)
        scaling = question_json["scaling"]
        cdf = np.asarray(mantic_run.forecast_payload(DISCRETE_POST_ID)["continuous_cdf"], dtype=float)
        grid = np.linspace(scaling["range_min"], scaling["range_max"], len(cdf))
        median_value = float(grid[int(np.searchsorted(cdf, 0.5))])
        low, high = BITCOIN_MEDIAN_RANGE
        assert low <= median_value <= high, f"published median {median_value} outside {BITCOIN_MEDIAN_RANGE}"

    def test_date_payload_is_a_13_point_cdf_the_server_accepts(self, mantic_run: ManticRun) -> None:
        """Both of post 651's bounds are closed, so the CDF is pinned to exactly 0.0 and 1.0 at its
        ends, and its 12 bins must each clear the server's ``0.01 / 12`` minimum step."""
        question_json = next(post["question"] for post in mantic_run.posts if post["id"] == DATE_POST_ID)
        scaling = question_json["scaling"]
        cdf_size = scaling["inbound_outcome_count"] + 1
        assert cdf_size == 13
        cdf = mantic_run.forecast_payload(DATE_POST_ID)["continuous_cdf"]

        assert len(cdf) == cdf_size
        probs = np.asarray(cdf, dtype=float)
        assert probs[0] == 0.0
        assert probs[-1] == 1.0
        assert np.all(np.diff(probs) >= 0.0), "the CDF must be non-decreasing"
        assert_server_accepts_cdf(
            probs,
            cdf_size=cdf_size,
            open_lower=scaling["open_lower_bound"],
            open_upper=scaling["open_upper_bound"],
        )

    def test_the_per_bin_members_certain_of_september_16_put_the_mass_in_bin_8(self, mantic_run: ManticRun) -> None:
        """The oracle for the bin convention: calendar day D = range_min + k days is bin k, whose mass
        is ``cdf[k + 1] - cdf[k]`` over ``(edge_k, edge_{k+1}]``, and 2026-09-16 is k = 8. Every member
        keyed its mass to that day's label, so the pooled forecast carries their mean there."""
        question_json = next(post["question"] for post in mantic_run.posts if post["id"] == DATE_POST_ID)
        edges = question_json["scaling"]["continuous_range"]
        assert edges[DATE_CERTAIN_BIN] == iso(DATE_CERTAIN_DAY)
        assert edges[DATE_CERTAIN_BIN + 1] == iso(DATE_CERTAIN_DAY + timedelta(days=1))
        assert DATE_LABELS[DATE_CERTAIN_BIN] == DATE_CERTAIN_LABEL

        probs = np.asarray(mantic_run.forecast_payload(DATE_POST_ID)["continuous_cdf"], dtype=float)
        bin_masses = np.diff(probs)
        assert int(np.argmax(bin_masses)) == DATE_CERTAIN_BIN
        assert bin_masses[DATE_CERTAIN_BIN] > 0.85
        assert bin_masses[DATE_CERTAIN_BIN] == pytest.approx(float(np.mean(DATE_MEMBER_CERTAINTIES)), abs=0.01)
        # The other trading days share the remainder; nothing else on the grid carries real mass.
        others = np.delete(bin_masses, DATE_CERTAIN_BIN)
        assert np.all(others < 0.02)

    def test_the_weekend_bins_publish_at_exactly_the_platform_minimum(self, mantic_run: ManticRun) -> None:
        """The point of per-bin elicitation on this question: a percentile declaration cannot say
        "zero on 12, 13 and 19 September", so PCHIP used to leak about a quarter of the mass onto
        days that cannot resolve. Every member declared 0 there; the floor blend lifts each to the
        server's ``round(0.01 / 12, 9)`` plus its 1e-9 margin, and the pool of three floors is the floor."""
        probs = np.asarray(mantic_run.forecast_payload(DATE_POST_ID)["continuous_cdf"], dtype=float)
        bin_masses = np.diff(probs)
        for weekend_bin in DATE_WEEKEND_BINS:
            assert DATE_LABELS[weekend_bin] in DATE_WEEKEND_LABELS
            assert server_min_step(12) <= bin_masses[weekend_bin] <= server_min_step(12) + 2e-9, bin_masses[weekend_bin]

    def test_the_date_comment_renders_dates_not_epoch_seconds(self, mantic_run: ManticRun) -> None:
        text: str = mantic_run.comment_payload(DATE_POST_ID)["text"]
        assert "2026-09-16" in text
        # An epoch second in 2026 is a ten-digit number starting 17; none may reach the reader.
        assert re.search(r"\b17\d{8}(?:\.\d+)?\b", text) is None, text[:2000]

    @pytest.mark.parametrize("post_id", list(ALL_POST_IDS))
    def test_each_comment_targets_its_own_post(self, mantic_run: ManticRun, post_id: int) -> None:
        payload = mantic_run.comment_payload(post_id)
        assert payload["on_post"] == post_id
        # The framework's comment body opens with the SUMMARY heading (after one leading newline).
        assert payload["text"].lstrip().startswith("# SUMMARY")
        assert payload["is_private"] is True
        assert payload["included_forecast"] is True


class TestDateTelemetry:
    """The date question is counted as its own type in the run log, on the epoch axis, elicited per bin."""

    def test_each_member_leaves_a_per_bin_member_forecast_line_with_qtype_date(self, mantic_run: ManticRun) -> None:
        """``qtype`` is the QUESTION type; ``elicitation=pmf`` says ``raw`` and ``published`` are the
        platform's 14-entry PMF, ``[below, p_0, ..., p_11, above]``, not percentile pairs."""
        lines = [line for line in mantic_run.marker_lines("MEMBER_FORECAST:") if f"question={DATE_POST_ID} " in line]
        assert len(lines) == FORECASTS_PER_QUESTION
        raw_certainties = set()
        for line in lines:
            assert " role=member qtype=date " in line
            # Both bounds closed: no mass outside the range, then the elicitation field closes the line.
            assert line.endswith(" oor_low=0.000000 oor_high=0.000000 elicitation=pmf")
            raw = json.loads(line.split(" raw=", 1)[1].split(" ", 1)[0])
            published = json.loads(line.split(" published=", 1)[1].split(" ", 1)[0])
            assert len(raw) == len(published) == len(DATE_LABELS) + 2
            assert raw[0] == raw[-1] == 0.0
            assert published[0] == published[-1] == 0.0
            raw_certainties.add(raw[1 + DATE_CERTAIN_BIN])
            for weekend_bin in DATE_WEEKEND_BINS:
                assert raw[1 + weekend_bin] == 0.0
                assert published[1 + weekend_bin] == pytest.approx(server_min_step(12), abs=2e-9)
        assert raw_certainties == set(DATE_MEMBER_CERTAINTIES)

    def test_each_per_bin_member_was_read_off_the_block_rung_as_a_pmf(self, mantic_run: ManticRun) -> None:
        """``EXTRACTION_RUNG`` names the BLOCK type the ladder parsed, ``pmf``, beside member lines that say
        ``qtype=date``; the question type is joined from those."""
        for post_id in (DATE_POST_ID, COARSE_DISCRETE_POST_ID):
            lines = [line for line in mantic_run.marker_lines("EXTRACTION_RUNG:") if f"question={post_id} " in line]
            assert len(lines) == FORECASTS_PER_QUESTION, lines
            for line in lines:
                assert " qtype=pmf rung=block block_present=True" in line

    def test_the_aggregate_marker_names_the_date_grid_and_the_pool(self, mantic_run: ManticRun) -> None:
        """Both of post 651's bounds are closed, so the tail floor has nothing to move: raw and
        published tails are zero and the floor reads zero. Per-bin members are pooled by the mean."""
        lines = [line for line in mantic_run.marker_lines("NUMERIC_AGGREGATE:") if f"question={DATE_POST_ID} " in line]
        assert lines == [
            f"NUMERIC_AGGREGATE: question={DATE_POST_ID} qtype=date cdf_size=13 oor_low=0.000000 oor_high=0.000000 "
            "oor_low_raw=0.000000 oor_high_raw=0.000000 tail_floor=0.000000 method=mean"
        ]

    def test_the_percentile_question_keeps_the_median(self, mantic_run: ManticRun) -> None:
        """450 bins is far above the per-bin threshold, so post 650's percentile members are medianed
        exactly as before: the pool did not widen to percentile members."""
        lines = [
            line for line in mantic_run.marker_lines("NUMERIC_AGGREGATE:") if f"question={DISCRETE_POST_ID} " in line
        ]
        assert len(lines) == 1
        assert " qtype=numeric cdf_size=451 " in lines[0]
        assert lines[0].endswith(" method=median")
        member_lines = [
            line for line in mantic_run.marker_lines("MEMBER_FORECAST:") if f"question={DISCRETE_POST_ID} " in line
        ]
        assert len(member_lines) == FORECASTS_PER_QUESTION
        assert all("elicitation" not in line for line in member_lines)

    def test_the_mantic_question_marker_still_names_the_date_question(self, mantic_run: ManticRun) -> None:
        (line,) = [line for line in mantic_run.mantic_question_marker_lines() if f"post={DATE_POST_ID} " in line]
        assert "type=date" in line
        assert "cdf_size=13" in line
        assert "date_granularity=day" in line


class TestTheOutOfRangeTailFloor:
    """The published discrete aggregate carries the Mantic tail floor beyond each open bound.

    Post 650 has both bounds open and every member's thirteen percentiles inside the range, so
    the aggregate's own tails are the structural 1% and the floor raises each to
    ``MANTIC_OUT_OF_RANGE_TAIL_FLOOR``. The payload the server receives is what is checked, so
    the floor is proven on the wire and not only at the seam.
    """

    def test_the_published_discrete_cdf_carries_the_floor_beyond_each_open_bound(self, mantic_run: ManticRun) -> None:
        question_json = next(post["question"] for post in mantic_run.posts if post["id"] == DISCRETE_POST_ID)
        scaling = question_json["scaling"]
        assert scaling["open_lower_bound"] is True
        assert scaling["open_upper_bound"] is True
        probs = np.asarray(mantic_run.forecast_payload(DISCRETE_POST_ID)["continuous_cdf"], dtype=float)
        assert probs[0] == MANTIC_OUT_OF_RANGE_TAIL_FLOOR
        assert 1.0 - probs[-1] == pytest.approx(MANTIC_OUT_OF_RANGE_TAIL_FLOOR)

    def test_the_discrete_aggregate_marker_records_raw_and_published_tails(self, mantic_run: ManticRun) -> None:
        lines = [
            line for line in mantic_run.marker_lines("NUMERIC_AGGREGATE:") if f"question={DISCRETE_POST_ID} " in line
        ]
        assert lines == [
            f"NUMERIC_AGGREGATE: question={DISCRETE_POST_ID} qtype=numeric cdf_size=451 oor_low=0.050000 "
            "oor_high=0.050000 oor_low_raw=0.010000 oor_high_raw=0.010000 tail_floor=0.050000 method=median"
        ]

    def test_the_members_keep_their_own_tails(self, mantic_run: ManticRun) -> None:
        lines = [
            line for line in mantic_run.marker_lines("MEMBER_FORECAST:") if f"question={DISCRETE_POST_ID} " in line
        ]
        assert len(lines) == FORECASTS_PER_QUESTION
        for line in lines:
            assert line.endswith(" oor_low=0.010000 oor_high=0.010000")


class TestTheTailFloorWhenOneTailIsFat:
    """The floor on the wire when the aggregate already piles most of its mass beyond one open bound.

    Every member's thirteen percentiles sit below post 650's range, so the aggregate carries about
    97.6% below the open lower bound and the builder's 1% above the upper. The server needs each of
    the 450 bins to keep its min step, so the upper tail can rise only to what that interior leaves
    (about 1.4%), never to 5%, and the fat tail is never reduced. Before the feasibility cap this
    shape made the seam's rebuild raise and the question was forfeited; here the payload the server
    receives passes its rules and the marker records the level actually applied.
    """

    def test_the_published_cdf_keeps_the_fat_tail_and_raises_the_thin_one_as_far_as_the_interior_allows(
        self, mantic_run_with_members_below_the_range: ManticRun
    ) -> None:
        probs = np.asarray(
            mantic_run_with_members_below_the_range.forecast_payload(DISCRETE_POST_ID)["continuous_cdf"], dtype=float
        )
        assert_server_accepts_cdf(probs, cdf_size=451, open_lower=True, open_upper=True)
        min_step, _ = grid_step_constraints(451)
        least_interior = 450 * min_step
        assert probs[0] > 0.95, "the members' mass sits below the range"
        room = 1.0 - probs[0] - least_interior
        assert 0.01 < room < MANTIC_OUT_OF_RANGE_TAIL_FLOOR
        assert 1.0 - probs[-1] == pytest.approx(room)
        assert float(probs[-1] - probs[0]) == pytest.approx(least_interior)

    def test_the_aggregate_marker_records_the_level_actually_applied(
        self, mantic_run_with_members_below_the_range: ManticRun
    ) -> None:
        run = mantic_run_with_members_below_the_range
        probs = np.asarray(run.forecast_payload(DISCRETE_POST_ID)["continuous_cdf"], dtype=float)
        (line,) = [line for line in run.marker_lines("NUMERIC_AGGREGATE:") if f"question={DISCRETE_POST_ID} " in line]
        fields = dict(token.split("=", 1) for token in line.split(": ", 1)[1].split(" "))
        min_step, _ = grid_step_constraints(451)
        room = 1.0 - probs[0] - 450 * min_step
        assert fields["oor_low"] == fields["oor_low_raw"] == f"{probs[0]:.6f}", "the fat tail is published as built"
        assert fields["oor_high_raw"] == "0.010000"
        assert float(fields["oor_high"]) == pytest.approx(room, abs=1e-6)
        assert fields["tail_floor"] == f"{room:.6f}", "the floor recorded is the level applied, not the nominal 5%"

    def test_every_question_still_publishes(self, mantic_run_with_members_below_the_range: ManticRun) -> None:
        run = mantic_run_with_members_below_the_range
        assert len(run.forecast_posts()) == FORECAST_QUESTION_COUNT
        assert len(run.comment_posts()) == FORECAST_QUESTION_COUNT
        assert not run.llm_calls.unexpected_prompts


class TestThePerBinPoolOnTheCoarseCountQuestion:
    """Post 643 on the wire: three per-bin members certain of ``1``, ``3`` and ``above_range``.

    Each declared 1.0 on its cell and 0 everywhere else, so the floor blend alone fills the other
    cells (each at the server minimum plus the 1e-9 margin) and the believed cell keeps about 0.99.
    The pointwise MEDIAN of those three CDFs would publish the middle member outright: 0.99 on ``3``,
    the floor on ``1`` and 0.001 above the range, about -230 Series 1 points whenever ``1`` or the
    escape resolves. The pointwise MEAN is the CDF of the mixture PMF, so each believed cell gets a
    third of the mass, every other bin stays at the floor (the mean of three floors), and the
    above-range third is already past the Mantic 5% tail floor, which therefore moves nothing.
    """

    def test_the_payload_is_a_22_value_cdf_the_server_accepts(self, mantic_run: ManticRun) -> None:
        question_json = next(post["question"] for post in mantic_run.posts if post["id"] == COARSE_DISCRETE_POST_ID)
        scaling = question_json["scaling"]
        assert scaling["inbound_outcome_count"] + 1 == 22
        assert scaling["open_lower_bound"] is False
        assert scaling["open_upper_bound"] is True
        probs = np.asarray(mantic_run.forecast_payload(COARSE_DISCRETE_POST_ID)["continuous_cdf"], dtype=float)
        assert len(probs) == 22
        assert probs[0] == 0.0
        assert probs[-1] <= 0.999
        assert_server_accepts_cdf(probs, cdf_size=22, open_lower=False, open_upper=True)

    def test_each_believed_cell_carries_a_third_and_every_other_bin_the_floor(self, mantic_run: ManticRun) -> None:
        probs = np.asarray(mantic_run.forecast_payload(COARSE_DISCRETE_POST_ID)["continuous_cdf"], dtype=float)
        bin_masses = np.diff(probs)
        above_range = 1.0 - probs[-1]
        for believed_bin in COUNT_BELIEVED_BINS:
            assert 0.30 <= bin_masses[believed_bin] <= 0.36, bin_masses[believed_bin]
        assert 0.30 <= above_range <= 0.36, above_range
        others = np.delete(bin_masses, list(COUNT_BELIEVED_BINS))
        assert np.all(others <= server_min_step(21) + 2e-9), others.max()
        assert np.all(others >= server_min_step(21)), others.min()

    def test_the_aggregate_marker_records_the_pool_and_an_inert_tail_floor(self, mantic_run: ManticRun) -> None:
        probs = np.asarray(mantic_run.forecast_payload(COARSE_DISCRETE_POST_ID)["continuous_cdf"], dtype=float)
        (line,) = [
            line
            for line in mantic_run.marker_lines("NUMERIC_AGGREGATE:")
            if f"question={COARSE_DISCRETE_POST_ID} " in line
        ]
        fields = dict(token.split("=", 1) for token in line.split(": ", 1)[1].split(" "))
        assert fields["qtype"] == "numeric"
        assert fields["cdf_size"] == "22"
        assert fields["method"] == "mean"
        assert fields["oor_low"] == fields["oor_low_raw"] == "0.000000"
        assert float(fields["oor_high"]) == pytest.approx(1.0 - probs[-1], abs=1e-6)
        assert fields["oor_high"] == fields["oor_high_raw"]
        assert float(fields["oor_high"]) > MANTIC_OUT_OF_RANGE_TAIL_FLOOR
        assert fields["tail_floor"] == "0.000000"

    def test_each_member_line_carries_its_sharp_declaration_and_the_blended_publication(
        self, mantic_run: ManticRun
    ) -> None:
        lines = [
            line
            for line in mantic_run.marker_lines("MEMBER_FORECAST:")
            if f"question={COARSE_DISCRETE_POST_ID} " in line
        ]
        assert len(lines) == FORECASTS_PER_QUESTION
        believed_cells = set()
        for line in lines:
            assert " role=member qtype=numeric " in line
            assert line.endswith(" elicitation=pmf")
            raw = json.loads(line.split(" raw=", 1)[1].split(" ", 1)[0])
            published = json.loads(line.split(" published=", 1)[1].split(" ", 1)[0])
            assert len(raw) == len(published) == len(COUNT_LABELS) + 2
            assert raw.count(1.0) == 1
            assert raw.count(0.0) == len(raw) - 1
            believed_cells.add(raw.index(1.0))
            assert published[raw.index(1.0)] > 0.98
            assert published[0] == 0.0  # closed lower bound
            oor_high = float(line.rsplit("oor_high=", 1)[1].split(" ", 1)[0])
            assert oor_high == pytest.approx(published[-1], abs=1e-6)
        assert believed_cells == {1 + 1, 1 + 3, len(COUNT_LABELS) + 1}

    def test_the_mantic_question_marker_names_the_coarse_grid(self, mantic_run: ManticRun) -> None:
        (line,) = [
            line for line in mantic_run.mantic_question_marker_lines() if f"post={COARSE_DISCRETE_POST_ID} " in line
        ]
        assert "type=discrete" in line
        assert "cdf_size=22" in line

    def test_the_report_and_comment_are_the_coarse_questions_own(self, mantic_run: ManticRun) -> None:
        assert isinstance(mantic_run.report_for(COARSE_DISCRETE_POST_ID), DiscreteReport)
        assert mantic_run.comment_payload(COARSE_DISCRETE_POST_ID)["on_post"] == COARSE_DISCRETE_POST_ID


class TestNothingReachesMetaculus:
    def test_every_request_went_to_the_mantic_api(self, mantic_run: ManticRun) -> None:
        assert mantic_run.requests, "the run made no requests at all"
        for recorded in mantic_run.requests:
            assert recorded.url.startswith(f"{MANTIC_API_BASE_URL}/"), recorded.url

    def test_no_request_mentions_metaculus(self, mantic_run: ManticRun) -> None:
        for recorded in mantic_run.requests:
            assert "metaculus.com" not in recorded.url, recorded.url

    def test_the_reports_carry_mantic_page_urls(self, mantic_run: ManticRun) -> None:
        for post_id in ALL_POST_IDS:
            page_url = mantic_run.report_for(post_id).question.page_url
            assert page_url is not None
            assert page_url.startswith(f"{MANTIC_SITE_URL}/questions/"), page_url


class TestPublishHardeningIsOnThePath:
    def test_every_publish_post_carries_the_forced_timeout(self, mantic_run: ManticRun) -> None:
        publish_requests = mantic_run.forecast_posts() + mantic_run.comment_posts()
        assert len(publish_requests) == 2 * FORECAST_QUESTION_COUNT
        for recorded in publish_requests:
            assert recorded.send_kwargs.get("timeout") == publish_hardening.PUBLISH_POST_TIMEOUT, (
                f"{recorded.path} carried timeout={recorded.send_kwargs.get('timeout')!r}"
            )

    def test_the_question_list_get_is_bounded_too(self, mantic_run: ManticRun) -> None:
        assert mantic_run.posts_requests()[0].send_kwargs.get("timeout") is not None


class TestTheEnsembleFannedOut:
    def test_three_forecasters_ran_on_each_question(self, mantic_run: ManticRun) -> None:
        calls = mantic_run.llm_calls.forecaster_calls
        assert len(calls) == FORECASTS_PER_QUESTION * FORECAST_QUESTION_COUNT
        assert {calls.count(title) for title in set(calls)} == {FORECASTS_PER_QUESTION}

    def test_no_parser_or_stacker_call_was_needed(self, mantic_run: ManticRun) -> None:
        """Every value came off the structured block (extraction rung 1) and every spread sat below
        its stacking threshold, so the twelve forecaster calls were the only LLM calls."""
        assert mantic_run.llm_calls.unexpected_prompts == []


class TestTheFrameworkDefaultTypeFilterLosesTheQuantitativeQuestion:
    """The negative control for the fetch fix, against the same fake transport.

    An unmodified ``MetaculusClient`` pointed at the Mantic base URL sends
    ``forecast_type=binary,numeric,...``; Mantic answers that with none of its quantitative
    questions, so the 450-bin bitcoin question never reaches the bot at all. ``ManticClient`` omits
    the parameter and gets it.
    """

    def test_the_framework_default_drops_post_650(self, mantic_posts_transport: list[RecordedRequest]) -> None:
        plain_client = MetaculusClient(base_url=MANTIC_API_BASE_URL, token=FAKE_TOKEN)
        fetched = plain_client.get_all_open_questions_from_tournament(MANTIC_TOURNAMENT_ID)
        post_ids = {question.id_of_post for question in fetched}
        assert DISCRETE_POST_ID not in post_ids
        assert COARSE_DISCRETE_POST_ID not in post_ids
        assert BINARY_POST_ID in post_ids
        assert "forecast_type" in mantic_posts_transport[0].query

    def test_the_mantic_client_keeps_post_650(self, mantic_posts_transport: list[RecordedRequest]) -> None:
        fetched = ManticClient(token=FAKE_TOKEN).get_all_open_questions_from_tournament(MANTIC_TOURNAMENT_ID)
        post_ids = {question.id_of_post for question in fetched}
        assert DISCRETE_POST_ID in post_ids
        assert post_ids == set(ALL_POST_IDS)
        assert "forecast_type" not in mantic_posts_transport[0].query
