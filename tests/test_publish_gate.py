"""The pre-publish close-time gate (metaculus_bot/publish_gate.py) and its wiring.

Regression cohort for q45085 (2026-08-03): a question fetched 22 seconds before its
close, forecast at full 3/3 strength, then POSTed and rejected
``405 {"error":"Question 45085 is already closed to forecasting !"}``. The 405
propagated out of ft's per-question handler into ``cli.main``'s
``log_report_summary``, which raised before the alertable block — so that run is the
only forecasting run since 2026-07-26 with no end-of-run summary, and every
degradation counter read zero.

Three invariants here, and the third is the one that keeps this cheap:

1. A closed question is skipped, warned once with a harvestable marker, counted, and
   the run continues.
2. An open question takes the byte-identical path it took before the gate existed —
   same call, same arguments, same return value.
3. The gate fails OPEN on anything it cannot read (no question, no close time): a
   silently withheld publish would be strictly worse than the 405 it replaces.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from forecasting_tools.data_models.binary_report import BinaryReport
from forecasting_tools.data_models.multiple_choice_report import (
    MultipleChoiceReport,
    PredictedOption,
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import NumericDistribution, NumericReport, Percentile
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
    QuestionState,
)
from forecasting_tools.helpers import metaculus_client as _ft_metaculus_client

from metaculus_bot import publish_gate, publish_hardening
from metaculus_bot.numeric.config import STANDARD_PERCENTILES

_NOW = datetime(2026, 8, 3, 12, 5, 6, tzinfo=UTC)
_CLOSE = datetime(2026, 8, 3, 12, 0, 0, tzinfo=UTC)
_DECLARED_VALUES = [5, 8, 12, 18, 28, 42, 50, 58, 72, 82, 88, 92, 96]


def _question(
    *,
    close_time: datetime | None,
    state: QuestionState | None = QuestionState.OPEN,
    qid: int = 45085,
) -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will the question close before we publish?",
        id_of_question=qid,
        id_of_post=45058,
        close_time=close_time,
        open_time=datetime(2026, 8, 3, 11, 0, 0, tzinfo=UTC),
        state=state,
    )


def _publishable_reports() -> list[BinaryReport | NumericReport | MultipleChoiceReport]:
    """One real report per published question type, ready to drive ft's own publish body.

    Each type publishes through its OWN method, so an ordering check that covers only
    one of them leaves the other two free to drift.
    """
    numeric_question = NumericQuestion(
        question_text="How many?",
        id_of_question=102,
        id_of_post=102,
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=False,
        open_upper_bound=False,
        zero_point=None,
        unit_of_measure="units",
        cdf_size=201,
    )
    declared = [Percentile(percentile=p, value=v) for p, v in zip(STANDARD_PERCENTILES, _DECLARED_VALUES, strict=True)]
    mc_question = MultipleChoiceQuestion(
        question_text="Which?",
        id_of_question=103,
        id_of_post=103,
        options=["A", "B", "C"],
    )
    return [
        BinaryReport(question=_question(close_time=_NOW + timedelta(days=1)), prediction=0.5, explanation="# pin"),
        NumericReport(
            question=numeric_question,
            prediction=NumericDistribution.from_question(declared, numeric_question),
            explanation="# pin",
        ),
        MultipleChoiceReport(
            question=mc_question,
            prediction=PredictedOptionList(
                predicted_options=[
                    PredictedOption(option_name="A", probability=0.5),
                    PredictedOption(option_name="B", probability=0.3),
                    PredictedOption(option_name="C", probability=0.2),
                ]
            ),
            explanation="# pin",
        ),
    ]


@pytest.fixture(autouse=True)
def _zero_counter():
    publish_gate.reset_publish_skipped_closed()
    yield
    publish_gate.reset_publish_skipped_closed()


class TestClosedToForecasting:
    """The pure decision, with no logging or counting in the way."""

    def test_close_time_already_passed_is_closed(self):
        verdict = publish_gate.closed_to_forecasting(_question(close_time=_CLOSE), _NOW)
        assert verdict is not None
        assert verdict.reason == publish_gate.REASON_CLOSE_TIME_PASSED
        assert verdict.overdue_s == pytest.approx(306.0)

    def test_open_question_is_not_closed(self):
        q = _question(close_time=_NOW + timedelta(minutes=30))
        assert publish_gate.closed_to_forecasting(q, _NOW) is None

    def test_exactly_at_close_is_closed(self):
        # The boundary belongs to the closed side: Metaculus stops accepting AT the
        # close time, and a publish landing on the tick would 405.
        assert publish_gate.closed_to_forecasting(_question(close_time=_NOW), _NOW) is not None

    def test_naive_close_time_is_read_as_utc(self):
        # Some ft call sites still hand naive datetimes through; treating one as local
        # time would move the boundary by hours in either direction.
        q = _question(close_time=datetime(2026, 8, 3, 12, 0, 0))
        verdict = publish_gate.closed_to_forecasting(q, _NOW)
        assert verdict is not None
        assert verdict.overdue_s == pytest.approx(306.0)

    def test_no_close_time_is_not_closed(self):
        # A question with no deadline has no window to miss — fail open.
        assert publish_gate.closed_to_forecasting(_question(close_time=None), _NOW) is None

    @pytest.mark.parametrize("state", [QuestionState.CLOSED, QuestionState.RESOLVED])
    def test_closed_state_beats_a_future_close_time(self, state: QuestionState):
        # The complement of the clock check: an admin closing or resolving a question
        # ahead of its scheduled close leaves close_time in the future while the API
        # rejects the forecast. Overdue is negative here, which is the point.
        q = _question(close_time=_NOW + timedelta(hours=2), state=state)
        verdict = publish_gate.closed_to_forecasting(q, _NOW)
        assert verdict is not None
        assert verdict.reason == publish_gate.REASON_STATE_CLOSED
        assert verdict.overdue_s == pytest.approx(-7200.0)

    def test_upcoming_state_is_deliberately_not_gated(self):
        # Documented non-goal: an UPCOMING question cannot reach the publish path from
        # any mode the bot runs, so the gate does not invent a branch for it.
        q = _question(close_time=_NOW + timedelta(hours=2), state=QuestionState.UPCOMING)
        assert publish_gate.closed_to_forecasting(q, _NOW) is None


class TestMarkerFormat:
    """Pinned against scripts/telemetry/markers.py's PUBLISH_SKIPPED_CLOSED spec."""

    def test_full_marker_line(self):
        q = _question(close_time=_CLOSE)
        verdict = publish_gate.closed_to_forecasting(q, _NOW)
        assert verdict is not None
        assert publish_gate.format_publish_skipped_marker(q, verdict, _NOW) == (
            "PUBLISH_SKIPPED_CLOSED: question=45085 reason=close_time_passed "
            "close_time=2026-08-03T12:00:00+00:00 now=2026-08-03T12:05:06+00:00 "
            "overdue_s=306 state=open"
        )

    def test_state_closed_without_a_close_time_renders_na(self):
        q = _question(close_time=None, state=QuestionState.CLOSED)
        verdict = publish_gate.closed_to_forecasting(q, _NOW)
        assert verdict is not None
        assert publish_gate.format_publish_skipped_marker(q, verdict, _NOW) == (
            "PUBLISH_SKIPPED_CLOSED: question=45085 reason=state_closed "
            "close_time=n/a now=2026-08-03T12:05:06+00:00 overdue_s=n/a state=closed"
        )


class TestSkipPublishIfClosed:
    def test_closed_question_counts_and_warns_once(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.publish_gate"):
            assert publish_gate.skip_publish_if_closed(_question(close_time=_CLOSE), _NOW) is True
        assert publish_gate.publish_skipped_closed_count() == 1
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert warnings[0].startswith("PUBLISH_SKIPPED_CLOSED: question=45085 reason=close_time_passed")

    def test_open_question_neither_counts_nor_warns(self, caplog: pytest.LogCaptureFixture):
        q = _question(close_time=_NOW + timedelta(hours=1))
        with caplog.at_level(logging.WARNING, logger="metaculus_bot.publish_gate"):
            assert publish_gate.skip_publish_if_closed(q, _NOW) is False
        assert publish_gate.publish_skipped_closed_count() == 0
        assert caplog.records == []

    def test_missing_question_fails_open(self):
        # The seam belongs to a third-party class; if it ever stops carrying a question
        # we publish unguarded rather than silently withholding a forecast.
        assert publish_gate.skip_publish_if_closed(None, _NOW) is False
        assert publish_gate.publish_skipped_closed_count() == 0

    def test_each_skipped_question_counts_separately(self):
        publish_gate.skip_publish_if_closed(_question(close_time=_CLOSE, qid=45085), _NOW)
        publish_gate.skip_publish_if_closed(_question(close_time=_CLOSE, qid=45093), _NOW)
        assert publish_gate.publish_skipped_closed_count() == 2

    def test_reset_zeroes_the_counter(self):
        publish_gate.skip_publish_if_closed(_question(close_time=_CLOSE), _NOW)
        publish_gate.reset_publish_skipped_closed()
        assert publish_gate.publish_skipped_closed_count() == 0

    def test_defaults_to_wall_clock_when_no_now_is_passed(self):
        # Prod passes no `now`; a close time far in the past must still gate.
        q = _question(close_time=datetime(2020, 1, 1, tzinfo=UTC))
        assert publish_gate.skip_publish_if_closed(q) is True


class TestGateRidesTheRealPublishSeam:
    """The gate must sit on ft's own async publish method, the last seam that knows
    which question is being published (publish_hardening layer 4)."""

    @pytest.fixture
    def offloaded_publish(self, monkeypatch: pytest.MonkeyPatch):
        """Install a recording stand-in for ft's publish, then wrap it with the offload."""
        published: list[Any] = []

        async def recording_publish(self: Any, metaculus_client: Any = None) -> str:
            published.append(self)
            return "published"

        monkeypatch.setattr(BinaryReport, publish_hardening._PUBLISH_METHOD, recording_publish)
        publish_hardening.apply_report_publish_offload()
        yield published
        monkeypatch.setattr(
            BinaryReport,
            publish_hardening._PUBLISH_METHOD,
            BinaryReport.__dict__[publish_hardening._PUBLISH_METHOD],
        )

    class _Report:
        def __init__(self, question: BinaryQuestion) -> None:
            self.question = question

    @pytest.mark.asyncio
    async def test_closed_question_never_reaches_the_post(self, offloaded_publish: list[Any]) -> None:
        report = self._Report(_question(close_time=datetime(2020, 1, 1, tzinfo=UTC)))
        result = await BinaryReport.publish_report_to_metaculus(report)  # type: ignore[arg-type]
        assert result is None, "a skipped publish returns None, matching ft's own return type"
        assert offloaded_publish == [], "the closed question must not have been POSTed"
        assert publish_gate.publish_skipped_closed_count() == 1

    @pytest.mark.asyncio
    async def test_open_question_publishes_unchanged(self, offloaded_publish: list[Any]) -> None:
        report = self._Report(_question(close_time=datetime.now(UTC) + timedelta(days=1)))
        result = await BinaryReport.publish_report_to_metaculus(report)  # type: ignore[arg-type]
        assert result == "published", "an open question must take the pre-gate path verbatim"
        assert offloaded_publish == [report]
        assert publish_gate.publish_skipped_closed_count() == 0

    @pytest.mark.asyncio
    async def test_the_gate_covers_every_published_report_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All three question types publish through their own report class, so a gate
        wired to only one would leave the other two 405-ing. The offload already pins
        the class tuple; this pins that the GATE rides it, in case the two ever split.
        """
        posted: list[str] = []

        for report_type in publish_hardening._PATCHED_REPORT_TYPES:

            async def recording_publish(self: Any, metaculus_client: Any = None, _name=report_type.__name__) -> None:
                posted.append(_name)

            monkeypatch.setattr(report_type, publish_hardening._PUBLISH_METHOD, recording_publish)
        publish_hardening.apply_report_publish_offload()
        try:
            closed = self._Report(_question(close_time=datetime(2020, 1, 1, tzinfo=UTC)))
            for report_type in publish_hardening._PATCHED_REPORT_TYPES:
                await report_type.publish_report_to_metaculus(closed)  # type: ignore[arg-type]
            assert posted == [], f"these report types published a closed question: {posted}"
            assert publish_gate.publish_skipped_closed_count() == len(publish_hardening._PATCHED_REPORT_TYPES)
        finally:
            for report_type in publish_hardening._PATCHED_REPORT_TYPES:
                monkeypatch.setattr(
                    report_type,
                    publish_hardening._PUBLISH_METHOD,
                    report_type.__dict__[publish_hardening._PUBLISH_METHOD],
                )

    @pytest.mark.asyncio
    async def test_the_prediction_posts_before_the_comment_on_every_report_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ft-side assumption ``PUBLISH_RESERVE_SECONDS`` is sized on.

        The reserve holds back 60 s for the publish — one POST, deliberately less than
        ``WALL_CLOCK_STACKING_MIN_BUDGET``'s 90 s, which covers both — because only the
        PREDICTION has to beat the close; a comment a few seconds late is still accepted.
        That was established by reading ft's three report bodies, not by a test. Reorder
        them and the reserve gets spent on the comment while the forecast takes the 405
        this whole feature exists to prevent, with nothing going red. Driven against ft's
        unwrapped coroutine so the pin is on ft's body, not on our own layers.
        """
        monkeypatch.setenv("METACULUS_TOKEN", "test-token")
        # ft sleeps 3.5-4.5s before each POST; irrelevant to the ordering invariant.
        monkeypatch.setattr("forecasting_tools.helpers.metaculus_client.time.sleep", lambda *_: None)
        posted_urls: list[str] = []

        def capturing_post(*args: Any, **kwargs: Any) -> Any:
            posted_urls.append(str(args[0] if args else kwargs.get("url", "")))
            response = MagicMock()
            response.status_code = 200
            response.raise_for_status.return_value = None
            response.json.return_value = {}
            return response

        monkeypatch.setattr(_ft_metaculus_client.requests, "post", capturing_post)

        for report in _publishable_reports():
            posted_urls.clear()
            raw = type(report).__dict__[publish_hardening._PUBLISH_METHOD]
            ft_publish = getattr(raw, "__wrapped__", raw)  # strip our offload if installed
            await ft_publish(report)

            assert len(posted_urls) == 2, f"{type(report).__name__}: expected prediction + comment, got {posted_urls}"
            assert posted_urls[0].endswith("/questions/forecast/"), (
                f"{type(report).__name__} no longer POSTs the prediction FIRST: {posted_urls}. "
                "PUBLISH_RESERVE_SECONDS reserves for one POST on the assumption that it is the prediction's"
            )
            assert posted_urls[1].endswith("/comments/create/"), posted_urls

    @pytest.mark.asyncio
    async def test_a_skip_does_not_stop_the_next_question(self, offloaded_publish: list[Any]) -> None:
        # The whole point of skipping rather than raising: q45085 took its run's
        # end-of-run summary down with it. Siblings must be unaffected.
        closed = self._Report(_question(close_time=datetime(2020, 1, 1, tzinfo=UTC), qid=45085))
        still_open = self._Report(_question(close_time=datetime.now(UTC) + timedelta(days=1), qid=45093))
        await BinaryReport.publish_report_to_metaculus(closed)  # type: ignore[arg-type]
        await BinaryReport.publish_report_to_metaculus(still_open)  # type: ignore[arg-type]
        assert offloaded_publish == [still_open]
        assert publish_gate.publish_skipped_closed_count() == 1
