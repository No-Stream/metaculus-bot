"""Tests for the close-aware per-question time budget (metaculus_bot/time_budget.py).

The invariant under test, per the operator mandate: a question with workable headroom
at run start always gets its PREDICTION submitted before close. Before this feature
the pipeline's only budget was the constant ``PER_QUESTION_WALL_CLOCK_DEADLINE``
(3510 s), sized against the CRON PERIOD rather than the question's deadline, and
``close_time`` was first consulted at the publish gate — after every paid call. So a
question closing in 20 minutes was handed 58.5 minutes of budget, and the full
pipeline's configured worst case is ~30 minutes.

Four behaviors are pinned here:

1. Budget math — ``min(static, close - now - publish reserve)``, gated on publishing
   so backtests over resolved questions keep the static budget.
2. Fast-path selection — below the threshold, the optional research stages are
   dropped, counted, and alertable.
3. The research phase is bounded, so a straggling provider cannot spend the
   forecast's time.
4. Ordering — tightest close first, which is also what the max-questions cap keeps.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from forecasting_tools import BinaryQuestion, GeneralLlm, ReasonedPrediction
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import (
    CRUX_SOFT_DEADLINE,
    PER_QUESTION_WALL_CLOCK_DEADLINE,
    PUBLISH_RESERVE_SECONDS,
    RESEARCH_PHASE_BUDGET_SHARE,
    TIME_BUDGET_FAST_PATH_THRESHOLD,
    WALL_CLOCK_STACKING_MIN_BUDGET,
)
from metaculus_bot.research.orchestrator import ResearchOrchestrator
from metaculus_bot.stacking_route import _enrichment_timeout, _skip_stacking_for_budget
from metaculus_bot.time_budget import (
    QuestionTimeBudget,
    build_question_time_budget,
    format_time_budget_marker,
)
from tests.conftest import gather_predictions_stub
from tests.pipeline_test_helpers import make_e2e_bot, make_real_binary_question


def _question(close_in: timedelta | None) -> BinaryQuestion:
    close_time = datetime.now(timezone.utc) + close_in if close_in is not None else None
    return make_real_binary_question(qid=7001, close_time=close_time)


def _budget(total_s: float, *, close_limited: bool = True) -> QuestionTimeBudget:
    """A budget positioned as if just granted, for the pure-math helpers."""
    return QuestionTimeBudget(
        total_s=total_s,
        started_at=time.monotonic(),
        close_time=datetime.now(timezone.utc) + timedelta(seconds=total_s),
        close_limited=close_limited,
    )


class TestBudgetMath:
    def test_not_publishing_keeps_the_static_budget_and_ignores_close_time(self):
        """A backtest forecasts RESOLVED questions, whose close time is in the past.

        Deriving a budget from that would hand every backtest question a negative
        budget and skip the whole run, so ``close_aware=False`` must not consult the
        field at all — including on a question that closed years ago.
        """
        question = make_real_binary_question(qid=7002, close_time=datetime(2020, 1, 1, tzinfo=timezone.utc))

        budget = build_question_time_budget(
            question, close_aware=False, static_deadline_s=PER_QUESTION_WALL_CLOCK_DEADLINE
        )

        assert budget.total_s == PER_QUESTION_WALL_CLOCK_DEADLINE
        assert budget.close_limited is False
        assert budget.is_exhausted is False
        assert budget.fast_path is False

    def test_a_distant_close_leaves_the_static_deadline_in_charge(self):
        budget = build_question_time_budget(
            _question(timedelta(days=365)), close_aware=True, static_deadline_s=PER_QUESTION_WALL_CLOCK_DEADLINE
        )

        assert budget.total_s == PER_QUESTION_WALL_CLOCK_DEADLINE
        assert budget.close_limited is False

    def test_a_missing_close_time_leaves_the_static_deadline_in_charge(self):
        budget = build_question_time_budget(
            _question(None), close_aware=True, static_deadline_s=PER_QUESTION_WALL_CLOCK_DEADLINE
        )

        assert budget.total_s == PER_QUESTION_WALL_CLOCK_DEADLINE
        assert budget.close_limited is False
        assert budget.close_time is None

    def test_a_near_close_shrinks_the_budget_by_the_publish_reserve(self):
        now = datetime(2026, 8, 3, 11, 40, tzinfo=timezone.utc)
        question = make_real_binary_question(qid=7003, close_time=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc))

        budget = build_question_time_budget(
            question, close_aware=True, static_deadline_s=PER_QUESTION_WALL_CLOCK_DEADLINE, now=now
        )

        assert budget.total_s == pytest.approx(20 * 60 - PUBLISH_RESERVE_SECONDS)
        assert budget.close_limited is True
        assert budget.fast_path is True
        assert budget.is_exhausted is False

    def test_a_naive_close_time_is_read_as_utc(self):
        """ft parses API timestamps tz-aware, but some call sites still hand in naive.

        A naive close time must not raise on the subtraction against an aware now, and
        must not silently shift the deadline by the local offset.
        """
        now = datetime(2026, 8, 3, 11, 30, tzinfo=timezone.utc)
        question = make_real_binary_question(qid=7004, close_time=datetime(2026, 8, 3, 12, 0))

        budget = build_question_time_budget(
            question, close_aware=True, static_deadline_s=PER_QUESTION_WALL_CLOCK_DEADLINE, now=now
        )

        assert budget.total_s == pytest.approx(30 * 60 - PUBLISH_RESERVE_SECONDS)

    def test_q45085_shape_is_exhausted_at_intake(self):
        """22 seconds of headroom: the prediction POST alone cannot fit, so no forecast
        this question could produce would be accepted. That is the 2026-08-03 q45085
        shape, which spent a full 3-forecaster ensemble and then took a 405."""
        now = datetime(2026, 8, 3, 11, 59, 38, tzinfo=timezone.utc)
        question = make_real_binary_question(qid=45085, close_time=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc))

        budget = build_question_time_budget(
            question, close_aware=True, static_deadline_s=PER_QUESTION_WALL_CLOCK_DEADLINE, now=now
        )

        assert budget.is_exhausted is True
        assert budget.close_limited is True

    def test_a_close_already_passed_is_exhausted(self):
        now = datetime(2026, 8, 3, 12, 5, tzinfo=timezone.utc)
        question = make_real_binary_question(qid=7005, close_time=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc))

        budget = build_question_time_budget(
            question, close_aware=True, static_deadline_s=PER_QUESTION_WALL_CLOCK_DEADLINE, now=now
        )

        assert budget.is_exhausted is True
        assert budget.total_s < 0

    def test_the_fast_path_threshold_sits_above_the_pipelines_worst_case(self):
        """Sanity on the constant itself: the static budget must never fast-path (or
        every ordinary question would run degraded), and a budget one second under the
        threshold must."""
        assert PER_QUESTION_WALL_CLOCK_DEADLINE > TIME_BUDGET_FAST_PATH_THRESHOLD
        assert _budget(TIME_BUDGET_FAST_PATH_THRESHOLD).fast_path is False
        assert _budget(TIME_BUDGET_FAST_PATH_THRESHOLD - 1).fast_path is True

    def test_research_phase_gets_a_share_of_what_remains(self):
        budget = _budget(1200)

        # Measured from `remaining`, which is ~total at grant time.
        assert budget.research_phase_deadline_s() == pytest.approx(1200 * RESEARCH_PHASE_BUDGET_SHARE, abs=1.0)

    def test_research_phase_deadline_never_goes_negative(self):
        """An overrun budget must report 0 (cancel now), not a negative timeout that
        ``asyncio.wait`` would reject."""
        assert _budget(-500).research_phase_deadline_s() == 0.0

    def test_elapsed_and_remaining_track_each_other(self):
        budget = _budget(100)

        assert budget.elapsed_s() >= 0.0
        assert budget.remaining_s() == pytest.approx(100 - budget.elapsed_s(), abs=0.5)

    def test_marker_line_states_the_budget_and_both_decisions(self):
        now = datetime(2026, 8, 3, 11, 40, tzinfo=timezone.utc)
        question = make_real_binary_question(qid=7006, close_time=datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc))
        budget = build_question_time_budget(
            question, close_aware=True, static_deadline_s=PER_QUESTION_WALL_CLOCK_DEADLINE, now=now
        )

        marker = format_time_budget_marker(question, budget)

        assert marker == (
            "TIME_BUDGET: question=7006 budget_s=1140 close_time=2026-08-03T12:00:00+00:00 "
            "close_limited=true fast_path=true"
        )

    def test_marker_line_renders_a_missing_close_time_as_the_none_sentinel(self):
        question = _question(None)
        budget = build_question_time_budget(
            question, close_aware=True, static_deadline_s=PER_QUESTION_WALL_CLOCK_DEADLINE
        )

        assert "close_time=n/a" in format_time_budget_marker(question, budget)


class TestFastPathProviderSelection:
    """The fast path keeps the primary provider and drops every optional one."""

    def _orchestrator(self) -> ResearchOrchestrator:
        return ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())

    @pytest.fixture(autouse=True)
    def _all_providers_enabled(self, monkeypatch):
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")
        for flag in (
            "NATIVE_SEARCH_ENABLED",
            "GEMINI_SEARCH_ENABLED",
            "FINANCIAL_DATA_ENABLED",
            "TS_ANCHOR_ENABLED",
            "PREDICTION_MARKETS_ENABLED",
            "RESOLUTION_SOURCE_ENABLED",
        ):
            monkeypatch.setenv(flag, "true")
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setenv("FRED_API_KEY", "key")

    def test_full_selection_includes_the_optional_providers(self):
        names = [name for _, name in self._orchestrator()._select_research_providers()]

        assert names[0] == "asknews"
        assert "native_search" in names
        assert "prediction_market" in names

    def test_fast_path_selection_keeps_only_the_primary(self):
        names = [name for _, name in self._orchestrator()._select_research_providers(primary_only=True)]

        assert names == ["asknews"]

    @pytest.mark.asyncio
    async def test_fast_path_with_no_primary_falls_back_to_the_empty_stub(self, monkeypatch):
        """The fast path must not accidentally re-enable the optional providers when the
        primary is unconfigured — it degrades to the same empty stub full selection uses.
        """
        for key in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "EXA_API_KEY", "PERPLEXITY_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        providers = self._orchestrator()._select_research_providers(primary_only=True)

        assert [name for _, name in providers] == ["none"]
        assert await providers[0][0](MagicMock()) == ""


class TestResearchPhaseDeadline:
    """A straggling provider is cancelled, not allowed to eat the forecast's time."""

    @pytest.mark.asyncio
    async def test_a_straggler_is_cancelled_and_the_partial_bundle_survives(self):
        orchestrator = ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())
        question = make_real_binary_question(qid=7101)
        cancelled = asyncio.Event()

        async def fast(_question) -> str:
            return "fast provider output"

        async def slow(_question) -> str:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return "never"

        # A budget whose research share is ~0.05s, so the slow provider is guaranteed
        # to still be running when the phase deadline lands.
        budget = _budget(0.1 / RESEARCH_PHASE_BUDGET_SHARE)
        combined, results, _ = await orchestrator._run_providers_parallel(
            question,
            [(fast, "native_search"), (slow, "gemini_search")],
            time_budget=budget,
        )

        assert cancelled.is_set()
        assert "fast provider output" in combined
        by_name = {result.name: result for result in results}
        assert by_name["native_search"].status == "ok"
        assert by_name["gemini_search"].status == "deadline"
        # A cancelled provider is a budget decision, not a provider defect: counting it
        # as a failure would make research_provider_failures stop meaning "broke".
        assert orchestrator.provider_failure_count == 0

    @pytest.mark.asyncio
    async def test_without_a_budget_the_phase_is_unbounded(self):
        """Every caller outside the per-question pipeline passes no budget, and must
        keep the pre-feature behavior of waiting for all providers."""
        orchestrator = ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())

        async def slowish(_question) -> str:
            await asyncio.sleep(0.05)
            return "eventually arrived"

        combined, results, _ = await orchestrator._run_providers_parallel(
            make_real_binary_question(qid=7102), [(slowish, "native_search")], time_budget=None
        )

        assert "eventually arrived" in combined
        assert results[0].status == "ok"

    @pytest.mark.asyncio
    async def test_a_broken_wrapper_still_raises_rather_than_faking_a_result(self):
        """``_run_one`` converts every PROVIDER exception into a ProviderResult, so an
        exception escaping it is a bug in our own wrapper. Swallowing it into a
        synthesized result would hide that class of defect entirely."""
        orchestrator = ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())

        async def exploding_run_one(_provider, _name):
            raise ValueError("wrapper bug")

        with pytest.raises(ValueError, match="wrapper bug"):
            await orchestrator._await_providers_within_deadline(
                [(AsyncMock(), "native_search")], exploding_run_one, None
            )


class TestGapFillSkippedOnTheFastPath:
    @pytest.mark.asyncio
    async def test_fast_path_skips_both_gap_fill_passes(self, monkeypatch, caplog):
        """Gap-fill is the research phase's largest optional cost (v1's configured worst
        case is 555 s), so it is the first thing a thin window drops."""
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        orchestrator = ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())

        async def provider(_question) -> str:
            return "x" * 2000  # comfortably over GAP_FILL_MIN_RESEARCH_CHARS

        with (
            patch.object(orchestrator, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass", new_callable=AsyncMock, return_value="v1 addendum"
            ) as v1,
            patch(
                "metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2",
                new_callable=AsyncMock,
                return_value="v2 findings",
            ) as v2,
        ):
            research = await orchestrator.run_research(
                make_real_binary_question(qid=7201), time_budget=_budget(TIME_BUDGET_FAST_PATH_THRESHOLD - 1)
            )

        v1.assert_not_called()
        v2.assert_not_called()
        assert "Targeted Gap-Fill" not in research
        assert "Agentic Research Findings" not in research
        assert any("GAP_FILL_SKIPPED_FOR_BUDGET" in message for message in caplog.messages)

    @pytest.mark.asyncio
    async def test_a_roomy_budget_still_runs_both_passes(self, monkeypatch):
        """The guarantee that nothing changes on an ordinary question: the static
        budget's research share (~1755 s) exceeds the phase's 1155 s worst case."""
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")
        orchestrator = ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())

        async def provider(_question) -> str:
            return "x" * 2000

        with (
            patch.object(orchestrator, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass", new_callable=AsyncMock, return_value="v1 addendum"
            ) as v1,
            patch(
                "metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2",
                new_callable=AsyncMock,
                return_value="## Agentic Research Findings\n\nv2 findings",
            ) as v2,
        ):
            research = await orchestrator.run_research(
                make_real_binary_question(qid=7202),
                time_budget=_budget(PER_QUESTION_WALL_CLOCK_DEADLINE, close_limited=False),
            )

        v1.assert_awaited_once()
        v2.assert_awaited_once()
        assert "Targeted Gap-Fill" in research
        assert "Agentic Research Findings" in research


class TestTightestCloseFirstOrdering:
    def _bot(self) -> TemplateForecaster:
        return TemplateForecaster(
            llms={
                "default": "mock",
                "summarizer": "mock_sum",
                "parser": "mock_parser",
                "researcher": "mock_researcher",
            },
            max_questions_per_run=2,
        )

    def _mock_question(self, qid: int, close_in: timedelta | None) -> MagicMock:
        question = MagicMock(spec=BinaryQuestion)
        question.already_forecasted = False
        question.id_of_question = qid
        question.close_time = datetime.now(timezone.utc) + close_in if close_in is not None else None
        return question

    @pytest.mark.asyncio
    async def test_the_cap_keeps_the_soonest_closing_questions(self, monkeypatch):
        """Without the sort the cap kept whatever order the tournament fetch returned,
        so a run with more questions than the cap could drop the one about to close and
        forecast one with hours left."""
        roomy = self._mock_question(1, timedelta(hours=5))
        tight = self._mock_question(2, timedelta(minutes=20))
        middling = self._mock_question(3, timedelta(hours=2))
        forwarded: list[list[int]] = []

        async def stub(_self, questions_arg, return_exceptions=False):
            forwarded.append([q.id_of_question for q in questions_arg])
            return []

        monkeypatch.setattr(ForecastBot, "forecast_questions", stub, raising=True)

        await self._bot().forecast_questions([roomy, tight, middling])

        assert forwarded == [[2, 3]]

    @pytest.mark.asyncio
    async def test_a_question_with_no_close_time_sorts_last(self, monkeypatch):
        """No deadline means no urgency, and it must not raise on a None-vs-datetime
        comparison either."""
        deadline_free = self._mock_question(1, None)
        tight = self._mock_question(2, timedelta(minutes=20))
        forwarded: list[list[int]] = []

        async def stub(_self, questions_arg, return_exceptions=False):
            forwarded.append([q.id_of_question for q in questions_arg])
            return []

        monkeypatch.setattr(ForecastBot, "forecast_questions", stub, raising=True)

        await self._bot().forecast_questions([deadline_free, tight])

        assert forwarded == [[2, 1]]


class TestStackingGateUnderACloseLimitedBudget:
    """The 90 s floor is right against the static deadline and wrong against a close."""

    def _bot(self, strategy: AggregationStrategy) -> TemplateForecaster:
        llm = MagicMock(spec=GeneralLlm)
        llm.model = "mock"
        llm.invoke = AsyncMock(return_value="reasoning")
        return TemplateForecaster(
            llms={
                "forecasters": [llm, llm],
                "stacker": llm,
                "analyzer": llm,
                "default": "mock",
                "summarizer": "mock_sum",
                "parser": "mock_parser",
                "researcher": "mock_researcher",
            },
            aggregation_strategy=strategy,
        )

    def test_a_static_budget_with_two_minutes_left_still_stacks(self):
        """Overrunning the static deadline costs nothing (the close is hours away), so
        this case must keep its pre-feature behavior exactly."""
        skipped = _skip_stacking_for_budget(
            self._bot(AggregationStrategy.CONDITIONAL_STACKING),
            make_real_binary_question(qid=7301),
            7301,
            _budget(120, close_limited=False),
        )

        assert skipped is False

    def test_a_close_limited_budget_with_two_minutes_left_skips_stacking(self):
        """Same 120 s, but now overrunning forfeits the question — and the stacker
        ladder alone can legitimately run 800 s."""
        bot = self._bot(AggregationStrategy.CONDITIONAL_STACKING)
        skipped = _skip_stacking_for_budget(bot, make_real_binary_question(qid=7302), 7302, _budget(120))

        assert skipped is True
        assert bot._stacker_outcome[7302] == "fallback_median"

    def test_a_close_limited_budget_with_room_for_the_whole_ladder_still_stacks(self):
        skipped = _skip_stacking_for_budget(
            self._bot(AggregationStrategy.CONDITIONAL_STACKING),
            make_real_binary_question(qid=7303),
            7303,
            _budget(3000),
        )

        assert skipped is False

    def test_the_enrichment_clamp_leaves_the_publish_reserve_alone(self):
        # Roomy: the stage's own soft deadline wins.
        assert _enrichment_timeout(CRUX_SOFT_DEADLINE, _budget(3000)) == pytest.approx(CRUX_SOFT_DEADLINE)
        # Tight: clamped to remaining minus the publish reserve.
        assert _enrichment_timeout(CRUX_SOFT_DEADLINE, _budget(150)) == pytest.approx(
            150 - WALL_CLOCK_STACKING_MIN_BUDGET, abs=1.0
        )
        # Exhausted: a positive floor, never 0 (wait_for(0) cancels before the first
        # step and would report a stage failure that never happened).
        assert _enrichment_timeout(CRUX_SOFT_DEADLINE, _budget(0)) == 1.0


class TestThinWindowThroughThePipeline:
    """Integration: the whole per-question pipeline under a thin window."""

    def _bot(self, **overrides) -> TemplateForecaster:
        return make_e2e_bot(
            AggregationStrategy.CONDITIONAL_STACKING,
            n_forecasters=3,
            publish_reports_to_metaculus=True,
            is_benchmarking=False,
            **overrides,
        )

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_a_thin_window_publishes_degraded_and_reddens_ci(self):
        bot = self._bot()
        question = make_real_binary_question(qid=7401, close_time=datetime.now(timezone.utc) + timedelta(minutes=20))
        predictions = [
            ReasonedPrediction(prediction_value=0.30, reasoning="Model: m1\n\nthin window"),
            ReasonedPrediction(prediction_value=0.32, reasoning="Model: m2\n\nthin window"),
            ReasonedPrediction(prediction_value=0.31, reasoning="Model: m3\n\nthin window"),
        ]
        seen_budgets = []

        async def capture_research(_question, time_budget=None):
            seen_budgets.append(time_budget)
            return "Canned research"

        with (
            patch.object(bot, "_get_notepad") as notepad,
            patch.object(bot, "run_research", new=capture_research),
            patch.object(
                bot, "_gather_predictions_with_wall_clock", new=gather_predictions_stub((predictions, [], None))
            ),
        ):
            notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            result = await bot._research_and_make_predictions(question)

        # Published: three raw predictions handed to the aggregator.
        assert len(result.predictions) == 3
        # Degraded, counted, and alertable.
        assert bot._time_budget_fast_path_count == 1
        assert bot.alertable_count >= 1
        # The research phase saw a close-limited budget, not the static one.
        (budget,) = seen_budgets
        assert budget is not None
        assert budget.fast_path is True
        assert budget.close_limited is True
        assert budget.total_s < TIME_BUDGET_FAST_PATH_THRESHOLD

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_a_roomy_window_runs_the_full_pipeline(self):
        bot = self._bot()
        question = make_real_binary_question(qid=7402, close_time=datetime.now(timezone.utc) + timedelta(hours=3))
        predictions = [
            ReasonedPrediction(prediction_value=0.30, reasoning="Model: m1\n\nroomy"),
            ReasonedPrediction(prediction_value=0.32, reasoning="Model: m2\n\nroomy"),
            ReasonedPrediction(prediction_value=0.31, reasoning="Model: m3\n\nroomy"),
        ]
        seen_budgets = []

        async def capture_research(_question, time_budget=None):
            seen_budgets.append(time_budget)
            return "Canned research"

        with (
            patch.object(bot, "_get_notepad") as notepad,
            patch.object(bot, "run_research", new=capture_research),
            patch.object(
                bot, "_gather_predictions_with_wall_clock", new=gather_predictions_stub((predictions, [], None))
            ),
        ):
            notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            result = await bot._research_and_make_predictions(question)

        assert len(result.predictions) == 3
        assert bot._time_budget_fast_path_count == 0
        (budget,) = seen_budgets
        assert budget is not None
        assert budget.fast_path is False
        assert budget.total_s == PER_QUESTION_WALL_CLOCK_DEADLINE

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_a_zero_headroom_question_is_skipped_before_any_spend(self):
        """The q45085 class. Today this question would research, run three forecasters,
        and then be refused by the publish gate; the budget check makes it cost nothing
        while producing the same skip and the same alertable counter."""
        bot = self._bot()
        question = make_real_binary_question(qid=7403, close_time=datetime.now(timezone.utc) + timedelta(seconds=5))
        research = AsyncMock(return_value="Canned research")
        gather = AsyncMock()

        with (
            patch.object(bot, "_get_notepad") as notepad,
            patch.object(bot, "run_research", new=research),
            patch.object(bot, "_gather_predictions_with_wall_clock", new=gather),
        ):
            notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            with pytest.raises(RuntimeError, match="no publishable time budget"):
                await bot._research_and_make_predictions(question)

        research.assert_not_awaited()
        gather.assert_not_awaited()
        assert bot._questions_failed_to_publish == 1
        assert bot._time_budget_fast_path_count == 0
