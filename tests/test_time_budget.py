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
import logging
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from forecasting_tools import BinaryQuestion, GeneralLlm, ReasonedPrediction
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import (
    CRUX_SOFT_DEADLINE,
    NATIVE_SEARCH_WALL_TIMEOUT,
    PER_QUESTION_WALL_CLOCK_DEADLINE,
    PUBLISH_RESERVE_SECONDS,
    RESEARCH_PHASE_BUDGET_SHARE,
    TIME_BUDGET_FAST_PATH_THRESHOLD,
    TIME_BUDGET_MIN_VIABLE_S,
    WALL_CLOCK_STACKING_MIN_BUDGET,
)
from metaculus_bot.publish_gate import reset_publish_skipped_closed
from metaculus_bot.research.orchestrator import ResearchOrchestrator
from metaculus_bot.research.provider_diagnostics import pop_provider_detail, record_provider_detail
from metaculus_bot.research.provider_fanout import await_providers_within_deadline
from metaculus_bot.stacking_route import (
    _enrichment_timeout,
    _skip_stacking_for_budget,
    _stacking_budget_required_s,
    _targeted_research_for_crux,
)
from metaculus_bot.time_budget import (
    QuestionTimeBudget,
    build_question_time_budget,
    format_time_budget_marker,
)
from tests.conftest import gather_predictions_stub
from tests.pipeline_test_helpers import make_e2e_bot, make_real_binary_question


@pytest.fixture(autouse=True)
def _isolate_publish_gate_counter():
    """The pipeline tests here drive the REAL publish gate, whose skip counter is a module
    global (prod resets it at run start). Without an after-each reset, a skip recorded in
    this file leaks into any later-collected suite's fresh-bot ``alertable_count == 0``
    assertion — observed as an order-dependent failure in test_degradation_counters."""
    reset_publish_skipped_closed()
    yield
    reset_publish_skipped_closed()


def _question(close_in: timedelta | None) -> BinaryQuestion:
    close_time = datetime.now(UTC) + close_in if close_in is not None else None
    return make_real_binary_question(qid=7001, close_time=close_time)


def _budget(total_s: float, *, close_limited: bool = True) -> QuestionTimeBudget:
    """A budget positioned as if just granted, for the pure-math helpers."""
    return QuestionTimeBudget(
        total_s=total_s,
        started_at=time.monotonic(),
        close_time=datetime.now(UTC) + timedelta(seconds=total_s),
        close_limited=close_limited,
    )


def _partly_spent_budget(total_s: float, *, spent_s: float, close_limited: bool = True) -> QuestionTimeBudget:
    """A budget granted ``spent_s`` ago, for the mid-question states.

    ``fast_path`` reads ``total_s`` while every stage deadline reads
    ``remaining_s()``, so the two only come apart once time has passed — which is
    the only way to reach the branches that cut a stage on a budget that never
    fast-pathed.
    """
    return QuestionTimeBudget(
        total_s=total_s,
        started_at=time.monotonic() - spent_s,
        close_time=datetime.now(UTC) + timedelta(seconds=total_s - spent_s),
        close_limited=close_limited,
    )


class TestBudgetMath:
    def test_not_publishing_keeps_the_static_budget_and_ignores_close_time(self):
        """A backtest forecasts RESOLVED questions, whose close time is in the past.

        Deriving a budget from that would hand every backtest question a negative
        budget and skip the whole run, so ``close_aware=False`` must not consult the
        field at all — including on a question that closed years ago.
        """
        question = make_real_binary_question(qid=7002, close_time=datetime(2020, 1, 1, tzinfo=UTC))

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
        now = datetime(2026, 8, 3, 11, 40, tzinfo=UTC)
        question = make_real_binary_question(qid=7003, close_time=datetime(2026, 8, 3, 12, 0, tzinfo=UTC))

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
        now = datetime(2026, 8, 3, 11, 30, tzinfo=UTC)
        question = make_real_binary_question(qid=7004, close_time=datetime(2026, 8, 3, 12, 0))

        budget = build_question_time_budget(
            question, close_aware=True, static_deadline_s=PER_QUESTION_WALL_CLOCK_DEADLINE, now=now
        )

        assert budget.total_s == pytest.approx(30 * 60 - PUBLISH_RESERVE_SECONDS)

    def test_q45085_shape_is_exhausted_at_intake(self):
        """22 seconds of headroom: the prediction POST alone cannot fit, so no forecast
        this question could produce would be accepted. That is the 2026-08-03 q45085
        shape, which spent a full 3-forecaster ensemble and then took a 405."""
        now = datetime(2026, 8, 3, 11, 59, 38, tzinfo=UTC)
        question = make_real_binary_question(qid=45085, close_time=datetime(2026, 8, 3, 12, 0, tzinfo=UTC))

        budget = build_question_time_budget(
            question, close_aware=True, static_deadline_s=PER_QUESTION_WALL_CLOCK_DEADLINE, now=now
        )

        assert budget.is_exhausted is True
        assert budget.close_limited is True

    def test_a_close_already_passed_is_exhausted(self):
        now = datetime(2026, 8, 3, 12, 5, tzinfo=UTC)
        question = make_real_binary_question(qid=7005, close_time=datetime(2026, 8, 3, 12, 0, tzinfo=UTC))

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

    def test_min_viable_floor_boundary(self):
        """A close-limited budget below the minimum-viable floor intake-skips: the
        primary-research-plus-one-forecaster path essentially never lands under it,
        so running would spend a full fan-out only for the min-forecasters guard to
        drop the question afterwards."""
        assert _budget(TIME_BUDGET_MIN_VIABLE_S - 1).is_exhausted is True
        assert _budget(TIME_BUDGET_MIN_VIABLE_S).is_exhausted is False

    def test_min_viable_floor_is_close_derived_only(self):
        # A deliberately tiny STATIC budget (tests, an operator override of the
        # static deadline) is a wall-clock experiment, not a hopeless close.
        assert _budget(TIME_BUDGET_MIN_VIABLE_S - 1, close_limited=False).is_exhausted is False
        # The arithmetic-unpublishable shape still skips on every path.
        assert _budget(0.0, close_limited=False).is_exhausted is True

    def test_research_phase_gets_its_share_at_grant_time(self):
        budget = _budget(1200)

        assert budget.research_phase_deadline_s() == pytest.approx(1200 * RESEARCH_PHASE_BUDGET_SHARE, abs=1.0)

    def test_research_phase_is_one_fixed_window_not_a_rolling_share(self):
        """The discriminating case: research consults the deadline at two sequential
        points (provider phase, then gap-fill), and a rolling 50%-of-remaining at each
        compounds to ~75% of the budget — leaving the fan-out under its own soft
        deadline on exactly the close-limited band. The window is fixed at grant:
        whatever research already spent comes out of ITS half."""
        budget = _partly_spent_budget(2000, spent_s=500)

        # Fixed window: 0.5*2000 - 500 = 500. A rolling share would read 0.5*1500 = 750.
        assert budget.research_phase_deadline_s() == pytest.approx(500, abs=1.0)

    def test_a_spent_research_window_reads_zero_with_budget_still_left(self):
        """Past the window, research gets nothing MORE even though the question still
        has budget — that remainder is the forecast's guaranteed half."""
        budget = _partly_spent_budget(2000, spent_s=1100)

        assert budget.remaining_s() > 0
        assert budget.research_phase_deadline_s() == 0.0

    def test_research_phase_deadline_never_goes_negative(self):
        """An overrun budget must report 0 (cancel now), not a negative timeout that
        ``asyncio.wait`` would reject."""
        assert _budget(-500).research_phase_deadline_s() == 0.0

    def test_elapsed_and_remaining_track_each_other(self):
        budget = _budget(100)

        assert budget.elapsed_s() >= 0.0
        assert budget.remaining_s() == pytest.approx(100 - budget.elapsed_s(), abs=0.5)

    def test_marker_line_states_the_budget_and_both_decisions(self):
        now = datetime(2026, 8, 3, 11, 40, tzinfo=UTC)
        question = make_real_binary_question(qid=7006, close_time=datetime(2026, 8, 3, 12, 0, tzinfo=UTC))
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
    """The fast path drops the two SLOW search providers and keeps everything else.

    Dropping the cheap hard-capped providers (resolution_source 45s,
    prediction_market 150s, ts_anchor 20s, financial classifier 30s) cannot
    shorten the phase — the primary itself is the longest configured pole — and
    would only discard the resolution ground truth. The shed is the measured
    tail: native_search (slowest provider on 51.5% of questions, up to 292s) and
    gemini_search.
    """

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

    def test_fast_path_selection_drops_the_slow_search_providers_and_keeps_the_cheap_ones(self):
        names = [name for _, name in self._orchestrator()._select_research_providers(fast_path=True)]

        assert names[0] == "asknews"
        assert "native_search" not in names
        assert "gemini_search" not in names
        assert "resolution_source" in names
        assert "prediction_market" in names
        assert "timeseries_anchor" in names
        assert "financial_data" in names

    def test_fast_path_with_no_primary_still_runs_the_cheap_providers(self, monkeypatch):
        """An unconfigured primary must not take the cheap hard-capped providers down
        with it on the fast path — they are independent and short."""
        for key in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "EXA_API_KEY", "PERPLEXITY_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(key, raising=False)

        names = [name for _, name in self._orchestrator()._select_research_providers(fast_path=True)]

        assert "native_search" not in names
        assert "gemini_search" not in names
        assert "resolution_source" in names

    @pytest.mark.asyncio
    async def test_run_research_hands_selection_the_budgets_fast_path(self):
        """The seam the isolation tests above cannot see: ``run_research`` must pass
        the BUDGET's fast_path into provider selection. A mutation hardcoding it off
        (re-running the slow search providers on exactly the thin questions whose
        problem is tail latency) previously left the whole suite green."""
        orchestrator = self._orchestrator()
        handed_downstream: list[list[str]] = []

        async def record(question, providers, time_budget=None):
            handed_downstream.append([name for _, name in providers])
            return "research " * 300, [], None

        with patch.object(orchestrator, "_run_providers_parallel", side_effect=record):
            await orchestrator.run_research(
                make_real_binary_question(qid=7301), time_budget=_budget(TIME_BUDGET_FAST_PATH_THRESHOLD - 1)
            )
            await orchestrator.run_research(
                make_real_binary_question(qid=7302), time_budget=_budget(TIME_BUDGET_FAST_PATH_THRESHOLD + 200)
            )

        fast_names, roomy_names = handed_downstream
        assert "native_search" not in fast_names
        assert "gemini_search" not in fast_names
        assert "native_search" in roomy_names
        assert "gemini_search" in roomy_names

    @pytest.mark.asyncio
    async def test_fast_path_with_nothing_configured_falls_back_to_the_empty_stub(self, monkeypatch):
        """With no primary AND every optional flag off, selection degrades to the same
        empty stub full selection uses rather than returning an empty list."""
        for key in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "EXA_API_KEY", "PERPLEXITY_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        for flag in (
            "NATIVE_SEARCH_ENABLED",
            "GEMINI_SEARCH_ENABLED",
            "FINANCIAL_DATA_ENABLED",
            "TS_ANCHOR_ENABLED",
            "PREDICTION_MARKETS_ENABLED",
            "RESOLUTION_SOURCE_ENABLED",
        ):
            monkeypatch.setenv(flag, "false")

        providers = self._orchestrator()._select_research_providers(fast_path=True)

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
    async def test_a_cancelled_provider_drains_its_detail_registry_entry(self):
        """CancelledError is a BaseException and used to escape BOTH of _run_one's
        drain paths, leaving the exact stale same-key registry entry the error
        drain's own comment says must not happen."""
        orchestrator = ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())
        question = make_real_binary_question(qid=7103)

        async def slow_with_detail(_question) -> str:
            record_provider_detail(7103, "gemini_search", {"partial": "detail"})
            await asyncio.sleep(30)
            return "never"

        await orchestrator._run_providers_parallel(
            question,
            [(slow_with_detail, "gemini_search")],
            time_budget=_budget(0.1 / RESEARCH_PHASE_BUDGET_SHARE),
        )

        assert pop_provider_detail(7103, "gemini_search") == {}

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

        async def exploding_run_one(_provider, _name):
            raise ValueError("wrapper bug")

        with pytest.raises(ValueError, match="wrapper bug"):
            await await_providers_within_deadline([(AsyncMock(), "native_search")], exploding_run_one, None)

    @pytest.mark.asyncio
    async def test_an_empty_provider_list_yields_no_results_instead_of_raising(self):
        """The bare ``asyncio.gather()`` this replaced accepted an empty list;
        ``asyncio.wait(set())`` raises ValueError. Selection always yields at least the
        "none" stub, so only a direct caller reaches the guard — but without it that
        caller gets an exception where it used to get ``[]``."""
        never_run = AsyncMock(side_effect=AssertionError("no provider should have been started"))

        assert await await_providers_within_deadline([], never_run, None) == []
        never_run.assert_not_awaited()


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


class TestGapFillCutByTheResearchPhaseBudget:
    """The belt to the fast path's braces: a pass that STARTED is still cut.

    The fast-path skip decides before the phase begins, off ``total_s``. These cover
    what happens when a budget that never fast-pathed runs low mid-question — a slow
    intake, a straggling provider — which is the only way the ``asyncio.wait_for``
    around each pass and the zero-remaining skip can fire.
    """

    @pytest.fixture(autouse=True)
    def _both_passes_enabled(self, monkeypatch):
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.setenv("GAP_FILL_V2_ENABLED", "true")

    @staticmethod
    async def _instant_provider(_question) -> str:
        await asyncio.sleep(0)  # one checkpoint, like a real provider's first await
        return "x" * 2000  # comfortably over GAP_FILL_MIN_RESEARCH_CHARS, so v1 activates

    @staticmethod
    async def _never_returns(*_args, **_kwargs) -> str:
        await asyncio.sleep(30)
        return "too late to matter"

    @pytest.mark.asyncio
    async def test_both_passes_are_cut_and_neither_counts_as_a_failure(self, caplog):
        """v2's cut must not bump ``gap_fill_v2_error_count``: that counter exists to
        redden CI on a DEAD v2 feature, and a budget cut is our own decision — alertable
        instead through ``research_budget_cut_count``. Folding the two together would
        make a run of thin-window questions look like v2 had broken."""
        orchestrator = ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())
        # Clears the fast-path threshold, so the pre-phase skip cannot be what fires,
        # but only ~2s of the fixed research WINDOW (total * share, anchored at grant)
        # is left — the wait_for lands mid-pass. The margin is deliberately seconds
        # rather than milliseconds: this suite runs on shared CI runners, and a window
        # that expires before the phase starts would take the SKIP branch and quietly
        # stop testing the cut.
        total_s = TIME_BUDGET_FAST_PATH_THRESHOLD + 200
        budget = _partly_spent_budget(total_s, spent_s=total_s * RESEARCH_PHASE_BUDGET_SHARE - 2)
        assert budget.fast_path is False
        assert 0.0 < budget.research_phase_deadline_s() < 5.0

        with (
            patch.object(
                orchestrator, "_select_research_providers", return_value=[(self._instant_provider, "native_search")]
            ),
            patch("metaculus_bot.research.targeted.run_gap_fill_pass", new=self._never_returns),
            patch("metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2", new=self._never_returns),
        ):
            research = await orchestrator.run_research(make_real_binary_question(qid=7203), time_budget=budget)

        assert any("GAP_FILL_V1_CUT_FOR_BUDGET" in message for message in caplog.messages), caplog.messages
        assert any("GAP_FILL_V2_CUT_FOR_BUDGET" in message for message in caplog.messages), caplog.messages
        assert not any("GAP_FILL_SKIPPED_FOR_BUDGET" in message for message in caplog.messages), caplog.messages
        # The bundle survives the cut with the primary provider's text intact.
        assert "x" * 2000 in research
        assert "Targeted Gap-Fill" not in research
        assert "Agentic Research Findings" not in research
        assert orchestrator.gap_fill_v2_error_count == 0
        # The cut IS alertable, once: both passes cut on one question dedupe to a
        # single research_budget_cut_count bump — the off-fast-path counter that
        # keeps this degradation out of the all-clear census.
        assert orchestrator.research_budget_cut_count == 1

    @pytest.mark.asyncio
    async def test_an_exhausted_research_budget_skips_the_passes_outside_the_fast_path(self, caplog):
        """The second disjunct of the skip: ``research_phase_deadline_s()`` reached 0 on
        a budget whose ``total_s`` never qualified as thin, so the marker has to say
        ``fast_path=false`` — otherwise the archive attributes the degradation to a
        close time when the real cause was time already burned."""
        orchestrator = ResearchOrchestrator(default_llm=MagicMock(), summarizer_llm=MagicMock())
        budget = _partly_spent_budget(
            PER_QUESTION_WALL_CLOCK_DEADLINE, spent_s=PER_QUESTION_WALL_CLOCK_DEADLINE + 10, close_limited=False
        )
        assert budget.fast_path is False
        assert budget.research_phase_deadline_s() == 0.0

        with (
            patch.object(
                orchestrator, "_select_research_providers", return_value=[(self._instant_provider, "native_search")]
            ),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass", new_callable=AsyncMock, return_value="v1 addendum"
            ) as v1,
            patch(
                "metaculus_bot.research.agentic_gap_fill.run_gap_fill_v2", new_callable=AsyncMock, return_value="v2"
            ) as v2,
        ):
            await orchestrator.run_research(make_real_binary_question(qid=7204), time_budget=budget)

        v1.assert_not_called()
        v2.assert_not_called()
        skip_lines = [message for message in caplog.messages if "GAP_FILL_SKIPPED_FOR_BUDGET" in message]
        assert len(skip_lines) == 1, caplog.messages
        assert "fast_path=false" in skip_lines[0]
        assert "research_phase_remaining=0s" in skip_lines[0]


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
        question.close_time = datetime.now(UTC) + close_in if close_in is not None else None
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
            self._bot(AggregationStrategy.CONDITIONAL_STACKING)._pipeline,
            make_real_binary_question(qid=7301),
            7301,
            _budget(120, close_limited=False),
        )

        assert skipped is False

    def test_a_close_limited_budget_with_two_minutes_left_skips_stacking(self, caplog):
        """Same 120 s, but now overrunning forfeits the question — and the stacker
        ladder alone can legitimately run 800 s."""
        bot = self._bot(AggregationStrategy.CONDITIONAL_STACKING)
        with caplog.at_level(logging.WARNING):
            skipped = _skip_stacking_for_budget(bot._pipeline, make_real_binary_question(qid=7302), 7302, _budget(120))

        assert skipped is True
        assert bot._pipeline.outcomes[7302] == "fallback_median"
        # The skip records the same reason + counter treatment as its siblings, so a
        # STACKER_SKIP_REASON cut cannot silently miss the budget bucket.
        assert bot._pipeline.skip_reasons[7302] == "wall_clock_budget"
        assert bot._pipeline.counters.conditional_stacking_skipped_count == 1
        # The WARN interpolates the floor ACTUALLY applied — the close-limited one
        # including the stacking path's worst case — not the static 90 s constant,
        # which would make the line arithmetically false (remaining=120 > 90).
        abort_line = next(m for m in caplog.messages if "WALLCLOCK_ABORT" in m)
        expected_floor = WALL_CLOCK_STACKING_MIN_BUDGET + _stacking_budget_required_s(bot._pipeline)
        assert f"< {expected_floor:.0f}s" in abort_line

    def test_a_close_limited_budget_with_room_for_the_whole_ladder_still_stacks(self):
        skipped = _skip_stacking_for_budget(
            self._bot(AggregationStrategy.CONDITIONAL_STACKING)._pipeline,
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

    @pytest.mark.asyncio
    async def test_both_enrichment_stages_actually_run_under_the_clamp(self):
        """The clamp has to be WIRED, not merely correct: dropping either call site
        would leave a stage free to run its full soft deadline (crux 180 s, targeted
        search 420 s) on a budget with two minutes left — and the unit test above would
        still pass. Recorded off ``asyncio.wait_for`` so the assertion is on the exact
        timeout each stage was given rather than on how long the test slept."""
        bot = self._bot(AggregationStrategy.CONDITIONAL_STACKING)
        analyzer_llm = bot._analyzer_llm
        assert analyzer_llm is not None, "the CONDITIONAL_STACKING bot factory configures one"
        budget = _budget(150)
        predictions: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.2, reasoning="Model: m1\n\nlow"),
            ReasonedPrediction(prediction_value=0.6, reasoning="Model: m2\n\nhigh"),
        ]
        recorded: list[float | None] = []
        real_wait_for = asyncio.wait_for

        async def recording_wait_for(awaitable, **kwargs):
            recorded.append(kwargs.get("timeout"))
            return await real_wait_for(awaitable, **kwargs)

        with (
            patch("asyncio.wait_for", recording_wait_for),
            patch(
                "metaculus_bot.stacking_route.extract_disagreement_crux",
                new_callable=AsyncMock,
                return_value="whether the BLS revision lands before the close",
            ),
            patch(
                "metaculus_bot.stacking_route.run_targeted_search",
                new_callable=AsyncMock,
                return_value="targeted findings",
            ),
        ):
            text = await _targeted_research_for_crux(
                bot._pipeline,
                make_real_binary_question(qid=7304),
                analyzer_llm=analyzer_llm,
                is_benchmarking=False,
                valid_predictions=predictions,
                time_budget=budget,
            )

        assert text == "targeted findings"
        expected = [
            _enrichment_timeout(CRUX_SOFT_DEADLINE, budget),
            _enrichment_timeout(NATIVE_SEARCH_WALL_TIMEOUT, budget),
        ]
        assert recorded == pytest.approx(expected, abs=1.0)
        # No None among them: an unbounded stage is the failure this guards against, and
        # both came out well below their own soft deadlines because 150 s of budget
        # cannot afford either at full length.
        bounded = [timeout for timeout in recorded if timeout is not None]
        assert len(bounded) == 2
        assert max(bounded) < CRUX_SOFT_DEADLINE

    @pytest.mark.asyncio
    async def test_a_clamped_crux_that_overruns_degrades_to_base_research(self, caplog):
        """What the clamp does when it bites. The stage is enrichment, so a cut must cost
        the question nothing beyond the targeted search — and the WARN has to name the
        CLAMPED bound rather than ``CRUX_SOFT_DEADLINE``, or a reader debugging a cut
        stage is handed a number (180 s) that never applied."""
        bot = self._bot(AggregationStrategy.CONDITIONAL_STACKING)
        analyzer_llm = bot._analyzer_llm
        assert analyzer_llm is not None
        predictions: list[ReasonedPrediction[PredictionTypes]] = [
            ReasonedPrediction(prediction_value=0.2, reasoning="Model: m1\n\nlow"),
            ReasonedPrediction(prediction_value=0.6, reasoning="Model: m2\n\nhigh"),
        ]
        # Fully spent, so the clamp lands on its 1 s floor — the shortest run this can
        # have while still giving the coroutine a first step.
        budget = _budget(0)
        assert _enrichment_timeout(CRUX_SOFT_DEADLINE, budget) == 1.0
        search = AsyncMock(return_value="never reached")

        with (
            patch("metaculus_bot.stacking_route.extract_disagreement_crux", new=self._hangs),
            patch("metaculus_bot.stacking_route.run_targeted_search", new=search),
            caplog.at_level(logging.WARNING, logger="metaculus_bot.stacking_route"),
        ):
            text = await _targeted_research_for_crux(
                bot._pipeline,
                make_real_binary_question(qid=7305),
                analyzer_llm=analyzer_llm,
                is_benchmarking=False,
                valid_predictions=predictions,
                time_budget=budget,
            )

        assert text == ""
        search.assert_not_awaited()
        assert bot._pipeline.counters.conditional_stacking_crux_failures == 1
        cut = [message for message in caplog.messages if message.startswith("CRUX_SOFT_DEADLINE:")]
        assert len(cut) == 1, caplog.messages
        assert "exceeded 1s" in cut[0], cut[0]
        assert str(CRUX_SOFT_DEADLINE) not in cut[0], "the constant is not the bound that applied"

    @staticmethod
    async def _hangs(*_args, **_kwargs) -> str:
        await asyncio.sleep(30)
        return "too late to matter"


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
        question = make_real_binary_question(qid=7401, close_time=datetime.now(UTC) + timedelta(minutes=20))
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
        question = make_real_binary_question(qid=7402, close_time=datetime.now(UTC) + timedelta(hours=3))
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
        question = make_real_binary_question(qid=7403, close_time=datetime.now(UTC) + timedelta(seconds=5))
        research = AsyncMock(return_value="Canned research")
        gather = AsyncMock()

        with (
            patch.object(bot, "_get_notepad") as notepad,
            patch.object(bot, "run_research", new=research),
            patch.object(bot, "_gather_predictions_with_wall_clock", new=gather),
        ):
            notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            with pytest.raises(RuntimeError, match="no viable time budget"):
                await bot._research_and_make_predictions(question)

        research.assert_not_awaited()
        gather.assert_not_awaited()
        # The intake skip shares the CLOSE gate's counter ("latency cost us this
        # question" has one home); questions_failed_to_publish stays the
        # min-forecasters floor's counter alone.
        assert bot._publish_skipped_closed_count == 1
        assert bot._questions_failed_to_publish == 0
        assert bot._time_budget_fast_path_count == 0

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_a_backtest_over_a_long_closed_question_is_not_skipped(self):
        """The wiring that keeps this feature out of the backtests. ``close_aware`` is
        ``publish_reports_to_metaculus``, and backtests forecast RESOLVED questions —
        hardcode it True and every one of them gets a negative budget and raises at
        intake, killing the whole run. The builder's unit test cannot see that; only the
        bot's own gate can."""
        bot = make_e2e_bot(
            AggregationStrategy.CONDITIONAL_STACKING,
            n_forecasters=3,
            publish_reports_to_metaculus=False,
            is_benchmarking=True,
        )
        question = make_real_binary_question(qid=7407, close_time=datetime(2024, 6, 1, tzinfo=UTC))
        predictions = [
            ReasonedPrediction(prediction_value=0.30, reasoning="Model: m1\n\nbacktest"),
            ReasonedPrediction(prediction_value=0.32, reasoning="Model: m2\n\nbacktest"),
            ReasonedPrediction(prediction_value=0.31, reasoning="Model: m3\n\nbacktest"),
        ]
        seen_budgets = []

        async def capture_research(_question, time_budget=None):
            await asyncio.sleep(0)
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
        assert bot._questions_failed_to_publish == 0
        assert bot._time_budget_fast_path_count == 0
        (budget,) = seen_budgets
        assert budget is not None
        assert budget.total_s == PER_QUESTION_WALL_CLOCK_DEADLINE
        # A past close must not even be recorded, or the marker line would imply it bound
        # something on a run that never publishes.
        assert budget.close_time is None
        assert budget.close_limited is False

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_the_forecaster_fanout_is_cut_by_the_close_derived_deadline(self, monkeypatch):
        """The seam the whole feature rests on: the fan-out's ``asyncio.wait`` cap comes
        from the budget, so a close-limited question cancels its stragglers seconds in
        rather than at the 3510 s static deadline. Nothing here monkeypatches
        ``PER_QUESTION_WALL_CLOCK_DEADLINE`` — with the static budget in charge the slow
        forecasters would run to completion and the outer ``wait_for`` would fail the
        test rather than let it hang. The min-viable intake floor is zeroed so the
        sub-second budget reaches the fan-out at all (the floor has its own tests)."""
        monkeypatch.setattr("metaculus_bot.time_budget.TIME_BUDGET_MIN_VIABLE_S", 0)
        bot = make_e2e_bot(
            AggregationStrategy.MEAN,
            n_forecasters=3,
            publish_reports_to_metaculus=True,
            is_benchmarking=False,
            min_forecasters_to_publish=1,
        )
        # Just over the publish reserve, so the budget is a fraction of a second.
        question = make_real_binary_question(
            qid=7404, close_time=datetime.now(UTC) + timedelta(seconds=PUBLISH_RESERVE_SECONDS + 0.6)
        )
        started: list[int] = []

        async def one_fast_two_hung(*_args, **_kwargs) -> ReasonedPrediction[PredictionTypes]:
            started.append(1)
            await asyncio.sleep(30 if len(started) > 1 else 0)
            return ReasonedPrediction(prediction_value=0.31, reasoning="Model: m1\n\nbeat the close")

        bot._forecaster_with_soft_deadline = one_fast_two_hung  # type: ignore[method-assign]

        with (
            patch.object(bot, "_get_notepad") as notepad,
            patch.object(bot, "run_research", new=AsyncMock(return_value="Canned research")),
        ):
            notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            result = await asyncio.wait_for(bot._research_and_make_predictions(question), timeout=15)

        assert len(result.predictions) == 1, "the one forecaster that beat the close still publishes"
        assert bot._forecasters_dropped_count == 2
        assert bot._time_budget_fast_path_count == 1

    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_every_question_logs_its_budget_marker_including_roomy_ones(self, caplog):
        """``TIME_BUDGET`` is the uncensored denominator a later round needs.
        ``CLOSE_MARGIN`` fires only after a successful submission, so it is censored on
        exactly the thin-window questions this feature exists for — which is why the
        marker has to be emitted for a roomy question too, not only when it bites."""
        bot = self._bot()
        predictions = [
            ReasonedPrediction(prediction_value=0.30, reasoning="Model: m1\n\nresearch"),
            ReasonedPrediction(prediction_value=0.32, reasoning="Model: m2\n\nresearch"),
            ReasonedPrediction(prediction_value=0.31, reasoning="Model: m3\n\nresearch"),
        ]
        roomy = make_real_binary_question(qid=7405, close_time=datetime.now(UTC) + timedelta(days=30))
        thin = make_real_binary_question(qid=7406, close_time=datetime.now(UTC) + timedelta(minutes=20))

        with (
            patch.object(bot, "_get_notepad") as notepad,
            patch.object(bot, "run_research", new=AsyncMock(return_value="Canned research")),
            patch.object(
                bot, "_gather_predictions_with_wall_clock", new=gather_predictions_stub((predictions, [], None))
            ),
            caplog.at_level(logging.INFO, logger="metaculus_bot.forecaster"),
        ):
            notepad.return_value = Mock(total_research_reports_attempted=0, total_predictions_attempted=0)
            await bot._research_and_make_predictions(roomy)
            await bot._research_and_make_predictions(thin)

        markers = [message for message in caplog.messages if message.startswith("TIME_BUDGET:")]
        assert len(markers) == 2, caplog.messages
        assert f"question=7405 budget_s={PER_QUESTION_WALL_CLOCK_DEADLINE} " in markers[0]
        assert "close_limited=false fast_path=false" in markers[0]
        assert "question=7406 " in markers[1]
        assert "close_limited=true fast_path=true" in markers[1]
