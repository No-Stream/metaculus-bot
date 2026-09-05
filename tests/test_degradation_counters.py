"""Run-level degradation accounting: the alertable sum and the two summary lines.

``alertable_count`` is what ``cli.py`` turns into an exit status, so a dropped or
double-counted term silently changes CI color. The summary lines are what a future
debugger reads first, so their keys have to name what they actually count — and the
telemetry parser reads them, so the text is a contract, not a convenience.
"""

import logging
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import Any

import pytest
from forecasting_tools.data_models.questions import BinaryQuestion

from main import TemplateForecaster
from metaculus_bot import publish_gate, publish_hardening
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.degradation_counters import (
    DegradationSnapshot,
    alertable_total,
    format_conditional_stacking_summary,
    format_degradation_summary,
)
from metaculus_bot.research import prediction_market, provider_health


def _bot(mock_general_llm, *, with_stacker: bool = False, **kwargs: Any) -> TemplateForecaster:
    """A one-forecaster bot. ``with_stacker`` adds the stacker/analyzer slots that
    CONDITIONAL_STACKING requires."""
    llms_config: dict[str, Any] = {
        "forecasters": [mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    if with_stacker:
        llms_config["stacker"] = mock_general_llm
        llms_config["analyzer"] = mock_general_llm
    return TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1, **kwargs)


def _snapshot(**overrides: int) -> DegradationSnapshot:
    values = {
        "forecasters_dropped": 0,
        "questions_failed_to_publish": 0,
        "stacker_primary_failed": 0,
        "stacker_fallback_used": 0,
        "stacker_fallback_failed": 0,
        "research_provider_failures": 0,
        "summarizer_failures": 0,
        "gap_fill_v2_errors": 0,
        "prediction_market_degraded": 0,
        "prediction_market_source_losses": 0,
        "provider_degradation": 0,
        "publish_attempt_failures": 0,
        "publish_skipped_closed": 0,
        "time_budget_fast_path": 0,
        "research_budget_cuts": 0,
        "conditional_stacking_triggered": 0,
        "conditional_stacking_skipped": 0,
        "conditional_stacking_skipped_single_forecaster": 0,
        "conditional_stacking_crux_failures": 0,
        "conditional_stacking_search_failures": 0,
    }
    values.update(overrides)
    return DegradationSnapshot(**values)


def test_snapshot_is_immutable_and_conditional_counts_are_not_alertable() -> None:
    snapshot = _snapshot(
        forecasters_dropped=2,
        conditional_stacking_triggered=3,
        conditional_stacking_skipped=5,
    )

    assert alertable_total(snapshot) == 2
    with pytest.raises(FrozenInstanceError):
        snapshot.forecasters_dropped = 4  # pyright: ignore[reportAttributeAccessIssue]


def test_formatters_accept_snapshot_without_a_forecaster() -> None:
    snapshot = _snapshot(
        forecasters_dropped=1,
        conditional_stacking_triggered=2,
        conditional_stacking_skipped_single_forecaster=3,
    )

    assert format_degradation_summary(snapshot).startswith("Degradation counters: forecasters_dropped=1")
    assert format_conditional_stacking_summary(snapshot) == (
        "Conditional stacking summary: triggered=2, skipped=0, skipped_single_forecaster=3, "
        "crux_failures=0, search_failures=0"
    )


def test_forecaster_reads_a_fresh_snapshot_after_counter_updates(mock_general_llm) -> None:
    bot = _bot(mock_general_llm)

    first_snapshot = bot._degradation_snapshot()
    bot._forecasters_dropped_count = 7
    second_snapshot = bot._degradation_snapshot()

    assert first_snapshot.forecasters_dropped == 0
    assert second_snapshot.forecasters_dropped == 7
    assert alertable_total(first_snapshot) == 0
    assert alertable_total(second_snapshot) == 7


def test_alertable_count_sums_all_degradation_counters(mock_general_llm, monkeypatch):
    """Property must sum all thirteen degradation counters. Using distinct powers of 2
    makes an off-by-one or missing-counter bug visible: the resulting sum
    uniquely identifies which subset was counted.
    """
    bot = _bot(mock_general_llm)

    bot._forecasters_dropped_count = 1
    bot._questions_failed_to_publish = 2
    bot._stacker_primary_failed_count = 4
    bot._stacker_fallback_used_count = 8
    bot._stacker_fallback_failed_count = 16
    bot._research_provider_failure_count = 32
    bot._gap_fill_v2_error_count = 64
    # prediction_market_degraded is read-only — it reads the prediction-market
    # module's per-run global — so stub the accessor the property imports rather
    # than bumping the counter 128 times.
    monkeypatch.setattr(prediction_market, "kalshi_catalogue_fetch_failures", lambda: 128)
    # Same shape for the source-loss counter (operator decision: any prediction-market
    # source losing a fetch reddens CI), so a dropped or double-counted ninth term
    # shows up in the sum.
    monkeypatch.setattr(prediction_market, "prediction_market_source_losses", lambda: 256)
    # Tenth term: a dead AskNews summarizer silently ships raw ungated articles on
    # every question.
    bot._summarizer_failure_count = 512
    # Eleventh term: a provider that populated but degraded — a liquidity field dead
    # across 100% of a venue's rows, or a venue contributing nothing while its
    # siblings answered. Same read-only shape as the two above, so stub the accessor
    # the property imports rather than recording 1024 observations.
    monkeypatch.setattr(provider_health, "provider_degradation_count", lambda: 1024)
    # Twelfth term: a publish POST that exhausted the publish-hardening retry
    # budget (q45085's 405 shape) — the module global the bot property reads.
    monkeypatch.setattr(publish_hardening, "_PUBLISH_ATTEMPT_FAILURES", 2048)
    # Thirteenth term: a question whose close time was too near for the full
    # pipeline, so the optional research stages were dropped (time_budget.py).
    bot._time_budget_fast_path_count = 4096
    # Fourteenth term: budget-driven research degradation OFF the fast path — a
    # provider cancelled at the research-window deadline or gap-fill cut/skipped
    # for budget on a question that never fast-pathed (orchestrator-side,
    # deduplicated per question).
    bot._research.research_budget_cut_count = 8192

    assert bot.alertable_count == 16383


def test_alertable_count_zero_by_default(mock_general_llm):
    """Fresh bot with no degradation events must report alertable_count == 0."""
    assert _bot(mock_general_llm).alertable_count == 0


@pytest.mark.asyncio
async def test_run_summary_lines_name_what_they_count(mock_general_llm, caplog):
    """The two end-of-run summary lines are what a future debugger reads first, so
    their keys have to name what they actually count.

    One run once shipped two lies: ``research_provider_timeouts=1`` described a hard
    403 that failed in 137ms, and ``skipped=0`` contradicted a "single forecaster
    survived" skip logged seven seconds earlier.
    """
    bot = _bot(mock_general_llm, with_stacker=True, aggregation_strategy=AggregationStrategy.CONDITIONAL_STACKING)
    bot._research_provider_failure_count = 3
    bot._summarizer_failure_count = 1
    bot._conditional_stacking_skipped_single_forecaster_count = 2

    with caplog.at_level(logging.INFO, logger="metaculus_bot.forecaster"):
        await bot.forecast_questions([])

    degradation = next(line for line in caplog.messages if line.startswith("Degradation counters:"))
    assert "research_provider_failures=3" in degradation
    assert "research_provider_timeouts" not in degradation
    assert "summarizer_failures=1" in degradation
    assert "prediction_market_source_losses=0" in degradation
    assert "prediction_market_platform_failures" not in degradation

    stacking = next(line for line in caplog.messages if line.startswith("Conditional stacking summary:"))
    assert "skipped_single_forecaster=2" in stacking


@pytest.mark.asyncio
async def test_provider_degradation_rides_the_run_summary(mock_general_llm, caplog):
    """The counter reaches the ``Degradation counters:`` line and the marker fires,
    both per run and both even at zero.

    This is the wiring that makes a degraded provider VISIBLE: the counter is the one
    grep that answers "why was this run red", and the marker is the positive
    counterpart — without it "no provider degraded" and "the check never ran" are the
    same absent line. The tail placement matters too, since the telemetry parser's
    optional-group tail keys on it.
    """
    bot = _bot(mock_general_llm)

    with caplog.at_level(logging.INFO):
        await bot.forecast_questions([])

    degradation = next(line for line in caplog.messages if line.startswith("Degradation counters:"))
    assert "provider_degradation=0" in degradation, degradation
    assert "publish_attempt_failures=0" in degradation, degradation
    assert "publish_skipped_closed=0" in degradation, degradation
    # The newest key is the tail, and the tail is where the telemetry parser's optional
    # groups end — appending past it without extending that regex breaks the whole
    # line's harvest, because the pattern is $-anchored.
    assert "time_budget_fast_path=0" in degradation, degradation
    assert degradation.endswith("research_budget_cuts=0"), degradation
    assert any(line.startswith("PROVIDER_DEGRADATION:") for line in caplog.messages), caplog.messages


@pytest.mark.asyncio
async def test_run_start_resets_provider_health_observations(mock_general_llm):
    """A leaked observation would make a healthy run report a degradation it inherited
    from an earlier run in the same process — and would poison every later
    ``alertable_count == 0`` assertion in the suite. ``forecast_questions`` is where
    the reset has to happen, alongside ``reset_pchip_stats``.
    """
    bot = _bot(mock_general_llm)

    provider_health.record_venue_observation(
        provider_health.VenueObservation(
            qid=1,
            venue="kalshi",
            candidates_pre_filter=3,
            rows_post_filter=3,
            liquidity_fields_present=frozenset(),
        )
    )
    assert bot.alertable_count == 1

    await bot.forecast_questions([])
    assert bot.alertable_count == 0


@pytest.mark.asyncio
async def test_forecast_questions_resets_publish_attempt_failures(mock_general_llm):
    """The publish-hardening retry-exhaustion counter is module state (the wrapper
    has no handle back to the bot), so forecast_questions must zero it at run start
    — same rationale as the prediction-market counters below."""
    bot = _bot(mock_general_llm)

    publish_hardening._bump_publish_attempt_failure()
    try:
        assert publish_hardening.publish_attempt_failures() == 1
        assert bot.alertable_count == 1

        await bot.forecast_questions([])

        assert publish_hardening.publish_attempt_failures() == 0
        assert bot.alertable_count == 0
    finally:
        publish_hardening.reset_publish_attempt_failures()


@pytest.mark.asyncio
async def test_close_time_skip_is_alertable_and_reset_per_run(mock_general_llm, caplog):
    """A publish skipped because the question had already closed means latency cost us
    a question, so it must redden CI rather than pass quietly — and like its
    publish-hardening sibling the counter is module state that forecast_questions has
    to zero, else it leaks into the next run in the same process."""
    bot = _bot(mock_general_llm)

    publish_gate.skip_publish_if_closed(
        BinaryQuestion(
            question_text="Will this have closed by publish time?",
            id_of_question=45085,
            close_time=datetime(2020, 1, 1, tzinfo=UTC),
        )
    )
    try:
        assert publish_gate.publish_skipped_closed_count() == 1
        assert bot.alertable_count == 1

        with caplog.at_level(logging.INFO):
            await bot.forecast_questions([])

        assert publish_gate.publish_skipped_closed_count() == 0
        assert bot.alertable_count == 0
        degradation = next(line for line in caplog.messages if line.startswith("Degradation counters:"))
        assert "publish_skipped_closed=0" in degradation, degradation
    finally:
        publish_gate.reset_publish_skipped_closed()


@pytest.mark.asyncio
async def test_forecast_questions_resets_prediction_market_counter(mock_general_llm):
    """Both prediction-market failure counters (Kalshi series index, source losses)
    live at module scope because the provider is a stateless callable, so
    forecast_questions must zero them at run start.

    Without the reset a previous run's — or a previous test's — failures leak into
    this run's alertable_count, reddening CI for degradation that already happened
    elsewhere.
    """
    bot = _bot(mock_general_llm)

    prediction_market._bump_kalshi_catalogue_failure()
    prediction_market._bump_source_loss()
    assert prediction_market.kalshi_catalogue_fetch_failures() == 1
    assert prediction_market.prediction_market_source_losses() == 1
    # Live module bumps (not stubbed accessors) reach alertable_count...
    assert bot.alertable_count == 2
    try:
        await bot.forecast_questions([])

        assert prediction_market.kalshi_catalogue_fetch_failures() == 0
        assert prediction_market.prediction_market_source_losses() == 0
        assert bot.alertable_count == 0
    finally:
        prediction_market.reset_series_degradation_counter()
        prediction_market.reset_source_loss_counter()
