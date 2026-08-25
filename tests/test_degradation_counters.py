"""Run-level degradation accounting: the alertable sum and the two summary lines.

``alertable_count`` is what ``cli.py`` turns into an exit status, so a dropped or
double-counted term silently changes CI color. The summary lines are what a future
debugger reads first, so their keys have to name what they actually count — and the
telemetry parser reads them, so the text is a contract, not a convenience.
"""

import logging
from typing import Any

import pytest

from main import TemplateForecaster
from metaculus_bot import publish_hardening
from metaculus_bot.aggregation_strategies import AggregationStrategy
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


def test_alertable_count_sums_all_degradation_counters(mock_general_llm, monkeypatch):
    """Property must sum all twelve degradation counters. Using distinct powers of 2
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

    assert bot.alertable_count == 4095


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
    assert degradation.endswith("publish_attempt_failures=0"), degradation
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
