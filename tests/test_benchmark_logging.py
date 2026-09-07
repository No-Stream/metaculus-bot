from types import SimpleNamespace
from typing import cast

from forecasting_tools import ForecastBot, GeneralLlm

from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.benchmark.logging import log_bot_lineup, log_stacking_summaries
from metaculus_bot.forecaster import TemplateForecaster


def _stacking_bot() -> TemplateForecaster:
    llm = GeneralLlm(model="test-model", temperature=0.0)
    return TemplateForecaster(
        aggregation_strategy=AggregationStrategy.STACKING,
        llms={
            "forecasters": [llm, llm],  # type: ignore[dict-item]  # supported bot extension to framework config
            "stacker": llm,
            "default": llm,
            "parser": llm,
            "researcher": llm,
            "summarizer": llm,
        },
        is_benchmarking=True,
    )


def test_real_stacking_bot_logging_reads_pipeline_configuration_and_counters(caplog) -> None:
    bot = _stacking_bot()
    bot.name = "stack-test"
    bot._pipeline.counters.stacking_fallback_count = 2
    bot._pipeline.counters.stacking_expected_combine_count = 3
    bot._pipeline.counters.stacking_unexpected_combine_count = 1

    caplog.clear()
    with caplog.at_level("INFO", logger="metaculus_bot.benchmark.logging"):
        log_bot_lineup([bot])
        log_stacking_summaries([bot])

    messages = [record.getMessage() for record in caplog.records]
    assert (
        "- Bot 1/1 | name=stack-test | strategy=STACKING | R×P=1×2 | stacker=test-model | "
        "base_forecasters(2)=['test-model', 'test-model'] | final_outputs_per_q=1"
    ) in messages
    assert "STACKING fallback summary | bot=stack-test | fallbacks=2 (fell back to MEAN due to errors)" in messages
    assert (
        "Stacking combine summary | bot=stack-test | expected=3 | unexpected=1 "
        "(base-aggregator combine across research reports)"
    ) in messages


def test_generic_forecast_bot_logging_keeps_framework_defaults(caplog) -> None:
    generic_bot = cast(
        ForecastBot,
        SimpleNamespace(
            name="generic",
            aggregation_strategy=None,
            research_reports_per_question=2,
            predictions_per_research_report=4,
        ),
    )

    with caplog.at_level("INFO", logger="metaculus_bot.benchmark.logging"):
        log_bot_lineup([generic_bot])
        log_stacking_summaries([generic_bot])

    messages = [record.getMessage() for record in caplog.records]
    assert "- Bot 1/1 | name=generic | strategy=(framework default) | R×P=2×4" in messages
    assert not any(
        message.startswith(("STACKING fallback summary", "Stacking combine summary")) for message in messages
    )
