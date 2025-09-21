import logging

import pytest
from forecasting_tools import GeneralLlm

from metaculus_bot import llm_setup
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.llm_setup import prepare_llm_config


def _base_llms() -> dict[str, object]:
    sentinel = GeneralLlm(model="sentinel", temperature=0.0)
    return {
        "default": sentinel,
        "parser": sentinel,
        "researcher": sentinel,
        "summarizer": sentinel,
    }


def test_prepare_llm_config_with_forecasters_updates_predictions():
    llm_a = GeneralLlm(model="a", temperature=0.0)
    llm_b = GeneralLlm(model="b", temperature=0.0)
    base = _base_llms()
    base["forecasters"] = [llm_a, llm_b]

    setup = prepare_llm_config(
        llms=base,
        aggregation_strategy=AggregationStrategy.MEAN,
        predictions_per_report=1,
    )

    assert setup.forecaster_llms == [llm_a, llm_b]
    assert setup.predictions_per_report == 2
    assert "forecasters" not in setup.normalized_llms
    assert setup.normalized_llms["default"] == llm_a


def test_prepare_llm_config_requires_core_roles():
    with pytest.raises(ValueError, match="Missing required LLM purposes"):
        prepare_llm_config(
            llms={"default": GeneralLlm(model="a", temperature=0.0)},
            aggregation_strategy=AggregationStrategy.MEAN,
            predictions_per_report=1,
        )


def test_prepare_llm_config_stacking_sets_default_to_stacker():
    stacker = GeneralLlm(model="stack", temperature=0.0)
    forecaster = GeneralLlm(model="fore", temperature=0.0)
    base = _base_llms()
    base["forecasters"] = [forecaster]
    base["stacker"] = stacker

    setup = prepare_llm_config(
        llms=base,
        aggregation_strategy=AggregationStrategy.STACKING,
        predictions_per_report=3,
    )

    assert setup.normalized_llms["default"] is stacker
    assert setup.stacker_llm is stacker
    assert setup.forecaster_llms == [forecaster]
    assert setup.predictions_per_report == 1


def test_prepare_llm_config_logs_invalid_stacker(monkeypatch, caplog):
    base = _base_llms()
    base["stacker"] = "oops"

    test_logger = logging.getLogger("metaculus_bot.llm_setup_test")
    caplog.set_level(logging.WARNING, logger=test_logger.name)
    monkeypatch.setattr(llm_setup, "logger", test_logger)

    setup = prepare_llm_config(
        llms=base,
        aggregation_strategy=AggregationStrategy.MEAN,
        predictions_per_report=1,
    )

    assert setup.stacker_llm is None
    assert "must be a GeneralLlm" in caplog.text
