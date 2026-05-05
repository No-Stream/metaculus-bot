import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import BinaryQuestion, GeneralLlm, MetaculusQuestion, ReasonedPrediction
from forecasting_tools.data_models.forecast_report import ResearchWithPredictions

from main import TemplateForecaster
from metaculus_bot.comment_trimming import TRIM_NOTICE
from metaculus_bot.constants import REPORT_SECTION_CHAR_LIMIT
from metaculus_bot.discrete_snap import OutcomeTypeResult

# `asyncio` is used by the soft-deadline test below; an explicit no-op reference
# prevents the formatter from pruning the import when the only usage sits far
# below the import block (formatter heuristic).
_ASYNCIO_SLEEP = asyncio.sleep


@pytest.fixture
def mock_general_llm():
    mock_llm = MagicMock(spec=GeneralLlm)
    mock_llm.model = "mock_model"
    mock_llm.invoke = AsyncMock(return_value="mock reasoning")
    return mock_llm


@pytest.fixture
def mock_metaculus_question():
    question = MagicMock(spec=MetaculusQuestion)
    question.page_url = "http://example.com/question"
    question.question_text = "Test Question"
    question.background_info = "Background info"
    question.resolution_criteria = "Resolution criteria"
    question.fine_print = "Fine print"
    question.unit_of_measure = "units"
    question.id_of_question = 123  # Add a mock ID for testing
    return question


@pytest.fixture
def mock_binary_question():
    question = MagicMock(spec=BinaryQuestion)
    question.page_url = "http://example.com/binary_question"
    question.question_text = "Binary Test Question"
    question.background_info = "Binary background info"
    question.resolution_criteria = "Binary resolution criteria"
    question.fine_print = "Binary fine print"
    question.unit_of_measure = "binary units"
    question.id_of_question = 456
    return question


@pytest.mark.asyncio
async def test_template_forecaster_init_with_forecasters(mock_general_llm):
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config)

    assert bot._forecaster_llms == llms_config["forecasters"]
    assert bot.predictions_per_research_report == 2
    assert bot.get_llm("default") == mock_general_llm  # Should be the first forecaster


@pytest.mark.asyncio
async def test_template_forecaster_init_without_forecasters():
    llms_config = {
        "default": GeneralLlm(model="test_default"),
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config, predictions_per_research_report=3)

    assert not bot._forecaster_llms
    assert bot.predictions_per_research_report == 3
    assert bot.get_llm("default").model == "test_default"


@pytest.mark.asyncio
async def test_template_forecaster_init_no_llms_provided():
    with pytest.raises(ValueError, match="Either 'forecasters' or a 'default' LLM must be provided."):
        TemplateForecaster(llms=None)


@pytest.mark.asyncio
async def test_template_forecaster_init_missing_required_llms():
    # Test missing parser and researcher
    incomplete_llms = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
    }
    with pytest.raises(ValueError, match="Missing required LLM purposes: parser, researcher"):
        TemplateForecaster(llms=incomplete_llms)

    # Test missing just researcher
    incomplete_llms = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
    }
    with pytest.raises(ValueError, match="Missing required LLM purposes: researcher"):
        TemplateForecaster(llms=incomplete_llms)


@pytest.mark.asyncio
async def test_template_forecaster_init_forecasters_not_list():
    llms_config = {
        "forecasters": "not_a_list",
        "default": GeneralLlm(model="test_default"),
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    with patch("metaculus_bot.llm_setup.logger.warning") as mock_warning:
        bot = TemplateForecaster(llms=llms_config)
        mock_warning.assert_called_once_with("'forecasters' key in llms must be a list of GeneralLlm objects.")
        assert not bot._forecaster_llms
        assert bot.predictions_per_research_report == 1  # Default value from parent class


@pytest.mark.asyncio
async def test_research_and_make_predictions_with_forecasters(mock_binary_question, mock_general_llm):
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1)

    # Mock internal methods
    bot._get_notepad = AsyncMock(
        return_value=MagicMock(total_research_reports_attempted=0, total_predictions_attempted=0)
    )
    bot.run_research = AsyncMock(return_value="mock research")
    bot.summarize_research = AsyncMock(return_value="mock summary")
    bot._make_prediction = AsyncMock(return_value=ReasonedPrediction(prediction_value=0.5, reasoning="test"))
    bot._gather_results_and_exceptions = AsyncMock(
        return_value=(
            [
                ReasonedPrediction(prediction_value=0.5, reasoning="test"),
                ReasonedPrediction(prediction_value=0.6, reasoning="test2"),
            ],
            [],
            None,
        )
    )

    # Wrap _forecaster_with_soft_deadline so we can count invocations. The
    # soft-deadline wrapper is the new per-forecaster entrypoint — it delegates
    # to _make_prediction internally. Tests mocking _gather_results_and_exceptions
    # short-circuit execution, so we assert on the wrapper being called (once
    # per forecaster) rather than the inner _make_prediction.
    bot._forecaster_with_soft_deadline = AsyncMock(
        return_value=ReasonedPrediction(prediction_value=0.5, reasoning="test")
    )

    result = await bot._research_and_make_predictions(mock_binary_question)

    bot._get_notepad.assert_called_once_with(mock_binary_question)
    bot.run_research.assert_called_once_with(mock_binary_question)
    # summarize_research is NOT called when use_research_summary_to_forecast=False (default)
    bot.summarize_research.assert_not_called()
    assert bot._forecaster_with_soft_deadline.call_count == 2  # Called once for each forecaster
    bot._forecaster_with_soft_deadline.assert_any_call(
        mock_binary_question, "mock research", mock_general_llm, mock_binary_question.id_of_question
    )
    assert isinstance(result, ResearchWithPredictions)
    assert (
        len(result.predictions) == 2
    )  # The mocked _gather_results_and_exceptions returns two ReasonedPrediction objects


@pytest.mark.asyncio
async def test_research_and_make_predictions_with_summarization_enabled(mock_binary_question, mock_general_llm):
    """Test that summarize_research IS called when use_research_summary_to_forecast=True"""
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, use_research_summary_to_forecast=True, min_forecasters_to_publish=1)

    # Mock internal methods
    bot._get_notepad = AsyncMock(
        return_value=MagicMock(total_research_reports_attempted=0, total_predictions_attempted=0)
    )
    bot.run_research = AsyncMock(return_value="mock research")
    bot.summarize_research = AsyncMock(return_value="mock summary")
    bot._make_prediction = AsyncMock(return_value=ReasonedPrediction(prediction_value=0.5, reasoning="test"))
    bot._gather_results_and_exceptions = AsyncMock(
        return_value=(
            [
                ReasonedPrediction(prediction_value=0.5, reasoning="test"),
                ReasonedPrediction(prediction_value=0.6, reasoning="test2"),
            ],
            [],
            None,
        )
    )

    # See sibling test above — soft-deadline wrapper is the per-forecaster
    # entrypoint, so assert on it rather than the inner _make_prediction.
    bot._forecaster_with_soft_deadline = AsyncMock(
        return_value=ReasonedPrediction(prediction_value=0.5, reasoning="test")
    )

    result = await bot._research_and_make_predictions(mock_binary_question)

    bot._get_notepad.assert_called_once_with(mock_binary_question)
    bot.run_research.assert_called_once_with(mock_binary_question)
    # summarize_research IS called when use_research_summary_to_forecast=True
    bot.summarize_research.assert_called_once_with(mock_binary_question, "mock research")
    assert bot._forecaster_with_soft_deadline.call_count == 2  # Called once for each forecaster
    bot._forecaster_with_soft_deadline.assert_any_call(
        mock_binary_question, "mock summary", mock_general_llm, mock_binary_question.id_of_question
    )  # Uses summary
    assert isinstance(result, ResearchWithPredictions)
    assert len(result.predictions) == 2


@pytest.mark.asyncio
async def test_research_and_make_predictions_without_forecasters(mock_binary_question):
    llms_config = {
        "default": GeneralLlm(model="test_default"),
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config, predictions_per_research_report=1)

    # Mock the super method call
    with patch(
        "forecasting_tools.forecast_bots.forecast_bot.ForecastBot._research_and_make_predictions",
        new_callable=AsyncMock,
    ) as mock_super_method:
        mock_super_method.return_value = ResearchWithPredictions(
            research_report="super research",
            summary_report="super summary",
            predictions=[ReasonedPrediction(prediction_value=0.6, reasoning="super test")],
        )
        result = await bot._research_and_make_predictions(mock_binary_question)
        mock_super_method.assert_called_once_with(mock_binary_question)
        assert isinstance(result, ResearchWithPredictions)
        assert result.research_report == "super research"


@pytest.mark.asyncio
async def test_make_prediction_with_provided_llm(mock_binary_question, mock_general_llm):
    llms_config = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config)
    bot._get_notepad = AsyncMock(return_value=MagicMock(total_predictions_attempted=0))
    bot._run_forecast_on_binary = AsyncMock(
        return_value=ReasonedPrediction(prediction_value=0.7, reasoning="binary forecast")
    )

    result = await bot._make_prediction(mock_binary_question, "some research", mock_general_llm)

    bot._get_notepad.assert_called_once_with(mock_binary_question)
    bot._run_forecast_on_binary.assert_called_once_with(mock_binary_question, "some research", mock_general_llm)
    assert result.prediction_value == 0.7
    assert "Model: mock_model" in result.reasoning
    assert "binary forecast" in result.reasoning


@pytest.mark.asyncio
async def test_make_prediction_without_provided_llm(mock_binary_question):
    mock_default_llm = MagicMock(spec=GeneralLlm)
    mock_default_llm.model = "default_mock_model"
    mock_default_llm.invoke = AsyncMock(return_value="default reasoning")

    llms_config = {
        "default": mock_default_llm,
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config)
    bot._get_notepad = AsyncMock(return_value=MagicMock(total_predictions_attempted=0))
    bot._run_forecast_on_binary = AsyncMock(
        return_value=ReasonedPrediction(prediction_value=0.8, reasoning="default binary forecast")
    )
    bot.get_llm = MagicMock(return_value=mock_default_llm)

    result = await bot._make_prediction(mock_binary_question, "some research")

    bot._get_notepad.assert_called_once_with(mock_binary_question)
    bot.get_llm.assert_called_once_with("default", "llm")
    bot._run_forecast_on_binary.assert_called_once_with(mock_binary_question, "some research", mock_default_llm)
    assert result.prediction_value == 0.8
    assert "Model: default_mock_model" in result.reasoning
    assert "default binary forecast" in result.reasoning


@pytest.mark.asyncio
async def test_run_forecast_on_binary_uses_provided_llm(mock_binary_question, mock_general_llm):
    llms_config = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config)

    # Mock structured_output to avoid external parsing LLM calls
    with patch(
        "main.structure_output",
        return_value=type("_Bin", (), {"prediction_in_decimal": 0.65})(),
    ) as mock_struct:
        result = await bot._run_forecast_on_binary(mock_binary_question, "some research", mock_general_llm)
        mock_general_llm.invoke.assert_called_once()
        mock_struct.assert_called_once()
        assert result.prediction_value == 0.65
        assert "mock reasoning" in result.reasoning


@pytest.mark.asyncio
async def test_run_forecast_on_multiple_choice_uses_provided_llm(mock_metaculus_question, mock_general_llm):
    llms_config = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config)
    mock_metaculus_question.options = ["A", "B"]

    # Mock structured_output for multiple-choice
    with patch("main.structure_output", return_value=MagicMock()) as mock_struct:
        result = await bot._run_forecast_on_multiple_choice(mock_metaculus_question, "some research", mock_general_llm)
        mock_general_llm.invoke.assert_called_once()
        mock_struct.assert_called_once()
        assert result.prediction_value is not None
        assert "mock reasoning" in result.reasoning


@pytest.mark.asyncio
async def test_run_forecast_on_numeric_uses_provided_llm(mock_metaculus_question, mock_general_llm):
    llms_config = {
        "default": "mock_default_model",
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config)

    # Mock bound_messages and structured_output to return a valid percentile list
    from forecasting_tools.data_models.numeric_report import Percentile as FTPercentile

    fake_percentiles = [
        FTPercentile(value=v, percentile=p)
        for v, p in zip(
            [0.5, 1, 2, 4, 5, 6, 7, 8, 9, 9.5, 9.75],
            [0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 0.95, 0.975],
        )
    ]
    # Provide minimal numeric bounds attributes expected by NumericDistribution.from_question
    mock_metaculus_question.open_upper_bound = False
    mock_metaculus_question.open_lower_bound = False
    mock_metaculus_question.upper_bound = 100
    mock_metaculus_question.lower_bound = 0
    mock_metaculus_question.zero_point = None
    mock_metaculus_question.cdf_size = 201

    with (
        patch("main.bound_messages", return_value=("", "")) as mock_bounds,
        patch(
            "main.structure_output",
            side_effect=[OutcomeTypeResult(is_discrete_integer=False), fake_percentiles],
        ) as mock_struct,
    ):
        result = await bot._run_forecast_on_numeric(mock_metaculus_question, "some research", mock_general_llm)
        mock_general_llm.invoke.assert_called_once()
        mock_bounds.assert_called_once()
        assert mock_struct.call_count == 2  # outcome type classification + percentiles
        assert result.prediction_value is not None
        assert "mock reasoning" in result.reasoning


def test_format_methods_trim_long_outputs():
    llms_config = {
        "default": GeneralLlm(model="test_default"),
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
    }
    bot = TemplateForecaster(llms=llms_config)

    long_research_body = "Line\n" + "A" * (REPORT_SECTION_CHAR_LIMIT + 500)
    long_reasoning = "Reasoning\n" + "B" * (REPORT_SECTION_CHAR_LIMIT + 800)
    research_with_predictions = ResearchWithPredictions(
        research_report=f"# Deep Dive\n{long_research_body}",
        summary_report="Summary",
        predictions=[ReasonedPrediction(prediction_value=0.5, reasoning=long_reasoning)],
    )

    formatted_research = bot._format_main_research(1, research_with_predictions)
    assert formatted_research.startswith("## Report 1 Research")
    assert TRIM_NOTICE in formatted_research
    assert len(formatted_research) <= REPORT_SECTION_CHAR_LIMIT

    formatted_rationales = bot._format_forecaster_rationales(1, research_with_predictions)
    assert formatted_rationales.startswith("## R1: Forecaster 1 Reasoning")
    assert TRIM_NOTICE in formatted_rationales
    assert len(formatted_rationales) <= REPORT_SECTION_CHAR_LIMIT


# ---------------------------------------------------------------------------
# F4: real _forecaster_with_soft_deadline timeout branch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_forecaster_with_soft_deadline_times_out_and_bumps_counter(
    mock_binary_question, mock_general_llm, monkeypatch: pytest.MonkeyPatch
):
    """The real wrapper must raise TimeoutError and bump _forecasters_dropped_count
    when _make_prediction exceeds FORECASTER_SOFT_DEADLINE. Prior tests replaced
    the wrapper with an AsyncMock, so the asyncio.wait_for branch was untested.
    """
    llms_config = {
        "forecasters": [mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1)

    # Tighten the deadline to a fraction of a second so the test is fast.
    monkeypatch.setattr("main.FORECASTER_SOFT_DEADLINE", 0.05)

    async def slow_make_prediction(question, research, llm):
        await asyncio.sleep(5)
        return ReasonedPrediction(prediction_value=0.5, reasoning="never returned")

    bot._make_prediction = AsyncMock(side_effect=slow_make_prediction)

    assert bot._forecasters_dropped_count == 0
    with pytest.raises(asyncio.TimeoutError):
        await bot._forecaster_with_soft_deadline(
            mock_binary_question, "research", mock_general_llm, mock_binary_question.id_of_question
        )
    assert bot._forecasters_dropped_count == 1


# ---------------------------------------------------------------------------
# F5: min-forecasters guard raise path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_min_forecasters_guard_raises_runtime_error_when_exception_group_none(
    mock_binary_question, mock_general_llm
):
    """If only 1/2 forecasters succeed and threshold is 3, with no exception_group,
    the guard raises RuntimeError and bumps _questions_failed_to_publish.
    """
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    # Warning expected on construction (threshold exceeds ensemble size); this is
    # the exact scenario we want to test.
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=3)

    bot._get_notepad = AsyncMock(
        return_value=MagicMock(total_research_reports_attempted=0, total_predictions_attempted=0)
    )
    bot.run_research = AsyncMock(return_value="mock research")
    bot._forecaster_with_soft_deadline = AsyncMock(
        return_value=ReasonedPrediction(prediction_value=0.5, reasoning="ok")
    )
    # 1 valid, no errors, no exception group: RuntimeError path.
    bot._gather_results_and_exceptions = AsyncMock(
        return_value=(
            [ReasonedPrediction(prediction_value=0.5, reasoning="ok")],
            [],
            None,
        )
    )

    assert bot._questions_failed_to_publish == 0
    with pytest.raises(RuntimeError, match="Only 1/2 forecasters succeeded"):
        await bot._research_and_make_predictions(mock_binary_question)
    assert bot._questions_failed_to_publish == 1


@pytest.mark.asyncio
async def test_min_forecasters_guard_reraises_exception_group_when_present(mock_binary_question, mock_general_llm):
    """If only 1/2 forecasters succeed and exception_group is non-None, the
    re-raise path preserves the exception chain by delegating to the framework
    helper (which raises an ExceptionGroup).
    """
    llms_config = {
        "forecasters": [mock_general_llm, mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=3)

    bot._get_notepad = AsyncMock(
        return_value=MagicMock(total_research_reports_attempted=0, total_predictions_attempted=0)
    )
    bot.run_research = AsyncMock(return_value="mock research")
    bot._forecaster_with_soft_deadline = AsyncMock(
        return_value=ReasonedPrediction(prediction_value=0.5, reasoning="ok")
    )

    inner = RuntimeError("forecaster 2 failed")
    # ExceptionGroup is a Python 3.11+ builtin; ruff target-version isn't pinned
    # here so suppress the false-positive F821.
    exc_group = ExceptionGroup("forecaster errors", [inner])  # noqa: F821
    bot._gather_results_and_exceptions = AsyncMock(
        return_value=(
            [ReasonedPrediction(prediction_value=0.5, reasoning="ok")],
            ["RuntimeError: forecaster 2 failed"],
            exc_group,
        )
    )

    with pytest.raises(ExceptionGroup) as exc_info:  # noqa: F821  # 3.11+ builtin
        await bot._research_and_make_predictions(mock_binary_question)

    # The framework helper wraps the exception group with a prepended message
    # but preserves the original wrapped exceptions.
    assert any(isinstance(e, RuntimeError) and "forecaster 2 failed" in str(e) for e in exc_info.value.exceptions)
    assert bot._questions_failed_to_publish == 1


# ---------------------------------------------------------------------------
# F9a: alertable_count sum
# ---------------------------------------------------------------------------


def test_alertable_count_sums_all_degradation_counters(mock_general_llm):
    """Property must sum all six degradation counters. Using distinct powers of 2
    makes an off-by-one or missing-counter bug visible: the resulting sum
    uniquely identifies which subset was counted.
    """
    llms_config = {
        "forecasters": [mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1)

    bot._forecasters_dropped_count = 1
    bot._questions_failed_to_publish = 2
    bot._stacker_primary_failed_count = 4
    bot._stacker_fallback_used_count = 8
    bot._stacker_fallback_failed_count = 16
    bot._research_provider_timeout_count = 32

    assert bot.alertable_count == 63


def test_alertable_count_zero_by_default(mock_general_llm):
    """Fresh bot with no degradation events must report alertable_count == 0."""
    llms_config = {
        "forecasters": [mock_general_llm],
        "summarizer": "mock_summarizer_model",
        "parser": "mock_parser_model",
        "researcher": "mock_researcher_model",
        "default": "mock_default_model",
    }
    bot = TemplateForecaster(llms=llms_config, min_forecasters_to_publish=1)

    assert bot.alertable_count == 0
