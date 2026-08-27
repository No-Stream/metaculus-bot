from __future__ import annotations

from abc import ABC
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from forecasting_tools.data_models.forecast_report import ForecastReport
from forecasting_tools.data_models.numeric_report import NumericDistribution, Percentile
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from pydantic import Field

from metaculus_bot.numeric.config import PCHIP_CDF_POINTS
from metaculus_bot.numeric.utils import (
    aggregate_binary_mean,
    aggregate_numeric,
    bound_messages,
    nominal_bounds,
)
from metaculus_bot.prompts import binary_prompt, multiple_choice_prompt, numeric_prompt
from metaculus_bot.utils.logging_utils import compact_log_report_summary

if TYPE_CHECKING:
    from forecasting_tools.helpers.metaculus_client import MetaculusClient


def _open_time() -> datetime:
    return datetime.now() - timedelta(days=30)


def _resolve_time() -> datetime:
    return datetime.now() + timedelta(days=365)


# ---------- Prompt builders -------------------------------------------------


def test_binary_prompt_contains_inputs():
    question = BinaryQuestion(
        id_of_question=2,
        id_of_post=2,
        page_url="example",
        question_text="Will it rain tomorrow?",
        background_info="Some background",
        resolution_criteria="Criteria",
        fine_print="Fine print",
        published_time=None,
        close_time=None,
        open_time=_open_time(),
        scheduled_resolution_time=_resolve_time(),
    )
    prompt = binary_prompt(question, "research snippet")
    assert "Will it rain tomorrow?" in prompt
    assert "research snippet" in prompt


def test_multiple_choice_prompt_contains_options():
    question = MultipleChoiceQuestion(
        id_of_question=3,
        id_of_post=3,
        page_url="example",
        question_text="Who will win?",
        options=["A", "B"],
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        open_time=_open_time(),
        scheduled_resolution_time=_resolve_time(),
    )
    prompt = multiple_choice_prompt(question, "mc research")
    assert "Who will win?" in prompt
    # Post-refactor: the STRUCTURED FORECAST JSON block is the sole per-option
    # forecast surface; real option names appear as JSON keys so strict parsers
    # can bind LLM output to question.options. The old trailing "{opt}: NN%"
    # prose lines are gone.
    assert '"A"' in prompt
    assert '"B"' in prompt
    assert "A: NN%" not in prompt
    assert "B: NN%" not in prompt


def test_numeric_prompt_bounds_and_research():
    question = NumericQuestion(
        id_of_question=4,
        id_of_post=4,
        page_url="example",
        question_text="How many widgets?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        open_time=_open_time(),
        scheduled_resolution_time=_resolve_time(),
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="widgets",
        zero_point=None,
    )
    prompt = numeric_prompt(question, "num research", "lower", "upper")
    assert "widgets" in prompt
    assert "num research" in prompt


def test_numeric_prompt_declared_percentiles_block_has_all_thirteen_keys():
    """Post-refactor: the numeric prompt's ONLY forecast surface is the
    STRUCTURED FORECAST JSON block. It must carry all 13 standard percentile
    keys ("0.01".."0.99") in the declared_percentiles example, in order."""
    question = NumericQuestion(
        id_of_question=5,
        id_of_post=5,
        page_url="example",
        question_text="Test numeric question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        open_time=_open_time(),
        scheduled_resolution_time=_resolve_time(),
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
    )
    prompt = numeric_prompt(question, "research", "", "")

    # Trailing prose "Percentile X: ..." block must be gone entirely.
    assert "Percentile 5:" not in prompt
    assert "Percentile 95:" not in prompt

    structured_section = prompt[prompt.find("STRUCTURED FORECAST") :]
    expected_keys = ["0.01", "0.025", "0.05", "0.1", "0.2", "0.4", "0.5", "0.6", "0.8", "0.9", "0.95", "0.975", "0.99"]
    indices = []
    for key in expected_keys:
        token = f'"{key}"'
        assert token in structured_section, f"missing percentile key {token} in declared_percentiles example"
        indices.append(structured_section.find(token))
    assert indices == sorted(indices), f"percentile keys out of order: {indices}"


# ---------- Numeric utils ---------------------------------------------------


def test_aggregate_numeric_mean_and_median():
    question = NumericQuestion(
        id_of_question=1,
        id_of_post=1,
        page_url="example",
        question_text="?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
    )
    # Note: numeric distribution will add 0% and 100% percentiles if they are not present,
    # so the values being tested are not at the boundaries.
    percentiles = [Percentile(value=v, percentile=p) for v, p in zip([10, 50, 90], [0.1, 0.5, 0.9], strict=True)]
    dist_a = NumericDistribution(declared_percentiles=percentiles, **question.model_dump())
    dist_b = NumericDistribution(declared_percentiles=percentiles, **question.model_dump())

    mean_result = aggregate_numeric([dist_a, dist_b], question, "mean")
    median_result = aggregate_numeric([dist_a, dist_b], question, "median")

    # Both mean and median aggregations now return a full 201-point distribution.
    # Since we are aggregating two identical distributions, the result should be
    # the same as the original interpolated CDF. We can check the 50th percentile.
    mean_p50 = next(p for p in mean_result.declared_percentiles if p.value == 50)
    median_p50 = next(p for p in median_result.declared_percentiles if p.value == 50)

    assert mean_p50.percentile == pytest.approx(0.5)
    assert median_p50.percentile == pytest.approx(0.5)


def test_aggregate_binary_mean():
    assert aggregate_binary_mean([0.4, 0.6]) == 0.5


def test_bound_messages():
    q = NumericQuestion(
        id_of_question=5,
        id_of_post=5,
        page_url="example",
        question_text="?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=0,
        upper_bound=10,
        open_lower_bound=True,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
    )
    upper, lower = bound_messages(q)
    assert "higher" in upper
    # With open lower bound, we now include a practical/display lower bound hint
    assert "0.0" in lower or lower == ""


def test_bound_messages_open_vs_closed_semantics():
    """Open bounds must signal the outcome can resolve beyond the range; closed bounds must not."""
    open_q = NumericQuestion(
        id_of_question=8,
        id_of_post=8,
        page_url="example",
        question_text="?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=75000000.0,
        upper_bound=150000000.0,
        open_lower_bound=True,
        open_upper_bound=True,
        unit_of_measure="$",
        zero_point=None,
    )
    upper, lower = bound_messages(open_q)
    # Open: explicitly licenses resolving past the displayed range, and directs
    # percentiles at/beyond the bound when warranted (the Toy Story 5 fix).
    assert "open" in upper.lower()
    assert "can resolve above" in upper
    assert "at or above" in upper
    assert "open" in lower.lower()
    assert "can resolve below" in lower
    assert "at or below" in lower

    closed_q = open_q.model_copy(update={"open_lower_bound": False, "open_upper_bound": False})
    upper_c, lower_c = bound_messages(closed_q)
    # Closed: hard limit, no "can resolve beyond" language.
    assert "closed" in upper_c.lower()
    assert "can not be higher" in upper_c
    assert "can resolve above" not in upper_c
    assert "closed" in lower_c.lower()
    assert "can not be lower" in lower_c
    assert "can resolve below" not in lower_c


def test_bound_messages_uses_nominal_bounds():
    q = NumericQuestion(
        id_of_question=6,
        id_of_post=6,
        page_url="example",
        question_text="?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
        nominal_lower_bound=5,
        nominal_upper_bound=42,
    )

    upper, lower = bound_messages(q)
    assert "42" in upper
    assert "5" in lower


def test_bound_messages_discrete_fallback():
    """Test that bound_messages derives nominal bounds for discrete questions when missing."""
    # Create a discrete question (cdf_size != 201) without nominal bounds
    q = NumericQuestion(
        id_of_question=7,
        id_of_post=7,
        page_url="example",
        question_text="Discrete question",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=-0.5,  # API bounds are typically off by 0.5 for discrete
        upper_bound=9.5,  # Representing 0-9 discrete values
        open_lower_bound=False,
        open_upper_bound=False,
        unit_of_measure="",
        zero_point=None,
        cdf_size=11,  # 10 discrete values + 1 = 11
    )

    upper, lower = bound_messages(q)
    # Should derive nominal bounds: step = (9.5 - (-0.5)) / (11 - 1) = 1.0
    # nominal_lower = -0.5 + 1.0/2 = 0.0, nominal_upper = 9.5 - 1.0/2 = 9.0
    assert "9.0" in upper
    assert "0.0" in lower


def _numeric_bounds_q(
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 100.0,
    cdf_size: int = PCHIP_CDF_POINTS,
) -> NumericQuestion:
    return NumericQuestion(
        id_of_question=20,
        id_of_post=20,
        page_url="example",
        question_text="?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        open_lower_bound=True,
        open_upper_bound=True,
        unit_of_measure="",
        zero_point=None,
        cdf_size=cdf_size,
    )


def test_nominal_bounds_continuous_returns_raw_bounds():
    """Continuous open question with no nominal_* attrs → raw (upper, lower)."""
    q = _numeric_bounds_q(lower_bound=0.0, upper_bound=100.0)
    upper, lower = nominal_bounds(q)
    assert (upper, lower) == (100.0, 0.0)


def test_nominal_bounds_discrete_half_step_derivation():
    """Discrete question (cdf_size != PCHIP_CDF_POINTS, no nominal attrs) → half-step derived."""
    # Q38195-class: raw bounds [-0.5, 7.5], cdf_size 9 (0..7 = 8 values + 1) → step 1.0.
    q = _numeric_bounds_q(lower_bound=-0.5, upper_bound=7.5, cdf_size=9)
    upper, lower = nominal_bounds(q)
    assert (upper, lower) == (7.0, 0.0)


def test_nominal_bounds_cdf_size_equals_pchip_points_returns_raw():
    """When cdf_size == PCHIP_CDF_POINTS the half-step branch is skipped → raw bounds."""
    q = _numeric_bounds_q(lower_bound=-0.5, upper_bound=7.5, cdf_size=PCHIP_CDF_POINTS)
    upper, lower = nominal_bounds(q)
    assert (upper, lower) == (7.5, -0.5)


# ---------- Compact logger --------------------------------------------------


class DummyQuestion(MetaculusQuestion, ABC):
    pass


class DummyReport(ForecastReport):
    # This is a dummy report for testing the compact logger.
    # It needs to be a valid ForecastReport, so we provide minimal implementations
    # for abstract methods and required fields.
    question: MetaculusQuestion = DummyQuestion(
        id_of_question=99,
        id_of_post=99,
        page_url="dummy_url",
        question_text="?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
    )
    explanation: str = "# Dummy"
    prediction: list[str] = Field(default_factory=list)

    @classmethod
    def make_readable_prediction(cls, prediction: list[str]) -> str:
        return "N/A"

    @classmethod
    async def aggregate_predictions(cls: type, predictions: list, question: MetaculusQuestion) -> DummyReport:
        raise NotImplementedError()

    async def publish_report_to_metaculus(self, metaculus_client: MetaculusClient | None = None) -> None:
        # metaculus_client added to match the 0.2.92 base signature
        # (publish_report_to_metaculus(self, metaculus_client=None)); this double
        # never publishes, so the arg is accepted and ignored.
        raise NotImplementedError()


def test_compact_logger_no_exception(caplog: pytest.LogCaptureFixture) -> None:
    """Test that the compact logger runs without exceptions on a dummy report."""
    compact_log_report_summary([DummyReport()])  # should not raise
