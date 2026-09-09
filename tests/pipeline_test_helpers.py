"""Shared test infrastructure for end-to-end pipeline tests.

Provides realistic question factories, canned LLM responses, and a prompt-routing
LlmRouter that dispatches responses based on prompt content. Designed so pipeline
tests exercise the FULL production code path with deterministic outputs.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import numpy as np
from forecasting_tools import BinaryQuestion, GeneralLlm, MultipleChoiceQuestion, NumericDistribution, NumericQuestion
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import DateQuestion, DiscreteQuestion, MetaculusQuestion

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import MANTIC_SITE_URL
from metaculus_bot.numeric.config import PMF_ELICITATION_MAX_BINS, PMF_FLOOR_MARGIN, grid_step_constraints
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.numeric.pchip_processing import create_pchip_numeric_distribution

# ---------------------------------------------------------------------------
# Canned LLM responses
# ---------------------------------------------------------------------------

CANNED_BINARY_REASONING = """\
## Analysis

The US unemployment rate has been relatively stable between 3.4% and 4.2% over the past year. \
Historical data shows that crossing the 5% threshold typically requires a significant economic shock \
such as a recession. Current leading indicators (initial jobless claims, ISM manufacturing PMI, \
yield curve) are mixed but do not strongly signal an imminent recession.

The Federal Reserve has maintained a restrictive stance, which could slow growth, but labor markets \
remain tight with job openings exceeding unemployed workers by approximately 1.3:1.

Key considerations:
- Base rate: US unemployment has exceeded 5% in roughly 4 of the last 20 years
- Current trajectory suggests gradual cooling, not a sharp spike
- No major recession trigger is currently identified by consensus forecasters

```json
{"question_type": "binary", "posterior_prob": 0.22}
```
"""

CANNED_NUMERIC_REASONING = """\
## Analysis

The US unemployment rate has been in the 3.4-4.2% range throughout the past year. Economic \
indicators suggest a gradual cooling of the labor market. The Fed's restrictive monetary policy \
is expected to continue moderating employment growth. Historical patterns suggest unemployment \
typically rises 0.3-0.8pp during a soft landing scenario.

```json
{
  "question_type": "numeric",
  "declared_percentiles": {
    "0.01": 3.0, "0.025": 3.2, "0.05": 3.4, "0.1": 3.6, "0.2": 3.8, "0.4": 4.1, "0.5": 4.3,
    "0.6": 4.5, "0.8": 5.0, "0.9": 5.6, "0.95": 6.2, "0.975": 7.0, "0.99": 7.8
  },
  "outcome_type": "continuous"
}
```
"""

CANNED_MC_REASONING = """\
## Analysis

Evaluating the three options based on current geopolitical and economic trends. Option A has \
strong institutional support and momentum. Option B represents the status quo with gradual erosion. \
Option C is a tail scenario requiring multiple unlikely events.

My assessment of probabilities:
- Option A: 45%
- Option B: 40%
- Option C: 15%

```json
{"question_type": "multiple_choice", "option_probs": {"Option A": 0.45, "Option B": 0.40, "Option C": 0.15}}
```
"""

CANNED_STACKER_BINARY_REASONING = """\
## Meta-Analysis

After reviewing the three base model analyses, I observe moderate agreement centered around 20-32%. \
Model 1 emphasizes labor market tightness (22%), Model 2 weights recession base rates more heavily (30%), \
and Model 3 takes an intermediate position (31%). The disagreement is driven primarily by differing \
priors on recession probability in the next 6 months.

Synthesizing: the labor market data is more current and reliable than base-rate extrapolation alone. \
I weight Model 1's reasoning slightly higher but incorporate the uncertainty flagged by Models 2 and 3.

```json
{"question_type": "binary", "posterior_prob": 0.25}
```
"""

CANNED_RESEARCH_TEXT = """\
## Research Summary

The US Bureau of Labor Statistics reported the unemployment rate at 4.1% in the most recent release \
(April 2026). Initial jobless claims have been trending slightly upward over the past 8 weeks, \
averaging 235,000 per week compared to 210,000 six months ago.

The Federal Reserve maintained the federal funds rate at 5.25-5.50% at its last meeting, with \
dot plot projections suggesting one rate cut by year-end. GDP growth for Q1 2026 came in at \
2.1% annualized, below the 2.8% consensus estimate.

Sources: BLS Employment Situation Report, Federal Reserve FOMC Minutes, BEA GDP Advance Estimate.
"""


# ---------------------------------------------------------------------------
# LlmRouter — dispatches canned responses based on prompt content
# ---------------------------------------------------------------------------


class LlmRouter:
    """Callable that routes LLM invocations to canned responses based on prompt content.

    Records all prompts received in ``self.calls`` for test assertions.
    """

    def __init__(
        self,
        *,
        forecaster_responses: dict[str, list[str]] | None = None,
        stacker_response: str = CANNED_STACKER_BINARY_REASONING,
        parser_responses: dict[str, Any] | None = None,
        research_response: str = CANNED_RESEARCH_TEXT,
        stacker_side_effect: BaseException | None = None,
    ) -> None:
        self.calls: list[str] = []
        self._forecaster_responses = forecaster_responses or {
            "binary": [CANNED_BINARY_REASONING] * 10,
            "numeric": [CANNED_NUMERIC_REASONING] * 10,
            "mc": [CANNED_MC_REASONING] * 10,
        }
        self._stacker_response = stacker_response
        self._stacker_side_effect = stacker_side_effect
        self._parser_responses = parser_responses or {}
        self._research_response = research_response
        self._forecaster_call_counts: dict[str, int] = {"binary": 0, "numeric": 0, "mc": 0}
        self._stacker_call_count = 0

    async def __call__(self, prompt: str, **kwargs: Any) -> str:

        await asyncio.sleep(0)
        self.calls.append(prompt)

        if self._is_stacker_prompt(prompt):
            self._stacker_call_count += 1
            if self._stacker_side_effect is not None:
                raise self._stacker_side_effect
            return self._stacker_response

        if self._is_forecaster_prompt(prompt):
            qtype = self._detect_question_type(prompt)
            idx = self._forecaster_call_counts[qtype]
            self._forecaster_call_counts[qtype] += 1
            responses = self._forecaster_responses[qtype]
            return responses[idx % len(responses)]

        return self._research_response

    def _is_stacker_prompt(self, prompt: str) -> bool:
        stacker_signals = ["meta-forecaster", "synthesize multiple expert", "Model 1 Analysis:"]
        return any(signal.lower() in prompt.lower() for signal in stacker_signals)

    def _is_forecaster_prompt(self, prompt: str) -> bool:
        """Every forecaster and stacker prompt asks for a fenced STRUCTURED FORECAST block.

        Stacker prompts match too; ``__call__`` routes them through ``_is_stacker_prompt`` first,
        so this signal only sees non-stacker LLM calls.
        """
        return "STRUCTURED FORECAST" in prompt

    def _detect_question_type(self, prompt: str) -> str:
        """Numeric prompts talk about percentiles, MC prompts list the options in resolution order; else binary."""
        if "percentile" in prompt.lower():
            return "numeric"
        if "options (in resolution order)" in prompt.lower():
            return "mc"
        return "binary"

    @property
    def stacker_was_called(self) -> bool:
        return self._stacker_call_count > 0

    @property
    def stacker_prompts(self) -> list[str]:
        return [c for c in self.calls if self._is_stacker_prompt(c)]


# ---------------------------------------------------------------------------
# Question factories — REAL instances (not MagicMock)
# ---------------------------------------------------------------------------

_OPEN_TIME = datetime(2026, 1, 1)
_RESOLVE_TIME = datetime(2026, 12, 31)


def metaculus_url(post_id: int | None) -> str:
    """The ``page_url`` the framework writes on every question it parses, whichever host served the payload."""
    return f"https://www.metaculus.com/questions/{post_id}/"


def mantic_url(post_id: int | None) -> str:
    """The ``page_url`` ``ManticClient`` rewrites onto every question it parses; ``question_platform`` reads the host."""
    return f"{MANTIC_SITE_URL}/questions/{post_id}/"


def on_mantic[Q: MetaculusQuestion](question: Q) -> Q:
    """``question`` as ``ManticClient`` hands it over.

    The framework parses every payload with a metaculus.com ``page_url``; the client rewrites it to the
    Mantic host, which is the one fact every platform gate reads.
    """
    return question.model_copy(update={"page_url": mantic_url(question.id_of_post)})


def make_real_binary_question(qid: int = 1001, close_time: datetime | None = None) -> BinaryQuestion:
    """A real BinaryQuestion. ``close_time`` drives the per-question time budget.

    Left None by default so the existing e2e suite keeps the pre-time-budget shape
    (no deadline -> the static budget); pass it to exercise a thin window.
    """
    return BinaryQuestion(
        question_text="Will the US unemployment rate exceed 5% by December 2026?",
        id_of_question=qid,
        id_of_post=qid + 10000,
        page_url=metaculus_url(qid),
        background_info=(
            "The US unemployment rate has been between 3.4% and 4.2% for the past year. "
            "Historical data shows that spikes above 5% are typically associated with recessions."
        ),
        resolution_criteria=(
            "Resolves YES if the Bureau of Labor Statistics reports a seasonally adjusted "
            "unemployment rate of 5.0% or higher for any month through December 2026."
        ),
        fine_print="Uses seasonally adjusted figures from the BLS Employment Situation report.",
        open_time=_OPEN_TIME,
        close_time=close_time,
        scheduled_resolution_time=_RESOLVE_TIME,
        api_json={"my_forecasts": {"latest": {"forecast_values": [0.3]}}},
    )


def make_real_numeric_question(
    qid: int = 2001,
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 20.0,
    open_lower_bound: bool = False,
    open_upper_bound: bool = True,
    zero_point: float | None = None,
) -> NumericQuestion:
    return NumericQuestion(
        question_text="What will the US unemployment rate be in December 2026?",
        id_of_question=qid,
        id_of_post=qid + 10000,
        page_url=metaculus_url(qid),
        background_info=(
            "The US unemployment rate is reported monthly by the Bureau of Labor Statistics. "
            "It has ranged from 3.4% to 4.2% over the past 12 months."
        ),
        resolution_criteria=(
            "Resolves to the seasonally adjusted unemployment rate reported by BLS for December 2026."
        ),
        fine_print="If the December 2026 report is revised, uses the initial release value.",
        open_time=_OPEN_TIME,
        scheduled_resolution_time=_RESOLVE_TIME,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        open_lower_bound=open_lower_bound,
        open_upper_bound=open_upper_bound,
        zero_point=zero_point,
        unit_of_measure="%",
        api_json={"my_forecasts": {"latest": {"forecast_values": None}}},
    )


def make_count_question(
    bins: int,
    *,
    page_url: str | None = None,
    open_lower: bool = False,
    open_upper: bool = True,
    qid: int = 700,
) -> DiscreteQuestion:
    """A count question on ``bins`` integer bins, 0 through ``bins - 1``, in the platform's discrete convention.

    Half-step range bounds (``-0.5`` to ``bins - 0.5``), ``cdf_size = bins + 1`` and nominal bounds on
    the first and last count: the shape the framework parses for every Metaculus discrete question and
    ``ManticClient`` for every Mantic quantitative one. ``page_url`` defaults to the Mantic host, where
    a grid of ``PMF_ELICITATION_MAX_BINS`` bins or fewer is elicited per bin; pass ``metaculus_url(qid)``
    for the same grid off the platform.
    """
    return DiscreteQuestion(
        id_of_question=qid,
        id_of_post=qid,
        page_url=mantic_url(qid) if page_url is None else page_url,
        question_text="How many?",
        background_info="",
        resolution_criteria="",
        fine_print="",
        published_time=None,
        close_time=None,
        lower_bound=-0.5,
        upper_bound=bins - 0.5,
        open_lower_bound=open_lower,
        open_upper_bound=open_upper,
        unit_of_measure="",
        zero_point=None,
        cdf_size=bins + 1,
        nominal_lower_bound=0.0,
        nominal_upper_bound=float(bins - 1),
    )


def make_real_date_question(
    qid: int = 4001,
    *,
    lower_bound: datetime = datetime(2026, 9, 8, tzinfo=UTC),
    upper_bound: datetime = datetime(2026, 9, 20, tzinfo=UTC),
    open_lower_bound: bool = False,
    open_upper_bound: bool = False,
    cdf_size: int = 13,
    date_granularity: str = "day",
    with_scaling: bool = True,
) -> DateQuestion:
    """A real DateQuestion in post 651's shape: twelve one-day bins, both bounds closed.

    ``api_json`` carries the ``question.scaling`` block the epoch adapter reads its nominal bounds
    from, in the wire convention: ``nominal_max`` one bin below ``range_max`` on a day- or
    week-granularity question, equal to it on a legacy one. ``with_scaling=False`` omits the block,
    the shape of a question not built from API JSON.
    """
    question_json: dict[str, Any] = {"date_granularity": date_granularity}
    if with_scaling:
        bin_width = (upper_bound - lower_bound) / (cdf_size - 1)
        nominal_max = upper_bound - bin_width if date_granularity in ("day", "week") else upper_bound
        question_json["scaling"] = {
            "range_min": lower_bound.timestamp(),
            "range_max": upper_bound.timestamp(),
            "nominal_min": lower_bound.timestamp(),
            "nominal_max": nominal_max.timestamp(),
            "zero_point": None,
            "inbound_outcome_count": cdf_size - 1,
        }
    return DateQuestion(
        question_text="On which date will the S&P 500 post its largest single-day percentage move?",
        id_of_question=qid,
        id_of_post=qid + 10000,
        page_url=mantic_url(qid),
        background_info="The window covers the trading days between the open and the close of the question.",
        resolution_criteria=(
            "Resolves to the UTC calendar date of the trading day with the largest absolute percentage "
            "move in the S&P 500 index during the window."
        ),
        fine_print="Ties resolve to the earlier date.",
        open_time=_OPEN_TIME,
        scheduled_resolution_time=_RESOLVE_TIME,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        open_lower_bound=open_lower_bound,
        open_upper_bound=open_upper_bound,
        cdf_size=cdf_size,
        api_json={"question": question_json, "my_forecasts": {"latest": {"forecast_values": None}}},
    )


def make_real_mc_question(
    qid: int = 3001,
    options: list[str] | None = None,
) -> MultipleChoiceQuestion:
    _options = options if options is not None else ["Option A", "Option B", "Option C"]
    return MultipleChoiceQuestion(
        question_text="Which economic scenario is most likely for the US in 2026?",
        id_of_question=qid,
        id_of_post=qid + 10000,
        page_url=metaculus_url(qid),
        background_info=(
            "Multiple economic scenarios are possible depending on Fed policy, "
            "geopolitical developments, and consumer spending trends."
        ),
        resolution_criteria="Resolves to the option that best describes the realized outcome by year-end.",
        fine_print="Resolution determined by a panel of three economists.",
        open_time=_OPEN_TIME,
        scheduled_resolution_time=_RESOLVE_TIME,
        options=_options,
        api_json={"my_forecasts": {"latest": {"forecast_values": None}}},
    )


# ---------------------------------------------------------------------------
# Bot factory
# ---------------------------------------------------------------------------


def make_e2e_bot(
    strategy: AggregationStrategy,
    n_forecasters: int = 3,
    **overrides: Any,
) -> TemplateForecaster:
    """Create a TemplateForecaster configured for deterministic e2e testing."""
    test_llm = GeneralLlm(model="test-model", temperature=0.0)
    defaults: dict[str, Any] = {
        "publish_reports_to_metaculus": False,
        "is_benchmarking": True,
        "research_reports_per_question": 1,
        "min_forecasters_to_publish": 2,
        "aggregation_strategy": strategy,
        "stacking_fallback_on_failure": True,
        "stacking_randomize_order": False,
        "llms": {
            "forecasters": [test_llm] * n_forecasters,
            "stacker": test_llm,
            "analyzer": test_llm,
            "default": test_llm,
            "parser": test_llm,
            "researcher": test_llm,
            "summarizer": test_llm,
        },
    }
    defaults.update(overrides)
    return TemplateForecaster(**defaults)


# ---------------------------------------------------------------------------
# Platform CDF acceptance replica
# ---------------------------------------------------------------------------


# The per-bin oracle set: five grid sizes across the elicitable range, the smallest coarse Mantic grid to the threshold.
ORACLE_BIN_COUNTS = (3, 4, 12, 21, PMF_ELICITATION_MAX_BINS)


def server_min_step(inbound: int) -> float:
    """The platform's per-bin minimum for ``inbound`` bins, rounded the way the server rounds it."""
    return round(0.01 / inbound, 9)


def server_max_step(inbound: int) -> float:
    """The platform's per-bin maximum for ``inbound`` bins, UNROUNDED, as the server compares it."""
    return 0.2 * 200 / inbound


def assert_server_accepts_cdf(probs: np.ndarray, *, cdf_size: int, open_lower: bool, open_upper: bool) -> None:
    """Fail exactly where the platform's ``continuous_cdf`` validator would reject the submission.

    Replicated from ``questions/serializers/common.py`` in the open-source Metaculus backend,
    which Mantic forked with identical constants: the CDF is rounded to 10 decimals, its PMF to
    9, and the rounded PMF is compared against the rounded min step and the UNROUNDED max step.
    """
    inbound = cdf_size - 1
    assert len(probs) == inbound + 1, f"len(continuous_cdf)={len(probs)} != inbound_outcome_count + 1={inbound + 1}"
    assert not np.any(np.isnan(probs))

    rounded = np.round(probs, 10)
    pmf = np.round(np.diff(rounded), 9)
    min_diff = server_min_step(inbound)
    max_diff = server_max_step(inbound)
    assert np.all(pmf >= min_diff), f"step below server min {min_diff}: min pmf {pmf.min()} at {int(np.argmin(pmf))}"
    assert np.all(pmf <= max_diff), f"step above server max {max_diff}: max pmf {pmf.max()} at {int(np.argmax(pmf))}"

    if open_lower:
        assert rounded[0] >= 0.001, f"open lower bound cdf[0]={rounded[0]} < 0.001"
    else:
        assert rounded[0] == 0.0, f"closed lower bound cdf[0]={rounded[0]} != 0.0"
    if open_upper:
        assert rounded[-1] <= 0.999, f"open upper bound cdf[-1]={rounded[-1]} > 0.999"
    else:
        assert rounded[-1] == 1.0, f"closed upper bound cdf[-1]={rounded[-1]} != 1.0"


def cdf_heights(distribution: NumericDistribution) -> np.ndarray:
    """The CDF heights of a built distribution, one per grid point, as the server receives them."""
    return np.asarray([point.percentile for point in distribution.get_cdf()], dtype=float)


def pmf_of(distribution: NumericDistribution) -> np.ndarray:
    """The per-bin mass of a built distribution: the first difference of its CDF heights."""
    return np.diff(cdf_heights(distribution))


def distribution_from_heights(heights: np.ndarray, question: NumericQuestion) -> NumericDistribution:
    """A published-shape distribution with exactly these CDF heights on the question's canonical grid."""
    values = build_cdf_value_grid(question.lower_bound, question.upper_bound, None, len(heights))
    declared = [Percentile(percentile=float(h), value=float(v)) for h, v in zip(heights, values, strict=True)]
    return create_pchip_numeric_distribution(
        pchip_cdf=[float(h) for h in heights], percentile_list=declared, question=question, zero_point=None
    )


def certain_of_bin(view: NumericQuestion, bin_index: int) -> NumericDistribution:
    """A per-bin member certain of one bin on ``view``'s grid: the platform floor on every other bin.

    The shape the per-bin floor blend hands the aggregator: a zero declared on a bin lands at
    ``min_step + PMF_FLOOR_MARGIN`` and the believed bin keeps the rest. Built straight from the
    heights, the way the tail floor rebuilds a published aggregate, so it needs no elicitation and no
    percentiles.
    """
    min_step, _ = grid_step_constraints(view.cdf_size)
    floor = min_step + PMF_FLOOR_MARGIN
    pmf = np.full(view.cdf_size - 1, floor)
    pmf[bin_index] = 1.0 - (pmf.size - 1) * floor
    heights = np.concatenate(([0.0], np.cumsum(pmf)))
    heights[-1] = 1.0
    return distribution_from_heights(heights, view)
