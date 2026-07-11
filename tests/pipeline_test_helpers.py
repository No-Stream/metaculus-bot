"""Shared test infrastructure for end-to-end pipeline tests.

Provides realistic question factories, canned LLM responses, and a prompt-routing
LlmRouter that dispatches responses based on prompt content. Designed so pipeline
tests exercise the FULL production code path with deterministic outputs.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from forecasting_tools import BinaryQuestion, GeneralLlm, MultipleChoiceQuestion, NumericQuestion

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy

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
        # All forecaster + stacker prompts now emit a fenced STRUCTURED FORECAST
        # JSON block. Stacker prompts also match — the __call__ order routes them
        # via ``_is_stacker_prompt`` first, so this signal only sees non-stacker
        # LLM calls. The old ``"Probability:"`` cue is gone: the base binary
        # prompt no longer requests a trailing "Probability: NN%" line.
        return "STRUCTURED FORECAST" in prompt

    def _detect_question_type(self, prompt: str) -> str:
        # Numeric prompt talks about "percentiles" extensively; MC prompt has an
        # "Options (in resolution order):" line the others lack. Otherwise binary.
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


def make_real_binary_question(qid: int = 1001) -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will the US unemployment rate exceed 5% by December 2026?",
        id_of_question=qid,
        id_of_post=qid + 10000,
        page_url=f"https://www.metaculus.com/questions/{qid}/",
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
        page_url=f"https://www.metaculus.com/questions/{qid}/",
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


def make_real_mc_question(
    qid: int = 3001,
    options: list[str] | None = None,
) -> MultipleChoiceQuestion:
    _options = options if options is not None else ["Option A", "Option B", "Option C"]
    return MultipleChoiceQuestion(
        question_text="Which economic scenario is most likely for the US in 2026?",
        id_of_question=qid,
        id_of_post=qid + 10000,
        page_url=f"https://www.metaculus.com/questions/{qid}/",
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
