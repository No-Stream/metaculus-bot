"""Configuration and helpers for assembling benchmark forecasting bots."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping

from forecasting_tools import GeneralLlm

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback
from metaculus_bot.llm_configs import PARSER_LLM, RESEARCHER_LLM, SUMMARIZER_LLM

MODEL_CONFIG: Dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 32_000,
    "stream": False,
    "timeout": 480,  # 8 minutes - reasoning models (o3, gpt-5.1) need extra time
    "allowed_tries": 3,
}

BENCHMARK_BOT_CONFIG: Dict[str, Any] = {
    "research_reports_per_question": 1,
    "predictions_per_research_report": 1,
    "use_research_summary_to_forecast": False,
    "publish_reports_to_metaculus": False,
    "folder_to_save_reports_to": None,
    "skip_previously_forecasted_questions": False,
    "research_provider": None,
    "max_questions_per_run": None,
    "is_benchmarking": True,
    "allow_research_fallback": False,
}

DEFAULT_HELPER_LLMS: Dict[str, GeneralLlm] = {
    "summarizer": SUMMARIZER_LLM,
    "parser": PARSER_LLM,
    "researcher": RESEARCHER_LLM,
}


# TODO: add various models from note e.g. gpt 5.2, g3 pro/flash, etc.
MODEL_CATALOG: Dict[str, GeneralLlm] = {
    "qwen3-235b": GeneralLlm(
        model="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
        **MODEL_CONFIG,
    ),
    "deepseek-3.2": GeneralLlm(
        model="openrouter/deepseek/deepseek-chat-v3.2",
        **MODEL_CONFIG,
    ),
    "kimi-k2": GeneralLlm(
        model="openrouter/moonshotai/kimi-k2-thinking",
        **MODEL_CONFIG,
    ),
    "glm-4.5": GeneralLlm(
        model="openrouter/z-ai/glm-4.5",
        **MODEL_CONFIG,
    ),
    "r1-0528": GeneralLlm(
        model="openrouter/deepseek/deepseek-r1-0528",
        **MODEL_CONFIG,
    ),
    "grok-4-fast": GeneralLlm(
        model="openrouter/x-ai/grok-4-fast",
        reasoning={"enabled": True},
        **MODEL_CONFIG,
    ),
    "claude-sonnet-4": build_llm_with_openrouter_fallback(
        model="openrouter/anthropic/claude-sonnet-4",
        reasoning={"max_tokens": 8_000},
        **MODEL_CONFIG,
    ),
    "gpt-5.1": build_llm_with_openrouter_fallback(
        model="openrouter/openai/gpt-5.1",
        reasoning={"effort": "high"},
        **MODEL_CONFIG,
    ),
    "o3": build_llm_with_openrouter_fallback(
        model="openrouter/openai/o3",
        reasoning={"effort": "high"},
        **MODEL_CONFIG,
    ),
    # --- Models below are defined for future testing ---
    # "gpt-5.2": build_llm_with_openrouter_fallback(
    #     model="openrouter/openai/gpt-5.2",
    #     reasoning={"effort": "high"},
    #     **MODEL_CONFIG,
    # ),
    # "gemini-3-pro": GeneralLlm(
    #     model="openrouter/google/gemini-3-pro-preview",
    #     **MODEL_CONFIG,
    # ),
    # "gemini-3-flash": GeneralLlm(
    #     model="openrouter/google/gemini-3-flash-preview",
    #     **MODEL_CONFIG,
    # ),
    # "claude-opus-4.5": build_llm_with_openrouter_fallback(
    #     model="openrouter/anthropic/claude-opus-4.5",
    #     reasoning={"max_tokens": 16_000},
    #     **MODEL_CONFIG,
    # ),
    # "grok-4.1-fast": build_llm_with_openrouter_fallback(
    #     model="openrouter/x-ai/grok-4.1-fast",
    #     reasoning={"effort": "high"},
    #     **MODEL_CONFIG,
    # ),
    # "claude-sonnet-4.5": build_llm_with_openrouter_fallback(
    #     model="openrouter/anthropic/claude-sonnet-4.5",
    #     reasoning={"max_tokens": 16_000},
    #     **MODEL_CONFIG,
    # ),
}

INDIVIDUAL_MODEL_SPECS: tuple[Mapping[str, GeneralLlm], ...] = (
    MappingProxyType({"name": "qwen3-235b", "forecaster": MODEL_CATALOG["qwen3-235b"]}),
    MappingProxyType({"name": "deepseek-3.2", "forecaster": MODEL_CATALOG["deepseek-3.2"]}),
    # MappingProxyType({"name": "kimi-k2", "forecaster": MODEL_CATALOG["kimi-k2"]}),
    # MappingProxyType({"name": "glm-4.5", "forecaster": MODEL_CATALOG["glm-4.5"]}),
    # MappingProxyType({"name": "r1-0528", "forecaster": MODEL_CATALOG["r1-0528"]}),
    # MappingProxyType({"name": "grok-4-fast", "forecaster": MODEL_CATALOG["grok-4-fast"]}),
    # MappingProxyType({"name": "claude-sonnet-4", "forecaster": MODEL_CATALOG["claude-sonnet-4"]}),
    MappingProxyType({"name": "gpt-5.1", "forecaster": MODEL_CATALOG["gpt-5.1"]}),
    # MappingProxyType({"name": "o3", "forecaster": MODEL_CATALOG["o3"]}),
)

STACKING_MODEL_SPECS: tuple[Mapping[str, GeneralLlm], ...] = (
    # MappingProxyType({"name": "stack-qwen3", "stacker": MODEL_CATALOG["qwen3-235b"]}),
    # MappingProxyType({"name": "stack-o3", "stacker": MODEL_CATALOG["o3"]}),
    # MappingProxyType({"name": "stack-claude4", "stacker": MODEL_CATALOG["claude-sonnet-4"]}),
    # MappingProxyType({"name": "stack-gpt5.1", "stacker": MODEL_CATALOG["gpt-5.1"]}),
)


def create_individual_bots(
    model_specs: Iterable[Mapping[str, GeneralLlm]],
    helper_llms: Dict[str, GeneralLlm],
    benchmark_config: Dict[str, Any],
    *,
    batch_size: int,
    research_cache: Dict[int, str],
) -> List[TemplateForecaster]:
    bots: List[TemplateForecaster] = []
    for spec in model_specs:
        bot = TemplateForecaster(
            **benchmark_config,
            aggregation_strategy=AggregationStrategy.MEAN,
            llms={"forecasters": [spec["forecaster"]], **helper_llms},
            max_concurrent_research=batch_size,
            research_cache=research_cache,
        )
        bot.name = spec["name"]
        bots.append(bot)
    return bots


def create_stacking_bots(
    stacking_specs: Iterable[Mapping[str, GeneralLlm]],
    base_forecasters: List[GeneralLlm],
    helper_llms: Dict[str, GeneralLlm],
    benchmark_config: Dict[str, Any],
    *,
    batch_size: int,
    research_cache: Dict[int, str],
) -> List[TemplateForecaster]:
    bots: List[TemplateForecaster] = []
    for spec in stacking_specs:
        bot = TemplateForecaster(
            **benchmark_config,
            aggregation_strategy=AggregationStrategy.STACKING,
            llms={
                "forecasters": base_forecasters,
                "stacker": spec["stacker"],
                **helper_llms,
            },
            max_concurrent_research=batch_size,
            research_cache=research_cache,
            stacking_fallback_on_failure=False,
            stacking_randomize_order=True,
        )
        bot.name = spec["name"]
        bots.append(bot)
    return bots


__all__ = [
    "BENCHMARK_BOT_CONFIG",
    "DEFAULT_HELPER_LLMS",
    "INDIVIDUAL_MODEL_SPECS",
    "MODEL_CONFIG",
    "MODEL_CATALOG",
    "STACKING_MODEL_SPECS",
    "create_individual_bots",
    "create_stacking_bots",
]
