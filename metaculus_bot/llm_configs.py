"""Centralised model configuration for TemplateForecaster.

Keeping these objects in a single module avoids merge-conflicts and makes it
possible to tweak/benchmark models without touching application code.
"""

from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback

__all__ = ["FORECASTER_LLMS", "SUMMARIZER_LLM", "PARSER_LLM", "RESEARCHER_LLM"]
REASONING_MODEL_CONFIG = {
    "temperature": 1.0,  # standard sampling params for recent reasoning models
    "top_p": 0.95,
    "max_tokens": 32_000,  # Prevent truncation issues with reasoning models
    "stream": False,
    "timeout": 300,
    "allowed_tries": 3,
}
QWEN_CONFIG = {  # developer recommends this for qwen models
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 32_000,
    "stream": False,
    "timeout": 300,
    "allowed_tries": 3,
}
DETERMINISTIC_MODEL_CONFIG = {  # used for basic parsing and summarization tasks
    "temperature": 0.0,
    "top_p": 0.9,
    "max_tokens": 32_000,
    "stream": False,
    "timeout": 300,
    "allowed_tries": 3,
}
ACCEPTABLE_QUANTS = [
    "fp8",
    "fp16",
    "bf16",
    "fp32",
    "unknown",
]

FORECASTER_LLMS = [
    # TODO: consider adding add'l LLMs to ensemble
    build_llm_with_openrouter_fallback(
        model="openrouter/openai/gpt-5.1",
        reasoning={"effort": "high"},
        **REASONING_MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/openai/o3",
        reasoning={"effort": "high"},
        **REASONING_MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/anthropic/claude-sonnet-4.5",
        reasoning={"max_tokens": 16_000},
        **REASONING_MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/x-ai/grok-4.1-fast",
        reasoning={"effort": "high"},
        **REASONING_MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/qwen/qwen3-235b-a22b-thinking-2507",
        provider={
            "quantizations": ACCEPTABLE_QUANTS,
        },
        **QWEN_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/moonshotai/kimi-k2-0905",
        provider={
            "quantizations": ACCEPTABLE_QUANTS,
        },
        **QWEN_CONFIG,  # non thinking model but has similar optimal params to qwen3-235b thinking model
    ),
]

SUMMARIZER_LLM: str = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5-mini")
# Parser should be a reliable, low-latency model for structure extraction
PARSER_LLM: str = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5-mini",
    reasoning={"effort": "low"},
    **DETERMINISTIC_MODEL_CONFIG,
)
# Researcher is only used by the base bot when internal research is invoked.
# Our implementation uses providers, but we still set it explicitly to avoid silent defaults.
RESEARCHER_LLM = build_llm_with_openrouter_fallback(
    model="openrouter/openai/gpt-5.1",
)
