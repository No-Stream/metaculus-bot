"""Centralised model configuration for TemplateForecaster.

Keeping these objects in a single module avoids merge-conflicts and makes it
possible to tweak/benchmark models without touching application code.
"""

from forecasting_tools import GeneralLlm

from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback

__all__ = [
    "FORECASTER_LLMS",
    "FORECASTER_MODEL_NAMES",
    "SUMMARIZER_LLM",
    "PARSER_LLM",
    "RESEARCHER_LLM",
    "STACKER_LLM",
    "STACKER_FALLBACK_LLM",
    "DISAGREEMENT_ANALYZER_LLM",
]
REASONING_MODEL_CONFIG = {
    "temperature": 1.0,  # standard sampling params for recent reasoning models
    "top_p": 0.95,
    "max_tokens": 32_000,  # Prevent truncation issues with reasoning models
    "stream": False,
    "timeout": 480,
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

FORECASTER_LLMS: list[GeneralLlm] = [
    build_llm_with_openrouter_fallback(
        model="openrouter/openai/gpt-5.4",
        reasoning={"effort": "high"},
        **REASONING_MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/openai/gpt-5.5",
        reasoning={"effort": "high"},
        **REASONING_MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/anthropic/claude-opus-4.7",
        reasoning={"enabled": True},
        extra_body={"verbosity": "high"},
        **REASONING_MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/anthropic/claude-opus-4.6",
        reasoning={"enabled": True},
        extra_body={"verbosity": "high"},
        **REASONING_MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/google/gemini-3.1-pro-preview",
        **REASONING_MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/x-ai/grok-4.1-fast",
        **REASONING_MODEL_CONFIG,
    ),
]


def _forecaster_display_name(llm: GeneralLlm) -> str:
    """Short label for a forecaster (e.g. 'claude-opus-4.7') — strips the 'openrouter/<provider>/' prefix.

    Used by performance_analysis.parsing to map 'Forecaster N' labels in bot comments
    back to a model name without having to hand-maintain a parallel list.
    """
    return llm.model.rsplit("/", 1)[-1]


FORECASTER_MODEL_NAMES: list[str] = [_forecaster_display_name(llm) for llm in FORECASTER_LLMS]

SUMMARIZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback("openrouter/google/gemini-3-flash-preview", timeout=120)
# Parser should be a reliable, low-latency model for structure extraction
PARSER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5-mini",
    reasoning={"effort": "low"},
    **DETERMINISTIC_MODEL_CONFIG,
)
# Researcher is only used by the base bot when internal research is invoked.
# Our implementation uses providers, but we still set it explicitly to avoid silent defaults.
RESEARCHER_LLM = build_llm_with_openrouter_fallback(model="openrouter/google/gemini-3-flash-preview", timeout=120)

# Stacker meta-model for conditional stacking (invoked only on high-disagreement questions).
#
# allowed_tries=1: a single 8-minute attempt with no retries. The outer
# STACKER_SOFT_DEADLINE (500s) catches wholly stuck calls; on failure we fall
# back to STACKER_FALLBACK_LLM rather than burning another 16 min of retries
# against the same Anthropic API that just stalled. Retrying against the same
# provider after a stall rarely succeeds (we're almost certainly re-rolling a
# dice with the same distribution), and the budget is better spent on a
# different-provider fallback.
STACKER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/anthropic/claude-opus-4.7",
    reasoning={"enabled": True},
    extra_body={"verbosity": "xhigh"},
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1},
)

# Fallback stacker used when the primary stacker times out or errors. Different
# provider on purpose: if Anthropic is thrashing, retrying against Anthropic
# is unlikely to recover; gpt-5.5 via OpenAI gives us an independent failure
# mode. Tighter timeout and single try since we're already running late on
# the critical path by the time this fires.
STACKER_FALLBACK_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.5",
    reasoning={"effort": "high"},
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1, "timeout": 300},
)

# Cheap model for identifying the crux of model disagreement (feeds into targeted research)
DISAGREEMENT_ANALYZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5-mini",
    reasoning={"effort": "low"},
    **DETERMINISTIC_MODEL_CONFIG,
)
