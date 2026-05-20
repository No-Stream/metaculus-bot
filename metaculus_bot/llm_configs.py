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
    "PREDICTION_MARKET_KEYWORD_LLM_CONFIG",
]
REASONING_MODEL_CONFIG = {
    "temperature": 1.0,  # standard sampling params for recent reasoning models
    "top_p": 0.95,
    "max_tokens": 64_000,  # Prevent truncation; all current forecasters/stackers support 64k output
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
        # Explicit max_tokens forces budget-based thinking. Without it, Opus 4.6 defaults to
        # "adaptive thinking" (OpenRouter 4.6 migration guide) which is unbounded and has
        # caused silent 600s soft-deadline stalls on hard questions (e.g. Q14333 on 2026-05-07).
        # Opus 4.7 is unaffected by the default change per the migration guide's scope.
        reasoning={"max_tokens": 32_000},
        extra_body={"verbosity": "high"},
        **REASONING_MODEL_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/google/gemini-3.1-pro-preview",
        **REASONING_MODEL_CONFIG,
    ),
    # 2026-05-18: migrated from x-ai/grok-4.1-fast (deprecated 2026-05-15 by xAI).
    # Added explicit reasoning effort=high to match the gpt-5.4/5.5 reasoning peers
    # (4.3 defaults to low effort if unspecified, vs. 4.1-fast which had no effort flag).
    build_llm_with_openrouter_fallback(
        model="openrouter/x-ai/grok-4.3",
        reasoning={"effort": "high"},
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

# Summarizer: compresses AskNews/Perplexity research output. Migrated 2026-05-17
# from gemini-3-flash-preview to gpt-5.4-mini for: (1) consistency with the rest
# of the OpenAI-based support stack (analyzer, parser, native search), (2) lower
# rate-limit exposure than the donated-key Google route, (3) the donated-key
# data-policy block on OpenAI is expected to be lifted; until then, summarizer
# bills to personal OPENROUTER_API_KEY (~$0.01/call × every Q).
SUMMARIZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.4-mini",
    reasoning={"effort": "low"},
    **DETERMINISTIC_MODEL_CONFIG,
)
# Parser should be a reliable, low-latency model for structure extraction
PARSER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5-mini",
    reasoning={"effort": "low"},
    **DETERMINISTIC_MODEL_CONFIG,
)
# Researcher slot in the forecasting-tools LLM config dict. Effectively dead
# code in our pipeline — we use research providers (AskNews/Gemini/native_search)
# rather than the framework's researcher path — but the slot must be populated
# to avoid silent framework defaults. Aliasing to SUMMARIZER_LLM rather than
# constructing a duplicate config: same model, same effort, same job tier, no
# reason to maintain two parallel definitions.
RESEARCHER_LLM = SUMMARIZER_LLM

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
    "openrouter/anthropic/claude-opus-4.5",
    reasoning={"max_tokens": 32_000},  # Opus 4.5 uses explicit thinking budget, not effort/verbosity levels
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

# Keyword-extraction LLM config for the prediction-market provider.
# Per G0 (2026-05-12 prediction_market_keyword_extraction_experiment.md):
# gpt-5-mini reasoning=low burns 128-512 tokens on invisible reasoning before
# emitting any visible response, so max_tokens=800 is load-bearing.
# Constructed per-call inside _run_llm rather than as a singleton because the
# provider is gated OFF by default and we don't want to pay construction cost
# (or break the existing test pattern that patches build_llm_with_openrouter_fallback).
PREDICTION_MARKET_KEYWORD_LLM_CONFIG: dict = {
    "model": "openrouter/openai/gpt-5-mini",
    "temperature": 0.0,
    "max_tokens": 800,
    "reasoning_effort": "low",
    "timeout": 60,
}


# Tier-B auxiliary: read-and-synthesize work that needs taste but not deep
# reasoning. Identifies the crux of forecaster disagreement; output text seeds
# the targeted-search query downstream. Dropped 2026-05-20 from medium→low
# effort alongside the broader tier-B consolidation (native_search also at low):
# 1-3 sentence crux extraction is structure-following with light judgment, not
# deep reasoning. Cost roughly halves (~$4 → ~$1.50/tournament).
DISAGREEMENT_ANALYZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.5",
    reasoning={"effort": "low"},
    **DETERMINISTIC_MODEL_CONFIG,
)
