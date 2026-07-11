"""Centralised model configuration for TemplateForecaster.

Keeping these objects in a single module avoids merge-conflicts and makes it
possible to tweak/benchmark models without touching application code.
"""

from typing import Any

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
# Reasoning models ignore (or degrade under) explicit sampling params, so we
# defer to provider defaults. temperature=None is load-bearing: GeneralLlm
# injects temperature=0 when the arg is omitted, so None is what makes litellm
# drop it; top_p flows via **kwargs and is simply never set.
REASONING_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
    "max_tokens": 64_000,  # Prevent truncation; all current forecasters/stackers support 64k output
    "stream": False,
    "timeout": 480,
    "allowed_tries": 3,
}
# Low-effort utility slots (parser, summarizer, analyzer). Same sampling-param
# rationale as REASONING_MODEL_CONFIG: temperature=None keeps litellm from
# injecting temperature=0; top_p left unset for provider defaults.
UTILITY_MODEL_CONFIG: dict[str, Any] = {
    "temperature": None,
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

# Per-instance allowed_tries=1 override (Round-2): forecaster .invoke is wrapped
# in the broad, 30s-elapsed-gated retry (forecaster_runners.py) so we can impose
# the universal "no retry after 30s" deadline-safety rule that forecasting-tools'
# un-gated tenacity cannot. Spread per-instance (NOT by mutating
# REASONING_MODEL_CONFIG) so PARSER_LLM / STACKER configs are untouched.
_FORECASTER_CONFIG = {**REASONING_MODEL_CONFIG, "allowed_tries": 1}

FORECASTER_LLMS: list[GeneralLlm] = [
    # 2026-07-09: OpenAI flagship (5.6 series, released today); replaces gpt-5.4. High effort —
    # forecaster quality is the product. (Date anchors the config era for residual analysis.)
    build_llm_with_openrouter_fallback(
        model="openrouter/openai/gpt-5.6-sol",
        reasoning={"effort": "high"},
        **_FORECASTER_CONFIG,
    ),
    # Kept (not migrated to sol) to preserve intra-OpenAI generation diversity
    # alongside gpt-5.6-sol in the ensemble.
    build_llm_with_openrouter_fallback(
        model="openrouter/openai/gpt-5.5",
        reasoning={"effort": "high"},
        **_FORECASTER_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/anthropic/claude-opus-4.8",
        reasoning={"enabled": True},
        extra_body={"verbosity": "high"},
        **_FORECASTER_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/anthropic/claude-opus-4.6",
        # Explicit max_tokens forces budget-based thinking. Without it, Opus 4.6 defaults to
        # "adaptive thinking" (OpenRouter 4.6 migration guide) which is unbounded and has
        # caused silent 600s soft-deadline stalls on hard questions (e.g. Q14333 on 2026-05-07).
        reasoning={"max_tokens": 32_000},
        extra_body={"verbosity": "high"},
        **_FORECASTER_CONFIG,
    ),
    build_llm_with_openrouter_fallback(
        model="openrouter/google/gemini-3.1-pro-preview",
        **_FORECASTER_CONFIG,
    ),
    # 2026-07-08: migrated from x-ai/grok-4.3 to x-ai/grok-4.5 (released today; xAI's
    # newest frontier reasoning model, 500K context, $2/$6 per M input/output tokens).
    # Prior hop 2026-05-18: x-ai/grok-4.1-fast (deprecated 2026-05-15 by xAI) → grok-4.3
    # with explicit reasoning effort=high to match the gpt-5.4/5.5 reasoning peers
    # (4.3 defaulted to low effort if unspecified, vs. 4.1-fast which had no effort flag).
    # effort=high kept from the 4.3 config — 4.5's default-effort behavior isn't yet
    # documented, so preserving the peer-parity setting rather than reverting to default.
    build_llm_with_openrouter_fallback(
        model="openrouter/x-ai/grok-4.5",
        reasoning={"effort": "high"},
        **_FORECASTER_CONFIG,
    ),
]


def _forecaster_display_name(llm: GeneralLlm) -> str:
    """Short label for a forecaster (e.g. 'claude-opus-4.7') — strips the 'openrouter/<provider>/' prefix.

    Used by performance_analysis.parsing to map 'Forecaster N' labels in bot comments
    back to a model name without having to hand-maintain a parallel list.
    """
    return llm.model.rsplit("/", 1)[-1]


FORECASTER_MODEL_NAMES: list[str] = [_forecaster_display_name(llm) for llm in FORECASTER_LLMS]

# Summarizer: compresses raw AskNews article markdown into an analyst briefing
# (AskNews-only; all other providers already emit LLM prose). NOT a saturated
# task — it decides what is load-bearing and which sources to trust in the
# briefing every forecaster reads, which rewards the top tier; low effort keeps
# latency in check (sol-low ≈ terra-medium quality per the AA Pareto frontier).
# allowed_tries=1 (Round-2): the summarizer invoke is wrapped in the broad,
# 30s-gated retry (orchestrator._summarize_asknews) to impose the universal
# "no retry after 30s" deadline rule. Per-instance override so PARSER_LLM (which
# also uses UTILITY_MODEL_CONFIG) keeps its allowed_tries=3.
SUMMARIZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.6-sol",
    reasoning={"effort": "low"},
    **{**UTILITY_MODEL_CONFIG, "allowed_tries": 1},
)
# Parser: deterministic extraction of percentiles/JSON from rationales — a
# capability-saturated task where mini is still cheaper than gpt-5.6-luna
# ($0.75/$4.50 vs $1/$6 per 1M) and keeps allowed_tries=3 for robustness.
PARSER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.4-mini",
    reasoning={"effort": "low"},
    **UTILITY_MODEL_CONFIG,
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
    "openrouter/anthropic/claude-fable-5",
    # Fable 5 uses effort-based adaptive thinking (none/low/medium/high/max), not a
    # max_tokens budget. effort=high + verbosity=high matches the opus-4.8 forecaster slot.
    reasoning={"effort": "high"},
    extra_body={"verbosity": "high"},
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1},
)

# Fallback stacker used when the primary stacker times out or errors.
# Reasoning slot → strongest OpenAI tier (gpt-5.6-sol) at high effort;
# deliberately cross-provider from the Anthropic Fable primary so an Anthropic
# stall doesn't take both attempts down. Tighter timeout and single try since
# we're already running late on the critical path by the time this fires.
STACKER_FALLBACK_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.6-sol",
    reasoning={"effort": "high"},
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1, "timeout": 300},
)

# Keyword-extraction LLM config for the prediction-market provider.
# Keyword extraction is capability-saturated; mini is the cheapest capable tier.
# Per G0 (2026-05-12 prediction_market_keyword_extraction_experiment.md):
# gpt-5.4-mini reasoning=low burns 128-512 tokens on invisible reasoning before
# emitting any visible response, so max_tokens=800 is load-bearing.
# Constructed per-call inside _run_llm rather than as a singleton because the
# provider is gated OFF by default and we don't want to pay construction cost
# (or break the existing test pattern that patches build_llm_with_openrouter_fallback).
# temperature=None (not omitted): GeneralLlm injects temperature=0 otherwise;
# reasoning models defer to provider defaults. top_p left unset.
PREDICTION_MARKET_KEYWORD_LLM_CONFIG: dict = {
    "model": "openrouter/openai/gpt-5.4-mini",
    "temperature": None,
    "max_tokens": 800,
    "reasoning_effort": "low",
    "timeout": 60,
}


# Tier-B auxiliary: read-and-synthesize work that needs taste but not deep
# reasoning. Identifies the crux of forecaster disagreement; output text seeds
# the targeted-search query downstream. Runs under CRUX_SOFT_DEADLINE (180s);
# effort deliberately low since 2026-05-20 for latency — the tier was upgraded
# instead (smarter-model-at-lower-effort beats more effort on a smaller model).
# allowed_tries=1 (Round-2): the crux-analyzer invoke is wrapped in the broad,
# 30s-gated retry (targeted.extract_disagreement_crux) to impose the universal
# "no retry after 30s" deadline rule on the conditional-stacking critical path.
# Per-instance override so PARSER_LLM keeps its allowed_tries=3.
DISAGREEMENT_ANALYZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.6-sol",
    reasoning={"effort": "low"},
    **{**UTILITY_MODEL_CONFIG, "allowed_tries": 1},
)
