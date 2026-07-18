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
    # 2026-07-09: OpenAI flagship (5.6 series, released today); replaces gpt-5.4.
    # 2026-07-15: effort high -> xhigh (top OpenAI tier; forecaster quality is the
    # product). Live-verified: OpenRouter's effort enum is
    # max|xhigh|high|medium|low|minimal|none and all four bumped models accept
    # xhigh (bogus values 400). NOTE: "max" is Anthropic-only — OpenAI's ceiling
    # is xhigh and OpenAI rejects max upstream even though OpenRouter's enum
    # validation admits it. (Dates anchor config eras for residual analysis.)
    build_llm_with_openrouter_fallback(
        model="openrouter/openai/gpt-5.6-sol",
        reasoning={"effort": "xhigh"},
        **_FORECASTER_CONFIG,
    ),
    # Kept (not migrated to sol) to preserve intra-OpenAI generation diversity
    # alongside gpt-5.6-sol in the ensemble. 2026-07-15: effort high -> xhigh.
    build_llm_with_openrouter_fallback(
        model="openrouter/openai/gpt-5.5",
        reasoning={"effort": "xhigh"},
        **_FORECASTER_CONFIG,
    ),
    # 2026-07-15: Fable-5 joins the forecaster roster (roster change = new config
    # era for residual analysis). Previously stacker-only — but stacking is
    # disabled in prod (all workflow yamls pin *_STACKING_ENABLED=false), so our
    # top Anthropic tier was idle in every prod run. Same effort=xhigh +
    # verbosity=high config as its stacker slot; "max" held back for latency
    # (FORECASTER_SOFT_DEADLINE=600s). Cost: $10/$50 per M in/out — 2x opus-4.8;
    # donated-key eligible (Anthropic provider).
    build_llm_with_openrouter_fallback(
        model="openrouter/anthropic/claude-fable-5",
        reasoning={"effort": "xhigh"},
        extra_body={"verbosity": "high"},
        **_FORECASTER_CONFIG,
    ),
    # 2026-07-15: enabled:True (provider-default adaptive thinking) -> explicit
    # effort=xhigh. Anthropic also exposes "max" one tier above xhigh — held back
    # deliberately for latency (FORECASTER_SOFT_DEADLINE=600s; unbounded adaptive
    # thinking caused silent 600s soft-deadline stalls on the retired opus-4.6
    # slot, e.g. Q14333 on 2026-05-07).
    build_llm_with_openrouter_fallback(
        model="openrouter/anthropic/claude-opus-4.8",
        reasoning={"effort": "xhigh"},
        extra_body={"verbosity": "high"},
        **_FORECASTER_CONFIG,
    ),
    # 2026-07-15: opus-4.6 retired from the roster — Fable-5 takes the second
    # Anthropic slot, keeping ensemble size at 6 and provider balance at
    # 2 Anthropic / 2 OpenAI / 1 Google / 1 xAI. (4.6 was the older Anthropic
    # tier; its adaptive-thinking stall workaround — reasoning={"max_tokens":
    # 32_000} instead of effort-based — goes with it. See git history.)
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
# (AskNews-only; all other providers already emit LLM prose). sol → terra
# 2026-07-18 operator decision: AskNews is an auxiliary/augmenting source
# (content audit: 16% unique content vs native-search 54% / gap-fill 59%), so
# the absolute-frontier tier isn't warranted. The role audit
# (scratch/research_role_audit_2026-07-17/) had sol 1st but verdict "MARGINAL
# EDGE" with terra 2nd (one attribution blur, no fabrications), and 4/5 briefing
# failures in the AskNews quality audit (scratch/asknews_quality_audit_2026-07-18/)
# were prompt-era (mini summarizer + missing no-forecast rule), not model-tier.
# Terra: −43% cost, ~50s vs ~118s wall. Effort stays low (latency).
# allowed_tries=1 (Round-2): the summarizer invoke is wrapped in the broad,
# 30s-gated retry (orchestrator._summarize_asknews) to impose the universal
# "no retry after 30s" deadline rule. Per-instance override so PARSER_LLM (which
# also uses UTILITY_MODEL_CONFIG) keeps its allowed_tries=3.
SUMMARIZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.6-terra",
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
    # Fable 5 uses effort-based adaptive thinking, not a max_tokens budget. Live-verified
    # OpenRouter effort enum: none/minimal/low/medium/high/xhigh/max. 2026-07-15: effort
    # high -> xhigh, matching the opus-4.8 forecaster slot. "max" (one tier above xhigh)
    # is deliberately held back for latency — the stacker runs under STACKER_SOFT_DEADLINE
    # (500s).
    reasoning={"effort": "xhigh"},
    extra_body={"verbosity": "high"},
    **{**REASONING_MODEL_CONFIG, "allowed_tries": 1},
)

# Fallback stacker used when the primary stacker times out or errors.
# Reasoning slot → strongest OpenAI tier (gpt-5.6-sol) at high effort;
# deliberately cross-provider from the Anthropic Fable primary so an Anthropic
# stall doesn't take both attempts down. Tighter timeout and single try since
# we're already running late on the critical path by the time this fires.
# Stays at high (not xhigh) for that same reason — the 2026-07-15 xhigh bump
# covers the primary stacker and forecaster slots, not this 300s-budget path.
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
# 2026-07-17: sol→terra per the role audit; terra 2nd (sol 3rd) at −49% cost;
# the role fires rarely (stacking disabled in prod).
# allowed_tries=1 (Round-2): the crux-analyzer invoke is wrapped in the broad,
# 30s-gated retry (targeted.extract_disagreement_crux) to impose the universal
# "no retry after 30s" deadline rule on the conditional-stacking critical path.
# Per-instance override so PARSER_LLM keeps its allowed_tries=3.
DISAGREEMENT_ANALYZER_LLM: GeneralLlm = build_llm_with_openrouter_fallback(
    "openrouter/openai/gpt-5.6-terra",
    reasoning={"effort": "low"},
    **{**UTILITY_MODEL_CONFIG, "allowed_tries": 1},
)
