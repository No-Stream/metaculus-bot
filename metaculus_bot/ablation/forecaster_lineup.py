"""Free-model forecaster lineup for the probabilistic-tools ablation benchmark.

The ablation benchmark needs cheap forecasters: we run the SAME N forecasters
once per question against pre-cached Gemini research, then feed those rationales
into BOTH stacker arms (tools-on, tools-off) — so the per-forecaster cost is
amortized across two arms. We pick free OpenRouter models so a 100-question
medium backtest doesn't burn budget on the forecaster stage.

**Routing posture**: ablation explicitly opts out of the donated-OpenRouter-key
path. ``build_free_forecaster_llms`` and ``build_free_parser_llm`` construct
plain ``GeneralLlm`` instances with no ``api_key`` set, so litellm picks up
the operator's paid ``OPENROUTER_API_KEY`` from env at invoke time. Reasons:

1. **Resource accounting.** The donated key is for production
   ensemble + stacker work; ablation runs are exploratory work charged to the
   operator. Routing free-tier traffic through the donated key would
   silently consume that quota.
2. **The donated-key allowed-providers list trips on free-tier providers.**
   Many ``:free`` model variants are served only by OpenInference/Venice/etc.
   The donated key returns 404 (we'd burn time on transient failures + the
   alerting non-zero exit) for any model whose only provider isn't in
   ``DONATED_KEY_PROVIDERS``. Plain GeneralLlm sidesteps this entirely.
3. **Cleaner first-light**: no fallback wrapping means failures are
   directly diagnostic — what you see is what hit OpenRouter.

Edit ``FREE_FORECASTER_MODELS`` to swap the lineup.
"""

from __future__ import annotations

from forecasting_tools import GeneralLlm

from metaculus_bot.benchmark.bot_factory import MODEL_CONFIG
from metaculus_bot.llm_configs import DETERMINISTIC_MODEL_CONFIG

__all__ = [
    "FREE_FORECASTER_MODELS",
    "FREE_PARSER_MODEL",
    "build_free_forecaster_llms",
    "build_free_parser_llm",
]

# Lineup history:
# * ``openrouter/openai/gpt-oss-120b:free`` was originally in this list but
#   routes through the donated-key wrapper (because it's OpenAI-prefixed),
#   which 404s on the donated key's allowed-providers list. The donated key
#   is intentionally NOT used in the ablation pipeline (see module docstring),
#   but even if we forced plain ``GeneralLlm`` for it, the served-by provider
#   (``open-inference``) is rate-limited enough that it'd be a low-utility
#   slot. Removed in task #16.
# * ``openrouter/z-ai/glm-4.5-air:free`` removed 2026-05-14 (Phase A.3 Package
#   3b) after qid 43171: GLM hallucinated TSA partial-week data and emitted a
#   "normal" distribution with σ=13K vs ensemble median σ ~965K (1.3% of
#   ensemble σ). Arm B's stacker over-weighted GLM and saturated the schema
#   floor (-220 log score). User signed off post-Phase-A.2.
# * ``qwen3-next-80b-a3b-instruct:free`` is retained despite chronic Venice
#   upstream rate-limiting — that's what the ``patient`` rate-limit-mode (CLI
#   default) is for. Dropping qwen would put us at 3 free models, which is
#   below the noise floor for ensemble diversity at 50q scale.
FREE_FORECASTER_MODELS: list[str] = [
    "openrouter/minimax/minimax-m2.5:free",
    "openrouter/google/gemma-4-26b-a4b-it:free",
    "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
    "openrouter/qwen/qwen3-next-80b-a3b-instruct:free",
]

# Parser stays on a Google-served free model so the donated-key wrapper
# doesn't apply at all (and the gemini/gemma family is reliable for
# structured-output extraction).
#
# Bake-off 2026-05-14 (task #18 / Bucket 2) picked gemma-4-31b-it over
# gemma-4-26b-a4b-it. Tested 8 free models on a real failing rationale that
# emitted nonstandard percentiles (0.1, 1, 25, 75, 99, 99.9 instead of the
# requested 2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5):
#
#   - gemma-4-26b-a4b-it (incumbent): emitted `null` for percentiles not
#     literally present in the text — over-applied the "do not guess"
#     instruction. Same failure across multiple trials.
#   - nemotron-3-super-120b, deepseek-v4-flash: returned the
#     "<<REQUESTED TYPE WAS NOT FOUND IN TEXT>>" sentinel — over-applied
#     the "if unrelated" instruction.
#   - qwen3-next-80b, llama-3.3-70b, hermes-3-405b: chronically rate-limited
#     upstream (Venice / etc free-tier providers).
#   - glm-4.5-air: 120s timeout consistently.
#   - gemma-4-31b-it: 3/3 PASS, ~10s, deterministic output, all 11 percentiles
#     interpolated correctly, within bounds. Minor: emits some adjacent
#     duplicate values (P40=P50, P80=P90), which the prod numeric_pipeline
#     handles via apply_jitter_for_duplicates.
FREE_PARSER_MODEL: str = "openrouter/google/gemma-4-31b-it:free"


def build_free_forecaster_llms() -> list[GeneralLlm]:
    """Construct plain ``GeneralLlm`` instances for each free forecaster model.

    Plain (no ``api_key`` arg) so litellm reads ``OPENROUTER_API_KEY`` from
    env at invoke time — the operator's paid key, not the donated one. See
    module docstring for the routing rationale.
    """
    return [GeneralLlm(model=model, **MODEL_CONFIG) for model in FREE_FORECASTER_MODELS]


def build_free_parser_llm() -> GeneralLlm:
    """Plain ``GeneralLlm`` parser at deterministic config.

    Mirrors the production PARSER_LLM contract — low temperature, deterministic
    output — but uses a free model so backtest runs don't burn paid-parser
    budget. Plain (no ``api_key``) so litellm picks up ``OPENROUTER_API_KEY``
    from env at invoke time.
    """
    return GeneralLlm(model=FREE_PARSER_MODEL, **DETERMINISTIC_MODEL_CONFIG)
