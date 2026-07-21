"""Run the research-role model audit arms (Zambia Q44229, fixed inputs).

Three NON-driver research roles, three candidate models each (9 paid calls):

- ``summarizer``    — the real ``asknews_summarizer_prompt`` over the frozen
  AskNews article dump (``inputs/asknews_raw.md``).
- ``native_search`` — the real ``web_research_prompt`` through
  ``build_native_search_llm`` (live OpenAI native web search, donated key).
- ``crux``          — the real ``disagreement_crux_prompt`` over the six
  forecaster rationales recovered from the 2026-07-17 smoke run.

Candidates (per role): gpt-5.6-sol @ low (incumbent), gpt-5.6-terra @ low
(cost candidate), gpt-5.6-luna @ medium (completeness candidate).

All arms use the production plumbing: ``build_llm_with_openrouter_fallback``
with the same UTILITY_MODEL_CONFIG-shaped kwargs the prod slots use
(temperature=None, stream=False, allowed_tries=1), and
``build_native_search_llm`` with an explicit reasoning_effort override for the
search role. Token usage comes from GeneralLlm's TextTokenCostResponse (the
object ``invoke`` unwraps), so counts are the provider-reported ones.

Usage:
    uv run python scratch/research_role_audit_2026-07-17/run_arms.py [role ...]

With no args runs all three roles. Arms within a role run in parallel; roles
run sequentially. Idempotent: an arm whose output file already exists is
skipped, so a partial failure can be resumed without double-spending.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from metaculus_bot.config import load_environment

load_environment()

from forecasting_tools import GeneralLlm  # noqa: E402

from metaculus_bot.fallback_openrouter import build_llm_with_openrouter_fallback  # noqa: E402
from metaculus_bot.prompts import (  # noqa: E402
    asknews_summarizer_prompt,
    disagreement_crux_prompt,
    web_research_prompt,
)
from metaculus_bot.research.providers import build_native_search_llm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger("research_role_audit")

BASE_DIR = Path(__file__).parent
INPUTS_DIR = BASE_DIR / "inputs"
ARMS_DIR = BASE_DIR / "arms"

# (candidate name, OpenAI model tier slug, reasoning effort)
CANDIDATES: list[tuple[str, str, str]] = [
    ("sol_low", "gpt-5.6-sol", "low"),
    ("terra_low", "gpt-5.6-terra", "low"),
    ("luna_medium", "gpt-5.6-luna", "medium"),
]

# Per-M-token USD pricing pulled from the OpenRouter /models endpoint 2026-07-17.
# Recorded here (not fetched at runtime) so cost rows are reproducible; the
# native-search web-tool fee is billed opaquely per request and reported
# separately in RESULTS.md.
PRICING_PER_M: dict[str, dict[str, float]] = {
    "gpt-5.6-sol": {"prompt": 5.0, "completion": 30.0},
    "gpt-5.6-terra": {"prompt": 2.5, "completion": 15.0},
    "gpt-5.6-luna": {"prompt": 1.0, "completion": 6.0},
}

ROLES = ("summarizer", "native_search", "crux")


def _load_inputs() -> tuple[dict, str, list[str]]:
    meta = json.loads((INPUTS_DIR / "question_meta.json").read_text(encoding="utf-8"))
    asknews_raw = (INPUTS_DIR / "asknews_raw.md").read_text(encoding="utf-8")
    crux_payload = json.loads((INPUTS_DIR / "crux_base_texts.json").read_text(encoding="utf-8"))
    base_texts = [entry["reasoning"] for entry in crux_payload]
    return meta, asknews_raw, base_texts


def _build_role_prompt(role: str, meta: dict, asknews_raw: str, base_texts: list[str]) -> str:
    if role == "summarizer":
        return asknews_summarizer_prompt(
            question_text=meta["question_text"],
            resolution_criteria=meta["resolution_criteria"],
            fine_print=meta["fine_print"],
            open_date=meta["open_date"],
            research=asknews_raw,
        )
    if role == "native_search":
        return web_research_prompt(meta["question_text"], is_benchmarking=False, citation_style="markdown")
    if role == "crux":
        return disagreement_crux_prompt(meta["question_text"], base_texts)
    raise ValueError(f"unknown role: {role}")


def _build_role_llm(role: str, model_tier: str, effort: str) -> GeneralLlm:
    if role == "native_search":
        # Explicit reasoning_effort override wins over the env read; verbosity
        # stays the env/default LOW — identical across arms.
        return build_native_search_llm(f"openai/{model_tier}", reasoning_effort=effort)
    # Summarizer / crux both run on UTILITY_MODEL_CONFIG-shaped kwargs with the
    # prod per-instance allowed_tries=1 override (llm_configs.py).
    return build_llm_with_openrouter_fallback(
        f"openrouter/openai/{model_tier}",
        reasoning={"effort": effort},
        temperature=None,
        max_tokens=32_000,
        stream=False,
        timeout=300,
        allowed_tries=1,
    )


async def _invoke_with_usage(llm: GeneralLlm, prompt: str) -> tuple[str, dict[str, Any]]:
    """Invoke via the same internal path GeneralLlm.invoke uses, keeping usage.

    ``invoke`` unwraps TextTokenCostResponse to a str; calling the underscore
    method keeps provider-reported token counts for the cost table. NOTE: for
    FallbackOpenRouterLlm this bypasses the donated->personal fallback wrapper
    (primary/donated key only) — acceptable for a one-shot audit where a key
    failure should surface loudly, not silently bill the personal key.
    """
    response = await llm._invoke_with_request_cost_time_and_token_limits_and_retry(prompt)  # noqa: SLF001
    usage = {
        "prompt_tokens": response.prompt_tokens_used,
        "completion_tokens": response.completion_tokens_used,
        "total_tokens": response.total_tokens_used,
        "framework_cost": response.cost,
    }
    return response.data, usage


async def run_arm(role: str, name: str, model_tier: str, effort: str, prompt: str) -> None:
    arm_dir = ARMS_DIR / role
    arm_dir.mkdir(parents=True, exist_ok=True)
    out_path = arm_dir / f"{name}.md"
    meta_path = arm_dir / f"{name}.meta.json"
    if out_path.exists():
        logger.info("SKIP %s/%s — output exists", role, name)
        return

    llm = _build_role_llm(role, model_tier, effort)
    logger.info("ARM START %s/%s model=%s effort=%s", role, name, llm.model, effort)
    started = time.monotonic()
    output, usage = await _invoke_with_usage(llm, prompt)
    wall_s = time.monotonic() - started

    pricing = PRICING_PER_M[model_tier]
    est_cost = (
        usage["prompt_tokens"] * pricing["prompt"] + usage["completion_tokens"] * pricing["completion"]
    ) / 1_000_000

    out_path.write_text(output, encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "role": role,
                "arm": name,
                "model": llm.model,
                "reasoning_effort": effort,
                "wall_s": round(wall_s, 1),
                "output_chars": len(output),
                "usage": usage,
                "est_cost_usd_ex_webtool": round(est_cost, 4),
                "pricing_per_m": pricing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("ARM DONE %s/%s wall_s=%.1f chars=%d est_cost=$%.4f", role, name, wall_s, len(output), est_cost)


async def main() -> None:
    requested = sys.argv[1:] or list(ROLES)
    unknown = set(requested) - set(ROLES)
    assert not unknown, f"unknown roles: {unknown}; valid: {ROLES}"

    meta, asknews_raw, base_texts = _load_inputs()
    for role in requested:
        prompt = _build_role_prompt(role, meta, asknews_raw, base_texts)
        logger.info("ROLE %s: prompt is %d chars", role, len(prompt))
        await asyncio.gather(*(run_arm(role, name, tier, effort, prompt) for name, tier, effort in CANDIDATES))


if __name__ == "__main__":
    asyncio.run(main())
