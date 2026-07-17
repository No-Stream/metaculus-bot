"""Driver-model replay harness for the gap-fill v2 agentic loop.

Replays the EXACT research task from the 2026-07-17 Zambia Q44229 smoke run
(question 44229, pre-gap-fill research bundle recovered from /tmp/v2-smoke.log)
through `run_agentic_loop` with different driver models, producing comparable
transcripts for a blinded judge.

The user brief is built ONCE and cached to disk (`user_brief.md`) so every arm
sees byte-identical input. The system prompt pins today="2026-07-17" (the
original smoke run's date). Tools are the real `build_gap_fill_tools` set
(AskNews / Exa / fetch / Gemini read_document) — live, paid, operator-approved.

Usage:
    uv run python scratch/driver_replay_2026-07-17/replay.py <model> <effort|default> <arm_name>

Examples:
    uv run python scratch/driver_replay_2026-07-17/replay.py openai/gpt-5.6-luna medium arm_luna_medium
    uv run python scratch/driver_replay_2026-07-17/replay.py anthropic/claude-sonnet-5 medium arm_sonnet5_medium
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from metaculus_bot.config import load_environment

load_environment()

import aiohttp  # noqa: E402
from forecasting_tools import MetaculusApi, NumericQuestion  # noqa: E402
from litellm import acompletion  # noqa: E402

from metaculus_bot.constants import OAI_ANTH_OPENROUTER_KEY_ENV, OPENROUTER_API_KEY_ENV  # noqa: E402
from metaculus_bot.fallback_openrouter import (  # noqa: E402
    should_retry_with_general_key,
    should_route_via_donated_key,
)
from metaculus_bot.research.agentic import LoopConfig, build_gap_fill_tools, run_agentic_loop  # noqa: E402
from metaculus_bot.research.agentic.driver_prompt import (  # noqa: E402
    build_ghost_prompt,
    build_system_prompt,
    build_user_brief,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logger = logging.getLogger("driver_replay")

BASE_DIR = Path(__file__).parent
QUESTION_URL = "https://www.metaculus.com/questions/44229/"
ORIGINAL_TODAY = "2026-07-17"  # date string from the original smoke run
BUNDLE_PATH = BASE_DIR / "bundle_pre_gap_fill.md"
BRIEF_PATH = BASE_DIR / "user_brief.md"

# Same values the original smoke run used (constants.py defaults, no env overrides).
MAX_TOOL_CALLS = 14
WALL_DEADLINE_S = 540.0
CONCLUDE_THRESHOLD_S = 90.0


def build_or_load_brief() -> str:
    """Build the user brief once and cache it; later arms reuse the exact bytes."""
    if BRIEF_PATH.exists():
        return BRIEF_PATH.read_text(encoding="utf-8")
    question = MetaculusApi.get_question_by_url(QUESTION_URL)
    assert isinstance(question, NumericQuestion), f"expected numeric question, got {type(question).__name__}"
    bundle = BUNDLE_PATH.read_text(encoding="utf-8")
    brief = build_user_brief(question, bundle)
    BRIEF_PATH.write_text(brief, encoding="utf-8")
    logger.info("Built and cached user brief (%d chars)", len(brief))
    return brief


def get_question_text() -> str:
    """Question topic for tool binding (read_document auto-escalation ask)."""
    meta_path = BASE_DIR / "question_text.txt"
    if meta_path.exists():
        return meta_path.read_text(encoding="utf-8")
    question = MetaculusApi.get_question_by_url(QUESTION_URL)
    meta_path.write_text(question.question_text, encoding="utf-8")
    return question.question_text


class TelemetryLlmCall:
    """Wraps an llm_call to record per-call usage and wall time.

    NOTE on reasoning_effort: production's build_default_llm_call passes
    ``reasoning_effort`` to litellm, which only works because forecasting_tools'
    GeneralLlm sets ``litellm.drop_params = True`` globally on first invoke — in
    the full bot process the param is silently DROPPED before reaching
    OpenRouter (prod bug, flagged separately). In this standalone harness we
    pass it through properly via ``allowed_openai_params`` so effort arms
    actually differ; OpenRouter supports ``reasoning_effort`` on all arm models
    (verified against its /models endpoint 2026-07-17).
    """

    def __init__(self, config: LoopConfig, effort: str | None) -> None:
        self._inner = _build_replay_llm_call(config, effort)
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, messages: list[dict[str, Any]], tools_json: list[dict[str, Any]] | None) -> Any:
        started = time.monotonic()
        response = await self._inner(messages, tools_json)
        elapsed = time.monotonic() - started
        usage = getattr(response, "usage", None)
        usage_dict: dict[str, Any] = {}
        if usage is not None:
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = getattr(usage, field, None)
                if value is not None:
                    usage_dict[field] = value
            details = getattr(usage, "completion_tokens_details", None)
            reasoning = getattr(details, "reasoning_tokens", None) if details is not None else None
            if reasoning is not None:
                usage_dict["reasoning_tokens"] = reasoning
        self.calls.append(
            {
                "wall_s": round(elapsed, 2),
                "n_messages": len(messages),
                "n_tools_offered": len(tools_json) if tools_json is not None else 0,
                "usage": usage_dict,
                "response_id": getattr(response, "id", None),
            }
        )
        return response


def _build_replay_llm_call(config: LoopConfig, effort: str | None):
    """Production build_default_llm_call's routing, with reasoning_effort actually delivered.

    litellm's OpenrouterConfig doesn't map ``reasoning_effort``, so we pass it
    through with ``allowed_openai_params`` (OpenRouter accepts the raw param).
    ``effort=None`` omits the param entirely (model default).
    """
    model = config.model if config.model.startswith("openrouter/") else f"openrouter/{config.model}"
    donated_key = os.getenv(OAI_ANTH_OPENROUTER_KEY_ENV)
    personal_key = os.getenv(OPENROUTER_API_KEY_ENV)
    use_fallback = should_route_via_donated_key(model) and donated_key and personal_key and donated_key != personal_key

    async def _call_once(
        messages: list[dict[str, Any]], tools_json: list[dict[str, Any]] | None, api_key: str | None
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "parallel_tool_calls": True,
            "temperature": None,
        }
        if effort is not None:
            kwargs["reasoning_effort"] = effort
            kwargs["allowed_openai_params"] = ["reasoning_effort"]
        if tools_json is not None:
            kwargs["tools"] = tools_json
        if api_key:
            kwargs["api_key"] = api_key
        return await acompletion(**kwargs)

    async def _call(messages: list[dict[str, Any]], tools_json: list[dict[str, Any]] | None) -> Any:
        if use_fallback:
            assert donated_key is not None and personal_key is not None
            try:
                return await _call_once(messages, tools_json, donated_key)
            except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # classifier re-raises non-key-scoped errors (mirrors production llm.py)
                if not should_retry_with_general_key(exc):
                    raise
                return await _call_once(messages, tools_json, personal_key)
        api_key = donated_key if should_route_via_donated_key(model) and donated_key else personal_key
        return await _call_once(messages, tools_json, api_key)

    return _call


async def fetch_openrouter_pricing(model: str) -> dict[str, float] | None:
    """Per-token prompt/completion USD pricing from the free OpenRouter models endpoint."""
    slug = model.removeprefix("openrouter/")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://openrouter.ai/api/v1/models", timeout=aiohttp.ClientTimeout(total=20)
            ) as resp:
                data = await resp.json()
        for entry in data.get("data", []):
            if entry.get("id") == slug:
                pricing = entry.get("pricing", {})
                return {
                    "prompt": float(pricing.get("prompt", 0)),
                    "completion": float(pricing.get("completion", 0)),
                }
    except (
        Exception
    ) as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # pricing is best-effort telemetry; never blocks the arm
        logger.warning("pricing fetch failed: %s", exc)
    return None


async def run_arm(model: str, effort: str | None, arm_name: str) -> None:
    arm_dir = BASE_DIR / arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)

    # Capture the loop's INFO lines (GAP_FILL_V2 / GHOST_FORECAST markers) per arm.
    file_handler = logging.FileHandler(arm_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    system_prompt = build_system_prompt(ORIGINAL_TODAY)
    user_brief = build_or_load_brief()
    question_text = get_question_text()
    tools = build_gap_fill_tools(question_text)

    config = LoopConfig(
        model=model,
        reasoning_effort=effort or "medium",  # LoopConfig requires a value; unused when effort is None
        max_tool_calls=MAX_TOOL_CALLS,
        wall_deadline_s=WALL_DEADLINE_S,
        conclude_threshold_s=CONCLUDE_THRESHOLD_S,
    )
    llm_call = TelemetryLlmCall(config, effort)

    logger.info("ARM START: %s model=%s effort=%s", arm_name, model, effort)
    started = time.monotonic()
    result = await run_agentic_loop(
        system_prompt,
        user_brief,
        tools,
        config,
        llm_call=llm_call,
        ghost_prompt=build_ghost_prompt(),
        log_prefix=f"arm={arm_name} ",
    )
    wall_s = time.monotonic() - started

    pricing = await fetch_openrouter_pricing(model)
    est_cost = None
    prompt_tokens = sum(c["usage"].get("prompt_tokens", 0) for c in llm_call.calls)
    completion_tokens = sum(c["usage"].get("completion_tokens", 0) for c in llm_call.calls)
    if pricing is not None:
        est_cost = prompt_tokens * pricing["prompt"] + completion_tokens * pricing["completion"]

    (arm_dir / "transcript.json").write_text(
        json.dumps(result.transcript, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (arm_dir / "findings.md").write_text(result.findings_markdown, encoding="utf-8")
    (arm_dir / "telemetry.json").write_text(
        json.dumps(dataclasses.asdict(result.telemetry), indent=2), encoding="utf-8"
    )
    ghost_payload = result.ghost.model_dump() if result.ghost is not None else None
    (arm_dir / "ghost.json").write_text(json.dumps(ghost_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (arm_dir / "llm_calls.json").write_text(json.dumps(llm_call.calls, indent=2), encoding="utf-8")
    meta = {
        "arm": arm_name,
        "model": model,
        "reasoning_effort": effort,
        "outer_wall_s": round(wall_s, 2),
        "loop_wall_s": result.telemetry.wall_s,
        "est_cost_usd": round(est_cost, 4) if est_cost is not None else None,
        "pricing_per_token": pricing,
        "prompt_tokens_total": prompt_tokens,
        "completion_tokens_total": completion_tokens,
        "n_llm_calls": len(llm_call.calls),
    }
    (arm_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "ARM DONE: %s wall_s=%.1f est_cost=%s findings=%d", arm_name, wall_s, est_cost, result.telemetry.findings_count
    )


def main() -> None:
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    model, effort_arg, arm_name = sys.argv[1], sys.argv[2], sys.argv[3]
    effort = None if effort_arg == "default" else effort_arg
    asyncio.run(run_arm(model, effort, arm_name))


if __name__ == "__main__":
    main()
