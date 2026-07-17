from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

from litellm import acompletion

from metaculus_bot.constants import OAI_ANTH_OPENROUTER_KEY_ENV, OPENROUTER_API_KEY_ENV
from metaculus_bot.fallback_openrouter import should_retry_with_general_key, should_route_via_donated_key
from metaculus_bot.research.agentic.types import LoopConfig

LlmCall = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None], Awaitable[Any]]


def build_default_llm_call(config: LoopConfig) -> LlmCall:
    model = config.model if config.model.startswith("openrouter/") else f"openrouter/{config.model}"
    donated_key = os.getenv(OAI_ANTH_OPENROUTER_KEY_ENV)
    personal_key = os.getenv(OPENROUTER_API_KEY_ENV)
    use_fallback = should_route_via_donated_key(model) and donated_key and personal_key and donated_key != personal_key

    async def _call_once(
        messages: list[dict[str, Any]], tools_json: list[dict[str, Any]] | None, api_key: str | None
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            # Shallow copy: litellm may mutate the caller's list in place in some
            # code paths; copying the container preserves the loop's append-only
            # prefix (dict identity is kept — providers cache on it).
            "messages": list(messages),
            "parallel_tool_calls": True,
            "reasoning_effort": config.reasoning_effort,
            "temperature": None,
        }
        if tools_json is not None:
            kwargs["tools"] = tools_json
        if api_key:
            kwargs["api_key"] = api_key
        return await acompletion(**kwargs)

    async def _call(messages: list[dict[str, Any]], tools_json: list[dict[str, Any]] | None) -> Any:
        if use_fallback:
            assert donated_key is not None
            assert personal_key is not None
            try:
                return await _call_once(messages, tools_json, donated_key)
            except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # classifier re-raises non-key-scoped errors
                if not should_retry_with_general_key(exc):
                    raise
                return await _call_once(messages, tools_json, personal_key)

        api_key = donated_key if should_route_via_donated_key(model) and donated_key else personal_key
        # TODO(unify-fallback-routing): share a raw-litellm retry wrapper with fallback_openrouter.py if this grows.
        return await _call_once(messages, tools_json, api_key)

    return _call
