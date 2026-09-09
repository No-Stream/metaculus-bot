from __future__ import annotations

import os
from collections.abc import Awaitable
from typing import Any, Protocol

from litellm import acompletion

from metaculus_bot.constants import OAI_ANTH_OPENROUTER_KEY_ENV, OPENROUTER_API_KEY_ENV
from metaculus_bot.credit_telemetry import DONATED_KEY_ALIAS, PERSONAL_KEY_ALIAS, llm_call_metadata
from metaculus_bot.fallback_openrouter import (
    record_donated_key_fallback,
    should_retry_with_general_key,
    should_route_via_donated_key,
)
from metaculus_bot.research.agentic.types import LoopConfig


class LlmCall(Protocol):
    """One driver completion: the message list, the tool list to offer, and ``tool_choice``.

    ``tool_choice`` is forwarded as the API parameter of that name; ``None`` leaves the
    provider default. The ghost phase offers the research turns' tool list with
    ``tool_choice="none"`` so its request still matches the cached prompt prefix
    (docs/agentic_gap_fill.md "The ghost forecast").
    """

    def __call__(
        self,
        messages: list[dict[str, Any]],
        tools_json: list[dict[str, Any]] | None,
        /,
        *,
        tool_choice: str | None = None,
    ) -> Awaitable[Any]: ...


# The CREDIT_ROLE_SPEND line for the v2 driver's tool-loop completions.
GAP_FILL_V2_DRIVER_ROLE = "gap_fill_v2_driver"


def build_default_llm_call(config: LoopConfig) -> LlmCall:
    model = config.model if config.model.startswith("openrouter/") else f"openrouter/{config.model}"
    donated_key = os.getenv(OAI_ANTH_OPENROUTER_KEY_ENV)
    personal_key = os.getenv(OPENROUTER_API_KEY_ENV)
    use_fallback = should_route_via_donated_key(model) and donated_key and personal_key and donated_key != personal_key

    async def _call_once(
        messages: list[dict[str, Any]],
        tools_json: list[dict[str, Any]] | None,
        *,
        tool_choice: str | None,
        api_key: str | None,
        key_alias: str,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": model,
            # Shallow copy: litellm may mutate the caller's list, and dict identity must survive for caching.
            "messages": list(messages),
            # CREDIT_ROLE_SPEND tag, stamped per call because the alias names the key this attempt bills.
            "metadata": llm_call_metadata(GAP_FILL_V2_DRIVER_ROLE, key_alias),
            "parallel_tool_calls": True,
            "reasoning_effort": config.reasoning_effort,
            # Without this whitelist litellm drops reasoning_effort; see docs/agentic_gap_fill.md.
            "allowed_openai_params": ["reasoning_effort"],
            "temperature": None,
            # Private litellm kwarg skipping a proxy-only eager import; see docs/agentic_gap_fill.md.
            "_skip_mcp_handler": True,
        }
        if tools_json is not None:
            kwargs["tools"] = tools_json
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if api_key:
            kwargs["api_key"] = api_key
        return await acompletion(**kwargs)

    async def _call(
        messages: list[dict[str, Any]],
        tools_json: list[dict[str, Any]] | None,
        *,
        tool_choice: str | None = None,
    ) -> Any:
        if use_fallback:
            assert donated_key is not None
            assert personal_key is not None
            try:
                return await _call_once(
                    messages, tools_json, tool_choice=tool_choice, api_key=donated_key, key_alias=DONATED_KEY_ALIAS
                )
            except Exception as exc:  # HARNESS-SCAN-EXEMPT-broad-except  # classifier re-raises non-key-scoped errors
                if not should_retry_with_general_key(exc):
                    raise
                # Without this the bot's highest-volume donated-key path failed to the paid key silently.
                await record_donated_key_fallback(model, exc)
                return await _call_once(
                    messages, tools_json, tool_choice=tool_choice, api_key=personal_key, key_alias=PERSONAL_KEY_ALIAS
                )

        use_donated = bool(should_route_via_donated_key(model) and donated_key)
        api_key = donated_key if use_donated else personal_key
        # Fallback decision shared with fallback_openrouter; only the transport differs.
        key_alias = DONATED_KEY_ALIAS if use_donated else PERSONAL_KEY_ALIAS
        return await _call_once(messages, tools_json, tool_choice=tool_choice, api_key=api_key, key_alias=key_alias)

    return _call
