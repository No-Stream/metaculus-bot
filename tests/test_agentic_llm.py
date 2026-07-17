"""Key-routing tests for the agentic loop's default LLM transport.

``build_default_llm_call`` composes the donated-vs-personal OpenRouter key
routing (prefix normalization, donated-key-first, classifier-gated fallback).
Every other agentic test injects ``llm_call=`` or patches the builder, so this
file is the only coverage of the billing-relevant glue itself. The routing
predicates (``should_route_via_donated_key`` / ``should_retry_with_general_key``)
are stubbed here — their own classification behavior is covered in
tests/test_fallback_openrouter.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from metaculus_bot.research.agentic import llm as agentic_llm
from metaculus_bot.research.agentic.types import LoopConfig

_DONATED = "sk-donated"
_PERSONAL = "sk-personal"


def _config(model: str = "openai/gpt-5.6-luna", effort: str = "medium") -> LoopConfig:
    return LoopConfig(model=model, reasoning_effort=effort)


def _messages() -> list[dict[str, Any]]:
    return [{"role": "system", "content": "sys"}, {"role": "user", "content": "brief"}]


def _last_kwargs(mock: AsyncMock) -> Any:
    assert mock.await_args is not None
    return mock.await_args.kwargs


@pytest.fixture()
def acompletion(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(agentic_llm, "acompletion", mock)
    return mock


def _set_keys(monkeypatch: pytest.MonkeyPatch, donated: str | None, personal: str | None) -> None:
    if donated is None:
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
    else:
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", donated)
    if personal is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", personal)


class TestKwargsPassthrough:
    @pytest.mark.asyncio
    async def test_prefix_added_and_call_kwargs(self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_keys(monkeypatch, donated=None, personal=_PERSONAL)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: False)
        call = agentic_llm.build_default_llm_call(_config(effort="high"))
        tools = [{"type": "function", "function": {"name": "fetch"}}]

        await call(_messages(), tools)

        kwargs = _last_kwargs(acompletion)
        assert kwargs["model"] == "openrouter/openai/gpt-5.6-luna"
        assert kwargs["parallel_tool_calls"] is True
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["temperature"] is None
        assert kwargs["tools"] == tools
        assert kwargs["api_key"] == _PERSONAL

    @pytest.mark.asyncio
    async def test_existing_prefix_not_doubled(self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch) -> None:
        _set_keys(monkeypatch, donated=None, personal=_PERSONAL)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: False)
        call = agentic_llm.build_default_llm_call(_config(model="openrouter/openai/gpt-5.6-luna"))

        await call(_messages(), None)

        assert _last_kwargs(acompletion)["model"] == "openrouter/openai/gpt-5.6-luna"
        assert "tools" not in _last_kwargs(acompletion)

    @pytest.mark.asyncio
    async def test_messages_list_is_shallow_copied(
        self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_keys(monkeypatch, donated=None, personal=_PERSONAL)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: False)
        call = agentic_llm.build_default_llm_call(_config())
        messages = _messages()

        await call(messages, None)

        sent = _last_kwargs(acompletion)["messages"]
        assert sent == messages
        assert sent is not messages  # container copied: litellm mutations can't corrupt the transcript
        assert sent[0] is messages[0]  # dict identity preserved: providers cache on it


class TestKeyRouting:
    @pytest.mark.asyncio
    async def test_donated_key_first_when_routing_allows(
        self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_keys(monkeypatch, donated=_DONATED, personal=_PERSONAL)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: True)

        call = agentic_llm.build_default_llm_call(_config())
        await call(_messages(), None)

        acompletion.assert_awaited_once()
        assert _last_kwargs(acompletion)["api_key"] == _DONATED

    @pytest.mark.asyncio
    async def test_fallback_to_personal_on_credential_error(
        self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_keys(monkeypatch, donated=_DONATED, personal=_PERSONAL)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: True)
        monkeypatch.setattr(agentic_llm, "should_retry_with_general_key", lambda exc: True)
        acompletion.side_effect = [RuntimeError("401 unauthorized: invalid api key"), {"ok": True}]

        call = agentic_llm.build_default_llm_call(_config())
        result = await call(_messages(), None)

        assert result == {"ok": True}
        assert acompletion.await_count == 2
        first, second = acompletion.await_args_list
        assert first.kwargs["api_key"] == _DONATED
        assert second.kwargs["api_key"] == _PERSONAL

    @pytest.mark.asyncio
    async def test_no_fallback_when_classifier_rejects(
        self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_keys(monkeypatch, donated=_DONATED, personal=_PERSONAL)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: True)
        monkeypatch.setattr(agentic_llm, "should_retry_with_general_key", lambda exc: False)
        acompletion.side_effect = RuntimeError("403 forbidden: moderation")

        call = agentic_llm.build_default_llm_call(_config())
        with pytest.raises(RuntimeError, match="moderation"):
            await call(_messages(), None)

        acompletion.assert_awaited_once()
        assert _last_kwargs(acompletion)["api_key"] == _DONATED

    @pytest.mark.asyncio
    async def test_identical_keys_disable_fallback(
        self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_keys(monkeypatch, donated=_PERSONAL, personal=_PERSONAL)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: True)
        monkeypatch.setattr(agentic_llm, "should_retry_with_general_key", lambda exc: True)
        acompletion.side_effect = RuntimeError("401 unauthorized")

        call = agentic_llm.build_default_llm_call(_config())
        with pytest.raises(RuntimeError, match="401"):
            await call(_messages(), None)

        acompletion.assert_awaited_once()  # same key twice would be pointless; the != guard skips the retry

    @pytest.mark.asyncio
    async def test_donated_only_uses_donated_without_fallback(
        self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_keys(monkeypatch, donated=_DONATED, personal=None)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: True)

        call = agentic_llm.build_default_llm_call(_config())
        await call(_messages(), None)

        acompletion.assert_awaited_once()
        assert _last_kwargs(acompletion)["api_key"] == _DONATED

    @pytest.mark.asyncio
    async def test_personal_only_when_routing_disallows_donated(
        self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_keys(monkeypatch, donated=_DONATED, personal=_PERSONAL)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: False)

        call = agentic_llm.build_default_llm_call(_config())
        await call(_messages(), None)

        acompletion.assert_awaited_once()
        assert _last_kwargs(acompletion)["api_key"] == _PERSONAL
