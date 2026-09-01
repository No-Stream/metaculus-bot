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

from metaculus_bot import fallback_openrouter
from metaculus_bot.credit_telemetry import DONATED_KEY_ALIAS, PERSONAL_KEY_ALIAS, llm_call_metadata
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


@pytest.fixture
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
        # litellm's OpenrouterConfig doesn't map reasoning_effort; without the
        # whitelist, litellm.drop_params=True (set globally by forecasting_tools)
        # silently strips it and drivers run at model-default effort.
        assert kwargs["allowed_openai_params"] == ["reasoning_effort"]
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
    async def test_credit_role_metadata_names_the_key_each_attempt_bills(
        self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The raw-acompletion transport stamps the CREDIT_ROLE_SPEND tag itself, per call,
        so a donated→personal fallback books its spend on the key that actually paid."""
        _set_keys(monkeypatch, donated=_DONATED, personal=_PERSONAL)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: True)
        monkeypatch.setattr(agentic_llm, "should_retry_with_general_key", lambda exc: True)
        acompletion.side_effect = [RuntimeError("401 unauthorized: invalid api key"), {"ok": True}]

        await agentic_llm.build_default_llm_call(_config())(_messages(), None)

        first, second = acompletion.await_args_list
        assert first.kwargs["metadata"] == llm_call_metadata(agentic_llm.GAP_FILL_V2_DRIVER_ROLE, DONATED_KEY_ALIAS)
        assert second.kwargs["metadata"] == llm_call_metadata(agentic_llm.GAP_FILL_V2_DRIVER_ROLE, PERSONAL_KEY_ALIAS)

    @pytest.mark.asyncio
    async def test_credit_role_metadata_on_the_single_key_path(
        self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _set_keys(monkeypatch, donated=None, personal=_PERSONAL)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: False)

        await agentic_llm.build_default_llm_call(_config())(_messages(), None)

        assert _last_kwargs(acompletion)["metadata"] == llm_call_metadata(
            agentic_llm.GAP_FILL_V2_DRIVER_ROLE, PERSONAL_KEY_ALIAS
        )

    @pytest.mark.asyncio
    async def test_fallback_is_counted_and_logged_like_the_wrapper(
        self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The v2 fallback must feed the same accounting as ``FallbackOpenRouterLlm``.

        gap-fill v2 runs on every question in all four prod workflows, so this
        hand-rolled donated→personal retry was the highest-volume uninstrumented
        personal-key spend path: no counter, no ``PAID PERSONAL-KEY FALLBACK`` WARN,
        no contribution to the end-of-run summary. After 2026-09-10 a v2-only
        fallback that should redden CI would silently not.
        """
        _set_keys(monkeypatch, donated=_DONATED, personal=_PERSONAL)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: True)
        monkeypatch.setattr(agentic_llm, "should_retry_with_general_key", lambda exc: True)
        acompletion.side_effect = [RuntimeError("401 unauthorized: invalid api key"), {"ok": True}]
        fallback_openrouter.reset_generic_key_fallback_count()
        fallback_openrouter.reset_credit_key_fallback_count()

        call = agentic_llm.build_default_llm_call(_config())
        with caplog.at_level("WARNING", logger="metaculus_bot.fallback_openrouter"):
            await call(_messages(), None)

        assert fallback_openrouter.get_generic_key_fallback_count() == 1
        # A 401 is not a credit shortfall, so the suppression subset stays empty:
        # generic adds once, at most one subset subtracts (CLAUDE.md invariant).
        assert fallback_openrouter.get_credit_key_fallback_count() == 0
        assert any("PAID PERSONAL-KEY FALLBACK" in message for message in caplog.messages)
        assert any("openrouter/openai/gpt-5.6-luna" in message for message in caplog.messages)

    @pytest.mark.asyncio
    async def test_credit_caused_fallback_counted_in_both_scalars_exactly_once(
        self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Credit-caused fallbacks are a strict SUBSET of the generic total: generic
        adds the event, the credit subset subtracts it back out during the suppression
        window. Double-counting either side breaks cli.py's alertable arithmetic."""
        _set_keys(monkeypatch, donated=_DONATED, personal=_PERSONAL)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: True)
        monkeypatch.setattr(agentic_llm, "should_retry_with_general_key", lambda exc: True)
        acompletion.side_effect = [RuntimeError("402 payment required: insufficient credits"), {"ok": True}]
        fallback_openrouter.reset_generic_key_fallback_count()
        fallback_openrouter.reset_credit_key_fallback_count()

        call = agentic_llm.build_default_llm_call(_config())
        await call(_messages(), None)

        assert fallback_openrouter.get_generic_key_fallback_count() == 1
        assert fallback_openrouter.get_credit_key_fallback_count() == 1

    @pytest.mark.asyncio
    async def test_no_fallback_means_no_counter_bump(
        self, acompletion: AsyncMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rejected fallback bills nothing to the personal key, so it must not count."""
        _set_keys(monkeypatch, donated=_DONATED, personal=_PERSONAL)
        monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: True)
        monkeypatch.setattr(agentic_llm, "should_retry_with_general_key", lambda exc: False)
        acompletion.side_effect = RuntimeError("403 forbidden: moderation")
        fallback_openrouter.reset_generic_key_fallback_count()

        call = agentic_llm.build_default_llm_call(_config())
        with pytest.raises(RuntimeError):
            await call(_messages(), None)

        assert fallback_openrouter.get_generic_key_fallback_count() == 0

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
