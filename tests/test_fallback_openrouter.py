from unittest.mock import AsyncMock

import pytest

from metaculus_bot.fallback_openrouter import (
    DONATED_KEY_PROVIDERS,
    FallbackOpenRouterLlm,
    build_llm_with_openrouter_fallback,
    should_retry_with_general_key,
    should_route_via_donated_key,
)


class TestPredicates:
    def test_should_route_via_donated_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # OpenAI / Anthropic always route via the donated key when one is configured.
        assert should_route_via_donated_key("openrouter/openai/gpt-5.1") is True
        assert should_route_via_donated_key("openrouter/anthropic/claude-sonnet-4") is True
        # Google is gated on GEMINI_USE_DONATED_OPENROUTER_KEY. With the toggle on,
        # flash models prefer the donated key — but gemini-3.1-pro is on the
        # DONATED_KEY_BLOCKED_GOOGLE_MODELS blocklist (free-tier BYOK → 429), so it
        # is pinned to the personal key even with the toggle ON.
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")
        assert should_route_via_donated_key("openrouter/google/gemini-3.5-flash") is True
        assert should_route_via_donated_key("openrouter/google/gemini-3.1-flash-lite") is True
        assert should_route_via_donated_key("openrouter/google/gemini-3-flash-preview") is True
        assert should_route_via_donated_key("openrouter/google/gemini-3.1-pro-preview") is False
        # Explicit toggle off: ALL Google calls go through the operator's personal key only.
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "false")
        assert should_route_via_donated_key("openrouter/google/gemini-3.1-pro-preview") is False
        assert should_route_via_donated_key("openrouter/google/gemini-3.5-flash") is False
        # Default with the env var unset is ON (donated key with personal fallback):
        # after Metaculus raised the Google rate limits (2026-06-16) the donated key
        # serves most Gemini. A flash model routes donated-first by default; the
        # gemini-3.1-pro slug stays blocklisted (pinned to personal) regardless.
        monkeypatch.delenv("GEMINI_USE_DONATED_OPENROUTER_KEY", raising=False)
        assert should_route_via_donated_key("openrouter/google/gemini-3.5-flash") is True
        assert should_route_via_donated_key("openrouter/google/gemini-3.1-pro-preview") is False
        # Providers NOT covered by the donated key.
        assert should_route_via_donated_key("openrouter/x-ai/grok-4.1-fast") is False
        assert should_route_via_donated_key("openrouter/qwen/qwen3-235b") is False
        # Non-OpenRouter slugs.
        assert should_route_via_donated_key("perplexity/sonar") is False
        # Defensive: bogus inputs return False rather than raising.
        assert should_route_via_donated_key("openrouter/") is False  # parts < 2
        assert should_route_via_donated_key("") is False

    def test_donated_key_providers_set(self) -> None:
        # Pin the membership so any drift surfaces in code review rather than
        # silently changing routing.
        assert DONATED_KEY_PROVIDERS == frozenset({"openai", "anthropic", "google"})

    @pytest.mark.parametrize(
        "message, expected",
        [
            ("HTTP 402 Payment Required", True),
            ("payment required", True),
            ("insufficient credit on key", True),
            ("401 Unauthorized", True),
            ("invalid API key", True),
            ("disabled api key", True),
            # Donated-key allowed-providers 404 quirk → should fall back.
            ("404 No allowed providers are available for the selected model.", True),
            ("no allowed providers", True),
            # Donated-key data-policy / guardrail 404 (added 2026-05-17 for OpenAI
            # native search migration) — donated key's data-collection guardrail
            # blocks OpenAI native search; the personal key has no such block.
            (
                "404 No endpoints available matching your guardrail restrictions and data policy. "
                "Configure: https://openrouter.ai/settings/privacy",
                True,
            ),
            ("matching your guardrail restrictions", True),
            ("data policy", True),
            # 429 rate-limit: textual detection falls back (BYOK quotas are per-key).
            ("429 Too Many Requests", True),
            ("Rate limit exceeded", True),
            # Belt-and-suspenders textual patterns for 429 edge cases.
            ('{"code":429, "message": "rate limited"}', True),
            ("rate-limited upstream by provider", True),
            # Negative: moderation / infrastructure errors do NOT fall back.
            ("403 Forbidden moderation", False),
            ("502 Bad Gateway", False),
            ("503 Service Unavailable", False),
        ],
    )
    def test_should_retry_with_general_key(self, message: str, expected: bool) -> None:
        assert should_retry_with_general_key(Exception(message)) is expected

    def test_litellm_rate_limit_error_triggers_fallback(self) -> None:
        """litellm.RateLimitError (typed 429) triggers fallback — BYOK quotas are independent."""
        import litellm

        exc = litellm.RateLimitError(
            message="Rate limit exceeded on openrouter",
            model="openrouter/google/gemini-3.1-pro-preview",
            llm_provider="openrouter",
        )
        assert should_retry_with_general_key(exc) is True

    def test_litellm_service_unavailable_does_not_trigger_fallback(self) -> None:
        """litellm.ServiceUnavailableError (503) does NOT trigger fallback — infrastructure issue."""
        import litellm

        exc = litellm.ServiceUnavailableError(
            message="503 Service Unavailable",
            model="openrouter/openai/gpt-5.1",
            llm_provider="openrouter",
        )
        assert should_retry_with_general_key(exc) is False


class TestFallbackOpenRouterLlm:
    @pytest.mark.asyncio
    async def test_primary_success_no_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.1",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        # Patch the internal primary call point to avoid network.
        monkeypatch.setattr(llm, "_invoke_once_using_primary", AsyncMock(return_value="answer"))

        out = await llm.invoke("hi")
        assert out == "answer"

    @pytest.mark.asyncio
    async def test_fallback_on_402(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/anthropic/claude-sonnet-4",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("HTTP 402 Payment Required: insufficient credit")),
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        out = await llm.invoke("hi")
        assert out == "ok"

    @pytest.mark.asyncio
    async def test_fallback_on_429_rate_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """429 on primary key falls back to secondary — BYOK quotas are independent."""
        import litellm

        llm = FallbackOpenRouterLlm(
            model="openrouter/google/gemini-3.1-pro-preview",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        exc = litellm.RateLimitError(
            message="Rate limit exceeded",
            model="openrouter/google/gemini-3.1-pro-preview",
            llm_provider="openrouter",
        )
        monkeypatch.setattr(llm, "_invoke_once_using_primary", AsyncMock(side_effect=exc))
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="fallback_ok"))

        out = await llm.invoke("hi")
        assert out == "fallback_ok"

    @pytest.mark.asyncio
    async def test_no_fallback_on_403(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.1",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("403 Forbidden moderation")),
        )

        with pytest.raises(Exception):
            await llm.invoke("hi")

    @pytest.mark.asyncio
    async def test_no_fallback_on_503(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """503 Service Unavailable re-raises without fallback — infrastructure issue, not key-scoped."""
        import litellm

        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.1",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )

        exc = litellm.ServiceUnavailableError(
            message="503 Service Unavailable",
            model="openrouter/openai/gpt-5.1",
            llm_provider="openrouter",
        )
        monkeypatch.setattr(llm, "_invoke_once_using_primary", AsyncMock(side_effect=exc))

        with pytest.raises(litellm.ServiceUnavailableError):
            await llm.invoke("hi")

    @pytest.mark.asyncio
    async def test_no_secondary_key_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.1",
            primary_api_key="special",
            secondary_api_key=None,
            temperature=0,
        )

        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("401 Unauthorized")),
        )

        with pytest.raises(Exception):
            await llm.invoke("hi")


class TestFallbackCounters:
    """Every donated->personal fallback must be counted + logged loudly so silent
    personal-key spend can't accumulate. ``_generic_key_fallback_count`` counts ALL
    fallback causes; ``_donated_404_fallback_count`` is the allowed-providers-404
    subset. cli.py folds the generic counter into its non-zero-exit alert.
    """

    def setup_method(self) -> None:
        from metaculus_bot.fallback_openrouter import (
            reset_donated_404_fallback_count,
            reset_generic_key_fallback_count,
        )

        reset_generic_key_fallback_count()
        reset_donated_404_fallback_count()

    def teardown_method(self) -> None:
        from metaculus_bot.fallback_openrouter import (
            reset_donated_404_fallback_count,
            reset_generic_key_fallback_count,
        )

        reset_generic_key_fallback_count()
        reset_donated_404_fallback_count()

    @pytest.mark.asyncio
    async def test_generic_fallback_bumps_counter_and_logs_every_time(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 401/402 (non-404) fallback bumps ONLY the generic counter, not the 404 subset,
        and logs a WARNING on EVERY fallback (no once-per-instance suppression)."""
        from metaculus_bot.fallback_openrouter import (
            get_donated_404_fallback_count,
            get_generic_key_fallback_count,
        )

        llm = FallbackOpenRouterLlm(
            model="openrouter/anthropic/claude-opus-4.8",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("HTTP 402 Payment Required: insufficient credit")),
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        import logging

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.fallback_openrouter"):
            assert await llm.invoke("hi") == "ok"
            assert await llm.invoke("hi") == "ok"

        # Generic counter bumped on both fallbacks; 404 subset untouched.
        assert get_generic_key_fallback_count() == 2
        assert get_donated_404_fallback_count() == 0
        # Loud WARNING on every fallback, not just the first.
        paid_warnings = [r for r in caplog.records if "PAID PERSONAL-KEY FALLBACK" in r.getMessage()]
        assert len(paid_warnings) == 2

    @pytest.mark.asyncio
    async def test_donated_404_fallback_bumps_both_counters(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A 404 'no allowed providers' fallback bumps BOTH the generic total and the 404 subset
        (a 404 fallback is still a personal-key fallback)."""
        from metaculus_bot.fallback_openrouter import (
            get_donated_404_fallback_count,
            get_generic_key_fallback_count,
        )

        llm = FallbackOpenRouterLlm(
            model="openrouter/openai/gpt-5.4",
            primary_api_key="special",
            secondary_api_key="general",
            temperature=0,
        )
        monkeypatch.setattr(
            llm,
            "_invoke_once_using_primary",
            AsyncMock(side_effect=Exception("404 No allowed providers are available for the selected model.")),
        )
        monkeypatch.setattr(llm, "_invoke_once_using_secondary", AsyncMock(return_value="ok"))

        import logging

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.fallback_openrouter"):
            assert await llm.invoke("hi") == "ok"

        assert get_generic_key_fallback_count() == 1
        assert get_donated_404_fallback_count() == 1
        assert any("no allowed providers" in r.getMessage() for r in caplog.records)


class TestBuilder:
    def test_builder_returns_wrapper_when_both_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5.1")
        assert isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_plain_when_only_general(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OAI_ANTH_OPENROUTER_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/openai/gpt-5.1")
        # Not wrapper, should be a GeneralLlm
        from forecasting_tools import GeneralLlm as GL

        assert isinstance(llm, GL)
        assert not isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_plain_for_non_donated_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Models from providers not in DONATED_KEY_PROVIDERS get a plain GeneralLlm
        (no donated-key wrapping). Grok via x-ai is the canonical example.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        llm = build_llm_with_openrouter_fallback("openrouter/x-ai/grok-4.1-fast")
        # Not wrapper, should be a GeneralLlm
        from forecasting_tools import GeneralLlm as GL

        assert isinstance(llm, GL)
        assert not isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_returns_wrapper_for_google_flash_when_donated_toggle_on(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flash Google models route via the donated wrapper when the toggle is ON.

        Originally added in task #12 (Google in DONATED_KEY_PROVIDERS). After the
        2026-06-16 rate-limit bump the donated key serves the flash models, so a
        flash slug returns a FallbackOpenRouterLlm (donated primary, personal
        fallback). gemini-3.1-pro is handled separately by the blocklist test.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")
        llm = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.5-flash")
        assert isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_plain_for_google_pro_blocklisted_when_toggle_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """gemini-3.1-pro is pinned to the personal key via the blocklist even with
        the donated toggle ON — so the builder returns a plain GeneralLlm (no
        donated attempt, no 429, no fallback-counter bump), while a flash model in
        the same env returns a FallbackOpenRouterLlm.

        Temporary workaround; see DONATED_KEY_BLOCKED_GOOGLE_MODELS
        (``TODO(gemini-3.1-pro-donated)``) and FUTURE.md.
        """
        from forecasting_tools import GeneralLlm as GL

        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "true")
        pro = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.1-pro-preview")
        assert isinstance(pro, GL)
        assert not isinstance(pro, FallbackOpenRouterLlm)
        flash = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.5-flash")
        assert isinstance(flash, FallbackOpenRouterLlm)

    def test_builder_plain_for_google_when_donated_toggle_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the donated-key toggle off, ALL Google calls bypass the donated
        wrapper entirely — the resulting LLM is a plain GeneralLlm using the
        operator's general OpenRouter key.
        """
        from forecasting_tools import GeneralLlm as GL

        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        monkeypatch.setenv("GEMINI_USE_DONATED_OPENROUTER_KEY", "false")
        llm = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.5-flash")
        assert isinstance(llm, GL)
        assert not isinstance(llm, FallbackOpenRouterLlm)

    def test_builder_returns_wrapper_for_google_flash_when_toggle_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default (env var unset) is ON: flash Google calls prefer the donated wrapper.

        After Metaculus raised the Google rate limits (2026-06-16) the donated key
        serves most Gemini, so with two distinct keys configured a flash slug
        returns a FallbackOpenRouterLlm (donated primary, personal fallback).
        gemini-3.1-pro stays blocklisted (plain GeneralLlm) — see the dedicated
        blocklist test.
        """
        monkeypatch.setenv("OAI_ANTH_OPENROUTER_KEY", "special")
        monkeypatch.setenv("OPENROUTER_API_KEY", "general")
        monkeypatch.delenv("GEMINI_USE_DONATED_OPENROUTER_KEY", raising=False)
        llm = build_llm_with_openrouter_fallback("openrouter/google/gemini-3.5-flash")
        assert isinstance(llm, FallbackOpenRouterLlm)


class TestDeprecationTripwire:
    """Post-submission deprecation tripwire (added 2026-05-17 after the
    2026-05-15 x-ai/grok-4.1-fast deprecation silently 404'd for ~2 days).
    """

    def test_records_deprecation_404(self) -> None:
        """A canonical OpenRouter deprecation 404 is recorded with model + msg."""
        from metaculus_bot.fallback_openrouter import (
            _DEPRECATION_ALERTS,
            _record_deprecation_if_matched,
            clear_deprecation_alerts,
        )

        clear_deprecation_alerts()
        msg = "Grok 4.1 Fast is deprecated. xAI recommends switching to Grok 4.3"
        matched = _record_deprecation_if_matched("x-ai/grok-4.1-fast", msg)
        assert matched is True
        assert _DEPRECATION_ALERTS == [("x-ai/grok-4.1-fast", msg)]
        clear_deprecation_alerts()

    def test_does_not_record_unrelated_error(self) -> None:
        """Plain 401/429/etc. don't match — false positives turn CI red without cause."""
        from metaculus_bot.fallback_openrouter import (
            _DEPRECATION_ALERTS,
            _record_deprecation_if_matched,
            clear_deprecation_alerts,
        )

        clear_deprecation_alerts()
        for msg in (
            "401 Unauthorized: invalid api key",
            "429 Too Many Requests: rate limit exceeded",
            "402 Payment Required: insufficient credit",
            "404 No allowed providers",
            "503 Service Unavailable",
        ):
            matched = _record_deprecation_if_matched("openrouter/openai/gpt-5.5", msg)
            assert matched is False, f"falsely matched: {msg}"
        assert _DEPRECATION_ALERTS == []
        clear_deprecation_alerts()

    def test_check_exits_on_populated_list(self) -> None:
        """Tripwire fires sys.exit(1) when at least one deprecation was seen."""
        from metaculus_bot.fallback_openrouter import (
            _record_deprecation_if_matched,
            check_deprecation_alerts_and_exit,
            clear_deprecation_alerts,
        )

        clear_deprecation_alerts()
        _record_deprecation_if_matched("x-ai/grok-4.1-fast", "Grok 4.1 Fast is deprecated.")
        with pytest.raises(SystemExit) as exc_info:
            check_deprecation_alerts_and_exit()
        assert exc_info.value.code == 1
        clear_deprecation_alerts()

    def test_check_returns_silently_on_empty(self) -> None:
        """No deprecation observed → tripwire is a no-op (run completes cleanly)."""
        from metaculus_bot.fallback_openrouter import (
            check_deprecation_alerts_and_exit,
            clear_deprecation_alerts,
        )

        clear_deprecation_alerts()
        # Must not raise SystemExit
        check_deprecation_alerts_and_exit()
