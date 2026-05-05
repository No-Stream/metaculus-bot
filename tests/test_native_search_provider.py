"""Tests for native search research provider."""

import asyncio
from unittest.mock import patch

import pytest


@pytest.mark.asyncio
async def test_native_search_provider_constructs_correct_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify native search provider constructs model name correctly."""
    monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "true")
    monkeypatch.setenv("NATIVE_SEARCH_MODEL", "x-ai/grok-4.1-fast")

    captured_model: str | None = None

    class MockLlm:
        def __init__(self, model: str, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal captured_model
            captured_model = model
            self.model = model

        async def invoke(self, prompt: str) -> str:
            return "Mock research response"

    with patch("metaculus_bot.research_providers.GeneralLlm", MockLlm):
        from metaculus_bot.research_providers import native_search_provider

        provider = native_search_provider()
        await provider("Will X happen?")

    assert captured_model == "openrouter/x-ai/grok-4.1-fast"


@pytest.mark.asyncio
async def test_native_search_provider_uses_custom_model_slug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify provider uses custom model slug when provided."""
    captured_model: str | None = None

    class MockLlm:
        def __init__(self, model: str, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal captured_model
            captured_model = model
            self.model = model

        async def invoke(self, prompt: str) -> str:
            return "Mock research response"

    with patch("metaculus_bot.research_providers.GeneralLlm", MockLlm):
        from metaculus_bot.research_providers import native_search_provider

        provider = native_search_provider(model_slug="openai/gpt-4o")
        await provider("Will X happen?")

    assert captured_model == "openrouter/openai/gpt-4o"


@pytest.mark.asyncio
async def test_native_search_provider_includes_prediction_markets_when_not_benchmarking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify prediction markets are included in prompt when not benchmarking."""
    captured_prompt: str | None = None

    class MockLlm:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.model = kwargs.get("model", "mock")

        async def invoke(self, prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Mock research response"

    with patch("metaculus_bot.research_providers.GeneralLlm", MockLlm):
        from metaculus_bot.research_providers import native_search_provider

        provider = native_search_provider(is_benchmarking=False)
        await provider("Will X happen?")

    assert captured_prompt is not None
    assert "Prediction market" in captured_prompt


@pytest.mark.asyncio
async def test_native_search_provider_excludes_prediction_markets_when_benchmarking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify prediction markets are excluded from prompt when benchmarking."""
    captured_prompt: str | None = None

    class MockLlm:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.model = kwargs.get("model", "mock")

        async def invoke(self, prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Mock research response"

    with patch("metaculus_bot.research_providers.GeneralLlm", MockLlm):
        from metaculus_bot.research_providers import native_search_provider

        provider = native_search_provider(is_benchmarking=True)
        await provider("Will X happen?")

    assert captured_prompt is not None
    assert "Prediction market" not in captured_prompt


@pytest.mark.asyncio
async def test_native_search_provider_prompt_includes_anti_hallucination_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify prompt includes anti-hallucination instructions."""
    captured_prompt: str | None = None

    class MockLlm:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.model = kwargs.get("model", "mock")

        async def invoke(self, prompt: str) -> str:
            nonlocal captured_prompt
            captured_prompt = prompt
            return "Mock research response"

    with patch("metaculus_bot.research_providers.GeneralLlm", MockLlm):
        from metaculus_bot.research_providers import native_search_provider

        provider = native_search_provider()
        await provider("Will X happen?")

    assert captured_prompt is not None
    assert "DO NOT hallucinate" in captured_prompt
    assert "only cite what you actually found" in captured_prompt


class TestParallelProviderSelection:
    """Tests for parallel provider selection in main.py."""

    def test_select_research_providers_returns_primary_only_when_native_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify only primary provider returned when native search disabled."""
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")
        monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        # Mock the TemplateForecaster minimally
        from main import TemplateForecaster

        with patch.object(TemplateForecaster, "__init__", lambda self: None):
            bot = TemplateForecaster.__new__(TemplateForecaster)
            bot._custom_research_provider = None
            bot.is_benchmarking = False
            bot.allow_research_fallback = True

            # Mock _select_research_provider to return a known provider
            async def mock_provider(q: str) -> str:
                return "primary research"

            with patch.object(bot, "_select_research_provider", return_value=(mock_provider, "asknews")):
                providers = bot._select_research_providers()

        assert len(providers) == 1
        assert providers[0][1] == "asknews"

    def test_select_research_providers_includes_native_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify native search provider added when enabled."""
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "true")
        monkeypatch.setenv("NATIVE_SEARCH_MODEL", "x-ai/grok-4.1-fast")
        monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from main import TemplateForecaster

        with patch.object(TemplateForecaster, "__init__", lambda self: None):
            bot = TemplateForecaster.__new__(TemplateForecaster)
            bot._custom_research_provider = None
            bot.is_benchmarking = False
            bot.allow_research_fallback = True

            async def mock_provider(q: str) -> str:
                return "primary research"

            with patch.object(bot, "_select_research_provider", return_value=(mock_provider, "asknews")):
                providers = bot._select_research_providers()

        assert len(providers) == 2
        provider_names = [name for _, name in providers]
        assert "asknews" in provider_names
        assert "native_search" in provider_names


class TestParallelExecution:
    """Tests for parallel provider execution."""

    @pytest.mark.asyncio
    async def test_run_providers_parallel_combines_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify parallel execution combines results from multiple providers."""
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")

        from main import TemplateForecaster

        with patch.object(TemplateForecaster, "__init__", lambda self: None):
            bot = TemplateForecaster.__new__(TemplateForecaster)
            bot.allow_research_fallback = False

            async def provider1(q: str) -> str:
                return "Research from provider 1"

            async def provider2(q: str) -> str:
                return "Research from provider 2"

            providers = [(provider1, "asknews"), (provider2, "native_search")]
            result = await bot._run_providers_parallel("Test question", providers)

        assert "Research from provider 1" in result
        assert "Research from provider 2" in result
        assert "## News Articles (AskNews)" in result
        assert "## Web Research (Native Search)" in result

    @pytest.mark.asyncio
    async def test_run_providers_parallel_handles_provider_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify parallel execution handles individual provider failures gracefully."""
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")

        from main import TemplateForecaster

        with patch.object(TemplateForecaster, "__init__", lambda self: None):
            bot = TemplateForecaster.__new__(TemplateForecaster)
            bot.allow_research_fallback = False
            # __init__ was bypassed; set the counter attribute that
            # _run_providers_parallel now increments on non-AskNews-403
            # provider failures.
            bot._research_provider_timeout_count = 0

            async def failing_provider(q: str) -> str:
                raise RuntimeError("Provider failed")

            async def working_provider(q: str) -> str:
                return "Research from working provider"

            providers = [(failing_provider, "failing"), (working_provider, "working")]
            result = await bot._run_providers_parallel("Test question", providers)

        # Should still have result from working provider
        assert "Research from working provider" in result
        # Should not crash or include empty result from failed provider

    @pytest.mark.asyncio
    async def test_run_providers_parallel_runs_concurrently(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify providers actually run in parallel, not sequentially."""
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")

        from main import TemplateForecaster

        execution_order: list[str] = []
        completion_order: list[str] = []

        with patch.object(TemplateForecaster, "__init__", lambda self: None):
            bot = TemplateForecaster.__new__(TemplateForecaster)
            bot.allow_research_fallback = False

            async def slow_provider(q: str) -> str:
                execution_order.append("slow_start")
                await asyncio.sleep(0.1)
                completion_order.append("slow")
                return "Slow result"

            async def fast_provider(q: str) -> str:
                execution_order.append("fast_start")
                await asyncio.sleep(0.01)
                completion_order.append("fast")
                return "Fast result"

            # Slow provider listed first, but fast should complete first if parallel
            providers = [(slow_provider, "slow"), (fast_provider, "fast")]
            await bot._run_providers_parallel("Test question", providers)

        # Both should start before either completes (parallel execution)
        assert execution_order == ["slow_start", "fast_start"]
        # Fast should complete before slow
        assert completion_order == ["fast", "slow"]


class TestAskNewsSubscriptionErrorHandling:
    """F10: AskNews off-season 403 vs operational failure branching in
    _run_providers_parallel.

    The provider loop must silence only the exact subscription-inactive
    signature (class-name + message match) — every other asknews failure must
    bump _research_provider_timeout_count so CI surfaces it.
    """

    @pytest.mark.asyncio
    async def test_asknews_subscription_inactive_not_counted_and_info_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """AskNews 403011 → counter stays 0, INFO log emitted, no WARNING."""
        import logging

        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")

        from main import TemplateForecaster

        # Local class named "ForbiddenError" so type(exc).__name__ matches the
        # SDK's real class name that is_asknews_subscription_error looks for.
        class ForbiddenError(Exception):
            pass

        with patch.object(TemplateForecaster, "__init__", lambda self: None):
            bot = TemplateForecaster.__new__(TemplateForecaster)
            bot.allow_research_fallback = False
            bot._research_provider_timeout_count = 0

            async def asknews_provider(q: str) -> str:
                raise ForbiddenError("403011 - subscription is not currently active")

            with caplog.at_level(logging.INFO, logger="main"):
                result = await bot._run_providers_parallel("test question", [(asknews_provider, "asknews")])

        assert result == "", "Failed provider yields empty result."
        assert bot._research_provider_timeout_count == 0, (
            "Off-season subscription-inactive must NOT count as an alertable failure."
        )
        # Matching INFO-level log (no WARNING) should be present.
        info_records = [r for r in caplog.records if r.levelno == logging.INFO and "asknews" in r.getMessage().lower()]
        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING and "asknews" in r.getMessage().lower()
        ]
        assert any("inactive" in r.getMessage().lower() or "off-season" in r.getMessage().lower() for r in info_records)
        assert warning_records == [], "Subscription-inactive must not log at WARNING."

    @pytest.mark.asyncio
    async def test_asknews_non_subscription_error_counted_and_warning_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """AskNews non-subscription failure → counter bumped, WARNING logged."""
        import logging

        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")

        from main import TemplateForecaster

        with patch.object(TemplateForecaster, "__init__", lambda self: None):
            bot = TemplateForecaster.__new__(TemplateForecaster)
            bot.allow_research_fallback = False
            bot._research_provider_timeout_count = 0

            async def asknews_provider(q: str) -> str:
                raise RuntimeError("connection timeout")

            with caplog.at_level(logging.WARNING, logger="main"):
                result = await bot._run_providers_parallel("test question", [(asknews_provider, "asknews")])

        assert result == ""
        assert bot._research_provider_timeout_count == 1
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("asknews" in r.getMessage().lower() for r in warning_records)
