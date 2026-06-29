"""Tests for native search research provider."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_SUMMARIZE_ASKNEWS_PATH = "metaculus_bot.research.orchestrator.ResearchOrchestrator._summarize_asknews"


async def _passthrough_summarize_asknews(self, question, research):
    """Identity stub for the AskNews summarizer.

    The combine test below routes an ``"asknews"``-named provider through the
    orchestrator, which invokes ``_summarize_asknews`` with the bogus
    ``GeneralLlm(model="test/model")`` summarizer. That raises BadRequestError;
    on failure the orchestrator now drops the AskNews block to empty (it no
    longer falls back to raw articles), which would erase the provider output
    the test asserts on. Patching the summarizer to a passthrough keeps the
    AskNews block intact without coupling the routing test to the failure path.
    """
    del self, question
    await asyncio.sleep(0)
    return research


def _make_q(text: str) -> MagicMock:
    """Build a minimal MetaculusQuestion-shaped mock for the new ResearchCallable
    contract. Tests only care about question_text on this path."""
    q = MagicMock()
    q.question_text = text
    return q


@pytest.mark.asyncio
async def test_native_search_provider_constructs_correct_model_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify native search provider constructs model name correctly via NATIVE_SEARCH_MODEL env override."""
    monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "true")
    monkeypatch.setenv("NATIVE_SEARCH_MODEL", "openai/gpt-5.5")

    captured_model: str | None = None
    captured_kwargs: dict | None = None

    class MockLlm:
        def __init__(self, model: str, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal captured_model, captured_kwargs
            captured_model = model
            captured_kwargs = kwargs
            self.model = model

        async def invoke(self, prompt: str) -> str:
            return "Mock research response"

    with patch("metaculus_bot.research.providers.build_llm_with_openrouter_fallback", MockLlm):
        from metaculus_bot.research.providers import native_search_provider

        provider = native_search_provider()
        await provider(_make_q("Will X happen?"))

    assert captured_model == "openrouter/openai/gpt-5.5"
    # Default reasoning effort + verbosity should be plumbed through.
    # `verbosity` is now top-level (canonical OpenRouter / litellm form);
    # `extra_body` is no longer used to smuggle it.
    assert captured_kwargs is not None
    # Default reasoning effort dropped medium→low on 2026-05-20; see
    # NATIVE_SEARCH_REASONING_EFFORT_DEFAULT in constants.py for rationale.
    assert captured_kwargs.get("reasoning") == {"effort": "low"}
    assert captured_kwargs.get("verbosity") == "low"
    assert "extra_body" not in captured_kwargs
    # 2026-05-20: pinned to allowed_tries=1 for native_search; the wall-clock
    # guard at the caller bounds the budget, retrying a malformed-whitespace
    # response from OpenRouter doesn't help (and burns the budget).
    assert captured_kwargs.get("allowed_tries") == 1


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

    with patch("metaculus_bot.research.providers.build_llm_with_openrouter_fallback", MockLlm):
        from metaculus_bot.research.providers import native_search_provider

        provider = native_search_provider(model_slug="openai/gpt-4o")
        await provider(_make_q("Will X happen?"))

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

    with patch("metaculus_bot.research.providers.build_llm_with_openrouter_fallback", MockLlm):
        from metaculus_bot.research.providers import native_search_provider

        provider = native_search_provider(is_benchmarking=False)
        await provider(_make_q("Will X happen?"))

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

    with patch("metaculus_bot.research.providers.build_llm_with_openrouter_fallback", MockLlm):
        from metaculus_bot.research.providers import native_search_provider

        provider = native_search_provider(is_benchmarking=True)
        await provider(_make_q("Will X happen?"))

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

    with patch("metaculus_bot.research.providers.build_llm_with_openrouter_fallback", MockLlm):
        from metaculus_bot.research.providers import native_search_provider

        provider = native_search_provider()
        await provider(_make_q("Will X happen?"))

    assert captured_prompt is not None
    assert "DO NOT hallucinate" in captured_prompt
    assert "only cite what you actually found" in captured_prompt


@pytest.mark.asyncio
async def test_native_search_provider_enforces_wall_clock_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung llm.invoke must be bounded by NATIVE_SEARCH_WALL_TIMEOUT.

    Regression test for the 2026-05-20 incident: an OpenRouter response
    dripped ~700 lines of whitespace over 8m37s before closing with malformed
    JSON, defeating the per-HTTP-request timeout entirely. The asyncio.wait_for
    wrapper in _native_search_provider._fetch is the wall-clock backstop;
    this test locks it in so a future refactor can't silently remove it.
    """
    # _fetch reads NATIVE_SEARCH_WALL_TIMEOUT via a function-scoped import from
    # constants, so we patch the constants module (not the research_providers
    # module) for the override to take effect at call time.
    monkeypatch.setattr("metaculus_bot.constants.NATIVE_SEARCH_WALL_TIMEOUT", 0.05)

    class HangingLlm:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.model = kwargs.get("model", "mock")

        async def invoke(self, prompt: str) -> str:
            # Sleep well past the 0.05s wall-clock cap; test passes only if
            # asyncio.wait_for cancels this before it returns.
            await asyncio.sleep(5)
            return "should never reach here"

    with patch("metaculus_bot.research.providers.build_llm_with_openrouter_fallback", HangingLlm):
        from metaculus_bot.research.providers import native_search_provider

        provider = native_search_provider()
        with pytest.raises(asyncio.TimeoutError):
            await provider(_make_q("Will X happen?"))


class TestBuildNativeSearchLlmOverrides:
    """Per-call reasoning_effort / verbosity overrides on build_native_search_llm.

    The gap-fill resolver needs gpt-5.4-mini at MEDIUM effort without perturbing
    the global NATIVE_SEARCH_REASONING_EFFORT env (which the main native_search
    provider reads at LOW). These tests lock the override semantics: explicit
    args win over env; None preserves the existing env-read behavior exactly.
    """

    def _captured_kwargs(self, *, model_slug=None, reasoning_effort=None, verbosity=None) -> dict:  # type: ignore[no-untyped-def]
        captured: dict = {}

        class MockLlm:
            def __init__(self, model: str, **kwargs):  # type: ignore[no-untyped-def]
                captured["model"] = model
                captured.update(kwargs)
                self.model = model

        with patch("metaculus_bot.research.providers.build_llm_with_openrouter_fallback", MockLlm):
            from metaculus_bot.research.providers import build_native_search_llm

            build_native_search_llm(model_slug, reasoning_effort=reasoning_effort, verbosity=verbosity)
        return captured

    def test_reasoning_effort_override_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit reasoning_effort="medium" beats NATIVE_SEARCH_REASONING_EFFORT=low."""
        monkeypatch.setenv("NATIVE_SEARCH_REASONING_EFFORT", "low")

        captured = self._captured_kwargs(model_slug="openai/gpt-5.4-mini", reasoning_effort="medium")

        assert "gpt-5.4-mini" in captured["model"]
        assert captured.get("reasoning") == {"effort": "medium"}

    def test_verbosity_override_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Explicit verbosity="high" beats NATIVE_SEARCH_VERBOSITY=low."""
        monkeypatch.setenv("NATIVE_SEARCH_VERBOSITY", "low")

        captured = self._captured_kwargs(verbosity="high")

        assert captured.get("verbosity") == "high"

    def test_none_preserves_env_default_behavior(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Passing nothing preserves the env-read path: env value flows through."""
        monkeypatch.setenv("NATIVE_SEARCH_REASONING_EFFORT", "medium")
        monkeypatch.setenv("NATIVE_SEARCH_VERBOSITY", "high")

        captured = self._captured_kwargs()

        assert captured.get("reasoning") == {"effort": "medium"}
        assert captured.get("verbosity") == "high"

    def test_empty_string_override_disables_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit empty-string override disables the kwarg (matches env semantics)."""
        monkeypatch.setenv("NATIVE_SEARCH_REASONING_EFFORT", "low")

        captured = self._captured_kwargs(reasoning_effort="", verbosity="")

        assert "reasoning" not in captured
        assert "verbosity" not in captured


class TestParallelProviderSelection:
    """Tests for parallel provider selection in main.py."""

    def test_select_research_providers_returns_primary_only_when_native_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify only primary provider returned when native search disabled."""
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")
        monkeypatch.setenv("FINANCIAL_DATA_ENABLED", "false")
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)

        mock_provider = AsyncMock(return_value="primary research")

        with patch.object(orch, "_select_research_provider", return_value=(mock_provider, "asknews")):
            providers = orch._select_research_providers()

        assert len(providers) == 1
        assert providers[0][1] == "asknews"

    def test_select_research_providers_includes_native_when_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify native search provider added when enabled."""
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "true")
        monkeypatch.setenv("NATIVE_SEARCH_MODEL", "openai/gpt-5.5")
        monkeypatch.setenv("FINANCIAL_DATA_ENABLED", "false")
        monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
        monkeypatch.setenv("ASKNEWS_SECRET", "secret")

        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm)

        mock_provider = AsyncMock(return_value="primary research")

        with patch.object(orch, "_select_research_provider", return_value=(mock_provider, "asknews")):
            providers = orch._select_research_providers()

        assert len(providers) == 2
        provider_names = [name for _, name in providers]
        assert "asknews" in provider_names
        assert "native_search" in provider_names


class TestParallelExecution:
    """Tests for parallel provider execution."""

    @pytest.mark.asyncio
    async def test_run_providers_parallel_combines_results(self) -> None:
        """Verify parallel execution combines results from multiple providers."""
        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)

        provider1 = AsyncMock(return_value="Research from provider 1")
        provider2 = AsyncMock(return_value="Research from provider 2")

        providers = [(provider1, "asknews"), (provider2, "native_search")]
        with patch(_SUMMARIZE_ASKNEWS_PATH, _passthrough_summarize_asknews):
            result, _ = await orch._run_providers_parallel(_make_q("Test question"), providers)

        assert "Research from provider 1" in result
        assert "Research from provider 2" in result
        assert "## News Articles (AskNews)" in result
        assert "## Web Research (Native Search)" in result

    @pytest.mark.asyncio
    async def test_run_providers_parallel_handles_provider_failure(self) -> None:
        """Verify parallel execution handles individual provider failures gracefully."""
        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)

        failing_provider = AsyncMock(side_effect=RuntimeError("Provider failed"))
        working_provider = AsyncMock(return_value="Research from working provider")

        providers = [(failing_provider, "failing"), (working_provider, "working")]
        result, _ = await orch._run_providers_parallel(_make_q("Test question"), providers)

        assert "Research from working provider" in result

    @pytest.mark.asyncio
    async def test_run_providers_parallel_runs_concurrently(self) -> None:
        """Verify providers actually run in parallel, not sequentially."""
        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)

        execution_order: list[str] = []
        completion_order: list[str] = []

        async def slow_provider(q):  # noqa: ASYNC910
            execution_order.append("slow_start")
            await asyncio.sleep(0.1)
            completion_order.append("slow")
            return "Slow result"

        async def fast_provider(q):  # noqa: ASYNC910
            execution_order.append("fast_start")
            await asyncio.sleep(0.01)
            completion_order.append("fast")
            return "Fast result"

        providers = [(slow_provider, "slow"), (fast_provider, "fast")]
        await orch._run_providers_parallel(_make_q("Test question"), providers)

        assert execution_order == ["slow_start", "fast_start"]
        assert completion_order == ["fast", "slow"]


class TestAskNewsSubscriptionErrorHandling:
    """F10: AskNews off-season 403 vs operational failure branching in
    _run_providers_parallel.

    The provider loop must silence only the exact subscription-inactive
    signature (class-name + message match) — every other asknews failure must
    bump timeout_count so CI surfaces it.
    """

    @pytest.mark.asyncio
    async def test_asknews_subscription_inactive_not_counted_and_info_logged(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """AskNews 403011 → counter stays 0, INFO log emitted, no WARNING."""
        import logging

        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)

        class ForbiddenError(Exception):
            pass

        async def asknews_provider(q):  # noqa: ASYNC910
            raise ForbiddenError("403011 - subscription is not currently active")

        with caplog.at_level(logging.INFO, logger="metaculus_bot.research.orchestrator"):
            result, _ = await orch._run_providers_parallel(_make_q("test question"), [(asknews_provider, "asknews")])

        assert result == "", "Failed provider yields empty result."
        assert orch.timeout_count == 0, "Off-season subscription-inactive must NOT count as an alertable failure."
        info_records = [r for r in caplog.records if r.levelno == logging.INFO and "asknews" in r.getMessage().lower()]
        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING and "asknews" in r.getMessage().lower()
        ]
        assert any("inactive" in r.getMessage().lower() or "off-season" in r.getMessage().lower() for r in info_records)
        assert warning_records == [], "Subscription-inactive must not log at WARNING."

    @pytest.mark.asyncio
    async def test_asknews_non_subscription_error_counted_and_warning_logged(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """AskNews non-subscription failure → counter bumped, WARNING logged."""
        import logging

        from forecasting_tools import GeneralLlm

        from metaculus_bot.research.orchestrator import ResearchOrchestrator

        mock_llm = GeneralLlm(model="test/model", temperature=0.0)
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)

        asknews_provider = AsyncMock(side_effect=RuntimeError("connection timeout"))

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.research.orchestrator"):
            result, _ = await orch._run_providers_parallel(_make_q("test question"), [(asknews_provider, "asknews")])

        assert result == ""
        assert orch.timeout_count == 1
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any("asknews" in r.getMessage().lower() for r in warning_records)
