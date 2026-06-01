"""Tests for the extracted ResearchOrchestrator.

Verifies that the orchestrator:
1. Delegates to providers correctly
2. Handles caching (benchmarking mode)
3. Runs gap-fill when enabled
4. Combines parallel provider results with headers
5. Falls back gracefully on provider failure
6. Exposes timeout_count for alerting
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import GeneralLlm

from metaculus_bot.research_orchestrator import ResearchOrchestrator


@pytest.fixture
def mock_llm() -> GeneralLlm:
    return GeneralLlm(model="test/model", temperature=0.0)


@pytest.fixture
def question() -> MagicMock:
    q = MagicMock()
    q.id_of_question = 42
    q.question_text = "Will X happen by 2027?"
    q.page_url = "https://metaculus.com/questions/42"
    q.resolution_criteria = "Resolves YES if X happens"
    q.fine_print = ""
    return q


@pytest.fixture
def orchestrator(mock_llm: GeneralLlm) -> ResearchOrchestrator:
    return ResearchOrchestrator(
        default_llm=mock_llm,
        summarizer_llm=mock_llm,
    )


class TestRunResearch:
    @pytest.mark.asyncio
    async def test_calls_providers_and_returns_combined_result(self, orchestrator, question):
        provider_a = AsyncMock(return_value="Result from provider A")
        provider_b = AsyncMock(return_value="Result from provider B")

        with (
            patch.object(
                orchestrator,
                "_select_research_providers",
                return_value=[(provider_a, "asknews"), (provider_b, "native_search")],
            ),
            # AskNews is summarized inline; the briefing replaces the raw articles.
            patch.object(
                orchestrator._summarizer_llm,
                "invoke",
                new_callable=AsyncMock,
                return_value="AskNews briefing",
            ),
        ):
            result = await orchestrator.run_research(question)

        assert "AskNews briefing" in result  # AskNews summarized
        assert "Result from provider A" not in result  # raw articles replaced
        assert "Result from provider B" in result  # native search raw
        assert "## News Articles (AskNews)" in result
        assert "## Web Research (Native Search)" in result

    @pytest.mark.asyncio
    async def test_uses_cache_in_benchmarking_mode(self, mock_llm, question):
        cache = {42: "cached research"}
        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            is_benchmarking=True,
            research_cache=cache,
        )

        result = await orch.run_research(question)
        assert result == "cached research"

    @pytest.mark.asyncio
    async def test_stores_to_cache_in_benchmarking_mode(self, mock_llm, question):
        cache: dict[int, str] = {}
        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            is_benchmarking=True,
            research_cache=cache,
        )

        provider = AsyncMock(return_value="fresh research")

        with patch.object(orch, "_select_research_providers", return_value=[(provider, "custom")]):
            result = await orch.run_research(question)

        assert result  # non-empty
        assert cache[42]  # stored

    @pytest.mark.asyncio
    async def test_gap_fill_appended_when_enabled(self, orchestrator, question, monkeypatch):
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")

        provider = AsyncMock(return_value="A" * 200)  # exceeds GAP_FILL_MIN_RESEARCH_CHARS

        with (
            patch.object(orchestrator, "_select_research_providers", return_value=[(provider, "custom")]),
            patch(
                "metaculus_bot.targeted_research.run_gap_fill_pass",
                new_callable=AsyncMock,
                return_value="gap fill addendum",
            ) as mock_gap_fill,
        ):
            result = await orchestrator.run_research(question)

        assert "gap fill addendum" in result
        assert "Targeted Gap-Fill" in result
        mock_gap_fill.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_timeout_count_incremented_on_provider_failure(self, orchestrator, question):
        failing = AsyncMock(side_effect=RuntimeError("timeout"))
        working = AsyncMock(return_value="ok")

        with patch.object(
            orchestrator,
            "_select_research_providers",
            return_value=[(failing, "native_search"), (working, "custom")],
        ):
            result = await orchestrator.run_research(question)

        assert orchestrator.timeout_count == 1
        assert "ok" in result


class TestAskNewsSummarization:
    @pytest.mark.asyncio
    async def test_invokes_summarizer_llm(self, orchestrator, question):
        with patch.object(orchestrator._summarizer_llm, "invoke", new_callable=AsyncMock, return_value="summary"):
            result = await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert result == "summary"

    @pytest.mark.asyncio
    async def test_empty_research_skips_summarizer(self, orchestrator, question):
        with patch.object(orchestrator._summarizer_llm, "invoke", new_callable=AsyncMock) as invoke:
            result = await orchestrator._summarize_asknews(question, "   ")

        assert result == "   "
        invoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_soft_falls_back_to_raw_on_summarizer_error(self, orchestrator, question):
        with patch.object(
            orchestrator._summarizer_llm,
            "invoke",
            new_callable=AsyncMock,
            side_effect=RuntimeError("summarizer down"),
        ):
            result = await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert result == "raw asknews articles"

    @pytest.mark.asyncio
    async def test_only_asknews_is_summarized(self, orchestrator, question):
        """AskNews output is summarized; every other provider passes through raw."""
        asknews = AsyncMock(return_value="raw asknews articles")
        native = AsyncMock(return_value="native search prose")

        with (
            patch.object(
                orchestrator,
                "_select_research_providers",
                return_value=[(asknews, "asknews"), (native, "native_search")],
            ),
            patch.object(
                orchestrator._summarizer_llm,
                "invoke",
                new_callable=AsyncMock,
                return_value="ASKNEWS BRIEFING",
            ) as invoke,
        ):
            result = await orchestrator.run_research(question)

        # Summarizer ran exactly once, and on the AskNews payload specifically.
        invoke.assert_awaited_once()
        assert "raw asknews articles" in invoke.await_args.args[0]
        assert "ASKNEWS BRIEFING" in result
        assert "raw asknews articles" not in result
        # Native search prose is delivered verbatim.
        assert "native search prose" in result

    @pytest.mark.asyncio
    async def test_asknews_fallback_prose_is_not_summarized(self, orchestrator, question, monkeypatch):
        """When AskNews fails and falls back to Perplexity/Exa, the fallback is
        already LLM prose, so it must NOT be summarized (no double-summarization)."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
        asknews = AsyncMock(side_effect=RuntimeError("asknews down"))

        with (
            patch.object(
                orchestrator,
                "_select_research_providers",
                return_value=[(asknews, "asknews")],
            ),
            patch.object(
                orchestrator,
                "_call_perplexity",
                new_callable=AsyncMock,
                return_value="perplexity fallback prose",
            ),
            patch.object(orchestrator._summarizer_llm, "invoke", new_callable=AsyncMock) as invoke,
        ):
            result = await orchestrator.run_research(question)

        # Fallback prose passes through verbatim; summarizer never runs.
        invoke.assert_not_awaited()
        assert "perplexity fallback prose" in result


class TestProviderSelection:
    def test_returns_empty_stub_when_no_providers_configured(self, mock_llm, monkeypatch):
        monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
        monkeypatch.delenv("ASKNEWS_SECRET", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "false")
        monkeypatch.setenv("GEMINI_SEARCH_ENABLED", "false")
        monkeypatch.setenv("FINANCIAL_DATA_ENABLED", "false")
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "false")
        monkeypatch.delenv("RESEARCH_PROVIDER", raising=False)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
        )
        providers = orch._select_research_providers()
        assert len(providers) == 1
        assert providers[0][1] == "none"

    def test_includes_native_search_when_enabled(self, mock_llm, monkeypatch):
        monkeypatch.setenv("NATIVE_SEARCH_ENABLED", "true")
        monkeypatch.setenv("NATIVE_SEARCH_MODEL", "openai/gpt-5.5")
        monkeypatch.setenv("FINANCIAL_DATA_ENABLED", "false")
        monkeypatch.setenv("GEMINI_SEARCH_ENABLED", "false")
        monkeypatch.setenv("PREDICTION_MARKETS_ENABLED", "false")
        monkeypatch.delenv("ASKNEWS_CLIENT_ID", raising=False)
        monkeypatch.delenv("ASKNEWS_SECRET", raising=False)
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("RESEARCH_PROVIDER", raising=False)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
        )
        providers = orch._select_research_providers()
        names = [n for _, n in providers]
        assert "native_search" in names


class TestConcurrencyLimiter:
    @pytest.mark.asyncio
    async def test_limits_concurrent_research_calls(self, mock_llm, question):
        max_concurrent = 2
        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            max_concurrent_research=max_concurrent,
        )
        active = {"count": 0, "max_seen": 0}
        lock = asyncio.Lock()

        async def slow_provider(q):  # noqa: ASYNC910 - intentionally simple test helper
            async with lock:
                active["count"] += 1
                active["max_seen"] = max(active["max_seen"], active["count"])
            await asyncio.sleep(0.05)
            async with lock:
                active["count"] -= 1
            return "done"

        with patch.object(orch, "_select_research_providers", return_value=[(slow_provider, "custom")]):
            results = await asyncio.gather(
                orch.run_research(question),
                orch.run_research(question),
                orch.run_research(question),
                orch.run_research(question),
            )

        assert all("done" in r for r in results)
        assert active["max_seen"] <= max_concurrent


class TestIntegrationWithTemplateForecaster:
    """Verify that TemplateForecaster delegates to ResearchOrchestrator correctly."""

    @pytest.mark.asyncio
    async def test_bot_run_research_delegates_to_orchestrator(self, question):
        from main import TemplateForecaster
        from metaculus_bot.aggregation_strategies import AggregationStrategy

        sentinel = GeneralLlm(model="test/sentinel", temperature=0.0)
        bot = TemplateForecaster(
            llms={"default": sentinel, "parser": sentinel, "researcher": sentinel, "summarizer": sentinel},
            aggregation_strategy=AggregationStrategy.MEAN,
        )

        mock_provider = AsyncMock(return_value="delegated research")

        with patch.object(bot._research, "_select_research_providers", return_value=[(mock_provider, "custom")]):
            result = await bot.run_research(question)

        assert "delegated research" in result

    @pytest.mark.asyncio
    async def test_bot_run_research_patchable_at_bot_level(self, question):
        """Existing tests patch bot.run_research directly; this must still work."""
        from main import TemplateForecaster
        from metaculus_bot.aggregation_strategies import AggregationStrategy

        sentinel = GeneralLlm(model="test/sentinel", temperature=0.0)
        bot = TemplateForecaster(
            llms={"default": sentinel, "parser": sentinel, "researcher": sentinel, "summarizer": sentinel},
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        bot.run_research = AsyncMock(return_value="patched result")  # ty: ignore[invalid-assignment]
        result = await bot.run_research(question)
        assert result == "patched result"

    def test_timeout_count_exposed_via_bot(self, question):
        """Bot exposes timeout count from the orchestrator for alerting."""
        from main import TemplateForecaster
        from metaculus_bot.aggregation_strategies import AggregationStrategy

        sentinel = GeneralLlm(model="test/sentinel", temperature=0.0)
        bot = TemplateForecaster(
            llms={"default": sentinel, "parser": sentinel, "researcher": sentinel, "summarizer": sentinel},
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        bot._research.timeout_count = 5
        assert bot._research_provider_timeout_count == 5
