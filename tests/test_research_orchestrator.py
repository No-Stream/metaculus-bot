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

from metaculus_bot.research.orchestrator import ResearchOrchestrator


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
                "metaculus_bot.research.targeted.run_gap_fill_pass",
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
    async def test_soft_falls_back_to_raw_on_transient_summarizer_error(self, orchestrator, question):
        """Transient LLM-provider errors (timeouts, API hiccups) soft-fail to the raw articles.

        Round-2: the summarizer invoke is wrapped in the broad 30s-gated retry, which
        retries this fast asyncio.TimeoutError the full backoff schedule before it
        surfaces and is caught by _SUMMARIZER_TRANSIENT_EXCEPTIONS → raw articles.
        asyncio.sleep is patched so the backoffs are instant.
        """
        with (
            patch("metaculus_bot.llm_retry.asyncio.sleep", new=AsyncMock()),
            patch.object(
                orchestrator._summarizer_llm,
                "invoke",
                new_callable=AsyncMock,
                side_effect=asyncio.TimeoutError("summarizer timed out"),
            ),
        ):
            result = await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert result == "raw asknews articles"

    @pytest.mark.asyncio
    async def test_non_transient_summarizer_error_propagates(self, orchestrator, question):
        """A genuine bug must ultimately crash, not silently degrade.

        Round-2: the summarizer invoke is now wrapped in the broad 30s-gated retry,
        which retries an AttributeError (not in PERMANENT_NO_RETRY_EXCEPTIONS) a few
        times before exhausting and propagating. The key contract is unchanged —
        the bug surfaces rather than being swallowed into a soft-fall-back to raw
        articles (which the _SUMMARIZER_TRANSIENT_EXCEPTIONS catch would NOT do for
        AttributeError anyway). asyncio.sleep is patched so the backoffs are instant.
        """
        with (
            patch("metaculus_bot.llm_retry.asyncio.sleep", new=AsyncMock()),
            patch.object(
                orchestrator._summarizer_llm,
                "invoke",
                new_callable=AsyncMock,
                side_effect=AttributeError("prompt builder bug"),
            ),
        ):
            with pytest.raises(AttributeError):
                await orchestrator._summarize_asknews(question, "raw asknews articles")

    @pytest.mark.asyncio
    async def test_fast_blip_on_summarizer_invoke_retries_then_succeeds(self, orchestrator, question):
        """A fast litellm.Timeout on the summarizer invoke is retried; the next call wins.

        Locks the Round-2 broad-retry wiring: SUMMARIZER_LLM is allowed_tries=1, so
        this gated wrapper is its sole retry layer.
        """
        import litellm.exceptions as litellm_exc

        invoke = AsyncMock(side_effect=[litellm_exc.Timeout("blip", model="m", llm_provider="openrouter"), "summary"])
        with (
            patch("metaculus_bot.llm_retry.asyncio.sleep", new=AsyncMock()),
            patch.object(orchestrator._summarizer_llm, "invoke", new=invoke),
        ):
            result = await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert result == "summary"
        assert invoke.await_count == 2

    @pytest.mark.asyncio
    async def test_slow_summarizer_failure_not_retried_soft_falls_to_raw(self, orchestrator, question):
        """A summarizer failure past the 30s gate is NOT retried; it soft-falls to raw articles.

        A slow litellm.Timeout is a transient-type exception that _SUMMARIZER_TRANSIENT_
        EXCEPTIONS catches, so after the gate blocks the retry the orchestrator returns
        the raw articles (one invoke, no deadline-risking re-fire).
        """
        import litellm.exceptions as litellm_exc

        from metaculus_bot.llm_retry import TRANSIENT_RETRY_MAX_ELAPSED_S

        invoke = AsyncMock(side_effect=litellm_exc.Timeout("stall", model="m", llm_provider="openrouter"))
        clock = iter([0.0] + [TRANSIENT_RETRY_MAX_ELAPSED_S + 5.0] * 20)

        with (
            patch("metaculus_bot.llm_retry.time.monotonic", lambda: next(clock)),
            patch.object(orchestrator._summarizer_llm, "invoke", new=invoke),
        ):
            result = await orchestrator._summarize_asknews(question, "raw asknews articles")

        assert result == "raw asknews articles"
        assert invoke.await_count == 1

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
        assert invoke.await_args is not None
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


class TestProviderDiagnosticsCapture:
    """Per-provider observability: _run_providers_parallel returns a ProviderResult list
    alongside the combined text, and run_research appends a diagnostics block."""

    @pytest.mark.asyncio
    async def test_status_ok_for_nonempty_provider(self, mock_llm, question):
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(return_value="some research prose")

        _, results = await orch._run_providers_parallel(question, [(provider, "native_search")])

        assert len(results) == 1
        assert results[0].name == "native_search"
        assert results[0].status == "ok"
        assert results[0].chars == len("some research prose")
        assert isinstance(results[0].latency_ms, int)
        assert results[0].latency_ms >= 0
        assert results[0].error_type is None
        assert results[0].error_message is None
        assert results[0].details == {}

    @pytest.mark.asyncio
    async def test_status_empty_for_blank_provider(self, mock_llm, question):
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(return_value="   ")

        _, results = await orch._run_providers_parallel(question, [(provider, "native_search")])

        assert results[0].status == "empty"
        assert results[0].chars == 0

    @pytest.mark.asyncio
    async def test_status_errored_for_raising_provider(self, mock_llm, question):
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(side_effect=RuntimeError("boom"))

        _, results = await orch._run_providers_parallel(question, [(provider, "native_search")])

        assert results[0].status == "errored"
        assert results[0].chars == 0
        assert results[0].error_type == "RuntimeError"
        assert results[0].error_message is not None
        assert "boom" in results[0].error_message
        assert orch.timeout_count == 1

    @pytest.mark.asyncio
    async def test_status_inactive_for_asknews_subscription_error(self, mock_llm, question):
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)

        class ForbiddenError(Exception):
            pass

        provider = AsyncMock(side_effect=ForbiddenError("403011 - subscription is not currently active"))

        _, results = await orch._run_providers_parallel(question, [(provider, "asknews")])

        assert results[0].status == "inactive"
        assert results[0].chars == 0
        assert results[0].error_type == "ForbiddenError"
        # Off-season is expected, not an alertable failure.
        assert orch.timeout_count == 0

    @pytest.mark.asyncio
    async def test_status_fallback_when_asknews_falls_back_to_prose(self, mock_llm, question, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=True)
        asknews = AsyncMock(side_effect=RuntimeError("asknews down"))

        with patch.object(
            orch,
            "_call_perplexity",
            new_callable=AsyncMock,
            return_value="perplexity fallback prose",
        ):
            text, results = await orch._run_providers_parallel(question, [(asknews, "asknews")])

        assert results[0].status == "fallback"
        assert results[0].name == "asknews"
        assert results[0].chars == len("perplexity fallback prose")
        assert "perplexity fallback prose" in text

    @pytest.mark.asyncio
    async def test_one_result_per_provider_mixed_statuses(self, mock_llm, question):
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        ok = AsyncMock(return_value="native prose")
        failing = AsyncMock(side_effect=RuntimeError("timeout"))

        _, results = await orch._run_providers_parallel(question, [(ok, "native_search"), (failing, "gemini_search")])

        by_name = {r.name: r for r in results}
        assert by_name["native_search"].status == "ok"
        assert by_name["gemini_search"].status == "errored"

    @pytest.mark.asyncio
    async def test_diagnostics_block_appended_to_research(self, mock_llm, question):
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        ok = AsyncMock(return_value="native prose")
        failing = AsyncMock(side_effect=RuntimeError("timeout"))

        with patch.object(
            orch,
            "_select_research_providers",
            return_value=[(ok, "native_search"), (failing, "gemini_search")],
        ):
            research = await orch.run_research(question)

        assert "## Provider Diagnostics" in research
        diag_lines = [line for line in research.splitlines() if line.startswith("- ")]
        assert any(line.startswith("- native_search: ok |") for line in diag_lines)
        errored_line = next(line for line in diag_lines if line.startswith("- gemini_search: errored |"))
        assert "RuntimeError" in errored_line

    @pytest.mark.asyncio
    async def test_diagnostics_block_after_gap_fill(self, mock_llm, question, monkeypatch):
        monkeypatch.setenv("GAP_FILL_ENABLED", "true")
        orch = ResearchOrchestrator(default_llm=mock_llm, summarizer_llm=mock_llm, allow_research_fallback=False)
        provider = AsyncMock(return_value="A" * 200)

        with (
            patch.object(orch, "_select_research_providers", return_value=[(provider, "native_search")]),
            patch(
                "metaculus_bot.research.targeted.run_gap_fill_pass",
                new_callable=AsyncMock,
                return_value="gap fill addendum",
            ),
        ):
            research = await orch.run_research(question)

        # Ordering: providers -> gap-fill -> provider diagnostics.
        assert research.index("Targeted Gap-Fill") < research.index("## Provider Diagnostics")

    @pytest.mark.asyncio
    async def test_sink_receives_attempted_and_succeeded_excluding_failures(self, mock_llm, question, monkeypatch):
        """End-to-end guard for the schema-v2 derivation feeding the research sink.

        ``providers_attempted`` is every selected provider in order; ``providers_succeeded``
        is ONLY the ok/fallback ones. This pins the exclusion of ``empty``, ``errored``, AND —
        most importantly — ``inactive`` (AskNews off-season), so a regression that derived
        ``providers_succeeded`` from the attempted list, or included ``inactive``, would fail here.
        Fallback is None-free in this orchestrator (allow_research_fallback=False), so the inactive
        AskNews subscription error coerces to status=``inactive`` rather than triggering a fallback.
        """
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)

        captured: dict = {}

        def sink(**kwargs) -> None:  # noqa: ANN003
            captured.update(kwargs)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            allow_research_fallback=False,
            research_sink=sink,
        )

        class ForbiddenError(Exception):
            pass

        ok = AsyncMock(return_value="real research prose")
        empty = AsyncMock(return_value="")
        errored = AsyncMock(side_effect=RuntimeError("provider blew up"))
        inactive = AsyncMock(side_effect=ForbiddenError("403011 - subscription is not currently active"))

        selected = [
            (ok, "native_search"),
            (empty, "gemini_search"),
            (errored, "financial_data"),
            (inactive, "asknews"),
        ]

        with patch.object(orch, "_select_research_providers", return_value=selected):
            await orch.run_research(question)

        # Sink fired (question fixture id_of_question=42 satisfies the qid gate).
        assert captured, "research_sink was never called"
        assert captured["providers_attempted"] == ["native_search", "gemini_search", "financial_data", "asknews"]
        # Only the ok provider succeeded — empty, errored, AND inactive are all excluded.
        assert captured["providers_succeeded"] == ["native_search"]
        assert "gemini_search" not in captured["providers_succeeded"]  # empty
        assert "financial_data" not in captured["providers_succeeded"]  # errored
        assert "asknews" not in captured["providers_succeeded"]  # inactive (off-season)

        # provider_results round-trips each per-provider status as a list of dicts.
        statuses = {r["name"]: r["status"] for r in captured["provider_results"]}
        assert statuses == {
            "native_search": "ok",
            "gemini_search": "empty",
            "financial_data": "errored",
            "asknews": "inactive",
        }

    @pytest.mark.asyncio
    async def test_sink_counts_fallback_as_succeeded(self, mock_llm, question, monkeypatch):
        """A ``fallback`` provider (AskNews failed, prose fallback supplied the result) counts as
        succeeded — it contributed usable research. Pins the second member of SUCCEEDED_STATUSES."""
        monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

        captured: dict = {}

        def sink(**kwargs) -> None:  # noqa: ANN003
            captured.update(kwargs)

        orch = ResearchOrchestrator(
            default_llm=mock_llm,
            summarizer_llm=mock_llm,
            allow_research_fallback=True,
            research_sink=sink,
        )
        asknews = AsyncMock(side_effect=RuntimeError("asknews down"))

        with (
            patch.object(orch, "_select_research_providers", return_value=[(asknews, "asknews")]),
            patch.object(orch, "_call_perplexity", new_callable=AsyncMock, return_value="perplexity fallback prose"),
        ):
            await orch.run_research(question)

        assert captured["providers_attempted"] == ["asknews"]
        assert captured["providers_succeeded"] == ["asknews"]  # fallback status counts as succeeded
        assert captured["provider_results"][0]["status"] == "fallback"


class TestDemoteInnerHeadings:
    """Provider bodies must not carry headings at/above the h2 provider header.

    Otherwise the framework's report_sections_to_markdown raises ("First section
    must be at the highest heading level") and degrades to the [Hashtag] fallback.
    """

    def test_demotes_h1_and_h2_to_at_least_h3(self) -> None:
        from metaculus_bot.research.orchestrator import _demote_inner_headings

        body = "# Historical Context\nbody\n## Recent Developments\nmore\n### Already deep\nkept"
        out = _demote_inner_headings(body)
        assert "### Historical Context" in out  # h1 -> h3
        assert "#### Recent Developments" in out  # h2 -> h4
        assert "### Already deep" in out  # h3 untouched
        assert not out.startswith("# "), "must not start with h1"
        assert "\n# " not in out, "no h1 on any line"
        assert not out.startswith("## "), "must not start with h2"
        assert "\n## " not in out, "no h2 on any line"

    def test_inline_hash_not_treated_as_heading(self) -> None:
        from metaculus_bot.research.orchestrator import _demote_inner_headings

        body = "Issue #42 and C# are not headings\n# Real Heading"
        out = _demote_inner_headings(body)
        assert "Issue #42 and C# are not headings" in out
        assert "### Real Heading" in out

    def test_combined_research_first_section_is_minimum_level(self) -> None:
        """After normalization, every provider's h2 header outranks its body.

        The first section of the combined string must be the minimum heading
        level so the framework wouldn't raise on it.
        """
        import re

        from metaculus_bot.research.orchestrator import _demote_inner_headings

        # Simulate the assembly: h2 provider header + a body that had its own h1.
        provider_body = "# Historical Context\nstuff\n## Recent\nmore"
        combined = f"## News Articles (AskNews)\n{_demote_inner_headings(provider_body)}"

        heading_levels = [len(m.group(1)) for m in re.finditer(r"^(#+)\s", combined, re.MULTILINE)]
        assert heading_levels, "precondition: combined must contain headings"
        # The first heading (provider header, h2) must be the minimum.
        assert heading_levels[0] == min(heading_levels)
        assert heading_levels[0] == 2
