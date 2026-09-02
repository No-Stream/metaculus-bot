from unittest.mock import AsyncMock, MagicMock, patch

import litellm.exceptions
import pytest
from forecasting_tools import GeneralLlm

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import (
    PERPLEXITY_RESEARCH_MODEL,
    PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER,
    PERPLEXITY_WALL_TIMEOUT,
)
from metaculus_bot.prompts import OUTSIDE_VENUE_MARKET_ODDS_POLICY
from metaculus_bot.research import orchestrator, providers


@pytest.fixture
def question() -> MagicMock:
    q = MagicMock()
    q.id_of_question = 999
    q.question_text = "Sample question?"
    q.page_url = "https://example.com/q/999"
    return q


@pytest.fixture
def base_llms() -> dict[str, GeneralLlm]:
    sentinel = GeneralLlm(model="sentinel", temperature=0.0)
    return {
        "default": sentinel,
        "parser": sentinel,
        "researcher": sentinel,
        "summarizer": sentinel,
    }


@pytest.mark.asyncio
async def test_run_research_falls_back_to_openrouter(monkeypatch, question, base_llms):
    bot = TemplateForecaster(llms=base_llms, aggregation_strategy=AggregationStrategy.MEAN)

    failing_provider = AsyncMock(side_effect=RuntimeError("primary failure"))
    monkeypatch.setattr(
        bot._research, "_select_research_providers", lambda fast_path=False: [(failing_provider, "asknews")]
    )

    fallback = AsyncMock(return_value="fallback research")
    monkeypatch.setattr(bot._research, "_call_perplexity", fallback)
    monkeypatch.setenv("OPENROUTER_API_KEY", "token")
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    result = await bot.run_research(question)

    assert "fallback research" in result
    assert failing_provider.await_count == 1
    fallback.assert_awaited_once_with(question.question_text, use_open_router=True)


@pytest.mark.asyncio
async def test_run_research_returns_empty_when_all_providers_fail(monkeypatch, question, base_llms):
    """When all providers fail, run_research degrades gracefully: empty forecaster-facing
    text, with the failure durably recorded in the comment-bound diagnostics block
    (popped via the orchestrator seam) for triage."""
    bot = TemplateForecaster(
        llms=base_llms,
        aggregation_strategy=AggregationStrategy.MEAN,
        allow_research_fallback=True,
    )

    failing_provider = AsyncMock(side_effect=RuntimeError("primary failure"))
    monkeypatch.setattr(
        bot._research, "_select_research_providers", lambda fast_path=False: [(failing_provider, "asknews")]
    )

    monkeypatch.setattr(bot._research, "_call_perplexity", AsyncMock(side_effect=RuntimeError("fallback fail")))
    monkeypatch.setattr(bot._research, "_call_exa_smart_searcher", AsyncMock(side_effect=RuntimeError("exa fail")))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    result = await bot.run_research(question)
    # Forecaster-facing text is empty — no provider content and no diagnostics block.
    assert "## News Articles (AskNews)" not in result
    assert "## Provider Diagnostics" not in result
    # The failure is recorded in the comment-bound diagnostics block instead.
    block = bot._research.pop_provider_diagnostics(question.id_of_question)
    assert "## Provider Diagnostics" in block
    assert "- asknews: errored | 0 chars |" in block
    assert "RuntimeError" in block
    assert failing_provider.await_count == 1


class TestPerplexityRetryHardening:
    """Both Perplexity call sites own their retry budget instead of inheriting one.

    Dormant in production — AskNews wins the provider priority ladder, so these only
    run when AskNews credentials are absent or its call fails — but they carry the same
    defect the 2026-07-26 dry-key run exposed elsewhere: a plain ``GeneralLlm`` with no
    ``allowed_tries`` inherits forecasting-tools' default of 2 with an UN-GATED
    ``random.uniform(5, 10)`` tenacity sleep, so a deterministic rejection still costs a
    blind multi-second sleep. Pinning ``allowed_tries=1`` and routing through
    ``invoke_with_transient_retry`` makes the elapsed-gated wrapper the sole retry layer.
    """

    @pytest.mark.asyncio
    async def test_provider_factory_site_pins_tries_and_wraps(self, question):
        captured: dict = {}

        def _capture_llm(**kwargs):
            captured["llm_kwargs"] = kwargs
            llm = MagicMock()
            llm.invoke = AsyncMock(return_value="perplexity prose")
            return llm

        async def _spy(make_awaitable, **kwargs):
            captured["retry_kwargs"] = kwargs
            return await make_awaitable()

        with (
            patch.object(providers, "GeneralLlm", _capture_llm),
            patch.object(providers, "invoke_with_transient_retry", _spy),
        ):
            out = await providers._perplexity_provider()(question)

        assert out == "perplexity prose"
        assert captured["llm_kwargs"]["allowed_tries"] == 1
        assert captured["retry_kwargs"]["label"] == "perplexity_research"
        assert captured["retry_kwargs"]["wall_timeout"] == PERPLEXITY_WALL_TIMEOUT

    @pytest.mark.asyncio
    async def test_orchestrator_site_pins_tries_and_wraps(self, question, base_llms):
        bot = TemplateForecaster(llms=base_llms, aggregation_strategy=AggregationStrategy.MEAN)
        captured: dict = {}

        def _capture_llm(**kwargs):
            captured["llm_kwargs"] = kwargs
            llm = MagicMock()
            llm.invoke = AsyncMock(return_value="perplexity prose")
            return llm

        async def _spy(make_awaitable, **kwargs):
            captured["retry_kwargs"] = kwargs
            return await make_awaitable()

        with (
            patch.object(orchestrator, "GeneralLlm", _capture_llm),
            patch.object(orchestrator, "invoke_with_transient_retry", _spy),
        ):
            out = await bot._research._call_perplexity(question.question_text)

        assert out == "perplexity prose"
        assert captured["llm_kwargs"]["allowed_tries"] == 1
        assert captured["retry_kwargs"]["label"] == "perplexity_research"
        assert captured["retry_kwargs"]["wall_timeout"] == PERPLEXITY_WALL_TIMEOUT

    @pytest.mark.asyncio
    async def test_hard_403_costs_one_attempt(self, question):
        """A drained-key 403 is terminal here too — no ladder, no blind sleep."""
        attempts = {"n": 0}

        def _dead_llm(**_kwargs):
            async def _invoke(_prompt):
                attempts["n"] += 1
                raise litellm.exceptions.APIError(
                    status_code=403,
                    message='{"error":{"message":"Key limit exceeded (total limit).","code":403}}',
                    llm_provider="openrouter",
                    model=PERPLEXITY_RESEARCH_MODEL,
                )

            llm = MagicMock()
            llm.invoke = _invoke
            return llm

        with patch.object(providers, "GeneralLlm", _dead_llm), pytest.raises(litellm.exceptions.APIError):
            await providers._perplexity_provider()(question)

        assert attempts["n"] == 1


class TestPerplexityModelIsOneConstant:
    """Both Perplexity call sites must resolve to the SAME model.

    They each carried their own literal until 2026-07-27 and silently drifted:
    the provider factory was pinned to Perplexity's non-reasoning tier while the
    orchestrator's AskNews-failure fallback used the reasoning tier, so which
    model answered a research call depended on which code path got there. Both
    now read ``PERPLEXITY_RESEARCH_MODEL`` from ``constants.py``; these tests
    pin that so a future edit to one site can't re-open the gap.
    """

    @staticmethod
    def _capture_llm(captured: dict):
        def _factory(**kwargs):
            captured["llm_kwargs"] = kwargs
            llm = MagicMock()
            llm.invoke = AsyncMock(return_value="perplexity prose")
            return llm

        return _factory

    @staticmethod
    async def _passthrough(make_awaitable, **_kwargs):
        return await make_awaitable()

    def test_openrouter_route_is_the_direct_slug_prefixed(self):
        # One model, two routes: the OpenRouter form must be the direct slug with
        # the ``openrouter/`` prefix that get_openrouter_api_key routes on, not an
        # independently-chosen model.
        assert f"openrouter/{PERPLEXITY_RESEARCH_MODEL}" == PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("use_open_router", "expected_model"),
        [
            (False, PERPLEXITY_RESEARCH_MODEL),
            (True, PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER),
        ],
    )
    async def test_provider_factory_site_uses_the_constant(self, question, use_open_router, expected_model):
        captured: dict = {}
        with (
            patch.object(providers, "GeneralLlm", self._capture_llm(captured)),
            patch.object(providers, "invoke_with_transient_retry", self._passthrough),
        ):
            await providers._perplexity_provider(use_open_router=use_open_router)(question)

        assert captured["llm_kwargs"]["model"] == expected_model

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("use_open_router", "expected_model"),
        [
            (False, PERPLEXITY_RESEARCH_MODEL),
            (True, PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER),
        ],
    )
    async def test_orchestrator_site_uses_the_constant(self, question, base_llms, use_open_router, expected_model):
        bot = TemplateForecaster(llms=base_llms, aggregation_strategy=AggregationStrategy.MEAN)
        captured: dict = {}
        with (
            patch.object(orchestrator, "GeneralLlm", self._capture_llm(captured)),
            patch.object(orchestrator, "invoke_with_transient_retry", self._passthrough),
        ):
            await bot._research._call_perplexity(question.question_text, use_open_router=use_open_router)

        assert captured["llm_kwargs"]["model"] == expected_model


class TestPerplexityMarketOddsPolicy:
    """Both Perplexity prompts must carry the SAME narrowed market-odds ask the first-pass
    web-research prompt does, from the one definition in ``prompts``.

    The narrowing (2026-09-01, operator-confirmed wording) points research AWAY from
    Polymarket / Kalshi / Manifold / PredictIt, because the live snapshot covers those four and
    the only measured harm from searching them was stale prices contradicting correct snapshot
    rows. It landed on ``web_research_prompt`` alone; these two sites kept asking for "all
    relevant prediction markets" afterwards. Dormant in prod, where AskNews credentials are
    always present and Perplexity never runs — but Perplexity IS the primary provider the moment
    they are absent, which is exactly when nobody is watching the prompt.
    """

    @staticmethod
    def _capture_prompt(captured: dict):
        def _factory(**_kwargs):
            async def _invoke(prompt: str) -> str:
                captured["prompt"] = prompt
                return "perplexity prose"

            llm = MagicMock()
            llm.invoke = _invoke
            return llm

        return _factory

    @staticmethod
    async def _passthrough(make_awaitable, **_kwargs):
        return await make_awaitable()

    async def _provider_prompt(self, question, *, is_benchmarking: bool) -> str:
        captured: dict = {}
        with (
            patch.object(providers, "GeneralLlm", self._capture_prompt(captured)),
            patch.object(providers, "invoke_with_transient_retry", self._passthrough),
        ):
            await providers._perplexity_provider(is_benchmarking=is_benchmarking)(question)
        return captured["prompt"]

    async def _orchestrator_prompt(self, question, base_llms, *, is_benchmarking: bool) -> str:
        bot = TemplateForecaster(
            llms=base_llms,
            aggregation_strategy=AggregationStrategy.MEAN,
            is_benchmarking=is_benchmarking,
        )
        captured: dict = {}
        with (
            patch.object(orchestrator, "GeneralLlm", self._capture_prompt(captured)),
            patch.object(orchestrator, "invoke_with_transient_retry", self._passthrough),
        ):
            await bot._research._call_perplexity(question.question_text)
        return captured["prompt"]

    @pytest.mark.asyncio
    async def test_provider_factory_site_carries_the_shared_policy(self, question):
        prompt = await self._provider_prompt(question, is_benchmarking=False)

        assert OUTSIDE_VENUE_MARKET_ODDS_POLICY in prompt
        assert "consider all relevant prediction markets" not in prompt

    @pytest.mark.asyncio
    async def test_orchestrator_site_carries_the_shared_policy(self, question, base_llms):
        prompt = await self._orchestrator_prompt(question, base_llms, is_benchmarking=False)

        assert OUTSIDE_VENUE_MARKET_ODDS_POLICY in prompt
        assert "briefly research prediction markets" not in prompt

    @pytest.mark.asyncio
    async def test_the_orchestrator_keeps_its_no_speculation_tail(self, question, base_llms):
        """Its own anti-fabrication rule about an EMPTY result, not a second market policy: an
        empty answer has to read as empty rather than as invented odds."""
        prompt = await self._orchestrator_prompt(question, base_llms, is_benchmarking=False)

        assert "DO NOT speculate what they would say" in prompt

    @pytest.mark.asyncio
    async def test_benchmarking_keeps_market_talk_out_of_both_prompts(self, question, base_llms):
        """The leakage carve-out is unchanged by the narrowing: a backtest sees no market ask at
        all, on either route."""
        provider_prompt = await self._provider_prompt(question, is_benchmarking=True)
        orchestrator_prompt = await self._orchestrator_prompt(question, base_llms, is_benchmarking=True)

        for prompt in (provider_prompt, orchestrator_prompt):
            assert OUTSIDE_VENUE_MARKET_ODDS_POLICY not in prompt
            assert "Market-implied or crowd odds" not in prompt
            assert "prediction market" not in prompt.lower()

    @pytest.mark.asyncio
    async def test_the_clean_indents_block_still_dedents(self, question, base_llms):
        """The reason the policy is interpolated as a one-line sentence rather than as the FOCUS
        AREAS bullet: the orchestrator's prompt body is one ``clean_indents`` block, and an
        interpolated value carrying a newline would put a column-0 line into it and leave every
        other line indented by 12 spaces."""
        prompt = await self._orchestrator_prompt(question, base_llms, is_benchmarking=False)

        assert not any(line.startswith(" ") for line in prompt.split("\n")), prompt
