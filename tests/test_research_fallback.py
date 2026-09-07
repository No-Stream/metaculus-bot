import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import litellm.exceptions
import pytest
from forecasting_tools import GeneralLlm, MetaculusQuestion

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy
from metaculus_bot.constants import (
    PERPLEXITY_RESEARCH_MODEL,
    PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER,
    PERPLEXITY_WALL_TIMEOUT,
)
from metaculus_bot.prompts import OUTSIDE_VENUE_MARKET_ODDS_POLICY
from metaculus_bot.research import orchestrator, providers

_FACTORY_EXA_PROMPT = (
    "You are an assistant to a superforecaster. The superforecaster will give"
    " you a question they intend to forecast on. To be a great assistant, you generate"
    " a concise but detailed rundown of the most relevant news, including if the question"
    " would resolve Yes or No based on current information. You do not produce forecasts yourself."
    "\n\nThe question is: Sample question?"
)
_ORCHESTRATOR_EXA_PROMPT = (
    "You are an assistant to a superforecaster. The superforecaster will give"
    "you a question they intend to forecast on. To be a great assistant, you generate"
    "a concise but detailed rundown of the most relevant news, including if the question"
    "would resolve Yes or No based on current information. You do not produce forecasts yourself."
    "\n\nThe question is: Sample question?"
)


def _factory_perplexity_prompt(*, is_benchmarking: bool) -> str:
    prediction_markets_instruction = (
        "" if is_benchmarking else f"In addition to news, cover: {OUTSIDE_VENUE_MARKET_ODDS_POLICY}\n"
    )
    return (
        "You are an assistant to a superforecaster.\n"
        "Generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.\n"
        f"{prediction_markets_instruction}"
        "Do not produce forecasts yourself. Provide data for the superforecaster.\n\n"
        "Question:\nSample question?"
    )


def _orchestrator_perplexity_prompt(*, is_benchmarking: bool) -> str:
    prediction_markets_instruction = (
        ""
        if is_benchmarking
        else (
            f"In addition to news, cover: {OUTSIDE_VENUE_MARKET_ODDS_POLICY} "
            "(If there are no relevant markets of that kind, simply skip reporting on this and "
            "DO NOT speculate what they would say.)"
        )
    )
    return (
        "\nYou are an assistant to a superforecaster.\n"
        "The superforecaster will give you a question they intend to forecast on.\n"
        "To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.\n"
        f"{prediction_markets_instruction}\n"
        "You DO NOT produce forecasts yourself; you must provide ALL relevant data to the superforecaster so they can make an expert judgment.\n\n"
        "Question:\nSample question?\n"
    )


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


class TestProviderInvocationContracts:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("site", "expected_prompt", "expected_citation_kwargs"),
        [
            (
                "factory",
                _FACTORY_EXA_PROMPT,
                {"include_works_cited_list": False, "use_brackets_around_citations": False},
            ),
            ("orchestrator", _ORCHESTRATOR_EXA_PROMPT, {}),
        ],
    )
    async def test_exa_prompt_and_constructor_shape_are_exact(
        self,
        question,
        base_llms,
        site: str,
        expected_prompt: str,
        expected_citation_kwargs: dict[str, bool],
    ) -> None:
        captured: dict = {}
        searcher = MagicMock()
        searcher.invoke = AsyncMock(return_value="exa prose")

        def capture_searcher(**kwargs):
            captured["searcher_kwargs"] = kwargs
            return searcher

        with patch.object(providers, "SmartSearcher", capture_searcher):
            if site == "factory":
                output = await providers._exa_provider(base_llms["default"])(question)
            else:
                bot = TemplateForecaster(llms=base_llms, aggregation_strategy=AggregationStrategy.MEAN)
                output = await bot._research._call_exa_smart_searcher(question.question_text)

        assert output == "exa prose"
        assert captured["searcher_kwargs"] == {
            "model": base_llms["default"],
            "temperature": None,
            "num_searches_to_run": 2,
            "num_sites_per_search": 10,
            **expected_citation_kwargs,
        }
        searcher.invoke.assert_awaited_once_with(expected_prompt)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("site", ["factory", "orchestrator"])
    @pytest.mark.parametrize("use_open_router", [False, True])
    @pytest.mark.parametrize("is_benchmarking", [False, True])
    async def test_perplexity_prompt_constructor_and_retry_shape_are_exact(
        self,
        question,
        base_llms,
        site: str,
        use_open_router: bool,
        is_benchmarking: bool,
    ) -> None:
        captured: dict = {}
        model = MagicMock()
        model.invoke = AsyncMock(return_value="perplexity prose")
        expected_model = PERPLEXITY_RESEARCH_MODEL_VIA_OPENROUTER if use_open_router else PERPLEXITY_RESEARCH_MODEL

        def capture_llm(**kwargs):
            captured["llm_kwargs"] = kwargs
            return model

        async def capture_retry(make_awaitable, **kwargs):
            captured["retry_kwargs"] = kwargs
            return await make_awaitable()

        with (
            patch.object(providers, "GeneralLlm", capture_llm),
            patch.object(providers, "invoke_with_transient_retry", capture_retry),
            patch.object(orchestrator, "get_openrouter_api_key", return_value="resolved-key") as get_key,
        ):
            if site == "factory":
                output = await providers._perplexity_provider(
                    use_open_router=use_open_router,
                    is_benchmarking=is_benchmarking,
                )(question)
                expected_prompt = _factory_perplexity_prompt(is_benchmarking=is_benchmarking)
            else:
                bot = TemplateForecaster(
                    llms=base_llms,
                    aggregation_strategy=AggregationStrategy.MEAN,
                    is_benchmarking=is_benchmarking,
                )
                output = await bot._research._call_perplexity(
                    question.question_text,
                    use_open_router=use_open_router,
                )
                expected_prompt = _orchestrator_perplexity_prompt(is_benchmarking=is_benchmarking)

        assert output == "perplexity prose"
        expected_llm_kwargs = {
            "model": expected_model,
            "temperature": None,
            "allowed_tries": 1,
            "metadata": {
                "role": "perplexity_research",
                "key_alias": "personal" if use_open_router else "direct",
            },
        }
        if site == "orchestrator":
            expected_llm_kwargs["api_key"] = "resolved-key" if use_open_router else None
        assert captured["llm_kwargs"] == expected_llm_kwargs
        assert captured["retry_kwargs"] == {
            "wall_timeout": PERPLEXITY_WALL_TIMEOUT,
            "label": "perplexity_research",
        }
        model.invoke.assert_awaited_once_with(expected_prompt)
        if site == "orchestrator" and use_open_router:
            get_key.assert_called_once_with(expected_model)
        else:
            get_key.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider_name", ["exa", "perplexity"])
    async def test_orchestrator_provider_wrappers_accept_question_objects_and_text(
        self,
        base_llms,
        provider_name: str,
    ) -> None:
        bot = TemplateForecaster(llms=base_llms, aggregation_strategy=AggregationStrategy.MEAN)
        question = MetaculusQuestion(question_text="Sample question?", page_url="https://example.com/questions/1")

        if provider_name == "exa":
            with patch.object(
                orchestrator, "_invoke_exa_research", new_callable=AsyncMock, return_value="research"
            ) as invoke:
                assert await bot._research._call_exa_smart_searcher(question) == "research"
                assert await bot._research._call_exa_smart_searcher(question.question_text) == "research"
            assert invoke.await_args_list[0] == invoke.await_args_list[1]
        else:
            with patch.object(
                orchestrator,
                "_invoke_perplexity_research",
                new_callable=AsyncMock,
                return_value="research",
            ) as invoke:
                assert await bot._research._call_perplexity(question, use_open_router=False) == "research"
                assert await bot._research._call_perplexity(question.question_text, use_open_router=False) == "research"
            assert invoke.await_args_list[0] == invoke.await_args_list[1]

    @pytest.mark.asyncio
    async def test_perplexity_retry_receives_a_fresh_awaitable_factory(self, question) -> None:
        model = MagicMock()
        model.invoke = AsyncMock(side_effect=["first attempt", "second attempt"])

        async def invoke_twice(make_awaitable, **_kwargs):
            await make_awaitable()
            return await make_awaitable()

        with (
            patch.object(providers, "GeneralLlm", return_value=model),
            patch.object(providers, "invoke_with_transient_retry", invoke_twice),
        ):
            output = await providers._perplexity_provider()(question)

        assert output == "second attempt"
        assert model.invoke.await_count == 2


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


class TestAskNewsFallbackOrder:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("present_key", "expected_provider", "expected_openrouter"),
        [
            ("OPENROUTER_API_KEY", "openrouter", True),
            ("PERPLEXITY_API_KEY", "perplexity", False),
            ("EXA_API_KEY", "exa", None),
        ],
    )
    async def test_each_credential_selects_its_existing_fallback_rung(
        self,
        monkeypatch,
        base_llms,
        present_key: str,
        expected_provider: str,
        expected_openrouter: bool | None,
    ) -> None:
        for key in ("OPENROUTER_API_KEY", "PERPLEXITY_API_KEY", "EXA_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv(present_key, "key")
        bot = TemplateForecaster(llms=base_llms, aggregation_strategy=AggregationStrategy.MEAN)

        with (
            patch.object(bot._research, "_call_perplexity", new_callable=AsyncMock, return_value="perplexity") as pplx,
            patch.object(bot._research, "_call_exa_smart_searcher", new_callable=AsyncMock, return_value="exa") as exa,
        ):
            result = await bot._research._attempt_research_fallback("Sample question?")

        assert result == (("exa", "exa") if expected_provider == "exa" else ("perplexity", expected_provider))
        if expected_provider == "exa":
            exa.assert_awaited_once_with("Sample question?")
            pplx.assert_not_awaited()
        else:
            pplx.assert_awaited_once_with("Sample question?", use_open_router=expected_openrouter)
            exa.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_first_present_fallback_failure_does_not_cascade(self, monkeypatch, base_llms) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
        monkeypatch.setenv("PERPLEXITY_API_KEY", "perplexity-key")
        monkeypatch.setenv("EXA_API_KEY", "exa-key")
        bot = TemplateForecaster(llms=base_llms, aggregation_strategy=AggregationStrategy.MEAN)

        perplexity = AsyncMock(side_effect=RuntimeError("selected fallback failed"))
        exa = AsyncMock(return_value="must not run")
        with (
            patch.object(bot._research, "_call_perplexity", perplexity),
            patch.object(bot._research, "_call_exa_smart_searcher", exa),
        ):
            result = await bot._research._attempt_research_fallback("Sample question?")

        assert result == (None, None)
        perplexity.assert_awaited_once_with("Sample question?", use_open_router=True)
        exa.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_cancellation_propagates(self, monkeypatch, base_llms) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
        bot = TemplateForecaster(llms=base_llms, aggregation_strategy=AggregationStrategy.MEAN)

        with (
            patch.object(bot._research, "_call_perplexity", new_callable=AsyncMock, side_effect=asyncio.CancelledError),
            pytest.raises(asyncio.CancelledError),
        ):
            await bot._research._attempt_research_fallback("Sample question?")


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
            patch.object(providers, "GeneralLlm", _capture_llm),
            patch.object(providers, "invoke_with_transient_retry", _spy),
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
            patch.object(providers, "GeneralLlm", self._capture_llm(captured)),
            patch.object(providers, "invoke_with_transient_retry", self._passthrough),
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
            patch.object(providers, "GeneralLlm", self._capture_prompt(captured)),
            patch.object(providers, "invoke_with_transient_retry", self._passthrough),
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
