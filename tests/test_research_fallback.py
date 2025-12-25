from unittest.mock import AsyncMock, MagicMock

import pytest
from forecasting_tools import GeneralLlm

from main import TemplateForecaster
from metaculus_bot.aggregation_strategies import AggregationStrategy


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
    # Patch the new _select_research_providers method
    monkeypatch.setattr(bot, "_select_research_providers", lambda: [(failing_provider, "asknews")])

    fallback = AsyncMock(return_value="fallback research")
    monkeypatch.setattr(bot, "_call_perplexity", fallback)
    monkeypatch.setenv("OPENROUTER_API_KEY", "token")
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    result = await bot.run_research(question)

    # Result now includes provider header for non-empty fallback
    assert "fallback research" in result
    assert failing_provider.await_count == 1
    fallback.assert_awaited_once_with(question.question_text, use_open_router=True)


@pytest.mark.asyncio
async def test_run_research_returns_empty_when_all_providers_fail(monkeypatch, question, base_llms):
    """When all providers fail, run_research returns empty string (graceful degradation)."""
    bot = TemplateForecaster(
        llms=base_llms,
        aggregation_strategy=AggregationStrategy.MEAN,
        allow_research_fallback=True,
    )

    failing_provider = AsyncMock(side_effect=RuntimeError("primary failure"))
    # Patch the new _select_research_providers method
    monkeypatch.setattr(bot, "_select_research_providers", lambda: [(failing_provider, "asknews")])

    monkeypatch.setattr(bot, "_call_perplexity", AsyncMock(side_effect=RuntimeError("fallback fail")))
    monkeypatch.setattr(bot, "_call_exa_smart_searcher", AsyncMock(side_effect=RuntimeError("exa fail")))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    # With parallel provider execution, failures are handled gracefully
    # and return empty string rather than raising
    result = await bot.run_research(question)
    assert result == ""
    assert failing_provider.await_count == 1
