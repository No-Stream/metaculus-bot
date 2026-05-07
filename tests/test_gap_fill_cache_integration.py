"""Integration test: cache stores research with the gap-fill addendum appended.

When GAP_FILL_ENABLED=true and run_gap_fill_pass returns content, the cached
value must include both the gap-fill addendum and the section header that
``run_research`` prepends to it.
"""

from unittest.mock import AsyncMock, patch

import pytest

from main import TemplateForecaster


@pytest.fixture
def mock_question(make_mock_question):
    return make_mock_question(
        question_id=77777,
        question_text="Will GDP grow in Q4?",
        resolution_criteria="Any positive real growth.",
        fine_print="BEA advance estimate.",
    )


@pytest.mark.asyncio
async def test_cache_stores_research_with_gap_fill_addendum(
    mock_question,
    test_llms,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With GAP_FILL_ENABLED=true, the cached research blob includes the addendum + header."""
    monkeypatch.setenv("GAP_FILL_ENABLED", "true")
    # Avoid activating other paid providers.
    monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
    monkeypatch.delenv("NATIVE_SEARCH_ENABLED", raising=False)
    monkeypatch.delenv("GEMINI_SEARCH_ENABLED", raising=False)

    shared_cache: dict[int, str] = {}

    bot = TemplateForecaster(
        is_benchmarking=True,
        research_cache=shared_cache,
        llms=test_llms,
    )
    bot._custom_research_provider = AsyncMock(
        return_value="first-pass research blob with enough substance to clear the length guard: "
        + ("lorem ipsum dolor sit amet " * 10)
    )

    # Mock run_gap_fill_pass at the point where main.py imports it (inside run_research).
    with patch(
        "metaculus_bot.targeted_research.run_gap_fill_pass",
        AsyncMock(return_value="MOCK_ADDENDUM"),
    ):
        result = await bot.run_research(mock_question)

    # Cached value should contain first-pass research, the gap-fill header, and the addendum.
    assert 77777 in shared_cache
    cached = shared_cache[77777]
    assert "first-pass research blob" in cached
    assert "## Targeted Gap-Fill (second pass)" in cached
    assert "MOCK_ADDENDUM" in cached

    # The returned result (== cached, since this is the initial store) likewise has it.
    assert "MOCK_ADDENDUM" in result
    assert "## Targeted Gap-Fill (second pass)" in result


@pytest.mark.asyncio
async def test_gap_fill_disabled_by_default(
    mock_question,
    test_llms,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without GAP_FILL_ENABLED set, ``run_gap_fill_pass`` must not be invoked."""
    monkeypatch.delenv("GAP_FILL_ENABLED", raising=False)
    monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
    monkeypatch.delenv("NATIVE_SEARCH_ENABLED", raising=False)
    monkeypatch.delenv("GEMINI_SEARCH_ENABLED", raising=False)

    bot = TemplateForecaster(
        is_benchmarking=True,
        research_cache={},
        llms=test_llms,
    )
    bot._custom_research_provider = AsyncMock(
        return_value="first-pass research blob with enough substance to clear the length guard: "
        + ("lorem ipsum dolor sit amet " * 10)
    )

    fake_gap_fill = AsyncMock(return_value="SHOULD_NOT_APPEAR")
    with patch("metaculus_bot.targeted_research.run_gap_fill_pass", fake_gap_fill):
        result = await bot.run_research(mock_question)

    fake_gap_fill.assert_not_called()
    assert "SHOULD_NOT_APPEAR" not in result
    assert "## Targeted Gap-Fill" not in result


@pytest.mark.asyncio
async def test_gap_fill_skipped_when_first_pass_is_tiny(
    mock_question,
    test_llms,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When first-pass research is below GAP_FILL_MIN_RESEARCH_CHARS, skip gap-fill.

    Guards against the failure mode where every first-pass provider soft-failed
    and emitted only its header line — we'd otherwise burn an analyzer + grounded-search
    budget trying to find gaps in effectively-empty research.
    """
    monkeypatch.setenv("GAP_FILL_ENABLED", "true")
    monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
    monkeypatch.delenv("NATIVE_SEARCH_ENABLED", raising=False)
    monkeypatch.delenv("GEMINI_SEARCH_ENABLED", raising=False)

    bot = TemplateForecaster(
        is_benchmarking=True,
        research_cache={},
        llms=test_llms,
    )
    # Short blob — provider header + a few words, well below GAP_FILL_MIN_RESEARCH_CHARS=200.
    bot._custom_research_provider = AsyncMock(return_value="tiny blob")

    fake_gap_fill = AsyncMock(return_value="SHOULD_NOT_APPEAR")
    with patch("metaculus_bot.targeted_research.run_gap_fill_pass", fake_gap_fill):
        result = await bot.run_research(mock_question)

    fake_gap_fill.assert_not_called()
    assert "SHOULD_NOT_APPEAR" not in result
    assert "## Targeted Gap-Fill" not in result


@pytest.mark.asyncio
async def test_gap_fill_empty_return_does_not_add_addendum(
    mock_question,
    test_llms,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``run_gap_fill_pass`` returns "" (its soft-fail signal), research is unchanged.

    ``run_gap_fill_pass`` owns the never-raises contract — on any upstream failure
    (timeout, API error, missing key, network) it logs and returns "". The caller
    in main.py relies on this: no try/except wrapper exists around the call. This
    test verifies the integration: empty return → no addendum header, cached
    research is just the first-pass blob.
    """
    monkeypatch.setenv("GAP_FILL_ENABLED", "true")
    monkeypatch.delenv("FINANCIAL_DATA_ENABLED", raising=False)
    monkeypatch.delenv("NATIVE_SEARCH_ENABLED", raising=False)
    monkeypatch.delenv("GEMINI_SEARCH_ENABLED", raising=False)

    bot = TemplateForecaster(
        is_benchmarking=True,
        research_cache={},
        llms=test_llms,
    )
    bot._custom_research_provider = AsyncMock(
        return_value="first-pass research blob with enough substance to clear the length guard: "
        + ("lorem ipsum dolor sit amet " * 10)
    )

    with patch(
        "metaculus_bot.targeted_research.run_gap_fill_pass",
        AsyncMock(return_value=""),
    ):
        result = await bot.run_research(mock_question)

    # Research still came back, without the gap-fill section.
    assert "first-pass research blob" in result
    assert "## Targeted Gap-Fill" not in result
