from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_os_getenv():
    with patch("os.getenv") as mock_getenv:
        yield mock_getenv


@pytest.fixture(autouse=True)
def _clear_gemini_client_cache():
    """Clear the module-global genai.Client lru_cache between tests.

    The Gemini provider caches one client per API key via functools.lru_cache;
    without clearing, a test that mocks genai.Client will see a stale cached
    mock from an earlier test that used a different mock.
    """
    from metaculus_bot import gemini_search_provider as gsp

    gsp._cached_client_for_key.cache_clear()
    yield
    gsp._cached_client_for_key.cache_clear()


@pytest.fixture
def test_llms():
    """Shared LLM config with a mock default and real parser/researcher/summarizer."""
    from metaculus_bot.llm_configs import PARSER_LLM, RESEARCHER_LLM, SUMMARIZER_LLM

    return {
        "default": MagicMock(),
        "parser": PARSER_LLM,
        "researcher": RESEARCHER_LLM,
        "summarizer": SUMMARIZER_LLM,
    }


def _build_mock_question(
    *,
    question_id: int,
    question_text: str,
    resolution_criteria: str | None = None,
    fine_print: str | None = None,
) -> MagicMock:
    question = MagicMock()
    question.id_of_question = question_id
    question.question_text = question_text
    question.page_url = f"https://example.com/q/{question_id}"
    if resolution_criteria is not None:
        question.resolution_criteria = resolution_criteria
    if fine_print is not None:
        question.fine_print = fine_print
    return question


@pytest.fixture
def make_mock_question():
    """Factory for building mock MetaculusQuestion objects with configurable fields."""
    return _build_mock_question
