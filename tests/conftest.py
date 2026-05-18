from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools import NumericQuestion


@pytest.fixture
def mock_os_getenv():
    with patch("os.getenv") as mock_getenv:
        yield mock_getenv


def make_mock_numeric_question(
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 100.0,
    open_lower_bound: bool = False,
    open_upper_bound: bool = False,
    zero_point: float | None = None,
    cdf_size: int | None = None,
    id_of_question: int = 42,
    page_url: str | None = None,
    question_text: str = "What will X be?",
    background_info: str = "bg",
    resolution_criteria: str = "rc",
    fine_print: str = "",
    unit_of_measure: str = "USD",
    nominal_lower_bound: float | None = None,
    nominal_upper_bound: float | None = None,
    id_of_post: int | None = None,
    with_open_resolve_times: bool = False,
) -> MagicMock:
    """Return a ``MagicMock(spec=NumericQuestion)`` with all common fields populated.

    Centralizes the small differences that used to live in ~8 inline helpers across
    the test suite. Field defaults match the most common shape (closed [0, 100] in
    USD with question id 42); per-test overrides land via keyword args.

    ``with_open_resolve_times=True`` populates ``open_time`` (now − 30d) and
    ``scheduled_resolution_time`` (now + 365d), required by helpers that call
    ``_forecasting_window_str``.
    """
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = id_of_question
    q.id_of_post = id_of_post if id_of_post is not None else id_of_question
    q.page_url = page_url if page_url is not None else f"https://example.com/q/{id_of_question}"
    q.question_text = question_text
    q.background_info = background_info
    q.resolution_criteria = resolution_criteria
    q.fine_print = fine_print
    q.unit_of_measure = unit_of_measure
    q.lower_bound = lower_bound
    q.upper_bound = upper_bound
    q.open_lower_bound = open_lower_bound
    q.open_upper_bound = open_upper_bound
    q.zero_point = zero_point
    q.cdf_size = cdf_size
    q.nominal_lower_bound = nominal_lower_bound
    q.nominal_upper_bound = nominal_upper_bound
    if with_open_resolve_times:
        q.open_time = datetime.now() - timedelta(days=30)
        q.scheduled_resolution_time = datetime.now() + timedelta(days=365)
    return q


@pytest.fixture
def make_mock_numeric_q():
    """Pytest-fixture wrapper around ``make_mock_numeric_question``."""
    return make_mock_numeric_question


@pytest.fixture(autouse=True)
def _clear_gemini_client_cache():
    """Clear the module-global genai.Client lru_cache between tests.

    The Gemini provider caches one client per API key via functools.lru_cache;
    without clearing, a test that mocks genai.Client will see a stale cached
    mock from an earlier test that used a different mock.

    Autouse-global because the gemini client cache is process-wide and can
    pollute even unrelated tests if any prior test loads the module (e.g. via
    a transitive import in main.py / research_providers.py). Scoping to
    gemini-named tests would miss those indirect-load cases. The clear is
    cheap (a single ``cache_clear`` on a 1-entry lru_cache) so the per-test
    cost is negligible — leaving the autouse global is the simpler,
    safer choice.
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
