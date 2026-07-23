import socket
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

_OPEN = datetime(2026, 1, 1)
_RESOLVE = datetime(2026, 5, 1)


# ---------------------------------------------------------------------------
# Network-egress guard (money-safety backstop)
# ---------------------------------------------------------------------------

# Hosts a socket may connect to without tripping the guard. Loopback only —
# everything else is a real host and must be stubbed by the test. AF_UNIX and
# socket.socketpair() are allowed unconditionally (they never carry an INET
# address); asyncio's self-pipe / event-loop internals rely on them, so blocking
# them would wedge the whole suite.
_ALLOWED_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost", "0.0.0.0", "::"})


def _host_from_address(address: Any) -> str | None:
    """Pull the host out of a connect() address (INET = (host, port), INET6 adds flow/scope)."""
    if isinstance(address, tuple) and address:
        return address[0]
    return None


def _address_is_blocked(family: int, address: Any) -> bool:
    """True iff this is an AF_INET/AF_INET6 connect to a non-loopback host."""
    if family not in (socket.AF_INET, socket.AF_INET6):
        # AF_UNIX and everything else (socketpair, unix domain sockets) is fine.
        return False
    host = _host_from_address(address)
    if host is None:
        return False
    return host not in _ALLOWED_HOSTS


@pytest.fixture(autouse=True)
def _block_network_egress(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Block all real network egress during the test suite (money-safety backstop).

    Monkeypatches ``socket.socket.connect`` / ``connect_ex`` to raise a clear
    ``RuntimeError`` for any AF_INET/AF_INET6 connect to a non-localhost host, so
    no test can silently reach a paid API. Modeled on how ``pytest-socket`` gates
    (localhost + AF_UNIX + socketpair allowed, real hosts blocked) without taking
    the dependency.

    Opt out with ``@pytest.mark.allow_network`` on a test that must reach a real
    host (the ``live`` suite is already deselected by ``addopts = -m 'not live'``,
    so this marker is a belt-and-suspenders escape hatch, not the primary gate).

    See also ``metaculus_bot.ablation.offline_replay.no_network()`` — a scoped
    context manager (ablation replay only) that blocks ``socket.getaddrinfo`` at
    the DNS level; this autouse guard is complementary, blocking ``connect`` /
    ``connect_ex`` at the socket level so it also catches literal-IP connects that
    skip DNS resolution entirely. Different scopes on purpose; don't consolidate.
    """
    if request.node.get_closest_marker("allow_network") is not None:
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address: Any) -> Any:
        if _address_is_blocked(self.family, address):
            raise RuntimeError(
                f"Network access blocked in tests: {address}. "
                "Stub the client or mark the test @pytest.mark.allow_network."
            )
        return real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> Any:
        if _address_is_blocked(self.family, address):
            raise RuntimeError(
                f"Network access blocked in tests: {address}. "
                "Stub the client or mark the test @pytest.mark.allow_network."
            )
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)


def make_mock_binary_question(qid: int = 1001) -> MagicMock:
    """Return a ``MagicMock(spec=BinaryQuestion)`` with standard fields populated."""
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = qid
    q.question_text = "Will it rain?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = f"https://example.com/q/{qid}"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


def make_mock_mc_question(qid: int = 1002, options: list[str] | None = None) -> MagicMock:
    """Return a ``MagicMock(spec=MultipleChoiceQuestion)`` with configurable options."""
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.id_of_question = qid
    q.question_text = "Which color?"
    q.options = options if options is not None else ["Red", "Blue", "Green"]
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = f"https://example.com/q/{qid}"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


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
def _enable_per_type_stacking(monkeypatch):
    """Force the per-type stacking gates ON for the whole test suite.

    Production defaults all three ``<TYPE>_STACKING_ENABLED`` flags to DISABLED
    (the stacker only runs when a deploy explicitly opts in). Most stacking
    tests, however, exist to exercise the stacking MECHANISM (crux extraction,
    targeted search, aggregation, fallbacks, thresholds) — they assume the
    stacker is reachable. Setting the flags here keeps those tests faithful to
    their intent without each having to opt in.

    Tests that assert the production DEFAULT (off-when-unset) or a specific
    polarity override their flag in the test body via ``monkeypatch.delenv`` /
    ``monkeypatch.setenv``; that runs after this setup fixture, so the later
    value wins.
    """
    monkeypatch.setenv("BINARY_STACKING_ENABLED", "true")
    monkeypatch.setenv("MC_STACKING_ENABLED", "true")
    monkeypatch.setenv("NUMERIC_STACKING_ENABLED", "true")


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
    from metaculus_bot.research import gemini_search as gsp

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
