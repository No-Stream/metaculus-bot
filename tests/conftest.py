import socket
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from scripts import gha_artifacts

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

    Two independent markers skip the guard:

    - ``@pytest.mark.allow_network`` — a normally-selected test that must reach a
      real host (belt-and-suspenders escape hatch).
    - ``@pytest.mark.live`` — the live suite (e.g. ``tests/test_smoke_real_llm.py``),
      which makes real API calls by design and carries no ``allow_network`` marker.

    Deselection and exemption are separate concerns. ``addopts = -m 'not live'``
    DESELECTS the live suite so a plain ``make test`` never runs it (and it never
    reaches this guard). ``make test_live`` (``pytest -m live``) re-selects it, at
    which point this exemption is what keeps the guard from blocking the real API
    calls those tests exist to make. So the ``live`` exemption is load-bearing for
    ``make test_live``, not merely belt-and-suspenders.

    See also ``metaculus_bot.ablation.offline_replay.no_network()`` — a scoped
    context manager (ablation replay only) that blocks ``socket.getaddrinfo`` at
    the DNS level; this autouse guard is complementary, blocking ``connect`` /
    ``connect_ex`` at the socket level so it also catches literal-IP connects that
    skip DNS resolution entirely. Different scopes on purpose; don't consolidate.
    """
    if request.node.get_closest_marker("allow_network") is not None:
        return
    if request.node.get_closest_marker("live") is not None:
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


# ---------------------------------------------------------------------------
# Persisted-artifact-store guard (data-safety backstop)


@pytest.fixture(autouse=True)
def _redirect_artifact_store(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the persisted artifact store at a temp dir for every test.

    ``backtests/gha_artifact_store/`` is the durable local copy of GHA artifacts that
    expire at 90 days, so a test writing into it plants fake data in real research
    evidence. Not hypothetical: a run-log test that simply omitted ``store_dir`` persisted
    a fixture artifact named ``research-2`` there, and the next real offline harvest
    ingested its one-line log into the live telemetry archive as an ``unknown`` run.

    ``scripts.gha_artifacts._resolve_store_dir`` reads this module attribute at CALL time
    precisely so this redirect works; a signature default would be captured at import and
    leave the omission unprotected. Tests that pass ``store_dir`` explicitly are
    unaffected.
    """
    monkeypatch.setattr(gha_artifacts, "DEFAULT_STORE_DIR", str(tmp_path_factory.mktemp("gha_artifact_store")))


# Shared failure fixtures
# ---------------------------------------------------------------------------

# The verbatim OpenRouter response that cost the 2026-07-26 tournament run two of three
# forecasters and most of the research stack. Copied character-for-character from the run
# log (the donated key at $0.00 of its $850 cap): HTTP 403 rather than the 402 OpenRouter's
# docs promise, the phrase "Key limit exceeded (total limit)", and a ``"code":403`` field in
# the body — which is what the old negative rule matched on, vetoing the fallback and
# leaving the funded personal key untried.
#
# Shared rather than copied per test file because the exact bytes are the assertion
# substrate: tests reason about what this string does NOT contain ("credit",
# "insufficient", "balance", "402") and about which status digits the 64-hex key hash
# happens to carry (none of 401/402/403/429 — the only "403" is the JSON code field).
# Divergent copies would quietly invalidate that reasoning.
PRODUCTION_KEY_LIMIT_403 = (
    "litellm.APIError: APIError: OpenrouterException - "
    '{"error":{"message":"Key limit exceeded (total limit). Manage it using '
    "https://openrouter.ai/workspaces/default/keys/"
    '8f5af82f134c33c0dbada6e1ce93b780819cc08716001bef5ab4af81791702bd","code":403}}'
)


def gather_predictions_stub(result: tuple[Any, Any, Any]) -> AsyncMock:
    """An ``AsyncMock`` stand-in for ``TemplateForecaster._gather_predictions_with_wall_clock``.

    ``_research_and_make_predictions`` (``metaculus_bot/forecaster.py``) builds one
    coroutine per forecaster by CALLING ``_forecaster_with_soft_deadline``, then hands
    the whole list to ``_gather_predictions_with_wall_clock``, which owns them from
    that point on. Tests that stub the forecaster with an ``AsyncMock`` make each of
    those calls produce a real coroutine object, so a plain ``MagicMock`` stand-in for
    gather silently drops them: every one is later garbage-collected unawaited and
    emits ``RuntimeWarning: coroutine 'AsyncMockMixin._execute_mock_call' was never
    awaited``. Those warnings are attributed to whichever unrelated test happened to
    trigger the collection, which is why they were so hard to place.

    This closes the coroutines it receives (honoring gather's ownership contract)
    without running the stubs, then returns ``result`` — the ``(valid_predictions,
    errors, exception_group)`` triple the real function returns. The returned mock
    records its calls normally, so assertions on gather's args still work.
    """

    async def _close_tasks_and_return(tasks, *_args, **_kwargs):
        for task in tasks:
            task.close()
        return result

    return AsyncMock(side_effect=_close_tasks_and_return)


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
