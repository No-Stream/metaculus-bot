"""Tests for the Gemini-only ablation research module.

These tests mock ``gemini_search_provider`` (callable) and, except where a test drives the
real ``run_gap_fill_pass`` to prove its soft-fail contract or its concurrency safety,
``run_gap_fill_pass`` itself. No live API calls. The module under test is
``metaculus_bot.ablation.research``.

The module-global patches (``GAP_FILL_MAX_GAPS`` and the gap-fill year rewrite) belong to
``run_gemini_research_for_qids``, which installs them once around the whole batch, so the
tests that pin them drive the batch wrapper rather than ``run_gemini_only_research``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from forecasting_tools import MetaculusQuestion

from metaculus_bot import constants
from metaculus_bot import prompts as prompts_module
from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.research import run_gemini_only_research, run_gemini_research_for_qids
from metaculus_bot.research import targeted

_AsyncBlobCallable = Callable[..., Awaitable[str]]

# Long enough that two questions overlap inside the gap-fill phase rather than running serially.
_ANALYZER_OVERLAP_SLEEP_S = 0.05


def _make_question(
    *,
    qid: int = 12345,
    open_time: datetime = datetime(2026, 1, 1),
    scheduled_resolution_time: datetime = datetime(2026, 5, 1),
    question_text: str = "Will X happen by 2026?",
) -> MetaculusQuestion:
    """Build a minimal MetaculusQuestion-shaped stub.

    The window-patch context manager reads ``open_time`` and ``scheduled_resolution_time``;
    the gemini provider reads ``question_text``; the gap-fill pass reads
    ``resolution_criteria`` and ``fine_print`` via ``getattr``.
    """
    q = SimpleNamespace(
        id_of_question=qid,
        open_time=open_time,
        scheduled_resolution_time=scheduled_resolution_time,
        question_text=question_text,
        resolution_criteria="Resolves YES if X happens.",
        fine_print="See bls.gov for source data.",
        page_url=f"https://example.com/q/{qid}",
    )
    return cast(MetaculusQuestion, q)


def _install_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gemini_blob: str | BaseException | _AsyncBlobCallable,
    gap_blob: str | BaseException | _AsyncBlobCallable,
) -> tuple[AsyncMock, AsyncMock, MagicMock]:
    """Patch the two upstream entry points used by ``run_gemini_only_research``.

    A ``str`` becomes that call's return value, an exception its raise, and a coroutine
    function is called through (a batch test steering the result per question, or observing
    module state mid-call). Returns (gemini_callable_mock, gap_fill_pass_mock, factory_mock)
    so individual tests can inspect call args: the Gemini provider is a factory returning a
    callable, and both are mocked so a test can assert on the factory's kwargs and on the
    question the callable received.
    """
    gemini_callable = (
        AsyncMock(return_value=gemini_blob) if isinstance(gemini_blob, str) else AsyncMock(side_effect=gemini_blob)
    )
    factory = MagicMock(return_value=gemini_callable)
    gap_fill = AsyncMock(return_value=gap_blob) if isinstance(gap_blob, str) else AsyncMock(side_effect=gap_blob)

    monkeypatch.setattr("metaculus_bot.ablation.research.gemini_search_provider", factory)
    monkeypatch.setattr("metaculus_bot.ablation.research.run_gap_fill_pass", gap_fill)

    return gemini_callable, gap_fill, factory


class _RecordingAnalyzerLlm:
    """Stand-in for the gap-fill analyzer LLM: records each call and returns one gap."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.max_gaps_seen: list[int] = []

    async def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self.max_gaps_seen.append(targeted.GAP_FILL_MAX_GAPS)
        await asyncio.sleep(_ANALYZER_OVERLAP_SLEEP_S)
        return (
            '```json\n{"gaps": [{"gap": "the missing figure", "why_matters": "it decides the question", '
            '"answerable_now": true, "already_in_first_pass": false, "same_need_as": null}]}\n```'
        )


def _install_real_gap_fill(monkeypatch: pytest.MonkeyPatch, *, first_pass: str) -> _RecordingAnalyzerLlm:
    """Wire the REAL ``run_gap_fill_pass`` with only its analyzer LLM and its gap resolver stubbed.

    ``_run_analyzer`` late-imports ``build_llm_with_openrouter_fallback``, so the fake LLM is
    installed on ``fallback_openrouter`` where the name is bound at call time. The analyzer
    sleeps mid-call, which is what puts two questions in the gap-fill phase at once.
    """
    analyzer_llm = _RecordingAnalyzerLlm()
    monkeypatch.setattr(
        "metaculus_bot.ablation.research.gemini_search_provider",
        MagicMock(return_value=AsyncMock(return_value=first_pass)),
    )
    monkeypatch.setattr(
        "metaculus_bot.fallback_openrouter.build_llm_with_openrouter_fallback",
        MagicMock(return_value=analyzer_llm),
    )
    monkeypatch.setattr(
        "metaculus_bot.research.targeted._resolve_single_gap",
        AsyncMock(return_value="resolved gap text"),
    )
    return analyzer_llm


@pytest.mark.asyncio
async def test_cache_hit_short_circuits_gemini(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-populated cache → no API calls."""
    question = _make_question(qid=42)
    cache.write_research(qid=42, blob="cached blob", meta={"gemini_search_used": True})

    gemini_callable, gap_fill, factory = _install_mocks(
        monkeypatch,
        gemini_blob="should not be called",
        gap_blob="should not be called",
    )

    blob, _meta = await run_gemini_only_research(question, cache)

    assert blob == "cached blob"
    factory.assert_not_called()
    gemini_callable.assert_not_called()
    gap_fill.assert_not_called()


@pytest.mark.asyncio
async def test_cache_hit_returns_cached_values(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached blob and meta dict are returned exactly as stored."""
    question = _make_question(qid=99)
    original_meta = {
        "gemini_search_used": True,
        "gap_fill_used": True,
        "gap_count": 3,
        "first_pass_chars": 1234,
        "gap_fill_chars": 567,
        "researched_at": "2026-05-13T10:00:00",
        "gemini_model": "gemini-3-flash-preview",
        "gap_fill_max_gaps": 3,
        "is_benchmarking": True,
    }
    cache.write_research(qid=99, blob="abc", meta=original_meta)

    _install_mocks(monkeypatch, gemini_blob="never", gap_blob="never")

    blob, meta = await run_gemini_only_research(question, cache)

    assert blob == "abc"
    for key, value in original_meta.items():
        assert meta[key] == value


@pytest.mark.asyncio
async def test_force_true_bypasses_cache(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """force=True ignores existing cache and re-fetches both APIs."""
    question = _make_question(qid=7)
    cache.write_research(qid=7, blob="OLD", meta={"first_pass_chars": 0})

    fresh_first_pass = "x" * 250  # exceed GAP_FILL_MIN_RESEARCH_CHARS
    gemini_callable, gap_fill, factory = _install_mocks(
        monkeypatch,
        gemini_blob=fresh_first_pass,
        gap_blob="fresh gap addendum",
    )

    _blob, _meta = await run_gemini_only_research(question, cache, force=True)

    factory.assert_called_once()
    gemini_callable.assert_called_once()
    gap_fill.assert_called_once()

    # Cache was overwritten with the new blob.
    cached = cache.read_research(qid=7)
    assert cached is not None
    new_blob, _ = cached
    assert "OLD" not in new_blob
    assert fresh_first_pass in new_blob


@pytest.mark.asyncio
async def test_cache_miss_invokes_both_apis(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh question → factory + callable + run_gap_fill_pass each called once."""
    question = _make_question(qid=1)
    long_first_pass = "first-pass research blob " * 20  # well above 200 chars
    gemini_callable, gap_fill, factory = _install_mocks(
        monkeypatch,
        gemini_blob=long_first_pass,
        gap_blob="gap addendum",
    )

    await run_gemini_only_research(question, cache)

    assert factory.call_count == 1
    assert gemini_callable.await_count == 1
    assert gap_fill.await_count == 1


@pytest.mark.asyncio
async def test_concatenated_blob_matches_production_format(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concatenation matches main.run_research's exact separator + header."""
    question = _make_question(qid=2)
    first_pass = "first pass result " * 20  # > 200 chars
    gap_addendum = "Gap fill text"
    _install_mocks(monkeypatch, gemini_blob=first_pass, gap_blob=gap_addendum)

    blob, _ = await run_gemini_only_research(question, cache)

    expected_separator = "\n\n---\n\n## Targeted Gap-Fill (second pass)\n\n"
    assert expected_separator in blob
    assert blob == f"{first_pass}{expected_separator}{gap_addendum}"


@pytest.mark.asyncio
async def test_gap_fill_skipped_when_first_pass_too_short(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-pass < GAP_FILL_MIN_RESEARCH_CHARS → gap-fill not called; meta marks gap_fill_used=False."""
    question = _make_question(qid=3)
    short_first_pass = "tiny"  # 4 chars, well below 200
    _gemini_callable, gap_fill, _factory = _install_mocks(
        monkeypatch,
        gemini_blob=short_first_pass,
        gap_blob="should not be called",
    )

    blob, meta = await run_gemini_only_research(question, cache)

    assert blob == short_first_pass
    gap_fill.assert_not_called()
    assert meta["gap_fill_used"] is False
    assert meta["gap_count"] == 0


@pytest.mark.asyncio
async def test_gap_fill_skipped_when_first_pass_is_only_whitespace_padded(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whitespace does not count toward the gate: production gates on ``len(research.strip())``."""
    question = _make_question(qid=13)
    padded_first_pass = f"{' ' * 250}tiny{' ' * 250}"
    _gemini_callable, gap_fill, _factory = _install_mocks(
        monkeypatch,
        gemini_blob=padded_first_pass,
        gap_blob="should not be called",
    )

    _blob, meta = await run_gemini_only_research(question, cache)

    gap_fill.assert_not_called()
    assert meta["gap_fill_used"] is False


@pytest.mark.asyncio
async def test_gap_fill_provider_failure_caches_first_pass_alone(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The analyzer's wall timeout, absorbed by the REAL ``run_gap_fill_pass``, leaves the first pass cached alone.

    ``run_gap_fill_pass`` is deliberately not mocked here: the soft-fail lives in it, so this
    drives it end to end with only its analyzer stubbed to fail the way a slow provider does.
    """
    question = _make_question(qid=4)
    long_first_pass = "first-pass result " * 20
    monkeypatch.setattr(
        "metaculus_bot.ablation.research.gemini_search_provider",
        MagicMock(return_value=AsyncMock(return_value=long_first_pass)),
    )
    monkeypatch.setattr("metaculus_bot.research.targeted._run_analyzer", AsyncMock(side_effect=TimeoutError()))

    blob, meta = await run_gemini_only_research(question, cache)

    assert blob == long_first_pass
    assert meta["gap_fill_used"] is False
    assert meta["gap_count"] == 0

    cached = cache.read_research(qid=4)
    assert cached is not None
    cached_blob, _ = cached
    assert cached_blob == long_first_pass


@pytest.mark.asyncio
async def test_gap_fill_bug_propagates_and_caches_nothing(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception ``run_gap_fill_pass`` lets out is a bug: it propagates and the qid is not cached."""
    question = _make_question(qid=14)
    _install_mocks(
        monkeypatch,
        gemini_blob="first-pass result " * 20,
        gap_blob=KeyError("gap"),
    )

    with pytest.raises(KeyError, match="gap"):
        await run_gemini_only_research(question, cache)

    assert cache.read_research(qid=14) is None


@pytest.mark.asyncio
async def test_primary_gemini_failure_reraises(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini callable raises → re-raised; cache is not written."""
    question = _make_question(qid=5)
    _install_mocks(
        monkeypatch,
        gemini_blob=RuntimeError("gemini fail"),
        gap_blob="never reached",
    )

    with pytest.raises(RuntimeError, match="gemini fail"):
        await run_gemini_only_research(question, cache)

    assert cache.read_research(qid=5) is None


@pytest.mark.asyncio
async def test_is_benchmarking_threaded_to_gemini(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """is_benchmarking=True passes through to gemini_search_provider factory and run_gap_fill_pass."""
    question = _make_question(qid=10)
    long_first_pass = "first pass " * 30

    _gemini_callable, gap_fill, factory = _install_mocks(
        monkeypatch,
        gemini_blob=long_first_pass,
        gap_blob="addendum",
    )

    await run_gemini_only_research(question, cache, is_benchmarking=True)

    factory.assert_called_once()
    assert factory.call_args.kwargs.get("is_benchmarking") is True

    gap_fill.assert_called_once()
    assert gap_fill.call_args.kwargs.get("is_benchmarking") is True


@pytest.mark.asyncio
async def test_run_gemini_only_research_passes_the_requested_model_to_the_provider(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI flag reaches the provider as ``model_slug``, which wins over the env var in ``_resolve_model``.

    The env var itself is left alone: a pre-existing shell value survives the call untouched.
    """
    question = _make_question(qid=1500)
    monkeypatch.setenv("GEMINI_SEARCH_MODEL", "preexisting-model")

    _gemini_callable, _gap_fill, factory = _install_mocks(
        monkeypatch,
        gemini_blob="first pass " * 30,
        gap_blob="addendum",
    )

    await run_gemini_only_research(question, cache, gemini_model="gemini-2.5-flash")

    factory.assert_called_once()
    assert factory.call_args.kwargs.get("model_slug") == "gemini-2.5-flash"
    assert factory.call_args.kwargs.get("is_benchmarking") is True
    assert os.environ.get("GEMINI_SEARCH_MODEL") == "preexisting-model"


@pytest.mark.asyncio
async def test_run_gemini_only_research_records_actual_model_in_meta(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Meta payload's ``gemini_model`` reflects the model actually requested."""
    question = _make_question(qid=1501)
    long_first_pass = "first pass " * 30
    _install_mocks(monkeypatch, gemini_blob=long_first_pass, gap_blob="addendum")

    _, meta = await run_gemini_only_research(question, cache, gemini_model="gemini-2.5-flash")

    assert meta["gemini_model"] == "gemini-2.5-flash"


@pytest.mark.asyncio
async def test_run_gemini_only_research_skips_gap_fill_when_disabled(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``enable_gap_fill=False`` short-circuits before ``run_gap_fill_pass`` is called.

    Meta payload must record ``gap_fill_enabled=False`` and ``gap_fill_used=False``.
    """
    question = _make_question(qid=1600)
    long_first_pass = "first pass " * 30  # > 200 chars
    _gemini_callable, gap_fill, _factory = _install_mocks(
        monkeypatch,
        gemini_blob=long_first_pass,
        gap_blob="should not be called",
    )

    blob, meta = await run_gemini_only_research(question, cache, enable_gap_fill=False)

    gap_fill.assert_not_called()
    assert blob == long_first_pass  # No gap-fill addendum.
    assert meta["gap_fill_enabled"] is False
    assert meta["gap_fill_used"] is False


@pytest.mark.asyncio
async def test_run_gemini_only_research_runs_gap_fill_when_enabled(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``enable_gap_fill=True`` goes through the existing gap-fill path."""
    question = _make_question(qid=1601)
    long_first_pass = "first pass " * 30
    _, gap_fill, _ = _install_mocks(
        monkeypatch,
        gemini_blob=long_first_pass,
        gap_blob="addendum text",
    )

    _, meta = await run_gemini_only_research(question, cache, enable_gap_fill=True)

    gap_fill.assert_called_once()
    assert meta["gap_fill_enabled"] is True
    assert meta["gap_fill_used"] is True


@pytest.mark.asyncio
async def test_meta_fields_populated(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every documented meta key is present in the returned dict."""
    question = _make_question(qid=11)
    long_first_pass = "first pass content " * 30
    gap_blob = "gap addendum text"
    _install_mocks(monkeypatch, gemini_blob=long_first_pass, gap_blob=gap_blob)

    _, meta = await run_gemini_only_research(question, cache, gap_fill_max_gaps=4, is_benchmarking=True)

    expected_keys = {
        "gemini_search_used",
        "gap_fill_used",
        "gap_count",
        "first_pass_chars",
        "gap_fill_chars",
        "researched_at",
        "gemini_model",
        "gap_fill_max_gaps",
        "is_benchmarking",
    }
    assert expected_keys.issubset(meta.keys())

    assert meta["gemini_search_used"] is True
    assert meta["gap_fill_used"] is True
    assert meta["first_pass_chars"] == len(long_first_pass)
    assert meta["gap_fill_chars"] == len(gap_blob)
    assert meta["gap_fill_max_gaps"] == 4
    assert meta["is_benchmarking"] is True
    # ISO datetime parses cleanly.
    datetime.fromisoformat(meta["researched_at"])


@pytest.mark.asyncio
async def test_batch_wrapper_runs_concurrently_under_semaphore(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """concurrency=2 over 4 questions caps in-flight calls at 2 simultaneously."""
    questions = [_make_question(qid=100 + i) for i in range(4)]
    long_first_pass = "first pass " * 30

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def slow_callable(_q: MetaculusQuestion) -> str:
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return long_first_pass

    _install_mocks(monkeypatch, gemini_blob=slow_callable, gap_blob="addendum")

    results = await run_gemini_research_for_qids(questions, cache, concurrency=2)

    assert len(results) == 4
    for q in questions:
        qid = q.id_of_question
        assert qid is not None
        assert results[qid] is not None
    assert max_in_flight <= 2
    # Sanity: at least 2 ran simultaneously, otherwise the semaphore is over-restricting.
    assert max_in_flight == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError("gemini wall"), KeyError("meta")], ids=["provider", "bug"])
async def test_batch_wrapper_logs_the_failed_question_with_its_traceback(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failure: Exception,
) -> None:
    """Whatever one question raises, the batch logs it at ERROR with the traceback, caches nothing for it and goes on."""
    failing = _make_question(qid=211)
    healthy = _make_question(qid=212)
    long_first_pass = "first pass " * 30

    async def selective_callable(q: MetaculusQuestion) -> str:
        await asyncio.sleep(0)
        if q.id_of_question == 211:
            raise failure
        return long_first_pass

    _install_mocks(monkeypatch, gemini_blob=selective_callable, gap_blob="addendum")

    with caplog.at_level(logging.ERROR, logger="metaculus_bot.ablation.research"):
        results = await run_gemini_research_for_qids([failing, healthy], cache, concurrency=2)

    assert results[211] is None
    healthy_result = results[212]
    assert healthy_result is not None
    healthy_blob, healthy_meta = healthy_result
    assert long_first_pass in healthy_blob
    assert healthy_meta["gemini_search_used"] is True
    assert cache.read_research(qid=211) is None
    assert cache.read_research(qid=212) is not None
    failure_records = [r for r in caplog.records if r.levelno == logging.ERROR and "qid 211" in r.getMessage()]
    assert len(failure_records) == 1
    assert failure_records[0].exc_info is not None
    assert failure_records[0].exc_info[1] is failure


@pytest.mark.asyncio
async def test_batch_wrapper_reraises_a_cancelled_child(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled child is the event loop shutting down, not this question failing, so it propagates.

    Its ``Exception`` counterpart is mapped to ``None`` and logged instead — see
    ``test_batch_wrapper_logs_the_failed_question_with_its_traceback``.
    """
    cancelled = _make_question(qid=221)
    healthy = _make_question(qid=222)
    long_first_pass = "first pass " * 30

    async def selective_callable(q: MetaculusQuestion) -> str:
        await asyncio.sleep(0)
        if q.id_of_question == 221:
            raise asyncio.CancelledError()
        return long_first_pass

    _install_mocks(monkeypatch, gemini_blob=selective_callable, gap_blob="addendum")

    with pytest.raises(asyncio.CancelledError):
        await run_gemini_research_for_qids([cancelled, healthy], cache, concurrency=2)


@pytest.mark.asyncio
async def test_batch_wrapper_caches_each_qid_separately(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a batch run, each successful qid has its own cache entry."""
    questions = [_make_question(qid=300 + i) for i in range(3)]
    long_first_pass = "first pass " * 30

    async def per_question_callable(q: MetaculusQuestion) -> str:
        await asyncio.sleep(0)
        return f"{long_first_pass}-{q.id_of_question}"

    _install_mocks(monkeypatch, gemini_blob=per_question_callable, gap_blob="addendum")

    await run_gemini_research_for_qids(questions, cache, concurrency=3)

    for q in questions:
        qid = q.id_of_question
        assert qid is not None
        cached = cache.read_research(qid=qid)
        assert cached is not None
        cached_blob, _ = cached
        assert f"-{q.id_of_question}" in cached_blob


@pytest.mark.asyncio
async def test_batch_wrapper_threads_gemini_model_and_gap_fill_flags(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_gemini_research_for_qids`` accepts ``gemini_model`` + ``enable_gap_fill``
    and threads them into each per-question call.
    """
    questions = [_make_question(qid=1700 + i) for i in range(2)]
    _gemini_callable, gap_fill, _factory = _install_mocks(
        monkeypatch,
        gemini_blob="first pass " * 30,
        gap_blob="addendum",
    )

    results = await run_gemini_research_for_qids(
        questions,
        cache,
        concurrency=2,
        gemini_model="gemini-2.5-flash",
        enable_gap_fill=False,
    )

    assert len(results) == 2
    # gap_fill never invoked because enable_gap_fill=False.
    gap_fill.assert_not_called()

    # Meta records the requested model and enable flag for each qid.
    for q in questions:
        qid = q.id_of_question
        assert qid is not None
        result = results[qid]
        assert result is not None
        _, meta = result
        assert meta["gemini_model"] == "gemini-2.5-flash"
        assert meta["gap_fill_enabled"] is False
        assert meta["gap_fill_used"] is False


@pytest.mark.asyncio
async def test_batch_wrapper_patches_gap_fill_max_gaps_in_both_modules(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``gap_fill_max_gaps`` overrides ``GAP_FILL_MAX_GAPS`` in BOTH modules for the batch, and restores both."""
    questions = [_make_question(qid=6)]
    captured: dict[str, int] = {}

    async def capture_max_gaps(*_args: object, **_kwargs: object) -> str:
        captured["constants"] = constants.GAP_FILL_MAX_GAPS
        captured["targeted_research"] = targeted.GAP_FILL_MAX_GAPS
        await asyncio.sleep(0)
        return "addendum"

    _install_mocks(monkeypatch, gemini_blob="first pass " * 30, gap_blob=capture_max_gaps)

    original_constants = constants.GAP_FILL_MAX_GAPS
    original_targeted = targeted.GAP_FILL_MAX_GAPS

    await run_gemini_research_for_qids(questions, cache, gap_fill_max_gaps=2)

    assert captured["constants"] == 2
    assert captured["targeted_research"] == 2
    assert original_constants == constants.GAP_FILL_MAX_GAPS
    assert original_targeted == targeted.GAP_FILL_MAX_GAPS


@pytest.mark.asyncio
async def test_batch_wrapper_restores_the_module_patches_when_the_batch_raises(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure that propagates out of the gather still leaves every patched module global restored."""
    questions = [_make_question(qid=8)]
    _install_mocks(
        monkeypatch,
        gemini_blob="first pass " * 30,
        gap_blob=asyncio.CancelledError(),
    )

    original_constants = constants.GAP_FILL_MAX_GAPS
    original_targeted_max_gaps = targeted.GAP_FILL_MAX_GAPS
    original_prompt = prompts_module.gap_fill_analyzer_prompt
    original_targeted_prompt = targeted.gap_fill_analyzer_prompt

    with pytest.raises(asyncio.CancelledError):
        await run_gemini_research_for_qids(questions, cache, gap_fill_max_gaps=2)

    assert original_constants == constants.GAP_FILL_MAX_GAPS
    assert original_targeted_max_gaps == targeted.GAP_FILL_MAX_GAPS
    assert prompts_module.gap_fill_analyzer_prompt is original_prompt
    assert targeted.gap_fill_analyzer_prompt is original_targeted_prompt


@pytest.mark.asyncio
async def test_batch_wrapper_holds_the_year_patch_while_gap_fill_runs(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch's year patch is active for the whole gather, so every question's gap-fill runs under it."""
    questions = [_make_question(qid=9)]
    entered = {"flag": False}

    @contextmanager
    def fake_patcher(_questions: list[MetaculusQuestion]) -> Iterator[None]:
        entered["flag"] = True
        try:
            yield
        finally:
            entered["flag"] = False

    monkeypatch.setattr("metaculus_bot.ablation.research.patched_gap_fill_year_for_questions", fake_patcher)

    flag_during_gap_fill: dict[str, bool] = {}

    async def gap_fill_observer(*_args: object, **_kwargs: object) -> str:
        flag_during_gap_fill["entered"] = entered["flag"]
        await asyncio.sleep(0)
        return "addendum"

    _install_mocks(monkeypatch, gemini_blob="first pass " * 30, gap_blob=gap_fill_observer)

    await run_gemini_research_for_qids(questions, cache)

    assert flag_during_gap_fill["entered"] is True
    assert entered["flag"] is False


@pytest.mark.asyncio
async def test_batch_wrapper_gap_fills_both_questions_when_their_gap_fills_overlap(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two questions in the gap-fill phase at once must BOTH get a gap-fill, each with its own year.

    This drives the REAL ``run_gap_fill_pass`` and the REAL year patch. Patching the prompt
    per question chained the second question's wrapper onto the first's, so the second found
    the year already rewritten, raised, and soft-failed to no gap-fill at all.
    """
    long_first_pass = "first pass " * 30
    earlier = _make_question(
        qid=401,
        open_time=datetime(2024, 1, 1),
        scheduled_resolution_time=datetime(2024, 5, 1),
        question_text="Will the earlier question happen?",
    )
    later = _make_question(
        qid=402,
        open_time=datetime(2025, 1, 1),
        scheduled_resolution_time=datetime(2025, 5, 1),
        question_text="Will the later question happen?",
    )
    analyzer_llm = _install_real_gap_fill(monkeypatch, first_pass=long_first_pass)

    original_prompt = prompts_module.gap_fill_analyzer_prompt
    original_max_gaps = constants.GAP_FILL_MAX_GAPS

    results = await run_gemini_research_for_qids([earlier, later], cache, concurrency=2, gap_fill_max_gaps=2)

    for qid in (401, 402):
        result = results[qid]
        assert result is not None
        _blob, meta = result
        assert meta["gap_fill_used"] is True
        assert meta["gap_count"] == 1

    assert len(analyzer_llm.prompts) == 2
    current_year = datetime.now(UTC).year
    assert not any(f"no {current_year} data" in prompt for prompt in analyzer_llm.prompts)
    assert sum("no 2023 data" in prompt for prompt in analyzer_llm.prompts) == 1
    assert sum("no 2024 data" in prompt for prompt in analyzer_llm.prompts) == 1

    assert prompts_module.gap_fill_analyzer_prompt is original_prompt
    assert targeted.gap_fill_analyzer_prompt is original_prompt
    assert original_max_gaps == constants.GAP_FILL_MAX_GAPS
    assert original_max_gaps == targeted.GAP_FILL_MAX_GAPS


@pytest.mark.asyncio
async def test_batch_wrapper_shows_every_analyzer_call_the_requested_max_gaps(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both questions' analyzer calls read the batch's ``gap_fill_max_gaps``, not the production default."""
    questions = [
        _make_question(qid=411, question_text="Will the first question happen?"),
        _make_question(qid=412, question_text="Will the second question happen?"),
    ]
    analyzer_llm = _install_real_gap_fill(monkeypatch, first_pass="first pass " * 30)

    await run_gemini_research_for_qids(questions, cache, concurrency=2, gap_fill_max_gaps=2)

    assert analyzer_llm.max_gaps_seen == [2, 2]
