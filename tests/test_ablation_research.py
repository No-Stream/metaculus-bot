"""Tests for the Gemini-only ablation research module.

These tests mock both ``gemini_search_provider`` (callable) and ``run_gap_fill_pass``.
No live API calls. The module under test is ``metaculus_bot.ablation.research``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from forecasting_tools import MetaculusQuestion

from metaculus_bot.ablation.cache import AblationCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


@pytest.fixture
def cache(tmp_path: Path) -> AblationCache:
    return AblationCache(tmp_path / "abl")


def _install_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gemini_blob: str | Exception,
    gap_blob: str | Exception,
    gemini_factory: MagicMock | None = None,
) -> tuple[AsyncMock, AsyncMock, MagicMock]:
    """Patch the two upstream entry points used by ``run_gemini_only_research``.

    Returns (gemini_callable_mock, gap_fill_pass_mock, factory_mock) so individual
    tests can inspect call args.

    The Gemini provider is a factory that returns a callable; we mock both the
    factory (so we can inspect ``is_benchmarking`` it received) and the callable
    (so we can inspect the question it received and steer the result).
    """
    if isinstance(gemini_blob, Exception):
        gemini_callable = AsyncMock(side_effect=gemini_blob)
    else:
        gemini_callable = AsyncMock(return_value=gemini_blob)

    factory = gemini_factory or MagicMock(return_value=gemini_callable)

    if isinstance(gap_blob, Exception):
        gap_fill = AsyncMock(side_effect=gap_blob)
    else:
        gap_fill = AsyncMock(return_value=gap_blob)

    monkeypatch.setattr("metaculus_bot.ablation.research.gemini_search_provider", factory)
    monkeypatch.setattr("metaculus_bot.ablation.research.run_gap_fill_pass", gap_fill)

    return gemini_callable, gap_fill, factory


# ---------------------------------------------------------------------------
# run_gemini_only_research — cache behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_short_circuits_gemini(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-populated cache → no API calls."""
    from metaculus_bot.ablation.research import run_gemini_only_research

    question = _make_question(qid=42)
    cache.write_research(qid=42, blob="cached blob", meta={"gemini_search_used": True})

    gemini_callable, gap_fill, factory = _install_mocks(
        monkeypatch,
        gemini_blob="should not be called",
        gap_blob="should not be called",
    )

    blob, meta = await run_gemini_only_research(question, cache)

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
    from metaculus_bot.ablation.research import run_gemini_only_research

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
    from metaculus_bot.ablation.research import run_gemini_only_research

    question = _make_question(qid=7)
    cache.write_research(qid=7, blob="OLD", meta={"first_pass_chars": 0})

    fresh_first_pass = "x" * 250  # exceed GAP_FILL_MIN_RESEARCH_CHARS
    gemini_callable, gap_fill, factory = _install_mocks(
        monkeypatch,
        gemini_blob=fresh_first_pass,
        gap_blob="fresh gap addendum",
    )

    blob, meta = await run_gemini_only_research(question, cache, force=True)

    factory.assert_called_once()
    gemini_callable.assert_called_once()
    gap_fill.assert_called_once()

    # Cache was overwritten with the new blob.
    cached = cache.read_research(qid=7)
    assert cached is not None
    new_blob, _ = cached
    assert "OLD" not in new_blob
    assert fresh_first_pass in new_blob


# ---------------------------------------------------------------------------
# run_gemini_only_research — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_miss_invokes_both_apis(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh question → factory + callable + run_gap_fill_pass each called once."""
    from metaculus_bot.ablation.research import run_gemini_only_research

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
    from metaculus_bot.ablation.research import run_gemini_only_research

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
    from metaculus_bot.ablation.research import run_gemini_only_research

    question = _make_question(qid=3)
    short_first_pass = "tiny"  # 4 chars, well below 200
    gemini_callable, gap_fill, factory = _install_mocks(
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
async def test_gap_fill_failure_caches_first_pass_alone(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gap-fill raising → first-pass blob still cached; function returns successfully."""
    from metaculus_bot.ablation.research import run_gemini_only_research

    question = _make_question(qid=4)
    long_first_pass = "first-pass result " * 20
    _install_mocks(
        monkeypatch,
        gemini_blob=long_first_pass,
        gap_blob=RuntimeError("gap-fill exploded"),
    )

    blob, meta = await run_gemini_only_research(question, cache)

    assert blob == long_first_pass
    assert meta["gap_fill_used"] is False
    assert meta["gap_count"] == 0

    cached = cache.read_research(qid=4)
    assert cached is not None
    cached_blob, _ = cached
    assert cached_blob == long_first_pass


# ---------------------------------------------------------------------------
# run_gemini_only_research — failure semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_primary_gemini_failure_reraises(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemini callable raises → re-raised; cache is not written."""
    from metaculus_bot.ablation.research import run_gemini_only_research

    question = _make_question(qid=5)
    _install_mocks(
        monkeypatch,
        gemini_blob=RuntimeError("gemini fail"),
        gap_blob="never reached",
    )

    with pytest.raises(RuntimeError, match="gemini fail"):
        await run_gemini_only_research(question, cache)

    assert cache.read_research(qid=5) is None


# ---------------------------------------------------------------------------
# run_gemini_only_research — patches & threading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gap_fill_max_gaps_monkey_patched(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gap_fill_max_gaps overrides constants.GAP_FILL_MAX_GAPS in BOTH modules during the call,
    and restores both after the call returns.
    """
    from metaculus_bot import constants
    from metaculus_bot.ablation.research import run_gemini_only_research
    from metaculus_bot.research import targeted

    question = _make_question(qid=6)
    long_first_pass = "first pass " * 30

    captured: dict[str, int] = {}

    async def capture_max_gaps(*_args: object, **_kwargs: object) -> str:
        captured["constants"] = constants.GAP_FILL_MAX_GAPS
        captured["targeted_research"] = targeted.GAP_FILL_MAX_GAPS
        await asyncio.sleep(0)
        return "addendum"

    monkeypatch.setattr(
        "metaculus_bot.ablation.research.gemini_search_provider",
        MagicMock(return_value=AsyncMock(return_value=long_first_pass)),
    )
    monkeypatch.setattr(
        "metaculus_bot.ablation.research.run_gap_fill_pass",
        capture_max_gaps,
    )

    original_constants = constants.GAP_FILL_MAX_GAPS
    original_tr = targeted.GAP_FILL_MAX_GAPS

    await run_gemini_only_research(question, cache, gap_fill_max_gaps=2)

    # During the gap-fill call, both module references were patched to 2.
    assert captured["constants"] == 2
    assert captured["targeted_research"] == 2

    # After the call, both are restored to original values.
    assert constants.GAP_FILL_MAX_GAPS == original_constants
    assert targeted.GAP_FILL_MAX_GAPS == original_tr


@pytest.mark.asyncio
async def test_gap_fill_max_gaps_restored_on_exception(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even if gap-fill raises, the GAP_FILL_MAX_GAPS patch is reverted (try/finally)."""
    from metaculus_bot import constants
    from metaculus_bot.ablation.research import run_gemini_only_research
    from metaculus_bot.research import targeted

    question = _make_question(qid=8)
    long_first_pass = "first pass " * 30

    monkeypatch.setattr(
        "metaculus_bot.ablation.research.gemini_search_provider",
        MagicMock(return_value=AsyncMock(return_value=long_first_pass)),
    )
    monkeypatch.setattr(
        "metaculus_bot.ablation.research.run_gap_fill_pass",
        AsyncMock(side_effect=RuntimeError("kaboom")),
    )

    original_constants = constants.GAP_FILL_MAX_GAPS
    original_tr = targeted.GAP_FILL_MAX_GAPS

    # Gap-fill failure is absorbed (matches production soft-fail), so this returns normally.
    await run_gemini_only_research(question, cache, gap_fill_max_gaps=2)

    assert constants.GAP_FILL_MAX_GAPS == original_constants
    assert targeted.GAP_FILL_MAX_GAPS == original_tr


@pytest.mark.asyncio
async def test_gap_fill_year_patched_during_gap_fill(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """patched_gap_fill_year_for_question is active when run_gap_fill_pass is called."""
    from metaculus_bot.ablation.research import run_gemini_only_research

    question = _make_question(qid=9)
    long_first_pass = "first pass " * 30

    entered = {"flag": False}

    from contextlib import contextmanager

    @contextmanager
    def fake_patcher(q: MetaculusQuestion):  # type: ignore[no-untyped-def]
        entered["flag"] = True
        yield

    monkeypatch.setattr("metaculus_bot.ablation.research.patched_gap_fill_year_for_question", fake_patcher)

    flag_during_gap_fill: dict[str, bool] = {}

    async def gap_fill_observer(*_args: object, **_kwargs: object) -> str:
        flag_during_gap_fill["entered"] = entered["flag"]
        await asyncio.sleep(0)
        return "addendum"

    monkeypatch.setattr(
        "metaculus_bot.ablation.research.gemini_search_provider",
        MagicMock(return_value=AsyncMock(return_value=long_first_pass)),
    )
    monkeypatch.setattr(
        "metaculus_bot.ablation.research.run_gap_fill_pass",
        gap_fill_observer,
    )

    await run_gemini_only_research(question, cache)

    assert flag_during_gap_fill["entered"] is True


@pytest.mark.asyncio
async def test_is_benchmarking_threaded_to_gemini(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """is_benchmarking=True passes through to gemini_search_provider factory and run_gap_fill_pass."""
    from metaculus_bot.ablation.research import run_gemini_only_research

    question = _make_question(qid=10)
    long_first_pass = "first pass " * 30

    gemini_callable, gap_fill, factory = _install_mocks(
        monkeypatch,
        gemini_blob=long_first_pass,
        gap_blob="addendum",
    )

    await run_gemini_only_research(question, cache, is_benchmarking=True)

    factory.assert_called_once()
    assert factory.call_args.kwargs.get("is_benchmarking") is True

    gap_fill.assert_called_once()
    assert gap_fill.call_args.kwargs.get("is_benchmarking") is True


# ---------------------------------------------------------------------------
# Meta payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_meta_fields_populated(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every documented meta key is present in the returned dict."""
    from metaculus_bot.ablation.research import run_gemini_only_research

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


# ---------------------------------------------------------------------------
# run_gemini_research_for_qids — batch wrapper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_wrapper_runs_concurrently_under_semaphore(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """concurrency=2 over 4 questions caps in-flight calls at 2 simultaneously."""
    from metaculus_bot.ablation.research import run_gemini_research_for_qids

    questions = [_make_question(qid=100 + i) for i in range(4)]
    long_first_pass = "first pass " * 30

    in_flight = 0
    max_in_flight = 0
    lock = asyncio.Lock()

    async def slow_callable(q: MetaculusQuestion) -> str:
        nonlocal in_flight, max_in_flight
        async with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        await asyncio.sleep(0.05)
        async with lock:
            in_flight -= 1
        return long_first_pass

    monkeypatch.setattr(
        "metaculus_bot.ablation.research.gemini_search_provider",
        MagicMock(return_value=slow_callable),
    )
    monkeypatch.setattr(
        "metaculus_bot.ablation.research.run_gap_fill_pass",
        AsyncMock(return_value="addendum"),
    )

    results = await run_gemini_research_for_qids(questions, cache, concurrency=2)

    assert len(results) == 4
    for q in questions:
        assert results[q.id_of_question] is not None
    assert max_in_flight <= 2
    # Sanity: at least 2 ran simultaneously, otherwise the semaphore is over-restricting.
    assert max_in_flight == 2


@pytest.mark.asyncio
async def test_batch_wrapper_per_question_failure_does_not_kill_batch(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Q1 raises in primary Gemini → results[Q1.id] is None; Q2 still succeeds."""
    from metaculus_bot.ablation.research import run_gemini_research_for_qids

    q1 = _make_question(qid=201)
    q2 = _make_question(qid=202)
    long_first_pass = "first pass " * 30

    async def selective_callable(q: MetaculusQuestion) -> str:
        await asyncio.sleep(0)
        if q.id_of_question == 201:
            raise RuntimeError("q1 failed hard")
        return long_first_pass

    monkeypatch.setattr(
        "metaculus_bot.ablation.research.gemini_search_provider",
        MagicMock(return_value=selective_callable),
    )
    monkeypatch.setattr(
        "metaculus_bot.ablation.research.run_gap_fill_pass",
        AsyncMock(return_value="addendum"),
    )

    results = await run_gemini_research_for_qids([q1, q2], cache, concurrency=2)

    assert results[201] is None
    assert results[202] is not None
    blob, meta = results[202]
    assert long_first_pass in blob
    assert meta["gemini_search_used"] is True


@pytest.mark.asyncio
async def test_batch_wrapper_caches_each_qid_separately(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a batch run, each successful qid has its own cache entry."""
    from metaculus_bot.ablation.research import run_gemini_research_for_qids

    questions = [_make_question(qid=300 + i) for i in range(3)]
    long_first_pass = "first pass " * 30

    async def callable_(q: MetaculusQuestion) -> str:
        await asyncio.sleep(0)
        return f"{long_first_pass}-{q.id_of_question}"

    monkeypatch.setattr(
        "metaculus_bot.ablation.research.gemini_search_provider",
        MagicMock(return_value=callable_),
    )
    monkeypatch.setattr(
        "metaculus_bot.ablation.research.run_gap_fill_pass",
        AsyncMock(return_value="addendum"),
    )

    await run_gemini_research_for_qids(questions, cache, concurrency=3)

    for q in questions:
        cached = cache.read_research(qid=q.id_of_question)
        assert cached is not None
        cached_blob, _ = cached
        assert f"-{q.id_of_question}" in cached_blob


# ---------------------------------------------------------------------------
# gemini_model parameter (CLI flag → env var override during call)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_gemini_only_research_sets_gemini_search_model_env_during_call(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """While the gemini callable runs, ``GEMINI_SEARCH_MODEL`` env var is the requested model.

    After the call returns, the env var is restored to whatever it was beforehand.
    The CLI flag is canonical — it must override any pre-existing shell setting.
    """
    import os

    from metaculus_bot.ablation.research import run_gemini_only_research

    question = _make_question(qid=1500)
    long_first_pass = "first pass " * 30

    # Set a sentinel pre-existing env value to ensure restoration works.
    monkeypatch.setenv("GEMINI_SEARCH_MODEL", "preexisting-model")

    captured: dict[str, str | None] = {}

    async def capture_env(_q: MetaculusQuestion) -> str:
        captured["during_call"] = os.environ.get("GEMINI_SEARCH_MODEL")
        await asyncio.sleep(0)
        return long_first_pass

    monkeypatch.setattr(
        "metaculus_bot.ablation.research.gemini_search_provider",
        MagicMock(return_value=capture_env),
    )
    monkeypatch.setattr(
        "metaculus_bot.ablation.research.run_gap_fill_pass",
        AsyncMock(return_value="addendum"),
    )

    await run_gemini_only_research(question, cache, gemini_model="gemini-2.5-flash")

    assert captured["during_call"] == "gemini-2.5-flash"
    # After the call, the original value is restored.
    assert os.environ.get("GEMINI_SEARCH_MODEL") == "preexisting-model"


@pytest.mark.asyncio
async def test_run_gemini_only_research_records_actual_model_in_meta(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Meta payload's ``gemini_model`` reflects the model actually requested."""
    from metaculus_bot.ablation.research import run_gemini_only_research

    question = _make_question(qid=1501)
    long_first_pass = "first pass " * 30
    _install_mocks(monkeypatch, gemini_blob=long_first_pass, gap_blob="addendum")

    _, meta = await run_gemini_only_research(question, cache, gemini_model="gemini-2.5-flash")

    assert meta["gemini_model"] == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# enable_gap_fill parameter (gates gap-fill stage entirely)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_gemini_only_research_skips_gap_fill_when_disabled(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``enable_gap_fill=False`` short-circuits before ``run_gap_fill_pass`` is called.

    Meta payload must record ``gap_fill_enabled=False`` and ``gap_fill_used=False``.
    """
    from metaculus_bot.ablation.research import run_gemini_only_research

    question = _make_question(qid=1600)
    long_first_pass = "first pass " * 30  # > 200 chars
    gemini_callable, gap_fill, factory = _install_mocks(
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
    from metaculus_bot.ablation.research import run_gemini_only_research

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


# ---------------------------------------------------------------------------
# Batch wrapper threads new kwargs through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_wrapper_threads_gemini_model_and_gap_fill_flags(
    cache: AblationCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_gemini_research_for_qids`` accepts ``gemini_model`` + ``enable_gap_fill``
    and threads them into each per-question call.
    """
    from metaculus_bot.ablation.research import run_gemini_research_for_qids

    questions = [_make_question(qid=1700 + i) for i in range(2)]
    long_first_pass = "first pass " * 30

    async def callable_(_q: MetaculusQuestion) -> str:
        await asyncio.sleep(0)
        return long_first_pass

    gap_fill = AsyncMock(return_value="addendum")
    monkeypatch.setattr(
        "metaculus_bot.ablation.research.gemini_search_provider",
        MagicMock(return_value=callable_),
    )
    monkeypatch.setattr(
        "metaculus_bot.ablation.research.run_gap_fill_pass",
        gap_fill,
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
        result = results[q.id_of_question]
        assert result is not None
        _, meta = result
        assert meta["gemini_model"] == "gemini-2.5-flash"
        assert meta["gap_fill_enabled"] is False
        assert meta["gap_fill_used"] is False
