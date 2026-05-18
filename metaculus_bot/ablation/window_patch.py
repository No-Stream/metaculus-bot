"""Context managers that anchor prompt-injected dates to a question's mid-window.

For ablation backtests on resolved questions, the production prompts'
``_forecasting_window_str`` reveals the resolution status by computing
"days from now" against ``datetime.now()``. These helpers monkey-patch
the prompt builders for the duration of a single question's forecast,
restoring the originals on exit.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot import prompts as prompts_module

__all__ = [
    "compute_mid_window_today",
    "patched_gap_fill_year_for_question",
    "patched_window_and_year_for_question",
    "patched_window_for_question",
]


def compute_mid_window_today(question: MetaculusQuestion | Any) -> datetime:
    """Return the mid-window datetime for ``question``.

    Mid-window = ``open_time + (scheduled_resolution_time - open_time) / 2``.
    Both timestamps are required; absence is a data bug, not a graceful path.
    """
    assert question.open_time is not None, "question.open_time is required"
    assert question.scheduled_resolution_time is not None, "question.scheduled_resolution_time is required"
    delta = question.scheduled_resolution_time - question.open_time
    return question.open_time + delta / 2


def _question_identity(question: Any) -> tuple[str, int]:
    """Hashable identity for question-routing in the patched function.

    Real questions populate ``id_of_question``; synthetic test fixtures
    sometimes leave it ``None`` and we fall back to ``id(question)`` so
    distinct objects don't collide.
    """
    qid = getattr(question, "id_of_question", None)
    if qid is None:
        return ("py_id", id(question))
    return ("metaculus_id", int(qid))


_window_patch_active: bool = False


@contextmanager
def patched_window_for_question(question: MetaculusQuestion | Any) -> Iterator[None]:
    """Monkey-patch ``_forecasting_window_str`` to anchor today on ``question``'s mid-window.

    Calls with a different question fall through to the original ``datetime.now()``
    implementation. Re-entrancy is unsupported and raises ``RuntimeError``.
    """
    global _window_patch_active
    if _window_patch_active:
        raise RuntimeError("patched_window_for_question is already active; nested patches unsupported")

    original = prompts_module._forecasting_window_str
    target_identity = _question_identity(question)
    mid_window = compute_mid_window_today(question)

    def _patched(q: Any) -> str:
        if _question_identity(q) != target_identity:
            return original(q)
        assert q.open_time is not None, "question.open_time is required"
        assert q.scheduled_resolution_time is not None, "question.scheduled_resolution_time is required"
        elapsed_days = (mid_window - q.open_time).days
        remaining_days = (q.scheduled_resolution_time - mid_window).days
        return (
            f"Today: {mid_window.strftime('%Y-%m-%d')}\n"
            f"Question opened: {q.open_time.strftime('%Y-%m-%d')} ({elapsed_days} days ago)\n"
            f"Scheduled to resolve: {q.scheduled_resolution_time.strftime('%Y-%m-%d')} "
            f"({remaining_days} days from now)\n"
            f"Forecasting window: open date → resolution date. "
            f"Events occurring BEFORE the open date do NOT resolve this question YES "
            f"unless the resolution criteria explicitly say they count. "
            f"If the question uses forward-looking language ('will X occur by DATE'), "
            f"interpret it as asking about the open→resolution window, not all of history."
        )

    setattr(prompts_module, "_forecasting_window_str", _patched)
    _window_patch_active = True
    try:
        yield
    finally:
        setattr(prompts_module, "_forecasting_window_str", original)
        _window_patch_active = False


@contextmanager
def patched_gap_fill_year_for_question(question: MetaculusQuestion | Any) -> Iterator[None]:
    """Patch ``gap_fill_analyzer_prompt`` to neutralize the ``{datetime.now().year}`` leak.

    The analyzer prompt interpolates the current year into a "stale info"
    rubric ("e.g., no 2026 data on a near-term question"). For ablation
    backtests we substitute a year that does not leak the question's
    resolution timing: ``scheduled_resolution_time.year - 1``.
    """
    assert question.scheduled_resolution_time is not None, "question.scheduled_resolution_time is required"
    replacement_year = question.scheduled_resolution_time.year - 1
    original = prompts_module.gap_fill_analyzer_prompt

    def _patched(*args: Any, **kwargs: Any) -> str:
        rendered = original(*args, **kwargs)
        pattern = rf"\bno {datetime.now().year} data\b"
        return re.sub(pattern, f"no {replacement_year} data", rendered)

    setattr(prompts_module, "gap_fill_analyzer_prompt", _patched)
    try:
        yield
    finally:
        setattr(prompts_module, "gap_fill_analyzer_prompt", original)


@contextmanager
def patched_window_and_year_for_question(question: MetaculusQuestion | Any) -> Iterator[None]:
    """Apply both ``patched_window_for_question`` and ``patched_gap_fill_year_for_question``."""
    with patched_window_for_question(question), patched_gap_fill_year_for_question(question):
        yield
