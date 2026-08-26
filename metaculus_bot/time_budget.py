"""Per-question time budget derived from the question's own close time.

The invariant this exists to establish: **a question with workable headroom at run
start always gets its PREDICTION submitted before close.** Operator mandate — a
question lost to our own latency is a big miss and should be impossible.

Before this module, the only per-question budget was
``PER_QUESTION_WALL_CLOCK_DEADLINE`` (3510 s), a constant sized against the
*cron period* (``METACULUS_CLOSE_WINDOW_SECONDS``), not against the question's
deadline. ``close_time`` was read in exactly two places, both downstream of every
paid call: ``close_margin`` (telemetry, after submission) and ``publish_gate``
(immediately before the POST). So a question closing in 20 minutes was handed the
same 58.5-minute budget as one closing in 2029, and the full pipeline's
*configured* worst case is ~1815 s (research 1155 + fan-out 600 + publish tail) —
30 minutes. The publish gate then caught the overrun and skipped, which is
correct but converts a savable question into a forfeit plus a full ensemble's
worth of spend.

``QuestionTimeBudget`` closes that by making the deadline the smaller of the
static cap and what the close time actually allows:

    total_s = min(static_deadline_s, close_time - now - PUBLISH_RESERVE_SECONDS)

Both existing budget consumers — the forecaster fan-out's ``asyncio.wait`` cap and
the stacking-skip gate — already read a remaining-seconds number, so they become
close-aware for free. The research phase, which consumed no budget at all, reads
``research_phase_deadline_s`` and ``fast_path``.

**Two clocks, on purpose.** ``total_s`` is derived from wall clock once at intake,
because that is the only way to compare against a calendar close time; elapsed is
then measured with ``time.monotonic()``, so a clock step (NTP correction on a
GitHub runner) cannot silently lengthen or shorten a live budget.

**Gated on publishing, not on benchmarking-ness.** ``build_question_time_budget``
takes ``close_aware`` and the caller passes ``publish_reports_to_metaculus`` — the
same gate ``close_margin`` uses. Backtests and ablations forecast *resolved*
questions whose close time is in the past; deriving a budget from that would hand
every one a negative budget and skip the whole run. A non-publishing run keeps
exactly the old static budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone

from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    PUBLISH_RESERVE_SECONDS,
    RESEARCH_PHASE_BUDGET_SHARE,
    TIME_BUDGET_FAST_PATH_THRESHOLD,
)
from metaculus_bot.time_utils import _as_utc


@dataclass(frozen=True)
class QuestionTimeBudget:
    """One question's wall-clock allowance, and how it is split across stages.

    ``total_s`` can be non-positive: a question fetched at or after its close (the
    q45085 shape, fetched 22 seconds before a close) cannot be published even by
    an instant forecast, and the honest budget for it is zero. Callers check
    ``is_exhausted`` at intake rather than spending an ensemble to discover it.
    """

    total_s: float
    started_at: float
    close_time: datetime | None
    close_limited: bool

    def elapsed_s(self) -> float:
        """Seconds spent on this question so far (monotonic, so clock-step safe)."""
        return time.monotonic() - self.started_at

    def remaining_s(self) -> float:
        """Seconds of budget left; goes negative once the budget is overrun."""
        return self.total_s - self.elapsed_s()

    @property
    def is_exhausted(self) -> bool:
        """True when there is no time to publish in, so no forecast can land."""
        return self.total_s <= 0.0

    @property
    def fast_path(self) -> bool:
        """True when the window is too thin for the full pipeline's worst case.

        Compared against ``total_s`` (the budget as granted) rather than
        ``remaining_s()`` so the answer is stable for the whole question: a
        mid-question flip would run half the optional stages and degrade the
        research bundle in a way no telemetry could reconstruct.
        """
        return self.total_s < TIME_BUDGET_FAST_PATH_THRESHOLD

    def research_phase_deadline_s(self) -> float:
        """Seconds the research phase may still consume, from now.

        ONE fixed research window — ``total_s * RESEARCH_PHASE_BUDGET_SHARE``,
        anchored at the budget's start — minus what has already elapsed. The
        window is deliberately NOT re-derived as a share of what remains at each
        call: research consults this at two sequential points (the provider
        phase, then each gap-fill pass), and taking 50% of the CURRENT remaining
        at each compounds to ~75% of the budget in the worst case, leaving the
        forecaster fan-out less than its own soft deadline on exactly the
        close-limited questions this budget exists to save. With a fixed window
        the forecast is guaranteed the complementary share whatever research
        does, and any pre-research spend (a slow intake) comes out of research's
        half rather than the forecast's. Never negative: an exhausted window
        yields 0, which cancels the provider phase immediately rather than
        reporting a nonsense deadline.

        Under the static 3510 s budget the window is ~1755 s, comfortably above
        the research phase's 1155 s configured worst case, so on a roomy question
        the deadline never fires and behavior is unchanged.
        """
        return max(0.0, self.total_s * RESEARCH_PHASE_BUDGET_SHARE - self.elapsed_s())


def build_question_time_budget(
    question: MetaculusQuestion,
    *,
    close_aware: bool,
    static_deadline_s: float,
    now: datetime | None = None,
) -> QuestionTimeBudget:
    """Grant one question its budget at intake.

    ``static_deadline_s`` is passed in rather than read here so the caller's
    module-level ``PER_QUESTION_WALL_CLOCK_DEADLINE`` stays the single knob tests
    monkeypatch.

    ``close_aware=False`` (any non-publishing run) returns the static budget and
    does not read ``close_time`` at all — on a backtest the field holds a past date
    that means nothing to the budget, so consulting it would only invite a reader of
    the marker line to think it bound something.
    """
    started_at = time.monotonic()
    if not close_aware:
        return QuestionTimeBudget(
            total_s=static_deadline_s,
            started_at=started_at,
            close_time=None,
            close_limited=False,
        )

    close_time = question.close_time
    close_utc = _as_utc(close_time) if close_time is not None else None
    if close_utc is None:
        return QuestionTimeBudget(
            total_s=static_deadline_s,
            started_at=started_at,
            close_time=None,
            close_limited=False,
        )

    moment = _as_utc(now) if now is not None else datetime.now(timezone.utc)
    close_allows_s = (close_utc - moment).total_seconds() - PUBLISH_RESERVE_SECONDS
    close_limited = close_allows_s < static_deadline_s
    return QuestionTimeBudget(
        total_s=min(static_deadline_s, close_allows_s),
        started_at=started_at,
        close_time=close_utc,
        close_limited=close_limited,
    )


def format_time_budget_marker(question: MetaculusQuestion, budget: QuestionTimeBudget) -> str:
    """Build the per-question ``TIME_BUDGET`` marker line.

    Emitted for EVERY question, not only thin ones: the archive currently has no
    way to say how often a thin window happens, because the only close-time
    telemetry (``CLOSE_MARGIN``) is emitted after a successful submission and is
    therefore censored on exactly the questions this feature exists for.
    """
    close_repr = budget.close_time.isoformat(timespec="seconds") if budget.close_time is not None else "n/a"
    return (
        f"TIME_BUDGET: question={question.id_of_question} "
        f"budget_s={round(budget.total_s)} "
        f"close_time={close_repr} "
        f"close_limited={str(budget.close_limited).lower()} "
        f"fast_path={str(budget.fast_path).lower()}"
    )
