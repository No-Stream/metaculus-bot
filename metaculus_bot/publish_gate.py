"""Pre-publish gate: never POST a forecast the question can no longer accept.

q45085 (2026-08-03) is the shape this exists for. The tournament list was fetched
at 11:59:38Z against a question that closed at 12:00:00Z; all three forecasters
returned, ``FORECASTERS_SURVIVED: survived=3/3`` logged, the median came out at
0.030 — and the publish POST came back ``405 {"error":"Question 45085 is already
closed to forecasting !"}`` at 12:05:06. Latency had already cost the question and
no local check can recover it, but crashing on a state we can read for free is
strictly worse than skipping: the ``HTTPError`` propagated out of ft's
per-question handler into ``cli.main``'s ``log_report_summary``, which raised
before the alertable block, so that run is the only forecasting run since
2026-07-26 with no end-of-run summary — and every degradation counter still read
zero.

So: compare the question's close time to now immediately before the publish, and
skip the WHOLE publish when the window has passed. Skipping the comment POST too
is deliberate — the published comment is the durable per-model record
``performance_analysis`` parses, so posting one for a forecast Metaculus never
accepted would seed the analysis with a forecast that does not exist on the
platform.

The close time read here is ``question.close_time`` (ft populates it from the
API's ``scheduled_close_time``), the same field ``close_margin`` reports on, so
the gate and the ``CLOSE_MARGIN`` marker can never disagree about the deadline.

**No safety margin, on purpose.** ft's publish body sleeps 3.5-4.5s twice
(``_sleep_between_requests``), so a question with seconds left can pass this gate
and still 405. Widening the gate to cover that would start skipping publishes that
WOULD have landed, and a forfeited question costs far more than a rejected POST —
which now costs one attempt instead of two, since ``publish_hardening`` no longer
retries a non-retryable 4xx. The residual race is left to the 405.

**Visible and alertable.** Each skip emits one ``PUBLISH_SKIPPED_CLOSED`` WARN
(specced in ``scripts/telemetry/markers.py``) and bumps a per-run counter that
rides the degradation line into ``alertable_count``, so CI goes red. A skip means
latency cost us a question, which is exactly what the operator wants paged. The run
continues — every other question still publishes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from forecasting_tools.data_models.questions import MetaculusQuestion, QuestionState

from metaculus_bot.time_utils import _as_utc

logger = logging.getLogger(__name__)

# Reason tags on the marker line. Two independent triggers, distinguishable in the
# archive because they mean different things about the run: the clock one is a
# latency problem we own, the state one is a question that was already shut when we
# picked it up.
REASON_CLOSE_TIME_PASSED = "close_time_passed"
REASON_STATE_CLOSED = "state_closed"

# States in which Metaculus rejects a forecast POST outright. ``state`` is a
# snapshot from list-fetch time, so it is NOT a substitute for the clock check (a
# question that closes mid-run still reads OPEN here) — it catches the complement:
# an admin closing or resolving a question ahead of its scheduled close, where the
# clock check would happily let the POST through. UPCOMING is deliberately absent:
# a not-yet-open question cannot reach the publish path from any mode the bot runs,
# and inventing a branch for it would be a guess about behavior we have never seen.
_CLOSED_STATES: frozenset[QuestionState] = frozenset({QuestionState.CLOSED, QuestionState.RESOLVED})

# Per-run count of questions whose publish was skipped because the question was
# already closed. Module-scoped for the same reason as
# ``publish_hardening._PUBLISH_ATTEMPT_FAILURES``: the gate runs inside a patched
# forecasting-tools method with no handle back to the bot. ``forecast_questions``
# resets it at run start. Incremented only from the event-loop thread (the gate
# runs in the async publish wrapper, before the to_thread offload), so unlike its
# publish_hardening sibling it needs no lock.
_PUBLISH_SKIPPED_CLOSED: int = 0


@dataclass(frozen=True)
class ClosedVerdict:
    """Why a question cannot accept a forecast, and by how much it missed."""

    reason: str
    close_time: datetime | None
    state: QuestionState | None
    overdue_s: float | None


def publish_skipped_closed_count() -> int:
    """Per-run count of publishes skipped because the question had already closed."""
    return _PUBLISH_SKIPPED_CLOSED


def reset_publish_skipped_closed() -> None:
    """Zero the counter at run start; without this it leaks across runs sharing a process."""
    global _PUBLISH_SKIPPED_CLOSED
    _PUBLISH_SKIPPED_CLOSED = 0


def closed_to_forecasting(question: MetaculusQuestion, now: datetime) -> ClosedVerdict | None:
    """Return why ``question`` can no longer be forecast, or None if it still can.

    The clock check is first and is the one that matters in prod: a question whose
    ``close_time`` has passed will reject the POST no matter what its cached state
    says.
    """
    close_time = question.close_time
    close_utc = _as_utc(close_time) if close_time is not None else None
    now_utc = _as_utc(now)
    overdue_s = (now_utc - close_utc).total_seconds() if close_utc is not None else None

    if close_utc is not None and now_utc >= close_utc:
        return ClosedVerdict(
            reason=REASON_CLOSE_TIME_PASSED,
            close_time=close_utc,
            state=question.state,
            overdue_s=overdue_s,
        )
    if question.state in _CLOSED_STATES:
        return ClosedVerdict(
            reason=REASON_STATE_CLOSED,
            close_time=close_utc,
            state=question.state,
            overdue_s=overdue_s,
        )
    return None


def format_publish_skipped_marker(question: MetaculusQuestion, verdict: ClosedVerdict, now: datetime) -> str:
    """Build the ``PUBLISH_SKIPPED_CLOSED`` marker line for one skipped publish.

    ``overdue_s`` can be negative under ``state_closed`` (closed ahead of schedule),
    which is informative rather than a bug — it says how much of the window the
    early close ate.
    """
    close_repr = verdict.close_time.isoformat(timespec="seconds") if verdict.close_time is not None else "n/a"
    overdue_repr = str(round(verdict.overdue_s)) if verdict.overdue_s is not None else "n/a"
    state_repr = verdict.state.value if verdict.state is not None else "n/a"
    return (
        f"PUBLISH_SKIPPED_CLOSED: question={question.id_of_question} "
        f"reason={verdict.reason} "
        f"close_time={close_repr} "
        f"now={_as_utc(now).isoformat(timespec='seconds')} "
        f"overdue_s={overdue_repr} "
        f"state={state_repr}"
    )


def skip_publish_if_closed(question: MetaculusQuestion | None, now: datetime | None = None) -> bool:
    """True when this question's publish must be skipped; counts and warns when so.

    Fails OPEN on a ``None`` question: the only caller is a wrapper around a
    forecasting-tools method, and if that seam ever stops carrying a question we
    want an unguarded publish (the status quo), never a silently withheld one.
    """
    global _PUBLISH_SKIPPED_CLOSED
    if question is None:
        return False

    moment = now if now is not None else datetime.now(timezone.utc)
    verdict = closed_to_forecasting(question, moment)
    if verdict is None:
        return False

    _PUBLISH_SKIPPED_CLOSED += 1
    logger.warning(format_publish_skipped_marker(question, verdict, moment))
    return True
