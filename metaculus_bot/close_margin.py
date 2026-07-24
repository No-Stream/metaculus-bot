"""Per-question CLOSE_MARGIN telemetry.

Emits a durable ``CLOSE_MARGIN`` marker line at submission time so the telemetry
pipeline can track how much of each question's open→close window remained when we
submitted. A shrinking margin (especially below ~30% of the window) is the
early-warning signal that GitHub Actions cron starvation is pushing submissions
toward the close deadline — and a missed close forfeits the whole spot score
(2026-07-18 latency/completeness audit: submission-latency p90 crept 24→58 min,
worst-case queue+pipeline already breaches the 90-minute question windows).

Emitted format (the source of truth for ``scripts/telemetry/markers.py``)::

    CLOSE_MARGIN: question=<id> close_time=<iso> submitted_at=<iso> \
        window_s=<int|n/a> margin_s=<int> margin_frac=<float|n/a>

``margin_s`` (close − submit, seconds) is always computable once ``close_time``
exists; ``window_s`` (open → close) and ``margin_frac`` (margin / window) need
``open_time`` and a positive window, else render ``n/a`` (matching the marker
parser's None sentinel). Returns ``None`` — skip, emit nothing — when
``close_time`` is absent: a question with no deadline has no margin to track.
"""

from __future__ import annotations

from datetime import datetime

from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.time_utils import _as_utc


def format_close_margin_marker(question: MetaculusQuestion, submitted_at: datetime) -> str | None:
    """Build the ``CLOSE_MARGIN`` marker line for one submitted question, or None to skip."""
    close_time = question.close_time
    if close_time is None:
        return None

    close_utc = _as_utc(close_time)
    submitted_utc = _as_utc(submitted_at)
    margin_s = (close_utc - submitted_utc).total_seconds()

    window_repr = "n/a"
    frac_repr = "n/a"
    open_time = question.open_time
    if open_time is not None:
        window_s = (close_utc - _as_utc(open_time)).total_seconds()
        if window_s > 0:
            window_repr = str(round(window_s))
            frac_repr = f"{margin_s / window_s:.4f}"

    return (
        f"CLOSE_MARGIN: question={question.id_of_question} "
        f"close_time={close_utc.isoformat(timespec='seconds')} "
        f"submitted_at={submitted_utc.isoformat(timespec='seconds')} "
        f"window_s={window_repr} margin_s={round(margin_s)} margin_frac={frac_repr}"
    )
