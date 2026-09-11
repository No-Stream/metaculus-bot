"""A date question viewed on the epoch-seconds axis, so the numeric pipeline can forecast it.

forecasting-tools and the Metaculus backend (which Mantic forked) both treat a date question as a
numeric question whose x-axis is epoch seconds: ``NumericDistribution.from_question`` converts the
datetime bounds with ``.timestamp()`` and sets ``is_date``, ``DateReport`` inherits
``NumericReport``'s publish verbatim, and the server validates a date CDF with the same
type-agnostic rules (length ``inbound + 1``, per-bin min and max step, bound pins). The repo's
numeric pipeline is typed on ``NumericQuestion`` with float bounds, so a ``DateQuestion`` is viewed
through :class:`EpochDateQuestion`, a real ``NumericQuestion`` subclass: every ``isinstance`` gate
in the sanitize, PCHIP, aggregation and publish path passes, and dates are converted back only
where a human reads them (the prompt's bound messages and parse notes, the gap-fill brief) or the
comment renders them (``is_date`` on the published distribution).

Two conventions live here and nowhere else:

* **Nominal bounds are READ, never derived.** Mantic sets ``nominal_max`` to the LAST bin's left
  edge on a day-granularity date question (post 651: ``range_max`` 2026-09-20T00:00Z,
  ``nominal_max`` 2026-09-19T00:00Z, the last answer date), while the repo's half-step derivation
  in ``numeric.utils.nominal_bounds`` implements the discrete centre-aligned convention. The
  adapter always carries nominal bounds, so that derivation is never entered for a date question.
* **A date-only value means NOON UTC of that day.** The platform buckets a resolution on calendar
  day D into the right-closed bin ``(D 00:00, D+1 00:00]``, so a forecaster's "2026-09-16" must land
  strictly inside the day rather than on the edge the 1e-10 bucket fudge decides.

Every timestamp is UTC. A naive value is UTC, never host-local: forecasting-tools' own date
template calls ``.timestamp()`` on a naive datetime, an 8-hour error on a Pacific-time laptop.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Any

from forecasting_tools.data_models.questions import DateQuestion, MetaculusQuestion, NumericQuestion

from metaculus_bot.question_types import QuestionType
from metaculus_bot.time_utils import _as_utc

__all__ = [
    "DATE_GRANULARITIES",
    "EpochDateQuestion",
    "as_epoch_question",
    "format_epoch",
    "numeric_qtype",
    "numeric_view",
    "parse_forecast_date",
    "question_json",
    "to_epoch",
]

# Mantic's ``date_granularity`` enum (``month`` is in its rules doc but not in the API; see module docstring).
DATE_GRANULARITIES: frozenset[str] = frozenset({"", "day", "week"})

_DATE_ONLY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_DATE_TIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:?\d{2})?")

_DATE_ONLY_HOUR_UTC = 12


class EpochDateQuestion(NumericQuestion):
    """A ``DateQuestion`` with its bounds in epoch seconds.

    Built only by :func:`as_epoch_question`. ``lower_bound`` / ``upper_bound`` are the
    ``.timestamp()`` of the tz-aware datetime bounds, ``nominal_lower_bound`` /
    ``nominal_upper_bound`` come from the API's ``scaling`` block (see the module docstring), and
    ``date_granularity`` is the platform's bin label, used only to pick a rendering in
    :func:`format_epoch`. Everything else is the source question's own field, copied.
    """

    date_granularity: str = ""


def to_epoch(moment: datetime) -> float:
    """Epoch seconds of ``moment``; a naive value is read as UTC."""
    return _as_utc(moment).timestamp()


def parse_forecast_date(text: str) -> datetime:
    """Parse a forecaster's strict ISO-8601 date or timestamp to tz-aware UTC.

    ``YYYY-MM-DD`` maps to 12:00:00 UTC of that day (module docstring: noon sits inside the
    platform's right-closed day bin). ``YYYY-MM-DDTHH:MM[:SS]`` is taken as written, with a
    ``Z`` or offset honoured and a naive time read as UTC. Anything else raises ``ValueError``,
    including a bare year, a year-month, and every non-ISO spelling.

    The two regexes are deliberately narrower than ``datetime.fromisoformat``, which also admits
    the basic ``YYYYMMDD`` and ISO-week forms nobody asked a forecaster for. This is the reader
    for a FORECAST VALUE and nothing else; the archive reader is ``time_utils.parse_iso_utc``,
    which is lenient, answers None on a bad value and keeps a date-only value at midnight. The
    two are named apart because a date read through the archive reader lands on a bin edge,
    where the platform's 1e-10 bucket fudge decides the day, with no error anywhere.
    """
    candidate = text.strip()
    if _DATE_ONLY_RE.fullmatch(candidate):
        day = date.fromisoformat(candidate)
        return datetime(day.year, day.month, day.day, _DATE_ONLY_HOUR_UTC, tzinfo=UTC)
    if _DATE_TIME_RE.fullmatch(candidate):
        return _as_utc(datetime.fromisoformat(candidate))
    raise ValueError(f"not a strict ISO-8601 date or timestamp: {text!r}")


def format_epoch(value: float, granularity: str) -> str:
    """Render an epoch-seconds value as the date a forecaster should read.

    ``YYYY-MM-DD`` for a ``day`` or ``week`` granularity question, whose bins are whole days and
    whose displayed bounds are calendar dates; ``YYYY-MM-DDTHH:MM:SSZ`` otherwise, since a
    legacy 200-bin date question's edges fall at arbitrary times of day.
    """
    moment = datetime.fromtimestamp(value, tz=UTC)
    if granularity in ("day", "week"):
        return moment.strftime("%Y-%m-%d")
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def question_json(question: MetaculusQuestion) -> dict[str, Any]:
    """The API's ``question`` object off ``api_json``, or ``{}`` for a question not built from API JSON.

    Mantic's per-question fields (``scaling``, ``date_granularity``, ``multi_resolution``,
    ``precision``) live under that key and nowhere on the ``forecasting_tools`` model. Metaculus
    payloads never carry the Mantic flags, so a prompt clause keyed on one disables itself there
    without a run-mode flag. The one accessor for that read, so every consumer tolerates a
    test-built question the same way.
    """
    payload = question.api_json.get("question")
    return payload if isinstance(payload, dict) else {}


def _scaling_block(payload: dict[str, Any]) -> dict[str, Any]:
    """The ``scaling`` dict of a ``question_json`` payload, empty when absent."""
    scaling = payload.get("scaling")
    return scaling if isinstance(scaling, dict) else {}


def _read_granularity(question: DateQuestion, payload: dict[str, Any]) -> str:
    granularity = payload.get("date_granularity") or ""
    if granularity not in DATE_GRANULARITIES:
        raise ValueError(
            f"Question {question.id_of_question}: unknown date_granularity {granularity!r}; "
            f"expected one of {sorted(DATE_GRANULARITIES)}"
        )
    return granularity


def as_epoch_question(question: DateQuestion) -> EpochDateQuestion:
    """The numeric-pipeline view of ``question``: the same question, bounds in epoch seconds.

    Nominal bounds are the API's ``scaling.nominal_min`` / ``nominal_max`` (epoch floats on the
    wire for a date question), falling back to the range bounds only when the payload carries
    none, which is the shape of a question built in a test rather than from API JSON. Both are
    read rather than derived because Mantic's date convention (``range_max = nominal_max + one
    bin``) and the repo's discrete half-step convention disagree.
    """
    lower_bound = to_epoch(question.lower_bound)
    upper_bound = to_epoch(question.upper_bound)
    payload = question_json(question)
    scaling = _scaling_block(payload)
    nominal_min = scaling.get("nominal_min")
    nominal_max = scaling.get("nominal_max")
    base_fields = question.model_dump(
        exclude={
            "question_type",
            "upper_bound",
            "lower_bound",
            "open_upper_bound",
            "open_lower_bound",
            "zero_point",
            "cdf_size",
            "previous_forecasts",  # NumericQuestion narrows its element type; the pipeline never reads it
        }
    )
    return EpochDateQuestion(
        **base_fields,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        open_upper_bound=question.open_upper_bound,
        open_lower_bound=question.open_lower_bound,
        zero_point=question.zero_point,
        cdf_size=question.cdf_size,
        nominal_lower_bound=float(nominal_min) if nominal_min is not None else lower_bound,
        nominal_upper_bound=float(nominal_max) if nominal_max is not None else upper_bound,
        date_granularity=_read_granularity(question, payload),
    )


def numeric_view(question: MetaculusQuestion) -> NumericQuestion:
    """The question the numeric math runs on: itself for a numeric question, the adapter for a date one."""
    if isinstance(question, NumericQuestion):
        return question
    if isinstance(question, DateQuestion):
        return as_epoch_question(question)
    raise TypeError(f"{type(question).__name__} has no numeric view")


def numeric_qtype(question: NumericQuestion) -> QuestionType:
    """The telemetry type of a numeric-axis question: ``date`` for the adapter, ``numeric`` otherwise.

    ``question_types.question_type_of`` cannot tell the two apart (the adapter IS a
    ``NumericQuestion``, and that leaf module may not import this package), so the two emitters
    that see the adapter (the member and stacker ``MEMBER_FORECAST`` lines) ask here.
    """
    return "date" if isinstance(question, EpochDateQuestion) else "numeric"
