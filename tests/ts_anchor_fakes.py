"""Shared fakes + question factories for the time-series-anchor test files.

Split out because three test modules key on the same two shapes and duplicating either
would let them drift apart:

- ``FakeHttp`` / ``_csv`` — the ``ts_fetch._http_get`` seam (raw CSV bytes by URL prefix).
  Used by ``test_ts_fetch.py`` (real parse + leakage guard) and by
  ``test_timeseries_anchor_provider.py``'s soft-fail tests (malformed / leaky responses).
- ``_make_numeric_q`` / ``_make_discrete_q`` / ``_DGS10_RC`` — the question mock every
  routing, render, provider and guard test builds on. Used by ``test_ts_routing.py`` and
  ``test_timeseries_anchor_provider.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from forecasting_tools import NumericQuestion
from forecasting_tools.data_models.questions import DiscreteQuestion

# A resolution-criteria string that routes deterministically to a non-revising FRED series.
_DGS10_RC = "Resolves per https://fred.stlouisfed.org/series/DGS10 on the resolution date."


class FakeHttp:
    """Drop-in for ``ts_fetch._http_get`` dispatching by URL prefix to the raw CSV
    bytes that prefix should return."""

    def __init__(self, handlers: dict[str, bytes]):
        self._handlers = handlers
        self.calls: list[tuple[str, dict[str, str]]] = []

    def __call__(self, url: str, params: dict[str, str]) -> bytes:
        self.calls.append((url, dict(params)))
        for prefix, body in self._handlers.items():
            if url.startswith(prefix):
                return body
        raise AssertionError(f"no handler for URL {url}")


def _csv(header_value_col: str, rows: list[tuple[str, str]]) -> bytes:
    body = f"observation_date,{header_value_col}\n" + "".join(f"{d},{v}\n" for d, v in rows)
    return body.encode("utf-8")


def _make_numeric_q(
    *,
    qid: int = 7001,
    question_text: str = "What will X be?",
    resolution_criteria: str = "rc",
    fine_print: str = "",
    open_time: datetime | None = None,
    scheduled_resolution_time: datetime | None = datetime(2027, 1, 1, tzinfo=UTC),
    lower_bound: float = 0.0,
    upper_bound: float = 1000.0,
    open_lower_bound: bool = False,
    open_upper_bound: bool = False,
) -> MagicMock:
    """A ``MagicMock(spec=NumericQuestion)`` with the fields the provider reads set to
    real values (unset MagicMock attrs are truthy and would corrupt routing / isinstance,
    and the bounds backstop needs real numeric bounds). The wide default range [0, 1000]
    comfortably contains the synthetic-series bands, so the backstop is a no-op unless a
    test opts into a mismatched range."""
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = qid
    q.question_text = question_text
    q.resolution_criteria = resolution_criteria
    q.fine_print = fine_print
    q.title = question_text
    q.open_time = open_time if open_time is not None else datetime(2026, 3, 15, tzinfo=UTC)
    q.scheduled_resolution_time = scheduled_resolution_time
    q.lower_bound = lower_bound
    q.upper_bound = upper_bound
    q.open_lower_bound = open_lower_bound
    q.open_upper_bound = open_upper_bound
    q.page_url = f"https://www.metaculus.com/questions/{qid}/"
    return q


def _make_discrete_q(**kwargs) -> MagicMock:
    """A ``DiscreteQuestion``-spec'd twin of ``_make_numeric_q``.

    ``DiscreteQuestion`` subclasses ``NumericQuestion``, so the provider's ``isinstance``
    gate admits it and routing (which is text-only) must reach the same verdict. A real
    subclass spec — not just a relabelled numeric mock — is what makes that a real check."""
    q = _make_numeric_q(**kwargs)
    discrete = MagicMock(spec=DiscreteQuestion)
    for attr in (
        "id_of_question",
        "question_text",
        "resolution_criteria",
        "fine_print",
        "title",
        "open_time",
        "scheduled_resolution_time",
        "lower_bound",
        "upper_bound",
        "open_lower_bound",
        "open_upper_bound",
        "page_url",
    ):
        setattr(discrete, attr, getattr(q, attr))
    return discrete
