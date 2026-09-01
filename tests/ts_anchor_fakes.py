"""Shared fakes + question factories for the time-series test files.

Split out because several test modules key on the same shapes and duplicating any of them
would let them drift apart:

- ``FakeHttp`` / ``_csv`` — the ``ts_fetch._http_get`` seam (raw CSV bytes by URL prefix).
  Used by ``test_ts_fetch.py`` (real parse + leakage guard) and by
  ``test_timeseries_anchor_provider.py``'s soft-fail tests (malformed / leaky responses).
- ``_make_numeric_q`` / ``_make_discrete_q`` / ``_DGS10_RC`` — the question mock every
  routing, render, provider and guard test builds on. Used by ``test_ts_routing.py`` and
  ``test_timeseries_anchor_provider.py``.
- ``random_walk_close_series`` / ``noise_dominated_close_series`` — the clean-versus-noisy
  price pair the variance-ratio screen is calibrated on. Used by
  ``test_timeseries_anchor_provider.py`` (the estimator itself) and
  ``test_financial_data_provider.py`` (the rendered noise flag), which is exactly the pair
  of consumers a private copy in either file would let disagree.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from forecasting_tools import NumericQuestion
from forecasting_tools.data_models.questions import DiscreteQuestion

# A resolution-criteria string that routes deterministically to a non-revising FRED series.
_DGS10_RC = "Resolves per https://fred.stlouisfed.org/series/DGS10 on the resolution date."

# Rows in the synthetic price pair below: ~265 business days is what the provider's own
# FINANCIAL_YFINANCE_LOOKBACK_DAYS window returns for an exchange-traded series, so the
# variance-ratio screen is calibrated at the sample size it will actually see in prod.
_SYNTHETIC_CLOSE_ROWS = 265
# Daily log-return sd of the underlying walk. 0.6%/day annualizes to ~9.5% on the
# trading-day basis — the order of magnitude USD/ZAR ran at in the q44797 window.
_SYNTHETIC_DAILY_SIGMA = 0.006


def random_walk_close_series(
    name: str = "CLEAN=X",
    *,
    seed: int = 0,
    n: int = _SYNTHETIC_CLOSE_ROWS,
    daily_sigma: float = _SYNTHETIC_DAILY_SIGMA,
    start: float = 16.4,
    end: str = "2026-07-17",
) -> pd.Series:
    """A clean geometric random walk on business days — the liquid-cross shape.

    Deterministic per ``seed``. Its variance ratio has no reason to sit away from 1 beyond
    sampling noise, which is what makes it the false-positive control for the screen."""
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(start * np.exp(np.cumsum(rng.normal(0.0, daily_sigma, n))), index=idx, name=name)


def noise_dominated_close_series(
    name: str = "NOISY=X",
    *,
    seed: int = 0,
    n: int = _SYNTHETIC_CLOSE_ROWS,
    daily_sigma: float = _SYNTHETIC_DAILY_SIGMA,
    quote_noise_sigma: float = _SYNTHETIC_DAILY_SIGMA,
    start: float = 16.4,
    end: str = "2026-07-17",
) -> pd.Series:
    """The pegged-cross shape: the same walk seen through independent quote noise.

    ``log p_t = log s_t + e_t`` with ``e`` iid, so every daily return carries
    ``e_t - e_{t-1}`` — a negatively autocorrelated component that inflates a one-day
    volatility estimate and then cancels over multi-day windows. At
    ``quote_noise_sigma == daily_sigma`` the noise is ~2/3 of return variance, which is the
    regime the q44797 verification measured on ``USDSZL=X`` (79% of variance vendor noise)
    and which puts VR(5) near 0.47 against ~1 for the walk above."""
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0.0, daily_sigma, n))
    quote_noise = rng.normal(0.0, quote_noise_sigma, n)
    return pd.Series(start * np.exp(walk + quote_noise), index=idx, name=name)


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
