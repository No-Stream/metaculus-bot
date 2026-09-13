"""Shared record builders and reference implementations for the performance-analysis tests.

One canonical home for the three record/post factories the analysis-cut, collector, dataset and
report modules all build their fixtures from, the fake ``requests`` response the collector's retry
tests serve, and the pre-fix PIT interpolation kept only so the value-grid tests can prove the
regression against it. All of it used to live at the top of
``tests/test_performance_analysis_extended.py``, which grew past the monolithic-file limit and was
split by concern into the ``tests/test_performance_analysis_*.py`` modules that now import from
here.

Not named ``test_*`` on purpose: pytest must import this module without collecting it.

Nothing here opens a socket or bills a key; the response object is a plain stub.
"""

from __future__ import annotations

import requests


def _old_interpolate_pit(resolution: float, lower_bound: float, upper_bound: float, cdf_values: list[float]) -> float:
    """The pre-fix linear-index implementation, kept here only to prove the regression.

    Maps the resolution to a CDF index assuming a LINEAR value grid. Correct for
    linear-scaled questions, wrong for log-scaled (zero_point) ones.
    """
    total_range = upper_bound - lower_bound
    if total_range <= 0:
        return 0.5
    fraction = (resolution - lower_bound) / total_range
    n = len(cdf_values)
    idx_float = fraction * (n - 1)
    idx_low = max(0, min(int(idx_float // 1), n - 2))
    idx_high = idx_low + 1
    weight = idx_float - idx_low
    return cdf_values[idx_low] * (1 - weight) + cdf_values[idx_high] * weight


def _binary_record(
    post_id: int,
    prob_yes: float,
    resolution: bool,
    per_model: dict[str, str] | None = None,
    category: str | None = None,
    **stacker_fields: object,
) -> dict:
    """Build a binary record. ``stacker_fields`` sets the stacker-detection
    signals (``was_stacked``, ``stacker_outcome``, ``comment_text``) that
    ``per_model_cohort`` reads; omit them for an ordinary unstacked record."""
    return {
        "post_id": post_id,
        "type": "binary",
        "our_prob_yes": prob_yes,
        "our_forecast_values": [1.0 - prob_yes, prob_yes],
        "resolution_parsed": resolution,
        "brier_score": (prob_yes - (1.0 if resolution else 0.0)) ** 2,
        "log_score": 0.0,
        "numeric_log_score": None,
        "mc_log_score": None,
        "per_model_forecasts": per_model or {},
        "metadata": {"category": category},
        **stacker_fields,
    }


def _binary_post(
    post_id: int,
    question_id: int,
    resolution: str = "yes",
    score_data: dict[str, float] | None = None,
) -> dict:
    """One resolved binary post in the shape ``fetch_resolved_questions`` returns.

    ``score_data`` defaults to a peer-scored forecast; pass ``{}`` for a post whose
    forecast carries no platform scores. ``nr_forecasters`` is deliberately set on the
    QUESTION here as a decoy, since the collector must read the crowd size off the post.
    """
    return {
        "id": post_id,
        "title": f"Q{post_id}",
        "question": {
            "id": question_id,
            "type": "binary",
            "resolution": resolution,
            "my_forecasts": {
                "latest": {
                    "forecast_values": [0.3, 0.7],
                    "score_data": {"peer_score": 1.0} if score_data is None else score_data,
                },
            },
            "scaling": {},
            "options": None,
            "open_lower_bound": False,
            "open_upper_bound": False,
            "nr_forecasters": 5,
            "title": f"Q{post_id}",
        },
        "projects": {},
    }


def _numeric_record(
    post_id: int,
    cdf: list[float],
    resolution: float,
    lower: float = 0.0,
    upper: float = 100.0,
    category: str | None = None,
) -> dict:
    return {
        "post_id": post_id,
        "type": "numeric",
        "our_forecast_values": cdf,
        "resolution_parsed": resolution,
        "scaling": {"range_min": lower, "range_max": upper},
        "open_lower_bound": False,
        "open_upper_bound": False,
        "brier_score": None,
        "log_score": None,
        "numeric_log_score": 0.0,
        "mc_log_score": None,
        "per_model_forecasts": {},
        "metadata": {"category": category},
    }


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload
