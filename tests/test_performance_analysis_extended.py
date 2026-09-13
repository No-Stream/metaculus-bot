"""Tests for extended performance-analysis cuts added for residual analysis.

Covers: no_bias_check, financial_vs_nonfinancial_pit, stacking_effectiveness,
disagreement_predicts_error.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar, cast

import numpy as np
import pytest
import requests

from metaculus_bot import performance_analysis
from metaculus_bot.numeric.pchip_cdf import build_cdf_value_grid
from metaculus_bot.performance_analysis import collector
from metaculus_bot.performance_analysis.analysis import (
    PitReading,
    _interpolate_pit,
    _single_curve_pit,
    binary_summary,
    declared_percentile_pit,
    disagreement_predicts_error,
    financial_vs_nonfinancial_pit,
    max_step_clamp_screen,
    mc_summary,
    no_bias_check,
    numeric_pit_analysis,
    out_of_range_pit_reading,
    per_model_binary_scores,
    per_model_cohort,
    stacking_effectiveness,
)
from metaculus_bot.performance_analysis.collector import (
    _process_post,
    build_performance_dataset,
    load_dataset,
    rescore_records,
    resolve_numeric_record_to_score_inputs,
)
from metaculus_bot.performance_analysis.parsing import anonymous_model_key, is_anonymous_model_key


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# no_bias_check
# ---------------------------------------------------------------------------


class TestMcSummary:
    def test_a_short_forecast_vector_is_dropped_from_mean_prob_correct(self, caplog):
        """A forecast vector shorter than its option list cannot say what probability was
        on the winner. The old ``else 0.0`` scored that PARSE gap as "we gave the correct
        option zero", dragging mean_prob_correct down on a defect rather than a forecast;
        the record must still count in count / mean_mc_log_score."""
        normal = {
            "type": "multiple_choice",
            "mc_log_score": -0.5,
            "resolution_parsed": "B",
            "options": ["A", "B"],
            "our_forecast_values": [0.3, 0.7],
            "post_id": 1,
        }
        short = {
            "type": "multiple_choice",
            "mc_log_score": -0.9,
            "resolution_parsed": "C",
            "options": ["A", "B", "C"],
            "our_forecast_values": [0.6, 0.4],
            "post_id": 2,
        }
        with caplog.at_level("WARNING"):
            summary = mc_summary([normal, short])
        assert summary["count"] == 2
        assert summary["mean_prob_correct"] == pytest.approx(0.7), "the short record contributes nothing"
        assert summary["mean_mc_log_score"] == pytest.approx(-0.7)
        assert summary["accuracy"] == pytest.approx(0.5)
        assert "shorter than its option list" in caplog.text


class TestNoBiasCheck:
    def test_detects_no_bias(self):
        """Predicting 30% when the actual YES rate is 43% is a -13pp NO-bias."""
        records = [_binary_record(i, 0.30, True) for i in range(43)] + [
            _binary_record(100 + i, 0.30, False) for i in range(57)
        ]
        result = no_bias_check(records)
        assert result["count"] == 100
        assert result["mean_predicted"] == pytest.approx(0.30)
        assert result["actual_yes_rate"] == pytest.approx(0.43)
        assert result["bias_pp"] == pytest.approx(-13.0)

    def test_reports_low_range_subset(self):
        """20 records inside the 0.10-0.30 bucket: mean predicted ~0.205, actual yes-rate 0.50
        (10 of 20 resolve YES). The 5 records at 0.70 sit outside the bucket and must not leak
        into the low_range stats."""
        low_range = (
            [_binary_record(i, 0.15, True) for i in range(4)]
            + [_binary_record(10 + i, 0.25, True) for i in range(6)]
            + [_binary_record(20 + i, 0.20, False) for i in range(10)]
        )
        other = [_binary_record(100 + i, 0.70, True) for i in range(5)]
        result = no_bias_check(low_range + other)
        assert "low_range" in result
        lr = result["low_range"]
        assert lr["count"] == 20
        assert lr["mean_predicted"] == pytest.approx(0.205, abs=0.01)
        assert lr["actual_yes_rate"] == pytest.approx(0.50)

    def test_empty_data(self):
        assert no_bias_check([])["count"] == 0


# ---------------------------------------------------------------------------
# financial_vs_nonfinancial_pit
# ---------------------------------------------------------------------------


class TestFinancialVsNonfinancialPit:
    def test_splits_by_category(self):
        """Simple linear CDFs, so the PIT each record reads is predictable."""
        linear_cdf = [i / 200 for i in range(201)]
        records = [
            _numeric_record(1, linear_cdf, resolution=25.0, category="Economy & Business"),
            _numeric_record(2, linear_cdf, resolution=75.0, category="Economy & Business"),
            _numeric_record(3, linear_cdf, resolution=50.0, category="Science & Tech"),
        ]
        result = financial_vs_nonfinancial_pit(records)
        assert result["financial"]["count"] == 2
        assert result["nonfinancial"]["count"] == 1

    def test_unknown_category_goes_to_nonfinancial(self):
        linear_cdf = [i / 200 for i in range(201)]
        records = [_numeric_record(1, linear_cdf, resolution=50.0, category=None)]
        result = financial_vs_nonfinancial_pit(records)
        assert result["nonfinancial"]["count"] == 1
        assert result["financial"]["count"] == 0


# ---------------------------------------------------------------------------
# stacking_effectiveness
# ---------------------------------------------------------------------------


class TestStackingEffectiveness:
    def test_computes_counterfactual_mean_brier_on_triggered(self):
        """Triggered means the per-model probability range exceeds the threshold."""
        high_spread = _binary_record(
            1,
            prob_yes=0.50,
            resolution=True,
            per_model={"m1": "10%", "m2": "90%"},  # prob range 0.80
        )
        low_spread = _binary_record(
            2,
            prob_yes=0.50,
            resolution=True,
            per_model={"m1": "48%", "m2": "52%"},  # prob range 0.04
        )
        result = stacking_effectiveness([high_spread, low_spread], threshold=0.20)
        assert result["triggered_count"] == 1
        assert result["skipped_count"] == 1

    def test_empty_data(self):
        assert stacking_effectiveness([], threshold=0.15)["triggered_count"] == 0

    def test_boundary_exact_match_skips(self):
        exact_match = _binary_record(
            1,
            prob_yes=0.50,
            resolution=True,
            per_model={"m1": "40%", "m2": "60%"},  # prob range exactly 0.20
        )
        result = stacking_effectiveness([exact_match], threshold=0.20)
        assert result["triggered_count"] == 0
        assert result["skipped_count"] == 1


# ---------------------------------------------------------------------------
# disagreement_predicts_error
# ---------------------------------------------------------------------------


class TestDisagreementPredictsError:
    def test_positive_correlation_on_disagreement_and_error(self):
        """The records are built so the high-spread questions are also the high-Brier ones."""
        records = []
        for i in range(10):
            spread_tight = {"m1": f"{50 + i}%", "m2": f"{50 - i}%"}  # low spread
            records.append(_binary_record(i, 0.50, resolution=True, per_model=spread_tight))
        for i in range(10):
            # High spread, Brier gets large when prob_yes is wrong
            spread_wide = {"m1": "90%", "m2": "10%"}
            records.append(
                _binary_record(100 + i, 0.10, resolution=True, per_model=spread_wide)  # Brier = 0.81
            )
        result = disagreement_predicts_error(records)
        # High-spread bucket should have worse (higher) Brier
        assert result["count"] >= 20
        assert result["spearman_rho"] is not None
        assert result["spearman_rho"] > 0.3

    def test_handles_few_records(self):
        """Under 3 records there is no meaningful correlation to compute.

        Each record still carries a per_model dict, so it actually contributes to the spread
        correlation and the None comes from the record count rather than from empty input.
        """
        records = [
            _binary_record(1, 0.5, True, per_model={"m1": "40%", "m2": "60%"}),
            _binary_record(2, 0.6, True, per_model={"m1": "50%", "m2": "70%"}),
        ]
        result = disagreement_predicts_error(records)
        assert result["count"] == 2
        assert result["spearman_rho"] is None  # n<3


# ---------------------------------------------------------------------------
# per_model_cohort — phantom "Forecaster N" buckets and stacked records
# ---------------------------------------------------------------------------


class TestPerModelCohort:
    """Per-model cuts must see only named base models.

    Two ways a non-model entry reaches ``per_model_forecasts``: an anonymous
    positional key (no ``Model:`` line to attribute the bullet) and a
    stacker-fired record (the one summary bullet holds the stacker's aggregate,
    not a base model's forecast). Measured on the 2026-04 dataset, 50 such
    forecasts were being scored as if ``Forecaster 1`` and ``Forecaster 2`` were
    ensemble members, making that bucket a stacker-vs-base-model mixture.
    """

    def test_anonymous_keys_dropped_named_models_kept(self):
        record = _binary_record(
            1,
            prob_yes=0.60,
            resolution=True,
            per_model={"gpt-5.6-sol": "70%", "Forecaster 1": "50%", "Forecaster 2 base": "40%"},
        )
        [(returned, per_model)] = per_model_cohort([record], cut="unit_test")
        assert returned is record
        assert per_model == {"gpt-5.6-sol": "70%"}

    @pytest.mark.parametrize(
        "stacker_fields",
        [
            {"was_stacked": True},
            {"stacker_outcome": "primary"},
            {"stacker_outcome": "fallback_llm"},
            {"comment_text": "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=primary -->\n"},
            {"comment_text": "*Forecaster 1*: 70%\n<!-- STACKED=true -->\n"},
        ],
    )
    def test_stacker_fired_records_excluded_entirely(self, stacker_fields):
        stacked = _binary_record(
            1, prob_yes=0.60, resolution=True, per_model={"claude-opus-4.8": "70%"}, **stacker_fields
        )
        assert per_model_cohort([stacked], cut="unit_test") == []

    def test_median_records_kept(self):
        """The mirror of the stacker-fired case: a record the detector confirms ran on MEDIAN
        keeps its per-model bullets."""
        unstacked = _binary_record(
            1,
            prob_yes=0.60,
            resolution=True,
            per_model={"claude-opus-4.8": "70%"},
            stacker_outcome="skipped",
        )
        [(_record, per_model)] = per_model_cohort([unstacked], cut="unit_test")
        assert per_model == {"claude-opus-4.8": "70%"}

    def test_high_spread_record_without_stacker_signals_is_kept(self):
        """``likely_stacker`` (high spread plus a published value far from the median) must NOT
        exclude a record: that shape is also what a MEAN-era aggregate looks like, and dropping
        it would silently remove the high-disagreement records these cuts exist to measure."""
        wide = _binary_record(1, prob_yes=0.10, resolution=True, per_model={"m1": "90%", "m2": "10%"})
        [(_record, per_model)] = per_model_cohort([wide], cut="unit_test")
        assert per_model == {"m1": "90%", "m2": "10%"}

    def test_exclusions_are_logged_with_counts_and_reason(self, caplog):
        records = [
            _binary_record(1, 0.6, True, per_model={"gpt-5.6-sol": "70%", "Forecaster 1": "50%"}),
            _binary_record(2, 0.6, True, per_model={"Forecaster 1": "50%", "Forecaster 2": "40%"}),
            _binary_record(3, 0.6, True, per_model={"claude-opus-4.8": "70%"}, was_stacked=True),
        ]
        with caplog.at_level(logging.INFO, logger="metaculus_bot.performance_analysis.analysis"):
            per_model_cohort(records, cut="my_cut")

        [line] = [r.getMessage() for r in caplog.records if "PER_MODEL_COHORT" in r.getMessage()]
        assert "cut=my_cut" in line
        assert "eligible_records=2" in line
        assert "excluded_stacked_records=1" in line
        assert "excluded_stacked_observations=1" in line
        assert "excluded_anonymous_observations=3" in line
        assert "reason=" in line

    def test_per_model_binary_scores_excludes_phantoms(self):
        records = [
            _binary_record(1, 0.6, True, per_model={"gpt-5.6-sol": "70%", "Forecaster 1": "10%"}),
            _binary_record(2, 0.4, False, per_model={"gpt-5.6-sol": "30%", "Forecaster 1": "90%"}),
            # Stacker-fired: its bullet is the aggregate, not a base model.
            _binary_record(3, 0.6, True, per_model={"gemini-3.1-pro-preview": "70%"}, was_stacked=True),
        ]
        scores = per_model_binary_scores(records)
        assert set(scores) == {"gpt-5.6-sol"}
        assert scores["gpt-5.6-sol"]["count"] == 2

    def test_aggregate_cuts_still_include_excluded_records(self):
        """The aggregates keep stacked and anonymously-attributed records by decision; only the
        per-MODEL cuts drop them, so both aggregate paths must count all three records here."""
        records = [
            _binary_record(1, 0.6, True, per_model={"Forecaster 1": "60%"}),
            _binary_record(2, 0.6, True, per_model={"claude-opus-4.8": "70%"}, was_stacked=True),
            _binary_record(3, 0.4, False, per_model={"gpt-5.6-sol": "40%"}),
        ]
        assert binary_summary(records)["count"] == 3
        assert no_bias_check(records)["count"] == 3
        # ...while the per-model cut sees one named model on one question.
        assert set(per_model_binary_scores(records)) == {"gpt-5.6-sol"}

    def test_spread_cuts_skip_stacked_records(self):
        stacked_wide = _binary_record(
            1, 0.5, True, per_model={"Forecaster 1": "10%", "Forecaster 2": "90%"}, was_stacked=True
        )
        named_wide = _binary_record(2, 0.5, True, per_model={"m1": "10%", "m2": "90%"})
        effectiveness = stacking_effectiveness([stacked_wide, named_wide], threshold=0.20)
        assert effectiveness["triggered_count"] == 1
        assert effectiveness["skipped_count"] == 0

        correlation = disagreement_predicts_error([stacked_wide, named_wide])
        assert correlation["count"] == 1


class TestAnonymousModelKey:
    """The producer and the predicate must agree — they are what keeps the
    phantom filter from drifting away from the key format it filters on."""

    @pytest.mark.parametrize("index", [1, 3, 12])
    @pytest.mark.parametrize("is_base_model", [False, True])
    def test_produced_keys_are_recognized(self, index, is_base_model):
        assert is_anonymous_model_key(anonymous_model_key(index, is_base_model=is_base_model))

    @pytest.mark.parametrize(
        "key",
        [
            "gpt-5.6-sol",
            "claude-opus-4.8",
            "gemini-3.1-pro-preview",
            # Near-misses, spelled out in the docstring below.
            "Forecaster",
            "Forecaster One",
            "Forecaster 1 (gpt-5.6-sol)",
            "*Forecaster 1*",
        ],
    )
    def test_model_names_are_not_anonymous(self, key):
        """A real model name is never anonymous, and neither are the near-misses: display names
        that merely start the same way as the positional format, and a bullet-shaped string."""
        assert not is_anonymous_model_key(key)


# ---------------------------------------------------------------------------
# collector — the retry that keeps one timeout from abandoning a whole sweep
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class TestApiGetRetry:
    """``_api_get`` retries a transient network failure, and its wall clock stays bounded.

    A read timeout used to propagate on its first occurrence, so one slow page abandoned a
    whole tournament sweep and wrote nothing after several minutes of paging. Nine residual
    rounds worked around it with a pasted ``resilient_pull.py``. The retry budget is SHARED
    with the 429 budget on purpose: the worst-case wall clock is then identical to the
    429-only retry this replaces, which is what makes the change strictly safer in a path
    whose overrun costs forecasts.
    """

    def _install(self, monkeypatch, outcomes: list[object]) -> tuple[list[float], list[dict]]:
        """Serve ``outcomes`` (a response or an exception instance) in order; record every sleep."""
        sleeps: list[float] = []
        calls: list[dict] = []
        remaining = list(outcomes)

        def fake_get(url: str, headers: dict | None = None, params: dict | None = None, timeout: float | None = None):
            calls.append({"url": url, "headers": dict(headers or {}), "timeout": timeout})
            outcome = remaining.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        def fake_sleep(seconds: float) -> None:
            sleeps.append(seconds)

        monkeypatch.setattr(collector.requests, "get", fake_get)
        monkeypatch.setattr(collector.time, "sleep", fake_sleep)
        return sleeps, calls

    def test_a_read_timeout_is_retried_and_the_next_attempt_wins(self, monkeypatch):
        sleeps, calls = self._install(
            monkeypatch,
            [requests.exceptions.ReadTimeout("slow page"), _FakeResponse(200, {"results": [{"id": 7}]})],
        )

        assert collector._api_get("/posts/", "token") == {"results": [{"id": 7}]}
        assert len(calls) == 2
        assert sleeps == [collector.RETRY_BACKOFF_SECS]

    def test_a_connection_error_is_retried_too(self, monkeypatch):
        self._install(
            monkeypatch, [requests.exceptions.ConnectionError("reset by peer"), _FakeResponse(200, {"results": []})]
        )

        assert collector._api_get("/comments/", "token") == {"results": []}

    def test_exhausted_transient_retries_raise_the_underlying_requests_error(self, monkeypatch):
        timeout = requests.exceptions.ReadTimeout("still slow")
        sleeps, calls = self._install(monkeypatch, [timeout] * collector.MAX_RETRIES)

        with pytest.raises(requests.exceptions.ReadTimeout):
            collector._api_get("/posts/", "token")

        # Every attempt is spent, and the last one does not sleep before giving up.
        assert len(calls) == collector.MAX_RETRIES
        assert len(sleeps) == collector.MAX_RETRIES - 1

    def test_the_transient_and_429_budgets_are_shared_so_the_attempt_ceiling_is_unchanged(self, monkeypatch):
        outcomes: list[object] = [requests.exceptions.ReadTimeout("slow"), _FakeResponse(429), _FakeResponse(429)]
        assert len(outcomes) == collector.MAX_RETRIES, "this pin assumes the three-attempt budget"
        sleeps, calls = self._install(monkeypatch, outcomes)

        with pytest.raises(requests.exceptions.RequestException):
            collector._api_get("/posts/", "token")

        assert len(calls) == collector.MAX_RETRIES
        # Worst-case wall clock: MAX_RETRIES reads at the request timeout, plus these backoffs.
        assert sleeps == [collector.RETRY_BACKOFF_SECS, collector.RETRY_BACKOFF_SECS * 2]

    def test_an_exhausted_429_names_the_rate_limit_rather_than_reporting_one_unlucky_request(self, monkeypatch):
        self._install(monkeypatch, [_FakeResponse(429)] * collector.MAX_RETRIES)

        with pytest.raises(requests.exceptions.HTTPError, match="429 rate limit"):
            collector._api_get("/posts/", "token")

    def test_a_server_error_is_not_retried(self, monkeypatch):
        _sleeps, calls = self._install(monkeypatch, [_FakeResponse(500)])

        with pytest.raises(requests.exceptions.HTTPError):
            collector._api_get("/posts/", "token")

        assert len(calls) == 1

    def test_every_attempt_carries_the_bounded_request_timeout(self, monkeypatch):
        _sleeps, calls = self._install(
            monkeypatch, [requests.exceptions.ReadTimeout("slow"), _FakeResponse(200, {"results": []})]
        )

        collector._api_get("/posts/", "token")

        assert [call["timeout"] for call in calls] == [collector.REQUEST_TIMEOUT_SECS] * 2


# ---------------------------------------------------------------------------
# collector — the resolved-posts pull reads the list pages alone
# ---------------------------------------------------------------------------


class TestFetchResolvedQuestions:
    """The scoring pull is the list pages and nothing else.

    Under ``with_cp=true`` a list page's question dict carries everything the collector
    reads: type, resolution, scaling, the open bounds, the timestamps and the token's own
    ``my_forecasts`` with ``forecast_values`` and ``score_data``, on a single-question post
    and on a group post's members alike (verified live 2026-09-09 against the detail
    payload, where only post-level keys the collector never reads differ). The pull used to
    page the list for ids and then GET every post one at a time behind a 0.5 s sleep, about
    99 s per hundred posts against 1.8 s for the page that already held them all. The
    pins: every page asks for ``with_cp``, the payloads come back as the list served them,
    and no per-post GET is ever issued.
    """

    def _install_pages(self, monkeypatch, pages: list[list[dict]]) -> list[dict]:
        seen: list[dict] = []
        remaining = list(pages)

        def fake_api_get(path: str, token: str, params: dict | None = None) -> dict:
            seen.append({"path": path, "token": token, "params": dict(params or {})})
            page = remaining.pop(0)
            return {"results": page, "next": "next-page" if remaining else None}

        monkeypatch.setattr(collector, "_api_get", fake_api_get)
        monkeypatch.setattr(collector, "FETCH_DELAY_SECS", 0.0)
        return seen

    def test_returns_the_list_payloads_across_pages_with_no_per_post_get(self, monkeypatch):
        first_page = [_binary_post(1, 11), _binary_post(2, 22)]
        second_page = [_binary_post(3, 33)]
        seen = self._install_pages(monkeypatch, [first_page, second_page])

        posts = collector.fetch_resolved_questions("spring-aib-2026", "token")

        assert posts == first_page + second_page
        assert [call["path"] for call in seen] == ["/posts/", "/posts/"]
        assert [call["params"]["offset"] for call in seen] == [0, collector.PAGE_SIZE]

    def test_every_list_page_asks_for_the_tokens_own_forecasts(self, monkeypatch):
        seen = self._install_pages(monkeypatch, [[_binary_post(1, 11)], [_binary_post(2, 22)]])

        collector.fetch_resolved_questions("spring-aib-2026", "personal-token")

        assert len(seen) == 2
        for call in seen:
            assert call["params"]["with_cp"] == "true"
            assert call["params"]["tournaments"] == "spring-aib-2026"
            assert call["params"]["statuses"] == "resolved"
            assert call["params"]["limit"] == collector.PAGE_SIZE
            assert call["token"] == "personal-token"

    def test_an_empty_tournament_is_one_page_and_no_posts(self, monkeypatch):
        seen = self._install_pages(monkeypatch, [[]])

        assert collector.fetch_resolved_questions("spring-aib-2026", "token") == []
        assert len(seen) == 1


# ---------------------------------------------------------------------------
# collector — the comment pull covers public and private comments
# ---------------------------------------------------------------------------


class TestFetchBotComments:
    """The comment pull has to list the private comments as well as the public ones.

    The bot POSTs every comment with ``is_private: true`` and Metaculus flips older ones
    public server-side, so the default author listing served the 1,054 summer comments and
    none of the six fall ones (verified live 2026-09-09), which is a residual round whose
    records carry no comment text, no ``bot_comment_created_at`` and no per-model parse. The
    pins: both param sets are requested, the return is their union, a comment served by both
    appears once, and each listing still pages to exhaustion.
    """

    def _install_pages(
        self, monkeypatch, public_pages: list[list[dict]], private_pages: list[list[dict]]
    ) -> list[dict]:
        seen: list[dict] = []
        remaining = {False: list(public_pages), True: list(private_pages)}

        def fake_api_get(path: str, token: str, params: dict | None = None) -> dict:
            call_params = dict(params or {})
            seen.append({"path": path, "token": token, "params": call_params})
            pages = remaining[bool(call_params.get("is_private"))]
            page = pages.pop(0)
            return {"results": page, "next": "next-page" if pages else None}

        monkeypatch.setattr(collector, "_api_get", fake_api_get)
        monkeypatch.setattr(collector, "FETCH_DELAY_SECS", 0.0)
        return seen

    @staticmethod
    def _comment(comment_id: int, post_id: int) -> dict:
        return {"id": comment_id, "on_post": post_id, "text": f"*Forecaster 1*: {comment_id}%\n"}

    def test_returns_the_union_of_the_public_and_private_listings(self, monkeypatch):
        public = self._comment(1, 11)
        private = self._comment(2, 22)
        seen = self._install_pages(monkeypatch, [[public]], [[private]])

        comments = collector.fetch_bot_comments(275109, "bot-token")

        assert comments == [public, private]
        assert [call["path"] for call in seen] == ["/comments/", "/comments/"]
        assert [call["params"].get("is_private") for call in seen] == [None, "true"]
        for call in seen:
            assert call["params"]["author"] == 275109
            assert call["params"]["limit"] == collector.PAGE_SIZE
            assert call["token"] == "bot-token"

    def test_a_comment_served_by_both_listings_appears_once(self, monkeypatch):
        """Metaculus flips comments public in place, so the two listings overlap during the
        flip and a duplicate would give one post two records of the same forecast."""
        flipped = self._comment(7, 77)
        self._install_pages(monkeypatch, [[flipped, self._comment(8, 88)]], [[flipped]])

        comments = collector.fetch_bot_comments(275109, "bot-token")

        assert [c["id"] for c in comments] == [7, 8]

    def test_each_listing_pages_to_exhaustion(self, monkeypatch):
        public_pages = [[self._comment(1, 11)], [self._comment(2, 22)]]
        private_pages = [[self._comment(3, 33)], [self._comment(4, 44)], [self._comment(5, 55)]]
        seen = self._install_pages(monkeypatch, public_pages, private_pages)

        comments = collector.fetch_bot_comments(275109, "bot-token")

        assert [c["id"] for c in comments] == [1, 2, 3, 4, 5]
        public_offsets = [c["params"]["offset"] for c in seen if c["params"].get("is_private") is None]
        private_offsets = [c["params"]["offset"] for c in seen if c["params"].get("is_private") == "true"]
        assert public_offsets == [0, collector.PAGE_SIZE]
        assert private_offsets == [0, collector.PAGE_SIZE, 2 * collector.PAGE_SIZE]

    def test_an_author_with_no_comments_is_one_page_per_listing(self, monkeypatch):
        seen = self._install_pages(monkeypatch, [[]], [[]])

        assert collector.fetch_bot_comments(275109, "bot-token") == []
        assert len(seen) == 2


# ---------------------------------------------------------------------------
# collector — bot_comment_created_at field
# ---------------------------------------------------------------------------


class TestCollectorCommentCreatedAt:
    """Records produced by the collector should surface the comment's
    ``created_at`` timestamp so cohort cuts can filter by submit-date (vs the
    coarser actual_resolve_time on the question)."""

    def test_record_includes_bot_comment_created_at(self):
        post = _binary_post(1, 11)
        comment = {
            "id": 999,
            "text": "*Forecaster 1*: 70%\n",
            "on_post": 1,
            "created_at": "2026-04-30T12:34:56Z",
        }
        records = _process_post(post, {1: comment})
        assert len(records) == 1
        assert records[0]["bot_comment_created_at"] == "2026-04-30T12:34:56Z"

    def test_record_has_none_when_comment_missing(self):
        post = _binary_post(2, 22)
        records = _process_post(post, {})
        assert len(records) == 1
        assert records[0]["bot_comment_created_at"] is None

    def test_crowd_size_is_read_off_the_post_not_the_question(self):
        """``nr_forecasters`` is a POST field; reading it off the question dict with a 0
        default made it read 0 in all 2196 archived records — "never read", rendered as a
        measured empty crowd. This fixture deliberately keeps the decoy on the question."""
        post = _binary_post(5, 55)
        post["nr_forecasters"] = 170
        records = _process_post(post, {})

        assert records[0]["metadata"]["nr_forecasters"] == 170

    def test_a_post_with_no_crowd_field_reads_none_not_zero(self):
        """None means "the post didn't say", which a crowd-size cut can drop.

        A 0 would average a fabricated empty crowd into the cut, and would also silently kill
        audit.py's ``n/a`` fallback, since a real 0 is not a missing key.
        """
        post = _binary_post(6, 66)
        post.pop("nr_forecasters", None)
        records = _process_post(post, {})

        assert records[0]["metadata"]["nr_forecasters"] is None

    def test_record_has_none_when_comment_lacks_created_at(self):
        post = _binary_post(3, 33)
        comment = {"id": 1000, "text": "*Forecaster 1*: 70%\n", "on_post": 3}
        records = _process_post(post, {3: comment})
        assert len(records) == 1
        assert records[0]["bot_comment_created_at"] is None

    def test_stacker_skip_reason_marker_round_trips_onto_record(self):
        """The additive STACKER_SKIP_REASON marker must reach the record dict, because its
        documented durable path is the published comment rather than the run log."""
        post = _binary_post(4, 44)
        comment = {
            "id": 1001,
            "text": "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=skipped -->\n<!-- STACKER_SKIP_REASON=single_forecaster -->\n",
            "on_post": 4,
        }
        records = _process_post(post, {4: comment})
        assert len(records) == 1
        assert records[0]["stacker_skip_reason"] == "single_forecaster"
        assert records[0]["stacker_outcome"] == "skipped"

    def test_stacker_skip_reason_none_without_marker(self):
        post = _binary_post(5, 55)
        comment = {"id": 1002, "text": "*Forecaster 1*: 70%\n", "on_post": 5}
        records = _process_post(post, {5: comment})
        assert records[0]["stacker_skip_reason"] is None

    def test_practice_posts_produce_no_records(self):
        """Practice questions are not tournament scoring surface, so they must never enter the
        dataset; they would otherwise land in every calibration cut."""
        post = _binary_post(6, 66)
        post["title"] = "[PRACTICE] Will this be scored?"
        comment = {"id": 1003, "text": "*Forecaster 1*: 70%\n", "on_post": 6}
        assert _process_post(post, {6: comment}) == []


class TestCollectorExcludesDateQuestions:
    """The live bot forecasts date questions (on the epoch-seconds axis), so a resolved one
    reaches the collector; the residual dataset stays date-free by decision, the same exclusion
    ``backtest/question_prep.py`` and ``ablation/run_pdf.py`` carry at their seams. The skip has
    to be explicit and named in the log: ``parse_resolution``'s unknown-type fallthrough would
    file the same question as a parser bug."""

    @pytest.mark.parametrize("resolution", ["above_upper_bound", "2026-07-20T12:00:00Z"])
    def test_a_resolved_date_question_is_skipped_by_type_with_one_named_warning(self, resolution, caplog):
        """The fixture is the wire shape of a resolved date question.

        Taken from ``tests/data/mantic_series1_date_post_500_2026_09_08.json``: ``type`` is
        ``date``, the ``scaling`` bounds are epoch seconds, and the resolution is either an
        out-of-range token or an ISO timestamp.
        """
        post = _binary_post(500, 5000, resolution=resolution)
        post["question"]["type"] = "date"
        post["question"]["open_upper_bound"] = True
        post["question"]["scaling"] = {"range_min": 1781708400.0, "range_max": 1786536000.0, "zero_point": None}
        post["question"]["my_forecasts"]["latest"]["forecast_values"] = [0.0, 0.4, 0.9]
        comment = {"id": 1004, "text": "*Forecaster 1*: 2026-07-20\n", "on_post": 500}

        with caplog.at_level(logging.WARNING, logger="metaculus_bot.performance_analysis"):
            assert _process_post(post, {500: comment}) == []

        warnings = [(r.name, r.getMessage()) for r in caplog.records if r.levelno == logging.WARNING]
        (message,) = [text for name, text in warnings if name.endswith(".collector")]
        assert "Q5000" in message
        assert "date question" in message
        assert not any("Unknown question type" in text for _, text in warnings)


# ---------------------------------------------------------------------------
# collector — stacker_outcome / stacker_outcome_source fields
# ---------------------------------------------------------------------------


class TestCollectorStackerOutcome:
    """Records produced by the collector should expose the tri-state
    ``stacker_outcome`` plus its provenance, computed from
    ``parse_inferred_stacker_outcome`` over the comment text. The legacy
    ``was_stacked`` field collapses median-fallback into False, so analyses
    that need to distinguish "stacker LLM ran" from "MEDIAN fallback" must
    consume ``stacker_outcome``.
    """

    def _run(self, post_id: int, question_id: int, comment_text: str | None) -> dict:
        post = _binary_post(post_id, question_id)
        if comment_text is None:
            records = _process_post(post, {})
        else:
            records = _process_post(post, {post_id: {"id": 999, "text": comment_text, "on_post": post_id}})
        assert len(records) == 1
        return records[0]

    def test_outcome_marker_primary(self):
        rec = self._run(1, 11, "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=primary -->\n")
        assert rec["stacker_outcome"] == "primary"
        assert rec["stacker_outcome_source"] == "marker_outcome"

    def test_outcome_marker_fallback_median_distinguished_from_skipped(self):
        """The load-bearing case: this used to round-trip as ``STACKED=true`` and
        ``was_stacked=True``, with no way to tell median-fallback from primary. The richer
        ``stacker_outcome="fallback_median"`` is now preserved on the record."""
        rec = self._run(2, 22, "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=fallback_median -->\n")
        assert rec["stacker_outcome"] == "fallback_median"
        assert rec["stacker_outcome_source"] == "marker_outcome"

    def test_outcome_marker_skipped(self):
        rec = self._run(3, 33, "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=skipped -->\n")
        assert rec["stacker_outcome"] == "skipped"
        assert rec["stacker_outcome_source"] == "marker_outcome"

    def test_outcome_marker_skipped_config_off(self):
        """A config-suppressed skip (the per-type gate off despite high spread) must survive the
        collector round-trip distinct from a plain "skipped": this is the field the
        0-of-22-numeric-suppression re-attribution needed."""
        rec = self._run(3, 33, "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=skipped_config_off -->\n")
        assert rec["stacker_outcome"] == "skipped_config_off"
        assert rec["stacker_outcome_source"] == "marker_outcome"

    def test_legacy_marker_only_maps_to_primary(self):
        rec = self._run(4, 44, "*Forecaster 1*: 70%\n<!-- STACKED=true -->\n")
        assert rec["stacker_outcome"] == "primary"
        assert rec["stacker_outcome_source"] == "marker_legacy"

    def test_legacy_marker_false_maps_to_skipped(self):
        rec = self._run(5, 55, "*Forecaster 1*: 70%\n<!-- STACKED=false -->\n")
        assert rec["stacker_outcome"] == "skipped"
        assert rec["stacker_outcome_source"] == "marker_legacy"

    def test_historical_body_inferred_primary(self):
        """A pre-marker comment from the spring-aib-2026 dataset: no ``STACKED=`` or
        ``STACKER_OUTCOME=`` marker, but the Forecaster 1 body opens with
        "## Stacker Meta-Analysis", which only the stacker pipeline produces."""
        comment = (
            "# SUMMARY\n"
            "*Forecaster 1*: 70%\n\n"
            "## R1: Forecaster 1 Reasoning\n"
            "Model: openrouter/anthropic/claude-opus-4.7\n\n"
            "## Stacker Meta-Analysis\n\n"
            "Synthesis of 6 base models below.\n"
        )
        rec = self._run(6, 66, comment)
        assert rec["stacker_outcome"] == "primary"
        assert rec["stacker_outcome_source"] == "historical_body"

    def test_no_signal_returns_none(self):
        rec = self._run(7, 77, "*Forecaster 1*: 70%\n")
        assert rec["stacker_outcome"] is None
        assert rec["stacker_outcome_source"] == "none"

    def test_missing_comment_returns_none(self):
        rec = self._run(8, 88, None)
        assert rec["stacker_outcome"] is None
        assert rec["stacker_outcome_source"] == "none"

    def test_outcome_marker_takes_precedence_over_legacy(self):
        """Both markers coexist for one round of back-compat, so the collector must prefer the
        richer ``STACKER_OUTCOME=`` signal, or median-fallback is silently downgraded to
        "primary"."""
        comment = "*Forecaster 1*: 70%\n<!-- STACKER_OUTCOME=fallback_median -->\n<!-- STACKED=false -->\n"
        rec = self._run(9, 99, comment)
        assert rec["stacker_outcome"] == "fallback_median"
        assert rec["stacker_outcome_source"] == "marker_outcome"


# ---------------------------------------------------------------------------
# _interpolate_pit — value-grid-aware PIT (regression: log-scaled questions)
# ---------------------------------------------------------------------------


class TestInterpolatePit:
    """PIT = F(resolution). F must be read against the ACTUAL value grid the CDF
    lives on (linear for linear questions, geometric for zero_point questions),
    not against a linear index map. The old linear-index map mis-buckets
    log-scaled resolutions by up to ~0.24."""

    def test_linear_question_matches_old_behavior(self):
        """On a linear grid the two maps are mathematically equivalent, so the value-grid
        interpolation must equal the old linear-index one within float tolerance."""
        lower, upper = 0.0, 100.0
        cdf = [float(value) for value in np.linspace(0.0, 1.0, 201)]  # straight-line CDF
        grid = list(build_cdf_value_grid(lower, upper, None, num_points=201))
        for resolution in (0.0, 12.3, 25.0, 50.0, 73.7, 100.0):
            new = _interpolate_pit(resolution, lower, upper, cdf, value_grid=grid, zero_point=None)
            old = _old_interpolate_pit(resolution, lower, upper, cdf)
            assert new == pytest.approx(old, abs=1e-9)

    def test_linear_endpoints_and_midpoint(self):
        lower, upper = 0.0, 100.0
        cdf = [float(value) for value in np.linspace(0.0, 1.0, 201)]
        grid = list(build_cdf_value_grid(lower, upper, None, num_points=201))
        assert _interpolate_pit(lower, lower, upper, cdf, value_grid=grid) == pytest.approx(cdf[0])
        assert _interpolate_pit(upper, lower, upper, cdf, value_grid=grid) == pytest.approx(cdf[-1])
        assert _interpolate_pit(50.0, lower, upper, cdf, value_grid=grid) == pytest.approx(0.5)

    def test_log_scaled_question_differs_and_is_correct(self):
        """On a log-scaled (zero_point) question the value grid is geometric, so the resolution
        lands on a different CDF index than the linear-index map put it.

        Resolution 31.6 is only ~0.1% along the range linearly but a meaningful chunk of
        probability on the geometric grid, which is where the fix has to bite: the new reading
        is the geometric-midpoint PIT (~0.5), not the near-zero PIT the linear-index map gives.
        """
        lower, upper, zero_point = 1.0, 1000.0, 0.0
        cdf = [float(value) for value in np.linspace(0.0, 1.0, 201)]  # uniform-in-index CDF
        geo_grid = build_cdf_value_grid(lower, upper, zero_point, num_points=201)

        resolution = 31.6  # ~10^1.5 -> roughly the geometric midpoint of [1, 1000]

        new = _interpolate_pit(resolution, lower, upper, cdf, value_grid=list(geo_grid), zero_point=zero_point)
        old = _old_interpolate_pit(resolution, lower, upper, cdf)

        expected = float(np.interp(resolution, geo_grid, np.asarray(cdf, dtype=float)))
        assert new == pytest.approx(expected, abs=1e-12)

        # The fix must bite: geometric vs linear-index map differ materially here.
        assert abs(new - old) > 0.2
        # And the new value is the geometric-midpoint PIT, not the linear-index map's near-zero one.
        assert new == pytest.approx(0.5, abs=0.02)
        assert old < 0.05

    def test_falls_back_to_zero_point_grid_when_value_grid_absent(self):
        """With no continuous_range supplied the geometric grid is reconstructed from zero_point,
        and the result must match interpolation against that rebuilt grid."""
        lower, upper, zero_point = 1.0, 1000.0, 0.0
        cdf = [float(value) for value in np.linspace(0.0, 1.0, 201)]
        resolution = 31.6

        no_grid = _interpolate_pit(resolution, lower, upper, cdf, value_grid=None, zero_point=zero_point)
        rebuilt = build_cdf_value_grid(lower, upper, zero_point, num_points=201)
        expected = float(np.interp(resolution, rebuilt, np.asarray(cdf, dtype=float)))
        assert no_grid == pytest.approx(expected, abs=1e-12)

    def test_mismatched_value_grid_length_falls_back(self):
        """A value_grid whose length disagrees with the CDF is ignored, and the grid is rebuilt
        from the bounds and zero_point instead."""
        lower, upper = 0.0, 100.0
        cdf = [float(value) for value in np.linspace(0.0, 1.0, 201)]
        bad_grid = [0.0, 50.0, 100.0]  # wrong length
        result = _interpolate_pit(50.0, lower, upper, cdf, value_grid=bad_grid, zero_point=None)
        assert result == pytest.approx(0.5)

    def test_degenerate_range_raises_instead_of_answering_with_the_best_case(self):
        """A zero-width question has no PIT. This used to return 0.5, which is the single most
        favorable value available — inside BOTH coverage bands — so a degenerate record
        silently improved every calibration statistic it entered. The caller screens the
        range now (see ``TestDeclaredPercentilePitDropsDegenerateRanges``)."""
        cdf = [float(value) for value in np.linspace(0.0, 1.0, 201)]

        with pytest.raises(ValueError, match="degenerate question range"):
            _interpolate_pit(5.0, 10.0, 10.0, cdf)


class TestInterpolatePitOutOfGrid:
    """The q44218 shape: a resolution BEYOND the grid must not be censored at cdf[0]/cdf[-1].

    With below-bound mass expressible on open bounds, cdf[0] can be ~0.9, so the grid
    clamp reads a below-grid resolution — a LOW-tail event — as a high PIT. Beyond the
    grid the PIT must come off the members' declared-percentile curves instead.
    """

    _LOWER: ClassVar[float] = 100.0
    _UPPER: ClassVar[float] = 200.0
    # 90% of the mass below the open lower bound (F(100) = 0.90), like q44218's 0.9168.
    _CDF: ClassVar[list[float]] = list(np.linspace(0.90, 0.975, 201))
    _PERCENTILES: ClassVar[dict[str, list[list[float]]]] = {
        "model-a": [[10.0, 80.0], [50.0, 90.0], [90.0, 105.0]],
        "model-b": [[10.0, 85.0], [50.0, 95.0], [90.0, 110.0]],
    }

    def _grid(self) -> list[float]:
        return list(build_cdf_value_grid(self._LOWER, self._UPPER, None, num_points=201))

    def test_below_grid_resolution_reads_low_tail_not_the_clamp(self):
        """Resolution 50 sits below every declared value of every member, so each curve reads its
        lowest declared percentile (P10, giving 0.10). The grid clamp would have said 0.90."""
        pit = _interpolate_pit(
            50.0,
            self._LOWER,
            self._UPPER,
            self._CDF,
            value_grid=self._grid(),
            per_model_percentiles=self._PERCENTILES,
        )
        assert pit == pytest.approx(0.10, abs=1e-9)

    def test_fallback_is_median_of_member_curves(self):
        """At resolution 95 model-a interpolates to 0.6333 and model-b reads its P50 of 0.50."""
        pit = _interpolate_pit(
            95.0,
            self._LOWER,
            self._UPPER,
            self._CDF,
            value_grid=self._grid(),
            per_model_percentiles=self._PERCENTILES,
        )
        assert pit == pytest.approx((0.6333333 + 0.50) / 2, abs=1e-6)

    def test_no_member_curves_keeps_grid_read(self):
        """The degraded path, with no per-model percentiles recoverable, keeps the grid-endpoint
        read."""
        pit = _interpolate_pit(50.0, self._LOWER, self._UPPER, self._CDF, value_grid=self._grid())
        assert pit == pytest.approx(0.90, abs=1e-9)

    def test_at_bound_resolution_keeps_endpoint_read(self):
        """AT a bound the clamp IS the correct PIT, since F(bound) equals cdf[0], so the
        declared-percentile fallback must engage only strictly beyond the grid."""
        pit = _interpolate_pit(
            self._LOWER,
            self._LOWER,
            self._UPPER,
            self._CDF,
            value_grid=self._grid(),
            per_model_percentiles=self._PERCENTILES,
        )
        assert pit == pytest.approx(0.90, abs=1e-9)

    def test_numeric_pit_analysis_uses_declared_fallback(self):
        record = {
            "post_id": 1,
            "type": "numeric",
            "our_forecast_values": self._CDF,
            "resolution_parsed": 50.0,
            "scaling": {
                "range_min": self._LOWER,
                "range_max": self._UPPER,
                "zero_point": None,
                "continuous_range": self._grid(),
            },
            "open_lower_bound": True,
            "open_upper_bound": True,
            "numeric_log_score": 0.0,
            "per_model_numeric_percentiles": self._PERCENTILES,
            "metadata": {"category": None},
        }
        result = numeric_pit_analysis([record])
        assert result["count"] == 1
        assert result["pit_values"][0] == pytest.approx(0.10, abs=1e-9)


class TestDeclaredPercentilePitDropsDegenerateRanges:
    """A zero-width question contributes no PIT rather than the most favorable one.

    ``_interpolate_pit`` used to answer 0.5 there, which is inside both coverage bands, so a
    degenerate record silently improved every calibration statistic it entered.
    """

    @staticmethod
    def _record(range_min: float, range_max: float) -> dict:
        return {
            "post_id": 1,
            "type": "numeric",
            "our_forecast_values": list(np.linspace(0.0, 1.0, 201)),
            "resolution_parsed": 5.0,
            "scaling": {"range_min": range_min, "range_max": range_max, "zero_point": None},
            "open_lower_bound": False,
            "open_upper_bound": False,
            "numeric_log_score": 0.0,
            "metadata": {"category": None},
        }

    def test_zero_width_record_is_dropped_not_scored_at_half(self):
        assert numeric_pit_analysis([self._record(10.0, 10.0)]) == {"count": 0}

    def test_an_inverted_range_is_dropped_too(self):
        assert numeric_pit_analysis([self._record(10.0, 5.0)]) == {"count": 0}

    def test_a_real_range_still_scores(self):
        result = numeric_pit_analysis([self._record(0.0, 100.0)])

        assert result["count"] == 1


class TestDeclaredPercentileCurveTolerance:
    """Member curves come out of comment TEXT, so the fallback must tolerate junk.

    Every unusable curve reads as no-curve (dropped from the median) rather than
    raising or contributing a garbage quantile — the callers then either median the
    surviving curves or fall back to the grid read.
    """

    _GOOD: ClassVar[list[list[float]]] = [[10.0, 85.0], [50.0, 95.0], [90.0, 110.0]]

    def test_non_numeric_declared_value_drops_only_that_curve(self):
        """One percentile line parsed to a non-number, so the median is taken over the surviving
        curve alone (model-b at 50 reads its P10 of 0.10) rather than over a coerced zero that
        would drag the quantile."""
        curves = cast(
            "dict[str, list[list[float]]]",
            {"model-a": [[10.0, "n/a"], [50.0, 90.0], [90.0, 105.0]], "model-b": self._GOOD},
        )
        assert declared_percentile_pit(curves, 50.0) == pytest.approx(0.10, abs=1e-9)

    def test_pair_missing_its_value_is_unusable(self):
        """A truncated line recovered as a bare percentile with no value beside it."""
        assert _single_curve_pit([[10.0], [50.0]], 50.0) is None

    def test_anonymous_keys_are_excluded_from_the_median_of_members(self):
        """A positional ``Forecaster N`` bucket on a stacker-fired record holds the STACKER's
        aggregate, so pooling it into a median-of-members counts the aggregate as an extra
        member and pulls the median toward itself. ``max_step_clamp_screen`` next door and
        ``per_model_cohort`` both filter these; this consumer used not to."""
        curves = cast(
            "dict[str, list[list[float]]]",
            {"model-a": self._GOOD, "Forecaster 1": [[10.0, 10.0], [50.0, 12.0], [90.0, 14.0]]},
        )

        # The anonymous curve would read ~0.90 at resolution 50 and swing the median.
        assert declared_percentile_pit(curves, 50.0) == pytest.approx(0.10, abs=1e-9)

    def test_an_all_anonymous_record_yields_none_rather_than_the_stacker_curve(self):
        curves = cast("dict[str, list[list[float]]]", {"Forecaster 1": self._GOOD})

        assert declared_percentile_pit(curves, 50.0) is None

    def test_duplicate_declared_values_stay_usable(self):
        """A flat tail, where P10 equals P50, is legitimate model output, so jitter it into
        strict monotonicity rather than discarding the whole curve."""
        flat_tail = [[10.0, 80.0], [50.0, 80.0], [90.0, 105.0]]
        assert _single_curve_pit(flat_tail, 50.0) == pytest.approx(0.10, abs=1e-9)
        # Between the duplicated value and P90 the curve still interpolates.
        mid = _single_curve_pit(flat_tail, 90.0)
        assert mid is not None
        assert 0.5 < mid < 0.9

    def test_non_finite_declared_values_are_unusable(self):
        """Jitter cannot rescue non-finite values, so the curve must read as no-curve instead of
        returning a nan PIT into the median.

        ``np.errstate`` only silences the expected nan arithmetic inside the guard, which is the
        code under test here.
        """
        with np.errstate(invalid="ignore"):
            assert _single_curve_pit([[10.0, float("inf")], [90.0, float("inf")]], 50.0) is None
            assert _single_curve_pit([[10.0, float("nan")], [50.0, 90.0]], 50.0) is None

    def test_all_curves_unusable_reads_as_no_fallback(self):
        """``declared_percentile_pit`` returning None is what makes ``_interpolate_pit`` and
        ``compute_pit_details`` keep the grid-endpoint read."""
        junk = cast("dict[str, list[list[float]]]", {"model-a": [[50.0, "junk"]]})
        assert declared_percentile_pit(junk, 50.0) is None
        assert declared_percentile_pit(None, 50.0) is None


class TestMaxStepClampScreen:
    """The q43913 signature: a published bin pinned at the per-bin max-step cap while
    every member's own declared curve wanted materially more mass there.

    The cap is era-correct — flat 0.2 before the grid-scaled cap reached main (b4e9df0),
    the grid's own ``grid_step_constraints`` max after — so a post-fix coarse-grid
    discrete that legitimately holds a 0.2 bin must NOT fire. The fixture member curves
    are 11-ANCHOR on purpose: the screen drops any member under MIN_SCOREABLE_ANCHORS,
    because its verdict turns on the MINIMUM member bin mass and a 3-anchor interpolation
    across one bin is not the distribution the model declared.
    """

    # 11-point integer grid; steps[1] (the [1, 2] bin) is exactly 0.20.
    _GRID: ClassVar[list[float]] = [float(v) for v in range(11)]
    _CDF: ClassVar[list[float]] = [0.0, 0.05, 0.25, 0.45, 0.65, 0.85, 0.90, 0.93, 0.96, 0.98, 1.0]
    # Both members concentrate ~0.70 of their mass on the [1, 2] bin.
    _LABELS: ClassVar[list[float]] = [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0]
    _MEMBERS: ClassVar[dict[str, list[list[float]]]] = {
        "model-a": [
            [label, value]
            for label, value in zip(
                _LABELS,
                [0.90, 1.00, 1.15, 1.30, 1.45, 1.55, 1.70, 1.85, 2.00, 2.30, 2.60],
                strict=True,
            )
        ],
        "model-b": [
            [label, value]
            for label, value in zip(
                _LABELS,
                [0.95, 1.05, 1.18, 1.32, 1.46, 1.56, 1.72, 1.90, 2.05, 2.35, 2.65],
                strict=True,
            )
        ],
    }
    _PRE_FIX_TS = "2026-06-11T00:00:00Z"
    _POST_FIX_TS = "2026-08-01T00:00:00Z"

    def _record(
        self, *, submitted, members=None, cdf=None, grid=None, resolution: float | str = 1.4, q_type="discrete"
    ) -> dict:
        grid = grid if grid is not None else self._GRID
        return {
            "type": q_type,
            "our_forecast_values": cdf if cdf is not None else self._CDF,
            "resolution_parsed": resolution,
            "scaling": {"range_min": grid[0], "range_max": grid[-1], "continuous_range": grid},
            "bot_comment_created_at": submitted,
            "per_model_numeric_percentiles": members if members is not None else self._MEMBERS,
        }

    def test_pre_fix_coarse_grid_clamp_is_suspected(self):
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS))
        assert screen["suspected"] is True
        assert screen["submitted_before_grid_scaled_cap"] is True
        assert screen["max_step_cap"] == pytest.approx(0.2)
        assert screen["published_bin_mass"] == pytest.approx(0.2, abs=1e-9)
        assert screen["min_member_bin_mass"] > 0.6

    def test_post_fix_coarse_grid_point_two_bin_does_not_fire(self):
        """After 9f1175c an 11-point grid's cap is 1.0, so a 0.2 bin means nothing."""
        screen = max_step_clamp_screen(self._record(submitted=self._POST_FIX_TS))
        assert screen["suspected"] is False
        assert screen["submitted_before_grid_scaled_cap"] is False
        assert screen["max_step_cap"] == pytest.approx(1.0)
        assert screen["resolution_bin_at_cap"] is False

    def test_post_fix_standard_grid_cap_still_fires(self):
        """On the 201-point grid the era-correct cap is still 0.2, so the screen keeps catching
        genuine clamps after the fix."""
        steps = np.full(200, 0.8 / 199)
        steps[100] = 0.2
        cdf = np.concatenate([[0.0], np.cumsum(steps)]).tolist()
        grid = np.linspace(0.0, 200.0, 201).tolist()
        # 11-anchor curves (see _MEMBERS): each puts ~0.70 on the [100, 101] bin.
        members = {
            "model-a": [
                [label, value]
                for label, value in zip(
                    self._LABELS,
                    [99.90, 100.00, 100.15, 100.30, 100.45, 100.55, 100.70, 100.85, 101.00, 101.30, 101.60],
                    strict=True,
                )
            ],
            "model-b": [
                [label, value]
                for label, value in zip(
                    self._LABELS,
                    [99.95, 100.05, 100.18, 100.32, 100.46, 100.56, 100.72, 100.90, 101.05, 101.35, 101.65],
                    strict=True,
                )
            ],
        }
        screen = max_step_clamp_screen(
            self._record(submitted=self._POST_FIX_TS, members=members, cdf=cdf, grid=grid, resolution=100.5)
        )
        assert screen["max_step_cap"] == pytest.approx(0.2)
        assert screen["suspected"] is True

    def _fine_grid_members(self) -> dict[str, list[list[float]]]:
        """Three 11-anchor curves each putting ~0.65-0.77 on the [100, 101] bin."""
        value_rows = [
            [99.90, 100.00, 100.15, 100.30, 100.45, 100.55, 100.70, 100.85, 101.00, 101.30, 101.60],
            [99.95, 100.05, 100.18, 100.32, 100.46, 100.56, 100.72, 100.90, 101.05, 101.35, 101.65],
            [99.85, 99.98, 100.12, 100.28, 100.43, 100.53, 100.68, 100.83, 100.98, 101.28, 101.55],
        ]
        return {
            f"model-{name}": [[label, value] for label, value in zip(self._LABELS, values, strict=True)]
            for name, values in zip("abc", value_rows, strict=True)
        }

    def _fine_grid_record(self, realized_bin_mass: float) -> dict:
        steps = np.full(200, (1.0 - realized_bin_mass) / 199)
        steps[100] = realized_bin_mass
        cdf = np.concatenate([[0.0], np.cumsum(steps)]).tolist()
        grid = np.linspace(0.0, 200.0, 201).tolist()
        return self._record(
            submitted=self._POST_FIX_TS, members=self._fine_grid_members(), cdf=cdf, grid=grid, resolution=100.5
        )

    def test_post_snap_near_cap_bin_is_suspected(self):
        """The q45065 shape: the snap alpha shaves the realized bin ~1.1% under the 0.2
        cap (0.1977991526), so the exact-equality screen read it as clear while all
        three members declared 0.65-0.77 there. The near-cap ratio must catch it."""
        screen = max_step_clamp_screen(self._fine_grid_record(0.1977991526))
        assert screen["suspected"] is True
        assert screen["resolution_bin_at_cap"] is False
        assert screen["resolution_bin_cap_bound"] is True
        assert screen["resolution_bin_cap_fraction"] == pytest.approx(0.989, abs=1e-3)

    def test_bin_well_below_the_cap_is_not_cap_bound(self):
        """0.15 against a 0.2 cap is 75%, below ``_CLAMP_CAP_NEAR_FRAC``, so the bin is not
        cap-bound even though every member wanted materially more mass there."""
        screen = max_step_clamp_screen(self._fine_grid_record(0.15))
        assert screen["resolution_bin_cap_bound"] is False
        assert screen["suspected"] is False

    def test_missing_timestamp_treated_as_pre_fix(self):
        screen = max_step_clamp_screen(self._record(submitted=None))
        assert screen["submitted_before_grid_scaled_cap"] is True
        assert screen["max_step_cap"] == pytest.approx(0.2)

    def test_unparseable_timestamp_treated_as_pre_fix(self):
        """Same rule as a missing timestamp: the undated (and undatable) archive records all
        predate the fix, so an unreadable timestamp must not be read as post-fix."""
        screen = max_step_clamp_screen(self._record(submitted="not-a-date"))
        assert screen["submitted_before_grid_scaled_cap"] is True
        assert screen["max_step_cap"] == pytest.approx(0.2)

    def test_timestamp_with_an_offset_is_compared_in_utc(self):
        """A post-fix instant written with a local offset must read as post-fix, because a naive
        offset-dropping comparison shifts it hours across the era boundary."""
        screen = max_step_clamp_screen(self._record(submitted="2026-07-21T11:07:37-07:00"))
        assert screen["submitted_before_grid_scaled_cap"] is False

    def test_members_not_materially_more_does_not_fire(self):
        """The members' own curves put ~0.2 on the bin too, so the cap coincided with what the
        ensemble wanted and nothing was overridden."""
        diffuse = {
            "model-a": [[10.0, 0.0], [50.0, 3.0], [90.0, 8.0]],
            "model-b": [[10.0, 0.5], [50.0, 3.5], [90.0, 8.5]],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=diffuse))
        assert screen["resolution_bin_at_cap"] is True
        assert screen["suspected"] is False

    def test_single_member_curve_does_not_fire(self):
        one = {"model-a": self._MEMBERS["model-a"]}
        assert max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=one))["suspected"] is False

    def test_anonymous_member_keys_are_excluded(self):
        """A positional key on a stacked record can hold the stacker's aggregate."""
        anon = {"Forecaster 1": self._MEMBERS["model-a"], "Forecaster 2": self._MEMBERS["model-b"]}
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=anon))
        assert screen["member_bin_masses"] == {}
        assert screen["suspected"] is False

    def test_resolution_exactly_on_grid_point_screens_the_bin_below(self):
        """A resolution sitting exactly ON a grid edge belongs to the bin BELOW it, which is the
        platform scorer's convention in ``resolution_to_bucket_index``.

        On this fixture resolution 2.0 must screen the [1, 2] bin whose 0.20 step sits at the
        pre-fix cap; the old ``side="right"`` screened [2, 3] and missed the q43913 signature
        entirely.
        """
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, resolution=2.0))
        assert screen["resolution_bin"] == [1.0, 2.0]
        assert screen["published_bin_mass"] == pytest.approx(0.2, abs=1e-9)
        assert screen["suspected"] is True

    def test_single_pair_member_curve_is_unusable(self):
        """One recovered (percentile, value) pair interpolates to a constant PIT at every
        resolution, so the member is dropped, leaving one usable curve, which is below the
        two-curve minimum the screen requires."""
        one_pair = {
            "model-a": [[50.0, 90.0]],
            "model-b": self._MEMBERS["model-b"],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=one_pair))
        assert list(screen["member_bin_masses"]) == ["model-b"]
        assert screen["suspected"] is False

    def test_a_sparse_member_curve_is_excluded_from_the_min(self):
        """The verdict turns on the MINIMUM member bin mass, so one sparse recovery can
        decide it — and a 3-anchor interpolation across one bin is not the distribution the
        model declared. q43913's KNOWN_BUG_QIDS entry survives this gate on its own
        11-anchor member; the 3-anchor sibling never decided that verdict."""
        mixed = {
            "model-a": self._MEMBERS["model-a"],
            "model-sparse": [[10.0, 0.0], [50.0, 3.0], [90.0, 8.0]],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=mixed))

        assert list(screen["member_bin_masses"]) == ["model-a"]
        # One usable curve is below the >=2-curves requirement, so nothing is suspected.
        assert screen["suspected"] is False

    def test_a_uniformly_sparse_record_reports_no_member_masses(self):
        """The sparse-era shape: no curve clears the anchor floor, so the screen has no member
        evidence at all rather than ranking equals against each other. A bin-mass comparison is
        absolute, unlike ``ranking_cohort``'s relative one."""
        sparse = {
            "model-a": [[10.0, 0.9], [50.0, 1.3], [90.0, 2.1]],
            "model-b": [[10.0, 0.95], [50.0, 1.4], [90.0, 2.2]],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=sparse))

        assert screen["member_bin_masses"] == {}
        assert screen["min_member_bin_mass"] is None
        assert screen["suspected"] is False

    def test_non_monotonic_member_curve_is_unusable(self):
        """Percentiles that DECREASE as values increase invert the curve, so it is dropped."""
        inverted = {
            "model-a": [[90.0, 0.9], [50.0, 1.3], [10.0, 2.1]],
            "model-b": self._MEMBERS["model-b"],
        }
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, members=inverted))
        assert list(screen["member_bin_masses"]) == ["model-b"]

    def test_non_numeric_resolution_and_type_gates(self):
        assert max_step_clamp_screen({"type": "binary"})["applicable"] is False
        screen = max_step_clamp_screen(self._record(submitted=self._PRE_FIX_TS, resolution="below_lower_bound"))
        assert screen["suspected"] is False
        assert screen["reason"] == "non-numeric resolution"

    def test_records_without_a_usable_grid_report_that_reason(self):
        """The screen needs the question's own value grid to locate the realized bin.

        Comment-backfilled records often carry no continuous_range, and a grid whose length
        disagrees with the published CDF cannot be indexed either, so both must report a reason
        instead of screening an arbitrary bin.
        """
        no_grid = self._record(submitted=self._PRE_FIX_TS)
        no_grid["scaling"] = {"range_min": 0.0, "range_max": 10.0}
        assert max_step_clamp_screen(no_grid)["reason"] == "no usable grid"

        mismatched = self._record(submitted=self._PRE_FIX_TS, grid=[0.0, 1.0, 2.0])
        assert max_step_clamp_screen(mismatched)["reason"] == "no usable grid"
        assert max_step_clamp_screen(mismatched)["suspected"] is False


class TestNumericPitAnalysisValueGrid:
    """End-to-end numeric_pit_analysis on a small mixed cohort: one linear-scaled
    record and one log-scaled (zero_point) record carrying continuous_range."""

    def _record(self, post_id, cdf, resolution, lower, upper, zero_point, continuous_range):
        return {
            "post_id": post_id,
            "type": "numeric",
            "our_forecast_values": cdf,
            "resolution_parsed": resolution,
            "scaling": {
                "range_min": lower,
                "range_max": upper,
                "zero_point": zero_point,
                "continuous_range": continuous_range,
            },
            "open_lower_bound": False,
            "open_upper_bound": False,
            "brier_score": None,
            "log_score": None,
            "numeric_log_score": 0.0,
            "mc_log_score": None,
            "per_model_forecasts": {},
            "metadata": {"category": None},
        }

    def test_continuous_range_used_directly_for_log_scaled(self):
        """One linear record at its midpoint resolution and one log-scaled record at its
        geometric midpoint both read PIT ~0.5, where the linear-index map called the log-scaled
        one ~0.03."""
        cdf = list(np.linspace(0.0, 1.0, 201))
        # Linear question, midpoint resolution -> PIT 0.5.
        lin_grid = list(build_cdf_value_grid(0.0, 100.0, None, num_points=201))
        linear_rec = self._record(1, cdf, 50.0, 0.0, 100.0, None, lin_grid)

        geo_grid = list(build_cdf_value_grid(1.0, 1000.0, 0.0, num_points=201))
        log_rec = self._record(2, cdf, 31.6, 1.0, 1000.0, 0.0, geo_grid)

        result = numeric_pit_analysis([linear_rec, log_rec])
        assert result["count"] == 2
        assert result["pit_values"][0] == pytest.approx(0.5)
        assert result["pit_values"][1] == pytest.approx(0.5, abs=0.02)
        # Both PITs land in the central coverage band.
        assert result["coverage_50"] == pytest.approx(1.0)

    def test_zero_point_zero_without_continuous_range_reconstructs_geometric(self):
        """Regression for the zero_point sentinel bug on the analysis fallback path.

        A log-scale record serializes ``zero_point == 0`` with a positive ``range_min`` but
        carries NO continuous_range, from an old archive or schema drift, and
        ``numeric_pit_analysis`` must then reconstruct the GEOMETRIC grid via
        ``grid_zero_point`` rather than a linear one. On [1, 1000] the geometric midpoint
        (~31.6) is PIT ~0.5, where the buggy linear-grid reconstruction called it near-zero.
        """
        cdf = list(np.linspace(0.0, 1.0, 201))
        log_rec = self._record(1, cdf, 31.6, 1.0, 1000.0, 0, None)
        result = numeric_pit_analysis([log_rec])
        assert result["count"] == 1
        assert result["pit_values"][0] == pytest.approx(0.5, abs=0.02)


class TestSetValuedOutOfRangePit:
    """An out-of-range resolution's PIT is a SET, and point statistics exclude it.

    The platform reports "beyond the displayed range" as a string, so the resolution VALUE
    is unknown and ``F(resolution)`` is only pinned to ``[cdf[-1], 1]`` (above) or
    ``[0, cdf[0]]`` (below). Forcing it to 1.0 / 0.0 counted q44842 as a high-side band
    miss: an open-bound record that deliberately published 13% of its mass above the
    displayed ceiling, resolved ``above_upper_bound``, and won spot peer +24.4.
    """

    @staticmethod
    def _record(resolution, *, cdf_start: float = 0.0, cdf_end: float = 1.0) -> dict:
        cdf = list(np.linspace(cdf_start, cdf_end, 201))
        return {
            "post_id": 1,
            "type": "numeric",
            "our_forecast_values": cdf,
            "resolution_parsed": resolution,
            "scaling": {"range_min": 0.0, "range_max": 100.0, "zero_point": None},
            "open_lower_bound": True,
            "open_upper_bound": True,
            "numeric_log_score": 0.0,
            "metadata": {"category": None},
        }

    def test_the_interval_is_read_off_our_own_published_tail_mass(self):
        above = out_of_range_pit_reading("above_upper_bound", list(np.linspace(0.0, 0.87, 201)))
        assert above is not None
        assert (above.low, above.high) == pytest.approx((0.87, 1.0))
        assert above.oob_side == "high"
        assert above.is_interval
        assert above.point is None

        below = out_of_range_pit_reading("below_lower_bound", list(np.linspace(0.13, 1.0, 201)))
        assert below is not None
        assert (below.low, below.high) == pytest.approx((0.0, 0.13))
        assert below.oob_side == "low"

        # Not an out-of-range marker at all.
        assert out_of_range_pit_reading("annulled", [0.0, 1.0]) is None
        assert out_of_range_pit_reading(50.0, [0.0, 1.0]) is None

    def test_a_closed_bound_interval_collapses_to_the_old_point_convention(self):
        """With no mass beyond the bound, ``[cdf[-1], 1]`` is ``[1, 1]``, so the set-valued
        reading degenerates to exactly the 1.0 the old convention forced and nothing changes on
        records that put nothing out of range."""
        reading = out_of_range_pit_reading("above_upper_bound", list(np.linspace(0.0, 1.0, 201)))
        assert reading is not None
        assert not reading.is_interval
        assert reading.point == pytest.approx(1.0)

    def test_a_point_reading_answers_the_band_predicates_like_a_scalar(self):
        point = PitReading.from_point(0.42)
        assert point.point == pytest.approx(0.42)
        assert point.intersects(0.10, 0.90)
        assert point.at_or_below(0.50)
        assert not point.at_or_below(0.40)
        assert not point.entirely_below(0.10)
        assert not point.entirely_above(0.90)

    def test_q44842_shape_is_covered_and_excluded_from_the_histogram(self):
        result = numeric_pit_analysis([self._record("above_upper_bound", cdf_end=0.87)])
        assert result["count"] == 1
        assert result["n_point"] == 0
        assert result["n_oob_interval"] == 1
        assert result["pit_values"] == []
        assert result["pit_intervals"] == [(pytest.approx(0.87), 1.0)]
        # [0.87, 1] intersects [0.05, 0.95] and [0.25, 0.75] it does not.
        assert result["coverage_90"] == pytest.approx(1.0)
        assert result["coverage_50"] == pytest.approx(0.0)
        assert sum(result["histogram"]) == 0

    def test_a_starved_tail_is_still_outside_the_coverage_band(self):
        """``cdf[-1] = 0.999`` is the open-bound structural floor, so [0.999, 1] lies wholly
        above 0.95 and this record is the band miss that the q44842 shape is not."""
        result = numeric_pit_analysis([self._record("above_upper_bound", cdf_end=0.999)])
        assert result["coverage_90"] == pytest.approx(0.0)

    def test_the_below_bound_mirror(self):
        covered = numeric_pit_analysis([self._record("below_lower_bound", cdf_start=0.13)])
        assert covered["coverage_90"] == pytest.approx(1.0)
        assert covered["n_oob_interval"] == 1
        starved = numeric_pit_analysis([self._record("below_lower_bound", cdf_start=0.001)])
        assert starved["coverage_90"] == pytest.approx(0.0)

    def test_point_records_and_intervals_share_the_coverage_denominator(self):
        data = [
            self._record(50.0),  # PIT 0.50 — covered
            self._record(1.0),  # PIT 0.01 — outside [0.05, 0.95]
            self._record("above_upper_bound", cdf_end=0.87),  # interval — covered
        ]
        result = numeric_pit_analysis(data)
        assert result["count"] == 3
        assert result["n_point"] == 2
        assert result["n_oob_interval"] == 1
        assert result["coverage_90"] == pytest.approx(2 / 3)
        # The histogram (a point statistic) counts only the two point readings.
        assert sum(result["histogram"]) == 2

    def test_the_report_discloses_the_excluded_count(self):
        report = performance_analysis.generate_report(
            [self._record(50.0), self._record("above_upper_bound", cdf_end=0.87)]
        )
        assert "## Numeric Questions" in report
        assert "Out-of-range resolutions (set-valued PIT" in report
        assert "excluded from the histogram): 1" in report


class TestRescoreRecords:
    """Stale stored scores in cached datasets must self-heal on load.

    Scores are pure functions of fields the record carries, but a cached JSON's
    score VALUES are whatever the scorer computed when the file was written — a
    scorer fix never reaches previously-saved files. The checked-in q38991
    fixture is the real record that carried a linear-bucket numeric_log_score of
    -193.29 for a month after the zero_point coercion fix, against a platform
    spot_baseline_score of 165.54.
    """

    _FIXTURE = Path(__file__).parent / "data" / "q38991_stale_zero_point_score.json"

    def _stale_record(self) -> dict:
        return json.loads(self._FIXTURE.read_text())

    def test_zero_point_record_rescores_to_platform_value(self):
        record = self._stale_record()
        assert record["scaling"]["zero_point"] == 0
        assert record["numeric_log_score"] == pytest.approx(-193.292, abs=1e-3)

        changed = rescore_records([record])

        assert changed == 1
        assert record["numeric_log_score"] == pytest.approx(record["metaculus_scores"]["spot_baseline_score"], abs=1e-9)

    def test_fresh_scores_left_untouched(self):
        record = self._stale_record()
        rescore_records([record])
        healed = record["numeric_log_score"]
        assert rescore_records([record]) == 0
        assert record["numeric_log_score"] == healed

    def test_unrecomputable_score_is_never_deleted(self):
        """Missing scaling bounds make recomputation yield None, so the stored value is kept."""
        record = self._stale_record()
        record["scaling"] = {}
        stored = record["numeric_log_score"]
        assert rescore_records([record]) == 0
        assert record["numeric_log_score"] == stored

    def test_load_dataset_heals_stale_scores(self, tmp_path: Path):
        path = tmp_path / "cached.json"
        path.write_text(json.dumps([self._stale_record()]))
        (loaded,) = load_dataset(str(path))
        assert loaded["numeric_log_score"] == pytest.approx(loaded["metaculus_scores"]["spot_baseline_score"], abs=1e-9)

    def test_malformed_record_is_skipped(self):
        assert rescore_records([{"no_type": True}, "not-a-dict"]) == 0  # type: ignore[list-item]

    def test_partial_records_are_skipped_not_crashed(self):
        """Rescoring takes arbitrary cached JSON, so every record missing a field that
        ``_compute_scores`` subscripts must be skipped rather than raise KeyError."""
        partial = [
            {"type": "binary", "resolution_parsed": True},  # no our_forecast_values
            {"type": "binary", "resolution_parsed": True, "our_forecast_values": [0.3, 0.7]},  # no our_prob_yes
            {  # numeric without open-bound flags
                "type": "numeric",
                "resolution_parsed": 5.0,
                "our_forecast_values": [0.0, 0.5, 1.0],
                "scaling": {"range_min": 0.0, "range_max": 10.0},
            },
        ]
        assert rescore_records(partial) == 0
        assert "brier_score" not in partial[0]

    def test_load_dataset_survives_partial_records(self, tmp_path: Path):
        path = tmp_path / "cached.json"
        path.write_text(json.dumps([{"type": "binary", "resolution_parsed": True}, self._stale_record()]))
        loaded = load_dataset(str(path))
        assert len(loaded) == 2
        assert loaded[1]["numeric_log_score"] == pytest.approx(
            loaded[1]["metaculus_scores"]["spot_baseline_score"], abs=1e-9
        )

    def test_load_dataset_is_idempotent_on_an_already_healed_file(self, tmp_path: Path):
        """Re-loading a file whose scores already agree with the scorer must leave every value
        byte-identical, because healing is a repair rather than a rewrite of live data."""
        healed = self._stale_record()
        rescore_records([healed])
        path = tmp_path / "healed.json"
        path.write_text(json.dumps([healed]))
        (reloaded,) = load_dataset(str(path))
        assert reloaded["numeric_log_score"] == healed["numeric_log_score"]

    def test_scoring_failure_on_a_record_without_post_id_only_warns(self, caplog):
        """Rescoring walks arbitrary cached JSON, so a record can lack post_id entirely, and the
        scoring-failure log lines must read it defensively: a subscript there turns one
        unscoreable record into a KeyError that kills the whole load."""
        unscoreable_numeric = {
            "type": "numeric",
            "resolution_parsed": 5.0,
            "our_forecast_values": [0.5],  # < 2 CDF points -> numeric_log_score raises
            "open_lower_bound": False,
            "open_upper_bound": False,
            "scaling": {"range_min": 0.0, "range_max": 10.0, "zero_point": None},
            "numeric_log_score": -1.0,
        }
        unscoreable_mc = {
            "type": "multiple_choice",
            "resolution_parsed": "B",
            "our_forecast_values": [1.0],  # fewer probabilities than options
            "options": ["A", "B"],
            "mc_log_score": -1.0,
        }
        with caplog.at_level(logging.WARNING):
            assert rescore_records([unscoreable_numeric, unscoreable_mc]) == 0

        messages = [r.getMessage() for r in caplog.records]
        assert any("Failed numeric scoring for post None" in m for m in messages)
        assert any("Failed MC scoring for post None" in m for m in messages)
        # The stored values survive: healing never deletes a score it can't recompute.
        assert unscoreable_numeric["numeric_log_score"] == -1.0
        assert unscoreable_mc["mc_log_score"] == -1.0


class TestBuildPerformanceDatasetResearchTags:
    """The dataset the analysis reads must arrive with the treatment tags stamped.

    ``attach_research_tags`` is a single call inside ``build_performance_dataset``;
    unit-testing the tagger alone leaves that pass-through deletable with a green
    suite, and every treated/untreated calibration cut reads these fields off the
    built dataset.
    """

    def test_records_carry_tags_from_the_archive_dir(self, tmp_path: Path, monkeypatch):
        (tmp_path / "11.json").write_text(
            json.dumps(
                {
                    "research_text": "## Time Series Anchor\nband\n## Agentic Research Findings\nfindings\n",
                    "source": "artifact",
                    "gap_fill_v2": {"steps": 4},
                }
            )
        )
        monkeypatch.setattr(
            collector, "fetch_resolved_questions", lambda tournament, token: [_binary_post(1, 11, score_data={})]
        )
        monkeypatch.setattr(
            collector,
            "fetch_bot_comments",
            lambda author_id, token: [{"id": 9, "text": "*Forecaster 1*: 70%\n", "on_post": 1}],
        )

        records = build_performance_dataset(tournament="t", token="fake", research_archive_dir=tmp_path)

        assert len(records) == 1
        assert records[0]["anchor_present"] is True
        assert records[0]["gfv2_present"] is True
        assert records[0]["gfv2_loop_ran"] is True
        assert records[0]["research_source_class"] == "artifact"

    def test_question_without_an_archive_record_gets_none_not_false(self, tmp_path: Path, monkeypatch):
        """Absence of evidence is not an untreated record: a missing archive file, or a whole
        missing archive, must never look like a measured False in the cuts."""
        monkeypatch.setattr(
            collector, "fetch_resolved_questions", lambda tournament, token: [_binary_post(2, 22, score_data={})]
        )
        monkeypatch.setattr(collector, "fetch_bot_comments", lambda author_id, token: [])

        records = build_performance_dataset(tournament="t", token="fake", research_archive_dir=tmp_path)

        assert records[0]["anchor_present"] is None
        assert records[0]["gfv2_present"] is None
        assert records[0]["anchor_confidence"] is None


class TestBuildPerformanceDatasetPriorDiff:
    """``prior_records=`` must actually reach the re-resolution diff.

    Same rule as the sibling class above: the call inside ``build_performance_dataset`` is a
    single line, so unit-testing ``diff_platform_rescores`` alone leaves the pass-through
    deletable with a green suite. It is the live-pull half of the q44798 detector, where
    Metaculus edits a resolution in place and no timestamp moves.
    """

    def _fetchers(self, monkeypatch, post: dict) -> None:
        monkeypatch.setattr(collector, "fetch_resolved_questions", lambda tournament, token: [post])
        monkeypatch.setattr(collector, "fetch_bot_comments", lambda author_id, token: [])

    def test_a_moved_resolution_is_tagged_on_the_built_records(self, tmp_path: Path, monkeypatch):
        self._fetchers(monkeypatch, _binary_post(3, 33, score_data={}))
        prior = [{"post_id": 3, "question_id": 33, "resolution_raw": "no", "resolution_parsed": False}]

        records = build_performance_dataset(
            tournament="t", token="fake", research_archive_dir=tmp_path, prior_records=prior
        )

        assert records[0]["platform_rescored"] is True
        assert "resolution_raw" in records[0]["platform_rescored_fields"]
        assert records[0]["prior_resolution"] == "no"

    def test_no_prior_leaves_the_tags_none_not_false(self, tmp_path: Path, monkeypatch):
        """ "Not compared" and "compared, nothing moved" are different facts, and a default build
        must produce the first, or every downstream cut reads silence as stability.

        The keys are absent rather than explicitly None on this path, which is the same "not
        compared" answer to every reader, since the diff and the renderer both use ``.get``.
        """
        self._fetchers(monkeypatch, _binary_post(4, 44, score_data={}))

        records = build_performance_dataset(tournament="t", token="fake", research_archive_dir=tmp_path)

        assert records[0].get("platform_rescored") is None
        assert records[0].get("platform_rescored_fields") is None


class TestPackageExports:
    """The residual rounds' out-of-band scripts import these off the package root, so
    the re-export list is a contract, not bookkeeping."""

    def test_new_analysis_helpers_are_re_exported(self):
        for name in (
            "attach_research_tags",
            "research_tags_for_qid",
            "research_tags_for_record",
            "max_step_clamp_screen",
            "rescore_records",
            "parse_stacker_skip_reason_marker",
        ):
            assert name in performance_analysis.__all__, name
            assert getattr(performance_analysis, name) is not None


class TestResolveNumericScoreInputsZeroPoint:
    """Regression for the zero_point sentinel bug in the record-scoring coercion:
    ``resolve_numeric_record_to_score_inputs`` must keep a serialized
    ``zero_point == 0`` (with a positive ``range_min``) as a genuine log-scale
    value, not collapse it to the linear ``None`` sentinel."""

    def _record(self, zero_point: float | int | None, range_min: float, range_max: float) -> dict:
        return {
            "type": "numeric",
            "resolution_parsed": (range_min + range_max) / 2.0,
            "scaling": {"range_min": range_min, "range_max": range_max, "zero_point": zero_point},
        }

    def test_zero_point_zero_stays_log_when_range_min_positive(self):
        """The sibling of the width_monitor fix: a log-scale question with a positive floor
        carries ``zero_point == 0``, which must survive as 0.0 so ``numeric_log_score`` buckets
        on the geometric grid."""
        inputs = resolve_numeric_record_to_score_inputs(self._record(0, 1.0, 1000.0))
        assert inputs is not None
        _res, _lo, _hi, zero_point = inputs
        assert zero_point == 0.0

    def test_zero_point_zero_dropped_when_range_min_nonpositive(self):
        """A non-positive floor rules out a log transform, so the axis is linear (None)."""
        inputs = resolve_numeric_record_to_score_inputs(self._record(0, 0.0, 100.0))
        assert inputs is not None
        assert inputs[3] is None

    def test_absent_zero_point_is_linear(self):
        inputs = resolve_numeric_record_to_score_inputs(self._record(None, 0.0, 100.0))
        assert inputs is not None
        assert inputs[3] is None

    def test_nonzero_zero_point_passthrough(self):
        inputs = resolve_numeric_record_to_score_inputs(self._record(50, 0.0, 100.0))
        assert inputs is not None
        assert inputs[3] == 50.0


class TestGenerateReport:
    """Pins the markdown skeleton of ``generate_report`` — the CLI's whole output.

    Each section is gated on its own count, so an empty cut must leave no heading
    behind, and the section ORDER (binary, per-model, numeric, MC) is what a reader
    diffs across residual rounds.
    """

    @staticmethod
    def _mc_record(post_id: int, log_score: float) -> dict:
        return {
            "post_id": post_id,
            "type": "multiple_choice",
            "mc_log_score": log_score,
            "resolution_parsed": "B",
            "options": ["A", "B"],
            "our_forecast_values": [0.3, 0.7],
            "brier_score": None,
            "log_score": None,
            "numeric_log_score": None,
            "per_model_forecasts": {},
            "metadata": {"category": None},
        }

    def test_empty_dataset_renders_only_the_header(self):
        report = performance_analysis.generate_report([])

        assert report.splitlines()[0] == "# Performance Analysis Report"
        assert "**Total questions:** 0" in report
        assert "## Binary Questions" not in report
        assert "## Per-Model Binary Scores" not in report
        assert "## Numeric Questions" not in report
        assert "## Multiple Choice Questions" not in report

    def test_type_counts_are_listed_alphabetically(self):
        data = [
            _binary_record(1, 0.7, True),
            self._mc_record(2, -0.5),
            _numeric_record(3, list(np.linspace(0.0, 1.0, 201)), 50.0),
        ]
        report = performance_analysis.generate_report(data)

        assert "**Total questions:** 3" in report
        assert "- binary: 1" in report
        assert "- multiple_choice: 1" in report
        assert "- numeric: 1" in report
        counts_block = report.split("**Total questions:** 3\n")[1].splitlines()[:3]
        assert counts_block == ["- binary: 1", "- multiple_choice: 1", "- numeric: 1"]

    def test_sections_appear_in_a_fixed_order(self):
        data = [
            _binary_record(1, 0.7, True, per_model={"model-a": "70.0%"}),
            _binary_record(2, 0.2, False, per_model={"model-a": "20.0%"}),
            self._mc_record(3, -0.5),
            _numeric_record(4, list(np.linspace(0.0, 1.0, 201)), 50.0),
        ]
        report = performance_analysis.generate_report(data)

        headings = [line for line in report.splitlines() if line.startswith("## ")]
        assert headings == [
            "## Binary Questions",
            "## Per-Model Binary Scores",
            "## Numeric Questions",
            "## Multiple Choice Questions",
        ]

    def test_binary_section_carries_summary_lines_and_a_calibration_table(self):
        data = [_binary_record(1, 0.7, True), _binary_record(2, 0.2, False)]
        report = performance_analysis.generate_report(data)

        assert "- Count: 2" in report
        assert "- Mean Brier: 0.0650" in report
        assert "- Direction Accuracy: 100.0%" in report
        assert "- Base Rate: 50.0%" in report
        assert "| Bucket | Predicted | Actual | Count |" in report

    def test_per_model_section_lists_each_recovered_member(self):
        data = [
            _binary_record(1, 0.7, True, per_model={"model-a": "70.0%", "model-b": "60.0%"}),
            _binary_record(2, 0.2, False, per_model={"model-a": "20.0%", "model-b": "30.0%"}),
        ]
        report = performance_analysis.generate_report(data)

        assert "### Calibration" in report
        assert "| model-a |" in report
        assert "| model-b |" in report

    def test_numeric_section_renders_ten_pit_histogram_bins(self):
        data = [_numeric_record(1, list(np.linspace(0.0, 1.0, 201)), 50.0)]
        report = performance_analysis.generate_report(data)

        assert "### PIT Histogram" in report
        assert "| 0.0-0.1 | 0 |" in report
        assert "| 0.5-0.6 | 1 |" in report
        assert sum(1 for line in report.splitlines() if line.startswith("| 0.")) == 10

    def test_mc_section_reports_accuracy_and_mean_scores(self):
        report = performance_analysis.generate_report([self._mc_record(1, -0.5)])

        assert "## Multiple Choice Questions" in report
        assert "- Accuracy (top pick correct): 100.0%" in report
        assert "- Mean Prob on Correct: 0.70" in report
        assert "- Mean MC Log Score: -0.50" in report
