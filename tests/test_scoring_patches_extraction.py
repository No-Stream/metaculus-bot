"""Branch pins for the scoring_patches community-data extractors.

These lock the full reason-code surface of ``_extract_mc_community_probs`` and
``_extract_numeric_community_cdf`` plus the diagnostic counters they bump, so the
extractors can be decomposed into helpers without silently moving a boundary. The
reason strings are the contract: ``calculate_multiple_choice_baseline_score``
passes them into ``log_mc_vector_mismatch`` as ``community_source``, and
``get_scoring_path_stats`` is what the run log reports.

Companion to tests/test_scoring_patches.py, which covers the happy paths and the
monkey-patch installation.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import Mock

import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from metaculus_bot.scoring_patches import (
    _extract_mc_community_probs,
    _extract_numeric_community_cdf,
    get_scoring_path_stats,
    log_score_scale_validation,
    reset_scoring_path_stats,
)


class _ExplodingDict(dict):
    """A dict whose ``get`` raises — the only way to reach the extractors' outer catch.

    RuntimeError deliberately: it is outside every plausible narrow exception tuple,
    so this pins that the outer handler has to stay broad.
    """

    def get(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("api_json read blew up")


def _rw_latest(payload: dict[str, Any]) -> dict[str, Any]:
    return {"question": {"type": "multiple_choice", "aggregations": {"recency_weighted": {"latest": payload}}}}


def _question(api_json: Any, options: list[str] | None = None) -> Mock:
    question = Mock()
    question.id_of_question = 42
    question.id_of_post = 4242
    question.api_json = api_json
    question.options = options
    return question


@pytest.fixture(autouse=True)
def _fresh_counters() -> None:
    """The extractors bump module globals; every test starts from zero."""
    reset_scoring_path_stats()


class TestMcCommunityProbsReasonCodes:
    """Every non-success exit of ``_extract_mc_community_probs``, by reason string."""

    @pytest.mark.parametrize(
        ("api_json", "options", "expected_reason"),
        [
            pytest.param(None, None, "missing_api_json", id="api_json_absent"),
            pytest.param("not-a-dict", None, "missing_api_json", id="api_json_wrong_type"),
            pytest.param({"question": {"type": "mc"}}, ["A"], "missing_aggregations", id="no_aggregations"),
            pytest.param({"question": {"aggregations": {}}}, ["A"], "missing_rw_latest", id="no_recency_weighted"),
            pytest.param(
                {"question": {"aggregations": {"recency_weighted": {}}}},
                ["A"],
                "missing_rw_latest",
                id="no_latest_node",
            ),
            pytest.param(
                _rw_latest({"forecast_values": [0.5, 0.5]}), None, "forecast_values_no_options", id="fv_no_options"
            ),
            pytest.param(
                _rw_latest({"forecast_values": [0.3, 0.3, 0.4]}),
                ["A", "B"],
                "forecast_values_length_mismatch",
                id="fv_length_mismatch",
            ),
            pytest.param(
                _rw_latest({"forecast_values": ["abc", "def"]}),
                ["A", "B"],
                "forecast_values_cast_error",
                id="fv_cast_valueerror",
            ),
            pytest.param(
                _rw_latest({"forecast_values": [{}, {}]}),
                ["A", "B"],
                "forecast_values_cast_error",
                id="fv_cast_typeerror",
            ),
            pytest.param(
                _rw_latest({"forecast_values": [1.5, -0.5]}),
                ["A", "B"],
                "forecast_values_out_of_range",
                id="fv_out_of_range",
            ),
            pytest.param(
                _rw_latest({"forecast_values": [0.2, 0.2]}), ["A", "B"], "forecast_values_bad_sum", id="fv_bad_sum"
            ),
            pytest.param(
                _rw_latest({"probability_yes_per_category": {"A": 0.1, "B": 0.2}}),
                ["A", "B"],
                "pyc_bad_sum",
                id="pyc_bad_sum",
            ),
            pytest.param(
                _rw_latest({"probability_yes_per_category": {"A": 0.5, "B": 0.5}}),
                None,
                "pyc_no_options",
                id="pyc_no_options",
            ),
            pytest.param(_rw_latest({"some_other_key": 1}), ["A", "B"], "no_forecast_data", id="neither_source"),
            pytest.param(_ExplodingDict(question={}), ["A", "B"], "exception", id="outer_exception"),
        ],
    )
    def test_returns_none_with_reason(self, api_json: Any, options: list[str] | None, expected_reason: str) -> None:
        probs, reason = _extract_mc_community_probs(_question(api_json, options))

        assert probs is None
        assert reason == expected_reason

    def test_every_failure_bumps_missing_community(self) -> None:
        """``mc_missing_community`` is the run-log rollup: no failure path may skip it."""
        for api_json, options in [
            (None, None),
            ({"question": {"type": "mc"}}, ["A"]),
            (_rw_latest({"forecast_values": [0.2, 0.2]}), ["A", "B"]),
            (_rw_latest({"some_other_key": 1}), ["A", "B"]),
            (_ExplodingDict(question={}), ["A", "B"]),
        ]:
            _extract_mc_community_probs(_question(api_json, options))

        assert get_scoring_path_stats()["mc_missing_community"] == 5

    @pytest.mark.parametrize(
        ("api_json", "counter"),
        [
            pytest.param(None, "mc_missing_api_json", id="api_json"),
            pytest.param({"question": {"type": "mc"}}, "mc_missing_aggregations", id="aggregations"),
            pytest.param(_rw_latest({"some_other_key": 1}), "mc_missing_prob_yes_per_category", id="no_source"),
        ],
    )
    def test_specific_breakdown_counter_is_bumped(self, api_json: Any, counter: str) -> None:
        _extract_mc_community_probs(_question(api_json, ["A", "B"]))

        assert get_scoring_path_stats()[counter] == 1


class TestMcCommunityProbsSuccess:
    def test_forecast_values_pass_through_when_already_normalized(self) -> None:
        probs, reason = _extract_mc_community_probs(_question(_rw_latest({"forecast_values": [0.3, 0.7]}), ["A", "B"]))

        assert reason == "forecast_values"
        assert probs == pytest.approx([0.3, 0.7])

    def test_forecast_values_within_rounding_tolerance_are_renormalized(self) -> None:
        """|sum - 1| in (1e-6, 1e-3] is renormalized rather than rejected."""
        probs, reason = _extract_mc_community_probs(
            _question(_rw_latest({"forecast_values": [0.3, 0.7001]}), ["A", "B"])
        )

        assert reason == "forecast_values"
        assert probs is not None
        assert sum(probs) == pytest.approx(1.0)
        assert probs[0] == pytest.approx(0.3 / 1.0001)

    def test_forecast_values_are_preferred_over_pyc(self) -> None:
        probs, reason = _extract_mc_community_probs(
            _question(
                _rw_latest({"forecast_values": [0.3, 0.7], "probability_yes_per_category": {"A": 0.9, "B": 0.1}}),
                ["A", "B"],
            )
        )

        assert reason == "forecast_values"
        assert probs == pytest.approx([0.3, 0.7])

    def test_pyc_is_aligned_to_question_option_order(self) -> None:
        probs, reason = _extract_mc_community_probs(
            _question(_rw_latest({"probability_yes_per_category": {"B": 0.75, "A": 0.25}}), ["A", "B"])
        )

        assert reason == "probability_yes_per_category"
        assert probs == pytest.approx([0.25, 0.75])

    def test_pyc_missing_option_becomes_zero_when_the_sum_still_holds(self) -> None:
        """The documented deliberate ``.get(opt, 0.0)``: warned about, but not rejected,
        because the sum gate is what rejects a materially incomplete ballot."""
        question = _question(_rw_latest({"probability_yes_per_category": {"A": 0.25, "B": 0.75}}), ["A", "B", "C"])

        probs, reason = _extract_mc_community_probs(question)

        assert reason == "probability_yes_per_category"
        assert probs == pytest.approx([0.25, 0.75, 0.0])

    def test_options_fall_back_to_the_api_json_copy(self) -> None:
        """``question.options`` absent → the question node's own ``options`` list is used."""
        api_json = _rw_latest({"forecast_values": [0.4, 0.6]})
        api_json["question"]["options"] = ["A", "B"]

        probs, reason = _extract_mc_community_probs(_question(api_json, None))

        assert reason == "forecast_values"
        assert probs == pytest.approx([0.4, 0.6])

    def test_success_does_not_bump_any_missing_counter(self) -> None:
        _extract_mc_community_probs(_question(_rw_latest({"forecast_values": [0.3, 0.7]}), ["A", "B"]))

        stats = get_scoring_path_stats()
        assert stats["mc_missing_community"] == 0
        assert stats["mc_missing_prob_yes_per_category"] == 0

    def test_flat_api_json_without_a_question_node_is_read_directly(self) -> None:
        """``api_json`` may be the question dict itself rather than {"question": {...}}."""
        api_json = {"aggregations": {"recency_weighted": {"latest": {"forecast_values": [0.5, 0.5]}}}}

        probs, reason = _extract_mc_community_probs(_question(api_json, ["A", "B"]))

        assert reason == "forecast_values"
        assert probs == pytest.approx([0.5, 0.5])


class TestNumericCommunityCdf:
    @pytest.mark.parametrize(
        "api_json",
        [
            pytest.param(None, id="api_json_absent"),
            pytest.param("not-a-dict", id="api_json_wrong_type"),
            pytest.param({"question": {}}, id="no_aggregations"),
            pytest.param({"question": {"aggregations": {}}}, id="no_recency_weighted"),
            pytest.param({"question": {"aggregations": {"recency_weighted": {}}}}, id="no_latest_node"),
            pytest.param(_rw_latest({"forecast_values": [0.5]}), id="single_point_cdf"),
            pytest.param(_rw_latest({"forecast_values": "not-a-list"}), id="forecast_values_wrong_type"),
            pytest.param(_rw_latest({}), id="no_forecast_values"),
            pytest.param(_ExplodingDict(question={}), id="outer_exception"),
        ],
    )
    def test_returns_none(self, api_json: Any) -> None:
        assert _extract_numeric_community_cdf(_question(api_json)) is None

    def test_returns_floats_for_a_usable_cdf(self) -> None:
        cdf = _extract_numeric_community_cdf(_question(_rw_latest({"forecast_values": [0, 0.5, 1]})))

        assert cdf == pytest.approx([0.0, 0.5, 1.0])
        assert all(isinstance(x, float) for x in cdf or [])

    def test_length_disagreeing_with_scaling_still_returns_the_cdf(self) -> None:
        """The expected-length check warns; it must not withhold the community data."""
        api_json = _rw_latest({"forecast_values": [0.0, 0.5, 1.0]})
        api_json["question"]["scaling"] = {"inbound_outcome_count": 200}

        assert _extract_numeric_community_cdf(_question(api_json)) == pytest.approx([0.0, 0.5, 1.0])

    def test_unparseable_scaling_does_not_break_extraction(self) -> None:
        api_json = _rw_latest({"forecast_values": [0.0, 0.5, 1.0]})
        api_json["question"]["scaling"] = {"inbound_outcome_count": "many"}

        assert _extract_numeric_community_cdf(_question(api_json)) == pytest.approx([0.0, 0.5, 1.0])


def _report_with_score(question_cls: type, score: float | None) -> Mock:
    report = Mock()
    # Mock(spec=...) satisfies isinstance, which is how the validator buckets by type.
    report.question = Mock(spec=question_cls)
    report.expected_baseline_score = score
    return report


def _benchmark(reports: list[Mock]) -> Mock:
    benchmark = Mock()
    benchmark.forecast_reports = reports
    return benchmark


class TestLogScoreScaleValidation:
    def test_buckets_scores_by_question_type(self, caplog: pytest.LogCaptureFixture) -> None:
        reports = [
            _report_with_score(BinaryQuestion, 10.0),
            _report_with_score(BinaryQuestion, -30.0),
            _report_with_score(NumericQuestion, -50.0),
            _report_with_score(MultipleChoiceQuestion, 5.0),
        ]

        with caplog.at_level(logging.INFO, logger="metaculus_bot.scoring_patches"):
            log_score_scale_validation([_benchmark(reports)])

        assert "Binary scores: count=2, range=[-30.0, 10.0]" in caplog.text
        assert "Numeric scores: count=1, range=[-50.0, -50.0]" in caplog.text
        assert "MC scores: count=1, range=[5.0, 5.0]" in caplog.text
        assert "=== END SCORE VALIDATION ===" in caplog.text

    def test_reports_no_data_per_missing_type(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="metaculus_bot.scoring_patches"):
            log_score_scale_validation([_benchmark([])])

        assert "Binary scores: no data" in caplog.text
        assert "Numeric scores: no data" in caplog.text
        assert "MC scores: no data" in caplog.text

    def test_none_scores_are_skipped_not_bucketed(self, caplog: pytest.LogCaptureFixture) -> None:
        reports = [_report_with_score(BinaryQuestion, None), _report_with_score(BinaryQuestion, 7.0)]

        with caplog.at_level(logging.INFO, logger="metaculus_bot.scoring_patches"):
            log_score_scale_validation([_benchmark(reports)])

        assert "Binary scores: count=1" in caplog.text

    def test_aggregates_across_multiple_benchmarks(self, caplog: pytest.LogCaptureFixture) -> None:
        first = _benchmark([_report_with_score(BinaryQuestion, 1.0)])
        second = _benchmark([_report_with_score(BinaryQuestion, 3.0)])

        with caplog.at_level(logging.INFO, logger="metaculus_bot.scoring_patches"):
            log_score_scale_validation([first, second])

        assert "Binary scores: count=2, range=[1.0, 3.0]" in caplog.text

    def test_a_raising_report_is_logged_not_propagated(self, caplog: pytest.LogCaptureFixture) -> None:
        """Validation is diagnostics-only, so a report that cannot score must not
        take down the caller mid-benchmark."""
        exploding = Mock()
        type(exploding).forecast_reports = property(lambda _self: (_ for _ in ()).throw(RuntimeError("boom")))

        with caplog.at_level(logging.ERROR, logger="metaculus_bot.scoring_patches"):
            log_score_scale_validation([exploding])

        assert "Error in score scale validation" in caplog.text
