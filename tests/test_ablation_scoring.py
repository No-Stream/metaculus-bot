"""Tests for paired-difference statistics and report generation in ablation/scoring.py."""

import math
import random
from unittest.mock import MagicMock

import numpy as np
import pytest
from forecasting_tools.data_models.questions import OutOfBoundsResolution

from metaculus_bot.ablation.scoring import (
    BINARY_LOG_SCORE_FLOOR,
    BRIER_FLOOR,
    NUMERIC_LOG_SCORE_FLOOR,
    PairedScore,
    PairedStats,
    aggregate_paired,
    bootstrap_mean_ci,
    bootstrap_median_ci,
    is_score_saturated,
    render_summary_markdown,
    score_arm_for_qid,
    sign_test,
    wilcoxon_signed_rank,
)
from metaculus_bot.backtest.scoring import GroundTruth
from metaculus_bot.scoring_common import PROB_CLAMP_MIN

# ---------------------------------------------------------------------------
# Synthetic-report fixtures
# ---------------------------------------------------------------------------


def _make_binary_report(prob: float) -> MagicMock:
    from forecasting_tools.data_models.binary_report import BinaryReport

    report = MagicMock(spec=BinaryReport)
    report.prediction = prob
    return report


def _make_numeric_report(*, lower: float, upper: float, cdf_xs: list[float], cdf_ys: list[float]) -> MagicMock:
    from forecasting_tools.data_models.numeric_report import NumericReport

    report = MagicMock(spec=NumericReport)

    cdf_points = []
    for x, y in zip(cdf_xs, cdf_ys, strict=True):
        point = MagicMock()
        point.value = float(x)
        point.percentile = float(y)
        cdf_points.append(point)

    prediction = MagicMock()
    prediction.cdf = cdf_points
    report.prediction = prediction

    question = MagicMock()
    question.lower_bound = float(lower)
    question.upper_bound = float(upper)
    question.open_lower_bound = False
    question.open_upper_bound = False
    question.zero_point = None
    report.question = question

    return report


def _uniform_numeric_report(lower: float = 0.0, upper: float = 10.0, n: int = 201) -> MagicMock:
    xs = list(np.linspace(lower, upper, n))
    ys = [i / (n - 1) for i in range(n)]
    return _make_numeric_report(lower=lower, upper=upper, cdf_xs=xs, cdf_ys=ys)


def _logistic_numeric_report(
    *, center: float, sharpness: float = 1.0, lower: float = 0.0, upper: float = 10.0, n: int = 201
) -> MagicMock:
    xs = np.linspace(lower, upper, n)
    ys = 1.0 / (1.0 + np.exp(-(xs - center) / sharpness))
    ys[0] = 0.0
    ys[-1] = 1.0
    return _make_numeric_report(lower=lower, upper=upper, cdf_xs=xs.tolist(), cdf_ys=ys.tolist())


def _make_mc_report(option_probs: dict[str, float]) -> MagicMock:
    from forecasting_tools.data_models.multiple_choice_report import MultipleChoiceReport

    report = MagicMock(spec=MultipleChoiceReport)
    options = list(option_probs.keys())

    question = MagicMock()
    question.options = options
    report.question = question

    predicted_options = []
    for option, prob in option_probs.items():
        po = MagicMock()
        po.option_name = option
        po.probability = prob
        predicted_options.append(po)
    prediction = MagicMock()
    prediction.predicted_options = predicted_options
    report.prediction = prediction

    return report


# ---------------------------------------------------------------------------
# score_arm_for_qid
# ---------------------------------------------------------------------------


class TestScoreArmForQid:
    def test_score_arm_binary_returns_brier_and_log(self):
        report_a = _make_binary_report(0.3)
        report_b = _make_binary_report(0.8)
        gt = GroundTruth(
            question_id=42,
            question_type="binary",
            resolution=True,
            resolution_string="yes",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="binary?",
        )

        scores = score_arm_for_qid(report_a, report_b, gt)

        assert len(scores) == 2
        metrics = {s.metric for s in scores}
        assert metrics == {"brier", "binary_log_score"}

        for s in scores:
            assert isinstance(s, PairedScore)
            assert s.qid == 42
            assert s.question_type == "binary"
            assert s.delta == pytest.approx(s.score_b - s.score_a)

    def test_score_arm_numeric_returns_log_score_and_crps(self):
        report_a = _uniform_numeric_report()
        report_b = _logistic_numeric_report(center=5.0, sharpness=0.3)
        gt = GroundTruth(
            question_id=99,
            question_type="numeric",
            resolution=5.0,
            resolution_string="5.0",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="numeric?",
        )

        scores = score_arm_for_qid(report_a, report_b, gt)

        assert len(scores) == 2
        metrics = {s.metric for s in scores}
        assert metrics == {"numeric_log_score", "crps"}

        for s in scores:
            assert s.qid == 99
            assert s.question_type == "numeric"

        # Concentrated CDF at resolution should be better on both metrics.
        log_score = next(s for s in scores if s.metric == "numeric_log_score")
        crps = next(s for s in scores if s.metric == "crps")

        assert log_score.score_b > log_score.score_a  # higher log score = better
        assert crps.score_b < crps.score_a  # lower CRPS = better

    def test_score_arm_multiple_choice_returns_one_score(self):
        report_a = _make_mc_report({"a": 0.4, "b": 0.3, "c": 0.3})
        report_b = _make_mc_report({"a": 0.8, "b": 0.1, "c": 0.1})
        gt = GroundTruth(
            question_id=7,
            question_type="multiple_choice",
            resolution="a",
            resolution_string="a",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="mc?",
        )

        scores = score_arm_for_qid(report_a, report_b, gt)

        assert len(scores) == 1
        assert scores[0].metric == "mc_log_score"
        assert scores[0].question_type == "multiple_choice"
        assert scores[0].score_b > scores[0].score_a  # higher = better

    def test_score_arm_canceled_resolution_returns_empty(self):
        report_a = _make_binary_report(0.3)
        report_b = _make_binary_report(0.7)
        gt = GroundTruth(
            question_id=1,
            question_type="binary",
            resolution="annulled",
            resolution_string="annulled",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="?",
        )

        scores = score_arm_for_qid(report_a, report_b, gt)
        assert scores == []

    def test_paired_score_higher_is_better_set_correctly(self):
        report_a = _make_binary_report(0.4)
        report_b = _make_binary_report(0.9)
        binary_gt = GroundTruth(
            question_id=1,
            question_type="binary",
            resolution=True,
            resolution_string="yes",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="?",
        )

        binary_scores = score_arm_for_qid(report_a, report_b, binary_gt)
        for s in binary_scores:
            if s.metric == "brier":
                assert s.higher_is_better is False
            elif s.metric == "binary_log_score":
                assert s.higher_is_better is True

        numeric_scores = score_arm_for_qid(
            _uniform_numeric_report(),
            _logistic_numeric_report(center=5.0),
            GroundTruth(
                question_id=2,
                question_type="numeric",
                resolution=5.0,
                resolution_string="5.0",
                community_prediction=None,
                actual_resolution_time=None,
                question_text="?",
            ),
        )
        for s in numeric_scores:
            if s.metric == "crps":
                assert s.higher_is_better is False
            elif s.metric == "numeric_log_score":
                assert s.higher_is_better is True

        mc_scores = score_arm_for_qid(
            _make_mc_report({"a": 0.4, "b": 0.6}),
            _make_mc_report({"a": 0.8, "b": 0.2}),
            GroundTruth(
                question_id=3,
                question_type="multiple_choice",
                resolution="a",
                resolution_string="a",
                community_prediction=None,
                actual_resolution_time=None,
                question_text="?",
            ),
        )
        for s in mc_scores:
            assert s.higher_is_better is True

    def test_score_arm_numeric_with_bool_resolution_returns_empty_and_warns(self, caplog):
        """M3: a bool resolution routed to a numeric scorer must be rejected.

        Pre-fix: the numeric branch dispatched to _score_numeric_arm without validating
        resolution type; ``bool`` is a subclass of ``int`` in Python, so float(True)=1.0
        silently fell into the lower-bound bucket and produced a numeric log score of 0.0.
        Post-fix: the bool case is rejected with a warning and returns []. Bool MUST be
        checked before int because ``isinstance(True, int)`` is True.
        """
        report_a = _uniform_numeric_report()
        report_b = _logistic_numeric_report(center=5.0)
        gt = GroundTruth(
            question_id=99,
            question_type="numeric",
            resolution=True,
            resolution_string="True",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="numeric mistakenly routed bool?",
        )

        with caplog.at_level("WARNING", logger="metaculus_bot.ablation.scoring"):
            scores = score_arm_for_qid(report_a, report_b, gt)

        assert scores == []
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("99" in m and "bool" in m.lower() for m in warning_messages), (
            f"expected warning mentioning qid 99 + bool; got: {warning_messages}"
        )

    def test_score_arm_numeric_with_below_lower_bound_resolution_scores_correctly(self):
        """M4: OutOfBoundsResolution.BELOW_LOWER_BOUND maps to lower-bound bucket."""
        # Logistic CDF centered at 50 over [0, 100] — most mass is in the middle, very
        # little below the lower bound. Resolution=BELOW_LOWER_BOUND should yield a
        # finite log score (negative because the prediction had little mass there).
        report_a = _logistic_numeric_report(center=50.0, sharpness=10.0, lower=0.0, upper=100.0)
        report_b = _logistic_numeric_report(center=50.0, sharpness=10.0, lower=0.0, upper=100.0)
        gt = GroundTruth(
            question_id=43129,
            question_type="numeric",
            resolution=OutOfBoundsResolution.BELOW_LOWER_BOUND,
            resolution_string="below_lower_bound",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="below-bound numeric?",
        )

        scores = score_arm_for_qid(report_a, report_b, gt)

        assert len(scores) == 2
        metrics = {s.metric for s in scores}
        assert metrics == {"numeric_log_score", "crps"}
        for s in scores:
            assert math.isfinite(s.score_a)
            assert math.isfinite(s.score_b)

    def test_score_arm_numeric_with_above_upper_bound_resolution_scores_correctly(self):
        """M4: OutOfBoundsResolution.ABOVE_UPPER_BOUND maps to upper-bound bucket."""
        report_a = _logistic_numeric_report(center=50.0, sharpness=10.0, lower=0.0, upper=100.0)
        report_b = _logistic_numeric_report(center=50.0, sharpness=10.0, lower=0.0, upper=100.0)
        gt = GroundTruth(
            question_id=43171,
            question_type="numeric",
            resolution=OutOfBoundsResolution.ABOVE_UPPER_BOUND,
            resolution_string="above_upper_bound",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="above-bound numeric?",
        )

        scores = score_arm_for_qid(report_a, report_b, gt)

        assert len(scores) == 2
        metrics = {s.metric for s in scores}
        assert metrics == {"numeric_log_score", "crps"}
        for s in scores:
            assert math.isfinite(s.score_a)
            assert math.isfinite(s.score_b)


# ---------------------------------------------------------------------------
# bootstrap_mean_ci
# ---------------------------------------------------------------------------


class TestBootstrapMeanCi:
    def test_bootstrap_mean_ci_recovers_known_mean(self):
        rng = random.Random(123)
        deltas = [rng.gauss(0.5, 0.1) for _ in range(100)]

        mean, ci_low, ci_high = bootstrap_mean_ci(deltas, n_bootstrap=2000, seed=1)

        assert mean == pytest.approx(0.5, abs=0.05)
        assert ci_low < 0.5 < ci_high
        assert ci_high - ci_low < 0.1  # tight on n=100

    def test_bootstrap_mean_ci_handles_n_lt_5(self):
        deltas = [0.1, -0.2, 0.3]
        mean, ci_low, ci_high = bootstrap_mean_ci(deltas, n_bootstrap=2000, seed=0)
        assert ci_low == mean
        assert ci_high == mean

    def test_bootstrap_mean_ci_with_seed_is_reproducible(self):
        deltas = [0.1, -0.05, 0.2, 0.0, 0.15, -0.1, 0.05, 0.3, -0.2, 0.1]

        result_1 = bootstrap_mean_ci(deltas, n_bootstrap=500, seed=42)
        result_2 = bootstrap_mean_ci(deltas, n_bootstrap=500, seed=42)
        result_different_seed = bootstrap_mean_ci(deltas, n_bootstrap=500, seed=43)

        assert result_1 == result_2
        # Different seed should generally give different CI bounds (but same mean).
        assert result_1[1] != result_different_seed[1] or result_1[2] != result_different_seed[2]

    def test_bootstrap_mean_ci_empty_returns_nan_triple(self):
        # M5: explicit n=0 boundary — empty input must yield (NaN, NaN, NaN).
        mean, lo, hi = bootstrap_mean_ci([], n_bootstrap=2000, seed=0)
        assert math.isnan(mean)
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_bootstrap_mean_ci_n_one_returns_value_triple(self):
        # M5: n=1 boundary — short-circuit returns (mean, mean, mean) where mean = the value.
        mean, lo, hi = bootstrap_mean_ci([0.42], n_bootstrap=2000, seed=0)
        assert mean == pytest.approx(0.42)
        assert lo == mean
        assert hi == mean

    def test_bootstrap_mean_ci_n_two_returns_mean_triple(self):
        # M5: n=2 boundary — n < BOOTSTRAP_MIN_N=5, so triple = (mean, mean, mean).
        mean, lo, hi = bootstrap_mean_ci([0.1, 0.3], n_bootstrap=2000, seed=0)
        assert mean == pytest.approx(0.2)
        assert lo == mean
        assert hi == mean

    def test_bootstrap_mean_ci_n_four_short_circuits(self):
        # M5: n=4 (just below BOOTSTRAP_MIN_N=5) verifies the short-circuit boundary.
        mean, lo, hi = bootstrap_mean_ci([0.0, 0.1, 0.2, 0.3], n_bootstrap=2000, seed=0)
        assert mean == pytest.approx(0.15)
        assert lo == mean
        assert hi == mean

    def test_bootstrap_mean_ci_n_five_runs_full_bootstrap(self):
        # M5: n=5 is the threshold — full bootstrap runs and produces a non-degenerate CI.
        mean, lo, hi = bootstrap_mean_ci([0.0, 0.1, 0.2, 0.3, 0.4], n_bootstrap=2000, seed=0)
        assert mean == pytest.approx(0.2)
        # With variation in the data and 2000 bootstrap samples, the CI should be non-degenerate.
        assert lo < mean < hi


class TestBootstrapMedianCi:
    """M5: explicit n=0/1/2/4/5 boundary coverage for bootstrap_median_ci."""

    def test_bootstrap_median_ci_n_one_returns_value_triple(self):
        median, lo, hi = bootstrap_median_ci([0.42], n_bootstrap=2000, seed=0)
        assert median == pytest.approx(0.42)
        assert lo == median
        assert hi == median

    def test_bootstrap_median_ci_n_two_returns_median_triple(self):
        # n=2 < BOOTSTRAP_MIN_N=5; median([0.1, 0.3]) = 0.2.
        median, lo, hi = bootstrap_median_ci([0.1, 0.3], n_bootstrap=2000, seed=0)
        assert median == pytest.approx(0.2)
        assert lo == median
        assert hi == median

    def test_bootstrap_median_ci_n_four_short_circuits(self):
        median, lo, hi = bootstrap_median_ci([0.0, 0.1, 0.2, 0.3], n_bootstrap=2000, seed=0)
        assert median == pytest.approx(0.15)
        assert lo == median
        assert hi == median

    def test_bootstrap_median_ci_n_five_runs_full_bootstrap(self):
        median, lo, hi = bootstrap_median_ci([0.0, 0.1, 0.2, 0.3, 0.4], n_bootstrap=2000, seed=0)
        assert median == pytest.approx(0.2)
        # With 5 samples and 2000 resamples the bootstrap is degenerate enough that
        # the bounds may equal median, but the function must run without raising.
        assert lo <= median <= hi


# ---------------------------------------------------------------------------
# sign_test
# ---------------------------------------------------------------------------


class TestSignTest:
    def test_sign_test_known_case(self):
        # 8 positive of 10 → two-sided binomial p ≈ 0.109375.
        deltas = [1.0] * 8 + [-1.0] * 2
        p = sign_test(deltas)
        assert p == pytest.approx(0.109375, abs=1e-3)

    def test_sign_test_all_ties_returns_one(self):
        assert sign_test([0.0, 0.0, 0.0]) == 1.0

    def test_sign_test_excludes_ties(self):
        # [1, -1, 0, 0, 1] → effective [1, -1, 1] → 2 of 3 positive.
        # Two-sided p for 2/3 = 1 (every outcome at least as extreme is included).
        p = sign_test([1.0, -1.0, 0.0, 0.0, 1.0])
        assert p == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# wilcoxon_signed_rank
# ---------------------------------------------------------------------------


class TestWilcoxonSignedRank:
    def test_wilcoxon_returns_none_for_small_n(self):
        assert wilcoxon_signed_rank([0.1, 0.2, 0.3]) is None
        assert wilcoxon_signed_rank([0.1, 0.2, 0.3, 0.4, 0.5]) is None

    def test_wilcoxon_basic(self):
        # Strongly positive deltas → very small two-sided p-value.
        deltas = [0.5, 0.4, 0.6, 0.3, 0.55, 0.45, 0.5, 0.35, 0.55, 0.4]
        p = wilcoxon_signed_rank(deltas)
        assert p is not None
        assert p < 0.01


# ---------------------------------------------------------------------------
# aggregate_paired
# ---------------------------------------------------------------------------


def _binary_paired_score(qid: int, score_a: float, score_b: float) -> PairedScore:
    return PairedScore(
        qid=qid,
        question_type="binary",
        metric="binary_log_score",
        score_a=score_a,
        score_b=score_b,
        delta=score_b - score_a,
        higher_is_better=True,
    )


def _numeric_paired_score(qid: int, metric: str, score_a: float, score_b: float, hib: bool) -> PairedScore:
    return PairedScore(
        qid=qid,
        question_type="numeric",
        metric=metric,
        score_a=score_a,
        score_b=score_b,
        delta=score_b - score_a,
        higher_is_better=hib,
    )


class TestAggregatePaired:
    def test_aggregate_paired_groups_by_metric_and_type(self):
        # 3 binary log-score, 3 numeric crps → expect:
        # 1 row per (metric, type) + 1 overall row per metric
        # = 2 per-type rows + 2 overall rows = 4 PairedStats.
        scores: list[PairedScore] = []
        for qid, (a, b) in enumerate([(0.5, 0.6), (0.4, 0.7), (0.3, 0.5)]):
            scores.append(_binary_paired_score(qid, a, b))
        for qid, (a, b) in enumerate([(0.4, 0.3), (0.5, 0.4), (0.6, 0.5)], start=10):
            scores.append(_numeric_paired_score(qid, "crps", a, b, hib=False))

        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)

        assert len(stats) == 4

        per_type = [s for s in stats if s.question_type is not None]
        overall = [s for s in stats if s.question_type is None]

        assert len(per_type) == 2
        assert len(overall) == 2

        per_type_keys = {(s.metric, s.question_type) for s in per_type}
        assert per_type_keys == {("binary_log_score", "binary"), ("crps", "numeric")}

        overall_metrics = {s.metric for s in overall}
        assert overall_metrics == {"binary_log_score", "crps"}

        for s in stats:
            assert s.n == 3
            assert isinstance(s, PairedStats)

    def test_aggregate_paired_overall_uses_all_questions_for_metric(self):
        # Two binary questions and one numeric — different metrics, but a
        # shared "binary_log_score" metric only on binary type. Overall
        # row for "binary_log_score" should aggregate the two binary entries only.
        scores = [
            _binary_paired_score(1, 0.0, 0.2),
            _binary_paired_score(2, 0.0, 0.4),
            _numeric_paired_score(3, "crps", 0.5, 0.4, hib=False),
        ]

        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)

        overall_binary = next(s for s in stats if s.metric == "binary_log_score" and s.question_type is None)
        assert overall_binary.n == 2
        assert overall_binary.mean_delta == pytest.approx(0.3)


class TestAggregatePairedNanDeltas:
    """C3: NaN delta in any group should be filtered (with a logged warning), not propagated.

    Pre-fix: a single NaN delta poisons mean_delta, bootstrap_ci_low/high, median_delta,
    median_ci_low/high, and mean_delta_clean for the entire group. Post-fix: NaN deltas
    are filtered before bootstrapping and a warning surfaces with the metric/qtype context.
    """

    def test_aggregate_paired_filters_nan_deltas_and_warns(self, caplog):
        # 1 NaN delta + 5 valid deltas. Filtered group has n=5 → bootstrap CI computed
        # over the 5 non-NaN entries, mean_delta is finite, warning logged.
        scores: list[PairedScore] = [
            PairedScore(
                qid=1,
                question_type="binary",
                metric="binary_log_score",
                score_a=0.0,
                score_b=float("nan"),
                delta=float("nan"),
                higher_is_better=True,
            ),
        ]
        for qid, b in enumerate([0.1, 0.2, 0.3, 0.4, 0.5], start=2):
            scores.append(_binary_paired_score(qid, 0.0, b))

        with caplog.at_level("WARNING", logger="metaculus_bot.ablation.scoring"):
            stats = aggregate_paired(scores, n_bootstrap=200, seed=0)

        per_type = next(s for s in stats if s.question_type == "binary")
        # n reflects the FILTERED count (NaN dropped).
        assert per_type.n == 5
        # Mean is computed over the 5 non-NaN deltas (0.1..0.5) → 0.3.
        assert not math.isnan(per_type.mean_delta)
        assert per_type.mean_delta == pytest.approx(0.3)
        assert not math.isnan(per_type.bootstrap_ci_low)
        assert not math.isnan(per_type.bootstrap_ci_high)

        # Warning surface: metric + qtype + count of dropped rows.
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("binary_log_score" in m and "binary" in m and "NaN" in m for m in warning_messages), (
            f"expected metric/qtype/NaN-count warning; got: {warning_messages}"
        )

    def test_aggregate_paired_all_nan_deltas_returns_empty_stats(self, caplog):
        # If ALL deltas in a group are NaN, the resulting group has n=0 and stats are NaN.
        scores: list[PairedScore] = [
            PairedScore(
                qid=qid,
                question_type="binary",
                metric="binary_log_score",
                score_a=0.0,
                score_b=float("nan"),
                delta=float("nan"),
                higher_is_better=True,
            )
            for qid in range(3)
        ]

        with caplog.at_level("WARNING", logger="metaculus_bot.ablation.scoring"):
            stats = aggregate_paired(scores, n_bootstrap=200, seed=0)

        per_type = next(s for s in stats if s.question_type == "binary")
        assert per_type.n == 0
        assert math.isnan(per_type.mean_delta)
        assert math.isnan(per_type.bootstrap_ci_low)
        assert math.isnan(per_type.bootstrap_ci_high)
        assert math.isnan(per_type.median_delta)
        assert per_type.n_clean == 0
        assert math.isnan(per_type.mean_delta_clean)


# ---------------------------------------------------------------------------
# render_summary_markdown
# ---------------------------------------------------------------------------


def _build_paired_scores() -> list[PairedScore]:
    scores = []
    for qid, (a, b) in enumerate([(0.5, 0.6), (0.4, 0.7), (0.3, 0.5)]):
        scores.append(_binary_paired_score(qid, a, b))
    for qid, (a, b) in enumerate([(0.5, 0.45), (0.6, 0.4), (0.7, 0.3)], start=10):
        scores.append(_numeric_paired_score(qid, "crps", a, b, hib=False))
    return scores


class TestRenderSummary:
    def test_render_summary_includes_all_sections(self):
        paired_scores = _build_paired_scores()
        stats = aggregate_paired(paired_scores, n_bootstrap=200, seed=0)
        meta = {
            "timestamp": "2026-05-13T10:00:00",
            "n_questions": 6,
            "model_lineup": ["m1", "m2"],
        }

        md = render_summary_markdown(stats, paired_scores, meta)

        assert "Overall summary" in md
        assert "Per-type breakdown" in md
        assert "Per-question diagnostic" in md
        # Low n=3 per group, should warn.
        assert "directional only" in md.lower()

    def test_render_summary_handles_empty_input(self):
        md = render_summary_markdown([], [], {"timestamp": "now", "n_questions": 0})
        assert "n=0" in md or "N=0" in md
        # Make sure it didn't crash / return empty.
        assert len(md) > 0


# ---------------------------------------------------------------------------
# Confounder fields on PairedScore + render_summary
# ---------------------------------------------------------------------------


class TestScoreArmConfounderFields:
    def test_score_arm_populates_confounder_fields_when_payloads_provided(self):
        report_a = _make_binary_report(0.3)
        report_b = _make_binary_report(0.8)
        gt = GroundTruth(
            question_id=42,
            question_type="binary",
            resolution=True,
            resolution_string="yes",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="?",
        )
        payload_a = {
            "stacker_model_used": "fallback",
            "n_forecasters_used": 4,
            "cross_model_aggregation": "",
        }
        payload_b = {
            "stacker_model_used": "primary",
            "n_forecasters_used": 5,
            "cross_model_aggregation": "## Cross-model aggregation\nstuff\n",
        }

        scores = score_arm_for_qid(report_a, report_b, gt, arm_a_payload=payload_a, arm_b_payload=payload_b)

        assert len(scores) == 2
        for s in scores:
            assert s.stacker_a_model_used == "fallback"
            assert s.stacker_b_model_used == "primary"
            assert s.n_forecasters_a == 4
            assert s.n_forecasters_b == 5
            assert s.tools_b_fired is True

    def test_score_arm_tools_b_fired_false_when_cross_model_empty(self):
        report_a = _make_binary_report(0.3)
        report_b = _make_binary_report(0.8)
        gt = GroundTruth(
            question_id=43,
            question_type="binary",
            resolution=True,
            resolution_string="yes",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="?",
        )
        payload_a = {
            "stacker_model_used": "primary",
            "n_forecasters_used": 5,
            "cross_model_aggregation": "",
        }
        payload_b = {
            "stacker_model_used": "primary",
            "n_forecasters_used": 5,
            "cross_model_aggregation": "",
        }

        scores = score_arm_for_qid(report_a, report_b, gt, arm_a_payload=payload_a, arm_b_payload=payload_b)

        assert all(s.tools_b_fired is False for s in scores)

    def test_score_arm_omits_confounder_fields_when_payloads_not_provided(self):
        report_a = _make_binary_report(0.3)
        report_b = _make_binary_report(0.8)
        gt = GroundTruth(
            question_id=44,
            question_type="binary",
            resolution=True,
            resolution_string="yes",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="?",
        )

        scores = score_arm_for_qid(report_a, report_b, gt)

        assert len(scores) == 2
        for s in scores:
            assert s.stacker_a_model_used is None
            assert s.stacker_b_model_used is None
            assert s.n_forecasters_a is None
            assert s.n_forecasters_b is None
            assert s.tools_b_fired is None


def _binary_paired_score_with_confounders(
    qid: int,
    score_a: float,
    score_b: float,
    *,
    a_used: str,
    b_used: str,
    n_a: int,
    n_b: int,
    tools_b: bool,
) -> PairedScore:
    return PairedScore(
        qid=qid,
        question_type="binary",
        metric="binary_log_score",
        score_a=score_a,
        score_b=score_b,
        delta=score_b - score_a,
        higher_is_better=True,
        stacker_a_model_used=a_used,
        stacker_b_model_used=b_used,
        n_forecasters_a=n_a,
        n_forecasters_b=n_b,
        tools_b_fired=tools_b,
    )


class TestRenderSummaryConfounderSection:
    def test_render_summary_includes_confounder_section_when_present(self):
        scores = [
            _binary_paired_score_with_confounders(
                1, 0.5, 0.6, a_used="primary", b_used="primary", n_a=5, n_b=5, tools_b=True
            ),
            _binary_paired_score_with_confounders(
                2, 0.4, 0.7, a_used="fallback", b_used="primary", n_a=4, n_b=5, tools_b=True
            ),
            _binary_paired_score_with_confounders(
                3, 0.3, 0.5, a_used="primary", b_used="fallback", n_a=5, n_b=4, tools_b=False
            ),
        ]
        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)
        md = render_summary_markdown(stats, scores, {"timestamp": "ts", "n_questions": 3})

        assert "Confounder summary" in md
        # Arm A: 2/3 primary, 1/3 fallback (qids 1, 3 primary; 2 fallback)
        assert "Arm A:" in md
        assert "2/3 primary" in md
        assert "1/3 fallback" in md
        # Arm B: 2/3 primary, 1/3 fallback (qids 1, 2 primary; 3 fallback)
        assert "Arm B:" in md

    def test_render_summary_includes_treatment_activation_line(self):
        scores = [
            _binary_paired_score_with_confounders(
                10, 0.5, 0.6, a_used="primary", b_used="primary", n_a=5, n_b=5, tools_b=True
            ),
            _binary_paired_score_with_confounders(
                11, 0.4, 0.7, a_used="primary", b_used="primary", n_a=5, n_b=5, tools_b=True
            ),
            _binary_paired_score_with_confounders(
                12, 0.3, 0.5, a_used="primary", b_used="primary", n_a=5, n_b=5, tools_b=False
            ),
        ]
        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)
        md = render_summary_markdown(stats, scores, {"timestamp": "ts", "n_questions": 3})

        # The activation line is per-question (3 unique qids, 2 fired).
        assert "Treatment activation" in md
        assert "arm B fired tools on 2/3 questions" in md

    def test_render_summary_diagnostic_table_includes_marker_columns(self):
        scores = [
            _binary_paired_score_with_confounders(
                100, 0.5, 0.6, a_used="primary", b_used="fallback", n_a=5, n_b=4, tools_b=True
            ),
        ]
        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)
        md = render_summary_markdown(stats, scores, {"timestamp": "ts", "n_questions": 1})

        assert "A_stacker" in md
        assert "B_stacker" in md
        assert "B_tools" in md
        # Both labels should appear in the row.
        assert "primary" in md
        assert "fallback" in md

    def test_render_summary_omits_confounder_section_when_data_absent(self):
        # Default _build_paired_scores has no confounder data.
        paired_scores = _build_paired_scores()
        stats = aggregate_paired(paired_scores, n_bootstrap=200, seed=0)
        md = render_summary_markdown(stats, paired_scores, {"timestamp": "ts", "n_questions": 6})

        assert "Confounder summary" not in md
        assert "Treatment activation" not in md
        assert "A_stacker" not in md
        assert "B_stacker" not in md
        assert "B_tools" not in md

    def test_render_summary_escapes_pipe_in_stacker_model_name(self):
        """m3: a stacker name containing '|' must be escaped before landing in a markdown table.

        Production code only writes literal "primary"/"fallback", but this guards future
        changes that pass through model identifiers from upstream config (which can legitimately
        contain '|', e.g., "claude-opus-4.7|via-openrouter"). An unescaped '|' breaks the
        table layout — the row gets extra columns, downstream renderers misalign.
        """
        scores = [
            _binary_paired_score_with_confounders(
                1, 0.5, 0.6, a_used="claude|opus", b_used="gpt|5.5", n_a=5, n_b=5, tools_b=True
            ),
        ]
        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)
        md = render_summary_markdown(stats, scores, {"timestamp": "ts", "n_questions": 1})

        # Escaped form (\|) must appear; raw "claude|opus" must not (verifies escaping triggered).
        assert "claude\\|opus" in md
        assert "gpt\\|5.5" in md
        # Verify table integrity: the diagnostic-table row for qid=1 must contain
        # the same number of pipe-delimited columns as the header. The header line:
        # "| qid | type | metric | A | B | Δ | direction | saturation | A_stacker | B_stacker | B_tools |"
        diag_lines = [ln for ln in md.splitlines() if ln.startswith("| 1 | binary |")]
        assert len(diag_lines) == 1, f"expected 1 diagnostic row for qid=1, got: {diag_lines}"
        # Count unescaped pipes in the row: split by `\|` first to drop escapes, then count |.
        row = diag_lines[0]
        unescaped_pipe_count = row.replace("\\|", "").count("|")
        # Header has 12 pipes (11 columns + 2 outer borders → 12 separators).
        assert unescaped_pipe_count == 12, (
            f"row has {unescaped_pipe_count} unescaped pipes; expected 12 for table integrity. row={row!r}"
        )


# ---------------------------------------------------------------------------
# Statistical-validity fixes
# ---------------------------------------------------------------------------


class TestPerGroupBootstrapSeeding:
    """Fix 1: each (metric, qtype) group must use a distinct bootstrap seed.

    Reproducibility is the simplest property to verify: calling aggregate_paired
    twice with the same top-level seed must produce identical PairedStats across
    all groups. This holds whether seeds are shared or distinct, but combined
    with the per-group-distinct-seed implementation it ensures the per-group CIs
    are not artificially correlated by sharing RNG state.
    """

    def test_per_group_bootstrap_uses_distinct_seeds(self):
        # Two metrics x two qtypes (binary/numeric for binary_log_score-like
        # metric is unusual but irrelevant — we just need >=2 groups so that a
        # seed-sharing bug would surface as identical CIs between groups whose
        # deltas happen to be equal arrays).
        scores: list[PairedScore] = []
        # Group 1: binary, metric=brier, deltas = [0.1, 0.2, 0.3, 0.4, 0.5]
        for qid, (a, b) in enumerate([(0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (0.0, 0.4), (0.0, 0.5)]):
            scores.append(
                PairedScore(
                    qid=qid,
                    question_type="binary",
                    metric="brier",
                    score_a=a,
                    score_b=b,
                    delta=b - a,
                    higher_is_better=False,
                )
            )
        # Group 2: numeric, metric=crps, identical deltas [0.1, 0.2, 0.3, 0.4, 0.5]
        for qid, (a, b) in enumerate([(0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (0.0, 0.4), (0.0, 0.5)], start=100):
            scores.append(
                PairedScore(
                    qid=qid,
                    question_type="numeric",
                    metric="crps",
                    score_a=a,
                    score_b=b,
                    delta=b - a,
                    higher_is_better=False,
                )
            )

        stats_run_1 = aggregate_paired(scores, n_bootstrap=200, seed=42)
        stats_run_2 = aggregate_paired(scores, n_bootstrap=200, seed=42)

        # Reproducibility property: same top-level seed → identical CIs.
        assert len(stats_run_1) == len(stats_run_2)
        for s1, s2 in zip(stats_run_1, stats_run_2, strict=True):
            assert s1.metric == s2.metric
            assert s1.question_type == s2.question_type
            assert s1.bootstrap_ci_low == s2.bootstrap_ci_low
            assert s1.bootstrap_ci_high == s2.bootstrap_ci_high

        # Distinct-seeds property: two groups with identical delta arrays should
        # produce DIFFERENT bootstrap CIs because each gets a distinct seed.
        # (With shared seeds + identical deltas, the CIs would collide bit-exact.)
        brier_per_type = next(s for s in stats_run_1 if s.metric == "brier" and s.question_type == "binary")
        crps_per_type = next(s for s in stats_run_1 if s.metric == "crps" and s.question_type == "numeric")
        assert brier_per_type.bootstrap_ci_low != crps_per_type.bootstrap_ci_low or (
            brier_per_type.bootstrap_ci_high != crps_per_type.bootstrap_ci_high
        )

    def test_aggregate_paired_changes_with_top_level_seed(self):
        # Same data, different top-level seeds should produce different per-group CIs.
        scores: list[PairedScore] = [_binary_paired_score(qid, 0.0, 0.1 * (qid + 1)) for qid in range(10)]

        stats_seed_0 = aggregate_paired(scores, n_bootstrap=200, seed=0)
        stats_seed_1 = aggregate_paired(scores, n_bootstrap=200, seed=1)

        # At least one CI bound should differ between runs.
        any_differ = any(
            s0.bootstrap_ci_low != s1.bootstrap_ci_low or s0.bootstrap_ci_high != s1.bootstrap_ci_high
            for s0, s1 in zip(stats_seed_0, stats_seed_1, strict=True)
        )
        assert any_differ

    def test_aggregate_paired_reproducible_across_processes(self):
        """C1: Same seed + same data must produce identical CIs across distinct Python processes.

        Python's `hash(tuple_of_strings)` is randomized per process unless PYTHONHASHSEED
        is fixed. This test launches two fresh subprocess invocations with PYTHONHASHSEED
        randomized, asserts the CI bounds match bit-exact. The fix is hashlib.sha256 on
        the encoded key (deterministic across processes).
        """
        import os
        import subprocess
        import sys
        import textwrap

        snippet = textwrap.dedent(
            """
            import os
            os.environ.setdefault("STREAMLIT_SUPPRESS_DEPRECATION_WARNING", "true")
            from metaculus_bot.ablation.scoring import PairedScore, aggregate_paired
            scores = [
                PairedScore(qid=i, question_type="binary", metric="brier",
                            score_a=0.0, score_b=0.1 * (i + 1),
                            delta=0.1 * (i + 1), higher_is_better=False)
                for i in range(8)
            ]
            stats = aggregate_paired(scores, n_bootstrap=200, seed=42)
            for s in stats:
                qtype = s.question_type if s.question_type is not None else "overall"
                print(f"{s.metric}|{qtype}|{s.bootstrap_ci_low:.6f}|{s.bootstrap_ci_high:.6f}")
            """
        )

        env = {**os.environ, "PYTHONHASHSEED": "random"}
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")

        result_1 = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env=env,
            cwd=repo_root,
            check=True,
        )
        result_2 = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True,
            text=True,
            env=env,
            cwd=repo_root,
            check=True,
        )

        # Filter to lines matching our expected metric|qtype|low|high format.
        out_1 = sorted(line for line in result_1.stdout.strip().splitlines() if line.count("|") == 3)
        out_2 = sorted(line for line in result_2.stdout.strip().splitlines() if line.count("|") == 3)
        assert out_1, f"subprocess 1 produced no parseable output: {result_1.stdout!r}\nstderr: {result_1.stderr!r}"
        assert out_1 == out_2, f"CIs differ across processes:\nrun1={out_1}\nrun2={out_2}"


class TestRenderSummaryWarnings:
    """Fix 2: distinguish overall vs per-type 'directional only' warnings.

    Fix 3: include multiple-testing, percentile-bootstrap, and selection-bias caveats.
    """

    def test_render_summary_distinguishes_overall_and_per_type_warnings(self):
        # Construct stats where overall n=50 (above threshold) but per-type
        # binary n=5 (below threshold). Expect: NO overall caveat, YES per-type caveat.
        per_type_binary = PairedStats(
            metric="brier",
            question_type="binary",
            n=5,
            mean_delta=0.01,
            bootstrap_ci_low=-0.02,
            bootstrap_ci_high=0.04,
            sign_test_p=0.5,
            wilcoxon_p=None,
            higher_is_better=False,
        )
        per_type_numeric = PairedStats(
            metric="brier",
            question_type="numeric",
            n=45,
            mean_delta=0.01,
            bootstrap_ci_low=-0.02,
            bootstrap_ci_high=0.04,
            sign_test_p=0.5,
            wilcoxon_p=0.4,
            higher_is_better=False,
        )
        overall = PairedStats(
            metric="brier",
            question_type=None,
            n=50,
            mean_delta=0.01,
            bootstrap_ci_low=-0.01,
            bootstrap_ci_high=0.03,
            sign_test_p=0.5,
            wilcoxon_p=0.4,
            higher_is_better=False,
        )

        md = render_summary_markdown(
            [per_type_binary, per_type_numeric, overall],
            [],
            {"timestamp": "now", "n_questions": 50},
        )

        # The "overall directional only" caveat must NOT fire (overall n=50 > LOW_N_THRESHOLD).
        # If it appears with that exact phrase, the per-type/overall scoping bug returned.
        lower = md.lower()
        assert "overall directional only" not in lower
        # The per-type caveat must fire (per-type binary n=5 < LOW_N_THRESHOLD).
        assert "per-type directional only" in lower
        # Per-type binary row should carry a marker (e.g., asterisk).
        assert "5*" in md or "5 *" in md

    def test_render_summary_distinguishes_when_overall_below_threshold(self):
        # Overall n=10 (below 30), per-type also below. Expect: BOTH caveats fire.
        per_type = PairedStats(
            metric="brier",
            question_type="binary",
            n=10,
            mean_delta=0.0,
            bootstrap_ci_low=0.0,
            bootstrap_ci_high=0.0,
            sign_test_p=1.0,
            wilcoxon_p=None,
            higher_is_better=False,
        )
        overall = PairedStats(
            metric="brier",
            question_type=None,
            n=10,
            mean_delta=0.0,
            bootstrap_ci_low=0.0,
            bootstrap_ci_high=0.0,
            sign_test_p=1.0,
            wilcoxon_p=None,
            higher_is_better=False,
        )

        md = render_summary_markdown(
            [per_type, overall],
            [],
            {"timestamp": "now", "n_questions": 10},
        )

        lower = md.lower()
        assert "overall directional only" in lower
        assert "per-type directional only" in lower

    def test_render_summary_includes_statistical_caveats(self):
        paired_scores = _build_paired_scores()
        stats = aggregate_paired(paired_scores, n_bootstrap=200, seed=0)
        meta = {"timestamp": "now", "n_questions": 6}

        md = render_summary_markdown(stats, paired_scores, meta)

        lower = md.lower()
        # Multiple-testing caveat.
        assert "multiple testing" in lower or "multiple-testing" in lower
        assert "no multiple-testing correction" in lower or "no multiple testing correction" in lower
        # Percentile-bootstrap caveat.
        assert "percentile-bootstrap" in lower or "percentile bootstrap" in lower
        assert "under-cover" in lower or "under cover" in lower
        # Selection-bias caveat.
        assert "selection bias" in lower or "leakage screen" in lower


# ---------------------------------------------------------------------------
# Saturation detection (Package 1: outlier-robust summary)
# ---------------------------------------------------------------------------


class TestSaturationDetection:
    def test_numeric_log_score_at_empirical_floor_is_saturated(self):
        # -219.9756 is the observed empirical floor on qids 43129/42747/43171.
        assert is_score_saturated("numeric_log_score", -219.9756) is True

    def test_numeric_log_score_just_above_threshold_is_not_saturated(self):
        # NUMERIC_LOG_SCORE_FLOOR=-220, frac=0.05 → cutoff at -220*0.95 = -209.0. -194 is above.
        assert is_score_saturated("numeric_log_score", -194.0) is False

    def test_numeric_log_score_well_above_floor_is_not_saturated(self):
        assert is_score_saturated("numeric_log_score", -100.0) is False
        assert is_score_saturated("numeric_log_score", 0.0) is False
        assert is_score_saturated("numeric_log_score", 50.0) is False

    def test_binary_log_score_at_floor_is_saturated(self):
        score = BINARY_LOG_SCORE_FLOOR + 5.0
        assert is_score_saturated("binary_log_score", score) is True

    def test_binary_log_score_above_threshold_is_not_saturated(self):
        # frac=0.05 → cutoff at FLOOR * 0.95 ≈ -1167.33. FLOOR * 0.9 ≈ -1105.89 is above cutoff.
        score = BINARY_LOG_SCORE_FLOOR * 0.9
        assert is_score_saturated("binary_log_score", score) is False

    def test_mc_log_score_floor_depends_on_n_options(self):
        # K=2: floor = 100*(log2(eps)/log2(2) + 1) = 100*(log2(eps) + 1) = same as binary.
        # K=4: floor = 100*(log2(eps)/log2(4) + 1) = 100*(log2(eps)/2 + 1).
        floor_k2 = 100.0 * (math.log2(PROB_CLAMP_MIN) + 1.0)
        floor_k4 = 100.0 * (math.log2(PROB_CLAMP_MIN) / 2.0 + 1.0)
        # At floor + 5 with the right K: saturated.
        assert is_score_saturated("mc_log_score", floor_k2 + 5.0, n_mc_options=2) is True
        assert is_score_saturated("mc_log_score", floor_k4 + 5.0, n_mc_options=4) is True
        # K=2's floor with K=10 should NOT be saturated (very different floors).
        floor_k10 = 100.0 * (math.log2(PROB_CLAMP_MIN) / math.log2(10.0) + 1.0)
        # floor_k2 << floor_k10, so floor_k2+5 is far below floor_k10's saturation cutoff
        # (which should mark it as saturated), but importantly, a score AT floor_k10 with K=2 isn't.
        # Score at floor_k10 (≈ -300) with K=2 (cutoff ≈ -1203) is well above K=2's saturation.
        assert is_score_saturated("mc_log_score", floor_k10, n_mc_options=2) is False

    def test_mc_log_score_no_options_returns_false(self):
        # Without n_mc_options we can't determine the floor → must return False.
        assert is_score_saturated("mc_log_score", -1000.0, n_mc_options=None) is False

    def test_brier_at_floor_is_saturated(self):
        # BRIER_FLOOR ≈ 0.9998. eps=0.01. So 0.96 is well within the saturated region.
        assert is_score_saturated("brier", BRIER_FLOOR - 0.005) is True

    def test_brier_far_from_floor_is_not_saturated(self):
        assert is_score_saturated("brier", 0.5) is False
        assert is_score_saturated("brier", 0.0) is False

    def test_crps_never_saturated(self):
        assert is_score_saturated("crps", 1000.0) is False
        assert is_score_saturated("crps", 0.0) is False
        assert is_score_saturated("crps", -50.0) is False

    def test_unknown_metric_returns_false(self):
        assert is_score_saturated("bogus", -10000.0) is False

    def test_brier_and_binary_log_saturate_consistently_on_same_forecast(self):
        """C2: cross-metric agreement on the same forecast.

        Pre-fix: a fixed absolute eps=25 was 1.0% of brier's [0,1] range but only 2.0% of
        binary log score's distance from 0 to its floor. Result: a forecast like p=1.4e-4 with
        outcome=True saturated brier (cutoff 0.9898) but NOT binary log score (cutoff -1203.77),
        even though both metrics describe the same "very confident wrong" forecast. The fix
        switches log-score eps to fractional (5% of distance from 0 to floor) so the boundary
        shifts in proportion to each metric's range and the same forecast gets a consistent flag.
        """
        from metaculus_bot.scoring_common import binary_log_score, brier_score

        # p=1.4e-4 / outcome=True is in the transition window:
        # - brier=0.999720 (saturated under both old and new eps)
        # - binary_log_score=-1180.23 (NOT saturated under old eps=25; cutoff=-1203.77)
        #                              (saturated under new frac=0.05; cutoff=-1167.33)
        p = 1.4e-4
        outcome = True
        b = brier_score(p, outcome)
        ll = binary_log_score(p, outcome)
        assert is_score_saturated("brier", b) is True
        assert is_score_saturated("binary_log_score", ll) is True

    def test_open_boundary_minimum_pmf_is_not_saturated(self):
        """C2: a legitimately-pinned open-bound minimum should not register as saturated.

        Numeric log score at the open-bound minimum PMF (cdf[0]=0.001) computes to -195.60
        via 50*ln(0.001/0.05). Under old eps=25, cutoff was -195.0 — so the *enforced
        minimum* sat just barely below the saturation cutoff, producing false-positive
        flags. Under new frac=0.05, cutoff = -220*0.95 = -209.0, comfortably above the
        legitimate boundary score.
        """
        score_at_open_bound_min = 50.0 * math.log(0.001 / 0.05)
        assert score_at_open_bound_min == pytest.approx(-195.60, abs=0.05)
        assert is_score_saturated("numeric_log_score", score_at_open_bound_min) is False


# ---------------------------------------------------------------------------
# Saturation flags through score_arm_for_qid
# ---------------------------------------------------------------------------


def _saturating_numeric_report(lower: float = 0.0, upper: float = 100.0) -> MagicMock:
    """Uniform 201-pt CDF — when resolution is far out of bounds, log-score saturates."""
    n = 201
    xs = list(np.linspace(lower, upper, n))
    ys = [i / (n - 1) for i in range(n)]
    return _make_numeric_report(lower=lower, upper=upper, cdf_xs=xs, cdf_ys=ys)


class TestScoreArmSaturationFlags:
    def test_numeric_arm_saturation_when_resolution_below_bounds(self):
        # Uniform CDF over [0,100], resolution=1000 → falls in above-bound bucket
        # which has min-step PMF mass → log-score near floor.
        report_a = _logistic_numeric_report(center=50.0, sharpness=10.0, lower=0.0, upper=100.0)
        report_b = _saturating_numeric_report(lower=0.0, upper=100.0)
        gt = GroundTruth(
            question_id=43171,
            question_type="numeric",
            resolution=1000.0,  # far above upper bound = 100
            resolution_string="1000.0",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="numeric?",
        )

        scores = score_arm_for_qid(report_a, report_b, gt)

        assert len(scores) == 2
        log_score = next(s for s in scores if s.metric == "numeric_log_score")
        # Both arms scored against an out-of-bounds resolution should saturate.
        # The uniform arm B hits floor cleanly. Arm A (logistic centered at 50) should also saturate
        # since logistic CDF at upper bucket is essentially 1.0 - tiny, but let's just check arm B.
        assert log_score.is_saturated_b is True

    def test_binary_arm_saturation_when_predicting_wrong_with_high_confidence(self):
        # Predicting prob=0.0001 but outcome=True → binary log score at floor.
        report_a = _make_binary_report(0.5)
        report_b = _make_binary_report(0.0001)  # extremely confident wrong
        gt = GroundTruth(
            question_id=99,
            question_type="binary",
            resolution=True,
            resolution_string="yes",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="?",
        )

        scores = score_arm_for_qid(report_a, report_b, gt)

        log_score = next(s for s in scores if s.metric == "binary_log_score")
        brier = next(s for s in scores if s.metric == "brier")

        assert log_score.is_saturated_b is True
        assert brier.is_saturated_b is True
        # Arm A (0.5 prob) should not saturate.
        assert log_score.is_saturated_a is False
        assert brier.is_saturated_a is False

    def test_mc_arm_saturation_uses_correct_k(self):
        # 4-option MC, correct option gets ~PROB_CLAMP_MIN → saturated.
        # Note: probabilities in the report don't need to be normalized for this test;
        # what matters is what mc_log_score sees.
        report_a = _make_mc_report({"a": 0.4, "b": 0.2, "c": 0.2, "d": 0.2})
        report_b = _make_mc_report({"a": PROB_CLAMP_MIN, "b": 0.4, "c": 0.3, "d": 0.3 - PROB_CLAMP_MIN})
        gt = GroundTruth(
            question_id=7,
            question_type="multiple_choice",
            resolution="a",
            resolution_string="a",
            community_prediction=None,
            actual_resolution_time=None,
            question_text="?",
        )

        scores = score_arm_for_qid(report_a, report_b, gt)
        assert len(scores) == 1
        assert scores[0].is_saturated_b is True
        assert scores[0].is_saturated_a is False

    def test_default_paired_score_has_saturation_false(self):
        ps = PairedScore(
            qid=1,
            question_type="binary",
            metric="brier",
            score_a=0.1,
            score_b=0.2,
            delta=0.1,
            higher_is_better=False,
        )
        assert ps.is_saturated_a is False
        assert ps.is_saturated_b is False


# ---------------------------------------------------------------------------
# Median Δ + bootstrap CI on median
# ---------------------------------------------------------------------------


class TestMedianDelta:
    def test_bootstrap_median_robust_to_single_outlier(self):
        # 11 deltas: 10 small + 1 huge outlier. Mean is dragged; median sticks near 0.
        # n must be large enough that bootstrap resamples reliably preserve the median's
        # robustness to the single outlier (with only n=5 the outlier can become majority
        # in a meaningful fraction of resamples).
        deltas = [0.0] * 10 + [-200.0]

        median, lo, hi = bootstrap_median_ci(deltas, n_bootstrap=5000, seed=0)
        mean, lo_mean, hi_mean = bootstrap_mean_ci(deltas, n_bootstrap=5000, seed=0)

        # Median is 0.0 (middle of 11 sorted values).
        assert median == pytest.approx(0.0, abs=1e-9)
        # Mean dragged way negative: -200/11 ≈ -18.18.
        assert mean == pytest.approx(-200.0 / 11.0, abs=1e-9)
        # Median CI should be tighter than mean CI on this data.
        median_width = hi - lo
        mean_width = hi_mean - lo_mean
        assert median_width < mean_width

    def test_bootstrap_median_ci_returns_triple_for_low_n(self):
        deltas = [0.1, 0.2, 0.3]  # n=3 < BOOTSTRAP_MIN_N=5
        median, lo, hi = bootstrap_median_ci(deltas, n_bootstrap=2000, seed=0)
        assert median == pytest.approx(0.2, abs=1e-9)
        assert lo == median
        assert hi == median

    def test_bootstrap_median_ci_handles_empty(self):
        median, lo, hi = bootstrap_median_ci([], n_bootstrap=200, seed=0)
        assert math.isnan(median)
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_bootstrap_median_ci_is_reproducible(self):
        deltas = [0.1, -0.05, 0.2, 0.0, 0.15, -0.1, 0.05, 0.3, -0.2, 0.1]
        result_1 = bootstrap_median_ci(deltas, n_bootstrap=500, seed=42)
        result_2 = bootstrap_median_ci(deltas, n_bootstrap=500, seed=42)
        result_diff = bootstrap_median_ci(deltas, n_bootstrap=500, seed=43)
        assert result_1 == result_2
        assert result_1[1] != result_diff[1] or result_1[2] != result_diff[2]

    def test_aggregate_paired_includes_median_in_stats(self):
        # 5 deltas including an outlier — make sure median is populated and not NaN.
        scores: list[PairedScore] = []
        for qid, (a, b) in enumerate([(0.0, 0.0), (0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (0.0, -100.0)]):
            scores.append(_binary_paired_score(qid, a, b))

        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)
        overall = next(s for s in stats if s.question_type is None)

        assert not math.isnan(overall.median_delta)
        assert overall.median_delta == pytest.approx(0.1, abs=1e-9)
        assert not math.isnan(overall.median_ci_low)
        assert not math.isnan(overall.median_ci_high)


# ---------------------------------------------------------------------------
# n_clean and mean_delta_clean (over non-saturated pairs only)
# ---------------------------------------------------------------------------


def _binary_paired_score_with_sat(
    qid: int, score_a: float, score_b: float, sat_a: bool = False, sat_b: bool = False
) -> PairedScore:
    return PairedScore(
        qid=qid,
        question_type="binary",
        metric="binary_log_score",
        score_a=score_a,
        score_b=score_b,
        delta=score_b - score_a,
        higher_is_better=True,
        is_saturated_a=sat_a,
        is_saturated_b=sat_b,
    )


class TestPairedStatsCleanMean:
    def test_n_clean_excludes_saturated_pairs(self):
        # 5 pairs: 3 clean, 2 saturated (1 a-sat, 1 b-sat).
        scores = [
            _binary_paired_score_with_sat(1, 0.0, 0.1, sat_a=False, sat_b=False),
            _binary_paired_score_with_sat(2, 0.0, 0.2, sat_a=False, sat_b=False),
            _binary_paired_score_with_sat(3, 0.0, 0.3, sat_a=False, sat_b=False),
            _binary_paired_score_with_sat(4, 0.0, -50.0, sat_a=True, sat_b=False),
            _binary_paired_score_with_sat(5, 0.0, -100.0, sat_a=False, sat_b=True),
        ]

        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)
        overall = next(s for s in stats if s.question_type is None)

        assert overall.n == 5
        assert overall.n_clean == 3
        # Mean of the 3 clean deltas: (0.1 + 0.2 + 0.3) / 3 = 0.2
        assert overall.mean_delta_clean == pytest.approx(0.2, abs=1e-9)

    def test_n_clean_zero_yields_nan_mean_clean(self):
        scores = [
            _binary_paired_score_with_sat(1, 0.0, -50.0, sat_a=True, sat_b=False),
            _binary_paired_score_with_sat(2, 0.0, -100.0, sat_a=False, sat_b=True),
            _binary_paired_score_with_sat(3, 0.0, 0.0, sat_a=True, sat_b=True),
        ]

        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)
        overall = next(s for s in stats if s.question_type is None)

        assert overall.n_clean == 0
        assert math.isnan(overall.mean_delta_clean)

    def test_all_clean_means_clean_equals_mean(self):
        scores = [_binary_paired_score_with_sat(qid, 0.0, 0.1 * (qid + 1)) for qid in range(5)]
        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)
        overall = next(s for s in stats if s.question_type is None)
        assert overall.n_clean == 5
        assert overall.mean_delta_clean == pytest.approx(overall.mean_delta, abs=1e-9)


# ---------------------------------------------------------------------------
# Summary rendering with saturation column + new headers
# ---------------------------------------------------------------------------


class TestSummaryWithSaturation:
    def test_per_question_diagnostic_includes_saturation_column(self):
        scores = [
            _binary_paired_score_with_sat(1, 0.5, 0.6, sat_a=False, sat_b=False),
            _binary_paired_score_with_sat(2, 0.4, -100.0, sat_a=False, sat_b=True),
            _binary_paired_score_with_sat(3, -100.0, 0.5, sat_a=True, sat_b=False),
        ]
        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)
        md = render_summary_markdown(stats, scores, {"timestamp": "ts", "n_questions": 3})

        # Per-question diagnostic must include a saturation column header.
        assert "saturation" in md.lower()

    def test_summary_marks_both_saturated_rows_as_draws(self):
        scores = [
            _binary_paired_score_with_sat(1, 0.5, 0.6, sat_a=False, sat_b=False),  # clean
            _binary_paired_score_with_sat(2, -100.0, 0.5, sat_a=True, sat_b=False),  # a_sat
            _binary_paired_score_with_sat(3, -100.0, -100.0, sat_a=True, sat_b=True),  # both
        ]
        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)
        md = render_summary_markdown(stats, scores, {"timestamp": "ts", "n_questions": 3})

        # Each label should appear in the rendered table.
        assert "clean" in md.lower()
        assert "a_sat" in md.lower()
        assert "b_sat" in md.lower() or "a_sat" in md.lower()  # both labels potentially elided
        assert "both" in md.lower()

    def test_caveats_explains_saturation(self):
        scores = [_binary_paired_score_with_sat(1, 0.5, 0.6, sat_a=False, sat_b=False)]
        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)
        md = render_summary_markdown(stats, scores, {"timestamp": "ts", "n_questions": 1})

        lower = md.lower()
        assert "saturat" in lower  # "saturated" or "saturation"

    def test_overall_table_includes_median_and_nosat_columns(self):
        scores = [_binary_paired_score_with_sat(qid, 0.0, 0.1 * (qid + 1)) for qid in range(5)]
        stats = aggregate_paired(scores, n_bootstrap=200, seed=0)
        md = render_summary_markdown(stats, scores, {"timestamp": "ts", "n_questions": 5})

        # Look for column headers in the rendered markdown.
        assert "Median Δ" in md or "median" in md.lower()
        assert "NoSat" in md or "nosat" in md.lower() or "non-saturated" in md.lower()

    def test_floor_constants_have_expected_relative_ordering(self):
        # These shouldn't change across PROB_CLAMP_MIN tweaks but should stay sane.
        # Numeric floor is at -220 (empirical).
        # Binary floor is much more negative (probability scale uses 100x scaling).
        # Brier floor is near 1.0.
        assert NUMERIC_LOG_SCORE_FLOOR == pytest.approx(-220.0, abs=1e-9)
        assert BINARY_LOG_SCORE_FLOOR < NUMERIC_LOG_SCORE_FLOOR  # binary much more negative
        assert 0.9 < BRIER_FLOOR < 1.0
