"""Tests for the W5 offline replay + iterated-k-fold CV harness.

Unit tests use SYNTHETIC fixtures (a tiny fake question + a few fake forecaster CDFs / probs)
so they don't depend on the 50 real prod qids being present — CI stays self-contained. One
integration-style test loads a couple of real qids end-to-end IF the prod cache exists, and
skips gracefully otherwise.

Mirrors the pytest-class style of tests/test_probabilistic_tools.py.
"""

from __future__ import annotations

import math
import socket
from pathlib import Path

import numpy as np
import pytest
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.ablation.cache import AblationCache
from metaculus_bot.ablation.offline_replay import (
    MEDIAN_BASELINE,
    SATURATION_THRESHOLD,
    BinaryRecord,
    ConfigCVResult,
    MCRecord,
    NetworkAccessDuringReplayError,
    NumericRecord,
    ReplayDataset,
    binary_overconfidence,
    build_binary_configs,
    build_mc_configs,
    build_numeric_configs,
    count_saturation_events,
    is_degenerate_config,
    iterated_kfold_cv,
    load_replay_dataset,
    no_network,
    score_all_binary,
    score_all_mc,
    score_all_numeric,
    score_binary,
    score_mc,
    score_numeric,
)

PROD_CACHE_ROOT = Path("backtests/ablation_prod")


# Synthetic fixtures


def _normal_cdf_percentiles(question: NumericQuestion, mean: float, sd: float, n: int = 201) -> list[Percentile]:
    """A constraint-loose 201-point normal CDF on the question grid (for aggregation tests)."""
    grid = np.linspace(float(question.lower_bound), float(question.upper_bound), n)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)((grid - mean) / (sd * math.sqrt(2.0))))
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[0] = 0.0
    cdf[-1] = 1.0
    cdf = np.maximum.accumulate(cdf)
    return [Percentile(value=float(grid[i]), percentile=float(cdf[i])) for i in range(n)]


@pytest.fixture
def numeric_question() -> NumericQuestion:
    return NumericQuestion(
        question_text="synthetic numeric",
        lower_bound=0.0,
        upper_bound=100.0,
        open_lower_bound=False,
        open_upper_bound=False,
        zero_point=None,
        id_of_question=9001,
    )


@pytest.fixture
def mc_question() -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="synthetic mc",
        options=["A", "B", "C"],
        id_of_question=9002,
    )


@pytest.fixture
def binary_question() -> BinaryQuestion:
    return BinaryQuestion(question_text="synthetic binary", id_of_question=9003)


# Zero-network guarantee


class TestZeroNetwork:
    def test_no_network_blocks_outbound_resolution(self):
        # Inside no_network(), any attempt to resolve a real hostname (the first step of
        # every outbound HTTP/LLM/provider call) must raise — making a live call impossible.
        with no_network():
            with pytest.raises(NetworkAccessDuringReplayError):
                socket.getaddrinfo("openrouter.ai", 443)

    def test_no_network_allows_localhost(self):
        # Local tooling resolution stays allowed so the guard is targeted at the live providers.
        with no_network():
            socket.getaddrinfo("localhost", 0)  # must not raise

    def test_no_network_restores_getaddrinfo(self):
        original = socket.getaddrinfo
        with no_network():
            pass
        assert socket.getaddrinfo is original

    def test_load_replay_dataset_makes_no_network_call(self):
        # Loading the dataset under no_network() must not raise — it is pure disk I/O.
        # Skip if the prod cache is absent (CI). An empty cache loads to an empty dataset.
        if not PROD_CACHE_ROOT.exists():
            pytest.skip("prod ablation cache not present")
        with no_network():
            dataset = load_replay_dataset(AblationCache(root=str(PROD_CACHE_ROOT)))
        assert isinstance(dataset, ReplayDataset)


# Binary configs + scoring


class TestBinaryConfigs:
    def test_median_baseline_is_median_of_p_models(self, binary_question):
        record = BinaryRecord(
            qid=1,
            question=binary_question,
            outcome=True,
            p_models=[0.3, 0.5, 0.7],
            p_maths=[],
        )
        configs = build_binary_configs()
        assert configs[MEDIAN_BASELINE](record) == pytest.approx(0.5)

    def test_shrinkage_with_no_p_math_falls_back_to_median(self, binary_question):
        record = BinaryRecord(qid=1, question=binary_question, outcome=True, p_models=[0.2, 0.4, 0.9], p_maths=[])
        configs = build_binary_configs()
        median = float(np.median(record.p_models))
        # Every shrinkage config must fall back to the median p_model when no p_math exists.
        for name, config in configs.items():
            assert config(record) == pytest.approx(median), name

    def test_shrinkage_pulls_toward_p_math(self, binary_question):
        # p_model median high (0.9), p_math median low (0.1). w=0.5 should pull the
        # pooled estimate roughly to the logit midpoint (~0.5 by symmetry).
        record = BinaryRecord(
            qid=1, question=binary_question, outcome=True, p_models=[0.9, 0.9, 0.9], p_maths=[0.1, 0.1, 0.1]
        )
        configs = build_binary_configs()
        out = configs["shrink_w0.5"](record)
        assert out == pytest.approx(0.5, abs=1e-6)
        # A smaller weight should stay closer to p_model (0.9).
        out_small = configs["shrink_w0.1"](record)
        assert out_small > out

    def test_overconfidence_zero_when_estimates_agree(self, binary_question):
        record = BinaryRecord(
            qid=1, question=binary_question, outcome=True, p_models=[0.3, 0.3, 0.3], p_maths=[0.3, 0.3, 0.3]
        )
        assert binary_overconfidence(record) == pytest.approx(0.0, abs=1e-9)

    def test_overconfidence_none_without_p_math(self, binary_question):
        record = BinaryRecord(qid=1, question=binary_question, outcome=True, p_models=[0.3], p_maths=[])
        assert binary_overconfidence(record) is None

    def test_score_binary_higher_for_confident_correct(self, binary_question):
        confident = BinaryRecord(qid=1, question=binary_question, outcome=True, p_models=[0.9], p_maths=[])
        unsure = BinaryRecord(qid=2, question=binary_question, outcome=True, p_models=[0.5], p_maths=[])
        assert score_binary(confident, 0.9)[0] > score_binary(unsure, 0.5)[0]


# MC configs + scoring


class TestMCConfigs:
    def _record(self, mc_question, vectors, correct_index=0) -> MCRecord:
        return MCRecord(
            qid=1,
            question=mc_question,
            option_order=["A", "B", "C"],
            correct_option_index=correct_index,
            option_vectors=vectors,
        )

    def test_median_baseline_sums_to_one(self, mc_question):
        record = self._record(
            mc_question,
            [{"A": 0.6, "B": 0.3, "C": 0.1}, {"A": 0.5, "B": 0.4, "C": 0.1}, {"A": 0.7, "B": 0.2, "C": 0.1}],
        )
        out = build_mc_configs()[MEDIAN_BASELINE](record)
        assert sum(out) == pytest.approx(1.0)
        assert len(out) == 3

    def test_pool_mc_differs_from_median_on_disagreement(self, mc_question):
        # Forecasters disagree sharply on A; geometric pooling (penalizes confident
        # disagreement) should differ from the per-option median.
        # Asymmetric disagreement (a perfectly symmetric setup would make both the median
        # and the geometric pool collapse to uniform, hiding the difference).
        record = self._record(
            mc_question,
            [{"A": 0.7, "B": 0.2, "C": 0.1}, {"A": 0.2, "B": 0.7, "C": 0.1}, {"A": 0.05, "B": 0.15, "C": 0.8}],
        )
        configs = build_mc_configs()
        median = configs[MEDIAN_BASELINE](record)
        pooled = configs["pool_mc"](record)
        assert pooled != pytest.approx(median, abs=1e-6)
        assert sum(pooled) == pytest.approx(1.0)

    def test_pool_mc_dirichlet_smooths_toward_uniform(self, mc_question):
        record = self._record(
            mc_question,
            [{"A": 0.9, "B": 0.05, "C": 0.05}, {"A": 0.9, "B": 0.05, "C": 0.05}, {"A": 0.9, "B": 0.05, "C": 0.05}],
        )
        configs = build_mc_configs()
        unsmoothed = configs["pool_mc"](record)
        smoothed = configs["pool_mc_dir10"](record)
        # Smoothing pulls the dominant option down toward 1/3.
        assert smoothed[0] < unsmoothed[0]
        assert sum(smoothed) == pytest.approx(1.0)

    def test_score_mc_rewards_mass_on_correct(self, mc_question):
        record = self._record(mc_question, [{"A": 0.9, "B": 0.05, "C": 0.05}], correct_index=0)
        good = score_mc(record, [0.9, 0.05, 0.05])
        bad = score_mc(record, [0.05, 0.9, 0.05])
        assert good > bad


# Numeric configs + scoring


class TestNumericConfigs:
    def _record(self, numeric_question, means_sds, resolution=50.0) -> NumericRecord:
        cdfs = [_normal_cdf_percentiles(numeric_question, m, s) for m, s in means_sds]
        return NumericRecord(qid=1, question=numeric_question, resolution_value=resolution, cdfs=cdfs)

    def test_all_configs_return_201_point_cdf(self, numeric_question):
        record = self._record(numeric_question, [(40, 10), (50, 10), (60, 10)])
        for name, config in build_numeric_configs().items():
            cdf = config(record)
            assert len(cdf) == 201, name
            assert cdf[0] == pytest.approx(0.0, abs=1e-6), name
            assert cdf[-1] == pytest.approx(1.0, abs=1e-6), name
            assert all(cdf[i + 1] >= cdf[i] - 1e-9 for i in range(len(cdf) - 1)), name

    def test_vincentize_differs_from_vertical_median_on_bimodal(self, numeric_question):
        # Two forecasters far apart: vertical median smears, Vincentization preserves
        # the central location. The two aggregates must differ materially.
        record = self._record(numeric_question, [(20, 5), (80, 5)])
        configs = build_numeric_configs()
        vertical = np.array(configs[MEDIAN_BASELINE](record))
        vincent = np.array(configs["vincentize_mean"](record))
        assert np.max(np.abs(vertical - vincent)) > 0.05

    def test_score_numeric_rewards_mass_near_resolution(self, numeric_question):
        on_target = self._record(numeric_question, [(50, 5), (50, 5), (50, 5)], resolution=50.0)
        off_target = self._record(numeric_question, [(10, 5), (10, 5), (10, 5)], resolution=50.0)
        configs = build_numeric_configs()
        good = score_numeric(on_target, configs[MEDIAN_BASELINE](on_target))[0]
        bad = score_numeric(off_target, configs[MEDIAN_BASELINE](off_target))[0]
        assert good > bad

    def test_log_pool_concentrates_on_agreement(self, numeric_question):
        # When forecasters agree, log_pool should be sharp and score well at the consensus.
        record = self._record(numeric_question, [(50, 8), (50, 8), (50, 8)], resolution=50.0)
        cdf = build_numeric_configs()["log_pool"](record)
        assert len(cdf) == 201
        assert math.isfinite(score_numeric(record, cdf)[0])


# Iterated k-fold CV


class TestIteratedKFoldCV:
    def test_returns_result_per_config(self):
        rng = np.random.default_rng(0)
        scores = {
            MEDIAN_BASELINE: rng.normal(0, 1, size=20),
            "challenger": rng.normal(0.5, 1, size=20),
        }
        results = iterated_kfold_cv(scores, k=5, iterations=10)
        assert set(results) == {MEDIAN_BASELINE, "challenger"}
        assert all(isinstance(r, ConfigCVResult) for r in results.values())

    def test_resample_count_is_k_times_iterations(self):
        scores = {MEDIAN_BASELINE: np.arange(20.0), "x": np.arange(20.0) + 1.0}
        results = iterated_kfold_cv(scores, k=5, iterations=10)
        # 5 folds x 10 iterations = 50 resamples (all folds non-empty at n=20, k=5).
        assert results["x"].n_resamples == 50

    def test_baseline_delta_is_zero(self):
        scores = {MEDIAN_BASELINE: np.arange(20.0), "x": np.arange(20.0) + 2.0}
        results = iterated_kfold_cv(scores, k=5, iterations=5)
        assert results[MEDIAN_BASELINE].delta_vs_median_mean == pytest.approx(0.0, abs=1e-12)
        assert results[MEDIAN_BASELINE].delta_vs_median_std == pytest.approx(0.0, abs=1e-12)

    def test_constant_offset_delta_recovered(self):
        # A config that adds a constant +2 to every question's score should have a
        # paired delta of exactly +2 with zero spread.
        base = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0])
        scores = {MEDIAN_BASELINE: base, "plus2": base + 2.0}
        results = iterated_kfold_cv(scores, k=5, iterations=10)
        assert results["plus2"].delta_vs_median_mean == pytest.approx(2.0, abs=1e-9)
        assert results["plus2"].delta_vs_median_std == pytest.approx(0.0, abs=1e-9)

    def test_reproducible(self):
        scores = {MEDIAN_BASELINE: np.arange(15.0), "x": np.arange(15.0) * 1.1}
        r1 = iterated_kfold_cv(scores, k=5, iterations=10)
        r2 = iterated_kfold_cv(scores, k=5, iterations=10)
        assert r1["x"].mean_log_score == r2["x"].mean_log_score
        assert r1["x"].delta_vs_median_std == r2["x"].delta_vs_median_std

    def test_unequal_lengths_raise(self):
        scores = {MEDIAN_BASELINE: np.arange(10.0), "x": np.arange(8.0)}
        with pytest.raises(ValueError, match="same number"):
            iterated_kfold_cv(scores)

    def test_missing_baseline_raises(self):
        scores = {"x": np.arange(10.0)}
        with pytest.raises(ValueError, match="baseline"):
            iterated_kfold_cv(scores)


class TestSaturation:
    def test_counts_blowups(self):
        scores = np.array([0.0, -50.0, -250.0, -1000.0, 10.0])
        assert count_saturation_events(scores) == 2  # only -250 and -1000 are below -200

    def test_threshold_is_negative(self):
        assert SATURATION_THRESHOLD < 0


class TestDegeneracy:
    def test_constant_scores_flagged_degenerate(self):
        # A config collapsed to a constant/uniform prediction scores ~the same on every
        # question -> near-zero spread -> degenerate (mirrors the tail-floor failure mode).
        assert is_degenerate_config(np.array([5.27, 5.27, 2.56, 5.27, 2.56]))

    def test_real_aggregator_not_degenerate(self):
        # A genuine aggregator's per-question scores span tens of points across a
        # heterogeneous question set.
        assert not is_degenerate_config(np.array([80.0, -30.0, 15.0, -45.0, 100.0]))

    def test_single_score_not_degenerate(self):
        # Too few points to judge; don't false-positive.
        assert not is_degenerate_config(np.array([5.0]))


# End-to-end scoring on synthetic records (no real data)


class TestScoreAll:
    def test_score_all_binary_shapes(self, binary_question):
        records = [
            BinaryRecord(qid=i, question=binary_question, outcome=bool(i % 2), p_models=[0.4, 0.6], p_maths=[0.5])
            for i in range(6)
        ]
        scored = score_all_binary(records, build_binary_configs())
        assert all(len(arr) == 6 for arr in scored.values())
        assert MEDIAN_BASELINE in scored

    def test_score_all_numeric_shapes(self, numeric_question):
        records = [
            NumericRecord(
                qid=i,
                question=numeric_question,
                resolution_value=50.0,
                cdfs=[_normal_cdf_percentiles(numeric_question, 45 + i, 10) for _ in range(3)],
            )
            for i in range(4)
        ]
        scored = score_all_numeric(records, build_numeric_configs())
        assert all(len(arr) == 4 and np.all(np.isfinite(arr)) for arr in scored.values())

    def test_score_all_mc_shapes(self, mc_question):
        records = [
            MCRecord(
                qid=i,
                question=mc_question,
                option_order=["A", "B", "C"],
                correct_option_index=i % 3,
                option_vectors=[{"A": 0.5, "B": 0.3, "C": 0.2}, {"A": 0.4, "B": 0.4, "C": 0.2}],
            )
            for i in range(5)
        ]
        scored = score_all_mc(records, build_mc_configs())
        assert all(len(arr) == 5 and np.all(np.isfinite(arr)) for arr in scored.values())


# Integration: real prod cache (skips gracefully if absent)


@pytest.mark.skipif(not PROD_CACHE_ROOT.exists(), reason="prod ablation cache not present")
class TestRealCacheIntegration:
    def test_load_returns_finite_scores_end_to_end(self):
        cache = AblationCache(root=str(PROD_CACHE_ROOT))
        dataset = load_replay_dataset(cache)
        assert isinstance(dataset, ReplayDataset)
        total = len(dataset.binary) + len(dataset.mc) + len(dataset.numeric)
        assert total > 0, "expected at least some replayable questions in the prod cache"

        # Score a couple of questions of each type end-to-end; everything finite.
        if dataset.binary:
            scored = score_all_binary(dataset.binary[:3], build_binary_configs())
            assert all(np.all(np.isfinite(arr)) for arr in scored.values())
        if dataset.mc:
            scored = score_all_mc(dataset.mc[:3], build_mc_configs())
            assert all(np.all(np.isfinite(arr)) for arr in scored.values())
        if dataset.numeric:
            scored = score_all_numeric(dataset.numeric[:3], build_numeric_configs())
            assert all(np.all(np.isfinite(arr)) for arr in scored.values())

    def test_cv_runs_on_real_data(self):
        cache = AblationCache(root=str(PROD_CACHE_ROOT))
        dataset = load_replay_dataset(cache)
        if len(dataset.binary) < 5:
            pytest.skip("not enough binary questions for CV")
        scored = score_all_binary(dataset.binary, build_binary_configs())
        results = iterated_kfold_cv(scored, k=5, iterations=10)
        assert MEDIAN_BASELINE in results
        assert all(math.isfinite(r.mean_log_score) for r in results.values())
