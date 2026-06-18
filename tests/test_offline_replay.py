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
from forecasting_tools.data_models.questions import OutOfBoundsResolution

from metaculus_bot.ablation.cache import AblationCache, model_slug_to_filename
from metaculus_bot.ablation.offline_replay import (
    MEDIAN_BASELINE,
    SATURATION_THRESHOLD,
    BinaryRecord,
    ConfigCVResult,
    MCRecord,
    NetworkAccessDuringReplayError,
    NumericRecord,
    ReplayDataset,
    _mc_correct_index,
    _resolution_to_float,
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


class TestBinaryClampConfigs:
    """Tighter probability clamps applied to the median aggregate (after-median primary +
    per-forecaster-before-median secondary). The cached prod probs are already [0.02, 0.98]-
    clamped, so these test a tighter ceiling/floor vs. the incumbent."""

    def test_clamp_configs_registered(self, binary_question):
        configs = build_binary_configs()
        for name in ("clamp_05_95", "clamp_05_95_premedian", "clamp_10_90", "clamp_10_90_premedian"):
            assert name in configs, name

    def test_after_median_clamp_caps_extreme_high(self, binary_question):
        # Median p_model 0.99 (above every clamp ceiling) must be pulled DOWN to the ceiling.
        record = BinaryRecord(qid=1, question=binary_question, outcome=True, p_models=[0.99, 0.99, 0.99], p_maths=[])
        configs = build_binary_configs()
        assert configs["clamp_05_95"](record) == pytest.approx(0.95)
        assert configs["clamp_10_90"](record) == pytest.approx(0.90)

    def test_after_median_clamp_caps_extreme_low(self, binary_question):
        # Median p_model 0.01 (below every clamp floor) must be lifted UP to the floor.
        record = BinaryRecord(qid=1, question=binary_question, outcome=False, p_models=[0.01, 0.01, 0.01], p_maths=[])
        configs = build_binary_configs()
        assert configs["clamp_05_95"](record) == pytest.approx(0.05)
        assert configs["clamp_10_90"](record) == pytest.approx(0.10)

    def test_clamp_leaves_midrange_unchanged(self, binary_question):
        # A mid-range median sits inside every clamp band -> the clamp is a no-op there.
        record = BinaryRecord(qid=1, question=binary_question, outcome=True, p_models=[0.4, 0.5, 0.6], p_maths=[])
        configs = build_binary_configs()
        median = float(np.median(record.p_models))
        for name in ("clamp_05_95", "clamp_05_95_premedian", "clamp_10_90", "clamp_10_90_premedian"):
            assert configs[name](record) == pytest.approx(median), name

    def test_clamp_output_never_exceeds_bounds(self, binary_question):
        # Across a sweep of medians spanning the full [0, 1] range, no clamp config ever
        # produces a probability outside its declared [low, high].
        configs = build_binary_configs()
        bounds = {"clamp_05_95": (0.05, 0.95), "clamp_10_90": (0.10, 0.90)}
        for p in (0.0, 0.001, 0.03, 0.2, 0.5, 0.8, 0.97, 0.999, 1.0):
            record = BinaryRecord(qid=1, question=binary_question, outcome=True, p_models=[p, p, p], p_maths=[])
            for name, (low, high) in bounds.items():
                out_after = configs[name](record)
                out_pre = configs[f"{name}_premedian"](record)
                assert low - 1e-9 <= out_after <= high + 1e-9, (name, p, out_after)
                assert low - 1e-9 <= out_pre <= high + 1e-9, (name, p, out_pre)

    def test_before_vs_after_median_diverge_when_clamp_shifts_median(self, binary_question):
        # The two clamp variants differ only when capping individual members changes the
        # even-count median average. Four probs [0.99, 0.99, 0.80, 0.10], clamp_05_95:
        #   after-median  = clamp(median([0.99,0.99,0.80,0.10]) = 0.895) = 0.895 (in band)
        #   before-median = median(clamp -> [0.95,0.95,0.80,0.10]) = (0.95+0.80)/2 = 0.875
        record = BinaryRecord(
            qid=1, question=binary_question, outcome=True, p_models=[0.99, 0.99, 0.80, 0.10], p_maths=[]
        )
        configs = build_binary_configs()
        assert configs["clamp_05_95"](record) == pytest.approx(0.895)
        assert configs["clamp_05_95_premedian"](record) == pytest.approx(0.875)


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


class TestMcCorrectIndex:
    def test_direct_string_match(self):
        assert _mc_correct_index(["A", "B", "C"], "B") == 1

    def test_float_int_canonicalization_fallback(self):
        # Resolution arrives float-formatted ('2.0') while options are integer strings.
        assert _mc_correct_index(["2", "3"], 2.0) == 0

    def test_returns_none_on_true_no_match(self):
        assert _mc_correct_index(["A", "B"], "Z") is None


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


# ===========================================================================
# F4 + F21 — synthetic-cache coverage for the offline_replay loader layer
#
# The entire data-loading layer (load_replay_dataset + _build_*_record helpers
# + _resolution_to_float + the type-check branches + the min-forecasters skip)
# only ran under skipif(not PROD_CACHE_ROOT.exists()) before this — i.e. NEVER
# in CI, since backtests/ is gitignored. These tests seed a REAL tmp
# AblationCache (a qids manifest + forecaster_outputs JSON) so the real
# deserialization + survivor-filter + question-shim path is exercised.
#
# A real tmp cache is preferred over monkeypatching cache.list_forecaster_outputs
# / read_qids_manifest: the manifest + forecaster payloads round-trip through
# the actual cache write API, _build_question_shim_from_manifest_entry, and
# deserialize_prediction_value — so a regression in any of those is caught,
# not just a regression in the in-memory record assembly.
# ===========================================================================

_REPLAY_DUMMY_MODELS = ("openrouter/test/m0", "openrouter/test/m1", "openrouter/test/m2")

_ELEVEN_PERCENTILE_OFFSETS = (
    (0.025, -30),
    (0.05, -25),
    (0.10, -20),
    (0.20, -12),
    (0.40, -5),
    (0.50, 0),
    (0.60, 5),
    (0.80, 12),
    (0.90, 20),
    (0.95, 25),
    (0.975, 30),
)


def _replay_manifest_entry(
    qid: int,
    qtype: str,
    *,
    gt_resolution,
    gt_string: str,
    extra_metadata: dict | None = None,
    options: list[str] | None = None,
) -> dict:
    """Build a manifest entry matching ``_build_manifest_entry``'s schema.

    ``gt_resolution`` is the already-serialized resolution (bool / float / str,
    or the ``_type``-tagged OutOfBoundsResolution dict) so callers control the
    exact ground-truth round-trip.
    """
    metadata: dict = {
        "open_time": "2026-01-01T00:00:00",
        "scheduled_resolution_time": "2026-05-01T00:00:00",
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    if options is not None:
        metadata["options"] = options
    return {
        "type": qtype,
        "tournament": "spring-aib-2026",
        "question_text": f"Q{qid}?",
        "page_url": f"https://www.metaculus.com/questions/{qid}",
        "id_of_post": qid,
        "resolution_criteria": "rc",
        "fine_print": "",
        "background_info": "bg",
        "ground_truth": {
            "question_id": qid,
            "question_type": _gt_type_for(qtype),
            "resolution": gt_resolution,
            "resolution_string": gt_string,
            "actual_resolution_time": "2026-05-01T00:00:00",
            "question_text": f"Q{qid}?",
            "page_url": f"https://www.metaculus.com/questions/{qid}",
        },
        "question_metadata": metadata,
    }


def _gt_type_for(qtype: str) -> str:
    return qtype


def _binary_replay_payload(model: str, prob: float) -> dict:
    return {
        "model": model,
        "prediction_value": {"type": "binary", "prob": prob},
        "reasoning": f"Model: {model}\n\nrationale text",
        "errors": [],
    }


def _mc_replay_payload(model: str, p_first: float, options: list[str]) -> dict:
    # Two-option simplex: first option gets ``p_first``, the rest the remainder.
    return {
        "model": model,
        "prediction_value": {
            "type": "multiple_choice",
            "options": [
                {"option_name": options[0], "probability": p_first},
                {"option_name": options[1], "probability": 1.0 - p_first},
            ],
        },
        "reasoning": f"Model: {model}\n\nrationale text",
        "errors": [],
    }


def _numeric_replay_payload(model: str, median: float) -> dict:
    """Numeric payload in the post-Bucket-1 full-CDF schema (matches
    ``serialize_prediction_value``): declared percentiles + a monotone linear
    201-point CDF so ``deserialize_prediction_value`` reconstructs a real
    PchipNumericDistribution with a valid ``.cdf``."""
    declared = [{"percentile": p, "value": median + d} for p, d in _ELEVEN_PERCENTILE_OFFSETS]
    cdf_probabilities = [0.001 + (0.998 * i / 200) for i in range(201)]
    return {
        "model": model,
        "prediction_value": {
            "type": "numeric",
            "declared_percentiles": declared,
            "cdf_probabilities": cdf_probabilities,
            "lower_bound": 0.0,
            "upper_bound": 100.0,
            "open_lower_bound": False,
            "open_upper_bound": False,
            "zero_point": None,
            "cdf_size": 201,
        },
        "reasoning": f"Model: {model}\n\nrationale text",
        "errors": [],
    }


_NUMERIC_BOUNDS_METADATA = {
    "lower_bound": 0.0,
    "upper_bound": 100.0,
    "open_lower_bound": False,
    "open_upper_bound": False,
    "zero_point": None,
    "unit_of_measure": None,
}


def _seed_n_forecasters(cache: AblationCache, qid: int, payloads: list[dict]) -> None:
    for payload in payloads:
        cache.write_forecaster_output(
            qid=qid,
            model_slug=model_slug_to_filename(payload["model"]),
            payload=payload,
        )


@pytest.fixture
def _seeded_replay_cache(tmp_path) -> AblationCache:
    """A tmp AblationCache seeded with one binary, one MC, and one numeric qid.

    Each qid has 2 surviving forecasters (>= ABLATION_MIN_FORECASTERS=2) so
    none is dropped by the min-forecasters guard.
    """
    cache = AblationCache(tmp_path / "abl")

    # Binary qid 11 — resolves YES (True).
    cache.append_qids_manifest({11: _replay_manifest_entry(11, "binary", gt_resolution=True, gt_string="YES")})
    _seed_n_forecasters(
        cache,
        11,
        [_binary_replay_payload(_REPLAY_DUMMY_MODELS[i], p) for i, p in enumerate((0.4, 0.6))],
    )

    # MC qid 22 — options Red/Blue, resolves Red (correct index 0).
    cache.append_qids_manifest(
        {
            22: _replay_manifest_entry(
                22, "multiple_choice", gt_resolution="Red", gt_string="Red", options=["Red", "Blue"]
            )
        }
    )
    _seed_n_forecasters(
        cache,
        22,
        [_mc_replay_payload(_REPLAY_DUMMY_MODELS[i], p, ["Red", "Blue"]) for i, p in enumerate((0.7, 0.5))],
    )

    # Numeric qid 33 — bounds [0, 100], resolves 50.0.
    cache.append_qids_manifest(
        {
            33: _replay_manifest_entry(
                33, "numeric", gt_resolution=50.0, gt_string="50.0", extra_metadata=_NUMERIC_BOUNDS_METADATA
            )
        }
    )
    _seed_n_forecasters(
        cache,
        33,
        [_numeric_replay_payload(_REPLAY_DUMMY_MODELS[i], m) for i, m in enumerate((45.0, 55.0))],
    )

    return cache


class TestLoadReplayDatasetFromSeededCache:
    def test_loads_one_record_per_type(self, _seeded_replay_cache: AblationCache) -> None:
        # Run under no_network() to also exercise the zero-API guarantee — pure
        # disk reads must never touch the network.
        with no_network():
            dataset = load_replay_dataset(_seeded_replay_cache)
        assert isinstance(dataset, ReplayDataset)
        assert len(dataset.binary) == 1
        assert len(dataset.mc) == 1
        assert len(dataset.numeric) == 1

    def test_binary_record_shape_and_outcome(self, _seeded_replay_cache: AblationCache) -> None:
        with no_network():
            dataset = load_replay_dataset(_seeded_replay_cache)
        record = dataset.binary[0]
        assert isinstance(record, BinaryRecord)
        assert record.qid == 11
        assert isinstance(record.question, BinaryQuestion)
        assert record.outcome is True
        # Both surviving forecasters' probabilities deserialized.
        assert sorted(record.p_models) == pytest.approx([0.4, 0.6])
        # No structured JSON block in the rationale -> no reconstructed p_math.
        assert record.p_maths == []

    def test_mc_record_shape_and_correct_index(self, _seeded_replay_cache: AblationCache) -> None:
        with no_network():
            dataset = load_replay_dataset(_seeded_replay_cache)
        record = dataset.mc[0]
        assert isinstance(record, MCRecord)
        assert record.qid == 22
        assert isinstance(record.question, MultipleChoiceQuestion)
        assert record.option_order == ["Red", "Blue"]
        assert record.correct_option_index == 0  # resolved "Red"
        assert len(record.option_vectors) == 2
        # Each vector covers every option and sums to ~1.
        for vec in record.option_vectors:
            assert set(vec) == {"Red", "Blue"}
            assert sum(vec.values()) == pytest.approx(1.0)

    def test_numeric_record_shape_and_resolution(self, _seeded_replay_cache: AblationCache) -> None:
        with no_network():
            dataset = load_replay_dataset(_seeded_replay_cache)
        record = dataset.numeric[0]
        assert isinstance(record, NumericRecord)
        assert record.qid == 33
        assert isinstance(record.question, NumericQuestion)
        # In-bounds float resolution passes straight through.
        assert record.resolution_value == pytest.approx(50.0)
        assert len(record.cdfs) == 2
        # Each forecaster's reconstructed CDF is the full 201-point grid.
        for cdf in record.cdfs:
            assert len(cdf) == 201
            assert all(isinstance(p, Percentile) for p in cdf)

    def test_loaded_records_score_finite_end_to_end(self, _seeded_replay_cache: AblationCache) -> None:
        # The whole point of the loader is feeding the scorers; confirm the
        # seeded records produce finite scores across every config.
        with no_network():
            dataset = load_replay_dataset(_seeded_replay_cache)
        assert all(
            np.all(np.isfinite(arr)) for arr in score_all_binary(dataset.binary, build_binary_configs()).values()
        )
        assert all(np.all(np.isfinite(arr)) for arr in score_all_mc(dataset.mc, build_mc_configs()).values())
        assert all(
            np.all(np.isfinite(arr)) for arr in score_all_numeric(dataset.numeric, build_numeric_configs()).values()
        )


class TestResolutionToFloat:
    """``_resolution_to_float`` maps an OutOfBoundsResolution just past the
    matching bound (mirrors ``numeric_log_score_from_report``)."""

    def _numeric_question(self, lower: float, upper: float) -> NumericQuestion:
        return NumericQuestion(
            question_text="oob",
            lower_bound=lower,
            upper_bound=upper,
            open_lower_bound=True,
            open_upper_bound=True,
            zero_point=None,
            id_of_question=7777,
        )

    def test_below_lower_bound_maps_to_lower_minus_one(self) -> None:
        q = self._numeric_question(5.0, 20.0)
        assert _resolution_to_float(OutOfBoundsResolution.BELOW_LOWER_BOUND, q) == pytest.approx(4.0)

    def test_above_upper_bound_maps_to_upper_plus_one(self) -> None:
        q = self._numeric_question(5.0, 20.0)
        assert _resolution_to_float(OutOfBoundsResolution.ABOVE_UPPER_BOUND, q) == pytest.approx(21.0)

    def test_in_bounds_float_passes_through(self) -> None:
        q = self._numeric_question(0.0, 100.0)
        assert _resolution_to_float(42.5, q) == pytest.approx(42.5)

    def test_oob_resolution_round_trips_through_loader(self, tmp_path) -> None:
        # End-to-end: an OutOfBoundsResolution serialized in the manifest
        # ground_truth (``_type``-tagged dict, per cli._serialize_resolution)
        # deserializes and maps to upper+1.0 via _build_numeric_record.
        cache = AblationCache(tmp_path / "abl")
        cache.append_qids_manifest(
            {
                88: _replay_manifest_entry(
                    88,
                    "numeric",
                    gt_resolution={"_type": "OutOfBoundsResolution", "value": "ABOVE_UPPER_BOUND"},
                    gt_string="above_upper_bound",
                    extra_metadata=_NUMERIC_BOUNDS_METADATA,
                )
            }
        )
        _seed_n_forecasters(
            cache,
            88,
            [_numeric_replay_payload(_REPLAY_DUMMY_MODELS[i], m) for i, m in enumerate((45.0, 55.0))],
        )
        with no_network():
            dataset = load_replay_dataset(cache)
        assert len(dataset.numeric) == 1
        # upper_bound (100.0) + 1.0
        assert dataset.numeric[0].resolution_value == pytest.approx(101.0)


class TestMinForecastersSkip:
    def test_qid_with_one_survivor_is_dropped(self, tmp_path) -> None:
        # One real forecaster + one failed (prediction_value=None, errors set)
        # -> only 1 survivor < ABLATION_MIN_FORECASTERS (2) -> qid dropped.
        cache = AblationCache(tmp_path / "abl")
        cache.append_qids_manifest({55: _replay_manifest_entry(55, "binary", gt_resolution=True, gt_string="YES")})
        good = _binary_replay_payload(_REPLAY_DUMMY_MODELS[0], 0.5)
        failed = {
            **_binary_replay_payload(_REPLAY_DUMMY_MODELS[1], 0.6),
            "prediction_value": None,
            "errors": ["model failed"],
        }
        _seed_n_forecasters(cache, 55, [good, failed])
        with no_network():
            dataset = load_replay_dataset(cache)
        assert dataset.binary == []
        assert dataset.mc == []
        assert dataset.numeric == []

    def test_two_survivors_are_kept(self, tmp_path) -> None:
        # Exactly at the threshold: 2 survivors -> kept.
        cache = AblationCache(tmp_path / "abl")
        cache.append_qids_manifest({56: _replay_manifest_entry(56, "binary", gt_resolution=False, gt_string="NO")})
        _seed_n_forecasters(
            cache,
            56,
            [_binary_replay_payload(_REPLAY_DUMMY_MODELS[i], p) for i, p in enumerate((0.3, 0.7))],
        )
        with no_network():
            dataset = load_replay_dataset(cache)
        assert len(dataset.binary) == 1
        assert dataset.binary[0].outcome is False


class TestIteratedKFoldCVEdgeCases:
    """F21: CV behavior when there are fewer questions than folds, or none."""

    def test_n_less_than_k_clamps_effective_k(self) -> None:
        # n=3 < k=5: effective_k clamps to n=3, so 3 non-empty folds per
        # iteration -> 3 * iterations resamples, and the summaries stay finite.
        scores = {MEDIAN_BASELINE: np.arange(3.0), "challenger": np.arange(3.0) + 1.0}
        results = iterated_kfold_cv(scores, k=5, iterations=10)
        assert results["challenger"].n_resamples == 30  # min(5, 3) folds * 10 iterations
        assert math.isfinite(results["challenger"].mean_log_score)
        assert math.isfinite(results["challenger"].delta_vs_median_mean)
        # The paired delta of a constant +1 offset is recovered exactly.
        assert results["challenger"].delta_vs_median_mean == pytest.approx(1.0, abs=1e-9)

    def test_n_zero_returns_all_nan_with_zero_resamples(self) -> None:
        scores = {MEDIAN_BASELINE: np.array([]), "challenger": np.array([])}
        results = iterated_kfold_cv(scores, k=5, iterations=10)
        assert set(results) == {MEDIAN_BASELINE, "challenger"}
        for result in results.values():
            assert result.n_resamples == 0
            assert math.isnan(result.full_data_log_score)
            assert math.isnan(result.mean_log_score)
            assert math.isnan(result.std_log_score)
            assert math.isnan(result.delta_vs_median_mean)
            assert math.isnan(result.delta_vs_median_std)
