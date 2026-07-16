"""Tests for the weighted-quantile aggregation primitives added for the coherence-weighting study.

The load-bearing property is operator nesting: the weighted operator must reduce to the
unweighted median combine when weights are equal, so any coherence arm's offset from the
median baseline is attributable to the WEIGHTS, not to a different operator. On even-M
questions the reduction is to machine precision (``np.median`` forms ``(a+b)/2`` while the
type-7 interpolation forms ``a + 0.5*(b-a)`` — a last-ULP difference), so equal-weight tests
assert closeness at <2e-15 rather than literal bit-identity; the empty-weight fallback path,
which routes to the exact median code, is bit-identical.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion

from metaculus_bot.ablation.offline_replay import (
    COHERENCE_SOFTWEIGHT,
    MEDIAN_BASELINE,
    BinaryRecord,
    MCRecord,
    NumericRecord,
    build_binary_configs,
    build_mc_configs,
    build_numeric_configs,
    weighted_cdf_median,
    weighted_quantile,
)

# Machine-precision tolerance for even-M median nesting (see module docstring).
NEST_TOL = 2e-15


class TestWeightedQuantile:
    def test_equal_weights_match_np_median(self) -> None:
        """Type-7 weighted quantile at equal weights reproduces np.median to machine precision."""
        rng = np.random.default_rng(0)
        max_diff = 0.0
        for n in (2, 3, 4, 5, 6, 7):
            for _ in range(3000):
                x = rng.normal(size=n)
                wq = weighted_quantile(x, np.ones(n), 0.5)
                max_diff = max(max_diff, abs(wq - float(np.median(x))))
        assert max_diff < NEST_TOL, f"equal-weight quantile diverged from np.median by {max_diff}"

    def test_equal_weights_match_np_quantile_hazen_offmedian(self) -> None:
        """At equal weights the Hazen positions nest np.quantile(method='hazen') at any q."""
        rng = np.random.default_rng(1)
        for _ in range(2000):
            x = rng.normal(size=6)
            for q in (0.1, 0.25, 0.75, 0.9):
                wq = weighted_quantile(x, np.ones(6), q)
                ref = float(np.quantile(x, q, method="hazen"))
                assert abs(wq - ref) < 1e-12

    def test_weight_shifts_quantile_toward_heavy_model(self) -> None:
        """Up-weighting one value pulls the 0.5 quantile toward it (symmetric response)."""
        x = np.array([0.0, 1.0, 10.0])
        base = weighted_quantile(x, np.ones(3), 0.5)
        heavy_high = weighted_quantile(x, np.array([1.0, 1.0, 5.0]), 0.5)
        heavy_low = weighted_quantile(x, np.array([5.0, 1.0, 1.0]), 0.5)
        # Hazen midpoint positions respond symmetrically: heavier top/bottom weight
        # moves the median strictly up/down (the type-7 form was insensitive at the top).
        assert heavy_low < base < heavy_high

    def test_all_weight_on_largest_returns_it(self) -> None:
        x = np.array([1.0, 2.0, 9.0])
        assert weighted_quantile(x, np.array([0.0, 0.0, 1.0]), 0.5) == pytest.approx(9.0)

    def test_negative_and_zero_total_weights_raise(self) -> None:
        with pytest.raises(ValueError):
            weighted_quantile([1.0, 2.0], [-1.0, 1.0], 0.5)
        with pytest.raises(ValueError):
            weighted_quantile([1.0, 2.0], [0.0, 0.0], 0.5)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            weighted_quantile([1.0, 2.0, 3.0], [1.0, 1.0], 0.5)


class TestWeightedCdfMedian:
    def _random_cdf_matrix(self, rng: np.random.Generator, n_models: int, n_grid: int = 201) -> np.ndarray:
        """A monotone-in-columns CDF-prob matrix like the per-model 201-pt CDFs."""
        base = np.sort(rng.uniform(0.0, 1.0, size=(n_models, n_grid)), axis=1)
        base[:, 0] = 0.0
        base[:, -1] = 1.0
        return base

    def test_equal_weights_match_vertical_median(self) -> None:
        """Weighted CDF median at equal weights == np.median column-median (post clip+accumulate)."""
        rng = np.random.default_rng(2)
        max_diff = 0.0
        for n_models in (3, 4, 5, 6):
            mat = self._random_cdf_matrix(rng, n_models)
            weighted = np.array(weighted_cdf_median(mat, np.ones(n_models)))
            ref = np.maximum.accumulate(np.clip(np.median(mat, axis=0), 0.0, 1.0))
            max_diff = max(max_diff, float(np.max(np.abs(weighted - ref))))
        assert max_diff < NEST_TOL, f"weighted CDF median diverged from vertical median by {max_diff}"

    def test_none_weights_equal_explicit_equal(self) -> None:
        rng = np.random.default_rng(3)
        mat = self._random_cdf_matrix(rng, 5)
        assert weighted_cdf_median(mat, None) == weighted_cdf_median(mat, np.ones(5))

    def test_output_monotone_and_bounded(self) -> None:
        rng = np.random.default_rng(4)
        mat = self._random_cdf_matrix(rng, 6)
        out = np.array(weighted_cdf_median(mat, np.array([0.4, 0.1, 0.1, 0.1, 0.1, 0.2])))
        assert np.all(np.diff(out) >= -1e-12)
        assert out.min() >= 0.0 and out.max() <= 1.0


# --- Config-level nesting (the arm-vs-baseline guarantee) --------------------


def _binary_record(p_models: list[float], models: tuple[str, ...]) -> BinaryRecord:
    # The binary median/weighted path never reads ``question``; a placeholder keeps the
    # test free of a heavy forecasting-tools question construction.
    return BinaryRecord(
        qid=1,
        question=cast(BinaryQuestion, object()),
        outcome=True,
        p_models=p_models,
        p_maths=[],
        models=models,
    )


def _mc_record(vectors: list[dict[str, float]], order: list[str], models: tuple[str, ...]) -> MCRecord:
    return MCRecord(
        qid=1,
        question=cast(MultipleChoiceQuestion, object()),
        option_order=order,
        correct_option_index=0,
        option_vectors=vectors,
        models=models,
    )


def _numeric_record(cdfs_probs: list[np.ndarray], models: tuple[str, ...]) -> NumericRecord:
    grid = np.linspace(0.0, 1.0, cdfs_probs[0].size)
    cdfs = [[Percentile(value=float(x), percentile=float(p)) for x, p in zip(grid, probs)] for probs in cdfs_probs]
    return NumericRecord(
        qid=1,
        question=cast(NumericQuestion, object()),
        resolution_value=0.5,
        cdfs=cdfs,
        models=models,
    )


class TestConfigNesting:
    def test_binary_coherence_equal_weights_matches_baseline(self) -> None:
        models = ("m1", "m2", "m3", "m4", "m5", "m6")
        rec = _binary_record([0.1, 0.3, 0.4, 0.5, 0.7, 0.9], models)
        equal = {(1, m): 1.0 for m in models}
        configs = build_binary_configs(weights_by_qid_model=equal)
        assert COHERENCE_SOFTWEIGHT in configs
        assert abs(configs[COHERENCE_SOFTWEIGHT](rec) - configs[MEDIAN_BASELINE](rec)) < NEST_TOL

    def test_binary_missing_weights_fall_back_to_exact_median(self) -> None:
        models = ("m1", "m2", "m3")
        rec = _binary_record([0.2, 0.5, 0.8], models)
        empty: dict[tuple[int, str], float] = {}
        cfg = build_binary_configs(weights_by_qid_model=empty)[COHERENCE_SOFTWEIGHT]
        # No mapped weights → exact median code path → bit-identical.
        assert cfg(rec) == build_binary_configs()[MEDIAN_BASELINE](rec)

    def test_binary_no_weights_omits_coherence_arm(self) -> None:
        assert COHERENCE_SOFTWEIGHT not in build_binary_configs()

    def test_mc_coherence_equal_weights_matches_baseline(self) -> None:
        order = ["a", "b", "c"]
        models = ("m1", "m2", "m3", "m4", "m5")
        rng = np.random.default_rng(5)
        vectors = []
        for _ in range(5):
            v = rng.uniform(0.05, 1.0, size=3)
            v = v / v.sum()
            vectors.append({o: float(p) for o, p in zip(order, v)})
        rec = _mc_record(vectors, order, models)
        equal = {(1, m): 1.0 for m in models}
        base = build_mc_configs()[MEDIAN_BASELINE](rec)
        coh = build_mc_configs(weights_by_qid_model=equal)[COHERENCE_SOFTWEIGHT](rec)
        assert np.max(np.abs(np.array(base) - np.array(coh))) < NEST_TOL

    def test_numeric_coherence_equal_weights_matches_baseline(self) -> None:
        rng = np.random.default_rng(6)
        probs = []
        for _ in range(6):
            c = np.sort(rng.uniform(0.0, 1.0, size=201))
            c[0], c[-1] = 0.0, 1.0
            probs.append(c)
        models = ("m1", "m2", "m3", "m4", "m5", "m6")
        rec = _numeric_record(probs, models)
        equal = {(1, m): 1.0 for m in models}
        base = np.array(build_numeric_configs()[MEDIAN_BASELINE](rec))
        coh = np.array(build_numeric_configs(weights_by_qid_model=equal)[COHERENCE_SOFTWEIGHT](rec))
        assert np.max(np.abs(base - coh)) < NEST_TOL

    def test_numeric_weighted_differs_from_median_under_skewed_weights(self) -> None:
        rng = np.random.default_rng(7)
        probs = []
        for _ in range(6):
            c = np.sort(rng.uniform(0.0, 1.0, size=201))
            c[0], c[-1] = 0.0, 1.0
            probs.append(c)
        models = ("m1", "m2", "m3", "m4", "m5", "m6")
        rec = _numeric_record(probs, models)
        skewed = {(1, m): (5.0 if i == 0 else 1.0) for i, m in enumerate(models)}
        base = np.array(build_numeric_configs()[MEDIAN_BASELINE](rec))
        coh = np.array(build_numeric_configs(weights_by_qid_model=skewed)[COHERENCE_SOFTWEIGHT](rec))
        assert np.max(np.abs(base - coh)) > 1e-6  # weights actually move the combine
