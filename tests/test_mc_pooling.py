from __future__ import annotations

import math

import numpy as np
import pytest

from metaculus_bot.probabilistic_tools.mc_pooling import pool_mc


def _elementwise_median(vectors: list[dict[str, float]]) -> dict[str, float]:
    """Reference: the (buggy) no-op aggregation we are replacing — per-option median."""
    keys = list(vectors[0].keys())
    return {k: float(np.median([v[k] for v in vectors])) for k in keys}


class TestPoolMcGeometric:
    def test_sums_to_one(self):
        result = pool_mc([{"A": 0.6, "B": 0.3, "C": 0.1}, {"A": 0.2, "B": 0.5, "C": 0.3}])
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)

    def test_preserves_option_keys(self):
        result = pool_mc([{"A": 0.8, "B": 0.2}, {"A": 0.2, "B": 0.8}])
        assert set(result.keys()) == {"A", "B"}

    def test_geometric_mean_of_split_vote(self):
        # Symmetric split: geometric pool of A = sqrt(0.8*0.2)=sqrt(0.16)=0.4,
        # B likewise 0.4; normalized -> 0.5/0.5.
        result = pool_mc([{"A": 0.8, "B": 0.2}, {"A": 0.2, "B": 0.8}])
        z = math.sqrt(0.16) + math.sqrt(0.16)
        assert result["A"] == pytest.approx(math.sqrt(0.16) / z)
        assert result["A"] == pytest.approx(0.5)
        assert result["B"] == pytest.approx(0.5)

    def test_geometric_differs_from_arithmetic_on_asymmetric_vote(self):
        # Arithmetic mean: A=0.7, B=0.3. Geometric: A=sqrt(0.45)~0.6708,
        # B=sqrt(0.05)~0.2236; normalized A~0.75, B~0.25. Geometric punishes
        # the confident-low (0.1) B vote harder, pulling A higher than the mean.
        vectors = [{"A": 0.9, "B": 0.1}, {"A": 0.5, "B": 0.5}]
        geo = pool_mc(vectors)
        a_geom = math.sqrt(0.9 * 0.5)
        b_geom = math.sqrt(0.1 * 0.5)
        z = a_geom + b_geom
        assert geo["A"] == pytest.approx(a_geom / z)
        assert geo["A"] > 0.7  # strictly above the arithmetic mean of 0.7
        assert geo["A"] == pytest.approx(0.75, abs=1e-3)

    def test_unanimous_vote_is_identity(self):
        # If every forecaster gives the identical vector, pooling returns it unchanged.
        vec = {"A": 0.5, "B": 0.3, "C": 0.2}
        result = pool_mc([vec, vec, vec])
        for k, v in vec.items():
            assert result[k] == pytest.approx(v)


class TestPoolMcNotMedianNoOp:
    def test_pool_is_not_elementwise_median(self):
        # Regression test for the no-op bug: the old MC pdf path copied option_probs
        # then median-aggregated, so "pdf MC" == "median MC". A real pool must differ.
        vectors = [
            {"A": 0.7, "B": 0.2, "C": 0.1},
            {"A": 0.2, "B": 0.7, "C": 0.1},
            {"A": 0.1, "B": 0.1, "C": 0.8},
        ]
        pooled = pool_mc(vectors)
        median = _elementwise_median(vectors)
        # At least one option must differ materially from the elementwise median.
        max_abs_diff = max(abs(pooled[k] - median[k]) for k in pooled)
        assert max_abs_diff > 1e-3, f"pool_mc collapsed to elementwise median: {pooled=} {median=}"


class TestPoolMcZeroHandling:
    def test_near_zero_option_no_inf_or_nan(self):
        # A forecaster assigning ~0 to an option must not produce inf/nan via log(0).
        vectors = [{"A": 0.0, "B": 1.0}, {"A": 0.4, "B": 0.6}]
        result = pool_mc(vectors)
        assert all(math.isfinite(v) for v in result.values())
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)
        assert result["A"] > 0.0  # floored away from zero, then re-weighted

    def test_exact_zero_in_every_vector(self):
        vectors = [{"A": 0.0, "B": 0.5, "C": 0.5}, {"A": 0.0, "B": 0.6, "C": 0.4}]
        result = pool_mc(vectors)
        assert all(math.isfinite(v) for v in result.values())
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)
        # A pooled near the floor — tiny but positive.
        assert 0.0 < result["A"] < 0.01


class TestPoolMcDirichletSmoothing:
    def test_smoothing_pulls_toward_uniform_as_concentration_drops(self):
        # The pooled vector is non-uniform; lower concentration => closer to uniform (1/3 each).
        vectors = [{"A": 0.7, "B": 0.2, "C": 0.1}, {"A": 0.6, "B": 0.3, "C": 0.1}]
        n_options = 3
        uniform = 1.0 / n_options

        no_smooth = pool_mc(vectors, concentration=None)
        high_conc = pool_mc(vectors, concentration=50.0)
        low_conc = pool_mc(vectors, concentration=2.0)

        dist_no = abs(no_smooth["A"] - uniform)
        dist_high = abs(high_conc["A"] - uniform)
        dist_low = abs(low_conc["A"] - uniform)

        # More smoothing (lower concentration) moves the dominant option closer to uniform.
        assert dist_low < dist_high < dist_no

    def test_smoothing_output_still_normalized(self):
        result = pool_mc([{"A": 0.8, "B": 0.15, "C": 0.05}], concentration=5.0)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-9)


class TestPoolMcValidation:
    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="non-empty"):
            pool_mc([])

    def test_rejects_mismatched_keys(self):
        with pytest.raises(ValueError, match="keys"):
            pool_mc([{"A": 0.5, "B": 0.5}, {"A": 0.3, "C": 0.7}])

    def test_rejects_out_of_range_probability(self):
        with pytest.raises(ValueError):
            pool_mc([{"A": 1.5, "B": -0.5}])

    def test_rejects_non_normalized_vector(self):
        with pytest.raises(ValueError, match="sum"):
            pool_mc([{"A": 0.2, "B": 0.2}])

    def test_rejects_non_positive_concentration(self):
        with pytest.raises(ValueError):
            pool_mc([{"A": 0.5, "B": 0.5}], concentration=0.0)
