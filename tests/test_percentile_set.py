"""Tests for the PercentileSet value object.

PercentileSet guarantees percentiles are always accessed by LABEL (e.g. "the P10
value"), never by list position. This structurally prevents the spread-metric
foot-gun where a hardcoded positional index (``[2, 5, 8]`` meaning P10/P50/P90)
silently shifts when the standard percentile set grows from 11 to 13.
"""

from __future__ import annotations

import pytest
from forecasting_tools.data_models.numeric_report import Percentile

from metaculus_bot.numeric.config import STANDARD_PERCENTILES
from metaculus_bot.numeric.percentile_set import PercentileSet

# Standard 11-set values used across tests: value_at(pct) == pct * 1000 so each
# label maps to a distinct, easily-checked value (e.g. P10 -> 100.0).
_STANDARD_VALUES: dict[float, float] = {p: p * 1000.0 for p in STANDARD_PERCENTILES}


def _standard_percentile_list() -> list[Percentile]:
    return [Percentile(percentile=p, value=v) for p, v in _STANDARD_VALUES.items()]


# ===========================================================================
# Construction and round-tripping
# ===========================================================================


class TestConstruction:
    def test_from_percentiles_round_trips(self):
        ps = PercentileSet.from_percentiles(_standard_percentile_list())
        assert ps.value_at(0.10) == pytest.approx(100.0)
        assert ps.value_at(0.50) == pytest.approx(500.0)
        assert ps.value_at(0.975) == pytest.approx(975.0)

    def test_from_mapping_round_trips(self):
        ps = PercentileSet.from_mapping(dict(_STANDARD_VALUES))
        assert ps.value_at(0.10) == pytest.approx(100.0)
        assert ps.value_at(0.90) == pytest.approx(900.0)

    def test_from_percentiles_and_from_mapping_agree(self):
        from_list = PercentileSet.from_percentiles(_standard_percentile_list())
        from_map = PercentileSet.from_mapping(dict(_STANDARD_VALUES))
        assert from_list.values_sorted() == from_map.values_sorted()


# ===========================================================================
# Construction-time validation
# ===========================================================================


class TestValidation:
    def test_wrong_count_raises(self):
        too_few = {p: v for p, v in _STANDARD_VALUES.items() if p != 0.975}
        with pytest.raises(ValueError):
            PercentileSet.from_mapping(too_few)

    def test_missing_p50_raises_naming_it(self):
        missing_median = {p: v for p, v in _STANDARD_VALUES.items() if p != 0.50}
        with pytest.raises(ValueError, match="0.5"):
            PercentileSet.from_mapping(missing_median)

    def test_extra_key_raises_naming_it(self):
        with_extra = dict(_STANDARD_VALUES)
        with_extra[0.11] = 123.0
        with pytest.raises(ValueError, match="0.11"):
            PercentileSet.from_mapping(with_extra)


# ===========================================================================
# Label lookup (the whole point)
# ===========================================================================


class TestLabelLookup:
    def test_value_at_is_insertion_order_independent(self):
        shuffled = _standard_percentile_list()
        # Reverse insertion order to prove lookup is by label, not position.
        ps = PercentileSet.from_percentiles(list(reversed(shuffled)))
        assert ps.value_at(0.10) == pytest.approx(100.0)
        assert ps.value_at(0.50) == pytest.approx(500.0)
        assert ps.value_at(0.90) == pytest.approx(900.0)

    def test_unknown_label_raises_keyerror_not_neighbor(self):
        ps = PercentileSet.from_percentiles(_standard_percentile_list())
        with pytest.raises(KeyError):
            ps.value_at(0.11)
        # Must NOT silently return a neighboring percentile's value.
        try:
            result = ps.value_at(0.11)
        except KeyError:
            result = None
        assert result is None

    def test_float_key_robustness_via_rounding(self):
        ps = PercentileSet.from_percentiles(_standard_percentile_list())
        # 0.1, 0.10, and 0.099999999 all round to the same 6-decimal key.
        assert ps.value_at(0.1) == pytest.approx(100.0)
        assert ps.value_at(0.10) == pytest.approx(100.0)
        assert ps.value_at(0.099999999) == pytest.approx(100.0)


# ===========================================================================
# Ordered / list views
# ===========================================================================


class TestOrderedViews:
    def test_values_sorted_is_ascending_percentile_order(self):
        ps = PercentileSet.from_percentiles(list(reversed(_standard_percentile_list())))
        expected = [v for _, v in sorted(_STANDARD_VALUES.items())]
        assert ps.values_sorted() == expected
        # Because values increase with percentile here, this is also ascending.
        assert ps.values_sorted() == sorted(ps.values_sorted())

    def test_as_percentile_list_round_trips(self):
        ps = PercentileSet.from_percentiles(_standard_percentile_list())
        reconstructed = ps.as_percentile_list()
        assert len(reconstructed) == len(STANDARD_PERCENTILES)
        rebuilt = PercentileSet.from_percentiles(reconstructed)
        assert rebuilt.values_sorted() == ps.values_sorted()
        # Reconstructed list is in ascending percentile order.
        labels = [round(p.percentile, 6) for p in reconstructed]
        assert labels == sorted(labels)

    def test_no_positional_integer_access(self):
        ps = PercentileSet.from_percentiles(_standard_percentile_list())
        with pytest.raises(TypeError):
            _ = ps[2]  # type: ignore[index]


# ===========================================================================
# Regression: documents the positional foot-gun this object prevents
# ===========================================================================


class TestPositionalFootgunRegression:
    def test_label_lookup_correct_for_current_11_set(self):
        ps = PercentileSet.from_percentiles(_standard_percentile_list())
        assert ps.value_at(0.10) == pytest.approx(100.0)
        assert ps.value_at(0.50) == pytest.approx(500.0)
        assert ps.value_at(0.90) == pytest.approx(900.0)

    def test_positional_indices_break_on_a_13_element_set(self):
        # Simulate the future 13-set: P1 and P99 added at the front and back.
        future_labels = [0.01, 0.025, 0.05, 0.10, 0.20, 0.40, 0.50, 0.60, 0.80, 0.90, 0.95, 0.975, 0.99]
        future_list = [Percentile(percentile=p, value=p * 1000.0) for p in future_labels]

        # OLD positional access (indices tuned for the 11-set) now points at the
        # WRONG percentiles once P1/P99 shift everything.
        assert future_list[2].percentile == pytest.approx(0.05)  # meant P10, got P5
        assert future_list[5].percentile == pytest.approx(0.40)  # meant P50, got P40
        assert future_list[8].percentile == pytest.approx(0.80)  # meant P90, got P80

        # LABEL lookup is invariant: P10 lives wherever it lives (position 3 here).
        future_map = {round(p.percentile, 6): p.value for p in future_list}
        assert future_map[0.10] == pytest.approx(future_list[3].value)
        assert future_map[0.50] == pytest.approx(future_list[6].value)
        assert future_map[0.90] == pytest.approx(future_list[9].value)


# ===========================================================================
# Regression: migrated spread_metrics produces identical values
# ===========================================================================


class TestSpreadMetricsUnchanged:
    def test_numeric_percentile_spread_matches_hand_computed(self):
        from forecasting_tools import NumericQuestion

        from metaculus_bot.spread_metrics import numeric_percentile_spread

        std_pcts = [2.5, 5, 10, 20, 40, 50, 60, 80, 90, 95, 97.5]

        def make(values: list[float]) -> list[Percentile]:
            return [Percentile(percentile=p / 100.0, value=v) for p, v in zip(std_pcts, values)]

        # Same fixture as the pre-existing closed-bounds spread test.
        model1 = make([10, 15, 20, 25, 35, 40, 45, 55, 60, 65, 70])
        model2 = make([30, 35, 40, 45, 55, 60, 65, 75, 80, 85, 90])
        question = NumericQuestion(
            question_text="How many?",
            id_of_question=3,
            page_url="https://example.com/q/3",
            background_info="",
            resolution_criteria="",
            fine_print="",
            lower_bound=0.0,
            upper_bound=100.0,
            open_lower_bound=False,
            open_upper_bound=False,
        )

        # P10: |40-20|/100=0.20; P50: |60-40|/100=0.20; P90: |80-60|/100=0.20.
        spread = numeric_percentile_spread([model1, model2], question)
        assert spread == pytest.approx(0.20, abs=1e-9)
