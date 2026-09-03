"""Tests for metaculus_bot.research.number_format, the fixed-point rule both the FRED block
and the inline-chart rung render through.

The module has two consumers with disjoint dependencies (``fred_rendering`` pulls pandas and
fredapi; ``resolution_chart_data`` is stdlib-only), which is why the rule lives on its own and
why its contract is asserted here rather than inside either consumer's suite. The interesting
inputs are not FRED decimals: float subtraction noise, a magnitude that rounds away at this
precision, a true zero, and a negative tiny value whose sign is not information.
"""

from __future__ import annotations

import pytest

from metaculus_bot.research.number_format import (
    DECIMAL_PLACES,
    format_decimal_change,
    format_decimal_value,
)


class TestFormatDecimalValue:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(331.893, "331.893", id="index_level_keeps_its_digits"),
            pytest.param(4.2, "4.2", id="rate_loses_its_trailing_zeros"),
            pytest.param(20.0, "20", id="integer_valued_float_renders_bare"),
            pytest.param(0.0, "0", id="true_zero"),
            pytest.param(331.893 - 331.020, "0.873", id="float_subtraction_noise_is_cleaned_up"),
            pytest.param(0.000001, "0.000001", id="smallest_representable_magnitude"),
        ],
    )
    def test_fixed_point_at_published_precision(self, value: float, expected: str) -> None:
        assert format_decimal_value(value) == expected

    @pytest.mark.parametrize("value", [6_699_580.0, -6_699_580.0, 1e12])
    def test_a_large_magnitude_never_goes_scientific(self, value: float) -> None:
        """The Fed balance sheet rendered as "6.7e+06" under the retired ``:.4g``, which is
        unreadable as a level and unusable as a forecast anchor."""
        rendered = format_decimal_value(value)
        assert "e" not in rendered.lower()
        assert float(rendered) == pytest.approx(value)

    @pytest.mark.parametrize("value", [-1e-7, -1e-9, -0.0, -1e-300])
    def test_a_magnitude_that_rounds_away_never_renders_minus_zero(self, value: float) -> None:
        """The guard the module documents: the sign of a quantity too small to show at this
        precision is not information, and "-0" reads as a real decline. This is the branch the
        FRED suite could not reach, since it only ever calls the value formatter through
        ``format_decimal_change``, which passes an absolute value."""
        assert format_decimal_value(value) == "0"

    def test_negative_values_above_the_precision_floor_keep_their_sign(self) -> None:
        assert format_decimal_value(-331.893) == "-331.893"

    def test_precision_is_the_module_constant(self) -> None:
        """Six decimals covers everything FRED publishes; a seventh is dropped rather than
        rounded into view."""
        assert DECIMAL_PLACES == 6
        assert format_decimal_value(1.23456789) == "1.234568"


class TestFormatDecimalChange:
    @pytest.mark.parametrize(
        ("change", "expected"),
        [
            pytest.param(0.05, "+0.05", id="rise"),
            pytest.param(-0.749, "-0.749", id="fall"),
            pytest.param(0.0, "+0", id="true_zero_is_positive"),
            pytest.param(-0.0, "+0", id="negative_zero_is_positive"),
            pytest.param(-1e-9, "+0", id="rounds_away_to_positive_zero"),
        ],
    )
    def test_sign_and_magnitude(self, change: float, expected: str) -> None:
        assert format_decimal_change(change) == expected

    def test_the_sign_is_carried_separately_from_the_magnitude(self) -> None:
        """``:+`` cannot drive a custom formatter, so the sign is prepended. A rendered change
        must therefore agree with the value formatter on the magnitude."""
        assert format_decimal_change(-331.893) == f"-{format_decimal_value(331.893)}"
