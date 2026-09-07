"""
Unit tests for numeric validation utilities.

Tests for percentile validation and processing functions extracted from main.py.
"""

from types import SimpleNamespace
from typing import cast

import pytest
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import NumericQuestion
from pydantic import ValidationError

from metaculus_bot.numeric.validation import (
    check_discrete_question_properties,
    detect_unit_mismatch,
    sort_by_percentile_level,
    validate_percentile_count_and_values,
)


def _make_question(
    open_upper=False,
    open_lower=False,
    lower=0.0,
    upper=100.0,
    zero_point=None,
    cdf_size=None,
):
    return cast(
        NumericQuestion,
        SimpleNamespace(
            open_upper_bound=open_upper,
            open_lower_bound=open_lower,
            upper_bound=upper,
            lower_bound=lower,
            zero_point=zero_point,
            id_of_question=999,
            cdf_size=cdf_size,
        ),
    )


def _standard_13() -> list[Percentile]:
    """The full standard 13-percentile set with value == percentile * 100."""
    return [
        Percentile(percentile=0.01, value=1.0),
        Percentile(percentile=0.025, value=2.5),
        Percentile(percentile=0.05, value=5.0),
        Percentile(percentile=0.10, value=10.0),
        Percentile(percentile=0.20, value=20.0),
        Percentile(percentile=0.40, value=40.0),
        Percentile(percentile=0.50, value=50.0),
        Percentile(percentile=0.60, value=60.0),
        Percentile(percentile=0.80, value=80.0),
        Percentile(percentile=0.90, value=90.0),
        Percentile(percentile=0.95, value=95.0),
        Percentile(percentile=0.975, value=97.5),
        Percentile(percentile=0.99, value=99.0),
    ]


class TestUnitMismatchFailsShut:
    """``detect_unit_mismatch`` may never report "no mismatch" because it broke.

    It used to wrap its arithmetic in ``except (AttributeError, TypeError,
    ValueError): return False, ""`` — the same answer a passing check gives — so a
    crash inside the guard silently published the order-of-magnitude error the
    guard exists to block. Errors now propagate; every caller treats that the way
    it treats a detected mismatch (drop the forecaster, or fall back to MEDIAN on
    the stacker path).
    """

    def test_unparseable_value_propagates(self):
        question = _make_question()
        broken = cast(
            list[Percentile], [SimpleNamespace(percentile=p.percentile, value="not-a-number") for p in _standard_13()]
        )

        with pytest.raises(ValueError, match="could not convert string to float"):
            detect_unit_mismatch(broken, question)

    def test_missing_value_attribute_propagates(self):
        question = _make_question()
        broken = cast(list[Percentile], [SimpleNamespace(percentile=0.5)])

        with pytest.raises(AttributeError):
            detect_unit_mismatch(broken, question)

    def test_empty_list_is_still_a_mismatch(self):
        """The one honest early return: nothing to judge means withhold."""
        mismatch, reason = detect_unit_mismatch([], _make_question())
        assert mismatch is True
        assert reason == "empty percentile values"

    def test_healthy_declaration_passes(self):
        assert detect_unit_mismatch(_standard_13(), _make_question()) == (False, "")


class TestPercentileValidation:
    """Test percentile validation functions."""

    def test_validate_percentile_count_and_values_success(self):
        """Test successful validation of the correct 13-percentile set."""
        # Should not raise any exception
        validate_percentile_count_and_values(_standard_13())

    def test_validate_percentile_count_wrong_count(self):
        """Test validation fails with wrong number of percentiles (too few)."""
        percentiles = [
            Percentile(percentile=0.025, value=2.5),
            Percentile(percentile=0.05, value=5.0),
            Percentile(percentile=0.10, value=10.0),
        ]

        with pytest.raises(ValidationError) as exc_info:
            validate_percentile_count_and_values(percentiles)

        assert "Expected 13 declared percentiles" in str(exc_info.value)

    def test_validate_percentile_rejects_old_11_set(self):
        """The old 11-set (no P1/P99) must now fail — it's missing two labels."""
        old_11 = [p for p in _standard_13() if p.percentile not in (0.01, 0.99)]
        assert len(old_11) == 11
        with pytest.raises(ValidationError) as exc_info:
            validate_percentile_count_and_values(old_11)
        assert "Expected 13 declared percentiles" in str(exc_info.value)

    def test_validate_percentile_rejects_12_set(self):
        """A 12-set (missing just P99) must fail, naming the expected count/set."""
        twelve = [p for p in _standard_13() if p.percentile != 0.99]
        assert len(twelve) == 12
        with pytest.raises(ValidationError) as exc_info:
            validate_percentile_count_and_values(twelve)
        msg = str(exc_info.value)
        assert "Expected 13 declared percentiles" in msg or "Expected percentile set" in msg

    def test_validate_percentile_wrong_values(self):
        """Test validation fails with wrong percentile values (right count, wrong label)."""
        percentiles = _standard_13()
        percentiles[0] = Percentile(percentile=0.03, value=3.0)  # Wrong (should be 0.01)

        with pytest.raises(ValidationError) as exc_info:
            validate_percentile_count_and_values(percentiles)

        assert "Expected percentile set" in str(exc_info.value)

    def test_count_error_names_full_13_label_set(self):
        """The count-error parenthetical must list the full 13 labels (incl. 1 and 99),
        generated from STANDARD_PERCENTILES — not the stale hardcoded 11-list."""
        with pytest.raises(ValidationError) as exc_info:
            validate_percentile_count_and_values([Percentile(percentile=0.5, value=1.0)])
        msg = str(exc_info.value)
        assert "1,2.5,5,10,20,40,50,60,80,90,95,97.5,99" in msg

    def test_wrong_value_error_names_full_13_label_set(self):
        """The wrong-set error must name the full 13-label expected set (incl. 1 and 99)."""
        percentiles = _standard_13()
        percentiles[0] = Percentile(percentile=0.03, value=3.0)
        with pytest.raises(ValidationError) as exc_info:
            validate_percentile_count_and_values(percentiles)
        msg = str(exc_info.value)
        assert "1,2.5,5,10,20,40,50,60,80,90,95,97.5,99" in msg

    def test_sort_by_percentile_level(self):
        """Test sorting declarations by percentile level."""
        # Create unsorted percentiles
        percentiles = [
            Percentile(percentile=0.90, value=90.0),
            Percentile(percentile=0.05, value=5.0),
            Percentile(percentile=0.60, value=60.0),
            Percentile(percentile=0.20, value=20.0),
        ]

        sorted_percentiles = sort_by_percentile_level(percentiles)

        # Check they are sorted by percentile level
        expected_order = [0.05, 0.20, 0.60, 0.90]
        actual_order = [p.percentile for p in sorted_percentiles]

        assert actual_order == expected_order

    def test_check_discrete_question_properties_discrete(self):
        """Test discrete question detection."""
        question = _make_question(cdf_size=100)  # Not 201, so discrete
        question.zero_point = 1.0

        is_discrete, should_force_none = check_discrete_question_properties(question, 201)

        assert is_discrete is True
        assert should_force_none is True

    def test_check_discrete_question_properties_continuous(self):
        """Test continuous question detection."""
        question = _make_question(cdf_size=201)  # 201, so continuous
        question.zero_point = None

        is_discrete, should_force_none = check_discrete_question_properties(question, 201)

        assert is_discrete is False
        assert should_force_none is False

    def test_check_discrete_question_properties_zero_point_equals_lower_bound(self):
        """Test zero_point equals lower_bound case."""
        question = _make_question(lower=0.0, cdf_size=201)
        question.zero_point = 0.0  # Same as lower_bound

        is_discrete, should_force_none = check_discrete_question_properties(question, 201)

        assert is_discrete is False
        assert should_force_none is True  # Should force zero_point to None

    def test_check_discrete_question_properties_no_cdf_size(self):
        """Test question with no cdf_size attribute."""
        question = _make_question()
        # Don't set cdf_size attribute at all
        if hasattr(question, "cdf_size"):
            delattr(question, "cdf_size")
        question.zero_point = None

        is_discrete, should_force_none = check_discrete_question_properties(question, 201)

        assert is_discrete is False
        assert should_force_none is False
