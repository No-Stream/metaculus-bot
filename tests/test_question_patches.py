"""Regression tests for the numeric-bounds monkey-patch.

The Metaculus API can return scaling values (range_max/range_min/zero_point) as
JSON integers (e.g. 200 instead of 200.0). On forecasting-tools 0.2.92 the
upstream ``_get_bounds_from_api_json`` float-casts range_max/range_min itself but
returns zero_point raw, violating its own ``float | None`` return annotation.
W6 narrowed our patch in metaculus_bot.question_patches to coerce only that
still-raw zero_point, so the returned tuple honors its declared contract for any
direct consumer of the classmethod.
"""

from forecasting_tools.data_models.questions import BoundedQuestionMixin


def _make_scaling_json(
    *,
    range_max: int | float = 200.0,
    range_min: int | float = 0.0,
    zero_point: int | float | None = None,
    open_upper_bound: bool = True,
    open_lower_bound: bool = True,
) -> dict:
    """Build minimal API JSON dict for _get_bounds_from_api_json."""
    return {
        "question": {
            "open_upper_bound": open_upper_bound,
            "open_lower_bound": open_lower_bound,
            "scaling": {
                "range_max": range_max,
                "range_min": range_min,
                "zero_point": zero_point,
            },
        },
    }


class TestGetBoundsIntCoercion:
    def test_int_bounds_coerced_to_float(self):
        result = BoundedQuestionMixin._get_bounds_from_api_json(_make_scaling_json(range_max=200, range_min=0))
        _, _, upper, lower, _ = result
        assert upper == 200.0 and isinstance(upper, float)
        assert lower == 0.0 and isinstance(lower, float)

    def test_float_bounds_still_work(self):
        result = BoundedQuestionMixin._get_bounds_from_api_json(_make_scaling_json(range_max=200.0, range_min=0.0))
        _, _, upper, lower, _ = result
        assert upper == 200.0 and isinstance(upper, float)
        assert lower == 0.0 and isinstance(lower, float)

    def test_int_zero_point_coerced(self):
        result = BoundedQuestionMixin._get_bounds_from_api_json(_make_scaling_json(zero_point=100))
        _, _, _, _, zp = result
        assert zp == 100.0 and isinstance(zp, float)

    def test_none_zero_point_preserved(self):
        result = BoundedQuestionMixin._get_bounds_from_api_json(_make_scaling_json(zero_point=None))
        _, _, _, _, zp = result
        assert zp is None

    def test_mixed_int_float_bounds(self):
        result = BoundedQuestionMixin._get_bounds_from_api_json(
            _make_scaling_json(range_max=200, range_min=0.5, zero_point=1)
        )
        _, _, upper, lower, zp = result
        assert upper == 200.0 and isinstance(upper, float)
        assert lower == 0.5 and isinstance(lower, float)
        assert zp == 1.0 and isinstance(zp, float)
