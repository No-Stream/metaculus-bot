"""Monkey-patch for upstream BoundedQuestionMixin._get_bounds_from_api_json.

The Metaculus API returns range_max/range_min as JSON integers for some
questions (e.g. 200 instead of 200.0). Python's json.loads preserves
int vs float, and the upstream isinstance(x, float) assertions reject ints.
This patch coerces int bounds to float before delegating to the original.

Upstream: forecasting_tools/data_models/questions.py, BoundedQuestionMixin
Remove once upstream adds float() coercion or relaxes the assertions.
"""

import logging

logger: logging.Logger = logging.getLogger(__name__)


def apply_question_patches() -> None:
    """Patch BoundedQuestionMixin._get_bounds_from_api_json to tolerate int bounds."""
    from forecasting_tools.data_models.questions import BoundedQuestionMixin

    _original_func = BoundedQuestionMixin._get_bounds_from_api_json.__func__

    @classmethod  # type: ignore[misc]
    def _patched(cls, api_json: dict) -> tuple[bool, bool, float, float, float | None]:
        scaling = api_json.get("question", {}).get("scaling", {})
        for key in ("range_max", "range_min", "zero_point"):
            val = scaling.get(key)
            if isinstance(val, int):
                scaling[key] = float(val)
        return _original_func(cls, api_json)

    BoundedQuestionMixin._get_bounds_from_api_json = _patched  # type: ignore[assignment]
    logger.info("Patched BoundedQuestionMixin._get_bounds_from_api_json for int→float coercion")
