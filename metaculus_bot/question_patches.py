"""Monkey-patch for upstream BoundedQuestionMixin._get_bounds_from_api_json.

Narrowed for forecasting-tools 0.2.92. Upstream now float-casts range_max /
range_min itself (forecasting_tools/data_models/questions.py), so the old
int→float coercion of the *bounds* is redundant and has been dropped. What
upstream still does NOT coerce is ``zero_point``: for an integer JSON zero_point
it returns the raw int, violating the method's own
``tuple[bool, bool, float, float, float | None]`` return annotation. Downstream
Pydantic model construction (NumericQuestion / DateQuestion /
NumericTimestampedDistribution, all with a ``zero_point: float | None`` field)
coerces int→float, so this is not currently load-bearing — but we keep the
narrowed coercion so the returned tuple honors its declared contract for any
direct consumer of the classmethod.

Import-order note: importing this module imports forecasting_tools, and so litellm.
``metaculus_bot/__init__`` must therefore keep its ``DISABLE_AIOHTTP_TRANSPORT``
setdefault ABOVE the ``from metaculus_bot.question_patches import ...`` line (it does,
with a comment saying so). The mechanism is not an import-time read: litellm re-reads
the variable on every transport construction, but the handlers it builds during its own
import freeze onto the aiohttp transport if the default arrives late.
``tests/test_aiohttp_transport_flag.py`` asserts the source order.

Upstream: forecasting_tools/data_models/questions.py, BoundedQuestionMixin.
Follow-on: full retirement is viable (verified — Pydantic coerces zero_point
downstream, so NumericQuestion.from_metaculus_api_json builds fine from int
scaling without this patch). Drop the patch and the apply line in
metaculus_bot/__init__.py once no consumer reads the raw tuple's zero_point.
"""

import logging

from forecasting_tools.data_models.questions import BoundedQuestionMixin

logger: logging.Logger = logging.getLogger(__name__)


def apply_question_patches() -> None:
    """Patch BoundedQuestionMixin._get_bounds_from_api_json to float-coerce zero_point.

    Upstream 0.2.92 already float-casts range_max/range_min, so this narrowed
    patch coerces only the still-raw zero_point slot before delegating.
    """
    # Idempotency guard: metaculus_bot import applies this once, but a module
    # reload (importlib.reload, which some tests do) re-invokes it. Without the
    # guard the second call captures the already-installed _patched as its
    # _original_func, nesting wrappers and corrupting the closure chain that
    # callers — and the upgrade seam test — read to recover the pristine upstream.
    if getattr(BoundedQuestionMixin._get_bounds_from_api_json.__func__, "_zero_point_patch_installed", False):
        return

    _original_func = BoundedQuestionMixin._get_bounds_from_api_json.__func__

    def _patched(cls, api_json: dict) -> tuple[bool, bool, float, float, float | None]:
        scaling = api_json.get("question", {}).get("scaling", {})
        zero_point = scaling.get("zero_point")
        if isinstance(zero_point, int):
            scaling["zero_point"] = float(zero_point)
        return _original_func(cls, api_json)

    setattr(_patched, "_zero_point_patch_installed", True)  # noqa: B010  # re-patch guard marker

    # Monkey-patch: reattach the classmethod descriptor. setattr keeps ty from
    # nominally comparing the two method types (the # type: ignore covers pyright).
    setattr(BoundedQuestionMixin, "_get_bounds_from_api_json", classmethod(_patched))  # type: ignore[assignment]  # noqa: B010
    logger.info("Patched BoundedQuestionMixin._get_bounds_from_api_json for zero_point int→float coercion")
