from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict


class StagePayload(TypedDict, total=False):
    success: bool
    arm: str
    reason: str
    stacker_prediction: Any
    stacker_meta_reasoning: str
    computed_quantities: dict
    cross_model_aggregation: str | None
    stacker_model_used: str | None
    n_forecasters_used: int
    ran_at: str
    tools_enabled_at_runtime: bool
    errors: list


def make_success_payload(
    *,
    arm: str,
    prediction: Any,
    meta_reasoning: str = "",
    computed_quantities: dict | None = None,
    cross_model_aggregation: str | None = "",
    model_used: str | None,
    n_forecasters: int,
    tools_enabled: bool = False,
    errors: list | None = None,
) -> dict[str, Any]:
    return {
        "success": True,
        "arm": arm,
        "stacker_prediction": prediction,
        "stacker_meta_reasoning": meta_reasoning,
        "computed_quantities": computed_quantities or {},
        "cross_model_aggregation": cross_model_aggregation,
        "stacker_model_used": model_used,
        "n_forecasters_used": n_forecasters,
        "ran_at": datetime.now().isoformat(),
        "tools_enabled_at_runtime": tools_enabled,
        "errors": errors or [],
    }


def make_error_payload(
    *,
    arm: str,
    reason: str,
    prediction: Any = None,
    meta_reasoning: str = "",
    computed_quantities: dict | None = None,
    cross_model_aggregation: str | None = "",
    model_used: str | None,
    n_forecasters: int,
    tools_enabled: bool = False,
    errors: list | None = None,
) -> dict[str, Any]:
    return {
        "success": False,
        "arm": arm,
        "reason": reason,
        "stacker_prediction": prediction,
        "stacker_meta_reasoning": meta_reasoning,
        "computed_quantities": computed_quantities or {},
        "cross_model_aggregation": cross_model_aggregation,
        "stacker_model_used": model_used,
        "n_forecasters_used": n_forecasters,
        "ran_at": datetime.now().isoformat(),
        "tools_enabled_at_runtime": tools_enabled,
        "errors": errors or [],
    }
