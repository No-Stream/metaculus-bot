"""Backward-compatibility shim — canonical location is run_simple_agg.py."""

from metaculus_bot.ablation.run_simple_agg import (  # noqa: F401
    SIMPLE_AGGREGATION_LABEL,
    run_median_for_qid,
)

__all__ = ["run_median_for_qid", "SIMPLE_AGGREGATION_LABEL"]
