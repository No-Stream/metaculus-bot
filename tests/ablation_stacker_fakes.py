"""Shared fakes for the ablation stacker-runner test modules.

Question factories, forecaster-payload factories and the coroutine driver shared by every
``tests/ablation/test_ablation_run_stacker*`` module. Split out of
``test_ablation_run_stacker.py`` (3502 lines) so the per-aspect modules share one copy of
the factories rather than redefining them. Holds no fixtures — those live in
``tests/ablation/conftest.py``, which covers the whole directory.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

from forecasting_tools import (
    BinaryQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

from metaculus_bot.ablation.cache import model_slug_to_filename

FEATURE_FLAG = "PROBABILISTIC_TOOLS_ENABLED"

_OPEN = datetime(2026, 1, 1)
_RESOLVE = datetime(2026, 5, 1)

# ---------------------------------------------------------------------------
# Question factories
# ---------------------------------------------------------------------------


def _make_binary_q(qid: int = 1) -> BinaryQuestion:
    q = MagicMock(spec=BinaryQuestion)
    q.id_of_question = qid
    q.question_text = "Will it rain?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = f"https://example.com/q/{qid}"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


def _make_mc_q(qid: int = 2) -> MultipleChoiceQuestion:
    q = MagicMock(spec=MultipleChoiceQuestion)
    q.id_of_question = qid
    q.question_text = "Which color?"
    q.options = ["Red", "Blue"]
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = f"https://example.com/q/{qid}"
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


def _make_numeric_q(qid: int = 3) -> NumericQuestion:
    q = MagicMock(spec=NumericQuestion)
    q.id_of_question = qid
    q.question_text = "What will X be?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = ""
    q.page_url = f"https://example.com/q/{qid}"
    q.unit_of_measure = "USD"
    q.lower_bound = 0.0
    q.upper_bound = 100.0
    q.open_lower_bound = False
    q.open_upper_bound = False
    q.nominal_lower_bound = None
    q.nominal_upper_bound = None
    q.zero_point = None
    q.cdf_size = 201
    q.open_time = _OPEN
    q.scheduled_resolution_time = _RESOLVE
    return q


# ---------------------------------------------------------------------------
# Forecaster payload factories
# ---------------------------------------------------------------------------


def _binary_payload(model: str = "openrouter/test/m1", value: float = 0.6) -> dict:
    return {
        "prediction_value": {"type": "binary", "prob": value},
        "reasoning": f"Model: {model}\n\nrationale text from {model}",
        "errors": [],
        "model": model,
    }


def _numeric_payload(model: str = "openrouter/test/m1", median: float = 50.0) -> dict:
    """Build a numeric forecaster payload in the post-Bucket-1 full-CDF schema.

    Schema is what ``serialize_prediction_value`` emits for a real
    ``NumericDistribution``: declared_percentiles + the constraint-enforced
    201-point CDF + bounds + zero_point + cdf_size. Tests assemble payloads
    directly here instead of running the serializer, so we synthesize a
    monotone linear CDF that spans the bounds.
    """
    declared = [
        {"percentile": 0.01, "value": median - 35},
        {"percentile": 0.025, "value": median - 30},
        {"percentile": 0.05, "value": median - 25},
        {"percentile": 0.10, "value": median - 20},
        {"percentile": 0.20, "value": median - 12},
        {"percentile": 0.40, "value": median - 5},
        {"percentile": 0.50, "value": median},
        {"percentile": 0.60, "value": median + 5},
        {"percentile": 0.80, "value": median + 12},
        {"percentile": 0.90, "value": median + 20},
        {"percentile": 0.95, "value": median + 25},
        {"percentile": 0.975, "value": median + 30},
        {"percentile": 0.99, "value": median + 35},
    ]
    cdf_probabilities = [0.001 + (0.998 * i / 200) for i in range(201)]
    return {
        "prediction_value": {
            "type": "numeric",
            "declared_percentiles": declared,
            "cdf_probabilities": cdf_probabilities,
            "lower_bound": 0.0,
            "upper_bound": 100.0,
            "open_lower_bound": False,
            "open_upper_bound": False,
            "zero_point": None,
            "cdf_size": 201,
        },
        "reasoning": f"Model: {model}\n\nrationale text from {model}",
        "errors": [],
        "model": model,
    }


def _mc_payload(model: str = "openrouter/test/m1") -> dict:
    return {
        "prediction_value": {
            "type": "multiple_choice",
            "options": [
                {"option_name": "Red", "probability": 0.6},
                {"option_name": "Blue", "probability": 0.4},
            ],
        },
        "reasoning": f"Model: {model}\n\nrationale text from {model}",
        "errors": [],
        "model": model,
    }


def _three_binary_forecasters() -> dict[str, dict]:
    """Pre-built dict of three valid binary forecaster payloads."""
    return {
        model_slug_to_filename("openrouter/test/m1"): _binary_payload("openrouter/test/m1", 0.6),
        model_slug_to_filename("openrouter/test/m2"): _binary_payload("openrouter/test/m2", 0.5),
        model_slug_to_filename("openrouter/test/m3"): _binary_payload("openrouter/test/m3", 0.4),
    }


def _three_numeric_forecasters() -> dict[str, dict]:
    return {
        model_slug_to_filename("openrouter/test/m1"): _numeric_payload("openrouter/test/m1", 50.0),
        model_slug_to_filename("openrouter/test/m2"): _numeric_payload("openrouter/test/m2", 55.0),
        model_slug_to_filename("openrouter/test/m3"): _numeric_payload("openrouter/test/m3", 60.0),
    }


def _three_mc_forecasters() -> dict[str, dict]:
    return {
        model_slug_to_filename("openrouter/test/m1"): _mc_payload("openrouter/test/m1"),
        model_slug_to_filename("openrouter/test/m2"): _mc_payload("openrouter/test/m2"),
        model_slug_to_filename("openrouter/test/m3"): _mc_payload("openrouter/test/m3"),
    }


def _capture_base_texts(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[str]:
    """Pull the base_texts argument from a captured stacker call (positional or kw)."""
    base_texts = args[4] if len(args) > 4 else kwargs.get("base_texts", [])
    assert base_texts is not None
    return list(base_texts)


# ===========================================================================
# Helper: run an async coroutine in a sync test
# ===========================================================================


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)
